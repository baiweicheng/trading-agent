"""Safe, deterministic configuration loading and validation.

The loader is intentionally the only configuration entry point. It accepts one
safe YAML mapping plus a small, explicit environment allowlist, merges source
values at leaves, validates the complete Pydantic model, and resolves relative
paths only after the project boundary is known.
"""

# ruff: noqa: E501, SIM102

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Final, TypeVar, cast

from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.events import MappingStartEvent, ScalarEvent, SequenceStartEvent
from ruamel.yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from quant_research_platform.config.models import (
    DEFAULT_BENCHMARK,
    DEFAULT_UNIVERSE,
    ExecutionConfig,
    PathConfig,
    ResolvedConfig,
    RetryPolicyConfig,
    RuntimeConfig,
    SecretConfig,
    UiConfig,
    UnresolvedSecret,
)
from quant_research_platform.config.project_root import (
    ProjectRootBoundaryError,
    RelativePathEscapeError,
    normalize_local_path,
    resolve_project_root,
)
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    Ok,
    Result,
)

OPERATION: Final = "configuration.resolve"
REDACTION_MARKER: Final = "[REDACTED]"

# Every supported override is named explicitly. Adding a schema field does not
# expose it to the process environment until this map is deliberately updated.
ENVIRONMENT_FIELD_PATHS: Final[Mapping[str, tuple[str, ...]]] = {
    "QRP_PATHS__DATA_ROOT": ("paths", "data_root"),
    "QRP_PATHS__ARTIFACT_ROOT": ("paths", "artifact_root"),
    "QRP_PATHS__METADATA_DB": ("paths", "metadata_db"),
    "QRP_PATHS__MLFLOW_DB": ("paths", "mlflow_db"),
    "QRP_PATHS__LOCAL_SECRETS_FILE": ("paths", "local_secrets_file"),
    "QRP_RETRY__ATTEMPTS": ("retry", "attempts"),
    "QRP_RETRY__INITIAL_DELAY_SECONDS": ("retry", "initial_delay_seconds"),
    "QRP_RETRY__MAX_DELAY_SECONDS": ("retry", "max_delay_seconds"),
    "QRP_RETRY__BACKOFF_MULTIPLIER": ("retry", "backoff_multiplier"),
    "QRP_DATA__UNIVERSE": ("data", "universe"),
    "QRP_DATA__REQUESTED_RANGE__START": ("data", "requested_range", "start"),
    "QRP_DATA__REQUESTED_RANGE__END": ("data", "requested_range", "end"),
    "QRP_DATA__BENCHMARK": ("data", "benchmark"),
    "QRP_DATA__PROVIDER": ("data", "provider"),
    "QRP_DATA__BATCH_SIZE": ("data", "batch_size"),
    "QRP_DATA__STALENESS_SESSIONS": ("data", "staleness_sessions"),
    "QRP_DATA__REVISION_OVERLAP_SESSIONS": (
        "data",
        "revision_overlap_sessions",
    ),
    "QRP_DATA__WRITE_CHUNK_ROWS": ("data", "write_chunk_rows"),
    "QRP_STRATEGY__IDENTIFIER": ("strategy", "identifier"),
    "QRP_STRATEGY__POSITION_COUNT": ("strategy", "position_count"),
    "QRP_STRATEGY__LONG_LOOKBACK_SESSIONS": (
        "strategy",
        "long_lookback_sessions",
    ),
    "QRP_STRATEGY__SKIP_RECENT_SESSIONS": ("strategy", "skip_recent_sessions"),
    "QRP_EXECUTION__INITIAL_EQUITY_USD": ("execution", "initial_equity_usd"),
    "QRP_EXECUTION__COMMISSION_BPS": ("execution", "commission_bps"),
    "QRP_EXECUTION__SLIPPAGE_BPS": ("execution", "slippage_bps"),
    "QRP_UI__PAGE_SIZE": ("ui", "page_size"),
    "QRP_RUNTIME__DETERMINISTIC_SEED": ("runtime", "deterministic_seed"),
    "QRP_SECRETS__HTTP_PROXY": ("secrets", "http_proxy"),
    "QRP_SECRETS__HTTPS_PROXY": ("secrets", "https_proxy"),
}

# Required fields such as data.requested_range are intentionally absent. Their
# absence is passed to Pydantic so users receive required-field diagnostics.
DOCUMENTED_DEFAULTS: Final[dict[str, object]] = {
    "paths": PathConfig().model_dump(mode="python"),
    "retry": RetryPolicyConfig.model_validate({}).model_dump(mode="python"),
    "data": {
        "universe": DEFAULT_UNIVERSE,
        "benchmark": DEFAULT_BENCHMARK,
        "provider": "yfinance",
        "batch_size": 5,
        "staleness_sessions": 1,
        "revision_overlap_sessions": 5,
        "write_chunk_rows": 50_000,
    },
    # position_count remains absent so ResolvedConfig can derive it from the
    # final normalized universe rather than a lower-precedence default universe.
    "strategy": {
        "identifier": "monthly_momentum_v1",
        "long_lookback_sessions": 252,
        "skip_recent_sessions": 21,
    },
    "execution": ExecutionConfig.model_validate({}).model_dump(mode="python"),
    "ui": UiConfig.model_validate({}).model_dump(mode="python"),
    "runtime": RuntimeConfig.model_validate({}).model_dump(mode="python"),
    "secrets": SecretConfig().model_dump(mode="python"),
}

_LIST_INDEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[([0-9]+)\]")
_UNIVERSE_POSITION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\buniverse\[([0-9]+)\]"
)
_FIELD_PATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:paths|retry|data|strategy|execution|ui|runtime|secrets)"
    r"(?:\.[a-z_]+)+(?:\[[0-9]+\])?"
)

T = TypeVar("T")


def _safe_yaml() -> YAML:
    """Construct a fresh safe parser with duplicate YAML keys disabled."""

    parser = YAML(typ="safe", pure=True)
    parser.allow_duplicate_keys = False
    return parser


def _format_location(mark: object | None) -> str:
    """Return a parser location without exposing source-line contents."""

    line = getattr(mark, "line", None)
    column = getattr(mark, "column", None)
    if isinstance(line, int) and isinstance(column, int):
        return f"line {line + 1}, column {column + 1}"
    return "an unknown location"


def _error(
    category: ErrorCategory,
    message: str,
    corrective_action: str,
    *,
    field_path: str | None = None,
) -> ActionableError:
    return ActionableError(
        operation=OPERATION,
        category=category,
        message=message,
        corrective_action=corrective_action,
        field_path=field_path,
    )


def _yaml_failure(error: YAMLError, secret_values: tuple[str, ...]) -> Err:
    mark = getattr(error, "problem_mark", None) or getattr(error, "context_mark", None)
    message = _sanitize(
        f"YAML parsing failed at {_format_location(mark)}.", secret_values
    )
    return _configuration_err(
        (
            _error(
                ErrorCategory.CONFIGURATION_SYNTAX,
                message,
                "Correct the YAML syntax and try configuration resolution again.",
            ),
        )
    )


def _reject_explicit_tags(
    yaml_document: str | bytes, secret_values: tuple[str, ...]
) -> Err | None:
    """Reject every explicit YAML tag before construction can interpret it."""

    try:
        for event in _safe_yaml().parse(yaml_document):
            if isinstance(event, (ScalarEvent, SequenceStartEvent, MappingStartEvent)):
                if event.tag is not None:
                    return _configuration_err(
                        (
                            _error(
                                ErrorCategory.CONFIGURATION_SYNTAX,
                                _sanitize(
                                    "YAML tags are not permitted in configuration "
                                    f"at {_format_location(event.start_mark)}.",
                                    secret_values,
                                ),
                                "Remove the YAML tag and use a plain safe YAML value.",
                            ),
                        )
                    )
    except YAMLError as error:
        return _yaml_failure(error, secret_values)
    return None


def _mapping_key(node: Node) -> str:
    if isinstance(node, ScalarNode):
        return str(node.value)
    return "<non-scalar-key>"


def _join_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _duplicate_key(
    node: Node | None, path: tuple[str, ...] = ()
) -> tuple[str, object] | None:
    """Find the first duplicate key, preserving its complete YAML nesting path."""

    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = _mapping_key(key_node)
            key_path = (*path, key)
            if key in seen:
                return _join_path(key_path), key_node.start_mark
            seen.add(key)
            duplicate = _duplicate_key(value_node, key_path)
            if duplicate is not None:
                return duplicate
    elif isinstance(node, SequenceNode):
        for index, item in enumerate(node.value):
            duplicate = _duplicate_key(item, (*path, f"[{index}]"))
            if duplicate is not None:
                return duplicate
    return None


def _load_yaml_mapping(
    yaml_document: str | bytes | None, secret_values: tuple[str, ...]
) -> Result[dict[str, object]]:
    """Parse one safe YAML document and require a mapping root."""

    if yaml_document is None:
        return Ok({})

    tag_error = _reject_explicit_tags(yaml_document, secret_values)
    if tag_error is not None:
        return tag_error

    try:
        composed = _safe_yaml().compose(yaml_document)
    except YAMLError as error:
        return _yaml_failure(error, secret_values)

    duplicate = _duplicate_key(composed)
    if duplicate is not None:
        duplicate_path, mark = duplicate
        return _configuration_err(
            (
                _error(
                    ErrorCategory.CONFIGURATION_DUPLICATE_KEY,
                    _sanitize(
                        f"YAML contains a duplicate key at {duplicate_path} "
                        f"({_format_location(mark)}).",
                        secret_values,
                    ),
                    "Keep one value for the duplicated configuration key.",
                    field_path=duplicate_path,
                ),
            )
        )

    try:
        loaded = _safe_yaml().load(yaml_document)
    except YAMLError as error:
        return _yaml_failure(error, secret_values)

    if not isinstance(loaded, Mapping):
        root_type = type(loaded).__name__
        return _configuration_err(
            (
                _error(
                    ErrorCategory.CONFIGURATION_SYNTAX,
                    _sanitize(
                        "YAML configuration root has type "
                        f"{root_type}; the required root type is mapping.",
                        secret_values,
                    ),
                    "Make the YAML document root a mapping of configuration sections.",
                ),
            )
        )
    return Ok({str(key): value for key, value in loaded.items()})


def _nested_model(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _allowed_siblings(model: type[BaseModel]) -> str:
    return ", ".join(model.model_fields)


def _known_yaml_values(
    value: object,
    model: type[BaseModel],
    path: tuple[str, ...] = (),
) -> tuple[object, list[ActionableError], set[str]]:
    """Remove unknown YAML leaves while reporting actionable sibling diagnostics."""

    if not isinstance(value, Mapping):
        return value, [], set()

    known: dict[str, object] = {}
    errors: list[ActionableError] = []
    structural_error_paths: set[str] = set()
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        child_path = (*path, key)
        rendered_path = _join_path(child_path)
        field = model.model_fields.get(key) if isinstance(raw_key, str) else None
        if field is None:
            errors.append(
                _error(
                    ErrorCategory.CONFIGURATION_UNKNOWN_KEY,
                    "Unknown configuration key "
                    f"{rendered_path}; allowed sibling keys are: "
                    f"{_allowed_siblings(model)}.",
                    "Remove the unknown key or rename it to an allowed sibling key.",
                    field_path=rendered_path,
                )
            )
            continue

        nested = _nested_model(field.annotation)
        if nested is None:
            # Plain YAML must never supply a secret. The marker is accepted as an
            # unresolved value so canonical redacted configuration can be reloaded.
            if model is SecretConfig and raw_value not in (None, REDACTION_MARKER):
                errors.append(
                    _error(
                        ErrorCategory.CONFIGURATION_INVALID_VALUE,
                        "Secret configuration values may only be supplied by mapped "
                        "QRP_ environment variables.",
                        "Remove the secret from YAML and set its explicitly mapped "
                        "environment variable instead.",
                        field_path=rendered_path,
                    )
                )
                known[key] = None
            else:
                known[key] = (
                    UnresolvedSecret()
                    if model is SecretConfig and raw_value == REDACTION_MARKER
                    else raw_value
                )
            continue

        if not isinstance(raw_value, Mapping):
            structural_error_paths.add(rendered_path)
            errors.append(
                _error(
                    ErrorCategory.CONFIGURATION_INVALID_VALUE,
                    f"Configuration section {rendered_path} must be a mapping.",
                    "Provide a mapping with the documented sibling fields.",
                    field_path=rendered_path,
                )
            )
            known[key] = raw_value
            continue

        child_known, child_errors, child_structural_paths = _known_yaml_values(
            raw_value, nested, child_path
        )
        known[key] = child_known
        errors.extend(child_errors)
        structural_error_paths.update(child_structural_paths)
    return known, errors, structural_error_paths


def _assign_leaf(
    target: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    current = target
    for component in path[:-1]:
        child = current.get(component)
        if not isinstance(child, dict):
            child = {}
            current[component] = child
        current = child
    current[path[-1]] = value


def _environment_values(
    environment: Mapping[str, str],
) -> tuple[dict[str, object], list[ActionableError], tuple[str, ...]]:
    """Return only allowlisted QRP environment leaves and unmapped-name errors."""

    values: dict[str, object] = {}
    errors: list[ActionableError] = []
    secret_values: list[str] = []
    for name, raw_value in environment.items():
        if not isinstance(name, str) or not name.startswith("QRP_"):
            continue
        path = ENVIRONMENT_FIELD_PATHS.get(name)
        if path is None:
            errors.append(
                _error(
                    ErrorCategory.CONFIGURATION_UNKNOWN_KEY,
                    f"Environment override {name} has no explicit configuration mapping.",
                    "Remove the variable or add an approved explicit field mapping.",
                    field_path=name,
                )
            )
            continue
        if not isinstance(raw_value, str):
            errors.append(
                _error(
                    ErrorCategory.CONFIGURATION_INVALID_VALUE,
                    f"Environment override {name} must be a string value.",
                    "Set the mapped environment variable to a string value.",
                    field_path=name,
                )
            )
            continue

        value: object = raw_value
        if path == ("data", "universe"):
            value = tuple(symbol.strip() for symbol in raw_value.split(","))
        if path[0] == "secrets":
            secret_values.append(raw_value)
        _assign_leaf(values, path, value)
    return values, errors, tuple(secret_values)


def _deep_merge(
    lower_precedence: Mapping[str, object], higher_precedence: Mapping[str, object]
) -> dict[str, object]:
    """Recursively merge maps so independent nested leaves retain precedence."""

    merged = deepcopy(dict(lower_precedence))
    for key, higher_value in higher_precedence.items():
        lower_value = merged.get(key)
        if isinstance(lower_value, Mapping) and isinstance(higher_value, Mapping):
            merged[key] = _deep_merge(lower_value, higher_value)
        else:
            merged[key] = deepcopy(higher_value)
    return merged


def _field_order_paths(
    model: type[BaseModel] = ResolvedConfig,
    path: tuple[str, ...] = (),
    order: tuple[int, ...] = (),
) -> dict[tuple[str, ...], tuple[int, ...]]:
    paths: dict[tuple[str, ...], tuple[int, ...]] = {}
    for index, (field_name, field) in enumerate(model.model_fields.items()):
        child_path = (*path, field_name)
        child_order = (*order, index)
        paths[child_path] = child_order
        nested = _nested_model(field.annotation)
        if nested is not None:
            paths.update(_field_order_paths(nested, child_path, child_order))
    return paths


FIELD_ORDER_PATHS: Final = _field_order_paths()


def _field_path_tokens(field_path: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names: list[str] = []
    indexes: list[int] = []
    for component in field_path.split("."):
        name = component.split("[", 1)[0]
        if name:
            names.append(name)
        indexes.extend(int(index) for index in _LIST_INDEX_PATTERN.findall(component))
    return tuple(names), tuple(indexes)


def _configuration_error_sort_key(
    error: ActionableError,
) -> tuple[tuple[int, ...], str, str]:
    field_path = error.field_path or ""
    names, indexes = _field_path_tokens(field_path)
    for length in range(len(names), 0, -1):
        known_order = FIELD_ORDER_PATHS.get(names[:length])
        if known_order is not None:
            remaining_names = len(names) - length
            return (
                (*known_order, *indexes, *([99_999] * remaining_names)),
                field_path,
                error.message,
            )
    return ((99_999,), field_path, error.message)


def _configuration_err(
    errors: tuple[ActionableError, ...] | list[ActionableError],
) -> Err:
    """Return errors in Pydantic schema field and list-index order."""

    return Err(
        tuple(sorted(errors, key=_configuration_error_sort_key)),
        preserve_order=True,
    )


def _pydantic_path(detail: Mapping[str, object]) -> str:
    raw_location = detail.get("loc", ())
    parts = (
        [str(part) for part in raw_location] if isinstance(raw_location, tuple) else []
    )
    path = ".".join(parts) or "configuration"
    message = str(detail.get("msg", "invalid configuration value"))
    explicit_path = _FIELD_PATH_PATTERN.search(message)
    if explicit_path is not None:
        path = explicit_path.group(0)
    universe_position = _UNIVERSE_POSITION_PATTERN.search(message)
    if universe_position is not None and path.endswith("data.universe"):
        path = f"{path}[{universe_position.group(1)}]"
    return path


def _expected_type(field_path: str) -> str:
    names, _ = _field_path_tokens(field_path)
    model: type[BaseModel] = ResolvedConfig
    for index, name in enumerate(names):
        field = model.model_fields.get(name)
        if field is None:
            break
        if index == len(names) - 1:
            annotation = field.annotation
            annotation_name = getattr(annotation, "__name__", None)
            return annotation_name or str(annotation)
        nested = _nested_model(field.annotation)
        if nested is None:
            break
        model = nested
    return "the documented field type"


def _pydantic_errors(
    validation_error: ValidationError,
    secret_values: tuple[str, ...],
    structural_error_paths: set[str],
) -> list[ActionableError]:
    errors: list[ActionableError] = []
    for detail in validation_error.errors(include_url=False):
        path = _pydantic_path(detail)
        if path in structural_error_paths:
            continue
        error_type = str(detail.get("type", ""))
        if error_type == "missing":
            message = f"Required configuration field {path} is missing."
        else:
            message = (
                f"Configuration field {path} is invalid: "
                f"{detail.get('msg', 'invalid value')}. Accepted type: "
                f"{_expected_type(path)}."
            )
        errors.append(
            _error(
                ErrorCategory.CONFIGURATION_INVALID_VALUE,
                _sanitize(message, secret_values),
                "Correct the field value and retry configuration resolution.",
                field_path=path,
            )
        )
    return errors


def _sanitize(text: str, secret_values: tuple[str, ...]) -> str:
    sanitized = text
    for secret in sorted(
        (value for value in secret_values if value), key=len, reverse=True
    ):
        sanitized = sanitized.replace(secret, REDACTION_MARKER)
    return sanitized


def _normalize_paths(
    config: ResolvedConfig, project_root: Path
) -> Result[ResolvedConfig]:
    normalized_values = config.paths.model_dump(mode="python")
    errors: list[ActionableError] = []
    for field_name in PathConfig.model_fields:
        value = getattr(config.paths, field_name)
        if value is None:
            continue
        field_path = f"paths.{field_name}"
        try:
            normalized_values[field_name] = normalize_local_path(
                value,
                project_root=project_root,
                field_path=field_path,
            )
        except RelativePathEscapeError:
            errors.append(
                _error(
                    ErrorCategory.CONFIGURATION_INVALID_VALUE,
                    "Configured relative path for "
                    f"{field_path} resolves outside Project_Root boundary "
                    f"{project_root}.",
                    "Choose a relative path inside Project_Root or provide an "
                    "explicit absolute path.",
                    field_path=field_path,
                )
            )
    if errors:
        return _configuration_err(errors)

    resolved_paths = PathConfig.model_validate(normalized_values)
    return Ok(config.model_copy(update={"paths": resolved_paths}))


class ConfigurationManager:
    """Resolve safe configuration without allowing invalid values to reach services."""

    def __init__(self, *, project_anchor: Path | None = None) -> None:
        self._project_anchor = project_anchor

    def resolve(
        self,
        yaml_document: str | bytes | None,
        environment: Mapping[str, str],
    ) -> Result[ResolvedConfig]:
        """Safely parse, merge, validate, and path-resolve one configuration."""

        environment_values, environment_errors, secret_values = _environment_values(
            environment
        )
        parsed = _load_yaml_mapping(yaml_document, secret_values)
        if isinstance(parsed, Err):
            return parsed

        yaml_values, yaml_errors, structural_error_paths = _known_yaml_values(
            parsed.value, ResolvedConfig
        )
        yaml_mapping = cast(Mapping[str, object], yaml_values)
        merged = _deep_merge(DOCUMENTED_DEFAULTS, yaml_mapping)
        merged = _deep_merge(merged, environment_values)

        errors = [*yaml_errors, *environment_errors]
        try:
            config = ResolvedConfig.model_validate(merged)
        except ValidationError as validation_error:
            errors.extend(
                _pydantic_errors(
                    validation_error, secret_values, structural_error_paths
                )
            )
            return _configuration_err(errors)
        if errors:
            return _configuration_err(errors)

        try:
            project_root = resolve_project_root(self._project_anchor)
        except ProjectRootBoundaryError as error:
            return _configuration_err(
                (
                    _error(
                        ErrorCategory.CONFIGURATION_INVALID_VALUE,
                        str(error),
                        "Resolve the project so exactly one pyproject.toml boundary "
                        "contains the package.",
                        field_path="project_root",
                    ),
                )
            )
        return _normalize_paths(config, project_root)

    def resolve_then(
        self,
        yaml_document: str | bytes | None,
        environment: Mapping[str, str],
        operation: Callable[[ResolvedConfig], Result[T]],
    ) -> Result[T]:
        """Invoke a downstream operation only when configuration resolves cleanly."""

        resolved = self.resolve(yaml_document, environment)
        if isinstance(resolved, Err):
            return resolved
        return operation(resolved.value)


def resolve_configuration(
    yaml_document: str | bytes | None,
    environment: Mapping[str, str],
    *,
    project_anchor: Path | None = None,
) -> Result[ResolvedConfig]:
    """Convenience entry point for callers that do not retain a manager instance."""

    return ConfigurationManager(project_anchor=project_anchor).resolve(
        yaml_document, environment
    )


__all__ = [
    "DOCUMENTED_DEFAULTS",
    "ENVIRONMENT_FIELD_PATHS",
    "REDACTION_MARKER",
    "ConfigurationManager",
    "resolve_configuration",
]
