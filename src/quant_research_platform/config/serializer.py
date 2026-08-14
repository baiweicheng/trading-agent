"""Canonical non-secret configuration serialization and reusable redaction.

The serializer is the only configuration-to-durable/display boundary.  It
projects resolved proxy credentials to presence states, writes deterministic
safe YAML, and reloads a redaction marker as a value-free unresolved secret.
``Redactor`` is deliberately independent of logging, MLflow, or Streamlit so
all of those sinks can use one idempotent sanitization implementation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from typing import Final
from urllib.parse import quote, quote_plus

from pydantic import BaseModel, SecretStr
from ruamel.yaml import YAML

from quant_research_platform.domain.canonical import canonical_date, canonical_decimal

from .models import (
    REDACTION_MARKER,
    DataConfig,
    ExecutionConfig,
    PathConfig,
    ResolvedConfig,
    RetryPolicyConfig,
    RuntimeConfig,
    SecretConfig,
    SecretValue,
    StrategyConfig,
    UiConfig,
    UnresolvedSecret,
)


class SecretPresence(StrEnum):
    """The safe observable state of a configuration secret field."""

    ABSENT = "absent"
    PRESENT_UNRESOLVED = "present_unresolved"
    PRESENT_REDACTED = "present_redacted"


class NonSecretSecretConfig(BaseModel):
    """Secret names and safe state markers without any credential values."""

    model_config = {"extra": "forbid", "frozen": True}

    http_proxy: SecretPresence = SecretPresence.ABSENT
    https_proxy: SecretPresence = SecretPresence.ABSENT


class NonSecretConfig(BaseModel):
    """The durable/display-safe projection of :class:`ResolvedConfig`."""

    model_config = {"extra": "forbid", "frozen": True}

    paths: PathConfig
    retry: RetryPolicyConfig
    data: DataConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    ui: UiConfig
    runtime: RuntimeConfig
    secrets: NonSecretSecretConfig


class RequiredSecretError(ValueError):
    """Raised when an operation requests credentials absent from a config."""

    def __init__(self, field_paths: Sequence[str]) -> None:
        self.field_paths = tuple(field_paths)
        fields_text = ", ".join(self.field_paths)
        super().__init__(
            f"Required secret values are unresolved for: {fields_text}. "
            "Supply them from an approved external secret source."
        )


class SecretLeakError(ValueError):
    """Raised when an artifact metadata boundary receives an unredacted secret."""

    def __init__(self) -> None:
        super().__init__(
            "Metadata contains a registered secret and cannot be published."
        )


_SECRET_FIELDS: Final[tuple[str, ...]] = ("http_proxy", "https_proxy")
_SECRET_PATHS: Final[dict[str, str]] = {
    field_name: field_name for field_name in _SECRET_FIELDS
} | {f"secrets.{field_name}": field_name for field_name in _SECRET_FIELDS}
_SAFE_LOG_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {"accept", "content-type", "user-agent", "x-request-id"}
)
_SIMPLE_YAML_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _secret_presence(value: SecretValue) -> SecretPresence:
    if value is None:
        return SecretPresence.ABSENT
    if isinstance(value, UnresolvedSecret):
        return SecretPresence.PRESENT_UNRESOLVED
    if isinstance(value, SecretStr):
        return SecretPresence.PRESENT_REDACTED
    raise TypeError("Unsupported secret value type")


def non_secret_config(config: ResolvedConfig | NonSecretConfig) -> NonSecretConfig:
    """Return a credential-free projection of a resolved configuration.

    Supplying an already-projected configuration is intentionally a no-op,
    making this boundary safe for durable, metadata, and presenter callers to
    use repeatedly.
    """

    if isinstance(config, NonSecretConfig):
        return config
    if not isinstance(config, ResolvedConfig):
        raise TypeError("Expected ResolvedConfig or NonSecretConfig")

    return NonSecretConfig(
        paths=config.paths,
        retry=config.retry,
        data=config.data,
        strategy=config.strategy,
        execution=config.execution,
        ui=config.ui,
        runtime=config.runtime,
        secrets=NonSecretSecretConfig(
            http_proxy=_secret_presence(config.secrets.http_proxy),
            https_proxy=_secret_presence(config.secrets.https_proxy),
        ),
    )


def _canonical_secret_value(presence: SecretPresence) -> str | None:
    if presence is SecretPresence.ABSENT:
        return None
    return REDACTION_MARKER


def _schema_mapping(value: object) -> object:
    """Project models recursively in declaration order without secret values."""

    if isinstance(value, NonSecretSecretConfig):
        return {
            field_name: _canonical_secret_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, SecretConfig):
        return {
            field_name: _canonical_secret_value(
                _secret_presence(getattr(value, field_name))
            )
            for field_name in type(value).model_fields
        }
    if isinstance(value, BaseModel):
        return {
            field_name: _schema_mapping(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {str(key): _schema_mapping(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_schema_mapping(item) for item in value]
    if isinstance(value, list):
        return [_schema_mapping(item) for item in value]
    if isinstance(value, SecretStr | UnresolvedSecret):
        return REDACTION_MARKER
    return value


def _is_yaml_container(value: object) -> bool:
    return isinstance(value, (Mapping, list, tuple))


def _yaml_scalar(value: object) -> str:
    """Render one supported value with a unique, safe YAML scalar spelling."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Canonical configuration YAML does not allow non-finite floats"
            )
        return repr(value)
    if isinstance(value, datetime):
        raise TypeError("Canonical configuration YAML does not support datetimes")
    if isinstance(value, date):
        return canonical_date(value)
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, SecretStr | UnresolvedSecret):
        value = REDACTION_MARKER
    if isinstance(value, str):
        # JSON strings are valid YAML quoted scalars and give deterministic
        # escaping for brackets, comment characters, control characters, and
        # Unicode while retaining human-readable UTF-8 text.
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(
        "Canonical configuration YAML does not support "
        f"{type(value).__name__} scalar values"
    )


def _yaml_key(value: str) -> str:
    return value if _SIMPLE_YAML_KEY.fullmatch(value) else _yaml_scalar(value)


def _yaml_lines(value: object, indent: int = 0) -> list[str]:
    """Render a nested YAML node using a fixed two-space indentation style."""

    prefix = " " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Canonical configuration YAML mapping keys must be strings"
                )
            rendered_key = _yaml_key(key)
            if _is_yaml_container(item):
                if not item:
                    lines.append(f"{prefix}{rendered_key}: {_yaml_lines(item, 0)[0]}")
                else:
                    lines.append(f"{prefix}{rendered_key}:")
                    lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{rendered_key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if _is_yaml_container(item):
                if not item:
                    lines.append(f"{prefix}- {_yaml_lines(item, 0)[0]}")
                else:
                    lines.append(f"{prefix}-")
                    lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _canonical_yaml_document(value: object) -> bytes:
    """Return UTF-8 LF-only YAML with exactly one trailing line feed."""

    text = "\n".join(_yaml_lines(value)).replace("\r\n", "\n").replace("\r", "\\r")
    return (text.rstrip("\n") + "\n").encode("utf-8")


class ConfigurationSerializer:
    """Serialize and reload canonical credential-free configuration YAML."""

    def serialize(self, config: ResolvedConfig | NonSecretConfig) -> bytes:
        """Serialize every schema field in declaration order without secrets."""

        return _canonical_yaml_document(_schema_mapping(non_secret_config(config)))

    dumps = serialize

    def deserialize(self, document: bytes | str) -> ResolvedConfig:
        """Reload canonical YAML while converting secret markers to unresolved state."""

        if isinstance(document, bytes):
            try:
                yaml_text = document.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Canonical configuration YAML must be valid UTF-8"
                ) from error
        elif isinstance(document, str):
            yaml_text = document
        else:
            raise TypeError("Canonical configuration YAML must be bytes or text")

        yaml = YAML(typ="safe")
        loaded = yaml.load(yaml_text)
        if not isinstance(loaded, Mapping):
            raise ValueError("Canonical configuration YAML must have a mapping root")
        loaded_mapping = dict(loaded)
        self._reject_unredacted_canonical_secrets(loaded_mapping)
        return ResolvedConfig.model_validate(loaded_mapping)

    loads = deserialize

    @staticmethod
    def _reject_unredacted_canonical_secrets(document: Mapping[str, object]) -> None:
        secrets = document.get("secrets")
        if not isinstance(secrets, Mapping):
            return
        for field_name in _SECRET_FIELDS:
            value = secrets.get(field_name)
            if value not in (None, REDACTION_MARKER):
                raise ValueError(
                    "Canonical configuration YAML may contain only null or "
                    "[REDACTED] for secret fields"
                )


def canonical_yaml(config: ResolvedConfig | NonSecretConfig) -> bytes:
    """Convenience function for canonical non-secret YAML serialization."""

    return ConfigurationSerializer().serialize(config)


def load_canonical_yaml(document: bytes | str) -> ResolvedConfig:
    """Convenience function for canonical YAML reload with unresolved secrets."""

    return ConfigurationSerializer().deserialize(document)


def require_resolved_secrets(
    config: ResolvedConfig,
    required_fields: Iterable[str],
) -> None:
    """Gate only an operation's declared secret dependencies.

    Operations with no required proxy credential remain runnable after a
    canonical configuration reload.  An operation that declares a secret is
    blocked until a mapped environment variable or approved local secret source
    resolves the requested field.
    """

    requested = tuple(required_fields)
    unknown = [field for field in requested if field not in _SECRET_PATHS]
    if unknown:
        raise ValueError("Unknown secret field requirement")

    requested_field_names = {
        _SECRET_PATHS[field_path] for field_path in requested
    }
    missing = tuple(
        f"secrets.{field_name}"
        for field_name in _SECRET_FIELDS
        if field_name in requested_field_names
        and not isinstance(getattr(config.secrets, field_name), SecretStr)
    )
    if missing:
        raise RequiredSecretError(missing)


class Redactor:
    """Idempotently remove registered literal and URL-encoded secret forms."""

    def __init__(self, secrets: Iterable[str | SecretStr] = ()) -> None:
        self._forms: tuple[str, ...] = ()
        self._pattern: re.Pattern[str] | None = None
        for secret in secrets:
            self.register(secret)

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> Redactor:
        """Create a redactor from only resolved configuration secret values."""

        values: list[SecretStr] = []
        for field_name in _SECRET_FIELDS:
            value = getattr(config.secrets, field_name)
            if isinstance(value, SecretStr):
                values.append(value)
        return cls(values)

    @property
    def registered_form_count(self) -> int:
        """Return the number of recognized spellings without exposing any of them."""

        return len(self._forms)

    def register(self, secret: str | SecretStr) -> None:
        """Register literal and common URL-encoded spellings of one secret."""

        value = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
        if not isinstance(value, str) or not value:
            raise ValueError("Registered secret values must be non-empty strings")
        if value == REDACTION_MARKER:
            raise ValueError("The redaction marker cannot be registered as a secret")

        forms = set(self._forms)
        direct_forms = {value, quote(value, safe=""), quote_plus(value, safe="")}
        forms.update(direct_forms)
        forms.update(quote(form, safe="") for form in direct_forms)
        self._forms = tuple(sorted(forms, key=lambda item: (-len(item), item)))
        self._pattern = (
            re.compile("|".join(re.escape(form) for form in self._forms))
            if self._forms
            else None
        )

    def redact_text(self, text: str) -> str:
        """Redact a text value while preserving existing markers unchanged."""

        if not isinstance(text, str):
            raise TypeError("Text redaction requires a string")
        if self._pattern is None:
            return text
        # Existing markers are protected first so a credential such as
        # ``REDACTED`` cannot recursively alter the marker on repeated calls.
        return REDACTION_MARKER.join(
            self._pattern.sub(REDACTION_MARKER, segment)
            for segment in text.split(REDACTION_MARKER)
        )

    sanitize_text = redact_text

    def redact_url(self, url: str) -> str:
        """Redact literal or URL-encoded credentials embedded in a URL."""

        return self.redact_text(url)

    sanitize_url = redact_url

    def redact_headers(
        self,
        headers: Mapping[str, str],
        *,
        allowed_names: Collection[str] = (),
    ) -> dict[str, str]:
        """Produce log-safe headers, retaining only explicitly safe header values."""

        permitted = _SAFE_LOG_HEADER_NAMES | {
            name.casefold() for name in allowed_names
        }
        sanitized: dict[str, str] = {}
        for name, value in headers.items():
            redacted_name = self.redact_text(name)
            if name.casefold() in permitted:
                sanitized[redacted_name] = self.redact_text(value)
            else:
                sanitized[redacted_name] = REDACTION_MARKER
        return sanitized

    sanitize_headers = redact_headers

    def redact_structured(self, value: object) -> object:
        """Recursively sanitize metadata, DTOs, Pydantic models, and dataclasses."""

        if isinstance(value, (SecretStr, UnresolvedSecret)):
            return REDACTION_MARKER
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, BaseException):
            return self.redact_text(str(value))
        if isinstance(value, BaseModel):
            return self.redact_structured(value.model_dump(mode="python"))
        if isinstance(value, Mapping):
            sanitized_mapping: dict[object, object] = {}
            for key, item in value.items():
                sanitized_key = self.redact_text(key) if isinstance(key, str) else key
                sanitized_mapping[sanitized_key] = self.redact_structured(item)
            return sanitized_mapping
        if isinstance(value, list):
            return [self.redact_structured(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_structured(item) for item in value)
        if isinstance(value, set):
            return {self.redact_structured(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self.redact_structured(item) for item in value)
        if is_dataclass(value) and not isinstance(value, type):
            changes = {
                field.name: self.redact_structured(getattr(value, field.name))
                for field in fields(value)
            }
            try:
                return replace(value, **changes)
            except (TypeError, ValueError):
                return changes
        return value

    sanitize_metadata = redact_structured
    redact_metadata = redact_structured
    sanitize_progress = redact_structured
    redact_progress = redact_structured
    sanitize_log = redact_structured
    redact_log = redact_structured
    sanitize_mlflow_fields = redact_structured
    redact_mlflow_fields = redact_structured
    sanitize_manifest = redact_structured
    redact_manifest = redact_structured
    sanitize_presenter_dto = redact_structured
    redact_presenter_dto = redact_structured

    def redact_error(self, error: object) -> object:
        """Return a redacted exception message or a sanitized structured error."""

        return self.redact_structured(error)

    sanitize_error = redact_error

    def contains_secret(self, value: object) -> bool:
        """Detect registered values recursively before a durable metadata write."""

        if isinstance(value, SecretStr):
            return True
        if isinstance(value, UnresolvedSecret):
            return False
        if isinstance(value, str):
            return self._contains_secret_text(value)
        if isinstance(value, BaseException):
            return self._contains_secret_text(str(value))
        if isinstance(value, BaseModel):
            return self.contains_secret(value.model_dump(mode="python"))
        if isinstance(value, Mapping):
            return any(
                self.contains_secret(key) or self.contains_secret(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(self.contains_secret(item) for item in value)
        if is_dataclass(value) and not isinstance(value, type):
            return any(
                self.contains_secret(getattr(value, field.name))
                for field in fields(value)
            )
        return False

    def assert_metadata_is_redacted(self, metadata: Mapping[str, object]) -> None:
        """Fail closed when final artifact metadata still contains a secret."""

        if self.contains_secret(metadata):
            raise SecretLeakError()

    def _contains_secret_text(self, text: str) -> bool:
        if self._pattern is None:
            return False
        return any(
            self._pattern.search(segment) is not None
            for segment in text.split(REDACTION_MARKER)
        )


__all__ = [
    "ConfigurationSerializer",
    "NonSecretConfig",
    "NonSecretSecretConfig",
    "REDACTION_MARKER",
    "Redactor",
    "RequiredSecretError",
    "SecretLeakError",
    "SecretPresence",
    "canonical_yaml",
    "load_canonical_yaml",
    "non_secret_config",
    "require_resolved_secrets",
]
