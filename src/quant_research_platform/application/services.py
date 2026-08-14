"""Typed, framework-independent public application facade.

The facade is the only application entry point used by presentation adapters.  It
keeps resolved configuration objects in a process-local registry, exposes only a
credential-free view to callers, and delegates each use case to an injected
application service or port.  Infrastructure and presentation concerns stay
outside this module.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar, cast
from uuid import UUID, uuid4

from ruamel.yaml import YAML

from ..config.loader import ConfigurationManager
from ..config.models import ResolvedConfig
from ..config.serializer import (
    NonSecretConfig,
    Redactor,
    non_secret_config,
)
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.execution import RunState
from .backtests import BacktestRequest
from .comparisons import ComparisonOutput
from .ingestion import IngestionRequest, IngestionResult
from .inspection import RunDetail, RunSummary, TablePage
from .snapshots import SnapshotDetail, SnapshotQuery, SnapshotSummary

T = TypeVar("T")

ProgressCallback = Callable[[object], None]
"""Structural callback accepted by synchronous application services."""


@dataclass(frozen=True, slots=True, repr=False)
class ConfigurationHandle:
    """Opaque process-local reference to one validated configuration.

    The token is an identifier only; the resolved configuration is never stored
    on the handle and therefore cannot be serialized or inspected through the
    public DTO.  A handle is useful only with the facade instance that issued
    it.
    """

    _token: UUID

    def __post_init__(self) -> None:
        if not isinstance(self._token, UUID):
            raise TypeError("configuration handle token must be a UUID")

    def __repr__(self) -> str:
        return "ConfigurationHandle(<opaque>)"


ResolvedConfigView: TypeAlias = NonSecretConfig
"""Credential-free configuration projection returned to presentation code."""


@dataclass(frozen=True, slots=True)
class ConfigurationResolution:
    """Validated configuration preview plus its process-local opaque handle."""

    handle: ConfigurationHandle
    view: ResolvedConfigView

    @property
    def non_secret(self) -> ResolvedConfigView:
        """Compatibility alias for callers using the security terminology."""

        return self.view


@dataclass(frozen=True, slots=True)
class RunQuery:
    """Typed, bounded run-discovery filters exposed by the facade."""

    run_id: str | None = None
    snapshot_id: str | None = None
    strategy_id: str | None = None
    universe: tuple[str, ...] | None = None
    evaluation_start: date | None = None
    evaluation_end: date | None = None
    state: RunState | str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    page: int = 0
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.run_id is not None:
            if not isinstance(self.run_id, str) or not self.run_id.strip():
                raise ValueError("run_id must be a non-blank string or None")
            object.__setattr__(self, "run_id", self.run_id.strip())
        if self.snapshot_id is not None:
            if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
                raise ValueError("snapshot_id must be a non-blank string or None")
            object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        if self.strategy_id is not None:
            if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
                raise ValueError("strategy_id must be a non-blank string or None")
            object.__setattr__(self, "strategy_id", " ".join(self.strategy_id.split()))
        if self.universe is not None:
            if isinstance(self.universe, (str, bytes)):
                raise TypeError("universe must be an immutable tuple of symbols")
            values = tuple(str(symbol).strip().upper() for symbol in self.universe)
            if not values or any(not symbol for symbol in values):
                raise ValueError("universe must contain non-blank symbols")
            if len(set(values)) != len(values):
                raise ValueError("universe must contain distinct symbols")
            object.__setattr__(self, "universe", values)
        for name in ("evaluation_start", "evaluation_end"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, datetime) or not isinstance(value, date)
            ):
                raise TypeError(f"{name} must be a calendar date or None")
        if (
            self.evaluation_start is not None
            and self.evaluation_end is not None
            and self.evaluation_start > self.evaluation_end
        ):
            raise ValueError("evaluation_start must not be after evaluation_end")
        for name in ("created_from", "created_to"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise TypeError(f"{name} must be an aware datetime or None")
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ValueError("created_from must not be after created_to")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 0:
            raise ValueError("page must be a non-negative integer")
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")
        if self.state is not None:
            object.__setattr__(self, "state", RunState(self.state))


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """Immutable bounded page used by facade discovery methods."""

    items: tuple[T, ...]
    page: int
    page_size: int
    total: int | None = None
    errors: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("items must be an immutable tuple")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 0:
            raise ValueError("page must be a non-negative integer")
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")
        if len(self.items) > self.page_size:
            raise ValueError("a page cannot contain more than page_size items")
        if self.total is not None and (
            isinstance(self.total, bool) or not isinstance(self.total, int) or self.total < 0
        ):
            raise ValueError("total must be a non-negative integer or None")
        if not isinstance(self.errors, tuple) or any(
            not isinstance(error, ActionableError) for error in self.errors
        ):
            raise TypeError("errors must contain ActionableError values")

    @property
    def has_next(self) -> bool:
        if self.total is not None:
            return (self.page + 1) * self.page_size < self.total
        return len(self.items) == self.page_size

    @property
    def records(self) -> tuple[T, ...]:
        """Compatibility alias for metadata repositories returning records."""

        return self.items

    @property
    def total_count(self) -> int | None:
        return self.total

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)


@dataclass(frozen=True, slots=True)
class _ConfigurationEntry:
    config: ResolvedConfig
    redactor: Redactor


class ResearchApplication:
    """Typed application facade composed entirely from injected services.

    The constructor accepts structural application services so tests can use
    local fakes and the Streamlit composition root can provide concrete
    adapters.  No service locator or presentation state is consulted.
    """

    def __init__(
        self,
        configuration_manager: ConfigurationManager | object | None = None,
        ingestion_service: object | None = None,
        snapshot_manager: object | None = None,
        backtest_service: object | None = None,
        comparison_service: object | None = None,
        inspection_service: object | None = None,
        *,
        config_manager: object | None = None,
        ingestion: object | None = None,
        snapshots: object | None = None,
        backtest: object | None = None,
        comparisons: object | None = None,
        inspection: object | None = None,
        metadata_store: object | None = None,
        run_search: object | None = None,
        logger: object | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.configuration_manager = (
            configuration_manager or config_manager or ConfigurationManager()
        )
        self.ingestion_service = ingestion_service or ingestion
        self.snapshot_manager = snapshot_manager or snapshots
        self.backtest_service = backtest_service or backtest
        self.comparison_service = comparison_service or comparisons
        self.inspection_service = inspection_service or inspection
        self.metadata_store = metadata_store
        self.run_search = run_search or metadata_store
        self.logger = logger
        self._redactor = redactor or Redactor()
        self._configurations: dict[UUID, _ConfigurationEntry] = {}

    def resolve_configuration(
        self,
        yaml_path: Path | None,
        *,
        ui_yaml_values: dict[str, object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> Result[ConfigurationResolution]:
        """Resolve one safe document and retain its exact frozen config privately."""

        correlation_id = str(uuid4())
        source: bytes | str | None
        try:
            source = self._configuration_document(yaml_path, ui_yaml_values)
        except OSError:
            return self._unexpected(
                "configuration.resolve",
                RuntimeError("configuration document could not be read"),
                correlation_id=correlation_id,
            )
        except Exception as error:
            return self._unexpected(
                "configuration.resolve", error, correlation_id=correlation_id
            )

        env = os.environ if environment is None else environment
        try:
            # Register mapped environment secrets before any diagnostics from a
            # resolver/adapter can be written by this facade.
            for name, value in env.items():
                if name.startswith("QRP_SECRETS__") and value:
                    self._redactor.register(value)
            resolve = self._method(self.configuration_manager, ("resolve",))
            if resolve is None:
                return Err(
                    (self._missing_service("configuration manager"),),
                    preserve_order=True,
                )
            resolved = self._invoke(
                resolve,
                positional=(source, env),
                values={"yaml_document": source, "environment": env},
            )
            if isinstance(resolved, Err):
                return self._sanitize_result(resolved, self._redactor)
            config = self._unwrap_result(resolved, "configuration.resolve")
            if not isinstance(config, ResolvedConfig):
                raise TypeError("configuration manager returned an invalid ResolvedConfig")
            view = non_secret_config(config)
            token = UUID(int=uuid4().int)
            self._configurations[token] = _ConfigurationEntry(
                config=config,
                redactor=Redactor.from_config(config),
            )
            return Ok(ConfigurationResolution(ConfigurationHandle(token), view))
        except Exception as error:
            return self._unexpected(
                "configuration.resolve", error, correlation_id=correlation_id
            )

    def invalidate_configuration(self, handle: ConfigurationHandle) -> Result[None]:
        """Explicitly retire a process-local handle so later use is rejected."""

        if not isinstance(handle, ConfigurationHandle):
            return Err((self._handle_error(),), preserve_order=True)
        if self._configurations.pop(handle._token, None) is None:
            return Err((self._handle_error(),), preserve_order=True)
        return Ok(None)

    def ingest(
        self,
        request: IngestionRequest,
        config: ConfigurationHandle,
        *,
        progress: ProgressCallback | None = None,
    ) -> Result[IngestionResult]:
        """Ingest using exactly the frozen configuration behind ``config``."""

        entry = self._configuration_entry(config)
        if isinstance(entry, Err):
            return entry
        service = self.ingestion_service
        if service is None:
            return Err((self._missing_service("ingestion"),), preserve_order=True)
        if not isinstance(request, IngestionRequest):
            return Err((self._input_error("request", "an IngestionRequest is required"),), preserve_order=True)
        method = self._method(service, ("ingest", "run", "execute", "ingest_data"))
        if method is None:
            return Err((self._missing_service("ingestion"),), preserve_order=True)
        return self._delegate_result(
            "ingestion.execute",
            method,
            positional=(request, entry.value.config),
            values={"request": request, "config": entry.value.config, "progress": progress, "progress_callback": progress},
            redactor=entry.value.redactor,
        )

    def list_snapshots(
        self, query: SnapshotQuery | None = None
    ) -> Page[SnapshotSummary]:
        """Return a bounded verified snapshot-discovery page."""

        service = self.snapshot_manager
        if service is None:
            return self._error_page(
                "snapshot.list", self._missing_service("snapshot discovery"), query
            )
        resolved_query = query or SnapshotQuery()
        method = self._method(service, ("list_snapshots", "list"))
        if method is None:
            return self._error_page(
                "snapshot.list", self._missing_service("snapshot discovery"), resolved_query
            )
        try:
            value = self._invoke(
                method,
                positional=(resolved_query,),
                values={"query": resolved_query},
            )
            if isinstance(value, Err):
                return self._page_from_errors(value.errors, resolved_query.page, resolved_query.page_size)
            return self._page_from(value, resolved_query.page, resolved_query.page_size)
        except Exception as error:
            return self._error_page(
                "snapshot.list",
                self._page_unexpected("snapshot.list", error),
                resolved_query,
            )

    def inspect_snapshot(self, snapshot_id: str) -> Result[SnapshotDetail | object]:
        """Open one checksum-verified immutable snapshot detail."""

        service = self.snapshot_manager or self.inspection_service
        method = self._method(service, ("inspect_snapshot", "inspect"))
        if method is None:
            return Err((self._missing_service("snapshot inspection"),), preserve_order=True)
        return self._delegate_result(
            "snapshot.inspect",
            method,
            positional=(snapshot_id,),
            values={"snapshot_id": snapshot_id},
            redactor=self._redactor,
        )

    def run_backtest(
        self,
        request: BacktestRequest,
        config: ConfigurationHandle,
        *,
        progress: ProgressCallback | None = None,
    ) -> Result[object]:
        """Run an audited backtest against one exact snapshot and config."""

        entry = self._configuration_entry(config)
        if isinstance(entry, Err):
            return entry
        service = self.backtest_service
        if service is None:
            return Err((self._missing_service("backtest"),), preserve_order=True)
        if not isinstance(request, BacktestRequest):
            return Err((self._input_error("request", "a BacktestRequest is required"),), preserve_order=True)
        method = self._method(service, ("run", "run_backtest", "execute"))
        if method is None:
            return Err((self._missing_service("backtest"),), preserve_order=True)
        return self._delegate_result(
            "backtest.execute",
            method,
            positional=(request, entry.value.config),
            values={"request": request, "config": entry.value.config, "progress": progress, "progress_callback": progress},
            redactor=entry.value.redactor,
        )

    def search_runs(self, query: RunQuery | None = None) -> Page[RunSummary]:
        """Search indexed run projections with deterministic bounded paging."""

        resolved_query = query or RunQuery()
        service = self.run_search
        if service is None:
            return self._error_page(
                "run.search", self._missing_service("run discovery"), resolved_query
            )
        method = self._method(service, ("search_runs", "list_runs", "search"))
        if method is None:
            return self._error_page(
                "run.search", self._missing_service("run discovery"), resolved_query
            )
        try:
            target_query = self._adapt_run_query(method, resolved_query)
            value = self._invoke(
                method,
                positional=(target_query,),
                values={"query": target_query},
            )
            if isinstance(value, Err):
                return self._page_from_errors(value.errors, resolved_query.page, resolved_query.page_size)
            raw_page = self._page_from(value, resolved_query.page, resolved_query.page_size)
            summaries = tuple(self._run_summary(item) for item in raw_page.items)
            return Page(
                items=summaries,
                page=raw_page.page,
                page_size=raw_page.page_size,
                total=raw_page.total,
                errors=raw_page.errors,
            )
        except Exception as error:
            return self._error_page(
                "run.search",
                self._page_unexpected("run.search", error),
                resolved_query,
            )

    def inspect_run(self, run_id: str | UUID) -> Result[RunDetail]:
        """Return a redacted run manifest, provenance, logs, and artifact index."""

        service = self.inspection_service
        method = self._method(service, ("inspect_run", "inspect_run_details"))
        if method is None:
            return Err((self._missing_service("run inspection"),), preserve_order=True)
        return self._delegate_result(
            "run.inspect",
            method,
            positional=(run_id,),
            values={"run_id": run_id, "id": run_id},
            redactor=self._redactor,
        )

    def compare_runs(self, run_ids: Sequence[str | UUID]) -> Result[ComparisonOutput | object]:
        """Validate and compare 2–10 successful, checksum-verified runs."""

        service = self.comparison_service
        method = self._method(service, ("compare", "compare_runs", "execute"))
        if method is None:
            return Err((self._missing_service("run comparison"),), preserve_order=True)
        if isinstance(run_ids, (str, bytes)):
            selected: Sequence[str | UUID] = (run_ids,)
        else:
            selected = tuple(run_ids)
        return self._delegate_result(
            "comparison.execute",
            method,
            positional=(selected,),
            values={"run_ids": selected, "ids": selected},
            redactor=self._redactor,
        )

    def page_artifact(
        self,
        checksum: str,
        page: int = 0,
        page_size: int | None = None,
        columns: Sequence[str] | None = None,
        *,
        order_by: Sequence[str] | None = None,
    ) -> Result[TablePage]:
        """Read one bounded projected table page through the inspection service."""

        service = self.inspection_service
        method = self._method(service, ("page_artifact", "page_table", "page_artifact_table"))
        if method is None:
            return Err((self._missing_service("artifact paging"),), preserve_order=True)
        return self._delegate_result(
            "artifact.page",
            method,
            positional=(checksum, page, page_size),
            values={
                "checksum": checksum,
                "page": page,
                "page_size": page_size,
                "columns": columns,
                "order_by": order_by,
            },
            redactor=self._redactor,
        )

    def open_artifact(self, checksum: str) -> Result[object]:
        """Open one full artifact through a lazy checksum-verified stream."""

        service = self.inspection_service
        method = self._method(service, ("open_artifact", "open_verified_artifact", "download_artifact"))
        if method is None:
            return Err((self._missing_service("artifact streaming"),), preserve_order=True)
        return self._delegate_result(
            "artifact.verify",
            method,
            positional=(checksum,),
            values={"checksum": checksum},
            redactor=self._redactor,
        )

    # Explicit aliases make the public verbs discoverable without creating a
    # second implementation path.
    resolve_config = resolve_configuration
    run = run_backtest
    inspect_artifact = open_artifact

    def _configuration_entry(
        self, handle: ConfigurationHandle
    ) -> Result[_ConfigurationEntry]:
        if not isinstance(handle, ConfigurationHandle):
            return Err((self._handle_error(),), preserve_order=True)
        entry = self._configurations.get(handle._token)
        if entry is None:
            return Err((self._handle_error(),), preserve_order=True)
        return Ok(entry)

    @staticmethod
    def _configuration_document(
        yaml_path: Path | None, ui_yaml_values: dict[str, object] | None
    ) -> bytes | str | None:
        if yaml_path is None and ui_yaml_values is None:
            return None
        if yaml_path is None:
            base: object = {}
        else:
            base_bytes = Path(yaml_path).read_bytes()
            if ui_yaml_values is None:
                return base_bytes
            parser = YAML(typ="safe", pure=True)
            parser.allow_duplicate_keys = False
            try:
                loaded = parser.load(base_bytes)
            except Exception:
                # Let ConfigurationManager produce the canonical parser error.
                return base_bytes
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, Mapping):
                return base_bytes
            base = loaded
        if not isinstance(ui_yaml_values, Mapping):
            raise TypeError("ui_yaml_values must be a mapping")
        merged = ResearchApplication._deep_merge(base, ui_yaml_values)
        return json.dumps(
            merged,
            ensure_ascii=False,
            separators=(",", ":"),
            default=ResearchApplication._json_default,
        ).encode("utf-8")

    @staticmethod
    def _deep_merge(lower: object, higher: Mapping[str, object]) -> dict[str, object]:
        result = deepcopy(dict(lower)) if isinstance(lower, Mapping) else {}
        for key, value in higher.items():
            existing = result.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                result[str(key)] = ResearchApplication._deep_merge(existing, value)
            else:
                result[str(key)] = deepcopy(value)
        return result

    @staticmethod
    def _json_default(value: object) -> str:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Path):
            return value.as_posix()
        return str(value)

    @staticmethod
    def _method(target: object | None, names: Sequence[str]) -> Callable[..., object] | None:
        if target is None:
            return None
        for name in names:
            candidate = getattr(target, name, None)
            if callable(candidate):
                return cast(Callable[..., object], candidate)
        return None

    @staticmethod
    def _invoke(
        method: Callable[..., object],
        *,
        positional: tuple[object, ...] = (),
        values: Mapping[str, object],
    ) -> object:
        """Call an injected structural port without inventing an API."""

        try:
            parameters = tuple(inspect.signature(method).parameters.values())
        except (TypeError, ValueError):
            return method(*positional, **dict(values))
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            names = tuple(
                parameter.name
                for parameter in parameters
                if parameter.kind
                in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            )
            consumed = set(names[: len(positional)])
            return method(
                *positional,
                **{key: value for key, value in values.items() if key not in consumed},
            )
        positional_parameters = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        )
        if len(positional) > len(positional_parameters):
            positional = ()
        consumed = {parameter.name for parameter in positional_parameters[: len(positional)]}
        accepted = {
            key: value
            for key, value in values.items()
            if key not in consumed
            and any(parameter.name == key for parameter in parameters)
        }
        if positional or accepted:
            return method(*positional, **accepted)
        return method()

    def _delegate_result(
        self,
        operation: str,
        method: Callable[..., object],
        *,
        positional: tuple[object, ...],
        values: Mapping[str, object],
        redactor: Redactor,
    ) -> Result[Any]:
        correlation_id = str(uuid4())
        try:
            value = self._invoke(method, positional=positional, values=values)
            if isinstance(value, Err):
                return cast(Result[Any], self._sanitize_result(value, redactor))
            if isinstance(value, Ok):
                return cast(Result[Any], value)
            return Ok(value)
        except Exception as error:
            return self._unexpected(
                operation,
                error,
                correlation_id=correlation_id,
                redactor=redactor,
            )

    def _sanitize_result(self, result: Err, redactor: Redactor) -> Err:
        sanitized: list[ActionableError] = []
        for error in result.errors:
            value = redactor.redact_error(error)
            if isinstance(value, ActionableError):
                sanitized.append(value)
            else:
                sanitized.append(error)
        return Err(tuple(sanitized), preserve_order=result.preserve_order)

    def _unexpected(
        self,
        operation: str,
        error: BaseException,
        *,
        correlation_id: str,
        redactor: Redactor | None = None,
    ) -> Err:
        safe_redactor = redactor or self._redactor
        safe_message = safe_redactor.redact_text(str(error)) if str(error) else ""
        self._log_unexpected(operation, correlation_id, safe_message, error, safe_redactor)
        actionable = ActionableError.from_unexpected_exception(
            operation, error, correlation_id=correlation_id
        )
        return Err((actionable,), preserve_order=True)

    def _log_unexpected(
        self,
        operation: str,
        correlation_id: str,
        safe_message: str,
        error: BaseException,
        redactor: Redactor,
    ) -> None:
        logger = self.logger
        if logger is None:
            return
        method = getattr(logger, "write", None)
        if not callable(method):
            return
        context = redactor.redact_structured(
            {
                "exception_type": type(error).__name__,
                "sanitized_exception_message": safe_message,
            }
        )
        try:
            self._invoke(
                method,
                values={
                    "level": "error",
                    "operation": operation,
                    "correlation_id": correlation_id,
                    "message": "Unexpected application-boundary exception.",
                    "category": ErrorCategory.INTERNAL_UNEXPECTED.value,
                    "context": context,
                    # Never pass the raw exception to an injected logger.  The
                    # concrete structured logger already receives safe context.
                    "exception": None,
                },
            )
        except Exception:
            return

    def _page_unexpected(
        self,
        operation: str,
        error: BaseException,
    ) -> ActionableError:
        correlation_id = str(uuid4())
        safe_message = self._redactor.redact_text(str(error)) if str(error) else ""
        self._log_unexpected(operation, correlation_id, safe_message, error, self._redactor)
        return ActionableError.from_unexpected_exception(
            operation, error, correlation_id=correlation_id
        )

    @staticmethod
    def _unwrap_result(value: object, operation: str) -> object:
        if isinstance(value, Ok):
            return value.value
        if isinstance(value, Err):
            raise ValueError(f"{operation} returned an error")
        return value

    @staticmethod
    def _missing_service(name: str) -> ActionableError:
        return ActionableError(
            operation="application.compose",
            category=ErrorCategory.STORAGE_IO,
            message=f"The {name} application service is not configured.",
            corrective_action=f"Inject the {name} application service and retry.",
        )

    @staticmethod
    def _input_error(field_path: str, message: str) -> ActionableError:
        return ActionableError(
            operation="application.input",
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message=message,
            corrective_action="Provide the documented typed input and retry.",
            field_path=field_path,
        )

    @staticmethod
    def _handle_error() -> ActionableError:
        return ActionableError(
            operation="configuration.handle",
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message="The configuration handle is unknown or stale in this process.",
            corrective_action="Resolve the configuration again in the current process, then retry.",
            field_path="config",
        )

    @staticmethod
    def _unexpected_error(operation: str, error: BaseException) -> ActionableError:
        del error
        return ActionableError.from_unexpected_exception(operation, RuntimeError("sanitized"))

    @staticmethod
    def _page_from(
        value: object, page: int, page_size: int
    ) -> Page[Any]:
        items_value = getattr(value, "items", None)
        if items_value is None:
            items_value = getattr(value, "records", ())
        if isinstance(items_value, Mapping) or isinstance(items_value, (str, bytes)):
            items = (items_value,)
        else:
            items = tuple(items_value or ())
        errors = getattr(value, "errors", ())
        total = getattr(value, "total", None)
        if total is None:
            total = getattr(value, "total_count", None)
        actual_page = getattr(value, "page", page)
        actual_size = getattr(value, "page_size", page_size)
        return Page(
            items=items,
            page=actual_page,
            page_size=min(actual_size, 100),
            total=total,
            errors=tuple(errors),
        )

    @staticmethod
    def _page_from_errors(
        errors: Sequence[ActionableError], page: int, page_size: int
    ) -> Page[Any]:
        return Page(
            items=(),
            page=page,
            page_size=min(page_size, 100),
            errors=tuple(errors),
        )

    @staticmethod
    def _error_page(
        operation: str, error: ActionableError, query: object | None
    ) -> Page[Any]:
        page = int(getattr(query, "page", 0))
        page_size = min(int(getattr(query, "page_size", 100)), 100)
        del operation
        return Page(items=(), page=page, page_size=page_size, errors=(error,))

    @staticmethod
    def _adapt_run_query(method: Callable[..., object], query: RunQuery) -> object:
        """Adapt to a repository-local query DTO without importing infrastructure."""

        function = getattr(method, "__func__", method)
        namespace = getattr(function, "__globals__", {})
        target_type = namespace.get("RunQuery")
        if not isinstance(target_type, type) or target_type is RunQuery:
            return query
        values: dict[str, object] = {
            "run_id": query.run_id,
            "snapshot_id": query.snapshot_id,
            "strategy_id": query.strategy_id,
            "universe": query.universe,
            "evaluation_start": query.evaluation_start,
            "evaluation_end": query.evaluation_end,
            "state": query.state,
            "created_from": query.created_from,
            "created_to": query.created_to,
            "page": query.page,
            "page_size": query.page_size,
        }
        annotation = getattr(target_type, "__annotations__", {})
        run_id_annotation = annotation.get("run_id")
        if query.run_id is not None and run_id_annotation is not str:
            try:
                values["run_id"] = UUID(query.run_id)
            except (ValueError, AttributeError):
                values["run_id"] = query.run_id
        return target_type(**values)

    @staticmethod
    def _run_summary(record: object) -> RunSummary:
        if isinstance(record, RunSummary):
            return record
        def field(name: str, default: object = None) -> object:
            if isinstance(record, Mapping):
                return record.get(name, default)
            return getattr(record, name, default)
        start = field("evaluation_start")
        end = field("evaluation_end")
        if isinstance(start, datetime) or not isinstance(start, date):
            raise TypeError("run record has no valid evaluation_start")
        if isinstance(end, datetime) or not isinstance(end, date):
            raise TypeError("run record has no valid evaluation_end")
        raw_universe = field("universe", ())
        universe: tuple[str, ...]
        if isinstance(raw_universe, str):
            universe = (raw_universe.strip().upper(),)
        elif isinstance(raw_universe, Sequence) and not isinstance(
            raw_universe, (str, bytes)
        ):
            universe = tuple(str(value).strip().upper() for value in raw_universe)
        else:
            raise TypeError("run record has no valid universe")
        return RunSummary(
            run_id=cast(UUID | str, field("run_id")),
            snapshot_id=str(field("snapshot_id", "")),
            state=field("state", "unknown"),
            strategy_id=str(field("strategy_id", "")),
            evaluation_start=start,
            evaluation_end=end,
            universe=universe,
            configuration_checksum=str(field("configuration_checksum", "")),
            environment_checksum=str(field("environment_checksum", "")),
            manifest_checksum=cast(str | None, field("manifest_checksum")),
            created_at=cast(datetime | None, field("created_at")),
            ended_at=cast(datetime | None, field("ended_at")),
        )


ResearchApplicationFacade = ResearchApplication

__all__ = [
    "ConfigurationHandle",
    "ConfigurationResolution",
    "LimitationDisclosure",
    "Page",
    "ProgressCallback",
    "ResearchApplication",
    "ResearchApplicationFacade",
    "ResolvedConfigView",
    "RunQuery",
]
