"""Local MLflow experiment-tracking boundary.

MLflow is deliberately used as an operational catalog here.  The platform's
DuckDB metadata store remains the lifecycle/index authority and the local
content-addressed artifact store remains the source of scientific bytes.  This
module therefore never uploads a table or chart to MLflow: it records compact,
redacted scalar projections and references (URI/checksum/size) only.
"""

# ruff: noqa: E501, UP035

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, cast
from uuid import UUID, uuid4

from ..domain.canonical import canonical_json_text, sha256_canonical_json
from ..domain.errors import ActionableError, ErrorCategory
from ..domain.evaluation import EvaluationMetrics, EvaluationResult, MetricValue
from ..domain.execution import RunState

if TYPE_CHECKING:
    from .duckdb_metadata import RunFinalization

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY = re.compile(
    r"(?:secret|password|passwd|token|api[_-]?key|credential|authorization|proxy)",
    re.IGNORECASE,
)
_REDACTION = "[REDACTED]"
_MAX_MLFLOW_PARAM_LENGTH = 250


class MlflowTrackerError(RuntimeError):
    """A local tracking failure that must not expose a provider exception."""


class SecretValueError(MlflowTrackerError):
    """Raised when an unsanitized secret would otherwise reach a tracking sink."""


class MlflowClientPort(Protocol):
    """Small structural subset of :class:`mlflow.tracking.MlflowClient`."""

    def get_experiment_by_name(self, name: str) -> object | None: ...

    def create_experiment(self, name: str, **kwargs: object) -> str: ...

    def create_run(self, experiment_id: str, **kwargs: object) -> object: ...

    def log_param(self, run_id: str, key: str, value: object) -> object: ...

    def log_metric(self, run_id: str, key: str, value: float, **kwargs: object) -> object: ...

    def set_tag(self, run_id: str, key: str, value: object) -> object: ...

    def set_terminated(self, run_id: str, **kwargs: object) -> object: ...

    def log_text(self, run_id: str, text: str, artifact_file: str) -> object: ...


class MetadataStorePort(Protocol):
    """Structural metadata methods used by the adapter."""

    def create_run(self, **kwargs: object) -> object: ...

    def set_mlflow_run_id(self, run_id: UUID, mlflow_run_id: str) -> object: ...

    def create_finalization_intent(self, run_id: UUID, finalization: RunFinalization, **kwargs: object) -> object: ...

    def mark_finalization_mlflow_synced(self, run_id: UUID, **kwargs: object) -> object: ...

    def get_finalization_intent(self, run_id: UUID) -> object: ...

    def finalize_run(self, run_id: UUID, finalization: RunFinalization, **kwargs: object) -> object: ...

    def get_run(self, run_id: UUID) -> object: ...


@dataclass(frozen=True, slots=True, init=False)
class RunInputs:
    """Non-secret scientific and operational inputs recorded before execution.

    ``run_id`` and ``platform_run_id`` are accepted as aliases because callers
    use both names at the application/infrastructure boundary.
    """

    platform_run_id: UUID
    snapshot_id: str
    strategy_identifier: str
    strategy_parameters: Mapping[str, object]
    evaluation_start: date
    evaluation_end: date
    configuration: object | None
    environment_fingerprint: object | None
    configuration_checksum: str
    environment_checksum: str
    universe: tuple[str, ...]
    deterministic_seed: int
    secret_values: tuple[str, ...]

    def __init__(
        self,
        platform_run_id: UUID | None = None,
        snapshot_id: str = "",
        strategy_identifier: str = "monthly_momentum_v1",
        strategy_parameters: Mapping[str, object] | object | None = None,
        evaluation_start: date | None = None,
        evaluation_end: date | None = None,
        configuration: object | None = None,
        environment_fingerprint: object | None = None,
        configuration_checksum: str = "",
        environment_checksum: str = "",
        universe: Sequence[str] = (),
        deterministic_seed: int = 0,
        secret_values: Sequence[str] = (),
        *,
        run_id: UUID | None = None,
    ) -> None:
        identifier = platform_run_id or run_id or uuid4()
        if not isinstance(identifier, UUID):
            raise TypeError("platform_run_id must be a UUID")
        if evaluation_start is None or evaluation_end is None:
            raise TypeError("evaluation_start and evaluation_end are required")
        if not isinstance(evaluation_start, date) or isinstance(evaluation_start, datetime):
            raise TypeError("evaluation_start must be a date")
        if not isinstance(evaluation_end, date) or isinstance(evaluation_end, datetime):
            raise TypeError("evaluation_end must be a date")
        if evaluation_start > evaluation_end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        if strategy_parameters is None:
            strategy_parameters = {}
        if isinstance(strategy_parameters, Mapping):
            parameters: Mapping[str, object] = dict(strategy_parameters)
        elif hasattr(strategy_parameters, "to_serializable"):
            parameters = cast(Mapping[str, object], strategy_parameters.to_serializable())
        else:
            parameters = {"value": strategy_parameters}
        object.__setattr__(self, "platform_run_id", identifier)
        object.__setattr__(self, "snapshot_id", str(snapshot_id))
        object.__setattr__(self, "strategy_identifier", str(strategy_identifier))
        object.__setattr__(self, "strategy_parameters", parameters)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_end", evaluation_end)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "environment_fingerprint", environment_fingerprint)
        object.__setattr__(self, "configuration_checksum", str(configuration_checksum))
        object.__setattr__(self, "environment_checksum", str(environment_checksum))
        object.__setattr__(self, "universe", tuple(str(item).strip().upper() for item in universe))
        object.__setattr__(self, "deterministic_seed", deterministic_seed)
        object.__setattr__(self, "secret_values", tuple(str(item) for item in secret_values if str(item)))

    @property
    def run_id(self) -> UUID:
        return self.platform_run_id


@dataclass(frozen=True, slots=True)
class RunHandle:
    """The platform/MLflow one-to-one mapping used by terminal calls."""

    platform_run_id: UUID
    mlflow_run_id: str | None
    state: RunState = RunState.RUNNING

    @property
    def run_id(self) -> UUID:
        return self.platform_run_id

    @property
    def mlflow_id(self) -> str | None:
        return self.mlflow_run_id


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """A verified CAS reference with a streaming opener, never materialized bytes."""

    checksum: str
    relative_uri: str
    byte_size: int
    media_type: str = "application/octet-stream"
    _stream_factory: Callable[[], Iterable[bytes]] | None = None

    def stream(self) -> Iterable[bytes]:
        if self._stream_factory is None:
            raise MlflowTrackerError("artifact has no configured local streaming store")
        return self._stream_factory()


@dataclass(frozen=True, slots=True)
class EvaluatedRun:
    """Optional small result carrier useful to callers outside BacktestService."""

    evaluation: object
    artifacts: tuple[object, ...] = ()
    manifest_checksum: str | None = None
    manifest_uri: str | None = None


@dataclass(frozen=True, slots=True)
class _Artifact:
    checksum: str
    role: str
    uri: str
    byte_size: int | None
    scientific: bool


# MLflow's Run object is intentionally not imported at module import time.  The
# project can import and test all non-MLflow infrastructure when an optional
# environment has not installed runtime dependencies yet.
def _default_client(uri: str) -> MlflowClientPort:
    try:
        from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]
    except Exception as error:  # pragma: no cover - depends on environment
        raise MlflowTrackerError("MLflow is not installed for local tracking") from error
    return cast(MlflowClientPort, MlflowClient(tracking_uri=uri))


def _local_sqlite_uri(value: Path | str) -> str:
    text = str(value)
    if text.startswith("sqlite:///"):
        return text
    if "://" in text:
        raise ValueError("local MLflow tracking must use a SQLite URI")
    path = Path(text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _plain(value: object, *, key: str = "") -> object:
    """Convert domain/Pydantic values without calling SecretStr.get_secret_value."""
    if _SENSITIVE_KEY.search(key):
        return _REDACTION
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime, Decimal)):
        return value.isoformat() if isinstance(value, (date, datetime)) else format(value, "f")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item, key=key) for item in value]
    if hasattr(value, "to_serializable"):
        return _plain(value.to_serializable(), key=key)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=False), key=key)
    if hasattr(value, "get_secret_value"):
        return _REDACTION
    return str(value)


def _redact_text(value: object, secrets: Sequence[str]) -> str:
    text = str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, _REDACTION)
    return text.replace("[REDACTED]", _REDACTION)


def _redact_object(value: object, secrets: Sequence[str]) -> object:
    plain = _plain(value)
    if isinstance(plain, str):
        return _redact_text(plain, secrets)
    if isinstance(plain, list):
        return [_redact_object(item, secrets) for item in plain]
    if isinstance(plain, dict):
        return {
            key: _REDACTION if _SENSITIVE_KEY.search(key) else _redact_object(item, secrets)
            for key, item in plain.items()
        }
    return plain


def _scalar(value: object, secrets: Sequence[str]) -> str:
    plain = _redact_object(value, secrets)
    if isinstance(plain, (dict, list)):
        text = canonical_json_text(cast(object, plain))
    else:
        text = _redact_text(plain, secrets)
    if any(secret and secret in text for secret in secrets):
        raise SecretValueError("secret-bearing MLflow value was not sanitized")
    if len(text) <= _MAX_MLFLOW_PARAM_LENGTH:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"[VALUE_CHECKSUM:{digest}]"


def _experiment_id(client: MlflowClientPort, name: str) -> str:
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        identifier = getattr(existing, "experiment_id", None)
        if identifier is None and isinstance(existing, Mapping):
            identifier = existing.get("experiment_id")
        if identifier is not None:
            return str(identifier)
    return str(client.create_experiment(name))


def _mlflow_run_id(run: object) -> str:
    identifier = getattr(run, "info", None)
    if identifier is not None:
        identifier = getattr(identifier, "run_id", None)
    if identifier is None:
        identifier = getattr(run, "run_id", None)
    if identifier is None and isinstance(run, Mapping):
        identifier = run.get("run_id") or (run.get("info") or {}).get("run_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise MlflowTrackerError("MLflow did not return a run identifier")
    return identifier


def _is_mlflow_id_collision(error: BaseException) -> bool:
    """Recognize only durable one-to-one ID collisions as retryable."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "unique constraint",
            "duplicate key",
            "already bound",
            "mlflow_run_id",
        )
    )


def _get(value: object, *names: str, default: object = None) -> object:
    for name in names:
        current = value
        for part in name.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                current = default
                break
        else:
            return current
    return default


def _as_artifact(value: object, ordinal: int) -> _Artifact | None:
    checksum = _get(value, "checksum", "sha256")
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        return None
    role = _get(value, "role", "name", default=f"artifact_{ordinal}")
    uri = _get(value, "relative_uri", "uri", "path", default=f"artifacts/sha256/{checksum}")
    size = _get(value, "byte_size", "size", default=None)
    scientific = bool(_get(value, "scientific", default=True))
    normalized_size = int(size) if isinstance(size, int) else int(size) if isinstance(size, str) and size.isdigit() else None
    return _Artifact(checksum, str(role), str(uri), normalized_size, scientific)


def _result_artifacts(result: object) -> tuple[_Artifact, ...]:
    candidates = _get(result, "artifacts", "artifact_references", default=())
    if candidates is None:
        candidates = ()
    if isinstance(candidates, Mapping):
        candidates = tuple(candidates.values())
    found = [_as_artifact(item, index) for index, item in enumerate(cast(Iterable[object], candidates))]
    return tuple(sorted((item for item in found if item is not None), key=lambda item: (item.role, item.checksum)))


def _metrics(result: object) -> tuple[EvaluationMetrics, ...]:
    evaluation = _get(result, "evaluation", "evaluation_result", default=result)
    if not isinstance(evaluation, EvaluationResult):
        return tuple(item for item in (_get(evaluation, "strategy_metrics"), _get(evaluation, "benchmark_metrics"), _get(evaluation, "differences")) if isinstance(item, EvaluationMetrics))
    return (evaluation.strategy_metrics, evaluation.benchmark_metrics, evaluation.differences)


def _enum_value(value: object) -> str:
    member = getattr(value, "value", value)
    return str(member)


def _metric_values(metrics: Iterable[EvaluationMetrics]) -> Iterable[tuple[str, MetricValue]]:
    for collection in metrics:
        for metric in collection.metrics:
            yield f"{_enum_value(collection.scope)}.{_enum_value(metric.name)}", metric


class LocalMlflowTracker:
    """Track platform runs in local SQLite MLflow and authoritative local stores."""

    def __init__(
        self,
        tracking_uri: Path | str = "data/mlflow.db",
        *,
        metadata_store: MetadataStorePort | None = None,
        artifact_store: object | None = None,
        client: MlflowClientPort | None = None,
        experiment_name: str = "quant_research_platform",
        metadata: MetadataStorePort | None = None,
        mlflow_uri: Path | str | None = None,
        mlflow_db: Path | str | None = None,
    ) -> None:
        if metadata_store is not None and metadata is not None:
            raise ValueError("supply either metadata_store or metadata")
        if mlflow_uri is not None and mlflow_db is not None:
            raise ValueError("supply either mlflow_uri or mlflow_db")
        metadata_store = metadata_store or metadata
        selected_uri = mlflow_uri or mlflow_db or tracking_uri
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must not be blank")
        self.tracking_uri = _local_sqlite_uri(selected_uri)
        self.client = client or _default_client(self.tracking_uri)
        self.metadata_store = metadata_store
        self.artifact_store = artifact_store
        self.experiment_name = experiment_name.strip()
        self._handles: dict[UUID, RunHandle] = {}
        self._terminal_payloads: dict[UUID, str] = {}

    def allocate_run(self, **values: object) -> RunHandle:
        """Persist the platform running row, then create and bind MLflow."""
        inputs = self._inputs(values)
        if self.metadata_store is not None:
            self.metadata_store.create_run(
                run_id=inputs.platform_run_id,
                snapshot_id=inputs.snapshot_id,
                strategy_id=inputs.strategy_identifier,
                evaluation_start=inputs.evaluation_start,
                evaluation_end=inputs.evaluation_end,
                universe=inputs.universe,
                configuration_checksum=inputs.configuration_checksum or sha256_canonical_json({}),
                environment_checksum=inputs.environment_checksum or sha256_canonical_json({"seed": inputs.deterministic_seed}),
                created_at=cast(datetime, values.get("created_at") or datetime.now(UTC)),
                started_at=cast(datetime, values.get("started_at") or datetime.now(UTC)),
            )
        try:
            experiment = _experiment_id(self.client, self.experiment_name)
            tags = self._input_tags(inputs)
            run_name = f"qrp-{inputs.platform_run_id}"
            run = self.client.create_run(experiment, tags=tags, run_name=run_name)
            mlflow_id = _mlflow_run_id(run)
            if self.metadata_store is not None:
                try:
                    self.metadata_store.set_mlflow_run_id(inputs.platform_run_id, mlflow_id)
                except Exception as mapping_error:
                    # MLflow IDs are globally unique in the real backend.  A
                    # local client double (or a recovered catalog) can still
                    # surface a collision while creating a fresh platform run.
                    # Retry the catalog allocation once, rather than treating
                    # the platform run as failed before execution starts.
                    if not _is_mlflow_id_collision(mapping_error):
                        raise
                    replacement = self.client.create_run(
                        experiment, tags=tags, run_name=run_name
                    )
                    mlflow_id = _mlflow_run_id(replacement)
                    self.metadata_store.set_mlflow_run_id(
                        inputs.platform_run_id, mlflow_id
                    )
            self._log_inputs(mlflow_id, inputs)
        except Exception as error:
            actionable = self._tracking_error("experiment.allocate", error, inputs.platform_run_id)
            self._terminalize_platform_failure(inputs.platform_run_id, actionable, values)
            raise MlflowTrackerError(actionable.message) from None
        handle = RunHandle(inputs.platform_run_id, mlflow_id)
        self._handles[inputs.platform_run_id] = handle
        return handle

    create_run = allocate_run
    start_run = allocate_run
    begin_run = allocate_run

    def finalize_success(self, run: RunHandle | UUID | object, result: object) -> RunHandle:
        handle = self._handle(run)
        metrics = _metrics(result)
        artifacts = _result_artifacts(result)
        from .duckdb_metadata import RunArtifactLink, RunFinalization

        finalization = RunFinalization(
            desired_state=RunState.SUCCEEDED,
            manifest_checksum=cast(str | None, _get(result, "manifest_checksum", default=_get(_get(result, "manifest", default=None), "checksum"))),
            manifest_uri=cast(str | None, _get(result, "manifest_uri", default=_get(_get(result, "manifest", default=None), "relative_uri"))),
            metrics=tuple(sorted(metrics, key=lambda item: _enum_value(item.scope))),
            artifacts=tuple(RunArtifactLink(item.checksum, item.role, item.scientific) for item in artifacts),
        )
        payload_key = finalization.payload_checksum
        if self._is_terminal(handle, RunState.SUCCEEDED, payload_key):
            return RunHandle(handle.platform_run_id, handle.mlflow_run_id, RunState.SUCCEEDED)
        if handle.state is not RunState.RUNNING:
            raise MlflowTrackerError("terminal runs are immutable; create a new Run ID")
        self._create_intent(handle.platform_run_id, finalization)
        try:
            self._log_terminal(handle, result, metrics, artifacts, RunState.SUCCEEDED)
        except Exception as error:
            actionable = self._tracking_error(
                "experiment.finalize.mlflow", error, handle.platform_run_id
            )
            with suppress(Exception):
                self._mark_mlflow_sync_failure(handle.platform_run_id, actionable)
            raise MlflowTrackerError(actionable.message) from None
        self._sync_intent(handle.platform_run_id)
        self._finalize_metadata(handle.platform_run_id, finalization, result)
        self._terminal_payloads[handle.platform_run_id] = payload_key
        terminal = RunHandle(handle.platform_run_id, handle.mlflow_run_id, RunState.SUCCEEDED)
        self._handles[handle.platform_run_id] = terminal
        return terminal

    succeed = finalize_success
    record_success = finalize_success
    complete_run = finalize_success

    def finalize_failure(
        self,
        run: RunHandle | UUID | object,
        errors: Sequence[ActionableError],
        diagnostics: Sequence[object] = (),
    ) -> RunHandle:
        handle = self._handle(run)
        safe_errors = tuple(error for error in errors if isinstance(error, ActionableError))
        if not safe_errors:
            safe_errors = (self._tracking_error("experiment.finalize_failure", RuntimeError("failure"), handle.platform_run_id),)
        artifacts = tuple(item for item in (_as_artifact(value, index) for index, value in enumerate(diagnostics)) if item is not None)
        from .duckdb_metadata import RunArtifactLink, RunFinalization

        finalization = RunFinalization(
            desired_state=RunState.FAILED,
            manifest_checksum=None,
            manifest_uri=None,
            artifacts=tuple(RunArtifactLink(item.checksum, item.role, item.scientific) for item in sorted(artifacts, key=lambda item: (item.role, item.checksum))),
            errors=safe_errors,
        )
        payload_key = finalization.payload_checksum
        if self._is_terminal(handle, RunState.FAILED, payload_key):
            return RunHandle(handle.platform_run_id, handle.mlflow_run_id, RunState.FAILED)
        if handle.state is not RunState.RUNNING:
            raise MlflowTrackerError("terminal runs are immutable; create a new Run ID")
        self._create_intent(handle.platform_run_id, finalization)
        try:
            self._log_terminal(
                handle, None, (), artifacts, RunState.FAILED, errors=safe_errors
            )
        except Exception as error:
            actionable = self._tracking_error(
                "experiment.finalize_failure.mlflow", error, handle.platform_run_id
            )
            with suppress(Exception):
                self._mark_mlflow_sync_failure(handle.platform_run_id, actionable)
            raise MlflowTrackerError(actionable.message) from None
        self._sync_intent(handle.platform_run_id)
        self._finalize_metadata(handle.platform_run_id, finalization, None)
        self._terminal_payloads[handle.platform_run_id] = payload_key
        terminal = RunHandle(handle.platform_run_id, handle.mlflow_run_id, RunState.FAILED)
        self._handles[handle.platform_run_id] = terminal
        return terminal

    fail = finalize_failure
    record_failure = finalize_failure
    fail_run = finalize_failure

    def open_verified_artifact(self, run_id: UUID | str, checksum: str) -> VerifiedArtifact:
        """Verify and return a streamable local artifact reference."""
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        if self.metadata_store is None:
            raise MlflowTrackerError("metadata store is required for artifact lookup")
        record = self.metadata_store.get_artifact(checksum)  # type: ignore[attr-defined]
        if str(_get(record, "availability", default="available")) not in {"available", "SnapshotAvailability.AVAILABLE"}:
            raise MlflowTrackerError("artifact is unavailable or failed integrity verification")
        if self.artifact_store is None:
            raise MlflowTrackerError("artifact store is required for verified access")
        reference = _get(self.artifact_store, "artifact_reference", default=None)
        del reference
        stream_method = getattr(self.artifact_store, "stream_artifact", None)
        if not callable(stream_method):
            raise MlflowTrackerError("artifact store does not provide verified streaming")
        try:
            from .filesystem_store import ArtifactReference
            record_size = _get(record, "byte_size", default=0)
            artifact_reference = ArtifactReference(
                checksum=checksum,
                byte_size=int(record_size) if isinstance(record_size, (int, str)) else 0,
                relative_uri=str(_get(record, "relative_uri")),
                metadata_checksum=checksum,
            )
        except Exception as error:
            raise MlflowTrackerError("artifact metadata could not be converted to a verified reference") from error
        return VerifiedArtifact(
            checksum=checksum,
            relative_uri=artifact_reference.relative_uri,
            byte_size=artifact_reference.byte_size,
            media_type=str(_get(record, "media_type", default="application/octet-stream")),
            _stream_factory=lambda: cast(Iterable[bytes], stream_method(artifact_reference)),
        )

    def _inputs(self, values: Mapping[str, object]) -> RunInputs:
        request = values.get("request") or values.get("backtest_request")
        snapshot_id = str(values.get("snapshot_id") or _get(request, "snapshot_id", default=""))
        evaluation_range = values.get("evaluation_range") or _get(request, "evaluation_range", default=None)
        start = values.get("evaluation_start") or _get(evaluation_range, "start", default=None)
        end = values.get("evaluation_end") or _get(evaluation_range, "end", default=None)
        config = values.get("config") or values.get("resolved_config")
        strategy_parameters = values.get("strategy_parameters") or _get(config, "strategy", default={})
        fingerprint = values.get("environment_fingerprint") or values.get("fingerprint")
        return RunInputs(
            platform_run_id=cast(UUID | None, values.get("run_id") or values.get("platform_run_id")),
            snapshot_id=snapshot_id,
            strategy_identifier=str(values.get("strategy_identifier") or values.get("strategy_id") or _get(config, "strategy.identifier", default="monthly_momentum_v1")),
            strategy_parameters=strategy_parameters,
            evaluation_start=cast(date, start),
            evaluation_end=cast(date, end),
            configuration=config,
            environment_fingerprint=fingerprint,
            configuration_checksum=str(values.get("configuration_checksum") or ""),
            environment_checksum=str(values.get("environment_checksum") or ""),
            universe=cast(Sequence[str], values.get("universe") or _get(config, "data.universe", default=())),
            deterministic_seed=int(cast(int | str, values.get("deterministic_seed", _get(config, "runtime.deterministic_seed", default=0)))),
            secret_values=cast(Sequence[str], values.get("secret_values", ())),
        )

    def _input_tags(self, inputs: RunInputs) -> dict[str, str]:
        return {
            "qrp.run_id": str(inputs.platform_run_id),
            "qrp.snapshot_id": _scalar(inputs.snapshot_id, inputs.secret_values),
            "qrp.strategy": _scalar(inputs.strategy_identifier, inputs.secret_values),
            "qrp.evaluation_start": inputs.evaluation_start.isoformat(),
            "qrp.evaluation_end": inputs.evaluation_end.isoformat(),
            "qrp.state": RunState.RUNNING.value,
            "qrp.deterministic_seed": str(inputs.deterministic_seed),
        }

    def _log_inputs(self, mlflow_id: str, inputs: RunInputs) -> None:
        params: dict[str, str] = {}
        params.update({f"strategy.{key}": _scalar(value, inputs.secret_values) for key, value in inputs.strategy_parameters.items()})
        params["snapshot_id"] = _scalar(inputs.snapshot_id, inputs.secret_values)
        params["evaluation_start"] = inputs.evaluation_start.isoformat()
        params["evaluation_end"] = inputs.evaluation_end.isoformat()
        params["universe"] = _scalar(inputs.universe, inputs.secret_values)
        params["configuration_checksum"] = _scalar(inputs.configuration_checksum, inputs.secret_values)
        params["environment_checksum"] = _scalar(inputs.environment_checksum, inputs.secret_values)
        for key, value in params.items():
            self.client.log_param(mlflow_id, key, value)
        config = _redact_object(inputs.configuration, inputs.secret_values) if inputs.configuration is not None else {}
        fingerprint = _redact_object(inputs.environment_fingerprint, inputs.secret_values) if inputs.environment_fingerprint is not None else {}
        self.client.log_text(mlflow_id, canonical_json_text({"configuration": config, "environment_fingerprint": fingerprint}), "inputs.json")

    def _log_terminal(
        self,
        handle: RunHandle,
        result: object | None,
        metrics: Iterable[EvaluationMetrics],
        artifacts: Iterable[_Artifact],
        state: RunState,
        *,
        errors: Sequence[ActionableError] = (),
    ) -> None:
        if handle.mlflow_run_id is None:
            return
        for name, metric in _metric_values(metrics):
            if metric.value is not None:
                self.client.log_metric(handle.mlflow_run_id, name, float(metric.value))
            elif metric.null_reason is not None:
                self.client.set_tag(handle.mlflow_run_id, f"metric.{name}.null_reason", _enum_value(metric.null_reason))
        references = []
        for artifact in artifacts:
            references.append({"role": artifact.role, "checksum": artifact.checksum, "uri": artifact.uri, "byte_size": artifact.byte_size, "scientific": artifact.scientific})
            self.client.set_tag(handle.mlflow_run_id, f"artifact.{artifact.role}.checksum", artifact.checksum)
            self.client.set_tag(handle.mlflow_run_id, f"artifact.{artifact.role}.uri", _scalar(artifact.uri, ()))
        if errors:
            self.client.log_text(handle.mlflow_run_id, canonical_json_text({"errors": [error.format_for_display() for error in errors]}), "diagnostics.json")
        if references:
            self.client.log_text(handle.mlflow_run_id, canonical_json_text({"references": references}), "artifact-references.json")
        if result is not None:
            manifest = _get(result, "manifest", "run_manifest", default=None)
            if manifest is not None:
                self.client.log_text(handle.mlflow_run_id, canonical_json_text(_plain(manifest)), "run-manifest.json")
        self.client.set_tag(handle.mlflow_run_id, "qrp.state", state.value)
        self.client.set_terminated(handle.mlflow_run_id, status="FINISHED" if state is RunState.SUCCEEDED else "FAILED", end_time=int(datetime.now(UTC).timestamp() * 1000))

    def _create_intent(self, run_id: UUID, finalization: RunFinalization) -> None:
        if self.metadata_store is not None:
            self.metadata_store.create_finalization_intent(run_id, finalization, created_at=datetime.now(UTC))

    def _sync_intent(self, run_id: UUID) -> None:
        if self.metadata_store is not None:
            self.metadata_store.mark_finalization_mlflow_synced(run_id, attempted_at=datetime.now(UTC))

    def _mark_mlflow_sync_failure(
        self, run_id: UUID, error: ActionableError
    ) -> None:
        if self.metadata_store is not None:
            self.metadata_store.mark_finalization_mlflow_synced(
                run_id, attempted_at=datetime.now(UTC), error=error
            )

    def _finalize_metadata(self, run_id: UUID, finalization: RunFinalization, result: object | None) -> None:
        if self.metadata_store is not None:
            self.metadata_store.finalize_run(run_id, finalization, ended_at=cast(datetime, _get(result, "ended_at", default=datetime.now(UTC))))

    def _handle(self, value: RunHandle | UUID | object) -> RunHandle:
        if isinstance(value, RunHandle):
            return value
        if isinstance(value, UUID):
            if value in self._handles:
                return self._handles[value]
            if self.metadata_store is not None:
                try:
                    record = self.metadata_store.get_run(value)
                    state = RunState(_enum_value(_get(record, "state", default=RunState.RUNNING.value)))
                    handle = RunHandle(value, cast(str | None, _get(record, "mlflow_run_id", default=None)), state)
                    self._handles[value] = handle
                    return handle
                except Exception:
                    pass
            raise MlflowTrackerError(f"run {value} is not known to this tracker")
        identifier = _get(value, "platform_run_id", "run_id")
        if isinstance(identifier, UUID):
            return RunHandle(identifier, cast(str | None, _get(value, "mlflow_run_id", default=None)), RunState(cast(str, _get(value, "state", default=RunState.RUNNING.value))))
        raise TypeError("run must be a RunHandle or platform Run ID")

    def _is_terminal(self, handle: RunHandle, state: RunState, payload: str) -> bool:
        """Return true only for an exact terminal payload replay.

        A terminal state alone is not an idempotency key: a second terminal
        request with different metrics, artifacts, or errors must be rejected.
        """
        if self._terminal_payloads.get(handle.platform_run_id) == payload:
            return True
        if self.metadata_store is None:
            return False
        try:
            record = self.metadata_store.get_run(handle.platform_run_id)
            persisted_state = RunState(_enum_value(_get(record, "state", default="running")))
            if persisted_state is not state:
                return False
            intent = self.metadata_store.get_finalization_intent(handle.platform_run_id)
            persisted_intent_state = RunState(
                _enum_value(_get(intent, "desired_state", default=RunState.RUNNING.value))
            )
            persisted_payload = str(_get(intent, "terminal_payload_checksum", default=""))
            return persisted_intent_state is state and persisted_payload == payload
        except Exception:
            # The normal create-intent path reports the durable repository
            # error.  A transient read failure must never make a conflicting
            # terminal request look idempotent.
            return False

    def _tracking_error(self, operation: str, error: BaseException, run_id: UUID) -> ActionableError:
        del error
        return ActionableError(
            operation=operation,
            category=ErrorCategory.EXPERIMENT_RECORDING,
            message="Local MLflow recording failed; the platform run was retained with diagnostics.",
            corrective_action="Inspect the local tracking database and retry with a new Run ID.",
            correlation_id=str(run_id),
        )

    def _terminalize_platform_failure(self, run_id: UUID, error: ActionableError, values: Mapping[str, object]) -> None:
        if self.metadata_store is None:
            return
        from .duckdb_metadata import RunFinalization

        try:
            finalization = RunFinalization(desired_state=RunState.FAILED, manifest_checksum=None, manifest_uri=None, errors=(error,))
            self.metadata_store.create_finalization_intent(run_id, finalization, created_at=datetime.now(UTC))
            # A failed run with no MLflow mapping is still a valid terminal
            # platform record; the repository's sync flag means there is no
            # pending MLflow work for this failed attempt.
            self.metadata_store.mark_finalization_mlflow_synced(run_id, attempted_at=datetime.now(UTC))
            self.metadata_store.finalize_run(run_id, finalization, ended_at=cast(datetime, values.get("ended_at") or datetime.now(UTC)))
        except Exception:
            # The original allocation error is the useful diagnostic.  Startup
            # reconciliation can inspect the still-running row if this fallback
            # itself is unavailable.
            return


MlflowTracker = LocalMlflowTracker
MLflowTracker = LocalMlflowTracker
LocalMLflowTracker = LocalMlflowTracker
ExperimentTracker = LocalMlflowTracker

__all__ = [
    "EvaluatedRun",
    "ExperimentTracker",
    "LocalMlflowTracker",
    "LocalMLflowTracker",
    "MLflowTracker",
    "MlflowClientPort",
    "MlflowTracker",
    "MlflowTrackerError",
    "RunHandle",
    "RunInputs",
    "SecretValueError",
    "VerifiedArtifact",
]
