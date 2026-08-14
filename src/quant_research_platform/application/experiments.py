"""Framework-independent experiment lifecycle orchestration.

The application tracker is the boundary between backtest/evaluation services and
local tracking adapters.  It keeps operational run identifiers and timestamps
out of scientific identity, prepares immutable artifact references before a
terminal transition, and turns adapter failures into typed, display-safe
results.  Concrete MLflow, DuckDB, and filesystem implementations are injected
through structural ports; this module does not import any infrastructure
framework.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from ..domain.canonical import sha256_bytes, sha256_canonical_json
from ..domain.errors import ActionableError, Err, ErrorCategory, LimitationDisclosure, Ok, Result
from ..domain.evaluation import EvaluationMetrics
from ..domain.execution import RunState
from ..domain.manifests import ContentAddressedObjectRef, ObjectKind

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MetadataPort(Protocol):
    """Durable platform metadata operations used by the boundary."""

    def get_run(self, run_id: UUID) -> object: ...

    def get_artifact(self, checksum: str) -> object: ...

    def set_artifact_availability(self, checksum: str, availability: str) -> object: ...


class TrackingPort(Protocol):
    """Local tracker operations; implementations may return values or Result."""


class ArtifactPort(Protocol):
    """Structural artifact-store operations used for publication and streaming."""


class ClockPort(Protocol):
    def utc_now(self) -> datetime: ...


class _SystemClock:
    def utc_now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True, init=False)
class RunInputs:
    """Validated scientific inputs recorded before execution begins.

    ``run_id`` is optional so the tracker can allocate it.  The alias
    ``platform_run_id`` is accepted for callers that already use the metadata
    vocabulary.
    """

    run_id: UUID | None
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
    limitation_disclosure: LimitationDisclosure

    def __init__(
        self,
        run_id: UUID | None = None,
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
        limitation_disclosure: LimitationDisclosure | None = None,
        *,
        platform_run_id: UUID | None = None,
    ) -> None:
        if run_id is not None and platform_run_id is not None and run_id != platform_run_id:
            raise ValueError("run_id and platform_run_id must agree")
        identifier = run_id or platform_run_id
        if identifier is not None and not isinstance(identifier, UUID):
            raise TypeError("run_id must be a UUID or None")
        if evaluation_start is None or evaluation_end is None:
            raise TypeError("evaluation_start and evaluation_end are required")
        if isinstance(evaluation_start, datetime) or not isinstance(evaluation_start, date):
            raise TypeError("evaluation_start must be a calendar date")
        if isinstance(evaluation_end, datetime) or not isinstance(evaluation_end, date):
            raise TypeError("evaluation_end must be a calendar date")
        if evaluation_start > evaluation_end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        snapshot = str(snapshot_id).strip()
        if not snapshot:
            raise ValueError("snapshot_id must not be blank")
        strategy = " ".join(str(strategy_identifier).split())
        if not strategy:
            raise ValueError("strategy_identifier must not be blank")
        if strategy_parameters is None:
            parameters: Mapping[str, object] = {}
        elif isinstance(strategy_parameters, Mapping):
            parameters = dict(strategy_parameters)
        elif hasattr(strategy_parameters, "to_serializable"):
            parameters = cast(Mapping[str, object], strategy_parameters.to_serializable())
        else:
            parameters = {"value": strategy_parameters}
        normalized_universe = tuple(str(symbol).strip().upper() for symbol in universe)
        if any(not symbol for symbol in normalized_universe):
            raise ValueError("universe must not contain blank symbols")
        if len(set(normalized_universe)) != len(normalized_universe):
            raise ValueError("universe must contain distinct symbols")
        disclosure = limitation_disclosure or LimitationDisclosure.current()
        if not isinstance(disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be LimitationDisclosure")
        object.__setattr__(self, "run_id", identifier)
        object.__setattr__(self, "snapshot_id", snapshot)
        object.__setattr__(self, "strategy_identifier", strategy)
        object.__setattr__(self, "strategy_parameters", parameters)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_end", evaluation_end)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "environment_fingerprint", environment_fingerprint)
        object.__setattr__(self, "configuration_checksum", str(configuration_checksum))
        object.__setattr__(self, "environment_checksum", str(environment_checksum))
        object.__setattr__(self, "universe", normalized_universe)
        object.__setattr__(self, "deterministic_seed", deterministic_seed)
        object.__setattr__(self, "secret_values", tuple(str(item) for item in secret_values if str(item)))
        object.__setattr__(self, "limitation_disclosure", disclosure)

    @property
    def platform_run_id(self) -> UUID | None:
        return self.run_id


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Opaque operational handle returned after a running row is allocated."""

    run_id: UUID
    mlflow_run_id: str | None = None
    state: RunState = RunState.RUNNING

    @property
    def platform_run_id(self) -> UUID:
        return self.run_id

    @property
    def mlflow_id(self) -> str | None:
        return self.mlflow_run_id


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """A checksummed artifact reference safe to put in a terminal payload."""

    checksum: str
    role: str
    relative_uri: str
    byte_size: int
    media_type: str = "application/octet-stream"
    schema_version: str | None = None
    row_count: int | None = None
    scientific: bool = True
    payload: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checksum, str) or _SHA256.fullmatch(self.checksum) is None:
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("artifact role must not be blank")
        if not isinstance(self.relative_uri, str) or not self.relative_uri.strip():
            raise ValueError("artifact URI must not be blank")
        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("artifact byte_size must be non-negative")
        if self.payload is not None:
            if not isinstance(self.payload, bytes):
                raise TypeError("artifact payload must be bytes")
            if len(self.payload) != self.byte_size:
                raise ValueError("artifact byte_size must equal payload length")
            if sha256_bytes(self.payload) != self.checksum:
                raise ValueError("artifact payload checksum does not match checksum")
        if self.row_count is not None and (isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 0):
            raise ValueError("artifact row_count must be non-negative")
        object.__setattr__(self, "role", " ".join(self.role.split()))


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """A lazy, checksum-verified stream without materializing artifact bytes."""

    run_id: UUID
    checksum: str
    relative_uri: str
    byte_size: int
    media_type: str
    availability: str
    _stream_factory: Callable[[], Iterable[bytes]]

    def stream(self) -> Iterable[bytes]:
        return self._stream_factory()

    @property
    def valid(self) -> bool:
        return self.availability == "available"


def _field(value: object, names: str | Sequence[str], default: object = None) -> object:
    candidates = (names,) if isinstance(names, str) else tuple(names)
    if isinstance(value, Mapping):
        for name in candidates:
            if name in value:
                return value[name]
        return default
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _unwrap(value: object, operation: str) -> object:
    if isinstance(value, Err):
        raise _ExpectedFailure(value.errors)
    if isinstance(value, Ok):
        return value.value
    if value is None:
        raise _ExpectedFailure((_error(operation, "The tracking adapter returned no result."),))
    return value


class _ExpectedFailure(Exception):
    def __init__(self, errors: Sequence[ActionableError]) -> None:
        self.errors = tuple(errors)
        super().__init__(self.errors[0].message if self.errors else "operation failed")


def _error(operation: str, message: str, *, category: ErrorCategory = ErrorCategory.EXPERIMENT_RECORDING, checksum: str | None = None, run_id: UUID | None = None) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=" ".join(message.splitlines()) or "The experiment operation failed.",
        corrective_action="Inspect the local run and artifact stores, then retry with a new Run ID.",
        checksum=checksum,
        correlation_id=str(run_id) if run_id is not None else None,
    )


def _invoke(method: Callable[..., object], *, positional: tuple[object, ...] = (), values: Mapping[str, object] = {}) -> object:
    """Call a structural port using only parameters declared by that port."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(*positional, **dict(values))
    parameters = tuple(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        explicit = tuple(parameter for parameter in parameters if parameter.name != "self" and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL})
        if not explicit or all(parameter.name in values for parameter in explicit):
            return method(**dict(values))
        return method(*positional, **dict(values))
    accepted: dict[str, object] = {}
    positional_only: list[object] = []
    position = 0
    for parameter in parameters:
        if parameter.name == "self" or parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if position < len(positional):
                positional_only.append(positional[position])
                position += 1
            continue
        if parameter.name in values:
            accepted[parameter.name] = values[parameter.name]
    if positional_only:
        return method(*tuple(positional_only), **accepted)
    if accepted or not positional:
        return method(**accepted)
    return method(*positional)


def _checksum(value: object, fallback: object) -> str:
    if isinstance(value, str) and _SHA256.fullmatch(value) is not None:
        return value
    return sha256_canonical_json(fallback)


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime, UUID)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    method = getattr(value, "to_serializable", None)
    if callable(method):
        return _plain(method())
    method = getattr(value, "model_dump", None)
    if callable(method):
        return _plain(method(exclude_none=False))
    return type(value).__name__


def _metrics(result: object) -> tuple[EvaluationMetrics, ...]:
    candidate = _field(result, ("evaluation", "evaluation_result"), result)
    if isinstance(candidate, EvaluationMetrics):
        return (candidate,)
    values = tuple(
        item for item in (
            _field(candidate, "strategy_metrics"),
            _field(candidate, "benchmark_metrics"),
            _field(candidate, "differences"),
        ) if isinstance(item, EvaluationMetrics)
    )
    return tuple(sorted(values, key=lambda item: str(getattr(item.scope, "value", item.scope))))


def _artifact_candidates(result: object) -> tuple[object, ...]:
    value = _field(result, ("artifacts", "artifact_references"), ())
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    try:
        return cast(tuple[object, ...], tuple(cast(Iterable[object], value)))
    except TypeError:
        return (value,)


def _descriptor(value: object, ordinal: int) -> ArtifactDescriptor | None:
    checksum = _field(value, ("checksum", "sha256"))
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        return None
    role = str(_field(value, ("role", "name"), f"artifact_{ordinal}"))
    uri = str(_field(value, ("relative_uri", "uri", "path"), f"artifacts/sha256/{checksum}"))
    payload = _field(value, ("payload", "bytes"), None)
    if payload is not None and not isinstance(payload, bytes):
        payload = None
    size = _field(value, ("byte_size", "size"), len(payload) if payload is not None else 0)
    try:
        byte_size = int(cast(int | str, size))
    except (TypeError, ValueError):
        byte_size = len(payload) if payload is not None else 0
    return ArtifactDescriptor(
        checksum=checksum,
        role=role,
        relative_uri=uri,
        byte_size=byte_size,
        media_type=str(_field(value, "media_type", "application/octet-stream")),
        schema_version=cast(str | None, _field(value, "schema_version")),
        row_count=cast(int | None, _field(value, "row_count")),
        scientific=bool(_field(value, "scientific", True)),
        payload=payload,
    )


class ExperimentTracker:
    """Typed application orchestration over local metadata, MLflow, and CAS."""

    operation_name = "experiment"

    def __init__(
        self,
        metadata_store: MetadataPort | None = None,
        mlflow_tracker: TrackingPort | None = None,
        artifact_store: ArtifactPort | None = None,
        *,
        tracker: TrackingPort | None = None,
        metadata: MetadataPort | None = None,
        clock: ClockPort | None = None,
    ) -> None:
        if metadata_store is not None and metadata is not None:
            raise ValueError("supply either metadata_store or metadata")
        if mlflow_tracker is not None and tracker is not None:
            raise ValueError("supply either mlflow_tracker or tracker")
        self.metadata_store = metadata_store or metadata
        self.mlflow_tracker = mlflow_tracker or tracker
        self.artifact_store = artifact_store
        self.clock = clock or _SystemClock()
        self._handles: dict[UUID, RunHandle] = {}
        self._payloads: dict[UUID, tuple[RunState, str]] = {}

    def create_run(self, inputs: RunInputs | Mapping[str, object] | None = None, **values: object) -> Result[RunHandle]:
        """Allocate a platform Run_ID and durable running row before work."""
        try:
            normalized = self._inputs(inputs, values)
            identifier = normalized.run_id or uuid4()
            delegated_values = self._tracking_values(normalized, identifier, values)
            # LocalMlflowTracker owns the exact metadata-first sequence when it
            # was constructed with this metadata store.  For simpler injected
            # fakes, create the row here before invoking their adapter.
            tracker_owns_metadata = self.metadata_store is not None and _field(self.mlflow_tracker, "metadata_store") is self.metadata_store
            if self.metadata_store is not None and not tracker_owns_metadata:
                self._metadata_create(normalized, identifier)
            if self.mlflow_tracker is None:
                if self.metadata_store is None:
                    raise _ExpectedFailure((_error("experiment.allocate", "No tracking or metadata adapter was configured."),))
                handle = RunHandle(identifier)
            else:
                method = self._method(self.mlflow_tracker, ("allocate_run", "create_run", "start_run", "begin_run"))
                allocated = _unwrap(_invoke(method, values=delegated_values), "experiment.allocate")
                handle = self._handle_from(allocated, identifier)
            self._handles[identifier] = handle
            return Ok(handle)
        except _ExpectedFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except Exception as failure:
            return Err((ActionableError.from_unexpected_exception("experiment.allocate", failure),), preserve_order=True)

    allocate_run = create_run
    start_run = create_run
    begin_run = create_run

    def succeed(self, run: RunHandle | UUID, result: object) -> Result[RunHandle]:
        """Publish/verify terminal references, then perform durable success replay."""
        try:
            handle = self._resolve_handle(run)
            descriptors = self._prepare_artifacts(result, handle.run_id)
            disclosure = _field(result, "limitation_disclosure", LimitationDisclosure.current())
            if not isinstance(disclosure, LimitationDisclosure):
                raise _ExpectedFailure((_error("experiment.finalize", "A run result lacks a valid limitation disclosure.", run_id=handle.run_id),))
            manifest_checksum, manifest_uri = self._manifest(result)
            if manifest_checksum is None:
                raise _ExpectedFailure((_error("experiment.finalize", "A successful run requires a checksummed Run Manifest.", run_id=handle.run_id),))
            scientific_checksums = tuple(item.checksum for item in descriptors if item.scientific)
            payload_checksum = sha256_canonical_json({
                "snapshot_id": _field(result, "snapshot_id", ""),
                "manifest_checksum": manifest_checksum,
                "scientific_artifacts": scientific_checksums,
                "disclosure": disclosure.version,
            })
            existing = self._payloads.get(handle.run_id)
            if existing is not None:
                if existing != (RunState.SUCCEEDED, payload_checksum):
                    raise _ExpectedFailure((_error("experiment.finalize", "The terminal run already has a different success payload.", run_id=handle.run_id),))
                return Ok(RunHandle(handle.run_id, handle.mlflow_run_id, RunState.SUCCEEDED))
            if handle.state is not RunState.RUNNING:
                raise _ExpectedFailure((_error("experiment.finalize", "Terminal runs are immutable; create a new Run ID.", run_id=handle.run_id),))
            method = self._method(self.mlflow_tracker, ("finalize_success", "succeed", "record_success", "complete_run"))
            finalized = _unwrap(_invoke(method, positional=(handle, result), values={"run": handle, "run_handle": handle, "run_id": handle.run_id, "result": result, "backtest_result": result, "artifacts": descriptors, "manifest_checksum": manifest_checksum, "manifest_uri": manifest_uri, "ended_at": self._now()}), "experiment.finalize")
            terminal = self._handle_from(finalized, handle.run_id, state=RunState.SUCCEEDED)
            self._payloads[handle.run_id] = (RunState.SUCCEEDED, payload_checksum)
            self._handles[handle.run_id] = terminal
            return Ok(terminal)
        except _ExpectedFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except Exception as failure:
            return Err((ActionableError.from_unexpected_exception("experiment.finalize", failure, correlation_id=str(run) if isinstance(run, UUID) else None),), preserve_order=True)

    finalize_success = succeed
    record_success = succeed
    complete_run = succeed

    def fail(self, run: RunHandle | UUID, errors: Sequence[ActionableError], diagnostics: Sequence[object] = ()) -> Result[RunHandle]:
        """Preserve diagnostic artifacts and terminalize the run as failed."""
        try:
            handle = self._resolve_handle(run)
            safe_errors = tuple(error for error in errors if isinstance(error, ActionableError))
            if not safe_errors:
                safe_errors = (_error("experiment.finalize_failure", "The run failed without a structured diagnostic.", run_id=handle.run_id),)
            descriptors = self._prepare_artifacts_from_values(diagnostics, handle.run_id)
            payload_checksum = sha256_canonical_json({
                "state": RunState.FAILED.value,
                "errors": [error.format_for_display() for error in safe_errors],
                "diagnostics": [item.checksum for item in descriptors],
            })
            existing = self._payloads.get(handle.run_id)
            if existing is not None:
                if existing != (RunState.FAILED, payload_checksum):
                    raise _ExpectedFailure((_error("experiment.finalize_failure", "The terminal run already has a different failure payload.", run_id=handle.run_id),))
                return Ok(RunHandle(handle.run_id, handle.mlflow_run_id, RunState.FAILED))
            if handle.state is not RunState.RUNNING:
                raise _ExpectedFailure((_error("experiment.finalize_failure", "Terminal runs are immutable; create a new Run ID.", run_id=handle.run_id),))
            method = self._method(self.mlflow_tracker, ("finalize_failure", "fail", "record_failure", "fail_run"))
            finalized = _unwrap(_invoke(method, positional=(handle, safe_errors), values={"run": handle, "run_handle": handle, "run_id": handle.run_id, "errors": safe_errors, "diagnostics": descriptors, "ended_at": self._now()}), "experiment.finalize_failure")
            terminal = self._handle_from(finalized, handle.run_id, state=RunState.FAILED)
            self._payloads[handle.run_id] = (RunState.FAILED, payload_checksum)
            self._handles[handle.run_id] = terminal
            return Ok(terminal)
        except _ExpectedFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except Exception as failure:
            return Err((ActionableError.from_unexpected_exception("experiment.finalize_failure", failure, correlation_id=str(run) if isinstance(run, UUID) else None),), preserve_order=True)

    finalize_failure = fail
    record_failure = fail
    fail_run = fail

    def open_verified_artifact(self, run_id: UUID | str, checksum: str) -> Result[VerifiedArtifact]:
        """Verify run association and checksum before returning lazy access."""
        try:
            identifier = UUID(str(run_id))
            if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
                raise _ExpectedFailure((_error("artifact.verify", "The artifact checksum is invalid.", category=ErrorCategory.INTEGRITY_CHECKSUM, checksum=str(checksum), run_id=identifier),))
            if self.metadata_store is not None:
                self.metadata_store.get_run(identifier)
                self._assert_run_artifact(identifier, checksum)
            if self.mlflow_tracker is not None:
                method = getattr(self.mlflow_tracker, "open_verified_artifact", None)
                if callable(method):
                    opened = _unwrap(_invoke(method, positional=(identifier, checksum), values={"run_id": identifier, "checksum": checksum}), "artifact.verify")
                    artifact = self._artifact_from_opened(identifier, checksum, opened)
                else:
                    artifact = self._open_from_store(identifier, checksum)
            else:
                artifact = self._open_from_store(identifier, checksum)
            self._verify_metadata_available(checksum)
            return Ok(self._guard_stream(artifact))
        except _ExpectedFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except Exception as failure:
            failed_identifier = UUID(str(run_id)) if _looks_like_uuid(run_id) else None
            self._mark_invalid(checksum)
            return Err((ActionableError(
                operation="artifact.verify",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                message="The run artifact failed checksum or publication verification and was marked invalid.",
                corrective_action="Restore the immutable artifact bytes or select another verified artifact.",
                checksum=checksum if isinstance(checksum, str) else None,
                correlation_id=str(failed_identifier) if failed_identifier else None,
            ),), preserve_order=True)

    open_artifact = open_verified_artifact
    verify_and_open_artifact = open_verified_artifact

    def _inputs(self, inputs: RunInputs | Mapping[str, object] | None, values: Mapping[str, object]) -> RunInputs:
        if isinstance(inputs, RunInputs):
            if values:
                raise ValueError("keyword values cannot accompany RunInputs")
            return inputs
        merged = dict(inputs) if isinstance(inputs, Mapping) else {}
        merged.update(values)
        request = merged.get("request") or merged.get("backtest_request")
        evaluation_range = merged.get("evaluation_range") or _field(request, "evaluation_range")
        start = merged.get("evaluation_start") or _field(evaluation_range, "start")
        end = merged.get("evaluation_end") or _field(evaluation_range, "end")
        config = merged.get("config") or merged.get("resolved_config")
        data = _field(config, "data", config)
        strategy = _field(config, "strategy", config)
        runtime = _field(config, "runtime", config)
        return RunInputs(
            run_id=cast(UUID | None, merged.get("run_id") or merged.get("platform_run_id")),
            snapshot_id=str(merged.get("snapshot_id") or _field(request, "snapshot_id", "")),
            strategy_identifier=str(merged.get("strategy_identifier") or merged.get("strategy_id") or _field(strategy, "identifier", "monthly_momentum_v1")),
            strategy_parameters=merged.get("strategy_parameters", strategy),
            evaluation_start=cast(date, start),
            evaluation_end=cast(date, end),
            configuration=config,
            environment_fingerprint=merged.get("environment_fingerprint") or merged.get("fingerprint"),
            configuration_checksum=_checksum(merged.get("configuration_checksum"), {"configuration": _plain(config)}),
            environment_checksum=_checksum(merged.get("environment_checksum"), {"environment": _plain(merged.get("environment_fingerprint") or merged.get("fingerprint"))}),
            universe=cast(Sequence[str], merged.get("universe") or _field(data, "universe", ())),
            deterministic_seed=int(cast(int | str, merged.get("deterministic_seed", _field(runtime, "deterministic_seed", 0)))),
            secret_values=cast(Sequence[str], merged.get("secret_values", ())),
            limitation_disclosure=cast(LimitationDisclosure | None, merged.get("limitation_disclosure")),
        )

    def _tracking_values(self, inputs: RunInputs, run_id: UUID, supplied: Mapping[str, object]) -> dict[str, object]:
        values = dict(supplied)
        values.update({
            "run_id": run_id,
            "platform_run_id": run_id,
            "snapshot_id": inputs.snapshot_id,
            "strategy_id": inputs.strategy_identifier,
            "strategy_identifier": inputs.strategy_identifier,
            "strategy_parameters": inputs.strategy_parameters,
            "evaluation_start": inputs.evaluation_start,
            "evaluation_end": inputs.evaluation_end,
            "configuration": inputs.configuration,
            "resolved_config": inputs.configuration,
            "environment_fingerprint": inputs.environment_fingerprint,
            "configuration_checksum": inputs.configuration_checksum,
            "environment_checksum": inputs.environment_checksum,
            "universe": inputs.universe,
            "deterministic_seed": inputs.deterministic_seed,
            "secret_values": inputs.secret_values,
            "created_at": self._now(),
            "started_at": self._now(),
            "state": RunState.RUNNING.value,
        })
        return values

    def _metadata_create(self, inputs: RunInputs, run_id: UUID) -> None:
        method = getattr(self.metadata_store, "create_run", None)
        if not callable(method):
            raise _ExpectedFailure((_error("experiment.allocate", "The metadata adapter cannot create a running run.", run_id=run_id),))
        _invoke(method, values={
            "run_id": run_id,
            "snapshot_id": inputs.snapshot_id,
            "strategy_id": inputs.strategy_identifier,
            "evaluation_start": inputs.evaluation_start,
            "evaluation_end": inputs.evaluation_end,
            "universe": inputs.universe,
            "configuration_checksum": inputs.configuration_checksum,
            "environment_checksum": inputs.environment_checksum,
            "created_at": self._now(),
            "started_at": self._now(),
        })

    def _prepare_artifacts(self, result: object, run_id: UUID) -> tuple[ArtifactDescriptor, ...]:
        return self._prepare_artifacts_from_values(_artifact_candidates(result), run_id)

    def _prepare_artifacts_from_values(self, values: Sequence[object], run_id: UUID) -> tuple[ArtifactDescriptor, ...]:
        descriptors = tuple(item for index, value in enumerate(values) if (item := _descriptor(value, index)) is not None)
        prepared: list[ArtifactDescriptor] = []
        for descriptor in sorted(descriptors, key=lambda item: (item.role, item.checksum)):
            current = descriptor
            if current.payload is not None and self.artifact_store is not None:
                current = self._publish_artifact(current, run_id)
            self._record_artifact(current)
            prepared.append(current)
        return tuple(prepared)

    def _publish_artifact(self, artifact: ArtifactDescriptor, run_id: UUID) -> ArtifactDescriptor:
        store = self.artifact_store
        assert store is not None
        metadata = {"artifact_kind": artifact.role, "checksum": artifact.checksum, "byte_size": artifact.byte_size, "media_type": artifact.media_type, "schema_version": artifact.schema_version, "row_count": artifact.row_count}
        try:
            publish = getattr(store, "publish_artifact", None)
            create_staging = getattr(store, "create_staging", None)
            stage_bytes = getattr(store, "stage_bytes", None)
            if callable(publish) and callable(create_staging) and callable(stage_bytes):
                staging = _invoke(
                    create_staging,
                    values={"operation_id": f"experiment-{artifact.checksum[:16]}-{uuid4().hex}"},
                )
                staged = _invoke(
                    stage_bytes,
                    positional=(staging, f"runs/{run_id}/{artifact.role}-{artifact.checksum}.bin", artifact.payload),
                    values={
                        "staging": staging,
                        "relative_path": f"runs/{run_id}/{artifact.role}-{artifact.checksum}.bin",
                        "data": artifact.payload,
                        "bytes": artifact.payload,
                        "expected_checksum": artifact.checksum,
                    },
                )
                reference = _invoke(
                    publish,
                    positional=(staged,),
                    values={"staged": staged, "artifact": staged, "metadata": metadata},
                )
                return ArtifactDescriptor(artifact.checksum, artifact.role, str(_field(reference, ("relative_uri", "uri"), artifact.relative_uri)), artifact.byte_size, artifact.media_type, artifact.schema_version, artifact.row_count, artifact.scientific, None)
            for name in ("publish_artifact", "put", "store", "write"):
                method = getattr(store, name, None)
                if not callable(method):
                    continue
                reference = _invoke(method, positional=(artifact.payload,), values={"payload": artifact.payload, "data": artifact.payload, "bytes": artifact.payload, "metadata": metadata, "role": artifact.role, "checksum": artifact.checksum})
                return ArtifactDescriptor(artifact.checksum, artifact.role, str(_field(reference, ("relative_uri", "uri"), artifact.relative_uri)), artifact.byte_size, artifact.media_type, artifact.schema_version, artifact.row_count, artifact.scientific, None)
            raise _ExpectedFailure((_error("artifact.publish", "The artifact store has no publication method.", run_id=run_id),))
        except _ExpectedFailure:
            raise
        except Exception as failure:
            raise _ExpectedFailure((ActionableError.from_unexpected_exception("artifact.publish", failure, correlation_id=artifact.checksum),)) from None

    def _record_artifact(self, artifact: ArtifactDescriptor) -> None:
        if self.metadata_store is None:
            return
        method = getattr(self.metadata_store, "record_artifact", None)
        if not callable(method):
            return
        reference = ContentAddressedObjectRef(
            object_kind=ObjectKind.ARTIFACT,
            checksum=artifact.checksum,
            relative_uri=artifact.relative_uri,
            schema_version=artifact.schema_version or "artifact_v1",
            row_count=artifact.row_count or 0,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
        )
        _invoke(method, values={"reference": reference, "artifact_kind": artifact.role, "created_at": self._now()})

    def _open_from_store(self, run_id: UUID, checksum: str) -> VerifiedArtifact:
        if self.artifact_store is None:
            raise _ExpectedFailure((_error("artifact.verify", "No artifact store was configured.", category=ErrorCategory.INTEGRITY_CHECKSUM, checksum=checksum, run_id=run_id),))
        record = self.metadata_store.get_artifact(checksum) if self.metadata_store is not None else None
        reference = _field(record, ("reference", "artifact_reference"), record)
        method = getattr(self.artifact_store, "open_verified_artifact", None) or getattr(self.artifact_store, "stream_artifact", None)
        if not callable(method):
            raise _ExpectedFailure((_error("artifact.verify", "The artifact store has no verified streaming method.", category=ErrorCategory.INTEGRITY_CHECKSUM, checksum=checksum, run_id=run_id),))
        opened = _invoke(method, positional=(reference,), values={"reference": reference, "checksum": checksum})
        stream = opened if callable(opened) else (lambda opened=opened: cast(Iterable[bytes], opened))
        return VerifiedArtifact(run_id, checksum, str(_field(record, "relative_uri", _field(reference, "relative_uri", ""))), int(cast(int | str, _field(record, "byte_size", _field(reference, "byte_size", 0)))), str(_field(record, "media_type", "application/octet-stream")), "available", cast(Callable[[], Iterable[bytes]], stream))

    def _artifact_from_opened(self, run_id: UUID, checksum: str, opened: object) -> VerifiedArtifact:
        stream_method = getattr(opened, "stream", None)
        if not callable(stream_method):
            if callable(opened):
                stream_method = opened
            else:
                stream_method = lambda: cast(Iterable[bytes], opened)
        record = self.metadata_store.get_artifact(checksum) if self.metadata_store is not None else None
        return VerifiedArtifact(run_id, checksum, str(_field(opened, "relative_uri", _field(record, "relative_uri", ""))), int(cast(int | str, _field(opened, "byte_size", _field(record, "byte_size", 0)))), str(_field(opened, "media_type", _field(record, "media_type", "application/octet-stream"))), "available", cast(Callable[[], Iterable[bytes]], stream_method))

    def _guard_stream(self, artifact: VerifiedArtifact) -> VerifiedArtifact:
        return VerifiedArtifact(
            run_id=artifact.run_id,
            checksum=artifact.checksum,
            relative_uri=artifact.relative_uri,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
            availability=artifact.availability,
            _stream_factory=lambda: self._stream_with_guard(artifact),
        )

    def _stream_with_guard(self, artifact: VerifiedArtifact) -> Iterable[bytes]:
        try:
            yield from artifact.stream()
        except Exception:
            self._mark_invalid(artifact.checksum)
            raise

    def _verify_metadata_available(self, checksum: str) -> None:
        if self.metadata_store is None:
            return
        record = self.metadata_store.get_artifact(checksum)
        availability = _field(record, "availability", "available")
        if str(getattr(availability, "value", availability)) not in {"available", "SnapshotAvailability.AVAILABLE"}:
            raise _ExpectedFailure((_error("artifact.verify", "The artifact is unavailable or already invalid.", category=ErrorCategory.INTEGRITY_CHECKSUM, checksum=checksum),))

    def _assert_run_artifact(self, run_id: UUID, checksum: str) -> None:
        for name in ("list_run_artifacts", "get_run_artifacts", "run_artifacts"):
            method = getattr(self.metadata_store, name, None)
            if not callable(method):
                continue
            links = _invoke(method, positional=(run_id,), values={"run_id": run_id})
            values = cast(tuple[object, ...], tuple(cast(Iterable[object], links or ())))
            if not any(str(_field(link, "checksum")) == checksum for link in values):
                raise _ExpectedFailure((_error("artifact.verify", "The artifact is not referenced by this run.", category=ErrorCategory.INTEGRITY_CHECKSUM, checksum=checksum, run_id=run_id),))
            return

    def _mark_invalid(self, checksum: object) -> None:
        if self.metadata_store is None or not isinstance(checksum, str):
            return
        method = getattr(self.metadata_store, "set_artifact_availability", None)
        if callable(method):
            try:
                method(checksum, "invalid")
            except Exception:
                pass

    def _resolve_handle(self, value: RunHandle | UUID) -> RunHandle:
        if isinstance(value, RunHandle):
            return value
        if not isinstance(value, UUID):
            raise _ExpectedFailure((_error("experiment.run", "The run handle is invalid."),))
        handle = self._handles.get(value)
        if handle is not None:
            return handle
        if self.metadata_store is not None:
            record = self.metadata_store.get_run(value)
            state = RunState(cast(str | RunState, _field(record, "state", RunState.RUNNING)))
            handle = RunHandle(value, cast(str | None, _field(record, "mlflow_run_id")), state)
            self._handles[value] = handle
            return handle
        raise _ExpectedFailure((_error("experiment.run", "The run is not known to this process.", run_id=value),))

    @staticmethod
    def _require_running(handle: RunHandle) -> None:
        if handle.state is not RunState.RUNNING:
            raise _ExpectedFailure((_error("experiment.finalize", "Terminal runs are immutable; create a new Run ID.", run_id=handle.run_id),))

    @staticmethod
    def _handle_from(value: object, run_id: UUID, *, state: RunState = RunState.RUNNING) -> RunHandle:
        if isinstance(value, RunHandle):
            return value
        identifier = _field(value, ("run_id", "platform_run_id", "id"), run_id)
        if not isinstance(identifier, UUID):
            identifier = run_id
        selected_state = _field(value, "state", state)
        return RunHandle(identifier, cast(str | None, _field(value, ("mlflow_run_id", "mlflow_id"))), RunState(cast(str | RunState, selected_state)))

    @staticmethod
    def _method(target: object | None, names: Sequence[str]) -> Callable[..., object]:
        if target is None:
            raise _ExpectedFailure((_error("experiment.adapter", "No tracking adapter was configured."),))
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return cast(Callable[..., object], method)
        raise _ExpectedFailure((_error("experiment.adapter", "The tracking adapter does not implement the required lifecycle method."),))

    @staticmethod
    def _manifest(result: object) -> tuple[str | None, str | None]:
        manifest = _field(result, ("manifest", "run_manifest"))
        checksum = _field(result, ("manifest_checksum", "run_manifest_checksum"), _field(manifest, ("manifest_checksum", "checksum")))
        uri = _field(result, ("manifest_uri", "run_manifest_uri"), _field(manifest, ("manifest_uri", "relative_uri")))
        return (str(checksum) if isinstance(checksum, str) else None, str(uri) if isinstance(uri, str) else None)

    def _now(self) -> datetime:
        value = self.clock.utc_now()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("clock must return an aware UTC timestamp")
        return value


def _looks_like_uuid(value: object) -> bool:
    try:
        UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


ExperimentTrackerService = ExperimentTracker

__all__ = [
    "ArtifactDescriptor",
    "ExperimentTracker",
    "ExperimentTrackerService",
    "MetadataPort",
    "RunHandle",
    "RunInputs",
    "TrackingPort",
    "VerifiedArtifact",
]
