"""Framework-independent multi-run comparison orchestration.

The comparison service is intentionally read-only with respect to runs.  It
validates the selected terminal records and their immutable manifest/artifact
references, projects only metric/equity data needed by the comparison, and
publishes one canonical comparison view through an injected artifact port.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from ..config.serializer import non_secret_config
from ..domain.canonical import canonical_json, sha256_bytes
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.evaluation import (
    EvaluationMetrics,
    MetricName,
    MetricNullReason,
    MetricScope,
    MetricValue,
)
from ..domain.execution import RunState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY = re.compile(
    r"(?:secret|password|passwd|token|credential|private|proxy|authorization|api[_-]?key)",
    re.I,
)


class ComparisonMetadataPort(Protocol):
    def get_run(self, run_id: UUID) -> object: ...


class ComparisonArtifactPort(Protocol):
    def open_verified_artifact(self, reference: object) -> object: ...


class ComparisonScannerPort(Protocol):
    def scan(
        self, refs: Sequence[object], columns: Sequence[str], **kwargs: object
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class DifferenceRow:
    """One deterministic provenance difference between selected runs."""

    category: str
    field_path: str
    values: tuple[object, ...]

    @property
    def path(self) -> str:
        return self.field_path

    def to_serializable(self) -> dict[str, object]:
        return {
            "category": self.category,
            "field_path": self.field_path,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class ComparisonCurve:
    """One run's original and aligned strategy/benchmark equity curves."""

    run_id: UUID | str
    snapshot_id: str
    evaluation_start: date
    evaluation_end: date
    strategy_metrics: EvaluationMetrics
    benchmark_metrics: EvaluationMetrics
    strategy_curve: tuple[tuple[date, Decimal], ...]
    benchmark_curve: tuple[tuple[date, Decimal], ...]

    @property
    def original_range(self) -> tuple[date, date]:
        return self.evaluation_start, self.evaluation_end

    def to_serializable(self) -> dict[str, object]:
        return {
            "benchmark_curve": [
                {"equity": value, "session": session}
                for session, value in self.benchmark_curve
            ],
            "benchmark_metrics": self.benchmark_metrics.to_serializable(),
            "evaluation_end": self.evaluation_end,
            "evaluation_start": self.evaluation_start,
            "run_id": str(self.run_id),
            "snapshot_id": self.snapshot_id,
            "strategy_curve": [
                {"equity": value, "session": session}
                for session, value in self.strategy_curve
            ],
            "strategy_metrics": self.strategy_metrics.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonArtifact:
    """Checksummed canonical comparison artifact."""

    checksum: str
    byte_size: int
    payload: bytes
    role: str = "comparison"
    media_type: str = "application/vnd.quant-research.canonical+json"
    schema_version: str = "comparison_v1"
    reference: object | None = None

    def __post_init__(self) -> None:
        if self.checksum != sha256_bytes(self.payload):
            raise ValueError("comparison artifact checksum does not match payload")
        if self.byte_size != len(self.payload):
            raise ValueError("comparison artifact byte size does not match payload")

    @property
    def bytes(self) -> bytes:
        return self.payload


@dataclass(frozen=True, slots=True)
class ComparisonOutput:
    """Complete immutable multi-run comparison DTO."""

    runs: tuple[ComparisonCurve, ...]
    aligned_sessions: tuple[date, ...]
    snapshot_differences: tuple[DifferenceRow, ...]
    configuration_differences: tuple[DifferenceRow, ...]
    environment_differences: tuple[DifferenceRow, ...]
    artifact: ComparisonArtifact
    limitation_disclosure: LimitationDisclosure

    @property
    def comparison_set(self) -> tuple[ComparisonCurve, ...]:
        return self.runs

    @property
    def selected_runs(self) -> tuple[ComparisonCurve, ...]:
        return self.runs

    @property
    def aligned_range(self) -> tuple[date, date] | None:
        if not self.aligned_sessions:
            return None
        return self.aligned_sessions[0], self.aligned_sessions[-1]

    @property
    def differences(self) -> tuple[DifferenceRow, ...]:
        return (
            self.snapshot_differences
            + self.configuration_differences
            + self.environment_differences
        )

    @property
    def artifact_checksum(self) -> str:
        return self.artifact.checksum

    def to_serializable(self) -> dict[str, object]:
        return {
            "aligned_sessions": list(self.aligned_sessions),
            "artifact_checksum": self.artifact.checksum,
            "configuration_differences": [
                row.to_serializable() for row in self.configuration_differences
            ],
            "environment_differences": [
                row.to_serializable() for row in self.environment_differences
            ],
            "runs": [run.to_serializable() for run in self.runs],
            "snapshot_differences": [
                row.to_serializable() for row in self.snapshot_differences
            ],
        }


# Names used by composition roots and tests in earlier implementation waves.
ComparisonResult = ComparisonOutput
ComparisonRun = ComparisonCurve
ComparisonDifference = DifferenceRow


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


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime, Decimal)):
        # Keep canonical scalar types intact.  The canonical encoder owns
        # normalization (notably Decimal trailing-zero removal), so converting
        # them to strings here would make an otherwise identical manifest fail
        # checksum verification.
        return value
    if isinstance(value, UUID):
        return str(value)
    method = getattr(value, "to_serializable", None)
    if callable(method):
        return _plain(method())
    method = getattr(value, "model_dump", None)
    if callable(method):
        return _plain(method(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


def _safe_mapping(
    value: object, redactor: object | None = None
) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    projected: object = value
    with suppress(TypeError, ValueError, AttributeError):
        projected = non_secret_config(cast(Any, value))
    redaction = getattr(redactor, "redact_structured", None) or getattr(
        redactor, "sanitize_metadata", None
    )
    if callable(redaction):
        try:
            projected = redaction(projected)
        except Exception:
            projected = "[REDACTED]"
    plain = _plain(projected)
    if not isinstance(plain, Mapping):
        return MappingProxyType({"value": plain})

    def clean(item: object, key: str | None = None) -> object:
        if key is not None and _SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(item, Mapping):
            return {str(k): clean(v, str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item

    return cast(Mapping[str, object], clean(plain))


def _legacy_plain(value: object) -> object:
    """Normalize legacy fixture manifest values before checksum comparison."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime, UUID, Decimal)):
        return str(value)
    method = getattr(value, "to_serializable", None)
    if callable(method):
        return _legacy_plain(method())
    if isinstance(value, Mapping):
        return {str(key): _legacy_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_legacy_plain(item) for item in value]
    return value


def _date(value: object, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} is not a calendar date") from error
    raise ValueError(f"{name} is not a calendar date")


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} is not a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a finite number") from error
    if not result.is_finite():
        raise ValueError(f"{name} is not finite")
    return result


def _unwrap(value: object, operation: str) -> object:
    if isinstance(value, Ok):
        return value.value
    if isinstance(value, Err):
        raise _ComparisonFailure(value.errors)
    if value is None:
        raise _ComparisonFailure(
            (_error(operation, "The comparison port returned no result."),)
        )
    return value


def _invoke(
    method: Callable[..., object],
    values: Mapping[str, object],
    positional: tuple[object, ...] = (),
) -> object:
    try:
        parameters = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return method(*positional)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return method(**dict(values))
    accepted: dict[str, object] = {}
    positional_only: list[object] = []
    index = 0
    for parameter in parameters:
        if (
            parameter.name == "self"
            or parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ):
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if index < len(positional):
                positional_only.append(positional[index])
                index += 1
        elif parameter.name in values:
            accepted[parameter.name] = values[parameter.name]
    if positional_only:
        return method(*positional_only, **accepted)
    if accepted or not positional:
        return method(**accepted)
    return method(*positional)


def _error(
    operation: str,
    message: str,
    *,
    category: ErrorCategory = ErrorCategory.COMPARISON_SELECTION,
    field_path: str | None = None,
    run_id: object | None = None,
    checksum: str | None = None,
) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=" ".join(message.splitlines()),
        corrective_action=(
            "Select 2–10 distinct successful runs with intact manifests and "
            "artifacts, then retry."
        ),
        field_path=field_path,
        correlation_id=str(run_id) if run_id is not None else None,
        checksum=checksum,
    )


class _ComparisonFailure(Exception):
    def __init__(self, errors: Sequence[ActionableError]) -> None:
        self.errors = tuple(errors)
        super().__init__(self.errors[0].message if self.errors else "comparison failed")


class ComparisonService:
    """Validate, project, align, and publish an ordered run comparison."""

    operation_name = "comparison.execute"

    def __init__(
        self,
        metadata_store: object | None = None,
        artifact_store: object | None = None,
        parquet_store: object | None = None,
        *,
        metadata: object | None = None,
        artifacts: object | None = None,
        scanner: object | None = None,
        manifest_store: object | None = None,
        redactor: object | None = None,
        limitation_disclosure: LimitationDisclosure | None = None,
    ) -> None:
        if metadata_store is not None and metadata is not None:
            raise ValueError("supply either metadata_store or metadata")
        if artifact_store is not None and artifacts is not None:
            raise ValueError("supply either artifact_store or artifacts")
        if parquet_store is not None and scanner is not None:
            raise ValueError("supply either parquet_store or scanner")
        self.metadata_store = metadata_store or metadata
        self.artifact_store = artifact_store or artifacts
        self.parquet_store = parquet_store or scanner
        self.manifest_store = manifest_store
        self.redactor = redactor
        self.limitation_disclosure = (
            limitation_disclosure or LimitationDisclosure.current()
        )

    def compare(self, run_ids: Sequence[UUID | str]) -> Result[ComparisonOutput]:
        try:
            identifiers = self._validate_selection(run_ids)
            records: list[object] = []
            manifests: list[object] = []
            for identifier in identifiers:
                record = self._get_run(identifier)
                state = getattr(
                    _field(record, "state", ""), "value", _field(record, "state", "")
                )
                if state != RunState.SUCCEEDED.value:
                    raise _ComparisonFailure(
                        (
                            _error(
                                "comparison.selection",
                                (
                                    f"Run {identifier} is not successful; "
                                    "only succeeded runs may be compared."
                                ),
                                run_id=identifier,
                            ),
                        )
                    )
                manifest = self._load_manifest(identifier, record)
                self._verify_manifest(identifier, record, manifest)
                self._verify_referenced_artifacts(identifier, record, manifest)
                records.append(record)
                manifests.append(manifest)

            curves: list[ComparisonCurve] = []
            for identifier, record, manifest in zip(
                identifiers, records, manifests, strict=True
            ):
                metrics = self._metrics(identifier, record, manifest)
                strategy_curve = self._curve(identifier, record, manifest, "strategy")
                benchmark_curve = self._curve(identifier, record, manifest, "benchmark")
                start = _date(
                    _field(
                        record, "evaluation_start", _field(manifest, "evaluation_start")
                    ),
                    "evaluation_start",
                )
                end = _date(
                    _field(
                        record, "evaluation_end", _field(manifest, "evaluation_end")
                    ),
                    "evaluation_end",
                )
                curves.append(
                    ComparisonCurve(
                        identifier,
                        str(
                            _field(
                                record,
                                "snapshot_id",
                                _field(manifest, "snapshot_id", ""),
                            )
                        ),
                        start,
                        end,
                        metrics[0],
                        metrics[1],
                        strategy_curve,
                        benchmark_curve,
                    )
                )

            aligned = self._intersection(curves)
            if not aligned:
                raise _ComparisonFailure(
                    (
                        _error(
                            "comparison.alignment",
                            (
                                "The selected runs have no common strategy and "
                                "benchmark equity sessions."
                            ),
                            field_path="aligned_sessions",
                        ),
                    )
                )
            aligned_curves = tuple(
                ComparisonCurve(
                    curve.run_id,
                    curve.snapshot_id,
                    curve.evaluation_start,
                    curve.evaluation_end,
                    curve.strategy_metrics,
                    curve.benchmark_metrics,
                    tuple(
                        (session, dict(curve.strategy_curve)[session])
                        for session in aligned
                    ),
                    tuple(
                        (session, dict(curve.benchmark_curve)[session])
                        for session in aligned
                    ),
                )
                for curve in curves
            )
            snapshot_differences = self._differences(
                "snapshot",
                [
                    self._snapshot_values(record, manifest)
                    for record, manifest in zip(records, manifests, strict=True)
                ],
            )
            configuration_differences = self._differences(
                "configuration",
                [
                    self._configuration_values(record, manifest)
                    for record, manifest in zip(records, manifests, strict=True)
                ],
            )
            environment_differences = self._differences(
                "environment",
                [
                    self._environment_values(record, manifest)
                    for record, manifest in zip(records, manifests, strict=True)
                ],
            )
            artifact = self._publish(
                self._artifact_payload(
                    identifiers,
                    aligned_curves,
                    aligned,
                    snapshot_differences,
                    configuration_differences,
                    environment_differences,
                )
            )
            return Ok(
                ComparisonOutput(
                    aligned_curves,
                    aligned,
                    snapshot_differences,
                    configuration_differences,
                    environment_differences,
                    artifact,
                    self._disclosure(records, manifests),
                )
            )
        except _ComparisonFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except (TypeError, ValueError, KeyError, InvalidOperation) as failure:
            return Err(
                (
                    _error(
                        self.operation_name,
                        str(failure) or "The selected runs are not comparable.",
                    ),
                ),
                preserve_order=True,
            )
        except Exception as failure:
            return Err(
                (
                    ActionableError.from_unexpected_exception(
                        self.operation_name, failure
                    ),
                ),
                preserve_order=True,
            )

    compare_runs = compare
    execute = compare
    run = compare

    def _validate_selection(
        self, run_ids: Sequence[UUID | str]
    ) -> tuple[UUID | str, ...]:
        if isinstance(run_ids, (str, bytes)):
            count = 1
            values: tuple[UUID | str, ...] = (run_ids,)
        else:
            values = tuple(run_ids)
            count = len(values)
        if count < 2:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.selection",
                        (
                            "At least 2 successful runs are required for a comparison; "
                            "the minimum is 2."
                        ),
                        field_path="run_ids",
                    ),
                )
            )
        if count > 10:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.selection",
                        (
                            "At most 10 successful runs may be compared; "
                            "the maximum is 10."
                        ),
                        field_path="run_ids",
                    ),
                )
            )
        normalized: list[UUID | str] = []
        seen: set[str] = set()
        for raw in values:
            try:
                identifier: UUID | str = (
                    raw if isinstance(raw, UUID) else UUID(str(raw))
                )
            except (ValueError, TypeError, AttributeError):
                identifier = str(raw).strip()
            key = str(identifier)
            if not key:
                raise _ComparisonFailure(
                    (
                        _error(
                            "comparison.selection",
                            "Run IDs must not be blank.",
                            field_path="run_ids",
                        ),
                    )
                )
            if key in seen:
                raise _ComparisonFailure(
                    (
                        _error(
                            "comparison.selection",
                            "The selected run IDs must be distinct.",
                            field_path="run_ids",
                            run_id=identifier,
                        ),
                    )
                )
            seen.add(key)
            normalized.append(identifier)
        return tuple(normalized)

    def _get_run(self, identifier: UUID | str) -> object:
        method = self._method(self.metadata_store, ("get_run", "load_run"))
        if method is None:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.selection",
                        (
                            "Run discovery is unavailable because no metadata index "
                            "is configured."
                        ),
                    ),
                )
            )
        try:
            return _unwrap(
                _invoke(
                    method, {"run_id": identifier, "id": identifier}, (identifier,)
                ),
                "comparison.selection",
            )
        except (KeyError, FileNotFoundError):
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.selection",
                        f"Run {identifier} was not found.",
                        run_id=identifier,
                    ),
                )
            ) from None

    def _load_manifest(self, identifier: UUID | str, record: object) -> object:
        value = _field(record, ("manifest", "run_manifest"))
        if value is not None:
            return self._decode(value)
        method = self._method(
            self.manifest_store,
            ("get_run_manifest", "read_run_manifest", "read_manifest", "load_manifest"),
        )
        if method is None:
            method = self._method(
                self.metadata_store,
                ("get_run_manifest", "read_run_manifest", "read_manifest"),
            )
        if method is None:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.integrity",
                        "The selected run has no readable Run Manifest.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                    ),
                )
            )
        try:
            return self._decode(
                _unwrap(
                    _invoke(
                        method, {"run_id": identifier, "id": identifier}, (identifier,)
                    ),
                    "comparison.integrity",
                )
            )
        except (KeyError, FileNotFoundError):
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.integrity",
                        "The selected Run Manifest is missing.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                    ),
                )
            ) from None

    @staticmethod
    def _decode(value: object) -> object:
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                return json.loads(bytes(value).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("manifest bytes are not canonical JSON") from error
        return value

    def _verify_manifest(
        self, identifier: UUID | str, record: object, manifest: object
    ) -> None:
        expected = _field(
            record,
            ("manifest_checksum", "run_manifest_checksum"),
            _field(manifest, ("manifest_checksum", "checksum")),
        )
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.integrity",
                        "The selected run has no valid Run Manifest checksum.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                    ),
                )
            )
        source = _field(record, ("manifest_bytes", "run_manifest_bytes"))
        if source is None:
            source = manifest
        if isinstance(source, (bytes, bytearray, memoryview)):
            actual = sha256_bytes(source)
        else:
            try:
                actual = sha256_bytes(canonical_json(_plain(source)))
            except (TypeError, ValueError):
                actual = ""
        if actual != expected:
            try:
                legacy_actual = sha256_bytes(canonical_json(_legacy_plain(source)))
            except (TypeError, ValueError):
                legacy_actual = ""
            if legacy_actual == expected:
                return
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.integrity",
                        "The selected Run Manifest failed checksum verification.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                        checksum=expected,
                    ),
                )
            )

    def _artifact_links(self, record: object, manifest: object) -> tuple[object, ...]:
        value = _field(
            record,
            ("artifacts", "artifact_references", "run_artifacts"),
            _field(
                manifest,
                ("artifacts", "artifact_references", "scientific_artifacts"),
                (),
            ),
        )
        if isinstance(value, Mapping):
            return tuple(value.values())
        if value is None:
            return ()
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(value)
        return (value,)

    def _verify_referenced_artifacts(
        self, identifier: UUID | str, record: object, manifest: object
    ) -> None:
        links = self._artifact_links(record, manifest)
        for link in links:
            checksum = _field(link, ("checksum", "sha256"))
            if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
                raise _ComparisonFailure(
                    (
                        _error(
                            "comparison.integrity",
                            "The selected run references an invalid artifact checksum.",
                            category=ErrorCategory.INTEGRITY_CHECKSUM,
                            run_id=identifier,
                        ),
                    )
                )
            metadata_method = self._method(
                self.metadata_store, ("get_artifact", "load_artifact")
            )
            metadata = None
            if metadata_method is not None:
                try:
                    metadata = _unwrap(
                        _invoke(metadata_method, {"checksum": checksum}, (checksum,)),
                        "comparison.integrity",
                    )
                except (KeyError, FileNotFoundError):
                    raise _ComparisonFailure(
                        (
                            _error(
                                "comparison.integrity",
                                "A selected run artifact is missing from the "
                                "artifact index.",
                                category=ErrorCategory.INTEGRITY_CHECKSUM,
                                checksum=checksum,
                                run_id=identifier,
                            ),
                        )
                    ) from None
                availability = str(
                    getattr(
                        _field(metadata, "availability", "available"),
                        "value",
                        _field(metadata, "availability", "available"),
                    )
                )
                if availability != "available":
                    raise _ComparisonFailure(
                        (
                            _error(
                                "comparison.integrity",
                                "A selected run artifact is unavailable or invalid.",
                                category=ErrorCategory.INTEGRITY_CHECKSUM,
                                checksum=checksum,
                                run_id=identifier,
                            ),
                        )
                    )
            opener = self._method(
                self.artifact_store,
                (
                    "open_verified_artifact",
                    "verify_artifact",
                    "open_artifact",
                    "stream_artifact",
                ),
            )
            if opener is not None:
                reference = metadata or link
                try:
                    opened = _unwrap(
                        _invoke(
                            opener,
                            {
                                "reference": reference,
                                "artifact": reference,
                                "checksum": checksum,
                                "run_id": identifier,
                            },
                            (reference,),
                        ),
                        "comparison.integrity",
                    )
                    if (
                        isinstance(opened, (bytes, bytearray, memoryview))
                        and sha256_bytes(opened) != checksum
                    ):
                        raise ValueError("artifact checksum mismatch")
                except _ComparisonFailure:
                    raise
                except Exception:
                    raise _ComparisonFailure(
                        (
                            _error(
                                "comparison.integrity",
                                "A selected run artifact failed checksum verification.",
                                category=ErrorCategory.INTEGRITY_CHECKSUM,
                                checksum=checksum,
                                run_id=identifier,
                            ),
                        )
                    ) from None

    def _metrics(
        self, identifier: UUID | str, record: object, manifest: object
    ) -> tuple[EvaluationMetrics, EvaluationMetrics]:
        candidate = _field(
            record,
            ("evaluation", "evaluation_result", "metrics"),
            _field(manifest, ("evaluation_result", "evaluation", "metrics")),
        )
        if candidate is not None:
            strategy = _field(candidate, ("strategy_metrics", "strategy"))
            benchmark = _field(candidate, ("benchmark_metrics", "benchmark"))
            if isinstance(strategy, EvaluationMetrics) and isinstance(
                benchmark, EvaluationMetrics
            ):
                return strategy, benchmark
        rows = _field(
            record,
            ("metric_rows", "metrics_rows"),
            _field(manifest, ("metric_rows", "metrics_rows")),
        )
        if rows is None:
            method = self._method(
                self.metadata_store,
                ("get_run_metrics", "list_run_metrics", "run_metrics"),
            )
            if method is not None:
                rows = _unwrap(
                    _invoke(
                        method,
                        {
                            "run_id": identifier,
                            "scopes": ("strategy", "benchmark"),
                            "columns": (
                                "scope",
                                "metric_name",
                                "metric_value",
                                "null_reason",
                            ),
                        },
                        (identifier,),
                    ),
                    "comparison.metrics",
                )
        if rows is None:
            rows = self._scan_role(
                identifier,
                record,
                manifest,
                "metrics",
                (
                    "scope",
                    "name",
                    "metric_name",
                    "value",
                    "metric_value",
                    "null_reason",
                ),
            )
        built = self._metrics_from_rows(rows)
        strategy_metrics, benchmark_metrics = built
        if strategy_metrics is None or benchmark_metrics is None:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.metrics",
                        (
                            "The selected run has no complete strategy and "
                            "benchmark metric rows."
                        ),
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                    ),
                )
            )
        return strategy_metrics, benchmark_metrics

    def _metrics_from_rows(
        self, rows: object
    ) -> tuple[EvaluationMetrics | None, EvaluationMetrics | None]:
        # Canonical table artifacts are wrapped as ``{"rows": ..., 
        # "schema_version": ...}``; comparison consumes the row payload and
        # must not interpret the envelope metadata as a metric scope.
        if isinstance(rows, Mapping) and "rows" in rows:
            rows = rows["rows"]
        if isinstance(rows, Mapping) and not any(
            key in rows for key in ("scope", "name", "metric_name")
        ):
            expanded: list[object] = []
            for scope, values in rows.items():
                items = values if isinstance(values, (list, tuple)) else (values,)
                for value in items:
                    if isinstance(value, Mapping):
                        expanded.append({**value, "scope": scope})
                    else:
                        expanded.append(value)
            rows = expanded
        grouped: dict[str, dict[str, MetricValue]] = {"strategy": {}, "benchmark": {}}
        for row in self._rows(rows):
            raw_scope = _field(row, "scope", "")
            scope = str(getattr(raw_scope, "value", raw_scope))
            if scope not in grouped:
                continue
            raw_name = _field(row, ("name", "metric_name"), "")
            name = str(getattr(raw_name, "value", raw_name))
            try:
                metric_name = MetricName(name)
            except ValueError:
                continue
            raw_value = _field(row, ("value", "metric_value"))
            raw_reason = _field(row, "null_reason")
            metric_value: Decimal | int | None
            if raw_value is None:
                metric_value = None
            elif metric_name is MetricName.UNFILLED_ORDERS:
                metric_value = int(str(raw_value))
            else:
                metric_value = _decimal(raw_value, name)
            null_reason: MetricNullReason | None = None
            if metric_value is None and raw_reason:
                null_reason = MetricNullReason(
                    str(getattr(raw_reason, "value", raw_reason))
                )
            grouped[scope][metric_name.value] = MetricValue(
                metric_name, metric_value, null_reason
            )
        result: list[EvaluationMetrics | None] = []
        for scope in ("strategy", "benchmark"):
            try:
                names = (
                    MetricName.TOTAL_RETURN,
                    MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
                    MetricName.ANNUALIZED_VOLATILITY,
                    MetricName.SHARPE_RATIO,
                    MetricName.MAXIMUM_DRAWDOWN,
                ) + (
                    (
                        MetricName.TURNOVER,
                        MetricName.TOTAL_COMMISSIONS,
                        MetricName.TOTAL_SLIPPAGE,
                        MetricName.UNFILLED_ORDERS,
                        MetricName.ENDING_CASH_BALANCE,
                    )
                    if scope == "strategy"
                    else ()
                )
                result.append(
                    EvaluationMetrics(
                        MetricScope(scope),
                        tuple(grouped[scope][name.value] for name in names),
                    )
                )
            except (KeyError, TypeError, ValueError):
                result.append(None)
        return result[0], result[1]

    def _curve(
        self, identifier: UUID | str, record: object, manifest: object, scope: str
    ) -> tuple[tuple[date, Decimal], ...]:
        names = (f"{scope}_equity", f"{scope}_curve", f"{scope}_equity_curve")
        value = _field(record, names, _field(manifest, names))
        if value is None:
            value = self._scan_role(
                identifier,
                record,
                manifest,
                f"{scope}_equity",
                ("session", "date", "equity", "value", "portfolio_equity"),
            )
        points: dict[date, Decimal] = {}
        if isinstance(value, Mapping) and not any(
            key in value for key in ("session", "date", "equity", "value")
        ):
            value = [{"session": key, "equity": item} for key, item in value.items()]
        for row in self._rows(value):
            if isinstance(row, (tuple, list)) and len(row) == 2:
                session_value, equity_value = row
            else:
                session_value = _field(row, ("session", "date"))
                equity_value = _field(
                    row, ("equity", "value", "portfolio_equity", "close")
                )
            session = _date(session_value, f"{scope}.session")
            number = _decimal(equity_value, f"{scope}.equity")
            if session in points and points[session] != number:
                raise _ComparisonFailure(
                    (
                        _error(
                            "comparison.alignment",
                            (
                                f"The selected run contains conflicting {scope} "
                                "equity rows."
                            ),
                            category=ErrorCategory.INTEGRITY_CHECKSUM,
                            run_id=identifier,
                        ),
                    )
                )
            points[session] = number
        if not points:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.alignment",
                        f"The selected run has no {scope} equity curve.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        run_id=identifier,
                    ),
                )
            )
        return tuple(sorted(points.items()))

    def _scan_role(
        self,
        identifier: UUID | str,
        record: object,
        manifest: object,
        role: str,
        columns: Sequence[str],
    ) -> object:
        links = self._artifact_links(record, manifest)
        link = next(
            (item for item in links if role in str(_field(item, "role", "")).lower()),
            None,
        )
        if link is None:
            return ()
        checksum = _field(link, "checksum")
        scanner = self._method(
            self.parquet_store, ("scan", "scan_batches", "read_projected")
        )
        if scanner is None:
            return ()
        try:
            return _unwrap(
                _invoke(
                    scanner,
                    {
                        "refs": (link,),
                        "references": (link,),
                        "columns": tuple(columns),
                        "predicate": None,
                        "checksum": checksum,
                        "run_id": identifier,
                    },
                    ((link,), tuple(columns), None),
                ),
                "comparison.scan",
            )
        except _ComparisonFailure:
            raise
        except Exception:
            raise _ComparisonFailure(
                (
                    _error(
                        "comparison.integrity",
                        "A selected comparison artifact could not be projected.",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        checksum=str(checksum),
                        run_id=identifier,
                    ),
                )
            ) from None

    @staticmethod
    def _rows(value: object) -> Iterable[object]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            return (value,)
        result: list[object] = []
        for batch in value:
            if hasattr(batch, "to_pylist"):
                result.extend(batch.to_pylist())
            elif isinstance(batch, Mapping):
                result.append(batch)
            elif isinstance(batch, (list, tuple)):
                batch_values = tuple(batch)
                if batch_values and all(
                    isinstance(item, (Mapping, list, tuple)) for item in batch_values
                ):
                    result.extend(batch_values)
                else:
                    result.append(batch)
            else:
                result.append(batch)
        return tuple(result)

    @staticmethod
    def _intersection(curves: Sequence[ComparisonCurve]) -> tuple[date, ...]:
        common: set[date] | None = None
        for curve in curves:
            sessions = {item[0] for item in curve.strategy_curve} & {
                item[0] for item in curve.benchmark_curve
            }
            common = sessions if common is None else common & sessions
        return tuple(sorted(common or ()))

    def _snapshot_values(
        self, record: object, manifest: object
    ) -> Mapping[str, object]:
        identity = _field(manifest, "content_identity", manifest)
        return {
            "snapshot_id": _field(
                record,
                "snapshot_id",
                _field(identity, "snapshot_id", _field(manifest, "snapshot_id")),
            ),
            "provenance": _safe_mapping(identity, self.redactor),
        }

    def _configuration_values(
        self, record: object, manifest: object
    ) -> Mapping[str, object]:
        return _safe_mapping(
            _field(
                record,
                ("configuration", "resolved_configuration", "non_secret_configuration"),
                _field(
                    manifest,
                    (
                        "configuration",
                        "resolved_configuration",
                        "non_secret_configuration",
                    ),
                ),
            ),
            self.redactor,
        )

    def _environment_values(
        self, record: object, manifest: object
    ) -> Mapping[str, object]:
        return _safe_mapping(
            _field(
                record,
                ("environment_fingerprint", "fingerprint"),
                _field(manifest, ("environment_fingerprint", "fingerprint")),
            ),
            self.redactor,
        )

    @staticmethod
    def _differences(
        category: str, values: Sequence[Mapping[str, object]]
    ) -> tuple[DifferenceRow, ...]:
        flattened: list[Mapping[str, object]] = []

        def walk(value: object, path: str, output: dict[str, object]) -> None:
            if isinstance(value, Mapping):
                if not value and path:
                    output[path] = {}
                for key in sorted(value):
                    walk(value[key], f"{path}.{key}" if path else str(key), output)
            else:
                output[path] = value

        for value in values:
            item: dict[str, object] = {}
            walk(value, "", item)
            flattened.append(item)
        paths = sorted({path for item in flattened for path in item})
        result = []
        for path in paths:
            row_values = tuple(item.get(path) for item in flattened)
            if len(set(canonical_json(item) for item in row_values)) > 1:
                result.append(DifferenceRow(category, path, row_values))
        return tuple(result)

    @staticmethod
    def _artifact_payload(
        identifiers: Sequence[UUID | str],
        curves: Sequence[ComparisonCurve],
        aligned: Sequence[date],
        *differences: Sequence[DifferenceRow],
    ) -> bytes:
        return canonical_json(
            {
                "aligned_sessions": list(aligned),
                "differences": [
                    [row.to_serializable() for row in group] for group in differences
                ],
                "runs": [curve.to_serializable() for curve in curves],
                "selected_run_ids": [str(identifier) for identifier in identifiers],
            }
        )

    def _publish(self, payload: bytes) -> ComparisonArtifact:
        checksum = sha256_bytes(payload)
        artifact = ComparisonArtifact(checksum, len(payload), payload)
        store = self.artifact_store
        if store is None:
            return artifact
        metadata = {
            "artifact_kind": artifact.role,
            "checksum": checksum,
            "byte_size": len(payload),
            "media_type": artifact.media_type,
            "schema_version": artifact.schema_version,
        }
        try:
            method = self._method(store, ("publish_artifact", "put", "store", "write"))
            if method is None:
                raise _ComparisonFailure(
                    (
                        _error(
                            "artifact.publish",
                            "The artifact store has no publication method.",
                            category=ErrorCategory.STORAGE_IO,
                            field_path="artifact_store",
                        ),
                    )
                )
            create_staging = getattr(store, "create_staging", None)
            stage_bytes = getattr(store, "stage_bytes", None)
            if create_staging is not None and stage_bytes is not None:
                staging = _invoke(
                    cast(Callable[..., object], create_staging),
                    {"operation_id": f"comparison-{checksum[:16]}-{uuid4().hex}"},
                )
                staged = _invoke(
                    cast(Callable[..., object], stage_bytes),
                    {
                        "staging": staging,
                        "relative_path": f"comparison/{checksum}.json",
                        "data": payload,
                        "bytes": payload,
                        "expected_checksum": checksum,
                    },
                    (staging, f"comparison/{checksum}.json", payload),
                )
                reference = _invoke(
                    method,
                    {"staged": staged, "artifact": staged, "metadata": metadata},
                    (staged,),
                )
            else:
                reference = _invoke(
                    method,
                    {
                        "payload": payload,
                        "data": payload,
                        "bytes": payload,
                        "metadata": metadata,
                        "checksum": checksum,
                        "role": artifact.role,
                    },
                    (payload,),
                )
            returned = _field(reference, "checksum")
            if returned is not None and returned != checksum:
                raise ValueError("published comparison artifact checksum differs")
            return ComparisonArtifact(
                checksum, len(payload), payload, reference=reference
            )
        except _ComparisonFailure:
            raise
        except Exception:
            raise _ComparisonFailure(
                (
                    _error(
                        "artifact.publish",
                        "The canonical comparison artifact could not be published.",
                        category=ErrorCategory.STORAGE_IO,
                        checksum=checksum,
                    ),
                )
            ) from None

    def _disclosure(
        self, records: Sequence[object], manifests: Sequence[object]
    ) -> LimitationDisclosure:
        for value in (*records, *manifests):
            disclosure = _field(value, "limitation_disclosure")
            if isinstance(disclosure, LimitationDisclosure):
                return disclosure
        return self.limitation_disclosure

    @staticmethod
    def _method(
        target: object | None, names: Sequence[str]
    ) -> Callable[..., object] | None:
        if target is None:
            return None
        for name in names:
            candidate = getattr(target, name, None)
            if callable(candidate):
                return cast(Callable[..., object], candidate)
        return None


Comparison = ComparisonService
ComparisonApplicationService = ComparisonService


__all__ = [
    "Comparison",
    "ComparisonApplicationService",
    "ComparisonArtifact",
    "ComparisonCurve",
    "ComparisonDifference",
    "ComparisonMetadataPort",
    "ComparisonOutput",
    "ComparisonResult",
    "ComparisonRun",
    "ComparisonScannerPort",
    "ComparisonService",
    "DifferenceRow",
]
