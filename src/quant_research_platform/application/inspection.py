"""Framework-independent inspection and bounded artifact access.

This module is the read-only application boundary for the inspection views.  It
composes the verified snapshot manager, run metadata index, artifact store, and
projected table scanner through small structural ports.  It intentionally does
not import Streamlit, DuckDB, PyArrow, or a filesystem implementation.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol, cast
from uuid import UUID

from ..config.serializer import Redactor, non_secret_config
from ..domain.canonical import canonical_json, sha256_bytes
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.manifests import ArtifactReference, ContentAddressedObjectRef, ObjectKind
from .experiments import VerifiedArtifact

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InspectionMetadataPort(Protocol):
    """Minimal metadata surface used by inspection."""

    def get_run(self, run_id: UUID) -> object: ...

    def get_artifact(self, checksum: str) -> object: ...


class InspectionArtifactPort(Protocol):
    """Verified artifact access surface."""

    def open_verified_artifact(self, reference: object) -> object: ...


class InspectionScannerPort(Protocol):
    """Projected table scan surface."""

    def scan(
        self, refs: Sequence[object], columns: Sequence[str], predicate: object = None
    ) -> object: ...


class InspectionSnapshotPort(Protocol):
    """Verified snapshot inspection surface."""

    def inspect_snapshot(self, snapshot_id: str) -> Result[object]: ...


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Redacted immutable metadata for one content-addressed artifact."""

    checksum: str
    artifact_kind: str
    relative_uri: str
    media_type: str
    byte_size: int
    row_count: int | None = None
    schema_version: str | None = None
    availability: str = "available"
    role: str | None = None
    scientific: bool | None = None
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checksum, str)
            or _SHA256.fullmatch(self.checksum) is None
        ):
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        for name in ("artifact_kind", "relative_uri", "media_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not " ".join(value.split()):
                raise ValueError(f"{name} must not be blank")
            object.__setattr__(
                self, name, " ".join(value.split()) if name != "relative_uri" else value
            )
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise ValueError("byte_size must be a non-negative integer")
        if self.row_count is not None and (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count < 0
        ):
            raise ValueError("row_count must be a non-negative integer or None")
        normalized = (
            str(getattr(self.availability, "value", self.availability)).strip().lower()
        )
        if normalized not in {"available", "unavailable", "invalid"}:
            raise ValueError("availability must be available, unavailable, or invalid")
        object.__setattr__(self, "availability", normalized)
        if self.role is not None:
            object.__setattr__(self, "role", " ".join(self.role.split()))
        if self.scientific is not None and not isinstance(self.scientific, bool):
            raise TypeError("scientific must be a bool or None")
        if not isinstance(self.columns, tuple) or any(
            not isinstance(column, str) or not column for column in self.columns
        ):
            raise TypeError("columns must be an immutable tuple of names")

    @property
    def kind(self) -> str:
        return self.artifact_kind

    @property
    def uri(self) -> str:
        return self.relative_uri

    @property
    def size(self) -> int:
        return self.byte_size

    @property
    def valid(self) -> bool:
        return self.availability == "available"


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Small run projection suitable for discovery and inspection headers."""

    run_id: UUID | str
    snapshot_id: str
    state: object
    strategy_id: str
    evaluation_start: date
    evaluation_end: date
    universe: tuple[str, ...]
    configuration_checksum: str
    environment_checksum: str
    manifest_checksum: str | None
    created_at: datetime | None
    ended_at: datetime | None


@dataclass(frozen=True, slots=True)
class RunDetail:
    """Verified, redacted run inspection projection.

    The manifest/configuration/fingerprint fields remain structural objects so
    this DTO can be used with both domain manifests and lightweight repository
    records.  Values crossing this boundary have already passed the injected
    redactor.
    """

    summary: RunSummary
    run_record: object
    manifest: object | None
    configuration: object | None
    environment_fingerprint: object | None
    validation_report: object | None
    logs: object | None
    artifacts: tuple[ArtifactMetadata, ...]
    limitation_disclosure: LimitationDisclosure

    @property
    def run_id(self) -> UUID | str:
        return self.summary.run_id

    @property
    def run_manifest(self) -> object | None:
        return self.manifest

    @property
    def fingerprint(self) -> object | None:
        return self.environment_fingerprint

    @property
    def validation(self) -> object | None:
        return self.validation_report

    @property
    def log_entries(self) -> object | None:
        return self.logs

    @property
    def artifact_metadata(self) -> tuple[ArtifactMetadata, ...]:
        return self.artifacts


@dataclass(frozen=True, slots=True)
class TablePage:
    """A deterministic bounded ordinary table page."""

    rows: tuple[Mapping[str, object], ...]
    page: int
    page_size: int
    total: int | None = None
    columns: tuple[str, ...] = ()
    artifact_checksum: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.page, bool)
            or not isinstance(self.page, int)
            or self.page < 0
        ):
            raise ValueError("page must be a non-negative integer")
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")
        if not isinstance(self.rows, tuple) or any(
            not isinstance(row, Mapping) for row in self.rows
        ):
            raise TypeError("rows must be an immutable tuple of mappings")
        if len(self.rows) > self.page_size:
            raise ValueError("a table page cannot contain more than page_size rows")
        if self.total is not None and (
            isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total < 0
        ):
            raise ValueError("total must be a non-negative integer or None")
        if not isinstance(self.columns, tuple) or any(
            not isinstance(column, str) or not column for column in self.columns
        ):
            raise TypeError("columns must be an immutable tuple of names")
        if (
            self.artifact_checksum is not None
            and _SHA256.fullmatch(self.artifact_checksum) is None
        ):
            raise ValueError("artifact_checksum must be a lowercase SHA-256 digest")

    @property
    def items(self) -> tuple[Mapping[str, object], ...]:
        return self.rows

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def has_next(self) -> bool:
        if self.total is not None:
            return (self.page + 1) * self.page_size < self.total
        return len(self.rows) == self.page_size


class _InspectionFailure(Exception):
    def __init__(self, error: ActionableError) -> None:
        super().__init__(error.message)
        self.error = error


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


def _uuid(value: UUID | str) -> UUID | str:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return str(value)


def _error(
    operation: str,
    category: ErrorCategory,
    message: str,
    corrective_action: str,
    *,
    checksum: str | None = None,
    run_id: UUID | str | None = None,
    field_path: str | None = None,
) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=message,
        corrective_action=corrective_action,
        checksum=checksum,
        correlation_id=str(run_id) if run_id is not None else None,
        field_path=field_path,
    )


def _invoke(
    method: Callable[..., object],
    values: Mapping[str, object],
    positional: tuple[object, ...] = (),
) -> object:
    """Invoke a structural port using only parameters it declares."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(*positional)
    parameters = tuple(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return method(**dict(values))
    accepted: dict[str, object] = {}
    positional_only: list[object] = []
    position = 0
    for parameter in parameters:
        if (
            parameter.name == "self"
            or parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ):
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
    return method(**accepted)


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


def _unwrap(value: object, operation: str) -> object:
    if isinstance(value, Ok):
        return value.value
    if isinstance(value, Err):
        raise _InspectionFailure(value.errors[0])
    if value is None:
        raise _InspectionFailure(
            _error(
                operation,
                ErrorCategory.STORAGE_IO,
                "The inspection store returned no record.",
                "Repair or reconcile the local metadata index, then retry.",
            )
        )
    return value


def _redact(value: object, redactor: Redactor | object | None) -> object:
    if redactor is None:
        return value
    method = _method(
        redactor, ("redact_structured", "sanitize_metadata", "redact_metadata")
    )
    if method is None:
        return value
    try:
        return method(value)
    except Exception:
        # A redactor must not make inspection leak an unredacted value.  A
        # failed sanitizer therefore yields a safe marker rather than input.
        return "[REDACTED]"


def _safe_configuration(value: object, redactor: Redactor | object | None) -> object:
    projected: object = value
    with suppress(TypeError, ValueError):
        projected = cast(object, non_secret_config(cast(Any, value)))
    return _redact(projected, redactor)


def _rows(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, Mapping):
                raise TypeError("table scanner returned a non-mapping row")
            yield item
        return
    to_pylist = getattr(value, "to_pylist", None)
    if callable(to_pylist):
        result = to_pylist()
        if not isinstance(result, list):
            raise TypeError("record batch conversion did not return a list")
        for item in result:
            if not isinstance(item, Mapping):
                raise TypeError("record batch contains a non-mapping row")
            yield item
        return
    try:
        iterator = iter(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError("table scanner did not return an iterable") from error
    for batch in iterator:
        yield from _rows(batch)


def _as_datetime(value: object) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _artifact_record(value: object, *, link: object | None = None) -> ArtifactMetadata:
    checksum = _field(value, ("checksum", "sha256"), _field(link, "checksum"))
    if not isinstance(checksum, str):
        raise ValueError("artifact record has no checksum")
    kind = str(_field(value, ("artifact_kind", "kind", "object_kind"), "artifact"))
    role = _field(link, "role", _field(value, "role", None))
    scientific = _field(link, "scientific", _field(value, "scientific", None))
    row_count_value = _field(value, "row_count")
    byte_size_value = _field(value, ("byte_size", "size"), 0)
    declared_columns = _field(value, ("columns", "column_names"), ())
    columns = tuple(cast(Sequence[str], declared_columns)) if declared_columns else ()
    return ArtifactMetadata(
        checksum=checksum,
        artifact_kind=str(kind),
        relative_uri=str(_field(value, ("relative_uri", "uri", "path"), "")),
        media_type=str(_field(value, "media_type", "application/octet-stream")),
        byte_size=int(cast(int | str, byte_size_value)),
        row_count=(
            None if row_count_value is None else int(cast(int | str, row_count_value))
        ),
        schema_version=cast(str | None, _field(value, "schema_version")),
        availability=str(
            getattr(
                _field(value, "availability", "available"),
                "value",
                _field(value, "availability", "available"),
            )
        ),
        role=str(role) if role is not None else None,
        scientific=cast(bool | None, scientific),
        columns=columns,
    )


class InspectionService:
    """Inspect verified snapshots/runs and serve bounded ordinary table pages."""

    def __init__(
        self,
        snapshot_manager: InspectionSnapshotPort | None = None,
        metadata_store: InspectionMetadataPort | object | None = None,
        artifact_store: InspectionArtifactPort | object | None = None,
        parquet_store: InspectionScannerPort | object | None = None,
        experiment_tracker: object | None = None,
        *,
        metadata: object | None = None,
        artifacts: object | None = None,
        scanner: object | None = None,
        redactor: Redactor | object | None = None,
        configured_page_size: int = 100,
        page_size: int | None = None,
        run_manifest_store: object | None = None,
    ) -> None:
        if metadata_store is not None and metadata is not None:
            raise ValueError("supply either metadata_store or metadata")
        if artifact_store is not None and artifacts is not None:
            raise ValueError("supply either artifact_store or artifacts")
        if parquet_store is not None and scanner is not None:
            raise ValueError("supply either parquet_store or scanner")
        self.snapshot_manager = snapshot_manager
        self.metadata_store = metadata_store or metadata
        self.artifact_store = artifact_store or artifacts
        self.parquet_store = parquet_store or scanner
        self.experiment_tracker = experiment_tracker
        self.run_manifest_store = run_manifest_store
        self.redactor = redactor
        bound = configured_page_size if page_size is None else page_size
        if (
            isinstance(bound, bool)
            or not isinstance(bound, int)
            or not 1 <= bound <= 100
        ):
            raise ValueError("configured_page_size must be between 1 and 100")
        self.configured_page_size = bound

    def inspect_snapshot(self, snapshot_id: str) -> Result[object]:
        """Delegate to the snapshot manager's checksum-verifying inspection."""
        method = _method(self.snapshot_manager, ("inspect_snapshot", "inspect"))
        if method is None:
            return Err(
                (
                    _error(
                        "snapshot.inspect",
                        ErrorCategory.STORAGE_IO,
                        "Snapshot inspection is unavailable.",
                        "Configure a verified snapshot manager, then retry.",
                    ),
                )
            )
        try:
            return cast(
                Result[object],
                _invoke(method, {"snapshot_id": snapshot_id}, (snapshot_id,)),
            )
        except _InspectionFailure as failure:
            return Err((failure.error,))
        except Exception as error:
            return Err(
                (
                    ActionableError.from_unexpected_exception(
                        "snapshot.inspect", error, correlation_id=snapshot_id
                    ),
                )
            )

    inspect_snapshot_details = inspect_snapshot

    def inspect_run(self, run_id: UUID | str) -> Result[RunDetail]:
        """Return run provenance and all linked artifact metadata safely."""
        identifier = _uuid(run_id)
        try:
            getter = _method(self.metadata_store, ("get_run", "load_run"))
            if getter is None:
                raise _InspectionFailure(
                    _error(
                        "run.inspect",
                        ErrorCategory.STORAGE_IO,
                        "Run inspection is unavailable because no metadata index "
                        "is configured.",
                        "Configure the metadata index and retry.",
                    )
                )
            record = _unwrap(
                _invoke(
                    getter, {"run_id": identifier, "id": identifier}, (identifier,)
                ),
                "run.inspect",
            )
            summary = self._run_summary(record, run_id)
            manifest = self._read_run_value(
                record,
                ("manifest", "run_manifest"),
                ("get_run_manifest", "read_run_manifest", "read_manifest"),
                identifier,
            )
            configuration = _field(
                record,
                ("configuration", "resolved_configuration", "non_secret_configuration"),
            )
            if configuration is None:
                configuration = _field(
                    manifest,
                    (
                        "configuration",
                        "resolved_configuration",
                        "non_secret_configuration",
                    ),
                )
            fingerprint = _field(record, ("environment_fingerprint", "fingerprint"))
            if fingerprint is None:
                fingerprint = _field(
                    manifest, ("environment_fingerprint", "fingerprint")
                )
            validation = self._read_run_value(
                record,
                ("validation_report", "validation"),
                ("get_validation_report", "read_validation_report"),
                identifier,
            )
            logs = self._read_run_value(
                record,
                ("logs", "log", "log_entries"),
                ("get_run_logs", "list_run_logs", "read_logs"),
                identifier,
            )
            links = self._run_artifact_links(record, identifier)
            artifacts: list[ArtifactMetadata] = []
            for link in links:
                checksum = _field(link, "checksum")
                if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
                    raise _InspectionFailure(
                        _error(
                            "run.inspect",
                            ErrorCategory.INTEGRITY_CHECKSUM,
                            "The run references an invalid artifact checksum.",
                            "Repair the run manifest or select another run.",
                            checksum=str(checksum) if checksum is not None else None,
                            run_id=identifier,
                        )
                    )
                metadata = self._get_artifact_record(checksum)
                artifacts.append(_artifact_record(metadata, link=link))
            disclosure = _field(
                record,
                "limitation_disclosure",
                _field(
                    manifest, "limitation_disclosure", LimitationDisclosure.current()
                ),
            )
            if not isinstance(disclosure, LimitationDisclosure):
                disclosure = LimitationDisclosure.current()
            detail = RunDetail(
                summary=summary,
                run_record=_redact(record, self.redactor),
                manifest=_redact(self._decode_document(manifest), self.redactor),
                configuration=_safe_configuration(configuration, self.redactor)
                if configuration is not None
                else None,
                environment_fingerprint=_redact(fingerprint, self.redactor),
                validation_report=_redact(
                    self._decode_document(validation), self.redactor
                ),
                logs=_redact(self._decode_document(logs), self.redactor),
                artifacts=tuple(
                    sorted(artifacts, key=lambda item: (item.role or "", item.checksum))
                ),
                limitation_disclosure=disclosure,
            )
            return Ok(detail)
        except _InspectionFailure as failure:
            return Err((failure.error,))
        except (KeyError, FileNotFoundError):
            return Err(
                (
                    _error(
                        "run.inspect",
                        ErrorCategory.STORAGE_IO,
                        "The requested run record is missing.",
                        "Reconcile the metadata index or select another Run_ID.",
                        run_id=identifier,
                    ),
                )
            )
        except Exception as error:
            return Err(
                (
                    ActionableError.from_unexpected_exception(
                        "run.inspect", error, correlation_id=str(identifier)
                    ),
                )
            )

    inspect_run_details = inspect_run

    def inspect_artifact(self, checksum: str) -> Result[ArtifactMetadata]:
        try:
            record = self._get_artifact_record(checksum)
            metadata = _artifact_record(record)
            if metadata.availability != "available":
                raise _InspectionFailure(
                    _error(
                        "artifact.inspect",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact is unavailable or already marked invalid.",
                        "Restore the artifact or select another checksum.",
                        checksum=checksum,
                    )
                )
            return Ok(metadata)
        except _InspectionFailure as failure:
            return Err((failure.error,))
        except (KeyError, FileNotFoundError):
            return Err(
                (
                    _error(
                        "artifact.inspect",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The requested artifact record is missing.",
                        "Restore the artifact metadata or select another checksum.",
                        checksum=checksum if isinstance(checksum, str) else None,
                    ),
                )
            )
        except Exception:
            return Err(
                (
                    _error(
                        "artifact.inspect",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact metadata is malformed or could not be verified.",
                        "Restore the artifact metadata or select another checksum.",
                        checksum=checksum if isinstance(checksum, str) else None,
                    ),
                )
            )

    get_artifact = inspect_artifact

    def page_artifact(
        self,
        checksum: str,
        page: int = 0,
        page_size: int | None = None,
        columns: Sequence[str] | None = None,
        *,
        order_by: Sequence[str] | None = None,
    ) -> Result[TablePage]:
        """Return one projected page without materializing the complete artifact."""
        try:
            if isinstance(page, bool) or not isinstance(page, int) or page < 0:
                raise ValueError("page must be a non-negative integer")
            requested = self.configured_page_size if page_size is None else page_size
            if (
                isinstance(requested, bool)
                or not isinstance(requested, int)
                or requested < 1
            ):
                raise ValueError("page_size must be a positive integer")
            effective = min(requested, self.configured_page_size, 100)
            metadata = cast(
                ArtifactMetadata,
                _unwrap(self.inspect_artifact(checksum), "artifact.page"),
            )
            selected_columns = self._columns(metadata, columns)
            selected_order = tuple(order_by or selected_columns)
            if not selected_order:
                raise _InspectionFailure(
                    _error(
                        "artifact.page",
                        ErrorCategory.STORAGE_IO,
                        "No table projection was supplied for this artifact.",
                        "Select one or more table columns, then retry.",
                        checksum=checksum,
                    )
                )
            scanner = _method(self.parquet_store, ("scan", "scan_batches", "read_page"))
            if scanner is None:
                raise _InspectionFailure(
                    _error(
                        "artifact.page",
                        ErrorCategory.STORAGE_IO,
                        "Ordinary table paging is unavailable for this artifact.",
                        "Configure the projected Parquet scanner and retry.",
                        checksum=checksum,
                    )
                )
            reference = self._scanner_reference(metadata)
            offset = page * effective
            native = _supports_window(scanner)
            scanned = _invoke(
                scanner,
                {
                    "refs": (reference,),
                    "references": (reference,),
                    "ref": reference,
                    "artifact": reference,
                    "columns": selected_columns,
                    "predicate": None,
                    "offset": offset,
                    "limit": effective,
                    "order_by": selected_order,
                    "checksum": checksum,
                },
                (reference, selected_columns, None),
            )
            values: list[Mapping[str, object]] = []
            seen = 0
            for row in _rows(scanned):
                if not native and seen < offset:
                    seen += 1
                    continue
                if len(values) >= effective:
                    break
                values.append(dict(row))
                seen += 1
            total = metadata.row_count
            result = TablePage(
                rows=tuple(values),
                page=page,
                page_size=effective,
                total=total,
                columns=selected_columns,
                artifact_checksum=metadata.checksum,
            )
            return Ok(result)
        except _InspectionFailure as failure:
            return Err((failure.error,))
        except (KeyError, FileNotFoundError, OSError, ValueError, TypeError):
            self._mark_invalid(checksum)
            return Err(
                (
                    _error(
                        "artifact.page",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact could not be read as a verified table.",
                        "Restore the checksummed artifact or select another artifact.",
                        checksum=checksum if isinstance(checksum, str) else None,
                    ),
                )
            )
        except Exception:
            self._mark_invalid(checksum)
            return Err(
                (
                    _error(
                        "artifact.page",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact failed checksum verification while being "
                        "read as a table.",
                        "Restore the checksummed artifact or select another artifact.",
                        checksum=checksum if isinstance(checksum, str) else None,
                    ),
                )
            )

    page_table = page_artifact
    page_artifact_table = page_artifact

    def open_artifact(
        self,
        checksum: str | UUID,
        artifact_checksum: str | None = None,
        *,
        run_id: UUID | str | None = None,
    ) -> Result[VerifiedArtifact]:
        """Open a full artifact through a lazy verified stream handle.

        ``open_artifact(checksum)`` is the general form.  The compatibility
        form ``open_artifact(run_id, checksum)`` verifies the run association
        through an injected experiment tracker before returning the stream.
        """
        if artifact_checksum is not None:
            run_id = checksum
            checksum = artifact_checksum
        if not isinstance(checksum, str):
            return Err(
                (
                    _error(
                        "artifact.verify",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact checksum is invalid.",
                        "Select a lowercase SHA-256 artifact checksum.",
                        checksum=str(checksum),
                    ),
                )
            )
        identifier = _uuid(run_id) if run_id is not None else UUID(int=0)
        try:
            metadata = cast(
                ArtifactMetadata,
                _unwrap(self.inspect_artifact(checksum), "artifact.verify"),
            )
            if run_id is not None:
                self._assert_run_artifact(identifier, checksum)
            if self.experiment_tracker is not None and run_id is not None:
                method = _method(
                    self.experiment_tracker, ("open_verified_artifact", "open_artifact")
                )
                if method is not None:
                    opened = _unwrap(
                        _invoke(
                            method,
                            {"run_id": identifier, "checksum": checksum},
                            (identifier, checksum),
                        ),
                        "artifact.verify",
                    )
                    return Ok(
                        self._verified_handle(identifier, checksum, metadata, opened)
                    )
            method = _method(
                self.artifact_store,
                ("open_verified_artifact", "stream_artifact", "open_artifact"),
            )
            if method is None:
                raise _InspectionFailure(
                    _error(
                        "artifact.verify",
                        ErrorCategory.STORAGE_IO,
                        "Verified artifact streaming is unavailable.",
                        "Configure the content-addressed artifact store and retry.",
                        checksum=checksum,
                    )
                )
            reference = self._artifact_reference(metadata)
            opened = _invoke(
                method,
                {"reference": reference, "artifact": reference, "checksum": checksum},
                (reference,),
            )
            return Ok(self._verified_handle(identifier, checksum, metadata, opened))
        except _InspectionFailure as failure:
            return Err((failure.error,))
        except Exception:
            self._mark_invalid(checksum)
            return Err(
                (
                    _error(
                        "artifact.verify",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The artifact failed checksum verification or could not "
                        "be opened.",
                        "Restore the immutable artifact bytes or select another "
                        "verified artifact.",
                        checksum=checksum if isinstance(checksum, str) else None,
                        run_id=identifier,
                    ),
                )
            )

    open_verified_artifact = open_artifact
    download_artifact = open_artifact

    def _assert_run_artifact(self, identifier: UUID | str, checksum: str) -> None:
        getter = _method(self.metadata_store, ("get_run", "load_run"))
        if getter is None:
            return
        try:
            record = _unwrap(
                _invoke(
                    getter, {"run_id": identifier, "id": identifier}, (identifier,)
                ),
                "artifact.verify",
            )
            links = self._run_artifact_links(record, identifier)
        except _InspectionFailure:
            raise
        except Exception:
            raise _InspectionFailure(
                _error(
                    "artifact.verify",
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The run artifact association could not be verified.",
                    "Reconcile the run metadata or select another artifact.",
                    checksum=checksum,
                    run_id=identifier,
                )
            ) from None
        if links and not any(
            str(_field(link, "checksum", "")) == checksum for link in links
        ):
            raise _InspectionFailure(
                _error(
                    "artifact.verify",
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The artifact is not referenced by this run.",
                    "Select an artifact listed by the selected run or inspect it "
                    "without a run association.",
                    checksum=checksum,
                    run_id=identifier,
                )
            )

    def _run_summary(self, record: object, requested_id: UUID | str) -> RunSummary:
        raw_id = _field(record, ("run_id", "platform_run_id"), requested_id)
        identifier = _uuid(cast(UUID | str, raw_id))
        start = _field(record, "evaluation_start")
        end = _field(record, "evaluation_end")
        if (
            not isinstance(start, date)
            or isinstance(start, datetime)
            or not isinstance(end, date)
            or isinstance(end, datetime)
        ):
            raise _InspectionFailure(
                _error(
                    "run.inspect",
                    ErrorCategory.STORAGE_IO,
                    "The run record has an invalid evaluation range.",
                    "Repair the run metadata or select another run.",
                    run_id=identifier,
                )
            )
        universe_value = _field(record, "universe", ())
        universe = (
            tuple(
                str(item).strip().upper()
                for item in cast(Iterable[object], cast(Any, universe_value))
            )
            if not isinstance(universe_value, str)
            else (universe_value.strip().upper(),)
        )
        return RunSummary(
            run_id=identifier,
            snapshot_id=str(_field(record, "snapshot_id", "")),
            state=_field(record, "state", "unknown"),
            strategy_id=str(_field(record, ("strategy_id", "strategy_identifier"), "")),
            evaluation_start=start,
            evaluation_end=end,
            universe=universe,
            configuration_checksum=str(
                _field(record, ("configuration_checksum", "config_checksum"), "")
            ),
            environment_checksum=str(_field(record, ("environment_checksum",), "")),
            manifest_checksum=cast(str | None, _field(record, "manifest_checksum")),
            created_at=_as_datetime(_field(record, "created_at")),
            ended_at=_as_datetime(_field(record, "ended_at")),
        )

    def _read_run_value(
        self,
        record: object,
        fields: Sequence[str],
        methods: Sequence[str],
        identifier: UUID | str,
    ) -> object | None:
        value = _field(record, fields)
        if value is not None:
            return self._decode_document(value)
        for target in (self.run_manifest_store, self.metadata_store):
            method = _method(target, methods)
            if method is not None:
                try:
                    return self._decode_document(
                        _invoke(
                            method,
                            {"run_id": identifier, "id": identifier},
                            (identifier,),
                        )
                    )
                except (KeyError, FileNotFoundError):
                    return None
        return None

    def _run_artifact_links(
        self, record: object, identifier: UUID | str
    ) -> tuple[object, ...]:
        value = _field(record, ("artifacts", "artifact_references", "run_artifacts"))
        if value is None:
            method = _method(
                self.metadata_store,
                ("list_run_artifacts", "get_run_artifacts", "run_artifacts"),
            )
            if method is None:
                return ()
            value = _invoke(
                method, {"run_id": identifier, "id": identifier}, (identifier,)
            )
        if isinstance(value, Mapping):
            return tuple(value.values())
        if value is None:
            return ()
        return tuple(cast(Iterable[object], cast(Any, value)))

    def _get_artifact_record(self, checksum: str) -> object:
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            raise _InspectionFailure(
                _error(
                    "artifact.inspect",
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The artifact checksum is invalid.",
                    "Select a lowercase SHA-256 artifact checksum.",
                    checksum=str(checksum),
                )
            )
        getter = _method(self.metadata_store, ("get_artifact", "load_artifact"))
        if getter is None:
            raise _InspectionFailure(
                _error(
                    "artifact.inspect",
                    ErrorCategory.STORAGE_IO,
                    "Artifact metadata is unavailable.",
                    "Configure the metadata index and retry.",
                    checksum=checksum,
                )
            )
        try:
            value = _invoke(getter, {"checksum": checksum}, (checksum,))
        except (KeyError, FileNotFoundError):
            raise _InspectionFailure(
                _error(
                    "artifact.inspect",
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The requested artifact record is missing.",
                    "Restore the artifact metadata or select another checksum.",
                    checksum=checksum,
                )
            ) from None
        try:
            return _unwrap(value, "artifact.inspect")
        except _InspectionFailure as failure:
            if failure.error.category is ErrorCategory.STORAGE_IO:
                raise _InspectionFailure(
                    _error(
                        "artifact.inspect",
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The requested artifact record is missing.",
                        "Restore the artifact metadata or select another checksum.",
                        checksum=checksum,
                    )
                ) from None
            raise

    @staticmethod
    def _decode_document(value: object) -> object:
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                return json.loads(bytes(value).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return "[UNREADABLE_DOCUMENT]"
        return value

    @staticmethod
    def _columns(
        metadata: ArtifactMetadata, columns: Sequence[str] | None
    ) -> tuple[str, ...]:
        if columns is not None:
            if isinstance(columns, (str, bytes)) or not columns:
                raise ValueError("columns must be a non-empty sequence")
            result = tuple(columns)
        else:
            declared = _field(metadata, ("columns", "column_names"), ())
            result = (
                tuple(cast(Sequence[str], declared))
                if declared
                else ("symbol", "session")
            )
        if any(not isinstance(column, str) or not column for column in result) or len(
            set(result)
        ) != len(result):
            raise ValueError("columns must contain distinct non-empty names")
        return result

    @staticmethod
    def _scanner_reference(metadata: ArtifactMetadata) -> ContentAddressedObjectRef:
        return ContentAddressedObjectRef(
            object_kind=ObjectKind.ARTIFACT,
            checksum=metadata.checksum,
            relative_uri=metadata.relative_uri,
            schema_version=metadata.schema_version or "artifact_v1",
            row_count=metadata.row_count or 0,
            byte_size=metadata.byte_size,
            media_type=metadata.media_type,
        )

    @staticmethod
    def _artifact_reference(metadata: ArtifactMetadata) -> ArtifactReference:
        metadata_document = {
            "artifact_kind": metadata.artifact_kind,
            "checksum": metadata.checksum,
            "byte_size": metadata.byte_size,
            "media_type": metadata.media_type,
            "schema_version": metadata.schema_version,
            "row_count": metadata.row_count,
        }
        return ArtifactReference(
            checksum=metadata.checksum,
            byte_size=metadata.byte_size,
            relative_uri=metadata.relative_uri,
            metadata_checksum=sha256_bytes(canonical_json(metadata_document)),
        )

    @staticmethod
    def _verified_handle(
        identifier: UUID | str,
        checksum: str,
        metadata: ArtifactMetadata,
        opened: object,
    ) -> VerifiedArtifact:
        actual_id = identifier if isinstance(identifier, UUID) else UUID(int=0)
        stream_method = getattr(opened, "stream", None)
        factory: Callable[[], Iterable[bytes]]
        if callable(stream_method):
            factory = cast(Callable[[], Iterable[bytes]], stream_method)
        elif callable(opened):
            factory = cast(Callable[[], Iterable[bytes]], opened)
        else:

            def factory() -> Iterable[bytes]:
                return cast(Iterable[bytes], opened)

        return VerifiedArtifact(
            run_id=actual_id,
            checksum=checksum,
            relative_uri=metadata.relative_uri,
            byte_size=metadata.byte_size,
            media_type=metadata.media_type,
            availability=metadata.availability,
            _stream_factory=factory,
        )

    def _mark_invalid(self, checksum: object) -> None:
        if not isinstance(checksum, str):
            return
        method = _method(
            self.metadata_store, ("set_artifact_availability", "mark_artifact_invalid")
        )
        if method is not None:
            with suppress(Exception):
                _invoke(
                    method,
                    {
                        "checksum": checksum,
                        "availability": "invalid",
                        "state": "invalid",
                    },
                    (checksum, "invalid"),
                )


def _supports_window(scanner: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(scanner).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name in {"offset", "limit"} for parameter in parameters
    ) or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


ArtifactInspectionService = InspectionService
InspectionApplicationService = InspectionService


__all__ = [
    "ArtifactMetadata",
    "ArtifactInspectionService",
    "InspectionApplicationService",
    "InspectionArtifactPort",
    "InspectionMetadataPort",
    "InspectionScannerPort",
    "InspectionService",
    "InspectionSnapshotPort",
    "RunDetail",
    "RunSummary",
    "TablePage",
]
