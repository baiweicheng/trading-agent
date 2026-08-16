"""Application-level assembly of immutable snapshot manifests.

The snapshot domain objects enforce the byte-level representation of a
manifest.  This module owns the use-case boundary that gathers the already
verified scientific facts produced by ingestion and keeps operational facts
separate from them.  It deliberately does not publish files or open a
snapshot; those concerns belong to the later snapshot/storage services.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from ..domain.canonical import canonical_json, sha256_bytes
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotLineage,
    SnapshotManifest,
    SnapshotSchemaVersions,
    VerifiedSnapshotHandle,
)
from ..domain.market import (
    DateRange,
    ProviderRequestMetadata,
    ValidationReport,
    ValidationSummary,
    normalize_symbol,
)
from ..domain.validation import ValidationOutput

ValidationFacts: TypeAlias = ValidationSummary | ValidationReport | ValidationOutput


def _as_tuple(value: Iterable[object], *, field_name: str) -> tuple[object, ...]:
    """Materialize an input collection without accepting a scalar by accident."""

    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise TypeError(f"{field_name} must be an iterable of values, not a scalar")
    return tuple(value)


def _normalize_symbols(
    values: Sequence[str] | Iterable[str], *, field_name: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise TypeError(f"{field_name} must be an iterable of symbols, not a scalar")
    normalized = tuple(normalize_symbol(value) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one symbol")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain distinct normalized symbols")
    return normalized


def _validation_summary(value: ValidationFacts) -> ValidationSummary:
    """Extract the deterministic compact summary from validation output."""

    if isinstance(value, ValidationSummary):
        return value
    if isinstance(value, ValidationReport):
        return value.summary
    if isinstance(value, ValidationOutput):
        return value.report.summary
    raise TypeError(
        "validation must be a ValidationSummary, ValidationReport, or ValidationOutput"
    )


def _validation_report(value: ValidationFacts) -> ValidationReport | None:
    if isinstance(value, ValidationReport):
        return value
    if isinstance(value, ValidationOutput):
        return value.report
    return None


def _assemble_object_references(
    values: Iterable[ContentAddressedObjectRef],
    *,
    validation_report_checksum: str,
) -> tuple[ContentAddressedObjectRef, ...]:
    """Validate the one-reference-per-object rule before domain construction.

    ``SnapshotContentIdentity`` already sorts refs and rejects repeated
    checksums.  The application boundary additionally rejects repeated logical
    URIs, which catches two different checksums being supplied for the same
    partition path.  The validation report is a separate manifest reference;
    representing it again as a validation object would make the same artifact
    appear twice, so that form is rejected as well.
    """

    references = _as_tuple(values, field_name="objects")
    if any(not isinstance(item, ContentAddressedObjectRef) for item in references):
        raise TypeError("objects must contain ContentAddressedObjectRef values")

    typed_references = tuple(
        item for item in references if isinstance(item, ContentAddressedObjectRef)
    )
    uri_owners: dict[str, str] = {}
    checksum_owners: dict[str, str] = {}
    for reference in typed_references:
        uri = reference.relative_uri
        checksum = reference.checksum
        if uri in uri_owners:
            raise ValueError(
                f"objects must reference each logical partition URI exactly once: {uri}"
            )
        if checksum in checksum_owners:
            raise ValueError(
                f"objects must reference each content checksum exactly once: {checksum}"
            )
        if (
            reference.object_kind is ObjectKind.VALIDATION
            and checksum == validation_report_checksum
        ):
            raise ValueError(
                "validation report checksum must be referenced exactly once; "
                "do not repeat it as a validation object"
            )
        uri_owners[uri] = checksum
        checksum_owners[checksum] = uri

    return tuple(sorted(typed_references, key=ContentAddressedObjectRef.sort_key))


class SnapshotManifestAssembler:
    """Assemble a content-derived :class:`SnapshotManifest`.

    All arguments describing scientific content are explicit.  ``created_at``,
    request metadata, detection times, job/path notes, and lineage are accepted
    only as operational metadata and never enter ``SnapshotContentIdentity``.
    The resulting ``SnapshotManifest.snapshot_id`` is therefore reusable for
    equivalent content produced on another local root or from another parent.
    """

    @classmethod
    def assemble(
        cls,
        *,
        provider: str,
        requested_range: DateRange,
        configured_universe: Sequence[str] | Iterable[str],
        benchmark_symbol: str,
        calendar: CalendarIdentity,
        configuration_checksum: str,
        objects: Iterable[ContentAddressedObjectRef] | None = None,
        object_references: Iterable[ContentAddressedObjectRef] | None = None,
        validation_report_checksum: str | None = None,
        validation: ValidationFacts | None = None,
        validation_report: ValidationReport | None = None,
        validation_summary: ValidationSummary | None = None,
        limitation_disclosure: LimitationDisclosure | None = None,
        covered_range: DateRange | None = None,
        schema_versions: SnapshotSchemaVersions | None = None,
        created_at: datetime | None = None,
        operational_metadata: OperationalMetadata | None = None,
        provider_requests: Sequence[ProviderRequestMetadata] | None = None,
        detection_times: Sequence[datetime] | None = None,
        job_id: str | None = None,
        local_manifest_path: str | None = None,
        notes: Mapping[str, object] | None = None,
        lineage: SnapshotLineage | None = None,
        parent_snapshot_id: str | None = None,
        operation_id: str | None = None,
    ) -> SnapshotManifest:
        """Build one immutable manifest from verified snapshot facts.

        ``validation`` is the preferred input when the caller has the complete
        validation service result.  The explicit ``validation_report`` and
        ``validation_summary`` forms keep the method useful for storage
        finalizers that stream the report and retain only its compact summary.
        A report's canonical content checksum is used only when a physical
        validation-artifact checksum was not supplied; callers that wrote a
        Parquet validation artifact should pass that artifact's SHA-256.
        """

        del cls

        if objects is not None and object_references is not None:
            raise ValueError("supply either objects or object_references, not both")
        references_input = objects if objects is not None else object_references
        if references_input is None:
            references_input = ()

        supplied_validation: ValidationFacts | None = validation
        if supplied_validation is not None and (
            validation_report is not None or validation_summary is not None
        ):
            raise ValueError(
                "supply validation, or validation_report/validation_summary, not both"
            )
        if validation_report is not None and validation_summary is not None:
            if validation_report.summary != validation_summary:
                raise ValueError(
                    "validation_report and validation_summary contain different facts"
                )
            supplied_validation = validation_report
        elif validation_report is not None:
            supplied_validation = validation_report
        elif validation_summary is not None:
            supplied_validation = validation_summary

        if supplied_validation is None:
            raise TypeError(
                "one of validation, validation_report, or validation_summary is required"
            )

        summary = _validation_summary(supplied_validation)
        report = _validation_report(supplied_validation)
        report_checksum = validation_report_checksum
        if report_checksum is None and report is not None:
            report_checksum = report.content_checksum
        if report_checksum is None:
            raise TypeError(
                "validation_report_checksum is required when only a ValidationSummary is supplied"
            )
        if not isinstance(report_checksum, str):
            raise TypeError("validation_report_checksum must be a SHA-256 string")

        refs = _assemble_object_references(
            references_input,
            validation_report_checksum=report_checksum,
        )
        disclosure = limitation_disclosure or LimitationDisclosure.current()
        versions = schema_versions or SnapshotSchemaVersions()
        universe = _normalize_symbols(
            configured_universe,
            field_name="configured_universe",
        )

        identity = SnapshotContentIdentity(
            provider=provider,
            requested_range=requested_range,
            covered_range=covered_range,
            configured_universe=universe,
            benchmark_symbol=normalize_symbol(benchmark_symbol),
            calendar=calendar,
            configuration_checksum=configuration_checksum,
            objects=refs,
            validation_report_checksum=report_checksum,
            validation_summary=summary,
            limitation_disclosure=disclosure,
            schema_versions=versions,
        )

        if operational_metadata is not None:
            if created_at is not None:
                raise ValueError(
                    "created_at must not be supplied with operational_metadata"
                )
            if any(
                value is not None
                for value in (
                    provider_requests,
                    detection_times,
                    job_id,
                    local_manifest_path,
                    notes,
                )
            ):
                raise ValueError(
                    "operational_metadata cannot be combined with its component fields"
                )
            metadata = operational_metadata
        else:
            if created_at is None:
                raise TypeError(
                    "created_at is required when operational_metadata is not supplied"
                )
            request_metadata = tuple(provider_requests or ())
            detection_metadata = tuple(detection_times or ())
            metadata = OperationalMetadata(
                created_at=created_at,
                provider_requests=request_metadata,
                detection_times=detection_metadata,
                job_id=job_id,
                local_manifest_path=local_manifest_path,
                notes={} if notes is None else notes,
            )

        if lineage is not None and any(
            value is not None for value in (parent_snapshot_id, operation_id)
        ):
            raise ValueError(
                "lineage cannot be combined with parent_snapshot_id or operation_id"
            )
        manifest_lineage = lineage or SnapshotLineage(
            parent_snapshot_id=parent_snapshot_id,
            operation_id=operation_id,
        )
        return SnapshotManifest(
            content_identity=identity,
            operational_metadata=metadata,
            lineage=manifest_lineage,
        )

    @classmethod
    def build(cls, **kwargs: object) -> SnapshotManifest:
        """Alias for :meth:`assemble` for callers using builder terminology."""

        return cls.assemble(**kwargs)  # type: ignore[arg-type]


SnapshotManifestBuilder = SnapshotManifestAssembler


def assemble_snapshot_manifest(**kwargs: object) -> SnapshotManifest:
    """Functional facade for :class:`SnapshotManifestAssembler`."""

    return SnapshotManifestAssembler.assemble(**kwargs)  # type: ignore[arg-type]


build_snapshot_manifest = assemble_snapshot_manifest


__all__ = [
    "SnapshotManifestAssembler",
    "SnapshotManifestBuilder",
    "ValidationFacts",
    "assemble_snapshot_manifest",
    "build_snapshot_manifest",
]


_SNAPSHOT_ID_PATTERN = re.compile(r"^snap_[0-9a-f]{64}$")


class SnapshotByteStore(Protocol):
    """Read-only byte boundary for published manifests and CAS objects."""

    def read_manifest(self, snapshot_id: str, relative_uri: str | None = None) -> bytes:
        """Read the complete published manifest for one snapshot."""

    def read_object(self, relative_uri: str) -> bytes:
        """Read one object by its manifest-relative URI."""


class SnapshotIndex(Protocol):
    """The minimal metadata-index surface required by snapshot inspection."""

    def get_snapshot(self, snapshot_id: str) -> object:
        """Return an indexed snapshot or raise when it is not indexed."""

    def list_snapshots(
        self,
        *,
        provider: str | None = None,
        availability: object | None = None,
        page: int = 0,
        page_size: int = 100,
    ) -> Sequence[object]:
        """Return a deterministic, bounded snapshot page."""

    def list_snapshot_objects(self, snapshot_id: str) -> Sequence[object]:
        """Return indexed object references for one snapshot."""


class SnapshotClock(Protocol):
    """UTC clock seam used when creating an immutable verification handle."""

    def utc_now(self) -> datetime:
        """Return an aware UTC timestamp."""


class SystemSnapshotClock:
    """Production clock for the operational ``verified_at`` field."""

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


class _SnapshotFailure(Exception):
    """Internal safe failure carrying no raw storage exception to callers."""

    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        corrective_action: str,
        *,
        field_path: str | None = None,
        checksum: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.corrective_action = corrective_action
        self.field_path = field_path
        self.checksum = checksum


class LocalPublishedSnapshotStore:
    """Read-only view of the published snapshot and content-addressed roots.

    This adapter intentionally has no write methods.  It resolves manifests only
    below ``snapshots/<snapshot-id>/manifest.json`` and resolves object URIs below
    the configured root, rejecting escaping paths and symlinks that leave the
    store.  CAS validation artifacts are discoverable by their checksum when a
    manifest keeps that artifact checksum separate from its partition references.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def read_manifest(self, snapshot_id: str, relative_uri: str | None = None) -> bytes:
        identifier = _require_snapshot_id(snapshot_id)
        expected = f"snapshots/{identifier}/manifest.json"
        uri = expected if relative_uri is None else _safe_snapshot_uri(relative_uri)
        if uri != expected:
            raise FileNotFoundError(
                "manifest is not at the published snapshot location"
            )
        return self._read_relative(uri)

    def read_object(self, relative_uri: str) -> bytes:
        return self._read_relative(relative_uri)

    def read_by_checksum(self, checksum: str) -> bytes:
        """Read a checksum-addressed validation artifact from the local CAS."""

        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        candidates = (
            self.root / "objects" / "sha256" / checksum[:2] / checksum,
            self.root / "artifacts" / "sha256" / checksum[:2] / checksum,
            self.root / "objects" / "validation" / f"sha256={checksum}.parquet",
            self.root / "objects" / "validation" / checksum,
        )
        found = False
        for candidate in candidates:
            if not candidate.exists():
                continue
            found = True
            data = self._read_candidate(candidate)
            if sha256_bytes(data) != checksum:
                raise ValueError("checksum-addressed validation bytes are corrupt")
            return data
        if found:  # pragma: no cover - every found candidate returns or raises above.
            raise ValueError("validation artifact could not be verified")
        raise FileNotFoundError("validation artifact is not published")

    def list_published_manifest_ids(self) -> tuple[str, ...]:
        """List only snapshot directories containing the exact publication file."""

        snapshots_root = self.root / "snapshots"
        if not snapshots_root.is_dir():
            return ()
        identifiers: list[str] = []
        for candidate in snapshots_root.iterdir():
            if not candidate.is_dir() or not _SNAPSHOT_ID_PATTERN.fullmatch(
                candidate.name
            ):
                continue
            manifest = candidate / "manifest.json"
            if manifest.is_file():
                identifiers.append(candidate.name)
        return tuple(sorted(identifiers))

    def _read_relative(self, relative_uri: str) -> bytes:
        return self._read_candidate(self._path_for(relative_uri))

    def _path_for(self, relative_uri: str) -> Path:
        uri = _safe_snapshot_uri(relative_uri)
        candidate = self.root.joinpath(*PurePosixPath(uri).parts)
        return self._resolve_inside(candidate)

    def _read_candidate(self, candidate: Path) -> bytes:
        resolved = self._resolve_inside(candidate)
        if not resolved.is_file():
            raise FileNotFoundError("published snapshot object is not a regular file")
        return resolved.read_bytes()

    def _resolve_inside(self, candidate: Path) -> Path:
        try:
            resolved_root = self.root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise
        if not resolved.is_relative_to(resolved_root):
            raise PermissionError("published snapshot path escapes its storage root")
        return resolved


@dataclass(frozen=True, slots=True)
class SnapshotQuery:
    """Bounded discovery filters for immutable snapshot summaries."""

    provider: str | None = None
    availability: str | None = None
    page: int = 0
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.provider is not None:
            if not isinstance(self.provider, str) or not self.provider.strip():
                raise ValueError("provider must be a non-blank string or None")
            object.__setattr__(self, "provider", " ".join(self.provider.split()))
        if self.availability is not None:
            normalized = _availability_text(self.availability)
            if normalized not in {"available", "unavailable", "invalid"}:
                raise ValueError(
                    "availability must be available, unavailable, or invalid"
                )
            object.__setattr__(self, "availability", normalized)
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


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """Redacted, bounded discovery projection for one Data_Snapshot."""

    snapshot_id: str
    provider: str
    requested_range: DateRange
    covered_range: DateRange | None
    configured_universe: tuple[str, ...]
    benchmark_symbol: str
    comparison_ready: bool
    availability: str
    created_at: datetime
    manifest_checksum: str
    content_identity_checksum: str
    parent_snapshot_id: str | None
    limitation_disclosure: LimitationDisclosure
    validation_summary: ValidationSummary | None = None
    integrity_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _require_snapshot_id(self.snapshot_id))
        if not isinstance(self.requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange")
        if self.covered_range is not None and not isinstance(
            self.covered_range, DateRange
        ):
            raise TypeError("covered_range must be a DateRange or None")
        if (
            not isinstance(self.configured_universe, tuple)
            or not self.configured_universe
        ):
            raise TypeError("configured_universe must be a non-empty tuple")
        normalized = tuple(
            normalize_symbol(symbol) for symbol in self.configured_universe
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("configured_universe must contain distinct symbols")
        object.__setattr__(self, "configured_universe", normalized)
        object.__setattr__(
            self, "benchmark_symbol", normalize_symbol(self.benchmark_symbol)
        )
        object.__setattr__(
            self, "provider", _required_display_text("provider", self.provider)
        )
        object.__setattr__(self, "availability", _availability_text(self.availability))
        object.__setattr__(
            self, "created_at", _require_aware_datetime("created_at", self.created_at)
        )
        object.__setattr__(
            self,
            "manifest_checksum",
            _require_digest("manifest_checksum", self.manifest_checksum),
        )
        object.__setattr__(
            self,
            "content_identity_checksum",
            _require_digest(
                "content_identity_checksum", self.content_identity_checksum
            ),
        )
        if self.parent_snapshot_id is not None:
            object.__setattr__(
                self,
                "parent_snapshot_id",
                _require_snapshot_id(self.parent_snapshot_id),
            )
        if not isinstance(self.comparison_ready, bool):
            raise TypeError("comparison_ready must be a bool")
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be a LimitationDisclosure")
        if self.validation_summary is not None and not isinstance(
            self.validation_summary, ValidationSummary
        ):
            raise TypeError("validation_summary must be a ValidationSummary or None")
        if self.integrity_error is not None:
            object.__setattr__(
                self,
                "integrity_error",
                _required_display_text("integrity_error", self.integrity_error),
            )

    @property
    def failed_symbols(self) -> tuple[str, ...]:
        return self.validation_summary.failed_symbols if self.validation_summary else ()

    @property
    def stale_symbols(self) -> tuple[str, ...]:
        return self.validation_summary.stale_symbols if self.validation_summary else ()

    @property
    def gap_count(self) -> int:
        return self.validation_summary.gap_count if self.validation_summary else 0


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    """Scientific and operational provenance exposed by snapshot inspection."""

    provider: str
    requested_range: DateRange
    covered_range: DateRange | None
    configured_universe: tuple[str, ...]
    benchmark_symbol: str
    calendar: CalendarIdentity
    schema_versions: SnapshotSchemaVersions
    configuration_checksum: str
    object_references: tuple[ContentAddressedObjectRef, ...]
    validation_report_checksum: str
    created_at: datetime
    provider_requests: tuple[ProviderRequestMetadata, ...]
    parent_snapshot_id: str | None
    operation_id: str | None


@dataclass(frozen=True, slots=True)
class SnapshotReadiness:
    """Explicit availability and benchmark-comparison readiness facts."""

    available: bool
    comparison_ready: bool
    failed_symbols: tuple[str, ...] = ()
    stale_symbols: tuple[str, ...] = ()
    gap_count: int = 0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("available", "comparison_ready"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in ("failed_symbols", "stale_symbols", "reasons"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) for item in value
            ):
                raise TypeError(f"{field_name} must be an immutable tuple of strings")
        if (
            isinstance(self.gap_count, bool)
            or not isinstance(self.gap_count, int)
            or self.gap_count < 0
        ):
            raise ValueError("gap_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SnapshotDetail:
    """Verified snapshot inspection DTO with provenance and readiness."""

    manifest: SnapshotManifest
    handle: VerifiedSnapshotHandle
    summary: SnapshotSummary
    provenance: SnapshotProvenance
    validation_summary: ValidationSummary
    readiness: SnapshotReadiness

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def limitation_disclosure(self) -> LimitationDisclosure:
        return self.manifest.limitation_disclosure

    @property
    def comparison_ready(self) -> bool:
        return self.readiness.comparison_ready

    @property
    def validation(self) -> ValidationSummary:
        return self.validation_summary


T_Snapshot = TypeVar("T_Snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotPage(Generic[T_Snapshot]):
    """Immutable bounded page returned by snapshot discovery."""

    items: tuple[T_Snapshot, ...]
    page: int
    page_size: int
    total: int | None = None
    errors: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("items must be an immutable tuple")
        SnapshotQuery(page=self.page, page_size=self.page_size)
        if self.total is not None and (isinstance(self.total, bool) or self.total < 0):
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

    def __iter__(self) -> Iterator[T_Snapshot]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


# The aliases make the application boundary read naturally for callers that use
# either the design terminology (SnapshotManager) or service terminology.
SnapshotListPage = SnapshotPage[SnapshotSummary]


def _required_display_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _require_digest(name: str, value: str) -> str:
    digest = _required_display_text(name, value)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_snapshot_id(value: str) -> str:
    identifier = _required_display_text("snapshot_id", value)
    if _SNAPSHOT_ID_PATTERN.fullmatch(identifier) is None:
        raise ValueError("snapshot_id must be a content-derived snapshot ID")
    return identifier


def _availability_text(value: object) -> str:
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str):
        raise TypeError("availability must be a string or string enum")
    normalized = normalized.strip().lower()
    if normalized not in {"available", "unavailable", "invalid"}:
        raise ValueError("unsupported snapshot availability")
    return normalized


def _require_aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _safe_snapshot_uri(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("snapshot URI must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("snapshot URI must not escape its storage root")
    return path.as_posix()


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"manifest field {field_name} must be a mapping")
    return value


def _exact_mapping(
    value: object,
    field_name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    result = _mapping(value, field_name)
    keys = set(result)
    allowed = required | (optional or set())
    if keys != allowed or not required.issubset(keys):
        raise ValueError(f"manifest field {field_name} has an invalid schema")
    return result


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"manifest field {field_name} must be text")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"manifest field {field_name} must be a list of text")
    return tuple(value)


def _date_value(value: object, field_name: str) -> date:
    try:
        parsed = date.fromisoformat(_string(value, field_name))
    except (TypeError, ValueError) as error:
        raise ValueError(f"manifest field {field_name} must be an ISO date") from error
    return parsed


def _datetime_value(value: object, field_name: str) -> datetime:
    text = _string(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"manifest field {field_name} must be an ISO timestamp"
        ) from error
    return _require_aware_datetime(field_name, parsed)


def _date_range(value: object, field_name: str) -> DateRange:
    mapping = _exact_mapping(value, field_name, {"start", "end"})
    return DateRange(
        _date_value(mapping["start"], f"{field_name}.start"),
        _date_value(mapping["end"], f"{field_name}.end"),
    )


def _optional_date_range(value: object, field_name: str) -> DateRange | None:
    return None if value is None else _date_range(value, field_name)


def _parse_validation_summary(value: object) -> ValidationSummary:
    mapping = _exact_mapping(
        value,
        "content_identity.validation_summary",
        {
            "accepted_row_count",
            "quarantined_row_count",
            "collapsed_duplicate_count",
            "gap_count",
            "failed_symbols",
            "retained_parent_coverage_symbols",
            "stale_symbols",
            "covered_range",
            "comparison_ready",
        },
    )
    return ValidationSummary(
        accepted_row_count=mapping["accepted_row_count"],
        quarantined_row_count=mapping["quarantined_row_count"],
        collapsed_duplicate_count=mapping["collapsed_duplicate_count"],
        gap_count=mapping["gap_count"],
        failed_symbols=_string_tuple(mapping["failed_symbols"], "failed_symbols"),
        retained_parent_coverage_symbols=_string_tuple(
            mapping["retained_parent_coverage_symbols"],
            "retained_parent_coverage_symbols",
        ),
        stale_symbols=_string_tuple(mapping["stale_symbols"], "stale_symbols"),
        covered_range=_optional_date_range(mapping["covered_range"], "covered_range"),
        comparison_ready=mapping["comparison_ready"],
    )


def _parse_object_reference(value: object) -> ContentAddressedObjectRef:
    mapping = _exact_mapping(
        value,
        "content_identity.objects[]",
        {
            "object_kind",
            "checksum",
            "relative_uri",
            "schema_version",
            "row_count",
            "byte_size",
            "symbol",
            "session_year",
            "media_type",
        },
    )
    return ContentAddressedObjectRef(
        object_kind=ObjectKind(mapping["object_kind"]),
        checksum=mapping["checksum"],
        relative_uri=mapping["relative_uri"],
        schema_version=mapping["schema_version"],
        row_count=mapping["row_count"],
        byte_size=mapping["byte_size"],
        symbol=mapping["symbol"],
        session_year=mapping["session_year"],
        media_type=mapping["media_type"],
    )


def _parse_disclosure(
    document: Mapping[str, Any], content: Mapping[str, Any]
) -> LimitationDisclosure:
    identity = _exact_mapping(
        content["limitation_disclosure"],
        "content_identity.limitation_disclosure",
        {"version", "text_checksum"},
    )
    full = _exact_mapping(
        document["limitation_disclosure"],
        "limitation_disclosure",
        {"version", "lines"},
    )
    version = _string(identity["version"], "limitation_disclosure.version")
    if version != _string(full["version"], "limitation_disclosure.version"):
        raise ValueError("manifest limitation disclosure versions do not match")
    lines = _string_tuple(full["lines"], "limitation_disclosure.lines")
    disclosure = LimitationDisclosure(version=version)
    if lines != disclosure.lines():
        raise ValueError("manifest limitation disclosure text is not supported")
    expected_checksum = sha256_bytes(
        canonical_json({"version": version, "lines": list(lines)})
    )
    if expected_checksum != _string(
        identity["text_checksum"], "limitation_disclosure.text_checksum"
    ):
        raise ValueError("manifest limitation disclosure checksum does not match")
    return disclosure


def _parse_manifest_document(document: Mapping[str, Any]) -> SnapshotManifest:
    top = _exact_mapping(
        document,
        "manifest",
        {
            "snapshot_id",
            "content_identity",
            "operational_metadata",
            "lineage",
            "limitation_disclosure",
        },
    )
    content = _exact_mapping(
        top["content_identity"],
        "content_identity",
        {
            "schema_versions",
            "provider",
            "requested_range",
            "covered_range",
            "configured_universe",
            "benchmark_symbol",
            "calendar",
            "configuration_checksum",
            "objects",
            "validation_report_checksum",
            "validation_summary",
            "failed_symbols",
            "retained_parent_coverage_symbols",
            "limitation_disclosure",
        },
    )
    versions_mapping = _exact_mapping(
        content["schema_versions"],
        "content_identity.schema_versions",
        {
            "manifest_schema_version",
            "raw_schema_version",
            "normalized_schema_version",
            "quarantine_schema_version",
            "validation_report_schema_version",
            "corporate_action_policy_version",
        },
    )
    versions = SnapshotSchemaVersions(**versions_mapping)
    calendar_mapping = _exact_mapping(
        content["calendar"],
        "content_identity.calendar",
        {"name", "version", "schedule_checksum"},
    )
    calendar = CalendarIdentity(**calendar_mapping)
    object_values = content["objects"]
    if not isinstance(object_values, list):
        raise ValueError("content_identity.objects must be a list")
    objects = tuple(_parse_object_reference(value) for value in object_values)
    disclosure = _parse_disclosure(top, content)
    identity = SnapshotContentIdentity(
        provider=_string(content["provider"], "content_identity.provider"),
        requested_range=_date_range(content["requested_range"], "requested_range"),
        covered_range=_optional_date_range(content["covered_range"], "covered_range"),
        configured_universe=_string_tuple(
            content["configured_universe"], "configured_universe"
        ),
        benchmark_symbol=_string(content["benchmark_symbol"], "benchmark_symbol"),
        calendar=calendar,
        configuration_checksum=_string(
            content["configuration_checksum"], "configuration_checksum"
        ),
        objects=objects,
        validation_report_checksum=_string(
            content["validation_report_checksum"], "validation_report_checksum"
        ),
        validation_summary=_parse_validation_summary(content["validation_summary"]),
        limitation_disclosure=disclosure,
        schema_versions=versions,
    )
    operational = _exact_mapping(
        top["operational_metadata"],
        "operational_metadata",
        {
            "created_at",
            "provider_requests",
            "detection_times",
            "job_id",
            "local_manifest_path",
            "notes",
        },
    )
    requests_value = operational["provider_requests"]
    if not isinstance(requests_value, list):
        raise ValueError("operational_metadata.provider_requests must be a list")
    requests: list[ProviderRequestMetadata] = []
    for item in requests_value:
        request = _exact_mapping(
            item,
            "operational_metadata.provider_requests[]",
            {
                "request_content_key",
                "retrieved_at",
                "response_status",
                "request_id",
                "retrieval_started_at",
            },
        )
        requests.append(
            ProviderRequestMetadata(
                request_content_key=_string(
                    request["request_content_key"], "request_content_key"
                ),
                retrieved_at=_datetime_value(request["retrieved_at"], "retrieved_at"),
                response_status=_string(request["response_status"], "response_status"),
                request_id=(
                    None
                    if request["request_id"] is None
                    else _string(request["request_id"], "request_id")
                ),
                retrieval_started_at=(
                    None
                    if request["retrieval_started_at"] is None
                    else _datetime_value(
                        request["retrieval_started_at"], "retrieval_started_at"
                    )
                ),
            )
        )
    detection_value = operational["detection_times"]
    if not isinstance(detection_value, list):
        raise ValueError("operational_metadata.detection_times must be a list")
    notes = _mapping(operational["notes"], "operational_metadata.notes")
    metadata = OperationalMetadata(
        created_at=_datetime_value(operational["created_at"], "created_at"),
        provider_requests=tuple(requests),
        detection_times=tuple(
            _datetime_value(item, "detection_time") for item in detection_value
        ),
        job_id=(
            None
            if operational["job_id"] is None
            else _string(operational["job_id"], "job_id")
        ),
        local_manifest_path=(
            None
            if operational["local_manifest_path"] is None
            else _string(operational["local_manifest_path"], "local_manifest_path")
        ),
        notes=notes,
    )
    lineage_mapping = _exact_mapping(
        top["lineage"], "lineage", {"parent_snapshot_id", "operation_id"}
    )
    lineage = SnapshotLineage(
        parent_snapshot_id=(
            None
            if lineage_mapping["parent_snapshot_id"] is None
            else _string(lineage_mapping["parent_snapshot_id"], "parent_snapshot_id")
        ),
        operation_id=(
            None
            if lineage_mapping["operation_id"] is None
            else _string(lineage_mapping["operation_id"], "operation_id")
        ),
    )
    manifest = SnapshotManifest(
        content_identity=identity,
        operational_metadata=metadata,
        lineage=lineage,
    )
    declared_id = _string(top["snapshot_id"], "snapshot_id")
    if declared_id != manifest.snapshot_id:
        raise ValueError("manifest snapshot ID does not match content identity")
    return manifest


def _decode_manifest(raw: bytes) -> SnapshotManifest:
    if not isinstance(raw, bytes):
        raise _SnapshotFailure(
            ErrorCategory.INTEGRITY_CHECKSUM,
            "Published snapshot manifest bytes are not readable.",
            "Repair or republish the complete snapshot manifest.",
        )
    try:

        def reject_duplicate_pairs(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            local_keys: set[str] = set()
            for key, value in pairs:
                if key in local_keys:
                    raise ValueError("duplicate manifest key")
                local_keys.add(key)
                result[key] = value
            return result

        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs
        )
        if not isinstance(document, Mapping):
            raise ValueError("manifest root is not a mapping")
        if canonical_json(document) != raw:
            raise ValueError("manifest is not canonical")
        manifest = _parse_manifest_document(document)
        if canonical_json(manifest.to_manifest_dict()) != raw:
            raise ValueError("manifest canonical content does not match domain fields")
        return manifest
    except _SnapshotFailure:
        raise
    except Exception as error:
        del error
        raise _SnapshotFailure(
            ErrorCategory.INTEGRITY_CHECKSUM,
            "Published snapshot manifest is incomplete or does not match its content identity.",
            "Restore the complete manifest or publish a new snapshot.",
        ) from None


def _payload_error(
    message: str,
    corrective_action: str,
    *,
    checksum: str | None = None,
    field_path: str | None = None,
) -> _SnapshotFailure:
    return _SnapshotFailure(
        ErrorCategory.INTEGRITY_CHECKSUM,
        message,
        corrective_action,
        checksum=checksum,
        field_path=field_path,
    )


class SnapshotManager:
    """Open, inspect, and list immutable published snapshots.

    ``SnapshotManager`` is deliberately read-only in this task.  Publication,
    atomic rename, and startup reconciliation belong to the storage task that
    follows.  Every successful open is pinned to a manifest plus all verified
    object bytes; no staging directory, mutable ``latest`` pointer, or partially
    indexed snapshot is ever returned as a handle.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        storage: SnapshotByteStore | None = None,
        metadata: SnapshotIndex | None = None,
        clock: SnapshotClock | None = None,
    ) -> None:
        if storage is None:
            if root is None:
                raise TypeError("root or storage must be supplied")
            storage = LocalPublishedSnapshotStore(root)
        self._storage = storage
        self._metadata = metadata
        self._clock = clock or SystemSnapshotClock()

    def open_verified(self, snapshot_id: str) -> Result[VerifiedSnapshotHandle]:
        """Verify a complete published snapshot before returning its immutable handle."""

        try:
            _, handle = self._open_manifest(snapshot_id)
            return Ok(handle)
        except _SnapshotFailure as failure:
            return Err((self._error("snapshot.open", snapshot_id, failure),))
        except Exception as error:
            actionable = ActionableError.from_unexpected_exception(
                "snapshot.open", error
            )
            return Err((actionable,))

    open_snapshot = open_verified

    def inspect_snapshot(self, snapshot_id: str) -> Result[SnapshotDetail]:
        """Return verified manifest provenance, validation, and readiness details."""

        try:
            manifest, handle = self._open_manifest(snapshot_id)
            summary = _summary_from_manifest(manifest, availability="available")
            detail = SnapshotDetail(
                manifest=manifest,
                handle=handle,
                summary=summary,
                provenance=SnapshotProvenance(
                    provider=manifest.content_identity.provider,
                    requested_range=manifest.content_identity.requested_range,
                    covered_range=manifest.content_identity.covered_range,
                    configured_universe=manifest.content_identity.configured_universe,
                    benchmark_symbol=manifest.content_identity.benchmark_symbol,
                    calendar=manifest.content_identity.calendar,
                    schema_versions=manifest.content_identity.schema_versions,
                    configuration_checksum=manifest.content_identity.configuration_checksum,
                    object_references=manifest.content_identity.objects,
                    validation_report_checksum=manifest.content_identity.validation_report_checksum,
                    created_at=manifest.operational_metadata.created_at,
                    provider_requests=manifest.operational_metadata.provider_requests,
                    parent_snapshot_id=manifest.lineage.parent_snapshot_id,
                    operation_id=manifest.lineage.operation_id,
                ),
                validation_summary=manifest.content_identity.validation_summary,
                readiness=_readiness(
                    manifest.content_identity.validation_summary,
                    available=True,
                ),
            )
            return Ok(detail)
        except _SnapshotFailure as failure:
            return Err((self._error("snapshot.inspect", snapshot_id, failure),))
        except Exception as error:
            actionable = ActionableError.from_unexpected_exception(
                "snapshot.inspect", error
            )
            return Err((actionable,))

    inspect = inspect_snapshot

    def list_snapshots(
        self, query: SnapshotQuery | None = None
    ) -> SnapshotPage[SnapshotSummary]:
        """List bounded summaries from the metadata index or published directories."""

        try:
            resolved_query = query or SnapshotQuery()
        except Exception as error:
            actionable = ActionableError.from_unexpected_exception(
                "snapshot.list", error
            )
            return SnapshotPage(items=(), page=0, page_size=100, errors=(actionable,))

        if self._metadata is not None:
            try:
                records = tuple(
                    self._metadata.list_snapshots(
                        provider=resolved_query.provider,
                        availability=resolved_query.availability,
                        page=resolved_query.page,
                        page_size=resolved_query.page_size,
                    )
                )
                summaries: list[SnapshotSummary] = []
                for record in records:
                    summary = self._summary_from_record(record)
                    if summary.availability == "available":
                        try:
                            identifier = _require_snapshot_id(summary.snapshot_id)
                            manifest = self._read_manifest_only(identifier)
                            self._assert_index_matches(record, manifest)
                            summary = _summary_from_manifest(
                                manifest, availability=summary.availability
                            )
                        except _SnapshotFailure as failure:
                            summary = replace(
                                summary,
                                integrity_error=failure.message,
                            )
                    summaries.append(summary)
                return SnapshotPage(
                    items=tuple(summaries),
                    page=resolved_query.page,
                    page_size=resolved_query.page_size,
                )
            except Exception as error:
                actionable = ActionableError.from_unexpected_exception(
                    "snapshot.list", error
                )
                return SnapshotPage(
                    items=(),
                    page=resolved_query.page,
                    page_size=resolved_query.page_size,
                    errors=(actionable,),
                )

        list_ids = getattr(self._storage, "list_published_manifest_ids", None)
        if not callable(list_ids):
            error = ActionableError(
                operation="snapshot.list",
                category=ErrorCategory.STORAGE_IO,
                message="Snapshot discovery is unavailable because no metadata index is configured.",
                corrective_action="Configure the metadata index or reconcile published snapshots.",
            )
            return SnapshotPage(
                items=(),
                page=resolved_query.page,
                page_size=resolved_query.page_size,
                errors=(error,),
            )
        try:
            summaries: list[SnapshotSummary] = []
            for identifier in list_ids():
                try:
                    manifest = self._read_manifest_only(identifier)
                except _SnapshotFailure:
                    continue
                summary = _summary_from_manifest(manifest, availability="available")
                if (
                    resolved_query.provider is not None
                    and summary.provider != resolved_query.provider
                ):
                    continue
                if (
                    resolved_query.availability is not None
                    and summary.availability != resolved_query.availability
                ):
                    continue
                summaries.append(summary)
            summaries.sort(
                key=lambda item: (-item.created_at.timestamp(), item.snapshot_id)
            )
            start = resolved_query.page * resolved_query.page_size
            end = start + resolved_query.page_size
            return SnapshotPage(
                items=tuple(summaries[start:end]),
                page=resolved_query.page,
                page_size=resolved_query.page_size,
                total=len(summaries),
            )
        except Exception as error:
            actionable = ActionableError.from_unexpected_exception(
                "snapshot.list", error
            )
            return SnapshotPage(
                items=(),
                page=resolved_query.page,
                page_size=resolved_query.page_size,
                errors=(actionable,),
            )

    list = list_snapshots

    def reject_mutation(
        self,
        snapshot_id: str | None = None,
        *,
        operation: str = "snapshot.mutate",
    ) -> Result[None]:
        """Return the required immutable-snapshot error for every write attempt."""

        del snapshot_id
        return Err(
            (
                ActionableError(
                    operation=operation,
                    category=ErrorCategory.STORAGE_ATOMICITY,
                    message="Published Data_Snapshots are immutable through platform operations.",
                    corrective_action="Publish a new Data_Snapshot instead of replacing its manifest or objects.",
                    field_path="snapshot_id",
                ),
            )
        )

    reject_published_mutation = reject_mutation
    guard_mutation = reject_mutation

    def publish(self, candidate: object, **options: object) -> Result[None]:
        """Keep this read-only access service from replacing a publication.

        Atomic publication is intentionally implemented by the following storage
        task.  Exposing a fail-closed method here prevents callers from treating
        an access service as a mutable ``latest`` writer in the meantime.
        """

        del candidate, options
        return self.reject_mutation(operation="snapshot.publish")

    def replace_manifest(self, snapshot_id: str, manifest_bytes: bytes) -> Result[None]:
        del manifest_bytes
        return self.reject_mutation(snapshot_id, operation="snapshot.replace_manifest")

    def replace_object(
        self, snapshot_id: str, relative_uri: str, object_bytes: bytes
    ) -> Result[None]:
        del relative_uri, object_bytes
        return self.reject_mutation(snapshot_id, operation="snapshot.replace_object")

    def update_snapshot(self, snapshot_id: str, **changes: object) -> Result[None]:
        del changes
        return self.reject_mutation(snapshot_id, operation="snapshot.update")

    def delete_snapshot(self, snapshot_id: str) -> Result[None]:
        return self.reject_mutation(snapshot_id, operation="snapshot.delete")

    mutate_snapshot = update_snapshot

    def _open_manifest(
        self, snapshot_id: str
    ) -> tuple[SnapshotManifest, VerifiedSnapshotHandle]:
        identifier = _require_snapshot_id(snapshot_id)
        record = self._indexed_record(identifier)
        if (
            record is not None
            and _availability_text(getattr(record, "availability", "available"))
            != "available"
        ):
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The snapshot is indexed but currently unavailable for use.",
                "Reconcile the snapshot index or publish a new complete snapshot.",
                field_path="snapshot.availability",
            )
        manifest_uri = self._manifest_uri(record, identifier)
        raw = self._read_manifest_bytes(identifier, manifest_uri)
        manifest = _decode_manifest(raw)
        if manifest.snapshot_id != identifier:
            raise _payload_error(
                "Published manifest does not match the requested Snapshot_ID.",
                "Restore the matching manifest or publish a new snapshot.",
                field_path="snapshot_id",
            )
        if record is not None:
            self._assert_index_matches(record, manifest)
        self._verify_references(manifest, record)
        verified_at = _require_aware_datetime("verified_at", self._clock.utc_now())
        return manifest, VerifiedSnapshotHandle.from_manifest(
            manifest, verified_at=verified_at
        )

    def _read_manifest_only(self, snapshot_id: str) -> SnapshotManifest:
        identifier = _require_snapshot_id(snapshot_id)
        raw = self._read_manifest_bytes(
            identifier, f"snapshots/{identifier}/manifest.json"
        )
        manifest = _decode_manifest(raw)
        if manifest.snapshot_id != identifier:
            raise _payload_error(
                "Published manifest does not match its snapshot directory.",
                "Restore the complete publication or publish a new snapshot.",
                field_path="snapshot_id",
            )
        return manifest

    def _indexed_record(self, identifier: str) -> object | None:
        if self._metadata is None:
            return None
        try:
            record = self._metadata.get_snapshot(identifier)
        except Exception as error:
            if error.__class__.__name__ in {
                "MetadataNotFoundError",
                "SnapshotNotFoundError",
            }:
                raise _SnapshotFailure(
                    ErrorCategory.STORAGE_IO,
                    "The published snapshot is not available in the metadata index.",
                    "Reconcile the metadata index or publish a new complete snapshot.",
                    field_path="snapshot.index",
                ) from None
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The snapshot metadata index could not be read.",
                "Repair or reconcile the metadata index before opening the snapshot.",
                field_path="snapshot.index",
            ) from None
        if record is None:
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The published snapshot is not available in the metadata index.",
                "Reconcile the metadata index or publish a new complete snapshot.",
                field_path="snapshot.index",
            )
        return record

    @staticmethod
    def _manifest_uri(record: object | None, identifier: str) -> str:
        expected = f"snapshots/{identifier}/manifest.json"
        if record is None:
            return expected
        uri = getattr(record, "manifest_uri", expected)
        try:
            normalized = _safe_snapshot_uri(uri)
        except Exception:
            raise _payload_error(
                "Snapshot index contains an invalid manifest location.",
                "Reconcile the index and publish a complete snapshot.",
                field_path="snapshot.manifest_uri",
            ) from None
        if normalized != expected:
            raise _payload_error(
                "Snapshot index does not point to the immutable published manifest.",
                "Reconcile the index or publish a new snapshot at its content-derived location.",
                field_path="snapshot.manifest_uri",
            )
        return normalized

    def _read_manifest_bytes(self, identifier: str, relative_uri: str) -> bytes:
        try:
            if relative_uri == f"snapshots/{identifier}/manifest.json":
                data = self._storage.read_manifest(identifier)
            else:
                data = self._storage.read_object(relative_uri)
        except _SnapshotFailure:
            raise
        except FileNotFoundError:
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The complete published snapshot manifest is unavailable.",
                "Restore the publication directory or publish a new snapshot.",
                field_path="snapshot.manifest",
            ) from None
        except Exception as error:
            # FilesystemStore verifies referenced CAS bytes while reading a
            # manifest. Preserve that integrity classification at the
            # application boundary instead of reporting corruption as I/O.
            if error.__class__.__name__ == "IntegrityVerificationError":
                raise _SnapshotFailure(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "A published snapshot object failed checksum verification.",
                    "Restore the referenced immutable object or publish a new snapshot.",
                    field_path="snapshot.objects",
                ) from None
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The complete published snapshot manifest could not be read.",
                "Repair the publication or reconcile the local storage index.",
                field_path="snapshot.manifest",
            ) from None
        if not isinstance(data, bytes):
            raise _SnapshotFailure(
                ErrorCategory.STORAGE_IO,
                "The published snapshot manifest reader returned invalid bytes.",
                "Repair the publication or publish a new snapshot.",
                field_path="snapshot.manifest",
            )
        return data

    def _read_object_bytes(
        self,
        relative_uri: str,
        *,
        checksum: str | None = None,
    ) -> bytes:
        try:
            data = self._storage.read_object(relative_uri)
        except FileNotFoundError:
            raise _SnapshotFailure(
                ErrorCategory.INTEGRITY_CHECKSUM,
                "A published snapshot object is missing.",
                "Restore the referenced object or publish a new snapshot.",
                checksum=checksum,
                field_path="snapshot.objects",
            ) from None
        except Exception as error:
            del error
            raise _SnapshotFailure(
                ErrorCategory.INTEGRITY_CHECKSUM,
                "A published snapshot object could not be read.",
                "Restore the referenced object or publish a new snapshot.",
                checksum=checksum,
                field_path="snapshot.objects",
            ) from None
        if not isinstance(data, bytes):
            raise _SnapshotFailure(
                ErrorCategory.INTEGRITY_CHECKSUM,
                "A published snapshot object reader returned invalid bytes.",
                "Restore the referenced object or publish a new snapshot.",
                checksum=checksum,
                field_path="snapshot.objects",
            )
        return data

    def _read_validation_bytes(self, checksum: str, record: object | None) -> bytes:
        relative_uri: str | None = None
        if self._metadata is not None:
            for method_name in ("get_artifact", "get_data_object"):
                method = getattr(self._metadata, method_name, None)
                if not callable(method):
                    continue
                try:
                    indexed = method(checksum)
                except Exception:
                    continue
                availability = getattr(indexed, "availability", "available")
                if _availability_text(availability) != "available":
                    raise _SnapshotFailure(
                        ErrorCategory.STORAGE_IO,
                        "The validation artifact is indexed as unavailable.",
                        "Reconcile the validation artifact or publish a new snapshot.",
                        checksum=checksum,
                        field_path="snapshot.validation_report_checksum",
                    )
                relative_uri = getattr(indexed, "relative_uri", None)
                if relative_uri is not None:
                    break
        if relative_uri is not None:
            return self._read_object_bytes(relative_uri, checksum=checksum)
        for method_name in ("read_by_checksum", "read_checksum"):
            method = getattr(self._storage, method_name, None)
            if not callable(method):
                continue
            try:
                data = method(checksum)
            except FileNotFoundError:
                continue
            except Exception as error:
                del error
                raise _SnapshotFailure(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The published validation artifact is corrupt or unreadable.",
                    "Restore the validation artifact or publish a new snapshot.",
                    checksum=checksum,
                    field_path="snapshot.validation_report_checksum",
                ) from None
            if isinstance(data, bytes):
                return data
        raise _SnapshotFailure(
            ErrorCategory.INTEGRITY_CHECKSUM,
            "The snapshot validation artifact referenced by the manifest is missing.",
            "Restore the validation artifact or publish a new snapshot.",
            checksum=checksum,
            field_path="snapshot.validation_report_checksum",
        )

    def _verify_references(
        self, manifest: SnapshotManifest, record: object | None
    ) -> None:
        verified_checksums: set[str] = set()
        for reference in manifest.content_identity.objects:
            data = self._read_object_bytes(
                reference.relative_uri,
                checksum=reference.checksum,
            )
            if (
                len(data) != reference.byte_size
                or sha256_bytes(data) != reference.checksum
            ):
                raise _SnapshotFailure(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "A published snapshot object failed checksum or byte-size verification.",
                    "Restore the referenced immutable object or publish a new snapshot.",
                    checksum=reference.checksum,
                    field_path="snapshot.objects",
                )
            verified_checksums.add(reference.checksum)
        report_checksum = manifest.content_identity.validation_report_checksum
        if report_checksum not in verified_checksums:
            report = self._read_validation_bytes(report_checksum, record)
            if sha256_bytes(report) != report_checksum:
                raise _SnapshotFailure(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The published validation artifact failed checksum verification.",
                    "Restore the validation artifact or publish a new snapshot.",
                    checksum=report_checksum,
                    field_path="snapshot.validation_report_checksum",
                )

    def _assert_index_matches(self, record: object, manifest: SnapshotManifest) -> None:
        comparisons = (
            ("snapshot_id", getattr(record, "snapshot_id", None), manifest.snapshot_id),
            (
                "manifest_checksum",
                getattr(record, "manifest_checksum", None),
                manifest.manifest_checksum,
            ),
            (
                "content_identity_checksum",
                getattr(record, "content_identity_checksum", None),
                manifest.content_identity_checksum,
            ),
            (
                "configuration_checksum",
                getattr(record, "configuration_checksum", None),
                manifest.content_identity.configuration_checksum,
            ),
            (
                "provider",
                getattr(record, "provider", None),
                manifest.content_identity.provider,
            ),
            (
                "benchmark_symbol",
                getattr(record, "benchmark_symbol", None),
                manifest.content_identity.benchmark_symbol,
            ),
            (
                "comparison_ready",
                getattr(record, "comparison_ready", None),
                manifest.content_identity.validation_summary.comparison_ready,
            ),
        )
        for field_name, actual, expected in comparisons:
            normalized_actual = (
                _availability_text(actual) if field_name == "availability" else actual
            )
            if normalized_actual != expected:
                raise _payload_error(
                    "Snapshot metadata does not match the published manifest.",
                    "Reconcile the metadata index or publish a new snapshot.",
                    field_path=f"snapshot.{field_name}",
                )
        for field_name, actual, expected in (
            (
                "requested_start",
                getattr(record, "requested_start", None),
                manifest.content_identity.requested_range.start,
            ),
            (
                "requested_end",
                getattr(record, "requested_end", None),
                manifest.content_identity.requested_range.end,
            ),
            (
                "covered_start",
                getattr(record, "covered_start", None),
                manifest.content_identity.covered_range.start
                if manifest.content_identity.covered_range
                else None,
            ),
            (
                "covered_end",
                getattr(record, "covered_end", None),
                manifest.content_identity.covered_range.end
                if manifest.content_identity.covered_range
                else None,
            ),
        ):
            if actual != expected:
                raise _payload_error(
                    "Snapshot metadata range does not match the published manifest.",
                    "Reconcile the metadata index or publish a new snapshot.",
                    field_path=f"snapshot.{field_name}",
                )
        indexed_universe = tuple(getattr(record, "universe", ()))
        if indexed_universe != manifest.content_identity.configured_universe:
            raise _payload_error(
                "Snapshot metadata universe does not match the published manifest.",
                "Reconcile the metadata index or publish a new snapshot.",
                field_path="snapshot.universe",
            )
        list_objects = getattr(self._metadata, "list_snapshot_objects", None)
        if callable(list_objects):
            try:
                rows = tuple(list_objects(manifest.snapshot_id))
            except Exception as error:
                del error
                raise _SnapshotFailure(
                    ErrorCategory.STORAGE_IO,
                    "Snapshot object indexing is unavailable.",
                    "Reconcile the metadata index before opening the snapshot.",
                    field_path="snapshot.objects",
                ) from None
            expected_rows = tuple(
                sorted(
                    (
                        reference.object_kind.value,
                        reference.checksum,
                        reference.symbol,
                        reference.session_year,
                        ordinal,
                    )
                    for ordinal, reference in enumerate(
                        manifest.content_identity.objects
                    )
                )
            )
            actual_rows = tuple(
                sorted(
                    (
                        str(getattr(row, "role", "")),
                        str(getattr(row, "checksum", "")),
                        getattr(row, "symbol", None),
                        getattr(row, "session_year", None),
                        getattr(row, "ordinal", -1),
                    )
                    for row in rows
                )
            )
            if actual_rows != expected_rows:
                raise _payload_error(
                    "Snapshot metadata object references do not match the manifest.",
                    "Reconcile the metadata index or publish a new snapshot.",
                    field_path="snapshot.objects",
                )

    def _summary_from_record(self, record: object) -> SnapshotSummary:
        covered = None
        covered_start = getattr(record, "covered_start", None)
        covered_end = getattr(record, "covered_end", None)
        if covered_start is not None and covered_end is not None:
            covered = DateRange(covered_start, covered_end)
        return SnapshotSummary(
            snapshot_id=getattr(record, "snapshot_id"),
            provider=getattr(record, "provider"),
            requested_range=DateRange(
                getattr(record, "requested_start"), getattr(record, "requested_end")
            ),
            covered_range=covered,
            configured_universe=tuple(getattr(record, "universe")),
            benchmark_symbol=getattr(record, "benchmark_symbol"),
            comparison_ready=bool(getattr(record, "comparison_ready")),
            availability=_availability_text(getattr(record, "availability")),
            created_at=getattr(record, "created_at"),
            manifest_checksum=getattr(record, "manifest_checksum"),
            content_identity_checksum=getattr(record, "content_identity_checksum"),
            parent_snapshot_id=getattr(record, "parent_snapshot_id", None),
            limitation_disclosure=LimitationDisclosure.current(),
        )

    def _error(
        self, operation: str, snapshot_id: str, failure: _SnapshotFailure
    ) -> ActionableError:
        try:
            normalized_id = _require_snapshot_id(snapshot_id)
        except Exception:
            normalized_id = None
        return ActionableError(
            operation=operation,
            category=failure.category,
            message=failure.message,
            corrective_action=failure.corrective_action,
            field_path=failure.field_path,
            checksum=failure.checksum,
            correlation_id=normalized_id,
        )


def _summary_from_manifest(
    manifest: SnapshotManifest, *, availability: str, integrity_error: str | None = None
) -> SnapshotSummary:
    identity = manifest.content_identity
    return SnapshotSummary(
        snapshot_id=manifest.snapshot_id,
        provider=identity.provider,
        requested_range=identity.requested_range,
        covered_range=identity.covered_range,
        configured_universe=identity.configured_universe,
        benchmark_symbol=identity.benchmark_symbol,
        comparison_ready=identity.validation_summary.comparison_ready,
        availability=availability,
        created_at=manifest.operational_metadata.created_at,
        manifest_checksum=manifest.manifest_checksum,
        content_identity_checksum=manifest.content_identity_checksum,
        parent_snapshot_id=manifest.lineage.parent_snapshot_id,
        limitation_disclosure=manifest.limitation_disclosure,
        validation_summary=identity.validation_summary,
        integrity_error=integrity_error,
    )


def _readiness(summary: ValidationSummary, *, available: bool) -> SnapshotReadiness:
    reasons: list[str] = []
    if not available:
        reasons.append("snapshot_unavailable")
    if not summary.comparison_ready:
        reasons.append("benchmark_comparison_not_ready")
    if summary.failed_symbols:
        reasons.append("failed_symbols_recorded")
    if summary.gap_count:
        reasons.append("data_gaps_recorded")
    if summary.stale_symbols:
        reasons.append("stale_symbols_recorded")
    return SnapshotReadiness(
        available=available,
        comparison_ready=summary.comparison_ready,
        failed_symbols=summary.failed_symbols,
        stale_symbols=summary.stale_symbols,
        gap_count=summary.gap_count,
        reasons=tuple(reasons),
    )


SnapshotService = SnapshotManager


__all__ = [
    "LocalPublishedSnapshotStore",
    "SnapshotByteStore",
    "SnapshotClock",
    "SnapshotDetail",
    "SnapshotListPage",
    "SnapshotManager",
    "SnapshotPage",
    "SnapshotProvenance",
    "SnapshotQuery",
    "SnapshotReadiness",
    "SnapshotService",
    "SnapshotSummary",
    "SystemSnapshotClock",
    "SnapshotManifestAssembler",
    "SnapshotManifestBuilder",
    "ValidationFacts",
    "assemble_snapshot_manifest",
    "build_snapshot_manifest",
]
