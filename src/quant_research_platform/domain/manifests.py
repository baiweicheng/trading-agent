"""Framework-free snapshot manifest and content-addressing value objects.

A snapshot's scientific identity is represented separately from timestamps, job
references, request history, lineage, and local locations.  Infrastructure is
responsible for byte verification; these objects make the exact verified facts
immutable and canonical once it has done so.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import cast

from .canonical import (
    CanonicalJSONValue,
    canonicalize,
    sha256_canonical_json,
)
from .errors import LimitationDisclosure
from .market import (
    DateRange,
    ProviderRequestMetadata,
    ValidationSummary,
    normalize_symbol,
)

SNAPSHOT_MANIFEST_SCHEMA_VERSION = "snapshot_manifest_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^snap_[0-9a-f]{64}$")


def _required_text(field_name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _checksum(field_name: str, value: str) -> str:
    checksum = _required_text(field_name, value)
    if _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return checksum


def _snapshot_id(field_name: str, value: str) -> str:
    snapshot_id = _required_text(field_name, value)
    if _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None:
        raise ValueError(f"{field_name} must be a content-derived snapshot ID")
    return snapshot_id


def _utc_timestamp(field_name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _non_negative_int(field_name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _freeze_json(value: CanonicalJSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(
    field_name: str, value: Mapping[str, object]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    canonical = canonicalize(value)
    if not isinstance(canonical, dict):  # Defensive: mappings canonicalize to dicts.
        raise AssertionError("canonical mapping expected")
    return cast(Mapping[str, object], _freeze_json(canonical))


def _canonical_mapping(value: Mapping[str, object]) -> dict[str, CanonicalJSONValue]:
    canonical = canonicalize(value)
    if not isinstance(canonical, dict):  # Defensive: mappings canonicalize to dicts.
        raise AssertionError("canonical mapping expected")
    return canonical


def _validate_relative_uri(value: str) -> str:
    uri = _required_text("relative_uri", value)
    if "\\" in uri:
        raise ValueError("relative_uri must use POSIX separators")
    path = PurePosixPath(uri)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError("relative_uri must be a non-escaping relative logical URI")
    return path.as_posix()


class ObjectKind(StrEnum):
    """Stable logical content collections referenced from a manifest."""

    RAW = "raw"
    NORMALIZED = "normalized"
    QUARANTINE = "quarantine"
    GAP = "gap"
    VALIDATION = "validation"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Publication-gated reference used to stream one verified artifact."""

    checksum: str
    byte_size: int
    relative_uri: str
    metadata_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksum", _checksum("checksum", self.checksum))
        object.__setattr__(
            self,
            "relative_uri",
            _validate_relative_uri(self.relative_uri),
        )
        object.__setattr__(
            self,
            "byte_size",
            _non_negative_int("byte_size", self.byte_size),
        )
        object.__setattr__(
            self,
            "metadata_checksum",
            _checksum("metadata_checksum", self.metadata_checksum),
        )


@dataclass(frozen=True, slots=True)
class ContentAddressedObjectRef:
    """One checksummed, location-independent object referenced by a manifest."""

    object_kind: ObjectKind
    checksum: str
    relative_uri: str
    schema_version: str
    row_count: int
    byte_size: int
    symbol: str | None = None
    session_year: int | None = None
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        try:
            object_kind = ObjectKind(self.object_kind)
        except ValueError as error:
            raise ValueError(
                f"unsupported object kind: {self.object_kind!r}"
            ) from error
        object.__setattr__(self, "object_kind", object_kind)
        object.__setattr__(self, "checksum", _checksum("checksum", self.checksum))
        object.__setattr__(
            self, "relative_uri", _validate_relative_uri(self.relative_uri)
        )
        object.__setattr__(
            self,
            "schema_version",
            _required_text("schema_version", self.schema_version),
        )
        object.__setattr__(
            self, "row_count", _non_negative_int("row_count", self.row_count)
        )
        object.__setattr__(
            self, "byte_size", _non_negative_int("byte_size", self.byte_size)
        )
        if self.symbol is not None:
            object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if self.session_year is not None:
            if isinstance(self.session_year, bool) or not isinstance(
                self.session_year, int
            ):
                raise TypeError("session_year must be an integer or None")
            if not 1 <= self.session_year <= 9999:
                raise ValueError("session_year must be a valid calendar year")
        object.__setattr__(
            self, "media_type", _required_text("media_type", self.media_type)
        )

    def sort_key(self) -> tuple[str, str, int, str, str]:
        return (
            self.object_kind.value,
            self.symbol or "",
            self.session_year if self.session_year is not None else -1,
            self.relative_uri,
            self.checksum,
        )

    def to_content_dict(self) -> dict[str, object]:
        return {
            "object_kind": self.object_kind.value,
            "checksum": self.checksum,
            "relative_uri": self.relative_uri,
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "symbol": self.symbol,
            "session_year": self.session_year,
            "media_type": self.media_type,
        }


ObjectRef = ContentAddressedObjectRef


@dataclass(frozen=True, slots=True)
class CalendarIdentity:
    """The pinned exchange calendar name, package version, and schedule digest."""

    name: str
    version: str
    schedule_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text("calendar name", self.name))
        object.__setattr__(
            self, "version", _required_text("calendar version", self.version)
        )
        object.__setattr__(
            self,
            "schedule_checksum",
            _checksum("schedule_checksum", self.schedule_checksum),
        )

    def to_content_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "schedule_checksum": self.schedule_checksum,
        }


@dataclass(frozen=True, slots=True)
class SnapshotSchemaVersions:
    """Version pins for every scientific snapshot representation and policy."""

    manifest_schema_version: str = SNAPSHOT_MANIFEST_SCHEMA_VERSION
    raw_schema_version: str = "raw_v1"
    normalized_schema_version: str = "daily_bar_v1"
    quarantine_schema_version: str = "quarantine_v1"
    validation_report_schema_version: str = "validation_report_v1"
    corporate_action_policy_version: str = "causal_forward_v1"

    def __post_init__(self) -> None:
        for field_name in (
            "manifest_schema_version",
            "raw_schema_version",
            "normalized_schema_version",
            "quarantine_schema_version",
            "validation_report_schema_version",
            "corporate_action_policy_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(field_name, getattr(self, field_name)),
            )

    def to_content_dict(self) -> dict[str, str]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "raw_schema_version": self.raw_schema_version,
            "normalized_schema_version": self.normalized_schema_version,
            "quarantine_schema_version": self.quarantine_schema_version,
            "validation_report_schema_version": self.validation_report_schema_version,
            "corporate_action_policy_version": self.corporate_action_policy_version,
        }


@dataclass(frozen=True, slots=True)
class SnapshotLineage:
    """Operational parentage retained outside the content-derived Snapshot ID."""

    parent_snapshot_id: str | None = None
    operation_id: str | None = None

    def __post_init__(self) -> None:
        if self.parent_snapshot_id is not None:
            object.__setattr__(
                self,
                "parent_snapshot_id",
                _snapshot_id("parent_snapshot_id", self.parent_snapshot_id),
            )
        if self.operation_id is not None:
            object.__setattr__(
                self, "operation_id", _required_text("operation_id", self.operation_id)
            )

    def to_operational_dict(self) -> dict[str, str | None]:
        return {
            "parent_snapshot_id": self.parent_snapshot_id,
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True, slots=True)
class OperationalMetadata:
    """Timestamped/local facts deliberately excluded from scientific identity."""

    created_at: datetime
    provider_requests: tuple[ProviderRequestMetadata, ...] = ()
    detection_times: tuple[datetime, ...] = ()
    job_id: str | None = None
    local_manifest_path: str | None = None
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "created_at", _utc_timestamp("created_at", self.created_at)
        )
        if not isinstance(self.provider_requests, tuple):
            raise TypeError("provider_requests must be an immutable tuple")
        if any(
            not isinstance(request, ProviderRequestMetadata)
            for request in self.provider_requests
        ):
            raise TypeError(
                "provider_requests may contain only ProviderRequestMetadata values"
            )
        if not isinstance(self.detection_times, tuple):
            raise TypeError("detection_times must be an immutable tuple")
        detection_times = tuple(
            _utc_timestamp("detection_time", value) for value in self.detection_times
        )
        object.__setattr__(self, "detection_times", detection_times)
        if self.job_id is not None:
            object.__setattr__(self, "job_id", _required_text("job_id", self.job_id))
        if self.local_manifest_path is not None:
            object.__setattr__(
                self,
                "local_manifest_path",
                _required_text("local_manifest_path", self.local_manifest_path),
            )
        object.__setattr__(self, "notes", _freeze_mapping("notes", self.notes))

    def to_operational_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "provider_requests": [
                request.to_operational_dict() for request in self.provider_requests
            ],
            "detection_times": list(self.detection_times),
            "job_id": self.job_id,
            "local_manifest_path": self.local_manifest_path,
            "notes": _canonical_mapping(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SnapshotContentIdentity:
    """Only deterministic scientific facts from which a Snapshot ID is derived."""

    provider: str
    requested_range: DateRange
    configured_universe: tuple[str, ...]
    benchmark_symbol: str
    calendar: CalendarIdentity
    configuration_checksum: str
    objects: tuple[ContentAddressedObjectRef, ...]
    validation_report_checksum: str
    validation_summary: ValidationSummary
    limitation_disclosure: LimitationDisclosure
    covered_range: DateRange | None = None
    schema_versions: SnapshotSchemaVersions = field(
        default_factory=SnapshotSchemaVersions
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _required_text("provider", self.provider))
        if not isinstance(self.requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange")
        if self.covered_range is not None:
            if not isinstance(self.covered_range, DateRange):
                raise TypeError("covered_range must be a DateRange or None")
            if (
                self.covered_range.start < self.requested_range.start
                or self.covered_range.end > self.requested_range.end
            ):
                raise ValueError("covered_range must be contained by requested_range")
        if not isinstance(self.configured_universe, tuple):
            raise TypeError("configured_universe must be an immutable tuple")
        universe = tuple(
            normalize_symbol(symbol) for symbol in self.configured_universe
        )
        if not universe:
            raise ValueError("configured_universe must not be empty")
        if len(set(universe)) != len(universe):
            raise ValueError(
                "configured_universe must contain distinct normalized symbols"
            )
        object.__setattr__(self, "configured_universe", universe)
        object.__setattr__(
            self, "benchmark_symbol", normalize_symbol(self.benchmark_symbol)
        )
        if not isinstance(self.calendar, CalendarIdentity):
            raise TypeError("calendar must be a CalendarIdentity")
        object.__setattr__(
            self,
            "configuration_checksum",
            _checksum("configuration_checksum", self.configuration_checksum),
        )
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be an immutable tuple")
        if any(
            not isinstance(reference, ContentAddressedObjectRef)
            for reference in self.objects
        ):
            raise TypeError("objects may contain only ContentAddressedObjectRef values")
        objects = tuple(sorted(self.objects, key=ContentAddressedObjectRef.sort_key))
        if len({reference.sort_key() for reference in objects}) != len(objects):
            raise ValueError("objects must not contain duplicate logical references")
        if len({reference.checksum for reference in objects}) != len(objects):
            raise ValueError("each referenced content checksum may appear only once")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(
            self,
            "validation_report_checksum",
            _checksum("validation_report_checksum", self.validation_report_checksum),
        )
        if not isinstance(self.validation_summary, ValidationSummary):
            raise TypeError("validation_summary must be a ValidationSummary")
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be a LimitationDisclosure")
        if not isinstance(self.schema_versions, SnapshotSchemaVersions):
            raise TypeError("schema_versions must be a SnapshotSchemaVersions")
        if (
            self.covered_range is not None
            and self.validation_summary.covered_range is not None
            and self.covered_range != self.validation_summary.covered_range
        ):
            raise ValueError(
                "covered_range must match validation_summary.covered_range"
            )

    @property
    def limitation_disclosure_checksum(self) -> str:
        return sha256_canonical_json(
            {
                "version": self.limitation_disclosure.version,
                "lines": list(self.limitation_disclosure.lines()),
            }
        )

    @property
    def snapshot_id(self) -> str:
        return "snap_" + sha256_canonical_json(self.to_content_dict())

    @property
    def content_checksum(self) -> str:
        return sha256_canonical_json(self.to_content_dict())

    @property
    def failed_symbols(self) -> tuple[str, ...]:
        return self.validation_summary.failed_symbols

    @property
    def retained_parent_coverage_symbols(self) -> tuple[str, ...]:
        return self.validation_summary.retained_parent_coverage_symbols

    def to_content_dict(self) -> dict[str, object]:
        return {
            "schema_versions": self.schema_versions.to_content_dict(),
            "provider": self.provider,
            "requested_range": self.requested_range.to_content_dict(),
            "covered_range": (
                self.covered_range.to_content_dict() if self.covered_range else None
            ),
            "configured_universe": list(self.configured_universe),
            "benchmark_symbol": self.benchmark_symbol,
            "calendar": self.calendar.to_content_dict(),
            "configuration_checksum": self.configuration_checksum,
            "objects": [reference.to_content_dict() for reference in self.objects],
            "validation_report_checksum": self.validation_report_checksum,
            "validation_summary": self.validation_summary.to_content_dict(),
            "failed_symbols": list(self.failed_symbols),
            "retained_parent_coverage_symbols": list(
                self.retained_parent_coverage_symbols
            ),
            "limitation_disclosure": {
                "version": self.limitation_disclosure.version,
                "text_checksum": self.limitation_disclosure_checksum,
            },
        }


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """A complete immutable manifest with separate scientific and operational data."""

    content_identity: SnapshotContentIdentity
    operational_metadata: OperationalMetadata
    lineage: SnapshotLineage = field(default_factory=SnapshotLineage)

    def __post_init__(self) -> None:
        if not isinstance(self.content_identity, SnapshotContentIdentity):
            raise TypeError("content_identity must be a SnapshotContentIdentity")
        if not isinstance(self.operational_metadata, OperationalMetadata):
            raise TypeError("operational_metadata must be an OperationalMetadata")
        if not isinstance(self.lineage, SnapshotLineage):
            raise TypeError("lineage must be a SnapshotLineage")

    @property
    def snapshot_id(self) -> str:
        return self.content_identity.snapshot_id

    @property
    def content_identity_checksum(self) -> str:
        return self.content_identity.content_checksum

    @property
    def manifest_checksum(self) -> str:
        return sha256_canonical_json(self.to_manifest_dict())

    @property
    def limitation_disclosure(self) -> LimitationDisclosure:
        return self.content_identity.limitation_disclosure

    def to_content_identity_dict(self) -> dict[str, object]:
        return self.content_identity.to_content_dict()

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "content_identity": self.to_content_identity_dict(),
            "operational_metadata": self.operational_metadata.to_operational_dict(),
            "lineage": self.lineage.to_operational_dict(),
            "limitation_disclosure": {
                "version": self.limitation_disclosure.version,
                "lines": list(self.limitation_disclosure.lines()),
            },
        }


@dataclass(frozen=True, slots=True)
class VerifiedSnapshotHandle:
    """An immutable pin to a snapshot whose manifest and all references verified."""

    snapshot_id: str
    content_identity_checksum: str
    manifest_checksum: str
    object_references: tuple[ContentAddressedObjectRef, ...]
    verified_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _snapshot_id("snapshot_id", self.snapshot_id)
        )
        object.__setattr__(
            self,
            "content_identity_checksum",
            _checksum("content_identity_checksum", self.content_identity_checksum),
        )
        object.__setattr__(
            self,
            "manifest_checksum",
            _checksum("manifest_checksum", self.manifest_checksum),
        )
        if not isinstance(self.object_references, tuple):
            raise TypeError("object_references must be an immutable tuple")
        if any(
            not isinstance(reference, ContentAddressedObjectRef)
            for reference in self.object_references
        ):
            raise TypeError(
                "object_references may contain only ContentAddressedObjectRef values"
            )
        references = tuple(
            sorted(self.object_references, key=ContentAddressedObjectRef.sort_key)
        )
        if len({reference.checksum for reference in references}) != len(references):
            raise ValueError("object_references must not repeat a content checksum")
        object.__setattr__(self, "object_references", references)
        object.__setattr__(
            self, "verified_at", _utc_timestamp("verified_at", self.verified_at)
        )

    @classmethod
    def from_manifest(
        cls, manifest: SnapshotManifest, *, verified_at: datetime
    ) -> VerifiedSnapshotHandle:
        """Create a pinned handle after infrastructure has verified every object."""
        if not isinstance(manifest, SnapshotManifest):
            raise TypeError("manifest must be a SnapshotManifest")
        return cls(
            snapshot_id=manifest.snapshot_id,
            content_identity_checksum=manifest.content_identity_checksum,
            manifest_checksum=manifest.manifest_checksum,
            object_references=manifest.content_identity.objects,
            verified_at=verified_at,
        )


SnapshotHandle = VerifiedSnapshotHandle


__all__ = [
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "CalendarIdentity",
    "ContentAddressedObjectRef",
    "ObjectKind",
    "ObjectRef",
    "OperationalMetadata",
    "SnapshotContentIdentity",
    "SnapshotHandle",
    "SnapshotLineage",
    "SnapshotManifest",
    "SnapshotSchemaVersions",
    "VerifiedSnapshotHandle",
]
