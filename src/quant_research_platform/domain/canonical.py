"""Canonical scientific-content encoding and checksum primitives.

This module is deliberately framework-independent. It defines the byte-level
representation used by scientific identities while keeping operational details
out of that representation through explicit manifest projection helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction
from typing import TypeAlias

ByteLike: TypeAlias = bytes | bytearray | memoryview
CanonicalJSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["CanonicalJSONValue"]
    | dict[str, "CanonicalJSONValue"]
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# These fields are operational facts. They must not affect reproducible
# scientific identity even if a caller accidentally places them in a content
# identity mapping.
OPERATIONAL_CONTENT_FIELDS = frozenset(
    {
        "created_at",
        "creation_timestamp",
        "retrieved_at",
        "retrieval_timestamp",
        "detected_at",
        "started_at",
        "ended_at",
        "updated_at",
        "progress_at",
        "progress_timestamp",
        "run_id",
        "job_id",
        "request_id",
        "operation_id",
        "mlflow_run_id",
        "correlation_id",
        "local_path",
        "absolute_path",
        "storage_path",
        "staging_path",
        "manifest_path",
        "output_path",
        "operational_metadata",
        "lineage",
        "parent_snapshot_id",
        "attempted_parent_snapshot_id",
    }
)


class CanonicalizationError(ValueError):
    """Raised when a value has no unambiguous canonical representation."""


class ChecksumMismatchError(CanonicalizationError):
    """Raised when bytes do not match an expected SHA-256 checksum."""


def normalize_unicode(value: str) -> str:
    """Return *value* in Unicode NFC form."""
    if not isinstance(value, str):
        raise TypeError("Unicode normalization requires a string")
    return unicodedata.normalize("NFC", value)


def canonical_date(value: date) -> str:
    """Return an ISO-8601 calendar-date representation."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError("Canonical dates must be date instances, not datetimes")
    return value.isoformat()


def canonical_timestamp(value: datetime) -> str:
    """Return an aware timestamp normalized to UTC and terminated with ``Z``."""
    if not isinstance(value, datetime):
        raise TypeError("Canonical timestamps must be datetime instances")
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanonicalizationError("Canonical timestamps must be timezone-aware")

    utc_value = value.astimezone(UTC)
    fraction = ""
    if utc_value.microsecond:
        fraction = f".{utc_value.microsecond:06d}".rstrip("0")
    return (
        f"{utc_value.year:04d}-{utc_value.month:02d}-{utc_value.day:02d}"
        f"T{utc_value.hour:02d}:{utc_value.minute:02d}:{utc_value.second:02d}"
        f"{fraction}Z"
    )


def canonical_decimal(value: Decimal) -> str:
    """Return a finite decimal without exponent notation or redundant zeros."""
    if not isinstance(value, Decimal):
        raise TypeError("Canonical decimals must be Decimal instances")
    if not value.is_finite():
        raise CanonicalizationError("Canonical decimals must be finite")
    if value.is_zero():
        return "0"

    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def canonical_rational(value: Fraction | int, denominator: int | None = None) -> str:
    """Return a reduced, sign-normalized rational as ``numerator/denominator``."""
    if isinstance(value, Fraction):
        if denominator is not None:
            raise TypeError("A Fraction must not be paired with a denominator")
        rational = value
    else:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Rational numerators must be integers")
        if isinstance(denominator, bool) or not isinstance(denominator, int):
            raise TypeError("Rational denominators must be integers")
        if denominator == 0:
            raise CanonicalizationError(
                "Canonical rationals cannot have denominator zero"
            )
        rational = Fraction(value, denominator)
    return f"{rational.numerator}/{rational.denominator}"


def canonicalize(value: object) -> CanonicalJSONValue:
    """Convert supported domain primitives into canonical JSON-compatible values.

    Decimal and rational values intentionally become canonical strings so their
    precision and exactness do not depend on a JSON number parser. Finite binary
    floats remain JSON numbers for raw/provider data compatibility.
    """
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            return normalize_unicode(value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(
                "Canonical JSON does not allow non-finite floats"
            )
        return value
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, Fraction):
        return canonical_rational(value)
    if isinstance(value, datetime):
        return canonical_timestamp(value)
    if isinstance(value, date):
        return canonical_date(value)
    if isinstance(value, Mapping):
        normalized: dict[str, CanonicalJSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    "Canonical JSON object keys must be strings"
                )
            normalized_key = normalize_unicode(key)
            if normalized_key in normalized:
                raise CanonicalizationError(
                    "Canonical JSON object keys collide after Unicode NFC normalization"
                )
            normalized[normalized_key] = canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    raise CanonicalizationError(
        f"Unsupported canonical JSON value type: {type(value).__name__}"
    )


def canonical_json_text(value: object) -> str:
    """Encode *value* as whitespace-free, NFC-normalized canonical JSON plus LF."""
    return json.dumps(
        canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def canonical_json(value: object) -> bytes:
    """Encode *value* as UTF-8 canonical JSON terminated by exactly one LF."""
    return canonical_json_text(value).encode("utf-8")


canonical_json_bytes = canonical_json


def _coerce_bytes(data: ByteLike) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Checksums require bytes, bytearray, or memoryview input")
    return bytes(data)


def sha256_bytes(data: ByteLike) -> str:
    """Return the lowercase SHA-256 hexadecimal digest for *data*."""
    return hashlib.sha256(_coerce_bytes(data)).hexdigest()


def sha256_canonical_json(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON document."""
    return sha256_bytes(canonical_json(value))


canonical_checksum = sha256_canonical_json


def _validated_sha256(checksum: str) -> str:
    if not isinstance(checksum, str) or _SHA256_HEX.fullmatch(checksum) is None:
        raise CanonicalizationError(
            "SHA-256 checksums must be exactly 64 lowercase hexadecimal characters"
        )
    return checksum


@dataclass(frozen=True, slots=True)
class ChecksumVerifiedBytes:
    """Immutable bytes coupled to a verified lowercase SHA-256 checksum."""

    data: bytes
    checksum: str

    def __post_init__(self) -> None:
        materialized = _coerce_bytes(self.data)
        expected = _validated_sha256(self.checksum)
        actual = sha256_bytes(materialized)
        if actual != expected:
            raise ChecksumMismatchError(
                f"SHA-256 mismatch: expected {expected}, received {actual}"
            )
        object.__setattr__(self, "data", materialized)

    @classmethod
    def from_bytes(cls, data: ByteLike) -> ChecksumVerifiedBytes:
        """Create a wrapper after deriving the checksum from supplied bytes."""
        materialized = _coerce_bytes(data)
        return cls(data=materialized, checksum=sha256_bytes(materialized))

    @classmethod
    def verify(cls, data: ByteLike, expected_checksum: str) -> ChecksumVerifiedBytes:
        """Create a wrapper only when *data* verifies against *expected_checksum*."""
        return cls(data=_coerce_bytes(data), checksum=expected_checksum)


VerifiedBytes = ChecksumVerifiedBytes


def verify_checksum(data: ByteLike, expected_checksum: str) -> ChecksumVerifiedBytes:
    """Verify bytes and return their immutable checksum-verified wrapper."""
    return ChecksumVerifiedBytes.verify(data, expected_checksum)


def _project_value(value: object) -> object:
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    "Scientific-content object keys must be strings"
                )
            normalized_key = normalize_unicode(key)
            if normalized_key in OPERATIONAL_CONTENT_FIELDS:
                continue
            projected[normalized_key] = _project_value(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [_project_value(item) for item in value]
    return value


def project_scientific_content(
    manifest: Mapping[str, object],
) -> dict[str, CanonicalJSONValue]:
    """Return the scientific projection of a snapshot or run manifest.

    If the manifest has a ``content_identity`` mapping, only that mapping is
    eligible for the projection. Otherwise the supplied mapping is treated as
    an identity candidate. In either case known operational timestamps, IDs,
    paths, lineage, and metadata are explicitly excluded at every nesting level.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("Scientific-content projection requires a mapping")

    candidate = manifest.get("content_identity", manifest)
    if not isinstance(candidate, Mapping):
        raise CanonicalizationError("Manifest content_identity must be a mapping")
    projected = _project_value(candidate)
    canonical = canonicalize(projected)
    # Defensive: mappings always canonicalize to dictionaries.
    if not isinstance(canonical, dict):
        raise AssertionError("Scientific-content projection must be a JSON object")
    return canonical


def project_snapshot_content_identity(
    manifest: Mapping[str, object],
) -> dict[str, CanonicalJSONValue]:
    """Explicitly project a snapshot manifest to scientific content identity."""
    return project_scientific_content(manifest)


def project_run_content_identity(
    manifest: Mapping[str, object],
) -> dict[str, CanonicalJSONValue]:
    """Explicitly project a run manifest to scientific content identity."""
    return project_scientific_content(manifest)


def canonical_content_identity(manifest: Mapping[str, object]) -> bytes:
    """Return the canonical scientific-content bytes for a manifest."""
    return canonical_json(project_scientific_content(manifest))


def content_identity_checksum(manifest: Mapping[str, object]) -> str:
    """Return the SHA-256 checksum of a manifest's scientific projection."""
    return sha256_bytes(canonical_content_identity(manifest))
