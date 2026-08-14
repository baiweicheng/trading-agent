"""Focused tests for canonical scientific-content primitives."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import pytest

from quant_research_platform.domain.canonical import (
    CanonicalizationError,
    ChecksumMismatchError,
    ChecksumVerifiedBytes,
    canonical_content_identity,
    canonical_date,
    canonical_decimal,
    canonical_json,
    canonical_rational,
    canonical_timestamp,
    content_identity_checksum,
    project_snapshot_content_identity,
    sha256_bytes,
)


def test_canonical_json_is_stable_nfc_utf8_and_key_ordered() -> None:
    first = {"z": "e\u0301", "a": {"beta": 2, "alpha": 1}}
    second = {"a": {"alpha": 1, "beta": 2}, "z": "é"}

    expected = b'{"a":{"alpha":1,"beta":2},"z":"\xc3\xa9"}\n'

    assert canonical_json(first) == expected
    assert canonical_json(first) == canonical_json(second)


def test_canonical_json_has_one_terminal_lf() -> None:
    encoded = canonical_json({"line": "first\nsecond"})

    assert encoded.endswith(b"\n")
    assert not encoded.endswith(b"\n\n")
    assert encoded.count(b"\n") == 1


def test_canonical_scalars_normalize_dates_timestamps_decimals_and_rationals() -> None:
    offset = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime(2024, 2, 29, 5, 30, 0, 120_000, tzinfo=offset)

    assert canonical_date(date(2024, 2, 29)) == "2024-02-29"
    assert canonical_timestamp(timestamp) == "2024-02-29T00:00:00.12Z"
    assert canonical_decimal(Decimal("-0.000")) == "0"
    assert canonical_decimal(Decimal("1200.3400")) == "1200.34"
    assert canonical_rational(Fraction(-6, -8)) == "3/4"
    assert canonical_rational(6, -8) == "-3/4"


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity")],
)
def test_canonical_json_rejects_non_finite_numbers(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json({"value": value})


def test_checksum_verified_bytes_rejects_tampering() -> None:
    payload = b"scientific-content\n"
    checksum = sha256_bytes(payload)

    verified = ChecksumVerifiedBytes.verify(payload, checksum)

    assert verified.data == payload
    assert verified.checksum == checksum
    with pytest.raises(ChecksumMismatchError):
        ChecksumVerifiedBytes.verify(b"operational-content\n", checksum)


def test_content_projection_keeps_science_excludes_operations() -> None:
    initial_manifest = {
        "content_identity": {
            "schema_version": "snapshot_v1",
            "objects": [{"checksum": "a" * 64, "row_count": 3}],
            "created_at": "2024-01-01T00:00:00Z",
            "run_id": "run-one",
            "local_path": "/tmp/first/snapshot",
            "lineage": {"parent_snapshot_id": "snap-parent"},
        },
        "operational_metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "job_id": "job-one",
        },
    }
    changed_operational_manifest = {
        "content_identity": {
            "schema_version": "snapshot_v1",
            "objects": [{"checksum": "a" * 64, "row_count": 3}],
            "created_at": "2025-12-31T23:59:59Z",
            "run_id": "run-two",
            "local_path": "/other-machine/snapshot",
            "lineage": {"parent_snapshot_id": "snap-other-parent"},
        },
        "operational_metadata": {
            "created_at": "2025-12-31T23:59:59Z",
            "job_id": "job-two",
        },
    }
    changed_scientific_manifest = {
        "content_identity": {
            "schema_version": "snapshot_v1",
            "objects": [{"checksum": "b" * 64, "row_count": 3}],
        }
    }

    projection = project_snapshot_content_identity(initial_manifest)

    assert projection == {
        "objects": [{"checksum": "a" * 64, "row_count": 3}],
        "schema_version": "snapshot_v1",
    }
    assert canonical_content_identity(initial_manifest) == canonical_content_identity(
        changed_operational_manifest
    )
    assert content_identity_checksum(initial_manifest) == content_identity_checksum(
        changed_operational_manifest
    )
    assert content_identity_checksum(initial_manifest) != content_identity_checksum(
        changed_scientific_manifest
    )
