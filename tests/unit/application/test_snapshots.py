"""Focused examples for snapshot identity and manifest assembly."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quant_research_platform.application.snapshots import SnapshotManifestAssembler
from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
)
from quant_research_platform.domain.market import (
    DateRange,
    ProviderRequestMetadata,
    SymbolValidationSummary,
    ValidationReport,
)

_A = "a" * 64
_B = "b" * 64


def _validation_report() -> ValidationReport:
    return ValidationReport(
        per_symbol=(
            SymbolValidationSummary(
                symbol="AAPL",
                accepted_count=2,
                quarantined_count=1,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            ),
            SymbolValidationSummary(
                symbol="SPY",
                accepted_count=2,
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            ),
        ),
        quarantined_by_reason=(("validation.row", 1),),
        gaps=(),
        calendar_version="exchange_calendars/4.5",
    )


def _objects() -> tuple[ContentAddressedObjectRef, ...]:
    return (
        ContentAddressedObjectRef(
            object_kind=ObjectKind.NORMALIZED,
            checksum=_B,
            relative_uri="objects/normalized/symbol=SPY/year=2024/sha256=b.parquet",
            schema_version="daily_bar_v1",
            row_count=2,
            byte_size=200,
            symbol="SPY",
            session_year=2024,
            media_type="application/vnd.apache.parquet",
        ),
        ContentAddressedObjectRef(
            object_kind=ObjectKind.RAW,
            checksum=_A,
            relative_uri="objects/raw/provider=yfinance/symbol=AAPL/year=2024/sha256=a.parquet",
            schema_version="raw_v1",
            row_count=2,
            byte_size=100,
            symbol="AAPL",
            session_year=2024,
            media_type="application/vnd.apache.parquet",
        ),
    )


def _assemble(
    *,
    created_at: datetime,
    local_path: str,
    parent_snapshot_id: str | None = None,
    operation_id: str | None = None,
    root_note: str = "first-root",
):
    report = _validation_report()
    request_key = "c" * 64
    return SnapshotManifestAssembler.assemble(
        provider="yfinance",
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        configured_universe=("aapl",),
        benchmark_symbol=" spy ",
        calendar=CalendarIdentity("XNYS", "exchange_calendars/4.5", "d" * 64),
        configuration_checksum="e" * 64,
        objects=reversed(_objects()),
        validation=report,
        limitation_disclosure=LimitationDisclosure.current(),
        created_at=created_at,
        provider_requests=(
            ProviderRequestMetadata(
                request_content_key=request_key,
                retrieved_at=created_at,
                response_status="succeeded",
                request_id="request-local",
            ),
        ),
        detection_times=(created_at,),
        job_id="job-local",
        local_manifest_path=local_path,
        notes={"root": root_note},
        parent_snapshot_id=parent_snapshot_id,
        operation_id=operation_id,
    )


def test_assembly_contains_required_identity_versions_checksums_and_counts() -> None:
    report = _validation_report()
    manifest = _assemble(
        created_at=datetime(2024, 1, 3, 21, tzinfo=UTC),
        local_path="/tmp/first/snapshots/manifest.json",
    )

    assert manifest.snapshot_id.startswith("snap_")
    assert len(manifest.snapshot_id) == 69
    identity = manifest.content_identity
    assert identity.provider == "yfinance"
    assert identity.configured_universe == ("AAPL",)
    assert identity.benchmark_symbol == "SPY"
    assert identity.requested_range == DateRange(date(2024, 1, 2), date(2024, 1, 3))
    assert identity.covered_range == identity.validation_summary.covered_range
    assert identity.schema_versions.corporate_action_policy_version == (
        "causal_forward_v1"
    )
    assert identity.calendar.schedule_checksum == "d" * 64
    assert identity.validation_report_checksum == report.content_checksum
    assert identity.validation_summary.accepted_row_count == 4
    assert identity.validation_summary.quarantined_row_count == 1
    assert [reference.checksum for reference in identity.objects] == [_B, _A]
    assert [reference.row_count for reference in identity.objects] == [2, 2]
    assert manifest.to_manifest_dict()["operational_metadata"][
        "local_manifest_path"
    ] == ("/tmp/first/snapshots/manifest.json")


def test_volatile_metadata_relocation_and_attempted_parent_do_not_change_snapshot_id() -> (
    None
):
    first = _assemble(
        created_at=datetime(2024, 1, 3, 21, tzinfo=UTC),
        local_path="/tmp/first/snapshots/manifest.json",
        operation_id="operation-first",
    )
    relocated = _assemble(
        created_at=datetime(2025, 2, 4, 22, tzinfo=UTC),
        local_path="/Users/other/data/snapshots/manifest.json",
        parent_snapshot_id=first.snapshot_id,
        operation_id="operation-second",
        root_note="relocated-root",
    )

    assert first.snapshot_id == relocated.snapshot_id
    assert first.content_identity_checksum == relocated.content_identity_checksum
    assert first.manifest_checksum != relocated.manifest_checksum
    assert relocated.lineage.parent_snapshot_id == first.snapshot_id
    assert relocated.operational_metadata.local_manifest_path != (
        first.operational_metadata.local_manifest_path
    )


def test_assembly_rejects_duplicate_partition_references_and_repeated_validation_artifact() -> (
    None
):
    report = _validation_report()
    duplicate_uri = _objects()[0]
    with pytest.raises(ValueError, match="logical partition URI exactly once"):
        SnapshotManifestAssembler.assemble(
            provider="yfinance",
            requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            configured_universe=("AAPL",),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "v1", "f" * 64),
            configuration_checksum="1" * 64,
            objects=(duplicate_uri, duplicate_uri),
            validation=report,
            validation_report_checksum="2" * 64,
            created_at=datetime(2024, 1, 3, tzinfo=UTC),
        )

    validation_ref = ContentAddressedObjectRef(
        object_kind=ObjectKind.VALIDATION,
        checksum=report.content_checksum,
        relative_uri="objects/validation/sha256=report.parquet",
        schema_version="validation_report_v1",
        row_count=1,
        byte_size=10,
    )
    with pytest.raises(ValueError, match="referenced exactly once"):
        SnapshotManifestAssembler.assemble(
            provider="yfinance",
            requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            configured_universe=("AAPL",),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "v1", "f" * 64),
            configuration_checksum="1" * 64,
            objects=(validation_ref,),
            validation=report,
            created_at=datetime(2024, 1, 3, tzinfo=UTC),
        )


def test_snapshot_assembly_requires_a_physical_report_checksum_for_compact_summary() -> (
    None
):
    with pytest.raises(TypeError, match="validation_report_checksum is required"):
        SnapshotManifestAssembler.assemble(
            provider="yfinance",
            requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            configured_universe=("AAPL",),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "v1", "f" * 64),
            configuration_checksum="1" * 64,
            validation_summary=_validation_report().summary,
            created_at=datetime(2024, 1, 3, tzinfo=UTC),
        )


def test_assembly_accepts_explicit_operational_metadata_without_leaking_it_into_identity() -> (
    None
):
    manifest = SnapshotManifestAssembler.assemble(
        provider="yfinance",
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "v1", "f" * 64),
        configuration_checksum="1" * 64,
        objects=(),
        validation=_validation_report(),
        operational_metadata=OperationalMetadata(
            created_at=datetime(2024, 1, 3, tzinfo=UTC),
            local_manifest_path=str(Path("/relocated/manifest.json")),
        ),
    )

    assert "operational_metadata" not in manifest.to_content_identity_dict()
    assert "lineage" not in manifest.to_content_identity_dict()
