"""Focused examples for market provenance and immutable snapshot manifest objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_research_platform.domain.canonical import project_snapshot_content_identity
from quant_research_platform.domain.errors import (
    ActionableError,
    ErrorCategory,
    LimitationDisclosure,
    ProviderFailureKind,
    ProviderFailureReason,
)
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotLineage,
    SnapshotManifest,
    VerifiedSnapshotHandle,
)
from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    DataGap,
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    ProviderRequestMetadata,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawDailyBar,
    SessionKey,
    SymbolOutcome,
    SymbolOutcomeStatus,
    SymbolValidationSummary,
    ValidationReport,
)

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_NOW = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)


def _record(*, symbol: str = "AAPL") -> ProviderRecord:
    request = ProviderRequest((symbol,), date(2024, 1, 2), date(2024, 1, 5))
    return ProviderRecord(
        provider="yfinance",
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=RawDailyBar(
            provider_date=date(2024, 1, 2),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            adj_close=Decimal("100.5"),
            volume=Decimal("1234"),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
            provider_fields={"source": "Yahoo Finance"},
        ),
        provider_fields={"exchange": "NMS", "extra": ["kept", 1]},
    )


def _candidate(record: ProviderRecord) -> DailyBarCandidate:
    action = CorporateAction(
        symbol=record.symbol,
        session=date(2024, 1, 2),
        raw_lineage=record.raw_lineage,
        source_fields=("Dividends", "Stock Splits"),
    )
    return DailyBarCandidate(
        symbol=record.symbol,
        session=date(2024, 1, 2),
        event_timestamp=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        raw_bar=record.raw_bar,
        raw_action=record.raw_action,
        corporate_action=action,
        adjusted_open=Decimal("100"),
        adjusted_high=Decimal("102"),
        adjusted_low=Decimal("99"),
        adjusted_close=Decimal("101"),
        adjusted_volume=Decimal("1234"),
        execution_adjusted_open=Decimal("100"),
        sizing_adjusted_close=Decimal("101"),
        cumulative_price_factor=Decimal("1"),
        cumulative_split_factor=Decimal("1"),
        policy_version="causal_forward_v1",
        raw_lineage=record.raw_lineage,
    )


def _report() -> ValidationReport:
    requested_range = DateRange(date(2024, 1, 2), date(2024, 1, 5))
    return ValidationReport(
        per_symbol=(
            SymbolValidationSummary(
                symbol="MSFT",
                accepted_count=0,
                quarantined_count=1,
                duplicate_count=0,
                gap_count=1,
                failed=True,
                retained_parent_coverage=True,
            ),
            SymbolValidationSummary(
                symbol="AAPL",
                accepted_count=1,
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
            ),
        ),
        quarantined_by_reason=(("validation.row", 1),),
        gaps=(
            DataGap(
                symbol="MSFT",
                expected_session=date(2024, 1, 2),
                requested_range=requested_range,
                parent_retained=True,
            ),
        ),
        calendar_version="exchange_calendars/4.5",
    )


def _manifest(
    *, created_at: datetime, local_path: str, parent: str | None = None
) -> SnapshotManifest:
    report = _report()
    identity = SnapshotContentIdentity(
        provider="yfinance",
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 5)),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
        configured_universe=("MSFT", "AAPL"),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "exchange_calendars/4.5", _E),
        configuration_checksum=_D,
        objects=(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.NORMALIZED,
                checksum=_B,
                relative_uri="objects/normalized/symbol=AAPL/year=2024/sha256=b.parquet",
                schema_version="daily_bar_v1",
                row_count=1,
                byte_size=321,
                symbol="AAPL",
                session_year=2024,
                media_type="application/vnd.apache.parquet",
            ),
            ContentAddressedObjectRef(
                object_kind=ObjectKind.RAW,
                checksum=_A,
                relative_uri="objects/raw/provider=yfinance/symbol=AAPL/year=2024/sha256=a.parquet",
                schema_version="raw_v1",
                row_count=1,
                byte_size=456,
                symbol="AAPL",
                session_year=2024,
                media_type="application/vnd.apache.parquet",
            ),
        ),
        validation_report_checksum=_C,
        validation_summary=report.summary,
        limitation_disclosure=LimitationDisclosure.current(),
    )
    return SnapshotManifest(
        content_identity=identity,
        operational_metadata=OperationalMetadata(
            created_at=created_at,
            job_id="job-local-1",
            local_manifest_path=local_path,
            notes={"stage": "published"},
        ),
        lineage=SnapshotLineage(parent_snapshot_id=parent)
        if parent
        else SnapshotLineage(),
    )


def test_provider_objects_are_immutable_and_preserve_canonical_raw_lineage() -> None:
    record = _record()
    request = ProviderRequest((" msft ", "aapl"), date(2024, 1, 2), date(2024, 1, 5))

    assert request.symbols == ("MSFT", "AAPL")
    assert (
        record.raw_lineage.provider_record_checksum == record.provider_record_checksum
    )
    assert record.sort_key() == (
        "AAPL",
        date(2024, 1, 2),
        record.provider_record_checksum,
    )
    assert record.provider_fields == {"exchange": "NMS", "extra": ("kept", 1)}
    with pytest.raises(FrozenInstanceError):
        record.symbol = "MSFT"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.provider_fields["exchange"] = "changed"  # type: ignore[index]


def test_provider_outcomes_are_complete_per_symbol_and_canonically_order_records() -> (
    None
):
    aapl = _record(symbol="AAPL")
    msft = _record(symbol="MSFT")
    request = ProviderRequest(("MSFT", "AAPL"), date(2024, 1, 2), date(2024, 1, 5))
    error = ActionableError(
        operation="provider.fetch",
        category=ErrorCategory.PROVIDER_TERMINAL,
        message="No provider record was returned for the requested range.",
        corrective_action="Verify the symbol and choose an available completed range.",
        symbol="MSFT",
    )

    result = ProviderBatchResult(
        request=request,
        outcomes=(
            SymbolOutcome(
                symbol="MSFT",
                status=SymbolOutcomeStatus.FAILURE,
                attempts=1,
                failure_kind=ProviderFailureKind.TERMINAL,
                failure_reason=ProviderFailureReason.EMPTY_RESPONSE,
                errors=(error,),
            ),
            SymbolOutcome(
                symbol="AAPL",
                status=SymbolOutcomeStatus.SUCCESS,
                attempts=1,
                records=(aapl,),
            ),
        ),
        operational_metadata=ProviderRequestMetadata(
            request_content_key=request.content_key,
            retrieved_at=_NOW,
            response_status="partially_succeeded",
        ),
    )

    assert result.status == "partially_succeeded"
    assert result.successful_records == (aapl,)
    with pytest.raises(ValueError, match="one result per requested symbol"):
        ProviderBatchResult(request=request, outcomes=(result.outcomes[0],))
    with pytest.raises(ValueError, match="successful outcomes"):
        SymbolOutcome(
            symbol="AAPL",
            status=SymbolOutcomeStatus.SUCCESS,
            attempts=1,
        )
    assert msft.symbol == "MSFT"  # Ensures fixture construction does not share symbols.


def test_normalized_candidates_preserve_raw_lineage_and_session_sort_key() -> None:
    record = _record()
    candidate = _candidate(record)

    assert candidate.session_key == SessionKey(" aapl ", date(2024, 1, 2))
    assert candidate.sort_key() == (
        "AAPL",
        date(2024, 1, 2),
        candidate.canonical_row_checksum,
    )
    assert candidate.raw_lineage == record.raw_lineage
    with pytest.raises(ValueError, match="must match candidate raw_lineage"):
        replace(
            candidate,
            corporate_action=CorporateAction(
                symbol="AAPL",
                session=date(2024, 1, 2),
                raw_lineage=_record(symbol="MSFT").raw_lineage,
            ),
        )


def test_quarantine_gap_and_report_keep_rejection_and_coverage_facts() -> None:
    record = _record()
    candidate = _candidate(record)
    quarantine = QuarantineRecord(
        source_kind=QuarantineSourceKind.DAILY_BAR_CANDIDATE,
        reason_codes=("ohlc.finite_positive", "high.envelope"),
        offending_values={"open": "0", "high": "-1"},
        symbol="aapl",
        session=date(2024, 1, 2),
        raw_lineage=record.raw_lineage,
        candidate_checksum=candidate.canonical_row_checksum,
        policy_version="causal_forward_v1",
    )
    report = _report()

    assert quarantine.primary_reason == "ohlc.finite_positive"
    assert report.per_symbol[0].symbol == "AAPL"
    assert report.summary.failed_symbols == ("MSFT",)
    assert report.summary.retained_parent_coverage_symbols == ("MSFT",)
    assert report.summary.gap_count == 1
    with pytest.raises(ValueError, match="at most one value per SessionKey"):
        ValidationReport(
            per_symbol=report.per_symbol,
            quarantined_by_reason=report.quarantined_by_reason,
            gaps=(report.gaps[0], report.gaps[0]),
        )


def test_snapshot_identity_excludes_operational_timestamps_paths_and_lineage() -> None:
    first = _manifest(
        created_at=datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
        local_path="/first-machine/data/snapshots",
    )
    second = _manifest(
        created_at=datetime(2025, 2, 3, 4, 5, tzinfo=UTC),
        local_path="/copied-machine/data/snapshots",
        parent=first.snapshot_id,
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.content_identity_checksum == second.content_identity_checksum
    assert first.manifest_checksum != second.manifest_checksum
    assert (
        project_snapshot_content_identity(second.to_manifest_dict())
        == first.to_content_identity_dict()
    )
    assert first.content_identity.objects[0].object_kind is ObjectKind.NORMALIZED
    assert first.content_identity.failed_symbols == ("MSFT",)
    assert first.content_identity.retained_parent_coverage_symbols == ("MSFT",)


def test_snapshot_refs_and_handles_reject_invalid_content_or_mutation() -> None:
    manifest = _manifest(created_at=_NOW, local_path="/tmp/snapshots")
    handle = VerifiedSnapshotHandle.from_manifest(manifest, verified_at=_NOW)

    assert handle.snapshot_id == manifest.snapshot_id
    assert handle.object_references == manifest.content_identity.objects
    with pytest.raises(FrozenInstanceError):
        handle.snapshot_id = "snap_" + _A  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-escaping"):
        ContentAddressedObjectRef(
            object_kind=ObjectKind.RAW,
            checksum=_A,
            relative_uri="../outside.parquet",
            schema_version="raw_v1",
            row_count=0,
            byte_size=0,
        )
    with pytest.raises(ValueError, match="covered_range must be contained"):
        SnapshotContentIdentity(
            provider="yfinance",
            requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 5)),
            covered_range=DateRange(date(2023, 12, 29), date(2024, 1, 2)),
            configured_universe=("AAPL",),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "v1", _E),
            configuration_checksum=_D,
            objects=(),
            validation_report_checksum=_C,
            validation_summary=_report().summary,
            limitation_disclosure=LimitationDisclosure.current(),
        )
