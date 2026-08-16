"""Focused tests for explicit Arrow schemas and canonical conversion."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pyarrow as pa
import pytest

from quant_research_platform.domain.execution import (
    DailyReturn,
    FillRecord,
    OrderRecord,
    PortfolioState,
    Position,
    deterministic_fill_id,
    deterministic_order_id,
)
from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    DataGap,
    DateRange,
    ProviderRecord,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawDailyBar,
    RawLineage,
    SymbolValidationSummary,
    ValidationReport,
)
from quant_research_platform.domain.strategy import RationalWeight, StrategyDecision
from quant_research_platform.infrastructure.schemas import (
    DAILY_BAR_V1,
    DECISIONS_V1,
    GAP_V1,
    METRICS_V1,
    MONTHLY_RETURN_V1,
    ORDERS_V1,
    PORTFOLIO_V1,
    POSITIONS_V1,
    QUARANTINE_V1,
    RAW_V1,
    RETURNS_V1,
    VALIDATION_REPORT_V1,
    canonical_rows,
    canonical_table,
    daily_bars_to_table,
    decisions_to_table,
    fills_to_table,
    gaps_to_table,
    monthly_returns_to_table,
    orders_to_table,
    portfolio_states_to_table,
    positions_to_table,
    quarantines_to_table,
    raw_records_to_table,
    returns_to_table,
    validate_canonical_table,
    validation_reports_to_table,
)

_CHECKSUM = "a" * 64


def _provider_record() -> ProviderRecord:
    return ProviderRecord(
        provider="yfinance",
        request_content_key=_CHECKSUM,
        symbol="AAPL",
        raw_bar=RawDailyBar(
            provider_date=date(2024, 1, 2),
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            adj_close=Decimal("100.5"),
            volume=Decimal("1000"),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
            provider_fields={"provider_note": "fixture"},
        ),
        provider_fields={"currency": "USD"},
    )


def _daily_bar() -> DailyBarCandidate:
    raw_record = _provider_record()
    lineage = RawLineage(
        provider=raw_record.provider,
        request_content_key=raw_record.request_content_key,
        provider_record_checksum=raw_record.provider_record_checksum,
    )
    return DailyBarCandidate(
        symbol="AAPL",
        session=date(2024, 1, 2),
        event_timestamp=datetime(2024, 1, 2, 21, tzinfo=UTC),
        raw_bar=raw_record.raw_bar,
        raw_action=raw_record.raw_action,
        corporate_action=CorporateAction(
            symbol="AAPL",
            session=date(2024, 1, 2),
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
            raw_lineage=lineage,
        ),
        adjusted_open=Decimal("100"),
        adjusted_high=Decimal("102"),
        adjusted_low=Decimal("99"),
        adjusted_close=Decimal("101"),
        adjusted_volume=Decimal("1000"),
        execution_adjusted_open=Decimal("100"),
        sizing_adjusted_close=Decimal("101"),
        cumulative_price_factor=Decimal("1"),
        cumulative_split_factor=Decimal("1"),
        policy_version="causal_forward_v1",
        raw_lineage=lineage,
    )


def _order_and_fill() -> tuple[OrderRecord, FillRecord]:
    order_id = deterministic_order_id(
        signal_session=date(2024, 1, 2),
        execution_session=date(2024, 1, 3),
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
    )
    order = OrderRecord(
        order_id=order_id,
        signal_session=date(2024, 1, 2),
        execution_session=date(2024, 1, 3),
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
        status="filled",
    )
    fill = FillRecord(
        fill_id=deterministic_fill_id(
            order_id=order_id,
            symbol="AAPL",
            session=date(2024, 1, 3),
            quantity=10,
            ordinal=0,
        ),
        order_id=order_id,
        symbol="AAPL",
        session=date(2024, 1, 3),
        quantity=10,
        ordinal=0,
        base_adjusted_open=Decimal("100"),
        fill_price=Decimal("100.1"),
        gross_notional=Decimal("1001"),
        commission=Decimal("1"),
        slippage_cost=Decimal("1"),
    )
    return order, fill


def _portfolio_state() -> PortfolioState:
    position = Position(
        symbol="AAPL",
        quantity=10,
        mark_price=Decimal("101"),
        market_value=Decimal("1010"),
    )
    return PortfolioState(
        session=date(2024, 1, 3),
        cash_balance=Decimal("98990"),
        positions=(position,),
        gross_exposure=Decimal("1010"),
        portfolio_equity=Decimal("100000"),
        leverage=Decimal("0.0101"),
    )


def _report() -> ValidationReport:
    return ValidationReport(
        per_symbol=(
            SymbolValidationSummary(
                symbol="AAPL",
                accepted_count=1,
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
            ),
        ),
        quarantined_by_reason=(),
        gaps=(),
        calendar_version="xnys-fixture",
    )


def _daily_bar_row() -> dict[str, object]:
    return canonical_rows(daily_bars_to_table([_daily_bar()]), DAILY_BAR_V1)[0]


def test_representative_records_round_trip_through_every_canonical_table() -> None:
    raw = _provider_record()
    bar = _daily_bar()
    lineage = bar.raw_lineage
    quarantine = QuarantineRecord(
        source_kind=QuarantineSourceKind.DAILY_BAR_CANDIDATE,
        reason_codes=("ohlc.finite_positive",),
        offending_values={"raw_close": "0"},
        policy_version="causal_forward_v1",
        symbol="AAPL",
        session=date(2024, 1, 2),
        raw_lineage=lineage,
        candidate_checksum=bar.canonical_row_checksum,
    )
    gap = DataGap(
        symbol="AAPL",
        expected_session=date(2024, 1, 3),
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
    )
    decision = StrategyDecision(
        signal_session=date(2024, 1, 2),
        symbol="AAPL",
        endpoint_252_session=date(2023, 1, 2),
        endpoint_252_close=Decimal("80"),
        endpoint_21_session=date(2023, 12, 1),
        endpoint_21_close=Decimal("90"),
        momentum_score=Decimal("0.125"),
        eligible=True,
        rank=1,
        target_weight=RationalWeight(1, 1),
        exclusion_reason=None,
    )
    order, fill = _order_and_fill()
    state = _portfolio_state()

    tables_and_schemas = (
        (raw_records_to_table([raw]), RAW_V1),
        (daily_bars_to_table([bar]), DAILY_BAR_V1),
        (quarantines_to_table([quarantine]), QUARANTINE_V1),
        (gaps_to_table([gap]), GAP_V1),
        (validation_reports_to_table([_report()]), VALIDATION_REPORT_V1),
        (decisions_to_table([decision]), DECISIONS_V1),
        (orders_to_table([order]), ORDERS_V1),
        (fills_to_table([fill]), "fills_v1"),
        (positions_to_table([state]), POSITIONS_V1),
        (portfolio_states_to_table([state]), PORTFOLIO_V1),
        (
            returns_to_table([DailyReturn(date(2024, 1, 3), Decimal("0.01"))]),
            RETURNS_V1,
        ),
        (
            canonical_table(
                METRICS_V1,
                [
                    {
                        "scope": "strategy",
                        "name": "total_return",
                        "value_decimal": Decimal("0.01"),
                        "value_integer": None,
                        "null_reason": None,
                    }
                ],
            ),
            METRICS_V1,
        ),
        (
            monthly_returns_to_table(
                [
                    {
                        "month": date(2024, 1, 1),
                        "scope": "strategy",
                        "return_value": Decimal("0.01"),
                    }
                ]
            ),
            MONTHLY_RETURN_V1,
        ),
    )

    for table, schema_name in tables_and_schemas:
        validate_canonical_table(table, schema_name)
        assert b"pandas" not in (table.schema.metadata or {})
        assert canonical_rows(table, schema_name) == canonical_rows(table, schema_name)

    raw_row = canonical_rows(raw_records_to_table([raw]), RAW_V1)[0]
    assert raw_row["provider_record_checksum"] == bytes.fromhex(
        raw.provider_record_checksum
    )
    assert canonical_rows(daily_bars_to_table([bar]), DAILY_BAR_V1)[0]["event_ts"] == (
        datetime(2024, 1, 2, 21, tzinfo=UTC)
    )
    assert len(canonical_rows(positions_to_table([state]), POSITIONS_V1)) == 2


def test_canonical_conversion_rejects_missing_non_null_and_non_utc_timestamps() -> None:
    row = _daily_bar_row()
    missing_symbol = dict(row)
    del missing_symbol["symbol"]
    with pytest.raises(ValueError, match="missing required non-null field: symbol"):
        canonical_table(DAILY_BAR_V1, [missing_symbol])

    non_utc = dict(row)
    non_utc["event_ts"] = datetime(2024, 1, 2, 22, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="must use UTC"):
        canonical_table(DAILY_BAR_V1, [non_utc])


def test_schema_validation_rejects_incompatible_versions_and_pandas_metadata() -> None:
    table = raw_records_to_table([_provider_record()])
    wrong_version = table.replace_schema_metadata(
        {b"qrp.schema_name": b"raw_v0", b"qrp.schema_version": b"raw_v0"}
    )
    with pytest.raises(ValueError, match="incompatible or missing schema name"):
        validate_canonical_table(wrong_version, RAW_V1)

    pandas_metadata = table.replace_schema_metadata(
        {
            b"qrp.schema_name": b"raw_v1",
            b"qrp.schema_version": b"raw_v1",
            b"pandas": b"{}",
        }
    )
    with pytest.raises(ValueError, match="pandas metadata"):
        validate_canonical_table(pandas_metadata, RAW_V1)


def test_canonical_conversion_rejects_noncanonical_enums_and_decimals() -> None:
    order, _ = _order_and_fill()
    order_row = order.to_serializable()
    order_row["status"] = "FILLED"
    with pytest.raises(ValueError, match="status must be one of"):
        canonical_table(ORDERS_V1, [order_row])

    portfolio_row = {
        "session": date(2024, 1, 3),
        "cash_balance": "100000.000000",
        "gross_exposure": Decimal("0"),
        "leverage": Decimal("0"),
        "portfolio_equity": Decimal("100000"),
    }
    with pytest.raises(TypeError, match="cash_balance must be a Decimal"):
        canonical_table(PORTFOLIO_V1, [portfolio_row])

    excessive_scale = dict(portfolio_row)
    excessive_scale["cash_balance"] = Decimal("100000.0000001")
    with pytest.raises(ValueError, match="canonical decimal scale"):
        canonical_table(PORTFOLIO_V1, [excessive_scale])


def test_canonical_table_sorts_permuted_rows_without_pandas_round_trip() -> None:
    later = {
        "month": date(2024, 2, 1),
        "scope": "strategy",
        "return_value": Decimal("0.02"),
    }
    earlier = {
        "month": date(2024, 1, 1),
        "scope": "strategy",
        "return_value": Decimal("0.01"),
    }
    table = canonical_table(MONTHLY_RETURN_V1, [later, earlier])

    assert [row["month"] for row in canonical_rows(table, MONTHLY_RETURN_V1)] == [
        date(2024, 1, 1),
        date(2024, 2, 1),
    ]
    assert isinstance(table, pa.Table)
