"""Focused examples for deterministic validation partitioning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    DateRange,
    ProviderRecord,
    ProviderRequest,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawDailyBar,
)
from quant_research_platform.domain.validation import ValidationService


class FixtureCalendar:
    name = "XNYS"
    version = "fixture-xnys-v1"

    def __init__(self, sessions: tuple[date, ...]) -> None:
        self._sessions = frozenset(sessions)

    def is_session(self, value: date) -> bool:
        return value in self._sessions


def _candidate(
    symbol: str = "AAPL",
    session: date = date(2024, 1, 2),
    *,
    open_: str = "100",
    high: str = "102",
    low: str = "99",
    close: str = "101",
    volume: str = "1000",
) -> DailyBarCandidate:
    request = ProviderRequest((symbol,), date(2024, 1, 2), date(2024, 1, 10))
    record = ProviderRecord(
        provider="yfinance",
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=RawDailyBar(
            provider_date=session,
            open=Decimal(open_),
            high=Decimal(high),
            low=Decimal(low),
            close=Decimal(close),
            adj_close=Decimal(close),
            volume=Decimal(volume),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
        ),
    )
    action = CorporateAction(
        symbol=symbol,
        session=session,
        raw_lineage=record.raw_lineage,
    )
    return DailyBarCandidate(
        symbol=symbol,
        session=session,
        event_timestamp=datetime(2024, 1, 2, 21, tzinfo=UTC),
        raw_bar=record.raw_bar,
        raw_action=record.raw_action,
        corporate_action=action,
        adjusted_open=Decimal(open_),
        adjusted_high=Decimal(high),
        adjusted_low=Decimal(low),
        adjusted_close=Decimal(close),
        adjusted_volume=Decimal(volume),
        execution_adjusted_open=Decimal(open_),
        sizing_adjusted_close=Decimal(close),
        cumulative_price_factor=Decimal("1"),
        cumulative_split_factor=Decimal("1"),
        policy_version="causal_forward_v1",
        raw_lineage=record.raw_lineage,
    )


def _expected(*sessions: date, symbol: str = "AAPL") -> dict[str, tuple[date, ...]]:
    return {symbol: sessions}


def test_all_row_and_envelope_rules_are_reported_in_fixed_order() -> None:
    session = date(2024, 1, 2)
    candidate = _candidate(
        session=session,
        open_="0",
        high="-1",
        low="2",
        close="0",
        volume="-1",
    )
    output = ValidationService(calendar=FixtureCalendar((session,))).validate(
        [candidate], _expected(session), 1
    )

    assert output.accepted == ()
    assert len(output.quarantined) == 1
    assert output.quarantined[0].reason_codes == (
        "ohlc.finite_positive",
        "volume.finite_nonnegative",
        "high.envelope",
        "low.envelope",
    )
    assert output.quarantined[0].offending_values["open"] == "0"
    assert output.quarantined[0].offending_values["volume"] == "-1"
    assert output.report.summary.quarantined_row_count == 1
    assert output.report.summary.gap_count == 1


def test_map_policy_and_lineage_rules_are_quarantined() -> None:
    session = date(2024, 1, 2)
    symbol_invalid = _candidate(session=session)
    object.__setattr__(symbol_invalid, "symbol", "")
    session_invalid = _candidate(session=date(2024, 1, 3))
    policy_invalid = replace(_candidate(session=session), policy_version="other-v1")
    lineage_invalid = _candidate(session=session)
    object.__setattr__(lineage_invalid, "raw_lineage", None)

    output = ValidationService(calendar=FixtureCalendar((session,))).validate(
        [symbol_invalid, session_invalid, policy_invalid, lineage_invalid],
        _expected(session),
        1,
    )

    reasons = [record.reason_codes for record in output.quarantined]
    assert ("symbol.nonempty",) in reasons
    assert ("session.xnys",) in reasons
    assert any(
        record.reason_codes[0] == "normalization.policy"
        for record in output.quarantined
    )
    assert any(
        record.reason_codes[0] == "lineage.raw_record" for record in output.quarantined
    )
    assert len(output.accepted) == 0


def test_equivalent_duplicates_collapse_and_conflicts_quarantine_every_member() -> None:
    session = date(2024, 1, 2)
    candidate = _candidate(session=session)
    equivalent = ValidationService(calendar=FixtureCalendar((session,))).validate(
        [candidate, candidate], _expected(session), 1
    )
    assert equivalent.accepted == (candidate,)
    assert equivalent.duplicate_count == 1
    assert equivalent.duplicate_counts[0][0].symbol == "AAPL"
    assert equivalent.report.summary.quarantined_row_count == 0

    conflicting = replace(
        candidate,
        adjusted_high=Decimal("104"),
        adjusted_close=Decimal("103"),
    )
    conflict_output = ValidationService(calendar=FixtureCalendar((session,))).validate(
        [candidate, conflicting], _expected(session), 1
    )

    assert conflict_output.accepted == ()
    assert len(conflict_output.quarantined) == 2
    assert all(
        record.reason_codes == ("duplicate.conflict",)
        for record in conflict_output.quarantined
    )
    assert conflict_output.report.summary.gap_count == 1
    assert conflict_output.report.summary.collapsed_duplicate_count == 0


def test_gaps_are_explicit_and_do_not_fabricate_bars() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    output = ValidationService(calendar=FixtureCalendar((first, second))).validate(
        [_candidate(session=first)], _expected(first, second), 1
    )

    assert [gap.expected_session for gap in output.gaps] == [second]
    assert output.gaps[0].requested_range == DateRange(first, second)
    assert [row.session for row in output.accepted] == [first]
    assert output.report.summary.gap_count == len(output.gaps) == 1


def test_staleness_uses_completed_session_count_and_threshold_boundary() -> None:
    sessions = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    )
    calendar = FixtureCalendar(sessions)
    not_stale = ValidationService(calendar=calendar).validate(
        [_candidate(session=sessions[0])], _expected(*sessions), 2
    )
    stale = ValidationService(calendar=calendar).validate(
        [_candidate(session=sessions[0])], _expected(*sessions), 1
    )

    assert not_stale.per_symbol[0].stale is False
    assert not_stale.per_symbol[0].staleness_lag_sessions == 0
    assert stale.per_symbol[0].stale is True
    assert stale.per_symbol[0].staleness_lag_sessions == 2


def test_spy_comparison_readiness_is_specific_to_the_evaluation_range() -> None:
    sessions = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    )
    expected = {"SPY": sessions}
    calendar = FixtureCalendar(sessions)
    candidate = _candidate(symbol="SPY", session=sessions[0])

    ready = ValidationService(calendar=calendar).validate(
        [candidate],
        expected,
        1,
        comparison_range=DateRange(sessions[0], sessions[0]),
    )
    not_ready = ValidationService(calendar=calendar).validate(
        [candidate],
        expected,
        1,
        comparison_range=DateRange(sessions[1], sessions[1]),
    )

    assert ready.per_symbol[0].comparison_ready is True
    assert ready.comparison_ready is True
    assert not_ready.per_symbol[0].comparison_ready is False
    assert not_ready.comparison_ready is False


def test_report_counts_and_content_identity_are_reconciled_and_deterministic() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    valid = _candidate(session=first)
    invalid = replace(
        _candidate(session=second),
        adjusted_open=Decimal("0"),
        adjusted_low=Decimal("-1"),
        adjusted_close=Decimal("0"),
    )
    incoming = QuarantineRecord(
        source_kind=QuarantineSourceKind.PROVIDER_RECORD,
        reason_codes=("session.non_xnys",),
        offending_values={"symbol": "AAPL", "provider_date": date(2024, 1, 1)},
    )
    calendar = FixtureCalendar((first, second))
    service = ValidationService(calendar=calendar)
    output_one = service.validate(
        [invalid, incoming, valid], _expected(first, second), 1
    )
    output_two = service.validate(
        [valid, incoming, invalid], _expected(first, second), 1
    )

    assert output_one.report.content_checksum == output_two.report.content_checksum
    assert output_one.report.to_content_dict() == output_two.report.to_content_dict()
    assert output_one.report.summary.accepted_row_count == len(output_one.accepted)
    assert output_one.report.summary.quarantined_row_count == len(
        output_one.quarantined
    )
    assert output_one.report.summary.gap_count == len(output_one.gaps)
    assert sum(summary.quarantined_count for summary in output_one.per_symbol) == len(
        output_one.quarantined
    )
    assert output_one.report.quarantined_by_reason == (
        ("ohlc.finite_positive", 1),
        ("session.non_xnys", 1),
    )


def test_sorted_streams_are_consumed_without_materializing_the_input() -> None:
    session = date(2024, 1, 2)
    candidate = _candidate(session=session)

    def stream() -> Iterator[DailyBarCandidate]:
        yield candidate

    output = ValidationService(calendar=FixtureCalendar((session,))).validate(
        stream(), _expected(session), 1
    )

    assert output.accepted == (candidate,)
