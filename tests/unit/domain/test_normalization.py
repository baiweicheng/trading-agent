"""Focused examples for causal corporate-action normalization."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from quant_research_platform.domain.market import (
    DailyBarCandidate,
    ProviderRecord,
    ProviderRequest,
    QuarantineRecord,
    RawCorporateAction,
    RawDailyBar,
)
from quant_research_platform.domain.normalization import (
    CausalForwardAdjustmentV1,
    Normalizer,
)


class FixtureCalendar:
    name = "XNYS"
    version = "fixture"

    def __init__(self, sessions: tuple[date, ...]) -> None:
        self._sessions = frozenset(sessions)

    def is_session(self, value: date) -> bool:
        return value in self._sessions

    def close_timestamp(self, session: date) -> datetime:
        if not self.is_session(session):
            raise ValueError("not a fixture session")
        return datetime.combine(session, datetime.min.time(), tzinfo=UTC).replace(
            hour=21
        )


def _record(
    session: date,
    *,
    open_: str | None = "100",
    high: str | None = "102",
    low: str | None = "99",
    close: str | None = "101",
    volume: str | None = "1000",
    adj_close: str | None = "100.5",
    dividend: str | None = "0",
    split_ratio: str | None = "1",
    provider_fields: dict[str, object] | None = None,
) -> ProviderRecord:
    request = ProviderRequest(("AAPL",), date(2024, 1, 2), date(2024, 1, 10))
    return ProviderRecord(
        provider="yfinance",
        request_content_key=request.content_key,
        symbol="AAPL",
        raw_bar=RawDailyBar(
            provider_date=session,
            open=Decimal(open_) if open_ is not None else None,
            high=Decimal(high) if high is not None else None,
            low=Decimal(low) if low is not None else None,
            close=Decimal(close) if close is not None else None,
            adj_close=Decimal(adj_close) if adj_close is not None else None,
            volume=Decimal(volume) if volume is not None else None,
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal(dividend) if dividend is not None else None,
            split_ratio=Decimal(split_ratio) if split_ratio is not None else None,
            provider_fields={"action_source": "fixture"},
        ),
        provider_fields=provider_fields or {"exchange": "NMS", "marker": "kept"},
    )


def _bars(values: list[object]) -> list[DailyBarCandidate]:
    return [value for value in values if isinstance(value, DailyBarCandidate)]


def _quarantines(values: list[object]) -> list[QuarantineRecord]:
    return [value for value in values if isinstance(value, QuarantineRecord)]


def test_no_action_preserves_raw_fields_and_uses_xnys_close_in_utc() -> None:
    session = date(2024, 1, 2)
    record = _record(session)
    result = list(Normalizer().normalize([record], FixtureCalendar((session,))))

    assert len(result) == 1
    candidate = _bars(result)[0]
    assert candidate.raw_bar == record.raw_bar
    assert candidate.raw_action == record.raw_action
    assert candidate.provider_adj_close == Decimal("100.5")
    assert candidate.event_timestamp == datetime(2024, 1, 2, 21, tzinfo=UTC)
    assert candidate.adjusted_open == Decimal("100")
    assert candidate.adjusted_high == Decimal("102")
    assert candidate.adjusted_low == Decimal("99")
    assert candidate.adjusted_close == Decimal("101")
    assert candidate.adjusted_volume == Decimal("1000")
    assert candidate.execution_adjusted_open == Decimal("100")
    assert candidate.sizing_adjusted_close == Decimal("101")
    assert candidate.cumulative_price_factor == Decimal("1")
    assert candidate.cumulative_split_factor == Decimal("1")
    assert candidate.policy_version == "causal_forward_v1"


def test_split_adjusts_prices_and_volume_but_not_as_of_execution_prices() -> None:
    first = date(2024, 1, 2)
    split_session = date(2024, 1, 3)
    records = [
        _record(first, close="100"),
        _record(
            split_session,
            open_="51",
            high="52",
            low="49",
            close="50",
            volume="200",
            split_ratio="2",
        ),
    ]
    result = list(
        Normalizer().normalize(records, FixtureCalendar((first, split_session)))
    )
    candidate = _bars(result)[1]

    assert candidate.adjusted_open == Decimal("102")
    assert candidate.adjusted_high == Decimal("104")
    assert candidate.adjusted_low == Decimal("98")
    assert candidate.adjusted_close == Decimal("100")
    assert candidate.adjusted_volume == Decimal("100")
    assert candidate.execution_adjusted_open == Decimal("51")
    assert candidate.sizing_adjusted_close == Decimal("50")
    assert candidate.cumulative_price_factor == Decimal("2")
    assert candidate.cumulative_split_factor == Decimal("2")
    assert candidate.corporate_action.split_ratio == Decimal("2")
    assert candidate.corporate_action.dividend == Decimal("0")


def test_dividend_uses_prior_close_and_does_not_change_volume() -> None:
    first = date(2024, 1, 2)
    dividend_session = date(2024, 1, 3)
    records = [
        _record(first, close="100"),
        _record(
            dividend_session,
            open_="99",
            high="100",
            low="98",
            close="99",
            dividend="1",
        ),
    ]
    result = list(
        Normalizer().normalize(records, FixtureCalendar((first, dividend_session)))
    )
    candidate = _bars(result)[1]

    assert candidate.cumulative_price_factor == Decimal(
        "1.010101010101010101"
    )
    assert candidate.adjusted_open == Decimal("99.99999999999999999999999999")
    assert candidate.adjusted_close == Decimal("99.99999999999999999999999999")
    assert candidate.adjusted_volume == Decimal("1000")
    assert candidate.corporate_action.dividend == Decimal("1")
    assert candidate.corporate_action.split_ratio == Decimal("1")


def test_same_session_split_is_applied_before_dividend() -> None:
    first = date(2024, 1, 2)
    action_session = date(2024, 1, 3)
    records = [
        _record(first, close="100"),
        _record(
            action_session,
            open_="49",
            high="50",
            low="48",
            close="49",
            volume="200",
            dividend="1",
            split_ratio="2",
        ),
    ]
    result = list(
        Normalizer().normalize(records, FixtureCalendar((first, action_session)))
    )
    candidate = _bars(result)[1]

    assert candidate.cumulative_price_factor == Decimal(
        "2.040816326530612245"
    )
    assert candidate.adjusted_close == Decimal("100")
    assert candidate.adjusted_volume == Decimal("100")
    assert candidate.corporate_action.source_fields == (
        "Dividends",
        "Stock Splits",
    )


def test_invalid_action_equations_are_quarantined_without_a_candidate() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    records = [
        _record(first, close="100"),
        _record(second, dividend="100", close="99"),
    ]
    result = list(Normalizer().normalize(records, FixtureCalendar((first, second))))
    quarantines = _quarantines(result)

    assert len(_bars(result)) == 1
    assert len(quarantines) == 1
    assert quarantines[0].primary_reason == "normalization.policy"
    assert quarantines[0].symbol == "AAPL"
    assert quarantines[0].session == second
    assert quarantines[0].policy_version == "causal_forward_v1"
    assert quarantines[0].offending_values["policy_reason"] == (
        "dividend_reference_non_positive"
    )


def test_non_session_is_quarantined_and_wholly_absent_observation_emits_no_bar(
) -> None:
    non_session = date(2024, 1, 1)
    missing = date(2024, 1, 2)
    present = date(2024, 1, 3)
    records = [
        _record(non_session),
        _record(
            missing,
            open_=None,
            high=None,
            low=None,
            close=None,
            volume=None,
            adj_close=None,
        ),
        _record(present),
    ]
    result = list(
        Normalizer().normalize(records, FixtureCalendar((missing, present)))
    )

    assert len(_bars(result)) == 1
    assert len(_quarantines(result)) == 1
    assert _quarantines(result)[0].primary_reason == "session.non_xnys"
    assert _quarantines(result)[0].session is None


def test_partial_observation_is_emitted_for_row_validation() -> None:
    session = date(2024, 1, 2)
    record = _record(session, high=None, low=None)
    result = list(Normalizer().normalize([record], FixtureCalendar((session,))))

    assert len(_quarantines(result)) == 0
    candidate = _bars(result)[0]
    assert candidate.raw_open == Decimal("100")
    assert candidate.raw_high is None
    assert candidate.raw_low is None
    assert candidate.adjusted_high is None
    assert candidate.adjusted_low is None


def test_later_actions_do_not_change_prior_candidates() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    later = date(2024, 1, 4)
    prefix = [_record(first, close="100"), _record(second, close="101")]
    with_later_action = [
        *prefix,
        _record(later, close="50", split_ratio="2"),
    ]
    calendar = FixtureCalendar((first, second, later))

    before = _bars(list(Normalizer().normalize(prefix, calendar)))
    after = _bars(list(Normalizer().normalize(with_later_action, calendar)))[:2]

    assert [candidate.to_content_dict() for candidate in before] == [
        candidate.to_content_dict() for candidate in after
    ]


def test_input_order_and_equivalent_duplicates_are_confluent() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    record_a = _record(first, close="100")
    record_b = _record(second, close="101", split_ratio="2")
    records = [record_b, record_a, record_a]
    calendar = FixtureCalendar((first, second))

    result = list(Normalizer().normalize(records, calendar))
    candidates = _bars(result)

    assert candidates[0] == candidates[1]
    assert candidates[2].session == second
    assert candidates[2].cumulative_price_factor == Decimal("2")
    assert [value.sort_key() for value in result] == sorted(
        value.sort_key() for value in result
    )


def test_non_positive_split_ratio_is_quarantined_without_a_candidate() -> None:
    session = date(2024, 1, 2)
    result = list(
        Normalizer().normalize(
            [_record(session, split_ratio="0")], FixtureCalendar((session,))
        )
    )

    assert len(_bars(result)) == 0
    assert len(_quarantines(result)) == 1
    quarantine = _quarantines(result)[0]
    assert quarantine.primary_reason == "normalization.policy"
    assert quarantine.offending_values["split_ratio"] == "0"
    assert quarantine.offending_values["policy_reason"] == (
        "split_ratio_non_positive_or_non_finite"
    )


def test_dividend_without_a_prior_raw_close_is_quarantined_without_a_candidate(
) -> None:
    session = date(2024, 1, 2)
    result = list(
        Normalizer().normalize(
            [_record(session, dividend="1")], FixtureCalendar((session,))
        )
    )

    assert len(_bars(result)) == 0
    assert len(_quarantines(result)) == 1
    quarantine = _quarantines(result)[0]
    assert quarantine.primary_reason == "normalization.policy"
    assert quarantine.offending_values["prior_raw_close"] is None
    assert quarantine.offending_values["policy_reason"] == (
        "dividend_missing_prior_close"
    )


def test_policy_declares_sources_equations_and_float64_boundary() -> None:
    policy = CausalForwardAdjustmentV1()

    declaration = policy.to_content_dict()
    assert declaration["policy_version"] == "causal_forward_v1"
    assert "raw_action.split_ratio" in declaration["source_fields"]
    assert "cumulative_price_factor" in declaration["equations"]
    assert declaration["rounding_treatment"]
    assert policy.round_float64(Decimal("0.1")) == 0.1
