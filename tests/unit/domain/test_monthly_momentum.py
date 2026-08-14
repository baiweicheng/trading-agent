"""Focused examples for the deterministic monthly momentum policy."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from quant_research_platform.domain.strategy import (
    PriceHistory,
    PriceObservation,
    RationalWeight,
    StrategyExclusionReason,
    monthly_momentum_v1,
)


def _sessions(count: int, *, start: date = date(2020, 1, 2)) -> tuple[date, ...]:
    return tuple(start + timedelta(days=index) for index in range(count))


def _history(
    sessions: tuple[date, ...],
    values: dict[str, tuple[Decimal, Decimal]],
    *,
    universe: tuple[str, ...],
    tradable: dict[tuple[str, date], bool] | None = None,
    omitted: set[tuple[str, date]] | None = None,
) -> PriceHistory:
    observations: list[PriceObservation] = []
    omitted = omitted or set()
    tradable = tradable or {}
    for symbol in universe:
        long_value, short_value = values[symbol]
        for index, session in enumerate(sessions):
            if (symbol, session) in omitted:
                continue
            if index == 1:
                close = long_value
            elif index == len(sessions) - 22:
                close = short_value
            else:
                close = Decimal("100")
            observations.append(
                PriceObservation(
                    symbol=symbol,
                    session=session,
                    adjusted_close=close,
                    tradable=tradable.get((symbol, session), True),
                )
            )
    return PriceHistory(observations=tuple(observations), sessions=sessions, universe=universe)


def test_month_end_sessions_use_the_last_available_session_of_each_month() -> None:
    sessions = (
        date(2024, 3, 27),
        date(2024, 3, 28),  # Friday 29th is intentionally absent (holiday fixture).
        date(2024, 4, 1),
    )
    history = PriceHistory(
        observations=tuple(
            PriceObservation("AAPL", session, Decimal("100"))
            for session in sessions
        ),
        sessions=sessions,
        universe=("AAPL",),
    )

    decisions = monthly_momentum_v1(history, position_count=1)

    assert tuple(decision.signal_session for decision in decisions) == (
        date(2024, 3, 28),
        date(2024, 4, 1),
    )
    assert all(
        decision.exclusion_reason is StrategyExclusionReason.WARM_UP_INCOMPLETE
        for decision in decisions
    )


def test_exact_253_session_warmup_boundary_allows_first_score() -> None:
    warmup_sessions = _sessions(253)
    eligible_sessions = _sessions(254)
    universe = ("AAPL",)
    warmup_history = _history(
        warmup_sessions,
        {"AAPL": (Decimal("100"), Decimal("120"))},
        universe=universe,
    )
    eligible_history = _history(
        eligible_sessions,
        {"AAPL": (Decimal("100"), Decimal("120"))},
        universe=universe,
    )

    before_boundary = monthly_momentum_v1(
        warmup_history,
        signal_session=warmup_sessions[-1],
        universe=universe,
        position_count=1,
    )[0]
    at_boundary = monthly_momentum_v1(
        eligible_history,
        signal_session=eligible_sessions[-1],
        universe=universe,
        position_count=1,
    )[0]

    assert before_boundary.eligible is False
    assert before_boundary.exclusion_reason is StrategyExclusionReason.WARM_UP_INCOMPLETE
    assert at_boundary.eligible is True
    assert at_boundary.momentum_score == Decimal("0.2")
    assert at_boundary.target_weight == RationalWeight(1, 1)
    assert at_boundary.endpoint_252_checksum is not None
    assert at_boundary.endpoint_21_checksum is not None


def test_endpoint_and_asset_status_exclusions_are_explicit() -> None:
    sessions = _sessions(254)
    universe = ("AAPL", "MSFT", "PG", "XOM")
    omitted = {
        ("AAPL", sessions[1]),
        ("MSFT", sessions[-22]),
    }
    tradable = {("PG", sessions[-1]): False}
    history = _history(
        sessions,
        {symbol: (Decimal("100"), Decimal("110")) for symbol in universe},
        universe=universe,
        omitted=omitted,
        tradable=tradable,
    )

    decisions = monthly_momentum_v1(
        history,
        signal_session=sessions[-1],
        universe=universe,
        position_count=1,
    )
    by_symbol = {decision.symbol: decision for decision in decisions}

    assert by_symbol["AAPL"].exclusion_reason is StrategyExclusionReason.MISSING_LONG_ENDPOINT
    assert by_symbol["MSFT"].exclusion_reason is StrategyExclusionReason.MISSING_SHORT_ENDPOINT
    assert by_symbol["PG"].exclusion_reason is StrategyExclusionReason.ASSET_NOT_TRADABLE
    assert by_symbol["PG"].momentum_score == Decimal("0.1")
    assert by_symbol["XOM"].eligible is True
    assert by_symbol["XOM"].target_weight == RationalWeight(1, 1)


def test_negative_scores_are_selected_and_ties_break_by_symbol() -> None:
    sessions = _sessions(254)
    universe = ("ZZZ", "AAA", "MID")
    history = _history(
        sessions,
        {
            "ZZZ": (Decimal("100"), Decimal("90")),
            "AAA": (Decimal("100"), Decimal("90")),
            "MID": (Decimal("100"), Decimal("80")),
        },
        universe=universe,
    )

    decisions = monthly_momentum_v1(
        history,
        signal_session=sessions[-1],
        universe=universe,
        position_count=2,
    )
    by_symbol = {decision.symbol: decision for decision in decisions}

    assert by_symbol["AAA"].rank == 1
    assert by_symbol["ZZZ"].rank == 2
    assert by_symbol["MID"].rank == 3
    assert by_symbol["AAA"].target_weight == RationalWeight(1, 2)
    assert by_symbol["ZZZ"].target_weight == RationalWeight(1, 2)
    assert by_symbol["MID"].target_weight == RationalWeight.zero()
    assert by_symbol["MID"].exclusion_reason is StrategyExclusionReason.NOT_SELECTED
    assert RationalWeight.sum(decision.target_weight for decision in decisions) == RationalWeight(1, 1)


def test_no_eligible_symbol_is_all_cash_and_output_is_repeatable() -> None:
    sessions = _sessions(254)
    universe = ("AAPL", "MSFT")
    history = _history(
        sessions,
        {symbol: (Decimal("100"), Decimal("120")) for symbol in universe},
        universe=universe,
        tradable={(symbol, sessions[-1]): False for symbol in universe},
    )

    first = monthly_momentum_v1(
        history,
        signal_session=sessions[-1],
        universe=universe,
        position_count=2,
    )
    second = monthly_momentum_v1(
        history,
        signal_session=sessions[-1],
        universe=universe,
        position_count=2,
    )

    assert first == second
    assert len(first) == len(universe)
    assert all(decision.eligible is False for decision in first)
    assert all(decision.target_weight == RationalWeight.zero() for decision in first)
    assert RationalWeight.sum(decision.target_weight for decision in first) == RationalWeight.zero()
    assert all(
        decision.exclusion_reason is StrategyExclusionReason.ASSET_NOT_TRADABLE
        for decision in first
    )
