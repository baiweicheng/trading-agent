"""Property tests for complete deterministic monthly momentum decisions."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.strategy import (
    WARM_UP_SESSIONS,
    PriceHistory,
    PriceObservation,
    StrategyExclusionReason,
    monthly_momentum_v1,
)

_SYMBOL_POOL = ("AAPL", "MSFT", "NVDA", "SPY", "XOM", "ZZZ")
_SIGNAL_SESSION = date(2024, 12, 31)
_BASE_CLOSE = Decimal("100")


@dataclass(frozen=True)
class MomentumCase:
    """A bounded synthetic XNYS history with one month-end signal."""

    universe: tuple[str, ...]
    sessions: tuple[date, ...]
    score_ticks: tuple[int, ...]
    long_available: tuple[bool, ...]
    short_available: tuple[bool, ...]
    tradable: tuple[bool, ...]
    position_count: int
    history: PriceHistory

    @property
    def signal_session(self) -> date:
        return self.sessions[-1]


@dataclass(frozen=True)
class ReferenceDecision:
    """The independent, minimal projection used to check production decisions."""

    symbol: str
    endpoint_252_session: date | None
    endpoint_252_close: Decimal | None
    endpoint_252_checksum: str | None
    endpoint_21_session: date | None
    endpoint_21_close: Decimal | None
    endpoint_21_checksum: str | None
    momentum_score: Decimal | None
    eligible: bool
    rank: int | None
    target_weight: Fraction
    exclusion_reason: StrategyExclusionReason | None


def _synthetic_xnys_sessions(count: int) -> tuple[date, ...]:
    """Return weekday-only sessions ending on the December month end."""

    sessions: list[date] = []
    candidate = _SIGNAL_SESSION
    while len(sessions) < count:
        if candidate.weekday() < 5:
            sessions.append(candidate)
        candidate -= timedelta(days=1)
    return tuple(reversed(sessions))


@st.composite
def momentum_cases(draw: st.DrawFn) -> MomentumCase:
    """Generate complete and incomplete histories through one signal close."""

    universe = tuple(
        draw(
            st.lists(
                st.sampled_from(_SYMBOL_POOL),
                min_size=1,
                max_size=len(_SYMBOL_POOL),
                unique=True,
            )
        )
    )
    session_count = draw(st.integers(min_value=1, max_value=260))
    sessions = _synthetic_xnys_sessions(session_count)

    score_mode = draw(st.sampled_from(("equal", "distinct", "mixed")))
    if score_mode == "equal":
        equal_tick = draw(st.integers(min_value=-900, max_value=1000))
        score_ticks = (equal_tick,) * len(universe)
    elif score_mode == "distinct":
        score_ticks = tuple(
            draw(
                st.lists(
                    st.integers(min_value=-900, max_value=1000),
                    min_size=len(universe),
                    max_size=len(universe),
                    unique=True,
                )
            )
        )
    else:
        score_ticks = tuple(
            draw(
                st.lists(
                    st.integers(min_value=-900, max_value=1000),
                    min_size=len(universe),
                    max_size=len(universe),
                )
            )
        )

    long_available = tuple(
        draw(st.lists(st.booleans(), min_size=len(universe), max_size=len(universe)))
    )
    short_available = tuple(
        draw(st.lists(st.booleans(), min_size=len(universe), max_size=len(universe)))
    )
    tradable = tuple(
        draw(st.lists(st.booleans(), min_size=len(universe), max_size=len(universe)))
    )
    position_count = draw(st.integers(min_value=1, max_value=len(universe)))

    signal_index = len(sessions) - 1
    long_index = signal_index - 252
    short_index = signal_index - 21
    observations: list[PriceObservation] = []
    for symbol_index, symbol in enumerate(universe):
        for session_index, session in enumerate(sessions):
            if session_index == long_index and not long_available[symbol_index]:
                continue
            if session_index == short_index and not short_available[symbol_index]:
                continue

            close = _BASE_CLOSE
            if session_index == short_index:
                # A tenth-dollar tick over $100 creates a finite score in
                # [-0.9, 1.0] while retaining exact Decimal arithmetic.
                close = _BASE_CLOSE + Decimal(score_ticks[symbol_index]) / Decimal("10")
            observations.append(
                PriceObservation(
                    symbol=symbol,
                    session=session,
                    adjusted_close=close,
                )
            )

    history = PriceHistory(
        observations=tuple(observations),
        sessions=sessions,
        universe=universe,
    )
    return MomentumCase(
        universe=universe,
        sessions=sessions,
        score_ticks=score_ticks,
        long_available=long_available,
        short_available=short_available,
        tradable=tradable,
        position_count=position_count,
        history=history,
    )


def _reference_decisions(case: MomentumCase) -> tuple[ReferenceDecision, ...]:
    """Apply the policy's stated sort/slice rules without calling its helpers."""

    by_key = {
        (observation.symbol, observation.session): observation
        for observation in case.history
    }
    signal_index = len(case.sessions) - 1
    long_session = case.sessions[signal_index - 252] if signal_index >= 252 else None
    short_session = case.sessions[signal_index - 21] if signal_index >= 21 else None
    warmup_complete = signal_index >= WARM_UP_SESSIONS

    rows: dict[str, dict[str, object]] = {}
    eligible_scores: list[tuple[str, Decimal]] = []
    for symbol in case.universe:
        long_observation = (
            by_key.get((symbol, long_session)) if long_session is not None else None
        )
        short_observation = (
            by_key.get((symbol, short_session)) if short_session is not None else None
        )
        long_available = long_session is not None and long_observation is not None
        short_available = short_session is not None and short_observation is not None
        endpoint_252_session = long_session if long_available else None
        endpoint_252_close = (
            long_observation.adjusted_close if long_observation is not None else None
        )
        endpoint_252_checksum = (
            long_observation.canonical_row_checksum
            if long_observation is not None
            else None
        )
        endpoint_21_session = short_session if short_available else None
        endpoint_21_close = (
            short_observation.adjusted_close if short_observation is not None else None
        )
        endpoint_21_checksum = (
            short_observation.canonical_row_checksum
            if short_observation is not None
            else None
        )

        reason: StrategyExclusionReason | None = None
        score: Decimal | None = None
        if not warmup_complete:
            reason = StrategyExclusionReason.WARM_UP_INCOMPLETE
        elif not long_available:
            reason = StrategyExclusionReason.MISSING_LONG_ENDPOINT
        elif not short_available:
            reason = StrategyExclusionReason.MISSING_SHORT_ENDPOINT
        else:
            assert long_observation is not None
            assert short_observation is not None
            with localcontext() as context:
                context.prec = 28
                score = (
                    short_observation.adjusted_close / long_observation.adjusted_close
                    - Decimal("1")
                )
            if not case.tradable[case.universe.index(symbol)]:
                reason = StrategyExclusionReason.ASSET_NOT_TRADABLE
            else:
                eligible_scores.append((symbol, score))

        rows[symbol] = {
            "endpoint_252_session": endpoint_252_session,
            "endpoint_252_close": endpoint_252_close,
            "endpoint_252_checksum": endpoint_252_checksum,
            "endpoint_21_session": endpoint_21_session,
            "endpoint_21_close": endpoint_21_close,
            "endpoint_21_checksum": endpoint_21_checksum,
            "momentum_score": score,
            "reason": reason,
        }

    eligible_scores.sort(key=lambda item: (-item[1], item[0]))
    ranks = {symbol: rank for rank, (symbol, _) in enumerate(eligible_scores, start=1)}
    selected_symbols = {symbol for symbol, _ in eligible_scores[: case.position_count]}
    selected_weight = (
        Fraction(1, len(selected_symbols)) if selected_symbols else Fraction(0, 1)
    )

    expected: list[ReferenceDecision] = []
    for symbol in case.universe:
        row = rows[symbol]
        rank = ranks.get(symbol)
        eligible = rank is not None
        reason = row["reason"]
        assert reason is None or isinstance(reason, StrategyExclusionReason)
        if eligible and symbol not in selected_symbols:
            reason = StrategyExclusionReason.NOT_SELECTED
        elif eligible and symbol in selected_symbols:
            reason = None
        expected.append(
            ReferenceDecision(
                symbol=symbol,
                endpoint_252_session=row["endpoint_252_session"],
                endpoint_252_close=row["endpoint_252_close"],
                endpoint_252_checksum=row["endpoint_252_checksum"],
                endpoint_21_session=row["endpoint_21_session"],
                endpoint_21_close=row["endpoint_21_close"],
                endpoint_21_checksum=row["endpoint_21_checksum"],
                momentum_score=row["momentum_score"],
                eligible=eligible,
                rank=rank,
                target_weight=selected_weight
                if symbol in selected_symbols
                else Fraction(0, 1),
                exclusion_reason=reason,
            )
        )
    return tuple(expected)


# Feature: quantitative-research-platform, Property 8: Monthly momentum decisions are complete, exact, and deterministic
# Validates: Requirements 8.1–8.15
@settings(max_examples=100, deadline=None)
@given(case=momentum_cases())
def test_monthly_momentum_decisions_are_complete_exact_and_deterministic(
    case: MomentumCase,
) -> None:
    """Decisions match an independent ranking/weight reference model."""

    tradability = dict(zip(case.universe, case.tradable, strict=True))
    first = monthly_momentum_v1(
        case.history,
        signal_session=case.signal_session,
        universe=case.universe,
        position_count=case.position_count,
        tradability=tradability,
    )
    second = monthly_momentum_v1(
        case.history,
        signal_session=case.signal_session,
        universe=case.universe,
        position_count=case.position_count,
        tradability=tradability,
    )
    expected = _reference_decisions(case)

    assert first == second
    assert canonical_json(
        [decision.to_serializable() for decision in first]
    ) == canonical_json([decision.to_serializable() for decision in second])
    assert len(first) == len(case.universe)
    assert tuple(decision.symbol for decision in first) == case.universe

    for actual, reference in zip(first, expected, strict=True):
        assert actual.symbol == reference.symbol
        assert actual.endpoint_252_session == reference.endpoint_252_session
        assert actual.endpoint_252_close == reference.endpoint_252_close
        assert actual.endpoint_252_checksum == reference.endpoint_252_checksum
        assert actual.endpoint_21_session == reference.endpoint_21_session
        assert actual.endpoint_21_close == reference.endpoint_21_close
        assert actual.endpoint_21_checksum == reference.endpoint_21_checksum
        assert actual.momentum_score == reference.momentum_score
        assert actual.eligible is reference.eligible
        assert actual.rank == reference.rank
        assert actual.target_weight.fraction == reference.target_weight
        assert actual.exclusion_reason is reference.exclusion_reason
        assert actual.target_weight.numerator >= 0
        assert actual.target_weight.denominator >= 1

    selected = [decision for decision in first if decision.target_weight.numerator > 0]
    total_weight = sum(
        (decision.target_weight.fraction for decision in first),
        Fraction(0, 1),
    )
    assert len(selected) <= case.position_count
    assert total_weight == (Fraction(1, 1) if selected else Fraction(0, 1))
    assert all(decision.eligible for decision in selected)
    assert all(
        decision.target_weight == type(decision.target_weight).zero()
        for decision in first
        if not decision.eligible
    )

    # Fewer than 253 preceding sessions is the warm-up period: decisions may
    # explain the exclusion, but no order-producing positive target is allowed.
    if len(case.sessions) - 1 < WARM_UP_SESSIONS:
        assert all(not decision.eligible for decision in first)
        assert all(decision.target_weight.numerator == 0 for decision in first)
        assert all(
            decision.exclusion_reason is StrategyExclusionReason.WARM_UP_INCOMPLETE
            for decision in first
        )
