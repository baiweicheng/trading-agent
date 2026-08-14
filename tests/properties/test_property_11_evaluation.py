# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, localcontext
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.evaluation import EvaluationService
from quant_research_platform.domain.canonical import sha256_bytes
from quant_research_platform.domain.errors import (
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.domain.evaluation import (
    MetricName,
    MetricNullReason,
    MetricScope,
)
from quant_research_platform.domain.execution import (
    DailyReturn,
    FillRecord,
    OrderRecord,
    OrderStatus,
    PortfolioState,
    deterministic_fill_id,
    deterministic_order_id,
    quantize_money,
)
from quant_research_platform.domain.market import DateRange

_ZERO = Decimal("0")
_ONE = Decimal("1")
_INITIAL_EQUITY = Decimal("100000")
_BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Bounded, finite inputs for one strategy/SPY evaluation."""

    sessions: tuple[date, ...]
    strategy_equity: tuple[Decimal, ...]
    spy_prices: tuple[Decimal, ...]
    spy_gap_indices: frozenset[int]
    commission_bps: Decimal
    slippage_bps: Decimal
    fill_quantity: int
    fill_base_open: Decimal


ReferenceMetric = tuple[Decimal | int | None, MetricNullReason | None]


@st.composite
def evaluation_cases(draw: st.DrawFn) -> EvaluationCase:
    """Generate positive curves, explicit benchmark gaps, and finite costs."""

    count = draw(st.integers(min_value=1, max_value=12))
    offsets = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=180),
                min_size=count,
                max_size=count,
                unique=True,
            )
        )
    )
    start = date(2024, 1, 15)
    sessions = tuple(start + timedelta(days=offset) for offset in offsets)
    strategy_equity = tuple(
        Decimal(value)
        for value in draw(
            st.lists(
                st.integers(min_value=1, max_value=200_000),
                min_size=count,
                max_size=count,
            )
        )
    )
    spy_prices = tuple(
        Decimal(value)
        for value in draw(
            st.lists(
                st.integers(min_value=1, max_value=20_000),
                min_size=count,
                max_size=count,
            )
        )
    )
    spy_gap_indices = frozenset(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=count - 1),
                max_size=count,
            )
        )
    )
    return EvaluationCase(
        sessions=sessions,
        strategy_equity=strategy_equity,
        spy_prices=spy_prices,
        spy_gap_indices=spy_gap_indices,
        commission_bps=Decimal(draw(st.integers(min_value=0, max_value=20_000))),
        slippage_bps=Decimal(draw(st.integers(min_value=0, max_value=5_000))),
        fill_quantity=draw(st.integers(min_value=1, max_value=20)),
        fill_base_open=Decimal(draw(st.integers(min_value=1, max_value=20_000))),
    )


def _returns_from_equity(equity: Sequence[Decimal]) -> tuple[Decimal, ...]:
    return tuple(
        _ZERO if index == 0 else equity[index] / equity[index - 1] - _ONE
        for index in range(len(equity))
    )


def _benchmark_returns(prices: Sequence[Decimal]) -> tuple[Decimal, ...]:
    return _returns_from_equity(prices)


def _benchmark_equity(returns: Sequence[Decimal]) -> tuple[Decimal, ...]:
    current = _INITIAL_EQUITY
    values: list[Decimal] = []
    for index, value in enumerate(returns):
        if index:
            current *= _ONE + value
        values.append(current)
    return tuple(values)


def _null(reason: MetricNullReason) -> ReferenceMetric:
    return None, reason


def _performance_metrics(
    equity: Sequence[Decimal], returns: Sequence[Decimal]
) -> dict[MetricName, ReferenceMetric]:
    if not equity:
        return {
            MetricName.TOTAL_RETURN: _null(MetricNullReason.NO_EVALUATION_SESSIONS),
            MetricName.COMPOUND_ANNUAL_GROWTH_RATE: _null(MetricNullReason.NO_EVALUATION_SESSIONS),
            MetricName.ANNUALIZED_VOLATILITY: _null(MetricNullReason.NO_EVALUATION_SESSIONS),
            MetricName.SHARPE_RATIO: _null(MetricNullReason.NO_EVALUATION_SESSIONS),
            MetricName.MAXIMUM_DRAWDOWN: _null(MetricNullReason.NO_EVALUATION_SESSIONS),
        }

    compounded = _ONE
    for value in returns:
        compounded *= _ONE + value

    if len(equity) < 2:
        cagr = _null(MetricNullReason.INSUFFICIENT_OBSERVATIONS)
    else:
        with localcontext() as context:
            context.prec = 40
            cagr = (
                equity[-1] / equity[0]
            ) ** (Decimal(252) / Decimal(len(returns))) - _ONE, None

    if len(returns) < 2:
        volatility = _null(MetricNullReason.INSUFFICIENT_OBSERVATIONS)
        sharpe = _null(MetricNullReason.INSUFFICIENT_OBSERVATIONS)
    else:
        mean = sum(returns, _ZERO) / Decimal(len(returns))
        variance = sum(((value - mean) ** 2 for value in returns), _ZERO) / Decimal(len(returns) - 1)
        with localcontext() as context:
            context.prec = 40
            standard_deviation = variance.sqrt()
            volatility = standard_deviation * Decimal(252).sqrt(), None
            if standard_deviation == _ZERO:
                sharpe = _null(MetricNullReason.ZERO_VOLATILITY)
            else:
                sharpe = (mean / standard_deviation * Decimal(252).sqrt(), None)

    peak = equity[0]
    maximum_drawdown = _ZERO
    for value in equity:
        peak = max(peak, value)
        maximum_drawdown = min(maximum_drawdown, value / peak - _ONE)

    return {
        MetricName.TOTAL_RETURN: (compounded - _ONE, None),
        MetricName.COMPOUND_ANNUAL_GROWTH_RATE: cagr,
        MetricName.ANNUALIZED_VOLATILITY: volatility,
        MetricName.SHARPE_RATIO: sharpe,
        MetricName.MAXIMUM_DRAWDOWN: (maximum_drawdown, None),
    }


def _reference_metrics(
    scope: MetricScope,
    equity: Sequence[Decimal],
    returns: Sequence[Decimal],
    *,
    fill: FillRecord,
) -> dict[MetricName, ReferenceMetric]:
    metrics = _performance_metrics(equity, returns)
    if scope is not MetricScope.STRATEGY:
        return metrics

    average_equity = sum(equity, _ZERO) / Decimal(len(equity))
    metrics.update(
        {
            MetricName.TURNOVER: (fill.gross_notional / average_equity, None),
            MetricName.TOTAL_COMMISSIONS: (fill.commission, None),
            MetricName.TOTAL_SLIPPAGE: (fill.slippage_cost, None),
            MetricName.UNFILLED_ORDERS: (0, None),
            MetricName.ENDING_CASH_BALANCE: (equity[-1], None),
        }
    )
    return metrics


def _reference_differences(
    strategy: Mapping[MetricName, ReferenceMetric],
    benchmark: Mapping[MetricName, ReferenceMetric],
) -> dict[MetricName, ReferenceMetric]:
    result: dict[MetricName, ReferenceMetric] = {}
    for name in (
        MetricName.TOTAL_RETURN,
        MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
        MetricName.ANNUALIZED_VOLATILITY,
        MetricName.SHARPE_RATIO,
        MetricName.MAXIMUM_DRAWDOWN,
    ):
        strategy_value, strategy_reason = strategy[name]
        benchmark_value, benchmark_reason = benchmark[name]
        if strategy_value is None:
            result[name] = _null(strategy_reason or MetricNullReason.NO_EVALUATION_SESSIONS)
        elif benchmark_value is None:
            result[name] = _null(benchmark_reason or MetricNullReason.NO_EVALUATION_SESSIONS)
        else:
            assert isinstance(strategy_value, Decimal)
            assert isinstance(benchmark_value, Decimal)
            result[name] = (strategy_value - benchmark_value, None)
    return result


def _monthly_reference(
    sessions: Sequence[date], returns: Sequence[Decimal]
) -> tuple[tuple[date, Decimal], ...]:
    rows: list[tuple[date, Decimal]] = []
    current_month: tuple[int, int] | None = None
    compounded = _ONE
    month_end: date | None = None
    for session, value in zip(sessions, returns, strict=True):
        year_month = (session.year, session.month)
        if current_month is not None and current_month != year_month:
            assert month_end is not None
            rows.append((month_end, compounded - _ONE))
            compounded = _ONE
        current_month = year_month
        compounded *= _ONE + value
        month_end = session
    if month_end is not None:
        rows.append((month_end, compounded - _ONE))
    return tuple(rows)


def _make_fill(case: EvaluationCase) -> tuple[OrderRecord, FillRecord]:
    session = case.sessions[0]
    signal_session = session - timedelta(days=1)
    order_id = deterministic_order_id(
        signal_session=signal_session,
        execution_session=session,
        symbol="AAPL",
        requested_quantity=case.fill_quantity,
        ordinal=0,
    )
    order = OrderRecord(
        order_id=order_id,
        signal_session=signal_session,
        execution_session=session,
        symbol="AAPL",
        requested_quantity=case.fill_quantity,
        ordinal=0,
        status=OrderStatus.FILLED,
    )
    fill_price = quantize_money(
        case.fill_base_open
        * (_ONE + case.slippage_bps / _BPS)
    )
    fill = FillRecord(
        fill_id=deterministic_fill_id(
            order_id=order_id,
            symbol="AAPL",
            session=session,
            quantity=case.fill_quantity,
            ordinal=0,
        ),
        order_id=order_id,
        symbol="AAPL",
        session=session,
        quantity=case.fill_quantity,
        ordinal=0,
        base_adjusted_open=case.fill_base_open,
        fill_price=fill_price,
        gross_notional=quantize_money(case.fill_quantity * fill_price),
        commission=quantize_money(
            case.fill_quantity * fill_price * case.commission_bps / _BPS
        ),
        slippage_cost=quantize_money(
            case.fill_quantity * abs(fill_price - case.fill_base_open)
        ),
    )
    return order, fill


def _core_output(case: EvaluationCase, *, reverse: bool = False) -> SimpleNamespace:
    order, fill = _make_fill(case)
    returns = _returns_from_equity(case.strategy_equity)
    states = tuple(
        PortfolioState(
            session=session,
            cash_balance=equity,
            positions=(),
            gross_exposure=_ZERO,
            portfolio_equity=equity,
            leverage=_ZERO,
        )
        for session, equity in zip(case.sessions, case.strategy_equity, strict=True)
    )
    daily_returns = tuple(
        DailyReturn(session, value)
        for session, value in zip(case.sessions, returns, strict=True)
    )
    if reverse:
        states = tuple(reversed(states))
        daily_returns = tuple(reversed(daily_returns))
    return SimpleNamespace(
        orders=(order,),
        fills=(fill,),
        portfolio_states=states,
        daily_returns=daily_returns,
        strategy_decisions=(),
    )


def _snapshot(case: EvaluationCase, *, reverse: bool = False) -> SimpleNamespace:
    bars = [
        {"session": session, "adjusted_close": price}
        for index, (session, price) in enumerate(zip(case.sessions, case.spy_prices, strict=True))
        if index not in case.spy_gap_indices
    ]
    if reverse:
        bars.reverse()
    return SimpleNamespace(
        benchmark_bars=bars,
        limitation_disclosure=LimitationDisclosure.current(),
    )


def _assert_close(actual: Decimal, expected: Decimal) -> None:
    scale = max(abs(actual), abs(expected), _ONE)
    tolerance = max(Decimal("1e-12"), scale * Decimal("1e-24"))
    assert abs(actual - expected) <= tolerance


def _assert_metric_set(
    actual: object,
    expected: Mapping[MetricName, ReferenceMetric],
) -> None:
    for name, (expected_value, expected_reason) in expected.items():
        metric = actual.metric(name)  # type: ignore[attr-defined]
        assert metric.null_reason == expected_reason
        if expected_value is None:
            assert metric.value is None
        elif isinstance(expected_value, int):
            assert metric.value == expected_value
        else:
            assert isinstance(metric.value, Decimal)
            _assert_close(metric.value, expected_value)


# Feature: quantitative-research-platform, Property 11: Evaluation is aligned, gap-safe, and deterministic
# Validates: Requirements 10.1–10.18, 17.25
@settings(max_examples=100, deadline=None)
@given(case=evaluation_cases())
def test_evaluation_is_aligned_gap_safe_and_deterministic(case: EvaluationCase) -> None:
    """Evaluation agrees with independent formulas and canonicalizes row order."""

    evaluation_range = DateRange(case.sessions[0], case.sessions[-1])
    first = EvaluationService().evaluate(
        _core_output(case),
        _snapshot(case),
        evaluation_range=evaluation_range,
    )

    if case.spy_gap_indices:
        assert isinstance(first, Err)
        expected_missing = tuple(
            session
            for index, session in enumerate(case.sessions)
            if index in case.spy_gap_indices
        )
        assert tuple(error.session for error in first.errors) == expected_missing
        assert all(error.category is ErrorCategory.VALIDATION_GAP for error in first.errors)
        assert all(error.symbol == "SPY" for error in first.errors)
        assert len(first.errors) == len(case.spy_gap_indices)
        return

    assert isinstance(first, Ok)
    evaluated = first.value
    strategy_returns = _returns_from_equity(case.strategy_equity)
    benchmark_returns = _benchmark_returns(case.spy_prices)
    benchmark_equity = _benchmark_equity(benchmark_returns)
    _, fill = _make_fill(case)
    expected_strategy = _reference_metrics(
        MetricScope.STRATEGY,
        case.strategy_equity,
        strategy_returns,
        fill=fill,
    )
    expected_benchmark = _reference_metrics(
        MetricScope.BENCHMARK,
        benchmark_equity,
        benchmark_returns,
        fill=fill,
    )
    expected_difference = _reference_differences(expected_strategy, expected_benchmark)

    assert evaluated.evaluation_range == evaluation_range
    assert tuple(item.session for item in evaluated.strategy_returns) == case.sessions
    assert tuple(item.return_value for item in evaluated.strategy_returns) == strategy_returns
    assert tuple(item.session for item in evaluated.benchmark_returns) == case.sessions
    assert tuple(item.return_value for item in evaluated.benchmark_returns) == benchmark_returns
    assert tuple(evaluated.strategy_equity) == tuple(zip(case.sessions, case.strategy_equity, strict=True))
    assert tuple(evaluated.benchmark_equity) == tuple(zip(case.sessions, benchmark_equity, strict=True))

    _assert_metric_set(evaluated.evaluation_result.strategy_metrics, expected_strategy)
    _assert_metric_set(evaluated.evaluation_result.benchmark_metrics, expected_benchmark)
    _assert_metric_set(evaluated.evaluation_result.differences, expected_difference)
    assert evaluated.total_commissions == fill.commission
    assert evaluated.total_slippage == fill.slippage_cost
    assert evaluated.ending_cash_balance == quantize_money(case.strategy_equity[-1])

    expected_strategy_monthly = _monthly_reference(case.sessions, strategy_returns)
    expected_benchmark_monthly = _monthly_reference(case.sessions, benchmark_returns)
    assert tuple(
        (item.month, item.return_value) for item in evaluated.strategy_monthly_returns
    ) == expected_strategy_monthly
    assert tuple(
        (item.month, item.return_value) for item in evaluated.benchmark_monthly_returns
    ) == expected_benchmark_monthly

    second = EvaluationService().evaluate(
        _core_output(case, reverse=True),
        _snapshot(case, reverse=True),
        evaluation_range=evaluation_range,
    )
    assert isinstance(second, Ok)
    assert evaluated.artifact_checksums == second.value.artifact_checksums
    assert evaluated.to_serializable() == second.value.to_serializable()
    for artifact in evaluated.artifacts:
        assert artifact.checksum == sha256_bytes(artifact.payload)
        assert artifact.payload.endswith(b"\n")
