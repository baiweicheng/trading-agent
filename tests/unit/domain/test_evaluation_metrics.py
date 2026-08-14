"""Focused examples for deterministic evaluation metric reference functions."""

# ruff: noqa: E501, I001

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

from quant_research_platform.domain.evaluation import (
    MetricName,
    MetricNullReason,
    MetricScope,
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_evaluation_metrics,
    calculate_maximum_drawdown,
    calculate_monthly_compounding,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_total_return_from_returns,
    calculate_total_commissions,
    calculate_total_slippage,
    calculate_turnover,
    strategy_minus_benchmark,
    total_commissions,
    total_slippage,
)
from quant_research_platform.domain.execution import (
    FillRecord,
    deterministic_fill_id,
    deterministic_order_id,
)


def _close(actual: Decimal, expected: Decimal, tolerance: Decimal = Decimal("1e-24")) -> None:
    assert abs(actual - expected) <= tolerance


def _fill(*, symbol: str, quantity: int, base_open: str, fill_price: str, commission: str, ordinal: int) -> FillRecord:
    signal = date(2024, 1, 31)
    execution = date(2024, 2, 1)
    order_id = deterministic_order_id(
        signal_session=signal,
        execution_session=execution,
        symbol=symbol,
        requested_quantity=quantity,
        ordinal=ordinal,
    )
    return FillRecord(
        fill_id=deterministic_fill_id(
            order_id=order_id,
            symbol=symbol,
            session=execution,
            quantity=quantity,
            ordinal=ordinal,
        ),
        order_id=order_id,
        symbol=symbol,
        session=execution,
        quantity=quantity,
        ordinal=ordinal,
        base_adjusted_open=Decimal(base_open),
        fill_price=Decimal(fill_price),
        gross_notional=abs(quantity * Decimal(fill_price)),
        commission=Decimal(commission),
        slippage_cost=abs(Decimal(fill_price) - Decimal(base_open)) * abs(quantity),
    )


def test_total_return_and_cagr_cover_empty_single_and_return_series_inputs() -> None:
    empty = calculate_total_return([])
    assert empty.value is None
    assert empty.null_reason is MetricNullReason.NO_EVALUATION_SESSIONS

    single = calculate_total_return([Decimal("100")])
    assert single.value == Decimal("0")
    assert calculate_cagr([Decimal("100")]).null_reason is MetricNullReason.INSUFFICIENT_OBSERVATIONS

    returns = calculate_total_return_from_returns([Decimal("0.10"), Decimal("-0.05")])
    assert returns.name is MetricName.TOTAL_RETURN
    assert returns.value == Decimal("0.0450")

    equity = calculate_total_return([Decimal("100"), Decimal("104.5")])
    assert equity.value == Decimal("0.045")

    with localcontext() as context:
        context.prec = 50
        expected = Decimal("1.045") ** (Decimal(252) / Decimal(2)) - Decimal("1")
    cagr = calculate_cagr([Decimal("100"), Decimal("104.5")], return_observations=2)
    assert cagr.value is not None
    _close(cagr.value, expected)


def test_sample_volatility_zero_rate_sharpe_and_signed_drawdown() -> None:
    returns = [Decimal("0.01"), Decimal("0.03"), Decimal("-0.01")]
    volatility = calculate_annualized_volatility(returns)
    assert volatility.value is not None
    with localcontext() as context:
        context.prec = 50
        sample_variance = Decimal("0.0008") / Decimal(2)
        expected_volatility = sample_variance.sqrt() * Decimal(252).sqrt()
    _close(volatility.value, expected_volatility)

    zero_sharpe = calculate_sharpe_ratio([Decimal("0.02"), Decimal("0.02")])
    assert zero_sharpe.value is None
    assert zero_sharpe.null_reason is MetricNullReason.ZERO_VOLATILITY

    drawdown = calculate_maximum_drawdown(
        [Decimal("100"), Decimal("90"), Decimal("99"), Decimal("110"), Decimal("77")]
    )
    assert drawdown.value == Decimal("-0.30")


def test_turnover_commissions_slippage_and_monthly_compounding_use_declared_formulas() -> None:
    fills = (
        _fill(
            symbol="AAPL",
            quantity=3,
            base_open="10",
            fill_price="10.01",
            commission="0.015",
            ordinal=0,
        ),
        _fill(
            symbol="MSFT",
            quantity=-2,
            base_open="20",
            fill_price="19.98",
            commission="0.02",
            ordinal=1,
        ),
    )
    assert total_commissions(fills) == Decimal("0.035000")
    assert total_slippage(fills) == Decimal("0.070000")
    assert calculate_total_commissions(fills).value == Decimal("0.035000")
    assert calculate_total_slippage(fills).value == Decimal("0.070000")

    turnover = calculate_turnover(
        fills,
        portfolio_equity=[Decimal("1000"), Decimal("1100")],
    )
    assert turnover.value is not None
    _close(turnover.value, Decimal("69.99") / Decimal("1050"))

    monthly = calculate_monthly_compounding(
        {
            date(2024, 2, 1): Decimal("-0.10"),
            date(2024, 1, 31): Decimal("0.20"),
            date(2024, 1, 30): Decimal("0.10"),
        }
    )
    assert [(item.month, item.return_value) for item in monthly] == [
        (date(2024, 1, 31), Decimal("0.32")),
        (date(2024, 2, 1), Decimal("-0.10")),
    ]


def test_evaluation_uses_explicit_returns_and_signed_strategy_minus_benchmark() -> None:
    strategy = calculate_evaluation_metrics(
        MetricScope.STRATEGY,
        [Decimal("100"), Decimal("110"), Decimal("104.5")],
        returns=[Decimal("0.10"), Decimal("-0.05")],
    )
    benchmark = calculate_evaluation_metrics(
        MetricScope.BENCHMARK,
        [Decimal("100"), Decimal("102"), Decimal("105")],
        returns=[Decimal("0.02"), Decimal("0.03")],
    )
    assert strategy.metric(MetricName.TOTAL_RETURN).value == Decimal("0.0450")
    with localcontext() as context:
        context.prec = 50
        expected_cagr = Decimal("1.045") ** (Decimal(252) / Decimal(2)) - Decimal("1")
    strategy_cagr = strategy.metric(MetricName.COMPOUND_ANNUAL_GROWTH_RATE).value
    assert strategy_cagr is not None
    _close(strategy_cagr, expected_cagr)

    differences = strategy_minus_benchmark(strategy, benchmark)
    assert differences.scope is MetricScope.DIFFERENCE
    assert differences.metric(MetricName.TOTAL_RETURN).value == Decimal("-0.0056")
    assert differences.metric(MetricName.MAXIMUM_DRAWDOWN).value == Decimal("-0.05")
