"""Focused offline tests for the cash-safe Zipline execution seam."""

# ruff: noqa: E501, I001

from __future__ import annotations

from datetime import date
from decimal import Decimal

from quant_research_platform.infrastructure.zipline_engine import CashSafeOpenBlotter


SESSION = date(2024, 2, 1)


def _order(
    order_id: str, symbol: str, amount: int, rank: int | None = None
) -> dict[str, object]:
    value: dict[str, object] = {
        "id": order_id,
        "symbol": symbol,
        "amount": amount,
        "filled": 0,
    }
    if rank is not None:
        value["decision_rank"] = rank
    return value


def test_sells_fund_buys_and_buy_remainder_is_cash_capped() -> None:
    blotter = CashSafeOpenBlotter(commission_bps=0, slippage_bps=0)

    result = blotter.execute_orders(
        [
            _order("buy-msft", "MSFT", 15, rank=1),
            _order("sell-aapl", "AAPL", -10),
        ],
        opens={"AAPL": Decimal("100"), "MSFT": Decimal("100")},
        cash=Decimal("0"),
        positions={"AAPL": 10},
        session=SESSION,
    )

    assert [(fill.symbol, fill.quantity) for fill in result.fills] == [
        ("AAPL", -10),
        ("MSFT", 10),
    ]
    assert result.cash_balance == Decimal("0.000000")
    assert result.positions == {"MSFT": 10}
    assert result.unfilled_orders[0].quantity == 5


def test_adverse_price_and_commission_use_actual_fill_quantity() -> None:
    blotter = CashSafeOpenBlotter()

    result = blotter.execute_orders(
        [_order("buy-aapl", "AAPL", 10)],
        opens={"AAPL": Decimal("100")},
        cash=Decimal("1001"),
        positions={},
        session=SESSION,
    )

    fill = result.fills[0]
    assert fill.quantity == 9
    assert fill.base_adjusted_open == Decimal("100.000000")
    assert fill.fill_price == Decimal("100.100000")
    assert fill.gross_notional == Decimal("900.900000")
    assert fill.commission == Decimal("0.450450")
    assert fill.slippage_cost == Decimal("0.900000")
    assert result.cash_balance >= 0


def test_missing_open_is_unfilled_and_actionable() -> None:
    blotter = CashSafeOpenBlotter(commission_bps=0, slippage_bps=0)

    result = blotter.execute_orders(
        [_order("missing-aapl", "AAPL", 1)],
        opens={},
        cash=Decimal("100"),
        positions={},
        session=SESSION,
    )

    assert result.fills == ()
    assert result.unfilled_orders[0].reason == "missing_or_non_positive_adjusted_open"
    assert result.actionable_errors[0].symbol == "AAPL"
    assert result.actionable_errors[0].session == SESSION


def test_commission_above_one_hundred_percent_caps_sell_without_negative_cash() -> None:
    blotter = CashSafeOpenBlotter(commission_bps=20000, slippage_bps=0)

    result = blotter.execute_orders(
        [_order("sell-aapl", "AAPL", -10)],
        opens={"AAPL": Decimal("10")},
        cash=Decimal("100"),
        positions={"AAPL": 10},
        session=SESSION,
    )

    assert result.fills[0].quantity == -10
    assert result.fills[0].commission == Decimal("200.000000")
    assert result.cash_balance == Decimal("0.000000")
    assert result.positions == {}


def test_non_positive_adverse_price_produces_no_zero_fill() -> None:
    blotter = CashSafeOpenBlotter(commission_bps=0, slippage_bps=20000)

    result = blotter.execute_orders(
        [_order("sell-aapl", "AAPL", -2)],
        opens={"AAPL": Decimal("10")},
        cash=Decimal("0"),
        positions={"AAPL": 2},
        session=SESSION,
    )

    assert result.fills == ()
    assert result.unfilled_orders[0].quantity == -2
    assert result.unfilled_orders[0].reason == "invalid_adjusted_open"
