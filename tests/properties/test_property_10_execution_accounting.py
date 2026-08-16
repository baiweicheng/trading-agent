"""Property tests for whole-share next-open execution and accounting."""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import cast

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.infrastructure.zipline_engine import CashSafeOpenBlotter

SESSION = date(2024, 2, 1)
_MONEY = Decimal("0.000001")
_BPS = Decimal("10000")
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class MarkAction:
    """One action-effective mark used by the independent ledger model."""

    symbol: str
    mark_price: Decimal
    cash_return: Decimal = _ZERO


@dataclass(frozen=True, slots=True)
class ExecutionCase:
    """All state needed to execute one deterministic open."""

    cash: Decimal
    positions: Mapping[str, int]
    orders: tuple[dict[str, object], ...]
    opens: Mapping[str, Decimal]
    actions: tuple[MarkAction, ...]
    commission_bps: Decimal
    slippage_bps: Decimal


@dataclass(frozen=True, slots=True)
class ReferenceFill:
    """The independent model's projection of one actual fill."""

    order_id: str
    symbol: str
    quantity: int
    base_open: Decimal
    fill_price: Decimal
    gross_notional: Decimal
    commission: Decimal
    slippage_cost: Decimal


@dataclass(frozen=True, slots=True)
class ReferenceUnfilled:
    """The independent model's projection of one remaining order."""

    order_id: str
    symbol: str
    quantity: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """The independent Decimal sell-first/buy-second ledger state."""

    cash: Decimal
    positions: Mapping[str, int]
    fills: tuple[ReferenceFill, ...]
    unfilled: tuple[ReferenceUnfilled, ...]


def _money(value: Decimal) -> Decimal:
    """Quantize monetary values exactly as the accounting contract requires."""

    with localcontext() as context:
        context.prec = 40
        return value.quantize(_MONEY, rounding=ROUND_HALF_EVEN)


def _price(ticks: int) -> Decimal:
    return Decimal(ticks) / Decimal("100")


def _commission(quantity: int, fill_price: Decimal, commission_bps: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 40
        return _money(abs(quantity) * fill_price * commission_bps / _BPS)


def _adverse_price(
    base_open: Decimal, quantity: int, slippage_bps: Decimal
) -> Decimal | None:
    with localcontext() as context:
        context.prec = 40
        rate = slippage_bps / _BPS
        candidate = base_open * (
            Decimal("1") + rate if quantity > 0 else Decimal("1") - rate
        )
    if not candidate.is_finite() or candidate <= _ZERO:
        return None
    return _money(candidate)


def _buy_is_affordable(
    cash: Decimal, fill_price: Decimal, quantity: int, commission_bps: Decimal
) -> bool:
    notional = _money(quantity * fill_price)
    commission = _commission(quantity, fill_price, commission_bps)
    return _money(cash - notional - commission) >= _ZERO


def _largest_affordable_buy(
    cash: Decimal,
    fill_price: Decimal,
    requested: int,
    commission_bps: Decimal,
) -> int:
    """Brute-force the greatest affordable whole share, independently."""

    for quantity in range(requested, 0, -1):
        if _buy_is_affordable(cash, fill_price, quantity, commission_bps):
            return quantity
    return 0


def _largest_affordable_sell(
    cash: Decimal,
    fill_price: Decimal,
    requested: int,
    holdings: int,
    commission_bps: Decimal,
) -> int:
    """Brute-force the greatest sell preserving non-negative cash."""

    for quantity in range(min(requested, holdings), 0, -1):
        notional = _money(quantity * fill_price)
        commission = _commission(-quantity, fill_price, commission_bps)
        if _money(cash + notional - commission) >= _ZERO:
            return quantity
    return 0


@st.composite
def execution_cases(draw: st.DrawFn) -> ExecutionCase:
    """Generate valid portfolios, order actions, marks, opens, and costs."""

    symbols = tuple(
        draw(
            st.lists(
                st.sampled_from(("AAPL", "MSFT", "NVDA", "XOM")),
                min_size=1,
                max_size=4,
                unique=True,
            )
        )
    )
    quantities = draw(
        st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=len(symbols),
            max_size=len(symbols),
        )
    )
    # At least one starting share makes the starting marked equity positive even
    # when the generated cash balance is zero.
    if not any(quantities):
        quantities[0] = 1
    positions = {
        symbol: quantity
        for symbol, quantity in zip(symbols, quantities, strict=True)
        if quantity
    }
    cash = _price(draw(st.integers(min_value=0, max_value=2_000_000)))
    marks = draw(
        st.lists(
            st.integers(min_value=1, max_value=100_000),
            min_size=len(symbols),
            max_size=len(symbols),
        )
    )
    actions = tuple(
        MarkAction(symbol=symbol, mark_price=_price(mark))
        for symbol, mark in zip(symbols, marks, strict=True)
    )

    open_values = draw(
        st.lists(
            st.one_of(
                st.none(),
                st.integers(min_value=1, max_value=100_000).map(_price),
            ),
            min_size=len(symbols),
            max_size=len(symbols),
        )
    )
    opens = {
        symbol: value
        for symbol, value in zip(symbols, open_values, strict=True)
        if value is not None
    }

    sell_specs = draw(
        st.lists(
            st.tuples(
                st.sampled_from(symbols),
                st.integers(min_value=1, max_value=40),
            ),
            min_size=1,
            max_size=3,
        )
    )
    buy_specs = draw(
        st.lists(
            st.tuples(
                st.sampled_from(symbols),
                st.integers(min_value=1, max_value=40),
                st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
            ),
            min_size=1,
            max_size=3,
        )
    )
    generated_orders: list[dict[str, object]] = []
    for ordinal, (symbol, quantity) in enumerate(sell_specs):
        generated_orders.append(
            {
                "id": f"sell-{ordinal}-{symbol}",
                "symbol": symbol,
                "amount": -quantity,
                "filled": 0,
            }
        )
    for offset, (symbol, quantity, rank) in enumerate(buy_specs):
        generated_orders.append(
            {
                "id": f"buy-{offset}-{symbol}",
                "symbol": symbol,
                "amount": quantity,
                "filled": 0,
                "decision_rank": rank,
            }
        )
    orders = tuple(draw(st.permutations(generated_orders)))

    return ExecutionCase(
        cash=cash,
        positions=positions,
        orders=orders,
        opens=opens,
        actions=actions,
        commission_bps=Decimal(draw(st.integers(min_value=0, max_value=30_000))),
        slippage_bps=Decimal(draw(st.integers(min_value=0, max_value=20_000))),
    )


def _reference_execution(case: ExecutionCase) -> ReferenceResult:
    """Execute without importing or calling any production arithmetic helper."""

    remaining = {
        str(order["id"]): cast(int, order["amount"]) - cast(int, order.get("filled", 0))
        for order in case.orders
    }
    by_id = {str(order["id"]): order for order in case.orders}
    positions = dict(case.positions)
    cash = _money(case.cash)
    fills: list[ReferenceFill] = []
    unfilled: list[ReferenceUnfilled] = []

    def sort_rank(order: Mapping[str, object]) -> int:
        rank = order.get("decision_rank")
        return 2**31 - 1 if rank is None else cast(int, rank)

    ordered_ids = [
        *sorted(
            (order_id for order_id, quantity in remaining.items() if quantity < 0),
            key=lambda order_id: (str(by_id[order_id]["symbol"]), order_id),
        ),
        *sorted(
            (order_id for order_id, quantity in remaining.items() if quantity > 0),
            key=lambda order_id: (
                sort_rank(by_id[order_id]),
                str(by_id[order_id]["symbol"]),
                order_id,
            ),
        ),
    ]

    for order_id in ordered_ids:
        order = by_id[order_id]
        symbol = str(order["symbol"])
        requested_remaining = remaining[order_id]
        base_open = case.opens.get(symbol)
        if base_open is None:
            unfilled.append(
                ReferenceUnfilled(
                    order_id,
                    symbol,
                    requested_remaining,
                    "missing_or_non_positive_adjusted_open",
                )
            )
            continue
        fill_price = _adverse_price(base_open, requested_remaining, case.slippage_bps)
        if fill_price is None:
            unfilled.append(
                ReferenceUnfilled(
                    order_id, symbol, requested_remaining, "invalid_adjusted_open"
                )
            )
            continue

        if requested_remaining < 0:
            held = positions.get(symbol, 0)
            requested = min(abs(requested_remaining), held)
            quantity = _largest_affordable_sell(
                cash, fill_price, requested, held, case.commission_bps
            )
            if quantity == 0:
                unfilled.append(
                    ReferenceUnfilled(
                        order_id,
                        symbol,
                        requested_remaining,
                        "position_or_commission_cash_constraint",
                    )
                )
                continue
            signed_quantity = -quantity
            commission = _commission(signed_quantity, fill_price, case.commission_bps)
            cash = _money(cash + _money(quantity * fill_price) - commission)
            positions[symbol] = held - quantity
        else:
            quantity = _largest_affordable_buy(
                cash, fill_price, requested_remaining, case.commission_bps
            )
            if quantity == 0:
                unfilled.append(
                    ReferenceUnfilled(
                        order_id,
                        symbol,
                        requested_remaining,
                        "cash_constraint_including_commission",
                    )
                )
                continue
            signed_quantity = quantity
            commission = _commission(signed_quantity, fill_price, case.commission_bps)
            cash = _money(cash - _money(quantity * fill_price) - commission)
            positions[symbol] = positions.get(symbol, 0) + quantity

        fills.append(
            ReferenceFill(
                order_id=order_id,
                symbol=symbol,
                quantity=signed_quantity,
                base_open=_money(base_open),
                fill_price=fill_price,
                gross_notional=_money(abs(signed_quantity) * fill_price),
                commission=commission,
                slippage_cost=_money(
                    abs(fill_price - _money(base_open)) * abs(signed_quantity)
                ),
            )
        )
        remainder = requested_remaining - signed_quantity
        remaining[order_id] = remainder
        if remainder:
            reason = (
                "position_constraint"
                if requested_remaining < 0 and positions.get(symbol, 0) == 0
                else "cash_constraint_including_commission"
            )
            unfilled.append(ReferenceUnfilled(order_id, symbol, remainder, reason))

    return ReferenceResult(
        cash=cash,
        positions={
            symbol: quantity for symbol, quantity in positions.items() if quantity
        },
        fills=tuple(fills),
        unfilled=tuple(unfilled),
    )


def _marked_state(
    cash: Decimal, positions: Mapping[str, int], actions: tuple[MarkAction, ...]
) -> tuple[Decimal, Decimal, Decimal]:
    """Return gross value, equity, and leverage for action-effective marks."""

    marked_values = []
    for action in actions:
        quantity = positions.get(action.symbol, 0)
        marked_values.append(_money(quantity * action.mark_price))
        assert action.cash_return == _ZERO
    gross = _money(sum(marked_values, _ZERO))
    equity = _money(cash + gross)
    assert equity > _ZERO
    leverage = gross / equity
    return gross, equity, leverage


# Feature: quantitative-research-platform, Property 10: Whole-share execution and accounting invariants
# Validates: Requirements 9.2, 9.6–9.18, 17.8, 17.21–17.22
@settings(max_examples=100, deadline=None)
@given(case=execution_cases())
def test_whole_share_execution_and_accounting_invariants(case: ExecutionCase) -> None:
    """Every fill and action-effective mark agrees with an independent ledger."""

    expected = _reference_execution(case)
    blotter = CashSafeOpenBlotter(
        commission_bps=case.commission_bps,
        slippage_bps=case.slippage_bps,
    )
    actual = blotter.execute_orders(
        case.orders,
        opens=cast(Mapping[object, object], dict(case.opens)),
        cash=case.cash,
        positions=cast(Mapping[object, object], dict(case.positions)),
        session=SESSION,
    )

    assert actual.cash_balance == expected.cash
    assert dict(actual.positions) == dict(expected.positions)
    assert len(actual.fills) == len(expected.fills)
    assert len(actual.unfilled_orders) == len(expected.unfilled)

    for actual_fill, reference_fill in zip(actual.fills, expected.fills, strict=True):
        assert actual_fill.order_id == reference_fill.order_id
        assert actual_fill.symbol == reference_fill.symbol
        assert isinstance(actual_fill.quantity, int) and actual_fill.quantity != 0
        assert actual_fill.quantity == reference_fill.quantity
        assert actual_fill.base_adjusted_open == reference_fill.base_open
        assert actual_fill.fill_price == reference_fill.fill_price
        assert actual_fill.gross_notional == reference_fill.gross_notional
        assert actual_fill.commission == reference_fill.commission
        assert actual_fill.slippage_cost == reference_fill.slippage_cost
        if actual_fill.quantity > 0:
            assert actual_fill.fill_price >= actual_fill.base_adjusted_open
        else:
            assert actual_fill.fill_price <= actual_fill.base_adjusted_open

    for actual_unfilled, reference_unfilled in zip(
        actual.unfilled_orders, expected.unfilled, strict=True
    ):
        assert actual_unfilled.order_id == reference_unfilled.order_id
        assert actual_unfilled.symbol == reference_unfilled.symbol
        assert actual_unfilled.quantity == reference_unfilled.quantity
        assert actual_unfilled.reason == reference_unfilled.reason
        assert actual_unfilled.quantity != 0
    assert all(
        error.symbol == unfilled.symbol
        for error, unfilled in zip(
            actual.actionable_errors, actual.unfilled_orders, strict=True
        )
    )

    # The independent model explicitly checks the largest affordable buy after
    # every preceding sell and buy, including commission rates above 100%.
    buy_fills = {fill.order_id: fill for fill in expected.fills if fill.quantity > 0}
    assert all(fill.quantity > 0 for fill in actual.fills if fill.order_id in buy_fills)
    assert all(
        isinstance(quantity, int) and quantity >= 0
        for quantity in actual.positions.values()
    )
    assert all(isinstance(cast(int, order["amount"]), int) for order in case.orders)
    assert all(isinstance(fill.quantity, int) for fill in actual.fills)

    expected_gross, expected_equity, expected_leverage = _marked_state(
        expected.cash, expected.positions, case.actions
    )
    actual_gross, actual_equity, actual_leverage = _marked_state(
        actual.cash_balance, actual.positions, case.actions
    )
    assert actual_gross == expected_gross
    assert abs(actual_equity - expected_equity) <= Decimal("0.01")
    assert actual_equity == expected_equity
    assert Decimal("0") <= expected_leverage <= Decimal("1")
    assert Decimal("0") <= actual_leverage <= Decimal("1")
    assert actual.cash_balance >= _ZERO

    # Marking is not a cash return: only fills change cash, never actions/marks.
    assert actual.cash_balance == expected.cash
    assert all(action.cash_return == _ZERO for action in case.actions)
