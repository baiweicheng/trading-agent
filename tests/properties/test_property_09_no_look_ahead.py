"""Property tests for prefix-based no-look-ahead behavior."""

# ruff: noqa: E501, I001

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from collections.abc import Callable
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.decisions import CausalDecisionDelivery
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import Ok
from quant_research_platform.infrastructure.xnys_calendar import XNYSCalendar
from quant_research_platform.infrastructure.zipline_engine import (
    BacktestEngine,
    CashSafeExecutionResult,
    CashSafeOpenBlotter,
)


_COMPLETED_AT = datetime(2025, 1, 31, tzinfo=UTC)
_CALENDAR = XNYSCalendar()
_SIGNAL_SESSION = date(2024, 12, 31)
_CALENDAR_SESSIONS = _CALENDAR.sessions(
    date(2023, 1, 1),
    date(2025, 1, 10),
    completed_at=_COMPLETED_AT,
)
_PREFIX_SESSIONS = tuple(
    session for session in _CALENDAR_SESSIONS if session <= _SIGNAL_SESSION
)[-254:]
assert len(_PREFIX_SESSIONS) == 254
assert _PREFIX_SESSIONS[-1] == _SIGNAL_SESSION
_FILL_SESSION = _CALENDAR.next_session(_SIGNAL_SESSION)
_POST_SESSIONS = tuple(
    session for session in _CALENDAR_SESSIONS if session > _FILL_SESSION
)[:3]
assert len(_POST_SESSIONS) == 3
_SESSIONS = (*_PREFIX_SESSIONS, _FILL_SESSION, *_POST_SESSIONS)
_SIGNAL_INDEX = len(_PREFIX_SESSIONS) - 1
_FILL_INDEX = _SIGNAL_INDEX + 1
_SYMBOL_POOL = ("AAPL", "MSFT", "PG")
_SNAPSHOT = SimpleNamespace(snapshot_id="snap_" + "a" * 64)


@dataclass(frozen=True)
class PrefixCase:
    """One pair of valid histories and the first session allowed to differ."""

    universe: tuple[str, ...]
    score_ticks: tuple[int, ...]
    suffix_multiplier: int
    boundary: str
    commission_bps: Decimal
    slippage_bps: Decimal

    @property
    def boundary_session(self) -> date:
        return _SIGNAL_SESSION if self.boundary == "signal" else _FILL_SESSION

    @property
    def first_changed_index(self) -> int:
        return _SIGNAL_INDEX + 1 if self.boundary == "signal" else _FILL_INDEX + 1

    @property
    def first_changed_session(self) -> date:
        return _SESSIONS[self.first_changed_index]


def _row_value(
    case: PrefixCase,
    symbol_index: int,
    session_index: int,
    *,
    mutated: bool,
) -> dict[str, object]:
    """Build one positive, tradable normalized-bar-like row."""

    symbol = case.universe[symbol_index]
    session = _SESSIONS[session_index]
    long_index = 1
    short_index = _SIGNAL_INDEX - 21
    base = Decimal(100 + symbol_index * 10)
    adjusted_close = base
    if session_index == long_index:
        adjusted_close = Decimal("100")
    elif session_index == short_index:
        adjusted_close = Decimal("100") + Decimal(case.score_ticks[symbol_index]) / Decimal("10")
    sizing_close = base
    if session_index == _SIGNAL_INDEX:
        sizing_close = base
    execution_open = Decimal(95 + symbol_index * 10)

    if mutated and session_index >= case.first_changed_index:
        adjusted_close *= case.suffix_multiplier
        sizing_close *= case.suffix_multiplier
        execution_open *= case.suffix_multiplier

    return {
        "symbol": symbol,
        "session": session,
        "adjusted_close": adjusted_close,
        "sizing_adjusted_close": sizing_close,
        "execution_adjusted_open": execution_open,
        "tradable": True,
    }


def _history(case: PrefixCase, *, mutated: bool) -> tuple[dict[str, object], ...]:
    return tuple(
        _row_value(case, symbol_index, session_index, mutated=mutated)
        for symbol_index in range(len(case.universe))
        for session_index in range(len(_SESSIONS))
    )


@st.composite
def prefix_cases(draw: st.DrawFn) -> PrefixCase:
    """Generate valid pairs with arbitrary positive post-boundary mutations."""

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
    return PrefixCase(
        universe=universe,
        score_ticks=tuple(
            draw(
                st.lists(
                    st.integers(min_value=-900, max_value=1000),
                    min_size=len(universe),
                    max_size=len(universe),
                )
            )
        ),
        suffix_multiplier=draw(st.integers(min_value=2, max_value=9)),
        boundary=draw(st.sampled_from(("signal", "fill"))),
        commission_bps=Decimal(draw(st.integers(min_value=0, max_value=50))),
        slippage_bps=Decimal(draw(st.integers(min_value=0, max_value=50))),
    )


def _row_projection(
    rows: tuple[dict[str, object], ...],
    through: date | None = None,
) -> list[dict[str, object]]:
    selected = rows if through is None else tuple(row for row in rows if row["session"] <= through)
    return [
        {
            "adjusted_close": row["adjusted_close"],
            "execution_adjusted_open": row["execution_adjusted_open"],
            "session": row["session"],
            "sizing_adjusted_close": row["sizing_adjusted_close"],
            "symbol": row["symbol"],
        }
        for row in sorted(selected, key=lambda item: (item["session"], item["symbol"]))
    ]


def _checksum(role: str, rows: list[dict[str, object]]) -> str:
    return sha256_bytes(canonical_json({"role": role, "rows": rows}))


def _fill_projection(result: CashSafeExecutionResult) -> list[dict[str, object]]:
    return [
        {
            "base_adjusted_open": fill.base_adjusted_open,
            "commission": fill.commission,
            "fill_price": fill.fill_price,
            "gross_notional": fill.gross_notional,
            "order_id": fill.order_id,
            "quantity": fill.quantity,
            "session": fill.session,
            "slippage_cost": fill.slippage_cost,
            "symbol": fill.symbol,
        }
        for fill in result.fills
    ]


def _valuation_projection(states: tuple[object, ...]) -> list[dict[str, object]]:
    return [state.to_serializable() for state in states]


def _prefix_checksum(
    role: str,
    values: tuple[object, ...],
    session_getter: Callable[[object], date],
    through: date,
    serializer: Callable[[object], object],
) -> str:
    rows = [
        serializer(value)
        for value in values
        if session_getter(value) <= through
    ]
    return _checksum(role, rows)  # type: ignore[arg-type]


def _first_difference_session(
    left: tuple[object, ...],
    right: tuple[object, ...],
    session_getter: Callable[[object], date],
    serializer: Callable[[object], object],
) -> date | None:
    """Return the first session whose canonical row set differs."""

    left_by_session: dict[date, list[object]] = {}
    right_by_session: dict[date, list[object]] = {}
    for value in left:
        left_by_session.setdefault(session_getter(value), []).append(serializer(value))
    for value in right:
        right_by_session.setdefault(session_getter(value), []).append(serializer(value))
    for session in sorted(set(left_by_session) | set(right_by_session)):
        if canonical_json(left_by_session.get(session, [])) != canonical_json(
            right_by_session.get(session, [])
        ):
            return session
    return None


def _orders_for_delivery(delivery: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "amount": order.requested_quantity,
            "decision_rank": order.decision_rank,
            "execution_session": order.execution_session,
            "filled": 0,
            "id": order.order_id,
            "signal_session": order.signal_session,
            "symbol": order.symbol,
        }
        for order in delivery.orders
    )


def _execute_delivery(
    case: PrefixCase,
    rows: tuple[dict[str, object], ...],
) -> tuple[object, CashSafeExecutionResult, tuple[object, ...]]:
    """Run decision, next-open fill, and engine valuation seams for one history."""

    class PrefixReader:
        def __init__(self, source_rows: tuple[dict[str, object], ...]) -> None:
            self.source_rows = source_rows
            self.end_sessions: list[date] = []

        def read_history(
            self,
            snapshot_handle: object,
            *,
            symbols: tuple[str, ...],
            end_session: date,
            fields: tuple[str, ...],
            start_session: date,
        ) -> tuple[dict[str, object], ...]:
            del snapshot_handle, fields, start_session
            assert symbols == case.universe
            self.end_sessions.append(end_session)
            assert end_session <= _SIGNAL_SESSION
            return tuple(
                row
                for row in self.source_rows
                if row["symbol"] in symbols and row["session"] <= end_session
            )

    reader = PrefixReader(rows)
    delivery_result = CausalDecisionDelivery(
        reader,
        calendar=_CALENDAR,
    ).deliver(
        _SNAPSHOT,
        _SIGNAL_SESSION,
        {"cash_balance": Decimal("100000"), "positions": {}},
        universe=case.universe,
        position_count=len(case.universe),
        execution_session=_FILL_SESSION,
    )
    assert isinstance(delivery_result, Ok), delivery_result
    delivery = delivery_result.value
    assert reader.end_sessions == [_SIGNAL_SESSION]
    assert delivery.signal_session == _SIGNAL_SESSION

    order_rows = _orders_for_delivery(delivery)
    opens = {
        row["symbol"]: row["execution_adjusted_open"]
        for row in rows
        if row["session"] == _FILL_SESSION
    }
    blotter = CashSafeOpenBlotter(
        commission_bps=case.commission_bps,
        slippage_bps=case.slippage_bps,
    )
    fills = blotter.execute_orders(
        order_rows,
        opens=opens,
        cash=delivery.marked_equity,
        positions={},
        session=_FILL_SESSION,
    )

    rows_by_key = {(row["symbol"], row["session"]): row for row in rows}
    valuation_sessions = (_SIGNAL_SESSION, _FILL_SESSION, *_POST_SESSIONS)
    states: list[object] = []
    for session_index, session in enumerate(valuation_sessions):
        if session_index == 0:
            cash = Decimal("100000.000000")
            positions: dict[str, int] = {}
        else:
            cash = fills.cash_balance
            positions = dict(fills.positions)
        position_rows = [
            {
                "amount": quantity,
                "last_sale_price": rows_by_key[(symbol, session)]["sizing_adjusted_close"],
                "symbol": symbol,
            }
            for symbol, quantity in sorted(positions.items())
            if quantity
        ]
        gross = sum(
            Decimal(str(item["amount"])) * Decimal(str(item["last_sale_price"]))
            for item in position_rows
        )
        equity = cash + gross
        states.append(
            BacktestEngine._portfolio_state(
                session,
                {
                    "ending_cash": cash,
                    "positions": position_rows,
                    "portfolio_value": equity,
                },
                object(),
            )
        )
    return delivery, fills, tuple(states)


# Feature: quantitative-research-platform, Property 9: Prefix equivalence enforces no look-ahead
# Validates: Requirements 9.3–9.5, 9.19–9.21, 17.9, 17.19–17.20
@settings(max_examples=100, deadline=None)
@given(case=prefix_cases())
def test_prefix_equivalence_enforces_no_look_ahead(case: PrefixCase) -> None:
    """Post-boundary mutations cannot alter an earlier scientific output prefix."""

    baseline_rows = _history(case, mutated=False)
    changed_rows = _history(case, mutated=True)
    boundary = case.boundary_session
    first_changed = case.first_changed_session

    assert _checksum("history", _row_projection(baseline_rows, boundary)) == _checksum(
        "history", _row_projection(changed_rows, boundary)
    )
    assert _checksum("history", _row_projection(baseline_rows)) != _checksum(
        "history", _row_projection(changed_rows)
    )

    baseline_delivery, baseline_fills, baseline_states = _execute_delivery(case, baseline_rows)
    changed_delivery, changed_fills, changed_states = _execute_delivery(case, changed_rows)

    def decision_rows(value: object) -> object:
        return value.to_serializable()

    def order_rows(value: object) -> object:
        return value.to_serializable()

    assert _prefix_checksum(
        "strategy_decisions",
        tuple(baseline_delivery.decisions),
        lambda value: value.signal_session,
        _SIGNAL_SESSION,
        decision_rows,
    ) == _prefix_checksum(
        "strategy_decisions",
        tuple(changed_delivery.decisions),
        lambda value: value.signal_session,
        _SIGNAL_SESSION,
        decision_rows,
    )
    assert _prefix_checksum(
        "orders",
        tuple(baseline_delivery.orders),
        lambda value: value.signal_session,
        _SIGNAL_SESSION,
        order_rows,
    ) == _prefix_checksum(
        "orders",
        tuple(changed_delivery.orders),
        lambda value: value.signal_session,
        _SIGNAL_SESSION,
        order_rows,
    )

    fill_boundary = _SIGNAL_SESSION if case.boundary == "signal" else _FILL_SESSION
    assert _prefix_checksum(
        "fills",
        tuple(baseline_fills.fills),
        lambda value: value.session,
        fill_boundary,
        lambda value: {
            "base_adjusted_open": value.base_adjusted_open,
            "commission": value.commission,
            "fill_price": value.fill_price,
            "gross_notional": value.gross_notional,
            "order_id": value.order_id,
            "quantity": value.quantity,
            "session": value.session,
            "slippage_cost": value.slippage_cost,
            "symbol": value.symbol,
        },
    ) == _prefix_checksum(
        "fills",
        tuple(changed_fills.fills),
        lambda value: value.session,
        fill_boundary,
        lambda value: {
            "base_adjusted_open": value.base_adjusted_open,
            "commission": value.commission,
            "fill_price": value.fill_price,
            "gross_notional": value.gross_notional,
            "order_id": value.order_id,
            "quantity": value.quantity,
            "session": value.session,
            "slippage_cost": value.slippage_cost,
            "symbol": value.symbol,
        },
    )
    assert _prefix_checksum(
        "portfolio_valuations",
        baseline_states,
        lambda value: value.session,
        fill_boundary,
        lambda value: value.to_serializable(),
    ) == _prefix_checksum(
        "portfolio_valuations",
        changed_states,
        lambda value: value.session,
        fill_boundary,
        lambda value: value.to_serializable(),
    )

    assert _first_difference_session(
        tuple(baseline_delivery.decisions),
        tuple(changed_delivery.decisions),
        lambda value: value.signal_session,
        decision_rows,
    ) is None
    assert _first_difference_session(
        tuple(baseline_delivery.orders),
        tuple(changed_delivery.orders),
        lambda value: value.signal_session,
        order_rows,
    ) is None
    assert (
        _first_difference_session(
            tuple(baseline_fills.fills),
            tuple(changed_fills.fills),
            lambda value: value.session,
            lambda value: {
                "base_adjusted_open": value.base_adjusted_open,
                "commission": value.commission,
                "fill_price": value.fill_price,
                "gross_notional": value.gross_notional,
                "order_id": value.order_id,
                "quantity": value.quantity,
                "session": value.session,
                "slippage_cost": value.slippage_cost,
                "symbol": value.symbol,
            },
        )
        is None
        or _first_difference_session(
            tuple(baseline_fills.fills),
            tuple(changed_fills.fills),
            lambda value: value.session,
            lambda value: {
                "base_adjusted_open": value.base_adjusted_open,
                "commission": value.commission,
                "fill_price": value.fill_price,
                "gross_notional": value.gross_notional,
                "order_id": value.order_id,
                "quantity": value.quantity,
                "session": value.session,
                "slippage_cost": value.slippage_cost,
                "symbol": value.symbol,
            },
        )
        >= first_changed
    )
    valuation_difference = _first_difference_session(
        baseline_states,
        changed_states,
        lambda value: value.session,
        lambda value: value.to_serializable(),
    )
    assert valuation_difference is None or valuation_difference >= first_changed

    if case.boundary == "signal":
        assert baseline_fills.fills
        assert baseline_fills.fills != changed_fills.fills
        assert valuation_difference == _FILL_SESSION
    else:
        assert baseline_fills.fills == changed_fills.fills
        assert valuation_difference == first_changed

    assert _fill_projection(baseline_fills)
    assert _valuation_projection(baseline_states)
