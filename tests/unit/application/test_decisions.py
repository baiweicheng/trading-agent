"""Focused causal decision-delivery and order-intent examples."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

from quant_research_platform.application.decisions import (
    CausalDecisionDelivery,
    DecisionBook,
    generate_order_intents,
)
from quant_research_platform.domain.errors import Ok
from quant_research_platform.domain.strategy import StrategyExclusionReason

SNAPSHOT = SimpleNamespace(snapshot_id="snap_" + "a" * 64)


class Calendar:
    def next_session(self, session: date) -> date:
        return session + timedelta(days=1)


def _history(count: int = 254) -> tuple[dict[str, object], ...]:
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(count))
    rows: list[dict[str, object]] = []
    for symbol, long_close, short_close, sizing_close in (
        ("AAPL", Decimal("100"), Decimal("123"), Decimal("123.45")),
        ("MSFT", Decimal("100"), Decimal("110"), Decimal("50")),
    ):
        for index, session in enumerate(sessions):
            close = (
                long_close
                if index == 1
                else short_close
                if index == count - 22
                else Decimal("100")
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "adjusted_close": close,
                    "sizing_adjusted_close": sizing_close,
                }
            )
    return tuple(rows)


def test_delivery_reads_only_through_signal_and_floors_targets() -> None:
    signal = date(2024, 9, 11)
    calls: list[date] = []

    class Reader:
        def read_history(
            self,
            snapshot: object,
            *,
            symbols: tuple[str, ...],
            end_session: date,
            fields: tuple[str, ...],
        ):
            assert snapshot is SNAPSHOT
            assert symbols == ("AAPL", "MSFT")
            assert "sizing_adjusted_close" in fields
            calls.append(end_session)
            assert end_session <= signal
            return _history()

    result = CausalDecisionDelivery(Reader(), calendar=Calendar()).deliver(
        SNAPSHOT,
        signal,
        {"cash_balance": Decimal("1000"), "positions": {"MSFT": 10}},
        universe=("AAPL", "MSFT"),
        position_count=1,
    )

    assert isinstance(result, Ok)
    assert calls == [signal]
    assert result.value.marked_equity == Decimal("1500")
    # AAPL receives floor(1500 / 123.45) = 12 shares; MSFT is fully liquidated.
    assert [
        (item.symbol, item.requested_quantity) for item in result.value.order_intents
    ] == [
        ("MSFT", -10),
        ("AAPL", 12),
    ]
    assert result.value.run_inputs.endpoint_offsets == (252, 21)
    assert result.value.run_inputs.policy_version == "causal_forward_v1"


def test_decision_book_reveals_only_exact_signal_session_and_warmup_has_no_orders() -> (
    None
):
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(253))
    rows = tuple(
        {
            "symbol": "AAPL",
            "session": session,
            "adjusted_close": Decimal("100"),
            "sizing_adjusted_close": Decimal("100"),
        }
        for session in sessions
    )
    result = CausalDecisionDelivery(calendar=Calendar()).deliver(
        SNAPSHOT,
        sessions[-1],
        {"cash_balance": Decimal("100000"), "positions": {"AAPL": 10}},
        universe=("AAPL",),
        position_count=1,
        history=rows,
    )

    assert isinstance(result, Ok)
    assert result.value.order_intents == ()
    assert all(
        item.exclusion_reason is StrategyExclusionReason.WARM_UP_INCOMPLETE
        for item in result.value.decisions
    )
    book: DecisionBook = result.value.decision_book
    assert book.reveal(sessions[-1]) == result.value.decisions
    assert book.reveal(sessions[-2]) == ()
    assert book.reveal(sessions[-1] + timedelta(days=1)) == ()


def test_standalone_order_intents_have_repeatable_scientific_ids() -> None:
    history = _history()
    service = CausalDecisionDelivery(calendar=Calendar())
    prepared = service.deliver(
        SNAPSHOT,
        date(2024, 9, 11),
        {"cash_balance": Decimal("1000"), "positions": {"MSFT": 10}},
        universe=("AAPL", "MSFT"),
        position_count=1,
        history=history,
    )
    assert isinstance(prepared, Ok)
    decisions = prepared.value.decisions
    prices = {
        ("AAPL", date(2024, 9, 11)): Decimal("123.45"),
        ("MSFT", date(2024, 9, 11)): Decimal("50"),
    }
    first = generate_order_intents(
        decisions,
        {"cash_balance": Decimal("1000"), "positions": {"MSFT": 10}},
        prices,
        date(2024, 9, 12),
    )
    second = generate_order_intents(
        decisions,
        {"cash_balance": Decimal("1000"), "positions": {"MSFT": 10}},
        prices,
        date(2024, 9, 12),
    )
    assert first == second
    assert all(item.to_order_record().order_id == item.order_id for item in first)
