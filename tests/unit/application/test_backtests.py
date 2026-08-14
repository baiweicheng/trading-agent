"""Focused accounting and orchestration tests for the backtest application boundary."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

from quant_research_platform.application.backtests import (
    BacktestRequest,
    BacktestService,
    audit_core_output,
)
from quant_research_platform.config.models import ResolvedConfig
from quant_research_platform.domain.errors import Err, Ok
from quant_research_platform.domain.execution import (
    OrderRecord,
    OrderStatus,
    Position,
    deterministic_fill_id,
    deterministic_order_id,
)
from quant_research_platform.domain.market import DateRange

SNAPSHOT_ID = "snap_" + "a" * 64
RANGE = DateRange(date(2024, 1, 2), date(2024, 1, 3))
CONFIG = SimpleNamespace(
    execution=SimpleNamespace(
        initial_equity_usd=Decimal("100000"),
        commission_bps=Decimal("5"),
        slippage_bps=Decimal("10"),
    )
)


def _state(
    session: date,
    *,
    cash: Decimal = Decimal("100000.000000"),
    positions: tuple[object, ...] = (),
) -> SimpleNamespace:
    gross = sum(
        (cast(Any, position).market_value for position in positions),
        Decimal("0"),
    )
    equity = cash + gross
    return SimpleNamespace(
        session=session,
        cash_balance=cash,
        positions=positions,
        gross_exposure=gross,
        portfolio_equity=equity,
        leverage=gross / equity,
    )


def _domain_unfilled_order() -> OrderRecord:
    return OrderRecord(
        order_id=deterministic_order_id(
            signal_session=RANGE.start,
            execution_session=RANGE.end,
            symbol="AAPL",
            requested_quantity=10,
            ordinal=0,
        ),
        signal_session=RANGE.start,
        execution_session=RANGE.end,
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
        status=OrderStatus.UNFILLED,
        unfilled_reason="missing_adjusted_open",
    )


def _output(
    *,
    orders: tuple[object, ...] = (),
    fills: tuple[object, ...] = (),
    states: tuple[object, ...] | None = None,
    returns: tuple[object, ...] | None = None,
) -> SimpleNamespace:
    states = states or (_state(RANGE.start), _state(RANGE.end))
    returns = returns or tuple(
        SimpleNamespace(session=cast(Any, state).session, return_value=Decimal("0"))
        for state in states
    )
    return SimpleNamespace(
        orders=orders,
        fills=fills,
        portfolio_states=states,
        daily_returns=returns,
    )


def _buy_order(status: OrderStatus = OrderStatus.FILLED) -> SimpleNamespace:
    order_id = deterministic_order_id(
        signal_session=RANGE.start,
        execution_session=RANGE.end,
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
    )
    return SimpleNamespace(
        order_id=order_id,
        signal_session=RANGE.start,
        execution_session=RANGE.end,
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
        status=status,
        unfilled_reason=None,
    )


def test_audit_reconciles_exact_fill_costs_cash_and_positions() -> None:
    order = _buy_order()
    fill_price = Decimal("100.100000")
    gross = Decimal("1001.000000")
    commission = Decimal("0.500500")
    fill = SimpleNamespace(
        fill_id=deterministic_fill_id(
            order_id=order.order_id,
            symbol="AAPL",
            session=RANGE.end,
            quantity=10,
            ordinal=0,
        ),
        order_id=order.order_id,
        symbol="AAPL",
        session=RANGE.end,
        quantity=10,
        ordinal=0,
        base_adjusted_open=Decimal("100.000000"),
        fill_price=fill_price,
        gross_notional=gross,
        commission=commission,
        slippage_cost=Decimal("1.000000"),
    )
    position = Position(
        symbol="AAPL",
        quantity=10,
        mark_price=fill_price,
        market_value=gross,
    )
    states = (
        _state(RANGE.start),
        _state(
            RANGE.end,
            cash=Decimal("98998.499500"),
            positions=(position,),
        ),
    )
    returns = (
        SimpleNamespace(session=RANGE.start, return_value=Decimal("0")),
        SimpleNamespace(session=RANGE.end, return_value=Decimal("0")),
    )

    result = audit_core_output(
        _output(orders=(order,), fills=(fill,), states=states, returns=returns),
        CONFIG,
        evaluation_range=RANGE,
    )

    assert isinstance(result, Ok)
    assert result.value.unfilled_orders == ()


def test_audit_stops_at_first_cost_invariant_and_preserves_diagnostic() -> None:
    order = _buy_order()
    fill = SimpleNamespace(
        fill_id="fill_bad",
        order_id=order.order_id,
        symbol="AAPL",
        session=RANGE.end,
        quantity=10,
        ordinal=0,
        base_adjusted_open=Decimal("100.000000"),
        fill_price=Decimal("100.100000"),
        gross_notional=Decimal("1001.000000"),
        commission=Decimal("999.000000"),
        slippage_cost=Decimal("1.000000"),
    )

    result = audit_core_output(
        _output(orders=(order,), fills=(fill,)),
        CONFIG,
        evaluation_range=RANGE,
    )

    assert isinstance(result, Err)
    assert len(result.errors) == 1
    assert result.errors[0].field_path == "fills.commission"
    assert result.errors[0].category.value == "backtest.invariant"


def test_unfilled_order_is_successful_but_disclosed() -> None:
    order = _domain_unfilled_order()

    result = audit_core_output(
        _output(orders=(order,)),
        CONFIG,
        evaluation_range=RANGE,
    )

    assert isinstance(result, Ok)
    assert result.value.passed
    assert result.value.unfilled_orders == (order,)
    assert result.value.unfilled_diagnostics[0].symbol == "AAPL"


class _Tracker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def allocate_run(self, **kwargs: object) -> dict[str, object]:
        self.events.append("allocate")
        assert kwargs["snapshot_id"] == SNAPSHOT_ID
        return {"run_id": "run-1"}

    def finalize_success(self, **kwargs: object) -> None:
        self.events.append("finalize")
        assert kwargs["audited_output"] is kwargs["output"]

    def finalize_failure(self, **kwargs: object) -> None:
        self.events.append("failure")


class _SnapshotManager:
    def __init__(self, events: list[str], snapshot_id: str = SNAPSHOT_ID) -> None:
        self.events = events
        self.snapshot_id = snapshot_id

    def open_verified(self, snapshot_id: str) -> object:
        self.events.append("snapshot")
        return SimpleNamespace(
            snapshot_id=self.snapshot_id,
            available=True,
            comparison_ready=True,
            covered_range=RANGE,
        )


class _Bundle:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def materialize(self, snapshot: object) -> object:
        self.events.append("bundle")
        return SimpleNamespace(
            snapshot_id=cast(SimpleNamespace, snapshot).snapshot_id,
            bundle_name="exact",
        )


class _Engine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def run(self, **kwargs: object) -> object:
        self.events.append("engine")
        return Ok(_output())


class _Evaluator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def evaluate(self, **kwargs: object) -> object:
        self.events.append("evaluate")
        assert kwargs["audited_output"] is kwargs["output"]
        return {"evaluated": True}


def _config_for_service() -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {},
            "data": {
                "universe": ["AAPL"],
                "requested_range": {
                    "start": RANGE.start,
                    "end": RANGE.end,
                },
            },
            "runtime": {"deterministic_seed": 7},
        }
    )


def test_service_allocates_before_pinning_and_passes_audited_output() -> None:
    events: list[str] = []
    service = BacktestService(
        tracker=_Tracker(events),
        snapshot_manager=_SnapshotManager(events),
        bundle_adapter=_Bundle(events),
        engine=_Engine(events),
        evaluator=_Evaluator(events),
    )

    result = service.run(
        BacktestRequest(SNAPSHOT_ID, RANGE),
        config=_config_for_service(),
    )

    assert isinstance(result, Ok)
    assert events[:2] == ["allocate", "snapshot"]
    assert events[-2:] == ["evaluate", "finalize"]


def test_service_rejects_snapshot_switch_without_materializing_bundle() -> None:
    events: list[str] = []
    service = BacktestService(
        tracker=_Tracker(events),
        snapshot_manager=_SnapshotManager(events, "snap_" + "b" * 64),
        bundle_adapter=_Bundle(events),
        engine=_Engine(events),
    )

    result = service.run(
        BacktestRequest(SNAPSHOT_ID, RANGE),
        config=_config_for_service(),
    )

    assert isinstance(result, Err)
    assert events == ["allocate", "snapshot", "failure"]
    assert "bundle" not in events
    assert result.errors[0].field_path == "snapshot_id"
