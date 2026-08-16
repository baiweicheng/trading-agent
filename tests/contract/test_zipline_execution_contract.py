"""Pinned Zipline execution, ledger, and overnight-gap contracts."""

from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from decimal import ROUND_FLOOR, Decimal
from importlib.metadata import version as installed_package_version
from pathlib import Path
from typing import Any

import pytest

from quant_research_platform.application.decisions import (
    DecisionBook,
    DecisionDeliveryResult,
    DecisionRunInputs,
    OrderIntent,
)
from quant_research_platform.config.models import ResolvedConfig
from quant_research_platform.domain.errors import Ok
from quant_research_platform.domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    OrderStatus,
    deterministic_order_id,
)
from quant_research_platform.domain.strategy import RationalWeight, StrategyDecision
from quant_research_platform.infrastructure.zipline_bundle import ZiplineBundleAdapter
from quant_research_platform.infrastructure.zipline_engine import (
    BacktestEngine,
    CashSafeOpenBlotter,
)
from tests.golden.test_zipline_actions_golden import (
    _FixtureCalendar as _ActionCalendar,
)
from tests.golden.test_zipline_actions_golden import (
    _FixtureDataSource as _ActionDataSource,
)
from tests.golden.test_zipline_actions_golden import (
    _FixtureSnapshotManager as _ActionSnapshotManager,
)
from tests.golden.test_zipline_actions_golden import (
    _load_fixture as _load_action_fixture,
)
from tests.golden.test_zipline_actions_golden import (
    _snapshot as _action_snapshot,
)
from tests.integration.test_zipline_engine import (
    _bundle,
    _config,
    _delivery,
    _EngineCalendar,
)

_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "overnight_gap" / "execution_contract.json"
)


def _fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("overnight-gap fixture must be a mapping")
    return value


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _order(item: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "id": item["id"],
        "symbol": item["symbol"],
        "amount": item["amount"],
        "filled": 0,
    }
    if "decision_rank" in item:
        value["decision_rank"] = item["decision_rank"]
    return value


@pytest.mark.integration
def test_pinned_zipline_execution_extension_contract() -> None:
    """The custom seam must match the one locked Zipline Reloaded API."""

    import zipline
    from zipline.finance.blotter import SimulationBlotter

    fixture = _fixture()
    expected_version = fixture["pinned_zipline_version"]
    assert installed_package_version("zipline-reloaded") == expected_version
    assert zipline.__version__ == expected_version
    assert issubclass(CashSafeOpenBlotter, SimulationBlotter)
    signature = inspect.signature(CashSafeOpenBlotter.get_transactions)
    assert tuple(signature.parameters) == (
        "self",
        "bar_data",
    )
    assert tuple(inspect.signature(CashSafeOpenBlotter.order).parameters)[:3] == (
        "self",
        "asset",
        "amount",
    )


@pytest.mark.integration
def test_overnight_gap_planner_contract() -> None:
    """The same Decimal planner used by Zipline is pinned to reviewed cases."""

    cases = _fixture()["planner_cases"]
    for case_name, case in cases.items():
        blotter = CashSafeOpenBlotter(
            commission_bps=case["commission_bps"],
            slippage_bps=case["slippage_bps"],
        )
        result = blotter.execute_orders(
            [_order(item) for item in case["orders"]],
            opens={symbol: _decimal(price) for symbol, price in case["opens"].items()},
            cash=_decimal(case["cash"]),
            positions={
                symbol: int(quantity) for symbol, quantity in case["positions"].items()
            },
            session=date(2024, 1, 4),
        )

        expected_fills = case["fills"]
        assert len(result.fills) == len(expected_fills), case_name
        for actual, expected in zip(result.fills, expected_fills, strict=True):
            assert actual.symbol == expected["symbol"], case_name
            assert actual.quantity == expected["quantity"], case_name
            for field in (
                "base_open",
                "fill_price",
                "gross_notional",
                "commission",
                "slippage_cost",
            ):
                if field in expected:
                    actual_field = (
                        "base_adjusted_open" if field == "base_open" else field
                    )
                    assert getattr(actual, actual_field) == _decimal(expected[field]), (
                        case_name
                    )

        assert [
            {
                "symbol": item.symbol,
                "quantity": item.quantity,
                "reason": item.reason,
            }
            for item in result.unfilled_orders
        ] == case["unfilled"], case_name
        assert result.cash_balance == _decimal(case["cash_after"]), case_name
        assert dict(result.positions) == case["positions_after"], case_name
        assert all(isinstance(fill.quantity, int) for fill in result.fills)
        assert all(quantity >= 0 for quantity in result.positions.values())
        assert result.cash_balance >= 0


@pytest.mark.integration
def test_real_zipline_event_loop_has_next_open_timing_and_core_ledger_artifacts(
    tmp_path: Path,
) -> None:
    """The pinned event loop must expose the reviewed overnight-gap output."""

    calendar = _EngineCalendar()
    manifest, manager, locator = _bundle(tmp_path, calendar)
    delivery = _delivery(manifest.snapshot_id)
    progress: list[object] = []
    result = BacktestEngine(
        snapshot_manager=manager,
        decision_delivery=delivery,
        calendar=calendar,
    ).run(locator, None, _config(), progress=progress.append)

    assert isinstance(result, Ok)
    output = result.value
    expected = _fixture()["event_loop"]
    assert output.initial_equity == _decimal(expected["initial_cash"])
    assert output.portfolio_states[0].cash_balance == _decimal(
        expected["first_state_cash"]
    )
    assert output.portfolio_states[0].portfolio_equity == INITIAL_PORTFOLIO_EQUITY
    assert output.orders and output.fills and output.strategy_decisions

    order = output.orders[0]
    fill = output.fills[0]
    assert order.status is OrderStatus.FILLED
    assert order.signal_session == date.fromisoformat(expected["signal_session"])
    assert (
        order.execution_session
        == fill.session
        == date.fromisoformat(expected["execution_session"])
    )
    assert fill.session > order.signal_session
    assert fill.base_adjusted_open == _decimal(expected["execution_open"])
    assert fill.fill_price == _decimal(expected["fill_price"])
    assert fill.quantity == expected["fill_quantity"]
    assert fill.gross_notional == _decimal(expected["gross_notional"])
    assert fill.commission == _decimal(expected["commission"])
    assert fill.slippage_cost == _decimal(expected["slippage_cost"])

    # Every core ledger role is present and has immutable, whole-share values.
    assert len(output.portfolio_states) == len(output.daily_returns)
    assert len(progress) == len(output.portfolio_states)
    assert len({item.order_id for item in output.orders}) == len(output.orders)
    assert len({item.fill_id for item in output.fills}) == len(output.fills)
    assert all(
        item.quantity != 0 and isinstance(item.quantity, int) for item in output.fills
    )
    for state in output.portfolio_states:
        assert state.cash_balance >= 0
        assert state.leverage >= 0
        assert state.leverage <= 1
        assert state.portfolio_equity == state.cash_balance + state.gross_exposure
        assert all(position.quantity >= 0 for position in state.positions)
        assert all(isinstance(position.quantity, int) for position in state.positions)


class _ActionDecisionCalendar(_ActionCalendar):
    """Decision-calendar seam with a signal before the split/dividend rows."""

    def month_end_sessions(self, _start: date, _end: date) -> tuple[date, ...]:
        return (date(2024, 1, 2),)

    def next_session(self, session: date) -> date:
        return session + timedelta(days=1)


def _action_config() -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {},
            "data": {
                "requested_range": {"start": "2024-01-02", "end": "2024-01-04"},
                "universe": ["AAA"],
            },
            "strategy": {"position_count": 1},
            "execution": {"commission_bps": "0", "slippage_bps": "0"},
        }
    )


class _ActionDelivery:
    def deliver(
        self,
        snapshot: object,
        session: date,
        portfolio: object,
        **kwargs: object,
    ) -> Ok:
        del portfolio, kwargs
        snapshot_id = str(snapshot.snapshot_id)
        order_id = deterministic_order_id(
            signal_session=session,
            execution_session=date(2024, 1, 3),
            symbol="AAA",
            requested_quantity=5,
            ordinal=0,
        )
        decision = StrategyDecision(
            signal_session=session,
            symbol="AAA",
            endpoint_252_session=session,
            endpoint_252_close=Decimal("100"),
            endpoint_21_session=session,
            endpoint_21_close=Decimal("100"),
            momentum_score=Decimal("0"),
            eligible=True,
            rank=1,
            target_weight=RationalWeight(1, 1),
            exclusion_reason=None,
        )
        intent = OrderIntent(
            order_id=order_id,
            signal_session=session,
            execution_session=date(2024, 1, 3),
            symbol="AAA",
            requested_quantity=5,
            ordinal=0,
            target_shares=5,
            current_shares=0,
            sizing_price=Decimal("100"),
            portfolio_equity=INITIAL_PORTFOLIO_EQUITY,
            decision_rank=1,
        )
        inputs = DecisionRunInputs(position_count=1, snapshot_id=snapshot_id)
        return Ok(
            DecisionDeliveryResult(
                snapshot_id=snapshot_id,
                signal_session=session,
                decisions=(decision,),
                order_intents=(intent,),
                run_inputs=inputs,
                marked_equity=INITIAL_PORTFOLIO_EQUITY,
                decision_book=DecisionBook((decision,), inputs),
            )
        )


@pytest.mark.integration
def test_real_zipline_ledger_applies_split_and_dividend_once_to_actual_shares(
    tmp_path: Path,
) -> None:
    """A real ledger action must change shares/cash without adjusted-price doubling."""

    fixture = _load_action_fixture()
    case = fixture["cases"][0]
    manifest, rows = _action_snapshot(case, fixture)
    action_calendar = _ActionCalendar(fixture["calendar"]["schedule_checksum"])
    adapter = ZiplineBundleAdapter(
        snapshot_manager=_ActionSnapshotManager(manifest),
        data_source=_ActionDataSource(rows),
        calendar=action_calendar,
        cache_root=tmp_path / "action-bundle",
    )
    materialized = adapter.materialize(manifest.snapshot_id)
    assert isinstance(materialized, Ok)

    result = BacktestEngine(
        snapshot_manager=_ActionSnapshotManager(manifest),
        decision_delivery=_ActionDelivery(),
        calendar=_ActionDecisionCalendar(fixture["calendar"]["schedule_checksum"]),
    ).run(materialized.value, None, _action_config())
    assert isinstance(result, Ok), getattr(result, "errors", result)

    output = result.value
    fill = output.fills[0]
    assert fill.session == date(2024, 1, 3)
    expected = _fixture()["actual_share_actions"]
    post_split_shares = int(
        (
            _decimal(expected["starting_shares"]) * _decimal(expected["split_ratio"])
        ).to_integral_value(rounding=ROUND_FLOOR)
    )
    assert fill.quantity == post_split_shares
    assert fill.base_adjusted_open == _decimal(
        expected["zipline_stored_post_action_open"]
    )

    # The order was submitted before the split, so Zipline adjusts its whole
    # share amount exactly once. The raw post-action open is retained as the
    # ledger price; feeding research-adjusted prices would double-adjust it.
    final_state = output.portfolio_states[-1]
    position = next(item for item in final_state.positions if item.symbol == "AAA")
    assert position.quantity == post_split_shares
    assert position.mark_price == Decimal("67.000000")
    assert final_state.portfolio_equity == (
        final_state.cash_balance + final_state.gross_exposure
    )

    # Assert the pinned adjustment reader contains one split and one dividend
    # row. This is deliberately against the real SQLite ledger writer rather
    # than only the platform projection, so extension/ledger drift fails.
    engine = BacktestEngine()
    bundle_data, registered_name = engine._load_bundle(materialized.value)
    try:
        connection = bundle_data.adjustment_reader.conn
        split_rows = connection.execute(
            "SELECT sid, ratio FROM splits ORDER BY sid"
        ).fetchall()
        dividend_rows = connection.execute(
            "SELECT sid, amount FROM dividend_payouts ORDER BY sid"
        ).fetchall()
    finally:
        if registered_name is not None:
            engine._unregister_bundle(registered_name)
    assert split_rows == [(0, 2 / 3)]
    assert dividend_rows == [(0, 0.5)]
    actual_dividend_cash = _decimal(dividend_rows[0][1]) * post_split_shares
    assert actual_dividend_cash == _decimal(expected["dividend_cash"])
    assert all(
        item.quantity >= 0
        for state in output.portfolio_states
        for item in state.positions
    )
    assert all(
        item.quantity == int(item.quantity)
        for state in output.portfolio_states
        for item in state.positions
    )
