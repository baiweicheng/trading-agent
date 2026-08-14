"""Real Zipline event-loop coverage for the pinned backtest adapter."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

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
    OrderStatus,
    deterministic_order_id,
)
from quant_research_platform.domain.manifests import SnapshotHandle
from quant_research_platform.domain.strategy import RationalWeight, StrategyDecision
from quant_research_platform.infrastructure.zipline_bundle import ZiplineBundleAdapter
from quant_research_platform.infrastructure.zipline_engine import BacktestEngine
from tests.contract.test_zipline_bundle_contract import (
    _FixtureCalendar,
    _FixtureSnapshotManager,
    _FixtureSource,
    _normalized_rows,
    _snapshot_manifest,
)


class _EngineCalendar(_FixtureCalendar):
    """Calendar seam with a signal close followed by a tradable session."""

    def month_end_sessions(self, _start: date, _end: date) -> tuple[date, ...]:
        return (date(2024, 1, 3),)

    def next_session(self, session: date) -> date:
        if session == date(2024, 1, 3):
            return date(2024, 1, 4)
        return date(2024, 1, 5)


def _config() -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {},
            "data": {
                "requested_range": {
                    "start": "2024-01-02",
                    "end": "2024-01-04",
                },
                "universe": ["AAPL", "MSFT"],
            },
        }
    )


def _bundle(tmp_path: Path, calendar: _EngineCalendar):
    manifest = _snapshot_manifest()
    handle = SnapshotHandle.from_manifest(
        manifest,
        verified_at=datetime(2024, 1, 10, 15, tzinfo=UTC),
    )
    manager = _FixtureSnapshotManager(manifest, handle)
    result = ZiplineBundleAdapter(
        snapshot_manager=manager,
        data_source=_FixtureSource(_normalized_rows()),
        calendar=calendar,
        zipline_root=tmp_path / "bundles",
    ).materialize(manifest.snapshot_id)
    assert isinstance(result, Ok)
    return manifest, manager, result.value


def _delivery(snapshot_id: str):
    class Delivery:
        def __init__(self) -> None:
            self.calls: list[tuple[object, date, object, dict[str, object]]] = []

        def deliver(
            self,
            snapshot: object,
            session: date,
            portfolio: object,
            **kwargs: object,
        ) -> Ok:
            self.calls.append((snapshot, session, portfolio, kwargs))
            selected = StrategyDecision(
                signal_session=session,
                symbol="AAPL",
                endpoint_252_session=date(2024, 1, 2),
                endpoint_252_close=Decimal("99"),
                endpoint_21_session=date(2024, 1, 2),
                endpoint_21_close=Decimal("99"),
                momentum_score=Decimal("0"),
                eligible=True,
                rank=1,
                target_weight=RationalWeight(1, 1),
                exclusion_reason=None,
            )
            not_selected = StrategyDecision(
                signal_session=session,
                symbol="MSFT",
                endpoint_252_session=date(2024, 1, 2),
                endpoint_252_close=Decimal("199"),
                endpoint_21_session=date(2024, 1, 2),
                endpoint_21_close=Decimal("199"),
                momentum_score=Decimal("0"),
                eligible=True,
                rank=2,
                target_weight=RationalWeight.zero(),
                exclusion_reason="not_selected",
            )
            execution_session = date(2024, 1, 4)
            quantity = 900
            order_id = deterministic_order_id(
                signal_session=session,
                execution_session=execution_session,
                symbol="AAPL",
                requested_quantity=quantity,
                ordinal=0,
            )
            intent = OrderIntent(
                order_id=order_id,
                signal_session=session,
                execution_session=execution_session,
                symbol="AAPL",
                requested_quantity=quantity,
                ordinal=0,
                target_shares=quantity,
                current_shares=0,
                sizing_price=Decimal("100"),
                portfolio_equity=Decimal("100000"),
                decision_rank=1,
            )
            inputs = DecisionRunInputs(position_count=2, snapshot_id=snapshot_id)
            decisions = (selected, not_selected)
            return Ok(
                DecisionDeliveryResult(
                    snapshot_id=snapshot_id,
                    signal_session=session,
                    decisions=decisions,
                    order_intents=(intent,),
                    run_inputs=inputs,
                    marked_equity=Decimal("100000"),
                    decision_book=DecisionBook(decisions, inputs),
                )
            )

    return Delivery()


@pytest.mark.integration
def test_real_event_loop_fills_next_session_and_extracts_core_roles(
    tmp_path: Path,
) -> None:
    calendar = _EngineCalendar()
    manifest, manager, locator = _bundle(tmp_path, calendar)
    delivery = _delivery(manifest.snapshot_id)
    progress = []

    result = BacktestEngine(
        snapshot_manager=manager,
        decision_delivery=delivery,
        calendar=calendar,
    ).run(locator, None, _config(), progress=progress.append)

    assert isinstance(result, Ok)
    output = result.value
    assert output.initial_equity == Decimal("100000.000000")
    assert len(output.portfolio_states) == len(output.daily_returns) == 3
    assert output.portfolio_states[0].cash_balance == Decimal("100000.000000")
    assert output.portfolio_states[-1].positions[0].symbol == "AAPL"
    assert output.portfolio_states[-1].positions[0].quantity == 900
    assert output.portfolio_states[-1].cash_balance >= 0
    assert output.portfolio_states[-1].portfolio_equity > 0
    assert output.portfolio_states[-1].leverage <= Decimal("1")
    assert len(output.strategy_decisions) == 2
    assert len(output.orders) == len(output.fills) == 1

    order = output.orders[0]
    fill = output.fills[0]
    assert order.status is OrderStatus.FILLED
    assert order.signal_session == date(2024, 1, 3)
    assert order.execution_session == fill.session == date(2024, 1, 4)
    assert fill.base_adjusted_open == Decimal("101.000000")
    assert fill.fill_price == Decimal("101.101000")
    assert fill.quantity == 900
    assert delivery.calls[0][1] == date(2024, 1, 3)
    assert delivery.calls[0][3]["execution_session"] == date(2024, 1, 4)
    assert [(update.completed_units, update.total_units) for update in progress] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]


@pytest.mark.integration
def test_default_delivery_reuses_one_verified_snapshot_pin(tmp_path: Path) -> None:
    calendar = _EngineCalendar()
    manifest, manager, locator = _bundle(tmp_path, calendar)
    rows = tuple(
        {
            "symbol": row["symbol"],
            "session": row["session"],
            "adjusted_close": row["adjusted_close"],
            "sizing_adjusted_close": row["raw_close"],
            "tradable": True,
        }
        for row in _normalized_rows()
    )

    class Reader:
        def read_history(
            self,
            snapshot: object,
            *,
            symbols: tuple[str, ...],
            end_session: date,
            fields: tuple[str, ...],
            **_kwargs: object,
        ) -> tuple[dict[str, object], ...]:
            assert fields
            return tuple(
                row
                for row in rows
                if row["symbol"] in symbols and row["session"] <= end_session
            )

    result = BacktestEngine(
        snapshot_manager=manager,
        snapshot_reader=Reader(),
        calendar=calendar,
    ).run(locator, None, _config())

    assert isinstance(result, Ok)
    # One verification occurs during bundle materialization and one at run
    # start; causal delivery reuses that handle for the signal callback.
    assert manager.opened == [manifest.snapshot_id, manifest.snapshot_id]
    assert len(result.value.strategy_decisions) == 2
    assert result.value.orders == ()
    assert result.value.fills == ()
