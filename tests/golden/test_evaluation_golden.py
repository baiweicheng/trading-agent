"""Reviewed offline metric, artifact, monthly-return, and SPY-gap goldens."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from quant_research_platform.application.evaluation import EvaluationService
from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.evaluation import (
    MetricScope,
    calculate_evaluation_metrics,
)
from quant_research_platform.domain.execution import (
    CoreBacktestOutput,
    DailyReturn,
    FillRecord,
    OrderRecord,
    OrderStatus,
    PortfolioState,
    Position,
    deterministic_fill_id,
    deterministic_order_id,
)
from quant_research_platform.domain.market import DateRange
from quant_research_platform.domain.strategy import (
    RationalWeight,
    StrategyDecision,
)

_STABLE_FIXTURE = Path(__file__).parent / "stable_run" / "evaluation.json"
_GAP_FIXTURE = Path(__file__).parent / "spy_gap" / "evaluation.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    return value


def _core(fixture: dict[str, Any]) -> CoreBacktestOutput:
    source = fixture["input"]
    sessions = tuple(date.fromisoformat(item) for item in source["sessions"])
    cash = Decimal(source["cash_balance"])
    positions: list[PortfolioState] = []
    for session, mark_text in zip(sessions, source["strategy_marks"], strict=True):
        mark = Decimal(mark_text)
        gross = Decimal(10) * mark
        equity = cash + gross
        positions.append(
            PortfolioState(
                session,
                cash,
                (Position("AAPL", 10, mark, gross),),
                gross,
                equity,
                (gross / equity).quantize(Decimal("0.000000000000000001")),
            )
        )

    def make_order(spec: dict[str, Any]) -> OrderRecord:
        order_id = deterministic_order_id(
            signal_session=date.fromisoformat(spec["signal_session"]),
            execution_session=date.fromisoformat(spec["execution_session"]),
            symbol=spec["symbol"],
            requested_quantity=spec["requested_quantity"],
            ordinal=spec["ordinal"],
        )
        return OrderRecord(
            order_id=order_id,
            signal_session=date.fromisoformat(spec["signal_session"]),
            execution_session=date.fromisoformat(spec["execution_session"]),
            symbol=spec["symbol"],
            requested_quantity=spec["requested_quantity"],
            ordinal=spec["ordinal"],
            decision_rank=spec["decision_rank"],
            status=OrderStatus(spec["status"]),
            unfilled_reason=spec.get("unfilled_reason"),
        )

    filled_order = make_order(source["filled_order"])
    unfilled_order = make_order(source["unfilled_order"])
    fill_source = source["fill"]
    fill = FillRecord(
        fill_id=deterministic_fill_id(
            order_id=filled_order.order_id,
            symbol=filled_order.symbol,
            session=filled_order.execution_session,
            quantity=filled_order.requested_quantity,
            ordinal=filled_order.ordinal,
        ),
        order_id=filled_order.order_id,
        symbol=filled_order.symbol,
        session=filled_order.execution_session,
        quantity=filled_order.requested_quantity,
        ordinal=filled_order.ordinal,
        base_adjusted_open=Decimal(fill_source["base_adjusted_open"]),
        fill_price=Decimal(fill_source["fill_price"]),
        gross_notional=Decimal(fill_source["gross_notional"]),
        commission=Decimal(fill_source["commission"]),
        slippage_cost=Decimal(fill_source["slippage_cost"]),
    )
    decision_source = source["decision"]
    decision = StrategyDecision(
        signal_session=date.fromisoformat(decision_source["signal_session"]),
        symbol=decision_source["symbol"],
        endpoint_252_session=date.fromisoformat(
            decision_source["endpoint_252_session"]
        ),
        endpoint_252_close=Decimal(decision_source["endpoint_252_close"]),
        endpoint_21_session=date.fromisoformat(decision_source["endpoint_21_session"]),
        endpoint_21_close=Decimal(decision_source["endpoint_21_close"]),
        momentum_score=Decimal(decision_source["momentum_score"]),
        eligible=decision_source["eligible"],
        rank=decision_source["rank"],
        target_weight=RationalWeight(1, 1),
        exclusion_reason=None,
        endpoint_252_checksum=decision_source["endpoint_252_checksum"],
        endpoint_21_checksum=decision_source["endpoint_21_checksum"],
    )
    return CoreBacktestOutput(
        orders=(filled_order, unfilled_order),
        fills=(fill,),
        portfolio_states=tuple(positions),
        daily_returns=tuple(
            DailyReturn(session, Decimal(value))
            for session, value in zip(sessions, source["strategy_returns"], strict=True)
        ),
        strategy_decisions=(decision,),
    )


def _snapshot(
    source: dict[str, Any], *, observed: list[int] | None = None
) -> SimpleNamespace:
    sessions = tuple(date.fromisoformat(item) for item in source["sessions"])
    prices = tuple(Decimal(item) for item in source["benchmark_adjusted_closes"])
    indexes = range(len(sessions)) if observed is None else observed
    return SimpleNamespace(
        benchmark_bars=[
            {"session": sessions[index], "adjusted_close": prices[index]}
            for index in indexes
        ],
        limitation_disclosure=LimitationDisclosure.current(),
    )


def test_stable_run_matches_reviewed_metrics_and_artifacts() -> None:
    fixture = _load(_STABLE_FIXTURE)
    source = fixture["input"]
    sessions = tuple(date.fromisoformat(item) for item in source["sessions"])
    result = EvaluationService().evaluate(
        _core(fixture),
        _snapshot(source),
        evaluation_range=DateRange(sessions[0], sessions[-1]),
    )

    assert isinstance(result, Ok), result
    output = result.value
    expected = fixture["expected"]
    assert (
        _plain(output.evaluation_result.to_serializable())
        == expected["evaluation_result"]
    )
    assert (
        _plain(
            {
                "strategy": [
                    item.to_serializable() for item in output.strategy_monthly_returns
                ],
                "benchmark": [
                    item.to_serializable() for item in output.benchmark_monthly_returns
                ],
            }
        )
        == expected["monthly_returns"]
    )
    assert output.artifacts.roles == tuple(expected["artifact_roles"])
    assert dict(output.artifact_checksums) == expected["artifact_checksums"]

    metric_payload = json.loads(output.artifacts.get("metrics").payload)
    expected_metric_rows = [
        {
            **row,
            "scope": scope,
            "value": (
                row["value"]
                if row["name"] == "unfilled_orders" or row["value"] is None
                else Decimal(row["value"])
            ),
        }
        for scope, key in (
            ("strategy", "strategy_metrics"),
            ("benchmark", "benchmark_metrics"),
            ("difference", "differences"),
        )
        for row in expected["evaluation_result"][key]["metrics"]
    ]
    expected_metric_payload = json.loads(
        canonical_json(
            {
                "rows": sorted(
                    expected_metric_rows,
                    key=lambda row: canonical_json(row).decode(),
                ),
                "schema_version": "metrics_v1",
            }
        )
    )
    assert metric_payload == expected_metric_payload

    monthly_payload = json.loads(output.artifacts.get("monthly_returns").payload)
    monthly_rows = [
        {
            "month": item["month"],
            "return_value": Decimal(item["return_value"]),
            "scope": scope,
        }
        for scope in ("strategy", "benchmark")
        for item in expected["monthly_returns"][scope]
    ]
    expected_monthly_payload = json.loads(
        canonical_json(
            {
                "rows": sorted(
                    monthly_rows,
                    key=lambda row: canonical_json(row).decode(),
                ),
                "schema_version": "monthly_return_v1",
            }
        )
    )
    assert monthly_payload == expected_monthly_payload

    for role in ("chart_drawdown", "chart_equity_curve", "chart_monthly_returns"):
        chart = json.loads(output.artifacts.get(role).payload)
        assert chart["$schema"].endswith("/vega-lite/v5.json")
        assert chart["mark"] == {"type": "line"}
        assert chart["encoding"]["x"] == {"field": "session", "type": "temporal"}

    for artifact in output.artifacts:
        assert artifact.payload.endswith(b"\n")
        assert artifact.byte_size == len(artifact.payload)


def test_spy_gap_blocks_and_preserves_cost_diagnostics() -> None:
    fixture = _load(_GAP_FIXTURE)
    stable = _load(_STABLE_FIXTURE)
    source = stable["input"]
    sessions = tuple(
        date.fromisoformat(item) for item in fixture["input"]["evaluation_sessions"]
    )
    result = EvaluationService().evaluate(
        _core(stable),
        _snapshot(source, observed=[0]),
        evaluation_range=DateRange(sessions[0], sessions[-1]),
    )

    assert isinstance(result, Err)
    expected = fixture["expected"]
    assert [(error.session.isoformat(), error.symbol) for error in result.errors] == [
        (session, expected["symbol"]) for session in expected["missing_sessions"]
    ]
    assert all(error.operation == expected["operation"] for error in result.errors)
    assert all(error.category.value == expected["category"] for error in result.errors)
    assert all(error.field_path == expected["field_path"] for error in result.errors)
    assert all(
        error.corrective_action == expected["corrective_action"]
        for error in result.errors
    )

    core = _core(stable)
    diagnostics = calculate_evaluation_metrics(
        MetricScope.STRATEGY,
        core.portfolio_states,
        returns=core.daily_returns,
        fills=core.fills,
        orders=core.orders,
        portfolio_equity=tuple(
            state.portfolio_equity for state in core.portfolio_states
        ),
    )
    actual_diagnostics = {
        name: _plain(diagnostics.metric(name).value)
        for name in expected["diagnostic_strategy_metrics"]
    }
    assert actual_diagnostics == expected["diagnostic_strategy_metrics"]
