"""Focused tests for SPY-aligned evaluation and canonical result artifacts."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from quant_research_platform.application.evaluation import EvaluationService
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.execution import (
    CoreBacktestOutput,
    DailyReturn,
    OrderStatus,
    PortfolioState,
    deterministic_order_id,
    OrderRecord,
)
from quant_research_platform.domain.market import DateRange


SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
EVALUATION_RANGE = DateRange(SESSIONS[0], SESSIONS[-1])


def _states() -> tuple[PortfolioState, ...]:
    return (
        PortfolioState(SESSIONS[0], Decimal("100000"), (), Decimal("0"), Decimal("100000"), Decimal("0")),
        PortfolioState(SESSIONS[1], Decimal("101000"), (), Decimal("0"), Decimal("101000"), Decimal("0")),
        PortfolioState(SESSIONS[2], Decimal("100500"), (), Decimal("0"), Decimal("100500"), Decimal("0")),
    )


def _unfilled_order() -> OrderRecord:
    order_id = deterministic_order_id(
        signal_session=date(2023, 12, 29),
        execution_session=SESSIONS[0],
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
    )
    return OrderRecord(
        order_id=order_id,
        signal_session=date(2023, 12, 29),
        execution_session=SESSIONS[0],
        symbol="AAPL",
        requested_quantity=10,
        ordinal=0,
        status=OrderStatus.UNFILLED,
        unfilled_reason="missing_adjusted_open",
    )


def _core_output(*, reverse: bool = False) -> CoreBacktestOutput | SimpleNamespace:
    states = _states()
    returns = (
        DailyReturn(SESSIONS[0], Decimal("0")),
        DailyReturn(SESSIONS[1], Decimal("0.01")),
        DailyReturn(SESSIONS[2], Decimal("-0.004950495049504950495049504950")),
    )
    if not reverse:
        return CoreBacktestOutput(
            orders=(_unfilled_order(),),
            fills=(),
            portfolio_states=states,
            daily_returns=returns,
            strategy_decisions=(),
        )
    return SimpleNamespace(
        orders=(_unfilled_order(),),
        fills=(),
        portfolio_states=list(reversed(states)),
        daily_returns=list(reversed(returns)),
        strategy_decisions=(),
    )


def _snapshot(bars: object) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_bars=bars,
        limitation_disclosure=LimitationDisclosure.current(),
    )


def test_clean_evaluation_emits_disclosed_metrics_and_every_canonical_artifact() -> None:
    result = EvaluationService().evaluate(
        _core_output(),
        _snapshot(
            [
                {"session": SESSIONS[0], "adjusted_close": Decimal("100")},
                {"session": SESSIONS[1], "adjusted_close": Decimal("102")},
                {"session": SESSIONS[2], "adjusted_close": Decimal("101")},
            ]
        ),
        evaluation_range=EVALUATION_RANGE,
    )

    assert isinstance(result, Ok)
    evaluated = result.value
    assert evaluated.limitation_disclosure.version == "limitation-disclosure/v1"
    assert evaluated.spy_gaps == ()
    assert evaluated.unfilled_orders[0].status is OrderStatus.UNFILLED
    assert evaluated.evaluation_result.strategy_metrics.metric("unfilled_orders").value == 1
    assert evaluated.ending_cash_balance == Decimal("100500.000000")
    assert evaluated.total_commissions == Decimal("0.000000")
    assert evaluated.total_slippage == Decimal("0.000000")

    required_roles = {
        "strategy_returns",
        "benchmark_returns",
        "strategy_equity",
        "benchmark_equity",
        "drawdown",
        "monthly_returns",
        "positions",
        "portfolio",
        "orders",
        "fills",
        "decisions",
        "metrics",
        "transactions",
    }
    assert required_roles <= set(evaluated.artifacts.roles)
    assert evaluated.artifacts.roles == tuple(sorted(evaluated.artifacts.roles))
    for artifact in evaluated.artifacts:
        assert artifact.byte_size == len(artifact.payload)
        assert artifact.checksum == __import__("hashlib").sha256(artifact.payload).hexdigest()
        assert artifact.payload.endswith(b"\n")


def test_spy_gap_blocks_comparison_and_enumerates_every_missing_session() -> None:
    result = EvaluationService().evaluate(
        _core_output(),
        _snapshot(
            [
                {"session": SESSIONS[0], "adjusted_close": Decimal("100")},
            ]
        ),
        evaluation_range=EVALUATION_RANGE,
    )

    assert isinstance(result, Err)
    assert [(error.session, error.symbol) for error in result.errors] == [
        (SESSIONS[1], "SPY"),
        (SESSIONS[2], "SPY"),
    ]
    assert all(error.category.value == "validation.gap" for error in result.errors)


def test_evaluation_is_confluent_for_input_order_and_positive_direct_returns() -> None:
    bars = [
        {"session": SESSIONS[0], "adjusted_close": Decimal("100")},
        {"session": SESSIONS[1], "adjusted_close": Decimal("102")},
        {"session": SESSIONS[2], "adjusted_close": Decimal("101")},
    ]
    first = EvaluationService().evaluate(
        _core_output(), _snapshot(bars), evaluation_range=EVALUATION_RANGE
    )
    second = EvaluationService().evaluate(
        _core_output(reverse=True),
        _snapshot(list(reversed(bars))),
        evaluation_range=EVALUATION_RANGE,
    )
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.artifact_checksums == second.value.artifact_checksums
    assert first.value.evaluation_result.to_serializable() == second.value.evaluation_result.to_serializable()

    direct_returns = EvaluationService().evaluate(
        _core_output(),
        SimpleNamespace(
            benchmark_returns={
                SESSIONS[0]: Decimal("0.01"),
                SESSIONS[1]: Decimal("0.02"),
                SESSIONS[2]: Decimal("0.03"),
            },
            limitation_disclosure=LimitationDisclosure.current(),
        ),
        evaluation_range=EVALUATION_RANGE,
    )
    assert isinstance(direct_returns, Ok)
    assert tuple(item.return_value for item in direct_returns.value.benchmark_returns) == (
        Decimal("0.01"),
        Decimal("0.02"),
        Decimal("0.03"),
    )


class _Batch:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return self.rows


class _BarReader:
    def read(self, *, snapshot: object, symbol: str, start: date, end: date) -> list[_Batch]:
        assert snapshot is not None
        assert symbol == "SPY"
        assert start == EVALUATION_RANGE.start
        assert end == EVALUATION_RANGE.end
        return [
            _Batch([
                {"session": SESSIONS[0], "adjusted_close": Decimal("100")},
                {"session": SESSIONS[1], "adjusted_close": Decimal("102")},
            ]),
            _Batch([
                {"session": SESSIONS[2], "adjusted_close": Decimal("101")},
            ]),
        ]


def test_evaluation_consumes_projected_benchmark_batches_without_unbounded_collection() -> None:
    result = EvaluationService(bar_reader=_BarReader()).evaluate(
        _core_output(),
        SimpleNamespace(limitation_disclosure=LimitationDisclosure.current()),
        evaluation_range=EVALUATION_RANGE,
    )

    assert isinstance(result, Ok)
    assert result.value.benchmark_returns[-1].session == SESSIONS[-1]
