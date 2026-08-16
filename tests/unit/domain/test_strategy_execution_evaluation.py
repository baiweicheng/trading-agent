"""Focused tests for strategy, execution, evaluation, run, and job value objects."""

# ruff: noqa: E501, I001

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.domain.evaluation import (
    DependencyVersion,
    EnvironmentFingerprint,
    MetricName,
    MetricNullReason,
    MetricValue,
    RunContentIdentity,
    RunManifest,
    RunOperationalMetadata,
    ScientificArtifactReference,
)
from quant_research_platform.domain.execution import (
    FillRecord,
    JobOperation,
    JobStage,
    JobState,
    OrderRecord,
    ProgressUpdate,
    RunState,
    TransactionCostModel,
    deterministic_fill_id,
    deterministic_order_id,
    is_legal_job_transition,
    is_legal_run_transition,
    quantize_money,
    require_legal_job_transition,
    require_legal_run_transition,
)
from quant_research_platform.domain.strategy import (
    MomentumStrategyParameters,
    RationalWeight,
    StrategyDecision,
    StrategyExclusionReason,
)


CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64
CHECKSUM_C = "c" * 64
CHECKSUM_D = "d" * 64
RUN_ID_ONE = UUID("00000000-0000-4000-8000-000000000001")
RUN_ID_TWO = UUID("00000000-0000-4000-8000-000000000002")
JOB_ID = UUID("00000000-0000-4000-8000-000000000003")
SIGNAL_SESSION = date(2024, 1, 31)
EXECUTION_SESSION = date(2024, 2, 1)


def _order_id(*, ordinal: int = 0, quantity: int = 3) -> str:
    return deterministic_order_id(
        signal_session=SIGNAL_SESSION,
        execution_session=EXECUTION_SESSION,
        symbol="AAPL",
        requested_quantity=quantity,
        ordinal=ordinal,
    )


def _environment_fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        python_version="3.11.9",
        operating_system="macOS 14.5",
        architecture="arm64",
        dependencies=(DependencyVersion("numpy", "2.0.1"),),
        source_revision="abc123",
        source_dirty=False,
        deterministic_seed=0,
        effective_source_checksum=CHECKSUM_C,
    )


def _content_identity() -> RunContentIdentity:
    return RunContentIdentity(
        schema_version="run_manifest_v1",
        snapshot_id=f"snap_{CHECKSUM_A}",
        strategy_identifier="monthly_momentum_v1",
        strategy_parameters=MomentumStrategyParameters(position_count=2),
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 12, 31),
        configuration_checksum=CHECKSUM_B,
        environment_fingerprint=_environment_fingerprint(),
        scientific_artifacts=(
            ScientificArtifactReference(role="orders", checksum=CHECKSUM_D),
        ),
    )


def test_rational_weights_reduce_compare_and_sum_exactly() -> None:
    first = RationalWeight(2, 4)
    second = RationalWeight(1, 2)
    selected = tuple(RationalWeight.equal_allocation(3) for _ in range(3))

    assert first == second
    assert first.to_canonical_string() == "1/2"
    assert RationalWeight.sum(selected) == RationalWeight(1, 1)
    assert RationalWeight.sum(()) == RationalWeight.zero()
    assert RationalWeight(1, 3).as_decimal() == Decimal(
        "0.3333333333333333333333333333"
    )


def test_deterministic_order_and_fill_ids_are_scientific_not_operational() -> None:
    order_id = _order_id()
    matching_order_id = deterministic_order_id(
        signal_session=SIGNAL_SESSION,
        execution_session=EXECUTION_SESSION,
        symbol=" aapl ",
        requested_quantity=3,
        ordinal=0,
    )
    fill_id = deterministic_fill_id(
        order_id=order_id,
        symbol="AAPL",
        session=EXECUTION_SESSION,
        quantity=3,
        ordinal=0,
    )

    assert order_id == matching_order_id
    assert order_id.startswith("order_")
    assert _order_id(ordinal=1) != order_id
    assert fill_id.startswith("fill_")
    assert (
        deterministic_fill_id(
            order_id=order_id,
            symbol="aapl",
            session=EXECUTION_SESSION,
            quantity=3,
            ordinal=0,
        )
        == fill_id
    )

    order = OrderRecord(
        order_id=order_id,
        signal_session=SIGNAL_SESSION,
        execution_session=EXECUTION_SESSION,
        symbol="aapl",
        requested_quantity=3,
        ordinal=0,
    )
    fill = FillRecord(
        fill_id=fill_id,
        order_id=order.order_id,
        symbol="AAPL",
        session=EXECUTION_SESSION,
        quantity=3,
        ordinal=0,
        base_adjusted_open=Decimal("10"),
        fill_price=Decimal("10.01"),
        gross_notional=Decimal("30.03"),
        commission=Decimal("0.015"),
        slippage_cost=Decimal("0.03"),
    )

    assert fill.order_id == order.order_id


def test_invalid_strategy_execution_and_lifecycle_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="denominator"):
        RationalWeight(1, 0)

    with pytest.raises(ValueError, match="ineligible decisions must have zero"):
        StrategyDecision(
            signal_session=SIGNAL_SESSION,
            symbol="AAPL",
            endpoint_252_session=None,
            endpoint_252_close=None,
            endpoint_21_session=None,
            endpoint_21_close=None,
            momentum_score=None,
            eligible=False,
            rank=None,
            target_weight=RationalWeight(1, 2),
            exclusion_reason=StrategyExclusionReason.MISSING_LONG_ENDPOINT,
        )

    with pytest.raises(ValueError, match="does not match"):
        OrderRecord(
            order_id="order_" + CHECKSUM_A,
            signal_session=SIGNAL_SESSION,
            execution_session=EXECUTION_SESSION,
            symbol="AAPL",
            requested_quantity=3,
            ordinal=0,
        )

    with pytest.raises(ValueError, match="non-negative"):
        TransactionCostModel(commission_bps=Decimal("-0.1"), slippage_bps=Decimal("0"))

    assert is_legal_job_transition(
        JobState.NOT_STARTED,
        JobState.RUNNING,
        operation=JobOperation.BACKTEST,
    )
    assert not is_legal_job_transition(
        JobState.RUNNING,
        JobState.PARTIALLY_SUCCEEDED,
        operation=JobOperation.BACKTEST,
    )
    assert is_legal_run_transition(RunState.RUNNING, RunState.SUCCEEDED)
    assert not is_legal_run_transition(RunState.SUCCEEDED, RunState.FAILED)
    with pytest.raises(ValueError, match="illegal job state transition"):
        require_legal_job_transition(
            JobState.NOT_STARTED,
            JobState.SUCCEEDED,
            operation=JobOperation.INGESTION,
        )
    with pytest.raises(ValueError, match="illegal run state transition"):
        require_legal_run_transition(RunState.SUCCEEDED, RunState.FAILED)
    with pytest.raises(ValueError, match="only for ingestion"):
        ProgressUpdate(
            job_id=JOB_ID,
            operation=JobOperation.BACKTEST,
            state=JobState.PARTIALLY_SUCCEEDED,
            stage=JobStage.COMPLETED,
            completed_units=1,
            total_units=1,
            elapsed_seconds=Decimal("1"),
        )


def test_money_basis_points_and_fill_outputs_quantize_to_six_decimal_places() -> None:
    cost_model = TransactionCostModel(
        commission_bps=Decimal("5.1234567"),
        slippage_bps=Decimal("10.0000004"),
    )
    order_id = _order_id()
    fill = FillRecord(
        fill_id=deterministic_fill_id(
            order_id=order_id,
            symbol="AAPL",
            session=EXECUTION_SESSION,
            quantity=3,
            ordinal=0,
        ),
        order_id=order_id,
        symbol="AAPL",
        session=EXECUTION_SESSION,
        quantity=3,
        ordinal=0,
        base_adjusted_open=Decimal("10.0000004"),
        fill_price=Decimal("10.0100004"),
        gross_notional=Decimal("30.0300003"),
        commission=Decimal("0.0150004"),
        slippage_cost=Decimal("0.0300001"),
    )

    assert quantize_money(Decimal("1.2345675")) == Decimal("1.234568")
    assert cost_model.commission_bps == Decimal("5.123457")
    assert cost_model.slippage_bps == Decimal("10.000000")
    assert fill.base_adjusted_open == Decimal("10.000000")
    assert fill.fill_price == Decimal("10.010000")
    assert fill.gross_notional == Decimal("30.030000")
    assert fill.commission == Decimal("0.015000")
    assert fill.slippage_cost == Decimal("0.030000")


def test_null_metric_reason_serializes_with_json_null() -> None:
    metric = MetricValue(
        name=MetricName.SHARPE_RATIO,
        value=None,
        null_reason=MetricNullReason.ZERO_VOLATILITY,
    )

    assert metric.to_serializable() == {
        "name": "sharpe_ratio",
        "null_reason": "zero_volatility",
        "value": None,
    }
    assert canonical_json(metric.to_serializable()) == (
        b'{"name":"sharpe_ratio","null_reason":"zero_volatility","value":null}\n'
    )
    with pytest.raises(ValueError, match="require null_reason"):
        MetricValue(name=MetricName.SHARPE_RATIO, value=None)
    with pytest.raises(ValueError, match="must not have null_reason"):
        MetricValue(
            name=MetricName.SHARPE_RATIO,
            value=Decimal("1.2"),
            null_reason=MetricNullReason.ZERO_VOLATILITY,
        )


def test_run_scientific_identity_excludes_operational_run_ids_and_timestamps() -> None:
    content_identity = _content_identity()
    first = RunManifest(
        content_identity=content_identity,
        operational_metadata=RunOperationalMetadata(
            run_id=RUN_ID_ONE,
            state=RunState.RUNNING,
            created_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            started_at=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    second = RunManifest(
        content_identity=content_identity,
        operational_metadata=RunOperationalMetadata(
            run_id=RUN_ID_TWO,
            state=RunState.RUNNING,
            created_at=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            started_at=datetime(2025, 1, 1, 12, 1, tzinfo=UTC),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )

    assert first.scientific_checksum == second.scientific_checksum
    assert first.canonical_scientific_bytes() == second.canonical_scientific_bytes()
    assert first.to_serializable()["operational_metadata"]["run_id"] == str(RUN_ID_ONE)
