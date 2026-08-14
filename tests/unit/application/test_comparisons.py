"""Focused offline tests for ordered, verified multi-run comparison."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

from quant_research_platform.application.comparisons import ComparisonService
from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.errors import Err, Ok
from quant_research_platform.domain.evaluation import (
    EvaluationMetrics,
    MetricName,
    MetricScope,
    MetricValue,
)

RUNS = tuple(UUID(int=index) for index in range(1, 3))
SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))


def _metrics(scope: MetricScope) -> EvaluationMetrics:
    names = (
        MetricName.TOTAL_RETURN,
        MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
        MetricName.ANNUALIZED_VOLATILITY,
        MetricName.SHARPE_RATIO,
        MetricName.MAXIMUM_DRAWDOWN,
    )
    if scope is MetricScope.STRATEGY:
        names += (
            MetricName.TURNOVER,
            MetricName.TOTAL_COMMISSIONS,
            MetricName.TOTAL_SLIPPAGE,
            MetricName.UNFILLED_ORDERS,
            MetricName.ENDING_CASH_BALANCE,
        )
    values = []
    for name in names:
        values.append(
            MetricValue(
                name,
                0
                if name is MetricName.UNFILLED_ORDERS
                else Decimal("-0.1")
                if name is MetricName.MAXIMUM_DRAWDOWN
                else Decimal("1"),
            )
        )
    return EvaluationMetrics(scope, tuple(values))


def _manifest(
    run: UUID, *, snapshot: str = "snap_" + "a" * 64, secret: str | None = None
) -> tuple[dict[str, object], str]:
    document: dict[str, object] = {
        "content_identity": {
            "snapshot_id": snapshot,
            "calendar": {"name": "XNYS", "version": "1"},
        },
        "evaluation_start": SESSIONS[0],
        "evaluation_end": SESSIONS[-1],
        "configuration": {
            "strategy": {"position_count": 2},
            "secrets": {"https_proxy": secret},
        },
        "environment_fingerprint": {
            "python_version": "3.11",
            "source_revision": str(run),
        },
        "evaluation_result": {
            "strategy_metrics": _metrics(MetricScope.STRATEGY),
            "benchmark_metrics": _metrics(MetricScope.BENCHMARK),
        },
        "artifacts": (),
        "limitation_disclosure": {"version": "limitation-disclosure/v1"},
    }
    # The service hashes the canonical serialized representation, as the local
    # manifest publication does for JSON manifest records.
    return document, _manifest_checksum(document)


def _manifest_checksum(document: object) -> str:
    def jsonable(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (date, Decimal, UUID)):
            return str(value)
        if hasattr(value, "to_serializable"):
            return jsonable(value.to_serializable())
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [jsonable(item) for item in value]
        return value

    return sha256(canonical_json(jsonable(document))).hexdigest()


class Metadata:
    def __init__(self, records: dict[UUID, object]) -> None:
        self.records = records

    def get_run(self, run_id: UUID) -> object:
        return self.records[run_id]


class Publisher:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def publish_artifact(self, payload: bytes, metadata: object) -> object:
        self.payloads.append(payload)
        assert metadata["checksum"] == sha256(payload).hexdigest()  # type: ignore[index]
        return SimpleNamespace(checksum=metadata["checksum"])  # type: ignore[index]


def _record(
    run: UUID,
    *,
    state: str = "succeeded",
    snapshot: str = "snap_" + "a" * 64,
    sessions: tuple[date, ...] = SESSIONS,
) -> object:
    manifest, checksum = _manifest(run, snapshot=snapshot)
    return SimpleNamespace(
        run_id=run,
        state=state,
        snapshot_id=snapshot,
        manifest_checksum=checksum,
        manifest=manifest,
        evaluation_start=sessions[0],
        evaluation_end=sessions[-1],
        strategy_equity=tuple(
            (session, 100000 + index) for index, session in enumerate(sessions)
        ),
        benchmark_equity=tuple(
            (session, 100000 + index * 2) for index, session in enumerate(sessions)
        ),
    )


def test_rejects_selection_bounds_before_loading_runs() -> None:
    metadata = Metadata({RUNS[0]: _record(RUNS[0])})
    service = ComparisonService(metadata=metadata)

    too_few = service.compare((RUNS[0],))
    too_many = service.compare(tuple(UUID(int=index) for index in range(1, 12)))

    assert isinstance(too_few, Err)
    assert "minimum" in too_few.errors[0].message
    assert isinstance(too_many, Err)
    assert "maximum" in too_many.errors[0].message


def test_rejects_duplicate_and_failed_runs() -> None:
    failed = UUID(int=3)
    metadata = Metadata(
        {RUNS[0]: _record(RUNS[0]), failed: _record(failed, state="failed")}
    )
    service = ComparisonService(metadata=metadata)

    duplicate = service.compare((RUNS[0], RUNS[0]))
    failed_result = service.compare((RUNS[0], failed))

    assert isinstance(duplicate, Err)
    assert "distinct" in duplicate.errors[0].message
    assert isinstance(failed_result, Err)
    assert "successful" in failed_result.errors[0].message


def test_preserves_metrics_aligns_intersection_and_publishes_artifact() -> None:
    second = UUID(int=2)
    metadata = Metadata(
        {RUNS[0]: _record(RUNS[0]), second: _record(second, sessions=SESSIONS[1:])}
    )
    publisher = Publisher()
    service = ComparisonService(metadata=metadata, artifacts=publisher)

    result = service.compare(RUNS)

    assert isinstance(result, Ok)
    output = result.value
    assert output.aligned_sessions == (SESSIONS[1], SESSIONS[2])
    assert tuple(run.run_id for run in output.runs) == RUNS
    assert output.runs[0].strategy_metrics == _metrics(MetricScope.STRATEGY)
    assert output.snapshot_differences == ()
    assert len(publisher.payloads) == 1
    assert output.artifact.checksum == sha256(publisher.payloads[0]).hexdigest()
    assert json.loads(output.artifact.payload.decode())["selected_run_ids"] == [
        str(run) for run in RUNS
    ]
    assert output.limitation_disclosure.version == "limitation-disclosure/v1"


def test_rejects_corrupt_manifest_and_no_common_sessions() -> None:
    corrupt = _record(RUNS[0])
    corrupt.manifest_checksum = "b" * 64  # type: ignore[attr-defined]
    service = ComparisonService(
        metadata=Metadata({RUNS[0]: corrupt, RUNS[1]: _record(RUNS[1])})
    )
    corrupt_result = service.compare(RUNS)
    assert isinstance(corrupt_result, Err)
    assert corrupt_result.errors[0].category.value == "integrity.checksum"

    disjoint = ComparisonService(
        metadata=Metadata(
            {
                RUNS[0]: _record(RUNS[0], sessions=(SESSIONS[0],)),
                RUNS[1]: _record(RUNS[1], sessions=(SESSIONS[2],)),
            }
        )
    )
    no_intersection = disjoint.compare(RUNS)
    assert isinstance(no_intersection, Err)
    assert "common" in no_intersection.errors[0].message


def test_configuration_differences_are_recursive_and_secrets_are_redacted() -> None:
    first = _record(RUNS[0])
    second = _record(RUNS[1], snapshot="snap_" + "b" * 64)
    second.manifest["configuration"]["strategy"]["position_count"] = 3  # type: ignore[index]
    second.manifest["configuration"]["secrets"]["https_proxy"] = "do-not-leak"  # type: ignore[index]
    # Recompute the checksum after changing scientific/configuration content.
    second.manifest_checksum = _manifest_checksum(second.manifest)  # type: ignore[attr-defined]
    result = ComparisonService(
        metadata=Metadata({RUNS[0]: first, RUNS[1]: second})
    ).compare(RUNS)

    assert isinstance(result, Ok)
    paths = {row.field_path for row in result.value.configuration_differences}
    assert "strategy.position_count" in paths
    assert "do-not-leak" not in str(result.value)
