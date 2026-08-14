"""Property tests for stable reruns across the local application pipeline."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.backtests import (
    BacktestRequest,
    BacktestService,
)
from quant_research_platform.application.evaluation import EvaluationService
from quant_research_platform.config.models import ResolvedConfig
from quant_research_platform.config.serializer import ConfigurationSerializer
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import Ok
from quant_research_platform.domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    CoreBacktestOutput,
    DailyReturn,
    PortfolioState,
)
from quant_research_platform.domain.market import DateRange

_SYMBOL_POOL = ("AAPL", "MSFT", "PG", "XOM")


@dataclass(frozen=True, slots=True)
class StableRerunCase:
    """Bounded local scientific and reproducibility inputs for one rerun pair."""

    sessions: tuple[date, ...]
    strategy_equity: tuple[Decimal, ...]
    spy_prices: tuple[Decimal, ...]
    universe: tuple[str, ...]
    position_count: int
    batch_size: int
    commission_bps: Decimal
    slippage_bps: Decimal
    deterministic_seed: int
    source_revision: str
    source_dirty: bool
    effective_source_checksum: str
    dependency_versions: tuple[tuple[str, str], ...]
    operating_system: str
    architecture: str
    writer_version: str
    adapter_version: str
    mutation: str

    @property
    def requested_range(self) -> DateRange:
        return DateRange(self.sessions[0], self.sessions[-1])

    @property
    def environment_fingerprint(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "dependencies": [
                {"name": name, "version": version}
                for name, version in self.dependency_versions
            ],
            "deterministic_seed": self.deterministic_seed,
            "effective_source_checksum": self.effective_source_checksum,
            "operating_system": self.operating_system,
            "python_version": "3.11",
            "source_dirty": self.source_dirty,
            "source_revision": self.source_revision,
        }

    @property
    def snapshot_id(self) -> str:
        """Derive a valid fixture Snapshot_ID from stable snapshot inputs only."""

        identity = {
            "adapter_version": self.adapter_version,
            "batch_size": self.batch_size,
            "configured_universe": list(self.universe),
            "dependency_versions": list(self.dependency_versions),
            "effective_source_checksum": self.effective_source_checksum,
            "requested_range": self.requested_range.to_content_dict(),
            "source_dirty": self.source_dirty,
            "source_revision": self.source_revision,
            "spy_prices": list(self.spy_prices),
            "writer_version": self.writer_version,
        }
        return "snap_" + sha256_bytes(canonical_json(identity))


def _checksum(label: str, value: object) -> str:
    return sha256_bytes(canonical_json({"label": label, "value": value}))


@st.composite
def stable_rerun_cases(draw: st.DrawFn) -> StableRerunCase:
    """Generate equivalent local snapshot/configuration/fingerprint inputs."""

    count = draw(st.integers(min_value=2, max_value=6))
    start = date(2024, 1, 2) + timedelta(
        days=draw(st.integers(min_value=0, max_value=40))
    )
    sessions = tuple(start + timedelta(days=offset) for offset in range(count))
    universe = tuple(
        draw(
            st.lists(
                st.sampled_from(_SYMBOL_POOL),
                min_size=1,
                max_size=3,
                unique=True,
            )
        )
    )
    position_count = draw(st.integers(min_value=1, max_value=len(universe)))
    strategy_tail = draw(
        st.lists(
            st.integers(min_value=1, max_value=200_000),
            min_size=count - 1,
            max_size=count - 1,
        )
    )
    strategy_equity = (INITIAL_PORTFOLIO_EQUITY, *map(Decimal, strategy_tail))
    spy_prices = tuple(
        map(
            Decimal,
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=200_000),
                    min_size=count,
                    max_size=count,
                )
            ),
        )
    )
    source_revision_number = draw(st.integers(min_value=0, max_value=10_000))
    source_revision = f"fixture-revision-{source_revision_number}"
    source_dirty = draw(st.booleans())
    dependency_versions = (
        ("fixture-engine", f"{draw(st.integers(min_value=1, max_value=9))}.0"),
        ("python", "3.11"),
    )
    writer_version = f"fixture-writer-{draw(st.integers(min_value=1, max_value=9))}"
    adapter_version = f"fixture-adapter-{draw(st.integers(min_value=1, max_value=9))}"
    effective_source_checksum = _checksum(
        "effective-source",
        {
            "dependencies": dependency_versions,
            "revision": source_revision,
            "writer": writer_version,
        },
    )
    return StableRerunCase(
        sessions=sessions,
        strategy_equity=strategy_equity,
        spy_prices=spy_prices,
        universe=universe,
        position_count=position_count,
        batch_size=draw(st.integers(min_value=1, max_value=10)),
        commission_bps=Decimal(draw(st.integers(min_value=0, max_value=50))),
        slippage_bps=Decimal(draw(st.integers(min_value=0, max_value=50))),
        deterministic_seed=draw(st.integers(min_value=0, max_value=4_294_967_295)),
        source_revision=source_revision,
        source_dirty=source_dirty,
        effective_source_checksum=effective_source_checksum,
        dependency_versions=dependency_versions,
        operating_system="fixture-os",
        architecture="fixture-architecture",
        writer_version=writer_version,
        adapter_version=adapter_version,
        mutation=draw(st.sampled_from(("strategy_equity", "benchmark_series"))),
    )


def _resolved_config(case: StableRerunCase) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {},
            "data": {
                "universe": list(case.universe),
                "requested_range": {
                    "start": case.sessions[0],
                    "end": case.sessions[-1],
                },
                "batch_size": case.batch_size,
            },
            "strategy": {"position_count": case.position_count},
            "execution": {
                "commission_bps": case.commission_bps,
                "slippage_bps": case.slippage_bps,
            },
            "runtime": {"deterministic_seed": case.deterministic_seed},
        }
    )


def _core_output(case: StableRerunCase) -> CoreBacktestOutput:
    """Build a valid deterministic all-cash core output for the engine seam."""

    states: list[PortfolioState] = []
    returns: list[DailyReturn] = []
    previous_equity: Decimal | None = None
    for session, equity in zip(case.sessions, case.strategy_equity, strict=True):
        state = PortfolioState(
            session=session,
            cash_balance=equity,
            positions=(),
            gross_exposure=Decimal("0"),
            portfolio_equity=equity,
            leverage=Decimal("0"),
        )
        states.append(state)
        returns.append(
            DailyReturn(
                session=session,
                return_value=(
                    Decimal("0")
                    if previous_equity is None
                    else equity / previous_equity - Decimal("1")
                ),
            )
        )
        previous_equity = equity
    return CoreBacktestOutput(
        orders=(),
        fills=(),
        portfolio_states=tuple(states),
        daily_returns=tuple(returns),
        strategy_decisions=(),
    )


@dataclass(frozen=True, slots=True)
class FixtureSnapshot:
    """Verified local snapshot projection consumed by bundle and evaluation seams."""

    snapshot_id: str
    benchmark_bars: tuple[dict[str, object], ...]
    environment_fingerprint: dict[str, object]
    available: bool = True
    comparison_ready: bool = True


class FixtureSnapshotManager:
    """In-memory verified snapshot port; it performs no external I/O."""

    def __init__(self, snapshot: FixtureSnapshot) -> None:
        self.snapshot = snapshot
        self.opened_ids: list[str] = []

    def open_verified(self, snapshot_id: str) -> Ok[FixtureSnapshot]:
        assert snapshot_id == self.snapshot.snapshot_id
        self.opened_ids.append(snapshot_id)
        return Ok(self.snapshot)


class FixtureBundleAdapter:
    """Exact snapshot-keyed derived-bundle seam used by BacktestService."""

    def __init__(self, case: StableRerunCase) -> None:
        self.case = case
        self.calls: list[str] = []
        self.locator = SimpleNamespace(
            bundle_name=f"fixture-{case.snapshot_id}-{case.adapter_version}",
            snapshot_id=case.snapshot_id,
            adapter_version=case.adapter_version,
            bundle_checksum=_checksum(
                "bundle",
                {
                    "adapter_version": case.adapter_version,
                    "snapshot_id": case.snapshot_id,
                    "writer_version": case.writer_version,
                },
            ),
        )

    def materialize(self, snapshot: FixtureSnapshot) -> Ok[object]:
        assert snapshot.snapshot_id == self.case.snapshot_id
        self.calls.append(snapshot.snapshot_id)
        return Ok(self.locator)


class FixtureEngine:
    """Deterministic local event-loop seam returning a domain CoreBacktestOutput."""

    def __init__(self, case: StableRerunCase) -> None:
        self.case = case
        self.calls: list[object] = []

    def run(
        self,
        bundle: object,
        request: BacktestRequest,
        config: ResolvedConfig,
        progress: object | None = None,
    ) -> Ok[CoreBacktestOutput]:
        del progress
        assert bundle.snapshot_id == self.case.snapshot_id
        assert request.snapshot_id == self.case.snapshot_id
        assert config.runtime.deterministic_seed == self.case.deterministic_seed
        self.calls.append((bundle, request, config))
        return Ok(_core_output(self.case))


@dataclass(frozen=True, slots=True)
class OperationalRun:
    """Operational run facts retained separately from the scientific manifest."""

    run_id: UUID
    created_at: datetime
    ended_at: datetime
    manifest: dict[str, object]
    manifest_checksum: str


class FixtureTracker:
    """Local tracker that records distinct IDs/times and a canonical science manifest."""

    def __init__(self, case: StableRerunCase, config: ResolvedConfig) -> None:
        self.case = case
        self.config = config
        self.pending: dict[UUID, datetime] = {}
        self.records: list[OperationalRun] = []

    def allocate_run(self, **values: object) -> UUID:
        run_id = values["run_id"]
        created_at = values["created_at"]
        assert isinstance(run_id, UUID)
        assert isinstance(created_at, datetime)
        self.pending[run_id] = created_at
        return run_id

    def finalize_success(self, run_id: UUID, result: object) -> Ok[None]:
        created_at = self.pending.pop(run_id)
        ended_at = getattr(result, "_fixture_ended_at", None)
        if not isinstance(ended_at, datetime):
            # BacktestService supplies its operational end time as part of the
            # tracker call; this fallback is only defensive for direct callers.
            ended_at = created_at + timedelta(microseconds=1)
        evaluation = result.evaluation
        config_checksum = sha256_bytes(ConfigurationSerializer().serialize(self.config))
        manifest = {
            "content_identity": {
                "adapter_version": self.case.adapter_version,
                "artifact_checksums": dict(evaluation.artifact_checksums),
                "configuration_checksum": config_checksum,
                "core_output": result.core_output.to_scientific_dict(),
                "environment_fingerprint": self.case.environment_fingerprint,
                "evaluation": evaluation.to_serializable(),
                "evaluation_range": result.evaluation_range.to_content_dict(),
                "snapshot_id": result.snapshot_id,
                "writer_version": self.case.writer_version,
            }
        }
        manifest_checksum = sha256_bytes(canonical_json(manifest))
        self.records.append(
            OperationalRun(
                run_id=run_id,
                created_at=created_at,
                ended_at=ended_at,
                manifest=manifest,
                manifest_checksum=manifest_checksum,
            )
        )
        return Ok(None)


def _snapshot(case: StableRerunCase) -> FixtureSnapshot:
    return FixtureSnapshot(
        snapshot_id=case.snapshot_id,
        benchmark_bars=tuple(
            {
                "session": session,
                "adjusted_close": price,
            }
            for session, price in zip(case.sessions, case.spy_prices, strict=True)
        ),
        environment_fingerprint=case.environment_fingerprint,
    )


class FixtureClock:
    """Clock with deterministic but distinct operational timestamps."""

    def __init__(self, base: datetime) -> None:
        self.current = base

    def utc_now(self) -> datetime:
        value = self.current
        self.current += timedelta(microseconds=1)
        return value


def _execute(
    case: StableRerunCase,
    *,
    clock_base: datetime,
) -> tuple[object, FixtureTracker, FixtureBundleAdapter, FixtureSnapshotManager, FixtureEngine]:
    config = _resolved_config(case)
    snapshot_manager = FixtureSnapshotManager(_snapshot(case))
    bundle_adapter = FixtureBundleAdapter(case)
    engine = FixtureEngine(case)
    tracker = FixtureTracker(case, config)
    service = BacktestService(
        tracker=tracker,
        snapshot_manager=snapshot_manager,
        bundle_adapter=bundle_adapter,
        engine=engine,
        evaluator=EvaluationService(),
        clock=FixtureClock(clock_base),
    )
    result = service.run(
        BacktestRequest(case.snapshot_id, case.requested_range),
        config,
    )
    assert isinstance(result, Ok), result
    assert len(tracker.records) == 1
    return result.value, tracker, bundle_adapter, snapshot_manager, engine


def _scientific_projection(result: object, record: OperationalRun) -> dict[str, object]:
    evaluation = result.evaluation
    return {
        "artifact_bytes": {
            artifact.role: artifact.payload for artifact in evaluation.artifacts
        },
        "artifact_checksums": dict(evaluation.artifact_checksums),
        "core_output": result.core_output.to_scientific_dict(),
        "evaluation": evaluation.to_serializable(),
        "manifest": record.manifest,
        "manifest_checksum": record.manifest_checksum,
        "metrics": evaluation.evaluation_result.to_serializable(),
    }


# Feature: quantitative-research-platform, Property 12: Stable reruns preserve scientific outputs and checksums
# Validates: Requirements 9.22–9.24, 10.16–10.17, 11.11–11.14, 17.10, 17.33–17.34
@settings(max_examples=100, deadline=None)
@given(case=stable_rerun_cases())
def test_stable_reruns_preserve_scientific_outputs_and_checksums(
    case: StableRerunCase,
) -> None:
    """Equivalent local pipelines preserve science while operational facts differ."""

    first, first_tracker, first_bundle, first_snapshots, first_engine = _execute(
        case,
        clock_base=datetime(2024, 2, 5, tzinfo=UTC),
    )
    second, second_tracker, second_bundle, second_snapshots, second_engine = _execute(
        case,
        clock_base=datetime(2024, 2, 6, tzinfo=UTC),
    )
    first_record = first_tracker.records[0]
    second_record = second_tracker.records[0]

    assert first.run_id != second.run_id
    assert first_record.created_at != second_record.created_at
    assert first_record.ended_at != second_record.ended_at
    assert first_record.run_id != second_record.run_id
    assert first_record.manifest_checksum == second_record.manifest_checksum
    assert _scientific_projection(first, first_record) == _scientific_projection(
        second, second_record
    )
    assert first.core_output.to_scientific_dict() == second.core_output.to_scientific_dict()
    assert first.evaluation.evaluation_result.to_serializable() == second.evaluation.evaluation_result.to_serializable()
    assert dict(first.evaluation.artifact_checksums) == dict(second.evaluation.artifact_checksums)
    assert {
        role: artifact.payload for role, artifact in ((item.role, item) for item in first.evaluation.artifacts)
    } == {
        role: artifact.payload for role, artifact in ((item.role, item) for item in second.evaluation.artifacts)
    }
    assert first_bundle.locator == second_bundle.locator
    assert first_snapshots.opened_ids == [case.snapshot_id]
    assert second_snapshots.opened_ids == [case.snapshot_id]
    assert len(first_engine.calls) == len(second_engine.calls) == 1
    assert first_bundle.calls == second_bundle.calls == [case.snapshot_id]

    if case.mutation == "strategy_equity":
        changed = replace(
            case,
            strategy_equity=(*case.strategy_equity[:-1], case.strategy_equity[-1] + Decimal("1")),
        )
    else:
        changed = replace(
            case,
            spy_prices=(*case.spy_prices[:-1], case.spy_prices[-1] + Decimal("1")),
        )
    changed_result, changed_tracker, _, _, _ = _execute(
        changed,
        clock_base=datetime(2024, 2, 7, tzinfo=UTC),
    )
    changed_record = changed_tracker.records[0]
    assert _scientific_projection(first, first_record) != _scientific_projection(
        changed_result, changed_record
    )
    assert (
        first.core_output.to_scientific_dict()
        != changed_result.core_output.to_scientific_dict()
        or dict(first.evaluation.artifact_checksums)
        != dict(changed_result.evaluation.artifact_checksums)
    )
