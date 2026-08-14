"""Focused offline tests for the DuckDB metadata repository."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from quant_research_platform.domain.evaluation import (
    MetricScope,
    calculate_evaluation_metrics,
    strategy_minus_benchmark,
)
from quant_research_platform.domain.errors import (
    ActionableError,
    ErrorCategory,
    LimitationDisclosure,
)
from quant_research_platform.domain.execution import (
    JobOperation,
    JobStage,
    JobState,
    ProgressUpdate,
    RunState,
)
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotManifest,
)
from quant_research_platform.domain.market import (
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
    SymbolValidationSummary,
    ValidationSummary,
)
from quant_research_platform.infrastructure.duckdb_metadata import (
    DuckDBMetadataStore,
    IllegalMetadataTransitionError,
    ImmutableMetadataError,
    MetadataNotFoundError,
    RunFinalization,
    RunQuery,
    SnapshotAvailability,
)

_NOW = datetime(2024, 1, 10, 15, tzinfo=UTC)
_START = date(2024, 1, 2)
_END = date(2024, 1, 5)


def _checksum(character: str) -> str:
    return character * 64


def _not_started(job_id: UUID) -> ProgressUpdate:
    return ProgressUpdate(
        job_id=job_id,
        operation=JobOperation.INGESTION,
        state=JobState.NOT_STARTED,
        stage=JobStage.NOT_STARTED,
        completed_units=0,
        total_units=1,
        elapsed_seconds=Decimal("0"),
    )


def _snapshot(created_at: datetime = _NOW) -> SnapshotManifest:
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=_checksum("a"),
        relative_uri="objects/normalized/symbol=AAPL/year=2024/sha256=a.parquet",
        schema_version="daily_bar_v1",
        row_count=1,
        byte_size=128,
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    identity = SnapshotContentIdentity(
        provider="yfinance",
        requested_range=DateRange(_START, _END),
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity(
            name="XNYS",
            version="4.0",
            schedule_checksum=_checksum("b"),
        ),
        configuration_checksum=_checksum("c"),
        objects=(reference,),
        validation_report_checksum=_checksum("d"),
        validation_summary=ValidationSummary(
            accepted_row_count=1,
            quarantined_row_count=0,
            collapsed_duplicate_count=0,
            gap_count=0,
            covered_range=DateRange(_START, _END),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    return SnapshotManifest(
        content_identity=identity,
        operational_metadata=OperationalMetadata(created_at=created_at),
    )


def _symbol_status() -> SymbolValidationSummary:
    return SymbolValidationSummary(
        symbol="AAPL",
        accepted_count=1,
        quarantined_count=0,
        duplicate_count=0,
        gap_count=0,
        covered_range=DateRange(_START, _END),
    )


def _provider_result() -> ProviderBatchResult:
    request = ProviderRequest(("AAPL",), _START, _END)
    record = ProviderRecord(
        provider="yfinance",
        request_content_key=request.content_key,
        symbol="AAPL",
        raw_bar=RawDailyBar(
            provider_date=_START,
            open="10",
            high="11",
            low="9",
            close="10",
            volume="100",
        ),
    )
    outcome = SymbolOutcome(
        symbol="AAPL",
        status=SymbolOutcomeStatus.SUCCESS,
        attempts=1,
        records=(record,),
    )
    return ProviderBatchResult(request=request, outcomes=(outcome,))


def _failure() -> ActionableError:
    return ActionableError(
        operation="backtest.execute",
        category=ErrorCategory.BACKTEST_INVARIANT,
        message="An accounting invariant failed.",
        corrective_action="Review the diagnostic artifact and create a new run.",
    )


def _successful_finalization() -> RunFinalization:
    strategy = calculate_evaluation_metrics(
        MetricScope.STRATEGY,
        (Decimal("100000"), Decimal("101000")),
    )
    benchmark = calculate_evaluation_metrics(
        MetricScope.BENCHMARK,
        (Decimal("100000"), Decimal("100500")),
    )
    return RunFinalization(
        desired_state=RunState.SUCCEEDED,
        manifest_checksum=_checksum("a"),
        manifest_uri="runs/example/manifest.json",
        metrics=tuple(
            sorted(
                (strategy_minus_benchmark(strategy, benchmark), benchmark, strategy),
                key=lambda item: item.scope.value,
            )
        ),
    )


def test_migrates_persists_and_reopens_all_operational_indexes(tmp_path: object) -> None:
    database_path = tmp_path / "metadata.duckdb"  # type: ignore[operator]
    job_id = UUID("00000000-0000-0000-0000-000000000001")
    manifest = _snapshot()

    with DuckDBMetadataStore(database_path) as store:
        store.create_job(_not_started(job_id), updated_at=_NOW)
        assert store.insert_snapshot(
            manifest,
            manifest_uri=f"snapshots/{manifest.snapshot_id}/manifest.json",
            symbol_statuses=(_symbol_status(),),
        )
        store.record_provider_batch(job_id=job_id, result=_provider_result(), occurred_at=_NOW)

    with DuckDBMetadataStore(database_path) as reopened:
        persisted_job = reopened.get_job(job_id)
        persisted_snapshot = reopened.get_snapshot(manifest.snapshot_id)
        assert persisted_job.state is JobState.NOT_STARTED
        assert persisted_snapshot.manifest_checksum == manifest.manifest_checksum
        assert reopened.list_snapshot_objects(manifest.snapshot_id)[0].checksum == _checksum("a")


def test_transaction_rolls_back_nested_repository_writes(tmp_path: object) -> None:
    database_path = tmp_path / "metadata.duckdb"  # type: ignore[operator]
    job_id = uuid4()
    store = DuckDBMetadataStore(database_path)

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            store.create_job(_not_started(job_id), updated_at=_NOW)
            raise RuntimeError("rollback")

    with pytest.raises(MetadataNotFoundError):
        store.get_job(job_id)
    assert store.create_job(_not_started(job_id), updated_at=_NOW).job_id == job_id
    store.close()


def test_snapshot_science_is_insert_only_but_availability_is_operational(tmp_path: object) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    first = _snapshot()
    replay = _snapshot(created_at=_NOW + timedelta(hours=1))
    first_uri = f"snapshots/{first.snapshot_id}/manifest.json"

    assert store.insert_snapshot(
        first,
        manifest_uri=first_uri,
        symbol_statuses=(_symbol_status(),),
    )
    assert not store.insert_snapshot(
        replay,
        manifest_uri=f"other-root/{replay.snapshot_id}/manifest.json",
        symbol_statuses=(_symbol_status(),),
    )
    unavailable = store.set_snapshot_availability(
        first.snapshot_id,
        SnapshotAvailability.INVALID,
    )

    assert unavailable.availability is SnapshotAvailability.INVALID
    assert unavailable.manifest_uri == first_uri
    assert store.list_snapshots(availability=SnapshotAvailability.INVALID) == (unavailable,)
    store.close()


def test_run_and_job_illegal_transitions_are_rejected_after_terminal_state(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    run_id = uuid4()
    snapshot_id = _snapshot().snapshot_id
    store.create_run(
        run_id=run_id,
        snapshot_id=snapshot_id,
        strategy_id="monthly_momentum_v1",
        evaluation_start=_START,
        evaluation_end=_END,
        universe=("AAPL",),
        configuration_checksum=_checksum("e"),
        environment_checksum=_checksum("f"),
        created_at=_NOW,
        started_at=_NOW,
    )
    failed = RunFinalization(
        desired_state=RunState.FAILED,
        manifest_checksum=None,
        manifest_uri=None,
        errors=(_failure(),),
    )
    store.create_finalization_intent(run_id, failed, created_at=_NOW)
    store.mark_finalization_mlflow_synced(run_id, attempted_at=_NOW)
    terminal = store.finalize_run(run_id, failed, ended_at=_NOW + timedelta(seconds=1))

    assert terminal.state is RunState.FAILED
    assert terminal.immutable
    with pytest.raises(ImmutableMetadataError, match="terminal"):
        store.set_mlflow_run_id(run_id, "mlflow-later")
    with pytest.raises(ImmutableMetadataError, match="terminal"):
        store.create_finalization_intent(run_id, failed, created_at=_NOW)

    job_id = uuid4()
    store.create_job(_not_started(job_id), updated_at=_NOW)
    invalid = ProgressUpdate(
        job_id=job_id,
        operation=JobOperation.INGESTION,
        state=JobState.SUCCEEDED,
        stage=JobStage.COMPLETED,
        completed_units=1,
        total_units=1,
        elapsed_seconds=Decimal("1"),
    )
    with pytest.raises(IllegalMetadataTransitionError, match="illegal job"):
        store.update_job(invalid, updated_at=_NOW)
    store.close()


def test_run_discovery_filters_projects_and_orders_pages_deterministically(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    snapshot_id = _snapshot().snapshot_id
    first_id = UUID("00000000-0000-0000-0000-000000000010")
    second_id = UUID("00000000-0000-0000-0000-000000000020")
    other_id = UUID("00000000-0000-0000-0000-000000000030")
    created = _NOW + timedelta(minutes=1)

    for run_id, run_snapshot, run_created in (
        (second_id, snapshot_id, created),
        (first_id, snapshot_id, created),
        (other_id, "snap_" + _checksum("9"), _NOW),
    ):
        store.create_run(
            run_id=run_id,
            snapshot_id=run_snapshot,
            strategy_id="monthly_momentum_v1",
            evaluation_start=_START,
            evaluation_end=_END,
            universe=("AAPL",),
            configuration_checksum=_checksum("e"),
            environment_checksum=_checksum("f"),
            created_at=run_created,
            started_at=run_created,
        )

    query = RunQuery(
        snapshot_id=snapshot_id,
        strategy_id="monthly_momentum_v1",
        universe=("AAPL",),
        state=RunState.RUNNING,
        page_size=1,
    )
    first_page = store.search_runs(query)
    second_page = store.search_runs(
        RunQuery(
            snapshot_id=snapshot_id,
            strategy_id="monthly_momentum_v1",
            universe=("AAPL",),
            state=RunState.RUNNING,
            page=1,
            page_size=1,
        )
    )

    assert first_page.total_count == 2
    assert [record.run_id for record in first_page.records] == [first_id]
    assert [record.run_id for record in second_page.records] == [second_id]
    assert first_page.records[0].configuration_checksum == _checksum("e")
    assert not hasattr(first_page.records[0], "configuration")
    store.close()


def _create_running_run(store: DuckDBMetadataStore, run_id: UUID) -> None:
    store.create_run(
        run_id=run_id,
        snapshot_id=_snapshot().snapshot_id,
        strategy_id="monthly_momentum_v1",
        evaluation_start=_START,
        evaluation_end=_END,
        universe=("AAPL",),
        configuration_checksum=_checksum("e"),
        environment_checksum=_checksum("f"),
        created_at=_NOW,
        started_at=_NOW,
    )


def test_successful_finalization_and_identical_replay_are_idempotent(tmp_path: object) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    run_id = uuid4()
    _create_running_run(store, run_id)
    finalization = _successful_finalization()

    intent = store.create_finalization_intent(run_id, finalization, created_at=_NOW)
    assert intent.terminal_payload_checksum == finalization.payload_checksum
    store.mark_finalization_mlflow_synced(run_id, attempted_at=_NOW)
    terminal = store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=1))
    replay = store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=2))

    assert terminal == replay
    assert replay.state is RunState.SUCCEEDED
    assert replay.immutable
    store.close()


def test_conflicting_terminal_payload_is_rejected_without_mutation(tmp_path: object) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    run_id = uuid4()
    _create_running_run(store, run_id)
    finalization = _successful_finalization()
    store.create_finalization_intent(run_id, finalization, created_at=_NOW)
    store.mark_finalization_mlflow_synced(run_id, attempted_at=_NOW)
    terminal = store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=1))

    conflicting = RunFinalization(
        desired_state=RunState.SUCCEEDED,
        manifest_checksum=_checksum("b"),
        manifest_uri=finalization.manifest_uri,
        metrics=finalization.metrics,
    )
    with pytest.raises(ImmutableMetadataError, match="terminal"):
        store.finalize_run(run_id, conflicting, ended_at=_NOW + timedelta(seconds=2))
    assert store.get_run(run_id) == terminal
    store.close()


def test_failed_mlflow_sync_keeps_intent_running_and_recovers(tmp_path: object) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    run_id = uuid4()
    _create_running_run(store, run_id)
    finalization = RunFinalization(
        desired_state=RunState.FAILED,
        manifest_checksum=None,
        manifest_uri=None,
        errors=(_failure(),),
    )
    store.create_finalization_intent(run_id, finalization, created_at=_NOW)
    store.mark_finalization_mlflow_synced(
        run_id,
        attempted_at=_NOW,
        error=_failure(),
    )
    with pytest.raises(IllegalMetadataTransitionError, match="MLflow"):
        store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=1))
    assert store.get_run(run_id).state is RunState.RUNNING
    assert store.pending_finalizations()[0].run_id == run_id

    store.mark_finalization_mlflow_synced(
        run_id,
        attempted_at=_NOW + timedelta(seconds=1),
    )
    recovered = store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=2))
    assert recovered.state is RunState.FAILED
    assert store.pending_finalizations() == ()
    store.close()


def test_terminal_transition_requires_intent_and_rejects_illegal_payload(tmp_path: object) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    run_id = uuid4()
    _create_running_run(store, run_id)
    finalization = RunFinalization(
        desired_state=RunState.FAILED,
        manifest_checksum=None,
        manifest_uri=None,
        errors=(_failure(),),
    )
    with pytest.raises(IllegalMetadataTransitionError, match="intent"):
        store.finalize_run(run_id, finalization, ended_at=_NOW + timedelta(seconds=1))
    assert store.get_run(run_id).state is RunState.RUNNING
    store.close()


def test_run_discovery_supports_all_filters_successful_candidates_and_index_projection(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    snapshot_a = _snapshot().snapshot_id
    snapshot_b = "snap_" + _checksum("9")
    run_specs = (
        (
            UUID("00000000-0000-0000-0000-000000000101"),
            snapshot_a,
            "monthly_momentum_v1",
            (_START, _END),
            ("AAPL", "MSFT"),
            _NOW + timedelta(hours=1),
            RunState.SUCCEEDED,
        ),
        (
            UUID("00000000-0000-0000-0000-000000000102"),
            snapshot_a,
            "mean_reversion_v1",
            (date(2024, 1, 3), date(2024, 1, 6)),
            ("AAPL",),
            _NOW + timedelta(days=1),
            RunState.FAILED,
        ),
        (
            UUID("00000000-0000-0000-0000-000000000103"),
            snapshot_b,
            "monthly_momentum_v1",
            (_START, _END),
            ("MSFT",),
            _NOW + timedelta(days=2),
            RunState.SUCCEEDED,
        ),
        (
            UUID("00000000-0000-0000-0000-000000000104"),
            snapshot_b,
            "monthly_momentum_v1",
            (date(2024, 1, 3), date(2024, 1, 7)),
            ("AAPL", "MSFT"),
            _NOW + timedelta(days=3),
            RunState.RUNNING,
        ),
    )

    for (
        run_id,
        snapshot_id,
        strategy,
        evaluation,
        universe,
        created_at,
        state,
    ) in run_specs:
        store.create_run(
            run_id=run_id,
            snapshot_id=snapshot_id,
            strategy_id=strategy,
            evaluation_start=evaluation[0],
            evaluation_end=evaluation[1],
            universe=universe,
            configuration_checksum=_checksum("e"),
            environment_checksum=_checksum("f"),
            created_at=created_at,
            started_at=created_at,
        )
        if state is RunState.SUCCEEDED:
            finalization = _successful_finalization()
        elif state is RunState.FAILED:
            finalization = RunFinalization(
                desired_state=RunState.FAILED,
                manifest_checksum=None,
                manifest_uri=None,
                errors=(_failure(),),
            )
        else:
            continue
        store.create_finalization_intent(run_id, finalization, created_at=created_at)
        store.mark_finalization_mlflow_synced(run_id, attempted_at=created_at)
        store.finalize_run(
            run_id,
            finalization,
            ended_at=created_at + timedelta(seconds=1),
        )

    # Each individual filter is applied to the indexed metadata projection.
    assert [item.run_id for item in store.search_runs(
        RunQuery(run_id=run_specs[0][0])
    ).records] == [
        run_specs[0][0]
    ]
    assert [item.run_id for item in store.search_runs(
        RunQuery(snapshot_id=snapshot_a)
    ).records] == [
        run_specs[1][0],
        run_specs[0][0],
    ]
    assert [
        item.run_id
        for item in store.search_runs(
            RunQuery(strategy_id="monthly_momentum_v1")
        ).records
    ] == [run_specs[3][0], run_specs[2][0], run_specs[0][0]]
    assert [
        item.run_id
        for item in store.search_runs(RunQuery(universe=("aapl", " msft "))).records
    ] == [run_specs[3][0], run_specs[0][0]]
    assert [
        item.run_id
        for item in store.search_runs(RunQuery(evaluation_start=_START)).records
    ] == [run_specs[2][0], run_specs[0][0]]
    assert [
        item.run_id
        for item in store.search_runs(RunQuery(evaluation_end=_END)).records
    ] == [run_specs[2][0], run_specs[0][0]]
    assert [
        item.run_id
        for item in store.search_runs(RunQuery(state=RunState.SUCCEEDED)).records
    ] == [run_specs[2][0], run_specs[0][0]]
    assert [
        item.run_id
        for item in store.search_runs(
            RunQuery(
                created_from=_NOW + timedelta(days=1),
                created_to=_NOW + timedelta(days=2),
            )
        ).records
    ] == [run_specs[2][0], run_specs[1][0]]

    # Combined filters select the exact ordered universe and are suitable for
    # successful-only comparison candidate discovery.
    combined = store.search_runs(
        RunQuery(
            snapshot_id=snapshot_a,
            strategy_id="monthly_momentum_v1",
            universe=("AAPL", "MSFT"),
            evaluation_start=_START,
            evaluation_end=_END,
            state="succeeded",
            created_from=_NOW,
            created_to=_NOW + timedelta(days=1),
            page_size=100,
        )
    )
    assert [item.run_id for item in combined.records] == [run_specs[0][0]]
    assert combined.records[0].state is RunState.SUCCEEDED
    assert not hasattr(combined.records[0], "metrics")
    assert not hasattr(combined.records[0], "scientific_rows")

    # The order remains stable for equal timestamps and adjacent pages are
    # disjoint, while very large valid page numbers remain bounded and empty.
    first = store.search_runs(RunQuery(page=0, page_size=2))
    second = store.search_runs(RunQuery(page=1, page_size=2))
    assert [item.run_id for item in first.records] == [run_specs[3][0], run_specs[2][0]]
    assert [item.run_id for item in second.records] == [
        run_specs[1][0],
        run_specs[0][0],
    ]
    assert set(first.records).isdisjoint(second.records)
    assert store.search_runs(RunQuery(page=10**100, page_size=100)).records == ()

    index_names = {
        str(row[0])
        for row in store._connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'run'"
        ).fetchall()
    }
    assert "run_evaluation_range_idx" in index_names
    store.close()


def test_run_query_rejects_unbounded_page_sizes(tmp_path: object) -> None:
    with pytest.raises(ValueError, match="page_size"):
        RunQuery(page_size=0)
    with pytest.raises(ValueError, match="page_size"):
        RunQuery(page_size=101)
    with pytest.raises(ValueError, match="page"):
        RunQuery(page=-1)
