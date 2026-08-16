"""Focused synchronous job lifecycle, throttling, isolation, and persistence tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from quant_research_platform.application.jobs import SynchronousJobManager
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import Err, ErrorCategory, Ok
from quant_research_platform.domain.execution import JobOperation, JobStage, JobState
from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.logging import StructuredJsonlLogger


class FakeClock:
    """A UTC/monotonic clock that tests can move without sleeping."""

    def __init__(self) -> None:
        self._now = datetime(2024, 1, 2, 15, tzinfo=UTC)
        self._monotonic = Decimal("0")

    def utc_now(self) -> datetime:
        return self._now

    def monotonic_seconds(self) -> Decimal:
        return self._monotonic

    def advance(self, seconds: str) -> None:
        value = Decimal(seconds)
        self._monotonic += value
        self._now += timedelta(seconds=float(value))


class CountingMetadataStore:
    """Count mutable job writes while using the real DuckDB implementation."""

    def __init__(self, path: Path) -> None:
        self.store = DuckDBMetadataStore(path)
        self.update_count = 0

    def create_job(self, *args: object, **kwargs: object) -> object:
        return self.store.create_job(*args, **kwargs)  # type: ignore[arg-type]

    def update_job(self, *args: object, **kwargs: object) -> object:
        self.update_count += 1
        return self.store.update_job(*args, **kwargs)  # type: ignore[arg-type]

    def append_job_event(self, *args: object, **kwargs: object) -> object:
        return self.store.append_job_event(*args, **kwargs)  # type: ignore[arg-type]


def _manager(
    tmp_path: Path,
    *,
    clock: FakeClock,
    secret: str = "credential-123",
) -> tuple[SynchronousJobManager, CountingMetadataStore, Path]:
    log_path = tmp_path / "logs" / "operations.jsonl"
    repository = CountingMetadataStore(tmp_path / "metadata.duckdb")
    redactor = Redactor((secret,))
    logger = StructuredJsonlLogger(log_path, redactor=redactor, utc_now=clock.utc_now)
    return (
        SynchronousJobManager(repository, logger, redactor=redactor, clock=clock),
        repository,
        log_path,
    )


@pytest.mark.parametrize(
    ("operation", "partial", "expected"),
    [
        (JobOperation.INGESTION, False, JobState.SUCCEEDED),
        (JobOperation.INGESTION, True, JobState.PARTIALLY_SUCCEEDED),
        (JobOperation.BACKTEST, False, JobState.SUCCEEDED),
    ],
)
def test_legal_job_transitions_are_persisted(
    tmp_path: Path,
    operation: JobOperation,
    partial: bool,
    expected: JobState,
) -> None:
    clock = FakeClock()
    manager, repository, _ = _manager(tmp_path, clock=clock)
    job = manager.create(operation, total_units=2)

    assert job.state is JobState.NOT_STARTED
    job.start()
    clock.advance("0.25")
    job.report(stage=JobStage.FETCHING, completed_units=1)
    terminal = job.complete(partially_succeeded=partial)

    assert terminal.state is expected
    assert terminal.elapsed_seconds == Decimal("0.25")
    assert repository.store.get_job(job.job_id).state is expected
    assert [
        event.sequence for event in repository.store.list_job_events(job.job_id)
    ] == [0, 1, 2, 3]
    repository.store.close()


def test_illegal_job_transitions_are_rejected_without_mutating_terminal_record(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    manager, repository, _ = _manager(tmp_path, clock=clock)
    backtest = manager.create(JobOperation.BACKTEST, total_units=1)

    with pytest.raises(ValueError, match="only while the job is running"):
        backtest.complete()
    backtest.start()
    with pytest.raises(ValueError, match="illegal job state transition"):
        backtest.complete(partially_succeeded=True)
    assert backtest.state is JobState.RUNNING
    backtest.complete()
    persisted = repository.store.get_job(backtest.job_id)
    with pytest.raises(ValueError, match="only while the job is running"):
        backtest.report(stage=JobStage.EXECUTING, completed_units=1)
    with pytest.raises(ValueError, match="only while the job is running"):
        backtest.fail(
            # The type is checked after lifecycle validation, so use a real error only where needed.
            object()  # type: ignore[arg-type]
        )
    assert repository.store.get_job(backtest.job_id) == persisted
    repository.store.close()


def test_progress_throttles_to_four_writes_per_second_and_flushes_terminal_state(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    manager, repository, _ = _manager(tmp_path, clock=clock)
    job = manager.create(JobOperation.INGESTION, total_units=5)
    job.start()

    for completed in range(1, 5):
        clock.advance("0.05")
        job.report(stage=JobStage.FETCHING, completed_units=completed)
    # The start transition is one write; the rapid reports have not crossed 250ms.
    assert repository.update_count == 1

    clock.advance("0.05")
    job.report(stage=JobStage.NORMALIZING, completed_units=5)
    assert repository.update_count == 2

    # A terminal update is persisted immediately even though the interval has not elapsed.
    clock.advance("0.01")
    terminal = job.complete()
    assert repository.update_count == 3
    assert terminal.elapsed_seconds == Decimal("0.26")
    assert repository.store.get_job(job.job_id).state is JobState.SUCCEEDED
    repository.store.close()


def test_warnings_are_redacted_accumulated_and_keep_symbol_failures_isolated(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    secret = "https://user:credential-123@proxy.example"
    manager, repository, log_path = _manager(tmp_path, clock=clock, secret=secret)
    job = manager.create(JobOperation.INGESTION, total_units=2)
    job.start()
    job.report(
        stage=JobStage.FETCHING,
        completed_units=1,
        warnings=(f"AAPL provider failure via {secret}",),
        context={"symbol": "AAPL", "url": secret},
    )
    clock.advance("0.25")
    job.report(
        stage=JobStage.FETCHING,
        completed_units=2,
        warnings=("MSFT completed successfully",),
        context={"symbol": "MSFT"},
    )
    job.complete(partially_succeeded=True)

    persisted = repository.store.get_job(job.job_id)
    assert persisted.state is JobState.PARTIALLY_SUCCEEDED
    assert persisted.warnings == (
        "AAPL provider failure via [REDACTED]",
        "MSFT completed successfully",
    )
    events = repository.store.list_job_events(job.job_id)
    assert any("AAPL" in event.context_json for event in events)
    assert any("MSFT" in event.context_json for event in events)
    contents = log_path.read_text(encoding="utf-8")
    assert secret not in contents
    assert "[REDACTED]" in contents
    repository.store.close()


def test_execute_converts_boundary_exception_persists_diagnostics_and_preserves_prior_job_after_reload(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    manager, repository, log_path = _manager(tmp_path, clock=clock)
    first_job = UUID("00000000-0000-0000-0000-000000000001")
    failing_job = UUID("00000000-0000-0000-0000-000000000002")

    successful = manager.execute(
        JobOperation.INGESTION,
        lambda job: "snap_previous",
        total_units=1,
        job_id=first_job,
    )
    assert isinstance(successful, Ok)
    assert successful.value.state is JobState.SUCCEEDED

    secret = "credential-123"

    def fail_after_progress(job: object) -> str:
        assert hasattr(job, "report")
        typed_job = job  # keep a real boundary exception with a secret payload
        typed_job.report(  # type: ignore[union-attr]
            stage=JobStage.FETCHING,
            completed_units=1,
            warnings=(f"provider warning {secret}",),
        )
        raise RuntimeError(f"provider response contained {secret}")

    failed = manager.execute(
        JobOperation.INGESTION,
        fail_after_progress,
        total_units=1,
        job_id=failing_job,
    )
    assert isinstance(failed, Err)
    error = failed.errors[0]
    assert error.category is ErrorCategory.INTERNAL_UNEXPECTED
    assert error.correlation_id == str(failing_job)
    assert repository.store.get_job(failing_job).state is JobState.FAILED
    assert repository.store.get_job(first_job).state is JobState.SUCCEEDED
    repository.store.close()

    with DuckDBMetadataStore(tmp_path / "metadata.duckdb") as reopened:
        assert reopened.get_job(first_job).state is JobState.SUCCEEDED
        failed_record = reopened.get_job(failing_job)
        assert failed_record.state is JobState.FAILED
        assert failed_record.error_json is not None
        assert secret not in failed_record.error_json
        assert len(reopened.list_job_events(failing_job)) >= 3

    diagnostics = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(item["correlation_id"] for item in diagnostics)
    assert secret not in log_path.read_text(encoding="utf-8")
