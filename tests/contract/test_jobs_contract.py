"""Durable synchronous-job contracts through DuckDB and JSONL adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from quant_research_platform.application.jobs import SynchronousJobManager
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.execution import JobOperation, JobStage, JobState
from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.logging import StructuredJsonlLogger


class ContractClock:
    """A deterministic UTC/monotonic clock for persistence assertions."""

    def __init__(self) -> None:
        self._utc = datetime(2024, 1, 2, 15, tzinfo=UTC)
        self._monotonic = Decimal("0")

    def utc_now(self) -> datetime:
        return self._utc

    def monotonic_seconds(self) -> Decimal:
        return self._monotonic

    def advance(self, seconds: str) -> None:
        duration = Decimal(seconds)
        self._monotonic += duration
        self._utc += timedelta(seconds=float(duration))


def test_partial_ingestion_progress_is_redacted_terminally_flushed_and_reloads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "metadata.duckdb"
    log_path = tmp_path / "logs" / "operations.jsonl"
    secret = "contract-secret"
    clock = ContractClock()
    callbacks = []

    with DuckDBMetadataStore(database) as repository:
        redactor = Redactor((secret,))
        manager = SynchronousJobManager(
            repository,
            StructuredJsonlLogger(log_path, redactor=redactor, utc_now=clock.utc_now),
            redactor=redactor,
            clock=clock,
        )
        job = manager.create(
            JobOperation.INGESTION,
            total_units=3,
            progress_callback=callbacks.append,
        )
        job.start(stage=JobStage.FETCHING)
        job.report(
            stage=JobStage.FETCHING,
            completed_units=3,
            warnings=(f"AAPL failed through {secret}", "MSFT completed"),
            context={"symbol": "AAPL", "credential": secret},
        )

        # The rapid progress update is emitted but deliberately not yet persisted.
        assert [event.sequence for event in repository.list_job_events(job.job_id)] == [
            0,
            1,
        ]
        clock.advance("0.10")
        terminal = job.complete(partially_succeeded=True)
        stored = repository.get_job(job.job_id)

        assert terminal.state is JobState.PARTIALLY_SUCCEEDED
        assert stored.state is JobState.PARTIALLY_SUCCEEDED
        assert stored.stage is JobStage.COMPLETED
        assert stored.warnings == ("AAPL failed through [REDACTED]", "MSFT completed")
        assert stored.started_at is not None
        assert stored.ended_at is not None
        assert [event.sequence for event in repository.list_job_events(job.job_id)] == [
            0,
            1,
            2,
        ]
        assert callbacks[-1].state is JobState.PARTIALLY_SUCCEEDED

    with DuckDBMetadataStore(database) as reopened:
        persisted = reopened.get_job(job.job_id)
        events = reopened.list_job_events(job.job_id)
        assert persisted.state is JobState.PARTIALLY_SUCCEEDED
        assert persisted.warnings == ("AAPL failed through [REDACTED]", "MSFT completed")
        assert events[-1].stage is JobStage.COMPLETED

    diagnostics = log_path.read_text(encoding="utf-8")
    assert secret not in diagnostics
    assert "[REDACTED]" in diagnostics
