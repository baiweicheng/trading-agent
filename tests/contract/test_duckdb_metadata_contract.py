"""DuckDB metadata boundary contracts using a real temporary database."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.domain.execution import (
    JobOperation,
    JobStage,
    JobState,
    ProgressUpdate,
)
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.infrastructure.duckdb_metadata import (
    DuckDBMetadataStore,
    IllegalMetadataTransitionError,
    ImmutableMetadataError,
    SnapshotAvailability,
)

_NOW = datetime(2024, 1, 10, 15, tzinfo=UTC)
_START = date(2024, 1, 2)
_END = date(2024, 1, 5)


def _checksum(character: str) -> str:
    return character * 64


def _snapshot() -> SnapshotManifest:
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
            version="fixture",
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
        operational_metadata=OperationalMetadata(created_at=_NOW),
    )


def _progress(job_id: UUID, state: JobState, stage: JobStage) -> ProgressUpdate:
    return ProgressUpdate(
        job_id=job_id,
        operation=JobOperation.INGESTION,
        state=state,
        stage=stage,
        completed_units=0 if state is JobState.NOT_STARTED else 1,
        total_units=1,
        elapsed_seconds=Decimal("0"),
    )


def test_metadata_migrates_reopens_and_enforces_insert_only_science(
    tmp_path: Path,
) -> None:
    database = tmp_path / "metadata.duckdb"
    manifest = _snapshot()
    manifest_uri = f"snapshots/{manifest.snapshot_id}/manifest.json"
    job_id = UUID("00000000-0000-0000-0000-000000000001")

    with DuckDBMetadataStore(database) as store:
        store.create_job(
            _progress(job_id, JobState.NOT_STARTED, JobStage.NOT_STARTED),
            updated_at=_NOW,
        )
        assert store.insert_snapshot(manifest, manifest_uri=manifest_uri)
        assert not store.insert_snapshot(manifest, manifest_uri=manifest_uri)

        conflicting_object = ContentAddressedObjectRef(
            object_kind=ObjectKind.NORMALIZED,
            checksum=_checksum("a"),
            relative_uri="objects/normalized/symbol=AAPL/year=2024/sha256=a.parquet",
            schema_version="daily_bar_v1",
            row_count=2,
            byte_size=128,
            symbol="AAPL",
            session_year=2024,
            media_type="application/vnd.apache.parquet",
        )
        with pytest.raises(
            ImmutableMetadataError, match="different immutable metadata"
        ):
            store.record_data_object(conflicting_object, created_at=_NOW)

        invalid = store.set_snapshot_availability(
            manifest.snapshot_id,
            SnapshotAvailability.INVALID,
        )
        assert invalid.manifest_uri == manifest_uri
        assert invalid.availability is SnapshotAvailability.INVALID

    with DuckDBMetadataStore(database) as reopened:
        snapshot = reopened.get_snapshot(manifest.snapshot_id)
        assert snapshot.manifest_uri == manifest_uri
        assert snapshot.availability is SnapshotAvailability.INVALID
        assert reopened.list_snapshot_objects(manifest.snapshot_id)[
            0
        ].checksum == _checksum("a")


def test_job_repository_rejects_illegal_and_terminal_progress_rewrites(
    tmp_path: Path,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    job_id = UUID("00000000-0000-0000-0000-000000000002")
    not_started = _progress(job_id, JobState.NOT_STARTED, JobStage.NOT_STARTED)
    running = _progress(job_id, JobState.RUNNING, JobStage.FETCHING)
    succeeded = _progress(job_id, JobState.SUCCEEDED, JobStage.COMPLETED)

    store.create_job(not_started, updated_at=_NOW)
    with pytest.raises(IllegalMetadataTransitionError, match="illegal job"):
        store.update_job(succeeded, updated_at=_NOW)

    store.update_job(running, updated_at=_NOW)
    terminal = store.update_job(succeeded, updated_at=_NOW)
    assert terminal.state is JobState.SUCCEEDED
    assert terminal.ended_at == _NOW

    with pytest.raises(IllegalMetadataTransitionError, match="terminal job progress"):
        store.update_job(succeeded, updated_at=_NOW)
    store.close()
