"""Durable snapshot, ingestion-failure, and incremental integration coverage.

All provider data in this module is generated locally.  The fixture deliberately
uses the canonical provider name ``yfinance`` because the raw schema is pinned to
that Phase 1 value, but it never imports or calls the Yahoo Finance adapter.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import UUID

import pytest

from quant_research_platform.application.incremental import (
    IncrementalMerger,
    IncrementalParent,
)
from quant_research_platform.application.ingestion import (
    DataIngestionService,
    IngestionRequest,
)
from quant_research_platform.application.jobs import SynchronousJobManager
from quant_research_platform.application.snapshots import SnapshotManager
from quant_research_platform.config.models import (
    DataConfig,
    DateRangeConfig,
    PathConfig,
    ResolvedConfig,
    RetryPolicyConfig,
)
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.canonical import sha256_bytes, sha256_canonical_json
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    ProviderFailureKind,
    ProviderFailureReason,
)
from quant_research_platform.domain.execution import JobStage, JobState
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
    RawCorporateAction,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
    ValidationSummary,
)
from quant_research_platform.domain.normalization import (
    CausalForwardAdjustmentV1,
    Normalizer,
)
from quant_research_platform.domain.validation import ValidationService
from quant_research_platform.infrastructure.duckdb_metadata import (
    DuckDBMetadataStore,
    IngestionOperationStatus,
    MetadataNotFoundError,
    SnapshotAvailability,
)
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    SnapshotPublicationCandidate,
)
from quant_research_platform.infrastructure.logging import StructuredJsonlLogger
from quant_research_platform.infrastructure.parquet_store import (
    ParquetStore,
    StagedParquetObject,
)


SESSIONS = tuple(date(2024, 1, day) for day in (2, 3, 4, 5))
PARENT_RANGE = DateRange(SESSIONS[0], SESSIONS[1])
EXTENDED_RANGE = DateRange(SESSIONS[0], SESSIONS[-1])
SYMBOLS = ("AAPL", "MSFT")
ALL_SYMBOLS = (*SYMBOLS, "SPY")
_NOW = datetime(2024, 1, 8, 22, tzinfo=UTC)

PUBLICATION_FAULT_POINTS = (
    "before_snapshot_object_checksum",
    "after_snapshot_object_checksum",
    "before_snapshot_object_promotion",
    "after_snapshot_object_promotion",
    "before_validation_checksum",
    "after_validation_checksum",
    "before_validation_promotion",
    "after_validation_promotion",
    "before_publication_write",
    "after_publication_write",
    "before_publication_directory_fsync",
    "after_publication_directory_fsync",
    "before_publication_rename",
    "after_publication_rename",
    "after_publication_parent_fsync",
    "before_duckdb_commit",
    "after_duckdb_commit",
)


class FixtureCalendar:
    """A deterministic XNYS-shaped calendar with no external calendar package."""

    name = "XNYS"
    version = "offline-fixture-1"

    def __init__(self, sessions: Sequence[date] = SESSIONS) -> None:
        self._sessions = tuple(sessions)
        self.identity = CalendarIdentity(
            self.name,
            self.version,
            sha256_canonical_json([session.isoformat() for session in self._sessions]),
        )

    def is_session(self, value: date) -> bool:
        return value in self._sessions

    def sessions(
        self,
        start: date,
        end: date,
        *,
        completed_at: datetime | None = None,
    ) -> tuple[date, ...]:
        del completed_at
        return tuple(
            session for session in self._sessions if start <= session <= end
        )

    def close_timestamp(self, session: date) -> datetime:
        if not self.is_session(session):
            raise ValueError("not an offline fixture session")
        return datetime.combine(session, datetime.min.time(), tzinfo=UTC).replace(
            hour=21
        )

    def schedule_checksum(self, start: date, end: date) -> str:
        return sha256_canonical_json(
            [session.isoformat() for session in self.sessions(start, end)]
        )


class FixedIngestionClock:
    def utc_now(self) -> datetime:
        return _NOW


class FixedJobClock:
    def __init__(self) -> None:
        self._ticks = 0

    def utc_now(self) -> datetime:
        return _NOW

    def monotonic_seconds(self) -> Decimal:
        self._ticks += 1
        return Decimal(self._ticks)


class OfflineYFinanceFixture:
    """Deterministic provider seam named yfinance without network behavior."""

    name = "yfinance"

    def __init__(
        self,
        calendar: FixtureCalendar,
        *,
        failed_symbols: Iterable[str] = (),
        revised_actions: Mapping[tuple[str, date], Decimal] | None = None,
        reverse_records: bool = False,
    ) -> None:
        self.calendar = calendar
        self.failed_symbols = {symbol.upper() for symbol in failed_symbols}
        self.revised_actions = dict(revised_actions or {})
        self.reverse_records = reverse_records
        self.requests: list[ProviderRequest] = []

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        self.requests.append(request)
        outcomes: list[SymbolOutcome] = []
        for symbol in request.symbols:
            if symbol in self.failed_symbols:
                outcomes.append(self._failure(symbol))
                continue
            records = self.records_for(request, symbol=symbol)
            if not records:
                outcomes.append(
                    self._failure(
                        symbol,
                        reason=ProviderFailureReason.EMPTY_RESPONSE,
                    )
                )
            else:
                outcomes.append(
                    SymbolOutcome(
                        symbol=symbol,
                        status=SymbolOutcomeStatus.SUCCESS,
                        attempts=1,
                        records=records,
                    )
                )
        return ProviderBatchResult(request=request, outcomes=tuple(outcomes))

    def records_for(
        self,
        request: ProviderRequest,
        *,
        symbol: str | None = None,
    ) -> tuple[ProviderRecord, ...]:
        selected = request.symbols if symbol is None else (symbol,)
        records: list[ProviderRecord] = []
        for requested_symbol in selected:
            base = {
                "AAPL": Decimal("100"),
                "MSFT": Decimal("200"),
                "SPY": Decimal("300"),
            }[requested_symbol]
            for ordinal, session in enumerate(
                self.calendar.sessions(request.start, request.end), start=1
            ):
                close = base + Decimal(ordinal)
                split_ratio = self.revised_actions.get(
                    (requested_symbol, session), Decimal("1")
                )
                records.append(
                    ProviderRecord(
                        provider=self.name,
                        request_content_key=request.content_key,
                        symbol=requested_symbol,
                        raw_bar=RawDailyBar(
                            provider_date=session,
                            open=close - Decimal("1"),
                            high=close + Decimal("1"),
                            low=close - Decimal("2"),
                            close=close,
                            adj_close=close,
                            volume=Decimal("1000"),
                        ),
                        raw_action=RawCorporateAction(
                            dividend=Decimal("0"),
                            split_ratio=split_ratio,
                            provider_fields={"fixture": "offline"},
                        ),
                        provider_fields={"fixture": "offline"},
                    )
                )
        if self.reverse_records:
            records.reverse()
        return tuple(records)

    @staticmethod
    def _failure(
        symbol: str,
        *,
        reason: ProviderFailureReason = ProviderFailureReason.EMPTY_RESPONSE,
    ) -> SymbolOutcome:
        error = ActionableError(
            operation="provider.fetch",
            category=ErrorCategory.PROVIDER_TERMINAL,
            message=f"Offline fixture returned no usable records for {symbol}.",
            corrective_action="Retry the affected symbol or inspect the provider adapter.",
            symbol=symbol,
        )
        return SymbolOutcome(
            symbol=symbol,
            status=SymbolOutcomeStatus.FAILURE,
            attempts=1,
            failure_kind=ProviderFailureKind.TERMINAL,
            failure_reason=reason,
            errors=(error,),
        )


class SnapshotParquetWriter:
    """Adapt logical Parquet URIs to the FilesystemStore CAS namespace.

    ``ParquetStore`` intentionally emits collection-relative paths such as
    ``raw/...``.  Snapshot CAS references require ``objects/...``.  The test
    adapter changes only the manifest-facing URI; the staged file path and its
    final bytes remain owned by the real Parquet writer.
    """

    def __init__(self, root: Path) -> None:
        self.store = ParquetStore(root, write_chunk_size=2, scan_batch_size=2)

    @staticmethod
    def _cas_namespace(
        values: Iterable[StagedParquetObject],
    ) -> tuple[StagedParquetObject, ...]:
        return tuple(
            replace(value, relative_uri=f"objects/{value.relative_uri}")
            for value in values
        )

    def write_raw(
        self,
        rows: Iterable[ProviderRecord],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[StagedParquetObject, ...]:
        return self._cas_namespace(
            self.store.write_raw(
                rows,
                write_chunk_size=write_chunk_size,
                staging=staging,
            )
        )

    def write_normalized(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[StagedParquetObject, ...]:
        return self._cas_namespace(
            self.store.write_normalized(
                rows,
                write_chunk_size=write_chunk_size,
                staging=staging,
            )
        )


@dataclass(frozen=True, slots=True)
class PublicationFixture:
    manifest: SnapshotManifest
    object_bytes: bytes
    report_bytes: bytes

    def candidate(self) -> SnapshotPublicationCandidate:
        reference = self.manifest.content_identity.objects[0]
        return SnapshotPublicationCandidate(
            self.manifest,
            staged_objects={reference.relative_uri: self.object_bytes},
            validation_report=self.report_bytes,
        )


def _publication_fixture(label: str) -> PublicationFixture:
    object_bytes = f"normalized fixture object {label}".encode("utf-8")
    report_bytes = f"validation fixture report {label}".encode("utf-8")
    object_checksum = sha256_bytes(object_bytes)
    report_checksum = sha256_bytes(report_bytes)
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=object_checksum,
        relative_uri=(
            "objects/normalized/symbol=AAPL/year=2024/"
            f"sha256={object_checksum}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=1,
        byte_size=len(object_bytes),
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    covered = DateRange(date(2024, 1, 2), date(2024, 1, 3))
    identity = SnapshotContentIdentity(
        provider="offline-fixture",
        requested_range=covered,
        covered_range=covered,
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "fixture", "a" * 64),
        configuration_checksum="b" * 64,
        objects=(reference,),
        validation_report_checksum=report_checksum,
        validation_summary=ValidationSummary(
            accepted_row_count=1,
            quarantined_row_count=0,
            collapsed_duplicate_count=0,
            gap_count=0,
            covered_range=covered,
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    return PublicationFixture(
        SnapshotManifest(
            content_identity=identity,
            operational_metadata=OperationalMetadata(
                created_at=datetime(2024, 1, 10, 15, tzinfo=UTC)
            ),
        ),
        object_bytes,
        report_bytes,
    )


def _config(*, end: date, revision_overlap: int = 0) -> ResolvedConfig:
    return ResolvedConfig(
        paths=PathConfig(),
        retry=RetryPolicyConfig(
            attempts=1,
            initial_delay_seconds=Decimal("0"),
            max_delay_seconds=Decimal("0"),
            backoff_multiplier=Decimal("1"),
        ),
        data=DataConfig(
            universe=SYMBOLS,
            requested_range=DateRangeConfig(start=SESSIONS[0], end=end),
            batch_size=3,
            staleness_sessions=10,
            revision_overlap_sessions=revision_overlap,
            write_chunk_rows=2,
        ),
    )


def _job_manager(root: Path, metadata: DuckDBMetadataStore) -> SynchronousJobManager:
    logger = StructuredJsonlLogger(
        root / "diagnostics.jsonl",
        redactor=Redactor(),
        utc_now=lambda: _NOW,
    )
    return SynchronousJobManager(
        metadata,
        logger,
        redactor=Redactor(),
        clock=FixedJobClock(),
    )


def _service(
    root: Path,
    calendar: FixtureCalendar,
    provider: OfflineYFinanceFixture,
    store: FilesystemStore,
    metadata: DuckDBMetadataStore,
    jobs: SynchronousJobManager,
    *,
    incremental_merger: IncrementalMerger | None = None,
) -> DataIngestionService:
    policy = CausalForwardAdjustmentV1()
    return DataIngestionService(
        provider,
        calendar,
        normalizer=Normalizer(policy),
        validator=ValidationService(calendar=calendar, benchmark_symbol="SPY"),
        parquet_store=SnapshotParquetWriter(root / "parquet"),
        snapshot_publisher=store,
        metadata=metadata,
        job_manager=jobs,
        policy=policy,
        incremental_merger=incremental_merger,
        clock=FixedIngestionClock(),
        sleep=lambda _seconds: None,
        redactor=Redactor(),
    )


def _parent_from_result(
    result: object,
    calendar: FixtureCalendar,
) -> IncrementalParent:
    manifest = result.manifest
    assert isinstance(manifest, SnapshotManifest)
    requested_range = manifest.content_identity.requested_range
    sessions = calendar.sessions(requested_range.start, requested_range.end)
    return IncrementalParent.from_manifest(
        manifest,
        accepted_rows=result.accepted_rows,
        provider_records=result.provider_records,
        expected_sessions={symbol: sessions for symbol in ALL_SYMBOLS},
        validation_report=result.validation.report,
    )


def _failed_operation(metadata: DuckDBMetadataStore):
    with metadata._lock:  # type: ignore[attr-defined]
        row = metadata._connection.execute(  # type: ignore[attr-defined]
            """
            SELECT operation_id FROM ingestion_operation
            WHERE status = 'failed'
            ORDER BY created_at DESC, operation_id DESC LIMIT 1
            """
        ).fetchone()
    assert row is not None
    return metadata.get_ingestion_operation(UUID(str(row[0])))


def _fixture_parent(
    calendar: FixtureCalendar,
    records: tuple[ProviderRecord, ...],
    requested_range: DateRange,
) -> IncrementalParent:
    normalized = tuple(Normalizer(CausalForwardAdjustmentV1()).normalize(records, calendar))
    sessions = calendar.sessions(requested_range.start, requested_range.end)
    expected = {symbol: sessions for symbol in ALL_SYMBOLS}
    validation = ValidationService(calendar=calendar, benchmark_symbol="SPY").validate(
        normalized,
        expected,
        10,
        requested_range=requested_range,
        benchmark_symbol="SPY",
        calendar=calendar,
    )
    manifest = SnapshotManifest(
        content_identity=SnapshotContentIdentity(
            provider="yfinance",
            requested_range=requested_range,
            covered_range=validation.report.summary.covered_range,
            configured_universe=SYMBOLS,
            benchmark_symbol="SPY",
            calendar=calendar.identity,
            configuration_checksum="c" * 64,
            objects=(),
            validation_report_checksum=validation.report.content_checksum,
            validation_summary=validation.report.summary,
            limitation_disclosure=LimitationDisclosure.current(),
        ),
        operational_metadata=OperationalMetadata(created_at=_NOW),
    )
    return IncrementalParent.from_manifest(
        manifest,
        accepted_rows=validation.accepted_rows,
        provider_records=records,
        expected_sessions=expected,
        validation_report=validation.report,
    )


@pytest.mark.integration
@pytest.mark.parametrize("fault_point", PUBLICATION_FAULT_POINTS)
def test_every_publication_boundary_preserves_prior_and_retry_identity(
    tmp_path: Path,
    fault_point: str,
) -> None:
    """A failed publication is either hidden or durably indexed only at commit."""

    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    prior = _publication_fixture("prior")
    candidate = _publication_fixture("candidate")
    store.publish_snapshot(prior.candidate(), operation_id="prior")

    fired: list[str] = []

    def fail_once(point: str) -> None:
        fired.append(point)
        if point == fault_point and point not in fired[:-1]:
            raise RuntimeError(f"injected publication fault: {point}")

    with pytest.raises(RuntimeError, match=fault_point):
        store.publish_snapshot(
            candidate.candidate(),
            operation_id="candidate-fault",
            fault_injector=fail_once,
        )
    assert fault_point in fired

    manager = SnapshotManager(storage=store, metadata=metadata)
    assert isinstance(manager.open_verified(prior.manifest.snapshot_id), Ok)
    if fault_point == "after_duckdb_commit":
        assert metadata.get_snapshot(candidate.manifest.snapshot_id).availability is SnapshotAvailability.AVAILABLE
    else:
        with pytest.raises(MetadataNotFoundError):
            metadata.get_snapshot(candidate.manifest.snapshot_id)

    retry = store.publish_snapshot(candidate.candidate(), operation_id="candidate-retry")
    assert retry.snapshot_id == candidate.manifest.snapshot_id
    assert metadata.get_snapshot(candidate.manifest.snapshot_id).availability is SnapshotAvailability.AVAILABLE
    assert isinstance(manager.open_verified(candidate.manifest.snapshot_id), Ok)

    clean_metadata = DuckDBMetadataStore(tmp_path / "clean-metadata.duckdb")
    clean_store = FilesystemStore(tmp_path / "clean-store", metadata=clean_metadata)
    clean = clean_store.publish_snapshot(candidate.candidate(), operation_id="clean")
    assert retry.manifest.snapshot_id == clean.manifest.snapshot_id
    assert retry.manifest.to_content_identity_dict() == clean.manifest.to_content_identity_dict()
    assert tuple(ref.checksum for ref in retry.manifest.content_identity.objects) == tuple(
        ref.checksum for ref in clean.manifest.content_identity.objects
    )
    for reference in candidate.manifest.content_identity.objects:
        assert store.read_object(reference.relative_uri) == candidate.object_bytes
        assert clean_store.read_object(reference.relative_uri) == candidate.object_bytes
    assert store.read_by_checksum(candidate.manifest.content_identity.validation_report_checksum) == candidate.report_bytes
    assert clean_store.read_by_checksum(candidate.manifest.content_identity.validation_report_checksum) == candidate.report_bytes
    assert tuple(item.snapshot_id for item in metadata.list_snapshots()) == (
        candidate.manifest.snapshot_id,
        prior.manifest.snapshot_id,
    ) or {
        item.snapshot_id for item in metadata.list_snapshots()
    } == {prior.manifest.snapshot_id, candidate.manifest.snapshot_id}
    metadata.close()
    clean_metadata.close()


@pytest.mark.integration
def test_real_ingestion_persists_cas_manifest_index_operation_and_progress(
    tmp_path: Path,
) -> None:
    calendar = FixtureCalendar()
    provider = OfflineYFinanceFixture(calendar)
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    jobs = _job_manager(tmp_path, metadata)

    result = _service(tmp_path, calendar, provider, store, metadata, jobs).ingest(
        _config(end=SESSIONS[-1])
    )

    assert isinstance(result, Ok)
    value = result.value
    assert value.job_state is JobState.SUCCEEDED
    assert value.snapshot_id.startswith("snap_")
    assert provider.requests
    assert all(request.provider == "yfinance" for request in provider.requests)
    assert all(request.symbols == ALL_SYMBOLS for request in provider.requests)

    record = metadata.get_snapshot(value.snapshot_id)
    assert record.availability is SnapshotAvailability.AVAILABLE
    assert record.manifest_uri == f"snapshots/{value.snapshot_id}/manifest.json"
    references = value.manifest.content_identity.objects
    assert references
    assert {reference.object_kind for reference in references} == {
        ObjectKind.RAW,
        ObjectKind.NORMALIZED,
    }
    assert all(reference.relative_uri.startswith("objects/") for reference in references)
    indexed_objects = metadata.list_snapshot_objects(value.snapshot_id)
    assert len(indexed_objects) == len(references)
    for reference in references:
        stored = store.read_object(reference.relative_uri)
        assert len(stored) == reference.byte_size
        assert sha256_bytes(stored) == reference.checksum
    report_bytes = store.read_by_checksum(
        value.manifest.content_identity.validation_report_checksum
    )
    assert sha256_bytes(report_bytes) == value.manifest.content_identity.validation_report_checksum

    assert value.operation_id is not None
    operation = metadata.get_ingestion_operation(value.operation_id)
    assert operation.status is IngestionOperationStatus.SUCCEEDED
    assert operation.result_snapshot_id == value.snapshot_id
    assert operation.mode == "full"
    assert value.job_id is not None
    job = metadata.get_job(value.job_id)
    assert job.state is JobState.SUCCEEDED
    assert job.stage is JobStage.COMPLETED
    assert job.completed_units == job.total_units == len(ALL_SYMBOLS)
    events = metadata.list_job_events(value.job_id)
    assert events
    assert tuple(event.sequence for event in events) == tuple(range(len(events)))
    with metadata._lock:  # type: ignore[attr-defined]
        request_count = metadata._connection.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) FROM provider_request WHERE job_id = ?",
            [str(value.job_id)],
        ).fetchone()[0]
        outcome_count = metadata._connection.execute(  # type: ignore[attr-defined]
            """
            SELECT COUNT(*) FROM provider_symbol_outcome outcome
            JOIN provider_request request ON request.request_id = outcome.request_id
            WHERE request.job_id = ?
            """,
            [str(value.job_id)],
        ).fetchone()[0]
    assert request_count == len(provider.requests) == 1
    assert outcome_count == len(ALL_SYMBOLS)
    assert (tmp_path / "diagnostics.jsonl").is_file()
    metadata.close()


@pytest.mark.integration
def test_ingestion_fault_leaves_prior_available_and_reconciles_on_retry(
    tmp_path: Path,
) -> None:
    calendar = FixtureCalendar()
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    first_store = FilesystemStore(tmp_path / "store", metadata=metadata)
    jobs = _job_manager(tmp_path, metadata)
    first = _service(
        tmp_path,
        calendar,
        OfflineYFinanceFixture(calendar),
        first_store,
        metadata,
        jobs,
    ).ingest(_config(end=PARENT_RANGE.end))
    assert isinstance(first, Ok)
    prior_id = first.value.snapshot_id

    fired = False

    def fail_after_rename(point: str) -> None:
        nonlocal fired
        if point == "after_publication_rename" and not fired:
            fired = True
            raise RuntimeError("offline injected rename interruption")

    interrupted_store = FilesystemStore(
        tmp_path / "store",
        metadata=metadata,
        failure_injector=fail_after_rename,
    )
    failed = _service(
        tmp_path,
        calendar,
        OfflineYFinanceFixture(calendar),
        interrupted_store,
        metadata,
        jobs,
    ).ingest(_config(end=EXTENDED_RANGE.end))
    assert isinstance(failed, Err)
    assert any(error.category is ErrorCategory.INTERNAL_UNEXPECTED for error in failed.errors)
    assert fired

    manager = SnapshotManager(storage=interrupted_store, metadata=metadata)
    assert isinstance(manager.open_verified(prior_id), Ok)
    assert tuple(item.snapshot_id for item in metadata.list_snapshots()) == (prior_id,)
    published_ids = set(interrupted_store.list_published_manifest_ids())
    candidate_ids = published_ids - {prior_id}
    assert len(candidate_ids) == 1
    candidate_id = next(iter(candidate_ids))
    operation = _failed_operation(metadata)
    assert operation.status is IngestionOperationStatus.FAILED
    failed_job = metadata.get_job(operation.job_id)
    assert failed_job.state is JobState.FAILED
    assert failed_job.stage is JobStage.FAILED

    restarted_store = FilesystemStore(tmp_path / "store", metadata=metadata)
    reconciliation = restarted_store.reconcile()
    assert reconciliation.indexed_snapshot_ids == (candidate_id,)
    assert isinstance(SnapshotManager(storage=restarted_store, metadata=metadata).open_verified(candidate_id), Ok)
    assert isinstance(manager.open_verified(prior_id), Ok)

    retry_provider = OfflineYFinanceFixture(calendar)
    retried = _service(
        tmp_path,
        calendar,
        retry_provider,
        restarted_store,
        metadata,
        jobs,
    ).ingest(_config(end=EXTENDED_RANGE.end))
    assert isinstance(retried, Ok)
    assert retried.value.snapshot_id == candidate_id
    assert retried.value.snapshot_reused is True
    assert len(metadata.list_snapshots()) == 2
    metadata.close()


@pytest.mark.integration
def test_incremental_service_uses_no_request_for_unchanged_and_rebuilds_revised_suffix(
    tmp_path: Path,
) -> None:
    calendar = FixtureCalendar()
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    jobs = _job_manager(tmp_path, metadata)
    initial_provider = OfflineYFinanceFixture(calendar)
    initial = _service(
        tmp_path,
        calendar,
        initial_provider,
        store,
        metadata,
        jobs,
    ).ingest(_config(end=PARENT_RANGE.end))
    assert isinstance(initial, Ok)
    parent = _parent_from_result(initial.value, calendar)

    unchanged_provider = OfflineYFinanceFixture(calendar)
    unchanged = _service(
        tmp_path,
        calendar,
        unchanged_provider,
        store,
        metadata,
        jobs,
        incremental_merger=IncrementalMerger(
            calendar,
            normalizer=Normalizer(CausalForwardAdjustmentV1()),
            policy=CausalForwardAdjustmentV1(),
            validator=ValidationService(calendar=calendar, benchmark_symbol="SPY"),
        ),
    ).ingest(
        IngestionRequest(parent_snapshot=parent),
        _config(end=PARENT_RANGE.end, revision_overlap=0),
    )
    assert isinstance(unchanged, Ok)
    assert unchanged.value.snapshot_id == parent.snapshot_id
    assert unchanged.value.snapshot_reused is True
    assert unchanged_provider.requests == []

    revised_provider = OfflineYFinanceFixture(
        calendar,
        revised_actions={("AAPL", PARENT_RANGE.end): Decimal("2")},
    )
    revised = _service(
        tmp_path,
        calendar,
        revised_provider,
        store,
        metadata,
        jobs,
        incremental_merger=IncrementalMerger(
            calendar,
            normalizer=Normalizer(CausalForwardAdjustmentV1()),
            policy=CausalForwardAdjustmentV1(),
            validator=ValidationService(calendar=calendar, benchmark_symbol="SPY"),
        ),
    ).ingest(
        IngestionRequest(parent_snapshot=parent),
        _config(end=EXTENDED_RANGE.end, revision_overlap=1),
    )
    assert isinstance(revised, Ok)
    assert revised.value.snapshot_id != parent.snapshot_id
    assert len(revised_provider.requests) == 1
    assert revised_provider.requests[0].start == PARENT_RANGE.end
    assert revised_provider.requests[0].end == EXTENDED_RANGE.end
    assert len({row.session_key for row in revised.value.accepted_rows}) == len(
        revised.value.accepted_rows
    )
    prior_by_key = {
        (row.symbol, row.session): row for row in parent.accepted_rows
    }
    revised_by_key = {
        (row.symbol, row.session): row for row in revised.value.accepted_rows
    }
    for key, row in prior_by_key.items():
        if key[1] < PARENT_RANGE.end:
            assert revised_by_key[key].to_content_dict() == row.to_content_dict()
    changed = revised_by_key[("AAPL", PARENT_RANGE.end)]
    assert changed.cumulative_price_factor == Decimal("2.000000000000000000")
    assert prior_by_key[("AAPL", PARENT_RANGE.end)].cumulative_price_factor == Decimal(
        "1.000000000000000000"
    )
    assert isinstance(
        SnapshotManager(storage=store, metadata=metadata).open_verified(
            revised.value.snapshot_id
        ),
        Ok,
    )
    metadata.close()


@pytest.mark.integration
def test_incremental_batch_order_confluence_and_failed_symbol_coverage() -> None:
    calendar = FixtureCalendar()
    base_provider = OfflineYFinanceFixture(calendar)
    parent_request = ProviderRequest(
        ALL_SYMBOLS,
        PARENT_RANGE.start,
        PARENT_RANGE.end,
        provider="yfinance",
    )
    parent_records = base_provider.records_for(parent_request)
    parent = _fixture_parent(calendar, parent_records, PARENT_RANGE)
    merger = IncrementalMerger(
        calendar,
        normalizer=Normalizer(CausalForwardAdjustmentV1()),
        policy=CausalForwardAdjustmentV1(),
        validator=ValidationService(calendar=calendar, benchmark_symbol="SPY"),
    )

    revised_provider = OfflineYFinanceFixture(
        calendar,
        revised_actions={("AAPL", PARENT_RANGE.end): Decimal("2")},
    )
    suffix_request = ProviderRequest(
        ALL_SYMBOLS,
        PARENT_RANGE.end,
        EXTENDED_RANGE.end,
        provider="yfinance",
    )
    suffix_records = revised_provider.records_for(suffix_request)
    first = merger.merge_or_raise(
        parent,
        EXTENDED_RANGE,
        revision_overlap=1,
        records=suffix_records,
    )
    permuted = merger.merge_or_raise(
        parent,
        EXTENDED_RANGE,
        revision_overlap=1,
        records=tuple(reversed(suffix_records)),
    )
    assert first.snapshot_id != parent.snapshot_id
    assert first.content_identity == permuted.content_identity
    assert tuple(row.to_content_dict() for row in first.accepted_rows) == tuple(
        row.to_content_dict() for row in permuted.accepted_rows
    )
    plan = first.plan
    assert plan.overlap_sessions == (PARENT_RANGE.end,)
    assert plan.later_sessions == SESSIONS[2:]
    assert plan.suffix_sessions == SESSIONS[1:]
    assert plan.boundary_session == PARENT_RANGE.end
    assert plan.suffix_range == DateRange(PARENT_RANGE.end, EXTENDED_RANGE.end)

    zero_plan = merger.plan(parent, EXTENDED_RANGE, revision_overlap=0)
    assert zero_plan.overlap_sessions == ()
    assert zero_plan.later_sessions == SESSIONS[2:]
    assert zero_plan.suffix_sessions == SESSIONS[2:]
    assert zero_plan.boundary_session == SESSIONS[2]
    with pytest.raises(ValueError, match="must not precede"):
        merger.plan(parent, DateRange(PARENT_RANGE.start, PARENT_RANGE.start), 1)
    with pytest.raises(ValueError, match="must equal"):
        merger.plan(parent, DateRange(date(2024, 1, 1), EXTENDED_RANGE.end), 1)

    retained = merger.merge_or_raise(
        parent,
        PARENT_RANGE,
        revision_overlap=1,
        failed_symbols=("AAPL",),
        records=(),
    )
    assert retained.failed_symbols == ("AAPL",)
    assert retained.retained_parent_coverage_symbols == ("AAPL",)
    assert retained.new_rows == ()
    assert any(row.symbol == "AAPL" for row in retained.accepted_rows)
    assert retained.limitation_disclosure.data_failures

    missing_parent = IncrementalParent.from_manifest(
        parent.manifest,
        accepted_rows=(row for row in parent.accepted_rows if row.symbol != "MSFT"),
        provider_records=(
            record for record in parent.provider_records if record.symbol != "MSFT"
        ),
        expected_sessions=parent.expected_map(),
        validation_report=parent.validation_report,
    )
    missing = merger.merge_or_raise(
        missing_parent,
        PARENT_RANGE,
        revision_overlap=1,
        failed_symbols=("MSFT",),
        records=(),
    )
    assert missing.failed_symbols == ("MSFT",)
    assert missing.failed_without_parent_coverage == ("MSFT",)
    assert "MSFT" not in missing.retained_parent_coverage_symbols
    assert not any(row.symbol == "MSFT" for row in missing.new_rows)
    assert len({row.session_key for row in missing.accepted_rows}) == len(
        missing.accepted_rows
    )


@pytest.mark.integration
def test_relocation_corruption_reconciliation_and_mutation_guards(
    tmp_path: Path,
) -> None:
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    prior = _publication_fixture("relocation-prior")
    fixture = _publication_fixture("relocation")
    store.publish_snapshot(prior.candidate(), operation_id="relocation-prior")
    store.publish_snapshot(fixture.candidate(), operation_id="relocation")

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copytree(store.objects_root, relocated / "objects")
    shutil.copytree(store.snapshots_root, relocated / "snapshots")
    relocated_manager = SnapshotManager(root=relocated)
    opened = relocated_manager.open_verified(fixture.manifest.snapshot_id)
    assert isinstance(opened, Ok)
    assert opened.value.snapshot_id == fixture.manifest.snapshot_id
    assert isinstance(relocated_manager.open_verified(prior.manifest.snapshot_id), Ok)
    listed = relocated_manager.list_snapshots()
    assert {
        item.snapshot_id for item in listed.items
    } == {prior.manifest.snapshot_id, fixture.manifest.snapshot_id}

    reference = fixture.manifest.content_identity.objects[0]
    store._cas_path(reference.relative_uri).write_bytes(b"corrupt immutable bytes")  # type: ignore[attr-defined]
    manager = SnapshotManager(storage=store, metadata=metadata)
    corrupt_open = manager.open_verified(fixture.manifest.snapshot_id)
    assert isinstance(corrupt_open, Err)
    assert corrupt_open.errors[0].category is ErrorCategory.INTEGRITY_CHECKSUM
    assert isinstance(manager.open_verified(prior.manifest.snapshot_id), Ok)
    reconciliation = store.reconcile()
    assert reconciliation.ignored_publication_ids == (fixture.manifest.snapshot_id,)
    assert reconciliation.unavailable_snapshot_ids == (fixture.manifest.snapshot_id,)
    assert metadata.get_snapshot(fixture.manifest.snapshot_id).availability is SnapshotAvailability.UNAVAILABLE
    assert metadata.get_snapshot(prior.manifest.snapshot_id).availability is SnapshotAvailability.AVAILABLE

    mutation_results = (
        manager.reject_mutation(fixture.manifest.snapshot_id),
        manager.publish(fixture.candidate()),
        manager.replace_manifest(fixture.manifest.snapshot_id, b"replacement"),
        manager.replace_object(fixture.manifest.snapshot_id, reference.relative_uri, b"replacement"),
        manager.update_snapshot(fixture.manifest.snapshot_id, availability="available"),
        manager.delete_snapshot(fixture.manifest.snapshot_id),
    )
    assert all(isinstance(result, Err) for result in mutation_results)
    assert all(
        result.errors[0].category is ErrorCategory.STORAGE_ATOMICITY
        for result in mutation_results
    )
    metadata.close()
