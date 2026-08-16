"""Staged, failure-isolated market-data ingestion orchestration.

The service in this module is deliberately an application boundary.  It knows
about domain records and the narrow ports used by provider, normalization,
validation, storage, snapshot, and job adapters, but it does not import a
provider SDK, Parquet implementation, database driver, or UI state.

A single ingestion attempt follows this order:

1. resolve the ordered universe plus SPY and, for an incremental request, the
   contiguous revision-overlap suffix;
2. fetch bounded provider batches with per-symbol retry isolation;
3. normalize and validate the complete logical candidate stream;
4. write raw, accepted, rejected, gap, and report candidates below staging;
5. assemble a content-derived manifest and publish it through the injected
   snapshot port; and
6. terminalize the durable job and ingestion-operation records.

Expected provider/data-quality failures are values at the public boundary.
Unexpected adapter failures are converted to sanitized :class:`ActionableError`
values and never escape the service.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast
from uuid import UUID, uuid4

from ..config.models import ResolvedConfig, RetryPolicyConfig
from ..config.serializer import ConfigurationSerializer, Redactor
from ..domain.canonical import canonical_json, sha256_bytes, sha256_canonical_json
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.execution import JobOperation, JobStage, JobState, ProgressUpdate
from ..domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotManifest,
)
from ..domain.market import (
    DailyBarCandidate,
    DataGap,
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    ProviderRequestMetadata,
    QuarantineRecord,
    SymbolOutcomeStatus,
    ValidationReport,
    normalize_symbol,
)
from ..domain.normalization import (
    CausalForwardAdjustmentV1,
    CorporateActionPolicy,
    Normalizer,
)
from ..domain.validation import ValidationOutput, ValidationService
from .incremental import IncrementalMerger, IncrementalParent
from .jobs import JobMetadataRepository, SynchronousJobManager
from .ports import MarketDataProvider, RetryClock, RetryPolicy, fetch_with_retry
from .snapshots import SnapshotManifestAssembler

ProgressCallback: TypeAlias = Callable[[ProgressUpdate], None]


class IngestionClock(Protocol):
    """Clock port used for operational timestamps and completed sessions."""

    def utc_now(self) -> datetime:
        """Return an aware UTC timestamp."""


class ExchangeCalendarPort(Protocol):
    """Calendar surface required by staged ingestion."""

    name: str
    version: str

    def is_session(self, value: date) -> bool:
        """Return whether a date is an official exchange session."""


class SnapshotPublisherPort(Protocol):
    """Write-side snapshot port used by the orchestration."""

    def publish_snapshot(self, candidate: object, **kwargs: object) -> object:
        """Publish or reuse one complete candidate snapshot."""


class ParentLoader(Protocol):
    """Load logical parent rows after a verified snapshot handle is opened."""

    def __call__(self, handle: object, manifest: SnapshotManifest) -> object:
        """Return an IncrementalParent or a mapping of parent logical rows."""


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """Application request for a full or revision-overlap ingestion.

    ``parent_snapshot`` is an optional in-process parent DTO used by local
    fixtures.  Production callers normally supply ``parent_snapshot_id`` and
    let the injected verifier/loader resolve it from immutable storage.
    """

    parent_snapshot_id: str | None = None
    parent_snapshot: object | None = None
    requested_range: DateRange | None = None

    def __post_init__(self) -> None:
        if self.parent_snapshot_id is not None:
            if (
                not isinstance(self.parent_snapshot_id, str)
                or not self.parent_snapshot_id.strip()
            ):
                raise ValueError(
                    "parent_snapshot_id must be a non-blank string or None"
                )
            object.__setattr__(
                self, "parent_snapshot_id", self.parent_snapshot_id.strip()
            )
        if self.requested_range is not None and not isinstance(
            self.requested_range, DateRange
        ):
            raise TypeError("requested_range must be a DateRange or None")
        if self.parent_snapshot is not None and isinstance(self.parent_snapshot, str):
            raise TypeError(
                "parent_snapshot must be a parent DTO, not a snapshot ID string"
            )
        if (
            self.parent_snapshot_id is not None
            and isinstance(self.parent_snapshot, SnapshotManifest)
            and self.parent_snapshot.snapshot_id != self.parent_snapshot_id
        ):
            raise ValueError("parent_snapshot_id does not match parent_snapshot")


# Descriptive aliases keep callers that use the glossary terminology working.
DataIngestionRequest = IngestionRequest


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Terminal scientific and operational projection of one ingestion attempt."""

    status: JobState | str
    snapshot_id: str
    requested_range: DateRange
    provider_batches: tuple[ProviderBatchResult, ...] = ()
    provider_records: tuple[ProviderRecord, ...] = ()
    accepted_rows: tuple[Any, ...] = ()
    quarantined_rows: tuple[QuarantineRecord, ...] = ()
    gaps: tuple[DataGap, ...] = ()
    validation: ValidationOutput | None = None
    manifest: SnapshotManifest | None = None
    publication: object | None = None
    failed_symbols: tuple[str, ...] = ()
    retained_parent_coverage_symbols: tuple[str, ...] = ()
    errors: tuple[ActionableError, ...] = ()
    limitation_disclosure: LimitationDisclosure = LimitationDisclosure.current()
    job_id: UUID | None = None
    correlation_id: str | None = None
    operation_id: UUID | None = None
    snapshot_reused: bool = False
    provider_requests: tuple[ProviderRequestMetadata, ...] = ()
    raw_objects: tuple[object, ...] = ()
    normalized_objects: tuple[object, ...] = ()
    quarantine_objects: tuple[object, ...] = ()
    gap_objects: tuple[object, ...] = ()
    validation_object: object | None = None

    def __post_init__(self) -> None:
        state = JobState(self.status)
        if state not in {
            JobState.SUCCEEDED,
            JobState.PARTIALLY_SUCCEEDED,
        }:
            raise ValueError("IngestionResult must represent a successful terminal job")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-blank string")
        if not isinstance(self.requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange")
        if self.validation is not None and not isinstance(
            self.validation, ValidationOutput
        ):
            raise TypeError("validation must be a ValidationOutput or None")
        if self.manifest is not None and not isinstance(
            self.manifest, SnapshotManifest
        ):
            raise TypeError("manifest must be a SnapshotManifest or None")
        for name in (
            "provider_batches",
            "provider_records",
            "accepted_rows",
            "quarantined_rows",
            "gaps",
            "failed_symbols",
            "retained_parent_coverage_symbols",
            "errors",
            "provider_requests",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        if any(
            not isinstance(item, ProviderBatchResult) for item in self.provider_batches
        ):
            raise TypeError("provider_batches must contain ProviderBatchResult values")
        if any(not isinstance(item, ProviderRecord) for item in self.provider_records):
            raise TypeError("provider_records must contain ProviderRecord values")
        if any(not isinstance(item, DailyBarCandidate) for item in self.accepted_rows):
            raise TypeError("accepted_rows must contain DailyBarCandidate values")
        if any(
            not isinstance(item, QuarantineRecord) for item in self.quarantined_rows
        ):
            raise TypeError("quarantined_rows must contain QuarantineRecord values")
        if any(not isinstance(item, DataGap) for item in self.gaps):
            raise TypeError("gaps must contain DataGap values")
        if any(not isinstance(item, ActionableError) for item in self.errors):
            raise TypeError("errors must contain ActionableError values")
        if any(
            not isinstance(item, ProviderRequestMetadata)
            for item in self.provider_requests
        ):
            raise TypeError(
                "provider_requests must contain ProviderRequestMetadata values"
            )
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be a LimitationDisclosure")
        object.__setattr__(self, "status", state)
        object.__setattr__(self, "provider_batches", tuple(self.provider_batches))
        object.__setattr__(
            self,
            "provider_records",
            tuple(sorted(self.provider_records, key=ProviderRecord.sort_key)),
        )
        object.__setattr__(
            self,
            "accepted_rows",
            tuple(sorted(self.accepted_rows, key=lambda value: value.sort_key())),
        )
        object.__setattr__(
            self,
            "quarantined_rows",
            tuple(sorted(self.quarantined_rows, key=QuarantineRecord.sort_key)),
        )
        object.__setattr__(self, "gaps", tuple(sorted(self.gaps, key=DataGap.sort_key)))
        object.__setattr__(
            self,
            "failed_symbols",
            tuple(sorted({normalize_symbol(value) for value in self.failed_symbols})),
        )
        object.__setattr__(
            self,
            "retained_parent_coverage_symbols",
            tuple(
                sorted(
                    {
                        normalize_symbol(value)
                        for value in self.retained_parent_coverage_symbols
                    }
                )
            ),
        )
        object.__setattr__(
            self, "errors", tuple(sorted(self.errors, key=ActionableError.sort_key))
        )
        object.__setattr__(self, "provider_requests", tuple(self.provider_requests))

    @property
    def job_state(self) -> JobState:
        """Return the terminal job state using the common job vocabulary."""

        return cast(JobState, self.status)

    @property
    def snapshot_id_or_none(self) -> str | None:
        """Compatibility projection for presenters that handle failed results."""

        return self.snapshot_id

    @property
    def accepted(self) -> tuple[Any, ...]:
        return self.accepted_rows

    @property
    def quarantined(self) -> tuple[QuarantineRecord, ...]:
        return self.quarantined_rows

    @property
    def data_gaps(self) -> tuple[DataGap, ...]:
        return self.gaps


DataIngestionResult = IngestionResult


@dataclass(frozen=True, slots=True)
class _IngestionFacts:
    """Internal normalized facts shared by full and incremental paths."""

    provider_batches: tuple[ProviderBatchResult, ...]
    provider_records: tuple[ProviderRecord, ...]
    validation: ValidationOutput
    failed_symbols: tuple[str, ...]
    retained_parent_coverage_symbols: tuple[str, ...]
    limitation_disclosure: LimitationDisclosure
    errors: tuple[ActionableError, ...]
    accepted_rows: tuple[Any, ...]
    quarantined_rows: tuple[QuarantineRecord, ...]
    gaps: tuple[DataGap, ...]
    manifest_from_merge: SnapshotManifest | None = None
    merge_status: str | None = None
    merge_publication: object | None = None
    merge_reused_objects: tuple[ContentAddressedObjectRef, ...] = ()


@dataclass(frozen=True, slots=True)
class _MemoryPublication:
    """Small fallback publication used only when no write port is injected."""

    manifest: SnapshotManifest
    reused: bool = False

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id


class _MemoryPublisher:
    """Process-local fallback that still preserves append-only scientific IDs."""

    def __init__(self) -> None:
        self._manifests: dict[str, SnapshotManifest] = {}

    def publish_snapshot(
        self, manifest: SnapshotManifest, **_: object
    ) -> _MemoryPublication:
        existing = self._manifests.get(manifest.snapshot_id)
        if existing is not None:
            if (
                existing.to_content_identity_dict()
                != manifest.to_content_identity_dict()
            ):
                raise ValueError(
                    "existing snapshot ID contains different scientific content"
                )
            return _MemoryPublication(existing, reused=True)
        self._manifests[manifest.snapshot_id] = manifest
        return _MemoryPublication(manifest, reused=False)


class _InlineJob:
    """Minimal job controller for isolated tests without a durable job port."""

    def __init__(
        self, total_units: int | None, callback: ProgressCallback | None
    ) -> None:
        self.job_id = uuid4()
        self.correlation_id = str(self.job_id)
        self.state = JobState.NOT_STARTED
        self._stage = JobStage.NOT_STARTED
        self._completed = 0
        self._total = total_units
        self._callback = callback
        self._warnings: list[str] = []

    def start(self, *, stage: JobStage | str = JobStage.PREPARING) -> ProgressUpdate:
        self.state = JobState.RUNNING
        self._stage = JobStage(stage)
        return self._emit()

    def report(
        self,
        *,
        stage: JobStage | str,
        completed_units: int,
        total_units: int | None = None,
        warnings: Iterable[str] = (),
        context: Mapping[str, object] | None = None,
    ) -> ProgressUpdate:
        del context
        if total_units is not None:
            if total_units < completed_units:
                raise ValueError("total_units must not be below completed_units")
            if self._total is not None and total_units != self._total:
                raise ValueError("total_units cannot change after it is established")
            self._total = total_units
        if completed_units < self._completed:
            raise ValueError("completed_units must not decrease")
        if self._total is not None and completed_units > self._total:
            raise ValueError("completed_units must not exceed total_units")
        self._completed = completed_units
        self._stage = JobStage(stage)
        self._warnings.extend(str(item) for item in warnings)
        return self._emit()

    def complete(self, *, partially_succeeded: bool = False) -> ProgressUpdate:
        self.state = (
            JobState.PARTIALLY_SUCCEEDED if partially_succeeded else JobState.SUCCEEDED
        )
        self._stage = JobStage.COMPLETED
        return self._emit()

    def fail(self, error: ActionableError) -> ProgressUpdate:
        self.state = JobState.FAILED
        self._stage = JobStage.FAILED
        return self._emit()

    def _emit(self) -> ProgressUpdate:
        update = ProgressUpdate(
            job_id=self.job_id,
            operation=JobOperation.INGESTION,
            state=self.state,
            stage=self._stage,
            completed_units=self._completed,
            total_units=self._total,
            elapsed_seconds=Decimal("0"),
            warnings=tuple(self._warnings),
        )
        if self._callback is not None:
            self._callback(update)
        return update


class DataIngestionService:
    """Coordinate one staged full or incremental market-data ingestion."""

    def __init__(
        self,
        provider: MarketDataProvider,
        calendar: ExchangeCalendarPort,
        normalizer: Normalizer | None = None,
        validator: object | None = None,
        parquet_store: object | None = None,
        snapshot_publisher: SnapshotPublisherPort | None = None,
        metadata: JobMetadataRepository | object | None = None,
        job_manager: SynchronousJobManager | object | None = None,
        *,
        policy: CorporateActionPolicy | None = None,
        incremental_merger: object | None = None,
        snapshot_manager: object | None = None,
        parent_loader: ParentLoader | None = None,
        clock: IngestionClock | None = None,
        sleep: RetryClock | None = None,
        redactor: Redactor | None = None,
        manifest_assembler: object = SnapshotManifestAssembler,
        **compatibility: object,
    ) -> None:
        if provider is None or calendar is None:
            raise TypeError("provider and calendar are required")
        self.provider = provider
        self.calendar = calendar
        self.normalizer = normalizer or Normalizer(policy)
        self.policy = (
            policy
            or getattr(self.normalizer, "policy", None)
            or CausalForwardAdjustmentV1()
        )
        self.validator = validator or ValidationService(
            calendar=calendar, benchmark_symbol="SPY"
        )
        self.parquet_store = (
            parquet_store
            or compatibility.pop("writer", None)
            or compatibility.pop("parquet_writer", None)
        )
        self.snapshot_publisher = (
            snapshot_publisher
            or compatibility.pop("publisher", None)
            or compatibility.pop("snapshot_store", None)
            or compatibility.pop("filesystem_store", None)
            or _MemoryPublisher()
        )
        self.metadata = metadata or compatibility.pop("metadata_store", None)
        self.job_manager = job_manager or compatibility.pop("jobs", None)
        self.snapshot_manager = snapshot_manager or compatibility.pop(
            "snapshot_verifier", None
        )
        self.parent_loader = parent_loader or cast(
            ParentLoader | None, compatibility.pop("load_parent", None)
        )
        self.clock = (
            clock
            or cast(IngestionClock, compatibility.pop("snapshot_clock", None))
            or _SystemIngestionClock()
        )
        configured_sleep = compatibility.pop("retry_sleep", None)
        self.sleep = (
            sleep
            if sleep is not None
            else (
                cast(RetryClock, configured_sleep)
                if configured_sleep is not None
                else lambda _seconds: None
            )
        )
        self.redactor = redactor or cast(
            Redactor | None, compatibility.pop("redactor", None)
        )
        self.manifest_assembler = manifest_assembler
        self.incremental_merger = incremental_merger
        if compatibility:
            unknown = ", ".join(sorted(compatibility))
            raise TypeError(f"unsupported DataIngestionService arguments: {unknown}")

    def ingest(
        self,
        request: IngestionRequest | None = None,
        config: ResolvedConfig | object | None = None,
        *,
        progress: ProgressCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Result[IngestionResult]:
        """Execute one staged ingestion and return a typed terminal result.

        The method accepts the design's ``ingest(request, config)`` shape and a
        convenient ``ingest(config)`` form for local callers.  Both forms use
        the same deterministic orchestration and injected ports.
        """

        if (
            config is None
            and request is not None
            and not isinstance(request, IngestionRequest)
        ):
            config = request
            request = None
        resolved_request = request or IngestionRequest()
        if config is None:
            return Err(
                (self._input_error("config is required", field_path="config"),),
                preserve_order=True,
            )
        callback = progress_callback or progress
        redactor = self._redactor_for(config)
        try:
            return self._ingest(resolved_request, config, callback, redactor)
        except _IngestionFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except Exception as error:
            return Err(
                (ActionableError.from_unexpected_exception("ingestion.execute", error),)
            )

    # Common application-service spellings.
    run = ingest
    execute = ingest
    ingest_data = ingest
    ingest_snapshot = ingest

    def _ingest(
        self,
        request: IngestionRequest,
        config: object,
        callback: ProgressCallback | None,
        redactor: Redactor,
    ) -> Result[IngestionResult]:
        universe = self._universe(config)
        benchmark = self._benchmark(config)
        symbols = self._ordered_symbols(universe, benchmark)
        requested_range = request.requested_range or self._requested_range(config)
        batch_size = self._integer_config(config, "data.batch_size", 5)
        staleness = self._integer_config(config, "data.staleness_sessions", 1)
        overlap = self._integer_config(config, "data.revision_overlap_sessions", 5)
        write_chunk_size = self._integer_config(config, "data.write_chunk_rows", 50_000)
        policy = self.policy
        configuration_checksum = self._configuration_checksum(config, redactor)
        provider_name = str(
            getattr(
                self.provider,
                "name",
                self._text_config(config, "data.provider", "yfinance"),
            )
        )

        parent_id = request.parent_snapshot_id or self._parent_id(
            request.parent_snapshot
        )
        parent = request.parent_snapshot
        merger = self._incremental_merger(benchmark)
        fetch_range: DateRange | None = requested_range
        if parent_id is not None and merger is None:
            raise _IngestionFailure((self._parent_error(parent_id),))
        if parent_id is not None and merger is not None:
            parent = self._resolve_parent(parent, parent_id)
            if parent is not None:
                try:
                    plan = self._call_plan(merger, parent, requested_range, overlap)
                    suffix = getattr(plan, "suffix_range", None)
                    fetch_range = suffix if isinstance(suffix, DateRange) else None
                except Exception as error:
                    raise _IngestionFailure(
                        (
                            self._input_error(
                                str(error), field_path="incremental.requested_range"
                            ),
                        )
                    ) from None
            elif fetch_range == requested_range:
                # A merger with its own verifier can resolve the ID at merge
                # time; the full request is the safe fallback when no parent
                # row loader is available to compute a suffix locally.
                fetch_range = requested_range

        job = cast(Any, self._create_job(len(symbols), callback))
        operation_id = uuid4()
        operation_created = False
        try:
            job.start(stage=JobStage.PREPARING)
            self._create_operation(
                operation_id,
                job,
                requested_range,
                parent_id,
                mode="incremental" if parent_id is not None else "full",
            )
            operation_created = True
            staging = self._create_staging(operation_id)
            batches, records, failed = self._fetch_batches(
                job,
                symbols,
                fetch_range,
                batch_size=batch_size,
                config=config,
                redactor=redactor,
            )
            facts = self._build_facts(
                request=request,
                parent=parent,
                parent_id=parent_id,
                merger=merger,
                requested_range=requested_range,
                fetch_range=fetch_range,
                symbols=symbols,
                overlap=overlap,
                batches=batches,
                provider_records=records,
                failed_symbols=failed,
                staleness_threshold=staleness,
                benchmark=benchmark,
                policy=policy,
                redactor=redactor,
                job=job,
            )
            if not facts.accepted_rows:
                raise _IngestionFailure(
                    facts.errors
                    or (
                        ActionableError(
                            operation="ingestion.validate",
                            category=ErrorCategory.SNAPSHOT_NOT_READY,
                            message=(
                                "No valid accepted market-data candidate can be "
                                "published."
                            ),
                            corrective_action=(
                                "Resolve provider, normalization, or validation "
                                "failures and retry the ingestion."
                            ),
                        ),
                    )
                )

            job.report(
                stage=JobStage.PUBLISHING,
                completed_units=len(symbols),
                total_units=len(symbols),
                warnings=tuple(error.format_for_display() for error in facts.errors),
                context={"accepted_rows": len(facts.accepted_rows)},
            )
            publication, manifest, objects, validation_source = self._stage_and_publish(
                facts=facts,
                requested_range=requested_range,
                symbols=universe,
                benchmark=benchmark,
                provider_name=provider_name,
                configuration_checksum=configuration_checksum,
                policy=policy,
                write_chunk_size=write_chunk_size,
                staging=staging,
                parent_id=parent_id,
                operation_id=operation_id,
                redactor=redactor,
            )
            status = self._terminal_status(facts, manifest, publication)
            partial = status is JobState.PARTIALLY_SUCCEEDED
            terminal = job.complete(partially_succeeded=partial)
            if operation_created:
                self._complete_operation(
                    operation_id, status.value, manifest.snapshot_id
                )
            self._record_objects(objects, manifest, validation_source)
            result = IngestionResult(
                status=terminal.state,
                snapshot_id=manifest.snapshot_id,
                requested_range=requested_range,
                provider_batches=facts.provider_batches,
                provider_records=facts.provider_records,
                accepted_rows=facts.accepted_rows,
                quarantined_rows=facts.quarantined_rows,
                gaps=facts.gaps,
                validation=facts.validation,
                manifest=manifest,
                publication=publication,
                failed_symbols=facts.failed_symbols,
                retained_parent_coverage_symbols=facts.retained_parent_coverage_symbols,
                errors=facts.errors,
                limitation_disclosure=facts.limitation_disclosure,
                job_id=getattr(job, "job_id", None),
                correlation_id=getattr(job, "correlation_id", None),
                operation_id=operation_id,
                snapshot_reused=bool(getattr(publication, "reused", False)),
                provider_requests=tuple(
                    batch.operational_metadata
                    for batch in facts.provider_batches
                    if batch.operational_metadata is not None
                ),
                raw_objects=tuple(objects.get("raw", ())),
                normalized_objects=tuple(objects.get("normalized", ())),
                quarantine_objects=tuple(objects.get("quarantine", ())),
                gap_objects=tuple(objects.get("gap", ())),
                validation_object=validation_source,
            )
            return Ok(result)
        except _IngestionFailure as failure:
            self._terminal_failure(job, failure.errors)
            if operation_created:
                self._complete_operation(operation_id, "failed", None)
            raise
        except Exception as error:
            actionable = ActionableError.from_unexpected_exception(
                "ingestion.execute",
                error,
                correlation_id=getattr(job, "correlation_id", None),
            )
            self._terminal_failure(job, (actionable,))
            if operation_created:
                self._complete_operation(operation_id, "failed", None)
            raise _IngestionFailure((actionable,)) from None

    def _fetch_batches(
        self,
        job: object,
        symbols: tuple[str, ...],
        requested_range: DateRange | None,
        *,
        batch_size: int,
        config: object,
        redactor: Redactor,
    ) -> tuple[
        tuple[ProviderBatchResult, ...], tuple[ProviderRecord, ...], tuple[str, ...]
    ]:
        if requested_range is None:
            # An unchanged incremental request has no provider suffix.  It is
            # not a provider failure: the merger will reuse the verified
            # parent's content and validation facts.
            self._job_report(job, JobStage.FETCHING, len(symbols), len(symbols))
            return (), (), ()
        policy = self._retry_policy(config)
        batches: list[ProviderBatchResult] = []
        records: list[ProviderRecord] = []
        failures: list[str] = []
        for start in range(0, len(symbols), batch_size):
            batch_symbols = symbols[start : start + batch_size]
            request = self._provider_request(batch_symbols, requested_range)
            self._job_report(
                job,
                JobStage.FETCHING,
                start,
                len(symbols),
                context={"symbols": batch_symbols},
            )
            started = self.clock.utc_now()
            try:
                result = fetch_with_retry(
                    self.provider,
                    request,
                    policy,
                    sleep=self.sleep,
                )
            except Exception as error:
                raise _IngestionFailure(
                    (self._provider_exception_error(error, batch_symbols),)
                ) from None
            ended = self.clock.utc_now()
            result = self._attach_request_metadata(result, started, ended)
            batches.append(result)
            for outcome in result.outcomes:
                if outcome.status is SymbolOutcomeStatus.SUCCESS:
                    records.extend(outcome.records)
                else:
                    failures.append(outcome.symbol)
            self._record_provider_batch(job, result, ended)
            warnings = tuple(
                redactor.redact_text(error.format_for_display())
                for outcome in result.outcomes
                for error in outcome.errors
            )
            self._job_report(
                job,
                JobStage.FETCHING,
                start + len(batch_symbols),
                len(symbols),
                warnings=warnings,
                context={"symbols": batch_symbols, "response_status": result.status},
            )
        unique_records = {record.provider_record_checksum: record for record in records}
        return (
            tuple(batches),
            tuple(sorted(unique_records.values(), key=ProviderRecord.sort_key)),
            tuple(sorted(set(failures))),
        )

    def _build_facts(
        self,
        *,
        request: IngestionRequest,
        parent: object | None,
        parent_id: str | None,
        merger: object | None,
        requested_range: DateRange,
        fetch_range: DateRange | None,
        symbols: tuple[str, ...],
        overlap: int,
        batches: tuple[ProviderBatchResult, ...],
        provider_records: tuple[ProviderRecord, ...],
        failed_symbols: tuple[str, ...],
        staleness_threshold: int,
        benchmark: str,
        policy: CorporateActionPolicy,
        redactor: Redactor,
        job: object,
    ) -> _IngestionFacts:
        if parent_id is not None and merger is not None:
            self._job_report(job, JobStage.NORMALIZING, len(symbols), len(symbols))
            try:
                merge = self._call_merge(
                    merger,
                    parent,
                    parent_id,
                    requested_range,
                    overlap,
                    provider_records,
                    batches,
                    failed_symbols,
                    staleness_threshold,
                )
            except _IngestionFailure:
                raise
            except Exception as error:
                raise _IngestionFailure(
                    (self._unexpected_error("incremental.merge", error),)
                ) from None
            merge_value = self._unwrap_result(merge, "incremental.merge")
            merge_value_any = cast(Any, merge_value)
            validation = cast(ValidationOutput, merge_value_any.validation)
            accepted = tuple(
                getattr(merge_value, "accepted_rows", validation.accepted_rows)
            )
            quarantined = tuple(
                getattr(merge_value, "quarantined_rows", validation.quarantined_rows)
            )
            gaps = tuple(getattr(merge_value, "gaps", validation.gaps))
            merge_failed = tuple(getattr(merge_value, "failed_symbols", failed_symbols))
            retained = tuple(
                getattr(merge_value, "retained_parent_coverage_symbols", ())
            )
            merge_errors = tuple(getattr(merge_value, "failure_errors", ()))
            errors = self._quality_errors(
                batches,
                failed_symbols=merge_failed,
                quarantined=quarantined,
                gaps=gaps,
                validation=validation,
                extra=merge_errors,
                redactor=redactor,
            )
            disclosure = LimitationDisclosure.current(data_failures=errors)
            return _IngestionFacts(
                provider_batches=batches,
                provider_records=tuple(
                    getattr(merge_value, "provider_records", provider_records)
                ),
                validation=validation,
                failed_symbols=tuple(merge_failed),
                retained_parent_coverage_symbols=retained,
                limitation_disclosure=disclosure,
                errors=errors,
                accepted_rows=accepted,
                quarantined_rows=quarantined,
                gaps=gaps,
                manifest_from_merge=getattr(merge_value, "manifest", None),
                merge_status=getattr(merge_value, "status", None),
                merge_publication=getattr(merge_value, "publication", None),
                merge_reused_objects=tuple(
                    getattr(merge_value, "reused_object_references", ())
                ),
            )

        self._job_report(job, JobStage.NORMALIZING, len(symbols), len(symbols))
        try:
            candidates = tuple(self._normalize(provider_records, policy))
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("ingestion.normalize", error),)
            ) from None
        self._job_report(job, JobStage.VALIDATING, len(symbols), len(symbols))
        expected = self._expected_sessions(requested_range, symbols)
        try:
            validation = self._validate(
                candidates,
                expected,
                staleness_threshold,
                requested_range=requested_range,
                benchmark=benchmark,
                failed_symbols=failed_symbols,
            )
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("ingestion.validate", error),)
            ) from None
        accepted = validation.accepted_rows
        quarantined = validation.quarantined_rows
        gaps = validation.gaps
        errors = self._quality_errors(
            batches,
            failed_symbols=failed_symbols,
            quarantined=quarantined,
            gaps=gaps,
            validation=validation,
            redactor=redactor,
        )
        disclosure = LimitationDisclosure.current(data_failures=errors)
        return _IngestionFacts(
            provider_batches=batches,
            provider_records=provider_records,
            validation=validation,
            failed_symbols=failed_symbols,
            retained_parent_coverage_symbols=(),
            limitation_disclosure=disclosure,
            errors=errors,
            accepted_rows=accepted,
            quarantined_rows=quarantined,
            gaps=gaps,
        )

    def _stage_and_publish(
        self,
        *,
        facts: _IngestionFacts,
        requested_range: DateRange,
        symbols: tuple[str, ...],
        benchmark: str,
        provider_name: str,
        configuration_checksum: str,
        policy: CorporateActionPolicy,
        write_chunk_size: int,
        staging: object | None,
        parent_id: str | None,
        operation_id: UUID,
        redactor: Redactor,
    ) -> tuple[object, SnapshotManifest, dict[str, tuple[object, ...]], object]:
        staging_path = self._staging_path(staging)
        raw_sources = self._write_collection(
            "raw",
            facts.provider_records,
            write_chunk_size=write_chunk_size,
            staging=staging_path,
        )
        normalized_sources = self._write_collection(
            "normalized",
            facts.accepted_rows,
            write_chunk_size=write_chunk_size,
            staging=staging_path,
        )
        quarantine_sources = self._write_collection(
            "quarantine",
            facts.quarantined_rows,
            write_chunk_size=write_chunk_size,
            staging=staging_path,
        )
        gap_sources = self._write_collection(
            "gap",
            facts.gaps,
            write_chunk_size=write_chunk_size,
            staging=staging_path,
        )
        validation_sources = self._write_collection(
            "validation",
            (facts.validation.report,),
            write_chunk_size=write_chunk_size,
            staging=staging_path,
        )
        validation_source: object
        validation_checksum: str
        if validation_sources:
            validation_source = validation_sources[0]
            validation_checksum = (
                self._object_checksum(validation_source)
                or facts.validation.report.content_checksum
            )
        else:
            validation_source = canonical_json(
                facts.validation.report.to_content_dict()
            )
            validation_checksum = sha256_bytes(validation_source)
        references = self._object_references(
            (*raw_sources, *normalized_sources, *quarantine_sources, *gap_sources),
            extra=facts.merge_reused_objects,
        )
        calendar_identity = self._calendar_identity(requested_range)
        manifest = self._assemble_manifest(
            provider=provider_name,
            requested_range=requested_range,
            covered_range=facts.validation.report.summary.covered_range,
            configured_universe=symbols,
            benchmark_symbol=benchmark,
            calendar=calendar_identity,
            configuration_checksum=configuration_checksum,
            objects=references,
            validation=facts.validation,
            validation_report_checksum=validation_checksum,
            disclosure=facts.limitation_disclosure,
            provider_requests=tuple(
                batch.operational_metadata
                for batch in facts.provider_batches
                if batch.operational_metadata is not None
            ),
            parent_snapshot_id=parent_id,
            operation_id=str(operation_id),
            created_at=self.clock.utc_now(),
        )
        sources = tuple(
            (*raw_sources, *normalized_sources, *quarantine_sources, *gap_sources)
        )
        publication = self._publish(
            manifest,
            sources=sources,
            validation_source=validation_source,
            statuses=facts.validation.report.per_symbol,
            staging=staging,
            operation_id=operation_id,
        )
        published_manifest = getattr(publication, "manifest", manifest)
        if isinstance(published_manifest, SnapshotManifest):
            manifest = published_manifest
        return (
            publication,
            manifest,
            {
                "raw": tuple(raw_sources),
                "normalized": tuple(normalized_sources),
                "quarantine": tuple(quarantine_sources),
                "gap": tuple(gap_sources),
            },
            validation_source,
        )

    def _create_job(
        self, total_units: int, callback: ProgressCallback | None
    ) -> object:
        manager = self.job_manager
        create = getattr(manager, "create", None) if manager is not None else None
        if callable(create):
            return create(
                JobOperation.INGESTION,
                total_units=total_units,
                progress_callback=callback,
            )
        return _InlineJob(total_units, callback)

    def _create_operation(
        self,
        operation_id: UUID,
        job: object,
        requested_range: DateRange,
        parent_id: str | None,
        *,
        mode: str,
    ) -> None:
        if self.metadata is None:
            return
        method = getattr(self.metadata, "create_ingestion_operation", None)
        if not callable(method):
            return
        try:
            self._invoke(
                method,
                operation_id=operation_id,
                job_id=cast(UUID, cast(Any, job).job_id),
                mode=mode,
                requested_start=requested_range.start,
                requested_end=requested_range.end,
                created_at=self.clock.utc_now(),
                parent_snapshot_id=parent_id,
            )
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("ingestion.operation", error),)
            ) from None

    def _complete_operation(
        self,
        operation_id: UUID,
        status: str,
        snapshot_id: str | None,
    ) -> None:
        if self.metadata is None:
            return
        method = getattr(self.metadata, "complete_ingestion_operation", None)
        if not callable(method):
            return
        try:
            self._invoke(
                method, operation_id, status=status, result_snapshot_id=snapshot_id
            )
        except Exception:
            # A terminal job is still more useful than replacing its primary
            # publication result with a second metadata cleanup error.
            return

    def _record_provider_batch(
        self,
        job: object,
        result: ProviderBatchResult,
        occurred_at: datetime,
    ) -> None:
        if self.metadata is None:
            return
        method = getattr(self.metadata, "record_provider_batch", None)
        if not callable(method):
            return
        try:
            self._invoke(
                method,
                job_id=cast(UUID, cast(Any, job).job_id),
                result=result,
                occurred_at=occurred_at,
            )
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("ingestion.provenance", error),)
            ) from None

    def _record_objects(
        self,
        objects: Mapping[str, Sequence[object]],
        manifest: SnapshotManifest,
        validation_source: object,
    ) -> None:
        if self.metadata is None:
            return
        record = getattr(self.metadata, "record_data_object", None)
        if not callable(record):
            return
        created_at = manifest.operational_metadata.created_at
        seen: set[str] = set()
        for reference in manifest.content_identity.objects:
            if reference.checksum in seen:
                continue
            seen.add(reference.checksum)
            try:
                self._invoke(record, reference, created_at=created_at)
            except Exception:
                return
        # The validation artifact is intentionally outside content_identity.objects
        # in the manifest, but metadata stores may expose a separate artifact port.
        validation_ref = self._as_object_ref(validation_source, ObjectKind.VALIDATION)
        artifact = getattr(self.metadata, "record_artifact", None)
        if validation_ref is not None and callable(artifact):
            try:
                self._invoke(
                    artifact,
                    validation_ref,
                    artifact_kind="validation",
                    created_at=created_at,
                )
            except Exception:
                return

    def _create_staging(self, operation_id: UUID) -> object | None:
        creator = getattr(self.snapshot_publisher, "create_staging", None)
        if not callable(creator):
            return None
        try:
            return cast(object, self._invoke(creator, str(operation_id)))
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("snapshot.stage", error),)
            ) from None

    def _publish(
        self,
        manifest: SnapshotManifest,
        *,
        sources: Sequence[object],
        validation_source: object,
        statuses: Sequence[object],
        staging: object | None,
        operation_id: UUID,
    ) -> object:
        publisher = self.snapshot_publisher
        method = getattr(publisher, "publish_snapshot", None) or getattr(
            publisher, "publish", None
        )
        if not callable(method):
            raise _IngestionFailure(
                (
                    self._unexpected_error(
                        "snapshot.publish", TypeError("publisher has no publish method")
                    ),
                )
            )
        kwargs: dict[str, object] = {
            "staged_objects": tuple(sources),
            "validation_report": validation_source,
            "symbol_statuses": tuple(statuses),
            "operation_id": str(operation_id),
        }
        if staging is not None:
            kwargs["staging"] = staging
        try:
            result = self._invoke(method, manifest, **kwargs)
            return self._unwrap_result(result, "snapshot.publish")
        except _IngestionFailure:
            raise
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("snapshot.publish", error),)
            ) from None

    def _assemble_manifest(self, **kwargs: object) -> SnapshotManifest:
        assembler = self.manifest_assembler
        method = getattr(assembler, "assemble", None) or assembler
        if not callable(method):
            raise _IngestionFailure(
                (
                    self._unexpected_error(
                        "snapshot.assemble",
                        TypeError("manifest assembler is not callable"),
                    ),
                )
            )
        try:
            value = self._invoke(method, **kwargs)
            value = self._unwrap_result(value, "snapshot.assemble")
            if not isinstance(value, SnapshotManifest):
                raise TypeError("manifest assembler did not return SnapshotManifest")
            return value
        except _IngestionFailure:
            raise
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error("snapshot.assemble", error),)
            ) from None

    def _normalize(
        self,
        records: Iterable[ProviderRecord],
        policy: CorporateActionPolicy,
    ) -> tuple[object, ...]:
        method = self.normalizer.normalize
        kwargs = {"policy": policy}
        return tuple(self._invoke(method, records, self.calendar, **kwargs))

    def _validate(
        self,
        candidates: Iterable[object],
        expected: Mapping[str, Sequence[date]],
        staleness: int,
        *,
        requested_range: DateRange,
        benchmark: str,
        failed_symbols: Sequence[str],
    ) -> ValidationOutput:
        method = cast(Any, self.validator).validate
        value = self._invoke(
            method,
            candidates,
            expected,
            staleness,
            requested_range=requested_range,
            benchmark_symbol=benchmark,
            failed_symbols=tuple(failed_symbols),
            calendar=self.calendar,
        )
        if not isinstance(value, ValidationOutput):
            raise TypeError("validator did not return ValidationOutput")
        return value

    def _call_plan(
        self,
        merger: object,
        parent: object,
        requested_range: DateRange,
        overlap: int,
    ) -> object:
        method = cast(Any, merger).plan
        return self._invoke(method, parent, requested_range, overlap)

    def _call_merge(
        self,
        merger: object,
        parent: object | None,
        parent_id: str | None,
        requested_range: DateRange,
        overlap: int,
        records: tuple[ProviderRecord, ...],
        batches: tuple[ProviderBatchResult, ...],
        failed_symbols: tuple[str, ...],
        staleness_threshold: int,
    ) -> object:
        method = cast(Any, merger).merge
        parent_arg: object = parent if parent is not None else parent_id
        kwargs: dict[str, object] = {
            "provider_outcomes": batches,
            "failed_symbols": failed_symbols,
            "staleness_threshold": staleness_threshold,
        }
        # IncrementalMerger accepts exactly one record spelling.  Prefer the
        # production ``records`` keyword, while allowing a narrow fake merger
        # that exposes only the compatibility ``provider_records`` spelling.
        try:
            parameters: Mapping[str, inspect.Parameter] = inspect.signature(
                method
            ).parameters
        except (TypeError, ValueError):
            parameters = {}
        has_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "records" in parameters or has_var_kwargs or not parameters:
            kwargs["records"] = records
        elif "provider_records" in parameters:
            kwargs["provider_records"] = records
        return self._invoke(method, parent_arg, requested_range, overlap, **kwargs)

    def _resolve_parent(self, parent: object | None, parent_id: str) -> object | None:
        if parent is not None:
            return parent
        if self.snapshot_manager is None or self.parent_loader is None:
            return None
        opener = getattr(self.snapshot_manager, "open_verified", None)
        if not callable(opener):
            return None
        opened = self._unwrap_result(
            self._invoke(opener, parent_id), "incremental.verify_parent"
        )
        manifest = getattr(opened, "manifest", None)
        if not isinstance(manifest, SnapshotManifest):
            inspector = getattr(self.snapshot_manager, "inspect_snapshot", None)
            if callable(inspector):
                inspected = self._unwrap_result(
                    self._invoke(inspector, parent_id), "incremental.inspect_parent"
                )
                manifest = getattr(inspected, "manifest", None)
        if not isinstance(manifest, SnapshotManifest):
            raise _IngestionFailure((self._parent_error(parent_id),))
        loaded = self.parent_loader(opened, manifest)
        if isinstance(loaded, IncrementalParent):
            return loaded
        if isinstance(loaded, Mapping):
            return IncrementalParent.from_manifest(
                manifest,
                accepted_rows=cast(Iterable[Any], loaded.get("accepted_rows", ())),
                provider_records=cast(
                    Iterable[ProviderRecord], loaded.get("provider_records", ())
                ),
                quarantined_rows=cast(
                    Iterable[QuarantineRecord], loaded.get("quarantined_rows", ())
                ),
                expected_sessions=cast(
                    Mapping[str, Sequence[date]] | None, loaded.get("expected_sessions")
                ),
                validation_report=cast(
                    ValidationReport | None, loaded.get("validation_report")
                ),
            )
        raise _IngestionFailure((self._parent_error(parent_id),))

    def _incremental_merger(self, benchmark: str) -> object | None:
        if self.incremental_merger is not None:
            return self.incremental_merger
        if self.snapshot_manager is None:
            return None
        return IncrementalMerger(
            self.calendar,
            normalizer=self.normalizer,
            policy=self.policy,
            validator=self.validator
            if isinstance(self.validator, ValidationService)
            else None,
            snapshot_manager=cast(Any, self.snapshot_manager),
            parent_loader=cast(Any, self.parent_loader),
        )

    def _expected_sessions(
        self,
        requested_range: DateRange,
        symbols: Sequence[str],
    ) -> dict[str, tuple[date, ...]]:
        completed_at = self.clock.utc_now()
        sessions_method = getattr(self.calendar, "sessions", None)
        if callable(sessions_method):
            try:
                sessions = tuple(
                    self._invoke(
                        sessions_method,
                        requested_range.start,
                        requested_range.end,
                        completed_at=completed_at,
                    )
                )
            except TypeError:
                sessions = tuple(
                    self._invoke(
                        sessions_method, requested_range.start, requested_range.end
                    )
                )
        else:
            is_session = getattr(self.calendar, "is_session", None)
            if not callable(is_session):
                raise TypeError("calendar must expose sessions() or is_session()")
            sessions = tuple(
                value
                for ordinal in range(
                    (requested_range.end - requested_range.start).days + 1
                )
                if (
                    value := requested_range.start.fromordinal(
                        requested_range.start.toordinal() + ordinal
                    )
                )
                and bool(is_session(value))
            )
        valid_sessions = tuple(
            sorted(
                {
                    value
                    for value in sessions
                    if isinstance(value, date)
                    and not isinstance(value, datetime)
                    and requested_range.start <= value <= requested_range.end
                }
            )
        )
        return {normalize_symbol(symbol): valid_sessions for symbol in symbols}

    def _calendar_identity(self, requested_range: DateRange) -> CalendarIdentity:
        checksum_method = getattr(self.calendar, "schedule_checksum", None)
        if callable(checksum_method):
            try:
                checksum = str(
                    self._invoke(
                        checksum_method, requested_range.start, requested_range.end
                    )
                )
            except Exception:
                checksum = ""
        else:
            checksum = ""
        if len(checksum) != 64:
            existing = getattr(self.calendar, "identity", None) or getattr(
                self.calendar, "calendar_identity", None
            )
            if isinstance(existing, CalendarIdentity):
                checksum = existing.schedule_checksum
                name = existing.name
                version = existing.version
            else:
                sessions = self._expected_sessions(requested_range, ("SPY",))["SPY"]
                checksum = sha256_canonical_json(
                    [session.isoformat() for session in sessions]
                )
                name = str(getattr(self.calendar, "name", "XNYS"))
                version = str(getattr(self.calendar, "version", "unknown"))
        else:
            name = str(getattr(self.calendar, "name", "XNYS"))
            version = str(getattr(self.calendar, "version", "unknown"))
        return CalendarIdentity(
            name=name,
            version=version,
            schedule_checksum=checksum,
        )

    def _provider_request(
        self, symbols: Sequence[str], requested_range: DateRange
    ) -> ProviderRequest:
        return ProviderRequest(
            tuple(symbols),
            requested_range.start,
            requested_range.end,
            provider=str(getattr(self.provider, "name", "yfinance")),
        )

    def _attach_request_metadata(
        self,
        result: ProviderBatchResult,
        started_at: datetime,
        ended_at: datetime,
    ) -> ProviderBatchResult:
        metadata = result.operational_metadata
        if metadata is None:
            metadata = ProviderRequestMetadata(
                request_content_key=result.request.content_key,
                retrieval_started_at=started_at,
                retrieved_at=ended_at,
                response_status=result.status,
            )
            return replace(result, operational_metadata=metadata)
        return result

    def _write_collection(
        self,
        role: str,
        rows: Iterable[object],
        *,
        write_chunk_size: int,
        staging: Path | None,
    ) -> tuple[object, ...]:
        materialized = tuple(rows)
        if not materialized or self.parquet_store is None:
            return ()
        method_names = {
            "raw": ("write_raw", "write_raw_collection"),
            "normalized": ("write_normalized", "write_normalized_collection"),
            "quarantine": (
                "write_quarantine",
                "write_quarantines",
                "write_quarantine_collection",
            ),
            "gap": ("write_gaps", "write_gap", "write_gap_collection"),
            "validation": (
                "write_validation_report",
                "write_validation_reports",
                "write_validation",
            ),
        }[role]
        method = next(
            (
                getattr(self.parquet_store, name, None)
                for name in method_names
                if callable(getattr(self.parquet_store, name, None))
            ),
            None,
        )
        if not callable(method):
            if role == "validation":
                # Validation remains publishable through the existing canonical
                # JSON fallback for compatibility with older injected writers.
                return ()
            raise _IngestionFailure(
                (
                    self._unexpected_error(
                        f"ingestion.write.{role}",
                        TypeError(
                            f"parquet writer has no required {role} collection method"
                        ),
                    ),
                )
            )
        kwargs: dict[str, object] = {
            "write_chunk_size": write_chunk_size,
            "staging": staging,
        }
        try:
            value = self._invoke(method, materialized, **kwargs)
        except Exception as error:
            raise _IngestionFailure(
                (self._unexpected_error(f"ingestion.write.{role}", error),)
            ) from None
        return self._materialize_output(value)

    @staticmethod
    def _materialize_output(value: object) -> tuple[object, ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            return tuple(value.values())
        if isinstance(value, (str, bytes, bytearray, Path)):
            return (value,)
        try:
            return tuple(cast(Iterable[object], value))
        except TypeError:
            return (value,)

    @staticmethod
    def _object_checksum(value: object) -> str | None:
        checksum = getattr(value, "checksum", None)
        return checksum if isinstance(checksum, str) and len(checksum) == 64 else None

    def _object_references(
        self,
        values: Iterable[object],
        *,
        extra: Iterable[ContentAddressedObjectRef] = (),
    ) -> tuple[ContentAddressedObjectRef, ...]:
        by_uri: dict[str, ContentAddressedObjectRef] = {}
        by_checksum: dict[str, ContentAddressedObjectRef] = {}
        for value in (*tuple(values), *tuple(extra)):
            reference = self._as_object_ref(value)
            if reference is None or reference.object_kind is ObjectKind.VALIDATION:
                continue
            prior_uri = by_uri.get(reference.relative_uri)
            if prior_uri is not None and prior_uri != reference:
                raise _IngestionFailure(
                    (
                        self._unexpected_error(
                            "snapshot.assemble",
                            ValueError("logical object URI has conflicting checksums"),
                        ),
                    )
                )
            prior_checksum = by_checksum.get(reference.checksum)
            if prior_checksum is not None and prior_checksum != reference:
                raise _IngestionFailure(
                    (
                        self._unexpected_error(
                            "snapshot.assemble",
                            ValueError(
                                "content checksum is referenced by conflicting objects"
                            ),
                        ),
                    )
                )
            by_uri[reference.relative_uri] = reference
            by_checksum[reference.checksum] = reference
        return tuple(sorted(by_uri.values(), key=ContentAddressedObjectRef.sort_key))

    def _as_object_ref(
        self,
        value: object,
        kind_hint: ObjectKind | None = None,
    ) -> ContentAddressedObjectRef | None:
        if isinstance(value, ContentAddressedObjectRef):
            return value
        candidate = getattr(value, "object_ref", None)
        if isinstance(candidate, ContentAddressedObjectRef):
            return candidate
        if isinstance(value, Mapping):
            try:
                return ContentAddressedObjectRef(
                    object_kind=value.get(
                        "object_kind", kind_hint or ObjectKind.ARTIFACT
                    ),
                    checksum=cast(str, value["checksum"]),
                    relative_uri=cast(str, value["relative_uri"]),
                    schema_version=cast(str, value.get("schema_version", "unknown")),
                    row_count=int(value.get("row_count", 0)),
                    byte_size=int(value.get("byte_size", 0)),
                    symbol=cast(str | None, value.get("symbol")),
                    session_year=cast(int | None, value.get("session_year")),
                    media_type=str(value.get("media_type", "application/octet-stream")),
                )
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def _staging_path(self, staging: object | None) -> Path | None:
        if staging is None:
            return None
        value = getattr(staging, "path", staging)
        if isinstance(value, (str, Path)):
            return Path(value)
        return None

    def _terminal_status(
        self, facts: _IngestionFacts, manifest: SnapshotManifest, publication: object
    ) -> JobState:
        del manifest, publication
        if facts.merge_status == "failed":
            return JobState.FAILED
        if (
            facts.failed_symbols
            or facts.quarantined_rows
            or facts.gaps
            or facts.validation.report.summary.stale_symbols
        ):
            return JobState.PARTIALLY_SUCCEEDED
        return JobState.SUCCEEDED

    def _quality_errors(
        self,
        batches: Sequence[ProviderBatchResult],
        *,
        failed_symbols: Sequence[str],
        quarantined: Sequence[QuarantineRecord],
        gaps: Sequence[DataGap],
        validation: ValidationOutput,
        redactor: Redactor | None = None,
        extra: Iterable[ActionableError] = (),
    ) -> tuple[ActionableError, ...]:
        errors: list[ActionableError] = list(extra)
        by_symbol: set[str] = set()
        for batch in batches:
            for outcome in batch.outcomes:
                if outcome.status is SymbolOutcomeStatus.FAILURE:
                    by_symbol.add(outcome.symbol)
                    errors.extend(outcome.errors)
                    if not outcome.errors:
                        errors.append(
                            self._provider_missing_error(
                                outcome.symbol, batch.request.requested_range
                            )
                        )
        for symbol in failed_symbols:
            normalized = normalize_symbol(symbol)
            if normalized not in by_symbol:
                errors.append(self._provider_missing_error(normalized, None))
        for row in quarantined:
            category = (
                ErrorCategory.NORMALIZATION_POLICY
                if any("normalization" in code for code in row.reason_codes)
                else (
                    ErrorCategory.VALIDATION_DUPLICATE_CONFLICT
                    if any("duplicate" in code for code in row.reason_codes)
                    else ErrorCategory.VALIDATION_ROW
                )
            )
            errors.append(
                ActionableError(
                    operation="ingestion.validate",
                    category=category,
                    message=(
                        f"A {row.symbol or 'provider'} record was quarantined "
                        "during validation."
                    ),
                    corrective_action=(
                        "Inspect the quarantine reason and retry after correcting "
                        "the source or policy input."
                    ),
                    symbol=row.symbol,
                    session=row.session,
                )
            )
        for gap in gaps:
            errors.append(
                ActionableError(
                    operation="ingestion.validate",
                    category=ErrorCategory.VALIDATION_GAP,
                    message=(
                        f"No accepted bar was returned for {gap.symbol} on "
                        f"{gap.expected_session.isoformat()}."
                    ),
                    corrective_action=(
                        "Retry the affected symbol or use the disclosed incomplete "
                        "snapshot only for supported analyses."
                    ),
                    symbol=gap.symbol,
                    session=gap.expected_session,
                )
            )
        for summary in validation.report.per_symbol:
            if summary.stale:
                errors.append(
                    ActionableError(
                        operation="ingestion.validate",
                        category=ErrorCategory.VALIDATION_STALE,
                        message=(
                            f"Accepted data for {summary.symbol} is stale by "
                            f"{summary.staleness_lag_sessions} exchange sessions."
                        ),
                        corrective_action=(
                            "Request a completed range through the latest available "
                            "exchange session and retry."
                        ),
                        symbol=summary.symbol,
                    )
                )
        sanitized: list[ActionableError] = []
        for error in errors:
            if not isinstance(error, ActionableError):
                continue
            value = error
            if redactor is not None:
                value = cast(ActionableError, redactor.redact_error(value))
            sanitized.append(value)
        unique = {error: error for error in sanitized}
        return tuple(sorted(unique.values(), key=ActionableError.sort_key))

    def _job_report(
        self,
        job: object,
        stage: JobStage,
        completed: int,
        total: int | None,
        *,
        warnings: Iterable[str] = (),
        context: Mapping[str, object] | None = None,
    ) -> None:
        method = getattr(job, "report", None)
        if callable(method):
            self._invoke(
                method,
                stage=stage,
                completed_units=completed,
                total_units=total,
                warnings=tuple(warnings),
                context=context or {},
            )

    def _terminal_failure(self, job: object, errors: Sequence[ActionableError]) -> None:
        method = getattr(job, "fail", None)
        if not callable(method):
            return
        if getattr(job, "state", None) is JobState.RUNNING:
            try:
                method(errors[0])
            except Exception:
                return

    def _retry_policy(self, config: object) -> RetryPolicy:
        policy = getattr(config, "retry", None)
        if isinstance(policy, RetryPolicyConfig):
            return policy
        if policy is not None and all(
            hasattr(policy, name)
            for name in (
                "attempts",
                "initial_delay_seconds",
                "max_delay_seconds",
                "backoff_multiplier",
            )
        ):
            return cast(RetryPolicy, policy)
        return RetryPolicyConfig.model_validate({})

    def _redactor_for(self, config: object) -> Redactor:
        if self.redactor is not None:
            return self.redactor
        if isinstance(config, ResolvedConfig):
            return Redactor.from_config(config)
        return Redactor()

    @staticmethod
    def _universe(config: object) -> tuple[str, ...]:
        data = getattr(config, "data", config)
        values = getattr(data, "universe", getattr(config, "universe", ()))
        if isinstance(values, str):
            values = (values,)
        normalized = tuple(normalize_symbol(value) for value in values)
        if not normalized:
            raise ValueError("configured universe must contain at least one symbol")
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _universe_from_records(
        records: Sequence[ProviderRecord], parent_id: str | None
    ) -> tuple[str, ...]:
        del parent_id
        return tuple(dict.fromkeys(record.symbol for record in records)) or ("SPY",)

    @staticmethod
    def _benchmark(config: object) -> str:
        data = getattr(config, "data", config)
        return normalize_symbol(
            str(getattr(data, "benchmark", getattr(config, "benchmark", "SPY")))
        )

    @staticmethod
    def _ordered_symbols(universe: Sequence[str], benchmark: str) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in (*universe, benchmark):
            symbol = normalize_symbol(value)
            if symbol not in seen:
                result.append(symbol)
                seen.add(symbol)
        return tuple(result)

    @staticmethod
    def _parent_id(parent: object | None) -> str | None:
        if isinstance(parent, IncrementalParent):
            return parent.snapshot_id
        if isinstance(parent, SnapshotManifest):
            return parent.snapshot_id
        value = getattr(parent, "snapshot_id", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _requested_range(config: object) -> DateRange:
        data = getattr(config, "data", config)
        value = getattr(
            data, "requested_range", getattr(config, "requested_range", None)
        )
        if isinstance(value, DateRange):
            return value
        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        if isinstance(start, date) and isinstance(end, date):
            return DateRange(start, end)
        raise ValueError("config.data.requested_range is required")

    @staticmethod
    def _integer_config(config: object, path: str, default: int) -> int:
        current: object = config
        for component in path.split("."):
            current = getattr(current, component, None)
            if current is None:
                return default
        minimum = (
            0
            if path
            in {
                "data.staleness_sessions",
                "data.revision_overlap_sessions",
            }
            else 1
        )
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or current < minimum
        ):
            bound = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{path} must be a {bound} integer")
        return current

    @staticmethod
    def _text_config(config: object, path: str, default: str) -> str:
        current: object = config
        for component in path.split("."):
            current = getattr(current, component, None)
            if current is None:
                return default
        return str(current)

    @staticmethod
    def _configuration_checksum(config: object, redactor: Redactor) -> str:
        value = getattr(config, "configuration_checksum", None) or getattr(
            config, "non_secret_checksum", None
        )
        if isinstance(value, str) and len(value) == 64:
            return value
        try:
            if isinstance(config, ResolvedConfig):
                return sha256_bytes(ConfigurationSerializer().serialize(config))
            dumped = (
                config.model_dump(mode="python")
                if hasattr(config, "model_dump")
                else redactor.redact_structured(config)
            )
            return sha256_canonical_json(cast(object, dumped))
        except Exception:
            return sha256_canonical_json(
                {
                    "universe": list(DataIngestionService._universe(config)),
                    "requested_range": DataIngestionService._requested_range(
                        config
                    ).to_content_dict(),
                }
            )

    @staticmethod
    def _provider_missing_error(
        symbol: str, requested_range: DateRange | None
    ) -> ActionableError:
        range_text = requested_range.to_content_dict() if requested_range else None
        suffix = (
            ""
            if range_text is None
            else f" for {range_text['start']} through {range_text['end']}"
        )
        return ActionableError(
            operation="ingestion.fetch",
            category=ErrorCategory.PROVIDER_TERMINAL,
            message=f"No usable provider records were returned for {symbol}{suffix}.",
            corrective_action=(
                "Retry the affected symbol with the configured provider policy "
                "or inspect the provider response."
            ),
            symbol=symbol,
        )

    @staticmethod
    def _provider_exception_error(
        error: BaseException, symbols: Sequence[str]
    ) -> ActionableError:
        del error
        return ActionableError(
            operation="ingestion.fetch",
            category=ErrorCategory.PROVIDER_TERMINAL,
            message=f"The provider request failed for {', '.join(symbols)}.",
            corrective_action=(
                "Retry the ingestion or inspect the provider adapter diagnostics."
            ),
            symbol=symbols[0] if len(symbols) == 1 else None,
        )

    @staticmethod
    def _parent_error(snapshot_id: str) -> ActionableError:
        return ActionableError(
            operation="incremental.verify_parent",
            category=ErrorCategory.INTEGRITY_CHECKSUM,
            message=(
                "The parent Data_Snapshot could not be verified for incremental use."
            ),
            corrective_action=(
                "Restore a checksum-verified parent snapshot or select another "
                "published snapshot."
            ),
            field_path="parent_snapshot_id",
            correlation_id=snapshot_id,
        )

    @staticmethod
    def _input_error(message: str, *, field_path: str) -> ActionableError:
        return ActionableError(
            operation="ingestion.input",
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message=" ".join(message.splitlines()) or "Invalid ingestion input.",
            corrective_action="Correct the ingestion request/configuration and retry.",
            field_path=field_path,
        )

    @staticmethod
    def _unexpected_error(operation: str, error: BaseException) -> ActionableError:
        # The domain conversion keeps the boundary payload generic while the
        # original exception remains available to the injected diagnostics sink.
        return ActionableError.from_unexpected_exception(operation, error)

    @staticmethod
    def _unwrap_result(value: object, operation: str) -> object:
        if isinstance(value, Err):
            raise _IngestionFailure(value.errors)
        if isinstance(value, Ok):
            return value.value
        if isinstance(value, (SnapshotManifest, ValidationOutput, IncrementalParent)):
            return value
        if value is None:
            raise _IngestionFailure(
                (
                    ActionableError(
                        operation=operation,
                        category=ErrorCategory.STORAGE_IO,
                        message="The injected application port returned no result.",
                        corrective_action="Repair the adapter and retry the operation.",
                    ),
                )
            )
        return value

    @staticmethod
    def _invoke(method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        """Call a port while allowing focused fakes to implement fewer keywords."""

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        parameters = signature.parameters
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return method(*args, **kwargs)
        accepted = {key: value for key, value in kwargs.items() if key in parameters}
        # A few small fakes use ``chunk_size`` or ``max_rows`` instead of the
        # production ``write_chunk_size`` spelling.
        if (
            "write_chunk_size" not in parameters
            and "chunk_size" in parameters
            and "write_chunk_size" in kwargs
        ):
            accepted["chunk_size"] = kwargs["write_chunk_size"]
        if (
            "max_rows" not in parameters
            and "chunk_size" in parameters
            and "max_rows" in kwargs
        ):
            accepted["chunk_size"] = kwargs["max_rows"]
        return method(*args, **accepted)


class _SystemIngestionClock:
    def utc_now(self) -> datetime:
        return datetime.now(UTC)


class _IngestionFailure(Exception):
    """Internal carrier for already-sanitized expected application errors."""

    def __init__(self, errors: Sequence[ActionableError]) -> None:
        values = tuple(errors)
        if not values:
            raise ValueError("ingestion failure requires at least one actionable error")
        super().__init__(values[0].message)
        self.errors = values


# Additional names are intentionally exported for composition roots and tests.
IngestionService = DataIngestionService
DataIngestion = DataIngestionService


__all__ = [
    "DataIngestion",
    "DataIngestionResult",
    "DataIngestionRequest",
    "DataIngestionService",
    "ExchangeCalendarPort",
    "IngestionClock",
    "IngestionRequest",
    "IngestionResult",
    "IngestionService",
    "ParentLoader",
    "ProgressCallback",
    "SnapshotPublisherPort",
]
