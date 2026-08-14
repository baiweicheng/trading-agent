"""Focused service-level coverage for staged data ingestion orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research_platform.application.incremental import (
    IncrementalMerger,
    IncrementalParent,
)
from quant_research_platform.application.ingestion import (
    DataIngestionService,
    IngestionRequest,
)
from quant_research_platform.config.models import (
    DataConfig,
    DateRangeConfig,
    PathConfig,
    ResolvedConfig,
    RetryPolicyConfig,
)
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    Ok,
    ProviderFailureKind,
    ProviderFailureReason,
)
from quant_research_platform.domain.market import (
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
)
from quant_research_platform.domain.normalization import (
    CausalForwardAdjustmentV1,
    Normalizer,
)
from quant_research_platform.domain.validation import ValidationService

START = date(2024, 1, 2)
SHORT_END = date(2024, 1, 3)
EXTENDED_END = date(2024, 1, 5)


class FixtureCalendar:
    """Small deterministic weekday calendar implementing the application seam."""

    name = "XNYS-fixture"
    version = "fixture-1"

    @staticmethod
    def is_session(value: date) -> bool:
        return value.weekday() < 5

    def sessions(
        self,
        start: date,
        end: date,
        *,
        completed_at: datetime | None = None,
    ) -> tuple[date, ...]:
        del completed_at
        return tuple(
            value
            for offset in range((end - start).days + 1)
            if self.is_session(value := start + timedelta(days=offset))
        )

    @staticmethod
    def close_timestamp(session: date) -> datetime:
        return datetime(
            session.year,
            session.month,
            session.day,
            21,
            0,
            tzinfo=UTC,
        )

    @staticmethod
    def schedule_checksum(start: date, end: date) -> str:
        del start, end
        return "a" * 64


class FixedClock:
    def utc_now(self) -> datetime:
        return datetime(2024, 1, 8, 22, 0, tzinfo=UTC)


class FixtureProvider:
    name = "fixture-provider"

    def __init__(
        self,
        calendar: FixtureCalendar,
        *,
        fail_symbols: Iterable[str] = (),
        bad_symbols: Iterable[str] = (),
        missing_dates: Iterable[date] = (),
        failure_text: str = "No usable provider records were returned.",
    ) -> None:
        self.calendar = calendar
        self.fail_symbols = {symbol.upper() for symbol in fail_symbols}
        self.bad_symbols = {symbol.upper() for symbol in bad_symbols}
        self.missing_dates = set(missing_dates)
        self.failure_text = failure_text
        self.requests: list[ProviderRequest] = []

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        self.requests.append(request)
        outcomes: list[SymbolOutcome] = []
        for symbol in request.symbols:
            if symbol in self.fail_symbols:
                outcomes.append(self._failure(symbol))
                continue
            records = tuple(
                self._record(symbol, session, request)
                for session in self.calendar.sessions(request.start, request.end)
                if session not in self.missing_dates
            )
            if not records:
                outcomes.append(
                    self._failure(symbol, reason=ProviderFailureReason.EMPTY_RESPONSE)
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

    def _failure(
        self,
        symbol: str,
        *,
        reason: ProviderFailureReason = ProviderFailureReason.EMPTY_RESPONSE,
    ) -> SymbolOutcome:
        error = ActionableError(
            operation="provider.fetch",
            category=ErrorCategory.PROVIDER_TERMINAL,
            message=f"{self.failure_text} ({symbol})",
            corrective_action=(
                "Retry the affected symbol or inspect the provider adapter."
            ),
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

    def _record(
        self, symbol: str, session: date, request: ProviderRequest
    ) -> ProviderRecord:
        if symbol in self.bad_symbols:
            open_price = Decimal("10")
            high = Decimal("9")
            low = Decimal("9")
            close = Decimal("10")
        else:
            open_price = Decimal("10")
            high = Decimal("11")
            low = Decimal("9")
            close = Decimal("10")
        return ProviderRecord(
            provider=self.name,
            request_content_key=request.content_key,
            symbol=symbol,
            raw_bar=RawDailyBar(
                provider_date=session,
                open=open_price,
                high=high,
                low=low,
                close=close,
                adj_close=close,
                volume=Decimal("100"),
            ),
        )


@dataclass(frozen=True)
class PublishedFixture:
    manifest: object
    reused: bool = False

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id


class RecordingPublisher:
    """Publisher spy that models immutable-ID reuse and pre-publication failure."""

    def __init__(self) -> None:
        self.staging_operations: list[str] = []
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.manifests: dict[str, object] = {}
        self.fail_next = False

    def create_staging(self, operation_id: str) -> object:
        self.staging_operations.append(operation_id)
        return SimpleNamespace(path=Path("fixture-staging") / operation_id)

    def publish_snapshot(self, manifest: object, **kwargs: object) -> PublishedFixture:
        self.calls.append((manifest, dict(kwargs)))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("publication failed for secret-token")
        snapshot_id = manifest.snapshot_id
        existing = self.manifests.get(snapshot_id)
        if existing is not None:
            if (
                existing.to_content_identity_dict()
                != manifest.to_content_identity_dict()
            ):
                raise ValueError(
                    "scientific content changed at an existing snapshot ID"
                )
            return PublishedFixture(existing, reused=True)
        self.manifests[snapshot_id] = manifest
        return PublishedFixture(manifest, reused=False)


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], int | None, Path | None]] = []

    def _write(
        self,
        role: str,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        self.calls.append((role, tuple(rows), write_chunk_size, staging))
        return ()

    def write_raw(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        return self._write(
            "raw", rows, write_chunk_size=write_chunk_size, staging=staging
        )

    def write_normalized(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        return self._write(
            "normalized", rows, write_chunk_size=write_chunk_size, staging=staging
        )

    def write_quarantine(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        return self._write(
            "quarantine", rows, write_chunk_size=write_chunk_size, staging=staging
        )

    def write_gaps(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        return self._write(
            "gap", rows, write_chunk_size=write_chunk_size, staging=staging
        )

    def write_validation(
        self,
        rows: Iterable[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        return self._write(
            "validation", rows, write_chunk_size=write_chunk_size, staging=staging
        )


def make_config(
    universe: tuple[str, ...] = ("AAPL", "MSFT"),
    *,
    end: date = SHORT_END,
    batch_size: int = 5,
    staleness_sessions: int = 1,
    revision_overlap_sessions: int = 5,
    write_chunk_rows: int = 2,
) -> ResolvedConfig:
    return ResolvedConfig(
        paths=PathConfig(),
        retry=RetryPolicyConfig(
            attempts=1,
            initial_delay_seconds=Decimal("0"),
            max_delay_seconds=Decimal("0"),
            backoff_multiplier=Decimal("1"),
        ),
        data=DataConfig(
            universe=universe,
            requested_range=DateRangeConfig(start=START, end=end),
            batch_size=batch_size,
            staleness_sessions=staleness_sessions,
            revision_overlap_sessions=revision_overlap_sessions,
            write_chunk_rows=write_chunk_rows,
        ),
    )


def make_service(
    provider: FixtureProvider,
    calendar: FixtureCalendar,
    publisher: RecordingPublisher,
    writer: RecordingWriter | None = None,
    *,
    redactor: Redactor | None = None,
    incremental_merger: IncrementalMerger | None = None,
) -> DataIngestionService:
    return DataIngestionService(
        provider,
        calendar,
        parquet_store=writer,
        snapshot_publisher=publisher,
        incremental_merger=incremental_merger,
        clock=FixedClock(),
        redactor=redactor or Redactor(),
    )


def make_parent(result: object, calendar: FixtureCalendar) -> IncrementalParent:
    manifest = result.manifest
    requested_range = manifest.content_identity.requested_range
    sessions = calendar.sessions(requested_range.start, requested_range.end)
    symbols = (
        *manifest.content_identity.configured_universe,
        manifest.content_identity.benchmark_symbol,
    )
    return IncrementalParent.from_manifest(
        manifest,
        accepted_rows=result.accepted_rows,
        provider_records=result.provider_records,
        expected_sessions={symbol: sessions for symbol in symbols},
        validation_report=result.validation.report,
    )


def make_merger(calendar: FixtureCalendar) -> IncrementalMerger:
    normalizer = Normalizer(CausalForwardAdjustmentV1())
    validator = ValidationService(calendar=calendar, benchmark_symbol="SPY")
    return IncrementalMerger(
        calendar,
        normalizer=normalizer,
        policy=CausalForwardAdjustmentV1(),
        validator=validator,
    )


def test_ingestion_success_batches_writes_provenance_and_monotonic_progress() -> None:
    calendar = FixtureCalendar()
    provider = FixtureProvider(calendar)
    publisher = RecordingPublisher()
    writer = RecordingWriter()
    updates = []
    config = make_config(
        universe=("AAPL", "MSFT", "SPY"),
        batch_size=2,
        staleness_sessions=0,
        revision_overlap_sessions=0,
        write_chunk_rows=2,
    )

    result = make_service(provider, calendar, publisher, writer).ingest(
        config,
        progress_callback=updates.append,
    )

    assert isinstance(result, Ok)
    value = result.value
    assert value.status.value == "succeeded"
    assert tuple(
        symbol for request in provider.requests for symbol in request.symbols
    ) == (
        "AAPL",
        "MSFT",
        "SPY",
    )
    assert all(len(request.symbols) <= 2 for request in provider.requests)
    assert all(
        batch.request == request
        for batch, request in zip(
            value.provider_batches, provider.requests, strict=True
        )
    )
    assert all(
        metadata.request_content_key == batch.request.content_key
        for metadata, batch in zip(
            value.provider_requests, value.provider_batches, strict=True
        )
    )
    assert all(
        metadata.retrieval_started_at is not None
        for metadata in value.provider_requests
    )
    assert {call[0] for call in writer.calls} == {"raw", "normalized", "validation"}
    assert all(call[2] == 2 for call in writer.calls)
    assert all(call[3] is not None for call in writer.calls)
    assert publisher.calls[0][1]["staging"] is not None

    completed = [update.completed_units for update in updates]
    assert completed == sorted(completed)
    assert completed[-1] == 3
    stages = [update.stage.value for update in updates]
    assert "fetching" in stages
    assert "normalizing" in stages
    assert "validating" in stages
    assert "publishing" in stages
    assert stages[-1] == "completed"
    assert all(isinstance(update.elapsed_seconds, Decimal) for update in updates)


def test_partial_batch_is_symbol_isolated_and_disclosed() -> None:
    calendar = FixtureCalendar()
    provider = FixtureProvider(calendar, fail_symbols=("MSFT",))
    publisher = RecordingPublisher()
    config = make_config(universe=("AAPL", "MSFT"), batch_size=3)

    result = make_service(provider, calendar, publisher).ingest(config)

    assert isinstance(result, Ok)
    value = result.value
    assert value.status.value == "partially_succeeded"
    assert value.failed_symbols == ("MSFT",)
    assert {row.symbol for row in value.accepted_rows} == {"AAPL", "SPY"}
    assert any(gap.symbol == "MSFT" for gap in value.gaps)
    assert any(error.symbol == "MSFT" for error in value.errors)
    assert value.limitation_disclosure.data_failures == value.errors
    assert "Recorded data failures:" in value.limitation_disclosure.format_for_display()
    assert publisher.manifests


def test_quarantine_gap_and_staleness_make_usable_ingestion_partial() -> None:
    calendar = FixtureCalendar()
    provider = FixtureProvider(calendar, bad_symbols=("AAPL",))
    publisher = RecordingPublisher()
    config = make_config(
        universe=("AAPL",),
        staleness_sessions=0,
        batch_size=2,
    )

    result = make_service(provider, calendar, publisher).ingest(config)

    assert isinstance(result, Ok)
    value = result.value
    assert value.status.value == "partially_succeeded"
    assert value.accepted_rows and {row.symbol for row in value.accepted_rows} == {
        "SPY"
    }
    assert value.quarantined_rows
    assert value.gaps
    assert "AAPL" in value.validation.report.summary.stale_symbols
    categories = {error.category for error in value.errors}
    assert ErrorCategory.VALIDATION_ROW in categories
    assert ErrorCategory.VALIDATION_GAP in categories
    assert ErrorCategory.VALIDATION_STALE in categories


def test_empty_provider_outcomes_fail_before_publication() -> None:
    calendar = FixtureCalendar()
    provider = FixtureProvider(calendar, fail_symbols=("AAPL", "MSFT", "SPY"))
    publisher = RecordingPublisher()
    config = make_config(universe=("AAPL", "MSFT"), batch_size=3)

    result = make_service(provider, calendar, publisher).ingest(config)

    assert isinstance(result, Err)
    assert result.errors
    assert any(
        error.category is ErrorCategory.PROVIDER_TERMINAL for error in result.errors
    )
    assert not publisher.manifests
    assert (
        any(
            error.category is ErrorCategory.SNAPSHOT_NOT_READY
            for error in result.errors
        )
        is False
    )


def test_publication_failure_preserves_prior_snapshot_and_returns_sanitized_error() -> (
    None
):
    calendar = FixtureCalendar()
    publisher = RecordingPublisher()
    config = make_config(universe=("AAPL",), batch_size=2)
    first_provider = FixtureProvider(calendar)
    first = make_service(first_provider, calendar, publisher).ingest(config)
    assert isinstance(first, Ok)
    prior_id = first.value.snapshot_id

    publisher.fail_next = True
    second_provider = FixtureProvider(calendar)
    second = make_service(second_provider, calendar, publisher).ingest(config)

    assert isinstance(second, Err)
    assert any(
        error.category is ErrorCategory.INTERNAL_UNEXPECTED for error in second.errors
    )
    assert tuple(publisher.manifests) == (prior_id,)
    assert publisher.manifests[prior_id].snapshot_id == prior_id


def test_unchanged_ingestion_reuses_snapshot_id() -> None:
    calendar = FixtureCalendar()
    publisher = RecordingPublisher()
    config = make_config(
        universe=("AAPL",),
        revision_overlap_sessions=0,
        staleness_sessions=0,
    )
    first_provider = FixtureProvider(calendar)
    first = make_service(first_provider, calendar, publisher).ingest(config)
    assert isinstance(first, Ok)
    parent = make_parent(first.value, calendar)

    second_provider = FixtureProvider(calendar)
    second_service = make_service(
        second_provider,
        calendar,
        publisher,
        incremental_merger=make_merger(calendar),
    )
    second = second_service.ingest(
        IngestionRequest(parent_snapshot=parent),
        config,
    )

    assert isinstance(second, Ok)
    assert second.value.snapshot_id == first.value.snapshot_id
    assert second.value.snapshot_reused is True
    assert second.value.provider_batches == ()
    assert second_provider.requests == []
    assert len(publisher.manifests) == 1


def test_incremental_failed_symbol_retains_parent_coverage() -> None:
    calendar = FixtureCalendar()
    publisher = RecordingPublisher()
    parent_config = make_config(universe=("AAPL", "MSFT"), end=SHORT_END)
    first_provider = FixtureProvider(calendar)
    first = make_service(first_provider, calendar, publisher).ingest(parent_config)
    assert isinstance(first, Ok)
    parent = make_parent(first.value, calendar)

    update_config = make_config(
        universe=("AAPL", "MSFT"),
        end=EXTENDED_END,
        revision_overlap_sessions=1,
        batch_size=3,
    )
    update_provider = FixtureProvider(calendar, fail_symbols=("MSFT",))
    update = make_service(
        update_provider,
        calendar,
        publisher,
        incremental_merger=make_merger(calendar),
    ).ingest(IngestionRequest(parent_snapshot=parent), update_config)

    assert isinstance(update, Ok)
    value = update.value
    assert value.status.value == "partially_succeeded"
    assert value.failed_symbols == ("MSFT",)
    assert value.retained_parent_coverage_symbols == ("MSFT",)
    assert {row.session for row in value.accepted_rows if row.symbol == "MSFT"} == {
        START,
        SHORT_END,
    }
    assert value.snapshot_id != first.value.snapshot_id
    assert update_provider.requests
    assert update_provider.requests[0].start == SHORT_END
    assert update_provider.requests[0].end == EXTENDED_END


def test_provider_error_and_progress_warning_are_redacted() -> None:
    calendar = FixtureCalendar()
    secret = "secret-token"
    provider = FixtureProvider(
        calendar,
        fail_symbols=("AAPL",),
        failure_text=f"provider credential {secret} was rejected",
    )
    publisher = RecordingPublisher()
    updates = []
    config = make_config(universe=("AAPL",), batch_size=2)

    result = make_service(
        provider,
        calendar,
        publisher,
        redactor=Redactor([secret]),
    ).ingest(config, progress=updates.append)

    assert isinstance(result, Ok)
    value = result.value
    assert value.status.value == "partially_succeeded"
    assert all(secret not in error.message for error in value.errors)
    assert all(
        secret not in warning for update in updates for warning in update.warnings
    )
    assert secret not in value.limitation_disclosure.format_for_display()
    assert any("[REDACTED]" in error.message for error in value.errors)


@pytest.mark.parametrize(
    "path", ("data.staleness_sessions", "data.revision_overlap_sessions")
)
def test_zero_valued_session_settings_are_accepted(path: str) -> None:
    calendar = FixtureCalendar()
    provider = FixtureProvider(calendar)
    publisher = RecordingPublisher()
    kwargs = {"staleness_sessions": 0, "revision_overlap_sessions": 0}
    config = make_config(universe=("AAPL",), **kwargs)

    result = make_service(provider, calendar, publisher).ingest(config)

    assert isinstance(result, Ok), path
