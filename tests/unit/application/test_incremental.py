"""Focused examples for revision-overlap incremental merging."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_research_platform.application.incremental import (
    IncrementalMerger,
    IncrementalParent,
)
from quant_research_platform.application.snapshots import SnapshotManifestAssembler
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import CalendarIdentity
from quant_research_platform.domain.market import (
    DateRange,
    ProviderRecord,
    RawCorporateAction,
    RawDailyBar,
)
from quant_research_platform.domain.normalization import Normalizer
from quant_research_platform.domain.validation import ValidationService


class FixtureCalendar:
    name = "XNYS"
    version = "fixture"

    def __init__(self, sessions: tuple[date, ...]) -> None:
        self._sessions = tuple(sessions)

    def is_session(self, value: date) -> bool:
        return value in self._sessions

    def close_timestamp(self, session: date) -> datetime:
        if not self.is_session(session):
            raise ValueError("not a fixture session")
        return datetime.combine(session, datetime.min.time(), tzinfo=UTC).replace(
            hour=21
        )

    def sessions(
        self, start: date, end: date, *, completed_at: datetime
    ) -> tuple[date, ...]:
        del completed_at
        return tuple(session for session in self._sessions if start <= session <= end)


def _record(
    symbol: str,
    session: date,
    *,
    close: str = "101",
    split_ratio: str = "1",
) -> ProviderRecord:
    from quant_research_platform.domain.market import ProviderRequest

    request = ProviderRequest((symbol,), date(2024, 1, 2), date(2024, 1, 10))
    close_value = Decimal(close)
    return ProviderRecord(
        provider="fixture",
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=RawDailyBar(
            provider_date=session,
            open=close_value - 1,
            high=close_value + 1,
            low=close_value - 2,
            close=close_value,
            adj_close=close_value,
            volume=Decimal("1000"),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal(split_ratio),
        ),
    )


def _parent(
    calendar: FixtureCalendar,
    records: tuple[ProviderRecord, ...],
    requested_range: DateRange,
    universe: tuple[str, ...] = ("AAPL", "MSFT"),
) -> IncrementalParent:
    normalized = tuple(Normalizer().normalize(records, calendar))
    expected = {
        symbol: calendar.sessions(
            requested_range.start,
            requested_range.end,
            completed_at=datetime.max.replace(tzinfo=UTC),
        )
        for symbol in (*universe, "SPY")
    }
    validation = ValidationService(calendar=calendar).validate(
        normalized,
        expected,
        1,
        requested_range=requested_range,
        benchmark_symbol="SPY",
    )
    report = validation.report
    manifest = SnapshotManifestAssembler.assemble(
        provider="fixture",
        requested_range=requested_range,
        covered_range=report.summary.covered_range,
        configured_universe=universe,
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "fixture", "a" * 64),
        configuration_checksum="b" * 64,
        objects=(),
        validation=report,
        limitation_disclosure=LimitationDisclosure.current(),
        created_at=datetime(2024, 1, 10, 12, tzinfo=UTC),
    )
    return IncrementalParent.from_manifest(
        manifest,
        accepted_rows=validation.accepted_rows,
        provider_records=records,
        expected_sessions=expected,
        validation_report=report,
    )


def _all_records(
    sessions: tuple[date, ...], *, split_session: date | None = None
) -> tuple[ProviderRecord, ...]:
    result: list[ProviderRecord] = []
    for symbol in ("AAPL", "MSFT", "SPY"):
        for session in sessions:
            result.append(
                _record(
                    symbol,
                    session,
                    split_ratio=(
                        "2"
                        if symbol == "AAPL" and session == split_session
                        else "1"
                    ),
                )
            )
    return tuple(result)


def test_plan_uses_nonzero_overlap_and_rejects_shrink_and_back_extension() -> None:
    sessions = tuple(date(2024, 1, day) for day in (2, 3, 4, 5))
    calendar = FixtureCalendar(sessions)
    parent_range = DateRange(sessions[0], sessions[2])
    parent = _parent(calendar, _all_records(sessions[:3]), parent_range)
    merger = IncrementalMerger(calendar)

    plan = merger.plan(parent, DateRange(sessions[0], sessions[3]), 2)

    assert plan.overlap_sessions == sessions[1:3]
    assert plan.later_sessions == (sessions[3],)
    assert plan.suffix_sessions == sessions[1:]
    assert plan.boundary_session == sessions[1]

    with pytest.raises(ValueError, match="must not precede"):
        merger.plan(parent, DateRange(sessions[0], sessions[1]), 2)
    with pytest.raises(ValueError, match="must equal"):
        merger.plan(parent, DateRange(sessions[0].replace(day=1), sessions[3]), 2)


def test_zero_overlap_starts_at_first_later_session_and_extended_rows_are_added(
) -> None:
    sessions = tuple(date(2024, 1, day) for day in (2, 3, 4, 5))
    calendar = FixtureCalendar(sessions)
    parent_range = DateRange(sessions[0], sessions[2])
    parent_records = _all_records(sessions[:3])
    parent = _parent(calendar, parent_records, parent_range)
    incoming = tuple(_record(symbol, sessions[3]) for symbol in ("AAPL", "MSFT", "SPY"))

    result = IncrementalMerger(calendar).merge_or_raise(
        parent,
        DateRange(sessions[0], sessions[3]),
        revision_overlap=0,
        records=incoming,
    )

    assert result.plan.boundary_session == sessions[3]
    assert result.plan.overlap_sessions == ()
    assert result.new_rows
    assert {row.session for row in result.new_rows} == {sessions[3]}
    assert len({row.session_key for row in result.accepted_rows}) == len(
        result.accepted_rows
    )
    assert result.snapshot_id != parent.snapshot_id


def test_unchanged_overlap_reuses_parent_id_but_revision_rebuilds_suffix_from_seed(
) -> None:
    sessions = tuple(date(2024, 1, day) for day in (2, 3, 4))
    calendar = FixtureCalendar(sessions)
    requested_range = DateRange(sessions[0], sessions[-1])
    original = _all_records(sessions)
    parent = _parent(calendar, original, requested_range)
    incoming = tuple(
        record for record in original if record.provider_date in sessions[1:]
    )

    unchanged = IncrementalMerger(calendar).merge_or_raise(
        parent,
        requested_range,
        revision_overlap=2,
        records=incoming,
    )
    assert unchanged.snapshot_id == parent.snapshot_id
    assert unchanged.reused_parent

    revised_records = tuple(
        _record(
            symbol,
            session,
            split_ratio=(
                "2" if symbol == "AAPL" and session == sessions[1] else "1"
            ),
        )
        for symbol in ("AAPL", "MSFT", "SPY")
        for session in sessions[1:]
    )
    revised = IncrementalMerger(calendar).merge_or_raise(
        parent,
        requested_range,
        revision_overlap=2,
        records=revised_records,
    )
    assert revised.snapshot_id != parent.snapshot_id
    revised_aapl = next(
        row
        for row in revised.accepted_rows
        if row.symbol == "AAPL" and row.session == sessions[1]
    )
    assert revised_aapl.cumulative_price_factor == Decimal("2.000000000000000000")
    original_aapl = next(
        row
        for row in parent.accepted_rows
        if row.symbol == "AAPL" and row.session == sessions[1]
    )
    assert original_aapl.cumulative_price_factor == Decimal("1.000000000000000000")


def test_failed_symbol_retains_parent_coverage_and_missing_parent_has_zero_new_content(
) -> None:
    sessions = tuple(date(2024, 1, day) for day in (2, 3, 4))
    calendar = FixtureCalendar(sessions)
    requested_range = DateRange(sessions[0], sessions[1])
    parent_records = tuple(
        record
        for record in _all_records(sessions[:2])
        if record.symbol != "MSFT"
    )
    parent = _parent(calendar, parent_records, requested_range)

    # AAPL has accepted parent rows and is explicitly failed on re-request.
    retained = IncrementalMerger(calendar).merge_or_raise(
        parent,
        requested_range,
        revision_overlap=1,
        failed_symbols=("AAPL",),
    )
    assert retained.failed_symbols == ("AAPL",)
    assert retained.retained_parent_coverage_symbols == ("AAPL",)
    assert retained.new_rows == ()
    assert any(error.symbol == "AAPL" for error in retained.failure_errors)
    assert retained.limitation_disclosure.data_failures

    # MSFT is not represented in the parent; no accepted content can be
    # fabricated for it when the provider fails.
    no_parent = IncrementalMerger(calendar).merge_or_raise(
        parent,
        requested_range,
        revision_overlap=1,
        failed_symbols=("MSFT",),
    )
    assert no_parent.failed_symbols == ("MSFT",)
    assert "MSFT" not in no_parent.retained_parent_coverage_symbols
    assert not any(row.symbol == "MSFT" for row in no_parent.new_rows)
    assert no_parent.failed_without_parent_coverage == ("MSFT",)
    assert len({row.session_key for row in no_parent.accepted_rows}) == len(
        no_parent.accepted_rows
    )


def test_snapshot_manager_failure_is_returned_as_actionable_error() -> None:
    class RejectingManager:
        def open_verified(self, snapshot_id: str):
            del snapshot_id
            from quant_research_platform.domain.errors import (
                ActionableError,
                Err,
                ErrorCategory,
            )

            return Err((ActionableError(
                operation="snapshot.open",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                message="corrupt parent",
                corrective_action="repair parent",
            ),))

    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    calendar = FixtureCalendar(sessions)
    parent = _parent(calendar, _all_records(sessions), DateRange(*sessions))
    result = IncrementalMerger(calendar, snapshot_manager=RejectingManager()).merge(
        parent.snapshot_id,
        parent.requested_range,
        records=(),
    )
    assert not isinstance(result, Ok)
    assert result.errors[0].category.value == "integrity.checksum"
