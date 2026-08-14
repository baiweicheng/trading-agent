"""Property test for deterministic validation partitioning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from hypothesis import given, settings, strategies as st

from quant_research_platform.domain.canonical import canonical_json_text
from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    DataGap,
    DateRange,
    ProviderRecord,
    ProviderRequest,
    RawCorporateAction,
    RawDailyBar,
    QuarantineRecord,
    SymbolValidationSummary,
    ValidationReport,
)
from quant_research_platform.domain.normalization import POLICY_VERSION
from quant_research_platform.domain.validation import ValidationService

_START = date(2024, 1, 2)
_REQUEST_END = date(2024, 1, 31)
_SYMBOLS = ("AAPL", "MSFT", "SPY")
_INVALID_KINDS = ("ohlc", "volume", "high", "low", "multi", "missing")
_RANDOM_GROUP_KINDS = (
    "valid",
    "equivalent",
    "invalid",
    "invalid_equivalent",
    "conflict",
    "invalid_conflict",
    "absent",
    "policy",
)


class FixtureCalendar:
    """Small deterministic calendar with no dependency on exchange data."""

    name = "XNYS"
    version = "fixture-xnys-v1"

    def __init__(self, sessions: tuple[date, ...]) -> None:
        self._sessions = frozenset(sessions)

    def is_session(self, value: date) -> bool:
        return value in self._sessions


@dataclass(frozen=True)
class CandidateGroup:
    """One generated candidate multiset for an expected key slot."""

    slot_key: tuple[str, date]
    kind: str
    members: tuple[DailyBarCandidate, ...]


@dataclass(frozen=True)
class ValidationCase:
    """All inputs for one generated validation partition."""

    sessions: tuple[date, ...]
    expected: Mapping[str, tuple[date, ...]]
    groups: tuple[CandidateGroup, ...]
    staleness_threshold: int
    failed_symbols: tuple[str, ...]
    retained_parent_coverage: tuple[str, ...]
    requested_range: DateRange
    comparison_range: DateRange

    @property
    def candidates(self) -> tuple[DailyBarCandidate, ...]:
        return tuple(member for group in self.groups for member in group.members)


@dataclass(frozen=True)
class ReferencePartition:
    """Independent reference projection for the validator's public output."""

    accepted: tuple[tuple[object, ...], ...]
    quarantined: tuple[tuple[object, ...], ...]
    duplicate_counts: tuple[tuple[str, date, int], ...]
    gaps: tuple[tuple[object, ...], ...]
    report: ValidationReport


def _candidate(
    symbol: str,
    session: date,
    seed: int,
    *,
    invalid_kind: str | None = None,
    invalid_policy: bool = False,
) -> DailyBarCandidate:
    """Build a candidate whose raw lineage is stable and independently traceable."""

    price = Decimal(100 + seed)
    raw_close = price + Decimal("1")
    request = ProviderRequest((symbol,), _START, _REQUEST_END)
    record = ProviderRecord(
        provider="fixture",
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=RawDailyBar(
            provider_date=session,
            open=price,
            high=price + Decimal("2"),
            low=price - Decimal("1"),
            close=raw_close,
            adj_close=raw_close,
            volume=Decimal(1000 + seed),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
        ),
    )
    action = CorporateAction(
        symbol=symbol,
        session=session,
        raw_lineage=record.raw_lineage,
        source_fields=("Dividends", "Stock Splits"),
    )
    candidate = DailyBarCandidate(
        symbol=symbol,
        session=session,
        event_timestamp=datetime(
            session.year, session.month, session.day, 21, tzinfo=UTC
        ),
        raw_bar=record.raw_bar,
        raw_action=record.raw_action,
        corporate_action=action,
        adjusted_open=price,
        adjusted_high=price + Decimal("2"),
        adjusted_low=price - Decimal("1"),
        adjusted_close=raw_close,
        adjusted_volume=Decimal(1000 + seed),
        execution_adjusted_open=price,
        sizing_adjusted_close=raw_close,
        cumulative_price_factor=Decimal("1"),
        cumulative_split_factor=Decimal("1"),
        policy_version=POLICY_VERSION,
        raw_lineage=record.raw_lineage,
    )

    if invalid_kind == "ohlc":
        candidate = replace(
            candidate,
            adjusted_open=Decimal("0"),
            adjusted_high=price + Decimal("2"),
            adjusted_low=Decimal("0"),
            adjusted_close=raw_close,
        )
    elif invalid_kind == "volume":
        candidate = replace(candidate, adjusted_volume=Decimal("-1"))
    elif invalid_kind == "high":
        candidate = replace(
            candidate,
            adjusted_high=price - Decimal("0.5"),
            adjusted_low=price - Decimal("2"),
        )
    elif invalid_kind == "low":
        candidate = replace(candidate, adjusted_low=price + Decimal("0.5"))
    elif invalid_kind == "multi":
        candidate = replace(
            candidate,
            adjusted_open=Decimal("0"),
            adjusted_high=Decimal("-1"),
            adjusted_low=Decimal("2"),
            adjusted_close=Decimal("0"),
            adjusted_volume=Decimal("-1"),
        )
    elif invalid_kind == "missing":
        candidate = replace(candidate, adjusted_close=None)
    elif invalid_kind is not None:
        raise AssertionError(f"unknown invalid candidate kind: {invalid_kind}")

    if invalid_policy:
        candidate = replace(candidate, policy_version="not-causal-v0")
    return candidate


def _group_kind(
    symbol_index: int,
    session_index: int,
    session_count: int,
    draw: st.DrawFn,
) -> str:
    """Guarantee representative groups while varying the remaining partition."""

    if symbol_index == 0:
        anchors = {
            0: "valid",
            1: "equivalent",
            2: "invalid",
            3: "invalid_equivalent",
            4: "conflict",
            session_count - 1: "absent",
        }
        if session_index in anchors:
            return anchors[session_index]
    elif symbol_index == 1:
        anchors = {0: "invalid_conflict", 1: "non_session", 2: "policy"}
        if session_index in anchors:
            return anchors[session_index]
    elif symbol_index == 2:
        anchors = {0: "valid", session_count - 1: "absent"}
        if session_index in anchors:
            return anchors[session_index]
    return draw(st.sampled_from(_RANDOM_GROUP_KINDS))


def _make_group(
    symbol: str,
    session: date,
    symbol_index: int,
    session_index: int,
    kind: str,
    seed: int,
    draw: st.DrawFn,
) -> CandidateGroup:
    """Create one candidate multiset without calling production validation code."""

    slot_key = (symbol, session)
    if kind == "absent":
        return CandidateGroup(slot_key, kind, ())

    if kind == "non_session":
        non_session = date(2025, 1, 1) + timedelta(days=symbol_index * 20 + session_index)
        return CandidateGroup(
            slot_key,
            kind,
            (_candidate(symbol, non_session, seed),),
        )

    if kind == "policy":
        return CandidateGroup(
            slot_key,
            kind,
            (_candidate(symbol, session, seed, invalid_policy=True),),
        )

    invalid_kind: str | None = None
    if kind in {"invalid", "invalid_equivalent", "invalid_conflict"}:
        invalid_kind = draw(st.sampled_from(_INVALID_KINDS))

    if kind == "valid":
        members = (_candidate(symbol, session, seed),)
    elif kind == "equivalent":
        base = _candidate(symbol, session, seed)
        members = tuple(base for _ in range(draw(st.integers(2, 4))))
    elif kind == "invalid":
        members = (_candidate(symbol, session, seed, invalid_kind=invalid_kind),)
    elif kind == "invalid_equivalent":
        base = _candidate(symbol, session, seed, invalid_kind=invalid_kind)
        members = tuple(base for _ in range(draw(st.integers(2, 4))))
    elif kind == "conflict":
        members = (
            _candidate(symbol, session, seed),
            _candidate(symbol, session, seed + 1),
        )
    elif kind == "invalid_conflict":
        members = (
            _candidate(symbol, session, seed, invalid_kind=invalid_kind),
            _candidate(symbol, session, seed + 1, invalid_kind=invalid_kind),
        )
    else:
        raise AssertionError(f"unknown candidate group kind: {kind}")
    return CandidateGroup(slot_key, kind, members)


@st.composite
def validation_cases(draw: st.DrawFn) -> ValidationCase:
    """Generate valid, invalid, duplicate, conflict, gap, and lineage facts."""

    session_count = draw(st.integers(min_value=6, max_value=8))
    sessions = tuple(_START + timedelta(days=index) for index in range(session_count))
    groups: list[CandidateGroup] = []
    for symbol_index, symbol in enumerate(_SYMBOLS):
        for session_index, session in enumerate(sessions):
            kind = _group_kind(symbol_index, session_index, session_count, draw)
            seed = draw(st.integers(min_value=0, max_value=10_000))
            groups.append(
                _make_group(
                    symbol,
                    session,
                    symbol_index,
                    session_index,
                    kind,
                    seed,
                    draw,
                )
            )

    failed_symbols = tuple(
        symbol for symbol in _SYMBOLS if draw(st.booleans())
    )
    retained_parent_coverage = tuple(
        symbol for symbol in _SYMBOLS if draw(st.booleans())
    )
    requested_range = DateRange(sessions[0], sessions[-1])
    return ValidationCase(
        sessions=sessions,
        expected={symbol: sessions for symbol in _SYMBOLS},
        groups=tuple(groups),
        staleness_threshold=draw(st.integers(min_value=0, max_value=3)),
        failed_symbols=failed_symbols,
        retained_parent_coverage=retained_parent_coverage,
        requested_range=requested_range,
        comparison_range=DateRange(sessions[-1], sessions[-1]),
    )


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (TypeError, ValueError):
        return None


def _reference_row_facts(
    candidate: DailyBarCandidate,
    sessions: frozenset[date],
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Apply row rules independently of ValidationService implementation details."""

    reasons: list[str] = []
    offending: dict[str, object] = {}
    symbol = candidate.symbol.strip().upper()
    if not symbol:
        reasons.append("symbol.nonempty")
        offending["symbol"] = candidate.symbol

    session_valid = candidate.session in sessions
    if not session_valid:
        reasons.append("session.xnys")
        offending["session"] = candidate.session

    adjusted = {
        "open": candidate.adjusted_open,
        "high": candidate.adjusted_high,
        "low": candidate.adjusted_low,
        "close": candidate.adjusted_close,
    }
    numbers = {name: _decimal(value) for name, value in adjusted.items()}
    if not all(
        value is not None and value.is_finite() and value > Decimal("0")
        for value in numbers.values()
    ):
        reasons.append("ohlc.finite_positive")
        offending.update(adjusted)

    volume = _decimal(candidate.adjusted_volume)
    if volume is None or not volume.is_finite() or volume < Decimal("0"):
        reasons.append("volume.finite_nonnegative")
        offending["volume"] = candidate.adjusted_volume

    high = numbers["high"]
    low = numbers["low"]
    open_value = numbers["open"]
    close = numbers["close"]
    if (
        high is not None
        and high.is_finite()
        and low is not None
        and low.is_finite()
        and open_value is not None
        and open_value.is_finite()
        and close is not None
        and close.is_finite()
        and not (high >= open_value and high >= low and high >= close)
    ):
        reasons.append("high.envelope")
        offending.update(adjusted)
    if (
        low is not None
        and low.is_finite()
        and high is not None
        and high.is_finite()
        and open_value is not None
        and open_value.is_finite()
        and close is not None
        and close.is_finite()
        and not (low <= open_value and low <= high and low <= close)
    ):
        reasons.append("low.envelope")
        offending.update(adjusted)

    if candidate.policy_version != POLICY_VERSION:
        reasons.append("normalization.policy")
        offending.update(
            {
                "policy_version": candidate.policy_version,
                "expected_policy_version": POLICY_VERSION,
                "split_ratio": candidate.corporate_action.split_ratio,
                "dividend": candidate.corporate_action.dividend,
                "cumulative_price_factor": candidate.cumulative_price_factor,
                "cumulative_split_factor": candidate.cumulative_split_factor,
            }
        )
    if candidate.raw_lineage is None:
        reasons.append("lineage.raw_record")
        offending["raw_lineage"] = None
    return tuple(reasons), offending


def _candidate_projection(candidate: DailyBarCandidate) -> tuple[object, ...]:
    lineage = candidate.raw_lineage
    return (
        candidate.symbol,
        candidate.session,
        candidate.canonical_row_checksum,
        lineage.provider_record_checksum if lineage is not None else None,
    )


def _quarantine_projection(record: QuarantineRecord) -> tuple[object, ...]:
    """Project quarantine detail without depending on object identity."""

    lineage = record.raw_lineage
    return (
        record.source_kind.value,
        record.symbol,
        record.session,
        tuple(record.reason_codes),
        record.policy_version,
        record.candidate_checksum,
        lineage.provider_record_checksum if lineage is not None else None,
        canonical_json_text(record.offending_values),
    )


def _reference_quarantine(
    candidate: DailyBarCandidate,
    sessions: frozenset[date],
    *,
    conflict_checksums: tuple[str, ...] | None = None,
) -> tuple[object, ...]:
    reasons, offending = _reference_row_facts(candidate, sessions)
    if conflict_checksums is not None:
        reasons = (*reasons, "duplicate.conflict")
        offending = {
            **offending,
            "session_key": {
                "symbol": candidate.symbol,
                "session": candidate.session,
            },
            "canonical_row_checksums": list(conflict_checksums),
        }
    session_known = candidate.session in sessions
    lineage = candidate.raw_lineage
    return (
        "daily_bar_candidate",
        candidate.symbol if session_known else None,
        candidate.session if session_known else None,
        reasons,
        candidate.policy_version,
        candidate.canonical_row_checksum,
        lineage.provider_record_checksum if lineage is not None else None,
        canonical_json_text(offending),
    )


def _quarantine_sort_key(
    value: tuple[object, ...],
) -> tuple[str, str, str, str, str, str, str]:
    session = value[2]
    return (
        str(value[0]),
        str(value[1] or ""),
        session.isoformat() if isinstance(session, date) else "",
        str(value[3][0]),
        str(value[5] or ""),
        str(value[6] or ""),
        str(value[7]),
    )


def _reference_partition(case: ValidationCase) -> ReferencePartition:
    """Partition candidate groups using a small independent reference model."""

    sessions = frozenset(case.sessions)
    accepted_candidates: list[DailyBarCandidate] = []
    quarantined: list[tuple[object, ...]] = []
    duplicate_counts: list[tuple[str, date, int]] = []

    for group in case.groups:
        if not group.members:
            continue
        members = group.members
        checksums = tuple(sorted({member.canonical_row_checksum for member in members}))
        if len(checksums) > 1:
            quarantined.extend(
                _reference_quarantine(
                    member,
                    sessions,
                    conflict_checksums=checksums,
                )
                for member in members
            )
            continue

        member_reasons = [
            _reference_row_facts(member, sessions)[0] for member in members
        ]
        if all(not reasons for reasons in member_reasons):
            representative = min(
                members,
                key=lambda member: (
                    member.canonical_row_checksum,
                    member.symbol,
                    member.session,
                ),
            )
            accepted_candidates.append(representative)
            if len(members) > 1:
                duplicate_counts.append(
                    (representative.symbol, representative.session, len(members) - 1)
                )
        else:
            quarantined.extend(
                _reference_quarantine(member, sessions) for member in members
            )

    accepted_candidates.sort(key=DailyBarCandidate.sort_key)
    accepted = tuple(_candidate_projection(candidate) for candidate in accepted_candidates)
    quarantined = tuple(sorted(quarantined, key=_quarantine_sort_key))
    duplicate_counts_tuple = tuple(sorted(duplicate_counts))
    accepted_keys = {
        (candidate.symbol, candidate.session) for candidate in accepted_candidates
    }

    gaps: list[DataGap] = []
    for symbol, expected_sessions in case.expected.items():
        for session in expected_sessions:
            if (symbol, session) not in accepted_keys:
                gaps.append(
                    DataGap(
                        symbol=symbol,
                        expected_session=session,
                        requested_range=case.requested_range,
                        parent_retained=symbol in case.retained_parent_coverage,
                    )
                )
    gaps.sort(key=DataGap.sort_key)
    gaps_projection = tuple(
        (
            gap.symbol,
            gap.expected_session,
            gap.requested_range.start,
            gap.requested_range.end,
            gap.parent_retained,
            gap.reason,
        )
        for gap in gaps
    )

    quarantined_by_symbol = Counter(
        value[1] for value in quarantined if isinstance(value[1], str)
    )
    duplicate_by_symbol: Counter[str] = Counter()
    for symbol, _session, count in duplicate_counts_tuple:
        duplicate_by_symbol[symbol] += count
    symbols = set(case.expected)
    symbols.update(case.failed_symbols)
    symbols.update(case.retained_parent_coverage)
    summaries: list[SymbolValidationSummary] = []
    for symbol in sorted(symbols):
        expected_sessions = tuple(case.expected.get(symbol, ()))
        accepted_sessions = sorted(
            session
            for accepted_symbol, session in {
                (value[0], value[1]) for value in accepted
            }
            if accepted_symbol == symbol
        )
        covered_range = (
            DateRange(accepted_sessions[0], accepted_sessions[-1])
            if accepted_sessions
            else None
        )
        latest_accepted = accepted_sessions[-1] if accepted_sessions else None
        lag = (
            len(expected_sessions)
            if latest_accepted is None
            else sum(session > latest_accepted for session in expected_sessions)
        )
        stale = bool(expected_sessions) and lag > case.staleness_threshold
        symbol_gaps = [gap for gap in gaps if gap.symbol == symbol]
        comparison_ready = not (
            symbol == "SPY"
            and any(
                case.comparison_range.start
                <= gap.expected_session
                <= case.comparison_range.end
                for gap in symbol_gaps
            )
        )
        summaries.append(
            SymbolValidationSummary(
                symbol=symbol,
                accepted_count=len(accepted_sessions),
                quarantined_count=quarantined_by_symbol.get(symbol, 0),
                duplicate_count=duplicate_by_symbol.get(symbol, 0),
                gap_count=len(symbol_gaps),
                stale=stale,
                staleness_lag_sessions=lag if stale else 0,
                failed=symbol in case.failed_symbols,
                retained_parent_coverage=symbol in case.retained_parent_coverage,
                covered_range=covered_range,
                comparison_ready=comparison_ready,
            )
        )

    reason_counts: Counter[str] = Counter()
    for value in quarantined:
        reason_counts.update(value[3])
    report = ValidationReport(
        per_symbol=tuple(summaries),
        quarantined_by_reason=tuple(sorted(reason_counts.items())),
        gaps=tuple(gaps),
        calendar_version=FixtureCalendar.version,
    )
    return ReferencePartition(
        accepted=accepted,
        quarantined=quarantined,
        duplicate_counts=duplicate_counts_tuple,
        gaps=gaps_projection,
        report=report,
    )


# Feature: quantitative-research-platform, Property 5: Validation partitions candidates without fabrication
# Validates: Requirements 5.1–5.23, 17.5, 17.11–17.15
@settings(max_examples=100, deadline=None)
@given(case=validation_cases())
def test_validation_partitions_candidates_without_fabrication(
    case: ValidationCase,
) -> None:
    """Validation matches the independent partition reference exactly."""

    service = ValidationService(calendar=FixtureCalendar(case.sessions))
    kwargs = {
        "requested_range": case.requested_range,
        "comparison_range": case.comparison_range,
        "failed_symbols": case.failed_symbols,
        "retained_parent_coverage": case.retained_parent_coverage,
    }
    actual = service.validate(
        case.candidates,
        case.expected,
        case.staleness_threshold,
        **kwargs,
    )
    repeated = service.validate(
        tuple(reversed(case.candidates)),
        case.expected,
        case.staleness_threshold,
        **kwargs,
    )
    expected = _reference_partition(case)

    actual_accepted = tuple(_candidate_projection(row) for row in actual.accepted_rows)
    actual_quarantined = tuple(
        _quarantine_projection(row) for row in actual.quarantined_rows
    )
    actual_duplicates = tuple(
        (key.symbol, key.session, count) for key, count in actual.duplicate_counts
    )
    actual_gaps = tuple(
        (
            gap.symbol,
            gap.expected_session,
            gap.requested_range.start,
            gap.requested_range.end,
            gap.parent_retained,
            gap.reason,
        )
        for gap in actual.gaps
    )

    assert actual_accepted == expected.accepted
    assert actual_quarantined == expected.quarantined
    assert actual_duplicates == expected.duplicate_counts
    assert actual_gaps == expected.gaps
    assert actual.report.to_content_dict() == expected.report.to_content_dict()

    accepted_keys = tuple((row.symbol, row.session) for row in actual.accepted_rows)
    assert len(accepted_keys) == len(set(accepted_keys))
    assert all(row.raw_lineage is not None for row in actual.accepted_rows)
    input_keys = {(row.symbol, row.session) for row in case.candidates}
    assert set(accepted_keys) <= input_keys
    assert all(
        (gap.symbol, gap.expected_session) not in set(accepted_keys)
        for gap in actual.gaps
    )

    quarantined_by_checksum = {
        row.candidate_checksum: row for row in actual.quarantined_rows
    }
    for group in case.groups:
        if "conflict" not in group.kind:
            continue
        conflict_key = group.slot_key
        assert conflict_key not in set(accepted_keys)
        for member in group.members:
            record = quarantined_by_checksum[member.canonical_row_checksum]
            assert "duplicate.conflict" in record.reason_codes

    assert actual.report.content_checksum == repeated.report.content_checksum
    assert actual.report.to_content_dict() == repeated.report.to_content_dict()
    assert tuple(_candidate_projection(row) for row in repeated.accepted_rows) == (
        actual_accepted
    )
    assert tuple(_quarantine_projection(row) for row in repeated.quarantined_rows) == (
        actual_quarantined
    )
    assert actual.report.summary.stale_symbols == tuple(
        summary.symbol for summary in expected.report.per_symbol if summary.stale
    )
