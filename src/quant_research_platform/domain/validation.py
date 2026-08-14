"""Deterministic, streaming validation of normalized market-data candidates.

The normalizer deliberately emits candidates for rows that may still be
incomplete.  This module is the acceptance boundary: it applies the fixed row
rules, keeps every rejected value in an append-only quarantine record, resolves
same-key duplicates, and compares accepted keys with the completed XNYS
sessions requested by the caller.

The validation loop is written for an already sorted partition stream.  A
canonical ``Sequence`` input is sorted at the boundary for convenient local
callers and deterministic tests; streaming readers are consumed directly and
must be ordered by ``(symbol, session, canonical_row_checksum)`` as produced by
the canonical Parquet layer.  Only the current key group and aggregate facts
are held by the grouping phase.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Protocol, TypeAlias

from .canonical import canonical_json_text, sha256_canonical_json
from .errors import QuarantineReason, ValidationReason
from .market import (
    CorporateAction,
    DailyBarCandidate,
    DataGap,
    DateRange,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawLineage,
    SessionKey,
    SymbolValidationSummary,
    ValidationReport,
)
from .normalization import POLICY_VERSION

_CHECKSUM_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class SessionCalendar(Protocol):
    """The small calendar surface needed by validation's map phase."""

    name: str
    version: str

    def is_session(self, value: date) -> bool: ...


CandidateInput: TypeAlias = DailyBarCandidate | QuarantineRecord
ExpectedSessions: TypeAlias = Mapping[str, Sequence[date]]


@dataclass(frozen=True, slots=True)
class ValidationOutput:
    """Accepted rows, rejection details, coverage facts, and their report.

    Tuples make the result repeatable for callers that need to inspect it more
    than once.  ``iter_accepted`` and ``iter_quarantined`` provide iterator
    views for storage adapters, while the validation service itself consumes an
    input partition stream one key group at a time.
    """

    accepted_rows: tuple[DailyBarCandidate, ...]
    quarantined_rows: tuple[QuarantineRecord, ...]
    gaps: tuple[DataGap, ...]
    per_symbol: tuple[SymbolValidationSummary, ...]
    duplicate_counts: tuple[tuple[SessionKey, int], ...]
    report: ValidationReport

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_rows, tuple):
            raise TypeError("accepted_rows must be an immutable tuple")
        if not isinstance(self.quarantined_rows, tuple):
            raise TypeError("quarantined_rows must be an immutable tuple")
        if not isinstance(self.gaps, tuple):
            raise TypeError("gaps must be an immutable tuple")
        if not isinstance(self.per_symbol, tuple):
            raise TypeError("per_symbol must be an immutable tuple")
        if not isinstance(self.duplicate_counts, tuple):
            raise TypeError("duplicate_counts must be an immutable tuple")
        if any(not isinstance(row, DailyBarCandidate) for row in self.accepted_rows):
            raise TypeError("accepted_rows may contain only DailyBarCandidate values")
        if any(
            not isinstance(row, QuarantineRecord) for row in self.quarantined_rows
        ):
            raise TypeError(
                "quarantined_rows may contain only QuarantineRecord values"
            )
        if any(not isinstance(gap, DataGap) for gap in self.gaps):
            raise TypeError("gaps may contain only DataGap values")
        if any(
            not isinstance(summary, SymbolValidationSummary)
            for summary in self.per_symbol
        ):
            raise TypeError(
                "per_symbol may contain only SymbolValidationSummary values"
            )
        for key, count in self.duplicate_counts:
            if not isinstance(key, SessionKey):
                raise TypeError("duplicate_counts keys must be SessionKey values")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("duplicate_counts values must be positive integers")
        if not isinstance(self.report, ValidationReport):
            raise TypeError("report must be a ValidationReport")

        summary = self.report.summary
        if summary.accepted_row_count != len(self.accepted_rows):
            raise ValueError("report accepted count does not match accepted rows")
        # ValidationReport's existing per-symbol projection cannot attribute a
        # symbol.nonempty record without inventing a normalized symbol. Such a
        # detail remains in quarantined_rows and in quarantined_by_reason.
        if summary.collapsed_duplicate_count != sum(
            count for _, count in self.duplicate_counts
        ):
            raise ValueError("report duplicate count does not match duplicate details")
        if summary.gap_count != len(self.gaps):
            raise ValueError("report gap count does not match gap details")

        object.__setattr__(self, "accepted_rows", tuple(sorted(
            self.accepted_rows, key=DailyBarCandidate.sort_key
        )))
        object.__setattr__(self, "quarantined_rows", tuple(sorted(
            self.quarantined_rows, key=_quarantine_sort_key
        )))
        object.__setattr__(self, "gaps", tuple(sorted(self.gaps, key=DataGap.sort_key)))
        object.__setattr__(
            self,
            "per_symbol",
            tuple(sorted(self.per_symbol, key=SymbolValidationSummary.sort_key)),
        )
        object.__setattr__(
            self,
            "duplicate_counts",
            tuple(sorted(self.duplicate_counts, key=lambda item: item[0].sort_key())),
        )

    @property
    def accepted(self) -> tuple[DailyBarCandidate, ...]:
        """Compatibility alias for accepted normalized bars."""

        return self.accepted_rows

    @property
    def quarantined(self) -> tuple[QuarantineRecord, ...]:
        """Compatibility alias for quarantine detail rows."""

        return self.quarantined_rows

    @property
    def quarantine(self) -> tuple[QuarantineRecord, ...]:
        """Short alias used by storage and inspection adapters."""

        return self.quarantined_rows

    @property
    def accepted_bars(self) -> tuple[DailyBarCandidate, ...]:
        return self.accepted_rows

    @property
    def data_gaps(self) -> tuple[DataGap, ...]:
        return self.gaps

    @property
    def duplicate_count(self) -> int:
        return self.report.summary.collapsed_duplicate_count

    @property
    def comparison_ready(self) -> bool:
        return self.report.summary.comparison_ready

    def iter_accepted(self) -> Iterator[DailyBarCandidate]:
        """Iterate accepted rows without exposing mutable internal state."""

        return iter(self.accepted_rows)

    def iter_quarantined(self) -> Iterator[QuarantineRecord]:
        """Iterate quarantine rows without exposing mutable internal state."""

        return iter(self.quarantined_rows)


@dataclass(frozen=True, slots=True)
class _CandidateFacts:
    """Mapped and row/policy-checked facts for one input candidate."""

    value: object
    candidate: DailyBarCandidate | None
    symbol: str | None
    session: date | None
    raw_lineage: RawLineage | None
    policy_version: str | None
    checksum: str
    reasons: tuple[str, ...]
    offending_values: Mapping[str, object]
    valid: bool
    group_key: tuple[str, str]


class ValidationService:
    """Apply deterministic validation and partition candidates by outcome."""

    def __init__(
        self,
        calendar: SessionCalendar | None = None,
        *,
        expected_policy_version: str | None = POLICY_VERSION,
        benchmark_symbol: str | None = "SPY",
        calendar_version: str | None = None,
    ) -> None:
        if expected_policy_version is not None:
            if (
                not isinstance(expected_policy_version, str)
                or not expected_policy_version.strip()
            ):
                raise ValueError(
                    "expected_policy_version must be non-empty text or None"
                )
            expected_policy_version = expected_policy_version.strip()
        if benchmark_symbol is not None:
            benchmark_symbol = _normalize_symbol(benchmark_symbol)
            if benchmark_symbol is None:  # pragma: no cover - defensive guard
                raise ValueError("benchmark_symbol must be a valid symbol or None")
        if calendar_version is not None:
            if not isinstance(calendar_version, str) or not calendar_version.strip():
                raise ValueError("calendar_version must be non-empty text or None")
            calendar_version = calendar_version.strip()

        self.calendar = calendar
        self.expected_policy_version = expected_policy_version
        self.benchmark_symbol = benchmark_symbol
        self.calendar_version = calendar_version

    def validate(
        self,
        candidates: Iterable[CandidateInput],
        expected: ExpectedSessions,
        staleness_threshold: int,
        *,
        requested_range: DateRange | None = None,
        requested_ranges: Mapping[str, DateRange] | None = None,
        comparison_range: DateRange | None = None,
        evaluation_range: DateRange | None = None,
        benchmark_symbol: str | None = None,
        failed_symbols: Iterable[str] = (),
        retained_parent_coverage: Iterable[str] = (),
        calendar: SessionCalendar | None = None,
    ) -> ValidationOutput:
        """Validate a sorted candidate stream and return deterministic partitions.

        ``expected`` contains completed XNYS session labels for each requested
        symbol.  A sequence input is canonical-sorted at the boundary.  Other
        iterables are treated as partition streams and must already be sorted
        by ``(symbol, session, canonical_row_checksum)``; this avoids
        materializing a complete partition when reading from Parquet.
        """

        _require_non_negative_int("staleness_threshold", staleness_threshold)
        expected_by_symbol = _normalize_expected(expected)
        normalized_requested_ranges = _normalize_ranges(requested_ranges)
        if requested_range is not None and not isinstance(requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange or None")
        if comparison_range is not None and not isinstance(comparison_range, DateRange):
            raise TypeError("comparison_range must be a DateRange or None")
        if evaluation_range is not None and not isinstance(evaluation_range, DateRange):
            raise TypeError("evaluation_range must be a DateRange or None")
        if (
            comparison_range is not None
            and evaluation_range is not None
            and comparison_range != evaluation_range
        ):
            raise ValueError(
                "comparison_range and evaluation_range must identify the same range"
            )
        effective_comparison_range = comparison_range or evaluation_range

        effective_calendar = calendar or self.calendar
        effective_benchmark = (
            self.benchmark_symbol
            if benchmark_symbol is None
            else _normalize_symbol(benchmark_symbol)
        )
        if benchmark_symbol is not None and effective_benchmark is None:
            raise ValueError("benchmark_symbol must be a valid symbol or None")

        failed = _normalize_symbol_set(failed_symbols, "failed_symbols")
        retained = _normalize_symbol_set(
            retained_parent_coverage, "retained_parent_coverage"
        )

        accepted_rows: list[DailyBarCandidate] = []
        quarantine_rows: list[QuarantineRecord] = []
        reason_counts: defaultdict[str, int] = defaultdict(int)
        quarantined_by_symbol: defaultdict[str, int] = defaultdict(int)
        accepted_by_symbol: defaultdict[str, list[date]] = defaultdict(list)
        duplicate_by_key: dict[SessionKey, int] = {}

        def record_quarantine(record: QuarantineRecord) -> None:
            quarantine_rows.append(record)
            for reason in record.reason_codes:
                reason_counts[reason] += 1
            symbol = _quarantine_symbol(record)
            if symbol is not None:
                quarantined_by_symbol[symbol] += 1

        current_key: tuple[str, str] | None = None
        current_group: list[_CandidateFacts] = []

        def flush_group() -> None:
            nonlocal current_group
            if not current_group:
                return
            checksums = sorted({facts.checksum for facts in current_group})
            if len(checksums) > 1:
                conflict_values = {
                    "session_key": {
                        "symbol": current_group[0].symbol,
                        "session": current_group[0].session,
                    },
                    "canonical_row_checksums": checksums,
                }
                for facts in current_group:
                    record_quarantine(
                        _facts_quarantine(
                            facts,
                            extra_reason=QuarantineReason.DUPLICATE_CONFLICT.value,
                            extra_values=conflict_values,
                        )
                    )
            elif (
                len(checksums) == 1
                and len(current_group) >= 1
                and all(
                    facts.valid and facts.candidate is not None
                    for facts in current_group
                )
            ):
                # Equal checksums are equal canonical rows.  ``min`` keeps the
                # representative deterministic even if a caller supplies two
                # equivalent instances with different object identities.
                representative = min(
                    current_group,
                    key=lambda facts: _facts_tiebreak_key(facts),
                )
                assert representative.candidate is not None
                accepted_rows.append(representative.candidate)
                assert representative.symbol is not None
                assert representative.session is not None
                accepted_by_symbol[representative.symbol].append(
                    representative.session
                )
                key = SessionKey(representative.symbol, representative.session)
                if len(current_group) > 1:
                    duplicate_by_key[key] = len(current_group) - 1
            else:
                for facts in current_group:
                    record_quarantine(_facts_quarantine(facts))
            current_group = []

        source: Iterable[CandidateInput]
        if isinstance(candidates, Sequence) and not isinstance(
            candidates, (str, bytes, bytearray)
        ):
            source = sorted(candidates, key=_input_sort_key)
        else:
            source = candidates

        for item in source:
            if isinstance(item, QuarantineRecord):
                # Normalization policy/non-session decisions are already
                # canonical rejection facts; carry them into this report. They
                # do not interrupt a candidate key group in a streaming input.
                record_quarantine(item)
                continue

            facts = _candidate_facts(
                item,
                expected_by_symbol=expected_by_symbol,
                calendar=effective_calendar,
                expected_policy_version=self.expected_policy_version,
            )
            if current_key is None:
                current_key = facts.group_key
            elif facts.group_key != current_key:
                if facts.group_key < current_key:
                    raise ValueError(
                        "streaming validation input must be sorted by symbol/session"
                    )
                flush_group()
                current_key = facts.group_key
            current_group.append(facts)
        flush_group()

        # Normalize and sort the accepted/rejected streams after processing so
        # report and artifact bytes do not depend on input order.
        accepted_rows.sort(key=DailyBarCandidate.sort_key)
        quarantine_rows.sort(key=_quarantine_sort_key)

        symbols = set(expected_by_symbol)
        symbols.update(accepted_by_symbol)
        symbols.update(quarantined_by_symbol)
        symbols.update(failed)
        symbols.update(retained)

        accepted_keys = {
            (row.symbol, row.session)
            for row in accepted_rows
        }
        gaps: list[DataGap] = []
        summaries: list[SymbolValidationSummary] = []
        normalized_requested_range = requested_range

        benchmark_requested = (
            effective_benchmark is not None
            and (
                effective_benchmark in expected_by_symbol
                or effective_comparison_range is not None
            )
        )
        if benchmark_requested:
            assert effective_benchmark is not None
            if effective_comparison_range is None:
                effective_comparison_range = _range_for_symbol(
                    effective_benchmark,
                    expected_by_symbol,
                    normalized_requested_range,
                    normalized_requested_ranges,
                )

        for symbol in sorted(symbols):
            expected_sessions = expected_by_symbol.get(symbol, ())
            symbol_gaps: list[DataGap] = []
            gap_range = _range_for_symbol(
                symbol,
                expected_by_symbol,
                normalized_requested_range,
                normalized_requested_ranges,
            )
            if expected_sessions and gap_range is None:
                # ``expected_sessions`` is non-empty, so this branch can only
                # be reached if a caller bypasses DateRange construction.
                gap_range = DateRange(expected_sessions[0], expected_sessions[-1])
            if gap_range is not None:
                for session in expected_sessions:
                    if (symbol, session) not in accepted_keys:
                        gap = DataGap(
                            symbol=symbol,
                            expected_session=session,
                            requested_range=gap_range,
                            parent_retained=symbol in retained,
                        )
                        gaps.append(gap)
                        symbol_gaps.append(gap)

            accepted_sessions = sorted(set(accepted_by_symbol.get(symbol, ())))
            covered_range = (
                DateRange(accepted_sessions[0], accepted_sessions[-1])
                if accepted_sessions
                else None
            )
            latest_expected = expected_sessions[-1] if expected_sessions else None
            latest_accepted = accepted_sessions[-1] if accepted_sessions else None
            if latest_expected is None:
                lag = 0
            elif latest_accepted is None:
                lag = len(expected_sessions)
            else:
                lag = sum(session > latest_accepted for session in expected_sessions)
            stale = bool(expected_sessions) and lag > staleness_threshold

            comparison_ready = True
            if (
                benchmark_requested
                and effective_benchmark is not None
                and symbol == effective_benchmark
            ):
                comparison_ready = (
                    symbol in expected_by_symbol
                    and not any(
                        _date_in_range(
                            gap.expected_session, effective_comparison_range
                        )
                        for gap in symbol_gaps
                    )
                )

            summaries.append(
                SymbolValidationSummary(
                    symbol=symbol,
                    accepted_count=len(accepted_sessions),
                    quarantined_count=quarantined_by_symbol.get(symbol, 0),
                    duplicate_count=sum(
                        count
                        for key, count in duplicate_by_key.items()
                        if key.symbol == symbol
                    ),
                    gap_count=len(symbol_gaps),
                    stale=stale,
                    staleness_lag_sessions=lag if stale else 0,
                    failed=symbol in failed,
                    retained_parent_coverage=symbol in retained,
                    covered_range=covered_range,
                    comparison_ready=comparison_ready,
                )
            )

        gaps.sort(key=DataGap.sort_key)
        report = ValidationReport(
            per_symbol=tuple(summaries),
            quarantined_by_reason=tuple(sorted(reason_counts.items())),
            gaps=tuple(gaps),
            calendar_version=_calendar_version(
                effective_calendar, self.calendar_version
            ),
        )
        duplicate_counts = tuple(
            sorted(duplicate_by_key.items(), key=lambda item: item[0].sort_key())
        )
        return ValidationOutput(
            accepted_rows=tuple(accepted_rows),
            quarantined_rows=tuple(quarantine_rows),
            gaps=tuple(gaps),
            per_symbol=report.per_symbol,
            duplicate_counts=duplicate_counts,
            report=report,
        )


# A concise alias is useful to callers that reserve ``ValidationService`` for
# an application port while keeping this concrete domain implementation.
Validator = ValidationService


def validate(
    candidates: Iterable[CandidateInput],
    expected: ExpectedSessions,
    staleness_threshold: int,
    **kwargs: Any,
) -> ValidationOutput:
    """Functional convenience wrapper around :class:`ValidationService`."""

    return ValidationService().validate(
        candidates,
        expected,
        staleness_threshold,
        **kwargs,
    )


def _candidate_facts(
    value: object,
    *,
    expected_by_symbol: Mapping[str, Sequence[date]],
    calendar: SessionCalendar | None,
    expected_policy_version: str | None,
) -> _CandidateFacts:
    if not isinstance(value, DailyBarCandidate):
        checksum = _fallback_checksum(value)
        return _CandidateFacts(
            value=value,
            candidate=None,
            symbol=None,
            session=None,
            raw_lineage=None,
            policy_version=None,
            checksum=checksum,
            reasons=(QuarantineReason.VALIDATION_ROW.value,),
            offending_values={"candidate_type": type(value).__name__},
            valid=False,
            group_key=("", ""),
        )

    candidate = value
    raw_symbol = getattr(candidate, "symbol", None)
    symbol = _normalize_symbol(raw_symbol)
    raw_session = getattr(candidate, "session", None)
    session = _as_date(raw_session)
    reasons: list[str] = []
    offending: dict[str, object] = {}

    if symbol is None:
        reasons.append(ValidationReason.SYMBOL_NONEMPTY.value)
        offending["symbol"] = _safe_value(raw_symbol)

    session_valid = session is not None
    if session is not None:
        if calendar is not None:
            try:
                session_valid = bool(calendar.is_session(session))
            except (IndexError, TypeError, ValueError):
                session_valid = False
        elif symbol is not None and symbol in expected_by_symbol:
            # The expected input is explicitly the completed XNYS session set.
            # This lets a calendar-free domain test still exercise map rules.
            session_valid = session in expected_by_symbol[symbol]
    if not session_valid:
        reasons.append(ValidationReason.SESSION_XNYS.value)
        offending["session"] = _safe_value(raw_session)

    adjusted_values = {
        "open": getattr(candidate, "adjusted_open", None),
        "high": getattr(candidate, "adjusted_high", None),
        "low": getattr(candidate, "adjusted_low", None),
        "close": getattr(candidate, "adjusted_close", None),
    }
    numeric_values = {
        name: _as_decimal(value) for name, value in adjusted_values.items()
    }
    if not all(
        decimal_value is not None
        and decimal_value.is_finite()
        and decimal_value > Decimal("0")
        for decimal_value in numeric_values.values()
    ):
        reasons.append(ValidationReason.OHLC_FINITE_POSITIVE.value)
        offending.update(
            {name: _safe_value(value) for name, value in adjusted_values.items()}
        )

    raw_volume = getattr(candidate, "adjusted_volume", None)
    volume = _as_decimal(raw_volume)
    if volume is None or not volume.is_finite() or volume < Decimal("0"):
        reasons.append(ValidationReason.VOLUME_FINITE_NONNEGATIVE.value)
        offending["volume"] = _safe_value(raw_volume)

    high = numeric_values["high"]
    low = numeric_values["low"]
    open_value = numeric_values["open"]
    close = numeric_values["close"]
    if (
        high is not None
        and high.is_finite()
        and open_value is not None
        and open_value.is_finite()
        and low is not None
        and low.is_finite()
        and close is not None
        and close.is_finite()
        and not (high >= open_value and high >= low and high >= close)
    ):
        reasons.append(ValidationReason.HIGH_ENVELOPE.value)
        offending.update(
            {
                "high": _safe_value(adjusted_values["high"]),
                "open": _safe_value(adjusted_values["open"]),
                "low": _safe_value(adjusted_values["low"]),
                "close": _safe_value(adjusted_values["close"]),
            }
        )
    if (
        low is not None
        and low.is_finite()
        and open_value is not None
        and open_value.is_finite()
        and high is not None
        and high.is_finite()
        and close is not None
        and close.is_finite()
        and not (low <= open_value and low <= high and low <= close)
    ):
        reasons.append(ValidationReason.LOW_ENVELOPE.value)
        offending.update(
            {
                "low": _safe_value(adjusted_values["low"]),
                "open": _safe_value(adjusted_values["open"]),
                "high": _safe_value(adjusted_values["high"]),
                "close": _safe_value(adjusted_values["close"]),
            }
        )

    raw_lineage_value = getattr(candidate, "raw_lineage", None)
    raw_lineage = (
        raw_lineage_value if isinstance(raw_lineage_value, RawLineage) else None
    )
    policy_version_value = getattr(candidate, "policy_version", None)
    policy_version = (
        policy_version_value if isinstance(policy_version_value, str) else None
    )
    policy_problem = False
    if (
        expected_policy_version is not None
        and policy_version != expected_policy_version
    ):
        policy_problem = True
    raw_action = getattr(candidate, "raw_action", None)
    corporate_action = getattr(candidate, "corporate_action", None)
    if not isinstance(raw_action, RawCorporateAction) or not isinstance(
        corporate_action, CorporateAction
    ):
        policy_problem = True
    else:
        raw_split = _as_decimal(raw_action.split_ratio)
        if raw_split is not None and (
            not raw_split.is_finite() or raw_split <= Decimal("0")
        ):
            policy_problem = True
        action_split = _as_decimal(corporate_action.split_ratio)
        if (
            action_split is None
            or not action_split.is_finite()
            or action_split <= Decimal("0")
        ):
            policy_problem = True
        action_dividend = _as_decimal(corporate_action.dividend)
        if action_dividend is None or not action_dividend.is_finite():
            policy_problem = True
        for factor_name in (
            "cumulative_price_factor",
            "cumulative_split_factor",
        ):
            factor = _as_decimal(getattr(candidate, factor_name, None))
            if factor is None or not factor.is_finite() or factor <= Decimal("0"):
                policy_problem = True
        if symbol is not None and session is not None:
            try:
                if corporate_action.session_key != SessionKey(symbol, session):
                    policy_problem = True
            except (TypeError, ValueError):
                policy_problem = True
        if raw_lineage is not None and corporate_action.raw_lineage not in (
            None,
            raw_lineage,
        ):
            policy_problem = True
    if policy_problem:
        reasons.append(QuarantineReason.NORMALIZATION_POLICY.value)
        offending.update(
            {
                "policy_version": _safe_value(policy_version_value),
                "expected_policy_version": expected_policy_version,
                "split_ratio": _safe_value(
                    getattr(corporate_action, "split_ratio", None)
                ),
                "dividend": _safe_value(getattr(corporate_action, "dividend", None)),
                "cumulative_price_factor": _safe_value(
                    getattr(candidate, "cumulative_price_factor", None)
                ),
                "cumulative_split_factor": _safe_value(
                    getattr(candidate, "cumulative_split_factor", None)
                ),
            }
        )

    if raw_lineage is None:
        reasons.append(ValidationReason.RAW_LINEAGE.value)
        offending["raw_lineage"] = _safe_value(raw_lineage_value)

    checksum = _safe_candidate_checksum(candidate)
    group_key = (
        _sort_text(symbol if symbol is not None else raw_symbol),
        _sort_text(session.isoformat() if session is not None else raw_session),
    )
    return _CandidateFacts(
        value=value,
        candidate=candidate,
        symbol=symbol,
        session=session if session_valid else None,
        raw_lineage=raw_lineage,
        policy_version=policy_version,
        checksum=checksum,
        reasons=tuple(reasons),
        offending_values=offending,
        valid=not reasons and symbol is not None and session_valid,
        group_key=group_key,
    )


def _facts_quarantine(
    facts: _CandidateFacts,
    *,
    extra_reason: str | None = None,
    extra_values: Mapping[str, object] | None = None,
) -> QuarantineRecord:
    reasons = list(facts.reasons)
    if extra_reason is not None and extra_reason not in reasons:
        reasons.append(extra_reason)
    if not reasons:
        reasons.append(QuarantineReason.VALIDATION_ROW.value)
    values = dict(facts.offending_values)
    if extra_values is not None:
        values.update({key: _safe_value(value) for key, value in extra_values.items()})
    known_key = facts.symbol is not None and facts.session is not None
    return QuarantineRecord(
        source_kind=QuarantineSourceKind.DAILY_BAR_CANDIDATE,
        reason_codes=tuple(reasons),
        offending_values=values,
        policy_version=facts.policy_version,
        symbol=facts.symbol if known_key else None,
        session=facts.session if known_key else None,
        raw_lineage=facts.raw_lineage,
        candidate_checksum=facts.checksum,
    )


def _facts_tiebreak_key(facts: _CandidateFacts) -> tuple[str, str, str]:
    return (
        facts.checksum,
        facts.symbol or "",
        facts.session.isoformat() if facts.session is not None else "",
    )


def _input_sort_key(value: CandidateInput) -> tuple[int, str, str, str, str]:
    if isinstance(value, QuarantineRecord):
        content = canonical_json_text(value.to_content_dict())
        sort_key = value.sort_key()
        return (1, *sort_key[:3], sort_key[3] + content)
    return (0, *_candidate_sort_key(value))


def _candidate_sort_key(value: DailyBarCandidate) -> tuple[str, str, str, str]:
    symbol = _normalize_symbol(getattr(value, "symbol", None)) or _sort_text(
        getattr(value, "symbol", None)
    )
    session = _as_date(getattr(value, "session", None))
    session_text = session.isoformat() if session is not None else _sort_text(
        getattr(value, "session", None)
    )
    return (
        symbol,
        session_text,
        _safe_candidate_checksum(value),
        canonical_json_text(_candidate_sort_content(value)),
    )


def _quarantine_sort_key(value: QuarantineRecord) -> tuple[str, ...]:
    return (*value.sort_key(), canonical_json_text(value.to_content_dict()))


def _candidate_sort_content(value: object) -> dict[str, object]:
    if isinstance(value, DailyBarCandidate):
        try:
            content = value.to_content_dict()
            if isinstance(content, dict):
                # A deliberately malformed candidate may contain a non-finite
                # Decimal. Validate that the content is sortable before
                # returning it; otherwise use the sanitized projection below.
                canonical_json_text(content)
                return content
        except (AttributeError, TypeError, ValueError):
            pass
    return {
        "candidate_type": type(value).__name__,
        "symbol": _safe_value(getattr(value, "symbol", None)),
        "session": _safe_value(getattr(value, "session", None)),
        "adjusted_open": _safe_value(getattr(value, "adjusted_open", None)),
        "adjusted_high": _safe_value(getattr(value, "adjusted_high", None)),
        "adjusted_low": _safe_value(getattr(value, "adjusted_low", None)),
        "adjusted_close": _safe_value(getattr(value, "adjusted_close", None)),
        "adjusted_volume": _safe_value(getattr(value, "adjusted_volume", None)),
        "raw_lineage": _safe_value(getattr(value, "raw_lineage", None)),
    }


def _safe_candidate_checksum(value: object) -> str:
    if isinstance(value, DailyBarCandidate):
        try:
            candidate_checksum = value.canonical_row_checksum
            if _CHECKSUM_RE.fullmatch(candidate_checksum):
                return candidate_checksum
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        return sha256_canonical_json(_candidate_sort_content(value))
    except (TypeError, ValueError):
        return sha256_canonical_json(
            {
                "candidate_type": type(value).__name__,
                "symbol": _sort_text(getattr(value, "symbol", None)),
                "session": _sort_text(getattr(value, "session", None)),
            }
        )


def _fallback_checksum(value: object) -> str:
    try:
        return sha256_canonical_json(
            {
                "candidate_type": type(value).__name__,
                "value": _safe_value(value),
            }
        )
    except (TypeError, ValueError):
        return sha256_canonical_json({"candidate_type": type(value).__name__})


def _normalize_expected(expected: ExpectedSessions) -> dict[str, tuple[date, ...]]:
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a symbol-to-session mapping")
    normalized: dict[str, tuple[date, ...]] = {}
    for raw_symbol, raw_sessions in expected.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol is None:
            raise ValueError("expected symbols must be non-empty normalized symbols")
        if isinstance(raw_sessions, (str, bytes, bytearray)):
            raise TypeError("expected sessions must be a sequence of dates")
        sessions = tuple(raw_sessions)
        normalized_sessions: list[date] = []
        for session in sessions:
            normalized_session = _as_date(session)
            if normalized_session is None:
                raise TypeError("expected sessions must contain calendar dates")
            normalized_sessions.append(normalized_session)
        normalized[symbol] = tuple(sorted(set(normalized_sessions)))
    return normalized


def _normalize_ranges(
    ranges: Mapping[str, DateRange] | None,
) -> dict[str, DateRange]:
    if ranges is None:
        return {}
    if not isinstance(ranges, Mapping):
        raise TypeError("requested_ranges must be a symbol-to-DateRange mapping")
    normalized: dict[str, DateRange] = {}
    for raw_symbol, value in ranges.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol is None:
            raise ValueError("requested_ranges symbols must be valid symbols")
        if not isinstance(value, DateRange):
            raise TypeError("requested_ranges values must be DateRange values")
        normalized[symbol] = value
    return normalized


def _normalize_symbol_set(values: Iterable[str], field_name: str) -> frozenset[str]:
    normalized: set[str] = set()
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol is None:
            raise ValueError(f"{field_name} must contain only non-empty symbols")
        normalized.add(symbol)
    return frozenset(normalized)


def _range_for_symbol(
    symbol: str,
    expected: Mapping[str, Sequence[date]],
    requested_range: DateRange | None,
    requested_ranges: Mapping[str, DateRange],
) -> DateRange | None:
    explicit = requested_ranges.get(symbol)
    if explicit is not None:
        return explicit
    if requested_range is not None:
        return requested_range
    sessions = expected.get(symbol, ())
    if not sessions:
        return None
    return DateRange(sessions[0], sessions[-1])


def _date_in_range(value: date, range_: DateRange | None) -> bool:
    return range_ is not None and range_.start <= value <= range_.end


def _calendar_version(
    calendar: SessionCalendar | None,
    configured: str | None,
) -> str:
    if configured is not None:
        return configured
    if calendar is not None:
        value = getattr(calendar, "version", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "xnys"


def _quarantine_symbol(record: QuarantineRecord) -> str | None:
    if record.symbol is not None:
        return record.symbol
    value = record.offending_values.get("symbol")
    return _normalize_symbol(value) if isinstance(value, str) else None


def _normalize_symbol(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    if re.fullmatch(r"[A-Z0-9.\-]+", normalized) is None:
        return None
    return normalized


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime) or not isinstance(value, date):
        return None
    return value


def _as_decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str, date, datetime)):
        return value
    if isinstance(value, Decimal):
        return value if value.is_finite() else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    try:
        canonical_json_text(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _sort_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def _require_non_negative_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


__all__ = [
    "CandidateInput",
    "ExpectedSessions",
    "SessionCalendar",
    "ValidationOutput",
    "ValidationService",
    "Validator",
    "validate",
]
