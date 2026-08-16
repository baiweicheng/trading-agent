# ruff: noqa: E501
"""Property tests for causal normalization determinism and confluence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import chain
from typing import TypeAlias

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    ProviderRecord,
    ProviderRequest,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawDailyBar,
)
from quant_research_platform.domain.normalization import (
    POLICY_VERSION,
    CausalForwardAdjustmentV1,
    Normalizer,
)

_VALUE: TypeAlias = DailyBarCandidate | QuarantineRecord

_SYMBOLS = ("AAPL", "MSFT", "SPY")
_VALID_DATE_POOL = (
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
    date(2024, 1, 12),
)
_INVALID_DATE_POOL = (
    date(2024, 1, 1),
    date(2024, 1, 6),
    date(2024, 1, 7),
    date(2024, 1, 13),
)
_LATER_SESSION = date(2024, 1, 16)
_FACTOR_QUANTUM = Decimal("0.000000000000000001")


class FixtureXNYSCalendar:
    """Small deterministic calendar that exercises the normalizer boundary."""

    name = "XNYS"
    version = "property-fixture-v1"

    def __init__(self, sessions: Iterable[date]) -> None:
        self._sessions = frozenset(sessions)

    def is_session(self, value: date) -> bool:
        return value in self._sessions

    def close_timestamp(self, session: date) -> datetime:
        if not self.is_session(session):
            raise ValueError("the fixture calendar has no close for this date")
        return datetime.combine(session, datetime.min.time(), tzinfo=UTC).replace(
            hour=21
        )


@dataclass(frozen=True)
class NormalizationCase:
    """Generated history plus an independently regrouped provider stream."""

    records: tuple[ProviderRecord, ...]
    symbols: tuple[str, ...]
    valid_sessions: tuple[date, ...]
    invalid_sessions: tuple[date, ...]
    absent_checksums: tuple[str, ...]
    permuted_records: tuple[ProviderRecord, ...]
    regrouped_batches: tuple[tuple[ProviderRecord, ...], ...]

    @property
    def calendar(self) -> FixtureXNYSCalendar:
        return FixtureXNYSCalendar(self.valid_sessions)


@dataclass
class ReferenceState:
    """State carried by the independent Decimal reference implementation."""

    prior_raw_close: Decimal | None = None
    cumulative_price_factor: Decimal = Decimal("1")
    cumulative_split_factor: Decimal = Decimal("1")


def _draw_bar(
    draw: st.DrawFn,
    session: date,
    *,
    absent: bool,
) -> RawDailyBar:
    if absent:
        return RawDailyBar(provider_date=session)

    base = draw(st.integers(min_value=20, max_value=250))
    low = base - draw(st.integers(min_value=0, max_value=5))
    high = base + draw(st.integers(min_value=0, max_value=5))
    open_ = draw(st.integers(min_value=low, max_value=high))
    close = draw(st.integers(min_value=low, max_value=high))
    volume = draw(st.integers(min_value=1, max_value=50_000))
    return RawDailyBar(
        provider_date=session,
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        adj_close=Decimal(close) + Decimal("0.125"),
        volume=Decimal(volume),
    )


def _draw_action(
    draw: st.DrawFn,
    *,
    prior_raw_close: Decimal | None,
) -> RawCorporateAction:
    """Draw only finite actions whose policy equation is well-defined."""

    split_ratio = draw(
        st.sampled_from(
            (None, Decimal("0.5"), Decimal("1"), Decimal("1.5"), Decimal("2"))
        )
    )
    if prior_raw_close is None:
        # A first observation cannot carry a dividend because the policy needs
        # the preceding raw close for its causal reference price.
        dividend = draw(st.sampled_from((None, Decimal("0"))))
    else:
        # Generated closes are at least 15 and split ratios are at most 2, so
        # every selected dividend remains below prior_close / split_ratio.
        dividend = draw(
            st.sampled_from((None, Decimal("0"), Decimal("0.25"), Decimal("1")))
        )
    return RawCorporateAction(
        dividend=dividend,
        split_ratio=split_ratio,
        provider_fields={"source": "hypothesis-property-04"},
    )


def _record(
    request: ProviderRequest,
    symbol: str,
    raw_bar: RawDailyBar,
    raw_action: RawCorporateAction,
    ordinal: int,
) -> ProviderRecord:
    return ProviderRecord(
        provider="fixture-provider",
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=raw_bar,
        raw_action=raw_action,
        provider_fields={"fixture": "property-04", "ordinal": ordinal},
    )


@st.composite
def normalization_cases(draw: st.DrawFn) -> NormalizationCase:
    """Generate finite multi-symbol histories and valid input regroupings."""

    symbols = tuple(
        draw(
            st.lists(
                st.sampled_from(_SYMBOLS),
                min_size=1,
                max_size=3,
                unique=True,
            )
        )
    )
    valid_sessions = tuple(
        sorted(
            draw(
                st.lists(
                    st.sampled_from(_VALID_DATE_POOL),
                    min_size=2,
                    max_size=7,
                    unique=True,
                )
            )
        )
    )
    invalid_sessions = tuple(
        sorted(
            draw(
                st.lists(
                    st.sampled_from(_INVALID_DATE_POOL),
                    min_size=1,
                    max_size=2,
                    unique=True,
                )
            )
        )
    )
    absent_index = draw(st.integers(min_value=1, max_value=len(valid_sessions) - 1))
    request = ProviderRequest(
        symbols=symbols,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        provider="fixture-provider",
    )

    records: list[ProviderRecord] = []
    absent_checksums: list[str] = []
    ordinal = 0
    for symbol in symbols:
        prior_raw_close: Decimal | None = None
        for index, session in enumerate(valid_sessions):
            is_absent = index == absent_index
            raw_bar = _draw_bar(draw, session, absent=is_absent)
            raw_action = _draw_action(draw, prior_raw_close=prior_raw_close)
            record = _record(request, symbol, raw_bar, raw_action, ordinal)
            records.append(record)
            ordinal += 1
            if is_absent:
                absent_checksums.append(record.provider_record_checksum)
            elif raw_bar.close is not None:
                prior_raw_close = raw_bar.close

        # Invalid labels are deliberately outside the fixture's XNYS set and
        # are otherwise finite provider records, so only session mapping differs.
        for invalid_session in invalid_sessions:
            record = _record(
                request,
                symbol,
                _draw_bar(draw, invalid_session, absent=False),
                RawCorporateAction(
                    provider_fields={"source": "hypothesis-property-04"}
                ),
                ordinal,
            )
            records.append(record)
            ordinal += 1

    canonical_records = tuple(records)
    permuted_records = tuple(draw(st.permutations(canonical_records)))

    # Draw a sequence of non-empty batches, each no larger than the provider's
    # maximum batch size.  Flattening them is the normalizer's batch-boundary
    # seam and preserves the independently drawn permutation exactly.
    batches: list[tuple[ProviderRecord, ...]] = []
    offset = 0
    while offset < len(permuted_records):
        remaining = len(permuted_records) - offset
        batch_size = draw(st.integers(min_value=1, max_value=min(10, remaining)))
        batches.append(permuted_records[offset : offset + batch_size])
        offset += batch_size

    return NormalizationCase(
        records=canonical_records,
        symbols=symbols,
        valid_sessions=valid_sessions,
        invalid_sessions=invalid_sessions,
        absent_checksums=tuple(absent_checksums),
        permuted_records=permuted_records,
        regrouped_batches=tuple(batches),
    )


def _quantize_factor(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        return value.quantize(_FACTOR_QUANTUM)


def _multiply(value: Decimal | None, factor: Decimal) -> Decimal | None:
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        return value * factor


def _divide(value: Decimal | None, divisor: Decimal) -> Decimal | None:
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        return value / divisor


def _reference_outputs(
    records: Iterable[ProviderRecord], calendar: FixtureXNYSCalendar
) -> list[_VALUE]:
    """Compute expected candidates without calling the production policy."""

    mapped = sorted(
        records,
        key=lambda record: (
            record.symbol,
            record.provider_date,
            record.provider_record_checksum,
        ),
    )
    states: dict[str, ReferenceState] = {}
    outputs: list[_VALUE] = []

    for record in mapped:
        if not calendar.is_session(record.provider_date):
            outputs.append(
                QuarantineRecord(
                    source_kind=QuarantineSourceKind.PROVIDER_RECORD,
                    reason_codes=("session.non_xnys",),
                    offending_values={
                        "provider_date": record.provider_date,
                        "symbol": record.symbol,
                        "provider_record_checksum": record.provider_record_checksum,
                    },
                    raw_lineage=record.raw_lineage,
                )
            )
            continue

        state = states.setdefault(record.symbol, ReferenceState())
        raw_action = record.raw_action
        split_ratio = (
            Decimal("1") if raw_action.split_ratio is None else raw_action.split_ratio
        )
        dividend = Decimal("0") if raw_action.dividend is None else raw_action.dividend
        assert split_ratio is not None
        assert dividend is not None

        with localcontext() as context:
            context.prec = 28
            context.rounding = ROUND_HALF_EVEN
            reference_price = (
                None
                if dividend == 0
                else state.prior_raw_close / split_ratio
                if state.prior_raw_close is not None
                else None
            )
            dividend_factor = Decimal("1")
            if dividend != 0:
                assert reference_price is not None
                denominator = reference_price - dividend
                assert denominator > 0
                dividend_factor = reference_price / denominator
            next_split_factor = state.cumulative_split_factor * split_ratio
            next_price_factor = (
                state.cumulative_price_factor * split_ratio * dividend_factor
            )

        raw = record.raw_bar
        if not all(
            value is None
            for value in (raw.open, raw.high, raw.low, raw.close, raw.volume)
        ):
            action = CorporateAction(
                symbol=record.symbol,
                session=record.provider_date,
                dividend=dividend,
                split_ratio=split_ratio,
                raw_lineage=record.raw_lineage,
                source_fields=tuple(
                    field
                    for field, value in (
                        ("Dividends", raw_action.dividend),
                        ("Stock Splits", raw_action.split_ratio),
                    )
                    if value is not None
                ),
            )
            outputs.append(
                DailyBarCandidate(
                    symbol=record.symbol,
                    session=record.provider_date,
                    event_timestamp=calendar.close_timestamp(record.provider_date),
                    raw_bar=raw,
                    raw_action=raw_action,
                    corporate_action=action,
                    adjusted_open=_multiply(raw.open, next_price_factor),
                    adjusted_high=_multiply(raw.high, next_price_factor),
                    adjusted_low=_multiply(raw.low, next_price_factor),
                    adjusted_close=_multiply(raw.close, next_price_factor),
                    adjusted_volume=_divide(raw.volume, next_split_factor),
                    execution_adjusted_open=raw.open,
                    sizing_adjusted_close=raw.close,
                    cumulative_price_factor=_quantize_factor(next_price_factor),
                    cumulative_split_factor=_quantize_factor(next_split_factor),
                    policy_version=POLICY_VERSION,
                    raw_lineage=record.raw_lineage,
                )
            )

        state.cumulative_price_factor = next_price_factor
        state.cumulative_split_factor = next_split_factor
        if raw.close is not None:
            state.prior_raw_close = raw.close

    return outputs


def _content_bytes(value: _VALUE) -> bytes:
    return canonical_json(value.to_content_dict())


def _sorted_content_bytes(values: Iterable[_VALUE]) -> bytes:
    """Canonicalize the output after sorting complete element bytes."""

    items = sorted((value.to_content_dict() for value in values), key=canonical_json)
    return canonical_json(items)


def _content_checksum(values: Iterable[_VALUE]) -> str:
    return sha256_bytes(_sorted_content_bytes(values))


def _source_checksum(value: _VALUE) -> str | None:
    if isinstance(value, DailyBarCandidate):
        return value.raw_lineage.provider_record_checksum
    if value.raw_lineage is not None:
        return value.raw_lineage.provider_record_checksum
    return None


def _filtered_content(
    values: Iterable[_VALUE], source_checksums: set[str]
) -> tuple[bytes, ...]:
    return tuple(
        sorted(
            _content_bytes(value)
            for value in values
            if _source_checksum(value) in source_checksums
        )
    )


def _later_split_record(case: NormalizationCase) -> ProviderRecord:
    """Create a valid action strictly after every generated session."""

    first_record = case.records[0]
    return ProviderRecord(
        provider=first_record.provider,
        request_content_key=first_record.request_content_key,
        symbol=case.symbols[0],
        raw_bar=RawDailyBar(
            provider_date=_LATER_SESSION,
            open=Decimal("80"),
            high=Decimal("82"),
            low=Decimal("79"),
            close=Decimal("81"),
            adj_close=Decimal("81.125"),
            volume=Decimal("2000"),
        ),
        raw_action=RawCorporateAction(
            dividend=Decimal("0"),
            split_ratio=Decimal("2"),
            provider_fields={"source": "causal-suffix"},
        ),
        provider_fields={"fixture": "property-04", "suffix": True},
    )


# Feature: quantitative-research-platform, Property 4: Causal normalization determinism and confluence
# Validates: Requirements 4.1–4.18, 7.10, 9.19, 17.4, 17.7.
@settings(max_examples=100, deadline=None)
@given(case=normalization_cases())
def test_causal_normalization_determinism_and_confluence(
    case: NormalizationCase,
) -> None:
    """Normalization matches an independent Decimal model for every input order."""

    policy = CausalForwardAdjustmentV1()
    actual = list(Normalizer(policy).normalize(case.records, case.calendar, policy))
    expected = _reference_outputs(case.records, case.calendar)
    permuted = list(
        Normalizer(policy).normalize(case.permuted_records, case.calendar, policy)
    )
    regrouped = list(
        Normalizer(policy).normalize(
            chain.from_iterable(case.regrouped_batches), case.calendar, policy
        )
    )

    # The independent reference covers both the exact Decimal equations and
    # the raw/provider action objects carried into each candidate.
    assert _sorted_content_bytes(actual) == _sorted_content_bytes(expected)
    assert _sorted_content_bytes(actual) == _sorted_content_bytes(permuted)
    assert _sorted_content_bytes(actual) == _sorted_content_bytes(regrouped)
    assert _content_checksum(actual) == _content_checksum(permuted)
    assert _content_checksum(actual) == _content_checksum(regrouped)

    source_by_checksum = {
        record.provider_record_checksum: record for record in case.records
    }
    for value in actual:
        if not isinstance(value, DailyBarCandidate):
            continue
        source = source_by_checksum[value.raw_lineage.provider_record_checksum]
        assert value.raw_bar == source.raw_bar
        assert value.raw_action == source.raw_action
        assert value.raw_lineage == source.raw_lineage
        assert value.provider_adj_close == source.raw_bar.adj_close

    # A valid session with a wholly absent observation contributes no output;
    # in particular, the normalizer never fabricates a DailyBar for its key.
    output_source_checksums = {
        checksum
        for value in actual
        if (checksum := _source_checksum(value)) is not None
    }
    assert output_source_checksums.isdisjoint(set(case.absent_checksums))

    # Every generated invalid XNYS label is retained as a non-session
    # quarantine decision, while every present valid session has one candidate.
    invalid_checksums = {
        record.provider_record_checksum
        for record in case.records
        if record.provider_date in case.invalid_sessions
    }
    invalid_outputs = {
        checksum
        for value in actual
        if isinstance(value, QuarantineRecord)
        and (checksum := _source_checksum(value)) is not None
    }
    assert invalid_outputs == invalid_checksums
    assert all(
        value.reason_codes == ("session.non_xnys",)
        for value in actual
        if isinstance(value, QuarantineRecord)
        and _source_checksum(value) in invalid_checksums
    )

    # Appending a later action cannot alter any candidate or quarantine in the
    # causal prefix, even though it changes the later cumulative factor.
    suffix_record = _later_split_record(case)
    extended_calendar = FixtureXNYSCalendar((*case.valid_sessions, _LATER_SESSION))
    prefix = list(Normalizer(policy).normalize(case.records, extended_calendar, policy))
    with_suffix = list(
        Normalizer(policy).normalize(
            (*case.records, suffix_record), extended_calendar, policy
        )
    )
    original_checksums = set(source_by_checksum)
    assert _filtered_content(prefix, original_checksums) == _filtered_content(
        with_suffix, original_checksums
    )

    # The regrouping is a genuine bounded-batch partition, not a changed data
    # set; this guards the input seam explicitly.
    assert tuple(chain.from_iterable(case.regrouped_batches)) == case.permuted_records
    assert all(1 <= len(batch) <= 10 for batch in case.regrouped_batches)
    assert len(case.regrouped_batches) >= 1
