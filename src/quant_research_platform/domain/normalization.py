"""Causal, deterministic normalization of provider market records.

The normalizer owns the platform's research-adjustment coordinate.  It does not
mutate provider records, synthesize observations, or apply provider ``Adj Close``
values to OHLC.  Validation remains a separate phase: partially populated rows
are emitted as candidates so their raw values and deterministic row violations
can be retained, while rows for which a corporate-action equation cannot be
solved are quarantined here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Final, Protocol, TypeAlias, cast, runtime_checkable

from .errors import QuarantineReason
from .market import (
    CorporateAction,
    DailyBarCandidate,
    ProviderRecord,
    QuarantineRecord,
    QuarantineSourceKind,
    RawCorporateAction,
    RawDailyBar,
)

POLICY_VERSION: Final = "causal_forward_v1"
DECIMAL_PRECISION: Final = 28
DECIMAL_ROUNDING: Final = ROUND_HALF_EVEN
FACTOR_SCALE: Final = 18
_FACTOR_QUANTUM: Final = Decimal("0.000000000000000001")
_ZERO: Final = Decimal("0")
_ONE: Final = Decimal("1")


@runtime_checkable
class CorporateActionPolicy(Protocol):
    """Protocol for a versioned provider-action adjustment policy."""

    @property
    def version(self) -> str: ...

    @property
    def source_fields(self) -> tuple[str, ...]: ...

    @property
    def decimal_precision(self) -> int: ...

    @property
    def rounding_mode(self) -> str: ...

    def calculate(
        self,
        raw_action: RawCorporateAction,
        prior_raw_close: Decimal | None,
        cumulative_price_factor: Decimal = _ONE,
        cumulative_split_factor: Decimal = _ONE,
    ) -> AdjustmentCalculation: ...

    def action_source_fields(
        self, raw_action: RawCorporateAction
    ) -> tuple[str, ...]: ...

    def to_content_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class AdjustmentCalculation:
    """Exact Decimal result of one session's corporate-action equation.

    ``reason`` is populated only when the equation is intentionally unresolved.
    The normalizer turns that result into a deterministic quarantine record
    rather than guessing a factor.
    """

    split_ratio: Decimal
    dividend: Decimal
    reference_price: Decimal | None
    dividend_factor: Decimal | None
    cumulative_price_factor: Decimal | None
    cumulative_split_factor: Decimal | None
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.reason is None


@dataclass(frozen=True, slots=True)
class CausalForwardAdjustmentV1:
    """The single approved causal total-return policy for Phase 1.

    Prices are adjusted forward from the action session.  A split is applied
    before a same-session dividend, volume is adjusted only by cumulative
    splits, and provider ``Adj Close`` is never used in the calculation.
    Arithmetic is performed in a 28-digit Decimal context.  The Arrow schema
    performs the one later conversion to IEEE-754 float64; ``round_float64``
    exposes that conversion for callers that need to make it explicit.
    """

    version: str = POLICY_VERSION
    source_fields: tuple[str, ...] = (
        "raw_action.split_ratio",
        "raw_action.dividend",
        "raw_bar.adj_close",
    )
    decimal_precision: int = DECIMAL_PRECISION
    rounding_mode: str = "ROUND_HALF_EVEN"
    split_treatment: str = (
        "multiply cumulative price and split factors by new_shares_per_old_share; "
        "divide volume by cumulative split factor"
    )
    dividend_treatment: str = (
        "cash dividend per post-split share; apply reference/(reference-dividend) "
        "after split conversion"
    )
    volume_treatment: str = (
        "divide raw volume by cumulative split factor; dividends do not alter volume"
    )
    rounding_treatment: str = (
        "Decimal precision 28 with ROUND_HALF_EVEN; cumulative audit factors are "
        "quantized to 18 places and accepted float fields convert once to "
        "IEEE-754 float64"
    )

    @property
    def policy_version(self) -> str:
        """Compatibility alias used by callers that name versions explicitly."""

        return self.version

    def action_source_fields(self, raw_action: RawCorporateAction) -> tuple[str, ...]:
        """Return the provider action columns present on this raw record."""

        fields: list[str] = []
        # Keep this order stable and human-readable in canonical action rows.
        if raw_action.dividend is not None:
            fields.append("Dividends")
        if raw_action.split_ratio is not None:
            fields.append("Stock Splits")
        return tuple(fields)

    def calculate(
        self,
        raw_action: RawCorporateAction,
        prior_raw_close: Decimal | None,
        cumulative_price_factor: Decimal = _ONE,
        cumulative_split_factor: Decimal = _ONE,
    ) -> AdjustmentCalculation:
        """Calculate cumulative causal factors for one action row.

        ``None`` action fields mean the provider did not supply that action and
        are interpreted as split ``1`` and dividend ``0``.  A missing prior
        close is required only when a non-zero dividend needs the reference
        price.  The returned factors include the current session's action.
        """

        raw_split_ratio = raw_action.split_ratio
        raw_dividend = raw_action.dividend
        split_ratio = (
            _ONE if raw_split_ratio is None else cast(Decimal, raw_split_ratio)
        )
        dividend = _ZERO if raw_dividend is None else cast(Decimal, raw_dividend)

        with localcontext() as context:
            context.prec = self.decimal_precision
            context.rounding = DECIMAL_ROUNDING

            if not split_ratio.is_finite() or split_ratio <= _ZERO:
                return AdjustmentCalculation(
                    split_ratio=split_ratio,
                    dividend=dividend,
                    reference_price=None,
                    dividend_factor=None,
                    cumulative_price_factor=None,
                    cumulative_split_factor=None,
                    reason="split_ratio_non_positive_or_non_finite",
                )

            reference_price: Decimal | None = None
            dividend_factor = _ONE
            if dividend != _ZERO:
                if prior_raw_close is None or not prior_raw_close.is_finite():
                    return AdjustmentCalculation(
                        split_ratio=split_ratio,
                        dividend=dividend,
                        reference_price=None,
                        dividend_factor=None,
                        cumulative_price_factor=None,
                        cumulative_split_factor=None,
                        reason="dividend_missing_prior_close",
                    )
                reference_price = prior_raw_close / split_ratio
                denominator = reference_price - dividend
                if not denominator.is_finite() or denominator <= _ZERO:
                    return AdjustmentCalculation(
                        split_ratio=split_ratio,
                        dividend=dividend,
                        reference_price=reference_price,
                        dividend_factor=None,
                        cumulative_price_factor=None,
                        cumulative_split_factor=None,
                        reason="dividend_reference_non_positive",
                    )
                dividend_factor = reference_price / denominator

            next_split_factor = cumulative_split_factor * split_ratio
            next_price_factor = cumulative_price_factor * split_ratio * dividend_factor
            if not next_split_factor.is_finite() or not next_price_factor.is_finite():
                return AdjustmentCalculation(
                    split_ratio=split_ratio,
                    dividend=dividend,
                    reference_price=reference_price,
                    dividend_factor=dividend_factor,
                    cumulative_price_factor=None,
                    cumulative_split_factor=None,
                    reason="cumulative_factor_non_finite",
                )

            return AdjustmentCalculation(
                split_ratio=split_ratio,
                dividend=dividend,
                reference_price=reference_price,
                dividend_factor=dividend_factor,
                cumulative_price_factor=next_price_factor,
                cumulative_split_factor=next_split_factor,
            )

    # ``adjust`` is a concise alias useful to policy callers and keeps the
    # policy API readable without creating a second implementation.
    adjust = calculate

    @staticmethod
    def round_float64(value: Decimal | None) -> float | None:
        """Convert an exact Decimal to one nearest-even IEEE-754 float64."""

        return None if value is None else float(value)

    # A second descriptive alias makes the conversion boundary explicit.
    to_float64 = round_float64

    def to_content_dict(self) -> dict[str, object]:
        """Return the declared policy inputs and treatment, without runtime state."""

        return {
            "policy_version": self.version,
            "source_fields": list(self.source_fields),
            "equations": {
                "cumulative_split_factor": "Fs_t = Fs_(t-1) * S_t",
                "reference_price": "reference_t = C_(t-1) / S_t",
                "dividend_factor": "G_t = reference_t / (reference_t - D_t)",
                "cumulative_price_factor": "Fp_t = Fp_(t-1) * S_t * G_t",
                "adjusted_ohlc": "adjusted = raw * Fp_t",
                "adjusted_volume": "adjusted_volume = raw_volume / Fs_t",
            },
            "split_treatment": self.split_treatment,
            "dividend_treatment": self.dividend_treatment,
            "volume_treatment": self.volume_treatment,
            "rounding_treatment": self.rounding_treatment,
            "audit_factor_scale": FACTOR_SCALE,
        }


@dataclass(frozen=True, slots=True)
class _MappedRecord:
    record: ProviderRecord
    session: date | None

    def sort_key(self) -> tuple[str, date, str]:
        # Provider dates are the stable fallback for a non-session quarantine;
        # valid records use their mapped XNYS label exactly.
        return (
            self.record.symbol,
            self.session if self.session is not None else self.record.provider_date,
            self.record.provider_record_checksum,
        )


@dataclass(frozen=True, slots=True)
class NormalizationSeed:
    """Causal state immediately before a normalized suffix.

    Incremental ingestion must recompute the overlap and every later session,
    but it must not restart corporate-action factors at one.  This immutable
    value object is the narrow state hand-off from a verified parent snapshot
    to :class:`Normalizer`.
    """

    prior_raw_close: Decimal | int | float | str | None = None
    cumulative_price_factor: Decimal | int | float | str = _ONE
    cumulative_split_factor: Decimal | int | float | str = _ONE

    def __post_init__(self) -> None:
        prior = (
            None if self.prior_raw_close is None else Decimal(str(self.prior_raw_close))
        )
        price = Decimal(str(self.cumulative_price_factor))
        split = Decimal(str(self.cumulative_split_factor))
        if prior is not None and not prior.is_finite():
            raise ValueError("prior_raw_close must be finite or None")
        if not price.is_finite() or price <= _ZERO:
            raise ValueError("cumulative_price_factor must be finite and positive")
        if not split.is_finite() or split <= _ZERO:
            raise ValueError("cumulative_split_factor must be finite and positive")
        object.__setattr__(self, "prior_raw_close", prior)
        object.__setattr__(self, "cumulative_price_factor", price)
        object.__setattr__(self, "cumulative_split_factor", split)

    def to_content_dict(self) -> dict[str, object]:
        """Return deterministic state for diagnostics, never a mutable state."""

        return {
            "prior_raw_close": self.prior_raw_close,
            "cumulative_price_factor": self.cumulative_price_factor,
            "cumulative_split_factor": self.cumulative_split_factor,
        }


@dataclass(slots=True)
class _SymbolState:
    prior_raw_close: Decimal | None = None
    cumulative_price_factor: Decimal = _ONE
    cumulative_split_factor: Decimal = _ONE


class Normalizer:
    """Deterministically map provider records to bars or quarantine decisions."""

    def __init__(self, policy: CorporateActionPolicy | None = None) -> None:
        default_policy: CorporateActionPolicy = CausalForwardAdjustmentV1()
        self.policy = default_policy if policy is None else policy

    def normalize(
        self,
        records: Iterable[ProviderRecord],
        calendar: ExchangeCalendar,
        policy: CorporateActionPolicy | None = None,
        *,
        seeds: Mapping[str, NormalizationSeed] | None = None,
    ) -> Iterator[DailyBarCandidate | QuarantineRecord]:
        """Yield sorted candidate/quarantine records for *records*.

        Records are materialized only to establish the required deterministic
        sort before stateful per-symbol processing.  Same-session records share
        one pre-session state, which makes equivalent duplicate records produce
        byte-equivalent candidates; validation later decides whether a key is
        accepted, collapsed, or conflicting.

        ``seeds`` is an optional causal state map used by incremental ingestion.
        Each seed describes the state immediately before the first session in
        ``records`` for that symbol.  Omitting it preserves the original full
        ingestion behavior (all factors start at one).
        """

        active_policy = policy or self.policy
        seed_states: dict[str, _SymbolState] = {}
        if seeds is not None:
            if not isinstance(seeds, Mapping):
                raise TypeError(
                    "seeds must be a mapping of symbols to NormalizationSeed"
                )
            for symbol, seed in seeds.items():
                if not isinstance(seed, NormalizationSeed):
                    raise TypeError("seeds must contain NormalizationSeed values")
                normalized_symbol = str(symbol).strip().upper()
                if not normalized_symbol:
                    raise ValueError("seed symbols must not be blank")
                seed_states[normalized_symbol] = _SymbolState(
                    prior_raw_close=cast(Decimal | None, seed.prior_raw_close),
                    cumulative_price_factor=cast(Decimal, seed.cumulative_price_factor),
                    cumulative_split_factor=cast(Decimal, seed.cumulative_split_factor),
                )

        mapped = [self._map_record(record, calendar) for record in records]
        mapped.sort(key=_MappedRecord.sort_key)

        states: dict[str, _SymbolState] = seed_states
        index = 0
        while index < len(mapped):
            mapped_record = mapped[index]
            record = mapped_record.record
            if mapped_record.session is None:
                yield self._non_session_quarantine(record)
                index += 1
                continue

            group_end = index + 1
            while group_end < len(mapped):
                next_record = mapped[group_end]
                if (
                    next_record.record.symbol != record.symbol
                    or next_record.session != mapped_record.session
                ):
                    break
                group_end += 1

            state = states.setdefault(record.symbol, _SymbolState())
            group = mapped[index:group_end]
            calculations: list[AdjustmentCalculation] = []
            invalid = False
            for item in group:
                calculation = self._calculate(active_policy, item.record, state)
                calculations.append(calculation)
                if not calculation.valid:
                    invalid = True
                    yield self._policy_quarantine(
                        item.record,
                        mapped_record.session,
                        state,
                        calculation,
                        active_policy,
                    )
                    continue
                candidate = self._candidate(
                    item.record,
                    mapped_record.session,
                    calculation,
                    active_policy,
                    calendar,
                )
                if candidate is not None:
                    yield candidate

            # Advance factors once per logical session.  This is essential for
            # equivalent duplicate records, which are collapsed by validation.
            representative = group[0].record
            if not invalid and calculations:
                representative_calculation = calculations[0]
                if representative_calculation.valid:
                    assert representative_calculation.cumulative_price_factor
                    assert representative_calculation.cumulative_split_factor
                    state.cumulative_price_factor = (
                        representative_calculation.cumulative_price_factor
                    )
                    state.cumulative_split_factor = (
                        representative_calculation.cumulative_split_factor
                    )
            # A raw close is still provider provenance even when the row is
            # partially invalid; it is the prior-close input for a later action.
            if representative.raw_bar.close is not None:
                state.prior_raw_close = cast(Decimal, representative.raw_bar.close)
            index = group_end

    def seed_states(
        self,
        records: Iterable[ProviderRecord],
        calendar: ExchangeCalendar,
        *,
        before_session: date | None,
        policy: CorporateActionPolicy | None = None,
    ) -> dict[str, NormalizationSeed]:
        """Reconstruct causal state immediately before ``before_session``.

        This deliberately mirrors the state-advance portion of
        :meth:`normalize`, including same-session duplicate handling and
        action-only rows.  It never emits or mutates candidates, and therefore
        can safely be used with verified parent records before an incremental
        revision boundary.
        """

        if before_session is not None and not isinstance(before_session, date):
            raise TypeError("before_session must be a calendar date or None")
        active_policy = policy or self.policy
        mapped = [self._map_record(record, calendar) for record in records]
        mapped.sort(key=_MappedRecord.sort_key)
        states: dict[str, _SymbolState] = {}
        index = 0
        while index < len(mapped):
            mapped_record = mapped[index]
            if mapped_record.session is None:
                index += 1
                continue
            if before_session is not None and mapped_record.session >= before_session:
                break
            group_end = index + 1
            while group_end < len(mapped):
                next_record = mapped[group_end]
                if (
                    next_record.record.symbol != mapped_record.record.symbol
                    or next_record.session != mapped_record.session
                ):
                    break
                group_end += 1

            representative = mapped_record.record
            state = states.setdefault(representative.symbol, _SymbolState())
            calculations: list[AdjustmentCalculation] = []
            invalid = False
            for item in mapped[index:group_end]:
                calculation = self._calculate(active_policy, item.record, state)
                calculations.append(calculation)
                if not calculation.valid:
                    invalid = True
            if not invalid and calculations and calculations[0].valid:
                calculation = calculations[0]
                assert calculation.cumulative_price_factor is not None
                assert calculation.cumulative_split_factor is not None
                state.cumulative_price_factor = calculation.cumulative_price_factor
                state.cumulative_split_factor = calculation.cumulative_split_factor
            if representative.raw_bar.close is not None:
                state.prior_raw_close = cast(Decimal, representative.raw_bar.close)
            index = group_end

        return {
            symbol: NormalizationSeed(
                prior_raw_close=state.prior_raw_close,
                cumulative_price_factor=state.cumulative_price_factor,
                cumulative_split_factor=state.cumulative_split_factor,
            )
            for symbol, state in sorted(states.items())
        }

    # Descriptive aliases keep the state hand-off discoverable to application
    # services without exposing the mutable private ``_SymbolState`` type.
    build_seeds = seed_states
    causal_seeds = seed_states

    def normalize_seeded(
        self,
        records: Iterable[ProviderRecord],
        calendar: ExchangeCalendar,
        seeds: Mapping[str, NormalizationSeed],
        policy: CorporateActionPolicy | None = None,
    ) -> Iterator[DailyBarCandidate | QuarantineRecord]:
        """Normalize a suffix from verified causal state."""

        return self.normalize(records, calendar, policy, seeds=seeds)

    @staticmethod
    def _map_record(
        record: ProviderRecord, calendar: ExchangeCalendar
    ) -> _MappedRecord:
        if not isinstance(record, ProviderRecord):
            raise TypeError("records must contain ProviderRecord values")
        session = (
            record.provider_date if calendar.is_session(record.provider_date) else None
        )
        return _MappedRecord(record=record, session=session)

    @staticmethod
    def _calculate(
        policy: CorporateActionPolicy,
        record: ProviderRecord,
        state: _SymbolState,
    ) -> AdjustmentCalculation:
        return policy.calculate(
            record.raw_action,
            state.prior_raw_close,
            state.cumulative_price_factor,
            state.cumulative_split_factor,
        )

    @staticmethod
    def _observation_absent(raw_bar: RawDailyBar) -> bool:
        return all(
            value is None
            for value in (
                raw_bar.open,
                raw_bar.high,
                raw_bar.low,
                raw_bar.close,
                raw_bar.volume,
            )
        )

    def _candidate(
        self,
        record: ProviderRecord,
        session: date,
        calculation: AdjustmentCalculation,
        policy: CorporateActionPolicy,
        calendar: ExchangeCalendar,
    ) -> DailyBarCandidate | None:
        if self._observation_absent(record.raw_bar):
            # Action-only rows may still advance causal state, but no market
            # observation is fabricated for the missing key.
            return None

        assert calculation.cumulative_price_factor is not None
        assert calculation.cumulative_split_factor is not None
        exact_price_factor = calculation.cumulative_price_factor
        exact_split_factor = calculation.cumulative_split_factor
        price_factor = self._quantize_factor(exact_price_factor)
        split_factor = self._quantize_factor(exact_split_factor)
        raw = record.raw_bar
        adjusted_open = self._multiply(raw.open, exact_price_factor)
        adjusted_high = self._multiply(raw.high, exact_price_factor)
        adjusted_low = self._multiply(raw.low, exact_price_factor)
        adjusted_close = self._multiply(raw.close, exact_price_factor)
        adjusted_volume = self._divide(raw.volume, exact_split_factor)
        action = CorporateAction(
            symbol=record.symbol,
            session=session,
            dividend=calculation.dividend,
            split_ratio=calculation.split_ratio,
            raw_lineage=record.raw_lineage,
            source_fields=policy.action_source_fields(record.raw_action),
        )
        return DailyBarCandidate(
            symbol=record.symbol,
            session=session,
            event_timestamp=calendar.close_timestamp(session),
            raw_bar=raw,
            raw_action=record.raw_action,
            corporate_action=action,
            adjusted_open=adjusted_open,
            adjusted_high=adjusted_high,
            adjusted_low=adjusted_low,
            adjusted_close=adjusted_close,
            adjusted_volume=adjusted_volume,
            execution_adjusted_open=raw.open,
            sizing_adjusted_close=raw.close,
            cumulative_price_factor=price_factor,
            cumulative_split_factor=split_factor,
            policy_version=policy.version,
            raw_lineage=record.raw_lineage,
        )

    @staticmethod
    def _quantize_factor(value: Decimal | None) -> Decimal:
        if value is None:  # pragma: no cover - guarded by the candidate caller
            raise ValueError("valid adjustment factors are required")
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = DECIMAL_ROUNDING
            return value.quantize(_FACTOR_QUANTUM)

    @staticmethod
    def _multiply(
        value: Decimal | int | float | str | None, factor: Decimal
    ) -> Decimal | None:
        if value is None:
            return None
        decimal_value = cast(Decimal, value)
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = DECIMAL_ROUNDING
            return decimal_value * factor

    @staticmethod
    def _divide(
        value: Decimal | int | float | str | None, divisor: Decimal
    ) -> Decimal | None:
        if value is None:
            return None
        decimal_value = cast(Decimal, value)
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = DECIMAL_ROUNDING
            return decimal_value / divisor

    @staticmethod
    def _non_session_quarantine(record: ProviderRecord) -> QuarantineRecord:
        return QuarantineRecord(
            source_kind=QuarantineSourceKind.PROVIDER_RECORD,
            reason_codes=(QuarantineReason.NON_SESSION.value,),
            offending_values={
                "provider_date": record.provider_date,
                "symbol": record.symbol,
                "provider_record_checksum": record.provider_record_checksum,
            },
            raw_lineage=record.raw_lineage,
        )

    @staticmethod
    def _policy_quarantine(
        record: ProviderRecord,
        session: date,
        state: _SymbolState,
        calculation: AdjustmentCalculation,
        policy: CorporateActionPolicy,
    ) -> QuarantineRecord:
        return QuarantineRecord(
            source_kind=QuarantineSourceKind.PROVIDER_RECORD,
            reason_codes=(QuarantineReason.NORMALIZATION_POLICY.value,),
            offending_values={
                "provider_date": record.provider_date,
                "symbol": record.symbol,
                "split_ratio": calculation.split_ratio,
                "dividend": calculation.dividend,
                "prior_raw_close": state.prior_raw_close,
                "reference_price": calculation.reference_price,
                "policy_reason": calculation.reason or "unknown",
                "provider_record_checksum": record.provider_record_checksum,
            },
            policy_version=policy.version,
            symbol=record.symbol,
            session=session,
            raw_lineage=record.raw_lineage,
        )


@runtime_checkable
class ExchangeCalendar(Protocol):
    """Narrow calendar surface consumed by :class:`Normalizer`."""

    def is_session(self, value: date) -> bool: ...

    def close_timestamp(self, session: date) -> datetime: ...


NormalizationOutput: TypeAlias = DailyBarCandidate | QuarantineRecord


def normalize(
    records: Iterable[ProviderRecord],
    calendar: ExchangeCalendar,
    policy: CorporateActionPolicy | None = None,
    *,
    seeds: Mapping[str, NormalizationSeed] | None = None,
) -> Iterator[NormalizationOutput]:
    """Functional convenience wrapper around :class:`Normalizer`."""

    return Normalizer(policy).normalize(records, calendar, seeds=seeds)


def seed_causal_state(
    records: Iterable[ProviderRecord],
    calendar: ExchangeCalendar,
    *,
    before_session: date | None,
    policy: CorporateActionPolicy | None = None,
) -> dict[str, NormalizationSeed]:
    """Build immutable causal seeds for an incremental normalization suffix."""

    return Normalizer(policy).seed_states(
        records,
        calendar,
        before_session=before_session,
    )


__all__ = [
    "AdjustmentCalculation",
    "CausalForwardAdjustmentV1",
    "CorporateActionPolicy",
    "DECIMAL_PRECISION",
    "DECIMAL_ROUNDING",
    "ExchangeCalendar",
    "FACTOR_SCALE",
    "Normalizer",
    "NormalizationOutput",
    "NormalizationSeed",
    "POLICY_VERSION",
    "normalize",
    "seed_causal_state",
]
