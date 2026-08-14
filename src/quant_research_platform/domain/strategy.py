"""Immutable value objects for monthly-momentum strategy decisions.

The strategy layer uses exact rational target weights.  It intentionally contains no
market-data, calendar, or engine imports: history lookup and ranking policies are
implemented in later layers while this module protects the resulting scientific
records from ambiguity.
"""

# ruff: noqa: E501, SIM102

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from fractions import Fraction
from math import gcd
from typing import Final, Protocol, TypeAlias, cast

from .canonical import canonical_rational, sha256_canonical_json


STRATEGY_IDENTIFIER: Final = "monthly_momentum_v1"
LONG_LOOKBACK_SESSIONS: Final = 252
SKIP_RECENT_SESSIONS: Final = 21
WARM_UP_SESSIONS: Final = 253
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class StrategyExclusionReason(StrEnum):
    """Machine-readable reasons that a strategy decision has zero target weight."""

    MISSING_LONG_ENDPOINT = "missing_long_endpoint"
    MISSING_SHORT_ENDPOINT = "missing_short_endpoint"
    WARM_UP_INCOMPLETE = "warm_up_incomplete"
    ASSET_NOT_TRADABLE = "asset_not_tradable"
    NOT_SELECTED = "not_selected"


def _require_date(name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a calendar date")
    return value


def _require_integer(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite_decimal(
    name: str, value: Decimal, *, positive: bool = False
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be blank")
    if any(character.isspace() for character in normalized):
        raise ValueError("symbol must not contain whitespace")
    return normalized


def _normalize_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    if not isinstance(reason, str):
        raise TypeError("exclusion_reason must be a string or StrategyExclusionReason")
    normalized = "_".join(reason.strip().lower().split())
    if not normalized:
        raise ValueError("exclusion_reason must not be blank")
    return normalized


def _normalize_checksum(name: str, checksum: str | None) -> str | None:
    if checksum is None:
        return None
    if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return checksum


@dataclass(frozen=True, slots=True)
class RationalWeight:
    """A non-negative, reduced exact portfolio target weight.

    Equality is mathematical rather than representational: ``2/4`` is reduced
    to ``1/2`` during construction and consequently compares equal to ``1/2``.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator = _require_integer("numerator", self.numerator)
        denominator = _require_integer("denominator", self.denominator, minimum=1)
        divisor = gcd(numerator, denominator)
        object.__setattr__(self, "numerator", numerator // divisor)
        object.__setattr__(self, "denominator", denominator // divisor)

    @classmethod
    def zero(cls) -> RationalWeight:
        """Return the exact all-cash target weight."""
        return cls(0, 1)

    @classmethod
    def equal_allocation(cls, count: int) -> RationalWeight:
        """Return the exact weight assigned to each of *count* selected symbols."""
        return cls(1, _require_integer("count", count, minimum=1))

    @classmethod
    def sum(cls, weights: Iterable[RationalWeight]) -> RationalWeight:
        """Return an exact sum without converting weights to binary floats."""
        total = Fraction(0, 1)
        for weight in weights:
            if not isinstance(weight, RationalWeight):
                raise TypeError("weights must contain only RationalWeight values")
            total += Fraction(weight.numerator, weight.denominator)
        return cls(total.numerator, total.denominator)

    @property
    def fraction(self) -> Fraction:
        """Expose the exact standard-library fraction representation."""
        return Fraction(self.numerator, self.denominator)

    def as_decimal(self, *, precision: int = 28) -> Decimal:
        """Return a display/calculation decimal without changing exact identity."""
        _require_integer("precision", precision, minimum=1)
        with localcontext() as context:
            context.prec = precision
            return Decimal(self.numerator) / Decimal(self.denominator)

    def to_canonical_string(self) -> str:
        """Return the canonical reduced ``numerator/denominator`` representation."""
        return canonical_rational(self.numerator, self.denominator)

    def to_serializable(self) -> str:
        """Return the canonical representation used by JSON/table serializers."""
        return self.to_canonical_string()


@dataclass(frozen=True, slots=True)
class MomentumStrategyParameters:
    """Fixed momentum parameters that must be recorded in run science metadata."""

    position_count: int
    long_lookback_sessions: int = 252
    skip_recent_sessions: int = 21

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_count",
            _require_integer("position_count", self.position_count, minimum=1),
        )
        if self.long_lookback_sessions != 252:
            raise ValueError("long_lookback_sessions must be fixed at 252")
        if self.skip_recent_sessions != 21:
            raise ValueError("skip_recent_sessions must be fixed at 21")

    def to_serializable(self) -> dict[str, int]:
        """Return the deterministic manifest projection for fixed score endpoints."""
        return {
            "long_lookback_sessions": self.long_lookback_sessions,
            "position_count": self.position_count,
            "skip_recent_sessions": self.skip_recent_sessions,
        }


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """One complete symbol decision made at a signal-session close."""

    signal_session: date
    symbol: str
    endpoint_252_session: date | None
    endpoint_252_close: Decimal | None
    endpoint_21_session: date | None
    endpoint_21_close: Decimal | None
    momentum_score: Decimal | None
    eligible: bool
    rank: int | None
    target_weight: RationalWeight
    exclusion_reason: StrategyExclusionReason | str | None
    endpoint_252_checksum: str | None = None
    endpoint_21_checksum: str | None = None

    def __post_init__(self) -> None:
        signal_session = _require_date("signal_session", self.signal_session)
        object.__setattr__(self, "signal_session", signal_session)
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))

        self._validate_endpoint(
            "endpoint_252", self.endpoint_252_session, self.endpoint_252_close
        )
        self._validate_endpoint(
            "endpoint_21", self.endpoint_21_session, self.endpoint_21_close
        )
        endpoint_252_session = self.endpoint_252_session
        endpoint_21_session = self.endpoint_21_session
        if endpoint_252_session is not None and endpoint_21_session is not None:
            if endpoint_252_session > endpoint_21_session:
                raise ValueError("endpoint_252_session must not be after endpoint_21_session")
        for endpoint_name, endpoint_session in (
            ("endpoint_252_session", endpoint_252_session),
            ("endpoint_21_session", endpoint_21_session),
        ):
            if endpoint_session is not None and endpoint_session > signal_session:
                raise ValueError(f"{endpoint_name} must not be after signal_session")

        endpoint_252_checksum = _normalize_checksum(
            "endpoint_252_checksum", self.endpoint_252_checksum
        )
        endpoint_21_checksum = _normalize_checksum(
            "endpoint_21_checksum", self.endpoint_21_checksum
        )
        if endpoint_252_checksum is not None and endpoint_252_session is None:
            raise ValueError("endpoint_252_checksum requires endpoint_252_session")
        if endpoint_21_checksum is not None and endpoint_21_session is None:
            raise ValueError("endpoint_21_checksum requires endpoint_21_session")
        object.__setattr__(self, "endpoint_252_checksum", endpoint_252_checksum)
        object.__setattr__(self, "endpoint_21_checksum", endpoint_21_checksum)

        if self.momentum_score is not None:
            object.__setattr__(
                self,
                "momentum_score",
                _require_finite_decimal("momentum_score", self.momentum_score),
            )
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be a bool")
        if self.rank is not None:
            object.__setattr__(
                self, "rank", _require_integer("rank", self.rank, minimum=1)
            )
        if not isinstance(self.target_weight, RationalWeight):
            raise TypeError("target_weight must be a RationalWeight")

        exclusion_reason = self._coerce_reason(self.exclusion_reason)
        object.__setattr__(self, "exclusion_reason", exclusion_reason)
        self._validate_eligibility(exclusion_reason)

    def _validate_endpoint(
        self, prefix: str, endpoint_session: date | None, endpoint_close: Decimal | None
    ) -> None:
        if (endpoint_session is None) != (endpoint_close is None):
            raise ValueError(f"{prefix}_session and {prefix}_close must be supplied together")
        if endpoint_session is not None:
            _require_date(f"{prefix}_session", endpoint_session)
            assert endpoint_close is not None
            _require_finite_decimal(f"{prefix}_close", endpoint_close, positive=True)

    @staticmethod
    def _coerce_reason(
        reason: StrategyExclusionReason | str | None,
    ) -> StrategyExclusionReason | None:
        normalized = _normalize_reason(reason)
        if normalized is None:
            return None
        try:
            return StrategyExclusionReason(normalized)
        except ValueError as error:
            raise ValueError(f"unsupported exclusion_reason: {reason!r}") from error

    def _validate_eligibility(self, reason: StrategyExclusionReason | None) -> None:
        if self.eligible:
            if (
                self.endpoint_252_session is None
                or self.endpoint_252_close is None
                or self.endpoint_21_session is None
                or self.endpoint_21_close is None
                or self.momentum_score is None
                or self.rank is None
            ):
                raise ValueError(
                    "eligible decisions require both endpoints, momentum_score, and rank"
                )
            if self.target_weight == RationalWeight.zero():
                if reason is not StrategyExclusionReason.NOT_SELECTED:
                    raise ValueError(
                        "an eligible zero-weight decision must use exclusion_reason "
                        "not_selected"
                    )
            elif reason is not None:
                raise ValueError("a selected decision must not have an exclusion_reason")
            return

        if self.rank is not None:
            raise ValueError("ineligible decisions must not have rank")
        if self.momentum_score is not None and (
            self.endpoint_252_session is None
            or self.endpoint_252_close is None
            or self.endpoint_21_session is None
            or self.endpoint_21_close is None
        ):
            raise ValueError(
                "an ineligible momentum score requires both score endpoints"
            )
        if self.target_weight != RationalWeight.zero():
            raise ValueError("ineligible decisions must have zero target_weight")
        if reason is None or reason is StrategyExclusionReason.NOT_SELECTED:
            raise ValueError(
                "ineligible decisions require a non-selection eligibility exclusion_reason"
            )

    def to_serializable(self) -> dict[str, object]:
        """Return a canonical-serializer-friendly decision representation."""
        return {
            "eligible": self.eligible,
            "endpoint_21_close": self.endpoint_21_close,
            "endpoint_21_checksum": self.endpoint_21_checksum,
            "endpoint_21_session": self.endpoint_21_session,
            "endpoint_252_close": self.endpoint_252_close,
            "endpoint_252_checksum": self.endpoint_252_checksum,
            "endpoint_252_session": self.endpoint_252_session,
            "exclusion_reason": (
                StrategyExclusionReason(self.exclusion_reason).value
                if self.exclusion_reason is not None
                else None
            ),
            "momentum_score": self.momentum_score,
            "rank": self.rank,
            "signal_session": self.signal_session,
            "symbol": self.symbol,
            "target_weight": self.target_weight.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class PriceObservation:
    """Small, framework-free input row accepted by ``monthly_momentum_v1``.

    Normalized ``DailyBarCandidate`` values are accepted directly as well.  This
    value object is useful for unit tests and for application ports that expose
    only the adjusted close and its canonical source-row checksum.
    """

    symbol: str
    session: date
    adjusted_close: Decimal | int | float | str
    checksum: str | None = None
    tradable: bool = True

    def __post_init__(self) -> None:
        symbol = _normalize_symbol(self.symbol)
        session = _require_date("session", self.session)
        if isinstance(self.adjusted_close, bool):
            raise TypeError("adjusted_close must be a finite positive Decimal")
        try:
            close = (
                self.adjusted_close
                if isinstance(self.adjusted_close, Decimal)
                else Decimal(str(self.adjusted_close))
            )
        except (InvalidOperation, ValueError) as error:
            raise TypeError("adjusted_close must be a finite positive Decimal") from error
        if not close.is_finite() or close <= 0:
            raise ValueError("adjusted_close must be finite and positive")
        if not isinstance(self.tradable, bool):
            raise TypeError("tradable must be a bool")
        checksum = _normalize_checksum("checksum", self.checksum)
        if checksum is None:
            checksum = sha256_canonical_json(
                {
                    "adjusted_close": close,
                    "session": session,
                    "symbol": symbol,
                }
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "adjusted_close", close)
        object.__setattr__(self, "checksum", checksum)

    @property
    def canonical_row_checksum(self) -> str:
        """Alias matching the normalized daily-bar field name."""

        assert self.checksum is not None
        return self.checksum

    def sort_key(self) -> tuple[str, date, str]:
        return (self.symbol, self.session, self.canonical_row_checksum)

    def to_serializable(self) -> dict[str, object]:
        return {
            "adjusted_close": self.adjusted_close,
            "canonical_row_checksum": self.canonical_row_checksum,
            "session": self.session,
            "symbol": self.symbol,
            "tradable": self.tradable,
        }


@dataclass(frozen=True, slots=True)
class PriceHistory:
    """Immutable compact history container for strategy callers and tests."""

    observations: tuple[PriceObservation, ...]
    sessions: tuple[date, ...] = ()
    universe: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple):
            observations = tuple(self.observations)
        else:
            observations = self.observations
        if any(not isinstance(item, PriceObservation) for item in observations):
            raise TypeError("observations must contain only PriceObservation values")
        supplied_sessions = self.sessions
        if not isinstance(supplied_sessions, tuple):
            supplied_sessions = tuple(supplied_sessions)
        normalized_sessions = tuple(
            sorted({_require_date("session", item) for item in supplied_sessions})
        )
        if not normalized_sessions:
            normalized_sessions = tuple(sorted({item.session for item in observations}))
        if not isinstance(self.universe, tuple):
            supplied_universe = tuple(self.universe)
        else:
            supplied_universe = self.universe
        if supplied_universe:
            normalized_universe: list[str] = []
            seen: set[str] = set()
            for symbol in supplied_universe:
                normalized = _normalize_symbol(symbol)
                if normalized in seen:
                    raise ValueError("universe must not contain duplicate symbols")
                seen.add(normalized)
                normalized_universe.append(normalized)
            universe = tuple(normalized_universe)
        else:
            universe = tuple(sorted({item.symbol for item in observations}))
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "sessions", normalized_sessions)
        object.__setattr__(self, "universe", universe)

    @classmethod
    def from_records(
        cls,
        records: Iterable[PriceObservation],
        *,
        sessions: Iterable[date] = (),
        universe: Iterable[str] = (),
    ) -> PriceHistory:
        """Build a history while keeping its session calendar explicit."""

        return cls(tuple(records), tuple(sessions), tuple(universe))

    def __iter__(self):
        return iter(self.observations)


class StrategyCalendar(Protocol):
    """The only calendar operation needed when deriving month-end signals."""

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]: ...


HistoryInput: TypeAlias = PriceHistory | Iterable[object] | Mapping[object, object]
TradabilityInput: TypeAlias = (
    Mapping[object, object] | Callable[[str, date], bool] | None
)


def _date_like(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    date_method = getattr(value, "date", None)
    if callable(date_method):
        converted = date_method()
        if isinstance(converted, date) and not isinstance(converted, datetime):
            return converted
    return None


def _is_price_scalar(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return False
    return decimal_value.is_finite()


def _mapping_value(mapping: Mapping[object, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _observation_from_value(
    value: object,
    *,
    symbol_hint: str | None = None,
    session_hint: date | None = None,
) -> PriceObservation | None:
    """Coerce common normalized-bar/fixture shapes without importing pandas."""

    if value is None:
        return None
    if isinstance(value, PriceObservation):
        if symbol_hint is not None and value.symbol != _normalize_symbol(symbol_hint):
            raise ValueError("history symbol does not match its symbol mapping")
        if session_hint is not None and value.session != session_hint:
            raise ValueError("history session does not match its session mapping")
        return value

    if isinstance(value, Mapping):
        symbol_value = _mapping_value(value, ("symbol", "ticker"))
        session_value = _mapping_value(value, ("session", "date", "provider_date"))
        close_value = _mapping_value(
            value, ("adjusted_close", "adj_close", "adjustedClose", "price", "close")
        )
        checksum_value = _mapping_value(
            value,
            (
                "canonical_row_checksum",
                "row_checksum",
                "checksum",
                "provider_record_checksum",
            ),
        )
        tradable_value = _mapping_value(value, ("tradable", "is_tradable"))
        symbol = symbol_hint if symbol_value is None else symbol_value
        session = session_hint if session_value is None else _date_like(session_value)
        if symbol is None or session is None or close_value is None:
            return None
        return PriceObservation(
            symbol=symbol,
            session=session,
            adjusted_close=close_value,
            checksum=checksum_value if isinstance(checksum_value, str) else None,
            tradable=True if tradable_value is None else tradable_value,
        )

    if isinstance(value, (tuple, list)):
        values = tuple(value)
        if len(values) >= 3 and isinstance(values[0], str):
            tuple_session = _date_like(values[1])
            if tuple_session is not None:
                return PriceObservation(
                    symbol=values[0],
                    session=tuple_session,
                    adjusted_close=values[2],
                    checksum=values[3] if len(values) > 3 and isinstance(values[3], str) else None,
                    tradable=values[4] if len(values) > 4 else True,
                )
        if len(values) >= 2:
            tuple_session = _date_like(values[0])
            if symbol_hint is not None and tuple_session is not None:
                return PriceObservation(
                    symbol=symbol_hint,
                    session=tuple_session,
                    adjusted_close=values[1],
                    checksum=values[2] if len(values) > 2 and isinstance(values[2], str) else None,
                    tradable=values[3] if len(values) > 3 else True,
                )

    symbol_value = getattr(value, "symbol", symbol_hint)
    session_value = getattr(value, "session", session_hint)
    close_value = getattr(value, "adjusted_close", None)
    if close_value is None:
        close_value = getattr(value, "adj_close", None)
    if close_value is None:
        close_value = getattr(value, "close", None)
    if symbol_value is None or session_value is None or close_value is None:
        if symbol_hint is not None and session_hint is not None and _is_price_scalar(value):
            return PriceObservation(symbol_hint, session_hint, value)
        return None
    session = _date_like(session_value)
    if session is None:
        raise TypeError("history rows must contain a calendar-date session")
    checksum_value = getattr(value, "canonical_row_checksum", None)
    if checksum_value is None:
        checksum_value = getattr(value, "checksum", None)
    tradable_value = getattr(value, "tradable", True)
    return PriceObservation(
        symbol=symbol_value,
        session=session,
        adjusted_close=close_value,
        checksum=checksum_value if isinstance(checksum_value, str) else None,
        tradable=tradable_value,
    )


def _iter_history_values(
    history: HistoryInput,
) -> Iterable[tuple[object, str | None, date | None]]:
    """Yield raw values plus optional symbol/session hints from common shapes."""

    if isinstance(history, PriceHistory):
        yield from ((item, None, None) for item in history.observations)
        return
    if isinstance(history, Mapping):
        for outer_key, outer_value in history.items():
            if isinstance(outer_key, tuple) and len(outer_key) == 2:
                first_date = _date_like(outer_key[0])
                second_date = _date_like(outer_key[1])
                if isinstance(outer_key[0], str) and second_date is not None:
                    yield outer_value, outer_key[0], second_date
                    continue
                if first_date is not None and isinstance(outer_key[1], str):
                    yield outer_value, outer_key[1], first_date
                    continue
            outer_date = _date_like(outer_key)
            if outer_date is not None and isinstance(outer_value, Mapping):
                for symbol, value in outer_value.items():
                    yield value, str(symbol), outer_date
                continue
            if isinstance(outer_key, str):
                symbol_hint = outer_key
                if isinstance(outer_value, Mapping):
                    if any(
                        key in outer_value
                        for key in ("adjusted_close", "adj_close", "close", "price")
                    ):
                        yield outer_value, symbol_hint, None
                    else:
                        for inner_key, value in outer_value.items():
                            inner_date = _date_like(inner_key)
                            if inner_date is not None:
                                yield value, symbol_hint, inner_date
                            else:
                                yield value, symbol_hint, None
                elif isinstance(outer_value, Iterable) and not isinstance(
                    outer_value, (str, bytes, bytearray)
                ):
                    for value in outer_value:
                        yield value, symbol_hint, None
                else:
                    yield outer_value, symbol_hint, None
                continue
            yield outer_value, None, None
        return
    for value in history:
        yield value, None, None


def _normalize_history(
    history: HistoryInput,
) -> tuple[dict[tuple[str, date], PriceObservation], tuple[date, ...]]:
    by_key: dict[tuple[str, date], PriceObservation] = {}
    for value, symbol_hint, session_hint in _iter_history_values(history):
        observation = _observation_from_value(
            value, symbol_hint=symbol_hint, session_hint=session_hint
        )
        if observation is None:
            continue
        key = (observation.symbol, observation.session)
        previous = by_key.get(key)
        if previous is None or observation.sort_key() < previous.sort_key():
            by_key[key] = observation
    supplied_sessions = history.sessions if isinstance(history, PriceHistory) else ()
    return by_key, tuple(sorted(set(supplied_sessions)))


def _lookup_tradability(
    tradability: TradabilityInput,
    symbol: str,
    session: date,
    observation: PriceObservation | None,
) -> bool:
    if tradability is None:
        return observation is not None and observation.tradable
    if callable(tradability):
        return bool(tradability(symbol, session))
    pair = (symbol, session)
    if pair in tradability:
        return bool(tradability[pair])
    if symbol in tradability:
        value = tradability[symbol]
        if isinstance(value, Mapping):
            if session in value:
                return bool(value[session])
            session_text = session.isoformat()
            if session_text in value:
                return bool(value[session_text])
        elif isinstance(value, (set, frozenset, tuple, list)):
            return session in value or session.isoformat() in value
        elif isinstance(value, bool):
            return value
    if session in tradability:
        value = tradability[session]
        if isinstance(value, Mapping) and symbol in value:
            return bool(value[symbol])
        if isinstance(value, bool):
            return value
    return observation is not None and observation.tradable


def _derived_month_end_sessions(
    sessions: tuple[date, ...],
    *,
    calendar: StrategyCalendar | None,
) -> tuple[date, ...]:
    if not sessions:
        return ()
    if calendar is not None:
        selected = calendar.month_end_sessions(sessions[0], sessions[-1])
        return tuple(sorted({_require_date("signal_session", session) for session in selected}))
    month_ends: dict[tuple[int, int], date] = {}
    for session in sessions:
        month_ends[(session.year, session.month)] = session
    return tuple(sorted(month_ends.values()))


def monthly_momentum_v1(
    history: HistoryInput,
    signal_session: date | Iterable[date] | None = None,
    universe: Iterable[str] | None = None,
    position_count: int | MomentumStrategyParameters | None = None,
    *,
    params: MomentumStrategyParameters | None = None,
    sessions: Iterable[date] | None = None,
    signal_sessions: Iterable[date] | None = None,
    tradability: TradabilityInput = None,
    asset_tradability: TradabilityInput = None,
    calendar: StrategyCalendar | None = None,
    portfolio: object | None = None,
) -> tuple[StrategyDecision, ...]:
    """Return complete deterministic decisions for one or more month ends.

    ``history`` may be a :class:`PriceHistory`, normalized daily-bar-like
    objects, ``PriceObservation`` values, or compact mappings such as
    ``{"AAPL": {session: adjusted_close}}``.  ``sessions`` is useful when a
    fixture intentionally omits an entire session for every symbol: endpoint
    indexing then still follows the XNYS session sequence rather than the
    observed-row sequence.

    By default the function derives the final available XNYS session in each
    calendar month as a signal session.  Supplying ``signal_session`` or
    ``signal_sessions`` evaluates only those caller-selected month ends.  A
    253-session warm-up is enforced before scoring; no order-producing decision
    can be inferred from a shorter prefix.
    """

    # The application protocol in the design passes ``signal_session`` first
    # and a portfolio object before momentum parameters.  Keep that call shape
    # usable while the pure policy itself ignores portfolio state.
    if isinstance(history, date):
        protocol_signal = history
        protocol_history = signal_session
        if protocol_history is None or isinstance(protocol_history, date):
            raise TypeError("protocol-style calls require history after signal_session")
        protocol_universe = universe
        protocol_params = position_count
        history = cast(HistoryInput, protocol_history)
        signal_session = protocol_signal
        if isinstance(protocol_params, MomentumStrategyParameters):
            if params is not None and params != protocol_params:
                raise ValueError("conflicting momentum parameters")
            params = protocol_params
            position_count = None
        else:
            position_count = protocol_params
        if protocol_universe is not None and not isinstance(
            protocol_universe, (str, bytes)
        ) and not hasattr(protocol_universe, "__iter__"):
            # A protocol-style third positional argument is portfolio state,
            # not the configured symbol universe.
            universe = None
    elif isinstance(signal_session, MomentumStrategyParameters):
        if params is not None and params != signal_session:
            raise ValueError("conflicting momentum parameters")
        params = signal_session
        signal_session = None

    if params is not None and not isinstance(params, MomentumStrategyParameters):
        raise TypeError("params must be MomentumStrategyParameters or None")
    if isinstance(position_count, MomentumStrategyParameters):
        if params is not None and params != position_count:
            raise ValueError("position_count and params specify different strategy parameters")
        params = position_count
        position_count = None
    by_key, history_sessions = _normalize_history(history)

    if universe is None:
        if isinstance(history, PriceHistory) and history.universe:
            configured_symbols = history.universe
        else:
            configured_symbols = tuple(sorted({symbol for symbol, _ in by_key}))
    else:
        if isinstance(universe, (str, bytes)):
            raise TypeError("universe must be an iterable of symbol strings")
        configured_list: list[str] = []
        seen_symbols: set[str] = set()
        for symbol in universe:
            normalized = _normalize_symbol(symbol)
            if normalized in seen_symbols:
                raise ValueError("universe must not contain duplicate symbols")
            seen_symbols.add(normalized)
            configured_list.append(normalized)
        configured_symbols = tuple(configured_list)
    if not configured_symbols:
        raise ValueError("universe must contain at least one symbol")

    if params is None:
        if position_count is None:
            resolved_position_count = min(5, len(configured_symbols))
        else:
            resolved_position_count = _require_integer(
                "position_count", position_count, minimum=1
            )
        params = MomentumStrategyParameters(position_count=resolved_position_count)
    elif position_count is not None:
        resolved_position_count = _require_integer(
            "position_count", position_count, minimum=1
        )
        if resolved_position_count != params.position_count:
            raise ValueError("position_count and params.position_count must agree")
    if params.position_count > len(configured_symbols):
        raise ValueError("position_count must not exceed universe length")

    if tradability is not None and asset_tradability is not None:
        raise ValueError("tradability and asset_tradability are aliases; supply one")
    active_tradability = tradability if tradability is not None else asset_tradability

    explicit_sessions = tuple(
        sorted({_require_date("session", item) for item in (sessions or ())})
    )
    all_sessions = tuple(
        sorted(
            set(history_sessions)
            | {observation.session for observation in by_key.values()}
            | set(explicit_sessions)
        )
    )

    requested_signals: tuple[date, ...] | None
    if signal_sessions is not None:
        if signal_session is not None:
            raise ValueError("signal_session and signal_sessions are aliases; supply one")
        requested_signals = tuple(
            sorted({_require_date("signal_session", item) for item in signal_sessions})
        )
    elif signal_session is None:
        requested_signals = None
    elif isinstance(signal_session, date):
        requested_signals = (_require_date("signal_session", signal_session),)
    else:
        requested_signals = tuple(
            sorted({_require_date("signal_session", item) for item in signal_session})
        )

    if requested_signals is None:
        selected_signals = _derived_month_end_sessions(all_sessions, calendar=calendar)
    else:
        selected_signals = requested_signals
    if not selected_signals:
        return ()

    timeline = tuple(sorted(set(all_sessions) | set(selected_signals)))
    output: list[StrategyDecision] = []
    for signal in selected_signals:
        signal_index = timeline.index(signal)
        long_session = (
            timeline[signal_index - LONG_LOOKBACK_SESSIONS]
            if signal_index >= LONG_LOOKBACK_SESSIONS
            else None
        )
        short_session = (
            timeline[signal_index - SKIP_RECENT_SESSIONS]
            if signal_index >= SKIP_RECENT_SESSIONS
            else None
        )
        warmup_complete = signal_index >= WARM_UP_SESSIONS
        endpoint_rows: dict[str, tuple[date | None, PriceObservation | None]] = {}
        for symbol in configured_symbols:
            long_row = (
                by_key.get((symbol, long_session)) if long_session is not None else None
            )
            short_row = (
                by_key.get((symbol, short_session)) if short_session is not None else None
            )
            endpoint_rows[symbol] = (long_session, long_row)
            endpoint_rows[f"{symbol}:short"] = (short_session, short_row)

        preliminary: dict[str, dict[str, object]] = {}
        eligible_scores: list[tuple[str, Decimal]] = []
        for symbol in configured_symbols:
            long_date, long_row = endpoint_rows[symbol]
            short_date, short_row = endpoint_rows[f"{symbol}:short"]
            long_available = long_date is not None and long_row is not None
            short_available = short_date is not None and short_row is not None
            reason: StrategyExclusionReason | None = None
            score: Decimal | None = None
            if not warmup_complete:
                reason = StrategyExclusionReason.WARM_UP_INCOMPLETE
            elif not long_available:
                reason = StrategyExclusionReason.MISSING_LONG_ENDPOINT
            elif not short_available:
                reason = StrategyExclusionReason.MISSING_SHORT_ENDPOINT
            else:
                assert long_row is not None and short_row is not None
                with localcontext() as context:
                    context.prec = 28
                    score = short_row.adjusted_close / long_row.adjusted_close - Decimal("1")
                if not _lookup_tradability(active_tradability, symbol, signal, by_key.get((symbol, signal))):
                    reason = StrategyExclusionReason.ASSET_NOT_TRADABLE
                else:
                    eligible_scores.append((symbol, score))
            preliminary[symbol] = {
                "endpoint_252_close": long_row.adjusted_close if long_row else None,
                "endpoint_252_checksum": long_row.canonical_row_checksum if long_row else None,
                "endpoint_252_session": long_date if long_row else None,
                "endpoint_21_close": short_row.adjusted_close if short_row else None,
                "endpoint_21_checksum": short_row.canonical_row_checksum if short_row else None,
                "endpoint_21_session": short_date if short_row else None,
                "exclusion_reason": reason,
                "momentum_score": score,
            }

        eligible_scores.sort(key=lambda item: (-item[1], item[0]))
        ranks = {symbol: rank for rank, (symbol, _) in enumerate(eligible_scores, start=1)}
        selected = {symbol for symbol, _ in eligible_scores[: params.position_count]}
        target_weight = RationalWeight.equal_allocation(len(selected)) if selected else RationalWeight.zero()

        for symbol in configured_symbols:
            values = preliminary[symbol]
            reason = cast(StrategyExclusionReason | None, values["exclusion_reason"])
            is_eligible = symbol in ranks
            rank = ranks.get(symbol)
            if is_eligible and symbol not in selected:
                reason = StrategyExclusionReason.NOT_SELECTED
            if is_eligible and symbol in selected:
                reason = None
            output.append(
                StrategyDecision(
                    signal_session=signal,
                    symbol=symbol,
                    endpoint_252_session=cast(date | None, values["endpoint_252_session"]),
                    endpoint_252_close=cast(Decimal | None, values["endpoint_252_close"]),
                    endpoint_21_session=cast(date | None, values["endpoint_21_session"]),
                    endpoint_21_close=cast(Decimal | None, values["endpoint_21_close"]),
                    momentum_score=cast(Decimal | None, values["momentum_score"]),
                    eligible=is_eligible,
                    rank=rank,
                    target_weight=target_weight if symbol in selected else RationalWeight.zero(),
                    exclusion_reason=reason,
                    endpoint_252_checksum=cast(str | None, values["endpoint_252_checksum"]),
                    endpoint_21_checksum=cast(str | None, values["endpoint_21_checksum"]),
                )
            )
    return tuple(output)


# A descriptive alias makes the policy callable from code that uses strategy
# objects rather than function-level policies.
monthly_momentum_v1_decide = monthly_momentum_v1


__all__ = [
    "LONG_LOOKBACK_SESSIONS",
    "MomentumStrategyParameters",
    "PriceHistory",
    "PriceObservation",
    "RationalWeight",
    "SKIP_RECENT_SESSIONS",
    "STRATEGY_IDENTIFIER",
    "StrategyDecision",
    "StrategyExclusionReason",
    "WARM_UP_SESSIONS",
    "monthly_momentum_v1",
    "monthly_momentum_v1_decide",
]
