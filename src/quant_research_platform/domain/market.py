"""Immutable market-data value objects with deterministic canonical identities.

The domain objects in this module deliberately carry no provider, calendar, or
storage implementation dependencies.  They preserve raw provider provenance,
make normalized session keys explicit, and keep volatile request timestamps in
separate operational records.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

from .canonical import CanonicalJSONValue, canonicalize, sha256_canonical_json
from .errors import (
    ActionableError,
    ProviderFailureKind,
    ProviderFailureReason,
    QuarantineReason,
    ValidationReason,
)

RAW_DATASET_SCHEMA_VERSION = "raw_v1"
DAILY_BAR_SCHEMA_VERSION = "daily_bar_v1"
QUARANTINE_SCHEMA_VERSION = "quarantine_v1"
GAP_SCHEMA_VERSION = "gap_v1"
VALIDATION_REPORT_SCHEMA_VERSION = "validation_report_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9.-]+$")

ReasonCode: TypeAlias = ValidationReason | QuarantineReason | str


def _required_text(field_name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def normalize_symbol(value: str) -> str:
    """Return the canonical ticker representation or reject an invalid symbol."""
    if not isinstance(value, str):
        raise TypeError("symbol must be a string")
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol must not be blank")
    if _SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError(
            "symbol must contain only uppercase letters, digits, '.', or '-'"
        )
    return symbol


def _date_only(field_name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _utc_timestamp(field_name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _checksum(field_name: str, value: str) -> str:
    normalized = _required_text(field_name, value)
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return normalized


def _coerce_decimal(
    field_name: str, value: Decimal | int | float | str | None
) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a Decimal-compatible number or None")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise TypeError(
            f"{field_name} must be a Decimal-compatible number or None"
        ) from error
    if not decimal_value.is_finite():
        raise ValueError(f"{field_name} must be finite when present")
    return decimal_value


def _non_negative_int(field_name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _freeze_json(value: CanonicalJSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_metadata(
    field_name: str, value: Mapping[str, object]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    canonical = canonicalize(value)
    if not isinstance(canonical, dict):  # Defensive: mappings canonicalize to dicts.
        raise AssertionError("canonical mapping expected")
    return cast(Mapping[str, object], _freeze_json(canonical))


def _canonical_metadata(value: Mapping[str, object]) -> dict[str, CanonicalJSONValue]:
    canonical = canonicalize(value)
    if not isinstance(canonical, dict):  # Defensive: mappings canonicalize to dicts.
        raise AssertionError("canonical mapping expected")
    return canonical


def _normalize_reason_code(value: ReasonCode) -> str:
    if isinstance(value, Enum):
        return _required_text("reason_code", str(value.value))
    return _required_text("reason_code", value)


@dataclass(frozen=True, slots=True, order=True)
class DateRange:
    """An inclusive date range shared by provider, gap, and manifest records."""

    start: date
    end: date

    def __post_init__(self) -> None:
        start = _date_only("start", self.start)
        end = _date_only("end", self.end)
        if start > end:
            raise ValueError("start must not be after end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def sort_key(self) -> tuple[date, date]:
        return (self.start, self.end)

    def to_content_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True, order=True)
class SessionKey:
    """The normalized-symbol/XNYS-session key for exactly one daily bar."""

    symbol: str
    session: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(self, "session", _date_only("session", self.session))

    def sort_key(self) -> tuple[str, date]:
        return (self.symbol, self.session)

    def to_content_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "session": self.session.isoformat()}


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """A bounded, inclusive request for daily records from one provider."""

    symbols: tuple[str, ...]
    start: date
    end: date
    provider: str = "yfinance"

    def __post_init__(self) -> None:
        if not isinstance(self.symbols, tuple):
            raise TypeError("symbols must be an immutable tuple")
        symbols = tuple(normalize_symbol(symbol) for symbol in self.symbols)
        if not 1 <= len(symbols) <= 10:
            raise ValueError("symbols must contain between 1 and 10 values")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be distinct after normalization")
        start = _date_only("start", self.start)
        end = _date_only("end", self.end)
        if start > end:
            raise ValueError("start must not be after end")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "provider", _required_text("provider", self.provider))

    @property
    def requested_range(self) -> DateRange:
        return DateRange(self.start, self.end)

    @property
    def content_key(self) -> str:
        return sha256_canonical_json(self.to_content_dict())

    def sort_key(self) -> tuple[str, date, date, tuple[str, ...]]:
        return (self.provider, self.start, self.end, self.symbols)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "symbols": list(self.symbols),
            "requested_range": self.requested_range.to_content_dict(),
        }


@dataclass(frozen=True, slots=True)
class RawDailyBar:
    """Unmodified daily observation fields returned by a provider."""

    provider_date: date
    open: Decimal | int | float | str | None = None
    high: Decimal | int | float | str | None = None
    low: Decimal | int | float | str | None = None
    close: Decimal | int | float | str | None = None
    adj_close: Decimal | int | float | str | None = None
    volume: Decimal | int | float | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_date", _date_only("provider_date", self.provider_date)
        )
        for field_name in ("open", "high", "low", "close", "adj_close", "volume"):
            object.__setattr__(
                self,
                field_name,
                _coerce_decimal(field_name, getattr(self, field_name)),
            )

    def to_content_dict(self) -> dict[str, object]:
        return {
            "provider_date": self.provider_date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adj_close": self.adj_close,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class RawCorporateAction:
    """Provider-reported, unadjusted action fields associated with a raw record."""

    dividend: Decimal | int | float | str | None = None
    split_ratio: Decimal | int | float | str | None = None
    provider_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dividend", _coerce_decimal("dividend", self.dividend))
        object.__setattr__(
            self, "split_ratio", _coerce_decimal("split_ratio", self.split_ratio)
        )
        object.__setattr__(
            self,
            "provider_fields",
            _freeze_metadata("provider_fields", self.provider_fields),
        )

    def to_content_dict(self) -> dict[str, object]:
        return {
            "dividend": self.dividend,
            "split_ratio": self.split_ratio,
            "provider_fields": _canonical_metadata(self.provider_fields),
        }


@dataclass(frozen=True, slots=True)
class RawLineage:
    """Stable reference from a normalized value back to preserved raw content."""

    provider: str
    request_content_key: str
    provider_record_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _required_text("provider", self.provider))
        object.__setattr__(
            self,
            "request_content_key",
            _checksum("request_content_key", self.request_content_key),
        )
        object.__setattr__(
            self,
            "provider_record_checksum",
            _checksum("provider_record_checksum", self.provider_record_checksum),
        )

    def sort_key(self) -> tuple[str, str, str]:
        return (self.provider, self.request_content_key, self.provider_record_checksum)

    def to_content_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "request_content_key": self.request_content_key,
            "provider_record_checksum": self.provider_record_checksum,
        }


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    """One preserved logical provider record and its request provenance."""

    provider: str
    request_content_key: str
    symbol: str
    raw_bar: RawDailyBar
    raw_action: RawCorporateAction = field(default_factory=RawCorporateAction)
    provider_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _required_text("provider", self.provider))
        object.__setattr__(
            self,
            "request_content_key",
            _checksum("request_content_key", self.request_content_key),
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.raw_bar, RawDailyBar):
            raise TypeError("raw_bar must be a RawDailyBar")
        if not isinstance(self.raw_action, RawCorporateAction):
            raise TypeError("raw_action must be a RawCorporateAction")
        object.__setattr__(
            self,
            "provider_fields",
            _freeze_metadata("provider_fields", self.provider_fields),
        )

    @property
    def provider_date(self) -> date:
        return self.raw_bar.provider_date

    @property
    def provider_record_checksum(self) -> str:
        return sha256_canonical_json(self.to_content_dict())

    @property
    def raw_lineage(self) -> RawLineage:
        return RawLineage(
            provider=self.provider,
            request_content_key=self.request_content_key,
            provider_record_checksum=self.provider_record_checksum,
        )

    def sort_key(self) -> tuple[str, date, str]:
        return (self.symbol, self.provider_date, self.provider_record_checksum)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "request_content_key": self.request_content_key,
            "symbol": self.symbol,
            "raw_bar": self.raw_bar.to_content_dict(),
            "raw_action": self.raw_action.to_content_dict(),
            "provider_fields": _canonical_metadata(self.provider_fields),
        }


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A canonical action preserved alongside one normalized daily-bar candidate."""

    symbol: str
    session: date
    dividend: Decimal | int | float | str = Decimal("0")
    split_ratio: Decimal | int | float | str = Decimal("1")
    raw_lineage: RawLineage | None = None
    source_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(self, "session", _date_only("session", self.session))
        dividend = _coerce_decimal("dividend", self.dividend)
        split_ratio = _coerce_decimal("split_ratio", self.split_ratio)
        if dividend is None or split_ratio is None:
            raise ValueError(
                "canonical corporate actions require dividend and split_ratio"
            )
        object.__setattr__(self, "dividend", dividend)
        object.__setattr__(self, "split_ratio", split_ratio)
        if self.raw_lineage is not None and not isinstance(
            self.raw_lineage, RawLineage
        ):
            raise TypeError("raw_lineage must be a RawLineage or None")
        if not isinstance(self.source_fields, tuple):
            raise TypeError("source_fields must be an immutable tuple")
        fields = tuple(
            sorted(
                {_required_text("source_field", value) for value in self.source_fields}
            )
        )
        object.__setattr__(self, "source_fields", fields)

    @property
    def session_key(self) -> SessionKey:
        return SessionKey(self.symbol, self.session)

    def sort_key(self) -> tuple[str, date, str]:
        lineage_checksum = (
            self.raw_lineage.provider_record_checksum if self.raw_lineage else ""
        )
        return (self.symbol, self.session, lineage_checksum)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "session": self.session.isoformat(),
            "dividend": self.dividend,
            "split_ratio": self.split_ratio,
            "raw_lineage": (
                self.raw_lineage.to_content_dict() if self.raw_lineage else None
            ),
            "source_fields": list(self.source_fields),
        }


@dataclass(frozen=True, slots=True)
class DailyBarCandidate:
    """A normalized candidate keyed by symbol/session before validation accepts it.

    Price and volume values deliberately permit ``None`` and invalid envelopes:
    validation owns those row rules and must be able to quarantine the original
    candidate rather than rejecting it at construction time.  Structural facts
    (UTC time, session key, policy version, and raw lineage) are enforced here.
    """

    symbol: str
    session: date
    event_timestamp: datetime
    raw_bar: RawDailyBar
    raw_action: RawCorporateAction
    corporate_action: CorporateAction
    adjusted_open: Decimal | int | float | str | None
    adjusted_high: Decimal | int | float | str | None
    adjusted_low: Decimal | int | float | str | None
    adjusted_close: Decimal | int | float | str | None
    adjusted_volume: Decimal | int | float | str | None
    execution_adjusted_open: Decimal | int | float | str | None
    sizing_adjusted_close: Decimal | int | float | str | None
    cumulative_price_factor: Decimal | int | float | str | None
    cumulative_split_factor: Decimal | int | float | str | None
    policy_version: str
    raw_lineage: RawLineage

    def __post_init__(self) -> None:
        symbol = normalize_symbol(self.symbol)
        session = _date_only("session", self.session)
        event_timestamp = _utc_timestamp("event_timestamp", self.event_timestamp)
        if not isinstance(self.raw_bar, RawDailyBar):
            raise TypeError("raw_bar must be a RawDailyBar")
        if not isinstance(self.raw_action, RawCorporateAction):
            raise TypeError("raw_action must be a RawCorporateAction")
        if not isinstance(self.corporate_action, CorporateAction):
            raise TypeError("corporate_action must be a CorporateAction")
        if self.corporate_action.session_key != SessionKey(symbol, session):
            raise ValueError(
                "corporate_action must use the candidate symbol and session"
            )
        if not isinstance(self.raw_lineage, RawLineage):
            raise TypeError("raw_lineage must be a RawLineage")
        if (
            self.corporate_action.raw_lineage is not None
            and self.corporate_action.raw_lineage != self.raw_lineage
        ):
            raise ValueError(
                "corporate_action raw_lineage must match candidate raw_lineage"
            )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "event_timestamp", event_timestamp)
        for field_name in (
            "adjusted_open",
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
            "adjusted_volume",
            "execution_adjusted_open",
            "sizing_adjusted_close",
            "cumulative_price_factor",
            "cumulative_split_factor",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_decimal(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "policy_version",
            _required_text("policy_version", self.policy_version),
        )

    @property
    def session_key(self) -> SessionKey:
        return SessionKey(self.symbol, self.session)

    @property
    def raw_open(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.open)

    @property
    def raw_high(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.high)

    @property
    def raw_low(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.low)

    @property
    def raw_close(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.close)

    @property
    def raw_volume(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.volume)

    @property
    def provider_adj_close(self) -> Decimal | None:
        return cast(Decimal | None, self.raw_bar.adj_close)

    @property
    def canonical_row_checksum(self) -> str:
        return sha256_canonical_json(self.to_content_dict())

    def sort_key(self) -> tuple[str, date, str]:
        return (self.symbol, self.session, self.canonical_row_checksum)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "session": self.session.isoformat(),
            "event_timestamp": self.event_timestamp,
            "raw_bar": self.raw_bar.to_content_dict(),
            "raw_action": self.raw_action.to_content_dict(),
            "corporate_action": self.corporate_action.to_content_dict(),
            "adjusted_open": self.adjusted_open,
            "adjusted_high": self.adjusted_high,
            "adjusted_low": self.adjusted_low,
            "adjusted_close": self.adjusted_close,
            "adjusted_volume": self.adjusted_volume,
            "execution_adjusted_open": self.execution_adjusted_open,
            "sizing_adjusted_close": self.sizing_adjusted_close,
            "cumulative_price_factor": self.cumulative_price_factor,
            "cumulative_split_factor": self.cumulative_split_factor,
            "policy_version": self.policy_version,
            "raw_lineage": self.raw_lineage.to_content_dict(),
        }


DailyBar = DailyBarCandidate


class SymbolOutcomeStatus(StrEnum):
    """The per-symbol result state for one provider batch."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class SymbolOutcome:
    """Independent records or a classified failure for one requested symbol."""

    symbol: str
    status: SymbolOutcomeStatus
    attempts: int
    records: tuple[ProviderRecord, ...] = ()
    failure_kind: ProviderFailureKind | None = None
    failure_reason: ProviderFailureReason | None = None
    errors: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        symbol = normalize_symbol(self.symbol)
        try:
            status = SymbolOutcomeStatus(self.status)
        except ValueError as error:
            raise ValueError(
                f"unsupported symbol outcome status: {self.status!r}"
            ) from error
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise TypeError("attempts must be an integer")
        if self.attempts < 1:
            raise ValueError("attempts must be at least one")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be an immutable tuple")
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be an immutable tuple")
        if any(not isinstance(record, ProviderRecord) for record in self.records):
            raise TypeError("records may contain only ProviderRecord values")
        if any(record.symbol != symbol for record in self.records):
            raise ValueError("records must belong to the outcome symbol")
        if any(not isinstance(error, ActionableError) for error in self.errors):
            raise TypeError("errors may contain only ActionableError values")

        records = tuple(sorted(self.records, key=ProviderRecord.sort_key))
        errors = tuple(sorted(self.errors, key=ActionableError.sort_key))
        failure_kind = (
            ProviderFailureKind(self.failure_kind)
            if self.failure_kind is not None
            else None
        )
        failure_reason = (
            ProviderFailureReason(self.failure_reason)
            if self.failure_reason is not None
            else None
        )
        if status is SymbolOutcomeStatus.SUCCESS:
            if not records:
                raise ValueError(
                    "successful outcomes must contain at least one ProviderRecord"
                )
            if failure_kind is not None or failure_reason is not None or errors:
                raise ValueError("successful outcomes must not contain failure details")
        else:
            if records:
                raise ValueError(
                    "failed outcomes must not contain ProviderRecord values"
                )
            if failure_kind is None or failure_reason is None or not errors:
                raise ValueError(
                    "failed outcomes require kind, reason, and actionable errors"
                )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "failure_kind", failure_kind)
        object.__setattr__(self, "failure_reason", failure_reason)

    def sort_key(self) -> tuple[str, str, int]:
        return (self.symbol, self.status.value, self.attempts)


@dataclass(frozen=True, slots=True)
class ProviderRequestMetadata:
    """Volatile, inspectable request history excluded from scientific identity."""

    request_content_key: str
    retrieved_at: datetime
    response_status: str
    request_id: str | None = None
    retrieval_started_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_content_key",
            _checksum("request_content_key", self.request_content_key),
        )
        object.__setattr__(
            self, "retrieved_at", _utc_timestamp("retrieved_at", self.retrieved_at)
        )
        if self.retrieval_started_at is not None:
            started_at = _utc_timestamp(
                "retrieval_started_at", self.retrieval_started_at
            )
            if started_at > self.retrieved_at:
                raise ValueError("retrieval_started_at must not be after retrieved_at")
            object.__setattr__(self, "retrieval_started_at", started_at)
        object.__setattr__(
            self,
            "response_status",
            _required_text("response_status", self.response_status),
        )
        if self.request_id is not None:
            object.__setattr__(
                self, "request_id", _required_text("request_id", self.request_id)
            )

    def to_operational_dict(self) -> dict[str, object]:
        return {
            "request_content_key": self.request_content_key,
            "retrieved_at": self.retrieved_at,
            "response_status": self.response_status,
            "request_id": self.request_id,
            "retrieval_started_at": self.retrieval_started_at,
        }


@dataclass(frozen=True, slots=True)
class ProviderBatchResult:
    """All independent per-symbol outcomes for one bounded provider request."""

    request: ProviderRequest
    outcomes: tuple[SymbolOutcome, ...]
    operational_metadata: ProviderRequestMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProviderRequest):
            raise TypeError("request must be a ProviderRequest")
        if not isinstance(self.outcomes, tuple):
            raise TypeError("outcomes must be an immutable tuple")
        if any(not isinstance(outcome, SymbolOutcome) for outcome in self.outcomes):
            raise TypeError("outcomes may contain only SymbolOutcome values")
        outcome_symbols = tuple(outcome.symbol for outcome in self.outcomes)
        if outcome_symbols != self.request.symbols:
            raise ValueError(
                "outcomes must contain one result per requested symbol in request order"
            )
        if self.operational_metadata is not None:
            if not isinstance(self.operational_metadata, ProviderRequestMetadata):
                raise TypeError(
                    "operational_metadata must be ProviderRequestMetadata or None"
                )
            if (
                self.operational_metadata.request_content_key
                != self.request.content_key
            ):
                raise ValueError(
                    "operational metadata must match the request content key"
                )

    @property
    def status(self) -> str:
        failures = sum(
            outcome.status is SymbolOutcomeStatus.FAILURE for outcome in self.outcomes
        )
        if failures == 0:
            return "succeeded"
        if failures == len(self.outcomes):
            return "failed"
        return "partially_succeeded"

    @property
    def successful_records(self) -> tuple[ProviderRecord, ...]:
        return tuple(record for outcome in self.outcomes for record in outcome.records)


class QuarantineSourceKind(StrEnum):
    """The source shape retained with a deterministic quarantine decision."""

    PROVIDER_RECORD = "provider_record"
    DAILY_BAR_CANDIDATE = "daily_bar_candidate"


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """An immutable rejection decision and canonical offending values."""

    source_kind: QuarantineSourceKind
    reason_codes: tuple[ReasonCode, ...]
    offending_values: Mapping[str, object]
    schema_version: str = QUARANTINE_SCHEMA_VERSION
    policy_version: str | None = None
    symbol: str | None = None
    session: date | None = None
    raw_lineage: RawLineage | None = None
    candidate_checksum: str | None = None

    def __post_init__(self) -> None:
        try:
            source_kind = QuarantineSourceKind(self.source_kind)
        except ValueError as error:
            raise ValueError(
                f"unsupported quarantine source kind: {self.source_kind!r}"
            ) from error
        if not isinstance(self.reason_codes, tuple):
            raise TypeError("reason_codes must be an immutable tuple")
        reason_codes = tuple(
            _normalize_reason_code(value) for value in self.reason_codes
        )
        if not reason_codes:
            raise ValueError("reason_codes must not be empty")
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("reason_codes must not contain duplicates")
        symbol = normalize_symbol(self.symbol) if self.symbol is not None else None
        session = (
            _date_only("session", self.session) if self.session is not None else None
        )
        if (symbol is None) != (session is None):
            raise ValueError("symbol and session must be supplied together when known")
        if self.raw_lineage is not None and not isinstance(
            self.raw_lineage, RawLineage
        ):
            raise TypeError("raw_lineage must be a RawLineage or None")
        candidate_checksum = (
            _checksum("candidate_checksum", self.candidate_checksum)
            if self.candidate_checksum is not None
            else None
        )
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(
            self,
            "offending_values",
            _freeze_metadata("offending_values", self.offending_values),
        )
        object.__setattr__(
            self,
            "schema_version",
            _required_text("schema_version", self.schema_version),
        )
        if self.policy_version is not None:
            object.__setattr__(
                self,
                "policy_version",
                _required_text("policy_version", self.policy_version),
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "candidate_checksum", candidate_checksum)

    @property
    def primary_reason(self) -> str:
        return self.reason_codes[0]

    def sort_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.source_kind.value,
            self.symbol or "",
            self.session.isoformat() if self.session else "",
            self.primary_reason,
            self.candidate_checksum or "",
            (
                self.raw_lineage.provider_record_checksum
                if self.raw_lineage is not None
                else ""
            ),
        )

    def to_content_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind.value,
            "symbol": self.symbol,
            "session": self.session.isoformat() if self.session else None,
            "reason_codes": list(self.reason_codes),
            "offending_values": _canonical_metadata(self.offending_values),
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "raw_lineage": (
                self.raw_lineage.to_content_dict()
                if self.raw_lineage is not None
                else None
            ),
            "candidate_checksum": self.candidate_checksum,
        }


@dataclass(frozen=True, slots=True, order=True)
class DataGap:
    """One expected session for which validation accepted no daily bar."""

    symbol: str
    expected_session: date
    requested_range: DateRange
    parent_retained: bool = False
    reason: str = "missing_accepted_bar"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(
            self,
            "expected_session",
            _date_only("expected_session", self.expected_session),
        )
        if not isinstance(self.requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange")
        if not isinstance(self.parent_retained, bool):
            raise TypeError("parent_retained must be a bool")
        object.__setattr__(self, "reason", _required_text("reason", self.reason))

    @property
    def session_key(self) -> SessionKey:
        return SessionKey(self.symbol, self.expected_session)

    def sort_key(self) -> tuple[str, date, str]:
        return (self.symbol, self.expected_session, self.reason)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "expected_session": self.expected_session.isoformat(),
            "requested_range": self.requested_range.to_content_dict(),
            "parent_retained": self.parent_retained,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SymbolValidationSummary:
    """Deterministic validation counts and coverage state for one requested symbol."""

    symbol: str
    accepted_count: int
    quarantined_count: int
    duplicate_count: int
    gap_count: int
    stale: bool = False
    staleness_lag_sessions: int = 0
    failed: bool = False
    retained_parent_coverage: bool = False
    covered_range: DateRange | None = None
    comparison_ready: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        for field_name in (
            "accepted_count",
            "quarantined_count",
            "duplicate_count",
            "gap_count",
            "staleness_lag_sessions",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(field_name, getattr(self, field_name)),
            )
        for field_name in (
            "stale",
            "failed",
            "retained_parent_coverage",
            "comparison_ready",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        if not self.stale and self.staleness_lag_sessions != 0:
            raise ValueError("staleness_lag_sessions must be zero when stale is false")
        if self.covered_range is not None and not isinstance(
            self.covered_range, DateRange
        ):
            raise TypeError("covered_range must be a DateRange or None")
        if self.accepted_count == 0 and self.covered_range is not None:
            raise ValueError("covered_range requires at least one accepted row")

    def sort_key(self) -> str:
        return self.symbol

    def to_content_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "accepted_count": self.accepted_count,
            "quarantined_count": self.quarantined_count,
            "duplicate_count": self.duplicate_count,
            "gap_count": self.gap_count,
            "stale": self.stale,
            "staleness_lag_sessions": self.staleness_lag_sessions,
            "failed": self.failed,
            "retained_parent_coverage": self.retained_parent_coverage,
            "covered_range": (
                self.covered_range.to_content_dict() if self.covered_range else None
            ),
            "comparison_ready": self.comparison_ready,
        }


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """The compact deterministic validation projection embedded in manifests."""

    accepted_row_count: int
    quarantined_row_count: int
    collapsed_duplicate_count: int
    gap_count: int
    failed_symbols: tuple[str, ...] = ()
    retained_parent_coverage_symbols: tuple[str, ...] = ()
    stale_symbols: tuple[str, ...] = ()
    covered_range: DateRange | None = None
    comparison_ready: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "accepted_row_count",
            "quarantined_row_count",
            "collapsed_duplicate_count",
            "gap_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(field_name, getattr(self, field_name)),
            )
        for field_name in (
            "failed_symbols",
            "retained_parent_coverage_symbols",
            "stale_symbols",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be an immutable tuple")
            normalized = tuple(sorted({normalize_symbol(value) for value in values}))
            object.__setattr__(self, field_name, normalized)
        if self.covered_range is not None and not isinstance(
            self.covered_range, DateRange
        ):
            raise TypeError("covered_range must be a DateRange or None")
        if not isinstance(self.comparison_ready, bool):
            raise TypeError("comparison_ready must be a bool")

    def to_content_dict(self) -> dict[str, object]:
        return {
            "accepted_row_count": self.accepted_row_count,
            "quarantined_row_count": self.quarantined_row_count,
            "collapsed_duplicate_count": self.collapsed_duplicate_count,
            "gap_count": self.gap_count,
            "failed_symbols": list(self.failed_symbols),
            "retained_parent_coverage_symbols": list(
                self.retained_parent_coverage_symbols
            ),
            "stale_symbols": list(self.stale_symbols),
            "covered_range": (
                self.covered_range.to_content_dict() if self.covered_range else None
            ),
            "comparison_ready": self.comparison_ready,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Deterministic facts for accepted, rejected, missing, and stale data."""

    per_symbol: tuple[SymbolValidationSummary, ...]
    quarantined_by_reason: tuple[tuple[str, int], ...]
    gaps: tuple[DataGap, ...]
    schema_version: str = VALIDATION_REPORT_SCHEMA_VERSION
    calendar_version: str = "xnys"

    def __post_init__(self) -> None:
        if not isinstance(self.per_symbol, tuple):
            raise TypeError("per_symbol must be an immutable tuple")
        if not isinstance(self.quarantined_by_reason, tuple):
            raise TypeError("quarantined_by_reason must be an immutable tuple")
        if not isinstance(self.gaps, tuple):
            raise TypeError("gaps must be an immutable tuple")
        if any(
            not isinstance(item, SymbolValidationSummary) for item in self.per_symbol
        ):
            raise TypeError(
                "per_symbol may contain only SymbolValidationSummary values"
            )
        if any(not isinstance(gap, DataGap) for gap in self.gaps):
            raise TypeError("gaps may contain only DataGap values")

        symbols = tuple(sorted(self.per_symbol, key=SymbolValidationSummary.sort_key))
        if len({item.symbol for item in symbols}) != len(symbols):
            raise ValueError("per_symbol must contain one summary per symbol")
        gaps = tuple(sorted(self.gaps, key=DataGap.sort_key))
        if len({gap.session_key for gap in gaps}) != len(gaps):
            raise ValueError("gaps must contain at most one value per SessionKey")

        reason_counts: list[tuple[str, int]] = []
        seen_reasons: set[str] = set()
        for reason, count in self.quarantined_by_reason:
            normalized_reason = _required_text("quarantine reason", reason)
            if normalized_reason in seen_reasons:
                raise ValueError("quarantined_by_reason must not repeat a reason")
            seen_reasons.add(normalized_reason)
            reason_counts.append(
                (normalized_reason, _non_negative_int("quarantine reason count", count))
            )
        reason_counts.sort(key=lambda item: item[0])

        if sum(item.gap_count for item in symbols) != len(gaps):
            raise ValueError(
                "per-symbol gap counts must equal the supplied gap records"
            )
        if {gap.symbol for gap in gaps} - {item.symbol for item in symbols}:
            raise ValueError("every gap must belong to a symbol validation summary")

        object.__setattr__(self, "per_symbol", symbols)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "quarantined_by_reason", tuple(reason_counts))
        object.__setattr__(
            self,
            "schema_version",
            _required_text("schema_version", self.schema_version),
        )
        object.__setattr__(
            self,
            "calendar_version",
            _required_text("calendar_version", self.calendar_version),
        )

    @property
    def summary(self) -> ValidationSummary:
        ranges = [item.covered_range for item in self.per_symbol if item.covered_range]
        covered_range = (
            DateRange(
                min(item.start for item in ranges),
                max(item.end for item in ranges),
            )
            if ranges
            else None
        )
        return ValidationSummary(
            accepted_row_count=sum(item.accepted_count for item in self.per_symbol),
            quarantined_row_count=sum(
                item.quarantined_count for item in self.per_symbol
            ),
            collapsed_duplicate_count=sum(
                item.duplicate_count for item in self.per_symbol
            ),
            gap_count=len(self.gaps),
            failed_symbols=tuple(
                item.symbol for item in self.per_symbol if item.failed
            ),
            retained_parent_coverage_symbols=tuple(
                item.symbol for item in self.per_symbol if item.retained_parent_coverage
            ),
            stale_symbols=tuple(item.symbol for item in self.per_symbol if item.stale),
            covered_range=covered_range,
            comparison_ready=all(item.comparison_ready for item in self.per_symbol),
        )

    @property
    def content_checksum(self) -> str:
        return sha256_canonical_json(self.to_content_dict())

    def to_content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calendar_version": self.calendar_version,
            "per_symbol": [item.to_content_dict() for item in self.per_symbol],
            "quarantined_by_reason": [
                {"reason": reason, "count": count}
                for reason, count in self.quarantined_by_reason
            ],
            "gaps": [gap.to_content_dict() for gap in self.gaps],
            "summary": self.summary.to_content_dict(),
        }


__all__ = [
    "DAILY_BAR_SCHEMA_VERSION",
    "GAP_SCHEMA_VERSION",
    "QUARANTINE_SCHEMA_VERSION",
    "RAW_DATASET_SCHEMA_VERSION",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "CorporateAction",
    "DailyBar",
    "DailyBarCandidate",
    "DataGap",
    "DateRange",
    "ProviderBatchResult",
    "ProviderRecord",
    "ProviderRequest",
    "ProviderRequestMetadata",
    "QuarantineRecord",
    "QuarantineSourceKind",
    "RawCorporateAction",
    "RawDailyBar",
    "RawLineage",
    "SessionKey",
    "SymbolOutcome",
    "SymbolOutcomeStatus",
    "SymbolValidationSummary",
    "ValidationReport",
    "ValidationSummary",
    "normalize_symbol",
]
