"""Narrow, offline-testable adapter around :mod:`yfinance` daily downloads.

The adapter translates one bounded, inclusive platform request into exactly one
``yfinance.download`` call. Retry timing intentionally remains in
``application.ports``; this adapter only maps one provider attempt into one
independent outcome per requested symbol. The yfinance import is deliberately
deferred until a production request is executed, while tests inject a local
download callable and therefore never contact the network.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final

from quant_research_platform.application.ports import MarketDataProvider
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import (
    ActionableError,
    ErrorCategory,
    ProviderFailureKind,
    ProviderFailureReason,
)
from quant_research_platform.domain.market import (
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    ProviderRequestMetadata,
    RawCorporateAction,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
)

DownloadCallable = Callable[..., object]
Clock = Callable[[], datetime]
RawNumeric = Decimal | int | float | str | None

_PROVIDER_NAME: Final = "yfinance"
_REQUIRED_PRICE_FIELDS: Final = frozenset(
    {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
)
_STANDARD_FIELDS: Final = _REQUIRED_PRICE_FIELDS | {"Dividends", "Stock Splits"}
_FIELD_ALIASES: Final = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
    "dividends": "Dividends",
    "stock splits": "Stock Splits",
    "stocksplits": "Stock Splits",
}


class _SchemaError(ValueError):
    """A provider response cannot be mapped to the approved daily contract."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _download_with_yfinance(**kwargs: object) -> object:
    """Import yfinance only when a real adapter request is executed."""

    import yfinance  # type: ignore[import-untyped]

    return yfinance.download(**kwargs)


def _daily_download_arguments(request: ProviderRequest) -> dict[str, object]:
    """Return the sole approved yfinance request shape for daily raw data."""

    return {
        "tickers": " ".join(request.symbols),
        "start": request.start.isoformat(),
        "end": (request.end + timedelta(days=1)).isoformat(),
        "interval": "1d",
        "auto_adjust": False,
        "back_adjust": False,
        "actions": True,
        "repair": False,
        "keepna": True,
        "prepost": False,
        "rounding": False,
        "threads": False,
        "progress": False,
    }


def _is_missing(value: object) -> bool:
    """Treat common pandas/NumPy missing scalars as an absent provider value."""

    if value is None:
        return True
    try:
        unequal = value != value
    except (TypeError, ValueError):
        return False
    if isinstance(unequal, bool):
        return unequal
    item = getattr(unequal, "item", None)
    if callable(item):
        try:
            return bool(item())
        except (TypeError, ValueError):
            return False
    return False


def _provider_value(value: object, redactor: Redactor) -> object:
    """Convert pandas/NumPy scalar values to canonical-domain-safe primitives."""

    if _is_missing(value):
        return None
    if isinstance(value, str):
        return redactor.redact_text(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.isoformat()
        return value.astimezone(UTC)
    if isinstance(value, (bool, int, float, Decimal, date)):
        return value
    to_python_datetime = getattr(value, "to_pydatetime", None)
    if callable(to_python_datetime):
        converted = to_python_datetime()
        if converted is not value:
            return _provider_value(converted, redactor)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _provider_value(converted, redactor)
    if isinstance(value, Mapping):
        return {
            redactor.redact_text(str(key)): _provider_value(item, redactor)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_provider_value(item, redactor) for item in value]
    return redactor.redact_text(str(value))


def _raw_numeric(value: object, redactor: Redactor) -> RawNumeric:
    """Map a raw numeric provider scalar while rejecting invalid values."""

    converted = _provider_value(value, redactor)
    if converted is None:
        return None
    if isinstance(converted, bool) or not isinstance(
        converted, (Decimal, int, float, str)
    ):
        raise _SchemaError("A required provider numeric field was not numeric.")
    if isinstance(converted, float) and not math.isfinite(converted):
        raise _SchemaError("A required provider numeric field was non-finite.")
    try:
        numeric = Decimal(str(converted))
    except (InvalidOperation, ValueError) as error:
        raise _SchemaError(
            "A required provider numeric field was not numeric."
        ) from error
    if not numeric.is_finite():
        raise _SchemaError("A required provider numeric field was non-finite.")
    return converted


def _provider_date(value: object) -> date:
    """Extract a calendar date from a pandas-like frame index value."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    to_python_datetime = getattr(value, "to_pydatetime", None)
    if callable(to_python_datetime):
        converted = to_python_datetime()
        if isinstance(converted, datetime):
            return converted.date()
    to_date = getattr(value, "date", None)
    if callable(to_date):
        converted = to_date()
        if isinstance(converted, date) and not isinstance(converted, datetime):
            return converted
    raise _SchemaError("The provider response index did not contain calendar dates.")


def _canonical_field_name(value: object) -> str:
    text = " ".join(str(value).split())
    if not text:
        raise _SchemaError("The provider response contained a blank column name.")
    return _FIELD_ALIASES.get(text.casefold(), text)


def _columns_by_symbol(
    frame: object, request: ProviderRequest
) -> dict[str, list[tuple[object, str]]]:
    """Normalize yfinance single- and multi-symbol column arrangements."""

    columns = getattr(frame, "columns", None)
    if columns is None:
        raise _SchemaError("The provider response did not expose tabular columns.")
    try:
        labels = list(columns)
    except TypeError as error:
        raise _SchemaError(
            "The provider response columns were not iterable."
        ) from error

    grouped: dict[str, list[tuple[object, str]]] = {
        symbol: [] for symbol in request.symbols
    }
    if not labels:
        return grouped

    multi_symbol = any(isinstance(label, tuple) for label in labels)
    if not multi_symbol:
        if len(request.symbols) != 1:
            raise _SchemaError(
                "A multi-symbol request requires explicit per-symbol columns."
            )
        symbol = request.symbols[0]
        grouped[symbol] = [(label, _canonical_field_name(label)) for label in labels]
        return grouped

    if not all(isinstance(label, tuple) and label for label in labels):
        raise _SchemaError("The provider response mixed incompatible column shapes.")

    requested = set(request.symbols)
    for label in labels:
        symbol_positions = [
            position
            for position, part in enumerate(label)
            if str(part).strip().upper() in requested
        ]
        if not symbol_positions:
            # A provider may include an unrequested diagnostic column. It is
            # not evidence that any requested symbol had a successful outcome.
            continue
        if len(symbol_positions) != 1:
            raise _SchemaError("A provider column mapped to multiple symbols.")
        symbol_position = symbol_positions[0]
        symbol = str(label[symbol_position]).strip().upper()
        field_parts = [
            part for position, part in enumerate(label) if position != symbol_position
        ]
        if len(field_parts) != 1:
            raise _SchemaError("A provider column did not identify one field.")
        grouped[symbol].append((label, _canonical_field_name(field_parts[0])))

    for symbol, entries in grouped.items():
        fields = [field for _, field in entries]
        if len(fields) != len(set(fields)):
            raise _SchemaError(
                f"The provider response repeated a daily field for {symbol}."
            )
    return grouped


def _frame_metadata(frame: object, redactor: Redactor) -> dict[str, object]:
    """Preserve canonicalizable DataFrame metadata without exposing secrets."""

    attrs = getattr(frame, "attrs", None)
    if not isinstance(attrs, Mapping) or not attrs:
        return {}
    metadata = _provider_value(attrs, redactor)
    if not isinstance(metadata, dict):
        raise _SchemaError("The provider response metadata was not a mapping.")
    return metadata


def _cell(frame: object, row_number: int, column: object) -> object:
    """Read a scalar through the small pandas-compatible ``iloc`` seam."""

    iloc = getattr(frame, "iloc", None)
    if iloc is None:
        raise _SchemaError("The provider response did not support positional rows.")
    try:
        row = iloc[row_number]
        return row[column]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _SchemaError("The provider response row could not be read.") from error


def _failure_reason_from_exception(error: BaseException) -> ProviderFailureReason:
    """Map transport/status exceptions without retaining their raw diagnostics."""

    status_code: int | None = None
    for source in (error, getattr(error, "response", None)):
        raw_status = getattr(source, "status_code", None)
        if isinstance(raw_status, int) and not isinstance(raw_status, bool):
            status_code = raw_status
            break
    if status_code in {408, 429} or (status_code is not None and status_code >= 500):
        return (
            ProviderFailureReason.RATE_LIMITED
            if status_code == 429
            else ProviderFailureReason.SERVER_ERROR
        )
    if status_code is not None and 400 <= status_code < 500:
        return ProviderFailureReason.CLIENT_ERROR

    exception_name = type(error).__name__.casefold()
    if "timeout" in exception_name:
        return ProviderFailureReason.TIMEOUT
    if "connection" in exception_name or "reset" in exception_name:
        return ProviderFailureReason.CONNECTION_RESET
    if "rate" in exception_name and "limit" in exception_name:
        return ProviderFailureReason.RATE_LIMITED
    if "ticker" in exception_name or "symbol" in exception_name:
        return ProviderFailureReason.INVALID_SYMBOL
    return ProviderFailureReason.UNEXPECTED


def _failure(symbol: str, reason: ProviderFailureReason) -> SymbolOutcome:
    """Build one safe, classified failure outcome for an individual symbol."""

    retryable = reason in {
        ProviderFailureReason.TIMEOUT,
        ProviderFailureReason.CONNECTION_RESET,
        ProviderFailureReason.RATE_LIMITED,
        ProviderFailureReason.SERVER_ERROR,
    }
    kind = ProviderFailureKind.RETRYABLE if retryable else ProviderFailureKind.TERMINAL
    category = (
        ErrorCategory.PROVIDER_RETRYABLE
        if retryable
        else ErrorCategory.PROVIDER_TERMINAL
    )
    if reason is ProviderFailureReason.EMPTY_RESPONSE:
        message = (
            "yfinance returned no daily records for the inclusive requested range."
        )
        action = "Review the symbol and requested range before retrying."
    elif reason is ProviderFailureReason.SCHEMA_INVALID:
        message = "yfinance returned a daily response with an unsupported schema."
        action = "Review the provider response schema and update the adapter policy."
    elif retryable:
        message = (
            "The yfinance daily-data request encountered a retryable transport failure."
        )
        action = "Retry the request after the provider becomes available."
    else:
        message = (
            "The yfinance daily-data request could not be completed for this symbol."
        )
        action = "Review the symbol and provider request before retrying."
    return SymbolOutcome(
        symbol=symbol,
        status=SymbolOutcomeStatus.FAILURE,
        attempts=1,
        failure_kind=kind,
        failure_reason=reason,
        errors=(
            ActionableError(
                operation="provider.fetch_daily",
                category=category,
                message=message,
                corrective_action=action,
                symbol=symbol,
            ),
        ),
    )


class YFinanceAdapter(MarketDataProvider):
    """Daily yfinance adapter with deterministic options and offline test seam."""

    name: str = _PROVIDER_NAME

    def __init__(
        self,
        *,
        download: DownloadCallable | None = None,
        now: Clock = _utc_now,
        redactor: Redactor | None = None,
    ) -> None:
        self._download = download or _download_with_yfinance
        self._now = now
        self._redactor = redactor or Redactor()

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        """Fetch one request with the approved raw-daily yfinance options."""

        if request.provider != self.name:
            raise ValueError(
                "YFinanceAdapter requires "
                f"provider={self.name!r}, got {request.provider!r}"
            )

        started_at = self._now()
        try:
            frame = self._download(**_daily_download_arguments(request))
        except Exception as error:
            outcomes = tuple(
                _failure(symbol, _failure_reason_from_exception(error))
                for symbol in request.symbols
            )
            return self._batch_result(request, outcomes, started_at)

        try:
            outcomes = self._outcomes_from_frame(frame, request)
        except _SchemaError:
            outcomes = tuple(
                _failure(symbol, ProviderFailureReason.SCHEMA_INVALID)
                for symbol in request.symbols
            )
        return self._batch_result(request, outcomes, started_at)

    def _batch_result(
        self,
        request: ProviderRequest,
        outcomes: tuple[SymbolOutcome, ...],
        started_at: datetime,
    ) -> ProviderBatchResult:
        provisional = ProviderBatchResult(request=request, outcomes=outcomes)
        metadata = ProviderRequestMetadata(
            request_content_key=request.content_key,
            retrieval_started_at=started_at,
            retrieved_at=self._now(),
            response_status=provisional.status,
        )
        return ProviderBatchResult(
            request=request,
            outcomes=outcomes,
            operational_metadata=metadata,
        )

    def _outcomes_from_frame(
        self, frame: object, request: ProviderRequest
    ) -> tuple[SymbolOutcome, ...]:
        """Construct independent outcomes from a single/multi-symbol DataFrame."""

        if not hasattr(frame, "empty") or not hasattr(frame, "index"):
            raise _SchemaError("yfinance did not return a DataFrame-like response.")
        try:
            is_empty = bool(frame.empty)
        except (TypeError, ValueError) as error:
            raise _SchemaError(
                "The provider response had an invalid empty flag."
            ) from error
        if is_empty:
            return tuple(
                _failure(symbol, ProviderFailureReason.EMPTY_RESPONSE)
                for symbol in request.symbols
            )

        columns = _columns_by_symbol(frame, request)
        metadata = _frame_metadata(frame, self._redactor)
        try:
            index_values = list(frame.index)
        except TypeError as error:
            raise _SchemaError(
                "The provider response index was not iterable."
            ) from error
        if not index_values:
            return tuple(
                _failure(symbol, ProviderFailureReason.EMPTY_RESPONSE)
                for symbol in request.symbols
            )
        provider_dates = tuple(_provider_date(value) for value in index_values)

        outcomes: list[SymbolOutcome] = []
        for symbol in request.symbols:
            entries = columns[symbol]
            if not entries:
                outcomes.append(_failure(symbol, ProviderFailureReason.EMPTY_RESPONSE))
                continue
            fields = {field for _, field in entries}
            if not _REQUIRED_PRICE_FIELDS.issubset(fields):
                outcomes.append(_failure(symbol, ProviderFailureReason.SCHEMA_INVALID))
                continue
            try:
                records = self._records_for_symbol(
                    frame,
                    symbol,
                    entries,
                    provider_dates,
                    request,
                    metadata,
                )
            except _SchemaError:
                outcomes.append(_failure(symbol, ProviderFailureReason.SCHEMA_INVALID))
                continue
            if records:
                outcomes.append(
                    SymbolOutcome(
                        symbol=symbol,
                        status=SymbolOutcomeStatus.SUCCESS,
                        attempts=1,
                        records=records,
                    )
                )
            else:
                outcomes.append(_failure(symbol, ProviderFailureReason.EMPTY_RESPONSE))
        return tuple(outcomes)

    def _records_for_symbol(
        self,
        frame: object,
        symbol: str,
        entries: list[tuple[object, str]],
        provider_dates: tuple[date, ...],
        request: ProviderRequest,
        metadata: Mapping[str, object],
    ) -> tuple[ProviderRecord, ...]:
        """Preserve raw fields, actions, extras, and frame metadata for one symbol."""

        columns = {field: label for label, field in entries}
        records: list[ProviderRecord] = []
        for row_number, provider_date in enumerate(provider_dates):
            values = {
                field: _provider_value(_cell(frame, row_number, label), self._redactor)
                for field, label in columns.items()
            }
            raw_values = {
                field: _raw_numeric(values[field], self._redactor)
                for field in _REQUIRED_PRICE_FIELDS
            }
            if not any(value is not None for value in raw_values.values()):
                continue
            extra_fields = {
                field: value
                for field, value in values.items()
                if field not in _STANDARD_FIELDS
            }
            record_metadata: dict[str, object] = {}
            if extra_fields:
                record_metadata["additional_fields"] = extra_fields
            if metadata:
                record_metadata["frame_metadata"] = dict(metadata)

            try:
                record = ProviderRecord(
                    provider=self.name,
                    request_content_key=request.content_key,
                    symbol=symbol,
                    raw_bar=RawDailyBar(
                        provider_date=provider_date,
                        open=raw_values["Open"],
                        high=raw_values["High"],
                        low=raw_values["Low"],
                        close=raw_values["Close"],
                        adj_close=raw_values["Adj Close"],
                        volume=raw_values["Volume"],
                    ),
                    raw_action=RawCorporateAction(
                        dividend=_raw_numeric(values.get("Dividends"), self._redactor),
                        split_ratio=_raw_numeric(
                            values.get("Stock Splits"), self._redactor
                        ),
                    ),
                    provider_fields=record_metadata,
                )
            except (TypeError, ValueError) as error:
                raise _SchemaError(
                    "The provider row could not be represented as a raw record."
                ) from error
            records.append(record)
        return tuple(records)


YFinanceProvider = YFinanceAdapter


__all__ = [
    "Clock",
    "DownloadCallable",
    "YFinanceAdapter",
    "YFinanceProvider",
]
