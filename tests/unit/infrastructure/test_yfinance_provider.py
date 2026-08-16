"""Offline contract-style tests for the narrow yfinance adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import (
    ProviderFailureKind,
    ProviderFailureReason,
)
from quant_research_platform.domain.market import ProviderRequest, SymbolOutcomeStatus
from quant_research_platform.infrastructure.yfinance_provider import YFinanceAdapter

_START = date(2024, 1, 2)
_END = date(2024, 1, 3)


def _request(symbols: tuple[str, ...] = ("AAPL",)) -> ProviderRequest:
    return ProviderRequest(symbols, _START, _END)


def _daily_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000, 1_100],
            "Dividends": [0.0, 0.25],
            "Stock Splits": [0.0, 1.0],
            "Capital Gains": [0.0, 0.1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    frame.attrs["fixture_source"] = "local"
    return frame


def _adapter(
    download: Callable[..., object], *, redactor: Redactor | None = None
) -> YFinanceAdapter:
    timestamps = iter(
        (
            datetime(2024, 1, 4, 14, tzinfo=UTC),
            datetime(2024, 1, 4, 14, 0, 1, tzinfo=UTC),
        )
    )
    return YFinanceAdapter(
        download=download,
        now=lambda: next(timestamps),
        redactor=redactor,
    )


def test_single_symbol_success_uses_exact_options_and_preserves_raw_fields() -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> object:
        calls.append(kwargs)
        return _daily_frame()

    request = _request()
    result = _adapter(download).fetch_daily(request)

    assert calls == [
        {
            "tickers": "AAPL",
            "start": "2024-01-02",
            "end": "2024-01-04",
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
    ]
    assert result.status == "succeeded"
    assert result.operational_metadata is not None
    assert result.operational_metadata.request_content_key == request.content_key
    assert result.operational_metadata.response_status == "succeeded"
    assert result.operational_metadata.retrieval_started_at == datetime(
        2024, 1, 4, 14, tzinfo=UTC
    )
    assert result.operational_metadata.retrieved_at == datetime(
        2024, 1, 4, 14, 0, 1, tzinfo=UTC
    )

    outcome = result.outcomes[0]
    assert outcome.status is SymbolOutcomeStatus.SUCCESS
    assert [record.provider_date for record in outcome.records] == [_START, _END]
    second = outcome.records[1]
    assert second.provider == "yfinance"
    assert second.request_content_key == request.content_key
    assert second.raw_bar.open == Decimal("101.0")
    assert second.raw_bar.adj_close == Decimal("101.5")
    assert second.raw_action.dividend == Decimal("0.25")
    assert second.raw_action.split_ratio == Decimal("1.0")
    additional_fields = second.provider_fields["additional_fields"]
    frame_metadata = second.provider_fields["frame_metadata"]
    assert isinstance(additional_fields, Mapping)
    assert isinstance(frame_metadata, Mapping)
    assert additional_fields["Capital Gains"] == 0.1
    assert frame_metadata["fixture_source"] == "local"


def test_multi_symbol_columns_keep_success_and_empty_outcomes_independent() -> None:
    base = _daily_frame().drop(columns=["Capital Gains"])
    empty = base.copy().astype(float)
    empty.loc[:, :] = float("nan")
    frame = pd.concat({"AAPL": base, "MSFT": empty}, axis=1)
    frame.columns = frame.columns.swaplevel(0, 1)

    result = _adapter(lambda **_: frame).fetch_daily(_request(("AAPL", "MSFT")))

    aapl, msft = result.outcomes
    assert result.status == "partially_succeeded"
    assert aapl.status is SymbolOutcomeStatus.SUCCESS
    assert len(aapl.records) == 2
    assert msft.status is SymbolOutcomeStatus.FAILURE
    assert msft.failure_kind is ProviderFailureKind.TERMINAL
    assert msft.failure_reason is ProviderFailureReason.EMPTY_RESPONSE


def test_empty_and_schema_invalid_frames_are_terminal_per_requested_symbol() -> None:
    empty_result = _adapter(lambda **_: pd.DataFrame()).fetch_daily(
        _request(("AAPL", "MSFT"))
    )
    assert [outcome.failure_reason for outcome in empty_result.outcomes] == [
        ProviderFailureReason.EMPTY_RESPONSE,
        ProviderFailureReason.EMPTY_RESPONSE,
    ]
    assert all(
        outcome.failure_kind is ProviderFailureKind.TERMINAL
        for outcome in empty_result.outcomes
    )

    invalid_frame = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2024-01-02"])
    )
    invalid_result = _adapter(lambda **_: invalid_frame).fetch_daily(_request())
    assert (
        invalid_result.outcomes[0].failure_reason
        is ProviderFailureReason.SCHEMA_INVALID
    )
    assert invalid_result.outcomes[0].failure_kind is ProviderFailureKind.TERMINAL


class StatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_transport_statuses_are_classified_without_leaking_diagnostics() -> None:
    secret = "https://user:super-secret@example.test"
    retryable = _adapter(
        lambda **_: (_ for _ in ()).throw(StatusError(429, secret)),
        redactor=Redactor([secret]),
    ).fetch_daily(_request())
    retryable_outcome = retryable.outcomes[0]
    assert retryable_outcome.failure_kind is ProviderFailureKind.RETRYABLE
    assert retryable_outcome.failure_reason is ProviderFailureReason.RATE_LIMITED
    assert secret not in retryable_outcome.errors[0].format_for_display()

    terminal = _adapter(
        lambda **_: (_ for _ in ()).throw(StatusError(404, secret)),
        redactor=Redactor([secret]),
    ).fetch_daily(_request())
    terminal_outcome = terminal.outcomes[0]
    assert terminal_outcome.failure_kind is ProviderFailureKind.TERMINAL
    assert terminal_outcome.failure_reason is ProviderFailureReason.CLIENT_ERROR
    assert secret not in terminal_outcome.errors[0].format_for_display()
