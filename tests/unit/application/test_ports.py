"""Focused offline tests for provider ports, batching, and retry orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest

from quant_research_platform.application.ports import (
    MarketDataProvider,
    RetryPolicy,
    build_provider_requests,
    classify_provider_failure,
    fetch_with_retry,
    ordered_symbol_batches,
    retry_delay_seconds,
)
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
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
)

_START = date(2024, 1, 2)
_END = date(2024, 1, 5)


class FakeClock:
    """Records deterministic retry delays without sleeping."""

    def __init__(self) -> None:
        self.delays: list[Decimal] = []

    def sleep(self, seconds: Decimal) -> None:
        self.delays.append(seconds)


ScriptStep = Callable[[ProviderRequest], ProviderBatchResult]


class ScriptedProvider:
    """A local provider whose responses are supplied one physical attempt at a time."""

    name = "fake"

    def __init__(self, script: tuple[ScriptStep, ...]) -> None:
        self._script = list(script)
        self.requests: list[ProviderRequest] = []

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        self.requests.append(request)
        if not self._script:
            raise AssertionError("provider was called more times than scripted")
        return self._script.pop(0)(request)


def _record(request: ProviderRequest, symbol: str) -> ProviderRecord:
    return ProviderRecord(
        provider=request.provider,
        request_content_key=request.content_key,
        symbol=symbol,
        raw_bar=RawDailyBar(
            provider_date=request.start,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
    )


def _failure(
    symbol: str,
    kind: ProviderFailureKind,
    reason: ProviderFailureReason,
) -> SymbolOutcome:
    category = (
        ErrorCategory.PROVIDER_RETRYABLE
        if kind is ProviderFailureKind.RETRYABLE
        else ErrorCategory.PROVIDER_TERMINAL
    )
    return SymbolOutcome(
        symbol=symbol,
        status=SymbolOutcomeStatus.FAILURE,
        attempts=1,
        failure_kind=kind,
        failure_reason=reason,
        errors=(
            ActionableError(
                operation="provider.fetch",
                category=category,
                message=f"The provider could not return daily data for {symbol}.",
                corrective_action="Review the symbol and retry only when appropriate.",
                symbol=symbol,
            ),
        ),
    )


def _result(
    request: ProviderRequest,
    states: dict[str, tuple[ProviderFailureKind, ProviderFailureReason] | None],
) -> ProviderBatchResult:
    outcomes: list[SymbolOutcome] = []
    for symbol in request.symbols:
        state = states[symbol]
        if state is None:
            outcomes.append(
                SymbolOutcome(
                    symbol=symbol,
                    status=SymbolOutcomeStatus.SUCCESS,
                    attempts=1,
                    records=(_record(request, symbol),),
                )
            )
        else:
            outcomes.append(_failure(symbol, *state))
    return ProviderBatchResult(request=request, outcomes=tuple(outcomes))


def _retryable_state() -> tuple[ProviderFailureKind, ProviderFailureReason]:
    return (ProviderFailureKind.RETRYABLE, ProviderFailureReason.RATE_LIMITED)


def test_requests_preserve_normalized_order_batch_at_ten_and_include_spy_once() -> None:
    requests = build_provider_requests(
        (" msft ", "aapl", "SPY", "AAPL", "xom"),
        start=_START,
        end=_END,
        batch_size=2,
        provider="fake",
    )

    assert tuple(symbol for request in requests for symbol in request.symbols) == (
        "MSFT",
        "AAPL",
        "SPY",
        "XOM",
    )
    assert all(1 <= len(request.symbols) <= 10 for request in requests)
    assert tuple(request.symbols for request in requests) == (
        ("MSFT", "AAPL"),
        ("SPY", "XOM"),
    )
    assert ordered_symbol_batches(("AAPL",), 10) == (("AAPL", "SPY"),)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (ProviderFailureReason.TIMEOUT, ProviderFailureKind.RETRYABLE),
        (ProviderFailureReason.SERVER_ERROR, ProviderFailureKind.RETRYABLE),
        (ProviderFailureReason.EMPTY_RESPONSE, ProviderFailureKind.TERMINAL),
        (ProviderFailureReason.UNEXPECTED, ProviderFailureKind.TERMINAL),
    ],
)
def test_failure_classification_has_retryable_and_terminal_defaults(
    reason: ProviderFailureReason, expected: ProviderFailureKind
) -> None:
    assert classify_provider_failure(reason) is expected


def test_retry_delays_are_capped_and_attempts_do_not_exceed_policy() -> None:
    request = ProviderRequest(("AAPL", "SPY"), _START, _END, provider="fake")
    provider = ScriptedProvider(
        tuple(
            lambda current: _result(
                current, {symbol: _retryable_state() for symbol in current.symbols}
            )
            for _ in range(4)
        )
    )
    assert isinstance(provider, MarketDataProvider)
    clock = FakeClock()
    policy = RetryPolicy(
        attempts=4,
        initial_delay_seconds=Decimal("1"),
        max_delay_seconds=Decimal("3"),
        backoff_multiplier=Decimal("2"),
    )

    result = fetch_with_retry(provider, request, policy, sleep=clock.sleep)

    assert [item.symbols for item in provider.requests] == [
        ("AAPL", "SPY"),
        ("AAPL", "SPY"),
        ("AAPL", "SPY"),
        ("AAPL", "SPY"),
    ]
    assert clock.delays == [Decimal("1"), Decimal("2"), Decimal("3")]
    assert [outcome.attempts for outcome in result.outcomes] == [4, 4]
    assert all(
        outcome.failure_kind is ProviderFailureKind.RETRYABLE
        for outcome in result.outcomes
    )
    assert retry_delay_seconds(policy, 2) == Decimal("1")
    assert retry_delay_seconds(policy, 4) == Decimal("3")


def test_terminal_outcome_stops_without_a_retry_delay() -> None:
    request = ProviderRequest(("AAPL",), _START, _END, provider="fake")
    provider = ScriptedProvider(
        (
            lambda current: _result(
                current,
                {
                    "AAPL": (
                        ProviderFailureKind.TERMINAL,
                        ProviderFailureReason.INVALID_SYMBOL,
                    )
                },
            ),
        )
    )
    clock = FakeClock()

    result = fetch_with_retry(
        provider,
        request,
        RetryPolicy(attempts=3),
        sleep=clock.sleep,
    )

    assert len(provider.requests) == 1
    assert clock.delays == []
    assert result.outcomes[0].attempts == 1
    assert result.outcomes[0].failure_kind is ProviderFailureKind.TERMINAL


def test_partial_success_keeps_outcomes_isolated_across_retries() -> None:
    request = ProviderRequest(("AAPL", "MSFT", "SPY"), _START, _END, provider="fake")
    provider = ScriptedProvider(
        (
            lambda current: _result(
                current,
                {
                    "AAPL": None,
                    "MSFT": _retryable_state(),
                    "SPY": (
                        ProviderFailureKind.TERMINAL,
                        ProviderFailureReason.EMPTY_RESPONSE,
                    ),
                },
            ),
            lambda current: _result(current, {"MSFT": None}),
        )
    )
    clock = FakeClock()

    result = fetch_with_retry(
        provider,
        request,
        RetryPolicy(
            attempts=3,
            initial_delay_seconds=Decimal("2"),
            max_delay_seconds=Decimal("8"),
            backoff_multiplier=Decimal("2"),
        ),
        sleep=clock.sleep,
    )

    assert [item.symbols for item in provider.requests] == [
        ("AAPL", "MSFT", "SPY"),
        ("MSFT",),
    ]
    assert clock.delays == [Decimal("2")]
    final_outcomes = [
        (outcome.symbol, outcome.status, outcome.attempts)
        for outcome in result.outcomes
    ]
    assert final_outcomes == [
        ("AAPL", SymbolOutcomeStatus.SUCCESS, 1),
        ("MSFT", SymbolOutcomeStatus.SUCCESS, 2),
        ("SPY", SymbolOutcomeStatus.FAILURE, 1),
    ]
    assert result.status == "partially_succeeded"
