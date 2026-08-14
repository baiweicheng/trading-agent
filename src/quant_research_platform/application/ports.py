"""Application ports and deterministic provider orchestration helpers.

The provider port models one physical request attempt.  Retrying is deliberately
kept at the application boundary so the adapter remains a narrow translation
of provider responses and tests can control time without network I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Protocol, TypeAlias, runtime_checkable

from ..config.models import RetryPolicyConfig
from ..domain.errors import ProviderFailureKind, ProviderFailureReason
from ..domain.market import (
    ProviderBatchResult,
    ProviderRequest,
    SymbolOutcome,
    SymbolOutcomeStatus,
    normalize_symbol,
)

RetryPolicy: TypeAlias = RetryPolicyConfig
"""The validated configuration policy used by deterministic retry helpers."""

RetryClock = Callable[[Decimal], None]
"""Sleeper seam used to make retry timing deterministic in application tests."""

_RETRYABLE_FAILURE_REASONS = frozenset(
    {
        ProviderFailureReason.TIMEOUT,
        ProviderFailureReason.CONNECTION_RESET,
        ProviderFailureReason.RATE_LIMITED,
        ProviderFailureReason.SERVER_ERROR,
    }
)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Narrow boundary for one daily-market-data provider request attempt.

    Implementations receive an already-normalized, bounded request and must
    return exactly one independent outcome per requested symbol in the same
    order.  They must not perform retry scheduling; :func:`fetch_with_retry`
    applies the configured deterministic policy around this one-attempt port.
    """

    @property
    def name(self) -> str:
        """Stable provider identifier used for request provenance."""

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        """Fetch one bounded inclusive daily-data request attempt."""


def classify_provider_failure(reason: ProviderFailureReason) -> ProviderFailureKind:
    """Classify a stable provider reason without exposing adapter exceptions."""

    normalized_reason = ProviderFailureReason(reason)
    if normalized_reason in _RETRYABLE_FAILURE_REASONS:
        return ProviderFailureKind.RETRYABLE
    return ProviderFailureKind.TERMINAL


def retry_delay_seconds(policy: RetryPolicy, attempt_number: int) -> Decimal:
    """Return the capped, no-jitter delay immediately before ``attempt_number``.

    Attempt numbering starts at one.  Therefore the first retry, attempt two,
    waits ``initial_delay_seconds``; each later retry follows the configured
    multiplier and remains capped at ``max_delay_seconds``.
    """

    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise TypeError("attempt_number must be an integer")
    if not 2 <= attempt_number <= policy.attempts:
        raise ValueError("attempt_number must identify a permitted retry attempt")

    exponent = attempt_number - 2
    delay = policy.initial_delay_seconds * (policy.backoff_multiplier**exponent)
    return min(policy.max_delay_seconds, delay)


def ordered_symbol_batches(
    symbols: Iterable[str],
    batch_size: int,
    *,
    benchmark_symbol: str = "SPY",
) -> tuple[tuple[str, ...], ...]:
    """Normalize, first-occurrence deduplicate, and batch symbols with SPY once.

    The configured universe's order is preserved.  The benchmark is appended
    only when it is absent after normalization, ensuring every logical request
    has exactly one independent outcome for each symbol it asks for.
    """

    if isinstance(symbols, str):
        raise TypeError("symbols must be an iterable of symbol strings, not a string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if not 1 <= batch_size <= 10:
        raise ValueError("batch_size must be between 1 and 10")

    ordered: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)

    if not ordered:
        raise ValueError("symbols must contain at least one normalized symbol")

    benchmark = normalize_symbol(benchmark_symbol)
    if benchmark not in seen:
        ordered.append(benchmark)

    return tuple(
        tuple(ordered[index : index + batch_size])
        for index in range(0, len(ordered), batch_size)
    )


def build_provider_requests(
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    batch_size: int,
    provider: str = "yfinance",
    benchmark_symbol: str = "SPY",
) -> tuple[ProviderRequest, ...]:
    """Create deterministic bounded inclusive requests for a universe and SPY."""

    return tuple(
        ProviderRequest(batch, start, end, provider=provider)
        for batch in ordered_symbol_batches(
            symbols,
            batch_size,
            benchmark_symbol=benchmark_symbol,
        )
    )


def _with_total_attempts(outcome: SymbolOutcome, attempts: int) -> SymbolOutcome:
    """Record the logical batch attempt count without altering outcome content."""

    return replace(outcome, attempts=attempts)


def fetch_with_retry(
    provider: MarketDataProvider,
    request: ProviderRequest,
    policy: RetryPolicy,
    *,
    sleep: RetryClock,
) -> ProviderBatchResult:
    """Fetch a batch with per-symbol retry isolation and bounded no-jitter delays.

    Successful and terminal outcomes are final immediately.  Only symbols with
    retryable failures remain in subsequent bounded requests, so a terminal
    outcome is never attempted again and a successful symbol cannot be changed
    by another symbol's later failure.  The returned outcomes retain the order
    of the initial logical request and record the attempt on which each became
    final.
    """

    pending_symbols = request.symbols
    final_outcomes: dict[str, SymbolOutcome] = {}

    for attempt_number in range(1, policy.attempts + 1):
        attempt_request = ProviderRequest(
            pending_symbols,
            request.start,
            request.end,
            provider=request.provider,
        )
        attempt_result = provider.fetch_daily(attempt_request)
        if attempt_result.request != attempt_request:
            raise ValueError("provider result request must match the attempt request")
        if any(outcome.attempts != 1 for outcome in attempt_result.outcomes):
            raise ValueError("one-attempt provider outcomes must report attempts=1")

        next_pending: list[str] = []
        for outcome in attempt_result.outcomes:
            final_outcome = _with_total_attempts(outcome, attempt_number)
            if outcome.status is SymbolOutcomeStatus.SUCCESS:
                final_outcomes[outcome.symbol] = final_outcome
                continue

            if outcome.failure_kind is ProviderFailureKind.TERMINAL:
                final_outcomes[outcome.symbol] = final_outcome
                continue

            if attempt_number == policy.attempts:
                final_outcomes[outcome.symbol] = final_outcome
                continue

            next_pending.append(outcome.symbol)

        if not next_pending:
            break

        next_attempt = attempt_number + 1
        sleep(retry_delay_seconds(policy, next_attempt))
        pending_symbols = tuple(next_pending)

    missing_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in final_outcomes
    )
    if missing_symbols:
        raise RuntimeError(
            "provider retry loop ended without outcomes for "
            f"{', '.join(missing_symbols)}"
        )

    return ProviderBatchResult(
        request=request,
        outcomes=tuple(final_outcomes[symbol] for symbol in request.symbols),
    )


def fetch_batched_daily(
    provider: MarketDataProvider,
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    batch_size: int,
    policy: RetryPolicy,
    sleep: RetryClock,
    benchmark_symbol: str = "SPY",
) -> tuple[ProviderBatchResult, ...]:
    """Fetch every deterministic universe/SPY batch sequentially under one policy."""

    requests = build_provider_requests(
        symbols,
        start=start,
        end=end,
        batch_size=batch_size,
        provider=provider.name,
        benchmark_symbol=benchmark_symbol,
    )
    return tuple(
        fetch_with_retry(provider, request, policy, sleep=sleep)
        for request in requests
    )


__all__ = [
    "MarketDataProvider",
    "ProviderBatchResult",
    "ProviderRequest",
    "RetryClock",
    "RetryPolicy",
    "SymbolOutcome",
    "SymbolOutcomeStatus",
    "build_provider_requests",
    "classify_provider_failure",
    "fetch_batched_daily",
    "fetch_with_retry",
    "ordered_symbol_batches",
    "retry_delay_seconds",
]
