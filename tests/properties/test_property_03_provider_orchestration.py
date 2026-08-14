"""Property test for bounded provider batching and retry isolation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Final

from hypothesis import given, settings, strategies as st

from quant_research_platform.application.ports import (
    MarketDataProvider,
    RetryPolicy,
    fetch_batched_daily,
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

_START: Final = date(2024, 1, 2)
_END: Final = date(2024, 1, 5)
_SYMBOLS: Final = (
    "AAPL",
    "AMZN",
    "AVGO",
    "BAC",
    "BRK.B",
    "COST",
    "CVX",
    "DIS",
    "GOOG",
    "HD",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "MA",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "ORCL",
    "PEP",
    "PG",
    "SPY",
    "TSLA",
    "UNH",
    "XOM",
)


class AttemptState(str, Enum):
    """One local fake-provider response for one symbol attempt."""

    SUCCESS = "success"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ProviderCase:
    """All generated inputs required for one orchestration execution."""

    universe: tuple[str, ...]
    batch_size: int
    policy: RetryPolicy
    attempt_states: Mapping[str, tuple[AttemptState, ...]]


@dataclass(frozen=True)
class ExpectedOutcome:
    """Independent reference-model summary of one final symbol result."""

    symbol: str
    status: SymbolOutcomeStatus
    attempts: int
    record_count: int
    failure_kind: ProviderFailureKind | None
    failure_reason: ProviderFailureReason | None


@dataclass(frozen=True)
class ReferenceResult:
    """Expected physical calls and logical outcomes for one generated case."""

    logical_batches: tuple[tuple[str, ...], ...]
    attempt_requests: tuple[tuple[str, ...], ...]
    delays: tuple[Decimal, ...]
    final_outcomes: tuple[ExpectedOutcome, ...]
    batch_statuses: tuple[str, ...]
    aggregate_status: str


class FakeClock:
    """Records deterministic retry waits without invoking real time or I/O."""

    def __init__(self) -> None:
        self.delays: list[Decimal] = []

    def sleep(self, seconds: Decimal) -> None:
        self.delays.append(seconds)


class ScriptedProvider:
    """Local provider that maps each physical request to generated outcomes."""

    name = "fake"

    def __init__(self, attempt_states: Mapping[str, tuple[AttemptState, ...]]) -> None:
        self._attempt_states = attempt_states
        self.requests: list[ProviderRequest] = []
        self.attempt_count_by_symbol: dict[str, int] = {
            symbol: 0 for symbol in attempt_states
        }
        self.terminal_request_index: dict[str, int] = {}

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        request_index = len(self.requests)
        self.requests.append(request)
        outcomes: list[SymbolOutcome] = []

        for symbol in request.symbols:
            attempt_index = self.attempt_count_by_symbol[symbol]
            states = self._attempt_states[symbol]
            if attempt_index >= len(states):
                raise AssertionError(f"provider was called beyond scripted attempts for {symbol}")
            self.attempt_count_by_symbol[symbol] = attempt_index + 1
            state = states[attempt_index]
            if state is AttemptState.SUCCESS:
                outcomes.append(_success_outcome(request, symbol))
            else:
                if state is AttemptState.TERMINAL:
                    self.terminal_request_index[symbol] = request_index
                outcomes.append(_failure_outcome(symbol, state))

        return ProviderBatchResult(request=request, outcomes=tuple(outcomes))


def _success_outcome(request: ProviderRequest, symbol: str) -> SymbolOutcome:
    """Return one valid successful fake response with preserved raw data."""

    record = ProviderRecord(
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
    return SymbolOutcome(
        symbol=symbol,
        status=SymbolOutcomeStatus.SUCCESS,
        attempts=1,
        records=(record,),
    )


def _failure_outcome(symbol: str, state: AttemptState) -> SymbolOutcome:
    """Return one classified, actionable fake failure response."""

    if state is AttemptState.RETRYABLE:
        kind = ProviderFailureKind.RETRYABLE
        reason = ProviderFailureReason.RATE_LIMITED
        category = ErrorCategory.PROVIDER_RETRYABLE
    elif state is AttemptState.TERMINAL:
        kind = ProviderFailureKind.TERMINAL
        reason = ProviderFailureReason.INVALID_SYMBOL
        category = ErrorCategory.PROVIDER_TERMINAL
    else:
        raise AssertionError("success states must create successful outcomes")

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
                message=f"The fake provider could not return daily data for {symbol}.",
                corrective_action="Review the generated provider response.",
                symbol=symbol,
            ),
        ),
    )


def _reference_logical_symbols(universe: Sequence[str]) -> tuple[str, ...]:
    """Independently union the ordered universe with the required benchmark once."""

    return (*universe, "SPY") if "SPY" not in universe else tuple(universe)


def _reference_batches(case: ProviderCase) -> tuple[tuple[str, ...], ...]:
    """Slice the independently derived logical symbol order into fixed batches."""

    symbols = _reference_logical_symbols(case.universe)
    return tuple(
        tuple(symbols[index : index + case.batch_size])
        for index in range(0, len(symbols), case.batch_size)
    )


def _reference_outcome(
    symbol: str, state: AttemptState, attempts: int
) -> ExpectedOutcome:
    """Map a terminal fake state to the public outcome summary."""

    if state is AttemptState.SUCCESS:
        return ExpectedOutcome(
            symbol=symbol,
            status=SymbolOutcomeStatus.SUCCESS,
            attempts=attempts,
            record_count=1,
            failure_kind=None,
            failure_reason=None,
        )
    if state is AttemptState.RETRYABLE:
        return ExpectedOutcome(
            symbol=symbol,
            status=SymbolOutcomeStatus.FAILURE,
            attempts=attempts,
            record_count=0,
            failure_kind=ProviderFailureKind.RETRYABLE,
            failure_reason=ProviderFailureReason.RATE_LIMITED,
        )
    return ExpectedOutcome(
        symbol=symbol,
        status=SymbolOutcomeStatus.FAILURE,
        attempts=attempts,
        record_count=0,
        failure_kind=ProviderFailureKind.TERMINAL,
        failure_reason=ProviderFailureReason.INVALID_SYMBOL,
    )


def _status(outcomes: Sequence[ExpectedOutcome]) -> str:
    """Classify success, failure, and partial success without using production code."""

    failure_count = sum(
        outcome.status is SymbolOutcomeStatus.FAILURE for outcome in outcomes
    )
    if failure_count == 0:
        return "succeeded"
    if failure_count == len(outcomes):
        return "failed"
    return "partially_succeeded"


def _reference_result(case: ProviderCase) -> ReferenceResult:
    """Execute the batching and retry specification as a small independent model."""

    batches = _reference_batches(case)
    attempt_requests: list[tuple[str, ...]] = []
    delays: list[Decimal] = []
    final_outcomes: list[ExpectedOutcome] = []
    batch_statuses: list[str] = []

    for batch in batches:
        pending_symbols = batch
        final_by_symbol: dict[str, ExpectedOutcome] = {}
        for attempt_number in range(1, case.policy.attempts + 1):
            attempt_requests.append(pending_symbols)
            next_pending: list[str] = []

            for symbol in pending_symbols:
                state = case.attempt_states[symbol][attempt_number - 1]
                if state is AttemptState.SUCCESS or state is AttemptState.TERMINAL:
                    final_by_symbol[symbol] = _reference_outcome(
                        symbol, state, attempt_number
                    )
                elif attempt_number == case.policy.attempts:
                    final_by_symbol[symbol] = _reference_outcome(
                        symbol, state, attempt_number
                    )
                else:
                    next_pending.append(symbol)

            if not next_pending:
                break

            delay = case.policy.initial_delay_seconds * (
                case.policy.backoff_multiplier ** (attempt_number - 1)
            )
            delays.append(min(case.policy.max_delay_seconds, delay))
            pending_symbols = tuple(next_pending)

        batch_outcomes = tuple(final_by_symbol[symbol] for symbol in batch)
        final_outcomes.extend(batch_outcomes)
        batch_statuses.append(_status(batch_outcomes))

    return ReferenceResult(
        logical_batches=batches,
        attempt_requests=tuple(attempt_requests),
        delays=tuple(delays),
        final_outcomes=tuple(final_outcomes),
        batch_statuses=tuple(batch_statuses),
        aggregate_status=_status(final_outcomes),
    )


def _outcome_signature(outcome: SymbolOutcome) -> ExpectedOutcome:
    """Project a production outcome to the independent reference shape."""

    return ExpectedOutcome(
        symbol=outcome.symbol,
        status=outcome.status,
        attempts=outcome.attempts,
        record_count=len(outcome.records),
        failure_kind=outcome.failure_kind,
        failure_reason=outcome.failure_reason,
    )


def _aggregate_status(results: Sequence[ProviderBatchResult]) -> str:
    """Model ingestion's aggregate provider status across all logical batches."""

    outcomes = tuple(outcome for result in results for outcome in result.outcomes)
    expected_outcomes = tuple(_outcome_signature(outcome) for outcome in outcomes)
    return _status(expected_outcomes)


@st.composite
def provider_cases(draw: st.DrawFn) -> ProviderCase:
    """Generate bounded universes, policies, and one outcome script per symbol."""

    universe = tuple(
        draw(
            st.lists(
                st.sampled_from(_SYMBOLS),
                min_size=1,
                max_size=25,
                unique=True,
            )
        )
    )
    attempts = draw(st.integers(min_value=1, max_value=5))
    initial_delay = draw(st.decimals(min_value=0, max_value=30, places=2))
    max_delay = draw(st.decimals(min_value=initial_delay, max_value=60, places=2))
    policy = RetryPolicy(
        attempts=attempts,
        initial_delay_seconds=initial_delay,
        max_delay_seconds=max_delay,
        backoff_multiplier=draw(st.decimals(min_value=1, max_value=4, places=2)),
    )
    attempt_states = {
        symbol: tuple(
            draw(
                st.lists(
                    st.sampled_from(tuple(AttemptState)),
                    min_size=attempts,
                    max_size=attempts,
                )
            )
        )
        for symbol in _reference_logical_symbols(universe)
    }
    return ProviderCase(
        universe=universe,
        batch_size=draw(st.integers(min_value=1, max_value=10)),
        policy=policy,
        attempt_states=attempt_states,
    )


# Feature: quantitative-research-platform, Property 3: Bounded provider batching, retry, and symbol isolation
# Validates: Requirements 3.1, 3.4–3.11, 3.15–3.18, 14.9, 15.1, 17.16–17.17.
@settings(max_examples=100, deadline=None)
@given(case=provider_cases())
def test_bounded_provider_batching_retry_and_symbol_isolation(case: ProviderCase) -> None:
    """The local orchestration matches batching and retry reference semantics."""

    expected = _reference_result(case)
    provider = ScriptedProvider(case.attempt_states)
    assert isinstance(provider, MarketDataProvider)
    clock = FakeClock()

    actual = fetch_batched_daily(
        provider,
        case.universe,
        start=_START,
        end=_END,
        batch_size=case.batch_size,
        policy=case.policy,
        sleep=clock.sleep,
    )

    assert tuple(result.request.symbols for result in actual) == expected.logical_batches
    assert tuple(request.symbols for request in provider.requests) == (
        expected.attempt_requests
    )
    assert tuple(clock.delays) == expected.delays
    assert all(1 <= len(request.symbols) <= 10 for request in provider.requests)
    assert all(len(batch) <= 10 for batch in expected.logical_batches)

    actual_outcomes = tuple(
        outcome for result in actual for outcome in result.outcomes
    )
    assert tuple(_outcome_signature(outcome) for outcome in actual_outcomes) == (
        expected.final_outcomes
    )
    assert tuple(result.status for result in actual) == expected.batch_statuses
    assert _aggregate_status(actual) == expected.aggregate_status

    logical_symbols = _reference_logical_symbols(case.universe)
    assert tuple(outcome.symbol for outcome in actual_outcomes) == logical_symbols
    assert len(set(logical_symbols)) == len(logical_symbols)
    assert all(
        provider.attempt_count_by_symbol[symbol] <= case.policy.attempts
        for symbol in logical_symbols
    )
    assert all(
        provider.attempt_count_by_symbol[outcome.symbol] == outcome.attempts
        for outcome in actual_outcomes
    )

    for symbol, terminal_request_index in provider.terminal_request_index.items():
        assert all(
            symbol not in request.symbols
            for request in provider.requests[terminal_request_index + 1 :]
        )
