"""Causal strategy-decision delivery and deterministic order intents.

This module is the application boundary between the verified snapshot and the
Zipline event loop.  It deliberately does not import Zipline or any storage
implementation.  A caller supplies a verified snapshot handle and a bounded
history reader; the reader is invoked with an inclusive signal-session cutoff,
then all sizing is performed from the rows returned by that read.

The important distinction in this module is between research prices and
execution/sizing prices.  ``monthly_momentum_v1`` ranks symbols with causal
research-adjusted closes, while order sizing and current-equity marking use the
same-session ``sizing_adjusted_close`` (the action-effective actual-share
coordinate).  No execution-session price is read here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from math import floor
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

from ..config.models import ResolvedConfig
from ..domain.errors import ActionableError, Err, ErrorCategory, Ok, Result
from ..domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    OrderRecord,
    OrderStatus,
    deterministic_order_id,
    quantize_money,
)
from ..domain.manifests import SnapshotManifest, VerifiedSnapshotHandle
from ..domain.strategy import (
    MomentumStrategyParameters,
    PriceHistory,
    PriceObservation,
    RationalWeight,
    STRATEGY_IDENTIFIER,
    StrategyDecision,
    StrategyExclusionReason,
    monthly_momentum_v1,
)


HistoryRow: TypeAlias = object
HistoryInput: TypeAlias = PriceHistory | Iterable[HistoryRow] | Mapping[object, object]
SnapshotInput: TypeAlias = VerifiedSnapshotHandle | SnapshotManifest | str | object


class SnapshotHistoryReader(Protocol):
    """Read-only, projected history access used by the decision service.

    Implementations must interpret ``end_session`` as an inclusive upper
    bound.  The service never asks this port for a future session.
    """

    def read_history(
        self,
        snapshot: VerifiedSnapshotHandle,
        *,
        symbols: tuple[str, ...],
        end_session: date,
        fields: tuple[str, ...],
    ) -> Iterable[HistoryRow]:
        """Return rows for ``symbols`` no later than ``end_session``."""


class DecisionSnapshotVerifier(Protocol):
    """Minimal snapshot verification port required when an ID is supplied."""

    def open_verified(self, snapshot_id: str) -> object:
        """Return a verified handle or an application ``Result``."""

    def inspect_snapshot(self, snapshot_id: str) -> object:
        """Optionally return a manifest-bearing inspection DTO."""


class DecisionPolicyVersion(StrEnum):
    """The current policy version recorded with decision inputs."""

    CAUSAL_FORWARD_V1 = "causal_forward_v1"


@dataclass(frozen=True, slots=True)
class DecisionRunInputs:
    """Deterministic strategy inputs that must be recorded before execution."""

    position_count: int
    long_lookback_sessions: int = 252
    skip_recent_sessions: int = 21
    policy_version: str = DecisionPolicyVersion.CAUSAL_FORWARD_V1.value
    strategy_identifier: str = STRATEGY_IDENTIFIER
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.position_count, bool) or not isinstance(self.position_count, int):
            raise TypeError("position_count must be an integer")
        if self.position_count < 1:
            raise ValueError("position_count must be at least one")
        if self.long_lookback_sessions != 252:
            raise ValueError("long_lookback_sessions must be fixed at 252")
        if self.skip_recent_sessions != 21:
            raise ValueError("skip_recent_sessions must be fixed at 21")
        for field_name in ("policy_version", "strategy_identifier"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
            object.__setattr__(self, field_name, " ".join(value.split()))
        if self.snapshot_id is not None:
            if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
                raise ValueError("snapshot_id must be a non-blank string or None")
            object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())

    @property
    def endpoint_offsets(self) -> tuple[int, int]:
        """Return the fixed long and recent endpoint offsets."""

        return (self.long_lookback_sessions, self.skip_recent_sessions)

    def to_serializable(self) -> dict[str, object]:
        """Return the non-operational run-input projection."""

        return {
            "long_lookback_sessions": self.long_lookback_sessions,
            "policy_version": self.policy_version,
            "position_count": self.position_count,
            "skip_recent_sessions": self.skip_recent_sessions,
            "strategy_identifier": self.strategy_identifier,
            "snapshot_id": self.snapshot_id,
        }


# Names used by the design and by callers that prefer the shorter terminology.
RunInputs = DecisionRunInputs
DecisionInputs = DecisionRunInputs


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A deterministic whole-share request queued for the next session.

    ``OrderIntent`` retains the sizing facts alongside the domain
    ``OrderRecord``.  This makes the order-sizing decision independently
    auditable while allowing the backtest adapter to consume the canonical
    ``OrderRecord`` representation.
    """

    order_id: str
    signal_session: date
    execution_session: date
    symbol: str
    requested_quantity: int
    ordinal: int
    target_shares: int
    current_shares: int
    sizing_price: Decimal
    portfolio_equity: Decimal
    decision_rank: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.signal_session, datetime) or not isinstance(self.signal_session, date):
            raise TypeError("signal_session must be a calendar date")
        if isinstance(self.execution_session, datetime) or not isinstance(self.execution_session, date):
            raise TypeError("execution_session must be a calendar date")
        if self.execution_session <= self.signal_session:
            raise ValueError("execution_session must be after signal_session")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-blank string")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        for field_name in ("requested_quantity", "ordinal", "target_shares", "current_shares"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.requested_quantity == 0:
            raise ValueError("requested_quantity must not be zero")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if self.target_shares < 0 or self.current_shares < 0:
            raise ValueError("share counts must be non-negative")
        if self.requested_quantity != self.target_shares - self.current_shares:
            raise ValueError("requested_quantity must equal target_shares - current_shares")
        if self.decision_rank is not None and (
            isinstance(self.decision_rank, bool)
            or not isinstance(self.decision_rank, int)
            or self.decision_rank < 1
        ):
            raise ValueError("decision_rank must be a positive integer or None")
        for field_name in ("sizing_price", "portfolio_equity"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{field_name} must be a finite positive Decimal")
        expected_id = deterministic_order_id(
            signal_session=self.signal_session,
            execution_session=self.execution_session,
            symbol=self.symbol,
            requested_quantity=self.requested_quantity,
            ordinal=self.ordinal,
        )
        if self.order_id != expected_id:
            raise ValueError("order_id does not match deterministic scientific inputs")

    @property
    def quantity(self) -> int:
        """Alias used by execution adapters."""

        return self.requested_quantity

    @property
    def target_delta(self) -> int:
        return self.requested_quantity

    def to_order_record(self) -> OrderRecord:
        """Convert the intent to the canonical pending order record."""

        return OrderRecord(
            order_id=self.order_id,
            signal_session=self.signal_session,
            execution_session=self.execution_session,
            symbol=self.symbol,
            requested_quantity=self.requested_quantity,
            ordinal=self.ordinal,
            decision_rank=self.decision_rank,
            status=OrderStatus.PENDING,
        )

    def to_serializable(self) -> dict[str, object]:
        return {
            "current_shares": self.current_shares,
            "decision_rank": self.decision_rank,
            "execution_session": self.execution_session,
            "order_id": self.order_id,
            "ordinal": self.ordinal,
            "portfolio_equity": self.portfolio_equity,
            "requested_quantity": self.requested_quantity,
            "signal_session": self.signal_session,
            "sizing_price": self.sizing_price,
            "symbol": self.symbol,
            "target_shares": self.target_shares,
        }


@dataclass(frozen=True, slots=True)
class DecisionBook:
    """Precomputed decisions that can be revealed only on their signal date."""

    decisions: tuple[StrategyDecision, ...]
    run_inputs: DecisionRunInputs

    def __post_init__(self) -> None:
        if not isinstance(self.decisions, tuple):
            raise TypeError("decisions must be an immutable tuple")
        if any(not isinstance(item, StrategyDecision) for item in self.decisions):
            raise TypeError("decisions must contain StrategyDecision values")
        if not isinstance(self.run_inputs, DecisionRunInputs):
            raise TypeError("run_inputs must be DecisionRunInputs")
        object.__setattr__(
            self,
            "decisions",
            tuple(self.decisions),
        )

    @property
    def signal_sessions(self) -> tuple[date, ...]:
        return tuple(sorted({item.signal_session for item in self.decisions}))

    def reveal(self, session: date) -> tuple[StrategyDecision, ...]:
        """Reveal only decisions whose signal session equals ``session``.

        Returning an empty tuple for every earlier or later session prevents an
        engine callback from accidentally using a future precomputed decision.
        """

        if isinstance(session, datetime) or not isinstance(session, date):
            raise TypeError("session must be a calendar date")
        return tuple(item for item in self.decisions if item.signal_session == session)

    decisions_for = reveal
    for_session = reveal

    def to_serializable(self) -> dict[str, object]:
        return {
            "decisions": [item.to_serializable() for item in self.decisions],
            "run_inputs": self.run_inputs.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class DecisionDeliveryResult:
    """Application result for one signal-session delivery."""

    snapshot_id: str | None
    signal_session: date
    decisions: tuple[StrategyDecision, ...]
    order_intents: tuple[OrderIntent, ...]
    run_inputs: DecisionRunInputs
    marked_equity: Decimal
    decision_book: DecisionBook

    def __post_init__(self) -> None:
        if isinstance(self.signal_session, datetime) or not isinstance(self.signal_session, date):
            raise TypeError("signal_session must be a calendar date")
        if not isinstance(self.decisions, tuple) or any(
            not isinstance(item, StrategyDecision) for item in self.decisions
        ):
            raise TypeError("decisions must be an immutable tuple of StrategyDecision values")
        if not isinstance(self.order_intents, tuple) or any(
            not isinstance(item, OrderIntent) for item in self.order_intents
        ):
            raise TypeError("order_intents must be an immutable tuple of OrderIntent values")
        if not isinstance(self.run_inputs, DecisionRunInputs):
            raise TypeError("run_inputs must be DecisionRunInputs")
        if not isinstance(self.decision_book, DecisionBook):
            raise TypeError("decision_book must be DecisionBook")
        if not isinstance(self.marked_equity, Decimal) or not self.marked_equity.is_finite() or self.marked_equity <= 0:
            raise ValueError("marked_equity must be a finite positive Decimal")
        if tuple(item.signal_session for item in self.decisions) != (self.signal_session,) * len(self.decisions):
            raise ValueError("all delivered decisions must use signal_session")
        if self.decision_book.reveal(self.signal_session) != self.decisions:
            raise ValueError("decision_book must contain the delivered decisions")

    @property
    def orders(self) -> tuple[OrderRecord, ...]:
        """Canonical order records consumed by a backtest adapter."""

        return tuple(item.to_order_record() for item in self.order_intents)

    @property
    def equity(self) -> Decimal:
        return self.marked_equity

    def to_serializable(self) -> dict[str, object]:
        return {
            "decisions": [item.to_serializable() for item in self.decisions],
            "marked_equity": self.marked_equity,
            "order_intents": [item.to_serializable() for item in self.order_intents],
            "run_inputs": self.run_inputs.to_serializable(),
            "signal_session": self.signal_session,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class _PortfolioSnapshot:
    holdings: Mapping[str, int]
    cash: Decimal
    supplied_equity: Decimal | None


class CausalDecisionDelivery:
    """Deliver causal decisions and create next-session whole-share intents.

    ``snapshot_reader`` is intentionally a narrow injected port.  It is called
    exactly once per ``deliver`` invocation with ``end_session=signal_session``
    unless explicit ``history`` is supplied.  A reader implementation can be a
    Parquet/DuckDB projection, but this application object never imports those
    frameworks and never asks for the execution session's prices.
    """

    _HISTORY_FIELDS: tuple[str, ...] = (
        "symbol",
        "session",
        "adjusted_close",
        "sizing_adjusted_close",
        "canonical_row_checksum",
        "tradable",
    )

    def __init__(
        self,
        snapshot_reader: SnapshotHistoryReader | object | None = None,
        *,
        snapshot_manager: DecisionSnapshotVerifier | None = None,
        snapshot_verifier: DecisionSnapshotVerifier | None = None,
        calendar: object | None = None,
        strategy: Callable[..., tuple[StrategyDecision, ...]] = monthly_momentum_v1,
        policy_version: str | None = None,
        resolved_config: ResolvedConfig | None = None,
    ) -> None:
        if snapshot_manager is not None and snapshot_verifier is not None:
            raise ValueError("supply snapshot_manager or snapshot_verifier, not both")
        self.snapshot_reader = snapshot_reader
        self.snapshot_manager = snapshot_manager or snapshot_verifier
        self.calendar = calendar
        self.strategy = strategy
        self.policy_version = policy_version
        self.resolved_config = resolved_config

    def prepare(
        self,
        snapshot: SnapshotInput,
        signal_sessions: Iterable[date],
        *,
        universe: Iterable[str] | None = None,
        position_count: int | None = None,
        history: HistoryInput | None = None,
    ) -> Result[DecisionBook]:
        """Precompute a signal-session decision book from causal prefixes.

        This method is useful to a Zipline adapter that precomputes all
        decisions before starting the event loop.  Each signal is still read
        independently through its own close, so later rows cannot influence an
        earlier decision.
        """

        try:
            sessions = tuple(sorted(set(_date_only(value, "signal_session") for value in signal_sessions)))
            if not sessions:
                raise ValueError("signal_sessions must contain at least one session")
            resolved = self._verify_snapshot(snapshot)
            resolved_universe, resolved_count = self._resolve_strategy_inputs(
                resolved, universe=universe, position_count=position_count
            )
            all_decisions: list[StrategyDecision] = []
            for signal in sessions:
                source = history if history is not None else self._read_history(
                    resolved,
                    signal,
                    resolved_universe,
                )
                decisions, _ = self._decisions_for_signal(
                    source,
                    signal,
                    resolved_universe,
                    resolved_count,
                )
                all_decisions.extend(decisions)
            inputs = self._run_inputs(
                resolved,
                position_count=resolved_count,
            )
            return Ok(DecisionBook(tuple(all_decisions), inputs))
        except _DecisionFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except (TypeError, ValueError) as error:
            return Err((self._input_error("decisions.prepare", error),), preserve_order=True)
        except Exception as error:
            return Err((ActionableError.from_unexpected_exception("decisions.prepare", error),))

    def deliver(
        self,
        snapshot: SnapshotInput,
        signal_session: date,
        portfolio: object | None = None,
        *,
        universe: Iterable[str] | None = None,
        position_count: int | None = None,
        history: HistoryInput | None = None,
        execution_session: date | None = None,
    ) -> Result[DecisionDeliveryResult]:
        """Deliver one signal-session decision and deterministic order intents."""

        try:
            signal = _date_only(signal_session, "signal_session")
            resolved = self._verify_snapshot(snapshot)
            resolved_universe, resolved_count = self._resolve_strategy_inputs(
                resolved, universe=universe, position_count=position_count
            )
            source = history if history is not None else self._read_history(
                resolved,
                signal,
                resolved_universe,
            )
            decisions, sizing_prices = self._decisions_for_signal(
                source,
                signal,
                resolved_universe,
                resolved_count,
            )
            inputs = self._run_inputs(resolved, position_count=resolved_count)
            book = DecisionBook(decisions, inputs)

            if self._is_warmup(decisions):
                marked_equity = self._mark_equity(
                    portfolio,
                    sizing_prices,
                    signal,
                    require_prices=False,
                )
                intents: tuple[OrderIntent, ...] = ()
            else:
                next_session = execution_session or self._next_session(signal)
                if next_session <= signal:
                    raise ValueError("execution_session must be after signal_session")
                intents, marked_equity = self._make_intents(
                    decisions,
                    portfolio,
                    sizing_prices,
                    signal,
                    next_session,
                )

            return Ok(
                DecisionDeliveryResult(
                    snapshot_id=inputs.snapshot_id,
                    signal_session=signal,
                    decisions=decisions,
                    order_intents=intents,
                    run_inputs=inputs,
                    marked_equity=marked_equity,
                    decision_book=book,
                )
            )
        except _DecisionFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except (TypeError, ValueError, ArithmeticError) as error:
            return Err((self._input_error("decisions.deliver", error),), preserve_order=True)
        except Exception as error:
            return Err((ActionableError.from_unexpected_exception("decisions.deliver", error),))

    def deliver_or_raise(self, *args: object, **kwargs: object) -> DecisionDeliveryResult:
        """Convenience form for local unit callers that prefer exceptions."""

        result = self.deliver(*cast(Any, args), **cast(Any, kwargs))
        if isinstance(result, Err):
            raise ValueError("; ".join(error.message for error in result.errors))
        return result.value

    def reveal(self, book: DecisionBook, session: date) -> tuple[StrategyDecision, ...]:
        """Reveal a prepared decision only at its exact signal session."""

        if not isinstance(book, DecisionBook):
            raise TypeError("book must be a DecisionBook")
        return book.reveal(session)

    decisions_for_session = reveal
    reveal_to_zipline = reveal
    decisions_for = reveal

    def _verify_snapshot(self, snapshot: SnapshotInput) -> VerifiedSnapshotHandle | SnapshotManifest | object:
        supplied_id = snapshot if isinstance(snapshot, str) else getattr(snapshot, "snapshot_id", None)
        if supplied_id is None:
            raise _DecisionFailure((
                ActionableError(
                    operation="decisions.snapshot",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    message="Decision delivery did not receive a verified snapshot.",
                    corrective_action="Pass a checksum-verified snapshot handle or a Snapshot_ID with a verifier.",
                    field_path="snapshot",
                ),
            ))
        if not isinstance(supplied_id, str) or not supplied_id.strip():
            raise _DecisionFailure((self._snapshot_error(),))
        if self.snapshot_manager is not None:
            opened = self.snapshot_manager.open_verified(supplied_id)
            resolved = _unwrap_result(opened, "snapshot.open")
            if not hasattr(resolved, "snapshot_id"):
                raise _DecisionFailure((self._snapshot_error(),))
            if getattr(resolved, "snapshot_id") != supplied_id:
                raise _DecisionFailure((self._snapshot_error(),))
            return resolved
        # A non-string object is assumed to be the result of an earlier
        # verification step.  The service cannot verify an opaque ID without
        # an injected manager, but it still requires the immutable ID field.
        if isinstance(snapshot, str):
            raise _DecisionFailure((
                ActionableError(
                    operation="decisions.snapshot",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    message="Decision delivery requires a verified snapshot handle.",
                    corrective_action="Open and verify the selected Snapshot_ID before delivering decisions.",
                    field_path="snapshot_id",
                ),
            ))
        return snapshot

    def _manifest_for(self, snapshot: object) -> SnapshotManifest | None:
        manifest = getattr(snapshot, "manifest", None)
        if isinstance(manifest, SnapshotManifest):
            return manifest
        if self.snapshot_manager is not None:
            snapshot_id = getattr(snapshot, "snapshot_id", None)
            inspector = getattr(self.snapshot_manager, "inspect_snapshot", None)
            if isinstance(snapshot_id, str) and callable(inspector):
                inspected = _unwrap_result(inspector(snapshot_id), "snapshot.inspect")
                candidate = getattr(inspected, "manifest", inspected)
                if isinstance(candidate, SnapshotManifest):
                    return candidate
        return None

    def _resolve_strategy_inputs(
        self,
        snapshot: object,
        *,
        universe: Iterable[str] | None,
        position_count: int | None,
    ) -> tuple[tuple[str, ...], int]:
        manifest = self._manifest_for(snapshot)
        if universe is None and self.resolved_config is not None:
            configured = self.resolved_config.data.universe
        elif universe is None and manifest is not None:
            configured = manifest.content_identity.configured_universe
        elif universe is None:
            raise ValueError("universe is required when no resolved configuration or manifest is available")
        else:
            configured = tuple(universe)
        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in configured:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("universe must contain non-blank symbols")
            value = symbol.strip().upper()
            if value in seen:
                raise ValueError("universe must contain distinct symbols")
            seen.add(value)
            normalized.append(value)
        if not normalized:
            raise ValueError("universe must contain at least one symbol")
        if position_count is None:
            if self.resolved_config is not None:
                count = self.resolved_config.strategy.position_count
            else:
                count = min(5, len(normalized))
        else:
            count = position_count
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(normalized):
            raise ValueError("position_count must be between 1 and the universe size")
        return tuple(normalized), count

    def _run_inputs(self, snapshot: object, *, position_count: int) -> DecisionRunInputs:
        manifest = self._manifest_for(snapshot)
        policy = self.policy_version
        if policy is None and manifest is not None:
            policy = manifest.content_identity.schema_versions.corporate_action_policy_version
        return DecisionRunInputs(
            snapshot_id=getattr(snapshot, "snapshot_id", None),
            position_count=position_count,
            policy_version=policy or DecisionPolicyVersion.CAUSAL_FORWARD_V1.value,
        )

    def _read_history(
        self,
        snapshot: object,
        signal_session: date,
        universe: tuple[str, ...],
    ) -> HistoryInput:
        reader = self.snapshot_reader
        if reader is None:
            raise _DecisionFailure((
                ActionableError(
                    operation="decisions.read_history",
                    category=ErrorCategory.STORAGE_IO,
                    message="No verified snapshot history reader is configured.",
                    corrective_action="Inject a projected reader for the selected snapshot.",
                    field_path="snapshot_reader",
                    session=signal_session,
                ),
            ))
        method: Callable[..., object] | None = None
        for name in ("read_history", "read_daily_bars", "read_bars", "read", "scan"):
            candidate = getattr(reader, name, None)
            if callable(candidate):
                method = cast(Callable[..., object], candidate)
                break
        if method is None and callable(reader):
            method = cast(Callable[..., object], reader)
        if method is None:
            raise TypeError("snapshot_reader must expose read_history(), scan(), or be callable")

        result = _call_reader(
            method,
            snapshot=snapshot,
            symbols=universe,
            end_session=signal_session,
            fields=self._HISTORY_FIELDS,
            start_session=self._history_start(signal_session),
        )
        result = _unwrap_result(result, "snapshot.history")
        if result is None:
            raise ValueError("snapshot history reader returned no rows")
        return cast(HistoryInput, result)

    def _history_start(self, signal_session: date) -> date:
        if self.calendar is not None:
            sessions_fn = getattr(self.calendar, "sessions", None)
            if callable(sessions_fn):
                try:
                    candidates = tuple(
                        sessions_fn(
                            signal_session - timedelta(days=700),
                            signal_session,
                            completed_at=datetime.max.replace(tzinfo=UTC),
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    candidates = ()
                normalized = tuple(
                    value for value in candidates if isinstance(value, date) and not isinstance(value, datetime) and value <= signal_session
                )
                if len(normalized) >= 254:
                    return normalized[-254]
        return signal_session - timedelta(days=700)

    def _decisions_for_signal(
        self,
        source: HistoryInput,
        signal_session: date,
        universe: tuple[str, ...],
        position_count: int,
    ) -> tuple[tuple[StrategyDecision, ...], Mapping[tuple[str, date], Decimal]]:
        observations, sizing_prices, sessions = _normalize_history(source, signal_session)
        if not observations:
            raise ValueError("verified snapshot has no history through signal_session")
        history = PriceHistory(
            observations=tuple(observations),
            sessions=sessions,
            universe=universe,
        )
        params = MomentumStrategyParameters(position_count=position_count)
        decisions = tuple(
            self.strategy(
                history,
                signal_session=signal_session,
                universe=universe,
                params=params,
            )
        )
        if tuple(item.symbol for item in decisions) != universe:
            by_symbol = {item.symbol: item for item in decisions}
            if set(by_symbol) != set(universe):
                raise ValueError("strategy must return exactly one decision per configured symbol")
            decisions = tuple(by_symbol[symbol] for symbol in universe)
        if len(decisions) != len(universe):
            raise ValueError("strategy must return exactly one decision per configured symbol")
        return decisions, sizing_prices

    def _next_session(self, signal_session: date) -> date:
        if self.calendar is None:
            raise ValueError("calendar is required to derive next-session execution")
        method = getattr(self.calendar, "next_session", None)
        if not callable(method):
            raise TypeError("calendar must expose next_session()")
        result = method(signal_session)
        return _date_only(result, "execution_session")

    @staticmethod
    def _is_warmup(decisions: Sequence[StrategyDecision]) -> bool:
        return bool(decisions) and all(
            item.exclusion_reason is StrategyExclusionReason.WARM_UP_INCOMPLETE
            for item in decisions
        )

    def _make_intents(
        self,
        decisions: tuple[StrategyDecision, ...],
        portfolio: object | None,
        sizing_prices: Mapping[tuple[str, date], Decimal],
        signal_session: date,
        execution_session: date,
    ) -> tuple[tuple[OrderIntent, ...], Decimal]:
        snapshot = _portfolio_snapshot(portfolio)
        marked_equity = self._mark_equity(portfolio, sizing_prices, signal_session)
        selected = {
            decision.symbol: decision
            for decision in decisions
            if decision.eligible and decision.target_weight != RationalWeight.zero()
        }
        target_shares: dict[str, int] = {}
        for symbol, selected_decision in selected.items():
            price = sizing_prices.get((symbol, signal_session))
            if price is None:
                raise ValueError(f"sizing_adjusted_close is unavailable for selected symbol {symbol}")
            target_shares[symbol] = _floor_target_shares(marked_equity, selected_decision.target_weight, price)

        all_symbols = set(snapshot.holdings) | set(selected)
        deltas: list[tuple[str, int, int, int | None, Decimal]] = []
        for symbol in all_symbols:
            current = snapshot.holdings.get(symbol, 0)
            target = target_shares.get(symbol, 0)
            delta = target - current
            if delta == 0:
                continue
            held_decision: StrategyDecision | None = selected.get(symbol)
            price = sizing_prices.get((symbol, signal_session))
            if price is None:
                raise ValueError(f"sizing_adjusted_close is unavailable for held symbol {symbol}")
            deltas.append((symbol, current, target, held_decision.rank if held_decision else None, price))

        # The order list is deterministic and also gives the later blotter the
        # intended sell-first / ranked-buy ordering without needing future data.
        deltas.sort(
            key=lambda item: (
                0 if item[2] - item[1] < 0 else 1,
                item[0] if item[2] - item[1] < 0 else (item[3] or len(decisions) + 1),
                item[0],
            )
        )
        intents: list[OrderIntent] = []
        for ordinal, (symbol, current, target, rank, price) in enumerate(deltas):
            quantity = target - current
            order_id = deterministic_order_id(
                signal_session=signal_session,
                execution_session=execution_session,
                symbol=symbol,
                requested_quantity=quantity,
                ordinal=ordinal,
            )
            intents.append(
                OrderIntent(
                    order_id=order_id,
                    signal_session=signal_session,
                    execution_session=execution_session,
                    symbol=symbol,
                    requested_quantity=quantity,
                    ordinal=ordinal,
                    target_shares=target,
                    current_shares=current,
                    sizing_price=price,
                    portfolio_equity=marked_equity,
                    decision_rank=rank,
                )
            )
        return tuple(intents), marked_equity

    @staticmethod
    def _mark_equity(
        portfolio: object | None,
        sizing_prices: Mapping[tuple[str, date], Decimal],
        signal_session: date,
        *,
        require_prices: bool = True,
    ) -> Decimal:
        snapshot = _portfolio_snapshot(portfolio)
        marked = snapshot.cash
        for symbol, quantity in snapshot.holdings.items():
            price = sizing_prices.get((symbol, signal_session))
            if price is None:
                if require_prices:
                    raise ValueError(f"sizing_adjusted_close is unavailable for held symbol {symbol}")
                continue
            marked += Decimal(quantity) * price
        if not snapshot.holdings and snapshot.supplied_equity is not None:
            marked = snapshot.supplied_equity
        if marked <= 0 or not marked.is_finite():
            raise ValueError("marked portfolio equity must be finite and positive")
        return marked

    @staticmethod
    def _snapshot_error() -> ActionableError:
        return ActionableError(
            operation="decisions.snapshot",
            category=ErrorCategory.INTEGRITY_CHECKSUM,
            message="The selected snapshot could not be verified for decision delivery.",
            corrective_action="Reconcile the snapshot and retry with a complete checksum-verified publication.",
            field_path="snapshot_id",
        )

    @staticmethod
    def _input_error(operation: str, error: BaseException) -> ActionableError:
        message = str(error).splitlines()[0] or "invalid decision-delivery input"
        return ActionableError(
            operation=operation,
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message=message,
            corrective_action="Use a verified snapshot, signal-close history, and valid whole-share portfolio inputs.",
        )


# Descriptive aliases for application composition roots and test fixtures.
DecisionDeliveryService = CausalDecisionDelivery
CausalDecisionService = CausalDecisionDelivery
DecisionService = CausalDecisionDelivery
DecisionDelivery = CausalDecisionDelivery
DecisionDeliveryOutput = DecisionDeliveryResult


def deliver_decisions(*args: object, **kwargs: object) -> Result[DecisionDeliveryResult]:
    """Functional facade for one causal delivery operation."""

    service = kwargs.pop("service", None)
    if service is None:
        service = CausalDecisionDelivery(
            snapshot_reader=kwargs.pop("snapshot_reader", None),
            snapshot_manager=cast(DecisionSnapshotVerifier | None, kwargs.pop("snapshot_manager", None)),
            calendar=kwargs.pop("calendar", None),
        )
    if not isinstance(service, CausalDecisionDelivery):
        raise TypeError("service must be a CausalDecisionDelivery")
    return cast(
        Result[DecisionDeliveryResult],
        cast(Any, service).deliver(*args, **cast(Any, kwargs)),
    )


def generate_order_intents(
    decisions: Sequence[StrategyDecision],
    portfolio: object | None = None,
    sizing_prices: Mapping[object, object] | None = None,
    execution_session: date | None = None,
    *,
    signal_session: date | None = None,
    calendar: object | None = None,
) -> tuple[OrderIntent, ...]:
    """Generate deterministic intents from already-computed decisions.

    This pure helper is useful at the Zipline seam and intentionally receives
    no execution-session prices.  ``sizing_prices`` is keyed by
    ``(symbol, signal_session)`` (a simple ``symbol -> price`` mapping is also
    accepted for one signal).
    """

    if not decisions:
        return ()
    signal = signal_session or decisions[0].signal_session
    if any(item.signal_session != signal for item in decisions):
        raise ValueError("all decisions must use one signal_session")
    if execution_session is None:
        if calendar is None or not callable(getattr(calendar, "next_session", None)):
            raise ValueError("execution_session or calendar.next_session is required")
        execution_session = getattr(calendar, "next_session")(signal)
    prices = _normalize_sizing_prices(sizing_prices or {}, signal)
    service = CausalDecisionDelivery(calendar=calendar)
    intents, _ = service._make_intents(
        tuple(decisions),
        portfolio,
        prices,
        signal,
        _date_only(execution_session, "execution_session"),
    )
    return intents


create_order_intents = generate_order_intents
build_order_intents = generate_order_intents


def _date_only(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _unwrap_result(value: object, operation: str) -> object:
    if isinstance(value, Err):
        raise _DecisionFailure(value.errors)
    if isinstance(value, Ok):
        return value.value
    if value is None:
        raise ValueError(f"{operation} returned no result")
    return value


class _DecisionFailure(Exception):
    def __init__(self, errors: Sequence[ActionableError]) -> None:
        errors_tuple = tuple(errors)
        if not errors_tuple:
            raise ValueError("decision failure requires at least one error")
        super().__init__(errors_tuple[0].message)
        self.errors = errors_tuple


def _call_reader(
    method: Callable[..., object],
    *,
    snapshot: object,
    symbols: tuple[str, ...],
    end_session: date,
    fields: tuple[str, ...],
    start_session: date,
) -> object:
    """Call a reader once, adapting only named, documented seam variants."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(snapshot, symbols=symbols, end_session=end_session, fields=fields)

    aliases: dict[str, object] = {
        "snapshot": snapshot,
        "snapshot_handle": snapshot,
        "handle": snapshot,
        "snapshot_id": getattr(snapshot, "snapshot_id", snapshot),
        "symbols": symbols,
        "universe": symbols,
        "fields": fields,
        "columns": fields,
        "end_session": end_session,
        "session_end": end_session,
        "end_date": end_session,
        "until_date": end_session,
        "to_date": end_session,
        "max_session": end_session,
        "to_session": end_session,
        "until": end_session,
        "start_session": start_session,
        "session_start": start_session,
        "from_session": start_session,
        "start_date": start_session,
        "from_date": start_session,
    }
    kwargs: dict[str, object] = {}
    positional: list[object] = []
    parameters = tuple(signature.parameters.values())
    has_var_keyword = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters)
    for parameter in parameters:
        if parameter.name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        value = aliases.get(parameter.name)
        if value is None and parameter.default is not inspect.Parameter.empty:
            continue
        if value is None:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(f"reader parameter {parameter.name} is unsupported")
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        elif parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            kwargs[parameter.name] = value
    if has_var_keyword:
        for name in ("symbols", "end_session", "fields"):
            kwargs.setdefault(name, aliases[name])
    return method(*positional, **kwargs)


def _iter_history_rows(
    value: object,
    *,
    symbol_hint: str | None = None,
    session_hint: date | None = None,
) -> Iterable[object]:
    """Flatten compact symbol/session mappings without importing pandas."""

    if isinstance(value, PriceHistory):
        yield from value.observations
        return
    if isinstance(value, Mapping):
        if _looks_like_row(value):
            yield value
            return
        for key, nested in value.items():
            key_date = _as_date(key)
            key_symbol = key.strip().upper() if isinstance(key, str) else None
            if key_date is not None:
                yield from _iter_history_rows(nested, symbol_hint=symbol_hint, session_hint=key_date)
            elif key_symbol is not None:
                yield from _iter_history_rows(nested, symbol_hint=key_symbol, session_hint=session_hint)
            else:
                yield from _iter_history_rows(nested, symbol_hint=symbol_hint, session_hint=session_hint)
        return
    if isinstance(value, (str, bytes, bytearray)):
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_history_rows(item, symbol_hint=symbol_hint, session_hint=session_hint)
        return
    if value is not None:
        yield value


def _looks_like_row(value: Mapping[object, object]) -> bool:
    return any(
        key in value
        for key in (
            "symbol",
            "ticker",
            "session",
            "date",
            "provider_date",
            "adjusted_close",
            "adjustedClose",
            "sizing_adjusted_close",
            "close",
        )
    )


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    date_method = getattr(value, "date", None)
    if callable(date_method):
        converted = date_method()
        if isinstance(converted, date) and not isinstance(converted, datetime):
            return converted
    return None


def _field(value: object, names: tuple[str, ...], default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _decimal(value: object, field_name: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a Decimal-compatible value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise TypeError(f"{field_name} must be a Decimal-compatible value") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _normalize_history(
    source: HistoryInput,
    signal_session: date,
) -> tuple[list[PriceObservation], Mapping[tuple[str, date], Decimal], tuple[date, ...]]:
    observations: dict[tuple[str, date], PriceObservation] = {}
    sizing: dict[tuple[str, date], Decimal] = {}
    sizing_candidates: dict[tuple[tuple[str, date], str], Decimal] = {}
    sessions: set[date] = set()
    supplied_sessions = source.sessions if isinstance(source, PriceHistory) else ()
    sessions.update(item for item in supplied_sessions if item <= signal_session)
    for row in _iter_history_rows(source):
        symbol_value = _field(row, ("symbol", "ticker"))
        session_value = _field(row, ("session", "date", "provider_date"))
        symbol = str(symbol_value).strip().upper() if symbol_value is not None else None
        session = _as_date(session_value)
        if symbol is None or not symbol or session is None or session > signal_session:
            continue
        adjusted = _field(row, ("adjusted_close", "adjustedClose", "adj_close", "close", "price"))
        if adjusted is None:
            continue
        checksum_value = _field(row, ("canonical_row_checksum", "row_checksum", "checksum"))
        checksum = checksum_value if isinstance(checksum_value, str) else None
        tradable_value = _field(row, ("tradable", "is_tradable"), True)
        observation = PriceObservation(
            symbol=symbol,
            session=session,
            adjusted_close=cast(Decimal | int | float | str, adjusted),
            checksum=checksum,
            tradable=bool(tradable_value),
        )
        key = (observation.symbol, observation.session)
        prior = observations.get(key)
        if prior is None or observation.canonical_row_checksum < prior.canonical_row_checksum:
            observations[key] = observation
        sizing_value = _field(
            row,
            (
                "sizing_adjusted_close",
                "execution_adjusted_close",
                "raw_close",
                "close",
                "adjusted_close",
            ),
        )
        sizing_price = _decimal(sizing_value, "sizing_adjusted_close")
        if sizing_price is not None:
            sizing_candidates[(key, observation.canonical_row_checksum)] = sizing_price
        sessions.add(session)
    for key, observation in observations.items():
        sizing_price = sizing_candidates.get((key, observation.canonical_row_checksum))
        if sizing_price is not None:
            sizing[key] = sizing_price
    return list(observations.values()), MappingProxyType(dict(sizing)), tuple(sorted(sessions))


def _normalize_sizing_prices(
    values: Mapping[object, object], signal_session: date
) -> Mapping[tuple[str, date], Decimal]:
    result: dict[tuple[str, date], Decimal] = {}
    for key, value in values.items():
        if isinstance(key, tuple) and len(key) == 2:
            symbol = str(key[0]).strip().upper()
            session = _as_date(key[1])
            if session is None:
                raise TypeError("sizing price keys must contain calendar dates")
        else:
            symbol = str(key).strip().upper()
            session = signal_session
        price = _decimal(value, "sizing_adjusted_close")
        if price is not None:
            result[(symbol, session)] = price
    return MappingProxyType(result)


def _portfolio_snapshot(portfolio: object | None) -> _PortfolioSnapshot:
    if portfolio is None:
        return _PortfolioSnapshot({}, INITIAL_PORTFOLIO_EQUITY, None)
    positions_value = _field(portfolio, ("positions", "holdings", "current_positions"), {})
    holdings: dict[str, int] = {}
    if isinstance(positions_value, Mapping):
        for symbol, value in positions_value.items():
            quantity = _quantity(value)
            if quantity:
                holdings[str(symbol).strip().upper()] = quantity
    elif isinstance(positions_value, Iterable) and not isinstance(positions_value, (str, bytes, bytearray)):
        for item in positions_value:
            symbol = _field(item, ("symbol", "asset", "ticker"))
            quantity_value = _field(item, ("quantity", "shares", "amount"))
            if symbol is None or quantity_value is None:
                continue
            normalized_quantity = _quantity(quantity_value)
            if normalized_quantity:
                holdings[str(symbol).strip().upper()] = normalized_quantity
    cash_value = _field(portfolio, ("cash_balance", "cash", "available_cash"))
    supplied_equity_value = _field(portfolio, ("portfolio_equity", "equity", "total_value"))
    if cash_value is None:
        if not holdings and supplied_equity_value is not None:
            cash = _positive_decimal(supplied_equity_value, "cash_balance")
        elif supplied_equity_value is not None:
            # The mark is recomputed from the supplied equity only when a
            # caller has not provided a separate cash field; positions still
            # require prices at the decision boundary.
            cash = _positive_decimal(supplied_equity_value, "cash_balance")
        else:
            cash = INITIAL_PORTFOLIO_EQUITY
    else:
        cash = _non_negative_decimal(cash_value, "cash_balance")
    supplied_equity = (
        None if supplied_equity_value is None else _positive_decimal(supplied_equity_value, "portfolio_equity")
    )
    return _PortfolioSnapshot(MappingProxyType(holdings), cash, supplied_equity)


def _quantity(value: object) -> int:
    quantity_value: object = value
    if hasattr(quantity_value, "quantity") and not isinstance(quantity_value, (int, Decimal)):
        quantity_value = getattr(quantity_value, "quantity")
    if isinstance(quantity_value, bool):
        raise TypeError("portfolio quantities must be non-negative whole shares")
    if isinstance(quantity_value, int):
        if quantity_value < 0:
            raise ValueError("portfolio quantities must be non-negative whole shares")
        return quantity_value
    if isinstance(quantity_value, Decimal) and quantity_value == quantity_value.to_integral_value():
        result = int(quantity_value)
        if result < 0:
            raise ValueError("portfolio quantities must be non-negative whole shares")
        return result
    raise TypeError("portfolio quantities must be non-negative whole shares")


def _positive_decimal(value: object, field_name: str) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _non_negative_decimal(value: object, field_name: str) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _floor_target_shares(equity: Decimal, weight: RationalWeight, price: Decimal) -> int:
    if not price.is_finite() or price <= 0:
        raise ValueError("sizing_adjusted_close must be finite and positive")
    with localcontext() as context:
        context.prec = 40
        notional = equity * Decimal(weight.numerator) / Decimal(weight.denominator)
        return int(floor(notional / price))


__all__ = [
    "CausalDecisionDelivery",
    "CausalDecisionService",
    "DecisionBook",
    "DecisionDelivery",
    "DecisionDeliveryOutput",
    "DecisionDeliveryResult",
    "DecisionDeliveryService",
    "DecisionInputs",
    "DecisionPolicyVersion",
    "DecisionRunInputs",
    "DecisionService",
    "DecisionSnapshotVerifier",
    "HistoryInput",
    "OrderIntent",
    "RunInputs",
    "SnapshotHistoryReader",
    "build_order_intents",
    "create_order_intents",
    "deliver_decisions",
    "generate_order_intents",
]
