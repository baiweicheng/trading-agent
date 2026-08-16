"""Immutable whole-share execution, portfolio, and job value objects."""

# ruff: noqa: E501, I001

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from enum import Enum, StrEnum
from typing import Final, TypeVar
from uuid import UUID

from .canonical import canonical_json, sha256_bytes
from .strategy import StrategyDecision

MONEY_QUANTUM: Final = Decimal("0.000001")
BASIS_POINT_QUANTUM: Final = Decimal("0.000001")
LEVERAGE_QUANTUM: Final = Decimal("0.000000000000000001")
INITIAL_PORTFOLIO_EQUITY: Final = Decimal("100000.000000")
_ORDER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"order_[0-9a-f]{64}")
_FILL_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"fill_[0-9a-f]{64}")


class OrderStatus(StrEnum):
    """Stable lifecycle labels for an immutable order-output record."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    UNFILLED = "unfilled"


class RunState(StrEnum):
    """The legal persisted lifecycle states for a platform run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobState(StrEnum):
    """The legal persisted lifecycle states for one local job."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"


class JobOperation(StrEnum):
    """Operations that can produce persisted, visible job progress."""

    INGESTION = "ingestion"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"
    COMPARISON = "comparison"


class JobStage(StrEnum):
    """Stable progress stages independent of presentation wording."""

    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    FETCHING = "fetching"
    NORMALIZING = "normalizing"
    VALIDATING = "validating"
    PUBLISHING = "publishing"
    MATERIALIZING = "materializing"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


def _require_date(name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a calendar date")
    return value


def _require_integer(name: str, value: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_decimal(name: str, value: Decimal) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def _quantize(name: str, value: Decimal, quantum: Decimal) -> Decimal:
    decimal_value = _require_decimal(name, value)
    try:
        with localcontext() as context:
            context.prec = 28
            return decimal_value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as error:
        raise ValueError(f"{name} cannot be quantized") from error


def quantize_money(value: Decimal) -> Decimal:
    """Quantize a finite USD value once using the mandated six-decimal scale."""
    return _quantize("money", value, MONEY_QUANTUM)


def quantize_basis_points(value: Decimal) -> Decimal:
    """Quantize a finite basis-point value without converting it to float."""
    return _quantize("basis_points", value, BASIS_POINT_QUANTUM)


def _quantize_leverage(value: Decimal) -> Decimal:
    return _quantize("leverage", value, LEVERAGE_QUANTUM)


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be blank")
    if any(character.isspace() for character in normalized):
        raise ValueError("symbol must not contain whitespace")
    return normalized


def _clean_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


_T = TypeVar("_T", bound=Enum)


def _coerce_enum(enum_type: type[_T], name: str, value: _T | str) -> _T:
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unsupported {name}: {value!r}") from error


def deterministic_order_id(
    *,
    signal_session: date,
    execution_session: date,
    symbol: str,
    requested_quantity: int,
    ordinal: int,
) -> str:
    """Derive a scientific order identifier without a run/job UUID or clock value."""
    signal = _require_date("signal_session", signal_session)
    execution = _require_date("execution_session", execution_session)
    if execution <= signal:
        raise ValueError("execution_session must be after signal_session")
    quantity = _require_integer("requested_quantity", requested_quantity)
    if quantity == 0:
        raise ValueError("requested_quantity must not be zero")
    sequence = _require_integer("ordinal", ordinal, minimum=0)
    digest = sha256_bytes(
        canonical_json(
            {
                "execution_session": execution,
                "ordinal": sequence,
                "requested_quantity": quantity,
                "signal_session": signal,
                "symbol": _normalize_symbol(symbol),
            }
        )
    )
    return f"order_{digest}"


def deterministic_fill_id(
    *,
    order_id: str,
    symbol: str,
    session: date,
    quantity: int,
    ordinal: int,
) -> str:
    """Derive a scientific fill identifier from deterministic execution facts."""
    if not isinstance(order_id, str) or _ORDER_ID_PATTERN.fullmatch(order_id) is None:
        raise ValueError("order_id must be a deterministic order ID")
    fill_quantity = _require_integer("quantity", quantity)
    if fill_quantity == 0:
        raise ValueError("quantity must not be zero")
    digest = sha256_bytes(
        canonical_json(
            {
                "order_id": order_id,
                "ordinal": _require_integer("ordinal", ordinal, minimum=0),
                "quantity": fill_quantity,
                "session": _require_date("session", session),
                "symbol": _normalize_symbol(symbol),
            }
        )
    )
    return f"fill_{digest}"


@dataclass(frozen=True, slots=True)
class TransactionCostModel:
    """Non-negative Decimal commission and adverse-slippage assumptions in bps."""

    commission_bps: Decimal
    slippage_bps: Decimal

    def __post_init__(self) -> None:
        commission = quantize_basis_points(self.commission_bps)
        slippage = quantize_basis_points(self.slippage_bps)
        if commission < 0 or slippage < 0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")
        object.__setattr__(self, "commission_bps", commission)
        object.__setattr__(self, "slippage_bps", slippage)

    @property
    def commission_rate(self) -> Decimal:
        return self.commission_bps / Decimal("10000")

    @property
    def slippage_rate(self) -> Decimal:
        return self.slippage_bps / Decimal("10000")


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """A deterministic whole-share order created after a signal-session close."""

    order_id: str
    signal_session: date
    execution_session: date
    symbol: str
    requested_quantity: int
    ordinal: int
    decision_rank: int | None = None
    status: OrderStatus | str = OrderStatus.PENDING
    unfilled_reason: str | None = None

    def __post_init__(self) -> None:
        signal_session = _require_date("signal_session", self.signal_session)
        execution_session = _require_date("execution_session", self.execution_session)
        symbol = _normalize_symbol(self.symbol)
        quantity = _require_integer("requested_quantity", self.requested_quantity)
        if quantity == 0:
            raise ValueError("requested_quantity must not be zero")
        ordinal = _require_integer("ordinal", self.ordinal, minimum=0)
        if self.decision_rank is not None:
            _require_integer("decision_rank", self.decision_rank, minimum=1)
        status = _coerce_enum(OrderStatus, "status", self.status)
        reason = _clean_text("unfilled_reason", self.unfilled_reason)
        if status in {OrderStatus.PENDING, OrderStatus.FILLED} and reason is not None:
            raise ValueError(f"{status.value} orders must not have an unfilled_reason")
        if (
            status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.UNFILLED}
            and reason is None
        ):
            raise ValueError(f"{status.value} orders require an unfilled_reason")

        expected_id = deterministic_order_id(
            signal_session=signal_session,
            execution_session=execution_session,
            symbol=symbol,
            requested_quantity=quantity,
            ordinal=ordinal,
        )
        if self.order_id != expected_id:
            raise ValueError(
                "order_id does not match its deterministic scientific inputs"
            )
        object.__setattr__(self, "signal_session", signal_session)
        object.__setattr__(self, "execution_session", execution_session)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "requested_quantity", quantity)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unfilled_reason", reason)

    def to_serializable(self) -> dict[str, object]:
        return {
            "decision_rank": self.decision_rank,
            "execution_session": self.execution_session,
            "order_id": self.order_id,
            "ordinal": self.ordinal,
            "requested_quantity": self.requested_quantity,
            "signal_session": self.signal_session,
            "status": OrderStatus(self.status).value,
            "symbol": self.symbol,
            "unfilled_reason": self.unfilled_reason,
        }


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One deterministic simulated fill using quantized Decimal cost arithmetic."""

    fill_id: str
    order_id: str
    symbol: str
    session: date
    quantity: int
    ordinal: int
    base_adjusted_open: Decimal
    fill_price: Decimal
    gross_notional: Decimal
    commission: Decimal
    slippage_cost: Decimal

    def __post_init__(self) -> None:
        if (
            not isinstance(self.order_id, str)
            or _ORDER_ID_PATTERN.fullmatch(self.order_id) is None
        ):
            raise ValueError("order_id must be a deterministic order ID")
        symbol = _normalize_symbol(self.symbol)
        session = _require_date("session", self.session)
        quantity = _require_integer("quantity", self.quantity)
        if quantity == 0:
            raise ValueError("quantity must not be zero")
        ordinal = _require_integer("ordinal", self.ordinal, minimum=0)
        expected_id = deterministic_fill_id(
            order_id=self.order_id,
            symbol=symbol,
            session=session,
            quantity=quantity,
            ordinal=ordinal,
        )
        if self.fill_id != expected_id:
            raise ValueError(
                "fill_id does not match its deterministic scientific inputs"
            )

        base_open = quantize_money(self.base_adjusted_open)
        fill_price = quantize_money(self.fill_price)
        if base_open <= 0 or fill_price <= 0:
            raise ValueError("base_adjusted_open and fill_price must be positive")
        if quantity > 0 and fill_price < base_open:
            raise ValueError("a buy fill_price must not be below base_adjusted_open")
        if quantity < 0 and fill_price > base_open:
            raise ValueError("a sell fill_price must not exceed base_adjusted_open")

        gross_notional = quantize_money(self.gross_notional)
        expected_notional = quantize_money(abs(quantity) * fill_price)
        if gross_notional != expected_notional:
            raise ValueError("gross_notional must equal abs(quantity) * fill_price")
        commission = quantize_money(self.commission)
        slippage_cost = quantize_money(self.slippage_cost)
        if commission < 0 or slippage_cost < 0:
            raise ValueError("commission and slippage_cost must be non-negative")
        expected_slippage = quantize_money(abs(fill_price - base_open) * abs(quantity))
        if slippage_cost != expected_slippage:
            raise ValueError(
                "slippage_cost must equal adverse price difference times quantity"
            )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "base_adjusted_open", base_open)
        object.__setattr__(self, "fill_price", fill_price)
        object.__setattr__(self, "gross_notional", gross_notional)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "slippage_cost", slippage_cost)

    def to_serializable(self) -> dict[str, object]:
        return {
            "base_adjusted_open": self.base_adjusted_open,
            "commission": self.commission,
            "fill_id": self.fill_id,
            "fill_price": self.fill_price,
            "gross_notional": self.gross_notional,
            "order_id": self.order_id,
            "ordinal": self.ordinal,
            "quantity": self.quantity,
            "session": self.session,
            "slippage_cost": self.slippage_cost,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class Position:
    """A non-zero long actual-share holding marked at an action-effective price."""

    symbol: str
    quantity: int
    mark_price: Decimal
    market_value: Decimal

    def __post_init__(self) -> None:
        symbol = _normalize_symbol(self.symbol)
        quantity = _require_integer("quantity", self.quantity, minimum=1)
        mark_price = quantize_money(self.mark_price)
        if mark_price <= 0:
            raise ValueError("mark_price must be positive")
        market_value = quantize_money(self.market_value)
        expected_value = quantize_money(quantity * mark_price)
        if market_value != expected_value:
            raise ValueError("market_value must equal quantity * mark_price")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "mark_price", mark_price)
        object.__setattr__(self, "market_value", market_value)

    def to_serializable(self) -> dict[str, object]:
        return {
            "mark_price": self.mark_price,
            "market_value": self.market_value,
            "quantity": self.quantity,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """One end-of-session cash, long-position, equity, and leverage snapshot."""

    session: date
    cash_balance: Decimal
    positions: tuple[Position, ...]
    gross_exposure: Decimal
    portfolio_equity: Decimal
    leverage: Decimal

    def __post_init__(self) -> None:
        session = _require_date("session", self.session)
        cash_balance = quantize_money(self.cash_balance)
        if cash_balance < 0:
            raise ValueError("cash_balance must be non-negative")
        if not isinstance(self.positions, tuple):
            raise TypeError("positions must be an immutable tuple")
        if any(not isinstance(position, Position) for position in self.positions):
            raise TypeError("positions must contain only Position values")
        symbols = tuple(position.symbol for position in self.positions)
        if symbols != tuple(sorted(symbols)) or len(symbols) != len(set(symbols)):
            raise ValueError("positions must be symbol-sorted and unique")

        expected_gross = quantize_money(
            sum((position.market_value for position in self.positions), Decimal("0"))
        )
        gross_exposure = quantize_money(self.gross_exposure)
        if gross_exposure != expected_gross:
            raise ValueError(
                "gross_exposure must equal the sum of marked position values"
            )
        portfolio_equity = quantize_money(self.portfolio_equity)
        if portfolio_equity != quantize_money(cash_balance + gross_exposure):
            raise ValueError(
                "portfolio_equity must equal cash_balance plus gross_exposure"
            )
        if portfolio_equity <= 0:
            raise ValueError("portfolio_equity must be positive")
        leverage = _quantize_leverage(self.leverage)
        expected_leverage = _quantize_leverage(gross_exposure / portfolio_equity)
        if leverage != expected_leverage:
            raise ValueError(
                "leverage must equal gross_exposure divided by portfolio_equity"
            )
        if not Decimal("0") <= leverage <= Decimal("1"):
            raise ValueError("leverage must be between 0 and 1")

        object.__setattr__(self, "session", session)
        object.__setattr__(self, "cash_balance", cash_balance)
        object.__setattr__(self, "gross_exposure", gross_exposure)
        object.__setattr__(self, "portfolio_equity", portfolio_equity)
        object.__setattr__(self, "leverage", leverage)

    def to_serializable(self) -> dict[str, object]:
        return {
            "cash_balance": self.cash_balance,
            "gross_exposure": self.gross_exposure,
            "leverage": self.leverage,
            "portfolio_equity": self.portfolio_equity,
            "positions": [position.to_serializable() for position in self.positions],
            "session": self.session,
        }


@dataclass(frozen=True, slots=True)
class DailyReturn:
    """A finite session return, expressed as Decimal rather than float."""

    session: date
    return_value: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", _require_date("session", self.session))
        value = _require_decimal("return_value", self.return_value)
        if value <= Decimal("-1"):
            raise ValueError("return_value must be greater than -1")
        object.__setattr__(self, "return_value", value)

    def to_serializable(self) -> dict[str, object]:
        return {"return_value": self.return_value, "session": self.session}


@dataclass(frozen=True, slots=True)
class CoreBacktestOutput:
    """Complete deterministic scientific output emitted before evaluation artifacts."""

    orders: tuple[OrderRecord, ...]
    fills: tuple[FillRecord, ...]
    portfolio_states: tuple[PortfolioState, ...]
    daily_returns: tuple[DailyReturn, ...]
    strategy_decisions: tuple[StrategyDecision, ...]
    initial_equity: Decimal = INITIAL_PORTFOLIO_EQUITY

    def __post_init__(self) -> None:
        self._validate_tuple("orders", self.orders, OrderRecord)
        self._validate_tuple("fills", self.fills, FillRecord)
        self._validate_tuple("portfolio_states", self.portfolio_states, PortfolioState)
        self._validate_tuple("daily_returns", self.daily_returns, DailyReturn)
        self._validate_tuple(
            "strategy_decisions", self.strategy_decisions, StrategyDecision
        )
        initial_equity = quantize_money(self.initial_equity)
        if initial_equity != INITIAL_PORTFOLIO_EQUITY:
            raise ValueError("initial_equity must be fixed at USD 100000.000000")
        object.__setattr__(self, "initial_equity", initial_equity)

        order_ids = tuple(order.order_id for order in self.orders)
        if len(order_ids) != len(set(order_ids)):
            raise ValueError("orders must have unique order_id values")
        order_by_id = {order.order_id: order for order in self.orders}
        fill_ids = tuple(fill.fill_id for fill in self.fills)
        if len(fill_ids) != len(set(fill_ids)):
            raise ValueError("fills must have unique fill_id values")
        for fill in self.fills:
            order = order_by_id.get(fill.order_id)
            if order is None:
                raise ValueError("every fill must reference an order in orders")
            if fill.symbol != order.symbol:
                raise ValueError("fill symbol must match its order symbol")
            if (fill.quantity > 0) != (order.requested_quantity > 0):
                raise ValueError("fill quantity direction must match its order")

        self._validate_session_order(
            "portfolio_states", tuple(state.session for state in self.portfolio_states)
        )
        return_sessions = tuple(item.session for item in self.daily_returns)
        self._validate_session_order("daily_returns", return_sessions)
        state_sessions = {state.session for state in self.portfolio_states}
        if any(session not in state_sessions for session in return_sessions):
            raise ValueError(
                "daily_returns must correspond to portfolio state sessions"
            )
        decision_keys = tuple(
            (decision.signal_session, decision.symbol)
            for decision in self.strategy_decisions
        )
        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError(
                "strategy_decisions must be unique per signal session and symbol"
            )

    @staticmethod
    def _validate_tuple(name: str, values: object, expected_type: type[object]) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be an immutable tuple")
        if any(not isinstance(value, expected_type) for value in values):
            raise TypeError(f"{name} must contain only {expected_type.__name__} values")

    @staticmethod
    def _validate_session_order(name: str, sessions: tuple[date, ...]) -> None:
        if sessions != tuple(sorted(sessions)) or len(sessions) != len(set(sessions)):
            raise ValueError(f"{name} must be session-sorted with no duplicates")

    def to_scientific_dict(self) -> dict[str, object]:
        """Return the deterministic, UUID-free projection for canonical artifacts."""
        return {
            "daily_returns": [item.to_serializable() for item in self.daily_returns],
            "fills": [item.to_serializable() for item in self.fills],
            "initial_equity": self.initial_equity,
            "orders": [item.to_serializable() for item in self.orders],
            "portfolio_states": [
                item.to_serializable() for item in self.portfolio_states
            ],
            "strategy_decisions": [
                item.to_serializable() for item in self.strategy_decisions
            ],
        }


def is_legal_job_transition(
    current: JobState | str,
    target: JobState | str,
    *,
    operation: JobOperation | str,
) -> bool:
    """Return whether one operational job state transition is legal."""
    current_state = _coerce_enum(JobState, "current job state", current)
    target_state = _coerce_enum(JobState, "target job state", target)
    job_operation = _coerce_enum(JobOperation, "job operation", operation)
    transitions = {
        JobState.NOT_STARTED: {JobState.RUNNING},
        JobState.RUNNING: {
            JobState.SUCCEEDED,
            JobState.PARTIALLY_SUCCEEDED,
            JobState.FAILED,
        },
        JobState.SUCCEEDED: set(),
        JobState.PARTIALLY_SUCCEEDED: set(),
        JobState.FAILED: set(),
    }
    if target_state not in transitions[current_state]:
        return False
    return not (
        target_state is JobState.PARTIALLY_SUCCEEDED
        and job_operation is not JobOperation.INGESTION
    )


def require_legal_job_transition(
    current: JobState | str,
    target: JobState | str,
    *,
    operation: JobOperation | str,
) -> JobState:
    """Validate and return a legal target job state, otherwise raise ValueError."""
    coerced_target = _coerce_enum(JobState, "target job state", target)
    if not is_legal_job_transition(current, coerced_target, operation=operation):
        raise ValueError(f"illegal job state transition: {current!r} -> {target!r}")
    return coerced_target


def is_legal_run_transition(current: RunState | str, target: RunState | str) -> bool:
    """Return whether a run can move from running to one terminal state."""
    current_state = _coerce_enum(RunState, "current run state", current)
    target_state = _coerce_enum(RunState, "target run state", target)
    return current_state is RunState.RUNNING and target_state in {
        RunState.SUCCEEDED,
        RunState.FAILED,
    }


def require_legal_run_transition(
    current: RunState | str, target: RunState | str
) -> RunState:
    """Validate and return a legal target run state, otherwise raise ValueError."""
    coerced_target = _coerce_enum(RunState, "target run state", target)
    if not is_legal_run_transition(current, coerced_target):
        raise ValueError(f"illegal run state transition: {current!r} -> {target!r}")
    return coerced_target


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One sanitized, operational progress snapshot identified by an opaque job UUID."""

    job_id: UUID
    operation: JobOperation | str
    state: JobState | str
    stage: JobStage | str
    completed_units: int
    total_units: int | None
    elapsed_seconds: Decimal
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, UUID):
            raise TypeError("job_id must be a UUID operational identifier")
        operation = _coerce_enum(JobOperation, "operation", self.operation)
        state = _coerce_enum(JobState, "state", self.state)
        stage = _coerce_enum(JobStage, "stage", self.stage)
        completed_units = _require_integer(
            "completed_units", self.completed_units, minimum=0
        )
        total_units = self.total_units
        if total_units is not None:
            total_units = _require_integer("total_units", total_units, minimum=0)
            if completed_units > total_units:
                raise ValueError("completed_units must not exceed total_units")
        elapsed_seconds = _require_decimal("elapsed_seconds", self.elapsed_seconds)
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if not isinstance(self.warnings, tuple):
            raise TypeError("warnings must be an immutable tuple")
        warnings = tuple(_clean_text("warning", warning) for warning in self.warnings)
        if any(warning is None for warning in warnings):
            raise AssertionError("warning normalization unexpectedly produced None")

        if state is JobState.NOT_STARTED and (
            stage is not JobStage.NOT_STARTED or completed_units != 0
        ):
            raise ValueError(
                "not_started progress must use not_started stage and zero work"
            )
        if state is JobState.RUNNING and stage in {
            JobStage.NOT_STARTED,
            JobStage.COMPLETED,
            JobStage.FAILED,
        }:
            raise ValueError("running progress must use an active stage")
        if (
            state in {JobState.SUCCEEDED, JobState.PARTIALLY_SUCCEEDED}
            and stage is not JobStage.COMPLETED
        ):
            raise ValueError("successful terminal progress must use completed stage")
        if state is JobState.FAILED and stage is not JobStage.FAILED:
            raise ValueError("failed terminal progress must use failed stage")
        if (
            state is JobState.PARTIALLY_SUCCEEDED
            and operation is not JobOperation.INGESTION
        ):
            raise ValueError("partially_succeeded is valid only for ingestion jobs")

        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "completed_units", completed_units)
        object.__setattr__(self, "total_units", total_units)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        object.__setattr__(self, "warnings", warnings)

    def to_serializable(self) -> dict[str, object]:
        return {
            "completed_units": self.completed_units,
            "elapsed_seconds": self.elapsed_seconds,
            "job_id": str(self.job_id),
            "operation": JobOperation(self.operation).value,
            "stage": JobStage(self.stage).value,
            "state": JobState(self.state).value,
            "total_units": self.total_units,
            "warnings": list(self.warnings),
        }


__all__ = [
    "BASIS_POINT_QUANTUM",
    "CoreBacktestOutput",
    "DailyReturn",
    "FillRecord",
    "INITIAL_PORTFOLIO_EQUITY",
    "JobOperation",
    "JobStage",
    "JobState",
    "LEVERAGE_QUANTUM",
    "MONEY_QUANTUM",
    "OrderRecord",
    "OrderStatus",
    "PortfolioState",
    "Position",
    "ProgressUpdate",
    "RunState",
    "TransactionCostModel",
    "deterministic_fill_id",
    "deterministic_order_id",
    "is_legal_job_transition",
    "is_legal_run_transition",
    "quantize_basis_points",
    "quantize_money",
    "require_legal_job_transition",
    "require_legal_run_transition",
]
