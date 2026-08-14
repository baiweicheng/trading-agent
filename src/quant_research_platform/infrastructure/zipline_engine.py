"""Cash-safe next-session execution at the Zipline boundary.

The platform owns order sizing and cost assumptions, while Zipline owns the
ledger, position tracker, and corporate-action processing.  ``CashSafeOpenBlotter``
therefore replaces only ``SimulationBlotter.get_transactions``.  It does not
replace ``process_splits`` or otherwise reinterpret Zipline's split/dividend
behaviour; returned transactions and commission events are consumed by the
normal Zipline simulation loop.

The class also exposes a small framework-free execution seam.  This is useful
for contract tests and for callers that need to audit the sell-first/cash-cap
arithmetic without constructing a complete Zipline algorithm.
"""

# The execution planner intentionally keeps long, explicit arithmetic branches;
# the stable public seam is more important here than line-level lint density.
# ruff: noqa: E501, I001, SIM102, F841, SIM105, B010

from __future__ import annotations

import random
import time as _time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_FLOOR,
    localcontext,
)
from types import MappingProxyType
from typing import Final, Protocol, cast
from uuid import UUID, uuid4

from ..application.decisions import CausalDecisionDelivery, DecisionDeliveryResult
from ..config.models import ResolvedConfig
from ..domain.errors import ActionableError, Err, ErrorCategory, Ok, Result
from ..domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    CoreBacktestOutput,
    DailyReturn,
    FillRecord,
    OrderRecord,
    OrderStatus,
    PortfolioState,
    Position,
    ProgressUpdate,
    JobOperation,
    JobStage,
    JobState,
    TransactionCostModel,
    deterministic_fill_id,
    quantize_money,
)
from ..domain.strategy import StrategyDecision
from .zipline_bundle import ZiplineBundleLocator

try:  # Zipline is an infrastructure boundary and is intentionally optional.
    from zipline.finance.blotter import SimulationBlotter as _SimulationBlotter
    from zipline.finance.execution import MarketOrder as _MarketOrder
    from zipline.finance.transaction import create_transaction as _create_transaction
except Exception:  # pragma: no cover - exercised only in a dependency-free install.
    _SimulationBlotter = object  # type: ignore[assignment,misc]
    _MarketOrder = None  # type: ignore[assignment,misc]
    _create_transaction = None  # type: ignore[assignment,misc]

try:  # Used only to disable Zipline's second commission calculation.
    from zipline.finance.commission import NoCommission as _NoCommission
except Exception:  # pragma: no cover - exercised only without Zipline.
    _NoCommission = None  # type: ignore[assignment,misc]


INITIAL_CASH: Final[Decimal] = Decimal("100000.000000")
_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")


class OpenPriceProvider(Protocol):
    """Return the execution-adjusted open for one asset/session."""

    def __call__(self, asset: object, session: date) -> object: ...


@dataclass(frozen=True, slots=True)
class CashSafeFill:
    """A Decimal audit record for one non-zero whole-share fill."""

    order_id: str
    symbol: str
    session: date
    quantity: int
    base_adjusted_open: Decimal
    fill_price: Decimal
    gross_notional: Decimal
    commission: Decimal
    slippage_cost: Decimal
    transaction: object | None = None

    @property
    def amount(self) -> int:
        """Zipline terminology alias for the signed share quantity."""

        return self.quantity

    @property
    def base_open(self) -> Decimal:
        return self.base_adjusted_open

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if isinstance(self.session, datetime) or not isinstance(self.session, date):
            raise TypeError("session must be a calendar date")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity == 0:
            raise ValueError("quantity must not be zero")
        for name in (
            "base_adjusted_open",
            "fill_price",
            "gross_notional",
            "commission",
            "slippage_cost",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
            if name in {"base_adjusted_open", "fill_price"} and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name in {"gross_notional", "commission", "slippage_cost"} and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.quantity > 0 and self.fill_price < self.base_adjusted_open:
            raise ValueError("buy fill price must not be below the base open")
        if self.quantity < 0 and self.fill_price > self.base_adjusted_open:
            raise ValueError("sell fill price must not exceed the base open")
        if self.gross_notional != quantize_money(
            abs(self.quantity) * self.fill_price
        ):
            raise ValueError("gross_notional must equal the actual fill notional")
        if self.slippage_cost != quantize_money(
            abs(self.fill_price - self.base_adjusted_open) * abs(self.quantity)
        ):
            raise ValueError("slippage_cost must equal adverse price slippage")


@dataclass(frozen=True, slots=True)
class UnfilledOrder:
    """A remaining order quantity and its actionable execution reason."""

    order_id: str
    symbol: str
    session: date
    quantity: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if isinstance(self.session, datetime) or not isinstance(self.session, date):
            raise TypeError("session must be a calendar date")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")
        if self.quantity == 0:
            raise ValueError("quantity must not be zero")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity


@dataclass(frozen=True, slots=True)
class CommissionCharge:
    """Commission event emitted by the framework-free execution seam."""

    order_id: str
    symbol: str
    quantity: int
    cost: Decimal


@dataclass(frozen=True, slots=True)
class CashSafeExecutionResult:
    """Complete result of one deterministic open-execution pass."""

    fills: tuple[CashSafeFill, ...]
    unfilled_orders: tuple[UnfilledOrder, ...]
    cash_balance: Decimal
    positions: Mapping[str, int]
    actionable_errors: tuple[ActionableError, ...]
    commission_charges: tuple[CommissionCharge, ...] = ()

    def __post_init__(self) -> None:
        cash = quantize_money(self.cash_balance)
        if cash < 0:
            raise ValueError("cash_balance must be non-negative")
        normalized: dict[str, int] = {}
        for symbol, quantity in self.positions.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("position symbols must be non-empty strings")
            if isinstance(quantity, bool) or not isinstance(quantity, int):
                raise TypeError("position quantities must be integers")
            if quantity < 0:
                raise ValueError("position quantities must be non-negative")
            if quantity:
                normalized[symbol.strip().upper()] = quantity
        if any(not isinstance(item, CashSafeFill) for item in self.fills):
            raise TypeError("fills must contain CashSafeFill values")
        if any(not isinstance(item, UnfilledOrder) for item in self.unfilled_orders):
            raise TypeError("unfilled_orders must contain UnfilledOrder values")
        if any(not isinstance(item, ActionableError) for item in self.actionable_errors):
            raise TypeError("actionable_errors must contain ActionableError values")
        object.__setattr__(self, "cash_balance", cash)
        object.__setattr__(self, "positions", MappingProxyType(normalized))
        object.__setattr__(self, "fills", tuple(self.fills))
        object.__setattr__(self, "unfilled_orders", tuple(self.unfilled_orders))
        object.__setattr__(self, "actionable_errors", tuple(self.actionable_errors))
        object.__setattr__(self, "commission_charges", tuple(self.commission_charges))

    @property
    def fills_by_order(self) -> Mapping[str, tuple[CashSafeFill, ...]]:
        grouped: dict[str, list[CashSafeFill]] = {}
        for fill in self.fills:
            grouped.setdefault(fill.order_id, []).append(fill)
        return MappingProxyType({key: tuple(value) for key, value in grouped.items()})

    @property
    def remaining_orders(self) -> tuple[UnfilledOrder, ...]:
        return self.unfilled_orders

    @property
    def cash(self) -> Decimal:
        return self.cash_balance


@dataclass(frozen=True, slots=True)
class _OrderView:
    order: object
    order_id: str
    asset: object | None
    symbol: str
    quantity: int
    decision_rank: int | None
    execution_session: date | None
    signal_session: date | None


class CashSafeOpenBlotter(_SimulationBlotter):
    """Zipline blotter with causal next-open, whole-share cash-safe fills.

    ``get_transactions`` has the exact interface expected by Zipline Reloaded
    3.1.x.  It returns normal Zipline transactions and commission events; the
    simulation loop subsequently submits those objects to the ledger.  The
    class never calls Zipline's default slippage or commission models, which
    prevents costs from being applied twice.

    For deterministic unit/contract tests, call :meth:`execute_orders` with
    an order sequence, an ``opens`` mapping, and an explicit cash/positions
    state.  The same Decimal arithmetic is used by ``get_transactions``.
    """

    operation_name: Final[str] = "backtest.execution"

    def __init__(
        self,
        commission_bps: Decimal | int | float | str = Decimal("5"),
        slippage_bps: Decimal | int | float | str = Decimal("10"),
        *,
        cost_model: TransactionCostModel | None = None,
        transaction_cost_model: TransactionCostModel | None = None,
        initial_cash: Decimal | int | float | str = INITIAL_CASH,
        cash: Decimal | int | float | str | None = None,
        positions: Mapping[object, object] | None = None,
        cash_provider: Callable[[], object] | object | None = None,
        position_provider: Callable[[], object] | object | None = None,
        open_price_provider: OpenPriceProvider | Callable[..., object] | None = None,
        ledger: object | None = None,
        cancel_policy: object | None = None,
    ) -> None:
        selected_model = cost_model or transaction_cost_model
        if cost_model is not None and transaction_cost_model is not None:
            if cost_model != transaction_cost_model:
                raise ValueError("cost_model and transaction_cost_model disagree")
        if selected_model is None:
            selected_model = TransactionCostModel(
                commission_bps=_decimal(commission_bps, "commission_bps"),
                slippage_bps=_decimal(slippage_bps, "slippage_bps"),
            )
        if not isinstance(selected_model, TransactionCostModel):
            raise TypeError("cost_model must be a TransactionCostModel")
        self.cost_model = selected_model
        self.commission_bps = selected_model.commission_bps
        self.slippage_bps = selected_model.slippage_bps
        self._cash = _non_negative_decimal(
            initial_cash if cash is None else cash, "initial_cash"
        )
        self._positions = _normalize_positions(positions or {})
        self._cash_provider = cash_provider
        self._position_provider = position_provider
        self._open_price_provider = open_price_provider
        self._ledger = ledger
        self._order_metadata: dict[str, dict[str, object]] = {}
        self.execution_records: list[CashSafeFill] = []
        self.unfilled_orders: list[UnfilledOrder] = []
        self.actionable_errors: list[ActionableError] = []

        if _SimulationBlotter is object:  # pragma: no cover - no Zipline install.
            return
        kwargs: dict[str, object] = {"cancel_policy": cancel_policy}
        if _NoCommission is not None:
            kwargs["equity_commission"] = _NoCommission()
        try:
            super().__init__(**kwargs)
        except TypeError:  # Support older compatible Zipline extension seams.
            super().__init__(cancel_policy=cancel_policy)

    # ------------------------------------------------------------------
    # Public, framework-free arithmetic seam
    # ------------------------------------------------------------------
    def execute_orders(
        self,
        orders: Iterable[object] | None = None,
        *,
        opens: Mapping[object, object] | None = None,
        cash: Decimal | int | float | str | None = None,
        positions: Mapping[object, object] | None = None,
        session: date | datetime | object | None = None,
        dt: object | None = None,
    ) -> CashSafeExecutionResult:
        """Execute one open at a specified session using Decimal arithmetic.

        Sells are sorted by normalized symbol.  Buys are sorted by decision
        rank, then symbol.  A buy is capped using the post-commission notional;
        a sell is capped by holdings and, for commission rates above 100%, by
        the cash required to absorb the negative net proceeds.
        """

        active_orders = tuple(orders) if orders is not None else self._all_open_orders()
        active_session = _session_from_value(session if session is not None else dt)
        if active_session is None:
            active_session = self._session_from_value(getattr(self, "current_dt", None))
        if active_session is None:
            raise ValueError("execution session is required")
        available_cash = self._read_cash(cash)
        holdings = _normalize_positions(
            self._read_positions(positions) if positions is not None else None
        )
        result = self._execute_core(
            active_orders,
            active_session,
            available_cash,
            holdings,
            open_lookup=lambda view: self._lookup_open(
                view, opens=opens, session=active_session
            ),
            transaction_dt=dt or getattr(self, "current_dt", None),
            build_transactions=False,
            reject_invalid=False,
        )
        self._cash = result.cash_balance
        self._positions = dict(result.positions)
        self.execution_records.extend(result.fills)
        self.unfilled_orders.extend(result.unfilled_orders)
        self.actionable_errors.extend(result.actionable_errors)
        return result

    # Descriptive aliases used by application adapters and tests.
    execute_open_orders = execute_orders
    process_open_orders = execute_orders
    simulate_open = execute_orders
    fill_orders = execute_orders

    def adverse_fill_price(
        self, base_adjusted_open: Decimal | int | float | str, quantity: int
    ) -> Decimal:
        """Return the adverse whole-fill price for a signed quantity."""

        base = _positive_decimal(base_adjusted_open, "base_adjusted_open")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity == 0:
            raise ValueError("quantity must be a non-zero integer")
        with localcontext() as context:
            context.prec = 28
            price = base * (
                _ONE + self.cost_model.slippage_rate
                if quantity > 0
                else _ONE - self.cost_model.slippage_rate
            )
        if not price.is_finite() or price <= 0:
            raise ValueError("adverse fill price must be finite and positive")
        return quantize_money(price)

    calculate_fill_price = adverse_fill_price
    fill_price = adverse_fill_price

    def commission_for(
        self,
        quantity: int,
        fill_price: Decimal | int | float | str,
    ) -> Decimal:
        """Calculate commission on actual (not requested) fill notional."""

        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise TypeError("quantity must be an integer")
        if quantity == 0:
            return Decimal("0.000000")
        price = _positive_decimal(fill_price, "fill_price")
        with localcontext() as context:
            context.prec = 28
            return quantize_money(abs(quantity) * price * self.cost_model.commission_rate)

    calculate_commission = commission_for

    def max_affordable_buy_quantity(
        self,
        cash: Decimal | int | float | str,
        fill_price: Decimal | int | float | str,
        requested_quantity: int,
    ) -> int:
        """Return the greatest non-negative buy quantity including commission."""

        if isinstance(requested_quantity, bool) or not isinstance(requested_quantity, int):
            raise TypeError("requested_quantity must be an integer")
        requested = abs(requested_quantity)
        if requested == 0:
            return 0
        available = _non_negative_decimal(cash, "cash")
        price = _positive_decimal(fill_price, "fill_price")
        denominator = price * (_ONE + self.cost_model.commission_rate)
        if denominator <= 0 or not denominator.is_finite():
            return 0
        with localcontext() as context:
            context.prec = 40
            candidate = int((available / denominator).to_integral_value(rounding=ROUND_FLOOR))
        candidate = min(requested, max(0, candidate))
        while candidate and not self._buy_affordable(available, price, candidate):
            candidate -= 1
        return candidate

    largest_affordable_buy = max_affordable_buy_quantity

    def max_affordable_sell_quantity(
        self,
        cash: Decimal | int | float | str,
        fill_price: Decimal | int | float | str,
        requested_quantity: int,
        holdings: int,
    ) -> int:
        """Return the greatest sell quantity preserving non-negative cash."""

        if isinstance(requested_quantity, bool) or not isinstance(requested_quantity, int):
            raise TypeError("requested_quantity must be an integer")
        if isinstance(holdings, bool) or not isinstance(holdings, int) or holdings < 0:
            raise ValueError("holdings must be a non-negative integer")
        requested = min(abs(requested_quantity), holdings)
        if requested == 0:
            return 0
        available = _non_negative_decimal(cash, "cash")
        price = _positive_decimal(fill_price, "fill_price")
        rate = self.cost_model.commission_rate
        if rate <= _ONE:
            return requested
        denominator = price * (rate - _ONE)
        if denominator <= 0 or not denominator.is_finite():
            return requested
        with localcontext() as context:
            context.prec = 40
            candidate = int((available / denominator).to_integral_value(rounding=ROUND_FLOOR))
        candidate = min(requested, max(0, candidate))
        while candidate and not self._sell_affordable(available, price, candidate):
            candidate -= 1
        return candidate

    largest_affordable_sell = max_affordable_sell_quantity

    # ------------------------------------------------------------------
    # Zipline blotter boundary
    # ------------------------------------------------------------------
    def order(
        self,
        asset: object,
        amount: int,
        style: object | None = None,
        order_id: str | None = None,
        *,
        decision_rank: int | None = None,
        execution_session: date | None = None,
        signal_session: date | None = None,
    ) -> str | None:
        """Place a normal Zipline order and retain execution metadata."""

        if style is None:
            if _MarketOrder is None:
                raise RuntimeError("Zipline MarketOrder is unavailable")
            style = _MarketOrder()
        if _SimulationBlotter is object:  # pragma: no cover - no Zipline install.
            raise RuntimeError("Zipline is required to place engine orders")
        created = super().order(asset, amount, style, order_id=order_id)
        if created is not None:
            self.register_order_metadata(
                created,
                decision_rank=decision_rank,
                execution_session=execution_session,
                signal_session=signal_session,
            )
        return created

    def register_order_metadata(self, order_id: str, **metadata: object) -> None:
        """Attach rank/session metadata without mutating Zipline's slotted Order."""

        if not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        cleaned = {
            key: value for key, value in metadata.items() if value is not None
        }
        for key in ("decision_rank",):
            value = cleaned.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError("decision_rank must be a positive integer")
        for key in ("execution_session", "signal_session"):
            value = cleaned.get(key)
            if value is not None and (
                isinstance(value, datetime) or not isinstance(value, date)
            ):
                raise TypeError(f"{key} must be a calendar date")
        self._order_metadata[order_id] = cleaned

    set_order_metadata = register_order_metadata

    def get_transactions(self, bar_data: object) -> tuple[list[object], list[dict[str, object]], list[object]]:
        """Create cash-safe transactions for the current Zipline session."""

        if _SimulationBlotter is object:  # pragma: no cover - no Zipline install.
            return [], [], []
        current_dt = getattr(self, "current_dt", None)
        session = _session_from_value(current_dt)
        if session is None:
            session = _session_from_value(getattr(bar_data, "current_dt", None))
        if session is None:
            raise ValueError("Zipline blotter current_dt must identify a session")
        orders = self._all_open_orders()
        cash = self._read_cash(None)
        holdings = _normalize_positions(self._read_positions(None))
        result, transactions, commissions, closed = self._execute_zipline_core(
            orders,
            session,
            cash,
            holdings,
            bar_data,
            current_dt,
        )
        self.execution_records.extend(result.fills)
        self.unfilled_orders.extend(result.unfilled_orders)
        self.actionable_errors.extend(result.actionable_errors)
        return transactions, commissions, closed

    def apply_to_ledger(
        self,
        result: CashSafeExecutionResult,
        ledger: object | None = None,
    ) -> None:
        """Apply a pure execution result through a supplied Zipline ledger.

        The normal event loop performs this itself.  This explicit method is
        provided for local integration seams and never applies corporate
        actions; ``process_splits`` remains the inherited Zipline method.
        """

        target = ledger or self._ledger
        if target is None:
            raise ValueError("a Zipline ledger is required")
        process_transaction = getattr(target, "process_transaction", None)
        process_commission = getattr(target, "process_commission", None)
        if not callable(process_transaction) or not callable(process_commission):
            raise TypeError("ledger must expose process_transaction/process_commission")
        for fill in result.fills:
            if fill.transaction is None:
                raise ValueError("execution result does not contain Zipline transactions")
            process_transaction(fill.transaction)
            if fill.commission > 0:
                process_commission(
                    {"asset": _order_asset(fill.order_id, self), "cost": float(fill.commission)}
                )

    @property
    def unfilled(self) -> tuple[UnfilledOrder, ...]:
        return tuple(self.unfilled_orders)

    @property
    def errors(self) -> tuple[ActionableError, ...]:
        return tuple(self.actionable_errors)

    # ------------------------------------------------------------------
    # Internal deterministic planner
    # ------------------------------------------------------------------
    def _execute_zipline_core(
        self,
        orders: Sequence[object],
        session: date,
        cash: Decimal,
        positions: dict[str, int],
        bar_data: object,
        current_dt: object,
    ) -> tuple[
        CashSafeExecutionResult,
        list[object],
        list[dict[str, object]],
        list[object],
    ]:
        transactions: list[object] = []
        commissions: list[dict[str, object]] = []
        closed_orders: list[object] = []
        views = self._order_views(orders)
        result = self._execute_core(
            orders,
            session,
            cash,
            positions,
            open_lookup=lambda view: self._lookup_open_from_bar(view, bar_data, session),
            transaction_dt=current_dt,
            build_transactions=True,
            reject_invalid=True,
        )
        for fill in result.fills:
            if fill.transaction is not None:
                transactions.append(fill.transaction)
                if fill.commission > 0:
                    order = next(
                        (view.order for view in views if view.order_id == fill.order_id),
                        None,
                    )
                    if order is not None:
                        commissions.append(
                            {
                                "asset": _order_asset_object(order),
                                "order": order,
                                "cost": float(fill.commission),
                            }
                        )
        for view in views:
            if _remaining_quantity(view.order) == 0:
                closed_orders.append(view.order)
        return result, transactions, commissions, closed_orders

    def _execute_core(
        self,
        orders: Sequence[object],
        session: date,
        cash: Decimal,
        positions: dict[str, int],
        *,
        open_lookup: Callable[[_OrderView], object],
        transaction_dt: object | None,
        build_transactions: bool,
        reject_invalid: bool,
    ) -> CashSafeExecutionResult:
        cash_balance = _non_negative_decimal(cash, "cash")
        projected = dict(positions)
        fills: list[CashSafeFill] = []
        unfilled: list[UnfilledOrder] = []
        errors: list[ActionableError] = []
        commissions: list[CommissionCharge] = []
        views = self._order_views(orders)
        # Mapping-shaped test orders cannot be mutated like Zipline's slotted
        # Order, so retain projected remaining quantities independently.
        remaining_state = {
            view.order_id: _remaining_quantity(view.order) for view in views
        }
        eligible: list[_OrderView] = []

        for view in views:
            remaining = remaining_state[view.order_id]
            if remaining == 0:
                continue
            if view.execution_session is not None:
                if session < view.execution_session:
                    continue
                if session > view.execution_session:
                    self._unfilled(
                        view,
                        session,
                        remaining,
                        "execution_session_passed",
                        unfilled,
                        errors,
                        reject=reject_invalid,
                    )
                    continue
            if view.signal_session is not None and session <= view.signal_session:
                continue
            eligible.append(view)

        sells = sorted(
            (view for view in eligible if view.quantity < 0),
            key=lambda view: (view.symbol, view.order_id),
        )
        buys = sorted(
            (view for view in eligible if view.quantity > 0),
            key=lambda view: (
                view.decision_rank if view.decision_rank is not None else 2**31 - 1,
                view.symbol,
                view.order_id,
            ),
        )
        for view in (*sells, *buys):
            remaining = remaining_state[view.order_id]
            if remaining == 0:
                continue
            try:
                raw_open: object | None = open_lookup(view)
                base_open = _positive_decimal(raw_open, "adjusted_open")
                fill_price = self.adverse_fill_price(base_open, remaining)
            except (TypeError, ValueError, InvalidOperation, ArithmeticError):
                reason = "missing_or_non_positive_adjusted_open"
                if raw_open is not None:
                    reason = "invalid_adjusted_open"
                self._unfilled(
                    view,
                    session,
                    remaining,
                    reason,
                    unfilled,
                    errors,
                    reject=reject_invalid,
                )
                continue
            del raw_open

            if view.quantity < 0:
                held = projected.get(view.symbol, 0)
                requested = min(abs(remaining), held)
                quantity = self.max_affordable_sell_quantity(
                    cash_balance, fill_price, requested, held
                )
                if quantity == 0:
                    reason = "position_or_commission_cash_constraint"
                    self._unfilled(
                        view,
                        session,
                        remaining,
                        reason,
                        unfilled,
                        errors,
                        reject=False,
                    )
                    continue
                signed_quantity = -quantity
                commission = self.commission_for(signed_quantity, fill_price)
                notional = quantize_money(quantity * fill_price)
                cash_after = quantize_money(cash_balance + notional - commission)
                if cash_after < 0:
                    # This can only occur because of the mandated output
                    # quantization, but decrementing makes the invariant exact.
                    while quantity and cash_after < 0:
                        quantity -= 1
                        signed_quantity = -quantity
                        commission = self.commission_for(signed_quantity, fill_price)
                        notional = quantize_money(quantity * fill_price)
                        cash_after = quantize_money(cash_balance + notional - commission)
                    if quantity == 0:
                        self._unfilled(
                            view,
                            session,
                            remaining,
                            "commission_cash_constraint",
                            unfilled,
                            errors,
                            reject=False,
                        )
                        continue
                projected[view.symbol] = held - quantity
                cash_balance = cash_after
            else:
                quantity = self.max_affordable_buy_quantity(
                    cash_balance, fill_price, remaining
                )
                if quantity == 0:
                    self._unfilled(
                        view,
                        session,
                        remaining,
                        "cash_constraint_including_commission",
                        unfilled,
                        errors,
                        reject=False,
                    )
                    continue
                signed_quantity = quantity
                commission = self.commission_for(quantity, fill_price)
                notional = quantize_money(quantity * fill_price)
                cash_after = quantize_money(cash_balance - notional - commission)
                if cash_after < 0:  # Defensive against quantization at the boundary.
                    while quantity and cash_after < 0:
                        quantity -= 1
                        commission = self.commission_for(quantity, fill_price)
                        notional = quantize_money(quantity * fill_price)
                        cash_after = quantize_money(cash_balance - notional - commission)
                    if quantity == 0:
                        self._unfilled(
                            view,
                            session,
                            remaining,
                            "cash_constraint_including_commission",
                            unfilled,
                            errors,
                            reject=False,
                        )
                        continue
                    signed_quantity = quantity
                projected[view.symbol] = projected.get(view.symbol, 0) + quantity
                cash_balance = cash_after

            transaction: object | None = None
            if build_transactions:
                transaction = self._make_transaction(
                    view.order, transaction_dt, fill_price, signed_quantity
                )
            fill_base_open = quantize_money(base_open)
            fill_execution_price = quantize_money(fill_price)
            fill = CashSafeFill(
                order_id=view.order_id,
                symbol=view.symbol,
                session=session,
                quantity=signed_quantity,
                base_adjusted_open=fill_base_open,
                fill_price=fill_execution_price,
                gross_notional=quantize_money(
                    abs(signed_quantity) * fill_execution_price
                ),
                commission=commission,
                slippage_cost=quantize_money(
                    abs(fill_execution_price - fill_base_open) * abs(signed_quantity)
                ),
                transaction=transaction,
            )
            fills.append(fill)
            commissions.append(
                CommissionCharge(view.order_id, view.symbol, signed_quantity, commission)
            )
            _increase_order_filled(view.order, signed_quantity, commission, transaction_dt)
            remaining = remaining - signed_quantity
            remaining_state[view.order_id] = remaining
            remainder = remaining
            if remainder:
                reason = (
                    "position_constraint"
                    if view.quantity < 0 and projected.get(view.symbol, 0) == 0
                    else "cash_constraint_including_commission"
                )
                self._unfilled(
                    view,
                    session,
                    remainder,
                    reason,
                    unfilled,
                    errors,
                    reject=False,
                )

        return CashSafeExecutionResult(
            fills=tuple(fills),
            unfilled_orders=tuple(unfilled),
            cash_balance=cash_balance,
            positions=projected,
            actionable_errors=tuple(errors),
            commission_charges=tuple(commissions),
        )

    def _unfilled(
        self,
        view: _OrderView,
        session: date,
        quantity: int,
        reason: str,
        unfilled: list[UnfilledOrder],
        errors: list[ActionableError],
        *,
        reject: bool,
    ) -> None:
        if quantity == 0:
            return
        record = UnfilledOrder(view.order_id, view.symbol, session, quantity, reason)
        unfilled.append(record)
        errors.append(
            ActionableError(
                operation=self.operation_name,
                category=ErrorCategory.BACKTEST_EXECUTION,
                message=(
                    f"Order {view.order_id} for {quantity} shares of {view.symbol} "
                    f"was not fully filled: {reason}."
                ),
                corrective_action=(
                    "Verify the next-session adjusted open and portfolio constraints; "
                    "retry with a valid snapshot if the market row is missing."
                ),
                symbol=view.symbol,
                session=session,
            )
        )
        if reject and reason in {
            "missing_or_non_positive_adjusted_open",
            "invalid_adjusted_open",
            "execution_session_passed",
        }:
            reject_order = getattr(self, "reject", None)
            if callable(reject_order):
                with _suppress_exceptions():
                    reject_order(view.order_id, reason)

    def _order_views(self, orders: Iterable[object]) -> tuple[_OrderView, ...]:
        views: list[_OrderView] = []
        for order in orders:
            order_id = _order_id(order)
            metadata = self._order_metadata.get(order_id, {})
            asset = _field(order, "asset", metadata.get("asset"))
            symbol = _symbol(asset or order)
            quantity = _remaining_quantity(order)
            rank_value = _field(order, "decision_rank", metadata.get("decision_rank"))
            rank = None if rank_value is None else _positive_int(rank_value, "decision_rank")
            execution = _date_field(
                _field(order, "execution_session", metadata.get("execution_session"))
            )
            signal = _date_field(
                _field(order, "signal_session", metadata.get("signal_session"))
            )
            views.append(
                _OrderView(order, order_id, asset, symbol, quantity, rank, execution, signal)
            )
        return tuple(views)

    def _all_open_orders(self) -> tuple[object, ...]:
        open_orders = getattr(self, "open_orders", {})
        if isinstance(open_orders, Mapping):
            values: list[object] = []
            for group in open_orders.values():
                if isinstance(group, Iterable) and not isinstance(group, (str, bytes)):
                    values.extend(group)
                else:
                    values.append(group)
            return tuple(values)
        return ()

    def _read_cash(self, supplied: object | None) -> Decimal:
        if supplied is not None:
            return _non_negative_decimal(supplied, "cash")
        source = self._cash_provider
        if callable(source):
            return _non_negative_decimal(source(), "cash")
        if source is not None:
            return _non_negative_decimal(source, "cash")
        ledger = self._ledger
        if ledger is not None:
            for candidate in (
                _field(ledger, "cash"),
                _field(_field(ledger, "portfolio"), "cash"),
                _field(_field(ledger, "portfolio"), "cash_balance"),
            ):
                if candidate is not None:
                    return _non_negative_decimal(candidate, "cash")
        return self._cash

    def _read_positions(self, supplied: object | None) -> Mapping[object, object]:
        if supplied is not None:
            return cast(Mapping[object, object], supplied)
        source = self._position_provider
        if callable(source):
            value = source()
            return cast(Mapping[object, object], value or {})
        if isinstance(source, Mapping):
            return source
        if source is not None:
            return cast(Mapping[object, object], source)
        ledger = self._ledger
        if ledger is not None:
            tracker = _field(ledger, "position_tracker")
            value = _field(tracker, "positions")
            if isinstance(value, Mapping):
                return value
            value = _field(_field(ledger, "portfolio"), "positions")
            if isinstance(value, Mapping):
                return value
        return self._positions

    def _lookup_open(
        self,
        view: _OrderView,
        *,
        opens: Mapping[object, object] | None,
        session: date | None = None,
    ) -> object:
        if opens is not None:
            found = _mapping_lookup(
                opens, view.asset, view.symbol, view.order_id, session=session
            )
            if found is not _MISSING:
                return _price_from_value(found)
        provider = self._open_price_provider
        if provider is not None:
            return _call_price_provider(provider, view.asset, view.symbol, view.order_id, None)
        return None

    def _lookup_open_from_bar(self, view: _OrderView, bar_data: object, session: date) -> object:
        provider = self._open_price_provider
        if provider is not None:
            value = _call_price_provider(provider, view.asset, view.symbol, view.order_id, session)
            if value is not None:
                return _price_from_value(value)
        return _bar_open(bar_data, view.asset, view.symbol)

    def _make_transaction(
        self,
        order: object,
        transaction_dt: object | None,
        fill_price: Decimal,
        quantity: int,
    ) -> object | None:
        if _create_transaction is None:
            return None
        if transaction_dt is None:
            transaction_dt = datetime.combine(date.today(), time.min, tzinfo=UTC)
        try:
            return _create_transaction(order, transaction_dt, float(fill_price), quantity)
        except (TypeError, ValueError, AttributeError):
            return None

    def _buy_affordable(self, cash: Decimal, price: Decimal, quantity: int) -> bool:
        return quantize_money(
            cash - quantize_money(quantity * price) - self.commission_for(quantity, price)
        ) >= 0

    def _sell_affordable(self, cash: Decimal, price: Decimal, quantity: int) -> bool:
        return quantize_money(
            cash + quantize_money(quantity * price) - self.commission_for(-quantity, price)
        ) >= 0


# ----------------------------------------------------------------------
# Small coercion helpers kept private so Zipline types do not leak inward.
# ----------------------------------------------------------------------

_MISSING = object()


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be Decimal-compatible")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be Decimal-compatible") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _positive_decimal(value: object, name: str) -> Decimal:
    result = _decimal(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_decimal(value: object, name: str) -> Decimal:
    result = _decimal(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _normalize_positions(values: Mapping[object, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for asset, value in values.items():
        quantity_value = _field(value, "amount", value)
        if quantity_value is None:
            quantity_value = _field(value, "quantity")
        if isinstance(quantity_value, bool) or not isinstance(quantity_value, int):
            if isinstance(quantity_value, Decimal) and quantity_value == quantity_value.to_integral_value():
                quantity_value = int(quantity_value)
            else:
                try:
                    quantity_value = int(quantity_value)
                except (TypeError, ValueError) as error:
                    raise TypeError("positions must contain whole-share quantities") from error
        if quantity_value < 0:
            raise ValueError("positions must be long-only and non-negative")
        if quantity_value:
            result[_symbol(asset)] = result.get(_symbol(asset), 0) + quantity_value
    return result


def _field(value: object, name: str, default: object = None) -> object:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _order_id(order: object) -> str:
    value = _field(order, "id")
    if value is None:
        value = _field(order, "order_id")
    if value is None:
        raise ValueError("order must expose id/order_id")
    return str(value)


def _order_asset_object(order: object) -> object:
    asset = _field(order, "asset")
    return asset if asset is not None else order


def _order_asset(order_id: str, blotter: CashSafeOpenBlotter) -> object:
    order = getattr(blotter, "orders", {}).get(order_id)
    return _order_asset_object(order) if order is not None else order_id


def _symbol(value: object) -> str:
    candidate = _field(value, "symbol")
    if candidate is None:
        candidate = _field(value, "ticker")
    if candidate is None and isinstance(value, str):
        candidate = value
    if candidate is None:
        candidate = str(value)
    normalized = str(candidate).strip().upper()
    if not normalized:
        raise ValueError("asset/order symbol must be non-empty")
    return normalized


def _remaining_quantity(order: object) -> int:
    open_amount = _field(order, "open_amount")
    if open_amount is not None and not callable(open_amount):
        value = open_amount
    elif callable(open_amount):
        value = open_amount()
    else:
        amount = _field(order, "amount")
        if amount is None:
            amount = _field(order, "requested_quantity")
        if amount is None:
            raise ValueError("order must expose amount/requested_quantity")
        filled = _field(order, "filled", 0)
        value = amount - filled
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, Decimal) and value == value.to_integral_value():
            value = int(value)
        else:
            raise TypeError("orders must contain whole-share quantities")
    return value


def _increase_order_filled(
    order: object,
    quantity: int,
    commission: Decimal,
    transaction_dt: object | None,
) -> None:
    if isinstance(order, Mapping):
        return
    if hasattr(order, "filled"):
        try:
            setattr(order, "filled", int(getattr(order, "filled", 0)) + quantity)
        except (AttributeError, TypeError):
            pass
    if hasattr(order, "commission"):
        try:
            setattr(order, "commission", float(getattr(order, "commission", 0)) + float(commission))
        except (AttributeError, TypeError):
            pass
    if transaction_dt is not None and hasattr(order, "dt"):
        try:
            setattr(order, "dt", transaction_dt)
        except (AttributeError, TypeError):
            pass


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _date_field(value: object) -> date | None:
    if value is None:
        return None
    return _session_from_value(value)


def _session_from_value(value: object | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    date_method = getattr(value, "date", None)
    if callable(date_method):
        result = date_method()
        if isinstance(result, datetime):
            return result.date()
        if isinstance(result, date):
            return result
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _price_from_value(value: object) -> object:
    if isinstance(value, Mapping):
        for name in ("adjusted_open", "execution_adjusted_open", "open", "price"):
            if name in value:
                return value[name]
    for name in ("adjusted_open", "execution_adjusted_open", "open", "price"):
        candidate = getattr(value, name, _MISSING)
        if candidate is not _MISSING:
            return candidate
    return value


def _mapping_lookup(
    mapping: Mapping[object, object],
    asset: object | None,
    symbol: str,
    order_id: str,
    *,
    session: date | None = None,
) -> object:
    keys_list: list[object] = []
    if session is not None:
        keys_list.extend(((symbol, session), (asset, session)))
    keys_list.extend(((symbol, order_id), symbol, order_id, asset))
    keys = tuple(keys_list)
    for key in keys:
        if key is None:
            continue
        try:
            if key in mapping:
                return mapping[key]
        except TypeError:
            continue
    if session is not None:
        try:
            nested = mapping.get(session, _MISSING)
        except (AttributeError, TypeError):
            nested = _MISSING
        if isinstance(nested, Mapping):
            return _mapping_lookup(
                nested, asset, symbol, order_id, session=None
            )
    return _MISSING


def _call_price_provider(
    provider: Callable[..., object],
    asset: object | None,
    symbol: str,
    order_id: str,
    session: date | None,
) -> object:
    attempts = (
        (asset, session),
        (symbol, session),
        (asset,),
        (symbol,),
        (order_id, session),
        (order_id,),
    )
    last_error: Exception | None = None
    for args in attempts:
        try:
            return provider(*args)
        except TypeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return None


def _bar_open(bar_data: object, asset: object | None, symbol: str) -> object:
    # A local projected reader may expose a row mapping directly.
    if isinstance(bar_data, Mapping):
        found = _mapping_lookup(bar_data, asset, symbol, "")
        if found is not _MISSING:
            return _price_from_value(found)
        for key in ("adjusted_open", "execution_adjusted_open", "open"):
            if key in bar_data:
                return bar_data[key]
    current = getattr(bar_data, "current", None)
    if callable(current):
        for field in ("adjusted_open", "execution_adjusted_open", "open"):
            try:
                candidate = _price_from_value(current(asset, field))
                # Zipline exposes unsupported adjusted fields as NaN on its
                # daily BarData.  Treat those as unavailable and fall back to
                # the raw open, which is the actual-share price in the
                # snapshot's raw-bars-plus-actions bundle.
                _positive_decimal(candidate, field)
                return candidate
            except (KeyError, AttributeError, TypeError, ValueError, IndexError, InvalidOperation):
                continue
    for key in ("adjusted_open", "execution_adjusted_open", "open"):
        candidate = _field(bar_data, key, _MISSING)
        if candidate is not _MISSING:
            return _price_from_value(candidate)
    try:
        row = bar_data[asset]  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError):
        try:
            row = bar_data[symbol]  # type: ignore[index]
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
    return _price_from_value(row)


class _suppress_exceptions:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        return True


__all__ = [
    "BacktestEngine",
    "CashSafeExecutionResult",
    "CashSafeFill",
    "CashSafeOpenBlotter",
    "CommissionCharge",
    "INITIAL_CASH",
    "OpenPriceProvider",
    "UnfilledOrder",
    "ZiplineBacktestEngine",
]


class _EngineFailure(Exception):
    """Internal carrier for one already-sanitized engine boundary error."""

    def __init__(self, error: ActionableError) -> None:
        super().__init__(error.message)
        self.error = error


class _EngineCalendar:
    """Date-only calendar facade used by causal decision delivery."""

    def __init__(self, calendar: object) -> None:
        self._calendar = calendar
        self.name = str(getattr(calendar, "name", "XNYS"))
        self.version = str(getattr(calendar, "version", "zipline"))

    @staticmethod
    def _date(value: object) -> date:
        converted = _session_from_value(value)
        if converted is None:
            raise ValueError("calendar returned a value without a session date")
        return converted

    def next_session(self, value: date) -> date:
        method = getattr(self._calendar, "next_session", None)
        if not callable(method):
            raise ValueError("calendar does not expose next_session")
        try:
            return self._date(method(value))
        except (TypeError, ValueError, AttributeError):
            try:
                import pandas as pd  # type: ignore[import-untyped]

                return self._date(method(pd.Timestamp(value)))
            except Exception as error:
                raise ValueError("calendar could not determine the next session") from error

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]:
        method = getattr(self._calendar, "month_end_sessions", None)
        if callable(method):
            try:
                return tuple(sorted({self._date(value) for value in method(start, end)}))
            except (TypeError, ValueError, AttributeError):
                pass
        sessions = self.sessions(start, end, completed_at=datetime.max.replace(tzinfo=UTC))
        month_ends: dict[tuple[int, int], date] = {}
        for session in sessions:
            month_ends[(session.year, session.month)] = session
        return tuple(month_ends.values())

    def sessions(
        self, start: date, end: date, *, completed_at: datetime
    ) -> tuple[date, ...]:
        method = getattr(self._calendar, "sessions", None)
        if callable(method):
            try:
                return tuple(
                    value
                    for value in (
                        self._date(item) for item in method(
                            start, end, completed_at=completed_at
                        )
                    )
                    if value <= completed_at.date()
                )
            except (TypeError, ValueError, AttributeError):
                pass
        range_method = getattr(self._calendar, "sessions_in_range", None)
        if not callable(range_method):
            return ()
        try:
            import pandas as pd  # type: ignore[import-untyped]

            values = range_method(pd.Timestamp(start), pd.Timestamp(end))
            return tuple(self._date(value) for value in values)
        except Exception as error:
            raise ValueError("calendar could not provide session history") from error


class BacktestEngine:
    """Zipline Reloaded adapter for one verified, daily snapshot backtest.

    The adapter deliberately owns the event-loop composition rather than using
    Zipline's mutable ``latest`` bundle selection.  A caller supplies the exact
    :class:`ZiplineBundleLocator` produced by ``ZiplineBundleAdapter`` and a
    snapshot verifier/history reader for causal strategy decisions.  Zipline
    remains responsible for its ledger and corporate-action lifecycle; the
    platform blotter supplies only the required next-open transaction seam.
    """

    operation_name: Final[str] = "backtest.execute"

    def __init__(
        self,
        snapshot_manager: object | None = None,
        decision_delivery: object | None = None,
        snapshot_reader: object | None = None,
        calendar: object | None = None,
        *,
        bundle_loader: Callable[[ZiplineBundleLocator], object] | None = None,
        metrics_set: object | None = None,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.decision_delivery = decision_delivery
        self.snapshot_reader = snapshot_reader
        self.calendar = calendar
        self.bundle_loader = bundle_loader
        self.metrics_set = metrics_set
        self.last_seed: int | None = None
        self.last_snapshot_id: str | None = None
        self.last_bundle: ZiplineBundleLocator | None = None

    def run(
        self,
        bundle: ZiplineBundleLocator | object,
        request: object | None,
        config: ResolvedConfig,
        progress: Callable[..., object] | None = None,
    ) -> Result[CoreBacktestOutput]:
        """Run one exact bundle and return a sanitized typed result."""

        registered_name: str | None = None
        try:
            if not isinstance(config, ResolvedConfig):
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.CONFIGURATION_INVALID_VALUE,
                        "Backtest requires a validated ResolvedConfig.",
                        "Resolve and validate configuration before starting the backtest.",
                        field_path="config",
                    )
                )
            locator = self._coerce_locator(bundle)
            snapshot = self._verify_snapshot(locator)
            start, end = self._request_range(request, config)
            if start > end:
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.CONFIGURATION_INVALID_VALUE,
                        "The backtest evaluation range is reversed.",
                        "Use an evaluation start no later than the evaluation end.",
                        field_path="evaluation_range",
                    )
                )
            self._validate_snapshot_request(locator, request, start, end, snapshot)
            self._seed(config.runtime.deterministic_seed)
            self.last_seed = config.runtime.deterministic_seed
            self.last_snapshot_id = locator.snapshot_id
            self.last_bundle = locator
            # Progress belongs to this execution, not to a prior run on the
            # same adapter instance.
            self._progress_started = None
            self._progress_job_id = None

            bundle_data, registered_name = self._load_bundle(locator)
            zipline_calendar = self._bundle_calendar(bundle_data)
            decision_calendar = _EngineCalendar(self.calendar or zipline_calendar)
            sessions = self._simulation_sessions(zipline_calendar, start, end)
            if not sessions:
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.SNAPSHOT_NOT_READY,
                        "The exact bundle contains no sessions in the evaluation range.",
                        "Select a completed range covered by the verified snapshot bundle.",
                        field_path="evaluation_range",
                    )
                )
            signal_sessions = frozenset(decision_calendar.month_end_sessions(start, end))
            delivery = self._decision_service(config, decision_calendar)
            submitted: dict[str, object] = {}
            decisions: list[StrategyDecision] = []
            delivery_results: list[DecisionDeliveryResult] = []
            runtime: dict[str, object | None] = {"algorithm": None}
            blotter = CashSafeOpenBlotter(
                commission_bps=config.execution.commission_bps,
                slippage_bps=config.execution.slippage_bps,
                initial_cash=INITIAL_PORTFOLIO_EQUITY,
                cash_provider=lambda: self._runtime_cash(runtime),
                position_provider=lambda: self._runtime_positions(runtime),
            )

            def initialize(context: object) -> None:
                setter = getattr(context, "set_max_leverage", None)
                if callable(setter):
                    setter(1.0)

            def handle_data(context: object, _bar_data: object) -> None:
                session = _session_from_value(getattr(context, "datetime", None))
                if session is None:
                    getter = getattr(context, "get_datetime", None)
                    session = _session_from_value(getter()) if callable(getter) else None
                if session is None:
                    raise _EngineFailure(
                        self._error(
                            ErrorCategory.BACKTEST_EXECUTION,
                            "Zipline did not expose a calendar session to the algorithm.",
                            "Retry with the pinned daily XNYS bundle.",
                            field_path="session",
                        )
                    )
                if session in signal_sessions:
                    delivered = self._deliver(
                        delivery,
                        snapshot,
                        session,
                        getattr(context, "portfolio", None),
                        config,
                        decision_calendar,
                    )
                    delivery_results.append(delivered)
                    decisions.extend(delivered.decisions)
                    for intent in delivered.order_intents:
                        actual_id = self._submit_order(context, intent)
                        if actual_id is None:
                            raise _EngineFailure(
                                self._error(
                                    ErrorCategory.BACKTEST_EXECUTION,
                                    f"Zipline rejected the order for {intent.symbol}.",
                                    "Verify the snapshot asset lifetime and retry the backtest.",
                                    symbol=intent.symbol,
                                    session=session,
                                )
                            )
                        submitted[actual_id] = intent
                        blotter.register_order_metadata(
                            actual_id,
                            decision_rank=intent.decision_rank,
                            execution_session=intent.execution_session,
                            signal_session=intent.signal_session,
                        )
                self._emit_progress(progress, session, sessions, len(sessions))

            algorithm, data_portal = self._algorithm(
                bundle_data=bundle_data,
                calendar=zipline_calendar,
                start=start,
                end=end,
                capital_base=INITIAL_PORTFOLIO_EQUITY,
                initialize=initialize,
                handle_data=handle_data,
                blotter=blotter,
            )
            runtime["algorithm"] = algorithm
            performance = algorithm.run(data_portal)
            output = self._extract_output(
                performance,
                submitted=submitted,
                blotter=blotter,
                decisions=decisions,
                asset_finder=bundle_data.asset_finder,
            )
            return Ok(output)
        except _EngineFailure as failure:
            return Err((failure.error,))
        except (TypeError, ValueError, ArithmeticError) as error:
            return Err((self._input_error(error),))
        except Exception as error:
            return Err((ActionableError.from_unexpected_exception(self.operation_name, error),))
        finally:
            if registered_name is not None:
                self._unregister_bundle(registered_name)

    execute = run
    run_backtest = run

    def _coerce_locator(self, bundle: object) -> ZiplineBundleLocator:
        if isinstance(bundle, ZiplineBundleLocator):
            locator = bundle
        else:
            required = ("bundle_name", "bundle_timestamp", "zipline_root", "snapshot_id", "adapter_version", "bundle_checksum")
            if any(not hasattr(bundle, field) for field in required):
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "Backtest did not receive an exact Zipline bundle locator.",
                        "Materialize and select a checksum-verified snapshot bundle before running.",
                        field_path="bundle",
                    )
                )
            try:
                locator = ZiplineBundleLocator(
                    bundle_name=str(bundle.bundle_name),  # type: ignore[attr-defined]
                    bundle_timestamp=bundle.bundle_timestamp,  # type: ignore[attr-defined]
                    zipline_root=bundle.zipline_root,  # type: ignore[attr-defined]
                    snapshot_id=str(bundle.snapshot_id),  # type: ignore[attr-defined]
                    adapter_version=str(bundle.adapter_version),  # type: ignore[attr-defined]
                    bundle_checksum=str(bundle.bundle_checksum),  # type: ignore[attr-defined]
                )
            except (TypeError, ValueError) as error:
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The Zipline bundle locator is invalid.",
                        "Materialize a complete checksum-verified bundle and retry.",
                        field_path="bundle",
                    )
                ) from error
        if locator.bundle_name.strip().lower() == "latest":
            raise _EngineFailure(
                self._error(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "Mutable latest-bundle resolution is not allowed.",
                    "Select the exact snapshot-specific bundle locator.",
                    field_path="bundle.bundle_name",
                )
            )
        manifest_path = locator.cache_path / "bundle_manifest.json"
        if manifest_path.is_file():
            try:
                import json

                manifest = json.loads(manifest_path.read_bytes())
                if manifest.get("snapshot_id") != locator.snapshot_id or manifest.get("bundle_checksum") != locator.bundle_checksum:
                    raise ValueError("bundle manifest identity mismatch")
            except Exception as error:
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The exact Zipline bundle failed manifest verification.",
                        "Rebuild the derived bundle from the verified snapshot.",
                        field_path="bundle_manifest",
                        checksum=locator.bundle_checksum,
                    )
                ) from error
        return locator

    def _verify_snapshot(self, locator: ZiplineBundleLocator) -> object:
        manager = self.snapshot_manager
        if manager is None:
            supplied = getattr(locator, "snapshot_handle", None)
            if supplied is not None:
                return supplied
            raise _EngineFailure(
                self._error(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "No snapshot verification service is configured.",
                    "Open the selected Snapshot_ID through SnapshotManager before running.",
                    field_path="snapshot_manager",
                )
            )
        opener = getattr(manager, "open_verified", None)
        if not callable(opener):
            raise _EngineFailure(
                self._error(
                    ErrorCategory.STORAGE_IO,
                    "The snapshot service cannot verify published snapshots.",
                    "Configure SnapshotManager.open_verified and retry.",
                    field_path="snapshot_manager.open_verified",
                )
            )
        opened = opener(locator.snapshot_id)
        if isinstance(opened, Err):
            first = opened.errors[0]
            raise _EngineFailure(first)
        snapshot = opened.value if isinstance(opened, Ok) else opened
        if getattr(snapshot, "snapshot_id", None) != locator.snapshot_id:
            raise _EngineFailure(
                self._error(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The verified snapshot does not match the exact bundle.",
                    "Re-open the bundle from the same Snapshot_ID and retry.",
                    field_path="snapshot_id",
                )
            )
        return snapshot

    def _request_range(self, request: object | None, config: ResolvedConfig) -> tuple[date, date]:
        source = request
        candidate = None
        if source is not None:
            candidate = getattr(source, "evaluation_range", None)
            if candidate is None:
                candidate = getattr(source, "requested_range", None)
        if candidate is not None:
            start = _date_attr(candidate, ("start", "evaluation_start", "requested_start"))
            end = _date_attr(candidate, ("end", "evaluation_end", "requested_end"))
            if start is not None and end is not None:
                return start, end
        start = _date_attr(source, ("evaluation_start", "start", "requested_start")) if source is not None else None
        end = _date_attr(source, ("evaluation_end", "end", "requested_end")) if source is not None else None
        if start is not None and end is not None:
            return start, end
        requested = config.data.requested_range
        return requested.start, requested.end

    def _validate_snapshot_request(
        self,
        locator: ZiplineBundleLocator,
        request: object | None,
        start: date,
        end: date,
        snapshot: object,
    ) -> None:
        requested_id = getattr(request, "snapshot_id", None) if request is not None else None
        if requested_id is not None and requested_id != locator.snapshot_id:
            raise _EngineFailure(
                self._error(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The requested Snapshot_ID differs from the bundle Snapshot_ID.",
                    "Select one verified snapshot and its exact derived bundle.",
                    field_path="snapshot_id",
                )
            )
        manifest = getattr(snapshot, "manifest", None)
        if manifest is None:
            inspector = getattr(self.snapshot_manager, "inspect_snapshot", None)
            if callable(inspector):
                inspected = inspector(locator.snapshot_id)
                inspected = inspected.value if isinstance(inspected, Ok) else inspected
                manifest = getattr(inspected, "manifest", inspected)
        identity = getattr(manifest, "content_identity", None)
        requested_range = getattr(identity, "requested_range", None)
        if requested_range is not None and (start < requested_range.start or end > requested_range.end):
            raise _EngineFailure(
                self._error(
                    ErrorCategory.SNAPSHOT_NOT_READY,
                    "The evaluation range is outside the verified snapshot range.",
                    "Choose an evaluation range covered by the selected snapshot.",
                    field_path="evaluation_range",
                )
            )

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np  # type: ignore[import-untyped]

            np.random.seed(seed)
        except Exception:
            # NumPy is a Zipline dependency, but deterministic Python execution
            # remains valid when a lightweight test double omits it.
            pass

    def _load_bundle(self, locator: ZiplineBundleLocator) -> tuple[object, str | None]:
        if self.bundle_loader is not None:
            loaded = self.bundle_loader(locator)
            if loaded is None:
                raise _EngineFailure(
                    self._error(
                        ErrorCategory.INTEGRITY_CHECKSUM,
                        "The exact Zipline bundle loader returned no data.",
                        "Rebuild and verify the selected derived bundle.",
                        field_path="bundle",
                    )
                )
            return loaded, None
        try:
            from zipline.data.bundles import bundles, load, register

            registered_name: str | None = None
            if locator.bundle_name not in bundles:
                register(
                    locator.bundle_name,
                    lambda *args, **kwargs: None,
                    calendar_name="XNYS",
                    create_writers=False,
                )
                registered_name = locator.bundle_name
            loaded = load(
                locator.bundle_name,
                {"ZIPLINE_ROOT": str(locator.cache_path)},
                locator.bundle_timestamp,
            )
            return loaded, registered_name
        except _EngineFailure:
            raise
        except Exception as error:
            raise _EngineFailure(
                self._error(
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The exact Zipline bundle could not be opened.",
                    "Rebuild the bundle from the verified snapshot and retry.",
                    field_path="bundle",
                    checksum=locator.bundle_checksum,
                )
            ) from error

    @staticmethod
    def _unregister_bundle(name: str) -> None:
        try:
            from zipline.data.bundles import unregister

            unregister(name)
        except Exception:
            pass

    @staticmethod
    def _bundle_calendar(bundle_data: object) -> object:
        reader = getattr(bundle_data, "equity_daily_bar_reader", None)
        calendar = getattr(reader, "trading_calendar", None)
        if calendar is None:
            reader = getattr(bundle_data, "equity_minute_bar_reader", None)
            calendar = getattr(reader, "trading_calendar", None)
        if calendar is None:
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.STORAGE_IO,
                    "The exact bundle has no pinned XNYS calendar.",
                    "Materialize the bundle with the pinned XNYS calendar.",
                    field_path="bundle.calendar",
                )
            )
        return calendar

    @staticmethod
    def _simulation_sessions(calendar: object, start: date, end: date) -> tuple[date, ...]:
        method = getattr(calendar, "sessions_in_range", None)
        if not callable(method):
            return ()
        try:
            import pandas as pd  # type: ignore[import-untyped]

            return tuple(_session_from_value(value) for value in method(pd.Timestamp(start), pd.Timestamp(end)) if _session_from_value(value) is not None)
        except Exception as error:
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.STORAGE_IO,
                    "The pinned XNYS session calendar could not be read.",
                    "Use the calendar recorded by the verified bundle.",
                    field_path="calendar.sessions",
                )
            ) from error

    def _decision_service(self, config: ResolvedConfig, calendar: _EngineCalendar) -> object:
        if self.decision_delivery is not None:
            return self.decision_delivery
        # The run has already opened and pinned the snapshot before entering
        # the event loop.  Passing no verifier prevents CausalDecisionDelivery
        # from resolving the mutable metadata store again for every signal;
        # it still receives the immutable handle and the projected reader.
        return CausalDecisionDelivery(
            snapshot_reader=self.snapshot_reader,
            snapshot_manager=None,
            calendar=calendar,
            resolved_config=config,
        )

    @staticmethod
    def _deliver(
        delivery: object,
        snapshot: object,
        session: date,
        portfolio: object,
        config: ResolvedConfig,
        calendar: _EngineCalendar,
    ) -> DecisionDeliveryResult:
        method = getattr(delivery, "deliver", None)
        if not callable(method):
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    "No causal decision-delivery service is configured.",
                    "Inject CausalDecisionDelivery with a verified snapshot history reader.",
                    field_path="decision_delivery",
                )
            )
        try:
            result = method(
                snapshot,
                session,
                portfolio,
                universe=config.data.universe,
                position_count=config.strategy.position_count,
                execution_session=calendar.next_session(session),
            )
        except TypeError:
            result = method(snapshot, session, portfolio)
        if isinstance(result, Err):
            raise _EngineFailure(result.errors[0])
        value = result.value if isinstance(result, Ok) else result
        if not isinstance(value, DecisionDeliveryResult):
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    "Decision delivery returned an unsupported result.",
                    "Use the platform DecisionDeliveryResult contract.",
                    field_path="decision_delivery.result",
                )
            )
        return value

    @staticmethod
    def _submit_order(context: object, intent: object) -> str | None:
        asset_finder = getattr(context, "asset_finder", None)
        lookup = getattr(asset_finder, "lookup_symbol", None)
        if not callable(lookup):
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    "Zipline cannot resolve a configured symbol asset.",
                    "Rebuild the bundle with the configured universe metadata.",
                    field_path="asset_finder",
                )
            )
        try:
            import pandas as pd  # type: ignore[import-untyped]

            as_of = pd.Timestamp(getattr(context, "datetime", datetime.now(UTC)))
            # Zipline asset lifetimes are stored as timezone-naive session
            # labels.  The event clock is UTC-aware, so normalize before the
            # strict symbol lookup to avoid a naive/aware comparison failure.
            if as_of.tzinfo is not None:
                as_of = as_of.tz_localize(None)
            asset = lookup(intent.symbol, as_of_date=as_of)
            if _MarketOrder is None:
                raise RuntimeError("MarketOrder is unavailable")
            order_method = getattr(context, "order", None)
            if not callable(order_method):
                return None
            return cast(str | None, order_method(asset, intent.requested_quantity, style=_MarketOrder()))
        except Exception as error:
            raise _EngineFailure(
                BacktestEngine._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    f"The configured symbol {intent.symbol} could not be submitted to Zipline.",
                    "Verify the exact bundle asset metadata and retry.",
                    symbol=intent.symbol,
                    session=intent.signal_session,
                )
            ) from error

    @staticmethod
    def _runtime_cash(runtime: Mapping[str, object | None]) -> object:
        algorithm = runtime.get("algorithm")
        portfolio = getattr(algorithm, "portfolio", None)
        value = getattr(portfolio, "cash", None)
        return INITIAL_PORTFOLIO_EQUITY if value is None else value

    @staticmethod
    def _runtime_positions(runtime: Mapping[str, object | None]) -> Mapping[object, object]:
        algorithm = runtime.get("algorithm")
        portfolio = getattr(algorithm, "portfolio", None)
        value = getattr(portfolio, "positions", None)
        return value if isinstance(value, Mapping) else {}

    def _algorithm(
        self,
        *,
        bundle_data: object,
        calendar: object,
        start: date,
        end: date,
        capital_base: Decimal,
        initialize: Callable[[object], None],
        handle_data: Callable[[object, object], None],
        blotter: CashSafeOpenBlotter,
    ) -> tuple[object, object]:
        try:
            import pandas as pd  # type: ignore[import-untyped]
            from zipline.algorithm import TradingAlgorithm
            from zipline.data.data_portal import DataPortal
            from zipline.finance.metrics import load as load_metrics
            from zipline.finance.trading import SimulationParameters
            from zipline.pipeline.data import USEquityPricing
            from zipline.pipeline.loaders import USEquityPricingLoader

            daily_reader = bundle_data.equity_daily_bar_reader
            minute_reader = bundle_data.equity_minute_bar_reader
            adjustment_reader = bundle_data.adjustment_reader
            data_portal = DataPortal(
                bundle_data.asset_finder,
                trading_calendar=calendar,
                first_trading_day=minute_reader.first_trading_day,
                equity_minute_reader=minute_reader,
                equity_daily_reader=daily_reader,
                adjustment_reader=adjustment_reader,
                future_minute_reader=minute_reader,
                future_daily_reader=daily_reader,
            )

            pricing_loader = USEquityPricingLoader.without_fx(
                daily_reader, adjustment_reader
            )

            def get_pipeline_loader(column: object) -> object:
                if column in USEquityPricing.columns:
                    return pricing_loader
                raise ValueError(f"No PipelineLoader registered for column {column!r}")

            benchmark = bundle_data.asset_finder.lookup_symbol(
                "SPY", as_of_date=pd.Timestamp(start)
            )
            benchmark_sid = benchmark.sid
            metrics = self.metrics_set
            if metrics is None:
                # Zipline 3.1.1's default metrics use np.NINF, removed by
                # NumPy 2.x.  Add the compatibility name only for the run.
                import numpy as np  # type: ignore[import-untyped]

                if not hasattr(np, "NINF"):
                    setattr(np, "NINF", -np.inf)
                metrics = load_metrics("default")
            algorithm = TradingAlgorithm(
                data_portal=data_portal,
                trading_calendar=calendar,
                sim_params=SimulationParameters(
                    start_session=pd.Timestamp(start),
                    end_session=pd.Timestamp(end),
                    trading_calendar=calendar,
                    capital_base=float(capital_base),
                    data_frequency="daily",
                ),
                get_pipeline_loader=get_pipeline_loader,
                initialize=initialize,
                handle_data=handle_data,
                benchmark_sid=benchmark_sid,
                metrics_set=metrics,
                blotter=blotter,
            )
            return algorithm, data_portal
        except _EngineFailure:
            raise
        except Exception as error:
            raise _EngineFailure(
                self._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    "The pinned Zipline event loop could not be configured.",
                    "Verify the locked Zipline bundle, XNYS calendar, and daily data projection.",
                    field_path="zipline",
                )
            ) from error

    def _extract_output(
        self,
        performance: object,
        *,
        submitted: Mapping[str, object],
        blotter: CashSafeOpenBlotter,
        decisions: Sequence[StrategyDecision],
        asset_finder: object,
    ) -> CoreBacktestOutput:
        rows = tuple(self._performance_rows(performance))
        if not rows:
            raise _EngineFailure(
                self._static_error(
                    ErrorCategory.BACKTEST_EXECUTION,
                    "Zipline produced no daily performance rows.",
                    "Run the exact bundle over at least one completed XNYS session.",
                    field_path="performance",
                )
            )
        states: list[PortfolioState] = []
        returns: list[DailyReturn] = []
        previous_equity: Decimal | None = None
        for session, row in rows:
            state = self._portfolio_state(session, row, asset_finder)
            if not states and state.cash_balance != INITIAL_PORTFOLIO_EQUITY:
                raise _EngineFailure(
                    self._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "The first portfolio state did not start with USD 100000 cash.",
                        "Retry with the fixed initial-equity configuration and a clean run.",
                        field_path="portfolio.initial_cash",
                    )
                )
            states.append(state)
            raw_return = _row_value(row, "returns", None)
            if raw_return is None or not _finite_number(raw_return):
                value = Decimal("0") if previous_equity is None else state.portfolio_equity / previous_equity - Decimal("1")
            else:
                value = _decimal_number(raw_return, "returns")
            if value <= Decimal("-1"):
                raise _EngineFailure(
                    self._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "Zipline produced a daily return at or below negative 100 percent.",
                        "Inspect the bundle prices and ledger action stream before retrying.",
                        field_path="portfolio.returns",
                        session=session,
                    )
                )
            returns.append(DailyReturn(session=session, return_value=value))
            previous_equity = state.portfolio_equity

        fills: list[FillRecord] = []
        fill_ordinals: dict[str, int] = {}
        for execution in blotter.execution_records:
            intent = submitted.get(execution.order_id)
            if intent is None:
                raise _EngineFailure(
                    self._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "A Zipline fill was not associated with a platform order intent.",
                        "Retry from the exact bundle without mutating the event-loop order map.",
                        field_path="fills.order_id",
                    )
                )
            ordinal = fill_ordinals.get(intent.order_id, 0)
            fill_ordinals[intent.order_id] = ordinal + 1
            fill_id = deterministic_fill_id(
                order_id=intent.order_id,
                symbol=intent.symbol,
                session=execution.session,
                quantity=execution.quantity,
                ordinal=ordinal,
            )
            fills.append(
                FillRecord(
                    fill_id=fill_id,
                    order_id=intent.order_id,
                    symbol=intent.symbol,
                    session=execution.session,
                    quantity=execution.quantity,
                    ordinal=ordinal,
                    base_adjusted_open=execution.base_adjusted_open,
                    fill_price=execution.fill_price,
                    gross_notional=execution.gross_notional,
                    commission=execution.commission,
                    slippage_cost=execution.slippage_cost,
                )
            )

        reasons: dict[str, str] = {}
        for unfilled in blotter.unfilled_orders:
            intent = submitted.get(unfilled.order_id)
            if intent is not None:
                reasons[intent.order_id] = unfilled.reason
        orders: list[OrderRecord] = []
        for internal_id, intent in submitted.items():
            del internal_id
            filled_quantity = sum(
                fill.quantity for fill in fills if fill.order_id == intent.order_id
            )
            requested = intent.requested_quantity
            if abs(filled_quantity) >= abs(requested):
                status = OrderStatus.FILLED
                reason = None
            elif filled_quantity:
                status = OrderStatus.PARTIALLY_FILLED
                reason = reasons.get(intent.order_id, "execution_remainder")
            else:
                status = OrderStatus.UNFILLED
                reason = reasons.get(intent.order_id, "execution_window_ended")
            orders.append(
                intent.to_order_record().__class__(
                    order_id=intent.order_id,
                    signal_session=intent.signal_session,
                    execution_session=intent.execution_session,
                    symbol=intent.symbol,
                    requested_quantity=requested,
                    ordinal=intent.ordinal,
                    decision_rank=intent.decision_rank,
                    status=status,
                    unfilled_reason=reason,
                )
            )
        orders.sort(key=lambda item: (item.signal_session, item.execution_session, item.ordinal, item.symbol))
        fills.sort(key=lambda item: (item.session, item.order_id, item.ordinal))
        decisions_sorted = tuple(sorted(decisions, key=lambda item: (item.signal_session, item.symbol)))
        try:
            return CoreBacktestOutput(
                orders=tuple(orders),
                fills=tuple(fills),
                portfolio_states=tuple(states),
                daily_returns=tuple(returns),
                strategy_decisions=decisions_sorted,
                initial_equity=INITIAL_PORTFOLIO_EQUITY,
            )
        except (TypeError, ValueError) as error:
            raise _EngineFailure(
                self._static_error(
                    ErrorCategory.BACKTEST_INVARIANT,
                    "The extracted Zipline output failed platform accounting invariants.",
                    "Inspect the exact ledger transactions and retry from the verified snapshot.",
                    field_path="core_output",
                )
            ) from error

    @staticmethod
    def _performance_rows(performance: object) -> Iterable[tuple[date, object]]:
        iterrows = getattr(performance, "iterrows", None)
        if callable(iterrows):
            for index, row in iterrows():
                session = _session_from_value(index) or _session_from_value(_row_value(row, "period_close", None))
                if session is not None:
                    yield session, row
            return
        if isinstance(performance, Mapping):
            for key, value in performance.items():
                session = _session_from_value(key)
                if session is not None:
                    yield session, value

    @classmethod
    def _portfolio_state(cls, session: date, row: object, asset_finder: object) -> PortfolioState:
        cash = _decimal_number(_row_value(row, "ending_cash", _row_value(row, "cash", INITIAL_PORTFOLIO_EQUITY)), "ending_cash")
        if cash < 0 and cash > Decimal("-0.000001"):
            cash = Decimal("0")
        if cash < 0:
            raise _EngineFailure(
                cls._static_error(
                    ErrorCategory.BACKTEST_INVARIANT,
                    "Zipline produced negative cash after a fill.",
                    "Inspect the custom blotter cash cap and transaction costs.",
                    field_path="portfolio.cash_balance",
                    session=session,
                )
            )
        positions: list[Position] = []
        for entry in _position_entries(_row_value(row, "positions", ())):
            amount = _integer_number(_row_value(entry, "amount", _row_value(entry, "quantity", 0)), "position.amount")
            if amount == 0:
                continue
            if amount < 0:
                raise _EngineFailure(
                    cls._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "Zipline produced a negative position quantity.",
                        "Retry with the long-only cash-safe blotter and verified bundle.",
                        field_path="portfolio.positions",
                        session=session,
                    )
                )
            sid = _row_value(entry, "sid", None)
            asset = _row_value(entry, "asset", None)
            if asset is None and sid is not None:
                try:
                    asset = asset_finder.retrieve_asset(int(sid))
                except Exception:
                    asset = None
            symbol = _symbol(asset) if asset is not None else str(_row_value(entry, "symbol", sid)).strip().upper()
            price_value = _row_value(entry, "last_sale_price", _row_value(entry, "price", None))
            if price_value is None:
                value = _row_value(entry, "value", None)
                price_value = Decimal(str(value)) / Decimal(amount) if value is not None else None
            if price_value is None:
                raise _EngineFailure(
                    cls._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "Zipline omitted a mark price for an open position.",
                        "Use a complete daily raw-price projection in the verified bundle.",
                        field_path="portfolio.positions.last_sale_price",
                        symbol=symbol,
                        session=session,
                    )
                )
            price = quantize_money(_decimal_number(price_value, "position.mark_price"))
            value = quantize_money(Decimal(amount) * price)
            positions.append(Position(symbol=symbol, quantity=amount, mark_price=price, market_value=value))
        positions.sort(key=lambda item: item.symbol)
        gross = quantize_money(sum((item.market_value for item in positions), Decimal("0")))
        equity = quantize_money(cash + gross)
        reported = _row_value(row, "portfolio_value", None)
        if reported is not None and _finite_number(reported):
            if abs(_decimal_number(reported, "portfolio_value") - equity) > Decimal("0.010001"):
                raise _EngineFailure(
                    cls._static_error(
                        ErrorCategory.BACKTEST_INVARIANT,
                        "Portfolio equity does not reconcile to cash plus marked positions.",
                        "Inspect the ledger action stream and raw-price bundle projection.",
                        field_path="portfolio.portfolio_equity",
                        session=session,
                    )
                )
        if equity <= 0:
            raise _EngineFailure(
                cls._static_error(
                    ErrorCategory.BACKTEST_INVARIANT,
                    "Portfolio equity is not positive.",
                    "Inspect the verified bundle prices and transaction ledger.",
                    field_path="portfolio.portfolio_equity",
                    session=session,
                )
            )
        leverage = gross / equity
        if leverage < 0 or leverage > Decimal("1.000000000001"):
            raise _EngineFailure(
                cls._static_error(
                    ErrorCategory.BACKTEST_INVARIANT,
                    "Portfolio leverage exceeded the long-only one-times bound.",
                    "Retry with max leverage set to 1.0 and the cash-safe blotter.",
                    field_path="portfolio.leverage",
                    session=session,
                )
            )
        return PortfolioState(
            session=session,
            cash_balance=quantize_money(cash),
            positions=tuple(positions),
            gross_exposure=gross,
            portfolio_equity=equity,
            leverage=leverage,
        )

    def _emit_progress(
        self,
        callback: Callable[..., object] | None,
        session: date,
        sessions: Sequence[date],
        total: int,
    ) -> None:
        if callback is None:
            return
        if getattr(self, "_progress_started", None) is None:
            self._progress_started = _time.monotonic()
            self._progress_job_id = uuid4()
        update = ProgressUpdate(
            job_id=cast(UUID, self._progress_job_id),
            operation=JobOperation.BACKTEST,
            state=JobState.RUNNING,
            stage=JobStage.EXECUTING,
            completed_units=min(len(sessions), sessions.index(session) + 1),
            total_units=total,
            elapsed_seconds=Decimal(str(max(0.0, _time.monotonic() - self._progress_started))),
        )
        try:
            callback(update)
        except TypeError:
            try:
                callback(update.completed_units, update.total_units, session)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _static_error(
        category: ErrorCategory,
        message: str,
        corrective_action: str,
        *,
        field_path: str | None = None,
        symbol: str | None = None,
        session: date | None = None,
        checksum: str | None = None,
    ) -> ActionableError:
        return ActionableError(
            operation=BacktestEngine.operation_name,
            category=category,
            message=message,
            corrective_action=corrective_action,
            field_path=field_path,
            symbol=symbol,
            session=session,
            checksum=checksum,
        )

    def _error(self, category: ErrorCategory, message: str, corrective_action: str, **kwargs: object) -> ActionableError:
        return self._static_error(category, message, corrective_action, **cast(dict[str, object], kwargs))

    @staticmethod
    def _input_error(error: BaseException) -> ActionableError:
        return BacktestEngine._static_error(
            ErrorCategory.CONFIGURATION_INVALID_VALUE,
            str(error).splitlines()[0] or "Invalid backtest input.",
            "Use one verified snapshot bundle and a validated backtest configuration.",
        )


ZiplineBacktestEngine = BacktestEngine


def _date_attr(value: object | None, names: Sequence[str]) -> date | None:
    if value is None:
        return None
    for name in names:
        candidate = getattr(value, name, None)
        converted = _session_from_value(candidate)
        if converted is not None:
            return converted
    return None


def _row_value(row: object, name: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            value = getter(name, default)
            return default if value is None and default is not None else value
        except Exception:
            pass
    return getattr(row, name, default)


def _position_entries(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in ("sid", "asset", "symbol", "amount", "quantity")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _finite_number(value: object) -> bool:
    try:
        return _decimal_number(value, "number").is_finite()
    except (TypeError, ValueError, InvalidOperation):
        return False


def _decimal_number(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _integer_number(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole-share integer")
    if isinstance(value, int):
        return value
    try:
        decimal_value = _decimal_number(value, name)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole-share integer") from error
    if decimal_value != decimal_value.to_integral_value():
        raise ValueError(f"{name} must be a whole-share integer")
    return int(decimal_value)
