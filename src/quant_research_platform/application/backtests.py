"""Audited backtest orchestration at the application boundary.

The application service in this module deliberately does not import Zipline,
DuckDB, MLflow, or a storage implementation.  It pins one verified snapshot,
materializes the exact snapshot-specific bundle, delegates the event loop to
``BacktestEngine``, and audits the complete core output before evaluation or
terminal experiment finalization.

The tracker and evaluator are structural ports.  This keeps the service usable
with the small local fakes used by contract tests and with the concrete tracker
and evaluation adapters added by later implementation waves, without allowing
those infrastructure details to leak into the application layer.
"""

# Ruff's line-length rule is intentionally relaxed here because this boundary
# carries long, display-safe diagnostic messages with structured field paths.
# ruff: noqa: E501, B009

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Protocol, cast
from uuid import UUID, uuid4

from ..config.models import ResolvedConfig
from ..config.serializer import ConfigurationSerializer
from ..domain.canonical import sha256_bytes, sha256_canonical_json
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    OrderStatus,
    quantize_money,
)
from ..domain.market import DateRange


class RunTrackerPort(Protocol):
    """Tracker boundary used by :class:`BacktestService`.

    Implementations may expose ``allocate_run`` or the more complete
    ``create_run`` spelling.  The service inspects the method signature and
    passes only the fields that implementation accepts.
    """

    def allocate_run(self, **kwargs: object) -> object:
        """Allocate and persist a running Run_ID before any snapshot read."""


class EvaluationPort(Protocol):
    """Evaluation boundary receiving the complete audited core output."""

    def evaluate(self, **kwargs: object) -> object:
        """Evaluate one audited core output against the pinned snapshot."""


class SnapshotPort(Protocol):
    """Minimal verified snapshot boundary."""

    def open_verified(self, snapshot_id: str) -> object:
        """Open and verify one exact immutable snapshot."""


class BundlePort(Protocol):
    """Materialize one exact snapshot-specific derived bundle."""

    def materialize(self, snapshot: object) -> object:
        """Return a checksum-verified exact bundle locator."""


class EnginePort(Protocol):
    """Backtest engine boundary."""

    def run(
        self,
        bundle: object,
        request: object,
        config: ResolvedConfig,
        progress: Callable[..., object] | None = None,
    ) -> object:
        """Execute the exact bundle and return core output or a Result."""


class BacktestClock(Protocol):
    """UTC clock seam for operational tracker timestamps."""

    def utc_now(self) -> datetime:
        """Return an aware UTC timestamp."""


class _SystemBacktestClock:
    def utc_now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """One explicit request to run against one pinned Snapshot_ID."""

    snapshot_id: str
    evaluation_range: DateRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-blank string")
        object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        if self.evaluation_range is not None and not isinstance(
            self.evaluation_range, DateRange
        ):
            raise TypeError("evaluation_range must be a DateRange or None")

    @property
    def range(self) -> DateRange | None:
        """Compatibility alias for callers using ``range`` terminology."""

        return self.evaluation_range


@dataclass(frozen=True, slots=True)
class AuditReport:
    """The immutable accounting audit attached to a successful run."""

    output: object
    unfilled_orders: tuple[object, ...] = ()
    unfilled_diagnostics: tuple[ActionableError, ...] = ()
    diagnostics: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.unfilled_orders, tuple):
            raise TypeError("unfilled_orders must be an immutable tuple")
        for name, value in (
            ("unfilled_diagnostics", self.unfilled_diagnostics),
            ("diagnostics", self.diagnostics),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if any(not isinstance(item, ActionableError) for item in value):
                raise TypeError(f"{name} must contain ActionableError values")

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    @property
    def audited_output(self) -> object:
        """Alias used by evaluation and experiment adapters."""

        return self.output


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete application result after audit, evaluation, and finalization."""

    run_id: object
    snapshot_id: str
    evaluation_range: DateRange
    core_output: object
    audit: AuditReport
    evaluation: object | None = None
    limitation_disclosure: LimitationDisclosure = LimitationDisclosure.current()
    diagnostics: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        if self.run_id is None:
            raise ValueError("run_id must be present")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-blank string")
        if not isinstance(self.evaluation_range, DateRange):
            raise TypeError("evaluation_range must be a DateRange")
        if not isinstance(self.audit, AuditReport) or not self.audit.passed:
            raise ValueError("successful BacktestResult requires a passed AuditReport")
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be LimitationDisclosure")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, ActionableError) for item in self.diagnostics
        ):
            raise TypeError("diagnostics must contain ActionableError values")
        object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())

    @property
    def output(self) -> object:
        """Compatibility alias for the complete core output."""

        return self.core_output


@dataclass(frozen=True, slots=True)
class BacktestFinalization:
    """Tracker-facing carrier that adds immutable publication references.

    The original :class:`BacktestResult` remains unchanged for application
    callers and legacy test doubles.  Concrete trackers receive this carrier
    only when a production manifest publisher is configured.
    """

    backtest_result: BacktestResult
    manifest: object
    manifest_checksum: str
    manifest_uri: str
    artifacts: tuple[object, ...]
    ended_at: datetime

    @property
    def run_id(self) -> object:
        return self.backtest_result.run_id

    @property
    def snapshot_id(self) -> str:
        return self.backtest_result.snapshot_id

    @property
    def evaluation_range(self) -> DateRange:
        return self.backtest_result.evaluation_range

    @property
    def core_output(self) -> object:
        return self.backtest_result.core_output

    @property
    def output(self) -> object:
        return self.backtest_result.core_output

    @property
    def audit(self) -> AuditReport:
        return self.backtest_result.audit

    @property
    def evaluation(self) -> object | None:
        return self.backtest_result.evaluation

    @property
    def limitation_disclosure(self) -> LimitationDisclosure:
        return self.backtest_result.limitation_disclosure

    @property
    def diagnostics(self) -> tuple[ActionableError, ...]:
        return self.backtest_result.diagnostics


# Descriptive aliases used by application composition roots and tests.
BacktestOutput = BacktestResult
BacktestAudit = AuditReport


@dataclass(frozen=True, slots=True)
class _AuditContext:
    commission_bps: Decimal
    slippage_bps: Decimal
    initial_equity: Decimal = INITIAL_PORTFOLIO_EQUITY

    @property
    def commission_rate(self) -> Decimal:
        return self.commission_bps / Decimal("10000")

    @property
    def slippage_rate(self) -> Decimal:
        return self.slippage_bps / Decimal("10000")


class _BacktestFailure(Exception):
    """Internal carrier for already-sanitized application errors."""

    def __init__(self, errors: Sequence[ActionableError]) -> None:
        values = tuple(errors)
        if not values:
            raise ValueError("backtest failure requires at least one error")
        super().__init__(values[0].message)
        self.errors = values


class _SystemClock:
    """Compatibility clock accepting the same interface as BacktestClock."""

    def utc_now(self) -> datetime:
        return datetime.now(UTC)


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite decimal") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _date_value(value: object, name: str) -> date | None:
    if value is None:
        return None
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
        if isinstance(converted, datetime):
            return converted.date()
        if isinstance(converted, date):
            return converted
    raise ValueError(f"{name} must identify a calendar date")


def _field(value: object, names: str | Sequence[str], default: object = None) -> object:
    candidates = (names,) if isinstance(names, str) else tuple(names)
    if isinstance(value, Mapping):
        for name in candidates:
            if name in value:
                return value[name]
        return default
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _quantity(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole-share integer")
    if isinstance(value, int):
        return value
    if (
        isinstance(value, Decimal)
        and value.is_finite()
        and value == value.to_integral_value()
    ):
        return int(value)
    raise ValueError(f"{name} must be a whole-share integer")


def _error(
    message: str,
    *,
    field_path: str,
    session: date | None = None,
    symbol: str | None = None,
    operation: str = "backtest.audit",
    category: ErrorCategory = ErrorCategory.BACKTEST_INVARIANT,
) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=message,
        corrective_action=(
            "Inspect the pinned ledger output and retry from the same verified "
            "snapshot after correcting the violated invariant."
        ),
        field_path=field_path,
        session=session,
        symbol=symbol,
    )


def _config_costs(config: object) -> _AuditContext:
    execution = getattr(config, "execution", config)
    commission = _decimal(
        getattr(execution, "commission_bps", getattr(config, "commission_bps", 5)),
        "commission_bps",
    )
    slippage = _decimal(
        getattr(execution, "slippage_bps", getattr(config, "slippage_bps", 10)),
        "slippage_bps",
    )
    initial = _decimal(
        getattr(
            execution,
            "initial_equity_usd",
            getattr(config, "initial_equity", INITIAL_PORTFOLIO_EQUITY),
        ),
        "initial_equity",
    )
    if commission < 0 or slippage < 0:
        raise ValueError("transaction costs must be non-negative")
    if initial != INITIAL_PORTFOLIO_EQUITY:
        raise ValueError("initial equity must be fixed at USD 100000")
    return _AuditContext(commission, slippage, initial)


def _order_status(value: object) -> OrderStatus:
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str):
        raise ValueError(f"unsupported order status: {value!r}")
    try:
        return OrderStatus(normalized)
    except ValueError as error:
        raise ValueError(f"unsupported order status: {value!r}") from error


def _audit_fill_costs(fill: object, context: _AuditContext) -> None:
    symbol = cast(str | None, _field(fill, "symbol"))
    session = _date_value(
        _field(fill, ("session", "execution_session")), "fill.session"
    )
    quantity = _quantity(_field(fill, ("quantity", "amount")), "fill.quantity")
    if quantity == 0:
        raise _BacktestFailure(
            (
                _error(
                    "A fill quantity was zero.",
                    field_path="fills.quantity",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    base = _decimal(
        _field(fill, ("base_adjusted_open", "base_open")), "fill.base_adjusted_open"
    )
    price = _decimal(_field(fill, ("fill_price", "price")), "fill.fill_price")
    gross = _decimal(
        _field(fill, ("gross_notional", "notional")), "fill.gross_notional"
    )
    commission = _decimal(_field(fill, "commission"), "fill.commission")
    slippage_cost = _decimal(_field(fill, "slippage_cost"), "fill.slippage_cost")
    if base <= 0 or price <= 0:
        raise _BacktestFailure(
            (
                _error(
                    "A fill used a non-positive price.",
                    field_path="fills.price",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    with localcontext() as decimal_context:
        decimal_context.prec = 40
        expected_price = base * (
            Decimal("1") + context.slippage_rate
            if quantity > 0
            else Decimal("1") - context.slippage_rate
        )
    expected_price = quantize_money(expected_price)
    if expected_price <= 0 or price != expected_price:
        raise _BacktestFailure(
            (
                _error(
                    "A fill price does not equal the configured adverse-slippage formula.",
                    field_path="fills.fill_price",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    expected_gross = quantize_money(abs(quantity) * price)
    if gross != expected_gross:
        raise _BacktestFailure(
            (
                _error(
                    "A fill gross notional does not equal actual whole-share fill notional.",
                    field_path="fills.gross_notional",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    expected_commission = quantize_money(expected_gross * context.commission_rate)
    if commission != expected_commission:
        raise _BacktestFailure(
            (
                _error(
                    "A fill commission does not equal the configured cost formula on actual notional.",
                    field_path="fills.commission",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    expected_slippage = quantize_money(abs(price - base) * abs(quantity))
    if slippage_cost != expected_slippage:
        raise _BacktestFailure(
            (
                _error(
                    "A fill slippage cost does not equal adverse price difference times quantity.",
                    field_path="fills.slippage_cost",
                    session=session,
                    symbol=symbol,
                ),
            )
        )


def _audit_position(position: object, *, session: date | None) -> Decimal:
    symbol = cast(str | None, _field(position, "symbol"))
    quantity = _quantity(_field(position, ("quantity", "amount")), "position.quantity")
    if quantity <= 0:
        raise _BacktestFailure(
            (
                _error(
                    "A position quantity was not a positive whole share count.",
                    field_path="portfolio.positions.quantity",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    mark_price = _decimal(
        _field(position, ("mark_price", "price", "last_sale_price")),
        "position.mark_price",
    )
    market_value = _decimal(
        _field(position, ("market_value", "value")), "position.market_value"
    )
    if mark_price <= 0:
        raise _BacktestFailure(
            (
                _error(
                    "A position mark price was not positive.",
                    field_path="portfolio.positions.mark_price",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    expected_value = quantize_money(Decimal(quantity) * quantize_money(mark_price))
    if market_value != expected_value:
        raise _BacktestFailure(
            (
                _error(
                    "A position market value does not reconcile to quantity times mark price.",
                    field_path="portfolio.positions.market_value",
                    session=session,
                    symbol=symbol,
                ),
            )
        )
    return market_value


def _audit_state(state: object, context: _AuditContext) -> None:
    session = _date_value(_field(state, "session"), "portfolio.session")
    cash = _decimal(_field(state, ("cash_balance", "cash")), "portfolio.cash_balance")
    gross = _decimal(
        _field(state, ("gross_exposure", "gross_position_value", "gross")),
        "portfolio.gross_exposure",
    )
    equity = _decimal(
        _field(state, ("portfolio_equity", "equity", "portfolio_value")),
        "portfolio.portfolio_equity",
    )
    leverage = _decimal(_field(state, "leverage"), "portfolio.leverage")
    if cash < 0:
        raise _BacktestFailure(
            (
                _error(
                    "Portfolio cash became negative.",
                    field_path="portfolio.cash_balance",
                    session=session,
                ),
            )
        )
    positions_value = _field(state, ("positions", "holdings"), ())
    if isinstance(positions_value, Mapping):
        positions = tuple(positions_value.values())
    elif isinstance(positions_value, Iterable) and not isinstance(
        positions_value, (str, bytes, bytearray)
    ):
        positions = tuple(positions_value)
    else:
        positions = ()
    calculated_gross = Decimal("0")
    symbols: set[str] = set()
    for position in positions:
        symbol = str(_field(position, ("symbol", "ticker"), "")).strip().upper()
        if symbol in symbols:
            raise _BacktestFailure(
                (
                    _error(
                        "A portfolio state contained duplicate position symbols.",
                        field_path="portfolio.positions",
                        session=session,
                        symbol=symbol,
                    ),
                )
            )
        symbols.add(symbol)
        calculated_gross += _audit_position(position, session=session)
    calculated_gross = quantize_money(calculated_gross)
    if gross < 0 or gross != calculated_gross:
        raise _BacktestFailure(
            (
                _error(
                    "Gross exposure does not equal the sum of marked position values.",
                    field_path="portfolio.gross_exposure",
                    session=session,
                ),
            )
        )
    reconciled_equity = quantize_money(cash + gross)
    if abs(equity - reconciled_equity) > Decimal("0.01"):
        raise _BacktestFailure(
            (
                _error(
                    "Portfolio equity does not reconcile to cash plus marked positions.",
                    field_path="portfolio.portfolio_equity",
                    session=session,
                ),
            )
        )
    if equity <= 0:
        raise _BacktestFailure(
            (
                _error(
                    "Portfolio equity was not positive.",
                    field_path="portfolio.portfolio_equity",
                    session=session,
                ),
            )
        )
    expected_leverage = gross / equity
    if (
        leverage < 0
        or leverage > Decimal("1")
        or abs(leverage - expected_leverage) > Decimal("0.000000000001")
    ):
        raise _BacktestFailure(
            (
                _error(
                    "Portfolio gross leverage was outside the inclusive [0, 1] bound or did not reconcile.",
                    field_path="portfolio.leverage",
                    session=session,
                ),
            )
        )


def _audit_fill_transitions(states: Sequence[object], fills: Sequence[object]) -> None:
    """Reconcile cash and actual-share quantities across fill sessions."""

    fills_by_session: dict[date, list[object]] = {}
    for fill in fills:
        session = _date_value(_field(fill, "session"), "fill.session")
        if session is None:
            continue
        fills_by_session.setdefault(session, []).append(fill)

    previous_state: object | None = None
    for state in states:
        session = _date_value(_field(state, "session"), "portfolio.session")
        if session is None:
            continue
        session_fills = fills_by_session.get(session, ())
        if not session_fills:
            previous_state = state
            continue
        if previous_state is None:
            raise _BacktestFailure(
                (
                    _error(
                        "A fill occurred before an initial portfolio state was available.",
                        field_path="fills.session",
                        session=session,
                    ),
                )
            )

        previous_positions_value = _field(previous_state, ("positions", "holdings"), ())
        current_positions_value = _field(state, ("positions", "holdings"), ())

        def quantities(value: object) -> dict[str, int]:
            entries = (
                tuple(value.values())
                if isinstance(value, Mapping)
                else tuple(value)
                if isinstance(value, Iterable)
                and not isinstance(value, (str, bytes, bytearray))
                else ()
            )
            result: dict[str, int] = {}
            for position in entries:
                symbol = str(_field(position, ("symbol", "ticker"), "")).strip().upper()
                result[symbol] = _quantity(
                    _field(position, ("quantity", "amount")),
                    "position.quantity",
                )
            return result

        expected_positions = quantities(previous_positions_value)
        for fill in session_fills:
            symbol = str(_field(fill, "symbol", "")).strip().upper()
            expected_positions[symbol] = expected_positions.get(symbol, 0) + _quantity(
                _field(fill, "quantity"), "fill.quantity"
            )
            if expected_positions[symbol] < 0:
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill reduced a position below zero shares.",
                            field_path="portfolio.positions.quantity",
                            session=session,
                            symbol=symbol,
                        ),
                    )
                )
            if expected_positions[symbol] == 0:
                del expected_positions[symbol]

        actual_positions = quantities(current_positions_value)
        if actual_positions != expected_positions:
            raise _BacktestFailure(
                (
                    _error(
                        "Portfolio positions do not reconcile to fills after the execution session.",
                        field_path="portfolio.positions",
                        session=session,
                    ),
                )
            )

        previous_cash = _decimal(
            _field(previous_state, ("cash_balance", "cash")),
            "portfolio.cash_balance",
        )
        expected_cash = previous_cash
        for fill in session_fills:
            gross = _decimal(
                _field(fill, ("gross_notional", "notional")),
                "fill.gross_notional",
            )
            commission = _decimal(_field(fill, "commission"), "fill.commission")
            quantity = _quantity(_field(fill, "quantity"), "fill.quantity")
            expected_cash += gross - commission if quantity < 0 else -gross - commission
        expected_cash = quantize_money(expected_cash)
        actual_cash = quantize_money(
            _decimal(
                _field(state, ("cash_balance", "cash")),
                "portfolio.cash_balance",
            )
        )
        if actual_cash != expected_cash:
            raise _BacktestFailure(
                (
                    _error(
                        "Portfolio cash does not reconcile to fills and exact transaction costs.",
                        field_path="portfolio.cash_balance",
                        session=session,
                    ),
                )
            )
        previous_state = state


def _audit_generic_events(output: object, states: Sequence[object]) -> None:
    """Audit optional action/mark event streams exposed by richer engines."""

    state_sessions = {
        _date_value(_field(item, "session"), "portfolio.session") for item in states
    }
    for collection_name in ("actions", "corporate_actions", "marks", "valuation_marks"):
        collection = getattr(output, collection_name, ())
        if collection is None:
            continue
        if isinstance(collection, (str, bytes, bytearray)):
            raise _BacktestFailure(
                (
                    _error(
                        "The ledger event collection has an invalid shape.",
                        field_path=collection_name,
                    ),
                )
            )
        try:
            events = tuple(collection)
        except TypeError as error:
            raise _BacktestFailure(
                (
                    _error(
                        "The ledger event collection is not iterable.",
                        field_path=collection_name,
                    ),
                )
            ) from error
        previous: date | None = None
        for event in events:
            event_session = _date_value(
                _field(event, ("session", "effective_date", "ex_date")),
                f"{collection_name}.session",
            )
            if event_session is None or event_session not in state_sessions:
                raise _BacktestFailure(
                    (
                        _error(
                            "A ledger action or mark has no corresponding portfolio session.",
                            field_path=f"{collection_name}.session",
                            session=event_session,
                        ),
                    )
                )
            if previous is not None and event_session < previous:
                raise _BacktestFailure(
                    (
                        _error(
                            "Ledger actions or marks are out of chronological order.",
                            field_path=collection_name,
                            session=event_session,
                        ),
                    )
                )
            previous = event_session


def audit_core_output(
    output: object,
    config: object,
    *,
    evaluation_range: DateRange | None = None,
) -> Result[AuditReport]:
    """Fail fast while checking every output accounting and chronology invariant.

    This function intentionally returns the first actionable invariant failure;
    stopping at that boundary prevents later corrupted rows from hiding the
    diagnostic that identifies the first invalid ledger transition.  Unfilled
    orders are valid disclosed outcomes and are retained in the success report.
    """

    try:
        context = _config_costs(config)
        required = ("orders", "fills", "portfolio_states", "daily_returns")
        if any(not hasattr(output, name) for name in required):
            raise _BacktestFailure(
                (
                    _error(
                        "The engine output omitted a required audited role.",
                        field_path="core_output",
                    ),
                )
            )
        orders = tuple(getattr(output, "orders"))
        fills = tuple(getattr(output, "fills"))
        states = tuple(getattr(output, "portfolio_states"))
        returns = tuple(getattr(output, "daily_returns"))
        if not states:
            raise _BacktestFailure(
                (
                    _error(
                        "The engine output contained no portfolio marks.",
                        field_path="portfolio_states",
                    ),
                )
            )
        first_state_session = _date_value(
            _field(states[0], "session"), "portfolio.session"
        )
        if (
            _decimal(
                _field(states[0], ("cash_balance", "cash")), "portfolio.cash_balance"
            )
            != context.initial_equity
        ):
            raise _BacktestFailure(
                (
                    _error(
                        "The first portfolio state did not start with USD 100000 cash.",
                        field_path="portfolio.initial_cash",
                        session=first_state_session,
                    ),
                )
            )
        previous_session: date | None = None
        for state in states:
            session = _date_value(_field(state, "session"), "portfolio.session")
            if session is None or (
                previous_session is not None and session <= previous_session
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "Portfolio marks are not strictly chronological.",
                            field_path="portfolio_states.session",
                            session=session,
                        ),
                    )
                )
            _audit_state(state, context)
            previous_session = session

        if evaluation_range is not None:
            if (
                first_state_session is None
                or first_state_session < evaluation_range.start
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "Portfolio output begins before the requested evaluation range.",
                            field_path="evaluation_range.start",
                            session=first_state_session,
                        ),
                    )
                )
            last_state_session = _date_value(
                _field(states[-1], "session"), "portfolio.session"
            )
            if last_state_session is None or last_state_session > evaluation_range.end:
                raise _BacktestFailure(
                    (
                        _error(
                            "Portfolio output extends beyond the requested evaluation range.",
                            field_path="evaluation_range.end",
                            session=last_state_session,
                        ),
                    )
                )

        order_by_id: dict[str, object] = {}
        previous_order_key: tuple[date, date, int, str] | None = None
        for order in orders:
            order_id = str(_field(order, "order_id", ""))
            if not order_id or order_id in order_by_id:
                raise _BacktestFailure(
                    (
                        _error(
                            "Orders must have unique non-empty identifiers.",
                            field_path="orders.order_id",
                        ),
                    )
                )
            order_by_id[order_id] = order
            symbol = cast(str | None, _field(order, "symbol"))
            signal = _date_value(
                _field(order, "signal_session"), "order.signal_session"
            )
            execution = _date_value(
                _field(order, "execution_session"), "order.execution_session"
            )
            quantity = _quantity(
                _field(order, "requested_quantity"), "order.requested_quantity"
            )
            if (
                quantity == 0
                or signal is None
                or execution is None
                or execution <= signal
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "An order was not a non-zero whole-share next-session order.",
                            field_path="orders",
                            session=execution,
                            symbol=symbol,
                        ),
                    )
                )
            ordinal = _quantity(_field(order, "ordinal"), "order.ordinal")
            order_key = (signal, execution, ordinal, symbol or "")
            if previous_order_key is not None and order_key < previous_order_key:
                raise _BacktestFailure(
                    (
                        _error(
                            "Orders are not in deterministic signal/execution chronology.",
                            field_path="orders",
                            session=signal,
                            symbol=symbol,
                        ),
                    )
                )
            previous_order_key = order_key
            status = _order_status(_field(order, "status"))
            reason = _field(order, "unfilled_reason")
            if status in {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.UNFILLED,
            } and (not isinstance(reason, str) or not reason.strip()):
                raise _BacktestFailure(
                    (
                        _error(
                            "An unfilled order did not preserve an actionable reason.",
                            field_path="orders.unfilled_reason",
                            session=execution,
                            symbol=symbol,
                        ),
                    )
                )

        previous_fill_key: tuple[date, str, int] | None = None
        filled_by_order: dict[str, int] = {}
        fill_ids: set[str] = set()
        for fill in fills:
            fill_id = str(_field(fill, "fill_id", ""))
            order_id = str(_field(fill, "order_id", ""))
            if not fill_id or fill_id in fill_ids:
                raise _BacktestFailure(
                    (
                        _error(
                            "Fills must have unique non-empty identifiers.",
                            field_path="fills.fill_id",
                        ),
                    )
                )
            fill_ids.add(fill_id)
            order = order_by_id.get(order_id)
            if order is None:
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill referenced no platform order.",
                            field_path="fills.order_id",
                        ),
                    )
                )
            fill_session = _date_value(_field(fill, "session"), "fill.session")
            execution = _date_value(
                _field(order, "execution_session"), "order.execution_session"
            )
            if fill_session is None or execution != fill_session:
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill occurred outside its order execution session.",
                            field_path="fills.session",
                            session=fill_session,
                            symbol=cast(str | None, _field(fill, "symbol")),
                        ),
                    )
                )
            quantity = _quantity(_field(fill, "quantity"), "fill.quantity")
            requested = _quantity(
                _field(order, "requested_quantity"), "order.requested_quantity"
            )
            if (
                quantity == 0
                or (quantity > 0) != (requested > 0)
                or abs(quantity) > abs(requested)
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill quantity violated whole-share order direction or requested-size bounds.",
                            field_path="fills.quantity",
                            session=fill_session,
                            symbol=cast(str | None, _field(fill, "symbol")),
                        ),
                    )
                )
            ordinal = _quantity(_field(fill, "ordinal"), "fill.ordinal")
            fill_key = (fill_session, order_id, ordinal)
            if previous_fill_key is not None and fill_key < previous_fill_key:
                raise _BacktestFailure(
                    (
                        _error(
                            "Fills are not in deterministic action chronology.",
                            field_path="fills",
                            session=fill_session,
                            symbol=cast(str | None, _field(fill, "symbol")),
                        ),
                    )
                )
            previous_fill_key = fill_key
            if cast(str | None, _field(fill, "symbol")) != cast(
                str | None, _field(order, "symbol")
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill symbol did not match its order symbol.",
                            field_path="fills.symbol",
                            session=fill_session,
                        ),
                    )
                )
            filled_by_order[order_id] = filled_by_order.get(order_id, 0) + quantity
            _audit_fill_costs(fill, context)
            if fill_session not in {
                _date_value(_field(state, "session"), "portfolio.session")
                for state in states
            }:
                raise _BacktestFailure(
                    (
                        _error(
                            "A fill had no portfolio state after the fill.",
                            field_path="portfolio_states",
                            session=fill_session,
                        ),
                    )
                )

        _audit_fill_transitions(states, fills)
        for order_id, order in order_by_id.items():
            requested = _quantity(
                _field(order, "requested_quantity"), "order.requested_quantity"
            )
            filled = filled_by_order.get(order_id, 0)
            status = _order_status(_field(order, "status"))
            if abs(filled) > abs(requested):
                raise _BacktestFailure(
                    (
                        _error(
                            "Cumulative fills exceeded the requested order quantity.",
                            field_path="fills.quantity",
                        ),
                    )
                )
            expected_status = (
                OrderStatus.FILLED
                if abs(filled) == abs(requested)
                else OrderStatus.PARTIALLY_FILLED
                if filled
                else OrderStatus.UNFILLED
            )
            if status is not OrderStatus.PENDING and status is not expected_status:
                raise _BacktestFailure(
                    (
                        _error(
                            "Order status did not reconcile to its fills.",
                            field_path="orders.status",
                            symbol=cast(str | None, _field(order, "symbol")),
                        ),
                    )
                )

        previous_return_session: date | None = None
        state_session_set = {
            _date_value(_field(state, "session"), "portfolio.session")
            for state in states
        }
        for daily_return in returns:
            session = _date_value(
                _field(daily_return, "session"), "daily_return.session"
            )
            value = _decimal(
                _field(daily_return, ("return_value", "returns", "value")),
                "daily_return.return_value",
            )
            if (
                session is None
                or session not in state_session_set
                or (
                    previous_return_session is not None
                    and session <= previous_return_session
                )
                or value <= Decimal("-1")
            ):
                raise _BacktestFailure(
                    (
                        _error(
                            "Daily returns did not reconcile to finite chronological portfolio marks.",
                            field_path="daily_returns",
                            session=session,
                        ),
                    )
                )
            previous_return_session = session
        _audit_generic_events(output, states)
        unfilled = tuple(
            order
            for order in orders
            if _order_status(_field(order, "status"))
            in {OrderStatus.PARTIALLY_FILLED, OrderStatus.UNFILLED}
        )
        unfilled_diagnostics = tuple(
            _error(
                "An order was not completely filled; the disclosed reason is retained.",
                field_path="orders.unfilled_reason",
                session=_date_value(
                    _field(order, "execution_session"), "order.execution_session"
                ),
                symbol=cast(str | None, _field(order, "symbol")),
                operation="backtest.execution",
                category=ErrorCategory.BACKTEST_EXECUTION,
            )
            for order in unfilled
        )
        return Ok(
            AuditReport(
                output=output,
                unfilled_orders=unfilled,
                unfilled_diagnostics=unfilled_diagnostics,
            )
        )
    except _BacktestFailure as failure:
        return Err(failure.errors, preserve_order=True)
    except (TypeError, ValueError, ArithmeticError, KeyError) as failure:
        return Err(
            (
                _error(
                    str(failure).splitlines()[0]
                    or "The engine output failed accounting audit.",
                    field_path="core_output",
                ),
            ),
            preserve_order=True,
        )


# Public spellings kept explicit for focused tests and composition roots.
audit_backtest_output = audit_core_output
audit_accounting = audit_core_output


def _unwrap(value: object, operation: str) -> object:
    if isinstance(value, Err):
        raise _BacktestFailure(value.errors)
    if isinstance(value, Ok):
        return value.value
    if value is None:
        raise _BacktestFailure(
            (
                _error(
                    f"{operation} returned no result.",
                    field_path=operation,
                    operation=operation,
                    category=ErrorCategory.STORAGE_IO,
                ),
            )
        )
    return value


def _invoke(
    method: Callable[..., object],
    *,
    positional: tuple[object, ...] = (),
    values: Mapping[str, object],
) -> object:
    """Invoke a structural port without duplicate positional/keyword values."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(*positional, **dict(values))

    parameters = tuple(signature.parameters.values())
    explicit = tuple(
        parameter
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    )
    positional_parameters = tuple(
        parameter
        for parameter in explicit
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    bound_names = {
        parameter.name
        for index, parameter in enumerate(positional_parameters)
        if index < len(positional)
    }
    accepted = {
        name: value
        for name, value in values.items()
        if name not in bound_names
        and any(parameter.name == name for parameter in explicit)
    }
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if has_var_kwargs:
        if not positional_parameters:
            # A ``**kwargs``-only structural port cannot receive positional
            # arguments.  Its accepted aliases are already represented in
            # ``values``; use those directly instead of causing a duplicate or
            # invalid positional binding.
            return method(**dict(values))
        accepted.update(
            {
                name: value
                for name, value in values.items()
                if name not in bound_names and name not in accepted
            }
        )
    if positional:
        return method(*positional, **accepted)
    if accepted:
        return method(**accepted)
    return method()


def _run_id(value: object) -> object:
    if isinstance(value, (Ok, Err)):
        value = _unwrap(value, "tracker.allocate_run")
    if isinstance(value, (UUID, str, int)):
        return value
    candidate = _field(value, ("run_id", "id"))
    if candidate is not None:
        return candidate
    raise _BacktestFailure(
        (
            _error(
                "The experiment tracker did not return a Run_ID.",
                field_path="run_id",
                operation="experiment.allocate",
                category=ErrorCategory.EXPERIMENT_RECORDING,
            ),
        )
    )


def _configuration_checksum(config: object) -> str:
    candidate = getattr(config, "configuration_checksum", None) or getattr(
        config, "non_secret_checksum", None
    )
    if isinstance(candidate, str) and len(candidate) == 64:
        return candidate
    try:
        if isinstance(config, ResolvedConfig):
            return sha256_bytes(ConfigurationSerializer().serialize(config))
        dumped = (
            config.model_dump(mode="python")
            if hasattr(config, "model_dump")
            else repr(config)
        )
        return sha256_canonical_json(dumped)
    except Exception:
        return sha256_canonical_json({"config_type": type(config).__name__})


def _range_from_config(config: object) -> DateRange:
    data = getattr(config, "data", config)
    value = getattr(data, "requested_range", getattr(config, "requested_range", None))
    start = _date_value(_field(value, "start"), "evaluation_range.start")
    end = _date_value(_field(value, "end"), "evaluation_range.end")
    if start is None or end is None:
        raise ValueError("a requested date range is required")
    return DateRange(start, end)


def _request_value(
    request: object | None,
    *,
    snapshot_id: str | None,
    evaluation_range: DateRange | None,
) -> BacktestRequest:
    if isinstance(request, BacktestRequest):
        if snapshot_id is not None and snapshot_id != request.snapshot_id:
            raise ValueError("snapshot_id arguments disagree")
        if (
            evaluation_range is not None
            and evaluation_range != request.evaluation_range
        ):
            raise ValueError("evaluation_range arguments disagree")
        return request
    requested_id = snapshot_id
    requested_range = evaluation_range
    if isinstance(request, str):
        requested_id = request if requested_id is None else requested_id
    elif request is not None:
        value = getattr(request, "snapshot_id", None)
        if isinstance(value, str):
            requested_id = value if requested_id is None else requested_id
        candidate = getattr(request, "evaluation_range", None)
        if candidate is None:
            candidate = getattr(request, "requested_range", None)
        if candidate is not None and requested_range is None:
            if isinstance(candidate, DateRange):
                requested_range = candidate
            else:
                start = _date_value(
                    _field(candidate, "start"), "evaluation_range.start"
                )
                end = _date_value(_field(candidate, "end"), "evaluation_range.end")
                if start is not None and end is not None:
                    requested_range = DateRange(start, end)
    if requested_id is None:
        raise ValueError("snapshot_id is required")
    return BacktestRequest(requested_id, requested_range)


def _manifest(snapshot: object) -> object | None:
    candidate = getattr(snapshot, "manifest", None)
    return (
        candidate
        if candidate is not None
        else getattr(snapshot, "snapshot_manifest", None)
    )


def _disclosure(snapshot: object) -> LimitationDisclosure:
    manifest = _manifest(snapshot)
    candidate = getattr(snapshot, "limitation_disclosure", None)
    if candidate is None and manifest is not None:
        candidate = getattr(manifest, "limitation_disclosure", None)
    return (
        candidate
        if isinstance(candidate, LimitationDisclosure)
        else LimitationDisclosure.current()
    )


def _snapshot_range(snapshot: object) -> DateRange | None:
    manifest = _manifest(snapshot)
    identity = getattr(manifest, "content_identity", None)
    candidates = (
        getattr(identity, "covered_range", None),
        getattr(snapshot, "covered_range", None),
        getattr(identity, "requested_range", None),
        getattr(snapshot, "requested_range", None),
    )
    for value in candidates:
        if isinstance(value, DateRange):
            return value
    return None


def _readiness(snapshot: object) -> tuple[bool | None, bool | None]:
    readiness = getattr(snapshot, "readiness", None)
    if readiness is None:
        readiness = getattr(snapshot, "validation", None)
    if readiness is None:
        return None, None
    available = getattr(readiness, "available", None)
    comparison_ready = getattr(readiness, "comparison_ready", None)
    if available is None and hasattr(snapshot, "available"):
        available = getattr(snapshot, "available")
    if comparison_ready is None and hasattr(snapshot, "comparison_ready"):
        comparison_ready = getattr(snapshot, "comparison_ready")
    return (
        bool(available) if available is not None else None,
        bool(comparison_ready) if comparison_ready is not None else None,
    )


class BacktestService:
    """Run one exact verified bundle, audit it, evaluate it, and finalize it."""

    operation_name = "backtest.execute"

    def __init__(
        self,
        tracker: RunTrackerPort | object | None = None,
        snapshot_manager: SnapshotPort | object | None = None,
        bundle_adapter: BundlePort | object | None = None,
        engine: EnginePort | object | None = None,
        evaluator: EvaluationPort | object | None = None,
        *,
        experiment_tracker: object | None = None,
        evaluation: object | None = None,
        manifest_publisher: object | None = None,
        clock: BacktestClock | object | None = None,
        **compatibility: object,
    ) -> None:
        self.tracker = (
            tracker or experiment_tracker or compatibility.pop("tracker_port", None)
        )
        self.snapshot_manager = snapshot_manager or compatibility.pop(
            "snapshot_verifier", None
        )
        self.bundle_adapter = (
            bundle_adapter
            or compatibility.pop("bundle", None)
            or compatibility.pop("bundle_materializer", None)
        )
        self.engine = engine or compatibility.pop("backtest_engine", None)
        self.evaluator = (
            evaluator or evaluation or compatibility.pop("evaluation_service", None)
        )
        self.manifest_publisher = (
            manifest_publisher
            or compatibility.pop("run_manifest_publisher", None)
            or compatibility.pop("manifest_store", None)
        )
        self.clock: BacktestClock = cast(
            BacktestClock,
            clock
            or compatibility.pop("backtest_clock", None)
            or _SystemBacktestClock(),
        )
        if compatibility:
            unknown = ", ".join(sorted(compatibility))
            raise TypeError(f"unsupported BacktestService arguments: {unknown}")
        if (
            self.tracker is None
            or self.snapshot_manager is None
            or self.bundle_adapter is None
            or self.engine is None
        ):
            raise TypeError(
                "tracker, snapshot_manager, bundle_adapter, and engine are required"
            )

    def run(
        self,
        request: BacktestRequest | object | str | ResolvedConfig | None = None,
        config: ResolvedConfig | None = None,
        *,
        snapshot_id: str | None = None,
        evaluation_range: DateRange | None = None,
        progress: Callable[..., object] | None = None,
        progress_callback: Callable[..., object] | None = None,
    ) -> Result[BacktestResult]:
        """Execute the run with tracker allocation as the first durable action."""

        if config is None and isinstance(request, ResolvedConfig):
            config = request
            request = None
        raw_request = request
        try:
            normalized_request = _request_value(
                raw_request,
                snapshot_id=snapshot_id,
                evaluation_range=evaluation_range,
            )
        except (TypeError, ValueError) as failure:
            return Err(
                (self._input_error(str(failure), field_path="request"),),
                preserve_order=True,
            )

        run_identifier: object | None = None
        primary_errors: tuple[ActionableError, ...] = ()
        try:
            # Allocate before config validation, snapshot verification, bundle
            # materialization, or engine execution so every failure remains
            # discoverable in the experiment history.
            run_identifier = self._allocate_run(normalized_request, config)
            if not isinstance(config, ResolvedConfig):
                raise _BacktestFailure(
                    (
                        self._input_error(
                            "a validated ResolvedConfig is required",
                            field_path="config",
                        ),
                    )
                )
            requested_range = normalized_request.evaluation_range or _range_from_config(
                config
            )
            snapshot = self._open_snapshot(normalized_request.snapshot_id)
            self._verify_snapshot(
                snapshot, normalized_request.snapshot_id, requested_range
            )
            locator = self._materialize_bundle(snapshot, normalized_request.snapshot_id)
            engine_request = BacktestRequest(
                normalized_request.snapshot_id, requested_range
            )
            engine_method = self._first_method(
                self.engine, ("run", "execute", "run_backtest")
            )
            engine_value = _invoke(
                engine_method,
                positional=(locator, engine_request, config),
                values={
                    "bundle": locator,
                    "locator": locator,
                    "request": engine_request,
                    "config": config,
                    "resolved_config": config,
                    "progress": progress_callback or progress,
                    "progress_callback": progress_callback or progress,
                },
            )
            output = _unwrap(engine_value, "backtest.engine")
            audit_result = audit_core_output(
                output, config, evaluation_range=requested_range
            )
            if isinstance(audit_result, Err):
                raise _BacktestFailure(audit_result.errors)
            audit = audit_result.value
            evaluation_value = self._evaluate(
                output=output,
                audit=audit,
                snapshot=snapshot,
                locator=locator,
                config=config,
                request=engine_request,
                run_id=run_identifier,
            )
            result = BacktestResult(
                run_id=run_identifier,
                snapshot_id=normalized_request.snapshot_id,
                evaluation_range=requested_range,
                core_output=output,
                audit=audit,
                evaluation=evaluation_value,
                limitation_disclosure=_disclosure(snapshot),
                diagnostics=audit.diagnostics,
            )
            finalization = self._prepare_finalization(result, config)
            self._finalize_success(result, finalization)
            return Ok(result)
        except _BacktestFailure as failure:
            primary_errors = failure.errors
        except (TypeError, ValueError, ArithmeticError, KeyError) as failure:
            primary_errors = (self._input_error(str(failure), field_path="backtest"),)
        except Exception as failure:
            primary_errors = (
                ActionableError.from_unexpected_exception(
                    self.operation_name,
                    failure,
                    correlation_id=str(run_identifier)
                    if run_identifier is not None
                    else None,
                ),
            )

        if run_identifier is not None:
            primary_errors = self._finalize_failure(run_identifier, primary_errors)
        return Err(primary_errors, preserve_order=True)

    execute = run
    run_backtest = run

    def _allocate_run(self, request: BacktestRequest, config: object | None) -> object:
        tracker = self.tracker
        method = self._first_method(
            tracker,
            ("allocate_run", "create_run", "start_run", "begin_run"),
        )
        now = self._now()
        requested_range = request.evaluation_range
        if requested_range is None and config is not None:
            try:
                requested_range = _range_from_config(config)
            except Exception:
                requested_range = None
        universe = (
            getattr(getattr(config, "data", config), "universe", ())
            if config is not None
            else ()
        )
        strategy = (
            getattr(
                getattr(config, "strategy", config), "identifier", "monthly_momentum_v1"
            )
            if config is not None
            else "monthly_momentum_v1"
        )
        values: dict[str, object] = {
            "run_id": uuid4(),
            "request": request,
            "backtest_request": request,
            "snapshot_id": request.snapshot_id,
            "strategy_id": strategy,
            "strategy_identifier": strategy,
            "strategy_parameters": getattr(config, "strategy", None),
            "config": config,
            "resolved_config": config,
            "configuration_checksum": _configuration_checksum(config)
            if config is not None
            else sha256_canonical_json({}),
            "environment_checksum": sha256_canonical_json(
                {
                    "deterministic_seed": getattr(
                        getattr(config, "runtime", None), "deterministic_seed", 0
                    )
                }
            ),
            "universe": tuple(universe),
            "evaluation_range": requested_range,
            "evaluation_start": requested_range.start if requested_range else None,
            "evaluation_end": requested_range.end if requested_range else None,
            "created_at": now,
            "started_at": now,
            "state": "running",
            "deterministic_seed": getattr(
                getattr(config, "runtime", None), "deterministic_seed", 0
            ),
        }
        try:
            allocated = _invoke(method, values=values)
            return _run_id(allocated)
        except _BacktestFailure:
            raise
        except Exception as failure:
            raise _BacktestFailure(
                (
                    ActionableError.from_unexpected_exception(
                        "experiment.allocate",
                        failure,
                    ),
                )
            ) from None

    def _open_snapshot(self, snapshot_id: str) -> object:
        opener = getattr(self.snapshot_manager, "open_verified", None) or getattr(
            self.snapshot_manager, "open_snapshot", None
        )
        if not callable(opener):
            raise _BacktestFailure(
                (
                    self._port_error(
                        "snapshot.open", "snapshot verifier has no open_verified method"
                    ),
                )
            )
        return _unwrap(
            _invoke(
                opener, positional=(snapshot_id,), values={"snapshot_id": snapshot_id}
            ),
            "snapshot.open",
        )

    def _verify_snapshot(
        self, snapshot: object, snapshot_id: str, requested_range: DateRange
    ) -> None:
        if getattr(snapshot, "snapshot_id", None) != snapshot_id:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "snapshot.open",
                        "verified snapshot changed its Snapshot_ID",
                        field_path="snapshot_id",
                    ),
                )
            )
        available, comparison_ready = _readiness(snapshot)
        if available is False:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "snapshot.readiness",
                        "the selected snapshot is unavailable",
                        field_path="snapshot.availability",
                    ),
                )
            )
        if comparison_ready is False:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "snapshot.readiness",
                        "the selected snapshot is not ready for benchmark evaluation",
                        field_path="snapshot.comparison_ready",
                    ),
                )
            )
        covered = _snapshot_range(snapshot)
        if covered is not None and (
            requested_range.start < covered.start or requested_range.end > covered.end
        ):
            raise _BacktestFailure(
                (
                    self._port_error(
                        "snapshot.range",
                        "evaluation range is outside the pinned snapshot range",
                        field_path="evaluation_range",
                    ),
                )
            )
        inspector = getattr(self.snapshot_manager, "inspect_snapshot", None)
        if (
            available is None or comparison_ready is None or covered is None
        ) and callable(inspector):
            inspected = _unwrap(
                _invoke(
                    inspector,
                    positional=(snapshot_id,),
                    values={"snapshot_id": snapshot_id},
                ),
                "snapshot.inspect",
            )
            if getattr(inspected, "snapshot_id", snapshot_id) != snapshot_id:
                raise _BacktestFailure(
                    (
                        self._port_error(
                            "snapshot.inspect",
                            "snapshot inspection changed the pinned Snapshot_ID",
                            field_path="snapshot_id",
                        ),
                    )
                )
            inspected_available, inspected_ready = _readiness(inspected)
            if inspected_available is False or inspected_ready is False:
                raise _BacktestFailure(
                    (
                        self._port_error(
                            "snapshot.readiness",
                            "the selected snapshot is not ready for backtesting",
                            field_path="snapshot.readiness",
                        ),
                    )
                )
            inspected_range = _snapshot_range(inspected)
            if inspected_range is not None and (
                requested_range.start < inspected_range.start
                or requested_range.end > inspected_range.end
            ):
                raise _BacktestFailure(
                    (
                        self._port_error(
                            "snapshot.range",
                            "evaluation range is outside the pinned snapshot range",
                            field_path="evaluation_range",
                        ),
                    )
                )

    def _materialize_bundle(self, snapshot: object, snapshot_id: str) -> object:
        method = self._first_method(
            self.bundle_adapter, ("materialize", "materialize_bundle")
        )
        value = _unwrap(
            _invoke(
                method,
                positional=(snapshot,),
                values={
                    "snapshot": snapshot,
                    "snapshot_handle": snapshot,
                    "snapshot_id": snapshot_id,
                },
            ),
            "zipline.bundle.materialize",
        )
        if getattr(value, "snapshot_id", None) != snapshot_id:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "zipline.bundle.materialize",
                        "the exact bundle is not pinned to the selected Snapshot_ID",
                        field_path="bundle.snapshot_id",
                    ),
                )
            )
        bundle_name = getattr(value, "bundle_name", None)
        if isinstance(bundle_name, str) and bundle_name.strip().lower() == "latest":
            raise _BacktestFailure(
                (
                    self._port_error(
                        "zipline.bundle.materialize",
                        "mutable latest-bundle resolution is not allowed",
                        field_path="bundle.bundle_name",
                    ),
                )
            )
        return value

    def _evaluate(self, **values: object) -> object | None:
        if self.evaluator is None:
            return None
        method = self._first_method(self.evaluator, ("evaluate", "run", "execute"))
        return _unwrap(
            _invoke(
                method,
                positional=(values["output"], values["snapshot"], values["config"]),
                values={
                    **values,
                    "core_output": values["output"],
                    "audited_output": values["output"],
                    "output": values["output"],
                    "backtest_output": values["output"],
                },
            ),
            "evaluation.execute",
        )

    def _prepare_finalization(
        self,
        result: BacktestResult,
        config: ResolvedConfig,
    ) -> BacktestResult | BacktestFinalization:
        publisher = self.manifest_publisher
        if publisher is None:
            return result
        method = self._first_method(
            publisher,
            ("publish", "publish_run_manifest", "assemble_and_publish"),
        )
        try:
            publication = _unwrap(
                _invoke(
                    method,
                    positional=(result, config),
                    values={
                        "result": result,
                        "backtest_result": result,
                        "config": config,
                        "resolved_config": config,
                        "evaluation": result.evaluation,
                    },
                ),
                "experiment.manifest",
            )
        except _BacktestFailure:
            raise
        except Exception as failure:
            raise _BacktestFailure(
                (
                    ActionableError.from_unexpected_exception(
                        "experiment.manifest",
                        failure,
                        correlation_id=str(result.run_id),
                    ),
                )
            ) from None
        checksum = _field(publication, ("manifest_checksum", "checksum"))
        uri = _field(publication, ("manifest_uri", "uri", "relative_uri"))
        manifest = _field(publication, ("manifest", "run_manifest"))
        artifacts = _field(publication, ("artifacts", "artifact_references"), ())
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "experiment.manifest",
                        "manifest publisher returned no valid checksum",
                        field_path="manifest_checksum",
                    ),
                )
            )
        if not isinstance(uri, str) or not uri.strip():
            raise _BacktestFailure(
                (
                    self._port_error(
                        "experiment.manifest",
                        "manifest publisher returned no URI",
                        field_path="manifest_uri",
                    ),
                )
            )
        if manifest is None:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "experiment.manifest",
                        "manifest publisher returned no manifest document",
                        field_path="manifest",
                    ),
                )
            )
        if isinstance(artifacts, Mapping):
            artifacts = tuple(artifacts.values())
        elif isinstance(artifacts, (str, bytes, bytearray)):
            artifacts = ()
        else:
            try:
                artifacts = tuple(cast(Iterable[object], artifacts))
            except TypeError:
                artifacts = ()
        return BacktestFinalization(
            backtest_result=result,
            manifest=manifest,
            manifest_checksum=checksum,
            manifest_uri=uri,
            artifacts=cast(tuple[object, ...], artifacts),
            ended_at=self._now(),
        )

    def _finalize_success(
        self,
        result: BacktestResult,
        finalization: BacktestResult | BacktestFinalization | None = None,
    ) -> None:
        method = self._optional_method(
            self.tracker,
            ("finalize_success", "record_success", "complete_run", "finalize"),
        )
        if method is None:
            method = self._optional_method(self.tracker, ("finalize_run",))
        if method is None:
            raise _BacktestFailure(
                (
                    self._port_error(
                        "experiment.finalize",
                        "experiment tracker has no success finalization method",
                        field_path="run_id",
                    ),
                )
            )
        target = finalization if finalization is not None else result
        values = {
            "run_id": result.run_id,
            "state": "succeeded",
            "status": "succeeded",
            "result": target,
            "backtest_result": result,
            "output": result.core_output,
            "core_output": result.core_output,
            "audited_output": result.core_output,
            "audit": result.audit,
            "evaluation": result.evaluation,
            "evaluation_result": result.evaluation,
            "snapshot_id": result.snapshot_id,
            "diagnostics": result.diagnostics,
            "manifest": _field(target, ("manifest", "run_manifest")),
            "manifest_checksum": _field(target, ("manifest_checksum", "checksum")),
            "manifest_uri": _field(target, ("manifest_uri", "uri", "relative_uri")),
            "artifacts": _field(target, ("artifacts", "artifact_references"), ()),
            "ended_at": _field(target, "ended_at", self._now()),
        }
        try:
            finalized = _invoke(
                method,
                positional=(result.run_id, target),
                values=values,
            )
            if isinstance(finalized, Err):
                raise _BacktestFailure(finalized.errors)

        except _BacktestFailure:
            raise
        except Exception as failure:
            raise _BacktestFailure(
                (
                    ActionableError.from_unexpected_exception(
                        "experiment.finalize",
                        failure,
                        correlation_id=str(result.run_id),
                    ),
                )
            ) from None

    def _finalize_failure(
        self, run_id: object, errors: tuple[ActionableError, ...]
    ) -> tuple[ActionableError, ...]:
        method = self._optional_method(
            self.tracker,
            ("finalize_failure", "record_failure", "fail_run", "fail", "finalize"),
        )
        if method is None:
            return errors
        values = {
            "run_id": run_id,
            "state": "failed",
            "status": "failed",
            "errors": errors,
            "diagnostics": errors,
            "ended_at": self._now(),
        }
        try:
            finalized = _invoke(method, positional=(run_id, errors), values=values)
            if isinstance(finalized, Err):
                return (*errors, *finalized.errors)
        except Exception as failure:
            tracking_error = ActionableError.from_unexpected_exception(
                "experiment.finalize_failure",
                failure,
                correlation_id=str(run_id),
            )
            return (*errors, tracking_error)
        return errors

    def _now(self) -> datetime:
        value = self.clock.utc_now()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() != UTC.utcoffset(value)
        ):
            raise ValueError("backtest clock must return an aware UTC timestamp")
        return value

    @staticmethod
    def _first_method(target: object, names: Sequence[str]) -> Callable[..., object]:
        method = BacktestService._optional_method(target, names)
        if method is None:
            raise _BacktestFailure(
                (
                    BacktestService._port_error(
                        names[0],
                        f"required port method is unavailable: {', '.join(names)}",
                    ),
                )
            )
        return method

    @staticmethod
    def _optional_method(
        target: object, names: Sequence[str]
    ) -> Callable[..., object] | None:
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                return cast(Callable[..., object], method)
        return None

    @staticmethod
    def _port_error(
        operation: str, message: str, *, field_path: str | None = None
    ) -> ActionableError:
        return ActionableError(
            operation=operation,
            category=ErrorCategory.EXPERIMENT_RECORDING
            if operation.startswith("experiment")
            else ErrorCategory.SNAPSHOT_NOT_READY,
            message=message,
            corrective_action="Repair the injected application port and retry the run from the same pinned Snapshot_ID.",
            field_path=field_path,
        )

    @staticmethod
    def _input_error(message: str, *, field_path: str) -> ActionableError:
        return ActionableError(
            operation="backtest.input",
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message=" ".join(message.splitlines()) or "Invalid backtest input.",
            corrective_action="Provide a validated configuration, one Snapshot_ID, and a covered evaluation range.",
            field_path=field_path,
        )


# Common application-service spellings.
Backtest = BacktestService
BacktestOrchestrator = BacktestService


__all__ = [
    "AuditReport",
    "Backtest",
    "BacktestAudit",
    "BacktestClock",
    "BacktestOrchestrator",
    "BacktestRequest",
    "BacktestFinalization",
    "BacktestResult",
    "BacktestService",
    "BacktestOutput",
    "BundlePort",
    "EnginePort",
    "EvaluationPort",
    "RunTrackerPort",
    "SnapshotPort",
    "audit_accounting",
    "audit_backtest_output",
    "audit_core_output",
]
