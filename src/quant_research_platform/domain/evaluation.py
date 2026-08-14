"""Immutable evaluation, experiment-run, and multi-run comparison value objects."""

# ruff: noqa: E501

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, localcontext
from enum import Enum, StrEnum
from typing import Final, TypeVar
from uuid import UUID

from .canonical import canonical_content_identity, content_identity_checksum
from .errors import ActionableError, LimitationDisclosure
from .execution import (
    INITIAL_PORTFOLIO_EQUITY,
    DailyReturn,
    FillRecord,
    OrderRecord,
    OrderStatus,
    PortfolioState,
    ProgressUpdate,
    RunState,
    quantize_money,
)
from .strategy import MomentumStrategyParameters

_CHECKSUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"snap_[0-9a-f]{64}")


class MetricName(StrEnum):
    """Stable names for required strategy, benchmark, and comparison metrics."""

    TOTAL_RETURN = "total_return"
    COMPOUND_ANNUAL_GROWTH_RATE = "compound_annual_growth_rate"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TURNOVER = "turnover"
    TOTAL_COMMISSIONS = "total_commissions"
    TOTAL_SLIPPAGE = "total_slippage"
    UNFILLED_ORDERS = "unfilled_orders"
    ENDING_CASH_BALANCE = "ending_cash_balance"


class MetricScope(StrEnum):
    """The source/signed-difference meaning of one complete metric row set."""

    STRATEGY = "strategy"
    BENCHMARK = "benchmark"
    DIFFERENCE = "difference"


class MetricNullReason(StrEnum):
    """Why a mathematically undefined metric is serialized as JSON null."""

    NO_EVALUATION_SESSIONS = "no_evaluation_sessions"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    ZERO_VOLATILITY = "zero_volatility"
    NON_POSITIVE_EQUITY = "non_positive_equity"


_PERFORMANCE_METRICS: Final[tuple[MetricName, ...]] = (
    MetricName.TOTAL_RETURN,
    MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
    MetricName.ANNUALIZED_VOLATILITY,
    MetricName.SHARPE_RATIO,
    MetricName.MAXIMUM_DRAWDOWN,
)
_STRATEGY_ONLY_METRICS: Final[tuple[MetricName, ...]] = (
    MetricName.TURNOVER,
    MetricName.TOTAL_COMMISSIONS,
    MetricName.TOTAL_SLIPPAGE,
    MetricName.UNFILLED_ORDERS,
    MetricName.ENDING_CASH_BALANCE,
)
METRIC_ORDER_BY_SCOPE: Final[dict[MetricScope, tuple[MetricName, ...]]] = {
    MetricScope.STRATEGY: _PERFORMANCE_METRICS + _STRATEGY_ONLY_METRICS,
    MetricScope.BENCHMARK: _PERFORMANCE_METRICS,
    MetricScope.DIFFERENCE: _PERFORMANCE_METRICS,
}
_MONEY_METRICS: Final[frozenset[MetricName]] = frozenset(
    {
        MetricName.TOTAL_COMMISSIONS,
        MetricName.TOTAL_SLIPPAGE,
        MetricName.ENDING_CASH_BALANCE,
    }
)
_NON_NEGATIVE_METRICS: Final[frozenset[MetricName]] = frozenset(
    {
        MetricName.TURNOVER,
        MetricName.TOTAL_COMMISSIONS,
        MetricName.TOTAL_SLIPPAGE,
        MetricName.ENDING_CASH_BALANCE,
    }
)


def _clean_required_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _require_date(name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a calendar date")
    return value


def _require_aware_timestamp(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_checksum(name: str, value: str) -> str:
    if not isinstance(value, str) or _CHECKSUM_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 checksum")
    return value


def _require_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("deterministic_seed must be an integer")
    if not 0 <= value <= 4_294_967_295:
        raise ValueError("deterministic_seed must be between 0 and 4294967295")
    return value


_T = TypeVar("_T", bound=Enum)


def _coerce_enum(
    enum_type: type[_T], name: str, value: _T | str
) -> _T:
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unsupported {name}: {value!r}") from error


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One typed metric value, or a null with an explicit machine-readable reason."""

    name: MetricName | str
    value: Decimal | int | None
    null_reason: MetricNullReason | str | None = None

    def __post_init__(self) -> None:
        name = _coerce_enum(MetricName, "metric name", self.name)
        if self.value is None:
            if self.null_reason is None:
                raise ValueError("null metric values require null_reason")
            null_reason = _coerce_enum(
                MetricNullReason, "metric null reason", self.null_reason
            )
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "null_reason", null_reason)
            return

        if self.null_reason is not None:
            raise ValueError("non-null metric values must not have null_reason")
        if name is MetricName.UNFILLED_ORDERS:
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise TypeError("unfilled_orders must be an integer")
            if self.value < 0:
                raise ValueError("unfilled_orders must be non-negative")
            normalized_value: Decimal | int = self.value
        else:
            if not isinstance(self.value, Decimal):
                raise TypeError(f"{name.value} must be a Decimal")
            if not self.value.is_finite():
                raise ValueError(f"{name.value} must be finite")
            normalized_value = (
                quantize_money(self.value) if name in _MONEY_METRICS else self.value
            )
            if name in _NON_NEGATIVE_METRICS and normalized_value < 0:
                raise ValueError(f"{name.value} must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", normalized_value)

    def to_serializable(self) -> dict[str, object]:
        """Preserve null reason next to JSON-null metric values."""
        return {
            "name": MetricName(self.name).value,
            "null_reason": (
                MetricNullReason(self.null_reason).value
                if self.null_reason is not None
                else None
            ),
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """One complete, ordered metric collection for a strategy/benchmark/difference."""

    scope: MetricScope | str
    metrics: tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        scope = _coerce_enum(MetricScope, "metric scope", self.scope)
        if not isinstance(self.metrics, tuple):
            raise TypeError("metrics must be an immutable tuple")
        if any(not isinstance(metric, MetricValue) for metric in self.metrics):
            raise TypeError("metrics must contain only MetricValue values")
        expected_names = METRIC_ORDER_BY_SCOPE[scope]
        actual_names = tuple(metric.name for metric in self.metrics)
        if actual_names != expected_names:
            raise ValueError(
                f"{scope.value} metrics must use the required deterministic metric order"
            )
        if scope is not MetricScope.DIFFERENCE:
            volatility = next(
                metric
                for metric in self.metrics
                if metric.name is MetricName.ANNUALIZED_VOLATILITY
            )
            if isinstance(volatility.value, Decimal) and volatility.value < 0:
                raise ValueError("annualized_volatility must be non-negative")
        max_drawdown = next(
            metric
            for metric in self.metrics
            if metric.name is MetricName.MAXIMUM_DRAWDOWN
        )
        if (
            scope is not MetricScope.DIFFERENCE
            and isinstance(max_drawdown.value, Decimal)
            and max_drawdown.value > 0
        ):
            raise ValueError("strategy and benchmark maximum_drawdown must be non-positive")
        object.__setattr__(self, "scope", scope)

    def metric(self, name: MetricName | str) -> MetricValue:
        """Return one named metric after validating the stable metric name."""
        expected = _coerce_enum(MetricName, "metric name", name)
        return next(metric for metric in self.metrics if metric.name is expected)

    def to_serializable(self) -> dict[str, object]:
        return {
            "metrics": [metric.to_serializable() for metric in self.metrics],
            "scope": MetricScope(self.scope).value,
        }


SESSIONS_PER_YEAR: Final[int] = 252


@dataclass(frozen=True, slots=True)
class MonthlyReturn:
    """One month-end compounded return in canonical session order."""

    month: date
    return_value: Decimal

    def __post_init__(self) -> None:
        month = _require_date("month", self.month)
        return_value = _as_decimal("return_value", self.return_value)
        if return_value <= Decimal("-1"):
            raise ValueError("return_value must be greater than -1")
        object.__setattr__(self, "month", month)
        object.__setattr__(self, "return_value", return_value)

    def to_serializable(self) -> dict[str, object]:
        return {"month": self.month, "return_value": self.return_value}


def _as_decimal(name: str, value: object) -> Decimal:
    """Coerce exact integer inputs while rejecting lossy floating-point inputs."""
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise TypeError(f"{name} must be a Decimal or integer")
    decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def _equity_values(values: Iterable[Decimal | int | PortfolioState]) -> tuple[Decimal, ...]:
    normalized: list[Decimal] = []
    for value in values:
        if isinstance(value, PortfolioState):
            normalized.append(value.portfolio_equity)
        else:
            normalized.append(_as_decimal("equity", value))
    return tuple(normalized)


def _return_values(values: Iterable[Decimal | int | DailyReturn]) -> tuple[Decimal, ...]:
    normalized: list[Decimal] = []
    for value in values:
        if isinstance(value, DailyReturn):
            normalized.append(value.return_value)
        else:
            return_value = _as_decimal("return", value)
            if return_value <= Decimal("-1"):
                raise ValueError("return values must be greater than -1")
            normalized.append(return_value)
    return tuple(normalized)


def _null_metric(name: MetricName, reason: MetricNullReason) -> MetricValue:
    return MetricValue(name=name, value=None, null_reason=reason)


def _positive_equity(values: tuple[Decimal, ...]) -> bool:
    return all(value > 0 for value in values)


def _decimal_sqrt(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 40
        return value.sqrt()


def _dated_returns(
    values: Mapping[date, Decimal | int] | Iterable[DailyReturn | tuple[date, Decimal | int]],
) -> tuple[tuple[date, Decimal], ...]:
    if isinstance(values, Mapping):
        candidates = values.items()
    else:
        candidates = values

    dated: list[tuple[date, Decimal]] = []
    for item in candidates:
        if isinstance(item, DailyReturn):
            session = item.session
            return_value = item.return_value
        else:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("dated returns must contain (session, return) pairs")
            session, raw_return = item
            if not isinstance(session, date) or isinstance(session, datetime):
                raise TypeError("return session must be a calendar date")
            return_value = _as_decimal("return", raw_return)
            if return_value <= Decimal("-1"):
                raise ValueError("return values must be greater than -1")
        dated.append((session, return_value))

    dated.sort(key=lambda item: item[0])
    sessions = tuple(session for session, _ in dated)
    if len(sessions) != len(set(sessions)):
        raise ValueError("dated returns must not contain duplicate sessions")
    return tuple(dated)


def calculate_total_return(
    equity: Iterable[Decimal | int | PortfolioState],
) -> MetricValue:
    """Calculate ending equity divided by starting equity minus one."""
    values = _equity_values(equity)
    if not values:
        return _null_metric(MetricName.TOTAL_RETURN, MetricNullReason.NO_EVALUATION_SESSIONS)
    if not _positive_equity(values):
        return _null_metric(MetricName.TOTAL_RETURN, MetricNullReason.NON_POSITIVE_EQUITY)
    if len(values) == 1:
        result = Decimal("0")
    else:
        result = values[-1] / values[0] - Decimal("1")
    return MetricValue(name=MetricName.TOTAL_RETURN, value=result)


def calculate_total_return_from_returns(
    returns: Iterable[Decimal | int | DailyReturn],
) -> MetricValue:
    """Calculate total return as ``product(1 + r_t) - 1``.

    ``calculate_total_return`` accepts an equity curve because that is the most
    useful representation when no return artifact is available.  Evaluation
    artifacts normally have both curves, however, and this explicit function
    keeps the declared return-series formula available without inferring whether
    arbitrary Decimal values represent prices/equity or returns.
    """
    values = _return_values(returns)
    if not values:
        return _null_metric(
            MetricName.TOTAL_RETURN,
            MetricNullReason.NO_EVALUATION_SESSIONS,
        )
    compounded = Decimal("1")
    for value in values:
        compounded *= Decimal("1") + value
    return MetricValue(name=MetricName.TOTAL_RETURN, value=compounded - Decimal("1"))


def calculate_cagr(
    equity: Iterable[Decimal | int | PortfolioState],
    *,
    return_observations: int | None = None,
) -> MetricValue:
    """Calculate compound annual growth using 252 sessions per year.

    By default, an equity curve contributes one return observation for every
    adjacent pair.  Callers that provide a separately stored return artifact may
    pass its observation count explicitly; this preserves the definition
    ``(E_end / E_start) ** (252 / N) - 1`` even when the two artifacts use
    different boundary conventions.
    """
    values = _equity_values(equity)
    if not values:
        return _null_metric(
            MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
            MetricNullReason.NO_EVALUATION_SESSIONS,
        )
    if len(values) < 2:
        return _null_metric(
            MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
            MetricNullReason.INSUFFICIENT_OBSERVATIONS,
        )
    if not _positive_equity(values):
        return _null_metric(
            MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
            MetricNullReason.NON_POSITIVE_EQUITY,
        )
    if return_observations is None:
        observation_count = len(values) - 1
    else:
        if isinstance(return_observations, bool) or not isinstance(return_observations, int):
            raise TypeError("return_observations must be an integer")
        if return_observations <= 0:
            return _null_metric(
                MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
                MetricNullReason.NO_EVALUATION_SESSIONS,
            )
        observation_count = return_observations
    periods = Decimal(observation_count)
    with localcontext() as context:
        context.prec = 40
        result = (
            values[-1] / values[0]
        ) ** (Decimal(SESSIONS_PER_YEAR) / periods) - Decimal("1")
    return MetricValue(name=MetricName.COMPOUND_ANNUAL_GROWTH_RATE, value=result)


def calculate_annualized_volatility(
    returns: Iterable[Decimal | int | DailyReturn],
) -> MetricValue:
    """Calculate sample daily volatility annualized by square-root 252."""
    values = _return_values(returns)
    if not values:
        return _null_metric(
            MetricName.ANNUALIZED_VOLATILITY,
            MetricNullReason.NO_EVALUATION_SESSIONS,
        )
    if len(values) < 2:
        return _null_metric(
            MetricName.ANNUALIZED_VOLATILITY,
            MetricNullReason.INSUFFICIENT_OBSERVATIONS,
        )
    mean = sum(values, Decimal("0")) / Decimal(len(values))
    sample_variance = sum(
        ((value - mean) ** 2 for value in values), Decimal("0")
    ) / Decimal(len(values) - 1)
    result = _decimal_sqrt(sample_variance * Decimal(SESSIONS_PER_YEAR))
    return MetricValue(name=MetricName.ANNUALIZED_VOLATILITY, value=result)


def calculate_sharpe_ratio(
    returns: Iterable[Decimal | int | DailyReturn],
) -> MetricValue:
    """Calculate the zero-risk-free-rate sample Sharpe ratio."""
    values = _return_values(returns)
    if not values:
        return _null_metric(MetricName.SHARPE_RATIO, MetricNullReason.NO_EVALUATION_SESSIONS)
    if len(values) < 2:
        return _null_metric(
            MetricName.SHARPE_RATIO,
            MetricNullReason.INSUFFICIENT_OBSERVATIONS,
        )
    mean = sum(values, Decimal("0")) / Decimal(len(values))
    sample_variance = sum(
        ((value - mean) ** 2 for value in values), Decimal("0")
    ) / Decimal(len(values) - 1)
    standard_deviation = _decimal_sqrt(sample_variance)
    if standard_deviation == 0:
        return _null_metric(MetricName.SHARPE_RATIO, MetricNullReason.ZERO_VOLATILITY)
    result = mean / standard_deviation * _decimal_sqrt(Decimal(SESSIONS_PER_YEAR))
    return MetricValue(name=MetricName.SHARPE_RATIO, value=result)


def calculate_maximum_drawdown(
    equity: Iterable[Decimal | int | PortfolioState],
) -> MetricValue:
    """Calculate the signed worst peak-to-trough equity drawdown."""
    values = _equity_values(equity)
    if not values:
        return _null_metric(
            MetricName.MAXIMUM_DRAWDOWN,
            MetricNullReason.NO_EVALUATION_SESSIONS,
        )
    if not _positive_equity(values):
        return _null_metric(
            MetricName.MAXIMUM_DRAWDOWN,
            MetricNullReason.NON_POSITIVE_EQUITY,
        )
    peak = values[0]
    maximum = Decimal("0")
    for value in values:
        if value > peak:
            peak = value
        drawdown = value / peak - Decimal("1")
        if drawdown < maximum:
            maximum = drawdown
    return MetricValue(name=MetricName.MAXIMUM_DRAWDOWN, value=maximum)


def _fill_values(fills: Iterable[FillRecord]) -> tuple[FillRecord, ...]:
    normalized = tuple(fills)
    if any(not isinstance(fill, FillRecord) for fill in normalized):
        raise TypeError("fills must contain only FillRecord values")
    return normalized


def total_commissions(fills: Iterable[FillRecord]) -> Decimal:
    """Return the quantized sum of commission charged on every fill."""
    total = sum((fill.commission for fill in _fill_values(fills)), Decimal("0"))
    return quantize_money(total)


def total_slippage(fills: Iterable[FillRecord]) -> Decimal:
    """Return the quantized sum of adverse slippage cost on every fill."""
    total = sum((fill.slippage_cost for fill in _fill_values(fills)), Decimal("0"))
    return quantize_money(total)


def _turnover_denominator(
    portfolio_equity: Decimal | int | PortfolioState | Iterable[Decimal | int | PortfolioState] | None,
    initial_equity: Decimal | int,
) -> Decimal:
    if portfolio_equity is None:
        return _as_decimal("initial_equity", initial_equity)
    if isinstance(portfolio_equity, PortfolioState):
        return portfolio_equity.portfolio_equity
    if isinstance(portfolio_equity, (Decimal, int)) and not isinstance(portfolio_equity, bool):
        return _as_decimal("portfolio_equity", portfolio_equity)
    values = _equity_values(portfolio_equity)  # type: ignore[arg-type]
    if not values:
        return Decimal("0")
    return sum(values, Decimal("0")) / Decimal(len(values))


def calculate_turnover(
    fills: Iterable[FillRecord],
    portfolio_equity: Decimal | int | PortfolioState | Iterable[Decimal | int | PortfolioState] | None = None,
    *,
    initial_equity: Decimal | int = INITIAL_PORTFOLIO_EQUITY,
) -> MetricValue:
    """Calculate traded notional divided by initial or average portfolio equity.

    Passing portfolio states (or an equity sequence) uses their arithmetic mean;
    omitting it uses the fixed initial USD 100,000 equity.
    """
    denominator = _turnover_denominator(portfolio_equity, initial_equity)
    if denominator <= 0:
        return _null_metric(MetricName.TURNOVER, MetricNullReason.NON_POSITIVE_EQUITY)
    traded_notional = sum(
        (fill.gross_notional for fill in _fill_values(fills)), Decimal("0")
    )
    return MetricValue(
        name=MetricName.TURNOVER,
        value=traded_notional / denominator,
    )


def calculate_total_commissions(fills: Iterable[FillRecord]) -> MetricValue:
    """Return total commissions as the strategy cost metric."""
    return MetricValue(
        name=MetricName.TOTAL_COMMISSIONS,
        value=total_commissions(fills),
    )


def calculate_total_slippage(fills: Iterable[FillRecord]) -> MetricValue:
    """Return total adverse slippage as the strategy cost metric."""
    return MetricValue(
        name=MetricName.TOTAL_SLIPPAGE,
        value=total_slippage(fills),
    )


def calculate_monthly_compounding(
    returns: Mapping[date, Decimal | int]
    | Iterable[DailyReturn | tuple[date, Decimal | int]],
) -> tuple[MonthlyReturn, ...]:
    """Compound daily returns by calendar month and label rows by month-end session."""
    dated = _dated_returns(returns)
    if not dated:
        return ()

    monthly: list[MonthlyReturn] = []
    current_year_month: tuple[int, int] | None = None
    current_return = Decimal("1")
    month_end: date | None = None
    for session, return_value in dated:
        year_month = (session.year, session.month)
        if current_year_month is not None and year_month != current_year_month:
            assert month_end is not None
            monthly.append(
                MonthlyReturn(month=month_end, return_value=current_return - Decimal("1"))
            )
            current_return = Decimal("1")
        current_year_month = year_month
        current_return *= Decimal("1") + return_value
        month_end = session
    assert month_end is not None
    monthly.append(
        MonthlyReturn(month=month_end, return_value=current_return - Decimal("1"))
    )
    return tuple(monthly)


def _difference_metric(
    name: MetricName,
    strategy: MetricValue,
    benchmark: MetricValue,
) -> MetricValue:
    if strategy.value is None:
        return _null_metric(name, MetricNullReason(strategy.null_reason))
    if benchmark.value is None:
        return _null_metric(name, MetricNullReason(benchmark.null_reason))
    if not isinstance(strategy.value, Decimal) or not isinstance(benchmark.value, Decimal):
        raise TypeError("performance metric values must be Decimal values")
    return MetricValue(name=name, value=strategy.value - benchmark.value)


def strategy_minus_benchmark(
    strategy: EvaluationMetrics,
    benchmark: EvaluationMetrics,
) -> EvaluationMetrics:
    """Return signed strategy-minus-benchmark performance differences."""
    if strategy.scope is not MetricScope.STRATEGY:
        raise ValueError("strategy metrics must have strategy scope")
    if benchmark.scope is not MetricScope.BENCHMARK:
        raise ValueError("benchmark metrics must have benchmark scope")
    metrics = tuple(
        _difference_metric(
            name,
            strategy.metric(name),
            benchmark.metric(name),
        )
        for name in _PERFORMANCE_METRICS
    )
    return EvaluationMetrics(scope=MetricScope.DIFFERENCE, metrics=metrics)


def _returns_from_equity(values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    if len(values) < 2 or not _positive_equity(values):
        return ()
    return tuple(
        current / previous - Decimal("1")
        for previous, current in zip(values, values[1:])
    )


def calculate_evaluation_metrics(
    scope: MetricScope | str,
    equity: Iterable[Decimal | int | PortfolioState],
    *,
    returns: Iterable[Decimal | int | DailyReturn] | None = None,
    fills: Iterable[FillRecord] = (),
    orders: Iterable[OrderRecord] = (),
    portfolio_equity: Decimal | int | PortfolioState | Iterable[Decimal | int | PortfolioState] | None = None,
) -> EvaluationMetrics:
    """Build the deterministic metric collection consumed by evaluation artifacts."""
    metric_scope = _coerce_enum(MetricScope, "metric scope", scope)
    equity_items = tuple(equity)
    equity_values = _equity_values(equity_items)
    if returns is None:
        return_values = _returns_from_equity(equity_values)
        total_return_metric = calculate_total_return(equity_values)
        cagr_metric = calculate_cagr(equity_values)
    else:
        return_values = _return_values(returns)
        total_return_metric = calculate_total_return_from_returns(return_values)
        cagr_metric = calculate_cagr(
            equity_values,
            return_observations=len(return_values),
        )
    performance = (
        total_return_metric,
        cagr_metric,
        calculate_annualized_volatility(return_values),
        calculate_sharpe_ratio(return_values),
        calculate_maximum_drawdown(equity_values),
    )
    if metric_scope is not MetricScope.STRATEGY:
        if metric_scope is MetricScope.BENCHMARK:
            return EvaluationMetrics(scope=metric_scope, metrics=performance)
        raise ValueError("difference metrics must be built with strategy_minus_benchmark")

    normalized_fills = _fill_values(fills)
    normalized_orders = tuple(orders)
    if any(not isinstance(order, OrderRecord) for order in normalized_orders):
        raise TypeError("orders must contain only OrderRecord values")
    ending_cash = (
        MetricValue(
            name=MetricName.ENDING_CASH_BALANCE,
            value=equity_items[-1].cash_balance,
        )
        if equity_items and isinstance(equity_items[-1], PortfolioState)
        else _null_metric(
            MetricName.ENDING_CASH_BALANCE,
            MetricNullReason.NO_EVALUATION_SESSIONS,
        )
    )
    strategy_only = (
        calculate_turnover(normalized_fills, portfolio_equity),
        calculate_total_commissions(normalized_fills),
        calculate_total_slippage(normalized_fills),
        MetricValue(
            name=MetricName.UNFILLED_ORDERS,
            value=sum(
                order.status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.UNFILLED}
                for order in normalized_orders
            ),
        ),
        ending_cash,
    )
    return EvaluationMetrics(
        scope=metric_scope,
        metrics=performance + strategy_only,
    )


# Short aliases keep the formula names convenient at the domain boundary while
# the calculate_* names remain explicit in application code.
total_return = calculate_total_return
total_return_from_returns = calculate_total_return_from_returns
compound_annual_growth_rate = calculate_cagr
cagr = calculate_cagr
annualized_volatility = calculate_annualized_volatility
sample_annualized_volatility = calculate_annualized_volatility
sharpe_ratio = calculate_sharpe_ratio
zero_rate_sharpe = calculate_sharpe_ratio
maximum_drawdown = calculate_maximum_drawdown
max_drawdown = calculate_maximum_drawdown
turnover = calculate_turnover
commissions = total_commissions
slippage = total_slippage
monthly_compounding = calculate_monthly_compounding
monthly_returns = calculate_monthly_compounding
metric_differences = strategy_minus_benchmark
strategy_benchmark_differences = strategy_minus_benchmark
calculate_strategy_minus_benchmark = strategy_minus_benchmark
build_evaluation_metrics = calculate_evaluation_metrics


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Comparable strategy/benchmark metric tables plus signed differences."""

    strategy_metrics: EvaluationMetrics
    benchmark_metrics: EvaluationMetrics
    differences: EvaluationMetrics

    def __post_init__(self) -> None:
        if self.strategy_metrics.scope is not MetricScope.STRATEGY:
            raise ValueError("strategy_metrics must have strategy scope")
        if self.benchmark_metrics.scope is not MetricScope.BENCHMARK:
            raise ValueError("benchmark_metrics must have benchmark scope")
        if self.differences.scope is not MetricScope.DIFFERENCE:
            raise ValueError("differences must have difference scope")

    def to_serializable(self) -> dict[str, object]:
        return {
            "benchmark_metrics": self.benchmark_metrics.to_serializable(),
            "differences": self.differences.to_serializable(),
            "strategy_metrics": self.strategy_metrics.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class DependencyVersion:
    """One sorted installed dependency record in an environment fingerprint."""

    name: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_required_text("dependency name", self.name).lower())
        object.__setattr__(self, "version", _clean_required_text("dependency version", self.version))

    def to_serializable(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Stable environment/source inputs recorded for rerun reproducibility."""

    python_version: str
    operating_system: str
    architecture: str
    dependencies: tuple[DependencyVersion, ...]
    source_revision: str
    source_dirty: bool
    deterministic_seed: int
    effective_source_checksum: str

    def __post_init__(self) -> None:
        for field_name in (
            "python_version",
            "operating_system",
            "architecture",
            "source_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_required_text(field_name, getattr(self, field_name)),
            )
        if not isinstance(self.source_dirty, bool):
            raise TypeError("source_dirty must be a bool")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be an immutable tuple")
        if any(not isinstance(dependency, DependencyVersion) for dependency in self.dependencies):
            raise TypeError("dependencies must contain only DependencyVersion values")
        dependency_names = tuple(dependency.name for dependency in self.dependencies)
        if dependency_names != tuple(sorted(dependency_names)):
            raise ValueError("dependencies must be sorted by normalized name")
        if len(dependency_names) != len(set(dependency_names)):
            raise ValueError("dependencies must have unique normalized names")
        object.__setattr__(self, "deterministic_seed", _require_seed(self.deterministic_seed))
        object.__setattr__(
            self,
            "effective_source_checksum",
            _require_checksum("effective_source_checksum", self.effective_source_checksum),
        )

    def to_serializable(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "dependencies": [dependency.to_serializable() for dependency in self.dependencies],
            "deterministic_seed": self.deterministic_seed,
            "effective_source_checksum": self.effective_source_checksum,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
            "source_dirty": self.source_dirty,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True, slots=True)
class ScientificArtifactReference:
    """A checksummed scientific artifact included in run content identity."""

    role: str
    checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _clean_required_text("artifact role", self.role))
        object.__setattr__(self, "checksum", _require_checksum("artifact checksum", self.checksum))

    def to_serializable(self) -> dict[str, str]:
        return {"checksum": self.checksum, "role": self.role}


@dataclass(frozen=True, slots=True)
class RunContentIdentity:
    """The deterministic scientific projection of a run manifest."""

    schema_version: str
    snapshot_id: str
    strategy_identifier: str
    strategy_parameters: MomentumStrategyParameters
    evaluation_start: date
    evaluation_end: date
    configuration_checksum: str
    environment_fingerprint: EnvironmentFingerprint
    scientific_artifacts: tuple[ScientificArtifactReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _clean_required_text("schema_version", self.schema_version))
        if not isinstance(self.snapshot_id, str) or _SNAPSHOT_ID_PATTERN.fullmatch(self.snapshot_id) is None:
            raise ValueError("snapshot_id must be a scientific snapshot ID")
        object.__setattr__(
            self,
            "strategy_identifier",
            _clean_required_text("strategy_identifier", self.strategy_identifier),
        )
        if not isinstance(self.strategy_parameters, MomentumStrategyParameters):
            raise TypeError("strategy_parameters must be MomentumStrategyParameters")
        start = _require_date("evaluation_start", self.evaluation_start)
        end = _require_date("evaluation_end", self.evaluation_end)
        if start > end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        object.__setattr__(self, "evaluation_start", start)
        object.__setattr__(self, "evaluation_end", end)
        object.__setattr__(
            self,
            "configuration_checksum",
            _require_checksum("configuration_checksum", self.configuration_checksum),
        )
        if not isinstance(self.environment_fingerprint, EnvironmentFingerprint):
            raise TypeError("environment_fingerprint must be EnvironmentFingerprint")
        if not isinstance(self.scientific_artifacts, tuple):
            raise TypeError("scientific_artifacts must be an immutable tuple")
        if any(
            not isinstance(artifact, ScientificArtifactReference)
            for artifact in self.scientific_artifacts
        ):
            raise TypeError(
                "scientific_artifacts must contain ScientificArtifactReference values"
            )
        artifact_keys = tuple(
            (artifact.role, artifact.checksum) for artifact in self.scientific_artifacts
        )
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError("scientific_artifacts must be sorted by role and checksum")
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("scientific_artifacts must not contain duplicate references")

    def to_serializable(self) -> dict[str, object]:
        return {
            "configuration_checksum": self.configuration_checksum,
            "environment_fingerprint": self.environment_fingerprint.to_serializable(),
            "evaluation_end": self.evaluation_end,
            "evaluation_start": self.evaluation_start,
            "schema_version": self.schema_version,
            "scientific_artifacts": [
                artifact.to_serializable() for artifact in self.scientific_artifacts
            ],
            "snapshot_id": self.snapshot_id,
            "strategy_identifier": self.strategy_identifier,
            "strategy_parameters": self.strategy_parameters.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class RunOperationalMetadata:
    """Mutable-history facts intentionally excluded from scientific run identity."""

    run_id: UUID
    state: RunState | str
    created_at: datetime
    started_at: datetime
    ended_at: datetime | None = None
    mlflow_run_id: str | None = None
    progress_updates: tuple[ProgressUpdate, ...] = ()
    errors: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be a UUID operational identifier")
        state = _coerce_enum(RunState, "run state", self.state)
        created_at = _require_aware_timestamp("created_at", self.created_at)
        started_at = _require_aware_timestamp("started_at", self.started_at)
        if started_at < created_at:
            raise ValueError("started_at must not precede created_at")
        ended_at = self.ended_at
        if ended_at is not None:
            ended_at = _require_aware_timestamp("ended_at", ended_at)
            if ended_at < started_at:
                raise ValueError("ended_at must not precede started_at")
        if state is RunState.RUNNING and ended_at is not None:
            raise ValueError("running runs must not have ended_at")
        if state is not RunState.RUNNING and ended_at is None:
            raise ValueError("terminal runs require ended_at")
        mlflow_run_id = self.mlflow_run_id
        if mlflow_run_id is not None:
            mlflow_run_id = _clean_required_text("mlflow_run_id", mlflow_run_id)
        if not isinstance(self.progress_updates, tuple):
            raise TypeError("progress_updates must be an immutable tuple")
        if any(not isinstance(update, ProgressUpdate) for update in self.progress_updates):
            raise TypeError("progress_updates must contain only ProgressUpdate values")
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be an immutable tuple")
        if any(not isinstance(error, ActionableError) for error in self.errors):
            raise TypeError("errors must contain only ActionableError values")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "mlflow_run_id", mlflow_run_id)
        object.__setattr__(
            self,
            "errors",
            tuple(sorted(self.errors, key=ActionableError.sort_key)),
        )

    def to_serializable(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "errors": [error.format_for_display() for error in self.errors],
            "mlflow_run_id": self.mlflow_run_id,
            "progress_updates": [
                update.to_serializable() for update in self.progress_updates
            ],
            "run_id": str(self.run_id),
            "started_at": self.started_at,
            "state": RunState(self.state).value,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """An immutable run document with explicit scientific/operational separation."""

    content_identity: RunContentIdentity
    operational_metadata: RunOperationalMetadata
    limitation_disclosure: LimitationDisclosure
    evaluation_result: EvaluationResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_identity, RunContentIdentity):
            raise TypeError("content_identity must be RunContentIdentity")
        if not isinstance(self.operational_metadata, RunOperationalMetadata):
            raise TypeError("operational_metadata must be RunOperationalMetadata")
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be LimitationDisclosure")
        state = self.operational_metadata.state
        if state is RunState.SUCCEEDED and self.evaluation_result is None:
            raise ValueError("succeeded runs require an evaluation_result")
        if state is RunState.RUNNING and self.evaluation_result is not None:
            raise ValueError("running runs must not have an evaluation_result")
        if self.evaluation_result is not None and not isinstance(
            self.evaluation_result, EvaluationResult
        ):
            raise TypeError("evaluation_result must be EvaluationResult or None")

    @property
    def scientific_checksum(self) -> str:
        """Checksum only the deterministic content-identity projection."""
        return content_identity_checksum({"content_identity": self.content_identity.to_serializable()})

    def canonical_scientific_bytes(self) -> bytes:
        """Return canonical bytes only for reproducible run science."""
        return canonical_content_identity(
            {"content_identity": self.content_identity.to_serializable()}
        )

    def to_serializable(self) -> dict[str, object]:
        return {
            "content_identity": self.content_identity.to_serializable(),
            "evaluation_result": (
                self.evaluation_result.to_serializable()
                if self.evaluation_result is not None
                else None
            ),
            "limitation_disclosure": {
                "lines": list(self.limitation_disclosure.lines()),
                "version": self.limitation_disclosure.version,
            },
            "operational_metadata": self.operational_metadata.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One successful run's immutable inputs to an ordered multi-run comparison."""

    run_id: UUID
    run_manifest_checksum: str
    snapshot_id: str
    evaluation_start: date
    evaluation_end: date
    strategy_metrics: EvaluationMetrics
    benchmark_metrics: EvaluationMetrics
    configuration_checksum: str
    environment_fingerprint: EnvironmentFingerprint

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be a UUID operational identifier")
        object.__setattr__(
            self,
            "run_manifest_checksum",
            _require_checksum("run_manifest_checksum", self.run_manifest_checksum),
        )
        if not isinstance(self.snapshot_id, str) or _SNAPSHOT_ID_PATTERN.fullmatch(self.snapshot_id) is None:
            raise ValueError("snapshot_id must be a scientific snapshot ID")
        start = _require_date("evaluation_start", self.evaluation_start)
        end = _require_date("evaluation_end", self.evaluation_end)
        if start > end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        object.__setattr__(self, "evaluation_start", start)
        object.__setattr__(self, "evaluation_end", end)
        if self.strategy_metrics.scope is not MetricScope.STRATEGY:
            raise ValueError("strategy_metrics must have strategy scope")
        if self.benchmark_metrics.scope is not MetricScope.BENCHMARK:
            raise ValueError("benchmark_metrics must have benchmark scope")
        object.__setattr__(
            self,
            "configuration_checksum",
            _require_checksum("configuration_checksum", self.configuration_checksum),
        )
        if not isinstance(self.environment_fingerprint, EnvironmentFingerprint):
            raise TypeError("environment_fingerprint must be EnvironmentFingerprint")

    def to_serializable(self) -> dict[str, object]:
        return {
            "benchmark_metrics": self.benchmark_metrics.to_serializable(),
            "configuration_checksum": self.configuration_checksum,
            "environment_fingerprint": self.environment_fingerprint.to_serializable(),
            "evaluation_end": self.evaluation_end,
            "evaluation_start": self.evaluation_start,
            "run_id": str(self.run_id),
            "run_manifest_checksum": self.run_manifest_checksum,
            "snapshot_id": self.snapshot_id,
            "strategy_metrics": self.strategy_metrics.to_serializable(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonSet:
    """An ordered, duplicate-free selection of two through ten successful run records."""

    records: tuple[ComparisonRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise TypeError("records must be an immutable tuple")
        if any(not isinstance(record, ComparisonRecord) for record in self.records):
            raise TypeError("records must contain only ComparisonRecord values")
        if not 2 <= len(self.records) <= 10:
            raise ValueError("comparison records must contain from 2 through 10 runs")
        run_ids = tuple(record.run_id for record in self.records)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("comparison records must have unique run_id values")

    def to_serializable(self) -> dict[str, object]:
        return {"records": [record.to_serializable() for record in self.records]}


__all__ = [
    "ComparisonRecord",
    "ComparisonSet",
    "DependencyVersion",
    "EnvironmentFingerprint",
    "EvaluationMetrics",
    "EvaluationResult",
    "METRIC_ORDER_BY_SCOPE",
    "MetricName",
    "MetricNullReason",
    "MetricScope",
    "MetricValue",
    "RunContentIdentity",
    "RunManifest",
    "RunOperationalMetadata",
    "ScientificArtifactReference",
    "annualized_volatility",
    "build_evaluation_metrics",
    "calculate_annualized_volatility",
    "calculate_cagr",
    "calculate_evaluation_metrics",
    "calculate_maximum_drawdown",
    "calculate_monthly_compounding",
    "calculate_sharpe_ratio",
    "calculate_strategy_minus_benchmark",
    "calculate_total_commissions",
    "calculate_total_return",
    "calculate_total_return_from_returns",
    "calculate_total_slippage",
    "calculate_turnover",
    "cagr",
    "commissions",
    "compound_annual_growth_rate",
    "max_drawdown",
    "maximum_drawdown",
    "metric_differences",
    "monthly_compounding",
    "monthly_returns",
    "sample_annualized_volatility",
    "sharpe_ratio",
    "slippage",
    "strategy_benchmark_differences",
    "total_commissions",
    "total_return",
    "total_return_from_returns",
    "total_slippage",
    "turnover",
    "zero_rate_sharpe",
]
