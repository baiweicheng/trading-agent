"""Deterministic SPY-aligned evaluation and canonical result artifacts.

The evaluator is deliberately an application-layer coordinator.  It does not
know about a particular Parquet, CAS, or database implementation; those are
injected through small structural seams.  The scientific result is assembled
from immutable domain records, while operational publication is optional and
performed only after every canonical payload has been checksummed.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Protocol, cast
from uuid import uuid4

from ..domain.canonical import canonical_json, sha256_bytes
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.evaluation import (
    EvaluationMetrics,
    EvaluationResult,
    MetricName,
    MetricScope,
    MonthlyReturn,
    calculate_evaluation_metrics,
    calculate_monthly_compounding,
    strategy_minus_benchmark,
)
from ..domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    DailyReturn,
    FillRecord,
    OrderRecord,
    OrderStatus,
    PortfolioState,
)
from ..domain.market import DateRange, normalize_symbol


class SnapshotScanPort(Protocol):
    """Minimal projected scan boundary used for normalized snapshot bars."""

    def scan(
        self, refs: Sequence[object], columns: Sequence[str], **kwargs: object
    ) -> object:
        """Return an iterable record-batch reader for the requested projection."""


class ArtifactPublicationPort(Protocol):
    """Optional artifact publication seam.

    Concrete local stores may expose ``publish_artifact`` (a staged-file API),
    ``put``/``store`` (a bytes API), or ``write``.  The evaluator adapts to any
    of those narrow forms without importing an infrastructure implementation.
    """


_JSON_MEDIA_TYPE = "application/json"
_CANONICAL_TABLE_MEDIA_TYPE = "application/vnd.quant-research.canonical+json"
_CHART_MEDIA_TYPE = "application/vnd.vega-lite+json"
_SCHEMA_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "returns": "returns_v1",
        "equity": "equity_curve_v1",
        "drawdown": "drawdown_v1",
        "monthly_returns": "monthly_return_v1",
        "positions": "positions_v1",
        "orders": "orders_v1",
        "fills": "fills_v1",
        "decisions": "decisions_v1",
        "metrics": "metrics_v1",
        "portfolio": "portfolio_v1",
        "chart": "vega_lite_v1",
        "transactions": "transactions_v1",
    }
)


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


def _unwrap(value: object, operation: str) -> object:
    if isinstance(value, Err):
        raise _EvaluationFailure(value.errors)
    if isinstance(value, Ok):
        return value.value
    if value is None:
        raise _EvaluationFailure(
            (
                _error(
                    operation,
                    ErrorCategory.STORAGE_IO,
                    f"{operation} returned no result.",
                    "Repair the injected snapshot or artifact port and retry evaluation.",
                    field_path=operation,
                ),
            )
        )
    return value


def _invoke(
    method: Callable[..., object],
    *,
    positional: tuple[object, ...] = (),
    values: Mapping[str, object] = {},
) -> object:
    """Call a structural port with only parameters that it declares."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(*positional, **dict(values))

    parameters = tuple(signature.parameters.values())
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    explicit = tuple(
        parameter
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    )
    if has_var_kwargs:
        # Positional arguments already bind the corresponding declared
        # parameters.  Do not forward their alias values again: a structural
        # adapter may accept **kwargs while still rejecting duplicate names.
        bound_names = {
            parameter.name
            for index, parameter in enumerate(explicit)
            if index < len(positional)
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        }
        accepted = {
            name: value for name, value in values.items() if name not in bound_names
        }
        if positional:
            return method(*positional, **accepted)
        return method(**accepted)

    accepted: dict[str, object] = {}
    positional_only: list[object] = []
    positional_index = 0
    for parameter in parameters:
        if (
            parameter.name == "self"
            or parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ):
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if positional_index < len(positional):
                positional_only.append(positional[positional_index])
                positional_index += 1
            continue
        if parameter.name in values:
            accepted[parameter.name] = values[parameter.name]
    if positional_only:
        return method(*tuple(positional_only), **accepted)
    if accepted or not positional:
        return method(**accepted)
    return method(*positional)


def _date(value: object, name: str = "session") -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO calendar date") from error
    date_method = getattr(value, "date", None)
    if callable(date_method):
        converted = date_method()
        if isinstance(converted, datetime):
            return converted.date()
        if isinstance(converted, date):
            return converted
    raise TypeError(f"{name} must identify a calendar date")


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


def _as_tuple(value: object, name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        # A mapping of session -> record is a convenient fixture form.  A row
        # mapping (one that contains a session field) is still one row.
        if any(key in value for key in ("session", "date", "symbol", "adjusted_close")):
            return (value,)
        return tuple(value.values())
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of records")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of records") from error


def _error(
    operation: str,
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
        operation=operation,
        category=category,
        message=message,
        corrective_action=corrective_action,
        field_path=field_path,
        symbol=symbol,
        session=session,
        checksum=checksum,
    )


class _EvaluationFailure(Exception):
    """Internal carrier for already-sanitized evaluation errors."""

    def __init__(self, errors: Sequence[ActionableError]) -> None:
        values = tuple(errors)
        if not values:
            raise ValueError("evaluation failure requires an actionable error")
        super().__init__(values[0].message)
        self.errors = values


@dataclass(frozen=True, slots=True)
class CanonicalArtifact:
    """One deterministic result artifact and its content checksum."""

    role: str
    checksum: str
    byte_size: int
    media_type: str
    schema_version: str
    row_count: int | None
    payload: bytes = field(repr=False)
    reference: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        role = " ".join(self.role.split()) if isinstance(self.role, str) else ""
        if not role:
            raise ValueError("artifact role must not be blank")
        if not isinstance(self.payload, bytes):
            raise TypeError("artifact payload must be bytes")
        actual_checksum = sha256_bytes(self.payload)
        if self.checksum != actual_checksum:
            raise ValueError("artifact checksum does not match canonical payload")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size != len(self.payload)
        ):
            raise ValueError("artifact byte_size must equal payload length")
        if self.row_count is not None and (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count < 0
        ):
            raise ValueError(
                "artifact row_count must be a non-negative integer or None"
            )
        object.__setattr__(self, "role", role)

    @property
    def bytes(self) -> bytes:
        """Compatibility alias for callers that call the payload ``bytes``."""
        return self.payload

    @property
    def uri(self) -> str | None:
        return cast(str | None, getattr(self.reference, "relative_uri", None))

    def to_serializable(self) -> dict[str, object]:
        return {
            "byte_size": self.byte_size,
            "checksum": self.checksum,
            "media_type": self.media_type,
            "reference": self.uri,
            "role": self.role,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
        }


# Common names used by application composition roots.
ResultArtifact = CanonicalArtifact
EvaluationArtifact = CanonicalArtifact


@dataclass(frozen=True, slots=True)
class CanonicalResultArtifacts:
    """Ordered, duplicate-free artifacts emitted by one evaluation."""

    items: tuple[CanonicalArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("artifact items must be an immutable tuple")
        if any(not isinstance(item, CanonicalArtifact) for item in self.items):
            raise TypeError("artifact items must contain CanonicalArtifact values")
        roles = tuple(item.role for item in self.items)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if roles != tuple(sorted(roles)):
            raise ValueError("artifacts must be sorted by role")

    @property
    def artifacts(self) -> tuple[CanonicalArtifact, ...]:
        return self.items

    @property
    def checksums(self) -> Mapping[str, str]:
        return MappingProxyType({item.role: item.checksum for item in self.items})

    @property
    def artifact_checksums(self) -> Mapping[str, str]:
        return self.checksums

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.items)

    def get(self, role: str) -> CanonicalArtifact:
        aliases = {
            "returns": "strategy_returns",
            "equity_curve": "strategy_equity",
            "drawdowns": "drawdown",
            "monthly_return": "monthly_returns",
            "position": "positions",
            "order": "orders",
            "fill": "fills",
            "decision": "decisions",
            "metric": "metrics",
            "transactions": "fills",
            "chart_equity": "chart_equity_curve",
        }
        # Prefer an explicit role when the collection contains it.  Aliases
        # remain compatibility fallbacks, so ``transactions`` does not hide
        # the distinct combined transaction artifact when one is published.
        for artifact in self.items:
            if artifact.role == role:
                return artifact
        selected = aliases.get(role, role)
        for artifact in self.items:
            if artifact.role == selected:
                return artifact
        raise KeyError(role)

    def __getitem__(self, role: str) -> CanonicalArtifact:
        return self.get(role)

    def __iter__(self) -> Iterator[CanonicalArtifact]:
        return iter(self.items)

    def to_serializable(self) -> list[dict[str, object]]:
        return [item.to_serializable() for item in self.items]


EvaluationArtifacts = CanonicalResultArtifacts


def _artifact(
    role: str,
    schema_version: str,
    rows: Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    row_count: int | None = None,
    media_type: str = _CANONICAL_TABLE_MEDIA_TYPE,
) -> tuple[str, bytes, int | None, str]:
    """Encode a canonical table envelope with stable row ordering."""
    if isinstance(rows, Mapping):
        document: object = rows
        count = row_count
    else:
        normalized = [dict(row) for row in rows]
        # Domain values are canonicalized by canonical_json.  The sort key is
        # the complete row payload, making equivalent inputs order-independent.
        normalized.sort(key=lambda row: canonical_json(row))
        document = {"rows": normalized, "schema_version": schema_version}
        count = len(normalized) if row_count is None else row_count
    payload = canonical_json(document)
    return sha256_bytes(payload), payload, count, media_type


def _chart_artifact(
    role: str, spec: Mapping[str, object]
) -> tuple[str, bytes, int | None, str]:
    payload = canonical_json(dict(spec))
    return sha256_bytes(payload), payload, None, _CHART_MEDIA_TYPE


def _to_serializable(value: object) -> object:
    method = getattr(value, "to_serializable", None)
    if callable(method):
        return method()
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    return value


def _row(value: object, names: Sequence[str]) -> dict[str, object]:
    return {name: _to_serializable(_field(value, name)) for name in names}


def _sessioned(values: Iterable[object], name: str) -> tuple[object, ...]:
    normalized = _as_tuple(values, name)
    keyed: list[tuple[date, object]] = []
    for item in normalized:
        keyed.append(
            (_date(_field(item, ("session", "date")), f"{name}.session"), item)
        )
    keyed.sort(key=lambda pair: pair[0])
    sessions = [item[0] for item in keyed]
    if len(sessions) != len(set(sessions)):
        raise ValueError(f"{name} must not contain duplicate sessions")
    return tuple(item for _, item in keyed)


def _metric_rows(
    metric_sets: Sequence[EvaluationMetrics],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for metric_set in metric_sets:
        scope = MetricScope(metric_set.scope).value
        for metric in metric_set.metrics:
            value = metric.value
            rows.append(
                {
                    "name": MetricName(metric.name).value,
                    "null_reason": (
                        str(metric.null_reason.value)
                        if metric.null_reason is not None
                        else None
                    ),
                    "scope": scope,
                    "value": value,
                }
            )
    return tuple(rows)


def _state_rows(states: Sequence[object]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for state in states:
        positions = _as_tuple(_field(state, ("positions", "holdings"), ()), "positions")
        rows.append(
            {
                "cash_balance": _field(state, ("cash_balance", "cash")),
                "gross_exposure": _field(state, ("gross_exposure", "gross")),
                "leverage": _field(state, "leverage"),
                "portfolio_equity": _field(
                    state, ("portfolio_equity", "equity", "portfolio_value")
                ),
                "positions": [
                    {
                        "mark_price": _field(
                            position, ("mark_price", "price", "last_sale_price")
                        ),
                        "market_value": _field(position, ("market_value", "value")),
                        "quantity": _field(position, ("quantity", "amount")),
                        "symbol": normalize_symbol(
                            str(_field(position, ("symbol", "ticker"), ""))
                        ),
                    }
                    for position in sorted(
                        positions,
                        key=lambda position: normalize_symbol(
                            str(_field(position, ("symbol", "ticker"), ""))
                        ),
                    )
                ],
                "session": _date(_field(state, "session")),
            }
        )
    return tuple(rows)


def _drawdown_rows(
    curves: Mapping[str, Sequence[tuple[date, Decimal]]],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for scope, points in curves.items():
        peak: Decimal | None = None
        for session, equity in sorted(points):
            peak = equity if peak is None or equity > peak else peak
            assert peak is not None
            rows.append(
                {
                    "drawdown": equity / peak - Decimal("1"),
                    "scope": scope,
                    "session": session,
                }
            )
    rows.sort(key=lambda row: (str(row["session"]), str(row["scope"])))
    return tuple(rows)


def _equity_rows(
    curves: Mapping[str, Sequence[tuple[date, Decimal]]],
) -> tuple[dict[str, object], ...]:
    rows = [
        {"equity": equity, "scope": scope, "session": session}
        for scope, points in curves.items()
        for session, equity in sorted(points)
    ]
    rows.sort(key=lambda row: (str(row["session"]), str(row["scope"])))
    return tuple(rows)


def _monthly_rows(
    strategy: Sequence[MonthlyReturn], benchmark: Sequence[MonthlyReturn]
) -> tuple[dict[str, object], ...]:
    rows = [
        {"month": item.month, "return_value": item.return_value, "scope": "strategy"}
        for item in strategy
    ]
    rows.extend(
        {"month": item.month, "return_value": item.return_value, "scope": "benchmark"}
        for item in benchmark
    )
    rows.sort(key=lambda row: (str(row["month"]), str(row["scope"])))
    return tuple(rows)


def _decision_rows(decisions: Sequence[object]) -> tuple[dict[str, object], ...]:
    names = (
        "signal_session",
        "symbol",
        "endpoint_252_session",
        "endpoint_252_close",
        "endpoint_21_session",
        "endpoint_21_close",
        "momentum_score",
        "eligible",
        "rank",
        "target_weight",
        "exclusion_reason",
    )
    rows: list[dict[str, object]] = []
    for decision in decisions:
        values = _row(decision, names)
        weight = _field(decision, "target_weight")
        if weight is not None:
            to_string = getattr(weight, "to_canonical_string", None)
            values["target_weight"] = (
                to_string() if callable(to_string) else str(weight)
            )
        reason = _field(decision, "exclusion_reason")
        if reason is not None:
            values["exclusion_reason"] = getattr(reason, "value", reason)
        rows.append(values)
    rows.sort(key=lambda value: (str(value["signal_session"]), str(value["symbol"])))
    return tuple(rows)


def _order_rows(orders: Sequence[object]) -> tuple[dict[str, object], ...]:
    names = (
        "order_id",
        "signal_session",
        "execution_session",
        "symbol",
        "requested_quantity",
        "ordinal",
        "decision_rank",
        "status",
        "unfilled_reason",
    )
    rows = [_row(order, names) for order in orders]
    for row in rows:
        row["status"] = getattr(row["status"], "value", row["status"])
    rows.sort(
        key=lambda value: (
            str(value["signal_session"]),
            str(value["execution_session"]),
            str(value["symbol"]),
            int(value["ordinal"]),
            str(value["order_id"]),
        )
    )
    return tuple(rows)


def _fill_rows(fills: Sequence[object]) -> tuple[dict[str, object], ...]:
    names = (
        "fill_id",
        "order_id",
        "symbol",
        "session",
        "quantity",
        "ordinal",
        "base_adjusted_open",
        "fill_price",
        "gross_notional",
        "commission",
        "slippage_cost",
    )
    rows = [_row(fill, names) for fill in fills]
    rows.sort(
        key=lambda value: (
            str(value["session"]),
            str(value["symbol"]),
            int(value["ordinal"]),
            str(value["fill_id"]),
        )
    )
    return tuple(rows)


def _position_rows(states: Sequence[object]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for state in states:
        session = _date(_field(state, "session"))
        cash = _field(state, ("cash_balance", "cash"))
        rows.append(
            {
                "market_value": cash,
                "mark_price": None,
                "quantity": None,
                "row_kind": "cash",
                "session": session,
                "symbol": None,
            }
        )
        positions = _as_tuple(_field(state, ("positions", "holdings"), ()), "positions")
        for position in positions:
            rows.append(
                {
                    "market_value": _field(position, ("market_value", "value")),
                    "mark_price": _field(
                        position, ("mark_price", "price", "last_sale_price")
                    ),
                    "quantity": _field(position, ("quantity", "amount")),
                    "row_kind": "position",
                    "session": session,
                    "symbol": normalize_symbol(
                        str(_field(position, ("symbol", "ticker"), ""))
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["session"]),
            str(row["row_kind"]),
            str(row["symbol"] or ""),
        )
    )
    return tuple(rows)


def _return_rows(
    returns: Sequence[DailyReturn], scope: str
) -> tuple[dict[str, object], ...]:
    return tuple(
        {"return_value": item.return_value, "scope": scope, "session": item.session}
        for item in sorted(returns, key=lambda value: value.session)
    )


def _chart_spec(
    title: str, y_field: str, data_name: str, *, percent: bool = False
) -> dict[str, object]:
    axis: dict[str, object] = {"title": title}
    if percent:
        axis["format"] = ".2%"
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": data_name},
        "encoding": {
            "color": {"field": "scope", "type": "nominal"},
            "x": {"field": "session", "type": "temporal"},
            "y": {"axis": axis, "field": y_field, "type": "quantitative"},
        },
        "mark": {"type": "line"},
        "title": title,
    }


def _extract_range(
    *,
    explicit: DateRange | None,
    output: object,
    config: object | None,
    snapshot: object | None,
    strategy_sessions: Sequence[date],
) -> DateRange:
    if explicit is not None:
        return explicit
    for owner in (output, config, snapshot):
        candidates = (
            _field(owner, ("evaluation_range", "requested_range")),
            _field(
                _field(owner, "data", None), ("requested_range", "evaluation_range")
            ),
        )
        manifest = _field(owner, ("manifest", "snapshot_manifest"))
        identity = _field(manifest, "content_identity")
        candidates += (
            _field(identity, ("covered_range", "requested_range")),
            _field(owner, ("covered_range",)),
        )
        for candidate in candidates:
            if isinstance(candidate, DateRange):
                return candidate
            if candidate is not None:
                start = _field(candidate, "start")
                end = _field(candidate, "end")
                if start is not None and end is not None:
                    return DateRange(
                        _date(start, "evaluation_range.start"),
                        _date(end, "evaluation_range.end"),
                    )
    if not strategy_sessions:
        raise ValueError("evaluation requires at least one strategy return session")
    return DateRange(min(strategy_sessions), max(strategy_sessions))


def _snapshot_refs(snapshot: object | None) -> tuple[object, ...]:
    if snapshot is None:
        return ()
    values: object = _field(snapshot, ("object_references", "objects"), ())
    if not values:
        handle = _field(snapshot, ("handle", "snapshot_handle"))
        values = _field(handle, ("object_references", "objects"), ())
    if not values:
        manifest = _field(snapshot, ("manifest", "snapshot_manifest"))
        identity = _field(manifest, "content_identity")
        values = _field(identity, ("objects", "object_references"), ())
    return _as_tuple(values, "snapshot object references")


def _is_normalized_reference(reference: object) -> bool:
    kind = getattr(reference, "object_kind", _field(reference, "role"))
    kind = getattr(kind, "value", kind)
    schema = _field(reference, "schema_version", "")
    return (
        kind in {"normalized", "daily_bar", "daily_bar_v1"} or schema == "daily_bar_v1"
    )


def _row_from_bar(value: object) -> tuple[date, Decimal]:
    session_value = _field(value, ("session", "date", "expected_session"))
    close_value = _field(value, ("adjusted_close", "research_adjusted_close", "close"))
    if close_value is None and isinstance(value, Mapping):
        close_value = value.get("sizing_adjusted_close")
    return _date(session_value), _decimal(close_value, "SPY adjusted_close")


def _bar_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in ("session", "date", "adjusted_close", "close", "return_value")
        ):
            return (value,)
        return tuple(value.values())
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("benchmark bars must not be scalar text")
    return tuple(value) if isinstance(value, Iterable) else (value,)


class _BenchmarkValues(dict[date, Decimal]):
    """Normalized benchmark observations with an explicit value interpretation."""

    def __init__(
        self, values: Mapping[date, Decimal], *, values_are_returns: bool = False
    ) -> None:
        super().__init__(values)
        self.values_are_returns = values_are_returns


def _read_batches(reader: object) -> tuple[object, ...]:
    """Read bounded batches only; no read_all/to_pandas path is used."""
    direct_to_pylist = getattr(reader, "to_pylist", None)
    if callable(direct_to_pylist):
        return tuple(direct_to_pylist())
    to_batches = getattr(reader, "to_batches", None)
    batches: Iterable[object] = (
        to_batches() if callable(to_batches) else _bar_values(reader)
    )
    rows: list[object] = []
    for batch in batches:
        to_pylist = getattr(batch, "to_pylist", None)
        if callable(to_pylist):
            rows.extend(to_pylist())
        elif isinstance(batch, Mapping) or hasattr(batch, "session"):
            rows.append(batch)
        else:
            rows.extend(_bar_values(batch))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    """Complete deterministic evaluation result and artifact index."""

    evaluation_result: EvaluationResult
    evaluation_range: DateRange
    strategy_returns: tuple[DailyReturn, ...]
    benchmark_returns: tuple[DailyReturn, ...]
    strategy_equity: tuple[tuple[date, Decimal], ...]
    benchmark_equity: tuple[tuple[date, Decimal], ...]
    strategy_monthly_returns: tuple[MonthlyReturn, ...]
    benchmark_monthly_returns: tuple[MonthlyReturn, ...]
    artifacts: CanonicalResultArtifacts
    limitation_disclosure: LimitationDisclosure
    unfilled_orders: tuple[object, ...] = ()
    unfilled_diagnostics: tuple[ActionableError, ...] = ()
    ending_cash_balance: Decimal | None = None
    total_commissions: Decimal = Decimal("0.000000")
    total_slippage: Decimal = Decimal("0.000000")
    spy_gaps: tuple[date, ...] = ()

    @property
    def result(self) -> EvaluationResult:
        return self.evaluation_result

    @property
    def metrics(self) -> EvaluationResult:
        return self.evaluation_result

    @property
    def artifact_checksums(self) -> Mapping[str, str]:
        return self.artifacts.checksums

    @property
    def differences(self) -> EvaluationMetrics:
        return self.evaluation_result.differences

    def to_serializable(self) -> dict[str, object]:
        return {
            "artifacts": self.artifacts.to_serializable(),
            "benchmark_metrics": self.evaluation_result.benchmark_metrics.to_serializable(),
            "differences": self.evaluation_result.differences.to_serializable(),
            "ending_cash_balance": self.ending_cash_balance,
            "evaluation_range": self.evaluation_range.to_content_dict(),
            "limitation_disclosure": {
                "lines": list(self.limitation_disclosure.lines()),
                "version": self.limitation_disclosure.version,
            },
            "spy_gaps": list(self.spy_gaps),
            "strategy_metrics": self.evaluation_result.strategy_metrics.to_serializable(),
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "unfilled_orders": [
                _to_serializable(item) for item in self.unfilled_orders
            ],
        }


EvaluatedRun = EvaluationOutput


class EvaluationService:
    """Evaluate one audited core output against SPY from its exact snapshot."""

    operation_name = "evaluation.execute"

    def __init__(
        self,
        snapshot_manager: object | None = None,
        parquet_store: SnapshotScanPort | object | None = None,
        artifact_store: ArtifactPublicationPort | object | None = None,
        *,
        snapshot_reader: object | None = None,
        bar_reader: object | None = None,
        **compatibility: object,
    ) -> None:
        self.snapshot_manager = snapshot_manager or compatibility.pop("snapshot", None)
        self.parquet_store = (
            parquet_store
            or compatibility.pop("parquet", None)
            or compatibility.pop("data_store", None)
        )
        self.artifact_store = (
            artifact_store
            or compatibility.pop("artifacts", None)
            or compatibility.pop("artifact_repository", None)
        )
        self.snapshot_reader = snapshot_reader or compatibility.pop("reader", None)
        self.bar_reader = bar_reader or compatibility.pop("normalized_reader", None)
        if compatibility:
            unknown = ", ".join(sorted(compatibility))
            raise TypeError(f"unsupported EvaluationService arguments: {unknown}")

    def evaluate(
        self,
        output: object | None = None,
        snapshot: object | None = None,
        config: object | None = None,
        *,
        core_output: object | None = None,
        audited_output: object | None = None,
        evaluation_range: DateRange | None = None,
        request: object | None = None,
        snapshot_id: str | None = None,
    ) -> Result[EvaluationOutput]:
        """Return a gap-safe, deterministic evaluation or actionable errors."""
        del request
        candidate = output or core_output or audited_output
        if candidate is None:
            return Err(
                (self._input_error("core backtest output is required", "core_output"),)
            )
        try:
            core = self._core_output(candidate)
            if (
                snapshot is None
                and snapshot_id is not None
                and self.snapshot_manager is not None
            ):
                opener = getattr(self.snapshot_manager, "open_verified", None)
                if callable(opener):
                    snapshot = _unwrap(
                        _invoke(
                            opener,
                            positional=(snapshot_id,),
                            values={"snapshot_id": snapshot_id},
                        ),
                        "snapshot.open",
                    )
            states = _sessioned(
                _field(core, "portfolio_states", ()), "portfolio_states"
            )
            raw_returns = _sessioned(_field(core, "daily_returns", ()), "daily_returns")
            strategy_returns, strategy_equity, range_value = self._strategy_series(
                states, raw_returns, candidate, config, snapshot, evaluation_range
            )
            benchmark_values = self._benchmark_values(snapshot, range_value)
            missing = tuple(
                session
                for session in (item.session for item in strategy_returns)
                if session not in benchmark_values
            )
            if missing:
                errors = tuple(
                    _error(
                        "evaluation.benchmark",
                        ErrorCategory.VALIDATION_GAP,
                        f"SPY has no accepted adjusted observation for evaluation session {session.isoformat()}.",
                        "Repair the SPY gap in the pinned snapshot or select a snapshot with complete benchmark coverage.",
                        field_path="benchmark.sessions",
                        symbol="SPY",
                        session=session,
                    )
                    for session in missing
                )
                self.last_errors = errors
                return Err(errors, preserve_order=True)
            benchmark_returns, benchmark_equity = self._benchmark_series(
                strategy_returns, benchmark_values
            )
            fills = _as_tuple(_field(core, "fills", ()), "fills")
            orders = _as_tuple(_field(core, "orders", ()), "orders")
            strategy_metrics = calculate_evaluation_metrics(
                MetricScope.STRATEGY,
                tuple(
                    state
                    for state in states
                    if _date(_field(state, "session"))
                    in {item.session for item in strategy_returns}
                )
                if states and all(isinstance(state, PortfolioState) for state in states)
                else tuple(equity for _, equity in strategy_equity),
                returns=tuple(item.return_value for item in strategy_returns),
                fills=cast(Iterable[FillRecord], fills),
                orders=cast(Iterable[OrderRecord], orders),
                portfolio_equity=tuple(equity for _, equity in strategy_equity),
            )
            benchmark_metrics = calculate_evaluation_metrics(
                MetricScope.BENCHMARK,
                tuple(equity for _, equity in benchmark_equity),
                returns=tuple(item.return_value for item in benchmark_returns),
            )
            differences = strategy_minus_benchmark(strategy_metrics, benchmark_metrics)
            result = EvaluationResult(strategy_metrics, benchmark_metrics, differences)
            strategy_monthly = calculate_monthly_compounding(strategy_returns)
            benchmark_monthly = calculate_monthly_compounding(benchmark_returns)
            disclosure = self._disclosure(snapshot)
            artifacts = self._build_artifacts(
                core=core,
                result=result,
                states=states,
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                strategy_equity=strategy_equity,
                benchmark_equity=benchmark_equity,
                strategy_monthly=strategy_monthly,
                benchmark_monthly=benchmark_monthly,
            )
            artifacts = self._publish_artifacts(artifacts)
            unfilled = tuple(
                order
                for order in orders
                if getattr(
                    getattr(order, "status", None),
                    "value",
                    getattr(order, "status", None),
                )
                in {OrderStatus.PARTIALLY_FILLED.value, OrderStatus.UNFILLED.value}
            )
            ending_cash = (
                _field(states[-1], ("cash_balance", "cash")) if states else None
            )
            output_value = EvaluationOutput(
                evaluation_result=result,
                evaluation_range=range_value,
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                strategy_equity=strategy_equity,
                benchmark_equity=benchmark_equity,
                strategy_monthly_returns=strategy_monthly,
                benchmark_monthly_returns=benchmark_monthly,
                artifacts=artifacts,
                limitation_disclosure=disclosure,
                unfilled_orders=unfilled,
                unfilled_diagnostics=(),
                ending_cash_balance=_decimal(ending_cash, "ending_cash_balance")
                if ending_cash is not None
                else None,
                total_commissions=self._metric_decimal(
                    strategy_metrics, MetricName.TOTAL_COMMISSIONS
                ),
                total_slippage=self._metric_decimal(
                    strategy_metrics, MetricName.TOTAL_SLIPPAGE
                ),
            )
            self.last_errors = ()
            self.last_result = output_value
            return Ok(output_value)
        except _EvaluationFailure as failure:
            self.last_errors = failure.errors
            return Err(failure.errors, preserve_order=True)
        except (
            TypeError,
            ValueError,
            ArithmeticError,
            KeyError,
            InvalidOperation,
        ) as failure:
            error = self._input_error(str(failure), "evaluation")
            self.last_errors = (error,)
            return Err((error,), preserve_order=True)
        except Exception as failure:
            error = ActionableError.from_unexpected_exception(
                self.operation_name, failure
            )
            self.last_errors = (error,)
            return Err((error,), preserve_order=True)

    run = evaluate
    execute = evaluate

    @staticmethod
    def _core_output(value: object) -> object:
        # BacktestResult and AuditReport expose these aliases.  A raw core
        # output is accepted unchanged.
        for name in ("audited_output", "core_output", "output"):
            candidate = getattr(value, name, None)
            if candidate is not None and candidate is not value:
                if name == "output" and not any(
                    hasattr(value, required)
                    for required in ("orders", "fills", "daily_returns")
                ):
                    return EvaluationService._core_output(candidate)
                return candidate
        if not all(
            hasattr(value, required)
            for required in ("orders", "fills", "portfolio_states", "daily_returns")
        ):
            raise ValueError(
                "core output must contain orders, fills, portfolio_states, and daily_returns"
            )
        return value

    def _strategy_series(
        self,
        states: Sequence[object],
        raw_returns: Sequence[object],
        candidate: object,
        config: object | None,
        snapshot: object | None,
        evaluation_range: DateRange | None,
    ) -> tuple[tuple[DailyReturn, ...], tuple[tuple[date, Decimal], ...], DateRange]:
        # The caller's candidate/configuration/snapshot may carry the requested
        # range, so retain them for the range extractor rather than discarding
        # them before it is called.
        state_map = {
            _date(_field(state, "session")): _decimal(
                _field(state, ("portfolio_equity", "equity", "portfolio_value")),
                "portfolio_equity",
            )
            for state in states
        }
        returns: list[DailyReturn] = []
        for item in raw_returns:
            session = _date(_field(item, "session"), "daily_return.session")
            value = _decimal(
                _field(item, ("return_value", "returns", "value")),
                "daily_return.return_value",
            )
            returns.append(DailyReturn(session, value))
        returns.sort(key=lambda item: item.session)
        if not returns:
            ordered_sessions = tuple(sorted(state_map))
            if len(ordered_sessions) < 1:
                raise ValueError("core output contains no evaluation sessions")
            previous: Decimal | None = None
            for session in ordered_sessions:
                equity = state_map[session]
                value = (
                    Decimal("0")
                    if previous is None
                    else equity / previous - Decimal("1")
                )
                returns.append(DailyReturn(session, value))
                previous = equity
        if len({item.session for item in returns}) != len(returns):
            raise ValueError("daily_returns must not contain duplicate sessions")
        requested = _extract_range(
            explicit=evaluation_range,
            output=candidate,
            config=config,
            snapshot=snapshot,
            strategy_sessions=[item.session for item in returns],
        )
        selected_returns = tuple(
            item for item in returns if requested.start <= item.session <= requested.end
        )
        if not selected_returns:
            raise ValueError("evaluation range contains no strategy return sessions")
        curve: list[tuple[date, Decimal]] = []
        for item in selected_returns:
            if item.session not in state_map:
                raise ValueError(
                    f"missing portfolio equity for evaluation session {item.session.isoformat()}"
                )
            curve.append((item.session, state_map[item.session]))
        return (
            selected_returns,
            tuple(curve),
            DateRange(selected_returns[0].session, selected_returns[-1].session),
        )

    def _benchmark_values(
        self, snapshot: object | None, range_value: DateRange
    ) -> Mapping[date, Decimal]:
        direct = self._direct_benchmark(snapshot)
        if direct is not None:
            name = next(
                name
                for name in (
                    "benchmark_bars",
                    "spy_bars",
                    "benchmark_series",
                    "spy_series",
                    "benchmark_returns",
                    "spy_returns",
                    "benchmark_values",
                    "spy_values",
                )
                if _field(snapshot, name) is direct
            )
            return self._normalize_benchmark(
                direct,
                values_are_returns=name in {"benchmark_returns", "spy_returns"},
            )
        refs = tuple(
            reference
            for reference in _snapshot_refs(snapshot)
            if _is_normalized_reference(reference)
        )
        if self.bar_reader is not None:
            method = getattr(self.bar_reader, "read", self.bar_reader)
            if callable(method):
                value = _invoke(
                    cast(Callable[..., object], method),
                    positional=(snapshot, "SPY", range_value),
                    values={
                        "snapshot": snapshot,
                        "snapshot_handle": snapshot,
                        "symbol": "SPY",
                        "benchmark_symbol": "SPY",
                        "start": range_value.start,
                        "end": range_value.end,
                        "session_start": range_value.start,
                        "session_end": range_value.end,
                    },
                )
                return self._normalize_benchmark(_unwrap(value, "snapshot.benchmark"))
        store = self.parquet_store
        if store is None:
            reader = self.snapshot_reader or _field(
                snapshot, ("reader", "store", "storage")
            )
            store = reader
        if store is not None:
            for method_name in (
                "scan",
                "scan_normalized",
                "read_normalized",
                "read_bars",
            ):
                method = getattr(store, method_name, None)
                if not callable(method):
                    continue
                try:
                    value = _invoke(
                        cast(Callable[..., object], method),
                        positional=(refs, ("symbol", "session", "adjusted_close")),
                        values={
                            "refs": refs,
                            "references": refs,
                            "columns": ("symbol", "session", "adjusted_close"),
                            "snapshot": snapshot,
                            "snapshot_handle": snapshot,
                            "symbol": "SPY",
                            "symbols": ("SPY",),
                            "benchmark_symbol": "SPY",
                            "session_start": range_value.start,
                            "session_end": range_value.end,
                            "start": range_value.start,
                            "end": range_value.end,
                        },
                    )
                    return self._normalize_benchmark(
                        _unwrap(value, "snapshot.benchmark")
                    )
                except TypeError:
                    continue
        return {}

    @staticmethod
    def _direct_benchmark(snapshot: object | None) -> object | None:
        if snapshot is None:
            return None
        for name in (
            "benchmark_bars",
            "spy_bars",
            "benchmark_series",
            "spy_series",
            "benchmark_returns",
            "spy_returns",
            "benchmark_values",
            "spy_values",
        ):
            value = _field(snapshot, name)
            if value is not None:
                return value
        return None

    @staticmethod
    def _normalize_benchmark(
        value: object, *, values_are_returns: bool = False
    ) -> Mapping[date, Decimal]:
        result: dict[date, Decimal] = {}
        if isinstance(value, Mapping) and not any(
            key in value
            for key in ("session", "date", "adjusted_close", "close", "return_value")
        ):
            rows: Iterable[object] = tuple(
                {
                    "session": session,
                    "return_value"
                    if values_are_returns
                    else "adjusted_close": observation,
                }
                for session, observation in value.items()
            )
        else:
            rows = _read_batches(value)
        for item in rows:
            if isinstance(item, DailyReturn) or (
                _field(item, "return_value") is not None
                and _field(item, "adjusted_close") is None
            ):
                session = _date(_field(item, "session"))
                close = _decimal(_field(item, "return_value"), "benchmark return")
                values_are_returns = True
            else:
                session, close = _row_from_bar(item)
            if session in result and result[session] != close:
                raise ValueError(
                    f"benchmark contains conflicting rows for {session.isoformat()}"
                )
            result[session] = close
        return _BenchmarkValues(result, values_are_returns=values_are_returns)

    @staticmethod
    def _benchmark_series(
        strategy_returns: Sequence[DailyReturn], values: Mapping[date, Decimal]
    ) -> tuple[tuple[DailyReturn, ...], tuple[tuple[date, Decimal], ...]]:
        # The benchmark starts at the same normalized USD 100,000 base as the
        # strategy.  Thus the first aligned session is a zero return; subsequent
        # returns use adjacent adjusted SPY closes.  If the caller supplied
        # returns rather than prices, values are already return values and are
        # used directly.
        sessions = tuple(item.session for item in strategy_returns)
        ordered_values = tuple(values[session] for session in sessions)
        looks_like_returns = bool(getattr(values, "values_are_returns", False))
        returns: list[DailyReturn] = []
        equity: list[tuple[date, Decimal]] = []
        current = INITIAL_PORTFOLIO_EQUITY
        for index, session in enumerate(sessions):
            if looks_like_returns:
                return_value = ordered_values[index]
            elif index == 0:
                return_value = Decimal("0")
            else:
                previous = ordered_values[index - 1]
                if previous <= 0:
                    raise ValueError("SPY adjusted close must be positive")
                return_value = ordered_values[index] / previous - Decimal("1")
            returns.append(DailyReturn(session, return_value))
            if index:
                current *= Decimal("1") + return_value
            equity.append((session, current))
        return tuple(returns), tuple(equity)

    @staticmethod
    def _disclosure(snapshot: object | None) -> LimitationDisclosure:
        value = _field(snapshot, "limitation_disclosure")
        if not isinstance(value, LimitationDisclosure):
            manifest = _field(snapshot, ("manifest", "snapshot_manifest"))
            value = _field(manifest, "limitation_disclosure")
        return (
            value
            if isinstance(value, LimitationDisclosure)
            else LimitationDisclosure.current()
        )

    def _build_artifacts(
        self,
        *,
        core: object,
        result: EvaluationResult,
        states: Sequence[object],
        strategy_returns: Sequence[DailyReturn],
        benchmark_returns: Sequence[DailyReturn],
        strategy_equity: Sequence[tuple[date, Decimal]],
        benchmark_equity: Sequence[tuple[date, Decimal]],
        strategy_monthly: Sequence[MonthlyReturn],
        benchmark_monthly: Sequence[MonthlyReturn],
    ) -> CanonicalResultArtifacts:
        curves = {"strategy": strategy_equity, "benchmark": benchmark_equity}
        payloads: list[CanonicalArtifact] = []

        def add(
            role: str,
            schema: str,
            rows: Sequence[Mapping[str, object]] | Mapping[str, object],
            *,
            media_type: str = _CANONICAL_TABLE_MEDIA_TYPE,
            row_count: int | None = None,
        ) -> None:
            checksum, payload, count, resolved_media = _artifact(
                role, schema, rows, row_count=row_count, media_type=media_type
            )
            payloads.append(
                CanonicalArtifact(
                    role, checksum, len(payload), resolved_media, schema, count, payload
                )
            )

        add(
            "benchmark_returns",
            _SCHEMA_VERSIONS["returns"],
            _return_rows(benchmark_returns, "benchmark"),
            row_count=len(benchmark_returns),
        )
        add(
            "benchmark_equity",
            _SCHEMA_VERSIONS["equity"],
            _equity_rows({"benchmark": benchmark_equity}),
            row_count=len(benchmark_equity),
        )
        add(
            "chart_drawdown",
            _SCHEMA_VERSIONS["chart"],
            _chart_spec(
                "Strategy and SPY drawdown", "drawdown", "drawdown", percent=True
            ),
            media_type=_CHART_MEDIA_TYPE,
        )
        add(
            "chart_equity_curve",
            _SCHEMA_VERSIONS["chart"],
            _chart_spec("Strategy and SPY equity", "equity", "equity_curve"),
            media_type=_CHART_MEDIA_TYPE,
        )
        add(
            "chart_monthly_returns",
            _SCHEMA_VERSIONS["chart"],
            _chart_spec(
                "Monthly returns", "return_value", "monthly_returns", percent=True
            ),
            media_type=_CHART_MEDIA_TYPE,
        )
        add(
            "decisions",
            _SCHEMA_VERSIONS["decisions"],
            _decision_rows(
                _as_tuple(_field(core, "strategy_decisions", ()), "strategy_decisions")
            ),
        )
        add(
            "drawdown",
            _SCHEMA_VERSIONS["drawdown"],
            _drawdown_rows(curves),
            row_count=len(strategy_equity) + len(benchmark_equity),
        )
        add(
            "fills",
            _SCHEMA_VERSIONS["fills"],
            _fill_rows(_as_tuple(_field(core, "fills", ()), "fills")),
        )
        add(
            "metrics",
            _SCHEMA_VERSIONS["metrics"],
            _metric_rows(
                (result.strategy_metrics, result.benchmark_metrics, result.differences)
            ),
        )
        add(
            "monthly_returns",
            _SCHEMA_VERSIONS["monthly_returns"],
            _monthly_rows(strategy_monthly, benchmark_monthly),
        )
        add(
            "orders",
            _SCHEMA_VERSIONS["orders"],
            _order_rows(_as_tuple(_field(core, "orders", ()), "orders")),
        )
        add(
            "portfolio",
            _SCHEMA_VERSIONS["portfolio"],
            _state_rows(states),
            row_count=len(states),
        )
        add("positions", _SCHEMA_VERSIONS["positions"], _position_rows(states))
        add(
            "strategy_equity",
            _SCHEMA_VERSIONS["equity"],
            _equity_rows({"strategy": strategy_equity}),
            row_count=len(strategy_equity),
        )
        add(
            "strategy_returns",
            _SCHEMA_VERSIONS["returns"],
            _return_rows(strategy_returns, "strategy"),
            row_count=len(strategy_returns),
        )
        # A compact combined transaction view is useful to inspection callers;
        # orders and fills remain separately checksummed above.
        add(
            "transactions",
            _SCHEMA_VERSIONS["transactions"],
            {
                "fills": list(
                    _fill_rows(_as_tuple(_field(core, "fills", ()), "fills"))
                ),
                "orders": list(
                    _order_rows(_as_tuple(_field(core, "orders", ()), "orders"))
                ),
                "schema_version": _SCHEMA_VERSIONS["transactions"],
            },
        )
        return CanonicalResultArtifacts(
            tuple(sorted(payloads, key=lambda item: item.role))
        )

    def _publish_artifacts(
        self, artifacts: CanonicalResultArtifacts
    ) -> CanonicalResultArtifacts:
        store = self.artifact_store
        if store is None:
            return artifacts
        published: list[CanonicalArtifact] = []
        for artifact in artifacts:
            try:
                reference = self._publish_one(store, artifact)
            except _EvaluationFailure:
                raise
            except Exception as failure:
                raise _EvaluationFailure(
                    (
                        ActionableError.from_unexpected_exception(
                            "artifact.publish",
                            failure,
                            correlation_id=artifact.checksum,
                        ),
                    )
                ) from None
            published.append(
                CanonicalArtifact(
                    artifact.role,
                    artifact.checksum,
                    artifact.byte_size,
                    artifact.media_type,
                    artifact.schema_version,
                    artifact.row_count,
                    artifact.payload,
                    reference,
                )
            )
        return CanonicalResultArtifacts(
            tuple(sorted(published, key=lambda item: item.role))
        )

    @staticmethod
    def _publish_one(store: object, artifact: CanonicalArtifact) -> object | None:
        metadata = {
            "artifact_kind": artifact.role,
            "byte_size": artifact.byte_size,
            "checksum": artifact.checksum,
            "media_type": artifact.media_type,
            "row_count": artifact.row_count,
            "schema_version": artifact.schema_version,
        }
        for method_name in ("publish_artifact", "put", "store", "write"):
            method = getattr(store, method_name, None)
            if not callable(method):
                continue
            if method_name == "publish_artifact" and callable(
                getattr(store, "create_staging", None)
            ):
                staging = _invoke(
                    cast(Callable[..., object], getattr(store, "create_staging")),
                    values={
                        "operation_id": f"evaluation-{artifact.checksum[:16]}-{uuid4().hex}"
                    },
                )
                staged = _invoke(
                    cast(Callable[..., object], getattr(store, "stage_bytes")),
                    positional=(
                        staging,
                        f"evaluation/{artifact.role}-{artifact.checksum}.json",
                        artifact.payload,
                    ),
                    values={
                        "staging": staging,
                        "relative_path": f"evaluation/{artifact.role}-{artifact.checksum}.json",
                        "data": artifact.payload,
                        "bytes": artifact.payload,
                        "expected_checksum": artifact.checksum,
                    },
                )
                return _invoke(
                    method,
                    positional=(staged,),
                    values={"staged": staged, "artifact": staged, "metadata": metadata},
                )
            return _invoke(
                method,
                positional=(artifact.payload,),
                values={
                    "payload": artifact.payload,
                    "data": artifact.payload,
                    "bytes": artifact.payload,
                    "role": artifact.role,
                    "metadata": metadata,
                    "checksum": artifact.checksum,
                },
            )
        raise _EvaluationFailure(
            (
                _error(
                    "artifact.publish",
                    ErrorCategory.STORAGE_IO,
                    "The artifact store does not expose a publication method.",
                    "Configure an artifact store with publish_artifact, put, store, or write support.",
                    field_path="artifact_store",
                ),
            )
        )

    @staticmethod
    def _metric_decimal(metrics: EvaluationMetrics, name: MetricName) -> Decimal:
        value = metrics.metric(name).value
        return value if isinstance(value, Decimal) else Decimal("0.000000")

    @staticmethod
    def _input_error(message: str, field_path: str) -> ActionableError:
        return _error(
            "evaluation.input",
            ErrorCategory.CONFIGURATION_INVALID_VALUE,
            " ".join(message.splitlines()) or "Invalid evaluation input.",
            "Provide a complete audited core output and a verified snapshot covering its evaluation sessions.",
            field_path=field_path,
        )


# Descriptive aliases used by composition roots and tests.
Evaluation = EvaluationService
EvaluationResultOutput = EvaluationOutput
CanonicalResultArtifact = CanonicalArtifact


__all__ = [
    "CanonicalArtifact",
    "CanonicalResultArtifact",
    "CanonicalResultArtifacts",
    "EvaluatedRun",
    "Evaluation",
    "EvaluationArtifact",
    "EvaluationArtifacts",
    "EvaluationOutput",
    "EvaluationResultOutput",
    "EvaluationService",
    "ResultArtifact",
    "SnapshotScanPort",
]
