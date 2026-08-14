"""Explicit Arrow schemas and canonical conversions for persisted tables.

The platform's table boundary is deliberately independent of pandas.  Every
schema has an immutable name/version in Arrow metadata, fixed field order and
nullability, and a conversion path that accepts only canonical values.  This
keeps Parquet byte identity and table validation deterministic before the
storage layer is introduced.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeAlias

import pyarrow as pa  # type: ignore[import-untyped]

from quant_research_platform.domain.canonical import canonical_json, canonical_rational
from quant_research_platform.domain.evaluation import EvaluationMetrics, MetricValue
from quant_research_platform.domain.execution import (
    DailyReturn,
    FillRecord,
    OrderRecord,
    PortfolioState,
)
from quant_research_platform.domain.market import (
    DailyBarCandidate,
    DataGap,
    ProviderRecord,
    QuarantineRecord,
    ValidationReport,
)
from quant_research_platform.domain.strategy import StrategyDecision

RAW_V1: Final = "raw_v1"
DAILY_BAR_V1: Final = "daily_bar_v1"
QUARANTINE_V1: Final = "quarantine_v1"
GAP_V1: Final = "gap_v1"
VALIDATION_REPORT_V1: Final = "validation_report_v1"
DECISIONS_V1: Final = "decisions_v1"
ORDERS_V1: Final = "orders_v1"
FILLS_V1: Final = "fills_v1"
POSITIONS_V1: Final = "positions_v1"
PORTFOLIO_V1: Final = "portfolio_v1"
RETURNS_V1: Final = "returns_v1"
METRICS_V1: Final = "metrics_v1"
MONTHLY_RETURN_V1: Final = "monthly_return_v1"
MONTHLY_RETURNS_V1: Final = MONTHLY_RETURN_V1

SCHEMA_NAME_METADATA_KEY: Final = b"qrp.schema_name"
SCHEMA_VERSION_METADATA_KEY: Final = b"qrp.schema_version"
_SCHEMA_METADATA_KEYS: Final = frozenset(
    {SCHEMA_NAME_METADATA_KEY, SCHEMA_VERSION_METADATA_KEY}
)

CHECKSUM_BINARY_TYPE: Final = pa.binary(32)
MONEY_DECIMAL_TYPE: Final = pa.decimal128(38, 6)
RATIO_DECIMAL_TYPE: Final = pa.decimal128(38, 18)
UTC_TIMESTAMP_TYPE: Final = pa.timestamp("us", tz="UTC")

# These options are scientific-output preconditions. Changing any option,
# PyArrow version, or write_chunk_size intentionally changes resulting bytes.
PARQUET_WRITE_OPTIONS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "version": "2.6",
        "data_page_version": "2.0",
        "compression": "zstd",
        "compression_level": 3,
        "use_dictionary": False,
        "write_statistics": False,
        "data_page_size": 1_048_576,
        "write_batch_size": 1_024,
        "use_compliant_nested_type": True,
        "store_schema": True,
    }
)

SchemaRow: TypeAlias = Mapping[str, object]

_CHECKSUM_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE: Final = re.compile(r"^[A-Z0-9.-]+$")
_RATIONAL_RE: Final = re.compile(r"^(0|[1-9][0-9]*)/[1-9][0-9]*$")

_ENUM_VALUES: Final[dict[tuple[str, str], frozenset[str]]] = {
    (QUARANTINE_V1, "source_kind"): frozenset(
        {"provider_record", "daily_bar_candidate"}
    ),
    (DECISIONS_V1, "exclusion_reason"): frozenset(
        {
            "missing_long_endpoint",
            "missing_short_endpoint",
            "asset_not_tradable",
            "not_selected",
        }
    ),
    (ORDERS_V1, "status"): frozenset(
        {"pending", "partially_filled", "filled", "unfilled"}
    ),
    (POSITIONS_V1, "row_kind"): frozenset({"cash", "position"}),
    (METRICS_V1, "scope"): frozenset({"strategy", "benchmark", "difference"}),
    (MONTHLY_RETURN_V1, "scope"): frozenset({"strategy", "benchmark"}),
    (METRICS_V1, "name"): frozenset(
        {
            "total_return",
            "compound_annual_growth_rate",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "turnover",
            "total_commissions",
            "total_slippage",
            "unfilled_orders",
            "ending_cash_balance",
        }
    ),
    (METRICS_V1, "null_reason"): frozenset(
        {
            "no_evaluation_sessions",
            "insufficient_observations",
            "zero_volatility",
            "non_positive_equity",
        }
    ),
}


def _required(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=False)


def _optional(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=True)


def _schema(name: str, fields: Sequence[pa.Field]) -> pa.Schema:
    """Build a versioned schema with only platform-owned metadata."""
    encoded = name.encode("ascii")
    return pa.schema(
        fields,
        metadata={
            SCHEMA_NAME_METADATA_KEY: encoded,
            SCHEMA_VERSION_METADATA_KEY: encoded,
        },
    )


RAW_V1_SCHEMA: Final = _schema(
    RAW_V1,
    (
        _required("provider", pa.string()),
        _required("request_content_key", pa.string()),
        _required("symbol", pa.string()),
        _required("provider_date", pa.date32()),
        _optional("open", pa.float64()),
        _optional("high", pa.float64()),
        _optional("low", pa.float64()),
        _optional("close", pa.float64()),
        _optional("adj_close", pa.float64()),
        _optional("volume", pa.float64()),
        _optional("dividends", pa.float64()),
        _optional("stock_splits", pa.float64()),
        _required("provider_fields_json", pa.string()),
        _required("provider_record_checksum", CHECKSUM_BINARY_TYPE),
    ),
)

DAILY_BAR_V1_SCHEMA: Final = _schema(
    DAILY_BAR_V1,
    (
        _required("symbol", pa.string()),
        _required("session", pa.date32()),
        _required("event_ts", UTC_TIMESTAMP_TYPE),
        _required("raw_open", pa.float64()),
        _required("raw_high", pa.float64()),
        _required("raw_low", pa.float64()),
        _required("raw_close", pa.float64()),
        _required("raw_volume", pa.float64()),
        _optional("provider_adj_close", pa.float64()),
        _required("dividend", pa.float64()),
        _required("split_ratio", pa.float64()),
        _required("adjusted_open", pa.float64()),
        _required("adjusted_high", pa.float64()),
        _required("adjusted_low", pa.float64()),
        _required("adjusted_close", pa.float64()),
        _required("adjusted_volume", pa.float64()),
        _required("execution_adjusted_open", pa.float64()),
        _required("sizing_adjusted_close", pa.float64()),
        _required("cumulative_price_factor", RATIO_DECIMAL_TYPE),
        _required("cumulative_split_factor", RATIO_DECIMAL_TYPE),
        _required("policy_version", pa.string()),
        _required("provider_record_checksum", CHECKSUM_BINARY_TYPE),
        _required("canonical_row_checksum", CHECKSUM_BINARY_TYPE),
    ),
)

QUARANTINE_V1_SCHEMA: Final = _schema(
    QUARANTINE_V1,
    (
        _required("source_kind", pa.string()),
        _optional("symbol", pa.string()),
        _optional("session", pa.date32()),
        _required("reason_codes", pa.list_(pa.string())),
        _required("offending_values_json", pa.string()),
        _required("schema_version", pa.string()),
        _optional("policy_version", pa.string()),
        _optional("provider_record_checksum", CHECKSUM_BINARY_TYPE),
        _optional("candidate_checksum", CHECKSUM_BINARY_TYPE),
    ),
)

GAP_V1_SCHEMA: Final = _schema(
    GAP_V1,
    (
        _required("symbol", pa.string()),
        _required("expected_session", pa.date32()),
        _required("requested_start", pa.date32()),
        _required("requested_end", pa.date32()),
        _required("parent_retained", pa.bool_()),
        _required("reason", pa.string()),
    ),
)

VALIDATION_REPORT_V1_SCHEMA: Final = _schema(
    VALIDATION_REPORT_V1,
    (
        _required("schema_version", pa.string()),
        _required("calendar_version", pa.string()),
        _required("accepted_count", pa.int64()),
        _required("quarantined_count", pa.int64()),
        _required("duplicate_count", pa.int64()),
        _required("gap_count", pa.int64()),
        _required("failed_symbols", pa.list_(pa.string())),
        _required("retained_parent_coverage_symbols", pa.list_(pa.string())),
        _required("stale_symbols", pa.list_(pa.string())),
        _optional("covered_start", pa.date32()),
        _optional("covered_end", pa.date32()),
        _required("comparison_ready", pa.bool_()),
        _required("per_symbol_json", pa.string()),
        _required("quarantined_by_reason_json", pa.string()),
        _required("gaps_json", pa.string()),
        _required("report_checksum", CHECKSUM_BINARY_TYPE),
    ),
)

DECISIONS_V1_SCHEMA: Final = _schema(
    DECISIONS_V1,
    (
        _required("signal_session", pa.date32()),
        _required("symbol", pa.string()),
        _optional("endpoint_252_session", pa.date32()),
        _optional("endpoint_252_close", RATIO_DECIMAL_TYPE),
        _optional("endpoint_21_session", pa.date32()),
        _optional("endpoint_21_close", RATIO_DECIMAL_TYPE),
        _optional("momentum_score", RATIO_DECIMAL_TYPE),
        _required("eligible", pa.bool_()),
        _optional("rank", pa.int32()),
        _required("target_weight", pa.string()),
        _optional("exclusion_reason", pa.string()),
    ),
)

ORDERS_V1_SCHEMA: Final = _schema(
    ORDERS_V1,
    (
        _required("order_id", pa.string()),
        _required("signal_session", pa.date32()),
        _required("execution_session", pa.date32()),
        _required("symbol", pa.string()),
        _required("requested_quantity", pa.int64()),
        _required("ordinal", pa.int32()),
        _optional("decision_rank", pa.int32()),
        _required("status", pa.string()),
        _optional("unfilled_reason", pa.string()),
    ),
)

FILLS_V1_SCHEMA: Final = _schema(
    FILLS_V1,
    (
        _required("fill_id", pa.string()),
        _required("order_id", pa.string()),
        _required("symbol", pa.string()),
        _required("session", pa.date32()),
        _required("quantity", pa.int64()),
        _required("ordinal", pa.int32()),
        _required("base_adjusted_open", MONEY_DECIMAL_TYPE),
        _required("fill_price", MONEY_DECIMAL_TYPE),
        _required("gross_notional", MONEY_DECIMAL_TYPE),
        _required("commission", MONEY_DECIMAL_TYPE),
        _required("slippage_cost", MONEY_DECIMAL_TYPE),
    ),
)

POSITIONS_V1_SCHEMA: Final = _schema(
    POSITIONS_V1,
    (
        _required("session", pa.date32()),
        _required("row_kind", pa.string()),
        _optional("symbol", pa.string()),
        _optional("quantity", pa.int64()),
        _optional("mark_price", MONEY_DECIMAL_TYPE),
        _required("market_value", MONEY_DECIMAL_TYPE),
    ),
)

PORTFOLIO_V1_SCHEMA: Final = _schema(
    PORTFOLIO_V1,
    (
        _required("session", pa.date32()),
        _required("cash_balance", MONEY_DECIMAL_TYPE),
        _required("gross_exposure", MONEY_DECIMAL_TYPE),
        _required("leverage", RATIO_DECIMAL_TYPE),
        _required("portfolio_equity", MONEY_DECIMAL_TYPE),
    ),
)

RETURNS_V1_SCHEMA: Final = _schema(
    RETURNS_V1,
    (
        _required("session", pa.date32()),
        _required("return_value", RATIO_DECIMAL_TYPE),
    ),
)

METRICS_V1_SCHEMA: Final = _schema(
    METRICS_V1,
    (
        _required("scope", pa.string()),
        _required("name", pa.string()),
        _optional("value_decimal", RATIO_DECIMAL_TYPE),
        _optional("value_integer", pa.int64()),
        _optional("null_reason", pa.string()),
    ),
)

MONTHLY_RETURN_V1_SCHEMA: Final = _schema(
    MONTHLY_RETURN_V1,
    (
        _required("month", pa.date32()),
        _required("scope", pa.string()),
        _required("return_value", RATIO_DECIMAL_TYPE),
    ),
)
MONTHLY_RETURNS_V1_SCHEMA: Final = MONTHLY_RETURN_V1_SCHEMA

# Lowercase aliases make each named schema easy to import in writer code.
raw_v1: Final = RAW_V1_SCHEMA
daily_bar_v1: Final = DAILY_BAR_V1_SCHEMA
quarantine_v1: Final = QUARANTINE_V1_SCHEMA
gap_v1: Final = GAP_V1_SCHEMA
validation_report_v1: Final = VALIDATION_REPORT_V1_SCHEMA
decisions_v1: Final = DECISIONS_V1_SCHEMA
orders_v1: Final = ORDERS_V1_SCHEMA
fills_v1: Final = FILLS_V1_SCHEMA
positions_v1: Final = POSITIONS_V1_SCHEMA
portfolio_v1: Final = PORTFOLIO_V1_SCHEMA
returns_v1: Final = RETURNS_V1_SCHEMA
metrics_v1: Final = METRICS_V1_SCHEMA
monthly_return_v1: Final = MONTHLY_RETURN_V1_SCHEMA
monthly_returns_v1: Final = MONTHLY_RETURN_V1_SCHEMA

SCHEMAS: Final[dict[str, pa.Schema]] = {
    RAW_V1: RAW_V1_SCHEMA,
    DAILY_BAR_V1: DAILY_BAR_V1_SCHEMA,
    QUARANTINE_V1: QUARANTINE_V1_SCHEMA,
    GAP_V1: GAP_V1_SCHEMA,
    VALIDATION_REPORT_V1: VALIDATION_REPORT_V1_SCHEMA,
    DECISIONS_V1: DECISIONS_V1_SCHEMA,
    ORDERS_V1: ORDERS_V1_SCHEMA,
    FILLS_V1: FILLS_V1_SCHEMA,
    POSITIONS_V1: POSITIONS_V1_SCHEMA,
    PORTFOLIO_V1: PORTFOLIO_V1_SCHEMA,
    RETURNS_V1: RETURNS_V1_SCHEMA,
    METRICS_V1: METRICS_V1_SCHEMA,
    MONTHLY_RETURN_V1: MONTHLY_RETURN_V1_SCHEMA,
}

_SORT_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    RAW_V1: ("symbol", "provider_date", "provider_record_checksum"),
    DAILY_BAR_V1: ("symbol", "session", "canonical_row_checksum"),
    QUARANTINE_V1: (
        "source_kind",
        "symbol",
        "session",
        "reason_codes",
        "candidate_checksum",
    ),
    GAP_V1: ("symbol", "expected_session", "reason"),
    VALIDATION_REPORT_V1: ("report_checksum",),
    DECISIONS_V1: ("signal_session", "symbol"),
    ORDERS_V1: ("signal_session", "execution_session", "symbol", "ordinal"),
    FILLS_V1: ("session", "symbol", "ordinal", "fill_id"),
    POSITIONS_V1: ("session", "row_kind", "symbol"),
    PORTFOLIO_V1: ("session",),
    RETURNS_V1: ("session",),
    METRICS_V1: ("scope", "name"),
    MONTHLY_RETURN_V1: ("month", "scope"),
}


def schema_for(schema_name: str) -> pa.Schema:
    """Return the exact registered schema for *schema_name* or raise clearly."""
    try:
        return SCHEMAS[schema_name]
    except KeyError as error:
        supported = ", ".join(sorted(SCHEMAS))
        raise ValueError(
            f"unsupported schema version {schema_name!r}; expected one of {supported}"
        ) from error


def _canonical_json_cell(value: object, field_name: str) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field_name} must contain canonical JSON") from error
        canonical = canonical_json(parsed).decode("utf-8").rstrip("\n")
        if value != canonical:
            raise ValueError(f"{field_name} must use canonical JSON encoding")
        return value
    try:
        return canonical_json(value).decode("utf-8").rstrip("\n")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be JSON-canonicalizable") from error


def _canonical_text(value: object, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    if value.encode("utf-8").decode("utf-8") != value:
        raise ValueError(f"{field_name} must be valid UTF-8 text")
    # NFC is intentionally checked without importing a second canonicalizer.
    if value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{field_name} must be Unicode NFC-normalized")
    return value


def _checksum_bytes(value: object, field_name: str) -> bytes:
    if isinstance(value, str):
        if _CHECKSUM_RE.fullmatch(value) is None:
            raise ValueError(f"{field_name} must be a lowercase SHA-256 checksum")
        return bytes.fromhex(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        result = bytes(value)
        if len(result) != 32:
            raise ValueError(f"{field_name} must contain exactly 32 bytes")
        return result
    raise TypeError(f"{field_name} must be checksum text or 32 raw bytes")


def _canonical_decimal(value: object, type_: pa.DataType, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal, not a float or string")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if not pa.types.is_decimal(type_):  # pragma: no cover - defensive caller guard
        raise TypeError(f"{field_name} does not have a decimal Arrow type")
    decimal_type = type_
    scale = decimal_type.scale
    precision = decimal_type.precision
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"{field_name} must use a finite decimal exponent")
    if -exponent > scale:
        raise ValueError(
            f"{field_name} exceeds the canonical decimal scale of {scale} places"
        )
    scaled = value.scaleb(scale)
    if scaled != scaled.to_integral_value():  # pragma: no cover - scale guard above
        raise ValueError(f"{field_name} cannot be represented at scale {scale}")
    if len(str(abs(int(scaled)))) > precision:
        raise ValueError(
            f"{field_name} exceeds the canonical decimal precision of {precision}"
        )
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Decimal, float, int)):
        raise TypeError(f"{field_name} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _canonical_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a UTC datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC, not a non-UTC timezone")
    if value.tzname() != "UTC":
        raise ValueError(f"{field_name} must use a UTC timezone label")
    return value.astimezone(UTC)


def _canonical_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _canonical_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _coerce_field_value(field: pa.Field, value: object) -> object:
    if value is None:
        if field.nullable:
            return None
        raise ValueError(f"missing required non-null field: {field.name}")

    if field.name.endswith("_json"):
        return _canonical_json_cell(value, field.name)
    if pa.types.is_string(field.type):
        return _canonical_text(value, field.name)
    if pa.types.is_fixed_size_binary(field.type):
        return _checksum_bytes(value, field.name)
    if pa.types.is_date32(field.type):
        return _canonical_date(value, field.name)
    if pa.types.is_timestamp(field.type):
        return _canonical_timestamp(value, field.name)
    if pa.types.is_decimal(field.type):
        return _canonical_decimal(value, field.type, field.name)
    if pa.types.is_floating(field.type):
        return _finite_float(value, field.name)
    if pa.types.is_integer(field.type):
        return _canonical_integer(value, field.name)
    if pa.types.is_boolean(field.type):
        if not isinstance(value, bool):
            raise TypeError(f"{field.name} must be a bool")
        return value
    if pa.types.is_list(field.type):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field.name} must be an immutable/list sequence")
        value_type = field.type.value_type
        if not pa.types.is_string(value_type):  # pragma: no cover - schema invariant
            raise TypeError(f"unsupported list type for {field.name}")
        return [_canonical_text(item, field.name) for item in value]
    raise TypeError(f"unsupported Arrow type for {field.name}: {field.type}")


def _require_symbol(value: object, field_name: str = "symbol") -> None:
    assert isinstance(value, str)
    if _SYMBOL_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an uppercase normalized ticker")


def _require_checksum_text(value: object, field_name: str) -> None:
    assert isinstance(value, str)
    if _CHECKSUM_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 checksum")


def _require_enum(schema_name: str, field_name: str, value: object) -> None:
    if value is None:
        return
    allowed = _ENUM_VALUES.get((schema_name, field_name))
    if allowed is None:
        return
    assert isinstance(value, str)
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {expected}")


def _require_rational(value: object) -> None:
    assert isinstance(value, str)
    if _RATIONAL_RE.fullmatch(value) is None:
        raise ValueError("target_weight must be a canonical non-negative rational")
    numerator_text, denominator_text = value.split("/")
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if canonical_rational(numerator, denominator) != value:
        raise ValueError("target_weight must be a reduced canonical rational")


def _validate_row_rules(schema_name: str, row: Mapping[str, object]) -> None:
    for field_name in row:
        _require_enum(schema_name, field_name, row[field_name])

    for field_name in ("symbol",):
        if field_name in row and row[field_name] is not None:
            _require_symbol(row[field_name], field_name)
    if schema_name == RAW_V1:
        if row["provider"] != "yfinance":
            raise ValueError("provider must be the canonical Phase 1 value 'yfinance'")
        _require_checksum_text(row["request_content_key"], "request_content_key")
    elif schema_name == DAILY_BAR_V1:
        if row["policy_version"] != "causal_forward_v1":
            raise ValueError("policy_version must be 'causal_forward_v1'")
    elif schema_name == QUARANTINE_V1:
        if row["schema_version"] != QUARANTINE_V1:
            raise ValueError("schema_version must equal 'quarantine_v1'")
        reasons = row["reason_codes"]
        assert isinstance(reasons, list)
        if not reasons or len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must be a non-empty duplicate-free list")
        if any(
            re.fullmatch(r"[a-z][a-z0-9_.-]*", reason) is None
            for reason in reasons
        ):
            raise ValueError("reason_codes must use canonical lowercase reason strings")
        if (row["symbol"] is None) != (row["session"] is None):
            raise ValueError(
                "quarantine symbol and session must both be present or null"
            )
    elif schema_name == GAP_V1:
        requested_start = row["requested_start"]
        requested_end = row["requested_end"]
        assert isinstance(requested_start, date)
        assert isinstance(requested_end, date)
        if requested_start > requested_end:
            raise ValueError("requested_start must not be after requested_end")
    elif schema_name == VALIDATION_REPORT_V1:
        if row["schema_version"] != VALIDATION_REPORT_V1:
            raise ValueError("schema_version must equal 'validation_report_v1'")
        if (row["covered_start"] is None) != (row["covered_end"] is None):
            raise ValueError(
                "covered_start and covered_end must both be present or null"
            )
        covered_start = row["covered_start"]
        covered_end = row["covered_end"]
        assert covered_start is None or isinstance(covered_start, date)
        assert covered_end is None or isinstance(covered_end, date)
        if covered_start is not None:
            assert covered_end is not None
            if covered_start > covered_end:
                raise ValueError("covered_start must not be after covered_end")
    elif schema_name == DECISIONS_V1:
        _require_rational(row["target_weight"])
        if row["eligible"] and row["rank"] is None:
            raise ValueError("eligible decisions must have a rank")
        if not row["eligible"] and row["rank"] is not None:
            raise ValueError("ineligible decisions must not have a rank")
    elif schema_name == ORDERS_V1:
        if row["requested_quantity"] == 0:
            raise ValueError("requested_quantity must not be zero")
        status = row["status"]
        reason = row["unfilled_reason"]
        if status in {"pending", "filled"} and reason is not None:
            raise ValueError(f"{status} orders must not have an unfilled_reason")
        if status in {"partially_filled", "unfilled"} and reason is None:
            raise ValueError(f"{status} orders require an unfilled_reason")
    elif schema_name == FILLS_V1:
        if row["quantity"] == 0:
            raise ValueError("fill quantity must not be zero")
    elif schema_name == POSITIONS_V1:
        kind = row["row_kind"]
        if kind == "cash":
            if any(
                row[name] is not None for name in ("symbol", "quantity", "mark_price")
            ):
                raise ValueError("cash rows must not contain position fields")
        else:
            if (
                row["symbol"] is None
                or row["quantity"] is None
                or row["mark_price"] is None
            ):
                raise ValueError(
                    "position rows require symbol, quantity, and mark_price"
                )
            quantity = row["quantity"]
            assert isinstance(quantity, int)
            if quantity <= 0:
                raise ValueError("position quantities must be positive")
    elif schema_name == METRICS_V1:
        name = row["name"]
        decimal_value = row["value_decimal"]
        integer_value = row["value_integer"]
        null_reason = row["null_reason"]
        if null_reason is not None:
            if decimal_value is not None or integer_value is not None:
                raise ValueError("null metrics must not contain a numeric value")
        elif name == "unfilled_orders":
            if integer_value is None or decimal_value is not None:
                raise ValueError("unfilled_orders must use value_integer")
        elif decimal_value is None or integer_value is not None:
            raise ValueError("non-count metrics must use value_decimal")
    elif schema_name == MONTHLY_RETURN_V1:
        month = row["month"]
        assert isinstance(month, date)
        if month.day != 1:
            raise ValueError("month must be the first calendar date of its month")


def _normalize_row(schema_name: str, row: SchemaRow) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise TypeError("canonical table rows must be mappings")
    schema = schema_for(schema_name)
    source = dict(row)
    expected_names = {field.name for field in schema}
    extra_names = sorted(set(source) - expected_names)
    if extra_names:
        names = ", ".join(extra_names)
        raise ValueError(f"unexpected fields for {schema_name}: {names}")

    normalized: dict[str, object] = {}
    for field in schema:
        value = source.get(field.name)
        normalized[field.name] = _coerce_field_value(field, value)
    _validate_row_rules(schema_name, normalized)
    return normalized


def _sort_component(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, list):
        return tuple(value)
    return value


def _row_sort_key(schema_name: str, row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(_sort_component(row[name]) for name in _SORT_FIELDS[schema_name])


def canonical_table(schema_name: str, rows: Iterable[SchemaRow]) -> pa.Table:
    """Convert records to a sorted, schema-exact, pandas-free Arrow table.

    The function refuses implicit float/string-to-decimal conversion, unknown
    enum variants, non-UTC timestamps, missing required fields, and schema
    drift.  Sorting happens after canonical field conversion, so input order
    never changes the table's logical representation.
    """
    normalized = [_normalize_row(schema_name, row) for row in rows]
    normalized.sort(key=lambda row: _row_sort_key(schema_name, row))
    schema = schema_for(schema_name)
    table = pa.Table.from_pylist(normalized, schema=schema)
    table = table.replace_schema_metadata(schema.metadata)
    validate_canonical_table(table, schema_name)
    return table


def validate_canonical_table(table: pa.Table, schema_name: str) -> None:
    """Reject an Arrow table whose schema, metadata, values, or order drifted."""
    if not isinstance(table, pa.Table):
        raise TypeError("table must be a pyarrow.Table")
    schema = schema_for(schema_name)
    metadata = table.schema.metadata or {}
    if b"pandas" in metadata:
        raise ValueError("pandas metadata is not permitted in canonical tables")
    if metadata.get(SCHEMA_NAME_METADATA_KEY) != schema_name.encode("ascii"):
        raise ValueError(f"incompatible or missing schema name for {schema_name}")
    if metadata.get(SCHEMA_VERSION_METADATA_KEY) != schema_name.encode("ascii"):
        raise ValueError(f"incompatible schema version for {schema_name}")
    if not table.schema.equals(schema, check_metadata=True):
        raise ValueError(f"table schema does not exactly match {schema_name}")

    normalized = [_normalize_row(schema_name, row) for row in table.to_pylist()]
    canonical_order = sorted(
        normalized,
        key=lambda row: _row_sort_key(schema_name, row),
    )
    if normalized != canonical_order:
        raise ValueError(f"{schema_name} rows are not in canonical sort order")


def canonical_rows(table: pa.Table, schema_name: str) -> tuple[dict[str, object], ...]:
    """Validate a table and return its canonical Python-row representation."""
    validate_canonical_table(table, schema_name)
    return tuple(_normalize_row(schema_name, row) for row in table.to_pylist())


def raw_records_to_table(records: Iterable[ProviderRecord]) -> pa.Table:
    """Convert immutable provider records to the authoritative raw schema."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, ProviderRecord):
            raise TypeError("records must contain ProviderRecord values")
        rows.append(
            {
                "provider": record.provider,
                "request_content_key": record.request_content_key,
                "symbol": record.symbol,
                "provider_date": record.provider_date,
                "open": record.raw_bar.open,
                "high": record.raw_bar.high,
                "low": record.raw_bar.low,
                "close": record.raw_bar.close,
                "adj_close": record.raw_bar.adj_close,
                "volume": record.raw_bar.volume,
                "dividends": record.raw_action.dividend,
                "stock_splits": record.raw_action.split_ratio,
                "provider_fields_json": {
                    "action": record.raw_action.provider_fields,
                    "record": record.provider_fields,
                },
                "provider_record_checksum": record.provider_record_checksum,
            }
        )
    return canonical_table(RAW_V1, rows)


def daily_bars_to_table(records: Iterable[DailyBarCandidate]) -> pa.Table:
    """Convert accepted normalized bars to the strict daily-bar schema."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, DailyBarCandidate):
            raise TypeError("records must contain DailyBarCandidate values")
        rows.append(
            {
                "symbol": record.symbol,
                "session": record.session,
                "event_ts": record.event_timestamp,
                "raw_open": record.raw_open,
                "raw_high": record.raw_high,
                "raw_low": record.raw_low,
                "raw_close": record.raw_close,
                "raw_volume": record.raw_volume,
                "provider_adj_close": record.provider_adj_close,
                "dividend": record.corporate_action.dividend,
                "split_ratio": record.corporate_action.split_ratio,
                "adjusted_open": record.adjusted_open,
                "adjusted_high": record.adjusted_high,
                "adjusted_low": record.adjusted_low,
                "adjusted_close": record.adjusted_close,
                "adjusted_volume": record.adjusted_volume,
                "execution_adjusted_open": record.execution_adjusted_open,
                "sizing_adjusted_close": record.sizing_adjusted_close,
                "cumulative_price_factor": record.cumulative_price_factor,
                "cumulative_split_factor": record.cumulative_split_factor,
                "policy_version": record.policy_version,
                "provider_record_checksum": record.raw_lineage.provider_record_checksum,
                "canonical_row_checksum": record.canonical_row_checksum,
            }
        )
    return canonical_table(DAILY_BAR_V1, rows)


def quarantines_to_table(records: Iterable[QuarantineRecord]) -> pa.Table:
    """Convert deterministic quarantine decisions without volatile detection times."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, QuarantineRecord):
            raise TypeError("records must contain QuarantineRecord values")
        rows.append(
            {
                "source_kind": record.source_kind,
                "symbol": record.symbol,
                "session": record.session,
                "reason_codes": record.reason_codes,
                "offending_values_json": record.offending_values,
                "schema_version": record.schema_version,
                "policy_version": record.policy_version,
                "provider_record_checksum": (
                    record.raw_lineage.provider_record_checksum
                    if record.raw_lineage is not None
                    else None
                ),
                "candidate_checksum": record.candidate_checksum,
            }
        )
    return canonical_table(QUARANTINE_V1, rows)


def gaps_to_table(records: Iterable[DataGap]) -> pa.Table:
    """Convert explicit missing-session facts without creating synthetic bars."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, DataGap):
            raise TypeError("records must contain DataGap values")
        rows.append(
            {
                "symbol": record.symbol,
                "expected_session": record.expected_session,
                "requested_start": record.requested_range.start,
                "requested_end": record.requested_range.end,
                "parent_retained": record.parent_retained,
                "reason": record.reason,
            }
        )
    return canonical_table(GAP_V1, rows)


def validation_reports_to_table(reports: Iterable[ValidationReport]) -> pa.Table:
    """Convert complete reports to one canonical, checksummed row per report."""
    rows: list[dict[str, object]] = []
    for report in reports:
        if not isinstance(report, ValidationReport):
            raise TypeError("reports must contain ValidationReport values")
        summary = report.summary
        rows.append(
            {
                "schema_version": report.schema_version,
                "calendar_version": report.calendar_version,
                "accepted_count": summary.accepted_row_count,
                "quarantined_count": summary.quarantined_row_count,
                "duplicate_count": summary.collapsed_duplicate_count,
                "gap_count": summary.gap_count,
                "failed_symbols": summary.failed_symbols,
                "retained_parent_coverage_symbols": (
                    summary.retained_parent_coverage_symbols
                ),
                "stale_symbols": summary.stale_symbols,
                "covered_start": (
                    summary.covered_range.start if summary.covered_range else None
                ),
                "covered_end": (
                    summary.covered_range.end if summary.covered_range else None
                ),
                "comparison_ready": summary.comparison_ready,
                "per_symbol_json": [
                    item.to_content_dict() for item in report.per_symbol
                ],
                "quarantined_by_reason_json": [
                    {"reason": reason, "count": count}
                    for reason, count in report.quarantined_by_reason
                ],
                "gaps_json": [gap.to_content_dict() for gap in report.gaps],
                "report_checksum": report.content_checksum,
            }
        )
    return canonical_table(VALIDATION_REPORT_V1, rows)


def decisions_to_table(records: Iterable[StrategyDecision]) -> pa.Table:
    """Convert complete strategy decisions with exact rational target weights."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, StrategyDecision):
            raise TypeError("records must contain StrategyDecision values")
        rows.append(
            {
                "signal_session": record.signal_session,
                "symbol": record.symbol,
                "endpoint_252_session": record.endpoint_252_session,
                "endpoint_252_close": record.endpoint_252_close,
                "endpoint_21_session": record.endpoint_21_session,
                "endpoint_21_close": record.endpoint_21_close,
                "momentum_score": record.momentum_score,
                "eligible": record.eligible,
                "rank": record.rank,
                "target_weight": record.target_weight.to_canonical_string(),
                "exclusion_reason": (
                    str(record.exclusion_reason)
                    if record.exclusion_reason is not None
                    else None
                ),
            }
        )
    return canonical_table(DECISIONS_V1, rows)


def orders_to_table(records: Iterable[OrderRecord]) -> pa.Table:
    """Convert deterministic whole-share order records."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, OrderRecord):
            raise TypeError("records must contain OrderRecord values")
        rows.append(record.to_serializable())
    return canonical_table(ORDERS_V1, rows)


def fills_to_table(records: Iterable[FillRecord]) -> pa.Table:
    """Convert deterministic fills with fixed-money Decimal fields."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, FillRecord):
            raise TypeError("records must contain FillRecord values")
        rows.append(record.to_serializable())
    return canonical_table(FILLS_V1, rows)


def positions_to_table(states: Iterable[PortfolioState]) -> pa.Table:
    """Expand each state into non-zero position rows and exactly one cash row."""
    rows: list[dict[str, object]] = []
    for state in states:
        if not isinstance(state, PortfolioState):
            raise TypeError("states must contain PortfolioState values")
        rows.append(
            {
                "session": state.session,
                "row_kind": "cash",
                "symbol": None,
                "quantity": None,
                "mark_price": None,
                "market_value": state.cash_balance,
            }
        )
        rows.extend(
            {
                "session": state.session,
                "row_kind": "position",
                "symbol": position.symbol,
                "quantity": position.quantity,
                "mark_price": position.mark_price,
                "market_value": position.market_value,
            }
            for position in state.positions
        )
    return canonical_table(POSITIONS_V1, rows)


def portfolio_states_to_table(states: Iterable[PortfolioState]) -> pa.Table:
    """Convert daily portfolio totals used for independent accounting checks."""
    rows: list[dict[str, object]] = []
    for state in states:
        if not isinstance(state, PortfolioState):
            raise TypeError("states must contain PortfolioState values")
        rows.append(
            {
                "session": state.session,
                "cash_balance": state.cash_balance,
                "gross_exposure": state.gross_exposure,
                "leverage": state.leverage,
                "portfolio_equity": state.portfolio_equity,
            }
        )
    return canonical_table(PORTFOLIO_V1, rows)


def returns_to_table(records: Iterable[DailyReturn]) -> pa.Table:
    """Convert ordered daily return records without binary floating-point drift."""
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, DailyReturn):
            raise TypeError("records must contain DailyReturn values")
        rows.append(record.to_serializable())
    return canonical_table(RETURNS_V1, rows)


def metrics_to_table(metric_sets: Iterable[EvaluationMetrics]) -> pa.Table:
    """Flatten complete metric scopes while preserving null reasons and value type."""
    rows: list[dict[str, object]] = []
    for metric_set in metric_sets:
        if not isinstance(metric_set, EvaluationMetrics):
            raise TypeError("metric_sets must contain EvaluationMetrics values")
        for metric in metric_set.metrics:
            rows.append(_metric_row(str(metric_set.scope), metric))
    return canonical_table(METRICS_V1, rows)


def _metric_row(scope: str, metric: MetricValue) -> dict[str, object]:
    value_decimal: Decimal | None
    value_integer: int | None
    if isinstance(metric.value, Decimal):
        value_decimal = metric.value
        value_integer = None
    elif isinstance(metric.value, int):
        value_decimal = None
        value_integer = metric.value
    else:
        value_decimal = None
        value_integer = None
    return {
        "scope": scope,
        "name": str(metric.name),
        "value_decimal": value_decimal,
        "value_integer": value_integer,
        "null_reason": (
            str(metric.null_reason) if metric.null_reason is not None else None
        ),
    }


def monthly_returns_to_table(rows: Iterable[SchemaRow]) -> pa.Table:
    """Convert monthly-compounded returns supplied as explicit canonical rows."""
    return canonical_table(MONTHLY_RETURN_V1, rows)


monthly_return_to_table = monthly_returns_to_table
raw_to_table = raw_records_to_table
daily_bar_to_table = daily_bars_to_table
quarantine_to_table = quarantines_to_table
gap_to_table = gaps_to_table
validation_report_to_table = validation_reports_to_table
portfolio_to_table = portfolio_states_to_table

__all__ = [
    "CHECKSUM_BINARY_TYPE",
    "DAILY_BAR_V1",
    "DAILY_BAR_V1_SCHEMA",
    "DECISIONS_V1",
    "DECISIONS_V1_SCHEMA",
    "FILLS_V1",
    "FILLS_V1_SCHEMA",
    "GAP_V1",
    "GAP_V1_SCHEMA",
    "METRICS_V1",
    "METRICS_V1_SCHEMA",
    "MONEY_DECIMAL_TYPE",
    "MONTHLY_RETURN_V1",
    "MONTHLY_RETURN_V1_SCHEMA",
    "MONTHLY_RETURNS_V1",
    "MONTHLY_RETURNS_V1_SCHEMA",
    "ORDERS_V1",
    "ORDERS_V1_SCHEMA",
    "PORTFOLIO_V1",
    "PORTFOLIO_V1_SCHEMA",
    "POSITIONS_V1",
    "POSITIONS_V1_SCHEMA",
    "QUARANTINE_V1",
    "QUARANTINE_V1_SCHEMA",
    "RATIO_DECIMAL_TYPE",
    "RAW_V1",
    "RAW_V1_SCHEMA",
    "RETURNS_V1",
    "RETURNS_V1_SCHEMA",
    "SCHEMAS",
    "SCHEMA_NAME_METADATA_KEY",
    "SCHEMA_VERSION_METADATA_KEY",
    "UTC_TIMESTAMP_TYPE",
    "VALIDATION_REPORT_V1",
    "VALIDATION_REPORT_V1_SCHEMA",
    "canonical_rows",
    "canonical_table",
    "daily_bar_to_table",
    "daily_bars_to_table",
    "decisions_to_table",
    "fills_to_table",
    "gap_to_table",
    "gaps_to_table",
    "metrics_to_table",
    "monthly_return_to_table",
    "monthly_returns_to_table",
    "orders_to_table",
    "portfolio_states_to_table",
    "portfolio_to_table",
    "positions_to_table",
    "quarantine_to_table",
    "quarantines_to_table",
    "raw_records_to_table",
    "raw_to_table",
    "returns_to_table",
    "schema_for",
    "validate_canonical_table",
    "validation_report_to_table",
    "validation_reports_to_table",
]
