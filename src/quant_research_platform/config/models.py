"""Frozen Pydantic models for validated Phase 1 configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

DEFAULT_UNIVERSE: Final[tuple[str, ...]] = ("AAPL", "JPM", "MSFT", "PG", "XOM")
DEFAULT_BENCHMARK: Final[Literal["SPY"]] = "SPY"
DEFAULT_INITIAL_EQUITY_USD: Final[Decimal] = Decimal("100000")
REDACTION_MARKER: Final = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class UnresolvedSecret:
    """A redacted secret whose value must be supplied by an external source.

    Canonical configuration YAML may retain the field name and this marker but
    must never turn the marker into a credential.  The object is deliberately
    value-free and safe to persist or show to users.
    """

    marker: Literal["[REDACTED]"] = REDACTION_MARKER

    def __str__(self) -> str:
        return REDACTION_MARKER


SecretValue: TypeAlias = SecretStr | UnresolvedSecret | None


class FrozenConfigModel(BaseModel):
    """Base schema that makes every configuration layer immutable and closed."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def _normalize_universe(value: object) -> tuple[str, ...]:
    """Return ordered normalized symbols or raise a position-specific error."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("universe must be an ordered sequence of symbols")

    normalized_symbols: list[str] = []
    seen: set[str] = set()
    for index, symbol in enumerate(value):
        if not isinstance(symbol, str):
            raise ValueError(f"universe[{index}] must be a string")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError(f"universe[{index}] must not be empty after normalization")
        if normalized in seen:
            raise ValueError(
                f"universe[{index}] duplicates normalized symbol {normalized!r}"
            )
        seen.add(normalized)
        normalized_symbols.append(normalized)
    return tuple(normalized_symbols)


def _default_position_count(value: object) -> int:
    """Compute the root-derived default without masking invalid universe errors."""
    try:
        normalized = _normalize_universe(value)
    except ValueError:
        return min(5, len(DEFAULT_UNIVERSE))
    if not normalized:
        return min(5, len(DEFAULT_UNIVERSE))
    return min(5, len(normalized))


class DateRangeConfig(FrozenConfigModel):
    """Inclusive ISO 8601 calendar-date interval requested from the provider."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> DateRangeConfig:
        if self.start > self.end:
            raise ValueError("requested range start must be no later than end")
        return self


class PathConfig(FrozenConfigModel):
    """Approved local paths before project-root resolution."""

    data_root: Path = Path("data")
    artifact_root: Path = Path("data/artifacts")
    metadata_db: Path = Path("data/metadata.duckdb")
    mlflow_db: Path = Path("data/mlflow.db")
    local_secrets_file: Path | None = Path("config/secrets.local.yaml")


class RetryPolicyConfig(FrozenConfigModel):
    """Bounded, deterministic retry policy for provider operations."""

    attempts: int = Field(3, ge=1, le=5)
    initial_delay_seconds: Decimal = Field(Decimal("1"), ge=0, le=60)
    max_delay_seconds: Decimal = Field(Decimal("8"), ge=0, le=60)
    backoff_multiplier: Decimal = Field(Decimal("2.0"), ge=1, le=4)

    @model_validator(mode="after")
    def validate_delay_relationship(self) -> RetryPolicyConfig:
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds must be at least initial_delay_seconds")
        return self


class DataConfig(FrozenConfigModel):
    """Provider, universe, range, ingestion-rate, and storage-chunk settings."""

    universe: tuple[str, ...] = Field(
        DEFAULT_UNIVERSE,
        min_length=1,
        max_length=25,
    )
    requested_range: DateRangeConfig
    benchmark: Literal["SPY"] = DEFAULT_BENCHMARK
    provider: Literal["yfinance"] = "yfinance"
    batch_size: int = Field(5, ge=1, le=10)
    staleness_sessions: int = Field(1, ge=0, le=252)
    revision_overlap_sessions: int = Field(5, ge=0, le=252)
    write_chunk_rows: int = Field(50_000, ge=1, le=100_000)

    @field_validator("universe", mode="before")
    @classmethod
    def normalize_symbols(cls, value: object) -> tuple[str, ...]:
        return _normalize_universe(value)


class StrategyConfig(FrozenConfigModel):
    """The single approved, interpretable Phase 1 momentum strategy."""

    identifier: Literal["monthly_momentum_v1"] = "monthly_momentum_v1"
    position_count: int = Field(5, ge=1)
    long_lookback_sessions: int = Field(252, ge=252, le=252)
    skip_recent_sessions: int = Field(21, ge=21, le=21)


class ExecutionConfig(FrozenConfigModel):
    """Fixed initial equity and finite, non-negative transaction cost assumptions."""

    initial_equity_usd: Decimal = Field(
        DEFAULT_INITIAL_EQUITY_USD,
        allow_inf_nan=False,
    )
    commission_bps: Decimal = Field(Decimal("5"), ge=0, allow_inf_nan=False)
    slippage_bps: Decimal = Field(Decimal("10"), ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_fixed_initial_equity(self) -> ExecutionConfig:
        if self.initial_equity_usd != DEFAULT_INITIAL_EQUITY_USD:
            raise ValueError("initial_equity_usd must be fixed at USD 100000")
        return self


class UiConfig(FrozenConfigModel):
    """Bounds for ordinary paginated Web UI tables."""

    page_size: int = Field(100, ge=1, le=100)


class RuntimeConfig(FrozenConfigModel):
    """Deterministic process settings recorded with scientific runs."""

    deterministic_seed: int = Field(0, ge=0, le=4_294_967_295)


class SecretConfig(FrozenConfigModel):
    """Optional proxy credentials accepted only by later approved secret sources."""

    http_proxy: SecretValue = None
    https_proxy: SecretValue = None

    @field_validator("http_proxy", "https_proxy", mode="before")
    @classmethod
    def preserve_redaction_marker_as_unresolved(cls, value: object) -> object:
        """Keep canonical markers value-free when redacted YAML is reloaded."""
        if value == REDACTION_MARKER:
            return UnresolvedSecret()
        return value


class ResolvedConfig(FrozenConfigModel):
    """Complete frozen configuration after source precedence and schema validation."""

    paths: PathConfig
    retry: RetryPolicyConfig = RetryPolicyConfig.model_validate({})
    data: DataConfig
    strategy: StrategyConfig = StrategyConfig.model_validate({})
    execution: ExecutionConfig = ExecutionConfig.model_validate({})
    ui: UiConfig = UiConfig.model_validate({})
    runtime: RuntimeConfig = RuntimeConfig.model_validate({})
    secrets: SecretConfig = SecretConfig()

    @model_validator(mode="before")
    @classmethod
    def inject_derived_position_count(cls, value: Any) -> Any:
        """Set an omitted position count to the smaller of five and universe size."""
        if not isinstance(value, Mapping):
            return value

        values = dict(value)
        data_value = values.get("data")
        if isinstance(data_value, DataConfig):
            universe: object = data_value.universe
        elif isinstance(data_value, Mapping):
            universe = data_value.get("universe", DEFAULT_UNIVERSE)
        else:
            universe = DEFAULT_UNIVERSE
        default_count = _default_position_count(universe)

        strategy_value = values.get("strategy")
        if strategy_value is None and "strategy" not in values:
            values["strategy"] = {"position_count": default_count}
        elif (
            isinstance(strategy_value, Mapping)
            and "position_count" not in strategy_value
        ):
            strategy = dict(strategy_value)
            strategy["position_count"] = default_count
            values["strategy"] = strategy
        return values

    @model_validator(mode="after")
    def validate_position_count(self) -> ResolvedConfig:
        universe_count = len(self.data.universe)
        # When the universe itself is invalid, report that field without adding
        # a derived position-count error that obscures the primary failure.
        if universe_count > 0 and self.strategy.position_count > universe_count:
            raise ValueError(
                "strategy.position_count must not exceed data.universe length "
                f"({universe_count})"
            )
        return self


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_INITIAL_EQUITY_USD",
    "DEFAULT_UNIVERSE",
    "DataConfig",
    "DateRangeConfig",
    "ExecutionConfig",
    "FrozenConfigModel",
    "PathConfig",
    "REDACTION_MARKER",
    "ResolvedConfig",
    "RetryPolicyConfig",
    "RuntimeConfig",
    "SecretConfig",
    "SecretValue",
    "StrategyConfig",
    "UiConfig",
    "UnresolvedSecret",
]
