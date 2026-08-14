"""Focused validation tests for frozen Phase 1 Pydantic configuration models."""

# ruff: noqa: E501, I001

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from quant_research_platform.config.models import (
    DEFAULT_INITIAL_EQUITY_USD,
    DEFAULT_UNIVERSE,
    DataConfig,
    DateRangeConfig,
    ExecutionConfig,
    PathConfig,
    ResolvedConfig,
    RetryPolicyConfig,
    RuntimeConfig,
    SecretConfig,
    StrategyConfig,
    UiConfig,
)


REQUESTED_RANGE = {"start": "2015-01-01", "end": "2024-12-31"}


def _resolved_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "paths": {},
        "data": {"requested_range": REQUESTED_RANGE},
    }
    payload.update(overrides)
    return payload


def _resolved_config(**overrides: Any) -> ResolvedConfig:
    return ResolvedConfig.model_validate(_resolved_payload(**overrides))


def test_documented_defaults_include_fixed_equity_and_spy_benchmark() -> None:
    config = _resolved_config()

    assert config.paths == PathConfig(
        data_root=Path("data"),
        artifact_root=Path("data/artifacts"),
        metadata_db=Path("data/metadata.duckdb"),
        mlflow_db=Path("data/mlflow.db"),
        local_secrets_file=Path("config/secrets.local.yaml"),
    )
    assert config.retry == RetryPolicyConfig(
        attempts=3,
        initial_delay_seconds=Decimal("1"),
        max_delay_seconds=Decimal("8"),
        backoff_multiplier=Decimal("2.0"),
    )
    assert config.data.universe == DEFAULT_UNIVERSE
    assert config.data.requested_range == DateRangeConfig(
        start=date(2015, 1, 1), end=date(2024, 12, 31)
    )
    assert config.data.benchmark == "SPY"
    assert config.data.provider == "yfinance"
    assert config.data.batch_size == 5
    assert config.data.staleness_sessions == 1
    assert config.data.revision_overlap_sessions == 5
    assert config.data.write_chunk_rows == 50_000
    assert config.strategy == StrategyConfig(position_count=5)
    assert config.execution.initial_equity_usd == DEFAULT_INITIAL_EQUITY_USD
    assert config.execution.commission_bps == Decimal("5")
    assert config.execution.slippage_bps == Decimal("10")
    assert config.ui.page_size == 100
    assert config.runtime.deterministic_seed == 0
    assert config.secrets == SecretConfig()


def test_all_documented_inclusive_bounds_are_accepted() -> None:
    requested_range = DateRangeConfig(start=date(2024, 1, 1), end=date(2024, 1, 1))
    universe = tuple(f"SYM{index}" for index in range(25))

    lower = ResolvedConfig(
        paths=PathConfig(),
        retry=RetryPolicyConfig(
            attempts=1,
            initial_delay_seconds=Decimal("0"),
            max_delay_seconds=Decimal("0"),
            backoff_multiplier=Decimal("1"),
        ),
        data=DataConfig(
            universe=(" aapl ",),
            requested_range=requested_range,
            batch_size=1,
            staleness_sessions=0,
            revision_overlap_sessions=0,
            write_chunk_rows=1,
        ),
        strategy=StrategyConfig(position_count=1),
        execution=ExecutionConfig(
            initial_equity_usd=Decimal("100000"),
            commission_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        ui=UiConfig(page_size=1),
        runtime=RuntimeConfig(deterministic_seed=0),
    )
    upper = ResolvedConfig(
        paths=PathConfig(),
        retry=RetryPolicyConfig(
            attempts=5,
            initial_delay_seconds=Decimal("60"),
            max_delay_seconds=Decimal("60"),
            backoff_multiplier=Decimal("4"),
        ),
        data=DataConfig(
            universe=universe,
            requested_range=requested_range,
            batch_size=10,
            staleness_sessions=252,
            revision_overlap_sessions=252,
            write_chunk_rows=100_000,
        ),
        strategy=StrategyConfig(position_count=25),
        execution=ExecutionConfig(
            initial_equity_usd=Decimal("100000"),
            commission_bps=Decimal("123456789.123"),
            slippage_bps=Decimal("987654321.987"),
        ),
        ui=UiConfig(page_size=100),
        runtime=RuntimeConfig(deterministic_seed=4_294_967_295),
    )

    assert lower.data.universe == ("AAPL",)
    assert lower.strategy.position_count == 1
    assert upper.data.universe == universe
    assert upper.strategy.position_count == 25
    assert upper.execution.commission_bps == Decimal("123456789.123")
    assert upper.execution.slippage_bps == Decimal("987654321.987")


def test_universe_is_normalized_ordered_unique_and_derives_position_count() -> None:
    config = _resolved_config(
        data={
            "requested_range": REQUESTED_RANGE,
            "universe": [" msft ", "aapl", "JPM"],
        }
    )

    assert config.data.universe == ("MSFT", "AAPL", "JPM")
    assert config.strategy.position_count == 3

    explicit = _resolved_config(
        data={"requested_range": REQUESTED_RANGE, "universe": ["AAPL", "MSFT"]},
        strategy={"position_count": 1},
    )
    assert explicit.strategy.position_count == 1


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: DataConfig(
                universe=(" ",), requested_range=DateRangeConfig(**REQUESTED_RANGE)
            ),
            r"universe\[0\]",
        ),
        (
            lambda: DataConfig(
                universe=(" aapl ", "AAPL"),
                requested_range=DateRangeConfig(**REQUESTED_RANGE),
            ),
            r"duplicates normalized symbol",
        ),
        (
            lambda: DateRangeConfig(start=date(2024, 1, 2), end=date(2024, 1, 1)),
            r"start must be no later",
        ),
        (
            lambda: RetryPolicyConfig(
                initial_delay_seconds=Decimal("2"), max_delay_seconds=Decimal("1")
            ),
            r"max_delay_seconds must be at least",
        ),
        (lambda: ExecutionConfig(commission_bps=Decimal("NaN")), r"finite|greater"),
        (lambda: ExecutionConfig(slippage_bps=Decimal("Infinity")), r"finite|less"),
        (lambda: ExecutionConfig(commission_bps=Decimal("-0.01")), r"greater than or equal"),
        (lambda: ExecutionConfig(initial_equity_usd=Decimal("99999.99")), r"fixed"),
        (lambda: DataConfig(requested_range=DateRangeConfig(**REQUESTED_RANGE), benchmark="QQQ"), r"SPY"),
        (lambda: DataConfig(requested_range=DateRangeConfig(**REQUESTED_RANGE), batch_size=11), r"less than or equal"),
        (lambda: DataConfig(requested_range=DateRangeConfig(**REQUESTED_RANGE), staleness_sessions=253), r"less than or equal"),
        (
            lambda: DataConfig(
                requested_range=DateRangeConfig(**REQUESTED_RANGE),
                revision_overlap_sessions=253,
            ),
            r"less than or equal",
        ),
        (
            lambda: DataConfig(
                requested_range=DateRangeConfig(**REQUESTED_RANGE), write_chunk_rows=100_001
            ),
            r"less than or equal",
        ),
        (lambda: UiConfig(page_size=101), r"less than or equal"),
        (lambda: RuntimeConfig(deterministic_seed=4_294_967_296), r"less than or equal"),
    ],
)
def test_invalid_values_and_cross_field_relationships_are_rejected(
    factory: Any, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        factory()


def test_position_count_must_not_exceed_normalized_universe_size() -> None:
    with pytest.raises(ValidationError, match=r"position_count must not exceed"):
        _resolved_config(
            data={"requested_range": REQUESTED_RANGE, "universe": ["AAPL", "MSFT"]},
            strategy={"position_count": 3},
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DateRangeConfig(start=date(2024, 1, 1), end=date(2024, 1, 2), extra=True),
        lambda: PathConfig(extra=True),
        lambda: RetryPolicyConfig(extra=True),
        lambda: DataConfig(requested_range=DateRangeConfig(**REQUESTED_RANGE), extra=True),
        lambda: StrategyConfig(extra=True),
        lambda: ExecutionConfig(extra=True),
        lambda: UiConfig(extra=True),
        lambda: RuntimeConfig(extra=True),
        lambda: SecretConfig(extra=True),
        lambda: _resolved_config(extra=True),
        lambda: _resolved_config(paths={"extra": True}),
        lambda: _resolved_config(data={"requested_range": REQUESTED_RANGE, "extra": True}),
    ],
)
def test_extra_keys_are_forbidden_at_every_configuration_level(factory: Any) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        factory()


def test_models_are_frozen_and_secret_values_remain_secret_strings() -> None:
    config = _resolved_config(secrets={"https_proxy": "https://user:password@proxy"})

    assert config.secrets.https_proxy is not None
    assert config.secrets.https_proxy.get_secret_value() == "https://user:password@proxy"
    with pytest.raises(ValidationError, match="frozen"):
        config.ui.page_size = 1
