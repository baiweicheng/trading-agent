"""Property tests for configuration source precedence and validation gating."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from hypothesis import given, settings, strategies as st
from ruamel.yaml import YAML

from quant_research_platform.config import loader as configuration_loader
from quant_research_platform.config.loader import ConfigurationManager
from quant_research_platform.domain.errors import Err, ErrorCategory, Ok, Result


LeafPath = tuple[str, ...]


@dataclass(frozen=True)
class Leaf:
    """One generated configuration leaf and its explicit environment name."""

    path: LeafPath
    environment_name: str


@dataclass(frozen=True)
class InvalidLeaf:
    """A deliberately invalid but safe-YAML-representable configuration value."""

    path: LeafPath
    yaml_value: object
    environment_value: str
    environment_supported: bool = True


@dataclass(frozen=True)
class ConfigurationCase:
    """A complete three-source configuration scenario and expected failures."""

    defaults: dict[str, object]
    yaml_leaf_values: dict[LeafPath, object]
    environment_leaf_values: dict[LeafPath, object]
    mode: str
    invalid_paths: tuple[LeafPath, ...]
    unknown_suffix: int | None


LEAVES = (
    Leaf(("paths", "data_root"), "QRP_PATHS__DATA_ROOT"),
    Leaf(("paths", "artifact_root"), "QRP_PATHS__ARTIFACT_ROOT"),
    Leaf(("paths", "metadata_db"), "QRP_PATHS__METADATA_DB"),
    Leaf(("paths", "mlflow_db"), "QRP_PATHS__MLFLOW_DB"),
    Leaf(("retry", "attempts"), "QRP_RETRY__ATTEMPTS"),
    Leaf(("retry", "initial_delay_seconds"), "QRP_RETRY__INITIAL_DELAY_SECONDS"),
    Leaf(("retry", "max_delay_seconds"), "QRP_RETRY__MAX_DELAY_SECONDS"),
    Leaf(("retry", "backoff_multiplier"), "QRP_RETRY__BACKOFF_MULTIPLIER"),
    Leaf(("data", "universe"), "QRP_DATA__UNIVERSE"),
    Leaf(("data", "requested_range", "start"), "QRP_DATA__REQUESTED_RANGE__START"),
    Leaf(("data", "requested_range", "end"), "QRP_DATA__REQUESTED_RANGE__END"),
    Leaf(("data", "benchmark"), "QRP_DATA__BENCHMARK"),
    Leaf(("data", "provider"), "QRP_DATA__PROVIDER"),
    Leaf(("data", "batch_size"), "QRP_DATA__BATCH_SIZE"),
    Leaf(("data", "staleness_sessions"), "QRP_DATA__STALENESS_SESSIONS"),
    Leaf(
        ("data", "revision_overlap_sessions"),
        "QRP_DATA__REVISION_OVERLAP_SESSIONS",
    ),
    Leaf(("data", "write_chunk_rows"), "QRP_DATA__WRITE_CHUNK_ROWS"),
    Leaf(("strategy", "identifier"), "QRP_STRATEGY__IDENTIFIER"),
    Leaf(("strategy", "position_count"), "QRP_STRATEGY__POSITION_COUNT"),
    Leaf(
        ("strategy", "long_lookback_sessions"),
        "QRP_STRATEGY__LONG_LOOKBACK_SESSIONS",
    ),
    Leaf(
        ("strategy", "skip_recent_sessions"),
        "QRP_STRATEGY__SKIP_RECENT_SESSIONS",
    ),
    Leaf(
        ("execution", "initial_equity_usd"),
        "QRP_EXECUTION__INITIAL_EQUITY_USD",
    ),
    Leaf(("execution", "commission_bps"), "QRP_EXECUTION__COMMISSION_BPS"),
    Leaf(("execution", "slippage_bps"), "QRP_EXECUTION__SLIPPAGE_BPS"),
    Leaf(("ui", "page_size"), "QRP_UI__PAGE_SIZE"),
    Leaf(("runtime", "deterministic_seed"), "QRP_RUNTIME__DETERMINISTIC_SEED"),
)

LEAF_BY_PATH = {leaf.path: leaf for leaf in LEAVES}
DATE_RANGE_PATHS = {
    ("data", "requested_range", "start"),
    ("data", "requested_range", "end"),
}
SIBLING_YAML_LEAF = LEAF_BY_PATH[("retry", "attempts")]
SIBLING_ENVIRONMENT_LEAF = LEAF_BY_PATH[("retry", "initial_delay_seconds")]

INVALID_LEAVES = (
    InvalidLeaf(("retry", "attempts"), [], "not-an-integer"),
    InvalidLeaf(("retry", "initial_delay_seconds"), -1, "-1"),
    InvalidLeaf(("retry", "max_delay_seconds"), 61, "61"),
    InvalidLeaf(("retry", "backoff_multiplier"), 0, "0"),
    InvalidLeaf(("data", "universe"), [], "", environment_supported=False),
    InvalidLeaf(("data", "requested_range", "start"), "not-a-date", "not-a-date"),
    InvalidLeaf(("data", "requested_range", "end"), "not-a-date", "not-a-date"),
    InvalidLeaf(("data", "provider"), "not-yfinance", "not-yfinance"),
    InvalidLeaf(("data", "batch_size"), 11, "11"),
    InvalidLeaf(("data", "staleness_sessions"), -1, "-1"),
    InvalidLeaf(("data", "revision_overlap_sessions"), 253, "253"),
    InvalidLeaf(("data", "write_chunk_rows"), 100_001, "100001"),
    InvalidLeaf(("strategy", "identifier"), "another_strategy", "another_strategy"),
    InvalidLeaf(("strategy", "position_count"), 0, "0"),
    InvalidLeaf(("strategy", "long_lookback_sessions"), 251, "251"),
    InvalidLeaf(("strategy", "skip_recent_sessions"), 20, "20"),
    InvalidLeaf(("execution", "commission_bps"), -1, "-1"),
    InvalidLeaf(("execution", "slippage_bps"), -1, "-1"),
    InvalidLeaf(("ui", "page_size"), 101, "101"),
    InvalidLeaf(("runtime", "deterministic_seed"), -1, "-1"),
)

TOP_LEVEL_ORDER = {
    "paths": 0,
    "retry": 1,
    "data": 2,
    "strategy": 3,
    "execution": 4,
    "ui": 5,
    "runtime": 6,
    "secrets": 7,
}
CHILD_ORDER = {
    "paths": {
        "data_root": 0,
        "artifact_root": 1,
        "metadata_db": 2,
        "mlflow_db": 3,
        "local_secrets_file": 4,
    },
    "retry": {
        "attempts": 0,
        "initial_delay_seconds": 1,
        "max_delay_seconds": 2,
        "backoff_multiplier": 3,
    },
    "data": {
        "universe": 0,
        "requested_range": 1,
        "benchmark": 2,
        "provider": 3,
        "batch_size": 4,
        "staleness_sessions": 5,
        "revision_overlap_sessions": 6,
        "write_chunk_rows": 7,
    },
    "strategy": {
        "identifier": 0,
        "position_count": 1,
        "long_lookback_sessions": 2,
        "skip_recent_sessions": 3,
    },
    "execution": {
        "initial_equity_usd": 0,
        "commission_bps": 1,
        "slippage_bps": 2,
    },
    "ui": {"page_size": 0},
    "runtime": {"deterministic_seed": 0},
}
DATE_RANGE_ORDER = {"start": 0, "end": 1}


def _base_defaults(*, include_requested_range: bool) -> dict[str, object]:
    """Return safe, complete defaults that may be varied by generated leaves."""

    data: dict[str, object] = {
        "universe": ["AAPL", "JPM", "MSFT", "PG", "XOM"],
        "benchmark": "SPY",
        "provider": "yfinance",
        "batch_size": 5,
        "staleness_sessions": 1,
        "revision_overlap_sessions": 5,
        "write_chunk_rows": 50_000,
    }
    if include_requested_range:
        data["requested_range"] = {"start": "2020-01-01", "end": "2025-12-31"}

    return {
        "paths": {
            "data_root": "data",
            "artifact_root": "data/artifacts",
            "metadata_db": "data/metadata.duckdb",
            "mlflow_db": "data/mlflow.db",
            "local_secrets_file": "config/secrets.local.yaml",
        },
        "retry": {
            "attempts": 3,
            "initial_delay_seconds": 1,
            "max_delay_seconds": 8,
            "backoff_multiplier": 2,
        },
        "data": data,
        "strategy": {
            "identifier": "monthly_momentum_v1",
            "long_lookback_sessions": 252,
            "skip_recent_sessions": 21,
        },
        "execution": {
            "initial_equity_usd": "100000",
            "commission_bps": 5,
            "slippage_bps": 10,
        },
        "ui": {"page_size": 100},
        "runtime": {"deterministic_seed": 0},
        "secrets": {"http_proxy": None, "https_proxy": None},
    }


def _safe_value_strategy(leaf: Leaf) -> st.SearchStrategy[object]:
    """Generate values that remain valid independently of source selection."""

    path = leaf.path
    if path[0] == "paths":
        return st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
            min_size=1,
            max_size=8,
        ).map(lambda segment: f"state/{segment}")
    if path == ("retry", "attempts"):
        return st.integers(min_value=1, max_value=5)
    if path == ("retry", "initial_delay_seconds"):
        return st.integers(min_value=0, max_value=5)
    if path == ("retry", "max_delay_seconds"):
        return st.integers(min_value=10, max_value=60)
    if path == ("retry", "backoff_multiplier"):
        return st.integers(min_value=1, max_value=4)
    if path == ("data", "universe"):
        return st.lists(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=4),
            min_size=5,
            max_size=5,
            unique=True,
        ).map(
            lambda symbols: [
                f" {symbol.lower()} " if index % 2 else symbol
                for index, symbol in enumerate(symbols)
            ]
        )
    if path == ("data", "requested_range", "start"):
        return st.dates(min_value=date(2020, 1, 1), max_value=date(2024, 12, 31)).map(
            str
        )
    if path == ("data", "requested_range", "end"):
        return st.dates(min_value=date(2025, 1, 1), max_value=date(2029, 12, 31)).map(
            str
        )
    if path in {
        ("data", "benchmark"),
        ("data", "provider"),
        ("strategy", "identifier"),
    }:
        return st.just(
            {
                ("data", "benchmark"): "SPY",
                ("data", "provider"): "yfinance",
                ("strategy", "identifier"): "monthly_momentum_v1",
            }[path]
        )
    if path == ("data", "batch_size"):
        return st.integers(min_value=1, max_value=10)
    if path in {
        ("data", "staleness_sessions"),
        ("data", "revision_overlap_sessions"),
    }:
        return st.integers(min_value=0, max_value=252)
    if path == ("data", "write_chunk_rows"):
        return st.integers(min_value=1, max_value=100_000)
    if path == ("strategy", "position_count"):
        return st.integers(min_value=1, max_value=5)
    if path == ("strategy", "long_lookback_sessions"):
        return st.just(252)
    if path == ("strategy", "skip_recent_sessions"):
        return st.just(21)
    if path == ("execution", "initial_equity_usd"):
        return st.just("100000")
    if path in {
        ("execution", "commission_bps"),
        ("execution", "slippage_bps"),
    }:
        return st.integers(min_value=0, max_value=10_000)
    if path == ("ui", "page_size"):
        return st.integers(min_value=1, max_value=100)
    if path == ("runtime", "deterministic_seed"):
        return st.integers(min_value=0, max_value=4_294_967_295)
    raise AssertionError(f"unhandled safe leaf: {path}")


def _assign_leaf(target: dict[str, object], path: LeafPath, value: object) -> None:
    current = target
    for component in path[:-1]:
        child = current.get(component)
        if not isinstance(child, dict):
            child = {}
            current[component] = child
        current = child
    current[path[-1]] = value


def _leaf_map(values: Mapping[LeafPath, object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for path, value in values.items():
        _assign_leaf(mapped, path, value)
    return mapped


def _right_biased_merge(
    lower_precedence: Mapping[str, object], higher_precedence: Mapping[str, object]
) -> dict[str, object]:
    """Small independent reference merge for nested configuration leaf maps."""

    merged = deepcopy(dict(lower_precedence))
    for key, higher_value in higher_precedence.items():
        lower_value = merged.get(key)
        if isinstance(lower_value, Mapping) and isinstance(higher_value, Mapping):
            merged[key] = _right_biased_merge(lower_value, higher_value)
        else:
            merged[key] = deepcopy(higher_value)
    return merged


def _environment_raw_value(leaf: Leaf, value: object) -> str:
    if leaf.path == ("data", "universe"):
        assert isinstance(value, list)
        return ",".join(value)
    return str(value)


def _environment_reference_map(
    values: Mapping[LeafPath, object],
) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for path, value in values.items():
        if path == ("data", "universe"):
            assert isinstance(value, list)
            converted: object = tuple(symbol.strip() for symbol in value)
        else:
            converted = str(value)
        _assign_leaf(mapped, path, converted)
    return mapped


def _environment_values(values: Mapping[LeafPath, object]) -> dict[str, str]:
    return {
        LEAF_BY_PATH[path].environment_name: _environment_raw_value(
            LEAF_BY_PATH[path], value
        )
        for path, value in values.items()
    }


def _yaml_document(values: Mapping[str, object]) -> str:
    yaml = YAML(typ="safe", pure=True)
    stream = StringIO()
    yaml.dump(dict(values), stream)
    return stream.getvalue()


def _reference_normalize(
    merged: Mapping[str, object], project_root: Path
) -> dict[LeafPath, object]:
    """Normalize generated values without invoking the Pydantic configuration model."""

    paths = merged["paths"]
    retry = merged["retry"]
    data = merged["data"]
    strategy = merged["strategy"]
    execution = merged["execution"]
    ui = merged["ui"]
    runtime = merged["runtime"]
    assert isinstance(paths, Mapping)
    assert isinstance(retry, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(strategy, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(ui, Mapping)
    assert isinstance(runtime, Mapping)

    universe_raw = data["universe"]
    assert isinstance(universe_raw, (list, tuple))
    universe = tuple(str(symbol).strip().upper() for symbol in universe_raw)
    range_raw = data["requested_range"]
    assert isinstance(range_raw, Mapping)

    normalized: dict[LeafPath, object] = {
        ("paths", "data_root"): (project_root / str(paths["data_root"])).resolve(
            strict=False
        ),
        ("paths", "artifact_root"): (
            project_root / str(paths["artifact_root"])
        ).resolve(strict=False),
        ("paths", "metadata_db"): (
            project_root / str(paths["metadata_db"])
        ).resolve(strict=False),
        ("paths", "mlflow_db"): (project_root / str(paths["mlflow_db"])).resolve(
            strict=False
        ),
        ("retry", "attempts"): int(str(retry["attempts"])),
        ("retry", "initial_delay_seconds"): Decimal(
            str(retry["initial_delay_seconds"])
        ),
        ("retry", "max_delay_seconds"): Decimal(str(retry["max_delay_seconds"])),
        ("retry", "backoff_multiplier"): Decimal(str(retry["backoff_multiplier"])),
        ("data", "universe"): universe,
        ("data", "requested_range", "start"): date.fromisoformat(
            str(range_raw["start"])
        ),
        ("data", "requested_range", "end"): date.fromisoformat(
            str(range_raw["end"])
        ),
        ("data", "benchmark"): str(data["benchmark"]),
        ("data", "provider"): str(data["provider"]),
        ("data", "batch_size"): int(str(data["batch_size"])),
        ("data", "staleness_sessions"): int(str(data["staleness_sessions"])),
        ("data", "revision_overlap_sessions"): int(
            str(data["revision_overlap_sessions"])
        ),
        ("data", "write_chunk_rows"): int(str(data["write_chunk_rows"])),
        ("strategy", "identifier"): str(strategy["identifier"]),
        ("strategy", "position_count"): int(
            str(strategy.get("position_count", min(5, len(universe))))
        ),
        ("strategy", "long_lookback_sessions"): int(
            str(strategy["long_lookback_sessions"])
        ),
        ("strategy", "skip_recent_sessions"): int(
            str(strategy["skip_recent_sessions"])
        ),
        ("execution", "initial_equity_usd"): Decimal(
            str(execution["initial_equity_usd"])
        ),
        ("execution", "commission_bps"): Decimal(
            str(execution["commission_bps"])
        ),
        ("execution", "slippage_bps"): Decimal(
            str(execution["slippage_bps"])
        ),
        ("ui", "page_size"): int(str(ui["page_size"])),
        ("runtime", "deterministic_seed"): int(
            str(runtime["deterministic_seed"])
        ),
    }
    return normalized


def _actual_values(config: object) -> dict[LeafPath, object]:
    """Project the same non-secret leaves from the real resolved configuration."""

    return {
        ("paths", "data_root"): config.paths.data_root,
        ("paths", "artifact_root"): config.paths.artifact_root,
        ("paths", "metadata_db"): config.paths.metadata_db,
        ("paths", "mlflow_db"): config.paths.mlflow_db,
        ("retry", "attempts"): config.retry.attempts,
        ("retry", "initial_delay_seconds"): config.retry.initial_delay_seconds,
        ("retry", "max_delay_seconds"): config.retry.max_delay_seconds,
        ("retry", "backoff_multiplier"): config.retry.backoff_multiplier,
        ("data", "universe"): config.data.universe,
        ("data", "requested_range", "start"): config.data.requested_range.start,
        ("data", "requested_range", "end"): config.data.requested_range.end,
        ("data", "benchmark"): config.data.benchmark,
        ("data", "provider"): config.data.provider,
        ("data", "batch_size"): config.data.batch_size,
        ("data", "staleness_sessions"): config.data.staleness_sessions,
        ("data", "revision_overlap_sessions"): config.data.revision_overlap_sessions,
        ("data", "write_chunk_rows"): config.data.write_chunk_rows,
        ("strategy", "identifier"): config.strategy.identifier,
        ("strategy", "position_count"): config.strategy.position_count,
        ("strategy", "long_lookback_sessions"): config.strategy.long_lookback_sessions,
        ("strategy", "skip_recent_sessions"): config.strategy.skip_recent_sessions,
        ("execution", "initial_equity_usd"): config.execution.initial_equity_usd,
        ("execution", "commission_bps"): config.execution.commission_bps,
        ("execution", "slippage_bps"): config.execution.slippage_bps,
        ("ui", "page_size"): config.ui.page_size,
        ("runtime", "deterministic_seed"): config.runtime.deterministic_seed,
    }


def _schema_order(path: str) -> tuple[int, ...]:
    """Independent schema-order key for known and unknown generated error paths."""

    parts = path.split(".")
    if not parts or parts[0] not in TOP_LEVEL_ORDER:
        return (99_999,)

    top_level = parts[0]
    order = [TOP_LEVEL_ORDER[top_level]]
    if len(parts) == 1:
        return tuple(order)

    child = parts[1].split("[", 1)[0]
    child_order = CHILD_ORDER.get(top_level, {}).get(child)
    if child_order is None:
        return (*order, 99_999)
    order.append(child_order)

    if top_level == "data" and child == "requested_range" and len(parts) > 2:
        date_component = parts[2].split("[", 1)[0]
        order.append(DATE_RANGE_ORDER.get(date_component, 99_999))
    return tuple(order)


def _expected_errors(case: ConfigurationCase) -> list[tuple[str, ErrorCategory]]:
    expected = [
        (".".join(path), ErrorCategory.CONFIGURATION_INVALID_VALUE)
        for path in case.invalid_paths
    ]
    if case.mode == "path":
        expected.append(("paths.data_root", ErrorCategory.CONFIGURATION_INVALID_VALUE))
    if case.mode == "missing":
        expected.append(("data.requested_range", ErrorCategory.CONFIGURATION_INVALID_VALUE))
    if case.unknown_suffix is not None:
        expected.extend(
            (
                (
                    f"data.unmapped_{case.unknown_suffix}",
                    ErrorCategory.CONFIGURATION_UNKNOWN_KEY,
                ),
                (
                    f"QRP_UNMAPPED_{case.unknown_suffix}",
                    ErrorCategory.CONFIGURATION_UNKNOWN_KEY,
                ),
            )
        )
    return sorted(expected, key=lambda error: (_schema_order(error[0]), error[0]))


def _draw_entries(
    draw: st.DrawFn,
    leaves: tuple[Leaf, ...],
    *,
    max_size: int,
) -> dict[LeafPath, object]:
    selected = draw(
        st.lists(
            st.sampled_from(leaves),
            min_size=0,
            max_size=max_size,
            unique_by=lambda leaf: leaf.path,
        )
    )
    return {leaf.path: draw(_safe_value_strategy(leaf)) for leaf in selected}


@st.composite
def configuration_cases(draw: st.DrawFn) -> ConfigurationCase:
    """Generate three-source leaf maps plus controlled invalid variants."""

    mode = draw(st.sampled_from(("valid", "schema", "path", "missing")))
    available_leaves = tuple(
        leaf for leaf in LEAVES if mode != "missing" or leaf.path not in DATE_RANGE_PATHS
    )
    precedence_leaves = tuple(
        leaf
        for leaf in available_leaves
        if leaf not in {SIBLING_YAML_LEAF, SIBLING_ENVIRONMENT_LEAF}
    )

    defaults = _base_defaults(include_requested_range=mode != "missing")
    default_values = _draw_entries(draw, available_leaves, max_size=8)
    yaml_leaf_values = _draw_entries(draw, available_leaves, max_size=8)
    environment_leaf_values = _draw_entries(draw, available_leaves, max_size=8)
    for path, value in default_values.items():
        _assign_leaf(defaults, path, value)

    shared_leaf = draw(st.sampled_from(precedence_leaves))
    _assign_leaf(defaults, shared_leaf.path, draw(_safe_value_strategy(shared_leaf)))
    yaml_leaf_values[shared_leaf.path] = draw(_safe_value_strategy(shared_leaf))
    environment_leaf_values[shared_leaf.path] = draw(
        _safe_value_strategy(shared_leaf)
    )

    # Always place independent siblings in different sources. This detects a
    # shallow merge that would drop a lower-precedence sibling mapping.
    yaml_leaf_values[SIBLING_YAML_LEAF.path] = draw(
        _safe_value_strategy(SIBLING_YAML_LEAF)
    )
    environment_leaf_values[SIBLING_ENVIRONMENT_LEAF.path] = draw(
        _safe_value_strategy(SIBLING_ENVIRONMENT_LEAF)
    )

    invalid_paths: tuple[LeafPath, ...] = ()
    if mode == "schema":
        invalid = draw(
            st.lists(
                st.sampled_from(INVALID_LEAVES),
                min_size=1,
                max_size=4,
                unique_by=lambda item: item.path,
            )
        )
        invalid_paths = tuple(item.path for item in invalid)
        for item in invalid:
            source = (
                "yaml"
                if not item.environment_supported
                else draw(st.sampled_from(("yaml", "environment")))
            )
            if source == "yaml":
                yaml_leaf_values[item.path] = item.yaml_value
                # Ensure the invalid YAML leaf remains effective rather than
                # being hidden by a higher-precedence environment override.
                environment_leaf_values.pop(item.path, None)
            else:
                environment_leaf_values[item.path] = item.environment_value
                # Keep the generated invalid value at the highest precedence
                # source so the expected validation error is deterministic.
                yaml_leaf_values.pop(item.path, None)
    elif mode == "path":
        path_value = f"../outside-{draw(st.integers(min_value=0, max_value=9999))}"
        if draw(st.booleans()):
            yaml_leaf_values[("paths", "data_root")] = path_value
            environment_leaf_values.pop(("paths", "data_root"), None)
        else:
            environment_leaf_values[("paths", "data_root")] = path_value
            yaml_leaf_values.pop(("paths", "data_root"), None)

    unknown_suffix = None
    if mode != "path" and draw(st.booleans()):
        unknown_suffix = draw(st.integers(min_value=0, max_value=9999))

    return ConfigurationCase(
        defaults=defaults,
        yaml_leaf_values=yaml_leaf_values,
        environment_leaf_values=environment_leaf_values,
        mode=mode,
        invalid_paths=invalid_paths,
        unknown_suffix=unknown_suffix,
    )


# Feature: quantitative-research-platform, Property 1: Leaf-wise configuration resolution and validation gate
# Validates: Requirements 2.6, 2.7, 2.9–2.24, 2.26–2.31, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.56–2.58.
@settings(max_examples=100, deadline=None)
@given(case=configuration_cases())
def test_leafwise_configuration_resolution_and_validation_gate(
    case: ConfigurationCase,
) -> None:
    """Resolution follows right-biased leaf precedence or stops before work begins."""

    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="qrp-property-01-") as temporary_directory:
        project_root = Path(temporary_directory) / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text(
            "[project]\nname = 'property-test'\n"
        )
        manager = ConfigurationManager(
            project_anchor=project_root / "src" / "package" / "entry.py"
        )

        yaml_values = _leaf_map(case.yaml_leaf_values)
        environment = _environment_values(case.environment_leaf_values)
        if case.unknown_suffix is not None:
            data_values = yaml_values.setdefault("data", {})
            assert isinstance(data_values, dict)
            data_values[f"unmapped_{case.unknown_suffix}"] = "unexpected"
            environment[f"QRP_UNMAPPED_{case.unknown_suffix}"] = "unexpected"

        calls: list[str] = []
        observed_configs: list[object] = []

        def downstream(config: object) -> Result[str]:
            calls.append("called")
            observed_configs.append(config)
            return Ok("called")

        with patch.object(configuration_loader, "DOCUMENTED_DEFAULTS", case.defaults):
            result = manager.resolve_then(
                _yaml_document(yaml_values), environment, downstream
            )

        expected_errors = _expected_errors(case)
        if expected_errors:
            assert isinstance(result, Err)
            assert calls == []
            assert observed_configs == []
            actual_errors = [
                (error.field_path, error.category) for error in result.errors
            ]
            assert actual_errors == expected_errors
            assert len(result.errors) == len(expected_errors)

            invalid_schema_paths = {".".join(path) for path in case.invalid_paths}
            for error in result.errors:
                if error.field_path in invalid_schema_paths:
                    assert "Accepted type:" in error.message
                if error.field_path == "data.requested_range" and case.mode == "missing":
                    assert "missing" in error.message
                if error.field_path == "paths.data_root" and case.mode == "path":
                    assert "Project_Root boundary" in error.message
            return

        assert isinstance(result, Ok)
        assert result.value == "called"
        assert calls == ["called"]
        assert len(observed_configs) == 1

        reference_environment = _environment_reference_map(case.environment_leaf_values)
        reference_merged = _right_biased_merge(
            case.defaults, _leaf_map(case.yaml_leaf_values)
        )
        reference_merged = _right_biased_merge(
            reference_merged, reference_environment
        )
        assert _actual_values(observed_configs[0]) == _reference_normalize(
            reference_merged, project_root
        )
