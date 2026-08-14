"""Representative configuration parser, precedence, path, and redaction examples."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_research_platform.config.loader import ConfigurationManager
from quant_research_platform.config.models import (
    DEFAULT_INITIAL_EQUITY_USD,
    DEFAULT_UNIVERSE,
    UnresolvedSecret,
)
from quant_research_platform.config.serializer import (
    ConfigurationSerializer,
    SecretPresence,
    non_secret_config,
)
from quant_research_platform.domain.errors import Err, ErrorCategory, Ok

REQUESTED_RANGE_YAML = """
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
"""


def _manager(tmp_path: Path) -> tuple[ConfigurationManager, Path]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    anchor = root / "src" / "package" / "entry.py"
    return ConfigurationManager(project_anchor=anchor), root


def _errors(result: object) -> tuple[Any, ...]:
    assert isinstance(result, Err)
    return result.errors


def _error_text(errors: tuple[Any, ...]) -> str:
    return "\n".join(error.format_for_display() for error in errors)


def test_parser_examples_report_locations_safe_tags_and_mapping_roots(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    secret = "https://user:unmistakable-secret@proxy.invalid"
    environment = {"QRP_SECRETS__HTTPS_PROXY": secret}

    malformed = _errors(manager.resolve("data:\n  requested_range: [", environment))
    assert malformed[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "line" in malformed[0].message
    assert "column" in malformed[0].message

    unsafe_tag = _errors(
        manager.resolve("data: !!python/object {value: 1}", environment)
    )
    assert unsafe_tag[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "tags are not permitted" in unsafe_tag[0].message

    non_mapping = _errors(manager.resolve("- one\n- two", environment))
    assert non_mapping[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "list" in non_mapping[0].message
    assert "mapping" in non_mapping[0].message

    rendered = _error_text((*malformed, *unsafe_tag, *non_mapping))
    assert secret not in rendered
    assert secret.encode("utf-8") not in rendered.encode("utf-8")


def test_nested_duplicate_and_unknown_keys_include_actionable_paths(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)

    duplicate = _errors(
        manager.resolve(
            """
data:
  requested_range:
    start: 2020-01-01
    start: 2020-01-02
    end: 2020-12-31
""",
            {},
        )
    )
    assert duplicate[0].category is ErrorCategory.CONFIGURATION_DUPLICATE_KEY
    assert duplicate[0].field_path == "data.requested_range.start"
    assert "data.requested_range.start" in duplicate[0].message

    unknown = _errors(
        manager.resolve(
            """
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
    finish: 2021-01-01
""",
            {},
        )
    )
    assert unknown[0].category is ErrorCategory.CONFIGURATION_UNKNOWN_KEY
    assert unknown[0].field_path == "data.requested_range.finish"
    assert "start" in unknown[0].message
    assert "end" in unknown[0].message


def test_manager_defaults_and_leafwise_mapped_environment_precedence(
    tmp_path: Path,
) -> None:
    manager, root = _manager(tmp_path)

    defaults = manager.resolve(REQUESTED_RANGE_YAML, {})
    assert isinstance(defaults, Ok)
    default_config = defaults.value
    assert default_config.paths.data_root == root / "data"
    assert default_config.paths.artifact_root == root / "data" / "artifacts"
    assert default_config.paths.metadata_db == root / "data" / "metadata.duckdb"
    assert default_config.paths.mlflow_db == root / "data" / "mlflow.db"
    expected_secrets_file = root / "config" / "secrets.local.yaml"
    assert default_config.paths.local_secrets_file == expected_secrets_file
    assert default_config.retry.attempts == 3
    assert default_config.retry.initial_delay_seconds == Decimal("1")
    assert default_config.retry.max_delay_seconds == Decimal("8")
    assert default_config.retry.backoff_multiplier == Decimal("2.0")
    assert default_config.data.universe == DEFAULT_UNIVERSE
    assert default_config.data.requested_range.start == date(2020, 1, 1)
    assert default_config.data.requested_range.end == date(2020, 12, 31)
    assert default_config.data.benchmark == "SPY"
    assert default_config.data.provider == "yfinance"
    assert default_config.data.batch_size == 5
    assert default_config.data.staleness_sessions == 1
    assert default_config.data.revision_overlap_sessions == 5
    assert default_config.data.write_chunk_rows == 50_000
    assert default_config.strategy.position_count == 5
    assert default_config.execution.initial_equity_usd == DEFAULT_INITIAL_EQUITY_USD
    assert default_config.execution.commission_bps == Decimal("5")
    assert default_config.execution.slippage_bps == Decimal("10")
    assert default_config.ui.page_size == 100
    assert default_config.runtime.deterministic_seed == 0
    assert default_config.secrets.http_proxy is None
    assert default_config.secrets.https_proxy is None

    resolved = manager.resolve(
        """
retry:
  attempts: 2
  initial_delay_seconds: 3
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
  batch_size: 4
""",
        {
            "QRP_RETRY__ATTEMPTS": "5",
            "QRP_DATA__BATCH_SIZE": "7",
        },
    )
    assert isinstance(resolved, Ok)
    assert resolved.value.retry.attempts == 5
    assert resolved.value.retry.initial_delay_seconds == Decimal("3")
    assert resolved.value.retry.max_delay_seconds == Decimal("8")
    assert resolved.value.data.batch_size == 7
    assert resolved.value.data.staleness_sessions == 1


def test_unmapped_environment_error_is_actionable_and_secret_safe(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    secret = "https://user:secret-never-printed@proxy.invalid"

    errors = _errors(
        manager.resolve(
            REQUESTED_RANGE_YAML,
            {
                "QRP_SECRETS__HTTPS_PROXY": secret,
                "QRP_DATA__UNMAPPED_BATCH_LIMIT": "8",
            },
        )
    )
    assert len(errors) == 1
    assert errors[0].category is ErrorCategory.CONFIGURATION_UNKNOWN_KEY
    assert errors[0].field_path == "QRP_DATA__UNMAPPED_BATCH_LIMIT"
    assert "explicit configuration mapping" in errors[0].message
    rendered = _error_text(errors)
    assert secret not in rendered
    assert secret.encode("utf-8") not in rendered.encode("utf-8")


def test_schema_errors_are_in_field_order_and_name_each_failing_path(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    secret = "https://user:field-order-secret@proxy.invalid"

    errors = _errors(
        manager.resolve(
            """
paths:
  data_root: {}
retry:
  attempts: 0
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
  batch_size: 11
strategy:
  position_count: 0
execution:
  commission_bps: -1
ui:
  page_size: 101
runtime:
  deterministic_seed: 4294967296
""",
            {"QRP_SECRETS__HTTPS_PROXY": secret},
        )
    )

    expected_paths = [
        "paths.data_root",
        "retry.attempts",
        "data.batch_size",
        "strategy.position_count",
        "execution.commission_bps",
        "ui.page_size",
        "runtime.deterministic_seed",
    ]
    assert [error.field_path for error in errors] == expected_paths
    rendered = _error_text(errors)
    for field_path in expected_paths:
        assert field_path in rendered
    assert secret not in rendered


def test_ambiguous_roots_and_path_boundary_examples_use_temporary_roots(
    tmp_path: Path,
) -> None:
    outer = tmp_path / "outer"
    nested = outer / "nested"
    nested.mkdir(parents=True)
    (outer / "pyproject.toml").write_text("[project]\nname = 'outer'\n")
    (nested / "pyproject.toml").write_text("[project]\nname = 'nested'\n")
    ambiguous_manager = ConfigurationManager(
        project_anchor=nested / "src" / "package" / "entry.py"
    )

    ambiguous = _errors(ambiguous_manager.resolve(REQUESTED_RANGE_YAML, {}))
    assert ambiguous[0].field_path == "project_root"
    assert "multiple pyproject.toml ancestors" in ambiguous[0].message

    manager, root = _manager(tmp_path)
    escape_target = root.parent / "escaped"
    escaped_document = REQUESTED_RANGE_YAML + "\npaths:\n  data_root: ../escaped\n"
    escaped = _errors(manager.resolve(escaped_document, {}))
    assert escaped[0].field_path == "paths.data_root"
    assert "Project_Root boundary" in escaped[0].message
    assert not escape_target.exists()

    absolute = tmp_path / "explicit-absolute-artifacts"
    absolute_result = manager.resolve(
        REQUESTED_RANGE_YAML + f"\npaths:\n  artifact_root: {absolute}\n", {}
    )
    assert isinstance(absolute_result, Ok)
    assert absolute_result.value.paths.artifact_root == absolute.resolve()
    assert not absolute.exists()


def test_redaction_marker_reload_stays_unresolved_through_configuration_manager(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    secret = "https://user:marker-reload-secret@proxy.invalid"
    resolved = manager.resolve(
        REQUESTED_RANGE_YAML,
        {"QRP_SECRETS__HTTPS_PROXY": secret},
    )
    assert isinstance(resolved, Ok)

    canonical = ConfigurationSerializer().serialize(resolved.value)
    assert secret.encode("utf-8") not in canonical
    reloaded = manager.resolve(canonical, {})

    assert isinstance(reloaded, Ok)
    assert isinstance(reloaded.value.secrets.https_proxy, UnresolvedSecret)
    assert (
        non_secret_config(reloaded.value).secrets.https_proxy
        is SecretPresence.PRESENT_UNRESOLVED
    )
