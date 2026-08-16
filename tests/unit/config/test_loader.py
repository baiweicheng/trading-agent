"""Focused tests for safe configuration loading, precedence, and path boundaries."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant_research_platform.config.loader import ConfigurationManager
from quant_research_platform.config.project_root import (
    ProjectRootBoundaryError,
    resolve_project_root,
)
from quant_research_platform.domain.errors import Err, ErrorCategory, Ok, Result

VALID_YAML = """
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


def test_safe_parser_reports_syntax_root_duplicate_and_tag_diagnostics(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)

    syntax_errors = _errors(manager.resolve("data: [", {}))
    assert syntax_errors[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "line" in syntax_errors[0].message

    root_errors = _errors(manager.resolve("- not-a-mapping", {}))
    assert root_errors[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "list" in root_errors[0].message
    assert "mapping" in root_errors[0].message

    duplicate_errors = _errors(
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
    assert duplicate_errors[0].category is ErrorCategory.CONFIGURATION_DUPLICATE_KEY
    assert duplicate_errors[0].field_path == "data.requested_range.start"

    tag_errors = _errors(manager.resolve("data: !!str unsafe", {}))
    assert tag_errors[0].category is ErrorCategory.CONFIGURATION_SYNTAX
    assert "tags are not permitted" in tag_errors[0].message


def test_unknown_nested_keys_include_allowed_sibling_diagnostic(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    result = manager.resolve(
        """
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
  batch_size: 4
  unsupported_batch_limit: 9
""",
        {},
    )

    errors = _errors(result)
    assert len(errors) == 1
    assert errors[0].category is ErrorCategory.CONFIGURATION_UNKNOWN_KEY
    assert errors[0].field_path == "data.unsupported_batch_limit"
    assert "allowed sibling keys" in errors[0].message
    assert "batch_size" in errors[0].message


def test_nested_yaml_and_explicit_environment_values_merge_leaf_by_leaf(
    tmp_path: Path,
) -> None:
    manager, root = _manager(tmp_path)
    result = manager.resolve(
        """
retry:
  attempts: 2
  initial_delay_seconds: 3
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
  batch_size: 4
execution:
  commission_bps: 7
""",
        {
            "QRP_RETRY__ATTEMPTS": "5",
            "QRP_DATA__BATCH_SIZE": "7",
            "QRP_EXECUTION__COMMISSION_BPS": "12.5",
            "QRP_SECRETS__HTTPS_PROXY": "https://user:password@proxy.invalid",
            "UNRELATED_VARIABLE": "ignored",
        },
    )

    assert isinstance(result, Ok)
    config = result.value
    assert config.retry.attempts == 5
    assert str(config.retry.initial_delay_seconds) == "3"
    assert str(config.retry.max_delay_seconds) == "8"
    assert config.data.batch_size == 7
    assert str(config.execution.commission_bps) == "12.5"
    assert config.paths.data_root == root / "data"
    assert config.secrets.https_proxy is not None
    assert (
        config.secrets.https_proxy.get_secret_value()
        == "https://user:password@proxy.invalid"
    )


def test_only_explicit_qrp_environment_leaves_are_accepted(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    result = manager.resolve(VALID_YAML, {"QRP_DATA__NOT_A_FIELD": "1"})

    errors = _errors(result)
    assert errors[0].category is ErrorCategory.CONFIGURATION_UNKNOWN_KEY
    assert errors[0].field_path == "QRP_DATA__NOT_A_FIELD"
    assert "explicit configuration mapping" in errors[0].message


def test_plain_yaml_cannot_supply_secret_values(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    secret = "https://user:credential@proxy.invalid"
    result = manager.resolve(VALID_YAML + f"\nsecrets:\n  https_proxy: {secret}\n", {})

    errors = _errors(result)
    assert errors[0].field_path == "secrets.https_proxy"
    assert errors[0].category is ErrorCategory.CONFIGURATION_INVALID_VALUE
    assert secret not in errors[0].format_for_display()


def test_relative_paths_are_project_rooted_absolute_paths_are_preserved_and_escapes_fail(
    tmp_path: Path,
) -> None:
    manager, root = _manager(tmp_path)
    absolute_artifacts = tmp_path / "external-artifacts"
    valid_result = manager.resolve(
        VALID_YAML
        + """
paths:
  data_root: state/data
  artifact_root: """
        + str(absolute_artifacts),
        {},
    )

    assert isinstance(valid_result, Ok)
    assert valid_result.value.paths.data_root == root / "state" / "data"
    assert valid_result.value.paths.artifact_root == absolute_artifacts.resolve()

    escape_target = tmp_path / "outside"
    escaped = manager.resolve(VALID_YAML + "\npaths:\n  data_root: ../outside\n", {})
    errors = _errors(escaped)
    assert errors[0].field_path == "paths.data_root"
    assert "Project_Root boundary" in errors[0].message
    assert not escape_target.exists()


def test_project_root_requires_one_pyproject_boundary(tmp_path: Path) -> None:
    root = tmp_path / "outer"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'outer'\n")
    (nested / "pyproject.toml").write_text("[project]\nname = 'nested'\n")

    with pytest.raises(ProjectRootBoundaryError, match="multiple pyproject.toml"):
        resolve_project_root(nested / "src" / "package" / "module.py")

    with pytest.raises(ProjectRootBoundaryError, match="no pyproject.toml"):
        resolve_project_root(tmp_path / "without-boundary" / "module.py")


def test_schema_errors_are_ordered_and_invalid_configuration_never_calls_downstream(
    tmp_path: Path,
) -> None:
    manager, _ = _manager(tmp_path)
    calls: list[str] = []

    def downstream(_: object) -> Result[str]:
        calls.append("called")
        return Ok("called")

    result = manager.resolve_then(
        """
retry:
  attempts: 0
data:
  requested_range:
    start: 2020-01-01
    end: 2020-12-31
  universe: [AAPL, " aapl "]
  batch_size: 11
execution:
  commission_bps: -1
ui:
  page_size: 101
""",
        {},
        downstream,
    )

    errors = _errors(result)
    assert calls == []
    assert [error.field_path for error in errors] == [
        "retry.attempts",
        "data.universe[1]",
        "data.batch_size",
        "execution.commission_bps",
        "ui.page_size",
    ]
