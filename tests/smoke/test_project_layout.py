"""Focused checks for the Phase 1 repository layout and source-control hygiene."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "quant_research_platform"
DATA_ROOT = PROJECT_ROOT / "data"

PACKAGE_DIRECTORIES = {
    PACKAGE_ROOT / name
    for name in ("domain", "config", "application", "infrastructure", "ui")
}
TEST_DIRECTORIES = {
    PROJECT_ROOT / "tests" / name
    for name in ("unit", "properties", "contract", "integration", "golden", "smoke")
}
DATA_DIRECTORIES = {
    DATA_ROOT / name
    for name in (
        "raw",
        "normalized",
        "quarantine",
        "snapshots",
        "runs",
        "staging",
        "objects",
        "artifacts",
        "zipline-bundles",
    )
}
IGNORED_PATHS = (
    ".env.local",
    "config/secrets.local.yaml",
    "data/staging/sample.parquet",
    "data/objects/sha256/sample",
    "data/artifacts/result.json",
    "data/zipline-bundles/sample.sqlite",
    "metadata.duckdb",
    "mlflow.db",
)
TRACKABLE_PATHS = (
    "src/quant_research_platform/domain/__init__.py",
    "config/default.yaml",
    "tests/golden/reviewed-fixture.json",
)


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode in (0, 1), (
        f"git check-ignore failed for {path!r}: exit {result.returncode}"
    )
    return result.returncode == 0


def test_phase_one_package_and_test_layout_is_present() -> None:
    for directory in PACKAGE_DIRECTORIES | TEST_DIRECTORIES:
        assert directory.is_dir(), f"missing required directory: {directory}"
        assert (directory / "__init__.py").is_file(), (
            f"missing package marker: {directory / '__init__.py'}"
        )

    assert (PROJECT_ROOT / "config" / "default.yaml").is_file()
    assert all(directory.is_dir() for directory in DATA_DIRECTORIES)


@pytest.mark.parametrize("path", IGNORED_PATHS)
def test_local_secrets_and_generated_data_are_ignored(path: str) -> None:
    assert _is_ignored(path), f"expected path to be ignored: {path}"


@pytest.mark.parametrize("path", TRACKABLE_PATHS)
def test_source_configuration_and_golden_fixtures_remain_trackable(path: str) -> None:
    assert not _is_ignored(path), f"expected path to remain trackable: {path}"
