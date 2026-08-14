"""Focused checks for the reproducible project foundation."""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
LOCK_FILE = PROJECT_ROOT / "uv.lock"
EXPECTED_RUNTIME = {
    "pydantic",
    "ruamel-yaml",
    "yfinance",
    "exchange-calendars",
    "pyarrow",
    "duckdb",
    "zipline-reloaded",
    "mlflow",
    "streamlit",
    "pandas",
    "numpy",
}
EXPECTED_DEVELOPMENT = {"pytest", "hypothesis", "ruff", "mypy", "coverage"}
EXCLUDED_PACKAGES = {
    "fastapi",
    "celery",
    "rq",
    "openai",
    "anthropic",
    "torch",
    "tensorflow",
    "alphalens",
    "alphalens-reloaded",
    "pyfolio",
    "pyfolio-reloaded",
    "robinhood",
}


def _package_name(requirement: str) -> str:
    """Extract a PEP 503-normalized package name from a PEP 508 requirement."""

    name = re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.lower())


def _locked_project(lock: dict[str, object]) -> dict[str, object]:
    """Return the editable project entry from uv's lock package list."""

    packages = lock["package"]
    assert isinstance(packages, list)
    project = next(
        package
        for package in packages
        if isinstance(package, dict)
        and package.get("name") == "quant-research-platform"
    )
    assert isinstance(project, dict)
    return project


def _metadata() -> dict[str, object]:
    with PROJECT_FILE.open("rb") as project_file:
        return tomllib.load(project_file)


def test_project_metadata_declares_supported_runtime_and_build() -> None:
    metadata = _metadata()

    assert metadata["build-system"] == {
        "requires": ["setuptools>=69.0,<81.0"],
        "build-backend": "setuptools.build_meta",
    }
    project = metadata["project"]
    assert project["requires-python"] == ">=3.11,<3.12"
    assert project["name"] == "quant-research-platform"


def test_project_metadata_declares_only_approved_direct_packages() -> None:
    metadata = _metadata()
    project = metadata["project"]
    runtime = {_package_name(value) for value in project["dependencies"]}
    development = {
        _package_name(value)
        for value in metadata["dependency-groups"]["dev"]
    }

    assert runtime == EXPECTED_RUNTIME
    assert development == EXPECTED_DEVELOPMENT
    assert not (runtime | development) & EXCLUDED_PACKAGES


def test_tool_configuration_declares_required_markers_and_profiles() -> None:
    metadata = _metadata()
    pytest_options = metadata["tool"]["pytest"]["ini_options"]
    markers = {entry.split(":", maxsplit=1)[0] for entry in pytest_options["markers"]}

    assert markers == {"integration", "external", "memory", "smoke"}
    assert metadata["tool"]["ruff"]["target-version"] == "py311"
    assert metadata["tool"]["mypy"]["python_version"] == "3.11"
    assert metadata["tool"]["coverage"]["run"]["branch"] is True
    assert set(metadata["tool"]["hypothesis"]["profiles"]) == {"default", "ci"}


@pytest.mark.smoke
def test_uv_lock_is_present_and_frozen_sync_is_current() -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to verify the reviewed lockfile"
    assert LOCK_FILE.is_file(), "uv.lock must be committed for frozen setup"

    metadata = _metadata()
    project = metadata["project"]
    with LOCK_FILE.open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    assert lock["version"] == 1
    assert lock["resolution-markers"]
    locked_project = _locked_project(lock)
    lock_specifier = SpecifierSet(lock["requires-python"])
    project_specifier = SpecifierSet(project["requires-python"])
    for python_version in (Version("3.10"), Version("3.11"), Version("3.12")):
        assert (python_version in lock_specifier) == (
            python_version in project_specifier
        )
    assert Version("3.11") in lock_specifier
    assert Version("3.12") not in lock_specifier

    locked_runtime = {
        item["name"]
        for item in locked_project["dependencies"]
        if isinstance(item, dict)
    }
    locked_development = {
        item["name"]
        for item in locked_project["dev-dependencies"]["dev"]
        if isinstance(item, dict)
    }
    assert locked_runtime == EXPECTED_RUNTIME
    assert locked_development == EXPECTED_DEVELOPMENT
    locked_packages = {
        _package_name(package["name"])
        for package in lock["package"]
        if isinstance(package, dict) and isinstance(package.get("name"), str)
    }
    assert not locked_packages & EXCLUDED_PACKAGES

    result = subprocess.run(
        [uv, "lock", "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    frozen_sync = subprocess.run(
        [uv, "sync", "--frozen", "--dev", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert frozen_sync.returncode == 0, frozen_sync.stderr or frozen_sync.stdout
