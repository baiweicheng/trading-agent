"""Single-shot smoke checks for the installable Phase 1 project boundary."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from ruamel.yaml import YAML

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
PACKAGE_NAME = "quant_research_platform"
EXPECTED_DEVELOPMENT_TOOLS = {
    "coverage",
    "hypothesis",
    "mypy",
    "pytest",
    "ruff",
}
IGNORED_SAMPLES = (
    ".env.local",
    "config/secrets.local.yaml",
    "data/metadata.duckdb",
    "data/mlflow.db",
    "data/staging/candidate.parquet",
    "data/objects/sha256/candidate.parquet",
    "data/artifacts/result.json",
    "data/zipline-bundles/snapshot.sqlite",
)


def _metadata() -> dict[str, object]:
    with PROJECT_FILE.open("rb") as project_file:
        return tomllib.load(project_file)


def _requirement_name(requirement: str) -> str:
    """Return the normalized distribution name from a PEP 508 requirement."""

    name = re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.lower())


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _is_ignored(path: str) -> bool:
    result = _run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=PROJECT_ROOT,
    )
    assert result.returncode in (0, 1), (
        f"git check-ignore failed for {path!r}: exit {result.returncode}"
    )
    return result.returncode == 0


def test_project_declares_python_311_and_required_dependency_groups() -> None:
    metadata = _metadata()
    project = metadata["project"]
    dependency_groups = metadata["dependency-groups"]

    assert sys.version_info[:2] == (3, 11)
    assert isinstance(project, dict)
    assert project["requires-python"] == ">=3.11,<3.12"
    assert isinstance(project["dependencies"], list)
    assert project["dependencies"], "runtime dependencies must be declared"

    assert isinstance(dependency_groups, dict)
    development = dependency_groups["dev"]
    assert isinstance(development, list)
    development_names = {
        _requirement_name(requirement)
        for requirement in development
        if isinstance(requirement, str)
    }
    assert development_names >= EXPECTED_DEVELOPMENT_TOOLS


def test_default_configuration_is_a_safe_mapping_with_required_sections() -> None:
    assert DEFAULT_CONFIG.is_file(), "config/default.yaml must be present"

    document = YAML(typ="safe").load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    assert {
        "paths",
        "retry",
        "data",
        "strategy",
        "execution",
        "ui",
        "runtime",
        "secrets",
    } <= document.keys()


@pytest.mark.parametrize("path", IGNORED_SAMPLES)
def test_secret_and_generated_samples_are_ignored(path: str) -> None:
    assert _is_ignored(path), f"expected path to be ignored: {path}"


def test_wheel_builds_and_imports_without_the_source_tree(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build the reviewed package"

    build = _run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    wheels = tuple(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found: {wheels}"
    wheel = wheels[0].resolve()
    import_script = "\n".join(
        (
            "import importlib",
            "import sys",
            "wheel = sys.argv[1]",
            "sys.path.insert(0, wheel)",
            f"package = importlib.import_module({PACKAGE_NAME!r})",
            "assert package.__file__ is not None",
            "assert package.__file__.startswith(wheel)",
        )
    )
    imported = _run(
        [sys.executable, "-I", "-c", import_script, str(wheel)],
        cwd=tmp_path,
    )
    assert imported.returncode == 0, imported.stderr or imported.stdout
