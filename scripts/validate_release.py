"""Run the single-shot, offline Phase 1 release validation gates.

The command intentionally uses only local tools and the already-synchronized
Python environment.  It never starts a server, watcher, deployment, or
external-provider test.
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_SUPPORTED_PYTHON = ">=3.11,<3.12"
_FORBIDDEN_DEPENDENCIES = frozenset(
    {
        "alphalens",
        "alpaca-trade-api",
        "anthropic",
        "boto3",
        "celery",
        "fastapi",
        "flask",
        "ib-insync",
        "openai",
        "pyfolio",
        "redis",
        "scikit-learn",
        "tensorflow",
        "torch",
    }
)
_FORBIDDEN_IMPORTS = frozenset(
    {
        "alphalens",
        "alpaca_trade_api",
        "anthropic",
        "boto3",
        "celery",
        "fastapi",
        "flask",
        "ib_insync",
        "openai",
        "pyfolio",
        "redis",
        "sklearn",
        "tensorflow",
        "torch",
    }
)


class CommandRunner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        check: bool,
        text: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class Gate:
    """One release gate and the command that repairs it."""

    name: str
    command: tuple[str, ...]
    corrective_command: str


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _tail(output: str, *, limit: int = 40) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(
        (f"... ({len(lines) - limit} earlier lines omitted)", *lines[-limit:])
    )


def run_gate(
    gate: Gate,
    project_root: Path,
    *,
    runner: CommandRunner = subprocess.run,
) -> int:
    """Run one command, report a concise failure, and return its exact status."""

    print(f"[release] {gate.name}: {_command_text(gate.command)}")
    try:
        completed = runner(
            gate.command,
            cwd=project_root,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as error:
        print(f"FAIL {gate.name}: {error}")
        print(f"Corrective command: {gate.corrective_command}")
        return 127

    if completed.returncode == 0:
        print(f"PASS {gate.name}")
        return 0

    output = "\n".join(value for value in (completed.stdout, completed.stderr) if value)
    print(f"FAIL {gate.name} (exit {completed.returncode})")
    if output:
        print(_tail(output))
    print(f"Corrective command: {gate.corrective_command}")
    return completed.returncode


def _dependency_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.split(r"\s*(?:[<>=!~;@]|\[)\s*", value, maxsplit=1)[0].strip().lower()


def _source_imports(source_root: Path) -> set[tuple[Path, str]]:
    imports: set[tuple[Path, str]] = set()
    for path in sorted(source_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            module: str | None
            if isinstance(node, ast.Import):
                module = node.names[0].name if node.names else None
            elif isinstance(node, ast.ImportFrom):
                module = node.module
            else:
                continue
            if module:
                imports.add((path, module.split(".", 1)[0]))
    return imports


def check_project_scope(project_root: Path) -> tuple[str, ...]:
    """Validate Python metadata and the explicitly excluded Phase 1 scope."""

    pyproject_path = project_root / "pyproject.toml"
    errors: list[str] = []
    try:
        document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return (f"cannot read pyproject.toml: {type(error).__name__}",)

    project = document.get("project")
    if not isinstance(project, dict):
        return ("pyproject.toml has no [project] table",)
    if project.get("requires-python") != _SUPPORTED_PYTHON:
        errors.append(f"requires-python must be {_SUPPORTED_PYTHON!r}")
    dependencies = project.get("dependencies", ())
    groups = document.get("dependency-groups", {})
    dev_dependencies = groups.get("dev", ()) if isinstance(groups, dict) else ()
    declared = {
        _dependency_name(value)
        for value in (*dependencies, *dev_dependencies)
        if _dependency_name(value)
    }
    for dependency in sorted(declared & _FORBIDDEN_DEPENDENCIES):
        errors.append(f"excluded dependency is declared: {dependency}")

    source_root = project_root / "src"
    if source_root.is_dir():
        for path, module in sorted(_source_imports(source_root)):
            if module in _FORBIDDEN_IMPORTS:
                errors.append(
                    f"excluded import {module!r} in {path.relative_to(project_root)}"
                )
    else:
        errors.append("src directory is missing")
    if not (project_root / "uv.lock").is_file():
        errors.append("uv.lock is missing")
    return tuple(errors)


def _python_command(*parts: str) -> tuple[str, ...]:
    return (sys.executable, *parts)


def build_gates(project_root: Path, build_directory: Path) -> tuple[Gate, ...]:
    """Build the deterministic local command sequence for one checkout."""

    uv = shutil.which("uv") or "uv"
    return (
        Gate(
            "frozen lock",
            (uv, "lock", "--check", "--offline"),
            "uv sync --frozen --offline",
        ),
        Gate(
            "package build",
            (uv, "build", "--wheel", "--offline", "--out-dir", str(build_directory)),
            "uv build --wheel --offline",
        ),
        Gate(
            "package import",
            _python_command(
                "-c",
                "import quant_research_platform; import quant_research_platform.ui.app",
            ),
            f"{sys.executable} -c 'import quant_research_platform'",
        ),
        Gate(
            "ruff lint",
            _python_command("-m", "ruff", "check", "src", "tests"),
            f"{sys.executable} -m ruff check src tests",
        ),
        Gate(
            "ruff format",
            _python_command("-m", "ruff", "format", "--check", "src", "tests"),
            f"{sys.executable} -m ruff format src tests",
        ),
        Gate(
            "mypy",
            _python_command("-m", "mypy", "src"),
            f"{sys.executable} -m mypy src",
        ),
        Gate(
            "offline test suite",
            _python_command(
                "-m",
                "pytest",
                "-q",
                "-x",
                "--no-header",
                "-p",
                "no:cacheprovider",
                "-m",
                "not external",
            ),
            (
                f"{sys.executable} -m pytest -q -x --no-header -p "
                'no:cacheprovider -m "not external"'
            ),
        ),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project directory containing pyproject.toml (default: repository root)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    scope_errors = check_project_scope(project_root)
    if scope_errors:
        print("FAIL project scope")
        for error in scope_errors:
            print(f"- {error}")
        print("Corrective command: inspect pyproject.toml and Phase 1 imports")
        return 1
    print("PASS project scope")

    with tempfile.TemporaryDirectory(prefix="qrp-release-") as temporary_directory:
        gates = build_gates(project_root, Path(temporary_directory))
        for gate in gates:
            status = run_gate(gate, project_root)
            if status:
                return status
    print("PASS release validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
