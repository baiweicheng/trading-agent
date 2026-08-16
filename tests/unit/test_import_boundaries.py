"""Static architecture checks for the Phase 1 package boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "quant_research_platform"
PACKAGE_PREFIX = "quant_research_platform"
LAYER_IMPORT_RESTRICTIONS = {
    "domain": frozenset({"application", "infrastructure", "ui"}),
    "application": frozenset({"infrastructure", "ui"}),
    "ui": frozenset({"domain", "infrastructure"}),
}
DOMAIN_FRAMEWORKS = frozenset(
    {
        "duckdb",
        "exchange_calendars",
        "mlflow",
        "pyarrow",
        "streamlit",
        "yfinance",
        "zipline",
    }
)


def _python_files(layer: str) -> tuple[Path, ...]:
    return tuple(sorted((PACKAGE_ROOT / layer).rglob("*.py")))


def _current_package(path: Path) -> tuple[str, ...]:
    relative_path = path.relative_to(SOURCE_ROOT).with_suffix("")
    return relative_path.parts[:-1]


def _import_from_target(
    node: ast.ImportFrom, current_package: tuple[str, ...]
) -> tuple[str, ...]:
    if node.level == 0:
        return tuple(node.module.split(".")) if node.module else ()

    parent_length = len(current_package) - node.level + 1
    base = current_package[: max(parent_length, 0)]
    module = tuple(node.module.split(".")) if node.module else ()
    return base + module


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current_package = _current_package(path)
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _import_from_target(node, current_package)
            if target:
                imported.add(".".join(target))
            for alias in node.names:
                if alias.name != "*" and target:
                    imported.add(".".join((*target, alias.name)))

    return tuple(sorted(imported))


def _imports_layer(module: str, layer: str) -> bool:
    layer_prefix = f"{PACKAGE_PREFIX}.{layer}"
    return module == layer_prefix or module.startswith(f"{layer_prefix}.")


def _is_allowed_composition_root_import(path: Path, module: str) -> bool:
    """The documented UI composition root is the adapter-wiring exception."""

    return path == PACKAGE_ROOT / "ui" / "app.py" and module.startswith(
        f"{PACKAGE_PREFIX}.infrastructure"
    )


def _format_violations(violations: Iterable[tuple[Path, str]]) -> str:
    return "\n".join(
        f"{path.relative_to(PROJECT_ROOT)} imports {module}"
        for path, module in violations
    )


@pytest.mark.parametrize(
    ("layer", "forbidden_layers"),
    tuple(LAYER_IMPORT_RESTRICTIONS.items()),
)
def test_layers_only_depend_inward(
    layer: str, forbidden_layers: frozenset[str]
) -> None:
    violations = [
        (path, module)
        for path in _python_files(layer)
        for module in _imported_modules(path)
        if any(_imports_layer(module, forbidden) for forbidden in forbidden_layers)
        and not _is_allowed_composition_root_import(path, module)
    ]

    assert not violations, (
        "Layer dependencies must remain presentation -> application -> domain:\n"
        f"{_format_violations(violations)}"
    )


def test_domain_does_not_import_infrastructure_frameworks() -> None:
    violations = [
        (path, module)
        for path in _python_files("domain")
        for module in _imported_modules(path)
        if module.split(".", maxsplit=1)[0] in DOMAIN_FRAMEWORKS
    ]

    assert not violations, (
        "Domain policies must remain independent of infrastructure frameworks:\n"
        f"{_format_violations(violations)}"
    )


def test_application_does_not_read_streamlit_session_state() -> None:
    violations: list[tuple[Path, str]] = []

    for path in _python_files("application"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        streamlit_aliases: set[str] = set()
        session_state_aliases: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", maxsplit=1)[0] == "streamlit":
                        streamlit_aliases.add(alias.asname or alias.name.split(".")[0])
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".", maxsplit=1)[0] == "streamlit"
            ):
                for alias in node.names:
                    if alias.name == "session_state":
                        session_state_aliases.add(alias.asname or alias.name)
                        violations.append((path, "streamlit.session_state import"))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "session_state"
                and isinstance(node.value, ast.Name)
                and node.value.id in streamlit_aliases
            ):
                violations.append((path, "streamlit.session_state"))
            elif (
                isinstance(node, ast.Name)
                and node.id in session_state_aliases
                and isinstance(node.ctx, ast.Load)
            ):
                violations.append((path, "streamlit.session_state alias"))

    assert not violations, (
        "Application services must not read Streamlit session state:\n"
        f"{_format_violations(violations)}"
    )
