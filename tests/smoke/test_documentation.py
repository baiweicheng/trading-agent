"""Offline checks for developer and research-operation documentation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "developer-guide.md",
    PROJECT_ROOT / "docs" / "research-operations.md",
)
RELATIVE_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
YAML_FENCE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)

pytestmark = pytest.mark.smoke


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_documentation_relative_links_resolve() -> None:
    for document in DOCUMENTS:
        for target in RELATIVE_LINK.findall(_text(document)):
            if "://" in target or target.startswith("#"):
                continue
            target_path = target.split("#", maxsplit=1)[0]
            assert target_path, f"empty relative link in {document}"
            relative_document = document.relative_to(PROJECT_ROOT)
            assert (document.parent / target_path).resolve().is_file(), (
                f"broken documentation link {target!r} in {relative_document}"
            )


def test_documentation_names_supported_single_shot_commands() -> None:
    combined = "\n".join(_text(path) for path in DOCUMENTS)
    required_fragments = (
        "uv sync --frozen",
        "uv lock --check",
        'uv run pytest -m "not external"',
        "uv run ruff check src tests",
        "uv run mypy src",
        "uv run streamlit run src/quant_research_platform/ui/app.py",
        "QRP_RUN_EXTERNAL_TESTS=1 uv run pytest",
        'FilesystemStore(Path("data"), metadata=metadata).reconcile()',
    )
    for fragment in required_fragments:
        assert fragment in combined, f"missing documented command/API: {fragment}"


def test_documented_yaml_examples_are_safe_mappings() -> None:
    parser = YAML(typ="safe")
    for document in DOCUMENTS:
        examples = YAML_FENCE.findall(_text(document))
        for example in examples:
            loaded = parser.load(example)
            assert isinstance(loaded, dict), (
                f"YAML example is not a mapping: {document}"
            )
            serialized = example.casefold()
            assert "password" not in serialized
            assert "token" not in serialized
            assert "api_key" not in serialized
            assert "proxy.invalid" not in serialized


def test_documentation_has_no_real_credentials_or_machine_specific_paths() -> None:
    combined = "\n".join(_text(path) for path in DOCUMENTS)
    forbidden_fragments = (
        "user:password",
        "literal-secret",
        "https://user:",
        "/Users/",
        "/private/var/",
    )
    for fragment in forbidden_fragments:
        assert fragment not in combined


def test_default_configuration_is_the_documented_safe_starting_point() -> None:
    default_config = PROJECT_ROOT / "config" / "default.yaml"
    document = YAML(typ="safe").load(default_config.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    assert document["data"]["benchmark"] == "SPY"
    assert document["strategy"]["identifier"] == "monthly_momentum_v1"
    assert document["execution"]["initial_equity_usd"] == "100000"
    assert document["secrets"]["http_proxy"] is None
    assert document["secrets"]["https_proxy"] is None
