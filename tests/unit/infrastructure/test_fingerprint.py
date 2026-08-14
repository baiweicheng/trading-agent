"""Focused deterministic environment/source fingerprint examples."""

from __future__ import annotations

import subprocess
from pathlib import Path

from quant_research_platform.infrastructure.fingerprint import (
    EnvironmentFingerprint,
    compute_effective_source_checksum,
    fingerprint_environment,
    source_file_fingerprints,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src" / "quant_research_platform").mkdir(parents=True)
    (root / "src" / "quant_research_platform" / "__init__.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'fingerprint-fixture'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def _git_commit(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fingerprint Tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
    )


def test_source_checksum_is_independent_of_root_and_input_order(tmp_path: Path) -> None:
    first = _project(tmp_path / "first")
    second = _project(tmp_path / "second")

    first_checksum = compute_effective_source_checksum(
        first,
        source_roots=(first / "src", first / "src" / "quant_research_platform"),
        lock_files=(first / "uv.lock", first / "uv.lock"),
    )
    second_checksum = compute_effective_source_checksum(
        second,
        source_roots=(second / "src" / "quant_research_platform", second / "src"),
        lock_files=(second / "uv.lock",),
    )

    assert first_checksum == second_checksum
    assert [entry.relative_path for entry in source_file_fingerprints(first)] == [
        "pyproject.toml",
        "src/quant_research_platform/__init__.py",
        "uv.lock",
    ]


def test_untracked_source_changes_change_checksum(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _git_commit(root)
    initial = compute_effective_source_checksum(root)

    (root / "src" / "quant_research_platform" / "new_module.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    assert compute_effective_source_checksum(root) != initial


def test_generated_cache_and_test_output_files_are_excluded(tmp_path: Path) -> None:
    root = _project(tmp_path)
    initial = compute_effective_source_checksum(root)

    (root / "data").mkdir()
    (root / "data" / "generated.parquet").write_bytes(b"generated")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "cache.json").write_bytes(b"cache")
    (root / "test-output").mkdir()
    (root / "test-output" / "report.json").write_bytes(b"output")
    (root / "src" / "quant_research_platform" / "__pycache__").mkdir()
    (root / "src" / "quant_research_platform" / "__pycache__" / "x.pyc").write_bytes(
        b"bytecode"
    )
    (root / "src" / "fingerprint_fixture.egg-info").mkdir()
    (root / "src" / "fingerprint_fixture.egg-info" / "PKG-INFO").write_bytes(
        b"generated metadata"
    )

    assert compute_effective_source_checksum(root) == initial


def test_dirty_source_state_is_recorded_and_disclosed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _git_commit(root)
    clean = fingerprint_environment(root, deterministic_seed=123)
    assert clean.source_revision
    assert clean.source_dirty is False
    assert clean.dirty_state == "clean"

    (root / "src" / "quant_research_platform" / "__init__.py").write_text(
        "VALUE = 99\n", encoding="utf-8"
    )
    dirty = fingerprint_environment(root, seed=123)

    assert dirty.source_revision == clean.source_revision
    assert dirty.source_dirty is True
    assert "dirty" in dirty.dirty_disclosure.lower()
    assert dirty.effective_source_checksum != clean.effective_source_checksum
    assert dirty.canonical_bytes().endswith(b"\n")
    assert isinstance(dirty, EnvironmentFingerprint)


def test_distributions_are_canonicalized_and_sorted() -> None:
    fingerprint = EnvironmentFingerprint(
        python_version="3.11.0",
        os_name="TestOS",
        os_version="1",
        architecture="test",
        installed_distributions=(
            ("Zed_Package", "2"),
            ("a-package", "1"),
        ),
        source_revision=None,
        source_dirty=True,
        deterministic_seed=0,
        effective_source_checksum="a" * 64,
    )

    assert fingerprint.installed_distributions == (
        ("a-package", "1"),
        ("zed-package", "2"),
    )
    assert fingerprint.to_dict()["source_dirty"] is True
    assert fingerprint.checksum == fingerprint.content_checksum
