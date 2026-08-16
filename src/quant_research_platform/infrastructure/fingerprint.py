"""Deterministic environment and source fingerprints for recorded runs.

The fingerprint is an immutable description of the inputs that can affect a
local scientific run.  Source identity is deliberately based on canonical
relative paths and file bytes rather than absolute checkout locations, and it
includes untracked source files.  Operational checkout paths and timestamps
are never part of the resulting identity.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from quant_research_platform.domain.canonical import (
    canonical_json,
    normalize_unicode,
    sha256_bytes,
    sha256_canonical_json,
)

FINGERPRINT_SCHEMA_VERSION: Final = "environment_fingerprint_v1"
SOURCE_CHECKSUM_SCHEMA_VERSION: Final = "effective_source_checksum_v1"
_RECOGNIZED_LOCK_FILES: Final[tuple[str, ...]] = (
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "pdm.lock",
    "requirements.lock",
)
_EXCLUDED_DIRECTORY_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".conda",
        ".direnv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".cache",
        ".uv",
        "build",
        "dist",
        ".eggs",
        "pip-wheel-metadata",
        "data",
        "staging",
        "objects",
        "cas",
        "artifacts",
        "snapshots",
        "runs",
        "derived",
        "zipline-bundles",
        "test-output",
        "test_outputs",
        "test-results",
        "test_results",
    }
)
_EXCLUDED_FILE_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".coverage",
        "coverage.xml",
        "pytest-report.xml",
    }
)
_EXCLUDED_FILE_SUFFIXES: Final[tuple[str, ...]] = (
    ".pyc",
    ".pyo",
    ".tmp",
)


class FingerprintError(ValueError):
    """Raised when a project cannot be fingerprinted safely."""


@dataclass(frozen=True, slots=True)
class SourceFileFingerprint:
    """The canonical identity facts for one included source file."""

    relative_path: str
    executable: bool
    checksum: str

    def __post_init__(self) -> None:
        path = _relative_posix_path(self.relative_path)
        if not isinstance(self.executable, bool):
            raise TypeError("executable must be a boolean")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checksum):
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        object.__setattr__(self, "relative_path", path)

    def to_content_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "executable": self.executable,
            "sha256": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Immutable, serializable environment facts used by a research run."""

    python_version: str
    os_name: str
    os_version: str
    architecture: str
    installed_distributions: tuple[tuple[str, str], ...]
    source_revision: str | None
    source_dirty: bool
    deterministic_seed: int
    effective_source_checksum: str

    def __post_init__(self) -> None:
        for field_name in (
            "python_version",
            "os_name",
            "os_version",
            "architecture",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
            object.__setattr__(self, field_name, normalize_unicode(value.strip()))
        if self.source_revision is not None:
            if not isinstance(self.source_revision, str) or not self.source_revision:
                raise ValueError("source_revision must be non-empty text or None")
            object.__setattr__(
                self, "source_revision", normalize_unicode(self.source_revision)
            )
        if not isinstance(self.source_dirty, bool):
            raise TypeError("source_dirty must be a boolean")
        if isinstance(self.deterministic_seed, bool) or not isinstance(
            self.deterministic_seed, int
        ):
            raise TypeError("deterministic_seed must be an integer")
        if not 0 <= self.deterministic_seed <= 4_294_967_295:
            raise ValueError("deterministic_seed must be between 0 and 4294967295")
        if not isinstance(self.installed_distributions, tuple):
            raise TypeError("installed_distributions must be an immutable tuple")
        normalized_distributions: list[tuple[str, str]] = []
        for item in self.installed_distributions:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "each installed distribution must be a (name, version) tuple"
                )
            name, version = item
            if not isinstance(name, str) or not isinstance(version, str):
                raise TypeError("distribution names and versions must be strings")
            normalized_distributions.append((_distribution_name(name), version.strip()))
        canonical_distributions = tuple(sorted(normalized_distributions))
        object.__setattr__(self, "installed_distributions", canonical_distributions)
        _validate_checksum("effective_source_checksum", self.effective_source_checksum)

    @property
    def seed(self) -> int:
        """Compatibility alias for the deterministic seed field."""

        return self.deterministic_seed

    @property
    def dirty_state(self) -> str:
        """Return a display-safe state suitable for run disclosures."""

        return "dirty" if self.source_dirty else "clean"

    @property
    def dirty_disclosure(self) -> str:
        """Return the explicit source-state disclosure required for a run."""

        if self.source_dirty:
            return (
                "Source checkout is dirty; this run is identified by its effective "
                "source checksum and must not be treated as a clean revision."
            )
        return "Source checkout is clean at the recorded source revision."

    def to_content_dict(self) -> dict[str, object]:
        """Return canonical scientific fingerprint fields without local paths."""

        return {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "python_version": self.python_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "installed_distributions": [
                {"name": name, "version": version}
                for name, version in self.installed_distributions
            ],
            "source_revision": self.source_revision,
            "source_dirty": self.source_dirty,
            "deterministic_seed": self.deterministic_seed,
            "effective_source_checksum": self.effective_source_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the durable representation used by manifests and trackers."""

        return self.to_content_dict()

    def canonical_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for the fingerprint."""

        return canonical_json(self.to_content_dict())

    @property
    def checksum(self) -> str:
        """Return the checksum of the complete fingerprint content."""

        return sha256_canonical_json(self.to_content_dict())

    @property
    def content_checksum(self) -> str:
        """Alias used by manifest code when naming scientific checksums."""

        return self.checksum


def _distribution_name(value: str) -> str:
    """Apply the PEP 503 normalization used for stable distribution names."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("distribution name must be non-empty text")
    return re.sub(r"[-_.]+", "-", value.strip()).lower()


def _validate_checksum(field_name: str, value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _relative_posix_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("relative path must be text")
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    path = Path(normalized)
    if path.is_absolute() or normalized in {"", "."}:
        raise ValueError("source paths must be non-empty relative paths")
    parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        raise ValueError("source paths must not escape the project root")
    return "/".join(parts)


def _project_root(value: Path | str | None) -> Path:
    root = Path.cwd() if value is None else Path(value)
    root = root.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FingerprintError(f"project root is not a directory: {root}")
    if not (root / "pyproject.toml").is_file():
        raise FingerprintError(f"project root has no pyproject.toml: {root}")
    return root


def _included_path(path: Path, project_root: Path) -> bool:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return False
    parts = tuple(part.lower() for part in relative.parts)
    if any(
        part in _EXCLUDED_DIRECTORY_NAMES or part.endswith((".egg-info", ".dist-info"))
        for part in parts[:-1]
    ):
        return False
    name = parts[-1]
    if name in _EXCLUDED_FILE_NAMES:
        return False
    return not name.endswith(_EXCLUDED_FILE_SUFFIXES)


def _source_files(
    root: Path, source_roots: Sequence[Path | str] | None
) -> tuple[Path, ...]:
    if source_roots is None:
        candidates: list[Path] = []
        src = root / "src"
        if src.is_dir():
            candidates.append(src)
        else:
            candidates.extend(path for path in root.iterdir() if path.suffix == ".py")
            candidates.extend(
                path
                for path in root.iterdir()
                if path.is_dir() and (path / "__init__.py").is_file()
            )
        source_roots = tuple(candidates)

    files: dict[str, Path] = {}
    for candidate in source_roots:
        source_root = Path(candidate).expanduser().resolve(strict=False)
        try:
            source_root.relative_to(root)
        except ValueError as error:
            raise FingerprintError(
                f"source root must be inside project root: {source_root}"
            ) from error
        if not source_root.exists():
            continue
        if source_root.is_file():
            paths: tuple[Path, ...] = (source_root,)
        else:
            paths = tuple(source_root.rglob("*"))
        for path in paths:
            if (
                path.is_symlink()
                or not path.is_file()
                or not _included_path(path, root)
            ):
                continue
            relative = _relative_posix_path(path.relative_to(root).as_posix())
            files[relative] = path
    return tuple(files[path] for path in sorted(files))


def _active_lock_files(
    root: Path, lock_files: Sequence[Path | str] | None
) -> tuple[Path, ...]:
    if lock_files is None:
        discovered = tuple(root / name for name in _RECOGNIZED_LOCK_FILES)
        return tuple(path for path in discovered if path.is_file())
    files: dict[str, Path] = {}
    for candidate in lock_files:
        path = Path(candidate).expanduser().resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise FingerprintError(
                f"lock file must be inside project root: {path}"
            ) from error
        if path.is_file():
            files[_relative_posix_path(path.relative_to(root).as_posix())] = path
    return tuple(files[key] for key in sorted(files))


def source_file_fingerprints(
    project_root: Path | str | None = None,
    *,
    source_roots: Sequence[Path | str] | None = None,
    lock_files: Sequence[Path | str] | None = None,
) -> tuple[SourceFileFingerprint, ...]:
    """Return sorted source/metadata entries used by the effective checksum."""

    root = _project_root(project_root)
    paths: dict[str, Path] = {}
    for path in _source_files(root, source_roots):
        paths[_relative_posix_path(path.relative_to(root).as_posix())] = path
    metadata_paths = (root / "pyproject.toml",) + _active_lock_files(root, lock_files)
    for path in metadata_paths:
        if not path.is_file():
            raise FingerprintError(f"fingerprint input is not a file: {path}")
        paths[_relative_posix_path(path.relative_to(root).as_posix())] = path

    entries: list[SourceFileFingerprint] = []
    for relative_path in sorted(paths):
        path = paths[relative_path]
        try:
            payload = path.read_bytes()
            mode = path.stat().st_mode
        except OSError as error:
            raise FingerprintError(
                f"cannot read fingerprint input: {relative_path}"
            ) from error
        entries.append(
            SourceFileFingerprint(
                relative_path=relative_path,
                executable=bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
                checksum=sha256_bytes(payload),
            )
        )
    return tuple(entries)


def compute_effective_source_checksum(
    project_root: Path | str | None = None,
    *,
    source_roots: Sequence[Path | str] | None = None,
    lock_files: Sequence[Path | str] | None = None,
) -> str:
    """Hash canonical relative source paths, executable bits, and file bytes."""

    entries = source_file_fingerprints(
        project_root, source_roots=source_roots, lock_files=lock_files
    )
    payload = {
        "schema_version": SOURCE_CHECKSUM_SCHEMA_VERSION,
        "files": [entry.to_content_dict() for entry in entries],
    }
    return sha256_canonical_json(payload)


def effective_source_checksum(
    project_root: Path | str | None = None,
    *,
    source_roots: Sequence[Path | str] | None = None,
    lock_files: Sequence[Path | str] | None = None,
) -> str:
    """Short alias for :func:`compute_effective_source_checksum`."""

    return compute_effective_source_checksum(
        project_root, source_roots=source_roots, lock_files=lock_files
    )


def collect_installed_distributions() -> tuple[tuple[str, str], ...]:
    """Collect exact installed distribution names and versions in sorted order."""

    distributions: list[tuple[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.name
        version = distribution.version
        if name and version:
            distributions.append((_distribution_name(name), version))
    return tuple(sorted(distributions))


def _git_state(project_root: Path) -> tuple[str | None, bool]:
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None, True
    if revision_result.returncode != 0 or status_result.returncode != 0:
        return None, True
    revision = revision_result.stdout.strip() or None
    return revision, bool(status_result.stdout.strip())


def fingerprint_environment(
    project_root: Path | str | None = None,
    deterministic_seed: int = 0,
    *,
    seed: int | None = None,
    source_roots: Sequence[Path | str] | None = None,
    lock_files: Sequence[Path | str] | None = None,
) -> EnvironmentFingerprint:
    """Build the deterministic environment/source fingerprint for a project.

    ``seed`` is accepted as a readable alias for ``deterministic_seed``; passing
    both values is rejected unless they agree.  Source paths are only used while
    calculating bytes and are never retained in the returned fingerprint.
    """

    if seed is not None:
        if deterministic_seed != 0 and deterministic_seed != seed:
            raise ValueError("seed and deterministic_seed disagree")
        deterministic_seed = seed
    root = _project_root(project_root)
    checksum = compute_effective_source_checksum(
        root, source_roots=source_roots, lock_files=lock_files
    )
    revision, dirty = _git_state(root)
    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        os_name=platform.system() or sys.platform,
        os_version=platform.version() or platform.release(),
        architecture=platform.machine() or platform.processor() or "unknown",
        installed_distributions=collect_installed_distributions(),
        source_revision=revision,
        source_dirty=dirty,
        deterministic_seed=deterministic_seed,
        effective_source_checksum=checksum,
    )


build_environment_fingerprint = fingerprint_environment
collect_environment_fingerprint = fingerprint_environment
compute_source_checksum = compute_effective_source_checksum


__all__ = [
    "EnvironmentFingerprint",
    "FingerprintError",
    "FINGERPRINT_SCHEMA_VERSION",
    "SOURCE_CHECKSUM_SCHEMA_VERSION",
    "SourceFileFingerprint",
    "build_environment_fingerprint",
    "collect_environment_fingerprint",
    "collect_installed_distributions",
    "compute_effective_source_checksum",
    "compute_source_checksum",
    "effective_source_checksum",
    "fingerprint_environment",
    "source_file_fingerprints",
]
