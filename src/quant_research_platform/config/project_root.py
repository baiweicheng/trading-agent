"""Project-root discovery and local configuration path containment helpers."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Final

PYPROJECT_FILENAME: Final = "pyproject.toml"


class ProjectRootBoundaryError(ValueError):
    """Raised when an anchor has zero or multiple project metadata boundaries."""

    def __init__(self, anchor: Path, boundaries: tuple[Path, ...]) -> None:
        self.anchor = anchor
        self.boundaries = boundaries
        if not boundaries:
            detail = f"no {PYPROJECT_FILENAME} ancestor was found"
        else:
            rendered = ", ".join(str(boundary) for boundary in boundaries)
            detail = f"multiple {PYPROJECT_FILENAME} ancestors were found ({rendered})"
        super().__init__(f"Project-root boundary for {anchor} is ambiguous: {detail}.")


class RelativePathEscapeError(ValueError):
    """Raised when a relative configured path resolves outside the project root."""

    def __init__(
        self, field_path: str, configured_path: Path, project_root: Path
    ) -> None:
        self.field_path = field_path
        self.configured_path = configured_path
        self.project_root = project_root
        super().__init__(
            f"Configured relative path for {field_path} resolves outside the "
            f"Project_Root boundary {project_root}."
        )


def _start_directory(anchor: Path | None) -> Path:
    """Resolve an anchor to the directory from which ancestor discovery starts."""

    candidate = Path(__file__) if anchor is None else Path(anchor)
    candidate = candidate.resolve(strict=False)
    if candidate.name == PYPROJECT_FILENAME or candidate.is_file() or candidate.suffix:
        return candidate.parent
    return candidate


def resolve_project_root(anchor: Path | None = None) -> Path:
    """Return the sole ancestor containing project build metadata.

    The search deliberately rejects both absent and nested metadata boundaries. A
    caller therefore never silently chooses a parent project when an inner project
    defines a separate package boundary.
    """

    start = _start_directory(anchor)
    candidates = (start, *start.parents)
    boundaries = tuple(
        candidate.resolve(strict=False)
        for candidate in candidates
        if (candidate / PYPROJECT_FILENAME).is_file()
    )
    if len(boundaries) != 1:
        raise ProjectRootBoundaryError(start, boundaries)
    return boundaries[0]


def normalize_local_path(
    configured_path: Path | str,
    *,
    project_root: Path,
    field_path: str,
) -> Path:
    """Normalize a local path without creating it or permitting relative escapes.

    Absolute paths remain allowed because they are explicit user choices. Relative
    paths are resolved against the verified project root and are rejected before a
    caller can create directories or files outside that boundary.
    """

    raw_path = Path(configured_path)
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)

    normalized_root = project_root.resolve(strict=False)
    normalized_path = (normalized_root / raw_path).resolve(strict=False)
    if not normalized_path.is_relative_to(normalized_root):
        raise RelativePathEscapeError(field_path, raw_path, normalized_root)
    return normalized_path


__all__ = [
    "PYPROJECT_FILENAME",
    "ProjectRootBoundaryError",
    "RelativePathEscapeError",
    "normalize_local_path",
    "resolve_project_root",
]
