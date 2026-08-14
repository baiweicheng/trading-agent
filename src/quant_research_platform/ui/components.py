"""Reusable, bounded Streamlit view components.

Components are intentionally thin presentation helpers.  They accept application
DTOs or presenter mappings, never inspect Streamlit session state, and import
Streamlit lazily so unit tests can use a small local UI double without starting a
server or contacting an external provider.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..application.services import ActionableError, Err, LimitationDisclosure
from ..config.serializer import Redactor
from .presenters import (
    Presentation,
    present_artifact,
    present_errors,
    present_limitation_disclosure,
    present_metrics,
    present_progress,
    present_table_page,
)


def _streamlit_module(st_module: Any | None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as streamlit  # noqa: PLC0415

    return streamlit


def _call_ui(ui: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(ui, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except TypeError:
        # Small test doubles often implement only the positional Streamlit
        # portion.  Falling back does not alter production behavior.
        return method(*args)


def _items(value: object) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def bounded_page_size(
    requested_page_size: int | None = None, configured_page_size: int = 100
) -> int:
    """Return the server-side ordinary-table bound, never greater than 100."""

    if isinstance(configured_page_size, bool) or configured_page_size < 1:
        raise ValueError("configured_page_size must be positive")
    if requested_page_size is not None and (
        isinstance(requested_page_size, bool) or requested_page_size < 1
    ):
        raise ValueError("requested_page_size must be positive or None")
    limit = min(100, configured_page_size)
    return min(limit, requested_page_size) if requested_page_size is not None else limit


def render_limitation_disclosure(
    disclosure: LimitationDisclosure,
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Keep the complete limitation disclosure visible on the current view."""

    rendered = present_limitation_disclosure(disclosure, redactor=redactor)
    ui = _streamlit_module(st_module)
    _call_ui(ui, "subheader", "Limitations and assumptions")
    text = "\n".join(str(line) for line in _items(rendered["lines"]))
    if _call_ui(ui, "info", text) is None:
        _call_ui(ui, "markdown", text)
    return rendered


def render_actionable_errors(
    errors: Err | Sequence[ActionableError],
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> list[Presentation]:
    """Render only structured, sanitized diagnostics."""

    rendered = present_errors(errors, redactor=redactor)
    ui = _streamlit_module(st_module)
    for error in rendered:
        text = (
            f"{error['operation']} [{error['category']}]: {error['message']} "
            f"Corrective action: {error['corrective_action']}"
        )
        _call_ui(ui, "error", text)
    return rendered


def render_progress(
    progress: object,
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Render bounded job progress, including sanitized accumulated warnings."""

    rendered = present_progress(progress, redactor=redactor)
    ui = _streamlit_module(st_module)
    completed = rendered["completed_units"]
    total = rendered["total_units"]
    ratio = 0.0
    if isinstance(completed, int) and isinstance(total, int) and total > 0:
        ratio = min(1.0, max(0.0, completed / total))
    stage = rendered["stage"]
    state = rendered["state"]
    _call_ui(ui, "progress", ratio, text=f"{stage} ({state})")
    _call_ui(
        ui,
        "caption",
        f"Completed {completed} of {total if total is not None else '?'} units",
    )
    for warning in _items(rendered["warnings"]):
        _call_ui(ui, "warning", warning)
    return rendered


def _metric_rows(rendered: Presentation) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if "metrics" in rendered:
        scope = rendered.get("scope")
        for metric in _items(rendered.get("metrics")):
            if not isinstance(metric, Mapping):
                continue
            row = dict(metric)
            if scope is not None:
                row["scope"] = scope
            rows.append(row)
    else:
        for scope, values in rendered.items():
            if not isinstance(values, Mapping) or "metrics" not in values:
                continue
            for metric in _items(values.get("metrics")):
                if not isinstance(metric, Mapping):
                    continue
                row = dict(metric)
                row["scope"] = scope
                rows.append(row)
    return rows


def render_metrics(
    metrics: object,
    *,
    title: str | None = None,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Render metric values while retaining explicit null reasons."""

    rendered = present_metrics(metrics, redactor=redactor)
    ui = _streamlit_module(st_module)
    if title:
        _call_ui(ui, "subheader", title)
    rows = _metric_rows(rendered)
    if rows:
        _call_ui(ui, "dataframe", rows, use_container_width=True, hide_index=True)
    else:
        _call_ui(ui, "info", "No metrics are available for this view.")
    return rendered


def render_chart(
    chart_spec: Mapping[str, object],
    *,
    title: str | None = None,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> dict[str, object]:
    """Render an application-produced canonical Vega-Lite chart specification."""

    if not isinstance(chart_spec, Mapping):
        raise TypeError("chart_spec must be a mapping")
    chart_value: object = dict(chart_spec)
    if redactor is not None:
        chart_value = redactor.redact_structured(chart_value)
    if not isinstance(chart_value, Mapping):
        raise TypeError("redaction must preserve a chart mapping")
    chart = dict(chart_value)
    ui = _streamlit_module(st_module)
    if title:
        _call_ui(ui, "subheader", title)
    _call_ui(ui, "vega_lite_chart", chart, use_container_width=True)
    return chart


render_canonical_chart = render_chart


def render_table_page(
    page: object,
    *,
    configured_page_size: int = 100,
    requested_page_size: int | None = None,
    title: str | None = None,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Render one ordinary table page without materializing additional rows."""

    limit = bounded_page_size(requested_page_size, configured_page_size)
    rendered = present_table_page(
        page,
        configured_page_size=configured_page_size,
        requested_page_size=limit,
        redactor=redactor,
    )
    ui = _streamlit_module(st_module)
    if title:
        _call_ui(ui, "subheader", title)
    rows = _items(rendered["rows"])
    if rows:
        _call_ui(ui, "dataframe", rows, use_container_width=True, hide_index=True)
    else:
        _call_ui(ui, "info", "No rows are available for this page.")
    page_value = rendered["page"]
    page_number = page_value if isinstance(page_value, int) else 0
    _call_ui(
        ui,
        "caption",
        f"Page {page_number + 1} · {rendered['row_count']} rows",
    )
    return rendered


# The explicit name makes the ordinary-table boundary visible at call sites.
render_paged_table = render_table_page


def _artifact_bytes(artifact: object, data: bytes | None) -> bytes | None:
    if data is not None:
        return data
    payload = getattr(artifact, "payload", None)
    if isinstance(payload, bytes):
        return payload
    stream = getattr(artifact, "stream", None)
    if callable(stream):
        chunks = stream()
        if isinstance(chunks, (bytes, bytearray)):
            return bytes(chunks)
        return b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))
    return None


def render_artifact_download(
    artifact: object,
    *,
    data: bytes | None = None,
    label: str | None = None,
    file_name: str | None = None,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Render a separate explicit download affordance for a verified artifact.

    This function never turns an artifact into an ordinary table.  A caller can
    pass a lazy ``VerifiedArtifact`` returned by the application facade; the
    stream is consumed only for the explicit download action.
    """

    metadata = present_artifact(artifact, redactor=redactor)
    ui = _streamlit_module(st_module)
    valid = metadata.get(
        "valid", metadata.get("availability", "available") == "available"
    )
    if valid is False:
        _call_ui(
            ui,
            "warning",
            "This artifact is not checksum-verified and cannot be downloaded.",
        )
        return {"artifact": metadata, "downloaded": False}

    checksum = str(metadata.get("checksum", "artifact"))
    role = str(metadata.get("role", "artifact"))
    if file_name is None:
        uri = metadata.get("uri")
        file_name = PurePosixPath(str(uri)).name if uri else f"{role}.bin"
    payload = _artifact_bytes(artifact, data)
    if payload is None:
        _call_ui(
            ui,
            "info",
            "Artifact verified. Open it through the application facade to "
            "download it.",
        )
        return {"artifact": metadata, "downloaded": False}

    button_label = label or f"Download {role}"
    clicked = _call_ui(
        ui,
        "download_button",
        button_label,
        data=payload,
        file_name=file_name,
        mime=metadata.get("media_type", "application/octet-stream"),
        key=f"artifact-download-{checksum}",
    )
    return {
        "artifact": metadata,
        "downloaded": bool(clicked),
        "file_name": file_name,
    }


def render_artifact_downloads(
    artifacts: Iterable[object],
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> list[Presentation]:
    """Render independent download affordances for artifact metadata/handles."""

    return [
        render_artifact_download(
            artifact, st_module=st_module, redactor=redactor
        )
        for artifact in artifacts
    ]


# Short aliases used by page modules.
render_disclosure = render_limitation_disclosure
render_errors = render_actionable_errors
render_download = render_artifact_download

__all__ = [
    "bounded_page_size",
    "render_actionable_errors",
    "render_artifact_download",
    "render_artifact_downloads",
    "render_canonical_chart",
    "render_chart",
    "render_disclosure",
    "render_download",
    "render_errors",
    "render_limitation_disclosure",
    "render_metrics",
    "render_paged_table",
    "render_progress",
    "render_table_page",
]
