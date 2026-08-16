"""Streamlit page for discovering and inspecting immutable data snapshots.

The page is deliberately a presentation adapter.  It talks to the typed
``ResearchApplication`` facade only, uses application snapshot DTOs and the
redacted presenters, and keeps ordinary artifact access paged.  In particular,
it never reads the local data root or attempts to replace a published snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any

from ...application.services import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    ResearchApplication,
)
from ...application.snapshots import SnapshotQuery
from ...config.serializer import Redactor
from ..components import (
    bounded_page_size,
    render_actionable_errors,
    render_artifact_download,
    render_limitation_disclosure,
    render_table_page,
)
from ..presenters import present_snapshot_detail

_PLACEHOLDER = "Select a checksum-verified snapshot"
_MAX_REFERENCE_OPTIONS = 100


def _ui_module(st_module: Any | None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as streamlit  # noqa: PLC0415

    return streamlit


def _call(ui: Any, name: str, *args: object, **kwargs: object) -> object:
    method = getattr(ui, name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except TypeError:
        # Keep small AppTest/local doubles useful without changing production
        # Streamlit behavior when a widget does not support an optional kwarg.
        return method(*args)


def _field(value: object, names: str | Sequence[str], default: object = None) -> object:
    candidates = (names,) if isinstance(names, str) else tuple(names)
    if isinstance(value, Mapping):
        for name in candidates:
            if name in value:
                return value[name]
        return default
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _items(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def _enum_text(value: object, default: str = "") -> str:
    value = getattr(value, "value", value)
    return str(value) if value is not None else default


def _text(value: object, redactor: Redactor | None = None) -> str:
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    elif isinstance(value, Enum):
        value = value.value
    result = str(value) if value is not None else ""
    return redactor.redact_text(result) if redactor is not None else result


def _error(
    operation: str, error: BaseException, redactor: Redactor | None
) -> ActionableError:
    message = str(error) or "The snapshot view encountered an unexpected error."
    if redactor is not None:
        message = redactor.redact_text(message)
    return ActionableError(
        operation=operation,
        category=ErrorCategory.INTERNAL_UNEXPECTED,
        message=message,
        corrective_action=(
            "Retry the view; if the problem persists, inspect the local diagnostic log."
        ),
    )


def _render_errors(
    errors: Iterable[ActionableError],
    *,
    ui: Any,
    redactor: Redactor | None,
) -> None:
    values = tuple(error for error in errors if isinstance(error, ActionableError))
    if values:
        render_actionable_errors(values, st_module=ui, redactor=redactor)


def _summary_row(summary: object, redactor: Redactor | None) -> dict[str, object]:
    """Build a bounded, scalar-only discovery row from a SnapshotSummary."""

    requested = _field(summary, "requested_range")
    covered = _field(summary, "covered_range")
    validation = _field(summary, "validation_summary")
    integrity_error = _field(summary, "integrity_error")
    return {
        "snapshot_id": _text(_field(summary, "snapshot_id"), redactor),
        "provider": _text(_field(summary, "provider"), redactor),
        "requested_start": _text(_field(requested, "start"), redactor),
        "requested_end": _text(_field(requested, "end"), redactor),
        "covered_start": _text(_field(covered, "start"), redactor),
        "covered_end": _text(_field(covered, "end"), redactor),
        "universe": ", ".join(
            _text(symbol, redactor)
            for symbol in _items(_field(summary, "configured_universe", ()))
        ),
        "benchmark": _text(_field(summary, "benchmark_symbol"), redactor),
        "comparison_ready": bool(_field(summary, "comparison_ready", False)),
        "availability": _enum_text(_field(summary, "availability"), "unknown"),
        "accepted_rows": _field(validation, "accepted_row_count", 0),
        "quarantined_rows": _field(validation, "quarantined_row_count", 0),
        "gaps": _field(validation, "gap_count", 0),
        "stale_symbols": ", ".join(
            _text(symbol, redactor)
            for symbol in _items(_field(validation, "stale_symbols", ()))
        ),
        "created_at": _text(_field(summary, "created_at"), redactor),
        "integrity": (
            "unverified: " + _text(integrity_error, redactor)
            if integrity_error
            else "verified on inspection"
        ),
    }


def _safe_summary_disclosure(
    summaries: Sequence[object], redactor: Redactor | None
) -> LimitationDisclosure:
    for summary in summaries:
        disclosure = _field(summary, "limitation_disclosure")
        if isinstance(disclosure, LimitationDisclosure):
            return disclosure
    # The disclosure is part of every real SnapshotSummary.  Keeping a current
    # fallback makes an empty/error discovery view explicit as well.
    del redactor
    return LimitationDisclosure.current()


def _reference_details(detail: object) -> tuple[dict[str, object], ...]:
    """Return deterministic, bounded metadata for manifest object references."""

    provenance = _field(detail, "provenance")
    references = list(_items(_field(provenance, "object_references", ())))
    values: list[dict[str, object]] = []
    for reference in references:
        checksum = _field(reference, "checksum")
        if not isinstance(checksum, str) or not checksum:
            continue
        kind = _field(reference, ("object_kind", "role", "kind"), "artifact")
        values.append(
            {
                "checksum": checksum,
                "role": _enum_text(kind, "artifact"),
                "uri": _field(reference, ("relative_uri", "uri"), ""),
                "schema_version": _field(reference, "schema_version", ""),
                "row_count": _field(reference, "row_count", 0),
                "byte_size": _field(reference, ("byte_size", "size"), 0),
            }
        )
    values.sort(key=lambda item: (str(item["role"]), str(item["checksum"])))
    return tuple(values[:_MAX_REFERENCE_OPTIONS])


def _validation_view(rendered: Mapping[str, object]) -> dict[str, object]:
    validation = rendered.get("validation")
    if not isinstance(validation, Mapping):
        return {"summary": validation}
    # Keep the ordinary view to compact validation facts.  Full validation
    # artifacts remain available through the separate paged/download paths.
    names = (
        "accepted_row_count",
        "quarantined_row_count",
        "collapsed_duplicate_count",
        "gap_count",
        "failed_symbols",
        "retained_parent_coverage_symbols",
        "stale_symbols",
        "covered_range",
        "comparison_ready",
        "quarantined_by_reason",
        "gaps",
        "reasons",
    )
    return {name: validation[name] for name in names if name in validation}


def _bounded_provenance_view(rendered: Mapping[str, object]) -> dict[str, object]:
    provenance = rendered.get("provenance")
    if not isinstance(provenance, Mapping):
        return {"provenance": provenance}
    names = (
        "provider",
        "requested_range",
        "covered_range",
        "configured_universe",
        "benchmark_symbol",
        "calendar",
        "schema_versions",
        "configuration_checksum",
        "validation_report_checksum",
        "created_at",
        "parent_snapshot_id",
        "operation_id",
    )
    result = {name: provenance[name] for name in names if name in provenance}
    references = provenance.get("object_references")
    if isinstance(references, (list, tuple)):
        result["object_reference_count"] = len(references)
    requests = provenance.get("provider_requests")
    if isinstance(requests, (list, tuple)):
        result["provider_request_count"] = len(requests)
    return result


def _render_verified_artifact(
    application: ResearchApplication,
    checksum: str,
    *,
    label: str,
    ui: Any,
    redactor: Redactor | None,
) -> dict[str, object]:
    """Verify and expose a download only after an explicit user action."""

    clicked = bool(_call(ui, "button", label, key=f"verify-download-{checksum}"))
    if not clicked:
        return {"checksum": checksum, "verified": False}
    try:
        opened = application.open_artifact(checksum)
    except Exception as error:  # The facade normally converts this to Err.
        _render_errors(
            (_error("artifact.verify", error, redactor),),
            ui=ui,
            redactor=redactor,
        )
        return {"checksum": checksum, "verified": False}
    if isinstance(opened, Err):
        _render_errors(opened.errors, ui=ui, redactor=redactor)
        return {"checksum": checksum, "verified": False}
    # The shared component consumes the lazy stream only for this explicit
    # download affordance; merely selecting an artifact never reads its bytes.
    rendered = render_artifact_download(
        opened.value,
        label=f"Download verified {label.removeprefix('Verify and prepare ')}",
        st_module=ui,
        redactor=redactor,
    )
    return {"checksum": checksum, "verified": True, "download": rendered}


def _render_detail(
    application: ResearchApplication,
    detail: object,
    *,
    ui: Any,
    configured_page_size: int,
    redactor: Redactor | None,
) -> dict[str, object]:
    try:
        rendered = present_snapshot_detail(detail, redactor=redactor)
    except Exception as error:
        _render_errors(
            (_error("snapshot.present", error, redactor),),
            ui=ui,
            redactor=redactor,
        )
        return {}

    snapshot_id = _text(rendered.get("snapshot_id"), redactor)
    _call(ui, "subheader", "Verified snapshot details")
    _call(
        ui,
        "success",
        f"Snapshot {snapshot_id} is checksum-verified and ready for inspection.",
    )
    _call(ui, "json", {"snapshot_id": snapshot_id, "summary": rendered.get("summary")})

    _call(ui, "subheader", "Provenance and reproducibility")
    _call(ui, "json", _bounded_provenance_view(rendered))

    _call(ui, "subheader", "Validation and comparison readiness")
    _call(ui, "json", _validation_view(rendered))
    readiness = rendered.get("readiness")
    if isinstance(readiness, Mapping):
        _call(ui, "json", {"readiness": readiness})
    comparison_ready = bool(rendered.get("comparison_ready", False))
    if comparison_ready:
        _call(
            ui,
            "success",
            "Benchmark comparison is ready for this snapshot's reported coverage.",
        )
    else:
        _call(
            ui,
            "warning",
            "Benchmark comparison is not ready; review the recorded gaps or "
            "failures before backtesting.",
        )

    refs = _reference_details(detail)
    provenance = _field(detail, "provenance")
    validation_checksum = _field(provenance, "validation_report_checksum")
    artifact_options: list[str] = []
    if isinstance(validation_checksum, str) and validation_checksum:
        artifact_options.append(validation_checksum)
    artifact_options.extend(
        str(item["checksum"])
        for item in refs
        if str(item["checksum"]) not in artifact_options
    )
    artifact_options = artifact_options[:_MAX_REFERENCE_OPTIONS]

    if artifact_options:
        _call(ui, "subheader", "Verified snapshot artifacts")
        _call(
            ui,
            "caption",
            "Choose one manifest-referenced object. Ordinary views are paged; "
            "complete bytes are available only through an explicit verified "
            "download.",
        )
        descriptors = {str(item["checksum"]): item for item in refs}
        if isinstance(validation_checksum, str) and validation_checksum:
            descriptors.setdefault(
                validation_checksum,
                {
                    "checksum": validation_checksum,
                    "role": "validation_report",
                    "uri": "content-addressed validation artifact",
                    "schema_version": "validation_report_v1",
                    "row_count": None,
                    "byte_size": None,
                },
            )
        selected = _call(
            ui,
            "selectbox",
            "Artifact to inspect",
            tuple(artifact_options),
            index=0,
            key=f"snapshot-artifact-{snapshot_id}",
        )
        selected_checksum = str(selected) if selected in artifact_options else None
        if selected_checksum is not None:
            _call(ui, "json", {"selected_artifact": descriptors[selected_checksum]})
            detail_page_raw = _call(
                ui,
                "number_input",
                "Artifact page",
                value=0,
                min_value=0,
                step=1,
                key=f"snapshot-artifact-page-{selected_checksum}",
            )
            detail_page = (
                int(detail_page_raw)
                if isinstance(detail_page_raw, int)
                and not isinstance(detail_page_raw, bool)
                else 0
            )
            page_clicked = bool(
                _call(
                    ui,
                    "button",
                    "Load artifact page",
                    key=f"snapshot-artifact-load-{selected_checksum}",
                )
            )
            if page_clicked:
                try:
                    page_result = application.page_artifact(
                        selected_checksum,
                        page=detail_page,
                        page_size=configured_page_size,
                    )
                except Exception as error:
                    page_result = Err((_error("artifact.page", error, redactor),))
                if isinstance(page_result, Err):
                    _render_errors(page_result.errors, ui=ui, redactor=redactor)
                else:
                    render_table_page(
                        page_result.value,
                        configured_page_size=configured_page_size,
                        requested_page_size=configured_page_size,
                        title="Artifact page",
                        st_module=ui,
                        redactor=redactor,
                    )
            download_result = _render_verified_artifact(
                application,
                selected_checksum,
                label="Verify and prepare selected artifact download",
                ui=ui,
                redactor=redactor,
            )
        else:
            download_result = {}
    else:
        download_result = {}

    disclosure = _field(detail, "limitation_disclosure")
    if isinstance(disclosure, LimitationDisclosure):
        render_limitation_disclosure(disclosure, st_module=ui, redactor=redactor)
    else:
        # ``present_snapshot_detail`` already enforces this in normal operation;
        # retain a visible disclosure if a lightweight test DTO is incomplete.
        render_limitation_disclosure(
            LimitationDisclosure.current(), st_module=ui, redactor=redactor
        )

    _call(
        ui,
        "info",
        "Published snapshots are immutable. To correct or revise data, publish "
        "a new Data_Snapshot; existing Snapshot_IDs and their checksums remain "
        "valid for prior research.",
    )
    return {
        "snapshot_id": snapshot_id,
        "snapshot": rendered,
        "artifact": download_result,
    }


def render_snapshots(
    application: ResearchApplication,
    *,
    st_module: Any | None = None,
    configured_page_size: int = 100,
    redactor: Redactor | None = None,
) -> dict[str, object]:
    """Render bounded snapshot discovery and checksum-verified inspection.

    ``configured_page_size`` is clamped server-side to the absolute 100-row
    ordinary-table limit.  The returned mapping is a redacted presentation
    value, which makes the function convenient for AppTest/local UI doubles
    without exposing application handles or raw artifact streams.
    """

    ui = _ui_module(st_module)
    limit = bounded_page_size(configured_page_size)
    _call(ui, "title", "Snapshots")
    _call(
        ui,
        "caption",
        "Discover immutable, checksummed data snapshots without loading "
        "complete artifacts.",
    )

    provider_value = _call(
        ui,
        "text_input",
        "Provider filter (optional)",
        value="",
        key="snapshot-provider-filter",
    )
    provider = str(provider_value).strip() if provider_value is not None else ""
    availability_value = _call(
        ui,
        "selectbox",
        "Availability",
        ("All", "available", "unavailable", "invalid"),
        index=0,
        key="snapshot-availability-filter",
    )
    availability = (
        None
        if availability_value in (None, "All")
        else str(availability_value).strip().lower()
    )
    page_value = _call(
        ui,
        "number_input",
        "Snapshot page",
        value=0,
        min_value=0,
        step=1,
        key="snapshot-page",
    )
    page = (
        int(page_value)
        if isinstance(page_value, int) and not isinstance(page_value, bool)
        else 0
    )
    requested_size_value = _call(
        ui,
        "number_input",
        "Snapshots per page",
        value=limit,
        min_value=1,
        max_value=limit,
        step=1,
        key="snapshot-page-size",
    )
    requested_size = (
        int(requested_size_value)
        if isinstance(requested_size_value, int)
        and not isinstance(requested_size_value, bool)
        else limit
    )
    page_size = bounded_page_size(requested_size, limit)

    try:
        query = SnapshotQuery(
            provider=provider or None,
            availability=availability,
            page=page,
            page_size=page_size,
        )
        page_result = application.list_snapshots(query)
    except Exception as error:
        _render_errors(
            (_error("snapshot.list", error, redactor),),
            ui=ui,
            redactor=redactor,
        )
        disclosure = LimitationDisclosure.current()
        render_limitation_disclosure(disclosure, st_module=ui, redactor=redactor)
        _call(
            ui,
            "info",
            "Published snapshots are immutable; publish a new snapshot for "
            "corrections.",
        )
        return {"query": {"page": page, "page_size": page_size}, "items": ()}

    page_errors = _items(_field(page_result, "errors", ()))
    _render_errors(page_errors, ui=ui, redactor=redactor)
    summaries = tuple(_items(_field(page_result, ("items", "records"), ())))
    rows = tuple(_summary_row(summary, redactor) for summary in summaries)
    if rows:
        render_table_page(
            SimpleNamespace(
                rows=rows,
                page=page,
                page_size=page_size,
                total=_field(page_result, ("total", "total_count")),
                columns=tuple(rows[0].keys()),
            ),
            configured_page_size=limit,
            requested_page_size=page_size,
            title="Published snapshots",
            st_module=ui,
            redactor=redactor,
        )
    else:
        _call(ui, "info", "No published snapshots match the selected filters.")

    has_next = bool(_field(page_result, "has_next", False))
    _call(
        ui,
        "caption",
        f"Snapshot page {page + 1} · showing at most {page_size} rows"
        + (" · more pages available" if has_next else ""),
    )
    render_limitation_disclosure(
        _safe_summary_disclosure(summaries, redactor),
        st_module=ui,
        redactor=redactor,
    )

    available: list[object] = []
    blocked: list[object] = []
    for summary in summaries:
        availability_text = _enum_text(_field(summary, "availability"), "unknown")
        integrity_error = _field(summary, "integrity_error")
        if availability_text == "available" and not integrity_error:
            available.append(summary)
        else:
            blocked.append(summary)
    if blocked:
        _call(
            ui,
            "warning",
            "Unavailable, invalid, or integrity-failed snapshots are shown for "
            "diagnostics but cannot be selected for use.",
        )

    options = (_PLACEHOLDER,) + tuple(
        _text(_field(summary, "snapshot_id"), redactor) for summary in available
    )
    selected = _call(
        ui,
        "selectbox",
        "Snapshot to inspect",
        options,
        index=0,
        key="snapshot-selection",
    )
    selected_id = str(selected) if selected in options[1:] else None
    detail_view: dict[str, object] = {}
    if selected_id is not None:
        _call(
            ui,
            "caption",
            "Selection is checked against the immutable manifest before details "
            "are shown.",
        )
        try:
            inspected = application.inspect_snapshot(selected_id)
        except Exception as error:
            inspected = Err((_error("snapshot.inspect", error, redactor),))
        if isinstance(inspected, Err):
            _render_errors(inspected.errors, ui=ui, redactor=redactor)
            _call(
                ui,
                "warning",
                "This snapshot cannot be used because checksum verification did "
                "not succeed. Select another published Snapshot_ID or publish "
                "a new snapshot.",
            )
        else:
            detail_view = _render_detail(
                application,
                inspected.value,
                ui=ui,
                configured_page_size=limit,
                redactor=redactor,
            )

    return {
        "query": {
            "provider": provider or None,
            "availability": availability,
            "page": page,
            "page_size": page_size,
        },
        "rows": rows,
        "selected_snapshot_id": selected_id,
        "detail": detail_view,
    }


render_snapshots_page = render_snapshots
render_page = render_snapshots

__all__ = ["render_page", "render_snapshots", "render_snapshots_page"]
