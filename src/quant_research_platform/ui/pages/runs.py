"""Run discovery and immutable inspection page.

The page uses only :class:`ResearchApplication` contracts.  Discovery is
bounded by ``RunQuery``; manifests and artifact bytes are obtained only through
checksum-verifying application methods and are rendered through redacted
presenters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Any

from ...application.services import (
    ActionableError,
    Err,
    LimitationDisclosure,
    Ok,
    ResearchApplication,
    RunQuery,
    RunState,
)
from ...config.serializer import Redactor
from ..components import (
    bounded_page_size,
    render_actionable_errors,
    render_artifact_download,
    render_limitation_disclosure,
    render_table_page,
)
from ..presenters import present_artifact, present_run_detail

_STATE_PAGE = "qrp.runs.page"
_STATE_SELECTED = "qrp.runs.selected_run_id"
_STATE_DETAIL = "qrp.runs.detail"
_STATE_ERRORS = "qrp.runs.errors"
_STATE_ARTIFACT_PAGE = "qrp.runs.artifact_page"


def _call(ui: Any, name: str, *args: object, **kwargs: object) -> object:
    method = getattr(ui, name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except TypeError:
        return method(*args)


def _state(ui: Any) -> Any:
    state = getattr(ui, "session_state", None)
    if state is None:
        state = {}
        with suppress(Exception):
            ui.session_state = state
    return state


def _get(state: Any, key: str, default: object = None) -> object:
    try:
        return state.get(key, default)
    except AttributeError:
        try:
            return state[key]
        except (KeyError, TypeError):
            return default


def _set(state: Any, key: str, value: object) -> None:
    try:
        state[key] = value
    except (AttributeError, TypeError):
        setattr(state, key, value)


def _delete(state: Any, key: str) -> None:
    try:
        state.pop(key, None)
    except AttributeError:
        with suppress(KeyError, TypeError):
            del state[key]


def _items(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def _actionable_errors(value: object) -> tuple[ActionableError, ...]:
    """Keep diagnostic rendering on the structured application-error boundary."""

    if isinstance(value, Err):
        return value.errors
    values = _items(value)
    return tuple(item for item in values if isinstance(item, ActionableError))


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


def _configured_page_size(state: Any) -> int:
    view = _get(state, "qrp.configuration_view")
    configuration = _field(view, "configuration", view)
    ui_config = _field(configuration, "ui", configuration)
    value = _field(ui_config, "page_size", 100)
    return bounded_page_size(value if isinstance(value, int) else 100)


def _state_value(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def _run_id(value: object) -> str:
    return str(_field(value, "run_id", ""))


def _render_discovery(
    application: ResearchApplication,
    ui: Any,
    state: Any,
    page_size: int,
) -> tuple[object, ...]:
    run_id_value = _call(
        ui,
        "text_input",
        "Run_ID (optional)",
        value="",
        key="qrp-runs-run-id",
    )
    snapshot_value = _call(
        ui,
        "text_input",
        "Snapshot_ID (optional)",
        value="",
        key="qrp-runs-snapshot-id",
    )
    strategy_value = _call(
        ui,
        "text_input",
        "Strategy identifier (optional)",
        value="",
        key="qrp-runs-strategy",
    )
    state_options = ("All states", "running", "succeeded", "failed")
    selected_state = _call(
        ui,
        "selectbox",
        "Run state",
        state_options,
        index=0,
        key="qrp-runs-state",
    )
    page_value = _get(state, _STATE_PAGE, 0)
    page_number = page_value if isinstance(page_value, int) and page_value >= 0 else 0
    query_page = _call(
        ui,
        "number_input",
        "Run results page",
        value=page_number,
        min_value=0,
        step=1,
        key="qrp-runs-page-input",
    )
    if isinstance(query_page, int) and not isinstance(query_page, bool):
        page_number = max(0, query_page)
    _set(state, _STATE_PAGE, page_number)

    normalized_state = None
    state_text = str(selected_state) if selected_state else state_options[0]
    if state_text != state_options[0]:
        normalized_state = RunState(state_text)
    query = RunQuery(
        run_id=str(run_id_value).strip() or None,
        snapshot_id=str(snapshot_value).strip() or None,
        strategy_id=str(strategy_value).strip() or None,
        state=normalized_state,
        page=page_number,
        page_size=page_size,
    )
    page = application.search_runs(query)
    if page.errors:
        render_actionable_errors(page.errors, st_module=ui)
    rows = [
        {
            "run_id": _run_id(item),
            "snapshot_id": _field(item, "snapshot_id", ""),
            "state": _state_value(_field(item, "state", "")),
            "strategy": _field(item, "strategy_id", ""),
            "evaluation_start": str(_field(item, "evaluation_start", "")),
            "evaluation_end": str(_field(item, "evaluation_end", "")),
        }
        for item in page.items
    ]
    _call(ui, "subheader", "Run discovery")
    if rows:
        _call(ui, "dataframe", rows, hide_index=True, use_container_width=True)
    else:
        _call(ui, "info", "No runs match the selected filters.")
    previous_clicked = bool(
        _call(ui, "button", "Previous run page", key="qrp-runs-previous")
    )
    next_clicked = bool(_call(ui, "button", "Next run page", key="qrp-runs-next"))
    if previous_clicked:
        _set(state, _STATE_PAGE, max(0, page_number - 1))
    elif next_clicked and page.has_next:
        _set(state, _STATE_PAGE, page_number + 1)
    _call(
        ui,
        "caption",
        "Run page "
        f"{page_number + 1}; at most {page_size} indexed summaries are loaded.",
    )
    return page.items


def _render_artifacts(
    application: ResearchApplication,
    ui: Any,
    detail: object,
    state: Any,
    page_size: int,
    *,
    redactor: Redactor | None,
) -> None:
    values = _items(_field(detail, ("artifacts", "artifact_metadata")))
    if not values:
        return
    metadata = [present_artifact(item, redactor=redactor) for item in values]
    _call(ui, "subheader", "Verified run artifacts")
    _call(ui, "dataframe", metadata, hide_index=True, use_container_width=True)
    for index, artifact in enumerate(values):
        rendered = present_artifact(artifact, redactor=redactor)
        checksum = rendered.get("checksum")
        if not isinstance(checksum, str):
            continue
        role = str(rendered.get("role", f"artifact-{index}"))
        clicked = bool(
            _call(
                ui,
                "button",
                f"Verify and prepare download: {role}",
                key=f"qrp-runs-download-{checksum}",
            )
        )
        if clicked:
            opened = application.open_artifact(checksum)
            if isinstance(opened, Ok):
                render_artifact_download(
                    opened.value,
                    label=f"Download {role}",
                    st_module=ui,
                    redactor=redactor,
                )
            else:
                render_actionable_errors(opened.errors, st_module=ui, redactor=redactor)

    table_candidates = []
    for artifact in values:
        rendered = present_artifact(artifact, redactor=redactor)
        role = str(rendered.get("role", ""))
        checksum = rendered.get("checksum")
        if role and isinstance(checksum, str):
            table_candidates.append((role, checksum))
    if not table_candidates:
        return
    options = tuple(role for role, _ in table_candidates)
    selected_role = _call(
        ui,
        "selectbox",
        "Ordinary run artifact table (bounded page)",
        options,
        index=0,
        key="qrp-runs-artifact-role",
    )
    role = str(selected_role) if selected_role else options[0]
    checksum = dict(table_candidates).get(role)
    if checksum is None:
        return
    page_value = _get(state, _STATE_ARTIFACT_PAGE, 0)
    page_number = page_value if isinstance(page_value, int) and page_value >= 0 else 0
    page_input = _call(
        ui,
        "number_input",
        "Artifact page",
        value=page_number,
        min_value=0,
        step=1,
        key="qrp-runs-artifact-page",
    )
    if isinstance(page_input, int) and not isinstance(page_input, bool):
        page_number = max(0, page_input)
        _set(state, _STATE_ARTIFACT_PAGE, page_number)
    load_clicked = bool(
        _call(
            ui,
            "button",
            "Load verified run artifact page",
            key="qrp-runs-load-page",
        )
    )
    if load_clicked:
        page_result = application.page_artifact(
            checksum,
            page=page_number,
            page_size=page_size,
        )
        if isinstance(page_result, Ok):
            render_table_page(
                page_result.value,
                configured_page_size=page_size,
                st_module=ui,
                redactor=redactor,
            )
        else:
            render_actionable_errors(
                page_result.errors,
                st_module=ui,
                redactor=redactor,
            )


def _render_detail(
    application: ResearchApplication,
    ui: Any,
    state: Any,
    detail: object,
    page_size: int,
    *,
    redactor: Redactor | None,
) -> None:
    try:
        rendered = present_run_detail(detail, redactor=redactor)
    except (TypeError, ValueError):
        _call(ui, "error", "The selected run could not be rendered safely.")
        return
    summary = _field(detail, "summary")
    run_state = _state_value(_field(summary, "state", ""))
    run_id = _run_id(summary or detail)
    _call(ui, "subheader", f"Run inspection: {run_id}")
    if run_state in {RunState.SUCCEEDED.value, RunState.FAILED.value}:
        _call(
            ui,
            "info",
            (
                "This is a terminal Run. Inputs, metrics, lifecycle state, manifest, "
                "and artifacts are immutable; create a new Run_ID for another "
                "execution."
            ),
        )
    elif run_state == RunState.RUNNING.value:
        _call(
            ui,
            "warning",
            "This Run is still running; terminal inspection is not yet immutable.",
        )
    _call(ui, "json", rendered)
    _call(ui, "subheader", "Manifest")
    _call(ui, "json", rendered.get("manifest"))
    _call(ui, "subheader", "Redacted configuration and environment fingerprint")
    _call(
        ui,
        "json",
        {
            "configuration": rendered.get("configuration"),
            "environment_fingerprint": rendered.get("environment_fingerprint"),
        },
    )
    _call(ui, "subheader", "Validation report and logs")
    _call(
        ui,
        "json",
        {"validation": rendered.get("validation"), "logs": rendered.get("logs")},
    )
    _render_artifacts(application, ui, detail, state, page_size, redactor=redactor)
    disclosure = _field(detail, "limitation_disclosure")
    if isinstance(disclosure, LimitationDisclosure):
        render_limitation_disclosure(disclosure, st_module=ui, redactor=redactor)


def render_runs(
    application: ResearchApplication,
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> None:
    """Render bounded run discovery, inspection, and verified artifact access."""

    if st_module is None:
        import streamlit as st  # noqa: PLC0415

        ui = st
    else:
        ui = st_module
    state = _state(ui)
    page_size = _configured_page_size(state)
    _call(ui, "title", "Runs")
    _call(
        ui,
        "caption",
        "Searches use bounded RunQuery pages; terminal runs are immutable.",
    )
    records = _render_discovery(application, ui, state, page_size)
    options = tuple(_run_id(item) for item in records if _run_id(item))
    if not options:
        _delete(state, _STATE_SELECTED)
        _delete(state, _STATE_DETAIL)
        return
    selected_value = _call(
        ui,
        "selectbox",
        "Run_ID to inspect",
        options,
        index=0,
        key="qrp-runs-selected",
    )
    selected = str(selected_value) if selected_value else options[0]
    if _get(state, _STATE_SELECTED) != selected:
        _set(state, _STATE_SELECTED, selected)
        _delete(state, _STATE_DETAIL)
    if bool(_call(ui, "button", "Inspect selected run", key="qrp-runs-inspect")):
        inspected = application.inspect_run(selected)
        if isinstance(inspected, Ok):
            _set(state, _STATE_DETAIL, inspected.value)
            _delete(state, _STATE_ERRORS)
        else:
            _set(state, _STATE_ERRORS, inspected.errors)
            render_actionable_errors(inspected.errors, st_module=ui, redactor=redactor)
            _call(
                ui,
                "warning",
                "The selected run remains discoverable; checksum-corrupt artifacts "
                "were not opened.",
            )

    detail = _get(state, _STATE_DETAIL)
    if detail is not None and not _actionable_errors(_get(state, _STATE_ERRORS)):
        _render_detail(application, ui, state, detail, page_size, redactor=redactor)
    errors = _actionable_errors(_get(state, _STATE_ERRORS))
    if errors:
        render_actionable_errors(errors, st_module=ui, redactor=redactor)


render_runs_page = render_runs
render = render_runs

__all__ = ["render", "render_runs", "render_runs_page"]
