"""Backtest page for the synchronous, verified-snapshot workflow.

This module is a thin presentation adapter.  It discovers snapshots and invokes
only the typed :class:`ResearchApplication` facade; storage, provider clients,
checksum verification, configuration secrets, and run lifecycle rules remain in
application services.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast

from ...application.backtests import BacktestRequest
from ...application.services import (
    ActionableError,
    ConfigurationHandle,
    Err,
    LimitationDisclosure,
    Ok,
    ResearchApplication,
)
from ...application.snapshots import SnapshotQuery
from ...config.serializer import Redactor
from ..components import (
    bounded_page_size,
    render_actionable_errors,
    render_artifact_download,
    render_limitation_disclosure,
    render_metrics,
    render_progress,
    render_table_page,
)
from ..presenters import (
    present_artifact,
    present_backtest_result,
    present_evaluation_output,
    present_progress,
)

_STATE_VERIFIED_SNAPSHOT = "qrp.backtest.verified_snapshot_id"
_STATE_VERIFIED_DETAIL = "qrp.backtest.verified_snapshot_detail"
_STATE_PROGRESS = "qrp.backtest.progress"
_STATE_RESULT = "qrp.backtest.result"
_STATE_ERRORS = "qrp.backtest.errors"
_STATE_ARTIFACT_PAGE = "qrp.backtest.artifact_page"


def _call(ui: Any, name: str, *args: object, **kwargs: object) -> object:
    method = getattr(ui, name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except TypeError:
        # Small AppTest doubles often implement only Streamlit's positional
        # arguments.  This fallback does not change production behavior.
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


def _snapshot_id(value: object) -> str:
    return str(_field(value, "snapshot_id", ""))


def _snapshot_available(value: object) -> bool:
    availability = _field(value, "availability", "available")
    availability_text = str(getattr(availability, "value", availability)).lower()
    return availability_text == "available" and not bool(
        _field(value, "integrity_error", None)
    )


def _snapshot_verified(detail: object, selected_id: str) -> bool:
    """Validate the redacted detail returned by the facade verification boundary."""

    if _snapshot_id(detail) != selected_id:
        return False
    if bool(_field(detail, "integrity_error", None)):
        return False
    summary = _field(detail, "summary")
    readiness = _field(detail, "readiness", summary)
    available = _field(readiness, "available")
    comparison_ready = _field(readiness, "comparison_ready")
    # ``inspect_snapshot`` verifies the manifest and every referenced checksum.
    # Do not require a concrete handle attribute so structural test facades can
    # return a redacted detail projection.
    return not (available is False or comparison_ready is False)


def _render_snapshot_candidates(
    application: ResearchApplication,
    ui: Any,
    state: Any,
    page_size: int,
) -> str | None:
    page = application.list_snapshots(
        SnapshotQuery(availability="available", page=0, page_size=page_size)
    )
    if page.errors:
        render_actionable_errors(page.errors, st_module=ui)
    candidates = tuple(item for item in page.items if _snapshot_available(item))
    rows = [
        {
            "snapshot_id": _snapshot_id(item),
            "provider": _field(item, "provider", ""),
            "requested_range": str(_field(item, "requested_range", "")),
            "covered_range": str(_field(item, "covered_range", "")),
            "comparison_ready": bool(_field(item, "comparison_ready", False)),
        }
        for item in candidates
    ]
    if rows:
        _call(ui, "subheader", "Available snapshot candidates")
        _call(ui, "dataframe", rows, hide_index=True, use_container_width=True)
    else:
        _call(ui, "info", "No available snapshots are ready for a verified backtest.")
        _delete(state, _STATE_VERIFIED_SNAPSHOT)
        _delete(state, _STATE_VERIFIED_DETAIL)
        return None

    options = tuple(_snapshot_id(item) for item in candidates)
    selected_value = _call(
        ui,
        "selectbox",
        "Snapshot ID",
        options,
        index=0,
        key="qrp-backtest-snapshot",
    )
    selected = str(selected_value) if selected_value else options[0]
    if _get(state, _STATE_VERIFIED_SNAPSHOT) != selected:
        _delete(state, _STATE_VERIFIED_SNAPSHOT)
        _delete(state, _STATE_VERIFIED_DETAIL)
    _call(
        ui,
        "caption",
        "Select one Snapshot_ID, then verify its immutable manifest before running.",
    )
    return selected


def _row_values(values: object) -> tuple[Mapping[str, object], ...]:
    """Convert a DTO sequence to safe table-shaped rows without raw repr output."""

    rows: list[Mapping[str, object]] = []
    for value in _items(values):
        if isinstance(value, Mapping):
            rows.append(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            rows.append({"session": value[0], "value": value[1]})
        else:
            rows.append({"value": value})
    return tuple(rows)


def _render_bounded_table(
    ui: Any,
    title: str,
    values: object,
    page_size: int,
    *,
    redactor: Redactor | None,
) -> None:
    rows = _row_values(values)
    if not rows:
        return
    _call(ui, "subheader", title)
    render_table_page(
        SimpleNamespace(
            rows=rows,
            page=0,
            page_size=min(page_size, 100),
            total=len(rows),
        ),
        configured_page_size=page_size,
        st_module=ui,
        redactor=redactor,
    )


def _artifact_values(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    return _items(value)


def _render_artifacts(
    application: ResearchApplication,
    ui: Any,
    state: Any,
    artifacts: object,
    page_size: int,
    *,
    redactor: Redactor | None,
) -> None:
    values = _artifact_values(artifacts)
    if not values:
        return
    metadata = [present_artifact(item, redactor=redactor) for item in values]
    _call(ui, "subheader", "Verified result artifacts")
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
                key=f"qrp-backtest-download-{checksum}",
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

    table_roles = {
        "strategy_equity",
        "benchmark_equity",
        "drawdown",
        "positions",
        "orders",
        "fills",
        "decisions",
        "strategy_returns",
        "benchmark_returns",
        "monthly_returns",
    }
    table_artifacts = [
        (str(present_artifact(item, redactor=redactor).get("role", "")), item)
        for item in values
    ]
    table_artifacts = [item for item in table_artifacts if item[0] in table_roles]
    if not table_artifacts:
        return
    options = tuple(role for role, _ in table_artifacts)
    selected = _call(
        ui,
        "selectbox",
        "Ordinary artifact table (bounded page)",
        options,
        index=0,
        key="qrp-backtest-artifact-role",
    )
    selected_role = str(selected) if selected else options[0]
    selected_artifact = dict(table_artifacts).get(selected_role)
    if selected_artifact is None:
        return
    selected_metadata = present_artifact(selected_artifact, redactor=redactor)
    selected_checksum = selected_metadata.get("checksum")
    if not isinstance(selected_checksum, str):
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
        key="qrp-backtest-artifact-page",
    )
    if isinstance(page_input, int) and not isinstance(page_input, bool):
        page_number = max(0, page_input)
        _set(state, _STATE_ARTIFACT_PAGE, page_number)
    load_clicked = bool(
        _call(ui, "button", "Load verified artifact page", key="qrp-backtest-load-page")
    )
    if load_clicked:
        page_result = application.page_artifact(
            selected_checksum,
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
                page_result.errors, st_module=ui, redactor=redactor
            )


def _render_result(
    application: ResearchApplication,
    ui: Any,
    state: Any,
    result: object,
    page_size: int,
    *,
    redactor: Redactor | None,
) -> None:
    try:
        rendered = present_backtest_result(result, redactor=redactor)
    except (TypeError, ValueError):
        _call(ui, "error", "The backtest result could not be rendered safely.")
        return
    run_id = rendered.get("run_id", "")
    _call(ui, "success", f"Backtest completed. Run_ID: {run_id}")
    _call(ui, "json", rendered)

    evaluation = _field(result, "evaluation")
    if evaluation is not None:
        try:
            evaluation_view = present_evaluation_output(evaluation, redactor=redactor)
        except (TypeError, ValueError):
            evaluation_view = {"status": "Evaluation details unavailable."}
        _call(ui, "subheader", "Evaluation and comparison")
        _call(ui, "json", evaluation_view)
        evaluation_result = _field(evaluation, ("evaluation_result", "result"))
        if evaluation_result is not None:
            render_metrics(
                _field(evaluation_result, "strategy_metrics"),
                title="Baseline strategy metrics",
                st_module=ui,
                redactor=redactor,
            )
            render_metrics(
                _field(evaluation_result, "benchmark_metrics"),
                title="SPY benchmark metrics",
                st_module=ui,
                redactor=redactor,
            )
            render_metrics(
                _field(evaluation_result, "differences"),
                title="Strategy minus SPY differences",
                st_module=ui,
                redactor=redactor,
            )
        _render_bounded_table(
            ui,
            "Strategy equity curve",
            _field(evaluation, "strategy_equity"),
            page_size,
            redactor=redactor,
        )
        _render_bounded_table(
            ui,
            "SPY equity curve",
            _field(evaluation, "benchmark_equity"),
            page_size,
            redactor=redactor,
        )
        _render_bounded_table(
            ui,
            "Drawdown",
            _field(evaluation, ("drawdown", "drawdowns")),
            page_size,
            redactor=redactor,
        )
        _render_bounded_table(
            ui,
            "Monthly returns",
            _field(evaluation, ("strategy_monthly_returns", "monthly_returns")),
            page_size,
            redactor=redactor,
        )
        _render_artifacts(
            application,
            ui,
            state,
            _field(evaluation, "artifacts"),
            page_size,
            redactor=redactor,
        )

    core = _field(result, ("core_output", "output"))
    _render_bounded_table(
        ui,
        "Positions and portfolio state",
        _field(core, "portfolio_states"),
        page_size,
        redactor=redactor,
    )
    _render_bounded_table(
        ui,
        "Transactions and fills",
        _field(core, "fills"),
        page_size,
        redactor=redactor,
    )
    _render_bounded_table(
        ui,
        "Orders",
        _field(core, "orders"),
        page_size,
        redactor=redactor,
    )
    _render_bounded_table(
        ui,
        "Strategy decisions",
        _field(core, "strategy_decisions"),
        page_size,
        redactor=redactor,
    )
    _render_bounded_table(
        ui,
        "Daily returns",
        _field(core, "daily_returns"),
        page_size,
        redactor=redactor,
    )

    diagnostics = _actionable_errors(
        _field(result, "diagnostics")
    ) + _actionable_errors(_field(_field(result, "audit"), "unfilled_diagnostics"))
    if diagnostics:
        _call(ui, "subheader", "Unfilled orders and diagnostics")
        render_actionable_errors(diagnostics, st_module=ui, redactor=redactor)
    disclosure = _field(result, "limitation_disclosure")
    if isinstance(disclosure, LimitationDisclosure):
        render_limitation_disclosure(disclosure, st_module=ui, redactor=redactor)


def render_backtest(
    application: ResearchApplication,
    *,
    st_module: Any | None = None,
    redactor: Redactor | None = None,
) -> None:
    """Render verified snapshot selection, synchronous execution, and results."""

    if st_module is None:
        import streamlit as st  # noqa: PLC0415

        ui = st
    else:
        ui = st_module
    state = _state(ui)
    page_size = _configured_page_size(state)
    _call(ui, "title", "Backtest")
    _call(
        ui,
        "caption",
        "Runs are synchronous and are pinned to one checksum-verified Data_Snapshot.",
    )

    selected_id = _render_snapshot_candidates(application, ui, state, page_size)
    if selected_id is not None:
        verify_clicked = bool(
            _call(ui, "button", "Verify selected snapshot", key="qrp-backtest-verify")
        )
        if verify_clicked:
            inspected = application.inspect_snapshot(selected_id)
            if isinstance(inspected, Ok) and _snapshot_verified(
                inspected.value, selected_id
            ):
                _set(state, _STATE_VERIFIED_SNAPSHOT, selected_id)
                _set(state, _STATE_VERIFIED_DETAIL, inspected.value)
                _call(
                    ui,
                    "success",
                    f"Snapshot_ID {selected_id} verified for backtesting.",
                )
            else:
                _delete(state, _STATE_VERIFIED_SNAPSHOT)
                _delete(state, _STATE_VERIFIED_DETAIL)
                if isinstance(inspected, Err):
                    render_actionable_errors(
                        inspected.errors,
                        st_module=ui,
                        redactor=redactor,
                    )
                else:
                    _call(
                        ui,
                        "error",
                        "The selected snapshot could not be checksum-verified.",
                    )

    verified_id = _get(state, _STATE_VERIFIED_SNAPSHOT)
    handle = _get(state, "qrp.configuration_handle")
    verified = isinstance(verified_id, str) and verified_id == selected_id
    has_handle = isinstance(handle, ConfigurationHandle)
    if not has_handle:
        _call(
            ui,
            "info",
            "Resolve configuration on Configure / Ingest before running a backtest.",
        )
    if not verified and selected_id is not None:
        _call(
            ui,
            "info",
            "Verify the selected snapshot before the Run backtest control is enabled.",
        )
    run_clicked = bool(
        _call(
            ui,
            "button",
            "Run backtest",
            disabled=not (verified and has_handle),
            type="primary",
            key="qrp-backtest-run",
        )
    )
    if run_clicked and verified and has_handle:
        progress_slot = _call(ui, "empty") or ui

        def on_progress(update: object) -> None:
            _set(state, _STATE_PROGRESS, present_progress(update, redactor=redactor))
            render_progress(update, st_module=progress_slot, redactor=redactor)

        result = application.run_backtest(
            BacktestRequest(str(verified_id)),
            cast(ConfigurationHandle, handle),
            progress=on_progress,
        )
        if isinstance(result, Ok):
            _set(state, _STATE_RESULT, result.value)
            _delete(state, _STATE_ERRORS)
            _render_result(
                application,
                ui,
                state,
                result.value,
                page_size,
                redactor=redactor,
            )
        else:
            _set(state, _STATE_ERRORS, result.errors)
            render_actionable_errors(
                result.errors,
                st_module=ui,
                redactor=redactor,
            )
            _call(
                ui,
                "warning",
                "The failed run was retained as an immutable diagnostic; "
                "prior runs remain available.",
            )

    progress_view = _get(state, _STATE_PROGRESS)
    if progress_view is not None and not run_clicked:
        _call(ui, "subheader", "Latest persisted backtest progress")
        _call(ui, "json", progress_view)
    previous_result = _get(state, _STATE_RESULT)
    if previous_result is not None and not run_clicked:
        _render_result(
            application,
            ui,
            state,
            previous_result,
            page_size,
            redactor=redactor,
        )
    previous_errors = _actionable_errors(_get(state, _STATE_ERRORS))
    if previous_errors and not run_clicked:
        render_actionable_errors(previous_errors, st_module=ui, redactor=redactor)


render_backtest_page = render_backtest
render = render_backtest

__all__ = ["render", "render_backtest", "render_backtest_page"]
