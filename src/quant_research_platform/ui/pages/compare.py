"""Streamlit Compare page.

The page is deliberately a presentation adapter.  Run discovery and comparison
validation are delegated to :class:`ResearchApplication`; this module never
opens metadata, artifact, Parquet, provider, or secret stores directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast

from ...application.services import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Page,
    ResearchApplication,
    RunQuery,
    RunState,
)
from ..components import (
    bounded_page_size,
    render_actionable_errors,
    render_artifact_download,
    render_chart,
    render_limitation_disclosure,
    render_table_page,
)
from ..presenters import present_comparison_output

_MIN_RUNS = 2
_MAX_RUNS = 10
_DEFAULT_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class _RunOption:
    """The redacted, bounded subset of a run summary used by the selector."""

    identifier: str
    state: str
    snapshot_id: str
    strategy_id: str
    evaluation_start: date | object
    evaluation_end: date | object

    @property
    def label(self) -> str:
        return (
            f"{self.identifier} · {self.strategy_id} · {self.snapshot_id} · "
            f"{self.evaluation_start} to {self.evaluation_end}"
        )


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
        # Lightweight AppTest doubles often omit optional Streamlit keyword
        # arguments.  The production call still receives the full arguments.
        try:
            return method(*args)
        except TypeError:
            return None


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


def _state_value(value: object) -> str:
    raw = _field(value, "state", "")
    return str(getattr(raw, "value", raw)).strip().lower()


def _identifier(value: object) -> str:
    raw = _field(value, ("run_id", "platform_run_id", "id"), "")
    return str(raw).strip()


def _option(value: object) -> _RunOption | None:
    identifier = _identifier(value)
    if not identifier:
        return None
    start = _field(value, "evaluation_start", "")
    end = _field(value, "evaluation_end", "")
    return _RunOption(
        identifier=identifier,
        state=_state_value(value),
        snapshot_id=str(_field(value, "snapshot_id", "")),
        strategy_id=str(_field(value, ("strategy_id", "strategy_identifier"), "")),
        evaluation_start=start,
        evaluation_end=end,
    )


def _page_items(value: object) -> tuple[object, ...]:
    if isinstance(value, Page):
        return value.items
    return _items(_field(value, ("items", "records"), ()))


def _page_errors(value: object) -> tuple[ActionableError, ...]:
    raw = _field(value, "errors", ())
    return tuple(error for error in _items(raw) if isinstance(error, ActionableError))


def _unexpected(operation: str) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=ErrorCategory.INTERNAL_UNEXPECTED,
        message=(
            "The Compare page could not complete the requested application operation."
        ),
        corrective_action=(
            "Retry the operation; if it continues to fail, inspect the local "
            "diagnostics."
        ),
    )


def _selection_error(message: str, *, field_path: str = "run_ids") -> ActionableError:
    return ActionableError(
        operation="comparison.selection",
        category=ErrorCategory.COMPARISON_SELECTION,
        message=message,
        corrective_action="Select 2–10 distinct successful runs, then retry.",
        field_path=field_path,
    )


def _selection_errors(
    selected: Sequence[str], options: Mapping[str, _RunOption]
) -> tuple[ActionableError, ...]:
    errors: list[ActionableError] = []
    count = len(selected)
    if count < _MIN_RUNS:
        errors.append(
            _selection_error(
                "At least 2 successful runs are required for a comparison; "
                "the minimum is 2."
            )
        )
    elif count > _MAX_RUNS:
        errors.append(
            _selection_error(
                "At most 10 successful runs may be compared; the maximum is 10."
            )
        )

    seen: set[str] = set()
    for identifier in selected:
        if identifier in seen:
            errors.append(
                _selection_error(
                    "The selected run IDs must be distinct.",
                    field_path=f"run_ids[{identifier}]",
                )
            )
            continue
        seen.add(identifier)
        option = options.get(identifier)
        if option is None:
            errors.append(
                _selection_error(
                    (
                        f"Run {identifier} was not found in the successful-run "
                        "discovery page."
                    ),
                    field_path="run_ids",
                )
            )
        elif option.state != RunState.SUCCEEDED.value:
            errors.append(
                _selection_error(
                    (
                        f"Run {identifier} is {option.state or 'not successful'}; "
                        "only successful runs may be compared."
                    ),
                    field_path=f"run_ids[{identifier}]",
                )
            )
    return tuple(errors)


def _safe_table(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    ui: Any,
    page_size: int,
) -> None:
    if not rows:
        _call(ui, "info", f"No {title.casefold()} are available.")
        return
    bounded = tuple(rows[:page_size])
    page = SimpleNamespace(
        rows=bounded,
        page=0,
        page_size=page_size,
        total=len(rows),
    )
    render_table_page(
        page,
        configured_page_size=page_size,
        requested_page_size=page_size,
        title=title,
        st_module=ui,
    )


def _difference_rows(
    differences: object, *, category: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for difference in _items(differences):
        if isinstance(difference, Mapping):
            path = difference.get("field_path", difference.get("path", ""))
            values = difference.get("values", ())
        else:
            path = _field(difference, ("field_path", "path"), "")
            values = _field(difference, "values", ())
        rows.append(
            {
                "category": category,
                "field_path": path,
                "values": tuple(_items(values)),
            }
        )
    return rows


def _metric_rows(runs: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in _items(runs):
        run_id = (
            run.get("run_id", "")
            if isinstance(run, Mapping)
            else _field(run, "run_id", "")
        )
        for scope_name, field_name in (
            ("strategy", "strategy_metrics"),
            ("benchmark", "benchmark_metrics"),
        ):
            metrics = _field(run, field_name)
            if not isinstance(metrics, Mapping):
                continue
            for metric in _items(metrics.get("metrics", ())):
                if not isinstance(metric, Mapping):
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "scope": scope_name,
                        "metric": metric.get("name", ""),
                        "value": metric.get("value"),
                        "null_reason": metric.get("null_reason"),
                    }
                )
    return rows


def _range_rows(
    runs: object, aligned_range: object
) -> list[dict[str, object]]:
    if isinstance(aligned_range, Mapping):
        aligned_start = aligned_range.get("start")
        aligned_end = aligned_range.get("end")
    else:
        aligned_start = aligned_end = None
    rows: list[dict[str, object]] = []
    for run in _items(runs):
        if not isinstance(run, Mapping):
            continue
        original = run.get("original_range")
        if isinstance(original, Mapping):
            original_start = original.get("start")
            original_end = original.get("end")
        else:
            original_start = original_end = None
        rows.append(
            {
                "run_id": run.get("run_id", ""),
                "snapshot_id": run.get("snapshot_id", ""),
                "original_start": original_start,
                "original_end": original_end,
                "aligned_start": aligned_start,
                "aligned_end": aligned_end,
            }
        )
    return rows


def _chart_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not number.is_finite():
        return None
    return float(number)


def _curve_chart(
    runs: object, curve_name: str, *, title: str
) -> Mapping[str, object] | None:
    values: list[dict[str, object]] = []
    for run in _items(runs):
        if not isinstance(run, Mapping):
            continue
        run_id = run.get("run_id", "")
        for point in _items(run.get(curve_name, ())):
            if isinstance(point, Mapping):
                session = point.get("session")
                equity_value = point.get("equity")
            elif isinstance(point, (tuple, list)) and len(point) == 2:
                session, equity_value = point
            else:
                continue
            equity = _chart_number(equity_value)
            if equity is None or session is None:
                continue
            values.append(
                {
                    "run_id": run_id,
                    "session": session,
                    "equity": equity,
                }
            )
    if not values:
        return None
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "data": {"values": values},
        "mark": {"type": "line", "point": False},
        "encoding": {
            "x": {"field": "session", "type": "temporal"},
            "y": {"field": "equity", "type": "quantitative"},
            "color": {"field": "run_id", "type": "nominal"},
            "tooltip": [
                {"field": "run_id", "type": "nominal"},
                {"field": "session", "type": "temporal"},
                {"field": "equity", "type": "quantitative"},
            ],
        },
    }


def _render_comparison(
    output: object,
    *,
    application: ResearchApplication,
    ui: Any,
    page_size: int,
    redactor: object | None,
) -> None:
    try:
        rendered = present_comparison_output(output, redactor=redactor)  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError):
        render_actionable_errors((_unexpected("comparison.present"),), st_module=ui)
        return

    output_disclosure = _field(output, "limitation_disclosure")
    if isinstance(output_disclosure, LimitationDisclosure):
        render_limitation_disclosure(
            output_disclosure, st_module=ui, redactor=cast(Any, redactor)
        )

    _call(ui, "subheader", "Snapshot provenance differences")
    _safe_table(
        _difference_rows(rendered.get("snapshot_differences"), category="snapshot"),
        title="Snapshot provenance differences",
        ui=ui,
        page_size=page_size,
    )
    _call(ui, "subheader", "Configuration differences")
    _safe_table(
        _difference_rows(
            rendered.get("configuration_differences"), category="configuration"
        ),
        title="Configuration differences",
        ui=ui,
        page_size=page_size,
    )
    _call(ui, "subheader", "Environment fingerprint differences")
    _safe_table(
        _difference_rows(
            rendered.get("environment_differences"), category="environment"
        ),
        title="Environment fingerprint differences",
        ui=ui,
        page_size=page_size,
    )

    runs = rendered.get("runs", ())
    _call(ui, "subheader", "Original and aligned evaluation ranges")
    _safe_table(
        _range_rows(runs, rendered.get("aligned_range")),
        title="Original and aligned evaluation ranges",
        ui=ui,
        page_size=page_size,
    )
    aligned_sessions = _items(rendered.get("aligned_sessions"))
    if aligned_sessions:
        _call(
            ui,
            "caption",
            (
                f"Aligned comparison sessions: {aligned_sessions[0]} through "
                f"{aligned_sessions[-1]} ({len(aligned_sessions)} sessions)."
            ),
        )

    _call(ui, "subheader", "Metrics for every selected run")
    _safe_table(
        _metric_rows(runs),
        title="Comparison metrics",
        ui=ui,
        page_size=page_size,
    )

    strategy_chart = _curve_chart(
        runs, "strategy_curve", title="Baseline strategy equity curves"
    )
    if strategy_chart is not None:
        render_chart(
            strategy_chart,
            title="Baseline strategy equity curves",
            st_module=ui,
        )
    else:
        _call(ui, "info", "No aligned baseline strategy equity curves are available.")
    benchmark_chart = _curve_chart(
        runs, "benchmark_curve", title="SPY benchmark equity curves"
    )
    if benchmark_chart is not None:
        render_chart(benchmark_chart, title="SPY benchmark equity curves", st_module=ui)
    else:
        _call(ui, "info", "No aligned SPY benchmark equity curves are available.")

    _render_verified_download(
        output,
        application=application,
        ui=ui,
        redactor=redactor,
    )


def _render_verified_download(
    output: object,
    *,
    application: ResearchApplication,
    ui: Any,
    redactor: object | None,
) -> None:
    """Offer only the complete verified comparison artifact as a download.

    Comparison tables and charts above are ordinary bounded views.  This path
    is intentionally separate and uses the checksummed artifact payload when
    available, otherwise asking the facade for its lazy verified stream.
    """

    artifact = _field(output, "artifact")
    checksum = _field(output, ("artifact_checksum",), _field(artifact, "checksum"))
    if not isinstance(checksum, str) or not checksum:
        render_actionable_errors(
            (_selection_error("The comparison has no checksummed artifact."),),
            st_module=ui,
        )
        return

    payload = _field(artifact, ("payload", "bytes"))
    if isinstance(payload, bytes) and sha256(payload).hexdigest() == checksum:
        _call(ui, "subheader", "Verified comparison artifact")
        render_artifact_download(
            artifact,
            data=payload,
            label="Download verified comparison artifact",
            file_name=f"comparison-{checksum[:16]}.json",
            st_module=ui,
            redactor=cast(Any, redactor),
        )
        return

    opener = getattr(application, "open_artifact", None)
    if not callable(opener):
        render_actionable_errors(
            (
                _selection_error(
                    "The comparison artifact could not be checksum-verified."
                ),
            ),
            st_module=ui,
        )
        return
    try:
        opened = opener(checksum)
    except Exception:
        opened = Err((_unexpected("artifact.verify"),), preserve_order=True)
    if isinstance(opened, Err):
        render_actionable_errors(opened, st_module=ui)
        return
    if not isinstance(opened, Ok):
        render_actionable_errors(
            (
                _selection_error(
                    "The comparison artifact could not be checksum-verified."
                ),
            ),
            st_module=ui,
        )
        return
    _call(ui, "subheader", "Verified comparison artifact")
    render_artifact_download(
        opened.value,
        label="Download verified comparison artifact",
        file_name=f"comparison-{checksum[:16]}.json",
        st_module=ui,
        redactor=cast(Any, redactor),
    )


def render_compare(
    application: ResearchApplication,
    *,
    st_module: Any | None = None,
    page: int = 0,
    page_size: int = _DEFAULT_PAGE_SIZE,
    redactor: object | None = None,
) -> None:
    """Render one deterministic Compare page pass.

    Successful runs are discovered through the typed facade with an explicit
    bounded page.  The multiselect return order is passed unchanged to
    ``compare_runs`` so the comparison artifact retains user-selected order.
    """

    ui = _ui_module(st_module)
    effective_page_size = bounded_page_size(page_size, page_size)
    if isinstance(page, bool) or not isinstance(page, int) or page < 0:
        page = 0

    _call(ui, "title", "Compare runs")
    _call(
        ui,
        "caption",
        (
            "Select 2–10 successful runs in the order they should appear in the "
            "comparison."
        ),
    )
    render_limitation_disclosure(
        LimitationDisclosure.current(),
        st_module=ui,
        redactor=cast(Any, redactor),
    )

    search = getattr(application, "search_runs", None)
    if not callable(search):
        render_actionable_errors(
            (_unexpected("run.search"),), st_module=ui, redactor=cast(Any, redactor)
        )
        return

    discovery_query = RunQuery(
        state=RunState.SUCCEEDED,
        page=page,
        page_size=effective_page_size,
    )
    try:
        discovered = search(discovery_query)
    except Exception:
        render_actionable_errors(
            (_unexpected("run.search"),), st_module=ui, redactor=cast(Any, redactor)
        )
        return
    if isinstance(discovered, Err):
        render_actionable_errors(
            discovered, st_module=ui, redactor=cast(Any, redactor)
        )
        return

    records = _page_items(discovered)
    discovery_errors = _page_errors(discovered)
    if discovery_errors:
        render_actionable_errors(
            discovery_errors, st_module=ui, redactor=cast(Any, redactor)
        )
    all_options = tuple(
        option
        for record in records
        if (option := _option(record)) is not None
    )
    options_by_id: dict[str, _RunOption] = {}
    for option in all_options:
        options_by_id.setdefault(option.identifier, option)
    successful = tuple(
        option
        for option in options_by_id.values()
        if option.state == RunState.SUCCEEDED.value
    )

    if successful:
        rows = [
            {
                "run_id": option.identifier,
                "snapshot_id": option.snapshot_id,
                "strategy": option.strategy_id,
                "evaluation_start": option.evaluation_start,
                "evaluation_end": option.evaluation_end,
                "state": option.state,
            }
            for option in successful
        ]
        _safe_table(
            rows,
            title="Successful runs",
            ui=ui,
            page_size=effective_page_size,
        )
    else:
        _call(ui, "info", "No successful terminal runs are available to compare.")

    options = tuple(option.identifier for option in successful)
    selected_value = _call(
        ui,
        "multiselect",
        "Runs (selection order is preserved)",
        options,
        default=(),
        format_func=lambda value: options_by_id[str(value)].label
        if str(value) in options_by_id
        else str(value),
        key="qrp-compare-selection",
    )
    if selected_value is None:
        selected: tuple[str, ...] = ()
    elif isinstance(selected_value, (str, bytes)):
        selected = (str(selected_value),)
    elif isinstance(selected_value, set):
        # A set is not a Streamlit return type, but sorting makes a test double
        # deterministic rather than allowing hash order to affect artifacts.
        selected = tuple(sorted(str(value) for value in selected_value))
    else:
        selected = tuple(str(value) for value in _items(selected_value))

    selection_errors = _selection_errors(selected, options_by_id)
    if selection_errors:
        render_actionable_errors(
            selection_errors, st_module=ui, redactor=cast(Any, redactor)
        )
    if len(selected) not in range(_MIN_RUNS, _MAX_RUNS + 1):
        _call(
            ui,
            "caption",
            "Comparison is enabled only for 2–10 distinct successful runs.",
        )

    compare_clicked = _call(
        ui,
        "button",
        "Compare selected runs",
        disabled=bool(selection_errors),
        type="primary",
    )
    if not compare_clicked or selection_errors:
        return

    compare = getattr(application, "compare_runs", None)
    if not callable(compare):
        render_actionable_errors(
            (_unexpected("comparison.execute"),),
            st_module=ui,
            redactor=cast(Any, redactor),
        )
        return
    try:
        result = compare(selected)
    except Exception:
        result = Err((_unexpected("comparison.execute"),), preserve_order=True)
    if isinstance(result, Err):
        render_actionable_errors(
            result, st_module=ui, redactor=cast(Any, redactor)
        )
        return
    if not isinstance(result, Ok):
        render_actionable_errors(
            (_unexpected("comparison.execute"),),
            st_module=ui,
            redactor=cast(Any, redactor),
        )
        return
    _render_comparison(
        result.value,
        application=application,
        ui=ui,
        page_size=effective_page_size,
        redactor=redactor,
    )


# Compatibility names keep the page discoverable to the composition root and
# AppTest fixtures without creating alternate rendering paths.
render_compare_page = render_compare
render = render_compare


__all__ = ["render", "render_compare", "render_compare_page"]
