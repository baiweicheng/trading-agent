"""Streamlit composition root and the Configure/Ingest page.

The UI is an adapter around :class:`ResearchApplication`.  Concrete ports are
constructed in one cached composition root, while page code deals only in
application requests/results and redacted presenter/component functions.  No
provider call is made during import or application construction; yfinance is
invoked only after a user submits the synchronous ingestion action.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext, suppress
from datetime import date
from pathlib import Path
from typing import Any, cast

from ..application.backtests import BacktestService
from ..application.comparisons import ComparisonService
from ..application.evaluation import EvaluationService
from ..application.ingestion import DataIngestionService, IngestionRequest
from ..application.inspection import InspectionService
from ..application.jobs import SynchronousJobManager
from ..application.services import ConfigurationHandle, ResearchApplication, RunQuery
from ..application.snapshots import SnapshotManager, SnapshotQuery
from ..config.loader import ENVIRONMENT_FIELD_PATHS, ConfigurationManager
from ..config.serializer import Redactor
from ..domain.errors import Ok
from .components import (
    render_actionable_errors,
    render_limitation_disclosure,
    render_progress,
)
from .pages.backtest import render_backtest
from .pages.runs import render_runs
from .pages.snapshots import render_snapshots
from .presenters import (
    present_configuration_resolution,
    present_errors,
    present_ingestion_result,
    present_progress,
)

PAGE_CONFIGURE_INGEST = "Configure / Ingest"
PAGE_SNAPSHOTS = "Snapshots"
PAGE_BACKTEST = "Backtest"
PAGE_RUNS = "Runs"
PAGE_COMPARE = "Compare"
NAVIGATION = (
    PAGE_CONFIGURE_INGEST,
    PAGE_SNAPSHOTS,
    PAGE_BACKTEST,
    PAGE_RUNS,
    PAGE_COMPARE,
)

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = "config/default.yaml"
_DEFAULT_START = date(2015, 1, 1)
_DEFAULT_END = date(2024, 12, 31)
_STATE_HANDLE = "qrp.configuration_handle"
_STATE_CONFIGURATION = "qrp.configuration_view"
_STATE_RESOLUTION_ERRORS = "qrp.configuration_errors"
_STATE_PROGRESS = "qrp.ingestion_progress"
_STATE_RESULT = "qrp.ingestion_result"
_STATE_RESULT_ERRORS = "qrp.ingestion_errors"


def build_application(project_root: Path | str | None = None) -> ResearchApplication:
    """Build the concrete local application graph without doing external I/O.

    The provider adapter is deliberately created without a download call.  Its
    yfinance import and network boundary remain lazy until ``ingest`` executes.
    ``project_root`` is an injection seam for AppTests and local smoke tests;
    normal Streamlit use resolves the repository root containing ``pyproject.toml``.
    """

    from ..infrastructure.duckdb_metadata import DuckDBMetadataStore
    from ..infrastructure.filesystem_store import FilesystemStore
    from ..infrastructure.logging import StructuredJsonlLogger
    from ..infrastructure.mlflow_tracker import LocalMlflowTracker
    from ..infrastructure.parquet_store import ParquetStore
    from ..infrastructure.run_manifest import RunManifestPublisher
    from ..infrastructure.snapshot_history import SnapshotParquetHistoryReader
    from ..infrastructure.xnys_calendar import XNYSCalendar
    from ..infrastructure.yfinance_provider import YFinanceAdapter
    from ..infrastructure.zipline_bundle import ZiplineBundleAdapter
    from ..infrastructure.zipline_engine import ZiplineBacktestEngine

    root = Path(project_root or _DEFAULT_PROJECT_ROOT).expanduser().resolve()
    data_root = root / "data"
    metadata_path = data_root / "metadata.duckdb"
    mlflow_path = data_root / "mlflow.db"
    log_path = data_root / "logs" / "platform.jsonl"

    redactor = Redactor()
    configuration_manager = ConfigurationManager(project_anchor=root)
    metadata = DuckDBMetadataStore(metadata_path)
    parquet = ParquetStore(data_root, cas_namespace="objects")
    history_reader = SnapshotParquetHistoryReader(parquet)
    filesystem = FilesystemStore(data_root, metadata=metadata, redactor=redactor)
    manifest_publisher = RunManifestPublisher(
        filesystem,
        metadata_store=metadata,
        project_root=root,
    )
    logger = StructuredJsonlLogger(log_path, redactor=redactor)
    jobs = SynchronousJobManager(metadata, logger, redactor=redactor)
    calendar = XNYSCalendar()
    provider = YFinanceAdapter(redactor=redactor)
    snapshot_manager = SnapshotManager(
        storage=filesystem,
        metadata=cast(Any, metadata),
    )

    ingestion = DataIngestionService(
        provider,
        calendar,
        parquet_store=parquet,
        snapshot_publisher=cast(Any, filesystem),
        metadata=metadata,
        job_manager=jobs,
        snapshot_manager=snapshot_manager,
        redactor=redactor,
    )

    tracker = LocalMlflowTracker(
        mlflow_path,
        metadata_store=cast(Any, metadata),
        artifact_store=filesystem,
    )
    evaluation = EvaluationService(
        snapshot_manager=snapshot_manager,
        parquet_store=parquet,
        artifact_store=filesystem,
    )
    bundle = ZiplineBundleAdapter(
        snapshot_manager=snapshot_manager,
        parquet_store=parquet,
        calendar=calendar,
        zipline_root=data_root / "zipline-bundles",
    )
    engine = ZiplineBacktestEngine(
        snapshot_manager=snapshot_manager,
        snapshot_reader=history_reader,
        calendar=calendar,
    )
    backtest = BacktestService(
        tracker=tracker,
        snapshot_manager=snapshot_manager,
        bundle_adapter=bundle,
        engine=engine,
        evaluator=evaluation,
        manifest_publisher=manifest_publisher,
    )
    comparison = ComparisonService(
        metadata_store=cast(Any, metadata),
        artifact_store=filesystem,
        parquet_store=parquet,
        manifest_store=filesystem,
        redactor=redactor,
    )
    inspection = InspectionService(
        snapshot_manager=snapshot_manager,
        metadata_store=cast(Any, metadata),
        artifact_store=filesystem,
        parquet_store=parquet,
        experiment_tracker=tracker,
        redactor=redactor,
    )
    return ResearchApplication(
        configuration_manager=configuration_manager,
        ingestion_service=ingestion,
        snapshot_manager=snapshot_manager,
        backtest_service=backtest,
        comparison_service=comparison,
        inspection_service=inspection,
        metadata_store=cast(Any, metadata),
        logger=logger,
        redactor=redactor,
    )


# Streamlit owns the cache.  The fallback keeps importing this module useful in
# a dependency-light unit test and never starts a server.
_streamlit: Any | None
try:  # pragma: no cover - the normal project environment has Streamlit.
    import streamlit as _streamlit
except ImportError:  # pragma: no cover
    _streamlit = None


if _streamlit is not None:

    @_streamlit.cache_resource(show_spinner=False)  # type: ignore[untyped-decorator]
    def cached_application(project_root: str = "") -> ResearchApplication:
        return build_application(Path(project_root) if project_root else None)

else:  # pragma: no cover

    def cached_application(project_root: str = "") -> ResearchApplication:
        return build_application(Path(project_root) if project_root else None)


def get_application(project_root: Path | str | None = None) -> ResearchApplication:
    """Return the process-cached composition root used by the Streamlit page."""

    key = "" if project_root is None else str(Path(project_root).expanduser().resolve())
    return cast(ResearchApplication, cached_application(key))


def _ui_module(st_module: Any | None = None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as streamlit  # noqa: PLC0415

    return streamlit


def _state(ui: Any) -> Any:
    state = getattr(ui, "session_state", None)
    if state is None:
        # A minimal local UI double may not implement session_state.  The
        # fallback is process-local UI state, never application-layer state.
        state = {}
        with suppress(Exception):
            ui.session_state = state
    return state


def _get_state(state: Any, key: str, default: object = None) -> object:
    try:
        return state.get(key, default)
    except AttributeError:
        try:
            return state[key]
        except (KeyError, TypeError):
            return default


def _set_state(state: Any, key: str, value: object) -> None:
    try:
        state[key] = value
    except (TypeError, AttributeError):
        setattr(state, key, value)


def _delete_state(state: Any, key: str) -> None:
    try:
        state.pop(key, None)
    except AttributeError:
        with suppress(KeyError, TypeError):
            del state[key]


def _call(ui: Any, name: str, *args: object, **kwargs: object) -> object:
    method = getattr(ui, name, None)
    if not callable(method):
        return None
    return method(*args, **kwargs)


def _context(ui: Any, name: str, key: str | None = None) -> Any:
    method = getattr(ui, name, None)
    if callable(method):
        value = method(key) if key is not None else method()
        if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
            return value
    return nullcontext()


def _number(
    ui: Any,
    label: str,
    value: int | float,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    key: str | None = None,
) -> int | float:
    kwargs: dict[str, object] = {}
    if minimum is not None:
        kwargs["min_value"] = minimum
    if maximum is not None:
        kwargs["max_value"] = maximum
    if step is not None:
        kwargs["step"] = step
    if key is not None:
        kwargs["key"] = key
    result = _call(ui, "number_input", label, value=value, **kwargs)
    return value if result is None else cast(int | float, result)


def _text(ui: Any, label: str, value: str, *, key: str | None = None) -> str:
    kwargs = {"key": key} if key is not None else {}
    result = _call(ui, "text_input", label, value, **kwargs)
    return value if result is None else str(result)


def _date_input(ui: Any, label: str, value: date, *, key: str | None = None) -> date:
    kwargs = {"key": key} if key is not None else {}
    result = _call(ui, "date_input", label, value, **kwargs)
    if isinstance(result, date) and not isinstance(result, tuple):
        return result
    return value


def _uploaded_yaml(ui: Any) -> tuple[Path | None, str | None]:
    uploaded = _call(
        ui,
        "file_uploader",
        "YAML document (optional)",
        type=("yaml", "yml"),
    )
    if uploaded is None:
        return None, None
    read = getattr(uploaded, "getvalue", None) or getattr(uploaded, "read", None)
    if not callable(read):
        return None, "The selected YAML document could not be read."
    try:
        payload = read()
        if not isinstance(payload, bytes):
            payload = str(payload).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            prefix="qrp-config-", suffix=".yaml", delete=False
        ) as handle:
            handle.write(payload)
        return Path(handle.name), None
    except OSError:
        return None, "The selected YAML document could not be staged for validation."


def _document_path(ui: Any, project_root: Path) -> Path | None:
    selection = _call(
        ui,
        "selectbox",
        "YAML source",
        (_DEFAULT_CONFIG_PATH, "No YAML document"),
        index=0,
    )
    if selection == "No YAML document" or selection is None:
        return None
    candidate = project_root / str(selection)
    return candidate if candidate.is_file() else None


def _effective_environment_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in ENVIRONMENT_FIELD_PATHS
            if name in os.environ and not name.startswith("QRP_SECRETS__")
        )
    )


def _configuration_values(ui: Any) -> dict[str, object]:
    universe_text = _text(
        ui,
        "Configured universe (comma-separated)",
        ",".join(("AAPL", "JPM", "MSFT", "PG", "XOM")),
        key="qrp-universe",
    )
    symbols = tuple(item.strip() for item in universe_text.split(","))
    symbol_count = max(1, len(symbols))
    position_default = min(5, symbol_count)
    start = _date_input(ui, "Requested range start", _DEFAULT_START, key="qrp-start")
    end = _date_input(ui, "Requested range end", _DEFAULT_END, key="qrp-end")
    attempts = int(
        _number(
            ui,
            "Retry attempts",
            3,
            minimum=1,
            maximum=5,
            step=1,
            key="qrp-attempts",
        )
    )
    initial_delay = _number(
        ui,
        "Retry initial delay (seconds)",
        1.0,
        minimum=0.0,
        maximum=60.0,
        step=0.5,
        key="qrp-initial-delay",
    )
    max_delay = _number(
        ui,
        "Retry maximum delay (seconds)",
        8.0,
        minimum=0.0,
        maximum=60.0,
        step=0.5,
        key="qrp-max-delay",
    )
    multiplier = _number(
        ui,
        "Retry backoff multiplier",
        2.0,
        minimum=1.0,
        maximum=4.0,
        step=0.1,
        key="qrp-multiplier",
    )
    batch_size = int(
        _number(
            ui,
            "Provider batch size",
            5,
            minimum=1,
            maximum=10,
            step=1,
            key="qrp-batch",
        )
    )
    position_count = int(
        _number(
            ui,
            "Position count",
            position_default,
            minimum=1,
            maximum=symbol_count,
            step=1,
            key="qrp-position-count",
        )
    )
    commission = _number(
        ui,
        "Commission (basis points)",
        5.0,
        minimum=0.0,
        step=0.1,
        key="qrp-commission",
    )
    slippage = _number(
        ui,
        "Slippage (basis points)",
        10.0,
        minimum=0.0,
        step=0.1,
        key="qrp-slippage",
    )
    seed = int(
        _number(
            ui,
            "Deterministic seed",
            0,
            minimum=0,
            maximum=4_294_967_295,
            step=1,
            key="qrp-seed",
        )
    )
    return {
        "retry": {
            "attempts": attempts,
            "initial_delay_seconds": initial_delay,
            "max_delay_seconds": max_delay,
            "backoff_multiplier": multiplier,
        },
        "data": {
            "universe": symbols,
            "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
            "batch_size": batch_size,
        },
        "strategy": {"position_count": position_count},
        "execution": {"commission_bps": commission, "slippage_bps": slippage},
        "runtime": {"deterministic_seed": seed},
    }


def _render_prior_records(
    application: ResearchApplication, ui: Any, page_size: int
) -> None:
    """Keep prior valid snapshots and runs discoverable after a new failure."""

    snapshots = None
    list_snapshots = getattr(application, "list_snapshots", None)
    if callable(list_snapshots):
        snapshots = list_snapshots(SnapshotQuery(page=0, page_size=page_size))
    if snapshots is not None and snapshots.errors:
        render_actionable_errors(snapshots.errors, st_module=ui)
    snapshot_rows = []
    if snapshots is not None:
        for item in snapshots.items:
            requested = getattr(item, "requested_range", None)
            snapshot_rows.append(
                {
                    "snapshot_id": getattr(item, "snapshot_id", ""),
                    "provider": getattr(item, "provider", ""),
                    "requested_range": str(requested),
                    "comparison_ready": getattr(item, "comparison_ready", False),
                    "availability": getattr(item, "availability", "unknown"),
                }
            )
    if snapshot_rows:
        _call(ui, "subheader", "Previously published snapshots")
        _call(ui, "dataframe", snapshot_rows, hide_index=True, use_container_width=True)

    runs = None
    search_runs = getattr(application, "search_runs", None)
    if callable(search_runs):
        runs = search_runs(RunQuery(page=0, page_size=page_size))
    if runs is not None and runs.errors:
        render_actionable_errors(runs.errors, st_module=ui)
    run_rows = []
    if runs is not None:
        for item in runs.items:
            run_rows.append(
                {
                    "run_id": str(getattr(item, "run_id", "")),
                    "snapshot_id": getattr(item, "snapshot_id", ""),
                    "state": str(
                        getattr(
                            getattr(item, "state", ""),
                            "value",
                            getattr(item, "state", ""),
                        )
                    ),
                    "strategy": getattr(item, "strategy_id", ""),
                    "evaluation_start": str(getattr(item, "evaluation_start", "")),
                    "evaluation_end": str(getattr(item, "evaluation_end", "")),
                }
            )
    if run_rows:
        _call(ui, "subheader", "Previously recorded runs")
        _call(ui, "dataframe", run_rows, hide_index=True, use_container_width=True)


def _render_result(result: object, ui: Any, redactor: Redactor | None) -> None:
    rendered = present_ingestion_result(result, redactor=redactor)
    status = str(rendered.get("status", ""))
    snapshot_id = rendered.get("snapshot_id")
    if status == "partially_succeeded":
        _call(
            ui,
            "warning",
            f"Ingestion partially succeeded. Snapshot ID: {snapshot_id}",
        )
    else:
        _call(ui, "success", f"Ingestion succeeded. Snapshot ID: {snapshot_id}")
    if rendered.get("errors"):
        render_actionable_errors(
            tuple(getattr(result, "errors", ())), st_module=ui, redactor=redactor
        )
    disclosure = getattr(result, "limitation_disclosure", None)
    if disclosure is not None:
        render_limitation_disclosure(disclosure, st_module=ui, redactor=redactor)
    _call(ui, "json", rendered)


def render_configure_ingest(
    application: ResearchApplication,
    *,
    st_module: Any | None = None,
    project_root: Path | str | None = None,
) -> None:
    """Render configuration resolution and the synchronous ingestion action."""

    ui = _ui_module(st_module)
    root = Path(project_root or _DEFAULT_PROJECT_ROOT).expanduser().resolve()
    state = _state(ui)
    _call(ui, "title", "Quantitative Research Platform")
    _call(
        ui,
        "caption",
        "Local Phase 1 workflow · configuration is validated before "
        "actions are enabled",
    )

    with _context(ui, "form", "configuration-form"):
        yaml_path = _document_path(ui, root)
        uploaded_path, upload_error = _uploaded_yaml(ui)
        values = _configuration_values(ui)
        resolve_clicked = _call(ui, "form_submit_button", "Validate configuration")

    if upload_error:
        _call(ui, "error", upload_error)
    if resolve_clicked:
        selected_path = uploaded_path or yaml_path
        try:
            resolution = application.resolve_configuration(
                selected_path, ui_yaml_values=values
            )
        finally:
            if uploaded_path is not None:
                with suppress(OSError):
                    uploaded_path.unlink()
        if isinstance(resolution, Ok):
            rendered = present_configuration_resolution(resolution.value)
            _set_state(state, _STATE_HANDLE, resolution.value.handle)
            _set_state(state, _STATE_CONFIGURATION, rendered)
            _delete_state(state, _STATE_RESOLUTION_ERRORS)
            _call(
                ui,
                "success",
                "Configuration resolved. Ingestion actions are enabled.",
            )
        else:
            _delete_state(state, _STATE_HANDLE)
            _delete_state(state, _STATE_CONFIGURATION)
            errors = present_errors(resolution.errors)
            _set_state(state, _STATE_RESOLUTION_ERRORS, errors)
            render_actionable_errors(resolution.errors, st_module=ui)
    elif uploaded_path is not None:
        with suppress(OSError):
            uploaded_path.unlink()

    configuration_view = _get_state(state, _STATE_CONFIGURATION)
    if configuration_view is not None:
        _call(ui, "subheader", "Resolved non-secret configuration")
        _call(ui, "json", configuration_view)
        mapped = _effective_environment_names()
        if mapped:
            _call(
                ui,
                "caption",
                "Explicit mapped environment overrides in effect: " + ", ".join(mapped),
            )
        else:
            _call(
                ui,
                "caption",
                "No non-secret mapped environment overrides are currently set.",
            )

    resolution_errors = _get_state(state, _STATE_RESOLUTION_ERRORS)
    if resolution_errors:
        _call(ui, "subheader", "Configuration diagnostics")
        for error in cast(list[Mapping[str, object]], resolution_errors):
            _call(
                ui,
                "error",
                f"{error.get('message', 'Configuration error')} "
                f"Corrective action: {error.get('corrective_action', '')}",
            )

    handle = _get_state(state, _STATE_HANDLE)
    valid_handle = isinstance(handle, ConfigurationHandle)
    _call(ui, "subheader", "Ingest")
    ingest_clicked = _call(
        ui,
        "button",
        "Ingest data",
        disabled=not valid_handle,
        type="primary",
    )
    if ingest_clicked and valid_handle:
        progress_slot = getattr(ui, "empty", lambda: ui)()

        def on_progress(update: object) -> None:
            _set_state(state, _STATE_PROGRESS, present_progress(update))
            render_progress(update, st_module=progress_slot)

        configuration_handle = cast(ConfigurationHandle, handle)
        result = application.ingest(
            IngestionRequest(), configuration_handle, progress=on_progress
        )
        if isinstance(result, Ok):
            _set_state(state, _STATE_RESULT, present_ingestion_result(result.value))
            _delete_state(state, _STATE_RESULT_ERRORS)
            _render_result(result.value, ui, None)
        else:
            _set_state(state, _STATE_RESULT_ERRORS, present_errors(result.errors))
            render_actionable_errors(result.errors, st_module=ui)
            _call(
                ui,
                "warning",
                "The ingestion did not replace previously published snapshots or runs.",
            )

    progress_view = _get_state(state, _STATE_PROGRESS)
    if progress_view is not None:
        _call(ui, "subheader", "Latest persisted job progress")
        _call(ui, "json", progress_view)
    result_view = _get_state(state, _STATE_RESULT)
    if result_view is not None and not ingest_clicked:
        _call(ui, "subheader", "Latest ingestion result")
        _call(ui, "json", result_view)
    result_errors = _get_state(state, _STATE_RESULT_ERRORS)
    if result_errors and not ingest_clicked:
        _call(ui, "subheader", "Latest ingestion diagnostics")
        for error in cast(list[Mapping[str, object]], result_errors):
            _call(
                ui,
                "error",
                f"{error.get('message', '')} "
                f"Corrective action: {error.get('corrective_action', '')}",
            )

    _render_prior_records(application, ui, page_size=100)


def _render_placeholder(page: str, ui: Any) -> None:
    _call(ui, "title", page)
    _call(
        ui,
        "info",
        "This view is wired through the same application facade; its detailed "
        "controls are introduced by the next UI task.",
    )
    _call(
        ui,
        "caption",
        "Previously published snapshots and runs remain available from "
        "Configure / Ingest.",
    )


def main(
    *,
    st_module: Any | None = None,
    application: ResearchApplication | None = None,
    project_root: Path | str | None = None,
) -> None:
    """Run one Streamlit render pass; importing this module starts no server."""

    ui = _ui_module(st_module)
    app = application or get_application(project_root)
    sidebar = getattr(ui, "sidebar", ui)
    selected = _call(sidebar, "radio", "Workflow", NAVIGATION, index=0)
    page = selected if selected in NAVIGATION else PAGE_CONFIGURE_INGEST
    if page == PAGE_CONFIGURE_INGEST:
        render_configure_ingest(app, st_module=ui, project_root=project_root)
    elif page == PAGE_SNAPSHOTS:
        render_snapshots(app, st_module=ui)
    elif page == PAGE_BACKTEST:
        render_backtest(app, st_module=ui)
    elif page == PAGE_RUNS:
        render_runs(app, st_module=ui)
    elif page == PAGE_COMPARE:
        from .pages.compare import render_compare  # noqa: PLC0415

        render_compare(app, st_module=ui)
    else:
        _render_placeholder(page, ui)


__all__ = [
    "NAVIGATION",
    "PAGE_BACKTEST",
    "PAGE_COMPARE",
    "PAGE_CONFIGURE_INGEST",
    "PAGE_RUNS",
    "PAGE_SNAPSHOTS",
    "build_application",
    "cached_application",
    "get_application",
    "main",
    "render_configure_ingest",
]


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script.
    main()
