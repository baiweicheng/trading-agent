from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from quant_research_platform.application.services import ConfigurationHandle, Page
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.ui.app import (
    PAGE_BACKTEST,
    PAGE_COMPARE,
    PAGE_CONFIGURE_INGEST,
    PAGE_RUNS,
    PAGE_SNAPSHOTS,
)

_SECRET = "literal-secret"
_SNAPSHOT_ID = "snap_" + "a" * 64
_DISCLOSURE = LimitationDisclosure.current()


def _range(start: date = date(2024, 1, 2), end: date = date(2024, 1, 5)) -> object:
    return SimpleNamespace(start=start, end=end)


def _error(
    operation: str = "ingestion.fetch",
    category: ErrorCategory = ErrorCategory.PROVIDER_TERMINAL,
    message: str = "The fixture provider failed for one symbol.",
) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=message,
        corrective_action="Inspect the retained diagnostics and retry the operation.",
        symbol="MSFT" if operation == "ingestion.fetch" else None,
    )


def _artifact(role: str = "equity_curve", payload: bytes = b"artifact") -> Any:
    return SimpleNamespace(
        checksum=sha256(payload).hexdigest(),
        artifact_kind=role,
        role=role,
        relative_uri=f"artifacts/{role}.json",
        media_type="application/json",
        byte_size=len(payload),
        row_count=150,
        availability="available",
        payload=payload,
    )


def _snapshot_summary() -> dict[str, object]:
    return {
        "snapshot_id": _SNAPSHOT_ID,
        "provider": "fixture",
        "requested_range": _range(date(2024, 1, 1), date(2024, 1, 10)),
        "covered_range": _range(),
        "configured_universe": ("AAPL", "MSFT"),
        "benchmark_symbol": "SPY",
        "comparison_ready": True,
        "availability": "available",
        "created_at": date(2024, 1, 10),
        "manifest_checksum": "b" * 64,
        "content_identity_checksum": "c" * 64,
        "limitation_disclosure": _DISCLOSURE,
        "validation_summary": {
            "accepted_row_count": 12,
            "quarantined_row_count": 1,
            "collapsed_duplicate_count": 0,
            "gap_count": 0,
            "failed_symbols": (),
            "retained_parent_coverage_symbols": (),
            "stale_symbols": (),
            "covered_range": _range(),
            "comparison_ready": True,
        },
    }


def _snapshot_detail() -> dict[str, object]:
    reference = _artifact("validation_report", b"validation")
    summary = _snapshot_summary()
    return {
        "snapshot_id": _SNAPSHOT_ID,
        "summary": summary,
        "provenance": {
            "provider": "fixture",
            "requested_range": summary["requested_range"],
            "covered_range": summary["covered_range"],
            "configured_universe": summary["configured_universe"],
            "benchmark_symbol": "SPY",
            "calendar": {"name": "XNYS", "version": "fixture"},
            "schema_versions": {"corporate_action_policy_version": "fixture"},
            "configuration_checksum": "d" * 64,
            "validation_report_checksum": reference.checksum,
            "object_references": (reference,),
            "provider_requests": (),
        },
        "validation_summary": summary["validation_summary"],
        "readiness": {
            "available": True,
            "comparison_ready": True,
            "failed_symbols": (),
            "stale_symbols": (),
            "gap_count": 0,
        },
        "limitation_disclosure": _DISCLOSURE,
    }


def _run_summary(identifier: str) -> dict[str, object]:
    return {
        "run_id": identifier,
        "snapshot_id": _SNAPSHOT_ID,
        "state": "succeeded",
        "strategy_id": "monthly_momentum_v1",
        "evaluation_start": date(2024, 1, 2),
        "evaluation_end": date(2024, 1, 5),
    }


def _run_detail(identifier: str = "run-1") -> dict[str, object]:
    return {
        "summary": _run_summary(identifier),
        "manifest": {"run_id": identifier, "secret": _SECRET},
        "configuration": {"data": {"provider": "fixture"}, "secret": _SECRET},
        "environment_fingerprint": {"python": "3.11", "platform": "test"},
        "validation_report": {"accepted": 12, "gaps": 0},
        "logs": ({"message": "fixture run completed"},),
        "artifacts": (_artifact(),),
        "limitation_disclosure": _DISCLOSURE,
    }


def _metrics(scope: str, value: str) -> dict[str, object]:
    return {
        "scope": scope,
        "metrics": (
            {"name": "total_return", "value": value, "null_reason": None},
            {"name": "sharpe", "value": None, "null_reason": "zero_volatility"},
        ),
    }


def _backtest_result() -> object:
    payload = _artifact("strategy_equity", b"strategy-equity")
    evaluation_result = {
        "strategy_metrics": _metrics("strategy", "0.10"),
        "benchmark_metrics": _metrics("benchmark", "0.05"),
        "differences": _metrics("difference", "0.05"),
    }
    evaluation = {
        "evaluation_range": _range(),
        "evaluation_result": evaluation_result,
        "strategy_equity": tuple(
            {
                "session": date(2024, 1, 2) + timedelta(days=index - 2),
                "equity": 100 + index,
            }
            for index in range(2, 6)
        ),
        "benchmark_equity": tuple(
            {
                "session": date(2024, 1, 2) + timedelta(days=index - 2),
                "equity": 100 + index / 2,
            }
            for index in range(2, 6)
        ),
        "drawdown": ({"session": date(2024, 1, 2), "drawdown": 0},),
        "strategy_monthly_returns": ({"month": "2024-01", "return": "0.10"},),
        "artifacts": (payload,),
        "limitation_disclosure": _DISCLOSURE,
        "ending_cash_balance": "90000",
        "total_commissions": "5",
        "total_slippage": "10",
        "spy_gaps": (),
        "unfilled_orders": (),
        "unfilled_diagnostics": (),
    }
    return SimpleNamespace(
        run_id="run-1",
        snapshot_id=_SNAPSHOT_ID,
        evaluation_range=_range(),
        core_output=SimpleNamespace(
            orders=({"symbol": "AAPL", "quantity": 1},),
            fills=({"symbol": "AAPL", "quantity": 1},),
            portfolio_states=({"session": date(2024, 1, 2), "cash": "90000"},),
            daily_returns=({"session": date(2024, 1, 2), "return": "0.01"},),
            strategy_decisions=({"symbol": "AAPL", "eligible": True},),
        ),
        audit=SimpleNamespace(
            unfilled_orders=({"symbol": "MSFT"},),
            unfilled_diagnostics=(
                _error(
                    "backtest.execute",
                    ErrorCategory.BACKTEST_EXECUTION,
                    "One fixture order was unfilled.",
                ),
            ),
        ),
        evaluation=evaluation,
        diagnostics=(),
        limitation_disclosure=_DISCLOSURE,
    )


def _comparison_output(selected: tuple[str, ...]) -> object:
    payload = b'{"comparison":"fixture"}'
    curves = tuple(
        {
            "run_id": identifier,
            "snapshot_id": f"snapshot-for-{identifier}",
            "evaluation_start": date(2024, 1, 2),
            "evaluation_end": date(2024, 1, 5),
            "strategy_metrics": _metrics("strategy", "0.10"),
            "benchmark_metrics": _metrics("benchmark", "0.05"),
            "strategy_curve": ((date(2024, 1, 2), "100"), (date(2024, 1, 5), "110")),
            "benchmark_curve": ((date(2024, 1, 2), "100"), (date(2024, 1, 5), "105")),
        }
        for identifier in selected
    )
    return SimpleNamespace(
        runs=curves,
        aligned_range=(date(2024, 1, 2), date(2024, 1, 5)),
        aligned_sessions=(date(2024, 1, 2), date(2024, 1, 5)),
        snapshot_differences=(
            SimpleNamespace(
                field_path="snapshot_id",
                values=("snapshot-a", "snapshot-b"),
            ),
        ),
        configuration_differences=(
            SimpleNamespace(field_path="execution.commission_bps", values=("5", "7")),
        ),
        environment_differences=(
            SimpleNamespace(field_path="platform", values=("macOS", "test")),
        ),
        artifact=_artifact("comparison", payload),
        artifact_checksum=sha256(payload).hexdigest(),
        limitation_disclosure=_DISCLOSURE,
    )


def _progress(operation: str) -> object:
    return SimpleNamespace(
        job_id=uuid4(),
        operation=operation,
        state="running",
        stage="executing",
        completed_units=2,
        total_units=4,
        elapsed_seconds=0.5,
        warnings=("provider warning",),
    )


class FakeWorkflowApplication:
    """Offline facade double used by all AppTests; it has no provider boundary."""

    def __init__(
        self,
        *,
        invalid_first_configuration: bool = False,
        run_count: int = 2,
    ) -> None:
        self.invalid_first_configuration = invalid_first_configuration
        self.resolve_calls = 0
        self.ingest_calls = 0
        self.backtest_calls = 0
        self.inspect_snapshot_calls: list[str] = []
        self.inspect_run_calls: list[str] = []
        self.opened_artifacts: list[str] = []
        self.paged_artifacts: list[tuple[str, int, int]] = []
        self.compared: list[tuple[str, ...]] = []
        self.provider_calls = 0
        self._handle = ConfigurationHandle(uuid4())
        self._ingestion_results: list[object] = [
            Ok(
                SimpleNamespace(
                    status="partially_succeeded",
                    snapshot_id=_SNAPSHOT_ID,
                    requested_range=_range(),
                    provider_batches=("batch-1",),
                    provider_records=("record-1",),
                    accepted_rows=("accepted-1",),
                    quarantined_rows=("quarantine-1",),
                    gaps=("gap-1",),
                    failed_symbols=("MSFT",),
                    retained_parent_coverage_symbols=(),
                    snapshot_reused=False,
                    validation={"gap_count": 1, "quarantined_row_count": 1},
                    errors=(),
                    limitation_disclosure=_DISCLOSURE,
                )
            ),
            Err((_error(),)),
        ]
        self._run_records = tuple(
            _run_summary(f"run-{index}") for index in range(1, run_count + 1)
        )

    def resolve_configuration(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.resolve_calls += 1
        if self.invalid_first_configuration and self.resolve_calls == 1:
            return Err(
                (
                    _error(
                        "configuration.resolve",
                        ErrorCategory.CONFIGURATION_INVALID_VALUE,
                        "Configuration validation failed for the fixture.",
                    ),
                )
            )
        return Ok(
            SimpleNamespace(
                handle=self._handle,
                view={
                    "data": {"provider": "fixture", "batch_size": 2},
                    "ui": {"page_size": 3},
                    "secrets": {"https_proxy": "[REDACTED]"},
                },
            )
        )

    def ingest(self, request: object, handle: object, *, progress: Any) -> object:
        del request, handle
        self.ingest_calls += 1
        progress(_progress("ingestion"))
        return self._ingestion_results.pop(0)

    def list_snapshots(self, query: Any) -> Page[object]:
        return Page(
            items=(_snapshot_summary(),),
            page=query.page,
            page_size=query.page_size,
            total=1,
        )

    def inspect_snapshot(self, snapshot_id: str) -> object:
        self.inspect_snapshot_calls.append(snapshot_id)
        return Ok(_snapshot_detail())

    def run_backtest(self, request: object, handle: object, *, progress: Any) -> object:
        del request, handle
        self.backtest_calls += 1
        progress(_progress("backtest"))
        return Ok(_backtest_result())

    def search_runs(self, query: Any) -> Page[object]:
        return Page(
            items=self._run_records,
            page=query.page,
            page_size=query.page_size,
            total=len(self._run_records),
        )

    def inspect_run(self, run_id: str) -> object:
        self.inspect_run_calls.append(run_id)
        return Ok(_run_detail(run_id))

    def open_artifact(self, checksum: str) -> object:
        self.opened_artifacts.append(checksum)
        artifacts = (
            _artifact(),
            _artifact("validation_report", b"validation"),
            _artifact("comparison", b'{"comparison":"fixture"}'),
            _artifact("strategy_equity", b"strategy-equity"),
        )
        for artifact in artifacts:
            if artifact.checksum == checksum:
                return Ok(artifact)
        return Err(
            (
                _error(
                    "artifact.verify",
                    ErrorCategory.INTEGRITY_CHECKSUM,
                    "The fixture artifact checksum is invalid.",
                ),
            )
        )

    def page_artifact(self, checksum: str, *, page: int, page_size: int) -> object:
        self.paged_artifacts.append((checksum, page, page_size))
        return Ok(
            SimpleNamespace(
                rows=tuple({"row": index} for index in range(150)),
                page=page,
                page_size=page_size,
                total=150,
            )
        )

    def compare_runs(self, run_ids: tuple[str, ...]) -> object:
        self.compared.append(tuple(run_ids))
        return Ok(_comparison_output(tuple(run_ids)))


def _app_script(application: object, project_root: object) -> None:
    from pathlib import Path
    from typing import cast

    import streamlit as st

    from quant_research_platform.application.services import ResearchApplication
    from quant_research_platform.ui.app import main

    main(
        st_module=st,
        application=cast(ResearchApplication, application),
        project_root=cast(Path, project_root),
    )


def _run_app(application: FakeWorkflowApplication, project_root: Path) -> AppTest:
    return AppTest.from_function(
        _app_script,
        args=(application, project_root),
        default_timeout=10,
    ).run()


def _widget(elements: Any, label: str) -> Any:
    for element in elements:
        if getattr(element, "label", None) == label:
            return element
    raise AssertionError(f"Streamlit widget not found: {label}")


def _values(elements: Any) -> str:
    try:
        values: tuple[Any, ...] = tuple(elements)
    except TypeError:
        values = (elements,)
    return "\n".join(str(getattr(element, "value", element)) for element in values)


def _ui_text(app_test: AppTest) -> str:
    names = (
        "title",
        "header",
        "subheader",
        "caption",
        "info",
        "warning",
        "error",
        "success",
        "json",
        "text",
        "markdown",
    )
    return "\n".join(_values(getattr(app_test, name)) for name in names)


def _dataframe_rows(app_test: AppTest) -> list[int]:
    rows: list[int] = []
    for element in app_test.dataframe:
        value = getattr(element, "value", None)
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple) and shape:
            rows.append(int(shape[0]))
        elif isinstance(value, (tuple, list)):
            rows.append(len(value))
    return rows


def test_apptest_configure_ingest_gates_progress_partial_failure_and_prior_results(
    tmp_path: Path,
) -> None:
    application = FakeWorkflowApplication(invalid_first_configuration=True)
    app_test = _run_app(application, tmp_path)

    _widget(app_test.button, "Validate configuration").click()
    app_test.run()
    assert "Corrective action" in _values(app_test.error)
    assert _widget(app_test.button, "Ingest data").disabled is True

    _widget(app_test.button, "Validate configuration").click()
    app_test.run()
    assert _widget(app_test.button, "Ingest data").disabled is False

    _widget(app_test.button, "Ingest data").click()
    app_test.run()
    assert application.ingest_calls == 1
    assert "partially succeeded" in _values(app_test.warning).lower()
    assert app_test.session_state["qrp.ingestion_progress"]["completed_units"] == 2
    assert "Recorded data failures" in _values(app_test.info)

    _widget(app_test.button, "Ingest data").click()
    app_test.run()
    assert application.ingest_calls == 2
    assert "did not replace previously published snapshots" in _values(app_test.warning)
    assert "fixture provider failed" in _values(app_test.error)

    # The next render retains the successful result and prior snapshot/run access.
    app_test.run()
    assert "Previously published snapshots" in _values(app_test.subheader)
    assert "Latest ingestion result" in _values(app_test.subheader)
    assert _SECRET not in _ui_text(app_test)
    assert application.provider_calls == 0
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))


def test_apptest_snapshots_inspection_immutability_paging_download_and_disclosure(
    tmp_path: Path,
) -> None:
    application = FakeWorkflowApplication()
    app_test = _run_app(application, tmp_path)

    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_SNAPSHOTS)
    app_test.run()
    assert app_test.title[0].value == "Snapshots"
    _widget(app_test.number_input, "Snapshots per page").set_value(3)
    _widget(app_test.selectbox, "Snapshot to inspect").set_value(_SNAPSHOT_ID)
    app_test.run()

    assert application.inspect_snapshot_calls == [_SNAPSHOT_ID]
    assert "Published snapshots are immutable" in _values(app_test.info)
    assert "Limitations and assumptions" in _values(app_test.subheader)
    assert "yfinance" in _values(app_test.info)
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))

    _widget(app_test.button, "Load artifact page").click()
    app_test.run()
    assert application.paged_artifacts
    assert application.paged_artifacts[-1][2] <= 100
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))

    _widget(app_test.button, "Verify and prepare selected artifact download").click()
    app_test.run()
    assert application.opened_artifacts
    assert app_test.download_button
    assert _SECRET not in _ui_text(app_test)


def test_apptest_backtest_requires_verified_snapshot_and_renders_result(
    tmp_path: Path,
) -> None:
    application = FakeWorkflowApplication()
    app_test = _run_app(application, tmp_path)

    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_BACKTEST)
    app_test.run()
    assert _widget(app_test.button, "Run backtest").disabled is True
    assert "Resolve configuration" in _values(app_test.info)

    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_CONFIGURE_INGEST)
    app_test.run()
    _widget(app_test.button, "Validate configuration").click()
    app_test.run()

    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_BACKTEST)
    app_test.run()
    _widget(app_test.button, "Verify selected snapshot").click()
    app_test.run()
    assert application.inspect_snapshot_calls == [_SNAPSHOT_ID]
    assert _widget(app_test.button, "Run backtest").disabled is False

    _widget(app_test.button, "Run backtest").click()
    app_test.run()
    assert application.backtest_calls == 1
    assert "Backtest completed. Run_ID: run-1" in _values(app_test.success)
    assert "Baseline strategy metrics" in _values(app_test.subheader)
    assert "SPY benchmark metrics" in _values(app_test.subheader)
    assert "Strategy minus SPY differences" in _values(app_test.subheader)
    assert "Unfilled orders and diagnostics" in _values(app_test.subheader)
    assert "Limitations and assumptions" in _values(app_test.subheader)
    assert app_test.session_state["qrp.backtest.progress"]["completed_units"] == 2
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))
    assert _SECRET not in _ui_text(app_test)


def test_apptest_runs_discovery_inspection_artifact_separation_and_navigation(
    tmp_path: Path,
) -> None:
    application = FakeWorkflowApplication()
    app_test = _run_app(application, tmp_path)

    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_RUNS)
    app_test.run()
    assert app_test.title[0].value == "Runs"
    _widget(app_test.button, "Inspect selected run").click()
    app_test.run()
    assert application.inspect_run_calls == ["run-1"]
    assert "terminal Run" in _values(app_test.info)
    assert "Manifest" in _values(app_test.subheader)
    assert "Validation report and logs" in _values(app_test.subheader)
    assert _SECRET not in _ui_text(app_test)

    # Inspection does not consume artifact bytes; each access is explicit.
    assert application.opened_artifacts == []
    _widget(app_test.button, "Load verified run artifact page").click()
    app_test.run()
    assert application.paged_artifacts[-1][2] <= 100
    _widget(app_test.button, "Verify and prepare download: equity_curve").click()
    app_test.run()
    assert application.opened_artifacts
    assert app_test.download_button
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))

    # Switching pages repeatedly is an in-process rerun, not a server restart.
    for page, title in (
        (PAGE_SNAPSHOTS, "Snapshots"),
        (PAGE_RUNS, "Runs"),
        (PAGE_BACKTEST, "Backtest"),
        (PAGE_CONFIGURE_INGEST, "Quantitative Research Platform"),
    ):
        _widget(app_test.sidebar.radio, "Workflow").set_value(page)
        app_test.run()
        assert not app_test.exception
        assert app_test.title[0].value == title
    assert application.provider_calls == 0


def test_apptest_compare_preserves_order_provenance_curves_download_and_bounds(
    tmp_path: Path,
) -> None:
    application = FakeWorkflowApplication(run_count=11)
    app_test = _run_app(application, tmp_path)
    _widget(app_test.sidebar.radio, "Workflow").set_value(PAGE_COMPARE)
    app_test.run()

    assert "minimum is 2" in _values(app_test.error)
    assert _widget(app_test.button, "Compare selected runs").disabled is True

    selected = ["run-2", "run-1"]
    _widget(app_test.multiselect, "Runs (selection order is preserved)").set_value(
        selected
    )
    app_test.run()
    _widget(app_test.button, "Compare selected runs").click()
    app_test.run()

    assert application.compared == [tuple(selected)]
    assert "Snapshot provenance differences" in _values(app_test.subheader)
    assert "Configuration differences" in _values(app_test.subheader)
    assert "Environment fingerprint differences" in _values(app_test.subheader)
    assert "Original and aligned evaluation ranges" in _values(app_test.subheader)
    assert "Comparison metrics" in _values(app_test.subheader)
    assert "Baseline strategy equity curves" in _values(app_test.subheader)
    assert "SPY benchmark equity curves" in _values(app_test.subheader)
    assert "Verified comparison artifact" in _values(app_test.subheader)
    assert len(app_test.get("vega_lite_chart")) == 2
    assert app_test.download_button
    assert "Limitations and assumptions" in _values(app_test.subheader)

    all_run_ids = [f"run-{index}" for index in range(1, 12)]
    _widget(app_test.multiselect, "Runs (selection order is preserved)").set_value(
        all_run_ids
    )
    app_test.run()
    assert "maximum is 10" in _values(app_test.error)
    assert _widget(app_test.button, "Compare selected runs").disabled is True
    assert application.compared == [tuple(selected)]
    assert all(row_count <= 100 for row_count in _dataframe_rows(app_test))
    assert _SECRET not in _ui_text(app_test)


def test_apptest_navigation_is_offline_and_never_calls_external_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import yfinance  # type: ignore[import-untyped]

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("AppTest must not call yfinance")

    monkeypatch.setattr(yfinance, "download", fail_if_called)
    application = FakeWorkflowApplication()
    app_test = _run_app(application, tmp_path)

    for page in (
        PAGE_CONFIGURE_INGEST,
        PAGE_SNAPSHOTS,
        PAGE_BACKTEST,
        PAGE_RUNS,
        PAGE_COMPARE,
    ):
        _widget(app_test.sidebar.radio, "Workflow").set_value(page)
        app_test.run()
        assert not app_test.exception

    assert application.provider_calls == 0
    assert not app_test.exception
