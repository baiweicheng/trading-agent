from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

from quant_research_platform.application.services import (
    ConfigurationHandle,
    Page,
)
from quant_research_platform.config.serializer import REDACTION_MARKER, Redactor
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.domain.execution import RunState
from quant_research_platform.ui.pages.backtest import render_backtest
from quant_research_platform.ui.pages.runs import render_runs


class FakeStreamlit:
    def __init__(
        self,
        *,
        clicked: set[str] | None = None,
        selections: dict[str, object] | None = None,
    ) -> None:
        self.clicked = clicked or set()
        self.selections = selections or {}
        self.session_state: dict[str, object] = {}
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def button(self, label: str, *args: object, **kwargs: object) -> bool:
        self._record("button", label, *args, **kwargs)
        return label in self.clicked and not bool(kwargs.get("disabled", False))

    def selectbox(
        self, label: str, options: tuple[object, ...], **kwargs: object
    ) -> object:
        self._record("selectbox", label, options, **kwargs)
        selected = self.selections.get(label)
        return selected if selected in options else options[int(kwargs.get("index", 0))]

    def text_input(self, label: str, value: str = "", **kwargs: object) -> str:
        self._record("text_input", label, value, **kwargs)
        selected = self.selections.get(label)
        return str(selected) if selected is not None else value

    def number_input(self, label: str, **kwargs: object) -> object:
        self._record("number_input", label, **kwargs)
        return kwargs.get("value", 0)

    def empty(self) -> FakeStreamlit:
        self._record("empty")
        return self

    def download_button(self, *args: object, **kwargs: object) -> bool:
        self._record("download_button", *args, **kwargs)
        return False

    def __getattr__(self, name: str):
        def method(*args: object, **kwargs: object) -> None:
            self._record(name, *args, **kwargs)

        return method


def _calls(
    ui: FakeStreamlit, name: str
) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    return [call for call in ui.calls if call[0] == name]


def _snapshot_page() -> Page[object]:
    summary = SimpleNamespace(
        snapshot_id="snap-1",
        availability="available",
        integrity_error=None,
        provider="fixture",
        requested_range=SimpleNamespace(start=date(2024, 1, 1), end=date(2024, 1, 2)),
        covered_range=None,
        comparison_ready=True,
    )
    return Page(items=(summary,), page=0, page_size=100, total=1)


def _progress() -> object:
    return SimpleNamespace(
        job_id=uuid4(),
        operation="backtest",
        state="running",
        stage="executing",
        completed_units=2,
        total_units=4,
        elapsed_seconds=0.5,
        warnings=("provider secret",),
    )


def _backtest_result() -> object:
    return SimpleNamespace(
        run_id="run-1",
        snapshot_id="snap-1",
        evaluation_range=SimpleNamespace(start=date(2024, 1, 1), end=date(2024, 1, 2)),
        core_output=SimpleNamespace(
            orders=(),
            fills=(),
            portfolio_states=(),
            daily_returns=(),
            strategy_decisions=(),
        ),
        audit=SimpleNamespace(unfilled_orders=(), unfilled_diagnostics=()),
        evaluation=None,
        diagnostics=(),
        limitation_disclosure=LimitationDisclosure.current(),
    )


def _error() -> ActionableError:
    return ActionableError(
        operation="backtest.execute",
        category=ErrorCategory.BACKTEST_EXECUTION,
        message="The fixture run failed.",
        corrective_action="Inspect the retained diagnostic and retry.",
    )


class BacktestApplication:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.list_queries: list[object] = []
        self.inspect_ids: list[str] = []
        self.run_calls: list[tuple[object, object, object]] = []

    def list_snapshots(self, query: object) -> Page[object]:
        self.list_queries.append(query)
        return _snapshot_page()

    def inspect_snapshot(self, snapshot_id: str) -> object:
        self.inspect_ids.append(snapshot_id)
        return Ok(
            SimpleNamespace(
                snapshot_id=snapshot_id,
                integrity_error=None,
                readiness=SimpleNamespace(available=True, comparison_ready=True),
            )
        )

    def run_backtest(self, request: object, handle: object, *, progress) -> object:
        self.run_calls.append((request, handle, progress))
        progress(_progress())
        return self.results.pop(0)


def test_backtest_requires_verification_and_handle_then_persists_progress() -> None:
    application = BacktestApplication([Ok(_backtest_result())])
    handle = ConfigurationHandle(uuid4())
    ui = FakeStreamlit(clicked={"Verify selected snapshot"})
    ui.session_state["qrp.configuration_handle"] = handle

    render_backtest(application, st_module=ui, redactor=Redactor(["secret"]))

    assert application.inspect_ids == ["snap-1"]
    assert application.run_calls == []
    run_button = [
        call for call in _calls(ui, "button") if call[1][0] == "Run backtest"
    ][-1]
    assert run_button[2]["disabled"] is False
    assert "qrp.backtest.verified_snapshot_id" in ui.session_state

    ui.clicked = {"Run backtest"}
    render_backtest(application, st_module=ui, redactor=Redactor(["secret"]))

    assert len(application.run_calls) == 1
    request, called_handle, _ = application.run_calls[0]
    assert request.snapshot_id == "snap-1"
    assert called_handle is handle
    assert ui.session_state["qrp.backtest.progress"]["warnings"] == [
        f"provider {REDACTION_MARKER}"
    ]
    assert ui.session_state["qrp.backtest.result"].run_id == "run-1"
    assert any(
        "Limitations and assumptions" in str(call[1][0])
        for call in _calls(ui, "subheader")
    )
    assert all(query.page_size <= 100 for query in application.list_queries)


def test_backtest_failure_retains_prior_successful_result() -> None:
    application = BacktestApplication([Ok(_backtest_result()), Err((_error(),))])
    ui = FakeStreamlit(clicked={"Verify selected snapshot"})
    ui.session_state["qrp.configuration_handle"] = ConfigurationHandle(uuid4())
    render_backtest(application, st_module=ui)

    ui.clicked = {"Run backtest"}
    render_backtest(application, st_module=ui)
    prior_result = ui.session_state["qrp.backtest.result"]

    render_backtest(application, st_module=ui)

    assert ui.session_state["qrp.backtest.result"] is prior_result
    assert ui.session_state["qrp.backtest.errors"] == (_error(),)
    assert len(application.run_calls) == 2


def _run_summary() -> object:
    return SimpleNamespace(
        run_id="run-1",
        snapshot_id="snap-1",
        state=RunState.SUCCEEDED,
        strategy_id="monthly_momentum_v1",
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )


def _artifact() -> object:
    return SimpleNamespace(
        checksum="a" * 64,
        artifact_kind="equity_curve",
        role="equity_curve",
        relative_uri="artifacts/equity.json",
        media_type="application/json",
        byte_size=4,
        row_count=2,
        availability="available",
        payload=b"rows",
    )


class RunsApplication:
    def __init__(self) -> None:
        self.queries: list[object] = []
        self.inspected: list[str] = []
        self.opened: list[str] = []
        self.paged: list[tuple[str, int, int]] = []

    def search_runs(self, query: object) -> Page[object]:
        self.queries.append(query)
        return Page(items=(_run_summary(),), page=0, page_size=100, total=1)

    def inspect_run(self, run_id: str) -> object:
        self.inspected.append(run_id)
        return Ok(
            SimpleNamespace(
                summary=_run_summary(),
                manifest={"handle": "opaque-handle", "secret": "literal-secret"},
                configuration={"secret": "literal-secret", "safe": "value"},
                environment_fingerprint={"python": "3.11"},
                validation_report={"accepted": 2},
                logs=[{"message": "safe"}],
                artifacts=(_artifact(),),
                limitation_disclosure=LimitationDisclosure.current(),
            )
        )

    def open_artifact(self, checksum: str) -> object:
        self.opened.append(checksum)
        return Ok(_artifact())

    def page_artifact(self, checksum: str, *, page: int, page_size: int) -> object:
        self.paged.append((checksum, page, page_size))
        return Ok(
            SimpleNamespace(
                rows=({"session": "2024-01-01", "value": 1},),
                page=page,
                page_size=page_size,
                total=1,
            )
        )


def test_runs_inspection_is_immutable_and_artifact_access_is_explicit_and_bounded() -> (
    None
):
    application = RunsApplication()
    ui = FakeStreamlit(clicked={"Inspect selected run"})
    redactor = Redactor(["literal-secret"])

    render_runs(application, st_module=ui, redactor=redactor)

    assert application.inspected == ["run-1"]
    assert application.opened == []
    rendered_text = repr(ui.calls)
    assert "This is a terminal Run" in rendered_text
    assert "literal-secret" not in rendered_text
    assert "opaque-handle" not in rendered_text
    assert all(query.page_size <= 100 for query in application.queries)

    ui.clicked = {"Verify and prepare download: equity_curve"}
    render_runs(application, st_module=ui, redactor=redactor)
    assert application.opened == ["a" * 64]

    ui.clicked = {"Load verified run artifact page"}
    render_runs(application, st_module=ui, redactor=redactor)
    assert application.paged == [("a" * 64, 0, 100)]
