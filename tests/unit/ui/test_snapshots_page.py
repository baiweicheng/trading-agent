from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

from quant_research_platform.application.services import Page
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.market import DateRange
from quant_research_platform.ui.pages.snapshots import render_snapshots

SNAPSHOT_ID = "snap_" + "a" * 64
NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)


class FakeStreamlit:
    def __init__(self, *, selected_snapshot: str | None = None) -> None:
        self.selected_snapshot = selected_snapshot
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def selectbox(
        self, label: str, options: tuple[object, ...], **kwargs: object
    ) -> object:
        self._record("selectbox", label, options, **kwargs)
        if label == "Snapshot to inspect" and self.selected_snapshot is not None:
            return self.selected_snapshot
        return options[0]

    def number_input(self, label: str, **kwargs: object) -> object:
        self._record("number_input", label, **kwargs)
        return kwargs["value"]

    def text_input(self, label: str, **kwargs: object) -> str:
        self._record("text_input", label, **kwargs)
        return ""

    def dataframe(self, *args: object, **kwargs: object) -> None:
        self._record("dataframe", *args, **kwargs)

    def __getattr__(self, name: str):
        def method(*args: object, **kwargs: object) -> None:
            self._record(name, *args, **kwargs)

        return method


def _summary(*, integrity_error: str | None = None) -> object:
    validation = SimpleNamespace(
        accepted_row_count=12,
        quarantined_row_count=1,
        collapsed_duplicate_count=0,
        gap_count=2,
        failed_symbols=("MSFT",),
        retained_parent_coverage_symbols=("AAPL",),
        stale_symbols=("MSFT",),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 10)),
        comparison_ready=False,
    )
    return SimpleNamespace(
        snapshot_id=SNAPSHOT_ID,
        provider="yfinance",
        requested_range=DateRange(date(2024, 1, 1), date(2024, 1, 10)),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 10)),
        configured_universe=("AAPL", "MSFT"),
        benchmark_symbol="SPY",
        comparison_ready=False,
        availability="available" if integrity_error is None else "invalid",
        created_at=NOW,
        manifest_checksum="b" * 64,
        content_identity_checksum="c" * 64,
        parent_snapshot_id=None,
        limitation_disclosure=LimitationDisclosure.current(),
        validation_summary=validation,
        integrity_error=integrity_error,
    )


def _detail() -> object:
    summary = _summary()
    return SimpleNamespace(
        snapshot_id=SNAPSHOT_ID,
        summary=summary,
        provenance=SimpleNamespace(
            provider="yfinance",
            requested_range=summary.requested_range,
            covered_range=summary.covered_range,
            configured_universe=summary.configured_universe,
            benchmark_symbol="SPY",
            calendar=SimpleNamespace(
                name="XNYS", version="fixture", schedule_checksum="d" * 64
            ),
            schema_versions=SimpleNamespace(
                corporate_action_policy_version="causal_forward_v1"
            ),
            configuration_checksum="e" * 64,
            object_references=(),
            validation_report_checksum=None,
            created_at=NOW,
            provider_requests=(),
            parent_snapshot_id=None,
            operation_id=None,
        ),
        validation_summary=summary.validation_summary,
        readiness=SimpleNamespace(
            available=True,
            comparison_ready=False,
            failed_symbols=("MSFT",),
            stale_symbols=("MSFT",),
            gap_count=2,
            reasons=("data_gaps_recorded", "failed_symbols_recorded"),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )


class FakeApplication:
    def __init__(self, summary: object, detail: object | None = None) -> None:
        self.summary = summary
        self.detail = detail
        self.query: object | None = None
        self.inspected: list[str] = []

    def list_snapshots(self, query: object) -> Page[object]:
        self.query = query
        return Page(
            items=(self.summary,),
            page=query.page,
            page_size=query.page_size,
            total=1,
        )

    def inspect_snapshot(self, snapshot_id: str) -> object:
        self.inspected.append(snapshot_id)
        return Ok(self.detail)


def _calls(
    ui: FakeStreamlit, name: str
) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    return [call for call in ui.calls if call[0] == name]


def test_snapshots_page_uses_bounded_query_and_renders_verified_details() -> None:
    summary = _summary()
    application = FakeApplication(summary, _detail())
    ui = FakeStreamlit(selected_snapshot=SNAPSHOT_ID)

    rendered = render_snapshots(
        application,
        st_module=ui,
        configured_page_size=25,
        redactor=Redactor(("proxy-secret",)),
    )

    assert application.query is not None
    assert application.query.page_size == 25
    assert application.inspected == [SNAPSHOT_ID]
    assert rendered["selected_snapshot_id"] == SNAPSHOT_ID
    assert rendered["detail"]["snapshot"]["snapshot_id"] == SNAPSHOT_ID
    assert any("Limitations and assumptions" in str(call) for call in ui.calls)
    assert any("immutable" in str(call).lower() for call in ui.calls)
    assert all(
        len(call[1][0]) <= 25
        for call in _calls(ui, "dataframe")
        if call[1]
    )


def test_corrupt_snapshot_is_visible_for_diagnostics_but_not_selectable() -> None:
    secret = "provider-proxy-secret"
    summary = _summary(integrity_error=f"checksum mismatch: {secret}")
    application = FakeApplication(summary, _detail())
    ui = FakeStreamlit(selected_snapshot=SNAPSHOT_ID)

    rendered = render_snapshots(
        application,
        st_module=ui,
        configured_page_size=1000,
        redactor=Redactor((secret,)),
    )

    assert application.inspected == []
    assert rendered["selected_snapshot_id"] is None
    selector = _calls(ui, "selectbox")[-1]
    assert selector[1][1] == ("Select a checksum-verified snapshot",)
    assert secret not in repr(ui.calls)
    assert any("cannot be selected" in str(call).lower() for call in ui.calls)


def _app_test_script() -> None:
    from types import SimpleNamespace

    from quant_research_platform.ui.pages.snapshots import render_snapshots

    class EmptyApp:
        def list_snapshots(self, query: object) -> object:
            return SimpleNamespace(
                items=(),
                page=query.page,
                page_size=query.page_size,
                total=0,
                errors=(),
            )

    render_snapshots(EmptyApp())


def test_snapshots_page_runs_in_process_streamlit_apptest() -> None:
    from streamlit.testing.v1 import AppTest

    result = AppTest.from_function(_app_test_script).run()

    assert not result.exception
    assert result.title[0].value == "Snapshots"
