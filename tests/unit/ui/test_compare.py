from __future__ import annotations

from datetime import date
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

from quant_research_platform.application.services import Page
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.ui.pages.compare import render_compare


class FakeStreamlit:
    def __init__(self, selected: object = (), clicked: bool = True) -> None:
        self.selected = selected
        self.clicked = clicked
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def multiselect(self, *args: object, **kwargs: object) -> object:
        self._record("multiselect", *args, **kwargs)
        return self.selected

    def button(self, *args: object, **kwargs: object) -> bool:
        self._record("button", *args, **kwargs)
        return self.clicked

    def dataframe(self, *args: object, **kwargs: object) -> None:
        self._record("dataframe", *args, **kwargs)

    def download_button(self, *args: object, **kwargs: object) -> bool:
        self._record("download_button", *args, **kwargs)
        return False

    def vega_lite_chart(self, *args: object, **kwargs: object) -> None:
        self._record("vega_lite_chart", *args, **kwargs)

    def __getattr__(self, name: str):
        def method(*args: object, **kwargs: object) -> None:
            self._record(name, *args, **kwargs)

        return method


class FakeApplication:
    def __init__(self, records: tuple[object, ...], result: object) -> None:
        self.records = records
        self.result = result
        self.search_query: object | None = None
        self.compared: tuple[str, ...] | None = None

    def search_runs(self, query: object) -> Page[object]:
        self.search_query = query
        return Page(self.records, query.page, query.page_size, len(self.records))

    def compare_runs(self, run_ids: tuple[str, ...]) -> object:
        self.compared = run_ids
        return self.result


def _summary(identifier: int, state: str = "succeeded") -> object:
    return SimpleNamespace(
        run_id=UUID(int=identifier),
        state=state,
        snapshot_id=f"snap_{identifier:064x}",
        strategy_id="monthly_momentum_v1",
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 4),
    )


def _error(message: str) -> ActionableError:
    return ActionableError(
        operation="comparison.alignment",
        category=ErrorCategory.COMPARISON_SELECTION,
        message=message,
        corrective_action="Select runs with a common evaluation session, then retry.",
    )


def _comparison_output() -> object:
    payload = b'{"selected_run_ids":[]}'
    checksum = sha256(payload).hexdigest()
    metric = {
        "scope": "strategy",
        "metrics": [{"name": "total_return", "value": "0.1", "null_reason": None}],
    }
    benchmark_metric = {
        "scope": "benchmark",
        "metrics": [{"name": "total_return", "value": "0.05", "null_reason": None}],
    }
    curves = (
        {
            "run_id": "run-1",
            "snapshot_id": "snap-1",
            "evaluation_start": date(2024, 1, 2),
            "evaluation_end": date(2024, 1, 4),
            "strategy_metrics": metric,
            "benchmark_metrics": benchmark_metric,
            "strategy_curve": ((date(2024, 1, 2), Decimal("100")),),
            "benchmark_curve": ((date(2024, 1, 2), Decimal("100")),),
        },
    )
    return SimpleNamespace(
        runs=curves,
        aligned_sessions=(date(2024, 1, 2),),
        snapshot_differences=(
            SimpleNamespace(
                category="snapshot",
                field_path="snapshot_id",
                values=("snap-1", "snap-2"),
            ),
        ),
        configuration_differences=(),
        environment_differences=(),
        artifact=SimpleNamespace(
            checksum=checksum,
            payload=payload,
            byte_size=len(payload),
            role="comparison",
            media_type="application/json",
            availability="available",
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )


def _calls(
    ui: FakeStreamlit, name: str
) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    return [call for call in ui.calls if call[0] == name]


def test_invalid_selection_is_bounded_structured_and_does_not_call_comparison() -> None:
    ui = FakeStreamlit(selected=(str(UUID(int=1)),))
    application = FakeApplication((_summary(1), _summary(2)), Ok(_comparison_output()))

    render_compare(application, st_module=ui, page_size=1000)

    assert application.compared is None
    assert application.search_query is not None
    assert application.search_query.page_size == 100
    button = _calls(ui, "button")[-1]
    assert button[2]["disabled"] is True
    assert any("minimum is 2" in str(call) for call in _calls(ui, "error"))
    assert any(
        "Limitations and assumptions" in str(call) for call in _calls(ui, "subheader")
    )


def test_valid_selection_renders_provenance_curves_and_download() -> None:
    first = str(UUID(int=1))
    second = str(UUID(int=2))
    ui = FakeStreamlit(selected=(second, first))
    application = FakeApplication((_summary(1), _summary(2)), Ok(_comparison_output()))

    render_compare(application, st_module=ui)

    assert application.compared == (second, first)
    assert _calls(ui, "button")[-1][2]["disabled"] is False
    assert len(_calls(ui, "vega_lite_chart")) == 2
    assert _calls(ui, "download_button")
    labels = " ".join(str(call) for call in ui.calls)
    assert "Verified comparison artifact" in labels
    assert "Snapshot provenance differences" in labels
    assert "Comparison metrics" in labels
    assert all("secret" not in labels.casefold() for _ in (0,))


def test_duplicate_and_nonterminal_selections_are_rejected_before_facade_compare() -> (
    None
):
    running = str(UUID(int=3))
    first = str(UUID(int=1))
    ui = FakeStreamlit(selected=(first, first, running))
    application = FakeApplication(
        (_summary(1), _summary(2), _summary(3, state="running")),
        Ok(_comparison_output()),
    )

    render_compare(application, st_module=ui)

    assert application.compared is None
    assert _calls(ui, "button")[-1][2]["disabled"] is True
    error_text = " ".join(str(call) for call in _calls(ui, "error"))
    assert "distinct" in error_text
    assert "successful" in error_text or "not found" in error_text


def test_no_common_session_error_is_structured_and_disclosure_remains_visible() -> None:
    ui = FakeStreamlit(selected=(str(UUID(int=1)), str(UUID(int=2))))
    result = Err(
        (_error("The selected runs have no common sessions."),),
        preserve_order=True,
    )
    application = FakeApplication((_summary(1), _summary(2)), result)

    render_compare(application, st_module=ui)

    assert application.compared is not None
    assert any(
        "no common sessions" in str(call).lower() for call in _calls(ui, "error")
    )
    assert any(call[0] == "info" for call in ui.calls)
