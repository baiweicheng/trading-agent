from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from quant_research_platform.config.serializer import REDACTION_MARKER, Redactor
from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.ui.components import (
    bounded_page_size,
    render_artifact_download,
    render_chart,
    render_limitation_disclosure,
    render_table_page,
)
from quant_research_platform.ui.presenters import (
    present_backtest_result,
    present_configuration_resolution,
    present_progress,
    present_table_page,
)


@dataclass(frozen=True)
class SecretBearingDto:
    safe_value: str
    secret_value: str
    configuration_handle: object


class FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def subheader(self, *args: object, **kwargs: object) -> None:
        self._record("subheader", *args, **kwargs)

    def info(self, *args: object, **kwargs: object) -> None:
        self._record("info", *args, **kwargs)

    def warning(self, *args: object, **kwargs: object) -> None:
        self._record("warning", *args, **kwargs)

    def dataframe(self, *args: object, **kwargs: object) -> None:
        self._record("dataframe", *args, **kwargs)

    def caption(self, *args: object, **kwargs: object) -> None:
        self._record("caption", *args, **kwargs)

    def vega_lite_chart(self, *args: object, **kwargs: object) -> None:
        self._record("vega_lite_chart", *args, **kwargs)

    def download_button(self, *args: object, **kwargs: object) -> bool:
        self._record("download_button", *args, **kwargs)
        return False



def test_safe_presenter_excludes_secrets_and_opaque_handles() -> None:
    from quant_research_platform.ui.presenters import _safe_value

    redactor = Redactor(["literal-secret"])
    result = _safe_value(
        SecretBearingDto("visible", "literal-secret", object()), redactor
    )

    assert result == {"safe_value": "visible", "secret_value": REDACTION_MARKER}
    assert "literal-secret" not in repr(result)
    assert "configuration_handle" not in result



def test_configuration_presenter_drops_opaque_handle_and_keeps_paths() -> None:
    from pathlib import Path

    resolution = SimpleNamespace(
        handle=object(),
        view={
            "data_root": Path("data"),
            "secrets": {"https_proxy": "[REDACTED]"},
        },
    )

    rendered = present_configuration_resolution(resolution)

    assert "handle" not in rendered
    configuration = rendered["configuration"]
    assert configuration["data_root"] == "data"
    assert configuration["secrets"]["https_proxy"] == REDACTION_MARKER


def test_progress_presenter_redacts_warning_text() -> None:
    update = SimpleNamespace(
        job_id=uuid4(),
        operation="ingestion",
        state="running",
        stage="fetching",
        completed_units=2,
        total_units=5,
        elapsed_seconds=Decimal("1.25"),
        warnings=("provider token is literal-secret",),
    )

    result = present_progress(update, redactor=Redactor(["literal-secret"]))

    assert result["completed_units"] == 2
    assert result["warnings"] == [f"provider token is {REDACTION_MARKER}"]
    assert "literal-secret" not in repr(result)



def test_presenters_require_limitation_disclosure() -> None:
    result = SimpleNamespace(
        run_id="run-1",
        snapshot_id="snap-1",
        evaluation_range=SimpleNamespace(start="2024-01-01", end="2024-01-02"),
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
    )

    with pytest.raises(TypeError, match="LimitationDisclosure"):
        present_backtest_result(result)

    result.limitation_disclosure = LimitationDisclosure.current()
    rendered = present_backtest_result(result)
    assert rendered["limitation_disclosure"]["version"]



def test_ordinary_table_presenter_and_component_are_bounded() -> None:
    page = SimpleNamespace(
        rows=tuple({"row": index} for index in range(150)),
        page=0,
        page_size=100,
        total=150,
        columns=("row",),
    )

    rendered = present_table_page(page, configured_page_size=37)
    assert rendered["page_size"] == 37
    assert rendered["row_count"] == 37
    assert len(rendered["rows"]) == 37
    assert bounded_page_size(1000, 37) == 37

    ui = FakeStreamlit()
    component_rendered = render_table_page(
        page, configured_page_size=25, requested_page_size=1000, st_module=ui
    )
    assert component_rendered["row_count"] == 25
    dataframe_calls = [call for call in ui.calls if call[0] == "dataframe"]
    assert len(dataframe_calls) == 1
    assert len(dataframe_calls[0][1][0]) == 25



def test_disclosure_and_download_are_visible_separate_affordances() -> None:
    ui = FakeStreamlit()
    disclosure = LimitationDisclosure.current()
    rendered_disclosure = render_limitation_disclosure(disclosure, st_module=ui)
    assert rendered_disclosure["version"] == disclosure.version
    assert any(call[0] == "info" for call in ui.calls)

    artifact = SimpleNamespace(
        checksum="a" * 64,
        role="equity_curve",
        relative_uri="artifacts/equity.json",
        byte_size=5,
        media_type="application/json",
        availability="available",
        payload=b"table",
    )
    download = render_artifact_download(artifact, st_module=ui)
    assert download["downloaded"] is False
    chart = render_chart(
        {"mark": "line", "data": {"values": []}}, st_module=ui
    )
    assert chart["mark"] == "line"
    assert any(call[0] == "vega_lite_chart" for call in ui.calls)
    assert all(call[0] != "dataframe" for call in ui.calls)
