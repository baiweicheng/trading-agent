from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

from quant_research_platform.application.services import ConfigurationHandle
from quant_research_platform.domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.ui import app


class FakeStreamlit:
    def __init__(self, *, submit: bool = False, ingest: bool = False) -> None:
        self.submit = submit
        self.ingest = ingest
        self.session_state: dict[str, object] = {}
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def form(self, key: str) -> object:
        self._record("form", key)
        return nullcontext()

    def file_uploader(self, *args: object, **kwargs: object) -> None:
        self._record("file_uploader", *args, **kwargs)
        return None

    def selectbox(self, label: str, options: tuple[str, ...], **kwargs: object) -> str:
        self._record("selectbox", label, options, **kwargs)
        return options[int(kwargs.get("index", 0))]

    def text_input(self, label: str, value: str, **kwargs: object) -> str:
        self._record("text_input", label, value, **kwargs)
        return value

    def date_input(self, label: str, value: object, **kwargs: object) -> object:
        self._record("date_input", label, value, **kwargs)
        return value

    def number_input(self, label: str, **kwargs: object) -> object:
        self._record("number_input", label, **kwargs)
        return kwargs["value"]

    def form_submit_button(self, *args: object, **kwargs: object) -> bool:
        self._record("form_submit_button", *args, **kwargs)
        return self.submit

    def button(self, *args: object, **kwargs: object) -> bool:
        self._record("button", *args, **kwargs)
        return self.ingest

    def empty(self) -> FakeStreamlit:
        self._record("empty")
        return self

    def __getattr__(self, name: str):
        def method(*args: object, **kwargs: object) -> None:
            self._record(name, *args, **kwargs)

        return method


class FakeApplication:
    def __init__(
        self, resolution: object, ingestion_result: object | None = None
    ) -> None:
        self.resolution = resolution
        self.ingestion_result = ingestion_result
        self.ingest_handle: ConfigurationHandle | None = None
        self.progress_seen: object | None = None

    def resolve_configuration(self, *args: object, **kwargs: object) -> object:
        return self.resolution

    def ingest(
        self,
        request: object,
        handle: ConfigurationHandle,
        *,
        progress,
    ) -> object:
        self.ingest_handle = handle
        update = SimpleNamespace(
            job_id=uuid4(),
            operation="ingestion",
            state="running",
            stage="fetching",
            completed_units=1,
            total_units=1,
            elapsed_seconds=0.5,
            warnings=("provider warning",),
        )
        progress(update)
        self.progress_seen = update
        return Ok(self.ingestion_result)


def _error() -> ActionableError:
    return ActionableError(
        operation="configuration.resolve",
        category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
        message="Configuration is invalid.",
        corrective_action="Correct the highlighted fields and validate again.",
        field_path="data.universe",
    )


def _resolution() -> object:
    return Ok(
        SimpleNamespace(
            handle=ConfigurationHandle(uuid4()),
            view={"provider": "yfinance", "secret": "literal-secret"},
        )
    )


def _ingestion_result() -> object:
    return SimpleNamespace(
        status="completed",
        snapshot_id="snapshot-123",
        errors=(),
        limitation_disclosure=LimitationDisclosure.current(),
    )


def _calls(
    ui: FakeStreamlit, name: str
) -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    return [call for call in ui.calls if call[0] == name]


def test_configuration_failure_keeps_ingest_disabled_and_renders_actionable_error() -> (
    None
):
    ui = FakeStreamlit(submit=True, ingest=True)
    application = FakeApplication(Err((_error(),)))

    app.render_configure_ingest(application, st_module=ui, project_root=".")

    button_call = _calls(ui, "button")[-1]
    assert button_call[2]["disabled"] is True
    assert application.ingest_handle is None
    assert app._STATE_HANDLE not in ui.session_state
    assert "Corrective action:" in " ".join(
        str(call[1][0]) for call in _calls(ui, "error")
    )


def test_valid_resolution_stores_only_opaque_handle_and_ingestion_progress() -> None:
    resolution = _resolution()
    handle = resolution.value.handle
    ui = FakeStreamlit(submit=True, ingest=True)
    application = FakeApplication(resolution, _ingestion_result())

    app.render_configure_ingest(application, st_module=ui, project_root=".")

    assert application.ingest_handle == handle
    assert ui.session_state[app._STATE_HANDLE] == handle
    rendered_configuration = ui.session_state[app._STATE_CONFIGURATION]
    assert "handle" not in rendered_configuration
    assert "literal-secret" not in repr(rendered_configuration)
    assert ui.session_state[app._STATE_PROGRESS]["completed_units"] == 1
    assert ui.session_state[app._STATE_RESULT]["snapshot_id"] == "snapshot-123"
    assert _calls(ui, "button")[-1][2]["disabled"] is False
    assert any(
        "Snapshot ID: snapshot-123" in str(call[1][0]) for call in _calls(ui, "success")
    )


def test_form_uses_stable_key_and_number_inputs_preserve_value_keyword() -> None:
    ui = FakeStreamlit()
    application = FakeApplication(Err((_error(),)))

    app.render_configure_ingest(application, st_module=ui, project_root=".")

    assert _calls(ui, "form")[0][1] == ("configuration-form",)
    assert _calls(ui, "number_input")
    assert all("value" in call[2] for call in _calls(ui, "number_input"))
