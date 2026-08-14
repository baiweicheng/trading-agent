"""Offline contract tests for the local MLflow tracking boundary."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("duckdb")

from quant_research_platform.domain.evaluation import MetricScope, calculate_evaluation_metrics
from quant_research_platform.domain.errors import ActionableError, ErrorCategory
from quant_research_platform.domain.execution import RunState
from quant_research_platform.infrastructure.mlflow_tracker import (
    LocalMlflowTracker,
    MlflowTrackerError,
    RunHandle,
)

NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)
SNAPSHOT = "snap_" + "a" * 64
MANIFEST = "b" * 64
ARTIFACT = "c" * 64
RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
SECOND_RUN_ID = UUID("00000000-0000-0000-0000-000000000103")


class RecordingMetadata:
    """Small metadata port that records the contract's durable ordering."""

    def __init__(self, timeline: list[str] | None = None) -> None:
        self.events: list[str] = []
        self.timeline = timeline if timeline is not None else self.events
        self.runs: dict[UUID, SimpleNamespace] = {}
        self.intents: dict[UUID, SimpleNamespace] = {}

    def _record(self, event: str) -> None:
        self.events.append(event)
        if self.timeline is not self.events:
            self.timeline.append(event)

    def create_run(self, **values: object) -> SimpleNamespace:
        run_id = values["run_id"]
        assert isinstance(run_id, UUID)
        self._record("metadata.create_run")
        record = SimpleNamespace(
            run_id=run_id,
            state=RunState.RUNNING,
            mlflow_run_id=None,
        )
        self.runs[run_id] = record
        return record

    def set_mlflow_run_id(self, run_id: UUID, mlflow_run_id: str) -> SimpleNamespace:
        self._record("metadata.set_mlflow_run_id")
        current = self.runs[run_id].mlflow_run_id
        if current is not None and current != mlflow_run_id:
            raise ValueError("MLflow run ID is already bound")
        self.runs[run_id].mlflow_run_id = mlflow_run_id
        return self.runs[run_id]

    def create_finalization_intent(
        self, run_id: UUID, finalization: object, **_: object
    ) -> SimpleNamespace:
        self._record("metadata.create_finalization_intent")
        intent = SimpleNamespace(
            run_id=run_id,
            desired_state=getattr(finalization, "desired_state"),
            terminal_payload_checksum=getattr(finalization, "payload_checksum"),
            mlflow_synced=False,
        )
        self.intents[run_id] = intent
        return intent

    def mark_finalization_mlflow_synced(
        self, run_id: UUID, **_: object
    ) -> SimpleNamespace:
        self._record("metadata.mark_finalization_mlflow_synced")
        self.intents[run_id].mlflow_synced = True
        return self.intents[run_id]

    def get_finalization_intent(self, run_id: UUID) -> SimpleNamespace:
        return self.intents[run_id]

    def finalize_run(
        self, run_id: UUID, finalization: object, **_: object
    ) -> SimpleNamespace:
        self._record("metadata.finalize_run")
        self.runs[run_id].state = getattr(finalization, "desired_state")
        return self.runs[run_id]

    def get_run(self, run_id: UUID) -> SimpleNamespace:
        return self.runs[run_id]


class RecordingClient:
    def __init__(self, timeline: list[str] | None = None) -> None:
        self.events: list[str] = []
        self.timeline = timeline if timeline is not None else self.events
        self.params: list[tuple[str, str, object]] = []
        self.tags: list[tuple[str, str, object]] = []
        self.text: list[tuple[str, str, str]] = []
        self.terminated: list[tuple[str, str]] = []
        self.run_count = 0
    def _record(self, event: str) -> None:
        self.events.append(event)
        if self.timeline is not self.events:
            self.timeline.append(event)

    def get_experiment_by_name(self, name: str) -> None:
        del name
        return None

    def create_experiment(self, name: str) -> str:
        assert name == "quant_research_platform"
        self._record("mlflow.create_experiment")
        return "experiment-1"

    def create_run(self, experiment_id: str, **kwargs: object) -> SimpleNamespace:
        assert experiment_id == "experiment-1"
        tags = kwargs["tags"]
        assert tags["qrp.state"] == "running"  # type: ignore[index]
        self.run_count += 1
        mlflow_id = f"mlflow-contract-{self.run_count}"
        self.tags.extend((mlflow_id, str(key), value) for key, value in tags.items())  # type: ignore[union-attr]
        self._record("mlflow.create_run")
        return SimpleNamespace(info=SimpleNamespace(run_id=mlflow_id))

    def log_param(self, run_id: str, key: str, value: object) -> None:
        self._record("mlflow.log_param")
        self.params.append((run_id, key, value))

    def log_metric(self, run_id: str, key: str, value: float, **_: object) -> None:
        self._record("mlflow.log_metric")

    def set_tag(self, run_id: str, key: str, value: object) -> None:
        self._record("mlflow.set_tag")
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, **kwargs: object) -> None:
        self._record("mlflow.set_terminated")
        self.terminated.append((run_id, str(kwargs["status"])))

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self._record("mlflow.log_text")
        self.text.append((run_id, text, artifact_file))


def _tracker() -> tuple[LocalMlflowTracker, RecordingMetadata, RecordingClient]:
    timeline: list[str] = []
    metadata = RecordingMetadata(timeline)
    client = RecordingClient(timeline)
    return LocalMlflowTracker(client=client, metadata_store=metadata), metadata, client


def _metrics() -> object:
    return calculate_evaluation_metrics(
        MetricScope.STRATEGY,
        (Decimal("100"), Decimal("101")),
    )


def _result(*, manifest_checksum: str = MANIFEST, with_bytes: bool = False) -> SimpleNamespace:
    artifact = SimpleNamespace(
        checksum=ARTIFACT,
        role="equity",
        relative_uri="runs/" + ARTIFACT + ".parquet",
        byte_size=17,
        scientific=True,
    )
    if with_bytes:
        artifact.payload = b"scientific bytes"
    return SimpleNamespace(
        evaluation=SimpleNamespace(strategy_metrics=_metrics()),
        manifest_checksum=manifest_checksum,
        manifest_uri="runs/manifest.json",
        artifacts=(artifact,),
    )


def _inputs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "snapshot_id": SNAPSHOT,
        "strategy_id": "monthly_momentum_v1",
        "strategy_parameters": {"position_count": 3},
        "evaluation_start": date(2024, 1, 2),
        "evaluation_end": date(2024, 1, 31),
        "universe": ("AAPL", "MSFT"),
        "configuration_checksum": "d" * 64,
        "environment_checksum": "e" * 64,
        "configuration": {
            "https_proxy": "https://user:password@example.invalid",
            "nested": {"token": "password"},
        },
        "secret_values": ("https://user:password@example.invalid", "password"),
    }
    values.update(overrides)
    return values


def test_metadata_allocation_precedes_mlflow_and_mapping_is_one_to_one() -> None:
    tracker, metadata, client = _tracker()

    handle = tracker.allocate_run(**_inputs())

    assert handle == RunHandle(RUN_ID, "mlflow-contract-1")
    assert metadata.events[0] == "metadata.create_run"
    assert metadata.runs[RUN_ID].mlflow_run_id == handle.mlflow_run_id
    second = tracker.allocate_run(**_inputs(run_id=SECOND_RUN_ID))
    assert second.mlflow_run_id != handle.mlflow_run_id
    assert metadata.runs[SECOND_RUN_ID].mlflow_run_id == second.mlflow_run_id
    assert client.tags.count(("mlflow-contract-1", "qrp.run_id", str(RUN_ID))) == 1

    with pytest.raises(ValueError, match="already bound"):
        metadata.set_mlflow_run_id(RUN_ID, "another-mlflow-run")


def test_inputs_are_redacted_and_terminal_mlflow_payload_is_reference_only() -> None:
    tracker, _, client = _tracker()
    handle = tracker.allocate_run(**_inputs())
    terminal = tracker.finalize_success(handle, _result(with_bytes=True))

    assert terminal.state is RunState.SUCCEEDED
    assert all("password" not in str(item) for item in client.params)
    assert all("password" not in text for _, text, _ in client.text)
    reference_text = next(text for _, text, file in client.text if file == "artifact-references.json")
    assert ARTIFACT in reference_text
    assert "scientific bytes" not in reference_text
    assert not any(name == "log_artifact" for name in dir(client))


def test_terminal_success_and_failure_order_intent_mlflow_sync_then_metadata() -> None:
    tracker, metadata, client = _tracker()
    handle = tracker.allocate_run(**_inputs())
    tracker.finalize_success(handle, _result())

    intent_index = metadata.events.index("metadata.create_finalization_intent")
    sync_index = metadata.events.index("metadata.mark_finalization_mlflow_synced")
    terminal_index = metadata.events.index("metadata.finalize_run")
    mlflow_terminal_index = metadata.timeline.index("mlflow.set_terminated")
    assert intent_index < sync_index < terminal_index
    assert metadata.timeline.index("metadata.create_finalization_intent") < mlflow_terminal_index < metadata.timeline.index("metadata.mark_finalization_mlflow_synced") < metadata.timeline.index("metadata.finalize_run")
    assert client.terminated == [("mlflow-contract-1", "FINISHED")]
    assert mlflow_terminal_index >= 0

    tracker2, metadata2, client2 = _tracker()
    failed_handle = tracker2.allocate_run(**_inputs(run_id=UUID("00000000-0000-0000-0000-000000000102")))
    error = ActionableError(
        operation="backtest.execute",
        category=ErrorCategory.BACKTEST_INVARIANT,
        message="ledger invariant failed",
        corrective_action="inspect diagnostics and retry with a new Run ID",
    )
    failed = tracker2.finalize_failure(failed_handle, (error,))
    assert failed.state is RunState.FAILED
    assert client2.terminated == [("mlflow-contract-1", "FAILED")]
    assert metadata2.events.index("metadata.create_finalization_intent") < metadata2.events.index("metadata.finalize_run")


def test_exact_terminal_replay_is_idempotent_but_conflicting_payload_is_rejected() -> None:
    tracker, metadata, client = _tracker()
    handle = tracker.allocate_run(**_inputs())
    first = tracker.finalize_success(handle, _result())
    replay = tracker.finalize_success(first, _result())

    assert replay == RunHandle(RUN_ID, "mlflow-contract-1", RunState.SUCCEEDED)
    assert client.terminated == [("mlflow-contract-1", "FINISHED")]
    assert metadata.events.count("metadata.finalize_run") == 1

    with pytest.raises(MlflowTrackerError, match="immutable"):
        tracker.finalize_success(first, _result(manifest_checksum="f" * 64))
