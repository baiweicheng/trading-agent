"""Offline tests for application experiment orchestration."""

from __future__ import annotations

from datetime import UTC, date, datetime
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

from quant_research_platform.application.experiments import (
    ArtifactDescriptor,
    ExperimentTracker,
    RunHandle,
    RunInputs,
)
from quant_research_platform.domain.errors import Err, ErrorCategory, LimitationDisclosure, Ok
from quant_research_platform.domain.execution import RunState


NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)
SNAPSHOT = "snap_" + "a" * 64
ARTIFACT_CHECKSUM = sha256(b"payload").hexdigest()
MANIFEST_CHECKSUM = "c" * 64
RUN_ONE = UUID("00000000-0000-0000-0000-000000000001")
RUN_TWO = UUID("00000000-0000-0000-0000-000000000002")


class FakeMetadata:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.runs: dict[UUID, SimpleNamespace] = {}
        self.artifacts: dict[str, SimpleNamespace] = {}
        self.links: dict[UUID, tuple[SimpleNamespace, ...]] = {}

    def create_run(self, **values: object) -> SimpleNamespace:
        run_id = values["run_id"]
        assert isinstance(run_id, UUID)
        self.events.append("metadata.create_run")
        record = SimpleNamespace(
            run_id=run_id,
            state=RunState.RUNNING,
            mlflow_run_id=None,
        )
        self.runs[run_id] = record
        return record

    def get_run(self, run_id: UUID) -> SimpleNamespace:
        return self.runs[run_id]

    def record_artifact(self, reference: object, **_: object) -> None:
        checksum = str(getattr(reference, "checksum"))
        self.events.append("metadata.record_artifact")
        self.artifacts[checksum] = SimpleNamespace(
            checksum=checksum,
            relative_uri=getattr(reference, "relative_uri"),
            byte_size=getattr(reference, "byte_size"),
            media_type=getattr(reference, "media_type"),
            availability="available",
        )

    def get_artifact(self, checksum: str) -> SimpleNamespace:
        return self.artifacts[checksum]

    def set_artifact_availability(self, checksum: str, availability: str) -> None:
        self.artifacts[checksum].availability = availability

    def list_run_artifacts(self, run_id: UUID) -> tuple[SimpleNamespace, ...]:
        return self.links.get(run_id, ())


class FakeTracker:
    def __init__(self, metadata: FakeMetadata) -> None:
        self.metadata = metadata
        self.handles: dict[UUID, RunHandle] = {}
        self.finalized: list[tuple[UUID, RunState]] = []
        self.open_error = False

    def allocate_run(self, **values: object) -> RunHandle:
        self.metadata.events.append("tracker.allocate_run")
        run_id = values["run_id"]
        assert isinstance(run_id, UUID)
        handle = RunHandle(run_id, f"mlflow-{run_id}")
        self.handles[run_id] = handle
        return handle

    def finalize_success(self, run: RunHandle, result: object) -> RunHandle:
        del result
        self.finalized.append((run.run_id, RunState.SUCCEEDED))
        self.metadata.runs[run.run_id].state = RunState.SUCCEEDED
        links = (SimpleNamespace(checksum=ARTIFACT_CHECKSUM, role="equity", scientific=True),)
        self.metadata.links[run.run_id] = links
        return RunHandle(run.run_id, run.mlflow_run_id, RunState.SUCCEEDED)

    def finalize_failure(self, run: RunHandle, errors: object, diagnostics: object = ()) -> RunHandle:
        assert tuple(errors)
        del diagnostics
        self.finalized.append((run.run_id, RunState.FAILED))
        self.metadata.runs[run.run_id].state = RunState.FAILED
        return RunHandle(run.run_id, run.mlflow_run_id, RunState.FAILED)

    def open_verified_artifact(self, run_id: UUID, checksum: str) -> object:
        if self.open_error:
            raise OSError("corrupt bytes")
        assert self.metadata.links[run_id][0].checksum == checksum
        return SimpleNamespace(
            checksum=checksum,
            relative_uri="artifacts/sha256/bb/" + checksum,
            byte_size=7,
            media_type="application/json",
            stream=lambda: iter((b"payload",)),
        )


class FakeArtifactStore:
    def publish_artifact(self, payload: bytes, metadata: object) -> SimpleNamespace:
        assert payload == b"payload"
        assert metadata["checksum"] == ARTIFACT_CHECKSUM  # type: ignore[index]
        return SimpleNamespace(
            checksum=ARTIFACT_CHECKSUM,
            relative_uri="artifacts/sha256/bb/" + ARTIFACT_CHECKSUM,
        )


def _inputs(run_id: UUID | None = None) -> RunInputs:
    return RunInputs(
        run_id=run_id,
        snapshot_id=SNAPSHOT,
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 9),
        strategy_parameters={"position_count": 2},
        universe=("AAPL", "MSFT"),
    )


def _result() -> SimpleNamespace:
    artifact = ArtifactDescriptor(
        checksum=ARTIFACT_CHECKSUM,
        role="equity",
        relative_uri="artifacts/sha256/bb/" + ARTIFACT_CHECKSUM,
        byte_size=7,
        media_type="application/json",
        payload=b"payload",
    )
    return SimpleNamespace(
        snapshot_id=SNAPSHOT,
        manifest_checksum=MANIFEST_CHECKSUM,
        manifest_uri="runs/example/manifest.json",
        limitation_disclosure=LimitationDisclosure.current(),
        artifacts=(artifact,),
    )


def test_allocation_persists_running_row_before_tracking_adapter() -> None:
    metadata = FakeMetadata()
    tracker = ExperimentTracker(metadata, FakeTracker(metadata))

    result = tracker.create_run(_inputs(RUN_ONE))

    assert isinstance(result, Ok)
    assert result.value.run_id == RUN_ONE
    assert metadata.events == ["metadata.create_run", "tracker.allocate_run"]
    assert metadata.runs[RUN_ONE].state is RunState.RUNNING


def test_success_publishes_artifact_and_terminal_replay_rejects_conflict() -> None:
    metadata = FakeMetadata()
    adapter = FakeTracker(metadata)
    tracker = ExperimentTracker(metadata, adapter, FakeArtifactStore())
    assert isinstance(tracker.create_run(_inputs(RUN_ONE)), Ok)

    first = tracker.succeed(RUN_ONE, _result())
    replay = tracker.succeed(RUN_ONE, _result())
    conflicting = tracker.succeed(RUN_ONE, SimpleNamespace(
        snapshot_id=SNAPSHOT,
        manifest_checksum="d" * 64,
        manifest_uri="runs/example/other.json",
        limitation_disclosure=LimitationDisclosure.current(),
        artifacts=(),
    ))

    assert isinstance(first, Ok)
    assert first.value.state is RunState.SUCCEEDED
    assert isinstance(replay, Ok)
    assert isinstance(conflicting, Err)
    assert metadata.artifacts[ARTIFACT_CHECKSUM].availability == "available"
    assert adapter.finalized == [(RUN_ONE, RunState.SUCCEEDED)]


def test_failure_preserves_actionable_diagnostics_and_isolated_prior_run() -> None:
    metadata = FakeMetadata()
    adapter = FakeTracker(metadata)
    tracker = ExperimentTracker(metadata, adapter)
    assert isinstance(tracker.create_run(_inputs(RUN_ONE)), Ok)
    assert isinstance(tracker.create_run(_inputs(RUN_TWO)), Ok)

    from quant_research_platform.domain.errors import ActionableError

    diagnostic = ActionableError(
        operation="backtest.execute",
        category=ErrorCategory.BACKTEST_INVARIANT,
        message="ledger invariant failed",
        corrective_action="inspect the diagnostic artifact and retry",
    )
    failed = tracker.fail(RUN_TWO, (diagnostic,))

    assert isinstance(failed, Ok)
    assert failed.value.state is RunState.FAILED
    assert metadata.runs[RUN_ONE].state is RunState.RUNNING
    assert metadata.runs[RUN_TWO].state is RunState.FAILED


def test_corrupt_artifact_returns_integrity_error_and_marks_invalid() -> None:
    metadata = FakeMetadata()
    adapter = FakeTracker(metadata)
    tracker = ExperimentTracker(metadata, adapter)
    assert isinstance(tracker.create_run(_inputs(RUN_ONE)), Ok)
    assert isinstance(tracker.succeed(RUN_ONE, _result()), Ok)
    adapter.open_error = True

    opened = tracker.open_verified_artifact(RUN_ONE, ARTIFACT_CHECKSUM)

    assert isinstance(opened, Err)
    assert opened.errors[0].category is ErrorCategory.INTEGRITY_CHECKSUM
    assert metadata.artifacts[ARTIFACT_CHECKSUM].availability == "invalid"


def test_operationally_distinct_runs_can_share_scientific_artifact() -> None:
    metadata = FakeMetadata()
    tracker = ExperimentTracker(metadata, FakeTracker(metadata), FakeArtifactStore())
    first = tracker.create_run(_inputs())
    second = tracker.create_run(_inputs())

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.run_id != second.value.run_id
    assert first.value.run_id != second.value.run_id
    assert metadata.events.count("metadata.create_run") == 2
