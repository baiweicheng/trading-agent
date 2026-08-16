"""Fault-injection coverage for the local run finalization protocol."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("duckdb")

from quant_research_platform.application.experiments import (  # noqa: E402
    ArtifactDescriptor,
    ExperimentTracker,
    RunInputs,
)
from quant_research_platform.domain.errors import (
    Err,
    LimitationDisclosure,
)  # noqa: E402
from quant_research_platform.domain.evaluation import (
    MetricScope,
    calculate_evaluation_metrics,
)  # noqa: E402
from quant_research_platform.domain.execution import RunState  # noqa: E402
from quant_research_platform.domain.manifests import (
    ContentAddressedObjectRef,
    ObjectKind,
)  # noqa: E402
from quant_research_platform.infrastructure.duckdb_metadata import (  # noqa: E402
    DuckDBMetadataStore,
    ImmutableMetadataError,
    RunArtifactLink,
    RunFinalization,
)
from quant_research_platform.infrastructure.mlflow_tracker import (  # noqa: E402
    LocalMlflowTracker,
    MlflowTrackerError,
    RunHandle,
)

NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)
SNAPSHOT = "snap_" + "a" * 64
MANIFEST = "b" * 64
ARTIFACT = "c" * 64
CORRUPT_ARTIFACT = "f" * 64
RUN_ONE = UUID("00000000-0000-0000-0000-000000000201")
RUN_TWO = UUID("00000000-0000-0000-0000-000000000202")


class FaultClient:
    """Deterministic MLflow client double with one-shot terminal failure."""

    def __init__(self, *, fail_terminal: bool = False) -> None:
        self.fail_terminal = fail_terminal
        self.runs = 0
        self.terminated: list[str] = []
        self.logged_text: list[str] = []

    def get_experiment_by_name(self, name: str) -> None:
        del name
        return None

    def create_experiment(self, name: str) -> str:
        assert name == "quant_research_platform"
        return "experiment-1"

    def create_run(self, experiment_id: str, **_: object) -> SimpleNamespace:
        assert experiment_id == "experiment-1"
        self.runs += 1
        return SimpleNamespace(info=SimpleNamespace(run_id=f"mlflow-{self.runs}"))

    def log_param(self, run_id: str, key: str, value: object) -> None:
        del run_id, key, value

    def log_metric(self, run_id: str, key: str, value: float, **_: object) -> None:
        del run_id, key, value

    def set_tag(self, run_id: str, key: str, value: object) -> None:
        del run_id, key, value

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        del run_id, artifact_file
        self.logged_text.append(text)

    def set_terminated(self, run_id: str, **_: object) -> None:
        if self.fail_terminal:
            raise OSError("injected MLflow terminalization failure")
        self.terminated.append(run_id)


class FaultMetadata:
    """Proxy that fails one durable step while retaining the real DuckDB store."""

    def __init__(self, store: DuckDBMetadataStore, failure: str | None = None) -> None:
        self.store = store
        self.failure = failure
        self.failed = False

    def _fail_once(self, point: str) -> None:
        if self.failure == point and not self.failed:
            self.failed = True
            raise OSError(f"injected {point} failure")

    def create_run(self, **values: object) -> object:
        return self.store.create_run(**values)  # type: ignore[arg-type]

    def set_mlflow_run_id(self, run_id: UUID, mlflow_run_id: str) -> object:
        return self.store.set_mlflow_run_id(run_id, mlflow_run_id)

    def create_finalization_intent(
        self, run_id: UUID, finalization: object, **values: object
    ) -> object:
        self._fail_once("intent_commit")
        return self.store.create_finalization_intent(run_id, finalization, **values)  # type: ignore[arg-type]

    def mark_finalization_mlflow_synced(self, run_id: UUID, **values: object) -> object:
        return self.store.mark_finalization_mlflow_synced(run_id, **values)  # type: ignore[arg-type]

    def get_finalization_intent(self, run_id: UUID) -> object:
        return self.store.get_finalization_intent(run_id)

    def finalize_run(
        self, run_id: UUID, finalization: object, **values: object
    ) -> object:
        self._fail_once("terminal_commit")
        return self.store.finalize_run(run_id, finalization, **values)  # type: ignore[arg-type]

    def get_run(self, run_id: UUID) -> object:
        return self.store.get_run(run_id)

    def get_artifact(self, checksum: str) -> object:
        return self.store.get_artifact(checksum)

    def list_pending_finalization_intents(self) -> object:
        return self.store.list_pending_finalization_intents()

    def record_artifact(self, **values: object) -> object:
        return self.store.record_artifact(**values)  # type: ignore[arg-type]

    def set_artifact_availability(self, checksum: str, availability: str) -> object:
        return self.store.set_artifact_availability(checksum, availability)


class FailingArtifactStore:
    def publish_artifact(self, payload: bytes, metadata: object) -> object:
        del payload, metadata
        raise OSError("injected artifact publication failure")


def _metrics() -> object:
    return calculate_evaluation_metrics(
        MetricScope.STRATEGY,
        (Decimal("100"), Decimal("101")),
    )


def _result(*, checksum: str = ARTIFACT, manifest: str = MANIFEST) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=SimpleNamespace(strategy_metrics=_metrics()),
        manifest_checksum=manifest,
        manifest_uri="runs/manifest.json",
        artifacts=(
            SimpleNamespace(
                checksum=checksum,
                role="equity",
                relative_uri="runs/equity.parquet",
                byte_size=17,
                scientific=True,
            ),
        ),
    )


def _artifact_descriptor() -> ArtifactDescriptor:
    return ArtifactDescriptor(
        checksum="c664a72fe58359a96d8db6722cb40ff2b51a235e12f02c9043a61b2405e5b1fb",
        role="equity",
        relative_uri="runs/equity.parquet",
        byte_size=16,
        media_type="application/octet-stream",
        payload=b"scientific bytes",
    )


def _inputs(run_id: UUID) -> RunInputs:
    return RunInputs(
        run_id=run_id,
        snapshot_id=SNAPSHOT,
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 31),
        strategy_parameters={"position_count": 3},
        universe=("AAPL", "MSFT"),
        configuration_checksum="d" * 64,
        environment_checksum="e" * 64,
        limitation_disclosure=LimitationDisclosure.current(),
    )


def _store_artifact(store: DuckDBMetadataStore, checksum: str = ARTIFACT) -> None:
    store.record_artifact(
        ContentAddressedObjectRef(
            object_kind=ObjectKind.ARTIFACT,
            checksum=checksum,
            relative_uri="runs/" + checksum + ".parquet",
            schema_version="artifact_v1",
            row_count=1,
            byte_size=17,
            media_type="application/octet-stream",
        ),
        artifact_kind="equity",
        created_at=NOW,
    )


def _allocate(
    store: DuckDBMetadataStore,
    run_id: UUID,
    *,
    client: FaultClient | None = None,
    failure: str | None = None,
) -> tuple[LocalMlflowTracker, RunHandle]:
    tracker = LocalMlflowTracker(
        client=client or FaultClient(),
        metadata_store=FaultMetadata(store, failure=failure),
    )
    handle = tracker.allocate_run(
        run_id=run_id,
        snapshot_id=SNAPSHOT,
        strategy_id="monthly_momentum_v1",
        strategy_parameters={"position_count": 3},
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 31),
        universe=("AAPL", "MSFT"),
        configuration_checksum="d" * 64,
        environment_checksum="e" * 64,
        created_at=NOW,
        started_at=NOW,
    )
    return tracker, handle


def test_failure_before_artifact_publication_preserves_running_run_and_prior_valid_run(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    prior_tracker, prior = _allocate(store, RUN_ONE)
    _store_artifact(store)
    prior_tracker.finalize_success(prior, _result())

    tracker = LocalMlflowTracker(client=FaultClient(), metadata_store=store)
    application = ExperimentTracker(
        metadata_store=store,
        mlflow_tracker=tracker,
        artifact_store=FailingArtifactStore(),
    )
    allocated = application.create_run(_inputs(RUN_TWO))
    assert not isinstance(allocated, Err)
    failed = application.succeed(
        allocated.value,
        SimpleNamespace(
            snapshot_id=SNAPSHOT,
            manifest_checksum=MANIFEST,
            manifest_uri="runs/manifest.json",
            limitation_disclosure=LimitationDisclosure.current(),
            artifacts=(_artifact_descriptor(),),
        ),
    )

    assert isinstance(failed, Err)
    assert store.get_run(RUN_TWO).state is RunState.RUNNING
    assert store.get_run(RUN_ONE).state is RunState.SUCCEEDED
    assert store.get_artifact(ARTIFACT).availability.value == "available"
    store.close()


def test_failure_at_intent_commit_is_retryable_after_artifact_publication(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    _store_artifact(store)
    tracker, handle = _allocate(store, RUN_ONE, failure="intent_commit")

    with pytest.raises(OSError, match="intent_commit"):
        tracker.finalize_success(handle, _result())
    assert store.get_run(RUN_ONE).state is RunState.RUNNING
    assert store.get_artifact(ARTIFACT).availability.value == "available"
    assert store.list_pending_finalization_intents() == ()

    recovered_tracker, recovered_handle = _allocate(store, RUN_TWO)
    recovered_tracker.finalize_success(recovered_handle, _result())
    retry = LocalMlflowTracker(client=FaultClient(), metadata_store=store)
    retry.finalize_success(RunHandle(RUN_ONE, handle.mlflow_run_id), _result())
    assert store.get_run(RUN_ONE).state is RunState.SUCCEEDED
    assert store.get_run(RUN_TWO).state is RunState.SUCCEEDED
    store.close()


def test_mlflow_terminalization_failure_leaves_pending_intent_for_restart_replay(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    _store_artifact(store)
    failing_client = FaultClient(fail_terminal=True)
    tracker, handle = _allocate(store, RUN_ONE, client=failing_client)

    with pytest.raises(MlflowTrackerError):
        tracker.finalize_success(handle, _result())

    pending = store.list_pending_finalization_intents()
    assert [intent.run_id for intent in pending] == [RUN_ONE]
    assert store.get_run(RUN_ONE).state is RunState.RUNNING
    assert store.get_finalization_intent(RUN_ONE).mlflow_synced is False

    healthy = LocalMlflowTracker(client=FaultClient(), metadata_store=store)
    recovered = healthy.finalize_success(
        RunHandle(RUN_ONE, handle.mlflow_run_id), _result()
    )
    assert recovered.state is RunState.SUCCEEDED
    assert store.list_pending_finalization_intents() == ()
    store.close()


def test_duckdb_terminal_commit_failure_replays_exact_intent_without_mutating_prior_run(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    _store_artifact(store)
    prior_tracker, prior = _allocate(store, RUN_ONE)
    prior_tracker.finalize_success(prior, _result())

    tracker, handle = _allocate(store, RUN_TWO, failure="terminal_commit")
    with pytest.raises(OSError, match="terminal_commit"):
        tracker.finalize_success(handle, _result())

    interrupted = store.get_run(RUN_TWO)
    assert interrupted.state is RunState.RUNNING
    intent = store.get_finalization_intent(RUN_TWO)
    assert intent.mlflow_synced is True
    assert store.get_run(RUN_ONE).state is RunState.SUCCEEDED

    restarted = LocalMlflowTracker(client=FaultClient(), metadata_store=store)
    recovered = restarted.finalize_success(
        RunHandle(RUN_TWO, handle.mlflow_run_id), _result()
    )
    assert recovered.state is RunState.SUCCEEDED
    assert store.get_run(RUN_TWO).state is RunState.SUCCEEDED
    assert store.get_run(RUN_ONE).state is RunState.SUCCEEDED
    assert (
        store.get_finalization_intent(RUN_TWO).terminal_payload_checksum
        == intent.terminal_payload_checksum
    )
    store.close()


def test_corrupt_artifact_invalidates_new_attempt_without_invalidating_prior_run(
    tmp_path: object,
) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")  # type: ignore[operator]
    _store_artifact(store)
    _store_artifact(store, CORRUPT_ARTIFACT)
    prior_tracker, prior = _allocate(store, RUN_ONE)
    prior_tracker.finalize_success(prior, _result())

    store.set_artifact_availability(CORRUPT_ARTIFACT, "invalid")
    tracker, handle = _allocate(store, RUN_TWO)
    finalization = RunFinalization(
        desired_state=RunState.SUCCEEDED,
        manifest_checksum=MANIFEST,
        manifest_uri="runs/manifest.json",
        metrics=(_metrics(),),
        artifacts=(RunArtifactLink(CORRUPT_ARTIFACT, "equity", True),),
    )
    store.create_finalization_intent(RUN_TWO, finalization, created_at=NOW)
    store.mark_finalization_mlflow_synced(RUN_TWO, attempted_at=NOW)
    with pytest.raises(ImmutableMetadataError, match="unavailable"):
        store.finalize_run(RUN_TWO, finalization, ended_at=NOW)
    assert store.get_run(RUN_TWO).state is RunState.RUNNING
    assert store.get_run(RUN_ONE).state is RunState.SUCCEEDED
    assert store.get_artifact(CORRUPT_ARTIFACT).availability.value == "invalid"
    assert store.get_artifact(ARTIFACT).availability.value == "available"
    del tracker, handle
    store.close()
