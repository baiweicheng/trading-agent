"""Focused identity-boundary and lifecycle examples for domain manifests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import pytest

from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.domain.evaluation import (
    EnvironmentFingerprint,
    RunContentIdentity,
    RunManifest,
    RunOperationalMetadata,
    ScientificArtifactReference,
)
from quant_research_platform.domain.execution import (
    JobOperation,
    JobState,
    RunState,
    is_legal_job_transition,
    is_legal_run_transition,
    require_legal_job_transition,
    require_legal_run_transition,
)
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotLineage,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.domain.strategy import MomentumStrategyParameters

_CHECKSUM_A = "a" * 64
_CHECKSUM_B = "b" * 64
_CHECKSUM_C = "c" * 64
_CHECKSUM_D = "d" * 64
_CHECKSUM_E = "e" * 64
_REQUESTED_RANGE = DateRange(date(2024, 1, 2), date(2024, 1, 5))
_CREATED_AT = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
_RUN_ID_ONE = UUID("00000000-0000-4000-8000-000000000011")
_RUN_ID_TWO = UUID("00000000-0000-4000-8000-000000000012")


def _snapshot_manifest(
    *,
    created_at: datetime = _CREATED_AT,
    local_path: str = "/first-machine/data/snapshots/manifest.json",
    job_id: str = "job-one",
    parent_snapshot_id: str | None = None,
    row_count: int = 1,
    object_checksum: str = _CHECKSUM_A,
    configuration_checksum: str = _CHECKSUM_B,
) -> SnapshotManifest:
    object_ref = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=object_checksum,
        relative_uri=(
            "objects/normalized/symbol=AAPL/year=2024/"
            f"sha256={object_checksum}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=row_count,
        byte_size=512,
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    content_identity = SnapshotContentIdentity(
        provider="yfinance",
        requested_range=_REQUESTED_RANGE,
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity(
            name="XNYS",
            version="exchange_calendars/4.5",
            schedule_checksum=_CHECKSUM_C,
        ),
        configuration_checksum=configuration_checksum,
        objects=(object_ref,),
        validation_report_checksum=_CHECKSUM_D,
        validation_summary=ValidationSummary(
            accepted_row_count=row_count,
            quarantined_row_count=0,
            collapsed_duplicate_count=0,
            gap_count=0,
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    return SnapshotManifest(
        content_identity=content_identity,
        operational_metadata=OperationalMetadata(
            created_at=created_at,
            detection_times=(created_at + timedelta(seconds=3),),
            job_id=job_id,
            local_manifest_path=local_path,
        ),
        lineage=SnapshotLineage(parent_snapshot_id=parent_snapshot_id),
    )


def _run_manifest(
    *,
    run_id: UUID,
    created_at: datetime,
    configuration_checksum: str = _CHECKSUM_B,
    artifact_checksum: str = _CHECKSUM_D,
    state: RunState = RunState.RUNNING,
) -> RunManifest:
    content_identity = RunContentIdentity(
        schema_version="run_manifest_v1",
        snapshot_id=f"snap_{_CHECKSUM_A}",
        strategy_identifier="monthly_momentum_v1",
        strategy_parameters=MomentumStrategyParameters(position_count=1),
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 5),
        configuration_checksum=configuration_checksum,
        environment_fingerprint=EnvironmentFingerprint(
            python_version="3.11.9",
            operating_system="macOS 14.5",
            architecture="arm64",
            dependencies=(),
            source_revision="abc123",
            source_dirty=False,
            deterministic_seed=0,
            effective_source_checksum=_CHECKSUM_C,
        ),
        scientific_artifacts=(
            ScientificArtifactReference(role="orders", checksum=artifact_checksum),
        ),
    )
    return RunManifest(
        content_identity=content_identity,
        operational_metadata=RunOperationalMetadata(
            run_id=run_id,
            state=state,
            created_at=created_at,
            started_at=created_at + timedelta(seconds=1),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )


def test_snapshot_identity_excludes_operational_time_path_job_and_lineage() -> None:
    first = _snapshot_manifest()
    copied = _snapshot_manifest(
        created_at=_CREATED_AT + timedelta(days=17),
        local_path="/copied-machine/research/snapshots/manifest.json",
        job_id="job-two",
        parent_snapshot_id=f"snap_{_CHECKSUM_E}",
    )

    assert copied.snapshot_id == first.snapshot_id
    assert copied.content_identity_checksum == first.content_identity_checksum
    assert copied.manifest_checksum != first.manifest_checksum
    assert copied.to_content_identity_dict() == first.to_content_identity_dict()


@pytest.mark.parametrize(
    ("change", "expected_path"),
    [
        (
            {"row_count": 2},
            ("objects", 0, "row_count"),
        ),
        (
            {"object_checksum": _CHECKSUM_E},
            ("objects", 0, "checksum"),
        ),
        (
            {"configuration_checksum": _CHECKSUM_E},
            ("configuration_checksum",),
        ),
    ],
)
def test_snapshot_identity_changes_for_scientific_rows_checksums_and_configuration(
    change: dict[str, int | str], expected_path: tuple[str | int, ...]
) -> None:
    baseline = _snapshot_manifest()
    changed = _snapshot_manifest(**change)

    assert changed.snapshot_id != baseline.snapshot_id
    assert changed.content_identity_checksum != baseline.content_identity_checksum

    baseline_value: object = baseline.to_content_identity_dict()
    changed_value: object = changed.to_content_identity_dict()
    for segment in expected_path:
        baseline_value = baseline_value[segment]  # type: ignore[index]
        changed_value = changed_value[segment]  # type: ignore[index]
    assert changed_value != baseline_value


def test_run_scientific_identity_excludes_operational_ids_and_times() -> None:
    first = _run_manifest(run_id=_RUN_ID_ONE, created_at=_CREATED_AT)
    rerun = _run_manifest(
        run_id=_RUN_ID_TWO,
        created_at=_CREATED_AT + timedelta(days=2),
    )

    assert rerun.scientific_checksum == first.scientific_checksum
    assert rerun.canonical_scientific_bytes() == first.canonical_scientific_bytes()
    assert rerun.operational_metadata.run_id != first.operational_metadata.run_id


@pytest.mark.parametrize(
    "change",
    [
        {"configuration_checksum": _CHECKSUM_E},
        {"artifact_checksum": _CHECKSUM_E},
    ],
)
def test_run_scientific_identity_changes_for_configuration_and_artifact_checksums(
    change: dict[str, str],
) -> None:
    baseline = _run_manifest(run_id=_RUN_ID_ONE, created_at=_CREATED_AT)
    changed = _run_manifest(
        run_id=_RUN_ID_TWO,
        created_at=_CREATED_AT + timedelta(days=1),
        **change,
    )

    assert changed.scientific_checksum != baseline.scientific_checksum
    assert changed.canonical_scientific_bytes() != baseline.canonical_scientific_bytes()


def test_terminal_run_metadata_is_an_immutable_record() -> None:
    terminal = RunOperationalMetadata(
        run_id=_RUN_ID_ONE,
        state=RunState.FAILED,
        created_at=_CREATED_AT,
        started_at=_CREATED_AT + timedelta(seconds=1),
        ended_at=_CREATED_AT + timedelta(seconds=2),
    )

    with pytest.raises(FrozenInstanceError):
        terminal.state = RunState.SUCCEEDED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("current", "target", "operation", "expected"),
    [
        (JobState.NOT_STARTED, JobState.RUNNING, JobOperation.INGESTION, True),
        (JobState.RUNNING, JobState.SUCCEEDED, JobOperation.BACKTEST, True),
        (
            JobState.RUNNING,
            JobState.PARTIALLY_SUCCEEDED,
            JobOperation.INGESTION,
            True,
        ),
        (
            JobState.RUNNING,
            JobState.PARTIALLY_SUCCEEDED,
            JobOperation.EVALUATION,
            False,
        ),
        (JobState.RUNNING, JobState.FAILED, JobOperation.COMPARISON, True),
        (JobState.NOT_STARTED, JobState.SUCCEEDED, JobOperation.INGESTION, False),
        (JobState.SUCCEEDED, JobState.FAILED, JobOperation.INGESTION, False),
        (JobState.FAILED, JobState.RUNNING, JobOperation.INGESTION, False),
    ],
)
def test_job_state_machine_accepts_only_declared_transitions(
    current: JobState,
    target: JobState,
    operation: JobOperation,
    expected: bool,
) -> None:
    assert is_legal_job_transition(current, target, operation=operation) is expected
    if expected:
        assert require_legal_job_transition(
            current,
            target,
            operation=operation,
        ) is target
    else:
        with pytest.raises(ValueError, match="illegal job state transition"):
            require_legal_job_transition(current, target, operation=operation)


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (RunState.RUNNING, RunState.SUCCEEDED, True),
        (RunState.RUNNING, RunState.FAILED, True),
        (RunState.SUCCEEDED, RunState.FAILED, False),
        (RunState.FAILED, RunState.SUCCEEDED, False),
    ],
)
def test_run_state_machine_terminalizes_once(
    current: RunState,
    target: RunState,
    expected: bool,
) -> None:
    assert is_legal_run_transition(current, target) is expected
    if expected:
        assert require_legal_run_transition(current, target) is target
    else:
        with pytest.raises(ValueError, match="illegal run state transition"):
            require_legal_run_transition(current, target)
