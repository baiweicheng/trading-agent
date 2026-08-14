"""Integration coverage for the typed application facade and local read paths."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import SecretStr

pytest.importorskip("duckdb")

from quant_research_platform.application.backtests import BacktestRequest  # noqa: E402
from quant_research_platform.application.ingestion import (  # noqa: E402
    IngestionRequest,
    IngestionResult,
)
from quant_research_platform.application.inspection import (  # noqa: E402
    InspectionService,
)
from quant_research_platform.application.jobs import SynchronousJobManager  # noqa: E402
from quant_research_platform.application.services import (  # noqa: E402
    Page,
    ResearchApplication,
    RunQuery,
)
from quant_research_platform.application.snapshots import (  # noqa: E402
    LocalPublishedSnapshotStore,
    SnapshotManager,
    SnapshotQuery,
)
from quant_research_platform.config.models import SecretConfig  # noqa: E402
from quant_research_platform.config.serializer import Redactor  # noqa: E402
from quant_research_platform.domain.canonical import canonical_json  # noqa: E402
from quant_research_platform.domain.errors import (  # noqa: E402
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.domain.evaluation import (  # noqa: E402
    MetricScope,
    calculate_evaluation_metrics,
)
from quant_research_platform.domain.execution import (  # noqa: E402
    JobOperation,
    JobStage,
    JobState,
)
from quant_research_platform.domain.manifests import (  # noqa: E402
    ContentAddressedObjectRef,
    ObjectKind,
)
from quant_research_platform.domain.market import (  # noqa: E402
    DateRange,
    ProviderRequest,
)
from quant_research_platform.infrastructure.duckdb_metadata import (  # noqa: E402
    DuckDBMetadataStore,
)
from quant_research_platform.infrastructure.filesystem_store import (  # noqa: E402
    ArtifactReference,
    FilesystemStore,
)
from quant_research_platform.infrastructure.logging import (  # noqa: E402
    StructuredJsonlLogger,
)
from quant_research_platform.infrastructure.mlflow_tracker import (  # noqa: E402
    LocalMlflowTracker,
)
from tests.integration.test_snapshot_ingestion_faults import (  # noqa: E402
    SESSIONS,
    FixedJobClock,
    FixtureCalendar,
    OfflineYFinanceFixture,
    _config,
    _publication_fixture,
)

SECRET = "https://user:password@proxy.invalid"
SNAPSHOT_B = "snap_" + "b" * 64
NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)
RUN_ONE = UUID("00000000-0000-0000-0000-000000000801")
RUN_TWO = UUID("00000000-0000-0000-0000-000000000802")


class FixedConfigurationManager:
    """Return a validated config while retaining the real facade boundary."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.calls: list[tuple[object, object]] = []

    def resolve(self, yaml_document: object, environment: object) -> Ok[object]:
        self.calls.append((yaml_document, environment))
        return Ok(self.config)


class FacadeIngestion:
    """Use a local fake provider behind a real persisted synchronous job."""

    def __init__(
        self,
        jobs: SynchronousJobManager,
        provider: OfflineYFinanceFixture,
        snapshot_id: str,
        modes: tuple[str, ...],
    ) -> None:
        self.jobs = jobs
        self.provider = provider
        self.snapshot_id = snapshot_id
        self.modes = modes
        self.calls = 0

    def ingest(
        self,
        request: IngestionRequest,
        config: object,
        *,
        progress: object = None,
    ) -> object:
        del config
        mode = self.modes[min(self.calls, len(self.modes) - 1)]
        self.calls += 1
        requested_range = request.requested_range or DateRange(
            SESSIONS[0], SESSIONS[-1]
        )

        def work(job: object) -> None:
            self.provider.fetch_daily(
                ProviderRequest(("AAPL",), requested_range.start, requested_range.end)
            )
            job.report(  # type: ignore[attr-defined]
                stage=JobStage.FETCHING,
                completed_units=1,
                total_units=1,
                warnings=(f"provider note contains {SECRET}",),
            )
            if mode == "failed":
                raise RuntimeError(f"provider failed with {SECRET}")

        outcome = self.jobs.execute(
            JobOperation.INGESTION,
            work,
            total_units=1,
            partially_succeeded=mode == "partial",
            progress_callback=progress if callable(progress) else None,
        )
        if isinstance(outcome, Err):
            return outcome
        failed_symbols = ("MSFT",) if mode == "partial" else ()
        return Ok(
            IngestionResult(
                status=outcome.value.state,
                snapshot_id=self.snapshot_id,
                requested_range=requested_range,
                failed_symbols=failed_symbols,
                job_id=outcome.value.job_id,
                correlation_id=outcome.value.correlation_id,
            )
        )
class ExplodingBacktest:
    def run(
        self, request: object, config: object, *, progress: object = None
    ) -> object:
        del request, config, progress
        raise RuntimeError(f"backtest provider used {SECRET}")


class CapturingLogger:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def write(self, **entry: object) -> None:
        self.entries.append(entry)


class Scanner:
    """Bounded table scanner double; bytes remain owned by the real CAS store."""

    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def scan(
        self,
        refs: object,
        columns: object,
        predicate: object = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        order_by: object = (),
    ) -> tuple[dict[str, object], ...]:
        self.calls.append(
            {
                "refs": refs,
                "columns": columns,
                "predicate": predicate,
                "offset": offset,
                "limit": limit,
                "order_by": order_by,
            }
        )
        end = None if limit is None else offset + limit
        return self.rows[offset:end]


class TrackingClient:
    """Local MLflow client seam used by the real tracker without network I/O."""

    def __init__(self) -> None:
        self.run_count = 0
        self.terminated: list[tuple[str, str]] = []

    def get_experiment_by_name(self, name: str) -> None:
        del name
        return None

    def create_experiment(self, name: str) -> str:
        assert name == "quant_research_platform"
        return "local-experiment"

    def create_run(self, experiment_id: str, **kwargs: object) -> SimpleNamespace:
        assert experiment_id == "local-experiment"
        assert kwargs["tags"]["qrp.state"] == "running"  # type: ignore[index]
        self.run_count += 1
        return SimpleNamespace(info=SimpleNamespace(run_id=f"mlflow-{self.run_count}"))

    def log_param(self, run_id: str, key: str, value: object) -> None:
        del run_id, key, value

    def log_metric(self, run_id: str, key: str, value: float, **kwargs: object) -> None:
        del run_id, key, value, kwargs

    def set_tag(self, run_id: str, key: str, value: object) -> None:
        del run_id, key, value

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        del run_id, text, artifact_file

    def set_terminated(self, run_id: str, **kwargs: object) -> None:
        self.terminated.append((run_id, str(kwargs["status"])))


class RunViews:
    """Enrich real DuckDB run projections with verified local run documents."""

    def __init__(self, metadata: DuckDBMetadataStore) -> None:
        self.metadata = metadata
        self.documents: dict[UUID, dict[str, object]] = {}

    def get_run(self, run_id: UUID) -> object:
        record = self.metadata.get_run(run_id)
        values = {
            name: getattr(record, name)
            for name in (
                "run_id",
                "mlflow_run_id",
                "snapshot_id",
                "state",
                "strategy_id",
                "evaluation_start",
                "evaluation_end",
                "universe",
                "configuration_checksum",
                "environment_checksum",
                "manifest_checksum",
                "manifest_uri",
                "created_at",
                "started_at",
                "ended_at",
                "error_json",
                "immutable",
            )
        }
        values.update(self.documents[run_id])
        return SimpleNamespace(**values)

    def get_artifact(self, checksum: str) -> object:
        return self.metadata.get_artifact(checksum)

    def set_artifact_availability(self, checksum: str, availability: str) -> object:
        return self.metadata.set_artifact_availability(checksum, availability)


class EagerVerifiedArtifactStore:
    """Make the lazy CAS stream verify before the facade returns a result."""

    def __init__(self, store: FilesystemStore) -> None:
        self.store = store

    def open_verified_artifact(self, reference: object) -> object:
        chunks = tuple(self.store.stream_artifact(reference))  # type: ignore[arg-type]
        return lambda: iter(chunks)


class VerifiedComparisonStore:
    """Verify comparison inputs through CAS while delegating output publication."""

    def __init__(
        self, store: FilesystemStore, references: dict[str, ArtifactReference]
    ) -> None:
        self.store = store
        self.references = references

    def open_verified_artifact(self, reference: object) -> bytes:
        checksum = str(reference.checksum)  # type: ignore[attr-defined]
        return b"".join(self.store.stream_artifact(self.references[checksum]))

    def publish_artifact(self, payload: bytes, *, metadata: object) -> object:
        assert sha256(payload).hexdigest() == metadata["checksum"]  # type: ignore[index]
        return SimpleNamespace(checksum=metadata["checksum"])  # type: ignore[index]


def _secret_config() -> object:
    config = _config(end=SESSIONS[-1])
    return config.model_copy(
        update={"secrets": SecretConfig(https_proxy=SecretStr(SECRET))}
    )


def _metrics() -> tuple[object, object]:
    values = (Decimal("100"), Decimal("101"), Decimal("102"))
    return (
        calculate_evaluation_metrics(MetricScope.STRATEGY, values),
        calculate_evaluation_metrics(MetricScope.BENCHMARK, values),
    )


def _manifest(
    snapshot_id: str,
    artifact_checksum: str,
    *,
    position_count: int,
    source_revision: str,
) -> tuple[dict[str, object], str]:
    document: dict[str, object] = {
        "content_identity": {
            "snapshot_id": snapshot_id,
            "calendar": {"name": "XNYS", "version": "offline-fixture-1"},
        },
        "configuration": {
            "strategy": {"position_count": position_count},
            "secrets": {"https_proxy": "[REDACTED]"},
        },
        "environment_fingerprint": {
            "python_version": "3.11",
            "source_revision": source_revision,
        },
        "limitation_disclosure": {"version": "limitation-disclosure/v1"},
        "artifacts": (
            {"checksum": artifact_checksum, "role": "equity", "scientific": True},
        ),
    }
    return document, sha256(canonical_json(document)).hexdigest()


def _fixture_application(
    tmp_path: Path,
) -> tuple[ResearchApplication, DuckDBMetadataStore, FilesystemStore, str]:
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    publication = _publication_fixture("facade")
    published = store.publish_snapshot(publication.candidate(), operation_id="facade")
    snapshot_manager = SnapshotManager(
        storage=LocalPublishedSnapshotStore(store.root), metadata=metadata
    )
    config = _secret_config()
    application = ResearchApplication(
        configuration_manager=FixedConfigurationManager(config),
        snapshot_manager=snapshot_manager,
    )
    return application, metadata, store, published.snapshot_id


def test_facade_handles_configuration_jobs_partial_failures_and_sanitized_errors(
    tmp_path: Path,
) -> None:
    calendar = FixtureCalendar()
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    publication = _publication_fixture("facade-jobs")
    published = store.publish_snapshot(
        publication.candidate(), operation_id="facade-jobs"
    )
    snapshot_manager = SnapshotManager(
        storage=LocalPublishedSnapshotStore(store.root), metadata=metadata
    )
    jobs = SynchronousJobManager(
        metadata,
        StructuredJsonlLogger(
            tmp_path / "diagnostics.jsonl",
            redactor=Redactor((SECRET,)),
            utc_now=lambda: NOW,
        ),
        redactor=Redactor((SECRET,)),
        clock=FixedJobClock(),
    )
    config = _secret_config()
    services = FacadeIngestion(
        jobs,
        OfflineYFinanceFixture(calendar),
        published.snapshot_id,
        ("success", "partial", "failed"),
    )
    logger = CapturingLogger()
    application = ResearchApplication(
        configuration_manager=FixedConfigurationManager(config),
        ingestion_service=services,
        snapshot_manager=snapshot_manager,
        backtest_service=ExplodingBacktest(),
        logger=logger,
    )

    resolution = application.resolve_configuration(None, environment={})
    assert isinstance(resolution, Ok)
    assert resolution.value.view.secrets.https_proxy.value == "present_redacted"
    handle = resolution.value.handle
    progress: list[object] = []

    first = application.ingest(IngestionRequest(), handle, progress=progress.append)
    partial = application.ingest(IngestionRequest(), handle, progress=progress.append)
    failed = application.ingest(IngestionRequest(), handle, progress=progress.append)

    assert isinstance(first, Ok)
    assert first.value.status is JobState.SUCCEEDED
    assert isinstance(partial, Ok)
    assert partial.value.status is JobState.PARTIALLY_SUCCEEDED
    assert partial.value.failed_symbols == ("MSFT",)
    assert isinstance(failed, Err)
    assert failed.errors[0].operation == "ingestion"
    assert failed.errors[0].corrective_action
    assert all(SECRET not in str(item) for item in (failed.errors + tuple(progress)))
    states = {getattr(item, "state", None) for item in progress}
    assert JobState.RUNNING in states
    assert JobState.SUCCEEDED in states or JobState.PARTIALLY_SUCCEEDED in states

    # A later failed operation does not hide either previously published result.
    listed = application.list_snapshots(SnapshotQuery(page=0, page_size=100))
    assert isinstance(listed, Page)
    assert listed.errors == ()
    assert {item.snapshot_id for item in listed.items} == {first.value.snapshot_id}
    inspected = application.inspect_snapshot(first.value.snapshot_id)
    assert isinstance(inspected, Ok)
    assert inspected.value.snapshot_id == first.value.snapshot_id

    unexpected = application.run_backtest(
        BacktestRequest(first.value.snapshot_id), handle
    )
    assert isinstance(unexpected, Err)
    assert unexpected.errors[0].category is ErrorCategory.INTERNAL_UNEXPECTED
    assert SECRET not in str(unexpected.errors[0])
    assert logger.entries
    assert SECRET not in repr(logger.entries)

    assert isinstance(application.invalidate_configuration(handle), Ok)
    stale = application.ingest(IngestionRequest(), handle)
    assert isinstance(stale, Err)
    assert stale.errors[0].category is ErrorCategory.CONFIGURATION_INVALID_VALUE
    assert services.calls == 3
    metadata.close()


def test_facade_inspects_verified_runs_pages_artifacts_and_comparison_provenance(
    tmp_path: Path,
) -> None:
    application, metadata, store, snapshot_id = _fixture_application(tmp_path)
    payload = b"immutable equity artifact bytes"
    artifact_metadata = {
        "artifact_kind": "equity",
        "checksum": sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "media_type": "application/octet-stream",
        "schema_version": "artifact_v1",
        "row_count": 6,
    }
    staging = store.create_staging("equity-artifact")
    staged = store.stage_bytes(
        staging,
        "equity/data.bin",
        payload,
        expected_checksum=artifact_metadata["checksum"],
    )
    artifact_reference = store.publish_artifact(staged, metadata=artifact_metadata)
    metadata.record_artifact(
        ContentAddressedObjectRef(
            object_kind=ObjectKind.ARTIFACT,
            checksum=artifact_reference.checksum,
            relative_uri=artifact_reference.relative_uri,
            schema_version="artifact_v1",
            row_count=6,
            byte_size=len(payload),
            media_type="application/octet-stream",
        ),
        artifact_kind="equity",
        created_at=NOW,
    )

    metrics = _metrics()
    tracker = LocalMlflowTracker(
        tracking_uri=tmp_path / "mlflow.db",
        metadata_store=metadata,
        client=TrackingClient(),
    )
    views = RunViews(metadata)
    for index, (run_id, run_snapshot, position_count, start) in enumerate(
        (
            (RUN_ONE, snapshot_id, 2, SESSIONS[0]),
            (RUN_TWO, SNAPSHOT_B, 3, SESSIONS[1]),
        )
    ):
        manifest, manifest_checksum = _manifest(
            run_snapshot,
            artifact_reference.checksum,
            position_count=position_count,
            source_revision=f"revision-{index}",
        )
        handle = tracker.allocate_run(
            run_id=run_id,
            snapshot_id=run_snapshot,
            strategy_id="monthly_momentum_v1",
            strategy_parameters={"position_count": position_count},
            evaluation_start=start,
            evaluation_end=SESSIONS[-1],
            universe=("AAPL", "MSFT"),
            configuration_checksum=("d" if index == 0 else "e") * 64,
            environment_checksum=("f" if index == 0 else "a") * 64,
            configuration=manifest["configuration"],
            environment_fingerprint=manifest["environment_fingerprint"],
            secret_values=(SECRET,),
            created_at=NOW + timedelta(seconds=index),
            started_at=NOW + timedelta(seconds=index),
        )
        tracker.finalize_success(
            handle,
            SimpleNamespace(
                evaluation=SimpleNamespace(
                    strategy_metrics=metrics[0], benchmark_metrics=metrics[1]
                ),
                manifest_checksum=manifest_checksum,
                manifest_uri=f"runs/{run_id}/manifest.json",
                artifacts=(
                    SimpleNamespace(
                        checksum=artifact_reference.checksum,
                        role="equity",
                        scientific=True,
                    ),
                ),
            ),
        )
        views.documents[run_id] = {
            "manifest": manifest,
            "configuration": manifest["configuration"],
            "environment_fingerprint": manifest["environment_fingerprint"],
            "validation_report": {"accepted": 6},
            "logs": (f"run {index} started",),
            "artifacts": (
                SimpleNamespace(
                    checksum=artifact_reference.checksum,
                    role="equity",
                    scientific=True,
                ),
            ),
            "evaluation": SimpleNamespace(
                strategy_metrics=metrics[0], benchmark_metrics=metrics[1]
            ),
            "strategy_equity": tuple(
                (session, Decimal("100000") + Decimal(index + offset))
                for offset, session in enumerate(SESSIONS[index:])
            ),
            "benchmark_equity": tuple(
                (session, Decimal("100000") + Decimal(2 * index + offset))
                for offset, session in enumerate(SESSIONS[index:])
            ),
            "limitation_disclosure": LimitationDisclosure.current(),
        }

    scanner = Scanner(
        tuple(
            {"session": index, "equity": Decimal("100000") + index}
            for index in range(6)
        )
    )
    inspection = InspectionService(
        metadata=views,
        artifacts=EagerVerifiedArtifactStore(store),
        scanner=scanner,
        redactor=Redactor((SECRET,)),
        configured_page_size=2,
    )
    comparison_store = VerifiedComparisonStore(
        store, {artifact_reference.checksum: artifact_reference}
    )
    from quant_research_platform.application.comparisons import ComparisonService

    comparison = ComparisonService(
        metadata=views,
        artifacts=comparison_store,
        redactor=Redactor((SECRET,)),
    )
    application.inspection_service = inspection
    application.comparison_service = comparison
    application.run_search = metadata

    snapshots = application.list_snapshots(SnapshotQuery(page=0, page_size=1))
    assert isinstance(snapshots, Page)
    assert len(snapshots.items) == 1
    assert snapshots.has_next
    assert snapshots.items[0].snapshot_id == snapshot_id
    run_page = application.search_runs(RunQuery(page=0, page_size=1))
    assert isinstance(run_page, Page)
    assert len(run_page.items) == 1
    assert run_page.total == 2
    assert run_page.has_next
    assert run_page.items[0].run_id == RUN_TWO

    detail = application.inspect_run(RUN_ONE)
    assert isinstance(detail, Ok)
    assert detail.value.run_id == RUN_ONE
    assert detail.value.artifact_metadata[0].checksum == artifact_reference.checksum
    assert detail.value.configuration["secrets"]["https_proxy"] == "[REDACTED]"  # type: ignore[index]
    assert SECRET not in str(detail.value)
    assert detail.value.limitation_disclosure.version == "limitation-disclosure/v1"

    first_page = application.page_artifact(
        artifact_reference.checksum,
        page=0,
        page_size=100,
        columns=("session", "equity"),
        order_by=("session",),
    )
    second_page = application.page_artifact(
        artifact_reference.checksum,
        page=1,
        page_size=100,
        columns=("session", "equity"),
        order_by=("session",),
    )
    assert isinstance(first_page, Ok)
    assert isinstance(second_page, Ok)
    assert first_page.value.page_size == second_page.value.page_size == 2
    assert len(first_page.value.rows) == len(second_page.value.rows) == 2
    assert first_page.value.rows[0]["session"] == 0
    assert second_page.value.rows[0]["session"] == 2
    assert all(call["limit"] == 2 for call in scanner.calls)

    opened = application.open_artifact(artifact_reference.checksum)
    assert isinstance(opened, Ok)
    assert b"".join(opened.value.stream()) == payload

    compared = application.compare_runs((RUN_ONE, RUN_TWO))
    assert isinstance(compared, Ok)
    assert compared.value.aligned_sessions == SESSIONS[1:]
    assert compared.value.snapshot_differences
    assert compared.value.configuration_differences
    assert compared.value.environment_differences
    assert compared.value.limitation_disclosure.version == "limitation-disclosure/v1"
    assert SECRET not in str(compared.value)
    assert compared.value.artifact.checksum == sha256(
        compared.value.artifact.payload
    ).hexdigest()

    too_few = application.compare_runs((RUN_ONE,))
    too_many = application.compare_runs(tuple(UUID(int=1000 + i) for i in range(11)))
    assert isinstance(too_few, Err)
    assert "minimum" in too_few.errors[0].message
    assert isinstance(too_many, Err)
    assert "maximum" in too_many.errors[0].message

    # Tampering is detected at the facade boundary and does not erase run discovery.
    (store.root / artifact_reference.relative_uri).write_bytes(b"tampered")
    corrupt = application.open_artifact(artifact_reference.checksum)
    assert isinstance(corrupt, Err)
    assert corrupt.errors[0].category is ErrorCategory.INTEGRITY_CHECKSUM
    assert (
        metadata.get_artifact(artifact_reference.checksum).availability.value
        == "invalid"
    )
    still_discoverable = application.search_runs(RunQuery(page=0, page_size=100))
    assert [item.run_id for item in still_discoverable.items] == [RUN_TWO, RUN_ONE]
    comparison_after_corruption = application.compare_runs((RUN_ONE, RUN_TWO))
    assert isinstance(comparison_after_corruption, Err)
    assert (
        comparison_after_corruption.errors[0].category
        is ErrorCategory.INTEGRITY_CHECKSUM
    )
    metadata.close()
