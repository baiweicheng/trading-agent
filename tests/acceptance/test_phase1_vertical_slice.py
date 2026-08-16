"""Final offline Phase 1 vertical-slice acceptance coverage.

This test intentionally composes the real application services and local
infrastructure through the public facade.  Provider, calendar, MLflow client,
and Zipline writer seams are deterministic local fixtures; no network or
server is started.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("streamlit")

from quant_research_platform.application.backtests import (  # noqa: E402
    BacktestRequest,
    BacktestService,
)
from quant_research_platform.application.comparisons import (
    ComparisonService,  # noqa: E402
)
from quant_research_platform.application.evaluation import (
    EvaluationService,  # noqa: E402
)
from quant_research_platform.application.ingestion import (  # noqa: E402
    DataIngestionService,
    IngestionRequest,
)
from quant_research_platform.application.inspection import (
    InspectionService,  # noqa: E402
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
from quant_research_platform.config.loader import ConfigurationManager  # noqa: E402
from quant_research_platform.config.serializer import (  # noqa: E402
    Redactor,
    non_secret_config,
)
from quant_research_platform.domain.canonical import (  # noqa: E402
    canonical_json,
    sha256_bytes,
)
from quant_research_platform.domain.errors import (  # noqa: E402
    Err,
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.domain.execution import (  # noqa: E402
    INITIAL_PORTFOLIO_EQUITY,
    OrderStatus,
)
from quant_research_platform.domain.manifests import (  # noqa: E402
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
)
from quant_research_platform.domain.market import DateRange  # noqa: E402
from quant_research_platform.infrastructure.duckdb_metadata import (  # noqa: E402
    DuckDBMetadataStore,
    ImmutableMetadataError,
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
from quant_research_platform.infrastructure.zipline_bundle import (  # noqa: E402
    ZiplineBundleAdapter,
)
from tests.integration import test_phase1_pipeline as pipeline  # noqa: E402
from tests.integration.test_streamlit_apptest import (  # noqa: E402
    FakeWorkflowApplication,
    _dataframe_rows,
    _run_app,
    _ui_text,
    _values,
    _widget,
)

SECRET = pipeline.SECRET


class AcceptanceParquetWriter(pipeline.SnapshotParquetWriter):
    """Use the real Parquet writer for validation-only collections too."""

    def _write_auxiliary(
        self,
        rows: Sequence[object],
        *,
        schema_name: str,
        object_kind: ObjectKind,
        converter: object,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        import pyarrow.parquet as pq

        from quant_research_platform.infrastructure.parquet_store import (
            PARQUET_WRITE_OPTIONS,
        )

        chunk_size = write_chunk_size or self.store.write_chunk_size
        output_root = Path(staging) if staging is not None else self.store.root
        outputs: list[object] = []
        for offset in range(0, len(rows), chunk_size):
            table = converter(rows[offset : offset + chunk_size])
            output = io.BytesIO()
            pq.write_table(
                table,
                output,
                row_group_size=chunk_size,
                **dict(PARQUET_WRITE_OPTIONS),
            )
            payload = output.getvalue()
            checksum = sha256_bytes(payload)
            relative_uri = f"objects/{schema_name}/sha256={checksum}.parquet"
            path = (
                output_root / "auxiliary" / schema_name / f"sha256={checksum}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            outputs.append(
                pipeline.StagedAuxiliaryObject(
                    object_ref=ContentAddressedObjectRef(
                        object_kind=object_kind,
                        checksum=checksum,
                        relative_uri=relative_uri,
                        schema_version=schema_name,
                        row_count=table.num_rows,
                        byte_size=len(payload),
                        media_type="application/vnd.apache.parquet",
                    ),
                    path=path,
                )
            )
        return tuple(outputs)

    def write_quarantine(
        self,
        rows: Sequence[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        from quant_research_platform.infrastructure.schemas import quarantines_to_table

        return self._write_auxiliary(
            tuple(rows),
            schema_name="quarantine_v1",
            object_kind=ObjectKind.QUARANTINE,
            converter=quarantines_to_table,
            write_chunk_size=write_chunk_size,
            staging=staging,
        )

    def write_gaps(
        self,
        rows: Sequence[object],
        *,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[object, ...]:
        from quant_research_platform.infrastructure.schemas import gaps_to_table

        return self._write_auxiliary(
            tuple(rows),
            schema_name="gap_v1",
            object_kind=ObjectKind.GAP,
            converter=gaps_to_table,
            write_chunk_size=write_chunk_size,
            staging=staging,
        )


class IncrementingClock:
    """Return distinct operational times without changing scientific inputs."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def utc_now(self) -> datetime:
        current = self.value
        self.value = current.replace(microsecond=current.microsecond + 1)
        return current


class AcceptanceCalendar(pipeline.LongFixtureCalendar):
    """Fixture calendar whose identity exactly matches the requested slice."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = CalendarIdentity(
            self.name,
            self.version,
            self.schedule_checksum(pipeline.START, pipeline.END),
        )


class AcceptanceProjection(pipeline.PublishedProjection):
    """Resolve normalized references from both public snapshot representations.

    Bundle materialization receives a manifest, while decision delivery and the
    local engine receive a verified handle.  The production reader port allows
    either shape; this acceptance projection keeps the fixture at that public
    boundary instead of teaching the test engine about storage internals.
    """

    def _rows(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        *,
        symbols: Sequence[str] = (),
        session_start: object = None,
        session_end: object = None,
    ) -> tuple[dict[str, object], ...]:
        projected_columns = tuple(dict.fromkeys(("symbol", *columns)))
        rows = super()._rows(
            references,
            projected_columns,
            symbols=symbols,
            session_start=session_start,
            session_end=session_end,
        )
        return tuple(
            {
                **row,
                **{
                    field: Decimal(str(row[field]))
                    for field in ("raw_open", "raw_close")
                    if field in row and row[field] is not None
                },
            }
            for row in rows
        )

    def scan(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        predicate: object | None = None,
        *,
        symbols: Sequence[str] | None = None,
        session_start: object = None,
        session_end: object = None,
    ) -> tuple[dict[str, object], ...]:
        return super().scan(
            references,
            columns,
            predicate=predicate,
            symbols=symbols,
            session_start=session_start,
            session_end=session_end,
        )

    def _normalized_refs(
        self, snapshot: object | None = None
    ) -> tuple[ContentAddressedObjectRef, ...]:
        references = getattr(snapshot, "object_references", None)
        if references is not None:
            return tuple(
                reference
                for reference in references
                if reference.object_kind is ObjectKind.NORMALIZED
            )
        return super()._normalized_refs(snapshot)


def _acceptance_manifest_plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date, UUID, Decimal)):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _acceptance_manifest_plain(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_acceptance_manifest_plain(item) for item in value]
    return str(value)


def _acceptance_manifest_for_run(
    result: object,
    config: object,
    links: Sequence[SimpleNamespace],
) -> tuple[dict[str, object], bytes, str]:
    evaluation = result.evaluation
    non_secret = non_secret_config(config).model_dump(mode="json")
    metric_document = json.loads(evaluation.artifacts["metrics"].payload)
    metric_rows = (
        metric_document["rows"]
        if isinstance(metric_document, dict)
        and isinstance(metric_document.get("rows"), list)
        else metric_document
    )
    manifest: dict[str, object] = {
        "content_identity": {
            "snapshot_id": result.snapshot_id,
            "strategy_id": config.strategy.identifier,
            "evaluation_range": result.evaluation_range.to_content_dict(),
            "configuration_checksum": sha256_bytes(canonical_json(non_secret)),
            "artifact_checksums": {item.role: item.checksum for item in links},
        },
        "snapshot_id": result.snapshot_id,
        "strategy_id": config.strategy.identifier,
        "evaluation_start": result.evaluation_range.start,
        "evaluation_end": result.evaluation_range.end,
        "configuration": non_secret,
        "environment_fingerprint": {
            "python_version": "3.11",
            "source_revision": "offline-fixture",
            "deterministic_seed": config.runtime.deterministic_seed,
        },
        "strategy_equity": evaluation.strategy_equity,
        "benchmark_equity": evaluation.benchmark_equity,
        "metric_rows": metric_rows,
        "artifacts": [
            {
                "checksum": item.checksum,
                "role": item.role,
                "scientific": item.scientific,
            }
            for item in links
        ],
        "limitation_disclosure": {
            "version": evaluation.limitation_disclosure.version,
            "lines": list(evaluation.limitation_disclosure.lines()),
        },
    }
    payload = canonical_json(_acceptance_manifest_plain(manifest))
    return manifest, payload, sha256_bytes(payload)


def _acceptance_record_evaluation_artifacts(
    metadata: DuckDBMetadataStore,
    evaluation: object,
    created_at: datetime,
) -> tuple[SimpleNamespace, ...]:
    links: list[SimpleNamespace] = []
    for artifact in evaluation.artifacts:
        reference = artifact.reference
        assert reference is not None
        row_count = artifact.row_count if isinstance(artifact.row_count, int) else 0
        metadata.record_artifact(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.ARTIFACT,
                checksum=artifact.checksum,
                relative_uri=reference.relative_uri,
                schema_version=artifact.schema_version,
                row_count=row_count,
                byte_size=artifact.byte_size,
                media_type=artifact.media_type,
            ),
            artifact_kind=artifact.role,
            created_at=created_at,
        )
        links.append(
            SimpleNamespace(
                checksum=artifact.checksum,
                role=artifact.role,
                scientific=True,
                uri=reference.relative_uri,
                byte_size=artifact.byte_size,
            )
        )
    return tuple(sorted(links, key=lambda item: (item.role, item.checksum)))


class AcceptanceTrackingAdapter(pipeline.TrackingAdapter):
    """Expose the tracker finalizer under the facade's ``run_id`` name."""

    def finalize_success(self, run_id: object, result: object) -> object:
        links = _acceptance_record_evaluation_artifacts(
            self.metadata,
            result.evaluation,
            datetime(2024, 2, 5, tzinfo=UTC),
        )
        manifest, payload, checksum = _acceptance_manifest_for_run(
            result,
            self.config,
            links,
        )
        staging = self.store.create_staging(f"manifest-{result.run_id}")
        staged = self.store.stage_bytes(
            staging,
            f"runs/{result.run_id}/manifest.json",
            payload,
            expected_checksum=checksum,
        )
        reference = self.store.publish_artifact(
            staged,
            metadata={
                "artifact_kind": "run_manifest",
                "checksum": checksum,
                "byte_size": len(payload),
                "media_type": "application/json",
                "schema_version": "run_manifest_v1",
                "row_count": None,
            },
        )
        self.metadata.record_artifact(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.ARTIFACT,
                checksum=checksum,
                relative_uri=reference.relative_uri,
                schema_version="run_manifest_v1",
                row_count=0,
                byte_size=len(payload),
                media_type="application/json",
            ),
            artifact_kind="run_manifest",
            created_at=datetime(2024, 2, 5, tzinfo=UTC),
        )
        identifier = (
            result.run_id
            if isinstance(result.run_id, UUID)
            else UUID(str(result.run_id))
        )
        self.run_views.documents[identifier] = {
            "manifest": manifest,
            "configuration": manifest["configuration"],
            "environment_fingerprint": manifest["environment_fingerprint"],
            "validation_report": {"snapshot_id": result.snapshot_id},
            "logs": ("offline local run",),
            "artifacts": links,
            "evaluation": result.evaluation,
            "evaluation_result": result.evaluation.evaluation_result,
            "strategy_equity": result.evaluation.strategy_equity,
            "benchmark_equity": result.evaluation.benchmark_equity,
            "limitation_disclosure": result.evaluation.limitation_disclosure,
        }
        ended_at = datetime(2024, 2, 5, 0, 0, len(self.run_views.documents), tzinfo=UTC)
        tracked = SimpleNamespace(
            manifest=manifest,
            manifest_checksum=checksum,
            manifest_uri=reference.relative_uri,
            evaluation=result.evaluation,
            artifacts=links,
            ended_at=ended_at,
        )
        return self.tracker.finalize_success(run_id, tracked)


class AcceptanceArtifactStore:
    """Reuse the real CAS store with distinct staging IDs per publication."""

    def __init__(self, store: FilesystemStore) -> None:
        self.store = store
        self.attempt = 0

    def create_staging(self, operation_id: str | None = None) -> object:
        self.attempt += 1
        suffix = f"-{self.attempt}"
        return self.store.create_staging(f"{operation_id or 'artifact'}{suffix}")

    def stage_bytes(
        self,
        staging: object,
        relative_path: str,
        data: bytes,
        *,
        expected_checksum: str | None = None,
    ) -> object:
        return self.store.stage_bytes(
            staging,
            relative_path,
            data,
            expected_checksum=expected_checksum,
        )

    def publish_artifact(
        self,
        staged: object,
        *,
        metadata: object,
    ) -> object:
        return self.store.publish_artifact(staged, metadata=metadata)


class AcceptanceArtifactScanner:
    """Project checksummed JSON artifacts into bounded ordinary-table pages."""

    def __init__(self, store: FilesystemStore) -> None:
        self.store = store

    def scan(
        self,
        refs: object,
        columns: Sequence[str],
        predicate: object | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        order_by: Sequence[str] = (),
        checksum: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        del predicate
        references = (
            tuple(refs) if not isinstance(refs, (str, bytes, bytearray)) else ()
        )
        reference = references[0] if references else None
        checksum_value = checksum or getattr(reference, "checksum", None)
        if not isinstance(checksum_value, str):
            raise ValueError("artifact checksum is required")
        publication = json.loads(
            (self.store.publications_root / f"{checksum_value}.json").read_bytes()
        )
        metadata = publication["metadata"]
        metadata_checksum = sha256_bytes(canonical_json(metadata))
        relative_uri = metadata.get(
            "relative_uri", getattr(reference, "relative_uri", None)
        )
        byte_size = metadata.get("byte_size", getattr(reference, "byte_size", None))
        if relative_uri is None or byte_size is None:
            raise ValueError("artifact publication reference is incomplete")
        artifact_reference = ArtifactReference(
            checksum=checksum_value,
            byte_size=int(byte_size),
            relative_uri=str(relative_uri),
            metadata_checksum=metadata_checksum,
        )
        document = json.loads(b"".join(self.store.stream_artifact(artifact_reference)))
        if isinstance(document, dict) and isinstance(document.get("rows"), list):
            source_rows = document["rows"]
        elif isinstance(document, list):
            source_rows = document
        else:
            source_rows = [document]
        rows = [
            {
                column: row[column]
                for column in columns
                if isinstance(row, dict) and column in row
            }
            for row in source_rows
        ]
        if order_by:
            rows.sort(key=lambda row: tuple(row.get(column) for column in order_by))
        selected = rows[offset:] if limit is None else rows[offset : offset + limit]
        return tuple(selected)


class AcceptanceApplication:
    """The concrete public application graph used by this acceptance test."""

    def __init__(self, root: Path) -> None:
        self.calendar = AcceptanceCalendar()
        self.metadata = DuckDBMetadataStore(root / "metadata.duckdb")
        self.store = FilesystemStore(root / "store", metadata=self.metadata)
        self.provider = pipeline.QualityOfflineProvider(self.calendar)
        self.jobs = SynchronousJobManager(
            self.metadata,
            StructuredJsonlLogger(
                root / "diagnostics.jsonl",
                redactor=Redactor((SECRET,)),
                utc_now=lambda: datetime(2024, 2, 5, tzinfo=UTC),
            ),
            redactor=Redactor((SECRET,)),
            clock=pipeline.FixedJobClock(),
        )
        self.ingestion = DataIngestionService(
            self.provider,
            self.calendar,
            parquet_store=AcceptanceParquetWriter(root / "parquet"),
            snapshot_publisher=self.store,
            metadata=self.metadata,
            job_manager=self.jobs,
            clock=SimpleNamespace(utc_now=lambda: datetime(2024, 2, 5, tzinfo=UTC)),
            sleep=lambda _seconds: None,
            redactor=Redactor((SECRET,)),
        )
        self.snapshot_manager = SnapshotManager(
            storage=LocalPublishedSnapshotStore(self.store.root),
            metadata=self.metadata,
        )
        self.configuration_manager = ConfigurationManager(
            project_anchor=Path(__file__).resolve()
        )
        self.application = ResearchApplication(
            configuration_manager=self.configuration_manager,
            ingestion_service=self.ingestion,
            snapshot_manager=self.snapshot_manager,
        )


def _make_application(root: Path) -> tuple[AcceptanceApplication, object, object]:
    fixture = AcceptanceApplication(root)
    config_path = root / "effective.yaml"
    config_path.write_bytes(pipeline._config_yaml(root))
    resolved = fixture.application.resolve_configuration(
        config_path,
        environment={"QRP_SECRETS__HTTPS_PROXY": SECRET},
    )
    assert isinstance(resolved, Ok), resolved
    assert resolved.value.view.secrets.https_proxy.value == "present_redacted"
    assert SECRET not in repr(resolved.value.view)
    return fixture, resolved.value.handle, resolved.value.view


def _wire_backtest(
    fixture: AcceptanceApplication, handle: object, view: object
) -> tuple[object, object, object]:
    projection = pipeline.PublishedProjection(fixture.store, [])
    tracker_client = pipeline.TrackingClient()
    tracker = LocalMlflowTracker(
        tracking_uri=fixture.store.root.parent / "mlflow.db",
        metadata_store=fixture.metadata,
        artifact_store=fixture.store,
        client=tracker_client,
    )
    run_views = pipeline.RunViews(fixture.metadata)
    reader = AcceptanceProjection(fixture.store, [])
    engine = pipeline.LocalMomentumEngine(
        fixture.snapshot_manager,
        reader,
        fixture.calendar,
    )
    tracking = AcceptanceTrackingAdapter(
        tracker,
        view,
        fixture.metadata,
        fixture.store,
        run_views,
    )
    backtest = BacktestService(
        tracker=tracking,
        snapshot_manager=fixture.snapshot_manager,
        bundle_adapter=ZiplineBundleAdapter(
            snapshot_manager=fixture.snapshot_manager,
            data_source=projection,
            calendar=fixture.calendar,
            zipline_root=fixture.store.root / "derived-runs",
            writer=pipeline._DeterministicWriter(),
        ),
        engine=engine,
        evaluator=EvaluationService(
            snapshot_manager=fixture.snapshot_manager,
            parquet_store=reader,
            artifact_store=AcceptanceArtifactStore(fixture.store),
        ),
        clock=IncrementingClock(datetime(2024, 2, 5, tzinfo=UTC)),
    )
    fixture.application.backtest_service = backtest
    fixture.application.run_search = fixture.metadata
    fixture.application.inspection_service = InspectionService(
        metadata=run_views,
        artifacts=pipeline.ArtifactVerifier(fixture.store, fixture.metadata),
        scanner=AcceptanceArtifactScanner(fixture.store),
        redactor=Redactor((SECRET,)),
        configured_page_size=100,
    )
    fixture.application.comparison_service = ComparisonService(
        metadata=run_views,
        artifacts=pipeline.ArtifactVerifier(fixture.store, fixture.metadata),
        redactor=Redactor((SECRET,)),
    )
    return projection, reader, tracker_client


def _assert_accounting(result: object) -> None:
    core = result.core_output
    assert core.initial_equity == INITIAL_PORTFOLIO_EQUITY
    assert core.orders and core.fills and core.strategy_decisions
    assert all(order.status is OrderStatus.FILLED for order in core.orders)
    for state in core.portfolio_states:
        assert state.cash_balance >= 0
        assert 0 <= state.leverage <= 1
        assert all(position.quantity >= 0 for position in state.positions)
        marked = sum((position.market_value for position in state.positions), 0)
        assert abs(state.portfolio_equity - state.cash_balance - marked) <= 0.01
    order_sessions = {order.order_id: order.execution_session for order in core.orders}
    assert all(fill.session == order_sessions[fill.order_id] for fill in core.fills)
    assert all(order.execution_session > order.signal_session for order in core.orders)


@pytest.mark.integration
def test_phase1_vertical_slice_acceptance(tmp_path: Path) -> None:
    """Accept Phase 1 only after the complete local workflow passes its gates."""

    fixture, handle, view = _make_application(tmp_path)
    try:
        progress: list[object] = []
        partial = fixture.application.ingest(
            IngestionRequest(), handle, progress=progress.append
        )
        assert isinstance(partial, Ok), partial
        partial_result = partial.value
        assert partial_result.job_state.value == "partially_succeeded"
        assert partial_result.failed_symbols == ("MSFT",)
        assert partial_result.quarantined_rows and partial_result.gaps
        assert partial_result.limitation_disclosure.data_failures
        assert partial_result.manifest is not None
        partial_roles = {
            item.object_kind
            for item in partial_result.manifest.content_identity.objects
        }
        assert {
            ObjectKind.RAW,
            ObjectKind.NORMALIZED,
            ObjectKind.QUARANTINE,
            ObjectKind.GAP,
        } <= partial_roles
        assert all(
            fixture.store.read_object(item.relative_uri)[:4] == b"PAR1"
            for item in partial_result.manifest.content_identity.objects
        )
        assert (
            fixture.metadata.get_job(partial_result.job_id).state.value
            == "partially_succeeded"
        )

        fixture.provider.failed_symbols.clear()
        fixture.provider.include_quality_issue = False
        clean = fixture.application.ingest(
            IngestionRequest(), handle, progress=progress.append
        )
        assert isinstance(clean, Ok), clean
        clean_result = clean.value
        assert clean_result.job_state.value == "succeeded"
        assert clean_result.failed_symbols == ()
        assert clean_result.quarantined_rows == ()
        assert clean_result.gaps == ()
        assert clean_result.manifest is not None
        assert {
            item.object_kind for item in clean_result.manifest.content_identity.objects
        } == {ObjectKind.RAW, ObjectKind.NORMALIZED}
        assert clean_result.snapshot_id != partial_result.snapshot_id
        assert isinstance(
            fixture.snapshot_manager.open_verified(clean_result.snapshot_id), Ok
        )
        assert isinstance(
            fixture.snapshot_manager.open_verified(partial_result.snapshot_id), Ok
        )
        assert isinstance(
            fixture.snapshot_manager.replace_manifest(
                clean_result.snapshot_id, b"mutate"
            ),
            Err,
        )

        snapshots = fixture.application.list_snapshots(
            SnapshotQuery(page=0, page_size=100)
        )
        assert isinstance(snapshots, Page)
        assert {item.snapshot_id for item in snapshots.items} >= {
            partial_result.snapshot_id,
            clean_result.snapshot_id,
        }
        inspected_snapshot = fixture.application.inspect_snapshot(
            clean_result.snapshot_id
        )
        assert isinstance(inspected_snapshot, Ok)
        assert inspected_snapshot.value.comparison_ready
        assert (
            inspected_snapshot.value.limitation_disclosure.version
            == LimitationDisclosure.current().version
        )

        bundle_projection = pipeline.PublishedProjection(fixture.store, [])
        bundle = ZiplineBundleAdapter(
            snapshot_manager=fixture.snapshot_manager,
            data_source=bundle_projection,
            calendar=fixture.calendar,
            zipline_root=tmp_path / "derived",
            writer=pipeline._DeterministicWriter(),
        ).materialize(clean_result.snapshot_id)
        assert isinstance(bundle, Ok), bundle
        assert bundle.value.snapshot_id == clean_result.snapshot_id
        assert bundle.value.bundle_name != "latest"
        bundle_manifest = json.loads(
            (bundle.value.cache_path / "bundle_manifest.json").read_bytes()
        )
        assert bundle_manifest["bundle_checksum"] == bundle.value.bundle_checksum

        _bundle_projection, reader, tracker_client = _wire_backtest(
            fixture, handle, view
        )
        request = BacktestRequest(
            clean_result.snapshot_id, DateRange(pipeline.START, pipeline.END)
        )
        first = fixture.application.run_backtest(request, handle)
        second = fixture.application.run_backtest(request, handle)
        assert isinstance(first, Ok), first
        assert isinstance(second, Ok), second
        _assert_accounting(first.value)
        _assert_accounting(second.value)
        assert first.value.run_id != second.value.run_id
        assert (
            first.value.core_output.to_scientific_dict()
            == second.value.core_output.to_scientific_dict()
        )
        assert (
            first.value.evaluation.artifact_checksums
            == second.value.evaluation.artifact_checksums
        )
        assert (
            first.value.evaluation.strategy_equity
            == second.value.evaluation.strategy_equity
        )
        assert first.value.evaluation.spy_gaps == ()
        assert (
            first.value.evaluation.limitation_disclosure.version
            == LimitationDisclosure.current().version
        )
        assert all(
            sha256_bytes(artifact.payload) == artifact.checksum
            for artifact in first.value.evaluation.artifacts
        )
        assert any(call[0] == "terminated" for call in tracker_client.calls)
        assert reader.calls and any(call["kind"] == "history" for call in reader.calls)
        assert all(
            call["session_end"] <= pipeline.END
            for call in reader.calls
            if call["kind"] == "history"
        )
        assert all(
            decision.endpoint_252_session is None
            or decision.endpoint_252_session <= decision.signal_session
            for decision in first.value.core_output.strategy_decisions
        )

        first_id = UUID(str(first.value.run_id))
        first_record = fixture.metadata.get_run(first_id)
        second_record = fixture.metadata.get_run(UUID(str(second.value.run_id)))
        assert first_record.state.value == "succeeded"
        assert first_record.immutable is True
        assert first_record.created_at != second_record.created_at
        assert first_record.started_at != second_record.started_at
        assert first_record.ended_at != second_record.ended_at
        assert first_record.manifest_checksum == second_record.manifest_checksum
        with pytest.raises(ImmutableMetadataError):
            fixture.metadata.set_mlflow_run_id(first_id, "mutate-terminal-run")

        discovered = fixture.application.search_runs(RunQuery(page=0, page_size=100))
        assert discovered.total == 2
        assert all(item.state.value == "succeeded" for item in discovered.items)
        inspected_run = fixture.application.inspect_run(first.value.run_id)
        assert isinstance(inspected_run, Ok)
        assert SECRET not in repr(inspected_run.value)
        artifact_checksum = first.value.evaluation.artifacts["strategy_equity"].checksum
        opened = fixture.application.open_artifact(artifact_checksum)
        assert isinstance(opened, Ok), opened
        assert (
            b"".join(opened.value.stream())
            == first.value.evaluation.artifacts["strategy_equity"].payload
        )
        paged = fixture.application.page_artifact(
            artifact_checksum, page=0, page_size=100
        )
        assert isinstance(paged, Ok), paged
        assert len(paged.value.rows) <= 100
        compared = fixture.application.compare_runs(
            (first.value.run_id, second.value.run_id)
        )
        assert isinstance(compared, Ok), compared
        assert compared.value.aligned_sessions
        assert (
            compared.value.limitation_disclosure.version
            == LimitationDisclosure.current().version
        )
        assert compared.value.artifact_checksum == sha256_bytes(
            compared.value.artifact.bytes
        )

        fixture.provider.failed_symbols.add("MSFT")
        later_failure = fixture.application.ingest(
            IngestionRequest(), handle, progress=progress.append
        )
        assert isinstance(later_failure, Ok)
        assert later_failure.value.job_state.value == "partially_succeeded"
        assert isinstance(
            fixture.snapshot_manager.open_verified(clean_result.snapshot_id), Ok
        )
        assert isinstance(fixture.application.inspect_run(first.value.run_id), Ok)
        assert all(SECRET not in repr(item) for item in progress)
        assert SECRET not in (tmp_path / "diagnostics.jsonl").read_text()
        assert SECRET not in repr(tracker_client.calls)

        ui = _run_app(FakeWorkflowApplication(run_count=2), tmp_path / "ui")
        _widget(ui.button, "Validate configuration").click()
        ui.run()
        _widget(ui.button, "Ingest data").click()
        ui.run()
        assert "Limitations and assumptions" in _values(ui.subheader)
        _widget(ui.sidebar.radio, "Workflow").set_value("Snapshots")
        ui.run()
        snapshot_selector = _widget(ui.selectbox, "Snapshot to inspect")
        snapshot_selector.set_value(snapshot_selector.options[-1])
        ui.run()
        assert "Published snapshots are immutable" in _values(ui.info)
        _widget(ui.sidebar.radio, "Workflow").set_value("Backtest")
        ui.run()
        _widget(ui.button, "Verify selected snapshot").click()
        ui.run()
        _widget(ui.button, "Run backtest").click()
        ui.run()
        assert "SPY benchmark metrics" in _values(ui.subheader)
        _widget(ui.sidebar.radio, "Workflow").set_value("Runs")
        ui.run()
        _widget(ui.button, "Inspect selected run").click()
        ui.run()
        assert "Manifest" in _values(ui.subheader)
        _widget(ui.sidebar.radio, "Workflow").set_value("Compare")
        ui.run()
        assert "minimum is 2" in _values(ui.error)
        _widget(ui.multiselect, "Runs (selection order is preserved)").set_value(
            ["run-2", "run-1"]
        )
        ui.run()
        _widget(ui.button, "Compare selected runs").click()
        ui.run()
        assert "Verified comparison artifact" in _values(ui.subheader)
        assert all(row_count <= 100 for row_count in _dataframe_rows(ui))
        assert SECRET not in _ui_text(ui)
    finally:
        fixture.metadata.close()
