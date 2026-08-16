from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

from pydantic import SecretStr

from quant_research_platform.application.backtests import BacktestRequest
from quant_research_platform.application.ingestion import IngestionRequest
from quant_research_platform.application.inspection import RunSummary
from quant_research_platform.application.services import (
    ConfigurationHandle,
    Page,
    ResearchApplication,
    RunQuery,
)
from quant_research_platform.config.models import ResolvedConfig
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import Err, ErrorCategory, Ok

SECRET = "https://user:password@proxy.invalid"
SNAPSHOT_ID = "snap_" + "a" * 64
NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)


def _config() -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {},
            "data": {
                "requested_range": {
                    "start": "2020-01-01",
                    "end": "2020-12-31",
                }
            },
            "secrets": {"https_proxy": SecretStr(SECRET)},
        }
    )


class ConfigurationManager:
    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config
        self.calls: list[tuple[object, object]] = []

    def resolve(self, yaml_document: object, environment: object) -> Ok[ResolvedConfig]:
        self.calls.append((yaml_document, environment))
        return Ok(self.config)


class Ingestion:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def ingest(
        self, request: object, config: object, *, progress: object = None
    ) -> Ok[str]:
        self.calls.append((request, config, progress))
        return Ok("ingested")


class Backtest:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def run(
        self, request: object, config: object, *, progress: object = None
    ) -> Ok[str]:
        self.calls.append((request, config, progress))
        return Ok("backtested")


class Logger:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def write(self, **entry: object) -> None:
        self.entries.append(entry)


def test_configuration_resolution_returns_redacted_view_and_opaque_handle() -> None:
    config = _config()
    manager = ConfigurationManager(config)
    app = ResearchApplication(configuration_manager=manager)

    result = app.resolve_configuration(None, ui_yaml_values={"data": {"batch_size": 3}})

    assert isinstance(result, Ok)
    resolution = result.value
    assert resolution.view.data.batch_size == 5
    assert resolution.view.secrets.https_proxy.value == "present_redacted"
    assert SECRET not in repr(resolution.view)
    assert SECRET not in repr(resolution.handle)
    assert isinstance(resolution.handle, ConfigurationHandle)
    assert manager.calls[0][0] is not None
    assert "batch_size" in str(manager.calls[0][0])


def test_valid_handle_delegates_the_same_frozen_configuration_to_ingestion_and_backtest() -> (
    None
):
    config = _config()
    manager = ConfigurationManager(config)
    ingestion = Ingestion()
    backtest = Backtest()
    app = ResearchApplication(
        configuration_manager=manager,
        ingestion_service=ingestion,
        backtest_service=backtest,
    )
    resolved = app.resolve_configuration(None)
    assert isinstance(resolved, Ok)
    handle = resolved.value.handle
    progress = object()
    ingestion_request = IngestionRequest()
    backtest_request = BacktestRequest(SNAPSHOT_ID)

    ingestion_result = app.ingest(ingestion_request, handle, progress=progress)
    backtest_result = app.run_backtest(backtest_request, handle, progress=progress)

    assert ingestion_result == Ok("ingested")
    assert backtest_result == Ok("backtested")
    assert ingestion.calls == [(ingestion_request, config, progress)]
    assert backtest.calls == [(backtest_request, config, progress)]


def test_unknown_and_invalidated_handles_are_rejected_before_service_invocation() -> (
    None
):
    config = _config()
    ingestion = Ingestion()
    app = ResearchApplication(
        configuration_manager=ConfigurationManager(config),
        ingestion_service=ingestion,
    )
    resolved = app.resolve_configuration(None)
    assert isinstance(resolved, Ok)
    handle = resolved.value.handle

    unknown = app.ingest(IngestionRequest(), ConfigurationHandle(uuid4()))
    invalidated = app.invalidate_configuration(handle)
    stale = app.ingest(IngestionRequest(), handle)
    invalid_type = app.ingest(IngestionRequest(), object())  # type: ignore[arg-type]

    for result in (unknown, invalidated, stale, invalid_type):
        assert isinstance(result, (Ok, Err))
    assert isinstance(unknown, Err)
    assert isinstance(invalidated, Ok)
    assert isinstance(stale, Err)
    assert isinstance(invalid_type, Err)
    assert all(
        error.category is ErrorCategory.CONFIGURATION_INVALID_VALUE
        for error in (unknown.errors + stale.errors + invalid_type.errors)
    )
    assert ingestion.calls == []


def test_discovery_and_artifact_operations_return_typed_pages_and_delegate() -> None:
    summary = RunSummary(
        run_id=UUID(int=1),
        snapshot_id=SNAPSHOT_ID,
        state="succeeded",
        strategy_id="monthly_momentum_v1",
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 9),
        universe=("AAPL",),
        configuration_checksum="b" * 64,
        environment_checksum="c" * 64,
        manifest_checksum="d" * 64,
        created_at=NOW,
        ended_at=NOW,
    )

    class Discovery:
        def __init__(self) -> None:
            self.snapshot_query: object | None = None
            self.run_query: object | None = None

        def inspect_snapshot(self, snapshot_id: str) -> Ok[str]:
            self.snapshot_query = snapshot_id
            return Ok("snapshot-detail")

        def list_snapshots(self, query: object) -> object:
            self.snapshot_query = query
            return SimpleNamespace(
                items=("snapshot",), page=0, page_size=100, total=1, errors=()
            )

        def search_runs(self, query: object) -> object:
            self.run_query = query
            return SimpleNamespace(
                records=(summary,), page=0, page_size=100, total_count=1
            )

    class Operations:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        def inspect_run(self, run_id: object) -> Ok[str]:
            self.calls.append(("run", (run_id,), {}))
            return Ok("run-detail")

        def compare(self, run_ids: object) -> Ok[str]:
            self.calls.append(("compare", (run_ids,), {}))
            return Ok("comparison")

        def page_artifact(
            self,
            checksum: str,
            page: int,
            page_size: int | None,
            columns: object = None,
            *,
            order_by: object = None,
        ) -> Ok[str]:
            self.calls.append(
                (
                    "page",
                    (checksum, page, page_size),
                    {"columns": columns, "order_by": order_by},
                )
            )
            return Ok("table-page")

        def open_artifact(self, checksum: str) -> Ok[str]:
            self.calls.append(("artifact", (checksum,), {}))
            return Ok("artifact-stream")

    discovery = Discovery()
    operations = Operations()
    app = ResearchApplication(
        snapshot_manager=discovery,
        run_search=discovery,
        inspection_service=operations,
        comparison_service=operations,
    )

    snapshots = app.list_snapshots()
    runs = app.search_runs(RunQuery(page=0, page_size=10))
    snapshot = app.inspect_snapshot(SNAPSHOT_ID)
    run = app.inspect_run(UUID(int=1))
    comparison = app.compare_runs(("run-a", "run-b"))
    table = app.page_artifact("e" * 64, 1, 20, ("session",), order_by=("session",))
    artifact = app.open_artifact("e" * 64)

    assert isinstance(snapshots, Page)
    assert snapshots.items == ("snapshot",)
    assert isinstance(runs, Page)
    assert runs.items == (summary,)
    assert isinstance(snapshot, Ok) and snapshot.value == "snapshot-detail"
    assert isinstance(run, Ok) and run.value == "run-detail"
    assert isinstance(comparison, Ok) and comparison.value == "comparison"
    assert isinstance(table, Ok) and table.value == "table-page"
    assert isinstance(artifact, Ok) and artifact.value == "artifact-stream"
    assert discovery.snapshot_query is not None
    assert isinstance(discovery.run_query, RunQuery)
    assert [call[0] for call in operations.calls] == [
        "run",
        "compare",
        "page",
        "artifact",
    ]


def test_unexpected_exception_is_sanitized_logged_with_correlation_id_and_no_raw_exception() -> (
    None
):
    logger = Logger()

    class ExplodingSnapshotService:
        def inspect_snapshot(self, snapshot_id: str) -> object:
            raise RuntimeError(f"provider failed using {SECRET} for {snapshot_id}")

    app = ResearchApplication(
        snapshot_manager=ExplodingSnapshotService(),
        logger=logger,
        redactor=Redactor((SECRET,)),
    )

    result = app.inspect_snapshot(SNAPSHOT_ID)

    assert isinstance(result, Err)
    error = result.errors[0]
    assert error.category is ErrorCategory.INTERNAL_UNEXPECTED
    assert error.correlation_id
    assert SECRET not in str(error)
    assert len(logger.entries) == 1
    entry_text = repr(logger.entries[0])
    assert SECRET not in entry_text
    assert logger.entries[0]["exception"] is None
    assert logger.entries[0]["correlation_id"] == error.correlation_id
