"""Offline local smoke coverage for the Phase 1 composition boundaries.

The smoke suite deliberately uses temporary roots and a deterministic provider.
The only provider-network seam lives in the separately marked external test in
``test_yfinance_contract.py`` and is skipped unless explicitly enabled.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from streamlit.testing.v1 import AppTest

from quant_research_platform.application.backtests import BacktestRequest
from quant_research_platform.application.ingestion import IngestionRequest
from quant_research_platform.application.services import Page
from quant_research_platform.domain.errors import ActionableError, ErrorCategory, Ok
from quant_research_platform.domain.execution import (
    INITIAL_PORTFOLIO_EQUITY,
    CoreBacktestOutput,
    DailyReturn,
    PortfolioState,
)
from quant_research_platform.domain.market import (
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    RawCorporateAction,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
)
from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.filesystem_store import FilesystemStore
from quant_research_platform.infrastructure.logging import StructuredJsonlLogger
from quant_research_platform.infrastructure.mlflow_tracker import LocalMlflowTracker
from quant_research_platform.infrastructure.xnys_calendar import XNYSCalendar
from quant_research_platform.ui.app import build_application
from tests.integration.test_snapshot_ingestion_faults import SnapshotParquetWriter

pytestmark = pytest.mark.smoke

SMOKE_START = date(2024, 1, 2)
SMOKE_END = date(2024, 1, 5)
SECRET = "https://user:password@proxy.invalid"


class OfflineProvider:
    """Small deterministic provider implementation with no network path."""

    name = "yfinance"

    def __init__(self, calendar: XNYSCalendar) -> None:
        self.calendar = calendar
        self.requests: list[ProviderRequest] = []

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        self.requests.append(request)
        outcomes: list[SymbolOutcome] = []
        sessions = self.calendar.sessions(
            request.start,
            request.end,
            completed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for symbol in request.symbols:
            base = Decimal("100") if symbol == "AAPL" else Decimal("300")
            records = tuple(
                ProviderRecord(
                    provider=self.name,
                    request_content_key=request.content_key,
                    symbol=symbol,
                    raw_bar=RawDailyBar(
                        provider_date=session,
                        open=base + ordinal,
                        high=base + ordinal + Decimal("2"),
                        low=base + ordinal - Decimal("1"),
                        close=base + ordinal + Decimal("1"),
                        adj_close=base + ordinal + Decimal("1"),
                        volume=Decimal("1000"),
                    ),
                    raw_action=RawCorporateAction(
                        dividend=Decimal("0"),
                        split_ratio=Decimal("1"),
                    ),
                    provider_fields={"fixture": "local-smoke"},
                )
                for ordinal, session in enumerate(sessions)
            )
            outcomes.append(
                SymbolOutcome(
                    symbol=symbol,
                    status=SymbolOutcomeStatus.SUCCESS,
                    attempts=1,
                    records=records,
                )
            )
        return ProviderBatchResult(request=request, outcomes=tuple(outcomes))


class FixtureBacktest:
    """Exercise the real evaluation seam without starting Zipline."""

    def __init__(self, application: object, calendar: XNYSCalendar) -> None:
        self.application = application
        self.calendar = calendar
        self.calls: list[BacktestRequest] = []

    def run(
        self,
        request: BacktestRequest,
        config: object,
        *,
        progress: object | None = None,
    ) -> object:
        del progress
        self.calls.append(request)
        snapshot_result = self.application.snapshot_manager.open_verified(
            request.snapshot_id
        )
        assert isinstance(snapshot_result, Ok)
        sessions = self.calendar.sessions(
            request.evaluation_range.start,
            request.evaluation_range.end,
            completed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        states = tuple(
            PortfolioState(
                session=session,
                cash_balance=INITIAL_PORTFOLIO_EQUITY,
                positions=(),
                gross_exposure=Decimal("0"),
                portfolio_equity=INITIAL_PORTFOLIO_EQUITY,
                leverage=Decimal("0"),
            )
            for session in sessions
        )
        output = CoreBacktestOutput(
            orders=(),
            fills=(),
            portfolio_states=states,
            daily_returns=tuple(
                DailyReturn(session=session, return_value=Decimal("0"))
                for session in sessions
            ),
            strategy_decisions=(),
        )
        evaluated = self.application.evaluator.evaluate(
            output,
            snapshot_result.value,
            config,
            evaluation_range=request.evaluation_range,
        )
        assert isinstance(evaluated, Ok), evaluated
        return Ok(
            SimpleNamespace(
                run_id="local-smoke-run",
                snapshot_id=request.snapshot_id,
                evaluation_range=request.evaluation_range,
                core_output=output,
                evaluation=evaluated.value,
            )
        )


class UiSmokeApplication:
    """Read-only facade double for one in-process Streamlit render pass."""

    def list_snapshots(self, query: object) -> Page[object]:
        return Page(items=(), page=query.page, page_size=query.page_size, total=0)

    def search_runs(self, query: object) -> Page[object]:
        return Page(items=(), page=query.page, page_size=query.page_size, total=0)


def _write_project_boundary(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'local-smoke-project'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )


def _write_smoke_config(root: Path) -> Path:
    path = root / "smoke.yaml"
    path.write_text(
        "data:\n"
        "  universe: [AAPL]\n"
        "  requested_range:\n"
        f"    start: {SMOKE_START.isoformat()}\n"
        f"    end: {SMOKE_END.isoformat()}\n"
        "  batch_size: 2\n"
        "  staleness_sessions: 10\n"
        "retry:\n"
        "  attempts: 1\n"
        "  initial_delay_seconds: 0\n"
        "  max_delay_seconds: 0\n"
        "  backoff_multiplier: 1\n",
        encoding="utf-8",
    )
    return path


def test_local_storage_and_mlflow_sqlite_smoke_use_temporary_roots(
    tmp_path: Path,
) -> None:
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    mlflow_path = tmp_path / "mlflow.db"
    try:
        assert (tmp_path / "metadata.duckdb").is_file()
        assert store.artifacts_root.is_dir()
        with store.publisher_lock():
            assert (store.lock_root / "publisher.lock").is_file()

        tracker = LocalMlflowTracker(mlflow_path)
        run = tracker.allocate_run(
            run_id=UUID("00000000-0000-0000-0000-000000001904"),
            snapshot_id="snap_" + "a" * 64,
            strategy_id="monthly_momentum_v1",
            evaluation_start=SMOKE_START,
            evaluation_end=SMOKE_END,
            universe=("AAPL",),
            configuration_checksum="b" * 64,
            environment_checksum="c" * 64,
            strategy_parameters={"position_count": 1},
            deterministic_seed=0,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        tracker.finalize_failure(
            run,
            (
                ActionableError(
                    operation="smoke",
                    category=ErrorCategory.EXPERIMENT_RECORDING,
                    message="local smoke diagnostic",
                    corrective_action="Inspect the local smoke output.",
                ),
            ),
        )
        assert mlflow_path.is_file()
    finally:
        metadata.close()


def test_offline_fixture_ingestion_evaluation_and_facade_are_local_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import yfinance

    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"offline smoke unexpectedly called yfinance: {args}, {kwargs}"
        )

    monkeypatch.setattr(yfinance, "download", fail_network)
    _write_project_boundary(tmp_path)
    config_path = _write_smoke_config(tmp_path)
    application = build_application(tmp_path)
    metadata = application.metadata_store
    calendar = application.ingestion_service.calendar
    provider = OfflineProvider(calendar)
    application.ingestion_service.provider = provider
    application.ingestion_service.calendar = calendar
    application.ingestion_service.parquet_store = SnapshotParquetWriter(
        tmp_path / "parquet"
    )
    application.evaluator = application.backtest_service.evaluator

    resolution = application.resolve_configuration(
        config_path,
        environment={"QRP_SECRETS__HTTPS_PROXY": SECRET},
    )
    assert isinstance(resolution, Ok)
    assert SECRET not in repr(resolution.value.view)
    handle = resolution.value.handle

    ingestion = application.ingest(IngestionRequest(), handle)
    assert isinstance(ingestion, Ok), ingestion
    assert ingestion.value.snapshot_id.startswith("snap_")
    assert provider.requests == [
        ProviderRequest(("AAPL", "SPY"), SMOKE_START, SMOKE_END)
    ]
    snapshot = application.snapshot_manager.open_verified(ingestion.value.snapshot_id)
    assert isinstance(snapshot, Ok)
    assert snapshot.value.snapshot_id == ingestion.value.snapshot_id

    fixture_backtest = FixtureBacktest(application, calendar)
    application.backtest_service = fixture_backtest
    backtest = application.run_backtest(
        BacktestRequest(ingestion.value.snapshot_id, ingestion.value.requested_range),
        handle,
    )
    assert isinstance(backtest, Ok), backtest
    assert fixture_backtest.calls[0].snapshot_id == ingestion.value.snapshot_id
    assert backtest.value.evaluation.artifact_checksums
    assert backtest.value.evaluation.limitation_disclosure.version

    logger = application.logger
    assert isinstance(logger, StructuredJsonlLogger)
    logger.write(
        level="warning",
        operation="smoke",
        correlation_id="local-smoke",
        message=f"diagnostic includes {SECRET}",
        context={"proxy": SECRET},
    )
    diagnostic_text = logger.path.read_text(encoding="utf-8")
    assert SECRET not in diagnostic_text
    assert "[REDACTED]" in diagnostic_text
    assert (
        metadata.get_snapshot(ingestion.value.snapshot_id).availability.value
        == "available"
    )
    metadata.close()


def _render_ui(application, root) -> None:
    import streamlit as st

    from quant_research_platform.ui.app import main

    main(st_module=st, application=application, project_root=root)


def test_streamlit_composition_and_apptest_start_in_process_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import yfinance

    monkeypatch.setattr(
        yfinance,
        "download",
        lambda *_args, **_kwargs: pytest.fail("UI startup must not call yfinance"),
    )
    app_test = AppTest.from_function(
        _render_ui,
        args=(UiSmokeApplication(), tmp_path),
        default_timeout=10,
    ).run()
    assert not app_test.exception
    assert app_test.title[0].value == "Quantitative Research Platform"
    assert not (tmp_path / "streamlit-server-started").exists()
