"""Offline Phase 1 vertical-slice integration coverage.

The fixture uses the real configuration, ingestion, Parquet/CAS/DuckDB snapshot,
bundle projection, evaluation, tracking, inspection, and comparison services.
Only the provider, calendar, MLflow client, and event-loop execution seam are
local deterministic fakes; no network or external service is contacted.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence
from uuid import UUID

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from quant_research_platform.application.backtests import (  # noqa: E402
    BacktestRequest,
    BacktestService,
)
from quant_research_platform.application.decisions import (  # noqa: E402
    CausalDecisionDelivery,
)
from quant_research_platform.application.evaluation import (  # noqa: E402
    EvaluationService,
)
from quant_research_platform.application.ingestion import (  # noqa: E402
    DataIngestionService,
    IngestionRequest,
)
from quant_research_platform.application.jobs import (  # noqa: E402
    SynchronousJobManager,
)
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
    LimitationDisclosure,
    Ok,
)
from quant_research_platform.domain.execution import (  # noqa: E402
    CoreBacktestOutput,
    DailyReturn,
    FillRecord,
    INITIAL_PORTFOLIO_EQUITY,
    OrderRecord,
    OrderStatus,
    PortfolioState,
    Position,
    deterministic_fill_id,
    quantize_money,
)
from quant_research_platform.domain.manifests import (  # noqa: E402
    ContentAddressedObjectRef,
    ObjectKind,
)
from quant_research_platform.domain.market import (  # noqa: E402
    DateRange,
    ProviderRecord,
    ProviderRequest,
    RawCorporateAction,
    RawDailyBar,
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
from quant_research_platform.infrastructure.zipline_bundle import (  # noqa: E402
    ZiplineBundleAdapter,
)
from quant_research_platform.infrastructure.zipline_engine import (  # noqa: E402
    CashSafeOpenBlotter,
)
from tests.integration.test_snapshot_ingestion_faults import (  # noqa: E402
    FixedJobClock,
    FixtureCalendar,
    OfflineYFinanceFixture,
    SnapshotParquetWriter as BaseSnapshotParquetWriter,
)
from tests.integration.test_zipline_bundle_rebuild import (  # noqa: E402
    _DeterministicWriter,
)


SECRET = "https://user:password@proxy.invalid"
START = date(2023, 1, 3)
END = date(2024, 2, 2)
CALENDAR_END = date(2024, 2, 29)


def _long_xnys_sessions() -> tuple[date, ...]:
    holidays = {
        date(2023, 1, 16),
        date(2023, 2, 20),
        date(2023, 4, 7),
        date(2023, 5, 29),
        date(2023, 6, 19),
        date(2023, 7, 4),
        date(2023, 9, 4),
        date(2023, 11, 23),
        date(2023, 12, 25),
        date(2024, 1, 1),
        date(2024, 2, 19),
    }
    sessions: list[date] = []
    current = START
    while current <= CALENDAR_END:
        if current.weekday() < 5 and current not in holidays:
            sessions.append(current)
        current += timedelta(days=1)
    return tuple(sessions)


class LongFixtureCalendar(FixtureCalendar):
    """Deterministic XNYS-shaped sessions with enough history for one signal."""

    def __init__(self) -> None:
        self.sessions_tuple = _long_xnys_sessions()
        super().__init__(self.sessions_tuple)

    def next_session(self, session: date) -> date:
        for candidate in self.sessions_tuple:
            if candidate > session:
                return candidate
        raise ValueError("fixture has no next session")

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]:
        month_ends: list[date] = []
        for session in self.sessions(start, end):
            next_session = self.next_session(session)
            if next_session.month != session.month:
                month_ends.append(session)
        return tuple(month_ends)


class QualityOfflineProvider(OfflineYFinanceFixture):
    """Extend the existing local provider with one deterministic bad row."""

    def __init__(self, calendar: LongFixtureCalendar) -> None:
        super().__init__(calendar, failed_symbols=("MSFT",))
        self.include_quality_issue = True

    def records_for(
        self,
        request: ProviderRequest,
        *,
        symbol: str | None = None,
    ) -> tuple[ProviderRecord, ...]:
        records = list(super().records_for(request, symbol=symbol))
        if self.include_quality_issue and (symbol is None or symbol == "AAPL"):
            records.append(
                ProviderRecord(
                    provider=self.name,
                    request_content_key=request.content_key,
                    symbol="AAPL",
                    raw_bar=RawDailyBar(
                        provider_date=date(2023, 1, 7),
                        open=Decimal("100"),
                        high=Decimal("101"),
                        low=Decimal("99"),
                        close=Decimal("100"),
                        adj_close=Decimal("100"),
                        volume=Decimal("1000"),
                    ),
                    raw_action=RawCorporateAction(
                        dividend=Decimal("0"),
                        split_ratio=Decimal("1"),
                        provider_fields={"fixture": "non-session"},
                    ),
                    provider_fields={"fixture": "non-session"},
                )
            )
        return tuple(records)


@dataclass(frozen=True)
class StagedAuxiliaryObject:
    """Checksum-addressed local Parquet output for validation-only collections."""

    object_ref: ContentAddressedObjectRef
    path: Path

    @property
    def checksum(self) -> str:
        return self.object_ref.checksum

    @property
    def relative_uri(self) -> str:
        return self.object_ref.relative_uri


class SnapshotParquetWriter(BaseSnapshotParquetWriter):
    """Reuse the production raw/normalized writer and add bounded quality tables."""

    def _write_auxiliary(
        self,
        rows: Sequence[object],
        *,
        schema_name: str,
        object_kind: ObjectKind,
        converter: object,
        write_chunk_size: int | None = None,
        staging: Path | None = None,
    ) -> tuple[StagedAuxiliaryObject, ...]:
        import pyarrow.parquet as pq

        from quant_research_platform.infrastructure.schemas import (
            PARQUET_WRITE_OPTIONS,
        )

        materialized = tuple(rows)
        if not materialized:
            return ()
        chunk_size = write_chunk_size or self.store.write_chunk_size
        if chunk_size <= 0:
            raise ValueError("write_chunk_size must be positive")
        output_root = Path(staging) if staging is not None else self.store.root
        outputs: list[StagedAuxiliaryObject] = []
        convert = converter
        for offset in range(0, len(materialized), chunk_size):
            table = convert(materialized[offset : offset + chunk_size])
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
                output_root
                / "auxiliary"
                / schema_name
                / f"sha256={checksum}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            outputs.append(
                StagedAuxiliaryObject(
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
    ) -> tuple[StagedAuxiliaryObject, ...]:
        from quant_research_platform.infrastructure.schemas import (
            quarantines_to_table,
        )

        return self._write_auxiliary(
            rows,
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
    ) -> tuple[StagedAuxiliaryObject, ...]:
        from quant_research_platform.infrastructure.schemas import gaps_to_table

        return self._write_auxiliary(
            rows,
            schema_name="gap_v1",
            object_kind=ObjectKind.GAP,
            converter=gaps_to_table,
            write_chunk_size=write_chunk_size,
            staging=staging,
        )


@dataclass
class PublishedProjection:
    """Projected Parquet reader used by decisions, bundle materialization, and evaluation."""

    store: FilesystemStore
    calls: list[dict[str, object]]

    def _normalized_refs(
        self, snapshot: object | None = None
    ) -> tuple[ContentAddressedObjectRef, ...]:
        manifest = getattr(snapshot, "manifest", snapshot)
        identity = getattr(manifest, "content_identity", None)
        references = getattr(identity, "objects", None)
        if references is None:
            references = getattr(snapshot, "object_references", ())
        return tuple(
            reference
            for reference in references
            if reference.object_kind is ObjectKind.NORMALIZED
        )

    def _rows(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        *,
        symbols: Sequence[str] = (),
        session_start: date | None = None,
        session_end: date | None = None,
    ) -> tuple[dict[str, object], ...]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        selected: list[dict[str, object]] = []
        allowed_symbols = set(symbols)
        for reference in references:
            parquet = pq.ParquetFile(
                pa.BufferReader(self.store.read_object(reference.relative_uri))
            )
            available_columns = set(parquet.schema_arrow.names)
            projected_columns = [
                column for column in columns if column in available_columns
            ]
            for batch in parquet.iter_batches(
                columns=projected_columns, batch_size=64, use_threads=False
            ):
                for row in batch.to_pylist():
                    if allowed_symbols and row["symbol"] not in allowed_symbols:
                        continue
                    if session_start is not None and row["session"] < session_start:
                        continue
                    if session_end is not None and row["session"] > session_end:
                        continue
                    selected.append(row)
        return tuple(
            sorted(
                (
                    {
                        **row,
                        **{
                            field: Decimal(str(row[field]))
                            for field in ("raw_open", "raw_close")
                            if field in row and row[field] is not None
                        },
                    }
                    for row in selected
                ),
                key=lambda row: (row["symbol"], row["session"]),
            )
        )

    def read_history(
        self,
        snapshot: object,
        *,
        symbols: tuple[str, ...],
        end_session: date,
        fields: tuple[str, ...],
        start_session: date | None = None,
    ) -> tuple[dict[str, object], ...]:
        self.calls.append(
            {
                "kind": "history",
                "columns": fields,
                "symbols": symbols,
                "session_start": start_session,
                "session_end": end_session,
            }
        )
        rows = self._rows(
            self._normalized_refs(snapshot),
            fields,
            symbols=symbols,
            session_start=start_session,
            session_end=end_session,
        )
        if "tradable" in fields:
            rows = tuple({**row, "tradable": True} for row in rows)
        assert all(row["session"] <= end_session for row in rows)
        return rows

    def scan(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        *,
        predicate: object | None = None,
        symbols: Sequence[str] | None = None,
        session_start: date | None = None,
        session_end: date | None = None,
        **_: object,
    ) -> tuple[dict[str, object], ...]:
        if predicate is not None:
            symbols = symbols or getattr(predicate, "symbols", None)
            session_start = session_start or getattr(predicate, "session_start", None)
            session_end = session_end or getattr(predicate, "session_end", None)
        self.calls.append(
            {
                "kind": "scan",
                "columns": tuple(columns),
                "symbols": tuple(symbols or ()),
                "session_start": session_start,
                "session_end": session_end,
            }
        )
        return self._rows(
            references,
            columns,
            symbols=tuple(symbols or ()),
            session_start=session_start,
            session_end=session_end,
        )


class LocalMomentumEngine:
    """Local event-loop seam using the real causal decision and cash-safe policies."""

    def __init__(
        self,
        snapshot_manager: SnapshotManager,
        reader: PublishedProjection,
        calendar: LongFixtureCalendar,
    ) -> None:
        self.snapshot_manager = snapshot_manager
        self.reader = reader
        self.calendar = calendar
        self.last_output: CoreBacktestOutput | None = None

    def run(
        self,
        bundle: object,
        request: BacktestRequest,
        config: object,
        progress: object | None = None,
    ) -> Ok[CoreBacktestOutput]:
        opened = self.snapshot_manager.open_verified(request.snapshot_id)
        assert isinstance(opened, Ok)
        snapshot = opened.value
        rows = self.reader._rows(
            self.reader._normalized_refs(snapshot),
            ("symbol", "session", "raw_open", "raw_close"),
            session_start=request.evaluation_range.start,
            session_end=request.evaluation_range.end,
        )
        bars = {(row["symbol"], row["session"]): row for row in rows}
        delivery = CausalDecisionDelivery(
            snapshot_reader=self.reader,
            calendar=self.calendar,
            resolved_config=config,
        )
        pending: dict[date, list[object]] = {}
        all_decisions: list[object] = []
        order_intents: dict[str, object] = {}
        fills: list[FillRecord] = []
        holdings: dict[str, int] = {}
        cash = INITIAL_PORTFOLIO_EQUITY
        states: list[PortfolioState] = []
        returns: list[DailyReturn] = []
        prior_equity: Decimal | None = None
        sessions = self.calendar.sessions(
            request.evaluation_range.start,
            request.evaluation_range.end,
        )
        for index, session in enumerate(sessions, start=1):
            active_orders = pending.pop(session, ())
            if active_orders:
                opens = {
                    (intent.symbol, intent.execution_session): bars[
                        (intent.symbol, session)
                    ]["raw_open"]
                    for intent in active_orders
                }
                executed = CashSafeOpenBlotter(
                    commission_bps=config.execution.commission_bps,
                    slippage_bps=config.execution.slippage_bps,
                ).execute_orders(
                    active_orders,
                    opens=opens,
                    cash=cash,
                    positions=holdings,
                    session=session,
                )
                cash = executed.cash_balance
                holdings = dict(executed.positions)
                for ordinal, fill in enumerate(executed.fills):
                    fills.append(
                        FillRecord(
                            fill_id=deterministic_fill_id(
                                order_id=fill.order_id,
                                symbol=fill.symbol,
                                session=fill.session,
                                quantity=fill.quantity,
                                ordinal=ordinal,
                            ),
                            order_id=fill.order_id,
                            symbol=fill.symbol,
                            session=fill.session,
                            quantity=fill.quantity,
                            ordinal=ordinal,
                            base_adjusted_open=fill.base_adjusted_open,
                            fill_price=fill.fill_price,
                            gross_notional=fill.gross_notional,
                            commission=fill.commission,
                            slippage_cost=fill.slippage_cost,
                        )
                    )

            positions = tuple(
                Position(
                    symbol=symbol,
                    quantity=quantity,
                    mark_price=quantize_money(
                        bars[(symbol, session)]["raw_close"]
                    ),
                    market_value=quantize_money(
                        quantity * Decimal(str(bars[(symbol, session)]["raw_close"]))
                    ),
                )
                for symbol, quantity in sorted(holdings.items())
                if quantity
            )
            gross = quantize_money(sum((item.market_value for item in positions), Decimal("0")))
            equity = quantize_money(cash + gross)
            state = PortfolioState(
                session=session,
                cash_balance=cash,
                positions=positions,
                gross_exposure=gross,
                portfolio_equity=equity,
                leverage=(gross / equity).quantize(Decimal("0.000000000000000001")),
            )
            states.append(state)
            return_value = Decimal("0") if prior_equity is None else equity / prior_equity - Decimal("1")
            returns.append(DailyReturn(session=session, return_value=return_value))
            prior_equity = equity

            if session in self.calendar.month_end_sessions(
                request.evaluation_range.start,
                request.evaluation_range.end,
            ):
                portfolio = SimpleNamespace(cash=cash, positions=holdings)
                delivered = delivery.deliver(
                    snapshot,
                    session,
                    portfolio,
                    universe=config.data.universe,
                    position_count=config.strategy.position_count,
                    execution_session=self.calendar.next_session(session),
                )
                assert isinstance(delivered, Ok)
                value = delivered.value
                all_decisions.extend(value.decisions)
                for intent in value.order_intents:
                    order_intents[intent.order_id] = intent
                    pending.setdefault(intent.execution_session, []).append(intent)
            if callable(progress):
                progress(index, len(sessions))

        orders: list[OrderRecord] = []
        for intent in order_intents.values():
            filled = sum(fill.quantity for fill in fills if fill.order_id == intent.order_id)
            status = OrderStatus.FILLED if filled == intent.requested_quantity else OrderStatus.UNFILLED
            orders.append(
                OrderRecord(
                    order_id=intent.order_id,
                    signal_session=intent.signal_session,
                    execution_session=intent.execution_session,
                    symbol=intent.symbol,
                    requested_quantity=intent.requested_quantity,
                    ordinal=intent.ordinal,
                    decision_rank=intent.decision_rank,
                    status=status,
                    unfilled_reason=None if status is OrderStatus.FILLED else "local execution remainder",
                )
            )
        output = CoreBacktestOutput(
            orders=tuple(sorted(orders, key=lambda item: item.order_id)),
            fills=tuple(sorted(fills, key=lambda item: item.fill_id)),
            portfolio_states=tuple(states),
            daily_returns=tuple(returns),
            strategy_decisions=tuple(
                sorted(all_decisions, key=lambda item: (item.signal_session, item.symbol))
            ),
        )
        self.last_output = output
        return Ok(output)


class TrackingClient:
    """Local MLflow client seam; it records every call for redaction assertions."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.run_count = 0

    def get_experiment_by_name(self, name: str) -> None:
        self.calls.append(("get_experiment", name))
        return None

    def create_experiment(self, name: str) -> str:
        self.calls.append(("create_experiment", name))
        return "offline-experiment"

    def create_run(self, experiment_id: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append(("create_run", experiment_id, kwargs))
        self.run_count += 1
        return SimpleNamespace(info=SimpleNamespace(run_id=f"offline-{self.run_count}"))

    def log_param(self, run_id: str, key: str, value: object) -> None:
        self.calls.append(("param", run_id, key, value))

    def log_metric(self, run_id: str, key: str, value: float, **kwargs: object) -> None:
        self.calls.append(("metric", run_id, key, value, kwargs))

    def set_tag(self, run_id: str, key: str, value: object) -> None:
        self.calls.append(("tag", run_id, key, value))

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.calls.append(("text", run_id, text, artifact_file))

    def set_terminated(self, run_id: str, **kwargs: object) -> None:
        self.calls.append(("terminated", run_id, kwargs))


class ArtifactVerifier:
    def __init__(self, store: FilesystemStore, metadata: DuckDBMetadataStore) -> None:
        self.store = store
        self.metadata = metadata

    def open_verified_artifact(self, reference: object) -> object:
        checksum = str(getattr(reference, "checksum", reference))
        record = self.metadata.get_artifact(checksum)
        publication = json.loads(
            (self.store.publications_root / f"{record.checksum}.json").read_bytes()
        )
        metadata_checksum = sha256_bytes(canonical_json(publication["metadata"]))
        artifact = ArtifactReference(
            checksum=record.checksum,
            byte_size=record.byte_size,
            relative_uri=record.relative_uri,
            metadata_checksum=metadata_checksum,
        )
        stream = self.store.stream_artifact(artifact)
        first = next(stream, None)

        def verified_stream() -> object:
            if first is not None:
                yield first
            yield from stream

        return verified_stream()

    def create_staging(self, operation_id: str | None = None) -> object:
        return self.store.create_staging(operation_id)

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
        metadata: Mapping[str, object],
    ) -> object:
        return self.store.publish_artifact(staged, metadata=metadata)


class RunViews:
    """Enrich immutable DuckDB records with locally retained manifest projections."""

    def __init__(self, metadata: DuckDBMetadataStore) -> None:
        self.metadata = metadata
        self.documents: dict[UUID, dict[str, object]] = {}

    def get_run(self, run_id: UUID | str) -> object:
        identifier = run_id if isinstance(run_id, UUID) else UUID(str(run_id))
        record = self.metadata.get_run(identifier)
        document = self.documents[identifier]
        values = {
            name: getattr(record, name)
            for name in (
                "run_id", "mlflow_run_id", "snapshot_id", "state", "strategy_id",
                "evaluation_start", "evaluation_end", "universe",
                "configuration_checksum", "environment_checksum", "manifest_checksum",
                "manifest_uri", "created_at", "started_at", "ended_at", "error_json",
                "immutable",
            )
        }
        values.update(document)
        return SimpleNamespace(**values)

    def get_artifact(self, checksum: str) -> object:
        return self.metadata.get_artifact(checksum)


def _config_yaml(root: Path) -> bytes:
    return (
        "paths:\n"
        f"  data_root: {root / 'data'}\n"
        f"  artifact_root: {root / 'artifacts'}\n"
        f"  metadata_db: {root / 'metadata.duckdb'}\n"
        f"  mlflow_db: {root / 'mlflow.db'}\n"
        "data:\n"
        "  universe: [AAPL, MSFT]\n"
        "  requested_range:\n"
        f"    start: {START.isoformat()}\n"
        f"    end: {END.isoformat()}\n"
        "  batch_size: 3\n"
        "  staleness_sessions: 10\n"
        "  write_chunk_rows: 64\n"
        "strategy:\n"
        "  position_count: 1\n"
        "execution:\n"
        "  commission_bps: 5\n"
        "  slippage_bps: 10\n"
        "runtime:\n"
        "  deterministic_seed: 7\n"
    ).encode("utf-8")


def _record_evaluation_artifacts(
    metadata: DuckDBMetadataStore,
    evaluation: object,
    created_at: datetime,
) -> tuple[SimpleNamespace, ...]:
    links: list[SimpleNamespace] = []
    for artifact in evaluation.artifacts:
        reference = artifact.reference
        assert reference is not None
        metadata.record_artifact(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.ARTIFACT,
                checksum=artifact.checksum,
                relative_uri=reference.relative_uri,
                schema_version=artifact.schema_version,
                row_count=(
                    artifact.row_count
                    if isinstance(artifact.row_count, int)
                    else 0
                ),
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


def _manifest_for_run(
    result: object,
    config: object,
    links: Sequence[SimpleNamespace],
) -> tuple[dict[str, object], bytes, str]:
    evaluation = result.evaluation
    non_secret = non_secret_config(config).model_dump(mode="json")
    manifest: dict[str, object] = {
        "content_identity": {
            "snapshot_id": result.snapshot_id,
            "strategy_id": config.strategy.identifier,
            "evaluation_range": result.evaluation_range.to_content_dict(),
            "configuration_checksum": sha256_bytes(canonical_json(non_secret)),
            "artifact_checksums": {
                item.role: item.checksum for item in links
            },
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
        "metric_rows": json.loads(evaluation.artifacts["metrics"].payload),
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
    payload = canonical_json(manifest)
    return manifest, payload, sha256_bytes(payload)


class TrackingAdapter:
    """Inject redacted secrets and retain manifest projections around the real tracker."""

    def __init__(
        self,
        tracker: LocalMlflowTracker,
        config: object,
        metadata: DuckDBMetadataStore,
        store: FilesystemStore,
        run_views: RunViews,
    ) -> None:
        self.tracker = tracker
        self.config = config
        self.metadata = metadata
        self.store = store
        self.run_views = run_views

    def allocate_run(self, **values: object) -> object:
        values["secret_values"] = (SECRET,)
        return self.tracker.allocate_run(**values)

    def finalize_success(self, run: object, result: object) -> object:
        links = _record_evaluation_artifacts(self.metadata, result.evaluation, datetime(2024, 2, 5, tzinfo=UTC))
        manifest, payload, checksum = _manifest_for_run(result, self.config, links)
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
        record = self.run_views.documents
        identifier = result.run_id if isinstance(result.run_id, UUID) else UUID(str(result.run_id))
        record[identifier] = {
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
        tracked = SimpleNamespace(
            manifest=manifest,
            manifest_checksum=checksum,
            manifest_uri=reference.relative_uri,
            evaluation=result.evaluation,
            ended_at=datetime(2024, 2, 5, tzinfo=UTC),
        )
        return self.tracker.finalize_success(run, tracked)

    def open_verified_artifact(self, run_id: UUID | str, checksum: str) -> object:
        return self.tracker.open_verified_artifact(run_id, checksum)


@pytest.mark.integration
def test_offline_phase1_pipeline_is_complete_redacted_and_reproducible(
    tmp_path: Path,
) -> None:
    calendar = LongFixtureCalendar()
    assert len(calendar.sessions(START, END)) >= 254
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    jobs = SynchronousJobManager(
        metadata,
        StructuredJsonlLogger(
            tmp_path / "diagnostics.jsonl",
            redactor=Redactor((SECRET,)),
            utc_now=lambda: datetime(2024, 2, 5, tzinfo=UTC),
        ),
        redactor=Redactor((SECRET,)),
        clock=FixedJobClock(),
    )
    provider = QualityOfflineProvider(calendar)
    ingestion = DataIngestionService(
        provider,
        calendar,
        normalizer=None,
        validator=None,
        parquet_store=SnapshotParquetWriter(tmp_path / "parquet"),
        snapshot_publisher=store,
        metadata=metadata,
        job_manager=jobs,
        clock=type("Clock", (), {"utc_now": lambda self: datetime(2024, 2, 5, tzinfo=UTC)})(),
        sleep=lambda _: None,
        redactor=Redactor((SECRET,)),
    )
    manager = SnapshotManager(
        storage=LocalPublishedSnapshotStore(store.root),
        metadata=metadata,
    )
    configuration_manager = ConfigurationManager(
        project_anchor=Path(__file__).resolve(),
    )
    application = ResearchApplication(
        configuration_manager=configuration_manager,
        ingestion_service=ingestion,
        snapshot_manager=manager,
    )

    yaml_path = tmp_path / "effective.yaml"
    yaml_path.write_bytes(_config_yaml(tmp_path))
    resolution = application.resolve_configuration(
        yaml_path,
        environment={"QRP_SECRETS__HTTPS_PROXY": SECRET},
    )
    assert isinstance(resolution, Ok)
    handle = resolution.value.handle
    assert resolution.value.view.secrets.https_proxy.value == "present_redacted"
    assert SECRET not in repr(resolution.value.view)

    progress: list[object] = []
    partial = application.ingest(IngestionRequest(), handle, progress=progress.append)
    assert isinstance(partial, Ok)
    assert partial.value.job_state.value == "partially_succeeded"
    assert partial.value.failed_symbols == ("MSFT",)
    assert partial.value.gaps
    assert partial.value.quarantined
    assert partial.value.limitation_disclosure.data_failures
    partial_manifest = partial.value.manifest
    assert partial_manifest is not None
    partial_roles = {
        reference.object_kind
        for reference in partial_manifest.content_identity.objects
    }
    assert {
        ObjectKind.RAW,
        ObjectKind.NORMALIZED,
        ObjectKind.QUARANTINE,
        ObjectKind.GAP,
    } <= partial_roles
    assert all(
        store.read_object(reference.relative_uri)[:4] == b"PAR1"
        for reference in partial_manifest.content_identity.objects
    )
    assert partial.value.job_id is not None
    assert metadata.get_job(partial.value.job_id).state.value == "partially_succeeded"

    provider.failed_symbols.clear()
    provider.include_quality_issue = False
    clean = application.ingest(IngestionRequest(), handle, progress=progress.append)
    assert isinstance(clean, Ok)
    assert clean.value.job_state.value == "succeeded"
    assert clean.value.failed_symbols == ()
    assert clean.value.gaps == ()
    assert clean.value.quarantined == ()
    assert clean.value.limitation_disclosure.version == LimitationDisclosure.current().version
    clean_manifest = clean.value.manifest
    clean_roles = {reference.object_kind for reference in clean_manifest.content_identity.objects}
    assert clean_roles == {ObjectKind.RAW, ObjectKind.NORMALIZED}
    assert clean.value.snapshot_id != partial.value.snapshot_id
    assert isinstance(manager.open_verified(partial.value.snapshot_id), Ok)
    assert isinstance(manager.open_verified(clean.value.snapshot_id), Ok)

    listed = application.list_snapshots(SnapshotQuery(page=0, page_size=100))
    assert isinstance(listed, Page)
    assert {item.snapshot_id for item in listed.items} == {
        partial.value.snapshot_id,
        clean.value.snapshot_id,
    }
    inspected_snapshot = application.inspect_snapshot(clean.value.snapshot_id)
    assert isinstance(inspected_snapshot, Ok)
    assert inspected_snapshot.value.snapshot_id == clean.value.snapshot_id
    assert inspected_snapshot.value.limitation_disclosure.version == LimitationDisclosure.current().version
    assert inspected_snapshot.value.comparison_ready

    projection = PublishedProjection(store, [])
    writer = _DeterministicWriter()
    bundle_adapter = ZiplineBundleAdapter(
        snapshot_manager=manager,
        data_source=projection,
        calendar=calendar,
        zipline_root=tmp_path / "derived",
        writer=writer,
    )
    bundle = bundle_adapter.materialize(clean.value.snapshot_id)
    assert isinstance(bundle, Ok)
    assert bundle.value.snapshot_id == clean.value.snapshot_id
    assert bundle.value.bundle_name != "latest"
    bundle_manifest = json.loads(
        (bundle.value.cache_path / "bundle_manifest.json").read_bytes()
    )
    assert bundle_manifest["bundle_checksum"] == bundle.value.bundle_checksum
    same_bundle = bundle_adapter.materialize(clean.value.snapshot_id)
    assert isinstance(same_bundle, Ok)
    assert same_bundle.value.bundle_checksum == bundle.value.bundle_checksum
    assert projection.calls
    assert all("adjusted" not in column for call in projection.calls for column in call["columns"])
    assert all(
        call["session_start"] is not None and call["session_end"] is not None
        for call in projection.calls
    )

    tracker_client = TrackingClient()
    tracker = LocalMlflowTracker(
        tracking_uri=tmp_path / "mlflow.db",
        metadata_store=metadata,
        artifact_store=store,
        client=tracker_client,
    )
    run_views = RunViews(metadata)
    reader = PublishedProjection(store, [])
    engine = LocalMomentumEngine(manager, reader, calendar)
    evaluator = EvaluationService(
        snapshot_manager=manager,
        parquet_store=reader,
        artifact_store=store,
    )
    tracking = TrackingAdapter(tracker, resolution.value.view, metadata, store, run_views)
    backtest = BacktestService(
        tracker=tracking,
        snapshot_manager=manager,
        bundle_adapter=ZiplineBundleAdapter(
            snapshot_manager=manager,
            data_source=projection,
            calendar=calendar,
            zipline_root=tmp_path / "derived-runs",
            writer=_DeterministicWriter(),
        ),
        engine=engine,
        evaluator=evaluator,
        clock=type("Clock", (), {"utc_now": lambda self: datetime(2024, 2, 5, tzinfo=UTC)})(),
    )

    application.backtest_service = backtest
    backtest_request = BacktestRequest(
        clean.value.snapshot_id,
        DateRange(START, END),
    )
    run_one = application.run_backtest(backtest_request, handle)
    assert isinstance(run_one, Ok), run_one
    assert run_one.value.core_output.initial_equity == INITIAL_PORTFOLIO_EQUITY
    assert run_one.value.core_output.strategy_decisions
    assert run_one.value.core_output.orders
    assert run_one.value.core_output.fills
    assert all(order.status is OrderStatus.FILLED for order in run_one.value.core_output.orders)
    evaluation = run_one.value.evaluation
    assert evaluation.spy_gaps == ()
    assert evaluation.strategy_equity and evaluation.benchmark_equity
    expected_roles = {
        "benchmark_returns", "benchmark_equity", "chart_drawdown", "chart_equity_curve",
        "chart_monthly_returns", "decisions", "drawdown", "fills", "metrics",
        "monthly_returns", "orders", "portfolio", "positions", "strategy_equity",
        "strategy_returns", "transactions",
    }
    assert set(evaluation.artifacts.roles) == expected_roles
    assert all(len(artifact.payload) == artifact.byte_size for artifact in evaluation.artifacts)
    assert all(sha256_bytes(artifact.payload) == artifact.checksum for artifact in evaluation.artifacts)
    assert evaluation.limitation_disclosure.version == LimitationDisclosure.current().version
    assert any(call["kind"] == "history" for call in reader.calls)
    assert all(
        max((row["session"] for row in reader._rows(
            reader._normalized_refs(manager.open_verified(clean.value.snapshot_id).value),
            call["columns"],
            symbols=call["symbols"],
            session_start=call["session_start"],
            session_end=call["session_end"],
        )), default=call["session_end"]) <= call["session_end"]
        for call in reader.calls
        if call["kind"] == "history"
    )

    run_two = application.run_backtest(backtest_request, handle)
    assert isinstance(run_two, Ok), run_two
    assert run_one.value.run_id != run_two.value.run_id
    assert run_one.value.core_output.to_scientific_dict() == run_two.value.core_output.to_scientific_dict()
    evaluation_two = run_two.value.evaluation
    assert evaluation.artifact_checksums == evaluation_two.artifact_checksums
    assert evaluation.strategy_equity == evaluation_two.strategy_equity
    manifest_one = run_views.documents[UUID(str(run_one.value.run_id))]["manifest"]
    manifest_two = run_views.documents[UUID(str(run_two.value.run_id))]["manifest"]
    assert manifest_one == manifest_two
    assert manifest_one["content_identity"]["artifact_checksums"] == {
        role: evaluation.artifacts[role].checksum for role in expected_roles
    }

    application.run_search = metadata
    run_page = application.search_runs(RunQuery(page=0, page_size=100))
    assert run_page.total == 2
    assert all(item.state.value == "succeeded" for item in run_page.items)
    application.metadata_store = run_views
    from quant_research_platform.application.comparisons import ComparisonService
    from quant_research_platform.application.inspection import InspectionService

    application.inspection_service = InspectionService(
        metadata=run_views,
        artifacts=ArtifactVerifier(store, metadata),
        redactor=Redactor((SECRET,)),
        configured_page_size=100,
    )
    application.comparison_service = ComparisonService(
        metadata=run_views,
        artifacts=ArtifactVerifier(store, metadata),
        redactor=Redactor((SECRET,)),
    )
    application.run_search = metadata

    inspected_run = application.inspect_run(run_one.value.run_id)
    assert isinstance(inspected_run, Ok)
    assert inspected_run.value.limitation_disclosure.version == LimitationDisclosure.current().version
    assert SECRET not in repr(inspected_run.value)
    artifact_checksum = evaluation.artifacts["strategy_equity"].checksum
    opened_artifact = application.open_artifact(artifact_checksum)
    assert isinstance(opened_artifact, Ok)
    assert b"".join(opened_artifact.value.stream()) == evaluation.artifacts["strategy_equity"].payload

    compared = application.compare_runs((run_one.value.run_id, run_two.value.run_id))
    assert isinstance(compared, Ok), compared
    assert compared.value.aligned_sessions
    assert compared.value.artifact_checksum == sha256_bytes(compared.value.artifact.bytes)
    assert compared.value.limitation_disclosure.version == LimitationDisclosure.current().version
    assert SECRET not in repr(compared.value)

    assert all(SECRET not in repr(item) for item in progress)
    assert SECRET not in (tmp_path / "diagnostics.jsonl").read_text()
    assert SECRET not in repr(tracker_client.calls)
    assert SECRET not in repr(partial.value)
    assert SECRET not in repr(clean.value)
    assert metadata.get_job(clean.value.job_id).state.value == "succeeded"
    metadata.close()
