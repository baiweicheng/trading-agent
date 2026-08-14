"""Bounded-memory behavior and local streaming benchmark coverage.

These tests intentionally observe the streaming seams instead of asserting a
machine-specific RSS number.  Guarded readers fail if a consumer switches to a
whole-table ``read_all``/``to_pandas`` path, while spies record the bounded
batch, chunk, projection, predicate, and page limits.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from quant_research_platform.application.decisions import CausalDecisionDelivery
from quant_research_platform.application.evaluation import EvaluationService
from quant_research_platform.application.inspection import InspectionService
from quant_research_platform.application.ports import fetch_batched_daily
from quant_research_platform.config.models import RetryPolicyConfig
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.execution import (
    CoreBacktestOutput,
    DailyReturn,
    PortfolioState,
)
from quant_research_platform.domain.market import (
    CorporateAction,
    DailyBarCandidate,
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    RawCorporateAction,
    RawDailyBar,
    RawLineage,
    SessionKey,
    SymbolOutcome,
    SymbolOutcomeStatus,
)
from quant_research_platform.domain.strategy import (
    RationalWeight,
    StrategyDecision,
    StrategyExclusionReason,
)
from quant_research_platform.domain.validation import ValidationService
from quant_research_platform.infrastructure import parquet_store as parquet_module
from quant_research_platform.infrastructure.parquet_store import ParquetStore

_START = date(2024, 1, 2)
_END = date(2024, 1, 5)
_CHECKSUM_A = "a" * 64
_SNAPSHOT_ID = "snap_" + "b" * 64


def _raw_row(index: int, *, symbol: str = "AAPL") -> dict[str, object]:
    day = _START + timedelta(days=index)
    close = float(100 + index)
    return {
        "provider": "yfinance",
        "request_content_key": _CHECKSUM_A,
        "symbol": symbol,
        "provider_date": day,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "adj_close": close,
        "volume": 1_000.0,
        "dividends": 0.0,
        "stock_splits": 1.0,
        "provider_fields_json": {"fixture": symbol},
        "provider_record_checksum": bytes.fromhex(f"{index + 1:064x}"),
    }


class _OnePassRows:
    """Iterable fixture that rejects eager sizing and whole-table APIs."""

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._rows = tuple(rows)
        self.iterations = 0
        self.yielded = 0

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        self.iterations += 1
        for row in self._rows:
            self.yielded += 1
            yield row

    def __len__(self) -> int:
        raise AssertionError("bounded processing must not size the complete source")

    def __length_hint__(self) -> int:
        raise AssertionError("bounded processing must not request a source length hint")

    def read_all(self) -> object:
        raise AssertionError("unbounded read_all is forbidden")

    def to_pandas(self) -> object:
        raise AssertionError("unbounded to_pandas is forbidden")


class _BatchProvider:
    name = "fixture"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    def fetch_daily(self, request: ProviderRequest) -> ProviderBatchResult:
        self.requests.append(request)
        outcomes = tuple(
            SymbolOutcome(
                symbol=symbol,
                status=SymbolOutcomeStatus.SUCCESS,
                attempts=1,
                records=(
                    ProviderRecord(
                        provider=request.provider,
                        request_content_key=request.content_key,
                        symbol=symbol,
                        raw_bar=RawDailyBar(
                            provider_date=request.start,
                            open=Decimal("100"),
                            high=Decimal("101"),
                            low=Decimal("99"),
                            close=Decimal("100"),
                            volume=Decimal("1000"),
                        ),
                    ),
                ),
            )
            for symbol in request.symbols
        )
        return ProviderBatchResult(request=request, outcomes=outcomes)


class _AllSessions:
    name = "fixture-xnys"
    version = "fixture-1"

    @staticmethod
    def is_session(value: date) -> bool:
        return True


class _StreamingCandidates:
    """One-pass candidate source with no sequence/materialization protocol."""

    def __init__(self, values: Sequence[DailyBarCandidate]) -> None:
        self._values = tuple(values)
        self.iterations = 0
        self.consumed: list[SessionKey] = []

    def __iter__(self) -> Iterator[DailyBarCandidate]:
        self.iterations += 1
        for value in self._values:
            self.consumed.append(value.session_key)
            yield value

    def __len__(self) -> int:
        raise AssertionError(
            "validation must consume a stream, not size all partitions"
        )

    def __getitem__(self, index: int) -> DailyBarCandidate:
        raise AssertionError(f"validation must not index a stream: {index}")

    def read_all(self) -> object:
        raise AssertionError("validation must not call read_all")

    def to_pandas(self) -> object:
        raise AssertionError("validation must not call to_pandas")


def _candidate(index: int, symbol: str) -> DailyBarCandidate:
    session = _START + timedelta(days=index)
    checksum = f"{index + 1:064x}"
    lineage = RawLineage("fixture", _CHECKSUM_A, checksum)
    raw_bar = RawDailyBar(
        provider_date=session,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1000"),
    )
    raw_action = RawCorporateAction(dividend=Decimal("0"), split_ratio=Decimal("1"))
    return DailyBarCandidate(
        symbol=symbol,
        session=session,
        event_timestamp=datetime(
            session.year, session.month, session.day, 21, tzinfo=UTC
        ),
        raw_bar=raw_bar,
        raw_action=raw_action,
        corporate_action=CorporateAction(
            symbol=symbol,
            session=session,
            dividend=Decimal("0"),
            split_ratio=Decimal("1"),
            raw_lineage=lineage,
        ),
        adjusted_open=Decimal("100"),
        adjusted_high=Decimal("101"),
        adjusted_low=Decimal("99"),
        adjusted_close=Decimal("100"),
        adjusted_volume=Decimal("1000"),
        execution_adjusted_open=Decimal("100"),
        sizing_adjusted_close=Decimal("100"),
        cumulative_price_factor=Decimal("1"),
        cumulative_split_factor=Decimal("1"),
        policy_version="causal_forward_v1",
        raw_lineage=lineage,
    )


def _candidate_stream() -> tuple[_StreamingCandidates, dict[str, tuple[date, ...]]]:
    """Create a deterministic, key-sorted multi-symbol validation stream."""

    expected = {
        symbol: tuple(_START + timedelta(days=index) for index in range(8))
        for symbol in ("AAPL", "MSFT")
    }
    values = tuple(
        _candidate(index, symbol)
        for symbol in ("AAPL", "MSFT")
        for index in range(8)
    )
    return _StreamingCandidates(values), expected


def test_provider_batches_are_bounded_and_recorded_in_manifest_inputs() -> None:
    provider = _BatchProvider()
    policy = RetryPolicyConfig(
        attempts=1,
        initial_delay_seconds=Decimal("0"),
        max_delay_seconds=Decimal("0"),
        backoff_multiplier=Decimal("1"),
    )
    batches = fetch_batched_daily(
        provider,
        (
            "AAPL",
            "MSFT",
            "PG",
            "XOM",
            "JPM",
            "COST",
            "IBM",
            "ORCL",
            "NVDA",
            "UNH",
            "META",
        ),
        start=_START,
        end=_END,
        batch_size=10,
        policy=policy,
        sleep=lambda _delay: None,
    )

    assert tuple(request.symbols for request in provider.requests) == (
        ("AAPL", "MSFT", "PG", "XOM", "JPM", "COST", "IBM", "ORCL", "NVDA", "UNH"),
        ("META", "SPY"),
    )
    assert all(len(request.symbols) <= 10 for request in provider.requests)
    assert all(
        result.request.symbols == request.symbols
        for result, request in zip(batches, provider.requests, strict=True)
    )
    assert tuple(
        outcome.symbol for result in batches for outcome in result.outcomes
    ) == (
        "AAPL",
        "MSFT",
        "PG",
        "XOM",
        "JPM",
        "COST",
        "IBM",
        "ORCL",
        "NVDA",
        "UNH",
        "META",
        "SPY",
    )


def test_validation_consumes_sorted_partitions_incrementally() -> None:
    candidates, expected = _candidate_stream()
    validation = ValidationService(calendar=_AllSessions()).validate(
        candidates,
        expected,
        staleness_threshold=0,
        requested_range=DateRange(_START, _START + timedelta(days=7)),
    )

    assert candidates.iterations == 1
    assert len(candidates.consumed) == 16
    assert len(validation.accepted_rows) == 16
    assert validation.quarantined_rows == ()
    assert validation.gaps == ()
    assert validation.report.summary.accepted_row_count == 16


class _ScannerGuard:
    def __init__(self, scanner: object) -> None:
        self._scanner = scanner

    def to_reader(self) -> pa.RecordBatchReader:
        return cast(pa.RecordBatchReader, self._scanner.to_reader())  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        if name in {"read_all", "to_pandas", "to_table"}:
            raise AssertionError(f"unbounded scanner method accessed: {name}")
        return getattr(self._scanner, name)


class _DatasetGuard:
    def __init__(self, dataset: ds.Dataset, calls: list[dict[str, object]]) -> None:
        self._dataset = dataset
        self._calls = calls

    def scanner(self, **kwargs: object) -> _ScannerGuard:
        self._calls.append(kwargs)
        return _ScannerGuard(self._dataset.scanner(**kwargs))


def test_chunked_parquet_write_and_projected_scan_remain_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_size = 32
    rows = _OnePassRows(tuple(_raw_row(index) for index in range(257)))
    canonical_calls: list[int] = []
    original_canonical_table = parquet_module.canonical_table

    def canonical_table_spy(schema_name: str, values: object) -> pa.Table:
        canonical_calls.append(len(cast(Sequence[object], values)))
        return original_canonical_table(schema_name, values)  # type: ignore[arg-type]

    monkeypatch.setattr(parquet_module, "canonical_table", canonical_table_spy)
    scanner_calls: list[dict[str, object]] = []

    def dataset_factory(paths: object, **kwargs: object) -> _DatasetGuard:
        dataset = ds.dataset(paths, **kwargs)  # type: ignore[arg-type]
        return _DatasetGuard(dataset, scanner_calls)

    store = ParquetStore(
        tmp_path,
        write_chunk_size=chunk_size,
        scan_batch_size=7,
        dataset_factory=dataset_factory,
    )
    objects = store.write_raw(rows)

    assert rows.iterations == 1
    assert rows.yielded == 257
    assert canonical_calls
    assert max(canonical_calls) <= chunk_size
    assert sum(item.row_count for item in objects) == 257
    assert all(item.row_count <= chunk_size for item in objects)

    reader = store.scan(
        objects,
        columns=("symbol", "provider_date", "close"),
        symbols=("AAPL",),
        session_start=_START,
        session_end=_START + timedelta(days=10),
        batch_size=7,
    )
    batches = tuple(batch for batch in reader if batch.num_rows)
    assert batches
    assert all(batch.num_rows <= 7 for batch in batches)
    assert scanner_calls[0]["columns"] == ["symbol", "provider_date", "close"]
    assert scanner_calls[0]["batch_size"] == 7
    assert scanner_calls[0]["filter"] is not None
    assert store.last_scan_plan is not None
    assert store.last_scan_plan.columns == ("symbol", "provider_date", "close")
    assert store.last_scan_plan.symbols == ("AAPL",)
    assert store.last_scan_plan.batch_size == 7


class _PageScanner:
    def __init__(self, row_count: int) -> None:
        self.calls: list[dict[str, object]] = []
        self.rows = tuple(
            {"session": index, "symbol": "AAPL", "close": index}
            for index in range(row_count)
        )

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
        if limit is None or limit > 100:
            raise AssertionError("pagination requested an unbounded page")
        return self.rows[offset : offset + limit]

    def read_all(self) -> object:
        raise AssertionError("pagination must not call read_all")

    def to_pandas(self) -> object:
        raise AssertionError("pagination must not call to_pandas")


class _ArtifactMetadata:
    def __init__(self, checksum: str) -> None:
        self.record = SimpleNamespace(
            checksum=checksum,
            artifact_kind="equity",
            relative_uri=f"artifacts/{checksum}.parquet",
            media_type="application/vnd.apache.parquet",
            byte_size=5_000,
            row_count=257,
            schema_version="daily_bar_v1",
            availability="available",
        )

    def get_artifact(self, checksum: str) -> object:
        return self.record

    def set_artifact_availability(self, _checksum: str, availability: str) -> None:
        self.record.availability = availability


class _StreamHandle:
    def stream(self) -> Iterable[bytes]:
        yield b"chunk-1"
        yield b"chunk-2"

    def read_all(self) -> object:
        raise AssertionError("artifact handles must remain streaming")

    def to_pandas(self) -> object:
        raise AssertionError("artifact handles must remain streaming")


class _ArtifactStore:
    def __init__(self) -> None:
        self.opened = 0

    def open_verified_artifact(self, reference: object) -> _StreamHandle:
        del reference
        self.opened += 1
        return _StreamHandle()


def test_artifact_pages_and_downloads_are_bounded_and_separate() -> None:
    checksum = sha256(b"table").hexdigest()
    metadata = _ArtifactMetadata(checksum)
    scanner = _PageScanner(257)
    artifacts = _ArtifactStore()
    service = InspectionService(
        metadata=metadata,
        scanner=scanner,
        artifacts=artifacts,
        configured_page_size=37,
    )

    first = service.page_artifact(
        checksum,
        page=0,
        page_size=10_000,
        columns=("session", "close"),
        order_by=("session",),
    )
    second = service.page_artifact(
        checksum,
        page=1,
        page_size=10_000,
        columns=("session", "close"),
        order_by=("session",),
    )
    opened = service.open_artifact(checksum)

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert isinstance(opened, Ok)
    assert first.value.page_size == 37
    assert len(first.value.rows) == 37
    assert len(second.value.rows) == 37
    assert first.value.rows[-1]["session"] + 1 == second.value.rows[0]["session"]
    assert all(call["limit"] == 37 for call in scanner.calls)
    assert all(call["columns"] == ("session", "close") for call in scanner.calls)
    assert artifacts.opened == 1
    assert list(opened.value.stream()) == [b"chunk-1", b"chunk-2"]
    assert scanner.calls  # full artifact opening never invokes the page scanner


def _ineligible_strategy(
    _history: object,
    *,
    signal_session: date,
    universe: tuple[str, ...],
    params: object,
) -> tuple[StrategyDecision, ...]:
    del params
    return tuple(
        StrategyDecision(
            signal_session=signal_session,
            symbol=symbol,
            endpoint_252_session=None,
            endpoint_252_close=None,
            endpoint_21_session=None,
            endpoint_21_close=None,
            momentum_score=None,
            eligible=False,
            rank=None,
            target_weight=RationalWeight.zero(),
            exclusion_reason=StrategyExclusionReason.MISSING_LONG_ENDPOINT,
        )
        for symbol in universe
    )


class _HistoryProjectionSpy:
    def __init__(self, signal_session: date) -> None:
        self.signal_session = signal_session
        self.calls: list[dict[str, object]] = []

    def read_history(
        self,
        snapshot: object,
        *,
        symbols: tuple[str, ...],
        end_session: date,
        fields: tuple[str, ...],
        start_session: date,
    ) -> tuple[Mapping[str, object], ...]:
        self.calls.append(
            {
                "snapshot": snapshot,
                "symbols": symbols,
                "end_session": end_session,
                "fields": fields,
                "start_session": start_session,
            }
        )
        return tuple(
            {
                "symbol": symbol,
                "session": self.signal_session,
                "adjusted_close": Decimal("100"),
                "sizing_adjusted_close": Decimal("100"),
                "canonical_row_checksum": "c" * 64,
                "tradable": True,
            }
            for symbol in symbols
        )


def test_backtest_decision_history_reads_only_active_projection_and_window() -> None:
    signal = date(2024, 6, 28)
    reader = _HistoryProjectionSpy(signal)
    service = CausalDecisionDelivery(
        snapshot_reader=reader,
        strategy=_ineligible_strategy,
    )

    result = service.deliver(
        SimpleNamespace(snapshot_id=_SNAPSHOT_ID),
        signal,
        universe=("AAPL", "MSFT"),
        position_count=1,
        execution_session=signal + timedelta(days=1),
    )

    assert isinstance(result, Ok)
    assert len(reader.calls) == 1
    call = reader.calls[0]
    assert call["symbols"] == ("AAPL", "MSFT")
    assert call["fields"] == (
        "symbol",
        "session",
        "adjusted_close",
        "sizing_adjusted_close",
        "canonical_row_checksum",
        "tradable",
    )
    assert call["end_session"] == signal
    assert cast(date, call["start_session"]) <= signal
    assert result.value.order_intents == ()


class _EvaluationBatch:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._rows = tuple(rows)

    def to_pylist(self) -> list[Mapping[str, object]]:
        return list(self._rows)

    def read_all(self) -> object:
        raise AssertionError("evaluation must consume bounded record batches")

    def to_pandas(self) -> object:
        raise AssertionError("evaluation must not materialize pandas tables")


class _EvaluationReader:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._rows = tuple(rows)

    def to_batches(self) -> Iterator[_EvaluationBatch]:
        for offset in range(0, len(self._rows), 2):
            yield _EvaluationBatch(self._rows[offset : offset + 2])

    def read_all(self) -> object:
        raise AssertionError("evaluation must not call read_all")

    def to_pandas(self) -> object:
        raise AssertionError("evaluation must not call to_pandas")


class _EvaluationProjectionSpy:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = tuple(rows)
        self.calls: list[dict[str, object]] = []

    def scan(
        self,
        refs: Sequence[object],
        columns: Sequence[str],
        **kwargs: object,
    ) -> _EvaluationReader:
        self.calls.append({"refs": refs, "columns": columns, **kwargs})
        if tuple(columns) != ("symbol", "session", "adjusted_close"):
            raise AssertionError(
                "evaluation requested columns outside its metric projection"
            )
        if kwargs.get("symbols") != ("SPY",):
            raise AssertionError("evaluation did not partition-scan SPY only")
        return _EvaluationReader(self.rows)


def _evaluation_output() -> CoreBacktestOutput:
    sessions = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    states = tuple(
        PortfolioState(
            session=session,
            cash_balance=equity,
            positions=(),
            gross_exposure=Decimal("0"),
            portfolio_equity=equity,
            leverage=Decimal("0"),
        )
        for session, equity in zip(
            sessions,
            (Decimal("100000"), Decimal("101000"), Decimal("100500")),
            strict=True,
        )
    )
    returns = (
        DailyReturn(sessions[0], Decimal("0")),
        DailyReturn(sessions[1], Decimal("0.01")),
        DailyReturn(sessions[2], Decimal("-0.004950495049504950495049504950")),
    )
    return CoreBacktestOutput(
        orders=(),
        fills=(),
        portfolio_states=states,
        daily_returns=returns,
        strategy_decisions=(),
    )


@pytest.mark.memory
def test_local_streaming_benchmark_larger_than_one_chunk_has_no_unbounded_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run a deterministic laptop-sized stream through write, scan, and evaluation."""

    # This is intentionally modest for local CI but larger than the configured
    # write chunk, so a whole-input implementation cannot hide behind one chunk.
    chunk_size = 64
    rows = _OnePassRows(tuple(_raw_row(index) for index in range(513)))
    chunk_calls: list[int] = []
    original_canonical_table = parquet_module.canonical_table

    def bounded_canonical_table(schema_name: str, values: object) -> pa.Table:
        size = len(cast(Sequence[object], values))
        chunk_calls.append(size)
        if size > chunk_size:
            raise AssertionError("benchmark observed an unbounded canonical table")
        return original_canonical_table(schema_name, values)  # type: ignore[arg-type]

    monkeypatch.setattr(parquet_module, "canonical_table", bounded_canonical_table)
    store = ParquetStore(tmp_path, write_chunk_size=chunk_size, scan_batch_size=11)
    objects = store.write_raw(rows)
    reader = store.scan(
        objects,
        columns=("symbol", "provider_date", "close"),
        symbols=("AAPL",),
        batch_size=11,
    )
    observed_batches = tuple(batch for batch in reader if batch.num_rows)

    assert rows.yielded == 513
    assert chunk_calls and max(chunk_calls) <= chunk_size
    assert sum(item.row_count for item in objects) == 513
    assert all(batch.num_rows <= 11 for batch in observed_batches)
    assert store.last_scan_plan is not None
    assert store.last_scan_plan.batch_size == 11
    assert store.last_scan_plan.columns == ("symbol", "provider_date", "close")

    evaluation_sessions = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    )
    projection = _EvaluationProjectionSpy(
        tuple(
            {
                "symbol": "SPY",
                "session": session,
                "adjusted_close": Decimal(str(100 + index)),
            }
            for index, session in enumerate(evaluation_sessions)
        )
    )
    snapshot = SimpleNamespace(
        object_references=(
            SimpleNamespace(object_kind="normalized", schema_version="daily_bar_v1"),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    result = EvaluationService(parquet_store=projection).evaluate(
        _evaluation_output(),
        snapshot,
        evaluation_range=DateRange(*evaluation_sessions[::2]),
    )

    assert isinstance(result, Ok)
    assert len(projection.calls) == 1
    assert projection.calls[0]["columns"] == ("symbol", "session", "adjusted_close")
    assert projection.calls[0]["symbols"] == ("SPY",)
    assert projection.calls[0]["session_start"] == evaluation_sessions[0]
    assert projection.calls[0]["session_end"] == evaluation_sessions[-1]
