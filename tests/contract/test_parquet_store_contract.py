"""Contract tests for bounded deterministic Parquet persistence and scans."""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from quant_research_platform.domain.manifests import ObjectKind
from quant_research_platform.infrastructure.parquet_store import (
    ParquetStore,
    ScanPredicate,
)
from quant_research_platform.infrastructure.schemas import DAILY_BAR_V1, RAW_V1, canonical_table

_CHECKSUM_A = "a" * 64
_CHECKSUM_B = "b" * 64


def _raw_row(day: date, *, symbol: str = "AAPL", close: float = 101.0) -> dict[str, object]:
    return {
        "provider": "yfinance",
        "request_content_key": _CHECKSUM_A,
        "symbol": symbol,
        "provider_date": day,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": close,
        "adj_close": close,
        "volume": 1_000.0,
        "dividends": 0.0,
        "stock_splits": 1.0,
        "provider_fields_json": {"fixture": symbol},
        "provider_record_checksum": bytes.fromhex(_CHECKSUM_B),
    }


def _normalized_row(session: date, *, symbol: str, close: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "session": session,
        "event_ts": datetime(session.year, session.month, session.day, 21, tzinfo=UTC),
        "raw_open": close - 1.0,
        "raw_high": close + 1.0,
        "raw_low": close - 2.0,
        "raw_close": close,
        "raw_volume": 1_000.0,
        "provider_adj_close": close,
        "dividend": 0.0,
        "split_ratio": 1.0,
        "adjusted_open": close - 1.0,
        "adjusted_high": close + 1.0,
        "adjusted_low": close - 2.0,
        "adjusted_close": close,
        "adjusted_volume": 1_000.0,
        "execution_adjusted_open": close - 1.0,
        "sizing_adjusted_close": close,
        "cumulative_price_factor": Decimal("1"),
        "cumulative_split_factor": Decimal("1"),
        "policy_version": "causal_forward_v1",
        "provider_record_checksum": bytes.fromhex(_CHECKSUM_A),
        "canonical_row_checksum": bytes.fromhex(_CHECKSUM_B),
    }


def _reader_rows(reader: pa.RecordBatchReader) -> list[dict[str, object]]:
    batches = list(reader)
    return pa.Table.from_batches(batches).to_pylist() if batches else []


def test_raw_writes_are_separate_sorted_chunked_and_single_row_group(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, write_chunk_size=2)
    rows = [_raw_row(date(2024, 1, day), close=float(100 + day)) for day in (4, 2, 3)]

    objects = store.write_raw(rows)

    assert len(objects) == 2
    assert [object_.row_count for object_ in objects] == [2, 1]
    assert {object_.partition.object_kind for object_ in objects} == {ObjectKind.RAW}
    assert {
        object_.relative_uri.split("/sha256=")[0] for object_ in objects
    } == {"raw/provider=yfinance/symbol=AAPL/year=2024"}
    for object_ in objects:
        parquet_file = pq.ParquetFile(object_.path)
        assert parquet_file.metadata.num_row_groups == 1
        assert parquet_file.metadata.row_group(0).num_rows <= 2
        assert object_.checksum == __import__("hashlib").sha256(object_.path.read_bytes()).hexdigest()

    rows_from_scan = _reader_rows(
        store.scan(
            objects,
            columns=("symbol", "provider_date", "close"),
            symbols=("AAPL",),
            years=(2024,),
            sessions=(date(2024, 1, 2), date(2024, 1, 3)),
            batch_size=1,
        )
    )
    assert [row["provider_date"] for row in rows_from_scan] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert all(set(row) == {"symbol", "provider_date", "close"} for row in rows_from_scan)
    assert store.last_scan_plan is not None
    assert store.last_scan_plan.columns == ("symbol", "provider_date", "close")
    assert store.last_scan_plan.batch_size == 1
    assert store.last_scan_plan.symbols == ("AAPL",)
    assert store.last_scan_plan.years == (2024,)


def test_normalized_writes_partition_by_symbol_year_and_scan_pushes_predicates(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, write_chunk_size=2)
    rows = [
        _normalized_row(date(2024, 1, 3), symbol="AAPL", close=103.0),
        _normalized_row(date(2023, 12, 29), symbol="AAPL", close=99.0),
        _normalized_row(date(2024, 1, 2), symbol="MSFT", close=202.0),
    ]

    objects = store.write_normalized(rows)
    result = _reader_rows(
        store.scan(
            objects,
            columns=("symbol", "session", "adjusted_close"),
            predicate=ScanPredicate.from_values(
                symbols=("AAPL",),
                years=(2024,),
                expression=pc.field("adjusted_close") > 100,
            ),
        )
    )

    assert [(object_.partition.symbol, object_.partition.session_year) for object_ in objects] == [
        ("AAPL", 2023),
        ("AAPL", 2024),
        ("MSFT", 2024),
    ]
    assert all(object_.partition.object_kind is ObjectKind.NORMALIZED for object_ in objects)
    assert result == [
        {"symbol": "AAPL", "session": date(2024, 1, 3), "adjusted_close": 103.0}
    ]
    assert store.last_scan_plan is not None
    assert store.last_scan_plan.source_count == 1
    assert store.last_scan_plan.has_expression


def test_canonical_permutations_produce_byte_identical_final_files(tmp_path: Path) -> None:
    rows = [
        _normalized_row(date(2024, 1, 4), symbol="AAPL", close=104.0),
        _normalized_row(date(2024, 1, 2), symbol="AAPL", close=102.0),
        _normalized_row(date(2024, 1, 3), symbol="AAPL", close=103.0),
    ]
    first = ParquetStore(tmp_path / "first", write_chunk_size=2).write_normalized(rows)
    second = ParquetStore(tmp_path / "second", write_chunk_size=2).write_normalized(list(reversed(rows)))

    assert [(item.relative_uri, item.checksum, item.row_count) for item in first] == [
        (item.relative_uri, item.checksum, item.row_count) for item in second
    ]
    for original, repeated in zip(first, second, strict=True):
        assert original.path.read_bytes() == repeated.path.read_bytes()


def test_writer_accepts_bounded_canonical_arrow_batches_and_scan_stays_streaming(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, write_chunk_size=2)
    canonical = canonical_table(
        RAW_V1,
        [_raw_row(date(2024, 1, day)) for day in (2, 3, 4)],
    )
    reader = pa.RecordBatchReader.from_batches(canonical.schema, canonical.to_batches())

    objects = store.write_raw(reader)

    assert [object_.row_count for object_ in objects] == [2, 1]
    source = inspect.getsource(ParquetStore.scan)
    assert "read_all(" not in source
    assert ".to_pandas(" not in source
    with pytest.raises(ValueError, match="columns"):
        store.scan(objects, columns=())


def test_writer_rejects_collection_schema_mismatches(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    normalized = canonical_table(
        DAILY_BAR_V1,
        [_normalized_row(date(2024, 1, 2), symbol="AAPL", close=102.0)],
    )

    with pytest.raises(Exception, match="schema"):
        store.write_raw(normalized)


class _LengthHintGuard:
    """A one-pass source that fails if a caller tries to size it up front."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = iter(rows)
        self.consumed = 0

    def __iter__(self) -> _LengthHintGuard:
        return self

    def __next__(self) -> dict[str, object]:
        row = next(self._rows)
        self.consumed += 1
        return row

    def __length_hint__(self) -> int:
        raise AssertionError("a chunked writer must not materialize the source")


class _ScannerSpy:
    """Expose only the streaming reader seam of an Arrow dataset scanner."""

    def __init__(self, scanner: object) -> None:
        self._scanner = scanner
        self.to_reader_called = False

    def to_reader(self) -> pa.RecordBatchReader:
        self.to_reader_called = True
        return self._scanner.to_reader()  # type: ignore[union-attr,no-any-return]


class _DatasetSpy:
    """Record projection/filter scanner construction without a table API."""

    def __init__(self, dataset: object, scanner_calls: list[dict[str, object]]) -> None:
        self._dataset = dataset
        self._scanner_calls = scanner_calls
        self.scanner_spy: _ScannerSpy | None = None

    def scanner(self, **kwargs: object) -> _ScannerSpy:
        self._scanner_calls.append(kwargs)
        scanner = self._dataset.scanner(**kwargs)  # type: ignore[union-attr]
        self.scanner_spy = _ScannerSpy(scanner)
        return self.scanner_spy


def test_parquet_schema_options_and_checksum_are_pinned_for_final_bytes(
    tmp_path: Path,
) -> None:
    from quant_research_platform.infrastructure.parquet_store import (
        PARQUET_WRITE_OPTIONS,
    )
    from quant_research_platform.infrastructure.schemas import schema_for

    store = ParquetStore(tmp_path, write_chunk_size=2)
    objects = store.write_raw(
        [_raw_row(date(2024, 1, day)) for day in (2, 3, 4)]
    )

    assert PARQUET_WRITE_OPTIONS["version"] == "2.6"
    assert PARQUET_WRITE_OPTIONS["compression"] == "zstd"
    for object_ in objects:
        parquet_file = pq.ParquetFile(object_.path)
        metadata = parquet_file.metadata
        assert parquet_file.schema_arrow.equals(schema_for(RAW_V1), check_metadata=True)
        assert metadata.format_version == "2.6"
        assert metadata.num_row_groups == 1
        column = metadata.row_group(0).column(0)
        assert column.compression == "ZSTD"
        assert column.statistics is None
        assert object_.checksum == __import__("hashlib").sha256(
            object_.path.read_bytes()
        ).hexdigest()


def test_chunked_writes_and_filtered_scans_use_only_streaming_seams(
    tmp_path: Path,
) -> None:
    import pyarrow.dataset as ds

    stream = _LengthHintGuard(
        [
            _raw_row(date(2024, 1, day), symbol="AAPL")
            for day in (2, 3, 4)
        ]
        + [
            _raw_row(date(2025, 1, day), symbol="MSFT")
            for day in (2, 3)
        ]
    )
    scanner_calls: list[dict[str, object]] = []
    factory_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    dataset_spies: list[_DatasetSpy] = []

    def dataset_factory(paths: object, **kwargs: object) -> _DatasetSpy:
        resolved_paths = tuple(str(path) for path in paths)  # type: ignore[union-attr]
        factory_calls.append((resolved_paths, kwargs))
        spy = _DatasetSpy(ds.dataset(resolved_paths, **kwargs), scanner_calls)
        dataset_spies.append(spy)
        return spy

    store = ParquetStore(
        tmp_path,
        write_chunk_size=2,
        scan_batch_size=1,
        dataset_factory=dataset_factory,
    )
    objects = store.write_raw(stream)
    reader = store.scan(
        objects,
        columns=("symbol", "provider_date", "close"),
        symbols=("AAPL",),
        years=(2024,),
        sessions=(date(2024, 1, 3),),
        batch_size=1,
    )

    # Dataset filters can yield an empty physical batch before the matching row.
    batch = next(candidate for candidate in reader if candidate.num_rows)
    assert batch.to_pylist() == [
        {"symbol": "AAPL", "provider_date": date(2024, 1, 3), "close": 101.0}
    ]

    assert stream.consumed == 5
    assert factory_calls
    assert all("symbol=AAPL" in path for path in factory_calls[0][0])
    assert scanner_calls[0]["columns"] == ["symbol", "provider_date", "close"]
    assert scanner_calls[0]["batch_size"] == 1
    assert scanner_calls[0]["filter"] is not None
    assert dataset_spies[0].scanner_spy is not None
    assert dataset_spies[0].scanner_spy.to_reader_called
