"""End-to-end normalization, validation, and Parquet integration coverage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import chain
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.market import (
    DailyBarCandidate,
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    ProviderRequest,
    ProviderRequestMetadata,
    QuarantineRecord,
    RawCorporateAction,
    RawDailyBar,
    SymbolOutcome,
    SymbolOutcomeStatus,
)
from quant_research_platform.domain.normalization import Normalizer
from quant_research_platform.domain.validation import (
    ValidationOutput,
    ValidationService,
)
from quant_research_platform.infrastructure.parquet_store import (
    PARQUET_WRITE_OPTIONS,
    ParquetStore,
    StagedParquetObject,
)
from quant_research_platform.infrastructure.schemas import (
    DAILY_BAR_V1_SCHEMA,
    GAP_V1,
    GAP_V1_SCHEMA,
    QUARANTINE_V1,
    QUARANTINE_V1_SCHEMA,
    RAW_V1_SCHEMA,
    VALIDATION_REPORT_V1,
    VALIDATION_REPORT_V1_SCHEMA,
    daily_bars_to_table,
    gaps_to_table,
    quarantines_to_table,
    raw_records_to_table,
    validation_reports_to_table,
)
from quant_research_platform.infrastructure.xnys_calendar import XNYSCalendar

_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "quality_issues" / "validation_cases.json"
)
_WRITE_CHUNK_SIZE = 2
NormalizationValue: TypeAlias = DailyBarCandidate | QuarantineRecord


@dataclass(frozen=True, slots=True)
class AuxiliaryObject:
    """A test-local checksummed object for schemas without a collection writer."""

    path: Path
    relative_uri: str
    checksum: str
    row_count: int
    schema_version: str


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """All scientific and operational outputs produced by one local run."""

    root: Path
    records: tuple[ProviderRecord, ...]
    provider_batch: ProviderBatchResult
    normalized_values: tuple[NormalizationValue, ...]
    validation: ValidationOutput
    raw_objects: tuple[StagedParquetObject, ...]
    normalized_objects: tuple[StagedParquetObject, ...]
    quarantine_objects: tuple[AuxiliaryObject, ...]
    gap_objects: tuple[AuxiliaryObject, ...]
    report_objects: tuple[AuxiliaryObject, ...]
    store: ParquetStore
    operational_metadata_path: Path


def _load_fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("validation fixture must contain a JSON object")
    return value


def _request(case: Mapping[str, Any]) -> ProviderRequest:
    request = case["request"]
    return ProviderRequest(
        tuple(str(symbol) for symbol in request["symbols"]),
        date.fromisoformat(str(request["start"])),
        date.fromisoformat(str(request["end"])),
    )


def _decimal(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _records(case: Mapping[str, Any]) -> tuple[ProviderRecord, ...]:
    request = _request(case)
    result: list[ProviderRecord] = []
    for raw in case["records"]:
        bar = raw["raw_bar"]
        action = raw["raw_action"]
        result.append(
            ProviderRecord(
                provider="yfinance",
                request_content_key=request.content_key,
                symbol=str(raw["symbol"]),
                raw_bar=RawDailyBar(
                    provider_date=date.fromisoformat(str(raw["provider_date"])),
                    open=_decimal(bar["open"]),
                    high=_decimal(bar["high"]),
                    low=_decimal(bar["low"]),
                    close=_decimal(bar["close"]),
                    adj_close=_decimal(bar["adj_close"]),
                    volume=_decimal(bar["volume"]),
                ),
                raw_action=RawCorporateAction(
                    dividend=_decimal(action["dividend"]),
                    split_ratio=_decimal(action["split_ratio"]),
                    provider_fields=dict(action.get("provider_fields", {})),
                ),
                provider_fields=dict(raw.get("provider_fields", {})),
            )
        )
    return tuple(result)


def _expected_sessions(case: Mapping[str, Any]) -> dict[str, tuple[date, ...]]:
    return {
        str(symbol): tuple(date.fromisoformat(str(value)) for value in sessions)
        for symbol, sessions in case["expected_sessions"].items()
    }


def _batches(
    records: tuple[ProviderRecord, ...], *, permuted: bool
) -> tuple[tuple[ProviderRecord, ...], ...]:
    """Return two deterministic batch layouts containing the same record multiset."""

    first_layout = (
        records[:2],
        records[2:5],
        records[5:],
    )
    if not permuted:
        return first_layout
    return tuple(tuple(reversed(batch)) for batch in reversed(first_layout))


def _stream_batches(
    batches: Sequence[Sequence[ProviderRecord]],
) -> Iterable[ProviderRecord]:
    return chain.from_iterable(batches)


def _provider_batch(
    request: ProviderRequest,
    records: Iterable[ProviderRecord],
    metadata: ProviderRequestMetadata,
) -> ProviderBatchResult:
    records_by_symbol: dict[str, list[ProviderRecord]] = {
        symbol: [] for symbol in request.symbols
    }
    for record in records:
        records_by_symbol[record.symbol].append(record)
    outcomes = tuple(
        SymbolOutcome(
            symbol=symbol,
            status=SymbolOutcomeStatus.SUCCESS,
            attempts=1,
            records=tuple(records_by_symbol[symbol]),
        )
        for symbol in request.symbols
    )
    return ProviderBatchResult(
        request=request,
        outcomes=outcomes,
        operational_metadata=metadata,
    )


def _write_auxiliary_table(
    root: Path,
    collection: str,
    schema_name: str,
    table: pa.Table,
    *,
    max_rows: int,
) -> tuple[AuxiliaryObject, ...]:
    """Write canonical auxiliary rows with the same pinned bytes as the store."""

    if table.schema.metadata is None:
        raise AssertionError("canonical auxiliary tables must carry schema metadata")
    collection_root = root / collection
    collection_root.mkdir(parents=True, exist_ok=True)
    objects: list[AuxiliaryObject] = []
    for ordinal, offset in enumerate(range(0, table.num_rows, max_rows)):
        portion = table.slice(offset, max_rows)
        temporary = collection_root / f".slice-{ordinal:04d}.parquet"
        pq.write_table(
            portion,
            temporary,
            row_group_size=max_rows,
            **{
                **dict(PARQUET_WRITE_OPTIONS),
                # The registered Arrow schema names list children ``item``;
                # preserve that schema when writing the test-only auxiliary
                # collections instead of applying PyArrow's ``element`` name.
                "use_compliant_nested_type": False,
            },
        )
        data = temporary.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        final_path = collection_root / f"sha256={checksum}.parquet"
        temporary.replace(final_path)
        objects.append(
            AuxiliaryObject(
                path=final_path,
                relative_uri=final_path.relative_to(root).as_posix(),
                checksum=checksum,
                row_count=portion.num_rows,
                schema_version=schema_name,
            )
        )
    return tuple(objects)


def _run_pipeline(
    case: Mapping[str, Any],
    root: Path,
    batches: tuple[tuple[ProviderRecord, ...], ...],
    metadata: ProviderRequestMetadata,
) -> PipelineResult:
    records = tuple(_stream_batches(batches))
    request = _request(case)
    provider_batch = _provider_batch(request, records, metadata)
    root.mkdir(parents=True, exist_ok=True)
    operational_metadata_path = root / "operational" / "provider-request.json"
    operational_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    operational_metadata_path.write_bytes(
        canonical_json(provider_batch.operational_metadata.to_operational_dict())  # type: ignore[union-attr]
    )

    calendar = XNYSCalendar()
    store = ParquetStore(root, write_chunk_size=_WRITE_CHUNK_SIZE, scan_batch_size=2)
    raw_objects = store.write_raw(_stream_batches(batches))
    normalized_values = tuple(
        Normalizer().normalize(_stream_batches(batches), calendar)
    )

    request_range = DateRange(
        date.fromisoformat(str(case["request"]["start"])),
        date.fromisoformat(str(case["request"]["end"])),
    )
    validation = ValidationService(calendar=calendar).validate(
        normalized_values,
        _expected_sessions(case),
        int(case["staleness_threshold"]),
        requested_range=request_range,
        comparison_range=(
            DateRange(
                date.fromisoformat(str(case["comparison_range"]["start"])),
                date.fromisoformat(str(case["comparison_range"]["end"])),
            )
            if "comparison_range" in case
            else None
        ),
    )
    normalized_objects = store.write_normalized(validation.accepted_rows)
    quarantine_objects = _write_auxiliary_table(
        root,
        "quarantine",
        QUARANTINE_V1,
        quarantines_to_table(validation.quarantined_rows),
        max_rows=_WRITE_CHUNK_SIZE,
    )
    gap_objects = _write_auxiliary_table(
        root,
        "gap",
        GAP_V1,
        gaps_to_table(validation.gaps),
        max_rows=_WRITE_CHUNK_SIZE,
    )
    report_objects = _write_auxiliary_table(
        root,
        "validation",
        VALIDATION_REPORT_V1,
        validation_reports_to_table((validation.report,)),
        max_rows=_WRITE_CHUNK_SIZE,
    )
    return PipelineResult(
        root=root,
        records=records,
        provider_batch=provider_batch,
        normalized_values=normalized_values,
        validation=validation,
        raw_objects=raw_objects,
        normalized_objects=normalized_objects,
        quarantine_objects=quarantine_objects,
        gap_objects=gap_objects,
        report_objects=report_objects,
        store=store,
        operational_metadata_path=operational_metadata_path,
    )


def _read_rows(
    store: ParquetStore,
    refs: Sequence[StagedParquetObject | AuxiliaryObject],
    columns: Sequence[str],
    **filters: object,
) -> list[dict[str, object]]:
    paths: tuple[StagedParquetObject | Path, ...] = tuple(
        ref if isinstance(ref, StagedParquetObject) else ref.path for ref in refs
    )
    reader = store.scan(paths, columns=columns, batch_size=2, **filters)  # type: ignore[arg-type]
    batches = tuple(reader)
    if not batches:
        return []
    return pa.Table.from_batches(batches).to_pylist()


def _gap_sort_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row["symbol"]),
        str(row["expected_session"]),
        str(row["reason"]),
    )


def _assert_object_schema(
    objects: Sequence[StagedParquetObject | AuxiliaryObject],
    expected_schema: pa.Schema,
    root: Path,
    max_rows: int,
) -> None:
    for object_ in objects:
        assert object_.path.is_relative_to(root)
        assert object_.path.relative_to(root).as_posix() == object_.relative_uri
        assert object_.row_count <= max_rows
        parquet_file = pq.ParquetFile(object_.path)
        assert parquet_file.metadata.num_row_groups == 1
        assert parquet_file.metadata.row_group(0).num_rows == object_.row_count
        assert parquet_file.schema_arrow.equals(expected_schema, check_metadata=True)
        assert hashlib.sha256(object_.path.read_bytes()).hexdigest() == object_.checksum


def _object_identity(
    objects: Sequence[StagedParquetObject | AuxiliaryObject],
) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (
            object_.relative_uri,
            object_.checksum,
            object_.row_count,
            object_.path.stat().st_size,
        )
        for object_ in objects
    )


def _assert_same_object_bytes(
    first: Sequence[StagedParquetObject | AuxiliaryObject],
    second: Sequence[StagedParquetObject | AuxiliaryObject],
) -> None:
    assert _object_identity(first) == _object_identity(second)
    for first_object, second_object in zip(first, second, strict=True):
        assert first_object.path.read_bytes() == second_object.path.read_bytes()


@pytest.mark.integration
def test_quality_fixture_streams_through_bounded_collections_and_filtered_rereads(
    tmp_path: Path,
) -> None:
    """Persist every outcome separately and verify canonical rereads/lineage."""

    fixture = _load_fixture()
    case = fixture["cases"]["non_session_gap_and_stale_symbols"]
    records = _records(case)
    request = _request(case)
    metadata = ProviderRequestMetadata(
        request_content_key=request.content_key,
        retrieved_at=datetime(2024, 1, 6, 15, tzinfo=UTC),
        response_status="200",
        request_id="quality-run-1",
        retrieval_started_at=datetime(2024, 1, 6, 14, tzinfo=UTC),
    )
    result = _run_pipeline(
        case,
        tmp_path / "quality-run",
        _batches(records, permuted=False),
        metadata,
    )

    assert result.provider_batch.status == "succeeded"
    assert result.provider_batch.operational_metadata is metadata
    assert result.operational_metadata_path.is_file()
    assert json.loads(result.operational_metadata_path.read_text(encoding="utf-8"))[
        "request_id"
    ] == ("quality-run-1")

    assert sum(object_.row_count for object_ in result.raw_objects) == len(records) == 9
    assert (
        sum(object_.row_count for object_ in result.normalized_objects)
        == len(result.validation.accepted_rows)
        == 7
    )
    assert (
        sum(object_.row_count for object_ in result.quarantine_objects)
        == len(result.validation.quarantined_rows)
        == 1
    )
    assert (
        sum(object_.row_count for object_ in result.gap_objects)
        == len(result.validation.gaps)
        == 5
    )
    assert sum(object_.row_count for object_ in result.report_objects) == 1
    assert result.validation.report.content_checksum == (
        "85942a573a78d9860c9dc0664eef0ba320d41b2e4281fc5374e39b1a0993cd2a"
    )

    _assert_object_schema(
        result.raw_objects, RAW_V1_SCHEMA, result.root, _WRITE_CHUNK_SIZE
    )
    _assert_object_schema(
        result.normalized_objects,
        DAILY_BAR_V1_SCHEMA,
        result.root,
        _WRITE_CHUNK_SIZE,
    )
    _assert_object_schema(
        result.quarantine_objects,
        QUARANTINE_V1_SCHEMA,
        result.root,
        _WRITE_CHUNK_SIZE,
    )
    _assert_object_schema(
        result.gap_objects, GAP_V1_SCHEMA, result.root, _WRITE_CHUNK_SIZE
    )
    _assert_object_schema(
        result.report_objects,
        VALIDATION_REPORT_V1_SCHEMA,
        result.root,
        _WRITE_CHUNK_SIZE,
    )

    raw_uris = {object_.relative_uri for object_ in result.raw_objects}
    normalized_uris = {object_.relative_uri for object_ in result.normalized_objects}
    quarantine_uris = {object_.relative_uri for object_ in result.quarantine_objects}
    gap_uris = {object_.relative_uri for object_ in result.gap_objects}
    assert all(uri.startswith("raw/provider=yfinance/symbol=") for uri in raw_uris)
    assert all(uri.startswith("normalized/symbol=") for uri in normalized_uris)
    assert all(uri.startswith("quarantine/") for uri in quarantine_uris)
    assert all(uri.startswith("gap/") for uri in gap_uris)
    assert raw_uris.isdisjoint(normalized_uris)
    assert raw_uris.isdisjoint(quarantine_uris | gap_uris)
    assert normalized_uris.isdisjoint(quarantine_uris | gap_uris)

    raw_rows = _read_rows(
        result.store,
        result.raw_objects,
        RAW_V1_SCHEMA.names,
    )
    assert raw_rows == raw_records_to_table(records).to_pylist()
    normalized_rows = _read_rows(
        result.store,
        result.normalized_objects,
        DAILY_BAR_V1_SCHEMA.names,
    )
    assert (
        normalized_rows
        == daily_bars_to_table(result.validation.accepted_rows).to_pylist()
    )
    quarantine_rows = _read_rows(
        result.store,
        result.quarantine_objects,
        QUARANTINE_V1_SCHEMA.names,
    )
    assert (
        quarantine_rows
        == quarantines_to_table(result.validation.quarantined_rows).to_pylist()
    )
    gap_rows = _read_rows(result.store, result.gap_objects, GAP_V1_SCHEMA.names)
    expected_gap_rows = gaps_to_table(result.validation.gaps).to_pylist()
    assert sorted(gap_rows, key=_gap_sort_key) == sorted(
        expected_gap_rows,
        key=_gap_sort_key,
    )
    report_rows = _read_rows(
        result.store,
        result.report_objects,
        VALIDATION_REPORT_V1_SCHEMA.names,
    )
    assert (
        report_rows
        == validation_reports_to_table((result.validation.report,)).to_pylist()
    )

    filtered_raw = _read_rows(
        result.store,
        result.raw_objects,
        ("symbol", "provider_date", "provider_record_checksum"),
        symbols=("AAPL",),
        sessions=(date(2024, 1, 3),),
    )
    assert len(filtered_raw) == 1
    assert filtered_raw[0]["provider_date"] == date(2024, 1, 3)
    filtered_normalized = _read_rows(
        result.store,
        result.normalized_objects,
        ("symbol", "session", "adjusted_close", "provider_record_checksum"),
        symbols=("AAPL",),
        session_start=date(2024, 1, 2),
        session_end=date(2024, 1, 3),
    )
    assert [(row["symbol"], row["session"]) for row in filtered_normalized] == [
        ("AAPL", date(2024, 1, 2)),
        ("AAPL", date(2024, 1, 3)),
    ]

    raw_by_checksum = {
        bytes.fromhex(record.provider_record_checksum): record for record in records
    }
    accepted_by_key = {
        candidate.session_key: candidate
        for candidate in result.validation.accepted_rows
    }
    for row in normalized_rows:
        checksum = row["provider_record_checksum"]
        assert isinstance(checksum, bytes)
        source = raw_by_checksum[checksum]
        key = (row["symbol"], row["session"])
        assert source.symbol == key[0]
        assert source.provider_date == key[1]
        candidate = accepted_by_key[
            next(
                candidate_key
                for candidate_key in accepted_by_key
                if candidate_key.symbol == key[0] and candidate_key.session == key[1]
            )
        ]
        assert checksum == bytes.fromhex(candidate.raw_lineage.provider_record_checksum)
        assert row["canonical_row_checksum"] == bytes.fromhex(
            candidate.canonical_row_checksum
        )
        assert candidate.raw_lineage.provider == source.provider
        assert candidate.raw_lineage.request_content_key == request.content_key

    accepted_session_keys = {
        (candidate.symbol, candidate.session)
        for candidate in result.validation.accepted_rows
    }
    assert all(
        (row["symbol"], row["expected_session"]) not in accepted_session_keys
        for row in gap_rows
    )


@pytest.mark.integration
def test_quality_fixture_permutations_preserve_scientific_objects_and_expose_metadata(
    tmp_path: Path,
) -> None:
    """Record/batch permutations change request history, not scientific objects."""

    fixture = _load_fixture()
    case = fixture["cases"]["non_session_gap_and_stale_symbols"]
    records = _records(case)
    request = _request(case)
    first_root = tmp_path / "first"
    second_root = tmp_path / "permuted"
    first_metadata = ProviderRequestMetadata(
        request_content_key=request.content_key,
        retrieved_at=datetime(2024, 1, 6, 15, tzinfo=UTC),
        response_status="200",
        request_id="request-first",
    )
    second_metadata = ProviderRequestMetadata(
        request_content_key=request.content_key,
        retrieved_at=datetime(2024, 1, 7, 16, tzinfo=UTC),
        response_status="200",
        request_id="request-second",
    )

    first = _run_pipeline(
        case,
        first_root,
        _batches(records, permuted=False),
        first_metadata,
    )
    second = _run_pipeline(
        case,
        second_root,
        _batches(records, permuted=True),
        second_metadata,
    )

    _assert_same_object_bytes(first.raw_objects, second.raw_objects)
    _assert_same_object_bytes(first.normalized_objects, second.normalized_objects)
    _assert_same_object_bytes(first.quarantine_objects, second.quarantine_objects)
    _assert_same_object_bytes(first.gap_objects, second.gap_objects)
    _assert_same_object_bytes(first.report_objects, second.report_objects)
    assert (
        first.validation.report.content_checksum
        == second.validation.report.content_checksum
    )
    assert [
        candidate.canonical_row_checksum for candidate in first.validation.accepted_rows
    ] == [
        candidate.canonical_row_checksum
        for candidate in second.validation.accepted_rows
    ]
    assert [value.to_content_dict() for value in first.normalized_values] == [
        value.to_content_dict() for value in second.normalized_values
    ]
    assert [gap.to_content_dict() for gap in first.validation.gaps] == [
        gap.to_content_dict() for gap in second.validation.gaps
    ]

    first_operational = first.provider_batch.operational_metadata
    second_operational = second.provider_batch.operational_metadata
    assert first_operational is not None
    assert second_operational is not None
    assert (
        first_operational.request_content_key
        == second_operational.request_content_key
        == (request.content_key)
    )
    assert (
        first_operational.to_operational_dict()
        != second_operational.to_operational_dict()
    )
    assert first_operational.request_id == "request-first"
    assert second_operational.request_id == "request-second"
    assert (
        first.operational_metadata_path.read_bytes()
        != second.operational_metadata_path.read_bytes()
    )
    assert json.loads(first.operational_metadata_path.read_text(encoding="utf-8"))[
        "request_id"
    ] == ("request-first")
    assert json.loads(second.operational_metadata_path.read_text(encoding="utf-8"))[
        "request_id"
    ] == ("request-second")
    assert first.operational_metadata_path.relative_to(first.root).as_posix() == (
        "operational/provider-request.json"
    )
    assert all(
        not object_.relative_uri.startswith("operational/")
        for object_ in (*first.raw_objects, *first.normalized_objects)
    )
