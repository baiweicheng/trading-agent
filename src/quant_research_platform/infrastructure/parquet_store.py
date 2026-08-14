"""Bounded deterministic Parquet persistence and projected streaming scans.

This module owns the PyArrow boundary for tabular scientific data.  Incoming
rows are staged in bounded fragments, DuckDB externally sorts each logical
partition, and fixed-size canonical slices are written with one pinned Parquet
row group each.  The final file bytes, rather than an in-memory table, define
the SHA-256 used by manifests.

``scan`` deliberately returns a :class:`pyarrow.RecordBatchReader`; it never
uses ``read_all`` or pandas.  Its explicit projection and predicate inputs
make the data-window boundary observable in tests and at application seams.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias, cast
from uuid import uuid4

import duckdb  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ..domain.manifests import ContentAddressedObjectRef, ObjectKind
from ..domain.market import DailyBarCandidate, ProviderRecord, normalize_symbol
from .schemas import (
    DAILY_BAR_V1,
    PARQUET_WRITE_OPTIONS,
    RAW_V1,
    SCHEMAS,
    canonical_table,
    daily_bars_to_table,
    raw_records_to_table,
    schema_for,
)

SchemaRow: TypeAlias = Mapping[str, object]
ParquetInput: TypeAlias = (
    pa.Table
    | pa.RecordBatchReader
    | Iterable[SchemaRow | pa.RecordBatch | ProviderRecord | DailyBarCandidate]
)
DatasetFactory: TypeAlias = Callable[..., ds.Dataset]

DEFAULT_WRITE_CHUNK_SIZE: Final = 50_000
MAX_WRITE_CHUNK_SIZE: Final = 100_000
DEFAULT_SCAN_BATCH_SIZE: Final = 65_536
PARQUET_MEDIA_TYPE: Final = "application/vnd.apache.parquet"

_PARTITION_COMPONENT_RE: Final = re.compile(r"^[A-Za-z0-9._-]+$")

# A partition already fixes symbol, so the sort keys need only make rows within
# that one logical partition canonical.  The checksum tie-breakers ensure that
# reordering provider batches or retries cannot change the final slice layout.
_PARTITION_SORT_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    RAW_V1: ("provider_date", "provider_record_checksum"),
    DAILY_BAR_V1: ("session", "canonical_row_checksum"),
}


class ParquetStoreError(RuntimeError):
    """Base class for safe Parquet persistence and scan failures."""


class ParquetWriteError(ParquetStoreError):
    """Raised when a row, partition, or deterministic write is invalid."""


class ParquetScanError(ParquetStoreError):
    """Raised when a projected streaming scan cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class LogicalPartition:
    """A deterministic raw or normalized physical partition identity.

    Raw records are isolated by provider/symbol/provider-date year.  Accepted
    daily bars are isolated by symbol/session year.  The logical directory
    excludes all staging paths and operation identifiers so it is safe to put
    into a scientific manifest after a final CAS publication step.
    """

    object_kind: ObjectKind | str
    schema_version: str
    symbol: str
    session_year: int
    provider: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = ObjectKind(self.object_kind)
        except ValueError as error:
            raise ValueError(f"unsupported logical object kind: {self.object_kind!r}") from error
        object.__setattr__(self, "object_kind", kind)
        schema_for(self.schema_version)
        symbol = normalize_symbol(self.symbol)
        object.__setattr__(self, "symbol", symbol)
        if isinstance(self.session_year, bool) or not isinstance(self.session_year, int):
            raise TypeError("session_year must be an integer")
        if not 1 <= self.session_year <= 9999:
            raise ValueError("session_year must be a valid calendar year")

        if kind is ObjectKind.RAW:
            if self.schema_version != RAW_V1:
                raise ValueError("raw partitions must use raw_v1")
            provider = _partition_component("provider", self.provider)
            object.__setattr__(self, "provider", provider)
        elif kind is ObjectKind.NORMALIZED:
            if self.schema_version != DAILY_BAR_V1:
                raise ValueError("normalized partitions must use daily_bar_v1")
            if self.provider is not None:
                raise ValueError("normalized partitions must not include a provider")
        else:
            raise ValueError("this store writes only raw and normalized partitions")

    @classmethod
    def raw(cls, *, provider: str, symbol: str, year: int) -> LogicalPartition:
        """Construct a raw provider/symbol/year partition."""
        return cls(
            object_kind=ObjectKind.RAW,
            schema_version=RAW_V1,
            provider=provider,
            symbol=symbol,
            session_year=year,
        )

    @classmethod
    def normalized(cls, *, symbol: str, year: int) -> LogicalPartition:
        """Construct a normalized symbol/session-year partition."""
        return cls(
            object_kind=ObjectKind.NORMALIZED,
            schema_version=DAILY_BAR_V1,
            symbol=symbol,
            session_year=year,
        )

    @property
    def relative_directory(self) -> PurePosixPath:
        """Return the immutable logical directory independent of local roots."""
        if self.object_kind is ObjectKind.RAW:
            assert self.provider is not None
            return PurePosixPath(
                f"raw/provider={self.provider}/symbol={self.symbol}/year={self.session_year:04d}"
            )
        return PurePosixPath(
            f"normalized/symbol={self.symbol}/year={self.session_year:04d}"
        )

    def sort_key(self) -> tuple[str, str, int, str]:
        """Order partitions before finalization without using operational paths."""
        return (
            self.object_kind.value,
            self.symbol,
            self.session_year,
            self.provider or "",
        )


@dataclass(frozen=True, slots=True)
class StagedParquetObject:
    """One final-byte-checked canonical Parquet slice in staging or a store root."""

    partition: LogicalPartition
    path: Path
    relative_uri: str
    checksum: str
    row_count: int
    byte_size: int
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.partition, LogicalPartition):
            raise TypeError("partition must be a LogicalPartition")
        path = Path(self.path)
        if not path.name.endswith(".parquet"):
            raise ValueError("path must name a Parquet file")
        object.__setattr__(self, "path", path)
        relative = PurePosixPath(self.relative_uri)
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise ValueError("relative_uri must be a non-escaping relative URI")
        expected_filename = f"sha256={self.checksum}.parquet"
        if relative.name != expected_filename:
            raise ValueError("relative_uri must use the final-byte checksum filename")
        if path.name != expected_filename:
            raise ValueError("path must use the final-byte checksum filename")
        if not isinstance(self.checksum, str) or re.fullmatch(r"[0-9a-f]{64}", self.checksum) is None:
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        for name, value in (("row_count", self.row_count), ("byte_size", self.byte_size), ("ordinal", self.ordinal)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def object_ref(self) -> ContentAddressedObjectRef:
        """Convert this staged table slice into the manifest-facing reference."""
        return ContentAddressedObjectRef(
            object_kind=self.partition.object_kind,
            checksum=self.checksum,
            relative_uri=self.relative_uri,
            schema_version=self.partition.schema_version,
            row_count=self.row_count,
            byte_size=self.byte_size,
            symbol=self.partition.symbol,
            session_year=self.partition.session_year,
            media_type=PARQUET_MEDIA_TYPE,
        )


# The design uses this shorter name in the ParquetStore protocol.
StagedObject = StagedParquetObject


@dataclass(frozen=True, slots=True)
class ScanPredicate:
    """Validated column and partition predicates for a bounded Parquet scan."""

    symbols: tuple[str, ...] | None = None
    years: tuple[int, ...] | None = None
    sessions: tuple[date, ...] | None = None
    session_start: date | None = None
    session_end: date | None = None
    expression: ds.Expression | None = None

    def __post_init__(self) -> None:
        if self.symbols is not None:
            if not isinstance(self.symbols, tuple):
                raise TypeError("symbols must be an immutable tuple or None")
            symbols = tuple(normalize_symbol(symbol) for symbol in self.symbols)
            if not symbols or len(set(symbols)) != len(symbols):
                raise ValueError("symbols must be a non-empty tuple of distinct symbols")
            object.__setattr__(self, "symbols", symbols)
        if self.years is not None:
            if not isinstance(self.years, tuple):
                raise TypeError("years must be an immutable tuple or None")
            years = tuple(sorted(set(self.years)))
            if not years or any(isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999 for year in years):
                raise ValueError("years must contain valid distinct calendar years")
            object.__setattr__(self, "years", years)
        if self.sessions is not None:
            if not isinstance(self.sessions, tuple):
                raise TypeError("sessions must be an immutable tuple or None")
            sessions = tuple(sorted(set(_require_date("session", session) for session in self.sessions)))
            if not sessions:
                raise ValueError("sessions must not be empty")
            object.__setattr__(self, "sessions", sessions)
        if self.session_start is not None:
            object.__setattr__(self, "session_start", _require_date("session_start", self.session_start))
        if self.session_end is not None:
            object.__setattr__(self, "session_end", _require_date("session_end", self.session_end))
        if self.session_start is not None and self.session_end is not None and self.session_start > self.session_end:
            raise ValueError("session_start must not be after session_end")
        if self.expression is not None and not isinstance(self.expression, ds.Expression):
            raise TypeError("expression must be a pyarrow.dataset.Expression or None")

    @classmethod
    def from_values(
        cls,
        *,
        symbols: Sequence[str] | None = None,
        years: Sequence[int] | None = None,
        sessions: Sequence[date] | None = None,
        session_start: date | None = None,
        session_end: date | None = None,
        expression: ds.Expression | None = None,
    ) -> ScanPredicate:
        """Create a predicate while accepting caller-friendly sequences."""
        return cls(
            symbols=tuple(symbols) if symbols is not None else None,
            years=tuple(years) if years is not None else None,
            sessions=tuple(sessions) if sessions is not None else None,
            session_start=session_start,
            session_end=session_end,
            expression=expression,
        )


@dataclass(frozen=True, slots=True)
class ScanPlan:
    """Inspectable scan boundary retained for focused memory-bound tests."""

    columns: tuple[str, ...]
    source_count: int
    symbols: tuple[str, ...] | None
    years: tuple[int, ...] | None
    sessions: tuple[date, ...] | None
    session_start: date | None
    session_end: date | None
    batch_size: int
    has_expression: bool


class ParquetStore:
    """Write canonical raw/normalized slices and scan them as record batches."""

    def __init__(
        self,
        root: Path | str,
        *,
        write_chunk_size: int = DEFAULT_WRITE_CHUNK_SIZE,
        scan_batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
        dataset_factory: DatasetFactory | None = None,
    ) -> None:
        self._root = Path(root).expanduser()
        self._write_chunk_size = _bounded_chunk_size(write_chunk_size)
        self._scan_batch_size = _bounded_scan_batch_size(scan_batch_size)
        self._dataset_factory = dataset_factory or cast(DatasetFactory, ds.dataset)
        self._last_scan_plan: ScanPlan | None = None

    @property
    def root(self) -> Path:
        """Return the local root without including it in scientific identities."""
        return self._root

    @property
    def write_chunk_size(self) -> int:
        """Return the pinned maximum canonical Parquet slice size."""
        return self._write_chunk_size

    @property
    def last_scan_plan(self) -> ScanPlan | None:
        """Expose the latest projected scan boundary for test instrumentation."""
        return self._last_scan_plan

    def write_raw(
        self,
        rows: ParquetInput,
        *,
        write_chunk_size: int | None = None,
        staging: Path | str | None = None,
    ) -> tuple[StagedParquetObject, ...]:
        """Write raw provider rows into separate provider/symbol/year collections."""
        return self._write_collection(
            rows,
            schema_name=RAW_V1,
            write_chunk_size=write_chunk_size,
            staging=staging,
        )

    def write_normalized(
        self,
        rows: ParquetInput,
        *,
        write_chunk_size: int | None = None,
        staging: Path | str | None = None,
    ) -> tuple[StagedParquetObject, ...]:
        """Write daily bars into separate canonical symbol/session-year collections."""
        return self._write_collection(
            rows,
            schema_name=DAILY_BAR_V1,
            write_chunk_size=write_chunk_size,
            staging=staging,
        )

    # Explicit aliases keep the collection role visible at use sites.
    write_raw_collection = write_raw
    write_normalized_collection = write_normalized

    def write_chunks(
        self,
        rows: ParquetInput,
        schema: pa.Schema | str,
        logical_partition: LogicalPartition,
        max_rows: int | None = None,
        staging: Path | str | None = None,
    ) -> tuple[StagedParquetObject, ...]:
        """Externally sort and write fixed slices for one known logical partition.

        This method is intentionally useful to ingestion orchestration that has
        already separated a partition.  ``write_raw`` and ``write_normalized``
        provide the higher-level streaming partitioning path.
        """
        schema_name, arrow_schema = _resolve_schema(schema)
        if not isinstance(logical_partition, LogicalPartition):
            raise TypeError("logical_partition must be a LogicalPartition")
        if logical_partition.schema_version != schema_name:
            raise ParquetWriteError("logical partition schema does not match the writer schema")
        max_rows = _bounded_chunk_size(max_rows or self._write_chunk_size)
        output_root = self._output_root(staging)
        operation_root = self._operation_root(output_root)
        try:
            paths = self._stage_partition_rows(
                rows,
                schema_name=schema_name,
                schema=arrow_schema,
                partition=logical_partition,
                max_rows=max_rows,
                operation_root=operation_root,
            )
            return self._finalize_partition(
                partition=logical_partition,
                schema=arrow_schema,
                input_paths=paths,
                max_rows=max_rows,
                output_root=output_root,
                operation_root=operation_root,
            )
        finally:
            shutil.rmtree(operation_root, ignore_errors=True)

    def scan(
        self,
        refs: Sequence[StagedParquetObject | ContentAddressedObjectRef | Path | str],
        columns: Sequence[str],
        predicate: ScanPredicate | ds.Expression | None = None,
        *,
        symbols: Sequence[str] | None = None,
        years: Sequence[int] | None = None,
        sessions: Sequence[date] | None = None,
        session_start: date | None = None,
        session_end: date | None = None,
        batch_size: int | None = None,
    ) -> pa.RecordBatchReader:
        """Return a projected, filtered :class:`RecordBatchReader`.

        The method only constructs a dataset scanner.  It does not collect
        batches, call ``Table.to_pandas``, or otherwise materialize the result.
        """
        normalized_columns = _required_columns(columns)
        resolved = self._resolve_scan_sources(refs)
        schema_name, schema = _resolve_schema_from_sources(resolved)
        _validate_projection(normalized_columns, schema)
        resolved_predicate = _merge_predicates(
            predicate,
            symbols=symbols,
            years=years,
            sessions=sessions,
            session_start=session_start,
            session_end=session_end,
        )
        filtered_sources = _filter_sources_for_partitions(resolved, resolved_predicate)
        if not filtered_sources:
            projected_schema = schema.select(normalized_columns)
            self._last_scan_plan = ScanPlan(
                columns=normalized_columns,
                source_count=0,
                symbols=resolved_predicate.symbols,
                years=resolved_predicate.years,
                sessions=resolved_predicate.sessions,
                session_start=resolved_predicate.session_start,
                session_end=resolved_predicate.session_end,
                batch_size=_bounded_scan_batch_size(batch_size or self._scan_batch_size),
                has_expression=resolved_predicate.expression is not None,
            )
            return pa.RecordBatchReader.from_batches(projected_schema, ())

        filter_expression = _dataset_filter(schema_name, resolved_predicate)
        paths = [str(source.path) for source in filtered_sources]
        requested_batch_size = _bounded_scan_batch_size(batch_size or self._scan_batch_size)
        self._last_scan_plan = ScanPlan(
            columns=normalized_columns,
            source_count=len(paths),
            symbols=resolved_predicate.symbols,
            years=resolved_predicate.years,
            sessions=resolved_predicate.sessions,
            session_start=resolved_predicate.session_start,
            session_end=resolved_predicate.session_end,
            batch_size=requested_batch_size,
            has_expression=resolved_predicate.expression is not None,
        )
        try:
            dataset = self._dataset_factory(paths, format="parquet")
            scanner = dataset.scanner(
                columns=list(normalized_columns),
                filter=filter_expression,
                batch_size=requested_batch_size,
                use_threads=False,
                batch_readahead=1,
                fragment_readahead=1,
            )
            return scanner.to_reader()
        except (OSError, pa.ArrowException) as error:
            raise ParquetScanError(f"cannot construct projected Parquet scan: {error}") from error

    scan_batches = scan

    def _write_collection(
        self,
        rows: ParquetInput,
        *,
        schema_name: str,
        write_chunk_size: int | None,
        staging: Path | str | None,
    ) -> tuple[StagedParquetObject, ...]:
        schema = schema_for(schema_name)
        max_rows = _bounded_chunk_size(write_chunk_size or self._write_chunk_size)
        output_root = self._output_root(staging)
        operation_root = self._operation_root(output_root)
        paths_by_partition: dict[LogicalPartition, list[Path]] = {}
        try:
            for table in _iter_canonical_tables(rows, schema_name, schema, max_rows):
                by_partition: dict[LogicalPartition, list[dict[str, object]]] = {}
                for row in table.to_pylist():
                    partition = _partition_for_row(schema_name, row)
                    by_partition.setdefault(partition, []).append(row)
                for partition, partition_rows in sorted(by_partition.items(), key=lambda item: item[0].sort_key()):
                    fragment = canonical_table(schema_name, partition_rows)
                    paths_by_partition.setdefault(partition, []).append(
                        self._write_input_fragment(
                            fragment,
                            partition=partition,
                            sequence=len(paths_by_partition.get(partition, ())),
                            operation_root=operation_root,
                        )
                    )

            finalized: list[StagedParquetObject] = []
            for partition in sorted(paths_by_partition, key=LogicalPartition.sort_key):
                finalized.extend(
                    self._finalize_partition(
                        partition=partition,
                        schema=schema,
                        input_paths=paths_by_partition[partition],
                        max_rows=max_rows,
                        output_root=output_root,
                        operation_root=operation_root,
                    )
                )
            return tuple(finalized)
        finally:
            shutil.rmtree(operation_root, ignore_errors=True)

    def _stage_partition_rows(
        self,
        rows: ParquetInput,
        *,
        schema_name: str,
        schema: pa.Schema,
        partition: LogicalPartition,
        max_rows: int,
        operation_root: Path,
    ) -> list[Path]:
        paths: list[Path] = []
        for table in _iter_canonical_tables(rows, schema_name, schema, max_rows):
            for row in table.to_pylist():
                if _partition_for_row(schema_name, row) != partition:
                    raise ParquetWriteError(
                        "write_chunks received a row outside its logical partition"
                    )
            paths.append(
                self._write_input_fragment(
                    table,
                    partition=partition,
                    sequence=len(paths),
                    operation_root=operation_root,
                )
            )
        return paths

    @staticmethod
    def _write_input_fragment(
        table: pa.Table,
        *,
        partition: LogicalPartition,
        sequence: int,
        operation_root: Path,
    ) -> Path:
        fragment_path = (
            operation_root
            / "input"
            / Path(partition.relative_directory.as_posix())
            / f"fragment-{sequence:08d}.parquet"
        )
        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pinned_parquet(table, fragment_path, row_group_size=max(1, table.num_rows))
        return fragment_path

    def _finalize_partition(
        self,
        *,
        partition: LogicalPartition,
        schema: pa.Schema,
        input_paths: Sequence[Path],
        max_rows: int,
        output_root: Path,
        operation_root: Path,
    ) -> tuple[StagedParquetObject, ...]:
        if not input_paths:
            return ()
        row_groups = self._externally_sorted_slices(
            input_paths=input_paths,
            schema=schema,
            schema_name=partition.schema_version,
            max_rows=max_rows,
            operation_root=operation_root,
        )
        objects: list[StagedParquetObject] = []
        final_directory = output_root / Path(partition.relative_directory.as_posix())
        final_directory.mkdir(parents=True, exist_ok=True)
        for ordinal, table in enumerate(row_groups):
            temporary_path = final_directory / f".slice-{uuid4().hex}.parquet"
            _write_pinned_parquet(table, temporary_path, row_group_size=max_rows)
            checksum = _sha256_file(temporary_path)
            byte_size = temporary_path.stat().st_size
            final_name = f"sha256={checksum}.parquet"
            final_path = final_directory / final_name
            if final_path.exists():
                if _sha256_file(final_path) != checksum:
                    raise ParquetWriteError("existing Parquet checksum path contains different bytes")
                temporary_path.unlink()
            else:
                temporary_path.replace(final_path)
            objects.append(
                StagedParquetObject(
                    partition=partition,
                    path=final_path,
                    relative_uri=(partition.relative_directory / final_name).as_posix(),
                    checksum=checksum,
                    row_count=table.num_rows,
                    byte_size=byte_size,
                    ordinal=ordinal,
                )
            )
        return tuple(objects)

    @staticmethod
    def _externally_sorted_slices(
        *,
        input_paths: Sequence[Path],
        schema: pa.Schema,
        schema_name: str,
        max_rows: int,
        operation_root: Path,
    ) -> Iterator[pa.Table]:
        """Use DuckDB's spill-capable sort then yield exact fixed row slices."""
        sort_fields = _PARTITION_SORT_FIELDS[schema_name]
        quoted_columns = ", ".join(_quote_identifier(field.name) for field in schema)
        quoted_sort_fields = ", ".join(_quote_identifier(name) for name in sort_fields)
        locations = ", ".join(_quote_sql_literal(str(path)) for path in input_paths)
        sql = (
            f"SELECT {quoted_columns} FROM read_parquet([{locations}], union_by_name = false) "
            f"ORDER BY {quoted_sort_fields}"
        )
        temp_directory = operation_root / "duckdb-spill"
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(f"SET temp_directory = {_quote_sql_literal(str(temp_directory))}")
            connection.execute("SET threads TO 1")
            reader = connection.execute(sql).to_arrow_reader(batch_size=max_rows)
            pending: list[pa.RecordBatch] = []
            pending_rows = 0
            for batch in reader:
                batch = _batch_with_schema(batch, schema)
                offset = 0
                while offset < batch.num_rows:
                    take = min(max_rows - pending_rows, batch.num_rows - offset)
                    pending.append(batch.slice(offset, take))
                    pending_rows += take
                    offset += take
                    if pending_rows == max_rows:
                        yield _table_from_batches(pending, schema)
                        pending = []
                        pending_rows = 0
            if pending_rows:
                yield _table_from_batches(pending, schema)
        except Exception as error:
            if isinstance(error, ParquetStoreError):
                raise
            raise ParquetWriteError(f"external sort or canonical Parquet write failed: {error}") from error
        finally:
            connection.close()

    def _output_root(self, staging: Path | str | None) -> Path:
        root = Path(staging).expanduser() if staging is not None else self._root
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _operation_root(output_root: Path) -> Path:
        path = output_root / ".parquet-sort" / uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _resolve_scan_sources(
        self,
        refs: Sequence[StagedParquetObject | ContentAddressedObjectRef | Path | str],
    ) -> tuple[_ScanSource, ...]:
        if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
            raise TypeError("refs must be a sequence of Parquet references")
        if not refs:
            raise ParquetScanError("scan requires at least one Parquet reference")
        sources = tuple(_scan_source(ref, self._root) for ref in refs)
        if any(not source.path.is_file() for source in sources):
            missing = next(source.path for source in sources if not source.path.is_file())
            raise ParquetScanError(f"referenced Parquet object does not exist: {missing}")
        return tuple(sorted(sources, key=_ScanSource.sort_key))


@dataclass(frozen=True, slots=True)
class _ScanSource:
    path: Path
    schema_name: str
    partition: LogicalPartition | None
    ordinal: int | None

    def sort_key(self) -> tuple[str, str, int, int, str]:
        if self.partition is None:
            return ("", "", -1, self.ordinal if self.ordinal is not None else -1, str(self.path))
        return (
            self.partition.object_kind.value,
            self.partition.symbol,
            self.partition.session_year,
            self.ordinal if self.ordinal is not None else -1,
            str(self.path),
        )


def _partition_component(name: str, value: str | None) -> str:
    if not isinstance(value, str) or not value or _PARTITION_COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use a safe non-empty partition component")
    return value


def _require_date(name: str, value: date) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a calendar date")
    return value


def _bounded_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_WRITE_CHUNK_SIZE:
        raise ValueError(f"write_chunk_size must be between 1 and {MAX_WRITE_CHUNK_SIZE}")
    return value


def _bounded_scan_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_WRITE_CHUNK_SIZE:
        raise ValueError(f"scan batch_size must be between 1 and {MAX_WRITE_CHUNK_SIZE}")
    return value


def _resolve_schema(schema: pa.Schema | str) -> tuple[str, pa.Schema]:
    if isinstance(schema, str):
        return schema, schema_for(schema)
    if not isinstance(schema, pa.Schema):
        raise TypeError("schema must be a registered schema name or pyarrow.Schema")
    for schema_name, candidate in SCHEMAS.items():
        if schema.equals(candidate, check_metadata=True):
            return schema_name, candidate
    raise ParquetWriteError("schema must exactly match a registered canonical schema")


def _validate_input_schema(schema: pa.Schema, expected: pa.Schema) -> None:
    if not schema.equals(expected, check_metadata=True):
        raise ParquetWriteError("input table schema must exactly match the canonical schema")


def _iter_canonical_tables(
    rows: ParquetInput,
    schema_name: str,
    schema: pa.Schema,
    max_rows: int,
) -> Iterator[pa.Table]:
    """Canonicalize only bounded source batches before external sort finalization."""
    if isinstance(rows, pa.Table):
        _validate_input_schema(rows.schema, schema)
        for batch in rows.to_batches(max_chunksize=max_rows):
            if batch.num_rows:
                yield canonical_table(schema_name, batch.to_pylist())
        return
    if isinstance(rows, pa.RecordBatchReader):
        _validate_input_schema(rows.schema, schema)
        for batch in rows:
            yield from _canonicalize_record_batch(batch, schema_name, max_rows)
        return

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise TypeError("rows must be an Arrow table/reader or iterable of canonical rows")
    pending: list[SchemaRow] = []
    for item in rows:
        if isinstance(item, pa.RecordBatch):
            if pending:
                yield canonical_table(schema_name, pending)
                pending = []
            yield from _canonicalize_record_batch(item, schema_name, max_rows, expected_schema=schema)
            continue
        pending.append(_input_row(item, schema_name))
        if len(pending) == max_rows:
            yield canonical_table(schema_name, pending)
            pending = []
    if pending:
        yield canonical_table(schema_name, pending)


def _canonicalize_record_batch(
    batch: pa.RecordBatch,
    schema_name: str,
    max_rows: int,
    *,
    expected_schema: pa.Schema | None = None,
) -> Iterator[pa.Table]:
    if expected_schema is not None:
        _validate_input_schema(batch.schema, expected_schema)
    for offset in range(0, batch.num_rows, max_rows):
        portion = batch.slice(offset, max_rows)
        if portion.num_rows:
            yield canonical_table(schema_name, portion.to_pylist())


def _input_row(item: object, schema_name: str) -> SchemaRow:
    if isinstance(item, Mapping):
        return cast(SchemaRow, item)
    if schema_name == RAW_V1 and isinstance(item, ProviderRecord):
        return cast(SchemaRow, raw_records_to_table((item,)).to_pylist()[0])
    if schema_name == DAILY_BAR_V1 and isinstance(item, DailyBarCandidate):
        return cast(SchemaRow, daily_bars_to_table((item,)).to_pylist()[0])
    raise TypeError(f"{schema_name} rows must be mappings or their matching domain records")


def _partition_for_row(schema_name: str, row: Mapping[str, object]) -> LogicalPartition:
    try:
        symbol = cast(str, row["symbol"])
        if schema_name == RAW_V1:
            provider_date = cast(date, row["provider_date"])
            return LogicalPartition.raw(
                provider=cast(str, row["provider"]),
                symbol=symbol,
                year=provider_date.year,
            )
        if schema_name == DAILY_BAR_V1:
            session = cast(date, row["session"])
            return LogicalPartition.normalized(symbol=symbol, year=session.year)
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ParquetWriteError(f"cannot derive a logical partition from {schema_name} row") from error
    raise ParquetWriteError(f"unsupported partitioned schema: {schema_name}")


def _write_pinned_parquet(table: pa.Table, path: Path, *, row_group_size: int) -> None:
    if table.num_rows <= 0:
        raise ParquetWriteError("Parquet slices must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(
            table,
            path,
            row_group_size=row_group_size,
            **dict(PARQUET_WRITE_OPTIONS),
        )
    except (OSError, pa.ArrowException) as error:
        raise ParquetWriteError(f"cannot write canonical Parquet bytes: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1_048_576):
            digest.update(data)
    return digest.hexdigest()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _batch_with_schema(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    table = pa.Table.from_batches((batch,))
    physical_schema = schema.remove_metadata()
    if not table.schema.remove_metadata().equals(physical_schema):
        table = table.cast(physical_schema)
    table = table.replace_schema_metadata(schema.metadata)
    return table.to_batches()[0]


def _table_from_batches(batches: Sequence[pa.RecordBatch], schema: pa.Schema) -> pa.Table:
    table = pa.Table.from_batches(batches)
    physical_schema = schema.remove_metadata()
    if not table.schema.remove_metadata().equals(physical_schema):
        table = table.cast(physical_schema)
    return table.replace_schema_metadata(schema.metadata)


def _required_columns(columns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError("columns must be a non-empty sequence of column names")
    names = tuple(columns)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("columns must contain at least one non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("columns must not repeat names")
    return names


def _validate_projection(columns: tuple[str, ...], schema: pa.Schema) -> None:
    unknown = sorted(set(columns) - set(schema.names))
    if unknown:
        raise ParquetScanError(f"scan requested unknown columns: {', '.join(unknown)}")


def _merge_predicates(
    predicate: ScanPredicate | ds.Expression | None,
    *,
    symbols: Sequence[str] | None,
    years: Sequence[int] | None,
    sessions: Sequence[date] | None,
    session_start: date | None,
    session_end: date | None,
) -> ScanPredicate:
    if predicate is None:
        base = ScanPredicate()
    elif isinstance(predicate, ScanPredicate):
        base = predicate
    elif isinstance(predicate, ds.Expression):
        base = ScanPredicate(expression=predicate)
    else:
        raise TypeError("predicate must be ScanPredicate, dataset expression, or None")
    explicit = ScanPredicate.from_values(
        symbols=symbols,
        years=years,
        sessions=sessions,
        session_start=session_start,
        session_end=session_end,
    )
    return ScanPredicate(
        symbols=_intersect_optional(base.symbols, explicit.symbols),
        years=_intersect_optional(base.years, explicit.years),
        sessions=_intersect_optional(base.sessions, explicit.sessions),
        session_start=_max_optional_date(base.session_start, explicit.session_start),
        session_end=_min_optional_date(base.session_end, explicit.session_end),
        expression=base.expression,
    )


def _intersect_optional(
    first: tuple[object, ...] | None,
    second: tuple[object, ...] | None,
) -> tuple[object, ...] | None:
    if first is None:
        return second
    if second is None:
        return first
    intersection = tuple(item for item in first if item in set(second))
    if not intersection:
        raise ParquetScanError("combined scan predicates have no matching partition values")
    return intersection


def _max_optional_date(first: date | None, second: date | None) -> date | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _min_optional_date(first: date | None, second: date | None) -> date | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _scan_source(
    ref: StagedParquetObject | ContentAddressedObjectRef | Path | str,
    root: Path,
) -> _ScanSource:
    if isinstance(ref, StagedParquetObject):
        return _ScanSource(
            path=ref.path,
            schema_name=ref.partition.schema_version,
            partition=ref.partition,
            ordinal=ref.ordinal,
        )
    if isinstance(ref, ContentAddressedObjectRef):
        partition = _partition_from_ref(ref)
        return _ScanSource(
            path=root / Path(ref.relative_uri),
            schema_name=ref.schema_version,
            partition=partition,
            ordinal=None,
        )
    if isinstance(ref, (Path, str)):
        path = Path(ref)
        schema_name = _schema_name_from_file(path)
        return _ScanSource(path=path, schema_name=schema_name, partition=None, ordinal=None)
    raise TypeError("Parquet scan refs must be staged objects, manifest refs, or paths")


def _partition_from_ref(ref: ContentAddressedObjectRef) -> LogicalPartition | None:
    if ref.object_kind is ObjectKind.RAW:
        provider_match = re.search(r"(?:^|/)provider=([^/]+)(?:/|$)", ref.relative_uri)
        if provider_match is None or ref.symbol is None or ref.session_year is None:
            return None
        return LogicalPartition.raw(
            provider=provider_match.group(1), symbol=ref.symbol, year=ref.session_year
        )
    if ref.object_kind is ObjectKind.NORMALIZED:
        if ref.symbol is None or ref.session_year is None:
            return None
        return LogicalPartition.normalized(symbol=ref.symbol, year=ref.session_year)
    return None


def _schema_name_from_file(path: Path) -> str:
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except (OSError, pa.ArrowException) as error:
        raise ParquetScanError(f"cannot inspect Parquet schema at {path}: {error}") from error
    for schema_name, candidate in SCHEMAS.items():
        if schema.equals(candidate, check_metadata=True):
            return schema_name
    raise ParquetScanError("Parquet file does not use a registered canonical schema")


def _resolve_schema_from_sources(sources: Sequence[_ScanSource]) -> tuple[str, pa.Schema]:
    schema_names = {source.schema_name for source in sources}
    if len(schema_names) != 1:
        raise ParquetScanError("a scan must not combine different schema versions")
    schema_name = schema_names.pop()
    return schema_name, schema_for(schema_name)


def _filter_sources_for_partitions(
    sources: Sequence[_ScanSource],
    predicate: ScanPredicate,
) -> tuple[_ScanSource, ...]:
    selected: list[_ScanSource] = []
    for source in sources:
        partition = source.partition
        if partition is not None:
            if predicate.symbols is not None and partition.symbol not in predicate.symbols:
                continue
            if predicate.years is not None and partition.session_year not in predicate.years:
                continue
        selected.append(source)
    return tuple(selected)


def _dataset_filter(schema_name: str, predicate: ScanPredicate) -> ds.Expression | None:
    expressions: list[ds.Expression] = []
    if predicate.symbols is not None:
        expressions.append(ds.field("symbol").isin(list(predicate.symbols)))
    session_column = "provider_date" if schema_name == RAW_V1 else "session"
    if predicate.years is not None:
        year_terms = [
            (ds.field(session_column) >= date(year, 1, 1))
            & (ds.field(session_column) <= date(year, 12, 31))
            for year in predicate.years
        ]
        expressions.append(_or_all(year_terms))
    if predicate.sessions is not None:
        expressions.append(ds.field(session_column).isin(list(predicate.sessions)))
    if predicate.session_start is not None:
        expressions.append(ds.field(session_column) >= predicate.session_start)
    if predicate.session_end is not None:
        expressions.append(ds.field(session_column) <= predicate.session_end)
    if predicate.expression is not None:
        expressions.append(predicate.expression)
    if not expressions:
        return None
    result = expressions[0]
    for expression in expressions[1:]:
        result = result & expression
    return result


def _or_all(expressions: Sequence[ds.Expression]) -> ds.Expression:
    if not expressions:
        raise ValueError("at least one expression is required")
    result = expressions[0]
    for expression in expressions[1:]:
        result = result | expression
    return result


ParquetStoreAdapter = ParquetStore

__all__ = [
    "DEFAULT_SCAN_BATCH_SIZE",
    "DEFAULT_WRITE_CHUNK_SIZE",
    "DatasetFactory",
    "LogicalPartition",
    "MAX_WRITE_CHUNK_SIZE",
    "PARQUET_MEDIA_TYPE",
    "PARQUET_WRITE_OPTIONS",
    "ParquetScanError",
    "ParquetStore",
    "ParquetStoreAdapter",
    "ParquetStoreError",
    "ParquetWriteError",
    "ScanPlan",
    "ScanPredicate",
    "StagedObject",
    "StagedParquetObject",
]
