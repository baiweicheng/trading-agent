"""Derived Zipline bundle cache and snapshot-projection integration tests.

The fixture is built through the local ingestion/publication path.  No provider
or Zipline network activity is used: the adapter receives published Parquet
bytes from the filesystem CAS and writes through a deterministic local writer
seam.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.application.snapshots import (
    SnapshotManager,
    SnapshotManifestAssembler,
)
from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ProviderRequest
from quant_research_platform.domain.normalization import (
    CausalForwardAdjustmentV1,
    Normalizer,
)
from quant_research_platform.domain.validation import ValidationService
from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    SnapshotPublicationCandidate,
)
from quant_research_platform.infrastructure.parquet_store import ParquetStore
from quant_research_platform.infrastructure.zipline_bundle import (
    ZiplineAsset,
    ZiplineBundleAdapter,
    ZiplineDailyBar,
    ZiplineDividend,
    ZiplineSplit,
)
from tests.integration.test_snapshot_ingestion_faults import (
    ALL_SYMBOLS,
    SESSIONS,
    SYMBOLS,
    FixtureCalendar,
    OfflineYFinanceFixture,
)


class _BundleCalendar(FixtureCalendar):
    """Fixture XNYS calendar with the auto-close seam required by the adapter."""

    def next_session(self, session: date) -> date:
        for candidate in SESSIONS:
            if candidate > session:
                return candidate
        raise ValueError("fixture has no next session")


@dataclass(frozen=True)
class _ScanCall:
    """The bounded projection requested from one published Parquet scan."""

    references: tuple[str, ...]
    columns: tuple[str, ...]
    symbols: tuple[str, ...]
    session_start: date | None
    session_end: date | None


@dataclass
class _PublishedParquetScanSpy:
    """Read real published Parquet while recording adapter scan boundaries."""

    store: FilesystemStore
    calls: list[_ScanCall] = field(default_factory=list)

    def scan(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        *,
        predicate: object | None = None,
    ) -> Iterable[Mapping[str, object]]:
        symbols = tuple(getattr(predicate, "symbols", ()) or ())
        session_start = getattr(predicate, "session_start", None)
        session_end = getattr(predicate, "session_end", None)
        self.calls.append(
            _ScanCall(
                references=tuple(reference.relative_uri for reference in references),
                columns=tuple(columns),
                symbols=symbols,
                session_start=session_start,
                session_end=session_end,
            )
        )

        selected_rows: list[Mapping[str, object]] = []
        for reference in references:
            parquet = pq.ParquetFile(
                pa.BufferReader(self.store.read_object(reference.relative_uri))
            )
            for batch in parquet.iter_batches(
                columns=list(columns), batch_size=2, use_threads=False
            ):
                for row in batch.to_pylist():
                    row_symbol = row["symbol"]
                    row_session = row["session"]
                    if symbols and row_symbol not in symbols:
                        continue
                    if session_start is not None and row_session < session_start:
                        continue
                    if session_end is not None and row_session > session_end:
                        continue
                    selected_rows.append(row)

        for row in sorted(
            selected_rows, key=lambda item: (item["symbol"], item["session"])
        ):
            yield row


@dataclass
class _DeterministicWriter:
    """Small writer seam that preserves the adapter's projected inputs."""

    calls: int = 0
    projections: list[
        tuple[
            tuple[ZiplineAsset, ...],
            tuple[ZiplineSplit, ...],
            tuple[ZiplineDividend, ...],
        ]
    ] = field(default_factory=list)

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[ZiplineAsset],
        daily_rows: Iterable[tuple[int, Iterable[ZiplineDailyBar]]],
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        del bundle_name, bundle_timestamp, start_session, end_session, calendar
        self.calls += 1
        materialized_rows = tuple((sid, tuple(rows)) for sid, rows in daily_rows)
        self.projections.append((tuple(assets), tuple(splits), tuple(dividends)))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "assets.json").write_bytes(
            canonical_json([asset.to_content_dict() for asset in assets])
        )
        (output_dir / "splits.json").write_bytes(
            canonical_json([split.to_content_dict() for split in splits])
        )
        (output_dir / "dividends.json").write_bytes(
            canonical_json([dividend.to_content_dict() for dividend in dividends])
        )
        with (output_dir / "daily.jsonl").open("wb") as handle:
            for _sid, rows in materialized_rows:
                for row in rows:
                    handle.write(canonical_json(row.to_content_dict()))


def _published_fixture(
    tmp_path: Path,
) -> tuple[_BundleCalendar, FilesystemStore, DuckDBMetadataStore, SnapshotManifest]:
    """Publish one complete local snapshot containing real Parquet objects."""

    calendar = _BundleCalendar()
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    requested_range = DateRange(SESSIONS[0], SESSIONS[-2])
    provider = OfflineYFinanceFixture(calendar)
    request = ProviderRequest(
        ALL_SYMBOLS,
        requested_range.start,
        requested_range.end,
        provider="yfinance",
    )
    records = provider.records_for(request)
    policy = CausalForwardAdjustmentV1()
    candidates = tuple(Normalizer(policy).normalize(records, calendar))
    validation = ValidationService(calendar=calendar, benchmark_symbol="SPY").validate(
        candidates,
        {
            symbol: calendar.sessions(SESSIONS[0], SESSIONS[-2])
            for symbol in ALL_SYMBOLS
        },
        10,
        requested_range=requested_range,
        benchmark_symbol="SPY",
        calendar=calendar,
    )
    parquet_store = ParquetStore(tmp_path / "parquet", write_chunk_size=2)
    staged = parquet_store.write_normalized(
        validation.accepted_rows,
        staging=tmp_path / "parquet-staging",
    )
    published_staged = tuple(
        replace(item, relative_uri=f"objects/{item.relative_uri}") for item in staged
    )
    references = tuple(item.object_ref for item in published_staged)
    manifest = SnapshotManifestAssembler.assemble(
        provider="yfinance",
        requested_range=requested_range,
        covered_range=validation.report.summary.covered_range,
        configured_universe=SYMBOLS,
        benchmark_symbol="SPY",
        calendar=CalendarIdentity(
            calendar.name,
            calendar.version,
            calendar.schedule_checksum(requested_range.start, requested_range.end),
        ),
        configuration_checksum="c" * 64,
        objects=references,
        validation=validation,
        limitation_disclosure=LimitationDisclosure.current(),
        created_at=datetime(2024, 1, 8, 22, tzinfo=UTC),
    )
    report_bytes = canonical_json(validation.report.to_content_dict())
    publication = store.publish_snapshot(
        SnapshotPublicationCandidate(
            manifest,
            staged_objects={
                reference.relative_uri: staged_object.path
                for reference, staged_object in zip(
                    references, published_staged, strict=True
                )
            },
            validation_report=report_bytes,
            symbol_statuses=validation.report.per_symbol,
        ),
        operation_id="bundle-fixture",
    )
    return calendar, store, metadata, publication.manifest


def _normalized_references(
    manifest: SnapshotManifest,
) -> tuple[ContentAddressedObjectRef, ...]:
    return tuple(
        reference
        for reference in manifest.content_identity.objects
        if reference.object_kind is ObjectKind.NORMALIZED
    )


def _adapter(
    *,
    manager: SnapshotManager,
    source: _PublishedParquetScanSpy,
    calendar: FixtureCalendar,
    root: Path,
    writer: _DeterministicWriter,
) -> ZiplineBundleAdapter:
    return ZiplineBundleAdapter(
        snapshot_manager=manager,
        data_source=source,
        calendar=calendar,
        zipline_root=root,
        writer=writer,
    )


@pytest.mark.integration
def test_verified_copy_reuses_bundle_identity_and_reads_only_configured_projection(
    tmp_path: Path,
) -> None:
    """A verified copy has the same derived identity and bounded raw projection."""

    calendar, store, metadata, manifest = _published_fixture(tmp_path)
    try:
        manager = SnapshotManager(storage=store, metadata=metadata)
        source = _PublishedParquetScanSpy(store)
        writer = _DeterministicWriter()
        first = _adapter(
            manager=manager,
            source=source,
            calendar=calendar,
            root=tmp_path / "derived",
            writer=writer,
        ).materialize(manifest.snapshot_id)

        assert isinstance(first, Ok)
        assert writer.calls == 1
        normalized_refs = _normalized_references(manifest)
        assert normalized_refs
        expected_refs = tuple(reference.relative_uri for reference in normalized_refs)
        assert source.calls
        assert all(call.references == expected_refs for call in source.calls)
        assert all(
            call.session_start == SESSIONS[0] and call.session_end == SESSIONS[-2]
            for call in source.calls
        )
        action_calls = [
            call
            for call in source.calls
            if call.columns == ("symbol", "session", "dividend", "split_ratio")
        ]
        daily_calls = [
            call
            for call in source.calls
            if call.columns
            == (
                "symbol",
                "session",
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "raw_volume",
            )
        ]
        assert len(action_calls) == 1
        assert action_calls[0].symbols == tuple(sorted(ALL_SYMBOLS))
        assert {call.symbols for call in daily_calls} == {
            (symbol,) for symbol in sorted(ALL_SYMBOLS)
        }
        assert len(daily_calls) == len(ALL_SYMBOLS)
        assert all(
            "adjusted" not in column for call in source.calls for column in call.columns
        )
        assert first.value.bundle_name != "latest"

        relocated = tmp_path / "relocated"
        relocated.mkdir()
        shutil.copytree(store.objects_root, relocated / "objects")
        shutil.copytree(store.snapshots_root, relocated / "snapshots")
        relocated_store = FilesystemStore(relocated)
        relocated_manager = SnapshotManager(root=relocated)
        relocated_source = _PublishedParquetScanSpy(relocated_store)
        relocated_writer = _DeterministicWriter()
        copied = _adapter(
            manager=relocated_manager,
            source=relocated_source,
            calendar=calendar,
            root=relocated / "derived",
            writer=relocated_writer,
        ).materialize(manifest.snapshot_id)

        assert isinstance(copied, Ok)
        assert copied.value.bundle_checksum == first.value.bundle_checksum
        assert copied.value.bundle_name == first.value.bundle_name
        assert copied.value.bundle_timestamp == first.value.bundle_timestamp
        assert copied.value.cache_path != first.value.cache_path
        assert relocated_writer.calls == 1
    finally:
        metadata.close()


@pytest.mark.integration
def test_invalid_derived_cache_is_discarded_and_rebuilt_from_verified_snapshot(
    tmp_path: Path,
) -> None:
    """Cache corruption never becomes authoritative and triggers one rebuild."""

    calendar, store, metadata, manifest = _published_fixture(tmp_path)
    try:
        manager = SnapshotManager(storage=store, metadata=metadata)
        source = _PublishedParquetScanSpy(store)
        writer = _DeterministicWriter()
        adapter = _adapter(
            manager=manager,
            source=source,
            calendar=calendar,
            root=tmp_path / "derived",
            writer=writer,
        )
        first = adapter.materialize(manifest.snapshot_id)
        assert isinstance(first, Ok)
        daily_path = first.value.cache_path / "daily.jsonl"
        original_daily_bytes = daily_path.read_bytes()
        daily_path.write_bytes(original_daily_bytes + b"corrupt cache bytes")

        rebuilt = adapter.materialize(manifest.snapshot_id)

        assert isinstance(rebuilt, Ok)
        assert writer.calls == 2
        assert rebuilt.value.bundle_checksum == first.value.bundle_checksum
        assert daily_path.read_bytes() == original_daily_bytes
        assert len(source.calls) == 8
        assert all(call.session_start == SESSIONS[0] for call in source.calls)
        assert all(call.session_end == SESSIONS[-2] for call in source.calls)
    finally:
        metadata.close()


@pytest.mark.integration
def test_corrupt_source_snapshot_is_rejected_before_bundle_cache_or_projection(
    tmp_path: Path,
) -> None:
    """A derived bundle cannot hide corruption in its authoritative snapshot."""

    calendar, store, metadata, manifest = _published_fixture(tmp_path)
    try:
        reference = _normalized_references(manifest)[0]
        source_path = store.objects_root / Path(reference.relative_uri).relative_to(
            "objects"
        )
        source_path.write_bytes(b"corrupt authoritative parquet bytes")

        manager = SnapshotManager(storage=store, metadata=metadata)
        source = _PublishedParquetScanSpy(store)
        writer = _DeterministicWriter()
        result = _adapter(
            manager=manager,
            source=source,
            calendar=calendar,
            root=tmp_path / "derived",
            writer=writer,
        ).materialize(manifest.snapshot_id)

        assert isinstance(result, Err)
        assert result.errors[0].category.value in {
            "integrity.checksum",
            "storage.io",
        }
        assert writer.calls == 0
        assert source.calls == []
        assert not (tmp_path / "derived" / manifest.snapshot_id).exists()
    finally:
        metadata.close()
