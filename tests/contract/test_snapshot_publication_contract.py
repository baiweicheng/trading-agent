"""Atomic snapshot publication and restart-reconciliation contracts."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quant_research_platform.application.snapshots import SnapshotManager
from quant_research_platform.domain.canonical import sha256_bytes
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.infrastructure.duckdb_metadata import (
    DuckDBMetadataStore,
    SnapshotAvailability,
)
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    SnapshotPublicationCandidate,
)

_NOW = datetime(2024, 1, 10, 15, tzinfo=UTC)


def _manifest(object_bytes: bytes, report_bytes: bytes) -> SnapshotManifest:
    object_checksum = sha256_bytes(object_bytes)
    report_checksum = sha256_bytes(report_bytes)
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=object_checksum,
        relative_uri=(
            "objects/normalized/symbol=AAPL/year=2024/"
            f"sha256={object_checksum}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=1,
        byte_size=len(object_bytes),
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    return SnapshotManifest(
        content_identity=SnapshotContentIdentity(
            provider="fixture",
            requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            configured_universe=("AAPL",),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "fixture", "a" * 64),
            configuration_checksum="b" * 64,
            objects=(reference,),
            validation_report_checksum=report_checksum,
            validation_summary=ValidationSummary(
                accepted_row_count=1,
                quarantined_row_count=0,
                collapsed_duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
            ),
            limitation_disclosure=LimitationDisclosure.current(),
        ),
        operational_metadata=OperationalMetadata(created_at=_NOW),
    )


def _candidate(
    manifest: SnapshotManifest,
    object_bytes: bytes,
    report_bytes: bytes,
) -> SnapshotPublicationCandidate:
    reference = manifest.content_identity.objects[0]
    return SnapshotPublicationCandidate(
        manifest,
        staged_objects={reference.relative_uri: object_bytes},
        validation_report=report_bytes,
    )


def test_publication_indexes_only_after_complete_rename_and_opens_verified_content(
    tmp_path: Path,
) -> None:
    object_bytes = b"canonical normalized object"
    report_bytes = b"canonical validation report"
    manifest = _manifest(object_bytes, report_bytes)
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)

    result = store.publish_snapshot(_candidate(manifest, object_bytes, report_bytes))

    assert result.snapshot_id == manifest.snapshot_id
    assert result.indexed
    assert (store.snapshots_root / manifest.snapshot_id / "manifest.json").is_file()
    assert (
        metadata.get_snapshot(manifest.snapshot_id).availability
        is SnapshotAvailability.AVAILABLE
    )
    opened = SnapshotManager(
        storage=store, metadata=metadata
    ).open_verified(manifest.snapshot_id)
    assert isinstance(opened, Ok)
    metadata.close()


def test_restart_reconciliation_indexes_complete_publication_left_after_rename_failure(
    tmp_path: Path,
) -> None:
    object_bytes = b"object survives rename failure"
    report_bytes = b"report survives rename failure"
    manifest = _manifest(object_bytes, report_bytes)
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    fired = False

    def fail_after_rename(point: str) -> None:
        nonlocal fired
        if point == "after_publication_rename" and not fired:
            fired = True
            raise RuntimeError("injected after rename")

    store = FilesystemStore(
        tmp_path / "store",
        metadata=metadata,
        failure_injector=fail_after_rename,
    )
    with pytest.raises(RuntimeError, match="after rename"):
        store.publish_snapshot(_candidate(manifest, object_bytes, report_bytes))
    assert metadata.list_snapshots() == ()
    assert (store.snapshots_root / manifest.snapshot_id / "manifest.json").is_file()

    restarted = FilesystemStore(tmp_path / "store", metadata=metadata)
    report = restarted.reconcile()
    assert report.indexed_snapshot_ids == (manifest.snapshot_id,)
    assert (
        metadata.get_snapshot(manifest.snapshot_id).availability
        is SnapshotAvailability.AVAILABLE
    )
    metadata.close()


def test_reconciliation_marks_indexed_snapshot_unavailable_when_directory_disappears(
    tmp_path: Path,
) -> None:
    object_bytes = b"object whose publication is removed"
    report_bytes = b"report whose publication is removed"
    manifest = _manifest(object_bytes, report_bytes)
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store = FilesystemStore(tmp_path / "store", metadata=metadata)
    store.publish_snapshot(_candidate(manifest, object_bytes, report_bytes))

    shutil.rmtree(store.snapshots_root / manifest.snapshot_id)
    report = store.reconcile()

    assert report.unavailable_snapshot_ids == (manifest.snapshot_id,)
    assert (
        metadata.get_snapshot(manifest.snapshot_id).availability
        is SnapshotAvailability.UNAVAILABLE
    )
    metadata.close()
