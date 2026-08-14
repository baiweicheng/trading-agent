"""Focused examples for verified snapshot access and append-only guards."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant_research_platform.application.snapshots import (
    LocalPublishedSnapshotStore,
    SnapshotManager,
    SnapshotQuery,
)
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
)
from quant_research_platform.domain.market import (
    DateRange,
    SymbolValidationSummary,
    ValidationReport,
)

_NOW = datetime(2024, 1, 3, 21, tzinfo=UTC)


def _report() -> ValidationReport:
    return ValidationReport(
        per_symbol=(
            SymbolValidationSummary(
                symbol="AAPL",
                accepted_count=1,
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
            ),
            SymbolValidationSummary(
                symbol="SPY",
                accepted_count=1,
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
            ),
        ),
        quarantined_by_reason=(),
        gaps=(),
        calendar_version="exchange_calendars/4.5",
    )


def _manifest_and_bytes(
    *,
    object_bytes: bytes = b"a normalized parquet object",
    configuration_character: str = "c",
    object_name: str = "a",
) -> tuple[Any, bytes, bytes, ContentAddressedObjectRef]:
    report_bytes = canonical_json(_report().to_content_dict())
    report_checksum = sha256_bytes(report_bytes)
    object_checksum = sha256_bytes(object_bytes)
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=object_checksum,
        relative_uri=(
            "objects/normalized/symbol=AAPL/year=2024/"
            f"sha256={object_name}-{object_checksum}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=1,
        byte_size=len(object_bytes),
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    from quant_research_platform.application.snapshots import SnapshotManifestAssembler

    manifest = SnapshotManifestAssembler.assemble(
        provider="yfinance",
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 2)),
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "exchange_calendars/4.5", "d" * 64),
        configuration_checksum=configuration_character * 64,
        objects=(reference,),
        validation=(_report()),
        validation_report_checksum=report_checksum,
        limitation_disclosure=LimitationDisclosure.current(),
        created_at=_NOW,
    )
    return manifest, object_bytes, report_bytes, reference


def _publish(root: Path, manifest: Any, object_bytes: bytes, report_bytes: bytes) -> None:
    object_ref = manifest.content_identity.objects[0]
    object_path = root / object_ref.relative_uri
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(object_bytes)
    report_checksum = manifest.content_identity.validation_report_checksum
    report_path = root / "objects" / "sha256" / report_checksum[:2] / report_checksum
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)
    manifest_path = root / "snapshots" / manifest.snapshot_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json(manifest.to_manifest_dict()))


def test_open_verifies_manifest_objects_and_exposes_inspection_details(tmp_path: Path) -> None:
    manifest, object_bytes, report_bytes, _ = _manifest_and_bytes()
    _publish(tmp_path, manifest, object_bytes, report_bytes)
    manager = SnapshotManager(tmp_path)

    opened = manager.open_verified(manifest.snapshot_id)
    inspected = manager.inspect_snapshot(manifest.snapshot_id)

    assert isinstance(opened, Ok)
    assert opened.value.snapshot_id == manifest.snapshot_id
    assert isinstance(inspected, Ok)
    assert inspected.value.provenance.provider == "yfinance"
    assert inspected.value.validation_summary.accepted_row_count == 2
    assert inspected.value.readiness.comparison_ready
    assert inspected.value.limitation_disclosure.version == "limitation-disclosure/v1"


def test_missing_or_corrupt_object_is_rejected_before_handle_is_returned(
    tmp_path: Path,
) -> None:
    manifest, object_bytes, report_bytes, reference = _manifest_and_bytes()
    _publish(tmp_path, manifest, object_bytes, report_bytes)
    manager = SnapshotManager(tmp_path)

    (tmp_path / reference.relative_uri).unlink()
    missing = manager.open_verified(manifest.snapshot_id)
    assert isinstance(missing, Err)
    assert missing.errors[0].category.value == "integrity.checksum"

    _publish(tmp_path, manifest, object_bytes, report_bytes)
    (tmp_path / reference.relative_uri).write_bytes(b"replacement bytes")
    corrupt = manager.open_verified(manifest.snapshot_id)
    assert isinstance(corrupt, Err)
    assert corrupt.errors[0].category.value == "integrity.checksum"


def test_manifest_directory_id_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, object_bytes, report_bytes, _ = _manifest_and_bytes()
    _publish(tmp_path, manifest, object_bytes, report_bytes)
    manifest_path = tmp_path / "snapshots" / manifest.snapshot_id / "manifest.json"
    document = __import__("json").loads(manifest_path.read_bytes())
    document["snapshot_id"] = "snap_" + "0" * 64
    manifest_path.write_bytes(canonical_json(document))

    result = SnapshotManager(tmp_path).open_verified(manifest.snapshot_id)

    assert isinstance(result, Err)
    assert result.errors[0].category.value == "integrity.checksum"


def test_relocated_published_root_resolves_same_snapshot_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    relocated = tmp_path / "relocated"
    manifest, object_bytes, report_bytes, _ = _manifest_and_bytes()
    _publish(source, manifest, object_bytes, report_bytes)
    import shutil

    shutil.copytree(source, relocated)

    result = SnapshotManager(relocated).open_verified(manifest.snapshot_id)

    assert isinstance(result, Ok)
    assert result.value.snapshot_id == manifest.snapshot_id


def test_list_returns_bounded_summaries_and_mutations_are_rejected(tmp_path: Path) -> None:
    first, first_bytes, first_report, _ = _manifest_and_bytes()
    second, second_bytes, second_report, _ = _manifest_and_bytes(
        object_bytes=b"a different normalized object",
        configuration_character="e",
        object_name="b",
    )
    _publish(tmp_path, first, first_bytes, first_report)
    _publish(tmp_path, second, second_bytes, second_report)
    manager = SnapshotManager(tmp_path)

    page = manager.list_snapshots(SnapshotQuery(page_size=1))
    mutation = manager.replace_manifest(first.snapshot_id, b"replacement")
    object_mutation = manager.replace_object(first.snapshot_id, "objects/x", b"x")

    assert page.errors == ()
    assert len(page.items) == 1
    assert page.total == 2
    assert page.has_next
    assert page.items[0].validation_summary is not None
    assert isinstance(mutation, Err)
    assert isinstance(object_mutation, Err)
    assert "Publish a new Data_Snapshot" in mutation.errors[0].corrective_action
    assert isinstance(manager.open_verified(first.snapshot_id), Ok)


def test_unavailable_index_blocks_open_but_does_not_delete_prior_manifest(
    tmp_path: Path,
) -> None:
    manifest, object_bytes, report_bytes, _ = _manifest_and_bytes()
    _publish(tmp_path, manifest, object_bytes, report_bytes)

    class UnavailableIndex:
        def get_snapshot(self, snapshot_id: str) -> object:
            del snapshot_id
            return type(
                "Record",
                (),
                {
                    "snapshot_id": manifest.snapshot_id,
                    "availability": "unavailable",
                },
            )()

    blocked = SnapshotManager(tmp_path, metadata=UnavailableIndex()).open_verified(
        manifest.snapshot_id
    )
    still_readable = SnapshotManager(tmp_path).open_verified(manifest.snapshot_id)

    assert isinstance(blocked, Err)
    assert blocked.errors[0].category.value == "storage.io"
    assert isinstance(still_readable, Ok)
