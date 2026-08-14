# ruff: noqa: E501
"""Property tests for immutable, confluent snapshot content identity."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.snapshots import (
    SnapshotManifestAssembler,
    SnapshotManager,
)
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotManifest,
    SnapshotSchemaVersions,
)
from quant_research_platform.domain.market import (
    DateRange,
    SymbolValidationSummary,
    ValidationReport,
)
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    SnapshotPublicationCandidate,
)


_ROW_SESSIONS = (
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
)
_SYMBOL_POOL = ("AAPL", "MSFT", "PG", "XOM")
_INTERRUPTION_POINTS = (
    "before_snapshot_object_checksum",
    "after_snapshot_object_checksum",
    "before_snapshot_object_promotion",
    "after_snapshot_object_promotion",
    "before_validation_checksum",
    "after_validation_checksum",
    "before_validation_promotion",
    "after_validation_promotion",
    "before_publication_write",
    "after_publication_write",
    "before_publication_directory_fsync",
    "after_publication_directory_fsync",
    "before_publication_rename",
    "after_publication_rename",
    "after_publication_parent_fsync",
)
_MUTATIONS = (
    "row",
    "provider",
    "configuration",
    "calendar",
    "schema",
    "validation",
)

# A row is deliberately a small logical scientific record.  The canonical
# writer below is independent from SnapshotManifestAssembler and has no batch,
# retry, timestamp, path, or parent-lineage fields.
ScientificRow = tuple[str, str, int]


@dataclass(frozen=True)
class SnapshotInputs:
    """Scientific inputs from which a reference publisher builds one manifest."""

    rows: tuple[ScientificRow, ...]
    requested_range: DateRange
    configured_universe: tuple[str, ...]
    provider: str
    configuration_checksum: str
    calendar_checksum: str
    validation: ValidationReport
    schema_versions: SnapshotSchemaVersions = SnapshotSchemaVersions()

    @property
    def symbols(self) -> tuple[str, ...]:
        return (*self.configured_universe, "SPY")


@dataclass(frozen=True)
class SnapshotCase:
    """Generated parent/merged content and operational publication variations."""

    parent: SnapshotInputs
    merged: SnapshotInputs
    permuted_rows: tuple[ScientificRow, ...]
    created_at: tuple[datetime, datetime, datetime]
    operation_suffix: str
    interruption_point: str
    mutation: str


def _digest(label: str, value: object) -> str:
    """Produce a valid scientific checksum from generated fixture values."""

    return sha256_bytes(canonical_json({"label": label, "value": value}))


def _row_document(row: ScientificRow) -> dict[str, object]:
    symbol, session, close = row
    return {"symbol": symbol, "session": session, "close": close}


def _canonical_partition_bytes(rows: Iterable[ScientificRow]) -> bytes:
    """Canonicalize one logical partition independently of input order."""

    documents = [_row_document(row) for row in rows]
    return canonical_json(sorted(documents, key=canonical_json))


def _object_references(
    inputs: SnapshotInputs,
    rows: Iterable[ScientificRow] | None = None,
) -> tuple[ContentAddressedObjectRef, ...]:
    """Build expected content-addressed refs from sorted logical partition rows."""

    source_rows = tuple(inputs.rows if rows is None else rows)
    references: list[ContentAddressedObjectRef] = []
    for symbol in inputs.symbols:
        partition_rows = tuple(row for row in source_rows if row[0] == symbol)
        payload = _canonical_partition_bytes(partition_rows)
        checksum = sha256_bytes(payload)
        references.append(
            ContentAddressedObjectRef(
                object_kind=ObjectKind.NORMALIZED,
                checksum=checksum,
                relative_uri=(
                    "objects/normalized/"
                    f"symbol={symbol}/year=2024/sha256={checksum}.json"
                ),
                schema_version=inputs.schema_versions.normalized_schema_version,
                row_count=len(partition_rows),
                byte_size=len(payload),
                symbol=symbol,
                session_year=2024,
                media_type="application/json",
            )
        )
    return tuple(sorted(references, key=ContentAddressedObjectRef.sort_key))


def _object_bytes(
    inputs: SnapshotInputs,
    rows: Iterable[ScientificRow] | None = None,
) -> Mapping[str, bytes]:
    """Return the bytes keyed by the exact logical URI in each object ref."""

    source_rows = tuple(inputs.rows if rows is None else rows)
    result: dict[str, bytes] = {}
    for reference in _object_references(inputs, source_rows):
        partition_rows = tuple(
            row for row in source_rows if row[0] == reference.symbol
        )
        result[reference.relative_uri] = _canonical_partition_bytes(partition_rows)
    return result


def _validation_report(
    rows: tuple[ScientificRow, ...],
    symbols: tuple[str, ...],
    *,
    calendar_version: str = "fixture-xnys-v1",
    quarantined_by_reason: tuple[tuple[str, int], ...] = (),
) -> ValidationReport:
    """Create a coherent, gap-free validation report for generated rows."""

    summaries: list[SymbolValidationSummary] = []
    for symbol in symbols:
        symbol_rows = tuple(row for row in rows if row[0] == symbol)
        sessions = tuple(sorted(date.fromisoformat(row[1]) for row in symbol_rows))
        covered_range = DateRange(sessions[0], sessions[-1])
        summaries.append(
            SymbolValidationSummary(
                symbol=symbol,
                accepted_count=len(symbol_rows),
                quarantined_count=0,
                duplicate_count=0,
                gap_count=0,
                covered_range=covered_range,
            )
        )
    return ValidationReport(
        per_symbol=tuple(summaries),
        quarantined_by_reason=quarantined_by_reason,
        gaps=(),
        calendar_version=calendar_version,
    )


def _reference_content_identity(
    inputs: SnapshotInputs,
) -> dict[str, object]:
    """Independently project scientific identity into canonical JSON fields."""

    disclosure = LimitationDisclosure.current()
    disclosure_checksum = sha256_bytes(
        canonical_json(
            {"version": disclosure.version, "lines": list(disclosure.lines())}
        )
    )
    report_checksum = sha256_bytes(canonical_json(inputs.validation.to_content_dict()))
    return {
        "schema_versions": inputs.schema_versions.to_content_dict(),
        "provider": inputs.provider,
        "requested_range": inputs.requested_range.to_content_dict(),
        "covered_range": inputs.validation.summary.covered_range.to_content_dict(),
        "configured_universe": list(inputs.configured_universe),
        "benchmark_symbol": "SPY",
        "calendar": CalendarIdentity(
            "XNYS", "fixture-xnys-v1", inputs.calendar_checksum
        ).to_content_dict(),
        "configuration_checksum": inputs.configuration_checksum,
        "objects": [
            reference.to_content_dict()
            for reference in _object_references(inputs)
        ],
        "validation_report_checksum": report_checksum,
        "validation_summary": inputs.validation.summary.to_content_dict(),
        "failed_symbols": [],
        "retained_parent_coverage_symbols": [],
        "limitation_disclosure": {
            "version": disclosure.version,
            "text_checksum": disclosure_checksum,
        },
    }


def _reference_snapshot_id(inputs: SnapshotInputs) -> str:
    return "snap_" + sha256_bytes(
        canonical_json(_reference_content_identity(inputs))
    )


def _manifest(
    inputs: SnapshotInputs,
    *,
    created_at: datetime,
    local_path: str,
    operation_id: str,
    parent_snapshot_id: str | None = None,
    refs: Iterable[ContentAddressedObjectRef] | None = None,
) -> SnapshotManifest:
    """Assemble a manifest with operational differences excluded from identity."""

    return SnapshotManifestAssembler.assemble(
        provider=inputs.provider,
        requested_range=inputs.requested_range,
        covered_range=inputs.validation.summary.covered_range,
        configured_universe=inputs.configured_universe,
        benchmark_symbol="SPY",
        calendar=CalendarIdentity(
            "XNYS", "fixture-xnys-v1", inputs.calendar_checksum
        ),
        configuration_checksum=inputs.configuration_checksum,
        objects=tuple(
            _object_references(inputs) if refs is None else refs
        ),
        validation=inputs.validation,
        limitation_disclosure=LimitationDisclosure.current(),
        schema_versions=inputs.schema_versions,
        created_at=created_at,
        provider_requests=(),
        detection_times=(created_at + timedelta(seconds=17),),
        job_id=f"job-{operation_id}",
        local_manifest_path=local_path,
        notes={"operation": operation_id, "root": local_path},
        parent_snapshot_id=parent_snapshot_id,
        operation_id=operation_id,
    )


def _candidate(
    manifest: SnapshotManifest,
    inputs: SnapshotInputs,
    rows: Iterable[ScientificRow] | None = None,
) -> SnapshotPublicationCandidate:
    """Create a publication candidate with verified fixture bytes."""

    return SnapshotPublicationCandidate(
        manifest,
        staged_objects=_object_bytes(inputs, rows),
        validation_report=canonical_json(inputs.validation.to_content_dict()),
    )


def _mutate_scientific_content(
    inputs: SnapshotInputs,
    mutation: str,
) -> SnapshotInputs:
    """Apply exactly one valid scientific revision, never an operational edit."""

    if mutation == "row":
        first_symbol, first_session, first_close = inputs.rows[0]
        revised_rows = (
            (first_symbol, first_session, first_close + 1),
            *inputs.rows[1:],
        )
        return replace(
            inputs,
            rows=revised_rows,
            validation=_validation_report(revised_rows, inputs.symbols),
        )
    if mutation == "provider":
        return replace(inputs, provider=inputs.provider + "-revision")
    if mutation == "configuration":
        return replace(
            inputs,
            configuration_checksum=_digest(
                "configuration-revision", inputs.configuration_checksum
            ),
        )
    if mutation == "calendar":
        return replace(
            inputs,
            calendar_checksum=_digest("calendar-revision", inputs.calendar_checksum),
        )
    if mutation == "schema":
        return replace(
            inputs,
            schema_versions=replace(
                inputs.schema_versions,
                normalized_schema_version="daily_bar_v2",
            ),
        )
    if mutation == "validation":
        first = inputs.validation.per_symbol[0]
        revised_summary = replace(first, quarantined_count=1)
        revised_report = replace(
            inputs.validation,
            per_symbol=(revised_summary, *inputs.validation.per_symbol[1:]),
            quarantined_by_reason=(("fixture.revision", 1),),
        )
        return replace(inputs, validation=revised_report)
    raise AssertionError(f"unsupported scientific mutation: {mutation}")


@st.composite
def snapshot_cases(draw: st.DrawFn) -> SnapshotCase:
    """Generate parent content, merged content, permutations, and interruptions."""

    universe = tuple(
        draw(
            st.lists(
                st.sampled_from(_SYMBOL_POOL),
                min_size=1,
                max_size=3,
                unique=True,
            )
        )
    )
    symbols = (*universe, "SPY")
    close_values = draw(
        st.lists(
            st.integers(min_value=10, max_value=10_000),
            min_size=len(symbols) * len(_ROW_SESSIONS),
            max_size=len(symbols) * len(_ROW_SESSIONS),
        )
    )
    rows = tuple(
        (
            symbol,
            session.isoformat(),
            close_values[index * len(_ROW_SESSIONS) + session_index],
        )
        for index, symbol in enumerate(symbols)
        for session_index, session in enumerate(_ROW_SESSIONS)
    )
    parent_rows = tuple(
        row for row in rows if row[1] != _ROW_SESSIONS[-1].isoformat()
    )
    parent = SnapshotInputs(
        rows=parent_rows,
        requested_range=DateRange(_ROW_SESSIONS[0], _ROW_SESSIONS[1]),
        configured_universe=universe,
        provider=draw(st.sampled_from(("fixture", "fixture-provider"))),
        configuration_checksum=_digest(
            "configuration", draw(st.integers(min_value=0, max_value=10_000))
        ),
        calendar_checksum=_digest(
            "calendar", draw(st.integers(min_value=0, max_value=10_000))
        ),
        validation=_validation_report(parent_rows, symbols),
    )
    merged = SnapshotInputs(
        rows=rows,
        requested_range=DateRange(_ROW_SESSIONS[0], _ROW_SESSIONS[-1]),
        configured_universe=universe,
        provider=parent.provider,
        configuration_checksum=parent.configuration_checksum,
        calendar_checksum=parent.calendar_checksum,
        validation=_validation_report(rows, symbols),
    )
    return SnapshotCase(
        parent=parent,
        merged=merged,
        permuted_rows=tuple(draw(st.permutations(rows))),
        created_at=tuple(
            datetime(2024, 1, 10, 12, tzinfo=UTC)
            + timedelta(minutes=draw(st.integers(min_value=0, max_value=100_000)))
            for _ in range(3)
        ),
        operation_suffix=draw(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                min_size=1,
                max_size=12,
            )
        ),
        interruption_point=draw(st.sampled_from(_INTERRUPTION_POINTS)),
        mutation=draw(st.sampled_from(_MUTATIONS)),
    )


# Feature: quantitative-research-platform, Property 6: Snapshot content identity is idempotent and confluent
# Validates: Requirements 6.4–6.9, 6.17, 7.5–7.10, 17.6, 17.18
@settings(max_examples=100, deadline=None)
@given(case=snapshot_cases())
def test_snapshot_content_identity_is_idempotent_and_confluent(
    case: SnapshotCase,
) -> None:
    """Equivalent retries, batches, roots, and lineage share one scientific ID."""

    parent_expected_identity = _reference_content_identity(case.parent)
    parent_manifest = _manifest(
        case.parent,
        created_at=case.created_at[0],
        local_path="/first-root/snapshots/manifest.json",
        operation_id=f"parent-{case.operation_suffix}",
    )
    assert parent_manifest.to_content_identity_dict() == parent_expected_identity
    assert parent_manifest.snapshot_id == _reference_snapshot_id(case.parent)

    expected_identity = _reference_content_identity(case.merged)
    expected_id = _reference_snapshot_id(case.merged)
    canonical_refs = _object_references(case.merged)
    permuted_refs = _object_references(case.merged, case.permuted_rows)
    assert permuted_refs == canonical_refs

    first_manifest = _manifest(
        case.merged,
        created_at=case.created_at[0],
        local_path="/first-root/snapshots/manifest.json",
        operation_id=f"first-{case.operation_suffix}",
        parent_snapshot_id=parent_manifest.snapshot_id,
        refs=canonical_refs,
    )
    retry_manifest = _manifest(
        case.merged,
        created_at=case.created_at[1],
        local_path="/relocated-root/snapshots/manifest.json",
        operation_id=f"retry-{case.operation_suffix}",
        refs=reversed(permuted_refs),
    )
    assert first_manifest.to_content_identity_dict() == expected_identity
    assert retry_manifest.to_content_identity_dict() == expected_identity
    assert first_manifest.snapshot_id == retry_manifest.snapshot_id == expected_id
    assert first_manifest.manifest_checksum != retry_manifest.manifest_checksum
    assert first_manifest.content_identity.objects == canonical_refs
    assert retry_manifest.content_identity.objects == canonical_refs

    changed_inputs = _mutate_scientific_content(case.merged, case.mutation)
    changed_manifest = _manifest(
        changed_inputs,
        created_at=case.created_at[2],
        local_path="/first-root/revision/manifest.json",
        operation_id=f"revision-{case.operation_suffix}",
        parent_snapshot_id=parent_manifest.snapshot_id,
    )
    assert changed_manifest.snapshot_id == _reference_snapshot_id(changed_inputs)
    assert changed_manifest.snapshot_id != expected_id

    with TemporaryDirectory() as temporary_root:
        source_root = Path(temporary_root) / "source"
        store = FilesystemStore(source_root)
        parent_publication = store.publish_snapshot(
            _candidate(parent_manifest, case.parent),
            operation_id=f"publish-parent-{case.operation_suffix}",
        )
        assert parent_publication.snapshot_id == parent_manifest.snapshot_id
        parent_manifest_bytes = store.read_manifest(parent_manifest.snapshot_id)
        parent_object_bytes = {
            reference.relative_uri: store.read_object(reference.relative_uri)
            for reference in parent_manifest.content_identity.objects
        }

        def interrupt(point: str) -> None:
            if point == case.interruption_point:
                raise RuntimeError(f"injected interruption: {point}")

        try:
            store.publish_snapshot(
                _candidate(first_manifest, case.merged, case.permuted_rows),
                operation_id=f"interrupted-{case.operation_suffix}",
                fault_injector=interrupt,
            )
        except RuntimeError as error:
            assert str(error) == (
                f"injected interruption: {case.interruption_point}"
            )
        else:  # Every generated point is reached by this non-empty candidate.
            raise AssertionError("the generated publication interruption was not reached")

        recovered = store.publish_snapshot(
            _candidate(retry_manifest, case.merged, case.permuted_rows),
            operation_id=f"recovered-{case.operation_suffix}",
        )
        repeated = store.publish_snapshot(
            _candidate(
                _manifest(
                    case.merged,
                    created_at=case.created_at[2],
                    local_path="/another-root/snapshots/manifest.json",
                    operation_id=f"repeat-{case.operation_suffix}",
                    refs=canonical_refs,
                ),
                case.merged,
            ),
            operation_id=f"repeated-{case.operation_suffix}",
        )
        revised = store.publish_snapshot(
            _candidate(changed_manifest, changed_inputs),
            operation_id=f"revised-{case.operation_suffix}",
        )

        assert recovered.snapshot_id == repeated.snapshot_id == expected_id
        assert tuple(
            (reference.relative_uri, reference.checksum)
            for reference in recovered.manifest.content_identity.objects
        ) == tuple(
            (reference.relative_uri, reference.checksum)
            for reference in canonical_refs
        )
        assert revised.snapshot_id == changed_manifest.snapshot_id

        # A later retry or scientific revision cannot mutate the already
        # published parent bytes or its content-derived identity.
        assert store.read_manifest(parent_manifest.snapshot_id) == parent_manifest_bytes
        assert {
            reference.relative_uri: store.read_object(reference.relative_uri)
            for reference in parent_manifest.content_identity.objects
        } == parent_object_bytes
        assert isinstance(
            SnapshotManager(source_root).open_verified(parent_manifest.snapshot_id),
            Ok,
        )

        relocated_root = Path(temporary_root) / "relocated"
        shutil.copytree(source_root, relocated_root)
        relocated_manager = SnapshotManager(relocated_root)
        relocated_child = relocated_manager.open_verified(expected_id)
        relocated_parent = relocated_manager.open_verified(parent_manifest.snapshot_id)
        assert isinstance(relocated_child, Ok)
        assert isinstance(relocated_parent, Ok)
        assert relocated_child.value.snapshot_id == expected_id
        assert relocated_child.value.object_references == canonical_refs
