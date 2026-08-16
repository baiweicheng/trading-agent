# ruff: noqa: E501
"""Property tests for atomic publication and immutable snapshot state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.snapshots import SnapshotManager
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotLineage,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    SnapshotPublicationCandidate,
)

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
    "before_duckdb_commit",
    "after_duckdb_commit",
)
_COMPLETE_ORPHAN_POINTS = frozenset(
    {
        "after_publication_rename",
        "after_publication_parent_fsync",
        "before_duckdb_commit",
        "after_duckdb_commit",
    }
)
_MUTATION_COMMANDS = (
    "reject",
    "replace_manifest",
    "replace_object",
    "update",
    "delete",
)


@dataclass(frozen=True)
class PublicationCommand:
    """One operation in the independent fake publication state machine."""

    name: str
    outcome: bool | None = None
    mutation: str | None = None


@dataclass(frozen=True)
class PublicationCase:
    """Generated fake commands plus one real-filesystem interruption point."""

    commands: tuple[PublicationCommand, ...]
    interruption_point: str
    nonce: int


class FakeFilesystemStateMachine:
    """Small reference model for visible complete publications.

    This model deliberately has no filesystem or production imports.  It only
    models the observable append-only states needed by the property: the old
    publication is always retained, a candidate is visible only after complete
    publication, and reconciliation never promotes a partial/corrupt candidate.
    """

    previous_id = "previous"
    candidate_id = "candidate"

    def __init__(self) -> None:
        self.visible: set[str] = {self.previous_id}
        self.candidate_state = "absent"
        self.mutation_rejections = 0

    def apply(self, command: PublicationCommand) -> None:
        if command.name == "stage_write":
            self.candidate_state = "staged" if command.outcome else "absent"
            self.visible.discard(self.candidate_id)
            return
        if command.name == "checksum":
            if self.candidate_state == "staged" and command.outcome:
                self.candidate_state = "checksum_verified"
            else:
                self.candidate_state = "invalid"
            self.visible.discard(self.candidate_id)
            return
        if command.name == "validation":
            if self.candidate_state == "checksum_verified" and command.outcome:
                self.candidate_state = "validated"
            else:
                self.candidate_state = "invalid"
            self.visible.discard(self.candidate_id)
            return
        if command.name == "publish":
            if self.candidate_state == "validated":
                self.candidate_state = "complete"
                self.visible.add(self.candidate_id)
            else:
                self.candidate_state = "invalid"
                self.visible.discard(self.candidate_id)
            return
        if command.name == "partial":
            self.candidate_state = "partial"
            self.visible.discard(self.candidate_id)
            return
        if command.name == "corrupt":
            self.candidate_state = "corrupt"
            self.visible.discard(self.candidate_id)
            return
        if command.name == "reconcile":
            if self.candidate_state == "complete":
                self.visible.add(self.candidate_id)
            else:
                self.visible.discard(self.candidate_id)
            return
        if command.name == "mutation":
            self.mutation_rejections += 1
            return
        raise AssertionError(f"unknown publication command: {command.name}")

    def apply_all(self, commands: Iterable[PublicationCommand]) -> None:
        for command in commands:
            self.apply(command)

    def assert_invariants(self) -> None:
        assert self.previous_id in self.visible
        assert self.visible <= {self.previous_id, self.candidate_id}
        if self.candidate_state != "complete":
            assert self.candidate_id not in self.visible


def _command_strategy() -> st.SearchStrategy[PublicationCommand]:
    return st.one_of(
        st.booleans().map(
            lambda outcome: PublicationCommand("stage_write", outcome=outcome)
        ),
        st.booleans().map(
            lambda outcome: PublicationCommand("checksum", outcome=outcome)
        ),
        st.booleans().map(
            lambda outcome: PublicationCommand("validation", outcome=outcome)
        ),
        st.sampled_from(("publish", "partial", "corrupt", "reconcile")).map(
            PublicationCommand
        ),
        st.sampled_from(_MUTATION_COMMANDS).map(
            lambda mutation: PublicationCommand("mutation", mutation=mutation)
        ),
    )


@st.composite
def publication_cases(draw: st.DrawFn) -> PublicationCase:
    """Generate representative command sequences with a valid publication path."""

    # The prefix guarantees coverage of the complete path, reconciliation, a
    # partial/corrupt candidate, and at least one immutable mutation attempt;
    # the suffix explores arbitrary retries and failures around that path.
    prefix = (
        PublicationCommand("stage_write", outcome=True),
        PublicationCommand("checksum", outcome=True),
        PublicationCommand("validation", outcome=True),
        PublicationCommand("publish"),
        PublicationCommand("reconcile"),
        PublicationCommand("partial"),
        PublicationCommand("corrupt"),
        PublicationCommand("mutation", mutation="reject"),
    )
    suffix = tuple(draw(st.lists(_command_strategy(), min_size=0, max_size=12)))
    return PublicationCase(
        commands=(*prefix, *suffix),
        interruption_point=draw(st.sampled_from(_INTERRUPTION_POINTS)),
        nonce=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )


def _manifest(
    object_bytes: bytes,
    report_bytes: bytes,
    *,
    created_at: datetime,
    parent_snapshot_id: str | None = None,
    operation_id: str,
) -> SnapshotManifest:
    """Build a tiny complete manifest whose object bytes are independently known."""

    object_checksum = sha256_bytes(object_bytes)
    report_checksum = sha256_bytes(report_bytes)
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=object_checksum,
        relative_uri=(
            f"objects/normalized/symbol=AAPL/year=2024/sha256={object_checksum}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=1,
        byte_size=len(object_bytes),
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    content = SnapshotContentIdentity(
        provider="fixture",
        requested_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        covered_range=DateRange(date(2024, 1, 2), date(2024, 1, 3)),
        configured_universe=("AAPL",),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "fixture-xnys-v1", "a" * 64),
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
    )
    return SnapshotManifest(
        content_identity=content,
        operational_metadata=OperationalMetadata(created_at=created_at),
        lineage=SnapshotLineage(
            parent_snapshot_id=parent_snapshot_id,
            operation_id=operation_id,
        ),
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


def _object_path(store: FilesystemStore, reference: ContentAddressedObjectRef) -> Path:
    parts = PurePosixPath(reference.relative_uri).parts
    assert parts[0] == "objects"
    return store.objects_root.joinpath(*parts[1:])


def _write_partial_publication(
    store: FilesystemStore, manifest: SnapshotManifest
) -> None:
    """Leave only a manifest to model an interruption before publication completion."""

    directory = store.snapshots_root / manifest.snapshot_id
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest.to_manifest_dict()))


def _publish_then_corrupt(
    store: FilesystemStore,
    manifest: SnapshotManifest,
    object_bytes: bytes,
    report_bytes: bytes,
) -> None:
    """Publish a complete candidate, then corrupt one referenced immutable object."""

    store.publish_snapshot(_candidate(manifest, object_bytes, report_bytes))
    reference = manifest.content_identity.objects[0]
    _object_path(store, reference).write_bytes(b"corrupt-object-bytes")


def _assert_mutation_rejected(
    manager: SnapshotManager,
    snapshot_id: str,
    mutation: str,
) -> None:
    """Exercise each read-service mutation guard and require an ``Err`` result."""

    if mutation == "reject":
        result = manager.reject_mutation(snapshot_id)
    elif mutation == "replace_manifest":
        result = manager.replace_manifest(snapshot_id, b"replacement")
    elif mutation == "replace_object":
        result = manager.replace_object(
            snapshot_id,
            "objects/normalized/symbol=AAPL/year=2024/sha256=0.parquet",
            b"replacement",
        )
    elif mutation == "update":
        result = manager.update_snapshot(snapshot_id, provider="changed")
    elif mutation == "delete":
        result = manager.delete_snapshot(snapshot_id)
    else:
        raise AssertionError(f"unknown mutation command: {mutation}")
    assert isinstance(result, Err)
    assert result.errors


# Feature: quantitative-research-platform, Property 7: Publication and immutability state machine preserves the last valid snapshot
# Validates: Requirements 6.10–6.16, 6.18, 7.1–7.4, 7.11–7.16, 14.11, 17.23
@settings(max_examples=100, deadline=None)
@given(case=publication_cases())
def test_publication_and_immutability_state_machine_preserves_last_valid_snapshot(
    case: PublicationCase,
) -> None:
    """Only complete publications become visible and the prior snapshot survives."""

    reference_model = FakeFilesystemStateMachine()
    reference_model.apply_all(case.commands)
    reference_model.assert_invariants()
    assert reference_model.mutation_rejections == sum(
        command.name == "mutation" for command in case.commands
    )

    with TemporaryDirectory() as temporary_root:
        root = Path(temporary_root) / "store"
        metadata = DuckDBMetadataStore(Path(temporary_root) / "metadata.duckdb")
        try:
            created_at = datetime(2024, 1, 10, 15, tzinfo=UTC)
            old_object = f"previous-object-{case.nonce}".encode()
            old_report = f"previous-report-{case.nonce}".encode()
            previous = _manifest(
                old_object,
                old_report,
                created_at=created_at,
                operation_id=f"previous-{case.nonce}",
            )
            new_object = f"new-object-{case.nonce}".encode()
            new_report = f"new-report-{case.nonce}".encode()
            candidate = _manifest(
                new_object,
                new_report,
                created_at=created_at.replace(day=11),
                parent_snapshot_id=previous.snapshot_id,
                operation_id=f"candidate-{case.nonce}",
            )
            assert previous.snapshot_id != candidate.snapshot_id

            store = FilesystemStore(root, metadata=metadata)
            store.publish_snapshot(_candidate(previous, old_object, old_report))

            fired = False

            def interrupt(point: str) -> None:
                nonlocal fired
                if point == case.interruption_point and not fired:
                    fired = True
                    raise RuntimeError(f"injected interruption: {point}")

            try:
                store.publish_snapshot(
                    _candidate(candidate, new_object, new_report),
                    operation_id=f"interrupted-{case.nonce}",
                    fault_injector=interrupt,
                )
            except RuntimeError as error:
                assert str(error) == f"injected interruption: {case.interruption_point}"
            else:
                raise AssertionError(
                    "the sampled publication interruption was not reached"
                )
            assert fired

            # A filesystem reader can discover only complete directories.  A
            # complete orphan may exist after rename, but it is not indexed for
            # application use until startup reconciliation runs.
            visible_before_reconcile = set(store.list_published_manifest_ids())
            assert previous.snapshot_id in visible_before_reconcile
            assert visible_before_reconcile <= {
                previous.snapshot_id,
                candidate.snapshot_id,
            }
            complete_orphan = case.interruption_point in _COMPLETE_ORPHAN_POINTS
            assert (
                candidate.snapshot_id in visible_before_reconcile
            ) is complete_orphan

            indexed_reader = SnapshotManager(storage=store, metadata=metadata)
            assert isinstance(indexed_reader.open_verified(previous.snapshot_id), Ok)
            if complete_orphan and case.interruption_point != "after_duckdb_commit" or not complete_orphan:
                assert isinstance(
                    indexed_reader.open_verified(candidate.snapshot_id), Err
                )

            if any(command.name == "partial" for command in case.commands):
                partial_manifest = _manifest(
                    f"partial-object-{case.nonce}".encode(),
                    f"partial-report-{case.nonce}".encode(),
                    created_at=created_at,
                    operation_id=f"partial-{case.nonce}",
                )
                _write_partial_publication(store, partial_manifest)

            corrupt_manifest: SnapshotManifest | None = None
            if any(command.name == "corrupt" for command in case.commands):
                corrupt_object = f"corrupt-object-{case.nonce}".encode()
                corrupt_report = f"corrupt-report-{case.nonce}".encode()
                corrupt_manifest = _manifest(
                    corrupt_object,
                    corrupt_report,
                    created_at=created_at,
                    operation_id=f"corrupt-{case.nonce}",
                )
                _publish_then_corrupt(
                    store,
                    corrupt_manifest,
                    corrupt_object,
                    corrupt_report,
                )

            # Reconciliation indexes a complete orphan, leaves partial/corrupt
            # directories ignored, and marks any formerly indexed corruption
            # unavailable without changing the previous valid snapshot.
            restarted = FilesystemStore(root, metadata=metadata)
            reconciliation = restarted.reconcile()
            assert previous.snapshot_id in (
                *reconciliation.indexed_snapshot_ids,
                *reconciliation.already_indexed_snapshot_ids,
            )
            reconciled_reader = SnapshotManager(storage=restarted, metadata=metadata)
            assert isinstance(reconciled_reader.open_verified(previous.snapshot_id), Ok)
            if complete_orphan:
                assert candidate.snapshot_id in (
                    *reconciliation.indexed_snapshot_ids,
                    *reconciliation.already_indexed_snapshot_ids,
                )
                assert isinstance(
                    reconciled_reader.open_verified(candidate.snapshot_id), Ok
                )
            else:
                assert isinstance(
                    reconciled_reader.open_verified(candidate.snapshot_id), Err
                )

            if any(command.name == "partial" for command in case.commands):
                assert (
                    partial_manifest.snapshot_id
                    not in restarted.list_published_manifest_ids()
                )
                assert isinstance(
                    SnapshotManager(storage=restarted).open_verified(
                        partial_manifest.snapshot_id
                    ),
                    Err,
                )
                assert (
                    partial_manifest.snapshot_id
                    in reconciliation.ignored_publication_ids
                )

            if corrupt_manifest is not None:
                assert (
                    corrupt_manifest.snapshot_id
                    not in restarted.list_published_manifest_ids()
                )
                assert isinstance(
                    SnapshotManager(storage=restarted).open_verified(
                        corrupt_manifest.snapshot_id
                    ),
                    Err,
                )
                assert (
                    corrupt_manifest.snapshot_id
                    in reconciliation.ignored_publication_ids
                )

            # Every generated mutation command is routed through the immutable
            # application guard; no command can replace, delete, or update the
            # previously published scientific content.
            for command in case.commands:
                if command.name == "mutation":
                    mutation = command.mutation
                    assert mutation is not None
                    _assert_mutation_rejected(
                        reconciled_reader,
                        previous.snapshot_id,
                        mutation,
                    )
            assert isinstance(reconciled_reader.open_verified(previous.snapshot_id), Ok)
        finally:
            metadata.close()
