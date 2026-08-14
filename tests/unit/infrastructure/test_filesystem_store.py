"""Focused real-filesystem coverage for durable staged CAS artifact storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_research_platform.config.serializer import Redactor, SecretLeakError
from quant_research_platform.domain.canonical import sha256_bytes
from quant_research_platform.infrastructure.filesystem_store import (
    ArtifactNotPublishedError,
    ArtifactReference,
    ContentAddressConflictError,
    CrossDevicePublicationError,
    ExclusiveWriteError,
    FilesystemStore,
    IntegrityVerificationError,
    PublisherLockError,
)


def test_staged_writes_are_exclusive_short_write_safe_and_fsynced(
    tmp_path: Path,
) -> None:
    fsync_calls: list[int] = []

    def short_write(file_descriptor: int, data: memoryview) -> int:
        return os.write(file_descriptor, data[:1])

    store = FilesystemStore(
        tmp_path / "store",
        write_function=short_write,
        fsync_function=fsync_calls.append,
    )
    staging = store.create_staging("ingest-1")
    staged = store.stage_bytes(staging, "raw/records.bin", b"durable bytes")

    assert staged.path.read_bytes() == b"durable bytes"
    assert staged.checksum == sha256_bytes(b"durable bytes")
    assert fsync_calls
    with pytest.raises(ExclusiveWriteError, match="already exists"):
        store.stage_bytes(staging, "raw/records.bin", b"replacement")


def test_staged_write_rejects_a_writer_that_makes_no_progress(tmp_path: Path) -> None:
    def zero_write(_file_descriptor: int, _data: memoryview) -> int:
        return 0

    store = FilesystemStore(tmp_path / "store", write_function=zero_write)
    staging = store.create_staging("ingest-2")

    with pytest.raises(ExclusiveWriteError, match="no progress"):
        store.stage_bytes(staging, "candidate.bin", b"bytes")
    assert not (staging.path / "candidate.bin").exists()


def test_store_rejects_roots_on_different_devices_before_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    objects_root = tmp_path / "separate-objects"

    def device_for(path: Path) -> int:
        expected_objects_root = objects_root.resolve(strict=False)
        return 2 if path.resolve(strict=False) == expected_objects_root else 1

    with pytest.raises(CrossDevicePublicationError, match="share one filesystem"):
        FilesystemStore(root, objects_root=objects_root, device_resolver=device_for)


def test_promotion_verifies_staged_bytes_reuses_identical_object_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    store = FilesystemStore(tmp_path / "store")
    first_area = store.create_staging("first")
    first_staged = store.stage_bytes(first_area, "object.bin", b"canonical object")
    first = store.promote(
        first_staged,
        relative_uri=f"objects/test/sha256={first_staged.checksum}.bin",
    )

    identical_area = store.create_staging("identical")
    identical_staged = store.stage_bytes(
        identical_area,
        "object.bin",
        b"canonical object",
    )
    reused = store.promote(identical_staged, relative_uri=first.relative_uri)

    conflict_area = store.create_staging("conflict")
    conflict_staged = store.stage_bytes(conflict_area, "object.bin", b"other bytes")
    with pytest.raises(ContentAddressConflictError, match="conflicts"):
        store.promote(conflict_staged, relative_uri=first.relative_uri)

    assert not first.reused
    assert reused.reused
    assert first.path.read_bytes() == b"canonical object"
    assert not identical_staged.path.exists()
    assert conflict_staged.path.exists()


def test_promotion_rejects_staged_checksum_mismatch_without_publishing(
    tmp_path: Path,
) -> None:
    store = FilesystemStore(tmp_path / "store")
    staging = store.create_staging("tampered")
    staged = store.stage_bytes(staging, "object.bin", b"original")
    staged.path.write_bytes(b"tampered")

    with pytest.raises(IntegrityVerificationError, match="SHA-256 mismatch"):
        store.promote_to_cas(staged)
    assert not list(store.objects_root.rglob("*"))


def test_artifact_publication_scans_metadata_and_streams_only_verified_publications(
    tmp_path: Path,
) -> None:
    secret = "https://proxy.example/?token=sensitive-value"
    store = FilesystemStore(tmp_path / "store", redactor=Redactor((secret,)))

    rejected_staging = store.create_staging("secret-metadata")
    rejected = store.stage_bytes(rejected_staging, "artifact.txt", b"private bytes")
    with pytest.raises(SecretLeakError):
        store.publish_artifact(rejected, metadata={"source": secret})
    assert not list(store.artifacts_root.rglob("*"))

    staging = store.create_staging("published")
    staged = store.stage_bytes(staging, "artifact.txt", b"verified bytes")
    reference = store.publish_artifact(
        staged,
        metadata={"source": "fixture", "credential": "[REDACTED]"},
    )
    assert b"".join(store.stream_artifact(reference, chunk_size=3)) == b"verified bytes"

    artifact_path = store._cas_path(reference.relative_uri)
    artifact_path.write_bytes(b"corrupt")
    with pytest.raises(
        IntegrityVerificationError,
        match="(byte size|SHA-256) mismatch",
    ):
        list(store.stream_artifact(reference))


def test_unreferenced_cas_objects_are_not_reader_visible(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path / "store")
    staging = store.create_staging("unreferenced")
    staged = store.stage_bytes(staging, "object.txt", b"unpublished")
    stored = store.promote_to_cas(staged, collection="artifacts")
    reference = ArtifactReference(
        checksum=stored.checksum,
        byte_size=stored.byte_size,
        relative_uri=stored.relative_uri,
        metadata_checksum=sha256_bytes(b"metadata"),
    )

    with pytest.raises(ArtifactNotPublishedError, match="not referenced"):
        list(store.stream_artifact(reference))


def test_publisher_lock_allows_only_one_writer_at_a_time(tmp_path: Path) -> None:
    first = FilesystemStore(tmp_path / "store")
    second = FilesystemStore(tmp_path / "store")

    with (
        first.publisher_lock(),
        pytest.raises(PublisherLockError, match="another local publisher"),
        second.publisher_lock(),
    ):
        pass

    with second.publisher_lock():
        pass
