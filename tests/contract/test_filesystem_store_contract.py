"""Filesystem publication contracts against temporary local roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.infrastructure import filesystem_store as filesystem_module
from quant_research_platform.infrastructure.filesystem_store import (
    FilesystemStore,
    IntegrityVerificationError,
    PublisherLockError,
)


def test_artifact_publication_fsyncs_and_renames_on_one_device_then_verifies_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync_descriptors: list[int] = []
    rename_pairs: list[tuple[Path, Path]] = []
    original_replace = filesystem_module.os.replace

    def tracking_replace(source: object, destination: object) -> None:
        source_path = Path(source)  # type: ignore[arg-type]
        destination_path = Path(destination)  # type: ignore[arg-type]
        rename_pairs.append((source_path, destination_path))
        original_replace(source, destination)

    monkeypatch.setattr(filesystem_module.os, "replace", tracking_replace)
    store = FilesystemStore(
        tmp_path / "store",
        fsync_function=fsync_descriptors.append,
    )
    fsync_descriptors.clear()

    staging = store.create_staging("artifact-contract")
    staged = store.stage_stream(
        staging,
        "reports/summary.json",
        (b"{", b'"status":"verified"', b"}"),
    )
    reference = store.publish_artifact(
        staged,
        metadata={"role": "validation-report", "source": "contract-fixture"},
    )

    cas_path = store._cas_path(reference.relative_uri)
    publication_path = store.publications_root / f"{reference.checksum}.json"
    assert b"".join(store.stream_artifact(reference, chunk_size=2)) == (
        b'{"status":"verified"}'
    )
    assert fsync_descriptors
    assert {destination for _, destination in rename_pairs} >= {
        cas_path,
        publication_path,
    }
    assert all(
        source.parent.stat().st_dev == destination.parent.stat().st_dev
        for source, destination in rename_pairs
    )

    cas_path.write_bytes(b"corrupt")
    with pytest.raises(
        IntegrityVerificationError, match="(byte size|SHA-256) mismatch"
    ):
        list(store.stream_artifact(reference))


def test_publisher_lock_excludes_a_second_store_until_the_first_releases_it(
    tmp_path: Path,
) -> None:
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
