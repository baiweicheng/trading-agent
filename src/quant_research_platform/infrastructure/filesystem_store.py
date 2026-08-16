"""Local durable staging, content-addressed storage, and verified artifact access.

The store deliberately exposes small filesystem primitives rather than a mutable
"latest" view.  Candidate bytes live only below ``staging`` until they verify,
CAS bytes are addressable only through a publication record, and every reader
re-verifies a published reference before yielding its contents.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, TypeAlias
from uuid import uuid4

from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.manifests import (
    ArtifactReference,
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, SymbolValidationSummary

try:  # The supported local platforms provide POSIX advisory file locking.
    import fcntl
except ImportError:  # pragma: no cover - guarded for unsupported platforms.
    fcntl = None  # type: ignore[assignment]


BytesChunk: TypeAlias = bytes | bytearray | memoryview
WriteFunction: TypeAlias = Callable[[int, memoryview], int]
FsyncFunction: TypeAlias = Callable[[int], None]
DeviceResolver: TypeAlias = Callable[[Path], int]
FaultInjector: TypeAlias = Callable[[str], None]

_CHUNK_SIZE: Final = 1024 * 1024
_SAFE_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID = re.compile(r"^snap_[0-9a-f]{64}$")


class SnapshotMetadataIndex(Protocol):
    """Minimal metadata port needed by the filesystem publication boundary."""

    def insert_snapshot(
        self,
        manifest: SnapshotManifest,
        *,
        manifest_uri: str,
        symbol_statuses: tuple[SymbolValidationSummary, ...] = (),
    ) -> bool:
        """Insert or idempotently verify one published snapshot."""

    def set_snapshot_availability(self, snapshot_id: str, availability: str) -> object:
        """Update only operational availability for an indexed snapshot."""


@dataclass(frozen=True, slots=True, init=False)
class SnapshotPublicationCandidate:
    """Verified-input description consumed by :meth:`publish_snapshot`.

    ``objects`` may contain :class:`StagedFile` values, Parquet-store staged
    objects, paths, bytes, or a mapping from manifest URI to one of those
    values.  The publication code never trusts metadata supplied by a caller;
    it checks the bytes against the immutable references in ``manifest``.
    """

    manifest: SnapshotManifest
    objects: tuple[object, ...]
    validation_report: object | None
    symbol_statuses: tuple[SymbolValidationSummary, ...]
    staging: StagingArea | None
    source_mapping: tuple[tuple[str, object], ...]

    def __init__(
        self,
        manifest: SnapshotManifest,
        objects: Iterable[object] = (),
        *,
        staged_objects: Iterable[object] | Mapping[str, object] | None = None,
        validation_report: object | None = None,
        symbol_statuses: Iterable[SymbolValidationSummary] = (),
        staging: StagingArea | None = None,
    ) -> None:
        source_mapping: tuple[tuple[str, object], ...] = ()
        if staged_objects is not None:
            if tuple(objects):
                raise ValueError("supply either objects or staged_objects, not both")
            if isinstance(staged_objects, Mapping):
                source_mapping = tuple(
                    (str(key), value) for key, value in staged_objects.items()
                )
                objects = tuple(value for _, value in source_mapping)
            else:
                objects = staged_objects
        materialized = tuple(objects)
        statuses = tuple(symbol_statuses)
        if not isinstance(manifest, SnapshotManifest):
            raise TypeError("manifest must be a SnapshotManifest")
        if any(not isinstance(status, SymbolValidationSummary) for status in statuses):
            raise TypeError(
                "symbol_statuses must contain SymbolValidationSummary values"
            )
        if len({status.symbol for status in statuses}) != len(statuses):
            raise ValueError("symbol_statuses must contain one row per symbol")
        if staging is not None and not isinstance(staging, StagingArea):
            raise TypeError("staging must be a StagingArea or None")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "objects", materialized)
        object.__setattr__(self, "validation_report", validation_report)
        object.__setattr__(
            self,
            "symbol_statuses",
            tuple(sorted(statuses, key=lambda item: item.sort_key())),
        )
        object.__setattr__(self, "staging", staging)
        object.__setattr__(self, "source_mapping", source_mapping)

    @property
    def staged_objects(self) -> tuple[object, ...]:
        """Alias used by ingestion callers that name the inputs explicitly."""
        return self.objects


SnapshotCandidate = SnapshotPublicationCandidate


@dataclass(frozen=True, slots=True)
class PublishedSnapshot:
    """Result of a complete filesystem publication and optional index commit."""

    manifest: SnapshotManifest
    directory: Path
    objects: tuple[ContentAddressedFile, ...]
    validation_report: ContentAddressedFile
    indexed: bool
    reused: bool

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def publication_dir(self) -> Path:
        return self.directory


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Observable result of startup publication/index reconciliation."""

    indexed_snapshot_ids: tuple[str, ...] = ()
    already_indexed_snapshot_ids: tuple[str, ...] = ()
    unavailable_snapshot_ids: tuple[str, ...] = ()
    ignored_publication_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def indexed(self) -> tuple[str, ...]:
        return self.indexed_snapshot_ids

    @property
    def unavailable(self) -> tuple[str, ...]:
        return self.unavailable_snapshot_ids

    @property
    def ignored(self) -> tuple[str, ...]:
        return self.ignored_publication_ids


SnapshotPublication = PublishedSnapshot
SnapshotReconciliationReport = ReconciliationReport


class FilesystemStoreError(RuntimeError):
    """Base class for local-storage failures that must stop publication."""


class StorageRootError(FilesystemStoreError):
    """Raised when a configured storage root is missing or not a directory."""


class CrossDevicePublicationError(FilesystemStoreError):
    """Raised before a rename that cannot be atomic on one filesystem."""


class UnsafeStoragePathError(FilesystemStoreError):
    """Raised for a path that escapes its declared storage collection."""


class ExclusiveWriteError(FilesystemStoreError):
    """Raised when a staged file already exists or a write makes no progress."""


class IntegrityVerificationError(FilesystemStoreError):
    """Raised when byte size or SHA-256 verification fails."""


class ContentAddressConflictError(FilesystemStoreError):
    """Raised when an existing CAS path has bytes other than the staged object."""


class PublisherLockError(FilesystemStoreError):
    """Raised when another publisher currently owns the store lock."""


class ArtifactNotPublishedError(FilesystemStoreError):
    """Raised when a reader is asked to access staging or unreferenced CAS bytes."""


@dataclass(frozen=True, slots=True)
class StagingArea:
    """An operation-private directory whose contents are never reader-visible."""

    operation_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class StagedFile:
    """A fully flushed, checksummed file that remains below a staging area."""

    operation_id: str
    relative_path: str
    path: Path
    checksum: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ContentAddressedFile:
    """A verified immutable byte object promoted into the local CAS."""

    checksum: str
    byte_size: int
    relative_uri: str
    path: Path
    reused: bool


def _default_write(file_descriptor: int, data: memoryview) -> int:
    return os.write(file_descriptor, data)


def _validate_checksum(checksum: str) -> str:
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        raise ValueError("checksum must be a lowercase SHA-256 hexadecimal digest")
    return checksum


def _safe_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    if isinstance(value, PurePosixPath):
        path = value
    elif isinstance(value, str):
        path = PurePosixPath(value)
    else:
        raise TypeError("relative paths must be strings or PurePosixPath values")
    if "\\" in str(path) or path.is_absolute() or not path.parts:
        raise UnsafeStoragePathError(
            "storage paths must be non-empty relative POSIX paths"
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeStoragePathError(
            "storage paths must not escape their declared root"
        )
    return path


def _path_inside(root: Path, relative_path: str | PurePosixPath) -> Path:
    relative = _safe_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve(strict=False)
    resolved_parent = candidate.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise UnsafeStoragePathError("storage path resolves outside its declared root")
    return candidate


def fsync_file(path: Path) -> None:
    """Flush an existing regular file and its data metadata to the local device."""

    candidate = Path(path)
    if not candidate.is_file():
        raise StorageRootError(f"file does not exist or is not regular: {candidate}")
    descriptor = os.open(candidate, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Flush a directory entry update so rename/create durability is explicit."""

    candidate = Path(path)
    if not candidate.is_dir():
        raise StorageRootError(f"directory does not exist: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(candidate, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path, *, chunk_size: int = _CHUNK_SIZE) -> tuple[str, int]:
    """Return the SHA-256 checksum and size without materializing the file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    candidate = Path(path)
    digest = hashlib.sha256()
    byte_size = 0
    with candidate.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


def verify_file_checksum(
    path: Path,
    expected_checksum: str,
    *,
    expected_size: int | None = None,
) -> None:
    """Verify file bytes against an expected checksum and optional exact size."""

    checksum = _validate_checksum(expected_checksum)
    actual_checksum, actual_size = sha256_file(path)
    if expected_size is not None and actual_size != expected_size:
        raise IntegrityVerificationError(
            f"byte size mismatch for {path}: expected {expected_size}, "
            f"got {actual_size}"
        )
    if actual_checksum != checksum:
        raise IntegrityVerificationError(
            f"SHA-256 mismatch for {path}: expected {checksum}, got {actual_checksum}"
        )


def validate_same_filesystem_roots(*roots: Path) -> tuple[Path, ...]:
    """Reject roots on different devices before an atomic publication can begin."""

    if not roots:
        raise ValueError("at least one storage root is required")
    resolved = tuple(Path(root).resolve(strict=True) for root in roots)
    if any(not root.is_dir() for root in resolved):
        raise StorageRootError("every storage root must be an existing directory")
    devices = {root.stat().st_dev for root in resolved}
    if len(devices) != 1:
        joined = ", ".join(str(root) for root in resolved)
        raise CrossDevicePublicationError(
            f"publication roots must share one filesystem device: {joined}"
        )
    return resolved


class FilesystemStore:
    """A same-filesystem local store for staged CAS and verified artifacts.

    ``objects`` and ``artifacts`` hold immutable content bytes.  They are not
    reader-visible until a publication record is atomically written below
    ``artifact-publications``.  Snapshot publication uses the lower-level
    staging/CAS helpers and later supplies its own immutable manifest.
    """

    def __init__(
        self,
        root: Path,
        *,
        staging_root: Path | None = None,
        objects_root: Path | None = None,
        artifacts_root: Path | None = None,
        snapshots_root: Path | None = None,
        lock_root: Path | None = None,
        metadata: SnapshotMetadataIndex | None = None,
        metadata_store: SnapshotMetadataIndex | None = None,
        redactor: Redactor | None = None,
        write_function: WriteFunction | None = None,
        fsync_function: FsyncFunction | None = None,
        device_resolver: DeviceResolver | None = None,
        failure_injector: FaultInjector | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        if failure_injector is not None and fault_injector is not None:
            raise ValueError("supply either failure_injector or fault_injector")
        if metadata is not None and metadata_store is not None:
            raise ValueError("supply either metadata or metadata_store")
        self.root = self._create_root(Path(root), "root")
        self.staging_root = self._create_root(
            staging_root or self.root / "staging", "staging_root"
        )
        self.objects_root = self._create_root(
            objects_root or self.root / "objects", "objects_root"
        )
        self.artifacts_root = self._create_root(
            artifacts_root or self.root / "artifacts", "artifacts_root"
        )
        self.snapshots_root = self._create_root(
            snapshots_root or self.root / "snapshots", "snapshots_root"
        )
        self.publications_root = self._create_root(
            self.root / "artifact-publications", "publications_root"
        )
        self.lock_root = self._create_root(
            lock_root or self.root / "locks", "lock_root"
        )
        self.metadata = metadata or metadata_store
        self._redactor = redactor or Redactor()
        self._write: WriteFunction = write_function or _default_write
        self._fsync: FsyncFunction = fsync_function or os.fsync
        self._device_resolver = device_resolver or self._device_from_stat
        self._failure_injector = failure_injector or fault_injector
        self._publisher_lock_depth = 0
        self._publisher_lock_descriptor: int | None = None
        self.validate_same_filesystem_roots()

    @staticmethod
    def _create_root(path: Path, name: str) -> Path:
        candidate = path.expanduser().resolve(strict=False)
        candidate.mkdir(parents=True, exist_ok=True)
        if not candidate.is_dir():
            raise StorageRootError(f"{name} must be a directory: {candidate}")
        return candidate.resolve(strict=True)

    @staticmethod
    def _device_from_stat(path: Path) -> int:
        return path.stat().st_dev

    def validate_same_filesystem_roots(self) -> None:
        """Validate all publication participants use the same filesystem device."""

        roots = (
            self.root,
            self.staging_root,
            self.objects_root,
            self.artifacts_root,
            self.snapshots_root,
            self.publications_root,
            self.lock_root,
        )
        devices = {self._device_resolver(root) for root in roots}
        if len(devices) != 1:
            joined = ", ".join(str(root) for root in roots)
            raise CrossDevicePublicationError(
                f"publication roots must share one filesystem device: {joined}"
            )

    def create_staging(self, operation_id: str | None = None) -> StagingArea:
        """Create one exclusive hidden operation directory below ``staging``."""

        token = operation_id or uuid4().hex
        if not isinstance(token, str) or _SAFE_OPERATION_ID.fullmatch(token) is None:
            raise ValueError("operation_id must contain only safe filename characters")
        path = self.staging_root / f".{token}.staging"
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ExclusiveWriteError(
                f"staging area already exists for operation {token}"
            ) from error
        self._fsync_directory(self.staging_root)
        self._fsync_directory(path)
        return StagingArea(operation_id=token, path=path)

    def stage_bytes(
        self,
        staging: StagingArea,
        relative_path: str,
        data: BytesChunk,
        *,
        expected_checksum: str | None = None,
    ) -> StagedFile:
        """Exclusively stage and flush one complete byte string."""

        return self.stage_stream(
            staging,
            relative_path,
            (data,),
            expected_checksum=expected_checksum,
        )

    def stage_stream(
        self,
        staging: StagingArea,
        relative_path: str,
        chunks: Iterable[BytesChunk],
        *,
        expected_checksum: str | None = None,
    ) -> StagedFile:
        """Exclusively stage chunks, looping on short writes and fsyncing output."""

        area = self._validated_staging_area(staging)
        relative = _safe_relative_path(relative_path)
        path = _path_inside(area.path, relative)
        self._ensure_directory(path.parent)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise ExclusiveWriteError(
                f"staged file already exists: {relative.as_posix()}"
            ) from error

        try:
            self._fault("before_write")
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("staged chunks must be bytes-like")
                self._write_all(descriptor, memoryview(chunk))
            self._fault("after_write")
            self._fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            self._fsync_directory(path.parent)
            raise
        else:
            os.close(descriptor)

        self._fsync_directory(path.parent)
        self._fault("before_checksum")
        checksum, byte_size = sha256_file(path)
        self._fault("after_checksum")
        if expected_checksum is not None:
            expected = _validate_checksum(expected_checksum)
            if checksum != expected:
                path.unlink(missing_ok=True)
                self._fsync_directory(path.parent)
                raise IntegrityVerificationError(
                    "staged file checksum mismatch: "
                    f"expected {expected}, got {checksum}"
                )
        return StagedFile(
            operation_id=area.operation_id,
            relative_path=relative.as_posix(),
            path=path,
            checksum=checksum,
            byte_size=byte_size,
        )

    def promote_to_cas(
        self,
        staged: StagedFile,
        *,
        collection: str = "objects",
        suffix: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> ContentAddressedFile:
        """Promote one staged file to its digest-derived CAS path or reuse it."""

        if collection not in {"objects", "artifacts"}:
            raise ValueError("collection must be either 'objects' or 'artifacts'")
        if not isinstance(suffix, str) or "/" in suffix or "\\" in suffix:
            raise ValueError("suffix must be a simple filename suffix")
        relative_uri = (
            f"{collection}/sha256/{staged.checksum[:2]}/{staged.checksum}{suffix}"
        )
        return self.promote(staged, relative_uri=relative_uri, metadata=metadata)

    def promote(
        self,
        staged: StagedFile,
        *,
        relative_uri: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ContentAddressedFile:
        """Promote bytes to an explicit CAS URI without replacing a conflict.

        Existing bytes are first checksum-verified and then compared in chunks.
        A path collision containing even different same-digest test bytes is
        rejected rather than silently overwritten.
        """

        if metadata is not None:
            self.validate_artifact_metadata(metadata)
        source = self._validated_staged_file(staged)
        destination = self._cas_path(relative_uri)
        with self.publisher_lock():
            self._fault("before_object_checksum")
            verify_file_checksum(
                source.path,
                source.checksum,
                expected_size=source.byte_size,
            )
            self._fault("after_object_checksum")
            self._ensure_directory(destination.parent)
            self._ensure_same_device(source.path.parent, destination.parent)
            if destination.exists():
                self._verify_existing_object(destination, source)
                source.path.unlink(missing_ok=True)
                self._fsync_directory(source.path.parent)
                self._fault("after_object_promotion")
                return ContentAddressedFile(
                    checksum=source.checksum,
                    byte_size=source.byte_size,
                    relative_uri=relative_uri,
                    path=destination,
                    reused=True,
                )

            try:
                self._fault("before_object_promotion")
                os.replace(source.path, destination)
                self._fault("after_object_promotion")
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise CrossDevicePublicationError(
                        "refusing non-atomic cross-device object promotion"
                    ) from error
                raise
            self._fsync_directory(destination.parent)
            verify_file_checksum(
                destination,
                source.checksum,
                expected_size=source.byte_size,
            )
            return ContentAddressedFile(
                checksum=source.checksum,
                byte_size=source.byte_size,
                relative_uri=relative_uri,
                path=destination,
                reused=False,
            )

    def publish_artifact(
        self,
        staged: StagedFile,
        *,
        metadata: Mapping[str, object],
    ) -> ArtifactReference:
        """Promote an artifact then atomically create its reader-visible reference."""

        self.validate_artifact_metadata(metadata)
        with self.publisher_lock():
            stored = self.promote_to_cas(
                staged,
                collection="artifacts",
                metadata=metadata,
            )
            metadata_bytes = canonical_json(dict(metadata))
            document = canonical_json(
                {
                    "checksum": stored.checksum,
                    "byte_size": stored.byte_size,
                    "relative_uri": stored.relative_uri,
                    "metadata": json.loads(metadata_bytes),
                }
            )
            metadata_checksum = sha256_bytes(metadata_bytes)
            publication = self.publications_root / f"{stored.checksum}.json"
            if publication.exists():
                existing = publication.read_bytes()
                if existing != document:
                    raise ContentAddressConflictError(
                        "published artifact metadata differs for existing checksum "
                        f"{stored.checksum}"
                    )
            else:
                self._write_publication(publication, document)
            return ArtifactReference(
                checksum=stored.checksum,
                byte_size=stored.byte_size,
                relative_uri=stored.relative_uri,
                metadata_checksum=metadata_checksum,
            )

    def publish_snapshot(
        self,
        candidate: SnapshotPublicationCandidate | SnapshotManifest,
        staged_objects: Iterable[object] | Mapping[str, object] | None = None,
        *,
        metadata: SnapshotMetadataIndex | None = None,
        validation_report: object | None = None,
        symbol_statuses: Iterable[SymbolValidationSummary] = (),
        operation_id: str | None = None,
        staging: StagingArea | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> PublishedSnapshot:
        """Publish one immutable snapshot with recoverable filesystem/DB ordering.

        The order is deliberately strict: staged bytes are verified, immutable
        CAS objects are promoted, a complete publication directory is fsynced,
        that directory is renamed to an absent content-derived ID, the parent is
        fsynced, and only then is DuckDB indexed.  A failure after the rename
        leaves a complete unindexed publication for :meth:`reconcile` rather
        than deleting a previously valid snapshot or exposing a partial one.
        """

        if metadata is None and hasattr(staged_objects, "insert_snapshot"):
            metadata = staged_objects  # type: ignore[assignment]
            staged_objects = None
        if isinstance(candidate, SnapshotManifest):
            candidate = SnapshotPublicationCandidate(
                candidate,
                () if staged_objects is None else staged_objects,
                validation_report=validation_report,
                symbol_statuses=symbol_statuses,
                staging=staging,
            )
        elif not isinstance(candidate, SnapshotPublicationCandidate):
            raise TypeError(
                "candidate must be a SnapshotManifest or SnapshotPublicationCandidate"
            )
        elif any(
            value is not None for value in (staged_objects, validation_report, staging)
        ) or tuple(symbol_statuses):
            raise ValueError(
                "publication overrides are accepted only with a SnapshotManifest"
            )

        manifest = candidate.manifest
        index = metadata or self.metadata
        injector = fault_injector or self._failure_injector
        staging = candidate.staging
        owns_staging = staging is None
        if staging is None:
            token = operation_id or manifest.lineage.operation_id or uuid4().hex
            staging = self.create_staging(token)
        else:
            self._validated_staging_area(staging)
        publication_moved = False
        destination = self.snapshots_root / manifest.snapshot_id
        promoted: dict[str, ContentAddressedFile] = {}
        validation_stored: ContentAddressedFile | None = None

        try:
            with self.publisher_lock():
                self.validate_same_filesystem_roots()
                references = tuple(manifest.content_identity.objects)
                source_map = self._candidate_source_map(
                    candidate.objects,
                    references,
                    candidate.source_mapping,
                )
                for reference in references:
                    source = source_map.get(reference.relative_uri)
                    staged = self._materialize_staged_source(
                        staging,
                        source,
                        relative_path=f"inputs/{self._safe_name(reference.relative_uri)}",
                        expected_checksum=reference.checksum,
                        expected_size=reference.byte_size,
                    )
                    if staged is None:
                        promoted[reference.relative_uri] = self._existing_cas_file(
                            reference
                        )
                        continue
                    self._fault("before_snapshot_object_checksum", injector)
                    verify_file_checksum(
                        staged.path,
                        reference.checksum,
                        expected_size=reference.byte_size,
                    )
                    self._fault("after_snapshot_object_checksum", injector)
                    self._fault("before_snapshot_object_promotion", injector)
                    promoted[reference.relative_uri] = self.promote(
                        staged,
                        relative_uri=reference.relative_uri,
                    )
                    self._fault("after_snapshot_object_promotion", injector)

                validation_reference = self._validation_reference(
                    manifest,
                    candidate.validation_report,
                    source_map,
                    staging,
                )
                validation_source = candidate.validation_report
                if validation_source is None:
                    validation_source = source_map.get(
                        f"__validation__:{manifest.content_identity.validation_report_checksum}"
                    )
                if validation_source is None:
                    validation_source = source_map.get(
                        f"__checksum__:{manifest.content_identity.validation_report_checksum}"
                    )
                validation_staged = self._materialize_staged_source(
                    staging,
                    validation_source,
                    relative_path=(
                        "inputs/validation-"
                        + manifest.content_identity.validation_report_checksum
                        + ".bin"
                    ),
                    expected_checksum=validation_reference.checksum,
                    expected_size=validation_reference.byte_size,
                )
                if validation_staged is None:
                    validation_stored = self._existing_cas_file(validation_reference)
                else:
                    self._fault("before_validation_checksum", injector)
                    verify_file_checksum(
                        validation_staged.path,
                        validation_reference.checksum,
                        expected_size=validation_reference.byte_size,
                    )
                    self._fault("after_validation_checksum", injector)
                    self._fault("before_validation_promotion", injector)
                    validation_stored = self.promote(
                        validation_staged,
                        relative_uri=validation_reference.relative_uri,
                    )
                    self._fault("after_validation_promotion", injector)

                manifest_bytes = canonical_json(manifest.to_manifest_dict())
                if sha256_bytes(manifest_bytes) != manifest.manifest_checksum:
                    raise IntegrityVerificationError(
                        "snapshot manifest bytes do not match its declared checksum"
                    )
                publication_stage = _path_inside(
                    staging.path,
                    PurePosixPath("publication") / manifest.snapshot_id,
                )
                self._ensure_directory(publication_stage)
                self._fault("before_publication_write", injector)
                self.stage_bytes(
                    staging,
                    f"publication/{manifest.snapshot_id}/manifest.json",
                    manifest_bytes,
                    expected_checksum=manifest.manifest_checksum,
                )
                index_bytes = self._publication_index_bytes(
                    manifest,
                    validation_reference,
                    candidate.symbol_statuses,
                )
                self.stage_bytes(
                    staging,
                    f"publication/{manifest.snapshot_id}/index.json",
                    index_bytes,
                    expected_checksum=sha256_bytes(index_bytes),
                )
                self._fault("after_publication_write", injector)
                self._fault("before_publication_directory_fsync", injector)
                self._fsync_directory(publication_stage)
                self._fault("after_publication_directory_fsync", injector)

                reused = destination.exists()
                if reused:
                    existing_manifest, _, _ = self._verify_snapshot_directory(
                        destination, expected_snapshot_id=manifest.snapshot_id
                    )
                    if (
                        existing_manifest.to_content_identity_dict()
                        != manifest.to_content_identity_dict()
                    ):
                        raise ContentAddressConflictError(
                            "existing snapshot ID contains different scientific content"
                        )
                    self._remove_tree(publication_stage)
                else:
                    self._ensure_same_device(
                        publication_stage.parent, self.snapshots_root
                    )
                    self._fault("before_publication_rename", injector)
                    os.replace(publication_stage, destination)
                    publication_moved = True
                    self._fault("after_publication_rename", injector)
                    self._fsync_directory(self.snapshots_root)
                    self._fault("after_publication_parent_fsync", injector)

                verified_manifest, _, _ = self._verify_snapshot_directory(
                    destination, expected_snapshot_id=manifest.snapshot_id
                )
                if (
                    verified_manifest.to_content_identity_dict()
                    != manifest.to_content_identity_dict()
                ):
                    raise ContentAddressConflictError(
                        "published snapshot content identity changed during publication"
                    )
                if index is not None:
                    self._fault("before_duckdb_commit", injector)
                    indexed = self._index_snapshot(
                        index,
                        verified_manifest,
                        validation_reference,
                        candidate.symbol_statuses,
                    )
                    self._fault("after_duckdb_commit", injector)
                else:
                    indexed = False
                assert validation_stored is not None
                return PublishedSnapshot(
                    manifest=verified_manifest,
                    directory=destination,
                    objects=tuple(
                        promoted[reference.relative_uri] for reference in references
                    ),
                    validation_report=validation_stored,
                    indexed=indexed,
                    reused=reused,
                )
        except BaseException:
            if owns_staging and not publication_moved and not destination.exists():
                self._remove_tree(staging.path)
            raise

    publish = publish_snapshot
    publish_data_snapshot = publish_snapshot
    publish_snapshot_atomic = publish_snapshot

    def reconcile(
        self,
        metadata: SnapshotMetadataIndex | None = None,
    ) -> ReconciliationReport:
        """Index complete orphan publications and invalidate broken index rows.

        Reconciliation scans only ``snapshots/<snapshot-id>`` directories.  It
        never scans or returns ``staging`` or bare CAS objects, so an interrupted
        candidate cannot become a reader-visible snapshot by accident.
        """

        index = metadata or self.metadata
        indexed: list[str] = []
        already_indexed: list[str] = []
        unavailable: list[str] = []
        ignored: list[str] = []
        errors: list[str] = []
        verified: dict[
            str,
            tuple[
                SnapshotManifest,
                ContentAddressedObjectRef,
                tuple[SymbolValidationSummary, ...],
            ],
        ] = {}
        with self.publisher_lock():
            for directory in sorted(
                self.snapshots_root.iterdir(), key=lambda item: item.name
            ):
                if (
                    not directory.is_dir()
                    or _SNAPSHOT_ID.fullmatch(directory.name) is None
                ):
                    continue
                try:
                    verified[directory.name] = self._verify_snapshot_directory(
                        directory, expected_snapshot_id=directory.name
                    )
                except Exception as error:
                    ignored.append(directory.name)
                    errors.append(f"{directory.name}: {type(error).__name__}")

            if index is not None:
                for snapshot_id, (manifest, validation_ref, statuses) in sorted(
                    verified.items()
                ):
                    try:
                        was_inserted = self._index_snapshot(
                            index, manifest, validation_ref, statuses
                        )
                        if was_inserted:
                            indexed.append(snapshot_id)
                        else:
                            already_indexed.append(snapshot_id)
                        self._set_available_if_possible(index, snapshot_id)
                    except Exception as error:
                        errors.append(f"{snapshot_id}: {type(error).__name__}")

                for record in self._all_snapshot_records(index):
                    snapshot_id = str(getattr(record, "snapshot_id", ""))
                    if snapshot_id not in verified:
                        try:
                            index.set_snapshot_availability(snapshot_id, "unavailable")
                        except Exception as error:
                            errors.append(
                                f"{snapshot_id}: availability {type(error).__name__}"
                            )
                        unavailable.append(snapshot_id)
            else:
                indexed.extend(sorted(verified))

        return ReconciliationReport(
            indexed_snapshot_ids=tuple(sorted(set(indexed))),
            already_indexed_snapshot_ids=tuple(sorted(set(already_indexed))),
            unavailable_snapshot_ids=tuple(sorted(set(unavailable))),
            ignored_publication_ids=tuple(sorted(set(ignored))),
            errors=tuple(errors),
        )

    reconcile_startup = reconcile
    reconcile_publications = reconcile
    reconcile_snapshots = reconcile
    startup_reconcile = reconcile

    def read_manifest(self, snapshot_id: str, relative_uri: str | None = None) -> bytes:
        """Read a manifest only from a complete snapshot directory."""
        if _SNAPSHOT_ID.fullmatch(snapshot_id) is None:
            raise ValueError("snapshot_id must be a content-derived Snapshot_ID")
        expected = f"snapshots/{snapshot_id}/manifest.json"
        if relative_uri is not None and relative_uri != expected:
            raise ArtifactNotPublishedError("snapshot manifest URI is not canonical")
        path = self.snapshots_root / snapshot_id / "manifest.json"
        if not path.is_file():
            raise ArtifactNotPublishedError("snapshot manifest is not published")
        self._verify_snapshot_directory(
            path.parent,
            expected_snapshot_id=snapshot_id,
        )
        return path.read_bytes()

    def read_object(self, relative_uri: str) -> bytes:
        """Read a verified CAS object referenced by a complete snapshot."""
        path = self._cas_path(relative_uri)
        if not path.is_file() or not self._is_published_object(relative_uri):
            raise ArtifactNotPublishedError(
                "CAS object is not referenced by a published snapshot"
            )
        return path.read_bytes()

    def read_by_checksum(self, checksum: str) -> bytes:
        """Read the separately referenced validation report by checksum."""
        _validate_checksum(checksum)
        relative_uri = self._validation_uri(checksum)
        path = self._cas_path(relative_uri)
        if not path.is_file() or not self._is_published_object(relative_uri):
            raise ArtifactNotPublishedError(
                "validation report is not referenced by a published snapshot"
            )
        verify_file_checksum(path, checksum)
        return path.read_bytes()

    def list_published_manifest_ids(self) -> tuple[str, ...]:
        """List only IDs with complete, checksum-verified publications."""
        identifiers: list[str] = []
        for directory in self.snapshots_root.iterdir():
            if not directory.is_dir() or _SNAPSHOT_ID.fullmatch(directory.name) is None:
                continue
            try:
                self._verify_snapshot_directory(
                    directory,
                    expected_snapshot_id=directory.name,
                )
            except Exception:
                continue
            identifiers.append(directory.name)
        return tuple(sorted(identifiers))

    def _is_published_object(self, relative_uri: str) -> bool:
        for snapshot_id in self._candidate_snapshot_ids():
            try:
                manifest, validation_reference, _ = self._verify_snapshot_directory(
                    self.snapshots_root / snapshot_id,
                    expected_snapshot_id=snapshot_id,
                )
            except Exception:
                continue
            if relative_uri == validation_reference.relative_uri or any(
                reference.relative_uri == relative_uri
                for reference in manifest.content_identity.objects
            ):
                return True
        return False

    def _candidate_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(
            directory.name
            for directory in self.snapshots_root.iterdir()
            if directory.is_dir() and _SNAPSHOT_ID.fullmatch(directory.name) is not None
        )

    def artifact_reference(self, checksum: str) -> ArtifactReference:
        """Load the publication-gated reference for one local CAS artifact."""

        digest = _validate_checksum(checksum)
        publication = self.publications_root / f"{digest}.json"
        if not publication.is_file():
            raise ArtifactNotPublishedError(
                "artifact is not referenced by a published artifact record"
            )
        try:
            document = json.loads(publication.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrityVerificationError(
                f"published artifact record is unreadable: {publication}"
            ) from error
        if not isinstance(document, dict):
            raise IntegrityVerificationError(
                "published artifact record is not a mapping"
            )
        metadata = document.get("metadata")
        relative_uri = document.get("relative_uri")
        byte_size = document.get("byte_size")
        if (
            document.get("checksum") != digest
            or not isinstance(metadata, Mapping)
            or not isinstance(relative_uri, str)
            or not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
        ):
            raise IntegrityVerificationError(
                "published artifact record has invalid reference metadata"
            )
        try:
            return ArtifactReference(
                checksum=digest,
                byte_size=byte_size,
                relative_uri=relative_uri,
                metadata_checksum=sha256_bytes(canonical_json(dict(metadata))),
            )
        except (TypeError, ValueError) as error:
            raise IntegrityVerificationError(
                "published artifact record has invalid reference fields"
            ) from error

    def stream_artifact(
        self,
        reference: ArtifactReference,
        *,
        chunk_size: int = _CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """Yield published artifact bytes only after manifest and file verification."""

        if not isinstance(reference, ArtifactReference):
            raise TypeError("reference must be an ArtifactReference")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        publication = self.publications_root / f"{reference.checksum}.json"
        if not publication.is_file():
            raise ArtifactNotPublishedError(
                "artifact is not referenced by a published artifact record"
            )
        try:
            document = json.loads(publication.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrityVerificationError(
                f"published artifact record is unreadable: {publication}"
            ) from error
        if not isinstance(document, dict) or (
            document.get("checksum") != reference.checksum
            or document.get("byte_size") != reference.byte_size
            or document.get("relative_uri") != reference.relative_uri
        ):
            raise IntegrityVerificationError(
                "published artifact record does not match reference"
            )
        try:
            metadata_checksum = sha256_bytes(canonical_json(document["metadata"]))
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityVerificationError(
                "published artifact record has invalid metadata"
            ) from error
        if metadata_checksum != reference.metadata_checksum:
            raise IntegrityVerificationError(
                "published artifact metadata does not match reference"
            )
        path = self._cas_path(reference.relative_uri)
        verify_file_checksum(
            path,
            reference.checksum,
            expected_size=reference.byte_size,
        )
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk

    open_verified_artifact = stream_artifact

    def validate_artifact_metadata(self, metadata: Mapping[str, object]) -> None:
        """Apply the final fail-closed scan before durable metadata publication."""

        if not isinstance(metadata, Mapping):
            raise TypeError("artifact metadata must be a mapping")
        self._redactor.assert_metadata_is_redacted(metadata)

    @contextmanager
    def publisher_lock(self) -> Iterator[None]:
        """Hold the non-blocking advisory lock for one local publishing process."""

        if self._publisher_lock_depth:
            self._publisher_lock_depth += 1
            try:
                yield
            finally:
                self._publisher_lock_depth -= 1
            return
        if fcntl is None:  # pragma: no cover - platform guard.
            raise PublisherLockError("publisher advisory locking is unavailable")

        lock_path = self.lock_root / "publisher.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PublisherLockError(
                    "another local publisher currently holds the store lock"
                ) from error
            self._publisher_lock_descriptor = descriptor
            self._publisher_lock_depth = 1
            yield
        finally:
            if self._publisher_lock_depth:
                self._publisher_lock_depth = 0
            self._publisher_lock_descriptor = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _fault(self, point: str, injector: FaultInjector | None = None) -> None:
        hook = injector or self._failure_injector
        if hook is not None:
            hook(point)

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._=-]+", "_", value)[:180]

    def _candidate_source_map(
        self,
        values: Iterable[object],
        references: Sequence[ContentAddressedObjectRef],
        explicit_mapping: Sequence[tuple[str, object]] = (),
    ) -> dict[str, object]:
        materialized = tuple(values)
        result: dict[str, object] = {key: value for key, value in explicit_mapping}
        unkeyed: list[object] = []
        for value in materialized:
            uri = getattr(value, "relative_uri", None)
            if isinstance(uri, str):
                result[uri] = value
            else:
                unkeyed.append(value)
        for reference, value in zip(references, unkeyed, strict=False):
            result.setdefault(reference.relative_uri, value)
        for value in materialized:
            checksum = getattr(value, "checksum", None)
            if isinstance(checksum, str) and checksum:
                result.setdefault(f"__checksum__:{checksum}", value)
        return result

    def _materialize_staged_source(
        self,
        staging: StagingArea,
        source: object | None,
        *,
        relative_path: str,
        expected_checksum: str,
        expected_size: int,
    ) -> StagedFile | None:
        if source is None:
            return None
        if isinstance(source, StagedFile):
            try:
                staged = self._validated_staged_file(source)
                verify_file_checksum(
                    staged.path,
                    expected_checksum,
                    expected_size=expected_size,
                )
                return staged
            except (StorageRootError, UnsafeStoragePathError):
                source = source.path
        elif isinstance(source, (bytes, bytearray, memoryview)):
            staged = self.stage_bytes(
                staging,
                relative_path,
                source,
                expected_checksum=expected_checksum,
            )
            if staged.byte_size != expected_size:
                raise IntegrityVerificationError(
                    "staged snapshot object size differs from its manifest"
                )
            return staged

        path_value = getattr(source, "path", source)
        if not isinstance(path_value, (str, Path)):
            raise TypeError(
                "snapshot object sources must be staged files, paths, or bytes"
            )
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise StorageRootError(
                f"snapshot object source is not a regular file: {path}"
            )

        def chunks() -> Iterator[bytes]:
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK_SIZE):
                    yield chunk

        staged = self.stage_stream(
            staging,
            relative_path,
            chunks(),
            expected_checksum=expected_checksum,
        )
        if staged.byte_size != expected_size:
            staged.path.unlink(missing_ok=True)
            self._fsync_directory(staged.path.parent)
            raise IntegrityVerificationError(
                "staged snapshot object size differs from its manifest"
            )
        return staged

    def _existing_cas_file(
        self, reference: ContentAddressedObjectRef
    ) -> ContentAddressedFile:
        path = self._cas_path(reference.relative_uri)
        verify_file_checksum(
            path, reference.checksum, expected_size=reference.byte_size
        )
        return ContentAddressedFile(
            checksum=reference.checksum,
            byte_size=reference.byte_size,
            relative_uri=reference.relative_uri,
            path=path,
            reused=True,
        )

    def _validation_reference(
        self,
        manifest: SnapshotManifest,
        explicit_source: object | None,
        source_map: Mapping[str, object],
        staging: StagingArea,
    ) -> ContentAddressedObjectRef:
        checksum = manifest.content_identity.validation_report_checksum
        source = explicit_source
        if source is None:
            source = source_map.get(f"__validation__:{checksum}")
        if source is None:
            source = source_map.get(f"__checksum__:{checksum}")
        path = getattr(source, "path", source)
        byte_size = getattr(source, "byte_size", None)
        if isinstance(source, (bytes, bytearray, memoryview)):
            byte_size = len(source)
        if isinstance(path, (str, Path)) and Path(path).is_file():
            byte_size = Path(path).stat().st_size
        if byte_size is None:
            fixed = self._cas_path(self._validation_uri(checksum))
            if fixed.is_file():
                byte_size = fixed.stat().st_size
            else:
                raise StorageRootError(
                    "validation report bytes were not supplied and are not already "
                    "in CAS"
                )
        if (
            isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size < 0
        ):
            raise ValueError(
                "validation report byte_size must be a non-negative integer"
            )
        row_count = getattr(source, "row_count", 0)
        schema_version = getattr(source, "schema_version", "validation_report_v1")
        media_type = getattr(source, "media_type", "application/vnd.apache.parquet")
        return ContentAddressedObjectRef(
            object_kind=ObjectKind.VALIDATION,
            checksum=checksum,
            relative_uri=self._validation_uri(checksum),
            schema_version=str(schema_version),
            row_count=int(row_count),
            byte_size=byte_size,
            media_type=str(media_type),
        )

    @staticmethod
    def _validation_uri(checksum: str) -> str:
        _validate_checksum(checksum)
        return f"objects/validation/sha256={checksum}.parquet"

    @staticmethod
    def _publication_index_bytes(
        manifest: SnapshotManifest,
        validation_reference: ContentAddressedObjectRef,
        statuses: Sequence[SymbolValidationSummary],
    ) -> bytes:
        payload = {
            "manifest_checksum": manifest.manifest_checksum,
            "snapshot_id": manifest.snapshot_id,
            "validation_object": validation_reference.to_content_dict(),
            "symbol_statuses": [
                status.to_content_dict()
                for status in sorted(statuses, key=lambda item: item.sort_key())
            ],
        }
        return canonical_json(payload)

    def _verify_snapshot_directory(
        self,
        directory: Path,
        *,
        expected_snapshot_id: str,
    ) -> tuple[
        SnapshotManifest,
        ContentAddressedObjectRef,
        tuple[SymbolValidationSummary, ...],
    ]:
        if _SNAPSHOT_ID.fullmatch(expected_snapshot_id) is None:
            raise ValueError(
                "expected_snapshot_id must be a content-derived Snapshot_ID"
            )
        if not directory.is_dir():
            raise StorageRootError("snapshot publication directory is absent")
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise IntegrityVerificationError("snapshot publication manifest is absent")
        raw = manifest_path.read_bytes()
        try:
            from ..application.snapshots import _decode_manifest

            manifest = _decode_manifest(raw)
        except Exception as error:
            raise IntegrityVerificationError(
                "snapshot publication manifest is incomplete or non-canonical"
            ) from error
        if manifest.snapshot_id != expected_snapshot_id:
            raise IntegrityVerificationError(
                "snapshot publication directory ID does not match its manifest"
            )
        if sha256_bytes(raw) != manifest.manifest_checksum:
            raise IntegrityVerificationError(
                "snapshot publication manifest checksum is invalid"
            )

        for reference in manifest.content_identity.objects:
            path = self._cas_path(reference.relative_uri)
            verify_file_checksum(
                path, reference.checksum, expected_size=reference.byte_size
            )

        index_path = directory / "index.json"
        validation_reference = self._validation_reference_from_index(
            manifest, index_path
        )
        validation_path = self._cas_path(validation_reference.relative_uri)
        verify_file_checksum(
            validation_path,
            validation_reference.checksum,
            expected_size=validation_reference.byte_size,
        )
        statuses = self._statuses_from_index(manifest, index_path)
        return manifest, validation_reference, statuses

    def _validation_reference_from_index(
        self,
        manifest: SnapshotManifest,
        index_path: Path,
    ) -> ContentAddressedObjectRef:
        checksum = manifest.content_identity.validation_report_checksum
        if index_path.is_file():
            try:
                document = json.loads(index_path.read_bytes())
                if not isinstance(document, dict):
                    raise ValueError("publication index is not a mapping")
                if document.get("snapshot_id") != manifest.snapshot_id:
                    raise ValueError("publication index snapshot ID mismatch")
                if document.get("manifest_checksum") != manifest.manifest_checksum:
                    raise ValueError("publication index manifest checksum mismatch")
                value = document.get("validation_object")
                if isinstance(value, dict):
                    reference = ContentAddressedObjectRef(
                        object_kind=value["object_kind"],
                        checksum=value["checksum"],
                        relative_uri=value["relative_uri"],
                        schema_version=value["schema_version"],
                        row_count=value["row_count"],
                        byte_size=value["byte_size"],
                        symbol=value.get("symbol"),
                        session_year=value.get("session_year"),
                        media_type=value["media_type"],
                    )
                    if reference.checksum != checksum:
                        raise ValueError("validation report checksum mismatch")
                    return reference
            except (
                OSError,
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                raise IntegrityVerificationError(
                    "snapshot publication index is corrupt"
                ) from error
        path = self._cas_path(self._validation_uri(checksum))
        if not path.is_file():
            raise IntegrityVerificationError("snapshot validation report is absent")
        return ContentAddressedObjectRef(
            object_kind=ObjectKind.VALIDATION,
            checksum=checksum,
            relative_uri=self._validation_uri(checksum),
            schema_version="validation_report_v1",
            row_count=0,
            byte_size=path.stat().st_size,
            media_type="application/vnd.apache.parquet",
        )

    def _statuses_from_index(
        self,
        manifest: SnapshotManifest,
        index_path: Path,
    ) -> tuple[SymbolValidationSummary, ...]:
        if not index_path.is_file():
            return ()
        raw = index_path.read_bytes()
        document = json.loads(raw)
        if canonical_json(document) != raw:
            raise IntegrityVerificationError(
                "snapshot publication index is non-canonical"
            )
        if not isinstance(document, dict):
            raise IntegrityVerificationError(
                "snapshot publication index is not a mapping"
            )
        raw_statuses = document.get("symbol_statuses", [])
        if not isinstance(raw_statuses, list):
            raise IntegrityVerificationError(
                "snapshot publication statuses are invalid"
            )
        statuses: list[SymbolValidationSummary] = []
        for raw_status in raw_statuses:
            if not isinstance(raw_status, dict):
                raise IntegrityVerificationError(
                    "snapshot publication status is invalid"
                )
            covered = raw_status.get("covered_range")
            covered_range = None
            if covered is not None:
                if not isinstance(covered, dict):
                    raise IntegrityVerificationError("snapshot status range is invalid")
                covered_range = DateRange(
                    date.fromisoformat(str(covered["start"])),
                    date.fromisoformat(str(covered["end"])),
                )
            statuses.append(
                SymbolValidationSummary(
                    symbol=raw_status["symbol"],
                    accepted_count=raw_status["accepted_count"],
                    quarantined_count=raw_status["quarantined_count"],
                    duplicate_count=raw_status["duplicate_count"],
                    gap_count=raw_status["gap_count"],
                    stale=raw_status.get("stale", False),
                    staleness_lag_sessions=raw_status.get("staleness_lag_sessions", 0),
                    failed=raw_status.get("failed", False),
                    retained_parent_coverage=raw_status.get(
                        "retained_parent_coverage", False
                    ),
                    covered_range=covered_range,
                    comparison_ready=raw_status.get("comparison_ready", True),
                )
            )
        normalized = tuple(sorted(statuses, key=lambda item: item.sort_key()))
        if len({status.symbol for status in normalized}) != len(normalized):
            raise IntegrityVerificationError(
                "snapshot publication contains duplicate symbol statuses"
            )
        allowed_symbols = (
            *manifest.content_identity.configured_universe,
            manifest.content_identity.benchmark_symbol,
        )
        if any(status.symbol not in allowed_symbols for status in normalized):
            raise IntegrityVerificationError(
                "snapshot publication contains an unknown symbol status"
            )
        return normalized

    def _index_snapshot(
        self,
        metadata: SnapshotMetadataIndex,
        manifest: SnapshotManifest,
        validation_reference: ContentAddressedObjectRef,
        statuses: tuple[SymbolValidationSummary, ...],
    ) -> bool:
        transaction = getattr(metadata, "transaction", None)
        if callable(transaction):
            with transaction():
                record_object = getattr(metadata, "record_data_object", None)
                if callable(record_object):
                    record_object(
                        validation_reference,
                        created_at=manifest.operational_metadata.created_at,
                    )
                return metadata.insert_snapshot(
                    manifest,
                    manifest_uri=f"snapshots/{manifest.snapshot_id}/manifest.json",
                    symbol_statuses=statuses,
                )
        return metadata.insert_snapshot(
            manifest,
            manifest_uri=f"snapshots/{manifest.snapshot_id}/manifest.json",
            symbol_statuses=statuses,
        )

    @staticmethod
    def _set_available_if_possible(
        metadata: SnapshotMetadataIndex, snapshot_id: str
    ) -> None:
        get_snapshot = getattr(metadata, "get_snapshot", None)
        set_availability = getattr(metadata, "set_snapshot_availability", None)
        if not callable(get_snapshot) or not callable(set_availability):
            return
        try:
            record = get_snapshot(snapshot_id)
        except Exception:
            return
        if (
            str(
                getattr(
                    getattr(record, "availability", None),
                    "value",
                    getattr(record, "availability", "available"),
                )
            )
            != "available"
        ):
            set_availability(snapshot_id, "available")

    @staticmethod
    def _all_snapshot_records(metadata: SnapshotMetadataIndex) -> tuple[object, ...]:
        listing = getattr(metadata, "list_snapshots", None)
        if not callable(listing):
            return ()
        records: list[object] = []
        page = 0
        while True:
            try:
                batch = tuple(listing(page=page, page_size=100))
            except TypeError:
                batch = tuple(listing())
                records.extend(batch)
                break
            records.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return tuple(records)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _validated_staging_area(self, staging: StagingArea) -> StagingArea:
        if not isinstance(staging, StagingArea):
            raise TypeError("staging must be a StagingArea")
        expected = self.staging_root / f".{staging.operation_id}.staging"
        if staging.path.resolve(strict=False) != expected.resolve(strict=False):
            raise UnsafeStoragePathError("staging area does not belong to this store")
        if not expected.is_dir():
            raise StorageRootError("staging area is missing")
        return staging

    def _validated_staged_file(self, staged: StagedFile) -> StagedFile:
        if not isinstance(staged, StagedFile):
            raise TypeError("staged must be a StagedFile")
        area_path = self.staging_root / f".{staged.operation_id}.staging"
        area = self._validated_staging_area(StagingArea(staged.operation_id, area_path))
        expected_path = _path_inside(area.path, staged.relative_path)
        if staged.path.resolve(strict=False) != expected_path.resolve(strict=False):
            raise UnsafeStoragePathError(
                "staged file does not belong to its staging area"
            )
        if not staged.path.is_file():
            raise StorageRootError("staged file is missing")
        _validate_checksum(staged.checksum)
        if staged.byte_size < 0:
            raise ValueError("staged byte_size must be non-negative")
        return staged

    def _cas_path(self, relative_uri: str) -> Path:
        relative = _safe_relative_path(relative_uri)
        collection, *remainder = relative.parts
        if collection == "objects":
            root = self.objects_root
        elif collection == "artifacts":
            root = self.artifacts_root
        else:
            raise ArtifactNotPublishedError(
                "readers and promoters may address only objects or artifacts CAS paths"
            )
        if not remainder:
            raise UnsafeStoragePathError(
                "CAS URI must include a path below its collection"
            )
        return _path_inside(root, PurePosixPath(*remainder))

    def _write_all(self, descriptor: int, data: memoryview) -> None:
        pending = data
        while pending:
            written = self._write(descriptor, pending)
            if (
                isinstance(written, bool)
                or not isinstance(written, int)
                or written <= 0
            ):
                raise ExclusiveWriteError("staged write made no progress")
            if written > len(pending):
                raise ExclusiveWriteError(
                    "staged write reported more bytes than requested"
                )
            pending = pending[written:]

    def _ensure_directory(self, directory: Path) -> None:
        if directory.is_dir():
            return
        missing: list[Path] = []
        cursor = directory
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        if not cursor.is_dir():
            raise StorageRootError(f"storage parent is not a directory: {cursor}")
        for candidate in reversed(missing):
            candidate.mkdir(mode=0o700)
            self._fsync_directory(candidate.parent)
            self._fsync_directory(candidate)

    def _ensure_same_device(
        self,
        source_parent: Path,
        destination_parent: Path,
    ) -> None:
        source_device = self._device_resolver(source_parent)
        destination_device = self._device_resolver(destination_parent)
        if source_device != destination_device:
            raise CrossDevicePublicationError(
                "staging and final CAS destinations must share one filesystem device"
            )

    def _verify_existing_object(self, destination: Path, source: StagedFile) -> None:
        try:
            verify_file_checksum(
                destination,
                source.checksum,
                expected_size=source.byte_size,
            )
        except IntegrityVerificationError as error:
            raise ContentAddressConflictError(
                f"existing CAS object conflicts with checksum {source.checksum}"
            ) from error
        if not self._files_equal(source.path, destination):
            raise ContentAddressConflictError(
                "existing CAS object has conflicting bytes for checksum "
                f"{source.checksum}"
            )

    @staticmethod
    def _files_equal(first: Path, second: Path) -> bool:
        with first.open("rb") as first_handle, second.open("rb") as second_handle:
            while True:
                first_chunk = first_handle.read(_CHUNK_SIZE)
                second_chunk = second_handle.read(_CHUNK_SIZE)
                if first_chunk != second_chunk:
                    return False
                if not first_chunk:
                    return True

    def _write_publication(self, destination: Path, document: bytes) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                self._write_all(descriptor, memoryview(document))
                self._fsync(descriptor)
            finally:
                os.close(descriptor)
            self._fsync_directory(destination.parent)
            self._ensure_same_device(temporary.parent, destination.parent)
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _fsync_directory(self, directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            self._fsync(descriptor)
        finally:
            os.close(descriptor)


LocalFilesystemStore = FilesystemStore


__all__ = [
    "ArtifactNotPublishedError",
    "ArtifactReference",
    "ContentAddressConflictError",
    "ContentAddressedFile",
    "CrossDevicePublicationError",
    "ExclusiveWriteError",
    "FaultInjector",
    "FilesystemStore",
    "FilesystemStoreError",
    "IntegrityVerificationError",
    "LocalFilesystemStore",
    "PublishedSnapshot",
    "PublisherLockError",
    "ReconciliationReport",
    "SnapshotCandidate",
    "SnapshotMetadataIndex",
    "SnapshotPublication",
    "SnapshotPublicationCandidate",
    "SnapshotReconciliationReport",
    "StagedFile",
    "StagingArea",
    "StorageRootError",
    "UnsafeStoragePathError",
    "fsync_directory",
    "fsync_file",
    "sha256_file",
    "validate_same_filesystem_roots",
    "verify_file_checksum",
]
