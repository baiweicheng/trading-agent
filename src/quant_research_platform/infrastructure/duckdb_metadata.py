"""Transactional DuckDB indexes for immutable platform metadata.

DuckDB is deliberately an *index* in this adapter.  Raw/normalized market rows,
validation detail tables, and scientific run output remain in Parquet or the
content-addressed store; this module persists only their checksums, logical
locations, lifecycle state, and compact query projections.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast
from uuid import UUID, uuid4

import duckdb

from ..domain.canonical import canonical_json_text, sha256_canonical_json
from ..domain.errors import ActionableError
from ..domain.evaluation import (
    EvaluationMetrics,
    MetricName,
    MetricNullReason,
    MetricScope,
    MetricValue,
)
from ..domain.execution import (
    JobOperation,
    JobStage,
    JobState,
    ProgressUpdate,
    RunState,
    require_legal_job_transition,
    require_legal_run_transition,
)
from ..domain.manifests import ContentAddressedObjectRef, SnapshotManifest
from ..domain.market import (
    ProviderBatchResult,
    SymbolValidationSummary,
    normalize_symbol,
)

SCHEMA_VERSION: Final = 3
_CHECKSUM_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE: Final = re.compile(r"^snap_[0-9a-f]{64}$")
# DuckDB integer parameters are bounded; an arbitrarily large page number is
# still a valid empty page, not a reason to fail a discovery request.
_MAX_SQL_OFFSET: Final = 2**63 - 1


class MetadataStoreError(RuntimeError):
    """Base class for safe metadata persistence failures."""


class MetadataNotFoundError(MetadataStoreError):
    """Raised when a requested operational index row does not exist."""


class ImmutableMetadataError(MetadataStoreError):
    """Raised when an insert-only record or terminal run would be changed."""


class IllegalMetadataTransitionError(MetadataStoreError):
    """Raised when a repository lifecycle transition is not legal."""


class SnapshotAvailability(StrEnum):
    """Operational availability state; it never changes snapshot science."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class IngestionOperationStatus(StrEnum):
    """Operational result states for one ingestion attempt."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestionOperationRecord:
    """One operational ingestion attempt and its optional published result."""

    operation_id: UUID
    job_id: UUID
    mode: str
    parent_snapshot_id: str | None
    result_snapshot_id: str | None
    requested_start: date
    requested_end: date
    status: IngestionOperationStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """The bounded snapshot projection required for discovery and verification."""

    snapshot_id: str
    parent_snapshot_id: str | None
    manifest_checksum: str
    manifest_uri: str
    content_identity_checksum: str
    configuration_checksum: str
    provider: str
    requested_start: date
    requested_end: date
    covered_start: date | None
    covered_end: date | None
    universe: tuple[str, ...]
    benchmark_symbol: str
    comparison_ready: bool
    created_at: datetime
    availability: SnapshotAvailability


@dataclass(frozen=True, slots=True)
class SnapshotObjectRecord:
    """A snapshot-to-CAS reference, ordered deterministically by role/ordinal."""

    snapshot_id: str
    checksum: str
    role: str
    symbol: str | None
    session_year: int | None
    ordinal: int


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """A content-addressed artifact index entry and operational validity state."""

    checksum: str
    artifact_kind: str
    relative_uri: str
    media_type: str
    byte_size: int
    row_count: int | None
    schema_version: str | None
    created_at: datetime
    availability: SnapshotAvailability


@dataclass(frozen=True, slots=True)
class RunArtifactLink:
    """One immutable run-to-artifact association included in terminal payloads."""

    checksum: str
    role: str
    scientific: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checksum", _require_checksum("checksum", self.checksum)
        )
        object.__setattr__(self, "role", _require_text("role", self.role))
        if not isinstance(self.scientific, bool):
            raise TypeError("scientific must be a boolean")

    def to_payload(self) -> dict[str, object]:
        return {
            "checksum": self.checksum,
            "role": self.role,
            "scientific": self.scientific,
        }


@dataclass(frozen=True, slots=True)
class RunFinalization:
    """Checksummed terminal payload written before a run becomes immutable."""

    desired_state: RunState | str
    manifest_checksum: str | None
    manifest_uri: str | None
    metrics: tuple[EvaluationMetrics, ...] = ()
    artifacts: tuple[RunArtifactLink, ...] = ()
    errors: tuple[ActionableError, ...] = ()

    def __post_init__(self) -> None:
        state = RunState(self.desired_state)
        if state is RunState.RUNNING:
            raise ValueError("terminal finalization state must be succeeded or failed")
        checksum = self.manifest_checksum
        uri = self.manifest_uri
        if (checksum is None) != (uri is None):
            raise ValueError(
                "manifest_checksum and manifest_uri must be supplied together"
            )
        if checksum is not None:
            object.__setattr__(
                self,
                "manifest_checksum",
                _require_checksum("manifest_checksum", checksum),
            )
            object.__setattr__(
                self, "manifest_uri", _require_relative_uri("manifest_uri", uri)
            )
        if not isinstance(self.metrics, tuple):
            raise TypeError("metrics must be an immutable tuple")
        if any(not isinstance(metric, EvaluationMetrics) for metric in self.metrics):
            raise TypeError("metrics must contain EvaluationMetrics values")
        metric_scopes = tuple(
            MetricScope(metric.scope).value for metric in self.metrics
        )
        if metric_scopes != tuple(sorted(metric_scopes)):
            raise ValueError("metrics must be sorted by scope")
        if len(set(metric_scopes)) != len(metric_scopes):
            raise ValueError("metrics must contain at most one collection per scope")
        if not isinstance(self.artifacts, tuple):
            raise TypeError("artifacts must be an immutable tuple")
        if any(
            not isinstance(artifact, RunArtifactLink) for artifact in self.artifacts
        ):
            raise TypeError("artifacts must contain RunArtifactLink values")
        artifact_keys = tuple((item.role, item.checksum) for item in self.artifacts)
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError("artifacts must be sorted by role and checksum")
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("artifacts must not repeat a role/checksum reference")
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be an immutable tuple")
        if any(not isinstance(error, ActionableError) for error in self.errors):
            raise TypeError("errors must contain ActionableError values")
        if state is RunState.SUCCEEDED:
            if checksum is None or not self.metrics:
                raise ValueError("succeeded finalization requires manifest and metrics")
            if self.errors:
                raise ValueError("succeeded finalization must not include errors")
        if state is RunState.FAILED and not self.errors:
            raise ValueError(
                "failed finalization requires at least one actionable error"
            )
        object.__setattr__(self, "desired_state", state)
        object.__setattr__(
            self,
            "errors",
            tuple(sorted(self.errors, key=ActionableError.sort_key)),
        )

    def payload(self) -> dict[str, object]:
        """Return the canonical terminal payload copied to terminal indexes."""
        return {
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "desired_state": RunState(self.desired_state).value,
            "errors": [_error_payload(error) for error in self.errors],
            "manifest_checksum": self.manifest_checksum,
            "manifest_uri": self.manifest_uri,
            "metrics": [metric.to_serializable() for metric in self.metrics],
        }

    @property
    def payload_checksum(self) -> str:
        return sha256_canonical_json(self.payload())


@dataclass(frozen=True, slots=True)
class FinalizationIntent:
    """Persisted finalization intent used for idempotent recovery."""

    run_id: UUID
    desired_state: RunState
    terminal_payload_checksum: str
    terminal_payload_json: str
    mlflow_synced: bool
    created_at: datetime
    last_attempt_at: datetime | None
    last_error_json: str | None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Run-discovery projection; no scientific table rows are duplicated here."""

    run_id: UUID
    mlflow_run_id: str | None
    snapshot_id: str
    state: RunState
    strategy_id: str
    evaluation_start: date
    evaluation_end: date
    universe: tuple[str, ...]
    configuration_checksum: str
    environment_checksum: str
    manifest_checksum: str | None
    manifest_uri: str | None
    created_at: datetime
    started_at: datetime
    ended_at: datetime | None
    error_json: str | None
    immutable: bool


@dataclass(frozen=True, slots=True)
class RunQuery:
    """Explicit, bounded run-discovery filters with deterministic pagination."""

    run_id: UUID | None = None
    snapshot_id: str | None = None
    strategy_id: str | None = None
    universe: tuple[str, ...] | None = None
    evaluation_start: date | None = None
    evaluation_end: date | None = None
    state: RunState | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    page: int = 0
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.run_id is not None and not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be a UUID or None")
        if self.snapshot_id is not None:
            object.__setattr__(
                self, "snapshot_id", _require_snapshot_id(self.snapshot_id)
            )
        if self.strategy_id is not None:
            object.__setattr__(
                self, "strategy_id", _require_text("strategy_id", self.strategy_id)
            )
        if self.universe is not None:
            if not isinstance(self.universe, tuple):
                raise TypeError("universe must be an immutable tuple or None")
            universe = tuple(normalize_symbol(symbol) for symbol in self.universe)
            if not universe or len(set(universe)) != len(universe):
                raise ValueError("universe filter must contain distinct symbols")
            object.__setattr__(self, "universe", universe)
        for name in ("evaluation_start", "evaluation_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_date(name, value))
        if (
            self.evaluation_start is not None
            and self.evaluation_end is not None
            and self.evaluation_start > self.evaluation_end
        ):
            raise ValueError("evaluation_start must not be after evaluation_end")
        if self.state is not None:
            object.__setattr__(self, "state", RunState(self.state))
        for name in ("created_from", "created_to"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_utc_datetime(name, value))
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ValueError("created_from must not be after created_to")
        if (
            isinstance(self.page, bool)
            or not isinstance(self.page, int)
            or self.page < 0
        ):
            raise ValueError("page must be a non-negative integer")
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class RunPage:
    """One deterministic bounded page of run summaries."""

    records: tuple[RunRecord, ...]
    total_count: int
    page: int
    page_size: int


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Persisted mutable progress state, immutable after a terminal transition."""

    job_id: UUID
    operation: JobOperation
    state: JobState
    stage: JobStage
    completed_units: int
    total_units: int | None
    started_at: datetime | None
    updated_at: datetime
    ended_at: datetime | None
    warnings: tuple[str, ...]
    error_json: str | None


@dataclass(frozen=True, slots=True)
class JobEvent:
    """A sequenced, sanitized operational event for later inspection."""

    job_id: UUID
    sequence: int
    occurred_at: datetime
    level: str
    stage: JobStage
    message: str
    context_json: str


_MIGRATION_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS ingestion_operation (
      operation_id UUID PRIMARY KEY,
      job_id UUID NOT NULL,
      mode VARCHAR NOT NULL,
      parent_snapshot_id VARCHAR,
      result_snapshot_id VARCHAR,
      requested_start DATE NOT NULL,
      requested_end DATE NOT NULL,
      status VARCHAR NOT NULL,
      created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_request (
      request_id UUID PRIMARY KEY,
      request_content_key VARCHAR NOT NULL,
      job_id UUID NOT NULL,
      provider VARCHAR NOT NULL,
      requested_start DATE NOT NULL,
      requested_end DATE NOT NULL,
      symbols_json JSON NOT NULL,
      retrieval_started_at TIMESTAMPTZ NOT NULL,
      retrieval_ended_at TIMESTAMPTZ,
      status VARCHAR NOT NULL,
      attempts INTEGER NOT NULL,
      error_json JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_symbol_outcome (
      request_id UUID NOT NULL,
      symbol VARCHAR NOT NULL,
      status VARCHAR NOT NULL,
      row_count BIGINT NOT NULL,
      failure_class VARCHAR,
      error_json JSON,
      PRIMARY KEY (request_id, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_object (
      checksum VARCHAR PRIMARY KEY,
      object_kind VARCHAR NOT NULL,
      relative_uri VARCHAR NOT NULL UNIQUE,
      schema_version VARCHAR NOT NULL,
      symbol VARCHAR,
      session_year INTEGER,
      byte_size UBIGINT NOT NULL,
      row_count UBIGINT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot (
      snapshot_id VARCHAR PRIMARY KEY,
      parent_snapshot_id VARCHAR,
      manifest_checksum VARCHAR NOT NULL,
      manifest_uri VARCHAR NOT NULL,
      content_identity_checksum VARCHAR NOT NULL,
      configuration_checksum VARCHAR NOT NULL,
      provider VARCHAR NOT NULL,
      requested_start DATE NOT NULL,
      requested_end DATE NOT NULL,
      covered_start DATE,
      covered_end DATE,
      universe_json JSON NOT NULL,
      benchmark_symbol VARCHAR NOT NULL,
      validation_summary_json JSON NOT NULL,
      comparison_ready BOOLEAN NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      availability VARCHAR NOT NULL DEFAULT 'available'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot_object (
      snapshot_id VARCHAR NOT NULL,
      checksum VARCHAR NOT NULL,
      role VARCHAR NOT NULL,
      symbol VARCHAR,
      session_year INTEGER,
      ordinal INTEGER NOT NULL,
      PRIMARY KEY (snapshot_id, role, ordinal)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot_symbol_status (
      snapshot_id VARCHAR NOT NULL,
      symbol VARCHAR NOT NULL,
      accepted_count BIGINT NOT NULL,
      gap_count BIGINT NOT NULL,
      quarantine_count BIGINT NOT NULL,
      stale BOOLEAN NOT NULL,
      lag_sessions INTEGER NOT NULL,
      failed BOOLEAN NOT NULL,
      retained_parent_coverage BOOLEAN NOT NULL,
      comparison_ready BOOLEAN NOT NULL,
      PRIMARY KEY (snapshot_id, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifact (
      checksum VARCHAR PRIMARY KEY,
      artifact_kind VARCHAR NOT NULL,
      relative_uri VARCHAR NOT NULL UNIQUE,
      media_type VARCHAR NOT NULL,
      byte_size UBIGINT NOT NULL,
      row_count UBIGINT,
      schema_version VARCHAR,
      created_at TIMESTAMPTZ NOT NULL,
      availability VARCHAR NOT NULL DEFAULT 'available'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run (
      run_id VARCHAR PRIMARY KEY,
      mlflow_run_id VARCHAR UNIQUE,
      snapshot_id VARCHAR NOT NULL,
      state VARCHAR NOT NULL,
      strategy_id VARCHAR NOT NULL,
      evaluation_start DATE NOT NULL,
      evaluation_end DATE NOT NULL,
      universe_json JSON NOT NULL,
      universe_key VARCHAR NOT NULL,
      config_checksum VARCHAR NOT NULL,
      environment_checksum VARCHAR NOT NULL,
      manifest_checksum VARCHAR,
      manifest_uri VARCHAR,
      created_at TIMESTAMPTZ NOT NULL,
      started_at TIMESTAMPTZ NOT NULL,
      ended_at TIMESTAMPTZ,
      error_json JSON,
      immutable BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_metric (
      run_id VARCHAR NOT NULL,
      scope VARCHAR NOT NULL,
      metric_name VARCHAR NOT NULL,
      metric_value DOUBLE,
      null_reason VARCHAR,
      PRIMARY KEY (run_id, scope, metric_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_artifact (
      run_id VARCHAR NOT NULL,
      checksum VARCHAR NOT NULL,
      role VARCHAR NOT NULL,
      scientific BOOLEAN NOT NULL,
      ordinal INTEGER NOT NULL,
      PRIMARY KEY (run_id, role, ordinal)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_finalization (
      run_id VARCHAR PRIMARY KEY,
      desired_state VARCHAR NOT NULL,
      terminal_payload_checksum VARCHAR NOT NULL,
      terminal_payload_json JSON NOT NULL,
      mlflow_synced BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TIMESTAMPTZ NOT NULL,
      last_attempt_at TIMESTAMPTZ,
      last_error_json JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job (
      job_id UUID PRIMARY KEY,
      operation VARCHAR NOT NULL,
      state VARCHAR NOT NULL,
      stage VARCHAR NOT NULL,
      completed_units BIGINT NOT NULL,
      total_units BIGINT,
      started_at TIMESTAMPTZ,
      updated_at TIMESTAMPTZ NOT NULL,
      ended_at TIMESTAMPTZ,
      warnings_json JSON NOT NULL,
      error_json JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_event (
      job_id UUID NOT NULL,
      sequence BIGINT NOT NULL,
      occurred_at TIMESTAMPTZ NOT NULL,
      level VARCHAR NOT NULL,
      stage VARCHAR NOT NULL,
      message VARCHAR NOT NULL,
      context_json JSON NOT NULL,
      PRIMARY KEY (job_id, sequence)
    )
    """,
    "CREATE INDEX IF NOT EXISTS provider_request_job_idx ON provider_request(job_id)",
    "CREATE INDEX IF NOT EXISTS data_object_partition_idx ON data_object(object_kind, symbol, session_year)",
    "CREATE INDEX IF NOT EXISTS snapshot_discovery_idx ON snapshot(created_at, snapshot_id)",
    "CREATE INDEX IF NOT EXISTS snapshot_provider_range_idx ON snapshot(provider, requested_start, requested_end)",
    "CREATE INDEX IF NOT EXISTS snapshot_object_checksum_idx ON snapshot_object(checksum)",
    "CREATE INDEX IF NOT EXISTS artifact_availability_idx ON artifact(availability, created_at)",
    "CREATE INDEX IF NOT EXISTS run_created_idx ON run(created_at, run_id)",
    "CREATE INDEX IF NOT EXISTS run_snapshot_idx ON run(snapshot_id, created_at)",
    "CREATE INDEX IF NOT EXISTS run_strategy_idx ON run(strategy_id, created_at)",
    "CREATE INDEX IF NOT EXISTS run_state_idx ON run(state, created_at)",
    "CREATE INDEX IF NOT EXISTS run_universe_idx ON run(universe_key)",
    "CREATE INDEX IF NOT EXISTS job_event_order_idx ON job_event(job_id, sequence)",
)

# Version 2 makes the finalization-intent table independently migratable for
# databases created by earlier releases.  The table is also present in the
# bootstrap statements above, so this migration is safe for new databases.
_MIGRATION_V2_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS run_finalization (
      run_id VARCHAR PRIMARY KEY,
      desired_state VARCHAR NOT NULL,
      terminal_payload_checksum VARCHAR NOT NULL,
      terminal_payload_json JSON NOT NULL,
      mlflow_synced BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TIMESTAMPTZ NOT NULL,
      last_attempt_at TIMESTAMPTZ,
      last_error_json JSON
    )
    """,
)

# Version 3 adds the range projection index used by the run-discovery date
# filters.  It is separate from the terminal payload and does not duplicate
# metrics, manifests, or scientific output rows.
_MIGRATION_V3_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS run_evaluation_range_idx ON run("
    "evaluation_start, evaluation_end, created_at, run_id)",
)


def _migration_statements(version: int) -> tuple[str, ...]:
    if version == 1:
        return _MIGRATION_STATEMENTS
    if version == 2:
        return _MIGRATION_V2_STATEMENTS
    if version == 3:
        return _MIGRATION_V3_STATEMENTS
    raise MetadataStoreError(f"unsupported metadata migration version {version}")


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _require_checksum(name: str, value: str) -> str:
    checksum = _require_text(name, value)
    if _CHECKSUM_RE.fullmatch(checksum) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 checksum")
    return checksum


def _require_snapshot_id(value: str) -> str:
    snapshot_id = _require_text("snapshot_id", value)
    if _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None:
        raise ValueError("snapshot_id must be a content-derived snapshot ID")
    return snapshot_id


def _require_relative_uri(name: str, value: str | None) -> str:
    if value is None:
        raise TypeError(f"{name} must be a string")
    uri = _require_text(name, value)
    if "\\" in uri:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(uri)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError(f"{name} must be a non-escaping relative URI")
    return path.as_posix()


def _require_date(name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a calendar date")
    return value


def _require_utc_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_json(value: object) -> str:
    """Serialize JSON metadata in the shared deterministic document form."""
    return canonical_json_text(value).rstrip("\n")


def _error_payload(error: ActionableError) -> dict[str, object]:
    return {
        "category": error.category.value,
        "checksum": error.checksum,
        "correlation_id": error.correlation_id,
        "corrective_action": error.corrective_action,
        "field_path": error.field_path,
        "message": error.message,
        "operation": error.operation,
        "session": error.session,
        "symbol": error.symbol,
    }


def _errors_json(errors: Sequence[ActionableError]) -> str | None:
    if not errors:
        return None
    return _canonical_json([_error_payload(error) for error in errors])


def _json_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    decoded = json.loads(str(value))
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise MetadataStoreError("stored JSON array has an invalid shape")
    return tuple(decoded)


def _json_text(value: object | None) -> str | None:
    return None if value is None else str(value)


def _verify_terminal_payload(
    payload_json: str, expected_checksum: str
) -> dict[str, object]:
    """Verify a persisted intent before it can participate in recovery."""
    try:
        decoded = json.loads(payload_json)
    except (TypeError, ValueError) as error:
        raise MetadataStoreError(
            "stored finalization payload is not valid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise MetadataStoreError("stored finalization payload must be a JSON object")
    actual_checksum = sha256_canonical_json(decoded)
    if actual_checksum != expected_checksum:
        raise MetadataStoreError("stored finalization payload checksum does not match")
    return cast(dict[str, object], decoded)


class DuckDBMetadataStore:
    """One local DuckDB connection with transactional, repository-owned guards."""

    def __init__(self, database_path: Path | str) -> None:
        database = str(database_path)
        if database != ":memory:":
            Path(database).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._database_path = database
        self._connection: Any = duckdb.connect(database)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._closed = False
        self.migrate()

    @property
    def database_path(self) -> str:
        """Return the configured database location without resolving it into identity."""
        return self._database_path

    def __enter__(self) -> DuckDBMetadataStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Close the owned connection; calling this method repeatedly is safe."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    @contextmanager
    def transaction(self) -> Iterator[DuckDBMetadataStore]:
        """Run repository calls atomically, rolling back every exception."""
        with self._lock:
            self._ensure_open()
            outermost = self._transaction_depth == 0
            if outermost:
                self._connection.execute("BEGIN TRANSACTION")
            self._transaction_depth += 1
            try:
                yield self
            except BaseException:
                self._transaction_depth -= 1
                if outermost:
                    self._connection.execute("ROLLBACK")
                raise
            else:
                self._transaction_depth -= 1
                if outermost:
                    self._connection.execute("COMMIT")

    def migrate(self) -> None:
        """Apply ordered metadata migrations without changing scientific bytes."""
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migration (
                  version INTEGER PRIMARY KEY,
                  applied_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            rows = self._connection.execute(
                "SELECT version FROM schema_migration ORDER BY version"
            ).fetchall()
            versions = {int(row[0]) for row in rows}
            if any(version > SCHEMA_VERSION for version in versions):
                raise MetadataStoreError(
                    "metadata database uses a newer schema version"
                )
            next_version = max(versions, default=0) + 1
            while next_version <= SCHEMA_VERSION:
                self._connection.execute("BEGIN TRANSACTION")
                try:
                    for statement in _migration_statements(next_version):
                        self._connection.execute(statement)
                    self._connection.execute(
                        "INSERT INTO schema_migration(version, applied_at) VALUES (?, ?)",
                        [next_version, datetime.now(tz=UTC)],
                    )
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                else:
                    self._connection.execute("COMMIT")
                next_version += 1

    def create_ingestion_operation(
        self,
        *,
        operation_id: UUID,
        job_id: UUID,
        mode: str,
        requested_start: date,
        requested_end: date,
        created_at: datetime,
        parent_snapshot_id: str | None = None,
    ) -> IngestionOperationRecord:
        """Insert an operational ingestion attempt before provider work begins."""
        if not isinstance(operation_id, UUID) or not isinstance(job_id, UUID):
            raise TypeError("operation_id and job_id must be UUID values")
        start = _require_date("requested_start", requested_start)
        end = _require_date("requested_end", requested_end)
        if start > end:
            raise ValueError("requested_start must not be after requested_end")
        parent = (
            _require_snapshot_id(parent_snapshot_id) if parent_snapshot_id else None
        )
        record = IngestionOperationRecord(
            operation_id=operation_id,
            job_id=job_id,
            mode=_require_text("mode", mode),
            parent_snapshot_id=parent,
            result_snapshot_id=None,
            requested_start=start,
            requested_end=end,
            status=IngestionOperationStatus.RUNNING,
            created_at=_require_utc_datetime("created_at", created_at),
        )
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO ingestion_operation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(record.operation_id),
                    str(record.job_id),
                    record.mode,
                    record.parent_snapshot_id,
                    None,
                    record.requested_start,
                    record.requested_end,
                    record.status.value,
                    record.created_at,
                ],
            )
        return record

    def complete_ingestion_operation(
        self,
        operation_id: UUID,
        *,
        status: IngestionOperationStatus | str,
        result_snapshot_id: str | None = None,
    ) -> IngestionOperationRecord:
        """Terminalize an ingestion attempt once, preserving its attempt history."""
        if not isinstance(operation_id, UUID):
            raise TypeError("operation_id must be a UUID")
        target = IngestionOperationStatus(status)
        if target is IngestionOperationStatus.RUNNING:
            raise IllegalMetadataTransitionError(
                "ingestion operations must end terminally"
            )
        snapshot_id = (
            _require_snapshot_id(result_snapshot_id) if result_snapshot_id else None
        )
        with self.transaction():
            row = self._connection.execute(
                "SELECT status FROM ingestion_operation WHERE operation_id = ?",
                [str(operation_id)],
            ).fetchone()
            if row is None:
                raise MetadataNotFoundError(
                    f"ingestion operation {operation_id} was not found"
                )
            current = IngestionOperationStatus(row[0])
            if current is not IngestionOperationStatus.RUNNING:
                raise IllegalMetadataTransitionError(
                    "ingestion operation is already terminal"
                )
            self._connection.execute(
                """
                UPDATE ingestion_operation
                SET status = ?, result_snapshot_id = ?
                WHERE operation_id = ?
                """,
                [target.value, snapshot_id, str(operation_id)],
            )
        return self.get_ingestion_operation(operation_id)

    def get_ingestion_operation(self, operation_id: UUID) -> IngestionOperationRecord:
        """Load one ingestion operation by its operational UUID."""
        if not isinstance(operation_id, UUID):
            raise TypeError("operation_id must be a UUID")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT operation_id, job_id, mode, parent_snapshot_id, result_snapshot_id,
                       requested_start, requested_end, status, created_at
                FROM ingestion_operation WHERE operation_id = ?
                """,
                [str(operation_id)],
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(
                f"ingestion operation {operation_id} was not found"
            )
        return IngestionOperationRecord(
            operation_id=UUID(str(row[0])),
            job_id=UUID(str(row[1])),
            mode=str(row[2]),
            parent_snapshot_id=_json_text(row[3]),
            result_snapshot_id=_json_text(row[4]),
            requested_start=cast(date, row[5]),
            requested_end=cast(date, row[6]),
            status=IngestionOperationStatus(str(row[7])),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[8])),
        )

    def record_provider_batch(
        self,
        *,
        job_id: UUID,
        result: ProviderBatchResult,
        request_id: UUID | None = None,
        occurred_at: datetime | None = None,
    ) -> UUID:
        """Persist a provider request plus independent per-symbol outcomes."""
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        if not isinstance(result, ProviderBatchResult):
            raise TypeError("result must be a ProviderBatchResult")
        identifier = request_id or uuid4()
        if not isinstance(identifier, UUID):
            raise TypeError("request_id must be a UUID or None")
        fallback_time = _require_utc_datetime(
            "occurred_at", occurred_at or datetime.now(tz=UTC)
        )
        metadata = result.operational_metadata
        started_at = metadata.retrieval_started_at if metadata else fallback_time
        ended_at = metadata.retrieved_at if metadata else fallback_time
        assert started_at is not None
        attempts = max(outcome.attempts for outcome in result.outcomes)
        errors = tuple(error for outcome in result.outcomes for error in outcome.errors)
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO provider_request VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(identifier),
                    result.request.content_key,
                    str(job_id),
                    result.request.provider,
                    result.request.start,
                    result.request.end,
                    _canonical_json(list(result.request.symbols)),
                    started_at,
                    ended_at,
                    result.status,
                    attempts,
                    _errors_json(errors),
                ],
            )
            for outcome in result.outcomes:
                error_json = _errors_json(outcome.errors)
                self._connection.execute(
                    """
                    INSERT INTO provider_symbol_outcome VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(identifier),
                        outcome.symbol,
                        outcome.status.value,
                        len(outcome.records),
                        outcome.failure_kind.value if outcome.failure_kind else None,
                        error_json,
                    ],
                )
        return identifier

    def record_data_object(
        self,
        reference: ContentAddressedObjectRef,
        *,
        created_at: datetime,
    ) -> None:
        """Insert an immutable object reference or verify an identical replay."""
        if not isinstance(reference, ContentAddressedObjectRef):
            raise TypeError("reference must be a ContentAddressedObjectRef")
        created = _require_utc_datetime("created_at", created_at)
        with self.transaction():
            self._record_data_object(reference, created)

    def record_artifact(
        self,
        reference: ContentAddressedObjectRef,
        *,
        artifact_kind: str | None = None,
        created_at: datetime,
    ) -> None:
        """Index an artifact by checksum while retaining all table content in CAS."""
        if not isinstance(reference, ContentAddressedObjectRef):
            raise TypeError("reference must be a ContentAddressedObjectRef")
        created = _require_utc_datetime("created_at", created_at)
        kind = _require_text(
            "artifact_kind", artifact_kind or reference.object_kind.value
        )
        with self.transaction():
            self._record_data_object(reference, created)
            existing = self._connection.execute(
                """
                SELECT artifact_kind, relative_uri, media_type, byte_size, row_count,
                       schema_version
                FROM artifact WHERE checksum = ?
                """,
                [reference.checksum],
            ).fetchone()
            expected = (
                kind,
                reference.relative_uri,
                reference.media_type,
                reference.byte_size,
                reference.row_count,
                reference.schema_version,
            )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO artifact(
                      checksum, artifact_kind, relative_uri, media_type, byte_size,
                      row_count, schema_version, created_at, availability
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        reference.checksum,
                        *expected,
                        created,
                        SnapshotAvailability.AVAILABLE.value,
                    ],
                )
            elif tuple(existing) != expected:
                raise ImmutableMetadataError(
                    "artifact checksum already indexes different immutable metadata"
                )

    def set_artifact_availability(
        self, checksum: str, availability: SnapshotAvailability | str
    ) -> ArtifactRecord:
        """Record an operational artifact integrity result without changing its bytes."""
        digest = _require_checksum("checksum", checksum)
        state = SnapshotAvailability(availability)
        with self.transaction():
            existing = self._connection.execute(
                "SELECT 1 FROM artifact WHERE checksum = ?", [digest]
            ).fetchone()
            if existing is None:
                raise MetadataNotFoundError(f"artifact {digest} was not found")
            self._connection.execute(
                "UPDATE artifact SET availability = ? WHERE checksum = ?",
                [state.value, digest],
            )
        return self.get_artifact(digest)

    def get_artifact(self, checksum: str) -> ArtifactRecord:
        """Load one artifact index row by checksum."""
        digest = _require_checksum("checksum", checksum)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT checksum, artifact_kind, relative_uri, media_type, byte_size,
                       row_count, schema_version, created_at, availability
                FROM artifact WHERE checksum = ?
                """,
                [digest],
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(f"artifact {digest} was not found")
        return ArtifactRecord(
            checksum=str(row[0]),
            artifact_kind=str(row[1]),
            relative_uri=str(row[2]),
            media_type=str(row[3]),
            byte_size=int(row[4]),
            row_count=int(row[5]) if row[5] is not None else None,
            schema_version=_json_text(row[6]),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[7])),
            availability=SnapshotAvailability(str(row[8])),
        )

    def insert_snapshot(
        self,
        manifest: SnapshotManifest,
        *,
        manifest_uri: str,
        symbol_statuses: tuple[SymbolValidationSummary, ...] = (),
    ) -> bool:
        """Index one published immutable snapshot and all referenced CAS objects.

        Returns ``True`` for a newly indexed snapshot and ``False`` when an
        equivalent scientific snapshot was already indexed.  The latter never
        rewrites the initial manifest or operational lineage.
        """
        if not isinstance(manifest, SnapshotManifest):
            raise TypeError("manifest must be a SnapshotManifest")
        uri = _require_relative_uri("manifest_uri", manifest_uri)
        if not isinstance(symbol_statuses, tuple):
            raise TypeError("symbol_statuses must be an immutable tuple")
        if any(
            not isinstance(item, SymbolValidationSummary) for item in symbol_statuses
        ):
            raise TypeError(
                "symbol_statuses must contain SymbolValidationSummary values"
            )
        statuses = tuple(sorted(symbol_statuses, key=SymbolValidationSummary.sort_key))
        if len({item.symbol for item in statuses}) != len(statuses):
            raise ValueError("symbol_statuses must contain one row per symbol")
        identity = manifest.content_identity
        created = manifest.operational_metadata.created_at
        expected = (
            manifest.content_identity_checksum,
            identity.configuration_checksum,
            identity.provider,
            identity.requested_range.start,
            identity.requested_range.end,
            identity.covered_range.start if identity.covered_range else None,
            identity.covered_range.end if identity.covered_range else None,
            _canonical_json(list(identity.configured_universe)),
            identity.benchmark_symbol,
            _canonical_json(identity.validation_summary.to_content_dict()),
            identity.validation_summary.comparison_ready,
        )
        with self.transaction():
            existing = self._connection.execute(
                """
                SELECT content_identity_checksum, configuration_checksum, provider,
                       requested_start, requested_end, covered_start, covered_end,
                       universe_json, benchmark_symbol, validation_summary_json,
                       comparison_ready
                FROM snapshot WHERE snapshot_id = ?
                """,
                [manifest.snapshot_id],
            ).fetchone()
            if existing is not None:
                actual = list(existing)
                actual[7] = str(actual[7])
                actual[9] = str(actual[9])
                if tuple(actual) != expected:
                    raise ImmutableMetadataError(
                        "snapshot ID already indexes different scientific content"
                    )
                self._verify_snapshot_references(manifest, statuses)
                return False

            for reference in identity.objects:
                self._record_data_object(reference, created)
            self._connection.execute(
                """
                INSERT INTO snapshot(
                  snapshot_id, parent_snapshot_id, manifest_checksum, manifest_uri,
                  content_identity_checksum, configuration_checksum, provider,
                  requested_start, requested_end, covered_start, covered_end,
                  universe_json, benchmark_symbol, validation_summary_json,
                  comparison_ready, created_at, availability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    manifest.snapshot_id,
                    manifest.lineage.parent_snapshot_id,
                    manifest.manifest_checksum,
                    uri,
                    *expected,
                    created,
                    SnapshotAvailability.AVAILABLE.value,
                ],
            )
            for ordinal, reference in enumerate(identity.objects):
                self._connection.execute(
                    """
                    INSERT INTO snapshot_object(
                      snapshot_id, checksum, role, symbol, session_year, ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        manifest.snapshot_id,
                        reference.checksum,
                        reference.object_kind.value,
                        reference.symbol,
                        reference.session_year,
                        ordinal,
                    ],
                )
            for status in statuses:
                self._insert_snapshot_symbol_status(manifest.snapshot_id, status)
        return True

    def set_snapshot_availability(
        self, snapshot_id: str, availability: SnapshotAvailability | str
    ) -> SnapshotRecord:
        """Update only a snapshot's operational availability/invalidity flag."""
        identifier = _require_snapshot_id(snapshot_id)
        state = SnapshotAvailability(availability)
        with self.transaction():
            existing = self._connection.execute(
                "SELECT 1 FROM snapshot WHERE snapshot_id = ?", [identifier]
            ).fetchone()
            if existing is None:
                raise MetadataNotFoundError(f"snapshot {identifier} was not found")
            self._connection.execute(
                "UPDATE snapshot SET availability = ? WHERE snapshot_id = ?",
                [state.value, identifier],
            )
        return self.get_snapshot(identifier)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        """Load one snapshot's compact metadata projection."""
        identifier = _require_snapshot_id(snapshot_id)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT snapshot_id, parent_snapshot_id, manifest_checksum, manifest_uri,
                       content_identity_checksum, configuration_checksum, provider,
                       requested_start, requested_end, covered_start, covered_end,
                       universe_json, benchmark_symbol, comparison_ready, created_at,
                       availability
                FROM snapshot WHERE snapshot_id = ?
                """,
                [identifier],
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(f"snapshot {identifier} was not found")
        return self._snapshot_record(row)

    def list_snapshots(
        self,
        *,
        provider: str | None = None,
        availability: SnapshotAvailability | str | None = None,
        page: int = 0,
        page_size: int = 100,
    ) -> tuple[SnapshotRecord, ...]:
        """Return a bounded, deterministic snapshot page without reading Parquet."""
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("page must be a non-negative integer")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")
        clauses: list[str] = []
        parameters: list[object] = []
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(_require_text("provider", provider))
        if availability is not None:
            clauses.append("availability = ?")
            parameters.append(SnapshotAvailability(availability).value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend([page_size, page * page_size])
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT snapshot_id, parent_snapshot_id, manifest_checksum, manifest_uri,
                       content_identity_checksum, configuration_checksum, provider,
                       requested_start, requested_end, covered_start, covered_end,
                       universe_json, benchmark_symbol, comparison_ready, created_at,
                       availability
                FROM snapshot"""
                + where
                + " ORDER BY created_at DESC, snapshot_id ASC LIMIT ? OFFSET ?",
                parameters,
            ).fetchall()
        return tuple(self._snapshot_record(row) for row in rows)

    def list_snapshot_objects(
        self, snapshot_id: str
    ) -> tuple[SnapshotObjectRecord, ...]:
        """List immutable snapshot references in their stored canonical order."""
        identifier = _require_snapshot_id(snapshot_id)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT snapshot_id, checksum, role, symbol, session_year, ordinal
                FROM snapshot_object WHERE snapshot_id = ?
                ORDER BY role ASC, ordinal ASC
                """,
                [identifier],
            ).fetchall()
        return tuple(
            SnapshotObjectRecord(
                snapshot_id=str(row[0]),
                checksum=str(row[1]),
                role=str(row[2]),
                symbol=_json_text(row[3]),
                session_year=int(row[4]) if row[4] is not None else None,
                ordinal=int(row[5]),
            )
            for row in rows
        )

    def create_run(
        self,
        *,
        run_id: UUID,
        snapshot_id: str,
        strategy_id: str,
        evaluation_start: date,
        evaluation_end: date,
        universe: tuple[str, ...],
        configuration_checksum: str,
        environment_checksum: str,
        created_at: datetime,
        started_at: datetime,
        mlflow_run_id: str | None = None,
    ) -> RunRecord:
        """Allocate a discoverable running run before snapshot verification/execution."""
        if not isinstance(run_id, UUID):
            raise TypeError("run_id must be a UUID")
        start = _require_date("evaluation_start", evaluation_start)
        end = _require_date("evaluation_end", evaluation_end)
        if start > end:
            raise ValueError("evaluation_start must not be after evaluation_end")
        if not isinstance(universe, tuple):
            raise TypeError("universe must be an immutable tuple")
        normalized_universe = tuple(normalize_symbol(symbol) for symbol in universe)
        if not normalized_universe or len(set(normalized_universe)) != len(
            normalized_universe
        ):
            raise ValueError("universe must contain distinct normalized symbols")
        created = _require_utc_datetime("created_at", created_at)
        started = _require_utc_datetime("started_at", started_at)
        if started < created:
            raise ValueError("started_at must not precede created_at")
        mlflow = (
            _require_text("mlflow_run_id", mlflow_run_id) if mlflow_run_id else None
        )
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO run(
                  run_id, mlflow_run_id, snapshot_id, state, strategy_id,
                  evaluation_start, evaluation_end, universe_json, universe_key,
                  config_checksum, environment_checksum, manifest_checksum,
                  manifest_uri, created_at, started_at, ended_at, error_json, immutable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(run_id),
                    mlflow,
                    _require_snapshot_id(snapshot_id),
                    RunState.RUNNING.value,
                    _require_text("strategy_id", strategy_id),
                    start,
                    end,
                    _canonical_json(list(normalized_universe)),
                    _canonical_json(list(normalized_universe)),
                    _require_checksum("configuration_checksum", configuration_checksum),
                    _require_checksum("environment_checksum", environment_checksum),
                    None,
                    None,
                    created,
                    started,
                    None,
                    None,
                    False,
                ],
            )
        return self.get_run(run_id)

    def set_mlflow_run_id(self, run_id: UUID, mlflow_run_id: str) -> RunRecord:
        """Set a running run's one-to-one MLflow mapping once, idempotently."""
        identifier = self._run_id_text(run_id)
        mlflow = _require_text("mlflow_run_id", mlflow_run_id)
        with self.transaction():
            row = self._connection.execute(
                "SELECT state, immutable, mlflow_run_id FROM run WHERE run_id = ?",
                [identifier],
            ).fetchone()
            self._require_mutable_running_run(identifier, row)
            current = _json_text(row[2])
            if current is None:
                self._connection.execute(
                    "UPDATE run SET mlflow_run_id = ? WHERE run_id = ?",
                    [mlflow, identifier],
                )
            elif current != mlflow:
                raise ImmutableMetadataError("MLflow run ID is already bound")
        return self.get_run(run_id)

    def create_finalization_intent(
        self,
        run_id: UUID,
        finalization: RunFinalization,
        *,
        created_at: datetime,
    ) -> FinalizationIntent:
        """Durably insert a checksummed terminal intent while the run is running."""
        identifier = self._run_id_text(run_id)
        if not isinstance(finalization, RunFinalization):
            raise TypeError("finalization must be a RunFinalization")
        created = _require_utc_datetime("created_at", created_at)
        payload = _canonical_json(finalization.payload())
        checksum = finalization.payload_checksum
        with self.transaction():
            run_row = self._connection.execute(
                "SELECT state, immutable, mlflow_run_id FROM run WHERE run_id = ?",
                [identifier],
            ).fetchone()
            self._require_mutable_running_run(identifier, run_row)
            existing = self._connection.execute(
                """
                SELECT desired_state, terminal_payload_checksum, terminal_payload_json
                FROM run_finalization WHERE run_id = ?
                """,
                [identifier],
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO run_finalization(
                      run_id, desired_state, terminal_payload_checksum,
                      terminal_payload_json, mlflow_synced, created_at,
                      last_attempt_at, last_error_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        identifier,
                        RunState(finalization.desired_state).value,
                        checksum,
                        payload,
                        False,
                        created,
                        None,
                        None,
                    ],
                )
            else:
                stored_payload = _verify_terminal_payload(
                    str(existing[2]), str(existing[1])
                )
                if stored_payload.get("desired_state") != str(existing[0]):
                    raise MetadataStoreError(
                        "stored finalization intent state does not match payload"
                    )
                if tuple(existing[:2]) != (
                    RunState(finalization.desired_state).value,
                    checksum,
                ):
                    raise ImmutableMetadataError(
                        "run finalization intent already has a different terminal payload"
                    )
        return self.get_finalization_intent(run_id)

    def mark_finalization_mlflow_synced(
        self,
        run_id: UUID,
        *,
        attempted_at: datetime,
        error: ActionableError | None = None,
    ) -> FinalizationIntent:
        """Record idempotent MLflow synchronization state for recovery on restart."""
        identifier = self._run_id_text(run_id)
        if error is not None and not isinstance(error, ActionableError):
            raise TypeError("error must be an ActionableError or None")
        attempted = _require_utc_datetime("attempted_at", attempted_at)
        with self.transaction():
            run_row = self._connection.execute(
                "SELECT state, immutable FROM run WHERE run_id = ?", [identifier]
            ).fetchone()
            self._require_mutable_running_run(identifier, run_row)
            row = self._connection.execute(
                "SELECT run_id FROM run_finalization WHERE run_id = ?", [identifier]
            ).fetchone()
            if row is None:
                raise MetadataNotFoundError(
                    f"run finalization intent for {run_id} was not found"
                )
            self._connection.execute(
                """
                UPDATE run_finalization
                SET mlflow_synced = ?, last_attempt_at = ?, last_error_json = ?
                WHERE run_id = ?
                """,
                [
                    error is None,
                    attempted,
                    _errors_json((error,)) if error else None,
                    identifier,
                ],
            )
        return self.get_finalization_intent(run_id)

    def finalize_run(
        self,
        run_id: UUID,
        finalization: RunFinalization,
        *,
        ended_at: datetime,
    ) -> RunRecord:
        """Atomically index terminal payloads and permanently guard the run."""
        identifier = self._run_id_text(run_id)
        if not isinstance(finalization, RunFinalization):
            raise TypeError("finalization must be a RunFinalization")
        ended = _require_utc_datetime("ended_at", ended_at)
        with self.transaction():
            run_row = self._connection.execute(
                """
                SELECT state, immutable, mlflow_run_id, started_at
                FROM run WHERE run_id = ?
                """,
                [identifier],
            ).fetchone()
            if run_row is None:
                raise MetadataNotFoundError(f"run {identifier} was not found")
            current_state = RunState(str(run_row[0]))
            if current_state is not RunState.RUNNING:
                intent = self._connection.execute(
                    """
                    SELECT desired_state, terminal_payload_checksum, terminal_payload_json
                    FROM run_finalization WHERE run_id = ?
                    """,
                    [identifier],
                ).fetchone()
                if (
                    bool(run_row[1])
                    and intent is not None
                    and current_state is finalization.desired_state
                    and str(intent[0]) == RunState(finalization.desired_state).value
                    and str(intent[1]) == finalization.payload_checksum
                ):
                    _verify_terminal_payload(str(intent[2]), str(intent[1]))
                    return self.get_run(run_id)
                raise ImmutableMetadataError(
                    "terminal runs are immutable; create a new Run ID"
                )
            try:
                require_legal_run_transition(current_state, finalization.desired_state)
            except ValueError as error:
                raise IllegalMetadataTransitionError(str(error)) from error
            self._require_mutable_running_run(identifier, run_row)
            started_at = _require_utc_datetime("started_at", cast(datetime, run_row[3]))
            if ended < started_at:
                raise ValueError("ended_at must not precede started_at")
            intent = self._connection.execute(
                """
                SELECT desired_state, terminal_payload_checksum, terminal_payload_json, mlflow_synced
                FROM run_finalization WHERE run_id = ?
                """,
                [identifier],
            ).fetchone()
            if intent is None:
                raise IllegalMetadataTransitionError(
                    "terminal run requires a finalization intent"
                )
            if str(intent[0]) != RunState(finalization.desired_state).value:
                raise ImmutableMetadataError(
                    "finalization state disagrees with saved intent"
                )
            if str(intent[1]) != finalization.payload_checksum:
                raise ImmutableMetadataError(
                    "finalization payload disagrees with saved intent"
                )
            _verify_terminal_payload(str(intent[2]), str(intent[1]))
            if not bool(intent[3]):
                raise IllegalMetadataTransitionError(
                    "MLflow terminalization has not been synchronized"
                )
            self._verify_terminal_artifacts(finalization.artifacts)
            self._connection.execute(
                "DELETE FROM run_metric WHERE run_id = ?", [identifier]
            )
            self._connection.execute(
                "DELETE FROM run_artifact WHERE run_id = ?", [identifier]
            )
            for metrics in finalization.metrics:
                for metric in metrics.metrics:
                    self._insert_metric(
                        identifier, MetricScope(metrics.scope).value, metric
                    )
            for ordinal, artifact in enumerate(finalization.artifacts):
                self._connection.execute(
                    """
                    INSERT INTO run_artifact(run_id, checksum, role, scientific, ordinal)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        identifier,
                        artifact.checksum,
                        artifact.role,
                        artifact.scientific,
                        ordinal,
                    ],
                )
            self._connection.execute(
                """
                UPDATE run
                SET state = ?, manifest_checksum = ?, manifest_uri = ?, ended_at = ?,
                    error_json = ?, immutable = TRUE
                WHERE run_id = ?
                """,
                [
                    RunState(finalization.desired_state).value,
                    finalization.manifest_checksum,
                    finalization.manifest_uri,
                    ended,
                    _errors_json(finalization.errors),
                    identifier,
                ],
            )
        return self.get_run(run_id)

    def get_finalization_intent(self, run_id: UUID) -> FinalizationIntent:
        """Load one durable terminal intent for restart/reconciliation logic."""
        identifier = self._run_id_text(run_id)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT run_id, desired_state, terminal_payload_checksum,
                       terminal_payload_json, mlflow_synced, created_at,
                       last_attempt_at, last_error_json
                FROM run_finalization WHERE run_id = ?
                """,
                [identifier],
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(
                f"run finalization intent for {run_id} was not found"
            )
        payload_json = str(row[3])
        payload = _verify_terminal_payload(payload_json, str(row[2]))
        if payload.get("desired_state") != str(row[1]):
            raise MetadataStoreError(
                "stored finalization intent state does not match payload"
            )
        return FinalizationIntent(
            run_id=UUID(str(row[0])),
            desired_state=RunState(str(row[1])),
            terminal_payload_checksum=str(row[2]),
            terminal_payload_json=payload_json,
            mlflow_synced=bool(row[4]),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[5])),
            last_attempt_at=(
                _require_utc_datetime("last_attempt_at", cast(datetime, row[6]))
                if row[6] is not None
                else None
            ),
            last_error_json=_json_text(row[7]),
        )

    def list_pending_finalization_intents(self) -> tuple[FinalizationIntent, ...]:
        """Return running runs whose terminal intent still needs reconciliation."""
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT f.run_id, f.desired_state, f.terminal_payload_checksum,
                       f.terminal_payload_json, f.mlflow_synced, f.created_at,
                       f.last_attempt_at, f.last_error_json
                FROM run_finalization AS f
                JOIN run AS r ON r.run_id = f.run_id
                WHERE r.state = ? AND r.immutable = FALSE AND f.mlflow_synced = FALSE
                ORDER BY f.created_at ASC, f.run_id ASC
                """,
                [RunState.RUNNING.value],
            ).fetchall()
        return tuple(self._finalization_record(row) for row in rows)

    # Compatibility spelling used by restart/reconciliation callers.
    pending_finalizations = list_pending_finalization_intents

    @staticmethod
    def _finalization_record(row: Sequence[object]) -> FinalizationIntent:
        payload_json = str(row[3])
        payload = _verify_terminal_payload(payload_json, str(row[2]))
        if payload.get("desired_state") != str(row[1]):
            raise MetadataStoreError(
                "stored finalization intent state does not match payload"
            )
        return FinalizationIntent(
            run_id=UUID(str(row[0])),
            desired_state=RunState(str(row[1])),
            terminal_payload_checksum=str(row[2]),
            terminal_payload_json=payload_json,
            mlflow_synced=bool(row[4]),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[5])),
            last_attempt_at=(
                _require_utc_datetime("last_attempt_at", cast(datetime, row[6]))
                if row[6] is not None
                else None
            ),
            last_error_json=_json_text(row[7]),
        )

    def get_run(self, run_id: UUID) -> RunRecord:
        """Load one run by its operational UUID."""
        identifier = self._run_id_text(run_id)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                self._run_select() + " WHERE run_id = ?", [identifier]
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(f"run {run_id} was not found")
        return self._run_record(row)

    def search_runs(self, query: RunQuery) -> RunPage:
        """Search only the run index with deterministic filtering and page bounds.

        The projection deliberately stays on ``run``.  Metrics, manifests,
        artifacts, and scientific tables are loaded by inspection/comparison
        services only after a caller has selected a run.
        """
        if not isinstance(query, RunQuery):
            raise TypeError("query must be a RunQuery")
        where, parameters = self._run_filters(query)
        offset = query.page * query.page_size
        with self._lock:
            self._ensure_open()
            count_row = self._connection.execute(
                "SELECT COUNT(*) FROM run" + where, parameters
            ).fetchone()
            # Avoid passing an unrepresentable OFFSET to DuckDB.  This keeps
            # pagination total and deterministic for any non-negative page.
            if offset > _MAX_SQL_OFFSET:
                rows: list[tuple[object, ...]] = []
            else:
                rows = self._connection.execute(
                    self._run_select()
                    + where
                    + " ORDER BY created_at DESC, run_id ASC LIMIT ? OFFSET ?",
                    [*parameters, query.page_size, offset],
                ).fetchall()
        return RunPage(
            records=tuple(self._run_record(row) for row in rows),
            total_count=int(count_row[0]),
            page=query.page,
            page_size=query.page_size,
        )

    def create_job(self, update: ProgressUpdate, *, updated_at: datetime) -> JobRecord:
        """Insert a not-started job; later progress is guarded by legal transitions."""
        if not isinstance(update, ProgressUpdate):
            raise TypeError("update must be a ProgressUpdate")
        if update.state is not JobState.NOT_STARTED:
            raise ValueError("new jobs must begin in not_started state")
        timestamp = _require_utc_datetime("updated_at", updated_at)
        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO job(
                  job_id, operation, state, stage, completed_units, total_units,
                  started_at, updated_at, ended_at, warnings_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(update.job_id),
                    JobOperation(update.operation).value,
                    JobState(update.state).value,
                    JobStage(update.stage).value,
                    update.completed_units,
                    update.total_units,
                    None,
                    timestamp,
                    None,
                    _canonical_json(list(update.warnings)),
                    None,
                ],
            )
        return self.get_job(update.job_id)

    def update_job(
        self,
        update: ProgressUpdate,
        *,
        updated_at: datetime,
        errors: tuple[ActionableError, ...] = (),
    ) -> JobRecord:
        """Persist legal progress/state changes and immediately retain terminal facts."""
        if not isinstance(update, ProgressUpdate):
            raise TypeError("update must be a ProgressUpdate")
        if not isinstance(errors, tuple) or any(
            not isinstance(error, ActionableError) for error in errors
        ):
            raise TypeError("errors must be a tuple of ActionableError values")
        timestamp = _require_utc_datetime("updated_at", updated_at)
        identifier = str(update.job_id)
        with self.transaction():
            row = self._connection.execute(
                "SELECT operation, state, started_at FROM job WHERE job_id = ?",
                [identifier],
            ).fetchone()
            if row is None:
                raise MetadataNotFoundError(f"job {update.job_id} was not found")
            operation = JobOperation(str(row[0]))
            current = JobState(str(row[1]))
            if operation is not update.operation:
                raise IllegalMetadataTransitionError("job operation cannot change")
            if current is update.state:
                if current is not JobState.RUNNING:
                    raise IllegalMetadataTransitionError(
                        "terminal job progress cannot change"
                    )
            else:
                try:
                    require_legal_job_transition(
                        current, update.state, operation=operation
                    )
                except ValueError as error:
                    raise IllegalMetadataTransitionError(str(error)) from error
            started_at = row[2]
            if update.state is JobState.RUNNING and started_at is None:
                started_at = timestamp
            ended_at = timestamp if update.state is not JobState.RUNNING else None
            self._connection.execute(
                """
                UPDATE job
                SET state = ?, stage = ?, completed_units = ?, total_units = ?,
                    started_at = ?, updated_at = ?, ended_at = ?, warnings_json = ?,
                    error_json = ?
                WHERE job_id = ?
                """,
                [
                    JobState(update.state).value,
                    JobStage(update.stage).value,
                    update.completed_units,
                    update.total_units,
                    started_at,
                    timestamp,
                    ended_at,
                    _canonical_json(list(update.warnings)),
                    _errors_json(errors),
                    identifier,
                ],
            )
        return self.get_job(update.job_id)

    def append_job_event(
        self,
        job_id: UUID,
        *,
        occurred_at: datetime,
        level: str,
        stage: JobStage | str,
        message: str,
        context: Mapping[str, object],
    ) -> JobEvent:
        """Append one sequenced sanitized event without permitting event rewrites."""
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        timestamp = _require_utc_datetime("occurred_at", occurred_at)
        normalized_stage = JobStage(stage)
        normalized_level = _require_text("level", level)
        normalized_message = _require_text("message", message)
        with self.transaction():
            found = self._connection.execute(
                "SELECT 1 FROM job WHERE job_id = ?", [str(job_id)]
            ).fetchone()
            if found is None:
                raise MetadataNotFoundError(f"job {job_id} was not found")
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM job_event WHERE job_id = ?",
                [str(job_id)],
            ).fetchone()
            sequence = int(row[0])
            context_json = _canonical_json(dict(context))
            self._connection.execute(
                """
                INSERT INTO job_event(job_id, sequence, occurred_at, level, stage, message, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(job_id),
                    sequence,
                    timestamp,
                    normalized_level,
                    normalized_stage.value,
                    normalized_message,
                    context_json,
                ],
            )
        return JobEvent(
            job_id=job_id,
            sequence=sequence,
            occurred_at=timestamp,
            level=normalized_level,
            stage=normalized_stage,
            message=normalized_message,
            context_json=context_json,
        )

    def get_job(self, job_id: UUID) -> JobRecord:
        """Load the latest persisted job state for inspection after a refresh."""
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT job_id, operation, state, stage, completed_units, total_units,
                       started_at, updated_at, ended_at, warnings_json, error_json
                FROM job WHERE job_id = ?
                """,
                [str(job_id)],
            ).fetchone()
        if row is None:
            raise MetadataNotFoundError(f"job {job_id} was not found")
        return JobRecord(
            job_id=UUID(str(row[0])),
            operation=JobOperation(str(row[1])),
            state=JobState(str(row[2])),
            stage=JobStage(str(row[3])),
            completed_units=int(row[4]),
            total_units=int(row[5]) if row[5] is not None else None,
            started_at=(
                _require_utc_datetime("started_at", cast(datetime, row[6]))
                if row[6] is not None
                else None
            ),
            updated_at=_require_utc_datetime("updated_at", cast(datetime, row[7])),
            ended_at=(
                _require_utc_datetime("ended_at", cast(datetime, row[8]))
                if row[8] is not None
                else None
            ),
            warnings=_json_tuple(row[9]),
            error_json=_json_text(row[10]),
        )

    def list_job_events(self, job_id: UUID) -> tuple[JobEvent, ...]:
        """Return all job events in total sequence order."""
        if not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID")
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT job_id, sequence, occurred_at, level, stage, message, context_json
                FROM job_event WHERE job_id = ? ORDER BY sequence ASC
                """,
                [str(job_id)],
            ).fetchall()
        return tuple(
            JobEvent(
                job_id=UUID(str(row[0])),
                sequence=int(row[1]),
                occurred_at=_require_utc_datetime(
                    "occurred_at", cast(datetime, row[2])
                ),
                level=str(row[3]),
                stage=JobStage(str(row[4])),
                message=str(row[5]),
                context_json=str(row[6]),
            )
            for row in rows
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise MetadataStoreError("metadata store is closed")

    def _record_data_object(
        self, reference: ContentAddressedObjectRef, created: datetime
    ) -> None:
        existing = self._connection.execute(
            """
            SELECT object_kind, relative_uri, schema_version, symbol, session_year,
                   byte_size, row_count
            FROM data_object WHERE checksum = ?
            """,
            [reference.checksum],
        ).fetchone()
        expected = (
            reference.object_kind.value,
            reference.relative_uri,
            reference.schema_version,
            reference.symbol,
            reference.session_year,
            reference.byte_size,
            reference.row_count,
        )
        if existing is None:
            collision = self._connection.execute(
                "SELECT checksum FROM data_object WHERE relative_uri = ?",
                [reference.relative_uri],
            ).fetchone()
            if collision is not None:
                raise ImmutableMetadataError(
                    "logical object URI already indexes different immutable bytes"
                )
            self._connection.execute(
                """
                INSERT INTO data_object(
                  checksum, object_kind, relative_uri, schema_version, symbol,
                  session_year, byte_size, row_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [reference.checksum, *expected, created],
            )
        elif tuple(existing) != expected:
            raise ImmutableMetadataError(
                "object checksum already indexes different immutable metadata"
            )

    def _verify_snapshot_references(
        self,
        manifest: SnapshotManifest,
        statuses: tuple[SymbolValidationSummary, ...],
    ) -> None:
        existing_refs = self.list_snapshot_objects(manifest.snapshot_id)
        expected_refs = tuple(
            SnapshotObjectRecord(
                snapshot_id=manifest.snapshot_id,
                checksum=reference.checksum,
                role=reference.object_kind.value,
                symbol=reference.symbol,
                session_year=reference.session_year,
                ordinal=ordinal,
            )
            for ordinal, reference in enumerate(manifest.content_identity.objects)
        )
        if existing_refs != tuple(
            sorted(expected_refs, key=lambda item: (item.role, item.ordinal))
        ):
            raise ImmutableMetadataError(
                "snapshot ID already indexes different object references"
            )
        if statuses:
            rows = self._connection.execute(
                """
                SELECT symbol, accepted_count, gap_count, quarantine_count, stale,
                       lag_sessions, failed, retained_parent_coverage, comparison_ready
                FROM snapshot_symbol_status WHERE snapshot_id = ? ORDER BY symbol ASC
                """,
                [manifest.snapshot_id],
            ).fetchall()
            expected = tuple(
                (
                    item.symbol,
                    item.accepted_count,
                    item.gap_count,
                    item.quarantined_count,
                    item.stale,
                    item.staleness_lag_sessions,
                    item.failed,
                    item.retained_parent_coverage,
                    item.comparison_ready,
                )
                for item in statuses
            )
            if tuple(rows) != expected:
                raise ImmutableMetadataError("snapshot symbol status is insert-only")

    def _insert_snapshot_symbol_status(
        self, snapshot_id: str, status: SymbolValidationSummary
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO snapshot_symbol_status(
              snapshot_id, symbol, accepted_count, gap_count, quarantine_count,
              stale, lag_sessions, failed, retained_parent_coverage, comparison_ready
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_id,
                status.symbol,
                status.accepted_count,
                status.gap_count,
                status.quarantined_count,
                status.stale,
                status.staleness_lag_sessions,
                status.failed,
                status.retained_parent_coverage,
                status.comparison_ready,
            ],
        )

    def _snapshot_record(self, row: Sequence[object]) -> SnapshotRecord:
        return SnapshotRecord(
            snapshot_id=str(row[0]),
            parent_snapshot_id=_json_text(row[1]),
            manifest_checksum=str(row[2]),
            manifest_uri=str(row[3]),
            content_identity_checksum=str(row[4]),
            configuration_checksum=str(row[5]),
            provider=str(row[6]),
            requested_start=cast(date, row[7]),
            requested_end=cast(date, row[8]),
            covered_start=cast(date, row[9]) if row[9] is not None else None,
            covered_end=cast(date, row[10]) if row[10] is not None else None,
            universe=_json_tuple(row[11]),
            benchmark_symbol=str(row[12]),
            comparison_ready=bool(row[13]),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[14])),
            availability=SnapshotAvailability(str(row[15])),
        )

    @staticmethod
    def _run_select() -> str:
        """Return the bounded metadata projection used by discovery and get_run."""
        return """
            SELECT run_id, mlflow_run_id, snapshot_id, state, strategy_id,
                   evaluation_start, evaluation_end, universe_json, config_checksum,
                   environment_checksum, manifest_checksum, manifest_uri, created_at,
                   started_at, ended_at, error_json, immutable
            FROM run
        """

    @staticmethod
    def _run_filters(query: RunQuery) -> tuple[str, list[object]]:
        """Build parameterized filters over indexed run columns only."""
        clauses: list[str] = []
        parameters: list[object] = []
        if query.run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(str(query.run_id))
        if query.snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            parameters.append(query.snapshot_id)
        if query.strategy_id is not None:
            clauses.append("strategy_id = ?")
            parameters.append(query.strategy_id)
        if query.universe is not None:
            # universe_key retains order intentionally: an ordered configured
            # universe is part of the run's scientific input identity.
            clauses.append("universe_key = ?")
            parameters.append(_canonical_json(list(query.universe)))
        if query.evaluation_start is not None:
            clauses.append("evaluation_start = ?")
            parameters.append(query.evaluation_start)
        if query.evaluation_end is not None:
            clauses.append("evaluation_end = ?")
            parameters.append(query.evaluation_end)
        if query.state is not None:
            clauses.append("state = ?")
            parameters.append(query.state.value)
        if query.created_from is not None:
            clauses.append("created_at >= ?")
            parameters.append(query.created_from)
        if query.created_to is not None:
            clauses.append("created_at <= ?")
            parameters.append(query.created_to)
        return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), parameters

    @staticmethod
    def _run_id_text(run_id: UUID) -> str:
        if not isinstance(run_id, UUID):
            raise TypeError("run_id must be a UUID")
        return str(run_id)

    @staticmethod
    def _run_record(row: Sequence[object]) -> RunRecord:
        return RunRecord(
            run_id=UUID(str(row[0])),
            mlflow_run_id=_json_text(row[1]),
            snapshot_id=str(row[2]),
            state=RunState(str(row[3])),
            strategy_id=str(row[4]),
            evaluation_start=cast(date, row[5]),
            evaluation_end=cast(date, row[6]),
            universe=_json_tuple(row[7]),
            configuration_checksum=str(row[8]),
            environment_checksum=str(row[9]),
            manifest_checksum=_json_text(row[10]),
            manifest_uri=_json_text(row[11]),
            created_at=_require_utc_datetime("created_at", cast(datetime, row[12])),
            started_at=_require_utc_datetime("started_at", cast(datetime, row[13])),
            ended_at=(
                _require_utc_datetime("ended_at", cast(datetime, row[14]))
                if row[14] is not None
                else None
            ),
            error_json=_json_text(row[15]),
            immutable=bool(row[16]),
        )

    @staticmethod
    def _require_mutable_running_run(
        identifier: str, row: Sequence[object] | None
    ) -> None:
        if row is None:
            raise MetadataNotFoundError(f"run {identifier} was not found")
        state = RunState(str(row[0]))
        immutable = bool(row[1])
        if immutable or state is not RunState.RUNNING:
            raise ImmutableMetadataError(
                "terminal runs are immutable; create a new Run ID"
            )

    def _verify_terminal_artifacts(
        self, artifacts: tuple[RunArtifactLink, ...]
    ) -> None:
        for artifact in artifacts:
            row = self._connection.execute(
                "SELECT availability FROM artifact WHERE checksum = ?",
                [artifact.checksum],
            ).fetchone()
            if row is None:
                raise MetadataNotFoundError(
                    f"artifact {artifact.checksum} was not indexed"
                )
            if SnapshotAvailability(str(row[0])) is not SnapshotAvailability.AVAILABLE:
                raise ImmutableMetadataError(
                    "terminal run cannot reference unavailable artifact"
                )

    def _insert_metric(self, run_id: str, scope: str, metric: MetricValue) -> None:
        numeric_value = None if metric.value is None else float(metric.value)
        self._connection.execute(
            """
            INSERT INTO run_metric(run_id, scope, metric_name, metric_value, null_reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                run_id,
                scope,
                MetricName(metric.name).value,
                numeric_value,
                MetricNullReason(metric.null_reason).value
                if metric.null_reason is not None
                else None,
            ],
        )


DuckDBMetadataRepository = DuckDBMetadataStore
"""Compatibility name for consumers that call the adapter a repository."""


__all__ = [
    "ArtifactRecord",
    "DuckDBMetadataRepository",
    "DuckDBMetadataStore",
    "FinalizationIntent",
    "IllegalMetadataTransitionError",
    "ImmutableMetadataError",
    "IngestionOperationRecord",
    "IngestionOperationStatus",
    "JobEvent",
    "JobRecord",
    "MetadataNotFoundError",
    "MetadataStoreError",
    "RunArtifactLink",
    "RunFinalization",
    "RunPage",
    "RunQuery",
    "RunRecord",
    "SCHEMA_VERSION",
    "SnapshotAvailability",
    "SnapshotObjectRecord",
    "SnapshotRecord",
]
