"""Synchronous local job orchestration with durable, redacted progress.

Jobs remain operational records: they are created in ``not_started``, move once to
``running``, then end exactly once.  The manager intentionally owns no queue or
worker process; callers execute their work synchronously through :meth:`execute`
or an explicitly managed :class:`SynchronousJob`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Generic, Protocol, TypeVar
from uuid import UUID, uuid4

from ..config.serializer import Redactor
from ..domain.errors import ActionableError, Err, Ok, Result
from ..domain.execution import (
    JobOperation,
    JobStage,
    JobState,
    ProgressUpdate,
    require_legal_job_transition,
)

T = TypeVar("T")

_PROGRESS_INTERVAL_SECONDS = Decimal("0.25")
"""At most four non-terminal persisted progress updates per second."""


class JobClock(Protocol):
    """Injectable wall/monotonic clock pair for jobs and deterministic tests."""

    def utc_now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""

    def monotonic_seconds(self) -> Decimal:
        """Return a non-decreasing monotonic elapsed-time source."""


class JobMetadataRepository(Protocol):
    """The narrow durable job-store surface used by this application service."""

    def create_job(self, update: ProgressUpdate, *, updated_at: datetime) -> object:
        """Persist one not-started job."""

    def update_job(
        self,
        update: ProgressUpdate,
        *,
        updated_at: datetime,
        errors: tuple[ActionableError, ...] = (),
    ) -> object:
        """Persist a legal current/terminal state update."""

    def append_job_event(
        self,
        job_id: UUID,
        *,
        occurred_at: datetime,
        level: str,
        stage: JobStage | str,
        message: str,
        context: Mapping[str, object],
    ) -> object:
        """Append one immutable, sanitized job event."""


class DiagnosticLogger(Protocol):
    """Structured local-diagnostics sink; implemented by infrastructure logging."""

    def write(
        self,
        *,
        level: str,
        operation: str,
        correlation_id: str,
        message: str,
        job_id: UUID | None = None,
        run_id: str | None = None,
        stage: JobStage | str | None = None,
        category: str | None = None,
        context: Mapping[str, object] | None = None,
        exception: BaseException | None = None,
    ) -> object:
        """Persist one sanitized JSONL diagnostic."""


@dataclass(frozen=True, slots=True)
class JobOutcome(Generic[T]):
    """The value and terminal operational state produced by one synchronous job."""

    job_id: UUID
    correlation_id: str
    state: JobState
    value: T

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, UUID):
            raise TypeError("job_id must be a UUID")
        if not isinstance(self.correlation_id, str) or not self.correlation_id.strip():
            raise ValueError("correlation_id must be a non-empty string")
        if self.state not in {JobState.SUCCEEDED, JobState.PARTIALLY_SUCCEEDED}:
            raise ValueError("JobOutcome requires a successful terminal job state")


class SystemJobClock:
    """Production clock implementation; tests provide a deterministic replacement."""

    def utc_now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic_seconds(self) -> Decimal:
        from time import monotonic

        return Decimal(str(monotonic()))


def _sanitize_error(
    error: ActionableError,
    *,
    redactor: Redactor,
    correlation_id: str,
) -> ActionableError:
    """Apply redaction and guarantee a correlation ID on persisted errors."""

    sanitized = redactor.redact_error(error)
    if not isinstance(sanitized, ActionableError):
        raise TypeError("redaction must retain an ActionableError instance")
    if sanitized.correlation_id is None:
        sanitized = replace(sanitized, correlation_id=correlation_id)
    return sanitized


class SynchronousJob:
    """One mutable in-process job controller backed by immutable progress values."""

    def __init__(
        self,
        *,
        repository: JobMetadataRepository,
        diagnostics: DiagnosticLogger,
        redactor: Redactor,
        clock: JobClock,
        job_id: UUID,
        operation: JobOperation,
        total_units: int | None,
        progress_callback: Callable[[ProgressUpdate], None] | None,
    ) -> None:
        self._repository = repository
        self._diagnostics = diagnostics
        self._redactor = redactor
        self._clock = clock
        self._job_id = job_id
        self._operation = operation
        self._total_units = total_units
        self._progress_callback = progress_callback
        self._correlation_id = str(job_id)
        self._state = JobState.NOT_STARTED
        self._stage = JobStage.NOT_STARTED
        self._completed_units = 0
        self._warnings: list[str] = []
        self._started_monotonic: Decimal | None = None
        self._last_persisted_monotonic: Decimal | None = None

        created = self._update()
        self._repository.create_job(created, updated_at=self._utc_now())
        self._append_event(
            level="info",
            message="Job created.",
            update=created,
            context={},
        )
        self._emit(created)

    @property
    def job_id(self) -> UUID:
        """Return the opaque operational identifier."""

        return self._job_id

    @property
    def correlation_id(self) -> str:
        """Return the diagnostic correlation identifier for this job."""

        return self._correlation_id

    @property
    def state(self) -> JobState:
        """Return the locally tracked durable lifecycle state."""

        return self._state

    @property
    def current_progress(self) -> ProgressUpdate:
        """Return the latest sanitized progress view without forcing a write."""

        return self._update()

    def start(self, *, stage: JobStage | str = JobStage.PREPARING) -> ProgressUpdate:
        """Persist the required ``not_started -> running`` transition immediately."""

        normalized_stage = JobStage(stage)
        if normalized_stage in {
            JobStage.NOT_STARTED,
            JobStage.COMPLETED,
            JobStage.FAILED,
        }:
            raise ValueError("a running job must start at an active stage")
        self._transition(JobState.RUNNING)
        self._stage = normalized_stage
        self._started_monotonic = self._monotonic_now()
        update = self._update()
        self._persist(
            update,
            force=True,
            level="info",
            message="Job started.",
            context={},
        )
        return update

    def report(
        self,
        *,
        stage: JobStage | str,
        completed_units: int,
        total_units: int | None = None,
        warnings: Iterable[str] = (),
        context: Mapping[str, object] | None = None,
    ) -> ProgressUpdate:
        """Record in-memory progress and persist at most four times a second.

        Every supplied warning is redacted and accumulated.  The callback receives
        every update immediately, while metadata persistence is deliberately
        throttled.  A later terminal update always flushes the latest warning set.
        """

        self._require_running()
        normalized_stage = JobStage(stage)
        if normalized_stage in {
            JobStage.NOT_STARTED,
            JobStage.COMPLETED,
            JobStage.FAILED,
        }:
            raise ValueError("running job progress must use an active stage")
        if isinstance(completed_units, bool) or not isinstance(completed_units, int):
            raise TypeError("completed_units must be an integer")
        if completed_units < self._completed_units:
            raise ValueError("completed_units must not decrease")

        if total_units is not None:
            if isinstance(total_units, bool) or not isinstance(total_units, int):
                raise TypeError("total_units must be an integer or None")
            if total_units < completed_units:
                raise ValueError("total_units must not be below completed_units")
            if self._total_units is not None and total_units != self._total_units:
                raise ValueError("total_units cannot change after it is established")
            self._total_units = total_units
        if self._total_units is not None and completed_units > self._total_units:
            raise ValueError("completed_units must not exceed total_units")

        self._stage = normalized_stage
        self._completed_units = completed_units
        sanitized_context = self._sanitize_context(context)
        warnings_added = self._accumulate_warnings(warnings)
        update = self._update()
        self._emit(update)

        for warning in warnings_added:
            self._diagnostics.write(
                level="warning",
                operation=self._operation.value,
                correlation_id=self._correlation_id,
                message=warning,
                job_id=self._job_id,
                stage=self._stage,
                category="job.warning",
                context=sanitized_context,
            )

        if self._should_persist():
            self._persist(
                update,
                force=False,
                level="warning" if warnings_added else "info",
                message=(
                    "Job progress warning recorded."
                    if warnings_added
                    else "Job progress updated."
                ),
                context=sanitized_context,
            )
        return update

    def complete(self, *, partially_succeeded: bool = False) -> ProgressUpdate:
        """Immediately persist a successful terminal update and all accumulated warnings."""

        self._require_running()
        target = (
            JobState.PARTIALLY_SUCCEEDED if partially_succeeded else JobState.SUCCEEDED
        )
        self._transition(target)
        self._stage = JobStage.COMPLETED
        update = self._update()
        self._persist(
            update,
            force=True,
            level="warning" if target is JobState.PARTIALLY_SUCCEEDED else "info",
            message=(
                "Job partially succeeded."
                if target is JobState.PARTIALLY_SUCCEEDED
                else "Job succeeded."
            ),
            context={},
        )
        return update

    def fail(self, error: ActionableError) -> ProgressUpdate:
        """Immediately persist a failed terminal update with a safe error payload."""

        self._require_running()
        if not isinstance(error, ActionableError):
            raise TypeError("error must be an ActionableError")
        sanitized_error = _sanitize_error(
            error,
            redactor=self._redactor,
            correlation_id=self._correlation_id,
        )
        self._transition(JobState.FAILED)
        self._stage = JobStage.FAILED
        update = self._update()
        self._persist(
            update,
            force=True,
            level="error",
            message="Job failed.",
            context={"error_category": sanitized_error.category.value},
            errors=(sanitized_error,),
        )
        return update

    def _transition(self, target: JobState) -> None:
        require_legal_job_transition(
            self._state,
            target,
            operation=self._operation,
        )
        self._state = target

    def _require_running(self) -> None:
        if self._state is not JobState.RUNNING:
            raise ValueError("job progress is allowed only while the job is running")

    def _accumulate_warnings(self, warnings: Iterable[str]) -> tuple[str, ...]:
        if isinstance(warnings, str):
            raise TypeError("warnings must be an iterable of strings, not a string")
        sanitized: list[str] = []
        for warning in warnings:
            if not isinstance(warning, str):
                raise TypeError("warnings must contain only strings")
            value = self._redactor.redact_text(warning)
            if not value.strip():
                raise ValueError("warnings must not contain blank text")
            sanitized.append(value)
        self._warnings.extend(sanitized)
        return tuple(sanitized)

    def _sanitize_context(
        self, context: Mapping[str, object] | None
    ) -> Mapping[str, object]:
        if context is None:
            return {}
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        sanitized = self._redactor.redact_structured(context)
        if not isinstance(sanitized, Mapping):
            raise TypeError("sanitized context must remain a mapping")
        return sanitized

    def _update(self) -> ProgressUpdate:
        return ProgressUpdate(
            job_id=self._job_id,
            operation=self._operation,
            state=self._state,
            stage=self._stage,
            completed_units=self._completed_units,
            total_units=self._total_units,
            elapsed_seconds=self._elapsed_seconds(),
            warnings=tuple(self._warnings),
        )

    def _persist(
        self,
        update: ProgressUpdate,
        *,
        force: bool,
        level: str,
        message: str,
        context: Mapping[str, object],
        errors: tuple[ActionableError, ...] = (),
    ) -> None:
        if not force and not self._should_persist():
            return
        occurred_at = self._utc_now()
        self._repository.update_job(update, updated_at=occurred_at, errors=errors)
        self._last_persisted_monotonic = self._monotonic_now()
        self._append_event(
            level=level,
            message=message,
            update=update,
            context=context,
        )
        self._diagnostics.write(
            level=level,
            operation=self._operation.value,
            correlation_id=self._correlation_id,
            message=message,
            job_id=self._job_id,
            stage=update.stage,
            category=(errors[0].category.value if errors else "job.progress"),
            context={
                "progress": update.to_serializable(),
                "context": context,
            },
        )
        self._emit(update)

    def _append_event(
        self,
        *,
        level: str,
        message: str,
        update: ProgressUpdate,
        context: Mapping[str, object],
    ) -> None:
        event_context = {
            "completed_units": update.completed_units,
            "elapsed_seconds": update.elapsed_seconds,
            "total_units": update.total_units,
            "warnings": list(update.warnings),
            **dict(context),
        }
        self._repository.append_job_event(
            self._job_id,
            occurred_at=self._utc_now(),
            level=level,
            stage=update.stage,
            message=self._redactor.redact_text(message),
            context=self._sanitize_context(event_context),
        )

    def _emit(self, update: ProgressUpdate) -> None:
        if self._progress_callback is not None:
            self._progress_callback(update)

    def _should_persist(self) -> bool:
        if self._last_persisted_monotonic is None:
            return True
        return (
            self._monotonic_now() - self._last_persisted_monotonic
            >= _PROGRESS_INTERVAL_SECONDS
        )

    def _elapsed_seconds(self) -> Decimal:
        if self._started_monotonic is None:
            return Decimal("0")
        elapsed = self._monotonic_now() - self._started_monotonic
        return max(Decimal("0"), elapsed)

    def _monotonic_now(self) -> Decimal:
        value = self._clock.monotonic_seconds()
        if not isinstance(value, Decimal) or not value.is_finite():
            raise TypeError("clock.monotonic_seconds() must return a finite Decimal")
        return value

    def _utc_now(self) -> datetime:
        value = self._clock.utc_now()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise TypeError("clock.utc_now() must return an aware datetime")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("clock.utc_now() must return a UTC datetime")
        return value


class SynchronousJobManager:
    """Create, execute, and safely terminalize local synchronous jobs."""

    def __init__(
        self,
        repository: JobMetadataRepository,
        diagnostics: DiagnosticLogger,
        *,
        redactor: Redactor | None = None,
        clock: JobClock | None = None,
        job_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._repository = repository
        self._diagnostics = diagnostics
        self._redactor = redactor or Redactor()
        self._clock = clock or SystemJobClock()
        self._job_id_factory = job_id_factory

    def create(
        self,
        operation: JobOperation | str,
        *,
        total_units: int | None = None,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
        job_id: UUID | None = None,
    ) -> SynchronousJob:
        """Create a durable job in ``not_started`` state without starting work."""

        normalized_operation = JobOperation(operation)
        identifier = job_id or self._job_id_factory()
        if not isinstance(identifier, UUID):
            raise TypeError("job_id_factory must return a UUID")
        return SynchronousJob(
            repository=self._repository,
            diagnostics=self._diagnostics,
            redactor=self._redactor,
            clock=self._clock,
            job_id=identifier,
            operation=normalized_operation,
            total_units=total_units,
            progress_callback=progress_callback,
        )

    def execute(
        self,
        operation: JobOperation | str,
        work: Callable[[SynchronousJob], T],
        *,
        total_units: int | None = None,
        partially_succeeded: bool = False,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
        job_id: UUID | None = None,
    ) -> Result[JobOutcome[T]]:
        """Run local work synchronously and turn unexpected boundary errors into values."""

        normalized_operation = JobOperation(operation)
        identifier = job_id or self._job_id_factory()
        if not isinstance(identifier, UUID):
            raise TypeError("job_id_factory must return a UUID")
        correlation_id = str(identifier)
        session: SynchronousJob | None = None
        try:
            session = self.create(
                normalized_operation,
                total_units=total_units,
                progress_callback=progress_callback,
                job_id=identifier,
            )
            session.start()
            value = work(session)
            terminal = session.complete(partially_succeeded=partially_succeeded)
            return Ok(
                JobOutcome(
                    job_id=identifier,
                    correlation_id=correlation_id,
                    state=terminal.state,
                    value=value,
                )
            )
        except Exception as exception:
            error = ActionableError.from_unexpected_exception(
                normalized_operation.value,
                exception,
                correlation_id=correlation_id,
            )
            self._diagnostics.write(
                level="error",
                operation=normalized_operation.value,
                correlation_id=correlation_id,
                message="Unexpected application-boundary exception.",
                job_id=identifier,
                stage=(session.current_progress.stage if session is not None else None),
                category=error.category.value,
                context={},
                exception=exception,
            )
            if session is not None and session.state is JobState.RUNNING:
                try:
                    session.fail(error)
                except Exception as terminal_exception:
                    self._diagnostics.write(
                        level="error",
                        operation=normalized_operation.value,
                        correlation_id=correlation_id,
                        message="Unable to persist failed job terminal state.",
                        job_id=identifier,
                        stage=JobStage.FAILED,
                        category=error.category.value,
                        context={},
                        exception=terminal_exception,
                    )
            return Err((error,))


__all__ = [
    "DiagnosticLogger",
    "JobClock",
    "JobMetadataRepository",
    "JobOutcome",
    "SynchronousJob",
    "SynchronousJobManager",
    "SystemJobClock",
]
