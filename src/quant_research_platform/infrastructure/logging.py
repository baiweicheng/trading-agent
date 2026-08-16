"""Sanitized, local-only structured JSON Lines diagnostics.

The logger is intentionally a small infrastructure adapter.  It writes one
canonical JSON document per line, never transmits telemetry, and uses the
configuration redactor before a message, context value, or exception is made
durable.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Final
from uuid import UUID

from pydantic import BaseModel

from ..config.models import UnresolvedSecret
from ..config.serializer import REDACTION_MARKER, Redactor
from ..domain.canonical import canonical_json_text
from ..domain.execution import JobStage

_ALLOWED_LEVELS: Final[frozenset[str]] = frozenset(
    {"debug", "info", "warning", "error"}
)


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _optional_text(name: str, value: str | None) -> str | None:
    return None if value is None else _required_text(name, value)


def _utc_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("occurred_at must be an aware datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("occurred_at must be in UTC")
    return value


def _json_safe(value: object) -> object:
    """Return an allowlisted canonical-JSON value without falling back to ``repr``."""

    if value is None or isinstance(
        value, (bool, int, float, str, Decimal, date, datetime)
    ):
        return value
    if isinstance(value, UUID | Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, UnresolvedSecret):
        return REDACTION_MARKER
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("structured log context keys must be strings")
            safe_mapping[key] = _json_safe(item)
        return safe_mapping
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # Do not stringify arbitrary values: __repr__ may contain credentials.
    return "[UNAVAILABLE]"


@dataclass(frozen=True, slots=True)
class StructuredLogRecord:
    """One durable diagnostic record returned after a successful JSONL append."""

    occurred_at: datetime
    level: str
    operation: str
    correlation_id: str
    message: str
    job_id: str | None
    run_id: str | None
    stage: str | None
    category: str | None
    exception_type: str | None
    context_json: str

    def payload(self) -> dict[str, object]:
        """Return the exact stable JSON object written to the log stream."""

        return {
            "category": self.category,
            "context": json.loads(self.context_json),
            "correlation_id": self.correlation_id,
            "exception_type": self.exception_type,
            "job_id": self.job_id,
            "level": self.level,
            "message": self.message,
            "occurred_at": self.occurred_at,
            "operation": self.operation,
            "run_id": self.run_id,
            "stage": self.stage,
        }


class StructuredJsonlLogger:
    """Append sanitized diagnostics to a local fsync'd UTF-8 JSONL file."""

    def __init__(
        self,
        path: Path | str,
        *,
        redactor: Redactor | None = None,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = Path(path)
        self._redactor = redactor or Redactor()
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Return the destination JSONL path without opening the file."""

        return self._path

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
    ) -> StructuredLogRecord:
        """Sanitize and durably append one diagnostic JSON line.

        Exception information is retained only after redaction.  Unsupported
        context values are replaced by ``[UNAVAILABLE]`` rather than serialized
        through potentially unsafe ``repr`` implementations.
        """

        normalized_level = _required_text("level", level).lower()
        if normalized_level not in _ALLOWED_LEVELS:
            raise ValueError(f"unsupported log level: {level!r}")
        normalized_operation = self._redactor.redact_text(
            _required_text("operation", operation)
        )
        normalized_correlation = self._redactor.redact_text(
            _required_text("correlation_id", correlation_id)
        )
        normalized_message = self._redactor.redact_text(
            _required_text("message", message)
        )
        normalized_run_id = self._sanitize_optional("run_id", run_id)
        normalized_category = self._sanitize_optional("category", category)
        normalized_stage = (
            None if stage is None else self._redactor.redact_text(JobStage(stage).value)
        )
        normalized_job_id = str(job_id) if job_id is not None else None
        if job_id is not None and not isinstance(job_id, UUID):
            raise TypeError("job_id must be a UUID or None")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping or None")

        raw_context: dict[str, object] = dict(context or {})
        exception_type: str | None = None
        if exception is not None:
            exception_type = self._redactor.redact_text(type(exception).__name__)
            raw_context["exception_message"] = str(exception)
            raw_context["traceback"] = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        sanitized_context = self._redactor.redact_structured(raw_context)
        if not isinstance(sanitized_context, Mapping):
            raise TypeError("sanitized log context must remain a mapping")
        safe_context = _json_safe(sanitized_context)
        if not isinstance(safe_context, Mapping):
            raise AssertionError("log context conversion must retain mapping shape")
        context_json = canonical_json_text(safe_context).rstrip("\n")
        occurred_at = _utc_timestamp(self._utc_now())

        record = StructuredLogRecord(
            occurred_at=occurred_at,
            level=normalized_level,
            operation=normalized_operation,
            correlation_id=normalized_correlation,
            message=normalized_message,
            job_id=normalized_job_id,
            run_id=normalized_run_id,
            stage=normalized_stage,
            category=normalized_category,
            exception_type=exception_type,
            context_json=context_json,
        )
        encoded = canonical_json_text(record.payload())
        self._append(encoded)
        return record

    def _sanitize_optional(self, name: str, value: str | None) -> str | None:
        text = _optional_text(name, value)
        return None if text is None else self._redactor.redact_text(text)

    def _append(self, encoded: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())


__all__ = ["StructuredJsonlLogger", "StructuredLogRecord"]
