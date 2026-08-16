"""Safe, immutable error, result, and disclosure domain primitives.

Application boundaries return :class:`Ok` or :class:`Err` values.  They never
return a raw exception, traceback, response body, or transport object to a
caller.  Infrastructure code can retain sanitized diagnostics locally, while
these value objects remain safe to persist or display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Final, Generic, Protocol, Self, TypeAlias, TypeVar, runtime_checkable


class ErrorCategory(StrEnum):
    """Stable, display-safe categories for application-boundary failures."""

    CONFIGURATION_SYNTAX = "configuration.syntax"
    CONFIGURATION_DUPLICATE_KEY = "configuration.duplicate_key"
    CONFIGURATION_UNKNOWN_KEY = "configuration.unknown_key"
    CONFIGURATION_INVALID_VALUE = "configuration.invalid_value"
    PROVIDER_RETRYABLE = "provider.retryable"
    PROVIDER_TERMINAL = "provider.terminal"
    NORMALIZATION_POLICY = "normalization.policy"
    VALIDATION_ROW = "validation.row"
    VALIDATION_DUPLICATE_CONFLICT = "validation.duplicate_conflict"
    VALIDATION_GAP = "validation.gap"
    VALIDATION_STALE = "validation.stale"
    STORAGE_IO = "storage.io"
    STORAGE_ATOMICITY = "storage.atomicity"
    INTEGRITY_CHECKSUM = "integrity.checksum"
    SNAPSHOT_NOT_READY = "snapshot.not_ready"
    BACKTEST_EXECUTION = "backtest.execution"
    BACKTEST_INVARIANT = "backtest.invariant"
    EXPERIMENT_RECORDING = "experiment.recording"
    COMPARISON_SELECTION = "comparison.selection"
    SECURITY_SECRET_DETECTED = "security.secret_detected"
    INTERNAL_UNEXPECTED = "internal.unexpected"


class ValidationReason(StrEnum):
    """Stable rule identifiers emitted by deterministic validation."""

    SYMBOL_NONEMPTY = "symbol.nonempty"
    SESSION_XNYS = "session.xnys"
    OHLC_FINITE_POSITIVE = "ohlc.finite_positive"
    VOLUME_FINITE_NONNEGATIVE = "volume.finite_nonnegative"
    HIGH_ENVELOPE = "high.envelope"
    LOW_ENVELOPE = "low.envelope"
    RAW_LINEAGE = "lineage.raw_record"


class QuarantineReason(StrEnum):
    """Stable top-level reasons for records excluded from accepted data."""

    NON_SESSION = "session.non_xnys"
    NORMALIZATION_POLICY = "normalization.policy"
    VALIDATION_ROW = "validation.row"
    DUPLICATE_CONFLICT = "duplicate.conflict"
    MISSING_RAW_LINEAGE = "lineage.raw_record"


class ProviderFailureKind(StrEnum):
    """Whether unchanged provider input is eligible for a retry."""

    RETRYABLE = "retryable"
    TERMINAL = "terminal"


class ProviderFailureReason(StrEnum):
    """Stable provider-failure reasons independent of adapter exceptions."""

    TIMEOUT = "timeout"
    CONNECTION_RESET = "connection_reset"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    INVALID_SYMBOL = "invalid_symbol"
    EMPTY_RESPONSE = "empty_response"
    SCHEMA_INVALID = "schema_invalid"
    CLIENT_ERROR = "client_error"
    UNEXPECTED = "unexpected"


class JobReason(StrEnum):
    """Stable operational reason codes; job states are defined separately."""

    COMPLETED = "completed"
    PUBLISHED = "published"
    PARTIAL_PROVIDER_FAILURE = "partial_provider_failure"
    PARTIAL_QUARANTINE = "partial_quarantine"
    PARTIAL_DATA_GAP = "partial_data_gap"
    PARTIAL_STALE_DATA = "partial_stale_data"
    REQUIRED_OUTPUT_UNPUBLISHED = "required_output_unpublished"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


_CHECKSUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_CONTEXT_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("field_path", "field"),
    ("symbol", "symbol"),
    ("session", "session"),
    ("checksum", "checksum"),
    ("correlation_id", "correlation"),
)


def _clean_required_text(name: str, value: str) -> str:
    """Normalize a display-safe single-line text field or reject it."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _clean_optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _clean_required_text(name, value)


@dataclass(frozen=True)
class ActionableError:
    """A structured error that is safe to return across an application boundary.

    ``message`` and ``corrective_action`` must already be sanitized, concise,
    single-line text.  Raw exceptions are deliberately not represented by a
    field on this type.  Use :meth:`from_unexpected_exception` when converting
    a caught exception at an application boundary.
    """

    operation: str
    category: ErrorCategory
    message: str
    corrective_action: str
    field_path: str | None = None
    symbol: str | None = None
    session: date | None = None
    checksum: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _clean_required_text("operation", self.operation)
        )
        try:
            category = ErrorCategory(self.category)
        except ValueError as error:
            raise ValueError(
                f"unsupported error category: {self.category!r}"
            ) from error
        object.__setattr__(self, "category", category)
        object.__setattr__(
            self, "message", _clean_required_text("message", self.message)
        )
        object.__setattr__(
            self,
            "corrective_action",
            _clean_required_text("corrective_action", self.corrective_action),
        )
        object.__setattr__(
            self, "field_path", _clean_optional_text("field_path", self.field_path)
        )

        symbol = _clean_optional_text("symbol", self.symbol)
        object.__setattr__(self, "symbol", symbol.upper() if symbol else None)

        if self.session is not None and (
            isinstance(self.session, datetime) or not isinstance(self.session, date)
        ):
            raise TypeError("session must be a calendar date")

        checksum = _clean_optional_text("checksum", self.checksum)
        if checksum is not None and _CHECKSUM_PATTERN.fullmatch(checksum) is None:
            raise ValueError(
                "checksum must be a lowercase SHA-256 hexadecimal digest"
            )
        object.__setattr__(self, "checksum", checksum)
        object.__setattr__(
            self,
            "correlation_id",
            _clean_optional_text("correlation_id", self.correlation_id),
        )

    @classmethod
    def from_unexpected_exception(
        cls,
        operation: str,
        exception: BaseException,
        *,
        correlation_id: str | None = None,
    ) -> Self:
        """Convert a caught exception without exposing its type or contents.

        The caller may log a separately sanitized diagnostic locally.  The
        returned value deliberately discards the exception after conversion.
        """

        del exception
        return cls(
            operation=operation,
            category=ErrorCategory.INTERNAL_UNEXPECTED,
            message="The operation encountered an unexpected internal failure.",
            corrective_action=(
                "Retry the operation; if it continues to fail, contact the maintainer "
                "with the correlation ID."
            ),
            correlation_id=correlation_id,
        )

    def sort_key(self) -> tuple[str, str, str, str, str, str, str, str, str]:
        """Return the canonical deterministic ordering key for error collections."""

        return (
            self.operation,
            self.category.value,
            self.field_path or "",
            self.symbol or "",
            self.session.isoformat() if self.session else "",
            self.checksum or "",
            self.correlation_id or "",
            self.message,
            self.corrective_action,
        )

    def format_for_display(self) -> str:
        """Format only structured, sanitized fields in a deterministic order."""

        context_parts = [
            f"{label}={getattr(self, attribute)}"
            for attribute, label in _CONTEXT_LABELS
            if getattr(self, attribute) is not None
        ]
        context = f" ({', '.join(context_parts)})" if context_parts else ""
        return (
            f"{self.operation} [{self.category.value}]{context}: {self.message} "
            f"Corrective action: {self.corrective_action}"
        )

    def __str__(self) -> str:
        return self.format_for_display()


T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Ok(Generic[T]):
    """A successful application result."""

    value: T


@dataclass(frozen=True)
class Err:
    """A non-empty collection of actionable errors.

    Most application failures use the canonical domain ordering. Configuration
    validation is the intentional exception: its diagnostics must follow the
    Pydantic schema and list-index order, so the configuration loader can opt
    into preserving a precomputed deterministic sequence.
    """

    errors: tuple[ActionableError, ...]
    preserve_order: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be an immutable tuple")
        if not isinstance(self.preserve_order, bool):
            raise TypeError("preserve_order must be a boolean")
        if not self.errors:
            raise ValueError("errors must contain at least one ActionableError")
        if any(not isinstance(error, ActionableError) for error in self.errors):
            raise TypeError("Err may contain only ActionableError values")
        if not self.preserve_order:
            object.__setattr__(
                self,
                "errors",
                tuple(sorted(self.errors, key=ActionableError.sort_key)),
            )


Result: TypeAlias = Ok[T] | Err


LIMITATION_DISCLOSURE_VERSION: Final = "limitation-disclosure/v1"
SUPPORTED_LIMITATION_DISCLOSURE_VERSIONS: Final = frozenset(
    {LIMITATION_DISCLOSURE_VERSION}
)
DEFAULT_FREE_SOURCE_NOTICE: Final = (
    "Market data is obtained from a free Yahoo Finance source through yfinance; "
    "availability and terms can change."
)
DEFAULT_UNIVERSE_NOTICE: Final = (
    "The configured universe is an explicit user-supplied ticker list and does not "
    "represent point-in-time index membership."
)
DEFAULT_SURVIVORSHIP_NOTICE: Final = (
    "The explicit universe can contain survivorship and selection bias because it does "
    "not reconstruct delistings, mergers, or historical membership."
)
DEFAULT_DATA_QUALITY_NOTICE: Final = (
    "Free-provider records can be incomplete, corrected, unavailable, or inconsistent; "
    "validation and provenance do not guarantee source truth."
)
DEFAULT_COST_ASSUMPTIONS: Final = "Configured commission and adverse-slippage assumptions are applied to simulated fills."
DEFAULT_EXECUTION_ASSUMPTIONS: Final = (
    "Configured execution uses long-only whole-share orders at the next eligible session "
    "open and is a research simulation, not live trading."
)


@dataclass(frozen=True)
class LimitationDisclosure:
    """Versioned limitations that must accompany data, snapshot, run, and comparison DTOs."""

    version: str = LIMITATION_DISCLOSURE_VERSION
    free_source_notice: str = DEFAULT_FREE_SOURCE_NOTICE
    universe_notice: str = DEFAULT_UNIVERSE_NOTICE
    survivorship_notice: str = DEFAULT_SURVIVORSHIP_NOTICE
    data_quality_notice: str = DEFAULT_DATA_QUALITY_NOTICE
    data_failures: tuple[ActionableError, ...] = ()
    cost_assumptions: str = DEFAULT_COST_ASSUMPTIONS
    execution_assumptions: str = DEFAULT_EXECUTION_ASSUMPTIONS

    def __post_init__(self) -> None:
        version = _clean_required_text("version", self.version)
        if version not in SUPPORTED_LIMITATION_DISCLOSURE_VERSIONS:
            raise ValueError(f"unsupported limitation disclosure version: {version}")
        object.__setattr__(self, "version", version)

        for field_name in (
            "free_source_notice",
            "universe_notice",
            "survivorship_notice",
            "data_quality_notice",
            "cost_assumptions",
            "execution_assumptions",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_required_text(field_name, getattr(self, field_name)),
            )

        if not isinstance(self.data_failures, tuple):
            raise TypeError("data_failures must be an immutable tuple")
        if any(not isinstance(error, ActionableError) for error in self.data_failures):
            raise TypeError("data_failures may contain only ActionableError values")
        object.__setattr__(
            self,
            "data_failures",
            tuple(sorted(self.data_failures, key=ActionableError.sort_key)),
        )

    @classmethod
    def current(cls, *, data_failures: tuple[ActionableError, ...] = ()) -> Self:
        """Build the current disclosure version with platform-approved wording."""

        return cls(data_failures=data_failures)

    def lines(self) -> tuple[str, ...]:
        """Return visible disclosure lines in their stable presentation order."""

        failure_line = (
            "Recorded data failures: none recorded."
            if not self.data_failures
            else "Recorded data failures: "
            + " | ".join(error.format_for_display() for error in self.data_failures)
        )
        return (
            f"Limitation disclosure version: {self.version}",
            self.free_source_notice,
            self.universe_notice,
            self.survivorship_notice,
            self.data_quality_notice,
            failure_line,
            self.cost_assumptions,
            self.execution_assumptions,
        )

    def format_for_display(self) -> str:
        """Render the complete visible disclosure without non-deterministic formatting."""

        return "\n".join(self.lines())


@runtime_checkable
class DisclosureCarrier(Protocol):
    """Structural contract for data, snapshot, run, and comparison DTOs."""

    @property
    def limitation_disclosure(self) -> LimitationDisclosure:
        """Return the disclosure required for the DTO's visible representation."""

        ...


__all__ = [
    "ActionableError",
    "DEFAULT_COST_ASSUMPTIONS",
    "DEFAULT_DATA_QUALITY_NOTICE",
    "DEFAULT_EXECUTION_ASSUMPTIONS",
    "DEFAULT_FREE_SOURCE_NOTICE",
    "DEFAULT_SURVIVORSHIP_NOTICE",
    "DEFAULT_UNIVERSE_NOTICE",
    "DisclosureCarrier",
    "Err",
    "ErrorCategory",
    "JobReason",
    "LIMITATION_DISCLOSURE_VERSION",
    "LimitationDisclosure",
    "Ok",
    "ProviderFailureKind",
    "ProviderFailureReason",
    "QuarantineReason",
    "Result",
    "SUPPORTED_LIMITATION_DISCLOSURE_VERSIONS",
    "ValidationReason",
]
