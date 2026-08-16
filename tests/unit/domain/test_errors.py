"""Focused tests for safe domain error, result, and disclosure primitives."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pytest

from quant_research_platform.domain.errors import (
    LIMITATION_DISCLOSURE_VERSION,
    ActionableError,
    DisclosureCarrier,
    Err,
    ErrorCategory,
    JobReason,
    LimitationDisclosure,
    Ok,
    ProviderFailureKind,
    ProviderFailureReason,
    QuarantineReason,
    ValidationReason,
)


def _error(
    *,
    operation: str = "configuration.resolve",
    category: ErrorCategory = ErrorCategory.CONFIGURATION_INVALID_VALUE,
    message: str = "Requested range start must not be after end.",
    corrective_action: str = "Set the start date no later than the end date.",
    **context: object,
) -> ActionableError:
    return ActionableError(
        operation=operation,
        category=category,
        message=message,
        corrective_action=corrective_action,
        **context,  # type: ignore[arg-type]
    )


def test_actionable_error_is_immutable_and_normalizes_structured_context() -> None:
    error = _error(
        field_path=" data.requested_range.start ",
        symbol=" aapl ",
        session=date(2024, 1, 2),
        checksum="a" * 64,
        correlation_id=" job-42 ",
    )

    assert error.field_path == "data.requested_range.start"
    assert error.symbol == "AAPL"
    assert error.session == date(2024, 1, 2)
    assert error.checksum == "a" * 64
    assert error.correlation_id == "job-42"
    with pytest.raises(FrozenInstanceError):
        error.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["operation", "message", "corrective_action"])
def test_actionable_error_rejects_missing_required_display_fields(field: str) -> None:
    values: dict[str, object] = {
        "operation": "configuration.resolve",
        "category": ErrorCategory.CONFIGURATION_INVALID_VALUE,
        "message": "A required field is invalid.",
        "corrective_action": "Correct the field and try again.",
    }
    values[field] = "   "

    with pytest.raises(ValueError, match=f"{field} must not be blank"):
        ActionableError(**values)  # type: ignore[arg-type]


def test_actionable_error_rejects_invalid_checksum_and_timestamp_session() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _error(checksum="not-a-checksum")

    with pytest.raises(TypeError, match="calendar date"):
        _error(session=datetime(2024, 1, 2, 16, 0))


def test_err_canonically_orders_errors_and_rejects_raw_exceptions() -> None:
    provider_error = _error(
        operation="provider.fetch",
        category=ErrorCategory.PROVIDER_TERMINAL,
        message="The provider returned no records for the requested symbol.",
        corrective_action="Verify the symbol and requested date range.",
        symbol="MSFT",
    )
    configuration_error = _error(
        field_path="data.universe[0]",
        message="The configured symbol is empty.",
        corrective_action="Provide a non-empty ticker symbol.",
    )

    result = Err((provider_error, configuration_error))

    assert result.errors == (configuration_error, provider_error)
    assert isinstance(Ok(value="snapshot-id"), Ok)
    with pytest.raises(ValueError, match="at least one"):
        Err(())
    with pytest.raises(TypeError, match="ActionableError"):
        Err((RuntimeError("database password=secret"),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("error", "expected_fragment"),
    [
        (
            _error(
                field_path="data.requested_range.start",
                message="The start date must not be after the end date.",
                corrective_action="Set a valid inclusive date range.",
            ),
            "field=data.requested_range.start",
        ),
        (
            _error(
                operation="provider.fetch",
                category=ErrorCategory.PROVIDER_TERMINAL,
                message="The provider returned no records for this symbol.",
                corrective_action="Verify the symbol and choose an available range.",
                symbol="SPY",
            ),
            "symbol=SPY",
        ),
        (
            _error(
                operation="snapshot.open",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                message="Referenced content failed checksum verification.",
                corrective_action="Restore verified snapshot content before retrying.",
                checksum="b" * 64,
            ),
            f"checksum={'b' * 64}",
        ),
        (
            _error(
                operation="backtest.execute",
                category=ErrorCategory.BACKTEST_EXECUTION,
                message="No valid next-session open was available for the order.",
                corrective_action="Inspect the symbol data and rerun with a ready snapshot.",
                symbol="AAPL",
                session=date(2024, 2, 1),
            ),
            "session=2024-02-01",
        ),
    ],
)
def test_representative_failures_have_sanitized_deterministic_display_text(
    error: ActionableError, expected_fragment: str
) -> None:
    formatted = error.format_for_display()

    assert expected_fragment in formatted
    assert f"[{error.category.value}]" in formatted
    assert "Corrective action:" in formatted
    assert "\n" not in formatted


def test_unexpected_exception_is_not_exposed_in_boundary_error() -> None:
    error = ActionableError.from_unexpected_exception(
        "snapshot.open",
        RuntimeError("database password=super-secret"),
        correlation_id="corr-27",
    )

    formatted = error.format_for_display()
    assert error.category is ErrorCategory.INTERNAL_UNEXPECTED
    assert "database password" not in formatted
    assert "super-secret" not in formatted
    assert "RuntimeError" not in formatted
    assert "correlation=corr-27" in formatted


def test_reason_taxonomies_expose_stable_machine_values() -> None:
    assert ValidationReason.HIGH_ENVELOPE.value == "high.envelope"
    assert QuarantineReason.DUPLICATE_CONFLICT.value == "duplicate.conflict"
    assert ProviderFailureKind.RETRYABLE.value == "retryable"
    assert ProviderFailureReason.RATE_LIMITED.value == "rate_limited"
    assert JobReason.PARTIAL_DATA_GAP.value == "partial_data_gap"


def test_current_disclosure_is_versioned_complete_and_deterministic() -> None:
    provider_error = _error(
        operation="provider.fetch",
        category=ErrorCategory.PROVIDER_TERMINAL,
        message="No records were available for the requested symbol.",
        corrective_action="Verify the symbol and date range.",
        symbol="SPY",
    )
    integrity_error = _error(
        operation="snapshot.open",
        category=ErrorCategory.INTEGRITY_CHECKSUM,
        message="Referenced partition content failed verification.",
        corrective_action="Restore verified content before opening the snapshot.",
    )

    disclosure = LimitationDisclosure.current(
        data_failures=(provider_error, integrity_error)
    )
    rendered = disclosure.format_for_display()

    assert disclosure.version == LIMITATION_DISCLOSURE_VERSION
    assert disclosure.data_failures == (provider_error, integrity_error)
    assert "free Yahoo Finance" in rendered
    assert "point-in-time index membership" in rendered
    assert "survivorship and selection bias" in rendered
    assert "incomplete, corrected, unavailable, or inconsistent" in rendered
    assert "Recorded data failures:" in rendered
    assert "commission and adverse-slippage" in rendered
    assert "next eligible session open" in rendered
    assert rendered == disclosure.format_for_display()


def test_disclosure_rejects_unknown_version_and_exposes_structural_contract() -> None:
    with pytest.raises(ValueError, match="unsupported limitation disclosure version"):
        LimitationDisclosure(version="limitation-disclosure/v0")

    class DataView:
        limitation_disclosure = LimitationDisclosure.current()

    assert isinstance(DataView(), DisclosureCarrier)
