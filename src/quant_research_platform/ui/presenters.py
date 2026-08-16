"""Redacted application-DTO presenters for the Streamlit adapter.

The presentation layer receives values produced by application services.  It does
not know about storage, provider clients, Streamlit session state, or opaque
configuration handles.  Presenter functions deliberately return small,
JSON-like mappings so pages can render them without serializing domain objects
or accidentally exposing a secret-bearing field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, TypeAlias
from uuid import UUID

from ..application.services import ActionableError, Err, LimitationDisclosure
from ..config.serializer import REDACTION_MARKER, Redactor

Presentation: TypeAlias = dict[str, object]

_SECRET_FIELD_WORDS = (
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "private",
    "authorization",
    "api_key",
    "apikey",
    "proxy",
)
_OPAQUE_FIELD_WORDS = (
    "handle",
    "token",
    "stream_factory",
    "payload",
    "raw_config",
    "resolved_config",
)


def _field(value: object, names: str | Sequence[str], default: object = None) -> object:
    candidates = (names,) if isinstance(names, str) else tuple(names)
    if isinstance(value, Mapping):
        for name in candidates:
            if name in value:
                return value[name]
        return default
    for name in candidates:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _items(value: object) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return ()
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def _count(value: object) -> int:
    return len(_items(value))


def _redact_text(value: str, redactor: Redactor | None) -> str:
    return redactor.redact_text(value) if redactor is not None else value


def _field_name_is_secret(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    if normalized in {"secrets", "secret_config"}:
        return False
    return any(word in normalized for word in _SECRET_FIELD_WORDS)


def _field_name_is_opaque(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return normalized.startswith("_") or any(
        word in normalized for word in _OPAQUE_FIELD_WORDS
    )


def _safe_value(value: object, redactor: Redactor | None = None) -> object:
    """Convert a DTO to a safe presentation value.

    Unknown objects are represented by their type name rather than ``str(value)``;
    this prevents an object's debug representation from becoming an unreviewed
    output sink.  Structured values are recursively redacted and known secret or
    opaque fields are removed/replaced before rendering.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime, UUID)):
        return value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    if isinstance(value, PurePath):
        return _redact_text(value.as_posix(), redactor)
    if isinstance(value, Enum):
        return _safe_value(value.value, redactor)
    if isinstance(value, str):
        return _redact_text(value, redactor)
    if isinstance(value, bytes):
        return REDACTION_MARKER
    if hasattr(value, "get_secret_value"):
        return REDACTION_MARKER

    if isinstance(value, Mapping):
        result: Presentation = {}
        for raw_key, raw_item in value.items():
            key = _redact_text(str(raw_key), redactor)
            if _field_name_is_opaque(key):
                continue
            if _field_name_is_secret(key):
                result[key] = REDACTION_MARKER
            else:
                result[key] = _safe_value(raw_item, redactor)
        return result

    serializable = getattr(value, "to_serializable", None)
    if callable(serializable):
        try:
            return _safe_value(serializable(), redactor)
        except (TypeError, ValueError):
            # Fall through to a structural read for lightweight DTO doubles.
            pass

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _safe_value(model_dump(mode="python"), redactor)
        except TypeError:
            return _safe_value(model_dump(), redactor)

    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for field in fields(value):
            name = field.name
            if _field_name_is_opaque(name):
                continue
            if _field_name_is_secret(name):
                result[name] = REDACTION_MARKER
            else:
                result[name] = _safe_value(getattr(value, name), redactor)
        return result

    if isinstance(value, (tuple, list)):
        return [_safe_value(item, redactor) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe_value(item, redactor) for item in sorted(value, key=str)]

    return type(value).__name__


def _required_disclosure(value: object) -> LimitationDisclosure:
    disclosure = _field(value, "limitation_disclosure")
    if not isinstance(disclosure, LimitationDisclosure):
        raise TypeError(
            "data, snapshot, result, and comparison presenters require "
            "a LimitationDisclosure"
        )
    return disclosure


def present_configuration_resolution(
    resolution: object, *, redactor: Redactor | None = None
) -> Presentation:
    """Present only the credential-free view of a configuration resolution.

    The returned mapping intentionally omits ``ConfigurationHandle`` and any
    private/full resolved configuration retained by the facade.
    """

    view = _field(resolution, ("view", "non_secret", "configuration"))
    if view is None:
        raise TypeError("resolution must contain a non-secret configuration view")
    rendered = _safe_value(view, redactor)
    if not isinstance(rendered, dict):
        raise TypeError("configuration view must render as a mapping")
    return {"configuration": rendered}


present_configuration = present_configuration_resolution


def present_limitation_disclosure(
    disclosure: LimitationDisclosure, *, redactor: Redactor | None = None
) -> Presentation:
    """Return the complete visible disclosure in a stable order."""

    if not isinstance(disclosure, LimitationDisclosure):
        raise TypeError("disclosure must be a LimitationDisclosure")
    return {
        "version": _safe_value(disclosure.version, redactor),
        "lines": [_redact_text(line, redactor) for line in disclosure.lines()],
        "data_failures": [
            present_actionable_error(error, redactor=redactor)
            for error in disclosure.data_failures
        ],
    }


def present_actionable_error(
    error: ActionableError, *, redactor: Redactor | None = None
) -> Presentation:
    """Format one already-sanitized actionable error without raw exception data."""

    if not isinstance(error, ActionableError):
        raise TypeError("error must be an ActionableError")
    result: Presentation = {
        "operation": _safe_value(error.operation, redactor),
        "category": _safe_value(error.category, redactor),
        "message": _safe_value(error.message, redactor),
        "corrective_action": _safe_value(error.corrective_action, redactor),
    }
    for name in ("field_path", "symbol", "session", "checksum", "correlation_id"):
        value = getattr(error, name)
        if value is not None:
            result[name] = _safe_value(value, redactor)
    return result


def present_errors(
    errors: Err | Sequence[ActionableError], *, redactor: Redactor | None = None
) -> list[Presentation]:
    """Present an ``Err`` or error sequence in its application-defined order."""

    values = errors.errors if isinstance(errors, Err) else tuple(errors)
    if any(not isinstance(error, ActionableError) for error in values):
        raise TypeError("errors must contain ActionableError values")
    return [present_actionable_error(error, redactor=redactor) for error in values]


def present_progress(
    progress: object, *, redactor: Redactor | None = None
) -> Presentation:
    """Present sanitized job progress and never include application state."""

    result: Presentation = {
        "job_id": _safe_value(_field(progress, "job_id"), redactor),
        "operation": _safe_value(_field(progress, "operation"), redactor),
        "state": _safe_value(_field(progress, "state"), redactor),
        "stage": _safe_value(_field(progress, "stage"), redactor),
        "completed_units": _safe_value(
            _field(progress, "completed_units", 0), redactor
        ),
        "total_units": _safe_value(_field(progress, "total_units"), redactor),
        "elapsed_seconds": _safe_value(
            _field(progress, "elapsed_seconds", Decimal("0")), redactor
        ),
        "warnings": [
            _redact_text(str(warning), redactor)
            for warning in _items(_field(progress, "warnings", ()))
        ],
    }
    return result


def _present_date_range(value: object, *, redactor: Redactor | None = None) -> object:
    if value is None:
        return None
    start = _field(value, "start")
    end = _field(value, "end")
    if start is not None and end is not None:
        return {
            "start": _safe_value(start, redactor),
            "end": _safe_value(end, redactor),
        }
    return _safe_value(value, redactor)


def present_artifact(
    artifact: object, *, redactor: Redactor | None = None
) -> Presentation:
    """Present artifact metadata only; payloads and lazy streams stay private."""

    result: Presentation = {}
    fields_to_copy = (
        ("checksum", ("checksum",)),
        ("role", ("role", "artifact_kind", "kind")),
        ("uri", ("relative_uri", "uri")),
        ("media_type", ("media_type",)),
        ("byte_size", ("byte_size", "size")),
        ("row_count", ("row_count",)),
        ("schema_version", ("schema_version",)),
        ("availability", ("availability",)),
        ("scientific", ("scientific",)),
        ("columns", ("columns",)),
    )
    for output_name, names in fields_to_copy:
        value = _field(artifact, names)
        if value is not None:
            result[output_name] = _safe_value(value, redactor)
    valid = _field(artifact, "valid")
    if valid is not None:
        result["valid"] = bool(valid)
    return result


def present_metric_value(
    metric: object, *, redactor: Redactor | None = None
) -> Presentation:
    return {
        "name": _safe_value(_field(metric, "name"), redactor),
        "value": _safe_value(_field(metric, "value"), redactor),
        "null_reason": _safe_value(_field(metric, "null_reason"), redactor),
    }


def present_metrics(
    metrics: object, *, redactor: Redactor | None = None
) -> Presentation:
    """Present one metric set or a strategy/benchmark/difference result."""

    if _field(metrics, "metrics") is not None:
        return {
            "scope": _safe_value(_field(metrics, "scope"), redactor),
            "metrics": [
                present_metric_value(item, redactor=redactor)
                for item in _items(_field(metrics, "metrics"))
            ],
        }
    names = ("strategy_metrics", "benchmark_metrics", "differences")
    if any(_field(metrics, name) is not None for name in names):
        return {
            name: present_metrics(_field(metrics, name), redactor=redactor)
            for name in names
            if _field(metrics, name) is not None
        }
    return _safe_value(metrics, redactor)  # type: ignore[return-value]


def present_table_page(
    page: object,
    *,
    configured_page_size: int = 100,
    requested_page_size: int | None = None,
    redactor: Redactor | None = None,
) -> Presentation:
    """Present an ordinary table page with an absolute 100-row ceiling."""

    if isinstance(configured_page_size, bool) or configured_page_size < 1:
        raise ValueError("configured_page_size must be positive")
    if requested_page_size is not None and (
        isinstance(requested_page_size, bool) or requested_page_size < 1
    ):
        raise ValueError("requested_page_size must be positive or None")

    rows = _items(_field(page, ("rows", "items"), ()))
    source_size = _field(page, "page_size", 100)
    if not isinstance(source_size, int) or isinstance(source_size, bool):
        source_size = 100
    limit = min(100, configured_page_size, source_size)
    if requested_page_size is not None:
        limit = min(limit, requested_page_size)
    shown_rows = rows[:limit]
    total_value = _field(page, "total")
    total = total_value if isinstance(total_value, int) else None
    page_value = _field(page, "page", 0)
    page_number = page_value if isinstance(page_value, int) else 0
    has_next = _field(page, "has_next")
    if callable(has_next):
        has_next = has_next()
    if has_next is None:
        has_next = (
            (page_number + 1) * limit < total
            if isinstance(total, int)
            else len(rows) > limit or len(shown_rows) == limit
        )
    result: Presentation = {
        "rows": [_safe_value(row, redactor) for row in shown_rows],
        "page": _safe_value(page_number, redactor),
        "page_size": limit,
        "row_count": len(shown_rows),
        "total": _safe_value(total, redactor),
        "has_next": bool(has_next),
    }
    columns = _field(page, "columns")
    if columns:
        result["columns"] = _safe_value(_items(columns), redactor)
    checksum = _field(page, "artifact_checksum")
    if checksum is not None:
        result["artifact_checksum"] = _safe_value(checksum, redactor)
    return result


def present_ingestion_result(
    result: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(result)
    validation = _field(result, "validation")
    report = _field(validation, "report") if validation is not None else None
    return {
        "status": _safe_value(_field(result, "status"), redactor),
        "snapshot_id": _safe_value(_field(result, "snapshot_id"), redactor),
        "requested_range": _present_date_range(
            _field(result, "requested_range"), redactor=redactor
        ),
        "provider_batch_count": _count(_field(result, "provider_batches")),
        "provider_record_count": _count(_field(result, "provider_records")),
        "accepted_row_count": _count(_field(result, "accepted_rows")),
        "quarantined_row_count": _count(_field(result, "quarantined_rows")),
        "gap_count": _count(_field(result, "gaps")),
        "failed_symbols": _safe_value(
            _items(_field(result, "failed_symbols")), redactor
        ),
        "retained_parent_coverage_symbols": _safe_value(
            _items(_field(result, "retained_parent_coverage_symbols")),
            redactor,
        ),
        "snapshot_reused": bool(_field(result, "snapshot_reused", False)),
        "validation": _safe_value(report or validation, redactor),
        "errors": present_errors(_items(_field(result, "errors")), redactor=redactor),
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


def present_snapshot_detail(
    detail: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(detail)
    summary = _field(detail, "summary")
    provenance = _field(detail, "provenance")
    readiness = _field(detail, "readiness")
    return {
        "snapshot_id": _safe_value(_field(detail, "snapshot_id"), redactor),
        "summary": _safe_value(summary, redactor),
        "provenance": _safe_value(provenance, redactor),
        "validation": _safe_value(
            _field(detail, ("validation_summary", "validation")), redactor
        ),
        "readiness": _safe_value(readiness, redactor),
        "comparison_ready": bool(_field(detail, "comparison_ready", False)),
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


def present_evaluation_output(
    output: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(output)
    evaluation_result = _field(output, ("evaluation_result", "result", "metrics"))
    artifacts = _field(output, "artifacts")
    artifact_values: tuple[Any, ...]
    if artifacts is not None and not isinstance(artifacts, Mapping):
        artifact_values = _items(artifacts)
    else:
        artifact_values = _items(_field(artifacts, ("items", "artifacts"), ()))
    return {
        "evaluation_range": _present_date_range(
            _field(output, "evaluation_range"), redactor=redactor
        ),
        "metrics": present_metrics(evaluation_result, redactor=redactor),
        "artifact_checksums": _safe_value(
            _field(output, "artifact_checksums", {}), redactor
        ),
        "artifacts": [
            present_artifact(artifact, redactor=redactor)
            for artifact in artifact_values
        ],
        "spy_gaps": _safe_value(_items(_field(output, "spy_gaps")), redactor),
        "unfilled_order_count": _count(_field(output, "unfilled_orders")),
        "unfilled_diagnostics": present_errors(
            _items(_field(output, "unfilled_diagnostics")), redactor=redactor
        ),
        "ending_cash_balance": _safe_value(
            _field(output, "ending_cash_balance"), redactor
        ),
        "total_commissions": _safe_value(_field(output, "total_commissions"), redactor),
        "total_slippage": _safe_value(_field(output, "total_slippage"), redactor),
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


def present_backtest_result(
    result: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(result)
    core = _field(result, ("core_output", "output"))
    audit = _field(result, "audit")
    evaluation = _field(result, "evaluation")
    diagnostics = _items(_field(result, "diagnostics"))
    return {
        "run_id": _safe_value(_field(result, "run_id"), redactor),
        "snapshot_id": _safe_value(_field(result, "snapshot_id"), redactor),
        "evaluation_range": _present_date_range(
            _field(result, "evaluation_range"), redactor=redactor
        ),
        "core_output_counts": {
            "orders": _count(_field(core, "orders")),
            "fills": _count(_field(core, "fills")),
            "portfolio_states": _count(_field(core, "portfolio_states")),
            "daily_returns": _count(_field(core, "daily_returns")),
            "strategy_decisions": _count(_field(core, "strategy_decisions")),
        },
        "unfilled_order_count": _count(_field(audit, "unfilled_orders")),
        "unfilled_diagnostics": present_errors(
            _items(_field(audit, "unfilled_diagnostics")), redactor=redactor
        ),
        "diagnostics": present_errors(diagnostics, redactor=redactor),
        "evaluation": (
            present_evaluation_output(evaluation, redactor=redactor)
            if evaluation is not None
            else None
        ),
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


def present_run_detail(
    detail: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(detail)
    artifacts = _items(_field(detail, ("artifacts", "artifact_metadata")))
    return {
        "summary": _safe_value(_field(detail, "summary"), redactor),
        "manifest": _safe_value(_field(detail, ("manifest", "run_manifest")), redactor),
        "configuration": _safe_value(_field(detail, "configuration"), redactor),
        "environment_fingerprint": _safe_value(
            _field(detail, ("environment_fingerprint", "fingerprint")), redactor
        ),
        "validation": _safe_value(
            _field(detail, ("validation_report", "validation")), redactor
        ),
        "logs": _safe_value(_field(detail, ("logs", "log_entries")), redactor),
        "artifacts": [
            present_artifact(artifact, redactor=redactor) for artifact in artifacts
        ],
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


def _present_curve(curve: object, *, redactor: Redactor | None = None) -> Presentation:
    return {
        "run_id": _safe_value(_field(curve, "run_id"), redactor),
        "snapshot_id": _safe_value(_field(curve, "snapshot_id"), redactor),
        "original_range": _present_date_range(
            {
                "start": _field(curve, "evaluation_start"),
                "end": _field(curve, "evaluation_end"),
            },
            redactor=redactor,
        ),
        "strategy_metrics": present_metrics(
            _field(curve, "strategy_metrics"), redactor=redactor
        ),
        "benchmark_metrics": present_metrics(
            _field(curve, "benchmark_metrics"), redactor=redactor
        ),
        "strategy_curve": _safe_value(
            _items(_field(curve, "strategy_curve")), redactor
        ),
        "benchmark_curve": _safe_value(
            _items(_field(curve, "benchmark_curve")), redactor
        ),
    }


def present_comparison_output(
    output: object, *, redactor: Redactor | None = None
) -> Presentation:
    disclosure = _required_disclosure(output)
    aligned_value = _field(output, "aligned_range")
    aligned_range: object | None = None
    if isinstance(aligned_value, (tuple, list)) and len(aligned_value) >= 2:
        aligned_range = _present_date_range(
            {"start": aligned_value[0], "end": aligned_value[1]},
            redactor=redactor,
        )
    return {
        "runs": [
            _present_curve(curve, redactor=redactor)
            for curve in _items(_field(output, ("runs", "comparison_set")))
        ],
        "aligned_range": aligned_range,
        "aligned_sessions": _safe_value(
            _items(_field(output, "aligned_sessions")), redactor
        ),
        "snapshot_differences": _safe_value(
            _items(_field(output, "snapshot_differences")), redactor
        ),
        "configuration_differences": _safe_value(
            _items(_field(output, "configuration_differences")), redactor
        ),
        "environment_differences": _safe_value(
            _items(_field(output, "environment_differences")), redactor
        ),
        "artifact": present_artifact(_field(output, "artifact"), redactor=redactor),
        "artifact_checksum": _safe_value(_field(output, "artifact_checksum"), redactor),
        "limitation_disclosure": present_limitation_disclosure(
            disclosure, redactor=redactor
        ),
    }


# Compatibility aliases keep page code readable and make the DTO boundary
# discoverable without introducing a second implementation path.
present_data_result = present_ingestion_result
present_snapshot = present_snapshot_detail
present_backtest = present_backtest_result
present_evaluation = present_evaluation_output
present_comparison = present_comparison_output
present_page = present_table_page
present_download = present_artifact

__all__ = [
    "Presentation",
    "present_actionable_error",
    "present_artifact",
    "present_backtest",
    "present_backtest_result",
    "present_comparison",
    "present_comparison_output",
    "present_configuration",
    "present_configuration_resolution",
    "present_data_result",
    "present_download",
    "present_errors",
    "present_evaluation",
    "present_evaluation_output",
    "present_ingestion_result",
    "present_limitation_disclosure",
    "present_metric_value",
    "present_metrics",
    "present_page",
    "present_progress",
    "present_run_detail",
    "present_snapshot",
    "present_snapshot_detail",
    "present_table_page",
]
