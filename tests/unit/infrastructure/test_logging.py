"""Focused tests for durable structured JSONL diagnostics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import pytest

from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.execution import JobStage
from quant_research_platform.infrastructure.logging import StructuredJsonlLogger

_NOW = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)


def test_jsonl_logger_persists_correlation_and_redacts_message_context_and_exception(
    tmp_path: Path,
) -> None:
    secret = "https://user:p@ss word@proxy.example/?token=a/b"
    encoded = quote(secret, safe="")
    path = tmp_path / "logs" / "diagnostics.jsonl"
    logger = StructuredJsonlLogger(
        path,
        redactor=Redactor((secret,)),
        utc_now=lambda: _NOW,
    )

    record = logger.write(
        level="error",
        operation="ingestion.fetch",
        correlation_id="corr-123",
        message=f"Provider returned {secret}",
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        stage=JobStage.FETCHING,
        category="provider.terminal",
        context={"url": encoded, "nested": [secret]},
        exception=RuntimeError(f"headers included {secret}"),
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["correlation_id"] == "corr-123"
    assert payload["job_id"] == "00000000-0000-0000-0000-000000000001"
    assert payload["stage"] == "fetching"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["context"]["url"] == "[REDACTED]"
    assert payload["context"]["nested"] == ["[REDACTED]"]
    assert "[REDACTED]" in lines[0]
    assert secret not in lines[0]
    assert encoded not in lines[0]
    assert record.context_json == json.dumps(
        payload["context"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def test_jsonl_logger_rejects_invalid_level_and_uses_safe_placeholder_for_unknown_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "diagnostics.jsonl"
    logger = StructuredJsonlLogger(path, utc_now=lambda: _NOW)

    with pytest.raises(ValueError, match="unsupported log level"):
        logger.write(
            level="trace",
            operation="ingestion.fetch",
            correlation_id="corr-123",
            message="invalid",
        )

    logger.write(
        level="info",
        operation="ingestion.fetch",
        correlation_id="corr-123",
        message="safe fallback",
        context={"unserializable": object()},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["context"] == {"unserializable": "[UNAVAILABLE]"}
