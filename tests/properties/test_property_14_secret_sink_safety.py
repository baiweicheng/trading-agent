"""Property tests for complete, idempotent secret redaction across sinks."""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters, digits
from urllib.parse import quote, quote_plus

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.config.models import ResolvedConfig
from quant_research_platform.config.serializer import (
    REDACTION_MARKER,
    Redactor,
    SecretLeakError,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SECRET_ALPHABET = ascii_letters + digits + "-_.:@/?&=+ "


@dataclass(frozen=True, slots=True)
class _PresenterDto:
    """Small presenter-shaped DTO with nested secret-bearing display text."""

    title: str
    body: str
    details: dict[str, str]


@dataclass(frozen=True, slots=True)
class _SecretSinkCase:
    """Generated literal/encoded values and structures used by every sink."""

    secret: str
    encoded: str
    plus_encoded: str
    double_encoded: str
    text: str
    url: str
    headers: dict[str, str]
    exception: RuntimeError
    configuration: ResolvedConfig
    progress: dict[str, object]
    logs: dict[str, object]
    mlflow_values: dict[str, object]
    manifest: dict[str, object]
    presenter: _PresenterDto


@st.composite
def secret_sink_cases(draw: st.DrawFn) -> _SecretSinkCase:
    """Generate one secret in literal and common URL-encoded spellings."""

    suffix = draw(st.text(alphabet=_SECRET_ALPHABET, min_size=1, max_size=12))
    secret = f"credential-{suffix}/token"
    encoded = quote(secret, safe="")
    plus_encoded = quote_plus(secret, safe="")
    double_encoded = quote(encoded, safe="")
    text = (
        f"provider failed with {secret}; encoded={encoded}; "
        f"plus={plus_encoded}; double={double_encoded}"
    )
    url = f"https://gateway.invalid/fetch?proxy={encoded}&token={double_encoded}"
    headers = {
        "Authorization": f"Bearer {secret}",
        "X-Credential": encoded,
        "Content-Type": f"application/json; token={plus_encoded}",
        "X-Request-ID": f"request-{secret}",
    }
    exception = RuntimeError(f"provider response included {secret} ({encoded})")
    configuration = ResolvedConfig.model_validate(
        {
            "paths": {
                "data_root": _PROJECT_ROOT / "data-property-14",
                "artifact_root": _PROJECT_ROOT / "artifacts-property-14",
            },
            "data": {
                "requested_range": {"start": "2024-01-02", "end": "2024-01-03"},
            },
            "secrets": {"https_proxy": secret},
        }
    )
    progress = {
        "stage": "fetching",
        "warnings": [text, encoded],
        "context": {"url": url, "headers": headers},
    }
    logs = {
        "message": text,
        "exception": exception,
        "context": {"configuration": configuration, "headers": headers},
    }
    mlflow_values = {
        "params": {"https_proxy": secret, "token": encoded},
        "tags": {"provider_url": url},
    }
    manifest = {
        "configuration": configuration,
        "metadata": {"secret": secret, "encoded": double_encoded},
        "provider": {"url": url, "headers": headers},
    }
    presenter = _PresenterDto(
        title="Provider diagnostics",
        body=f"Unable to fetch {secret}",
        details={"url": url, "credential": encoded},
    )
    return _SecretSinkCase(
        secret,
        encoded,
        plus_encoded,
        double_encoded,
        text,
        url,
        headers,
        exception,
        configuration,
        progress,
        logs,
        mlflow_values,
        manifest,
        presenter,
    )


def _assert_no_registered_form(value: object, case: _SecretSinkCase) -> None:
    """Ensure no literal or URL-encoded secret spelling crossed a sink."""

    rendered = repr(value)
    for form in (
        case.secret,
        case.encoded,
        case.plus_encoded,
        case.double_encoded,
    ):
        assert form not in rendered


def _assert_idempotent_and_safe(
    name: str,
    sanitizer: Callable[[object], object],
    value: object,
    redactor: Redactor,
    case: _SecretSinkCase,
) -> object:
    """Check a sink's fixed point, marker replacement, and secret absence."""

    sanitized = sanitizer(value)
    repeated = sanitizer(sanitized)
    assert repeated == sanitized, f"{name} redaction is not idempotent"
    assert redactor.contains_secret(sanitized) is False
    assert REDACTION_MARKER in repr(sanitized), f"{name} lost its redaction marker"
    _assert_no_registered_form(sanitized, case)
    return sanitized


# Feature: quantitative-research-platform, Property 14: Secret redaction is idempotent and complete across sinks
# Validates: Requirements 16.1, 16.4, 16.6–16.9, 11.19, 13.20.
@settings(max_examples=100, deadline=None)
@given(case=secret_sink_cases())
def test_secret_redaction_is_idempotent_and_complete_across_sinks(
    case: _SecretSinkCase,
) -> None:
    """Every configured display/durable sink removes all registered secret forms."""

    redactor = Redactor((case.secret,))

    _assert_idempotent_and_safe(
        "text", lambda value: redactor.redact_text(value), case.text, redactor, case
    )
    _assert_idempotent_and_safe(
        "url", lambda value: redactor.redact_url(value), case.url, redactor, case
    )
    _assert_idempotent_and_safe(
        "headers",
        lambda value: redactor.redact_headers(value),
        case.headers,
        redactor,
        case,
    )
    _assert_idempotent_and_safe(
        "error",
        lambda value: redactor.redact_error(value),
        case.exception,
        redactor,
        case,
    )
    _assert_idempotent_and_safe(
        "configuration",
        lambda value: redactor.redact_structured(value),
        case.configuration,
        redactor,
        case,
    )
    _assert_idempotent_and_safe(
        "progress",
        lambda value: redactor.redact_progress(value),
        case.progress,
        redactor,
        case,
    )
    _assert_idempotent_and_safe(
        "log",
        lambda value: redactor.redact_log(value),
        case.logs,
        redactor,
        case,
    )
    _assert_idempotent_and_safe(
        "mlflow",
        lambda value: redactor.redact_mlflow_fields(value),
        case.mlflow_values,
        redactor,
        case,
    )
    sanitized_manifest = _assert_idempotent_and_safe(
        "manifest",
        lambda value: redactor.redact_manifest(value),
        case.manifest,
        redactor,
        case,
    )
    sanitized_presenter = _assert_idempotent_and_safe(
        "presenter",
        lambda value: redactor.redact_presenter_dto(value),
        case.presenter,
        redactor,
        case,
    )

    assert isinstance(sanitized_presenter, _PresenterDto)
    assert sanitized_presenter.body == f"Unable to fetch {REDACTION_MARKER}"
    assert isinstance(sanitized_manifest, dict)

    with pytest.raises(SecretLeakError):
        redactor.assert_metadata_is_redacted(case.manifest)
    redactor.assert_metadata_is_redacted(sanitized_manifest)

    # The complete nested sink bundle is also safe to persist after one pass.
    raw_bundle = {
        "configuration": case.configuration,
        "progress": case.progress,
        "logs": case.logs,
        "mlflow": case.mlflow_values,
        "manifest": case.manifest,
        "presenter": case.presenter,
        "exception": case.exception,
        "url": case.url,
        "headers": case.headers,
    }
    sanitized_bundle = redactor.redact_structured(raw_bundle)
    assert redactor.redact_structured(sanitized_bundle) == sanitized_bundle
    assert redactor.contains_secret(sanitized_bundle) is False
    _assert_no_registered_form(sanitized_bundle, case)
    redactor.assert_metadata_is_redacted(sanitized_bundle)
