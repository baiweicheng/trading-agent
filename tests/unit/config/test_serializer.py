"""Focused tests for canonical non-secret configuration and reusable redaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pytest
from pydantic import SecretStr
from ruamel.yaml import YAML

from quant_research_platform.config.models import ResolvedConfig, UnresolvedSecret
from quant_research_platform.config.serializer import (
    REDACTION_MARKER,
    ConfigurationSerializer,
    Redactor,
    RequiredSecretError,
    SecretLeakError,
    SecretPresence,
    non_secret_config,
    require_resolved_secrets,
)
from quant_research_platform.domain.errors import ActionableError, ErrorCategory


@dataclass(frozen=True)
class _PresenterDto:
    message: str
    nested: dict[str, str]


def _config(
    *, http_proxy: str | None = None, https_proxy: str | None = None
) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "paths": {
                "data_root": Path("research-data"),
                "artifact_root": Path("research-data/artifacts"),
            },
            "retry": {
                "attempts": 4,
                "initial_delay_seconds": "1.50",
                "max_delay_seconds": "8.00",
                "backoff_multiplier": "2.00",
            },
            "data": {
                "universe": [" msft ", "AAPL"],
                "requested_range": {"start": "2020-01-02", "end": "2024-12-31"},
                "batch_size": 2,
            },
            "strategy": {"position_count": 2},
            "execution": {"commission_bps": "5.00", "slippage_bps": "10.0"},
            "ui": {"page_size": 25},
            "runtime": {"deterministic_seed": 42},
            "secrets": {"http_proxy": http_proxy, "https_proxy": https_proxy},
        }
    )


def test_equivalent_non_secret_configurations_serialize_to_identical_canonical_bytes() -> (
    None
):
    serializer = ConfigurationSerializer()
    first_secret = "http://alice:one secret@proxy.example:8080/?token=a/b"
    second_secret = "http://bob:another secret@proxy.example:8080/?token=c/d"

    first = _config(http_proxy=first_secret, https_proxy=first_secret)
    equivalent_non_secret = _config(http_proxy=second_secret, https_proxy=second_secret)

    first_bytes = serializer.serialize(first)
    second_bytes = serializer.serialize(equivalent_non_secret)
    text = first_bytes.decode("utf-8")
    parsed = YAML(typ="safe").load(text)

    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert not first_bytes.endswith(b"\n\n")
    assert b"\r" not in first_bytes
    assert first_secret.encode("utf-8") not in first_bytes
    assert quote(first_secret, safe="").encode("utf-8") not in first_bytes
    assert parsed["retry"]["initial_delay_seconds"] == 1.5
    assert parsed["execution"]["commission_bps"] == 5.0
    assert parsed["secrets"] == {
        "http_proxy": REDACTION_MARKER,
        "https_proxy": REDACTION_MARKER,
    }

    root_keys = [
        line.split(":", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith(" ") and line.endswith(":")
    ]
    assert root_keys == list(ResolvedConfig.model_fields)
    assert 'http_proxy: "[REDACTED]"' in text


def test_marker_reload_is_unresolved_and_non_secret_projection_never_contains_secret() -> (
    None
):
    serializer = ConfigurationSerializer()
    secret = "https://name:p@ss word@proxy.example"
    resolved = _config(https_proxy=secret)

    view = non_secret_config(resolved)
    serialized = serializer.serialize(resolved)
    reloaded = serializer.deserialize(serialized)

    assert isinstance(resolved.secrets.https_proxy, SecretStr)
    assert view.secrets.http_proxy is SecretPresence.ABSENT
    assert view.secrets.https_proxy is SecretPresence.PRESENT_REDACTED
    assert secret not in repr(view)
    assert secret.encode("utf-8") not in serialized
    assert isinstance(reloaded.secrets.https_proxy, UnresolvedSecret)
    assert reloaded.secrets.http_proxy is None
    assert (
        non_secret_config(reloaded).secrets.https_proxy
        is SecretPresence.PRESENT_UNRESOLVED
    )

    # A persisted canonical marker can never be interpreted as an actual credential.
    with pytest.raises(ValueError, match="only null or"):
        serializer.deserialize(
            serialized.replace(REDACTION_MARKER.encode(), b"literal-secret")
        )


def test_unresolved_secrets_gate_only_operations_that_declare_them() -> None:
    serializer = ConfigurationSerializer()
    resolved = _config(https_proxy="https://name:password@proxy.example")
    reloaded = serializer.deserialize(serializer.serialize(resolved))

    require_resolved_secrets(reloaded, ())
    require_resolved_secrets(resolved, ("https_proxy",))

    with pytest.raises(RequiredSecretError, match=r"secrets\.https_proxy"):
        require_resolved_secrets(reloaded, ("https_proxy",))
    with pytest.raises(ValueError, match="Unknown secret"):
        require_resolved_secrets(reloaded, ("secrets.unknown",))


def test_redactor_sanitizes_every_supported_sink_idempotently() -> None:
    secret = "https://user:p@ss word@proxy.example:8443/?token=a/b"
    encoded = quote(secret, safe="")
    redactor = Redactor.from_config(_config(https_proxy=secret))
    error = ActionableError(
        operation="provider.fetch",
        category=ErrorCategory.PROVIDER_TERMINAL,
        message=f"Provider rejected {secret}",
        corrective_action=f"Replace credential {encoded}",
    )
    metadata = {
        "url": f"https://gateway.example/?proxy={encoded}",
        "header": f"Bearer {secret}",
        "error": error,
        "config": _config(https_proxy=secret),
        "items": [secret, {"secret-key": secret}],
    }
    presenter = _PresenterDto(message=secret, nested={"proxy": encoded})

    sanitized_metadata = redactor.redact_metadata(metadata)
    sanitized_presenter = redactor.redact_presenter_dto(presenter)

    assert redactor.registered_form_count >= 3
    assert redactor.redact_url(encoded) == REDACTION_MARKER
    assert redactor.redact_headers({"Authorization": f"Bearer {secret}"}) == {
        "Authorization": REDACTION_MARKER
    }
    assert redactor.redact_error(RuntimeError(secret)) == REDACTION_MARKER
    assert redactor.redact_progress(metadata) == sanitized_metadata
    assert redactor.redact_log(metadata) == sanitized_metadata
    assert redactor.redact_mlflow_fields(metadata) == sanitized_metadata
    assert redactor.redact_manifest(metadata) == sanitized_metadata
    assert isinstance(sanitized_presenter, _PresenterDto)
    assert sanitized_presenter.message == REDACTION_MARKER
    assert sanitized_presenter.nested["proxy"] == REDACTION_MARKER
    assert secret not in repr(sanitized_metadata)
    assert encoded not in repr(sanitized_metadata)
    assert redactor.redact_metadata(sanitized_metadata) == sanitized_metadata
    assert not redactor.contains_secret(sanitized_metadata)

    with pytest.raises(SecretLeakError):
        redactor.assert_metadata_is_redacted(metadata)
    redactor.assert_metadata_is_redacted(sanitized_metadata)
