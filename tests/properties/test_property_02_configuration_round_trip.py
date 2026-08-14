"""Property tests for canonical redacted configuration serialization."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from string import ascii_letters, ascii_uppercase, digits
from typing import Any
from urllib.parse import quote, quote_plus

from hypothesis import given, settings, strategies as st
from pydantic import BaseModel, SecretStr
from ruamel.yaml import YAML

from quant_research_platform.config.loader import ConfigurationManager
from quant_research_platform.config.models import ResolvedConfig, UnresolvedSecret
from quant_research_platform.config.serializer import (
    REDACTION_MARKER,
    ConfigurationSerializer,
    SecretPresence,
    non_secret_config,
)
from quant_research_platform.domain.errors import Ok

# Feature: quantitative-research-platform, Property 2: Canonical redacted configuration round trip
# Validates: Requirements 2.59–2.72, 16.5, 16.8–16.10, 17.2.

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANAGER = ConfigurationManager(project_anchor=_PROJECT_ROOT / "tests" / "entry.py")
_SYMBOLS = tuple(
    f"{first}{second}" for first in ascii_uppercase for second in ascii_uppercase
)
_PATH_ALPHABET = ascii_letters + digits + "-_é"
_SECRET_COMPONENT_ALPHABET = ascii_letters + digits


@st.composite
def _proxy_secrets(draw: st.DrawFn) -> tuple[str, str]:
    """Generate non-empty secrets with literal and URL-encoded spellings."""

    component = st.text(
        alphabet=_SECRET_COMPONENT_ALPHABET,
        min_size=1,
        max_size=12,
    )
    username = draw(component)
    password = draw(component)
    token = draw(component)
    http_proxy = (
        f"http://{username}:{password} secret@proxy.invalid/?token={token}/a b"
    )

    username = draw(component)
    password = draw(component)
    token = draw(component)
    https_proxy = (
        f"https://{username}:{password} secret@proxy.invalid/?token={token}/a b"
    )
    return http_proxy, https_proxy


@st.composite
def _valid_resolved_configurations(draw: st.DrawFn) -> ResolvedConfig:
    """Generate valid, fully populated configurations with resolved secrets."""

    path_segment = draw(st.text(alphabet=_PATH_ALPHABET, min_size=1, max_size=12))
    universe = draw(
        st.lists(
            st.sampled_from(_SYMBOLS),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    range_start = draw(
        st.dates(min_value=date(2000, 1, 1), max_value=date(2025, 12, 30))
    )
    range_end = draw(st.dates(min_value=range_start, max_value=date(2025, 12, 31)))
    initial_delay = draw(st.decimals(min_value=0, max_value=60, places=2))
    max_delay = draw(st.decimals(min_value=initial_delay, max_value=60, places=2))
    http_proxy, https_proxy = draw(_proxy_secrets())

    return ResolvedConfig.model_validate(
        {
            "paths": {
                "data_root": _PROJECT_ROOT / f"data-{path_segment}",
                "artifact_root": _PROJECT_ROOT / f"artifacts-{path_segment}",
                "metadata_db": _PROJECT_ROOT / f"metadata-{path_segment}.duckdb",
                "mlflow_db": _PROJECT_ROOT / f"mlflow-{path_segment}.db",
                "local_secrets_file": _PROJECT_ROOT
                / f"secrets-{path_segment}.local.yaml",
            },
            "retry": {
                "attempts": draw(st.integers(min_value=1, max_value=5)),
                "initial_delay_seconds": initial_delay,
                "max_delay_seconds": max_delay,
                "backoff_multiplier": draw(
                    st.decimals(min_value=1, max_value=4, places=2)
                ),
            },
            "data": {
                "universe": universe,
                "requested_range": {"start": range_start, "end": range_end},
                "batch_size": draw(st.integers(min_value=1, max_value=10)),
                "staleness_sessions": draw(st.integers(min_value=0, max_value=252)),
                "revision_overlap_sessions": draw(
                    st.integers(min_value=0, max_value=252)
                ),
                "write_chunk_rows": draw(
                    st.integers(min_value=1, max_value=100_000)
                ),
            },
            "strategy": {
                "position_count": draw(st.integers(min_value=1, max_value=len(universe)))
            },
            "execution": {
                "commission_bps": draw(
                    st.decimals(min_value=0, max_value=100_000, places=3)
                ),
                "slippage_bps": draw(
                    st.decimals(min_value=0, max_value=100_000, places=3)
                ),
            },
            "ui": {"page_size": draw(st.integers(min_value=1, max_value=100))},
            "runtime": {
                "deterministic_seed": draw(
                    st.integers(min_value=0, max_value=4_294_967_295)
                )
            },
            "secrets": {"http_proxy": http_proxy, "https_proxy": https_proxy},
        }
    )


def _assert_schema_order(value: object, model: type[BaseModel]) -> None:
    """Assert mappings recursively retain their Pydantic declaration order."""

    assert isinstance(value, Mapping)
    assert list(value) == list(model.model_fields)
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _assert_schema_order(value[field_name], annotation)


def _equivalent_non_secret_projection(config: ResolvedConfig) -> dict[str, Any]:
    """Compare safe values while treating redacted and unresolved as present."""

    projection = non_secret_config(config).model_dump(mode="python")
    projection["secrets"] = {
        field_name: getattr(non_secret_config(config).secrets, field_name)
        is not SecretPresence.ABSENT
        for field_name in ("http_proxy", "https_proxy")
    }
    return projection


def _assert_secret_forms_are_absent(document: bytes, config: ResolvedConfig) -> None:
    """Reject literal, single-encoded, and double-encoded credential forms."""

    for secret in (config.secrets.http_proxy, config.secrets.https_proxy):
        assert isinstance(secret, SecretStr)
        value = secret.get_secret_value()
        encoded_forms = (
            value,
            quote(value, safe=""),
            quote_plus(value, safe=""),
            quote(quote(value, safe=""), safe=""),
        )
        for form in encoded_forms:
            assert form.encode("utf-8") not in document


@settings(max_examples=100, deadline=None)
@given(config=_valid_resolved_configurations())
def test_canonical_redacted_configuration_round_trip(config: ResolvedConfig) -> None:
    """Canonical bytes round trip under one project root without environment input."""

    serializer = ConfigurationSerializer()
    first_serialization = serializer.serialize(config)
    second_serialization = serializer.serialize(config)
    safe_view_serialization = serializer.serialize(non_secret_config(config))

    assert first_serialization == second_serialization == safe_view_serialization
    canonical_text = first_serialization.decode("utf-8")
    assert canonical_text.encode("utf-8") == first_serialization
    assert b"\r" not in first_serialization
    assert first_serialization.endswith(b"\n")
    assert not first_serialization.endswith(b"\n\n")
    _assert_secret_forms_are_absent(first_serialization, config)

    parsed_yaml = YAML(typ="safe").load(canonical_text)
    _assert_schema_order(parsed_yaml, type(non_secret_config(config)))
    assert parsed_yaml["secrets"] == {
        "http_proxy": REDACTION_MARKER,
        "https_proxy": REDACTION_MARKER,
    }

    parsed_result = _MANAGER.resolve(first_serialization, {})
    assert isinstance(parsed_result, Ok)
    reloaded = parsed_result.value
    assert isinstance(reloaded.secrets.http_proxy, UnresolvedSecret)
    assert isinstance(reloaded.secrets.https_proxy, UnresolvedSecret)
    assert _equivalent_non_secret_projection(reloaded) == _equivalent_non_secret_projection(
        config
    )
    assert serializer.serialize(reloaded) == first_serialization
