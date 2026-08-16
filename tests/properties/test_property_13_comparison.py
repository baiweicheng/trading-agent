"""Property tests for ordered, provenance-preserving run comparison."""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.comparisons import ComparisonService
from quant_research_platform.domain.canonical import canonical_json
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok
from quant_research_platform.domain.evaluation import (
    EvaluationMetrics,
    MetricName,
    MetricScope,
    MetricValue,
)

_BASE_DATE = date(2024, 1, 2)
_MARKER = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class ComparisonCase:
    """A bounded comparison selection and its fake immutable dependencies."""

    selection: tuple[UUID, ...]
    records: Mapping[UUID, object]
    artifact_payloads: Mapping[str, bytes]
    mode: str


class MetadataStore:
    """Minimal local metadata port used by the property test."""

    def __init__(self, records: Mapping[UUID, object]) -> None:
        self.records = records

    def get_run(self, run_id: UUID) -> object:
        return self.records[run_id]


class ArtifactStore:
    """Verified-artifact port with an injectable corruption case."""

    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        self.payloads = payloads

    def open_verified_artifact(self, reference: object) -> bytes:
        if not isinstance(reference, Mapping):
            raise TypeError("comparison artifact references must be mappings")
        checksum = reference["checksum"]
        return self.payloads[str(checksum)]

    def publish_artifact(
        self, payload: bytes, metadata: Mapping[str, object]
    ) -> object:
        checksum = str(metadata["checksum"])
        assert sha256(payload).hexdigest() == checksum
        return SimpleNamespace(checksum=checksum)


def _plain(value: object) -> object:
    """Independently normalize manifest values for checksum calculation."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, Decimal, UUID)):
        return str(value)
    method = getattr(value, "to_serializable", None)
    if callable(method):
        return _plain(method())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return str(value)


def _manifest_checksum(manifest: Mapping[str, object]) -> str:
    return sha256(canonical_json(_plain(manifest))).hexdigest()


def _metrics(scope: MetricScope, seed: int) -> EvaluationMetrics:
    """Build complete valid metric sets while retaining per-run differences."""

    performance = (
        MetricName.TOTAL_RETURN,
        MetricName.COMPOUND_ANNUAL_GROWTH_RATE,
        MetricName.ANNUALIZED_VOLATILITY,
        MetricName.SHARPE_RATIO,
        MetricName.MAXIMUM_DRAWDOWN,
    )
    names = performance
    if scope is MetricScope.STRATEGY:
        names += (
            MetricName.TURNOVER,
            MetricName.TOTAL_COMMISSIONS,
            MetricName.TOTAL_SLIPPAGE,
            MetricName.UNFILLED_ORDERS,
            MetricName.ENDING_CASH_BALANCE,
        )
    values: list[MetricValue] = []
    for ordinal, name in enumerate(names):
        if name is MetricName.UNFILLED_ORDERS:
            value: Decimal | int = seed % 2
        elif name is MetricName.MAXIMUM_DRAWDOWN:
            value = Decimal("-0.1") - Decimal(seed) / Decimal("1000")
        else:
            value = Decimal("1") + Decimal(seed + ordinal) / Decimal("1000")
        values.append(MetricValue(name, value))
    return EvaluationMetrics(scope, tuple(values))


def _curve(
    sessions: Sequence[date], seed: int, multiplier: int
) -> tuple[tuple[date, Decimal], ...]:
    return tuple(
        (session, Decimal(100_000 + seed * 100 + ordinal * multiplier))
        for ordinal, session in enumerate(sessions)
    )


def _difference_projection(rows: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            cast(Any, row).category,
            cast(Any, row).field_path,
            cast(Any, row).values,
        )
        for row in rows
    )


def _redacted_mapping(value: object) -> object:
    """Apply the independent key-based secret projection used by this model."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                _MARKER
                if any(
                    token in str(key).lower()
                    for token in (
                        "secret",
                        "password",
                        "passwd",
                        "token",
                        "credential",
                        "private",
                        "proxy",
                        "authorization",
                        "api_key",
                        "api-key",
                    )
                )
                else _redacted_mapping(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_redacted_mapping(item) for item in value]
    return value


def _flatten(value: object, path: str, output: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        if not value and path:
            output[path] = {}
        for key in sorted(value):
            _flatten(value[key], f"{path}.{key}" if path else str(key), output)
    else:
        output[path] = value


def _reference_differences(
    category: str, values: Sequence[Mapping[str, object]]
) -> tuple[tuple[object, ...], ...]:
    flattened: list[dict[str, object]] = []
    for value in values:
        item: dict[str, object] = {}
        _flatten(_redacted_mapping(value), "", item)
        flattened.append(item)
    paths = sorted({path for item in flattened for path in item})
    return tuple(
        (category, path, tuple(item.get(path) for item in flattened))
        for path in paths
        if len({canonical_json(item.get(path)) for item in flattened}) > 1
    )


def _reference_provenance_differences(
    selection: Sequence[UUID], records: Mapping[UUID, object]
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    snapshot_values: list[Mapping[str, object]] = []
    configuration_values: list[Mapping[str, object]] = []
    environment_values: list[Mapping[str, object]] = []
    for run_id in selection:
        record = cast(Any, records[run_id])
        manifest = record.manifest
        snapshot_values.append(
            {
                "snapshot_id": record.snapshot_id,
                "provenance": manifest["content_identity"],
            }
        )
        configuration_values.append(manifest["configuration"])
        environment_values.append(manifest["environment_fingerprint"])
    return (
        _reference_differences("snapshot", snapshot_values),
        _reference_differences("configuration", configuration_values),
        _reference_differences("environment", environment_values),
    )


def _expected_alignment(
    selection: Sequence[UUID], records: Mapping[UUID, object]
) -> tuple[date, ...]:
    common: set[date] | None = None
    for run_id in selection:
        record = cast(Any, records[run_id])
        strategy = {session for session, _ in record.strategy_equity}
        benchmark = {session for session, _ in record.benchmark_equity}
        run_sessions = strategy & benchmark
        common = run_sessions if common is None else common & run_sessions
    return tuple(sorted(common or ()))


def _valid_run_ids(count: int) -> tuple[UUID, ...]:
    return tuple(UUID(int=index) for index in range(1, count + 1))


@st.composite
def comparison_cases(draw: st.DrawFn) -> ComparisonCase:
    """Generate successful and rejected comparison selections."""

    record_count = draw(st.integers(min_value=2, max_value=4))
    run_ids = _valid_run_ids(record_count)
    mode = draw(
        st.sampled_from(
            (
                "valid",
                "too_few",
                "too_many",
                "duplicate",
                "failed",
                "missing",
                "no_intersection",
                "corrupt_manifest",
                "corrupt_artifact",
            )
        )
    )
    common_count = draw(st.integers(min_value=1, max_value=4))
    common_start = draw(st.integers(min_value=0, max_value=20))
    common = tuple(
        _BASE_DATE + timedelta(days=common_start + offset)
        for offset in range(common_count)
    )
    prefixes = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=record_count,
            max_size=record_count,
        )
    )
    suffixes = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=record_count,
            max_size=record_count,
        )
    )
    snapshot_variants = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=record_count,
            max_size=record_count,
        )
    )
    configuration_variants = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=record_count,
            max_size=record_count,
        )
    )
    environment_variants = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=record_count,
            max_size=record_count,
        )
    )
    metric_variants = draw(
        st.lists(
            st.integers(min_value=0, max_value=9),
            min_size=record_count,
            max_size=record_count,
        )
    )

    records: dict[UUID, object] = {}
    artifact_payloads: dict[str, bytes] = {}
    for index, run_id in enumerate(run_ids):
        prefix = tuple(
            _BASE_DATE + timedelta(days=common_start - value - 1)
            for value in range(prefixes[index], 0, -1)
        )
        suffix = tuple(
            _BASE_DATE + timedelta(days=common_start + common_count + value)
            for value in range(suffixes[index])
        )
        sessions = (*prefix, *common, *suffix)
        snapshot = "snap_" + f"{index + snapshot_variants[index] + 1:064x}"
        artifact_payload = f"artifact-{index}".encode()
        artifact_checksum = sha256(artifact_payload).hexdigest()
        artifact_payloads[artifact_checksum] = artifact_payload
        manifest: dict[str, object] = {
            "content_identity": {
                "snapshot_id": snapshot,
                "calendar": {
                    "name": "XNYS",
                    "version": str(1 + snapshot_variants[index]),
                },
            },
            "configuration": {
                "strategy": {"position_count": 1 + configuration_variants[index]},
                "secrets": {
                    "https_proxy": f"secret-{index}-{configuration_variants[index]}"
                },
            },
            "environment_fingerprint": {
                "python_version": "3.11",
                "source_revision": f"revision-{environment_variants[index]}",
            },
            "artifacts": ({"checksum": artifact_checksum, "role": "metrics"},),
        }
        strategy_metrics = _metrics(MetricScope.STRATEGY, metric_variants[index])
        benchmark_metrics = _metrics(MetricScope.BENCHMARK, metric_variants[index] + 1)
        record = SimpleNamespace(
            run_id=run_id,
            state="failed" if mode == "failed" and index == 0 else "succeeded",
            snapshot_id=snapshot,
            manifest=manifest,
            manifest_checksum=_manifest_checksum(manifest),
            evaluation_start=sessions[0],
            evaluation_end=sessions[-1],
            evaluation=SimpleNamespace(
                strategy_metrics=strategy_metrics,
                benchmark_metrics=benchmark_metrics,
            ),
            strategy_equity=_curve(sessions, metric_variants[index], 1),
            benchmark_equity=_curve(sessions, metric_variants[index] + 1, 2),
        )
        if mode == "corrupt_manifest" and index == 0:
            record.manifest_checksum = (
                "0" * 64 if record.manifest_checksum != "0" * 64 else "1" * 64
            )
        records[run_id] = record

    if mode == "too_few":
        selection = (run_ids[0],)
    elif mode == "too_many":
        selection = tuple(UUID(int=index) for index in range(1, 12))
    elif mode == "duplicate":
        selection = (run_ids[0], run_ids[0])
    elif mode == "missing":
        selection = (run_ids[0], UUID(int=10_000))
    else:
        selection = run_ids

    if mode == "no_intersection":
        record = records[run_ids[-1]]
        disjoint = _BASE_DATE + timedelta(days=200)
        record.strategy_equity = ((disjoint, Decimal("100000")),)
        record.manifest_checksum = _manifest_checksum(record.manifest)

    if mode == "corrupt_artifact":
        first_record = records[run_ids[0]]
        first_checksum = first_record.manifest["artifacts"][0]["checksum"]  # type: ignore[index]
        artifact_payloads[str(first_checksum)] = b"not-the-checksummed-artifact"

    return ComparisonCase(selection, records, artifact_payloads, mode)


def _is_expected_success(mode: str) -> bool:
    return mode == "valid"


# Feature: quantitative-research-platform, Property 13: Comparison validation and alignment preserve provenance
# Validates: Requirements 12.2–12.14, 13.13–13.14, 17.26–17.27
@settings(max_examples=100, deadline=None)
@given(case=comparison_cases())
def test_comparison_validation_and_alignment_preserve_provenance(
    case: ComparisonCase,
) -> None:
    """Comparison validates its selection and never changes run provenance."""

    service = ComparisonService(
        metadata=MetadataStore(case.records),
        artifacts=ArtifactStore(case.artifact_payloads),
    )
    result = service.compare(case.selection)

    if not _is_expected_success(case.mode):
        assert isinstance(result, Err)
        assert result.errors
        if case.mode in {"corrupt_manifest", "corrupt_artifact"}:
            assert result.errors[0].category.value == "integrity.checksum"
        elif case.mode == "no_intersection":
            assert result.errors[0].category.value == "comparison.selection"
        else:
            assert result.errors[0].category.value == "comparison.selection"
        return

    assert isinstance(result, Ok)
    output = result.value
    assert tuple(run.run_id for run in output.runs) == case.selection
    assert output.aligned_sessions == _expected_alignment(case.selection, case.records)
    assert output.limitation_disclosure == LimitationDisclosure.current()

    expected_snapshot, expected_configuration, expected_environment = (
        _reference_provenance_differences(case.selection, case.records)
    )
    assert _difference_projection(output.snapshot_differences) == expected_snapshot
    assert (
        _difference_projection(output.configuration_differences)
        == expected_configuration
    )
    assert (
        _difference_projection(output.environment_differences) == expected_environment
    )

    for comparison_run, run_id in zip(output.runs, case.selection, strict=True):
        record = cast(Any, case.records[run_id])
        assert comparison_run.run_id == run_id
        assert comparison_run.snapshot_id == record.snapshot_id
        assert comparison_run.original_range == (
            record.evaluation_start,
            record.evaluation_end,
        )
        assert comparison_run.strategy_metrics == record.evaluation.strategy_metrics
        assert comparison_run.benchmark_metrics == record.evaluation.benchmark_metrics
        expected_sessions = set(output.aligned_sessions)
        assert tuple(session for session, _ in comparison_run.strategy_curve) == tuple(
            session
            for session, _ in record.strategy_equity
            if session in expected_sessions
        )
        assert tuple(session for session, _ in comparison_run.benchmark_curve) == tuple(
            session
            for session, _ in record.benchmark_equity
            if session in expected_sessions
        )
        assert all(
            session in expected_sessions for session, _ in comparison_run.strategy_curve
        )
        assert all(
            session in expected_sessions
            for session, _ in comparison_run.benchmark_curve
        )

    assert output.artifact.checksum == sha256(output.artifact.payload).hexdigest()
    assert output.artifact.payload.endswith(b"\n")
