"""Reviewed causal-action and data-quality golden contract tests.

These tests intentionally load JSON fixture inputs into the real domain models,
then run the production normalizer and validator.  Expectations are field-level
canonical projections and report checksums rather than rendered snapshots.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_research_platform.domain.canonical import (
    canonical_decimal,
    canonical_timestamp,
)
from quant_research_platform.domain.market import (
    DailyBarCandidate,
    DateRange,
    ProviderRecord,
    ProviderRequest,
    QuarantineRecord,
    RawCorporateAction,
    RawDailyBar,
)
from quant_research_platform.domain.normalization import (
    CausalForwardAdjustmentV1,
    Normalizer,
)
from quant_research_platform.domain.validation import ValidationService
from quant_research_platform.infrastructure.xnys_calendar import XNYSCalendar

_ACTIONS_PATH = Path(__file__).parent / "actions" / "causal_forward_v1.json"
_QUALITY_PATH = Path(__file__).parent / "quality_issues" / "validation_cases.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path.name} must contain a JSON object")
    return value


def _decimal(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _request(case: Mapping[str, Any]) -> ProviderRequest:
    request = case["request"]
    return ProviderRequest(
        tuple(str(symbol) for symbol in request["symbols"]),
        date.fromisoformat(str(request["start"])),
        date.fromisoformat(str(request["end"])),
    )


def _records(case: Mapping[str, Any]) -> tuple[ProviderRecord, ...]:
    request = _request(case)
    result: list[ProviderRecord] = []
    for raw in case["records"]:
        bar = raw["raw_bar"]
        action = raw["raw_action"]
        result.append(
            ProviderRecord(
                provider="yfinance",
                request_content_key=request.content_key,
                symbol=str(raw["symbol"]),
                raw_bar=RawDailyBar(
                    provider_date=date.fromisoformat(str(raw["provider_date"])),
                    open=_decimal(bar["open"]),
                    high=_decimal(bar["high"]),
                    low=_decimal(bar["low"]),
                    close=_decimal(bar["close"]),
                    adj_close=_decimal(bar["adj_close"]),
                    volume=_decimal(bar["volume"]),
                ),
                raw_action=RawCorporateAction(
                    dividend=_decimal(action["dividend"]),
                    split_ratio=_decimal(action["split_ratio"]),
                    provider_fields=dict(action.get("provider_fields", {})),
                ),
                provider_fields=dict(raw.get("provider_fields", {})),
            )
        )
    return tuple(result)


def _canonical_number(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Decimal):
        raise AssertionError(f"expected a Decimal value, got {type(value).__name__}")
    return canonical_decimal(value)


def _candidate_projection(candidate: DailyBarCandidate) -> dict[str, Any]:
    """Project every scientific adjustment field used by the reviewed fixture."""

    return {
        "symbol": candidate.symbol,
        "session": candidate.session.isoformat(),
        "event_timestamp": canonical_timestamp(candidate.event_timestamp),
        "raw_close": _canonical_number(candidate.raw_close),
        "raw_volume": _canonical_number(candidate.raw_volume),
        "dividend": _canonical_number(candidate.corporate_action.dividend),
        "split_ratio": _canonical_number(candidate.corporate_action.split_ratio),
        "source_fields": list(candidate.corporate_action.source_fields),
        "adjusted": {
            "open": _canonical_number(candidate.adjusted_open),
            "high": _canonical_number(candidate.adjusted_high),
            "low": _canonical_number(candidate.adjusted_low),
            "close": _canonical_number(candidate.adjusted_close),
            "volume": _canonical_number(candidate.adjusted_volume),
        },
        "execution_adjusted_open": _canonical_number(candidate.execution_adjusted_open),
        "sizing_adjusted_close": _canonical_number(candidate.sizing_adjusted_close),
        "cumulative_price_factor": _canonical_number(candidate.cumulative_price_factor),
        "cumulative_split_factor": _canonical_number(candidate.cumulative_split_factor),
        "policy_version": candidate.policy_version,
        "provider_record_checksum": candidate.raw_lineage.provider_record_checksum,
        "canonical_row_checksum": candidate.canonical_row_checksum,
    }


def _quarantine_projection(record: QuarantineRecord) -> dict[str, Any]:
    return {
        "source_kind": record.source_kind.value,
        "symbol": record.symbol,
        "session": record.session.isoformat() if record.session else None,
        "reason_codes": list(record.reason_codes),
        "policy_version": record.policy_version,
        "offending_values": {
            str(key): _canonical_number(value)
            if isinstance(value, Decimal)
            else (value.isoformat() if isinstance(value, date) else value)
            for key, value in record.offending_values.items()
        },
        "candidate_checksum": record.candidate_checksum,
    }


def _assert_candidate_fields(
    actual: Iterable[DailyBarCandidate], expected: Iterable[Mapping[str, Any]]
) -> None:
    actual_rows = [_candidate_projection(candidate) for candidate in actual]
    expected_rows = list(expected)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        for field, expected_value in expected_row.items():
            if not expected_value:
                raise AssertionError(
                    f"fixture expectation for {field} must contain a reviewed value"
                )
            if field == "adjusted":
                for adjusted_field, adjusted_value in expected_value.items():
                    assert actual_row[field][adjusted_field] == adjusted_value
            elif field == "adjusted_close":
                assert actual_row["adjusted"]["close"] == expected_value
            else:
                assert actual_row[field] == expected_value


def _assert_quarantine_fields(
    actual: Iterable[QuarantineRecord], expected: Iterable[Mapping[str, Any]]
) -> None:
    actual_rows = [_quarantine_projection(record) for record in actual]
    expected_rows = list(expected)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        for field in ("source_kind", "symbol", "session", "reason_codes"):
            if field in expected_row:
                assert actual_row[field] == expected_row[field]
        for field, expected_value in expected_row.get("offending_values", {}).items():
            assert actual_row["offending_values"][field] == expected_value
        for field in ("policy_version", "candidate_checksum"):
            if field in expected_row:
                assert actual_row[field] == expected_row[field]
        for field in (
            "policy_reason",
            "split_ratio",
            "dividend",
            "prior_raw_close",
            "reference_price",
        ):
            if field in expected_row:
                assert actual_row["offending_values"][field] == expected_row[field]


def _assert_candidate_source_lineage(
    candidates: Iterable[DailyBarCandidate], records: Iterable[ProviderRecord]
) -> None:
    by_checksum = {record.provider_record_checksum: record for record in records}
    for candidate in candidates:
        source = by_checksum[candidate.raw_lineage.provider_record_checksum]
        assert candidate.raw_bar == source.raw_bar
        assert candidate.raw_action == source.raw_action
        assert candidate.raw_lineage == source.raw_lineage


def _normalization_case(
    case: Mapping[str, Any], calendar: XNYSCalendar
) -> tuple[tuple[DailyBarCandidate, ...], tuple[QuarantineRecord, ...]]:
    values = tuple(Normalizer().normalize(_records(case), calendar))
    candidates = tuple(
        value for value in values if isinstance(value, DailyBarCandidate)
    )
    quarantines = tuple(
        value for value in values if isinstance(value, QuarantineRecord)
    )
    _assert_candidate_source_lineage(candidates, _records(case))
    return candidates, quarantines


def test_action_fixture_declares_and_applies_causal_policy() -> None:
    fixture = _load_json(_ACTIONS_PATH)
    calendar = XNYSCalendar()
    assert calendar.name == fixture["calendar_name"]
    assert calendar.version == fixture["calendar_version"]

    policy = CausalForwardAdjustmentV1()
    policy_fixture = fixture["policy"]
    assert policy.version == policy_fixture["version"]
    assert policy.decimal_precision == policy_fixture["decimal_precision"]
    assert policy.rounding_mode == policy_fixture["rounding_mode"]
    policy_content = policy.to_content_dict()
    assert policy_content["source_fields"] == policy_fixture["source_fields"]
    assert policy_content["equations"] == policy_fixture["equations"]

    case = fixture["cases"]["split_dividend_and_later_action"]
    records = _records(case)
    candidates, quarantines = _normalization_case(case, calendar)
    assert not quarantines
    actual = [_candidate_projection(candidate) for candidate in candidates]
    assert actual == case["expected_candidates"]

    prefix_end = date.fromisoformat(case["causal_prefix_sessions"][-1])
    prefix_records = tuple(
        record for record in records if record.provider_date <= prefix_end
    )
    prefix_candidates, prefix_quarantines = _normalization_case(
        {
            **case,
            "records": [
                {
                    "symbol": record.symbol,
                    "provider_date": record.provider_date.isoformat(),
                    "raw_bar": {
                        name: _canonical_number(getattr(record.raw_bar, name))
                        for name in (
                            "open",
                            "high",
                            "low",
                            "close",
                            "adj_close",
                            "volume",
                        )
                    },
                    "raw_action": {
                        "dividend": _canonical_number(record.raw_action.dividend),
                        "split_ratio": _canonical_number(record.raw_action.split_ratio),
                        "provider_fields": dict(record.raw_action.provider_fields),
                    },
                    "provider_fields": dict(record.provider_fields),
                }
                for record in prefix_records
            ],
        },
        calendar,
    )
    assert not prefix_quarantines
    assert [_candidate_projection(candidate) for candidate in prefix_candidates] == (
        actual[:3]
    )


def test_action_fixture_policy_failures_are_explicit_and_non_fabricating() -> None:
    fixture = _load_json(_ACTIONS_PATH)
    calendar = XNYSCalendar()
    for name in ("invalid_split_ratio", "invalid_dividend_equation"):
        case = fixture["cases"][name]
        candidates, quarantines = _normalization_case(case, calendar)
        expected_candidates = case["expected_candidates"]
        _assert_candidate_fields(candidates, expected_candidates)
        _assert_quarantine_fields(quarantines, case["expected_policy_quarantines"])
        assert all(
            record.primary_reason == "normalization.policy" for record in quarantines
        )


def _expected_sessions(case: Mapping[str, Any]) -> dict[str, tuple[date, ...]]:
    return {
        str(symbol): tuple(date.fromisoformat(str(value)) for value in sessions)
        for symbol, sessions in case["expected_sessions"].items()
    }


def _actual_duplicate_projection(output: Any) -> list[dict[str, Any]]:
    return [
        {"symbol": key.symbol, "session": key.session.isoformat(), "count": count}
        for key, count in output.duplicate_counts
    ]


def _actual_gap_projection(output: Any) -> list[dict[str, Any]]:
    return [
        {
            "symbol": gap.symbol,
            "expected_session": gap.expected_session.isoformat(),
            "parent_retained": gap.parent_retained,
            "reason": gap.reason,
        }
        for gap in output.gaps
    ]


def test_quality_fixture_runs_real_validator_and_matches_canonical_report() -> None:
    fixture = _load_json(_QUALITY_PATH)
    calendar = XNYSCalendar()
    assert calendar.name == fixture["calendar_name"]
    assert calendar.version == fixture["calendar_version"]

    for case in fixture["cases"].values():
        records = _records(case)
        normalized = tuple(Normalizer().normalize(records, calendar))
        output = ValidationService(calendar=calendar).validate(
            normalized,
            _expected_sessions(case),
            int(case["staleness_threshold"]),
            requested_range=DateRange(
                date.fromisoformat(str(case["request"]["start"])),
                date.fromisoformat(str(case["request"]["end"])),
            ),
            comparison_range=(
                DateRange(
                    date.fromisoformat(str(case["comparison_range"]["start"])),
                    date.fromisoformat(str(case["comparison_range"]["end"])),
                )
                if "comparison_range" in case
                else None
            ),
        )
        expected = case["expected"]
        _assert_candidate_fields(output.accepted_rows, expected["accepted"])
        _assert_quarantine_fields(output.quarantined_rows, expected["quarantines"])
        assert _actual_duplicate_projection(output) == expected["duplicate_counts"]
        assert _actual_gap_projection(output) == expected["gaps"]
        assert [summary.to_content_dict() for summary in output.per_symbol] == expected[
            "per_symbol"
        ]
        assert [
            {"reason": reason, "count": count}
            for reason, count in output.report.quarantined_by_reason
        ] == expected["quarantined_by_reason"]
        assert output.report.content_checksum == expected["report_checksum"]


def test_quality_fixture_is_deterministic_under_record_permutation() -> None:
    fixture = _load_json(_QUALITY_PATH)
    calendar = XNYSCalendar()
    for case in fixture["cases"].values():
        records = _records(case)
        expected = _expected_sessions(case)
        kwargs: dict[str, Any] = {
            "requested_range": DateRange(
                date.fromisoformat(str(case["request"]["start"])),
                date.fromisoformat(str(case["request"]["end"])),
            ),
        }
        if "comparison_range" in case:
            kwargs["comparison_range"] = DateRange(
                date.fromisoformat(str(case["comparison_range"]["start"])),
                date.fromisoformat(str(case["comparison_range"]["end"])),
            )
        first = ValidationService(calendar=calendar).validate(
            tuple(Normalizer().normalize(records, calendar)),
            expected,
            int(case["staleness_threshold"]),
            **kwargs,
        )
        second = ValidationService(calendar=calendar).validate(
            tuple(Normalizer().normalize(tuple(reversed(records)), calendar)),
            expected,
            int(case["staleness_threshold"]),
            **kwargs,
        )
        assert first.report.to_content_dict() == second.report.to_content_dict()
        assert first.report.content_checksum == second.report.content_checksum
        assert tuple(
            row.canonical_row_checksum for row in first.accepted_rows
        ) == tuple(row.canonical_row_checksum for row in second.accepted_rows)
