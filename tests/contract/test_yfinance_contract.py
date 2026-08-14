"""Offline yfinance boundary contracts backed by reviewed local fixtures."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from quant_research_platform.application.ports import fetch_with_retry
from quant_research_platform.config.models import RetryPolicyConfig
from quant_research_platform.domain.market import ProviderRequest, SymbolOutcome
from quant_research_platform.infrastructure.yfinance_provider import YFinanceAdapter

_GOLDEN_PATH = Path(__file__).parents[1] / "golden" / "yfinance_contract.json"


def _golden() -> dict[str, Any]:
    value = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("yfinance contract fixture must be a JSON object")
    return value


def _request(case: dict[str, Any]) -> ProviderRequest:
    return ProviderRequest(
        tuple(str(symbol) for symbol in case["symbols"]),
        date.fromisoformat(str(case["start"])),
        date.fromisoformat(str(case["end"])),
    )


def _adapter(download: Callable[..., object]) -> YFinanceAdapter:
    retrieved_at = datetime(2024, 1, 4, 14, tzinfo=UTC)
    return YFinanceAdapter(download=download, now=lambda: retrieved_at)


def _single_symbol_frame(case: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(
        case["columns"], index=pd.to_datetime(case["index"])
    )
    frame.attrs.update(case["attrs"])
    return frame


def _multi_symbol_frame(case: dict[str, Any]) -> pd.DataFrame:
    frames = {
        str(symbol): pd.DataFrame(columns, index=pd.to_datetime(case["index"]))
        for symbol, columns in case["columns"].items()
    }
    frame = pd.concat(frames, axis=1)
    frame.columns = frame.columns.swaplevel(0, 1)
    return frame


def _record_rows(outcome: SymbolOutcome) -> list[dict[str, str]]:
    return [
        {
            "date": record.provider_date.isoformat(),
            "open": str(record.raw_bar.open),
            "high": str(record.raw_bar.high),
            "low": str(record.raw_bar.low),
            "close": str(record.raw_bar.close),
            "adj_close": str(record.raw_bar.adj_close),
            "volume": str(record.raw_bar.volume),
            "dividend": str(record.raw_action.dividend),
            "split_ratio": str(record.raw_action.split_ratio),
        }
        for record in outcome.records
    ]


def test_yfinance_call_options_and_single_symbol_actions_match_golden() -> None:
    case = _golden()["single_symbol_actions"]
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> object:
        calls.append(kwargs)
        return _single_symbol_frame(case)

    result = _adapter(download).fetch_daily(_request(case))

    assert calls == [case["call_options"]]
    assert result.status == case["expected_status"]
    outcome = result.outcomes[0]
    assert outcome.status.value == "success"
    assert _record_rows(outcome) == case["expected_rows"]

    second_record = outcome.records[1]
    additional = second_record.provider_fields["additional_fields"]
    assert additional[case["expected_additional_field"]["name"]] == case[
        "expected_additional_field"
    ]["value"]
    assert second_record.provider_fields["frame_metadata"] == case["attrs"]


def test_yfinance_multi_symbol_frame_preserves_partial_outcomes() -> None:
    case = _golden()["multi_symbol_partial"]

    result = _adapter(lambda **_: _multi_symbol_frame(case)).fetch_daily(
        _request(case)
    )

    assert result.status == case["expected_status"]
    actual_outcomes = [
        {
            "symbol": outcome.symbol,
            "status": outcome.status.value,
            "failure_kind": (
                outcome.failure_kind.value if outcome.failure_kind is not None else None
            ),
            "failure_reason": (
                outcome.failure_reason.value
                if outcome.failure_reason is not None
                else None
            ),
        }
        for outcome in result.outcomes
    ]
    assert actual_outcomes == case["expected_outcomes"]
    actual_dates = [
        record.provider_date.isoformat()
        for record in result.outcomes[0].records
    ]
    assert actual_dates == [
        "2024-01-02",
        "2024-01-03",
    ]


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"fixture HTTP status {status_code}")
        self.status_code = status_code


def test_yfinance_transport_exceptions_map_to_reviewed_failure_outcomes() -> None:
    request = _request(_golden()["single_symbol_actions"])

    for case in _golden()["exception_mapping"]:
        def download(
            *, _status_code: int = int(case["status_code"]), **_: object
        ) -> object:
            raise _StatusError(_status_code)

        outcome = _adapter(download).fetch_daily(request).outcomes[0]

        assert outcome.failure_kind is not None
        assert outcome.failure_reason is not None
        assert outcome.failure_kind.value == case["failure_kind"]
        assert outcome.failure_reason.value == case["failure_reason"]
        assert outcome.errors[0].category.value == case["error_category"]


@pytest.mark.smoke
@pytest.mark.external
def test_external_yfinance_one_batch_smoke_is_explicitly_opt_in() -> None:
    """Exercise one real short request without asserting provider price values."""

    if os.environ.get("QRP_RUN_EXTERNAL_TESTS") != "1":
        pytest.skip("set QRP_RUN_EXTERNAL_TESTS=1 to enable yfinance network smoke")

    request = ProviderRequest(("SPY",), date(2024, 1, 2), date(2024, 1, 3))
    retry = RetryPolicyConfig()
    result = fetch_with_retry(
        YFinanceAdapter(), request, retry, sleep=lambda _delay: None
    )

    assert result.request == request
    assert tuple(outcome.symbol for outcome in result.outcomes) == ("SPY",)
    outcome = result.outcomes[0]
    assert outcome.attempts <= retry.attempts
    assert outcome.status.value == "success"
    assert all(
        request.start <= record.provider_date <= request.end
        for record in outcome.records
    )
