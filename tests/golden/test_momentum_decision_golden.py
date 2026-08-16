"""Reviewed XNYS momentum and causal decision-artifact golden tests."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from quant_research_platform.application.decisions import CausalDecisionDelivery
from quant_research_platform.domain.canonical import canonical_json, sha256_bytes
from quant_research_platform.domain.errors import Ok
from quant_research_platform.domain.strategy import (
    RationalWeight,
    monthly_momentum_v1,
)
from quant_research_platform.infrastructure.xnys_calendar import XNYSCalendar

_FIXTURE_PATH = Path(__file__).parent / "daily_clean" / "daily_clean.json"
_COMPLETED_AT = datetime(2025, 1, 10, tzinfo=UTC)


def _load_fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("daily_clean fixture must contain an object")
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sessions(fixture: dict[str, Any], calendar: XNYSCalendar) -> tuple[date, ...]:
    window = fixture["session_window"]
    sessions = calendar.sessions(
        date.fromisoformat(str(window["start"])),
        date.fromisoformat(str(window["end"])),
        completed_at=_COMPLETED_AT,
    )
    assert len(sessions) == window["count"]
    assert sessions[0].isoformat() == window["start"]
    assert sessions[-1].isoformat() == window["end"]
    return sessions


def _history_rows(
    fixture: dict[str, Any],
    sessions: tuple[date, ...],
    *,
    post_signal_multiplier: int = 1,
) -> tuple[dict[str, object], ...]:
    signal = date.fromisoformat(str(fixture["signal_session"]))
    execution = date.fromisoformat(str(fixture["execution_session"]))
    rows: list[dict[str, object]] = []
    symbols = tuple(fixture["universe"]) + (str(fixture["benchmark_symbol"]),)
    for symbol in symbols:
        values = fixture["symbols"][symbol]
        for index, session in enumerate(sessions):
            adjusted_close = (
                values["long_close"]
                if index == 1
                else values["short_close"]
                if index == len(sessions) - 22
                else values["default_close"]
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "adjusted_close": adjusted_close,
                    "sizing_adjusted_close": (
                        values["signal_sizing_close"]
                        if session == signal
                        else adjusted_close
                    ),
                    "tradable": True,
                }
            )
        post_signal_close = str(
            Decimal(str(values["post_signal_close"])) * Decimal(post_signal_multiplier)
        )
        rows.append(
            {
                "symbol": symbol,
                "session": execution,
                "adjusted_close": post_signal_close,
                "sizing_adjusted_close": values["signal_sizing_close"],
                "tradable": True,
            }
        )
    return tuple(rows)


def _decision_projection(decisions: tuple[Any, ...]) -> list[object]:
    return [_jsonable(decision.to_serializable()) for decision in decisions]


def _endpoint_projection(decisions: tuple[Any, ...]) -> list[dict[str, object]]:
    return [
        {
            "symbol": decision.symbol,
            "long_session": (
                decision.endpoint_252_session.isoformat()
                if decision.endpoint_252_session is not None
                else None
            ),
            "long_close": (
                str(decision.endpoint_252_close)
                if decision.endpoint_252_close is not None
                else None
            ),
            "long_checksum": decision.endpoint_252_checksum,
            "short_session": (
                decision.endpoint_21_session.isoformat()
                if decision.endpoint_21_session is not None
                else None
            ),
            "short_close": (
                str(decision.endpoint_21_close)
                if decision.endpoint_21_close is not None
                else None
            ),
            "short_checksum": decision.endpoint_21_checksum,
        }
        for decision in decisions
    ]


def _intent_projection(intents: tuple[Any, ...]) -> list[object]:
    return [_jsonable(intent.to_serializable()) for intent in intents]


def _decision_artifact(decisions: tuple[Any, ...]) -> dict[str, object]:
    return {"role": "strategy_decisions", "rows": _decision_projection(decisions)}


def _intent_artifact(intents: tuple[Any, ...]) -> dict[str, object]:
    return {"role": "order_intents", "rows": _intent_projection(intents)}


def _fixture_inputs(
    fixture: dict[str, Any],
) -> tuple[date, tuple[str, ...], int, object, dict[str, object]]:
    signal = date.fromisoformat(str(fixture["signal_session"]))
    universe = tuple(str(symbol) for symbol in fixture["universe"])
    snapshot = SimpleNamespace(snapshot_id=str(fixture["snapshot_id"]))
    portfolio = {
        "cash_balance": Decimal(str(fixture["portfolio"]["cash_balance"])),
        "positions": dict(fixture["portfolio"]["positions"]),
    }
    return signal, universe, int(fixture["position_count"]), snapshot, portfolio


def test_daily_clean_momentum_boundary_matches_reviewed_decision_artifact() -> None:
    fixture = _load_fixture()
    calendar = XNYSCalendar()
    assert calendar.name == fixture["calendar"]["name"]
    assert calendar.version == fixture["calendar"]["version"]

    sessions = _sessions(fixture, calendar)
    signal, universe, position_count, _, _ = _fixture_inputs(fixture)
    assert signal == sessions[-1]
    assert calendar.month_end_sessions(sessions[0], signal)[-1] == signal
    assert sessions[-22] == date(2024, 11, 29)
    assert calendar.next_session(signal).isoformat() == fixture["execution_session"]

    rows = _history_rows(fixture, sessions)
    decisions = monthly_momentum_v1(
        tuple(row for row in rows if row["symbol"] in universe),
        signal_session=signal,
        universe=universe,
        position_count=position_count,
    )
    expected = fixture["expected"]

    assert _endpoint_projection(decisions) == expected["endpoint_rows"]
    assert _decision_projection(decisions) == expected["decisions"]
    assert [decision.rank for decision in decisions] == [1, 2, 3]
    assert [decision.target_weight.to_canonical_string() for decision in decisions] == [
        "1/2",
        "1/2",
        "0/1",
    ]
    assert RationalWeight.sum(
        decision.target_weight for decision in decisions
    ) == RationalWeight(1, 1)
    assert (
        sha256_bytes(canonical_json(_decision_artifact(decisions)))
        == expected["decision_artifact_checksum"]
    )


def test_daily_clean_causal_delivery_and_deterministic_artifacts() -> None:
    fixture = _load_fixture()
    calendar = XNYSCalendar()
    sessions = _sessions(fixture, calendar)
    signal, universe, position_count, snapshot, portfolio = _fixture_inputs(fixture)
    baseline_rows = _history_rows(fixture, sessions)
    changed_after_signal_rows = _history_rows(
        fixture, sessions, post_signal_multiplier=2
    )

    class Reader:
        def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
            self.rows = rows
            self.calls: list[dict[str, object]] = []

        def read_history(
            self,
            snapshot_handle: object,
            *,
            symbols: tuple[str, ...],
            end_session: date,
            fields: tuple[str, ...],
            start_session: date,
        ) -> tuple[dict[str, object], ...]:
            assert snapshot_handle is snapshot
            self.calls.append(
                {
                    "symbols": symbols,
                    "end_session": end_session,
                    "fields": fields,
                    "start_session": start_session,
                }
            )
            assert end_session == signal
            assert max(row["session"] for row in self.rows) == date.fromisoformat(
                str(fixture["execution_session"])
            )
            return self.rows

    first_reader = Reader(baseline_rows)
    first_result = CausalDecisionDelivery(first_reader, calendar=calendar).deliver(
        snapshot,
        signal,
        portfolio,
        universe=universe,
        position_count=position_count,
    )
    assert isinstance(first_result, Ok), first_result
    first = first_result.value
    expected = fixture["expected"]

    assert first_reader.calls[0]["symbols"] == universe
    assert first_reader.calls[0]["end_session"] == signal
    assert first_reader.calls[0]["fields"] == (
        "symbol",
        "session",
        "adjusted_close",
        "sizing_adjusted_close",
        "canonical_row_checksum",
        "tradable",
    )
    assert first.marked_equity == Decimal(expected["marked_equity"])
    assert _decision_projection(first.decisions) == expected["decisions"]
    assert _intent_projection(first.order_intents) == expected["order_intents"]
    assert [intent.symbol for intent in first.order_intents] == [
        "PG",
        "AAPL",
        "MSFT",
    ]
    assert [intent.requested_quantity for intent in first.order_intents] == [
        -40,
        8,
        20,
    ]
    assert (
        sha256_bytes(canonical_json(_intent_artifact(first.order_intents)))
        == expected["order_intent_artifact_checksum"]
    )
    assert (
        sha256_bytes(canonical_json(first.to_serializable()))
        == expected["delivery_artifact_checksum"]
    )
    assert first.run_inputs.to_serializable() == {
        "long_lookback_sessions": 252,
        "policy_version": "causal_forward_v1",
        "position_count": 2,
        "skip_recent_sessions": 21,
        "snapshot_id": fixture["snapshot_id"],
        "strategy_identifier": "monthly_momentum_v1",
    }
    assert first.decision_book.reveal(signal) == first.decisions
    next_session = date.fromisoformat(str(fixture["execution_session"]))
    assert first.decision_book.reveal(next_session) == ()

    changed_reader = Reader(changed_after_signal_rows)
    changed_result = CausalDecisionDelivery(changed_reader, calendar=calendar).deliver(
        snapshot,
        signal,
        portfolio,
        universe=universe,
        position_count=position_count,
    )
    assert isinstance(changed_result, Ok), changed_result
    assert changed_result.value.to_serializable() == first.to_serializable()
    assert changed_reader.calls[0]["end_session"] == signal


def test_daily_clean_exact_warmup_has_no_order_intents() -> None:
    fixture = _load_fixture()
    calendar = XNYSCalendar()
    sessions = _sessions(fixture, calendar)
    signal, universe, position_count, snapshot, portfolio = _fixture_inputs(fixture)
    incomplete_rows = tuple(
        row for row in _history_rows(fixture, sessions) if row["session"] != sessions[0]
    )

    result = CausalDecisionDelivery(calendar=calendar).deliver(
        snapshot,
        signal,
        portfolio,
        universe=universe,
        position_count=position_count,
        history=incomplete_rows,
    )

    assert isinstance(result, Ok), result
    assert len(incomplete_rows) > 0
    assert all(
        decision.exclusion_reason.value == "warm_up_incomplete"
        for decision in result.value.decisions
    )
    assert result.value.order_intents == ()
    assert result.value.decision_book.reveal(signal) == result.value.decisions
