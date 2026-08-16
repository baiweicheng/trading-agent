"""Pinned XNYS schedule contracts backed by reviewed local fixtures."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from importlib.metadata import version as installed_package_version
from pathlib import Path
from typing import Any

import pytest

from quant_research_platform.domain.canonical import sha256_canonical_json
from quant_research_platform.infrastructure.xnys_calendar import (
    IncompleteSessionWarning,
    XNYSCalendar,
)

_GOLDEN_PATH = Path(__file__).parents[1] / "golden" / "xnys_calendar_contract.json"


def _golden() -> dict[str, Any]:
    value = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("XNYS calendar contract fixture must be a JSON object")
    return value


def _date(value: object) -> date:
    return date.fromisoformat(str(value))


def _utc(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def test_xnys_schedule_version_and_digest_match_reviewed_fixture() -> None:
    golden = _golden()
    schedule = golden["digest_fixture"]
    calendar = XNYSCalendar()

    assert calendar.name == golden["calendar_name"]
    assert calendar.version == golden["package_version"]
    assert calendar.version == installed_package_version(golden["package_distribution"])

    expected_digest = sha256_canonical_json(schedule["rows"])
    assert (
        calendar.schedule_checksum(_date(schedule["start"]), _date(schedule["end"]))
        == expected_digest
    )
    assert (
        calendar.schedule_checksum(_date(schedule["start"]), _date(schedule["end"]))
        == expected_digest
    )

    for row in schedule["rows"]:
        assert calendar.close_timestamp(_date(row["session"])) == _utc(row["close_utc"])


def test_xnys_session_membership_month_ends_and_completed_cutoff_match_golden() -> None:
    golden = _golden()
    calendar = XNYSCalendar()
    membership = golden["session_membership"]

    assert all(calendar.is_session(_date(value)) for value in membership["sessions"])
    assert not any(
        calendar.is_session(_date(value)) for value in membership["non_sessions"]
    )
    assert [
        calendar.next_session(_date(case["value"])).isoformat()
        for case in membership["next_sessions"]
    ] == [case["next"] for case in membership["next_sessions"]]

    month_ends = golden["month_ends"]
    assert [
        session.isoformat()
        for session in calendar.month_end_sessions(
            _date(month_ends["start"]), _date(month_ends["end"])
        )
    ] == month_ends["sessions"]

    for case in golden["close_timestamps"]:
        assert calendar.close_timestamp(_date(case["session"])) == _utc(
            case["close_utc"]
        )

    completed = golden["completed_range"]
    with pytest.warns(IncompleteSessionWarning, match="future or incomplete"):
        actual_sessions = calendar.sessions(
            _date(completed["start"]),
            _date(completed["end"]),
            completed_at=_utc(completed["completed_at"]),
        )
    assert [session.isoformat() for session in actual_sessions] == completed[
        "expected_sessions"
    ]
