"""Offline fixtures for the pinned XNYS calendar adapter."""

from __future__ import annotations

from datetime import UTC, date, datetime
from importlib.metadata import version as installed_package_version

import pytest

from quant_research_platform.domain.canonical import sha256_canonical_json
from quant_research_platform.infrastructure.xnys_calendar import (
    EXCHANGE_CALENDARS_DISTRIBUTION,
    ExchangeCalendar,
    IncompleteSessionWarning,
    XNYSCalendar,
)


@pytest.fixture
def calendar() -> XNYSCalendar:
    return XNYSCalendar()


def test_adapter_exposes_pinned_xnys_identity_and_session_membership(
    calendar: XNYSCalendar,
) -> None:
    assert isinstance(calendar, ExchangeCalendar)
    assert calendar.name == "XNYS"
    assert calendar.version == installed_package_version(
        EXCHANGE_CALENDARS_DISTRIBUTION
    )

    assert calendar.is_session(date(2024, 7, 3))
    assert not calendar.is_session(date(2024, 7, 4))  # Independence Day
    assert not calendar.is_session(date(2024, 7, 6))  # Saturday
    assert calendar.next_session(date(2024, 7, 3)) == date(2024, 7, 5)
    assert calendar.next_session(date(2024, 7, 4)) == date(2024, 7, 5)


def test_official_utc_closes_cover_daylight_saving_and_shortened_sessions(
    calendar: XNYSCalendar,
) -> None:
    assert calendar.close_timestamp(date(2024, 3, 8)) == datetime(
        2024, 3, 8, 21, 0, tzinfo=UTC
    )
    assert calendar.close_timestamp(date(2024, 3, 11)) == datetime(
        2024, 3, 11, 20, 0, tzinfo=UTC
    )
    assert calendar.close_timestamp(date(2024, 11, 29)) == datetime(
        2024, 11, 29, 18, 0, tzinfo=UTC
    )


def test_completed_sessions_stop_at_official_close_and_warn_for_incomplete_range(
    calendar: XNYSCalendar,
) -> None:
    with pytest.warns(IncompleteSessionWarning, match="future or incomplete"):
        completed = calendar.sessions(
            date(2024, 11, 29),
            date(2024, 12, 2),
            completed_at=datetime(2024, 11, 29, 18, 0, tzinfo=UTC),
        )

    assert completed == (date(2024, 11, 29),)

    with pytest.warns(IncompleteSessionWarning):
        before_early_close = calendar.sessions(
            date(2024, 11, 29),
            date(2024, 11, 29),
            completed_at=datetime(2024, 11, 29, 17, 59, 59, tzinfo=UTC),
        )
    assert before_early_close == ()


def test_month_end_sessions_and_canonical_schedule_digest_use_reviewed_rows(
    calendar: XNYSCalendar,
) -> None:
    assert calendar.month_end_sessions(date(2024, 5, 1), date(2024, 7, 31)) == (
        date(2024, 5, 31),
        date(2024, 6, 28),
        date(2024, 7, 31),
    )

    expected_rows = [
        {
            "session": "2024-03-08",
            "open_utc": "2024-03-08T14:30:00Z",
            "close_utc": "2024-03-08T21:00:00Z",
        },
        {
            "session": "2024-03-11",
            "open_utc": "2024-03-11T13:30:00Z",
            "close_utc": "2024-03-11T20:00:00Z",
        },
    ]
    assert calendar.schedule_checksum(
        date(2024, 3, 8), date(2024, 3, 11)
    ) == sha256_canonical_json(expected_rows)


def test_calendar_input_boundaries_are_explicit(calendar: XNYSCalendar) -> None:
    with pytest.raises(ValueError, match="not an XNYS session"):
        calendar.close_timestamp(date(2024, 7, 4))
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.sessions(
            date(2024, 7, 3),
            date(2024, 7, 3),
            completed_at=datetime(2024, 7, 3, 17, 0),
        )
    with pytest.raises(ValueError, match="start must not be after end"):
        calendar.month_end_sessions(date(2024, 7, 5), date(2024, 7, 3))
