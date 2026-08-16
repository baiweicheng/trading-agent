"""Pinned XNYS session and schedule adapter.

This module is the sole boundary around :mod:`exchange_calendars` for Phase 1.
It exposes date-based XNYS session operations while retaining the provider's
UTC market opens and closes in the canonical schedule digest used by snapshot
identity.  Expected sessions are deliberately restricted to sessions whose
official close had occurred at the caller's retrieval time; later sessions are
warned about rather than represented as data gaps.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from importlib.metadata import version as installed_package_version
from typing import Any, Final, Protocol, cast, runtime_checkable
from warnings import warn

import exchange_calendars as exchange_calendars  # type: ignore[import-untyped]

from quant_research_platform.domain.canonical import sha256_canonical_json

CALENDAR_NAME: Final = "XNYS"
EXCHANGE_CALENDARS_DISTRIBUTION: Final = "exchange-calendars"


class IncompleteSessionWarning(UserWarning):
    """A requested XNYS session was excluded because its close is not complete."""


@runtime_checkable
class ExchangeCalendar(Protocol):
    """Date-oriented exchange-calendar boundary consumed by platform services."""

    name: str
    version: str

    def sessions(
        self, start: date, end: date, *, completed_at: datetime
    ) -> tuple[date, ...]:
        """Return requested XNYS sessions whose official close has completed."""

    def is_session(self, value: date) -> bool:
        """Return whether *value* is an XNYS session label."""

    def next_session(self, value: date) -> date:
        """Return the first XNYS session after a session or non-session date."""

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]:
        """Return the final XNYS session in each calendar month in the range."""

    def close_timestamp(self, session: date) -> datetime:
        """Return the official XNYS close for *session*, normalized to UTC."""

    def schedule_checksum(self, start: date, end: date) -> str:
        """Return the canonical digest of XNYS schedule rows in the range."""


def _require_date(field_name: str, value: date) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _require_ordered_range(start: date, end: date) -> tuple[date, date]:
    normalized_start = _require_date("start", start)
    normalized_end = _require_date("end", end)
    if normalized_start > normalized_end:
        raise ValueError("start must not be after end")
    return normalized_start, normalized_end


def _require_utc_timestamp(field_name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _to_date(value: Any) -> date:
    converted = value.date()
    if isinstance(converted, datetime) or not isinstance(converted, date):
        raise TypeError("exchange calendar returned a value without a calendar date")
    return cast(date, converted)


def _to_utc_datetime(value: Any) -> datetime:
    converted = value.to_pydatetime()
    if not isinstance(converted, datetime):
        raise TypeError("exchange calendar returned a value without a datetime")
    if converted.tzinfo is None or converted.utcoffset() is None:
        raise ValueError("exchange calendar returned a timezone-naive timestamp")
    return converted.astimezone(UTC)


class XNYSCalendar:
    """The pinned ``exchange_calendars`` XNYS implementation.

    The adapter intentionally has no caller-selectable calendar code.  A
    snapshot's calendar name, installed package version, and canonical schedule
    checksum make the exact session schedule inspectable and reproducible.
    """

    name: str = CALENDAR_NAME

    def __init__(self) -> None:
        self.version = installed_package_version(EXCHANGE_CALENDARS_DISTRIBUTION)
        self._calendar: Any = exchange_calendars.get_calendar(self.name)

    def _sessions_in_range(self, start: date, end: date) -> tuple[date, ...]:
        start, end = _require_ordered_range(start, end)
        return tuple(
            session_date
            for label in self._calendar.sessions
            if start <= (session_date := _to_date(label)) <= end
        )

    def _schedule_rows(self, start: date, end: date) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "session": session.isoformat(),
                "open_utc": _to_utc_datetime(self._calendar.session_open(session))
                .isoformat()
                .replace("+00:00", "Z"),
                "close_utc": self.close_timestamp(session)
                .isoformat()
                .replace("+00:00", "Z"),
            }
            for session in self._sessions_in_range(start, end)
        )

    def sessions(
        self, start: date, end: date, *, completed_at: datetime
    ) -> tuple[date, ...]:
        """Return sessions closed by ``completed_at`` without inventing gaps.

        An official close exactly equal to ``completed_at`` is complete.  If
        any requested sessions are future or currently incomplete, they are
        omitted and one warning identifies the omitted range/count.
        """

        completed_at = _require_utc_timestamp("completed_at", completed_at)
        requested = self._sessions_in_range(start, end)
        completed = tuple(
            session
            for session in requested
            if self.close_timestamp(session) <= completed_at
        )
        excluded_count = len(requested) - len(completed)
        if excluded_count:
            first_excluded = requested[len(completed)]
            warn(
                (
                    f"Excluded {excluded_count} future or incomplete XNYS session(s) "
                    f"from {first_excluded.isoformat()} through "
                    f"{requested[-1].isoformat()}; their official close is after "
                    f"completed_at={completed_at.isoformat().replace('+00:00', 'Z')}."
                ),
                IncompleteSessionWarning,
                stacklevel=2,
            )
        return completed

    def is_session(self, value: date) -> bool:
        """Return whether *value* maps to an official XNYS session label."""

        value = _require_date("value", value)
        try:
            return bool(self._calendar.is_session(value))
        except (IndexError, ValueError):
            return False

    def next_session(self, value: date) -> date:
        """Return the next XNYS session after *value*.

        A non-session date resolves to its following session, while a session
        resolves strictly forward to the next session.
        """

        value = _require_date("value", value)
        if self.is_session(value):
            return _to_date(self._calendar.next_session(value))
        return _to_date(self._calendar.date_to_session(value, direction="next"))

    def month_end_sessions(self, start: date, end: date) -> tuple[date, ...]:
        """Return each calendar month's final XNYS session in range order."""

        month_ends: list[date] = []
        for session in self._sessions_in_range(start, end):
            if month_ends and (session.year, session.month) == (
                month_ends[-1].year,
                month_ends[-1].month,
            ):
                month_ends[-1] = session
            else:
                month_ends.append(session)
        return tuple(month_ends)

    def close_timestamp(self, session: date) -> datetime:
        """Return XNYS's official session close as an aware UTC timestamp."""

        session = _require_date("session", session)
        if not self.is_session(session):
            raise ValueError(f"{session.isoformat()} is not an XNYS session")
        return _to_utc_datetime(self._calendar.session_close(session))

    def schedule_checksum(self, start: date, end: date) -> str:
        """Hash canonical ordered ``(session, open_utc, close_utc)`` rows."""

        return sha256_canonical_json(list(self._schedule_rows(start, end)))


XNYSCalendarAdapter = XNYSCalendar


__all__ = [
    "CALENDAR_NAME",
    "EXCHANGE_CALENDARS_DISTRIBUTION",
    "ExchangeCalendar",
    "IncompleteSessionWarning",
    "XNYSCalendar",
    "XNYSCalendarAdapter",
]
