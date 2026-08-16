"""Golden split/dividend projections and exactly-once ledger checks.

The fixture is deliberately independent from the bundle adapter's action
projection.  Platform rows are compared with the derived raw-only bundle and
with a small Decimal ledger reference, so a missing, duplicated, or inverted
action cannot make the test pass accidentally.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import pytest

from quant_research_platform.domain.errors import LimitationDisclosure
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotHandle,
    SnapshotManifest,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.infrastructure.zipline_bundle import (
    ZiplineBundleAdapter,
)

_FIXTURE_PATH = Path(__file__).parent / "zipline_actions" / "split_dividend_cases.json"
_MONEY_QUANTUM = Decimal("0.000001")


def _load_fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise AssertionError("split/dividend fixture must contain a cases list")
    return value


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _quantize_money(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANTUM)


def _fixture_rows(case: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in case["sessions"]:
        raw = item["raw"]
        action = item["action"]
        rows.append(
            {
                "symbol": case["symbol"],
                "session": date.fromisoformat(item["session"]),
                "raw_open": _decimal(raw["open"]),
                "raw_high": _decimal(raw["high"]),
                "raw_low": _decimal(raw["low"]),
                "raw_close": _decimal(raw["close"]),
                "raw_volume": _decimal(raw["volume"]),
                "dividend": _decimal(action["dividend"]),
                "split_ratio": _decimal(action["split_ratio"]),
            }
        )
        # The adapter always includes the benchmark in its derived bundle.
        rows.append(
            {
                "symbol": "SPY",
                "session": date.fromisoformat(item["session"]),
                "raw_open": Decimal("400"),
                "raw_high": Decimal("401"),
                "raw_low": Decimal("399"),
                "raw_close": Decimal("400"),
                "raw_volume": Decimal("1000"),
                "dividend": Decimal("0"),
                "split_ratio": Decimal("1"),
            }
        )
    return rows


class _FixtureCalendar:
    name = "XNYS"
    version = "4.13.2"

    def __init__(self, schedule_checksum: str) -> None:
        self._schedule_checksum = schedule_checksum

    def schedule_checksum(self, start: date, end: date) -> str:
        del start, end
        return self._schedule_checksum

    def next_session(self, session: date) -> date:
        return session + timedelta(days=1)


class _FixtureDataSource:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.calls: list[tuple[tuple[str, ...], object]] = []

    def scan(
        self,
        references: object,
        columns: tuple[str, ...],
        *,
        predicate: object,
    ) -> list[dict[str, object]]:
        del references
        self.calls.append((columns, predicate))
        symbols = getattr(predicate, "symbols")
        start = getattr(predicate, "session_start")
        end = getattr(predicate, "session_end")
        return [
            {column: row[column] for column in columns}
            for row in self._rows
            if row["symbol"] in symbols and start <= row["session"] <= end
        ]


class _FixtureSnapshotManager:
    def __init__(self, manifest: SnapshotManifest) -> None:
        self.manifest = manifest
        self.handle = SnapshotHandle.from_manifest(
            manifest, verified_at=datetime(2024, 1, 1, tzinfo=UTC)
        )

    def open_verified(self, snapshot_id: str) -> SnapshotHandle:
        assert snapshot_id == self.handle.snapshot_id
        return self.handle

    def inspect_snapshot(self, snapshot_id: str) -> SnapshotManifest:
        assert snapshot_id == self.handle.snapshot_id
        return self.manifest


def _snapshot(
    case: dict[str, Any], fixture: dict[str, Any]
) -> tuple[SnapshotManifest, list[dict[str, object]]]:
    rows = _fixture_rows(case)
    sessions = tuple(date.fromisoformat(item["session"]) for item in case["sessions"])
    requested_range = DateRange(min(sessions), max(sessions))
    object_ref = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum="d" * 64,
        relative_uri=f"normalized/{case['symbol']}.parquet",
        schema_version="daily_bar_v1",
        row_count=len(rows),
        byte_size=1,
        symbol=case["symbol"],
        session_year=sessions[0].year,
    )
    calendar_info = fixture["calendar"]
    identity = SnapshotContentIdentity(
        provider="yfinance",
        requested_range=requested_range,
        configured_universe=(case["symbol"],),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity(
            name=calendar_info["name"],
            version=calendar_info["version"],
            schedule_checksum=calendar_info["schedule_checksum"],
        ),
        configuration_checksum="b" * 64,
        objects=(object_ref,),
        validation_report_checksum="c" * 64,
        validation_summary=ValidationSummary(
            accepted_row_count=len(rows),
            quarantined_row_count=0,
            collapsed_duplicate_count=0,
            gap_count=0,
            covered_range=requested_range,
            comparison_ready=True,
        ),
        limitation_disclosure=LimitationDisclosure.current(),
        covered_range=requested_range,
    )
    manifest = SnapshotManifest(
        content_identity=identity,
        operational_metadata=OperationalMetadata(
            created_at=datetime(2024, 1, 1, tzinfo=UTC)
        ),
    )
    return manifest, rows


def _ledger_reference(
    case: dict[str, Any], rows: list[dict[str, object]]
) -> dict[str, Decimal | int]:
    ledger = case["ledger"]
    by_session = {
        row["session"]: row for row in rows if row["symbol"] == case["symbol"]
    }
    shares = _decimal(ledger["starting_shares"])
    cash = _decimal(ledger["starting_cash"])
    split_row = by_session[date.fromisoformat(ledger["split_session"])]
    pre_action_close = _decimal(ledger["pre_action_close"])
    split_ratio = _decimal(
        next(
            item["action"]["split_ratio"]
            for item in case["sessions"]
            if item["session"] == ledger["split_session"]
        )
    )
    before_split_shares = shares
    post_split_shares = shares * split_ratio
    actual_shares = post_split_shares.to_integral_value(rounding=ROUND_FLOOR)
    cash_in_lieu = (post_split_shares - actual_shares) * _decimal(
        split_row["raw_close"]
    )
    cash += cash_in_lieu
    shares = actual_shares
    split_value_before = before_split_shares * pre_action_close
    split_value_after = shares * _decimal(split_row["raw_close"]) + cash_in_lieu

    dividend_row = by_session[date.fromisoformat(ledger["dividend_session"])]
    dividend = _decimal(
        next(
            item["action"]["dividend"]
            for item in case["sessions"]
            if item["session"] == ledger["dividend_session"]
        )
    )
    dividend_cash = shares * dividend
    cash += dividend_cash
    return {
        "actual_shares": int(shares),
        "cash_in_lieu": _quantize_money(cash_in_lieu),
        "split_value_before": _quantize_money(split_value_before),
        "split_value_after": _quantize_money(split_value_after),
        "dividend_cash": _quantize_money(dividend_cash),
        "total_cash": _quantize_money(cash),
        "dividend_close": _decimal(dividend_row["raw_close"]),
    }


def _platform_actions(case: dict[str, Any]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "session": item["session"],
            "split_ratio": item["action"]["split_ratio"],
            "dividend": item["action"]["dividend"],
        }
        for item in case["sessions"]
        if item["action"]["split_ratio"] != "1" or item["action"]["dividend"] != "0"
    )


@pytest.mark.integration
def test_split_dividend_golden_bundle_applies_canonical_actions_once(
    tmp_path: Path,
) -> None:
    fixture = _load_fixture()
    for case in fixture["cases"]:
        manifest, rows = _snapshot(case, fixture)
        source = _FixtureDataSource(rows)
        adapter = ZiplineBundleAdapter(
            snapshot_manager=_FixtureSnapshotManager(manifest),
            data_source=source,
            calendar=_FixtureCalendar(fixture["calendar"]["schedule_checksum"]),
            cache_root=tmp_path / case["name"],
        )

        result = adapter.materialize(manifest.snapshot_id)
        assert hasattr(result, "value"), getattr(result, "errors", result)
        locator = result.value
        bundle_root = locator.cache_path
        derived_splits = json.loads((bundle_root / "splits.json").read_text())
        derived_dividends = json.loads((bundle_root / "dividends.json").read_text())
        derived_assets = json.loads((bundle_root / "assets.json").read_text())
        derived_daily = [
            json.loads(line)
            for line in (bundle_root / "daily.jsonl").read_text().splitlines()
        ]

        expected_actions = case["expected_bundle_actions"]
        assert derived_splits == expected_actions["splits"]
        assert derived_dividends == expected_actions["dividends"]
        assert len(derived_splits) == 1
        assert len(derived_dividends) == 1
        assert derived_assets[0]["symbol"] == case["symbol"]
        assert derived_assets[0]["sid"] == 0
        assert derived_assets[1]["symbol"] == "SPY"
        assert derived_assets[1]["sid"] == 1

        expected_daily = [
            {
                "sid": 0,
                "session": item["session"],
                "open": item["raw"]["open"],
                "high": item["raw"]["high"],
                "low": item["raw"]["low"],
                "close": item["raw"]["close"],
                "volume": item["raw"]["volume"],
            }
            for item in case["sessions"]
        ]
        assert [row for row in derived_daily if row["sid"] == 0] == expected_daily
        assert all(
            set(row) == {"sid", "session", "open", "high", "low", "close", "volume"}
            for row in derived_daily
        )

        platform_actions = _platform_actions(case)
        derived_action_keys = tuple(
            {
                "session": row["effective_date"],
                "split_ratio": str(
                    _decimal(row["new_shares"]) / _decimal(row["old_shares"])
                ),
                "dividend": "0",
            }
            for row in derived_splits
        ) + tuple(
            {
                "session": row["ex_date"],
                "split_ratio": "1",
                "dividend": row["amount"],
            }
            for row in derived_dividends
        )
        assert tuple(
            sorted(platform_actions, key=lambda item: item["session"])
        ) == tuple(sorted(derived_action_keys, key=lambda item: item["session"]))

        expected_ledger = case["ledger"]
        actual_ledger = _ledger_reference(case, rows)
        assert (
            actual_ledger["actual_shares"] == expected_ledger["expected_actual_shares"]
        )
        assert actual_ledger["cash_in_lieu"] == _decimal(
            expected_ledger["expected_cash_in_lieu"]
        )
        assert actual_ledger["split_value_before"] == _decimal(
            expected_ledger["expected_split_value_before"]
        )
        assert actual_ledger["split_value_after"] == _decimal(
            expected_ledger["expected_split_value_after"]
        )
        assert actual_ledger["dividend_cash"] == _decimal(
            expected_ledger["expected_dividend_cash"]
        )
        assert actual_ledger["total_cash"] == _decimal(
            expected_ledger["expected_total_cash"]
        )
        assert isinstance(actual_ledger["actual_shares"], int)

        requested_columns = [columns for columns, _ in source.calls]
        assert requested_columns[0] == ("symbol", "session", "dividend", "split_ratio")
        assert all(
            "adjusted_open" not in columns
            and "adjusted_close" not in columns
            and "provider_adj_close" not in columns
            for columns in requested_columns
        )
