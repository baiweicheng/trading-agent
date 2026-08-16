"""Focused offline tests for verified snapshot-to-Zipline projection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from quant_research_platform.domain.canonical import sha256_canonical_json
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
from quant_research_platform.domain.manifests import (
    CalendarIdentity,
    ContentAddressedObjectRef,
    ObjectKind,
    OperationalMetadata,
    SnapshotContentIdentity,
    SnapshotManifest,
    VerifiedSnapshotHandle,
)
from quant_research_platform.domain.market import DateRange, ValidationSummary
from quant_research_platform.infrastructure.zipline_bundle import (
    ZiplineBundleAdapter,
    ZiplineDividend,
    ZiplineSplit,
)


@dataclass
class FixtureCalendar:
    name: str = "XNYS"
    version: str = "fixture-xnys-v1"

    def schedule_checksum(self, start: date, end: date) -> str:
        return sha256_canonical_json(
            {"start": start.isoformat(), "end": end.isoformat()}
        )

    def next_session(self, session: date) -> date:
        return session + timedelta(days=1)


class FixtureSource:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = tuple(rows)
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def scan(
        self,
        refs: object,
        columns: Sequence[str],
        *,
        predicate: object,
    ) -> Iterable[Mapping[str, object]]:
        del refs
        symbols = getattr(predicate, "symbols", None)
        session_start = getattr(predicate, "session_start", None)
        session_end = getattr(predicate, "session_end", None)
        selected = tuple(
            row
            for row in self.rows
            if (symbols is None or row["symbol"] in symbols)
            and (session_start is None or row["session"] >= session_start)
            and (session_end is None or row["session"] <= session_end)
        )
        normalized_columns = tuple(columns)
        self.calls.append((normalized_columns, normalized_columns))
        return tuple(
            {column: row[column] for column in normalized_columns} for row in selected
        )


class FixtureWriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[object],
        daily_rows: Iterable[tuple[int, Iterable[object]]],
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        del bundle_name, bundle_timestamp, start_session, end_session, calendar
        captured_daily: list[tuple[int, tuple[object, ...]]] = []
        for sid, rows in daily_rows:
            captured_daily.append((sid, tuple(rows)))
        self.calls.append(
            {
                "assets": tuple(assets),
                "daily": tuple(captured_daily),
                "splits": tuple(splits),
                "dividends": tuple(dividends),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fixture-bundle.bin").write_bytes(
            repr(self.calls[-1]).encode("utf-8")
        )


class FixtureSnapshotManager:
    def __init__(self, manifest: SnapshotManifest) -> None:
        self._manifest = manifest
        self.handle = VerifiedSnapshotHandle.from_manifest(
            manifest,
            verified_at=datetime(2024, 1, 10, 22, tzinfo=UTC),
        )
        self.open_calls = 0
        self.inspect_calls = 0

    def open_verified(self, snapshot_id: str) -> object:
        self.open_calls += 1
        assert snapshot_id == self.handle.snapshot_id
        return Ok(self.handle)

    def inspect_snapshot(self, snapshot_id: str) -> object:
        self.inspect_calls += 1
        assert snapshot_id == self.handle.snapshot_id
        return Ok(SimpleNamespace(manifest=self.manifest))

    @property
    def manifest(self) -> SnapshotManifest:
        return self._manifest

    @manifest.setter
    def manifest(self, value: SnapshotManifest) -> None:
        self._manifest = value


def _fixture_manifest() -> SnapshotManifest:
    requested = DateRange(date(2024, 1, 2), date(2024, 1, 4))
    calendar_checksum = sha256_canonical_json(
        {"start": requested.start.isoformat(), "end": requested.end.isoformat()}
    )
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum="a" * 64,
        relative_uri=(
            "objects/normalized/symbol=AAPL/year=2024/"
            + "sha256="
            + "a" * 64
            + ".parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=9,
        byte_size=1,
        symbol="AAPL",
        session_year=2024,
        media_type="application/vnd.apache.parquet",
    )
    return SnapshotManifest(
        content_identity=SnapshotContentIdentity(
            provider="yfinance",
            requested_range=requested,
            covered_range=requested,
            configured_universe=("MSFT", "AAPL"),
            benchmark_symbol="SPY",
            calendar=CalendarIdentity("XNYS", "fixture-xnys-v1", calendar_checksum),
            configuration_checksum="b" * 64,
            objects=(reference,),
            validation_report_checksum="c" * 64,
            validation_summary=ValidationSummary(
                accepted_row_count=9,
                quarantined_row_count=0,
                collapsed_duplicate_count=0,
                gap_count=0,
                covered_range=requested,
            ),
            limitation_disclosure=LimitationDisclosure.current(),
        ),
        operational_metadata=OperationalMetadata(
            created_at=datetime(2024, 1, 10, 22, tzinfo=UTC)
        ),
    )


def _rows() -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    bases = {"AAPL": 100, "MSFT": 200, "SPY": 300}
    for symbol, base in bases.items():
        for offset, session in enumerate(
            (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
        ):
            close = Decimal(base + offset)
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "raw_open": close - 1,
                    "raw_high": close + 1,
                    "raw_low": close - 2,
                    "raw_close": close,
                    "raw_volume": Decimal("1000"),
                    "dividend": Decimal("0.5")
                    if symbol == "AAPL" and offset == 2
                    else Decimal("0"),
                    "split_ratio": Decimal("2")
                    if symbol == "AAPL" and offset == 1
                    else Decimal("1"),
                }
            )
    return tuple(rows)


def test_materialize_is_deterministic_raw_only_and_rebuilds_corrupt_cache(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest()
    manager = FixtureSnapshotManager(manifest)
    source = FixtureSource(_rows())
    writer = FixtureWriter()
    calendar = FixtureCalendar()
    adapter = ZiplineBundleAdapter(
        manager,
        source,
        calendar,
        tmp_path / "zipline-bundles",
        writer=writer,
    )

    first = adapter.materialize(manifest.snapshot_id)
    assert isinstance(first, Ok)
    first_locator = first.value
    first_call = writer.calls[0]

    assert [asset.symbol for asset in first_call["assets"]] == [
        "AAPL",
        "MSFT",
        "SPY",
    ]
    assert [asset.sid for asset in first_call["assets"]] == [0, 1, 2]
    assert all(
        "adjusted" not in field
        for _, rows in first_call["daily"]
        for row in rows
        for field in row.to_content_dict()
    )
    daily_by_sid = dict(first_call["daily"])
    assert (
        daily_by_sid[0][0].open,
        daily_by_sid[0][0].high,
        daily_by_sid[0][0].low,
        daily_by_sid[0][0].close,
        daily_by_sid[0][0].volume,
    ) == (
        Decimal("99"),
        Decimal("101"),
        Decimal("98"),
        Decimal("100"),
        Decimal("1000"),
    )
    assert source.calls[0][0] == (
        "symbol",
        "session",
        "dividend",
        "split_ratio",
    )
    assert all(
        columns
        == (
            "symbol",
            "session",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_volume",
        )
        for columns, _ in source.calls[1:]
    )
    assert [dividend.amount for dividend in first_call["dividends"]] == [Decimal("0.5")]
    assert first_locator.bundle_name != "latest"
    assert first_locator.cache_path.joinpath("bundle_manifest.json").is_file()

    second = adapter.materialize(manifest.snapshot_id)
    assert isinstance(second, Ok)
    assert second.value == first_locator
    assert len(writer.calls) == 1
    assert manager.open_calls == 2
    assert manager.inspect_calls == 2

    corruptible = first_locator.cache_path / "fixture-bundle.bin"
    corruptible.write_bytes(b"corrupt derived bytes")
    rebuilt = adapter.materialize(manifest.snapshot_id)
    assert isinstance(rebuilt, Ok)
    assert rebuilt.value == first_locator
    assert len(writer.calls) == 2
    assert rebuilt.value.bundle_checksum == first_locator.bundle_checksum
