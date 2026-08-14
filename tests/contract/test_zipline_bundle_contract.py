"""Pinned Zipline bundle boundary contracts backed by local snapshot fixtures."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib.metadata import version as installed_package_version
from pathlib import Path

from quant_research_platform.domain.canonical import (
    canonical_json,
    sha256_canonical_json,
)
from quant_research_platform.domain.errors import LimitationDisclosure, Ok
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
    ADAPTER_VERSION,
    ZiplineAsset,
    ZiplineBundleAdapter,
    ZiplineDailyBar,
    ZiplineDividend,
    ZiplineSplit,
)

_PINNED_ZIPLINE_VERSION = "3.1.1"
_NOW = datetime(2024, 1, 10, 15, tzinfo=UTC)
_START = date(2024, 1, 2)
_END = date(2024, 1, 4)
_SESSIONS = (_START, date(2024, 1, 3), _END)
_SNAPSHOT_OBJECT_CHECKSUM = "a" * 64


@dataclass
class _FixtureCalendar:
    """A deterministic XNYS seam for the tiny offline snapshot."""

    name: str = "XNYS"
    version: str = "fixture-xnys"
    schedule_digest: str = "b" * 64

    def schedule_checksum(self, start: date, end: date) -> str:
        assert (start, end) == (_START, _END)
        return self.schedule_digest

    def next_session(self, session: date) -> date:
        assert session == _END
        return date(2024, 1, 5)


@dataclass
class _FixtureSource:
    rows: tuple[dict[str, object], ...]
    calls: list[tuple[tuple[str, ...], tuple[str, ...], object | None]] = field(
        default_factory=list
    )

    def scan(
        self,
        references: Sequence[ContentAddressedObjectRef],
        columns: Sequence[str],
        *,
        predicate: object | None = None,
    ) -> Iterable[Mapping[str, object]]:
        del references
        normalized_columns = tuple(columns)
        symbols = getattr(predicate, "symbols", None)
        self.calls.append((normalized_columns, tuple(symbols or ()), predicate))
        selected = [
            row
            for row in self.rows
            if symbols is None or row["symbol"] in symbols
        ]
        selected.sort(key=lambda row: (str(row["symbol"]), row["session"]))
        return tuple(
            {column: row[column] for column in normalized_columns} for row in selected
        )


@dataclass
class _WriterCall:
    assets: tuple[ZiplineAsset, ...]
    daily_rows: tuple[tuple[int, tuple[ZiplineDailyBar, ...]], ...]
    splits: tuple[ZiplineSplit, ...]
    dividends: tuple[ZiplineDividend, ...]
    start_session: date
    end_session: date
    calendar: object


@dataclass
class _FixtureWriter:
    source: _FixtureSource
    calls: list[_WriterCall] = field(default_factory=list)
    scan_calls_before_daily_rows: int | None = None

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[ZiplineAsset],
        daily_rows: Iterable[tuple[int, Iterable[ZiplineDailyBar]]],
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        self.scan_calls_before_daily_rows = len(self.source.calls)
        materialized_daily_rows = tuple(
            (sid, tuple(rows)) for sid, rows in daily_rows
        )
        self.calls.append(
            _WriterCall(
                assets=tuple(assets),
                daily_rows=materialized_daily_rows,
                splits=tuple(splits),
                dividends=tuple(dividends),
                start_session=start_session,
                end_session=end_session,
                calendar=calendar,
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "assets.json").write_bytes(
            canonical_json([asset.to_content_dict() for asset in assets])
        )
        (output_dir / "splits.json").write_bytes(
            canonical_json([split.to_content_dict() for split in splits])
        )
        (output_dir / "dividends.json").write_bytes(
            canonical_json([dividend.to_content_dict() for dividend in dividends])
        )
        with (output_dir / "daily.jsonl").open("wb") as handle:
            for _sid, rows in materialized_daily_rows:
                for row in rows:
                    handle.write(canonical_json(row.to_content_dict()))


@dataclass
class _FixtureSnapshotManager:
    manifest: SnapshotManifest
    handle: SnapshotHandle
    opened: list[str] = field(default_factory=list)
    inspected: list[str] = field(default_factory=list)

    def open_verified(self, snapshot_id: str) -> object:
        self.opened.append(snapshot_id)
        return Ok(self.handle)

    def inspect_snapshot(self, snapshot_id: str) -> object:
        self.inspected.append(snapshot_id)
        return self.manifest


def _snapshot_manifest() -> SnapshotManifest:
    reference = ContentAddressedObjectRef(
        object_kind=ObjectKind.NORMALIZED,
        checksum=_SNAPSHOT_OBJECT_CHECKSUM,
        relative_uri=(
            "objects/normalized/symbol=all/year=2024/"
            f"sha256={_SNAPSHOT_OBJECT_CHECKSUM}.parquet"
        ),
        schema_version="daily_bar_v1",
        row_count=9,
        byte_size=1,
        media_type="application/vnd.apache.parquet",
    )
    identity = SnapshotContentIdentity(
        provider="fixture",
        requested_range=DateRange(_START, _END),
        covered_range=DateRange(_START, _END),
        configured_universe=("MSFT", "aapl"),
        benchmark_symbol="SPY",
        calendar=CalendarIdentity("XNYS", "fixture-xnys", "b" * 64),
        configuration_checksum="c" * 64,
        objects=(reference,),
        validation_report_checksum="d" * 64,
        validation_summary=ValidationSummary(
            accepted_row_count=9,
            quarantined_row_count=0,
            collapsed_duplicate_count=0,
            gap_count=0,
            covered_range=DateRange(_START, _END),
        ),
        limitation_disclosure=LimitationDisclosure.current(),
    )
    return SnapshotManifest(
        content_identity=identity,
        operational_metadata=OperationalMetadata(created_at=_NOW),
    )


def _normalized_rows() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    values = {
        "AAPL": Decimal("100"),
        "MSFT": Decimal("200"),
        "SPY": Decimal("300"),
    }
    for symbol, base in values.items():
        for offset, session in enumerate(_SESSIONS):
            close = base + offset
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "raw_open": close - Decimal("1"),
                    "raw_high": close + Decimal("1"),
                    "raw_low": close - Decimal("2"),
                    "raw_close": close,
                    "raw_volume": Decimal("1000"),
                    # These values intentionally disagree with raw prices.  The
                    # bundle must never select research-adjusted ledger prices.
                    "adjusted_open": Decimal("9000") + offset,
                    "adjusted_close": Decimal("9001") + offset,
                    "dividend": (
                        Decimal("0.25")
                        if symbol == "MSFT" and offset == 1
                        else Decimal("0")
                    ),
                    "split_ratio": (
                        Decimal("2")
                        if symbol == "AAPL" and offset == 1
                        else Decimal("1")
                    ),
                }
            )
    return tuple(rows)


def test_locked_zipline_version_and_bundle_contract() -> None:
    import zipline

    assert installed_package_version("zipline-reloaded") == _PINNED_ZIPLINE_VERSION
    assert zipline.__version__ == _PINNED_ZIPLINE_VERSION


def test_bundle_projection_is_raw_lazy_daily_and_exactly_snapshot_selected(
    tmp_path: Path,
) -> None:
    manifest = _snapshot_manifest()
    handle = SnapshotHandle.from_manifest(manifest, verified_at=_NOW)
    manager = _FixtureSnapshotManager(manifest, handle)
    source = _FixtureSource(_normalized_rows())
    calendar = _FixtureCalendar()
    writer = _FixtureWriter(source)
    adapter = ZiplineBundleAdapter(
        snapshot_manager=manager,
        data_source=source,
        calendar=calendar,
        zipline_root=tmp_path / "bundles",
        writer=writer,
    )

    result = adapter.materialize(manifest.snapshot_id)

    assert isinstance(result, Ok)
    locator = result.value
    assert manager.opened == [manifest.snapshot_id]
    assert manager.inspected == [manifest.snapshot_id]
    assert locator.snapshot_id == manifest.snapshot_id
    assert locator.adapter_version == ADAPTER_VERSION
    assert locator.bundle_name == f"qrp_{manifest.snapshot_id}_{ADAPTER_VERSION}"
    assert locator.bundle_name != "latest"
    assert locator.cache_path == (
        tmp_path / "bundles" / manifest.snapshot_id / ADAPTER_VERSION
    ).resolve()
    assert len(writer.calls) == 1

    call = writer.calls[0]
    assert call.start_session == _START
    assert call.end_session == _END
    assert call.calendar is calendar
    assert [asset.symbol for asset in call.assets] == ["AAPL", "MSFT", "SPY"]
    assert [asset.sid for asset in call.assets] == [0, 1, 2]
    assert all(asset.exchange == "XNYS" for asset in call.assets)
    assert all(asset.start_date == _START for asset in call.assets)
    assert all(asset.end_date == _END for asset in call.assets)
    assert all(asset.auto_close_date == date(2024, 1, 5) for asset in call.assets)

    assert [sid for sid, _ in call.daily_rows] == [0, 1, 2]
    assert writer.scan_calls_before_daily_rows == 1
    assert len(source.calls) == 4
    assert all(len(rows) == 3 for _, rows in call.daily_rows)
    aapl_rows = call.daily_rows[0][1]
    assert aapl_rows[0].open == Decimal("99")
    assert aapl_rows[1].close == Decimal("101")
    assert set(aapl_rows[0].to_content_dict()) == {
        "sid",
        "session",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    daily_columns = [
        columns
        for columns, _, _ in source.calls
        if columns == (
            "symbol",
            "session",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_volume",
        )
    ]
    assert len(daily_columns) == 3
    assert all(columns == daily_columns[0] for columns in daily_columns)
    assert not any(
        "adjusted" in column for columns, _, _ in source.calls for column in columns
    )

    assert len(call.splits) == 1
    split = call.splits[0]
    assert split.sid == 0
    assert split.effective_date == date(2024, 1, 3)
    assert split.old_shares == Decimal("1")
    assert split.new_shares == Decimal("2")
    assert split.ratio == Decimal("0.5")

    assert len(call.dividends) == 1
    dividend = call.dividends[0]
    assert dividend.sid == 1
    assert dividend.ex_date == date(2024, 1, 3)
    assert dividend.amount == Decimal("0.25")
    assert dividend.declared_date is None
    assert dividend.record_date is None
    assert dividend.pay_date is None

    metadata = json.loads(
        (locator.cache_path / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    expected_action_checksum = sha256_canonical_json(
        {
            "splits": [split.to_content_dict()],
            "dividends": [dividend.to_content_dict()],
        }
    )
    assert metadata["action_checksum"] == expected_action_checksum
    assert metadata["raw_columns"] == ["open", "high", "low", "close", "volume"]
    assert metadata["minute_data"] is False
    assert not any(
        "minute" in path.name.lower() for path in locator.cache_path.iterdir()
    )

    repeated = adapter.materialize(manifest.snapshot_id)
    assert isinstance(repeated, Ok)
    assert repeated.value == locator
    assert len(writer.calls) == 1
    assert manager.opened == [manifest.snapshot_id, manifest.snapshot_id]
    assert manager.inspected == [manifest.snapshot_id, manifest.snapshot_id]


def test_locked_zipline_writer_and_daily_reader_keep_raw_daily_projection(
    tmp_path: Path,
) -> None:
    from zipline.data.bcolz_daily_bars import BcolzDailyBarReader

    manifest = _snapshot_manifest()
    handle = SnapshotHandle.from_manifest(manifest, verified_at=_NOW)
    adapter = ZiplineBundleAdapter(
        snapshot_manager=_FixtureSnapshotManager(manifest, handle),
        data_source=_FixtureSource(_normalized_rows()),
        calendar=_FixtureCalendar(),
        zipline_root=tmp_path / "bundles",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = adapter.materialize(manifest.snapshot_id)

    assert isinstance(result, Ok)
    locator = result.value
    daily_path = next(locator.cache_path.rglob("daily_equities.bcolz"))
    reader = BcolzDailyBarReader(daily_path)
    opens, closes, volumes = reader.load_raw_arrays(
        ["open", "close", "volume"],
        datetime(2024, 1, 2),
        datetime(2024, 1, 4),
        [0, 1, 2],
    )
    assert opens.tolist() == [
        [99.0, 199.0, 299.0],
        [100.0, 200.0, 300.0],
        [101.0, 201.0, 301.0],
    ]
    assert closes.tolist() == [
        [100.0, 200.0, 300.0],
        [101.0, 201.0, 301.0],
        [102.0, 202.0, 302.0],
    ]
    assert volumes.tolist() == [[1000, 1000, 1000]] * 3

    minute_path = next(locator.cache_path.rglob("minute_equities.bcolz"))
    assert [
        path
        for path in minute_path.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    ] == []
