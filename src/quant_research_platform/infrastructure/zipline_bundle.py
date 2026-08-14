"""Verified snapshot-to-Zipline bundle materialization.

The platform snapshot is the source of truth.  This module is the only place
where a verified normalized Parquet snapshot is projected into Zipline's
bundle writers.  It intentionally keeps the Zipline import lazy: unit and
property tests can exercise the projection with a local writer seam, while a
normal installation uses the pinned Zipline Reloaded writers.

Only raw, action-effective OHLCV values are sent to the ledger.  The platform's
causal research-adjusted columns are never selected by the bundle scan.  The
canonical split/dividend stream is sent separately, once, so Zipline can apply
actual-share changes and dividend cash without double-adjusting prices.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ..domain.canonical import (
    canonical_json,
    canonical_timestamp,
    sha256_bytes,
    sha256_canonical_json,
)
from ..domain.errors import ActionableError, Err, ErrorCategory, Ok, Result
from ..domain.manifests import (
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotContentIdentity,
    SnapshotHandle,
    SnapshotManifest,
)
from ..domain.market import normalize_symbol
from .parquet_store import ScanPredicate

ADAPTER_VERSION: Final = "zipline_bundle_v1"
BUNDLE_MANIFEST_VERSION: Final = "zipline_bundle_manifest_v1"
DEFAULT_ZIPLINE_ROOT: Final = Path("data/zipline-bundles")
_ZIPLINE_SNAPSHOT_PATTERN: Final = re.compile(r"^snap_[0-9a-f]{64}$")
_SAFE_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9._-]+$")
_DAILY_COLUMNS: Final = (
    "symbol",
    "session",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
)
_ACTION_COLUMNS: Final = ("symbol", "session", "dividend", "split_ratio")
_BUNDLE_MANIFEST_NAME: Final = "bundle_manifest.json"


class ZiplineBundleError(RuntimeError):
    """Base class for bundle projection, cache, and writer failures."""


class ZiplineBundleIntegrityError(ZiplineBundleError):
    """Raised when a derived bundle or its source cannot be verified."""


class ZiplineBundleWriterError(ZiplineBundleError):
    """Raised when the pinned third-party writer cannot materialize a bundle."""


@dataclass(frozen=True, slots=True)
class ZiplineAsset:
    """The asset-lifetime projection supplied to ``AssetDBWriter``."""

    sid: int
    symbol: str
    exchange: str
    start_date: date
    end_date: date
    auto_close_date: date

    def __post_init__(self) -> None:
        if isinstance(self.sid, bool) or not isinstance(self.sid, int) or self.sid < 0:
            raise ValueError("sid must be a non-negative integer")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.exchange, str) or not self.exchange.strip():
            raise ValueError("exchange must be a non-empty string")
        object.__setattr__(self, "exchange", " ".join(self.exchange.split()))
        for field_name in ("start_date", "end_date", "auto_close_date"):
            value = getattr(self, field_name)
            if isinstance(value, datetime) or not isinstance(value, date):
                raise TypeError(f"{field_name} must be a calendar date")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.auto_close_date <= self.end_date:
            raise ValueError("auto_close_date must be after end_date")

    def to_content_dict(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "auto_close_date": self.auto_close_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ZiplineDailyBar:
    """One raw daily OHLCV row; no research-adjusted field is representable."""

    sid: int
    session: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        if isinstance(self.sid, bool) or not isinstance(self.sid, int) or self.sid < 0:
            raise ValueError("sid must be a non-negative integer")
        if isinstance(self.session, datetime) or not isinstance(self.session, date):
            raise TypeError("session must be a calendar date")
        for field_name in ("open", "high", "low", "close", "volume"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{field_name} must be a finite Decimal")
            if field_name != "volume" and value <= 0:
                raise ValueError(f"{field_name} must be positive")
            if field_name == "volume" and value < 0:
                raise ValueError("volume must be non-negative")

    def to_content_dict(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "session": self.session.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class ZiplineSplit:
    """A canonical split converted from provider ``new shares / old shares``."""

    sid: int
    effective_date: date
    old_shares: Decimal
    new_shares: Decimal
    ratio: Decimal

    def __post_init__(self) -> None:
        if isinstance(self.sid, bool) or not isinstance(self.sid, int) or self.sid < 0:
            raise ValueError("sid must be a non-negative integer")
        if isinstance(self.effective_date, datetime) or not isinstance(
            self.effective_date, date
        ):
            raise TypeError("effective_date must be a calendar date")
        for field_name in ("old_shares", "new_shares", "ratio"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{field_name} must be a positive finite Decimal")
        expected = self.old_shares / self.new_shares
        if self.ratio != expected:
            raise ValueError("Zipline split ratio must be old_shares / new_shares")

    def to_content_dict(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "effective_date": self.effective_date.isoformat(),
            "old_shares": self.old_shares,
            "new_shares": self.new_shares,
            "ratio": self.ratio,
        }


@dataclass(frozen=True, slots=True)
class ZiplineDividend:
    """A canonical ex-date dividend with unavailable dates left null."""

    sid: int
    ex_date: date
    amount: Decimal
    declared_date: date | None = None
    record_date: date | None = None
    pay_date: date | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sid, bool) or not isinstance(self.sid, int) or self.sid < 0:
            raise ValueError("sid must be a non-negative integer")
        if isinstance(self.ex_date, datetime) or not isinstance(self.ex_date, date):
            raise TypeError("ex_date must be a calendar date")
        if not isinstance(self.amount, Decimal) or not self.amount.is_finite():
            raise ValueError("amount must be a finite Decimal")
        for field_name in ("declared_date", "record_date", "pay_date"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, datetime) or not isinstance(value, date)
            ):
                raise TypeError(f"{field_name} must be a calendar date or None")

    def to_content_dict(self) -> dict[str, object]:
        return {
            "sid": self.sid,
            "ex_date": self.ex_date.isoformat(),
            "amount": self.amount,
            "declared_date": (
                self.declared_date.isoformat() if self.declared_date else None
            ),
            "record_date": (
                self.record_date.isoformat() if self.record_date else None
            ),
            "pay_date": self.pay_date.isoformat() if self.pay_date else None,
        }


@dataclass(frozen=True, slots=True)
class ZiplineBundleLocator:
    """An exact, immutable Zipline bundle selection.

    ``zipline_root`` points at the snapshot/adapter-specific root containing
    the bundle data.  It is intentionally not a mutable bundle name or a
    ``latest`` alias.
    """

    bundle_name: str
    bundle_timestamp: datetime
    zipline_root: Path
    snapshot_id: str
    adapter_version: str
    bundle_checksum: str

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_name, str) or not self.bundle_name.strip():
            raise ValueError("bundle_name must be non-empty")
        object.__setattr__(self, "bundle_name", " ".join(self.bundle_name.split()))
        if not isinstance(self.bundle_timestamp, datetime):
            raise TypeError("bundle_timestamp must be a datetime")
        if (
            self.bundle_timestamp.tzinfo is None
            or self.bundle_timestamp.utcoffset() != timedelta(0)
        ):
            raise ValueError("bundle_timestamp must be UTC")
        object.__setattr__(
            self, "bundle_timestamp", self.bundle_timestamp.astimezone(UTC)
        )
        root = Path(self.zipline_root).expanduser().resolve(strict=False)
        object.__setattr__(self, "zipline_root", root)
        if _ZIPLINE_SNAPSHOT_PATTERN.fullmatch(self.snapshot_id) is None:
            raise ValueError("snapshot_id must be a content-derived Snapshot_ID")
        if not isinstance(self.adapter_version, str):
            raise ValueError("adapter_version must be a safe non-empty name")
        if _SAFE_NAME_PATTERN.fullmatch(self.adapter_version) is None:
            raise ValueError("adapter_version must be a safe non-empty name")
        if not isinstance(self.bundle_checksum, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.bundle_checksum
        ) is None:
            raise ValueError("bundle_checksum must be a lowercase SHA-256 digest")

    @property
    def cache_path(self) -> Path:
        """Return the exact cache root used by this locator."""

        return self.zipline_root


class SnapshotVerifier(Protocol):
    """The small verification/inspection seam required by the adapter."""

    def open_verified(self, snapshot_id: str) -> object:
        """Verify and return a snapshot handle or a typed Result."""

    def inspect_snapshot(self, snapshot_id: str) -> object:
        """Return verified manifest details or a typed Result."""


class SnapshotObjectSource(Protocol):
    """A source of already-published normalized Parquet objects."""

    def read_object(self, relative_uri: str) -> bytes:
        """Read one published object by its manifest URI."""


BundleDailyRows: TypeAlias = Iterable[tuple[int, Iterable[ZiplineDailyBar]]]


class BundleWriter(Protocol):
    """Writer seam isolating the pinned third-party Zipline API."""

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[ZiplineAsset],
        daily_rows: BundleDailyRows,
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        """Write one complete derived bundle from platform-owned projections."""


@dataclass(frozen=True, slots=True)
class _SnapshotContext:
    manifest: SnapshotManifest
    handle: SnapshotHandle


@dataclass(frozen=True, slots=True)
class _BundleBounds:
    first: date
    last: date


class _ReplayableDailyRows:
    """Replay one lazy daily-row stream without issuing a second source scan."""

    def __init__(self, source: BundleDailyRows) -> None:
        self._source = iter(source)
        self._cache: list[tuple[int, tuple[ZiplineDailyBar, ...]]] = []
        self._exhausted = False

    def __iter__(self) -> Iterator[tuple[int, Iterable[ZiplineDailyBar]]]:
        index = 0
        while True:
            if index < len(self._cache):
                yield self._cache[index]
                index += 1
                continue
            if self._exhausted:
                return
            try:
                sid, rows = next(self._source)
            except StopIteration:
                self._exhausted = True
                return
            cached = (sid, tuple(rows))
            self._cache.append(cached)
            index += 1
            yield cached


class _BundleFailure(Exception):
    """Internal exception carrying a safe actionable error."""

    def __init__(
        self,
        message: str,
        corrective_action: str,
        *,
        category: ErrorCategory = ErrorCategory.INTEGRITY_CHECKSUM,
        field_path: str | None = None,
        checksum: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.corrective_action = corrective_action
        self.category = category
        self.field_path = field_path
        self.checksum = checksum


class _PortableBundleWriter:
    """Deterministic local writer used when Zipline is unavailable.

    The production path selects ``_ZiplineWriter`` first.  This writer keeps
    the projection and cache contract testable in an offline environment and
    produces a faithful, raw-only derived representation rather than silently
    importing an alternate data source.
    """

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[ZiplineAsset],
        daily_rows: BundleDailyRows,
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        del bundle_name, bundle_timestamp, start_session, end_session, calendar
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

        daily_path = output_dir / "daily.jsonl"
        with daily_path.open("wb") as handle:
            seen_sids: set[int] = set()
            for sid, rows in daily_rows:
                if sid in seen_sids:
                    raise ZiplineBundleWriterError("daily rows repeated one SID")
                seen_sids.add(sid)
                previous_session: date | None = None
                for row in rows:
                    if row.sid != sid:
                        raise ZiplineBundleWriterError(
                            "daily row SID does not match its writer group"
                        )
                    if previous_session is not None and row.session <= previous_session:
                        raise ZiplineBundleWriterError(
                            "daily rows for one SID must be strictly increasing"
                        )
                    previous_session = row.session
                    handle.write(canonical_json(row.to_content_dict()))
        # An empty minute collection is intentional: this adapter is daily-only.


class _ZiplineWriter:
    """Use Zipline Reloaded's registered-bundle writer boundary when available."""

    def __init__(self, fallback: BundleWriter | None = None) -> None:
        self._fallback = fallback or _PortableBundleWriter()

    def write(
        self,
        *,
        output_dir: Path,
        bundle_name: str,
        bundle_timestamp: datetime,
        assets: Sequence[ZiplineAsset],
        daily_rows: BundleDailyRows,
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        start_session: date,
        end_session: date,
        calendar: object,
    ) -> None:
        try:
            import pandas as pd  # type: ignore[import-untyped]
            from zipline.data.bundles import (  # type: ignore[import-untyped]
                ingest,
                register,
                unregister,
            )
        except ModuleNotFoundError:
            self._fallback.write(
                output_dir=output_dir,
                bundle_name=bundle_name,
                bundle_timestamp=bundle_timestamp,
                assets=assets,
                daily_rows=daily_rows,
                splits=splits,
                dividends=dividends,
                start_session=start_session,
                end_session=end_session,
                calendar=calendar,
            )
            return
        except Exception as error:
            raise ZiplineBundleWriterError(
                "The pinned Zipline writer could not be imported."
            ) from error

        output_dir.mkdir(parents=True, exist_ok=True)
        start = pd.Timestamp(start_session)
        end = pd.Timestamp(end_session)
        timestamp = pd.Timestamp(bundle_timestamp).tz_convert("UTC")

        asset_frame = pd.DataFrame(
            [
                {
                    "sid": asset.sid,
                    "symbol": asset.symbol,
                    "asset_name": asset.symbol,
                    "start_date": pd.Timestamp(asset.start_date, tz="UTC"),
                    "end_date": pd.Timestamp(asset.end_date, tz="UTC"),
                    "first_traded": pd.Timestamp(asset.start_date, tz="UTC"),
                    "auto_close_date": pd.Timestamp(
                        asset.auto_close_date, tz="UTC"
                    ),
                    "exchange": asset.exchange,
                }
                for asset in assets
            ]
        ).set_index("sid")

        split_frame = pd.DataFrame(
            [
                {
                    "sid": split.sid,
                    "effective_date": pd.Timestamp(split.effective_date, tz="UTC"),
                    "ratio": float(split.ratio),
                }
                for split in splits
            ],
            columns=("sid", "effective_date", "ratio"),
        )
        dividend_frame = pd.DataFrame(
            [
                {
                    "sid": dividend.sid,
                    "ex_date": pd.Timestamp(dividend.ex_date, tz="UTC"),
                    "amount": float(dividend.amount),
                    "declared_date": (
                        pd.Timestamp(dividend.declared_date, tz="UTC")
                        if dividend.declared_date
                        else pd.NaT
                    ),
                    "record_date": (
                        pd.Timestamp(dividend.record_date, tz="UTC")
                        if dividend.record_date
                        else pd.NaT
                    ),
                    "pay_date": (
                        pd.Timestamp(dividend.pay_date, tz="UTC")
                        if dividend.pay_date
                        else pd.NaT
                    ),
                }
                for dividend in dividends
            ],
            columns=(
                "sid",
                "ex_date",
                "amount",
                "declared_date",
                "record_date",
                "pay_date",
            ),
        )

        def ingest_function(
            environ: Mapping[str, object],
            asset_db_writer: object,
            minute_bar_writer: object,
            daily_bar_writer: object,
            adjustment_writer: object,
            zipline_calendar: object,
            zipline_start: object,
            zipline_end: object,
            cache: object,
            show_progress: bool,
            generated_output_dir: str,
        ) -> None:
            del (
                environ,
                minute_bar_writer,
                zipline_calendar,
                zipline_start,
                zipline_end,
                cache,
                generated_output_dir,
            )
            if asset_db_writer is None or daily_bar_writer is None:
                raise ZiplineBundleWriterError("Zipline did not create daily writers")
            write_assets = getattr(asset_db_writer, "write", None)
            write_daily = getattr(daily_bar_writer, "write", None)
            write_adjustments = getattr(adjustment_writer, "write", None)
            if not callable(write_assets) or not callable(write_daily) or not callable(
                write_adjustments
            ):
                raise ZiplineBundleWriterError(
                    "The pinned Zipline writer extension seam is unavailable"
                )
            write_assets(equities=asset_frame)

            def frames() -> Iterator[tuple[int, Any]]:
                for sid, rows in daily_rows:
                    materialized = list(rows)
                    index = pd.DatetimeIndex(
                        [pd.Timestamp(row.session, tz="UTC") for row in materialized]
                    )
                    frame = pd.DataFrame(
                        {
                            "open": [float(row.open) for row in materialized],
                            "high": [float(row.high) for row in materialized],
                            "low": [float(row.low) for row in materialized],
                            "close": [float(row.close) for row in materialized],
                            "volume": [float(row.volume) for row in materialized],
                        },
                        index=index,
                    )
                    yield sid, frame

            write_daily(frames(), show_progress=show_progress)
            write_adjustments(
                splits=split_frame if not split_frame.empty else None,
                dividends=dividend_frame if not dividend_frame.empty else None,
            )

        environment = {"ZIPLINE_ROOT": str(output_dir.resolve())}
        registered = False
        try:
            register(
                bundle_name,
                ingest_function,
                calendar_name="XNYS",
                start_session=start,
                end_session=end,
            )
            registered = True
            ingest(
                bundle_name,
                environ=environment,
                timestamp=timestamp,
                show_progress=False,
            )
        except Exception as error:
            if isinstance(error, ZiplineBundleError):
                raise
            raise ZiplineBundleWriterError(
                "The pinned Zipline writer failed to materialize the bundle."
            ) from error
        finally:
            if registered:
                with suppress(Exception):
                    unregister(bundle_name)


class ZiplineBundleAdapter:
    """Materialize one exact, verified Zipline bundle from a snapshot.

    Parameters are intentionally ports rather than global Zipline configuration:

    ``snapshot_manager`` verifies/open-inspects the snapshot; ``data_source``
    either exposes ``ParquetStore.scan`` or ``read_object``; ``calendar`` is the
    pinned XNYS date adapter; and ``writer`` isolates the version-pinned
    third-party writer.  A normal composition root can omit ``writer`` and the
    adapter uses Zipline Reloaded lazily, with a deterministic offline writer
    fallback when the optional runtime is not installed.
    """

    def __init__(
        self,
        snapshot_manager: SnapshotVerifier | None = None,
        data_source: object | None = None,
        calendar: object | None = None,
        zipline_root: Path | str = DEFAULT_ZIPLINE_ROOT,
        *,
        snapshot_store: object | None = None,
        parquet_store: object | None = None,
        writer: BundleWriter | None = None,
        adapter_version: str = ADAPTER_VERSION,
        cache_root: Path | str | None = None,
    ) -> None:
        if not isinstance(adapter_version, str):
            raise ValueError("adapter_version must be a safe non-empty name")
        if _SAFE_NAME_PATTERN.fullmatch(adapter_version) is None:
            raise ValueError("adapter_version must be a safe non-empty name")
        self._adapter_version = adapter_version
        self._snapshot_manager = snapshot_manager
        self._data_source = data_source or parquet_store or snapshot_store
        self._snapshot_store = snapshot_store or data_source
        self._calendar = calendar
        self._zipline_root = Path(zipline_root).expanduser().resolve(strict=False)
        self._cache_root = (
            Path(cache_root).expanduser().resolve(strict=False)
            if cache_root
            else self._zipline_root
        )
        self._writer = writer or _ZiplineWriter()

    @property
    def adapter_version(self) -> str:
        return self._adapter_version

    @property
    def zipline_root(self) -> Path:
        return self._zipline_root

    def materialize(
        self, snapshot: SnapshotHandle | str | object
    ) -> Result[ZiplineBundleLocator]:
        """Verify, cache-check, and materialize one exact snapshot bundle."""

        try:
            context = self._verified_context(snapshot)
            identity = context.manifest.content_identity
            calendar = self._calendar_or_error()
            self._verify_calendar_identity(identity, calendar)
            bundle_name = self._bundle_name(
                context.handle.snapshot_id, self._adapter_version
            )
            cache_dir = self._cache_directory(context.handle.snapshot_id)
            cached = self._read_verified_cache(
                cache_dir,
                context=context,
                bundle_name=bundle_name,
            )
            if cached is not None:
                return Ok(cached)

            refs = self._normalized_references(identity.objects)
            symbols = self._bundle_symbols(identity)
            bounds, splits, dividends = self._project_actions_and_bounds(
                refs,
                symbols=symbols,
                requested_start=identity.requested_range.start,
                requested_end=identity.requested_range.end,
            )
            assets = tuple(
                ZiplineAsset(
                    sid=index,
                    symbol=symbol,
                    exchange=identity.calendar.name,
                    start_date=bounds[symbol].first,
                    end_date=bounds[symbol].last,
                    auto_close_date=self._next_session(
                        calendar, bounds[symbol].last
                    ),
                )
                for index, symbol in enumerate(symbols)
            )
            asset_checksum = sha256_canonical_json(
                [asset.to_content_dict() for asset in assets]
            )
            policy_version = identity.schema_versions.corporate_action_policy_version
            policy_checksum = sha256_canonical_json(
                {"version": policy_version}
            )
            action_checksum = sha256_canonical_json(
                {
                    "splits": [split.to_content_dict() for split in splits],
                    "dividends": [
                        dividend.to_content_dict() for dividend in dividends
                    ],
                }
            )
            bundle_timestamp = self._deterministic_timestamp(
                context.handle.snapshot_id
            )
            self._materialize_cache(
                cache_dir,
                context=context,
                bundle_name=bundle_name,
                bundle_timestamp=bundle_timestamp,
                calendar=calendar,
                refs=refs,
                symbols=symbols,
                assets=assets,
                splits=splits,
                dividends=dividends,
                asset_checksum=asset_checksum,
                policy_version=policy_version,
                policy_checksum=policy_checksum,
                action_checksum=action_checksum,
                requested_start=identity.requested_range.start,
                requested_end=identity.requested_range.end,
            )
            verified = self._read_verified_cache(
                cache_dir,
                context=context,
                bundle_name=bundle_name,
            )
            if verified is None:  # pragma: no cover - defensive postcondition
                raise ZiplineBundleIntegrityError(
                    "newly materialized Zipline bundle failed verification"
                )
            return Ok(verified)
        except _BundleFailure as failure:
            return Err(
                (
                    ActionableError(
                        operation="zipline.bundle.materialize",
                        category=failure.category,
                        message=failure.message,
                        corrective_action=failure.corrective_action,
                        field_path=failure.field_path,
                        checksum=failure.checksum,
                    ),
                )
            )
        except ZiplineBundleError as error:
            return Err(
                (
                    ActionableError(
                        operation="zipline.bundle.materialize",
                        category=ErrorCategory.STORAGE_IO,
                        message=str(error),
                        corrective_action=(
                            "Repair the derived bundle writer or retry materialization "
                            "from the verified snapshot."
                        ),
                        field_path="bundle",
                    ),
                )
            )
        except Exception as error:
            return Err(
                (
                    ActionableError.from_unexpected_exception(
                        "zipline.bundle.materialize", error
                    ),
                )
            )

    materialize_bundle = materialize

    def _verified_context(
        self, snapshot: SnapshotHandle | str | object
    ) -> _SnapshotContext:
        requested_handle = snapshot if hasattr(snapshot, "snapshot_id") else None
        snapshot_id = getattr(snapshot, "snapshot_id", snapshot)
        if (
            not isinstance(snapshot_id, str)
            or _ZIPLINE_SNAPSHOT_PATTERN.fullmatch(snapshot_id) is None
        ):
            raise _BundleFailure(
                "The bundle request did not contain a valid Snapshot_ID.",
                (
                    "Select a checksum-verified published snapshot before "
                    "materializing a bundle."
                ),
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="snapshot_id",
            )
        manager = self._snapshot_manager
        if manager is None:
            raise _BundleFailure(
                "No snapshot verification service is configured.",
                (
                    "Inject SnapshotManager and retry after opening the "
                    "published snapshot."
                ),
                category=ErrorCategory.STORAGE_IO,
                field_path="snapshot_manager",
            )
        opener = getattr(manager, "open_verified", None)
        if not callable(opener):
            raise _BundleFailure(
                "The configured snapshot service cannot verify snapshots.",
                "Configure a SnapshotManager with open_verified support.",
                category=ErrorCategory.STORAGE_IO,
                field_path="snapshot_manager.open_verified",
            )
        opened = self._unwrap_result(opener(snapshot_id), "snapshot.open")
        if not hasattr(opened, "snapshot_id"):
            raise _BundleFailure(
                "Snapshot verification returned no immutable handle.",
                "Reconcile the snapshot publication and retry.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="snapshot.handle",
            )
        handle = cast(SnapshotHandle, opened)
        if requested_handle is not None and (
            handle.snapshot_id != getattr(requested_handle, "snapshot_id", None)
            or handle.content_identity_checksum
            != getattr(requested_handle, "content_identity_checksum", None)
            or handle.manifest_checksum
            != getattr(requested_handle, "manifest_checksum", None)
        ):
            raise _BundleFailure(
                "The requested snapshot handle changed during verification.",
                "Re-resolve the snapshot and retry with one verified handle.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="snapshot.handle",
            )

        manifest: object | None = None
        inspector = getattr(manager, "inspect_snapshot", None)
        if callable(inspector):
            inspected = self._unwrap_result(inspector(snapshot_id), "snapshot.inspect")
            manifest = getattr(inspected, "manifest", inspected)
        if manifest is None:
            for attribute in ("manifest", "snapshot_manifest"):
                candidate = getattr(snapshot, attribute, None)
                if candidate is not None:
                    manifest = candidate
                    break
        if manifest is None:
            for attribute in ("manifest_for", "get_manifest"):
                resolver = getattr(manager, attribute, None)
                if callable(resolver):
                    manifest = self._unwrap_result(
                        resolver(snapshot_id), "snapshot.manifest"
                    )
                    break
        if not isinstance(manifest, SnapshotManifest):
            raise _BundleFailure(
                "Snapshot verification did not expose its immutable manifest.",
                (
                    "Configure snapshot inspection or provide a "
                    "manifest-bearing snapshot detail."
                ),
                category=ErrorCategory.STORAGE_IO,
                field_path="snapshot.manifest",
            )
        if (
            manifest.snapshot_id != handle.snapshot_id
            or manifest.content_identity_checksum
            != handle.content_identity_checksum
            or manifest.manifest_checksum != handle.manifest_checksum
        ):
            raise _BundleFailure(
                "The verified snapshot manifest does not match its handle.",
                (
                    "Reconcile the snapshot publication and retry with one "
                    "verified snapshot."
                ),
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="snapshot.manifest",
                checksum=handle.content_identity_checksum,
            )
        return _SnapshotContext(manifest=manifest, handle=handle)

    @staticmethod
    def _unwrap_result(value: object, operation: str) -> object:
        if isinstance(value, Ok):
            return value.value
        if isinstance(value, Err):
            first = value.errors[0]
            raise _BundleFailure(
                first.message,
                first.corrective_action,
                category=first.category,
                field_path=first.field_path,
                checksum=first.checksum,
            )
        if value is None:
            raise _BundleFailure(
                f"{operation} returned no result.",
                "Repair the snapshot service and retry.",
                category=ErrorCategory.STORAGE_IO,
                field_path=operation,
            )
        return value

    def _calendar_or_error(self) -> object:
        if self._calendar is not None:
            return self._calendar
        try:
            from .xnys_calendar import XNYSCalendar

            self._calendar = XNYSCalendar()
            return self._calendar
        except Exception as error:
            raise _BundleFailure(
                "The pinned XNYS calendar is unavailable.",
                (
                    "Install the locked exchange-calendars dependency or inject "
                    "the verified calendar adapter."
                ),
                category=ErrorCategory.STORAGE_IO,
                field_path="calendar",
            ) from error

    @staticmethod
    def _verify_calendar_identity(
        identity: SnapshotContentIdentity, calendar: object
    ) -> None:
        expected = identity.calendar
        actual_name = getattr(calendar, "name", None)
        actual_version = getattr(calendar, "version", None)
        if actual_name != expected.name or actual_version != expected.version:
            raise _BundleFailure(
                "The bundle calendar does not match the snapshot calendar identity.",
                "Use the pinned XNYS calendar recorded by the snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="calendar",
                checksum=expected.schedule_checksum,
            )
        checksum_fn = getattr(calendar, "schedule_checksum", None)
        requested = identity.requested_range
        if callable(checksum_fn):
            try:
                actual_checksum = checksum_fn(requested.start, requested.end)
            except Exception as error:
                raise _BundleFailure(
                    "The bundle calendar schedule could not be verified.",
                    (
                        "Repair or inject the pinned XNYS calendar before "
                        "materializing the bundle."
                    ),
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    field_path="calendar.schedule_checksum",
                    checksum=expected.schedule_checksum,
                ) from error
            if actual_checksum != expected.schedule_checksum:
                raise _BundleFailure(
                    "The bundle calendar schedule differs from the snapshot identity.",
                    "Use the snapshot's exact calendar version and schedule.",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    field_path="calendar.schedule_checksum",
                    checksum=expected.schedule_checksum,
                )

    @staticmethod
    def _bundle_symbols(identity: SnapshotContentIdentity) -> tuple[str, ...]:
        universe = tuple(identity.configured_universe)
        benchmark = normalize_symbol(identity.benchmark_symbol)
        return tuple(sorted(set((*universe, benchmark))))

    @staticmethod
    def _normalized_references(
        references: Sequence[ContentAddressedObjectRef],
    ) -> tuple[ContentAddressedObjectRef, ...]:
        selected = tuple(
            reference
            for reference in references
            if reference.object_kind is ObjectKind.NORMALIZED
            and reference.schema_version == "daily_bar_v1"
        )
        if not selected:
            raise _BundleFailure(
                "The verified snapshot has no daily normalized Parquet objects.",
                "Publish a snapshot containing accepted daily bars before backtesting.",
                category=ErrorCategory.SNAPSHOT_NOT_READY,
                field_path="snapshot.objects",
            )
        return tuple(sorted(selected, key=ContentAddressedObjectRef.sort_key))

    @staticmethod
    def _bundle_name(snapshot_id: str, adapter_version: str | None = None) -> str:
        version = adapter_version or ADAPTER_VERSION
        return f"qrp_{snapshot_id}_{version}"

    def _cache_directory(self, snapshot_id: str) -> Path:
        return self._cache_root / snapshot_id / self._adapter_version

    @staticmethod
    def _deterministic_timestamp(snapshot_id: str) -> datetime:
        digest = bytes.fromhex(snapshot_id.removeprefix("snap_"))
        seconds = int.from_bytes(digest[:8], "big") % (60 * 60 * 24 * 365 * 100)
        return datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)

    @staticmethod
    def _next_session(calendar: object, session: date) -> date:
        next_fn = getattr(calendar, "next_session", None)
        if not callable(next_fn):
            raise _BundleFailure(
                "The XNYS calendar cannot determine asset auto-close dates.",
                "Inject a calendar adapter that implements next_session.",
                category=ErrorCategory.STORAGE_IO,
                field_path="calendar.next_session",
            )
        try:
            result = next_fn(session)
        except Exception as error:
            raise _BundleFailure(
                "The XNYS calendar could not determine an asset auto-close date.",
                "Repair the pinned calendar or verify the snapshot coverage.",
                category=ErrorCategory.STORAGE_IO,
                field_path="calendar.next_session",
            ) from error
        if (
            isinstance(result, datetime)
            or not isinstance(result, date)
            or result <= session
        ):
            raise _BundleFailure(
                "The calendar returned an invalid asset auto-close date.",
                "Use the pinned XNYS calendar implementation.",
                category=ErrorCategory.STORAGE_IO,
                field_path="calendar.next_session",
            )
        return cast(date, result)

    def _project_actions_and_bounds(
        self,
        refs: Sequence[ContentAddressedObjectRef],
        *,
        symbols: Sequence[str],
        requested_start: date,
        requested_end: date,
    ) -> tuple[
        dict[str, _BundleBounds],
        tuple[ZiplineSplit, ...],
        tuple[ZiplineDividend, ...],
    ]:
        sid_by_symbol = {symbol: sid for sid, symbol in enumerate(symbols)}
        bounds: dict[str, _BundleBounds] = {}
        splits_by_key: dict[tuple[int, date], ZiplineSplit] = {}
        dividends_by_key: dict[tuple[int, date], ZiplineDividend] = {}
        for row in self._scan_rows(
            refs,
            columns=_ACTION_COLUMNS,
            symbols=tuple(symbols),
            session_start=requested_start,
            session_end=requested_end,
        ):
            symbol = self._symbol_value(row)
            if symbol not in sid_by_symbol:
                continue
            session = self._as_date(self._row_value(row, "session"), "session")
            current = bounds.get(symbol)
            if current is None:
                bounds[symbol] = _BundleBounds(session, session)
            else:
                bounds[symbol] = _BundleBounds(
                    min(current.first, session), max(current.last, session)
                )
            split_ratio = self._decimal_value(
                self._row_value(row, "split_ratio"), "split_ratio"
            )
            dividend = self._decimal_value(
                self._row_value(row, "dividend"), "dividend"
            )
            sid = sid_by_symbol[symbol]
            if split_ratio != Decimal("1"):
                if split_ratio <= 0:
                    raise _BundleFailure(
                        "A canonical split ratio is not positive.",
                        (
                            "Quarantine the invalid action and publish a "
                            "corrected snapshot."
                        ),
                        category=ErrorCategory.NORMALIZATION_POLICY,
                        field_path="daily_bar.split_ratio",
                        checksum=self._checksum_for_row(row),
                    )
                split_action = ZiplineSplit(
                    sid=sid,
                    effective_date=session,
                    old_shares=Decimal("1"),
                    new_shares=split_ratio,
                    ratio=Decimal("1") / split_ratio,
                )
                key = (sid, session)
                prior = splits_by_key.get(key)
                if prior is not None and prior != split_action:
                    raise _BundleFailure(
                        "Conflicting canonical split actions were found for "
                        "one SID/session.",
                        (
                            "Validate duplicate normalized rows and publish a "
                            "new snapshot."
                        ),
                        category=ErrorCategory.VALIDATION_DUPLICATE_CONFLICT,
                        field_path="daily_bar.split_ratio",
                    )
                splits_by_key[key] = split_action
            if dividend != Decimal("0"):
                if not dividend.is_finite():
                    raise _BundleFailure(
                        "A canonical dividend amount is not finite.",
                        (
                            "Quarantine the invalid action and publish a "
                            "corrected snapshot."
                        ),
                        category=ErrorCategory.NORMALIZATION_POLICY,
                        field_path="daily_bar.dividend",
                    )
                dividend_action = ZiplineDividend(
                    sid=sid, ex_date=session, amount=dividend
                )
                key = (sid, session)
                prior_dividend = dividends_by_key.get(key)
                if prior_dividend is not None and prior_dividend != dividend_action:
                    raise _BundleFailure(
                        "Conflicting canonical dividend actions were found for "
                        "one SID/session.",
                        (
                            "Validate duplicate normalized rows and publish a "
                            "new snapshot."
                        ),
                        category=ErrorCategory.VALIDATION_DUPLICATE_CONFLICT,
                        field_path="daily_bar.dividend",
                    )
                dividends_by_key[key] = dividend_action

        missing = tuple(symbol for symbol in symbols if symbol not in bounds)
        if missing:
            raise _BundleFailure(
                (
                    "The verified snapshot has no accepted daily bars for one or "
                    "more bundle symbols."
                ),
                (
                    "Ingest complete coverage for the configured symbols and "
                    "SPY before backtesting."
                ),
                category=ErrorCategory.SNAPSHOT_NOT_READY,
                field_path="snapshot.normalized_objects",
            )
        return (
            bounds,
            tuple(
                sorted(
                    splits_by_key.values(),
                    key=lambda item: (item.sid, item.effective_date),
                )
            ),
            tuple(
                sorted(
                    dividends_by_key.values(),
                    key=lambda item: (item.sid, item.ex_date),
                )
            ),
        )

    def _materialize_cache(
        self,
        cache_dir: Path,
        *,
        context: _SnapshotContext,
        bundle_name: str,
        bundle_timestamp: datetime,
        calendar: object,
        refs: Sequence[ContentAddressedObjectRef],
        symbols: Sequence[str],
        assets: Sequence[ZiplineAsset],
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        asset_checksum: str,
        policy_version: str,
        policy_checksum: str,
        action_checksum: str,
        requested_start: date,
        requested_end: date,
    ) -> None:
        parent = cache_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self._adapter_version}.", dir=parent)
        )
        try:
            daily_rows = _ReplayableDailyRows(
                self._daily_rows(
                    refs,
                    symbols=symbols,
                    sid_by_symbol={symbol: sid for sid, symbol in enumerate(symbols)},
                    requested_start=requested_start,
                    requested_end=requested_end,
                )
            )
            self._writer.write(
                output_dir=staging,
                bundle_name=bundle_name,
                bundle_timestamp=bundle_timestamp,
                assets=assets,
                daily_rows=daily_rows,
                splits=splits,
                dividends=dividends,
                start_session=requested_start,
                end_session=requested_end,
                calendar=calendar,
            )
            # Keep a small canonical projection next to the third-party
            # bundle.  It makes the exact action stream and raw daily input
            # inspectable without depending on Zipline's private file layout.
            self._write_projection_files(
                staging,
                assets=assets,
                splits=splits,
                dividends=dividends,
                daily_rows=daily_rows,
            )
            files = self._file_records(staging)
            if not files:
                raise ZiplineBundleWriterError(
                    "The Zipline writer produced no derived bundle files"
                )
            bundle_checksum = self._bundle_checksum(
                snapshot_id=context.handle.snapshot_id,
                snapshot_checksum=context.handle.content_identity_checksum,
                policy_version=policy_version,
                policy_checksum=policy_checksum,
                calendar_name=context.manifest.content_identity.calendar.name,
                calendar_version=context.manifest.content_identity.calendar.version,
                calendar_checksum=context.manifest.content_identity.calendar.schedule_checksum,
                adapter_version=self._adapter_version,
                action_checksum=action_checksum,
                asset_checksum=asset_checksum,
                source_checksums=tuple(sorted(ref.checksum for ref in refs)),
                files=files,
            )
            metadata = {
                "manifest_version": BUNDLE_MANIFEST_VERSION,
                "snapshot_id": context.handle.snapshot_id,
                "snapshot_checksum": context.handle.content_identity_checksum,
                "snapshot_manifest_checksum": context.handle.manifest_checksum,
                "policy_version": policy_version,
                "policy_checksum": policy_checksum,
                "calendar_name": context.manifest.content_identity.calendar.name,
                "calendar_version": context.manifest.content_identity.calendar.version,
                "calendar_checksum": (
                    context.manifest.content_identity.calendar.schedule_checksum
                ),
                "adapter_version": self._adapter_version,
                "bundle_name": bundle_name,
                "bundle_timestamp": canonical_timestamp(bundle_timestamp),
                "asset_checksum": asset_checksum,
                "action_checksum": action_checksum,
                "source_object_checksums": sorted(ref.checksum for ref in refs),
                "symbols": list(symbols),
                "raw_columns": ["open", "high", "low", "close", "volume"],
                "minute_data": False,
                "files": list(files),
                "bundle_checksum": bundle_checksum,
            }
            (staging / _BUNDLE_MANIFEST_NAME).write_bytes(canonical_json(metadata))
            self._verify_cache_directory(
                staging,
                context=context,
                bundle_name=bundle_name,
                adapter_version=self._adapter_version,
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.replace(staging, cache_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _write_projection_files(
        output_dir: Path,
        *,
        assets: Sequence[ZiplineAsset],
        splits: Sequence[ZiplineSplit],
        dividends: Sequence[ZiplineDividend],
        daily_rows: BundleDailyRows,
    ) -> None:
        """Persist stable, raw-only inspection files for every writer backend."""
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
            for _sid, rows in daily_rows:
                for row in rows:
                    handle.write(canonical_json(row.to_content_dict()))

    def _read_verified_cache(
        self,
        cache_dir: Path,
        *,
        context: _SnapshotContext,
        bundle_name: str,
    ) -> ZiplineBundleLocator | None:
        if not cache_dir.is_dir():
            return None
        try:
            metadata_path = cache_dir / _BUNDLE_MANIFEST_NAME
            metadata_bytes = metadata_path.read_bytes()
            metadata = json.loads(metadata_bytes)
            if canonical_json(metadata) != metadata_bytes:
                raise ZiplineBundleIntegrityError("bundle metadata is not canonical")
            self._verify_cache_metadata(
                metadata,
                context=context,
                bundle_name=bundle_name,
                adapter_version=self._adapter_version,
            )
            self._verify_cache_directory(
                cache_dir,
                context=context,
                bundle_name=bundle_name,
                adapter_version=self._adapter_version,
            )
            timestamp = self._parse_timestamp(metadata["bundle_timestamp"])
            return ZiplineBundleLocator(
                bundle_name=metadata["bundle_name"],
                bundle_timestamp=timestamp,
                zipline_root=cache_dir,
                snapshot_id=metadata["snapshot_id"],
                adapter_version=metadata["adapter_version"],
                bundle_checksum=metadata["bundle_checksum"],
            )
        except Exception:
            # A derived cache is disposable.  It is never returned when any
            # byte or metadata check fails; materialize() rebuilds it below.
            shutil.rmtree(cache_dir, ignore_errors=True)
            return None

    def _verify_cache_directory(
        self,
        cache_dir: Path,
        *,
        context: _SnapshotContext,
        bundle_name: str,
        adapter_version: str,
    ) -> None:
        metadata_path = cache_dir / _BUNDLE_MANIFEST_NAME
        metadata = json.loads(metadata_path.read_bytes())
        self._verify_cache_metadata(
            metadata,
            context=context,
            bundle_name=bundle_name,
            adapter_version=adapter_version,
        )
        expected_files = tuple(metadata["files"])
        actual_files = self._file_records(cache_dir)
        if tuple(expected_files) != actual_files:
            raise ZiplineBundleIntegrityError("derived bundle file checksums differ")
        expected_bundle_checksum = metadata["bundle_checksum"]
        actual_bundle_checksum = self._bundle_checksum(
            snapshot_id=metadata["snapshot_id"],
            snapshot_checksum=metadata["snapshot_checksum"],
            policy_version=metadata["policy_version"],
            policy_checksum=metadata["policy_checksum"],
            calendar_name=metadata["calendar_name"],
            calendar_version=metadata["calendar_version"],
            calendar_checksum=metadata["calendar_checksum"],
            adapter_version=metadata["adapter_version"],
            action_checksum=metadata["action_checksum"],
            asset_checksum=metadata["asset_checksum"],
            source_checksums=tuple(metadata["source_object_checksums"]),
            files=actual_files,
        )
        if actual_bundle_checksum != expected_bundle_checksum:
            raise ZiplineBundleIntegrityError("derived bundle checksum differs")

    @staticmethod
    def _verify_cache_metadata(
        metadata: object,
        *,
        context: _SnapshotContext,
        bundle_name: str,
        adapter_version: str,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise ZiplineBundleIntegrityError("bundle metadata must be a mapping")
        identity = context.manifest.content_identity
        expected = {
            "manifest_version": BUNDLE_MANIFEST_VERSION,
            "snapshot_id": context.handle.snapshot_id,
            "snapshot_checksum": context.handle.content_identity_checksum,
            "policy_version": identity.schema_versions.corporate_action_policy_version,
            "calendar_name": identity.calendar.name,
            "calendar_version": identity.calendar.version,
            "calendar_checksum": identity.calendar.schedule_checksum,
            "adapter_version": adapter_version,
            "bundle_name": bundle_name,
        }
        for field_name, expected_value in expected.items():
            if field_name == "adapter_version":
                if (
                    not isinstance(expected_value, str)
                    or metadata.get(field_name) != expected_value
                ):
                    raise ZiplineBundleIntegrityError("bundle adapter version differs")
            elif metadata.get(field_name) != expected_value:
                raise ZiplineBundleIntegrityError(
                    f"bundle metadata field differs: {field_name}"
                )
        if metadata.get("raw_columns") != ["open", "high", "low", "close", "volume"]:
            raise ZiplineBundleIntegrityError(
                "bundle contains a non-raw price projection"
            )
        if metadata.get("minute_data") is not False:
            raise ZiplineBundleIntegrityError(
                "bundle unexpectedly contains minute data"
            )
        if not isinstance(metadata.get("files"), list) or not isinstance(
            metadata.get("source_object_checksums"), list
        ):
            raise ZiplineBundleIntegrityError("bundle metadata file lists are invalid")

    @staticmethod
    def _file_records(root: Path) -> tuple[dict[str, object], ...]:
        result: list[dict[str, object]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == _BUNDLE_MANIFEST_NAME:
                continue
            data = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            result.append(
                {
                    "path": relative,
                    "checksum": sha256_bytes(data),
                    "byte_size": len(data),
                }
            )
        return tuple(result)

    @staticmethod
    def _bundle_checksum(
        *,
        snapshot_id: str,
        snapshot_checksum: str,
        policy_version: str,
        policy_checksum: str,
        calendar_name: str,
        calendar_version: str,
        calendar_checksum: str,
        adapter_version: str,
        action_checksum: str,
        asset_checksum: str,
        source_checksums: Sequence[str],
        files: Sequence[Mapping[str, object]],
    ) -> str:
        return sha256_canonical_json(
            {
                "snapshot_id": snapshot_id,
                "snapshot_checksum": snapshot_checksum,
                "policy_version": policy_version,
                "policy_checksum": policy_checksum,
                "calendar": {
                    "name": calendar_name,
                    "version": calendar_version,
                    "checksum": calendar_checksum,
                },
                "adapter_version": adapter_version,
                "action_checksum": action_checksum,
                "asset_checksum": asset_checksum,
                "source_object_checksums": list(source_checksums),
                "files": list(files),
            }
        )

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise ZiplineBundleIntegrityError("bundle timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ZiplineBundleIntegrityError("bundle timestamp is invalid") from error
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ZiplineBundleIntegrityError("bundle timestamp is not UTC")
        return parsed.astimezone(UTC)

    def _daily_rows(
        self,
        refs: Sequence[ContentAddressedObjectRef],
        *,
        symbols: Sequence[str],
        sid_by_symbol: Mapping[str, int],
        requested_start: date,
        requested_end: date,
    ) -> Iterator[tuple[int, Iterable[ZiplineDailyBar]]]:
        for symbol in symbols:
            sid = sid_by_symbol[symbol]
            source_rows = self._scan_rows(
                refs,
                columns=_DAILY_COLUMNS,
                symbols=(symbol,),
                session_start=requested_start,
                session_end=requested_end,
            )

            def rows_for_symbol(
                values: Iterable[Mapping[str, object]] = source_rows,
                expected_sid: int = sid,
            ) -> Iterator[ZiplineDailyBar]:
                previous_session: date | None = None
                for row in values:
                    session = self._as_date(self._row_value(row, "session"), "session")
                    if previous_session is not None and session <= previous_session:
                        raise ZiplineBundleWriterError(
                            "normalized daily rows are not strictly increasing"
                        )
                    previous_session = session
                    row_symbol = self._symbol_value(row)
                    if (
                        row_symbol not in sid_by_symbol
                        or sid_by_symbol[row_symbol] != expected_sid
                    ):
                        raise ZiplineBundleWriterError(
                            "normalized daily row symbol does not match its SID"
                        )
                    yield ZiplineDailyBar(
                        sid=expected_sid,
                        session=session,
                        open=self._decimal_value(
                            self._row_value(row, "raw_open"), "raw_open"
                        ),
                        high=self._decimal_value(
                            self._row_value(row, "raw_high"), "raw_high"
                        ),
                        low=self._decimal_value(
                            self._row_value(row, "raw_low"), "raw_low"
                        ),
                        close=self._decimal_value(
                            self._row_value(row, "raw_close"), "raw_close"
                        ),
                        volume=self._decimal_value(
                            self._row_value(row, "raw_volume"), "raw_volume"
                        ),
                    )

            yield sid, rows_for_symbol()

    def _scan_rows(
        self,
        refs: Sequence[ContentAddressedObjectRef],
        *,
        columns: Sequence[str],
        symbols: Sequence[str],
        session_start: date,
        session_end: date,
    ) -> Iterator[Mapping[str, object]]:
        source = self._data_source
        if source is None:
            raise _BundleFailure(
                "No published snapshot data source is configured.",
                "Inject the verified Parquet store or snapshot object reader.",
                category=ErrorCategory.STORAGE_IO,
                field_path="data_source",
            )
        scanner = getattr(source, "scan", None)
        predicate = ScanPredicate.from_values(
            symbols=symbols,
            session_start=session_start,
            session_end=session_end,
        )
        if callable(scanner):
            try:
                result = scanner(refs, tuple(columns), predicate=predicate)
            except TypeError:
                result = scanner(
                    refs,
                    tuple(columns),
                    symbols=tuple(symbols),
                    session_start=session_start,
                    session_end=session_end,
                )
            yield from self._iter_scan_result(result)
            return

        for reference in refs:
            data = self._read_source_object(source, reference)
            try:
                parquet_file = pq.ParquetFile(pa.BufferReader(data))
                batches = parquet_file.iter_batches(
                    columns=list(columns), batch_size=65_536, use_threads=False
                )
                for batch in batches:
                    for row in batch.to_pylist():
                        row_symbol = self._symbol_value(row)
                        session = self._as_date(
                            self._row_value(row, "session"), "session"
                        )
                        if row_symbol in symbols and (
                            session_start <= session <= session_end
                        ):
                            yield row
            except Exception as error:
                raise _BundleFailure(
                    "A normalized snapshot Parquet object could not be read.",
                    "Restore the verified snapshot object or publish a new snapshot.",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    checksum=reference.checksum,
                    field_path="snapshot.normalized_objects",
                ) from error

    @staticmethod
    def _iter_scan_result(result: object) -> Iterator[Mapping[str, object]]:
        if isinstance(result, pa.RecordBatchReader):
            for batch in result:
                yield from batch.to_pylist()
            return
        if isinstance(result, pa.Table):
            for batch in result.to_batches(max_chunksize=65_536):
                yield from batch.to_pylist()
            return
        if isinstance(result, Iterable) and not isinstance(
            result, (str, bytes, bytearray)
        ):
            for item in result:
                if not isinstance(item, Mapping):
                    raise _BundleFailure(
                        "The Parquet scan returned a non-mapping row.",
                        "Use the canonical daily_bar_v1 scanner.",
                        category=ErrorCategory.STORAGE_IO,
                        field_path="data_source.scan",
                    )
                yield cast(Mapping[str, object], item)
            return
        raise _BundleFailure(
            "The normalized data source returned an unsupported scan result.",
            "Inject a projected RecordBatchReader or iterable of canonical rows.",
            category=ErrorCategory.STORAGE_IO,
            field_path="data_source.scan",
        )

    @staticmethod
    def _read_source_object(
        source: object, reference: ContentAddressedObjectRef
    ) -> bytes:
        reader = getattr(source, "read_object", None)
        if callable(reader):
            data = reader(reference.relative_uri)
        elif isinstance(source, Mapping):
            data = source[reference.relative_uri]
        elif isinstance(source, (str, Path)):
            data = (Path(source) / reference.relative_uri).read_bytes()
        else:
            raise _BundleFailure(
                "The normalized data source cannot read published objects.",
                "Inject ParquetStore or a snapshot object reader.",
                category=ErrorCategory.STORAGE_IO,
                field_path="data_source.read_object",
            )
        if not isinstance(data, bytes):
            raise _BundleFailure(
                "The normalized data source returned invalid bytes.",
                "Use a verified snapshot object reader.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                checksum=reference.checksum,
                field_path="data_source.read_object",
            )
        if sha256_bytes(data) != reference.checksum:
            raise _BundleFailure(
                "A normalized snapshot object failed checksum verification.",
                "Restore the referenced object or publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                checksum=reference.checksum,
                field_path="snapshot.normalized_objects",
            )
        return data

    @staticmethod
    def _row_value(row: Mapping[str, object], field_name: str) -> object:
        try:
            return row[field_name]
        except KeyError as error:
            raise _BundleFailure(
                f"The normalized scan omitted required field {field_name!r}.",
                "Read the canonical daily_bar_v1 projection.",
                category=ErrorCategory.STORAGE_IO,
                field_path=f"daily_bar.{field_name}",
            ) from error

    @staticmethod
    def _symbol_value(row: Mapping[str, object]) -> str:
        value = ZiplineBundleAdapter._row_value(row, "symbol")
        if not isinstance(value, str):
            raise _BundleFailure(
                "Normalized symbol is not text.",
                "Repair the daily_bar_v1 object and publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="daily_bar.symbol",
            )
        try:
            return normalize_symbol(value)
        except (TypeError, ValueError) as error:
            raise _BundleFailure(
                "Normalized symbol is not a valid ticker.",
                "Repair the daily_bar_v1 object and publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path="daily_bar.symbol",
            ) from error

    @staticmethod
    def _as_date(value: object, field_name: str) -> date:
        if isinstance(value, datetime) or not isinstance(value, date):
            raise _BundleFailure(
                f"Normalized {field_name} is not a calendar date.",
                "Repair the daily_bar_v1 object and publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path=f"daily_bar.{field_name}",
            )
        return value

    @staticmethod
    def _decimal_value(value: object, field_name: str) -> Decimal:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
            try:
                result = Decimal(str(value))
            except (InvalidOperation, ValueError) as error:
                raise _BundleFailure(
                    f"Normalized {field_name} is not a decimal value.",
                    "Repair the daily_bar_v1 object and publish a new snapshot.",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    field_path=f"daily_bar.{field_name}",
                ) from error
        else:
            raise _BundleFailure(
                f"Normalized {field_name} is not a decimal value.",
                "Repair the daily_bar_v1 object and publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path=f"daily_bar.{field_name}",
            )
        if not result.is_finite():
            raise _BundleFailure(
                f"Normalized {field_name} is not finite.",
                "Repair the daily_bar_v1 object and publish a new snapshot.",
                category=ErrorCategory.INTEGRITY_CHECKSUM,
                field_path=f"daily_bar.{field_name}",
            )
        return result

    @staticmethod
    def _checksum_for_row(row: Mapping[str, object]) -> str | None:
        value = row.get("canonical_row_checksum")
        if isinstance(value, bytes) and len(value) == 32:
            return value.hex()
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            return value
        return None


# Public aliases make the seams easy to discover without exposing Zipline's
# version-specific writer classes to the application layer.
ZiplineBundleWriter = BundleWriter
ZiplineDailyRow = ZiplineDailyBar


__all__ = [
    "ADAPTER_VERSION",
    "BUNDLE_MANIFEST_VERSION",
    "BundleWriter",
    "DEFAULT_ZIPLINE_ROOT",
    "ZiplineAsset",
    "ZiplineBundleAdapter",
    "ZiplineBundleError",
    "ZiplineBundleIntegrityError",
    "ZiplineBundleLocator",
    "ZiplineBundleWriter",
    "ZiplineBundleWriterError",
    "ZiplineDailyBar",
    "ZiplineDailyRow",
    "ZiplineDividend",
    "ZiplineSplit",
]
