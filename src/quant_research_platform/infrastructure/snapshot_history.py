"""Verified snapshot projections for causal strategy history reads.

The decision layer deliberately knows nothing about Parquet or CAS paths.  This
adapter is the infrastructure boundary that turns an immutable verified
snapshot handle into a bounded daily-bar projection while preserving the
snapshot's exact object references.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from typing import Final

from ..domain.manifests import (
    ContentAddressedObjectRef,
    ObjectKind,
    VerifiedSnapshotHandle,
)
from ..domain.market import normalize_symbol
from .parquet_store import DEFAULT_SCAN_BATCH_SIZE, MAX_WRITE_CHUNK_SIZE, ParquetStore
from .schemas import DAILY_BAR_V1

_HISTORY_FIELDS: Final = frozenset(
    {
        "symbol",
        "session",
        "adjusted_close",
        "sizing_adjusted_close",
        "canonical_row_checksum",
        "tradable",
    }
)
_PHYSICAL_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "session",
    "adjusted_close",
    "sizing_adjusted_close",
    "canonical_row_checksum",
)
_CHECKSUM_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class SnapshotHistoryReadError(RuntimeError):
    """Raised when a verified snapshot cannot provide daily-bar history."""


class SnapshotParquetHistoryReader:
    """Read bounded, normalized history from one verified snapshot handle.

    The reader never scans a mutable collection directory.  It selects only
    normalized references pinned by the supplied handle and delegates path
    resolution, CAS handling, projection, and inclusive predicates to the
    configured :class:`ParquetStore`.
    """

    def __init__(
        self,
        parquet_store: ParquetStore,
        *,
        scan_batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
    ) -> None:
        if not isinstance(parquet_store, ParquetStore):
            raise TypeError("parquet_store must be a ParquetStore")
        if (
            isinstance(scan_batch_size, bool)
            or not isinstance(scan_batch_size, int)
            or not 1 <= scan_batch_size <= MAX_WRITE_CHUNK_SIZE
        ):
            raise ValueError(
                f"scan_batch_size must be between 1 and {MAX_WRITE_CHUNK_SIZE}"
            )
        self.parquet_store = parquet_store
        self.scan_batch_size = scan_batch_size

    def read_history(
        self,
        snapshot: VerifiedSnapshotHandle,
        *,
        symbols: tuple[str, ...],
        end_session: date,
        fields: tuple[str, ...],
        start_session: date | None = None,
    ) -> Iterator[dict[str, object]]:
        """Yield an inclusive, projected daily-bar history one Arrow batch at a time."""
        handle = _verified_handle(snapshot)
        requested_symbols = _symbols(symbols)
        end = _session(end_session, "end_session")
        start = (
            None if start_session is None else _session(start_session, "start_session")
        )
        if start is not None and start > end:
            raise ValueError("start_session must not be after end_session")
        requested_fields = _fields(fields)
        references = _normalized_references(handle.object_references)
        if not references:
            raise SnapshotHistoryReadError(
                "verified snapshot has no normalized daily-bar objects"
            )

        physical_fields = tuple(
            field for field in _PHYSICAL_FIELDS if field in requested_fields
        )
        if not physical_fields:
            # A logical projection containing only the derived tradability flag
            # still needs a stable row identity from the physical table.
            physical_fields = ("symbol", "session")

        reader = self.parquet_store.scan(
            references,
            columns=physical_fields,
            symbols=requested_symbols,
            session_start=start,
            session_end=end,
            batch_size=self.scan_batch_size,
        )
        for batch in reader:
            for row in batch.to_pylist():
                yield {field: _logical_value(field, row) for field in requested_fields}

    # Structural compatibility aliases used by older application seams.
    read_daily_bars = read_history
    read_bars = read_history
    read = read_history


def _verified_handle(value: object) -> VerifiedSnapshotHandle:
    if not isinstance(value, VerifiedSnapshotHandle):
        raise TypeError("snapshot history requires a verified snapshot handle")
    return value


def _normalized_references(
    references: Sequence[ContentAddressedObjectRef],
) -> tuple[ContentAddressedObjectRef, ...]:
    return tuple(
        sorted(
            (
                reference
                for reference in references
                if reference.object_kind is ObjectKind.NORMALIZED
                and reference.schema_version == DAILY_BAR_V1
            ),
            key=ContentAddressedObjectRef.sort_key,
        )
    )


def _symbols(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("symbols must be a non-empty tuple")
    normalized = tuple(normalize_symbol(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("symbols must contain distinct values")
    return normalized


def _fields(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("fields must be a non-empty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("fields must contain non-empty names")
    if len(set(values)) != len(values):
        raise ValueError("fields must not contain duplicates")
    unknown = sorted(set(values) - _HISTORY_FIELDS)
    if unknown:
        raise ValueError(f"unsupported history fields: {', '.join(unknown)}")
    return values


def _session(value: date, field_name: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _logical_value(field: str, row: dict[str, object]) -> object:
    if field == "tradable":
        # daily_bar_v1 contains only accepted normalized rows.  Tradability is
        # therefore a derived logical projection, not a persisted column.
        return True
    value = row.get(field)
    if field == "canonical_row_checksum":
        return _checksum_text(value)
    return value


def _checksum_text(value: object) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        result = bytes(value).hex()
    elif isinstance(value, str):
        result = value.lower()
    else:
        raise SnapshotHistoryReadError(
            "normalized history returned an invalid canonical_row_checksum"
        )
    if _CHECKSUM_RE.fullmatch(result) is None:
        raise SnapshotHistoryReadError(
            "normalized history returned a non-SHA-256 canonical_row_checksum"
        )
    return result


ParquetSnapshotHistoryReader = SnapshotParquetHistoryReader

__all__ = [
    "ParquetSnapshotHistoryReader",
    "SnapshotHistoryReadError",
    "SnapshotParquetHistoryReader",
]
