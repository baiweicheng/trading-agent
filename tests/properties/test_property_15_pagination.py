"""Property tests for bounded ordinary artifact-table pagination."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_research_platform.application.inspection import InspectionService
from quant_research_platform.domain.errors import Ok

CHECKSUM = sha256(b"table").hexdigest()
_COLUMNS = ("session", "value")


@dataclass(frozen=True, slots=True)
class PaginationCase:
    """Generated inputs for one bounded table-pagination scenario."""

    artifact_size: int
    page: int
    requested_size: int
    configured_size: int


@st.composite
def pagination_cases(draw: st.DrawFn) -> PaginationCase:
    """Generate artifact sizes and all page-size inputs independently."""

    return PaginationCase(
        artifact_size=draw(st.integers(min_value=0, max_value=500)),
        page=draw(st.integers(min_value=0, max_value=20)),
        requested_size=draw(st.integers(min_value=1, max_value=250)),
        configured_size=draw(st.integers(min_value=1, max_value=100)),
    )


class ArtifactMetadata:
    """Minimal metadata repository for a generated table artifact."""

    def __init__(self, row_count: int) -> None:
        self.record = SimpleNamespace(
            checksum=CHECKSUM,
            artifact_kind="table",
            relative_uri=f"artifacts/sha256/aa/{CHECKSUM}.parquet",
            media_type="application/vnd.apache.parquet",
            byte_size=5,
            row_count=row_count,
            schema_version="table_v1",
            availability="available",
            columns=_COLUMNS,
        )

    def get_artifact(self, checksum: str) -> object:
        assert checksum == CHECKSUM
        return self.record


class WindowScanner:
    """Scanner spy that applies the native bounded window it receives."""

    def __init__(self, artifact_size: int) -> None:
        self.rows = tuple(
            {"session": index, "value": index * 2}
            for index in range(artifact_size)
        )
        self.calls: list[dict[str, object]] = []

    def scan(
        self,
        refs: object,
        columns: object,
        predicate: object = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        order_by: object = (),
    ) -> tuple[dict[str, int], ...]:
        self.calls.append(
            {
                "refs": refs,
                "columns": columns,
                "predicate": predicate,
                "offset": offset,
                "limit": limit,
                "order_by": order_by,
            }
        )
        end = None if limit is None else offset + limit
        return self.rows[offset:end]


class DownloadStore:
    """Lazy download spy used to prove downloads bypass ordinary table paging."""

    def __init__(self) -> None:
        self.open_calls = 0

    def open_verified_artifact(self, reference: object):
        del reference
        self.open_calls += 1
        return lambda: iter((b"table",))


# Feature: quantitative-research-platform, Property 15: Ordinary table pagination is absolutely bounded
# Validates: Requirements 13.15–13.17, 15.6, 17.28
@given(case=pagination_cases())
@settings(max_examples=100, deadline=None)
def test_ordinary_table_pagination_is_bounded_and_separate(
    case: PaginationCase,
) -> None:
    """Pages are deterministic bounded slices and full downloads use another path."""

    metadata = ArtifactMetadata(case.artifact_size)
    scanner = WindowScanner(case.artifact_size)
    downloads = DownloadStore()
    service = InspectionService(
        metadata=metadata,
        scanner=scanner,
        artifacts=downloads,
        configured_page_size=case.configured_size,
    )

    first = service.page_artifact(
        CHECKSUM,
        page=case.page,
        page_size=case.requested_size,
        columns=_COLUMNS,
        order_by=("session",),
    )
    repeat = service.page_artifact(
        CHECKSUM,
        page=case.page,
        page_size=case.requested_size,
        columns=_COLUMNS,
        order_by=("session",),
    )
    adjacent = service.page_artifact(
        CHECKSUM,
        page=case.page + 1,
        page_size=case.requested_size,
        columns=_COLUMNS,
        order_by=("session",),
    )

    assert isinstance(first, Ok)
    assert isinstance(repeat, Ok)
    assert isinstance(adjacent, Ok)

    effective_size = min(case.requested_size, case.configured_size, 100)
    first_start = case.page * effective_size
    adjacent_start = (case.page + 1) * effective_size
    expected_first = scanner.rows[first_start : first_start + effective_size]
    expected_adjacent = scanner.rows[adjacent_start : adjacent_start + effective_size]

    assert first.value.page_size == effective_size
    assert repeat.value.page_size == effective_size
    assert adjacent.value.page_size == effective_size
    assert len(first.value.rows) <= 100
    assert len(repeat.value.rows) <= 100
    assert len(adjacent.value.rows) <= 100
    assert first.value.rows == expected_first
    assert repeat.value.rows == expected_first
    assert adjacent.value.rows == expected_adjacent
    assert first.value.rows == repeat.value.rows
    assert {
        row["session"] for row in first.value.rows
    }.isdisjoint(row["session"] for row in adjacent.value.rows)

    assert len(scanner.calls) == 3
    for call, expected_page in zip(
        scanner.calls,
        (case.page, case.page, case.page + 1),
        strict=True,
    ):
        assert call["offset"] == expected_page * effective_size
        assert call["limit"] == effective_size
        assert call["columns"] == _COLUMNS
        assert call["order_by"] == ("session",)

    assert downloads.open_calls == 0
    downloaded = service.open_artifact(CHECKSUM)
    assert isinstance(downloaded, Ok)
    assert downloads.open_calls == 1
    assert len(scanner.calls) == 3
