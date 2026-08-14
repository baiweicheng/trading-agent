"""Focused offline tests for snapshot/run/artifact inspection and paging."""

from __future__ import annotations

from datetime import UTC, date, datetime
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

from quant_research_platform.application.inspection import InspectionService
from quant_research_platform.config.serializer import Redactor
from quant_research_platform.domain.errors import Err, LimitationDisclosure, Ok

NOW = datetime(2024, 1, 10, 12, tzinfo=UTC)
RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
CHECKSUM = sha256(b"table").hexdigest()
SNAPSHOT_ID = "snap_" + "a" * 64


class Metadata:
    def __init__(self) -> None:
        self.invalidated: list[str] = []
        self.artifacts = {
            CHECKSUM: SimpleNamespace(
                checksum=CHECKSUM,
                artifact_kind="equity",
                relative_uri="artifacts/sha256/aa/" + CHECKSUM + ".parquet",
                media_type="application/vnd.apache.parquet",
                byte_size=5,
                row_count=6,
                schema_version="daily_bar_v1",
                availability="available",
            )
        }
        self.run = SimpleNamespace(
            run_id=RUN_ID,
            snapshot_id=SNAPSHOT_ID,
            state="succeeded",
            strategy_id="monthly_momentum_v1",
            evaluation_start=date(2024, 1, 2),
            evaluation_end=date(2024, 1, 9),
            universe=("AAPL", "MSFT"),
            configuration_checksum="b" * 64,
            environment_checksum="c" * 64,
            manifest_checksum="d" * 64,
            created_at=NOW,
            ended_at=NOW,
            configuration={
                "secrets": {"https_proxy": "super-secret"},
                "safe": "visible",
            },
            environment_fingerprint={"python_version": "3.11"},
            validation_report={"accepted": 2},
            logs=("run started",),
        )
        self.links = (
            SimpleNamespace(checksum=CHECKSUM, role="equity", scientific=True),
        )

    def get_artifact(self, checksum: str) -> object:
        return self.artifacts[checksum]

    def set_artifact_availability(self, checksum: str, availability: str) -> None:
        self.invalidated.append(checksum)
        self.artifacts[checksum].availability = availability

    def get_run(self, run_id: UUID) -> object:
        assert run_id == RUN_ID
        return self.run

    def list_run_artifacts(self, run_id: UUID) -> tuple[object, ...]:
        assert run_id == RUN_ID
        return self.links


class NativeScanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.rows = tuple(
            {"session": index, "symbol": "AAPL", "close": index} for index in range(6)
        )

    def scan(
        self,
        refs: object,
        columns: object,
        predicate: object = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        order_by: object = (),
    ) -> tuple[dict[str, object], ...]:
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


class LazyArtifactStore:
    def __init__(self, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.opened = 0

    def open_verified_artifact(self, reference: object):
        self.opened += 1
        if self.corrupt:
            raise OSError("checksum mismatch")
        return lambda: iter((b"table",))


class Snapshot:
    def inspect_snapshot(self, snapshot_id: str):
        return Ok(
            SimpleNamespace(
                snapshot_id=snapshot_id,
                limitation_disclosure=LimitationDisclosure.current(),
            )
        )


def test_snapshot_inspection_delegates_to_verified_snapshot_manager() -> None:
    service = InspectionService(snapshot_manager=Snapshot())

    result = service.inspect_snapshot(SNAPSHOT_ID)

    assert isinstance(result, Ok)
    assert result.value.snapshot_id == SNAPSHOT_ID


def test_run_inspection_returns_artifacts_and_redacts_configuration() -> None:
    metadata = Metadata()
    service = InspectionService(metadata=metadata, redactor=Redactor(("super-secret",)))

    result = service.inspect_run(RUN_ID)

    assert isinstance(result, Ok)
    assert result.value.summary.run_id == RUN_ID
    assert result.value.artifacts[0].role == "equity"
    assert result.value.configuration["secrets"]["https_proxy"] == "[REDACTED]"  # type: ignore[index]
    assert "super-secret" not in str(result.value.configuration)
    assert result.value.validation_report == {"accepted": 2}


def test_missing_artifact_record_is_an_actionable_integrity_result() -> None:
    metadata = Metadata()
    metadata.links = (
        SimpleNamespace(checksum="e" * 64, role="missing", scientific=True),
    )
    service = InspectionService(metadata=metadata)

    result = service.inspect_run(RUN_ID)

    assert isinstance(result, Err)
    assert result.errors[0].category.value == "integrity.checksum"
    assert "artifact" in result.errors[0].corrective_action.lower()


def test_paging_clamps_page_size_orders_and_uses_native_window() -> None:
    metadata = Metadata()
    scanner = NativeScanner()
    service = InspectionService(
        metadata=metadata, scanner=scanner, configured_page_size=3
    )

    first = service.page_artifact(
        CHECKSUM,
        page=0,
        page_size=100,
        columns=("session", "close"),
        order_by=("session",),
    )
    second = service.page_artifact(
        CHECKSUM,
        page=1,
        page_size=100,
        columns=("session", "close"),
        order_by=("session",),
    )

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.page_size == 3
    assert [row["session"] for row in first.value.rows] == [0, 1, 2]
    assert [row["session"] for row in second.value.rows] == [3, 4, 5]
    assert first.value.has_next
    assert not second.value.has_next
    assert scanner.calls[0]["offset"] == 0
    assert scanner.calls[0]["limit"] == 3
    assert scanner.calls[0]["columns"] == ("session", "close")
    assert scanner.calls[0]["order_by"] == ("session",)


def test_full_download_is_lazy_and_separate_from_table_paging() -> None:
    metadata = Metadata()
    store = LazyArtifactStore()
    scanner = NativeScanner()
    service = InspectionService(metadata=metadata, artifacts=store, scanner=scanner)

    opened = service.open_artifact(CHECKSUM)

    assert isinstance(opened, Ok)
    assert store.opened == 1
    assert scanner.calls == []
    assert list(opened.value.stream()) == [b"table"]


def test_corrupt_download_returns_integrity_error_and_marks_artifact_invalid() -> None:
    metadata = Metadata()
    service = InspectionService(
        metadata=metadata, artifacts=LazyArtifactStore(corrupt=True)
    )

    result = service.open_artifact(CHECKSUM)

    assert isinstance(result, Err)
    assert result.errors[0].category.value == "integrity.checksum"
    assert metadata.invalidated == [CHECKSUM]
