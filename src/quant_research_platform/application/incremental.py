"""Revision-overlap incremental snapshot merging.

This module owns the scientific part of an incremental update.  It deliberately
keeps publication separate from merging: a verified parent is read, the
contiguous overlap/later-session suffix is normalized from the parent's causal
state, the complete logical result is validated, and a content-derived result
ID is produced.  Existing parent objects are never edited.  A later ingestion
or publication service can use ``reused_object_references`` and
``rebuilt_partition_keys`` to stage a new immutable publication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

from ..domain.canonical import sha256_canonical_json
from ..domain.errors import (
    ActionableError,
    Err,
    ErrorCategory,
    LimitationDisclosure,
    Ok,
    Result,
)
from ..domain.manifests import (
    ContentAddressedObjectRef,
    ObjectKind,
    SnapshotManifest,
    VerifiedSnapshotHandle,
)
from ..domain.market import (
    DailyBarCandidate,
    DataGap,
    DateRange,
    ProviderBatchResult,
    ProviderRecord,
    QuarantineRecord,
    SymbolOutcome,
    SymbolOutcomeStatus,
    ValidationReport,
    normalize_symbol,
)
from ..domain.normalization import (
    CorporateActionPolicy,
    NormalizationSeed,
    Normalizer,
)
from ..domain.validation import ValidationOutput, ValidationService

CompletedAt = datetime.max.replace(tzinfo=UTC)
RecordInput: TypeAlias = (
    Iterable[ProviderRecord]
    | Mapping[str, Iterable[ProviderRecord] | ProviderRecord | SymbolOutcome]
    | ProviderBatchResult
    | Iterable[SymbolOutcome]
)


class SnapshotVerifier(Protocol):
    """Minimal verified-parent port used by :class:`IncrementalMerger`."""

    def open_verified(self, snapshot_id: str) -> Result[VerifiedSnapshotHandle]:
        """Verify every manifest/object checksum before an update reads it."""


class SnapshotContentLoader(Protocol):
    """Optional loader for rows not embedded in a snapshot handle."""

    def __call__(
        self, handle: VerifiedSnapshotHandle, manifest: SnapshotManifest
    ) -> IncrementalParent | Mapping[str, object]:
        """Return the verified parent's logical rows and raw provenance."""


class _IncrementalFailure(Exception):
    """Internal exception carrying already-safe application errors."""

    def __init__(self, errors: Sequence[ActionableError]) -> None:
        errors_tuple = tuple(errors)
        if not errors_tuple:
            raise ValueError("incremental failure requires at least one error")
        super().__init__(errors_tuple[0].message)
        self.errors = errors_tuple


@dataclass(frozen=True, slots=True)
class IncrementalParent:
    """Verified parent content supplied to an incremental merge.

    ``SnapshotManifest`` and its object references are scientific/operational
    metadata from a verified publication.  The row tuples are the corresponding
    logical content loaded through a checksum-verified Parquet reader.  Keeping
    them separate makes this DTO useful both with the real storage adapters and
    with bounded local fixtures.
    """

    manifest: SnapshotManifest
    accepted_rows: tuple[DailyBarCandidate, ...] = ()
    provider_records: tuple[ProviderRecord, ...] = ()
    quarantined_rows: tuple[QuarantineRecord, ...] = ()
    expected_sessions: tuple[tuple[str, tuple[date, ...]], ...] = ()
    validation_report: ValidationReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SnapshotManifest):
            raise TypeError("manifest must be a SnapshotManifest")
        if not isinstance(self.accepted_rows, tuple):
            raise TypeError("accepted_rows must be an immutable tuple")
        if any(not isinstance(row, DailyBarCandidate) for row in self.accepted_rows):
            raise TypeError("accepted_rows must contain DailyBarCandidate values")
        if not isinstance(self.provider_records, tuple):
            raise TypeError("provider_records must be an immutable tuple")
        if any(not isinstance(row, ProviderRecord) for row in self.provider_records):
            raise TypeError("provider_records must contain ProviderRecord values")
        if not isinstance(self.quarantined_rows, tuple):
            raise TypeError("quarantined_rows must be an immutable tuple")
        if any(not isinstance(row, QuarantineRecord) for row in self.quarantined_rows):
            raise TypeError(
                "quarantined_rows must contain QuarantineRecord values"
            )
        if self.validation_report is not None and not isinstance(
            self.validation_report, ValidationReport
        ):
            raise TypeError("validation_report must be a ValidationReport or None")

        expected: list[tuple[str, tuple[date, ...]]] = []
        seen_symbols: set[str] = set()
        for symbol, sessions in self.expected_sessions:
            normalized = normalize_symbol(symbol)
            if normalized in seen_symbols:
                raise ValueError("expected_sessions must contain one entry per symbol")
            if not isinstance(sessions, tuple):
                raise TypeError("expected session values must be immutable tuples")
            normalized_sessions = tuple(sorted(set(sessions)))
            if any(
                isinstance(session, datetime) or not isinstance(session, date)
                for session in normalized_sessions
            ):
                raise TypeError("expected sessions must contain calendar dates")
            expected.append((normalized, normalized_sessions))
            seen_symbols.add(normalized)
        object.__setattr__(
            self,
            "accepted_rows",
            tuple(sorted(self.accepted_rows, key=DailyBarCandidate.sort_key)),
        )
        object.__setattr__(
            self,
            "provider_records",
            tuple(sorted(self.provider_records, key=ProviderRecord.sort_key)),
        )
        object.__setattr__(
            self,
            "quarantined_rows",
            tuple(sorted(self.quarantined_rows, key=_quarantine_sort_key)),
        )
        object.__setattr__(self, "expected_sessions", tuple(sorted(expected)))

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def requested_range(self) -> DateRange:
        return self.manifest.content_identity.requested_range

    @property
    def configured_symbols(self) -> tuple[str, ...]:
        identity = self.manifest.content_identity
        return (*identity.configured_universe, identity.benchmark_symbol)

    @property
    def object_references(self) -> tuple[ContentAddressedObjectRef, ...]:
        return self.manifest.content_identity.objects

    def expected_map(self) -> dict[str, tuple[date, ...]]:
        return {symbol: sessions for symbol, sessions in self.expected_sessions}

    @classmethod
    def from_manifest(
        cls,
        manifest: SnapshotManifest,
        *,
        accepted_rows: Iterable[DailyBarCandidate] = (),
        provider_records: Iterable[ProviderRecord] = (),
        quarantined_rows: Iterable[QuarantineRecord] = (),
        expected_sessions: Mapping[str, Sequence[date]] | None = None,
        validation_report: ValidationReport | None = None,
    ) -> IncrementalParent:
        expected = () if expected_sessions is None else tuple(
            (normalize_symbol(symbol), tuple(sessions))
            for symbol, sessions in expected_sessions.items()
        )
        return cls(
            manifest=manifest,
            accepted_rows=tuple(accepted_rows),
            provider_records=tuple(provider_records),
            quarantined_rows=tuple(quarantined_rows),
            expected_sessions=expected,
            validation_report=validation_report,
        )


# The shorter name reads naturally at application call sites.
ParentSnapshot = IncrementalParent


@dataclass(frozen=True, slots=True)
class IncrementalUpdateRequest:
    """Validated range/overlap inputs for one incremental update."""

    requested_range: DateRange
    revision_overlap: int = 5
    staleness_threshold: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.requested_range, DateRange):
            raise TypeError("requested_range must be a DateRange")
        _require_non_negative_int("revision_overlap", self.revision_overlap)
        if self.revision_overlap > 252:
            raise ValueError("revision_overlap must not exceed 252 sessions")
        _require_non_negative_int("staleness_threshold", self.staleness_threshold)
        if self.staleness_threshold > 252:
            raise ValueError("staleness_threshold must not exceed 252 sessions")


@dataclass(frozen=True, slots=True)
class IncrementalPlan:
    """The exact contiguous suffix requested from the provider."""

    parent_snapshot_id: str
    parent_range: DateRange
    requested_range: DateRange
    revision_overlap: int
    parent_sessions: tuple[date, ...]
    overlap_sessions: tuple[date, ...]
    later_sessions: tuple[date, ...]
    suffix_sessions: tuple[date, ...]
    boundary_session: date | None

    @property
    def is_extension(self) -> bool:
        return self.requested_range.end > self.parent_range.end

    @property
    def has_provider_suffix(self) -> bool:
        return bool(self.suffix_sessions)

    @property
    def suffix_range(self) -> DateRange | None:
        if not self.suffix_sessions:
            return None
        return DateRange(self.suffix_sessions[0], self.suffix_sessions[-1])

    @property
    def later_session_count(self) -> int:
        return len(self.later_sessions)


@dataclass(frozen=True, slots=True)
class IncrementalMergeResult:
    """Validated scientific result and publication plan for one update."""

    parent_snapshot_id: str
    snapshot_id: str
    plan: IncrementalPlan
    status: str
    accepted_rows: tuple[DailyBarCandidate, ...]
    quarantined_rows: tuple[QuarantineRecord, ...]
    gaps: tuple[DataGap, ...]
    validation: ValidationOutput
    provider_records: tuple[ProviderRecord, ...]
    failed_symbols: tuple[str, ...]
    retained_parent_coverage_symbols: tuple[str, ...]
    failure_errors: tuple[ActionableError, ...]
    limitation_disclosure: LimitationDisclosure
    reused_object_references: tuple[ContentAddressedObjectRef, ...] = ()
    rebuilt_partition_keys: tuple[tuple[str, str, int], ...] = ()
    new_rows: tuple[DailyBarCandidate, ...] = ()
    content_identity: Mapping[str, object] = field(default_factory=dict)
    manifest: SnapshotManifest | None = None
    publication: object | None = None

    def __post_init__(self) -> None:
        if not self.snapshot_id.startswith("snap_"):
            raise ValueError("snapshot_id must be content-derived")
        if self.status not in {"succeeded", "partially_succeeded", "failed"}:
            raise ValueError("unsupported incremental status")
        if not isinstance(self.plan, IncrementalPlan):
            raise TypeError("plan must be an IncrementalPlan")
        if not isinstance(self.validation, ValidationOutput):
            raise TypeError("validation must be a ValidationOutput")
        if not isinstance(self.limitation_disclosure, LimitationDisclosure):
            raise TypeError("limitation_disclosure must be a LimitationDisclosure")
        if not isinstance(self.content_identity, Mapping):
            raise TypeError("content_identity must be a mapping")
        object.__setattr__(
            self,
            "accepted_rows",
            tuple(sorted(self.accepted_rows, key=DailyBarCandidate.sort_key)),
        )
        object.__setattr__(
            self,
            "quarantined_rows",
            tuple(sorted(self.quarantined_rows, key=_quarantine_sort_key)),
        )
        object.__setattr__(self, "gaps", tuple(sorted(self.gaps, key=DataGap.sort_key)))
        object.__setattr__(
            self,
            "provider_records",
            tuple(sorted(self.provider_records, key=ProviderRecord.sort_key)),
        )
        object.__setattr__(
            self, "failed_symbols", _normalized_symbols(self.failed_symbols)
        )
        object.__setattr__(
            self,
            "retained_parent_coverage_symbols",
            _normalized_symbols(self.retained_parent_coverage_symbols),
        )
        object.__setattr__(
            self,
            "failure_errors",
            tuple(sorted(self.failure_errors, key=ActionableError.sort_key)),
        )
        object.__setattr__(
            self,
            "reused_object_references",
            tuple(
                sorted(
                    self.reused_object_references,
                    key=ContentAddressedObjectRef.sort_key,
                )
            ),
        )
        object.__setattr__(
            self,
            "rebuilt_partition_keys",
            tuple(sorted(set(self.rebuilt_partition_keys))),
        )
        object.__setattr__(
            self,
            "content_identity",
            MappingProxyType(dict(self.content_identity)),
        )

    @property
    def reused_parent(self) -> bool:
        return self.snapshot_id == self.parent_snapshot_id

    @property
    def failed_without_parent_coverage(self) -> tuple[str, ...]:
        return tuple(
            symbol
            for symbol in self.failed_symbols
            if symbol not in self.retained_parent_coverage_symbols
        )

    @property
    def retained_symbols(self) -> tuple[str, ...]:
        return self.retained_parent_coverage_symbols

    @property
    def accepted(self) -> tuple[DailyBarCandidate, ...]:
        return self.accepted_rows

    @property
    def quarantined(self) -> tuple[QuarantineRecord, ...]:
        return self.quarantined_rows

    @property
    def reused_objects(self) -> tuple[ContentAddressedObjectRef, ...]:
        return self.reused_object_references

    @property
    def rebuilt_partitions(self) -> tuple[tuple[str, str, int], ...]:
        return self.rebuilt_partition_keys


# Descriptive aliases used by callers that name the operation a merge/update.
IncrementalResult = IncrementalMergeResult
IncrementalUpdateResult = IncrementalMergeResult


class IncrementalMerger:
    """Plan and merge a revision-overlap update from one verified parent."""

    def __init__(
        self,
        calendar: object,
        *,
        normalizer: Normalizer | None = None,
        policy: CorporateActionPolicy | None = None,
        validator: ValidationService | None = None,
        snapshot_manager: SnapshotVerifier | None = None,
        parent_loader: SnapshotContentLoader | None = None,
    ) -> None:
        if calendar is None:
            raise TypeError("calendar is required")
        self.calendar: Any = calendar
        self.normalizer = normalizer or Normalizer(policy)
        self.policy = policy
        self.validator = validator
        self.snapshot_manager = snapshot_manager
        self.parent_loader = parent_loader

    def plan(
        self,
        parent: IncrementalParent | SnapshotManifest | object,
        requested_range: DateRange,
        revision_overlap: int = 5,
    ) -> IncrementalPlan:
        """Verify range invariants and compute one contiguous suffix.

        ``requested_range.start`` must equal the parent start.  A smaller end is
        rejected rather than silently converted into a full/back-extension
        ingestion.  With zero overlap the suffix starts at the first XNYS
        session strictly after the parent end; with non-zero overlap it starts
        at the first of the parent's final ``revision_overlap`` sessions.
        """

        resolved_parent = self._coerce_parent(parent)
        request = IncrementalUpdateRequest(requested_range, revision_overlap)
        parent_range = resolved_parent.requested_range
        if request.requested_range.start != parent_range.start:
            raise ValueError(
                "incremental requested start must equal the parent requested start"
            )
        if request.requested_range.end < parent_range.end:
            raise ValueError(
                "incremental requested end must not precede the parent requested end"
            )

        expected = resolved_parent.expected_map()
        parent_sessions = tuple(
            sorted(
                {
                    session
                    for sessions in expected.values()
                    for session in sessions
                    if parent_range.start <= session <= parent_range.end
                }
            )
        )
        if not parent_sessions:
            parent_sessions = self._calendar_sessions(
                parent_range.start, parent_range.end
            )

        boundary: date | None = None
        if revision_overlap:
            if parent_sessions:
                boundary = parent_sessions[
                    max(0, len(parent_sessions) - revision_overlap)
                ]
            else:
                parent_calendar_sessions = self._calendar_sessions(
                    parent_range.start, parent_range.end
                )
                boundary = (
                    parent_calendar_sessions[
                        max(0, len(parent_calendar_sessions) - revision_overlap)
                    ]
                    if parent_calendar_sessions
                    else None
                )
            suffix_start = boundary
            overlap_sessions = (
                tuple(session for session in parent_sessions if session >= boundary)
                if boundary is not None
                else ()
            )
        else:
            boundary = None
            suffix_start = parent_range.end + timedelta(days=1)
            overlap_sessions = ()

        later_sessions = tuple(
            session
            for session in self._calendar_sessions(
                parent_range.end + timedelta(days=1), request.requested_range.end
            )
            if session > parent_range.end
        )
        if suffix_start is None:
            suffix_sessions = later_sessions
            boundary = suffix_sessions[0] if suffix_sessions else None
        else:
            suffix_sessions = tuple(
                session
                for session in self._calendar_sessions(
                    suffix_start, request.requested_range.end
                )
                if session >= suffix_start
            )
            if boundary is None and suffix_sessions:
                # Zero-overlap updates still need a causal boundary: it is the
                # first strictly later XNYS session, not an absent state.
                boundary = suffix_sessions[0]
        return IncrementalPlan(
            parent_snapshot_id=resolved_parent.snapshot_id,
            parent_range=parent_range,
            requested_range=request.requested_range,
            revision_overlap=revision_overlap,
            parent_sessions=parent_sessions,
            overlap_sessions=overlap_sessions,
            later_sessions=later_sessions,
            suffix_sessions=suffix_sessions,
            boundary_session=boundary,
        )

    def merge(
        self,
        parent: IncrementalParent | SnapshotManifest | str | object | None = None,
        requested_range: DateRange | None = None,
        revision_overlap: int = 5,
        *,
        parent_snapshot_id: str | None = None,
        records: RecordInput = (),
        provider_records: RecordInput | None = None,
        provider_outcomes: ProviderBatchResult | Iterable[SymbolOutcome] | None = None,
        failed_symbols: Iterable[str] = (),
        parent_records: Iterable[ProviderRecord] | None = None,
        parent_rows: Iterable[DailyBarCandidate] | None = None,
        parent_quarantined_rows: Iterable[QuarantineRecord] | None = None,
        expected_sessions: Mapping[str, Sequence[date]] | None = None,
        staleness_threshold: int = 1,
    ) -> Result[IncrementalMergeResult]:
        """Merge a verified update and return sanitized errors instead of raising."""

        try:
            if requested_range is None:
                raise ValueError(
                    "requested_range is required for an incremental update"
                )
            resolved_parent = self._resolve_and_verify_parent(
                parent,
                parent_snapshot_id=parent_snapshot_id,
            )
            if provider_records is not None:
                if records != ():
                    raise ValueError(
                        "supply either records or provider_records, not both"
                    )
                records = provider_records
            plan = self.plan(resolved_parent, requested_range, revision_overlap)
            request = IncrementalUpdateRequest(
                requested_range,
                revision_overlap,
                staleness_threshold,
            )
            del request
            incoming_records, outcome_failures, outcome_symbols = _materialize_input(
                records,
                provider_outcomes,
            )
            explicit_failed = _normalized_symbols(failed_symbols)
            all_failed = _normalized_symbols(
                (*explicit_failed, *outcome_symbols)
            )
            failure_errors = _merge_errors(outcome_failures, all_failed)

            symbols = resolved_parent.configured_symbols
            unknown = tuple(
                sorted({record.symbol for record in incoming_records} - set(symbols))
            )
            if unknown:
                raise ValueError(
                    "incremental provider records contain symbols outside the parent "
                    f"universe: {', '.join(unknown)}"
                )

            expected = self._expected_sessions(
                resolved_parent,
                requested_range,
                expected_sessions,
                symbols,
            )
            boundary = plan.boundary_session
            suffix_start = _suffix_start(plan)
            parent_rows_all = tuple(
                resolved_parent.accepted_rows
                if parent_rows is None
                else tuple(parent_rows)
            )
            parent_records_all = tuple(
                resolved_parent.provider_records
                if parent_records is None
                else tuple(parent_records)
            )
            parent_quarantine_all = tuple(
                resolved_parent.quarantined_rows
                if parent_quarantined_rows is None
                else tuple(parent_quarantined_rows)
            )

            parent_rows_by_symbol: dict[str, tuple[DailyBarCandidate, ...]] = {
                symbol: tuple(
                    row for row in parent_rows_all if row.symbol == symbol
                )
                for symbol in symbols
            }
            retained_symbols = tuple(
                sorted(
                    symbol
                    for symbol in all_failed
                    if parent_rows_by_symbol.get(symbol)
                )
            )

            retained_rows = self._retained_rows(
                parent_rows_all,
                plan,
                all_failed,
            )
            retained_quarantine = self._retained_quarantines(
                parent_quarantine_all,
                plan,
                all_failed,
            )
            retained_records = self._retained_records(
                parent_records_all,
                plan,
                all_failed,
            )
            suffix_records = tuple(
                record
                for record in incoming_records
                if record.symbol not in all_failed
                and _record_in_suffix(record, suffix_start, requested_range.end)
            )

            seeds = self._causal_seeds(
                resolved_parent,
                parent_records_all,
                parent_rows_all,
                boundary,
            )
            normalized_suffix = tuple(
                self._normalize_suffix(suffix_records, seeds)
            )
            suffix_candidates = tuple(
                value
                for value in normalized_suffix
                if isinstance(value, (DailyBarCandidate, QuarantineRecord))
                and (
                    not isinstance(value, DailyBarCandidate)
                    or _date_in_suffix(value.session, suffix_start, requested_range.end)
                )
            )
            candidates = tuple(
                (*retained_rows, *retained_quarantine, *suffix_candidates)
            )

            benchmark = resolved_parent.manifest.content_identity.benchmark_symbol
            validator = self.validator or ValidationService(
                calendar=self.calendar,
                benchmark_symbol=benchmark,
            )
            validation = validator.validate(
                candidates,
                expected,
                staleness_threshold,
                requested_range=requested_range,
                benchmark_symbol=benchmark,
                failed_symbols=all_failed,
                retained_parent_coverage=retained_symbols,
                calendar=self.calendar,
            )
            merged_records = _unique_records(
                (*retained_records, *suffix_records)
            )
            new_rows = tuple(
                row
                for row in validation.accepted_rows
                if boundary is not None
                and row.symbol not in all_failed
                and row.session >= boundary
            )
            disclosure = LimitationDisclosure.current(data_failures=failure_errors)
            unchanged = self._is_scientifically_unchanged(
                resolved_parent,
                requested_range,
                validation,
                all_failed,
                retained_symbols,
            )
            if unchanged:
                snapshot_id = resolved_parent.snapshot_id
                content_identity: Mapping[str, object] = (
                    resolved_parent.manifest.to_content_identity_dict()
                )
                manifest = resolved_parent.manifest
            else:
                content_identity = self._new_content_identity(
                    resolved_parent,
                    requested_range,
                    validation,
                    merged_records,
                    all_failed,
                    retained_symbols,
                    disclosure,
                )
                snapshot_id = "snap_" + sha256_canonical_json(content_identity)
                manifest = None

            reused_objects, rebuilt = self._partition_plan(
                resolved_parent.object_references,
                plan,
                all_failed,
                validation,
                merged_records,
            )
            usable = bool(validation.accepted_rows)
            status = (
                "failed"
                if not usable and not parent_rows_all
                else (
                    "partially_succeeded"
                    if (
                        all_failed
                        or validation.quarantined_rows
                        or validation.gaps
                        or validation.report.summary.stale_symbols
                    )
                    else "succeeded"
                )
            )
            return Ok(
                IncrementalMergeResult(
                    parent_snapshot_id=resolved_parent.snapshot_id,
                    snapshot_id=snapshot_id,
                    plan=plan,
                    status=status,
                    accepted_rows=validation.accepted_rows,
                    quarantined_rows=validation.quarantined_rows,
                    gaps=validation.gaps,
                    validation=validation,
                    provider_records=merged_records,
                    failed_symbols=all_failed,
                    retained_parent_coverage_symbols=retained_symbols,
                    failure_errors=failure_errors,
                    limitation_disclosure=disclosure,
                    reused_object_references=reused_objects,
                    rebuilt_partition_keys=rebuilt,
                    new_rows=new_rows,
                    content_identity=content_identity,
                    manifest=manifest,
                )
            )
        except _IncrementalFailure as failure:
            return Err(failure.errors, preserve_order=True)
        except (TypeError, ValueError) as error:
            return Err((self._input_error(error),), preserve_order=True)
        except Exception as error:
            # Application boundary must not leak storage/provider exceptions.
            return Err(
                (
                    ActionableError.from_unexpected_exception(
                        "incremental.merge", error
                    ),
                )
            )

    def merge_or_raise(
        self,
        parent: IncrementalParent | SnapshotManifest | str | object,
        requested_range: DateRange,
        revision_overlap: int = 5,
        **kwargs: object,
    ) -> IncrementalMergeResult:
        """Convenience form for callers that want exceptions at a unit boundary."""

        result = self.merge(
            parent,
            requested_range,
            revision_overlap,
            **cast(dict[str, Any], kwargs),
        )
        if isinstance(result, Err):
            raise ValueError("; ".join(error.message for error in result.errors))
        return result.value

    # Common service naming aliases.
    update = merge
    merge_incremental = merge

    def _coerce_parent(
        self, parent: IncrementalParent | SnapshotManifest | object
    ) -> IncrementalParent:
        if isinstance(parent, IncrementalParent):
            return parent
        if isinstance(parent, SnapshotManifest):
            return IncrementalParent.from_manifest(parent)
        manifest = getattr(parent, "manifest", None)
        if isinstance(manifest, SnapshotManifest):
            return IncrementalParent(
                manifest=manifest,
                accepted_rows=tuple(getattr(parent, "accepted_rows", ())),
                provider_records=tuple(getattr(parent, "provider_records", ())),
                quarantined_rows=tuple(getattr(parent, "quarantined_rows", ())),
                expected_sessions=_expected_tuple(
                    getattr(parent, "expected_sessions", ())
                ),
                validation_report=getattr(parent, "validation_report", None),
            )
        raise TypeError("parent must be an IncrementalParent or SnapshotManifest")

    def _resolve_and_verify_parent(
        self,
        parent: IncrementalParent | SnapshotManifest | str | object | None,
        *,
        parent_snapshot_id: str | None,
    ) -> IncrementalParent:
        supplied_id = parent_snapshot_id
        if isinstance(parent, str):
            if supplied_id is not None:
                raise ValueError("parent snapshot ID was supplied twice")
            supplied_id = parent
            parent = None
        if parent is None:
            if supplied_id is None:
                raise ValueError("a parent snapshot or parent_snapshot_id is required")
            if self.snapshot_manager is None:
                raise _IncrementalFailure((self._parent_error(supplied_id),))
            parent = self._load_verified_parent(supplied_id)
        resolved = self._coerce_parent(parent)
        if supplied_id is not None and resolved.snapshot_id != supplied_id:
            raise _IncrementalFailure((self._parent_error(supplied_id),))
        if self.snapshot_manager is not None:
            self._verify_with_manager(resolved.snapshot_id)
        if self.parent_loader is not None and not (
            resolved.accepted_rows or resolved.provider_records
        ):
            if self.snapshot_manager is None:
                raise _IncrementalFailure((self._parent_error(resolved.snapshot_id),))
            opened = self.snapshot_manager.open_verified(resolved.snapshot_id)
            handle = _unwrap_handle(opened)
            loaded = self.parent_loader(handle, resolved.manifest)
            resolved = _merge_loaded_parent(resolved, loaded)
        return resolved

    def _load_verified_parent(self, snapshot_id: str) -> IncrementalParent:
        assert self.snapshot_manager is not None
        opened = self.snapshot_manager.open_verified(snapshot_id)
        handle = _unwrap_handle(opened)
        manifest = getattr(handle, "manifest", None)
        if not isinstance(manifest, SnapshotManifest):
            inspector = getattr(self.snapshot_manager, "inspect_snapshot", None)
            if callable(inspector):
                inspected = inspector(snapshot_id)
                if isinstance(inspected, Ok):
                    manifest = getattr(inspected.value, "manifest", None)
        if not isinstance(manifest, SnapshotManifest):
            raise _IncrementalFailure((self._parent_error(snapshot_id),))
        return IncrementalParent.from_manifest(manifest)

    def _verify_with_manager(self, snapshot_id: str) -> None:
        assert self.snapshot_manager is not None
        try:
            _unwrap_handle(self.snapshot_manager.open_verified(snapshot_id))
        except _IncrementalFailure:
            raise
        except Exception:
            raise _IncrementalFailure((self._parent_error(snapshot_id),)) from None

    def _calendar_sessions(self, start: date, end: date) -> tuple[date, ...]:
        if start > end:
            return ()
        method = getattr(self.calendar, "sessions", None)
        if callable(method):
            try:
                result = method(start, end, completed_at=CompletedAt)
            except TypeError:
                result = method(start, end)
            sessions = tuple(result)
            return tuple(
                sorted(
                    {
                        session
                        for session in sessions
                        if isinstance(session, date)
                        and not isinstance(session, datetime)
                        and start <= session <= end
                    }
                )
            )
        is_session = getattr(self.calendar, "is_session", None)
        if not callable(is_session):
            raise TypeError("calendar must expose sessions() or is_session()")
        values: list[date] = []
        current = start
        while current <= end:
            if is_session(current):
                values.append(current)
            current += timedelta(days=1)
        return tuple(values)

    def _expected_sessions(
        self,
        parent: IncrementalParent,
        requested_range: DateRange,
        supplied: Mapping[str, Sequence[date]] | None,
        symbols: Sequence[str],
    ) -> dict[str, tuple[date, ...]]:
        parent_expected = parent.expected_map()
        supplied_expected = (
            {}
            if supplied is None
            else {
                normalize_symbol(symbol): tuple(sorted(set(sessions)))
                for symbol, sessions in supplied.items()
            }
        )
        result: dict[str, tuple[date, ...]] = {}
        calendar_sessions = self._calendar_sessions(
            requested_range.start, requested_range.end
        )
        for symbol in symbols:
            if symbol in supplied_expected:
                values = supplied_expected[symbol]
            elif symbol in parent_expected:
                values = tuple(
                    session
                    for session in parent_expected[symbol]
                    if requested_range.start <= session <= requested_range.end
                )
                if requested_range.end > parent.requested_range.end:
                    later = tuple(
                        session
                        for session in calendar_sessions
                        if session > parent.requested_range.end
                    )
                    values = tuple(sorted(set((*values, *later))))
            else:
                values = calendar_sessions
            result[symbol] = tuple(sorted(set(values)))
        return result

    def _causal_seeds(
        self,
        parent: IncrementalParent,
        parent_records: Sequence[ProviderRecord],
        parent_rows: Sequence[DailyBarCandidate],
        boundary: date | None,
    ) -> dict[str, NormalizationSeed]:
        if boundary is None:
            return {}
        seeds: dict[str, NormalizationSeed] = {}
        if parent_records:
            seeds.update(
                self.normalizer.seed_states(
                    parent_records,
                    self.calendar,
                    before_session=boundary,
                    policy=self.policy,
                )
            )
        for symbol in parent.configured_symbols:
            if symbol in seeds:
                continue
            prior = [
                row
                for row in parent_rows
                if row.symbol == symbol and row.session < boundary
            ]
            if not prior:
                continue
            latest = max(prior, key=DailyBarCandidate.sort_key)
            seeds[symbol] = NormalizationSeed(
                prior_raw_close=latest.raw_close,
                cumulative_price_factor=latest.cumulative_price_factor or 1,
                cumulative_split_factor=latest.cumulative_split_factor or 1,
            )
        return seeds

    def _normalize_suffix(
        self,
        records: Sequence[ProviderRecord],
        seeds: Mapping[str, NormalizationSeed],
    ) -> tuple[DailyBarCandidate | QuarantineRecord, ...]:
        if not records:
            return ()
        seeded = getattr(self.normalizer, "normalize_seeded", None)
        if callable(seeded):
            return tuple(seeded(records, self.calendar, seeds, self.policy))
        # A custom normalizer may implement only the original protocol.  Do
        # not silently restart causal factors when seeds are non-empty.
        if seeds:
            raise TypeError("incremental normalizer must support causal seeds")
        return tuple(self.normalizer.normalize(records, self.calendar, self.policy))

    @staticmethod
    def _retained_rows(
        rows: Sequence[DailyBarCandidate],
        plan: IncrementalPlan,
        failed: Sequence[str],
    ) -> tuple[DailyBarCandidate, ...]:
        if not plan.has_provider_suffix:
            return tuple(rows)
        failed_set = set(failed)
        boundary = plan.boundary_session
        assert boundary is not None
        return tuple(
            row
            for row in rows
            if row.symbol in failed_set or row.session < boundary
        )

    @staticmethod
    def _retained_quarantines(
        rows: Sequence[QuarantineRecord],
        plan: IncrementalPlan,
        failed: Sequence[str],
    ) -> tuple[QuarantineRecord, ...]:
        if not plan.has_provider_suffix:
            return tuple(rows)
        failed_set = set(failed)
        boundary = plan.boundary_session
        assert boundary is not None
        return tuple(
            row
            for row in rows
            if row.symbol in failed_set
            or row.session is None
            or row.session < boundary
        )

    @staticmethod
    def _retained_records(
        rows: Sequence[ProviderRecord],
        plan: IncrementalPlan,
        failed: Sequence[str],
    ) -> tuple[ProviderRecord, ...]:
        if not plan.has_provider_suffix:
            return tuple(rows)
        failed_set = set(failed)
        suffix_start = _suffix_start(plan)
        return tuple(
            row
            for row in rows
            if row.symbol in failed_set or row.provider_date < suffix_start
        )

    @staticmethod
    def _is_scientifically_unchanged(
        parent: IncrementalParent,
        requested_range: DateRange,
        validation: ValidationOutput,
        failed: Sequence[str],
        retained: Sequence[str],
    ) -> bool:
        if requested_range != parent.requested_range or failed or retained:
            return False
        if _row_content(validation.accepted_rows) != _row_content(parent.accepted_rows):
            return False
        if parent.quarantined_rows and _quarantine_content(
            validation.quarantined_rows
        ) != _quarantine_content(parent.quarantined_rows):
            return False
        if parent.validation_report is not None:
            return (
                validation.report.to_content_dict()
                == parent.validation_report.to_content_dict()
            )
        return (
            validation.report.summary
            == parent.manifest.content_identity.validation_summary
        )

    @staticmethod
    def _new_content_identity(
        parent: IncrementalParent,
        requested_range: DateRange,
        validation: ValidationOutput,
        merged_records: Sequence[ProviderRecord],
        failed: Sequence[str],
        retained: Sequence[str],
        disclosure: LimitationDisclosure,
    ) -> Mapping[str, object]:
        base = parent.manifest.content_identity
        payload: dict[str, object] = {
            "identity_schema": "incremental_snapshot_identity_v1",
            "provider": base.provider,
            "requested_range": requested_range.to_content_dict(),
            "configured_universe": list(base.configured_universe),
            "benchmark_symbol": base.benchmark_symbol,
            "calendar": base.calendar.to_content_dict(),
            "configuration_checksum": base.configuration_checksum,
            "schema_versions": base.schema_versions.to_content_dict(),
            "accepted_rows": [
                row.to_content_dict() for row in validation.accepted_rows
            ],
            "quarantined_rows": [
                row.to_content_dict() for row in validation.quarantined_rows
            ],
            "gaps": [gap.to_content_dict() for gap in validation.gaps],
            "validation_report": validation.report.to_content_dict(),
            "failed_symbols": list(_normalized_symbols(failed)),
            "retained_parent_coverage_symbols": list(_normalized_symbols(retained)),
            "limitation_disclosure": {
                "version": disclosure.version,
                "lines": list(disclosure.lines()),
            },
            # Raw rows that do not become accepted candidates (for example a
            # non-session row) remain scientific provenance through the merged
            # canonical record checksum, while request IDs/timestamps do not.
            "provider_record_checksums": sorted(
                record.provider_record_checksum for record in merged_records
            ),
        }
        return MappingProxyType(payload)

    @staticmethod
    def _partition_plan(
        references: Sequence[ContentAddressedObjectRef],
        plan: IncrementalPlan,
        failed: Sequence[str],
        validation: ValidationOutput,
        records: Sequence[ProviderRecord],
    ) -> tuple[
        tuple[ContentAddressedObjectRef, ...],
        tuple[tuple[str, str, int], ...],
    ]:
        if not plan.has_provider_suffix:
            return tuple(references), ()
        boundary = plan.boundary_session
        assert boundary is not None
        failed_set = set(failed)
        affected = {
            *(
                row.symbol
                for row in validation.accepted_rows
                if row.session >= boundary and row.symbol not in failed_set
            ),
            *(
                record.symbol
                for record in records
                if record.provider_date >= boundary and record.symbol not in failed_set
            ),
        }
        rebuilt: set[tuple[str, str, int]] = set()
        reused: list[ContentAddressedObjectRef] = []
        for reference in references:
            symbol = reference.symbol
            year = reference.session_year
            is_affected = symbol in affected and (
                year is None or year >= boundary.year
            )
            if reference.object_kind in {ObjectKind.VALIDATION, ObjectKind.QUARANTINE}:
                is_affected = True
            if is_affected:
                rebuilt.add(
                    (reference.object_kind.value, symbol or "", year or -1)
                )
            else:
                reused.append(reference)
        for row in validation.accepted_rows:
            if row.session >= boundary and row.symbol not in failed_set:
                rebuilt.add((ObjectKind.NORMALIZED.value, row.symbol, row.session.year))
        for record in records:
            if (
                record.provider_date >= boundary
                and record.symbol not in failed_set
            ):
                rebuilt.add(
                    (ObjectKind.RAW.value, record.symbol, record.provider_date.year)
                )
        return tuple(
            sorted(reused, key=ContentAddressedObjectRef.sort_key)
        ), tuple(sorted(rebuilt))

    @staticmethod
    def _input_error(error: BaseException) -> ActionableError:
        message = str(error).splitlines()[0] or "invalid incremental update input"
        return ActionableError(
            operation="incremental.merge",
            category=ErrorCategory.CONFIGURATION_INVALID_VALUE,
            message=message,
            corrective_action=(
                "Use the parent requested start and a non-decreasing requested end, "
                "then retry with a valid revision overlap."
            ),
            field_path="incremental.requested_range",
        )

    @staticmethod
    def _parent_error(snapshot_id: str) -> ActionableError:
        return ActionableError(
            operation="incremental.verify_parent",
            category=ErrorCategory.INTEGRITY_CHECKSUM,
            message=(
                "The parent Data_Snapshot could not be verified for incremental use."
            ),
            corrective_action=(
                "Restore the checksum-verified parent snapshot or select another "
                "published Data_Snapshot."
            ),
            field_path="parent_snapshot_id",
            correlation_id=snapshot_id,
        )


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _normalized_symbols(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("symbols must be an iterable, not a scalar")
    return tuple(sorted({normalize_symbol(value) for value in values}))


def _quarantine_sort_key(
    value: QuarantineRecord,
) -> tuple[str, str, str, str, str, str]:
    return value.sort_key()


def _row_content(rows: Iterable[DailyBarCandidate]) -> tuple[dict[str, object], ...]:
    return tuple(
        row.to_content_dict()
        for row in sorted(rows, key=DailyBarCandidate.sort_key)
    )


def _quarantine_content(
    rows: Iterable[QuarantineRecord],
) -> tuple[dict[str, object], ...]:
    return tuple(
        row.to_content_dict()
        for row in sorted(rows, key=_quarantine_sort_key)
    )


def _date_in_suffix(value: date, start: date | None, end: date) -> bool:
    return start is not None and start <= value <= end


def _suffix_start(plan: IncrementalPlan) -> date:
    if plan.revision_overlap and plan.boundary_session is not None:
        return plan.boundary_session
    return plan.parent_range.end + timedelta(days=1)


def _record_in_suffix(record: ProviderRecord, start: date, end: date) -> bool:
    return start <= record.provider_date <= end


def _unique_records(records: Iterable[ProviderRecord]) -> tuple[ProviderRecord, ...]:
    by_checksum: dict[str, ProviderRecord] = {}
    for record in records:
        by_checksum.setdefault(record.provider_record_checksum, record)
    return tuple(sorted(by_checksum.values(), key=ProviderRecord.sort_key))


def _expected_tuple(value: object) -> tuple[tuple[str, tuple[date, ...]], ...]:
    if isinstance(value, Mapping):
        return tuple(
            (normalize_symbol(symbol), tuple(sessions))
            for symbol, sessions in value.items()
        )
    if value is None:
        return ()
    return tuple(value)  # type: ignore[arg-type]


def _unwrap_handle(value: object) -> VerifiedSnapshotHandle:
    if isinstance(value, Err):
        raise _IncrementalFailure(value.errors)
    if isinstance(value, Ok):
        value = value.value
    if not isinstance(value, VerifiedSnapshotHandle):
        raise _IncrementalFailure(
            (
                ActionableError(
                    operation="incremental.verify_parent",
                    category=ErrorCategory.INTEGRITY_CHECKSUM,
                    message=(
                        "Parent verification did not return a verified snapshot handle."
                    ),
                    corrective_action=(
                        "Reconcile the snapshot store and retry the update."
                    ),
                    field_path="parent_snapshot_id",
                ),
            )
        )
    return value


def _materialize_input(
    records: RecordInput,
    provider_outcomes: ProviderBatchResult | Iterable[SymbolOutcome] | None,
) -> tuple[tuple[ProviderRecord, ...], tuple[ActionableError, ...], tuple[str, ...]]:
    all_records: list[ProviderRecord] = []
    failures: list[ActionableError] = []
    failed_symbols: list[str] = []

    def consume(value: object) -> None:
        if isinstance(value, ProviderRecord):
            all_records.append(value)
            return
        if isinstance(value, SymbolOutcome):
            if value.status is SymbolOutcomeStatus.SUCCESS:
                all_records.extend(value.records)
            else:
                failed_symbols.append(value.symbol)
                failures.extend(value.errors)
            return
        if isinstance(value, ProviderBatchResult):
            for outcome in value.outcomes:
                consume(outcome)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                consume(item)
            return
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError("provider records must not be supplied as text")
        try:
            for item in cast(Iterable[object], value):
                consume(item)
        except TypeError as error:
            raise TypeError(
                "records must contain ProviderRecord or SymbolOutcome values"
            ) from error

    consume(records)
    if provider_outcomes is not None:
        consume(provider_outcomes)
    return (
        _unique_records(all_records),
        tuple(failures),
        _normalized_symbols(failed_symbols),
    )


def _merge_errors(
    errors: Iterable[ActionableError], failed_symbols: Sequence[str]
) -> tuple[ActionableError, ...]:
    by_symbol = {error.symbol for error in errors if error.symbol}
    result = list(errors)
    for symbol in failed_symbols:
        if symbol in by_symbol:
            continue
        result.append(
            ActionableError(
                operation="incremental.provider",
                category=ErrorCategory.PROVIDER_TERMINAL,
                message=f"No usable provider content was returned for {symbol}.",
                corrective_action=(
                    "Retry the symbol with the configured provider policy or inspect "
                    "the retained parent coverage."
                ),
                symbol=symbol,
            )
        )
    return tuple(sorted(result, key=ActionableError.sort_key))


def _merge_loaded_parent(
    original: IncrementalParent,
    loaded: IncrementalParent | Mapping[str, object],
) -> IncrementalParent:
    if isinstance(loaded, IncrementalParent):
        if loaded.snapshot_id != original.snapshot_id:
            raise _IncrementalFailure(
                (
                    ActionableError(
                        operation="incremental.verify_parent",
                        category=ErrorCategory.INTEGRITY_CHECKSUM,
                        message=(
                            "Loaded parent content does not match its verified "
                            "Snapshot_ID."
                        ),
                        corrective_action=(
                            "Reload content through the verified parent handle."
                        ),
                        field_path="parent_snapshot_id",
                        correlation_id=original.snapshot_id,
                    ),
                )
            )
        return loaded
    if not isinstance(loaded, Mapping):
        raise TypeError("parent_loader must return IncrementalParent or a mapping")
    return replace(
        original,
        accepted_rows=tuple(
            cast(
                Iterable[DailyBarCandidate],
                loaded.get("accepted_rows", original.accepted_rows),
            )
        ),
        provider_records=tuple(
            cast(
                Iterable[ProviderRecord],
                loaded.get("provider_records", original.provider_records),
            )
        ),
        quarantined_rows=tuple(
            cast(
                Iterable[QuarantineRecord],
                loaded.get("quarantined_rows", original.quarantined_rows),
            )
        ),
        expected_sessions=_expected_tuple(
            loaded.get("expected_sessions", original.expected_sessions)
        ),
        validation_report=cast(
            ValidationReport | None,
            loaded.get("validation_report", original.validation_report),
        ),
    )


def merge_incremental(
    parent: IncrementalParent | SnapshotManifest | str | object,
    requested_range: DateRange,
    *,
    calendar: object,
    revision_overlap: int = 5,
    **kwargs: object,
) -> Result[IncrementalMergeResult]:
    """Functional facade for one revision-overlap update."""

    return IncrementalMerger(calendar).merge(
        parent,
        requested_range,
        revision_overlap,
        **cast(dict[str, Any], kwargs),
    )


IncrementalUpdateService = IncrementalMerger
IncrementalSnapshotMerger = IncrementalMerger
plan_incremental_update = IncrementalMerger.plan


__all__ = [
    "IncrementalMerger",
    "IncrementalParent",
    "IncrementalPlan",
    "IncrementalMergeResult",
    "IncrementalResult",
    "IncrementalSnapshotMerger",
    "IncrementalUpdateRequest",
    "IncrementalUpdateResult",
    "IncrementalUpdateService",
    "ParentSnapshot",
    "SnapshotContentLoader",
    "SnapshotVerifier",
    "merge_incremental",
    "plan_incremental_update",
]
