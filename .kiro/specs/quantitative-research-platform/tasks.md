# Implementation Plan: Quantitative Research Platform — Phase 1 Vertical Slice

## Overview

This plan implements only the approved local Phase 1 vertical slice: validated configuration, yfinance daily US-equity ingestion, immutable Parquet/DuckDB snapshots, one Zipline Reloaded monthly momentum baseline, next-session whole-share accounting, SPY evaluation, immutable local MLflow runs, run comparison and inspection, and a Streamlit workflow. Tasks are incremental prompts for a code-generation agent; each implementation step is wired into the preceding slice and receives focused automated validation before the next boundary is added.

The plan intentionally excludes FastAPI, distributed/background queues, cloud infrastructure, LLMs, machine learning, additional strategy/factor libraries, Alphalens/Pyfolio, broker adapters, paper trading, live execution, and all later-roadmap capabilities.

## Tasks

- [ ] 1. Establish the Prompt Zero Python foundation before engine logic
  - [ ] 1.1 Record the Phase 1 technology decision in `docs/decisions/0001-phase1-local-stack.md`
    - Document the approved modular-monolith boundaries and the concrete roles of Python 3.11, Pydantic v2, ruamel.yaml, yfinance, exchange-calendars XNYS, PyArrow/Parquet, DuckDB, Zipline Reloaded, local MLflow/SQLite, Streamlit, Pytest, and Hypothesis.
    - Record why platform-owned Parquet/CAS data remains authoritative, why Zipline bundles are derived caches, and why FastAPI, queues, cloud services, LLM/ML capabilities, Alphalens/Pyfolio, and execution adapters are excluded from this slice.
    - Focused validation: review the decision against the constitution priority order and confirm every selected package has a current Phase 1 responsibility before any engine module is implemented.
    - _Requirements: 1.1, 1.6–1.11, 15.1–15.6_

  - [ ] 1.2 Create `pyproject.toml` and the reviewed `uv.lock` workflow for Python 3.11
    - Define installable package/build metadata, `requires-python = ">=3.11,<3.12"`, runtime dependencies, and development groups for Pytest, Hypothesis, Ruff, mypy, and coverage; pin the complete compatible environment in `uv.lock` and require frozen synchronization for reproducible setup.
    - Configure Ruff, mypy, Pytest markers (`integration`, `external`, `memory`, `smoke`), coverage, and Hypothesis profiles in project metadata without adding excluded dependencies.
    - Focused validation: add lock-consistency and metadata assertions that reject an unsupported Python range, missing dependency group, unresolved lock, or excluded package.
    - _Requirements: 1.1–1.7, 17.1, 17.29–17.31_

  - [ ] 1.3 Create the Phase 1 source, test, configuration, and generated-data layout
    - Create `src/quant_research_platform/{domain,config,application,infrastructure,ui}`, `tests/{unit,properties,contract,integration,golden,smoke}`, `config/default.yaml`, and ignored local `data/` roots with minimal package markers only; do not scaffold future execution, ML, LLM, API, or queue modules.
    - Update `.gitignore` for virtual environments, Python/tool caches, coverage/build output, local secret-file patterns, DuckDB/MLflow databases, staging/CAS data, artifacts, and derived Zipline bundles while keeping reviewed golden fixtures trackable.
    - Create initial `README.md` setup guidance for Python 3.11, frozen dependency sync, local paths, test/quality commands, secret handling, and the Phase 1 scope boundary.
    - Focused validation: assert ignored secret/generated-data samples are excluded while source, default configuration, and golden fixtures remain visible to source control.
    - _Requirements: 1.1–1.5, 1.10–1.11, 16.1–16.3_

  - [ ]* 1.4 Add project-boundary, build, import, and architecture smoke tests
    - Add `tests/smoke/test_project_setup.py` for package build/import, Python range, dependency groups, default configuration presence, and ignore rules.
    - Add `tests/unit/test_import_boundaries.py` to enforce `presentation -> application -> domain`, prevent domain imports of infrastructure frameworks, and ensure the application layer does not read Streamlit state.
    - Focused validation: run single-shot build/import, Ruff, mypy, and targeted Pytest commands under Python 3.11 without starting Streamlit or any watcher.
    - _Requirements: 1.1–1.5, 1.8–1.10, 17.1_

- [ ] 2. Checkpoint — Verify the Prompt Zero foundation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Implement canonical domain primitives and scientific identity rules
  - [ ] 3.1 Implement errors, results, disclosures, and stable reason taxonomies in `domain/errors.py`
    - Define immutable `ActionableError`, `Ok`/`Err` result types, validation/quarantine/provider/job reason enums, and the versioned `LimitationDisclosure` required by data, snapshot, run, and comparison DTOs.
    - Require one corrective action and structured optional field/symbol/session/checksum/correlation context while preventing raw exceptions from crossing application boundaries.
    - Focused validation: unit-test deterministic ordering, required fields, disclosure content/versioning, and sanitized formatting for representative configuration, provider, integrity, and backtest failures.
    - _Requirements: 3.19, 10.18, 11.20, 12.14, 14.6–14.8_

  - [ ] 3.2 Implement canonical encoding and checksum helpers in `domain/canonical.py`
    - Implement UTF-8/LF canonical JSON primitives, Unicode NFC normalization, canonical dates/UTC timestamps/decimals/rationals, SHA-256 helpers, content projections, and checksum-verified byte wrappers.
    - Keep volatile timestamps, IDs, local paths, and operational lineage out of scientific content identity through explicit projection functions rather than caller convention.
    - Focused validation: unit-test byte stability, one-terminal-LF behavior, key ordering, non-finite-number rejection, and checksum changes for scientific fields but not excluded operational metadata.
    - _Requirements: 6.5–6.9, 6.17, 10.16–10.17, 11.11–11.14_

  - [ ] 3.3 Implement market, manifest, and storage-reference value objects in `domain/market.py` and `domain/manifests.py`
    - Define immutable provider requests/records/outcomes, `SessionKey`, raw and normalized bar/action models, quarantine/gap/report models, content-addressed object references, snapshot content identity, operational metadata, and verified snapshot handles.
    - Encode raw lineage, schema/policy/calendar versions, row counts, validation summaries, failed-symbol/retained-coverage facts, and limitation disclosures without infrastructure imports.
    - Focused validation: construct representative valid/invalid objects and verify immutability, canonical sort keys, lineage requirements, and separation of scientific and operational fields.
    - _Requirements: 3.12–3.19, 4.1–4.15, 5.7–5.18, 6.4–6.9_

  - [ ] 3.4 Implement strategy, execution, evaluation, run, and job value objects in `domain/strategy.py`, `domain/execution.py`, and `domain/evaluation.py`
    - Define exact `RationalWeight`, `StrategyDecision`, deterministic order/fill IDs, orders, fills, positions, portfolio states, core outputs, metrics/null reasons, environment/run manifests, comparison records, progress updates, and legal job/run state enums.
    - Represent monetary calculations and basis points with `Decimal`, whole-share quantities with integers, and scientific IDs independently from operational run/job IDs.
    - Focused validation: unit-test rational equality/sums, order-ID determinism, illegal state/value rejection, monetary quantization, and serialization of null metric reasons.
    - _Requirements: 8.12–8.15, 9.6–9.18, 10.6–10.15, 11.3–11.16, 14.1–14.5_

  - [ ]* 3.5 Add domain primitive and canonical-manifest unit tests
    - Add focused tests under `tests/unit/domain/` for canonical encodings, checksums, disclosures, manifest projections, result types, immutable records, exact weights, deterministic IDs, and legal job/run transitions.
    - Focused validation: use reviewed examples showing operational time/path changes preserve identity while scientific row/checksum/configuration changes do not.
    - _Requirements: 5.21–5.23, 6.7–6.9, 8.15, 11.12–11.16, 14.1–14.5_

- [ ] 4. Implement validated configuration, project-root resolution, and redaction
  - [ ] 4.1 Implement frozen Pydantic v2 configuration models in `config/models.py`
    - Define every approved path, requested range, universe/benchmark, retry, batch, staleness, overlap, write chunk, strategy, cost, page-size, seed, and proxy-secret field with documented defaults and inclusive bounds.
    - Normalize universe symbols while preserving distinct order; derive default position count; reject empty/duplicate symbols, invalid date order, non-finite costs, invalid retry relationships, and extras at every nesting level.
    - Focused validation: unit-test all documented defaults and boundary values, including the fixed USD 100,000 initial equity and `SPY` benchmark.
    - _Requirements: 2.1–2.2, 2.19–2.53_

  - [ ] 4.2 Implement safe loading, precedence, and path boundaries in `config/loader.py` and `config/project_root.py`
    - Use ruamel.yaml safe mode with duplicate keys/tags disabled; return parser-location, root-type, duplicate-path, and allowed-sibling diagnostics; merge defaults, effective YAML, and only explicitly mapped `QRP_` environment leaves.
    - Resolve exactly one project `pyproject.toml` boundary; normalize relative paths against it, prevent escapes before creating directories, preserve absolute paths, and return all schema errors in field/list-index order before invoking downstream services.
    - Focused validation: use dependency spies to prove invalid configuration triggers no operation and representative nested-source examples to prove leaf-wise precedence.
    - _Requirements: 2.3–2.18, 2.54–2.58, 16.1, 16.10_

  - [ ] 4.3 Implement canonical non-secret serialization and the reusable redactor in `config/serializer.py`
    - Emit every non-secret field in schema order using canonical scalars, UTF-8, LF-only output, and exactly one final LF; emit `[REDACTED]` as the whole value for present secrets and reload it as unresolved.
    - Register literal and URL-encoded secret forms; provide idempotent sanitization for text, structured metadata, URLs, headers, errors, progress, logs, MLflow fields, manifests, and presenter DTOs.
    - Focused validation: test equivalent configurations produce identical bytes, resolved secrets never cross the non-secret boundary, and unresolved required secrets block only operations that need them.
    - _Requirements: 2.59–2.72, 16.4–16.10_

  - [ ]* 4.4 Write the Hypothesis test for Property 1 in `tests/properties/test_property_01_configuration_resolution.py`
    - Generate safe defaults/YAML/mapped-environment leaf maps, unknown/unmapped names, invalid types/bounds, sibling overrides, and downstream-call spies; compare resolution to a simple right-biased reference merge plus reference normalizers.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 1: Leaf-wise configuration resolution and validation gate`.
    - Focused validation: require at least 100 examples and assert complete schema-ordered errors and zero downstream calls on every invalid result.
    - **Property 1: Leaf-wise configuration resolution and validation gate**
    - **Validates: Requirements 2.6, 2.7, 2.9–2.24, 2.26–2.31, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50, 2.52, 2.56–2.58**

  - [ ]* 4.5 Write the Hypothesis test for Property 2 in `tests/properties/test_property_02_configuration_round_trip.py`
    - Generate valid resolved configurations and non-empty secrets; serialize repeatedly, parse under the same project root without environment overrides, and compare equivalent non-secret projections and exact bytes.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 2: Canonical redacted configuration round trip`.
    - Focused validation: assert UTF-8/LF/schema order, one final LF, unresolved redacted secrets, and absence of literal and encoded secret forms for at least 100 examples.
    - **Property 2: Canonical redacted configuration round trip**
    - **Validates: Requirements 2.59–2.72, 16.5, 16.8–16.10, 17.2**

  - [ ]* 4.6 Add configuration parser, precedence, bounds, path, and redaction example tests
    - Add `tests/unit/config/` cases for malformed YAML locations, unsafe tags, non-mapping roots, nested duplicate/unknown keys, all defaults, field-order errors, ambiguous roots, path escapes, explicit absolute paths, mapped/unmapped variables, and marker reload behavior.
    - Focused validation: use temporary project roots and synthetic environments only; ensure error messages contain actionable field/key paths and never contain secret bytes.
    - _Requirements: 2.3–2.18, 2.25, 2.32–2.58, 17.3_

- [ ] 5. Implement XNYS sessions and the narrow yfinance provider boundary
  - [ ] 5.1 Implement `ExchangeCalendar` and the pinned XNYS adapter in `infrastructure/xnys_calendar.py`
    - Expose completed sessions, session membership, next session, month ends, official UTC close times, package version, and a canonical schedule digest; stop expected sessions at the latest officially completed close.
    - Focused validation: verify known weekends, holidays, daylight-saving transitions, shortened sessions, month ends, and future/incomplete-session warnings against reviewed fixtures.
    - _Requirements: 4.1–4.4, 5.2, 5.14–5.17, 9.16_

  - [ ] 5.2 Define `MarketDataProvider`, request/result contracts, batching, and retry policy helpers in `application/ports.py`
    - Require one independent `SymbolOutcome` per normalized requested symbol, deterministic ordered batches of 1–10 symbols including SPY exactly once, bounded attempts, capped no-jitter delays, and retryable/terminal failure classes.
    - Focused validation: exercise fake clock/provider outcomes to verify order preservation, exact delay sequences, attempt caps, terminal stop, and per-symbol partial-success isolation.
    - _Requirements: 3.1–3.11, 14.9, 15.1_

  - [ ] 5.3 Implement the yfinance adapter in `infrastructure/yfinance_provider.py`
    - Translate inclusive end dates to yfinance's exclusive end; set the exact approved daily download options; normalize single/multi-symbol column shapes; preserve available raw fields/actions/metadata and request provenance; classify transport/status/schema/empty outcomes.
    - Keep yfinance imports and exceptions inside the adapter, disable provider threading, sanitize diagnostics, and never let properties or normal tests perform external calls.
    - Focused validation: replay local DataFrame/exception fixtures for successful, partial, empty, schema-invalid, retryable, and terminal responses and assert the exact call arguments.
    - _Requirements: 3.2–3.18, 14.9_

  - [ ]* 5.4 Write the Hypothesis test for Property 3 in `tests/properties/test_property_03_provider_orchestration.py`
    - Generate ordered universes, batch sizes, retry policies, and per-attempt symbol outcomes using a fake provider/clock; compare batching, delay, attempts, final symbol outcomes, and partial-success status to a reference model.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 3: Bounded provider batching, retry, and symbol isolation`.
    - Focused validation: require at least 100 examples, no external I/O, no batch above 10, no attempts beyond policy, and no repeated terminal attempt.
    - **Property 3: Bounded provider batching, retry, and symbol isolation**
    - **Validates: Requirements 3.1, 3.4–3.11, 3.15–3.18, 14.9, 15.1, 17.16–17.17**

  - [ ]* 5.5 Add yfinance and XNYS contract tests with local golden fixtures
    - Add `tests/contract/test_yfinance_contract.py` for call options, frame shapes, actions, partial symbols, and exception mapping, plus `tests/contract/test_xnys_calendar_contract.py` for schedule/version/digest semantics.
    - Focused validation: keep ordinary contract tests offline; mark one separately gated one-batch request as `external` for later smoke validation.
    - _Requirements: 3.2–3.18, 4.1–4.4, 9.16, 17.16–17.17, 17.30–17.31_

- [ ] 6. Implement deterministic Parquet, DuckDB metadata, content-addressed storage, and jobs
  - [ ] 6.1 Define explicit Arrow schemas and canonical table conversion in `infrastructure/schemas.py`
    - Implement `raw_v1`, `daily_bar_v1`, quarantine, gap, validation report, decisions, orders, fills, positions, portfolio, returns, metrics, and monthly-return schemas with stable enum/string/decimal/timestamp rules and no pandas metadata.
    - Focused validation: round-trip representative records and reject missing non-null fields, non-UTC event timestamps, incompatible versions, and non-canonical enum/decimal values.
    - _Requirements: 3.12–3.17, 4.3–4.13, 5.7–5.18, 10.12–10.15_

  - [ ] 6.2 Implement bounded deterministic Parquet writes and projected scans in `infrastructure/parquet_store.py`
    - Write raw and normalized collections separately; partition normalized rows by symbol/year; externally sort logical partitions; emit fixed `write_chunk_size` slices with pinned Parquet settings and SHA-256 over final bytes.
    - Expose `RecordBatchReader` scans with required columns, symbols, years, sessions, predicates, and no unconditional `read_all()`/`to_pandas()` path.
    - Focused validation: writer/scanner spies assert chunk, row-group, column, predicate, and ordering bounds; repeated canonical input produces byte-identical files under the locked environment.
    - _Requirements: 6.1–6.2, 6.5, 7.8, 15.2–15.5_

  - [ ] 6.3 Implement the DuckDB schema and repositories in `infrastructure/duckdb_metadata.py`
    - Create transactional repositories for ingestion/provider outcomes, objects, snapshots/status, artifacts, runs/metrics/finalization intents, jobs/events, and indexes matching the design; keep scientific tables in Parquet/CAS rather than duplicating them.
    - Enforce insert-only snapshot science, legal run/job transitions, terminal guards, deterministic filtering/order, and operational availability/invalidity updates.
    - Focused validation: run temporary-database migration/reopen tests, transaction rollback tests, illegal-transition tests, and query projection/order assertions.
    - _Requirements: 6.3, 11.3–11.18, 12.1, 14.13_

  - [ ] 6.4 Implement staging, CAS artifacts, integrity checks, and the publisher lock in `infrastructure/filesystem_store.py`
    - Provide same-filesystem root validation, exclusive staged writes, file/directory fsync, content-address promotion/reuse after byte verification, verified artifact streaming, final metadata secret scanning, and one-writer advisory locking.
    - Keep staging and unreferenced CAS objects invisible to readers; reject cross-device publication and conflicting bytes at an existing checksum.
    - Focused validation: use temporary real filesystems to inject short writes, checksum mismatches, cross-device/root errors, existing-object reuse, secret metadata, and lock contention.
    - _Requirements: 6.5–6.6, 6.10–6.13, 7.1–7.4, 7.11, 16.6_

  - [ ] 6.5 Implement synchronous jobs and sanitized structured logging in `application/jobs.py` and `infrastructure/logging.py`
    - Enforce `not_started -> running -> succeeded|partially_succeeded|failed`, ingestion-only partial success, persisted progress/events, four-writes-per-second throttling with immediate terminal writes, elapsed time, totals, and accumulated sanitized warnings.
    - Convert boundary exceptions into `ActionableError`, persist sanitized JSONL diagnostics with correlation IDs, and retain prior valid results after later failures.
    - Focused validation: fake-clock tests cover every legal/illegal transition, throttling, terminal flush, redaction, symbol isolation, and repository persistence/reload.
    - _Requirements: 13.4–13.5, 13.9, 13.11, 14.1–14.13_

  - [ ]* 6.6 Add DuckDB/PyArrow/filesystem/job contract tests
    - Add contract suites for schema fidelity, deterministic Parquet settings/checksums, filter/projection pushdown, repository guards, CAS verification, fsync/rename seams, publisher locking, and progress persistence.
    - Focused validation: use temporary roots/databases and bounded fixtures; prove no whole-dataset materialization is required for chunked writes or filtered reads.
    - _Requirements: 6.1–6.3, 6.5–6.6, 14.1–14.13, 15.2–15.6_

- [ ] 7. Implement causal normalization, corporate actions, validation, quarantine, gaps, and staleness
  - [ ] 7.1 Implement `CausalForwardAdjustmentV1` and `Normalizer` in `domain/normalization.py`
    - Sort records by `(symbol, session, provider_record_checksum)`, map only XNYS sessions, preserve raw/provider fields, represent XNYS close in UTC, and compute causal adjusted OHLCV with Decimal precision and declared split/dividend/volume/rounding treatment.
    - Emit no candidate for wholly absent observations; emit deterministic quarantine records for non-sessions and invalid action equations; preserve actions and `Adj Close` only as provenance/diagnostics.
    - Focused validation: unit examples cover no action, split, dividend, same-session split/dividend, invalid ratios/equations, absent/partial rows, and later-action causality.
    - _Requirements: 4.1–4.18_

  - [ ]* 7.2 Write the Hypothesis test for Property 4 in `tests/properties/test_property_04_normalization_confluence.py`
    - Generate finite provider histories, valid/invalid XNYS labels, actions, record permutations, and valid batch regroupings; compare output to an independent Decimal reference implementation.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 4: Causal normalization determinism and confluence`.
    - Focused validation: require at least 100 examples and assert raw preservation, causal prefixes, absent-row behavior, sorted byte equivalence, and equal checksums across permutations.
    - **Property 4: Causal normalization determinism and confluence**
    - **Validates: Requirements 4.1–4.18, 7.10, 9.19, 17.4, 17.7**

  - [ ] 7.3 Implement streaming deterministic validation in `domain/validation.py`
    - Apply map, row, action-policy, and duplicate phases in fixed reason order; accept one equivalent duplicate, quarantine every conflicting/invalid member, accept no conflicting key, require raw lineage, and aggregate partition streams one key group at a time.
    - Compare accepted keys to expected completed XNYS sessions; record one gap without fabricating a bar, staleness lag, per-symbol failure/retained coverage, range-specific SPY readiness, and a canonical validation report.
    - Focused validation: unit cases cover every row/envelope rule, multiple ordered reasons, equivalent/conflicting duplicates, gaps, stale boundaries, report/detail count reconciliation, and deterministic repetition.
    - _Requirements: 5.1–5.23, 15.3_

  - [ ]* 7.4 Write the Hypothesis test for Property 5 in `tests/properties/test_property_05_validation_partition.py`
    - Generate candidate multisets with valid/invalid rows, equivalent/conflicting key groups, expected session sets, and lineage; compare accepted, quarantine, duplicate, gap, staleness, and report outputs to a reference partitioner.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 5: Validation partitions candidates without fabrication`.
    - Focused validation: require at least 100 examples and prove accepted-key uniqueness, all-conflict quarantine, zero conflict acceptance, deterministic reasons, exact gap reporting, and zero fabricated bars.
    - **Property 5: Validation partitions candidates without fabrication**
    - **Validates: Requirements 5.1–5.23, 17.5, 17.11–17.15**

  - [ ]* 7.5 Create reviewed action and quality golden fixtures
    - Add `tests/golden/actions/` and `tests/golden/quality_issues/` with raw input plus expected causal-adjusted values, policy errors, invalid OHLC, equivalent/conflicting duplicates, non-session dates, gaps, stale symbols, quarantine reasons, and canonical report checksums.
    - Focused validation: load fixtures through the real normalizer/validator and compare field-level canonical outputs rather than opaque screenshots.
    - _Requirements: 4.10–4.18, 5.1–5.23, 17.11–17.15_

  - [ ]* 7.6 Add normalization-to-Parquet integration tests
    - Stream raw golden records through normalization, validation, bounded Parquet writes, and filtered rereads; verify raw/normalized/quarantine/gap separation, exact lineage, partition paths, schemas, counts, and deterministic checksums.
    - Focused validation: rerun with record and batch permutations and assert identical normalized scientific objects while operational request metadata remains independently inspectable.
    - _Requirements: 3.12–3.17, 4.16–4.18, 5.21–5.23, 6.1–6.2_

- [ ] 8. Implement immutable, atomic, idempotent, and incremental snapshot ingestion
  - [ ] 8.1 Implement snapshot identity/manifest assembly in `application/snapshots.py`
    - Build canonical snapshot content identity from policy/schema/calendar/configuration/disclosure versions, ordered universe/benchmark/ranges, sorted object checksums/counts, and deterministic validation/failure/retained-coverage facts; keep timestamps, paths, jobs, and lineage operational.
    - Derive `snap_<sha256>`, reference every partition/validation artifact once, and support equivalent-content reuse independent of local root and attempted parent.
    - Focused validation: example manifests prove required fields/checksums are present and volatile metadata or relocation cannot alter Snapshot ID.
    - _Requirements: 6.4–6.9, 6.17_

  - [ ] 8.2 Implement verified snapshot open/list and immutability guards in `application/snapshots.py`
    - Resolve only complete published manifests, verify all referenced bytes before returning a handle, index/list summaries, expose validation/provenance/readiness details, and reject all platform mutation attempts with guidance to publish a new snapshot.
    - Focused validation: test valid open, missing/corrupt objects, manifest mismatch, relocation, unavailable indexing, and attempted manifest/object replacement while prior snapshots remain readable.
    - _Requirements: 6.10–6.18, 13.6–13.7_

  - [ ] 8.3 Implement atomic publication and startup reconciliation in `infrastructure/filesystem_store.py`
    - Finalize deterministic logical partitions, verify/checksum staged objects, promote CAS bytes, fsync a complete publication directory, atomically rename it to an absent snapshot ID, fsync the parent, then transactionally index DuckDB.
    - Reconcile complete unindexed publications after crashes; mark rows with absent/corrupt directories unavailable; never expose staging or partial candidates; preserve the latest valid snapshot on every failure point.
    - Focused validation: inject failure before/after each write, checksum, object promotion, publication rename, and DuckDB commit, then restart and assert the observable append-only state machine.
    - _Requirements: 6.10–6.16, 7.1–7.4, 14.11_

  - [ ] 8.4 Implement revision-overlap incremental merging in `application/incremental.py`
    - Verify the parent; require an unchanged start and non-decreasing end; compute the contiguous overlap/later-session suffix; seed causal normalization from verified pre-boundary state; reuse immutable pre-boundary objects and rebuild affected suffix partitions.
    - Retain parent content for failed symbols when available, record zero new content otherwise, disclose retained coverage/failures, reuse an existing ID for unchanged science, and create a new ID for genuine revisions without mutating the parent.
    - Focused validation: examples cover overlap zero/non-zero, unchanged/revised actions, extended ranges, invalid shrinking/back-extension, failed symbols with/without parent data, and no duplicate accepted partitions.
    - _Requirements: 7.5–7.16, 14.10_

  - [ ] 8.5 Implement staged `DataIngestionService` orchestration in `application/ingestion.py`
    - Union ordered universe with SPY once; process bounded batches and write chunks; preserve provider request/outcome provenance; normalize, validate, finalize, publish/reuse a snapshot, and persist job/progress/status through injected ports.
    - Return `partially_succeeded` exactly when usable published data has any symbol failure, quarantine, gap, or staleness; return failed when no valid candidate can publish; attach limitation disclosures and preserve all prior snapshots.
    - Focused validation: fake-provider service tests cover success, partial batches, empty outcomes, pre-publication failures, unchanged retry, incremental parent retention, progress counters, and sanitized errors.
    - _Requirements: 1.8, 3.1–3.19, 5.18–5.20, 7.1–7.16, 14.1–14.11, 15.1, 15.7_

  - [ ]* 8.6 Write the Hypothesis test for Property 6 in `tests/properties/test_property_06_snapshot_identity.py`
    - Generate valid parent snapshots, scientific rows/identity fields, batch permutations, interruptions/retries, operational timestamps, and relocated roots; compare IDs/object references to a canonical reference publisher.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 6: Snapshot content identity is idempotent and confluent`.
    - Focused validation: require at least 100 examples and prove equivalent reruns/interruption recovery/relocation share identity while every generated scientific change creates a different identity without parent mutation.
    - **Property 6: Snapshot content identity is idempotent and confluent**
    - **Validates: Requirements 6.4–6.9, 6.17, 7.5–7.10, 17.6, 17.18**

  - [ ]* 8.7 Write the Hypothesis test for Property 7 in `tests/properties/test_property_07_publication_state_machine.py`
    - Generate staged-write/checksum/validation/publication/mutation command sequences against a fake filesystem state machine, supplemented by sampled real temporary-filesystem interruption points.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 7: Publication and immutability state machine preserves the last valid snapshot`.
    - Focused validation: require at least 100 generated state sequences and assert readers observe only the complete previous/new snapshot, reconciliation handles complete orphans, corrupt/partial candidates stay unavailable, and mutation always fails.
    - **Property 7: Publication and immutability state machine preserves the last valid snapshot**
    - **Validates: Requirements 6.10–6.16, 6.18, 7.1–7.4, 7.11–7.16, 14.11, 17.23**

  - [ ]* 8.8 Add snapshot/ingestion fault-injection and incremental integration tests
    - Add temporary-root tests for every publication boundary, interrupted-and-retried versus uninterrupted identity, unchanged/revised overlap, batch-order confluence, failed symbols with/without parent content, startup reconciliation, copy/relocation, corruption, and mutation rejection.
    - Focused validation: verify DuckDB indexes, manifests, CAS objects, status/progress, and prior snapshot availability after each injected failure; never call Yahoo Finance.
    - _Requirements: 6.10–6.18, 7.1–7.16, 14.10–14.13, 17.18, 17.23_

- [ ] 9. Checkpoint — Verify the reproducible data plane
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement the verified snapshot-to-Zipline bundle adapter
  - [ ] 10.1 Implement `ZiplineBundleAdapter` in `infrastructure/zipline_bundle.py`
    - Verify the snapshot first; assign deterministic SIDs; write XNYS asset metadata and lazy raw daily OHLCV; map canonical splits/dividends exactly once; record snapshot/policy/calendar/adapter/action/bundle checksums; return an exact ingestion locator rather than `latest`.
    - Keep research-adjusted bars out of ledger prices, never acquire source data through Zipline, and cache only checksum-verified derived bundles keyed by snapshot and adapter version.
    - Focused validation: materialize the tiny fixture bundle twice and assert deterministic SIDs, raw values, action tables, exact locator, cache verification, and rebuild after derived-cache corruption.
    - _Requirements: 6.12–6.15, 9.1, 9.16–9.17, 15.4_

  - [ ]* 10.2 Add the pinned Zipline bundle contract test in `tests/contract/test_zipline_bundle_contract.py`
    - Verify asset lifetimes/auto-close, deterministic SIDs, lazy daily rows, no minute data, split `old/new` mapping, dividend nullable dates, action checksum, and exact snapshot ingestion selection against the locked Zipline version.
    - Focused validation: fail if the third-party writer/reader extension seam changes or if adjusted research prices or duplicate actions reach the ledger.
    - _Requirements: 9.1, 9.16–9.17_

  - [ ]* 10.3 Add split/dividend golden bundle fixtures and exactly-once action tests
    - Add `tests/golden/zipline_actions/` for forward/reverse splits and dividends with expected actual-share changes, cash-in-lieu, dividend cash, value continuity, and integer quantities.
    - Focused validation: independently compare platform raw/actions, derived bundle data, and Zipline ledger effects to prove the canonical action stream is applied once and only once.
    - _Requirements: 4.10–4.13, 9.6, 9.12, 9.17–9.18_

  - [ ]* 10.4 Add derived-bundle rebuild and projection integration tests
    - Verify a copied/verified snapshot materializes the same bundle identity, an invalid cache is rebuilt, a corrupt source snapshot is rejected, and only configured symbols/fields/session windows are read.
    - Focused validation: instrument Parquet scans and ensure the derived bundle is never accepted as a source of snapshot or artifact truth.
    - _Requirements: 6.12–6.17, 15.4, 15.8_

- [ ] 11. Implement complete and deterministic monthly momentum decisions
  - [ ] 11.1 Implement `monthly_momentum_v1` in `domain/strategy.py`
    - Select XNYS month-end signal sessions; enforce 253-session warm-up; calculate adjusted-close `t-252` to `t-21` returns; record endpoint values/checksums; mark endpoint/tradability exclusions; rank score descending then symbol ascending.
    - Select up to position count even for negative scores, assign exact equal rational weights summing to one, emit one decision for every configured symbol, and choose all cash when none is eligible.
    - Focused validation: table-driven cases cover holidays/month ends, exact warm-up boundary, unavailable endpoints, non-tradable assets, tied/negative scores, position-count caps, all-cash, and repeated identical output.
    - _Requirements: 8.1–8.15_

  - [ ] 11.2 Implement causal decision delivery and order-intent generation in `application/decisions.py`
    - Read history only through signal close from the pinned snapshot, reveal precomputed decisions to Zipline only on their signal session, mark current equity using action-effective close, compute whole-share target deltas, and request full liquidation of unselected holdings.
    - Record fixed endpoints/position count/policy version in run inputs and produce deterministic scientific order IDs without reading next-session prices.
    - Focused validation: spies reject any post-signal read; examples verify target-share flooring, liquidation intents, deterministic order IDs, and no warm-up orders.
    - _Requirements: 8.2–8.14, 9.3–9.5, 9.19, 9.21_

  - [ ]* 11.3 Write the Hypothesis test for Property 8 in `tests/properties/test_property_08_momentum_decisions.py`
    - Generate ordered universes, synthetic XNYS histories, tradability/endpoint masks, equal/distinct scores, and valid position counts; compare decisions to a simple sort/slice/reference-weight model.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 8: Monthly momentum decisions are complete, exact, and deterministic`.
    - Focused validation: require at least 100 examples and assert one decision per symbol, explicit exclusions, deterministic rank/ties, exact weight sum, position cap, and no warm-up orders.
    - **Property 8: Monthly momentum decisions are complete, exact, and deterministic**
    - **Validates: Requirements 8.1–8.15**

  - [ ]* 11.4 Add momentum boundary and decision-artifact golden tests
    - Extend `tests/golden/daily_clean/` to cover enough sessions for a rebalance and add expected endpoint rows, decisions, ranks, rational targets, and deterministic order intents.
    - Focused validation: compare canonical decision artifacts/checksums and verify changing data strictly after signal close does not change the fixture's earlier decisions.
    - _Requirements: 8.1–8.15, 9.19, 17.19_

- [ ] 12. Implement next-session Zipline execution, accounting, and backtest orchestration
  - [ ] 12.1 Implement `CashSafeOpenBlotter` in `infrastructure/zipline_engine.py`
    - On the first tradable session after signal, reject missing/non-positive opens; calculate adverse sell/buy prices; calculate commission on actual fill notional; process sells by symbol then buys by decision rank/symbol; cap quantities to holdings/cash including commission rates above 100%.
    - Emit no zero fill, preserve unfilled remainders with actionable reasons, submit transactions through Zipline's ledger, and retain the pinned engine's canonical action/cash-in-lieu behavior.
    - Focused validation: examples cover one sell/buy, overnight gap, missing open, unaffordable buy, high commission, sell-first funding, adverse slippage, exact costs, and largest-affordable whole shares.
    - _Requirements: 9.3–9.13, 9.17_

  - [ ] 12.2 Implement the Zipline `BacktestEngine` adapter in `infrastructure/zipline_engine.py`
    - Configure the pinned verified bundle, XNYS sessions, long-only/max-leverage controls, USD 100,000 cash, zero cash return, deterministic seed, decision delivery, custom blotter, and bounded progress; extract canonical orders, fills, positions, cash, returns, equity, and decisions.
    - Pin exactly one snapshot and exact bundle before reading, project only configured symbols/fields/window, and prohibit mutable/latest bundle resolution.
    - Focused validation: run the tiny fixture through the real event loop and verify first state, next-session fill timing/base open, action handling, progress sessions, and complete core output roles.
    - _Requirements: 6.14–6.15, 9.1–9.6, 9.15–9.17, 9.24, 13.7–13.9, 15.4, 15.8_

  - [ ] 12.3 Implement accounting audits and `BacktestService` in `application/backtests.py`
    - Allocate Run ID before execution through the tracker port; verify snapshot/range/readiness; run the exact bundle; audit integer/non-negative quantities, cash, gross leverage, action/fill chronology, exact cost formulas, and equity reconciliation after each action/fill/mark.
    - Stop on invariant failure with preserved diagnostics; allow disclosed unfilled orders when the ledger remains valid; pass complete audited output to evaluation and experiment finalization.
    - Focused validation: inject each invariant violation and prove immediate failure, diagnostic preservation, prior-run/snapshot availability, and no switch away from the pinned snapshot.
    - _Requirements: 9.2, 9.6–9.24, 11.3–11.10, 14.12_

  - [ ]* 12.4 Write the Hypothesis test for Property 9 in `tests/properties/test_property_09_no_look_ahead.py`
    - Generate paired valid histories equal through a chosen signal or fill boundary and arbitrarily different afterward; execute through decision/fill/valuation seams and compare canonical prefixes/checksums.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 9: Prefix equivalence enforces no look-ahead`.
    - Focused validation: require at least 100 paired examples and assert decisions/orders through signal and fills/valuations through fill remain equal, with changes permitted only after the first changed-information boundary.
    - **Property 9: Prefix equivalence enforces no look-ahead**
    - **Validates: Requirements 9.3–9.5, 9.19–9.21, 17.9, 17.19–17.20**

  - [ ]* 12.5 Write the Hypothesis test for Property 10 in `tests/properties/test_property_10_execution_accounting.py`
    - Generate valid starting portfolios, orders, opens/missing opens, actions, and finite non-negative costs; compare every action/fill/mark to an independent Decimal sell-first/buy-second accounting model.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 10: Whole-share execution and accounting invariants`.
    - Focused validation: require at least 100 examples and assert integers, adverse prices, exact commission/slippage, largest affordable buys, non-negative positions/cash, leverage `[0,1]`, zero cash return, and cent-level equity reconciliation.
    - **Property 10: Whole-share execution and accounting invariants**
    - **Validates: Requirements 9.2, 9.6–9.18, 17.8, 17.21–17.22**

  - [ ]* 12.6 Add pinned Zipline execution and overnight-gap golden contract tests
    - Add `tests/contract/test_zipline_execution_contract.py` and `tests/golden/overnight_gap/` for next-open timing, sell-before-buy, costs, cash caps, unfilled orders, actual-share split/dividend effects, and core ledger artifacts.
    - Focused validation: fail on third-party extension API/ledger behavior drift and explicitly verify USD 100,000 initial cash, whole shares, no shorting/leverage, no same-bar fill, and no double action application.
    - _Requirements: 9.1–9.18, 17.21–17.22_

- [ ] 13. Implement SPY-aligned deterministic evaluation and artifacts
  - [ ] 13.1 Implement metric/reference functions in `domain/evaluation.py`
    - Implement total return, CAGR, sample annualized volatility, zero-rate Sharpe with null reasons, signed maximum drawdown, turnover, commissions, slippage, monthly compounding, and strategy-minus-benchmark differences using 252 sessions.
    - Focused validation: unit examples cover zero/one observation, zero volatility, positive/negative paths, drawdowns, monthly boundaries, cost totals, and formula agreement with hand calculations.
    - _Requirements: 10.6–10.11, 10.16_

  - [ ] 13.2 Implement `EvaluationService` and canonical result artifacts in `application/evaluation.py`
    - Load adjusted SPY from the same verified snapshot; align exact evaluation sessions; block on and enumerate every SPY gap; compute both metric sets; emit canonical returns/equity/drawdown/monthly/position/order/fill/decision/metric tables and Vega-Lite specs with checksums.
    - Include limitation disclosures, unfilled orders, ending cash, and cost summaries; query only required Parquet columns through DuckDB/filtered scans.
    - Focused validation: evaluate the clean and SPY-gap fixtures, verify every artifact role/checksum, and prove repeated input/order permutations yield identical scientific outputs.
    - _Requirements: 10.1–10.18, 15.5_

  - [ ]* 13.3 Write the Hypothesis test for Property 11 in `tests/properties/test_property_11_evaluation.py`
    - Generate finite strategy/SPY session series, explicit SPY gap sets, positive equity paths, and execution costs; compare alignment, errors, metrics, monthly rows, differences, and checksums to independent formulas.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 11: Evaluation is aligned, gap-safe, and deterministic`.
    - Focused validation: require at least 100 examples and assert all missing sessions are reported, comparison is blocked on any gap, 252/zero-rate formulas hold, null reasons replace infinities, and row-order changes do not alter output.
    - **Property 11: Evaluation is aligned, gap-safe, and deterministic**
    - **Validates: Requirements 10.1–10.18, 17.25**

  - [ ]* 13.4 Add metric, artifact, and SPY-gap golden tests
    - Add `tests/golden/stable_run/` evaluation expectations and `tests/golden/spy_gap/` blocked-comparison expectations, including reviewed metric values, canonical tables/specs, and checksums.
    - Focused validation: compare field-level outputs, verify every missing SPY session is actionable, and ensure strategy-only cost/unfilled/cash metrics remain available for diagnostics.
    - _Requirements: 10.1–10.18, 17.25_

- [ ] 14. Implement immutable local MLflow run lifecycle and artifact tracking
  - [ ] 14.1 Implement deterministic environment/source fingerprinting in `infrastructure/fingerprint.py`
    - Record exact Python/OS/architecture, sorted installed distributions, source revision/dirty state, seed, and effective source checksum over relevant source, `pyproject.toml`, and lock bytes while excluding generated/cache/test-output paths.
    - Focused validation: temporary-repository tests prove order/path normalization, included untracked source changes alter the checksum, excluded generated files do not, and dirty runs remain disclosed.
    - _Requirements: 11.5–11.6, 9.24_

  - [ ] 14.2 Implement the local MLflow adapter in `infrastructure/mlflow_tracker.py`
    - Use `MlflowClient` with the configured local SQLite URI and local artifact references; map one platform Run ID to one MLflow ID; record redacted scalar inputs and running state before backtest; log reference artifacts/metrics and terminal status idempotently.
    - Keep authoritative scientific artifacts in CAS, reject secret-bearing values, and retain a discoverable platform run when MLflow creation fails.
    - Focused validation: temporary SQLite tests cover running/succeeded/failed records, one-to-one mapping, redacted values, reference-only large artifacts, and idempotent terminal replay.
    - _Requirements: 11.1–11.10, 11.19–11.20, 14.12_

  - [ ] 14.3 Implement run finalization intents and terminal repository guards in `infrastructure/duckdb_metadata.py`
    - Publish and verify artifacts/manifest, persist a checksummed finalization intent while the platform run remains running/finalizing, synchronize MLflow, then atomically copy payload indexes and mark the run terminal/immutable.
    - Reconcile crashes before/after MLflow terminalization, support failed runs without an MLflow mapping, and reject any platform mutation of terminal inputs, metrics, state, manifest, or artifacts.
    - Focused validation: inject failures at every intent/MLflow/DuckDB boundary and prove replay reaches one coherent terminal state without exposing premature success or mutating earlier runs.
    - _Requirements: 11.7–11.18, 14.12_

  - [ ] 14.4 Implement `ExperimentTracker` orchestration and verified run-artifact access in `application/experiments.py`
    - Allocate the platform Run ID and DuckDB running row first; record snapshot/configuration/strategy/range/fingerprint; finalize succeeded or failed runs with manifests/checksums/disclosures; preserve diagnostic artifacts; verify every artifact before opening.
    - Separate Run ID/times/logs from scientific identity, include Snapshot ID and every scientific checksum, and return an actionable error plus invalid status on corruption.
    - Focused validation: examples cover pre-execution failure, successful finalization, failed diagnostics, corruption, terminal mutation, later-run failure isolation, and operationally distinct runs sharing scientific artifacts.
    - _Requirements: 11.3–11.20, 12.12–12.13_

  - [ ]* 14.5 Add MLflow/DuckDB finalization contract and fault-injection tests
    - Add `tests/contract/test_mlflow_tracker_contract.py` for local lifecycle/mapping/redaction plus `tests/integration/test_run_finalization_recovery.py` for failures before/after artifact publication, intent commit, MLflow terminalization, and DuckDB terminal commit.
    - Focused validation: restart reconciliation must converge idempotently; terminal mutation must fail; failed attempts must preserve Run ID, diagnostics, prior runs, and verified artifacts.
    - _Requirements: 11.1–11.20, 14.12, 17.24_

- [ ] 15. Checkpoint — Verify the complete research engine and immutable run lifecycle
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement run discovery, comparison, inspection, pagination, and the application facade
  - [ ] 16.1 Implement indexed run discovery in `infrastructure/duckdb_metadata.py`
    - Query by Run ID, creation range, Snapshot ID, strategy, ordered universe, evaluation range, and terminal state with deterministic ordering and bounded pages; return only DTO fields required by application services.
    - Focused validation: seed mixed run records and verify every individual/combined filter, stable pagination, successful-only comparison candidates, and no direct ad hoc UI SQL.
    - _Requirements: 12.1, 13.12, 15.6_

  - [ ] 16.2 Implement `ComparisonService` in `application/comparisons.py`
    - Accept ordered 2–10 distinct successful runs; verify manifests/artifacts; produce recursive non-secret snapshot/config/environment differences first; retain original metrics/ranges; align strategy and benchmark curves to exact session intersection; attach disclosures.
    - Reject count, duplicate, non-successful, missing, or corrupt selections before producing comparison output and never silently recompute original run metrics.
    - Focused validation: integration examples cover 2, 10, 1, and 11 runs; mixed snapshots/configurations/fingerprints/ranges; duplicate/failed runs; corrupt artifacts; and empty/non-empty session intersections.
    - _Requirements: 12.2–12.14, 13.13–13.14_

  - [ ]* 16.3 Write the Hypothesis test for Property 13 in `tests/properties/test_property_13_comparison.py`
    - Generate ordered run selections with valid/invalid sizes, states, duplicates, artifact integrity, ranges, curves, configurations, and fingerprints; compare acceptance, errors, differences, and intersection alignment to reference models.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 13: Comparison validation and alignment preserve provenance`.
    - Focused validation: require at least 100 examples and assert input order/original metrics/ranges are preserved, every provenance difference is shown, and corruption blocks comparison.
    - **Property 13: Comparison validation and alignment preserve provenance**
    - **Validates: Requirements 12.2–12.14, 13.13–13.14, 17.26–17.27**

  - [ ] 16.4 Implement the typed `ResearchApplication` facade and configuration-handle registry in `application/services.py`
    - Expose configuration, ingestion, snapshot, backtest, run search/inspection, comparison, page, and artifact methods returning typed `Result`; associate opaque process-local handles with the exact frozen resolved configuration and reject stale/unknown handles.
    - Catch unexpected exceptions once, sanitize/log them with correlation IDs, preserve prior results, and inject concrete adapters through constructors rather than importing UI state or a service locator.
    - Focused validation: contract tests call every facade method with valid/error inputs, prove secrets cannot be inspected/serialized through handles, and verify no infrastructure exception crosses the boundary.
    - _Requirements: 1.8–1.9, 2.13–2.14, 13.2–13.12, 14.6–14.8_

  - [ ] 16.5 Implement snapshot/run/artifact inspection and bounded paging in `application/inspection.py`
    - Return verified manifest, non-secret configuration, fingerprint, validation report, logs, artifact metadata/download streams, and ordinary table pages using `min(requested, configured, 100)` with deterministic offsets/order.
    - Keep full downloads on a separate verified streaming path and project/filter only page columns/rows through DuckDB/Parquet.
    - Focused validation: scanner spies assert page limits/projections, adjacent pages neither overlap nor skip, corrupt artifacts are rejected, and downloads do not materialize ordinary table rows.
    - _Requirements: 12.12–12.13, 13.12, 13.15–13.17, 15.5–15.6_

  - [ ]* 16.6 Write the Hypothesis test for Property 15 in `tests/properties/test_property_15_pagination.py`
    - Generate artifact sizes, pages, requested sizes, and configured sizes; compare output to a deterministic bounded slice and inspect scanner limit/offset/projection calls.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 15: Ordinary table pagination is absolutely bounded`.
    - Focused validation: require at least 100 examples and assert no page exceeds 100, unchanged adjacent pages are exact/disjoint, and download/page paths remain separate.
    - **Property 15: Ordinary table pagination is absolutely bounded**
    - **Validates: Requirements 13.15–13.17, 15.6, 17.28**

  - [ ]* 16.7 Write the Hypothesis test for Property 14 in `tests/properties/test_property_14_secret_sink_safety.py`
    - Generate non-empty secrets and exceptions, URLs, headers, nested configuration, progress, logs, MLflow values, manifest metadata, and presenter text containing literal/encoded forms; pass them through each real sink sanitizer and artifact metadata gate.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 14: Secret redaction is idempotent and complete across sinks`.
    - Focused validation: require at least 100 examples, assert repeated redaction is identical, only whole `[REDACTED]` markers/presence indicators remain, no secret sequence appears, and unsanitized metadata fails closed.
    - **Property 14: Secret redaction is idempotent and complete across sinks**
    - **Validates: Requirements 16.1, 16.4, 16.6–16.9, 11.19, 13.20**

  - [ ]* 16.8 Add application-facade, jobs, inspection, and comparison integration tests
    - Compose fake providers plus real temporary storage/metadata/tracking adapters; exercise typed successes/errors, stale configuration handles, run search, manifest/log/artifact inspection, downloads, pagination, comparison bounds, checksum failures, and prior-result access after later failures.
    - Focused validation: ensure all returned diagnostics are actionable/sanitized, all access verifies checksums, and no presentation or infrastructure exception leaks through the facade.
    - _Requirements: 1.8–1.9, 12.1–12.14, 13.11–13.17, 14.6–14.13_

- [ ] 17. Implement the Streamlit end-to-end visual workflow
  - [ ] 17.1 Implement redacted presenters and shared Streamlit view components in `ui/presenters.py` and `ui/components.py`
    - Format only application DTOs; require `LimitationDisclosure` for every data/snapshot/result/comparison presenter; render actionable errors, job progress, metrics, canonical charts, paged tables, and verified download handles without exposing secret/configuration handles.
    - Focused validation: unit-test presenter output for secret absence, required disclosures, partial/failure status, page bounds, and representative null/error metric states.
    - _Requirements: 13.4–13.6, 13.9–13.10, 13.15–13.20_

  - [ ] 17.2 Implement `ui/app.py` composition/navigation and Configure/Ingest pages
    - Build the single composition root with `st.cache_resource`; construct concrete adapters and `ResearchApplication`; provide YAML selection and approved editable controls; validate before enabling actions; invoke synchronous ingestion and render persisted progress/result without direct infrastructure imports.
    - Preserve the opaque configuration handle only in process/session state; require re-resolution after restart; keep prior snapshots/runs accessible after partial/failure results.
    - Focused validation: use mocked facade calls to verify validation gates, environment precedence visibility, control bounds, synchronous progress updates, sanitized warnings, partial-success display, and Snapshot ID/error outcomes.
    - _Requirements: 13.1–13.5, 13.11, 13.18–13.20_

  - [ ] 17.3 Implement the Snapshots page in `ui/pages/snapshots.py`
    - Provide bounded search/listing, checksum-verified selection, provenance/range/calendar/policy/configuration summaries, validation counts/reasons, gaps, staleness, failed/retained symbols, comparison readiness, disclosures, paged details, and verified downloads.
    - Focused validation: presenter/facade spies verify every required field, no page above the configured/absolute cap, corrupt snapshots cannot be selected for use, and disclosures remain visible.
    - _Requirements: 13.6–13.7, 13.15–13.20_

  - [ ] 17.4 Implement Backtest and Runs pages in `ui/pages/backtest.py` and `ui/pages/runs.py`
    - Require explicit verified Snapshot ID; invoke synchronous backtest; show session progress/warnings; display Run ID, strategy/SPY metrics/differences, curves, drawdowns, positions, transactions, costs, cash, manifests, redacted configuration/fingerprint, validation/log details, and verified artifact downloads.
    - Focused validation: mocked success/unfilled/failure/corruption cases verify controls, prior-run retention, all result sections, checksum enforcement, pagination, and disclosure/secret safety.
    - _Requirements: 13.7–13.12, 13.15–13.20_

  - [ ] 17.5 Implement the Compare page in `ui/pages/compare.py`
    - Provide ordered selection of 2–10 successful runs, disable invalid selections with the accepted range, and show snapshot/configuration/fingerprint differences before metrics plus original/aligned ranges and all strategy/SPY curves.
    - Focused validation: mocked 1/2/10/11-run, failed-run, differing-provenance, artifact-corruption, and no-common-session cases verify no invalid comparison call and persistent visible disclosures.
    - _Requirements: 12.2–12.14, 13.13–13.20_

  - [ ]* 17.6 Add Streamlit AppTest coverage for the complete visual workflow
    - Add in-process AppTests for configuration gates, ingest/backtest progress, partial/failure diagnostics, prior-result access, snapshot/run inspection, comparison bounds, pagination, downloads, and visible disclosures on data, result, and comparison views.
    - Focused validation: use mocked application services and fixture artifacts only; assert no development server starts, no secret text appears, and ordinary table renders never exceed 100/configured rows.
    - _Requirements: 13.1–13.20, 17.28, 17.35_

- [ ] 18. Checkpoint — Verify application services and Streamlit workflow
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. Complete integrated reproducibility, memory, documentation, and release validation
  - [ ]* 19.1 Add the local full-pipeline integration fixture and test in `tests/integration/test_phase1_pipeline.py`
    - Build a temporary application with effective YAML, fake provider, XNYS fixture, real raw/normalized/quarantine/gap Parquet, atomic snapshot/DuckDB, derived Zipline bundle, momentum execution, SPY evaluation, CAS artifacts, and local MLflow finalization.
    - Verify artifact opening and run/snapshot discovery through the facade, including one partial-ingestion diagnostic path and the clean successful path; do not call an external service.
    - Focused validation: assert every Phase 1 output role, checksum, manifest link, disclosure, job/run state, and prior-result isolation from configuration through artifact verification.
    - _Requirements: 1.8–1.10, 17.32_

  - [ ]* 19.2 Write the Hypothesis test for Property 12 in `tests/properties/test_property_12_stable_rerun.py`
    - Generate bounded local fixture runs satisfying identical snapshot/configuration/source/fingerprint/dependency/seed/writer/adapter preconditions; execute the complete snapshot-to-bundle-to-backtest-to-evaluation pipeline twice with distinct operational IDs/times.
    - Include this exact comment: `# Feature: quantitative-research-platform, Property 12: Stable reruns preserve scientific outputs and checksums`.
    - Focused validation: require at least 100 bounded examples with a deadline-free Hypothesis profile, assert role-for-role core output/metric/manifest/artifact byte equivalence, and assert operational records remain distinct.
    - **Property 12: Stable reruns preserve scientific outputs and checksums**
    - **Validates: Requirements 9.22–9.24, 10.16–10.17, 11.11–11.14, 17.10, 17.33–17.34**

  - [ ]* 19.3 Add bounded-memory behavior tests and a marked local streaming benchmark
    - Add scanner/writer/provider/UI spies proving batches stay at or below 10 symbols, chunks at or below `write_chunk_size`, validation aggregates incrementally, backtests request only active symbols/fields/windows, metrics project required columns, and pages stay bounded.
    - Add a `memory`-marked dataset larger than one chunk that fails on unbounded `read_all()`/`to_pandas()` paths without imposing a fragile absolute RSS threshold.
    - Focused validation: inspect call traces, batch/record-batch sizes, projections, predicates, and manifest-recorded limits on an 18 GB-laptop-suitable local run.
    - _Requirements: 15.1–15.8_

  - [ ]* 19.4 Add local smoke tests and the explicitly gated external-provider smoke seam
    - Add single-shot smoke coverage for build/install/import, temporary DuckDB/MLflow/artifact/lock creation, one fixture ingestion/backtest via `ResearchApplication`, and Streamlit composition/AppTest import without a server.
    - Add one opt-in `external` yfinance request for a short completed range and one small batch under configured retry limits; keep it excluded from default/property runs and sanitize any diagnostics.
    - Focused validation: default smoke execution remains offline and deterministic; the opt-in seam asserts only the narrow provider contract, not scientific golden values.
    - _Requirements: 1.1–1.9, 3.2–3.9, 17.30–17.32_

  - [ ] 19.5 Finalize developer and research-operation guidance in `README.md` and `docs/`
    - Document Python 3.11 frozen setup, default/local-secret configuration, source/data layout, offline and optional test commands, fixture ingest/backtest workflow, Streamlit launch command for the user to run manually, artifact/snapshot/run inspection, recovery/reconciliation, and checksum/immutability expectations.
    - Document free-source, explicit-universe, point-in-time/survivorship, quality/completeness, cost/execution, synchronous-job, local-trust, and research-not-trading limitations plus the excluded future scope.
    - Focused validation: add link/command/config-example checks and verify guidance names only implemented Phase 1 modules and never includes credentials or generated local paths.
    - _Requirements: 1.10–1.11, 3.19, 10.18, 11.20, 12.14, 13.18–13.20, 16.1–16.3_

  - [ ] 19.6 Add a single-shot local release-validation entry point in `scripts/validate_release.py`
    - Orchestrate frozen-lock verification, build/import, Ruff, mypy, offline unit/property/contract/integration/AppTest/smoke suites, golden checksum verification, and excluded-dependency/scope checks without starting watchers, servers, deployment, or external tests.
    - Emit a concise local report identifying the failed gate and corrective command while preserving full tool exit status.
    - Focused validation: unit-test command construction/error propagation and execute the entry point against a temporary or current checkout in single-run mode.
    - _Requirements: 1.1–1.11, 17.1–17.35_

  - [ ]* 19.7 Add the final end-to-end Phase 1 acceptance test in `tests/acceptance/test_phase1_vertical_slice.py`
    - Execute configuration parsing, fixture ingestion, validation/quarantine/gaps, atomic snapshot publication/verification, Zipline bundle creation, momentum decisions, next-session accounting, SPY evaluation, MLflow run finalization, run discovery/comparison, artifact inspection, and Streamlit AppTest views through public interfaces.
    - Rerun under Stable Rerun conditions and verify identical Core Backtest Output and scientific checksums while Run IDs/timestamps differ; inject one later failure and prove prior valid snapshots/runs remain usable.
    - Focused validation: require all disclosures, redaction, memory/page bounds, immutable mutation rejection, no-look-ahead evidence, and release-validation gates to pass before accepting the vertical slice.
    - _Requirements: 1.8–1.11, 6.10–6.18, 9.19–9.24, 11.15–11.20, 13.1–13.20, 14.1–14.13, 15.1–15.8, 16.1–16.10, 17.32–17.35_

- [ ] 20. Final checkpoint — Validate the first usable Phase 1 vertical slice
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks in Kiro and can be skipped for a faster implementation pass; all non-optional tasks are core implementation or required project/reproducibility artifacts.
- Every numbered design correctness property appears exactly once as a dedicated Hypothesis task with the exact `# Feature: quantitative-research-platform, Property N: ...` comment convention.
- Property tests use deterministic/local fakes and fixtures; only the separately marked `external` smoke seam may contact yfinance.
- Checkpoints are validation boundaries, not deployment gates. The plan contains no cloud, staging, production, broker, paper/live trading, FastAPI, distributed queue, LLM, ML, Alphalens, or Pyfolio work.
- The dependency graph is intentionally conservative: each wave starts only after all earlier waves complete, and tasks that may modify the same file are separated.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4"] },
    { "id": 3, "tasks": ["3.1", "3.2"] },
    { "id": 4, "tasks": ["3.3", "3.4"] },
    { "id": 5, "tasks": ["3.5"] },
    { "id": 6, "tasks": ["4.1"] },
    { "id": 7, "tasks": ["4.2", "4.3"] },
    { "id": 8, "tasks": ["4.4", "4.5", "4.6"] },
    { "id": 9, "tasks": ["5.1", "5.2"] },
    { "id": 10, "tasks": ["5.3"] },
    { "id": 11, "tasks": ["5.4", "5.5"] },
    { "id": 12, "tasks": ["6.1", "6.3"] },
    { "id": 13, "tasks": ["6.2", "6.4", "6.5"] },
    { "id": 14, "tasks": ["6.6"] },
    { "id": 15, "tasks": ["7.1"] },
    { "id": 16, "tasks": ["7.3"] },
    { "id": 17, "tasks": ["7.2", "7.4", "7.5"] },
    { "id": 18, "tasks": ["7.6"] },
    { "id": 19, "tasks": ["8.1"] },
    { "id": 20, "tasks": ["8.2"] },
    { "id": 21, "tasks": ["8.3"] },
    { "id": 22, "tasks": ["8.4"] },
    { "id": 23, "tasks": ["8.5"] },
    { "id": 24, "tasks": ["8.6", "8.7"] },
    { "id": 25, "tasks": ["8.8"] },
    { "id": 26, "tasks": ["10.1", "11.1"] },
    { "id": 27, "tasks": ["10.2", "10.3", "11.2"] },
    { "id": 28, "tasks": ["10.4", "11.3", "11.4"] },
    { "id": 29, "tasks": ["12.1"] },
    { "id": 30, "tasks": ["12.2"] },
    { "id": 31, "tasks": ["12.3"] },
    { "id": 32, "tasks": ["12.4", "12.5", "12.6"] },
    { "id": 33, "tasks": ["13.1"] },
    { "id": 34, "tasks": ["13.2"] },
    { "id": 35, "tasks": ["13.3", "13.4"] },
    { "id": 36, "tasks": ["14.1", "14.2"] },
    { "id": 37, "tasks": ["14.3"] },
    { "id": 38, "tasks": ["14.4"] },
    { "id": 39, "tasks": ["14.5"] },
    { "id": 40, "tasks": ["16.1", "16.5"] },
    { "id": 41, "tasks": ["16.2"] },
    { "id": 42, "tasks": ["16.3", "16.6"] },
    { "id": 43, "tasks": ["16.4"] },
    { "id": 44, "tasks": ["16.7", "16.8"] },
    { "id": 45, "tasks": ["17.1"] },
    { "id": 46, "tasks": ["17.2"] },
    { "id": 47, "tasks": ["17.3", "17.4", "17.5"] },
    { "id": 48, "tasks": ["17.6"] },
    { "id": 49, "tasks": ["19.1", "19.3"] },
    { "id": 50, "tasks": ["19.2", "19.4"] },
    { "id": 51, "tasks": ["19.5"] },
    { "id": 52, "tasks": ["19.6"] },
    { "id": 53, "tasks": ["19.7"] }
  ]
}
```
