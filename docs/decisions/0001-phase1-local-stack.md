# ADR 0001: Phase 1 Local Research Stack

- **Status:** Accepted
- **Date:** 2026-07-20
- **Scope:** First usable Phase 1 vertical slice
- **Requirements:** 1.1, 1.6–1.11, 15.1–15.6

## Context

Phase 1 must deliver a complete, reproducible research loop for one developer on a local Apple-silicon laptop: resolve configuration, ingest free daily US-equity data, validate and preserve immutable snapshots, run one interpretable long-only momentum baseline, compare it with SPY, record experiments and artifacts, and inspect the workflow through a visual UI. It must be bounded enough to implement and validate before adding later roadmap capabilities.

The [project constitution](../../project_constitution.md) establishes local-first development, free-only resources, modularity, loose coupling, replaceability, reproducibility, reuse before reinventing, and incremental delivery. The Phase 1 [requirements](../../.kiro/specs/quantitative-research-platform/requirements.md) and [design](../../.kiro/specs/quantitative-research-platform/design.md) refine those principles into a synchronous local modular monolith.

## Decision

Use a **ports-and-adapters modular monolith** running in one local Python process. The system is modular by responsibility, but it is not split into network services or background workers.

Dependency direction is inward:

```text
Streamlit presentation -> application services -> domain policies and models
                                              ^
                                              |
                         infrastructure adapters implement ports
```

- **Domain** contains framework-independent value objects and deterministic policies: configuration values, sessions, bars, actions, validation decisions, strategy decisions, execution arithmetic, metrics, manifests, checksums, and actionable errors.
- **Application** contains synchronous use-case coordinators: configuration resolution, ingestion, validation, snapshot management, backtesting, evaluation, experiment tracking, comparison, job progress, and artifact inspection. It exposes typed operations to the UI and does not read UI session state.
- **Infrastructure** contains replaceable adapters for market data, the XNYS calendar, Parquet/CAS storage, DuckDB metadata, the Zipline engine, MLflow, the filesystem, environment fingerprints, and logging.
- **Presentation** contains Streamlit pages and presenters. It invokes application services and displays redacted DTOs; it does not contain research rules or import infrastructure implementations directly.

All operations are synchronous and local in this slice. There is one composition root that wires infrastructure adapters into application services. This preserves clean boundaries without introducing deployment, coordination, or operational complexity that Phase 1 does not need.

## Selected technology and current Phase 1 responsibility

Each selected package has a concrete responsibility now; none is included only as a placeholder for a future phase.

| Technology | Phase 1 responsibility | Boundary and ownership rule |
|---|---|---|
| **Python 3.11** | Supported runtime for the installable project and all local services, with `>=3.11,<3.12`. | The runtime is pinned and fingerprinted for reproducible runs; no newer Python line is required by this slice. |
| **Pydantic v2** | Define and validate typed configuration models, bounds, cross-field rules, resolved settings, and redacted non-secret views. | Configuration validation belongs at the configuration boundary; domain and application code consume validated values rather than untyped dictionaries. |
| **ruamel.yaml** | Safely parse the one supported YAML configuration document while preserving deterministic, human-readable YAML serialization and detecting duplicate/unknown configuration keys. | YAML is an input/serialization format, not a source of business logic. Secret values come only from mapped environment variables or ignored local secret files. |
| **yfinance** | Implement the initial narrow market-data provider adapter for free daily US-equity and SPY records, including provider provenance and retry/error classification. | The adapter is behind the `Market_Data_Provider` port. Provider data is staged and normalized by the platform; yfinance does not define platform identity or storage. |
| **exchange-calendars (XNYS)** | Supply the pinned official NYSE session schedule used for date-range interpretation, expected-session checks, next-session execution, gap detection, and schedule digests. | Session semantics come from XNYS rather than weekday arithmetic or provider row presence. The calendar version/schedule digest is recorded for reproducibility. |
| **PyArrow / Parquet** | Write and read bounded, deterministic, partitioned tables for raw records, normalized bars, quarantine, validation outputs, and tabular run artifacts. | The platform controls schemas, canonical ordering, checksums, chunking, and partition pruning. Parquet is the durable scientific data representation. |
| **DuckDB** | Provide the local metadata/index store and filtered analytical queries over Parquet: snapshots, runs, validation, jobs, artifacts, and pagination. | DuckDB indexes and queries platform-owned content; it is not the sole copy of scientific data and does not replace immutable Parquet objects. |
| **Zipline Reloaded** | Execute the single monthly rebalanced, long-only momentum baseline and provide the event-driven ledger boundary for orders, fills, positions, returns, and equity. | Zipline receives a verified snapshot through an adapter. Its bundle is disposable, snapshot-keyed derived state, not source-data ownership. |
| **MLflow with a local SQLite backend** | Catalog local runs, lifecycle state, parameters/tags, metrics, and references to checksummed artifacts without requiring a server. | MLflow is an experiment catalog. The platform's manifests, checksums, and local CAS artifact store remain authoritative. |
| **Streamlit** | Provide the visual local workflow for configuration, ingestion, snapshot selection, backtest execution, run inspection, comparison, progress, errors, and artifact downloads. | Streamlit is a presentation adapter only; all research behavior is invoked through application services. |
| **Pytest** | Run unit, contract, integration, smoke, and representative external-boundary tests for the Phase 1 package. | Tests validate domain/application behavior and explicitly bounded infrastructure boundaries; no watcher or always-on service is required. |
| **Hypothesis** | Exercise variable-input domain and application properties, especially configuration, session, normalization, snapshot identity, bounded processing, accounting, and deterministic evaluation behavior. | Property tests complement focused examples and are tied to named requirements; external providers remain covered by representative integration tests. |

The selected stack is free/open-source and supports the constitution's local hardware and memory constraints. Ingestion is batched at no more than ten symbols, writes are chunked, validation aggregates incrementally, backtests project only the active window and required fields, analytical reads use filtered Parquet/DuckDB access, and ordinary UI tables page at no more than 100 rows.

## Data and cache authority

**Platform-owned Parquet and content-addressed storage (CAS) are authoritative.** The platform owns raw provider records, normalized data, quarantine and gap reports, manifests, validation outputs, and canonical run artifacts. A published snapshot or run is immutable and is resolved through its verified manifest and checksums. This is required for scientific reproducibility, stable content-derived identities, provider-revision handling, recovery after partial operations, and independence from any one engine or catalog.

**Zipline bundles are derived caches.** The snapshot-to-Zipline adapter materializes a disposable bundle from a checksum-verified platform snapshot, supplying the platform's raw bars and canonical corporate-action stream exactly once. A bundle is keyed by snapshot identity and may be deleted and rebuilt. Zipline must never download, mutate, or become the canonical owner of source data; a rebuild must produce the same engine input from the same verified snapshot and policy inputs.

**DuckDB and MLflow are indexes/catalogs, not substitutes for authoritative content.** DuckDB provides local query/index transactions and reconciliation metadata. MLflow records experiment lifecycle and references. A catalog failure must not make prior authoritative Parquet/CAS objects scientifically disappear, and an object is opened only after manifest checksum verification.

## Explicit exclusions from this slice

The following are intentionally deferred and must not be scaffolded as Phase 1 implementation modules:

- **FastAPI or another network API:** the local Streamlit UI calls application services directly; a network boundary adds deployment and API lifecycle complexity without improving the single-developer loop.
- **Queues, workers, and distributed/background execution:** jobs are synchronous and local. Queue durability, coordination, and scaling are outside the bounded Phase 1 operating model.
- **Cloud services and hosted infrastructure:** the platform must run with free local resources, preserve research privacy, and avoid cloud cost and vendor lock-in in this slice.
- **LLM capabilities and machine-learning capabilities:** Phase 1 establishes a reproducible classical quantitative baseline first. LLM-assisted research and ML/model-training infrastructure are roadmap capabilities, not dependencies or empty extension points here.
- **Alphalens Reloaded and Pyfolio Reloaded:** the constitution recognizes the broader Zipline-Reloaded ecosystem, but this slice requires only the baseline backtest, deterministic evaluation metrics, comparison, and artifact outputs. These libraries may be evaluated by a later specification when their capabilities are needed.
- **Execution adapters, brokers, paper trading, and live execution:** the product is a research platform, not an execution system. Backtest fills are simulated inside the defined Zipline boundary using next-session adjusted-open assumptions, whole shares, commission, slippage, and long-only constraints. No broker or exchange adapter is part of Phase 1.

These exclusions are scope controls, not claims that the capabilities lack long-term value. Adding one requires a new requirements/design decision that preserves the modular boundaries and authority model above.

## Validation against the constitution priority order

The decision was reviewed in the constitution's stated order: **correctness, maintainability, simplicity, extensibility, reproducibility, developer productivity, then performance**.

1. **Correctness:** XNYS defines expected sessions; platform normalization and validation own adjustment and gap semantics; immutable manifests/checksums protect scientific identity; Zipline receives verified inputs exactly once; application boundaries return actionable errors.
2. **Maintainability:** domain, application, infrastructure, and presentation responsibilities are explicit; adapters are narrow; UI state and infrastructure details do not leak into core use cases.
3. **Simplicity:** one local process, synchronous jobs, one supported provider, one strategy, local files, DuckDB, and SQLite avoid premature services, queues, and distributed coordination.
4. **Extensibility:** ports isolate the provider, calendar, storage, engine, tracker, and UI without speculative plugin frameworks. A later replacement can be introduced behind the same boundary when a real second implementation is needed.
5. **Reproducibility:** Python/dependency fingerprints, calendar digests, resolved non-secret configuration, snapshots, manifests, CAS checksums, deterministic Parquet, and run artifacts are retained.
6. **Developer productivity:** Streamlit provides a visual workflow; mature open-source packages handle parsing, calendar, tabular I/O, analytical querying, backtesting, tracking, and testing instead of being rebuilt.
7. **Performance:** bounded batches/chunks, partition pruning, DuckDB filtered scans, and derived Zipline bundles respect the 18 GB local-memory ceiling without optimizing prematurely for distributed scale.

### Focused validation result

- Every selected package listed above has a current Phase 1 responsibility and a defined boundary; no selected package is an unowned future placeholder.
- The architecture is a local modular monolith, not a networked service or distributed system, matching the approved Phase 1 scope.
- Platform Parquet/CAS authority, DuckDB/MLflow catalog roles, and disposable Zipline cache behavior are explicit and non-overlapping.
- The excluded technologies are not selected as Phase 1 dependencies and are deferred rather than indirectly required.
- This ADR is a decision record only; no engine module or future-scope implementation is introduced by it.

## Consequences

Positive consequences include a complete local research loop with clear ownership, deterministic data identity, bounded processing, replaceable infrastructure adapters, and a small operational footprint. The trade-off is that Phase 1 supports one local synchronous workflow and one baseline strategy rather than cloud scale, multiple strategy libraries, model training, or live execution. Those capabilities remain possible later, but they must be added deliberately through new scope and design decisions rather than leaking into this foundation.
