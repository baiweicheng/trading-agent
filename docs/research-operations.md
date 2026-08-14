# Research Operations Guide

This runbook describes the local Phase 1 research loop and its scientific guardrails. It is intentionally explicit about what the platform records, what it refuses to fabricate, and how to recover from local failures.

## Operating model

The platform is a synchronous, single-developer local workflow:

```text
configuration -> provider batches -> raw Parquet
             -> XNYS normalization/validation -> normalized/quarantine/gap Parquet
             -> verified manifest + CAS objects -> immutable Snapshot_ID
             -> snapshot-keyed Zipline bundle -> momentum backtest
             -> SPY-aligned evaluation -> checksummed artifacts + immutable Run_ID
             -> discovery, inspection, and 2–10-run comparison
```

The Streamlit UI calls the same `ResearchApplication` facade used by local integration tests. There is no background worker or resume daemon. Closing the process can interrupt a job; completed publications remain recoverable through manifests and reconciliation.

## 1. Resolve configuration before data work

Start with [`config/default.yaml`](../config/default.yaml), make a reviewed copy if a different date range or explicit ticker list is needed, and resolve it through the Configure / Ingest page or the `ResearchApplication.resolve_configuration()` method. The required date range is inclusive. The default universe is ordered and explicit; `SPY` is fetched separately as the benchmark unless it is also explicitly listed.

A minimal programmatic flow is:

```python
from pathlib import Path

from quant_research_platform.domain.errors import Ok
from quant_research_platform.ui.app import build_application

application = build_application()
resolution = application.resolve_configuration(Path("config/default.yaml"))
if not isinstance(resolution, Ok):
    for error in resolution.errors:
        print(error.format_for_display())
    raise SystemExit(1)

handle = resolution.value.handle  # opaque; keep it in this process only
```

The resolver validates the complete merged configuration before ingestion or backtesting. It reports parser location, duplicate/unknown key paths, invalid bounds/types, date order, universe normalization errors, project-root/path violations, and unmapped `QRP_` variables as actionable diagnostics. A secret field is never printed in full. See the [developer configuration section](developer-guide.md#configuration).

## 2. Full ingestion and validation

Use Configure / Ingest and click **Ingest data** only after configuration resolves. The application service:

1. unions the configured strategy symbols with `SPY` once, preserving deterministic order;
2. fetches successive provider batches of no more than 10 symbols, with the configured retry policy;
3. preserves provider records and request provenance in raw storage;
4. maps provider dates to the pinned XNYS calendar;
5. applies the versioned `causal_forward_v1` adjustment policy without synthesizing missing bars;
6. validates OHLCV, actions, duplicate Session Keys, session coverage, gaps, and staleness;
7. writes raw, normalized, quarantine, gap, and validation candidates below staging;
8. checksums the staged objects and publishes a complete manifest atomically; and
9. records the job/operation outcome in DuckDB and returns a Snapshot ID.

A successful result can be either:

- `succeeded`: usable data with no recorded symbol failure, quarantine row, gap, or stale status; or
- `partially_succeeded`: a usable snapshot was published, but one or more symbols had provider failures, quarantined records, gaps, or staleness.

Partial success is not silently promoted to clean success. The returned result and limitation disclosure retain failed symbols, quarantine reasons, gaps, stale symbols, retained parent coverage, and comparison readiness. No missing market observation is filled with a fabricated row.

For deterministic offline evidence, run the local fixture pipeline rather than downloading real data:

```bash
uv run pytest tests/smoke/test_local_phase1_smoke.py -q
uv run pytest tests/integration/test_phase1_pipeline.py -m integration -q
```

Those tests use local provider/calendar seams and temporary roots. They exercise configuration, raw/normalized/quarantine/gap Parquet, publication, snapshot verification, bundle projection, momentum decisions, next-session accounting, SPY evaluation, artifact verification, run tracking, comparison, and redaction without network access.

## 3. Snapshot identity, immutability, and incremental updates

A published snapshot is resolved by a content-derived `Snapshot_ID` of the form `snap_<sha256>`. Scientific identity includes the provider/range/universe/benchmark, XNYS calendar identity, policy/schema versions, non-secret configuration checksum, validation summary, limitation-disclosure version, and sorted referenced object checksums. Local paths, retrieval/creation/job timestamps, and operational IDs do not change that identity.

Before use, `SnapshotManager.open_verified(snapshot_id)` verifies the manifest and every referenced object. `inspect_snapshot(snapshot_id)` returns the verified provenance, covered range, validation summary, comparison readiness, and artifact references. Use these methods instead of opening a Parquet path directly.

The platform’s immutability rules are deliberate:

- staging is not reader-visible;
- an existing CAS checksum may be reused only after byte verification;
- a published manifest and referenced objects are not replaced in place;
- mutation attempts return an actionable storage/atomicity error and require a new snapshot;
- a provider revision creates new scientific content and therefore a new Snapshot ID;
- a later failed operation does not delete or rewrite an earlier valid snapshot;
- copying verified manifest/object bytes to another valid local root preserves the Snapshot ID.

For an incremental update, request a new non-decreasing range with `IngestionRequest(parent_snapshot_id=...)` (or the verified parent DTO used by local fixtures). The configured `revision_overlap_sessions` re-requests the latest parent XNYS sessions plus later sessions. An unchanged request can resolve to the existing Snapshot ID without duplicate accepted partitions. A revised overlap action rebuilds the affected suffix in a new snapshot while preserving the parent. If a symbol fails and parent coverage exists, that prior coverage is retained and disclosed; without parent coverage, the symbol contributes no accepted rows.

Incremental planning rejects a range that back-extends or changes the parent start. Treat a range shrink/back-extension as a new full-ingestion request rather than mutating an existing snapshot. The local incremental and fault tests provide reviewed examples:

```bash
uv run pytest tests/integration/test_snapshot_ingestion_faults.py -m integration -q
uv run pytest tests/contract/test_snapshot_publication_contract.py -q
```

## 4. Publication recovery and reconciliation

The storage model uses one publisher lock and same-filesystem staging/final paths. It is recoverable, not a distributed transaction across filesystem and DuckDB.

Expected recovery behavior:

| Interruption | Expected observation | Recovery |
| --- | --- | --- |
| Before checksum/publication | Candidate remains unpublished; prior snapshot remains resolvable. | Retry with the same logical inputs. |
| During staging or CAS promotion | Staging/unreferenced bytes are not snapshot-visible. | Retry or clean stale staging after confirming no complete publication references it. |
| After publication directory rename, before DuckDB commit | A complete publication may exist without an index row. | Recreate the storage adapter and call `FilesystemStore.reconcile()`; it verifies and indexes complete publications. |
| Missing/corrupt published directory/object | The indexed snapshot becomes unavailable/invalid operationally. | Do not repair it in place; use a prior valid Snapshot ID or publish a new snapshot. |
| MLflow/DuckDB terminalization interruption | A run remains discoverable with its assigned Run ID and diagnostics/finalization intent. | Re-run local reconciliation/finalization through the tracker/application recovery path; never edit terminal rows manually. |

An advanced local reconciliation call is intentionally infrastructure-level:

```python
from pathlib import Path

from quant_research_platform.infrastructure.duckdb_metadata import DuckDBMetadataStore
from quant_research_platform.infrastructure.filesystem_store import FilesystemStore

metadata = DuckDBMetadataStore(Path("data/metadata.duckdb"))
try:
    report = FilesystemStore(Path("data"), metadata=metadata).reconcile()
    print("indexed:", report.indexed_snapshot_ids)
    print("unavailable:", report.unavailable_snapshot_ids)
    print("ignored:", report.ignored_publication_ids)
finally:
    metadata.close()
```

Run this only against the intended local root. It does not make corrupt bytes valid and does not rewrite scientific identity. The fault-injection integration tests are the authoritative recovery examples:

```bash
uv run pytest tests/integration/test_snapshot_ingestion_faults.py -m integration -q
```

## 5. Backtest and portfolio assumptions

Select an available snapshot on the **Backtest** page, verify it, resolve configuration in the current process, and click **Run backtest**. The service allocates the platform Run ID before execution, pins exactly one verified Snapshot ID, materializes a snapshot-keyed derived Zipline bundle, runs the engine, audits the core output, evaluates it against SPY from that snapshot, publishes checksummed artifacts, and finalizes the run as succeeded or failed.

The only implemented strategy is `monthly_momentum_v1`:

- the last XNYS session of each calendar month is the signal session;
- the score is adjusted-close return from 252 sessions before to 21 sessions before that signal;
- at least 253 preceding sessions are required before orders are created;
- eligible symbols are ranked by descending score, then ascending ticker on ties;
- up to `position_count` symbols receive equal target weights; no eligible symbol means all cash;
- decisions are visible only on their signal session and use no future data;
- orders execute on the next XNYS session’s open, never on the signal-session close;
- quantities are whole shares; sells occur before buys;
- buys are capped by available cash after adverse slippage and commission;
- positions are long-only and non-negative, cash is non-negative, leverage is capped at 1.0, and cash earns 0%;
- initial portfolio equity is fixed at USD 100,000;
- default costs are 5 bps commission and 10 bps adverse slippage;
- the Zipline bundle receives raw bars and the canonical corporate-action stream exactly once; research-adjusted bars are not supplied as ledger prices.

Missing next-session opens produce unfilled-order diagnostics rather than fabricated fills. A valid run can therefore contain disclosed unfilled orders; invariant failures such as negative cash, invalid quantities, look-ahead chronology, or equity mismatch fail the run and preserve diagnostics.

Run-level tests for these rules are local and deterministic:

```bash
uv run pytest tests/contract/test_zipline_execution_contract.py -q
uv run pytest tests/integration/test_zipline_engine.py -q
uv run pytest tests/properties/test_property_09_no_look_ahead.py tests/properties/test_property_10_execution_accounting.py -q
```

## 6. Evaluation, artifact inspection, and comparison

Evaluation uses strategy returns and adjusted SPY values from the same verified snapshot and aligned XNYS sessions. Any missing SPY session blocks comparison and returns every missing session as an actionable error. When gap-free data exists, the output includes:

- total return, CAGR, annualized volatility, zero-risk-free-rate Sharpe, and maximum drawdown for strategy and SPY;
- strategy-minus-SPY differences for those comparable metrics;
- turnover, total commissions, total slippage, unfilled orders, and ending cash for the strategy;
- checksummed strategy/benchmark returns and equity curves;
- drawdown, monthly returns, positions, orders, fills/transactions, decisions, metrics, and canonical chart-spec artifacts.

`EvaluationService` reads only required projected columns and sessions. UI ordinary tables are bounded to `min(requested_page_size, configured_page_size, 100)`. Full artifacts are opened through a separate verified stream after an explicit download action. Use `page_artifact()` for inspection and `open_artifact()` for a complete verified download; never assume that an unreferenced CAS file is a valid artifact.

The **Runs** page searches by Run ID, Snapshot ID, strategy, state, and bounded page. Terminal runs are immutable: inspect their manifest, non-secret configuration, environment fingerprint, validation report, logs, and artifact checksums; start a new run for a changed input. The **Compare** page accepts 2–10 distinct successful runs, verifies the required artifacts, reports snapshot/configuration/environment differences, retains original metric/range provenance, and aligns only the displayed curves to the common session intersection. It does not silently replace the original metrics.

Representative checks:

```bash
uv run pytest tests/properties/test_property_11_evaluation.py -q
uv run pytest tests/properties/test_property_13_comparison.py -q
uv run pytest tests/integration/test_application_facade.py -m integration -q
```

## 7. What to inspect after an operation

For ingestion, retain:

- `job_state` and operation status;
- Snapshot ID and whether the snapshot was reused;
- provider requests/outcomes and attempts;
- accepted/quarantined/gap counts and reason codes;
- failed symbols and retained parent coverage;
- requested/covered ranges and benchmark comparison readiness;
- manifest, object checksums, configuration checksum, calendar/policy versions, and limitation disclosure.

For a run, retain:

- platform Run ID and terminal state;
- pinned Snapshot ID and evaluation range;
- strategy identifier/parameters and deterministic seed;
- environment/source/dependency fingerprint;
- audit diagnostics, unfilled orders, costs, ending cash, and metrics;
- manifest checksum and every scientific artifact checksum;
- redacted configuration, validation report, logs, and limitation disclosure.

A stable rerun under identical snapshot, non-secret configuration, source/dependency/environment fingerprint, writer/adapter versions, and seed should produce identical scientific output/checksums while operational Run IDs and timestamps remain distinct. This is a reproducibility condition, not a promise of cross-version or cross-platform byte identity.

## Troubleshooting

### Configuration will not resolve

Check that the YAML root is a mapping, the date range is inclusive and ordered, every nested key is known, universe symbols normalize to distinct non-empty values, and any `QRP_` variable is in the explicit allowlist. Run:

```bash
uv run pytest tests/unit/config/test_loader.py -q
```

If the error mentions the project boundary, invoke commands from the repository containing the intended `pyproject.toml`; remove ambiguous nested project boundaries or use an approved absolute path. Do not “fix” an escape by creating directories outside the root.

### Frozen setup or build fails

Confirm Python 3.11 and run:

```bash
uv lock --check
uv sync --frozen --dev --dry-run
uv build --wheel
```

Do not run an unlocked sync as a substitute for reviewing a lock change.

### Ingestion is partial or comparison is not ready

Open the ingestion result and snapshot validation view. A partial result is expected when a provider symbol failed, a row was quarantined, a session is missing, or a symbol is stale. A missing SPY session in the evaluation range blocks comparison. Repair the source request/range or ingest a new snapshot; do not fabricate bars or edit the old snapshot.

### Provider access fails

Normal tests never need network access. For a real run, check date range, symbol spelling, network availability, retry bounds, and yfinance diagnostics. Keep provider attempts bounded. Use the external smoke only when intentionally testing the boundary:

```bash
QRP_RUN_EXTERNAL_TESTS=1 uv run pytest tests/contract/test_yfinance_contract.py -m external
```

A free provider can revise, omit, or reject records. Preserve the failed outcome in the snapshot/run disclosure rather than treating a successful subset as complete evidence.

### Snapshot or artifact checksum fails

Stop using the affected Snapshot ID or artifact. Call the verified inspection path to obtain the actionable integrity error, confirm the local root, and run `FilesystemStore.reconcile()` if the failure followed an interrupted publication. Prior snapshots and unrelated artifacts should remain available. Never overwrite a checksum path or manifest in place.

### A run cannot be changed

This is expected for a terminal run. Run inputs, lifecycle state, metrics, manifest, and artifact links are immutable through platform operations. Resolve a new configuration or create a new Run ID.

### The process stopped during a job

There is no background resume. On restart, inspect the metadata and publication directories, reconcile complete unindexed publications, and retry with identical inputs. A failed/incomplete candidate must not replace the last valid snapshot. Use the fault-injection tests as the expected-state reference.

### Memory or page-bound regression appears

Run the bounded-memory checks:

```bash
uv run pytest tests/integration/test_memory_bounds.py -m memory -q
```

The expected limits are at most 10 symbols per provider batch, at most `write_chunk_rows` rows per canonical write chunk, projected/filtered Parquet reads, and no more than 100 rows in an ordinary UI page. Full artifact download is intentionally separate from page rendering.

### A secret appears in diagnostics

Stop and rotate the exposed credential outside the repository. The redactor covers errors, URLs, headers, structured logs, MLflow fields, manifests, artifacts, progress, and presenters; artifact metadata publication fails closed if a registered secret remains. Check that the value was supplied only through a mapped environment variable and that no generated file was staged for Git.

## Limitations and scientific assumptions

Every snapshot, run, and comparison includes a visible limitation disclosure. Interpret results with these assumptions:

1. **Free-source quality:** yfinance/Yahoo Finance is a free convenience source. Availability, schema, terms, corrections, and completeness can change; the adapter preserves provenance and failures but cannot guarantee institutional feed quality.
2. **Explicit universe and survivorship:** the configured ticker list is current-user supplied. It does not reconstruct historical index membership, delistings, mergers, or symbol changes, so selection and survivorship bias can be material.
3. **Daily US equities only:** XNYS daily sessions are the scheduling basis. Intraday, non-US markets, crypto, forex, options, and shorting are outside this slice. SPY is the required ETF benchmark.
4. **Corporate-action coordinates:** the causal adjusted series is a research signal/benchmark coordinate. The Zipline ledger uses raw action-effective prices and the same canonical action stream once, so positions remain actual equity shares. Dividend timing is limited to provider ex-date/amount fields, and split fractional residual handling follows the pinned engine behavior.
5. **Provider action semantics:** the implementation assumes yfinance split ratios mean new shares per old share and dividends are cash per post-split share on ex-date. A schema/semantic mismatch requires an adapter or policy revision.
6. **Asset status:** accepted snapshot coverage is a pragmatic tradability rule, not authoritative point-in-time listing metadata.
7. **Publication atomicity:** same-filesystem directory rename is atomic for the publication boundary, but filesystem and DuckDB do not form a distributed transaction. Network filesystems and concurrent writers are unsupported.
8. **Stable reruns:** scientific byte identity requires the recorded source/dependency/platform/writer/adapter conditions and deterministic seed. A version change may legitimately produce a new identity.
9. **Local trust boundary:** platform APIs reject mutation and verify checksums, but a user with filesystem/process access can alter local files. Verification detects alteration; it cannot protect raw process memory or out-of-band edits.
10. **Synchronous jobs:** killing the process interrupts work. Reconciliation preserves completed publications, but there is no distributed retry, daemon, or cross-machine resume.
11. **Research, not trading:** commission and slippage are transparent simplified assumptions, not market-impact estimates. Outputs are simulations and provide no broker, paper-trading, or live-execution functionality.

These are scientific disclosures, not investment advice. Do not present a backtest as a forecast or a live-trading recommendation.

## Future Scope and Phase 1 exclusions

The current package deliberately does not include FastAPI or another network API, Celery/RQ or background workers, cloud storage/hosted services, ML/model-training pipelines, LLM features, Alphalens Reloaded, Pyfolio Reloaded, additional strategy/factor libraries, broker/execution adapters, paper trading, live execution, or non-US/intraday/options workflows. The only strategy identifier implemented for this slice is `monthly_momentum_v1`.

The constitution describes later roadmap phases for quantitative research expansion, AI-assisted research, and eventual human-approved execution. Those remain Future_Scope. They must enter through a new requirements/design review and must not be implied by current module names or by the derived Zipline backtest bundle.
