# Developer Guide

This guide describes the supported local development path for Phase 1. It intentionally documents the repository as implemented: one Python process, synchronous application services, local files/databases, and a Streamlit presentation adapter. There is no required server, worker, queue, cloud account, or external service for the normal test suite.

## Runtime and frozen environment

Use Python 3.11 only. The supported project range is `>=3.11,<3.12`; the range is declared in [`pyproject.toml`](../pyproject.toml) and mirrored by [`uv.lock`](../uv.lock). From the project root:

```bash
uv sync --frozen
uv lock --check
uv sync --frozen --dev --dry-run
```

`uv sync --frozen` must not rewrite the lock file. If the lock is intentionally changed, review the dependency change and regenerate it in a deliberate change, then verify it again with `uv lock --check`. The direct runtime stack is deliberately limited to Pydantic, ruamel.yaml, yfinance, exchange-calendars, PyArrow/Parquet, DuckDB, Zipline Reloaded, MLflow, Streamlit, pandas, and NumPy. Development tools are Pytest, Hypothesis, Ruff, mypy, and coverage.

Build and import the wheel without using a development server:

```bash
uv build --wheel
uv run python -c "import quant_research_platform; print(quant_research_platform.__file__)"
```

## Repository and local-data boundaries

The **Project_Root** is the directory containing the package `pyproject.toml`. Configuration resolution requires an unambiguous project boundary. Relative paths are resolved against that root, normalized without creating directories first, and rejected if they escape it. Absolute paths are allowed and normalized as absolute paths. This prevents a YAML document such as `../outside` from silently redirecting local state.

Tracked source and reviewed fixtures belong in `src/`, `config/default.yaml`, `docs/`, and `tests/golden/`. The following local state is generated and ignored:

| Path | Ownership and use |
| --- | --- |
| `data/raw/` | Raw provider records, before normalization. |
| `data/normalized/` | Accepted canonical daily bars. Normalized partitions are symbol/session-year scoped. |
| `data/quarantine/` | Rejected rows, reason codes, and offending-value details. |
| `data/staging/` | Operation-private candidates. Readers must never resolve staging paths. |
| `data/objects/` | Content-addressed snapshot and validation objects. Bytes are verified by SHA-256 before publication/use. |
| `data/snapshots/` | Published snapshot directories and immutable manifests. |
| `data/runs/` | Local run publication/operational records. |
| `data/artifacts/` | Checksummed evaluation, comparison, chart-spec, and diagnostic artifacts. |
| `data/zipline-bundles/` | Disposable, snapshot-keyed derived Zipline bundles. |
| `data/metadata.duckdb` | DuckDB index for requests, objects, snapshots, runs, artifacts, jobs, and reconciliation state. |
| `data/mlflow.db` | Local MLflow SQLite catalog. MLflow stores compact redacted values and references; it is not the authoritative table store. |
| `data/logs/` | Sanitized structured JSONL operational logs. |

The repository may contain an ignored `mlruns/` directory from older local experiments; the Phase 1 composition root uses `data/mlflow.db` instead. Do not commit either catalog or generated data.

## Configuration

[`config/default.yaml`](../config/default.yaml) is a safe starting document. It contains no credentials and currently requests `2015-01-01` through `2024-12-31` for the default universe `AAPL`, `JPM`, `MSFT`, `PG`, and `XOM`, with `SPY` as the separate benchmark. The important defaults are:

- provider batch size: 5 symbols, bounded to 1–10;
- retry policy: 3 attempts, 1-second initial delay, 8-second maximum delay, multiplier 2.0;
- staleness threshold: 1 XNYS session;
- revision overlap: 5 XNYS sessions;
- write chunk: 50,000 rows, bounded to 1–100,000;
- strategy: `monthly_momentum_v1`, 252-session lookback, 21-session skip, up to 5 positions;
- initial equity: fixed at USD 100,000;
- commission/slippage defaults: 5/10 basis points;
- UI page size: 100 rows maximum;
- deterministic seed: 0.

A minimal custom document must still provide `data.requested_range`; other fields use documented defaults:

```yaml
data:
  requested_range:
    start: "2023-01-03"
    end: "2024-02-02"
  universe:
    - AAPL
    - MSFT
  batch_size: 2
strategy:
  position_count: 1
```

The configuration resolver applies leaf-wise precedence in this order:

1. documented defaults;
2. the supplied YAML mapping (including values edited in the Streamlit form);
3. explicitly mapped environment variables.

Streamlit form values are merged into the effective YAML document before resolution; they are not a fourth precedence tier. Environment variables remain highest precedence. Unknown YAML keys, duplicate nested keys, unsafe YAML tags, non-mapping roots, invalid types/bounds, invalid date order, duplicate normalized symbols, and unmapped `QRP_` variables return actionable errors before an application operation begins.

Only variables in the explicit allowlist in [`config/loader.py`](../src/quant_research_platform/config/loader.py) are accepted. Examples without credentials:

```bash
export QRP_DATA__BATCH_SIZE=3
export QRP_DATA__WRITE_CHUNK_ROWS=10000
export QRP_RUNTIME__DETERMINISTIC_SEED=7
# A real proxy value belongs only in the local environment, never in docs:
# export QRP_SECRETS__HTTPS_PROXY='<proxy-url>'
```

`QRP_SECRETS__HTTP_PROXY` and `QRP_SECRETS__HTTPS_PROXY` are the supported secret fields. Plain tracked YAML cannot supply a credential. The default path field `paths.local_secrets_file` points to `config/secrets.local.yaml`, and that filename is ignored by Git; the current public resolver accepts secret values through explicitly mapped environment variables, so do not assume that merely creating the file loads it. If a local-secret integration is added later, it must preserve the same explicit mapping and redaction rules.

Resolved configuration is frozen. Durable records and UI presenters receive the `NonSecretConfig` projection: ordinary values remain visible, while secret fields expose only presence state or `[REDACTED]`. Canonical serialization is UTF-8, LF-terminated, deterministic YAML in schema order. Never place a real credential, token, proxy URL, or generated absolute local path in README, docs, fixtures, logs, manifests, MLflow parameters, or screenshots.

## Tests and quality commands

The repository defines the `integration`, `external`, `memory`, and `smoke` markers in [`pyproject.toml`](../pyproject.toml). Tests use temporary roots and local fixtures unless explicitly described otherwise. Useful single-shot commands are:

```bash
# Full offline suite; the external-provider test is excluded.
uv run pytest -m "not external"

# Focused setup and local composition checks.
uv run pytest tests/smoke -m smoke
uv run pytest tests/smoke/test_project_metadata.py
uv run pytest tests/smoke/test_local_phase1_smoke.py

# Named local vertical slice and recovery evidence.
uv run pytest tests/integration/test_phase1_pipeline.py -m integration
uv run pytest tests/integration/test_snapshot_ingestion_faults.py -m integration
uv run pytest tests/integration/test_streamlit_apptest.py -m integration

# Variable-input correctness properties.
uv run pytest tests/properties

# Bounded-memory behavior; this is still local and deterministic.
uv run pytest tests/integration/test_memory_bounds.py -m memory

# Static/format checks.
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run coverage run -m pytest -m "not external"
uv run coverage report
```

The Hypothesis profile in the project metadata runs at least 100 examples for each property test. Property tests use local providers/fakes and never contact Yahoo Finance. The memory tests inspect batch sizes, chunk sizes, projected columns, predicates, and page limits rather than asserting a machine-specific RSS value.

### External provider smoke opt-in

The external seam is exactly one short SPY request in [`tests/contract/test_yfinance_contract.py`](../tests/contract/test_yfinance_contract.py). It is marked `smoke` and `external`, is skipped by default, and does not assert provider price values. Enable it only when network access is intentional:

```bash
QRP_RUN_EXTERNAL_TESTS=1 uv run pytest tests/contract/test_yfinance_contract.py -m external
```

It is not part of property tests, ordinary offline validation, or the fixture pipeline. A provider failure should be reported as an external-boundary diagnostic, not “fixed” by changing scientific fixtures.

## Streamlit workflow

Launch the app manually only when you want a local UI:

```bash
uv run streamlit run src/quant_research_platform/ui/app.py
```

Importing `quant_research_platform.ui.app` starts no server. In tests, `main()` is called through Streamlit `AppTest`, so no watcher or development server is required. The composition root uses the process cache for concrete adapters; a process restart intentionally requires configuration resolution again.

The implemented navigation is:

1. **Configure / Ingest** — choose `config/default.yaml` or upload a YAML document, edit approved non-secret controls, resolve configuration, and enable ingestion only after validation succeeds. Ingestion is synchronous; progress, sanitized warnings, the resulting Snapshot ID, partial-success state, and prior records remain visible.
2. **Snapshots** — list bounded available snapshots, inspect one manifest after checksum verification, review provenance, covered range, validation counts, quarantine/gap/staleness facts, benchmark readiness, and page or explicitly download referenced artifacts.
3. **Backtest** — select one available Snapshot ID, verify it, run the synchronous backtest, and inspect strategy/SPY metrics, differences, equity/drawdown/monthly tables, costs, positions, transactions, unfilled orders, and verified artifacts.
4. **Runs** — search bounded run summaries, inspect a run manifest/configuration/fingerprint/validation/log view, and separately page or download verified artifacts.
5. **Compare** — select 2–10 successful runs in order. The page reports snapshot/configuration/environment differences first, preserves each original evaluation range, aligns displayed curves to their common session intersection, and exposes a checksummed comparison artifact.

Ordinary tables are server-bounded to the configured page size and never exceed 100 rows. Full artifact access is a separate explicit checksum-verified stream; selecting a table does not materialize the full artifact.

## Public application facade

Presentation code and local callers should use `ResearchApplication` from [`application/services.py`](../src/quant_research_platform/application/services.py), not direct UI state or ad hoc database SQL. The main operations are:

- `resolve_configuration(...)` → a redacted view plus an opaque process-local `ConfigurationHandle`;
- `ingest(IngestionRequest(), handle, progress=...)`;
- `list_snapshots(SnapshotQuery(...))` and `inspect_snapshot(snapshot_id)`;
- `run_backtest(BacktestRequest(snapshot_id, evaluation_range), handle, progress=...)`;
- `search_runs(RunQuery(...))` and `inspect_run(run_id)`;
- `compare_runs((run_id_a, run_id_b, ...))` for 2–10 successful runs;
- `page_artifact(checksum, page=..., page_size=...)` and `open_artifact(checksum)`.

Expected provider, validation, storage, checksum, and configuration failures return `Err` with sanitized `ActionableError` values. A handle is not a serialized configuration and cannot expose secret values; after process restart or explicit invalidation, resolve again.

## Architecture references

The package is organized as `domain`, `config`, `application`, `infrastructure`, and `ui`. The [accepted ADR](decisions/0001-phase1-local-stack.md) explains why platform Parquet/CAS is authoritative, DuckDB and MLflow are catalogs, Zipline bundles are derived caches, and network APIs/queues/ML/LLM/execution integrations are excluded from this slice.
