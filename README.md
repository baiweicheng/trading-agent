# Quantitative Research Platform

Phase 1 is a local, single-developer research loop for reproducible US-equity research. It resolves validated configuration, acquires free daily data through the narrow yfinance adapter, preserves raw and normalized Parquet/CAS content, indexes local metadata in DuckDB, runs one monthly long-only momentum baseline through Zipline Reloaded, evaluates it against SPY, records local runs with MLflow/SQLite, and exposes the workflow through Streamlit.

This is research software, not a broker, trading bot, or investment recommendation. Read the [research-operations guide](docs/research-operations.md) before treating a result as evidence.

## Start here

- [Developer guide](docs/developer-guide.md): Python 3.11 setup, frozen `uv` workflow, configuration precedence, paths, secrets, tests, quality checks, and Streamlit launch.
- [Research operations guide](docs/research-operations.md): ingestion, immutable snapshots, incremental updates, recovery/reconciliation, backtests, evaluation, comparison, artifact inspection, assumptions, and troubleshooting.
- [ADR 0001: Phase 1 local stack](docs/decisions/0001-phase1-local-stack.md): technology boundaries, storage authority, and exclusions.
- [Default configuration](config/default.yaml): a safe, credential-free YAML starting point.

## Quick start

Use Python 3.11 (`>=3.11,<3.12`). Install [uv](https://docs.astral.sh/uv/) and synchronize the reviewed environment without changing the lock file:

```bash
uv sync --frozen
uv run pytest -m "not external"
uv run ruff check src tests
uv run mypy src
```

The normal test command is offline. The only network test is an explicitly gated, one-batch yfinance smoke test; see the [developer guide](docs/developer-guide.md#external-provider-smoke-opt-in).

To use the visual workflow manually after setup:

```bash
uv run streamlit run src/quant_research_platform/ui/app.py
```

The app validates configuration before enabling ingestion, then provides Configure / Ingest, Snapshots, Backtest, Runs, and Compare pages. Streamlit is only a presentation adapter; application behavior is exposed by the typed `ResearchApplication` facade.

## Local state and source control

The project root is the directory containing the package `pyproject.toml`. Relative configured paths are normalized against that root and cannot escape it; an absolute path is preserved after normalization. Generated local state is kept below `data/` and ignored by Git. The important runtime areas are:

```text
data/
  raw/                 raw provider records as partitioned Parquet
  normalized/         accepted daily bars as partitioned Parquet
  quarantine/         rejected records and reason details
  staging/             unpublished operation candidates
  objects/             content-addressed snapshot objects
  snapshots/          published immutable snapshot manifests
  runs/               local run records and manifest references
  artifacts/          checksummed tabular/chart artifacts
  zipline-bundles/    disposable snapshot-keyed derived bundles
data/metadata.duckdb  local metadata/index database
data/mlflow.db        local MLflow SQLite tracking catalog
data/logs/            sanitized operational JSONL logs
```

Parquet and content-addressed objects are the scientific source of truth. DuckDB indexes metadata and supports filtered discovery; MLflow catalogs run lifecycle and compact references. A Zipline bundle is derived cache state and may be rebuilt from a verified snapshot. Do not edit published snapshots, manifests, or terminal runs in place; publish a new snapshot or start a new run.

## Phase 1 boundary

This slice intentionally covers only configuration, free-source daily ingestion, XNYS session-aware normalization and validation, quarantine/gaps/staleness, immutable snapshots, the interpretable `monthly_momentum_v1` baseline, conservative next-session whole-share simulation, SPY evaluation, local experiment tracking, bounded run comparison/inspection, and the Streamlit workflow. It excludes network APIs, background queues/workers, cloud services, machine learning, LLM capabilities, additional factor/strategy libraries, Alphalens/Pyfolio, broker adapters, paper trading, and live execution. These are Future_Scope and require a separate reviewed specification.

Every snapshot, run, and comparison carries a limitation disclosure covering free-source quality, explicit-universe and survivorship limitations, recorded failures, cost assumptions, and execution assumptions. See the [research-operations guide](docs/research-operations.md#limitations-and-scientific-assumptions) for the full disclosure.
