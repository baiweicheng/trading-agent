# Trading Agent

Trading Agent is a local-first, paper-trading-only MVP for building and testing a deterministic trading system before any live broker integration exists.

This scaffold does not connect to a broker, database, network service, or LLM. It only establishes the package structure and developer commands for the next implementation steps.

## Safety

Version 1 is paper trading only. There is no live broker integration and no path for an LLM to place trades. Future trading behavior should keep risk checks, audit logs, and human approval ahead of order execution.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run the CLI help:

```bash
python -m trading_agent.cli
```

After installing the package, the console script is also available:

```bash
trading-agent
```
