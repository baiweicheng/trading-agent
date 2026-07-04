"""Command line entry point for the trading agent scaffold."""

from __future__ import annotations

from trading_agent import __version__


def main() -> None:
    """Print basic package information for the scaffold."""
    print(f"trading-agent {__version__}")
    print("Local-first paper trading agent scaffold.")


if __name__ == "__main__":
    main()
