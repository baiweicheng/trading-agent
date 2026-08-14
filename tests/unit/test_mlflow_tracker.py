# ruff: noqa: E501

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from uuid import UUID

from quant_research_platform.infrastructure.mlflow_tracker import (
    LocalMlflowTracker,
    RunInputs,
)


class FakeClient:
    def __init__(self) -> None:
        self.params: list[tuple[str, str, object]] = []
        self.metrics: list[tuple[str, str, float]] = []
        self.tags: list[tuple[str, str, object]] = []
        self.text: list[tuple[str, str, str]] = []
        self.terminated: list[tuple[str, str]] = []
        self.runs = 0

    def get_experiment_by_name(self, name: str) -> None:
        del name
        return None

    def create_experiment(self, name: str) -> str:
        assert name == "quant_research_platform"
        return "experiment-1"

    def create_run(self, experiment_id: str, **kwargs: object) -> SimpleNamespace:
        assert experiment_id == "experiment-1"
        assert kwargs["tags"]["qrp.state"] == "running"  # type: ignore[index]
        self.runs += 1
        return SimpleNamespace(info=SimpleNamespace(run_id=f"mlflow-{self.runs}"))

    def log_param(self, run_id: str, key: str, value: object) -> None:
        self.params.append((run_id, key, value))

    def log_metric(self, run_id: str, key: str, value: float, **kwargs: object) -> None:
        del kwargs
        self.metrics.append((run_id, key, value))

    def set_tag(self, run_id: str, key: str, value: object) -> None:
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, **kwargs: object) -> None:
        self.terminated.append((run_id, str(kwargs["status"])))

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.text.append((run_id, text, artifact_file))


def test_allocation_logs_only_redacted_scalar_inputs_and_local_references() -> None:
    client = FakeClient()
    tracker = LocalMlflowTracker(client=client)
    run_id = UUID("00000000-0000-0000-0000-000000000001")

    handle = tracker.allocate_run(
        run_id=run_id,
        snapshot_id="snap_" + "a" * 64,
        strategy_id="monthly_momentum_v1",
        strategy_parameters={"position_count": 3, "proxy": "https://user:password@example.invalid"},
        evaluation_start=date(2024, 1, 2),
        evaluation_end=date(2024, 1, 31),
        universe=("AAPL", "MSFT"),
        configuration={
            "endpoint": "https://user:password@example.invalid",
            "secrets": {"https_proxy": "https://user:password@example.invalid"},
        },
        secret_values=("https://user:password@example.invalid", "password"),
    )

    assert handle.mlflow_run_id == "mlflow-1"
    assert all("password" not in str(item) for item in client.params)
    assert all("password" not in str(item) for item in client.text)
    assert any(key == "strategy.proxy" and value == "[REDACTED]" for _, key, value in client.params)
    assert not any("log_artifact" in name for name in dir(client))


def test_run_inputs_accepts_platform_and_run_id_aliases() -> None:
    identifier = UUID("00000000-0000-0000-0000-000000000002")
    first = RunInputs(
        platform_run_id=identifier,
        snapshot_id="snap_" + "b" * 64,
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    second = RunInputs(
        run_id=identifier,
        snapshot_id="snap_" + "b" * 64,
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    assert first.run_id == second.platform_run_id == identifier
