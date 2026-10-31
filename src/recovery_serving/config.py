"""Load a small, explicit experiment configuration."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if config.get("backend") != "mock":
        raise ValueError("Only the mock backend is implemented; see docs/architecture.md")
    if config.get("policy") not in {"fifo", "edf"}:
        raise ValueError("policy must be fifo or edf")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if not isinstance(config.get("concurrency"), int) or config["concurrency"] < 1:
        raise ValueError("concurrency must be a positive integer")
    if config.get("deadline_budget_s", 0) <= 0:
        raise ValueError("deadline_budget_s must be positive")
    for name in ("prefill_s", "token_s"):
        if config.get("mock", {}).get(name, -1) < 0:
            raise ValueError(f"mock.{name} must be nonnegative")
    workload = config.get("workload", {})
    for name in ("input_tokens", "output_tokens"):
        bounds = workload.get(name)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(not isinstance(value, int) for value in bounds)
            or not 1 <= bounds[0] <= bounds[1]
        ):
            raise ValueError(f"workload.{name} must be [positive minimum, maximum]")
    if not workload.get("phases"):
        raise ValueError("At least one workload phase is required")
    for phase in workload["phases"]:
        if phase["duration_s"] <= 0 or phase["rate_rps"] < 0:
            raise ValueError("Phase duration must be positive and request rate nonnegative")
    return config
