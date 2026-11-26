"""Load the CPU smoke harness or the iteration-level replay configuration."""

import math
from pathlib import Path

import yaml

from .experiment import VARIANTS
from .replay import ReplaySettings, StepTiming


def _positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _bounds(shape):
    for name in ("input_tokens", "output_tokens"):
        bounds = shape.get(name)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(v) is not int for v in bounds)
            or not 1 <= bounds[0] <= bounds[1]
        ):
            raise ValueError(f"workload.{name} must be [positive minimum, maximum]")


def load_config(path: str | Path, *, scenario: str | None = None) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    backend = config.get("backend")
    if backend not in {"mock", "replay"}:
        raise ValueError("backend must be mock or replay")
    if type(config.get("seed")) is not int or not _positive(config.get("deadline_budget_s")):
        raise ValueError("expected an integer seed and positive deadline_budget_s")
    workload = config["workload"]
    scenarios = config.pop("scenarios", {})
    if scenario is not None:
        if backend != "replay" or scenario not in scenarios:
            raise ValueError(f"unknown replay scenario: {scenario}")
        workload.update(scenarios[scenario])
        config["scenario"] = scenario
    if backend == "mock":
        if config.get("policy") not in {"fifo", "edf"}:
            raise ValueError("mock policy must be fifo or edf")
        if type(config.get("concurrency")) is not int or config["concurrency"] < 1:
            raise ValueError("concurrency must be a positive integer")
        for name in ("prefill_s", "token_s"):
            if config.get("mock", {}).get(name, -1) < 0:
                raise ValueError(f"mock.{name} must be nonnegative")
    else:
        if config.get("policy") not in {"compare", *VARIANTS}:
            raise ValueError("unknown replay comparison")
        ReplaySettings(**config["replay"])
        for mode in ("high", "low"):
            StepTiming(**config["timing"][mode])
        if not config["timing"].get("source"):
            raise ValueError("identify the timing source")
        if workload.get("source") == "burstgpt":
            return config
        if workload.get("source", "synthetic") != "synthetic":
            raise ValueError("unknown workload source")
        if (
            type(config.get("calibration_seed")) is not int
            or type(config.get("calibration_size")) is not int
            or config["calibration_size"] < 1
        ):
            raise ValueError(
                "synthetic replay needs a calibration seed and positive calibration_size"
            )
    if "classes" in workload:
        if not workload["classes"]:
            raise ValueError("expected at least one workload class")
        for item in workload["classes"]:
            _bounds(item)
            if not _positive(item.get("weight")) or not _positive(
                item.get("deadline_budget_s", config["deadline_budget_s"])
            ):
                raise ValueError("workload classes require positive weights and deadline budgets")
    else:
        _bounds(workload)
    if not workload.get("phases"):
        raise ValueError("at least one workload phase is required")
    for phase in workload["phases"]:
        if (
            not _positive(phase["duration_s"])
            or not math.isfinite(phase["rate_rps"])
            or phase["rate_rps"] < 0
        ):
            raise ValueError("phase duration must be positive and rate nonnegative")
    return config
