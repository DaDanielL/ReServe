"""Run all admission policies on one frozen trace and one observation window."""

import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .metrics import summarize_replay
from .replay import Replay, ReplaySettings, StepTiming
from .scheduling import LengthPredictor, RecoveryAwareScheduler
from .workload import (
    burstgpt_trace,
    generate_trace,
    peak_window,
    steady_control,
    synthetic_calibration,
    trace_lines,
)

VARIANTS = {
    "fixed_high": ("high", "fifo"),
    "fixed_low": ("low", "fifo"),
    "pressure": ("dynamic", "fifo"),
    "cost": ("dynamic", "cost"),
    "drain": ("dynamic", "drain"),
    "recovery": ("dynamic", "recovery"),
}


def prepare(config):
    workload = config["workload"]
    if workload.get("source", "synthetic") == "burstgpt":
        trace, calibration, window, meta = burstgpt_trace(
            Path(workload["path"]), workload["burstgpt"], config["deadline_budget_s"]
        )
        bursts = peak_window(trace, window, workload["burst_bin_s"], workload["burst_factor"])
    else:
        seed = config["calibration_seed"]
        if seed == config["seed"]:
            raise ValueError("use a calibration seed distinct from the replay seed")
        trace = generate_trace(config)
        calibration = synthetic_calibration(config, seed, config["calibration_size"])
        meta = {"source": "synthetic", "seed": config["seed"], "calibration_seed": seed}
        window, bursts = 0.0, []
        for phase in workload["phases"]:
            end = window + phase["duration_s"]
            if phase["name"] == "burst":
                bursts.append((window, end))
            window = end
    if config.get("steady_control"):
        trace, bursts = steady_control(trace, window), []
    return trace, calibration, window, bursts, meta


def run_experiment(config: dict, output: Path, environment: dict) -> dict:
    s = ReplaySettings(**config["replay"])
    timings = {m: StepTiming(**config["timing"][m]) for m in ("high", "low")}
    trace, calibration, window, bursts, source = prepare(config)
    scale = config.get("prediction_scale", 1.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("prediction_scale must be finite and positive")
    estimates = ((p, min(s.max_new_tokens, max(1, round(n * scale)))) for p, n in calibration)
    predictor = LengthPredictor(estimates, s.quantile)
    policy = config["policy"]
    variants = VARIANTS if policy == "compare" else {policy: VARIANTS[policy]}
    horizon = window + s.followup_s
    raw = trace_lines(trace)
    cal = json.dumps(calibration, separators=(",", ":")) + "\n"
    metadata = {
        "measurement_kind": "cpu_timing_simulation",
        "timing_source": config["timing"]["source"],
        "config": config,
        "environment": environment,
        "workload": source,
        "workload_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "calibration_sha256": hashlib.sha256(cal.encode()).hexdigest(),
        "calibration_requests": len(calibration),
        "arrival_window_s": window,
        "observation_window_s": horizon,
        "offline_bursts_s": bursts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".replay-") as tmp:
        staging = Path(tmp) / "run"
        staging.mkdir()
        (staging / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (staging / "workload.jsonl").write_text(raw)
        (staging / "calibration.json").write_text(cal)
        summaries = {}
        for name, (mode, admission) in variants.items():
            folder = staging / name
            folder.mkdir()
            scheduler = RecoveryAwareScheduler(predictor, s.lookahead, s.max_wait_s)
            with (folder / "events.jsonl").open("w") as events:

                def emit(event):
                    events.write(json.dumps(event, separators=(",", ":")) + "\n")

                result = Replay(trace, horizon, s, timings, scheduler, mode, admission, emit).run()
            with (folder / "requests.csv").open("w", newline="") as handle:
                # An empty cohort still produces a CSV with the normal schema.
                writer = csv.DictWriter(handle, fieldnames=REQUEST_COLUMNS)
                writer.writeheader()
                writer.writerows(result.rows)
            summaries[name] = summarize_replay(result, horizon, bursts, s, mode)
            (folder / "summary.json").write_text(json.dumps(summaries[name], indent=2) + "\n")
        (staging / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return summaries


REQUEST_COLUMNS = (
    "request_id",
    "arrival_s",
    "input_tokens",
    "output_tokens",
    "deadline_s",
    "phase",
    "status",
    "reason",
    "started_s",
    "finished_s",
    "generated_tokens",
    "low_tokens",
    "queue_s",
    "observed_wait_s",
    "ttft_s",
    "e2e_s",
    "deadline_met",
)
