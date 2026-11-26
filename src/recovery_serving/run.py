"""Run the open-loop CPU smoke harness or an iteration-level scheduling replay."""

import argparse
import asyncio
import csv
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from .backends.mock import MockBackend
from .config import load_config
from .experiment import VARIANTS, run_experiment
from .metrics import summarize
from .scheduling import priority
from .workload import generate_trace


def environment() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in ("recovery-serving", "PyYAML", "torch", "transformers", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": packages,
    }


async def run(config: dict, output: Path) -> dict:
    if config["backend"] == "replay":
        return run_experiment(config, output, environment())
    # Refuse to overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=False)
    trace = generate_trace(config)
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": config,
                "environment": environment(),
                "measurement_kind": "artificial_setup_check",
                "limitations": "Mock backend; client admission only; no precision recovery implemented",
            },
            indent=2,
        )
        + "\n"
    )
    (output / "workload.jsonl").write_text(
        "".join(json.dumps(request.to_dict()) + "\n" for request in trace)
    )
    backend = MockBackend(**config["mock"])
    queue = asyncio.PriorityQueue()
    rows = []
    started = perf_counter()
    with (output / "events.jsonl").open("w") as events:

        def event(kind: str, request_id: str):
            events.write(
                json.dumps(
                    {
                        "event": kind,
                        "request_id": request_id,
                        "t_s": perf_counter() - started,
                        "backend": "mock",
                    }
                )
                + "\n"
            )
            events.flush()

        async def worker():
            while True:
                _, _, request = await queue.get()
                try:
                    if request is None:
                        return
                    service_start = perf_counter() - started
                    event("request_started", request.request_id)
                    row = {**request.to_dict(), "backend": "mock", "started_s": service_start}
                    try:
                        result = await backend.generate(request)
                        finished = result.finished_at - started
                        row.update(
                            status="ok",
                            finished_s=finished,
                            ttft_s=result.first_token_at - started - request.arrival_s,
                            e2e_s=finished - request.arrival_s,
                            queue_s=service_start - request.arrival_s,
                            deadline_met=finished <= request.deadline_s,
                            error="",
                        )
                        event("request_finished", request.request_id)
                    except Exception as error:
                        row.update(
                            status="error",
                            finished_s=perf_counter() - started,
                            ttft_s=None,
                            e2e_s=None,
                            queue_s=service_start - request.arrival_s,
                            deadline_met=False,
                            error=str(error),
                        )
                        event("request_failed", request.request_id)
                    rows.append(row)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(config["concurrency"])]
        for index, request in enumerate(trace):
            await asyncio.sleep(max(0.0, started + request.arrival_s - perf_counter()))
            event("request_arrived", request.request_id)
            queue.put_nowait((priority(request, config["policy"]), index, request))
        # Include the complete planned arrival window, even if its final phase has no requests.
        window = sum(phase["duration_s"] for phase in config["workload"]["phases"])
        await asyncio.sleep(max(0.0, started + window - perf_counter()))
        await queue.join()
        for index in range(len(workers)):
            queue.put_nowait((float("inf"), index, None))
        await asyncio.gather(*workers)
    duration_s = perf_counter() - started
    rows.sort(key=lambda row: row["arrival_s"])
    if rows:
        with (output / "requests.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (output / "requests.csv").write_text("request_id,status\n")
    summary = summarize(rows, duration_s)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", help="select a named workload variant from the replay config")
    parser.add_argument("--seed", type=int, help="override the evaluation trace seed")
    parser.add_argument(
        "--policy", choices=["compare", *VARIANTS], help="run a replay comparison or one variant"
    )
    parser.add_argument(
        "--steady-control", action="store_true", help="evenly space the same replay requests"
    )
    parser.add_argument("--burstgpt", type=Path, help="override the CSV path in a BurstGPT config")
    parser.add_argument(
        "--prediction-scale", type=float, help="scale calibration lengths to test prediction error"
    )
    parser.add_argument(
        "--deadline-budget-s", type=float, help="default deadline budget from arrival"
    )
    parser.add_argument("--restore-s", type=float, help="override simulated weight-copy duration")
    args = parser.parse_args()
    config = load_config(args.config, scenario=args.scenario)
    if (
        args.policy
        or args.steady_control
        or args.burstgpt
        or args.prediction_scale is not None
        or args.restore_s is not None
    ) and config["backend"] != "replay":
        parser.error("these comparison options require a replay config")
    if args.deadline_budget_s is not None:
        if not math.isfinite(args.deadline_budget_s) or args.deadline_budget_s <= 0:
            parser.error("--deadline-budget-s must be finite and positive")
        config["deadline_budget_s"] = args.deadline_budget_s
    if args.prediction_scale is not None:
        config["prediction_scale"] = args.prediction_scale
    if args.restore_s is not None:
        config["replay"]["restore_s"] = args.restore_s
    if args.seed is not None:
        config["seed"] = args.seed
    if args.policy:
        config["policy"] = args.policy
    if args.steady_control:
        config["steady_control"] = True
    if args.burstgpt:
        if config["workload"].get("source") != "burstgpt":
            parser.error("--burstgpt requires a BurstGPT workload")
        config["workload"]["path"] = str(args.burstgpt)
    summary = asyncio.run(run(config, args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
