"""Run ReServe's scheduler experiments from a checkout of frank-scheduler.

From the repository root, run python scripts/run_reserve_experiments.py pilot.
Use --plan to print the commands or --help for settings. Run stages sequentially.
A comparison lasts until all its policies/repetitions have finished. Completed
comparisons resume on rerun. The repository's live runner restarts an interrupted
comparison from its beginning. Using a different --out keeps old results intact.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LIVE = ["-m", "recovery_serving.live"]
REPLAY = ["-m", "recovery_serving.run"]
STAGES = ("pilot", "replay", "gpu", "extended")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def case_signature(repo):
    # Include tracked local modifications so a resume cannot silently mix code.
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"])
    return {
        "commit": git(repo, "rev-parse", "HEAD"),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def launch(opts):
    repo = opts.repo.resolve()
    if not (repo / "src/recovery_serving/live.py").is_file():
        raise SystemExit("Run from the ReServe frank-scheduler checkout or pass --repo PATH.")
    out = (repo / opts.out).resolve()
    data = (repo / "data/live").resolve()
    code = case_signature(repo)
    env = dict(
        os.environ, PYTHONPATH=str(repo / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    phases = STAGES if opts.stage == "all" else [opts.stage]
    gpu_info = None
    if not opts.plan and any(phase in phases for phase in ("gpu", "pilot", "extended")):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable to this Python. Check nvidia-smi and the PyTorch build."
            )
        gpu_info = [torch.cuda.get_device_name(0), str(torch.__version__), str(torch.version.cuda)]
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            "| PyTorch:",
            torch.__version__,
            "| CUDA:",
            torch.version.cuda,
        )
    print("Source commit:", code["commit"], "| Output:", out, flush=True)

    def run_case(name, args, *, live=False, figs=False):
        target = out / name
        cmd = [sys.executable, *map(str, args), "--output", str(target)]
        print(f"\n[{name}] {subprocess.list2cmdline(cmd)}", flush=True)
        if opts.plan:
            return target
        signature = {**code, "args": cmd[1:]}
        if live:
            signature["gpu"] = gpu_info
        # Verify the source budget file when replaying frozen deadlines across loads.
        if "--budgets" in cmd:
            budget = Path(cmd[cmd.index("--budgets") + 1])
            signature["budget_sha256"] = hashlib.sha256(budget.read_bytes()).hexdigest()
        marker = out / "_commands" / (name + ".json")
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != signature:
                raise SystemExit(
                    f"{name} has different settings or source. Use a fresh --out directory."
                )
        elif target.exists():
            raise SystemExit(
                f"{target} already exists without a matching command. Choose a fresh --out."
            )
        else:
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(signature, stream, indent=2)
        if target.exists():
            if not all((target / f).is_file() for f in ("summary.json", "run.json")):
                raise SystemExit(f"Incomplete existing result at {target}. Choose a fresh --out.")
            print("Reusing completed comparison.", flush=True)
        elif live:
            subprocess.run(cmd, cwd=repo, env=env, check=True)
        else:
            with marker.with_suffix(".log").open("w", encoding="utf-8") as output_log:
                subprocess.run(cmd, cwd=repo, env=env, check=True, stdout=output_log)
        if live:
            write_live_csv(target)
            if figs:
                write_live_plots(target)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            for label, stats in sorted(summary.items()):
                if label.startswith("rep-01"):
                    print(
                        f"  {label}: {stats['natural_eos']}/{stats['requests']} natural completions, "
                        f"{stats['length_capped']} capped, {stats['low_switches']} high-to-low switches."
                    )
        return target

    high = opts.high_kv_tokens
    cap = opts.max_new_tokens
    live = LIVE + [
        "--device",
        "cuda:0",
        "--dtype",
        "float16",
        "--data-dir",
        data,
        "--seed",
        opts.seed,
        "--max-prompt-tokens",
        512,
    ]
    if opts.model_dir:
        live += ["--model-dir", opts.model_dir.resolve()]
    jit = live + ["--source", "jitserve", "--max-new-tokens", cap, "--high-kv-tokens", high]
    full = ["--calibration", 24, "--requests", 24, "--repeats", opts.repeats]
    seed_tag = f"s{opts.seed}"
    if "pilot" in phases:
        # Both pilot cohorts are prefixes of the eventual 24 calibration prompts.
        # The evaluation prompts and arrival window are held out from tuning.
        run_case(
            "gpu-wiring-" + seed_tag,
            live
            + [
                "--source",
                "jitserve",
                "--burst-start",
                28800,
                "--calibration",
                2,
                "--requests",
                3,
                "--max-new-tokens",
                4,
                "--down-queue",
                1,
                "--up-queue",
                0,
                "--variants",
                "fixed_high",
                "recovery",
            ],
            live=True,
        )
        run_case(
            "gpu-pilot-" + seed_tag,
            jit
            + [
                "--burst-start",
                28800,
                "--arrival-scale",
                opts.scale,
                "--calibration",
                6,
                "--requests",
                8,
                "--variants",
                "fixed_high",
                "cost",
                "recovery",
            ],
            live=True,
            figs=True,
        )

    if "replay" in phases:
        # Software checks, synthetic stress tests, and one held-out public trace.
        base = REPLAY + ["--config", "configs/scheduler.yaml"]
        severe = []
        for case, flags in [
            ("steady", ["--steady-control"]),
            ("mild", ["--scenario", "mild"]),
            ("severe", []),
            ("long-tail", ["--scenario", "long-tail"]),
            ("underpredict", ["--prediction-scale", "0.5"]),
            ("overpredict", ["--prediction-scale", "1.5"]),
            ("zero-transfer", ["--restore-s", "0"]),
            ("tight-deadline", ["--deadline-budget-s", "0.4"]),
        ]:
            for seed in (41, 42, 43):
                path = run_case(f"replay-{case}-{seed}", base + ["--seed", seed, *flags])
                if case == "severe":
                    severe.append(path)
        trace = data / "BurstGPT_1.csv"
        if not opts.plan:
            sys.path.insert(0, str(repo / "src"))
            from recovery_serving.live_data import source_file

            trace = source_file("burstgpt", data)
        run_case(
            "replay-burstgpt",
            REPLAY + ["--config", "configs/scheduler-burstgpt.yaml", "--burstgpt", trace],
        )
        plot = repo / "scripts/plot_replay.py"
        destination = out / "replay-severe-figures"
        if opts.plan:
            print(
                subprocess.list2cmdline(
                    [sys.executable, str(plot), *map(str, severe), "--output", str(destination)]
                )
            )
        elif not destination.exists():
            with tempfile.TemporaryDirectory(dir=out) as temp:
                stage = Path(temp) / "figures"
                subprocess.run(
                    [sys.executable, str(plot), *map(str, severe), "--output", str(stage)],
                    cwd=repo,
                    env=env,
                    check=True,
                )
                stage.rename(destination)

    if "gpu" in phases:
        nominal = run_case(
            "gpu-jit-nominal-" + seed_tag,
            jit + full + ["--arrival-scale", opts.scale],
            live=True,
            figs=True,
        )
        for name, scale in (("wider-gaps", opts.scale * 4), ("tighter-gaps", opts.scale / 4)):
            run_case(
                f"gpu-jit-{name}-{seed_tag}",
                jit
                + full
                + [
                    "--arrival-scale",
                    scale,
                    "--deadline-mode",
                    "provided",
                    "--budgets",
                    nominal / "budgets.csv",
                ],
                live=True,
                figs=True,
            )

    if "extended" in phases:
        for mode in ("artifact", "jitserve-20s"):
            run_case(
                f"gpu-jit-{mode}-{seed_tag}",
                jit + full + ["--arrival-scale", opts.scale, "--deadline-mode", mode],
                live=True,
                figs=True,
            )
        run_case(
            "gpu-qwen-math-" + seed_tag,
            live
            + full
            + [
                "--source",
                "qwen-math",
                "--arrival-scale",
                opts.scale,
                "--max-new-tokens",
                2048,
                "--high-kv-tokens",
                4096,
            ],
            live=True,
            figs=True,
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="ReServe checkout (default: current directory)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/device-suite"),
        help="Output root, relative to repo or absolute",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Live prompt-split seed; replay always uses 41, 42, 43"
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Full live measurements per policy on each frozen cohort",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.02,
        help="Nominal multiplier on real BurstGPT interarrival gaps",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=256, help="JITServe generation cap (math uses 2048)"
    )
    p.add_argument(
        "--high-kv-tokens", type=int, default=1024, help="JITServe high-mode logical KV allowance"
    )
    p.add_argument(
        "--model-dir", type=Path, help="Optional pinned offline Qwen checkpoint directory"
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help="List commands without downloading inputs or running models",
    )
    opts = p.parse_args()
    if (
        opts.repeats < 1
        or not math.isfinite(opts.scale)
        or opts.scale <= 0
        or opts.max_new_tokens < 2
    ):
        p.error("Use positive repeats and arrival scale, and an output cap of at least two tokens.")
    launch(opts)


def write_live_csv(output_folder):
    """Summarize a single live run in live-comparison.csv. Deltas are policy minus cost."""
    folder = Path(output_folder)
    summaries = json.loads((folder / "summary.json").read_text(encoding="utf-8"))

    def percentile99(values):
        if not values:
            return None
        values.sort()
        index = (len(values) - 1) * 0.99
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    group_fields = (
        "requests",
        "natural_eos",
        "length_capped",
        "deadline_met",
        "deadline_attainment",
        "deadline_goodput_rps",
        "generated_tokens_per_s",
        "p95_ttft_s",
        "p95_tpot_s",
        "p95_queue_s",
        "p95_e2e_natural_s",
        "low_switches",
    )
    output = []
    for name, stats in sorted(summaries.items()):
        prefix, repeat, policy = name.split("-", 2)
        if prefix != "rep":
            raise ValueError("Expected a recovery_serving.live summary in " + str(folder))
        cost = summaries.get(f"rep-{repeat}-cost")
        requests = json.loads((folder / name / "requests.json").read_text(encoding="utf-8"))
        long_group = stats["by_reference_length"]["long"]
        common = {"repeat": int(repeat), "policy": policy}
        common.update({field: stats[field] for field in group_fields})
        common.update(
            {
                "p99_ttft_s": percentile99([r["ttft_s"] for r in requests]),
                "p99_e2e_natural_s": percentile99(
                    [r["e2e_s"] for r in requests if r["status"] == "eos"]
                ),
                "long_offered": long_group["offered"],
                "long_on_time": long_group["on_time"],
                "long_capped": long_group["length_capped"],
                "long_p95_queue_s": long_group["p95_queue_s"],
            }
        )
        common["goodput_delta_vs_cost_rps"] = (
            stats["deadline_goodput_rps"] - cost["deadline_goodput_rps"] if cost else None
        )
        ref = (
            {(e["burst_start_s"], e["burst_end_s"]): e for e in cost["burst_recovery"]}
            if cost
            else {}
        )
        episodes = stats["burst_recovery"] or [{"status": "no_burst"}]
        for episode in episodes:
            start, end = episode.get("burst_start_s"), episode.get("burst_end_s")
            baseline = ref.get((start, end), {})
            delta = None
            if episode["status"] == baseline.get("status") == "recovered":
                delta = episode["recovery_s"] - baseline["recovery_s"]
            output.append(
                {
                    **common,
                    "burst_start_s": start,
                    "burst_end_s": end,
                    "recovery_status": episode["status"],
                    "recovery_s": episode.get("recovery_s"),
                    "observed_until_s": episode.get("observed_until_s"),
                    "cost_recovery_status": baseline.get(
                        "status", "no_burst" if cost and not cost["burst_recovery"] else None
                    ),
                    "recovery_delta_vs_cost_s": delta,
                }
            )
    if not output:
        raise ValueError("No policy measurements in " + str(folder))
    destination = folder / "live-comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return destination


def write_live_plots(folder):
    """Render two figures for a CUDA comparison containing cost and recovery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    folder = Path(folder)
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    summaries = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if not str(meta["hardware"]["device"]).startswith("cuda") or not {
        "rep-01-recovery",
        "rep-01-cost",
    }.issubset(summaries):
        return
    event_path = folder / "rep-01-recovery" / "events.jsonl"
    with event_path.open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    high = next(e["high_blocks"] for e in events if e["event"] == "initial_mode")
    target_tokens = 16 * max(1, high - meta["settings"]["staging_blocks"])
    iterations = [e for e in events if e["event"] == "iteration"]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for i, (start, end) in enumerate(meta["offline_bursts_s"]):
        ax.axvspan(start, end, color="#d2c49c", alpha=0.35, label="Trace burst" if i == 0 else None)
    for field, label, color in [
        ("kv_tokens_before_release", "Before KV release", "#36688d"),
        ("kv_tokens", "After KV release", "#08937f"),
    ]:
        ax.step(
            [e["time_s"] for e in iterations],
            [e[field] for e in iterations],
            where="post",
            color=color,
            label=label,
        )
    ax.axhline(target_tokens, color="#202226", ls="--", label="Restore target (token equivalent)")
    for event, label, color in [
        ("low_active", "Low active", "#9e4d1b"),
        ("restore_requested", "Restore requested", "#70489c"),
        ("high_ready", "High ready", "#127135"),
    ]:
        for i, e in enumerate(e for e in events if e["event"] == event):
            ax.axvline(e["time_s"], color=color, ls=":", label=label if i == 0 else None)
    ax.set(
        title="Recovery policy · repetition 1",
        xlabel="Elapsed time (s)",
        ylabel="Logical KV tokens",
    )
    ax.legend(fontsize=8, ncol=3)
    fig.text(
        0.01,
        -0.035,
        "Each request is rounded to 16-token blocks for admission. Lines show unrounded token counts.",
        fontsize=8,
    )
    fig.savefig(folder / "gpu-occupancy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    colors = {
        "fixed_high": "#6b6b6b",
        "fixed_low": "#a66f18",
        "pressure": "#7762ab",
        "cost": "#37669d",
        "drain": "#189383",
        "recovery": "#ba462b",
    }
    statuses = ["fixed_mode", "no_switch", "censored", "no_burst"]
    fig, (timed, other) = plt.subplots(
        1, 2, sharey=True, figsize=(10, 5), width_ratios=(1.5, 1), layout="constrained"
    )
    paired, present = {}, set()
    for name, stats in sorted(summaries.items()):
        rep, policy = name.rsplit("-", 1)
        present.add(policy)
        goodput = stats["deadline_goodput_rps"]
        for episode in stats["burst_recovery"] or [{"status": "no_burst"}]:
            status = episode["status"]
            if status == "recovered":
                x = episode["recovery_s"]
                key = (rep, episode["burst_start_s"], episode["burst_end_s"])
                paired.setdefault(key, {})[policy] = (x, goodput)
                timed.scatter(x, goodput, color=colors[policy], s=36, zorder=3, alpha=0.85)
            else:
                offset = 0.055 * (list(colors).index(policy) - 2.5)
                other.scatter(
                    statuses.index(status) + offset, goodput, color=colors[policy], s=36, alpha=0.8
                )
    for pairs in paired.values():
        if "cost" in pairs and "recovery" in pairs:
            timed.plot(*zip(pairs["cost"], pairs["recovery"]), color=".65", lw=1, zorder=1)
    timed.set(
        title="Episodes with measured recovery",
        xlabel="High ready minus burst end (s)",
        ylabel="On-time completions/s",
    )
    other.set(title="Episodes without a recovery time", xlim=(-0.4, 3.4))
    other.set_xticks(
        range(len(statuses)), ["Fixed mode", "No switch", "Censored", "No burst"], rotation=20
    )
    handles = [
        Line2D([], [], marker="o", ls="", color=c, label=p.replace("_", " "))
        for p, c in colors.items()
        if p in present
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "GPU recovery and deadline goodput · lines pair cost and recovery within a repetition and burst"
    )
    fig.savefig(folder / "gpu-recovery-goodput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
