"""Plot an occupancy timeline and compare deadline goodput with sustained recovery."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text())


def timeline(folder, policy, output):
    meta = read_json(folder / "run.json")
    with (folder / policy / "events.jsonl").open() as handle:
        events = [json.loads(line) for line in handle]
    cfg = meta["config"]["replay"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    steps = [e for e in events if e["event"] == "iteration_finished"]
    ax.step(
        [0] + [e["t_s"] for e in steps],
        [0] + [e["live_blocks"] for e in steps],
        where="post",
        color="#234863",
        label="Live KV after each step",
    )
    ax.axhline(
        cfg["high_blocks"] - cfg["staging_blocks"],
        color="#a74238",
        ls="--",
        label="Safe restore budget",
    )
    for i, (start, end) in enumerate(meta["offline_bursts_s"]):
        ax.axvspan(
            start, end, color="#cadae7", alpha=0.5, label="Arrival burst" if i == 0 else None
        )
    labels = {
        "memory_eligible": ("Memory eligible", "#80713a"),
        "high_ready": ("Weights ready", "#197151"),
    }
    seen = set()
    copy_start = None
    for event in events:
        kind, t = event["event"], event["t_s"]
        if kind in labels:
            label, color = labels[kind]
            ax.axvline(
                t,
                color=color,
                lw=1.1,
                ls=":" if kind == "memory_eligible" else "-",
                label=label if kind not in seen else None,
            )
        if kind == "transfer_started":
            copy_start = t
        if kind == "transfer_finished" and copy_start is not None:
            ax.axvspan(
                copy_start,
                t,
                color="#e0a452",
                alpha=0.6,
                label="Weight transfer" if kind not in seen else None,
            )
            copy_start = None
        seen.add(kind)
    if copy_start is not None:
        ax.axvspan(
            copy_start,
            meta["observation_window_s"],
            color="#e0a452",
            alpha=0.6,
            label="Transfer in progress",
        )
    ax.set(
        title=f"{policy} admission · CPU replay · {meta['timing_source']}",
        xlabel="Time from start (s)",
        ylabel="KV blocks",
    )
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(output / "occupancy.png", dpi=160)
    plt.close(fig)


def comparison(folders, output):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    missing = Counter()
    handles, rows = {}, []
    colors = {"pressure": "#596779", "cost": "#116b8e", "drain": "#a35726", "recovery": "#357343"}
    for folder in folders:
        data = read_json(folder / "summary.json")
        meta = read_json(folder / "run.json")
        cost = data.get("cost")
        for name, summary in data.items():
            episodes = summary["recovery"] or [{"status": "no_burst", "recovery_s": None}]
            for i, ep in enumerate(episodes):
                ready, rate = ep["recovery_s"], summary["deadline_goodput_rps"]
                reference = (
                    cost["recovery"][i]["recovery_s"]
                    if cost and i < len(cost["recovery"])
                    else None
                )
                rows.append(
                    {
                        "run": folder.name,
                        "trace_sha256": meta["workload_sha256"],
                        "variant": name,
                        "burst": i + 1 if summary["recovery"] else None,
                        "status": ep["status"],
                        "recovery_s": ready,
                        "goodput_rps": rate,
                        "on_time_fraction": summary["deadline_attainment"],
                        "delta_recovery_vs_cost_s": ready - reference
                        if ready is not None and reference is not None
                        else None,
                        "delta_goodput_vs_cost_rps": rate - cost["deadline_goodput_rps"]
                        if cost
                        else None,
                    }
                )
                if ready is not None:
                    handles[name] = ax.scatter(rate, ready, color=colors.get(name), label=name)
                elif ep["status"] != "fixed_mode":
                    missing[ep["status"]] += 1
    if handles:
        ax.legend(handles.values(), handles.keys())
    else:
        ax.text(
            0.5, 0.5, "No sustained returns in these windows", ha="center", transform=ax.transAxes
        )
    ax.set(
        title="Sustained precision recovery and deadline goodput · CPU replay",
        xlabel="Requests completed on time per second",
        ylabel="Seconds from burst end to high precision",
    )
    if missing:
        fig.supxlabel(
            "   ".join(f"{k.replace('_', ' ')}: {n}" for k, n in sorted(missing.items())),
            fontsize=9,
        )
    fig.savefig(output / "recovery_goodput.png", dpi=160)
    plt.close(fig)
    with (output / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs", type=Path, nargs="+", help="comparison directories, one per seed"
    )
    parser.add_argument(
        "--timeline-policy", default="recovery", help="policy shown from the first run"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(
        read_json(d / "run.json")["measurement_kind"] != "cpu_timing_simulation"
        for d in args.run_dirs
    ):
        parser.error("expected replay outputs")
    if not (args.run_dirs[0] / args.timeline_policy / "events.jsonl").is_file():
        parser.error("timeline policy is absent from the first run")
    args.output.mkdir(parents=True, exist_ok=False)
    timeline(args.run_dirs[0], args.timeline_policy, args.output)
    comparison(args.run_dirs, args.output)


if __name__ == "__main__":
    main()
