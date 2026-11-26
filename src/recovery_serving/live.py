"""Reproducible wall-clock scheduler experiment on a pinned Qwen checkpoint."""

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from dataclasses import replace
from pathlib import Path

from .experiment import VARIANTS
from .live_benchmark import profile_prompts, run_policy, warmup
from .live_data import SOURCES, Profile, arrival_trace, digest, load_prompts, split_prompts
from .metrics import percentile
from .run import environment
from .scheduling import OnlineRequest
from .workload import peak_window


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, help="Offline folder containing the pinned checkpoint")
    p.add_argument("--data-dir", type=Path, default=Path("data/live"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--source", choices=["jitserve", "qwen-math"], default="jitserve")
    p.add_argument(
        "--deadline-mode",
        choices=["device-p95", "artifact", "jitserve-20s", "provided"],
        default="device-p95",
    )
    p.add_argument("--budgets", type=Path, help="CSV with source_id, prompt_group, budget_s")
    p.add_argument("--calibration", type=int, default=24)
    p.add_argument("--requests", type=int, default=24)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--burst-start", type=int, default=126000, help="Original BurstGPT seconds")
    p.add_argument("--arrival-scale", type=float, default=0.02, help="1 preserves trace gaps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--swap-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--high-kv-tokens",
        type=int,
        default=1024,
        help="Logical KV headroom in high mode, rounded down to a 16-token block",
    )
    p.add_argument("--staging-blocks", type=int, default=2)
    p.add_argument("--down-queue", type=int, default=3)
    p.add_argument("--up-queue", type=int, default=1)
    p.add_argument(
        "--long-output-tokens",
        type=int,
        default=128,
        help="Historical answer length threshold for the fairness cohorts",
    )
    p.add_argument("--lookahead", type=int, default=16)
    p.add_argument("--max-wait", type=float, default=3.0)
    return p


def validate(a):
    if (
        min(
            a.calibration,
            a.requests,
            a.max_prompt_tokens,
            a.batch_size,
            a.repeats,
            a.lookahead,
            a.long_output_tokens,
            a.swap_layers,
        )
        < 1
        or a.max_new_tokens < 2
    ):
        raise ValueError("Use positive cohort sizes and limits, and at least two output tokens")
    if a.burst_start < 0 or not 0 <= a.up_queue < a.down_queue:
        raise ValueError("Use a nonnegative trace start and 0 <= up-queue < down-queue")
    if not 0 <= a.staging_blocks < a.high_kv_tokens // 16:
        raise ValueError("KV headroom must exceed the restore staging margin")
    if not all(math.isfinite(x) and x > 0 for x in (a.arrival_scale, a.max_wait)):
        raise ValueError("arrival-scale and max-wait must be finite and positive")
    if len(set(a.variants)) != len(a.variants):
        raise ValueError("Each variant may appear once per repetition")
    if a.deadline_mode == "artifact" and a.source != "jitserve":
        raise ValueError("Only JITServe releases stored benchmark budgets")
    if (a.deadline_mode == "provided") != (a.budgets is not None):
        raise ValueError("Use --deadline-mode provided with --budgets CSV")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record_prompt(prompt):
    src = prompt.source
    return {
        "source_id": src.source_id,
        "prompt_group": prompt.group,
        "qwen_input_tokens": len(prompt.tokens),
        "input_ids_sha256": hashlib.sha256(json.dumps(prompt.tokens).encode()).hexdigest(),
        "source_input_tokens": src.reference_input,
        "source_output_tokens": src.reference_outputs,
        "source_budget_s": src.artifact_budget_s,
    }


def supplied_budgets(path):
    budgets = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_id"], row["prompt_group"])
            value = float(row["budget_s"])
            if key in budgets or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Duplicate or invalid budget for {key[0]}")
            budgets[key] = value
    return budgets


def execute(a, folder, engine):
    prompts = load_prompts(a.source, a.data_dir)
    cal, test = split_prompts(
        prompts, engine.encode, a.calibration, a.requests, a.max_prompt_tokens, a.seed
    )
    if (
        max(len(p.tokens) for p in cal + test) + a.max_new_tokens
        > engine.model.config.max_position_embeddings
    ):
        raise ValueError("Prompt plus output cap exceeds Qwen context length")
    if max(len(p.tokens) for p in test) + a.max_new_tokens > a.high_kv_tokens // 16 * 16:
        raise ValueError("Increase high-kv-tokens or reduce the prompt or generation limit")
    trace = arrival_trace(a.data_dir, len(test), a.burst_start, a.arrival_scale)
    custom = supplied_budgets(a.budgets) if a.budgets else {}
    if a.budgets and any((p.source.source_id, p.group) not in custom for p in test):
        raise ValueError("The budget file is missing a matching evaluation prompt")
    measured, decode_probes = {}, {}
    degrees = ["high"] if a.variants == ["fixed_high"] else ["high", "low"]
    for mode in degrees:
        engine.set_mode(mode)
        decode_probes[mode] = warmup(engine, cal[0])
        print(f"Profiling {len(cal)} held-out calibration prompts at {mode} precision", flush=True)
        measured[mode] = profile_prompts(engine, cal, a.max_new_tokens)
        write_json(folder / f"calibration-{mode}.json", measured[mode])
    profiles = {mode: Profile(rows, decode_probes[mode]) for mode, rows in measured.items()}
    high = profiles["high"]
    online, manifest = [], []
    for p, arrival in zip(test, trace):
        rid, t = p.source.source_id, arrival["arrival_s"]
        if a.deadline_mode == "provided":
            deadline = t + custom[(rid, p.group)]
            prediction = high.predictor.remaining(OnlineRequest(rid, t, deadline, len(p.tokens)))
        else:
            deadline, prediction = high.assign(p, t, a.deadline_mode)
        online.append(OnlineRequest(rid, t, deadline, len(p.tokens)))
        manifest.append(
            {
                **record_prompt(p),
                **arrival,
                "deadline_s": deadline,
                "budget_s": deadline - t,
                "predicted_output_tokens": prediction,
            }
        )
    write_json(folder / "workload.json", manifest)
    write_json(folder / "calibration-inputs.json", [record_prompt(p) for p in cal])
    with (folder / "budgets.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["source_id", "prompt_group", "budget_s"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(manifest)
    # Use complete equal-duration bins. Offline burst labels cannot affect the decisions.
    bin_s = a.arrival_scale  # one original second per bin
    offsets = [t["original_s"] - trace[0]["original_s"] for t in trace]
    original = [replace(r, arrival_s=t) for r, t in zip(online, offsets)]
    bursts = [(lo * bin_s, hi * bin_s) for lo, hi in peak_window(original, offsets[-1] + 1, 1, 1.5)]
    args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    metadata = {
        "measurement_kind": "wall_clock_model_inference",
        "hardware": engine.describe(),
        "environment": environment(),
        "settings": args,
        "offline_bursts_s": bursts,
        "prompt_source": SOURCES[a.source],
        "arrival_source": SOURCES["burstgpt"],
        "workload_sha256": digest(folder / "workload.json"),
        "budget_input_sha256": digest(a.budgets) if a.budgets else None,
        "calibration_length_capped": sum(r["status"] == "length_cap" for r in measured["high"]),
        "warmup_decode_s": decode_probes,
        "generation": "greedy argmax, no sampling or repetition penalty, EOS counted",
    }
    write_json(folder / "run.json", metadata)
    summaries = {}
    for rep in range(a.repeats):
        order = list(a.variants)
        random.Random(a.seed + rep).shuffle(order)
        for name in order:
            label = f"rep-{rep + 1:02d}-{name}"
            sub = folder / label
            sub.mkdir()
            mode, policy = VARIANTS[name]
            print(f"Measuring {label}", flush=True)
            with (sub / "events.jsonl").open("w", encoding="utf-8") as stream:

                def emit(obj):
                    stream.write(json.dumps(obj, separators=(",", ":")) + "\n")

                rows, summary = run_policy(
                    engine, test, online, profiles, mode, policy, a, bursts, emit
                )
            # Length prediction error is observable only for naturally terminated outputs.
            errors = [
                abs(r["generated_tokens"] - m["predicted_output_tokens"])
                for r, m in zip(rows, manifest)
                if r["status"] == "eos"
            ]
            summary["length_mae_natural_tokens"] = sum(errors) / len(errors) if errors else None
            summary["length_mae_samples"] = len(errors)
            # Cohorts use a source reference from before the experiment and include capped jobs.
            for r, m in zip(rows, manifest):
                refs = sorted(m["source_output_tokens"])
                r["long_reference"] = refs[(len(refs) - 1) // 2] >= a.long_output_tokens
            summary["by_reference_length"] = {}
            for label_name, is_long in (("long", True), ("short", False)):
                group = [r for r in rows if r["long_reference"] == is_long]
                summary["by_reference_length"][label_name] = {
                    "offered": len(group),
                    "on_time": sum(r["deadline_met"] for r in group),
                    "length_capped": sum(r["status"] == "length_cap" for r in group),
                    "p95_queue_s": percentile([r["queue_s"] for r in group], 0.95),
                }
            write_json(sub / "requests.json", rows)
            write_json(sub / "summary.json", summary)
            summaries[label] = summary
        write_json(folder / "summary.json", summaries)
    return summaries


def main(argv=None):
    a = parser().parse_args(argv)
    validate(a)
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    from .live_model import QwenEngine

    output.parent.mkdir(parents=True, exist_ok=True)
    engine = QwenEngine(a.model_dir, a.device, a.dtype, a.swap_layers)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".live-") as tmp:
        staging = Path(tmp) / "report"
        staging.mkdir()
        execute(a, staging, engine)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
