"""Small reporting helpers shared by the smoke harness and future real runs."""


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict], duration_s: float) -> dict:
    successful = [row for row in rows if row["status"] == "ok"]
    on_time = [row for row in successful if row["deadline_met"]]
    return {
        "backend": "mock",
        "measurement_kind": "artificial_setup_check",
        "submitted": len(rows),
        "completed": len(successful),
        "failed": len(rows) - len(successful),
        "on_time_completed": len(on_time),
        "deadline_attainment": len(on_time) / len(rows) if rows else None,
        "duration_s": duration_s,
        "deadline_goodput_rps": len(on_time) / duration_s if duration_s > 0 else None,
        "p95_e2e_s": percentile([row["e2e_s"] for row in successful], 0.95),
        "p95_ttft_s": percentile([row["ttft_s"] for row in successful], 0.95),
    }


def recovery_episodes(bursts, transitions, horizon, sustain_s, dynamic=True):
    """Score offline burst windows. A sustained return may precede the end of arrivals."""
    outcomes = []
    for i, (start, end) in enumerate(bursts):
        stop = bursts[i + 1][0] if i + 1 < len(bursts) else horizon
        row = {
            "burst_start_s": start,
            "burst_end_s": end,
            "observed_until_s": stop,
            "status": "no_switch" if dynamic else "fixed_mode",
            "ready_s": None,
            "recovery_s": None,
            "trigger_to_ready_s": None,
        }
        outcomes.append(row)
        if not dynamic:
            continue
        modes = [
            (t, e)
            for t, e in transitions
            if e in {"low_active", "high_ready"}
            and (t < stop or (i == len(bursts) - 1 and t == stop))
        ]
        intervals = [
            (t, e, modes[j + 1][0] if j + 1 < len(modes) else stop)
            for j, (t, e) in enumerate(modes)
        ]
        lows = [t for t, e, until in intervals if e == "low_active" and t < end and until > start]
        if not lows:
            # A backlog may trigger the first downgrade only after arrivals subside.
            lows = [t for t, e, _ in intervals if e == "low_active" and end <= t < stop][:1]
        if not lows:
            continue
        row["status"] = "censored"
        for ready, e, until in intervals:
            if e == "high_ready" and ready >= lows[-1] and ready + sustain_s <= until:
                row.update(status="recovered", ready_s=ready, recovery_s=ready - end)
                triggers = [
                    t
                    for t, event in transitions
                    if event == "restore_requested" and lows[-1] <= t <= ready
                ]
                if triggers:
                    row["trigger_to_ready_s"] = ready - triggers[-1]
                break
    return outcomes


def summarize_replay(result, horizon, bursts, settings, mode):
    rows = result.rows
    complete = [r for r in rows if r["status"] == "ok"]
    on_time = sum(r["deadline_met"] for r in rows)
    generated = sum(r["generated_tokens"] for r in rows)
    low_tokens = sum(r["low_tokens"] for r in rows)
    low_start = 0.0 if mode == "low" else None
    low_s = 0.0
    switches = 0
    for t, event in result.transitions:
        if event == "low_active":
            low_start = t
            switches += 1
        elif event == "high_ready":
            low_s += t - low_start
            low_start = None
            switches += 1
    if low_start is not None:
        low_s += horizon - low_start
    summary = {
        "low_mode_s": low_s,
        "precision_switches": switches,
        "measurement_kind": "cpu_timing_simulation",
        "duration_s": horizon,
        "submitted": len(rows),
        "completed": len(complete),
        "on_time_completed": on_time,
        "deadline_goodput_rps": on_time / horizon,
        "deadline_attainment": on_time / len(rows) if rows else None,
        "generated_tokens": generated,
        "token_throughput_s": generated / horizon,
        "low_tokens": low_tokens,
        "low_token_fraction": low_tokens / generated if generated else None,
        "scheduler_decisions": len(result.decision_ns),
        "scheduler_mean_us": sum(result.decision_ns) / (1000 * len(result.decision_ns))
        if result.decision_ns
        else None,
        "scheduler_p95_us": percentile([ns / 1000 for ns in result.decision_ns], 0.95),
        "recovery": recovery_episodes(
            bursts, result.transitions, horizon, settings.sustain_s, mode == "dynamic"
        ),
    }
    for status in ("rejected", "timed_out", "length_limited", "unfinished"):
        summary[status] = sum(r["status"] == status for r in rows)
    for name in ("ttft_s", "e2e_s", "queue_s", "tbt_s"):
        values = result.gaps if name == "tbt_s" else [r[name] for r in rows if r[name] is not None]
        for p in (95, 99):
            summary[f"p{p}_{name}"] = percentile(values, p / 100)
    by_length = {}
    for label, is_long in (("short", False), ("long", True)):
        group = [r for r in rows if (r["output_tokens"] >= settings.long_output_tokens) == is_long]
        hits = sum(r["deadline_met"] for r in group)
        by_length[label] = {
            "offered": len(group),
            "on_time": hits,
            "deadline_attainment": hits / len(group) if group else None,
            "never_started": sum(r["started_s"] is None for r in group),
            "p95_queue_s_started": percentile(
                [r["queue_s"] for r in group if r["queue_s"] is not None], 0.95
            ),
            "p95_observed_wait_s_all": percentile([r["observed_wait_s"] for r in group], 0.95),
        }
    summary["by_output_length"] = by_length
    return summary
