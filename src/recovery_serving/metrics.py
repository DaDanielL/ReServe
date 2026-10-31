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
