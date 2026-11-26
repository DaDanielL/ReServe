"""Generate the same open-loop arrival trace for every policy comparison."""

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_s: float
    input_tokens: int
    output_tokens: int
    deadline_s: float  # Absolute within the trace, measured from experiment start.
    phase: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_trace(config: dict) -> list[Request]:
    rng = random.Random(config["seed"])
    workload = config["workload"]
    requests = []
    phase_start = 0.0
    for phase in workload["phases"]:
        phase_end = phase_start + phase["duration_s"]
        arrival = phase_start
        rate = phase["rate_rps"]
        while rate > 0:
            arrival += rng.expovariate(rate)
            if arrival >= phase_end:
                break
            shape = sample_shape(rng, workload)
            requests.append(
                Request(
                    request_id=f"req-{len(requests):06d}",
                    arrival_s=arrival,
                    input_tokens=rng.randint(*shape["input_tokens"]),
                    output_tokens=rng.randint(*shape["output_tokens"]),
                    deadline_s=arrival
                    + shape.get("deadline_budget_s", config["deadline_budget_s"]),
                    phase=phase["name"],
                )
            )
        phase_start = phase_end
    return requests


def sample_shape(rng: random.Random, workload: dict) -> dict:
    classes = workload.get("classes")
    if classes:
        return rng.choices(classes, weights=[item["weight"] for item in classes])[0]
    return workload


def synthetic_calibration(config: dict, seed: int, count: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        shape = sample_shape(rng, config["workload"])
        pairs.append((rng.randint(*shape["input_tokens"]), rng.randint(*shape["output_tokens"])))
    return pairs


def steady_control(trace: list[Request], duration_s: float) -> list[Request]:
    """Spread the identical request shapes and deadline budgets evenly across the window."""
    result = []
    for i, req in enumerate(trace):
        arrival = (i + 0.5) * duration_s / len(trace)
        result.append(
            replace(
                req,
                arrival_s=arrival,
                deadline_s=arrival + (req.deadline_s - req.arrival_s),
                phase="steady",
            )
        )
    return result


def trace_lines(trace: list[Request]) -> str:
    return "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in trace)


def burstgpt_trace(path: Path, settings: dict, budget_s: float):
    """Read disjoint half-open windows of one model/service from the public BurstGPT CSV."""
    windows = [tuple(settings[name]) for name in ("calibration_window_s", "evaluation_window_s")]
    if any(
        len(w) != 2 or not all(math.isfinite(v) for v in w) or not 0 <= w[0] < w[1] for w in windows
    ):
        raise ValueError("BurstGPT windows must be finite [start, end) pairs")
    (c0, c1), (e0, e1) = windows
    if max(c0, e0) < min(c1, e1):
        raise ValueError("calibration and evaluation windows overlap")
    scale = settings["time_scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("BurstGPT time_scale must be finite and positive")
    trace, calibration = [], []
    failed = {"evaluation": 0, "calibration": 0}
    required = {"Timestamp", "Model", "Log Type", "Request tokens", "Response tokens"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(
                f"Missing BurstGPT columns: {sorted(required - set(rows.fieldnames or []))}"
            )
        for i, row in enumerate(rows, 2):
            if row["Model"] != settings["model"] or row["Log Type"] != settings["log_type"]:
                continue
            t = float(row["Timestamp"])
            if not math.isfinite(t):
                raise ValueError(f"Nonfinite BurstGPT timestamp on CSV line {i}")
            if not (c0 <= t < c1 or e0 <= t < e1):
                continue
            prompt, output = int(row["Request tokens"]), int(row["Response tokens"])
            split = "calibration" if c0 <= t < c1 else "evaluation"
            if output == 0:
                failed[split] += 1
                continue
            if min(prompt, output) < 1:
                raise ValueError(f"Invalid token count on CSV line {i}")
            if split == "calibration":
                calibration.append((prompt, output))
            else:
                arrival = (t - e0) * scale
                trace.append(
                    Request(f"burstgpt-{i}", arrival, prompt, output, arrival + budget_s, "trace")
                )
    if not calibration or not trace:
        raise ValueError("BurstGPT requires eligible requests in both selected windows")
    trace.sort(key=lambda r: r.arrival_s)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    meta = {
        "source": "BurstGPT",
        "source_sha256": digest.hexdigest(),
        "settings": settings,
        "excluded_source_failures": failed,
    }
    return trace, calibration, (e1 - e0) * scale, meta


def peak_window(trace: list[Request], duration_s: float, bin_s: float, factor: float):
    """Offline burst label: contiguous bins over factor times the mean around the peak."""
    if not (0 < bin_s <= duration_s and factor > 1):
        raise ValueError("burst bins must fit the window and the burst factor must exceed one")
    n = math.ceil(duration_s / bin_s)
    counts = [0] * n
    for request in trace:
        counts[min(n - 1, int(request.arrival_s / bin_s))] += 1
    # Scale the last (possibly partial) bin by its observed duration.
    rates = [count / min(bin_s, duration_s - i * bin_s) for i, count in enumerate(counts)]
    threshold = factor * len(trace) / duration_s
    left = right = max(range(n), key=rates.__getitem__)
    if not trace or rates[left] < threshold:
        return []
    while left and rates[left - 1] >= threshold:
        left -= 1
    while right + 1 < n and rates[right + 1] >= threshold:
        right += 1
    return [(left * bin_s, min(duration_s, (right + 1) * bin_s))]
