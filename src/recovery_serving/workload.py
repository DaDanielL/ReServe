"""Generate the same open-loop arrival trace for every policy comparison."""

import random
from dataclasses import asdict, dataclass


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
            requests.append(
                Request(
                    request_id=f"req-{len(requests):06d}",
                    arrival_s=arrival,
                    input_tokens=rng.randint(*workload["input_tokens"]),
                    output_tokens=rng.randint(*workload["output_tokens"]),
                    deadline_s=arrival + config["deadline_budget_s"],
                    phase=phase["name"],
                )
            )
        phase_start = phase_end
    return requests
