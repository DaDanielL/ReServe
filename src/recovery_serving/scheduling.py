"""Client admission baselines, not token-level engine scheduling."""

from .workload import Request


def priority(request: Request, policy: str) -> float:
    if policy == "fifo":
        return request.arrival_s
    if policy == "edf":
        return request.deadline_s
    raise ValueError(f"Unknown policy: {policy}")


class RecoveryAwareScheduler:
    """Future engine-side policy: needs per-request estimates and allocator state."""

    def schedule(self, requests, memory_state):
        raise NotImplementedError("Implement after measuring the recovery bottleneck")
