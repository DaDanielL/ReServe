"""Client queue priorities and causal, iteration-level admission policies."""

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil

from .precision import PrecisionMode
from .workload import Request

_POLICY_ALIASES = {"cost": "time-based", "recovery": "space-based"}
ADMISSION_POLICIES = {"fifo", "time-based", "drain", "space-based"}


def canonical_policy(name: str) -> str:
    """Resolve a policy identifier from a command, configuration, or saved run."""
    return _POLICY_ALIASES.get(name, name)


def priority(request: Request, policy: str) -> float:
    if policy == "fifo":
        return request.arrival_s
    if policy == "edf":
        return request.deadline_s
    raise ValueError(f"Unknown policy: {policy}")


def blocks(tokens: int, block_tokens: int) -> int:
    return (tokens + block_tokens - 1) // block_tokens


@dataclass(frozen=True)
class OnlineRequest:
    """Only information available at this iteration boundary, in trace-relative seconds."""

    request_id: str
    arrival_s: float
    deadline_s: float
    input_tokens: int
    generated: int = 0
    live_blocks: int = 0


class LengthPredictor:
    """Empirical output distribution conditioned on prompt size and tokens survived."""

    def __init__(self, calibration: Iterable[tuple[int, int]], quantile: float = 0.8):
        if not 0 < quantile <= 1:
            raise ValueError("length quantile must be in (0, 1]")
        buckets = defaultdict(list)
        totals = []
        for prompt, output in calibration:
            if prompt < 1 or output < 1:
                raise ValueError("calibration lengths must be positive")
            buckets[prompt.bit_length()].append(output)
            totals.append(output)
        if not totals:
            raise ValueError("length prediction requires a nonempty calibration split")
        self.buckets = {size: sorted(values) for size, values in buckets.items()}
        self.totals = sorted(totals)
        self.quantile = quantile

    def _survivors(self, request: OnlineRequest) -> tuple[list[int], int]:
        samples = self.buckets.get(request.input_tokens.bit_length(), self.totals)
        start = bisect_right(samples, request.generated)
        if start == len(samples):
            samples = self.totals
            start = bisect_right(samples, request.generated)
        return samples, start

    def remaining(self, request: OnlineRequest) -> int:
        samples, start = self._survivors(request)
        if start == len(samples):
            # Outside calibration support, retain a conservative positive tail.
            return self.totals[ceil(self.quantile * len(self.totals)) - 1]
        rank = start + ceil(self.quantile * (len(samples) - start)) - 1
        return samples[rank] - request.generated

    def projected_blocks(
        self,
        request: OnlineRequest,
        horizon: int,
        block_tokens: int,
        max_new_tokens: int | None = None,
    ) -> float:
        end = request.generated + horizon
        if max_new_tokens is not None and end >= max_new_tokens:
            return 0.0
        samples, start = self._survivors(request)
        if start == len(samples):
            # No empirical evidence that this unusually long request will finish soon.
            survival = 1.0
        else:
            survival = (len(samples) - bisect_right(samples, end)) / (len(samples) - start)
        live_if_running = max(request.live_blocks, blocks(request.input_tokens + end, block_tokens))
        return survival * live_if_running


@dataclass(frozen=True)
class AdmissionState:
    """Budgets available to NEW requests after protecting the running batch's next step.

    reserve_blocks is the remaining public growth allowance. high_target_blocks is
    the total safe cache occupancy for a switch, including the device's staging margin.
    """

    now_s: float
    mode: PrecisionMode
    recovery_pressure: bool
    slots: int
    prefill_tokens: int
    available_blocks: int
    reserve_blocks: int
    high_target_blocks: int
    block_tokens: int
    max_new_tokens: int
    deferral_s: float = 0.0


class RecoveryAwareScheduler:
    def __init__(self, predictor: LengthPredictor, lookahead: int, max_wait_s: float):
        if lookahead < 1 or max_wait_s <= 0:
            raise ValueError("lookahead and aging threshold must be positive")
        self.predictor = predictor
        self.lookahead = lookahead
        self.max_wait_s = max_wait_s

    def schedule(
        self,
        waiting: Sequence[OnlineRequest],
        running: Sequence[OnlineRequest],
        state: AdmissionState,
        service_seconds: Callable[[OnlineRequest, int], float],
        policy: str = "space-based",
    ) -> list[str]:
        """Return IDs to admit in this batch. The device owns all hard limits."""
        policy = canonical_policy(policy)
        if policy not in ADMISSION_POLICIES:
            raise ValueError(f"Unknown admission policy: {policy}")
        if state.slots <= 0 or not waiting:
            return []

        def rank(req: OnlineRequest) -> tuple:
            if policy == "fifo":
                return (0, req.arrival_s, req.request_id)
            if state.now_s - req.arrival_s >= self.max_wait_s:
                return (0, req.arrival_s, req.request_id)  # Oldest protected request first.
            remaining = min(state.max_new_tokens, self.predictor.remaining(req))
            slack = req.deadline_s - state.now_s - service_seconds(req, remaining)
            return (1, slack, req.arrival_s, req.request_id)

        recovering = state.recovery_pressure and state.mode == PrecisionMode.LOW
        forecast = recovering and policy == "space-based"
        projected = (
            sum(
                self.predictor.projected_blocks(
                    r, self.lookahead, state.block_tokens, state.max_new_tokens
                )
                for r in running
            )
            if forecast
            else 0.0
        )
        free, reserve, tokens = state.available_blocks, state.reserve_blocks, state.prefill_tokens
        selected = []
        for key, req in sorted(((rank(r), r) for r in waiting), key=lambda pair: pair[0]):
            initial = blocks(req.input_tokens + 1, state.block_tokens)
            bound = blocks(req.input_tokens + state.max_new_tokens, state.block_tokens)
            if req.input_tokens > tokens or initial > free or bound > reserve:
                # Let an aged request accumulate room as running requests finish.
                if policy != "fifo" and key[0] == 0 and (running or selected):
                    break
                continue
            future = (
                self.predictor.projected_blocks(
                    req, self.lookahead, state.block_tokens, state.max_new_tokens
                )
                if forecast
                else 0.0
            )
            protected = policy != "fifo" and (key[0] == 0 or key[1] <= state.deferral_s)
            # At least one request can start on an empty device, even under a drain guard.
            if recovering and (running or selected) and not protected:
                if policy == "drain" or (
                    forecast and projected + future > state.high_target_blocks
                ):
                    continue
            selected.append(req.request_id)
            free -= initial
            reserve -= bound
            tokens -= req.input_tokens
            projected += future
            if len(selected) == state.slots:
                break
        return selected
