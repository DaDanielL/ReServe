"""Single-device logical-clock replay for testing admission independently of CUDA."""

import math
from dataclasses import dataclass
from time import perf_counter_ns

from .precision import PrecisionMode
from .scheduling import AdmissionState, OnlineRequest, RecoveryAwareScheduler, blocks
from .workload import Request


@dataclass(frozen=True)
class ReplaySettings:
    block_tokens: int
    high_blocks: int
    low_blocks: int
    staging_blocks: int
    max_batch: int
    batch_tokens: int
    max_new_tokens: int
    down_queue: int
    up_queue: int
    min_low_s: float
    restore_s: float
    lookahead: int
    max_wait_s: float
    followup_s: float
    sustain_s: float
    long_output_tokens: int
    quantile: float = 0.8
    queue_timeout_s: float | None = None

    def __post_init__(self):
        for name in (
            "block_tokens",
            "high_blocks",
            "low_blocks",
            "max_batch",
            "batch_tokens",
            "max_new_tokens",
            "down_queue",
            "lookahead",
            "long_output_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"replay.{name} must be a positive integer")
        if (
            type(self.staging_blocks) is not int
            or not 0 <= self.staging_blocks < self.high_blocks <= self.low_blocks
        ):
            raise ValueError("require 0 <= staging_blocks < high_blocks <= low_blocks")
        if type(self.up_queue) is not int or not 0 <= self.up_queue < self.down_queue:
            raise ValueError("require 0 <= up_queue < down_queue")
        if self.max_batch > self.batch_tokens:
            raise ValueError("each running request needs a decode token in the batch budget")
        for name in ("min_low_s", "restore_s", "max_wait_s", "followup_s", "sustain_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"replay.{name} must be finite and nonnegative")
        if self.max_wait_s <= 0 or not 0 < self.quantile <= 1:
            raise ValueError("invalid aging threshold or length quantile")
        if self.queue_timeout_s is not None and not (
            math.isfinite(self.queue_timeout_s) and self.queue_timeout_s > 0
        ):
            raise ValueError("queue_timeout_s must be finite and positive")


@dataclass(frozen=True)
class StepTiming:
    """Seconds for one mixed iteration. Coefficients can be fitted to device profiles."""

    step_s: float
    prefill_token_s: float
    decode_request_s: float
    decode_context_token_s: float = 0.0

    def __post_init__(self):
        if self.step_s <= 0 or any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("timing coefficients must be finite and nonnegative, with step_s > 0")

    def step(self, prefill_tokens: int, decodes: int, context_tokens: int) -> float:
        return (
            self.step_s
            + self.prefill_token_s * prefill_tokens
            + self.decode_request_s * decodes
            + self.decode_context_token_s * context_tokens
        )

    def service(self, req: OnlineRequest, remaining: int) -> float:
        # Full prefill emits the first token. Later decode steps read the growing context.
        decodes = remaining - 1
        contexts = decodes * (req.input_tokens + 1) + decodes * (decodes - 1) / 2
        return (
            self.step(req.input_tokens, 0, 0)
            + decodes * (self.step_s + self.decode_request_s)
            + contexts * self.decode_context_token_s
        )


@dataclass
class Record:
    request: Request
    started: float | None = None
    first: float | None = None
    last: float | None = None
    terminal: float | None = None
    status: str = "unfinished"
    reason: str = ""
    generated: int = 0
    low_tokens: int = 0

    def view(self, block_tokens: int) -> OnlineRequest:
        r = self.request
        live = 0 if self.started is None else blocks(r.input_tokens + self.generated, block_tokens)
        return OnlineRequest(
            r.request_id, r.arrival_s, r.deadline_s, r.input_tokens, self.generated, live
        )

    def row(self, horizon: float) -> dict:
        r = self.request
        stop = self.terminal if self.terminal is not None else horizon
        return {
            **r.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "started_s": self.started,
            "finished_s": self.terminal,
            "generated_tokens": self.generated,
            "low_tokens": self.low_tokens,
            "queue_s": None if self.started is None else self.started - r.arrival_s,
            "observed_wait_s": (stop if self.started is None else self.started) - r.arrival_s,
            "ttft_s": None if self.first is None else self.first - r.arrival_s,
            "e2e_s": self.terminal - r.arrival_s if self.status == "ok" else None,
            "deadline_met": self.status == "ok" and self.terminal <= r.deadline_s,
        }


@dataclass
class ReplayResult:
    rows: list[dict]
    gaps: list[float]
    transitions: list[tuple[float, str]]
    decision_ns: list[int]


class Replay:
    """Aggregate elastic cache with a serialized restore, for the CPU reference fixture."""

    def __init__(self, trace, horizon, settings, timings, scheduler, mode, policy, emit):
        if mode not in {"dynamic", "high", "low"} or not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("invalid replay mode or observation horizon")
        if policy not in {"cost", "drain", "recovery", "fifo"}:
            raise ValueError("invalid replay admission policy")
        seen = set()
        previous = 0.0
        for request in trace:
            if (
                request.request_id in seen
                or not previous <= request.arrival_s <= horizon
                or not math.isfinite(request.deadline_s)
                or request.deadline_s < request.arrival_s
                or min(request.input_tokens, request.output_tokens) < 1
            ):
                raise ValueError(
                    "trace requires unique IDs, ordered in-window arrivals, valid lengths and deadlines"
                )
            seen.add(request.request_id)
            previous = request.arrival_s
        self.s: ReplaySettings = settings
        self.timings: dict[str, StepTiming] = timings
        self.scheduler: RecoveryAwareScheduler = scheduler
        self.policy, self.device_mode, self.emit = policy, mode, emit
        self.mode = PrecisionMode.LOW if mode == "low" else PrecisionMode.HIGH
        self.ceiling = settings.high_blocks if mode == "high" else settings.low_blocks
        self.horizon = horizon
        self.records = [Record(r) for r in trace]
        self.next_arrival = 0
        self.waiting = {}
        self.active = {}
        self.now = 0.0
        self.low_since = 0.0
        self.restore_intent = False
        self.memory_is_eligible = None
        self.transitions = []
        self.decisions = []
        self.gaps = []

    def event(self, kind, **fields):
        if kind in {
            "low_active",
            "restore_requested",
            "memory_eligible",
            "transfer_started",
            "transfer_finished",
            "high_ready",
        }:
            self.transitions.append((self.now, kind))
        self.emit({"t_s": self.now, "event": kind, "mode": self.mode.value, **fields})

    def bound(self, req: Request) -> int:
        return blocks(req.input_tokens + self.s.max_new_tokens, self.s.block_tokens)

    def advance(self, now):
        """Arrivals are open-loop, including during a step, a restore or idle time."""
        while self.next_arrival < len(self.records):
            record = self.records[self.next_arrival]
            r = record.request
            if r.arrival_s > now:
                break
            self.next_arrival += 1
            self.now = r.arrival_s
            self.event("request_arrived", request_id=r.request_id)
            if r.input_tokens > self.s.batch_tokens or self.bound(r) > self.ceiling:
                record.status, record.terminal = "rejected", self.now
                record.reason = (
                    "prompt_budget"
                    if r.input_tokens > self.s.batch_tokens
                    else "public_growth_reserve"
                )
                self.event("request_rejected", request_id=r.request_id, reason=record.reason)
            else:
                self.waiting[r.request_id] = record
        self.now = now
        if self.s.queue_timeout_s is not None:
            for rid, record in list(self.waiting.items()):
                expiry = record.request.arrival_s + self.s.queue_timeout_s
                if expiry <= now:
                    self.waiting.pop(rid)
                    record.status, record.terminal, record.reason = (
                        "timed_out",
                        expiry,
                        "queue_timeout",
                    )
                    self.event("request_timed_out", request_id=rid, expired_s=expiry)

    def update_eligibility(self, next_live: int) -> bool:
        eligible = next_live <= self.s.high_blocks - self.s.staging_blocks
        if eligible != self.memory_is_eligible:
            self.event(
                "memory_eligible" if eligible else "memory_ineligible", next_step_blocks=next_live
            )
            self.memory_is_eligible = eligible
        return eligible

    def update_mode(self, next_live):
        """Return True when a restore consumed this boundary, including at zero copy time."""
        if self.device_mode != "dynamic":
            return False
        s = self.s
        needs_low_when_idle = (
            not self.active
            and bool(self.waiting)
            and any(
                blocks(r.request.input_tokens + 1, s.block_tokens) > s.high_blocks
                for r in self.waiting.values()
            )
        )
        if self.mode == PrecisionMode.HIGH:
            if (
                next_live > s.high_blocks
                or len(self.waiting) >= s.down_queue
                or needs_low_when_idle
            ):
                self.mode = PrecisionMode.LOW
                self.low_since = self.now
                self.memory_is_eligible = None
                self.event("low_active")
            else:
                return False
        eligible = self.update_eligibility(next_live)
        intent = (
            len(self.waiting) <= s.up_queue
            and self.now >= self.low_since + s.min_low_s
            and not needs_low_when_idle
        )
        if intent != self.restore_intent:
            self.event("restore_requested" if intent else "restore_cancelled")
            self.restore_intent = intent
        if not intent or not eligible:
            return False
        self.event("transfer_started")
        end = self.now + s.restore_s
        self.advance(min(end, self.horizon))
        if end <= self.horizon:
            self.event("transfer_finished")
            self.mode = PrecisionMode.HIGH
            self.event("high_ready")
        self.restore_intent = False
        return True

    def run(self) -> ReplayResult:
        s = self.s
        self.event("initial_mode", high_target_blocks=s.high_blocks - s.staging_blocks)
        self.advance(0.0)
        while self.now < self.horizon:
            old = list(self.active.values())
            running = [r.view(s.block_tokens) for r in old]
            next_live = sum(
                blocks(r.input_tokens + r.generated + 1, s.block_tokens) for r in running
            )
            if self.update_mode(next_live):
                continue
            if not old and not self.waiting:
                wake = self.horizon
                if self.next_arrival < len(self.records):
                    wake = min(wake, self.records[self.next_arrival].request.arrival_s)
                dwell_end = self.low_since + s.min_low_s
                if (
                    self.device_mode == "dynamic"
                    and self.mode == PrecisionMode.LOW
                    and self.now < dwell_end
                ):
                    wake = min(wake, dwell_end)
                self.advance(wake)
                continue
            capacity = s.low_blocks if self.mode == PrecisionMode.LOW else s.high_blocks
            reserved = sum(self.bound(r.request) for r in old)
            timing = self.timings[self.mode.value]
            context = sum(r.input_tokens + r.generated for r in running)
            state = AdmissionState(
                self.now,
                self.mode,
                self.restore_intent,
                s.max_batch - len(old),
                s.batch_tokens - len(old),
                capacity - next_live,
                self.ceiling - reserved,
                s.high_blocks - s.staging_blocks,
                s.block_tokens,
                s.max_new_tokens,
                timing.step(0, len(old), context),
            )
            waiting = [r.view(s.block_tokens) for r in self.waiting.values()]
            started = perf_counter_ns()
            selected = self.scheduler.schedule(waiting, running, state, timing.service, self.policy)
            self.decisions.append(perf_counter_ns() - started)
            prefill = 0
            for rid in selected:
                record = self.waiting.pop(rid)
                self.active[rid] = record
                record.started = self.now
                prefill += record.request.input_tokens
                reserved += self.bound(record.request)
                next_live += blocks(record.request.input_tokens + 1, s.block_tokens)
                self.event("request_admitted", request_id=rid)
            if not self.active or next_live > capacity or reserved > self.ceiling:
                raise RuntimeError("admission violated progress or capacity guard")
            if self.device_mode == "dynamic" and self.mode == PrecisionMode.LOW:
                self.update_eligibility(next_live)
            self.event(
                "iteration_started",
                prefill_tokens=prefill,
                decode_requests=len(old),
                peak_blocks=next_live,
                capacity_blocks=capacity,
                reserved_blocks=reserved,
            )
            end = self.now + timing.step(prefill, len(old), context)
            self.advance(min(end, self.horizon))
            if end > self.horizon:
                break
            live = next_live
            for rid, record in list(self.active.items()):
                if record.first is None:
                    record.first = self.now
                    self.event("first_token", request_id=rid)
                else:
                    self.gaps.append(self.now - record.last)
                record.last = self.now
                record.generated += 1
                record.low_tokens += int(self.mode == PrecisionMode.LOW)
                if record.generated == min(record.request.output_tokens, s.max_new_tokens):
                    record.status = (
                        "ok"
                        if record.generated == record.request.output_tokens
                        else "length_limited"
                    )
                    record.terminal = self.now
                    if record.status == "length_limited":
                        record.reason = "max_new_tokens"
                    self.active.pop(rid)
                    live -= blocks(record.request.input_tokens + record.generated, s.block_tokens)
                    reserved -= self.bound(record.request)
                    self.event("request_finished", request_id=rid, status=record.status)
            self.event(
                "iteration_finished",
                live_blocks=live,
                peak_blocks=next_live,
                reserved_blocks=reserved,
                capacity_blocks=capacity,
                queued=len(self.waiting),
                active=len(self.active),
            )
        return ReplayResult(
            [r.row(self.horizon) for r in self.records], self.gaps, self.transitions, self.decisions
        )
