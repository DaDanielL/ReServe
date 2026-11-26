"""Wall-clock iteration admission and measurement on one actual model."""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .live_data import Encoded
from .metrics import percentile, recovery_episodes
from .precision import PrecisionMode
from .scheduling import AdmissionState, OnlineRequest, RecoveryAwareScheduler, blocks


@dataclass
class Job:
    prompt: Encoded
    arrival: float
    deadline: float
    cache: object = None
    tokens: list = field(default_factory=list)
    started: float | None = None
    first: float | None = None
    last: float | None = None
    low_tokens: int = 0
    max_gap: float = 0.0

    @property
    def key(self):
        return self.prompt.source.source_id

    def view(self, block_size):
        n = len(self.prompt.tokens)
        emitted = len(self.tokens)
        return OnlineRequest(
            self.key,
            self.arrival,
            self.deadline,
            n,
            emitted,
            blocks(n + emitted - 1, block_size) if emitted else 0,
        )

    def row(self, engine):
        n = len(self.tokens)
        eos = self.tokens[-1] in engine.eos
        return {
            "source_id": self.key,
            "prompt_group": self.prompt.group,
            "arrival_s": self.arrival,
            "deadline_s": self.deadline,
            "started_s": self.started,
            "first_token_s": self.first,
            "finished_s": self.last,
            "queue_s": self.started - self.arrival,
            "ttft_s": self.first - self.arrival,
            "tpot_s": (self.last - self.first) / (n - 1) if n > 1 else None,
            "max_intertoken_gap_s": self.max_gap if n > 1 else None,
            "e2e_s": self.last - self.arrival,
            "input_tokens": len(self.prompt.tokens),
            "generated_tokens": n,
            "low_tokens": self.low_tokens,
            "token_ids": self.tokens,
            "status": "eos" if eos else "length_cap",
            "deadline_met": eos and self.last <= self.deadline,
        }


def forward_group(engine, group, clock):
    ids = [job.tokens[-1:] if job.tokens else job.prompt.tokens for job in group]
    caches = [job.cache for job in group]
    start = clock()
    tokens, past = engine.step(ids, caches)
    end = clock()
    for job, token, cache in zip(group, tokens, past):
        if job.first is None:
            job.started = start
            job.first = end
        else:
            job.max_gap = max(job.max_gap, end - job.last)
        job.tokens.append(token)
        job.cache = cache
        job.last = end
        job.low_tokens += int(engine.mode == PrecisionMode.LOW)


def warmup(engine, prompt):
    """Exercise both forward paths. Keep a measured decode fallback for one-token cohorts."""
    first, cache = engine.step([prompt.tokens], [None])
    start = time.perf_counter()
    engine.step([[first[0]]], cache)
    return time.perf_counter() - start


def profile_prompts(engine, prompts, limit):
    rows = []
    for prompt in prompts:
        job = Job(prompt, 0.0, 0.0)
        start = time.perf_counter()
        while len(job.tokens) < limit:
            forward_group(engine, [job], lambda: time.perf_counter() - start)
            if job.tokens[-1] in engine.eos:
                break
        row = job.row(engine)
        rows.append(
            {
                k: row[k]
                for k in (
                    "source_id",
                    "prompt_group",
                    "input_tokens",
                    "generated_tokens",
                    "ttft_s",
                    "tpot_s",
                    "e2e_s",
                    "status",
                    "token_ids",
                )
            }
        )
    return rows


def run_policy(engine, prompts, online, profiles, device_mode, policy, opts, bursts, emit):
    """Inputs in online have public IDs, prompt length, arrivals and frozen deadlines only."""
    mode = PrecisionMode.LOW if device_mode == "low" else PrecisionMode.HIGH
    engine.set_mode(mode)
    jobs = [Job(p, r.arrival_s, r.deadline_s) for p, r in zip(prompts, online)]
    block = 16
    high = opts.high_kv_tokens // block
    extra = engine.saved_bytes // (block * engine.kv_bytes_per_token)
    low = high + extra
    if high <= 0 or any(
        blocks(len(j.prompt.tokens) + opts.max_new_tokens, block) > high for j in jobs
    ):
        raise ValueError("high-kv-tokens must fit the largest retained prompt plus its output cap")
    if engine.device.type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(engine.device)
    scheduler = RecoveryAwareScheduler(profiles["high"].predictor, opts.lookahead, opts.max_wait)
    waiting, active = {}, {}
    transitions, completed, decision_times = [], [], []
    next_job = 0
    intent = False
    target = max(1, high - opts.staging_blocks)
    start = time.perf_counter()

    def clock():
        return time.perf_counter() - start

    def event(name, **kw):
        t = clock()
        emit({"event": name, "time_s": t, "mode": engine.mode.value, **kw})
        if name in {"low_active", "restore_requested", "high_ready"}:
            transitions.append((t, name))

    def switch(to):
        before = clock()
        event("switch_started", to=to.value, **engine.device_memory())
        engine.set_mode(to)
        duration = clock() - before
        event(
            "high_ready" if to == PrecisionMode.HIGH else "low_active",
            transfer_s=duration,
            **engine.device_memory(),
        )

    event("initial_mode", high_blocks=high, low_blocks=low, **engine.device_memory())
    while next_job < len(jobs) or waiting or active:
        now = clock()
        while next_job < len(jobs) and jobs[next_job].arrival <= now:
            j = jobs[next_job]
            waiting[j.key] = j
            next_job += 1
        running = [j.view(block) for j in active.values()]
        next_live = sum(blocks(r.input_tokens + r.generated, block) for r in running)
        if device_mode == "dynamic":
            if engine.mode == PrecisionMode.HIGH and (
                len(waiting) >= opts.down_queue or next_live > high
            ):
                switch(PrecisionMode.LOW)
                intent = False
            if engine.mode == PrecisionMode.LOW:
                wants_high = len(waiting) <= opts.up_queue
                if wants_high and not intent:
                    event("restore_requested")
                intent = wants_high
                if intent and next_live <= target:
                    switch(PrecisionMode.HIGH)
                    intent = False
        if not waiting and not active:
            if next_job < len(jobs):
                time.sleep(max(0, jobs[next_job].arrival - clock()))
            continue
        capacity = low if engine.mode == PrecisionMode.LOW else high
        bound = high if device_mode == "high" else low
        reserved = sum(blocks(r.input_tokens + opts.max_new_tokens, block) for r in running)
        state = AdmissionState(
            clock(),
            engine.mode,
            intent,
            opts.batch_size - len(active),
            opts.batch_size * opts.max_prompt_tokens,
            capacity - next_live,
            bound - reserved,
            target,
            block,
            opts.max_new_tokens,
            profiles[engine.mode.value].service(running[0], 1) if running else 0.0,
        )
        decision_start = time.perf_counter_ns()
        admit = scheduler.schedule(
            [j.view(block) for j in waiting.values()],
            running,
            state,
            profiles[engine.mode.value].service,
            policy,
        )
        decision_times.append(time.perf_counter_ns() - decision_start)
        for key in admit:
            active[key] = waiting.pop(key)
        if not active:
            raise RuntimeError("No admissible work on an empty device")
        # Equal-length forwards use real tensor batching with unpadded KV. Other groups
        # run sequentially. Give already-streaming requests their decode step first.
        groups = defaultdict(list)
        for j in active.values():
            n = len(j.prompt.tokens)
            group_key = (0, n + len(j.tokens) - 1) if j.tokens else (1, n)
            groups[group_key].append(j)
        step_kv = sum(len(j.prompt.tokens) + len(j.tokens) for j in active.values())
        for _, group in sorted(groups.items()):
            forward_group(engine, group, clock)
            for j in group:
                if j.tokens[-1] in engine.eos or len(j.tokens) == opts.max_new_tokens:
                    completed.append(j.row(engine))
                    j.cache = None
                    del active[j.key]
        event(
            "iteration",
            admitted=admit,
            waiting=len(waiting),
            running=len(active),
            forward_batches=[len(g) for _, g in sorted(groups.items())],
            kv_tokens_before_release=step_kv,
            kv_tokens=sum(len(j.prompt.tokens) + len(j.tokens) - 1 for j in active.values()),
            **engine.device_memory(),
        )
    if device_mode == "dynamic" and engine.mode == PrecisionMode.LOW:
        if not intent:
            event("restore_requested")
        switch(PrecisionMode.HIGH)
    elapsed = clock()
    by_source = {r["source_id"]: r for r in completed}
    rows = [by_source[j.key] for j in jobs]
    ontime = sum(r["deadline_met"] for r in rows)
    horizon = max(r.deadline_s for r in online)
    summary = {
        "requests": len(rows),
        "natural_eos": sum(r["status"] == "eos" for r in rows),
        "length_capped": sum(r["status"] == "length_cap" for r in rows),
        "deadline_met": ontime,
        "deadline_attainment": ontime / len(rows),
        "deadline_goodput_rps": ontime / horizon,
        "common_deadline_window_s": horizon,
        "elapsed_through_drain_s": elapsed,
        "generated_tokens_per_s": sum(r["generated_tokens"] for r in rows) / elapsed,
        "p95_ttft_s": percentile([r["ttft_s"] for r in rows], 0.95),
        "p95_tpot_s": percentile([r["tpot_s"] for r in rows if r["tpot_s"] is not None], 0.95),
        "p95_queue_s": percentile([r["queue_s"] for r in rows], 0.95),
        "p95_e2e_natural_s": percentile([r["e2e_s"] for r in rows if r["status"] == "eos"], 0.95),
        "decision_p95_us": percentile(decision_times, 0.95) / 1000,
        "low_tokens": sum(r["low_tokens"] for r in rows),
        "low_switches": sum(name == "low_active" for _, name in transitions),
        "burst_recovery": recovery_episodes(
            bursts, transitions, elapsed, 0.0, device_mode == "dynamic"
        ),
        **engine.device_memory(),
    }
    return rows, summary
