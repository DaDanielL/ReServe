"""Pinned public workloads, disjoint prompt cohorts, and causal deadline calibration."""

import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .scheduling import LengthPredictor, OnlineRequest

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JIT = "ef3ab0c3f3cf42baa6dc3f928bee7675c2efa0c6"
BURST = "d895a53bb7b8ec137d0d2fe203b335835a78c10a"
MATH = "d22169cef97b94649b3811bc94051495f0331829"
MATH_DIR = "datasets_math500.jsonl--T-0.8--top_p-1.0--n-128--seed-96--agg_strategy-last"
SOURCES = {
    "jitserve": (
        "lmsys.json",
        f"https://raw.githubusercontent.com/UIUC-MLSys/JITServe/{JIT}/traces/lmsys.json",
        "134099f6cebab24ad48530701dd47e1d7c20c820d3ed9c460ffe4f811a2fcb0a",
    ),
    "burstgpt": (
        "BurstGPT_1.csv",
        f"https://raw.githubusercontent.com/HPMLL/BurstGPT/{BURST}/data/BurstGPT_1.csv",
        "46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a",
    ),
    "qwen-math": (
        "math.parquet",
        "https://huggingface.co/datasets/kheuton/Qwen2.5-1.5B-Instruct-bon-completions"
        f"/resolve/{MATH}/{MATH_DIR}/train-00000-of-00001.parquet",
        "600cc4225fbf670a889c3f1c665a18fb83225d6a4918c63b2938724bd4203e9a",
    ),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_file(key, folder):
    name, url, expected = SOURCES[key]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch for {path}. Supply the pinned source file.")
        return path
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as dest:
        temp = Path(dest.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temp.open("wb") as target:
            shutil.copyfileobj(source, target)
        if digest(temp) != expected:
            raise ValueError(f"Downloaded file does not match published input pin: {url}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class Prompt:
    source_id: str
    text: str
    reference_input: int | None
    reference_outputs: tuple[int, ...]
    artifact_budget_s: float | None = None


@dataclass(frozen=True)
class Encoded:
    source: Prompt
    tokens: list[int]
    group: str


def load_prompts(source, folder):
    path = source_file(source, folder)
    if source == "jitserve":
        # Type 1 is a single request whose complete answer has a time budget.
        # Type 0 has streaming semantics, type 2 has a dependency graph.
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        return [
            Prompt(
                f"lmsys:{i}", r["prompt"], r["prompt_len"], (r["output_len"],), r["deadline"] / 1000
            )
            for i, r in enumerate(rows)
            if r["request_type"] == 1
        ]
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["unique_id", "problem", "completion_tokens"])
    return [
        Prompt(r["unique_id"], r["problem"], None, tuple(r["completion_tokens"]))
        for r in rows.to_pylist()
    ]


def split_prompts(rows, encode, ncal, neval, max_prompt, seed):
    # Identical user text, including repeated rollouts, belongs to just one split.
    unique = {}
    for row in rows:
        normalized = " ".join(row.text.split())
        key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if normalized and key not in unique:
            unique[key] = row
    candidates = sorted(unique)
    random.Random(seed).shuffle(candidates)
    chosen = []
    for key in candidates:
        row = unique[key]
        ids = encode(row.text)
        if 0 < len(ids) <= max_prompt:
            chosen.append(Encoded(row, ids, key))
        if len(chosen) == ncal + neval:
            return chosen[:ncal], chosen[ncal:]
    raise ValueError(f"Found only {len(chosen)} distinct prompts within the context limit")


def arrival_trace(folder, count, start, scale):
    """Use successful real ChatGPT conversation rows, retaining simultaneous arrivals."""
    path = source_file("burstgpt", folder)
    selected = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if row["Model"] != "ChatGPT" or row["Log Type"] != "Conversation log":
                continue
            timestamp = int(row["Timestamp"])
            if timestamp < start or int(row["Response tokens"]) <= 0:
                continue
            if selected and timestamp < selected[-1]["original_s"]:
                raise ValueError("BurstGPT timestamps out of order")
            origin = selected[0]["original_s"] if selected else timestamp
            selected.append(
                {
                    "csv_line": line,
                    "original_s": timestamp,
                    "arrival_s": (timestamp - origin) * scale,
                    "burstgpt_input": int(row["Request tokens"]),
                    "burstgpt_output": int(row["Response tokens"]),
                }
            )
            if len(selected) == count:
                return selected
    raise ValueError(f"Only {len(selected)} eligible BurstGPT arrivals after {start}s")


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


class Profile:
    """Empirical solo prefill and decode times from a disjoint, measured cohort."""

    def __init__(self, rows, decode_probe_s=None):
        self.rows = rows
        self.predictor = LengthPredictor((r["input_tokens"], r["generated_tokens"]) for r in rows)
        self.gaps = [r["tpot_s"] for r in rows if r["tpot_s"] is not None]
        self.decode_probe_s = decode_probe_s

    def service(self, request, length, q=0.5):
        bucket = [
            r
            for r in self.rows
            if r["input_tokens"].bit_length() == request.input_tokens.bit_length()
        ]
        selected = bucket if len(bucket) >= 5 else self.rows
        if length == 1:
            return nearest_rank([r["ttft_s"] for r in selected], q)
        fallback = nearest_rank(self.gaps, 0.5) if self.gaps else self.decode_probe_s
        if fallback is None:
            raise ValueError("Extrapolating a multi-token answer requires a measured decode gap")
        times = [
            r["ttft_s"] + (length - 1) * (r["tpot_s"] if r["tpot_s"] is not None else fallback)
            for r in selected
        ]
        return nearest_rank(times, q)

    def assign(self, prompt, arrival, mode):
        online = OnlineRequest(prompt.source.source_id, arrival, 0.0, len(prompt.tokens))
        length = self.predictor.remaining(online)
        if mode == "device-p95":
            budget = self.service(online, length, 0.95)
        elif mode == "jitserve-20s":
            budget = 20.0
        elif mode == "artifact":
            budget = prompt.source.artifact_budget_s
            if budget is None:
                raise ValueError("Artifact budgets belong only to the JITServe workload")
        else:
            raise ValueError(f"Unknown deadline mode {mode}")
        return arrival + budget, length
