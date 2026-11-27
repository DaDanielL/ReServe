import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recovery_serving.config import load_config
from recovery_serving.experiment import prepare
from recovery_serving.live import parser, supplied_budgets, validate
from recovery_serving.live_benchmark import run_policy
from recovery_serving.live_data import Encoded, Profile, Prompt, arrival_trace, split_prompts
from recovery_serving.precision import PrecisionMode
from recovery_serving.scheduling import OnlineRequest
from scripts.run_reserve_experiments import live_runs

ROOT = Path(__file__).resolve().parents[1]


def row(text, i, length=2):
    return Prompt(str(i), text, None, (length,), 1.25)


class DeadlinesAndData(unittest.TestCase):
    def test_grouped_disjoint_split(self):
        prompts = [
            row("one plus two", 1),
            row("one   plus two", 2),
            row("x", 3),
            row("an overlong prompt", 4),
            row("ok", 5),
        ]
        cal, test = split_prompts(prompts, lambda s: [0] * len(s), 1, 2, 12, 99)
        self.assertEqual(len({p.group for p in cal + test}), 3)
        self.assertTrue(all(len(p.tokens) <= 12 for p in cal + test))

    def test_only_calibration_determines_device_target(self):
        observations = [
            {"input_tokens": 3, "generated_tokens": n, "ttft_s": 1.0, "tpot_s": 0.2}
            for n in (2, 3, 4, 5, 6)
        ]
        prof = Profile(observations)
        p = Encoded(row("hello", 3), [1, 2, 3], "g")
        d, length = prof.assign(p, 10.0, "device-p95")
        self.assertEqual(length, 5)
        self.assertAlmostEqual(d, 11.8)
        hidden = replace(
            p, source=replace(p.source, reference_outputs=(1000,), artifact_budget_s=100)
        )
        self.assertEqual(prof.assign(hidden, 10.0, "device-p95"), (d, length))
        self.assertEqual(prof.assign(p, 10.0, "artifact")[0], 11.25)
        self.assertEqual(prof.assign(p, 10.0, "jitserve-20s")[0], 30.0)

    def test_one_token_cohort_uses_measured_decode_only_for_extrapolation(self):
        obs = [{"input_tokens": 2, "generated_tokens": 1, "ttft_s": 0.25, "tpot_s": None}]
        prof = Profile(obs, decode_probe_s=0.05)
        request = OnlineRequest("test", 0, 1, 2)
        self.assertEqual(prof.service(request, 1), 0.25)
        self.assertAlmostEqual(prof.service(request, 3), 0.35)

    def test_jitserve_only_budgets_and_input_validation(self):
        a = parser().parse_args(
            ["--source", "qwen-math", "--deadline-mode", "artifact", "--output", "test"]
        )
        with self.assertRaisesRegex(ValueError, "Only JITServe"):
            validate(a)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budgets.csv"
            path.write_text("source_id,prompt_group,budget_s\na,h,0.5\n")
            self.assertEqual(supplied_budgets(path), {("a", "h"): 0.5})
            path.write_text("source_id,prompt_group,budget_s\na,h,nan\n")
            with self.assertRaises(ValueError):
                supplied_budgets(path)

    def test_real_trace_gaps_failures_and_replay_deadlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"]
                )
                for t, p, n in [
                    (1, 3, 2),
                    (2, 3, 6),
                    (3, 3, 0),
                    (10, 3, 1),
                    (10, 3, 99),
                    (20, 3, 1),
                ]:
                    writer.writerow([t, "ChatGPT", p, n, "Conversation log"])
            with patch("recovery_serving.live_data.source_file", return_value=path):
                trace = arrival_trace(Path(tmp), 3, 3, 0.5)
            self.assertEqual([r["arrival_s"] for r in trace], [0.0, 0.0, 5.0])
            cfg = load_config(ROOT / "configs/scheduler-burstgpt.yaml")
            w = cfg["workload"]
            w["path"] = str(path)
            w["burstgpt"].update(
                calibration_window_s=[0, 4], evaluation_window_s=[10, 21], time_scale=1.0
            )
            requests, _, _, _, _ = prepare(cfg)
            self.assertEqual(
                [round(r.deadline_s - r.arrival_s, 5) for r in requests], [0.75, 0.75, 0.75]
            )


class ToyEngine:
    device = SimpleNamespace(type="cpu")
    eos = {9}
    kv_bytes_per_token = 1
    saved_bytes = 64

    def set_mode(self, mode):
        self.mode = PrecisionMode(mode)

    def device_memory(self):
        return {}

    def step(self, token_lists, caches):
        # One prompt finishes at the first forward. The other hits the output cap.
        return [9 if ids[0] == 1 else 5 for ids in token_lists], [True] * len(caches)


class LiveExecution(unittest.TestCase):
    def test_current_and_historical_policy_names_in_commands_and_reports(self):
        for time_name, space_name in (("time-based", "space-based"), ("cost", "recovery")):
            with self.subTest(time=time_name), tempfile.TemporaryDirectory() as tmp:
                args = parser().parse_args(["--output", tmp, "--variants", time_name, space_name])
                validate(args)
                self.assertEqual(args.variants, ["time-based", "space-based"])
                folder = Path(tmp)
                stats = {f"rep-02-{name}": {"requests": 24} for name in (time_name, space_name)}
                (folder / "summary.json").write_text(json.dumps(stats))
                indexed = live_runs(folder)
                for name, saved in zip(args.variants, stats):
                    self.assertEqual(indexed[2, name], (folder / saved, stats[saved]))

    def test_wall_clock_scoring_counts_capped_requests_and_recovery(self):
        prompts = [Encoded(row("a", 1), [1], "x"), Encoded(row("b", 2), [2], "y")]
        online = [OnlineRequest(p.source.source_id, 0.0, 10.0, 1) for p in prompts]
        prof = Profile([{"input_tokens": 1, "generated_tokens": 2, "ttft_s": 0.1, "tpot_s": 0.1}])
        a = parser().parse_args(
            ["--output", "dummy", "--down-queue", "2", "--up-queue", "0", "--max-new-tokens", "3"]
        )
        events = []
        rows, summary = run_policy(
            ToyEngine(),
            prompts,
            online,
            {"low": prof, "high": prof},
            "dynamic",
            "space-based",
            a,
            [(0, 0.1)],
            events.append,
        )
        self.assertEqual([r["status"] for r in rows], ["eos", "length_cap"])
        self.assertEqual(summary["deadline_met"], 1)
        self.assertEqual(summary["deadline_attainment"], 0.5)
        self.assertEqual(summary["deadline_goodput_rps"], 0.1)
        self.assertIn("high_ready", [e["event"] for e in events])
        self.assertIn([2], [e.get("forward_batches") for e in events])


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "install the live extra for weight and KV tests",
)
class ModelExecution(unittest.TestCase):
    def test_mixed_mode_cache_and_grouped_decode_match_independent_forwards(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from recovery_serving.live_model import QwenEngine, SwitchableLinear

        torch.manual_seed(7)
        engine = object.__new__(QwenEngine)
        engine.device = torch.device("cpu")
        cfg = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        engine.model = Qwen2ForCausalLM(cfg).eval()
        layer = engine.model.model.layers[0].mlp
        original = layer.gate_proj.weight.detach().clone()
        layer.gate_proj = SwitchableLinear(layer.gate_proj)
        engine.swappable = [layer.gate_proj]
        engine.mode = PrecisionMode.HIGH
        a, b = [1, 3, 5], [1, 7, 2]
        initial, caches = engine.step([a, b], [None, None])
        solo = [engine.step([p], [None]) for p in (a, b)]
        self.assertEqual(initial, [s[0][0] for s in solo])
        engine.set_mode("low")
        both, caches = engine.step([[v] for v in initial], caches)
        separate = [engine.step([[s[0][0]]], s[1]) for s in solo]
        self.assertEqual(both, [s[0][0] for s in separate])
        self.assertIsNone(layer.gate_proj.high)
        engine.set_mode("high")
        self.assertTrue(torch.equal(layer.gate_proj.high, original))
        after, _ = engine.step([[v] for v in both], caches)
        self.assertEqual(after, [engine.step([[s[0][0]]], s[1])[0][0] for s in separate])
