import csv
import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

from recovery_serving.config import load_config
from recovery_serving.experiment import run_experiment
from recovery_serving.metrics import recovery_episodes, summarize_replay
from recovery_serving.precision import PrecisionMode
from recovery_serving.replay import Replay, ReplaySettings, StepTiming
from recovery_serving.scheduling import (
    AdmissionState,
    LengthPredictor,
    OnlineRequest,
    RecoveryAwareScheduler,
)
from recovery_serving.workload import Request, burstgpt_trace, peak_window, steady_control

ROOT = Path(__file__).resolve().parents[1]


def reference_settings(**changes):
    s = ReplaySettings(**load_config(ROOT / "configs/scheduler.yaml")["replay"])
    s = replace(
        s,
        block_tokens=1,
        high_blocks=100,
        low_blocks=160,
        staging_blocks=5,
        max_batch=2,
        batch_tokens=180,
        max_new_tokens=2,
        down_queue=2,
        up_queue=1,
        min_low_s=0,
        restore_s=0.25,
        sustain_s=0.5,
        long_output_tokens=2,
    )
    return replace(s, **changes)


def simulate(trace, settings=None, horizon=6, mode="dynamic", scheduler=None):
    s = settings or reference_settings()
    timing = StepTiming(1.0, 0, 0)
    events = []
    scheduler = scheduler or RecoveryAwareScheduler(LengthPredictor([(39, 1), (89, 2)]), 1, 3)
    result = Replay(
        trace, horizon, s, {"high": timing, "low": timing}, scheduler, mode, "cost", events.append
    ).run()
    return result, events


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.model = LengthPredictor([(16, 1), (64, 100), (128, 100)])
        self.scheduler = RecoveryAwareScheduler(self.model, lookahead=4, max_wait_s=3)
        self.state = AdmissionState(1, PrecisionMode.LOW, True, 1, 200, 40, 80, 20, 8, 128, 0.1)
        self.running = [OnlineRequest("live", 0, 100, 128, 8, 17)]
        self.big = OnlineRequest("big", 0.5, 6, 64)
        self.small = OnlineRequest("small", 0.6, 7, 16)

    def select(self, policy, state=None, candidates=None):
        return self.scheduler.schedule(
            candidates or [self.big, self.small],
            self.running,
            state or self.state,
            lambda req, n: n * 0.01,
            policy,
        )

    def test_forecast_accounts_for_survival_and_entire_cache_release(self):
        model = LengthPredictor([(32, 2), (32, 4), (32, 10)])
        req = OnlineRequest("r", 0, 10, 32, 2, 5)
        self.assertEqual(model.remaining(req), 8)
        # Two requests survived 2 outputs. Only the 10-token one survives to step 5.
        self.assertEqual(model.projected_blocks(req, 3, 8), 2.5)
        self.assertEqual(model.projected_blocks(replace(req, generated=4), 6, 8), 0)
        self.assertGreater(model.projected_blocks(replace(req, generated=100), 10, 8), 0)

    def test_common_ranking_and_distinct_recovery_guards(self):
        self.assertEqual(self.select("cost"), ["big"])
        self.assertEqual(self.select("drain"), [])
        self.assertEqual(self.select("recovery"), ["small"])
        for policy in ("drain", "recovery"):
            self.assertEqual(
                self.select(policy, replace(self.state, recovery_pressure=False)), ["big"]
            )
            self.assertEqual(
                self.select(policy, replace(self.state, mode=PrecisionMode.HIGH)), ["big"]
            )

    def test_urgent_and_aged_requests_bypass_preference_but_obey_capacity(self):
        urgent = replace(self.big, deadline_s=2.05)  # 0.05s slack, 0.1s to the next step.
        aged = replace(self.big, arrival_s=0, deadline_s=99)
        for policy in ("drain", "recovery"):
            self.assertEqual(self.select(policy, candidates=[urgent]), ["big"])
            self.assertEqual(self.select(policy, replace(self.state, now_s=4), [aged]), ["big"])
            self.assertEqual(
                self.select(policy, replace(self.state, available_blocks=1), [urgent]), []
            )
        self.assertEqual(self.select("cost", replace(self.state, prefill_tokens=20)), ["small"])
        self.assertEqual(self.select("cost", replace(self.state, reserve_blocks=1)), [])

    def test_aged_request_accumulates_room_while_the_running_batch_finishes(self):
        aged = replace(self.big, arrival_s=0)
        young = replace(self.small, arrival_s=3.9)
        state = replace(self.state, now_s=4, prefill_tokens=20)
        for policy in ("cost", "drain", "recovery"):
            self.assertEqual(self.select(policy, state, [aged, young]), [])
        self.assertEqual(self.select("fifo", state, [aged, young]), ["small"])

    def test_selected_candidates_share_one_forecast_and_budget(self):
        state = replace(self.state, slots=2, high_target_blocks=29)
        big2 = replace(self.big, request_id="big2", deadline_s=6.5)
        self.assertEqual(self.select("recovery", state, [self.big, big2]), ["big"])
        self.assertEqual(self.select("cost", state, [self.big, big2]), ["big", "big2"])


class ReplayTests(unittest.TestCase):
    def test_130_to_90_blocks_then_weights_ready_after_copy(self):
        trace = [
            Request("A", 0, 39, 1, 6, "burst"),
            Request("B", 0, 89, 2, 6, "burst"),
            Request("C", 1.1, 5, 1, 6, "tail"),
        ]
        result, events = simulate(trace)
        first = next(e for e in events if e["event"] == "iteration_finished")
        self.assertEqual((first["peak_blocks"], first["live_blocks"]), (130, 90))
        stamps = {e["event"]: e["t_s"] for e in events}
        self.assertEqual(stamps["memory_eligible"], 1)
        self.assertEqual(stamps["high_ready"], 1.25)
        self.assertEqual([r["started_s"] for r in result.rows], [0, 0, 1.25])
        self.assertEqual([r["ttft_s"] for r in result.rows][:2], [1, 1])
        self.assertEqual(result.gaps, [1.25])
        self.assertEqual([r["low_tokens"] for r in result.rows], [1, 1, 0])
        self.assertEqual([e["t_s"] for e in events], sorted(e["t_s"] for e in events))
        fast, _ = simulate(trace, reference_settings(restore_s=0))
        self.assertIn((1, "high_ready"), fast.transitions)

    def test_predictor_never_receives_length_phase_or_future_burst(self):
        seen = []

        class Observer(RecoveryAwareScheduler):
            def schedule(self, waiting, running, state, service_seconds, policy="recovery"):
                seen.append((state.now_s, tuple(waiting), tuple(running)))
                return super().schedule(waiting, running, state, service_seconds, policy)

        observer = Observer(LengthPredictor([(8, 2)]), 1, 3)
        a = Request("a", 0, 8, 2, 9, "burst")
        simulate([a], horizon=0.5, scheduler=observer)
        first = list(seen)
        seen.clear()
        simulate([replace(a, output_tokens=99, phase="secret")], horizon=0.5, scheduler=observer)
        self.assertEqual(first, seen)
        self.assertEqual(
            {f.name for f in fields(OnlineRequest)},
            {"request_id", "input_tokens", "arrival_s", "deadline_s", "generated", "live_blocks"},
        )

    def test_timeout_rejection_capping_and_unfinished_stay_in_denominator(self):
        s = reference_settings(max_batch=1, queue_timeout_s=0.5)
        trace = [
            Request("one", 0, 1, 1, 0.9, "s"),
            Request("wait", 0.1, 1, 1, 10, "s"),
            Request("reject", 0.2, 999, 1, 10, "s"),
            Request("cap", 1.1, 1, 3, 10, "s"),
            Request("partial", 3.2, 1, 2, 10, "s"),
        ]
        result, _ = simulate(trace, s, horizon=4.4, mode="high")
        summary = summarize_replay(result, 4.4, [], s, "high")
        self.assertEqual(
            [r["status"] for r in result.rows],
            ["ok", "timed_out", "rejected", "length_limited", "unfinished"],
        )
        self.assertEqual(summary["submitted"], 5)
        self.assertEqual(summary["deadline_attainment"], 0)
        self.assertEqual(summary["deadline_goodput_rps"], 0)
        self.assertEqual(summary["generated_tokens"], 4)
        self.assertEqual(summary["by_output_length"]["long"]["offered"], 2)
        self.assertEqual(summary["timed_out"], 1)

    def test_early_no_switch_censored_and_idle_recovery(self):
        cases = [
            ([], "no_switch", None),
            ([(0.1, "low_active"), (0.5, "high_ready")], "recovered", -0.5),
            ([(0.3, "low_active"), (1.9, "high_ready")], "censored", None),
            ([(1.25, "low_active"), (1.5, "high_ready")], "recovered", 0.5),
            ([(0.1, "low_active"), (0.2, "high_ready"), (0.3, "low_active")], "censored", None),
        ]
        for changes, status, delta in cases:
            row = recovery_episodes([(0, 1)], changes, 2, 0.25)[0]
            self.assertEqual((row["status"], row["recovery_s"]), (status, delta))
        trace = [Request("a", 0, 1, 1, 10, "burst"), Request("b", 0, 1, 1, 10, "burst")]
        result, _ = simulate(trace, reference_settings(min_low_s=2), horizon=3)
        self.assertIn((2.25, "high_ready"), result.transitions)
        # A new burst censors the preceding episode before it can satisfy its dwell.
        episodes = recovery_episodes(
            [(0, 1), (1.2, 1.3)], [(0, "low_active"), (1.1, "high_ready")], 3, 0.3
        )
        self.assertEqual([e["status"] for e in episodes], ["censored", "no_switch"])


class WorkloadTests(unittest.TestCase):
    def test_burstgpt_disjoint_windows_failures_stable_ties_and_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"]
                )
                writer.writerows(
                    [
                        (0, "M", 9, 2, "C"),
                        (2, "M", 4, 0, "C"),
                        (10, "M", 8, 1, "C"),
                        (10, "M", 7, 2, "C"),
                        (14, "M", 5, 0, "C"),
                        (19, "M", 8, 2, "C"),
                        (19, "M", 3, 2, "API"),
                        (20, "M", 1, 1, "C"),
                    ]
                )
            settings = {
                "model": "M",
                "log_type": "C",
                "time_scale": 0.1,
                "calibration_window_s": [0, 10],
                "evaluation_window_s": [10, 20],
            }
            trace, cal, window, meta = burstgpt_trace(path, settings, 0.5)
            self.assertEqual(cal, [(9, 2)])
            self.assertEqual([r.arrival_s for r in trace], [0, 0, 0.9])
            self.assertEqual([r.input_tokens for r in trace], [8, 7, 8])
            self.assertEqual(window, 1)
            self.assertEqual(meta["excluded_source_failures"], {"evaluation": 1, "calibration": 1})
            self.assertEqual(peak_window(trace, window, 0.25, 2), [(0, 0.25)])
            steady = steady_control(trace, window)
            self.assertEqual([r.output_tokens for r in steady], [1, 2, 2])
            self.assertAlmostEqual(steady[-1].deadline_s - steady[-1].arrival_s, 0.5)
            with self.assertRaisesRegex(ValueError, "overlap"):
                burstgpt_trace(path, {**settings, "calibration_window_s": [0, 11]}, 0.5)

    def test_matrix_uses_one_workload_and_rejects_reused_output(self):
        config = load_config(ROOT / "configs/scheduler.yaml", scenario="mild")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix"
            runs = run_experiment(config, path, {})
            self.assertEqual(
                set(runs), {"fixed_high", "fixed_low", "pressure", "cost", "drain", "recovery"}
            )
            cohorts = []
            for name, stats in runs.items():
                with (path / name / "requests.csv").open() as handle:
                    cohorts.append(
                        [
                            (r["request_id"], r["deadline_s"], r["output_tokens"])
                            for r in csv.DictReader(handle)
                        ]
                    )
                self.assertEqual(stats["duration_s"], runs["cost"]["duration_s"])
            self.assertTrue(all(x == cohorts[0] for x in cohorts))
            self.assertTrue(json.loads((path / "run.json").read_text())["calibration_sha256"])
            with self.assertRaises(FileExistsError):
                run_experiment(config, path, {})
            with self.assertRaisesRegex(ValueError, "distinct"):
                run_experiment({**config, "seed": config["calibration_seed"]}, path, {})
