import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from recovery_serving.config import load_config
from recovery_serving.metrics import summarize
from recovery_serving.run import run
from recovery_serving.scheduling import priority
from recovery_serving.workload import Request, generate_trace

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_trace_is_reproducible_and_within_phase_boundaries(self):
        config = load_config(ROOT / "configs/smoke.yaml")
        trace = generate_trace(config)
        self.assertEqual(trace, generate_trace(config))
        self.assertGreater(len(trace), 0)
        self.assertEqual([r.arrival_s for r in trace], sorted(r.arrival_s for r in trace))
        self.assertEqual(len({r.request_id for r in trace}), len(trace))
        boundaries = {"steady": (0, 0.15), "burst": (0.15, 0.45), "recovery": (0.45, 0.60)}
        for request in trace:
            lower, upper = boundaries[request.phase]
            self.assertTrue(lower <= request.arrival_s < upper)
            self.assertGreater(request.deadline_s, request.arrival_s)

    def test_edf_can_prioritize_a_later_arrival_with_an_earlier_deadline(self):
        a = Request("a", 0, 8, 8, 10, "test")
        b = Request("b", 1, 8, 8, 2, "test")
        self.assertLess(priority(a, "fifo"), priority(b, "fifo"))
        self.assertLess(priority(b, "edf"), priority(a, "edf"))

    def test_failure_remains_in_deadline_attainment_denominator(self):
        rows = [
            {"status": "ok", "deadline_met": True, "e2e_s": 0.1, "ttft_s": 0.02},
            {"status": "error", "deadline_met": False},
        ]
        summary = summarize(rows, 2)
        self.assertEqual(summary["deadline_attainment"], 0.5)
        self.assertEqual(summary["deadline_goodput_rps"], 0.5)
        self.assertEqual(summary["failed"], 1)
        self.assertIsNone(summarize([], 1)["p95_ttft_s"])

    def test_invalid_concurrency_is_rejected(self):
        config = load_config(ROOT / "configs/smoke.yaml")
        config["concurrency"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_run_writes_trace_results_and_refuses_overwrite(self):
        config = load_config(ROOT / "configs/smoke.yaml")
        config["mock"] = {"prefill_s": 0, "token_s": 0}
        config["workload"]["phases"] = [{"name": "tiny", "duration_s": 0.002, "rate_rps": 3000}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = asyncio.run(run(config, output))
            expected = len(generate_trace(config))
            self.assertEqual(summary["submitted"], expected)
            self.assertEqual(summary["completed"], expected)
            with (output / "requests.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected)
            self.assertTrue(all(float(row["queue_s"]) >= 0 for row in rows))
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["measurement_kind"], "artificial_setup_check")
            with self.assertRaises(FileExistsError):
                asyncio.run(run(config, output))

    def test_backend_failure_does_not_deadlock_the_queue(self):
        config = load_config(ROOT / "configs/smoke.yaml")
        config["workload"]["phases"] = [{"name": "tiny", "duration_s": 0.002, "rate_rps": 3000}]

        async def fail(*args):
            raise RuntimeError("intentional test failure")

        with tempfile.TemporaryDirectory() as directory:
            with patch("recovery_serving.run.MockBackend.generate", new=fail):
                summary = asyncio.run(run(config, Path(directory) / "run"))
            self.assertGreater(summary["failed"], 0)
            self.assertEqual(summary["completed"], 0)


if __name__ == "__main__":
    unittest.main()
