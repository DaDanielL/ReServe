"""Plot smoke-run arrivals, completions, and latency. Requires the analysis extra."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
with (args.run_dir / "requests.csv").open() as handle:
    rows = list(csv.DictReader(handle))
successful = [row for row in rows if row["status"] == "ok"]
fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
axes[0].step(
    [float(row["arrival_s"]) for row in rows],
    range(1, len(rows) + 1),
    where="post",
    label="Arrivals",
)
finished = sorted(float(row["finished_s"]) for row in successful)
axes[0].step(finished, range(1, len(finished) + 1), where="post", label="Completions")
axes[0].set_ylabel("Cumulative requests")
axes[0].legend()
axes[1].scatter(
    [float(row["arrival_s"]) for row in successful],
    [float(row["e2e_s"]) for row in successful],
    s=18,
)
axes[1].set(xlabel="Seconds since start", ylabel="End-to-end latency (s)")
fig.suptitle("MOCK setup check - artificial delays, no LLM performance results")
destination = args.run_dir / "timeline.png"
fig.savefig(destination, dpi=150)
print(destination)
