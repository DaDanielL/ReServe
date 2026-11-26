# ReServe

Recovery-Aware Scheduling and KV-Cache Placement for Dynamic-Precision LLM Serving.

ESE6800 research starter for Daniel Lin and Zichen Fu.

**Research question:** Can request scheduling and KV-cache placement shorten the
return to higher weight precision after a traffic burst, while preserving
deadline goodput and fair treatment of long requests?

## Run the actual Qwen scheduler

Install the PyTorch build for your CPU or NVIDIA driver, then run from this branch:

```bash
python -m pip install -e ".[live]"
python -m recovery_serving.live --device cuda:0 --output results/qwen-local
```

The command downloads the pinned Qwen2.5-1.5B-Instruct checkpoint and public workload, profiles disjoint prompts, freezes request deadlines, and executes all six policies on one device. For CPU use `--device cpu`. The first GPU run uses FP16. The flags `--source qwen-math` and `--deadline-mode artifact` select the public Qwen math cohort and the JITServe stored benchmark budgets, respectively. Artifact budgets apply to JITServe prompts. [Local run guide and paper evidence](docs/live-experiment.md) · [Kaggle GPU notebook](notebooks/kaggle_live.ipynb)

## What works now

- Seeded Poisson arrivals across steady, burst, and recovery traffic phases.
- A CPU-only mock backend and FIFO / earliest-deadline-first client admission.
- Per-request timings, an event log, configuration and Git metadata, and summary metrics.
- A Kaggle launcher, plotting script, environment check, and GitHub CPU checks.
- A deterministic CPU replay with deadline-slack, drain, and recovery-aware admission.
- Six matched comparisons, synthetic bursts, a BurstGPT CSV reader, per-request and recovery metrics, and plots.

The mock backend uses artificial sleeps to check the client harness. The replay backend
models individual prefill and decode steps on a logical clock. Its default GPU timings,
cache capacities, and serialized weight copy are fixtures. For measured model execution and live precision switching use the Qwen command above.

## Run the scheduler comparison

From the repository root with Python 3.10 or newer:

```bash
python -m pip install -e '.[analysis,dev]'
recovery-run --config configs/scheduler.yaml --output results/scheduler-42
python scripts/plot_replay.py results/scheduler-42 --output results/figures-42
```

Each variant receives the same calibration data, arrivals, deadlines, and observation
window. The run saves its inputs, per-policy events and request rows, and a comparison
summary. The plotting command produces an occupancy timeline, recovery versus goodput,
and a CSV of paired differences. See the [scheduler and evaluation guide](docs/scheduler.md)
for definitions, replay assumptions, BurstGPT commands, and the device interface.

## Start locally

Use Python 3.10+; the smoke harness has one small dependency and needs no GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis,dev]'
python -m unittest discover -s tests -v
recovery-run --config configs/smoke.yaml --output results/smoke-001
python scripts/plot_run.py results/smoke-001
```

On Windows, activate with `.venv\Scripts\activate`. Run all commands from the
repository root. Every run needs a new output directory; existing results are
never overwritten. Without an editable install, use `PYTHONPATH=src python -m
recovery_serving.run` in place of `recovery-run` on Linux/macOS.

## Project map

| Path | Purpose |
| --- | --- |
| `configs/` | Versioned experiment settings |
| `src/recovery_serving/workload.py` | Synthetic and BurstGPT request traces |
| `src/recovery_serving/scheduling.py` | Client priorities and iteration-level admission policies |
| `src/recovery_serving/precision.py` | Future switching and memory-state interface |
| `src/recovery_serving/backends/` | Mock execution and real-engine integration boundary |
| `src/recovery_serving/run.py` | CLI for the mock and replay backends |
| `src/recovery_serving/replay.py` | Logical-clock engine and reference memory controller |
| `src/recovery_serving/experiment.py` | Paired comparison, input hashes, and output files |
| `src/recovery_serving/metrics.py` | Service, fairness, and sustained recovery metrics |
| `notebooks/kaggle_start.ipynb` | Thin Kaggle launcher |
| `scripts/` | Environment check and analysis |
| `tests/` | CPU checks for scheduling, traces, reporting, and the runner |
| `docs/` | Kaggle setup, implementation boundaries, experiment plan |
| `data/`, `results/` | Local inputs and outputs, ignored by Git |

## GPU dependencies

Check `python scripts/check_environment.py` in Kaggle first. Keep both partners on
the same GPU type for comparative measurements. Begin with one GPU and a small
dense model that fits at the higher precision with room for KV cache.

Choose an initial development path:

```bash
# Hugging Face prototype / fixed-precision baselines
python -m pip install -e '.[hf,monitor,analysis]'

# OR a vLLM serving baseline / eventual engine fork
python -m pip install -e '.[vllm,monitor,analysis]'
```

These extras are dependency declarations, **not a tested GPU lockfile**. vLLM,
PyTorch, CUDA, and quantization kernels must be compatible with the actual Kaggle
GPU. Avoid independently upgrading PyTorch after installing vLLM. Once one setup
works, save `python -m pip freeze` with its run outputs and document the working
versions in `docs/kaggle.md`. Restart the notebook kernel if needed after installation.

## Mock harness outputs

Each directory contains `run.json`, `workload.jsonl`, `events.jsonl`,
`requests.csv`, and `summary.json`. `run.json` records the config, relevant package
versions, Git commit, dirty status, and the fact that this is a mock run.

Deadlines are measured from planned request arrival. The producer submits arrivals
independently of prior completions; requests wait in a priority queue when all
client workers are busy. In the supplied configs every request has the same
deadline budget, so EDF and FIFO have identical ordering. Vary budgets by request
class when studying heterogeneous deadlines.

In replay, the simulator alone receives true output lengths. The scheduler receives
observable progress and estimates remaining length from a separate calibration set.

## Work together

Keep shared logic in Python modules and launch it from notebooks. Use short feature
branches, pull requests, and the checklist in [CONTRIBUTING.md](CONTRIBUTING.md).
Keep credentials, model weights, datasets, and generated runs out of Git. Store
tokens in Kaggle Secrets and retain experiment outputs outside a live session.

See [Kaggle setup](docs/kaggle.md), [architecture](docs/architecture.md), and the
[experiment plan](docs/experiment_plan.md). The first real milestone is a
fixed-precision model serving a repeatable burst with measured KV occupancy and
latency, before attempting live precision switching.
