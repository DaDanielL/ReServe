# Working together

- Create a feature branch, for example `feature/workload-replay` or `feature/kv-metrics`.
- Keep commits small enough for your partner to review.
- Use Python modules for experiment logic; keep notebooks as launchers and analysis views.
- Before a pull request, run:

```bash
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

For an experiment change, record the config, trace seed, model revision, GPU type,
dependency versions, and Git commit. Run baselines against the same arrival trace.
Compare FIFO/EDF under heterogeneous deadlines before interpreting any difference.
Treat simulated or mocked measurements as such in every figure and report.

Suggested first split: one partner owns fixed-precision serving and GPU profiling;
the other owns trace replay, logging, and baseline policies. Agree on event fields
before connecting the two. Neither part requires implementing the research policy first.

GitHub CPU checks do not verify GPU compatibility. Run a small Kaggle smoke check
for changes to model execution, quantization, memory allocation, or CUDA streams.

No license has been selected. Decide together before distributing the research code
under an open-source license.
