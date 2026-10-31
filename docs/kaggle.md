# Kaggle workflow

## First session

1. Create a private Kaggle notebook and select an available GPU for real inference.
   The CPU smoke check does not need GPU allocation.
2. Put a working copy of this repository under `/kaggle/working/ReServe`.
   Use a public clone URL only for a public repository. For a private repository,
   a simple starting route is to download its ZIP yourself, attach it as a private
   Kaggle Dataset, and extract/copy it into `/kaggle/working`. Kaggle input mounts
   are read-only. Never place a personal token inside a clone URL or notebook cell.
3. Open/upload `notebooks/kaggle_start.ipynb`. Set `REPO_DIR` if your path differs.
4. Run the environment check, base installation, smoke run, and plot cells.
5. Verify saved files before ending the session. Save a notebook version with
   outputs or download the run directory; do not depend on a live session's files.

Use Kaggle Secrets for any Hugging Face or GitHub credentials needed by your
chosen workflow. Do not print them. The starter launcher itself needs no secrets.

## Real inference environment

The launcher installs only the base package and plotting dependency. Select the
HF or vLLM optional dependencies deliberately after recording the actual GPU.
Some GPU/quantization combinations are unsupported. T4 has no native FP8
computation, and lower-precision weight execution may use different kernel paths.

Install one working GPU stack, restart the kernel when appropriate, and save:

```bash
python scripts/check_environment.py > results/environment.txt
python -m pip freeze > results/requirements-kaggle.txt
```

The environment checker prints hardware and package metadata, not tokens. Copy
the tested versions below once a real inference setup works. CPU checks alone do
not certify this combination.

| Component | Tested value |
| --- | --- |
| Kaggle accelerator | TODO |
| Python / PyTorch / CUDA | TODO |
| Inference engine + exact version or fork commit | TODO |
| Model ID + revision | TODO |
| Quantization method / kernel | TODO |

Keep both partners on the same hardware and stack for comparisons. If Kaggle
assigns two GPUs, keep the first implementation single-GPU and verify that the
engine actually uses only that GPU.
