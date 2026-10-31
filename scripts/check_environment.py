"""Print non-secret environment details before starting a Kaggle experiment."""

import importlib.metadata
import json
import platform
import subprocess

info = {"python": platform.python_version(), "platform": platform.platform()}
for package in ("torch", "transformers", "vllm", "bitsandbytes"):
    try:
        info[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        info[package] = "not installed"
try:
    import torch

    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = torch.version.cuda
    info["gpus"] = [
        {
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "total_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
except ImportError:
    pass
print(json.dumps(info, indent=2))
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi unavailable: normal on a CPU-only development machine")
