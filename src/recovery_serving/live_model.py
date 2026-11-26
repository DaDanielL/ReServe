"""Single-device Qwen runtime with persistent per-request KV and reversible MLP weight swaps."""

import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .live_data import MODEL, REVISION, digest
from .precision import PrecisionMode

# Hash the files actually used by the fast tokenizer, EOS configuration and model.
CHECKPOINT = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}


def checkpoint(folder):
    if folder is None:
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=list(CHECKPOINT)))
    for name, expected in CHECKPOINT.items():
        if digest(folder / name) != expected:
            raise ValueError(f"{name} differs from the pinned Qwen checkpoint")
    return folder


class SwitchableLinear(nn.Module):
    """Symmetric per-output-channel W4. Unpack one tile into the compute dtype at a time."""

    tile_rows = 256

    def __init__(self, linear):
        super().__init__()
        master = linear.weight.detach().cpu()
        if linear.bias is not None or master.size(1) % 2:
            raise ValueError("Expected a bias-free Qwen MLP with even input width")
        packed = torch.empty((master.size(0), master.size(1) // 2), dtype=torch.uint8)
        scales = torch.empty((master.size(0), 1), dtype=master.dtype)
        for start in range(0, master.size(0), self.tile_rows):
            segment = slice(start, start + self.tile_rows)
            tile = master[segment].float()
            scale = tile.abs().amax(dim=1, keepdim=True) / 7
            scale = torch.where(scale == 0, 1.0, scale)
            codes = ((tile / scale).round().clamp(-7, 7) + 8).to(torch.uint8)
            packed[segment] = codes[:, ::2] | (codes[:, 1::2] << 4)
            scales[segment] = scale
        # CPU backing copies are outside the model's registered device buffers.
        self.host = (master, packed, scales)
        self.register_buffer("high", master, persistent=False)
        self.register_buffer("packed", None, persistent=False)
        self.register_buffer("scales", None, persistent=False)
        self.saved_bytes = master.nbytes - packed.nbytes - scales.nbytes

    def activate(self, mode, device):
        self.high = self.packed = self.scales = None
        master, packed, scales = self.host
        if mode == PrecisionMode.HIGH:
            self.high = master.to(device)
        else:
            self.packed = packed.to(device)
            self.scales = scales.to(device)

    def forward(self, hidden):
        if self.high is not None:
            return F.linear(hidden, self.high)
        pieces = []
        for start in range(0, self.packed.size(0), self.tile_rows):
            segment = slice(start, start + self.tile_rows)
            tile = self.packed[segment]
            pairs = torch.stack((tile & 15, tile >> 4), dim=-1).flatten(1)
            weight = (pairs.to(hidden.dtype) - 8) * self.scales[segment]
            pieces.append(F.linear(hidden, weight))
        return torch.cat(pieces, dim=-1)


class QwenEngine:
    def __init__(self, folder, device, dtype, swap_layers):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but PyTorch cannot see a GPU")
        if dtype == "auto":
            dtype = "float16" if self.device.type == "cuda" else "float32"
        if (
            self.device.type == "cuda"
            and dtype == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("This GPU does not support BF16. Use float16.")
        self.dtype = getattr(torch, dtype)
        folder = checkpoint(folder)
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            folder,
            torch_dtype=self.dtype,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.eos = set(self.model.generation_config.eos_token_id)
        layers = self.model.model.layers
        if not 1 <= swap_layers <= len(layers):
            raise ValueError(f"swap-layers must be between 1 and {len(layers)}")
        self.swappable = []
        for layer in layers[-swap_layers:]:
            for name in ("gate_proj", "up_proj", "down_proj"):
                replacement = SwitchableLinear(getattr(layer.mlp, name))
                setattr(layer.mlp, name, replacement)
                self.swappable.append(replacement)
        self.model.to(self.device)
        self.mode = PrecisionMode.HIGH
        self.sync()
        cfg = self.model.config
        self.kv_bytes_per_token = (
            2
            * len(layers)
            * cfg.num_key_value_heads
            * (cfg.hidden_size // cfg.num_attention_heads)
            * self.dtype.itemsize
        )
        self.saved_bytes = sum(layer.saved_bytes for layer in self.swappable)

    def encode(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
        )

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def set_mode(self, mode):
        mode = PrecisionMode(mode)
        if mode != self.mode:
            for layer in self.swappable:
                layer.activate(mode, self.device)
            self.sync()
            self.mode = mode

    def device_memory(self):
        if self.device.type != "cuda":
            return {}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
        }

    def describe(self):
        cuda = self.device.type == "cuda"
        gpu_inventory = None
        if cuda:
            try:
                gpu_inventory = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,uuid,driver_version,memory.total",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    .strip()
                    .splitlines()
                )
            except (OSError, subprocess.CalledProcessError):
                pass
        return {
            "gpu_inventory": gpu_inventory,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cuda_allocator_free_bytes_at_capture": torch.cuda.mem_get_info(self.device)[0]
            if cuda
            else None,
            "model": MODEL,
            "revision": REVISION,
            "checkpoint_sha256": CHECKPOINT,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device)
            if cuda
            else (platform.processor() or platform.machine()),
            "cuda_total_bytes": torch.cuda.get_device_properties(self.device).total_memory
            if cuda
            else None,
            "cuda_runtime": torch.version.cuda,
            "dtype": str(self.dtype),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "cpu_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "swap_layers": len(self.swappable) // 3,
            "w4_weight_bytes_saved": self.saved_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "attention": "sdpa",
        }

    @torch.inference_mode()
    def step(self, token_lists, caches):
        """One batched forward. Each member has the same input length and past length."""
        ids = torch.tensor(token_lists, dtype=torch.long, device=self.device)
        if caches[0] is None:
            past = None
        elif len(caches) == 1:
            past = caches[0]
        else:
            past = DynamicCache.from_batch_splits(caches)
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=1)
        next_ids = out.logits[:, -1].argmax(-1).tolist()
        cache = out.past_key_values
        per_request = [cache] if len(next_ids) == 1 else cache.batch_split(len(next_ids), 1)
        self.sync()
        return next_ids, per_request
