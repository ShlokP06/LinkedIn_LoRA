from typing import Any
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from src.lora.lora_linear import lora_state_dict
from safetensors.torch import save_file

def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)
    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dtype_from_str(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32 
    }[name]

def lr_lambda(step: int, warmup: int):
    if step < warmup:
        return float(step+1) / float(max(1, warmup))
    return 1.0

def sample_timesteps(batch_size, device):
    u = torch.randn(batch_size, device=device)
    t_norm = torch.sigmoid(u)
    t = (t_norm * 1000).long().clamp(1, 999)
    return t, t_norm

def add_noise(latents: torch.Tensor, noise: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
    t = t_norm.view(-1, 1, 1, 1)
    return (1-t)*latents + t * noise

def prune_old_checkpoints(output_dir: Path, max_keep: int, log) -> None:
    ckpts = sorted(output_dir.glob("lora_step_*.safetensors"))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        try:
            old.unlink()
            log.info(f"Pruned old checkpoint: {old.name}")
        except OSError:
            pass

def save_lora_checkpoint(
        model: torch.nn.Module,
        step: int, output_dir: Path, save_dtype: torch.dtype, max_keep: int, log
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict(model)
    sd = {k: v.detach().to(save_dtype).cpu().contiguous() for k, v in sd.items()}
    path = output_dir / f"lora_step_{step:06d}.safetensors"
    save_file(sd, str(path))
    prune_old_checkpoints(output_dir, max_keep, log)
    log.info(f"Saved LoRA checkpoint: {path.name} ({len(sd)} tensors.")
    return path


