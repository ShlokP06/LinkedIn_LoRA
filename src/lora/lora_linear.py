import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, r:int, lora_alpha: int, lora_dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.original = original
        self.original.requires_grad_(False)
        self.lora_A = nn.Parameter(original.weight.new_zeros((r, original.in_features)))
        self.lora_B = nn.Parameter(original.weight.new_zeros((original.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        if self.merged:
            return base
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora_out
    
    def merge_weights(self):
        if not self.merged:
            self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerged_weights(self):
        if self.merged:
            self.original.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

TARGET_MODULES = ["to_ q", "to_v", "to_k", "to_out.0"]

def inject_lora(model: nn.Module, r: int, lora_alpha:int):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in TARGET_MODULES):
            continue
        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr, LoRALayer(module, r, lora_alpha))

    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(("lora_A", "lora_B"))

def lora_state_dict(model: nn.Module) -> dict:
    src = model.module if hasattr(model, "module") else model
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}

    
