import math
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora_update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + lora_update * self.scaling


def inject_lora(
    module: nn.Module,
    target_keywords: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    target_keywords = tuple(target_keywords)
    replaced = []

    def _inject(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and any(keyword in full_name for keyword in target_keywords):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                _inject(child, full_name)

    _inject(module)
    return replaced


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def mark_only_lora_trainable(module: nn.Module):
    for name, param in module.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)
