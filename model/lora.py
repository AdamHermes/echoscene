import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.reset_parameters()
        self.freeze_base()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        lora_delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        return self.base(x) + lora_delta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if weight_key in state_dict and prefix + "base.weight" not in state_dict:
            state_dict[prefix + "base.weight"] = state_dict.pop(weight_key)
        if bias_key in state_dict and prefix + "base.bias" not in state_dict:
            state_dict[prefix + "base.bias"] = state_dict.pop(bias_key)
        state_dict.setdefault(prefix + "lora_A", self.lora_A.detach())
        state_dict.setdefault(prefix + "lora_B", self.lora_B.detach())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class LoRAConv1d(nn.Module):
    def __init__(self, base_conv: nn.Conv1d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if base_conv.groups != 1:
            raise ValueError("LoRAConv1d only supports groups=1.")

        self.base = base_conv
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = base_conv.in_channels * base_conv.kernel_size[0]
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_conv.out_channels, rank))
        self.reset_parameters()
        self.freeze_base()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        weight = torch.matmul(self.lora_B, self.lora_A).view(
            self.base.out_channels,
            self.base.in_channels,
            self.base.kernel_size[0],
        )
        lora_delta = F.conv1d(
            self.dropout(x),
            weight * self.scale,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
        )
        return base_out + lora_delta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        if weight_key in state_dict and prefix + "base.weight" not in state_dict:
            state_dict[prefix + "base.weight"] = state_dict.pop(weight_key)
        if bias_key in state_dict and prefix + "base.bias" not in state_dict:
            state_dict[prefix + "base.bias"] = state_dict.pop(bias_key)
        state_dict.setdefault(prefix + "lora_A", self.lora_A.detach())
        state_dict.setdefault(prefix + "lora_B", self.lora_B.detach())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


def _is_lora_module(module: nn.Module) -> bool:
    return isinstance(module, (LoRALinear, LoRAConv1d))


def mark_only_lora_as_trainable(model: nn.Module) -> Tuple[int, int]:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if _is_lora_module(module):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if _is_lora_module(module):
            yield module.lora_A
            yield module.lora_B


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> int:
    from model.networks.diffusion_layout.denoise_net import AttentionBlock
    from model.networks.diffusion_shape.attention import CrossAttention

    injected = 0
    for module in model.modules():
        if isinstance(module, AttentionBlock):
            for child_name in ("qkv", "proj_out"):
                child = getattr(module, child_name, None)
                if isinstance(child, nn.Conv1d):
                    setattr(module, child_name, LoRAConv1d(child, rank=rank, alpha=alpha, dropout=dropout))
                    injected += 1

        if isinstance(module, CrossAttention):
            for child_name in ("to_q", "to_k", "to_v"):
                child = getattr(module, child_name, None)
                if isinstance(child, nn.Linear):
                    setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                    injected += 1

            if isinstance(module.to_out, nn.Sequential) and len(module.to_out) > 0:
                child = module.to_out[0]
                if isinstance(child, nn.Linear):
                    module.to_out[0] = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    injected += 1

    return injected
