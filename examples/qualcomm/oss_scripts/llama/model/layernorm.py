# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import nn


class Norm(torch.nn.Module, ABC):
    """Abstract base class for normalization layers with unified interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for normalization layer.
        Args:
            x: Input tensor
        Returns:
            Normalized tensor
        """
        pass


NORM_REGISTRY: Dict[str, Type[Norm]] = {}


def register_norm(name: str):
    """Decorator to register norm classes"""

    def decorator(cls: Type[Norm]):
        NORM_REGISTRY[name] = cls
        return cls

    return decorator


@register_norm("layernorm")
class LayerNorm(torch.nn.LayerNorm, Norm):
    def __init__(self, hidden_size: int, eps=1e-5):
        super().__init__(hidden_size, eps=eps)


@register_norm("gemma2")
@register_norm("gemma3")
@register_norm("rmsnorm")
class RMSNorm(torch.nn.RMSNorm, Norm):
    def __init__(self, hidden_size: int, eps=1e-5):
        super().__init__(hidden_size, eps=eps)


@register_norm("qwen3_5")
class Qwen35RMSNorm(Norm):
    def __init__(
        self, hidden_size: int, eps: float = 1e-5, add_unit_offset: bool = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        output = x_float * torch.rsqrt(
            (x_float * x_float).mean(-1, keepdim=True) + self.eps
        )
        if self.add_unit_offset:
            return (output * (1.0 + self.weight.float())).to(x.dtype)
        return (output * self.weight.float()).to(x.dtype)


class Qwen35RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
