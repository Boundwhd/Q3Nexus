import torch
from typing import Tuple
from . import _C

def rmsnorm_bf16xbf16(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    return _C.rmsnorm_bf16xbf16(hidden_states, weight, epsilon)


def fused_add_rmsnorm_bf16xbf16(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out_hidden_states: torch.Tensor,
    out_residual: torch.Tensor,
    epsilon: float = 1e-6
) -> None:
    return _C.fused_add_rmsnorm_bf16xbf16(hidden_states, residual, weight, out_hidden_states, out_residual, epsilon)


def linear_bf16xbf16(
    hidden_states: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    return _C.linear_bf16xbf16(hidden_states, weight)

def silu_bf16xbf16(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return _C.silu_bf16xbf16(hidden_states)

def rope_cos_sin_bf16(
    positions: torch.Tensor,    # [0, 1, 2, 3, 4] or [5]
    inv_freq: torch.Tensor,     # [head_dim / 2]
    cos: torch.Tensor,          # [5, head_dim] or [1, head_dim]
    sin: torch.Tensor,          # [5, head_dim] or [1, head_dim]
) -> None:
    return _C.rope_cos_sin_bf16(positions, inv_freq, cos, sin)