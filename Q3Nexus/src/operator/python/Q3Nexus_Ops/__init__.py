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