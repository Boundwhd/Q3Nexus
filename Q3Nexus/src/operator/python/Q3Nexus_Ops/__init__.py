import torch
from . import _C

def rmsnorm_bf16xbf16(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    return _C.rmsnorm_bf16xbf16(hidden_states, weight, epsilon)