import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import Q3Nexus_Ops

class RMSnorm(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(
        self, 
        x: torch.Tensor, 
        residual: Optional[torch.tensor] = None
    ) -> Union[torch.tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            fused_add_rmsnorm_bf16xbf16(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        Q3Nexus_Ops.rmsnorm_bf16xbf16(x, self.weight, out)
        return out

        