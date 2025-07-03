import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

# ----------------- RMSNorm -----------------
# -------------------------------------------
from Q3Nexus_Ops import rmsnorm_bf16xbf16, fused_add_rmsnorm_bf16xbf16
class RMSNorm(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
    def forward(
        self, 
        x: torch.Tensor, 
        residual: Optional[torch.tensor] = None
    ) -> Union[torch.tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            fused_add_rmsnorm_bf16xbf16(x, residual, self.weight, x, residual)
            return x, residual
        out = rmsnorm_bf16xbf16(x, self.weight, self.variance_epsilon)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_size={self.hidden_size}, eps={self.variance_epsilon})"


# ----------------- MLP -----------------
# ---------------------------------------
from Q3Nexus_Ops import silu_bf16xbf16
class MLP(nn.Module):
    def __init__ (
        self,
        hidden_size: int,
        intermediate_size: int
    ) -> None :
        super().__init__()
        self.fused_gate_up_weight = nn.Parameter(torch.ones([2 * intermediate_size, hidden_size]))
        self.down_weight = nn.Parameter(torch.ones([hidden_size, intermediate_size]))
    def forward(
        self, 
        hidden_states
    ) -> torch.Tensor:
        hidden_states = torch.matmul(hidden_states, self.fused_gate_up_weight.T)    # hidden_states [batch, seq, 2 * intermediate_size]
        hidden_states = silu_bf16xbf16(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.down_weight.T)
        return hidden_states

if __name__ == "__main__":
    pass