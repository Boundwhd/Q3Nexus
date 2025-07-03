import torch
import time
import random
import numpy as np
from Q3Nexus_Ops import silu_bf16xbf16

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

batch_size       = 32
seq_len          = 1024
intermediate_dim = 512   
NUM_WARMUPS      = 10
NUM_RUNS         = 100
EPSILON          = 1e-3   

def pytorch_silu(hidden):
    """
    hidden: [B, S, 2*D] bfloat16
    return: [B, S, D] bfloat16
    """
    gate, up = hidden[..., :intermediate_dim], hidden[..., intermediate_dim:]
    gate_silu = (gate.float() / (1.0 + torch.exp(-gate.float()))).bfloat16()
    return (gate_silu.float() * up.float()).bfloat16()

def generate_test_tensor(dtype=torch.bfloat16):
    return torch.randn(batch_size, seq_len, 2*intermediate_dim,
                       dtype=dtype, device='cuda')

def test_silu():
    print("="*60)
    print(" Testing silu_bf16xbf16 ")
    print("="*60)

    hidden = generate_test_tensor()

    for _ in range(NUM_WARMUPS):
        custom_out = silu_bf16xbf16(hidden)
        torch_out  = pytorch_silu(hidden)

    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(NUM_RUNS):
        custom_out = silu_bf16xbf16(hidden)
    torch.cuda.synchronize()
    custom_time = (time.time() - start) / NUM_RUNS * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(NUM_RUNS):
        torch_out = pytorch_silu(hidden)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / NUM_RUNS * 1000

    speedup = torch_time / custom_time if custom_time > 0 else float('inf')
    print(f"Custom op time:    {custom_time:.4f} ms/run")
    print(f"PyTorch time:      {torch_time:.4f} ms/run")
    print(f"Speedup:           {speedup:.2f}x")
    print("-"*60)

    num_samples = 10
    indices = [
        (random.randrange(batch_size),
         random.randrange(seq_len),
         random.randrange(intermediate_dim))
        for _ in range(num_samples)
    ]

    print("\nRandom sample comparison:")
    print(f"{'Index':>15} | {'Custom':>15} | {'PyTorch':>15} | {'AbsDiff':>12} | {'RelDiff':>10}")
    for (b, s, d) in indices:
        c = custom_out[b, s, d].item()
        t = torch_out[b, s, d].item()
        abs_d = abs(c - t)
        rel_d = abs_d / (abs(t) + 1e-9)
        status = "OK" if abs_d < EPSILON else "FAIL"
        print(f"({b:2},{s:4},{d:4}) | {c:15.8e} | {t:15.8e} | {abs_d:12.4e} | {rel_d:10.4e} | {status}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        exit(1)
    torch.cuda.set_device(0)
    test_silu()
