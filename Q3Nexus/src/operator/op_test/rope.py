import torch
import time
import random
import numpy as np
from typing import Tuple
from Q3Nexus_Ops import rope_cos_sin_bf16

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

seq_len = 1024
head_dim = 128
NUM_WARMUPS = 10
NUM_RUNS = 100
EPSILON = 1e-3

def pytorch_rope_reference(
    positions: torch.Tensor,  
    inv_freq: torch.Tensor,
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)  
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    return cos, sin

def generate_test_tensors(dtype=torch.bfloat16):
    positions = torch.arange(seq_len, device='cuda', dtype=torch.int32)
    inv_freq = torch.randn(head_dim // 2, device='cuda', dtype=dtype)
    cos = torch.empty(seq_len, head_dim, device='cuda', dtype=dtype)
    sin = torch.empty(seq_len, head_dim, device='cuda', dtype=dtype)
    return positions, inv_freq, cos, sin

def test_rope_cos_sin():
    print("="*60)
    print(" Testing rope_cos_sin_bf16 ")
    print("="*60)

    positions, inv_freq, cos, sin = generate_test_tensors()
    
    for _ in range(NUM_WARMUPS):
        rope_cos_sin_bf16(positions, inv_freq, cos, sin)
        ref_cos, ref_sin = pytorch_rope_reference(positions.cpu(), inv_freq.cpu().float(), head_dim)

    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(NUM_RUNS):
        rope_cos_sin_bf16(positions, inv_freq, cos, sin)
    torch.cuda.synchronize()
    custom_time = (time.time() - start) / NUM_RUNS * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(NUM_RUNS):
        ref_cos, ref_sin = pytorch_rope_reference(positions.cpu(), inv_freq.cpu().float(), head_dim)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / NUM_RUNS * 1000

    speedup = ref_time / custom_time if custom_time > 0 else float('inf')
    print(f"Custom op time:    {custom_time:.4f} ms/run")
    print(f"Reference time:    {ref_time:.4f} ms/run")
    print(f"Speedup:           {speedup:.2f}x")
    print("-"*60)

    ref_cos, ref_sin = pytorch_rope_reference(positions.cpu(), inv_freq.cpu().float(), head_dim)
    ref_cos = ref_cos.to('cuda').bfloat16()
    ref_sin = ref_sin.to('cuda').bfloat16()

    num_samples = 10
    indices = [
        (random.randrange(seq_len),
         random.randrange(head_dim))
        for _ in range(num_samples)
    ]

    print("\nRandom sample comparison (cos):")
    print(f"{'Index':>12} | {'Custom':>12} | {'Reference':>12} | {'AbsDiff':>10} | {'RelDiff':>10}")
    for (s, d) in indices:
        c = cos[s, d].item()
        t = ref_cos[s, d].item()
        abs_d = abs(c - t)
        rel_d = abs_d / (abs(t) + 1e-9)
        status = "OK" if abs_d < EPSILON else "FAIL"
        print(f"({s:4},{d:4}) | {c:15.8e} | {t:15.8e} | {abs_d:12.4e} | {rel_d:10.4e} | {status}")

    print("\nRandom sample comparison (sin):")
    print(f"{'Index':>12} | {'Custom':>12} | {'Reference':>12} | {'AbsDiff':>10} | {'RelDiff':>10}")
    for (s, d) in indices:
        c = sin[s, d].item()
        t = ref_sin[s, d].item()
        abs_d = abs(c - t)
        rel_d = abs_d / (abs(t) + 1e-9)
        status = "OK" if abs_d < EPSILON else "FAIL"
        print(f"({s:4},{d:4}) | {c:15.8e} | {t:15.8e} | {abs_d:12.4e} | {rel_d:10.4e} | {status}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        exit(1)
    torch.cuda.set_device(0)
    test_rope_cos_sin()