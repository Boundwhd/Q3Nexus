import torch
import time
import random
import numpy as np
from Q3Nexus_Ops import linear_bf16xbf16

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

batch_size = 32
seq_len = 1024
in_dim = 1024
out_dim = 2048
NUM_WARMUPS = 10
NUM_RUNS = 100
EPSILON = 1e-6

def pytorch_linear(x, A):
    return torch.matmul(x, A.t())

def generate_test_tensors(dtype=torch.bfloat16):
    x = torch.randn(batch_size, seq_len, in_dim, dtype=dtype, device='cuda')
    A = torch.randn(out_dim, in_dim, dtype=dtype, device='cuda')
    return x, A

def test_linear():
    print("="*50)
    print("Testing linear_bf16xbf16")
    print("="*50)
    
    x, A = generate_test_tensors()

    for _ in range(NUM_WARMUPS):
        custom_out = linear_bf16xbf16(x, A)
        torch_out = pytorch_linear(x, A)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        custom_out = linear_bf16xbf16(x, A)
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / NUM_RUNS * 1000
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        torch_out = pytorch_linear(x, A)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / NUM_RUNS * 1000 
    
    speedup = torch_time / custom_time if custom_time > 0 else 0
    print(f"Custom operator time: {custom_time:.6f} ms per run")
    print(f"PyTorch implementation time: {torch_time:.6f} ms per run")
    print(f"Speedup: {speedup:.2f}x")
    print("-"*50)
    
    num_samples = 10
    indices = [(random.randint(0, batch_size-1), 
                random.randint(0, seq_len-1), 
                random.randint(0, out_dim-1)) 
               for _ in range(num_samples)]
    
    print("\nRandom sample comparison:")
    print(f"{'Index':>15} | {'Custom':>15} | {'PyTorch':>15} | {'AbsDiff':>12} | {'RelDiff':>10}")
    for i, (b, s, h) in enumerate(indices):
        c_val = custom_out[b, s, h].item()
        t_val = torch_out[b, s, h].item()
        abs_d = abs(c_val - t_val)
        rel_d = abs_d / (abs(t_val) + 1e-9)
        print(f"({b:3},{s:3},{h:4}) | {c_val:15.8e} | {t_val:15.8e} | {abs_d:12.4e} | {rel_d:10.4e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        exit(1)
        
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    
    test_linear()