import torch
import time
import random
import numpy as np
from Q3Nexus_Ops import rmsnorm_bf16xbf16, fused_add_rmsnorm_bf16xbf16

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

BATCH_SIZE = 32
SEQ_LEN = 1024
HIDDEN_SIZE = 2048
NUM_WARMUPS = 10
NUM_RUNS = 100
EPSILON = 1e-6

def rmsnorm_torch(hidden_states, weight, epsilon=1e-6):
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states

def generate_test_tensors(dtype=torch.bfloat16):
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device='cuda')
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(HIDDEN_SIZE, dtype=dtype, device='cuda')
    return hidden_states, residual, weight

def test_rmsnorm():
    print("="*50)
    print("Testing rmsnorm_bf16xbf16")
    print("="*50)
    
    hidden_states, _, weight = generate_test_tensors()
    
    for _ in range(NUM_WARMUPS):
        _ = rmsnorm_bf16xbf16(hidden_states, weight, EPSILON)
        _ = rmsnorm_torch(hidden_states, weight, EPSILON)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        custom_out = rmsnorm_bf16xbf16(hidden_states, weight, EPSILON)
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / NUM_RUNS * 1000
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        torch_out = rmsnorm_torch(hidden_states, weight, EPSILON)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / NUM_RUNS * 1000 
    
    speedup = torch_time / custom_time
    print(f"Custom operator time: {custom_time:.3f} ms per run")
    print(f"PyTorch implementation time: {torch_time:.3f} ms per run")
    print(f"Speedup: {speedup:.2f}x")
    print("-"*50)
    
    num_samples = 10
    indices = [(random.randint(0, BATCH_SIZE-1), 
                random.randint(0, SEQ_LEN-1), 
                random.randint(0, HIDDEN_SIZE-1)) 
               for _ in range(num_samples)]
    
    print("\nRandom sample comparison:")
    print(f"{'Index':>15} | {'Custom':>15} | {'PyTorch':>15} | {'AbsDiff':>12} | {'RelDiff':>10}")
    for i, (b, s, h) in enumerate(indices):
        c_val = custom_out[b, s, h].item()
        t_val = torch_out[b, s, h].item()
        abs_d = abs(c_val - t_val)
        rel_d = abs_d / (abs(t_val) + 1e-9)
        print(f"({b:3},{s:3},{h:4}) | {c_val:15.8e} | {t_val:15.8e} | {abs_d:12.4e} | {rel_d:10.4e}")

def test_fused_add_rmsnorm():
    print("\n" + "="*50)
    print("Testing fused_add_rmsnorm_bf16xbf16")
    print("="*50)
    
    hidden_states, residual, weight = generate_test_tensors()
    
    custom_out_hidden = torch.empty_like(hidden_states)
    custom_out_residual = torch.empty_like(hidden_states)
    
    for _ in range(NUM_WARMUPS):
        fused_add_rmsnorm_bf16xbf16(
            hidden_states, residual, weight, 
            custom_out_hidden, custom_out_residual, EPSILON
        )
        torch_residual_out = hidden_states + residual
        torch_hidden_out = rmsnorm_torch(torch_residual_out, weight, EPSILON)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        fused_add_rmsnorm_bf16xbf16(
            hidden_states, residual, weight, 
            custom_out_hidden, custom_out_residual, EPSILON
        )
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) / NUM_RUNS * 1000
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(NUM_RUNS):
        torch_residual_out = hidden_states + residual
        torch_hidden_out = rmsnorm_torch(torch_residual_out, weight, EPSILON)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / NUM_RUNS * 1000  # ms per run
    
    speedup = torch_time / custom_time
    print(f"Custom operator time: {custom_time:.6f} ms per run")
    print(f"PyTorch implementation time: {torch_time:.6f} ms per run")
    print(f"Speedup: {speedup:.2f}x")
    print("-"*50)
    
    torch_residual_out = hidden_states + residual
    torch_hidden_out = rmsnorm_torch(torch_residual_out, weight, EPSILON)
    
    num_samples = 5
    indices = [(random.randint(0, BATCH_SIZE-1), 
                random.randint(0, SEQ_LEN-1), 
                random.randint(0, HIDDEN_SIZE-1)) 
               for _ in range(num_samples)]
    
    print("\nRandom sample comparison (Hidden States):")
    print(f"{'Index':>15} | {'Custom':>15} | {'PyTorch':>15} | {'AbsDiff':>12} | {'RelDiff':>10}")
    for i, (b, s, h) in enumerate(indices):
        c_val = custom_out_hidden[b, s, h].item()
        t_val = torch_hidden_out[b, s, h].item()
        abs_d = abs(c_val - t_val)
        rel_d = abs_d / (abs(t_val) + 1e-9)
        print(f"({b:3},{s:3},{h:4}) | {c_val:15.8e} | {t_val:15.8e} | {abs_d:12.4e} | {rel_d:10.4e}")
    
    print("\nRandom sample comparison (Residual Output):")
    print(f"{'Index':>15} | {'Custom':>15} | {'PyTorch':>15} | {'AbsDiff':>12} | {'RelDiff':>10}")
    for i, (b, s, h) in enumerate(indices):
        c_val = custom_out_residual[b, s, h].item()
        t_val = torch_residual_out[b, s, h].item()
        abs_d = abs(c_val - t_val)
        rel_d = abs_d / (abs(t_val) + 1e-9)
        print(f"({b:3},{s:3},{h:4}) | {c_val:15.8e} | {t_val:15.8e} | {abs_d:12.4e} | {rel_d:10.4e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        exit(1)
        
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    
    test_rmsnorm()
    test_fused_add_rmsnorm()