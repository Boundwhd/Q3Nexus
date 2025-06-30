import torch
import time
from Q3Nexus_Ops import rmsnorm_bf16xbf16

torch.manual_seed(42)
x = torch.randn(8, 256, 1024, dtype=torch.bfloat16).cuda()
w = torch.randn(1024, dtype=torch.bfloat16).cuda()
epsilon = 1e-5

print("Randomly sampled 5 data points:")
for _ in range(5):
    i, j, k = torch.randint(0, 8, (1,)).item(), torch.randint(0, 256, (1,)).item(), torch.randint(0, 1024, (1,)).item()
    print(f"x[{i},{j},{k}] = {x[i,j,k].item():.4f}, w[{k}] = {w[k].item():.4f}")

start = time.time()
out = rmsnorm_bf16xbf16(x, w, epsilon)
torch.cuda.synchronize()
print(f"\nFirst execution time: {(time.time()-start)*1000:.3f} ms")

start = time.time()
for _ in range(100):
    out = rmsnorm_bf16xbf16(x, w, epsilon)
torch.cuda.synchronize()
print(f"Average execution time: {(time.time()-start)*1000/100:.3f} ms")

print("\nVerification on 5 random points:")
for _ in range(5):
    i, j = torch.randint(0, 8, (1,)).item(), torch.randint(0, 256, (1,)).item()
    cuda_result = out[i,j].mean().item()
    vec = x[i,j].float()
    var = vec.pow(2).mean()
    ref = (vec * torch.rsqrt(var + epsilon) * w.float()).mean()
    print(f"Position[{i},{j}] -> CUDA: {cuda_result:.4f} | Ref: {ref.item():.4f} | Diff: {abs(cuda_result - ref.item()):.4f}")