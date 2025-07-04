import torch
import time
import numpy as np
from typing import Tuple

# 假设已经编译好的自定义算子库
from Q3Nexus_Ops import rope_cos_sin_bf16

def pytorch_rope_reference(
    positions: torch.Tensor,  # [seq_len]
    inv_freq: torch.Tensor,   # [head_dim//2]
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch参考实现（单序列版本）"""
    # 扩展维度 [seq_len, head_dim//2]
    freqs = positions.float()[:, None] * inv_freq[None, :]
    
    # 复制频率以填充整个头维度
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    
    # 计算cos和sin
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    
    return cos, sin

def test_rope_single_sequence(positions_list: list, head_dim: int = 128):
    """测试单个序列的旋转位置编码"""
    # 创建inv_freq在CPU上（避免在GPU上生成影响计时）
    inv_freq_cpu = 1.0 / (10000 ** (torch.arange(0, head_dim//2) / (head_dim//2)))
    inv_freq = inv_freq_cpu.to(torch.bfloat16).cuda()
    
    for positions in positions_list:
        # 转换为张量
        pos_tensor = torch.tensor(positions, dtype=torch.int32).cuda()
        seq_len = len(positions)
        
        print(f"\n{'='*60}")
        print(f" 测试场景: positions={positions}")
        print(f" 序列长度: {seq_len}, 头维度: {head_dim}")
        print('='*60)
        
        # 预热
        for _ in range(10):
            cos_ref, sin_ref = pytorch_rope_reference(pos_tensor, inv_freq, head_dim)
            cos_custom = torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
            sin_custom = torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
            rope_cos_sin_bf16(pos_tensor, inv_freq, cos_custom, sin_custom)
        
        # 参考实现
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            cos_ref, sin_ref = pytorch_rope_reference(pos_tensor, inv_freq, head_dim)
        torch.cuda.synchronize()
        ref_time = (time.time() - start_time) / 100 * 1000
        
        # 自定义算子
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            rope_cos_sin_bf16(
                pos_tensor, 
                inv_freq, 
                torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda"),
                torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
            )
        torch.cuda.synchronize()
        custom_time = (time.time() - start_time) / 100 * 1000
        
        # 计算误差
        cos_custom = torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        sin_custom = torch.empty(seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
        rope_cos_sin_bf16(pos_tensor, inv_freq, cos_custom, sin_custom)
        cos_ref, sin_ref = pytorch_rope_reference(pos_tensor, inv_freq, head_dim)
        
        # 计算误差
        cos_error = torch.max(torch.abs(cos_custom - cos_ref)).item()
        sin_error = torch.max(torch.abs(sin_custom - sin_ref)).item()
        
        # 打印结果
        print(f"PyTorch参考实现耗时: {ref_time:.4f} ms")
        print(f"自定义算子耗时:     {custom_time:.4f} ms")
        print(f"加速比:          {ref_time / custom_time:.2f}x")
        print(f"最大COS误差:      {cos_error:.6f}")
        print(f"最大SIN误差:      {sin_error:.6f}")
        
        # 打印位置0的结果（前5个元素）
        print("\n位置0的前5个元素比较 (COS):")
        print(f"  PyTorch: {cos_ref[0, :5].cpu().numpy().round(4)}")
        print(f"  Custom:  {cos_custom[0, :5].cpu().numpy().round(4)}")
        
        # 打印最后一个位置的结果（前5个元素）
        if seq_len > 1:
            last_idx = seq_len - 1
            print(f"\n位置{last_idx}的前5个元素比较 (COS):")
            print(f"  PyTorch: {cos_ref[last_idx, :5].cpu().numpy().round(4)}")
            print(f"  Custom:  {cos_custom[last_idx, :5].cpu().numpy().round(4)}")
        
        print("-"*60)

if __name__ == "__main__":
    # 配置头维度
    HEAD_DIM = 128
    
    # 定义测试场景
    test_cases = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 长序列
        [11]                                  # 短序列（单位置）
    ]
    
    print("\n" + "="*50)
    print(" 旋转位置编码 (RoPE) 算子测试 ")
    print("="*50)
    
    # 运行测试
    test_rope_single_sequence(test_cases, HEAD_DIM)
    
    print("\n测试完成！")