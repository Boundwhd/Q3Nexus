import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "Ampere"

include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")
]

nvcc_optimization_flags = [
    # --------------------------
    # 基础优化选项
    # --------------------------
    "-O3",  # 最高级别优化
    "--use_fast_math",  # 启用快速数学库（牺牲精度换取速度）
    
    # --------------------------
    # 半精度计算支持
    # --------------------------
    "-U__CUDA_NO_HALF_OPERATORS__",  # 启用half类型的运算符
    "-U__CUDA_NO_HALF_CONVERSIONS__", # 启用half与其他类型的转换
    "-U__CUDA_NO_HALF2_OPERATORS__",  # 启用half2类型的运算符
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",  # 启用bfloat16支持
    
    # --------------------------
    # 实验性功能
    # --------------------------
    # "--expt-relaxed-constexpr",  # 放宽constexpr限制（允许设备函数中使用更多C++特性）
    # "--expt-extended-lambda",    # 支持扩展lambda功能（允许在GPU代码中使用更复杂的lambda）
    
    # --------------------------
    # 架构相关优化
    # --------------------------
    # "--ftz=true",       # 将非正规浮点数刷新为零（提高计算效率）
    # "--prec-div=false", # 禁用高精度除法（提高速度）
    # "--prec-sqrt=false",# 禁用高精度平方根（提高速度）
    # "--fmad=true",      # 启用融合乘加（FMA）优化
    
    # --------------------------
    # 调试和分析
    # --------------------------
    # "--ptxas-options=-v",  # 显示寄存器使用情况
    # "--source-in-ptx",     # 在PTX中保留源代码信息（便于调试）
    
    # --------------------------
    # 其他优化
    # --------------------------
    # "--default-stream=per-thread",  # 使用每线程默认流（提高多流性能）
    # "--extra-device-vectorization"  # 增强设备代码向量化
]

setup(
    name="Q3Nexus-Operator",
    version="1.0.0",
    description="Optimized Q3Nexus CUDA Operators for PyTorch",
    packages=["Q3Nexus_Ops"],
    package_dir={"": "python"},
    package_data={
        "Q3Nexus_Ops": ["*.pyi", "*.so"]
    },
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="Q3Nexus_Ops._C",  
            sources=[
                "python/bindings.cpp", 
                *[os.path.join("csrc", f) for f in os.listdir("csrc") if f.endswith(".cu")]
            ],
            include_dirs=include_dirs,
            # libraries=["cudart", "c10", "torch", "torch_cpu", "torch_python"],
            extra_compile_args={
                "cxx": [
                    "-O3"
                ],
                "nvcc": nvcc_optimization_flags
            },
            define_macros=[
                ("USE_CUDA", "1"),
                ("NDEBUG", "1")
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=True,
            build_dir="build"
        )
    },
    zip_safe=False
)