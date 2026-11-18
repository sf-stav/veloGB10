"""
Compile CUDA kernels to PTX for GB10 (sm_121).
Requires nvcc from CUDA 12.1 or later.
"""
import subprocess
import sys
import os
from pathlib import Path

KERNELS = [
    ("gemm_nvfp4.cu", "gemm_nvfp4.ptx"),
    ("fused_decode.cu", "fused_decode.ptx"),
    ("rms_norm.cu", "rms_norm.ptx"),
    ("silu_gate.cu", "silu_gate.ptx"),
]

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC = Path(CUDA_HOME) / "bin" / "nvcc"
KERNEL_DIR = Path(__file__).parent
OUT_DIR = Path(__file__).parent.parent / "src" / "ptx"

ARCH = "--gpu-architecture=sm_121"
LINK = "--default-stream=per-thread"


def compile_kernel(src: Path, dst: Path) -> bool:
    cmd = [
        str(NVCC),
        ARCH,
        "--ptx",
        LINK,
        "-o", str(dst),
        str(src),
    ]
    print(f"Compiling {src.name} -> {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {src.name}")
        print(result.stderr)
        return False
    print(f"OK: {dst.name}")
    return True


def main():
    if not NVCC.exists():
        print(f"ERROR: nvcc not found at {NVCC}")
        print("Install CUDA toolkit and set CUDA_HOME")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for src_name, dst_name in KERNELS:
        src = KERNEL_DIR / src_name
        dst = OUT_DIR / dst_name
        if not src.exists():
            print(f"WARNING: {src} not found")
            continue
        if not compile_kernel(src, dst):
            all_ok = False

    if all_ok:
        print("\nAll kernels compiled successfully!")
        print(f"PTX files are in: {OUT_DIR}")
    else:
        print("\nSome kernels failed to compile")
        sys.exit(1)


if __name__ == "__main__":
    main()
