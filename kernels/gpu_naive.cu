// =============================================================================
// kernels/gpu_naive.cu — Naive-N0.5-Flash family kernels (PLAN/NAIVE_N05_PORT.md, S3).
//
// Owner directive 2026-10-16: Flash-Next code is off-limits. Everything here is a COPY of the
// engine's EXL3 kernels (kernels/exl3_bench.cu, S-A3-c..n) re-parameterized for this pack:
//   * half-integer trellis rates (3.5 bpw = 56 u16 words per 16x16 tile) next to the integer ones
//     (the 6-bit lm_head = 96 words). The template parameter is BPB, the bits per 16 weights (=
//     u16 words per tile): 48 / 56 / 64 / 80 / 96. The ONLY place a half rate differs from an
//     integer one is the window END of temporal index t (nv_win_end): integer K -> (t+1)*K,
//     half K (quantizer mask 0xAAAA: 3,4,3,4,... bits) -> S(t) = (t>>4)*BPB + ka*j + (j>>1),
//     j = (t&15)+1. Verified on the CPU (src/naive/frac.rs) against the exllamav3 frac.cu
//     semantics; with MASK = 0 the arithmetic is the engine's integer decoder bit for bit.
//   * the codebook (mul1), the dp4a decode, the tmap fragment placement, the orthonormal 128-block
//     Hadamard, the m16 x N128 HMMA tile, the K-ascending fp32 accumulation and the split-K
//     fixed-order combine are the engine's verbatim (batch invariance: AGENTS §2.4).
//
// Target: sm_121 baseline (fp16 mma.m16n8k16, dp4a, shuffles). Loaded by src/naive/gpu.rs only.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define NV_MUL1 0x83DCD12Du

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void nv_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

// PDL entry (every entry kernel: wait for the programmatic predecessor, release dependents; both
// are no-ops on a plain launch).
#define NV_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

__device__ __forceinline__ float nv_h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half nv_f2h(float f) { return __float2half_rn(f); }

#include "gpu_naive/01_decode_hadamard_gemm_wide.cuh"
#include "gpu_naive/02_layer_glue_attention.cuh"
#include "gpu_naive/03_gpu_sampling.cuh"
