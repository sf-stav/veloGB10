// =============================================================================
// kernels/gpu_dsv41.cu — the DeepSeek-V4.1-Flash kernel module `gpu_dsv41` (sm_121a).
// Loaded by src/dsv41/gpu/mod.rs only. The ABI (streams, PDL, format tags, cache layouts) is
// kernels/gpu_dsv41/00_abi.cuh; read it before adding a kernel.
//
// SECTION BANDS. Each kernel leg owns one band of file numbers in kernels/gpu_dsv41/ and its HEAD
// file below. The head file is included here automatically once it exists (__has_include), so a
// leg never edits this file: it creates its head file and includes its other band files from it
// (`#include "NN_more.cuh"`, same directory). Every *.cuh in the directory is part of
// KERNEL_BUILD_ID (build.rs), so an edit to any band file rebuilds and re-stamps the PTX.
//
//   band   head file              leg   what
//   01     01_abi_probe.cuh       I0    the ABI probe (build id, sizes, value-code decode)
//   02-04  02_fwd.cuh             SL    per-layer forward glue (row RoPE, SwiGLU, f32 + bf16 add-and-round)
//   05-09  05_tp.cuh              TPC   TP head exchange (vocab-parallel greedy / sample / verify), smoke producers; TP's 05_tp_fwd.cuh (forward glue) and FW's 06_fw.cuh (INT4/INT5 2026-11-19: SL's head was 05_fwd.cuh and moved to 02; 05-09 is TP's/FW's; the TP leg's report says 06_tp_fwd.cuh, the file is 05_tp_fwd.cuh)
//   10-19  10_trellis.cuh         KE    EXL3 trellis GEMV / coop / wide (K = 2..8), grouped MoE, plain linears
//   20-29  20_fp8_dense.cuh       KN1   native FP8 32x32 block GEMV / MMA, act_quant
//   30-39  30_mxfp4_moe.cuh       KN2   native MXFP4 experts with FP8 activations
//   40-49  40_kv.cuh              KC    compressor + main-KV / index-K / SWA writes
//   50-59  50_indexer.cuh         KI    indexer scoring, top-512, candidate pool, block selection
//   60-69  60_attn.cuh            KA    sparse attention over selected rows + SWA (decode / verify / prefill)
//   70-79  70_mixers.cuh          KM    mHC mixers (Sinkhorn), norms, router, Engram gate
//   80-89  80_engram.cuh          EIO   Engram row decode (FP8 records -> bf16), row cache helpers
//   90-99  90_dspark.cuh          DS    DSpark round glue (Markov head, confidence, block verify)
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif

#include "gpu_dsv41/00_abi.cuh"
#include "gpu_dsv41/01_abi_probe.cuh"
#if __has_include("gpu_dsv41/05_tp.cuh")
#include "gpu_dsv41/05_tp.cuh"
#endif

#if __has_include("gpu_dsv41/02_fwd.cuh")
#include "gpu_dsv41/02_fwd.cuh"
#endif

#if __has_include("gpu_dsv41/10_trellis.cuh")
#include "gpu_dsv41/10_trellis.cuh"
#endif
#if __has_include("gpu_dsv41/20_fp8_dense.cuh")
#include "gpu_dsv41/20_fp8_dense.cuh"
#endif
#if __has_include("gpu_dsv41/30_mxfp4_moe.cuh")
#include "gpu_dsv41/30_mxfp4_moe.cuh"
#endif
#if __has_include("gpu_dsv41/40_kv.cuh")
#include "gpu_dsv41/40_kv.cuh"
#endif
#if __has_include("gpu_dsv41/50_indexer.cuh")
#include "gpu_dsv41/50_indexer.cuh"
#endif
#if __has_include("gpu_dsv41/60_attn.cuh")
#include "gpu_dsv41/60_attn.cuh"
#endif
#if __has_include("gpu_dsv41/70_mixers.cuh")
#include "gpu_dsv41/70_mixers.cuh"
#endif
#if __has_include("gpu_dsv41/80_engram.cuh")
#include "gpu_dsv41/80_engram.cuh"
#endif
#if __has_include("gpu_dsv41/90_dspark.cuh")
#include "gpu_dsv41/90_dspark.cuh"
#endif
