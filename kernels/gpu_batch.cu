// Batched (continuous-batching) kernels for Qwen3.5-0.8B decode.
// ALL activations are bf16 (__nv_bfloat16) — eliminates f32↔bf16 conversion kernels.
// Internal computation is f32. State arrays (KV cache, conv, recurrent) stay f32.
// Activations are column-major [feat, B] (B sequences = B columns; seq j at offset j*feat).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "tp_doorbell.h"

// ---- Build-ID stamp: makes a stale PTX impossible to run silently ----
// build.rs hashes the .cu bytes and passes the result as -DKERNEL_BUILD_ID. GpuModel::load reads this
// global back out of the loaded module and asserts it equals the ID compiled into the BINARY. A fresh
// binary next to old kernels then fails loudly at startup instead of launching a kernel whose ABI it no
// longer agrees with -- which is how we once got CUDA_ERROR_ILLEGAL_ADDRESS out of correct code.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }


__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// Attention head-dim envelope: the decode/prefill attention kernels take hd as a runtime argument and
// size their per-lane register slices for hd/32 <= SK_DPL_MAX, i.e. any hd that is a positive multiple
// of 32 up to SK_HD_MAX (qwen3.5: 256; hy_v3: 128; DeepSeek: 512). A bigger hd needs only these raised.
#define SK_HD_MAX 512
#define SK_DPL_MAX (SK_HD_MAX / 32)

#define GRID1(total) ((int)(((total) + 255) / 256))

#include "gpu_batch/01_norms_tp_transport.cuh"
#include "gpu_batch/02_tp_prefill_conv_delta.cuh"
#include "gpu_batch/03_gdn_prefill_kv_q4.cuh"
#include "gpu_batch/04_kv_k8v4_tq_misc.cuh"
#include "gpu_batch/05_prefill_attn_splitk.cuh"
#include "gpu_batch/06_splitk_k8v4_k8v8.cuh"
#include "gpu_batch/07_splitk_tq_gq.cuh"
#include "gpu_batch/08_attn_reduce_sample.cuh"
#include "gpu_batch/09_gemm_mma_fp4.cuh"
#include "gpu_batch/10_gemm_dsv4_moe_ref.cuh"
#include "gpu_batch/11_moe_fp4_warp_tc.cuh"
#include "gpu_batch/12_moe_fold_descriptor.cuh"
#include "gpu_batch/13_gdn_rollback_fp8_ple.cuh"
#include "gpu_batch/14_qsa_indexer.cuh"
#include "gpu_batch/15_gptq.cuh"
