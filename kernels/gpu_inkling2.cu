// =============================================================================
// kernels/gpu_inkling2.cu — the Inkling-2 (looped, `inkling2_loop`) family kernel module `gpu_inkling2` (sm_121a).
// Own tree: nothing here is shared with another family (kernels other families use are COPIED, never edited;
// PLAN/INKLING2_OWNER_NOTES_2026-11-22.md). Loaded by src/inkling2/gpu/ only (ONE module load, every band's
// entry list in src/inkling2/gpu/mod.rs FNS). Plan: PLAN/INKLING2_MINI_PORT.md (+ the S01 report's D-list,
// which is authoritative where the plan differs); the exact reference is src/inkling2/cpu.rs.
//
// SECTION BANDS (one translation unit; every *.cuh in kernels/gpu_inkling2/ is part of KERNEL_BUILD_ID, see
// build.rs). A leg owns a band of file numbers:
//   01-09  K1  common helpers (ika_*), the fused BF16 projection GEMM (decode + prefill)
//   10-19  K1  attention (prep, split-KV decode / tiled prefill with relative bias + sinks + gate; combine)
//   20-29  K1  static short conv + residual, conv-ring store
//   30-39  K1  AttnRes (block attention residual, 1..4 queries)
//   40-49  K1  MessageGate (rms-cat, forget / gain epilogue)
//   50-59  K1  norms (weighted / unweighted RMSNorm), embed + embed_norm
//   60-79  K2  MoE (router, NVFP4 / MXFP8 experts, softcapped SwiGLU, combine), LM head + argmax (66: S7 verify widths)
//   80-84  S3  model glue (residual adds, sampler, misc)
//   85-99  S4  lane-batched decode entries (device-side positions: the decode step as one CUDA graph)
//   a0-af  S6  the bundled drafter (mtp.safetensors) + exact speculative accept (kernels ink6_*)
//   b0-bf  S5  vision tower (inkv_*: patches, embed + ln_pre, LN, RoPE, tensor-core attention, merger glue)
//   d0-df  TP2 tensor-parallel copies (attention per rank heads, generated: scripts/inkling2/gen_tp_attn.py; vocab-parallel sampler)
// Names: K1 kernels `ink_*` (helpers `ika_*`), K2 kernels `ink2_*`, S3 kernels `ink3_*`.
// Every K2 entry opens with INK2_PDL_ENTRY() so a PDL-launched neighbour may overlap it.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdint.h>

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void ink2_kernel_build_id(unsigned long long* out) { asm volatile("griddepcontrol.wait;" ::: "memory"); *out = KERNEL_BUILD_ID; }

#define INK2_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                              asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)
// S7: weight-streaming kernels start their weight fetch BEFORE waiting on the predecessor (PDL lets the CTAs start
// while the previous kernel's small ops run): one thread per CTA asks L2 for the first `bytes` of the CTA's own
// contiguous weight slice (weights never depend on the predecessor). Then the usual entry (trigger + wait).
__device__ __forceinline__ void ink2_l2_prefetch(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
#define INK2_PDL_ENTRY_PREFETCH(ptr, bytes) do { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); \
        if (threadIdx.x == 0 && (bytes) > 0) ink2_l2_prefetch((ptr), (bytes));                                   \
        asm volatile("griddepcontrol.wait;" ::: "memory"); } while (0)
#ifndef INK2_PF_BYTES
#define INK2_PF_BYTES 32768u   // per-CTA prefetch (x ~150-400 resident CTAs: 5-12 MB, well inside L2)
#endif

#include "gpu_inkling2/01_k1_common.cuh"
#include "gpu_inkling2/05_proj_gemm.cuh"
#include "gpu_inkling2/10_attn.cuh"
#include "gpu_inkling2/11_attn_tc.cuh"
#include "gpu_inkling2/06_proj_tile.cuh"   // after 11: uses ika_cp16
#include "gpu_inkling2/20_conv.cuh"
#include "gpu_inkling2/30_attnres.cuh"
#include "gpu_inkling2/40_msggate.cuh"
#include "gpu_inkling2/50_norm.cuh"
#include "gpu_inkling2/60_moe_common.cuh"
#include "gpu_inkling2/61_router.cuh"
#include "gpu_inkling2/62_experts_decode.cuh"
#include "gpu_inkling2/63_experts_prefill.cuh"
#include "gpu_inkling2/64_combine.cuh"
#include "gpu_inkling2/65_lm_head.cuh"
#include "gpu_inkling2/66_experts_verify.cuh"
#include "gpu_inkling2/85_lanes.cuh"
#include "gpu_inkling2/b0_vit.cuh"      // S5 vision tower (band b0-bf)
#include "gpu_inkling2/c0_actq_common.cuh"
#include "gpu_inkling2/c1_actq_quant.cuh"
#include "gpu_inkling2/c2_actq_decode.cuh"
#include "gpu_inkling2/c3_actq_prefill.cuh"
#if __has_include("gpu_inkling2/80_glue.cuh")
#include "gpu_inkling2/80_glue.cuh"
#endif
#include "gpu_inkling2/a0_mtp.cuh"
// S-INK2-TP2: tensor-parallel copies (band d0-df): attention at 24 / W heads per rank, the vocab-parallel sampler
#include "gpu_inkling2/d0_tp_attn.cuh"
#include "gpu_inkling2/d8_tp_glue.cuh"
