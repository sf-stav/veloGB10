// =============================================================================
// kernels/exl3_bench.cu — S-A3-c: EXL3 weight-path GEMM
// (trellis decode fused into fp16 HMMA), probe/bench-only module.
//
// Arch: sm_121 (fp16 mma.sync.m16n8k16 and dp4a are baseline CC 12.1 features —
// no family/architecture suffix needed; see gb10-cuda skill target rules).
// Loaded lazily ONLY by --probe-exl3-kernel / --probe-exl3-binv / --bench-exl3-gemm
// (precedent: mxfp4_bench). NOT in the serving manifest; no serving-path change.
//
// Decode math mirrors src/exl3.rs `decode_block` + `tmap` bit-exactly (that decode
// is G-A3-1-green vs ext.reconstruct over 21.3M blocks, S-A3-b):
//   ring  = the block's 16*bits int16 viewed as 8*bits LE uint32
//   t     = temporal index; window END b1 = (t+257)*bits, START b0 = b1-16
//   idx   = ((ring[i1%W] >> s) | (ring[i0%W] << (32-s))) & 0xFFFF, s = ((i1+1)*32)-b1
//           (s==0 -> ring[i1%W] & 0xFFFF; __funnelshift_r reproduces both branches)
//   w     = hfma(cvt(sum), 0x1EEE, 0xC931), sum = dp4a_u32(idx*0x83DCD12D, 0x01010101, 0x400)
//           sum = 1024 + UNSIGNED bytesum (matches the Rust LUT; fused fp16 FMA path
//           verified equal to the double-rounded f32 LUT on all 65536 entries)
//   place W[16kb+r][16nb+c] = decoded[tmap(r,c)],
//   tmap(r,c) = 8*l + (r%2) + 2*((r/8)%2) + 4*(c/8) + 32*(c%2), l = 8*((c/2)%4) + ((r%8)/2)
//
// GEMM: y_raw[m,n] = sum_k xh[m,k] * W[n,k]  (W = decoded trellis, fp32 accum)
//   CTA: 256 threads = 8 warps, tile m16 x N128; warp w owns columns [n0+16w, n0+16w+16).
//   Per k16 step each warp decodes ONE 16x16 trellis block DIRECTLY into two
//   m16n8k16 B-fragments (n-tiles h=0,1) — no weight SMEM round-trip.
//   B-fragment slot (lane l, tile h, slot s in 0..3):
//     r = 2*(l%4) + (s&1) + 8*(s>>1),  c = 8*h + (l>>2),  t = tmap(r,c).
//   A operand: xh staged in SMEM per k16 step (rows >= M zero-filled), manual
//   fragment build. Accumulators: fp32, K ascending, fixed order (batch invariance:
//   M only pads the m16 tile with zeros; the per-element k-reduction order is fixed
//   by the mma instruction and the K loop — AGENTS 2.4 / G-A3-5).
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_fp8.h>
#include "tp_doorbell.h"   // TP-G: the doorbell ABI (tp_dev_ctx, flag offsets) for the folded decode all-reduce

#define EXL3_MUL1 0x83DCD12Du

// ---------------------------------------------------------------------------
// Decode with precomputed (i0m, i1m, sf) — the hoisted-constant form used by the
// GEMM body. THE one decode device function (AGENTS 2.8: every consumer shares it).
// Returns the fp16 bit pattern of the decoded weight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t exl3_dq_from(const uint32_t* __restrict__ ring,
                                                 int i0m, int i1m, int sf) {
    // Oracle placement: the LATER word (i1) is LOW in the merged pair, the EARLIER
    // word (i0) HIGH — idx = ((ring[i1] >> s) | (ring[i0] << (32-s))) & 0xFFFF,
    // s == 0 -> ring[i1] & 0xFFFF. __funnelshift_r(lo, hi, s) = ((hi:lo) >> s) low 32,
    // so lo = ring[i1m] and hi = ring[i0m] (swapped args are invisible when i0 == i1,
    // i.e. exactly the ~50% of b=3 windows that don't cross a word boundary — the
    // second gate run's 47% mismatch fingerprint).
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    // UNSIGNED dp4a overload: value = 1024 + unsigned byte-sum s (must match Rust LUT).
    // S-A3-n (SASS diff vs the reference implementation's exl3_moe_coop_a: ZERO I2F there, one
    // I2F.F16.U32 per weight here — a reduced-rate conversion on the decode's
    // critical path): accumulate 0x6400 instead of 1024. 0x6400 is the fp16 bit
    // pattern of 1024.0 and s <= 4*255 = 1020 < 1024, so 0x6400 + s IS the fp16
    // encoding of 1024 + s (exponent 25, mantissa s; ulp = 1 on [1024, 2048)) —
    // exactly what cvt.rn.f16.u32(1024 + s) returned. Bit-identical, I2F gone.
    // (The spec §3.2 "0x6400" was this trick, not a typo.)
    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum); // fp16 bits of 1024 + s
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE),   // k_inv
                                  __ushort_as_half((unsigned short)0xC931)); // k_bias
    return (uint16_t)__half_as_ushort(w16);               // single-rounded fp16 FMA
}

// Decode by temporal index t (computes the window indices from t).
__device__ __forceinline__ uint16_t exl3_dq_t(const uint32_t* __restrict__ ring,
                                              int t, int bits) {
    const int b1 = (t + 257) * bits;      // window END (unwrapped bit)
    const int b0 = b1 - 16;               // window START
    const int i0 = b0 >> 5;
    const int i1 = (b1 - 1) >> 5;
    const int sf = ((i1 + 1) << 5) - b1;  // 0..31
    const unsigned w = (unsigned)(bits * 8);
    return exl3_dq_from(ring, (int)(i0 % w), (int)(i1 % w), sf);
}

// W1 (docs/EXL3_FRAC_K_SPEC.md): fractional K = KA + 0.5 window math. Weight pairs
// consume bits2 = 2*KA+1 bits (even-weight window KA bits, odd-weight KA+1); within the
// 8-window group the top window w7 ENDS at e7 = ((g>>1)+4)*bits2 + 128*bits2 (g = t & ~7),
// and weight t (j = t & 7) has its 16-bit window END at e7 - [(7-j)*KA + ((14-j)>>1) - 3].
// The tile ring is 4*bits2 uint32 words (= 8*k2 uint16 = the trellis last dim).
template <int KA>
__device__ __forceinline__ int exl3_half_b1(int t) {
    const int g = t & ~7, j = t & 7;
    const int bits2 = 2 * KA + 1;
    return ((g >> 1) + 4) * bits2 + 128 * bits2 - (((7 - j) * KA) + ((14 - j) >> 1) - 3);
}

// Runtime-ka form (decode_dump path); the templated bodies use exl3_half_b1<KA>.
__device__ __forceinline__ uint16_t exl3_dq_t_half(const uint32_t* __restrict__ ring,
                                                   int t, int ka) {
    const int bits2 = 2 * ka + 1;
    const int g = t & ~7, j = t & 7;
    const int e7 = ((g >> 1) + 4) * bits2 + 128 * bits2;
    const int b1 = e7 - (((7 - j) * ka) + ((14 - j) >> 1) - 3);
    const int b0 = b1 - 16;
    const int i0 = b0 >> 5;
    const int i1 = (b1 - 1) >> 5;
    const int sf = ((i1 + 1) << 5) - b1;
    const unsigned w = (unsigned)(4 * bits2);
    return exl3_dq_from(ring, (int)(i0 % w), (int)(i1 % w), sf);
}

// k2-dispatching decode (k2 = 2*K + half): the single entry the runtime-arg kernels use.
__device__ __forceinline__ uint16_t exl3_dq_k2(const uint32_t* __restrict__ ring, int t, int k2) {
    return (k2 & 1) ? exl3_dq_t_half(ring, t, k2 >> 1) : exl3_dq_t(ring, t, k2 >> 1);
}

// K35 (2026-10-31): the WIDTH CODE of the `bits` argument. The expert / wide-tile kernels switch on `bits`:
// an integer K arrives as K (2..=8); a fractional K = KA + 0.5 (k2 odd) arrives as EXL3_FRAC_BITS(KA) =
// 16 + KA (17 / 18 / 19), so a fractional width can never alias an integer one — k2 = 7 passed as k2/2 = 3
// would take the 3-bit bodies on a 56-word ring (the silent-garbage class). The fractional bodies are the
// HALF twins of exl3_gemm_body / exl3_gemm_body_a1 / exl3_gemm_body_wide (window math exl3_half_b1<KA>, ring
// 4*(2*KA+1) words per 16x16 tile). Host twin: src/exl3_forward.rs exl3_kernel_bits().
#define EXL3_FRAC_BITS(KA) (16 + (KA))

// ---------------------------------------------------------------------------
// Kernel 1: decode gate (G-A3-1 through the kernel path).
// One warp per 16x16 trellis block; thread lane decodes t in {lane + 32*step}.
// rc_of_t[256]: inverse tmap LUT (host-built from the unit-tested Rust bijection):
// rc_of_t[t] = r*16 + c. Output: W row-major [kb*16, nb*16] fp16 bit patterns.
// ---------------------------------------------------------------------------
// PDL (A5 D14, 2026-09-26): every entry kernel starts by waiting for its programmatic predecessor
// and releasing its own dependents. On a plain launch both instructions are no-ops; with
// GB10_EXL3_PDL=1 the launcher sets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION so the
// next grid becomes resident while this one's last wave drains (launch gap + tail overlap).
// The dependent release fires only after ALL CTAs of a grid have started, so co-resident
// grid-barrier kernels (xq_hc_fuse*) can never be starved by a waiting successor.
// P7 B7: the K=6 arms below are always compiled in — ONE module serves every width;
// the exl3_bench_k6.ptx per-process swap (4.05-bpw packs) is gone.
#define XQ_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

// TUNE T0 (PLAN/AUTOTUNE_DESIGN.md finding §14.1): the stale-kernel handshake. build.rs compiles
// this module with -DKERNEL_BUILD_ID (the hash of every .cu it builds); FwdModel::load launches this
// entry from the LOADED module and refuses a mismatch — the served EXL3 path used to run an old
// exl3_bench.ptx silently whenever the kernel names still resolved.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

#include "exl3_bench/01_tune_lmh_gemm.cuh"
#include "exl3_bench/02_wide_grouped_forward.cuh"
#include "exl3_bench/03_gdn_router_dhead.cuh"
#include "exl3_bench/04_sampler_penalties.cuh"
#include "exl3_bench/05_draft_head_pen.cuh"
#include "exl3_bench/06_router_fold_r16.cuh"
#include "exl3_bench/07_moe_wp20_epilogues.cuh"
#include "exl3_bench/08_moe_persistent.cuh"
#include "exl3_bench/09_moe_pk_widths_ple.cuh"
#include "exl3_bench/10_prefill_chunk.cuh"
#include "exl3_bench/11_prefill_attention.cuh"
#include "exl3_bench/12_prefill_moe_compact.cuh"
#include "exl3_bench/13_prefill_moe_pf2.cuh"
#include "exl3_bench/14_hc_mixer.cuh"
#include "exl3_bench/15_splitk_chain.cuh"
#include "exl3_bench/16_dense_dks_verify.cuh"
#include "exl3_bench/17_attn_dense_v3.cuh"
#include "exl3_bench/18_draft_slice_qsa.cuh"
#include "exl3_bench/19_qsa_xcheck_sel.cuh"
#include "exl3_bench/20_sel_tc_posture1.cuh"
#include "exl3_bench/21_small_levers.cuh"
#include "exl3_bench/22_dsplitk_ep.cuh"
#include "exl3_bench/23_tp_allreduce.cuh"
