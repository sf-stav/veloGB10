// =============================================================================
// kernels/gpu_dsv41/30_mxfp4_moe.cuh — band 30 HEAD (leg KN2): DeepSeek-V4.1 native routed experts,
// MXFP4 weights (E2M1 x2 + one ue8m0 per 32 inputs of a row) with FP8 E4M3 activations (ue8m0 per
// 32), the reference's `linear()` -> act_quant + fp4_gemm (REF_MAP §2.2, §2.5, §3.6, §4 items 12-14).
//
// ARITHMETIC CONTRACT (bit-exact to the reference's fp4_gemm, every row class):
//   for kb in 0 .. K/32 (ascending):
//       d   = RNE_f32( dot32(W[row, kb], X[tok, kb]) * 2^(sa + sb - 254) )
//       acc = RNE_f32( acc + d )
//   out = bf16(acc)
// `d` comes from ONE block-scaled tensor-core MMA with a ZERO accumulator per 32-wide k-block
// (mma.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X, e2m1 x e4m3, ue8m0); the k-block sum is
// an ordinary FP32 add on the CUDA cores, in ascending k order. Measured on GB10 (KN2 probe,
// 2026-11-18): this equals the sequential FP32 model of TileLang's fp4_gemm in 0 of 1.2M outputs
// over three activation distributions (incl. 7 decades inside a block); letting the tensor core
// accumulate ACROSS k-blocks (C = previous D) differs in 64-99 % of outputs, max rel error 7e-3 —
// the tensor core's accumulator is not an FP32 RNE adder. Never accumulate across k-blocks in the
// MMA. Because every class computes each output with the same sequence, every row class (GEMV /
// cooperative / wide) gives bit-identical results: batch invariance holds ACROSS classes.
//
// FRAGMENTS (m16n8k32, empirically verified on GB10 by the KN2 probe): lane l, g = l >> 2, t = l & 3.
//   A (weights, 16 rows x 32 k, e2m1 in 8-bit containers, code << 2):
//     a0: row g,   k 4t..4t+3     a1: row g+8, k 4t..4t+3
//     a2: row g,   k 16+4t..+3    a3: row g+8, k 16+4t..+3      (byte i = k + i)
//   B (activations, 32 k x 8 tokens, e4m3): b0: token g, k 4t..4t+3; b1: token g, k 16+4t..+3
//   C/D: d0: (row g, tok 2t) d1: (row g, tok 2t+1) d2: (row g+8, tok 2t) d3: (row g+8, tok 2t+1)
//   SFA (selector {byte v, thread 0}): lane (g, t=0) byte v = row g, lane (g, t=1) byte v = row g+8
//   SFB (selector {byte v, thread 0}): lane (g, t=0) byte v = token g
//
// PACKED WEIGHT FORMAT "mx1" (src/dsv41/weights/native/mxfp4_pack.rs builds it; one buffer per
// layer and matrix pair, experts at a uniform stride):
//   m-tile block = codes [nkb][32 lanes][8 B] then scales [nkq = ceil(nkb/4)][16 u32]
//     codes: lane (g,t) u64, nibble n = 4r + i is A reg r byte i (layout above)
//     scales: u32 index 2g + h (h = 0 / 1 -> row g / g+8), byte v = scale of that row at kb 4q + v
//   FC1 (w13): m-tile j = [w1 rows i0+8j .. +7 ; w3 rows i0+8j .. +7] (gate in rows 0-7, up in 8-15,
//     so SwiGLU pairs live in one lane), K = dim. FC2 (w2): m-tile j = w2 rows 16j .. 16j+15,
//     K = this rank's inner slice. Inner slices are multiples of 32 (act_quant groups never split).
//
// ACTIVATION FORMAT: e4m3 [rows][ld] row-major (the reference's act_quant output) + ue8m0
// [rows][lds], lds a multiple of 4 (the MMA reads 4 k-blocks' scales as one u32).
//
// Files: 30 (this: helpers, act_quant, route plan), 31 grouped decode / verify kernels (<= 32 rows),
// 32 wide (prefill) kernels.
// =============================================================================
#pragma once

#define D41_MX_CODES_UNIT 256      // bytes per (m-tile, k-block)
#define D41_MX_SCALES_UNIT 64      // bytes per (m-tile, k-block quad)
#define D41_MX_GROWS 32            // rows served by the grouped kernels (31)
#define D41_MX_MAXR 192            // routes in the grouped kernels: 32 rows x top-6

// ---- small helpers --------------------------------------------------------------------------
__device__ __forceinline__ float d41_mx_bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

// 4 packed e2m1 nibbles (low / high 16 bits of v) -> 4 bytes, each code << 2 (f8f6f4 container)
__device__ __forceinline__ uint32_t d41_mx_lo4(uint32_t v) {
    uint32_t x = __byte_perm(v, 0u, 0x1100);
    return ((x & 0x000F000Fu) << 2) | ((x >> 2) & 0x3C003C00u);
}
__device__ __forceinline__ uint32_t d41_mx_hi4(uint32_t v) {
    uint32_t x = __byte_perm(v, 0u, 0x3322);
    return ((x & 0x000F000Fu) << 2) | ((x >> 2) & 0x3C003C00u);
}

// one block-scaled MMA with a ZERO accumulator: d = RNE(dot32 * 2^(sa+sb-254)), byte BID of the scale regs
template <int BID>
__device__ __forceinline__ void d41_mx_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, %11, {%12,0}, %13, {%12,0};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f), "r"(sfa), "n"(BID), "r"(sfb));
}
__device__ __forceinline__ void d41_mx_mma_v(int v, float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    switch (v) {   // v is a compile-time constant after unrolling: the switch folds away
    case 0: d41_mx_mma<0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    case 1: d41_mx_mma<1>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    case 2: d41_mx_mma<2>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    default: d41_mx_mma<3>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    }
}

// streaming loads: weights are read once per call (no L1 allocation); activations are shared
__device__ __forceinline__ uint2 d41_mx_ldw(const uint8_t* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t d41_mx_ldw32(const uint8_t* p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t d41_mx_lda(const uint8_t* p) { return __ldg((const unsigned int*)p); }

// act_quant scale (kernel.py fast_round_scale): smallest power of two >= amax * fp32(1/448), by the
// exponent bits of the fp32 product (amax already floored at 1e-4, so the product is normal).
__device__ __forceinline__ float d41_mx_pow2_scale(float amax) {
    const float v = amax * (1.0f / 448.0f);
    const uint32_t b = __float_as_uint(v);
    const int e = (int)((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}
__device__ __forceinline__ uint8_t d41_mx_e4m3(float v, float sc) {
    const float q = fminf(fmaxf(__fdiv_rn(v, sc), -448.0f), 448.0f);
    return (uint8_t)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ uint8_t d41_mx_e8m0_of(float pow2) { return (uint8_t)((__float_as_uint(pow2) >> 23) & 0xFFu); }

// torch semantics: clamp keeps NaN (compare-and-select, not fminf / fmaxf)
__device__ __forceinline__ float d41_mx_min_keepnan(float x, float hi) { return x > hi ? hi : x; }
__device__ __forceinline__ float d41_mx_clamp_keepnan(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// the reference Expert's SwiGLU on fp4_gemm's bf16 outputs (model.py 845-853): clamps, silu as torch's
// CUDA kernel computes it (x / (1 + exp(-x)), precise expf, IEEE divide), then the routing weight,
// then the bf16 cast that feeds w2's act_quant.
__device__ __forceinline__ float d41_mx_swiglu(float gate_bf, float up_bf, float w, float limit) {
    const float u = d41_mx_clamp_keepnan(up_bf, -limit, limit);
    const float g = d41_mx_min_keepnan(gate_bf, limit);
    const float s = __fdiv_rn(g, 1.0f + expf(-g));
    return d41_mx_bf16r(w * (s * u));
}

// ---- act_quant (kernel.py act_quant, non-inplace, block 32, ue8m0) ----------------------------
// x bf16 [rows][ldx] -> q e4m3 [rows][ldq], s ue8m0 [rows][lds]. One warp per (row, 32-group).
extern "C" __global__ void __launch_bounds__(256)
d41_mx_act_quant(const __nv_bfloat16* __restrict__ x, int ldx, int rows, int K,
                 uint8_t* __restrict__ q, int ldq, uint8_t* __restrict__ s, int lds) {
    D41_PDL_ENTRY();
    const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    const int gpr = K >> 5;
    if (gw >= rows * gpr) return;
    const int row = gw / gpr, grp = gw - row * gpr;
    const float v = __bfloat162float(x[(size_t)row * ldx + grp * 32 + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    amax = fmaxf(amax, 1e-4f);
    const float sc = d41_mx_pow2_scale(amax);
    q[(size_t)row * ldq + grp * 32 + lane] = d41_mx_e4m3(v, sc);
    if (lane == 0) s[(size_t)row * lds + grp] = d41_mx_e8m0_of(sc);
}

// PK2: the same act_quant writing each 32-byte k-block in the WIDE kernels' permuted order: byte k at
// d41_mx_kperm(k) = ((k & 15) >> 2) * 8 + (k >> 4) * 4 + (k & 3), so lane t's two B registers (k 4t..4t+3
// and 16+4t..16+4t+3 of m16n8k32) are one 8-byte load at 8t. Values and scales are identical.
__device__ __forceinline__ int d41_mx_kperm(int k) { return ((k & 15) >> 2) * 8 + (k >> 4) * 4 + (k & 3); }
extern "C" __global__ void __launch_bounds__(256)
d41_mx_act_quant_p(const __nv_bfloat16* __restrict__ x, int ldx, int rows, int K,
                   uint8_t* __restrict__ q, int ldq, uint8_t* __restrict__ s, int lds) {
    D41_PDL_ENTRY();
    const int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    const int gpr = K >> 5;
    if (gw >= rows * gpr) return;
    const int row = gw / gpr, grp = gw - row * gpr;
    const float v = __bfloat162float(x[(size_t)row * ldx + grp * 32 + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    amax = fmaxf(amax, 1e-4f);
    const float sc = d41_mx_pow2_scale(amax);
    q[(size_t)row * ldq + grp * 32 + d41_mx_kperm(lane)] = d41_mx_e4m3(v, sc);
    if (lane == 0) s[(size_t)row * lds + grp] = d41_mx_e8m0_of(sc);
}

// ---- route plan (grouped kernels): distinct LOCAL experts in ascending id order, and each
// expert's routes in ascending route order. O(nr^2) over <= 192 routes in shared memory. ----------
struct D41MxPlan {
    int n_dist;
    short e_of[D41_MX_MAXR];     // route -> local expert (-1: another rank's expert)
    short dist_e[D41_MX_MAXR];   // distinct local experts, ascending
    short start[D41_MX_MAXR];    // first index in `sorted` of each distinct expert
    short cnt[D41_MX_MAXR];      // routes of each distinct expert
    short sorted[D41_MX_MAXR];   // routes sorted by (expert, route)
    unsigned char first[D41_MX_MAXR];
};

__device__ __forceinline__ void d41_mx_build_plan(D41MxPlan& P, const int* __restrict__ ids, int nr, int e_first, int n_local) {
    const int tid = threadIdx.x, nth = blockDim.x;
    if (tid == 0) P.n_dist = 0;
    for (int r = tid; r < nr; r += nth) {
        const int e = ids[r] - e_first;
        P.e_of[r] = (e >= 0 && e < n_local) ? (short)e : (short)-1;
    }
    __syncthreads();
    for (int r = tid; r < nr; r += nth) {
        const int e = P.e_of[r];
        if (e < 0) { P.first[r] = 0; continue; }
        int lt = 0, eqb = 0;
        for (int s = 0; s < nr; s++) {
            const int f = P.e_of[s];
            lt += (f >= 0 && f < e);
            eqb += (s < r && f == e);
        }
        P.sorted[lt + eqb] = (short)r;
        P.first[r] = (eqb == 0);
    }
    __syncthreads();
    for (int r = tid; r < nr; r += nth) {
        if (!P.first[r]) continue;
        const int e = P.e_of[r];
        int dr = 0, lt = 0, cnt = 0;
        for (int s = 0; s < nr; s++) {
            const int f = P.e_of[s];
            dr += (P.first[s] && f < e);
            lt += (f >= 0 && f < e);
            cnt += (f == e);
        }
        P.dist_e[dr] = (short)e; P.start[dr] = (short)lt; P.cnt[dr] = (short)cnt;
        atomicAdd(&P.n_dist, 1);
    }
    __syncthreads();
}

#include "31_mxfp4_decode.cuh"
#if __has_include("32_mxfp4_wide.cuh")
#include "32_mxfp4_wide.cuh"
#endif
