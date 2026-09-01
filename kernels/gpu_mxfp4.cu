// gpu_mxfp4.cu — serving kernels for the MXFP4-NATIVE mode (QWEN_MXFP4_NATIVE_DESIGN.md).
//
// The mode runs the Qwen models' fp4 GEMMs on Blackwell's native block-scaled FP4 path —
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .f32.e2m1.e2m1.f32.ue4m3      (SASS: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X)
// — instead of the dequantize-to-bf16 HMMA chain. STORAGE STAYS NVFP4: the only artifact-side
// change is the lossless load-time repack (src/mxfp4.rs) into the fragment/scale layout the
// instruction expects. All fragment/scale layouts below were EMPIRICALLY VERIFIED on GB10
// (probe 1899/1899, kernels/mxfp4_bench.cu header comment, 2026-08-06).
//
//   A fragment (lane g = lane>>2, t = lane&3; reg a_r, nibble j = LSB-first):
//     a0: row g     k 8t..8t+7      a1: row g+8   k 8t..8t+7
//     a2: row g     k 8t+32..8t+39  a3: row g+8   k 8t+32..8t+39
//   B fragment (b_r, nibble j): token n = g, k = 8t + j + 32*r   (ONE token row per lane)
//   C fragment (reg i): row g + 8*(i>=2), col 2t + (i&1)
//   SFA (lane (g,t), byte v): row g + 8*(t&1), kblock v  -- lanes t in {2,3} ignored
//   SFB (lane (g,t), byte v): token g, kblock v          -- only lanes t == 0 read
//   kblock v covers k in [16v, 16v+16) of the instruction's 64-wide K.
//
// Repacked weight layouts (per 16-row tile mt, 64-K kstep ks):
//   Aimg[(mt*nks+ks)*128 + lane*4 + r]   u32, 512 B per (tile,kstep), lane-major
//   SFAw[(mt*nks+ks)*16  + g*2 + (t&1)]  u32, 64 B per (tile,kstep), t<=1 valid
//   gs[mt]                                f32 per 16-row tile (applied once in the epilogue)
// Activation layouts (8-token group per kstep):
//   Bp[((q*nks+ks)*32 + lane)*2 + r]     u32, r=0 -> b0 (k 8t..8t+7), r=1 -> b1 (k+32)
//   SFB[(q*nks+ks)*32 + lane]            u32, byte v = scale(token g, kblock ks*4+v), t==0 only
//
// Batch-invariance within the mode (design §5): N padded to 8 (the OMMA's N); the k-visit
// order and the cross-warp reduction are FIXED and N-independent; column 0 of an N-wide
// verify is bit-identical to a decode BY CONSTRUCTION (padded token rows are zero, their
// scales zero, and the store guard only affects columns >= N).
//
// COMPILE: this file uses sm_121a family-specific features (the mma and cvt.e2m1x2 reject
// plain sm_121). build.rs compiles it with --gpu-architecture=sm_121a; the serving manifest
// stays sm_121.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstdio>

extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

typedef __nv_bfloat16 bf16;

// e2m1 nibble -> float (the 16 codes). Mirrors gpu_batch.cu's e2m1_f.
__device__ __forceinline__ float e2m1_f(uint8_t c) {
    const float v[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                         -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return v[c & 15];
}

#define MMA_NW 8                       // warps per block (split the kstep chain, fixed order)
#define MMA_SMEM (8 * 256)             // [8 acc slots (2 groups x 4)][8 warps][32 lanes] f32

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

// Two f32 -> one byte of two e2m1 nibbles (low nibble = src[0]), round-to-nearest-even with
// satfinite, the hardware instruction (sm_121a).
__device__ __forceinline__ unsigned char cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n}"
        : "=r"(tmp) : "f"(lo), "f"(hi));
    return (unsigned char)(tmp & 0xff);
}

// ue4m3 encode of |x| rounded UP (sign bit 0 — the OMMA ignores it). Mirrors the quantizer.
__device__ __host__ __forceinline__ unsigned char e4m3_ceil(float x) {
    // E4M3 has NO infinity and 0x7F is NaN: the largest finite code is 0x7E = 448. Saturate there
    // (the codes then clip through cvt.satfinite) — 0x7F would poison the whole MMA output.
    if (!(x > 0.f)) return 0x00;
    if (x >= 448.0f) return 0x7E;
    int e;
    float m = frexpf(x, &e);
    int e4 = e + 6;
    int mant = (int)ceilf((m - 0.5f) * 16.0f);
    if (mant >= 8) { mant = 0; e4++; }
    if (e4 < 0) {
        int sm = (int)ceilf(x * 512.0f);
        return (unsigned char)(sm > 7 ? 7 : sm);
    }
    if (e4 > 15 || (e4 == 15 && mant >= 7)) return 0x7E;
    return (unsigned char)((e4 << 3) | mant);
}

__device__ __host__ __forceinline__ float ue4m3_f(unsigned char s) {
    int e = (s >> 3) & 0xF, m = s & 7;
    if (e == 0) return (float)m * 0.001953125f;
    return (1.0f + m / 8.0f) * exp2f((float)e - 7);
}

// ---------------------------------------------------------------------------
// Activation quant + pack: bf16 X [N][K] -> Bp/SFB (8-token group; token rows >= N are
// zeroed so padded OMMA columns are exactly 0 and never stored). One block per 64-K kstep;
// warp n = token n.  The same pack feeds EVERY tile and warp (activations are shared).
// ---------------------------------------------------------------------------
extern "C" __global__ void mxfp4_quant_pack_b(const bf16* __restrict__ X, int K, int N,
                                              uint32_t* __restrict__ Bp, uint32_t* __restrict__ SFB) {
    const int ks = blockIdx.x;
    const int n = threadIdx.x >> 5;          // token 0..7 (>= N: zero row)
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    __shared__ float sh[8][64];
    if (n < N) {
        const bf16* row = X + (long long)n * K + (long long)ks * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(row[lane * 2 + i]);
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = 0.0f;
    }
    __syncthreads();

    // per-token per-kblock scales (4 kblocks of 16 per token)
    __shared__ float sc[8][4];
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();

    // pack b0/b1: nibble j of b_r = code of X[token g][ks*64 + 8t + j + 32r]
    uint32_t b0 = 0, b1 = 0;
    const float inv0 = sc[g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][t >> 1]);
    const float inv1 = sc[g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][2 + (t >> 1)]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][8 * t + 2 * j] * inv0, sh[g][8 * t + 2 * j + 1] * inv0);
        b0 |= cb << (8 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][32 + 8 * t + 2 * j] * inv1, sh[g][32 + 8 * t + 2 * j + 1] * inv1);
        b1 |= cb << (8 * j);
    }
    Bp[((long long)ks * 32 + lane) * 2 + 0] = b0;
    Bp[((long long)ks * 32 + lane) * 2 + 1] = b1;

    // SFB: u32 per lane; byte v = scale(token g, kblock ks*4 + v). Only t == 0 is read.
    uint32_t sfb = 0;
    if (t == 0) {
        unsigned char b0c = (unsigned char)sc[g][0], b1c = (unsigned char)sc[g][1];
        unsigned char b2c = (unsigned char)sc[g][2], b3c = (unsigned char)sc[g][3];
        sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
    }
    SFB[(long long)ks * 32 + lane] = sfb;
}

// ---------------------------------------------------------------------------
// The serving GEMV: C[col][M] += Wt[16-row tile][K] x B[8-token group][K], OMMA native.
// Persistent-grid port of gemm_mma_fp4_b's schedule (gpu_batch.cu:3219) — the tile->block
// map depends on the weight shape only, never on N; k-visit order and the cross-warp
// reduction are fixed; no atomics. Batch-invariance within the mode: N padded to 8,
// padded token rows/scales are zero, stores guarded by `col < N` (column 0 unaffected).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemv_native_b(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const uint32_t* __restrict__ Bp,
    const uint32_t* __restrict__ SFBw, int ntm, int nks, int M, int N)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int ng = (N + 7) >> 3;               // 8-token OMMA groups (1 or 2)
    __shared__ float sh[MMA_SMEM];

    for (int mt = blockIdx.x; mt < ntm; mt += gridDim.x) {
        __syncthreads();                       // sh reuse barrier (write->read->next write)
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)mt * nks * 128;
        const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)mt * nks * 16;

        // Fixed k-visit order per tile: warp w takes ksteps w, w+8, ... (N-independent).
        for (int ks = warp; ks < nks; ks += MMA_NW) {
            const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
            const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
            const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
            const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
            const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
            for (int q = 0; q < ng; q++) {
                const uint32_t b0 = Bp[(((size_t)q * nks + ks) * 32 + lane) * 2 + 0];
                const uint32_t b1 = Bp[(((size_t)q * nks + ks) * 32 + lane) * 2 + 1];
                const uint32_t sfb = SFBw[((size_t)q * nks + ks) * 32 + lane];
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                    : "+f"(acc[q][0]), "+f"(acc[q][1]), "+f"(acc[q][2]), "+f"(acc[q][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
                      "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
            }
        }

        // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
        for (int q = 0; q < 2; q++)
#pragma unroll
            for (int i = 0; i < 4; i++) sh[(q * 4 + i) * 256 + warp * 32 + lane] = acc[q][i];
        __syncthreads();
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        if (rslot < 4 * ng) {
            const int q = rslot >> 2, i = rslot & 3;
            const int col = q * 8 + 2 * t + (i & 1);
            if (col < N) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
                const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
                C[(long long)col * M + m] = f2b(v * gs[mt]);   // gs applied exactly once
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FUSED activation quant + OMMA GEMV (EXPERT_FUSED_QUANT_RESPONSE.md §4.2, F1) — replaces the
// mxfp4_quant_pack_b + mxfp4_gemv_native_b pair on the fused dispatch path
// (GB10_MXFP4_FUSED, default ON; GB10_MXFP4_FUSED=0 keeps today's separate launches).
// Each warp quantizes, per 64-K kstep, exactly the batch window it is about to feed to the
// OMMA: same elements (bf16 staging is bit-exact via b2f), same ascending fmaxf/fabsf amax
// chains, same amax/6.0f division, same e4m3_ceil, same 1.0f/ue4m3_f inverse, same
// cvt.rn.satfinite.e2m1x2 hardware pack — byte-identical Bp/SFB fragments by construction
// (§3, §7). NR ∈ {1, 8, 16} = real-row capacity (1: N==1 — row 0 staged, dead rows emit
// zero words directly; 8: 2..=N<=8 — 8 rows staged, rows >= N zero-staged; 16: 9..=N<=16 —
// two 8-row passes feeding acc[0]/acc[1]). Rows >= N zero-staged -> amax 0 -> scale byte 0
// -> inv 0 -> +0 codes -> b=0, sfb=0, identical to the standalone kernel's zeroed rows, so
// the padded OMMA columns stay exactly 0 and the `col < N` epilogue guard is unchanged.
// Staging is per-warp (shw[8][64] bf16 + scw[8][4] f32), synchronized with __syncwarp() —
// no cross-warp data flows in the quant path; the epilogue's block-wide reduce barrier is
// unchanged. Grid/block/k-visit order/cross-warp reduction: verbatim from mxfp4_gemv_native_b.
// NOTE (dense N>8 corner): the production quant_pack_b writes only Bp group 0, so today's
// chain's columns 8..15 at N in 9..16 read UNINITIALIZED group-1 scratch; they are never
// consumed at any production width (verify reads column 0). The fused NR=16 kernel computes
// the two real 8-row groups the instruction semantics require (deterministic; the probe
// compares against a two-group reference, not the latent-gap chain).
// ---------------------------------------------------------------------------
template <int NR>
__device__ __forceinline__ void gemv_native_fused_body(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm, int nks, int M, int N)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    // Compile-time epilogue slot bound: NR=16 serves N in 9..=16 -> 2 groups, else 1 group
    // (the dispatch picks NR by N, so the runtime ng is determined by the template).
    constexpr int NG = (NR == 16) ? 2 : 1;
    const int K = nks * 64;
    __shared__ float sh[MMA_SMEM];
    __shared__ bf16 shw[MMA_NW][8][64];
    __shared__ float scw[MMA_NW][8][4];

    for (int mt = blockIdx.x; mt < ntm; mt += gridDim.x) {
        __syncthreads();                       // sh reuse barrier (write->read->next write)
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)mt * nks * 128;
        const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)mt * nks * 16;

        // Fixed k-visit order per tile: warp w takes ksteps w, w+8, ... (N-independent).
        for (int ks = warp; ks < nks; ks += MMA_NW) {
            // (b)-(d) STAGE -> SCALE -> PACK -> MMA per 8-row group (NG=2 only for NR=16).
            // One uniform code path for every NR: rows >= N are zero-staged, so dead rows go
            // through the standalone kernel's exact inv-0 path (scale 0 -> inv 0 -> +0 codes)
            // and the fragments are byte-identical to mxfp4_quant_pack_b's zeroed rows. The
            // __syncwarp() after the mma orders the next pass's stage writes against this
            // pass's pack reads. The A-side loads are per-pass (the quant compute hides under
            // the weight-load latency; NR=16 reloads them for pass 1 instead of holding 5
            // registers live across both passes).
#pragma unroll
            for (int q = 0; q < NG; q++) {
                // (a) A-side loads FIRST — issued before the stage so the quant ALU below
                // executes in the weight-load latency shadow.
                const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
                const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
                const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
                const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
                const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
                // (b) STAGE the pass's 8 rows, 2 bf16 elems/lane/row (coalesced 128-B warp
                // transaction per row, the standalone kernel's exact pattern).
#pragma unroll 4
                for (int n = 0; n < 8; n++) {
                    const int row = q * 8 + n;
                    if (row < N) {
                        const bf16* rp = X + (size_t)row * K + (size_t)ks * 64 + lane * 2;
                        shw[warp][n][lane * 2 + 0] = rp[0];
                        shw[warp][n][lane * 2 + 1] = rp[1];
                    } else {
                        shw[warp][n][lane * 2 + 0] = f2b(0.f);
                        shw[warp][n][lane * 2 + 1] = f2b(0.f);
                    }
                }
                __syncwarp();
                // (c) SCALE: lane (n,v) = (lane>>2, lane&3) computes row n's kblock v — the
                // same ascending fmaxf(fabsf()) chain over the same 16 values, amax/6.0f,
                // e4m3_ceil (dead rows -> amax 0 -> scale byte 0).
                {
                    const int n = lane >> 2, v = lane & 3;
                    float amax = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(b2f(shw[warp][n][v * 16 + i])));
                    scw[warp][n][v] = e4m3_ceil(amax / 6.0f);
                }
                __syncwarp();
                // (d) PACK + MMA: lane (g,t) packs token (q*8+g)'s fragment, all lanes active.
                const float inv0 = scw[warp][g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][t >> 1]);
                const float inv1 = scw[warp][g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][2 + (t >> 1)]);
                uint32_t b0 = 0, b1 = 0, sfb = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][8 * t + 2 * j]) * inv0, b2f(shw[warp][g][8 * t + 2 * j + 1]) * inv0);
                    b0 |= cb << (8 * j);
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][32 + 8 * t + 2 * j]) * inv1, b2f(shw[warp][g][32 + 8 * t + 2 * j + 1]) * inv1);
                    b1 |= cb << (8 * j);
                }
                if (t == 0) {
                    unsigned char b0c = (unsigned char)scw[warp][g][0], b1c = (unsigned char)scw[warp][g][1];
                    unsigned char b2c = (unsigned char)scw[warp][g][2], b3c = (unsigned char)scw[warp][g][3];
                    sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                    : "+f"(acc[q][0]), "+f"(acc[q][1]), "+f"(acc[q][2]), "+f"(acc[q][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
                      "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
                __syncwarp();   // pass boundary: next pass's stage writes wait for this pack's reads
            }
        }

        // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
        for (int q = 0; q < 2; q++)
#pragma unroll
            for (int i = 0; i < 4; i++) sh[(q * 4 + i) * 256 + warp * 32 + lane] = acc[q][i];
        __syncthreads();
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        if (rslot < 4 * NG) {
            const int q = rslot >> 2, i = rslot & 3;
            const int col = q * 8 + 2 * t + (i & 1);
            if (col < N) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
                const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
                C[(long long)col * M + m] = f2b(v * gs[mt]);   // gs applied exactly once
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemv_native_fused_b1(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm, int nks, int M, int N)
{
    gemv_native_fused_body<1>(C, Wt, SFAw, gs, X, ntm, nks, M, N);
}

extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemv_native_fused_b8(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm, int nks, int M, int N)
{
    gemv_native_fused_body<8>(C, Wt, SFAw, gs, X, ntm, nks, M, N);
}

extern "C" __global__ __launch_bounds__(256, 5) void mxfp4_gemv_native_fused_b16(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm, int nks, int M, int N)
{
    gemv_native_fused_body<16>(C, Wt, SFAw, gs, X, ntm, nks, M, N);
}

// ---------------------------------------------------------------------------
// OMMA-layout element accessor + dequant-to-bf16 (the prefill path's twin of
// gpu_batch.cu fp4_tiled_at / dequant_fp4_tiled_b). Economy mode stores ONLY the
// OMMA repack (Aimg/SFAw/gs) in the W's qweight/scales/gs fields — prefill (batch >
// MAX_VERIFY) dequantizes from that layout before cuBLAS. Inverting the verified
// A-fragment map: element (row r, col c) lives in reg (hr | (q<<1)) nibble j of
// lane (g,t); its scale is SFAw byte (c>>4) of the u32 (g, hr) [row g+8*hr].
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp4_omma_at(const uint8_t* Aimg, const uint8_t* SFAw,
                                             const float* gs, int nks, int row, int c) {
    const int r = row & 15, cc = c & 63;
    const int g = r & 7, hr = r >> 3;
    const int t = (cc >> 3) & 3, j = cc & 7, q = cc >> 5;
    const int lane = g * 4 + t, reg = hr | (q << 1);
    const long long base = ((long long)(row >> 4) * nks + (c >> 6)) * 128 + lane * 4 + reg;
    const uint32_t u = *reinterpret_cast<const uint32_t*>(Aimg + base * 4);
    const uint8_t nib = (uint8_t)((u >> (4 * j)) & 0xF);
    const uint32_t sf = *reinterpret_cast<const uint32_t*>(SFAw
        + ((((long long)(row >> 4) * nks + (c >> 6)) * 16 + g * 2 + hr) * 4));
    const uint8_t sb = (uint8_t)((sf >> (8 * ((c >> 4) & 3))) & 0xFF);
    return e2m1_f(nib) * ue4m3_f(sb) * gs[row >> 4];
}

extern "C" __global__ void mxfp4_dequant_tiled_b(bf16* out, const uint8_t* Aimg,
                                                 const uint8_t* SFAw, const float* gs,
                                                 int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    out[i] = f2b(fp4_omma_at(Aimg, SFAw, gs, K >> 6, (int)(i / K), (int)(i % K)));
}

// Economy-mode embed gather: the OMMA-layout twin of embed_gather_fp4_tiled_b — the embed's
// qweight/scales ARE the OMMA repack when the mode runs (a standard-layout read there would
// silently permute the embeddings, which zeroes MTP acceptance).
extern "C" __global__ void mxfp4_embed_gather_tiled_b(bf16* out, const uint8_t* Aimg,
                                                      const uint8_t* SFAw, const float* gs,
                                                      const int* tokens, int h, int batch) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)h * batch) return;
    out[i] = f2b(fp4_omma_at(Aimg, SFAw, gs, h >> 6, tokens[i / h], i % h));
}

// ---- P4 B2 helpers ----
__device__ __forceinline__ void mxf4_mma(
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, unsigned sfa, unsigned sfb,
    float& d0, float& d1, float& d2, float& d3)
{
    const unsigned short z = 0;
    asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%0,  %1,  %2,  %3},"
    "{%10},"
    "{%11, %12},"
    "{%13},"
    "{%14, %15};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "r"(sfa), "h"(z), "h"(z),
      "r"(sfb), "h"(z), "h"(z));
}
__device__ __forceinline__ uint8_t f2ue4m3(float x) {
    // e4m3 satfinite via the cuda_fp8 intrinsic (RNE, proven on sm_121a). Positive
    // domain (amax-derived scales), so sign bit never set — e4m3 == ue4m3 bytes here.
    __nv_fp8_storage_t r = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return (uint8_t)r;
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ===========================================================================
// P4 B2: W4A4 prefill path (K-B quantizer + K-A GEMM). Added 2026-08-18.
//
// K-B `mxfp4_quant_prefill_b`: bf16 activations X (batch x K, row-major) ->
// e2m1 nibbles Bq (batch x K/2 bytes, row-major, k-packed low nibble first)
// + ue4m3 scales SB (batch x K/16, one per 16-k group). Per-group amax is
// computed in fp32, scale = amax/6, quantize round-nearest-even satfinite via
// the sm_121a cvt.e2m1x2 instruction (same as the existing pack helpers).
//
// K-A `mxfp4_gemm_prefill_b`: the bit-exact BK=256 2-stage cp.async OMMA GEMM
// (tool_probe/p4_gemm_perf6.cu lineage, maxerr 0 vs CPU at 512^3/1024^3/2048^3,
// 94.6 TF @2048 on metal) operating on the NATIVE row-major fp4 layouts:
//   A = weights in the OMMA repack (Aimg/SFAw/gs) is NOT used here — instead the
//   standard row-major qweight (outn x K nibbles) + scales (outn x K/16) layout,
//   the same bytes the dequant path reads. Rationale: one layout serves both
//   decode (W4A16 GEMV reads qweight) and prefill; the repack is decode-only.
//   The gs (per-16-row f32 global scale) applies ONLY to the OMMA repack, so it
//   is NOT applied here.
// C = out is bf16 (batch x outn, row-major) — written from the f32 accumulators.
// Constraints: K multiple of 256; outn multiple of 128; batch multiple of 128
// (caller pads — prefill windows are power-of-two-ish and scratch-padded).
// ===========================================================================
__device__ __forceinline__ unsigned char f2e2m1x2_rne(float lo, float hi) {
    // e2m1 pair via the cuda_fp4 intrinsic (RNE satfinite; PLAN/02 P4-entry probe).
    __nv_fp4x2_storage_t r = __nv_cvt_float2_to_fp4x2(make_float2(lo, hi), __NV_E2M1, cudaRoundNearest);
    return (unsigned char)r;
}

extern "C" __global__ void mxfp4_quant_prefill_b(
    uint8_t* __restrict__ Bq, uint8_t* __restrict__ SB,
    const bf16* __restrict__ X, int batch, int K)
{
    // one thread per 16-k group; groups are contiguous per row
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long ng = (long long)batch * (K >> 4);
    if (g >= ng) return;
    const int k0 = (int)(g % (K >> 4)) << 4;
    const int r = (int)(g / (K >> 4));
    float v[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v[i] = b2f(X[(long long)r * K + k0 + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float scale = amax / 6.0f;
    const uint8_t sb = f2ue4m3(scale);
    SB[g] = sb;
    const float inv = scale > 0.f ? 1.0f / (ue4m3_f(sb)) : 0.f;
    #pragma unroll
    for (int b = 0; b < 8; b++)
        Bq[g * 8 + b] = (uint8_t)f2e2m1x2_rne(v[2*b] * inv, v[2*b+1] * inv);
}

extern "C" __global__ void __launch_bounds__(256) mxfp4_gemm_prefill_b(
    const uint8_t* __restrict__ A,   // weights: outn x K/2 nibble-packed (row-major)
    const uint8_t* __restrict__ Bq,  // acts:    batch x K/2 nibble-packed (row-major)
    const uint8_t* __restrict__ SFA, // outn x K/16
    const uint8_t* __restrict__ SB,  // batch x K/16
    const float* __restrict__ gs,    // per-16-row global scale (indexed m>>4), as dequant applies
    bf16* __restrict__ C,            // batch x outn
    int M, int N, int K)             // M=outn, N=batch, K=K (perf6 naming)
{
    // ---- verbatim perf6 body (mxf4_mma wrapper above already in file) ----
    #define ARSTRIDE 144
    #define SCRSTRIDE 16
    #define STAGES 2
    #define A_TILEP (128 * ARSTRIDE)
    #define SC_TILEP (128 * SCRSTRIDE)
    #define STAGE_BYTESP (2*A_TILEP + 2*SC_TILEP)
    __shared__ unsigned char smem[STAGES * STAGE_BYTESP];
    unsigned char* A_s[STAGES]; unsigned char* B_s[STAGES];
    unsigned char* SFA_s[STAGES]; unsigned char* SFB_s[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        A_s[s]   = smem + s * STAGE_BYTESP;
        B_s[s]   = A_s[s] + A_TILEP;
        SFA_s[s] = B_s[s] + A_TILEP;
        SFB_s[s] = SFA_s[s] + SC_TILEP;
    }
    const int t = threadIdx.x;
    const int warp = t / 32, lane = t % 32;
    const int wm = warp / 4, wn = warp % 4;
    const int mb = blockIdx.y * 128, nb = blockIdx.x * 128;
    const int kc = K / 256;

    float acc[4][4][4] = {};
    unsigned a[4][4], b[4][2];

    // prologue: stage chunks 0..STAGES-1 fully (A, B, scales per chunk)
    #pragma unroll
    for (int c0 = 0; c0 < STAGES; c0++) {
        const int kb = c0 * 256;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            int idx = q * 256 + t, r = idx >> 3, off = (idx & 7) * 16;
            cp16(A_s[c0] + r * ARSTRIDE + off, A + (long long)(mb + r) * (K / 2) + kb/2 + off);
            cp16(B_s[c0] + r * ARSTRIDE + off, Bq + (long long)(nb + r) * (K / 2) + kb/2 + off);
        }
        {
            int r2 = t & 127;
            unsigned sA = (unsigned)__cvta_generic_to_shared(SFA_s[c0] + r2 * SCRSTRIDE);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sA),
                "l"(SFA + (long long)(mb + r2) * (K / 16) + kb / 16));
            unsigned sB = (unsigned)__cvta_generic_to_shared(SFB_s[c0] + r2 * SCRSTRIDE);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sB),
                "l"(SB + (long long)(nb + r2) * (K / 16) + kb / 16));
        }
        cp_commit();
    }

    for (int c = 0; c < kc; c++) {
        const int s = c % STAGES;
        cp_wait<STAGES - 2>();
        __syncthreads();

        #pragma unroll
        for (int ka = 0; ka < 4; ka++) {
            #pragma unroll
            for (int ma = 0; ma < 4; ma++) {
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    int mloc = wm * 64 + ma * 16 + (lane >> 2) + 8 * (w & 1);
                    a[ma][w] = *(const unsigned*)(A_s[s] + mloc * ARSTRIDE + ka * 32 + ((w >> 1) * 32 + (lane & 3) * 8) / 2);
                }
            }
            #pragma unroll
            for (int na = 0; na < 4; na++) {
                #pragma unroll
                for (int w = 0; w < 2; w++) {
                    int cloc = wn * 32 + na * 8 + (lane >> 2);
                    b[na][w] = *(const unsigned*)(B_s[s] + cloc * ARSTRIDE + ka * 32 + (w * 32 + (lane & 3) * 8) / 2);
                }
            }
            #pragma unroll
            for (int ma = 0; ma < 4; ma++) {
                // OMMA SFA lane mapping (verified, gpu_mxfp4.cu header): lane (g,t) byte v
                // covers row g + 8*(t&1), kblock v. t in {2,3} ignored by hardware.
                int mloc = wm * 64 + ma * 16 + (lane >> 2) + 8 * (lane & 1);
                unsigned sfa = *(const unsigned*)(SFA_s[s] + mloc * SCRSTRIDE + ka * 4);
                #pragma unroll
                for (int na = 0; na < 4; na++) {
                    int cloc = wn * 32 + na * 8 + (lane >> 2);
                    unsigned sfb = *(const unsigned*)(SFB_s[s] + cloc * SCRSTRIDE + ka * 4);
                    mxf4_mma(a[ma][0],a[ma][1],a[ma][2],a[ma][3], b[na][0],b[na][1], sfa, sfb,
                             acc[ma][na][0],acc[ma][na][1],acc[ma][na][2],acc[ma][na][3]);
                }
            }
        }
        __syncthreads();
        int pf = c + STAGES - 1;
        if (pf < kc) {
            int sp = pf % STAGES;
            const int kb = pf * 256;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int idx = q * 256 + t, r = idx >> 3, off = (idx & 7) * 16;
                cp16(A_s[sp] + r * ARSTRIDE + off, A + (long long)(mb + r) * (K / 2) + kb/2 + off);
                cp16(B_s[sp] + r * ARSTRIDE + off, Bq + (long long)(nb + r) * (K / 2) + kb/2 + off);
            }
            {
                int r2 = t & 127;
                unsigned sA = (unsigned)__cvta_generic_to_shared(SFA_s[sp] + r2 * SCRSTRIDE);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sA),
                    "l"(SFA + (long long)(mb + r2) * (K / 16) + kb / 16));
                unsigned sB = (unsigned)__cvta_generic_to_shared(SFB_s[sp] + r2 * SCRSTRIDE);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sB),
                    "l"(SB + (long long)(nb + r2) * (K / 16) + kb / 16));
            }
            cp_commit();
        }
    }

    #pragma unroll
    for (int ma = 0; ma < 4; ma++)
        #pragma unroll
        for (int na = 0; na < 4; na++) {
            int r0 = mb + wm * 64 + ma * 16 + (lane >> 2);
            int c0 = nb + wn * 32 + na * 8 + (lane & 3) * 2;
            #ifdef PF4_NO_GS
            const float g = 1.0f;
            #else
            const float g = gs[r0 >> 4];   // r1 = r0+8 shares the 16-row tile
            #endif
            // ENGINE LAYOUT: C is TOKEN-major (out[token * M + feature]); the perf6
            // standalone wrote feature-major C[m*N + n]. Store transposed.
            // OMMA C fragment (verified mapping): acc[i] = feature g + 8*(i>=2),
            // token 2t + (i&1). So: acc0=(r0,c0) acc1=(r0,c0+1) acc2=(r0+8,c0) acc3=(r0+8,c0+1).
            int r1 = r0 + 8, c1t = c0 + 1;
            C[(long long)c0  * M + r0] = f2b(acc[ma][na][0] * g);
            C[(long long)c1t * M + r0] = f2b(acc[ma][na][1] * g);
            C[(long long)c0  * M + r1] = f2b(acc[ma][na][2] * g);
            C[(long long)c1t * M + r1] = f2b(acc[ma][na][3] * g);
        }
    #undef ARSTRIDE
    #undef SCRSTRIDE
    #undef STAGES
    #undef A_TILEP
    #undef SC_TILEP
    #undef STAGE_BYTESP
}

// ---------------------------------------------------------------------------
// P4 B2 repack: standard tiled MMA layout -> row-major, for the W4A4 prefill
// GEMM (mxfp4_gemm_prefill_b reads row-major [M, K/2] qweight + [M, K/16]
// scales). Element mapping is fp4_tiled_at's (gpu_batch.cu), inverted: the
// output byte (row, k/2) packs nibbles k (lo) and k+1 (hi).
// Scales: Sct[((row>>4)*nblk + (k>>4))*16 + (row&15)] -> Srm[row*nblk + (k>>4)].
// Runs ONCE per tensor (cached in Mxfp4State.rm), not per prefill.
// ---------------------------------------------------------------------------
extern "C" __global__ void mxfp4_repack_rm_b(
    uint8_t* __restrict__ Wrm, uint8_t* __restrict__ Srm,
    const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    int M, int K)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int nblk = K >> 4;
    const long long nbytes = (long long)M * (K >> 1);
    if (i < nbytes) {
        const int row = (int)(i / (K >> 1));
        const int k = (int)((i % (K >> 1)) * 2);
        const int r = row & 15;
        uint8_t b = 0;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int c = k + h, cc = c & 15;
            const int lane = (r & 7) * 4 + ((cc & 7) >> 1);
            const int j    = (r >> 3) | ((cc >> 3) << 1);
            const long long tile = (long long)(row >> 4) * nblk + (c >> 4);
            const uint8_t byte = Wt[tile * 128 + lane * 4 + j];
            const uint8_t nib  = (cc & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
            b |= (uint8_t)(nib << (4 * h));
        }
        Wrm[i] = b;
    }
    // scales ride the same launch: element si = i - nbytes covers M*nblk
    const long long si = i - nbytes;
    if (si >= 0 && si < (long long)M * nblk) {
        const int row = (int)(si / nblk);
        const int blk = (int)(si % nblk);
        Srm[si] = Sct[((long long)(row >> 4) * nblk + blk) * 16 + (row & 15)];
    }
}

// ---------------------------------------------------------------------------
// P4 TTFT (2026-08-25): OMMA-native PREFILL GEMM v2 + batched act quantizer.
// Champion lineage: tool_probe/mxf4_gemm_v2b_champ.cu — MATCH relL2 1.654e-3 vs f64
// oracle, 213-253 TF/s (old W4A4 path: 94.6 @2048; nvjet bf16: 100). Key facts
// (measured, PLAN/13):
//   * OMMA mxf4nvf4 issue latency ~45-90 cyc/warp: 8 warps/SM cap ~150 TF/s. 16 warps
//     (2 blocks x 8, acc 64 = 32x64 warp tile) reach 230-287 compute-only.
//   * LDGSTS returns 16 B/cyc/SM on L2 hits; first-touch Wt misses cap fill at ~4.7 B/cyc
//     unless the token-fastest raster makes 8 co-resident blocks share each Wt tile.
//   * ptxas sm_121f can DROP mma D-fragment stores under acc-128 full-unroll pressure —
//     acc-64 configs have not shown it; the engine-side CHECK mode + xchain gates watch.
//   * SFB lane packing: word (q,ks,lane) is read at every lane but only t==0 lanes are
//     consumed (token g = lane>>2); quantizer writes real bytes on t==0, 0 elsewhere.
// A = weights (m = feature, Aimg/SFAw/gs); B = acts (n = token, Bp/SFB). Out token-major.
// Constraints: Mf%128==0, K%256==0, Nt%128==0 (caller pads Nt to 128).
// ---------------------------------------------------------------------------
extern "C" __global__ void mxfp4_quant_pack_prefill_b(
    const bf16* __restrict__ X, int K, int M, int Mpad,
    uint32_t* __restrict__ Bp, uint32_t* __restrict__ SFB)
{
    // Batched port of mxfp4_quant_pack_b (decode-proven arithmetic, statement-identical):
    // grid (nks, Mpad/8); block 256 = 8 token rows x 32 lanes; rows >= M zero-fill
    // (padded rows produce zero nibbles + zero scales -> exact 0 contributions).
    const int ks = blockIdx.x, q = blockIdx.y;
    const int nks = gridDim.x;
    const int n = threadIdx.x >> 5;          // token 0..7 within the group
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int row = q * 8 + n;
    __shared__ float sh[8][64];
    if (row < M) {
        const bf16* xr = X + (long long)row * K + (long long)ks * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(xr[lane * 2 + i]);
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = 0.0f;
    }
    __syncthreads();
    __shared__ float sc[8][4];
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();
    uint32_t b0 = 0, b1 = 0;
    const float inv0 = sc[g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][t >> 1]);
    const float inv1 = sc[g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][2 + (t >> 1)]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][8 * t + 2 * j] * inv0, sh[g][8 * t + 2 * j + 1] * inv0);
        b0 |= cb << (8 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][32 + 8 * t + 2 * j] * inv1, sh[g][32 + 8 * t + 2 * j + 1] * inv1);
        b1 |= cb << (8 * j);
    }
    Bp[((size_t)q * nks + ks) * 64 + lane * 2 + 0] = b0;
    Bp[((size_t)q * nks + ks) * 64 + lane * 2 + 1] = b1;
    uint32_t sfb = 0;
    if (t == 0) {
        unsigned char b0c = (unsigned char)sc[g][0], b1c = (unsigned char)sc[g][1];
        unsigned char b2c = (unsigned char)sc[g][2], b3c = (unsigned char)sc[g][3];
        sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
    }
    SFB[((size_t)q * nks + ks) * 32 + lane] = sfb;
}

#define PF2_BM 128
#define PF2_BN 128
#define PF2_STAGE_A 4096
#define PF2_STAGE_B 4096
#define PF2_STAGE_SFA 512
#define PF2_STAGE_SFB 2048
#define PF2_STAGE_BYTES (PF2_STAGE_A + PF2_STAGE_B + PF2_STAGE_SFA + PF2_STAGE_SFB)  // 10752
#define PF2_NSTAGES 4

extern "C" __global__ __launch_bounds__(256, 2) void mxfp4_gemm_prefill_v2_b(
    const uint8_t* __restrict__ Wt8,     // Aimg: [(Mf/16)*nks][128] u32 lane-major
    const uint8_t* __restrict__ SFAw8,   // [(Mf/16)*nks][16] u32
    const uint32_t* __restrict__ Bp,     // [(Nt/8)*nks][64]
    const uint32_t* __restrict__ SFBw,   // [(Nt/8)*nks][32]
    const float* __restrict__ gs,        // [Mf/16]
    bf16* __restrict__ Out,              // token-major [Nt][Mf]
    int Mf, int Nt, int K)
{
    const uint32_t* Wt = reinterpret_cast<const uint32_t*>(Wt8);
    const uint32_t* SFAw = reinterpret_cast<const uint32_t*>(SFAw8);
    const int nks = K >> 6;
    const int tid = threadIdx.x;
    const int lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;        // 4m x 2n warp tiles, 32x64 each

    // token-fastest raster, group width 8: 8 co-resident blocks share each Wt tile (L2)
    const int tm = (Mf + PF2_BM - 1) / PF2_BM, tn = (Nt + PF2_BN - 1) / PF2_BN;
    const int gw = 8;
    const int per_ng = tm * gw;
    const int ng = blockIdx.x / per_ng, rem = blockIdx.x % per_ng;
    const int bm = (rem / gw) * PF2_BM;
    const int bn = (ng * gw + rem % gw) * PF2_BN;

    extern __shared__ uint8_t smem[];
    auto load_stage = [&](int s, int ks) {
        uint8_t* base = smem + s * PF2_STAGE_BYTES;
        // flat chunk map: [0,256) A | [256,512) B | [512,544) SFA | [544,672) SFB
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx = tid + i * 256;
            if (idx >= 672) break;
            if (idx < 256) {                                    // A: 8 slices x 512B
                int sl = idx >> 5, ch = idx & 31;
                int mt = (bm >> 4) + sl;
                if (mt * 16 < Mf)
                    cp16(base + sl * 512 + ch * 16, Wt + ((size_t)mt * nks + ks) * 128 + ch * 4);
                else
                    *(uint4*)(base + sl * 512 + ch * 16) = make_uint4(0u, 0u, 0u, 0u);
            } else if (idx < 512) {                             // B: 16 slices x 256B
                int j = idx - 256, q = j >> 4, ch = j & 15;
                int qg = (bn >> 3) + q;
                if (qg * 8 < Nt)
                    cp16(base + PF2_STAGE_A + q * 256 + ch * 16, Bp + ((size_t)qg * nks + ks) * 64 + ch * 4);
                else
                    *(uint4*)(base + PF2_STAGE_A + q * 256 + ch * 16) = make_uint4(0u, 0u, 0u, 0u);
            } else if (idx < 544) {                             // SFA: 8 slices x 64B
                int j = idx - 512, sl = j >> 2, ch = j & 3;
                int mt = (bm >> 4) + sl;
                if (mt * 16 < Mf)
                    cp16(base + PF2_STAGE_A + PF2_STAGE_B + sl * 64 + ch * 16,
                         SFAw + ((size_t)mt * nks + ks) * 16 + ch * 4);
                else
                    *(uint4*)(base + PF2_STAGE_A + PF2_STAGE_B + sl * 64 + ch * 16) = make_uint4(0u, 0u, 0u, 0u);
            } else {                                            // SFB: 16 slices x 128B
                int j = idx - 544, q = j >> 3, ch = j & 7;
                int qg = (bn >> 3) + q;
                if (qg * 8 < Nt)
                    cp16(base + PF2_STAGE_A + PF2_STAGE_B + PF2_STAGE_SFA + q * 128 + ch * 16,
                         SFBw + ((size_t)qg * nks + ks) * 32 + ch * 4);
                else
                    *(uint4*)(base + PF2_STAGE_A + PF2_STAGE_B + PF2_STAGE_SFA + q * 128 + ch * 16) =
                        make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_commit();
    };

    load_stage(0, 0);
    if (nks > 1) load_stage(1, 1);
    if (nks > 2) load_stage(2, 2);

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

    for (int kt = 0; kt < nks; kt++) {
        const int s = kt % PF2_NSTAGES;
        const int committed = (kt + 3 < nks) ? (kt + 3) : nks;
        const int need = committed - kt - 1;
        if (need <= 0) cp_wait<0>();
        else if (need == 1) cp_wait<1>();
        else cp_wait<2>();
        __syncthreads();
        if (kt + 3 < nks) load_stage((kt + 3) % PF2_NSTAGES, kt + 3);

        const uint8_t* st = smem + s * PF2_STAGE_BYTES;
        const uint32_t* sfa_s = (const uint32_t*)(st + PF2_STAGE_A + PF2_STAGE_B);
        const uint32_t* sfb_s = (const uint32_t*)(st + PF2_STAGE_A + PF2_STAGE_B + PF2_STAGE_SFA);

        uint32_t a[2][4], b[8][2], sfa[2], sfb[8];
#pragma unroll
        for (int ma = 0; ma < 2; ma++) {
            const uint4* p = (const uint4*)(st + ((wm * 2 + ma) * 128 + lane * 4) * 4);
            a[ma][0] = p->x; a[ma][1] = p->y; a[ma][2] = p->z; a[ma][3] = p->w;
        }
#pragma unroll
        for (int na = 0; na < 8; na++) {
            const uint2* p = (const uint2*)(st + PF2_STAGE_A + ((wn * 8 + na) * 32 + lane) * 8);
            b[na][0] = p->x; b[na][1] = p->y;
        }
#pragma unroll
        for (int ma = 0; ma < 2; ma++)
            sfa[ma] = (t <= 1) ? sfa_s[(wm * 2 + ma) * 16 + g * 2 + t] : 0u;
#pragma unroll
        for (int na = 0; na < 8; na++)
            sfb[na] = sfb_s[(wn * 8 + na) * 32 + lane];
#pragma unroll
        for (int ma = 0; ma < 2; ma++)
#pragma unroll
            for (int na = 0; na < 8; na++)
                mxf4_mma(a[ma][0], a[ma][1], a[ma][2], a[ma][3],
                         b[na][0], b[na][1], sfa[ma], sfb[na],
                         acc[ma][na][0], acc[ma][na][1], acc[ma][na][2], acc[ma][na][3]);
    }

    // smem-transpose epilogue: alias the dead stage smem; token-major tile, 8-feat pad;
    // 16-B vector Out stores (the raw fragment store pattern is 2-B token-strided).
    __syncthreads();
    bf16* tile = (bf16*)smem;                 // [128 tok][136 feat] = 34,816 B <= 43,008
    const int TSTRIDE = 136;
#pragma unroll
    for (int ma = 0; ma < 2; ma++) {
#pragma unroll
        for (int na = 0; na < 8; na++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                int fl = wm * 32 + ma * 16 + g + 8 * (c >= 2);
                int tl = wn * 64 + na * 8 + 2 * t + (c & 1);
                // clamp for the ragged-M tail: pad columns would index gs one row past its
                // [Mf/16] end (OOB read); their tile values are never stored anyway
                const int gi = (bm + fl) < Mf ? ((bm + fl) >> 4) : ((Mf - 1) >> 4);
                tile[tl * TSTRIDE + fl] = f2b(acc[ma][na][c] * gs[gi]);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; i++) {             // 2048 chunks / 256 threads
        int idx = tid + i * 256;
        int tl = idx >> 4, f8 = (idx & 15) * 8;
        int tok = bn + tl, feat = bm + f8;
        if (tok >= Nt) continue;
        if (feat + 7 < Mf) {
            *(uint4*)(Out + (size_t)tok * Mf + feat) = *(const uint4*)(tile + tl * TSTRIDE + f8);
        } else {
            // ragged M tail (mixer projections: e.g. GDN fused mtot=16480 = 128*128+96) —
            // element stores for the final partial 8-feat chunk
#pragma unroll
            for (int e = 0; e < 8; e++)
                if (feat + e < Mf)
                    Out[(size_t)tok * Mf + feat + e] = tile[tl * TSTRIDE + f8 + e];
        }
    }
}
