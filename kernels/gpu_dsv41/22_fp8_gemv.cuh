// FP8 block-scaled GEMV / small-M kernel (rows 1..32). Out features sit on the MMA's M axis (A = weights),
// tokens on its N axis (B = activations, 8 per n-tile). One CTA = MT*16 output rows x KS warps splitting K
// (units of 4 K-blocks interleaved over warps, ascending within a warp); the KS partial sums are combined in
// smem in the fixed order 0..KS-1, so the result does not depend on `rows` (batch invariant across the
// GEMV (<=2) and cooperative (3..16) classes: NT = 1 (<=8 rows) or 2 (<=16) only changes the live token columns).
// Weights stream through a WARP-PRIVATE cp.async ring (S stages of R rows x 128 B): 16 B coalesced segments
// give DRAM whole 128 B lines (8-byte per-lane loads of 32 B sectors topped out at ~140 GB/s). Fragments are
// read from smem with the swizzle  chunk' = ((j ^ (row & 3)) << 2) | t  (conflict-free).
// K-slot permutation (consistent in A and B): MMA k-slot 4t+i <- byte 8t+i, 16+4t+i <- 8t+4+i of the 32-block.
#pragma once

#include "23_fp8_gemm_prims.cuh"

template <int MT, int NT, int KS, int S, int OT>
__device__ __forceinline__ void d41_fp8_gemv_body(const d41_fp8_args a) {
    D41_PDL_ENTRY();
    extern __shared__ __align__(128) unsigned char d41_gv_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int R = MT * 16, T = NT * 8, STG = R * 128;
    unsigned char* ring = d41_gv_smem + warp * S * STG;
    const int nbase = a.n0 + blockIdx.x * R;
    const int nend = a.n0 + a.n_count;
    const unsigned char* wsrow = a.ws + (long long)(nbase >> 5) * a.ldws + a.k0b;
    const int units = (a.nkb + 3) >> 2;

    auto issue = [&](int u, int buf) {
        unsigned char* dst = ring + buf * STG;
        const int kb0 = u * 4;
        #pragma unroll
        for (int i = 0; i < R * 8 / 32; ++i) {
            const int idx = lane + i * 32, row = idx >> 3, seg = idx & 7;
            const int gr = nbase + row;
            const bool ok = (u < units) && gr < nend && (kb0 + (seg >> 1)) < a.nkb;
            const unsigned char* src = ok ? a.w + (long long)gr * a.ldw + (long long)(a.k0b + kb0) * 32 + seg * 16 : a.w;
            d41_cp16_zfill(dst + row * 128 + ((seg ^ ((row & 3) << 1)) << 4), src, ok);
        }
    };
    struct XU { uint2 x[NT][4]; uint32_t sa[NT][2]; uint32_t sb; };
    auto load_x = [&](int u, XU& U) {
        const int kb0 = u * 4;
        const int cnt = (u < units) ? min(4, a.nkb - kb0) : 0;
        #pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int tg = n * 8 + g;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                U.x[n][j] = (tg < a.rows && j < cnt) ? __ldg((const uint2*)(a.xq + (long long)tg * a.ldx + (long long)(kb0 + j) * 32 + t * 8)) : make_uint2(0u, 0u);
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = n * 8 + 2 * t + c;
                U.sa[n][c] = (tk < a.rows && cnt > 0) ? d41_ld_sc4(a.xs + (long long)tk * a.ldxs + kb0, cnt) : 0u;
            }
        }
        U.sb = cnt > 0 ? d41_ld_sc4(wsrow + kb0, cnt) : 0u;
    };

    float acc[MT][NT][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n)
            #pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][n][c] = 0.f;

    const int myunits = warp < units ? (units - warp + KS - 1) / KS : 0;
    #pragma unroll
    for (int s = 0; s < S - 1; ++s) { issue(warp + s * KS, s); d41_cp_commit(); }
    XU xc, xn;
    load_x(warp, xc);
    for (int i = 0; i < myunits; ++i) {
        const int u = warp + i * KS;
        d41_cp_wait<S - 2>();
        __syncwarp();
        issue(warp + (i + S - 1) * KS, (i + S - 1) % S); d41_cp_commit();
        load_x(u + KS, xn);
        const unsigned char* st = ring + (i % S) * STG;
        const int cnt = min(4, a.nkb - u * 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < cnt) {
                const int ch = (((j ^ (g & 3)) << 2) | t) * 8;
                const float sb = d41_sc_f(xc.sb, j);
                #pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const uint2 a0 = *(const uint2*)(st + (m * 16 + g) * 128 + ch);
                    const uint2 a1 = *(const uint2*)(st + (m * 16 + 8 + g) * 128 + ch);
                    #pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        float c4[4];
                        d41_mma_e4m3_k32(c4, a0.x, a1.x, a0.y, a1.y, xc.x[n][j].x, xc.x[n][j].y);
                        const float f0 = sb * d41_sc_f(xc.sa[n][0], j), f1 = sb * d41_sc_f(xc.sa[n][1], j);
                        acc[m][n][0] = fmaf(c4[0], f0, acc[m][n][0]);
                        acc[m][n][1] = fmaf(c4[1], f1, acc[m][n][1]);
                        acc[m][n][2] = fmaf(c4[2], f0, acc[m][n][2]);
                        acc[m][n][3] = fmaf(c4[3], f1, acc[m][n][3]);
                    }
                }
            }
        }
        xc = xn;
    }
    d41_cp_wait<0>();
    __syncthreads();                                              // the rings are free: reuse as the partial buffer
    float* part = (float*)d41_gv_smem;                            // [KS][T][R]
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {
            float* p = part + (warp * T + n * 8 + 2 * t) * R + m * 16 + g;
            p[0] = acc[m][n][0]; p[R] = acc[m][n][1]; p[8] = acc[m][n][2]; p[R + 8] = acc[m][n][3];
        }
    __syncthreads();
    for (int i = threadIdx.x; i < R * T; i += KS * 32) {
        const int row = i % R, tok = i / R;
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < KS; ++w) s += part[(w * T + tok) * R + row];
        if (tok < a.rows && nbase + row < nend) d41_store_out(a.y, OT, (long long)tok * a.ldy + (nbase + row - a.n0), s, a.accum);
    }
}

// smem = KS * S * MT*16*128 bytes = 48 KB for every entry (no opt-in needed)
#define D41_GEMV_ENTRY(MT, NT, KS, S, NAME) \
    extern "C" __global__ void __launch_bounds__(KS * 32) NAME##_bf16(const d41_fp8_args a) { d41_fp8_gemv_body<MT, NT, KS, S, 0>(a); } \
    extern "C" __global__ void __launch_bounds__(KS * 32) NAME##_f32(const d41_fp8_args a) { d41_fp8_gemv_body<MT, NT, KS, S, 1>(a); }
D41_GEMV_ENTRY(1, 1, 8, 3, d41_fp8_gemv_m1n1k8)
D41_GEMV_ENTRY(1, 2, 8, 3, d41_fp8_gemv_m1n2k8)
D41_GEMV_ENTRY(2, 1, 4, 3, d41_fp8_gemv_m2n1k4)
D41_GEMV_ENTRY(2, 2, 4, 3, d41_fp8_gemv_m2n2k4)
// 17..32 rows (the 4-lane verify width and the batch-4 decode widths stay on the SAME split-K structure)
D41_GEMV_ENTRY(1, 3, 8, 3, d41_fp8_gemv_m1n3k8)
D41_GEMV_ENTRY(1, 4, 8, 3, d41_fp8_gemv_m1n4k8)
D41_GEMV_ENTRY(2, 3, 4, 3, d41_fp8_gemv_m2n3k4)
D41_GEMV_ENTRY(2, 4, 4, 3, d41_fp8_gemv_m2n4k4)
