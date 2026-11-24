// FP8 block-scaled tensor-core GEMM for prefill (rows >= 17). CTA tile 128 output rows x 64 tokens, K stage
// = 4 K-blocks (128), 3-stage cp.async pipeline, 8 warps each 32 x 32 (2 m16 tiles of weights x 4 n8 tiles of
// tokens). Same operand roles and K-slot permutation as the GEMV: A = weights, B = activations; every K
// block is one fresh-accumulator MMA scaled by sa*sb into an f32 accumulator, blocks in ascending order.
// smem rows are 128 B (4 blocks); 8-byte fragment loads are conflict-free with the swizzle
//   chunk' = ((j ^ (row & 3)) << 2) | t   (j = block in stage, t = lane & 3).
// Grid: 1-D, tiles_n * tiles_m, grouped raster (see the body).
#pragma once

#define D41_GM_BN 128
#define D41_GM_BM 64
#define D41_GM_STAGES 3
#define D41_GM_STAGE_BYTES ((D41_GM_BN + D41_GM_BM) * 128)

template <int OT>
__device__ __forceinline__ void d41_fp8_gemm_body(const d41_fp8_args a) {
    D41_PDL_ENTRY();
    extern __shared__ __align__(128) unsigned char d41_gm_smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wn = warp & 3, wm = warp >> 2;
    // 1-D grid, grouped raster: a group of `gn` n-tiles sweeps all token tiles (concurrent CTAs share both the
    // group's weight tiles and a few token tiles in L2); weights and activations are each read ~once per group
    const int tiles_n = (a.n_count + D41_GM_BN - 1) / D41_GM_BN, tiles_m = (a.rows + D41_GM_BM - 1) / D41_GM_BM;
    const int gn = max(1, a.gn), span = gn * tiles_m, grp = blockIdx.x / span, first_n = grp * gn;
    const int gsz = min(gn, tiles_n - first_n), r_in = blockIdx.x - grp * span;
    const int tn = first_n + r_in % gsz, tm = r_in / gsz;
    const int tok0 = tm * D41_GM_BM;
    const int nrow0 = a.n0 + tn * D41_GM_BN;
    const int nend = a.n0 + a.n_count;
    const int nst = (a.nkb + 3) >> 2;

    auto issue = [&](int stage_k, int buf) {
        unsigned char* sA = d41_gm_smem + buf * D41_GM_STAGE_BYTES;
        unsigned char* sB = sA + D41_GM_BN * 128;
        const int kbase = stage_k * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                 // A: 128 rows x 8 segments
            const int idx = tid + i * 256, row = idx >> 3, seg = idx & 7;
            const int gr = nrow0 + row;
            const bool ok = gr < nend && (kbase + (seg >> 1)) < a.nkb;
            const unsigned char* src = ok ? a.w + (long long)gr * a.ldw + (long long)(a.k0b + kbase) * 32 + seg * 16 : a.w;
            d41_cp16_zfill(sA + row * 128 + ((seg ^ ((row & 3) << 1)) << 4), src, ok);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {                 // B: 64 tokens x 8 segments
            const int idx = tid + i * 256, row = idx >> 3, seg = idx & 7;
            const int gr = tok0 + row;
            const bool ok = gr < a.rows && (kbase + (seg >> 1)) < a.nkb;
            const unsigned char* src = ok ? a.xq + (long long)gr * a.ldx + (long long)kbase * 32 + seg * 16 : a.xq;
            d41_cp16_zfill(sB + row * 128 + ((seg ^ ((row & 3) << 1)) << 4), src, ok);
        }
    };

    // scales of one stage for this lane: weight block row of its 32 rows, 8 token columns
    // (n0 is only 16-aligned, so the two m16 tiles of a warp may sit in different 32-row scale blocks)
    const unsigned char* wsrow[2];
    bool wrow_ok[2];
    #pragma unroll
    for (int m = 0; m < 2; ++m) { wsrow[m] = a.ws + (long long)((nrow0 + wn * 32 + m * 16) >> 5) * a.ldws + a.k0b; wrow_ok[m] = (nrow0 + wn * 32 + m * 16) < nend; }
    uint32_t sbc[2] = {0, 0}, sac[4][2], sbn[2], san[4][2];
    auto load_sc = [&](int stage_k, uint32_t (&sb)[2], uint32_t (&sa)[4][2]) {
        const int kb0 = stage_k * 4;
        const int cnt = stage_k < nst ? min(4, a.nkb - kb0) : 0;
        #pragma unroll
        for (int m = 0; m < 2; ++m) sb[m] = (wrow_ok[m] && cnt > 0) ? d41_ld_sc4(wsrow[m] + kb0, cnt) : 0u;
        #pragma unroll
        for (int n = 0; n < 4; ++n)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = tok0 + wm * 32 + n * 8 + 2 * t + c;
                sa[n][c] = (tk < a.rows && cnt > 0) ? d41_ld_sc4(a.xs + (long long)tk * a.ldxs + kb0, cnt) : 0u;
            }
    };

    float acc[2][4][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int n = 0; n < 4; ++n)
            #pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][n][c] = 0.f;

    int pfb[2] = {0, 0}, pfa[4][2];
    float cp[2][4][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int n = 0; n < 4; ++n) { cp[m][n][0] = cp[m][n][1] = cp[m][n][2] = cp[m][n][3] = 0.f; pfa[n][0] = pfa[n][1] = 0; }
    #pragma unroll
    for (int s = 0; s < D41_GM_STAGES - 1; ++s) { if (s < nst) issue(s, s); d41_cp_commit(); }
    load_sc(0, sbc, sac);

    for (int it = 0; it < nst; ++it) {
        d41_cp_wait<D41_GM_STAGES - 2>();
        __syncthreads();
        { const int nk = it + D41_GM_STAGES - 1; if (nk < nst) issue(nk, nk % D41_GM_STAGES); d41_cp_commit(); }
        load_sc(it + 1, sbn, san);

        const unsigned char* sA = d41_gm_smem + (it % D41_GM_STAGES) * D41_GM_STAGE_BYTES;
        const unsigned char* sB = sA + D41_GM_BN * 128;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ch = (((j ^ (g & 3)) << 2) | t) * 8;
            uint2 af[2][2], bf[4];
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int h = 0; h < 2; ++h) af[m][h] = *(const uint2*)(sA + (wn * 32 + m * 16 + h * 8 + g) * 128 + ch);
            #pragma unroll
            for (int n = 0; n < 4; ++n) bf[n] = *(const uint2*)(sB + (wm * 32 + n * 8 + g) * 128 + ch);
            // issue this block's MMAs first, then fold the PREVIOUS block's results: the tensor pipe stays
            // busy while the FMAs of the last block run (scales are exact powers of two: C * sa * sb is exact)
            float cn[2][4][4];
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int n = 0; n < 4; ++n)
                    d41_mma_e4m3_k32(cn[m][n], af[m][0].x, af[m][1].x, af[m][0].y, af[m][1].y, bf[n].x, bf[n].y);
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const float f0 = __int_as_float(pfb[m] + pfa[n][0]), f1 = __int_as_float(pfb[m] + pfa[n][1]);
                    acc[m][n][0] = fmaf(cp[m][n][0], f0, acc[m][n][0]);
                    acc[m][n][1] = fmaf(cp[m][n][1], f1, acc[m][n][1]);
                    acc[m][n][2] = fmaf(cp[m][n][2], f0, acc[m][n][2]);
                    acc[m][n][3] = fmaf(cp[m][n][3], f1, acc[m][n][3]);
                }
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                pfb[m] = (int)d41_sc_exp(sbc[m], j) - (127 << 23);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    cp[m][n][0] = cn[m][n][0]; cp[m][n][1] = cn[m][n][1]; cp[m][n][2] = cn[m][n][2]; cp[m][n][3] = cn[m][n][3];
                    pfa[n][0] = (int)d41_sc_exp(sac[n][0], j); pfa[n][1] = (int)d41_sc_exp(sac[n][1], j);
                }
            }
        }
        sbc[0] = sbn[0]; sbc[1] = sbn[1];
        #pragma unroll
        for (int n = 0; n < 4; ++n) { sac[n][0] = san[n][0]; sac[n][1] = san[n][1]; }
    }
    #pragma unroll
    for (int m = 0; m < 2; ++m)       // drain: fold the last block
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const float f0 = __int_as_float(pfb[m] + pfa[n][0]), f1 = __int_as_float(pfb[m] + pfa[n][1]);
            acc[m][n][0] = fmaf(cp[m][n][0], f0, acc[m][n][0]);
            acc[m][n][1] = fmaf(cp[m][n][1], f1, acc[m][n][1]);
            acc[m][n][2] = fmaf(cp[m][n][2], f0, acc[m][n][2]);
            acc[m][n][3] = fmaf(cp[m][n][3], f1, acc[m][n][3]);
        }
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = nrow0 + wn * 32 + m * 16 + h * 8 + g;
            if (row >= nend) continue;
            #pragma unroll
            for (int n = 0; n < 4; ++n)
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int tk = tok0 + wm * 32 + n * 8 + 2 * t + c;
                    if (tk < a.rows) d41_store_out(a.y, OT, (long long)tk * a.ldy + (row - a.n0), acc[m][n][h * 2 + c], a.accum);
                }
        }
}

extern "C" __global__ void d41_fp8_gemm_bf16(const d41_fp8_args a) { d41_fp8_gemm_body<0>(a); }
extern "C" __global__ void d41_fp8_gemm_f32(const d41_fp8_args a) { d41_fp8_gemm_body<1>(a); }
