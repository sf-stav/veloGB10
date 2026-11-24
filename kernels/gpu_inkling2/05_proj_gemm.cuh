// gpu_inkling2/05_proj_gemm.cuh — the fused BF16 projection GEMM (K1). Section of gpu_inkling2.cu.
//
// Y[M][ldy] (fp32) = X[M][K] (bf16) . W^T, W the checkpoint's bf16 [N][K] stored in the fragment-native
// layout "B32" (a pure u16 permutation made at load, src/inkling2/gpu/k1.rs `b32_pack`, N padded to a
// multiple of 16 with zero rows; the FP4L family's layout): W'[t][g][h][lane][8] holds weight row
// 16t + 8h + gid, k = 32g + 8tig + j at u16 j (gid = lane >> 2, tig = lane & 3). One 32-k group of a
// 16-row tile is 1 KB contiguous; a warp's 16-B loads ARE its B fragments.
//
// Activations are bf16 (the bf16 reference casts every linear's input to bf16: products exact in fp32).
// The k-permutation (lane t owns actual k 8t..8t+7 of each 32-k group in both A and B) is fixed, so the
// reduction order of every output is fixed.
//
// FUSION by SEGMENTS: up to 3 activation streams share one launch. Tiles [0, t1) read X0, [t1, t2) X1,
// [t2, ...) X2 (all [M][K]). A sliding layer concatenates [q | gate | r | sink | k | v] (one input); a full
// layer [q | gate | r | sink] on X0 (q input), [k | k-dyn] on X1, [v | v-dyn] on X2. Segment boundaries are
// 16-row tile boundaries (each segment padded to 16 rows at load).
//
// ink_proj_dec<KS> (M <= 16, decode / verify): block 256 = 8 warps = (8 / KS) tiles x KS k-slices; the KS
//   slices of a tile reduce through shared memory in ascending slice order (fixed). Grid ceil(T / (8/KS)).
//   (K / 32) % (KS * grid.y) == 0. grid.y > 1 = cross-CTA k-slices: slice y writes its PARTIAL sums to Y + y * ysl
//   (the consumer adds the slices in ascending order: ink_msggate_out). Batch invariant: a token row's output depends only on its own X row (rows >= M read
//   a clamped valid row and are never stored), and the reduction order does not depend on M.
// ink_proj_wide (any M, prefill): grid (ceil(T / 8), ceil(M / 64)), one 16-row tile per warp, 4 token tiles
//   of 16 per CTA reusing each weight fragment 4 times. Rows independent (sequence-parallel safe).

__device__ __forceinline__ const uint16_t* ika_seg_x(const uint16_t* X0, const uint16_t* X1, const uint16_t* X2,
                                                    int tile, int t1, int t2) {
    return tile < t1 ? X0 : (tile < t2 ? X1 : X2);
}

// One 32-k group of one 16-row weight tile against token rows gid (xa) and gid + 8 (xb).
__device__ __forceinline__ void ika_bf_group(float acc[4], const uint4 xa, const uint4 xb, const uint4 w) {
    ika_mma_bf(acc, xa.x, xb.x, xa.y, xb.y, w.x, w.y);
    ika_mma_bf(acc, xa.z, xb.z, xa.w, xb.w, w.z, w.w);
}

// S7 PDL prefetch: this CTA's weight slice = its TPB tiles x (K / 32) groups x 1 KB, contiguous; the cross-CTA
// k-slices (grid.y) split it further: slice y owns groups [y G / gy, ...) of every tile (not contiguous across
// tiles), so the prefetch takes the first tile's part of this slice.
__device__ __forceinline__ const void* ika_proj_pf_ptr(const uint16_t* W, int K, int ntiles, int tpb) {
    const int G = K >> 5, gy = G / gridDim.y;
    const int tile = min((int)blockIdx.x * tpb, ntiles - 1);
    return W + ((size_t)tile * G + (size_t)blockIdx.y * gy) * 512;
}
__device__ __forceinline__ unsigned ika_proj_pf_bytes(int K, int ntiles, int tpb) {
    const int G = K >> 5, gy = G / gridDim.y;
    const int tiles = min(tpb, ntiles - (int)blockIdx.x * tpb);
    const unsigned region = gridDim.y > 1 ? (unsigned)gy * 1024u : (unsigned)max(tiles, 0) * (unsigned)G * 1024u;
    return min(region, INK2_PF_BYTES);
}
template <int KS>
__device__ __forceinline__ void ink_proj_dec_body(const uint16_t* __restrict__ W, const uint16_t* __restrict__ X0,
        const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2, int t1, int t2,
        float* __restrict__ Y, int ldy, int M, int K, int ntiles, int ysl) {
    constexpr int TPB = 8 / KS;
    __shared__ float red[8][32][8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tl = warp / KS, ks = warp % KS;
    const int tile = blockIdx.x * TPB + tl;
    const bool live = tile < ntiles;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    if (live) {
        const uint16_t* X = ika_seg_x(X0, X1, X2, tile, t1, t2);
        const int G = K >> 5, gy = G / gridDim.y, gs = gy / KS;   // grid.y = cross-CTA k-slices (partials)
        const int g0 = blockIdx.y * gy + ks * gs, g1 = g0 + gs;
        const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
        const uint16_t* xa = X + (size_t)ta * K + 8 * tig;
        const uint16_t* xb = X + (size_t)tb * K + 8 * tig;
        const uint16_t* wt = W + (size_t)tile * G * 512 + lane * 8;
        int g = g0;
        for (; g + 4 <= g1; g += 4) {
            uint4 b0[4], b1[4], a4[4], bb4[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                b0[u] = ika_ldw4(wt + (size_t)(g + u) * 512);
                b1[u] = ika_ldw4(wt + (size_t)(g + u) * 512 + 256);
                a4[u] = ika_lda4(xa + (g + u) * 32);
                bb4[u] = ika_lda4(xb + (g + u) * 32);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                ika_bf_group(acc[0], a4[u], bb4[u], b0[u]);
                ika_bf_group(acc[1], a4[u], bb4[u], b1[u]);
            }
        }
        for (; g < g1; ++g) {
            const uint4 b0 = ika_ldw4(wt + (size_t)g * 512), b1 = ika_ldw4(wt + (size_t)g * 512 + 256);
            const uint4 a = ika_lda4(xa + g * 32), b = ika_lda4(xb + g * 32);
            ika_bf_group(acc[0], a, b, b0);
            ika_bf_group(acc[1], a, b, b1);
        }
    }
    if (KS > 1) {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) red[warp][lane][h * 4 + r] = acc[h][r];
        __syncthreads();
        if (ks != 0) return;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                float s = red[warp][lane][h * 4 + r];
                #pragma unroll
                for (int j = 1; j < KS; ++j) s += red[warp + j][lane][h * 4 + r];
                acc[h][r] = s;
            }
    }
    if (!live) return;
    Y += (size_t)blockIdx.y * ysl;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int n = tile * 16 + 8 * h + 2 * tig;
        if (gid < M) *reinterpret_cast<float2*>(&Y[(size_t)gid * ldy + n]) = make_float2(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *reinterpret_cast<float2*>(&Y[(size_t)(gid + 8) * ldy + n]) = make_float2(acc[h][2], acc[h][3]);
    }
}

#define INK_PROJ_DEC(KS)                                                                                         \
extern "C" __global__ __launch_bounds__(256) void ink_proj_dec_ks##KS(const uint16_t* __restrict__ W,           \
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,       \
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles, int ysl) {                    \
    INK2_PDL_ENTRY_PREFETCH(ika_proj_pf_ptr(W, K, ntiles, 8 / KS), ika_proj_pf_bytes(K, ntiles, 8 / KS));      \
    ink_proj_dec_body<KS>(W, X0, X1, X2, t1, t2, Y, ldy, M, K, ntiles, ysl);                                     \
}
INK_PROJ_DEC(1)
INK_PROJ_DEC(2)
INK_PROJ_DEC(4)
INK_PROJ_DEC(8)

// S7: ink_proj_dec3<KS> — ink_proj_dec<KS> for 17..48 rows (a verify of several lanes): the same tiles, k-slices,
// slice reduction order and per-row MMA chain, with up to three 16-row token groups sharing every weight fragment
// (the weight is read once whatever M). A row's output is bit-identical to the same row of ink_proj_dec (and so to
// a 1-row decode). Same grid / ysl semantics.
template <int KS>
__device__ __forceinline__ void ink_proj_dec3_body(const uint16_t* __restrict__ W, const uint16_t* __restrict__ X0,
        const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2, int t1, int t2,
        float* __restrict__ Y, int ldy, int M, int K, int ntiles, int ysl) {
    constexpr int TPB = 8 / KS, NG = 3;
    __shared__ float red[8][32][8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tl = warp / KS, ks = warp % KS;
    const int tile = blockIdx.x * TPB + tl;
    const bool live = tile < ntiles;
    const int ng = (M + 15) >> 4;
    float acc[NG][2][4];
    #pragma unroll
    for (int q = 0; q < NG; ++q)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[q][h][r] = 0.f;
    if (live) {
        const uint16_t* X = ika_seg_x(X0, X1, X2, tile, t1, t2);
        const int G = K >> 5, gy = G / gridDim.y, gs = gy / KS;
        const int g0 = blockIdx.y * gy + ks * gs, g1 = g0 + gs;
        const uint16_t* xa[NG];
        const uint16_t* xb[NG];
        #pragma unroll
        for (int q = 0; q < NG; ++q) {
            xa[q] = X + (size_t)min(16 * q + gid, M - 1) * K + 8 * tig;
            xb[q] = X + (size_t)min(16 * q + gid + 8, M - 1) * K + 8 * tig;
        }
        const uint16_t* wt = W + (size_t)tile * G * 512 + lane * 8;
        int g = g0;
        for (; g + 4 <= g1; g += 4) {
            uint4 b0[4], b1[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                b0[u] = ika_ldw4(wt + (size_t)(g + u) * 512);
                b1[u] = ika_ldw4(wt + (size_t)(g + u) * 512 + 256);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                #pragma unroll
                for (int q = 0; q < NG; ++q) {
                    if (q < ng) {
                        const uint4 a = ika_lda4(xa[q] + (g + u) * 32), b = ika_lda4(xb[q] + (g + u) * 32);
                        ika_bf_group(acc[q][0], a, b, b0[u]);
                        ika_bf_group(acc[q][1], a, b, b1[u]);
                    }
                }
            }
        }
        for (; g < g1; ++g) {
            const uint4 b0 = ika_ldw4(wt + (size_t)g * 512), b1 = ika_ldw4(wt + (size_t)g * 512 + 256);
            #pragma unroll
            for (int q = 0; q < NG; ++q) {
                if (q < ng) {
                    const uint4 a = ika_lda4(xa[q] + g * 32), b = ika_lda4(xb[q] + g * 32);
                    ika_bf_group(acc[q][0], a, b, b0);
                    ika_bf_group(acc[q][1], a, b, b1);
                }
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < NG; ++q) {
        if (KS > 1) {
            if (q > 0) __syncthreads();
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) red[warp][lane][h * 4 + r] = acc[q][h][r];
            __syncthreads();
            if (ks == 0) {
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        float s = red[warp][lane][h * 4 + r];
                        #pragma unroll
                        for (int j = 1; j < KS; ++j) s += red[warp + j][lane][h * 4 + r];
                        acc[q][h][r] = s;
                    }
            }
        }
        if (live && ks == 0 && q < ng) {
            float* Yq = Y + (size_t)blockIdx.y * ysl;
            const int ta = 16 * q + gid, tb = ta + 8;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = tile * 16 + 8 * h + 2 * tig;
                if (ta < M) *reinterpret_cast<float2*>(&Yq[(size_t)ta * ldy + n]) = make_float2(acc[q][h][0], acc[q][h][1]);
                if (tb < M) *reinterpret_cast<float2*>(&Yq[(size_t)tb * ldy + n]) = make_float2(acc[q][h][2], acc[q][h][3]);
            }
        }
    }
}
#define INK_PROJ_DEC3(KS)                                                                                        \
extern "C" __global__ __launch_bounds__(256) void ink_proj_dec3_ks##KS(const uint16_t* __restrict__ W,          \
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,       \
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles, int ysl) {                    \
    INK2_PDL_ENTRY_PREFETCH(ika_proj_pf_ptr(W, K, ntiles, 8 / KS), ika_proj_pf_bytes(K, ntiles, 8 / KS));      \
    ink_proj_dec3_body<KS>(W, X0, X1, X2, t1, t2, Y, ldy, M, K, ntiles, ysl);                                    \
}
INK_PROJ_DEC3(1)
INK_PROJ_DEC3(2)
INK_PROJ_DEC3(4)
INK_PROJ_DEC3(8)

extern "C" __global__ __launch_bounds__(256) void ink_proj_wide(const uint16_t* __restrict__ W,
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles) {
    INK2_PDL_ENTRY();
    constexpr int MT = 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    if (tile >= ntiles) return;
    const uint16_t* X = ika_seg_x(X0, X1, X2, tile, t1, t2);
    const int t0 = blockIdx.y * 16 * MT;
    const int G = K >> 5;
    const uint16_t* wt = W + (size_t)tile * G * 512 + lane * 8;
    const uint16_t* xa[MT];
    const uint16_t* xb[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        xa[mt] = X + (size_t)min(t0 + 16 * mt + gid, M - 1) * K + 8 * tig;
        xb[mt] = X + (size_t)min(t0 + 16 * mt + gid + 8, M - 1) * K + 8 * tig;
    }
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
    for (int g = 0; g < G; ++g) {
        const uint4 c0 = ika_ldw4(wt + (size_t)g * 512), c1 = ika_ldw4(wt + (size_t)g * 512 + 256);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const uint4 a = ika_lda4(xa[mt] + g * 32), b = ika_lda4(xb[mt] + g * 32);
            ika_bf_group(acc[mt][0], a, b, c0);
            ika_bf_group(acc[mt][1], a, b, c1);
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int ta = t0 + 16 * mt + gid, tb = ta + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = tile * 16 + 8 * h + 2 * tig;
            if (ta < M) *reinterpret_cast<float2*>(&Y[(size_t)ta * ldy + n]) = make_float2(acc[mt][h][0], acc[mt][h][1]);
            if (tb < M) *reinterpret_cast<float2*>(&Y[(size_t)tb * ldy + n]) = make_float2(acc[mt][h][2], acc[mt][h][3]);
        }
    }
}
