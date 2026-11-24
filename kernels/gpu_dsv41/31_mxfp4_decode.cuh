// =============================================================================
// kernels/gpu_dsv41/31_mxfp4_decode.cuh — band 30 (leg KN2): grouped routed-expert kernels for
// decode and DSpark verify, <= 32 rows (row classes GEMV <= 2, cooperative 3-16, and the 17-32
// rows of 3-4 verifying lanes). Template NT = token tiles of 8 per expert: an expert can be routed
// by at most `rows` tokens, so NT = ceil(rows / 8) covers every routing.
//
//   d41_mx_fc1_g{1..4}: per (distinct local expert, 32 inner outputs) one CTA of 4 warps, warp w =
//     m-tile 4q + w (8 gate + 8 up rows); gate / up = bf16(fp4_gemm), SwiGLU with clamps and the
//     routing weight, bf16, then act_quant per 32 inner values -> hq e4m3 [route][ldh] + hs ue8m0.
//     Persistent over (expert, group) items; the grid is fixed per row count (graph-capturable).
//   d41_mx_fc2_g{1..4}: per 16 output columns one warp; it walks the distinct local experts in
//     ascending id, runs w2 on that expert's routes, rounds each expert's output to bf16 (the
//     reference's per-expert rounding; f32 partials when this rank holds an inner slice) and adds it
//     into the token's FP32 sum IN ASCENDING EXPERT ORDER (REF_MAP §4 item 12). One write of y.
// Every weight byte is read once per call; each output is the 30-file arithmetic contract.
// =============================================================================
#pragma once

struct D41MxFc1Args {
    const uint8_t* w;              // packed w13 (mx1), local expert 0
    unsigned long long e_stride;   // bytes per local expert
    unsigned long long mt_stride;  // bytes per m-tile block
    const uint8_t* xq;             // e4m3 [rows][ldx]
    const uint8_t* xs;             // ue8m0 [rows][ldxs]
    const int* ids;                // [rows * top_k] global expert ids
    const float* rw;               // [rows * top_k] routing weights (renormalized, x route_scale)
    uint8_t* hq;                   // e4m3 [route][ldh]
    uint8_t* hs;                   // ue8m0 [route][ldhs]
    __nv_bfloat16* dbg;            // optional [route][2][slice] gate / up (bf16 fp4_gemm outputs)
    int rows, top_k, e_first, n_local;
    int nkb;                       // dim / 32
    int slice;                     // inner width on this rank (multiple of 32)
    int ldx, ldxs, ldh, ldhs;
    float limit;
    int pad_;
};

struct D41MxFc2Args {
    const uint8_t* w;              // packed w2 (mx1), local expert 0
    unsigned long long e_stride;
    unsigned long long mt_stride;
    const uint8_t* hq;             // e4m3 [route][ldh]
    const uint8_t* hs;             // ue8m0 [route][ldhs]
    const int* ids;
    float* y;                      // f32 [rows][ldy]
    float* route_out;              // optional f32 [route][dim]: each route's own output (exactness mode)
    int rows, top_k, e_first, n_local;
    int nkb;                       // slice / 32
    int dim;
    int ldh, ldhs, ldy;
    int round_bf16;                // 1: per-expert bf16 rounding (whole experts); 0: f32 partial (inner slice)
    int accumulate;                // 1: y += sum; 0: y = sum
    int pad_;
};

static_assert(sizeof(D41MxFc1Args) == 128, "D41MxFc1Args mirrors src/dsv41/weights/native/mxfp4_launch.rs Fc1Args");
static_assert(sizeof(D41MxFc2Args) == 112, "D41MxFc2Args mirrors src/dsv41/weights/native/mxfp4_launch.rs Fc2Args");

// prefetch depth in quads (register ring): deeper for the narrow classes, which are latency-bound
#define D41_MX_DEPTH(NT) ((NT) == 1 ? 4 : ((NT) == 2 ? 3 : 2))

// ---- one quad (4 k-blocks) of loads ----------------------------------------------------------
template <int NT>
struct D41MxQuad {
    uint2 a[4];
    uint32_t b[NT][4][2];
    uint32_t sfa;
    uint32_t sfb[NT];
};

template <int NT>
__device__ __forceinline__ void d41_mx_load_quad(D41MxQuad<NT>& Q, const uint8_t* A, const uint8_t* As, int kq, int nkb,
                                                 int lane, int g, int t, const uint8_t* (&brow_q)[NT],
                                                 const uint8_t* (&brow_s)[NT]) {
    Q.sfa = d41_mx_ldw32(As + kq * D41_MX_SCALES_UNIT + (2 * g + (t & 1)) * 4);
#pragma unroll
    for (int v = 0; v < 4; v++) {
        const int kb = 4 * kq + v;
        Q.a[v] = kb < nkb ? d41_mx_ldw(A + (size_t)kb * D41_MX_CODES_UNIT + lane * 8) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const uint8_t* bq = brow_q[nt];
        Q.sfb[nt] = bq ? d41_mx_lda(brow_s[nt] + 4 * kq) : 0u;
#pragma unroll
        for (int v = 0; v < 4; v++) {
            const int kb = 4 * kq + v;
            const bool ok = bq != nullptr && kb < nkb;
            Q.b[nt][v][0] = ok ? d41_mx_lda(bq + kb * 32 + 4 * t) : 0u;
            Q.b[nt][v][1] = ok ? d41_mx_lda(bq + kb * 32 + 16 + 4 * t) : 0u;
        }
    }
}

template <int NT>
__device__ __forceinline__ void d41_mx_compute_quad(const D41MxQuad<NT>& Q, int kq, int nkb, float (&acc)[NT][4]) {
#pragma unroll
    for (int v = 0; v < 4; v++) {
        if (4 * kq + v < nkb) {   // warp-uniform
            const uint32_t a0 = d41_mx_lo4(Q.a[v].x), a1 = d41_mx_hi4(Q.a[v].x);
            const uint32_t a2 = d41_mx_lo4(Q.a[v].y), a3 = d41_mx_hi4(Q.a[v].y);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                float d[4];
                d41_mx_mma_v(v, d, a0, a1, a2, a3, Q.b[nt][v][0], Q.b[nt][v][1], Q.sfa, Q.sfb[nt]);
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][i] += d[i];
            }
        }
    }
}

// ---- FC1 ---------------------------------------------------------------------------------------
template <int NT>
__device__ __forceinline__ void d41_mx_fc1_g_body(const D41MxFc1Args& p) {
    __shared__ D41MxPlan P;
    __shared__ float red[4][NT * 8];
    d41_mx_build_plan(P, p.ids, p.rows * p.top_k, p.e_first, p.n_local);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int nq = p.slice >> 5, nkq = (p.nkb + 3) >> 2;
    const int nitems = P.n_dist * nq;
    for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
        const int d = item / nq, q = item - d * nq;
        const int e = P.dist_e[d], c = P.cnt[d], st = P.start[d];
        const int j = 4 * q + warp;
        const uint8_t* A = p.w + (size_t)e * p.e_stride + (size_t)j * p.mt_stride;
        const uint8_t* As = A + (size_t)p.nkb * D41_MX_CODES_UNIT;
        const uint8_t* bq[NT];
        const uint8_t* bs[NT];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int s = nt * 8 + g;
            const int row = s < c ? P.sorted[st + s] / p.top_k : -1;
            bq[nt] = row >= 0 ? p.xq + (size_t)row * p.ldx : nullptr;
            bs[nt] = row >= 0 ? p.xs + (size_t)row * p.ldxs : nullptr;
        }
        float acc[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[nt][i] = 0.0f;
        constexpr int D = D41_MX_DEPTH(NT);
        D41MxQuad<NT> Q[D];
#pragma unroll
        for (int i = 0; i < D - 1; i++)
            if (i < nkq) d41_mx_load_quad<NT>(Q[i], A, As, i, p.nkb, lane, g, t, bq, bs);
        for (int kq0 = 0; kq0 < nkq; kq0 += D) {
#pragma unroll
            for (int i = 0; i < D; i++) {
                const int kq = kq0 + i;
                if (kq < nkq) {
                    if (kq + D - 1 < nkq) d41_mx_load_quad<NT>(Q[(i + D - 1) % D], A, As, kq + D - 1, p.nkb, lane, g, t, bq, bs);
                    d41_mx_compute_quad<NT>(Q[i], kq, p.nkb, acc);
                }
            }
        }
        // ---- epilogue: SwiGLU per (inner row 8j+g, token slot), act_quant over the CTA's 32 inner values
        float hb[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int s = nt * 8 + 2 * t + i;
                const bool ok = s < c;
                const int r = ok ? P.sorted[st + s] : 0;
                const float gb = d41_mx_bf16r(acc[nt][i]), ub = d41_mx_bf16r(acc[nt][2 + i]);
                const float h = ok ? d41_mx_swiglu(gb, ub, p.rw[r], p.limit) : 0.0f;
                hb[nt][i] = h;
                if (ok && p.dbg) {
                    __nv_bfloat16* dg = p.dbg + (size_t)r * 2 * p.slice;
                    dg[j * 8 + g] = __float2bfloat16(gb);
                    dg[p.slice + j * 8 + g] = __float2bfloat16(ub);
                }
                float am = fabsf(h);
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                if (g == 0) red[warp][s] = am;
            }
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int s = nt * 8 + 2 * t + i;
                if (s < c) {
                    const int r = P.sorted[st + s];
                    float am = fmaxf(fmaxf(red[0][s], red[1][s]), fmaxf(red[2][s], red[3][s]));
                    am = fmaxf(am, 1e-4f);
                    const float sc = d41_mx_pow2_scale(am);
                    p.hq[(size_t)r * p.ldh + j * 8 + g] = d41_mx_e4m3(hb[nt][i], sc);
                    if (warp == 0 && g == 0) p.hs[(size_t)r * p.ldhs + q] = d41_mx_e8m0_of(sc);
                }
            }
        }
        __syncthreads();
    }
}

// ---- FC2 + ordered combine --------------------------------------------------------------------
template <int NT>
__device__ __forceinline__ void d41_mx_fc2_g_body(const D41MxFc2Args& p) {
    __shared__ D41MxPlan P;
    __shared__ float yacc[4][16][D41_MX_GROWS];
    d41_mx_build_plan(P, p.ids, p.rows * p.top_k, p.e_first, p.n_local);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int j = blockIdx.x * 4 + warp;           // 16 output columns
    const int nkq = (p.nkb + 3) >> 2;
    for (int i = lane; i < 16 * D41_MX_GROWS; i += 32) (&yacc[warp][0][0])[i] = 0.0f;
    __syncwarp();
    const int nd = P.n_dist;
    const int total = nd * nkq;
    const bool live = j * 16 < p.dim;
    // the flat (expert, quad) walk with one quad of prefetch
    auto setup = [&](int it, const uint8_t*& A, const uint8_t*& As, const uint8_t* (&bq)[NT], const uint8_t* (&bs)[NT], int& kq) {
        const int d = it / nkq;
        kq = it - d * nkq;
        const int e = P.dist_e[d], c = P.cnt[d], st = P.start[d];
        A = p.w + (size_t)e * p.e_stride + (size_t)j * p.mt_stride;
        As = A + (size_t)p.nkb * D41_MX_CODES_UNIT;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int s = nt * 8 + g;
            const int r = s < c ? P.sorted[st + s] : -1;
            bq[nt] = r >= 0 ? p.hq + (size_t)r * p.ldh : nullptr;
            bs[nt] = r >= 0 ? p.hs + (size_t)r * p.ldhs : nullptr;
        }
    };
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[nt][i] = 0.0f;
    if (live && total > 0) {
        constexpr int D = D41_MX_DEPTH(NT);
        D41MxQuad<NT> Q[D];
        auto load_it = [&](int it, D41MxQuad<NT>& q) {
            const uint8_t *A, *As;
            const uint8_t* bq[NT];
            const uint8_t* bs[NT];
            int kq;
            setup(it, A, As, bq, bs, kq);
            d41_mx_load_quad<NT>(q, A, As, kq, p.nkb, lane, g, t, bq, bs);
        };
        auto finish = [&](int it) {   // after the last quad of an expert: round, add into the token sums
            const int d = it / nkq;
            if (it - d * nkq != nkq - 1) return;
            const int c = P.cnt[d], st = P.start[d];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int s = nt * 8 + 2 * t + (i & 1);
                    const int col = g + 8 * (i >> 1);
                    float v = acc[nt][i];
                    acc[nt][i] = 0.0f;
                    if (s < c) {
                        const int r = P.sorted[st + s];
                        if (p.round_bf16) v = d41_mx_bf16r(v);
                        if (p.route_out) p.route_out[(size_t)r * p.dim + j * 16 + col] = v;
                        yacc[warp][col][r / p.top_k] += v;
                    }
                }
            }
            __syncwarp();
        };
#pragma unroll
        for (int i = 0; i < D - 1; i++)
            if (i < total) load_it(i, Q[i]);
        for (int it0 = 0; it0 < total; it0 += D) {
#pragma unroll
            for (int i = 0; i < D; i++) {
                const int it = it0 + i;
                if (it < total) {
                    if (it + D - 1 < total) load_it(it + D - 1, Q[(i + D - 1) % D]);
                    d41_mx_compute_quad<NT>(Q[i], it % nkq, p.nkb, acc);
                    finish(it);
                }
            }
        }
    }
    __syncwarp();
    if (live) {
        for (int i = lane; i < 16 * p.rows; i += 32) {
            const int row = i >> 4, col = i & 15;
            float* yp = p.y + (size_t)row * p.ldy + j * 16 + col;
            const float v = yacc[warp][col][row];
            *yp = p.accumulate ? *yp + v : v;
        }
    }
}

#define D41_MX_GROUPED(NT)                                                                            \
    extern "C" __global__ void __launch_bounds__(128) d41_mx_fc1_g##NT(const D41MxFc1Args p) {        \
        D41_PDL_ENTRY();                                                                              \
        d41_mx_fc1_g_body<NT>(p);                                                                     \
    }                                                                                                 \
    extern "C" __global__ void __launch_bounds__(128) d41_mx_fc2_g##NT(const D41MxFc2Args p) {        \
        D41_PDL_ENTRY();                                                                              \
        d41_mx_fc2_g_body<NT>(p);                                                                     \
    }
D41_MX_GROUPED(1)
D41_MX_GROUPED(2)
D41_MX_GROUPED(3)
D41_MX_GROUPED(4)
#undef D41_MX_GROUPED
