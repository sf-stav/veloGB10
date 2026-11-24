// gpu_inkling2/66_experts_verify.cuh (S7): routed experts at VERIFY widths, 17..48 token rows (a speculative verify of
// 2-4 lanes x up to 12 rows). Below 17 rows the decode kernels (62) run every active expert over all M rows; above
// 16 that wastes tensor work on rows that did not pick the expert, and the prefill tile kernels (63, 64-row x
// 128-column CTAs, 256 threads) leave most of a CTA idle at 1-4 rows per expert. Here each expert computes ONLY the
// rows that picked it (ink2_moe_pf_route's compact order: cnt / offs / rowtok), in 16-row groups that share one
// dequantised weight fragment, with the decode kernels' geometry (grid (N/64, E), 128 threads = 4 warps x one
// 16-row weight tile; experts with no row exit at once). Every expert's weight bytes are read once per call (the
// union), whatever the number of rows.
//   GU: W13 x Xh[rowtok[...]] -> softcapped SwiGLU -> Din compact rows [8M][N/2] f16.
//   DN: W2 x Din compact rows -> Y compact rows [8M][N] fp32 (unweighted; ink2_moe_combine reads rowidx).
// Exactness: a row's MMA sequence (k order, fragment arithmetic) and epilogue are those of ik2m_nv4_step /
// ik2m_mx8_block + ik2m_expert_store, split into "dequantise the fragment" and "MMA it against a row group": the
// output of every row is bit-identical to the decode / prefill kernels (gated by the S7 probe: every verify row ==
// the 1-row decode).

#define IK2V_NG 3   // max 16-row groups per expert (48 rows)

// ---- NVFP4: the dequantised B fragments of one 64-k step (ik2m_nv4_step's first half, verbatim)
__device__ __forceinline__ void ik2v_nv4_deq(const uint32_t n0x, const uint32_t n0y, const uint32_t n1x, const uint32_t n1y,
                                             const uint32_t sb0, const uint32_t sb1, uint32_t b0[8], uint32_t b1[8]) {
    ik2m_e2m1x8(n0x, &b0[0]); ik2m_e2m1x8(n0y, &b0[4]);
    ik2m_e2m1x8(n1x, &b1[0]); ik2m_e2m1x8(n1y, &b1[4]);
    const uint32_t f0 = ik2m_e4m3_bcast(sb0), f1 = ik2m_e4m3_bcast(sb1);
    #pragma unroll
    for (int j = 0; j < 8; ++j) { b0[j] = ik2m_hmul2(b0[j], f0); b1[j] = ik2m_hmul2(b1[j], f1); }
}
// ... and its MMA half against one row group (same instruction order as ik2m_nv4_step)
__device__ __forceinline__ void ik2v_nv4_mma(float acc[2][4], const uint32_t b0[8], const uint32_t b1[8],
                                             const uint4 xa0, const uint4 xa1, const uint4 xb0, const uint4 xb1) {
    ik2m_mma(acc[0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
    ik2m_mma(acc[0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
    ik2m_mma(acc[0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
    ik2m_mma(acc[0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
    ik2m_mma(acc[1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
    ik2m_mma(acc[1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
    ik2m_mma(acc[1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
    ik2m_mma(acc[1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
}

template <bool GU>
__device__ __forceinline__ void ik2v_nv4_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    constexpr int U = 4;
    const int e = blockIdx.y;
    const int c = cnt[e];
    if (c == 0) return;
    if (c > 16 * IK2V_NG) __trap();
    const int ng = (c + 15) >> 4, base = offs[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int KB = K >> 7;
    const size_t tw_stride = (size_t)KB * 1024 + ((K & 64) ? 512 : 0);
    const size_t ts_stride = (size_t)KB * 128 + ((K & 64) ? 64 : 0);
    const uint8_t* we = W + (size_t)e * N * (K >> 1) + (size_t)tile * tw_stride;
    const uint8_t* se = S + (size_t)e * N * (K >> 4) + (size_t)tile * ts_stride;
    const uint8_t* wt = we + lane * 16;
    const uint8_t* st = se + lane * 4;
    const __half* xa[IK2V_NG];
    const __half* xb[IK2V_NG];
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g) {
        const int ra = base + min(16 * g + gid, c - 1), rb = base + min(16 * g + gid + 8, c - 1);
        xa[g] = X + (size_t)(GU ? rowtok[ra] : ra) * K + 16 * tig;
        xb[g] = X + (size_t)(GU ? rowtok[rb] : rb) * K + 16 * tig;
    }
    float acc[IK2V_NG][2][4];
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[g][h][r] = 0.f;
    for (int b = 0; b < KB; b += U) {
        uint4 w0[U], w1[U];
        uint32_t sc[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                w0[u] = ik2m_ldw4(wt + (size_t)(b + u) * 1024);
                w1[u] = ik2m_ldw4(wt + (size_t)(b + u) * 1024 + 512);
                sc[u] = __ldg(reinterpret_cast<const uint32_t*>(st + (size_t)(b + u) * 128));
            }
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                const int k = (b + u) * 128;
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    uint32_t b0[8], b1[8];
                    if (half == 0) ik2v_nv4_deq(w0[u].x, w0[u].y, w1[u].x, w1[u].y, sc[u] & 0xFF, (sc[u] >> 16) & 0xFF, b0, b1);
                    else ik2v_nv4_deq(w0[u].z, w0[u].w, w1[u].z, w1[u].w, (sc[u] >> 8) & 0xFF, sc[u] >> 24, b0, b1);
                    const int kk = k + 64 * half;
                    #pragma unroll
                    for (int g = 0; g < IK2V_NG; ++g) {
                        if (g < ng)
                            ik2v_nv4_mma(acc[g], b0, b1, ik2m_lda4(xa[g] + kk), ik2m_lda4(xa[g] + kk + 8),
                                         ik2m_lda4(xb[g] + kk), ik2m_lda4(xb[g] + kk + 8));
                    }
                }
            }
        }
    }
    if constexpr (!GU) {
        if (K & 64) {
            const uint2 w0h = __ldg(reinterpret_cast<const uint2*>(we + (size_t)KB * 1024 + lane * 8));
            const uint2 w1h = __ldg(reinterpret_cast<const uint2*>(we + (size_t)KB * 1024 + 256 + lane * 8));
            const uint16_t sch = __ldg(reinterpret_cast<const uint16_t*>(se + (size_t)KB * 128 + lane * 2));
            uint32_t b0[8], b1[8];
            ik2v_nv4_deq(w0h.x, w0h.y, w1h.x, w1h.y, sch & 0xFF, sch >> 8, b0, b1);
            const int kk = KB * 128;
            #pragma unroll
            for (int g = 0; g < IK2V_NG; ++g) {
                if (g < ng)
                    ik2v_nv4_mma(acc[g], b0, b1, ik2m_lda4(xa[g] + kk), ik2m_lda4(xa[g] + kk + 8),
                                 ik2m_lda4(xb[g] + kk), ik2m_lda4(xb[g] + kk + 8));
            }
        }
    }
    const float gs = GS[2 * e + (GU ? 0 : 1)];
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g) {
        if (g < ng) {
            const int ra = 16 * g + gid, rb = ra + 8;
            ik2m_expert_store<GU>(acc[g], gs, tile, tig, base + ra, base + rb, ra < c, rb < c, N, Din, Yf);
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void ink2_nv4_gu_vt(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    INK2_PDL_ENTRY();
    ik2v_nv4_body<true>(W, S, GS, X, rowtok, Din, nullptr, cnt, offs, K, N);
}
extern "C" __global__ __launch_bounds__(128) void ink2_nv4_dn_vt(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    INK2_PDL_ENTRY();
    ik2v_nv4_body<false>(W, S, GS, Din, nullptr, nullptr, Y, cnt, offs, K, N);
}

// ---- MXFP8: ik2m_mx8_block split into the fragment conversion and the per-group MMA + exact scale FMAs
template <bool GU>
__device__ __forceinline__ void ik2v_mx8_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    constexpr int U = 4;
    const int e = blockIdx.y;
    const int c = cnt[e];
    if (c == 0) return;
    if (c > 16 * IK2V_NG) __trap();
    const int ng = (c + 15) >> 4, base = offs[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int KB = K >> 6;
    const uint8_t* wt = W + (size_t)e * N * K + (size_t)tile * KB * 1024 + lane * 16;
    const uint8_t* st = S + (size_t)e * N * (K >> 5) + (size_t)tile * KB * 32 + tig * 8;
    const __half* xa[IK2V_NG];
    const __half* xb[IK2V_NG];
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g) {
        const int ra = base + min(16 * g + gid, c - 1), rb = base + min(16 * g + gid + 8, c - 1);
        xa[g] = X + (size_t)(GU ? rowtok[ra] : ra) * K + 8 * tig;
        xb[g] = X + (size_t)(GU ? rowtok[rb] : rb) * K + 8 * tig;
    }
    float acc[IK2V_NG][2][4];
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[g][h][r] = 0.f;
    for (int b = 0; b < KB; b += U) {
        uint4 w0[U], w1[U];
        uint2 sc[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                w0[u] = ik2m_ldw4(wt + (size_t)(b + u) * 1024);
                w1[u] = ik2m_ldw4(wt + (size_t)(b + u) * 1024 + 512);
                sc[u] = __ldg(reinterpret_cast<const uint2*>(st + (size_t)(b + u) * 32));
            }
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                const int k = (b + u) * 64;
                #pragma unroll
                for (int gs = 0; gs < 2; ++gs) {
                    const uint32_t c0lo = gs ? w0[u].z : w0[u].x, c0hi = gs ? w0[u].w : w0[u].y;
                    const uint32_t c1lo = gs ? w1[u].z : w1[u].x, c1hi = gs ? w1[u].w : w1[u].y;
                    const uint32_t f0a = ik2m_e4m3x2(c0lo), f0b = ik2m_e4m3x2(c0lo >> 16);
                    const uint32_t f0c = ik2m_e4m3x2(c0hi), f0d = ik2m_e4m3x2(c0hi >> 16);
                    const uint32_t f1a = ik2m_e4m3x2(c1lo), f1b = ik2m_e4m3x2(c1lo >> 16);
                    const uint32_t f1c = ik2m_e4m3x2(c1hi), f1d = ik2m_e4m3x2(c1hi >> 16);
                    const float f00 = ik2m_ue8m0((sc[u].x >> (8 * gs)) & 0xFF), f01 = ik2m_ue8m0((sc[u].x >> (8 * (2 + gs))) & 0xFF);
                    const float f10 = ik2m_ue8m0((sc[u].y >> (8 * gs)) & 0xFF), f11 = ik2m_ue8m0((sc[u].y >> (8 * (2 + gs))) & 0xFF);
                    #pragma unroll
                    for (int g = 0; g < IK2V_NG; ++g) {
                        if (g < ng) {
                            const uint4 ax = ik2m_lda4(xa[g] + k + 32 * gs), bx = ik2m_lda4(xb[g] + k + 32 * gs);
                            float p0[4] = {0.f, 0.f, 0.f, 0.f}, p1[4] = {0.f, 0.f, 0.f, 0.f};
                            ik2m_mma(p0, ax.x, bx.x, ax.y, bx.y, f0a, f0b);
                            ik2m_mma(p0, ax.z, bx.z, ax.w, bx.w, f0c, f0d);
                            ik2m_mma(p1, ax.x, bx.x, ax.y, bx.y, f1a, f1b);
                            ik2m_mma(p1, ax.z, bx.z, ax.w, bx.w, f1c, f1d);
                            float (*a)[4] = acc[g];
                            a[0][0] = fmaf(p0[0], f00, a[0][0]); a[0][1] = fmaf(p0[1], f01, a[0][1]);
                            a[0][2] = fmaf(p0[2], f00, a[0][2]); a[0][3] = fmaf(p0[3], f01, a[0][3]);
                            a[1][0] = fmaf(p1[0], f10, a[1][0]); a[1][1] = fmaf(p1[1], f11, a[1][1]);
                            a[1][2] = fmaf(p1[2], f10, a[1][2]); a[1][3] = fmaf(p1[3], f11, a[1][3]);
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < IK2V_NG; ++g) {
        if (g < ng) {
            const int ra = 16 * g + gid, rb = ra + 8;
            ik2m_expert_store<GU>(acc[g], 1.0f, tile, tig, base + ra, base + rb, ra < c, rb < c, N, Din, Yf);
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void ink2_mx8_gu_vt(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    INK2_PDL_ENTRY();
    ik2v_mx8_body<true>(W, S, X, rowtok, Din, nullptr, cnt, offs, K, N);
}
extern "C" __global__ __launch_bounds__(128) void ink2_mx8_dn_vt(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ cnt, const int* __restrict__ offs, int K, int N) {
    INK2_PDL_ENTRY();
    ik2v_mx8_body<false>(W, S, Din, nullptr, nullptr, Y, cnt, offs, K, N);
}
