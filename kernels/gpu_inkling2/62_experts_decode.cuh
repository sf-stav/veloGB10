// gpu_inkling2/62_experts_decode.cuh: routed experts at decode / verify widths (M <= 16 tokens), the union-of-experts
// schedule (the fp4l xf_nv4_moe shape): every ACTIVE expert slot (ink2_router_topk's idxmap / esel) computes all M
// token rows once — the weight bytes of each selected expert are streamed exactly once per step whatever M is, so the
// cost grows with the expert union, not with M x 8.
//   GU: W13 (tile-interleaved gate|up, see 60) x Xh [M][K] -> softcapped SwiGLU -> Din [slot][M][N/2] f16.
//   DN: W2 x Din[slot] -> Y [slot][M][N] fp32 = acc * gs (unweighted; ink2_moe_combine applies the route weights).
// grid (N/64, emax), block 128 (4 warps x one 16-row tile). emax = min(8M, E) (slots >= *esel exit at once).
// Batch invariance: a token row is one MMA A row with a fixed k sequence; rows >= M read a clamped valid row and are
// never stored.
// GS [E][2] fp32: {w13 multiplier, w2 multiplier} (NVFP4: amax[e] / outer_scale_denominator; MXFP8 passes nullptr).

// ---- NVFP4 (F128) ----
template <bool GU>
__device__ __forceinline__ void ik2m_nv4_dec_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, long long x_slot_stride, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    constexpr int U = 4;
    const int slot = blockIdx.y;
    if (slot >= *esel) return;
    const int e = idxmap[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int KB = K >> 7;
    const size_t tw_stride = (size_t)KB * 1024 + ((K & 64) ? 512 : 0);
    const size_t ts_stride = (size_t)KB * 128 + ((K & 64) ? 64 : 0);
    const uint8_t* we = W + (size_t)e * N * (K >> 1) + (size_t)tile * tw_stride;
    const uint8_t* se = S + (size_t)e * N * (K >> 4) + (size_t)tile * ts_stride;
    const uint8_t* wt = we + lane * 16;
    const uint8_t* st = se + lane * 4;
    const __half* Xs = X + (size_t)slot * x_slot_stride;
    const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
    const __half* xa = Xs + (size_t)ta * K + 16 * tig;
    const __half* xb = Xs + (size_t)tb * K + 16 * tig;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
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
                ik2m_nv4_step(acc, w0[u].x, w0[u].y, w1[u].x, w1[u].y, sc[u] & 0xFF, (sc[u] >> 16) & 0xFF,
                              ik2m_lda4(xa + k), ik2m_lda4(xa + k + 8), ik2m_lda4(xb + k), ik2m_lda4(xb + k + 8));
                ik2m_nv4_step(acc, w0[u].z, w0[u].w, w1[u].z, w1[u].w, (sc[u] >> 8) & 0xFF, sc[u] >> 24,
                              ik2m_lda4(xa + k + 64), ik2m_lda4(xa + k + 72), ik2m_lda4(xb + k + 64), ik2m_lda4(xb + k + 72));
            }
        }
    }
    if constexpr (!GU) {
        if (K & 64) {
            const uint2 w0h = __ldg(reinterpret_cast<const uint2*>(we + (size_t)KB * 1024 + lane * 8));
            const uint2 w1h = __ldg(reinterpret_cast<const uint2*>(we + (size_t)KB * 1024 + 256 + lane * 8));
            const uint16_t sch = __ldg(reinterpret_cast<const uint16_t*>(se + (size_t)KB * 128 + lane * 2));
            const int k = KB * 128;
            ik2m_nv4_step(acc, w0h.x, w0h.y, w1h.x, w1h.y, sch & 0xFF, sch >> 8,
                          ik2m_lda4(xa + k), ik2m_lda4(xa + k + 8), ik2m_lda4(xb + k), ik2m_lda4(xb + k + 8));
        }
    }
    const float gs = GS[2 * e + (GU ? 0 : 1)];
    const int row0 = slot * M;
    ik2m_expert_store<GU>(acc, gs, tile, tig, row0 + gid, row0 + gid + 8, gid < M, gid + 8 < M, N, Din, Yf);
}
extern "C" __global__ __launch_bounds__(128) void ink2_nv4_gu_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, __half* __restrict__ Din,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_nv4_dec_body<true>(W, S, GS, X, 0, Din, nullptr, idxmap, esel, M, K, N);
}
extern "C" __global__ __launch_bounds__(128) void ink2_nv4_dn_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_nv4_dec_body<false>(W, S, GS, Din, (long long)M * K, nullptr, Y, idxmap, esel, M, K, N);
}

// ---- MXFP8 (F64) ----
template <bool GU>
__device__ __forceinline__ void ik2m_mx8_dec_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, long long x_slot_stride, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    constexpr int U = 4;
    const int slot = blockIdx.y;
    if (slot >= *esel) return;
    const int e = idxmap[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int KB = K >> 6;
    const uint8_t* wt = W + (size_t)e * N * K + (size_t)tile * KB * 1024 + lane * 16;
    const uint8_t* st = S + (size_t)e * N * (K >> 5) + (size_t)tile * KB * 32 + tig * 8;
    const __half* Xs = X + (size_t)slot * x_slot_stride;
    const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
    const __half* xa = Xs + (size_t)ta * K + 8 * tig;
    const __half* xb = Xs + (size_t)tb * K + 8 * tig;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
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
                ik2m_mx8_block(acc, w0[u], w1[u], sc[u], ik2m_lda4(xa + k), ik2m_lda4(xa + k + 32),
                               ik2m_lda4(xb + k), ik2m_lda4(xb + k + 32));
            }
        }
    }
    const int row0 = slot * M;
    ik2m_expert_store<GU>(acc, 1.0f, tile, tig, row0 + gid, row0 + gid + 8, gid < M, gid + 8 < M, N, Din, Yf);
}
extern "C" __global__ __launch_bounds__(128) void ink2_mx8_gu_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, __half* __restrict__ Din,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_mx8_dec_body<true>(W, S, X, 0, Din, nullptr, idxmap, esel, M, K, N);
}
extern "C" __global__ __launch_bounds__(128) void ink2_mx8_dn_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_mx8_dec_body<false>(W, S, Din, (long long)M * K, nullptr, Y, idxmap, esel, M, K, N);
}
