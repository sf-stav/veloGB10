// gpu_inkling2/c2_actq_decode.cuh — leg Q1: native block-scaled expert GEMMs at decode / verify widths (M <= 16),
// the K2 union-of-experts schedule (62): every active slot streams its expert's weights once for all M tokens.
// Weights = MMA A (one 16-row tile per warp: w13 rows 0-7 gate / 8-15 up of 8 channels), tokens = MMA B (n8 tiles:
// tokens 0-7, 8-15). C: c0 (row g, tok 2t), c1 (row g, tok 2t+1), c2 (row g+8, tok 2t), c3 (row g+8, tok 2t+1).
// The tensor core accumulates across k (as the reference's CUTLASS / CuTe grouped GEMMs do); epilogues = c1's
// ink2q_gu_* / ink2q_dn_* (the reference's bf16 roundings and NVFP4 outer scales).
//
// GS [E][4] = {w13 amax, w13 denominator, w2 amax, w2 denominator} (NVFP4; MXFP8 ignores it).
// GU: Xc/Xs/Xa = the quantised token rows [M] -> Act [slot][M][N/2] bf16 (rows not routed to the slot: gamma 0).
// DN: Xc/Xs/Xa = the quantised Act rows [slot][M] -> Y [slot][M][N] fp32 (final bf16-valued rows; combined with
//     weight 1 by ink2_moe_combine: the routing weight is already inside Act).
// grid (N/64, emax), block 128 (4 warps x one 16-row tile).

template <bool GU>
__device__ __forceinline__ void ink2q_nv4_dec_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs, const float* __restrict__ Xa,
        const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act, float* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, float gcap, float ucap) {
    constexpr int U = 8;
    const int slot = blockIdx.y;
    if (slot >= *esel) return;
    const int e = idxmap[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int NS = K >> 6;
    const uint8_t* wt = W + (size_t)e * N * (K >> 1) + (size_t)tile * NS * 512 + lane * 16;
    const uint8_t* st = S + (size_t)e * N * (K >> 4) + (size_t)tile * NS * 64 + (g + 8 * (t & 1)) * 4;
    const int xr0 = GU ? 0 : slot * M;
    const int tk0 = min(g, M - 1), tk1 = min(8 + g, M - 1);
    const uint8_t* xc0 = Xc + (size_t)(xr0 + tk0) * (K >> 1) + 4 * t;
    const uint8_t* xc1 = Xc + (size_t)(xr0 + tk1) * (K >> 1) + 4 * t;
    const uint8_t* xs0 = Xs + (size_t)(xr0 + tk0) * (K >> 4);
    const uint8_t* xs1 = Xs + (size_t)(xr0 + tk1) * (K >> 4);
    const bool two = M > 8;
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
    for (int s0 = 0; s0 < NS; s0 += U) {
        uint4 w[U];
        uint32_t sa[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (s0 + u < NS) {
                w[u] = ink2q_ldw4(wt + (size_t)(s0 + u) * 512);
                sa[u] = __ldg(reinterpret_cast<const uint32_t*>(st + (size_t)(s0 + u) * 64));
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = s0 + u;
            if (s < NS) {
                const uint32_t b0 = __ldg(reinterpret_cast<const uint32_t*>(xc0 + 32 * s));
                const uint32_t b1 = __ldg(reinterpret_cast<const uint32_t*>(xc0 + 32 * s + 16));
                const uint32_t sb = __ldg(reinterpret_cast<const uint32_t*>(xs0 + 4 * s));
                ink2q_mma_nv4(acc[0], w[u].x, w[u].y, w[u].z, w[u].w, b0, b1, sa[u], sb);
                if (two) {
                    const uint32_t c0 = __ldg(reinterpret_cast<const uint32_t*>(xc1 + 32 * s));
                    const uint32_t c1 = __ldg(reinterpret_cast<const uint32_t*>(xc1 + 32 * s + 16));
                    const uint32_t sc = __ldg(reinterpret_cast<const uint32_t*>(xs1 + 4 * s));
                    ink2q_mma_nv4(acc[1], w[u].x, w[u].y, w[u].z, w[u].w, c0, c1, sa[u], sc);
                }
            }
        }
    }
    const float wam = GS[4 * e + (GU ? 0 : 2)], wden = GS[4 * e + (GU ? 1 : 3)];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int n = 8 * j + 2 * t + q;
            if (n >= M) continue;
            const int row = slot * M + n;
            if constexpr (GU) {
                const int NI = N >> 1, ch = 8 * tile + g;
                const float sc = ink2q_row_scale(Xa[n], wam, wden);
                Act[(size_t)row * NI + ch] = __float2bfloat16_rn(ink2q_gu_nv(acc[j][q], acc[j][2 + q], sc, gam[row], gcap, ucap));
            } else {
                const float rs = ink2q_row_scale(Xa[row], wam, wden);
                Y[(size_t)row * N + 16 * tile + g] = ink2q_dn_nv(acc[j][q], rs);
                Y[(size_t)row * N + 16 * tile + 8 + g] = ink2q_dn_nv(acc[j][2 + q], rs);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void ink2q_nv4_gu_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs, const float* __restrict__ Xa,
        const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, float gcap, float ucap) {
    INK2_PDL_ENTRY();
    ink2q_nv4_dec_body<true>(W, S, GS, Xc, Xs, Xa, gam, Act, nullptr, idxmap, esel, M, K, N, gcap, ucap);
}
extern "C" __global__ __launch_bounds__(128) void ink2q_nv4_dn_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs, const float* __restrict__ Xa,
        float* __restrict__ Y, const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ink2q_nv4_dec_body<false>(W, S, GS, Xc, Xs, Xa, nullptr, nullptr, Y, idxmap, esel, M, K, N, 0.f, 0.f);
}

// ---- MXFP8: 32-k MMA steps, scales one byte per (row, step), 4 steps per u32 (byte = step % 4) ----
// Safe for all K % 64 == 0 (when K % 128 == 64, row stride K/32 is 2-byte aligned, not 4-byte aligned).
__device__ __forceinline__ uint32_t ink2q_ld_mx8_sb(const uint8_t* p, int q, int ns) {
    const uint32_t lo = (uint32_t)__ldg(reinterpret_cast<const uint16_t*>(p + 4 * q));
    if (4 * q + 4 <= ns) {
        const uint32_t hi = (uint32_t)__ldg(reinterpret_cast<const uint16_t*>(p + 4 * q + 2));
        return lo | (hi << 16);
    }
    return lo;
}
template <bool GU>
__device__ __forceinline__ void ink2q_mx8_dec_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs,
        const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act, float* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, float gcap, float ucap) {
    const int slot = blockIdx.y;
    if (slot >= *esel) return;
    const int e = idxmap[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int NS = K >> 5, NQ = (NS + 3) >> 2;
    const uint8_t* wt = W + (size_t)e * N * K + (size_t)tile * NS * 512 + lane * 16;
    const uint8_t* st = S + (size_t)e * (N >> 4) * NQ * 64 + (size_t)tile * NQ * 64 + (g + 8 * (t & 1)) * 4;
    const int xr0 = GU ? 0 : slot * M;
    const int tk0 = min(g, M - 1), tk1 = min(8 + g, M - 1);
    const uint8_t* xc0 = Xc + (size_t)(xr0 + tk0) * K + 4 * t;
    const uint8_t* xc1 = Xc + (size_t)(xr0 + tk1) * K + 4 * t;
    const uint8_t* xs0 = Xs + (size_t)(xr0 + tk0) * (K >> 5);
    const uint8_t* xs1 = Xs + (size_t)(xr0 + tk1) * (K >> 5);
    const bool two = M > 8;
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
    for (int q0 = 0; q0 < NQ; q0 += 2) {
        uint4 w[8];
        uint32_t sa[2];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
            if (q0 + qq < NQ) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (4 * (q0 + qq) + u < NS) w[4 * qq + u] = ink2q_ldw4(wt + (size_t)(4 * (q0 + qq) + u) * 512);
                }
                sa[qq] = __ldg(reinterpret_cast<const uint32_t*>(st + (size_t)(q0 + qq) * 64));
            }
        }
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
            const int q = q0 + qq;
            if (q < NQ) {
                const uint32_t sb0 = ink2q_ld_mx8_sb(xs0, q, NS);
                const uint32_t sb1 = two ? ink2q_ld_mx8_sb(xs1, q, NS) : 0u;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int s = 4 * q + u;
                    if (s >= NS) break;
                    const uint4 a = w[4 * qq + u];
                    const uint32_t b0 = __ldg(reinterpret_cast<const uint32_t*>(xc0 + 32 * s));
                    const uint32_t b1 = __ldg(reinterpret_cast<const uint32_t*>(xc0 + 32 * s + 16));
                    ink2q_mma_mx8_v(u, acc[0], a.x, a.y, a.z, a.w, b0, b1, sa[qq], sb0);
                    if (two) {
                        const uint32_t c0 = __ldg(reinterpret_cast<const uint32_t*>(xc1 + 32 * s));
                        const uint32_t c1 = __ldg(reinterpret_cast<const uint32_t*>(xc1 + 32 * s + 16));
                        ink2q_mma_mx8_v(u, acc[1], a.x, a.y, a.z, a.w, c0, c1, sa[qq], sb1);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int n = 8 * j + 2 * t + q;
            if (n >= M) continue;
            const int row = slot * M + n;
            if constexpr (GU) {
                const int NI = N >> 1, ch = 8 * tile + g;
                Act[(size_t)row * NI + ch] = __float2bfloat16_rn(ink2q_gu_mx(acc[j][q], acc[j][2 + q], gam[row], gcap, ucap));
            } else {
                Y[(size_t)row * N + 16 * tile + g] = ink2q_dn_mx(acc[j][q]);
                Y[(size_t)row * N + 16 * tile + 8 + g] = ink2q_dn_mx(acc[j][2 + q]);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void ink2q_mx8_gu_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs,
        const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, float gcap, float ucap) {
    INK2_PDL_ENTRY();
    ink2q_mx8_dec_body<true>(W, S, Xc, Xs, gam, Act, nullptr, idxmap, esel, M, K, N, gcap, ucap);
}
extern "C" __global__ __launch_bounds__(128) void ink2q_mx8_dn_dec(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs,
        float* __restrict__ Y, const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N) {
    INK2_PDL_ENTRY();
    ink2q_mx8_dec_body<false>(W, S, Xc, Xs, nullptr, nullptr, Y, idxmap, esel, M, K, N, 0.f, 0.f);
}
