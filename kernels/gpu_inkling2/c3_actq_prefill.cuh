// gpu_inkling2/c3_actq_prefill.cuh — leg Q1: native block-scaled expert GEMMs at PREFILL widths, over K2's device
// routing (ink2_moe_pf_route: compact rows expert-major, 64-row tiles). CTA = 8 warps x one 16-row weight tile
// (128 weight rows) x 64 compact rows (8 n8 tiles). Activations (codes + block scales) are staged per 128-k block in
// smem (2 cp.async stages, padded rows: conflict-free B-fragment reads); each warp streams its own weight tile into
// registers one block ahead. Per-row arithmetic: the decode kernels' MMA kinds and c1's epilogues.
// GU: Xc/Xs/Xa = the quantised token rows, gathered by rowtok -> Act [R][N/2] bf16 (R = compact row).
// DN: Xc/Xs/Xa = the quantised Act rows [R] -> Y [R][N] fp32 (bf16-valued; combine with weight 1).
// grid (N/128, max_tiles), block 256. TR = 128: CTAs of odd 64-row tiles exit; the even one takes both.

__device__ __forceinline__ void ink2q_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
template <int B>
__device__ __forceinline__ void ink2q_cpsmall(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(s), "l"(gmem), "n"(B) : "memory");
}
__device__ __forceinline__ void ink2q_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void ink2q_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <bool GU, bool NV, int TR>
__device__ __forceinline__ void ink2q_pf_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs, const float* __restrict__ Xa,
        const int* __restrict__ rowmap, const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act, float* __restrict__ Y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N, float gcap, float ucap) {
    constexpr int RB = NV ? 64 : 128;      // code bytes per row per 128-k block
    constexpr int RS = RB + 16;            // padded smem row stride
    constexpr int SB = NV ? 8 : 4;         // scale bytes per row per block
    constexpr int WU = NV ? 2 : 4;         // MMA steps (= weight uint4 per lane) per block
    constexpr int NJ = TR / 8;             // n8 token tiles per CTA (TR = 64 or 128 compact rows)
    __shared__ __align__(16) uint8_t smX[2][TR * RS];
    __shared__ __align__(16) uint8_t smS[2][TR * SB];
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int e = tiles[2 * tile], r0 = tiles[2 * tile + 1];
    if (r0 % TR) return;                   // TR = 128: the CTA of an expert's even 64-row tile takes the pair
    const int rows = min(TR, cnt[e] - r0);
    const int base = offs[e] + r0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int t16 = blockIdx.x * 8 + warp;
    const int KB = K >> 7;
    const int rowbytes = NV ? (K >> 1) : K, srow = NV ? (K >> 4) : (K >> 5);
    const int NS = NV ? (K >> 6) : (K >> 5);
    const int NQ = (NS + 3) >> 2;
    const uint8_t* wt = W + (size_t)e * N * rowbytes + (size_t)t16 * NS * 512 + lane * 16;
    const uint8_t* stp = NV ? S + (size_t)e * N * (K >> 4) + (size_t)t16 * NS * 64 + (g + 8 * (t & 1)) * 4
                            : S + (size_t)e * (N >> 4) * NQ * 64 + (size_t)t16 * NQ * 64 + (g + 8 * (t & 1)) * 4;
    auto load = [&](int st, int b) {
        constexpr int CPR = RB / 16;
#pragma unroll
        for (int j = 0; j < TR * CPR / 256; ++j) {
            const int c = tid + 256 * j, row = c / CPR, ch = c % CPR;
            const int rr = base + min(row, rows - 1);
            const int src = rowmap ? rowmap[rr] : rr;
            ink2q_cp16(&smX[st][row * RS + ch * 16], Xc + (size_t)src * rowbytes + b * RB + ch * 16);
        }
        if (tid < TR) {
            const int rr = base + min(tid, rows - 1);
            const int src = rowmap ? rowmap[rr] : rr;
            const uint8_t* sp = Xs + (size_t)src * srow + b * SB;
            if constexpr (NV) {
                // srow = K/16 is 4-byte aligned for all K % 64 == 0 (not 8-byte aligned when K % 128 == 64)
                const uint32_t v0 = __ldg(reinterpret_cast<const uint32_t*>(sp));
                const uint32_t v1 = __ldg(reinterpret_cast<const uint32_t*>(sp + 4));
                *reinterpret_cast<uint32_t*>(&smS[st][tid * 8]) = v0;
                *reinterpret_cast<uint32_t*>(&smS[st][tid * 8 + 4]) = v1;
            } else {
                // srow = K/32 is 2-byte aligned for all K % 64 == 0 (not 4-byte aligned when K % 128 == 64)
                const uint32_t lo = (uint32_t)__ldg(reinterpret_cast<const uint16_t*>(sp));
                const uint32_t hi = (uint32_t)__ldg(reinterpret_cast<const uint16_t*>(sp + 2));
                *reinterpret_cast<uint32_t*>(&smS[st][tid * 4]) = lo | (hi << 16);
            }
        }
    };
    auto loadw = [&](int b, uint4 (&w)[WU], uint32_t (&sa)[2]) {
#pragma unroll
        for (int u = 0; u < WU; ++u) w[u] = ink2q_ldw4(wt + (size_t)(WU * b + u) * 512);
        if constexpr (NV) {
            sa[0] = __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)(2 * b) * 64));
            sa[1] = __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)(2 * b + 1) * 64));
        } else {
            sa[0] = __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)b * 64));
            sa[1] = 0u;
        }
    };
    float acc[NJ][4];
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
    uint4 w[WU], wn[WU];
    uint32_t sa[2], san[2];
    if (KB > 0) {
        load(0, 0);
        ink2q_cp_commit();
        loadw(0, w, sa);
    }
    for (int b = 0; b < KB; ++b) {
        const int st = b & 1;
        if (b + 1 < KB) { load(st ^ 1, b + 1); loadw(b + 1, wn, san); }
        ink2q_cp_commit();
        ink2q_cp_wait<1>();
        __syncthreads();
        if constexpr (NV) {
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int r = 8 * j + g;
                    const uint8_t* p = &smX[st][r * RS + 32 * s2 + 4 * t];
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
                    const uint32_t sb = *reinterpret_cast<const uint32_t*>(&smS[st][r * 8 + 4 * s2]);
                    ink2q_mma_nv4(acc[j], w[s2].x, w[s2].y, w[s2].z, w[s2].w, b0, b1, sa[s2], sb);
                }
            }
        } else {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int r = 8 * j + g;
                    const uint8_t* p = &smX[st][r * RS + 32 * u + 4 * t];
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
                    const uint32_t sb = *reinterpret_cast<const uint32_t*>(&smS[st][r * 4]);
                    ink2q_mma_mx8_v(u, acc[j], w[u].x, w[u].y, w[u].z, w[u].w, b0, b1, sa[0], sb);
                }
            }
        }
        if (b + 1 < KB) {
#pragma unroll
            for (int u = 0; u < WU; ++u) w[u] = wn[u];
            sa[0] = san[0]; sa[1] = san[1];
        }
        __syncthreads();
    }
    if constexpr (!GU) {
        if (K & 64) {
            constexpr int RBH = RB / 2;
            constexpr int CPRH = RBH / 16;
            constexpr int WUH = WU / 2;
            const int n_cp = TR * CPRH;
            for (int c = tid; c < n_cp; c += 256) {
                const int row = c / CPRH, ch = c % CPRH;
                const int rr = base + min(row, rows - 1);
                const int src = rowmap ? rowmap[rr] : rr;
                ink2q_cp16(&smX[0][row * RS + ch * 16], Xc + (size_t)src * rowbytes + KB * RB + ch * 16);
            }
            ink2q_cp_commit();
            if (tid < TR) {
                const int rr = base + min(tid, rows - 1);
                const int src = rowmap ? rowmap[rr] : rr;
                if constexpr (NV) {
                    *reinterpret_cast<uint32_t*>(&smS[0][tid * 8]) = __ldg(reinterpret_cast<const uint32_t*>(Xs + (size_t)src * srow + KB * 8));
                } else {
                    *reinterpret_cast<uint16_t*>(&smS[0][tid * 4]) = __ldg(reinterpret_cast<const uint16_t*>(Xs + (size_t)src * srow + KB * 4));
                }
            }
#pragma unroll
            for (int u = 0; u < WUH; ++u) w[u] = ink2q_ldw4(wt + (size_t)(WU * KB + u) * 512);
            sa[0] = NV ? __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)(2 * KB) * 64))
                       : __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)KB * 64));
            ink2q_cp_wait<0>();
            __syncthreads();
            if constexpr (NV) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int r = 8 * j + g;
                    const uint8_t* p = &smX[0][r * RS + 4 * t];
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
                    const uint32_t sb = *reinterpret_cast<const uint32_t*>(&smS[0][r * 8]);
                    ink2q_mma_nv4(acc[j], w[0].x, w[0].y, w[0].z, w[0].w, b0, b1, sa[0], sb);
                }
            } else {
#pragma unroll
                for (int u = 0; u < 2; ++u) {
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        const int r = 8 * j + g;
                        const uint8_t* p = &smX[0][r * RS + 32 * u + 4 * t];
                        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
                        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
                        const uint32_t sb = (uint32_t)(*reinterpret_cast<const uint16_t*>(&smS[0][r * 4]));
                        ink2q_mma_mx8_v(u, acc[j], w[u].x, w[u].y, w[u].z, w[u].w, b0, b1, sa[0], sb);
                    }
                }
            }
            __syncthreads();
        }
    }
    const float wam = NV ? GS[4 * e + (GU ? 0 : 2)] : 0.f, wden = NV ? GS[4 * e + (GU ? 1 : 3)] : 0.f;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int lr = 8 * j + 2 * t + q;
            if (lr >= rows) continue;
            const int R = base + lr;
            if constexpr (GU) {
                const int NI = N >> 1, ch = 8 * t16 + g;
                float v;
                if constexpr (NV) v = ink2q_gu_nv(acc[j][q], acc[j][2 + q], ink2q_row_scale(Xa[rowmap[R]], wam, wden), gam[R], gcap, ucap);
                else v = ink2q_gu_mx(acc[j][q], acc[j][2 + q], gam[R], gcap, ucap);
                Act[(size_t)R * NI + ch] = __float2bfloat16_rn(v);
            } else {
                if constexpr (NV) {
                    const float rs = ink2q_row_scale(Xa[R], wam, wden);
                    Y[(size_t)R * N + 16 * t16 + g] = ink2q_dn_nv(acc[j][q], rs);
                    Y[(size_t)R * N + 16 * t16 + 8 + g] = ink2q_dn_nv(acc[j][2 + q], rs);
                } else {
                    Y[(size_t)R * N + 16 * t16 + g] = ink2q_dn_mx(acc[j][q]);
                    Y[(size_t)R * N + 16 * t16 + 8 + g] = ink2q_dn_mx(acc[j][2 + q]);
                }
            }
        }
    }
}

#define INK2Q_PF(NAME, GU, NV, TR, MINB)                                                                               \
    extern "C" __global__ __launch_bounds__(256, MINB) void NAME(                                                       \
            const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,               \
            const uint8_t* __restrict__ Xc, const uint8_t* __restrict__ Xs, const float* __restrict__ Xa,             \
            const int* __restrict__ rowmap, const float* __restrict__ gam, __nv_bfloat16* __restrict__ Act,           \
            float* __restrict__ Y, const int* __restrict__ tiles, const int* __restrict__ ntiles,                     \
            const int* __restrict__ offs, const int* __restrict__ cnt, int K, int N, float gcap, float ucap) {        \
        INK2_PDL_ENTRY();                                                                                              \
        ink2q_pf_body<GU, NV, TR>(W, S, GS, Xc, Xs, Xa, rowmap, gam, Act, Y, tiles, ntiles, offs, cnt, K, N, gcap, ucap); \
    }
INK2Q_PF(ink2q_nv4_gu_pf, true, true, 64, 2)
INK2Q_PF(ink2q_nv4_dn_pf, false, true, 64, 2)
INK2Q_PF(ink2q_mx8_gu_pf, true, false, 64, 2)
INK2Q_PF(ink2q_mx8_dn_pf, false, false, 64, 2)
// 128 compact rows per CTA (an expert's 64-row tile pair): each weight fragment feeds 16 MMAs, weights re-read half as often
INK2Q_PF(ink2q_nv4_gu_pf128, true, true, 128, 1)
INK2Q_PF(ink2q_nv4_dn_pf128, false, true, 128, 1)
INK2Q_PF(ink2q_mx8_gu_pf128, true, false, 128, 1)
INK2Q_PF(ink2q_mx8_dn_pf128, false, false, 128, 1)
