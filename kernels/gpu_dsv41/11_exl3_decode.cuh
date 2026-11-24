// 11_exl3_decode.cuh (leg KE): decode + Hadamard + the weight-stationary tile loop.
// Math mirrors src/exl3.rs decode_block (G-A3-1 green): mul1 codebook, funnel-shift window, tmap placement.
// WIDTH CODE `bits`: 2..8 = integer K; 17/18/19 = fractional K = 1.5 / 2.5 / 3.5 (16 + KA).
#pragma once

#define D41X_MUL1 0x83DCD12Du
#define D41X_FRAC(KA) (16 + (KA))
#define D41X_KBLK 8           // k16 steps per MMA accumulation chain (KE2; see d41x_tiles)

__device__ __forceinline__ uint16_t d41x_dq_from(const uint32_t* __restrict__ ring, int i0m, int i1m, int sf) {
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * D41X_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum);
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931));
    return (uint16_t)__half_as_ushort(w16);
}
__device__ __forceinline__ unsigned d41x_dq_idx(unsigned lo, unsigned hi, int sf) {
    return __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
}
__device__ __forceinline__ uint16_t d41x_dq_val(unsigned idx) {
    const unsigned sum = __dp4a(idx * D41X_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum);
    return (uint16_t)__half_as_ushort(__hfma(h, __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
}

// KA = bits (integer) or KA of a fractional KA + 0.5; H = fractional.
template <int KA, bool H>
struct D41xW {
    static constexpr int WR = H ? 4 * (2 * KA + 1) : 8 * KA;   // ring u32 words per 16x16 tile
    __device__ static __forceinline__ int b1(int t) {
        if (H) {
            const int g = t & ~7, j = t & 7, bits2 = 2 * KA + 1;
            return ((g >> 1) + 4) * bits2 + 128 * bits2 - (((7 - j) * KA) + ((14 - j) >> 1) - 3);
        }
        return (t + 257) * KA;
    }
};

// per-lane decode slots: slot = h*4 + s (h: n8 half, s: B-fragment register slot)
struct D41xSlots { int i0[8]; int i1[8]; int sf[8]; };
template <int KA, bool H>
__device__ __forceinline__ void d41x_slots(int lane, D41xSlots& S) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = D41xW<KA, H>::b1(t);
            const int i0 = (b1 - 16) >> 5, i1 = (b1 - 1) >> 5;
            S.i0[h * 4 + s] = i0 % D41xW<KA, H>::WR;
            S.i1[h * 4 + s] = i1 % D41xW<KA, H>::WR;
            S.sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
}

// Unnormalised-then-orthonormal Walsh-Hadamard over 128 values, 4 per lane, fixed butterfly order.
// KE2: every operation is an explicitly rounded intrinsic (.rn PTX), so ptxas can never contract a caller's product
// into the first butterfly stage: every kernel that uses it (decode, prefill, generic, dense) gets the same bits.
__device__ __forceinline__ void d41x_had128(float v[4], int lane) {
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int len = 1 << s;
        #pragma unroll
        for (int r = 0; r < 4; ++r) if ((r & len) == 0) { const float a = v[r], b = v[r + len]; v[r] = __fadd_rn(a, b); v[r + len] = __fsub_rn(a, b); }
    }
    #pragma unroll
    for (int s = 2; s < 7; ++s) {
        const unsigned mask = 1u << (s - 2);
        #pragma unroll
        for (int r = 0; r < 4; ++r) { const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask); v[r] = (lane & mask) ? __fsub_rn(t, v[r]) : __fadd_rn(v[r], t); }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __fmul_rn(v[r], 0.08838834764831845f);
}

// KE2: the MoE / dense per-element math, ONE definition shared by every kernel (generic, decode, prefill), each
// rounding pinned with an intrinsic.
// prep: fp16(H128(x * suh)) for one 128-block, 4 values per lane (x already loaded as f32).
__device__ __forceinline__ void d41x_prep4(float v[4], const float s[4], int lane) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __fmul_rn(v[r], s[r]);
    d41x_had128(v, lane);
}
// SwiGLU with the reference clamps, the routing weight and the bf16 rounding point, then w2's input scale:
// returns bf16(w * silu(min(g, L)) * clamp(u, -L, L)) * suh2. g / u are after their output H128 and svh.
__device__ __forceinline__ float d41x_act1(float gg, float uu, float w, float suh2, float limit) {
    if (limit > 0.f) { uu = fminf(fmaxf(uu, -limit), limit); gg = fminf(gg, limit); }
    const float sg = __fdiv_rn(gg, __fadd_rn(1.f, __expf(-gg)));
    const float act = __fmul_rn(__fmul_rn(sg, uu), w);
    return __fmul_rn(__bfloat162float(__float2bfloat16_rn(act)), suh2);
}

__device__ __forceinline__ float d41x_ld(const void* p, int code, size_t i) {   // 0 bf16, 1 f16, 2 f32
    if (code == 0) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    if (code == 1) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}
__device__ __forceinline__ void d41x_st(void* p, int code, size_t i, float v) {
    if (code == 0) ((__nv_bfloat16*)p)[i] = __float2bfloat16_rn(v);
    else if (code == 1) ((__half*)p)[i] = __float2half_rn(v);
    else ((float*)p)[i] = v;
}

// The tile loop. One warp owns one 16-column n-tile and a k16 range; MT m16 tiles share each decode.
//  ring0: u32 address of tile (first kb, this n-tile); kstride: u32 words between successive kb (nb_stride * WR)
//  A: fp16 activations (row 0, column of the first kb), row stride lda; rows [mrow0, rows) live, MT tiles of 16.
//  acc[mt][h][4]: m16n8k16 C fragment (rows gid, gid+8; cols 2*tig, 2*tig+1 of n8 half h).
// K order is strictly ascending kb, one mma per (mt, h) per step: independent of MT and of the row count,
// so row i of an N-row call is bit-identical to the 1-row call for the same (nkb, split) partition.
template <int KA, bool H, int U>
__device__ __forceinline__ void d41x_dec_group(const uint32_t* __restrict__ ring, size_t kstride, const D41xSlots& S,
                                               unsigned (&bf)[U][2][2]) {
    unsigned lo[U][8], hi[U][8];
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int q = 0; q < 8; ++q) { lo[u][q] = ring[u * kstride + S.i1[q]]; hi[u][q] = ring[u * kstride + S.i0[q]]; }
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned r0 = d41x_dq_val(d41x_dq_idx(lo[u][h*4+0], hi[u][h*4+0], S.sf[h*4+0]));
            const unsigned r1 = d41x_dq_val(d41x_dq_idx(lo[u][h*4+1], hi[u][h*4+1], S.sf[h*4+1]));
            const unsigned r2 = d41x_dq_val(d41x_dq_idx(lo[u][h*4+2], hi[u][h*4+2], S.sf[h*4+2]));
            const unsigned r3 = d41x_dq_val(d41x_dq_idx(lo[u][h*4+3], hi[u][h*4+3], S.sf[h*4+3]));
            bf[u][h][0] = r0 | (r1 << 16);
            bf[u][h][1] = r2 | (r3 << 16);
        }
}

template <int MT, int U>
__device__ __forceinline__ void d41x_mma_group(const __half* const (&a_p)[MT][2], const bool (&a_ok)[MT][2], int koff,
                                               const unsigned (&bf)[U][2][2], float (&acc)[MT][2][4]) {
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            const __half* p0 = a_p[m][0] + koff + 16 * u;
            const __half* p1 = a_p[m][1] + koff + 16 * u;
            const uint32_t a0 = a_ok[m][0] ? *(const uint32_t*)p0 : 0u;
            const uint32_t a1 = a_ok[m][1] ? *(const uint32_t*)p1 : 0u;
            const uint32_t a2 = a_ok[m][0] ? *(const uint32_t*)(p0 + 8) : 0u;
            const uint32_t a3 = a_ok[m][1] ? *(const uint32_t*)(p1 + 8) : 0u;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][h][0]), "+f"(acc[m][h][1]), "+f"(acc[m][h][2]), "+f"(acc[m][h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf[u][h][0]), "r"(bf[u][h][1]));
        }
}

template <int KA, bool H, int MT>
__device__ __forceinline__ void d41x_tiles(const uint32_t* __restrict__ ring0, size_t kstride, int nkb,
                                           const __half* __restrict__ A, int lda, int rows, int mrow0,
                                           const D41xSlots& S, int lane, float (&acc)[MT][2][4]) {
    const int gid = lane >> 2, tig = lane & 3;
    const __half* a_p[MT][2];
    bool a_ok[MT][2];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int row = mrow0 + 16 * m + gid + 8 * q;
            a_ok[m][q] = row < rows;
            a_p[m][q] = A + (size_t)(a_ok[m][q] ? row : 0) * lda + 2 * tig;
        }
    constexpr int U = MT <= 2 ? 4 : (MT == 4 ? 2 : 1);
    // KE2: the GB10 f16 MMA accumulator loses precision over long chains (6e-6 rel-L2 over 320 k16 steps, max element
    // 7.5e-3; an 8-step chain + an f32 add: 2.2e-7). So: ONE chain per k-block of D41X_KBLK k16 steps (counted from the
    // range start, shape only), each k-block's partial added into `acc` in f32 (__fadd_rn), ascending.
    for (int kb0 = 0; kb0 < nkb; kb0 += D41X_KBLK) {
        const int kb1 = min(kb0 + D41X_KBLK, nkb);
        float blk[MT][2][4];
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) blk[m][h][r] = 0.f;
        int kb = kb0;
        for (; kb + U <= kb1; kb += U) {
            unsigned bf[U][2][2];
            d41x_dec_group<KA, H, U>(ring0 + (size_t)kb * kstride, kstride, S, bf);
            d41x_mma_group<MT, U>(a_p, a_ok, kb * 16, bf, blk);
        }
        for (; kb < kb1; ++kb) {
            unsigned bf[1][2][2];
            d41x_dec_group<KA, H, 1>(ring0 + (size_t)kb * kstride, kstride, S, bf);
            d41x_mma_group<MT, 1>(a_p, a_ok, kb * 16, bf, blk);
        }
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[m][h][r] = __fadd_rn(acc[m][h][r], blk[m][h][r]);
    }
}
