// exl3_bench/07_moe_wp20_epilogues.cuh: A5 WP20 expert word diet and MoE glue as last-arrival epilogues.
// Target is plain sm_121; nothing here needs an f/a feature.
// Section 7 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ===========================================================================
// A5 WP20 (PLAN/SURPASS_PLAN_2026-09-26.md): expert WORD DIET + MoE glue as
// LAST-ARRIVAL EPILOGUES (the reference implementation's exl3_moe_coop structure on our a1b3 body).
// Target: plain sm_121 — nothing here needs an f/a feature (fp16 mma.m16n8k16, dp4a,
// 64-bit funnel shifts, device-scope atomics/fences are baseline CC 12.1).
//
// Rung 1 (V1) — the decode's word supply. A lane's 8 B-fragment slots j = 4h+q are
// 8 CONSECUTIVE trellis indices t0+j (tmap(r,c) with r = 2(l%4)+(q&1)+8(q>>1),
// c = 8h+l/4 reduces to t = t0(l) + 4h + q). Their 16-bit windows [b1-16, b1),
// b1 = (t+257)*BITS, span 16 + 7*BITS bits starting at bit 24g+755 (b3; g = t0/8),
// i.e. start%32 in {3,11,19,27} -> ALWAYS inside the two ring words A = start>>5 and
// A+1 (mod W = 8*BITS). So each lane loads those 2 words ONCE per k16 step and
// slices every window from the 64-bit pair Q = (w[A] << 32) | w[A+1]:
//   idx_j = (Q >> (32(A+2) - b1_j)) & 0xFFFF
// which is exactly the old funnelshift_r(w[i1], w[i0], 32(i1+1)-b1) & 0xFFFF for
// every (i0, i1) in {(A,A), (A,A+1), (A+1,A+1)} — the SAME stream bits, so the same
// idx, the same dp4a/hfma, the same fragments, the same mma chain: BITWISE. It
// replaces the 16 SHFL per lane per step of the A1_SHFL block (1 LDG + 16 SHFL ->
// 2 LDG, all MIO). (Index math checked for every lane/slot at b3 and b4 against the
// old decode on random rings; the binv probe checks the entry against the barrier body.)
//
// Rung 2 (V9) — the MoE glue moves into the two expert GEMMs' epilogues:
//  gate/up (xq_moe_gu_epi): every CTA applies ITS tile's svh (== xq_had_svh_multi),
//    publishes the f16 tile to ygu, and arrives on a per-(expert, tile mod P) counter;
//    the SECOND arrival of the (gate p, up p+P) pair does silu(g)*u (== xq_moe_gate_mul)
//    and the down input transform (== xq_had_suh_multi, per-expert input) for its
//    128-col block -> xhd. Expert 0's CTAs also compute the shared-expert gate
//    sigmoid per row (== the first half of xq_moe_combine, its 1024-thread tree
//    emulated exactly) -> sgv[m].
//  down (xq_moe_dn_epi): every CTA applies its svh (== xq_had_svh_multi) -> yd and
//    arrives on a per-N-tile counter against the device esel; the LAST arrival computes
//    moe_out = f16(sgv*ysh + sum_j wts*yd) for the tile in the exact j order
//    (== xq_moe_combine; the shared expert now runs BEFORE the routed down).
//  Every rounding point is the old kernels': same f16 stores/reloads, the same
//  butterfly, and every rounding pinned with intrinsics:
//   - combine: the old PTX's own explicit mul.f32 then fma.rn chain; sigmoid dot:
//     fma.rn partials + add tree (NVVM emitted those roundings explicitly).
//   - svh / silu*mul: no product feeds an add (mul -> cvt), nothing to contract.
//   - suh (x*s products feeding the FIRST butterfly stage): the old PTX leaves
//     mul.f32/add.f32 unrounded and ptxas contracts them context-dependently. WP20-v1c:
//     the epilogue's suh is PINNED op by op to xq_had_suh_multi's served SASS (FMUL
//     q = x1*s1; FFMA x0*s0 + q; FFMA x0*s0 - q; FADD / FSEL+FADD butterfly; FMUL by
//     1/sqrt(128)) with explicit .rn intrinsics — see wp20_suh_had. (v1/v1b's runtime
//     81-convention calibration could never succeed: x and s are f16 values, so x*s is
//     EXACT in f32 and every contraction gives the same bits — all 81 matched, "no
//     unique match", rung 2 OFF.) The boot check (xq_wp20_suh_cal) now only VERIFIES
//     the pinned sequence against the JIT'd xq_had_suh_multi, else the old path runs.
//  Counters self re-arm (the last arrival zeroes its counter; ONE real zero at
//  allocation) — graph-replay safe, no memset node.
//  Top-k (WP20-v1b): the down epilogue's combine is a compile-time TOPK template (the
//  j loops unroll into registers — AGENTS §4: no runtime-indexed per-thread arrays), one
//  entry per served top-k: xq_moe_dn_epi[_sh]_k8 and _k10 (Qwen3.8-Flash-Next = top-10).
//  The launcher (FwdModel::wp20_select) picks the instance from config and falls back
//  LOUDLY for any other top-k; the kernel traps on a top-k/instance mismatch.
// ===========================================================================

// WP20-v1c: == xq_had_suh_multi's per-lane body, PINNED op by op to its SERVED SASS:
// v = H128(x * s), 4 per lane, unrounded f32 out (the callers round with xq_f2h, the old
// kernel's F2FP.F16.F32 RN). The sequence is read off `ptxas -arch=sm_121` of the shipped
// PTX (the driver JIT's SASS; cuobjdump -sass -fun xq_had_suh_multi):
//   stage len 1  : FMUL q1 = x1*s1 ; FFMA a0 = x0*s0 + q1 ; FFMA a1 = x0*s0 - q1  (pair 2,3 alike)
//   stage len 2  : FADD a0 + a2 ; FADD a1 + a3 ; FADD a0 - a2 ; FADD a1 - a3
//   len 4 .. 64  : SHFL.BFLY t ; FSEL w = (lane & mask) ? -v : v ; FADD t + w
//   normalize    : FMUL v * 0x3db504f3 (1/sqrt(128) in f32)
// Every op is an explicit .rn intrinsic (fma.rn / mul.rn / add.rn / sub.rn in the PTX), which
// ptxas may neither contract nor split, so every inlining context (xq_moe_gu_epi[_sh],
// xq_wp20_suh_cal) executes exactly this sequence — no runtime convention, nothing to
// calibrate. It is ALSO value-exact against any contraction ptxas could pick for the old
// source: x and s are f16 values, so x*s is exact in f32 (11 + 11 significand bits <= 24,
// |x*s| in [2^-48, 2^32): no rounding, no underflow, no overflow), hence
// fma(x0,s0,q) == rn(rn(x0*s0) + q) == fma(-x1,s1,rn(x0*s0)) bit for bit (incl. signed zeros).
// That exactness is also why v1/v1b's 81-convention calibration always found 81 matches and
// never a "unique" one (the WP20 FALLBACK seen at boot on preview p2).
__device__ __forceinline__ void wp20_suh_had(const float x[4], const float s[4], float v[4], int lane) {
    {
        const float q1 = __fmul_rn(x[1], s[1]);
        const float q3 = __fmul_rn(x[3], s[3]);
        const float a0 = __fmaf_rn(x[0], s[0], q1);
        const float a1 = __fmaf_rn(x[0], s[0], -q1);
        const float a2 = __fmaf_rn(x[2], s[2], q3);
        const float a3 = __fmaf_rn(x[2], s[2], -q3);
        v[0] = __fadd_rn(a0, a2);                 // exl3_had128 stage len = 2
        v[1] = __fadd_rn(a1, a3);
        v[2] = __fsub_rn(a0, a2);
        v[3] = __fsub_rn(a1, a3);
    }
    #pragma unroll
    for (int s2 = 2; s2 < 7; ++s2) {              // exl3_had128 cross-lane stages
        const unsigned mask = (unsigned)(1u << (s2 - 2));
        const bool hi = (lane & mask) != 0;       // hi: t - v, else v + t (== t + v)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask);
            v[r] = __fadd_rn(t, hi ? -v[r] : v[r]);
        }
    }
    const float hnorm = 0.08838834764831845f;     // 1/sqrt(128), exl3_had128's
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __fmul_rn(v[r], hnorm);
}

// Boot VERIFICATION of the pinned sequence (FwdModel::load via exl3_bench::wp20_calibrate):
// xq_had_suh_multi's exact signature, indexing, loads and f16 store, with its arithmetic
// replaced by wp20_suh_had. The host launches both on the same inputs and requires every
// output byte equal; any mismatch keeps rung 2 off (loud WP20 FALLBACK).
extern "C" __global__ void xq_wp20_suh_cal(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                           const int* __restrict__ idxmap,
                                           __half* __restrict__ xh, int M, int K,
                                           const int* __restrict__ esel,
                                           int per_expert_in) {
    int nblk = K >> 7;
    int blk = blockIdx.x % nblk;
    long long em = blockIdx.x / nblk;
    int m = (int)(em % M);
    int e = (int)(em / M);
    if (e >= *esel) return;
    int lane = threadIdx.x & 31;
    long long base_in = (long long)(per_expert_in ? e : 0) * M * K + (long long)m * K
                      + (long long)blk * 128 + lane * 4;
    long long base_out = ((long long)e * M + m) * K + (long long)blk * 128 + lane * 4;
    const __half* sr = suh_all + (long long)idxmap[e] * K + (long long)blk * 128 + lane * 4;
    float xs[4], ss[4], v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) { xs[r] = xq_h2f(x[base_in + r]); ss[r] = xq_h2f(sr[r]); }
    wp20_suh_had(xs, ss, v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = xq_f2h(v[r]);
}

// 4 consecutive f16 (element j in bits [16j, 16j+16) of the pair) <-> one 8-B access.
__device__ __forceinline__ void wp20_unpack4(uint2 w, float v[4]) {
    v[0] = xq_h2f(__ushort_as_half((unsigned short)(w.x & 0xFFFFu)));
    v[1] = xq_h2f(__ushort_as_half((unsigned short)(w.x >> 16)));
    v[2] = xq_h2f(__ushort_as_half((unsigned short)(w.y & 0xFFFFu)));
    v[3] = xq_h2f(__ushort_as_half((unsigned short)(w.y >> 16)));
}
__device__ __forceinline__ uint2 wp20_pack4(const __half o[4]) {
    return make_uint2((unsigned)__half_as_ushort(o[0]) | ((unsigned)__half_as_ushort(o[1]) << 16),
                      (unsigned)__half_as_ushort(o[2]) | ((unsigned)__half_as_ushort(o[3]) << 16));
}

// A5-K7 (FOLD = true, moe.gu_fold): the gate/up A operand is FORMED here instead of copied from
// xh_e — xh = x (the [M][K] hidden, shared by every expert), suh = the expert's suh row; per
// (row, 128-block) A = f16(wp20_suh_had(x, suh)): the pinned, boot-verified xq_had_suh_multi
// sequence on the same h2f'd halves, then xq_f2h (== the bytes xq_had_suh_multi wrote to xh_e,
// which the FOLD = false copy stages). The expert trellis's first PF_D k16 rows are L2-prefetched
// BEFORE the latency-bound transform (value-transparent), so the DRAM stream starts under it.
// CF-P1e (16-row class): KCH > 1 stages A in KCH K-chunks (smem = M*(K/KCH+8)*2 instead of M*(K+8)*2), re-staging
// between chunks behind two barriers. The k16 loop, its order, the ring decode and the mma chain are untouched, so
// the result is bit-identical to KCH = 1; it exists so a 9..16-row gate/up (K = 2560: 51..82 KB at KCH = 1, one CTA
// per SM) keeps two to three co-resident CTAs. KCH = 1 (every existing instantiation) compiles to the code below
// exactly as before (the KCH > 1 statements are discarded by `if constexpr`).
template <int BITS, bool DIET, bool FOLD = false, int KCH = 1>
__device__ __forceinline__ void wp20_a1_mainloop(const uint16_t* __restrict__ trellis,
                                                 const __half* __restrict__ xh,
                                                 __half* a1_sa,
                                                 int M, int K, int N, int cta_tile,
                                                 float (&acc)[2][4],
                                                 const __half* __restrict__ suh = nullptr) {
    static_assert(KCH >= 1 && !(FOLD && KCH > 1), "wp20_a1_mainloop: K-chunked staging is for the unfolded gate/up only");
    constexpr bool CH = KCH > 1;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int KCw = CH ? K / KCH : K;                 // staged K width (== K when not chunked)
    const int SK = KCw + 8;
    // S-A3-n F4 / S-A3-o O1 L2 prefetch constants (value-transparent)
    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < BITS * 32;
    if constexpr (FOLD) {
        const uint32_t* ring0 = (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
        if (pf_lane) {
            const char* p0 = (const char*)ring0 + lane * 32;
            const size_t st = (size_t)nb * 8u * BITS * 4;
            #pragma unroll
            for (int d = 0; d < PF_D; ++d)
                if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * st));
        }
        constexpr int FU = 4;                             // transforms in flight per warp
        const int nblk = K >> 7, nt = M * nblk;
        for (int b0 = warp; b0 < nt; b0 += 8 * FU) {      // warp-uniform (full-warp shuffles)
            float xs[FU][4], ss[FU][4];
            #pragma unroll
            for (int u = 0; u < FU; ++u) {
                const int bi = b0 + 8 * u;
                if (bi < nt) {
                    const int r = bi / nblk, blk = bi - r * nblk;
                    wp20_unpack4(__ldg((const uint2*)(xh + (size_t)r * K + blk * 128 + lane * 4)), xs[u]);
                    wp20_unpack4(__ldg((const uint2*)(suh + blk * 128 + lane * 4)), ss[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < FU; ++u) {
                const int bi = b0 + 8 * u;
                if (bi < nt) {
                    const int r = bi / nblk, blk = bi - r * nblk;
                    float v[4];
                    wp20_suh_had(xs[u], ss[u], v, lane);      // == xq_had_suh_multi (pinned SASS sequence)
                    __half o[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
                    *(uint2*)&a1_sa[(size_t)r * SK + blk * 128 + lane * 4] = wp20_pack4(o);
                }
            }
        }
    } else {
        const int vpr = KCw >> 3;
        for (int i = tid; i < M * vpr; i += blockDim.x) {
            const int r = i / vpr, c = i - r * vpr;
            *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + c * 8];
        }
    }
    __syncthreads();

    constexpr int W = 8 * BITS;
    int s_i0m[8], s_i1m[8], s_sf[8];      // SHFL body (DIET = false) only
    int offA = 0, offB = 0, sh7 = 0;      // diet body only
    if constexpr (DIET) {
        static_assert(BITS == 3 || BITS == 4, "word diet: 2 ring words per lane per step hold for b3/b4 only");
        // slot j = 0 (h = 0, s = 0) through the SAME tmap arithmetic as the old body
        const int r = 2 * (lane & 3), c = lane >> 2;
        const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
        const int t0 = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
        const int b10 = (t0 + 257) * BITS;        // window END of slot 0; slot j ends at b10 + j*BITS
        const int A = (b10 - 16) >> 5;            // first ring word any of the 8 windows touches
        offA = A % W;
        offB = (A + 1) % W;
        sh7 = 32 * (A + 2) - (b10 + 7 * BITS);    // >= 0: slot 7's window ends inside word A+1
    } else {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
                const int c = 8 * h + (lane >> 2);
                const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
                const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                                  + 4 * (c >> 3) + 32 * (c & 1);
                const int b1 = (t + 257) * BITS;
                const int i0 = (b1 - 16) >> 5;
                const int i1 = (b1 - 1) >> 5;
                s_i0m[h * 4 + s] = i0 % (BITS * 8);
                s_i1m[h * 4 + s] = i1 % (BITS * 8);
                s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;

    // S-A3-n F4 / S-A3-o O1 L2 prefetch, unchanged (value-transparent; FOLD issued it above).
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;
    if (!FOLD && pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    int kb0 = 0;                                       // first k16 step of the staged chunk (KCH > 1 only)
    for (int kb = 0; kb < KB; ++kb) {
        if constexpr (CH) {
            if (kb - kb0 == (KCw >> 4)) {              // chunk boundary: uniform across the CTA
                __syncthreads();                       // every warp is done reading the previous chunk
                const int vpr2 = KCw >> 3;
                const int koff = kb * 16;              // == chunk index * KCw
                for (int i = tid; i < M * vpr2; i += blockDim.x) {
                    const int r = i / vpr2, c = i - r * vpr2;
                    *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + koff + c * 8];
                }
                __syncthreads();
                kb0 = kb;
            }
        }
        if (pf_lane && kb + PF_D < KB)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
        pf_ptr += pf_step;
        uint32_t bfrag[2][2];
        if constexpr (DIET) {
            const unsigned wA = __ldg(ring + offA);
            const unsigned wB = __ldg(ring + offB);
            const unsigned long long P = ((((unsigned long long)wA) << 32) | (unsigned long long)wB) >> sh7;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    const unsigned idx = (unsigned)(P >> (BITS * (7 - j))) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
        } else {
            // the A1_SHFL block of exl3_gemm_body_a1, verbatim
            const unsigned w0 = (lane < W) ? __ldg(ring + lane) : 0u;
            const unsigned w1 = (W > 32 && lane + 32 < W) ? __ldg(ring + lane + 32) : 0u;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (W > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
        }
        ring += ring_kstride;
        const int col = CH ? (kb - kb0) * 16 : kb * 16;
        const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }
}

// The CTA's f16 tile (the exact bytes the old epilogue stored as y_raw) into smem [M][128].
// Caller guarantees every warp is done reading the A operand from this smem.
__device__ __forceinline__ void wp20_stage_tile(__half* yt, const float (&acc)[2][4], int M) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lc = 16 * warp + 8 * h + 2 * tig;
        if (gid < M) *(__half2*)&yt[gid * 128 + lc] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *(__half2*)&yt[(gid + 8) * 128 + lc] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
}

// == xq_had_svh_multi on one 128-col block of one row: y = f16(H128(yraw) * svh), lane owns
// elements lane*4..+3 (yr: the f16 tile row in smem; sr: the block's svh). Returned packed.
__device__ __forceinline__ uint2 wp20_svh_row(const __half* yr, const __half* sr, int lane) {
    float v[4], s[4];
    wp20_unpack4(*(const uint2*)(yr + lane * 4), v);
    exl3_had128(v, lane);
    wp20_unpack4(__ldg((const uint2*)(sr + lane * 4)), s);
    __half o[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j] * s[j]);
    return wp20_pack4(o);
}

// Last-arrival election: returns true (CTA-uniform) in exactly ONE of the `target` CTAs that
// arrive on `cnt` in this launch — the last — after every arriving CTA's prior global stores
// are visible to it; that CTA re-arms the counter to 0 for the next launch.
__device__ __forceinline__ bool wp20_arrive(unsigned* cnt, unsigned target, int* s_flag) {
    // CUTLASS GenericBarrier's publication (split-K serial / stream-K fixup): the CTA barrier
    // orders every thread's stores before thread 0's gpu-scope acq_rel fence + counter bump;
    // the elected thread's fence after the bump is the acquire side, the second barrier hands
    // it to the CTA. Readers use ld.global.cg (L2) for the other CTAs' tiles.
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const unsigned old = atomicAdd(cnt, 1u);
        const int last = old == target - 1u;
        if (last) {
            atomicExch(cnt, 0u);                       // all `target` arrivals are in: re-arm
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        *s_flag = last;
    }
    __syncthreads();
    return *s_flag != 0;
}

// FOLD (A5-K7, entry xq_moe_gu_epi_f): A is formed in the prologue from x and suh_gu_all[ix]
// (wp20_a1_mainloop<.., FOLD>) — the xq_had_suh_multi launch before this kernel goes; xh unused.
template <bool DIET, bool FOLD = false, int KCH = 1, int BITS = 3>
__device__ __forceinline__ void wp20_moe_gu(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ xh, __half* __restrict__ ygu,
                                            int M, int K, int N, const int* __restrict__ esel,
                                            const int* __restrict__ idxmap,
                                            const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                                            __half* __restrict__ xhd, unsigned* __restrict__ cnt,
                                            const __half* __restrict__ x, const __half* __restrict__ sg,
                                            float* __restrict__ sgv,
                                            const __half* __restrict__ suh_gu_all = nullptr) {
    extern __shared__ __half a1_sa[];
    __shared__ int s_flag;
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ix = idxmap[expert];
    float acc[2][4];
    if constexpr (FOLD)
        wp20_a1_mainloop<BITS, DIET, true>(base + (size_t)offs[expert], x, a1_sa, M, K, N, tile, acc,
                                        suh_gu_all + (long long)ix * K);
    else
        wp20_a1_mainloop<BITS, DIET, false, KCH>(base + (size_t)offs[expert], xh + (long long)expert * M * K, a1_sa,
                                              M, K, N, tile, acc);
    __syncthreads();                                   // every warp is done reading A
    __half* yt = a1_sa;                                // [M][128]
    wp20_stage_tile(yt, acc, M);
    __syncthreads();
    // own svh (== xq_had_svh_multi, block `tile`): kept in smem, published to ygu
    const int cta_n = tile * 128;
    __half* ye = ygu + (long long)expert * M * N;
    for (int r = warp; r < M; r += 8) {
        const uint2 o = wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
        *(uint2*)(yt + r * 128 + lane * 4) = o;
        *(uint2*)(ye + (long long)r * N + cta_n + lane * 4) = o;
    }
    const int P = tiles >> 1;                          // gate tiles [0, P), up tiles [P, 2P)
    const int p = tile < P ? tile : tile - P;
    if (wp20_arrive(cnt + expert * P + p, 2u, &s_flag)) {
        // second arrival of the (gate p, up p+P) pair: silu*mul + the down input transform
        const int KD = N >> 1;                         // down K = moe_intermediate
        const int ptile = tile < P ? tile + P : tile - P;
        const bool own_gate = tile < P;
        for (int r = warp; r < M; r += 8) {
            const uint2 pw = __ldcg((const uint2*)(ye + (long long)r * N + ptile * 128 + lane * 4));
            const uint2 ow = *(const uint2*)(yt + r * 128 + lane * 4);
            float g[4], u[4], sv[4], dv[4], v[4];
            wp20_unpack4(own_gate ? ow : pw, g);
            wp20_unpack4(own_gate ? pw : ow, u);
            wp20_unpack4(__ldg((const uint2*)(suh_d_all + (long long)ix * KD + p * 128 + lane * 4)), sv);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                dv[j] = xq_h2f(xq_f2h(xq_silu(g[j]) * u[j]));   // == xq_moe_gate_mul (the f16 din)
            wp20_suh_had(dv, sv, v, lane);                        // == xq_had_suh_multi (pinned SASS sequence)
            __half o[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
            *(uint2*)(xhd + ((long long)expert * M + r) * KD + p * 128 + lane * 4) = wp20_pack4(o);
        }
    }
    // Shared-expert gate sigmoid, once per row, by expert 0's CTAs (row r -> tile r % tiles):
    // xq_moe_combine's per-row prologue with its 1024-thread shape emulated (virtual thread
    // vt = tid + 256q owns c = vt, vt+1024, ... ascending; the same halving tree), and the
    // old PTX roundings pinned (fma.rn partials, add tree, xq_sig).
    if (expert == 0) {
        float* red = (float*)a1_sa;                    // 4 KB <= the A-once smem (K = hidden)
        for (int r = tile; r < M; r += tiles) {
            __syncthreads();                           // smem reuse (tile staging / previous row)
            #pragma unroll 1
            for (int q = 0; q < 4; ++q) {
                const int vt = tid + 256 * q;
                float part = 0.0f;
                #pragma unroll 1
                for (int c = vt; c < K; c += 1024)
                    part = __fmaf_rn(xq_h2f(sg[c]), xq_h2f(x[(long long)r * K + c]), part);
                red[vt] = part;
            }
            __syncthreads();
            for (int s2 = 512; s2 > 0; s2 >>= 1) {
                for (int i = tid; i < s2; i += 256) red[i] = __fadd_rn(red[i], red[i + s2]);
                __syncthreads();
            }
            if (tid == 0) sgv[r] = xq_sig(red[0]);
        }
    }
}

// Down epilogue, templated on the combine's top-k (instances: 8, 10 — see the entries below).
// Shared memory (dynamic, the launcher passes a1_smem(M, K) = M*(K+8)*2 B, K = moe_intermediate):
//   mainloop: the A operand [M][K+8] f16;  after it (A dead): yt [M][128] f16 at 0, then the
//   combine tables s_slot [M][TOPK] i32 + s_w [M][TOPK] f32 at byte 256*M.
//   Need 256*M + 8*M*TOPK <= M*(K+8)*2  <=>  8*TOPK <= 2K - 240: at K = 640 any TOPK <= 130
//   fits for EVERY M (M = 8, TOPK = 10: 2688 B of the 10368 B); the launcher re-checks.
template <bool DIET, int TOPK, int BITS = 3>
__device__ __forceinline__ void wp20_moe_dn(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ xhd, __half* __restrict__ yd,
                                            int M, int K, int N, const int* __restrict__ esel,
                                            const int* __restrict__ idxmap, const __half* __restrict__ svh_all,
                                            unsigned* __restrict__ cnt, const __half* __restrict__ ysh,
                                            const int* __restrict__ ids, const float* __restrict__ wts,
                                            const int* __restrict__ slotmap, const float* __restrict__ sgv,
                                            __half* __restrict__ out, int topk) {
    static_assert(TOPK >= 1 && TOPK <= 16, "wp20_moe_dn: TOPK instance out of the planned range");
    extern __shared__ __half a1_sa[];
    __shared__ int s_flag;
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    const int live = *esel;
    if (expert >= live) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ix = idxmap[expert];
    float acc[2][4];
    wp20_a1_mainloop<BITS, DIET>(base + (size_t)offs[expert], xhd + (long long)expert * M * K, a1_sa,
                              M, K, N, tile, acc);
    __syncthreads();
    __half* yt = a1_sa;                                // [M][128]
    wp20_stage_tile(yt, acc, M);
    // The combine tables' routing rows (ids, wts: [M][TOPK], written by the router launches
    // before this grid) go to smem NOW by cp.async (baseline sm_80+ 4-B .ca form), one entry per
    // thread (M*TOPK <= 16*16 = blockDim at the a1 cap): ids -> s_slot (staging; the last arrival
    // rewrites it in place to slotmap[id]), wts -> s_w (final). No register ever holds them, so
    // nothing can stall on them before the arrival (a register prefetch did: ptxas MOVed the
    // loaded id right after the next barrier, in every CTA); each copying thread waits on its
    // own group after the svh stores (long since landed), and the arrival's barriers publish
    // the smem to the CTA. The last arrival then pays only the slotmap hop, not ids -> slotmap.
    // Region [256*M, 256*M + 8*M*TOPK) B: A is dead (the barrier above), yt is [0, 256*M).
    const int nrt = M * TOPK;
    int* s_slot = (int*)(a1_sa + M * 128);             // [M][TOPK] (fit: see the header comment)
    float* s_w = (float*)(s_slot + nrt);
    const bool rt_mine = tid < nrt;
    if (rt_mine) {
        const unsigned d_id = (unsigned)__cvta_generic_to_shared(s_slot + tid);
        const unsigned d_w = (unsigned)__cvta_generic_to_shared(s_w + tid);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d_id), "l"(ids + tid) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d_w), "l"(wts + tid) : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    __syncthreads();
    const int cta_n = tile * 128;
    for (int r = warp; r < M; r += 8)                  // == xq_had_svh_multi -> yd
        *(uint2*)(yd + ((long long)expert * M + r) * N + cta_n + lane * 4) =
            wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
    if (rt_mine) asm volatile("cp.async.wait_all;\n" ::: "memory");   // before exit / publication
    if (!wp20_arrive(cnt + tile, (unsigned)live, &s_flag)) return;
    // Last arrival for this N-tile: == xq_moe_combine on columns [cta_n, cta_n + 128).
    if (topk != TOPK) __trap();                        // launcher bug: never silently mis-combine
    if (rt_mine) s_slot[tid] = slotmap[s_slot[tid]];   // own entry: id -> slot (== slotmap[ids[i]])
    for (int i = tid + blockDim.x; i < nrt; i += blockDim.x) {   // M*TOPK > blockDim: never at the a1 cap
        s_slot[i] = slotmap[ids[i]];
        s_w[i] = wts[i];
    }
    __syncthreads();
    // 4 consecutive columns per thread (8-B loads/stores; 32 threads per row, so M <= 8 is ONE
    // pass with all TOPK L2 loads in flight at once). Per element the SAME op sequence as
    // xq_moe_combine: a = rn(sgv * ysh); for j ascending: a = fma.rn(w_j, y_j, a); f16(a).
    for (int e = tid; e < M * 32; e += blockDim.x) {
        const int r = e >> 5, i = cta_n + ((e & 31) << 2);
        const int* sl = s_slot + r * TOPK;
        const float* sw = s_w + r * TOPK;
        uint2 raw[TOPK];
        #pragma unroll
        for (int j = 0; j < TOPK; ++j) {               // unconditional (a -1 slot reads slot 0, unused)
            const int s = sl[j] < 0 ? 0 : sl[j];
            raw[j] = __ldcg((const uint2*)(yd + ((long long)s * M + r) * N + i));
        }
        float ys[4], a[4];
        wp20_unpack4(__ldg((const uint2*)(ysh + (long long)r * N + i)), ys);
        const float g = sgv[r];
        #pragma unroll
        for (int q = 0; q < 4; ++q) a[q] = __fmul_rn(g, ys[q]);
        #pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            if (sl[j] >= 0) {
                float y[4];
                wp20_unpack4(raw[j], y);
                const float w = sw[j];
                #pragma unroll
                for (int q = 0; q < 4; ++q) a[q] = __fmaf_rn(w, y[q], a[q]);
            }
        }
        __half o[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) o[q] = xq_f2h(a[q]);
        *(uint2*)(out + (long long)r * N + i) = wp20_pack4(o);
    }
}

// Rung 1 alone (GB10_WP20_EPI=0): the diet body behind xq_gemm_grouped_a1b3's exact contract.
template <bool DIET, int BITS = 3>
__device__ __forceinline__ void wp20_grouped_plain(const uint16_t* __restrict__ base,
                                                   const uint64_t* __restrict__ offs,
                                                   const __half* __restrict__ xh,
                                                   __half* __restrict__ y,
                                                   int M, int K, int N,
                                                   const int* __restrict__ esel) {
    extern __shared__ __half a1_sa[];
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    float acc[2][4];
    wp20_a1_mainloop<BITS, DIET>(base + (size_t)offs[expert], xh + (long long)expert * M * K, a1_sa,
                              M, K, N, tile, acc);
    __half* ye = y + (long long)expert * M * N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = tile * 128 + 16 * warp + 8 * h + 2 * tig;
        if (gid < M) *(__half2*)&ye[(size_t)gid * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *(__half2*)&ye[(size_t)(gid + 8) * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3_wd(const uint16_t* __restrict__ base,
                        const uint64_t* __restrict__ offs,
                        const __half* __restrict__ xh,
                        __half* __restrict__ y,
                        int M, int K, int N, int bits,
                        const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    wp20_grouped_plain<true>(base, offs, xh, y, M, K, N, esel);
}

#define WP20_GU_ENTRY(NAME, DIET)                                                                   \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<DIET>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, xhd, cnt,   \
                      x, sg, sgv);                                                                 \
}
WP20_GU_ENTRY(xq_moe_gu_epi, true)
WP20_GU_ENTRY(xq_moe_gu_epi_sh, false)
// CF-P1e (16-row class): the gate/up epilogue with A staged in 2 / 4 K-chunks (9..16 rows; dynamic smem
// M*(K/KCH+8)*2). Bit-identical to xq_moe_gu_epi (same mainloop order); the 8-row entries above are untouched.
#define WP20_GU_ENTRY_KC(NAME, KCH)                                                                 \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<true, false, KCH>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, \
                                  xhd, cnt, x, sg, sgv);                                           \
}
WP20_GU_ENTRY_KC(xq_moe_gu_epi_c2, 2)
WP20_GU_ENTRY_KC(xq_moe_gu_epi_c4, 4)
// CF-P1g step 2 (4.05 bpw, K = 4 experts): the same epilogue kernels instantiated for 4-bit trellis (the word diet holds for
// b3/b4; ring = 8*BITS = 32 words, one per lane). KCH 1 = <= 9 rows, 2 / 4 = the K-chunked 10..16-row class.
#define WP20_GU_ENTRY_B4(NAME, KCH)                                                                 \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<true, false, KCH, 4>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, \
                                     xhd, cnt, x, sg, sgv);                                        \
}
WP20_GU_ENTRY_B4(xq_moe_gu_epi_b4, 1)
WP20_GU_ENTRY_B4(xq_moe_gu_epi_b4_c2, 2)
WP20_GU_ENTRY_B4(xq_moe_gu_epi_b4_c4, 4)

// P7 B3 (TP2BIT5 width closure, K = 5 experts — e.g. layer-0 of the 4bpw Flash-Next pack):
// the SHFL twins at BITS = 5. The word diet is b3/b4-only by static_assert (ring = 8*BITS words
// vs 32 lanes), so every 5-bit entry is SHFL; bodies and contracts are otherwise the b4 set's.
#define WP20_GU_ENTRY_B5(NAME, KCH)                                                                 \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<false, false, KCH, 5>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, \
                                     xhd, cnt, x, sg, sgv);                                        \
}
WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5, 1)
WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5_c2, 2)
WP20_GU_ENTRY_B5(xq_moe_gu_epi_b5_c4, 4)
// P7 B3 leg 2 (K = 2/6/7 experts): the same SHFL twin shape at any width (the two-ring-word
// select covers bits <= 7; the word diet stays b3/b4-only by static_assert). Top-10 down only.
#define WP20_GU_ENTRY_BW(NAME, KCH, B)                                                              \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<false, false, KCH, B>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, \
                                     xhd, cnt, x, sg, sgv);                                        \
}
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b2, 1, 2)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b2_c2, 2, 2)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b2_c4, 4, 2)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b6, 1, 6)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b6_c2, 2, 6)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b6_c4, 4, 6)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b7, 1, 7)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b7_c2, 2, 7)
WP20_GU_ENTRY_BW(xq_moe_gu_epi_b7_c4, 4, 7)
// A5-K7 (moe.gu_fold): xq_moe_gu_epi with xq_had_suh_multi folded into the prologue. Same 16
// params (xh ignored) + suh_gu_all (the gate/up suh table, [experts][K]). Bitwise == the
// xq_had_suh_multi -> xq_moe_gu_epi pair (--probe-exl3-binv EXL3-MOE-EPI, "fold" rows).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_gu_epi_f(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,
                const int* __restrict__ esel, const int* __restrict__ idxmap,
                const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,
                const __half* __restrict__ sg, float* __restrict__ sgv, const __half* __restrict__ suh_gu_all) {
    XQ_PDL_ENTRY();
    wp20_moe_gu<true, true>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, xhd, cnt,
                            x, sg, sgv, suh_gu_all);
}

// One entry per (decode body, top-k). Adding a top-k = one more pair here + its name in
// src/exl3_forward.rs wp20_dn_fn (the launcher logs a loud fallback for any top-k without one).
#define WP20_DN_ENTRY(NAME, DIET, TOPK)                                                             \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,                              \
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,  \
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,    \
     int topk) {                                                                                   \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_dn<DIET, TOPK>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,  \
                            wts, slotmap, sgv, out, topk);                                         \
}
WP20_DN_ENTRY(xq_moe_dn_epi_k8, true, 8)
WP20_DN_ENTRY(xq_moe_dn_epi_sh_k8, false, 8)
WP20_DN_ENTRY(xq_moe_dn_epi_k10, true, 10)
WP20_DN_ENTRY(xq_moe_dn_epi_sh_k10, false, 10)
// CF-P1g step 2: the 4-bit down epilogue (top-10) and the rung-1 / plain a1 entries for K = 4 experts.
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_dn_epi_b4_k10(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,
     const int* __restrict__ esel, const int* __restrict__ idxmap,
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,
     int topk) {
    XQ_PDL_ENTRY();
    wp20_moe_dn<true, 10, 4>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,
                             wts, slotmap, sgv, out, topk);
}
// P7 B3: the 5-bit down epilogue (top-10, SHFL; no diet twin exists at K = 5).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_dn_epi_b5_k10(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,
     const int* __restrict__ esel, const int* __restrict__ idxmap,
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,
     int topk) {
    XQ_PDL_ENTRY();
    wp20_moe_dn<false, 10, 5>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,
                              wts, slotmap, sgv, out, topk);
}
// P7 B3 leg 2: the 2/6/7-bit down epilogues (top-10, SHFL; no diet twin exists there).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_dn_epi_b2_k10(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,
     const int* __restrict__ esel, const int* __restrict__ idxmap,
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,
     int topk) {
    XQ_PDL_ENTRY();
    wp20_moe_dn<false, 10, 2>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,
                                wts, slotmap, sgv, out, topk);
}
// P7 B3 leg 2: the 2/6/7-bit down epilogues (top-10, SHFL; no diet twin exists there).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_dn_epi_b6_k10(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,
     const int* __restrict__ esel, const int* __restrict__ idxmap,
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,
     int topk) {
    XQ_PDL_ENTRY();
    wp20_moe_dn<false, 10, 6>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,
                                wts, slotmap, sgv, out, topk);
}
// P7 B3 leg 2: the 2/6/7-bit down epilogues (top-10, SHFL; no diet twin exists there).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_dn_epi_b7_k10(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,
     const int* __restrict__ esel, const int* __restrict__ idxmap,
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,
     int topk) {
    XQ_PDL_ENTRY();
    wp20_moe_dn<false, 10, 7>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,
                                wts, slotmap, sgv, out, topk);
}
// P7 B3: rung-1 grouped entry for 5-bit experts (SHFL a1 body; bitwise == xq_gemm_grouped_xh).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b5(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<5>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}
// P7 B3 leg 2: rung-1 grouped entries for 2/6/7-bit experts (SHFL a1 body; bitwise == xq_gemm_grouped_xh).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b2(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<2>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}
// P7 B3 leg 2: rung-1 grouped entries for 2/6/7-bit experts (SHFL a1 body; bitwise == xq_gemm_grouped_xh).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b6(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<6>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}
// P7 B3 leg 2: rung-1 grouped entries for 2/6/7-bit experts (SHFL a1 body; bitwise == xq_gemm_grouped_xh).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b7(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<7>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}
// K8EXP (2026-11-07): the K = 8 (k2 16) grouped expert entry — the MTP head of the issue-#26 4bpw pack (mtp_bits 8)
// and the SAGE tails. Its own entry, NOT a case in xq_gemm_grouped_xh: adding case 8 there re-allocated that kernel
// (92 -> 87 registers, ptxas sm_121), and xh is the only decode body of the half-bit packs (1.66 / 2.68 / 3.68), so
// it stays byte-for-byte as it was. Same 9-argument signature and grid as xh (grouped_fn returns this name for bits
// 8 at every width); `bits` is unused. A-once body when M*(K+8)*2 fits the dynamic smem the launcher passes
// (a1_smem): WRING = 64 ring words = exactly the two coalesced SHFL source words per lane (w0 = 0..31,
// w1 = 32..63); otherwise the barrier body = the served dense K = 8 body (exl3_hmma_gemm case 16, PR#27-D2).
// Gate: --probe-exl3-bitwidths section 7 (multi-expert, poisoned, vs the host oracle and the m16 reference).
extern "C" __global__ void xq_gemm_grouped_b8(const uint16_t* __restrict__ base,
                                              const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh,
                                              __half* __restrict__ y,
                                              int M, int K, int N, int bits,
                                              const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    (void)bits;
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    __half* ye = y + (long long)expert * M * N;
    if (exl3_a1_fits(M, K)) exl3_gemm_body_a1<8>(tr, xe, ye, M, K, N, tile);
    else                    exl3_gemm_body<8>(tr, xe, ye, M, K, N, tile);
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b4(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<4>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b4_wd(const uint16_t* __restrict__ base,
                        const uint64_t* __restrict__ offs,
                        const __half* __restrict__ xh,
                        __half* __restrict__ y,
                        int M, int K, int N, int bits,
                        const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    wp20_grouped_plain<true, 4>(base, offs, xh, y, M, K, N, esel);
}

