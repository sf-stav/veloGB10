// =============================================================================
// kernels/gpu_dsv41/52_idx_score.cuh — leg KI / P7PF: fused indexer score + selection (never stores scores).
//
// Grid (G, n_rb): CTA (c, b) takes row block b (<= R rows of one lane, sharing that lane's index-K
// cache) and key chunk c of its tiles. Warps: 4 key warps x RG row groups x TG tile groups (r1 uses
// 8 warps = 256 threads at 2 CTAs/SM; r2/r4/r8 use 16 warps = 512 threads). Prologue: RoPE + FP4
// quantisation of each row's 32 x 128 query into MMA B-fragment order in shared memory, and the
// scaled head weights, overlapping cp.async prefetch of the first STAGES - 1 key super-tiles. Main
// loop: key tiles of 128 (16 blocks of 8), newest first, through a multi-stage cp.async ring; per row
// the block-scaled FP4 MMA gives 32 heads x 32 keys per warp, the epilogue rounds / rectifies /
// weights / sums the heads exactly as the reference (51_idx_common.cuh), and a running filter keeps
// every key that can still be in the row's top 512 (radix-select compaction to 512 when the row's
// buffer fills; the threshold only rises). Output: one chunk -> the final ascending position list;
// several chunks -> a sorted local top-512 per chunk, merged by d41_idx_merge (53). Mode 1 visits the
// union of the row block's candidate blocks (layers above the candidate source) and masks each row by
// its own set. The candidate source (layer 20) also writes the per-8-block maximum score (write_bmax).
// =============================================================================
#pragma once

template <int R, int STAGES, int TG, int CAP>
struct d41_idx_smem {
    static_assert(STAGES >= 2, "multi-stage ring requires STAGES >= 2 so stage s is idle during compaction");
    static constexpr int KT = TG * D41_IDX_TILE;                        // keys per stage
    static constexpr int BUF = 0;                                        // u64 [R][CAP] (aliases CS [R][32][2] during prep)
    static_assert(R * CAP * 8 >= R * 32 * 2 * 4, "BUF smaller than RoPE CS table");
    static constexpr int KQ = BUF + R * CAP * 8;                         // u8 [STAGES][KT][64] (16-B chunks swizzled; idle stage s holds radix scratch)
    static_assert(KT * 64 >= 1088 && (R == 1 || KT * 64 >= R * 576), "KQ stage smaller than radix compaction scratch");
    static constexpr int KS = KQ + STAGES * KT * 64;                     // u8 [STAGES][KT][4]
    static constexpr int QF = KS + STAGES * KT * 4;                      // u32 [R][2 ks][2 half][32 lane][4 nt]
    static constexpr int QS = QF + R * 2 * 2 * 32 * 4 * 4;              // u16 [R][2 ks][8 g][4 nt]
    static constexpr int WS = QS + R * 2 * 8 * 4 * 2;                    // u32 [R][4 t][4 nt] (bf16x2)
    static constexpr int ENT = WS + R * 4 * 4 * 4;                       // u32 [STAGES][TG * 16]
    static constexpr int THR = ENT + STAGES * TG * 16 * 4;               // u64 [R]
    static constexpr int CNT = THR + R * 8;                              // int [R]
    static constexpr int NV = CNT + R * 4;                               // int [R]
    static constexpr int RAW_TOTAL = (NV + R * 4 + 15) / 16 * 16;
    static constexpr int MSM_BYTES = ((int)sizeof(d41_idx_msm) + 15) / 16 * 16;
    static constexpr int TOTAL = RAW_TOTAL > MSM_BYTES ? RAW_TOTAL : MSM_BYTES;
    static constexpr int CS = 0;                                         // float [R][32][2] (cos, sin) in BUF during prep
};

__device__ __forceinline__ void d41_cp16(void* dst, const void* src, int bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void d41_cp16_full(void* dst, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}

__device__ __forceinline__ float d41_idx_load_in(const d41_idx_args& a, unsigned long long base, long long idx) {
    if (a.in_f32) return d41_bf16r(((const float*)base)[idx]);
    return __bfloat162float(((const __nv_bfloat16*)base)[idx]);
}

// Number of 128-key tiles of row block b (visible range, or the union in mode 1).
__device__ __forceinline__ int d41_idx_tiles(const d41_idx_args& a, int b, int R) {
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    if (a.mode == 1) return (((const int*)a.ucnt)[b] + 15) / 16;
    int nmax = 0;
    for (int r = 0; r < rb.nrows && r < R; r++) {
        int n = d41_idx_nvis(((const int*)a.pos)[rb.row0 + r], a.ratio);
        if (n > a.n_keys_cap) n = a.n_keys_cap;
        nmax = n > nmax ? n : nmax;
    }
    return (nmax + D41_IDX_TILE - 1) / D41_IDX_TILE;
}

// Load the TG tiles of super-tile `it` (tiles t_hi - 1 - it*TG - tg, tg = 0..TG-1, those >= t_lo).
template <int R, int STAGES, int TG, int CAP>
__device__ __forceinline__ void d41_idx_issue(const d41_idx_args& a, const d41_idx_rb& rb, int b, int ucnt,
                                              unsigned char* sm, int stage, int it, int t_lo, int t_hi) {
    using L = d41_idx_smem<R, STAGES, TG, CAP>;
    unsigned char* kq_s = sm + L::KQ + stage * L::KT * 64;
    unsigned char* ks_s = sm + L::KS + stage * L::KT * 4;
    const unsigned char* kq = (const unsigned char*)rb.kq;
    const unsigned char* ks = (const unsigned char*)rb.ks;
    const int tid = threadIdx.x, nt = blockDim.x;
    constexpr int NQ = L::KT * 4, NS = L::KT / 4;
    const int first_tile = t_hi - 1 - it * TG;
    const int last_tile = first_tile - (TG - 1);
    // Fast interior path for mode-0 full tiles wholly inside n_keys_cap: constant-16B cp.async without per-chunk bounds math.
    if (a.mode == 0 && last_tile >= t_lo && (long long)(first_tile + 1) * D41_IDX_TILE <= a.n_keys_cap) {
        for (int q = tid; q < NQ; q += nt) {
            const int kr = q >> 2, c = q & 3;
            const long long p = (long long)(first_tile - kr / D41_IDX_TILE) * D41_IDX_TILE + (kr % D41_IDX_TILE);
            d41_cp16_full(kq_s + kr * 64 + ((c ^ ((kr >> 1) & 3)) << 4), kq + p * 64 + c * 16);
        }
        for (int q = tid; q < NS; q += nt) {
            const int kr = q * 4;
            const long long p = (long long)(first_tile - kr / D41_IDX_TILE) * D41_IDX_TILE + (kr % D41_IDX_TILE);
            d41_cp16_full(ks_s + kr * 4, ks + p * 4);
        }
        d41_cp_commit();
        return;
    }
    const unsigned* uni = (const unsigned*)a.uni + (long long)b * a.ucap;
    for (int q = tid; q < NQ + NS; q += nt) {
        const bool scale = q >= NQ;
        const int kr = scale ? (q - NQ) * 4 : q >> 2, c = q & 3;   // key row within the stage
        const int tile = first_tile - kr / D41_IDX_TILE, kt = kr % D41_IDX_TILE;
        long long p; bool ok = tile >= t_lo;
        if (a.mode == 0) { p = (long long)tile * D41_IDX_TILE + kt; }
        else {
            const int ei = tile * 16 + (kt >> 3);
            ok = ok && ei < ucnt;
            const unsigned e = ok ? uni[ei] : 0u;
            p = (long long)(e >> 8) * D41_IDX_CAND_BLOCK + (kt & 7);
        }
        ok = ok && p < a.n_keys_cap;
        // a scale chunk holds 4 keys: copy only the ones inside the cache (cp.async zero-fills the rest)
        const int sbytes = ok ? (int)min(16ll, (a.n_keys_cap - p) * 4) : 0;
        if (scale) d41_cp16(ks_s + kr * 4, ok ? (const void*)(ks + p * 4) : (const void*)ks, sbytes);
        else d41_cp16(kq_s + kr * 64 + ((c ^ ((kr >> 1) & 3)) << 4), ok ? (const void*)(kq + p * 64 + c * 16) : (const void*)kq, ok ? 16 : 0);
    }
    if (a.mode == 1 && tid < TG * 16) {
        const int tile = first_tile - tid / 16, ei = tile * 16 + tid % 16;
        ((unsigned*)(sm + L::ENT))[stage * TG * 16 + tid] = (tile >= t_lo && ei < ucnt) ? uni[ei] : 0u;
    }
    d41_cp_commit();
}

// Prologue: RoPE + FP4 of the rows' queries into B-fragment order; scaled head weights.
template <int R, int STAGES, int TG, int CAP>
__device__ __forceinline__ void d41_idx_prep(const d41_idx_args& a, const d41_idx_rb& rb, unsigned char* sm) {
    using L = d41_idx_smem<R, STAGES, TG, CAP>;
    const int tid = threadIdx.x, ntd = blockDim.x;
    float* cs = (float*)(sm + L::CS);
    unsigned* qf = (unsigned*)(sm + L::QF);
    unsigned char* qs = sm + L::QS;
    unsigned short* ws = (unsigned short*)(sm + L::WS);
    // head weights first (their loads overlap the query loads below)
    for (int i = tid; i < rb.nrows * 32; i += ntd) {
        const int r = i >> 5, h = i & 31;
        const float wv = d41_idx_load_in(a, a.w, (long long)(rb.row0 + r) * a.w_ld + h);
        const __nv_bfloat16 we = __float2bfloat16_rn(__fmul_rn(wv, 0.015625f));
        const int nt = h >> 3, t = (h & 7) >> 1;
        ws[((r * 4 + t) * 4 + nt) * 2 + (h & 1)] = __bfloat16_as_ushort(we);
    }
    // every (row, head, group of 32) task; with 512 threads and <= 4 rows each thread holds one task,
    // so its loads are in flight while the RoPE table is computed
    for (int base_task = 0; base_task < rb.nrows * 128; base_task += ntd) {
        const int task = base_task + tid;
        const bool have = task < rb.nrows * 128;
        const int r = have ? task >> 7 : 0, h = (task & 127) >> 2, gq = task & 3;
        const long long row = rb.row0 + r;
        float x[32];
        const long long base = row * a.iq_ld + h * D41_IDX_D + gq * 32;
        if (have) {
            if (a.in_f32) {
#pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    const float4 v4 = *(const float4*)((const float*)a.iq + base + j);
                    x[j] = d41_bf16r(v4.x); x[j + 1] = d41_bf16r(v4.y); x[j + 2] = d41_bf16r(v4.z); x[j + 3] = d41_bf16r(v4.w);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j += 8) {
                    const uint4 v8 = *(const uint4*)((const __nv_bfloat16*)a.iq + base + j);
                    const unsigned u[4] = {v8.x, v8.y, v8.z, v8.w};
#pragma unroll
                    for (int k = 0; k < 4; k++) { x[j + 2 * k] = __uint_as_float(u[k] << 16); x[j + 2 * k + 1] = __uint_as_float(u[k] & 0xFFFF0000u); }
                }
            }
        }
        if (base_task == 0) {
            for (int i = tid; i < rb.nrows * 32; i += ntd) {
                const int rr = i >> 5, f = i & 31;
                const int p = ((const int*)a.pos)[rb.row0 + rr];
                const float ang = __fmul_rn((float)p, a.inv_freq[f]);
                float sn, cn;
                d41_sincos(ang, sn, cn);   // == sincosf (00_abi.cuh), no local memory
                cs[(rr * 32 + f) * 2] = cn;
                cs[(rr * 32 + f) * 2 + 1] = sn;
            }
            __syncthreads();
        }
        if (!have) continue;
        if (gq >= 2) {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int f = (gq - 2) * 16 + j;
                const float c = cs[(r * 32 + f) * 2], s = cs[(r * 32 + f) * 2 + 1];
                // torch's CUDA complex multiply (what REFb ran), the shared d41_rope_fwd of 00_abi.cuh
                float re = x[2 * j], im = x[2 * j + 1];
                d41_rope_fwd(re, im, c, s);
                x[2 * j] = d41_bf16r(re);
                x[2 * j + 1] = d41_bf16r(im);
            }
        }
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);          // 6 * 2^-126
        const unsigned sb = __float_as_uint(__fmul_rn(amax, 0.16666666666666666f));
        unsigned e = (sb >> 23) & 0xFFu;
        if (sb & 0x7FFFFFu) e += 1;
        const float rcp = __uint_as_float((254u - e) << 23);   // 1 / scale, exact (a power of two, e in 1..253)
        const int ks = gq >> 1, half = gq & 1, g = h & 7, nt = h >> 3;
        unsigned words[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            unsigned word = 0;
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int j = t * 8 + m * 2;
                word |= d41_cvt_e2m1x2(__fmul_rn(x[j], rcp), __fmul_rn(x[j + 1], rcp)) << (8 * m);
            }
            words[t] = word;
            qf[((((r * 2 + ks) * 2 + half) * 32 + g * 4 + t) * 4) + nt] = word;
        }
        qs[((((r * 2 + ks) * 8 + g) * 4 + nt) * 2) + half] = (unsigned char)e;
        if (a.dbg_q && blockIdx.x == 0) {   // canonical layout: codes [32][64 B], then scales [32][4]
            unsigned char* dq = (unsigned char*)a.dbg_q + row * (D41_IDX_H * 68);
            *(uint4*)(dq + h * 64 + gq * 16) = make_uint4(words[0], words[1], words[2], words[3]);
            dq[D41_IDX_H * 64 + h * 4 + gq] = (unsigned char)e;
        }
    }
}

// D41_IDX_PROF (development builds only, never in the engine): per-CTA clock64 stamps
// [start, after prep, after main loop, after finish, compactions, after merge] into d41_idx_prof[cta][8].
#ifdef D41_IDX_PROF
__device__ long long d41_idx_prof[4096][8];
#define D41_IDX_STAMP(k) do { if (threadIdx.x == 0) d41_idx_prof[blockIdx.y * gridDim.x + blockIdx.x][k] = clock64(); } while (0)
#define D41_IDX_COUNT(k) do { if (threadIdx.x == 0) d41_idx_prof[blockIdx.y * gridDim.x + blockIdx.x][k] += 1; } while (0)
#else
#define D41_IDX_STAMP(k) do { } while (0)
#define D41_IDX_COUNT(k) do { } while (0)
#endif

// The CTA: 4 key warps x RG row groups x TG tile groups warps. Warp (kw, rg, tg) scores keys
// [32 kw, + 32) of tile tg of every super-tile, for rows rg, rg + RG, ... of the row block.
template <int R, int STAGES, int RG, int TG, int CAP>
__device__ __forceinline__ void d41_idx_score_body(const d41_idx_args& a) {
    using L = d41_idx_smem<R, STAGES, TG, CAP>;
    constexpr int NT = 128 * RG * TG;
    extern __shared__ __align__(16) unsigned char d41_idx_sm[];
    unsigned char* sm = d41_idx_sm;
    const int b = blockIdx.y, c = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
    const int kw = w & 3, rg = (w >> 2) % RG, tg = (w >> 2) / RG;
    if (b >= a.n_rb) return;
    D41_IDX_STAMP(0);
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    int* nv = (int*)(sm + L::NV);
    int* cnt = (int*)(sm + L::CNT);
    unsigned long long* thr = (unsigned long long*)(sm + L::THR);
    unsigned long long* buf = (unsigned long long*)(sm + L::BUF);
    const bool stripe = a.stripe_hi > 0 && a.mode == 0;
    const int n_vis_tiles = d41_idx_tiles(a, b, R);
    const int t_base = stripe ? a.stripe_lo / D41_IDX_TILE : 0;
    const int t_end = stripe ? min(n_vis_tiles, a.stripe_hi / D41_IDX_TILE) : n_vis_tiles;
    const int n_tiles = max(0, t_end - t_base);
    const int geff = d41_idx_geff(n_tiles, a.G, a.min_tiles);
    const int lstride = a.lists > 0 ? a.lists : a.G;
    if (c >= geff) {
        if (stripe && tid < rb.nrows) ((int*)a.pcnt)[((long long)b * a.R + tid) * lstride + a.list_off + c] = 0;
        return;
    }
    if (tid < R) {
        int n = 0;
        if (tid < rb.nrows) {
            n = d41_idx_nvis(((const int*)a.pos)[rb.row0 + tid], a.ratio);
            if (n > a.n_keys_cap) { d41_idx_err(a, D41_IDX_ERR_POS); n = a.n_keys_cap; }
        }
        nv[tid] = n; cnt[tid] = 0; thr[tid] = 0ull;
    }
    const int ucnt = a.mode == 1 ? ((const int*)a.ucnt)[b] : 0;
    const int t_lo = t_base + (int)((long long)c * n_tiles / geff), t_hi = t_base + (int)((long long)(c + 1) * n_tiles / geff);
    const int count = (t_hi - t_lo + TG - 1) / TG;   // super-tiles of TG tiles, newest first
    for (int k = 0; k < STAGES - 1; k++) {
        if (k < count) d41_idx_issue<R, STAGES, TG, CAP>(a, rb, b, ucnt, sm, k, k, t_lo, t_hi);
        else d41_cp_commit();
    }
    d41_idx_prep<R, STAGES, TG, CAP>(a, rb, sm);
    __syncthreads();
    D41_IDX_STAMP(1);

    const unsigned* qf = (const unsigned*)(sm + L::QF);
    const unsigned short* qs = (const unsigned short*)(sm + L::QS);
    const unsigned* ws = (const unsigned*)(sm + L::WS);
    const unsigned* ent = (const unsigned*)(sm + L::ENT);

    for (int it = 0; it < count; it++) {
        const int s = it % STAGES;
        const int nx = it + STAGES - 1;
        if (nx < count) d41_idx_issue<R, STAGES, TG, CAP>(a, rb, b, ucnt, sm, nx % STAGES, nx, t_lo, t_hi);
        else d41_cp_commit();
        d41_cp_wait<STAGES - 1>();
        __syncthreads();
        const int ti = t_hi - 1 - it * TG - tg;   // this warp group's tile
        if (ti >= t_lo) {
            const unsigned char* kqs = sm + L::KQ + (s * L::KT + tg * D41_IDX_TILE) * 64;
            const unsigned char* kss = sm + L::KS + (s * L::KT + tg * D41_IDX_TILE) * 4;
            unsigned af[2][2][4], sfa[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const int r0 = 32 * kw + 16 * mt + g;
#pragma unroll
                for (int ks = 0; ks < 2; ks++) {
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const int ch = 2 * ks + hh;
                        af[mt][ks][hh * 2] = *(const unsigned*)(kqs + r0 * 64 + ((ch ^ ((r0 >> 1) & 3)) << 4) + 4 * t);
                        af[mt][ks][hh * 2 + 1] = *(const unsigned*)(kqs + (r0 + 8) * 64 + ((ch ^ (((r0 + 8) >> 1) & 3)) << 4) + 4 * t);
                    }
                    sfa[mt][ks] = t <= 1 ? (unsigned)*(const unsigned short*)(kss + (r0 + 8 * t) * 4 + 2 * ks) : 0u;
                }
            }
            // this lane's key after the butterfly: tile key 32 kw + 8t + g = block slot 4 kw + t, offset g
            const int slot = 4 * kw + t;
            long long p;
            unsigned rmask = 0xFFFFFFFFu;
            bool inb;
            if (a.mode == 0) { p = (long long)ti * D41_IDX_TILE + 32 * kw + 8 * t + g; inb = true; }
            else {
                const int ei = ti * 16 + slot;
                const unsigned e = ent[s * TG * 16 + tg * 16 + slot];
                inb = ei < ucnt;
                p = (long long)(e >> 8) * D41_IDX_CAND_BLOCK + g;
                rmask = e & 0xFFu;
            }
            for (int r = rg; r < rb.nrows; r += RG) {
                float d[2][4][4];
#pragma unroll
                for (int mt = 0; mt < 2; mt++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++)
#pragma unroll
                        for (int k = 0; k < 4; k++) d[mt][nt][k] = 0.f;
#pragma unroll
                for (int ks = 0; ks < 2; ks++) {
                    const uint4 b0 = *(const uint4*)(qf + (((r * 2 + ks) * 2 + 0) * 32 + lane) * 4);
                    const uint4 b1 = *(const uint4*)(qf + (((r * 2 + ks) * 2 + 1) * 32 + lane) * 4);
                    uint2 sbv = make_uint2(0, 0);
                    if (t == 0) sbv = *(const uint2*)(qs + ((r * 2 + ks) * 8 + g) * 4);
                    const unsigned sfb[4] = {sbv.x & 0xFFFFu, sbv.x >> 16, sbv.y & 0xFFFFu, sbv.y >> 16};
                    const unsigned bb0[4] = {b0.x, b0.y, b0.z, b0.w}, bb1[4] = {b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                    for (int nt = 0; nt < 4; nt++)
#pragma unroll
                        for (int mt = 0; mt < 2; mt++)
                            d41_mma_mxf4(af[mt][ks][0], af[mt][ks][1], af[mt][ks][2], af[mt][ks][3], bb0[nt], bb1[nt],
                                         sfa[mt][ks], sfb[nt], d[mt][nt][0], d[mt][nt][1], d[mt][nt][2], d[mt][nt][3]);
                }
                const uint4 wv = *(const uint4*)(ws + (r * 4 + t) * 4);
                const unsigned w2u[4] = {wv.x, wv.y, wv.z, wv.w};
                float pk[4];
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    const int mt = kk >> 1, c0 = (kk & 1) * 2;
                    float acc = 0.f;
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        __nv_bfloat162 v = __floats2bfloat162_rn(d[mt][nt][c0], d[mt][nt][c0 + 1]);
                        v = __hmax2(v, __float2bfloat162_rn(0.f));
                        v = __hmul2(v, *(const __nv_bfloat162*)&w2u[nt]);
                        acc = __fadd_rn(acc, __low2float(v));
                        acc = __fadd_rn(acc, __high2float(v));
                    }
                    pk[kk] = acc;
                }
                // transpose-reduce over the quad: lane t ends with key t = (p0 + p1) + (p2 + p3)
                const float s1 = (t & 1) ? pk[0] : pk[1], s2 = (t & 1) ? pk[2] : pk[3];
                const float r1 = __shfl_xor_sync(0xFFFFFFFFu, s1, 1), r2 = __shfl_xor_sync(0xFFFFFFFFu, s2, 1);
                const float qa = (t & 1) ? __fadd_rn(pk[1], r1) : __fadd_rn(pk[0], r1);
                const float qb = (t & 1) ? __fadd_rn(pk[3], r2) : __fadd_rn(pk[2], r2);
                const float sx = (t & 2) ? qa : qb;
                const float rx = __shfl_xor_sync(0xFFFFFFFFu, sx, 2);
                const float fin = (t & 2) ? __fadd_rn(qb, rx) : __fadd_rn(qa, rx);
                const unsigned s16 = d41_idx_sortable(__bfloat16_as_ushort(__float2bfloat16_rn(fin)));
                const bool vis = inb && p < nv[r] && ((rmask >> r) & 1u);
                const unsigned long long key = d41_idx_key(s16, (unsigned)p);
                const bool app = vis && key > thr[r];
                const unsigned m = __ballot_sync(0xFFFFFFFFu, app);
                if (m) {
                    const int leader = __ffs(m) - 1;
                    int base = 0;
                    if (lane == leader) base = atomicAdd(&cnt[r], __popc(m));
                    base = __shfl_sync(0xFFFFFFFFu, base, leader);
                    if (app) buf[r * CAP + base + __popc(m & ((1u << lane) - 1u))] = key;
                }
                if (a.write_bmax) {
                    unsigned bm = vis ? s16 : 0u;
                    bm = max(bm, __shfl_xor_sync(0xFFFFFFFFu, bm, 4));
                    bm = max(bm, __shfl_xor_sync(0xFFFFFFFFu, bm, 8));
                    bm = max(bm, __shfl_xor_sync(0xFFFFFFFFu, bm, 16));
                    const long long blk = (long long)ti * 16 + slot;
                    if (g == 0 && a.mode == 0 && blk < a.nb_cap)
                        ((unsigned short*)a.bmax)[(long long)(rb.row0 + r) * a.nb_cap + blk] = (unsigned short)bm;
                }
                if (a.dbg && inb && p < a.dbg_cap)
                    ((unsigned short*)a.dbg)[(long long)(rb.row0 + r) * a.dbg_cap + p] = (unsigned short)(vis ? s16 : 0u);
            }
        }
        __syncthreads();
        // Stage s is now completely consumed and idle until iteration it + STAGES (in-flight cp.async only targets stages != s).
        unsigned char* s_scr = sm + L::KQ + s * L::KT * 64;
        if (R > 1 && rb.nrows > 1 && it + 1 < count) {
            // keep each full row's best 512 (radix select, no sort): the full rows in parallel, one thread group per row (PK1)
            bool any = false;
            for (int r = 0; r < rb.nrows; r++) any |= cnt[r] > CAP - L::KT;   // uniform: cnt is stable since the barrier
            if (any) {
                const int gq = tid / (NT / R);
                const d41_idx_grp grp{tid % (NT / R), NT / R, 1 + gq};
                if (gq < rb.nrows && cnt[gq] > CAP - L::KT) {
                    int* gsc = (int*)(s_scr + gq * 576);
                    unsigned long long* v = buf + gq * CAP;
                    const int n = cnt[gq];
                    const unsigned long long k512 = d41_idx_kth<true>(v, n, D41_IDX_TOPK, gsc, grp);
                    d41_idx_keep_ge<(CAP * R + NT - 1) / NT>(v, n, k512, gsc, grp);
                    if (grp.tid == 0) { cnt[gq] = D41_IDX_TOPK; thr[gq] = k512; }
                    D41_IDX_COUNT(4);
                }
                __syncthreads();
            }
        } else {
            int* sc = (int*)s_scr;
            for (int r = 0; r < rb.nrows && it + 1 < count; r++) {   // (the last super-tile: the finish selects)
                const int n = cnt[r];
                if (n > CAP - L::KT) {   // keep the row's best 512: radix select, no sort
                    unsigned long long* v = buf + r * CAP;
                    const unsigned long long k512 = d41_idx_kth(v, n, D41_IDX_TOPK, sc);
                    d41_idx_keep_ge<(CAP + NT - 1) / NT>(v, n, k512, sc);
                    if (tid == 0) { cnt[r] = D41_IDX_TOPK; thr[r] = k512; }
                    D41_IDX_COUNT(4);
                    __syncthreads();
                }
            }
        }
    }
    d41_cp_wait<0>();
    __syncthreads();
    D41_IDX_STAMP(2);
    // finish: the chunk's top <= 512 per row. Several rows: one warp-aligned thread group per row, in parallel (PK1), each
    // with its own named barrier and its own scratch in the idle key stages / query fragments (the same operations per row
    // as the whole-CTA path, so the same selection); one row: the whole CTA.
    const bool final_here = geff == 1 && !stripe;
    const bool sorted_lists = geff > D41_IDX_SMALL_G || stripe;   // the merge binary-searches sorted lists
    const int ngrp = (R > 1 && rb.nrows > 1) ? R : 1;
    constexpr int GSPAN = R > 1 ? ((L::ENT - L::KQ) / R) / 16 * 16 : 0;   // bytes of one group's scratch
    static_assert(R == 1 || GSPAN >= 1088 + (512 + 16) * 4, "group scratch does not fit the idle stages");
    static_assert(L::ENT - L::KQ >= 1088 + (512 + 16) * 4, "1-group scratch does not fit the idle stages");
    const int gq = tid / (NT / ngrp);                                // this thread's group = its row
    const d41_idx_grp grp = ngrp == 1 ? d41_idx_cta() : d41_idx_grp{tid % (NT / ngrp), NT / ngrp, 1 + gq};
    int* gsc = ngrp == 1 ? (int*)(sm + L::KQ) : (int*)(sm + L::KQ + gq * GSPAN);
    unsigned* tmp = ngrp == 1 ? (unsigned*)(sm + L::KQ + 1088) : (unsigned*)(sm + L::KQ + gq * GSPAN + 1088);
    const int tmp_words = ngrp == 1 ? (L::ENT - L::KQ - 1088) / 4 : (GSPAN - 1088) / 4;
    for (int r = ngrp == 1 ? 0 : gq; r < rb.nrows; r += ngrp) {
        const int n0 = cnt[r];
        unsigned long long* v = buf + r * CAP;
        int n = n0;
        if (n0 > D41_IDX_TOPK) {
            const unsigned long long k512 = d41_idx_kth(v, n0, D41_IDX_TOPK, gsc, grp);
            if (ngrp == 1) d41_idx_keep_ge<(CAP + NT - 1) / NT>(v, n0, k512, gsc, grp);
            else d41_idx_keep_ge<(CAP * R + NT - 1) / NT>(v, n0, k512, gsc, grp);
            n = D41_IDX_TOPK;
        }
        if (final_here) {
            d41_idx_write_final(a, rb.row0 + r, v, n, tmp, tmp_words, nv[r], grp);
        } else {
            if (sorted_lists) {
                for (int i = n + grp.tid; i < D41_IDX_TOPK; i += grp.nt) v[i] = 0ull;
                d41_idx_gsync(grp);
                d41_idx_sort_desc<D41_IDX_TOPK>(v, grp);
            }
            const long long li = ((long long)b * a.R + r) * lstride + a.list_off + c;
            unsigned long long* out = (unsigned long long*)a.part + li * D41_IDX_TOPK;
            for (int i = grp.tid; i < n; i += grp.nt) out[i] = v[i];
            if (grp.tid == 0) ((int*)a.pcnt)[li] = n;
        }
    }
    if (ngrp > 1) __syncthreads();
    D41_IDX_STAMP(3);
    // fused merge: the last chunk CTA of the row block merges every row (threadfence reduction)
    if (!final_here && !stripe && a.tick) {
        __threadfence();
        __syncthreads();
        int* misc = (int*)(sm + L::KQ);
        if (tid == 0) misc[0] = atomicAdd((int*)a.tick + b, 1) == geff - 1;
        __syncthreads();
        if (misc[0]) {
            __threadfence();
            d41_idx_msm* m = (d41_idx_msm*)(sm + L::BUF);
            for (int r = 0; r < rb.nrows; r++) d41_idx_merge_row<NT>(a, rb.row0 + r, ((long long)b * a.R + r) * lstride, geff, m);
            if (tid == 0) ((int*)a.tick)[b] = 0;
            D41_IDX_STAMP(5);
        }
    }
}

// R rows | STAGES | RG row groups | TG tile groups | CAP candidate buffer per row
// r1 runs 256 threads at 2 CTAs/SM (49,616 B <= 50,176 B -> 96 concurrent CTAs, 4 SM-level stages in flight);
// r2 / r4 run 3-stage cp.async rings; r8 runs 2-stage cp.async ring at full CAP=1024 (101,120 B).
extern "C" __global__ void __launch_bounds__(256, 2) d41_idx_score_r1(const d41_idx_args a) { D41_PDL_ENTRY(); d41_idx_score_body<1, 2, 1, 2, 1536>(a); }
extern "C" __global__ void __launch_bounds__(512) d41_idx_score_r2(const d41_idx_args a) { D41_PDL_ENTRY(); d41_idx_score_body<2, 3, 2, 2, 1024>(a); }
extern "C" __global__ void __launch_bounds__(512) d41_idx_score_r4(const d41_idx_args a) { D41_PDL_ENTRY(); d41_idx_score_body<4, 3, 4, 1, 1024>(a); }
extern "C" __global__ void __launch_bounds__(512) d41_idx_score_r8(const d41_idx_args a) { D41_PDL_ENTRY(); d41_idx_score_body<8, 2, 4, 1, 1024>(a); }

// Sizes the host must agree with (checked by the ABI probe): args, rb, dynamic smem of r1/r2/r4/r8,
// and the threads per CTA of r1/r2/r4/r8.
extern "C" __global__ void d41_idx_abi(int* out) {
    out[0] = (int)sizeof(d41_idx_args);
    out[1] = (int)sizeof(d41_idx_rb);
    out[2] = d41_idx_smem<1, 2, 2, 1536>::TOTAL;
    out[3] = d41_idx_smem<2, 3, 2, 1024>::TOTAL;
    out[4] = d41_idx_smem<4, 3, 1, 1024>::TOTAL;
    out[5] = d41_idx_smem<8, 2, 1, 1024>::TOTAL;
    out[6] = 256; out[7] = 512; out[8] = 512; out[9] = 512;
}
