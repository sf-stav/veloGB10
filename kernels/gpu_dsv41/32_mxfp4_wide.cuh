// =============================================================================
// kernels/gpu_dsv41/32_mxfp4_wide.cuh — band 30 (leg KN2): routed experts for prefill chunks
// (> 32 rows). Same arithmetic contract as 30 / 31 (one zero-accumulator block-scaled MMA per
// k-block, FP32 k-block sum in ascending order), so a row's result is bit-identical to the grouped
// kernels' and to a 1-row decode call: chunked prefill = single shot, bit for bit.
//
//   d41_mx_plan_w     one CTA: per local expert route counts, offsets, the route lists, and the
//                     token-tile list (expert, first route) with TT = 128 routes per tile
//   d41_mx_fc1_w      CTA = (token tile, 96 inner outputs): 12 warps x 1 m-tile (8 gate + 8 up rows)
//                     x 16 n-tiles of 8 tokens; the tile's activations are staged in shared memory
//                     (cp.async, two quad stages); the weights stream from global (3-quad register ring).
//                     Epilogue = the decode one: SwiGLU, routing weight, bf16, act_quant per 32 -> hq / hs
//   d41_mx_fc2_w      CTA = (token tile, 128 output columns): 8 warps x 1 m-tile; w2 on the tile's
//                     routes -> out2 [route][dim]: bf16 values for whole experts (the reference's
//                     per-expert rounding, half the traffic), f32 partials for an inner slice
// PK2 (2026-11-19): 128-route tiles, one m-tile per warp (64 accumulators: 12 / 16 warps per SM instead of
// 6 / 8), a 3-quad weight ring. The per-output arithmetic is unchanged (bit-identical, --check-wide).
//   d41_mx_combine_w  y[row] (+)= sum of the row's local routes in ascending expert id
// Order of routes inside an expert's list does not matter: every MMA column (token) is independent.
// =============================================================================
#pragma once

#define D41_MX_TT 128                  // routes per token tile (PK2: 128, was 64: half the weight passes per expert)
#define D41_MX_TT64_MAX_RPE 48         // PK3: 64-route tiles while the mean routes per expert (rows x top-k / experts) is at most this
                                       // (mxfp4_launch.rs WIDE_TT64_MAX_RPE mirrors it; crossover measured on .13 between 3K and 4K rows
                                       // of 384 x top-6, PLAN/DSV41_PK3_REPORT.md)
#define D41_MX_BROW 160                // bytes per staged token row: 4 k-blocks x 32 + 32 pad (PK2: rows 8 banks apart,
                                       // the 8-byte B loads of a half-warp hit 32 distinct banks)

struct D41MxPlanWArgs {
    const int* ids;                    // [rows * top_k]
    int* cnt;                          // [n_local]
    int* off;                          // [n_local + 1]
    int* cur;                          // [n_local] scratch cursors
    int* list;                         // [rows * top_k] route ids grouped by expert
    int* tile_e;                       // [max_tiles] local expert of each tile
    int* tile_s;                       // [max_tiles] first index in `list`
    int* n_tiles;                      // [1]
    int rows, top_k, e_first, n_local;
    int max_tiles, tt;                 // PK3: tt = routes per token tile (64 or 128; 0 = D41_MX_TT)
};

struct D41MxWideArgs {
    const uint8_t* w;                  // packed (mx1) base, local expert 0
    unsigned long long e_stride, mt_stride;
    const uint8_t* aq;                 // FC1: xq [rows][lda]; FC2: hq [route][lda]
    const uint8_t* as;                 // FC1: xs [rows][ldas]; FC2: hs [route][ldas]
    const int* list; const int* off; const int* tile_e; const int* tile_s; const int* n_tiles;
    const float* rw;                   // FC1: routing weights [route]
    uint8_t* hq; uint8_t* hs;          // FC1 outputs [route][ldh], [route][ldhs]
    float* out2;                       // FC2 output [route][dim]
    int top_k, nkb, slice, dim;
    int lda, ldas, ldh, ldhs;
    int fc1, round_bf16;
    float limit;
    int pad_;
};
static_assert(sizeof(D41MxPlanWArgs) == 88, "mirrors mxfp4_launch.rs PlanWArgs");
static_assert(sizeof(D41MxWideArgs) == 160, "mirrors mxfp4_launch.rs WideArgs");

// ---- route plan for a chunk ---------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(1024) d41_mx_plan_w(const D41MxPlanWArgs p) {
    D41_PDL_ENTRY();
    __shared__ int scnt[512];
    __shared__ int soff[513];
    const int tid = threadIdx.x, nr = p.rows * p.top_k;
    for (int e = tid; e < p.n_local; e += blockDim.x) scnt[e] = 0;
    __syncthreads();
    for (int r = tid; r < nr; r += blockDim.x) {
        const int e = p.ids[r] - p.e_first;
        if (e >= 0 && e < p.n_local) atomicAdd(&scnt[e], 1);
    }
    __syncthreads();
    if (tid == 0) {   // n_local <= 512: a serial scan is a few microseconds
        const int tt = p.tt > 0 ? p.tt : D41_MX_TT;
        int s = 0, t = 0;
        for (int e = 0; e < p.n_local; e++) {
            soff[e] = s;
            const int c = scnt[e];
            for (int k = 0; k < c; k += tt) { if (t < p.max_tiles) { p.tile_e[t] = e; p.tile_s[t] = s + k; } t++; }
            s += c;
        }
        soff[p.n_local] = s;
        *p.n_tiles = t < p.max_tiles ? t : p.max_tiles;
    }
    __syncthreads();
    for (int e = tid; e < p.n_local; e += blockDim.x) { p.cnt[e] = scnt[e]; p.off[e] = soff[e]; scnt[e] = soff[e]; }
    if (tid == 0) p.off[p.n_local] = soff[p.n_local];
    __syncthreads();
    for (int r = tid; r < nr; r += blockDim.x) {
        const int e = p.ids[r] - p.e_first;
        if (e >= 0 && e < p.n_local) p.list[atomicAdd(&scnt[e], 1)] = r;
    }
}

__device__ __forceinline__ void d41_mx_cp16(void* smem, const void* g, bool ok) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void d41_mx_cp4(void* smem, const void* g, bool ok) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    const int n = ok ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(sa), "l"(g), "r"(n));
}

// ---- FC1 / FC2 on a token tile ----------------------------------------------------------------
// stage the tile's activations of quad kq: TT rows x (up to) 128 B + TT x u32 scales
// NTH = the CTA's thread count (PK3: a compile-time stride; a blockDim stride made ptxas divide for the trip count and keep it)
template <int TT, int NTH>
__device__ __forceinline__ void d41_mx_stage(uint8_t* sb, uint32_t* ss, const D41MxWideArgs& p, const int* srow, int c, int kq) {
    const int nbytes = min(128, (p.nkb - 4 * kq) * 32);    // valid bytes of this quad
#pragma unroll
    for (int i = threadIdx.x; i < TT * 8; i += NTH) {
        const int tok = i >> 3, ch = i & 7;
        const bool ok = tok < c && ch * 16 < nbytes;
        const uint8_t* src = p.aq + (size_t)(ok ? srow[tok] : 0) * p.lda + kq * 128 + ch * 16;
        d41_mx_cp16(sb + tok * D41_MX_BROW + ch * 16, ok ? src : p.aq, ok);
    }
#pragma unroll
    for (int tok = threadIdx.x; tok < TT; tok += NTH) {
        const bool ok = tok < c;
        d41_mx_cp4(ss + tok, ok ? p.as + (size_t)srow[tok] * p.ldas + 4 * kq : p.as, ok);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// CTA = NW warps x one m-tile each (FC1: 12 m-tiles = 96 inner values = 3 act_quant groups, so every
// slice width 2304 / 1152 / 576 / 288 divides; FC2: 8 m-tiles = 128 output columns, two CTAs per SM) x
// TT = 128 routes (16 n-tiles). One staged activation tile serves all NW m-tiles; each warp streams its
// m-tile's weights through a 3-quad register ring (loads for quad q + 2 issued while quad q is computed).
// Activations (FC1: xq from d41_mx_act_quant_p; FC2: hq from FC1's epilogue) are stored k-permuted inside
// each 32-byte k-block (d41_mx_kperm): one 8-byte shared load per B fragment. FULL = all 128 routes live.
template <int NW, int WPM, int TT, int RG, bool FC1, bool FULL>
__device__ __forceinline__ void d41_mx_wide_tile(const D41MxWideArgs& p, int e, int s0, int c, int q, uint8_t (*sbuf)[TT * D41_MX_BROW],
                                                 uint32_t (*sscl)[TT], const int* srow, const int* sroute) {
    constexpr int NT = TT / 8 / WPM;          // n-tiles per warp (WPM warps share an m-tile, each a slice of the routes)
    constexpr int MT = NW / WPM;              // m-tiles per CTA
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int mw = warp / WPM, th = warp % WPM;   // m-tile of this warp, its route slice
    const int j0 = q * MT + mw;               // this warp's m-tile
    const int n0 = th * NT;                   // first n-tile of the slice
    const uint8_t* A = p.w + (size_t)e * p.e_stride + (size_t)j0 * p.mt_stride;
    const uint8_t* As = A + (size_t)p.nkb * D41_MX_CODES_UNIT;
    const int nkq = (p.nkb + 3) >> 2;
    const int ntv = FULL ? NT : min(NT, max(0, ((c + 7) >> 3) - n0));   // live n-tiles of the slice
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[nt][i] = 0.0f;

    d41_mx_stage<TT, NW * 32>(sbuf[0], sscl[0], p, srow, c, 0);
    uint2 ra[RG][4];                          // RG-quad weight register ring (PK3: 2 or 3)
    uint32_t rsa[RG];
    auto load_a = [&](int kq, uint2 (&a)[4], uint32_t& sa) {
        if (kq >= nkq) return;
        sa = d41_mx_ldw32(As + kq * D41_MX_SCALES_UNIT + (2 * g + (t & 1)) * 4);
#pragma unroll
        for (int v = 0; v < 4; v++) {
            const int kb = 4 * kq + v;
            a[v] = kb < p.nkb ? d41_mx_ldw(A + (size_t)kb * D41_MX_CODES_UNIT + lane * 8) : make_uint2(0u, 0u);
        }
    };
    // PK3: n-tile outer, k-block inner (was k-block outer): the quad's 4 k-blocks of A decoded once, one token-scale word per
    // n-tile. Every output still adds its k-blocks in ascending order (v = 0..3 within the quad): the same bits.
    auto compute = [&](int kq, const uint8_t* sb, const uint32_t* ss, const uint2 (&a)[4], uint32_t sa) {
        const int nv = min(4, p.nkb - 4 * kq);
        uint32_t ad[4][4];
#pragma unroll
        for (int v = 0; v < 4; v++) { ad[v][0] = d41_mx_lo4(a[v].x); ad[v][1] = d41_mx_hi4(a[v].x); ad[v][2] = d41_mx_lo4(a[v].y); ad[v][3] = d41_mx_hi4(a[v].y); }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            if (!FULL && nt >= ntv) break;
            const uint32_t sbw = ss[(n0 + nt) * 8 + g];
            const uint8_t* brow = sb + ((n0 + nt) * 8 + g) * D41_MX_BROW + 8 * t;
#pragma unroll
            for (int v = 0; v < 4; v++) {
                if (v >= nv) break;
                const uint2 bb = *(const uint2*)(brow + v * 32);
                float d[4];
                d41_mx_mma_v(v, d, ad[v][0], ad[v][1], ad[v][2], ad[v][3], bb.x, bb.y, sa, sbw);
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][i] += d[i];
            }
        }
    };
    load_a(0, ra[0], rsa[0]);
    if constexpr (RG == 3) load_a(1, ra[1], rsa[1]);
    // quad kq: smem stage kq & 1, ring slot kq % RG (unrolled by lcm(2, RG) so both indices are compile-time)
    auto step = [&](int kq, int sbi, uint2 (&a)[4], uint32_t sa, uint2 (&an)[4], uint32_t& san) {
        if (kq + 1 < nkq) { d41_mx_stage<TT, NW * 32>(sbuf[sbi ^ 1], sscl[sbi ^ 1], p, srow, c, kq + 1); asm volatile("cp.async.wait_group 1;" ::: "memory"); }
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        load_a(kq + RG - 1, an, san);
        __syncthreads();
        compute(kq, sbuf[sbi], sscl[sbi], a, sa);
        __syncthreads();
    };
    if constexpr (RG == 3) {
        for (int kq = 0; kq < nkq; kq += 6) {
            step(kq, 0, ra[0], rsa[0], ra[2], rsa[2]);
            if (kq + 1 < nkq) step(kq + 1, 1, ra[1], rsa[1], ra[0], rsa[0]);
            if (kq + 2 < nkq) step(kq + 2, 0, ra[2], rsa[2], ra[1], rsa[1]);
            if (kq + 3 < nkq) step(kq + 3, 1, ra[0], rsa[0], ra[2], rsa[2]);
            if (kq + 4 < nkq) step(kq + 4, 0, ra[1], rsa[1], ra[0], rsa[0]);
            if (kq + 5 < nkq) step(kq + 5, 1, ra[2], rsa[2], ra[1], rsa[1]);
        }
    } else {
        for (int kq = 0; kq < nkq; kq += 2) {
            step(kq, 0, ra[0], rsa[0], ra[1], rsa[1]);
            if (kq + 1 < nkq) step(kq + 1, 1, ra[1], rsa[1], ra[0], rsa[0]);
        }
    }

    // PK3: q and this warp's m-tile recomputed from %ctaid.x (a volatile read ptxas cannot merge with the prologue's) so they are
    // not held, or spilled, across the main loop; the values are the prologue's
    {
        unsigned bx;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(bx));
        const unsigned ng = FC1 ? (unsigned)(p.slice / (8 * MT)) : (unsigned)(p.dim / (16 * MT));
        q = (int)(bx % ng);
    }
    const int j0e = q * MT + mw;
    if (FC1) {
        // SwiGLU + routing weight + bf16; act_quant per 32 inner values = 4 m-tiles = 4 warps
        float (*red)[TT] = (float (*)[TT])&sbuf[0][0];   // the stages are idle after the last barrier
        float hb[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int s = (n0 + nt) * 8 + 2 * t + i;
                const bool ok = s < c;
                const float w = ok ? p.rw[sroute[s]] : 0.0f;
                const float gb = d41_mx_bf16r(acc[nt][i]), ub = d41_mx_bf16r(acc[nt][2 + i]);
                const float h = ok ? d41_mx_swiglu(gb, ub, w, p.limit) : 0.0f;
                hb[nt][i] = h;
                float am = fabsf(h);
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
                if (g == 0) red[mw][s] = am;
            }
        }
        __syncthreads();
        const int grp = q * (MT / 4) + mw / 4;        // global act_quant group index of this warp's group
        const int kin = j0e * 8 + g;                  // inner index = FC2's k: stored k-permuted in its 32-block
        const int hpos = (kin & ~31) + d41_mx_kperm(kin & 31);
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int s = (n0 + nt) * 8 + 2 * t + i;
                if (s < c) {
                    const int r = sroute[s];
                    const int w0 = (mw / 4) * 4;
                    float am = red[w0][s];
#pragma unroll
                    for (int k = 1; k < 4; k++) am = fmaxf(am, red[w0 + k][s]);
                    am = fmaxf(am, 1e-4f);
                    const float sc = d41_mx_pow2_scale(am);
                    p.hq[(size_t)r * p.ldh + hpos] = d41_mx_e4m3(hb[nt][i], sc);
                    if (mw % 4 == 0 && g == 0) p.hs[(size_t)r * p.ldhs + grp] = d41_mx_e8m0_of(sc);
                }
            }
        }
    } else {
        // per-route outputs: bf16 values (whole experts, the reference's per-expert rounding) or f32 partials
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int s = (n0 + nt) * 8 + 2 * t + (i & 1);
                if (s < c) {
                    const int col = j0e * 16 + g + 8 * (i >> 1);
                    const size_t at = (size_t)sroute[s] * p.dim + col;
                    if (p.round_bf16) ((__nv_bfloat16*)p.out2)[at] = __float2bfloat16(acc[nt][i]);
                    else p.out2[at] = acc[nt][i];
                }
            }
        }
    }
}
template <int NW, int WPM, int TT, int RG, bool FC1, bool GF>
__device__ __forceinline__ void d41_mx_wide_body(const D41MxWideArgs& p) {
    __shared__ __align__(16) uint8_t sbuf[2][TT * D41_MX_BROW];
    __shared__ uint32_t sscl[2][TT];
    __shared__ int srow[TT];              // FC1: activation row of each token; FC2: route id
    __shared__ int sroute[TT];
    static_assert(!FC1 || (NW / WPM) * TT * 4 <= 2 * TT * D41_MX_BROW, "act_quant amax table aliases the stages");
    const int ngroups = FC1 ? (p.slice / (8 * (NW / WPM))) : (p.dim / (16 * (NW / WPM)));
    const int tile = blockIdx.x / ngroups, q = blockIdx.x - tile * ngroups;
    if (tile >= *p.n_tiles) return;
    const int e = p.tile_e[tile], s0 = p.tile_s[tile];
    const int c = min(TT, p.off[e + 1] - s0);
    for (int i = threadIdx.x; i < TT; i += blockDim.x) {
        const int r = i < c ? p.list[s0 + i] : 0;
        sroute[i] = r;
        srow[i] = FC1 ? r / p.top_k : r;
    }
    __syncthreads();
    if (GF && c == TT) d41_mx_wide_tile<NW, WPM, TT, RG, FC1, true>(p, e, s0, c, q, sbuf, sscl, srow, sroute);
    else d41_mx_wide_tile<NW, WPM, TT, RG, FC1, false>(p, e, s0, c, q, sbuf, sscl, srow, sroute);
}

// CTA shapes (FC1: NW m-tiles must divide every slice's m-tile count 36 / 72 / 144 / 288 and be a
// multiple of 4 (act_quant groups): 4, 12 or 36; FC2: must divide 320)
#define D41_MX_FC1W_WPM 1                // FC1: warps per m-tile (2 = 64 routes each: 24 warps at 80 registers spilled and lost, PK2)
#define D41_MX_FC1W_WARPS (12 * D41_MX_FC1W_WPM)
#define D41_MX_FC2W_WPM 1
#define D41_MX_FC2W_WARPS (8 * D41_MX_FC2W_WPM)
#define D41_MX_FC1W_INNER (12 * 8)
#ifndef D41_MX_FC1W_GF
#define D41_MX_FC1W_GF 1                 // a guard-free instantiation for full token tiles (PK2)
#endif
#ifndef D41_MX_FC2W_GF
#define D41_MX_FC2W_GF 0                 // PK3: FC2 without the guard-free instantiation (it spilled at the 128-register cap; measured
                                         // as fast or faster with the 2-quad ring, PLAN/DSV41_PK3_REPORT.md)
#endif
#ifndef D41_MX_FC1W_RING
#define D41_MX_FC1W_RING 3               // weight register ring depth in quads (FC1: K = 5120, 40 quads)
#endif
#ifndef D41_MX_FC2W_RING
#define D41_MX_FC2W_RING 2               // PK3: 2 for FC2 (K = the slice, 4.5 quads at TP4; 16 warps per SM; the 3-quad ring spilled)
#endif
#define D41_MX_FC2W_COLS (8 * 16)
extern "C" __global__ void __launch_bounds__(D41_MX_FC1W_WARPS * 32, 1) d41_mx_fc1_w(const D41MxWideArgs p) {
    D41_PDL_ENTRY();
    d41_mx_wide_body<D41_MX_FC1W_WARPS, D41_MX_FC1W_WPM, D41_MX_TT, D41_MX_FC1W_RING, true, D41_MX_FC1W_GF>(p);
}
extern "C" __global__ void __launch_bounds__(D41_MX_FC2W_WARPS * 32, 2) d41_mx_fc2_w(const D41MxWideArgs p) {
    D41_PDL_ENTRY();
    d41_mx_wide_body<D41_MX_FC2W_WARPS, D41_MX_FC2W_WPM, D41_MX_TT, D41_MX_FC2W_RING, false, D41_MX_FC2W_GF>(p);
}
// PK3: 64-route token tiles for short prefill chunks (about 64 routes per expert or fewer: the 128-route tiles run mostly empty
// there). The per-output arithmetic is the 128-route kernels' (every MMA column is one token): the same bits.
#ifndef D41_MX_FC1W64_MINB
#define D41_MX_FC1W64_MINB 1
#endif
#ifndef D41_MX_FC2W64_MINB
#define D41_MX_FC2W64_MINB 2
#endif
extern "C" __global__ void __launch_bounds__(D41_MX_FC1W_WARPS * 32, D41_MX_FC1W64_MINB) d41_mx_fc1_w64(const D41MxWideArgs p) {
    D41_PDL_ENTRY();
    d41_mx_wide_body<D41_MX_FC1W_WARPS, D41_MX_FC1W_WPM, 64, D41_MX_FC1W_RING, true, D41_MX_FC1W_GF>(p);
}
extern "C" __global__ void __launch_bounds__(D41_MX_FC2W_WARPS * 32, D41_MX_FC2W64_MINB) d41_mx_fc2_w64(const D41MxWideArgs p) {
    D41_PDL_ENTRY();
    d41_mx_wide_body<D41_MX_FC2W_WARPS, D41_MX_FC2W_WPM, 64, D41_MX_FC2W_RING, false, D41_MX_FC2W_GF>(p);
}

// ---- ordered combine: y[row][col] (+)= sum over the row's local routes in ascending expert id ----
extern "C" __global__ void __launch_bounds__(256)
d41_mx_combine_w(const float* __restrict__ out2, const int* __restrict__ ids, float* __restrict__ y, int ldy,
                 int rows, int top_k, int dim, int e_first, int n_local, int accumulate, int out2_bf16) {
    D41_PDL_ENTRY();
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= dim) return;
    // rank of every local slot by expert id (distinct per row), then add in rank order; all
    // per-thread arrays are compile-time indexed (top_k <= 8)
    int eid[8];
    bool ok[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int e = k < top_k ? ids[row * top_k + k] : -1;
        eid[k] = e;
        ok[k] = k < top_k && e - e_first >= 0 && e - e_first < n_local;
    }
    int rank[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int rk = 0;
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) rk += (ok[k2] && eid[k2] < eid[k]);
        rank[k] = rk;
    }
    float acc = 0.0f;
#pragma unroll
    for (int rr = 0; rr < 8; rr++)
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (ok[k] && rank[k] == rr) {
                const size_t at = (size_t)(row * top_k + k) * dim + col;
                acc += out2_bf16 ? __bfloat162float(((const __nv_bfloat16*)out2)[at]) : out2[at];
            }
    float* yp = y + (size_t)row * ldy + col;
    *yp = accumulate ? *yp + acc : acc;
}
