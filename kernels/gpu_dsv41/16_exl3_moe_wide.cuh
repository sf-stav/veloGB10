// 16_exl3_moe_wide.cuh (leg KE2): routed-MoE PREFILL (> 32 rows): grouped, weight-stationary, tensor cores.
//   d41x_mp_plan1   per 256-row block: per-expert row masks -> block histogram + each pair's rank in its block
//   d41x_mp_plan2   one CTA: per-expert block bases, pair offsets, the slot list (shared expert first) and tile lists
//   d41x_mp_plan3   per 256-row block: pair index of every (row, k), the pair's row and weight
//   d41x_mp_gu      one CTA per tile (slot, 128-row tile, 128-block, matrix): the input transform H128(x * suh) fused
//                   into the A staging, the trellis of the block decoded per warp from an smem stage, m16n8k16 MMAs;
//                   the second matrix of a (slot, row tile, block) pair to finish runs SwiGLU + w2's input -> xh2
//   d41x_mp_dn      one CTA per tile (slot, 128-row tile, output 128-block): w2 over the slice, H128 -> h2
//   d41x_mp_combine y[row] = (y + routed experts in ascending id) + shared expert, from h2
// Every row is BIT-IDENTICAL to the decode kernels' (15_*) and the generic path's: the same A values (the 8-lane
// Hadamard below has d41x_had128's butterfly tree), one MMA chain per 8-step k-block added in f32 in ascending order,
// the same pinned act / H128 / combine math. The probe gates prefill rows against decode rows bit for bit.
// Pairs: [0, rows) = the shared expert's rows (when given), then the routed pairs grouped by ascending expert id,
// ascending row inside an expert (deterministic; no atomics decide an order).
#pragma once

#define D41X_MP_RB 256           // plan row block
#define D41X_MP_TM 128           // rows per down / combine tile
#define D41X_MP_TMG 64           // PK2: rows per gate / up and down tile
#define D41X_MP_DN_NB 1          // PK2: output blocks per down tile (1: 256 threads, 2 CTAs per SM; 2: 512 threads, 1 CTA)
#define D41X_MP_MAXSL 1025       // slots: 1024 experts + the shared expert
// meta (int) layout
#define D41X_MP_SE 16
#define D41X_MP_SC (D41X_MP_SE + D41X_MP_MAXSL)
#define D41X_MP_SO (D41X_MP_SC + D41X_MP_MAXSL)
#define D41X_MP_SB (D41X_MP_SO + D41X_MP_MAXSL)
#define D41X_MP_GT (D41X_MP_SB + D41X_MP_MAXSL)
#define D41X_MP_DT (D41X_MP_GT + D41X_MP_MAXSL + 1)
#define D41X_MP_EO (D41X_MP_DT + D41X_MP_MAXSL + 1)
#define D41X_MP_META_INTS (D41X_MP_EO + 1024)

struct D41xMpArgs {
    const D41xExpert* ex; const D41xExpert* sh;
    const int* ids; const float* wts;
    const void* x; int xcode, ldx;
    int* hist; short* lrank; int* pos; int* prow; float* pw; int* meta; int* pcnt;
    int* gtab; int* dtab;   // PK2: slot of every gate / up and down tile (plan3; no per-tile binary search in the GEMMs)
    float* graw; __half* xh2; float* h2;
    float* out; int ldo;
    float limit, limit_sh;
    int n_experts, rows, topk, dim, inter_ld, kmax, kmin, nrb, ncnt;   // SX1: kmax / kmin informational
    int h2bf, pad_;          // PK3: h2bf = 1 (--dsv41-prefill-bf16-partials on): h2 holds bf16(H128(w2 sum) * svh) per pair
};

__device__ __forceinline__ bool d41x_mp_held(const D41xMpArgs& a, int e) { return e >= 0 && e < a.n_experts && a.ex[e].nblk > 0; }

// ---- plan ------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) d41x_mp_plan1(D41xMpArgs a) {
    D41_PDL_ENTRY();
    extern __shared__ unsigned d41x_mp_bm[];                 // [n_experts][8] row masks of this 256-row block
    const int t = threadIdx.x, rb = blockIdx.x, r0 = rb * D41X_MP_RB;
    const int nr = min(D41X_MP_RB, a.rows - r0), P = nr * a.topk;
    for (int i = t; i < a.n_experts * 8; i += 256) d41x_mp_bm[i] = 0u;
    __syncthreads();
    for (int i = t; i < P; i += 256) {
        const int e = a.ids[(size_t)r0 * a.topk + i], rl = i / a.topk;
        if (d41x_mp_held(a, e)) atomicOr(&d41x_mp_bm[e * 8 + (rl >> 5)], 1u << (rl & 31));
    }
    __syncthreads();
    for (int e = t; e < a.n_experts; e += 256) {
        int c = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) c += __popc(d41x_mp_bm[e * 8 + w]);
        a.hist[(size_t)rb * a.n_experts + e] = c;
    }
    for (int i = t; i < P; i += 256) {
        const int e = a.ids[(size_t)r0 * a.topk + i], rl = i / a.topk;
        short rk = -1;
        if (d41x_mp_held(a, e)) {
            int c = __popc(d41x_mp_bm[e * 8 + (rl >> 5)] & ((1u << (rl & 31)) - 1u));
            for (int w = 0; w < (rl >> 5); ++w) c += __popc(d41x_mp_bm[e * 8 + w]);
            rk = (short)c;
        }
        a.lrank[(size_t)r0 * a.topk + i] = rk;
    }
}

// block-wide exclusive scan of 4 ints over 1024 threads; returns totals
__device__ __forceinline__ void d41x_scan4(int (&v)[4], int (&tot)[4], int (*ws)[4]) {
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    int inc[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        inc[q] = v[q];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xFFFFFFFFu, inc[q], o); if (lane >= o) inc[q] += y; }
        if (lane == 31) ws[wid][q] = inc[q];
    }
    __syncthreads();
    if (wid == 0) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int w = ws[lane][q], iw = w;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xFFFFFFFFu, iw, o); if (lane >= o) iw += y; }
            ws[lane][q] = iw - w;
            if (lane == 31) ws[32][q] = iw;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < 4; ++q) { v[q] = ws[wid][q] + inc[q] - v[q]; tot[q] = ws[33 - 1][q]; }
}

extern "C" __global__ void __launch_bounds__(1024) d41x_mp_plan2(D41xMpArgs a) {
    D41_PDL_ENTRY();
    __shared__ int ws[33][4];
    const int t = threadIdx.x, E = a.n_experts, NJ = a.dim >> 7;
    int c = 0;
    if (t < E) for (int rb = 0; rb < a.nrb; ++rb) { const int h = a.hist[(size_t)rb * E + t]; a.hist[(size_t)rb * E + t] = c; c += h; }
    const int nb = (t < E && c > 0) ? a.ex[t].nblk : 0;
    const int nrtg = (c + D41X_MP_TMG - 1) / D41X_MP_TMG;
    // gate / up: (64-row tile, block pair, matrix); down: (64-row tile, output block pair)
    int v[4] = {c, c > 0 ? 1 : 0, nrtg * ((nb + 1) >> 1) * 2, nrtg * (NJ / D41X_MP_DN_NB)}, tot[4];
    d41x_scan4(v, tot, ws);
    const int base = a.sh ? 1 : 0, rows_sh = a.sh ? a.rows : 0;
    const int srtg = (a.rows + D41X_MP_TMG - 1) / D41X_MP_TMG;
    const int sh_gt = a.sh ? srtg * ((a.sh->nblk + 1) >> 1) * 2 : 0, sh_dt = a.sh ? srtg * (NJ / D41X_MP_DN_NB) : 0;
    int* m = a.meta;
    if (t < E) {
        m[D41X_MP_EO + t] = rows_sh + v[0];
        if (c > 0) {
            const int s = base + v[1];
            m[D41X_MP_SE + s] = t; m[D41X_MP_SC + s] = c; m[D41X_MP_SO + s] = rows_sh + v[0]; m[D41X_MP_SB + s] = nb;
            m[D41X_MP_GT + s] = sh_gt + v[2]; m[D41X_MP_DT + s] = sh_dt + v[3];
        }
    }
    if (t == 0) {
        if (a.sh) { m[D41X_MP_SE] = -1; m[D41X_MP_SC] = a.rows; m[D41X_MP_SO] = 0; m[D41X_MP_SB] = a.sh->nblk; m[D41X_MP_GT] = 0; m[D41X_MP_DT] = 0; }
        const int ns = base + tot[1];
        m[0] = ns; m[1] = sh_gt + tot[2]; m[2] = sh_dt + tot[3]; m[3] = rows_sh + tot[0];
        m[D41X_MP_GT + ns] = sh_gt + tot[2]; m[D41X_MP_DT + ns] = sh_dt + tot[3];
    }
}

__device__ __forceinline__ int d41x_mp_find(const int* m, int base, int ns, int t) {
    int lo = 0, hi = ns - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (m[base + mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
}
extern "C" __global__ void __launch_bounds__(256) d41x_mp_plan3(D41xMpArgs a) {
    D41_PDL_ENTRY();
    const int t = threadIdx.x, rb = blockIdx.x, r0 = rb * D41X_MP_RB;
    const int nr = min(D41X_MP_RB, a.rows - r0), P = nr * a.topk;
    for (int i = t; i < P; i += 256) {
        const size_t g = (size_t)r0 * a.topk + i;
        const int e = a.ids[g];
        int p = -1;
        if (d41x_mp_held(a, e)) {
            p = a.meta[D41X_MP_EO + e] + a.hist[(size_t)rb * a.n_experts + e] + a.lrank[g];
            a.prow[p] = r0 + i / a.topk; a.pw[p] = a.wts[g];
        }
        a.pos[g] = p;
    }
    for (int i = rb * 256 + t; i < a.ncnt; i += gridDim.x * 256) a.pcnt[i] = 0;   // act-pair counters for this call
    const int ns = a.meta[0];
    for (int i = rb * 256 + t; i < a.meta[1]; i += gridDim.x * 256) a.gtab[i] = d41x_mp_find(a.meta, D41X_MP_GT, ns, i);
    for (int i = rb * 256 + t; i < a.meta[2]; i += gridDim.x * 256) a.dtab[i] = d41x_mp_find(a.meta, D41X_MP_DT, ns, i);
}

// ---- shared helpers ----------------------------------------------------------------------------------------------------
// H128 over 128 values held 16 per lane by an 8-lane group (element i = l8 * 16 + r): d41x_had128's tree exactly
// (stage s pairs elements differing in bit s; the lower index gets a + b, the upper a - b; same ops, same order).
__device__ __forceinline__ void d41x_had128_g8(float (&v)[16], int l8) {
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int len = 1 << s;
        #pragma unroll
        for (int r = 0; r < 16; ++r) if ((r & len) == 0) { const float x = v[r], y = v[r + len]; v[r] = __fadd_rn(x, y); v[r + len] = __fsub_rn(x, y); }
    }
    // PK2: the cross-lane stages as one FMA with a +-1 factor: fma(v, -1, u) = RN(u - v) (the upper element's fsub(u, v)) and
    // fma(v, +1, u) = RN(u + v) = fadd(v, u) (the lower element's), bit for bit incl. signed zeros (v * +-1 is exact).
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        const unsigned mask = 1u << s;
        const float sg = (l8 & mask) ? -1.f : 1.f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) { const float u = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask); v[r] = __fmaf_rn(v[r], sg, u); }
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r) v[r] = __fmul_rn(v[r], 0.08838834764831845f);
}
__device__ __forceinline__ void d41x_ldsm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// A tile in smem: [128 rows][16 chunks of 16 B], chunk c of row r stored at chunk c ^ (r & 7) (conflict-free ldmatrix)
__device__ __forceinline__ unsigned d41x_mp_aoff(int row, int chunk) { return (unsigned)(row * 256 + ((chunk ^ (row & 7)) << 4)); }
// 16 consecutive x values of a row as f32: X4 = f32 input (64 B), else bf16 (xcode 0) / f16 (xcode 1) (32 B)
template <bool X4>
__device__ __forceinline__ void d41x_mp_xload(const void* x, size_t idx, uint4 (&raw)[X4 ? 4 : 2]) {
    if constexpr (X4) { const uint4* p = (const uint4*)((const float*)x + idx); raw[0] = __ldg(p); raw[1] = __ldg(p + 1); raw[2] = __ldg(p + 2); raw[3] = __ldg(p + 3); }
    else { const uint4* p = (const uint4*)((const unsigned short*)x + idx); raw[0] = __ldg(p); raw[1] = __ldg(p + 1); }
}
template <bool X4>
__device__ __forceinline__ void d41x_mp_xcvt(int xcode, const uint4 (&raw)[X4 ? 4 : 2], float (&v)[16]) {
    const unsigned* w = (const unsigned*)raw;
    if constexpr (X4) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(w[i]);
    } else if (xcode == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { v[2 * i] = __uint_as_float(w[i] << 16); v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u); }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const __half2 h = *(const __half2*)&w[i]; v[2 * i] = __low2float(h); v[2 * i + 1] = __high2float(h); }
    }
}

struct D41xMpTile { int s, rt, b, mat, cnt, off, nblk, K; const D41xExpert* E; bool shx; };
__device__ __forceinline__ void d41x_mp_slot(const D41xMpArgs& a, int s, D41xMpTile& T) {
    const int* m = a.meta;
    T.s = s; T.shx = m[D41X_MP_SE + s] < 0;
    T.E = T.shx ? a.sh : a.ex + m[D41X_MP_SE + s];
    T.cnt = m[D41X_MP_SC + s]; T.off = m[D41X_MP_SO + s]; T.nblk = m[D41X_MP_SB + s];
    T.K = T.E->k1;    // SX1: placeholder; the gate / up tile sets its matrix's code (k1 / k3), the down tile k2
}
__device__ __forceinline__ int d41x_mp_row(const D41xMpArgs& a, const D41xMpTile& T, int pair) { return T.shx ? pair : a.prow[pair]; }

// ---- gate / up GEMM tile (PK2: 64 rows x 2 blocks, 16 warps) -------------------------------------------------------------
// A tile = (slot, 64-row tile, block PAIR (b0, b0 + 1), matrix): 16 warps, warp w = block b0 + (w >> 3), n-tile w & 7 (16
// columns), all 64 rows: blk / sum [4 m16][2 n8][4] (64 accumulator registers, 128 per thread: one 512-thread CTA per SM, four
// warps per scheduler). The input transform A(kb) = H128(x * suh) of the 64 rows is computed ONCE for both blocks (KE2's
// 128 x 128 tiles computed it once per block): warp w transforms rows 4w .. 4w + 3 (8 lanes per row), into the A stage while
// the previous k-block's MMAs run (double buffered); B(kb) = both blocks' 8 k-steps x 8 tiles (cp.async). A pair whose second
// block does not exist (odd slice width) leaves warps 8-15 without MMAs. Empty m16 tiles are neither transformed nor
// multiplied (FULL = all 64 rows live: no guards). Arithmetic per column unchanged (KE2): one chain of 8 MMAs per k-block
// (the first with a zero accumulator operand: the same value as an explicit zero), f32 add, ascending k-blocks.
#define D41X_MP_ASTG (D41X_MP_TM * 256)
#define D41X_MP_ASTG_G (D41X_MP_TMG * 256)
#define D41X_MP_GU_THREADS 512
#ifndef D41X_MP_SUH_LATE
#define D41X_MP_SUH_LATE 1
#endif
__device__ __forceinline__ void d41x_mma_z(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
template <int K, bool X4, bool FULL>
__device__ __forceinline__ void d41x_mp_gu_tile(const D41xMpArgs& a, const D41xMpTile& T, int tile, unsigned char* smem) {
    constexpr int NR = X4 ? 4 : 2;
    constexpr int TB = 32 * K, BSTG = 8 * 8 * TB;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l8 = lane & 7, g4 = lane >> 3;
    const int bsel = warp >> 3, nt = warp & 7;
    const bool two = T.b + 1 < T.nblk;                 // the pair's second block exists
    const bool mwarp = bsel == 0 || two;               // this warp multiplies
    const unsigned s_u = (unsigned)__cvta_generic_to_shared(smem);
    auto a_u = [&](int buf) { return s_u + (unsigned)(buf * D41X_MP_ASTG_G); };
    auto b_u = [&](int buf) { return s_u + (unsigned)(2 * D41X_MP_ASTG_G + buf * 2 * BSTG); };
    auto b_p = [&](int buf) { return smem + 2 * D41X_MP_ASTG_G + buf * 2 * BSTG; };
    const int NKB = a.dim >> 7;
    const int rows = FULL ? D41X_MP_TMG : min(D41X_MP_TMG, T.cnt - T.rt * D41X_MP_TMG);
    const int pair0 = T.off + T.rt * D41X_MP_TMG;
    const unsigned char* trb = (const unsigned char*)T.E->tr[T.mat] + (size_t)(8 * T.b) * TB;
    const long long kstr = (long long)T.E->nbs13 * TB;
    const __half* suh = T.E->suh[T.mat];
    const unsigned long long pol = d41x_pol_ef();
    // the transform row of this lane: 4 warp + g4
    const int xr_row = 4 * warp + g4;
    const bool xok = FULL || xr_row < rows;
    const unsigned xrow = xok ? (unsigned)d41x_mp_row(a, T, pair0 + xr_row) * (unsigned)a.ldx : 0u;   // < 2^32 (host cap)
    const bool xwarp = FULL || 4 * warp < rows;        // a warp whose 4 rows are all past the tile transforms nothing
    const int nmt = FULL ? 4 : (rows + 15) >> 4;       // live m16 tiles
    const int nbt = two ? 2 : 1;                       // blocks to stage
    auto issue_b = [&](int kb, int buf) {
        // both blocks' runs: per k-step, 8 tiles of block b0 then (two) 8 tiles of b0 + 1: contiguous 16 tiles in the trellis
        const int nch = nbt * BSTG / 16;
        for (int c = t; c < nch; c += D41X_MP_GU_THREADS) {
            const int per = nbt * 8 * TB / 16;
            const int kk = c / per, cc = c - kk * per;
            d41x_cp16_ef(b_u(buf) + (unsigned)(kk * 16 * TB + cc * 16), trb + (long long)(kb * 8 + kk) * kstr + cc * 16, pol);
        }
        d41_cp_commit();
    };
    uint4 xr[NR];
    uint4 sq[2];                       // this lane's 16 suh values of the k-block, packed
    auto load_suh = [&](int kb) { const uint4* p = (const uint4*)(suh + kb * 128 + l8 * 16); sq[0] = __ldg(p); sq[1] = __ldg(p + 1); };
    auto xload = [&](int kb) { if (xok) d41x_mp_xload<X4>(a.x, (size_t)(xrow + (unsigned)(kb * 128 + l8 * 16)), xr); };
    auto xform = [&](int buf, int kb) {
        // every lane of a live warp runs the butterfly (full-mask shuffles); a row past the tile computes zeros (never read)
#if D41X_MP_SUH_LATE
        load_suh(kb);   // PK3: suh read here (an L1 / L2 hit) instead of held across the MMA chain: 8 fewer live registers
#endif
        float v[16];
        d41x_mp_xcvt<X4>(a.xcode, xr, v);
        if (!xok) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = 0.f;
        }
        const unsigned* sw = (const unsigned*)sq;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2 h = *(const __half2*)&sw[i];
            v[2 * i] = __fmul_rn(v[2 * i], __low2float(h)); v[2 * i + 1] = __fmul_rn(v[2 * i + 1], __high2float(h));
        }
        d41x_had128_g8(v, l8);
        unsigned pk[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const __half2 h = __floats2half2_rn(v[2 * i], v[2 * i + 1]); pk[i] = *(const unsigned*)&h; }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_u(buf) + d41x_mp_aoff(xr_row, 2 * l8)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(a_u(buf) + d41x_mp_aoff(xr_row, 2 * l8 + 1)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
    };
    // ldmatrix addresses: row 16 m + (lane & 15) (row & 7 = lane & 7), chunk 2 kk + (lane >> 4): offset = abase + m * 4096 +
    // ((2 kk ^ (lane & 6)) << 4) with abase holding the row and the low chunk bit
    const unsigned abase = (unsigned)((lane & 15) * 256 + (((lane >> 4) ^ (lane & 1)) << 4));
    const unsigned l6 = (unsigned)(lane & 6);
    // prologue: B(0), A(0)
    issue_b(0, 0);
#if !D41X_MP_SUH_LATE
    load_suh(0);
#endif
    if (xwarp) { xload(0); xform(0, 0); }
    D41xDiet<K, true> D; D.init(lane);
    float sum[4][2][4];
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) sum[m][h][r] = 0.f;
    d41_cp_wait<0>();
    __syncthreads();
    for (int kb = 0; kb < NKB; ++kb) {
        const int cur = kb & 1, nxt = cur ^ 1;
        const bool more = kb + 1 < NKB;
#if D41X_MP_SUH_LATE
        if (more) { issue_b(kb + 1, nxt); if (xwarp) xload(kb + 1); }
#else
        if (more) { issue_b(kb + 1, nxt); load_suh(kb + 1); if (xwarp) xload(kb + 1); }
#endif
        if (mwarp) {
            float blk[4][2][4];
            const unsigned char* bt = b_p(cur) + (bsel * 8 + nt) * TB;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                unsigned bf[2][2];
                D.decode((const uint32_t*)(bt + kk * 16 * TB), bf);
                const unsigned ak = a_u(cur) + abase + (((unsigned)(2 * kk) ^ l6) << 4);
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (!FULL && m >= nmt) break;   // no MMAs on a tile's empty m16 rows (CTA-uniform)
                    unsigned a0, a1, a2, a3;
                    d41x_ldsm4(ak + (unsigned)(m * 4096), a0, a1, a2, a3);
                    if (kk == 0) { d41x_mma_z(blk[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]); d41x_mma_z(blk[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]); }
                    else { d41x_mma(blk[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]); d41x_mma(blk[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]); }
                }
                if (more && xwarp && kk == 3) xform(nxt, kb + 1);
            }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (!FULL && m >= nmt) break;
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) sum[m][h][r] = __fadd_rn(sum[m][h][r], blk[m][h][r]);
            }
        } else if (more && xwarp) xform(nxt, kb + 1);
        d41_cp_wait<0>();
        __syncthreads();
    }
    // raw column sums -> graw
    const int gid = lane >> 2, tig = lane & 3, col0 = (8 * (T.b + bsel) + nt) * 16;
    if (mwarp) {
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r0 = 16 * m + gid, r1 = r0 + 8, col = col0 + 8 * h + 2 * tig;
                if (r0 < rows) *(float2*)(a.graw + ((size_t)(pair0 + r0) * 2 + T.mat) * a.inter_ld + col) = make_float2(sum[m][h][0], sum[m][h][1]);
                if (r1 < rows) *(float2*)(a.graw + ((size_t)(pair0 + r1) * 2 + T.mat) * a.inter_ld + col) = make_float2(sum[m][h][2], sum[m][h][3]);
            }
    }
    __syncthreads();
    __shared__ int flag;
    if (t == 0) { __threadfence(); flag = (atomicAdd(a.pcnt + (tile >> 1), 1) == 1); }
    __syncthreads();
    if (!flag) return;
    __threadfence();
    // act for the pair (both matrices done): rows of the tile, the live blocks
    const float lim = T.shx ? a.limit_sh : a.limit;
    for (int jj = warp; jj < rows * nbt; jj += 16) {
        const int j = jj >> (two ? 1 : 0), bb = two ? (jj & 1) : 0;
        const int pair = pair0 + j;
        const int c0 = (T.b + bb) * 128 + lane * 4;
        const float w = T.shx ? 1.f : __ldg(a.pw + pair);
        const float4 g4v = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 0) * a.inter_ld + c0));
        const float4 u4v = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 1) * a.inter_ld + c0));
        float gg[4] = {g4v.x, g4v.y, g4v.z, g4v.w}, uu[4] = {u4v.x, u4v.y, u4v.z, u4v.w}, v[4];
        d41x_had128(gg, lane); d41x_had128(uu, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            v[r] = d41x_act1(__fmul_rn(gg[r], __half2float(T.E->svh[0][c0 + r])), __fmul_rn(uu[r], __half2float(T.E->svh[1][c0 + r])), w, __half2float(T.E->suh[2][c0 + r]), lim);
        d41x_had128(v, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
        *(uint2*)(a.xh2 + (size_t)pair * a.inter_ld + c0) = *(const uint2*)o;
    }
    if (t == 0) a.pcnt[tile >> 1] = 0;
}

// smem: 2 A stages (2 x 16 KB) + 2 B stages (2 x 2 blocks x 2 KB x K). Entries by input width (separate register
// allocations): d41x_mp_gu_x2 (bf16 / f16 x), d41x_mp_gu_x4 (f32 x); PK3: the bf16 / f16 entry is split by trellis width,
// d41x_mp_gu_x2 (K 2..4: the routed K2 / K3 and shared K4 experts of the EAQ packs) and d41x_mp_gu_x2h (K 5..8): with all
// seven widths in one function ptxas spilled (every width alone fits 128 registers). Each entry serves the tiles of its
// widths and returns on the others; the host launches the entries the call's widths need (both on the same tile table).
template <bool X4, int KLO, int KHI>
__device__ __forceinline__ void d41x_mp_gu_entry(const D41xMpArgs& a) {
    extern __shared__ __align__(128) unsigned char d41x_mp_smem[];
    const int tile = blockIdx.x;
    const int* m = a.meta;
    if (tile >= m[1]) return;
    D41xMpTile T;
    const int s = __ldg(a.gtab + tile);
    d41x_mp_slot(a, s, T);
    const int local = tile - m[D41X_MP_GT + s];
    const int nbp = (T.nblk + 1) >> 1;
    T.mat = local & 1;
    T.K = T.mat ? T.E->k3 : T.E->k1;   // SX1: this tile's matrix at its own width code
    T.b = 2 * ((local >> 1) % nbp); T.rt = (local >> 1) / nbp;   // rt: 64-row tile; b: the pair's first block
    if (T.K < KLO || T.K > KHI) return;   // the other entry serves this tile's width
    const bool full = T.cnt - T.rt * D41X_MP_TMG >= D41X_MP_TMG;
#define D41X_MP_GUK(KK) case KK: if constexpr (KK >= KLO && KK <= KHI) { if (full) d41x_mp_gu_tile<KK, X4, true>(a, T, tile, d41x_mp_smem); else d41x_mp_gu_tile<KK, X4, false>(a, T, tile, d41x_mp_smem); } break;
    switch (T.K) { D41X_MP_GUK(2) D41X_MP_GUK(3) D41X_MP_GUK(4) D41X_MP_GUK(5) D41X_MP_GUK(6) D41X_MP_GUK(7) D41X_MP_GUK(8) default: break; }
#undef D41X_MP_GUK
}
extern "C" __global__ void __launch_bounds__(D41X_MP_GU_THREADS, 1) d41x_mp_gu_x2(D41xMpArgs a) { D41_PDL_ENTRY(); d41x_mp_gu_entry<false, 2, 4>(a); }
extern "C" __global__ void __launch_bounds__(D41X_MP_GU_THREADS, 1) d41x_mp_gu_x2h(D41xMpArgs a) { D41_PDL_ENTRY(); d41x_mp_gu_entry<false, 5, 8>(a); }
extern "C" __global__ void __launch_bounds__(D41X_MP_GU_THREADS, 1) d41x_mp_gu_x4(D41xMpArgs a) { D41_PDL_ENTRY(); d41x_mp_gu_entry<true, 2, 8>(a); }

// ---- down GEMM tile (PK2: 64 rows x 2 output blocks, 16 warps) -------------------------------------------------------------
// A tile = (slot, 64-row tile, output block PAIR (j0, j0 + 1)): 16 warps, warp w = output block j0 + (w >> 3), n-tile w & 7,
// all 64 rows: blk / sum [4 m16][2 n8][4]. A(kb) = the 64 xh2 rows (cp.async, shared by both blocks), B(kb) = both blocks'
// 8 k-steps x 8 tiles of w2 (contiguous 16 tiles per k-step), double buffered. Epilogue: sums staged in smem (over the idle
// stages), H128 per (row, block) -> h2. Empty m16 tiles are not multiplied (FULL: no guards). Arithmetic per column
// unchanged (KE2): one chain of 8 MMAs per k-block (the first with a zero accumulator operand), f32 add, ascending.
#define D41X_MP_DN_THREADS (256 * D41X_MP_DN_NB)
#define D41X_MP_DN_LD (128 * D41X_MP_DN_NB + 8)   // staged row stride (floats): row-to-row bank shift 8
template <int K, bool FULL>
__device__ __forceinline__ void d41x_mp_dn_tile(const D41xMpArgs& a, const D41xMpTile& T, unsigned char* smem) {
    constexpr int NB = D41X_MP_DN_NB;
    constexpr int TB = 32 * K, BSTG = 8 * 8 * TB;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int bsel = warp >> 3, nt = warp & 7;
    const unsigned s_u = (unsigned)__cvta_generic_to_shared(smem);
    auto a_u = [&](int buf) { return s_u + (unsigned)(buf * D41X_MP_ASTG_G); };
    auto b_u = [&](int buf) { return s_u + (unsigned)(2 * D41X_MP_ASTG_G + buf * NB * BSTG); };
    auto b_p = [&](int buf) { return smem + 2 * D41X_MP_ASTG_G + buf * NB * BSTG; };
    const int NKB = T.nblk;
    const int rows = FULL ? D41X_MP_TMG : min(D41X_MP_TMG, T.cnt - T.rt * D41X_MP_TMG);
    const int pair0 = T.off + T.rt * D41X_MP_TMG;
    const int j0 = T.b;
    const unsigned char* trb = (const unsigned char*)T.E->tr[2] + (size_t)(8 * j0) * TB;
    const long long kstr = (long long)(a.dim / 16) * TB;
    const unsigned long long pol = d41x_pol_ef();
    const int nmt = FULL ? 4 : (rows + 15) >> 4;
    auto issue = [&](int kb, int buf) {
        for (int c = t; c < NB * BSTG / 16; c += D41X_MP_DN_THREADS) {
            const int kk = c / (NB * 8 * TB / 16), cc = c - kk * (NB * 8 * TB / 16);
            d41x_cp16_ef(b_u(buf) + (unsigned)(kk * NB * 8 * TB + cc * 16), trb + (long long)(kb * 8 + kk) * kstr + cc * 16, pol);
        }
        for (int c = t; c < rows * 16; c += D41X_MP_DN_THREADS) {
            const int r = c >> 4, ch = c & 15;
            d41x_cp16(a_u(buf) + d41x_mp_aoff(r, ch), a.xh2 + (size_t)(pair0 + r) * a.inter_ld + kb * 128 + ch * 8);
        }
        d41_cp_commit();
    };
    const unsigned abase = (unsigned)((lane & 15) * 256 + (((lane >> 4) ^ (lane & 1)) << 4));
    const unsigned l6 = (unsigned)(lane & 6);
    D41xDiet<K, true> D; D.init(lane);
    float sum[4][2][4];
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) sum[m][h][r] = 0.f;
    issue(0, 0);
    for (int kb = 0; kb < NKB; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < NKB) { issue(kb + 1, cur ^ 1); d41_cp_wait<1>(); } else d41_cp_wait<0>();
        __syncthreads();
        float blk[4][2][4];
        const unsigned char* bt = b_p(cur) + (bsel * 8 + nt) * TB;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            unsigned bf[2][2];
            D.decode((const uint32_t*)(bt + kk * NB * 8 * TB), bf);
            const unsigned ak = a_u(cur) + abase + (((unsigned)(2 * kk) ^ l6) << 4);
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (!FULL && m >= nmt) break;
                unsigned a0, a1, a2, a3;
                d41x_ldsm4(ak + (unsigned)(m * 4096), a0, a1, a2, a3);
                if (kk == 0) { d41x_mma_z(blk[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]); d41x_mma_z(blk[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]); }
                else { d41x_mma(blk[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]); d41x_mma(blk[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]); }
            }
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (!FULL && m >= nmt) break;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) sum[m][h][r] = __fadd_rn(sum[m][h][r], blk[m][h][r]);
        }
        __syncthreads();
    }
    // stage [64][LD] f32 over the (now idle) stages, then H128 per (row, block) -> h2
    float* st = (float*)smem;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (!FULL && m >= nmt) break;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r0 = 16 * m + gid, col = bsel * 128 + nt * 16 + 8 * h + 2 * tig;
            *(float2*)(st + r0 * D41X_MP_DN_LD + col) = make_float2(sum[m][h][0], sum[m][h][1]);
            *(float2*)(st + (r0 + 8) * D41X_MP_DN_LD + col) = make_float2(sum[m][h][2], sum[m][h][3]);
        }
    }
    __syncthreads();
    // PK3 bf16 mode: the pair's contribution v * svh (the expert's output on this rank's slice) rounded to bf16 here, half the
    // bytes of the f32 h2; the default stores the f32 v and the combine applies svh (bit-identical to KE2 / PK2)
    const __half* svh2 = T.E->svh[2];
    for (int rb = warp; rb < rows * NB; rb += 8 * NB) {
        const int r = NB == 2 ? rb >> 1 : rb, bb = NB == 2 ? rb & 1 : 0;
        float v[4];
        const float4 q = *(const float4*)(st + r * D41X_MP_DN_LD + bb * 128 + lane * 4);
        v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
        d41x_had128(v, lane);
        const int c0 = (j0 + bb) * 128 + lane * 4;
        if (a.h2bf) {
            const uint2 sv = __ldg((const uint2*)(svh2 + c0));
            const __half2 s01 = *(const __half2*)&sv.x, s23 = *(const __half2*)&sv.y;
            const __nv_bfloat162 o01 = __floats2bfloat162_rn(__fmul_rn(v[0], __low2float(s01)), __fmul_rn(v[1], __high2float(s01)));
            const __nv_bfloat162 o23 = __floats2bfloat162_rn(__fmul_rn(v[2], __low2float(s23)), __fmul_rn(v[3], __high2float(s23)));
            *(uint2*)((__nv_bfloat16*)a.h2 + (size_t)(pair0 + r) * a.dim + c0) = make_uint2(*(const unsigned*)&o01, *(const unsigned*)&o23);
        } else {
            *(float4*)(a.h2 + (size_t)(pair0 + r) * a.dim + c0) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(D41X_MP_DN_THREADS, 2 / D41X_MP_DN_NB) d41x_mp_dn(D41xMpArgs a) {
    D41_PDL_ENTRY();
    extern __shared__ __align__(128) unsigned char d41x_mp_smem[];
    const int tile = blockIdx.x;
    const int* m = a.meta;
    if (tile >= m[2]) return;
    D41xMpTile T;
    const int s = __ldg(a.dtab + tile);
    d41x_mp_slot(a, s, T);
    const int local = tile - m[D41X_MP_DT + s];
    const int NJP = (a.dim >> 7) / D41X_MP_DN_NB;  // output block groups
    T.b = D41X_MP_DN_NB * (local % NJP); T.rt = local / NJP; T.mat = 2;   // rt: 64-row tile; b: the group's first output block
    T.K = T.E->k2;    // SX1: w2 at its own width code
    const bool full = T.cnt - T.rt * D41X_MP_TMG >= D41X_MP_TMG;
#define D41X_MP_DNK(KK) case KK: if (full) d41x_mp_dn_tile<KK, true>(a, T, d41x_mp_smem); else d41x_mp_dn_tile<KK, false>(a, T, d41x_mp_smem); break;
    switch (T.K) { D41X_MP_DNK(2) D41X_MP_DNK(3) D41X_MP_DNK(4) D41X_MP_DNK(5) D41X_MP_DNK(6) D41X_MP_DNK(7) D41X_MP_DNK(8) default: break; }
#undef D41X_MP_DNK
}

// ---- combine: grid (rows, dim / 512), 128 threads, 4 columns each --------------------------------------------------------
// 4 bf16 h2 values (PK3 bf16 mode) at element offset i of the bf16 view of h2, as f32
__device__ __forceinline__ void d41x_mp_bf4(const float* h2, size_t i, float (&v)[4]) {
    const uint2 u = __ldcg((const uint2*)((const __nv_bfloat16*)h2 + i));
    v[0] = __uint_as_float(u.x << 16); v[1] = __uint_as_float(u.x & 0xFFFF0000u);
    v[2] = __uint_as_float(u.y << 16); v[3] = __uint_as_float(u.y & 0xFFFF0000u);
}
extern "C" __global__ void __launch_bounds__(128) d41x_mp_combine(D41xMpArgs a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x, c0 = blockIdx.y * 512 + threadIdx.x * 4;
    if (c0 >= a.dim) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int prev = -1;
    for (int j = 0; j < a.topk; ++j) {
        int best = 0x7FFFFFFF, bp = -1;
        for (int k = 0; k < a.topk; ++k) {
            const int p = a.pos[(size_t)r * a.topk + k];
            if (p < 0) continue;
            const int e = a.ids[(size_t)r * a.topk + k];
            if (e > prev && e < best) { best = e; bp = p; }
        }
        if (bp < 0) break;
        prev = best;
        if (a.h2bf) {   // PK3: bf16 contributions (svh applied by the down tile), added in f32 in ascending expert id
            float v[4];
            d41x_mp_bf4(a.h2, (size_t)bp * a.dim + c0, v);
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[q] = __fadd_rn(acc[q], v[q]);
            continue;
        }
        const D41xExpert& E = a.ex[best];
        const float4 v4 = __ldcg((const float4*)(a.h2 + (size_t)bp * a.dim + c0));
        const float v[4] = {v4.x, v4.y, v4.z, v4.w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] = __fmaf_rn(v[q], __half2float(E.svh[2][c0 + q]), acc[q]);
    }
    float* o = a.out + (size_t)r * a.ldo + c0;
    float ov[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) ov[q] = __fadd_rn(o[q], acc[q]);
    if (a.sh && a.h2bf) {
        float v[4];
        d41x_mp_bf4(a.h2, (size_t)r * a.dim + c0, v);
        #pragma unroll
        for (int q = 0; q < 4; ++q) ov[q] = __fadd_rn(ov[q], v[q]);
    } else if (a.sh) {
        const float4 v4 = __ldcg((const float4*)(a.h2 + (size_t)r * a.dim + c0));
        const float v[4] = {v4.x, v4.y, v4.z, v4.w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) ov[q] = __fadd_rn(ov[q], __fmaf_rn(v[q], __half2float(a.sh->svh[2][c0 + q]), 0.f));
    }
    *(float4*)o = make_float4(ov[0], ov[1], ov[2], ov[3]);
}

extern "C" __global__ void d41x_mp_sizes(int* o) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    o[0] = (int)sizeof(D41xMpArgs); o[1] = D41X_MP_META_INTS; o[2] = D41X_MP_ASTG; o[3] = D41X_MP_TM; o[4] = D41X_MP_RB; o[5] = D41X_MP_TMG; o[6] = D41X_MP_ASTG_G; o[7] = D41X_MP_DN_NB;
}
