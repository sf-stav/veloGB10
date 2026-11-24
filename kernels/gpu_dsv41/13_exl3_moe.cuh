// 13_exl3_moe.cuh (leg KE): grouped MoE decode over a per-expert INNER-WIDTH slice table (T0 §3.1 / §11).
// Per expert e the table holds the 128-block slice [blk0, blk0 + nblk) of the inner width (2304 = 18 blocks) as
// pointers already offset to that slice: w1 / w3 are cut along their output (n-tiles), w2 along its input (k16
// steps). nblk == 0 = not held by this rank (EP = whole-or-empty slices). Chain per (row, expert):
//   g = H(w1 raw) svh1, u = H(w3 raw) svh3, a = bf16( weight * silu(min(g, L)) * clamp(u, +-L) ),
//   y_e = H( w2 raw( H(a * suh2) ) ) svh2;  out[row] (+)= sum over the row's experts, ascending expert id, f32.
// Weight-stationary: every distinct expert is decoded once and applied to all of its (<= 32) rows.
// The shared expert is a 1-expert table with ids == 0 and weight 1.0 (exact), same kernels.
#pragma once

struct D41xExpert {
    const uint16_t* tr[3];   // w1, w3, w2 trellis, tile (0,0) of the slice
    const __half* suh[3];    // w1, w3: [dim] ; w2: slice start
    const __half* svh[3];    // w1, w3: slice start ; w2: [dim]
    int nbs13;               // resident row length (n-tiles) of w1 / w3
    int nblk;                // slice width in 128-blocks (0 = not held)
    int k1, k3, k2;          // leg SX1: the width code of w1, w3 and w2, each read at its own (a pack may mix them)
    int kpad;
};

struct D41xMoeArgs {
    const D41xExpert* ex;
    const int* ids; const float* wts;              // [rows][topk]
    const void* x; int xcode, ldx;                 // activations [rows][ldx]
    int* ctl;                                      // [0] n_slots, [1] n_pairs
    int* slot_expert; int* slot_off; int* slot_cnt;
    int* pair_expert; int* pair_row; float* pair_w; int* pair_of; int* row_order;
    __half* xh1; float* y1; __half* xh2; float* y2;
    float* out; int ldo; int accum;
    float limit;
    int n_experts, rows, topk, dim, inter_ld, nblk_max, kmax, sms;   // kmax: informational (every matrix is read at its entry's code)
};

// ---- routing compaction: ONE CTA of 512 threads, n_experts <= 1024 (two per thread) ----------------------
extern "C" __global__ void __launch_bounds__(512) d41x_moe_route(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    __shared__ int s_cnt[1024], s_off[1024], s_slot[1024], s_wsum[2][32];
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int P = a.rows * a.topk;
    for (int p = t; p < P; p += blockDim.x) a.pair_of[p] = -1;
    for (int e = t; e < 1024; e += blockDim.x) {
        int c = 0;
        if (e < a.n_experts && a.ex[e].nblk > 0) for (int p = 0; p < P; ++p) c += (a.ids[p] == e);
        s_cnt[e] = c;
    }
    __syncthreads();
    // exclusive scans of cnt and of (cnt > 0): each thread owns experts 2t, 2t+1
    int c0 = s_cnt[2 * t], c1 = s_cnt[2 * t + 1];
    int f0 = c0 > 0, f1 = c1 > 0;
    int sc = c0 + c1, sf = f0 + f1, ic = sc, iff = sf;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int a1 = __shfl_up_sync(0xFFFFFFFFu, ic, o), a2 = __shfl_up_sync(0xFFFFFFFFu, iff, o);
        if (lane >= o) { ic += a1; iff += a2; }
    }
    if (lane == 31) { s_wsum[0][wid] = ic; s_wsum[1][wid] = iff; }
    __syncthreads();
    int wc = 0, wf = 0;
    for (int w = 0; w < wid; ++w) { wc += s_wsum[0][w]; wf += s_wsum[1][w]; }
    const int ec = wc + ic - sc, ef = wf + iff - sf;   // exclusive for this thread's pair
    s_off[2 * t] = ec;       s_off[2 * t + 1] = ec + c0;
    s_slot[2 * t] = ef;      s_slot[2 * t + 1] = ef + f0;
    if (t == blockDim.x - 1) { a.ctl[0] = ef + sf; a.ctl[1] = ec + sc; }
    __syncthreads();
    for (int e = 2 * t; e < 2 * t + 2; ++e) {
        const int c = s_cnt[e];
        if (c == 0) continue;
        const int s = s_slot[e], off = s_off[e];
        a.slot_expert[s] = e; a.slot_off[s] = off; a.slot_cnt[s] = c;
        int n = 0;
        for (int p = 0; p < P; ++p) if (a.ids[p] == e) {
            const int pr = off + n++;
            a.pair_expert[pr] = e; a.pair_row[pr] = p / a.topk; a.pair_w[pr] = a.wts[p]; a.pair_of[p] = pr;
        }
    }
    __syncthreads();
    for (int r = t; r < a.rows; r += blockDim.x) {
        int prev = -1;
        for (int j = 0; j < a.topk; ++j) {
            int best = 0x7FFFFFFF;
            for (int k = 0; k < a.topk; ++k) { const int q = a.pair_of[r * a.topk + k]; if (q > prev && q < best) best = q; }
            a.row_order[r * a.topk + j] = (best == 0x7FFFFFFF) ? -1 : best;
            if (best != 0x7FFFFFFF) prev = best;
        }
    }
}

// ---- prep: xh1[(pair*2 + mat)][dim] = fp16( H128( x[row] * suh_mat ) ) ------------------------------------------
extern "C" __global__ void __launch_bounds__(32) d41x_moe_prep1(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    const int p = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x;
    if (p >= a.ctl[1]) return;
    const int e = a.pair_expert[p], row = a.pair_row[p];
    #pragma unroll
    for (int mat = 0; mat < 2; ++mat) {
        float v[4], sv[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) { v[r] = d41x_ld(a.x, a.xcode, (size_t)row * a.ldx + blk * 128 + lane * 4 + r); sv[r] = __half2float(a.ex[e].suh[mat][blk * 128 + lane * 4 + r]); }
        d41x_prep4(v, sv, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r) a.xh1[((size_t)p * 2 + mat) * a.dim + blk * 128 + lane * 4 + r] = __float2half_rn(v[r]);
    }
}

// ---- GEMM 1 (w1 and w3 of every active slot): persistent CTAs over (slot, mat, 128-block) items, KS = 1 -----------
// KE2: ONE ascending pass over the whole K per output column (MMA chains per 8-step k-block, added in f32: d41x_tiles),
// was 8 K-splits summed in smem: the partition the decode (15_*) and prefill (16_*) kernels use, so every path gives the
// same bits for a row.
extern "C" __global__ void __launch_bounds__(256, 1) d41x_moe_gemm1(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    const int T = a.nblk_max;
    const int total = a.ctl[0] * 2 * T;
    for (int it = blockIdx.x; it < total; it += gridDim.x) {
        const int slot = it / (2 * T), mat = (it / T) & 1, blk = it % T;
        const int e = a.slot_expert[slot];
        const D41xExpert& ex = a.ex[e];
        if (blk < ex.nblk) {
            const int off = a.slot_off[slot], cnt = a.slot_cnt[slot];
            D41xGemmArgs g;
            g.tr = ex.tr[mat]; g.xh = a.xh1 + ((size_t)off * 2 + mat) * a.dim; g.y = a.y1 + ((size_t)off * 2 + mat) * a.inter_ld; g.gstride = 0;
            g.nb_stride = ex.nbs13; g.nb0 = 0; g.nbn = ex.nblk * 8; g.kb0 = 0; g.nkb = a.dim / 16;
            g.lda = 2 * a.dim; g.rows = cnt; g.ldy = 2 * a.inter_ld; g.ks = 1; g.g = 1; g.bits = mat ? ex.k3 : ex.k1;
            if (cnt <= 16) d41x_gemm_dispatch<1>(g, blk, 0, 0); else d41x_gemm_dispatch<2>(g, blk, 0, 0);
        }
        __syncthreads();
    }
}

// ---- activation: SwiGLU (+ routing weight, bf16 round) then w2's prep, per (pair, 128-block of the slice) -------
extern "C" __global__ void __launch_bounds__(32) d41x_moe_act(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    const int p = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x;
    if (p >= a.ctl[1]) return;
    const int e = a.pair_expert[p];
    const D41xExpert& ex = a.ex[e];
    if (blk >= ex.nblk) return;
    float g[4], u[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        g[r] = a.y1[((size_t)p * 2 + 0) * a.inter_ld + blk * 128 + lane * 4 + r];
        u[r] = a.y1[((size_t)p * 2 + 1) * a.inter_ld + blk * 128 + lane * 4 + r];
    }
    d41x_had128(g, lane); d41x_had128(u, lane);
    const float w = a.pair_w[p];
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int c = blk * 128 + lane * 4 + r;
        v[r] = d41x_act1(__fmul_rn(g[r], __half2float(ex.svh[0][c])), __fmul_rn(u[r], __half2float(ex.svh[1][c])), w, __half2float(ex.suh[2][c]), a.limit);
    }
    d41x_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) a.xh2[(size_t)p * a.inter_ld + blk * 128 + lane * 4 + r] = __float2half_rn(v[r]);
}

// ---- GEMM 2 (w2): items (slot, group of 8 output n-tiles), KS = 1, K = the slice -----------------------------------
extern "C" __global__ void __launch_bounds__(256, 1) d41x_moe_gemm2(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    const int G = (a.dim / 16 + 7) / 8;
    const int total = a.ctl[0] * G;
    for (int it = blockIdx.x; it < total; it += gridDim.x) {
        const int slot = it / G, grp = it % G;
        const D41xExpert& ex = a.ex[a.slot_expert[slot]];
        const int off = a.slot_off[slot], cnt = a.slot_cnt[slot];
        D41xGemmArgs g;
        g.tr = ex.tr[2]; g.xh = a.xh2 + (size_t)off * a.inter_ld; g.y = a.y2 + (size_t)off * a.dim; g.gstride = 0;
        g.nb_stride = a.dim / 16; g.nb0 = 0; g.nbn = a.dim / 16; g.kb0 = 0; g.nkb = ex.nblk * 8;
        g.lda = a.inter_ld; g.rows = cnt; g.ldy = a.dim; g.ks = 1; g.g = 1; g.bits = ex.k2;
        if (cnt <= 16) d41x_gemm_dispatch<1>(g, grp, 0, 0); else d41x_gemm_dispatch<2>(g, grp, 0, 0);
        __syncthreads();
    }
}

// ---- combine: out[row] (+)= sum over the row's pairs (ascending expert id) of H128(y2) * svh2 ----------------------
extern "C" __global__ void __launch_bounds__(32) d41x_moe_post2(D41xMoeArgs a) {
    D41_PDL_ENTRY();
    const int blk = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    if (row >= a.rows) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < a.topk; ++j) {
        const int p = a.row_order[row * a.topk + j];
        if (p < 0) break;
        const D41xExpert& ex = a.ex[a.pair_expert[p]];
        float v[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = a.y2[(size_t)p * a.dim + blk * 128 + lane * 4 + r];
        d41x_had128(v, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[r] = __fmaf_rn(v[r], __half2float(ex.svh[2][blk * 128 + lane * 4 + r]), acc[r]);
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float* o = &a.out[(size_t)row * a.ldo + blk * 128 + lane * 4 + r];
        *o = a.accum ? __fadd_rn(*o, acc[r]) : acc[r];
    }
}
