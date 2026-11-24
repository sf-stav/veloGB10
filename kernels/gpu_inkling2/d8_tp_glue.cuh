// gpu_inkling2/d8_tp_glue.cuh — tensor-parallel glue (leg S-INK2-TP2). Section of gpu_inkling2.cu.
//
// The vocab-parallel sampler. Each rank holds the logits of its vocabulary rows [voff, voff + n) (65_lm_head.cuh with
// row_base = voff). The Gumbel noise is the TP1 sampler's hash of (seed, GLOBAL id), so a token's perturbed score is
// the same whatever rank holds it:
//   ink_tp_gumbel_part: 80_glue.cuh's ink3_gumbel_part with the global id = voff + i (grid (G, M), block 256);
//   ink_tp_gumbel_rec:  80_glue.cuh's ink3_gumbel_pick, writing the rank's (best, id) into its slot of an exchange
//                       record [M][W][2] fp32 whose other slots are zero (grid (M), block 256). An all-reduce sum of
//                       the records IS their gather (x + 0 = x exactly; ids < 2^24 are exact in fp32);
//   ink_tp_rank_pick:   per row, the best of the W records, ties to the lowest id (the TP1 rule) (grid 1, block 32).
// T <= 0 rows are greedy (no noise), as in TP1.
//   ink_tp_logits_place: rows [M][n] of this rank's logits into a zeroed [M][V] buffer at column voff (the CPU-sampled
//                       rows' full-vocabulary gather: all-reduce sum of the placed rows).

extern "C" __global__ __launch_bounds__(256) void ink_tp_gumbel_part(const float* __restrict__ logits, int ldl, int n, int voff,
        const unsigned long long* __restrict__ seeds, const float* __restrict__ inv_t,
        float* __restrict__ pmax, int* __restrict__ pidx) { INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int row = blockIdx.y, G = gridDim.x;
    const int per = (n + G - 1) / G;
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    const float it = inv_t[row];
    const unsigned long long seed = seeds[row];
    const float* lr = logits + (size_t)row * ldl;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {
        float v = lr[i];
        const int gi = voff + i;
        if (it > 0.f) {
            const unsigned long long h = ink3_mix(seed ^ ink3_mix((unsigned long long)gi));
            const double w = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
            const double e = -log1p(-w);
            v = v * it - (float)log(e);
        }
        if (v > best) { best = v; bi = gi; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sv[threadIdx.x], b = sv[threadIdx.x + s];
            const int ia = si[threadIdx.x], ib = si[threadIdx.x + s];
            if (b > a || (b == a && ib < ia)) { sv[threadIdx.x] = b; si[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmax[row * G + blockIdx.x] = sv[0]; pidx[row * G + blockIdx.x] = si[0]; }
}

extern "C" __global__ __launch_bounds__(256) void ink_tp_gumbel_rec(const float* __restrict__ pmax, const int* __restrict__ pidx,
        int G, float* __restrict__ rec, int rank, int world) { INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int row = blockIdx.x;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int g = threadIdx.x; g < G; g += 256) {
        const float v = pmax[row * G + g]; const int i = pidx[row * G + g];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sv[threadIdx.x], b = sv[threadIdx.x + s];
            const int ia = si[threadIdx.x], ib = si[threadIdx.x + s];
            if (b > a || (b == a && ib < ia)) { sv[threadIdx.x] = b; si[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x < 2 * world) {
        const int r = threadIdx.x >> 1, f = threadIdx.x & 1;
        float v = 0.f;
        if (r == rank) v = f ? (float)si[0] : sv[0];
        rec[((size_t)row * world + r) * 2 + f] = v;
    }
}

extern "C" __global__ __launch_bounds__(32) void ink_tp_rank_pick(const float* __restrict__ rec, int world, int m, int* __restrict__ out) { INK2_PDL_ENTRY();
    const int row = threadIdx.x;
    if (row >= m) return;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int r = 0; r < world; ++r) {
        const float v = rec[((size_t)row * world + r) * 2];
        const int i = (int)rec[((size_t)row * world + r) * 2 + 1];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    out[row] = bi;
}

extern "C" __global__ __launch_bounds__(256) void ink_tp_logits_place(const float* __restrict__ logits, int ldl, int n, int voff,
        float* __restrict__ full, int V) { INK2_PDL_ENTRY();
    const int row = blockIdx.y;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < V; i += gridDim.x * 256) {
        const int j = i - voff;
        full[(size_t)row * V + i] = (j >= 0 && j < n) ? logits[(size_t)row * ldl + j] : 0.f;
    }
}

// Phase B: pack this rank's [n][vn] logits rows into gather[(rank * n + row) * per ..] (zero-padding vn..per on the last rank)
// so a 2-rail byte-exact all-gather transfers only (W-1)/W of the vocabulary (6.85x less wire/rail at TP8 vs 1-rail all-reduce).
extern "C" __global__ __launch_bounds__(256) void ink_tp_logits_pack(const float* __restrict__ logits, int ldl, int vn,
        float* __restrict__ gather, int per, int rank, int n) { INK2_PDL_ENTRY();
    const int row = blockIdx.y;
    if (row >= n) return;
    const float* src = logits + (size_t)row * ldl;
    float* dst = gather + ((size_t)rank * n + row) * per;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < per; i += gridDim.x * 256) {
        dst[i] = (i < vn) ? src[i] : 0.f;
    }
}

// Phase B: unpack all-gathered [world][n][per] rows into contiguous full [n][V] (col = r * per + j < V).
extern "C" __global__ __launch_bounds__(256) void ink_tp_logits_unpack(const float* __restrict__ gather, int per,
        float* __restrict__ full, int V, int n, int world) { INK2_PDL_ENTRY();
    const int row = blockIdx.y;
    if (row >= n) return;
    float* dst = full + (size_t)row * V;
    for (int col = blockIdx.x * 256 + threadIdx.x; col < V; col += gridDim.x * 256) {
        const int r = col / per;
        const int j = col - r * per;
        dst[col] = (r < world) ? gather[((size_t)r * n + row) * per + j] : 0.f;
    }
}

// -----------------------------------------------------------------------------------------
// Drop 1 (#1): Vocab-Parallel Drafter LM Head + Vocab-Parallel Exact Speculative Accept
// -----------------------------------------------------------------------------------------

// Per-lane pointers for packing one Markov step across up to 4 active lanes.
struct InkTpMtpPackArgs {
    const float* fpm[4];
    const float* fps[4];
    const float* gpm[4];
    const int* gpi[4];
};
struct InkTpMtpFinArgs {
    float* lse_q[4];
    float* conf[4];
    int* tok[4];
};

// Pack this rank's local (max_q, sum_q, best_gumbel, global_id) for each active lane l = blockIdx.x
// into rec[(l * world + r) * 4 + 0..4] (other ranks' slots zeroed for exact sum-gather).
// grid (L, 1, 1), block 256.
extern "C" __global__ __launch_bounds__(256) void ink_tp_mtp_step_pack(
        InkTpMtpPackArgs a, int fold_g, int smp_g, float* __restrict__ rec, int rank, int world) {
    INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ float s_lse_m, s_lse_s;
    const int l = blockIdx.x;
    // 1. Reduce fold_g (max, sumexp) partials on thread 0 (fold_g <= 384, ~1 us).
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int g = 0; g < fold_g; ++g) M = fmaxf(M, a.fpm[l][g]);
        double S = 0.0;
        for (int g = 0; g < fold_g; ++g) {
            const float m = a.fpm[l][g];
            if (m > -INFINITY) S += (double)a.fps[l][g] * exp((double)(m - M));
        }
        s_lse_m = M;
        s_lse_s = (float)S;
    }
    // 2. Reduce smp_g (best_gumbel, global_id) partials across all 256 threads (ties -> lowest id).
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int g = threadIdx.x; g < smp_g; g += 256) {
        const float v = a.gpm[l][g]; const int id = a.gpi[l][g];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float x = sv[threadIdx.x], y = sv[threadIdx.x + s];
            const int ix = si[threadIdx.x], iy = si[threadIdx.x + s];
            if (y > x || (y == x && iy < ix)) { sv[threadIdx.x] = y; si[threadIdx.x] = iy; }
        }
        __syncthreads();
    }
    if (threadIdx.x < 4 * world) {
        const int r = threadIdx.x >> 2, f = threadIdx.x & 3;
        float out = 0.f;
        if (r == rank) {
            out = (f == 0) ? s_lse_m : (f == 1) ? s_lse_s : (f == 2) ? sv[0] : (float)si[0];
        }
        rec[((size_t)l * world + r) * 4 + f] = out;
    }
}

// After all-reducing `rec` [L][world][4], finalize global lse_q, conf, and winning global draft token
// for each active lane l = blockIdx.x. grid (L, 1, 1), block 32.
extern "C" __global__ __launch_bounds__(32) void ink_tp_mtp_step_fin(
        const float* __restrict__ rec, int world, InkTpMtpFinArgs o) {
    INK2_PDL_ENTRY();
    if (threadIdx.x) return;
    const int l = blockIdx.x;
    float M = -INFINITY;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int r = 0; r < world; ++r) {
        const size_t base = ((size_t)l * world + r) * 4;
        M = fmaxf(M, rec[base + 0]);
        const float v = rec[base + 2];
        const int id = (int)rec[base + 3];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    double S = 0.0;
    for (int r = 0; r < world; ++r) {
        const size_t base = ((size_t)l * world + r) * 4;
        const float m = rec[base + 0];
        if (m > -INFINITY) S += (double)rec[base + 1] * exp((double)(m - M));
    }
    *o.lse_q[l] = M + (float)log(S);
    *o.conf[l] = (float)(1.0 / S);
    *o.tok[l] = bi;
}

// Vocab-parallel sampled accept (Stage 1 pack): per verify row (0..=c), reduce LSE_G partials of this
// rank's target logits and extract (tl[row, d], dl[row, d]) if draft token d = tok[row + 1] lies in
// [voff, voff + vn). Packs [local_M, local_S, tl_d, dl_d] into rec[(row * world + r) * 4 + 0..4].
// grid (c + 1, 1, 1), block 32.
extern "C" __global__ __launch_bounds__(32) void ink_tp_accept_pack(
        const float* __restrict__ lpm, const float* __restrict__ lps, int lse_g,
        const int* __restrict__ tok, int c,
        const float* __restrict__ tl, int ldl, const float* __restrict__ dl, int ldd,
        int voff, int vn, float* __restrict__ rec, int rank, int world) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.x;
    __shared__ float vals[4];
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int g = 0; g < lse_g; ++g) M = fmaxf(M, lpm[row * lse_g + g]);
        double S = 0.0;
        for (int g = 0; g < lse_g; ++g) {
            const float m = lpm[row * lse_g + g];
            if (m > -INFINITY) S += (double)lps[row * lse_g + g] * exp((double)(m - M));
        }
        float tld = 0.f, dld = 0.f;
        if (row < c) {
            const int d = tok[row + 1];
            const int j = d - voff;
            if (j >= 0 && j < vn) {
                tld = tl[(size_t)row * ldl + j];
                dld = dl[(size_t)row * ldd + j];
            }
        }
        vals[0] = M; vals[1] = (float)S; vals[2] = tld; vals[3] = dld;
    }
    __syncthreads();
    if (threadIdx.x < 4 * world) {
        const int r = threadIdx.x >> 2, f = threadIdx.x & 3;
        rec[((size_t)row * world + r) * 4 + f] = (r == rank) ? vals[f] : 0.f;
    }
}

// Vocab-parallel sampled accept (Stage 1 decide): after all-reducing rec [c + 1][world][4], compute
// exact global lse_p[0..=c] and run the exact sequential accept loop into res[0] on every rank.
// grid (1, 1, 1), block 32.
extern "C" __global__ __launch_bounds__(32) void ink_tp_accept_decide(
        const float* __restrict__ rec, int world, int c,
        const float* __restrict__ lse_q, float* __restrict__ lse_p,
        float it, unsigned long long seed, int* __restrict__ res) {
    INK2_PDL_ENTRY();
    if (threadIdx.x) return;
    for (int row = 0; row <= c; ++row) {
        float M = -INFINITY;
        for (int r = 0; r < world; ++r) M = fmaxf(M, rec[((size_t)row * world + r) * 4 + 0]);
        double S = 0.0;
        for (int r = 0; r < world; ++r) {
            const float m = rec[((size_t)row * world + r) * 4 + 0];
            if (m > -INFINITY) S += (double)rec[((size_t)row * world + r) * 4 + 1] * exp((double)(m - M));
        }
        lse_p[row] = M + (float)log(S);
    }
    int a = 0;
    for (int j = 0; j < c; ++j) {
        float tld = 0.f, dld = 0.f;
        for (int r = 0; r < world; ++r) {
            tld += rec[((size_t)j * world + r) * 4 + 2];
            dld += rec[((size_t)j * world + r) * 4 + 3];
        }
        const double lp = (double)(tld * it) - (double)lse_p[j];
        const double lq = (double)(dld * it) - (double)lse_q[j];
        const double u = ink6_u01(seed, (unsigned long long)j);
        if (!(log(u) < lp - lq)) break;
        ++a;
    }
    res[0] = a;
}

// Vocab-parallel sampled accept (Stage 2 final draw): same as ink6_final_part over this rank's
// vocabulary slice [voff, voff + vn) using global id `voff + i` for the exact TP1 Gumbel hash.
// grid (G, 1, 1), block 256 -> (pmax, pidx) partials for ink_tp_gumbel_rec.
extern "C" __global__ __launch_bounds__(256) void ink6_tp_final_part(
        const float* __restrict__ tl, int ldl, const float* __restrict__ dl, int ldd,
        int vn, int voff, const float* __restrict__ lse_p, const float* __restrict__ lse_q,
        float it, const int* __restrict__ res, int c, unsigned long long seed,
        float* __restrict__ pmax, int* __restrict__ pidx) {
    INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int a = res[0];
    const bool resid = a < c;
    const int G = gridDim.x;
    const int per = (vn + G - 1) / G;
    const int i0 = blockIdx.x * per, i1 = min(vn, i0 + per);
    const float* tr = tl + (size_t)a * ldl;
    const float* dr = dl + (size_t)a * ldd;
    const float lp0 = lse_p[a], lq0 = resid ? lse_q[a] : 0.f;
    const unsigned long long sd = ink3_mix(seed ^ (unsigned long long)(a + 1) * 0x9e3779b97f4a7c15ULL);
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {
        float lw;
        const float lp = tr[i] * it - lp0;
        if (resid) {
            const float lq = dr[i] * it - lq0;
            if (lp <= lq) continue;
            lw = lp + log1pf(-expf(lq - lp));
        } else lw = lp;
        const int gi = voff + i;
        const unsigned long long h = ink3_mix(sd ^ ink3_mix((unsigned long long)gi));
        const double wv = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        const double e = -log1p(-wv);
        const float v = lw - (float)log(e);
        if (v > best) { best = v; bi = gi; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float x = sv[threadIdx.x], y = sv[threadIdx.x + s];
            const int ix = si[threadIdx.x], iy = si[threadIdx.x + s];
            if (y > x || (y == x && iy < ix)) { sv[threadIdx.x] = y; si[threadIdx.x] = iy; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}

// -----------------------------------------------------------------------------------------
// Drop 2 (#2): Sequence-Parallel Prefill Short Causal Conv (k = 4) + Residual
// -----------------------------------------------------------------------------------------
// Computes `out[t] = base[t] + (sum_{lag=0..3} w[lag] * x_{t-lag} + x_t) * div` on a rank's
// [my_r0 .. my_r1) row slice (`my_m` rows). When `t - lag < 0`:
//   - on rank > 0 (`halo != nullptr`): reads `halo[3 + t - lag]` (the previous rank's last 3 rows);
//   - on rank 0 (`halo == nullptr`): reads `ring[(p - lag) & ring_mask]` when `p - lag >= 0`, else 0.
// Bit-identical to `ink_sconv` on the full chunk. grid (ceil(C / 256), my_m), block 256.
extern "C" __global__ __launch_bounds__(256) void ink_sconv_sp(
        const float* __restrict__ x, const float* __restrict__ halo,
        const float* __restrict__ ring, int ring_mask, const float* __restrict__ w,
        const float* __restrict__ base, float* __restrict__ out, int C, int p0, float div) {
    INK2_PDL_ENTRY();
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y, p = p0 + t;
    if (c >= C) return;
    const float x0 = x[(size_t)t * C + c];
    float acc = 0.f;
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        float xv;
        if (lag == 0) xv = x0;
        else if (t - lag >= 0) xv = x[(size_t)(t - lag) * C + c];
        else if (halo) xv = halo[(size_t)(3 + t - lag) * C + c];
        else if (p - lag >= 0) xv = ring[(size_t)((p - lag) & ring_mask) * C + c];
        else xv = 0.f;
        acc = fmaf(xv, w[lag * C + c], acc);
    }
    const float y = (acc + x0) * div;
    out[(size_t)t * C + c] = base ? base[(size_t)t * C + c] + y : y;
}

// =============================================================================
// DeepSpeed-Ulysses All-to-All Sequence-Parallel Attention Pack / Unpack Kernels
// Credit: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang,
//         Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He (2023),
//         "DeepSpeed Ulysses: System Optimizations for Enabling Training of
//          Extreme Long Sequence Transformer Models", arXiv:2309.14509.
//
// In SP prefill (world in {4, 8}), each rank projects its own `my_m` tokens across
// all heads (`proj_full` -> `y_full`), transposes `[my_m, H_all] -> [m, H_local]`
// via 1-RTT dual-rail All-to-All, runs local head-sharded attention (`attn_prep` +
// `attn_tc2`), transposes BF16 `att` `[m, H_local] -> [my_m, H_all]` via 1-RTT
// All-to-All, and projects `wo_full` locally with zero attention AG/RS collectives.
// =============================================================================

struct InkUlyBlocks {
    int r0[8];
    int m[8];
};

// Per-rank query/KV head layout for arbitrary-W Ulysses (TP2 / TP3 / TP4 / TP6 / TP8).
struct InkUlyHeads {
    int h0[8];
    int nq[8];
    int g0[8];
    int nkv[8];
};

// Pack `y_full[my_m, ldy_f]` into `a2a_src[world][max_my_m][y_stride]` (FP32, 100% bit-exact, arbitrary W).
// U1 optimization: `gate_attn` stays local on the token-owner rank and is applied post-A2A #2 in
// `ink_ulysses_att_unpack`, cutting A2A #1 wire by 35.4% while keeping bit-exact FP32*sigmoid rounding.
// Grid: (world, my_m, 1), block: 256.
extern "C" __global__ __launch_bounds__(256) void ink_ulysses_y_pack(
        const float* __restrict__ y_full, int ldy_f,
        float* __restrict__ a2a_src, int y_stride, int max_my_m,
        InkUlyHeads hds,
        int oq, int orr, int os, int ok, int okd, int ov, int ovd) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    const float* src_row = y_full + (size_t)t * ldy_f;
    float* dst = a2a_src + ((size_t)p * max_my_m + t) * y_stride;
    const int h0 = hds.h0[p], nq = hds.nq[p], g0 = hds.g0[p], nkv = hds.nkv[p];
    const int qlen = nq * 128;
    const int rlen = nq * 16;
    const int slen = nq;
    const int klen = nkv * 128;
    const int dlen = (okd >= 0) ? (nkv * 4) : 0;
    const int c0 = 0;
    const int c1 = c0 + qlen;
    const int c2 = c1 + rlen;
    const int c3 = c2 + slen;
    const int c4 = c3 + klen;
    const int c5 = c4 + dlen;
    const int c6 = c5 + klen;
    const int c7 = c6 + dlen;
    for (int j = threadIdx.x; j < c7; j += blockDim.x) {
        int col;
        if (j < c1)      col = oq  + h0 * 128 + (j - c0);
        else if (j < c2) col = orr + h0 * 16  + (j - c1);
        else if (j < c3) col = os  + h0       + (j - c2);
        else if (j < c4) col = ok  + g0 * 128 + (j - c3);
        else if (j < c5) col = okd + g0 * 4   + (j - c4);
        else if (j < c6) col = ov  + g0 * 128 + (j - c5);
        else             col = ovd + g0 * 4   + (j - c6);
        dst[j] = src_row[col];
    }
}

// Unpack `a2a_dst[world][max_my_m][y_stride]` into `ws_y[m, ldy_tp]` at TP head offsets (no `gate_attn`).
// Grid: (world, max_my_m, 1), block: 256.
extern "C" __global__ __launch_bounds__(256) void ink_ulysses_y_unpack(
        const float* __restrict__ a2a_dst, int y_stride, int max_my_m,
        float* __restrict__ ws_y, int ldy_tp,
        int nq, int nkv,
        int oq, int orr, int os, int ok, int okd, int ov, int ovd,
        InkUlyBlocks blk) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= blk.m[p]) return;
    const float* src = a2a_dst + ((size_t)p * max_my_m + t) * y_stride;
    float* dst_row = ws_y + (size_t)(blk.r0[p] + t) * ldy_tp;
    const int qlen = nq * 128;
    const int rlen = nq * 16;
    const int slen = nq;
    const int klen = nkv * 128;
    const int dlen = (okd >= 0) ? (nkv * 4) : 0;
    const int c0 = 0;
    const int c1 = c0 + qlen;
    const int c2 = c1 + rlen;
    const int c3 = c2 + slen;
    const int c4 = c3 + klen;
    const int c5 = c4 + dlen;
    const int c6 = c5 + klen;
    const int c7 = c6 + dlen;
    for (int j = threadIdx.x; j < c7; j += blockDim.x) {
        int col;
        if (j < c1)      col = oq  + (j - c0);
        else if (j < c2) col = orr + (j - c1);
        else if (j < c3) col = os  + (j - c2);
        else if (j < c4) col = ok  + (j - c3);
        else if (j < c5) col = okd + (j - c4);
        else if (j < c6) col = ov  + (j - c5);
        else             col = ovd + (j - c6);
        dst_row[col] = src[j];
    }
}

// Pack ungated FP32 `att_f32[m, nq_hd]` into `a2a_src[world][max_my_m][att_stride]` by token-owner rank `p`.
// Grid: (world, max_my_m, 1), block: 256.
extern "C" __global__ __launch_bounds__(256) void ink_ulysses_att_pack(
        const float* __restrict__ att_f32,
        float* __restrict__ a2a_src,
        int max_my_m, int nq_hd, int att_stride, InkUlyBlocks blk) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= blk.m[p]) return;
    const float* src = att_f32 + (size_t)(blk.r0[p] + t) * nq_hd;
    float* dst = a2a_src + ((size_t)p * max_my_m + t) * att_stride;
    for (int j = threadIdx.x; j < nq_hd; j += blockDim.x) dst[j] = src[j];
}

// Unpack ungated FP32 `a2a_dst[world][max_my_m][att_stride]`, apply local `sigmoid(y_full[og_f + ...])`,
// and round to BF16 `att_full[my_m, 24 * 128]` — bit-identical to TP1 attention epilogue on TP2/TP3/TP4/TP6/TP8!
// Grid: (world, my_m, 1), block: 256.
extern "C" __global__ __launch_bounds__(256) void ink_ulysses_att_unpack(
        const float* __restrict__ a2a_dst,
        const float* __restrict__ y_full, int ldy_f, int og_f,
        unsigned short* __restrict__ att_full,
        int max_my_m, int att_stride, int all_hd, InkUlyHeads hds) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    const int col0 = hds.h0[p] * 128;
    const int pnq_hd = hds.nq[p] * 128;
    const float* src = a2a_dst + ((size_t)p * max_my_m + t) * att_stride;
    const float* gate = y_full + (size_t)t * ldy_f + og_f + col0;
    unsigned short* dst = att_full + (size_t)t * all_hd + col0;
    for (int j = threadIdx.x; j < pnq_hd; j += blockDim.x) dst[j] = ika_f2b(src[j] * ika_sig(gate[j]));
}

// =============================================================================
// U7a: Opt-in BF16 Ulysses Wire Compression (`--inkling2-ulysses-wire bf16`, arbitrary W)
// Packs Q, K, V in BF16 (2 B) and keeps r, s, k_delta, v_delta in full FP32 (4 B),
// and ships ungated attention output in BF16 on A2A #2.
// =============================================================================

extern "C" __global__ __launch_bounds__(256) void ink_ulysses_y_pack_b16(
        const float* __restrict__ y_full, int ldy_f,
        unsigned char* __restrict__ a2a_src, int row_bytes, int max_my_m,
        InkUlyHeads hds,
        int oq, int orr, int os, int ok, int okd, int ov, int ovd) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    const float* src_row = y_full + (size_t)t * ldy_f;
    unsigned char* dst_row = a2a_src + ((size_t)p * max_my_m + t) * row_bytes;
    const int h0 = hds.h0[p], nq = hds.nq[p], g0 = hds.g0[p], nkv = hds.nkv[p];
    const int qlen = nq * 128;
    const int klen = nkv * 128;
    const int b16_n = qlen + 2 * klen;
    unsigned short* dst16 = (unsigned short*)dst_row;
    for (int j = threadIdx.x; j < b16_n; j += blockDim.x) {
        int col;
        if (j < qlen)             col = oq + h0 * 128 + j;
        else if (j < qlen + klen) col = ok + g0 * 128 + (j - qlen);
        else                      col = ov + g0 * 128 + (j - qlen - klen);
        dst16[j] = ika_f2b(src_row[col]);
    }
    const int rlen = nq * 16;
    const int slen = nq;
    const int dlen = (okd >= 0) ? (nkv * 4) : 0;
    const int f32_n = rlen + slen + 2 * dlen;
    float* dst32 = (float*)(dst_row + (size_t)b16_n * 2);
    for (int j = threadIdx.x; j < f32_n; j += blockDim.x) {
        int col;
        if (j < rlen)                    col = orr + h0 * 16 + j;
        else if (j < rlen + slen)        col = os  + h0      + (j - rlen);
        else if (j < rlen + slen + dlen) col = okd + g0 * 4  + (j - rlen - slen);
        else                             col = ovd + g0 * 4  + (j - rlen - slen - dlen);
        dst32[j] = src_row[col];
    }
}

extern "C" __global__ __launch_bounds__(256) void ink_ulysses_y_unpack_b16(
        const unsigned char* __restrict__ a2a_dst, int row_bytes, int max_my_m,
        float* __restrict__ ws_y, int ldy_tp,
        int nq, int nkv,
        int oq, int orr, int os, int ok, int okd, int ov, int ovd,
        InkUlyBlocks blk) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= blk.m[p]) return;
    const unsigned char* src_row = a2a_dst + ((size_t)p * max_my_m + t) * row_bytes;
    float* dst_row = ws_y + (size_t)(blk.r0[p] + t) * ldy_tp;
    const int qlen = nq * 128;
    const int klen = nkv * 128;
    const int b16_n = qlen + 2 * klen;
    const unsigned short* src16 = (const unsigned short*)src_row;
    for (int j = threadIdx.x; j < b16_n; j += blockDim.x) {
        int col;
        if (j < qlen)            col = oq + j;
        else if (j < qlen + klen) col = ok + (j - qlen);
        else                     col = ov + (j - qlen - klen);
        dst_row[col] = ika_b2f(src16[j]);
    }
    const int rlen = nq * 16;
    const int slen = nq;
    const int dlen = (okd >= 0) ? (nkv * 4) : 0;
    const int f32_n = rlen + slen + 2 * dlen;
    const float* src32 = (const float*)(src_row + (size_t)b16_n * 2);
    for (int j = threadIdx.x; j < f32_n; j += blockDim.x) {
        int col;
        if (j < rlen)                   col = orr + j;
        else if (j < rlen + slen)       col = os  + (j - rlen);
        else if (j < rlen + slen + dlen) col = okd + (j - rlen - slen);
        else                            col = ovd + (j - rlen - slen - dlen);
        dst_row[col] = src32[j];
    }
}

extern "C" __global__ __launch_bounds__(256) void ink_ulysses_att_pack_b16(
        const float* __restrict__ att_f32,
        unsigned short* __restrict__ a2a_src,
        int max_my_m, int nq_hd, int att_stride, InkUlyBlocks blk) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= blk.m[p]) return;
    const float* src = att_f32 + (size_t)(blk.r0[p] + t) * nq_hd;
    unsigned short* dst = a2a_src + ((size_t)p * max_my_m + t) * att_stride;
    for (int j = threadIdx.x; j < nq_hd; j += blockDim.x) dst[j] = ika_f2b(src[j]);
}

extern "C" __global__ __launch_bounds__(256) void ink_ulysses_att_unpack_b16(
        const unsigned short* __restrict__ a2a_dst,
        const float* __restrict__ y_full, int ldy_f, int og_f,
        unsigned short* __restrict__ att_full,
        int max_my_m, int att_stride, int all_hd, InkUlyHeads hds) {
    INK2_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    const int col0 = hds.h0[p] * 128;
    const int pnq_hd = hds.nq[p] * 128;
    const unsigned short* src = a2a_dst + ((size_t)p * max_my_m + t) * att_stride;
    const float* gate = y_full + (size_t)t * ldy_f + og_f + col0;
    unsigned short* dst = att_full + (size_t)t * all_hd + col0;
    for (int j = threadIdx.x; j < pnq_hd; j += blockDim.x) dst[j] = ika_f2b(ika_b2f(src[j]) * ika_sig(gate[j]));
}
