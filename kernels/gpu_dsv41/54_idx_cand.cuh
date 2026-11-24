// =============================================================================
// kernels/gpu_dsv41/54_idx_cand.cuh — leg KI: the layer-20 candidate blocks (model.py
// `select_candidate_blocks`, PLAN/DSV41_REF_MAP.md §3.5) and their per-row-block union.
//
// d41_idx_cand (one CTA per row): blocks of 8 compressed positions, block score = the max of its
// visible scores (written by the score kernel with write_bmax); the block holding the row's newest
// visible position is pinned (+inf); the top min(2048, n_blocks) blocks are kept, ties to the LOWEST
// block id. Output: the kept block ids ascending, their count, and a bitmap per row. Below 2,049
// blocks (16,384 positions) every block is kept (the reference's no-op).
// Selection is a radix select on the 16-bit block score (two 8-bit histograms) and an ordered
// compaction in block-id order, so the output is ascending without a sort.
//
// d41_idx_union (one CTA per row block): OR of the rows' bitmaps -> ascending list of entries
// (block << 8 | row mask), which the score kernel visits in mode 1.
// =============================================================================
#pragma once

#define D41_IDX_CT 512   // threads of the cand / union kernels (16 warps)

extern "C" __global__ void __launch_bounds__(D41_IDX_CT) d41_idx_cand(const d41_idx_args a) {
    D41_PDL_ENTRY();
    __shared__ int h1[256], h2[256];
    __shared__ int wgt[16], weq[16], wbase[16], wtake[16];
    __shared__ int s_T, s_q;
    const int b = blockIdx.x / a.R, r = blockIdx.x % a.R, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    if (b >= a.n_rb) return;
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    if (r >= rb.nrows) return;
    const long long row = rb.row0 + r;
    int n = d41_idx_nvis(((const int*)a.pos)[row], a.ratio);
    if (n > a.n_keys_cap) n = a.n_keys_cap;
    int nb = (n + D41_IDX_CAND_BLOCK - 1) / D41_IDX_CAND_BLOCK;
    if (nb > a.nb_cap) nb = a.nb_cap;
    int* cand = (int*)a.cand + row * D41_IDX_CAND_BLOCKS;
    unsigned* bm = (unsigned*)a.bm + row * a.nbw_cap;
    const unsigned short* bmax = (const unsigned short*)a.bmax + row * a.nb_cap;
    for (int i = tid; i < a.nbw_cap; i += D41_IDX_CT) bm[i] = 0u;
    __syncthreads();
    if (nb <= D41_IDX_CAND_BLOCKS) {
        for (int i = tid; i < D41_IDX_CAND_BLOCKS; i += D41_IDX_CT) cand[i] = i < nb ? i : -1;
        for (int i = tid; i < (nb + 31) / 32; i += D41_IDX_CT) {
            const int rem = nb - i * 32;
            bm[i] = rem >= 32 ? 0xFFFFFFFFu : ((1u << rem) - 1u);
        }
        if (tid == 0) ((int*)a.cand_cnt)[row] = nb;
        return;
    }
    const int M = nb - 1, need = D41_IDX_CAND_BLOCKS - 1;   // blocks [0, M) compete; block M is pinned
    for (int i = tid; i < 256; i += D41_IDX_CT) { h1[i] = 0; h2[i] = 0; }
    __syncthreads();
    for (int i = tid; i < M; i += D41_IDX_CT) atomicAdd(&h1[bmax[i] >> 8], 1);
    __syncthreads();
    if (tid == 0) {
        int cum = 0, H = 255;
        for (; H > 0; H--) { if (cum + h1[H] >= need) break; cum += h1[H]; }
        s_T = H; s_q = cum;   // s_q = count strictly above high byte H
    }
    __syncthreads();
    const int H = s_T, gt1 = s_q;
    for (int i = tid; i < M; i += D41_IDX_CT) { const unsigned v = bmax[i]; if ((int)(v >> 8) == H) atomicAdd(&h2[v & 255], 1); }
    __syncthreads();
    if (tid == 0) {
        int cum = gt1, L = 255;
        for (; L > 0; L--) { if (cum + h2[L] >= need) break; cum += h2[L]; }
        s_T = (H << 8) | L; s_q = need - cum;   // ties at T to take, lowest ids first
    }
    __syncthreads();
    const unsigned T = (unsigned)s_T;
    const int q = s_q;
    // ordered compaction over warp segments of [0, M)
    const int seg = (M + 15) / 16, s0 = w * seg, s1 = min(M, s0 + seg);
    int cg = 0, ce = 0;
    for (int i0 = s0; i0 < s1; i0 += 32) {
        const int i = i0 + lane;
        const unsigned v = i < s1 ? bmax[i] : 0u;
        cg += __popc(__ballot_sync(0xFFFFFFFFu, i < s1 && v > T));
        ce += __popc(__ballot_sync(0xFFFFFFFFu, i < s1 && v == T));
    }
    if (lane == 0) { wgt[w] = cg; weq[w] = ce; }
    __syncthreads();
    if (tid == 0) {
        int eqb = 0, kb = 0;
        for (int k = 0; k < 16; k++) {
            int tk = q - eqb; tk = tk < 0 ? 0 : (tk > weq[k] ? weq[k] : tk);
            wtake[k] = tk; wbase[k] = kb;
            eqb += weq[k]; kb += wgt[k] + tk;
        }
    }
    __syncthreads();
    int base = wbase[w], eq_seen = 0;
    const int take = wtake[w];
    const unsigned lt = (1u << lane) - 1u;
    for (int i0 = s0; i0 < s1; i0 += 32) {
        const int i = i0 + lane;
        const unsigned v = i < s1 ? bmax[i] : 0u;
        const bool isg = i < s1 && v > T, ise = i < s1 && v == T;
        const unsigned me = __ballot_sync(0xFFFFFFFFu, ise);
        const bool keep = isg || (ise && eq_seen + __popc(me & lt) < take);
        const unsigned mk = __ballot_sync(0xFFFFFFFFu, keep);
        if (keep) {
            const int o = base + __popc(mk & lt);
            cand[o] = i;
            atomicOr(&bm[i >> 5], 1u << (i & 31));
        }
        base += __popc(mk);
        eq_seen += __popc(me);
    }
    if (tid == 0) {
        cand[need] = M;
        atomicOr(&bm[M >> 5], 1u << (M & 31));
        ((int*)a.cand_cnt)[row] = D41_IDX_CAND_BLOCKS;
    }
}

extern "C" __global__ void __launch_bounds__(D41_IDX_CT) d41_idx_union(const d41_idx_args a) {
    D41_PDL_ENTRY();
    __shared__ int wcnt[16], wbase[17];
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    if (b >= a.n_rb) return;
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    int nbw = 0;
    for (int r = 0; r < rb.nrows && r < a.R; r++) {
        int n = d41_idx_nvis(((const int*)a.pos)[rb.row0 + r], a.ratio);
        if (n > a.n_keys_cap) n = a.n_keys_cap;
        int nb = (n + D41_IDX_CAND_BLOCK - 1) / D41_IDX_CAND_BLOCK;
        if (nb > a.nb_cap) nb = a.nb_cap;
        nbw = max(nbw, (nb + 31) / 32);
    }
    const int nrows = min(rb.nrows, a.R);
    auto word = [&](int i) -> unsigned {
        unsigned u = 0;
        for (int r = 0; r < nrows; r++) u |= ((const unsigned*)a.bm)[(long long)(rb.row0 + r) * a.nbw_cap + i];
        return u;
    };
    const int seg = (nbw + 15) / 16, s0 = w * seg, s1 = min(nbw, s0 + seg);
    int cnt = 0;
    for (int i = s0 + lane; i < s1; i += 32) cnt += __popc(word(i));
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xFFFFFFFFu, cnt, o);
    if (lane == 0) wcnt[w] = cnt;
    __syncthreads();
    if (tid == 0) { wbase[0] = 0; for (int k = 0; k < 16; k++) wbase[k + 1] = wbase[k] + wcnt[k]; }
    __syncthreads();
    const int total = wbase[16];
    if (total > a.ucap && tid == 0) d41_idx_err(a, D41_IDX_ERR_UCAP);
    unsigned* out = (unsigned*)a.uni + (long long)b * a.ucap;
    int base = wbase[w];
    for (int i0 = s0; i0 < s1; i0 += 32) {
        const int i = i0 + lane;
        unsigned u = i < s1 ? word(i) : 0u;
        int c = __popc(u), x = c;
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xFFFFFFFFu, x, o); if (lane >= o) x += y; }
        int o = base + x - c;
        while (u) {
            const int j = __ffs(u) - 1;
            u &= u - 1;
            unsigned mask = 0;
            for (int r = 0; r < nrows; r++)
                mask |= ((((const unsigned*)a.bm)[(long long)(rb.row0 + r) * a.nbw_cap + i] >> j) & 1u) << r;
            if (o < a.ucap) out[o] = ((unsigned)(i * 32 + j) << 8) | mask;
            o++;
        }
        base += __shfl_sync(0xFFFFFFFFu, x, 31);
    }
    if (tid == 0) ((int*)a.ucnt)[b] = total < a.ucap ? total : a.ucap;
}
