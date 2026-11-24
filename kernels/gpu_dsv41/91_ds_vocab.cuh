// d41_ds_vpart / d41_ds_vrank / d41_ds_vmerge / d41_ds_accept (leg DS): the vocab passes of the DSpark round
// over THIS rank's vocab shard [v0, v0 + vlocal), the per-rank reduce, the cross-rank merge (the exchange of
// T0 §3.4: one 64-B record per row and rank, all-gathered by the TP layer; at TP1 the rank's records are the
// gathered buffer) and the per-lane accept. See 90_dspark.cuh for the semantics.
#pragma once

#define D41_DS_MODE_VERIFY 0   // target rows of a verify: statistics, the bonus draw, the probe = that row's draft
#define D41_DS_MODE_DRAFT 1    // drafter rows: statistics and the draft draw (draft namespace)
#define D41_DS_MODE_RESID 2    // the rejected row of each lane: the residual max(q - alpha p, 0) draw

struct d41_ds_vpart_args {
    const float* logits;            // row (l, j) at logits + l * lane_stride + j * row_stride (f32, this rank's shard)
    long long lane_stride, row_stride;
    const float* dlogits;           // residual: the draft rows, (l, j) at dlogits + l * d_lane_stride + j * d_row_stride
    long long d_lane_stride, d_row_stride;
    const d41_ds_lane* lanes;       // [nlanes]
    const float* cut_a;             // target / draft cut per (l, j): [nlanes][cut_rpl] (nullable: no cut)
    const int* cut_id;
    const float* dcut_a;            // residual: the draft rows' cut [nlanes][D41_DS_BLOCK] (nullable: none)
    const int* dcut_id;
    const float* dlse;              // residual: the draft rows' log-sum-exp [nlanes][D41_DS_BLOCK]
    const d41_ds_vrec* tres;        // residual: the target rows' merged records [nlanes][rpl_t]
    const int* ids;                 // the ids table [6][nlanes] (verify probes: ids[(j + 1) * nlanes + l]; nullable)
    d41_ds_vrec* part;              // [rows][gridDim.x]
    int nlanes, rpl, j0, cut_rpl;   // rows = nlanes * rpl (verify / draft) or nlanes (residual); row j = j0 + r % rpl
    int vlocal, v0, chunk, mode;
    int rpl_t;
    float alpha;
    int stop_ids[4];                // leg P7.2: model EOS / stop token ids (-1 = unused)
};
static_assert(sizeof(d41_ds_vpart_args) == 15 * 8 + 14 * 4, "d41_ds_vpart_args layout");

__device__ __forceinline__ bool d41_ds_is_stop(int id, const int (&stop_ids)[4]) {
    return id == stop_ids[0] || id == stop_ids[1] || id == stop_ids[2] || id == stop_ids[3];
}

// The statistics record of one (row, block) for the verify / draft modes, reduced by the calling CTA
// (D41_DS_THREADS threads; thread t takes ids lo + t, lo + t + 256, ...; the fixed butterfly, then the warps in
// order): `out` holds it in thread 0. d41_ds_vpart's per-row scan (LDCG: read the row through L2).
template <bool LDCG>
__device__ __forceinline__ void d41_ds_row_scan(d41_ds_vrec& acc, const d41_ds_lane& L, int mode, int j, const float* row, int lo, int hi, int v0,
                                                float ca, int cid, int probe, bool sup_eos, const int (&stop_ids)[4]) {
    const int t = threadIdx.x;
    const float inv_t = L.inv_t;
    const bool greedy = inv_t == 0.0f;
    const long long tpos = mode == D41_DS_MODE_DRAFT ? (long long)L.wend + 2 + j : (long long)L.vbase + 1 + j;
    const unsigned long long seed = mode == D41_DS_MODE_DRAFT ? d41_ds_draft_seed(L) : L.seed;
    const unsigned long long key = d41_ds_key(seed, (unsigned long long)tpos * 3ull);
    for (int v = lo + t; v < hi; v += D41_DS_THREADS) {
        const int id = v0 + v;
        const bool masked = sup_eos && d41_ds_is_stop(id, stop_ids);
        float x0 = masked ? -INFINITY : (LDCG ? __ldcg(row + v) : row[v]);
        if (x0 != x0) x0 = -INFINITY;
        if (!masked && (x0 > acc.amax || (x0 == acc.amax && id < acc.aid))) { acc.amax = x0; acc.aid = id; }
        const bool kept = x0 > ca || (x0 == ca && id <= cid);
        if (!greedy && kept && x0 != -INFINITY) {
            const float x = __fmul_rn(x0, inv_t);
            if (x > acc.m) { acc.s = __fadd_rn(acc.m == -INFINITY ? 0.0f : __fmul_rn(acc.s, expf(acc.m - x)), 1.0f); acc.m = x; }
            else acc.s = __fadd_rn(acc.s, expf(x - acc.m));
            const float sc = __fadd_rn(x, d41_ds_gumbel(key, id));
            if (sc > acc.gmax || (sc == acc.gmax && id < acc.gid)) { acc.gmax = sc; acc.gid = id; acc.gx = x; }
        }
        if (id == probe) { acc.pheld = 1; acc.px = greedy ? x0 : (kept && x0 != -INFINITY ? __fmul_rn(x0, inv_t) : -INFINITY); }
    }
}

// the CTA reduction of the threads' records (the fixed butterfly, then the warps in ascending order) into thread 0
__device__ __forceinline__ void d41_ds_cta_merge(d41_ds_vrec& acc, d41_ds_vrec* s_w) {
    const int t = threadIdx.x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) d41_ds_rec_shfl_merge(acc, o);
    if ((t & 31) == 0) s_w[t >> 5] = acc;
    __syncthreads();
    if (t == 0) {
        d41_ds_vrec m = s_w[0];
#pragma unroll
        for (int w = 1; w < D41_DS_THREADS / 32; ++w) d41_ds_rec_merge(m, s_w[w]);
        acc = m;
    }
}

// grid (nblk, rows), D41_DS_THREADS threads: block b reduces local ids [b * chunk, min((b + 1) * chunk, vlocal)).
extern "C" __global__ void __launch_bounds__(D41_DS_THREADS) d41_ds_vpart(d41_ds_vpart_args a) {
    D41_PDL_ENTRY();
    const int b = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    __shared__ d41_ds_vrec s_w[D41_DS_THREADS / 32];
    d41_ds_vrec acc;
    d41_ds_rec_empty(acc);
    int l, j;
    bool act;
    if (a.mode == D41_DS_MODE_RESID) {
        l = r;
        j = a.lanes[l].n_acc;
        act = a.lanes[l].active && a.lanes[l].need_res;
    } else {
        l = r / a.rpl;
        j = a.j0 + r % a.rpl;
        act = a.lanes[l].active && (a.mode == D41_DS_MODE_DRAFT || j <= a.lanes[l].k);
    }
    if (act) {
        const d41_ds_lane& L = a.lanes[l];
        const float inv_t = L.inv_t;
        const bool sup_eos = (j < L.suppress_eos) && (a.stop_ids[0] >= 0);
        const long long tpos = a.mode == D41_DS_MODE_DRAFT ? (long long)L.wend + 2 + j : (long long)L.vbase + 1 + j;
        const unsigned long long seed = a.mode == D41_DS_MODE_DRAFT ? d41_ds_draft_seed(L) : L.seed;
        const unsigned long long key = d41_ds_key(seed, (unsigned long long)tpos * 3ull + (a.mode == D41_DS_MODE_RESID ? 1ull : 0ull));
        const int lo = b * a.chunk, hi = min(lo + a.chunk, a.vlocal);
        const float* row = (a.mode != D41_DS_MODE_DRAFT && L.trow0_p1 > 0)
            ? (a.logits + (long long)(L.trow0_p1 - 1 + j) * a.row_stride)
            : (a.logits + (long long)l * a.lane_stride + (long long)j * a.row_stride);
        const int ci = l * a.cut_rpl + j;
        const float ca = a.cut_a ? a.cut_a[ci] : -INFINITY;
        const int cid = a.cut_id ? a.cut_id[ci] : 0x7FFFFFFF;
        if (a.mode != D41_DS_MODE_RESID) {
            const int probe = (a.mode == D41_DS_MODE_VERIFY && a.ids != nullptr && j < L.k) ? a.ids[(j + 1) * a.nlanes + l] : -1;
            d41_ds_row_scan<false>(acc, L, a.mode, j, row, lo, hi, a.v0, ca, cid, probe, sup_eos, a.stop_ids);
        } else {
            const float* drow = a.dlogits + (long long)l * a.d_lane_stride + (long long)j * a.d_row_stride;
            const float lq = d41_ds_lse(a.tres[l * a.rpl_t + j]);
            const int di = l * D41_DS_BLOCK + j;
            const float lp = a.dlse[di];
            const float dca = a.dcut_a ? a.dcut_a[di] : -INFINITY;
            const int dcid = a.dcut_id ? a.dcut_id[di] : 0x7FFFFFFF;
            for (int v = lo + t; v < hi; v += D41_DS_THREADS) {
                const int id = a.v0 + v;
                const bool masked = sup_eos && d41_ds_is_stop(id, a.stop_ids);
                float aq = masked ? -INFINITY : row[v];
                float ap = masked ? -INFINITY : drow[v];
                if (aq != aq) aq = -INFINITY;
                if (ap != ap) ap = -INFINITY;
                const bool kq = (aq > ca || (aq == ca && id <= cid)) && aq != -INFINITY;
                const bool kp = (ap > dca || (ap == dca && id <= dcid)) && ap != -INFINITY && lp != -INFINITY;
                const float q = kq ? expf(__fmul_rn(aq, inv_t) - lq) : 0.0f;
                const float p = kp ? expf(__fmul_rn(ap, inv_t) - lp) : 0.0f;
                const float rr = __fsub_rn(q, __fmul_rn(a.alpha, p));
                if (rr > 0.0f) {
                    acc.rsum = __fadd_rn(acc.rsum, rr);
                    const float sc = __fadd_rn(logf(rr), d41_ds_gumbel(key, id));
                    if (sc > acc.rmax || (sc == acc.rmax && id < acc.rid)) { acc.rmax = sc; acc.rid = id; }
                }
            }
        }
    }
    d41_ds_cta_merge(acc, s_w);
    if (t == 0) a.part[(long long)r * gridDim.x + b] = acc;
}

struct d41_ds_vrank_args {
    const d41_ds_vrec* part;        // [rows][nblk]
    d41_ds_vrec* out;               // [rows] this rank's records (the exchange's send buffer)
    int rows, nblk;
};
// grid (ceil(rows / 32)), 32 threads: one thread per row, blocks merged in ascending order.
extern "C" __global__ void __launch_bounds__(32) d41_ds_vrank(d41_ds_vrank_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x * 32 + threadIdx.x;
    if (r >= a.rows) return;
    d41_ds_vrec m = a.part[(long long)r * a.nblk];
    for (int b = 1; b < a.nblk; ++b) d41_ds_rec_merge(m, a.part[(long long)r * a.nblk + b]);
    a.out[r] = m;
}

#define D41_DS_KIND_STATS 0     // write the merged records (verify rows; probes)
#define D41_DS_KIND_PICK 1      // drafter rows: the drafted id, its log-probability and the row's log-sum-exp
#define D41_DS_KIND_FINISH 2    // residual rows: the token after a rejection, then the lane's round result

struct d41_ds_vmerge_args {
    const d41_ds_vrec* recs;        // [nranks][rows]: the gathered records
    d41_ds_vrec* res;               // [rows] merged (nullable for PICK / FINISH)
    d41_ds_lane* lanes;
    int* ids;                       // PICK: ids[(j + 1) * nlanes + l]
    float* dlp;                     // PICK: [nlanes][5] the drafted token's processed log-probability (x - lse)
    float* dlse;                    // PICK: [nlanes][5] the row's log-sum-exp over the kept set
    int* out_tok;                   // FINISH: [nlanes][D41_DS_OUT_TOK]
    const d41_ds_vrec* tres;        // FINISH: the target rows' records [nlanes][rpl_t]
    int nranks, rows, nlanes, rpl;
    int j0, kind, rpl_t, test_flags;
};
// grid (ceil(rows / 32)), 32 threads: one thread per row, ranks merged in ascending order.
extern "C" __global__ void __launch_bounds__(32) d41_ds_vmerge(d41_ds_vmerge_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x * 32 + threadIdx.x;
    if (r >= a.rows) return;
    d41_ds_vrec m = a.recs[r];
    for (int k = 1; k < a.nranks; ++k) d41_ds_rec_merge(m, a.recs[(long long)k * a.rows + r]);
    if (a.res) a.res[r] = m;
    if (a.kind == D41_DS_KIND_PICK) {
        const int l = r / a.rpl, j = a.j0 + r % a.rpl;
        if (!a.lanes[l].active) return;
        const bool greedy = a.lanes[l].inv_t == 0.0f;
        const float lse = d41_ds_lse(m);
        const int pick = greedy ? m.aid : m.gid;
        a.ids[(j + 1) * a.nlanes + l] = pick;
        a.dlp[l * D41_DS_BLOCK + j] = greedy ? 0.0f : __fsub_rn(m.gx, lse);
        a.dlse[l * D41_DS_BLOCK + j] = lse;
    } else if (a.kind == D41_DS_KIND_FINISH) {
        const int l = r;
        d41_ds_lane& L = a.lanes[l];
        if (!L.active) return;
        const int n = L.n_acc;
        L.zmass = 0;
        if (L.need_res) {
            L.zmass = (m.rsum > 0.0f && m.rid != 0x7FFFFFFF) ? 0 : 1;
            const d41_ds_vrec& tr = a.tres[l * a.rpl_t + n];
            int tok = (m.rsum > 0.0f && m.rid != 0x7FFFFFFF) ? m.rid : tr.aid;   // zero residual mass: the first max logit
            if (a.test_flags & 2) tok = tr.gid;                                  // negative control: a fresh draw from q
            a.out_tok[l * D41_DS_OUT_TOK + n] = tok;
        }
        L.n_emit = n + 1;
        L.wend = L.vbase + n;
        L.tok = a.out_tok[l * D41_DS_OUT_TOK + n];
        L.rounds += 1u;
        L.drafted += (unsigned)L.k;
        L.accepted += (unsigned)n;
    }
}

struct d41_ds_accept_args {
    d41_ds_lane* lanes;
    const d41_ds_vrec* tres;        // [nlanes][rpl_t] merged target rows (row j decides the token at vbase + 1 + j)
    const int* ids;                 // [6][nlanes]: ids[j + 1] = draft d_{j+1}
    const float* dlp;               // [nlanes][5] the drafts' processed log-probabilities
    int* out_tok;                   // [nlanes][D41_DS_OUT_TOK]
    int nlanes, rpl_t, test_flags;
    float pad;
};
// grid (nlanes), 32 threads (thread 0 walks the drafts: the decision is sequential and tiny).
extern "C" __global__ void __launch_bounds__(32) d41_ds_accept(d41_ds_accept_args a) {
    D41_PDL_ENTRY();
    const int l = blockIdx.x;
    if (threadIdx.x != 0 || l >= a.nlanes) return;
    d41_ds_lane& L = a.lanes[l];
    if (!L.active) return;
    const bool greedy = L.inv_t == 0.0f;
    const int k = L.k;
    int n = 0;
    for (; n < k; ++n) {
        const int d = a.ids[(n + 1) * a.nlanes + l];
        const d41_ds_vrec& tr = a.tres[l * a.rpl_t + n];
        bool ok;
        if (greedy) {
            ok = d == tr.aid;
        } else if (L.copy) {
            // leg CD1: a copied token is a function of the committed history, so it is kept iff it equals the target's own
            // keyed draw at its position (the draw a plain decode step makes there); a mismatch emits that draw (below)
            ok = d == tr.gid;
        } else {
            const float lq = tr.pheld ? __fsub_rn(tr.px, d41_ds_lse(tr)) : -INFINITY;
            const float lp = a.dlp[l * D41_DS_BLOCK + n];
            const float lr = (a.test_flags & 1) ? __fsub_rn(lp, lq) : __fsub_rn(lq, lp);   // flag 1: the swapped rule (negative control)
            const double u = d41_ds_uniform(L.seed, (unsigned long long)((long long)L.vbase + 1 + n) * 3ull + 2ull);
            ok = lq != -INFINITY && u < exp((double)fminf(lr, 0.0f));
        }
        if (!ok) break;
        a.out_tok[l * D41_DS_OUT_TOK + n] = d;
    }
    L.n_acc = n;
    L.need_res = 0;
    const d41_ds_vrec& tr = a.tres[l * a.rpl_t + n];
    if (n == k) a.out_tok[l * D41_DS_OUT_TOK + n] = greedy ? tr.aid : tr.gid;        // all accepted: the bonus draw
    else if (greedy) a.out_tok[l * D41_DS_OUT_TOK + n] = tr.aid;                     // greedy: the exact argmax
    else if (L.copy) a.out_tok[l * D41_DS_OUT_TOK + n] = tr.gid;                     // leg CD1: the plain decode's own draw, no residual
    else L.need_res = 1;                                                             // the residual draw follows
}
