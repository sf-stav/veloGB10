// exl3_bench/05_draft_head_pen.cuh: DHEADP penalized draft passes, embed/cos gathers, row splices, had_suh multi.
// DRAFT-ONLY: verify, its penalties and every emitted token are untouched.
// Section 5 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// =============================================================================
// DHEADP (w3, 2026-09-27): DHEAD for PENALIZED draft passes (the WP15 draft mirror) — DRAFT-ONLY
// (the verify, its penalties and every emitted token are untouched; only which token is drafted
// can change). The old penalized draft tail ran the FULL 65,536-id slice head (exl3_chain) so
// xq_pen_draft could rewrite the window's tokens before argmax_rows + xq_rowmax_f16.
//
// Bound. Under the served semantics (xq_pen_tok: rep >= 1 divides positive / multiplies negative
// logits, pres >= 0 and freq >= 0 subtract) a penalty never RAISES a logit v >= -65504 (only -inf
// rises, to -65504). So for a block b the screen's unpenalized upper bound
//     UB_b = bmax[b] + delta * sigma * max|svh_b|     (DHEAD's per-block screen-noise margin)
// also bounds every PENALIZED logit in b. Pass A rescores DHEAD's candidate set C1 exactly (the
// same selection, partials and combine as xq_dh_rescore), penalizes those logits with the draft
// mirror's per-token formula and counts, and takes their best penalized value L — a value some
// token really has, so a LOWER bound on the penalized maximum. A block outside C1 can hold the
// penalized argmax (or tie it with a lower id) only if UB_b >= L: pass B rescores exactly those
// blocks (C2), penalizes them, and folds them into the max. Blocks with UB_b < L cannot win.
// So the draft equals the old penalized head's draft whenever DHEAD's own screen-noise model holds
// for the blocks it excludes (the same assumption the unpenalized DHEAD makes; XCHECK measures it).
//
// Fallback (device-side, graph-safe): if more than `cap` blocks qualify, or the params break the
// monotonicity (rep < 1, pres < 0, freq < 0 — the host routes those requests to the old head, this
// is the belt-and-braces), or L is degenerate (NaN / <= -65504), C2 = EVERY block outside C1: the
// pass computes all N logits exactly, penalized — the old head's function, independent of the
// screen — at about the old head's cost. Counted per reason in pws (per-request log line).
// A row with pen[0] == 0 is unpenalized: C2 is empty and the result is xq_dh_rescore's, bit for bit.
// DETERMINISM: every value is a fixed-order computation (ascending-s split combine per candidate,
// integer window counts); the fold is a max over (value, id) keys — order-free.
//
// pws (u32 words; mirrored by src/exl3_forward/dhead.rs pws_layout):
#define DHP_BEST 0   // u64: running penalized best key (xq_am_key order)
#define DHP_N2 2     // pass-B candidate count (0 = C1 decided the pass)
#define DHP_DONE 3   // pass-B finished-candidate counter (self re-arming)
#define DHP_META 4   // {b, p, slot, 0}: pass A's copy of draft_meta
#define DHP_PEN 8    // row-0 penalty params (8 words): pass A's copy
#define DHP_DBG 16   // {ns1, n2, flags, L bits, nq, qualifying, K1 lo, K1 hi} (XCHECK readback)
#define DHP_STAT 24  // cumulative {passes, expanded, sum n2 expanded, fb overflow, fb non-monotone,
                     //             fb degenerate, sum n2 fallback, unpenalized-row passes}
#define DHP_HDR 32   // then cntc[nblk4] (per-candidate split counters), c2[nblk4], part[nblk][S][128]
#define DHP_F_OVER 1u
#define DHP_F_NONMONO 2u
#define DHP_F_DEGEN 4u
#define DHP_F_UNPEN 8u
#define DHP_F_FB 16u
__device__ __forceinline__ int dhp_nblk4(int nblk) { return (nblk + 3) & ~3; }

struct DhSel {                                       // dh_select's shared scratch
    unsigned om[8];
    int nq[8];
    int bs[2][8];
    int wh[DH_KPT * 8];
    int we[DH_KPT * 8];
    int ph[DH_KPT * 8];
    int pe[DH_KPT * 8];
    int tot[2];
    int cand[DH_TCAP];
};

// xq_dh_rescore's step 1, op for op (xq_dh_rescore itself is left byte-identical, so the
// unpenalized served PTX cannot move): the candidate blocks -> sm.cand[0..ns) (qualified above tau
// by ascending id, then the tau ties), returns ns; nq_out = the qualified count. 256 threads.
__device__ __forceinline__ int dh_select(const float* __restrict__ bmax, const float* __restrict__ bsig,
                                         int nblk, float sig, float delta, int tmax, DhSel& sm, int& nq_out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float bsmax = bsig[nblk];
    unsigned ob[DH_KPT];
    unsigned om = 0u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        float v = -INFINITY;
        if (b < nblk) { v = bmax[b]; if (!(v == v)) v = -INFINITY; }
        ob[i] = b < nblk ? dh_ord(v) : 0u;
        om = max(om, ob[i]);
    }
    om = __reduce_max_sync(0xffffffffu, om);
    if (lane == 0) sm.om[warp] = om;
    __syncthreads();
    unsigned go = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) go = max(go, sm.om[w]);
    const float gmax = dh_unord(go);
    unsigned qm = 0u;
    int qc = 0;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        if (b < nblk) {
            const float marg = fmaxf(delta * sig * (bsig[b] + bsmax), 0.f);   // NaN -> 0
            if (dh_unord(ob[i]) + marg >= gmax) { qm |= 1u << i; ++qc; }
        }
    }
    qc = __reduce_add_sync(0xffffffffu, qc);
    if (lane == 0) sm.nq[warp] = qc;
    __syncthreads();
    int nq = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nq += sm.nq[w];
    unsigned hm = qm, em = 0u;
    if (nq > tmax) {                                  // uniform branch
        unsigned tau = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned t = tau | (1u << bit);
            int c = 0;
            #pragma unroll
            for (int i = 0; i < DH_KPT; ++i) c += (((qm >> i) & 1u) && ob[i] >= t) ? 1 : 0;
            c = __reduce_add_sync(0xffffffffu, c);
            if (lane == 0) sm.bs[bit & 1][warp] = c;
            __syncthreads();
            int tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += sm.bs[bit & 1][w];
            if (tot >= tmax) tau = t;
        }
        hm = 0u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            if ((qm >> i) & 1u) {
                if (ob[i] > tau) hm |= 1u << i;
                else if (ob[i] == tau) em |= 1u << i;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
        const unsigned be = __ballot_sync(0xffffffffu, (em >> i) & 1u);
        if (lane == 0) { sm.wh[i * 8 + warp] = __popc(bh); sm.we[i * 8 + warp] = __popc(be); }
    }
    __syncthreads();
    if (tid < 2 * DH_KPT * 8) {
        const int e = tid & (DH_KPT * 8 - 1);
        const int* src = tid < DH_KPT * 8 ? sm.wh : sm.we;
        int pfx = 0;
        for (int f = 0; f < e; ++f) pfx += src[f];
        (tid < DH_KPT * 8 ? sm.ph : sm.pe)[e] = pfx;
        if (e == DH_KPT * 8 - 1) sm.tot[tid < DH_KPT * 8 ? 0 : 1] = pfx + src[e];
    }
    __syncthreads();
    const int nh = sm.tot[0];
    const int ns = min(min(nh + sm.tot[1], tmax), DH_TCAP);
    {
        const unsigned lt = (1u << lane) - 1u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const unsigned mh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
            const unsigned me = __ballot_sync(0xffffffffu, (em >> i) & 1u);
            const int b = tid + 256 * i;
            if ((hm >> i) & 1u) {
                const int rk = sm.ph[i * 8 + warp] + __popc(mh & lt);
                if (rk < ns) sm.cand[rk] = b;
            }
            if ((em >> i) & 1u) {
                const int rk = nh + sm.pe[i * 8 + warp] + __popc(me & lt);
                if (rk < ns) sm.cand[rk] = b;
            }
        }
    }
    __syncthreads();
    nq_out = nq;
    return ns;
}

// Ascending-id compaction of a per-thread block mask (bit i <-> block tid + 256 i): out[rank] = the
// block (global stores); returns the count. s_w / s_p: DH_KPT * 8 ints of smem; 256 threads.
__device__ __forceinline__ int dh_rank1(unsigned m, int* s_w, int* s_p, int* s_tot, unsigned* out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bm = __ballot_sync(0xffffffffu, (m >> i) & 1u);
        if (lane == 0) s_w[i * 8 + warp] = __popc(bm);
    }
    __syncthreads();
    if (tid < DH_KPT * 8) {
        int pfx = 0;
        for (int f = 0; f < tid; ++f) pfx += s_w[f];
        s_p[tid] = pfx;
        if (tid == DH_KPT * 8 - 1) *s_tot = pfx + s_w[tid];
    }
    __syncthreads();
    const unsigned lt = (1u << lane) - 1u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bm = __ballot_sync(0xffffffffu, (m >> i) & 1u);
        if ((m >> i) & 1u) out[s_p[i * 8 + warp] + __popc(bm & lt)] = (unsigned)(tid + 256 * i);
    }
    return *s_tot;
}

// xq_dh_rescore's step 2, op for op: the exact split-K partial of candidate block cb over k16 rows
// [kb0, kb0 + R). The caller has issued the s_x2 loads (R * 8 words of xh); the __syncthreads here
// publishes them with the ring. The caller syncs before s_ring4 is reused.
__device__ __forceinline__ void dh_part(const uint16_t* __restrict__ tr, int cb, int kb0, int R, int nb16,
                                        int bits, uint4* s_ring4, const unsigned* s_x2, float (&acc)[2][4]) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int U4 = bits; // k2: uint4 chunks per trellis row (== 2*BITS integer, == 2KA+1 half)
    const int rowu4 = 8 * U4;
    for (int idx = tid; idx < R * rowu4; idx += 256) {
        const int row = idx / rowu4, rem = idx - row * rowu4;
        const int w = rem / U4, q = rem - w * U4;
        const uint4* src = reinterpret_cast<const uint4*>(tr + ((size_t)(kb0 + row) * nb16 + (size_t)cb * 8) * (8 * bits));
        s_ring4[(w * R + row) * U4 + q] = __ldcs(src + rem);
    }
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    const uint32_t* ringw = reinterpret_cast<const uint32_t*>(s_ring4 + warp * R * U4);
    switch (bits) {
        case 4: dh_rescore_mma<2>(ringw, s_x2, R, lane, acc); break;
        case 3: dh_rescore_mma<1, true>(ringw, s_x2, R, lane, acc); break;
        case 6: dh_rescore_mma<3>(ringw, s_x2, R, lane, acc); break;
        case 8: dh_rescore_mma<4>(ringw, s_x2, R, lane, acc); break;
        case 10: dh_rescore_mma<5>(ringw, s_x2, R, lane, acc); break;
        case 12: dh_rescore_mma<6>(ringw, s_x2, R, lane, acc); break;
        default: break;
    }
}
// row 0 of the partial's two n8 tiles -> part_cs[128], then a device-scope fence (step 2's tail)
__device__ __forceinline__ void dh_part_store(float* __restrict__ part_cs, const float (&acc)[2][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if ((lane >> 2) == 0) {                           // cols 16w + 8h + 2tig
        float* pp = part_cs + warp * 16 + 2 * (lane & 3);
        pp[0] = acc[0][0]; pp[1] = acc[0][1]; pp[8] = acc[1][0]; pp[9] = acc[1][1];
        __threadfence();
    }
}

// The draft-mirror window of pass i (xq_pen_draft's past, xq_pen_span's factors): positions q in
// [max(0, past - S - D + 1), past), past = p + 1 + i; token(q) = H[q & rmask] (q < p), b (q == p),
// d_dev[q - p - 1] (q > p); ids outside the draft slice [0, N) are never penalized there.
// slot_of(t) = the smem counter index of t, or -1 (untracked). Integer atomics: order-free.
template <class SlotOf>
__device__ __forceinline__ void dhp_scan(const unsigned* __restrict__ P, const int* __restrict__ H, int rmask,
                                         int p, int b, const int* __restrict__ d_dev, int past, int N,
                                         SlotOf slot_of, unsigned* fm, unsigned* fs) {
    const int S = (int)P[4], D = (int)P[5];
    const int lo = max(0, past - S - D + 1);
    for (int q = lo + (int)threadIdx.x; q < past; q += blockDim.x) {
        const int t = q < p ? H[q & rmask] : (q == p ? b : d_dev[q - p - 1]);
        if (t < 0 || t >= N) continue;
        const int e = slot_of(t);
        if (e < 0) continue;
        const int dist = past - q;
        const unsigned f = D > 0 ? (unsigned)min(D, D + S - dist) : 1u;
        atomicMax(&fm[e], f);
        atomicAdd(&fs[e], f);
    }
}

// One candidate's 128 logits (xq_dh_rescore's step 3 ops: ascending-s partial sum -> f16 yraw ->
// had128 -> x svh -> f16), each penalized by xq_pen_tok where fm[c] != 0 (flag == 0: none), keyed
// (value desc, id asc). One warp; returns the warp-max key.
__device__ __forceinline__ unsigned long long dhp_combine(const float* __restrict__ part_c, int S, int bb,
                                                          const __half* __restrict__ svh,
                                                          const unsigned* fm, const unsigned* fs,
                                                          unsigned flag, float rep, float pres, float freq,
                                                          float scale, int lane) {
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float* pc = part_c + lane * 4 + q;
        float a = 0.f;
        for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
        v[q] = __half2float(__float2half_rn(a));      // yraw (exl3_gemm_body's f16 RN store)
    }
    exl3_had128(v, lane);                             // exl3_had_svh_dev's ops
    unsigned long long best = 0ull;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c = lane * 4 + q, col = bb * 128 + c;
        float y = __half2float(__float2half_rn(v[q] * __half2float(svh[col])));
        if (flag != 0u) {
            const unsigned m = fm[c];
            if (m != 0u) y = __half2float(__ushort_as_half(xq_pen_tok(y, m, fs, c, flag, rep, pres, freq, scale)));
        }
        const unsigned long long kk = xq_am_key(y, col);
        best = kk > best ? kk : best;
    }
    return xq_am_wmax(best);
}

// The pass's outputs from the final key (xq_dh_rescore's four writes, same formulas).
__device__ __forceinline__ void dhp_finish(unsigned* pws, int pass_i, int* out_id, float* dconf,
                                           int* d_dev, int* toks) {
    const unsigned long long k = atomicMax(reinterpret_cast<unsigned long long*>(pws + DHP_BEST), 0ull);
    const int id = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
    out_id[0] = id;                                   // sc.argmax
    dconf[pass_i] = dh_unord((unsigned)(k >> 32));    // == xq_rowmax_f16 of the penalized row's max
    d_dev[pass_i] = id;
    toks[0] = id;
}

// Pass A: grid (S, tmax), block 256, dynamic smem = max(R * 8 * 32 * hbits, 2 * tmax * 512) bytes.
// geomA = pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32 (16 b) | rlog2 << 48 (5 b) | cap << 53
// geomB = N | delta (f32 bits) << 32. pen = sc.pen (row 0), meta = sc.draft_meta, hist = sc.pen_hist.
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore_pa(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        const float* __restrict__ bsig, unsigned* __restrict__ ws, unsigned* __restrict__ pws,
        const unsigned* __restrict__ pen, const int* __restrict__ meta, const int* __restrict__ d_dev,
        const int* __restrict__ hist, long long geomA, long long geomB) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];            // [8 warps][R rows][2 * bits]
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ DhSel s_sel;
    __shared__ unsigned long long s_best[8];
    __shared__ unsigned char s_map[DH_KPT * 256];     // last CTA: block -> C1 rank (0xFF = not in C1)
    __shared__ int s_red[8];
    __shared__ int s_last;
    const int pass_i = (int)(geomA & 0xFF), bits = (int)((geomA >> 8) & 0xFF);
    const int S = (int)((geomA >> 16) & 0xFF), tmax = (int)((geomA >> 24) & 0xFF);
    const int K = (int)((geomA >> 32) & 0xFFFF), rlog2 = (int)((geomA >> 48) & 0x1F);
    const int cap = (int)((geomA >> 53) & 0x7FF);
    const int N = (int)(geomB & 0xFFFFFFFFll);
    const float delta = __uint_as_float((unsigned)((unsigned long long)geomB >> 32));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7, KW = K >> 2;
    const float* bmax = reinterpret_cast<const float*>(ws + dh_off_bmax(K));
    unsigned* cnt = ws + dh_off_cnt(K, nblk);
    int* dbg = reinterpret_cast<int*>(ws + dh_off_dbg(K, nblk));
    float* part = reinterpret_cast<float*>(ws + dh_off_part(K, nblk));
    const float sig = __uint_as_float(ws[KW + 2]);

    // ---- 1. C1 = DHEAD's candidate set (identical in every CTA) ----
    int nq = 0;
    const int ns = dh_select(bmax, bsig, nblk, sig, delta, tmax, s_sel, nq);
    if (blockIdx.x == 0 && blockIdx.y == 0) {         // XCHECK readback (DHEAD's dbg words)
        if (tid < ns) dbg[1 + tid] = s_sel.cand[tid];
        if (tid == 0) { dbg[0] = ns; dbg[DH_TCAP + 1] = nq; }
    }
    const int rc = blockIdx.y, sp = blockIdx.x;
    if (rc >= ns) return;                             // uniform per CTA

    // ---- 2. exact partial of candidate rc over K-split sp ----
    const int R = (K >> 4) / S, kb0 = sp * R;
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    float acc[2][4];
    dh_part(tr, s_sel.cand[rc], kb0, R, N >> 4, bits, s_ring4, s_x2, acc);
    dh_part_store(part + (size_t)(rc * S + sp) * 128, acc);
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = (atomicAdd(cnt, 1u) == (unsigned)(ns * S) - 1u) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) return;

    // ---- 3. last CTA: penalized C1 combine -> K1 (L), the bound, the pass-B list ----
    __threadfence();
    const unsigned flag = pen[0];
    const float rep = __uint_as_float(pen[1]), pres = __uint_as_float(pen[2]), freq = __uint_as_float(pen[3]);
    const int PD = (int)pen[5];
    const float scale = PD > 0 ? (float)PD : 1.0f;
    const int b0 = meta[0], p0 = meta[1], slot = meta[2];
    const int rmask = (1 << rlog2) - 1;
    unsigned* fm = reinterpret_cast<unsigned*>(s_ring4);  // [ns][128] window counts (ring is dead now)
    unsigned* fs = fm + ns * 128;
    for (int e = tid; e < ns * 128; e += 256) { fm[e] = 0u; fs[e] = 0u; }
    for (int bq = tid; bq < nblk; bq += 256) s_map[bq] = 0xFFu;
    if (tid < 8) pws[DHP_PEN + tid] = pen[tid];
    if (tid < 4) pws[DHP_META + tid] = tid < 3 ? (unsigned)meta[tid] : 0u;
    __syncthreads();
    if (tid < ns) s_map[s_sel.cand[tid]] = (unsigned char)tid;
    __syncthreads();
    if (flag != 0u) {
        const int* H = hist + (long long)slot * (rmask + 1);
        dhp_scan(pen, H, rmask, p0, b0, d_dev, p0 + 1 + pass_i, N,
                 [&](int t) { const int r = s_map[t >> 7]; return r == 0xFF ? -1 : ((r << 7) | (t & 127)); },
                 fm, fs);
        __syncthreads();
    }
    unsigned long long best = 0ull;
    for (int rr = warp; rr < ns; rr += 8) {
        const unsigned long long k = dhp_combine(part + (size_t)rr * S * 128, S, s_sel.cand[rr], svh,
                                                 fm + rr * 128, fs + rr * 128, flag, rep, pres, freq, scale, lane);
        best = k > best ? k : best;
    }
    if (lane == 0) s_best[warp] = best;
    __syncthreads();
    unsigned long long k1 = 0ull;
    #pragma unroll
    for (int w = 0; w < 8; ++w) k1 = s_best[w] > k1 ? s_best[w] : k1;
    const float L = dh_unord((unsigned)(k1 >> 32));   // the best penalized C1 logit: a lower bound
    const bool nonmono = ((flag & 1u) && !(rep >= 1.0f)) || ((flag & 2u) && !(pres >= 0.0f && freq >= 0.0f));
    const bool degen = !(L > -65504.0f);              // NaN, -inf, or the -65504 clamp floor
    unsigned mq = 0u, mall = 0u;
    int cq = 0;
    if (flag != 0u) {
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const int b = tid + 256 * i;
            if (b < nblk && s_map[b] == 0xFFu) {
                mall |= 1u << i;
                const float ub = bmax[b] + fmaxf(delta * sig * bsig[b], 0.f);
                if (!(ub < L)) { mq |= 1u << i; ++cq; }   // UB >= L (a NaN bound qualifies)
            }
        }
    }
    cq = __reduce_add_sync(0xffffffffu, cq);
    if (lane == 0) s_red[warp] = cq;
    __syncthreads();
    int nqual = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nqual += s_red[w];
    const bool over = nqual > cap;
    const bool fb = flag != 0u && (nonmono || degen || over);
    int n2 = 0;
    if (flag != 0u)                                   // uniform
        n2 = dh_rank1(fb ? mall : mq, s_sel.wh, s_sel.ph, &s_sel.tot[0], pws + DHP_HDR + dhp_nblk4(nblk));
    if (tid == 0) {
        unsigned fl = 0u;
        if (flag == 0u) fl = DHP_F_UNPEN;
        else fl = (nonmono ? DHP_F_NONMONO : 0u) | (degen ? DHP_F_DEGEN : 0u) | (over ? DHP_F_OVER : 0u)
                | (fb ? DHP_F_FB : 0u);
        *reinterpret_cast<unsigned long long*>(pws + DHP_BEST) = k1;
        pws[DHP_N2] = (unsigned)n2;
        unsigned* dg = pws + DHP_DBG;
        dg[0] = (unsigned)ns; dg[1] = (unsigned)n2; dg[2] = fl; dg[3] = __float_as_uint(L);
        dg[4] = (unsigned)nq; dg[5] = (unsigned)nqual; dg[6] = (unsigned)k1; dg[7] = (unsigned)(k1 >> 32);
        unsigned* st = pws + DHP_STAT;
        st[0] += 1u;
        if (flag == 0u) st[7] += 1u;
        else if (fb) {
            if (nonmono) st[4] += 1u; else if (degen) st[5] += 1u; else st[3] += 1u;
            st[6] += (unsigned)n2;
        } else if (n2 > 0) { st[1] += 1u; st[2] += (unsigned)n2; }
        cnt[0] = 0u;                                  // re-arm (shared with xq_dh_rescore: stream-ordered)
    }
}

// Pass B: grid (S, G), block 256, dynamic smem = R * 8 * 32 * hbits bytes; same geomA / geomB.
// CTA (sp, y) computes split sp of candidates y, y + G, ... < n2; the CTA completing a candidate's
// S splits combines + penalizes it and atomicMax-folds its key; the CTA completing the n2-th
// candidate writes the outputs. n2 == 0: CTA (0, 0) writes the outputs from pass A's key.
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore_pb(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        unsigned* __restrict__ pws, int* __restrict__ d_dev, const int* __restrict__ hist,
        int* __restrict__ out_id, float* __restrict__ dconf, int* __restrict__ toks,
        long long geomA, long long geomB) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ unsigned s_fm[128];
    __shared__ unsigned s_fs[128];
    __shared__ int s_last;
    const int pass_i = (int)(geomA & 0xFF), bits = (int)((geomA >> 8) & 0xFF);
    const int S = (int)((geomA >> 16) & 0xFF);
    const int K = (int)((geomA >> 32) & 0xFFFF), rlog2 = (int)((geomA >> 48) & 0x1F);
    const int N = (int)(geomB & 0xFFFFFFFFll);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7;
    const int n2 = (int)pws[DHP_N2];
    if (n2 == 0) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) dhp_finish(pws, pass_i, out_id, dconf, d_dev, toks);
        return;
    }
    if ((int)blockIdx.y >= n2) return;
    unsigned* cntc = pws + DHP_HDR;
    const unsigned* c2 = pws + DHP_HDR + dhp_nblk4(nblk);
    float* part = reinterpret_cast<float*>(pws + DHP_HDR + 2 * dhp_nblk4(nblk));
    const unsigned* P = pws + DHP_PEN;
    const unsigned flag = P[0];
    const float rep = __uint_as_float(P[1]), pres = __uint_as_float(P[2]), freq = __uint_as_float(P[3]);
    const int PD = (int)P[5];
    const float scale = PD > 0 ? (float)PD : 1.0f;
    const int b0 = (int)pws[DHP_META], p0 = (int)pws[DHP_META + 1], slot = (int)pws[DHP_META + 2];
    const int rmask = (1 << rlog2) - 1;
    const int* H = hist + (long long)slot * (rmask + 1);
    const int R = (K >> 4) / S, sp = blockIdx.x, kb0 = sp * R;
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    for (int rc = blockIdx.y; rc < n2; rc += gridDim.y) {
        const int cb = (int)c2[rc];
        __syncthreads();                              // the previous candidate's smem readers are done
        float acc[2][4];
        dh_part(tr, cb, kb0, R, N >> 4, bits, s_ring4, s_x2, acc);
        dh_part_store(part + (size_t)(rc * S + sp) * 128, acc);
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            s_last = (atomicAdd(cntc + rc, 1u) == (unsigned)S - 1u) ? 1 : 0;
        }
        __syncthreads();
        if (!s_last) continue;                        // uniform per CTA
        // this CTA completed candidate rc: its window counts, combine, fold
        __threadfence();
        if (tid < 128) { s_fm[tid] = 0u; s_fs[tid] = 0u; }
        __syncthreads();
        if (flag != 0u) {
            dhp_scan(P, H, rmask, p0, b0, d_dev, p0 + 1 + pass_i, N,
                     [&](int t) { return (t >> 7) == cb ? (t & 127) : -1; }, s_fm, s_fs);
            __syncthreads();
        }
        if (warp == 0) {
            const unsigned long long k = dhp_combine(part + (size_t)rc * S * 128, S, cb, svh, s_fm, s_fs,
                                                     flag, rep, pres, freq, scale, lane);
            if (lane == 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(pws + DHP_BEST), k);
                cntc[rc] = 0u;                        // re-arm this candidate's split counter
                __threadfence();
                if (atomicAdd(pws + DHP_DONE, 1u) == (unsigned)n2 - 1u) {
                    __threadfence();
                    dhp_finish(pws, pass_i, out_id, dconf, d_dev, toks);
                    pws[DHP_DONE] = 0u;               // re-arm
                }
            }
        }
    }
}

// ---- embed gather + hc-stream expansion: resid[b][s*h+i] = f32(embed[tok[b]][i]).
extern "C" __global__ void xq_embed_resid(float* __restrict__ resid, const __half* __restrict__ emb,
                                          const int* __restrict__ toks, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (i >= rw * B) return;
    int b = (int)(i / rw);
    int r = (int)(i % rw);
    resid[i] = xq_h2f(emb[(long long)toks[b] * h + (r % h)]);
}

// ---- S-A3-f Item 1: RoPE table gather. cos/sin are precomputed ON HOST at load
// (bitwise-identical to the old per-step powf/cos/sin loop) into tab[max_pos][stride];
// each step copies the m active lanes' rows into the persistent per-step buffer.
// A row-copy of host-computed bits — numerics identical by construction.
extern "C" __global__ void xq_cos_gather(const float* __restrict__ tab, float* __restrict__ dst,
                                         const int* __restrict__ pos, int stride, int M) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * stride;
    if (i >= total) return;
    int b = (int)(i / stride);
    dst[i] = tab[(long long)pos[b] * stride + (int)(i - (long long)b * stride)];
}

// ---- VIS-2: RoPE row gather through a per-slot mrope map (every main-attention gather site).
// Row b is KV position p = pos[b] of slot s (slot_src ? slot_src[b * slot_stride] : slot_const).
// Inside the slot's mapped prompt (p < rplen[s]) the source row is rmap[s * max_pos + p]; past it,
// p + rdelta[s] (text after the last image runs rdelta <= 0 below its KV index, HF get_rope_index).
// Source rows >= max_pos live in the per-slot image-row table mtab (rows max_pos + s * R + j).
// A slot with rplen = rdelta = 0 — every text-only request — reads tab[p]: bit-identical to
// xq_cos_gather. An out-of-range slot (a stale id) also falls back to tab[p].
extern "C" __global__ void xq_cos_gather_v(const float* __restrict__ tab, const float* __restrict__ mtab,
                                           float* __restrict__ dst, const int* __restrict__ pos,
                                           const int* __restrict__ slot_src, int slot_stride, int slot_const,
                                           const int* __restrict__ rmap, const int* __restrict__ rplen,
                                           const int* __restrict__ rdelta, int nslots, int max_pos,
                                           int stride, int M) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * stride;
    if (i >= total) return;
    int b = (int)(i / stride);
    int p = pos[b];
    int s = slot_src ? slot_src[(long long)b * slot_stride] : slot_const;
    long long idx = p;
    if (s >= 0 && s < nslots) {
        idx = p < rplen[s] ? (long long)rmap[(long long)s * max_pos + p] : (long long)p + rdelta[s];
    }
    const float* src = idx < max_pos ? tab + idx * stride : mtab + (idx - max_pos) * stride;
    dst[i] = src[(int)(i - (long long)b * stride)];
}

// ---- VIS-2: image-embedding splice into the embedded residual of a prefill chunk.
// resid [c][hc][h] f32 (xq_embed_resid's layout); src_row[b] >= 0 = the row of img [n][h] for chunk
// row b (written into every hyper-connection stream: the reference splices BEFORE the hc repeat);
// src_row[b] < 0 leaves the token embedding.
extern "C" __global__ void xq_splice_rows(float* __restrict__ resid, const float* __restrict__ img,
                                          const int* __restrict__ src_row, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (i >= rw * B) return;
    int b = (int)(i / rw);
    int s = src_row[b];
    if (s < 0) return;
    resid[i] = img[(long long)s * h + (int)((i % rw) % h)];
}

// ---- per-expert input Hadamard: xh[e,m,:] = H(x[m,:] * suh_all[idxmap[e],:]).
// grid E*M*(K/128), block 32; suh PRE-butterfly (their convention, S-A3-c).
// exl3_had128 normalizes internally — no extra scale here.
extern "C" __global__ void xq_had_suh_multi(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                            const int* __restrict__ idxmap,
                                            __half* __restrict__ xh, int M, int K,
                                            const int* __restrict__ esel,
                                            int per_expert_in) {
    XQ_PDL_ENTRY();
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
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(x[base_in + r]) * xq_h2f(sr[r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = xq_f2h(v[r]);
}

// ---- S-A3-z lever 2: FUSED routing (decode / verify, M <= 8). Bit-identical to the path it
// replaces: xq_router_ks (ks = 64 K-slices) + xq_router_combine, then xq_router_topk + xq_moe_route.
// (a) xq_router_fused: block = 4 experts x 64 slices; thread (e, s) reads its 40-wide weight slice
//     ONCE and accumulates every row with xq_router_ks's exact per-(row, expert, slice) order
//     (float4 groups of 8, ascending); the 64 partials of (row, expert) are then summed ascending
//     s = 0..63 by one thread — xq_router_combine's ((p0 + p1) + p2) + ... order. Old kernel re-read
//     each weight slice once per row from uncoalesced n*K strides.
#define XQ_RF_KS 64
#define XQ_RF_EPB 4          // experts per block (256 / XQ_RF_KS)
#define XQ_RF_MAXM 8
extern "C" __global__ void __launch_bounds__(256)
xq_router_fused(float* __restrict__ out, const __half* __restrict__ w, const __half* __restrict__ x,
                int M, int N, int K) {
    XQ_PDL_ENTRY();
    __shared__ float part[XQ_RF_MAXM][XQ_RF_EPB][XQ_RF_KS];
    const int tid = threadIdx.x;
    const int el = tid / XQ_RF_KS, s = tid % XQ_RF_KS;
    const int n = blockIdx.x * XQ_RF_EPB + el;
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    float acc[XQ_RF_MAXM];
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) acc[r] = 0.0f;
    if (n < N && slice == 40) {
        // K = 2560 (this model): 5 float4 groups, fully unrolled — all weight loads issue up front.
        float4 wv5[5];
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (n < N) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int k = (s8 << 3); k < slice; ++k) {
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[k]) * __half2float(x[(size_t)r * K + k0 + k]);
        }
    }
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) if (r < M) part[r][el][s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_RF_EPB) {
        const int r = tid / XQ_RF_EPB, e = tid % XQ_RF_EPB;
        const int nn = blockIdx.x * XQ_RF_EPB + e;
        if (nn < N) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r][e][q];
            out[(size_t)r * N + nn] = a;
        }
    }
}

// (b) xq_router_topk_route: grid (B), block 256 (== xq_router_topk's launch). Softmax is
//     xq_router_topk's code verbatim (same reductions => same prob bits). Selection: exact top-k
//     by (prob desc, id asc) — per-warp top-k by shuffle argmax over the warp's strided entries,
//     then warp 0 merges the 8 x k candidates by the same order. The top-k SET and ORDER of a
//     total order are unique, so ids/wts equal xq_router_topk's; renorm = its ascending-j sum.
//     The LAST block to finish (counter, reset by that block) runs xq_moe_route's scan verbatim.
#define XQ_RTR_MAXK 16
extern "C" __global__ void __launch_bounds__(256)
xq_router_topk_route(int* __restrict__ ids, float* __restrict__ wts, const float* __restrict__ logits,
                     int ne, int k, int B, int* __restrict__ done_cnt,
                     int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                     unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                     unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    extern __shared__ float sm[];
    const int b = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, nw = bs >> 5;
    float* prob = sm + bs;
    float* cv = sm + bs + ne;                         // [nw * XQ_RTR_MAXK] candidate values
    int* ci = (int*)(cv + nw * XQ_RTR_MAXK);          // [nw * XQ_RTR_MAXK] candidate ids
    __shared__ int last;
    if (b < B) {
        const float* lg = logits + (long long)b * ne;
        // ---- softmax: xq_router_topk verbatim ----
        float mx = -INFINITY;
        for (int i = tid; i < ne; i += bs) mx = fmaxf(mx, lg[i]);
        sm[tid] = mx; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] = fmaxf(sm[tid], sm[tid + s2]); __syncthreads(); }
        float gmax = sm[0];
        __syncthreads();
        float part = 0.0f;
        for (int i = tid; i < ne; i += bs) part += __expf(lg[i] - gmax);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        float denom = sm[0];
        for (int i = tid; i < ne; i += bs) prob[i] = __expf(lg[i] - gmax) / denom;
        __syncthreads();
        // ---- per-warp top-k over this warp's entries {i : (i / 32) % nw == warp} ----
        for (int j = 0; j < k; j++) {
            float bv = -2.0f; int bi = ne;
            for (int i = warp * 32 + lane; i < ne; i += bs)
                if (prob[i] > bv || (prob[i] == bv && i < bi)) { bv = prob[i]; bi = i; }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                cv[warp * XQ_RTR_MAXK + j] = bv; ci[warp * XQ_RTR_MAXK + j] = bi;
                if (bi < ne) prob[bi] = -3.0f;        // below the -2 sentinel: never re-picked
            }
            __syncwarp();
        }
        __syncthreads();
        // ---- merge (warp 0): k picks over nw*k candidates, same total order ----
        if (warp == 0) {
            const int nc = nw * k;
            for (int j = 0; j < k; j++) {
                float bv = -2.0f; int bi = ne, bslot = -1;
                for (int c = lane; c < nc; c += 32) {
                    const int wq = c / k, jj = c % k;
                    const float v = cv[wq * XQ_RTR_MAXK + jj]; const int id = ci[wq * XQ_RTR_MAXK + jj];
                    if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; bslot = wq * XQ_RTR_MAXK + jj; }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                    const int os = __shfl_xor_sync(0xffffffffu, bslot, o);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bslot = os; }
                }
                if (lane == 0) {
                    int id = bi;
                    if (id < 0 || id >= ne) id = 0;   // xq_router_topk's NaN/Inf clamp
                    ids[(long long)b * k + j] = id;
                    wts[(long long)b * k + j] = bv;
                    if (bslot >= 0) cv[bslot] = -4.0f;
                }
                __syncwarp();
            }
            if (lane == 0) {
                float sum = 0.0f;
                for (int j = 0; j < k; j++) sum += wts[(long long)b * k + j];
                for (int j = 0; j < k; j++) wts[(long long)b * k + j] /= sum;
            }
        }
    }
    // ---- last block: xq_moe_route verbatim ----
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the scan runs in SHARED memory: a serial thread-0 walk over global (volatile) ids + a global
    // slotmap cost ~0.4 us of L2 latency per step (24 us/launch measured) — same first-seen order.
    __shared__ int cnt;
    __shared__ int sids[XQ_RTR_MAXK * XQ_RF_MAXM];
    __shared__ int sidx[XQ_RTR_MAXK * XQ_RF_MAXM];
    int* sslot = (int*)sm;                           // ne ints (the softmax scratch is dead here)
    for (int t = tid; t < B * k; t += bs) sids[t] = __ldcg(ids + t);
    for (int e = tid; e < ne; e += bs) sslot[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < B * k; t++) {
            const int e = sids[t];
            if (e >= 0 && e < ne && sslot[e] == -1) { sslot[e] = c; sidx[c] = e; c++; }
        }
        cnt = c; esel_dev[0] = c;
        *done_cnt = 0;                               // re-arm for the next launch / graph replay
    }
    __syncthreads();
    for (int e = tid; e < ne; e += bs) slotmap[e] = sslot[e];
    for (int s = tid; s < cnt; s += bs) {
        const int e = sidx[s];
        idxmap[s] = e;
        offs_gu[s] = (unsigned long long)e * gu_words;
        offs_d[s] = (unsigned long long)e * d_words;
    }
}

// ---- WP10 (PLAN/SURPASS_PLAN_2026-09-26): ROUTING FOLD. xq_router_fused + xq_router_topk_route in
// ONE launch: grid (ne/4), block 256 — the GEMV grid of xq_router_fused; the LAST block to finish
// (self re-arming counter; publication = cooperative-groups grid-sync pattern: bar.sync -> tid0
// fence + atomic; winner: fence -> bar.sync -> __ldcg) runs softmax / top-k / renorm / first-seen
// route for all M <= 8 rows, warp r <-> row r. Bit-identity contract vs the two-kernel path (itself
// == xq_router_ks + combine + xq_router_topk + xq_moe_route):
//  logits  — every (row, expert) value is the SAME thread-local chain (40-wide slice, float4 groups
//            of 8, ascending k, acc += w*x) + the SAME one-thread ascending s = 0..63 partial sum.
//            REMAP only changes WHICH thread owns (expert, slice) and pads the smem partial layout.
//  softmax — lane l holds the virtual xq_router_topk threads tid = l + 32q (q < 8), elements
//            i = tid + 256p (ascending p, the same per-thread folds). The 256-thread halving tree
//            is replayed exactly: q += q+4 / q+2 / q+1 in registers (s2 = 128, 64, 32), then
//            shfl_down 16..1 (s2 = 16..1): same ops, same order, same (low, high) operand order =>
//            same gmax / denom bits; prob = the same __expf(lg - gmax) / denom.
//  top-k   — (prob desc, id asc) is a total order on the non-NaN probs, so the top-k sequence is
//            unique: k warp-wide shuffle argmaxes == topk_route's per-warp + merge. An all-NaN row
//            picks (-2, ne) -> id 0, w -2 at every j, exactly as topk_route.
//  renorm  — sum = ((0 + w0) + w1) + ... ascending j (topk_route's lane-0 loop), w_j / sum.
//  route   — first-seen over ids row-major: first(e) = min position (smem atomicMin: a min is
//            order-free), slot(e) = #{first positions < first(e)} (ballot / popc prefix) =>
//            identical slotmap / idxmap / esel / offs_gu / offs_d (written for s < esel only).
// REMAP (xq_router_fold): warp w owns slices 8w..8w+7 of all 4 experts (lane = el*8 + s%8), so a
// warp's x float4 loads touch 8 distinct 16-B chunks in 5 lines (was 32 in 20: the m-scaling
// L1 wavefront term), weights unchanged (4 x 640 B = 20 lines); partials padded (el stride 72,
// row stride 289) so both the partial stores and the 64-deep ascending sums are bank-conflict-free
// (the legacy [8][4][64] layout put all M*4 summing lanes on one bank per step).
// xq_router_fold_m0 = xq_router_fused's exact lane map + layout (isolates the fold; GB10_WP10_GEMV=0).
#define XQ_RFD_NEMAX 512                               // the tail holds ne/32 <= 16 probs per lane
#define XQ_RFD_PS (XQ_RF_KS + 8)                       // REMAP partial el stride (72 == 8 mod 32)
#define XQ_RFD_RS (XQ_RF_EPB * XQ_RFD_PS + 1)          // REMAP partial row stride (289 == 1 mod 32)
// A5-K2 COAL (xq_router_fold_c; ledger #4): the 80-B-per-thread weight read (5 LDG.128 at an 80-B
// lane stride: 32 sector requests per warp instruction for 16 sectors of data) streams the 2.62 MB
// router at ~146 GB/s (17.9 us standalone) where a coalesced read of the same bytes takes 11.9 us
// (221 GB/s) whatever the grid (48..640 CTAs: the CTA count is NOT the limit). COAL loads each warp's
// 4 x 640-B expert chunks with coalesced cp.async.cg (lane-contiguous 16-B pieces) into a
// warp-private smem slab (dynamic, 8 x 2560 B), __syncwarp, then every thread reads ITS five
// float4 back (LDS.128, 80-B lane stride = conflict-free per quarter-warp) and runs the unchanged
// chain; per q it issues the M x loads before the FMAs (register-only reordering of loads). Bits:
// the same (w, x) values reach the same FFMA sequence per (row, expert, slice) — identical.
// TP-G (T2 lever #5) EP: the route tail runs xq_moe_route_ep's semantics instead — first-seen over
// ids row-major restricted to the experts THIS rank holds (ep_lut[e] >= 0), idxmap / offs_* in LOCAL
// stacked indices (ep_lut[e]), slotmap[e] = -1 for a remote expert. The slot of a local first position
// = #local first positions before it (ballot/popc prefix over the local-first flags) == route_ep's
// serial counter: identical tables, one launch fewer per MoE layer. EP = false compiles it out.
template <bool REMAP, bool COAL = false, bool EP = false>
__device__ __forceinline__ void xq_router_fold_body(
        float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
        int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
        int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
        unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
        unsigned long long gu_words, unsigned long long d_words, const int* __restrict__ ep_lut = nullptr) {
    constexpr int PS = REMAP ? XQ_RFD_PS : XQ_RF_KS;
    constexpr int RS = REMAP ? XQ_RFD_RS : XQ_RF_EPB * XQ_RF_KS;
    static_assert(XQ_RF_MAXM * RS >= XQ_RFD_NEMAX + 2 * XQ_RF_MAXM * XQ_RTR_MAXK + 8, "tail smem alias");
    __shared__ float part[XQ_RF_MAXM * RS];
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int el = REMAP ? (lane >> 3) : (tid / XQ_RF_KS);
    const int s = REMAP ? ((warp << 3) | (lane & 7)) : (tid % XQ_RF_KS);
    const int n = blockIdx.x * XQ_RF_EPB + el;
    // ---- GEMV: xq_router_fused's per-thread body, verbatim ----
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    float acc[XQ_RF_MAXM];
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) acc[r] = 0.0f;
    if constexpr (COAL) {
        static_assert(REMAP, "COAL uses the REMAP lane map (warp = 4 experts x 8 slices)");
        extern __shared__ __align__(16) float4 xq_rfc_ws[];          // [8 warps][4 experts][40]
        if (slice == 40) {                                           // host: always (K == 2560)
            float4* ws = xq_rfc_ws + warp * (XQ_RF_EPB * 40);
            #pragma unroll
            for (int q = 0; q < 5; ++q) {
                const int f = (q << 5) | lane, ef = f / 40, c = f - ef * 40;
                const int nf = blockIdx.x * XQ_RF_EPB + ef;
                if (nf < N)
                    lmh_cp16((unsigned)__cvta_generic_to_shared(ws + f),
                             w + (size_t)nf * K + warp * (8 * 40) + (c << 3));
            }
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;\n" ::: "memory");
            __syncwarp();
            if (n < N) {
                float4 wv5[5];
                #pragma unroll
                for (int q = 0; q < 5; ++q) wv5[q] = ws[el * 40 + (lane & 7) * 5 + q];
                #pragma unroll
                for (int q = 0; q < 5; ++q) {
                    const __half* wh = (const __half*)&wv5[q];
                    float4 xv[XQ_RF_MAXM];
                    #pragma unroll
                    for (int r = 0; r < XQ_RF_MAXM; r++)
                        if (r < M) xv[r] = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    #pragma unroll
                    for (int r = 0; r < XQ_RF_MAXM; r++) {
                        if (r < M) {
                            const __half* xh_ = (const __half*)&xv[r];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                        }
                    }
                }
            }
        }
    } else {
    if (n < N && slice == 40) {
        float4 wv5[5];
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (n < N) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int kk = (s8 << 3); kk < slice; ++kk) {
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[kk]) * __half2float(x[(size_t)r * K + k0 + kk]);
        }
    }
    }   // !COAL
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) if (r < M) part[r * RS + el * PS + s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_RF_EPB) {
        const int r = tid / XQ_RF_EPB, e = tid % XQ_RF_EPB;
        const int nn = blockIdx.x * XQ_RF_EPB + e;
        if (nn < N) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r * RS + e * PS + q];
            out[(size_t)r * N + nn] = a;
        }
    }
    // ---- publish + elect the last block (every block's logits visible before its ticket) ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
        if (last) __threadfence();
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    // ---- tail (last block only). `part` is dead: reuse it for the route scratch ----
    int* s_first = reinterpret_cast<int*>(part);                 // [ne]  first position of expert e
    int* s_ids = s_first + XQ_RFD_NEMAX;                         // [M*k] picked ids, row-major
    int* s_slot = s_ids + XQ_RF_MAXM * XQ_RTR_MAXK;              // [M*k] slot of a first position
    int* s_wc = s_slot + XQ_RF_MAXM * XQ_RTR_MAXK;               // [8]   per-warp first counts
    for (int e = tid; e < N; e += blockDim.x) s_first[e] = 0x7fffffff;
    if (warp < M) {
        const int r = warp;
        const float* lg = out + (size_t)r * N;
        float v[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) { const int i = lane + 32 * t; v[t] = (i < N) ? __ldcg(lg + i) : 0.0f; }
        // max: virtual thread q folds i = l+32q, then l+32q+256 (ascending), then the 256-tree.
        float mq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            mq[q] = -INFINITY;
            if (lane + 32 * q < N) mq[q] = fmaxf(mq[q], v[q]);
            if (lane + 32 * q + 256 < N) mq[q] = fmaxf(mq[q], v[q + 8]);
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) mq[q] = fmaxf(mq[q], mq[q + 4]);           // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) mq[q] = fmaxf(mq[q], mq[q + 2]);           // s2 = 64
        float mx = fmaxf(mq[0], mq[1]);                                          // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
        const float gmax = __shfl_sync(0xffffffffu, mx, 0);
        // denominator: the same per-virtual-thread partials and the same halving tree.
        float ev[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) ev[t] = __expf(v[t] - gmax);
        float pq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            pq[q] = 0.0f;
            if (lane + 32 * q < N) pq[q] += ev[q];
            if (lane + 32 * q + 256 < N) pq[q] += ev[q + 8];
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) pq[q] += pq[q + 4];                          // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) pq[q] += pq[q + 2];                          // s2 = 64
        float sm_ = pq[0] + pq[1];                                               // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sm_ += __shfl_down_sync(0xffffffffu, sm_, o);
        const float denom = __shfl_sync(0xffffffffu, sm_, 0);
        float p[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) p[t] = (lane + 32 * t < N) ? ev[t] / denom : -4.0f;  // -4: never picked
        // top-k under (prob desc, id asc); lane j keeps pick j; every lane folds the same sum.
        float myw = 0.0f, sum = 0.0f;
        int myid = 0;
        for (int j = 0; j < k; ++j) {
            float bv = -2.0f; int bi = N;
            if constexpr (COAL) {
                // A5-K2: the same pick, shorter chains. (prob desc, id asc) is a total order on the
                // non-NaN probs and every candidate here is a prob in [0, 1] (ev <= 1 after the max
                // shift; NaN / picked -3 / padding -4 are never picked: they never beat the (-2, N)
                // seed). So: map non-candidates (!(p >= 0)) to -5, take the lane's best by a 4-level
                // tree (ids ascend with t, so a pair keeps its lower id unless the higher one is
                // strictly greater), then the warp's best as (max key, min id among max-key lanes)
                // with redux.sync, key = bits(p) + 1 (order-preserving for p >= 0; 0 = no
                // candidate). bv carries the exact prob bits back. == the linear scan + butterfly.
                float tv[16]; int ti[16];
                #pragma unroll
                for (int t = 0; t < 16; ++t) { tv[t] = (p[t] >= 0.0f) ? p[t] : -5.0f; ti[t] = lane + 32 * t; }
                #pragma unroll
                for (int st = 1; st < 16; st <<= 1) {
                    #pragma unroll
                    for (int t = 0; t < 16; t += 2 * st)
                        if (tv[t + st] > tv[t]) { tv[t] = tv[t + st]; ti[t] = ti[t + st]; }
                }
                const unsigned key = (tv[0] >= 0.0f) ? __float_as_uint(tv[0]) + 1u : 0u;
                const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
                const unsigned imin = __reduce_min_sync(0xffffffffu, (key == kmax) ? (unsigned)ti[0] : 0xffffffffu);
                if (kmax != 0u) { bv = __uint_as_float(kmax - 1u); bi = (int)imin; }
            } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int i = lane + 32 * t;
                if (p[t] > bv || (p[t] == bv && i < bi)) { bv = p[t]; bi = i; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            }   // !COAL
            #pragma unroll
            for (int t = 0; t < 16; ++t) if (lane + 32 * t == bi) p[t] = -3.0f;
            const int id = (bi < 0 || bi >= N) ? 0 : bi;   // topk_route's NaN/Inf clamp
            if (lane == j) { myw = bv; myid = id; }
            sum += bv;
        }
        if (lane < k) {
            ids[(size_t)r * k + lane] = myid;
            wts[(size_t)r * k + lane] = myw / sum;
            s_ids[r * k + lane] = myid;
        }
    }
    __syncthreads();
    // ---- first-seen route (xq_moe_route semantics) ----
    const int BK = M * k;                                        // host: <= 128 <= blockDim
    if (tid < BK) { const int e = s_ids[tid]; if (e >= 0 && e < N) atomicMin(&s_first[e], tid); }
    __syncthreads();
    int e_t = -1; bool first = false;
    if (tid < BK) { e_t = s_ids[tid]; first = (e_t >= 0 && e_t < N) && s_first[e_t] == tid; }
    int le_t = e_t;                                              // the stacked index idxmap/offs carry
    if constexpr (EP) {
        if (first) { le_t = __ldg(ep_lut + e_t); first = le_t >= 0; }   // a remote expert takes no slot
    }
    const unsigned bal = __ballot_sync(0xffffffffu, first);
    if (lane == 0) s_wc[warp] = __popc(bal);
    __syncthreads();
    int base = 0, total = 0;
    #pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) { const int c = s_wc[w8]; total += c; if (w8 < warp) base += c; }
    if (first) {
        const int slot = base + __popc(bal & ((1u << lane) - 1u));
        s_slot[tid] = slot;
        idxmap[slot] = le_t;
        offs_gu[slot] = (unsigned long long)le_t * gu_words;
        offs_d[slot] = (unsigned long long)le_t * d_words;
    }
    if (tid == 0) { esel_dev[0] = total; *done_cnt = 0; }   // re-arm for the next launch / graph replay
    __syncthreads();
    for (int e = tid; e < N; e += blockDim.x) {
        const int f = s_first[e];
        bool live = f != 0x7fffffff;
        if constexpr (EP) live = live && __ldg(ep_lut + e) >= 0;       // s_slot[f] is unwritten for a remote e
        slotmap[e] = live ? s_slot[f] : -1;
    }
}

