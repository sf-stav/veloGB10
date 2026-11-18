// exl3_bench/04_sampler_penalties.cuh: WP15 repetition/presence/frequency penalties, argmax, stochastic sampling (WP24 real-q).
// Penalties follow the reference ComboSampler semantics, applied in place to fp16 logit rows.
// Section 4 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- WP15: repetition / presence / frequency penalties with the REFERENCE implementation's semantics
// (exllamav3 ComboSampler: SS_RepP then SS_PresFreqP, exllamav3_ext/generator/rep_pen.cu), applied
// in place to fp16 logit rows BEFORE xq_argmax / xq_sample_rows, at every decode row and every
// verify row. Per-row params pen[row*8 + {0 flag (bit0 rep, bit1 pres/freq; 0 = the row is
// untouched), 1 rep_p (f32), 2 pres_p (f32), 3 freq_p (f32), 4 sustain, 5 decay, 6-7 0}].
// Window: past positions i with past_len - i < sustain + decay (the reference implementation's `i > past_len - S - D`),
// factor(dist) = 1 for dist <= S, else 1 - (dist - S) / D (clamped); the rep factor and the presence
// factor are the MAX over a token's occurrences (its most recent), the frequency term the SUM.
//   rep:  w = v > 0 ? v / rep_p : v * rep_p;  fr = f + 1e-30;  v = v * ((1 - fr) + 1e-30) + w * fr
//   pres/freq: v -= freq_p * sum(factor);  v -= f * pres_p
// Factors are exact integers in 1/D units (the reference implementation's float factor is k/D, exact for D = 1024),
// accumulated with integer atomics: deterministic, unlike the reference implementation's float atomicAdd; the frequency
// sum can differ from the reference implementation's by float rounding only. Only tokens seen in the window are
// rewritten — for every other token the reference implementation's formula returns v bit-exactly — so a penalized row
// costs O(window) work per 4096-id span, and an unpenalized row (flag 0) exits at once.
// Past tokens: a per-slot device ring `hist` of R = 1 << rlog2 ids indexed by position & (R-1)
// (positions < base: committed history) plus the in-flight row tokens (positions >= base).
#define XQ_PEN_SPAN 4096
// One token's penalized fp16 logit (xq_pen_span's per-token ops; DHEADP's rescore calls the SAME
// function, so a draft's penalized logit is the draft mirror's to the bit): v = the fp16 logit as
// f32, fm / fs[j] = the token's max / summed window factors in 1/D units (fm > 0), scale = D (or 1);
// fs[j] is read only when the pres/freq bit is set (xq_pen_span's load placement: its PTX is unchanged).
// MONOTONICITY (DHEADP's bound, w3 2026-09-27): with rep >= 1, pres >= 0, freq >= 0 the result is
// <= v for every v >= -65504 (rep: w <= v, and v*f1 + w*fr exceeds v by at most a few f32 ulps —
// far under half an fp16 ulp, so the fp16 RN never rounds above v; pres/freq subtract >= 0; the
// +65504 clamp only lowers). Only v = -inf rises (to -65504). rep < 1 or pres/freq < 0 break it.
__device__ __forceinline__ unsigned short xq_pen_tok(float v, unsigned int fm, const unsigned int* fs,
                                                     int j, unsigned int flag, float rep, float pres,
                                                     float freq, float scale) {
    const float f = (float)fm / scale;
    if (flag & 1u) {
        // literally the reference implementation's expression (rep_pen.cu apply_rep_pens_kernel)
        const float w = v > 0.0f ? v / rep : v * rep;
        const float fr = f + 1e-30f;
        const float f1 = (1.0f - fr) + 1e-30f;
        v = v * f1 + w * fr;
    }
    if (flag & 2u) {
        v -= freq * ((float)fs[j] / scale);
        v -= f * pres;
    }
    // keep the row finite in fp16 (an overflow to +inf would poison the sampler's softmax
    // shift); NaN stays NaN (neither comparison holds)
    if (v > 65504.0f) v = 65504.0f; else if (v < -65504.0f) v = -65504.0f;
    return __half_as_ushort(__float2half_rn(v));
}
__device__ __forceinline__ void xq_pen_span(unsigned short* __restrict__ L, int V, int span0,
                                            const unsigned int* __restrict__ P,
                                            const int* __restrict__ H, int rmask,
                                            int base, int first, const int* __restrict__ ext,
                                            int past_len, unsigned int* fmax, unsigned int* fsum) {
    const unsigned int flag = P[0];
    const float rep = __uint_as_float(P[1]);
    const float pres = __uint_as_float(P[2]);
    const float freq = __uint_as_float(P[3]);
    const int S = (int)P[4], D = (int)P[5];
    const int n = min(XQ_PEN_SPAN, V - span0);
    for (int j = threadIdx.x; j < XQ_PEN_SPAN; j += blockDim.x) { fmax[j] = 0u; fsum[j] = 0u; }
    __syncthreads();
    const int lo = max(0, past_len - S - D + 1);
    for (int i = lo + (int)threadIdx.x; i < past_len; i += blockDim.x) {
        const int t = i < base ? H[i & rmask] : (i == base ? first : ext[i - base - 1]);
        const int j = t - span0;
        if (j < 0 || j >= n) continue;
        const int dist = past_len - i;
        const unsigned int f = D > 0 ? (unsigned int)min(D, D + S - dist) : 1u;
        atomicMax(&fmax[j], f);
        atomicAdd(&fsum[j], f);
    }
    __syncthreads();
    const float scale = D > 0 ? (float)D : 1.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const unsigned int fm = fmax[j];
        if (fm == 0u) continue;
        L[span0 + j] = xq_pen_tok(__half2float(__ushort_as_half(L[span0 + j])), fm, fsum, j,
                                  flag, rep, pres, freq, scale);
    }
}

// Decode / verify rows. grid = (ceil(V / XQ_PEN_SPAN), m). Row r = (slot, pos) from the slot-pos
// table, input token toks[r] at pos. Rows of one slot at consecutive positions form a run (the
// verify: b, d0.. at p..p+k); row r's past = hist[..base) + toks[r0..=r] where r0 = the run's
// first row and base = its position — so verify row r sees exactly the drafts before it, i.e.
// the sequence a plain decode of the same prefix would see. Block x == 0 then commits toks[r]
// at pos into the ring: every reader of this launch reads positions < base <= pos, the ring holds
// >= window + verify width positions (host-checked), and a rejected draft's entry lies beyond the
// next round's base, so it is rewritten before any launch can read it.
extern "C" __global__ void __launch_bounds__(256) xq_pen_rows(
        unsigned short* __restrict__ logits, int V, const unsigned int* __restrict__ pen,
        const int* __restrict__ toks, const int* __restrict__ slots, int* __restrict__ hist, int rlog2) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.y;
    const unsigned int* P = pen + row * 8;
    if (P[0] == 0u) return;
    __shared__ unsigned int fmax[XQ_PEN_SPAN];
    __shared__ unsigned int fsum[XQ_PEN_SPAN];
    const int slot = slots[2 * row], pos = slots[2 * row + 1];
    int r0 = row;
    while (r0 > 0 && slots[2 * (r0 - 1)] == slot && slots[2 * (r0 - 1) + 1] == slots[2 * r0 + 1] - 1) r0--;
    const int base = slots[2 * r0 + 1];
    const int R = 1 << rlog2;
    int* H = hist + (long long)slot * R;
    xq_pen_span(logits + (long long)row * V, V, blockIdx.x * XQ_PEN_SPAN, P, H, R - 1,
                base, toks[r0], toks + r0 + 1, pos + 1, fmax, fsum);
    if (blockIdx.x == 0 && threadIdx.x == 0) H[pos & (R - 1)] = toks[row];
}

// Draft mirror (draft-only): the same adjustment on draft pass i's logits row (the pruned head's
// leading Vd ids). meta = {b, p, slot, 0} (sc.draft_meta); pass i predicts position p+1+i, so its
// past = hist[..p) + b (at p) + d_0..d_{i-1} (d_dev, the chain's own drafts). Params = row 0 (the
// lane). No commit: the verify commits the round's inputs.
extern "C" __global__ void __launch_bounds__(256) xq_pen_draft(
        unsigned short* __restrict__ logits, int Vd, const unsigned int* __restrict__ pen,
        const int* __restrict__ meta, const int* __restrict__ d_dev, int i,
        const int* __restrict__ hist, int rlog2) {
    XQ_PDL_ENTRY();
    if (pen[0] == 0u) return;
    __shared__ unsigned int fmax[XQ_PEN_SPAN];
    __shared__ unsigned int fsum[XQ_PEN_SPAN];
    const int b = meta[0], p = meta[1], slot = meta[2];
    const int R = 1 << rlog2;
    xq_pen_span(logits, Vd, blockIdx.x * XQ_PEN_SPAN, pen, hist + (long long)slot * R, R - 1,
                p, b, d_dev, p + 1 + i, fmax, fsum);
}

// W2-EXL3: JSON-schema constrained decoding. grid = (ceil(V / 2048), m), block 256: row r
// with flags[r] != 0 keeps only the ids whose bit is set in its allowed bitset
// words[r * nw .. (r + 1) * nw) (bit t of word t >> 5); every other logit becomes -inf (f16 0xFC00),
// which the argmax never picks and the sampler gives zero mass. Runs after xq_pen_rows and before
// xq_argmax_rows / xq_sample_rows; launched only in steps that stage a mask.
extern "C" __global__ void __launch_bounds__(256) xq_vmask_rows(
        unsigned short* __restrict__ logits, int V, const unsigned int* __restrict__ words, int nw,
        const int* __restrict__ flags) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.y;
    if (flags[row] == 0) return;
    const unsigned int* W = words + (long long)row * nw;
    unsigned short* L = logits + (long long)row * V;
    const int base = blockIdx.x * 2048;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int t = base + k * 256 + (int)threadIdx.x;
        if (t < V && ((W[t >> 5] >> (t & 31)) & 1u) == 0u) L[t] = (unsigned short)0xFC00u;
    }
}

// =====================================================================
// WP24: exact speculative sampling with STOCHASTIC drafts (real-q). Opt-in
// (--spec-sampling ratio); nothing in this section runs on the default match path.
//
// Draft side (xq_rq_draft, one launch per draft pass, replaces that pass's argmax): the top-32
// of the pass's pruned-head logits (key desc, id asc — list[0] is xq_argmax_rows' id), then
// (mode 2) the proposal q' = that list truncated to the request's top-k, tempered by tau_d,
// cut to the smallest top-p prefix, renormalised: EXACTLY the distribution the pass samples
// its draft from (uniform = hash(seed, meta[3] + i, XQ_RQ_DOM_DRAFT)). (ids, logits, q') are
// stored per pass (one XQ_RQ_ROW-word row). Mode 1 = the list only (GB10_WP24_DUMP rung 0;
// the draft stays xq_argmax_rows' id).
// Verify side (xq_sample_rows_rq): the row's target p is xq_sample_rows' distribution
// (T -> top-k -> min-p -> top-p over the penalised logits, same code). Row i < nratio accepts
// d_i with probability min(1, p(d_i)/q'(d_i)); on reject it samples norm(max(0, p - q'))
// (only the listed ids are adjusted; d_i itself is 0) and the bonus row samples p. xq_accept
// runs unchanged (a = the first row whose emitted id != d_i), and
//   P(emit x) = min(p, q')(x) + (1 - sum min(p, q')) * max(0, p - q')(x) / sum max(0, p - q') = p(x)
// — distribution-exact against the plain sampler (speculative sampling, Leviathan / Chen).
// RNG domains: draft draw, accept test and residual draw are separate streams; the bonus /
// plain rows use domain 0 = xq_sample_rows' own draw (same counter -> the same id).
// w5 port (onto the DHEAD line): a real-q (mode 2) draft pass runs the FULL pruned-slice head
// (exl3_chain -> [xq_pen_draft] -> this kernel -> xq_rowmax_f16), not DHEAD's screen + rescore:
// DHEAD's candidate blocks are chosen to contain the ARGMAX (screen bound vs the best block), so
// they cannot guarantee the exact top-K. A5-L4 (spec.ratio_dhead=1, opt-in): unpenalized real-q
// passes run DHEAD + xq_rq_draft_dh instead — q' over the candidate set (exact for any q'). The dump (mode 1) keeps the served DHEAD/DHEADP draft
// and recomputes the slice logits only for its list. dconf (the DDS confidence) stays the pass's
// max draft logit (xq_rowmax_f16), never a function of the sampled draft.
// Known epsilon (float rounding only): a rejected ratio row whose residual mass R is 0 (p == q'
// to rounding) emits d, and a row with no target survivor keeps xq_argmax's id; xq_accept counts
// either as an accept. Both events have probability ~0.
// =====================================================================
#define XQ_RQ_K 32           // candidate list width (one warp)
#define XQ_RQ_NB_MAX 32      // stage-1 blocks per row
#define XQ_RQ_SLICE 8192     // max stage-1 slice (fp16 in smem, 16 KB)
#define XQ_RQ_ROW 96         // u32 words per list row: ids[32] | logits f32[32] | q' f32[32]
#define XQ_RQ_DOM_ACC 1u
#define XQ_RQ_DOM_RES 2u
#define XQ_RQ_DOM_DRAFT 3u
// uniform in (0,1) from (seed, counter, domain); dom 0 is bit-for-bit xq_sample_rows' draw.
__device__ __forceinline__ float xq_rq_u24(unsigned long long seed, unsigned int ctr, unsigned int dom) {
    return ((float)(xq_mix64(seed ^ xq_mix64(((unsigned long long)dom << 32) + (unsigned long long)ctr + 1ull)) >> 40)
            + 0.5f) * (1.0f / 16777216.0f);
}
// order key: larger = higher logit, ties -> lower id (xq_argmax_rows' comparator: -0 == +0);
// NaN -> 0 = "no candidate" (xq_argmax_rows never picks a NaN either).
__device__ __forceinline__ unsigned long long xq_rq_key(unsigned short b, int i) {
    if ((b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u) return 0ull;
    if (b == 0x8000u) b = 0u;
    return ((unsigned long long)xq_skey(b) << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned)i);
}

// Warp 0 of a row's final top-32 (`top`: sorted desc, invalid = 0 trailing; lane j = candidate j)
// -> (mode 2) the proposal q' and the draft sampled from it (-> out[0], returned; counter
// meta[3] + ci), and the list row L = {ids | logits | q'}. A copy of xq_rq_draft's warp-0 tail, op
// for op (xq_rq_draft keeps its inline copy so its PTX stays byte-identical); used by xq_rq_draft_dh.
// Returns -1 when mode != 2.
__device__ __forceinline__ int xq_rq_emit(const unsigned long long* top, int lane, int mode,
        const unsigned int* __restrict__ dsamp, const int* __restrict__ meta, int ci,
        int* __restrict__ out, unsigned int* __restrict__ L) {
    int pick = -1;
    const unsigned long long kj = top[lane];
    const bool valid = kj != 0ull;
    int id = valid ? (int)(0xFFFFFFFFu - (unsigned)(kj & 0xFFFFFFFFull)) : -1;
    const float lg = valid ? xq_skey_val((unsigned)(kj >> 32)) : -INFINITY;
    float q = 0.0f;
    if (mode == 2) {
        const float tau = fmaxf(__uint_as_float(dsamp[1]), 1e-6f), invT = 1.0f / tau;
        const float top_p = __uint_as_float(dsamp[2]);
        const unsigned int top_k = dsamp[3];
        const unsigned long long seed = ((unsigned long long)dsamp[5] << 32) | (unsigned long long)dsamp[4];
        const unsigned int ctr = (unsigned int)meta[3] + (unsigned int)ci;
        const bool any = __shfl_sync(0xffffffffu, (int)valid, 0) != 0;
        pick = 0;
        if (!any) {
            // no finite-key candidate at all: the delta proposal q' = 1{0} (still exact — the
            // verify then runs the match rule's law against token 0)
            id = lane == 0 ? 0 : -1;
            q = lane == 0 ? 1.0f : 0.0f;
        } else {
            const float l0 = __shfl_sync(0xffffffffu, lg, 0);
            const bool keep = valid && (top_k == 0u || (unsigned)lane < top_k);
            float w = keep ? __expf((lg - l0) * invT) : 0.0f;
            float tot = w;
            for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (top_p < 1.0f && tot > 0.0f) {
                // smallest prefix whose mass reaches top_p * tot: keep j iff mass(0..j-1) < target
                float incl = w;
                for (int o = 1; o < 32; o <<= 1) {
                    const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += nn;
                }
                float excl = __shfl_up_sync(0xffffffffu, incl, 1);
                if (lane == 0) excl = 0.0f;
                if (!(excl < top_p * tot)) w = 0.0f;
            }
            float z = w;
            for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
            if (!(z > 0.0f && z < INFINITY)) {               // non-finite list: delta on candidate 0
                w = lane == 0 ? 1.0f : 0.0f;
                z = 1.0f;
            }
            q = w / z;
            // inverse CDF over the kept prefix in list order (warp-uniform, deterministic)
            const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_DRAFT) * z;
            float incl = w;
            for (int o = 1; o < 32; o <<= 1) {
                const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += nn;
            }
            const unsigned int hit = __ballot_sync(0xffffffffu, w > 0.0f && incl > target);
            const unsigned int pos = __ballot_sync(0xffffffffu, w > 0.0f);
            const int pl = hit ? __ffs(hit) - 1 : 31 - __clz(pos);
            pick = __shfl_sync(0xffffffffu, id, pl);
        }
        if (lane == 0) out[0] = pick;
    }
    L[lane] = (unsigned int)id;
    L[32 + lane] = __float_as_uint(lg);
    L[64 + lane] = __float_as_uint(q);
    return pick;
}

// grid (nb <= XQ_RQ_NB_MAX, rows), block 256. Row r = logits row r (stride V); its list goes to
// list + (i0 + r) * XQ_RQ_ROW and (mode 2) its sampled id to out_ids[r], counter meta[3] + i0 + r.
// Stage 1: block b stages its slice in smem and extracts its top-32 by 32 rounds of (block max
// of the per-thread best key -> the owner drops it and rescans its strided elements); stage 2:
// the LAST block of the row (threadfence + per-row counter, self re-arming like xq_argmax_rows)
// merges the nb block lists the same way — exact: keys are distinct (the id is in the low word)
// and every global top-32 key is in its block's top-32. Deterministic: the order is total.
extern "C" __global__ void __launch_bounds__(256) xq_rq_draft(
        int* __restrict__ out_ids, const unsigned short* __restrict__ logits, int V,
        unsigned long long* __restrict__ part, unsigned int* __restrict__ cnt,
        unsigned int* __restrict__ list, const unsigned int* __restrict__ dsamp,
        const int* __restrict__ meta, int i0, int mode) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned short sv[XQ_RQ_SLICE];   // stage 1 slice; stage 2 candidates (u64)
    __shared__ unsigned long long wk[2][32];
    __shared__ unsigned long long top[XQ_RQ_K];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const unsigned short* row = logits + (long long)r * V;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;     // host: chunk <= XQ_RQ_SLICE
    const int lo = min(b * chunk, V), hi = min(lo + chunk, V), n = hi - lo;
    // mode 2 needs only the first top_k candidates (q' truncates there): extract that many
    const unsigned int tk = mode == 2 ? dsamp[3] : 0u;
    const int kr = (tk > 0u && tk < (unsigned int)XQ_RQ_K) ? (int)tk : XQ_RQ_K;
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    if ((reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull) {
        const int nv = n >> 3;                              // lo is 8-aligned: row + lo stays 16-B aligned
        const uint4* rv = reinterpret_cast<const uint4*>(row + lo);
        uint4* sv4 = reinterpret_cast<uint4*>(sv);
        for (int q = tid; q < nv; q += bs) sv4[q] = __ldg(rv + q);
        for (int j = nv * 8 + tid; j < n; j += bs) sv[j] = row[lo + j];
    } else {
        for (int j = tid; j < n; j += bs) sv[j] = row[lo + j];
    }
    __syncthreads();
    unsigned long long own = 0ull;
    for (int j = tid; j < n; j += bs) { const unsigned long long x = xq_rq_key(sv[j], lo + j); own = x > own ? x : own; }
    for (int it = 0; it < kr; it++) {
        // double-buffered warp partials: one barrier per round (a thread can only rewrite
        // wk[it & 1] after every thread passed round it + 1's barrier, i.e. finished reading it)
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;                               // block-uniform (all read the same partials)
        if (tid == 0) top[it] = k;
        if (own == k) {                                     // keys are unique: exactly one owner
            sv[(int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull)) - lo] = 0xFFFFu;   // -NaN: dropped
            own = 0ull;
            for (int j = tid; j < n; j += bs) { const unsigned long long x = xq_rq_key(sv[j], lo + j); own = x > own ? x : own; }
        }
    }
    __syncthreads();
    if (tid < XQ_RQ_K) part[((long long)r * nb + b) * XQ_RQ_K + tid] = top[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
    __syncthreads();
    if (!last) return;
    __threadfence();
    unsigned long long* cand = reinterpret_cast<unsigned long long*>(sv);    // nb * 32 <= 1024 keys (8 KB)
    const int nc = nb * XQ_RQ_K;
    for (int j = tid; j < nc; j += bs) cand[j] = __ldcg(part + (long long)r * nb * XQ_RQ_K + j);
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    __syncthreads();
    own = 0ull;
    for (int j = tid; j < nc; j += bs) own = cand[j] > own ? cand[j] : own;
    for (int it = 0; it < kr; it++) {
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;
        if (tid == 0) top[it] = k;
        if (own == k) {
            own = 0ull;
            for (int j = tid; j < nc; j += bs) {
                if (cand[j] == k) cand[j] = 0ull;
                own = cand[j] > own ? cand[j] : own;
            }
        }
    }
    __syncthreads();
    if (tid == 0) cnt[r] = 0u;                              // re-arm for the next launch / replay
    if (wid != 0) return;
    // warp 0: lane j = candidate j (sorted desc; invalid entries trail)
    const unsigned long long kj = top[lane];
    const bool valid = kj != 0ull;
    int id = valid ? (int)(0xFFFFFFFFu - (unsigned)(kj & 0xFFFFFFFFull)) : -1;
    const float lg = valid ? xq_skey_val((unsigned)(kj >> 32)) : -INFINITY;
    float q = 0.0f;
    if (mode == 2) {
        const float tau = fmaxf(__uint_as_float(dsamp[1]), 1e-6f), invT = 1.0f / tau;
        const float top_p = __uint_as_float(dsamp[2]);
        const unsigned int top_k = dsamp[3];
        const unsigned long long seed = ((unsigned long long)dsamp[5] << 32) | (unsigned long long)dsamp[4];
        const unsigned int ctr = (unsigned int)meta[3] + (unsigned int)(i0 + r);
        const bool any = __shfl_sync(0xffffffffu, (int)valid, 0) != 0;
        int pick = 0;
        if (!any) {
            // no finite-key candidate at all: the delta proposal q' = 1{0} (still exact — the
            // verify then runs the match rule's law against token 0)
            id = lane == 0 ? 0 : -1;
            q = lane == 0 ? 1.0f : 0.0f;
        } else {
            const float l0 = __shfl_sync(0xffffffffu, lg, 0);
            const bool keep = valid && (top_k == 0u || (unsigned)lane < top_k);
            float w = keep ? __expf((lg - l0) * invT) : 0.0f;
            float tot = w;
            for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (top_p < 1.0f && tot > 0.0f) {
                // smallest prefix whose mass reaches top_p * tot: keep j iff mass(0..j-1) < target
                float incl = w;
                for (int o = 1; o < 32; o <<= 1) {
                    const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += nn;
                }
                float excl = __shfl_up_sync(0xffffffffu, incl, 1);
                if (lane == 0) excl = 0.0f;
                if (!(excl < top_p * tot)) w = 0.0f;
            }
            float z = w;
            for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
            if (!(z > 0.0f && z < INFINITY)) {               // non-finite list: delta on candidate 0
                w = lane == 0 ? 1.0f : 0.0f;
                z = 1.0f;
            }
            q = w / z;
            // inverse CDF over the kept prefix in list order (warp-uniform, deterministic)
            const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_DRAFT) * z;
            float incl = w;
            for (int o = 1; o < 32; o <<= 1) {
                const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += nn;
            }
            const unsigned int hit = __ballot_sync(0xffffffffu, w > 0.0f && incl > target);
            const unsigned int pos = __ballot_sync(0xffffffffu, w > 0.0f);
            const int pl = hit ? __ffs(hit) - 1 : 31 - __clz(pos);
            pick = __shfl_sync(0xffffffffu, id, pl);
        }
        if (lane == 0) out_ids[r] = pick;
    }
    unsigned int* L = list + (long long)(i0 + r) * XQ_RQ_ROW;
    L[lane] = (unsigned int)id;
    L[32 + lane] = __float_as_uint(lg);
    L[64 + lane] = __float_as_uint(q);
}

// A5-L4 (spec.ratio_dhead, opt-in): a real-q (mode 2) draft pass on DHEAD instead of the full
// slice head. Launched right after xq_dh_rescore of pass i (same stream, one CTA): re-derives the
// ns x 128 exact candidate logits from the rescore's fp32 split-K partials with ITS step-3 ops
// (ascending-s sum -> f16 yraw -> had128 -> x svh -> f16 RN; the same values the rescore's argmax
// compared), takes their top-kr with xq_rq_draft's keys and order, and runs xq_rq_draft's mode-2
// tail (xq_rq_emit): q' = the request's top-k -> tau -> top-p law over DHEAD's CANDIDATE set, the
// draft ~ q' (counter meta[3] + i, domain XQ_RQ_DOM_DRAFT), list row i. Tokens outside the
// candidate blocks get q' = 0: they are never drafted, and the verify (xq_sample_rows_rq) reads
// q' only through list row i, so its residual max(0, p - q') is p there — speculative sampling
// is exact for ANY q' as long as the draft is drawn from the q' the verify reads (this row).
// Overwrites the rescore's argmax / d_dev[i] / toks[0] with the sampled draft; dconf[i] stays
// the rescore's in-candidate max (the DDS stop reads only draft-side quantities).
// geom = xq_dh_rescore's (pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32).
#define XQ_RQDH_MAX (DH_TCAP * 128)
extern "C" __global__ void __launch_bounds__(256) xq_rq_draft_dh(
        int* __restrict__ out_id, int* __restrict__ d_dev, int* __restrict__ toks,
        const __half* __restrict__ svh, const unsigned* __restrict__ ws,
        unsigned int* __restrict__ list, const unsigned int* __restrict__ dsamp,
        const int* __restrict__ meta, long long geom, int N) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned short sv[XQ_RQDH_MAX];
    __shared__ int s_cand[DH_TCAP];
    __shared__ unsigned long long wk[2][32];
    __shared__ unsigned long long top[XQ_RQ_K];
    const int pass_i = (int)(geom & 0xFF), S = (int)((geom >> 16) & 0xFF), K = (int)(geom >> 32);
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;
    const int nblk = N >> 7;
    const int* dbg = reinterpret_cast<const int*>(ws + dh_off_dbg(K, nblk));
    const float* part = reinterpret_cast<const float*>(ws + dh_off_part(K, nblk));
    const int ns = max(min(__ldcg(dbg), DH_TCAP), 0);
    if (tid < ns) s_cand[tid] = __ldcg(dbg + 1 + tid);
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    __syncthreads();
    // 1. the candidate logits (xq_dh_rescore step 3, op for op) -> sv[rr * 128 + col]
    for (int rr = wid; rr < ns; rr += nw) {
        const int bb = s_cand[rr];
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* pc = part + (size_t)rr * S * 128 + lane * 4 + q;
            float a = 0.f;
            for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
            v[q] = __half2float(__float2half_rn(a));
        }
        exl3_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = bb * 128 + lane * 4 + q;
            sv[rr * 128 + lane * 4 + q] = __half_as_ushort(__float2half_rn(v[q] * __half2float(svh[col])));
        }
    }
    __syncthreads();
    // 2. top-kr (xq_rq_draft stage 1 over the candidate positions; key = (logit, lower id))
    const unsigned int tk = dsamp[3];
    const int kr = (tk > 0u && tk < (unsigned int)XQ_RQ_K) ? (int)tk : XQ_RQ_K;
    const int n = ns * 128;
    unsigned long long own = 0ull;
    int oj = -1;
    for (int j = tid; j < n; j += bs) {
        const unsigned long long x = xq_rq_key(sv[j], s_cand[j >> 7] * 128 + (j & 127));
        if (x > own) { own = x; oj = j; }
    }
    for (int it = 0; it < kr; it++) {
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;                               // block-uniform
        if (tid == 0) top[it] = k;
        if (own == k) {                                     // keys are unique: exactly one owner
            sv[oj] = 0xFFFFu;                               // -NaN: dropped
            own = 0ull; oj = -1;
            for (int j = tid; j < n; j += bs) {
                const unsigned long long x = xq_rq_key(sv[j], s_cand[j >> 7] * 128 + (j & 127));
                if (x > own) { own = x; oj = j; }
            }
        }
    }
    __syncthreads();
    if (wid != 0) return;
    // 3. q' + the draft + list row pass_i (xq_rq_draft's mode-2 tail)
    const int pick = xq_rq_emit(top, lane, 2, dsamp, meta, pass_i, out_id, list + (long long)pass_i * XQ_RQ_ROW);
    if (lane == 0) { d_dev[pass_i] = pick; toks[0] = pick; }
}

// residual weight of token i (plain weight w > 0 = a target survivor): d itself 0; a listed id
// max(0, w - q'(i) * tot) (= tot * max(0, p - q')); any other survivor w. The bloom filter
// (1024 bits over the <= 32 listed ids) keeps the list scan off almost every element.
__device__ __forceinline__ float xq_rq_resw(int i, float w, int d, float tot, const int* l_id,
                                            const float* l_q, const unsigned int* bloom) {
    if (w <= 0.0f || i == d) return 0.0f;
    if ((bloom[(i >> 5) & 31] >> (i & 31)) & 1u) {
        for (int j = 0; j < XQ_RQ_K; j++)
            if (l_id[j] == i) return fmaxf(0.0f, w - __fmul_rn(l_q[j], tot));   // no FMA: both passes agree
    }
    return w;
}

// WP24 verify sampler: xq_sample_rows (steps 1-5 below are its code verbatim) + the real-q
// rule on rows r < nratio staged with flag 2 (the draft d_r = d_vec[r], its proposal = list row
// r). Every other sampled row — the bonus row, flag-1 rows — is xq_sample_rows exactly (same
// draw, same id). Launched only by real-q verify graphs (never on the default path).
extern "C" __global__ void __launch_bounds__(1024) xq_sample_rows_rq(
        int* __restrict__ out_ids, const __half* __restrict__ logits, int V,
        const unsigned int* __restrict__ samp, const int* __restrict__ d_vec,
        const unsigned int* __restrict__ list, int nratio) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.x;
    const unsigned int* P = samp + row * 8;
    if (P[0] == 0u) return;
    const float T = fmaxf(__uint_as_float(P[1]), 1e-6f);
    const float invT = 1.0f / T;
    const float top_p = __uint_as_float(P[2]);
    const unsigned int top_k = P[3];
    const unsigned long long seed = ((unsigned long long)P[5] << 32) | (unsigned long long)P[4];
    const unsigned int ctr = P[6];
    const float min_p = __uint_as_float(P[7]);
    const unsigned short* L = reinterpret_cast<const unsigned short*>(logits) + (long long)row * V;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;

    __shared__ unsigned int ush[32];
    __shared__ float fsh[32];
    __shared__ unsigned int hist[256];
    __shared__ float fh[256];
    __shared__ unsigned int s_u[4];
    __shared__ float s_f[2];
    __shared__ int l_id[XQ_RQ_K];
    __shared__ float l_q[XQ_RQ_K];
    __shared__ unsigned int bloom[32];
    __shared__ int s_d, s_acc;
    __shared__ float s_tot;

    // 1. max key -> lmax (softmax shift)
    unsigned int mk = 0u;
    for (int i = tid; i < V; i += bs) mk = max(mk, xq_skey(L[i]));
    mk = xq_bmax_u32(mk, ush);
    const float lmax = xq_skey_val(mk);

    // 2. top-k key threshold tau (radix select over the 16-bit key, ties included)
    unsigned int tau = 0u;
    if (top_k > 0u && top_k < (unsigned int)V) {
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) atomicAdd(&hist[xq_skey(L[i]) >> 8], 1u);
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, hb = 0u, need = 1u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= top_k) { hb = (unsigned int)b; need = top_k - cum; break; }
                cum += hist[b];
            }
            s_u[0] = hb; s_u[1] = need;
        }
        __syncthreads();
        const unsigned int hb = s_u[0], need = s_u[1];
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if ((k >> 8) == hb) atomicAdd(&hist[k & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, lb = 0u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= need) { lb = (unsigned int)b; break; }
                cum += hist[b];
            }
            s_u[2] = (hb << 8) | lb;
        }
        __syncthreads();
        tau = s_u[2];
    }
    const float mfloor = (min_p > 0.0f) ? lmax + T * __logf(min_p) : -INFINITY;

    // 3. candidate mass S (deterministic)
    float part = 0.0f;
    for (int i = tid; i < V; i += bs) {
        unsigned int k = xq_skey(L[i]);
        if (k >= tau) {
            float x = xq_skey_val(k);
            if (x >= mfloor) part += __expf((x - lmax) * invT);
        }
    }
    const float S = xq_bsum_f32(part, fsh);

    // 4. top-p key threshold theta (mass radix select; boundary ties included)
    unsigned int thr = tau;
    if (top_p < 1.0f && S > 0.0f) {
        const float target = top_p * S;
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k >> 8], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int hb = 0u; float rem = 0.0f; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= target) { hb = (unsigned int)b; rem = target - cum; found = 1; break; }
                cum += fh[b];
            }
            if (!found) { hb = (unsigned int)max(last, 0); rem = 3.0e38f; }
            s_u[0] = hb; s_f[0] = rem;
        }
        __syncthreads();
        const unsigned int hb = s_u[0];
        const float rem = s_f[0];
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau && (k >> 8) == hb) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k & 255u], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int lb = 0u; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= rem) { lb = (unsigned int)b; found = 1; break; }
                cum += fh[b];
            }
            if (!found) lb = (unsigned int)max(last, 0);
            s_u[3] = max((hb << 8) | lb, tau);
        }
        __syncthreads();
        thr = s_u[3];
    }

    // 5. per-warp survivor mass of p: warp w owns [w*chunk, (w+1)*chunk), lanes stride 32.
    const int chunk = (V + nw - 1) / nw;
    const int w0 = wid * chunk, w1 = min(V, w0 + chunk);
    float lsum = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) lsum += __expf((x - lmax) * invT);
        }
    }
    for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) fsh[wid] = lsum;
    const bool ratio = P[0] == 2u && row < nratio;          // block-uniform
    if (ratio && tid < XQ_RQ_K) {
        const unsigned int* Lr = list + (long long)row * XQ_RQ_ROW;
        l_id[tid] = (int)Lr[tid];
        l_q[tid] = __uint_as_float(Lr[64 + tid]);
        bloom[tid] = 0u;
    }
    __syncthreads();
    if (!ratio) {
        // plain row (the bonus row, flag-1 rows): xq_sample_rows' inverse CDF, verbatim
        if (tid == 0) {
            const float u = ((float)(xq_mix64(seed ^ xq_mix64((unsigned long long)ctr + 1ull)) >> 40) + 0.5f)
                            * (1.0f / 16777216.0f);
            float tot = 0.0f;
            for (int w = 0; w < nw; w++) tot += fsh[w];
            const float target = u * tot;
            float cum = 0.0f; int sel = -1, last = 0;
            for (int w = 0; w < nw; w++) {
                if (fsh[w] > 0.0f) last = w;
                if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
                cum += fsh[w];
            }
            if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
            s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
        }
        __syncthreads();
        if (wid == (int)s_u[0]) {
            const float want = s_f[0];
            float base = 0.0f;
            int pick = -1, lastidx = -1;
            for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
                const int i = i0 + lane;
                float wgt = 0.0f;
                if (i < w1) {
                    unsigned int k = xq_skey(L[i]);
                    if (k >= thr) {
                        float x = xq_skey_val(k);
                        if (x >= mfloor) wgt = __expf((x - lmax) * invT);
                    }
                }
                float incl = wgt;
                for (int o = 1; o < 32; o <<= 1) {
                    float n = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += n;
                }
                unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
                unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
                if (any) lastidx = i0 + 31 - __clz(any);
                if (hit) pick = i0 + __ffs(hit) - 1;
                base += __shfl_sync(0xffffffffu, incl, 31);
            }
            if (lane == 0 && (pick >= 0 || lastidx >= 0)) out_ids[row] = (pick >= 0) ? pick : lastidx;
        }
        return;
    }

    // 6. ratio row: accept d with probability min(1, p(d) / q'(d)) <=> u * q'(d) * tot < w(d)
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < nw; w++) tot += fsh[w];
        const int d = d_vec[row];
        float qd = 0.0f;
        for (int j = 0; j < XQ_RQ_K; j++) if (l_id[j] == d && l_q[j] > 0.0f) { qd = l_q[j]; break; }
        if (!(qd > 0.0f)) {
            // d outside its pass's list (a pass always samples from its list, so only a stale /
            // foreign list lands here): the delta proposal 1{d} — exactly the match rule's law
            for (int j = 0; j < XQ_RQ_K; j++) { l_id[j] = -1; l_q[j] = 0.0f; }
            l_id[0] = d; l_q[0] = 1.0f; qd = 1.0f;
        }
        float wd = 0.0f;
        if (d >= 0 && d < V) {
            unsigned int k = xq_skey(L[d]);
            if (k >= thr) {
                float x = xq_skey_val(k);
                if (x >= mfloor) wd = __expf((x - lmax) * invT);
            }
        }
        const float ua = xq_rq_u24(seed, ctr, XQ_RQ_DOM_ACC);
        s_acc = (tot > 0.0f && ua * qd * tot < wd) ? 1 : 0;
        s_d = d; s_tot = tot;
    }
    __syncthreads();
    const float tot = s_tot;
    const int d = s_d;
    if (!(tot > 0.0f)) return;                              // no survivor at all: keep xq_argmax's id
    if (s_acc) { if (tid == 0) out_ids[row] = d; return; }
    if (tid < XQ_RQ_K && l_id[tid] >= 0) atomicOr(&bloom[(l_id[tid] >> 5) & 31], 1u << (l_id[tid] & 31));
    __syncthreads();
    // 7. reject: sample norm(max(0, p - q')) — per-warp residual mass, then the same inverse CDF
    float rs = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) rs += xq_rq_resw(i, __expf((x - lmax) * invT), d, tot, l_id, l_q, bloom);
        }
    }
    for (int o = 16; o; o >>= 1) rs += __shfl_xor_sync(0xffffffffu, rs, o);
    if (lane == 0) fsh[wid] = rs;                           // tid 0 read fsh before the last barrier
    __syncthreads();
    if (tid == 0) {
        float R = 0.0f;
        for (int w = 0; w < nw; w++) R += fsh[w];
        const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_RES) * R;
        float cum = 0.0f; int sel = -1, last = 0;
        for (int w = 0; w < nw; w++) {
            if (fsh[w] > 0.0f) last = w;
            if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
            cum += fsh[w];
        }
        if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
        s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
        // R == 0: p == q' up to rounding, so a reject had probability ~0 — emit d (the limit law)
        s_u[1] = R > 0.0f ? 1u : 0u;
    }
    __syncthreads();
    if (s_u[1] == 0u) { if (tid == 0) out_ids[row] = d; return; }
    if (wid == (int)s_u[0]) {
        const float want = s_f[0];
        float base = 0.0f;
        int pick = -1, lastidx = -1;
        for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
            const int i = i0 + lane;
            float wgt = 0.0f;
            if (i < w1) {
                unsigned int k = xq_skey(L[i]);
                if (k >= thr) {
                    float x = xq_skey_val(k);
                    if (x >= mfloor) wgt = xq_rq_resw(i, __expf((x - lmax) * invT), d, tot, l_id, l_q, bloom);
                }
            }
            float incl = wgt;
            for (int o = 1; o < 32; o <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
            unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
            if (any) lastidx = i0 + 31 - __clz(any);
            if (hit) pick = i0 + __ffs(hit) - 1;
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) out_ids[row] = (pick >= 0) ? pick : (lastidx >= 0 ? lastidx : d);
    }
}

