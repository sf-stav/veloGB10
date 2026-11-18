// exl3_bench/19_qsa_xcheck_sel.cuh: WP18 XCHECK diagnostics and the QSA selected-set attention (sel splitk) kernels.
// XCHECK kernels are eager and diagnostics only.
// Section 19 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// WP18 XCHECK (eager, diagnostics only): bitwise diff of a scored slab tile against the old full
// matrix (qsa_score_prefill_b's output, row pitch nblk_stride; slab row r = chunk row t_first + r).
// cnt[0] += differing entries, cnt[1] = min packed (chunk row << 16 | block) of a difference,
// cnt[2] += compared entries. Grid (rows), block 256.
extern "C" __global__ void xq_wp18_cmp(int* __restrict__ cnt, const float* __restrict__ slab,
                                       const float* __restrict__ full, const struct QsaParamsX* __restrict__ p,
                                       int pos_first, int t_first, int kt, int nblk_stride) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int nb = (pos_first + r + 1) / p->ratio;
    const int kT = kt * WP18_T;
    const int nk = min(max(nb - kT, 0), WP18_T);
    const float* a = slab + (long long)r * WP18_T;
    const float* b = full + (long long)(t_first + r) * nblk_stride + kT;
    int nd = 0;
    for (int jj = threadIdx.x; jj < nk; jj += blockDim.x) {
        if (__float_as_uint(a[jj]) != __float_as_uint(b[jj])) {
            nd++;
            atomicMin(&cnt[1], ((t_first + r) << 16) | ((kT + jj) & 0xFFFF));
        }
    }
    if (nd) atomicAdd(&cnt[0], nd);
    if (threadIdx.x == 0 && nk > 0) atomicAdd(&cnt[2], nk);
}

// ---- 4. K/V cache write + roped-q staging for the gathered decode. Grid
// (m*nkv) 1-D: row = blockIdx.x/nkv, kvh = blockIdx.x%nkv; block hd (=256).
// (a) K: the EXISTING shared xq_k_norm_rope (bit-identical cache words vs
// xq_attn_decode / xq_attn_decode_group / xq_attn_kv_prefill) then the cache
// write with xq_attn_decode's exact slotbase/kcbase/vcbase arithmetic; V: raw
// f16 -> f32. (b) q: each q head of the kv group normed+roped EXACTLY as
// xq_attn_decode's q section (v^2 tree, (1+w) scale, partial rotate-half with
// c/s; sequential heads over the shared qbuf), staged as f32 to qstage rows
// [row*nh+h][hd] for the splitk kernel. qnkn = q norm [hd] | k norm [hd] f16.
extern "C" __global__ void xq_attn_prep(float* __restrict__ qstage, float* __restrict__ kcvc,
                                        const __half* __restrict__ qg, const __half* __restrict__ kp,
                                        const __half* __restrict__ vp, const __half* __restrict__ qnkn,
                                        const float* __restrict__ cossin, const int* __restrict__ slotpos,
                                        int nh_packed, int hd_packed, int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int row = blockIdx.x / nkv;
    const int kvh = blockIdx.x % nkv;
    const int gqa = nh / nkv;
    const int tid = threadIdx.x;
    const int slot = slotpos[row * 2 + 0];
    const int p = slotpos[row * 2 + 1];
    const float* c = cossin + (long long)row * 2 * rdim;
    const float* s = c + rdim;
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    // ---- (a) K: shared fn = bit-identical cache words; V: raw f16 -> f32
    xq_k_norm_rope(kbuf, red, kp + (long long)row * nkv * hd + (long long)kvh * hd,
                   kn_w, c, s, tid, hd, rdim, eps);
    __syncthreads();
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)row * nkv * hd + (long long)kvh * hd + tid]));
    __syncthreads();
    // ---- (b) q: rmsnorm (1+w), rope, f32 staging — xq_attn_decode's q order,
    // sequential heads over the shared qbuf (syncthreads as in the original).
    for (int h = kvh * gqa; h < kvh * gqa + gqa; h++) {
        float qv = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)h * hd * 2 + tid]);
        red[tid] = qv * qv; __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
        qbuf[tid] = qv; __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        }
        __syncthreads();
        qstage[((long long)row * nh + h) * hd + tid] = xq_kv_rot(qbuf[tid], kc.fmt, tid & 31);
        __syncthreads();
    }
}

// ---- 5. gathered flash-decode split-K: xq_attn_decode's two-pass softmax
// replaced by an ONLINE (streaming) softmax over the row's selection list —
// rank r -> cache column sel[row*sel_max+r], warp-uniform load; -1 padding
// entries skipped. Grid (m*nkv*nsplits) 1-D: sp = blockIdx.x % nsplits,
// kvh = (blockIdx.x/nsplits) % nkv, row = blockIdx.x/(nsplits*nkv); block
// 256 = 8 warps; warp w owns head kvh*gqa + w and, when w+8 < gqa, ALSO head
// kvh*gqa + w + 8 (sequential, separate register sets; this model: gqa 12,
// warps 8 -> warps 0..3 take two heads). Lane owns V dims [lane*8, lane*8+8)
// (hd = 256 = blockDim.x); logit = dot * rsqrtf(hd) exactly matching
// xq_attn_softmax's `red[0] * scale`; all state f32, __expf only, entries
// processed r ASCENDING. Empty split (begin >= nsel): pm = -INFINITY, pl = 0,
// pacc = 0 — the combine skips those splits. Partials per (row, head, split):
// pm/pl [hg*nsplits + sp], pacc [(hg*nsplits + sp)*hd + d], hg = row*nh + h.
extern "C" __global__ void xq_attn_sel_splitk(float* __restrict__ pm, float* __restrict__ pl,
                                              float* __restrict__ pacc, const float* __restrict__ qstage,
                                              const float* __restrict__ kcvc, const int* __restrict__ sel,
                                              const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                              int nh_packed, long long geom, int split_len) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    for (int hh = 0; hh < 2; hh++) {             // head slot 0: warp; slot 1: warp+8
        const int w = warp + hh * 8;
        if (w >= gqa) break;                     // second slot only when warp+8 < gqa
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int r = begin; r < end; r++) {
            const int t = sel[(long long)row * sel_max + r];   // warp-uniform (-1 padded)
            if (t < 0) continue;
            float krow8[8];
            xq_kv_ld8(krow8, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qrow[lane * 8 + i] * krow8[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            const float s = dot * rsqrtf((float)hd);
            const float m_new = fmaxf(m, s);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(s - m_new);
            l = l * scale_f + wt;
            float vrow8[8];
            xq_kv_ld8(vrow8, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vrow8[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- 5a. S-A3-v: xq_attn_sel_splitk with 128-bit loads (prefill-only twin; decode keeps
// splitk). splitk's scalar lane*8+i loads cost 8 L1 wavefronts EACH (24 per entry per head):
// the prefill gather was LSU-bound (133 ms/call @8.5K), not DRAM-bound — a tiled 4-row walk
// with smem-staged K/V (cross-row reuse) measured 142 ms, and with float4 loads 47 ms, vs
// this kernel's high-occupancy one-row layout at ~28 ms. Same values, same expression order:
// bit-identical partials to splitk (GB10_EXL3_SEL_XCHECK, all 48 calls @8.5K).
extern "C" __global__ void xq_attn_sel_splitk4(float* __restrict__ pm, float* __restrict__ pl,
                                              float* __restrict__ pacc, const float* __restrict__ qstage,
                                              const float* __restrict__ kcvc, const int* __restrict__ sel,
                                              const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                              int nh_packed, long long geom, int split_len) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    for (int hh = 0; hh < 2; hh++) {             // head slot 0: warp; slot 1: warp+8
        const int w = warp + hh * 8;
        if (w >= gqa) break;                     // second slot only when warp+8 < gqa
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float qv[8];
        *reinterpret_cast<float4*>(qv) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int r = begin; r < end; r++) {
            const int t = sel[(long long)row * sel_max + r];   // warp-uniform (-1 padded)
            if (t < 0) continue;
            float kv[8], vv[8];
            xq_kv_ld8(kv, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qv[i] * kv[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            const float s = dot * rsqrtf((float)hd);
            const float m_new = fmaxf(m, s);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(s - m_new);
            l = l * scale_f + wt;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vv[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- 5b. WP14 rungs 1/2 (PLAN/SURPASS_PLAN_2026-09-26.md): head-grouped gathered split-K,
// BIT-IDENTICAL partials to 5./5a. (GB10_EXL3_GATHER_XCHECK=1 diffs them). 5a's CTA is 8 warps
// over gqa=12 heads, so warps 0..3 walk their split TWICE (heads w and w+8), and every warp of
// every walk issues its own dependent sel -> K/V row loads (~65 serial DRAM round trips per walk,
// 12x redundant row requests per CTA). Here:
//  - warp w owns heads w, w+nwarps, ... (HPW per warp; blockDim = 32*ceil(gqa/HPW)): ONE walk;
//  - the split's selection entries are staged once in smem (no sel load on the critical path);
//  - the CTA stages each selected K and V row ONCE into a cp.async ring (XQ_HG_STAGES entries,
//    16-B copies, bypass-L1) and every warp reads its lane's 8 dims from smem, so up to
//    XQ_HG_STAGES-1 rows are in flight while the current one is consumed (one barrier per entry).
// The same grid / splits / split_len as 5a (identical partial grouping), entries ascending, -1
// entries skipped, and per head per entry 5a's arithmetic pinned against contraction to the SASS
// 5a compiles to: dot = FFMA chain from +0 over dims ascending, xor tree 16..1 (FADD), broadcast of
// lane 0, s = FMUL(dot, rsqrtf(hd)) with hd a RUNTIME value (5a: blockDim.x — never constant
// folded), m_new = fmaxf, __expf both ways, l = FFMA(l, scale, wt), acc = FFMA(acc, scale,
// FMUL(wt, v)). K/V values are the same bytes read through the same xq_kv_ld8 decode. hd must be
// 256 (lane owns 8 dims) and a row must fit XQ_HG_ROWB; the host dispatch checks both.
// split_hd = split_len | hd << 16.
#define XQ_HG_STAGES 8           // 7 selected rows in flight per CTA (16 KiB of f32 K+V)
#define XQ_HG_CHUNK 256
#define XQ_HG_ROWB 1024
// Stage entry e of the current chunk into ring slot e % XQ_HG_STAGES (whole CTA, 16 B per copy),
// then commit ONE group (empty for e >= n or t < 0: keeps the wait_group accounting aligned).
__device__ __forceinline__ void xq_hg_issue(char (*ring)[2][XQ_HG_ROWB], const int* s_t, int e, int n,
                                            const XqKv& kc) {
    if (e < n) {
        const int t = s_t[e];
        if (t >= 0) {
            const int st = e % XQ_HG_STAGES;
            const int c16 = (int)(kc.rb >> 4);
            for (int c = threadIdx.x; c < 2 * c16; c += blockDim.x) {
                const int isv = c >= c16;
                const int off = (c - isv * c16) << 4;
                const char* src = (isv ? kc.v : kc.k) + (long long)t * kc.rb + off;
                moe_coop_cp16(&ring[st][isv][off], src, 16);
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int HPW>
__device__ __forceinline__ void xq_attn_sel_hg_body(float* __restrict__ pm, float* __restrict__ pl,
                                                    float* __restrict__ pacc, const float* __restrict__ qstage,
                                                    const float* __restrict__ kcvc, const int* __restrict__ sel,
                                                    const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                                    int nh_packed, long long geom, int split_hd) {
    __shared__ __align__(16) char ring[XQ_HG_STAGES][2][XQ_HG_ROWB];
    __shared__ int s_t[XQ_HG_CHUNK];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int split_len = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int gqa = nh / nkv;
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    const float rsq = rsqrtf((float)hd);
    float qv[HPW][8], acc[HPW][8], mx[HPW], lx[HPW];
    #pragma unroll
    for (int g = 0; g < HPW; g++) {
        mx[g] = -INFINITY; lx[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[g][i] = 0.0f; qv[g][i] = 0.0f; }
        const int w = warp + g * nwarps;
        if (w < gqa) {
            const float* qrow = qstage + ((long long)row * nh + kvh * gqa + w) * hd;
            *reinterpret_cast<float4*>(qv[g]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
            *reinterpret_cast<float4*>(qv[g] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        }
    }
    for (int cb = begin; cb < end; cb += XQ_HG_CHUNK) {
        const int n = min(XQ_HG_CHUNK, end - cb);
        __syncthreads();                                 // the previous chunk's ring/s_t readers are done
        for (int i = threadIdx.x; i < n; i += blockDim.x) s_t[i] = sel[(long long)row * sel_max + cb + i];
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < XQ_HG_STAGES - 1; e++) xq_hg_issue(ring, s_t, e, n, kc);
        for (int r = 0; r < n; r++) {
            // groups committed: (STAGES-1) + r; entry r is group r -> at most STAGES-2 may pend
            asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_HG_STAGES - 2) : "memory");
            __syncthreads();                             // entry r visible; slot (r-1)%S free
            xq_hg_issue(ring, s_t, r + XQ_HG_STAGES - 1, n, kc);
            const int t = s_t[r];                        // block-uniform
            if (t < 0) continue;
            const int st = r % XQ_HG_STAGES;
            float kv[8], vv[8];
            xq_kv_ld8(kv, ring[st][0], kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, ring[st][1], kc.fmt, hd, lane * 8);
            #pragma unroll
            for (int g = 0; g < HPW; g++) {
                if (warp + g * nwarps >= gqa) continue;  // warp-uniform
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) dot = __fmaf_rn(qv[g][i], kv[i], dot);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
                dot = __shfl_sync(0xffffffffu, dot, 0);
                const float s = __fmul_rn(dot, rsq);
                const float m_new = fmaxf(mx[g], s);
                const float scale_f = __expf(mx[g] - m_new);
                const float wt = __expf(s - m_new);
                lx[g] = __fmaf_rn(lx[g], scale_f, wt);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = __fmaf_rn(acc[g][i], scale_f, __fmul_rn(wt, vv[i]));
                mx[g] = m_new;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // only empty groups can be pending
    #pragma unroll
    for (int g = 0; g < HPW; g++) {
        const int w = warp + g * nwarps;
        if (w >= gqa) continue;
        const long long hg = (long long)row * nh + kvh * gqa + w;
        if (lane == 0) {
            pm[hg * nsplits + sp] = mx[g];
            pl[hg * nsplits + sp] = lx[g];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[g][i];
    }
}
// Rung 1: one head per warp (block 32*gqa = 384 at gqa 12; served at m = 1). Rung 2: three heads
// per warp (block 32*ceil(gqa/3) = 128; served at m >= 2) — one smem K/V read feeds three heads.
extern "C" __global__ void __launch_bounds__(384)
xq_attn_sel_hg1(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    xq_attn_sel_hg_body<1>(pm, pl, pacc, qstage, kcvc, sel, pos_sel, slotpos, nh_packed, geom, split_hd);
}
extern "C" __global__ void __launch_bounds__(128)
xq_attn_sel_hg3(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    xq_attn_sel_hg_body<3>(pm, pl, pacc, qstage, kcvc, sel, pos_sel, slotpos, nh_packed, geom, split_hd);
}

// ---- 5d. W4/QSA (decode wave 4, gap #4): the ROW-SHARED union gather for multi-row verifies.
// BIT-IDENTICAL partials to 5./5a./5b. (GB10_W4QSA_XCHECK=1 diffs them against the old dispatch).
// 5b's grid is (row, kv head, split): at a 32K verify (m = 8) the 8 rows' CTAs for one (kv head,
// split) each walk their own ~65-entry sub-list and each stage every selected K/V row again,
// although the rows sit at consecutive positions and select heavily overlapping blocks. Here:
//  - grid (kv head, split, head group of XQ_QU_HPW = 4 heads), grp fastest so the 3 CTAs sharing
//    one (kv head, split) run side by side (their K/V reads coalesce in L2); block 32*m:
//    warp r = verify ROW r, owning the group's 4 heads of that row (4 register chains);
//  - the CTA builds the UNION of the m rows' split sub-lists (per slot group; rows of different
//    slots never share an entry) in ascending token order with a per-entry row mask — a
//    windowed token bitmap (XQ_QU_W tokens per window) + popc prefix: O(1) per element;
//  - each union entry's K and V rows are staged ONCE (cp.async ring, XQ_QU_RC chunks of XQ_QU_S
//    entries, one barrier per chunk; f32 rows land in a lo/hi dim layout so every LDS.128 is
//    conflict-free; other formats are decoded ONCE per CTA by xq_kv_ld8 — the served reader —
//    into an f32 lo/hi tile) and every row whose mask bit is set consumes it.
// Exactness: row r visits exactly its own sub-list entries [sp*L, min((sp+1)*L, nsel)), in list
// order (the union is ascending and each row's valid list is strictly ascending, so its masked
// entries ARE its list), with the same split/partial grouping as 5b. Per head per entry the
// arithmetic is 5b's pinned sequence: dot = FFMA chain from +0 over the lane's 8 dims ascending,
// the xor tree 16..1 (FADD own + partner), s = FMUL(dot, rsqrtf(hd)) with hd RUNTIME, fmaxf,
// __expf both ways, l = FFMA(l, scale, wt), acc = FFMA(acc, scale, FMUL(wt, v)). The 4 chains of a
// warp share ONE transposed tree: level 16 exchanges halves (lanes < 16 keep chains 0,1, lanes
// >= 16 keep 2,3), level 8 keeps one chain per lane (lane>>3), levels 4,2,1 are the plain
// butterfly — every level adds exactly the (own, partner) pair 5b adds for that chain, so the
// sum is the same value, bit for bit (a butterfly leaves every lane with the same bits: FADD is
// commutative), and lanes 8c..8c+7 hold chain c's dot. The softmax step then runs once per chain
// (one lane group) and scale/wt are broadcast for the accumulate. Any row sub-list that is NOT
// strictly ascending / contains a -1 (never produced by xq_qsa_topk_asc*) switches the CTA to an
// exact fallback: each row alone, its list verbatim (order, duplicates, -1 skips as in 5b).
// Contract (host-checked): hd == 256, gqa % 4 == 0, 1 <= m <= XQ_QU_MAXR, split_len <= XQ_QU_MAXL,
// K/V row bytes <= 1024, dynamic smem = xq_qu_smem_bytes below (mirrored by the host).
#define XQ_QU_HPW 4
#define XQ_QU_S 2                        // union entries per chunk (one barrier per chunk)
#define XQ_QU_RC 4                       // chunks in the ring: RC-1 in flight while one is consumed
#define XQ_QU_NE (XQ_QU_S * XQ_QU_RC)
#define XQ_QU_MAXR 16
#define XQ_QU_MAXL 128
#define XQ_QU_W 8192                     // bitmap window (tokens)
#define XQ_QU_WW (XQ_QU_W / 32)
// smem: ring NE*2*rb | [non-f32] f32 tile S*2*hd*4 | sel m*L | u m*L | msk m*L | bm WW | wp WW
//       | n MAXR | slot MAXR | red 32 (ints)
// Stage chunk c (entries c*S .. c*S+S-1 of the union) into its ring slots, then commit ONE group
// (empty past the union / for -1 entries: keeps the wait_group accounting aligned). f32 rows are
// permuted to the lo/hi layout (16-B chunk k -> float4 slot (k&1)*32 + (k>>1)); others are raw.
__device__ __forceinline__ void xq_qu_issue(char* ring, const int* s_u, const unsigned* s_msk,
                                            const int* s_slot, int c, int N, const char* kcvc,
                                            long long rb, long long mp, int nkv, int kvh, int fmt) {
    const int e0 = c * XQ_QU_S;
    const int c16 = (int)(rb >> 4);
    const int per_e = 2 * c16;
    const int tot = min(XQ_QU_S, N - e0) * per_e;          // <= 0 past the union
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int j = q / per_e;
        const int e = e0 + j;
        const int t = s_u[e];
        if (t < 0) continue;
        const int k = q - j * per_e;
        const int isv = k >= c16;
        const int kk = k - isv * c16;                      // 16-B chunk of the row
        const int slot = s_slot[__ffs(s_msk[e]) - 1];      // every row of an entry shares the slot
        const char* src = kcvc + ((((long long)slot * 2 + isv) * nkv + kvh) * mp + t) * rb + ((long long)kk << 4);
        const int dk = fmt == XQ_KV_F32 ? ((kk & 1) * 32 + (kk >> 1)) : kk;
        moe_coop_cp16(ring + (long long)(e % XQ_QU_NE) * 2 * rb + isv * rb + (dk << 4), src, 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
extern "C" __global__ void __launch_bounds__(XQ_QU_MAXR * 32)
xq_attn_sel_union(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                  const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                  const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                  int nh_packed, long long geom, int split_hd, int m) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) char xq_qu_sm[];
    const unsigned FULL = 0xffffffffu;
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int mpf = (int)(geom >> 32);
    const int L = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int gqa = nh / nkv;
    const int ngrp = gqa / XQ_QU_HPW;
    const int grp = blockIdx.x % ngrp;
    const int sp = (blockIdx.x / ngrp) % nsplits;
    const int kvh = blockIdx.x / (ngrp * nsplits);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nthr = blockDim.x, nwarps = nthr >> 5;
    const int fmt = xq_kv_fmt(mpf);
    const long long mp = (long long)(mpf & 0x0FFFFFFF);
    const long long rb = xq_kv_rowb(fmt, hd);
    const int ml = m * L;
    const bool is_f32 = fmt == XQ_KV_F32;
    const int slot_b = 2 * (int)rb;                                // one ring entry: K row | V row
    const int tile_o = XQ_QU_NE * slot_b;                          // non-f32 only: f32 lo/hi tile
    char* ring = xq_qu_sm;
    int* s_sel = reinterpret_cast<int*>(xq_qu_sm + tile_o + (is_f32 ? 0 : XQ_QU_S * 2 * hd * 4));
    int* s_u = s_sel + ml;
    unsigned* s_msk = reinterpret_cast<unsigned*>(s_u + ml);
    unsigned* s_bm = s_msk + ml;
    int* s_wp = reinterpret_cast<int*>(s_bm + XQ_QU_WW);
    int* s_n = s_wp + XQ_QU_WW;
    int* s_slot = s_n + XQ_QU_MAXR;
    int* s_red = s_slot + XQ_QU_MAXR;
    const int begin = sp * L;

    // ---- the rows' split sub-lists (5b's begin/end), slots, masks cleared
    if (tid < m) {
        const int nsel = pos_sel[tid] + 1;
        s_n[tid] = max(0, min(begin + L, nsel) - begin);
        s_slot[tid] = slotpos[tid * 2 + 0];
    }
    for (int i = tid; i < ml; i += nthr) s_msk[i] = 0u;
    __syncthreads();
    for (int idx = tid; idx < ml; idx += nthr) {
        const int r = idx / L, i = idx - r * L;
        s_sel[idx] = i < s_n[r] ? sel[(long long)r * sel_max + begin + i] : 0x7fffffff;
    }
    __syncthreads();
    int bad = 0;
    for (int idx = tid; idx < ml; idx += nthr) {
        const int r = idx / L, i = idx - r * L;
        if (i < s_n[r]) {
            const int v = s_sel[idx];
            bad |= (v < 0) | (i > 0 && v <= s_sel[idx - 1]);
        }
    }
    bad = __syncthreads_or(bad);

    // ---- union (block-uniform N: every thread derives it from the same smem)
    int N = 0;
    if (bad) {
        // exact fallback: each row alone, its list verbatim
        for (int r = 0; r < m; r++) {
            const int n = s_n[r];
            for (int i = tid; i < n; i += nthr) { s_u[N + i] = s_sel[r * L + i]; s_msk[N + i] = 1u << r; }
            N += n;
        }
    } else {
        for (int g = 0; g < m; g++) {
            const int sg = s_slot[g];
            bool lead = true;
            for (int r = 0; r < g; r++) lead = lead && (s_slot[r] != sg);
            if (!lead) continue;                                   // group already done
            int lo = 0x7fffffff, hi_all = -1;
            for (int r = g; r < m; r++) {
                if (s_slot[r] == sg && s_n[r] > 0) {
                    lo = min(lo, s_sel[r * L]);
                    hi_all = max(hi_all, s_sel[r * L + s_n[r] - 1]);
                }
            }
            while (lo <= hi_all) {                                 // one token window [lo, lo+W)
                const int hi = lo + XQ_QU_W;
                for (int w = tid; w < XQ_QU_WW; w += nthr) s_bm[w] = 0u;
                __syncthreads();
                for (int idx = tid; idx < ml; idx += nthr) {
                    const int r = idx / L, i = idx - r * L;
                    const int v = s_sel[idx];
                    if (i < s_n[r] && s_slot[r] == sg && v >= lo && v < hi)
                        atomicOr(&s_bm[(v - lo) >> 5], 1u << ((v - lo) & 31));
                }
                __syncthreads();
                // exclusive popc prefix over the window's words, offset by N (contiguous per thread)
                const int per = (XQ_QU_WW + nthr - 1) / nthr;
                const int w0 = tid * per;
                int loc = 0;
                for (int k = 0; k < per; k++) if (w0 + k < XQ_QU_WW) loc += __popc(s_bm[w0 + k]);
                int inc = loc;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, inc, off);
                    if (lane >= off) inc += x;
                }
                if (lane == 31) s_red[warp] = inc;
                __syncthreads();
                int wbase = 0, tot = 0;
                for (int w = 0; w < nwarps; w++) { const int x = s_red[w]; if (w < warp) wbase += x; tot += x; }
                int run = N + wbase + inc - loc;
                for (int k = 0; k < per; k++) {
                    if (w0 + k < XQ_QU_WW) { s_wp[w0 + k] = run; run += __popc(s_bm[w0 + k]); }
                }
                __syncthreads();
                for (int idx = tid; idx < ml; idx += nthr) {
                    const int r = idx / L, i = idx - r * L;
                    const int v = s_sel[idx];
                    if (i < s_n[r] && s_slot[r] == sg && v >= lo && v < hi) {
                        const int d = v - lo;
                        const int u = s_wp[d >> 5] + __popc(s_bm[d >> 5] & ((1u << (d & 31)) - 1u));
                        s_u[u] = v;                                // equal writers write equal values
                        atomicOr(&s_msk[u], 1u << r);
                    }
                }
                N += tot;
                // next window: the group's smallest element >= hi (rows are strictly ascending);
                // none when the window reached the group's largest element (the common case)
                int nlo = 0x7fffffff;
                if (hi_all >= hi) for (int r = g; r < m; r++) {
                    if (s_slot[r] != sg) continue;
                    const int n = s_n[r];
                    int pos = 0;
                    #pragma unroll
                    for (int step = 128; step > 0; step >>= 1)
                        if (pos + step <= n && s_sel[r * L + pos + step - 1] < hi) pos += step;
                    if (pos < n) nlo = min(nlo, s_sel[r * L + pos]);
                }
                lo = nlo;
                __syncthreads();                                   // s_bm / s_wp / s_red reuse
            }
        }
    }
    __syncthreads();

    // ---- q and state: warp = row, 4 heads (chains) of this CTA's group
    const int r = warp;
    const int h0 = kvh * gqa + grp * XQ_QU_HPW;
    const float rsq = rsqrtf((float)hd);
    float qv[XQ_QU_HPW][8], acc[XQ_QU_HPW][8];
    #pragma unroll
    for (int c = 0; c < XQ_QU_HPW; c++) {
        const float* qrow = qstage + ((long long)r * nh + h0 + c) * hd;
        *reinterpret_cast<float4*>(qv[c]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv[c] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[c][i] = 0.0f;
    }
    float mx = -INFINITY, lx = 0.0f;                               // chain lane>>3 (lanes 8c..8c+7)
    const unsigned rbit = 1u << r;
    const bool b16 = (lane & 16) != 0, b8 = (lane & 8) != 0;

    const int nchunk = (N + XQ_QU_S - 1) / XQ_QU_S;
    #pragma unroll
    for (int c = 0; c < XQ_QU_RC - 1; c++) xq_qu_issue(ring, s_u, s_msk, s_slot, c, N, (const char*)kcvc, rb, mp, nkv, kvh, fmt);
    for (int c = 0; c < nchunk; c++) {
        // groups committed: (RC-1) + c; chunk c is group c -> at most RC-2 may pend
        asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_QU_RC - 2) : "memory");
        __syncthreads();                                           // chunk c visible; chunk c-1's slots free
        xq_qu_issue(ring, s_u, s_msk, s_slot, c + XQ_QU_RC - 1, N, (const char*)kcvc, rb, mp, nkv, kvh, fmt);
        if (!is_f32) {
            // decode chunk c ONCE per CTA (xq_kv_ld8 = the served reader) into the f32 lo/hi tile;
            // the previous chunk's tile readers finished before the barrier above
            for (int it = tid; it < XQ_QU_S * 64; it += nthr) {
                const int j = it >> 6, isv = (it >> 5) & 1, gl = it & 31;
                const int e = c * XQ_QU_S + j;
                if (e < N && s_u[e] >= 0) {
                    float o[8];
                    xq_kv_ld8(o, xq_qu_sm + (e % XQ_QU_NE) * slot_b + isv * (int)rb, fmt, hd, gl * 8);
                    float4* tt = reinterpret_cast<float4*>(xq_qu_sm + tile_o) + (j * 2 + isv) * 64;
                    tt[gl] = make_float4(o[0], o[1], o[2], o[3]);
                    tt[32 + gl] = make_float4(o[4], o[5], o[6], o[7]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < XQ_QU_S; j++) {
            const int e = c * XQ_QU_S + j;
            if (e >= N) break;                                     // block-uniform
            if (!(s_msk[e] & rbit)) continue;                      // warp-uniform: not this row's entry
            if (s_u[e] < 0) continue;                              // fallback -1 entry (5b skips it)
            const float4* kt = reinterpret_cast<const float4*>(
                xq_qu_sm + (is_f32 ? (e % XQ_QU_NE) * slot_b : tile_o + j * 2 * hd * 4));
            float kv[8];
            {
                const float4 a = kt[lane], b = kt[32 + lane];
                kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w;
                kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
            }
            float d[XQ_QU_HPW];
            #pragma unroll
            for (int ch = 0; ch < XQ_QU_HPW; ch++) {
                d[ch] = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) d[ch] = __fmaf_rn(qv[ch][i], kv[i], d[ch]);
            }
            // transposed xor tree (see the header): lane holds chain lane>>3's dot at the end
            const float k0 = b16 ? d[2] : d[0], k1 = b16 ? d[3] : d[1];
            const float s0 = b16 ? d[0] : d[2], s1 = b16 ? d[1] : d[3];
            const float e0 = __fadd_rn(k0, __shfl_xor_sync(FULL, s0, 16));
            const float e1 = __fadd_rn(k1, __shfl_xor_sync(FULL, s1, 16));
            float x = __fadd_rn(b8 ? e1 : e0, __shfl_xor_sync(FULL, b8 ? e0 : e1, 8));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 4));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 2));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 1));
            const float s = __fmul_rn(x, rsq);
            const float m_new = fmaxf(mx, s);
            const float scale_f = __expf(mx - m_new);
            const float wt = __expf(s - m_new);
            lx = __fmaf_rn(lx, scale_f, wt);
            mx = m_new;
            float vv[8];
            {
                const float4 a = kt[64 + lane], b = kt[96 + lane];
                vv[0] = a.x; vv[1] = a.y; vv[2] = a.z; vv[3] = a.w;
                vv[4] = b.x; vv[5] = b.y; vv[6] = b.z; vv[7] = b.w;
            }
            #pragma unroll
            for (int ch = 0; ch < XQ_QU_HPW; ch++) {
                const float sc = __shfl_sync(FULL, scale_f, ch * 8);
                const float w = __shfl_sync(FULL, wt, ch * 8);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[ch][i] = __fmaf_rn(acc[ch][i], sc, __fmul_rn(w, vv[i]));
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");          // only empty groups can pend
    #pragma unroll
    for (int ch = 0; ch < XQ_QU_HPW; ch++) {
        const long long hg = (long long)r * nh + h0 + ch;
        if (lane == ch * 8) {
            pm[hg * nsplits + sp] = mx;
            pl[hg * nsplits + sp] = lx;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[ch][i];
    }
}
// W4/QSA probe-only (--probe-exl3-binv sparse section): fill positions [p0, p0 + gridDim.x) of
// every (slot, K|V, kv head) plane (blockIdx.y = (slot*2 + kv)*nkv + h) with hashed pseudo-random
// values through the SERVED row writer xq_kv_put (every KV format; blockDim == hd), so a 128K
// synthetic context costs one launch and every position holds distinct content.
extern "C" __global__ void xq_qu_synth_kv(float* __restrict__ kcvc, int mpf, int hd, int nkv, int p0,
                                          unsigned seed) {
    XQ_PDL_ENTRY();
    const int t = p0 + (int)blockIdx.x;
    const int plane = (int)blockIdx.y;
    const int h = plane % nkv, kv = (plane / nkv) & 1, slot = plane / (2 * nkv);
    const XqKv kc = xq_kv_make(kcvc, mpf, hd, nkv, slot, h);
    unsigned x = seed ^ ((unsigned)plane * 0x9E3779B1u) ^ ((unsigned)t * 0x85EBCA6Bu) ^ (threadIdx.x * 0xC2B2AE35u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    const float v = ((float)(x & 0xFFFFFFu) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * (kv == 0 ? 1.5f : 1.0f);
    xq_kv_put((kv ? kc.v : kc.k) + (long long)t * kc.rb, kc.fmt, hd, threadIdx.x, v);
}

// ---- 5c. PQ8 item 3: the PREFILL gather for the 8-bit KV layouts (q8; the probe-only q4x shares its
// layout and reader). BIT-IDENTICAL partials to xq_attn_sel_splitk4 (GB10_PQ8_XCHECK diffs them):
// the same grid (rows*nkv*nsplits: sp = bid % nsplits, kvh = (bid/nsplits) % nkv, row =
// bid/(nsplits*nkv)), split_len and partial layout; entries ascending, -1 entries skipped; per head
// per entry splitk4's arithmetic as its PTX reads: dot = fma.rn chain from +0 over the lane's 8 dims
// ascending, xor tree 16..1 (dot + shfl), lane-0 broadcast, s = dot * rsqrtf(hd) with hd RUNTIME
// (MUFU.RSQ, as splitk4's blockDim.x), m_new = fmaxf(m, s), __expf both ways, l = fma(l, scale,
// wt), acc = fma(acc, scale, wt * v). K/V values: xq_kv_ld8's q8 expression on the same bytes.
// What changes is where the values come from. splitk4 (8 warps over gqa 12 heads, warps 0..3 walk
// their split twice) makes EVERY head-walk re-load its lane's 8 codes + scale and re-run the unpack
// (12x per entry per CTA), behind a per-entry dependent sel -> K/V load chain: PROBES_14 found the
// prefill gather latency/issue-bound (21% of 32K prefill kernel time; an 8-bit cache made it
// slower, fp8 +13.6%). Here:
//  - warp w owns heads w, w+4, w+8 (XQ_PQ8_HPW = 3 x 4 warps: gqa == 12 exactly, host-checked):
//    one walk, three independent head chains per warp (ILP), nothing walked twice;
//  - the split's selection entries are staged once in smem;
//  - the CTA unpacks each selected K and V row ONCE into an f32 smem tile of XQ_PQ8_TE entries,
//    stored lo/hi (dims 8L..8L+3 at float4 [L], 8L+4..8L+7 at float4 [32+L]) so each warp's
//    4 LDS.128 per entry are conflict-free;
//  - the NEXT tile's raw bytes (8 codes + the group's f16 scale per thread-item) are loaded into
//    registers before the current tile's math, so global latency hides behind a tile of compute.
// hd must be 256 (lane owns 8 dims; host-checked). split_hd = split_len | hd << 16.
#define XQ_PQ8_HPW 3
#define XQ_PQ8_NW 4
#define XQ_PQ8_TE 8
#define XQ_PQ8_CH 1024
#define XQ_PQ8_IPT ((XQ_PQ8_TE * 2 * 32) / (XQ_PQ8_NW * 32))   // raw items per thread per tile (4)
// q8 dims d0..d0+7 as raw words: the 8 codes + the 32-group's f16 scale bits (d0 % 8 == 0).
struct XqQ8Raw { uint2 u; unsigned short h; };
__device__ __forceinline__ void xq_q8_raw(XqQ8Raw& r, const char* row, int hd, int d0) {
    r.u = *reinterpret_cast<const uint2*>(row + d0);
    r.h = reinterpret_cast<const unsigned short*>(row + hd)[d0 >> 5];
}
// == xq_kv_ld8's q8 branch on the same bytes: sm = fl(f16(scale) * 2^-7), hs = 0.5 * sm, o = xq_q8v.
__device__ __forceinline__ void xq_q8_deq8(float* o, const XqQ8Raw& r) {
    const float sm = __fmul_rn(__half2float(__ushort_as_half(r.h)), 0.0078125f);
    const float hs = 0.5f * sm;
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = xq_q8v(r.u.x, i, sm, hs);
    #pragma unroll
    for (int i = 0; i < 4; i++) o[4 + i] = xq_q8v(r.u.y, i, sm, hs);
}
// q4k dims d0..d0+7 as raw bytes in the SAME register pair (u.x = the 4 code bytes, h = the
// 32-group's f16 scale bits at row + hd/2; d0 % 8 == 0). == xq_kv_ld8's q4k branch: sm =
// fl(f16(scale) * 2^-3), hs = 0.5 * sm, o = fma(code - 8, sm, hs).
__device__ __forceinline__ void xq_q4k_raw(XqQ8Raw& r, const char* row, int hd, int d0) {
    r.u.x = *reinterpret_cast<const unsigned*>(row + (d0 >> 1));
    r.h = reinterpret_cast<const unsigned short*>(row + hd / 2)[d0 >> 5];
}
__device__ __forceinline__ void xq_q4k_deq8(float* o, const XqQ8Raw& r) {
    const float sm = __fmul_rn(__half2float(__ushort_as_half(r.h)), 0.125f);
    const float hs = 0.5f * sm;
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = __fmaf_rn((float)((int)((r.u.x >> (4 * i)) & 0xF) - 8), sm, hs);
}
extern "C" __global__ void __launch_bounds__(XQ_PQ8_NW * 32, 4)
xq_attn_sel_pq8(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float4 tile[XQ_PQ8_TE][2][64];   // [entry][K|V][lo 0..31 | hi 32..63]
    __shared__ int s_t[XQ_PQ8_CH];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int split_len = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gqa = nh / nkv;
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    const float rsq = rsqrtf((float)hd);
    const bool is4 = kc.fmt == XQ_KV_Q4K;      // block-uniform: the packed 4-bit row (raw twin loaders)
    float qv[XQ_PQ8_HPW][8], acc[XQ_PQ8_HPW][8], mx[XQ_PQ8_HPW], lx[XQ_PQ8_HPW];
    #pragma unroll
    for (int g = 0; g < XQ_PQ8_HPW; g++) {
        mx[g] = -INFINITY; lx[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[g][i] = 0.0f; qv[g][i] = 0.0f; }
        const int w = warp + g * XQ_PQ8_NW;
        const float* qrow = qstage + ((long long)row * nh + kvh * gqa + w) * hd;
        *reinterpret_cast<float4*>(qv[g]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv[g] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    // thread item k of a tile: row-slot rs = warp + NW*k -> entry rs >> 1, K|V = rs & 1, dims lane*8..+7
    // (a warp reads one row's 256 contiguous code bytes + its 16 scale bytes per item: coalesced;
    //  q4k: the 128 code bytes of the packed row + the same 16 scale bytes).
    XqQ8Raw raw[XQ_PQ8_IPT];
    for (int cb = begin; cb < end; cb += XQ_PQ8_CH) {
        const int n = min(XQ_PQ8_CH, end - cb);
        __syncthreads();                                   // previous chunk's s_t / tile readers done
        for (int i = tid; i < n; i += XQ_PQ8_NW * 32) s_t[i] = sel[(long long)row * sel_max + cb + i];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < XQ_PQ8_IPT; k++) {             // prefetch tile 0 (warp-uniform branches)
            const int rs = warp + XQ_PQ8_NW * k, e = rs >> 1;
            const int t = e < n ? s_t[e] : -1;
            if (t >= 0) {
                const char* row = ((rs & 1) ? kc.v : kc.k) + (long long)t * kc.rb;
                if (is4) xq_q4k_raw(raw[k], row, hd, lane * 8);
                else     xq_q8_raw(raw[k], row, hd, lane * 8);
            }
        }
        for (int e0 = 0; e0 < n; e0 += XQ_PQ8_TE) {
            const int ne = min(XQ_PQ8_TE, n - e0);
            __syncthreads();                               // previous tile's readers done
            #pragma unroll
            for (int k = 0; k < XQ_PQ8_IPT; k++) {         // unpack ONCE per CTA into the f32 tile
                const int rs = warp + XQ_PQ8_NW * k, e = rs >> 1;
                if (e < ne && s_t[e0 + e] >= 0) {
                    float o[8];
                    if (is4) xq_q4k_deq8(o, raw[k]);
                    else     xq_q8_deq8(o, raw[k]);
                    tile[e][rs & 1][lane] = make_float4(o[0], o[1], o[2], o[3]);
                    tile[e][rs & 1][32 + lane] = make_float4(o[4], o[5], o[6], o[7]);
                }
            }
            __syncthreads();
            if (e0 + XQ_PQ8_TE < n) {                      // next tile's bytes in flight during the math
                #pragma unroll
                for (int k = 0; k < XQ_PQ8_IPT; k++) {
                    const int rs = warp + XQ_PQ8_NW * k, e = e0 + XQ_PQ8_TE + (rs >> 1);
                    const int t = e < n ? s_t[e] : -1;
                    if (t >= 0) {
                        const char* row = ((rs & 1) ? kc.v : kc.k) + (long long)t * kc.rb;
                        if (is4) xq_q4k_raw(raw[k], row, hd, lane * 8);
                        else     xq_q8_raw(raw[k], row, hd, lane * 8);
                    }
                }
            }
            for (int e = 0; e < ne; e++) {
                if (s_t[e0 + e] < 0) continue;             // block-uniform (-1 padding)
                float kv[8], vv[8];
                {
                    const float4 a = tile[e][0][lane], b = tile[e][0][32 + lane];
                    kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w;
                    kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
                    const float4 c = tile[e][1][lane], d = tile[e][1][32 + lane];
                    vv[0] = c.x; vv[1] = c.y; vv[2] = c.z; vv[3] = c.w;
                    vv[4] = d.x; vv[5] = d.y; vv[6] = d.z; vv[7] = d.w;
                }
                // the three heads' chains stage by stage (independent: ILP 3); each head's own op
                // sequence is splitk4's, in splitk4's order. gqa == XQ_PQ8_NW*XQ_PQ8_HPW (host).
                float dot[XQ_PQ8_HPW];
                #pragma unroll
                for (int g = 0; g < XQ_PQ8_HPW; g++) {
                    dot[g] = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) dot[g] = __fmaf_rn(qv[g][i], kv[i], dot[g]);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int g = 0; g < XQ_PQ8_HPW; g++)
                        dot[g] = __fadd_rn(dot[g], __shfl_xor_sync(0xffffffffu, dot[g], off));
                }
                #pragma unroll
                for (int g = 0; g < XQ_PQ8_HPW; g++) {
                    const float s = __fmul_rn(__shfl_sync(0xffffffffu, dot[g], 0), rsq);
                    const float m_new = fmaxf(mx[g], s);
                    const float scale_f = __expf(mx[g] - m_new);
                    const float wt = __expf(s - m_new);
                    lx[g] = __fmaf_rn(lx[g], scale_f, wt);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) acc[g][i] = __fmaf_rn(acc[g][i], scale_f, __fmul_rn(wt, vv[i]));
                    mx[g] = m_new;
                }
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < XQ_PQ8_HPW; g++) {
        const int w = warp + g * XQ_PQ8_NW;
        const long long hg = (long long)row * nh + kvh * gqa + w;
        if (lane == 0) {
            pm[hg * nsplits + sp] = mx[g];
            pl[hg * nsplits + sp] = lx[g];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[g][i];
    }
}

