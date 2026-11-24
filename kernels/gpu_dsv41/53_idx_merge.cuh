// =============================================================================
// kernels/gpu_dsv41/53_idx_merge.cuh — leg KI: merge the per-chunk local top-512 lists of a row into
// its final selection (the same code merges per-rank key stripes under TP: T0 §3.8 / §11 KI).
//
// Each list is sorted descending by the total-order key (score desc, position asc); keys are unique,
// so the top 512 of the union is unique and equals the unsplit selection bit for bit. A lower bound of
// the final 512th key from the lists' heads first; then lists are consumed in batches: per list only
// its prefix above the running 512th key can matter (binary search); a batch plus the running set
// goes through a radix select of the 512th key (no sorts). Output: ascending positions, -1 padded (d41_idx_write_final).
// Used by the score kernel's last chunk CTA (fused, `tick`) and by d41_idx_merge (TP stripes).
// Lists written by other CTAs are read with ld.global.cg (L2), never through a possibly stale L1.
// =============================================================================
#pragma once

__device__ __forceinline__ int d41_idx_tiles_any(const d41_idx_args& a, int b) {
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    if (a.mode == 1) return (((const int*)a.ucnt)[b] + 15) / 16;
    int nmax = 0;
    for (int r = 0; r < rb.nrows && r < a.R; r++) {
        int n = d41_idx_nvis(((const int*)a.pos)[rb.row0 + r], a.ratio);
        if (n > a.n_keys_cap) n = a.n_keys_cap;
        nmax = n > nmax ? n : nmax;
    }
    return (nmax + D41_IDX_TILE - 1) / D41_IDX_TILE;
}

// Bitonic sort, runtime power-of-two n, descending.
__device__ __forceinline__ void d41_idx_sort_desc_n(unsigned long long* v, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
                int lo = ((i & ~(j - 1)) << 1) | (i & (j - 1)), hi = lo | j;
                unsigned long long x = v[lo], y = v[hi];
                bool desc = (lo & k) == 0;
                if ((x < y) == desc) { v[lo] = y; v[hi] = x; }
            }
            __syncthreads();
        }
    }
}

// Up to this many chunks the chunk lists stay unsorted and the merge radix-selects over all of them.
#define D41_IDX_SMALL_G 8
// Shared scratch of one row merge (~37 KB; the score kernel's idle buffers hold it).
struct d41_idx_msm {
    unsigned long long S[D41_IDX_TOPK];        // running selection (unsorted)
    unsigned long long T[D41_IDX_SMALL_G * D41_IDX_TOPK];   // list prefixes + S, or every list (small G)
    int sc[256 + 16];                          // radix-select scratch (8-B aligned: follows u64 arrays)
    unsigned long long thr;
    int lens[32], offs[33], take, cnt;
};

__device__ __forceinline__ int d41_ldcg_i(const int* p) { return __ldcg(p); }
__device__ __forceinline__ unsigned long long d41_ldcg_u64(const unsigned long long* p) { return __ldcg(p); }

// Merge `geff` lists of row `row` (lists at part[li0 + c], each sorted descending, counts at
// pcnt[li0 + c]) and write the final selection. Whole CTA of NT threads.
template <int NT>
__device__ __forceinline__ void d41_idx_merge_row(const d41_idx_args& a, int row, long long li0, int geff, d41_idx_msm* m) {
    const int tid = threadIdx.x;
    const int* pcnt = (const int*)a.pcnt;
    const unsigned long long* part = (const unsigned long long*)a.part;
    int nvis = d41_idx_nvis(((const int*)a.pos)[row], a.ratio);
    if (nvis > a.n_keys_cap) nvis = a.n_keys_cap;
    if (tid == 0) { m->cnt = 0; m->thr = 0ull; }
    if (geff <= D41_IDX_SMALL_G) {   // unsorted lists: one radix select over all of them
        if (tid == 0) { int o = 0; for (int c = 0; c < geff; c++) { m->offs[c] = o; o += d41_ldcg_i(pcnt + li0 + c); } m->offs[geff] = o; }
        __syncthreads();
        const int u = m->offs[geff];
        for (int i = tid; i < geff * D41_IDX_TOPK; i += NT) {
            const int c = i / D41_IDX_TOPK, k = i % D41_IDX_TOPK;
            if (k < m->offs[c + 1] - m->offs[c]) m->T[m->offs[c] + k] = d41_ldcg_u64(part + (li0 + c) * D41_IDX_TOPK + k);
        }
        __syncthreads();
        int n = u;
        if (u > D41_IDX_TOPK) {
            const unsigned long long k512 = d41_idx_kth(m->T, u, D41_IDX_TOPK, m->sc);
            d41_idx_keep_ge<(D41_IDX_SMALL_G * D41_IDX_TOPK + NT - 1) / NT>(m->T, u, k512, m->sc);
            n = D41_IDX_TOPK;
        }
        d41_idx_write_final(a, row, m->T, n, (unsigned*)(m->T + D41_IDX_TOPK), (D41_IDX_SMALL_G - 1) * D41_IDX_TOPK * 2, nvis);
        return;
    }
    __syncthreads();
    if (geff < 2 * D41_IDX_TOPK) {   // lower bound from the heads: the 513th largest of every list's top hk
        unsigned* hoff = (unsigned*)m->S;
        const int hk = max(1, (2 * D41_IDX_TOPK) / geff);
        if (tid == 0) {
            int o = 0;
            for (int c = 0; c < geff; c++) { hoff[c] = o; o += min(hk, d41_ldcg_i(pcnt + li0 + c)); }
            hoff[geff] = o;
        }
        __syncthreads();
        const int tot = (int)hoff[geff];
        if (tot > D41_IDX_TOPK) {
            for (int i = tid; i < geff * hk; i += NT) {
                const int c = i / hk, k = i % hk;
                if (k < (int)(hoff[c + 1] - hoff[c])) m->T[hoff[c] + k] = d41_ldcg_u64(part + (li0 + c) * D41_IDX_TOPK + k);
            }
            __syncthreads();
            const unsigned long long t0 = d41_idx_kth(m->T, tot, D41_IDX_TOPK + 1, m->sc);
            if (tid == 0) m->thr = t0;
        }
        __syncthreads();
    }
    for (int c0 = 0; c0 < geff;) {
        const int nl = min(32, geff - c0);
        if (tid < 32) {
            int mm = 0;
            if (tid < nl) {
                const long long li = li0 + c0 + tid;
                const int n = d41_ldcg_i(pcnt + li);
                const unsigned long long* v = part + li * D41_IDX_TOPK;
                const unsigned long long th = m->thr;
                int lo = 0, hi = n;   // first index with v <= th (v is descending)
                while (lo < hi) { int mid = (lo + hi) >> 1; if (d41_ldcg_u64(v + mid) > th) lo = mid + 1; else hi = mid; }
                mm = lo;
            }
            m->lens[tid] = mm;
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, k = 0;
            m->offs[0] = 0;
            while (k < nl && acc + m->lens[k] <= 2 * D41_IDX_TOPK) { acc += m->lens[k]; m->offs[++k] = acc; }
            m->take = k;
        }
        __syncthreads();
        const int take = m->take, total = m->offs[take], na = m->cnt;
        for (int i = tid; i < total; i += NT) {
            int k = 0;
            while (m->offs[k + 1] <= i) k++;
            m->T[i] = d41_ldcg_u64(part + (li0 + c0 + k) * D41_IDX_TOPK + (i - m->offs[k]));
        }
        for (int i = tid; i < na; i += NT) m->T[total + i] = m->S[i];
        __syncthreads();
        const int u = total + na;
        if (u > D41_IDX_TOPK) {
            const unsigned long long k512 = d41_idx_kth(m->T, u, D41_IDX_TOPK, m->sc);
            d41_idx_keep_ge<(3 * D41_IDX_TOPK + NT - 1) / NT>(m->T, u, k512, m->sc);   // u <= 1536
            for (int i = tid; i < D41_IDX_TOPK; i += NT) m->S[i] = m->T[i];
            if (tid == 0) { m->cnt = D41_IDX_TOPK; m->thr = k512; }
        } else {
            for (int i = tid; i < u; i += NT) m->S[i] = m->T[i];
            if (tid == 0) m->cnt = u;
        }
        __syncthreads();
        c0 += take;
    }
    d41_idx_write_final(a, row, m->S, m->cnt, (unsigned*)m->T, D41_IDX_SMALL_G * D41_IDX_TOPK * 2, nvis);
}

// Stand-alone merge (TP key stripes: `lists` > 0 lists per row; or the unfused chunked path).
extern "C" __global__ void __launch_bounds__(512) d41_idx_merge(const d41_idx_args a) {
    D41_PDL_ENTRY();
    __shared__ d41_idx_msm m;
    const int b = blockIdx.x / a.R, r = blockIdx.x % a.R;
    if (b >= a.n_rb) return;
    const d41_idx_rb rb = ((const d41_idx_rb*)a.rbs)[b];
    if (r >= rb.nrows) return;
    int geff, stride;
    if (a.lists > 0) { geff = a.lists; stride = a.lists; }     // explicit lists (TP key stripes)
    else {
        geff = d41_idx_geff(d41_idx_tiles_any(a, b), a.G, a.min_tiles);
        if (geff <= 1 || a.tick) return;   // the score kernel finished this row itself
        stride = a.G;
    }
    d41_idx_merge_row<512>(a, rb.row0 + r, ((long long)b * a.R + r) * stride, geff, &m);
}
