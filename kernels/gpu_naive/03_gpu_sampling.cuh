// gpu_naive/03_gpu_sampling.cuh: GPU sampling head: splitmix RNG, the multi-block exact sampler (prep/stats/gumbel/pick/mass/accept) and its radix top-k fallback.
// The sample is EXACT: rejection sampling from the full softmax equals sampling from the renormalized nucleus.
// Section 3 of 3 of gpu_naive.cu: included in order by gpu_naive.cu, one translation unit; text is moved verbatim.
// =============================================================================
// S6: GPU sampling (the head). sp = [T, top_p, top_k, min_p, rep, pres, freq, _] (device f32).
// =============================================================================
__device__ __forceinline__ unsigned long long nv_splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// =============================================================================
// S6 F8: the fast exact sampler (multi-block, ~20 small kernels, ~0.1 ms) replacing the single-block
// radix top-k over the vocabulary (0.68 ms). Blocks of 1024 vocabulary entries (SMP_B = ceil(V/1024)).
//   prep:   scaled + penalized logits -> out[V]; per-block max -> bmax[b]
//   stats:  stats[0] = max(bmax)                      z: per-block sum exp(out - max) -> bz;  stats[1] = sum(bz)
//   try t (unrolled 3x in the graph, each gated by acc[0] == 0):
//     gumbel: per-block argmax of out[i] + g_i (g_i = -log(-log(u_i)), u_i = splitmix(seed, step, t, i));
//             T <= 0: g = 0 (the plain argmax; ties -> lower index)  -> bval[b], bidx[b]
//     pick:   c = argmax over the blocks (ties -> lower index)       -> cand[0]
//     mass:   per-block sum of p_i over i with out[i] > out[c], and the count of them -> bmass[b], bcnt[b]
//     accept: nucleus test: mass < top_p (strict), count < top_k (if top_k > 0), p_c >= min_p * p_max;
//             T <= 0 accepts unconditionally. Accepted -> tok[0] = c, acc[0] = 1.
//   fallback (acc[0] still 0 after the tries: a probability <= (1 - nucleus mass)^3, or a tight top-k):
//     the exact single-block radix top-k + nv_sample_cands, both exiting at once when acc[0] == 1.
//   note:   counts[tok] += 1; rng[1] += 1 (the step).
// The sample is EXACT: rejection sampling from the full softmax, accepting iff the draw lies in the
// nucleus = sampling from the renormalized nucleus.
// =============================================================================
#define SMP_T 1024
// Distributed form (S6c, vocab-parallel head): this rank holds the logits of the vocabulary slice
// [v0, v0 + Vl). Every cross-rank value travels in the exchange buffer xb as ZERO-PADDED SLOTS: a
// rank writes its own slot and zeros the others, the fp32 sum all-reduce (exact: adding zeros) leaves
// every rank with every slot = an all-gather on the doorbell transport. World 1: xb holds the one
// slot and no reduce runs. Slot layouts (W = world, r = rank):
//   XB_STATS  [4r + 0..3] = (local max, local Z relative to the local max, cancel flag (rank 0), 0)
//   XB_TRY(t) [4r + 0..3] = (gumbel argmax value, its GLOBAL index, its scaled logit s, 0)
//   XB_MASS(t)[4r + 0..3] = (mass above the candidate, count above, 0, 0)
//   XB_CAND   [2 * (r * CANDL + j) + 0..1] = (s, global index) of the rank's exact top-CANDL (the fallback)
#define XB_STATS 0
#define XB_TRY(t) (16 + 16 * (t))
#define XB_MASS(t) (64 + 16 * (t))
#define XB_CAND 128
#define XB_FLOATS (XB_CAND + 2 * 1024)
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_prep(float* __restrict__ out, float* __restrict__ bmax, const __half* __restrict__ logits,
            const int* __restrict__ counts, const unsigned char* __restrict__ seen, const float* __restrict__ sp, int Vl, int v0) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    const float T = sp[0], rep = sp[4], pres = sp[5], freq = sp[6];
    const float inv_t = (T > 0.f) ? 1.f / T : 1.f;
    float l = -INFINITY;
    if (v < Vl) {
        l = nv_h2f(logits[v]);
        const int c = counts[v0 + v];
        if (rep != 1.f && rep > 0.f && (seen[v0 + v] || c > 0)) l = (l > 0.f) ? l / rep : l * rep;
        if (c > 0) l -= pres + freq * (float)c;
        l *= inv_t;
        out[v] = l;
    }
    red[tid] = l; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) bmax[blockIdx.x] = red[0];
}
// local max -> lmax[0]
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_stats(const float* __restrict__ bmax, float* __restrict__ lmax, int nb) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x;
    float m = -INFINITY;
    for (int b = tid; b < nb; b += SMP_T) m = fmaxf(m, bmax[b]);
    red[tid] = m; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) lmax[0] = red[0];
}
// per-block sum exp(out - local max) -> bz
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_z(const float* __restrict__ out, const float* __restrict__ lmax, float* __restrict__ bz, int Vl) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    red[tid] = (v < Vl) ? __expf(out[v] - lmax[0]) : 0.f; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) bz[blockIdx.x] = red[0];
}
// the STATS slot: (local max, local Z, cancel flag from rank 0), the other slots zeroed; acc = 0
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_z2(const float* __restrict__ bz, const float* __restrict__ lmax, int nb, float* __restrict__ xb, int rank, int world,
          const int* __restrict__ cancel, int* __restrict__ acc) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x;
    float z = 0.f;
    for (int b = tid; b < nb; b += SMP_T) z += bz[b];
    red[tid] = z; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid < 4 * world) xb[XB_STATS + tid] = 0.f;
    __syncthreads();
    if (tid == 0) {
        xb[XB_STATS + 4 * rank + 0] = lmax[0];
        xb[XB_STATS + 4 * rank + 1] = red[0];
        xb[XB_STATS + 4 * rank + 2] = (rank == 0) ? (float)cancel[0] : 0.f;
        acc[0] = 0;
    }
}
// after the STATS exchange: the global max and Z (Z = sum_r Z_r * exp(max_r - max)), the cancel flag
__device__ __forceinline__ void nv_smp_global_stats(const float* __restrict__ xb, int world, float& mx, float& z) {
    mx = -INFINITY;
    for (int r = 0; r < world; r++) mx = fmaxf(mx, xb[XB_STATS + 4 * r]);
    z = 0.f;
    for (int r = 0; r < world; r++) z += xb[XB_STATS + 4 * r + 1] * __expf(xb[XB_STATS + 4 * r] - mx);
}
__device__ __forceinline__ float nv_gumbel(unsigned long long seed, unsigned long long step, int t, int i) {
    const unsigned long long r = nv_splitmix(seed ^ (step * 0x9E3779B97F4A7C15ull) ^ ((unsigned long long)(t + 1) << 56) ^ ((unsigned long long)i * 0xD1B54A32D192ED03ull));
    const float u = ((float)(r >> 40) + 0.5f) * (1.f / 16777216.f);   // (0, 1)
    return -__logf(-__logf(u));
}
// per-block argmax of out[i] + gumbel(global index) (T <= 0: plain argmax). Ties -> lower index.
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_gumbel(const float* __restrict__ out, const float* __restrict__ sp, const unsigned long long* __restrict__ rng,
              int t, float* __restrict__ bval, int* __restrict__ bidx, int Vl, int v0, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float rv[SMP_T];
    __shared__ int ri[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    float val = -INFINITY;
    if (v < Vl) { val = out[v]; if (sp[0] > 0.f) val += nv_gumbel(rng[0], rng[1], t, v0 + v); }
    rv[tid] = val; ri[tid] = v0 + v; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) {
        if (tid < o) {
            const float a = rv[tid], b = rv[tid + o];
            if (b > a || (b == a && ri[tid + o] < ri[tid])) { rv[tid] = b; ri[tid] = ri[tid + o]; }
        }
        __syncthreads();
    }
    if (tid == 0) { bval[blockIdx.x] = rv[0]; bidx[blockIdx.x] = ri[0]; }
}
// the rank's candidate -> its TRY slot (value, global index, s), other slots zeroed
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_pick(const float* __restrict__ bval, const int* __restrict__ bidx, int nb, const float* __restrict__ out, int v0,
            float* __restrict__ xb, int t, int rank, int world, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float rv[SMP_T];
    __shared__ int ri[SMP_T];
    const int tid = threadIdx.x;
    float val = -INFINITY; int idx = 0x7fffffff;
    for (int b = tid; b < nb; b += SMP_T) { const float x = bval[b]; const int i = bidx[b]; if (x > val || (x == val && i < idx)) { val = x; idx = i; } }
    rv[tid] = val; ri[tid] = idx; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) {
        if (tid < o) {
            const float a = rv[tid], b = rv[tid + o];
            if (b > a || (b == a && ri[tid + o] < ri[tid])) { rv[tid] = b; ri[tid] = ri[tid + o]; }
        }
        __syncthreads();
    }
    if (tid < 4 * world) xb[XB_TRY(t) + tid] = 0.f;
    __syncthreads();
    if (tid == 0) {
        xb[XB_TRY(t) + 4 * rank + 0] = rv[0];
        xb[XB_TRY(t) + 4 * rank + 1] = (float)ri[0];
        xb[XB_TRY(t) + 4 * rank + 2] = out[ri[0] - v0];
    }
}
// after the TRY exchange: the global candidate (max value, ties -> lower index) and its s
__device__ __forceinline__ void nv_smp_global_cand(const float* __restrict__ xb, int t, int world, int& ci, float& sc) {
    float best = -INFINITY; ci = 0x7fffffff; sc = 0.f;
    for (int r = 0; r < world; r++) {
        const float v = xb[XB_TRY(t) + 4 * r];
        const int i = (int)xb[XB_TRY(t) + 4 * r + 1];
        if (v > best || (v == best && i < ci)) { best = v; ci = i; sc = xb[XB_TRY(t) + 4 * r + 2]; }
    }
}
// per-block: mass and count of the LOCAL entries strictly above the candidate's s
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_mass(const float* __restrict__ out, const float* __restrict__ xb, int t, int world,
            float* __restrict__ bmass, int* __restrict__ bcnt, int Vl, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float red[SMP_T];
    __shared__ int cnt[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    int ci; float sc; nv_smp_global_cand(xb, t, world, ci, sc);
    float m = 0.f; int c = 0;
    if (v < Vl && out[v] > sc) { m = __expf(out[v] - mx) / z; c = 1; }
    red[tid] = m; cnt[tid] = c; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) { red[tid] += red[tid + o]; cnt[tid] += cnt[tid + o]; } __syncthreads(); }
    if (tid == 0) { bmass[blockIdx.x] = red[0]; bcnt[blockIdx.x] = (float)cnt[0] == 0.f ? 0 : cnt[0]; bcnt[blockIdx.x] = cnt[0]; }
}
// the rank's (mass, count) -> its MASS slot
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_mass2(const float* __restrict__ bmass, const int* __restrict__ bcnt, int nb, float* __restrict__ xb, int t, int rank, int world,
             const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float red[SMP_T];
    __shared__ int cnt[SMP_T];
    const int tid = threadIdx.x;
    float m = 0.f; int c = 0;
    for (int b = tid; b < nb; b += SMP_T) { m += bmass[b]; c += bcnt[b]; }
    red[tid] = m; cnt[tid] = c; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) { red[tid] += red[tid + o]; cnt[tid] += cnt[tid + o]; } __syncthreads(); }
    if (tid < 4 * world) xb[XB_MASS(t) + tid] = 0.f;
    __syncthreads();
    if (tid == 0) { xb[XB_MASS(t) + 4 * rank + 0] = red[0]; xb[XB_MASS(t) + 4 * rank + 1] = (float)cnt[0]; }
}
// after the MASS exchange: the nucleus test, identical on every rank
extern "C" __global__ void nv_smp_accept(const float* __restrict__ xb, int t, int world, const float* __restrict__ sp,
                                         int* __restrict__ tok, int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (threadIdx.x != 0 || acc[0]) return;
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    int ci; float sc; nv_smp_global_cand(xb, t, world, ci, sc);
    float mass = 0.f; int cnt = 0;
    for (int r = 0; r < world; r++) { mass += xb[XB_MASS(t) + 4 * r]; cnt += (int)xb[XB_MASS(t) + 4 * r + 1]; }
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    bool ok = true;
    if (T > 0.f) {
        if (top_p > 0.f && top_p < 1.f && mass >= top_p) ok = false;
        if (top_k > 0 && cnt >= top_k) ok = false;
        if (min_p > 0.f) { const float pc = __expf(sc - mx) / z; if (pc < min_p / z) ok = false; }
    }
    if (ok) { tok[0] = ci; acc[0] = 1; }
}
// the exact fallback, distributed: this rank's top-CANDL (sel over the local slice, from nv_topk) ->
// (s, global index) pairs in its CAND slot (other slots zeroed; nsel < CANDL pads with -inf)
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_cands_out(const int* __restrict__ sel, const int* __restrict__ nsel, const float* __restrict__ out, int v0,
                 float* __restrict__ xb, int candl, int rank, int world) {
    NV_PDL_ENTRY();
    const int tid = threadIdx.x;
    for (int j = tid; j < candl * world; j += SMP_T) { xb[XB_CAND + 2 * j] = 0.f; xb[XB_CAND + 2 * j + 1] = 0.f; }
    __syncthreads();
    const int n = *nsel;
    for (int j = tid; j < candl; j += SMP_T) {
        const int slot = XB_CAND + 2 * (rank * candl + j);
        if (j < n) { const int i = sel[j]; xb[slot] = out[i]; xb[slot + 1] = (float)(v0 + i); }
        else { xb[slot] = -INFINITY; xb[slot + 1] = -1.f; }
    }
}
// after the CAND exchange: sort the gathered candidates (p desc, index asc), the cuts, one draw —
// identical on every rank. Only when no Gumbel try was accepted.
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_sample_cands(int* __restrict__ tok, const float* __restrict__ xb, int world, int candl, const float* __restrict__ sp,
                    const unsigned long long* __restrict__ rng, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float sp_v[1024];
    __shared__ int sp_i[1024];
    __shared__ float red[1024];
    const int tid = threadIdx.x;
    const int n = candl * world;    // <= 1024
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    if (tid < n) {
        const float s = xb[XB_CAND + 2 * tid]; const int i = (int)xb[XB_CAND + 2 * tid + 1];
        sp_i[tid] = (i < 0) ? 0x7fffffff : i;
        sp_v[tid] = (i < 0) ? -INFINITY : ((T > 0.f) ? __expf(s - mx) / z : s);
    } else { sp_i[tid] = 0x7fffffff; sp_v[tid] = -INFINITY; }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const float a = sp_v[tid], b = sp_v[ixj];
                const int ia = sp_i[tid], ib = sp_i[ixj];
                const bool a_first = (a > b) || (a == b && ia < ib);
                const bool up = ((tid & k) == 0);
                if (up != a_first) { sp_v[tid] = b; sp_v[ixj] = a; sp_i[tid] = ib; sp_i[ixj] = ia; }
            }
            __syncthreads();
        }
    }
    int live = 0;
    for (int i = 0; i < 1024; i++) { if (sp_v[i] == -INFINITY) break; live++; }   // (sorted: the -inf pads are last)
    if (T <= 0.f) { if (tid == 0) tok[0] = sp_i[0]; return; }
    int keep = live;
    if (top_k > 0) keep = min(keep, top_k);
    if (min_p > 0.f) {
        const float floor_p = sp_v[0] * min_p;
        red[tid] = (tid < keep && sp_v[tid] >= floor_p) ? 1.f : 0.f; __syncthreads();
        for (int o = 512; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
        keep = max(1, (int)red[0]); __syncthreads();
    }
    red[tid] = (tid < keep) ? sp_v[tid] : 0.f; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (top_p > 0.f && top_p < 1.f) {
        __shared__ int s_cut;
        if (tid == 0) s_cut = keep;
        __syncthreads();
        if (tid < keep && red[tid] >= top_p && (tid == 0 || red[tid - 1] < top_p)) s_cut = tid + 1;
        __syncthreads();
        keep = min(keep, s_cut);
    }
    if (tid == 0) {
        const float total = red[keep - 1];
        const unsigned long long r = nv_splitmix(rng[0] ^ (rng[1] * 0x9E3779B97F4A7C15ull));
        const float u = (float)(r >> 40) * (1.f / 16777216.f) * total;
        int pick = keep - 1;
        for (int i = 0; i < keep; i++) { if (u < red[i]) { pick = i; break; } }
        tok[0] = sp_i[pick];
    }
}
// ---- exact speculative acceptance (work item A of PLAN/NAIVE_DRAFT2_PROMPT_2026-10-19.md), for
// SAMPLED drafts. ONE BLOCK PER DRAFT ROW j (grid k); no exchange, so the head may run it while the
// node ranks skip it (their doorbell epochs are unaffected).
//
//   p_trunc  = the SAME distribution nv_smp_sample_cands draws from: the sorted all-gathered
//              candidates, cut by top_k / min_p / top_p and renormalised by the nucleus mass.
//   q(v)     = exp(qs[j][v] - qstat[2j]) / qstat[2j+1]   (the head's full-vocabulary drafter logits,
//              already scaled by 1/T, for draft row j)
//
// rule (exact speculative sampling, Leviathan 2023 / Chen 2023):
//   u < min(1, p(d)/q(d))  ->  accept d, the block continues
//   otherwise              ->  commit one draw from norm(max(0, p - q)), the block STOPS
//
// out: acc[j] = 1/0, rtok[j] = the residual draw (meaningful only when rejected). `on[0]` == 0 (the
// greedy path, or a node rank) makes the whole kernel a no-op so the greedy token stream stays
// bit-identical and the graph layout is shared.
// grid (1) per draft row, block 1024; launched inside the verify row loop.
extern "C" __global__ void __launch_bounds__(1024)
nv_spec_accept_rows(const int* __restrict__ on, const float* __restrict__ xb, int world, int n,
                    const float* __restrict__ sp, int row, const int* __restrict__ draft, const float* __restrict__ qs,
                    const float* __restrict__ qstat, const unsigned long long* __restrict__ rngp,
                    int* __restrict__ acc, int* __restrict__ rtok) {
    NV_PDL_ENTRY();
    if (!on[0]) return;
    // ONE launch per row, inside the row loop: `xb` holds THAT row's gathered candidate list and
    // XB_STATS that row's (max, z), both overwritten by the next row. The accept draw's seed/step are
    // read from device memory because the graph bakes kernel ARGUMENTS at capture.
    const int j = row;
    const unsigned long long rng = rngp[0];
    const int step = (int)rngp[1];
    // rngp[2] = 1: the drafts are argmax draws (F = 0 / the sharded drafter): q is ONE-HOT at the draft, so the
    // test is u < p(d) and the residual is p without d (no q buffer is written in that mode)
    const int onehot = (int)rngp[2];
    const int d = draft[j];
    __shared__ float sp_v[1024];
    __shared__ int sp_i[1024];
    __shared__ float red[1024];
    __shared__ int s_cut;
    const int tid = threadIdx.x;
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    if (tid < n) {
        const float s = xb[XB_CAND + 2 * tid]; const int i = (int)xb[XB_CAND + 2 * tid + 1];
        sp_i[tid] = (i < 0) ? 0x7fffffff : i;
        sp_v[tid] = (i < 0) ? -INFINITY : ((T > 0.f) ? __expf(s - mx) / z : s);
    } else { sp_i[tid] = 0x7fffffff; sp_v[tid] = -INFINITY; }
    __syncthreads();
    // p descending, index ascending: the order nv_smp_sample_cands samples in
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            const int ixj = tid ^ jj;
            if (ixj > tid) {
                const float a = sp_v[tid], b = sp_v[ixj];
                const int ia = sp_i[tid], ib = sp_i[ixj];
                const bool a_first = (a > b) || (a == b && ia < ib);
                const bool up = ((tid & k) == 0);
                if (up != a_first) { sp_v[tid] = b; sp_v[ixj] = a; sp_i[tid] = ib; sp_i[ixj] = ia; }
            }
            __syncthreads();
        }
    }
    int live = 0;
    for (int i = 0; i < 1024; i++) { if (sp_v[i] == -INFINITY) break; live++; }
    if (T <= 0.f) return;                       // greedy never reaches this kernel (guard kept explicit)
    int keep = live;
    if (top_k > 0) keep = min(keep, top_k);
    if (min_p > 0.f) {
        const float floor_p = sp_v[0] * min_p;
        red[tid] = (tid < keep && sp_v[tid] >= floor_p) ? 1.f : 0.f; __syncthreads();
        for (int o = 512; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
        keep = max(1, (int)red[0]); __syncthreads();
    }
    red[tid] = (tid < keep) ? sp_v[tid] : 0.f; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (top_p > 0.f && top_p < 1.f) {
        if (tid == 0) s_cut = keep;
        __syncthreads();
        if (tid < keep && red[tid] >= top_p && (tid == 0 || red[tid - 1] < top_p)) s_cut = tid + 1;
        __syncthreads();
        keep = min(keep, s_cut);
    }
    const float total = (keep > 0) ? red[keep - 1] : 0.f;
    // No candidate list (the gather did not run) or no mass: nothing to test against. Reject the row
    // and hand the decision back (rtok = -1 -> the host commits the target's own draw for the row and
    // counts the fallback) instead of indexing the list at -1.
    if (keep <= 0 || total <= 0.f) { if (tid == 0) { acc[j] = 0; rtok[j] = -1; } return; }   // -1: no candidate list
    // the accept test, and the residual weights r(i) = max(0, p_trunc(i) - q(id_i))
    float ri = 0.f;
    if (tid < keep && total > 0.f) {
        const float p = sp_v[tid] / total;
        const int id = sp_i[tid];
        const float q = onehot ? ((id == d) ? 1.f : 0.f) : ((id >= 0) ? __expf(qs[id] - qstat[2 * j]) / qstat[2 * j + 1] : 0.f);
        ri = fmaxf(0.f, p - q);
    }
    __syncthreads();
    red[tid] = ri; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (tid == 0) {
        float pd = 0.f;
        for (int i = 0; i < keep; i++) if (sp_i[i] == d) { pd = sp_v[i] / total; break; }
        const float qd = onehot ? 1.f : ((d >= 0) ? __expf(qs[d] - qstat[2 * j]) / qstat[2 * j + 1] : 0.f);
        const unsigned long long r0 = nv_splitmix(rng ^ ((unsigned long long)(step + 1) * 0x9E3779B97F4A7C15ull)
                                                      ^ ((unsigned long long)(j + 1) * 0xA24BAED4963EE407ull));
        const float u = (float)(r0 >> 40) * (1.f / 16777216.f);          // [0, 1)
        const bool ok = (qd <= 0.f) || (u * qd < pd);                     // u < min(1, pd/qd)
        acc[j] = ok ? 1 : 0;
        if (!ok) {
            const float tot = red[keep - 1];                              // = 1 - sum_c min(p, q)
            if (!(tot > 0.f)) { rtok[j] = -2; }                           // -2: empty residual (q >= p on every candidate, or NaN): host fallback
            else {
                const unsigned long long r1 = nv_splitmix(r0 ^ 0x5EC0DEADBEEFCAFEull);
                const float u2 = (float)(r1 >> 40) * (1.f / 16777216.f) * tot;
                int pick = keep - 1;
                for (int i = 0; i < keep; i++) { if (u2 < red[i]) { pick = i; break; } }
                rtok[j] = sp_i[pick];
            }
        }
    }
}
// the fallback gate for the exact path + the per-step bookkeeping
extern "C" __global__ void nv_smp_note(int* __restrict__ counts, const int* __restrict__ tok, unsigned long long* __restrict__ rng,
                                       const float* __restrict__ xb, int* __restrict__ flags, int count) {
    NV_PDL_ENTRY();
    if (threadIdx.x == 0) { if (count) atomicAdd(&counts[tok[0]], 1); rng[1] += 1ull; flags[0] = (int)xb[XB_STATS + 2]; }
}
// ---- work item C (PLAN/NAIVE_DRAFT2_PROMPT_2026-10-19.md §6): the verify block's rows sampled PHASE-MAJOR ----
// rows_pack: dir 0 gathers section [off, off+n) of every row's exchange buffer (row stride `stride` floats)
// into a contiguous staging area [m][n] so ONE all-reduce carries every row's slots; dir 1 scatters it back.
extern "C" __global__ void nv_rows_pack(float* __restrict__ xs, float* __restrict__ xb, int m, int stride, int off, int n, int dir) {
    NV_PDL_ENTRY();
    const int j = blockIdx.x;
    if (j >= m) return;
    float* row = xb + (size_t)j * stride + off;
    float* st = xs + (size_t)j * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { if (dir == 0) st[i] = row[i]; else row[i] = st[i]; }
}
// rng_rows: the per-row path advances the lane's step once per row (nv_smp_note), so row j draws from
// (seed, step + j). The batched rows read exactly those pairs from their own copies (bitwise the same draws).
extern "C" __global__ void nv_rng_rows(const unsigned long long* __restrict__ rng, unsigned long long* __restrict__ rows, int m) {
    NV_PDL_ENTRY();
    const int j = threadIdx.x;
    if (j < m) { rows[2 * j] = rng[0]; rows[2 * j + 1] = rng[1] + (unsigned long long)j; }
}
// the committed tokens of a speculative round -> the frequency / presence table (eager, after the round)
extern "C" __global__ void nv_smp_count(int* __restrict__ counts, const int* __restrict__ toks, int n) {
    NV_PDL_ENTRY();
    const int i = threadIdx.x;
    if (i < n) atomicAdd(&counts[toks[i]], 1);
}
