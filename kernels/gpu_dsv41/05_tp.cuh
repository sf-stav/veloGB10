// =============================================================================
// kernels/gpu_dsv41/05_tp.cuh — band 05 (leg TPC): the TP communication layer's device side.
// Host side: src/dsv41/tp/kern.rs (entry list `FNS`, loaded under its own module name).
//
// 1. The vocab-parallel head exchange (PLAN/DSV41_TP_DESIGN.md 3.4): every rank reduces its vocabulary
//    shard of the processed logits to ONE 16-word record per row (local maximum + id, local sum of
//    exp, the keyed Gumbel maximum, the probe token's logit if this rank owns it), the records are
//    gathered (an fp32 sum over zero-padded slots: x + 0 is exact), and every rank combines them in
//    rank order into the greedy token, the sampled token, the log-sum and the accept decision of
//    DeepSeek's token-level rule (u < min(1, p_target / q_draft)). The residual of the first rejected
//    row, max(p - alpha q, 0) with alpha = exp(-5e-5), is sampled by a second, same-shaped exchange.
//    Zero residual mass -> the first maximum logit. RNG streams (X-DI 4.2): fresh sample pos*3 + 0,
//    residual pos*3 + 1, accept uniform pos*3 + 2; drafter draws add 1,000,000,007 to the seed.
//    Every max is lexicographic (value, then the LOWEST id), so greedy and sampled tokens do not depend
//    on the vocabulary split at all; the log-sum combines per-shard (m, s) pairs in rank order.
// 2. The smoke's synthetic producers and checkers: partials of the real frame shapes from a keyed hash
//    (full 24-bit mantissas, exponents spread over 2^-8..2^7, so summation order shows in the bits), and a
//    checker that recomputes every rank's partial and sums them in the transport's association.
// Reference-grade: one block per row for the head; the TP forward's P-legs tune it (see the leg report).
// =============================================================================

#define D41TP_HEAD_REC 16
#define D41TP_RES_REC 4
#define D41TP_HEAD_THREADS 256
#define D41TP_DRAFT_SEED_OFFSET 1000000007ull

// One row's sampling parameters (32 B; mirrored by src/dsv41/tp/comm.rs `TpRow`).
struct d41tp_row {
    unsigned long long seed;   // the request's seed
    unsigned long long pos;    // absolute position of the token this row samples
    float temp;                // processed logit = logit / temp; 0 = greedy (logit itself, no noise)
    int probe;                 // the draft token this row verifies (-1 = none)
    unsigned flags;            // bit 0: a drafter draw (seed + 1,000,000,007)
    unsigned pad;
};

// One row's result (32 B; mirrored by `TpOut`).
struct d41tp_out {
    int greedy;        // the first maximum processed logit (lowest id on ties)
    int sample;        // the keyed Gumbel-max sample (= greedy for a greedy row)
    float lse;         // log-sum-exp of the processed logits (rank-order combine of the shards)
    float max_x;       // the maximum processed logit
    float probe_lp;    // log-probability of `probe` under this distribution (-inf if none / masked)
    int accept;        // 1 = the probe (a draft) is accepted, 0 = rejected, -1 = no probe
    int residual;      // the residual sample of a rejected row (-1 until the residual exchange ran)
    unsigned flags;    // bit 0: zero residual mass (residual = greedy); bits 8..31: non-finite logits seen
};

// The by-value argument of the two shard kernels (mirrored by src/dsv41/tp/kern.rs `LocalArgs`; 96 B).
struct d41tp_local_args {
    const float* logits;       // this rank's processed-logit shard, row r at logits + r * ld
    long long ld;
    const float* dlogits;      // residual: the drafter's processed logits of the same rows / shard
    long long dld;
    const d41tp_row* rows_p;
    const d41tp_out* tgt;      // residual: the target exchange's outputs (lse)
    const d41tp_out* dout;     // residual: the drafter exchange's outputs (lse)
    float* rec;                // the gather frame
    float* part;               // per-(row, chunk) partial records [rows][nb][16] (stage 1a -> 1b scratch)
    int vlen, voff, rank, world, zero_others, chunk;   // chunk: shard elements per block (nb = ceil(vlen / chunk))
};

__device__ __forceinline__ unsigned long long d41tp_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// The per-row key of one RNG stream; per-element draws are mix64(key + id).
__device__ __forceinline__ unsigned long long d41tp_stream_key(const d41tp_row& r, unsigned stream) {
    const unsigned long long seed = r.seed + ((r.flags & 1u) ? D41TP_DRAFT_SEED_OFFSET : 0ull);
    return d41tp_mix64(seed ^ d41tp_mix64(r.pos * 3ull + (unsigned long long)stream));
}
// Gumbel(0,1) from 32 hash bits: E = -log(u) with u = (h + 0.5) / 2^32 taken from whichever side keeps
// full float resolution (log1p near u = 1), then g = -log(E). Range about [-3.13, 22.9]: a token more than
// ~26 nats below the row's maximum can never be drawn (total mass < 129,280 e^-26 = 6.5e-7).
__device__ __forceinline__ float d41tp_gumbel(unsigned long long key, int id) {
    const unsigned h = (unsigned)(d41tp_mix64(key + (unsigned long long)(unsigned)id) >> 32);
    float e;
    if (h < 0x80000000u) {
        const float u = ((float)h + 0.5f) * 2.3283064365386963e-10f;
        e = -logf(u);
    } else {
        const float w = ((float)(0xFFFFFFFFu - h) + 0.5f) * 2.3283064365386963e-10f;   // 1 - u
        e = -log1pf(-w);
    }
    return -logf(e);
}
// One uniform in (0, 1) per row (24 bits): the accept draw.
__device__ __forceinline__ float d41tp_uniform(unsigned long long key) {
    return ((float)(d41tp_mix64(key) >> 40) + 0.5f) * 5.9604644775390625e-08f;
}

// lexicographic max: (v, id) beats (bv, bid) when larger, or equal with a lower id (id < 0 = empty)
__device__ __forceinline__ bool d41tp_beats(float v, int id, float bv, int bid) {
    if (id < 0) return false;
    if (bid < 0) return true;
    return v > bv || (v == bv && id < bid);
}
// (m, s) pair merge in a fixed order: a first, then b
__device__ __forceinline__ void d41tp_lse_merge(float& m, float& s, float bm, float bs) {
    if (bm == -INFINITY) return;
    if (m == -INFINITY) { m = bm; s = bs; return; }
    const float M = fmaxf(m, bm);
    s = s * expf(m - M) + bs * expf(bm - M);
    m = M;
}

// ---- head exchange, stage 1a: one partial record per (row, chunk of the shard) -------------------------
// grid = (rows, nb), block = D41TP_HEAD_THREADS. logits: this rank's shard, row r at logits + r * ld, ids voff ..
// voff + vlen - 1; block (r, b) covers shard elements [b * chunk, min(vlen, (b + 1) * chunk)). Partial record at
// part[(r * nb + b) * 16]: [0] max processed logit, [1] its id (-1 none), [2] sum exp(x - max), [3] Gumbel max
// value, [4] its id, [5] its processed logit, [6] the probe's processed logit (0 if not here), [7] 1 if here,
// [8] non-finite logits seen (NaN / +inf; -inf is a masked token, not counted).
extern "C" __global__ void __launch_bounds__(D41TP_HEAD_THREADS)
d41tp_head_local(const d41tp_local_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x, b = blockIdx.y, nb = gridDim.y;
    const d41tp_row rp = a.rows_p[r];
    const float* L = a.logits + (long long)r * a.ld;
    const bool greedy = !(rp.temp > 0.f);
    const unsigned long long key = d41tp_stream_key(rp, 0u);
    const int j0 = b * a.chunk, j1 = min(a.vlen, j0 + a.chunk);
    float m = -INFINITY, s = 0.f, gv = -INFINITY, gx = 0.f, px = 0.f, pown = 0.f;
    int mid = -1, gid = -1, nan = 0;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        const float l = L[j];
        const int id = a.voff + j;
        if (isnan(l) || l == INFINITY) { nan++; continue; }
        const float x = greedy ? l : __fdiv_rn(l, rp.temp);
        if (id == rp.probe) { px = x; pown = 1.f; }
        if (x == -INFINITY) continue;
        if (x > m || mid < 0) {
            s = (mid < 0) ? 1.f : s * expf(m - x) + 1.f;
            m = x; mid = id;
        } else {
            s += expf(x - m);
        }
        if (!greedy) {
            const float v = x + d41tp_gumbel(key, id);
            if (gid < 0 || v > gv) { gv = v; gid = id; gx = x; }
        }
    }
    if (greedy) { gv = m; gid = mid; gx = m; }
    __shared__ float sm[D41TP_HEAD_THREADS], ss[D41TP_HEAD_THREADS], sgv[D41TP_HEAD_THREADS], sgx[D41TP_HEAD_THREADS], spx[D41TP_HEAD_THREADS], spo[D41TP_HEAD_THREADS];
    __shared__ int smid[D41TP_HEAD_THREADS], sgid[D41TP_HEAD_THREADS], snan[D41TP_HEAD_THREADS];
    const int t = threadIdx.x;
    sm[t] = m; ss[t] = s; smid[t] = mid; sgv[t] = gv; sgid[t] = gid; sgx[t] = gx; spx[t] = px; spo[t] = pown; snan[t] = nan;
    __syncthreads();
    // fixed pairing tree: t takes t + stride
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            const int u = t + stride;
            float mm = sm[t], s2 = ss[t];
            if (smid[u] >= 0) {
                if (smid[t] < 0) { mm = sm[u]; s2 = ss[u]; }
                else d41tp_lse_merge(mm, s2, sm[u], ss[u]);
            }
            const bool take = d41tp_beats(sm[u], smid[u], sm[t], smid[t]);
            ss[t] = s2;
            if (take) { sm[t] = sm[u]; smid[t] = smid[u]; }
            if (d41tp_beats(sgv[u], sgid[u], sgv[t], sgid[t])) { sgv[t] = sgv[u]; sgid[t] = sgid[u]; sgx[t] = sgx[u]; }
            spx[t] = spx[t] + spx[u];
            spo[t] = spo[t] + spo[u];
            snan[t] = snan[t] + snan[u];
        }
        __syncthreads();
    }
    if (t == 0) {
        float* out = a.part + ((long long)r * nb + b) * D41TP_HEAD_REC;
        out[0] = smid[0] >= 0 ? sm[0] : -INFINITY;
        out[1] = (float)smid[0];
        out[2] = smid[0] >= 0 ? ss[0] : 0.f;
        out[3] = sgid[0] >= 0 ? sgv[0] : -INFINITY;
        out[4] = (float)sgid[0];
        out[5] = sgx[0];
        out[6] = spx[0];
        out[7] = spo[0];
        out[8] = (float)snan[0];
    }
}

// ---- head exchange, stage 1b: merge the row's nb partials in chunk order into this rank's record --------
// grid = rows, block = 32. rec: the gather frame [rows][world][16]; this rank writes slot `rank` and, when
// zero_others, zeros the other ranks' slots of its rows (the fp32 sum over ranks is then the gather).
extern "C" __global__ void d41tp_head_merge(const d41tp_local_args a, int nb) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x, t = threadIdx.x;
    float* out = a.rec + ((long long)r * a.world + a.rank) * D41TP_HEAD_REC;
    if (t == 0) {
        float m = -INFINITY, s = 0.f, gv = -INFINITY, gx = 0.f, px = 0.f, pown = 0.f, nan = 0.f;
        int mid = -1, gid = -1;
        for (int b = 0; b < nb; b++) {
            const float* P = a.part + ((long long)r * nb + b) * D41TP_HEAD_REC;
            const int pid = (int)P[1], pg = (int)P[4];
            if (pid >= 0) {
                if (mid < 0) { s = P[2]; } else { float mm = m; d41tp_lse_merge(mm, s, P[0], P[2]); }
                if (d41tp_beats(P[0], pid, m, mid)) { m = P[0]; mid = pid; }
            }
            if (d41tp_beats(P[3], pg, gv, gid)) { gv = P[3]; gid = pg; gx = P[5]; }
            px = px + P[6];
            pown = pown + P[7];
            nan = nan + P[8];
        }
        out[0] = mid >= 0 ? m : -INFINITY;
        out[1] = (float)mid;
        out[2] = mid >= 0 ? s : 0.f;
        out[3] = gid >= 0 ? gv : -INFINITY;
        out[4] = (float)gid;
        out[5] = gx;
        out[6] = px;
        out[7] = pown;
        out[8] = nan;
    }
    if (t >= 9 && t < D41TP_HEAD_REC) out[t] = 0.f;
    if (a.zero_others) {
        for (int i = t; i < a.world * D41TP_HEAD_REC; i += blockDim.x) {
            const int q = i / D41TP_HEAD_REC;
            if (q != a.rank) a.rec[((long long)r * a.world + q) * D41TP_HEAD_REC + (i % D41TP_HEAD_REC)] = 0.f;
        }
    }
}

// ---- head exchange, stage 2: combine the gathered records (every rank, identical arithmetic) ---------
// One thread per row. (Accepting drafts and drawing the residual is DS's sampler, band 90: leg SE2 removed the duplicate here.)
extern "C" __global__ void d41tp_head_combine(const float* __restrict__ rec, int rows, int world, const d41tp_row* __restrict__ rows_p,
                                              d41tp_out* __restrict__ out) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const d41tp_row rp = rows_p[r];
    float m = -INFINITY, gv = -INFINITY, px = 0.f, pown = 0.f;
    int mid = -1, gid = -1, nan = 0;
    for (int q = 0; q < world; q++) {
        const float* R = rec + ((long long)r * world + q) * D41TP_HEAD_REC;
        const int id = (int)R[1], g = (int)R[4];
        if (d41tp_beats(R[0], id, m, mid)) { m = R[0]; mid = id; }
        if (d41tp_beats(R[3], g, gv, gid)) { gv = R[3]; gid = g; }
        px = px + R[6];
        pown = pown + R[7];
        nan += (int)R[8];
    }
    float S = 0.f;
    for (int q = 0; q < world; q++) {
        const float* R = rec + ((long long)r * world + q) * D41TP_HEAD_REC;
        if ((int)R[1] >= 0 && R[0] != -INFINITY) S = S + R[2] * expf(R[0] - m);
    }
    const float lse = (mid >= 0) ? m + logf(S) : -INFINITY;
    const bool greedy = !(rp.temp > 0.f);
    d41tp_out o;
    o.greedy = mid;
    o.sample = greedy ? mid : gid;
    o.lse = lse;
    o.max_x = m;
    o.probe_lp = (rp.probe >= 0 && pown > 0.f && px != -INFINITY) ? px - lse : -INFINITY;
    o.accept = -1;
    o.residual = -1;
    o.flags = ((unsigned)min(nan, 0xFFFFFF)) << 8;
    out[r] = o;
}

// ---- the smoke's synthetic producers / checkers ---------------------------------------------------------
__device__ __forceinline__ float d41tp_synth_val(unsigned long long key, int rank, long long idx) {
    const unsigned long long h = d41tp_mix64(key ^ d41tp_mix64(((unsigned long long)(rank + 1) << 48) ^ (unsigned long long)idx));
    const float u = (float)(h >> 40) * 5.9604644775390625e-08f;   // [0, 1), 24 bits, exact
    const int e = (int)((h >> 8) & 15ull) - 8;
    return (2.f * u - 1.f) * ldexpf(1.f, e);
}
// The frame layout (src/dsv41/tp/frame.rs FrameSpec): [rows][hidden] then [rows][mix_stride] (cols >= mix_used
// are zero padding) then [rows][extra].
__device__ __forceinline__ bool d41tp_is_pad(long long i, int rows, int hidden, int mix_stride, int mix_used) {
    const long long h = (long long)rows * hidden;
    if (i < h) return false;
    const long long j = i - h;
    if (j >= (long long)rows * mix_stride) return false;
    return (int)(j % mix_stride) >= mix_used;
}
extern "C" __global__ void d41tp_synth_frame(float* __restrict__ f, long long n, int rows, int hidden, int mix_stride, int mix_used,
                                             unsigned long long key, int rank) {
    D41_PDL_ENTRY();
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        f[i] = d41tp_is_pad(i, rows, hidden, mix_stride, mix_used) ? 0.f : d41tp_synth_val(key, rank, i);
}

// The transport's association over W rank partials (tp_xport::rd_tree_reference): round j pairs the ranks
// differing in bit order[j] (order = first, first + 1, ... mod rounds), lower + upper. Compile-time W and
// FIRST, so every index is a constant after unrolling (no local memory).
template <int W, int FIRST>
__device__ __forceinline__ float d41tp_tree(unsigned long long key, long long idx, bool pad) {
    float v[W];
#pragma unroll
    for (int q = 0; q < W; q++) v[q] = pad ? 0.f : d41tp_synth_val(key, q, idx);
    constexpr int R = (W == 2) ? 1 : (W == 4) ? 2 : 3;
#pragma unroll
    for (int j = 0; j < R; j++) {
        const int bit = 1 << ((FIRST + j) % R);
#pragma unroll
        for (int q = 0; q < W; q++) if ((q & bit) == 0) v[q] = v[q] + v[q | bit];
    }
    return v[0];
}
template <int W, int FIRST>
__device__ void d41tp_check_body(const float* f, long long n, int rows, int hidden, int mix_stride, int mix_used,
                                 unsigned long long key, unsigned* bad) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float want = d41tp_tree<W, FIRST>(key, i, d41tp_is_pad(i, rows, hidden, mix_stride, mix_used));
        if (__float_as_uint(want) != __float_as_uint(f[i])) {
            atomicAdd(&bad[0], 1u);
            atomicMin(&bad[1], (unsigned)min(i, 0xFFFFFFFFll));
        }
    }
}
template <int W>
__device__ void d41tp_check_seq(const float* f, long long n, int rows, int hidden, int mix_stride, int mix_used,
                                unsigned long long key, unsigned* bad) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const bool pad = d41tp_is_pad(i, rows, hidden, mix_stride, mix_used);
        float want = pad ? 0.f : d41tp_synth_val(key, 0, i);
#pragma unroll
        for (int q = 1; q < W; q++) want = want + (pad ? 0.f : d41tp_synth_val(key, q, i));
        if (__float_as_uint(want) != __float_as_uint(f[i])) {
            atomicAdd(&bad[0], 1u);
            atomicMin(&bad[1], (unsigned)min(i, 0xFFFFFFFFll));
        }
    }
}
// bad[0] += mismatches; bad[1] = min mismatching index (pre-set to 0xFFFFFFFF by the host).
extern "C" __global__ void d41tp_synth_check(const float* __restrict__ f, long long n, int rows, int hidden, int mix_stride, int mix_used,
                                             unsigned long long key, int world, int first, unsigned* __restrict__ bad) {
    D41_PDL_ENTRY();
    if (world == 1) {
        for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
            const float want = d41tp_is_pad(i, rows, hidden, mix_stride, mix_used) ? 0.f : d41tp_synth_val(key, 0, i);
            if (__float_as_uint(want) != __float_as_uint(f[i])) { atomicAdd(&bad[0], 1u); atomicMin(&bad[1], (unsigned)min(i, 0xFFFFFFFFll)); }
        }
    } else if (world == 2) d41tp_check_body<2, 0>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 3) d41tp_check_seq<3>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 4 && first == 0) d41tp_check_body<4, 0>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 4) d41tp_check_body<4, 1>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 6) d41tp_check_seq<6>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 8 && first == 0) d41tp_check_body<8, 0>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 8 && first == 1) d41tp_check_body<8, 1>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (world == 8) d41tp_check_body<8, 2>(f, n, rows, hidden, mix_stride, mix_used, key, bad);
    else if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&bad[0], 0x80000000u);
}

// Synthetic processed logits over the GLOBAL vocabulary (row r, id g): uniform in [-8, 8) from the key, with
// planted cases: rows r % 3 == 0 carry an exact tie at 40.0 between a low id (rank 0's shard) and a high id
// (the last rank's shard); rows r % 3 == 1 a single peak at the high id; one id in 4,096 is masked (-inf).
extern "C" __global__ void d41tp_synth_logits(float* __restrict__ L, long long ld, int rows, int vocab, unsigned long long key) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y;
    if (r >= rows) return;
    const int lo = 17 + (r * 131) % 1000, hi = vocab - 1 - (r * 977) % 1000;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < vocab; g += gridDim.x * blockDim.x) {
        const unsigned long long h = d41tp_mix64(key ^ d41tp_mix64(((unsigned long long)r << 32) ^ (unsigned long long)g));
        float x = ((float)(h >> 40) * 5.9604644775390625e-08f - 0.5f) * 16.f;
        if (((h >> 8) & 4095ull) == 0ull) x = -INFINITY;
        if ((r % 3 == 0 && (g == lo || g == hi)) || (r % 3 == 1 && g == hi)) x = 40.f;
        L[(long long)r * ld + g] = x;
    }
}

// A timing gate: one thread spins `ns` on the global timer, so the host can enqueue a whole round behind it and the
// round then runs GPU-bound (the smoke's per-collective timings are not host-launch-bound).
extern "C" __global__ void d41tp_spin(unsigned long long ns) {
    D41_PDL_ENTRY();
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { __nanosleep(1000); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

// leg TP: the TP forward glue (host side src/dsv41/tp/tpk.rs)
#include "05_tp_fwd.cuh"
// leg FW: the serving engine's glue (taps, byte-exact gathers; host side src/dsv41/tp/tpk.rs)
#include "06_fw.cuh"
// leg PX1: the dsv41 copies of the world-2 exchange pair (PDL stage / collect; host side src/dsv41/tp/xchg.rs)
#include "07_xchg.cuh"
