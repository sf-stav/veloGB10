// d41_ds_mbias (leg PH1): the drafter's Markov bias with a cache of frequent tokens' rows, our own bits kept.
//
// One Markov chain step adds bias(t) = markov_head . markov_embed[t] (rank 256, this rank's vocab rows) to each lane's
// logits row. DS's sequence computes it every step with d41x_plain_gemv (Accum::Add: o = res, o += y, one f32 add):
// 33 MB of the head read per step at TP2. Here:
// * FILL (at load): rows = the cached tokens' embedding rows; y = the cache rows [K][nn], OVERWRITTEN with res (no add,
//   so a -0.0 stays -0.0).
// * STEP: if every ACTIVE lane's chain input id has a cache slot, each active lane's row gets y = cache[slot] + y (one
//   f32 add of the same two operands: the same bits) and the head is not read; otherwise every row is computed and
//   added exactly as d41x_plain_gemv does (inactive rows too, as it does).
// * leg P4, pad bit 0: a STEP where some active lane misses serves the lanes that HIT from the cache (the same add) and computes
//   only the other rows (the missing lanes, and the inactive ones as before): the head row is read once either way, the
//   arithmetic of the rows left out is skipped. Same bits per row as the all-rows step.
// * STEP_MISS (the probe): STEP with every lane treated as a miss.
// The per-column arithmetic is d41x_plain_cols' for K = 256 (lane k = 8 lane + j holds 8 products, fmaf over j from
// 0, then the xor tree 16 .. 1): the value of (row, column) never depends on the grid, the row count or the column's
// CTA, so FILL rows == STEP rows bit for bit (proven at load on every cached token, `mcache::selfcheck`). Grid-stride
// over the columns (gridDim.x CTAs, 8 warps a CTA, a warp a column); the hit path is a thread a column.
#pragma once

#define D41_DS_MB_K 256
#define D41_DS_MB_RB 8
#define D41_DS_MB_STEP 0
#define D41_DS_MB_FILL 1
#define D41_DS_MB_STEP_MISS 2

struct d41_ds_mc_dyn {
    int ring_head;
    int n_pend;
    int stopped;
    int pad0;
    int pend_id[D41_DS_MB_RB];
    int pend_slot[D41_DS_MB_RB];
    int write_slot[D41_DS_MB_RB];
};

struct d41_ds_mc_admit_args {
    int* slot;                      // [vocab] i32
    const int* ids;                 // [rows] i32 (step's chain inputs)
    const d41_ds_lane* lanes;       // [rows]
    d41_ds_mc_dyn* dyn;             // device dynamic ring + early-stop state
    int* dyn_owner;                 // [dyn_cap] i32
    const float* conf;              // [rows][conf_ld] f32 (for early-stop check)
    const double* early_t;          // [6] f64 round cost table (null or early_t[0] <= 0 = early-stop off)
    unsigned long long* cnt;        // [4] u64: [hit, miss, dyn_fills, early_skipped]
    double conf_offset;
    double lookahead_m;             // > 0: Stop::Lookahead; <= 0: Stop::First
    int rows, vocab, static_k, dyn_cap, step, k_max, conf_ld, early_on;
};
static_assert(sizeof(d41_ds_mc_admit_args) == 8 * 8 + 2 * 8 + 8 * 4, "d41_ds_mc_admit_args layout");

// grid (1), 32 threads: commits previous step's dynamic cache fills, checks conservative exact early-stop on conf[0..=step],
// and admits new missing active tokens into the dynamic ring for on-miss writeback in d41_ds_mbias.
extern "C" __global__ void __launch_bounds__(32) d41_ds_mc_admit(d41_ds_mc_admit_args a) {
    D41_PDL_ENTRY();
    __shared__ double ch[5];
    if (threadIdx.x != 0 || !a.dyn) return;
    d41_ds_mc_dyn* st = a.dyn;
    if (a.step == 0) st->stopped = 0;
    for (int p = 0; p < st->n_pend; ++p) {
        const int pid = st->pend_id[p], ps = st->pend_slot[p];
        if (pid >= 0 && pid < a.vocab && ps >= 0) a.slot[pid] = ps;
    }
    st->n_pend = 0;
    for (int r = 0; r < D41_DS_MB_RB; ++r) st->write_slot[r] = -1;
    if (st->stopped) return;

    if (a.early_on && a.early_t && a.conf && a.early_t[0] > 0.0 && a.rows <= D41_DS_MB_RB && a.k_max >= 1 && a.k_max <= 5) {
        int n_act = 0, l0 = -1;
        for (int r = 0; r < a.rows; ++r) if (a.lanes[r].active) { n_act++; l0 = r; }
        if (n_act == 1 && l0 >= 0 && a.step < a.k_max) {
            // Exact 1-lane stopping check on S_1..S_{step+1} (all computed before step's Markov head runs!).
            // With 1e-4 conservative rate slack for f32 round-cost broadcast rounding, st->stopped records step + 1
            // and rank 0 reads st->stopped back to cap depth <= step (100% F_{step}-measurable exact stopping time).
            double p = 1.0;
            const float* cz = a.conf + (size_t)l0 * a.conf_ld;
            for (int i = 0; i <= a.step; ++i) {
                const double x = (double)cz[i] + a.conf_offset;
                const double sig = 1.0 / (1.0 + exp(-x));
                p *= sig;
                ch[i] = p;
            }
            bool stopped_by_j = false;
            for (int j = 0; j <= a.step; ++j) {
                double ej = 1.0;
                for (int i = 0; i < j; ++i) ej += ch[i];
                const double rj = (ej / a.early_t[j]) * 0.9999;
                bool any_beats = false;
                if (a.lookahead_m > 0.0) {
                    for (int cc = j + 1; cc <= a.k_max; ++cc) {
                        double ec = 1.0;
                        double dec = ch[j];
                        for (int i = 0; i < cc; ++i) {
                            if (i <= j) ec += ch[i];
                            else { dec *= a.lookahead_m; ec += dec; }
                        }
                        if (ec / a.early_t[cc] > rj) { any_beats = true; break; }
                    }
                } else {
                    const double e1 = ej + ch[j];
                    if (e1 / a.early_t[j + 1] > rj) any_beats = true;
                }
                if (!any_beats) { stopped_by_j = true; break; }
            }
            if (stopped_by_j) { st->stopped = a.step + 1; return; }
        }
    }

    if (a.dyn_cap <= 0 || !a.dyn_owner || a.rows > D41_DS_MB_RB) return;
    for (int r = 0; r < a.rows; ++r) {
        if (!a.lanes[r].active) continue;
        const int id = a.ids[r];
        if (id < 0 || id >= a.vocab || a.slot[id] >= 0) continue;
        bool dup = false;
        for (int p = 0; p < st->n_pend; ++p) if (st->pend_id[p] == id) { dup = true; break; }
        if (dup) continue;
        if (st->n_pend >= D41_DS_MB_RB || st->n_pend >= a.dyn_cap) break;
        const int idx = st->ring_head;
        st->ring_head = (idx + 1 >= a.dyn_cap) ? 0 : (idx + 1);
        const int s = a.static_k + idx;
        const int old = a.dyn_owner[idx];
        if (old >= 0 && old < a.vocab && a.slot[old] == s) a.slot[old] = -1;
        a.dyn_owner[idx] = id;
        st->write_slot[r] = s;
        st->pend_id[st->n_pend] = id;
        st->pend_slot[st->n_pend] = s;
        st->n_pend++;
        if (a.cnt) atomicAdd(a.cnt + 2, 1ull);
    }
}

struct d41_ds_mbias_args {
    const void* w;                  // Markov head rows of this rank [nn][256] (wcode 0 bf16 / 1 f16)
    const __nv_bfloat16* x;         // the gathered Markov embedding rows: row r at x + r * ldx
    float* y;                       // output rows: row r at y + r * ldy (STEP: the logits rows; FILL: the cache)
    const int* ids;                 // STEP: the chain input id of row r (the ids table row `step`)
    const int* slot;                // [vocab] i32: the cache row of an id, -1 = none
    float* cache;                   // [K + dyn_cap][nn] f32 (written on dynamic miss when dyn != null)
    const d41_ds_lane* lanes;       // STEP: row r = lane r (inactive lanes neither decide nor are written on a hit)
    unsigned long long* cnt;        // STEP: [4] [hit, miss, dyn_fills, early_skipped] (null = not counted)
    const d41_ds_mc_dyn* dyn;       // optional dynamic cache + early-stop state (null = off)
    unsigned long long pad_ptr;     // keep 8-byte alignment
    int wcode, nn, ldx, ldy, rows, mode, vocab, pad;   // pad bit 0 (leg P4): per-lane hit / miss in a STEP (--dsv41-markov-cache-lanes)
};
static_assert(sizeof(d41_ds_mbias_args) == 10 * 8 + 8 * 4, "d41_ds_mbias_args layout");

__device__ __forceinline__ bool d41_ds_mb_skipped(unsigned m, int r) { return r < 32 && ((m >> r) & 1u); }

// one column of rows r0 .. r0 + RB from its loaded head row `raw` (the d41x_plain_cols<RB> arithmetic for K = 256);
// lane 0 gets the sums (rows past a.rows are not summed: warp-uniform)
template <int RB>
__device__ __forceinline__ void d41_ds_mb_col(const d41_ds_mbias_args& a, uint4 raw, const float (&xr)[RB][8], int r0, unsigned skip, float* res) {
    float w[8];
    if (a.wcode == 0) d41_ds_bf8(raw, w); else d41_ds_h8(raw, w);
#pragma unroll
    for (int i = 0; i < RB; ++i) {
        float acc = 0.f;
        if (r0 + i < a.rows && !d41_ds_mb_skipped(skip, r0 + i)) {
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(w[j], xr[i][j], acc);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        }
        res[i] = acc;
    }
}

// Vectorized cache-hit / partial-hit bias add: y[r][n] = cache[slot_r][n] + y[r][n] (exact f32 sum).
template <int MAXR>
__device__ __forceinline__ void d41_ds_mb_hit_add(const d41_ds_mbias_args& a, const int (&sl)[D41_DS_MB_RB], bool hit, unsigned hm) {
    const int t = threadIdx.x;
    const bool vec4 = ((a.nn | a.ldy) & 3) == 0 && (((unsigned long long)a.y | (unsigned long long)a.cache) & 15ull) == 0ull;
    if (vec4) {
        const int nn4 = a.nn >> 2;
        for (int i4 = blockIdx.x * 256 + t; i4 < nn4; i4 += gridDim.x * 256) {
            const size_t n = (size_t)i4 << 2;
#pragma unroll
            for (int r = 0; r < MAXR; ++r) {
                if (r < a.rows && sl[r] >= 0 && (hit || ((hm >> r) & 1u))) {
                    float4* yp = (float4*)(a.y + (size_t)r * a.ldy + n);
                    const float4 c4 = *(const float4*)(a.cache + (size_t)sl[r] * a.nn + n);
                    const float4 y4 = *yp;
                    float4 o4;
                    o4.x = __fadd_rn(c4.x, y4.x);
                    o4.y = __fadd_rn(c4.y, y4.y);
                    o4.z = __fadd_rn(c4.z, y4.z);
                    o4.w = __fadd_rn(c4.w, y4.w);
                    *yp = o4;
                }
            }
        }
    } else {
        for (int n = blockIdx.x * 256 + t; n < a.nn; n += gridDim.x * 256) {
#pragma unroll
            for (int r = 0; r < MAXR; ++r) {
                if (r < a.rows && sl[r] >= 0 && (hit || ((hm >> r) & 1u))) {
                    float* yp = a.y + (size_t)r * a.ldy + n;
                    float o = __fadd_rn(a.cache[(size_t)sl[r] * a.nn + n], *yp);
                    *yp = o;
                }
            }
        }
    }
}

// leg P7.6: compacted missing-row GEMV over M <= 4 missing rows (M = 1, 2, or 4), 8 columns per warp step,
// reduced via d41_ds_rs8 (10 shuffles instead of 40, identical IEEE xor-tree sums) and written coalesced by lanes 0..7.
template <int M>
__device__ __forceinline__ void d41_ds_mb_miss(const d41_ds_mbias_args& a, const int (&mr)[4], int m_miss) {
    const int t = threadIdx.x, lane = t & 31;
    const int gw = blockIdx.x * 8 + (t >> 5), nw = gridDim.x * 8;
    float xr[M][8];
    int ws[M];
#pragma unroll
    for (int i = 0; i < M; ++i) {
        if (i < m_miss) {
            const int r_idx = mr[i];
            const uint4 raw = *(const uint4*)(a.x + (size_t)r_idx * a.ldx + lane * 8);
            d41_ds_bf8(raw, xr[i]);
            ws[i] = (a.dyn && r_idx < D41_DS_MB_RB) ? a.dyn->write_slot[r_idx] : -1;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) xr[i][j] = 0.f;
            ws[i] = -1;
        }
    }
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    const unsigned short* mhp = (const unsigned short*)a.w + lane * 8;
    for (int n0 = gw * 8; n0 < a.nn; n0 += nw * 8) {
        uint4 raw[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int n = min(n0 + u, a.nn - 1);
            raw[u] = d41_ds_ld_last(mhp + (size_t)n * D41_DS_MB_K, pol);
        }
        float pp[M][8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            float wv[8];
            if (a.wcode == 0) d41_ds_bf8(raw[u], wv); else d41_ds_h8(raw[u], wv);
#pragma unroll
            for (int i = 0; i < M; ++i) {
                float acc = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc = fmaf(wv[j], xr[i][j], acc);
                pp[i][u] = acc;
            }
        }
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if (i < m_miss) {
                const float v = d41_ds_rs8(pp[i], lane);
                const float g = __shfl_sync(0xFFFFFFFFu, v, (lane & 7) << 2);
                const int n = n0 + lane;
                if (lane < 8 && n < a.nn) {
                    const int r_idx = mr[i];
                    float* yp = a.y + (size_t)r_idx * a.ldy + n;
                    *yp = __fadd_rn(g, *yp);
                    if (ws[i] >= 0) a.cache[(size_t)ws[i] * a.nn + n] = g;
                }
            }
        }
    }
}

// RB rows a pass (registers scale with it), U head rows a warp loads at once (used for MODE_FILL and 5..8-lane STEP)
template <int RB, int U>
__device__ __forceinline__ void d41_ds_mbias_body(const d41_ds_mbias_args& a) {
    const int t = threadIdx.x, lane = t & 31;
    if (a.mode == D41_DS_MB_STEP && a.dyn && a.dyn->stopped != 0) {
        if (blockIdx.x == 0 && t == 0 && a.cnt) atomicAdd(a.cnt + 3, 1ull);
        return;
    }
    // the step's decision (every thread computes it from the same loads)
    bool hit = a.mode == D41_DS_MB_STEP;
    unsigned hm = 0;   // leg P4: the lanes served from the cache although another lane missed (bit r = row r)
    int sl[D41_DS_MB_RB];
#pragma unroll
    for (int i = 0; i < D41_DS_MB_RB; ++i) sl[i] = -1;
    if (hit) {
        hit = a.rows <= D41_DS_MB_RB;   // the cache path serves <= 8 lanes
#pragma unroll
        for (int r = 0; r < D41_DS_MB_RB; ++r) {
            if (r < a.rows && a.lanes[r].active) {
                const int id = a.ids[r];
                const int s = (id >= 0 && id < a.vocab) ? a.slot[id] : -1;
                if (s < 0) hit = false;
                sl[r] = s;
            } else if (r < a.rows) {
                hm |= 1u << r;   // skip inactive lanes on miss too
            }
        }
        if (blockIdx.x == 0 && t == 0 && a.cnt) atomicAdd(a.cnt + (hit ? 0 : 1), 1ull);
        if (!hit && (a.pad & 1) && a.rows <= D41_DS_MB_RB) {
#pragma unroll
            for (int r = 0; r < D41_DS_MB_RB; ++r) if (r < a.rows && sl[r] >= 0) hm |= 1u << r;
        }
    }
    if (hit || hm) {
        d41_ds_mb_hit_add<D41_DS_MB_RB>(a, sl, hit, hm);
        if (hit) return;
    }
    if (a.mode != D41_DS_MB_FILL && a.rows <= D41_DS_MB_RB) {
        int mr[4] = {0, 0, 0, 0};
        int m_miss = 0;
        int m_all[D41_DS_MB_RB];
#pragma unroll
        for (int r = 0; r < D41_DS_MB_RB; ++r) {
            if (r < a.rows && !d41_ds_mb_skipped(hm, r)) m_all[m_miss++] = r;
        }
        if (m_miss == 0) return;
        if (m_miss <= 4) {
#pragma unroll
            for (int i = 0; i < 4; ++i) if (i < m_miss) mr[i] = m_all[i];
            if (m_miss == 1) d41_ds_mb_miss<1>(a, mr, 1);
            else if (m_miss == 2) d41_ds_mb_miss<2>(a, mr, 2);
            else d41_ds_mb_miss<4>(a, mr, m_miss);
            return;
        }
    }
    const bool fill = a.mode == D41_DS_MB_FILL;
    const int gw = blockIdx.x * 8 + (t >> 5), nw = gridDim.x * 8;
    for (int r0 = 0; r0 < a.rows; r0 += RB) {
        // this lane's 8 x values of each row (d41x_ld of a bf16 row)
        float xr[RB][8];
#pragma unroll
        for (int i = 0; i < RB; ++i) {
            if (r0 + i < a.rows && !d41_ds_mb_skipped(hm, r0 + i)) {
                const uint4 raw = *(const uint4*)(a.x + (size_t)(r0 + i) * a.ldx + lane * 8);
                d41_ds_bf8(raw, xr[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) xr[i][j] = 0.f;
            }
        }
        // U adjacent columns a warp at a time, their head rows (2 KB) loaded together
        for (int n0 = gw * U; n0 < a.nn; n0 += nw * U) {
            uint4 raw[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int n = min(n0 + u, a.nn - 1);
                raw[u] = *(const uint4*)((const char*)a.w + ((size_t)n * D41_DS_MB_K + lane * 8) * 2);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int n = n0 + u;
                if (n >= a.nn) break;   // warp-uniform
                float res[RB];
                d41_ds_mb_col<RB>(a, raw[u], xr, r0, hm, res);
                if (lane == 0) {
#pragma unroll
                    for (int i = 0; i < RB; ++i) if (r0 + i < a.rows && !d41_ds_mb_skipped(hm, r0 + i)) {
                        float* yp = a.y + (size_t)(r0 + i) * a.ldy + n;
                        float o = res[i];
                        if (!fill) o += *yp;
                        *yp = o;
                        if (!fill && a.dyn) {
                            const int r_idx = r0 + i;
                            const int ws = (r_idx < D41_DS_MB_RB) ? a.dyn->write_slot[r_idx] : -1;
                            if (ws >= 0) a.cache[(size_t)ws * a.nn + n] = res[i];
                        }
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void d41_ds_mbias_r4_body(const d41_ds_mbias_args& a) {
    const int t = threadIdx.x;
    if (a.mode == D41_DS_MB_STEP && a.dyn && a.dyn->stopped != 0) {
        if (blockIdx.x == 0 && t == 0 && a.cnt) atomicAdd(a.cnt + 3, 1ull);
        return;
    }
    bool hit = a.mode == D41_DS_MB_STEP;
    unsigned hm = 0;
    int sl[D41_DS_MB_RB] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (hit) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < a.rows && a.lanes[r].active) {
                const int id = a.ids[r];
                const int s = (id >= 0 && id < a.vocab) ? a.slot[id] : -1;
                if (s < 0) hit = false;
                sl[r] = s;
            } else if (r < a.rows) {
                hm |= 1u << r;
            }
        }
        if (blockIdx.x == 0 && t == 0 && a.cnt) atomicAdd(a.cnt + (hit ? 0 : 1), 1ull);
        if (!hit && (a.pad & 1)) {
#pragma unroll
            for (int r = 0; r < 4; ++r) if (r < a.rows && sl[r] >= 0) hm |= 1u << r;
        }
    }
    if (hit || hm) {
        d41_ds_mb_hit_add<4>(a, sl, hit, hm);
        if (hit) return;
    }
    int mr[4] = {0, 0, 0, 0};
    int m_miss = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (r < a.rows && !d41_ds_mb_skipped(hm, r)) mr[m_miss++] = r;
    }
    if (m_miss == 0) return;
    if (m_miss == 1) d41_ds_mb_miss<1>(a, mr, 1);
    else if (m_miss == 2) d41_ds_mb_miss<2>(a, mr, 2);
    else d41_ds_mb_miss<4>(a, mr, m_miss);
}

// grid (G), 256 threads. d41_ds_mbias: any rows (8 a pass; FILL, STEP of 5-8 lanes); d41_ds_mbias_r4: <= 4 rows.
extern "C" __global__ void __launch_bounds__(256) d41_ds_mbias(d41_ds_mbias_args a) { D41_PDL_ENTRY(); d41_ds_mbias_body<8, 2>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_ds_mbias_r4(d41_ds_mbias_args a) { D41_PDL_ENTRY(); d41_ds_mbias_r4_body(a); }
