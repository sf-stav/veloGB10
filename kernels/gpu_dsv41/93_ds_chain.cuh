// d41_ds_mstep_l{2,4} / d41_ds_vscan / d41_ds_vfin / d41_ds_conf / d41_ds_plain_r{16,24,32} (leg PK5): the
// drafter's head side with fewer launches and fewer bytes, every value bit-identical to the launches it replaces.
//
// One Markov chain step of `nlanes` lanes = d41_ds_mstep + d41_ds_vscan + d41_ds_vfin (it replaces d41x_gather +
// d41x_plain_gemv + d41_ds_vpart + d41_ds_vrank + d41_ds_vmerge; at TP > 1 d41_ds_vfin becomes d41_ds_vrank, the
// exchange and d41_ds_vmerge):
// * d41_ds_mstep_lN (N = 2 for 1-2 lanes, 4 for 3-4; registers scale with it): every CTA reads the lanes' Markov embedding rows
//   e_l = markov_embed[ids[step][l]] (256 bf16; CTA 0 also writes them into the confidence head's input, cin columns
//   d .. d + 256 of row `step`); a CTA owns 256 vocab columns of this rank's Markov head (8 warps x 32 columns, 8 per
//   batch with their loads issued together); per column the warp computes mh[v] . e_l with d41x_plain_gemv's
//   arithmetic (lane k = 8 lane + j, fmaf over j; the xor tree 16 .. 1 as a reduce-scatter that pairs the same
//   operands), adds it to the logits row (logits[l][step][v] += bias, one f32 add) and writes each id's Gumbel score
//   of the draft draw (x + g, the vocab pass's arithmetic) to a scratch row. L2: the CTA order alternates by step
//   (ascending, then descending), so a step starts on the columns the previous one read last (TP1 / TP2 shards are
//   larger than L2), and the Markov head is read with an evict_last policy.
// * d41_ds_vscan: d41_ds_vpart's DRAFT-mode records per (lane, 2,048-id block), the scores read back.
// * d41_ds_vfin (TP1): d41_ds_vrank's block merge and d41_ds_vmerge's PICK in one launch.
// d41_ds_conf — the confidence head: one warp per row, the plain GEMV's per-row arithmetic (it ran one warp for
//   ALL rows of its single output column).
// d41_ds_plain_r{16,24,32} — a plain BF16 / F16 GEMV that keeps up to 32 rows in registers: one pass over the
//   matrix (d41x_plain_gemv re-reads it for every 8 rows); per-row arithmetic identical.
#pragma once

#define D41_DS_MS_COLS 256                       // vocab columns per CTA
#define D41_DS_MS_SUB (2048 / D41_DS_MS_COLS)   // CTAs per 2,048-id vocab block (vocab::CHUNK)
#define D41_DS_MS_R 256                          // Markov rank
#define D41_DS_MS_LANES 4                        // lanes per call (instances l1 / l2 / l4)

struct d41_ds_mstep_args {
    const __nv_bfloat16* me;        // Markov embedding [vocab][256] (replicated)
    const void* mh;                 // Markov head rows of this rank [vlocal][256] (bf16 / f16: mh_code)
    const int* table;               // the chain inputs [6][nlanes]: row `step`
    __nv_bfloat16* cin;             // confidence input: row (l, step) columns d .. d + 256 at cin + l * cin_ls + step * ld + d
    float* logits;                  // row (l, step) at logits + l * lane_stride + step * row_stride
    long long lane_stride, row_stride, cin_ls;
    const d41_ds_lane* lanes;
    float* gsc;                     // [nlanes][vlocal]: the draft draw's Gumbel score x + g of every id (-inf: none)
    int nlanes, step, ld, d;
    int vlocal, v0, mh_code, pad;   // mh_code: the Markov head's dtype, 0 bf16 / 1 f16
};

__device__ __forceinline__ void d41_ds_bf8(uint4 raw, float* w) {
    const unsigned u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) { w[2 * i] = __uint_as_float(u[i] << 16); w[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u); }
}
__device__ __forceinline__ void d41_ds_h8(uint4 raw, float* w) {
    const unsigned u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) { w[2*i] = __half2float(__ushort_as_half((unsigned short)(u[i] & 0xFFFF))); w[2*i+1] = __half2float(__ushort_as_half((unsigned short)(u[i] >> 16))); }
}
// a read-only 16-B load with an L2 cache policy (not volatile: the compiler may batch and schedule these loads)
__device__ __forceinline__ uint4 d41_ds_ld_last(const void* p, unsigned long long pol) {
    uint4 r;
    asm("ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
    return r;
}

// The PICK epilogue of d41_ds_vmerge for row r = lane l at chain row j (nranks = 1: the record is the merged one).
__device__ __forceinline__ void d41_ds_pick(const d41_ds_vrec& m, d41_ds_lane* lanes, int* ids, float* dlp, float* dlse, int nlanes, int l, int j) {
    if (!lanes[l].active) return;
    const bool greedy = lanes[l].inv_t == 0.0f;
    const float lse = d41_ds_lse(m);
    const int pick = greedy ? m.aid : m.gid;
    ids[(j + 1) * nlanes + l] = pick;
    dlp[l * D41_DS_BLOCK + j] = greedy ? 0.0f : __fsub_rn(m.gx, lse);
    dlse[l * D41_DS_BLOCK + j] = lse;
}

// The xor-tree sum of d41x_plain_gemv (v += shfl_xor(v, o), o = 16 .. 1) of 8 columns at once, as a reduce-scatter:
// every addition pairs the same two operands as the tree (IEEE addition commutes), so each column's sum is the
// tree's bits; 10 shuffles instead of 40. On return lane L holds column (L >> 2) & 7 of the 8.
__device__ __forceinline__ float d41_ds_rs8(const float* p, int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float q[4], r[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float snd = b4 ? p[i] : p[i + 4]; q[i] = (b4 ? p[i + 4] : p[i]) + __shfl_xor_sync(0xFFFFFFFFu, snd, 16); }
#pragma unroll
    for (int i = 0; i < 2; ++i) { const float snd = b3 ? q[i] : q[i + 2]; r[i] = (b3 ? q[i + 2] : q[i]) + __shfl_xor_sync(0xFFFFFFFFu, snd, 8); }
    float v = (b2 ? r[1] : r[0]) + __shfl_xor_sync(0xFFFFFFFFu, b2 ? r[0] : r[1], 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    return v;
}

template <int NL>
__device__ __forceinline__ void d41_ds_mstep_body(const d41_ds_mstep_args& a) {
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int nl = a.nlanes;   // <= NL
    const int bid = (a.step & 1) ? (int)gridDim.x - 1 - (int)blockIdx.x : (int)blockIdx.x;
    const int b = bid / D41_DS_MS_SUB, sb = bid % D41_DS_MS_SUB;
    (void)t;
    // 1. the lanes' embedding rows: lane k = 8 * lane + j holds its 8 values (the plain GEMV's x operand)
    float x[NL][8];
#pragma unroll
    for (int l = 0; l < NL; ++l) {
        if (l < nl) {
            const int id = __ldg(a.table + a.step * nl + l);
            const uint4 raw = *(const uint4*)(a.me + (size_t)id * D41_DS_MS_R + lane * 8);
            d41_ds_bf8(raw, x[l]);
            if (bid == 0 && w == 0) *(uint4*)(a.cin + (long long)l * a.cin_ls + (long long)a.step * a.ld + a.d + lane * 8) = raw;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) x[l][j] = 0.0f;
        }
    }
    // 2. 32 columns per warp in batches of 8 (their 8 loads issued together; columns past the shard re-read its last
    //    row and are dropped): the 8 dot products per lane, the tree sums (reduce-scatter), lane L keeps column L
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    const int c0 = b * 2048 + sb * D41_DS_MS_COLS + w * 32;
    float mine[NL];
#pragma unroll
    for (int l = 0; l < NL; ++l) mine[l] = 0.0f;
    const unsigned short* mhp = (const unsigned short*)a.mh + lane * 8;
    for (int c8 = 0; c8 < 32; c8 += 8) {
        if (c0 + c8 >= a.vlocal) break;   // warp-uniform
        uint4 raw[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) raw[u] = d41_ds_ld_last(mhp + (size_t)min(c0 + c8 + u, a.vlocal - 1) * D41_DS_MS_R, pol);
        float pp[NL][8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            float wv[8];
            if (a.mh_code == 0) d41_ds_bf8(raw[u], wv); else d41_ds_h8(raw[u], wv);
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                float acc = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc = fmaf(wv[j], x[l][j], acc);
                pp[l][u] = acc;
            }
        }
#pragma unroll
        for (int l = 0; l < NL; ++l) {
            if (l < nl) {
                const float v = d41_ds_rs8(pp[l], lane);                                // column (lane >> 2) & 7
                const float g = __shfl_sync(0xFFFFFFFFu, v, (lane & 7) << 2);          // column lane & 7
                if ((lane >> 3) == (c8 >> 3)) mine[l] = g;
            }
        }
    }
    // the logits rows += bias (d41x_plain_gemv: o = res; o += y, one f32 add), and each id's Gumbel score of the
    // draft draw (x + g, the vocab pass's own arithmetic, computed here in parallel for the block scan below)
    {
        const int col = c0 + lane;
        if (col < a.vlocal) {
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                if (l < nl) {
                    float* y = a.logits + (long long)l * a.lane_stride + (long long)a.step * a.row_stride + col;
                    float x0 = mine[l] + *y;
                    *y = x0;
                    const d41_ds_lane& L = a.lanes[l];
                    float sc = -INFINITY;
                    if (x0 != x0) x0 = -INFINITY;
                    if (L.active && L.inv_t != 0.0f && x0 != -INFINITY) {
                        const unsigned long long key = d41_ds_key(d41_ds_draft_seed(L), (unsigned long long)((long long)L.wend + 2 + a.step) * 3ull);   // PS1: per-round draft noise (= d41_ds_vpart)
                        sc = __fadd_rn(__fmul_rn(x0, L.inv_t), d41_ds_gumbel(key, a.v0 + col));
                    }
                    a.gsc[(long long)l * a.vlocal + col] = sc;
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) d41_ds_mstep_l2(d41_ds_mstep_args a) { D41_PDL_ENTRY(); d41_ds_mstep_body<2>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_ds_mstep_l4(d41_ds_mstep_args a) { D41_PDL_ENTRY(); d41_ds_mstep_body<4>(a); }

// d41_ds_vscan — the draft rows' vocab pass after d41_ds_mstep: d41_ds_vpart's DRAFT-mode record of (lane, block)
//   with the Gumbel scores read back from d41_ds_mstep (gsc) instead of recomputed; the same thread <-> id map
//   (thread t: ids lo + t, lo + t + 256, ...), per-thread order, butterfly and warp order: the same bits.
struct d41_ds_vscan_args {
    const float* logits;            // row (l, step) at logits + l * lane_stride + step * row_stride
    long long lane_stride, row_stride;
    const float* gsc;               // [nlanes][vlocal]
    const d41_ds_lane* lanes;
    d41_ds_vrec* part;              // [nlanes][nblk]
    int step, vlocal, v0, pad;
};
// grid (nblk, nlanes), D41_DS_THREADS threads.
extern "C" __global__ void __launch_bounds__(D41_DS_THREADS) d41_ds_vscan(d41_ds_vscan_args a) {
    D41_PDL_ENTRY();
    const int b = blockIdx.x, l = blockIdx.y, t = threadIdx.x;
    __shared__ d41_ds_vrec s_w[D41_DS_THREADS / 32];
    const int lo = b * 2048, hi = min(lo + 2048, a.vlocal);
    const float* row = a.logits + (long long)l * a.lane_stride + (long long)a.step * a.row_stride;
    const float* grow = a.gsc + (long long)l * a.vlocal;
    float xv[8], sv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int v = lo + t + k * D41_DS_THREADS;
        xv[k] = v < hi ? row[v] : 0.0f;
        sv[k] = v < hi ? grow[v] : -INFINITY;
    }
    d41_ds_vrec r;
    d41_ds_rec_empty(r);
    if (a.lanes[l].active) {
        const float inv_t = a.lanes[l].inv_t;
        const bool greedy = inv_t == 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int v = lo + t + k * D41_DS_THREADS;
            if (v < hi) {
                float x0 = xv[k];
                if (x0 != x0) x0 = -INFINITY;
                const int id = a.v0 + v;
                if (x0 > r.amax || (x0 == r.amax && id < r.aid)) { r.amax = x0; r.aid = id; }
                if (!greedy && x0 != -INFINITY) {
                    const float xx = __fmul_rn(x0, inv_t);
                    if (xx > r.m) { r.s = __fadd_rn(r.m == -INFINITY ? 0.0f : __fmul_rn(r.s, expf(r.m - xx)), 1.0f); r.m = xx; }
                    else r.s = __fadd_rn(r.s, expf(xx - r.m));
                    const float sc = sv[k];
                    if (sc > r.gmax || (sc == r.gmax && id < r.gid)) { r.gmax = sc; r.gid = id; r.gx = xx; }
                }
            }
        }
    }
    d41_ds_cta_merge(r, s_w);
    if (t == 0) a.part[(long long)l * gridDim.x + b] = r;
}

// d41_ds_vfin — TP1: d41_ds_vrank (blocks merged in ascending order) and d41_ds_vmerge's PICK epilogue in one launch.
struct d41_ds_vfin_args {
    const d41_ds_vrec* part;        // [nlanes][nblk]
    d41_ds_vrec* recs;              // [nlanes] the merged records
    d41_ds_lane* lanes;
    int* ids; float* dlp; float* dlse;
    int nlanes, nblk, step, pick;
};
#define D41_DS_VFIN_MAX (4 * 64)   // lanes x vocab blocks per rank (129,280 / 2,048 at TP1)
// grid (1), 256 threads: the records staged in shared memory (parallel loads), then thread l merges lane l's blocks
// in ascending order.
extern "C" __global__ void __launch_bounds__(256) d41_ds_vfin(d41_ds_vfin_args a) {
    D41_PDL_ENTRY();
    __shared__ d41_ds_vrec s_p[D41_DS_VFIN_MAX];
    const int n = a.nlanes * a.nblk;
    for (int i = threadIdx.x; i < n && i < D41_DS_VFIN_MAX; i += blockDim.x) s_p[i] = a.part[i];
    __syncthreads();
    const int l = threadIdx.x;
    if (l >= a.nlanes) return;
    const d41_ds_vrec* p = s_p + l * a.nblk;
    d41_ds_vrec m = p[0];
    for (int k = 1; k < a.nblk; ++k) d41_ds_rec_merge(m, p[k]);
    a.recs[l] = m;
    if (a.pick) d41_ds_pick(m, a.lanes, a.ids, a.dlp, a.dlse, a.nlanes, l, a.step);
}

struct d41_ds_conf_args {
    const void* w;                  // [1][K] bf16 / f16 (wcode 0 / 1)
    const __nv_bfloat16* x;         // [rows][ldx]
    float* y;                       // [rows] (stride ldy)
    int wcode, K, ldx, ldy, rows, pad;
};
// grid (rows), 32 threads: d41x_plain_gemv's arithmetic for one output column, one warp per row.
extern "C" __global__ void __launch_bounds__(32) d41_ds_conf(d41_ds_conf_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= a.rows) return;
    float acc = 0.f;
    for (int k = lane * 8; k < a.K; k += 256) {
        float w[8];
        const uint4 raw = *(const uint4*)((const char*)a.w + (size_t)k * 2);
        if (a.wcode == 0) d41_ds_bf8(raw, w); else d41_ds_h8(raw, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(w[j], __bfloat162float(a.x[(size_t)r * a.ldx + k + j]), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) a.y[(size_t)r * a.ldy] = acc;
}

struct d41_ds_plain_args {
    const void* w;                  // [nn][K] bf16 / f16 (rows n0.. of the resident tensor)
    const void* x;                  // [rows][ldx] (xcode 0 bf16 / 1 f16 / 2 f32)
    void* y;                        // [rows][ldy] (ycode)
    int wcode, K, nn, xcode, ldx, rows, ycode, ldy, accum, pad;
};
__device__ __forceinline__ float d41_ds_ldx(const void* p, int code, size_t i) {
    if (code == 0) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    if (code == 1) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}
__device__ __forceinline__ void d41_ds_sty(void* p, int code, size_t i, float v) {
    if (code == 0) ((__nv_bfloat16*)p)[i] = __float2bfloat16_rn(v);
    else if (code == 1) ((__half*)p)[i] = __float2half_rn(v);
    else ((float*)p)[i] = v;
}
// grid (ceil(nn / 8)), 256 threads: one warp per output column, all rows (<= RB) in one pass over its weights.
template <int RB>
__device__ __forceinline__ void d41_ds_plain_body(const d41_ds_plain_args& a) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= a.nn) return;
    const int lane = threadIdx.x & 31;
    float acc[RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) acc[i] = 0.f;
    for (int k = lane * 8; k < a.K; k += 256) {
        float w[8];
        const uint4 raw = *(const uint4*)((const char*)a.w + ((size_t)n * a.K + k) * 2);
        if (a.wcode == 0) d41_ds_bf8(raw, w); else d41_ds_h8(raw, w);
        if (a.xcode == 0) {   // bf16 rows: one 16-B load per row (the 8 values the lane's fma chain takes)
#pragma unroll
            for (int i = 0; i < RB; ++i) {
                if (i < a.rows) {
                    float xv[8];
                    d41_ds_bf8(*(const uint4*)((const __nv_bfloat16*)a.x + (size_t)i * a.ldx + k), xv);
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i] = fmaf(w[j], xv[j], acc[i]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < RB; ++i) {
                if (i < a.rows) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i] = fmaf(w[j], d41_ds_ldx(a.x, a.xcode, (size_t)i * a.ldx + k + j), acc[i]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RB; ++i) {
        float v = acc[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if (lane == 0 && i < a.rows) {
            const size_t oi = (size_t)i * a.ldy + n;
            float o = v;
            if (a.accum) o += d41_ds_ldx(a.y, a.ycode, oi);
            d41_ds_sty(a.y, a.ycode, oi, o);
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) d41_ds_plain_r8(d41_ds_plain_args a) { D41_PDL_ENTRY(); d41_ds_plain_body<8>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_ds_plain_r16(d41_ds_plain_args a) { D41_PDL_ENTRY(); d41_ds_plain_body<16>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_ds_plain_r24(d41_ds_plain_args a) { D41_PDL_ENTRY(); d41_ds_plain_body<24>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_ds_plain_r32(d41_ds_plain_args a) { D41_PDL_ENTRY(); d41_ds_plain_body<32>(a); }
