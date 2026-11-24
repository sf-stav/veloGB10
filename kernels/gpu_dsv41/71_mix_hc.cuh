// 71_mix_hc.cuh — mHC (hc 4): the K-sliced mix partial, the finish (sum of slots + Sinkhorn), hc_pre(+norm), hc_post.
// Reference: model.py 940-996 (hc_mixes / hc_pre / hc_post / the delayed pre-mix), kernel.py 407-474.
#pragma once

// ---- d41_mix_partial: the 24 dot products of hc_fn [24, K] with a K slice of the stream, plus the slice's sum of squares.
// x: bf16 stream rows (row r at x + r*x_ld; the 4 copies contiguous: flat index c*5120 + d). The slice is [k0, k0 + klen) of
// that flat index; fn points at column k0 of hc_fn (f32, row m at fn + m*fn_ld). One block = 512 consecutive k of the slice
// (blockIdx.x) x up to rg rows (blockIdx.y). The block's 25 values per row go to
//   part[blockIdx.x * chunk_stride + row * row_stride + m]       (m 0..23 dots, m = 24 sum of squares)
// and are summed in a fixed order later (d41_mix_finish), so a TP rank's partial rides the sublayer reduce unchanged.
struct d41_mix_partial_args {
    const d41_bf16* x; long long x_ld;
    const float* fn; long long fn_ld;
    float* part; long long chunk_stride; long long row_stride;
    // optional rank reduction ("last block sums"): the LAST block of a row group sums that group's chunk partials in a fixed
    // order (4 contiguous chunk ranges, then 0+1+2+3) into rank_part[row * rp_row_stride + m] and re-arms its counter. This is
    // the 25-f32-per-row value a TP rank sends as the rider; counters[blockIdx.y] must be zero at the first launch.
    float* rank_part; long long rp_row_stride; unsigned* counters;
    int k0; int klen; int rows; int rg;
};
__device__ __forceinline__ void d41_mix_partial_body(const d41_mix_partial_args& a) {
    __shared__ float sm[24][4][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kk = blockIdx.x * 512 + tid * 4;
    if (blockIdx.x * 512 >= a.klen) return;
    const bool in_k = kk < a.klen;
    float4 f[24];
#pragma unroll
    for (int m = 0; m < 24; ++m)
        f[m] = in_k ? __ldg(reinterpret_cast<const float4*>(a.fn + (long long)m * a.fn_ld + kk)) : make_float4(0.f, 0.f, 0.f, 0.f);
    const int r0 = blockIdx.y * a.rg;
    const int rn = min(a.rg, a.rows - r0);
#pragma unroll 2
    for (int ri = 0; ri < rn; ++ri) {
        float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
        if (in_k) {
            const d41_bf16* xr = a.x + (long long)(r0 + ri) * a.x_ld + a.k0 + kk;
            uint2 raw = __ldg(reinterpret_cast<const uint2*>(xr));
            x0 = d41_bf2f_u16((unsigned short)(raw.x & 0xFFFFu)); x1 = d41_bf2f_u16((unsigned short)(raw.x >> 16));
            x2 = d41_bf2f_u16((unsigned short)(raw.y & 0xFFFFu)); x3 = d41_bf2f_u16((unsigned short)(raw.y >> 16));
        }
        float v[32];
#pragma unroll
        for (int m = 0; m < 24; ++m) v[m] = fmaf(f[m].w, x3, fmaf(f[m].z, x2, fmaf(f[m].y, x1, f[m].x * x0)));
        v[24] = fmaf(x3, x3, fmaf(x2, x2, fmaf(x1, x1, x0 * x0)));
#pragma unroll
        for (int m = 25; m < 32; ++m) v[m] = 0.f;
        sm[ri][warp][lane] = d41_rs32(v, lane);
    }
    __syncthreads();
    for (int idx = tid; idx < rn * D41_HC_PART; idx += 128) {
        const int ri = idx / D41_HC_PART, m = idx - ri * D41_HC_PART;
        const float s = ((sm[ri][0][m] + sm[ri][1][m]) + sm[ri][2][m]) + sm[ri][3][m];
        a.part[(long long)blockIdx.x * a.chunk_stride + (long long)(r0 + ri) * a.row_stride + m] = s;
    }
    if (a.counters) {
        __shared__ unsigned last;
        __threadfence();
        __syncthreads();
        if (tid == 0) last = atomicAdd(&a.counters[blockIdx.y], 1u) == gridDim.x - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        const int nch = gridDim.x, per = (nch + 3) >> 2, sub = tid & 3;
        const int c0 = sub * per, c1 = min(nch, c0 + per);
        // 4 outputs per thread-group per trip (their loads are all in flight together; the tail is latency-bound)
        for (int it = 0; it * 128 < rn * D41_HC_PART; ++it) {          // uniform trip count: the shuffles need every lane
            float vals[4][10];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int o = it * 128 + u * 32 + (tid >> 2);
                const int oc = o < rn * D41_HC_PART ? o : 0;
                const int ri = oc / D41_HC_PART, m = oc - ri * D41_HC_PART;
                const float* base = a.part + (long long)(r0 + ri) * a.row_stride + m;
#pragma unroll
                for (int q = 0; q < 10; ++q) vals[u][q] = (c0 + q < c1) ? __ldcg(base + (long long)(c0 + q) * a.chunk_stride) : 0.f;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int o = it * 128 + u * 32 + (tid >> 2);
                const bool ok = o < rn * D41_HC_PART;
                const int oc = ok ? o : 0;
                const int ri = oc / D41_HC_PART, m = oc - ri * D41_HC_PART;
                float acc = 0.f;
                bool first = true;
#pragma unroll
                for (int q = 0; q < 10; ++q) { if (c0 + q < c1) { acc = first ? vals[u][q] : acc + vals[u][q]; first = false; } }
                float s1 = __shfl_down_sync(0xffffffffu, acc, 1), s2 = __shfl_down_sync(0xffffffffu, acc, 2), s3 = __shfl_down_sync(0xffffffffu, acc, 3);
                if (ok && sub == 0) a.rank_part[(long long)(r0 + ri) * a.rp_row_stride + m] = ((acc + s1) + s2) + s3;
            }
        }
        if (tid == 0) a.counters[blockIdx.y] = 0u;
    }
}
extern "C" __global__ void __launch_bounds__(128) d41_mix_partial(d41_mix_partial_args a) {
    D41_PDL_ENTRY();
    d41_mix_partial_body(a);
}

// leg PA1 (the hc side stream at TP2, PLAN/DSV41_RX6_REPORT.md 2.2): BOTH ranks' K slices of the mix in one launch, each
// half exactly the launch its rank runs (same chunks, same last-block sums): blockIdx.z = the half (the rank whose slice it
// is), h[z] = that rank's arguments (its k0 / fn columns, its own counters and rank_part). d41_mix_finish with nslots = 2
// then sums rank 0's + rank 1's values, the one IEEE add the world-2 all-reduce performed on the rider.
struct d41_mix_partial2_args { d41_mix_partial_args h[2]; };
extern "C" __global__ void __launch_bounds__(128) d41_mix_partial2(const __grid_constant__ d41_mix_partial2_args a) {
    D41_PDL_ENTRY();
    d41_mix_partial_body(a.h[blockIdx.z]);
}
// leg P6-5 (the hc side stream at TP4): all 4 ranks' K slices in one launch; d41_mix_finish with nslots = 4 then sums
// (p0 + p2) + (p1 + p3), matching the world-4 all-reduce's exact association bit for bit.
struct d41_mix_partial4_args { d41_mix_partial_args h[4]; };
extern "C" __global__ void __launch_bounds__(128) d41_mix_partial4(const __grid_constant__ d41_mix_partial4_args a) {
    D41_PDL_ENTRY();
    d41_mix_partial_body(a.h[blockIdx.z]);
}
// Phase A (TP8 SP prefill + hc side stream): all 8 ranks' K slices in one launch; d41_mix_finish with nslots = 8 sums
// ((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7)), matching the world-8 all-reduce's exact association bit for bit.
struct d41_mix_partial8_args { d41_mix_partial_args h[8]; };
extern "C" __global__ void __launch_bounds__(128) d41_mix_partial8(const __grid_constant__ d41_mix_partial8_args a) {
    D41_PDL_ENTRY();
    d41_mix_partial_body(a.h[blockIdx.z]);
}

// ---- d41_mix_finish: sum the partial slots in slot order, scale by the stream's rsqrt, hc_split_sinkhorn.
// part[s * slot_stride + row * row_stride + m], s = 0 .. nslots-1 (rank-major, chunk-minor; ANY fixed order is deterministic).
// out: one record of 24 f32 per row: pre[0..3], post[4..7], comb[8..23] (comb[j][k] at 8 + 4j + k).
struct d41_mix_finish_args {
    const float* part; long long slot_stride; long long row_stride;
    const float* scale;      // [3]
    const float* base;       // [24]
    float* out; long long out_ld;
    int nslots; int rows;
    float inv_k;             // 1 / (4 * 5120), the mean over the whole flattened stream
    float eps_norm;          // rms_norm_eps (1e-20)
    float eps_hc;            // hc_eps (1e-6)
    int sinkhorn_iters;      // 20
};
__device__ __forceinline__ float d41_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
extern "C" __global__ void __launch_bounds__(128) d41_mix_finish(d41_mix_finish_args a) {
    D41_PDL_ENTRY();
    const int lane = threadIdx.x & 31, hb = lane & 16, li = lane & 15;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5) * 2 + (lane >> 4);
    const int rc = min(r, a.rows - 1);
    const float* pr = a.part + (long long)rc * a.row_stride;
    float da = 0.f, db = 0.f, ss = 0.f;
    if (a.nslots == 4) {
        // leg P6-5: match the world-4 all-reduce's pairwise association (p0 + p2) + (p1 + p3) + nslots=1 finish (0.f + sum).
        const float* p0 = pr;
        const float* p1 = pr + a.slot_stride;
        const float* p2 = pr + 2 * a.slot_stride;
        const float* p3 = pr + 3 * a.slot_stride;
        da = 0.f + ((p0[li] + p2[li]) + (p1[li] + p3[li]));
        if (li < 8) db = 0.f + ((p0[16 + li] + p2[16 + li]) + (p1[16 + li] + p3[16 + li]));
        ss = 0.f + ((p0[24] + p2[24]) + (p1[24] + p3[24]));
    } else if (a.nslots == 8) {
        // Phase A (TP8): match the world-8 all-reduce's canonical 3-round tree association + nslots=1 finish (0.f + sum).
        const float* p0 = pr;
        const float* p1 = pr + a.slot_stride;
        const float* p2 = pr + 2 * a.slot_stride;
        const float* p3 = pr + 3 * a.slot_stride;
        const float* p4 = pr + 4 * a.slot_stride;
        const float* p5 = pr + 5 * a.slot_stride;
        const float* p6 = pr + 6 * a.slot_stride;
        const float* p7 = pr + 7 * a.slot_stride;
        da = 0.f + (((p0[li] + p2[li]) + (p4[li] + p6[li])) + ((p1[li] + p3[li]) + (p5[li] + p7[li])));
        if (li < 8) {
            const int j = 16 + li;
            db = 0.f + (((p0[j] + p2[j]) + (p4[j] + p6[j])) + ((p1[j] + p3[j]) + (p5[j] + p7[j])));
        }
        ss = 0.f + (((p0[24] + p2[24]) + (p4[24] + p6[24])) + ((p1[24] + p3[24]) + (p5[24] + p7[24])));
    } else {
        int s = 0;
        for (; s + 8 <= a.nslots; s += 8) {          // loads hoisted 8 deep (latency), adds in slot order
            float pa[8], pb[8], ps[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float* p = pr + (long long)(s + q) * a.slot_stride;
                pa[q] = p[li]; pb[q] = li < 8 ? p[16 + li] : 0.f; ps[q] = p[24];
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) { da += pa[q]; db += pb[q]; ss += ps[q]; }
        }
        for (; s < a.nslots; ++s) {
            const float* p = pr + (long long)s * a.slot_stride;
            da += p[li];
            if (li < 8) db += p[16 + li];
            ss += p[24];
        }
    }
    const float rs = rsqrtf(ss * a.inv_k + a.eps_norm);
    const float ma = da * rs, mb = db * rs;                       // mixes[li], mixes[16 + li]
    const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
    float outv = 0.f;                                              // pre / post value for li < 8
    if (li < 4) outv = d41_sigmoid(fmaf(ma, s0, a.base[li])) + a.eps_hc;
    else if (li < 8) outv = 2.0f * d41_sigmoid(fmaf(ma, s1, a.base[li]));
    const float sa = __shfl_sync(0xffffffffu, ma, hb + 8 + (li & 7));
    const float sb = __shfl_sync(0xffffffffu, mb, hb + (li & 7));
    float c = fmaf(li < 8 ? sa : sb, s2, a.base[8 + li]);         // comb[j = li >> 2][k = li & 3]
    // softmax over k, + eps
    float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    c = expf(c - mx);
    // 4-lane sums as the reference kernel's xor butterfly: (c0 + c2) + (c1 + c3)
#define D41_ROWSUM(v) ({ float t_ = (v) + __shfl_xor_sync(0xffffffffu, (v), 2); t_ + __shfl_xor_sync(0xffffffffu, t_, 1); })
#define D41_COLSUM(v) ({ float t_ = (v) + __shfl_xor_sync(0xffffffffu, (v), 8); t_ + __shfl_xor_sync(0xffffffffu, t_, 4); })
    c = c / D41_ROWSUM(c) + a.eps_hc;
    c = c / (D41_COLSUM(c) + a.eps_hc);
    for (int it = 1; it < a.sinkhorn_iters; ++it) {
        c = c / (D41_ROWSUM(c) + a.eps_hc);
        c = c / (D41_COLSUM(c) + a.eps_hc);
    }
#undef D41_ROWSUM
#undef D41_COLSUM
    if (r < a.rows) {
        float* o = a.out + (long long)r * a.out_ld;
        if (li < 8) o[li] = outv;
        o[8 + li] = c;
    }
}

// ---- d41_hc_pre_norm: y = bf16(sum_j pre[j] * x_j) (fp32, j ascending), then RMSNorm(y) in the same pass (hc_pre -> attn_norm /
// ffn_norm / the head norm). y_raw (optional) gets the un-normed bf16 y (DSpark's confidence head reads it); y_norm (optional).
// pre: f32, row r at pre + r*pre_stride (the delayed pre-mix: a mix record of the PREVIOUS sublayer; layer 0 passes a constant
// one-hot [1,0,0,0] with pre_stride 0). w: norm weight (f32 / bf16) or null = unweighted.
struct d41_hc_pre_norm_args {
    const d41_bf16* x; long long x_ld;
    const float* pre; long long pre_stride;
    const void* w;
    d41_bf16* y_norm; long long yn_ld;
    d41_bf16* y_raw; long long yr_ld;
    unsigned w_fmt; int dim; int rows;
    float eps; float inv_dim;
};
// the shared body: IN = hc stream (4 copies) or a plain row
// PW (leg PA1): the pre-mix and the norm weights are loaded BEFORE the PDL wait (the caller does not wait first): pre is the
// record of a mix whose producer is at least two launches back on this stream or joined by an event (never this kernel's
// PDL predecessor), the weights are constants; x is read after the wait. Same operations, same order, same bits.
template <bool HC, bool PW>
__device__ __forceinline__ void d41_norm_body(const d41_bf16* x, long long x_ld, const float* pre, long long pre_stride, const void* w, unsigned w_fmt,
                                              d41_bf16* y_norm, long long yn_ld, d41_bf16* y_raw, long long yr_ld, int dim, float eps, float inv_dim) {
    __shared__ float sm[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    const d41_bf16* xr = x + (long long)row * x_ld;
    float y[6][4];
    float wv[6][4];                    // the norm weights, fetched with x (one memory latency, not two: PK1)
    float ss = 0.f;
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    if (HC) { const float* pp = pre + (long long)row * pre_stride; p0 = pp[0]; p1 = pp[1]; p2 = pp[2]; p3 = pp[3]; }
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = (tid + 256 * i) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) wv[i][e] = (w && y_norm && d < dim) ? d41_ldw(w, w_fmt, d + e) : 1.0f;
    }
    if constexpr (PW) D41_PDL_ENTRY();
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = (tid + 256 * i) * 4;
        if (d < dim) {
            float t[4];
            if (HC) {
                float c[4][4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint2 raw = __ldg(reinterpret_cast<const uint2*>(xr + (long long)j * dim + d));
                    c[j][0] = d41_bf2f_u16((unsigned short)(raw.x & 0xFFFFu)); c[j][1] = d41_bf2f_u16((unsigned short)(raw.x >> 16));
                    c[j][2] = d41_bf2f_u16((unsigned short)(raw.y & 0xFFFFu)); c[j][3] = d41_bf2f_u16((unsigned short)(raw.y >> 16));
                }
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float s = __fmul_rn(p0, c[0][e]);
                    s = __fadd_rn(s, __fmul_rn(p1, c[1][e]));
                    s = __fadd_rn(s, __fmul_rn(p2, c[2][e]));
                    s = __fadd_rn(s, __fmul_rn(p3, c[3][e]));
                    t[e] = s;
                }
            } else {
                uint2 raw = __ldg(reinterpret_cast<const uint2*>(xr + d));
                t[0] = d41_bf2f_u16((unsigned short)(raw.x & 0xFFFFu)); t[1] = d41_bf2f_u16((unsigned short)(raw.x >> 16));
                t[2] = d41_bf2f_u16((unsigned short)(raw.y & 0xFFFFu)); t[3] = d41_bf2f_u16((unsigned short)(raw.y >> 16));
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) { d41_bf16 b = __float2bfloat16_rn(t[e]); y[i][e] = d41_bf2f(b); ss = fmaf(y[i][e], y[i][e], ss); }
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) y[i][e] = 0.f;
        }
    }
    float v[1] = {ss};
    d41_block_sum256<1>(v, sm);
    const float rstd = rsqrtf(v[0] * inv_dim + eps);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = (tid + 256 * i) * 4;
        if (d < dim) {
            if (y_raw) {
                d41_bf16* o = y_raw + (long long)row * yr_ld + d;
#pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = __float2bfloat16_rn(y[i][e]);
            }
            if (y_norm) {
                d41_bf16* o = y_norm + (long long)row * yn_ld + d;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float xn = y[i][e] * rstd;
                    o[e] = __float2bfloat16_rn(wv[i][e] * xn);
                }
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) d41_hc_pre_norm(d41_hc_pre_norm_args a) {
    D41_PDL_ENTRY();
    d41_norm_body<true, false>(a.x, a.x_ld, a.pre, a.pre_stride, a.w, a.w_fmt, a.y_norm, a.yn_ld, a.y_raw, a.yr_ld, a.dim, a.eps, a.inv_dim);
}
// leg PA1: d41_hc_pre_norm with the pre record and the norm weights loaded before the PDL wait (theirs: `_hc_finish_s`
// issues its inputs before ACQBULK). Only for a pre record produced >= 2 launches back or across an event join (TpModel::body).
extern "C" __global__ void __launch_bounds__(256) d41_hc_pre_norm_pw(d41_hc_pre_norm_args a) {
    d41_norm_body<true, true>(a.x, a.x_ld, a.pre, a.pre_stride, a.w, a.w_fmt, a.y_norm, a.yn_ld, a.y_raw, a.yr_ld, a.dim, a.eps, a.inv_dim);
}

// ---- d41_hc_post: out_k = bf16(post[k] * x + sum_j comb[j][k] * resid_j)  (fp32, products rounded separately, j ascending).
// xs: the sublayer output (bf16, dim); resid / out: the hc stream (4 copies of dim at stride dim). out MAY alias resid (each
// thread reads its 4 copies of its element before writing them). post / comb: from the sublayer's own mix record.
struct d41_hc_post_args {
    const d41_bf16* xs; long long xs_ld;
    const d41_bf16* resid; long long r_ld;
    const float* post; long long post_stride;     // post[0..3] of row r at post + r * post_stride
    const float* comb; long long comb_stride;     // comb[j*4+k] of row r at comb + r * comb_stride
    d41_bf16* out; long long o_ld;
    int dim; int rows;
};
// F32 (leg PF1, the external reference's `_hc_post_only` reads the summed partials itself): xs is the REDUCED fp32
// sublayer frame (row r at xs32 + r * x32_ld), rounded to bf16 here exactly as d41f_add_round(s = 0) did into the bf16 xs
// this kernel read before; the same bits, one launch fewer per sublayer.
template <bool F32>
__device__ __forceinline__ void d41_hc_post_body(const d41_hc_post_args& a, const float* xs32, long long x32_ld) {
    const int row = blockIdx.y;
    const int d = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (d >= a.dim) return;
    const float* po = a.post + (long long)row * a.post_stride;
    const float* cb = a.comb + (long long)row * a.comb_stride;
    float x[4], r[4][4];
    if constexpr (F32) {
        const float4 v = *reinterpret_cast<const float4*>(xs32 + (long long)row * x32_ld + d);
        x[0] = __bfloat162float(__float2bfloat16_rn(v.x)); x[1] = __bfloat162float(__float2bfloat16_rn(v.y));
        x[2] = __bfloat162float(__float2bfloat16_rn(v.z)); x[3] = __bfloat162float(__float2bfloat16_rn(v.w));
    } else {
        uint2 raw = __ldg(reinterpret_cast<const uint2*>(a.xs + (long long)row * a.xs_ld + d));
        x[0] = d41_bf2f_u16((unsigned short)(raw.x & 0xFFFFu)); x[1] = d41_bf2f_u16((unsigned short)(raw.x >> 16));
        x[2] = d41_bf2f_u16((unsigned short)(raw.y & 0xFFFFu)); x[3] = d41_bf2f_u16((unsigned short)(raw.y >> 16));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint2 raw = *reinterpret_cast<const uint2*>(a.resid + (long long)row * a.r_ld + (long long)j * a.dim + d);
        r[j][0] = d41_bf2f_u16((unsigned short)(raw.x & 0xFFFFu)); r[j][1] = d41_bf2f_u16((unsigned short)(raw.x >> 16));
        r[j][2] = d41_bf2f_u16((unsigned short)(raw.y & 0xFFFFu)); r[j][3] = d41_bf2f_u16((unsigned short)(raw.y >> 16));
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float pk = po[k];
        const float c0 = cb[k], c1 = cb[4 + k], c2 = cb[8 + k], c3 = cb[12 + k];
        float o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float s = __fmul_rn(c0, r[0][e]);
            s = __fadd_rn(s, __fmul_rn(c1, r[1][e]));
            s = __fadd_rn(s, __fmul_rn(c2, r[2][e]));
            s = __fadd_rn(s, __fmul_rn(c3, r[3][e]));
            o[e] = __fadd_rn(__fmul_rn(pk, x[e]), s);
        }
        d41_bf16 b0 = __float2bfloat16_rn(o[0]), b1 = __float2bfloat16_rn(o[1]), b2 = __float2bfloat16_rn(o[2]), b3 = __float2bfloat16_rn(o[3]);
        uint2 w;
        w.x = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
        w.y = (unsigned)__bfloat16_as_ushort(b2) | ((unsigned)__bfloat16_as_ushort(b3) << 16);
        *reinterpret_cast<uint2*>(a.out + (long long)row * a.o_ld + (long long)k * a.dim + d) = w;
    }
}
extern "C" __global__ void __launch_bounds__(256) d41_hc_post(d41_hc_post_args a) {
    D41_PDL_ENTRY();
    d41_hc_post_body<false>(a, nullptr, 0);
}
struct d41_hc_post32_args { d41_hc_post_args p; const float* xs32; long long x32_ld; };
extern "C" __global__ void __launch_bounds__(256) d41_hc_post_f32(d41_hc_post32_args a) {
    D41_PDL_ENTRY();
    d41_hc_post_body<true>(a.p, a.xs32, a.x32_ld);
}
