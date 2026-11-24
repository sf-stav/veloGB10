// gpu_inkling2/10_attn.cuh — attention (K1). Section of gpu_inkling2.cu.
//
// Geometry: 24 Q heads, 8 KV heads (group 3), head_dim 128, d_rel 16, every layer. NO RoPE.
// Per decoder-layer call on M new rows at positions p0 .. p0+M-1 (one sequence):
//
//   1. ink_proj_* (05) -> Y[M][ldy] fp32 holding, at column offsets: q (24x128), gate (24x128), r (24x16),
//      sink (24), k (8x128, pre-conv), v (8x128, pre-conv), and on full layers kd / vd (8x4 dynamic taps).
//   2. ink_attn_prep: Q = L2Clip(q * q_scale, 22.6) (* LogScale on full layers), R = L2Clip(r * r_scale, 8)
//      (* LogScale), K = L2Clip(conv(k)), V = rms(conv(v)) (unweighted, eps); K / V written to the f16 cache
//      at slot (pos & cap_mask). D1 / D2: K has NO learned scale and LogScale never touches K.
//   3. ink_attn: for each (row, head): softmax over [q.k * 128^-0.5 + bias(q - k), sink] in fp32, value
//      sum without the sink, times sigmoid(gate) -> bf16 [M][3072] (the wo input). bias(d) =
//      r . rel_proj[:, d] for 0 <= d < ext (sliding: window 1024 = ext; full: ext 1024), else 0.
//      Split-KV: grid.x = chunks of `chunk` keys aligned to ABSOLUTE key index (chunk depends only on p0, so
//      row 0 of an M-row call sees exactly the M = 1 call's chunks and 64-key subtiles; subtiles where a
//      query has no unmasked key are skipped without touching its running state). nsplit == 1: finalize in
//      place (also the prefill path: grid.z = 16-row tiles). nsplit > 1: partials (m, l, acc[128]) per
//      (row, head, chunk), merged by ink_attn_combine in ascending chunk order.
//
// KV cache: per logical slot, K and V f16 [cap][8][128], cap a power of two, slot = pos & cap_mask. Sliding
// slots may be a ring (cap >= window + rows of the largest call - 1); full slots cap >= max context.
// Conv input ring: per logical slot and conv, fp32 [R][C], R = 32 (power of two), row = pos & (R - 1). See
// 20_conv.cuh for the speculative-commit design (commit = advance pos; zero copies).

#define IKA_HD 128
#define IKA_DREL 16
#define IKA_SUB 64           // keys per subtile
#define IKA_KSTR 130         // padded K row stride in smem (halves): 65 words -> conflict-free row reads
#define IKA_QMAX 48          // 16 rows x 3 heads per CTA
#define IKA_QPW 6            // queries per warp (8 warps)

struct InkPrepArgs {
    const float* Y; int ldy;
    int oq, orr, ok, ov, okd, ovd;      // column offsets in Y (okd / ovd < 0: static convs)
    const float* q_scale;               // [24*128]
    const float* r_scale;               // [24*16]
    float qk_clip, rel_clip;
    int full; float log_alpha, log_floor;
    const float* cw_k; const float* cw_v;   // conv taps [4 lag][1024] (lag 0 = current token)
    float conv_div;                     // 1/sqrt(5) static, 1/3 dynamic
    float* ring_k; float* ring_v; int ring_mask; int ring_write;
    __half* kc; __half* vc; unsigned cap_mask;
    float eps;
    float* Qf; float* Rf;               // [M][24][128], [M][24][16]
    int p0, M;
};

__device__ __forceinline__ float ika_log_scale(int pos, float alpha, float floor_) {
    return 1.0f + alpha * logf(fmaxf((float)(pos + 1) / floor_, 1.0f));
}

// grid (M), block 1024: warps 0..23 Q / R heads, warps 24..31 K / V heads. The body is shared with the
// lane-batched decode entry (85_lanes.cuh), which calls it with t = 0 on a per-row copy of the arguments.
__device__ __forceinline__ void ika_attn_prep_body(const InkPrepArgs& a, const int t) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = a.p0 + t;
    const float* y = a.Y + (size_t)t * a.ldy;
    const float ls = a.full ? ika_log_scale(p, a.log_alpha, a.log_floor) : 1.0f;
    if (warp < 24) {
        const int h = warp;
        const float4 qv = *reinterpret_cast<const float4*>(y + a.oq + h * IKA_HD + 4 * lane);
        const float4 sc = *reinterpret_cast<const float4*>(a.q_scale + h * IKA_HD + 4 * lane);
        float q0 = qv.x * sc.x, q1 = qv.y * sc.y, q2 = qv.z * sc.z, q3 = qv.w * sc.w;
        const float ss = ika_wsum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        const float f = fminf(1.0f, a.qk_clip / (sqrtf(ss) + 1e-6f)) * ls;
        *reinterpret_cast<float4*>(a.Qf + ((size_t)t * 24 + h) * IKA_HD + 4 * lane) = make_float4(q0 * f, q1 * f, q2 * f, q3 * f);
        float r = lane < IKA_DREL ? y[a.orr + h * IKA_DREL + lane] * a.r_scale[h * IKA_DREL + lane] : 0.f;
        const float rs = ika_wsum(r * r);
        const float fr = fminf(1.0f, a.rel_clip / (sqrtf(rs) + 1e-6f)) * ls;
        if (lane < IKA_DREL) a.Rf[((size_t)t * 24 + h) * IKA_DREL + lane] = r * fr;
        return;
    }
    const int g = warp - 24;
    #pragma unroll
    for (int kv = 0; kv < 2; ++kv) {
        const int off = kv ? a.ov : a.ok;
        const int od = kv ? a.ovd : a.okd;
        const float* cw = kv ? a.cw_v : a.cw_k;
        const float* ring = kv ? a.ring_v : a.ring_k;
        float o[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = g * IKA_HD + 4 * lane + i;
            const float x0 = y[off + c];
            float acc = 0.f;
            #pragma unroll
            for (int lag = 0; lag < 4; ++lag) {
                float xv;
                if (lag == 0) xv = x0;
                else if (t - lag >= 0) xv = a.Y[(size_t)(t - lag) * a.ldy + off + c];
                else if (p - lag >= 0) xv = ring[(size_t)((p - lag) & a.ring_mask) * 1024 + c];
                else xv = 0.f;
                float w = cw[lag * 1024 + c];
                if (od >= 0) w += y[od + g * 4 + lag];
                acc = fmaf(xv, w, acc);
            }
            o[i] = (acc + x0) * a.conv_div;
        }
        const float ss = ika_wsum(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
        const float f = kv ? rsqrtf(ss / (float)IKA_HD + a.eps) : fminf(1.0f, a.qk_clip / (sqrtf(ss) + 1e-6f));
        __half* cache = (kv ? a.vc : a.kc) + (size_t)(p & a.cap_mask) * 1024 + g * IKA_HD + 4 * lane;
        *reinterpret_cast<__half2*>(cache) = __floats2half2_rn(o[0] * f, o[1] * f);
        *reinterpret_cast<__half2*>(cache + 2) = __floats2half2_rn(o[2] * f, o[3] * f);
    }
    if (a.ring_write) {
        // the call's conv INPUTS go to the ring (rows p0..p0+M-1; no reader of this call touches them: M <= R-3)
        #pragma unroll
        for (int kv = 0; kv < 2; ++kv) {
            const int off = kv ? a.ov : a.ok;
            float* ring = kv ? a.ring_v : a.ring_k;
            const int c = g * IKA_HD + 4 * lane;
            *reinterpret_cast<float4*>(ring + (size_t)(p & a.ring_mask) * 1024 + c) =
                *reinterpret_cast<const float4*>(y + off + c);
        }
    }
}
extern "C" __global__ __launch_bounds__(1024) void ink_attn_prep(const InkPrepArgs a) { INK2_PDL_ENTRY(); ika_attn_prep_body(a, blockIdx.x); }

struct InkAttnArgs {
    const float* Qf; const float* Rf;
    const __half* kc; const __half* vc; unsigned cap_mask;
    const float* proj; int ext;          // rel_logits_proj [16][ext] fp32
    int window;                          // 0 = full attention
    int p0, M;
    int chunk, c0;                       // split: chunk keys per CTA (multiple of 64), first chunk index
    const float* Y; int ldy, os, og;     // sink / gate columns (finalize)
    uint16_t* out;                       // bf16 [M][24*128]
    float* pm; float* pl; float* pacc;   // partials [M][24][nsplit] (+[128])
};

// grid (nsplit, 8 kv heads, ceil(M / 16)), block 256, dynamic smem = 4*(qcap*144 + 16*80) + 64*(130+128)*2
// bytes, qcap = 3*min(16, M) (src/inkling2/gpu/k1.rs attn_smem).
// Body shared with the lane-batched decode entry (85_lanes.cuh): `nsplit` / `bx` / `bz` stand for gridDim.x /
// blockIdx.x / blockIdx.z of the plain launch.
__device__ __forceinline__ void ika_attn_body(const InkAttnArgs& a, const int nsplit, const int bx, const int bz) {
    extern __shared__ __align__(16) unsigned char smraw[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = blockIdx.y;
    const int t0 = bz * 16;
    const int mt = min(16, a.M - t0);
    const int nq = 3 * mt;
    const int qcap = 3 * min(16, a.M);                                // same for every CTA of the launch
    float* sQ = reinterpret_cast<float*>(smraw);                     // [qcap][128]
    float* sR = sQ + qcap * IKA_HD;                                   // [qcap][16]
    float* sP = sR + qcap * IKA_DREL;                                 // [16][80] proj columns
    __half* sK = reinterpret_cast<__half*>(sP + IKA_DREL * 80);       // [64][130]
    __half* sV = sK + IKA_SUB * IKA_KSTR;                             // [64][128]

    for (int i = threadIdx.x; i < nq * IKA_HD; i += blockDim.x) {
        const int qi = i / IKA_HD, d = i % IKA_HD;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        sQ[i] = a.Qf[((size_t)t * 24 + h) * IKA_HD + d];
    }
    for (int i = threadIdx.x; i < nq * IKA_DREL; i += blockDim.x) {
        const int qi = i / IKA_DREL, d = i % IKA_DREL;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        sR[i] = a.Rf[((size_t)t * 24 + h) * IKA_DREL + d];
    }
    const int P0 = a.p0 + t0, P1 = P0 + mt - 1;
    const int kmin = a.window ? max(0, P0 - a.window + 1) : 0;
    const long long cs = (long long)(a.c0 + bx) * a.chunk;
    const int lo = (int)max((long long)kmin, cs);
    const int hi = (int)min((long long)P1 + 1, cs + a.chunk);       // exclusive
    const float scale = rsqrtf((float)IKA_HD);

    float m_run[IKA_QPW], l_run[IKA_QPW], acc[IKA_QPW][4];
    #pragma unroll
    for (int j = 0; j < IKA_QPW; ++j) {
        m_run[j] = -INFINITY; l_run[j] = 0.f;
        #pragma unroll
        for (int d = 0; d < 4; ++d) acc[j][d] = 0.f;
    }
    for (int kb = (lo / IKA_SUB) * IKA_SUB; kb < hi; kb += IKA_SUB) {
        __syncthreads();
        // K / V subtile (rows outside [lo, hi) are zero: never read uninitialised cache)
        for (int i = threadIdx.x; i < IKA_SUB * 64; i += blockDim.x) {
            const int r = i >> 6, w2 = i & 63;     // 64 half2 per row
            const int kp = kb + r;
            __half2 kv2 = __float2half2_rn(0.f), vv2 = __float2half2_rn(0.f);
            if (kp >= lo && kp < hi) {
                const size_t base = (size_t)(kp & a.cap_mask) * 1024 + kvh * IKA_HD + 2 * w2;
                kv2 = *reinterpret_cast<const __half2*>(a.kc + base);
                vv2 = *reinterpret_cast<const __half2*>(a.vc + base);
            }
            *reinterpret_cast<__half2*>(sK + r * IKA_KSTR + 2 * w2) = kv2;
            *reinterpret_cast<__half2*>(sV + r * IKA_HD + 2 * w2) = vv2;
        }
        // proj columns for distances dmin .. dmin+79 (dmin = P0 - (kb + 63))
        const int dmin = P0 - (kb + IKA_SUB - 1);
        for (int i = threadIdx.x; i < IKA_DREL * 80; i += blockDim.x) {
            const int dd = i / 80, c = i % 80;
            const int dist = dmin + c;
            sP[i] = (dist >= 0 && dist < a.ext) ? a.proj[(size_t)dd * a.ext + dist] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < IKA_QPW; ++j) {
            const int qi = warp + 8 * j;
            if (qi >= nq) break;
            const int P = P0 + qi / 3;
            const float* q = sQ + qi * IKA_HD;
            const float* rr = sR + qi * IKA_DREL;
            float s[2];
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int r = lane + 32 * u, kp = kb + r;
                const __half2* kr = reinterpret_cast<const __half2*>(sK + r * IKA_KSTR);
                float dot = 0.f;
                #pragma unroll 16
                for (int d2 = 0; d2 < IKA_HD / 2; ++d2) {
                    const float2 kf = __half22float2(kr[d2]);
                    dot = fmaf(q[2 * d2], kf.x, dot);
                    dot = fmaf(q[2 * d2 + 1], kf.y, dot);
                }
                const int dist = P - kp;
                float b = 0.f;
                const int col = dist - dmin;
                #pragma unroll
                for (int dd = 0; dd < IKA_DREL; ++dd) b = fmaf(rr[dd], sP[dd * 80 + col], b);
                const bool ok = kp >= lo && kp < hi && dist >= 0 && (a.window == 0 || dist < a.window);
                s[u] = ok ? fmaf(dot, scale, b) : -INFINITY;
            }
            const float msub = ika_wmax(fmaxf(s[0], s[1]));
            if (msub == -INFINITY) continue;          // nothing visible here: state untouched
            const float mnew = fmaxf(m_run[j], msub);
            const float fold = __expf(m_run[j] - mnew); // m_run = -inf -> 0
            const float e0 = __expf(s[0] - mnew), e1 = __expf(s[1] - mnew);
            l_run[j] = l_run[j] * fold + ika_wsum(e0 + e1);
            m_run[j] = mnew;
            float o0 = acc[j][0] * fold, o1 = acc[j][1] * fold, o2 = acc[j][2] * fold, o3 = acc[j][3] * fold;
            #pragma unroll 8
            for (int r = 0; r < IKA_SUB; ++r) {
                const float pr = __shfl_sync(0xffffffffu, r < 32 ? e0 : e1, r & 31);
                const uint2 vv = *reinterpret_cast<const uint2*>(sV + r * IKA_HD + 4 * lane);
                const float2 va = __half22float2(*reinterpret_cast<const __half2*>(&vv.x));
                const float2 vb = __half22float2(*reinterpret_cast<const __half2*>(&vv.y));
                o0 = fmaf(pr, va.x, o0); o1 = fmaf(pr, va.y, o1); o2 = fmaf(pr, vb.x, o2); o3 = fmaf(pr, vb.y, o3);
            }
            acc[j][0] = o0; acc[j][1] = o1; acc[j][2] = o2; acc[j][3] = o3;
        }
    }
    #pragma unroll
    for (int j = 0; j < IKA_QPW; ++j) {
        const int qi = warp + 8 * j;
        if (qi >= nq) break;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        if (nsplit == 1) {
            const float* y = a.Y + (size_t)t * a.ldy;
            const float sink = y[a.os + h];
            const float m = fmaxf(m_run[j], sink);
            const float fold = __expf(m_run[j] - m);
            const float inv = 1.0f / (l_run[j] * fold + __expf(sink - m));
            const float w = fold * inv;
            if (a.og < 0) {
                float* fp = reinterpret_cast<float*>(a.out) + (size_t)t * 24 * IKA_HD + h * IKA_HD + 4 * lane;
                *reinterpret_cast<float4*>(fp) = make_float4(acc[j][0] * w, acc[j][1] * w, acc[j][2] * w, acc[j][3] * w);
            } else {
                const float4 gt = *reinterpret_cast<const float4*>(y + a.og + h * IKA_HD + 4 * lane);
                uint16_t* o = a.out + (size_t)t * 24 * IKA_HD + h * IKA_HD + 4 * lane;
                uint2 pk;
                pk.x = (uint32_t)ika_f2b(acc[j][0] * w * ika_sig(gt.x)) | ((uint32_t)ika_f2b(acc[j][1] * w * ika_sig(gt.y)) << 16);
                pk.y = (uint32_t)ika_f2b(acc[j][2] * w * ika_sig(gt.z)) | ((uint32_t)ika_f2b(acc[j][3] * w * ika_sig(gt.w)) << 16);
                *reinterpret_cast<uint2*>(o) = pk;
            }
        } else {
            const size_t pi = ((size_t)t * 24 + h) * nsplit + bx;
            if (lane == 0) { a.pm[pi] = m_run[j]; a.pl[pi] = l_run[j]; }
            *reinterpret_cast<float4*>(a.pacc + pi * IKA_HD + 4 * lane) = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
    }
}
extern "C" __global__ __launch_bounds__(256) void ink_attn(const InkAttnArgs a) { INK2_PDL_ENTRY(); ika_attn_body(a, gridDim.x, blockIdx.x, blockIdx.z); }

// grid (M, 24), block 128: merge the nsplit partials of (row, head) in ascending chunk order with the sink,
// times sigmoid(gate) -> bf16. Empty chunks (l == 0) are skipped (they hold no key of this row).
__device__ __forceinline__ void ika_attn_combine_body(const InkAttnArgs& a, const int nsplit, const int t) {
    const int h = blockIdx.y, d = threadIdx.x;
    const float* y = a.Y + (size_t)t * a.ldy;
    const float sink = y[a.os + h];
    const size_t pb = ((size_t)t * 24 + h) * nsplit;
    float m = sink;
    for (int c = 0; c < nsplit; ++c) if (a.pl[pb + c] > 0.f) m = fmaxf(m, a.pm[pb + c]);
    float sum = __expf(sink - m), o = 0.f;
    for (int c = 0; c < nsplit; ++c) {
        const float l = a.pl[pb + c];
        if (!(l > 0.f)) continue;
        const float f = __expf(a.pm[pb + c] - m);
        sum = fmaf(l, f, sum);
        o = fmaf(a.pacc[(pb + c) * IKA_HD + d], f, o);
    }
    if (a.og < 0) {
        reinterpret_cast<float*>(a.out)[(size_t)t * 24 * IKA_HD + h * IKA_HD + d] = o / sum;
    } else {
        a.out[(size_t)t * 24 * IKA_HD + h * IKA_HD + d] = ika_f2b(o / sum * ika_sig(y[a.og + h * IKA_HD + d]));
    }
}
extern "C" __global__ __launch_bounds__(128) void ink_attn_combine(const InkAttnArgs a, int nsplit) { INK2_PDL_ENTRY(); ika_attn_combine_body(a, nsplit, blockIdx.x); }
