// gpu_inkling2/30_attnres.cuh — Block Attention Residual (K1). Section of gpu_inkling2.cu.
//
// For each token row: values V_j = rms(v_j) (unweighted, FIXED eps 1e-6, D6) over v = [blocks..., partial]
// (nb + 1 <= 6 entries), and for each query q_i (nq <= 4): out_i = softmax_j(q_i . V_j / sqrt(D)) . V.
// One pass computes ||v_j||^2 and q_i . v_j together (q . rms(v) = rinv * q . v), one pass writes the
// outputs. Used for the layer inputs (1 query sliding, 3 full), the MLP input, the final input and the
// cross-pass MessageAttnRes (4 queries). fp32 throughout. Block reductions in a fixed order (01).
//
// grid (M), block 256, D % 256 == 0 (D = 2304: 9 per thread).

#define IKA_AR_MAXV 6
#define IKA_AR_MAXQ 4

struct InkArArgs {
    const float* v[IKA_AR_MAXV];    // [M][D] each: blocks then the partial last; nv = nb + 1
    int nv;
    const float* q; int nq;         // [nq][D]
    float* out[IKA_AR_MAXQ];        // [M][D] each
    int D;
};

extern "C" __global__ __launch_bounds__(256) void ink_attnres(const InkArArgs a) {
    INK2_PDL_ENTRY();
    __shared__ float red[8 * IKA_AR_MAXV * (1 + IKA_AR_MAXQ)];
    __shared__ float tot[IKA_AR_MAXV * (1 + IKA_AR_MAXQ)];
    __shared__ float pw[IKA_AR_MAXQ][IKA_AR_MAXV];
    const int t = blockIdx.x;
    const int D = a.D;
    constexpr int NV = IKA_AR_MAXV * (1 + IKA_AR_MAXQ);
    float part[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) part[i] = 0.f;
    for (int c = threadIdx.x; c < D; c += 256) {
        float qv[IKA_AR_MAXQ];
        #pragma unroll
        for (int i = 0; i < IKA_AR_MAXQ; ++i) qv[i] = i < a.nq ? a.q[(size_t)i * D + c] : 0.f;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) {
            const float x = j < a.nv ? a.v[j][(size_t)t * D + c] : 0.f;
            part[j * 5] = fmaf(x, x, part[j * 5]);
            #pragma unroll
            for (int i = 0; i < IKA_AR_MAXQ; ++i) part[j * 5 + 1 + i] = fmaf(qv[i], x, part[j * 5 + 1 + i]);
        }
    }
    ika_bsum_tot<NV, 8>(part, red, tot);
    #pragma unroll
    for (int i = 0; i < NV; ++i) part[i] = tot[i];
    float rinv[IKA_AR_MAXV];
    #pragma unroll
    for (int j = 0; j < IKA_AR_MAXV; ++j) rinv[j] = j < a.nv ? rsqrtf(part[j * 5] / (float)D + 1e-6f) : 0.f;
    if (threadIdx.x == 0) {
        const float sc = rsqrtf((float)D);
        #pragma unroll
        for (int i = 0; i < IKA_AR_MAXQ; ++i) {
            float s[IKA_AR_MAXV], m = -INFINITY;
            #pragma unroll
            for (int j = 0; j < IKA_AR_MAXV; ++j) {
                s[j] = j < a.nv ? part[j * 5 + 1 + i] * rinv[j] * sc : -INFINITY;
                m = fmaxf(m, s[j]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int j = 0; j < IKA_AR_MAXV; ++j) { s[j] = j < a.nv ? expf(s[j] - m) : 0.f; sum += s[j]; }
            #pragma unroll
            for (int j = 0; j < IKA_AR_MAXV; ++j) pw[i][j] = s[j] / sum * rinv[j];   // weight of the RAW value
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < D; c += 256) {
        float x[IKA_AR_MAXV];
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) x[j] = j < a.nv ? a.v[j][(size_t)t * D + c] : 0.f;
        #pragma unroll
        for (int i = 0; i < IKA_AR_MAXQ; ++i) {
            float o = 0.f;
            #pragma unroll
            for (int j = 0; j < IKA_AR_MAXV; ++j) o = fmaf(pw[i][j], x[j], o);
            if (i < a.nq) a.out[i][(size_t)t * D + c] = o;
        }
    }
}

// ink_attnres_cl — the same math, one row spread over a CLUSTER of 8 CTAs (D / 8 channels each, one channel per
// thread: block D / 8 threads, D = 2304 -> 288). Each CTA reduces its 30 partials (||v_j||^2, q_i . v_j) in a
// fixed order into shared memory; after a cluster barrier every CTA reads the 8 partial sets through distributed
// shared memory in ascending rank order (identical sums everywhere), forms the softmax weights and writes its own
// channels. A single-CTA row (ink_attnres) is latency-bound at decode (one SM streams 6 x 2304 x 4 B twice);
// here 8 SMs share it. grid (8, M), cluster (8, 1, 1). Same sums as ink_attnres up to order; rows independent.
// Dispatch (k1.rs attnres): this kernel for M <= 16 (decode / verify), ink_attnres for prefill chunks.
#include <cooperative_groups.h>
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(288) ink_attnres_cl(const InkArArgs a) {
    INK2_PDL_ENTRY();
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    constexpr int NV = IKA_AR_MAXV * (1 + IKA_AR_MAXQ);
    __shared__ float red[9 * NV];
    __shared__ float mine[NV];
    __shared__ float tot[NV];
    __shared__ float pw[IKA_AR_MAXQ][IKA_AR_MAXV];
    const int t = blockIdx.y, D = a.D;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;      // one channel per thread
    float x[IKA_AR_MAXV], qv[IKA_AR_MAXQ];
    #pragma unroll
    for (int j = 0; j < IKA_AR_MAXV; ++j) x[j] = j < a.nv ? a.v[j][(size_t)t * D + c] : 0.f;
    #pragma unroll
    for (int i = 0; i < IKA_AR_MAXQ; ++i) qv[i] = i < a.nq ? a.q[(size_t)i * D + c] : 0.f;
    float part[NV];
    #pragma unroll
    for (int j = 0; j < IKA_AR_MAXV; ++j) {
        part[j * 5] = x[j] * x[j];
        #pragma unroll
        for (int i = 0; i < IKA_AR_MAXQ; ++i) part[j * 5 + 1 + i] = qv[i] * x[j];
    }
    ika_bsum_tot<NV, 9>(part, red, mine);
    cl.sync();
    if (threadIdx.x < NV) {           // value i: the 8 CTAs' partials in ascending rank order
        float s = 0.f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) s += cl.map_shared_rank(mine, r)[threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
    if (threadIdx.x < IKA_AR_MAXQ) {  // query i
        const int i = threadIdx.x;
        const float sc = rsqrtf((float)D);
        float rinv[IKA_AR_MAXV], s[IKA_AR_MAXV], m = -INFINITY;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) {
            rinv[j] = j < a.nv ? rsqrtf(tot[j * 5] / (float)D + 1e-6f) : 0.f;
            s[j] = j < a.nv ? tot[j * 5 + 1 + i] * rinv[j] * sc : -INFINITY;
            m = fmaxf(m, s[j]);
        }
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) { s[j] = j < a.nv ? expf(s[j] - m) : 0.f; sum += s[j]; }
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) pw[i][j] = s[j] / sum * rinv[j];
    }
    cl.sync();   // pw ready; and no CTA leaves while a peer may still read its `mine`
    #pragma unroll
    for (int i = 0; i < IKA_AR_MAXQ; ++i) {
        float o = 0.f;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) o = fmaf(pw[i][j], x[j], o);
        if (i < a.nq) a.out[i][(size_t)t * D + c] = o;
    }
}

// Drop 3 (#4a): Fused cluster AttnRes + RMSNorm for decode/verify widths (zero intermediate global write/read of
// ws.ar[..], eliminates 60 ink_rmsnorm launches per step across TP1-TP8). Exact per-row cluster arithmetic ->
// verify == decode bit-exact. grid (8, M), cluster (8, 1, 1), block 288 (D = 2304).
struct InkArNormArgs {
    InkArArgs ar;           // nv, v[6], q, nq (1..3); ar.out unused
    const float* nw[3];     // per-query RMSNorm weight [D] (nullable)
    uint16_t* ob[3];        // per-query bf16 output [M][D] (nullable)
    float* of[3];           // per-query fp32 output [M][D] (nullable)
    float eps;
};
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(288) ink_attnres_cl_norm(const InkArNormArgs p) {
    INK2_PDL_ENTRY();
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const InkArArgs& a = p.ar;
    constexpr int NV = IKA_AR_MAXV * (1 + IKA_AR_MAXQ);
    __shared__ float red[9 * NV];
    __shared__ float mine[NV];
    __shared__ float tot[NV];
    __shared__ float pw[IKA_AR_MAXQ][IKA_AR_MAXV];
    const int t = blockIdx.y, D = a.D;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float x[IKA_AR_MAXV], qv[IKA_AR_MAXQ];
    #pragma unroll
    for (int j = 0; j < IKA_AR_MAXV; ++j) x[j] = j < a.nv ? a.v[j][(size_t)t * D + c] : 0.f;
    #pragma unroll
    for (int i = 0; i < IKA_AR_MAXQ; ++i) qv[i] = i < a.nq ? a.q[(size_t)i * D + c] : 0.f;
    float part[NV];
    #pragma unroll
    for (int j = 0; j < IKA_AR_MAXV; ++j) {
        part[j * 5] = x[j] * x[j];
        #pragma unroll
        for (int i = 0; i < IKA_AR_MAXQ; ++i) part[j * 5 + 1 + i] = qv[i] * x[j];
    }
    ika_bsum_tot<NV, 9>(part, red, mine);
    cl.sync();
    if (threadIdx.x < NV) {
        float s = 0.f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) s += cl.map_shared_rank(mine, r)[threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
    if (threadIdx.x < IKA_AR_MAXQ) {
        const int i = threadIdx.x;
        const float sc = rsqrtf((float)D);
        float rinv[IKA_AR_MAXV], s[IKA_AR_MAXV], m = -INFINITY;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) {
            rinv[j] = j < a.nv ? rsqrtf(tot[j * 5] / (float)D + 1e-6f) : 0.f;
            s[j] = j < a.nv ? tot[j * 5 + 1 + i] * rinv[j] * sc : -INFINITY;
            m = fmaxf(m, s[j]);
        }
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) { s[j] = j < a.nv ? expf(s[j] - m) : 0.f; sum += s[j]; }
        #pragma unroll
        for (int j = 0; j < IKA_AR_MAXV; ++j) pw[i][j] = s[j] / sum * rinv[j];
    }
    cl.sync();
    float o[3] = {0.f, 0.f, 0.f};
    float ss[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (i < a.nq) {
            float v = 0.f;
            #pragma unroll
            for (int j = 0; j < IKA_AR_MAXV; ++j) v = fmaf(pw[i][j], x[j], v);
            o[i] = v;
            ss[i] = v * v;
        }
    }
    ika_bsum_tot<3, 9>(ss, red, mine);
    cl.sync();
    if (threadIdx.x < 3) {
        float s = 0.f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) s += cl.map_shared_rank(mine, r)[threadIdx.x];
        tot[threadIdx.x] = rsqrtf(s / (float)D + p.eps);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (i < a.nq) {
            const float y = p.nw[i] ? p.nw[i][c] * (o[i] * tot[i]) : (o[i] * tot[i]);
            if (p.ob[i]) p.ob[i][(size_t)t * D + c] = ika_f2b(y);
            if (p.of[i]) p.of[i][(size_t)t * D + c] = y;
        }
    }
    cl.sync();
}
