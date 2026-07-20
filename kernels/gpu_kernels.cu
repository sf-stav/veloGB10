// Elementwise + recurrent kernels for Qwen3.5-0.8B GPU forward (f32).
// All pointers are device pointers unless noted. Compiled to PTX for sm_121.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---- Build-ID stamp: makes a stale PTX impossible to run silently ----
// build.rs hashes the .cu bytes and passes the result as -DKERNEL_BUILD_ID. GpuModel::load reads this
// global back out of the loaded module and asserts it equals the ID compiled into the BINARY. A fresh
// binary next to old kernels then fails loudly at startup instead of launching a kernel whose ABI it no
// longer agrees with -- which is how we once got CUDA_ERROR_ILLEGAL_ADDRESS out of correct code.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }


#define WARP 32

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---- RMSNorm (Qwen3.5): out = x * rsqrt(mean(x^2)+eps) * (1+w) ----
// one block per vector of length n (n <= 1024, single block reduce)
extern "C" __global__ void rmsnorm_qwen(float* out, const float* x, const float* w, int n, float eps) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < n) ? x[tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)n + eps);
    if (tid < n) out[tid] = x[tid] * inv * (1.0f + w[tid]);
}

// ---- Gated RMSNorm (linear attn): out = rms(x) * w * silu(z) ----
// one block per head; normalize m elements at offset head*m.
extern "C" __global__ void rmsnorm_gated(float* out, const float* x, const float* z, const float* w, int m, float eps) {
    extern __shared__ float s[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    float v = (tid < m) ? x[head * m + tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)m + eps);
    if (tid < m) out[head * m + tid] = x[head * m + tid] * inv * w[tid] * silu_f(z[head * m + tid]);
}

// ---- per-head RMSNorm on q/k: normalize each head's hd-vector with shared weight w[hd] ----
extern "C" __global__ void rmsnorm_perhead(float* out, const float* x, const float* w, int nh, int hd, float eps) {
    // one block per head, blockDim.x >= hd (hd=256)
    int head = blockIdx.x;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < hd) ? x[head * hd + tid] : 0.0f;
    v = v * v;
    s[tid] = v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) out[head * hd + tid] = x[head * hd + tid] * inv * (1.0f + w[tid]);
}

// ---- silu(gate)*up for MLP ----
extern "C" __global__ void silu_mul(float* out, const float* gate, const float* up, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) out[i] = silu_f(gate[i]) * up[i];
}

// ---- residual add: out = a + b ----
extern "C" __global__ void add_residual(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---- apply sigmoid gate to attention output (in place) ----
extern "C" __global__ void sigmoid_gate(float* attn, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) attn[i] *= sigmoid_f(gate[i]);
}

// ---- rotate_half RoPE on first rdim of each head (in place) ----
// x layout: [nh, hd]; operates per head, first rdim dims.
extern "C" __global__ void rope_rot_half(float* x, const float* cos, const float* sin, int nh, int hd, int rdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * (rdim / 2);
    if (idx >= total) return;
    int pair = idx % (rdim / 2);
    int head = idx / (rdim / 2);
    int half = rdim / 2;
    int base = head * hd;
    float x1 = x[base + pair];
    float x2 = x[base + pair + half];
    float c = cos[pair];
    float s = sin[pair];
    x[base + pair] = x1 * c - x2 * s;
    x[base + pair + half] = x2 * c + x1 * s;
}

// ---- GQA attention single-token decode ----
// q: [nh, hd]; k_cache/v_cache: [nkv, kv_stride, hd]; valid positions = pos_count (<= kv_stride).
// out: [nh, hd]; one block per query head, blockDim.x = hd (256).
extern "C" __global__ void gqa_attention(float* out, const float* q,
                                         const float* k_cache, const float* v_cache,
                                         int pos_count, int kv_stride, int nh, int nkv, int hd, float scale) {
    int qh = blockIdx.x;
    int kvh = qh / (nh / nkv);
    int d = threadIdx.x; // 0..hd-1
    extern __shared__ float sh[];
    float* scores = sh;          // pos_count
    float* red = sh + pos_count; // blockDim.x reduce scratch
    const float* qv = q + qh * hd;

    // scores[t] = scale * sum_d q[d]*k[t,d]
    for (int t = 0; t < pos_count; t++) {
        const float* kv = k_cache + (kvh * kv_stride + t) * hd;
        float dot = qv[d] * kv[d];
        red[d] = dot;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) scores[t] = red[0] * scale;
        __syncthreads();
    }
    float mx = -1e30f;
    if (d == 0) {
        for (int t = 0; t < pos_count; t++) mx = fmaxf(mx, scores[t]);
        red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    if (d == 0) {
        float se = 0.0f;
        for (int t = 0; t < pos_count; t++) { scores[t] = __expf(scores[t] - mx); se += scores[t]; }
        red[0] = se;
    }
    __syncthreads();
    float inv = 1.0f / red[0];
    float acc = 0.0f;
    for (int t = 0; t < pos_count; t++) {
        acc += scores[t] * inv * v_cache[(kvh * kv_stride + t) * hd + d];
    }
    out[qh * hd + d] = acc;
}

// ---- write current k,v into cache at position pos (stride = kv_stride) ----
extern "C" __global__ void write_kv(float* k_cache, float* v_cache,
                                    const float* k_new, const float* v_new,
                                    int pos, int kv_stride, int nkv, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nkv * hd;
    if (idx >= total) return;
    int h = idx / hd;
    int d = idx % hd;
    k_cache[(h * kv_stride + pos) * hd + d] = k_new[idx];
    v_cache[(h * kv_stride + pos) * hd + d] = v_new[idx];
}

// ---- split q_proj output [nh, hd*2] into q[nh,hd] and gate[nh,hd] ----
extern "C" __global__ void split_qgate(float* q, float* gate, const float* qg, int nh, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nh * hd;
    if (idx >= total) return;
    int head = idx / hd;
    int d = idx % hd;
    q[idx] = qg[head * hd * 2 + d];
    gate[idx] = qg[head * hd * 2 + hd + d];
}

// ---- conv1d depthwise causal step (in place): shift state, conv, silu ----
// x: [conv_dim] new sample (in/out); state: [conv_dim, k]; w: [conv_dim, k]
extern "C" __global__ void conv1d_step(float* x, float* state, const float* w, int conv_dim, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float* st = state + c * k;
    for (int j = 1; j < k; j++) st[j - 1] = st[j];
    st[k - 1] = x[c];
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
    x[c] = silu_f(acc);
}

// ---- Gated delta-rule recurrent step (linear attention), one block per head ----
// Inputs (post-conv): qkv: [conv_dim] split q[nh*kd],k[nh*kd],v[nh*vd]; b[nh],a[nh]
// state: [nh, kd, vd]; out: [nh, vd] (core attention output, pre-norm)
// one block per head, blockDim = kd (128). kd==vd here.
extern "C" __global__ void delta_step(float* out,
                                      const float* q_in, const float* k_in, const float* v_in,
                                      const float* b_in, const float* a_in,
                                      float* state,
                                      int nh, int kd, int vd,
                                      const float* a_log, const float* dt_bias) {
    int head = blockIdx.x;
    int a = threadIdx.x; // 0..kd-1
    extern __shared__ float sh[];
    float* Srow = sh;                 // kd
    float* kv_mem = sh + kd;          // vd
    float* vbuf = sh + kd + vd;       // vd
    float* delta = sh + kd + 2 * vd;  // vd
    float* qrow = sh + kd + 3 * vd;   // kd
    float* krow = sh + kd + 3 * vd + kd; // kd

    float* S = state + head * kd * vd; // [kd, vd]
    float beta = sigmoid_f(b_in[head]);
    float sp = a_in[head] + dt_bias[head];
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    float gt = __expf(-__expf(a_log[head]) * sp);

    // q,k: l2norm per head; q *= scale
    float qv = q_in[head * kd + a];
    float kv = k_in[head * kd + a];
    Srow[a] = qv * qv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float qn = rsqrtf(Srow[0] + 1e-6f);
    __syncthreads(); // ensure all read Srow[0] before we reuse Srow
    qv *= qn;
    Srow[a] = kv * kv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float kn = rsqrtf(Srow[0] + 1e-6f);
    __syncthreads();
    kv *= kn;                       // normalized k
    float scale = 1.0f / sqrtf((float)kd);
    qv *= scale;
    qrow[a] = qv;
    krow[a] = kv;                   // store normalized k
    __syncthreads();

    // S *= gt
    for (int bb = 0; bb < vd; bb++) S[a * vd + bb] *= gt;
    __syncthreads();

    // kv_mem[bb] = sum_a S[a,bb]*k[a]  (normalized k)
    int bb = a;
    float km = 0.0f;
    for (int aa = 0; aa < kd; aa++) km += S[aa * vd + bb] * krow[aa];
    kv_mem[bb] = km;
    vbuf[bb] = v_in[head * vd + bb];
    __syncthreads();

    // delta[bb] = (v[bb]-kv_mem[bb])*beta
    delta[bb] = (vbuf[bb] - kv_mem[bb]) * beta;
    __syncthreads();

    // S[a,bb] += k[a]*delta[bb]  (normalized k)
    float kk = krow[a];
    for (int bbb = 0; bbb < vd; bbb++) S[a * vd + bbb] += kk * delta[bbb];
    __syncthreads();

    // out[bb] = sum_a S[a,bb]*q[a]
    float o = 0.0f;
    for (int aa = 0; aa < kd; aa++) o += S[aa * vd + bb] * qrow[aa];
    out[head * vd + bb] = o;
}

// argmax is done on device (two-pass) below.

// ---- bf16<->f32 conversions ----
extern "C" __global__ void f32tobf16(__nv_bfloat16* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
extern "C" __global__ void bf16tof32(float* dst, const __nv_bfloat16* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// ---- fused: residual += mixer, then out = rmsnorm(residual, w) * (1+w) ----
// one block over vector length n.
extern "C" __global__ void fused_residual_rmsnorm(float* out, float* residual, const float* mixer,
                                                  const float* w, int n, float eps) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float v = (tid < n) ? (residual[tid] + mixer[tid]) : 0.0f;
    residual[tid] = v;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    if (tid < n) out[tid] = v * inv * (1.0f + w[tid]);
}

// ---- argmax pass 1: each block reduces a chunk of logits -> (val,idx) into globals ----
extern "C" __global__ void argmax_pass1(int* out_idx, float* out_val, const float* logits, int n) {
    int bid = blockIdx.x;
    int bs = blockDim.x;
    extern __shared__ float sv[];
    int* si = (int*)(sv + bs);
    int tid = threadIdx.x;
    int gid = bid * bs + tid;
    float val = -1e30f; int idx = -1;
    if (gid < n) { val = logits[gid]; idx = gid; }
    sv[tid] = val; si[tid] = idx;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) { if (sv[tid + s2] > sv[tid]) { sv[tid] = sv[tid + s2]; si[tid] = si[tid + s2]; } }
        __syncthreads();
    }
    if (tid == 0) { out_val[bid] = sv[0]; out_idx[bid] = si[0]; }
}

// ---- argmax pass 2: reduce per-block winners (m of them) into one index ----
extern "C" __global__ void argmax_pass2(int* token, const int* idxs, const float* vals, int m) {
    extern __shared__ float sv[];
    int* si = (int*)(sv + blockDim.x);
    int tid = threadIdx.x;
    float val = -1e30f; int idx = -1;
    if (tid < m) { val = vals[tid]; idx = idxs[tid]; }
    sv[tid] = val; si[tid] = idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) { if (sv[tid + s2] > sv[tid]) { sv[tid] = sv[tid + s2]; si[tid] = si[tid + s2]; } }
        __syncthreads();
    }
    if (tid == 0) token[0] = si[0];
}

