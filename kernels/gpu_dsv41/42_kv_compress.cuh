// d41_kc_compress / d41_kc_stash (leg KC): the ratio-r compressor and its frontier ring.
//
// model.py Compressor (429-485) + _compress_kv (740-764), for decode (1 row), verify (k rows) and
// chunked prefill (any rows), all one code path:
//   * ratio r >= 2: kv / score are the f32 outputs of the fp32 wkv / wgate GEMMs of THIS call's rows
//     [p0, p0 + rows). A group g covers positions [g r, g r + r). Group g is emitted by this call iff
//     its last position falls inside the call: g in [p0 / r, (p0 + rows) / r - 1]. Its member rows
//     come from the call's projections (q >= p0) or from the frontier ring (q < p0, written by an
//     earlier call; slot = q % R, R = r + 5 rows). pooled = sum_m kv_m * softmax_m(score) per channel
//     (fp32, ascending m), latent = RMSNorm(bf16(pooled)).
//   * ratio 1: latent = RMSNorm(wkv(x)) with the bf16 GEMM output; group g = position g.
//   * the pre-RoPE latent goes to `latent_out` (the index-K projection reads it), then RoPE on the
//     last 64 dims at position g r (a latent stands for the first token of its group), bf16 round,
//     fp4_act_quant(16, e4m3 scales) and the 288-byte record store at main-KV row g.
// d41_kc_stash then copies the last min(rows, R) projection rows into the ring (a SEPARATE launch,
// after the compress kernel: the compress blocks of the first group read the ring the stash overwrites).
#pragma once

__device__ __forceinline__ void d41_kc_pool_acc(float& acc, float kv, float score, float mx, float den) {
    const float w = __fdiv_rn(expf(__fsub_rn(score, mx)), den);
    acc = __fadd_rn(acc, __fmul_rn(kv, w));
}

struct d41_kc_compress_args {
    const float* kv;            // [rows][512] f32 (ratio >= 2)
    const float* score;         // [rows][512] f32 (ratio >= 2)
    const __nv_bfloat16* kv16;  // [rows][512] bf16 (ratio 1)
    const float* ring_kv;       // [R][512] f32 frontier (ratio >= 2)
    const float* ring_score;    // [R][512] f32 frontier (ratio >= 2)
    const void* norm_w;         // [512]
    const float* inv_freq;      // [32] rotary frequencies of this layer class
    __nv_bfloat16* latent_out;  // [groups][512] pre-RoPE normed latent (nullable)
    unsigned char* kv_main;     // main-KV cache: 288-byte record per group, indexed by the absolute group (nullable)
    long long p0;               // absolute position of the call's first row
    int rows, ratio, ring_rows, g_first, n_groups;
    unsigned norm_fmt;
    float eps;
    int* err;                   // nullable; 1 = ring missing, 2 = bad geometry
};

extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41_kc_compress(d41_kc_compress_args a) {
    D41_PDL_ENTRY();
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int j = blockIdx.x;
    if (j >= a.n_groups) return;
    const int g = a.g_first + j;
    const int r = a.ratio;
    __shared__ float s_red[4];
    float v[4];
    if (r >= 2) {
        if (a.ring_rows <= 0 || r > 16) { if (t == 0 && a.err) *a.err = 2; return; }
        const long long q0 = (long long)g * r;
        float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, den[4] = {0.f, 0.f, 0.f, 0.f};
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float* sp;
            if (q >= a.p0) sp = a.score + (size_t)(q - a.p0) * D41_KC_DIM;
            else { if (!a.ring_score) { if (t == 0 && a.err) *a.err = 1; return; } sp = a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM; }
            const float4 x = *(const float4*)(sp + 4 * t);
            mx[0] = fmaxf(mx[0], x.x); mx[1] = fmaxf(mx[1], x.y); mx[2] = fmaxf(mx[2], x.z); mx[3] = fmaxf(mx[3], x.w);
        }
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float* sp = q >= a.p0 ? a.score + (size_t)(q - a.p0) * D41_KC_DIM : a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM;
            const float4 x = *(const float4*)(sp + 4 * t);
            den[0] = __fadd_rn(den[0], expf(__fsub_rn(x.x, mx[0]))); den[1] = __fadd_rn(den[1], expf(__fsub_rn(x.y, mx[1])));
            den[2] = __fadd_rn(den[2], expf(__fsub_rn(x.z, mx[2]))); den[3] = __fadd_rn(den[3], expf(__fsub_rn(x.w, mx[3])));
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float *sp, *kp;
            if (q >= a.p0) { sp = a.score + (size_t)(q - a.p0) * D41_KC_DIM; kp = a.kv + (size_t)(q - a.p0) * D41_KC_DIM; }
            else { sp = a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM; kp = a.ring_kv + (size_t)(q % a.ring_rows) * D41_KC_DIM; }
            const float4 x = *(const float4*)(sp + 4 * t);
            const float4 k = *(const float4*)(kp + 4 * t);
            d41_kc_pool_acc(acc[0], k.x, x.x, mx[0], den[0]); d41_kc_pool_acc(acc[1], k.y, x.y, mx[1], den[1]);
            d41_kc_pool_acc(acc[2], k.z, x.z, mx[2], den[2]); d41_kc_pool_acc(acc[3], k.w, x.w, mx[3], den[3]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = d41_rbf(acc[c]);
    } else {
        if (g < a.p0 || g >= a.p0 + a.rows) { if (t == 0 && a.err) *a.err = 2; return; }
        const __nv_bfloat16* p = a.kv16 + (size_t)(g - a.p0) * D41_KC_DIM + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = d41_bf(p[c]);
    }
    // RMSNorm(512): var = mean(x^2), x * rsqrt(var + eps), weight * x, bf16
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) ss = __fadd_rn(ss, __fmul_rn(v[c], v[c]));
    ss = d41_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(s_red[0], s_red[1]), __fadd_rn(s_red[2], s_red[3]));
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)D41_KC_DIM), a.eps));
    float y[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) y[c] = d41_rbf(__fmul_rn(d41_load_w(a.norm_w, a.norm_fmt, 4 * t + c), __fmul_rn(v[c], rs)));
    if (a.latent_out) {
        __nv_bfloat16* o = a.latent_out + (size_t)j * D41_KC_DIM + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) o[c] = __float2bfloat16_rn(y[c]);
    }
    if (!a.kv_main) return;
    // RoPE on the last 64 channels at position g * r (threads 112..127 hold them), bf16 rounded
    if (t >= (D41_KC_DIM - D41_KC_ROPE) / 4) {
        const int i0 = (4 * t - (D41_KC_DIM - D41_KC_ROPE)) >> 1;
        float c0, s0, c1, s1;
        d41_kc_rope_cs(a.inv_freq, (long long)g * r, i0, c0, s0);
        d41_kc_rope_cs(a.inv_freq, (long long)g * r, i0 + 1, c1, s1);
        d41_kc_rope_pair(y[0], y[1], c0, s0); d41_kc_rope_pair(y[2], y[3], c1, s1);
#pragma unroll
        for (int c = 0; c < 4; ++c) y[c] = d41_rbf(y[c]);
    }
    // fp4_act_quant(16, e4m3 scales): 4 threads per group
    float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    am = d41_group_max<4>(am);
    am = fmaxf(am, 6.0f * 0.001953125f);
    const unsigned sb = d41_e4m3_byte(__fdiv_rn(am, 6.0f));
    const float sc = d41_e4m3(sb);
    unsigned code[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) code[c] = d41_e2m1_code(d41_clamp6(__fdiv_rn(y[c], sc)));
    unsigned char* rec = a.kv_main + (size_t)g * sizeof(d41_kv_main_row);
    rec[2 * t] = (unsigned char)(code[0] | (code[1] << 4));
    rec[2 * t + 1] = (unsigned char)(code[2] | (code[3] << 4));
    if ((t & 3) == 0) rec[256 + (t >> 2)] = (unsigned char)sb;
}

struct d41_kc_stash_args {
    const float* kv; const float* score;   // [rows][512] f32 projections of the call
    float* ring_kv; float* ring_score;     // [R][512]
    long long p0; int rows, ring_rows;
};
// grid.x = min(rows, R): copies the call's LAST min(rows, R) rows to slots (pos % R).
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41_kc_stash(d41_kc_stash_args a) {
    D41_PDL_ENTRY();
    const int n = a.rows < a.ring_rows ? a.rows : a.ring_rows;
    if ((int)blockIdx.x >= n) return;
    const int row = a.rows - n + blockIdx.x;
    const size_t dst = (size_t)((a.p0 + row) % a.ring_rows) * D41_KC_DIM, src = (size_t)row * D41_KC_DIM;
    *(float4*)(a.ring_kv + dst + 4 * threadIdx.x) = *(const float4*)(a.kv + src + 4 * threadIdx.x);
    *(float4*)(a.ring_score + dst + 4 * threadIdx.x) = *(const float4*)(a.score + src + 4 * threadIdx.x);
}
