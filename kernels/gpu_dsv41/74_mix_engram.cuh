// 74_mix_engram.cuh — the Engram gate (model.py 355-365), per (token, hc copy c) over dim:
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps);  dot = sum(h * (wq * wk) * key) * rstd * dim^-0.5
//   gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot));      out = bf16(h + gate * value)   (the same value row on all 4 copies)
// h = the hc stream (bf16, 4 copies of dim), key / value = the bf16 wkv output (key copy c at kv + c*dim, value at kv + 4*dim),
// wq / wk f32 or bf16 [4][dim]. gate_on[row] == 0 shuts the gate (image spans); null = all on. out MAY alias h.
#pragma once
struct d41_engram_gate_args {
    const d41_bf16* h; long long h_ld;
    const d41_bf16* kv; long long kv_ld;
    const void* wq; const void* wk;
    const unsigned char* gate_on;
    d41_bf16* out; long long o_ld;
    unsigned wq_fmt; unsigned wk_fmt; int dim; int rows;
    float eps; float inv_dim; float inv_sqrt_dim; float clamp_min;
};
extern "C" __global__ void __launch_bounds__(256) d41_engram_gate(d41_engram_gate_args a) {
    D41_PDL_ENTRY();
    __shared__ float sm[3 * 8];
    const int c = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    const d41_bf16* hr = a.h + (long long)row * a.h_ld + (long long)c * a.dim;
    const d41_bf16* kr = a.kv + (long long)row * a.kv_ld + (long long)c * a.dim;
    const d41_bf16* vr = a.kv + (long long)row * a.kv_ld + (long long)D41_HC * a.dim;
    float hv[6][4], kv[6][4];
    float s[3] = {0.f, 0.f, 0.f};      // sum h^2, sum key^2, sum h*w*key
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = (tid + 256 * i) * 4;
        if (d < a.dim) {
            uint2 rh = *reinterpret_cast<const uint2*>(hr + d), rk = __ldg(reinterpret_cast<const uint2*>(kr + d));
            hv[i][0] = d41_bf2f_u16((unsigned short)(rh.x & 0xFFFFu)); hv[i][1] = d41_bf2f_u16((unsigned short)(rh.x >> 16));
            hv[i][2] = d41_bf2f_u16((unsigned short)(rh.y & 0xFFFFu)); hv[i][3] = d41_bf2f_u16((unsigned short)(rh.y >> 16));
            kv[i][0] = d41_bf2f_u16((unsigned short)(rk.x & 0xFFFFu)); kv[i][1] = d41_bf2f_u16((unsigned short)(rk.x >> 16));
            kv[i][2] = d41_bf2f_u16((unsigned short)(rk.y & 0xFFFFu)); kv[i][3] = d41_bf2f_u16((unsigned short)(rk.y >> 16));
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const long long wi = (long long)c * a.dim + d + e;
                const float w = __fmul_rn(d41_ldw(a.wq, a.wq_fmt, wi), d41_ldw(a.wk, a.wk_fmt, wi));
                s[0] = fmaf(hv[i][e], hv[i][e], s[0]);
                s[1] = fmaf(kv[i][e], kv[i][e], s[1]);
                s[2] += __fmul_rn(__fmul_rn(hv[i][e], w), kv[i][e]);
            }
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) { hv[i][e] = 0.f; kv[i][e] = 0.f; }
        }
    }
    d41_block_sum256<3>(s, sm);
    const float rstd = rsqrtf(s[0] * a.inv_dim + a.eps) * rsqrtf(s[1] * a.inv_dim + a.eps);
    const float dot = (s[2] * rstd) * a.inv_sqrt_dim;
    const float mag = sqrtf(fmaxf(fabsf(dot), a.clamp_min));
    float gate = d41_sigmoid(copysignf(mag, dot));
    if (a.gate_on && !a.gate_on[row]) gate = 0.f;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = (tid + 256 * i) * 4;
        if (d < a.dim) {
            uint2 rv = __ldg(reinterpret_cast<const uint2*>(vr + d));
            float vv[4] = {d41_bf2f_u16((unsigned short)(rv.x & 0xFFFFu)), d41_bf2f_u16((unsigned short)(rv.x >> 16)),
                           d41_bf2f_u16((unsigned short)(rv.y & 0xFFFFu)), d41_bf2f_u16((unsigned short)(rv.y >> 16))};
            d41_bf16 b[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) b[e] = __float2bfloat16_rn(__fadd_rn(hv[i][e], __fmul_rn(gate, vv[e])));
            uint2 w;
            w.x = (unsigned)__bfloat16_as_ushort(b[0]) | ((unsigned)__bfloat16_as_ushort(b[1]) << 16);
            w.y = (unsigned)__bfloat16_as_ushort(b[2]) | ((unsigned)__bfloat16_as_ushort(b[3]) << 16);
            *reinterpret_cast<uint2*>(a.out + (long long)row * a.o_ld + (long long)c * a.dim + d) = w;
        }
    }
}
