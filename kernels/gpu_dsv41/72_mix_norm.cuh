// 72_mix_norm.cuh — RMSNorm, generic dim (<= 6144, multiple of 4): y = bf16(w * (x * rsqrt(mean(x^2) + eps))), fp32 inside.
// model.py 281-293 (attn_norm, ffn_norm, q_norm, kv_norm, the compressor / indexer norms, the head norm, DSpark norms).
// x bf16 [rows][x_ld]; w f32 / bf16 / null (unweighted); eps is a parameter (the model's rms_norm_eps is 1e-20).
#pragma once
struct d41_rmsnorm_args {
    const d41_bf16* x; long long x_ld;
    const void* w;
    d41_bf16* y; long long y_ld;
    unsigned w_fmt; int dim; int rows;
    float eps; float inv_dim;
};
extern "C" __global__ void __launch_bounds__(256) d41_rmsnorm(d41_rmsnorm_args a) {
    D41_PDL_ENTRY();
    d41_norm_body<false, false>(a.x, a.x_ld, nullptr, 0, a.w, a.w_fmt, a.y, a.y_ld, nullptr, 0, a.dim, a.eps, a.inv_dim);
}
