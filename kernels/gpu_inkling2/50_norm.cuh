// gpu_inkling2/50_norm.cuh — norms and the embedding (K1). Section of gpu_inkling2.cu.
//
// ink_rmsnorm: y = w * x * rsqrt(mean(x^2) + eps) (w nullable = the unweighted RMSNorm), fp32 in; writes
//   bf16 (the next linear's input: attn_norm / {q,k,v}_input_norm / mlp_norm / final norm) and/or fp32.
//   Each row's sum of squares is reduced in a fixed order (01), so rows are batch invariant.
// ink_embed_norm: h = embed_norm(embed[id]) (bf16 table rows, fp32 out).
// grid (M), block 256.

extern "C" __global__ __launch_bounds__(256) void ink_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
        uint16_t* __restrict__ ob, float* __restrict__ of, int D, float eps) {
    INK2_PDL_ENTRY();
    __shared__ float red[8];
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * D;
    float ss[1] = {0.f};
    for (int c = threadIdx.x; c < D; c += 256) ss[0] = fmaf(xr[c], xr[c], ss[0]);
    ika_bsum<1, 8>(ss, red);
    const float r = rsqrtf(ss[0] / (float)D + eps);
    for (int c = threadIdx.x; c < D; c += 256) {
        const float y = w ? w[c] * (xr[c] * r) : xr[c] * r;
        if (ob) ob[(size_t)t * D + c] = ika_f2b(y);
        if (of) of[(size_t)t * D + c] = y;
    }
}

extern "C" __global__ __launch_bounds__(256) void ink_embed_norm(const int* __restrict__ ids, const uint16_t* __restrict__ emb,
        const float* __restrict__ w, float* __restrict__ out, int D, float eps) {
    INK2_PDL_ENTRY();
    __shared__ float red[8];
    const int t = blockIdx.x;
    const uint16_t* er = emb + (size_t)ids[t] * D;
    float ss[1] = {0.f};
    for (int c = threadIdx.x; c < D; c += 256) { const float v = ika_b2f(er[c]); ss[0] = fmaf(v, v, ss[0]); }
    ika_bsum<1, 8>(ss, red);
    const float r = rsqrtf(ss[0] / (float)D + eps);
    for (int c = threadIdx.x; c < D; c += 256) out[(size_t)t * D + c] = w[c] * (ika_b2f(er[c]) * r);
}

// Drop 3 (#4a): Batched 3-stream RMSNorm (q/k/v attention inputs on loop passes 1 & 2) in ONE launch.
// Bit-identical to 3 sequential ink_rmsnorm launches; cuts 48 launches per decode/verify step.
// grid (M, 1, 3), block 256.
struct InkNorm3Args {
    const float* x[3];
    const float* w[3];
    uint16_t* ob[3];
    float* of[3];
};
extern "C" __global__ __launch_bounds__(256) void ink_rmsnorm_x3(const InkNorm3Args a, int D, float eps) {
    INK2_PDL_ENTRY();
    __shared__ float red[8];
    const int t = blockIdx.x, z = blockIdx.z;
    const float* xr = a.x[z] + (size_t)t * D;
    const float* w = a.w[z];
    uint16_t* ob = a.ob[z];
    float* of = a.of[z];
    float ss[1] = {0.f};
    for (int c = threadIdx.x; c < D; c += 256) ss[0] = fmaf(xr[c], xr[c], ss[0]);
    ika_bsum<1, 8>(ss, red);
    const float r = rsqrtf(ss[0] / (float)D + eps);
    for (int c = threadIdx.x; c < D; c += 256) {
        const float y = w ? w[c] * (xr[c] * r) : xr[c] * r;
        if (ob) ob[(size_t)t * D + c] = ika_f2b(y);
        if (of) of[(size_t)t * D + c] = y;
    }
}
