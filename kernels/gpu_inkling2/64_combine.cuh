// gpu_inkling2/64_combine.cuh: the weighted expert combine, fp32.
//   out[t][n] = sum_{k = 0..7} wts[t][k] * Y[rowidx[t][k]][n]  (k in ascending expert id: a fixed order per token)
// Y is the decode expert output [slot][M][N] (rowidx = slot * M + t) or the prefill compact rows [8M][N]
// (rowidx = compact row): the same per-token arithmetic either way. grid M, block 256 (float4 columns). N % 4 == 0.
extern "C" __global__ __launch_bounds__(256) void ink2_moe_combine(
        const float* __restrict__ Y, const int* __restrict__ rowidx, const float* __restrict__ wts,
        float* __restrict__ out, int N) {
    INK2_PDL_ENTRY();
    constexpr int TOPK = 8;
    const int t = blockIdx.x;
    __shared__ int s_r[TOPK];
    __shared__ float s_w[TOPK];
    if (threadIdx.x < TOPK) { s_r[threadIdx.x] = rowidx[(size_t)t * TOPK + threadIdx.x]; s_w[threadIdx.x] = wts[(size_t)t * TOPK + threadIdx.x]; }
    __syncthreads();
    for (int c = threadIdx.x; c < N / 4; c += blockDim.x) {
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const float4 y = __ldg(reinterpret_cast<const float4*>(Y + (size_t)s_r[k] * N) + c);
            const float w = s_w[k];
            a.x = fmaf(w, y.x, a.x); a.y = fmaf(w, y.y, a.y); a.z = fmaf(w, y.z, a.z); a.w = fmaf(w, y.w, a.w);
        }
        reinterpret_cast<float4*>(out + (size_t)t * N)[c] = a;
    }
}
