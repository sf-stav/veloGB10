// gpu_inkling2/80_glue.cuh — model glue (leg S3). Section of gpu_inkling2.cu.
//
// GPU sampler for temperature-only sampling (the stock pack's generation_config: temperature 1.0, no top-p /
// top-k): Gumbel-max over the full softcapped logits row, which is EXACT sampling from softmax(logits / T)
// (argmax_i (l_i / T + G_i), G_i iid Gumbel(0, 1)). The noise is a counter-based hash of (seed, i), so a row's
// draw does not depend on the launch shape. Ties resolve to the lowest id (the greedy rule).
//   ink3_gumbel_part: grid (G, M), block 256 — per (CTA, row) the (max, lowest id) of its vocab slice.
//   ink3_gumbel_pick: grid (M), block 256 — reduces the G partials of each row in ascending CTA order.
// T <= 0 is greedy (no noise): the same kernels then compute the plain argmax.

__device__ __forceinline__ unsigned long long ink3_mix(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

extern "C" __global__ __launch_bounds__(256) void ink3_gumbel_part(const float* __restrict__ logits, int ldl, int n,
        const unsigned long long* __restrict__ seeds, const float* __restrict__ inv_t,
        float* __restrict__ pmax, int* __restrict__ pidx) {
    INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int row = blockIdx.y, G = gridDim.x;
    const int per = (n + G - 1) / G;
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    const float it = inv_t[row];
    const unsigned long long seed = seeds[row];
    const float* lr = logits + (size_t)row * ldl;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {
        float v = lr[i];
        if (it > 0.f) {
            const unsigned long long h = ink3_mix(seed ^ ink3_mix((unsigned long long)i));
            // G = -log(-log u), u in (0, 1) from 53 bits. -log u is formed as -log1p(-(1 - u)) from the exact 1 - u, so the
            // large-G tail (u -> 1) keeps full relative precision (a fast log there would add O(1) noise to the top draws).
            const double w = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);   // 1 - u, in (0, 1)
            const double e = -log1p(-w);                                                 // -log u > 0
            v = v * it - (float)log(e);
        }
        if (v > best) { best = v; bi = i; }   // ascending i per thread: first max kept
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sv[threadIdx.x], b = sv[threadIdx.x + s];
            const int ia = si[threadIdx.x], ib = si[threadIdx.x + s];
            if (b > a || (b == a && ib < ia)) { sv[threadIdx.x] = b; si[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmax[row * G + blockIdx.x] = sv[0]; pidx[row * G + blockIdx.x] = si[0]; }
}

extern "C" __global__ __launch_bounds__(256) void ink3_gumbel_pick(const float* __restrict__ pmax, const int* __restrict__ pidx,
        int G, int* __restrict__ out) {
    INK2_PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int row = blockIdx.x;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int g = threadIdx.x; g < G; g += 256) {
        const float v = pmax[row * G + g]; const int i = pidx[row * G + g];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sv[threadIdx.x], b = sv[threadIdx.x + s];
            const int ia = si[threadIdx.x], ib = si[threadIdx.x + s];
            if (b > a || (b == a && ib < ia)) { sv[threadIdx.x] = b; si[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = si[0];
}
