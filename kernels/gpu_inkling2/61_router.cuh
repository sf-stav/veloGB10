// gpu_inkling2/61_router.cuh: the Inkling-2 MoE router (AGENTS §7: fp32 router).
//   logits = x @ W^T in fp32 from the stored bf16 gate weight and the fp32 MoE input x;
//   score = sigmoid(logit); selection = top-8 of (score + correction bias), ties -> lower expert id;
//   weight = score / (sum of the 8 selected scores + 1e-20) * route_scale (scores UN-biased).
// Batch invariance: every token's logits / picks depend only on its own x row (fixed per-lane k order, fixed
// butterfly reduction), so row 0 of an M-row launch is bit-identical to an M = 1 launch.

// ------------------------------------------------------------------------------------------------
// ink2_router_logits — grid (E/wpb, ceil(M/tpb)), block 32*wpb (warp = one expert). Decode: wpb 8 (16 CTAs; measured
//   faster than wpb 4 at M = 1..16); prefill: wpb 16 (each CTA re-reads its token rows for 16 experts, so the
//   fp32 activations cross L2 8 times instead of 32). The per-token arithmetic does not depend on wpb. K == 2304 (NW = 9 uint4 of bf16 per lane held in registers for the whole token loop).
//   Each token's chain is one fixed sequence (a lane's 72 k in order, then the xor butterfly), so the logits do not
//   depend on M, tpb or wpb.
//   X [M][K] fp32 -> logits [M][E] fp32; the blockIdx.x == 0 CTAs also write Xh [M][K] = f16(X) (the expert GEMMs'
//   A operand).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float ik2m_router_dot(const uint4 (&w)[9], const float* __restrict__ xr, int lane) {
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int k = (i * 32 + lane) * 8;
        const float4 xa = __ldg(reinterpret_cast<const float4*>(xr + k));
        const float4 xb = __ldg(reinterpret_cast<const float4*>(xr + k + 4));
        const uint32_t ww[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
        const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc = fmaf(__uint_as_float(ww[j] << 16), xs[2 * j], acc);
            acc = fmaf(__uint_as_float(ww[j] & 0xFFFF0000u), xs[2 * j + 1], acc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    return acc;
}
// S4: four tokens' chains interleaved (four independent accumulators, each the exact ik2m_router_dot sequence):
// the one-token loop was latency bound (a 72-deep FMA chain + butterfly per token, 0.72 ms per 2,048-row visit).
__device__ __forceinline__ void ik2m_router_dot4(const uint4 (&w)[9], const float* __restrict__ xr, int K, int lane, float (&out)[4]) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int k = (i * 32 + lane) * 8;
        const uint32_t ww[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xr + (size_t)u * K + k));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(xr + (size_t)u * K + k + 4));
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[u] = fmaf(__uint_as_float(ww[j] << 16), xs[2 * j], acc[u]);
                acc[u] = fmaf(__uint_as_float(ww[j] & 0xFFFF0000u), xs[2 * j + 1], acc[u]);
            }
        }
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        float a = acc[u];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        out[u] = a;
    }
}
extern "C" __global__ __launch_bounds__(512) void ink2_router_logits(
        const float* __restrict__ X, const uint16_t* __restrict__ Wr, float* __restrict__ logits,
        __half* __restrict__ Xh, int M, int E, int tpb) {
    INK2_PDL_ENTRY();
    constexpr int K = 2304, NW = K / 256;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t0 = blockIdx.y * tpb, t1 = min(M, t0 + tpb);
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < (t1 - t0) * (K / 2); i += blockDim.x) {
            const int t = t0 + i / (K / 2), k = 2 * (i % (K / 2));
            const float2 v = *reinterpret_cast<const float2*>(&X[(size_t)t * K + k]);
            *reinterpret_cast<__half2*>(&Xh[(size_t)t * K + k]) = __floats2half2_rn(v.x, v.y);
        }
    }
    const int e = blockIdx.x * (blockDim.x >> 5) + warp;
    if (e >= E) return;
    uint4 w[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) w[i] = __ldg(reinterpret_cast<const uint4*>(Wr + (size_t)e * K + (i * 32 + lane) * 8));
    int t = t0;
    for (; t + 4 <= t1; t += 4) {
        float a4[4];
        ik2m_router_dot4(w, X + (size_t)t * K, K, lane, a4);
        if (lane == 0) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) logits[(size_t)(t + u) * E + e] = a4[u];
        }
    }
    for (; t < t1; ++t) {
        const float a = ik2m_router_dot(w, X + (size_t)t * K, lane);
        if (lane == 0) logits[(size_t)t * E + e] = a;
    }
}

// ------------------------------------------------------------------------------------------------
// ink2_router_topk — grid ceil(M/16), block 512 (16 warps, warp = one token). E == 128, top-8.
//   ids / wts [M][8]: the 8 picks in ASCENDING EXPERT ID order (the fixed combine order) with their final weights.
//   build_slots != 0 (decode, gridDim.x == 1, M <= 16): also the union-of-experts slot map — esel = number of
//   distinct experts, idxmap[slot] = expert (ascending), rowidx[t][k] = slot(ids[t][k]) * M + t (the row of the
//   decode expert output Y [slot][M][N] that the combine reads). Prefill builds rowidx in ink2_moe_pf_route.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512) void ink2_router_topk(
        const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids, float* __restrict__ wts,
        int* __restrict__ rowidx, int* __restrict__ idxmap, int* __restrict__ esel, int M, float rscale, int build_slots) {
    INK2_PDL_ENTRY();
    constexpr int E = 128, TOPK = 8, PER = E / 32;
    __shared__ int s_present[E], s_slot[E];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x * 16 + warp;
    const bool live = t < M;
    int myid = 0, myrank = 0;
    if (build_slots && threadIdx.x < E) s_present[threadIdx.x] = 0;
    if (live) {
        float sc[PER], bs[PER];
        #pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + 32 * j;
            const float s = 1.0f / (1.0f + expf(-logits[(size_t)t * E + e]));
            sc[j] = s;
            bs[j] = s + bias[e];
        }
        int pick[TOPK];
        float ps[TOPK];
        #pragma unroll
        for (int kk = 0; kk < TOPK; ++kk) {
            float bv = -INFINITY, sv = 0.f;
            int bi = E;
            #pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int e = lane + 32 * j;
                if (bs[j] > bv || (bs[j] == bv && e < bi)) { bv = bs[j]; bi = e; sv = sc[j]; }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                const float os = __shfl_xor_sync(0xffffffffu, sv, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; sv = os; }
            }
            pick[kk] = bi;
            ps[kk] = sv;
            #pragma unroll
            for (int j = 0; j < PER; ++j) if (lane + 32 * j == bi) bs[j] = -INFINITY;
        }
        // the reference's renorm: the sum in pick order (biased score descending == its sort order)
        float sum = 0.f;
        #pragma unroll
        for (int kk = 0; kk < TOPK; ++kk) sum += ps[kk];
        float myw = 0.f;
        #pragma unroll
        for (int kk = 0; kk < TOPK; ++kk) if (kk == lane) { myid = pick[kk]; myw = ps[kk] / (sum + 1e-20f) * rscale; }
        if (lane < TOPK) {
            int rank = 0;
            #pragma unroll
            for (int kk = 0; kk < TOPK; ++kk) rank += pick[kk] < myid ? 1 : 0;
            ids[(size_t)t * TOPK + rank] = myid;
            wts[(size_t)t * TOPK + rank] = myw;
            myrank = rank;
        }
    }
    if (!build_slots) return;
    __syncthreads();
    if (live && lane < TOPK) s_present[myid] = 1;
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int e = 0; e < E; ++e) if (s_present[e]) { s_slot[e] = n; idxmap[n] = e; ++n; }
        *esel = n;
    }
    __syncthreads();
    if (live && lane < TOPK) rowidx[(size_t)t * TOPK + myrank] = s_slot[myid] * M + t;
}
