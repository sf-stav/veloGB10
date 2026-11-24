// gpu_inkling2/01_k1_common.cuh — K1 device helpers (prefix ika_). Section of gpu_inkling2.cu.
//
// Reduction contract (batch invariance, AGENTS §2.4): every reduction below has a FIXED order that
// depends only on the row length / thread geometry, never on how many rows a call carries. A row is
// always reduced by the same threads in the same order, so row 0 of an M-row call is bit-identical to
// the M = 1 call.
#pragma once

__device__ __forceinline__ float ika_b2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t ika_f2b(float v) {
    __nv_bfloat16 h = __float2bfloat16_rn(v);
    return *reinterpret_cast<uint16_t*>(&h);
}
__device__ __forceinline__ float ika_sig(float v) { return 1.0f / (1.0f + __expf(-v)); }

// Streamed-once 16-B load (weights): no L1 allocation.
__device__ __forceinline__ uint4 ika_ldw4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ika_lda4(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// Warp all-reduce sum / max, xor butterfly: every lane ends with the same value (fixed order).
__device__ __forceinline__ float ika_wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float ika_wmax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block sum of NV values per thread (blockDim.x = 32 * NW): per-warp xor tree, then warp partials
// summed in ascending warp order by every thread. `red` = shared [NW][NV]. Result in v[].
template <int NV, int NW>
__device__ __forceinline__ void ika_bsum(float (&v)[NV], float* red) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = ika_wsum(v[i]);
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < NV; ++i) red[warp * NV + i] = v[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < NV; ++i) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) s += red[w * NV + i];
        v[i] = s;
    }
    __syncthreads();
}

__device__ __forceinline__ void ika_mma_bf(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Same sums as ika_bsum (per-warp xor tree, warp partials in ascending warp order), but the cross-warp pass is
// done ONCE per value (thread i sums value i) instead of redundantly by every thread: NV x NW shared loads per
// thread was the whole cost of a 30-value block reduction. `red` = shared [NW][NV], `tot` = shared [NV].
template <int NV, int NW>
__device__ __forceinline__ void ika_bsum_tot(float (&v)[NV], float* red, float* tot) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = ika_wsum(v[i]);
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < NV; ++i) red[warp * NV + i] = v[i];
    }
    __syncthreads();
    if (threadIdx.x < NV) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) s += red[w * NV + threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
}
