// =============================================================================
// 70_mixers.cuh — band 70-79 (leg KM): mHC mixers, RMS norms, the router, the Engram gate.
// Head file of the band (included by kernels/gpu_dsv41.cu); it pulls in the band's other files.
//   71_mix_hc.cuh      mHC: K-sliced hc_fn partial, finish (rank-order sum + Sinkhorn), hc_pre(+norm), hc_post
//   72_mix_norm.cuh    RMSNorm (generic dim, bf16 in / bf16 out, f32 / bf16 weight)
//   73_mix_router.cuh  router: fp32 gate GEMV (bf16 / f16 / f32 weights) + sqrtsoftplus / noaux_tc / top-k select
//   74_mix_engram.cuh  Engram gate (signed-sqrt sigmoid gate, value added to the 4 copies)
// Reference semantics: PLAN/DSV41_REF_MAP.md 2.7, 3.2, 3.6, 3.7 (model.py 328-365, 794-829, 940-996;
// kernel.py 407-474). Every entry takes ONE by-value argument struct (host mirror: src/dsv41/gpu/mixers.rs).
// Determinism: every reduction has a fixed order independent of the row count (batch invariant), no atomics.
// =============================================================================
#pragma once

#define D41_HC 4
#define D41_HC_MIX 24            // (2 + hc) * hc : pre 4, post 4, comb 16
#define D41_HC_PART 25           // 24 dot products + the sum of squares of the stream slice

typedef __nv_bfloat16 d41_bf16;

__device__ __forceinline__ float d41_bf2f(d41_bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float d41_bf2f_u16(unsigned short u) { return __uint_as_float(((unsigned)u) << 16); }

// element i of a weight / parameter vector of format fmt (D41_FMT_BF16 / F16 / F32)
__device__ __forceinline__ float d41_ldw(const void* p, unsigned fmt, long long i) {
    if (fmt == D41_FMT_F32) return ((const float*)p)[i];
    if (fmt == D41_FMT_F16) return __half2float(((const __half*)p)[i]);
    return d41_bf2f(((const d41_bf16*)p)[i]);
}

// Reduce-scatter 32 values per lane across the warp: lane l ends with sum over lanes of v[l], in 31 shuffles.
// The order is a fixed tree, identical for every call (a row's result never depends on the other rows).
template <int H>
__device__ __forceinline__ void d41_rs_step(float (&v)[32], bool hi) {
#pragma unroll
    for (int i = 0; i < H; ++i) {
        float send = hi ? v[i] : v[i + H];
        float keep = hi ? v[i + H] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, H);
    }
}
__device__ __forceinline__ float d41_rs32(float (&v)[32], int lane) {
    d41_rs_step<16>(v, (lane & 16) != 0);
    d41_rs_step<8>(v, (lane & 8) != 0);
    d41_rs_step<4>(v, (lane & 4) != 0);
    d41_rs_step<2>(v, (lane & 2) != 0);
    d41_rs_step<1>(v, (lane & 1) != 0);
    return v[0];
}

// Block sum of NV values for a block of 256 threads (8 warps): xor tree per warp, then the warps in order.
template <int NV>
__device__ __forceinline__ void d41_block_sum256(float (&v)[NV], float* sm /* NV * 8 floats */) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int q = 0; q < NV; ++q) {
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) v[q] += __shfl_xor_sync(0xffffffffu, v[q], o);
        if (lane == 0) sm[q * 8 + warp] = v[q];
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < NV; ++q) {
        float s = sm[q * 8];
#pragma unroll
        for (int w = 1; w < 8; ++w) s += sm[q * 8 + w];
        v[q] = s;
    }
    __syncthreads();
}

#include "71_mix_hc.cuh"
#include "72_mix_norm.cuh"
#include "73_mix_router.cuh"
#include "74_mix_engram.cuh"
#include "75_mix_qkv.cuh"
