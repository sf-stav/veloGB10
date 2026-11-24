// Band 40 helpers (leg KC). Every arithmetic step that model.py / kernel.py round in fp32 uses the
// explicit-rounding intrinsics (__fmul_rn ...) so nvcc's FMA contraction cannot change a bit.
#pragma once
#include <cuda_fp8.h>
#include <cstddef>

#define D41_KC_DIM 512      // head_dim: the main KV latent, the SWA row
#define D41_KC_ROPE 64      // rotary tail
#define D41_KC_THREADS 128  // one block per 512-wide row, 4 channels per thread

// bf16 <-> f32 (RNE)
__device__ __forceinline__ float d41_bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float d41_rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// RMSNorm weight read (the pack stores norms as bf16, f16 or f32): fmt tags D41_FMT_BF16 / F16 / F32.
__device__ __forceinline__ float d41_load_w(const void* w, unsigned fmt, int i) {
    if (fmt == D41_FMT_F32) return ((const float*)w)[i];
    if (fmt == D41_FMT_F16) return __half2float(((const __half*)w)[i]);
    return __bfloat162float(((const __nv_bfloat16*)w)[i]);
}

// fp32 -> e2m1 code (round to nearest EVEN code) of |v| <= 6; ties: .25->0 .75->1 1.25->1 1.75->2 2.5->2 3.5->4 5->4.
// The sign bit is set only for a non-zero magnitude (the reference stores -0 as +0).
__device__ __forceinline__ unsigned d41_e2m1_code(float v) {
    const float a = fabsf(v);
    unsigned m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return m | ((v < 0.0f && m) ? 8u : 0u);
}
__device__ __forceinline__ float d41_clamp6(float v) { return fminf(fmaxf(v, -6.0f), 6.0f); }

// kernel.py fast_round_scale: the exponent byte of the smallest power of two >= v (v positive, normal).
__device__ __forceinline__ unsigned d41_pow2ceil_byte(float v) {
    const unsigned b = __float_as_uint(v);
    return ((b >> 23) & 0xFFu) + ((b & 0x7FFFFFu) != 0u);
}
__device__ __forceinline__ float d41_pow2_of_byte(unsigned e) { return __uint_as_float(e << 23); }

// e4m3 byte of fp32 (RNE, saturating at +-448: CUDA's satfinite cast).
__device__ __forceinline__ unsigned d41_e4m3_byte(float v) {
    return (unsigned)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// Max over `n` consecutive lanes (n = 4, 8, 16) of a value, within one warp (lanes aligned to n).
template <int N> __device__ __forceinline__ float d41_group_max(float v) {
#pragma unroll
    for (int o = 1; o < N; o <<= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

// Sum of squares of `n_per` values per thread (ascending), then a fixed butterfly over the warp.
__device__ __forceinline__ float d41_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

// RoPE of the cache writes: the shared arithmetic of 00_abi.cuh (SL RoPE unification; were the separately
// rounded products and sincosf here). cos / sin of pos * inv_freq[i], then the forward rotation.
__device__ __forceinline__ void d41_kc_rope_cs(const float* inv_freq, long long pos, int i, float& c, float& s) { d41_rope_cs(pos, inv_freq[i], c, s); }
__device__ __forceinline__ void d41_kc_rope_pair(float& a, float& b, float c, float s) { d41_rope_fwd(a, b, c, s); }
