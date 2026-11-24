// act_quant (reference kernel.py 22-124, REF_MAP 2.2): per row, per group of 32 along K:
//   amax = max(max|x|, 1e-4); s = 2^ceil(log2(amax * fp32(1/448))) by the exponent-bit rule;
//   q = e4m3(clamp(x / s, +-448)) (RNE); scale byte = the IEEE exponent field of s (e8m0).
// One warp per group, lane = element: coalesced, exact (all fp32 elementwise).
#pragma once

__device__ __forceinline__ void d41_act_quant_group(float v, float& s_out, unsigned& e8, unsigned char& q, int lane) {
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    a = fmaxf(a, 1e-4f);
    const float inv448 = 0.0022321429569274187f;   // fp32(1/448)
    int bits = __float_as_int(a * inv448);
    unsigned e = (unsigned)((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1u : 0u);
    float s = __int_as_float((int)(e << 23));
    float r = fminf(fmaxf(v / s, -448.f), 448.f);
    q = (unsigned char)__nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3);
    s_out = s; e8 = e;
}

extern "C" __global__ void d41_fp8_act_quant_bf16(const __nv_bfloat16* __restrict__ x, long long ldx, int rows, int K,
                                                  unsigned char* __restrict__ q, long long ldq, unsigned char* __restrict__ sc, long long lds) {
    D41_PDL_ENTRY();
    int KB = K >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long gi = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (gi >= (long long)rows * KB) return;
    int r = (int)(gi / KB), kb = (int)(gi % KB);
    float v = __bfloat162float(x[(long long)r * ldx + kb * 32 + lane]);
    float s; unsigned e; unsigned char qq;
    d41_act_quant_group(v, s, e, qq, lane);
    q[(long long)r * ldq + kb * 32 + lane] = qq;
    if (lane == 0) sc[(long long)r * lds + kb] = (unsigned char)e;
}
extern "C" __global__ void d41_fp8_act_quant_f32(const float* __restrict__ x, long long ldx, int rows, int K,
                                                 unsigned char* __restrict__ q, long long ldq, unsigned char* __restrict__ sc, long long lds) {
    D41_PDL_ENTRY();
    int KB = K >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long gi = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (gi >= (long long)rows * KB) return;
    int r = (int)(gi / KB), kb = (int)(gi % KB);
    float v = x[(long long)r * ldx + kb * 32 + lane];
    float s; unsigned e; unsigned char qq;
    d41_act_quant_group(v, s, e, qq, lane);
    q[(long long)r * ldq + kb * 32 + lane] = qq;
    if (lane == 0) sc[(long long)r * lds + kb] = (unsigned char)e;
}
// inplace: bf16 -> bf16 of the dequantised value (reference `inplace=True`: window KV, WO-A input)
extern "C" __global__ void d41_fp8_act_quant_inplace_bf16(const __nv_bfloat16* __restrict__ x, long long ldx, int rows, int K,
                                                          __nv_bfloat16* __restrict__ y, long long ldy) {
    D41_PDL_ENTRY();
    int KB = K >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long gi = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (gi >= (long long)rows * KB) return;
    int r = (int)(gi / KB), kb = (int)(gi % KB);
    float v = __bfloat162float(x[(long long)r * ldx + kb * 32 + lane]);
    float s; unsigned e; unsigned char qq;
    d41_act_quant_group(v, s, e, qq, lane);
    y[(long long)r * ldy + kb * 32 + lane] = __float2bfloat16_rn(d41_e4m3(qq) * s);
}
