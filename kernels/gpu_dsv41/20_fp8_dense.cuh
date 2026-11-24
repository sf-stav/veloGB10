// =============================================================================
// kernels/gpu_dsv41/20_fp8_dense.cuh — band 20-29 (leg KN1): native DeepSeek-V4.1 FP8 dense linears.
// Weight: E4M3 [N][K], one E8M0 scale per 32 x 32 block ([N/32][K/32] bytes). Activation: `act_quant`
// (E4M3, one E8M0 per 32 of a row, scales [rows][K/32]). C = bf16/f32( sum_kb sa*sb * (A_kb . B_kb^T) ),
// every 32-wide K block summed by ONE fp8 MMA (m16n8k32, fresh accumulator) and the blocks accumulated
// in f32 in the reference's order within a lane's K slice (REF_MAP 2.4).
//
//   21_fp8_act_quant.cuh   d41_fp8_act_quant_{bf16,f32}, d41_fp8_act_quant_inplace_bf16
//   22_fp8_gemv.cuh        d41_fp8_gemv_*   rows 1..16: the GEMV (<=2) and cooperative (3..16) classes
//   23_fp8_gemm.cuh        d41_fp8_gemm_*   rows >= 17: the wide (prefill) class
//
// Slice interface (T0 §11): output rows [n0, n0+n_count) (n0 % 16 == 0) and K blocks [k0b, k0b+nkb) of the
// full weight; the activation operand is the rank's LOCAL slice (its own scales). N, K are never assumed
// to be multiples of anything but 16 (rows) and 32 (K).
// =============================================================================
#pragma once

struct d41_fp8_args {
    const unsigned char* w;    // [N_full][ldw] e4m3
    const unsigned char* ws;   // [N_full/32][ldws] e8m0
    const unsigned char* xq;   // [rows][ldx] e4m3 (local K slice starts at byte 0)
    const unsigned char* xs;   // [rows][ldxs] e8m0
    void* y;                   // [rows][ldy], column 0 = output row n0
    long long ldw, ldws, ldx, ldxs, ldy;
    int n0, n_count, k0b, nkb, rows;
    int accum;                 // 0 overwrite, 1 add into y
    int gn;                    // GEMM only: n-tiles per raster group (L2 reuse); 1-D grid = tiles_n * tiles_m
};

__device__ __forceinline__ void d41_mma_e4m3_k32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

// up to 4 consecutive scale bytes, packed little-endian; missing ones are 0 (their data is zeroed too)
__device__ __forceinline__ uint32_t d41_ld_sc4(const unsigned char* p, int cnt) {
    if (cnt >= 4 && (((unsigned long long)p) & 3ull) == 0) return __ldg((const uint32_t*)p);
    uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) if (i < cnt) v |= (uint32_t)__ldg(p + i) << (8 * i);
    return v;
}
// the scale byte j as the exponent field of a float (bit 23 up): scale = 2^(e-127); the product of two scales
// is then ONE integer add (exponent fields add; bias removed once). e = 0 / 255 (denormal / NaN codes) are not
// produced by act_quant nor present in the checkpoint (checked by the census) and are not special-cased here.
__device__ __forceinline__ uint32_t d41_sc_exp(uint32_t packed, int j) { return ((packed >> (8 * j)) & 0xFFu) << 23; }
__device__ __forceinline__ float d41_sc_f(uint32_t packed, int j) { return d41_e8m0((packed >> (8 * j)) & 0xFFu); }

__device__ __forceinline__ void d41_store_out(void* y, int ot, long long idx, float v, int accum) {
    if (ot == 0) { __nv_bfloat16* p = (__nv_bfloat16*)y + idx; float o = accum ? __bfloat162float(*p) + v : v; *p = __float2bfloat16_rn(o); }
    else { float* p = (float*)y + idx; *p = accum ? *p + v : v; }
}

#include "21_fp8_act_quant.cuh"
#include "22_fp8_gemv.cuh"
#include "23_fp8_gemm.cuh"
