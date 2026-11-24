// cp.async primitives shared by the FP8 GEMV and GEMM (band 20)
#pragma once

__device__ __forceinline__ void d41_cp16_zfill(void* smem, const void* gmem, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}

