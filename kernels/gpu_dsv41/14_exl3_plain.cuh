// 14_exl3_plain.cuh (leg KE): plain linears mixed into EXL3 packs (router gate F16, indexer weights_proj F16,
// BF16 compressor / indexer linears, DSpark Markov head) and the vocab-parallel embedding gather.
// y[r][n] (+)= sum_k W[n][k] * x[r][k], f32 accumulation, one warp per output column, lanes stride k by 8 and
// combine through a fixed xor tree: every row's value is independent of the row count (batch invariant).
#pragma once

template <int RB>
__device__ __forceinline__ void d41x_plain_cols(const void* __restrict__ W, int wcode, int K, const void* __restrict__ x, int xcode, int ldx,
                                                int rows, int row0, int n, float* __restrict__ res) {
    const int lane = threadIdx.x & 31;
    float acc[RB];
    #pragma unroll
    for (int i = 0; i < RB; ++i) acc[i] = 0.f;
    for (int k = lane * 8; k < K; k += 256) {
        float w[8];
        const uint4 raw = *(const uint4*)((const char*)W + ((size_t)n * K + k) * 2);
        const unsigned u[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (wcode == 0) { w[2*i] = __uint_as_float(u[i] << 16); w[2*i+1] = __uint_as_float(u[i] & 0xFFFF0000u); }
            else { w[2*i] = __half2float(__ushort_as_half((unsigned short)(u[i] & 0xFFFF))); w[2*i+1] = __half2float(__ushort_as_half((unsigned short)(u[i] >> 16))); }
        }
        #pragma unroll
        for (int i = 0; i < RB; ++i) {
            const int r = row0 + i;
            if (r < rows) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i] = fmaf(w[j], d41x_ld(x, xcode, (size_t)r * ldx + k + j), acc[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < RB; ++i) {
        float v = acc[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        res[i] = v;
    }
}

// W [N][K] (wcode 0 bf16 / 1 f16) rows n0..n0+nn of the resident tensor; grid (ceil(nn / 8), 1), 256 threads.
extern "C" __global__ void __launch_bounds__(256) d41x_plain_gemv(const void* __restrict__ W, int wcode, int K, int nn,
                                                                 const void* __restrict__ x, int xcode, int ldx, int rows,
                                                                 void* __restrict__ y, int ycode, int ldy, int accum) {
    D41_PDL_ENTRY();
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= nn) return;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        float res[8];
        d41x_plain_cols<8>(W, wcode, K, x, xcode, ldx, rows, r0, n, res);
        if ((threadIdx.x & 31) == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (r0 + i < rows) {
                const size_t oi = (size_t)(r0 + i) * ldy + n;
                float o = res[i];
                if (accum) o += d41x_ld(y, ycode, oi);
                d41x_st(y, ycode, oi, o);
            }
        }
    }
}

// d41x_plain_gemv_rg (leg PP1): the same GEMV for calls of more than 32 rows (a prefill piece), one 8-row group per block row:
// grid (ceil(nn / 8), ceil(rows / 8)), 256 threads. Each (row group, column) is the d41x_plain_cols call d41x_plain_gemv makes
// for it (same bits); a 2,048-row call on a 32-column weight (the indexer weights_proj) runs on 1,024 blocks instead of 4.
extern "C" __global__ void __launch_bounds__(256) d41x_plain_gemv_rg(const void* __restrict__ W, int wcode, int K, int nn,
                                                                    const void* __restrict__ x, int xcode, int ldx, int rows,
                                                                    void* __restrict__ y, int ycode, int ldy, int accum) {
    D41_PDL_ENTRY();
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * 8;
    if (n >= nn || r0 >= rows) return;
    float res[8];
    d41x_plain_cols<8>(W, wcode, K, x, xcode, ldx, rows, r0, n, res);
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) if (r0 + i < rows) {
            const size_t oi = (size_t)(r0 + i) * ldy + n;
            float o = res[i];
            if (accum) o += d41x_ld(y, ycode, oi);
            d41x_st(y, ycode, oi, o);
        }
    }
}

// d41x_plain_gemv_w (leg P4): the same GEMV for a NARROW weight (nn <= 256 columns: the indexer weights_proj, the router gate) at
// decode / verify widths (1..=32 rows): one warp per block, grid (nn, ceil(rows / 2)), 32 threads. Each (2-row group, column) is a
// d41x_plain_cols<2> call: every row's accumulation is the one d41x_plain_gemv makes for it (RB only sizes the accumulator array) (same bits); d41x_plain_gemv runs a 32-column weight on 4 blocks.
#define D41_PLAIN_W_RB 2
extern "C" __global__ void __launch_bounds__(32) d41x_plain_gemv_w(const void* __restrict__ W, int wcode, int K, int nn,
                                                                  const void* __restrict__ x, int xcode, int ldx, int rows,
                                                                  void* __restrict__ y, int ycode, int ldy, int accum) {
    D41_PDL_ENTRY();
    const int n = blockIdx.x;
    const int r0 = blockIdx.y * D41_PLAIN_W_RB;
    if (n >= nn || r0 >= rows) return;
    float res[D41_PLAIN_W_RB];
    d41x_plain_cols<D41_PLAIN_W_RB>(W, wcode, K, x, xcode, ldx, rows, r0, n, res);
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int i = 0; i < D41_PLAIN_W_RB; ++i) if (r0 + i < rows) {
            const size_t oi = (size_t)(r0 + i) * ldy + n;
            float o = res[i];
            if (accum) o += d41x_ld(y, ycode, oi);
            d41x_st(y, ycode, oi, o);
        }
    }
}

// out[r][:] = E[ids[r] - v0][:] when v0 <= id < v1 else 0.  E bf16 [v1 - v0][dim]; out bf16 [rows][ldo].
extern "C" __global__ void __launch_bounds__(256) d41x_gather(const __nv_bfloat16* __restrict__ E, const int* __restrict__ ids,
                                                              int v0, int v1, int dim, int rows, __nv_bfloat16* __restrict__ out, int ldo) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y;
    if (r >= rows) return;
    const int id = ids[r];
    const bool mine = id >= v0 && id < v1;
    for (int c = blockIdx.x * 256 + threadIdx.x; c < dim; c += gridDim.x * 256)
        out[(size_t)r * ldo + c] = mine ? E[(size_t)(id - v0) * dim + c] : __float2bfloat16_rn(0.f);
}
