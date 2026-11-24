// gpu_inkling2/c1_actq_quant.cuh — leg Q1: row quantisers of the expert inputs (the reference's
// nvfp4_quantize_routed / mxfp8_quantize_routed bodies without the routing gather: every row is quantised once, a
// token's x row is the same bytes whichever expert reads it), plus the GU epilogue's arithmetic and a fixture kernel.
//   IN_F32: the input is fp32 (our MoE input) rounded to bf16 first (the reference quantises its bf16 x); else bf16.
//   Rows: `rows`, or (*nrows_dev) * nmul when nrows_dev != nullptr (decode: active slots x M).
// grid: min(rows, 4 x SMs) CTAs, grid-stride over rows (the reference's launch shape).

template <int K, bool IN_F32>
__device__ __forceinline__ void ink2q_nv4_quant_body(const void* __restrict__ xin, uint8_t* __restrict__ codes,
                                                     uint8_t* __restrict__ scales, float* __restrict__ amax_out,
                                                     int rows, const int* __restrict__ nrows_dev, int nmul) {
    constexpr int kThreads = ink2q_split_threads(K), kSplit = ink2q_split(K), E = 16 / kSplit;
    static_assert(K % 64 == 0 && kSplit > 1 && E == 4, "ink2q_nv4_quant: width");
    __shared__ float red[2][kThreads / 32];
    if (nrows_dev) rows = *nrows_dev * nmul;
    const int block = threadIdx.x / kSplit, part = threadIdx.x % kSplit;
    const bool active = block < K / 16;
    int parity = 0;
    for (int row = blockIdx.x; row < rows; row += gridDim.x, parity ^= 1) {
        float v[E];
#pragma unroll
        for (int i = 0; i < E; ++i) v[i] = 0.0f;
        if (active) {
            const size_t o = (size_t)row * K + block * 16 + part * E;
            if constexpr (IN_F32) {
                const float4 f = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xin) + o);
                v[0] = ink2q_bf(f.x); v[1] = ink2q_bf(f.y); v[2] = ink2q_bf(f.z); v[3] = ink2q_bf(f.w);
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(xin) + o);
                const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                for (int i = 0; i < E; ++i) v[i] = __bfloat162float(h[i]);
            }
        }
        float block_max = ink2q_part_max<E>(v);
        float amax = ink2q_block_max_reduce<kThreads>(block_max, red[parity]);
        uint32_t packed;
        uint8_t scale;
        ink2q_quantize_part<E>(v, amax, block_max, packed, scale);
        if (active) {
            if (threadIdx.x == 0) amax_out[row] = amax;
            *reinterpret_cast<uint16_t*>(codes + (size_t)row * (K / 2) + block * 8 + part * (E / 2)) = uint16_t(packed);
            if (part == 0) scales[(size_t)row * (K / 16) + block] = scale;
        }
    }
}

template <int K, bool IN_F32>
__device__ __forceinline__ void ink2q_mx8_quant_body(const void* __restrict__ xin, uint8_t* __restrict__ codes,
                                                     uint8_t* __restrict__ exps, int rows,
                                                     const int* __restrict__ nrows_dev, int nmul) {
    static_assert(K % 64 == 0 && K / 16 <= 1024, "ink2q_mx8_quant: width");
    if (nrows_dev) rows = *nrows_dev * nmul;
    const int block = threadIdx.x;
    const bool active = block < K / 16;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float v[16];
        if (active) {
            const size_t o = (size_t)row * K + block * 16;
            if constexpr (IN_F32) {
                const float4* f = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xin) + o);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 q = f[i];
                    v[4 * i] = ink2q_bf(q.x); v[4 * i + 1] = ink2q_bf(q.y); v[4 * i + 2] = ink2q_bf(q.z); v[4 * i + 3] = ink2q_bf(q.w);
                }
            } else {
                const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(xin) + o);
                uint4 raw[2] = {src[0], src[1]};
                const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(raw);
#pragma unroll
                for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(h[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = 0.0f;
        }
        uint4 packed;
        uint8_t exponent;
        ink2q_mxfp8_half_block(v, packed, exponent);
        if (active) {
            *reinterpret_cast<uint4*>(codes + (size_t)row * K + block * 16) = packed;
            if (block % 2 == 0) exps[(size_t)row * (K / 32) + block / 2] = exponent;
        }
    }
}

#define INK2Q_NV4_QUANT(NAME, K, F32)                                                                                  \
    extern "C" __global__ void __launch_bounds__(ink2q_split_threads(K)) NAME(                                        \
            const void* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,                    \
            float* __restrict__ amax, int rows, const int* __restrict__ nrows_dev, int nmul) {                        \
        INK2_PDL_ENTRY();                                                                                              \
        ink2q_nv4_quant_body<K, F32>(x, codes, scales, amax, rows, nrows_dev, nmul);                                   \
    }
#define INK2Q_MX8_QUANT(NAME, K, F32)                                                                                  \
    extern "C" __global__ void __launch_bounds__(ink2q_mx_threads(K)) NAME(                                           \
            const void* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ exps, int rows,            \
            const int* __restrict__ nrows_dev, int nmul) {                                                             \
        INK2_PDL_ENTRY();                                                                                              \
        ink2q_mx8_quant_body<K, F32>(x, codes, exps, rows, nrows_dev, nmul);                                           \
    }
INK2Q_NV4_QUANT(ink2q_nv4_quant_f2304, 2304, true)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b2304, 2304, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b1152, 1152, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b640, 640, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b576, 576, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b512, 512, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b384, 384, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b320, 320, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b256, 256, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b192, 192, false)
INK2Q_NV4_QUANT(ink2q_nv4_quant_b128, 128, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_f2304, 2304, true)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b2304, 2304, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b1152, 1152, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b640, 640, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b576, 576, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b512, 512, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b384, 384, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b320, 320, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b256, 256, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b192, 192, false)
INK2Q_MX8_QUANT(ink2q_mx8_quant_b128, 128, false)

// The NVFP4 per-row outer scale of a GEMM output: (x_amax * w_amax) * rcp_rn(1536 * w_den) (the reference's
// _nvfp4_row_scale / the swiglu kernel's `scale`, same association).
__device__ __forceinline__ float ink2q_row_scale(float x_amax, float w_amax, float w_den) {
    return __fmul_rn(__fmul_rn(x_amax, w_amax), __frcp_rn(__fmul_rn(1536.0f, w_den)));
}
// GU epilogue -> the bf16 down-projection input (gamma = the routing weight, applied before the quantiser as the
// reference does). raw_*: the fp32 GEMM accumulators; the reference's GEMM writes them as bf16 first.
__device__ __forceinline__ float ink2q_gu_nv(float raw_g, float raw_u, float scale, float gamma, float gcap, float ucap) {
    const float gate = ink2q_bf(__fmul_rn(ink2q_bf(raw_g), scale));
    const float up = ink2q_bf(__fmul_rn(ink2q_bf(raw_u), scale));
    return ink2q_bf(ink2q_capped_swiglu(gate, up, gamma, gcap, ucap));
}
__device__ __forceinline__ float ink2q_gu_mx(float raw_g, float raw_u, float gamma, float gcap, float ucap) {
    return ink2q_bf(ink2q_capped_swiglu(ink2q_bf(raw_g), ink2q_bf(raw_u), gamma, gcap, ucap));
}
// DN epilogue: the reference's GEMM output bf16(raw), NVFP4 scaled per row and rounded to bf16 again (routed_partial).
__device__ __forceinline__ float ink2q_dn_nv(float raw, float rs) { return ink2q_bf(__fmul_rn(ink2q_bf(raw), rs)); }
__device__ __forceinline__ float ink2q_dn_mx(float raw) { return ink2q_bf(raw); }

// Fixture kernel: the GU epilogue over a bf16 [rows][2N] (gate, up interleaved) accumulator -> bf16 act [rows][N].
// w = {w_amax, w_den}; nv = 0 -> the MXFP8 form (no scale). grid rows, block 256.
extern "C" __global__ void ink2q_gu_epi_test(const __nv_bfloat16* __restrict__ acc, const float* __restrict__ acc_amax,
                                             const float* __restrict__ w, const float* __restrict__ gam,
                                             __nv_bfloat16* __restrict__ act, int N, int nv, float gcap, float ucap) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.x;
    const float sc = nv ? ink2q_row_scale(acc_amax[r], w[0], w[1]) : 1.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float g = __bfloat162float(acc[(size_t)r * 2 * N + 2 * i]), u = __bfloat162float(acc[(size_t)r * 2 * N + 2 * i + 1]);
        const float v = nv ? ink2q_gu_nv(g, u, sc, gam[r], gcap, ucap) : ink2q_gu_mx(g, u, gam[r], gcap, ucap);
        act[(size_t)r * N + i] = __float2bfloat16_rn(v);
    }
}

// Routing weight per expert row: gam[rowidx[t][k]] = wts[t][k]; rows [0, nzero) are zeroed first (decode: the union
// slots' non-routed token rows). One CTA.
extern "C" __global__ void __launch_bounds__(256) ink2q_gamma(const int* __restrict__ rowidx, const float* __restrict__ wts,
                                                             float* __restrict__ gam, int pairs, int nzero) {
    INK2_PDL_ENTRY();
    for (int i = threadIdx.x; i < nzero; i += blockDim.x) gam[i] = 0.0f;
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) gam[rowidx[p]] = wts[p];
}
// Prefill version: grid-stride, no zeroing (every compact row is routed).
extern "C" __global__ void __launch_bounds__(256) ink2q_gamma_pf(const int* __restrict__ rowidx, const float* __restrict__ wts,
                                                                float* __restrict__ gam, int pairs) {
    INK2_PDL_ENTRY();
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += gridDim.x * blockDim.x) gam[rowidx[p]] = wts[p];
}
