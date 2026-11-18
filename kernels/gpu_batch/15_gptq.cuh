// gpu_batch/15_gptq.cuh: GPTQ to NVFP4 quantizer kernels (src/gptq.rs --gptq): absmax, Hadamard, sweep, calibration profile.
// Quantizer-only; not on a serving path.
// Section 15 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================== GPTQ → NVFP4 (src/gptq.rs, the --gptq quantizer) =====================
// W is f32 row-major [M, K] on device (a working copy of the bf16 source weight, optionally
// micro-rotated). U is the upper Cholesky factor of the damped inverse Hessian (f32 row-major
// [K, K]; only j <= k is read). The sweep quantizes one 128-column block for ALL rows at once —
// one thread per row, columns strictly in order with the GPTQ error feedback inside the block —
// and writes the NVFP4 codes (nibbles, [M, K/2]) and the per-16 E4M3 block scales ([M, K/16]) in
// the artifact's layout; the cross-block propagation W[:, c1:] -= Err · U[c0:c1, c1:] is a cuBLAS
// GEMM issued by the host between sweeps.

__device__ __forceinline__ unsigned char gptq_e2m1(float x) {   // == quant::f32_to_e2m1 (RNE, ties to even code)
    const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    unsigned char sign = x < 0.f ? 8 : 0;
    float a = fminf(fabsf(x), 6.f);
    unsigned char best = 0; float be = 1e30f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float e = fabsf(grid[i] - a);
        if (e < be - 1e-9f || (fabsf(e - be) <= 1e-9f && (i % 2) == 0)) { be = e; best = (unsigned char)i; }
    }
    return sign | best;
}
__device__ __forceinline__ float gptq_e2m1_val(unsigned char c) {
    const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = grid[c & 7]; return (c & 8) ? -v : v;
}

extern "C" __global__ void gptq_bf16_to_f32_b(float* out, const __nv_bfloat16* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// |x| max over n floats (atomicMax on the non-negative float bit pattern; *out must start at 0).
extern "C" __global__ void gptq_absmax_b(unsigned int* out, const __nv_bfloat16* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? fabsf(__bfloat162float(x[i])) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(v));
}

// NVFP4 activation headroom calibration. One logical sample is the amax of a contiguous
// 16-element activation block, matching the dynamic UE4M3 block scale consumed by W4A4. The
// histogram is logarithmic so corpora with very different activation ranges can be merged by
// adding counts instead of merging already-rounded global scales.
//
// stats layout: [512 log2 histogram bins | zero-block count | non-finite-block count], all u64.
// `running_max` is the positive f32 bit-pattern, so integer atomicMax has the desired ordering.
extern "C" __global__ void igs_hist_b(unsigned long long* stats, unsigned int* running_max,
                                      const __nv_bfloat16* x, long long n) {
    constexpr int NBINS = 512;
    constexpr float LOG2_MIN = -40.0f;
    constexpr float LOG2_MAX = 40.0f;
    __shared__ unsigned int shist[NBINS];
    __shared__ unsigned int sextra[2];
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) shist[i] = 0;
    if (threadIdx.x < 2) sextra[threadIdx.x] = 0;
    __syncthreads();

    const long long block = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nblocks = (n + 15) / 16;
    if (block < nblocks) {
        const long long base = block * 16;
        float amax = 0.0f;
        bool invalid = false;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const long long i = base + j;
            if (i < n) {
                const float value = __bfloat162float(x[i]);
                invalid |= !isfinite(value);
                if (!invalid) amax = fmaxf(amax, fabsf(value));
            }
        }
        if (invalid) {
            atomicAdd(&sextra[1], 1u);
        } else if (amax == 0.0f) {
            atomicAdd(&sextra[0], 1u);
        } else {
            atomicMax(running_max, __float_as_uint(amax));
            const float frac = (log2f(amax) - LOG2_MIN) / (LOG2_MAX - LOG2_MIN);
            int bin = (int)floorf(frac * (float)NBINS);
            bin = max(0, min(NBINS - 1, bin));
            atomicAdd(&shist[bin], 1u);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) {
        const unsigned int count = shist[i];
        if (count) atomicAdd(&stats[i], (unsigned long long)count);
    }
    if (threadIdx.x < 2 && sextra[threadIdx.x]) {
        atomicAdd(&stats[NBINS + threadIdx.x], (unsigned long long)sextra[threadIdx.x]);
    }
}

// In-place 16-point Hadamard (orthonormal, H/4) on every 16-vector of a [M, K] f32 matrix.
// axis 0: vectors along the rows (elements row*K + 16b + i) — the K (input) dimension;
// axis 1: vectors along the columns (elements (16b+i)*K + col) — for the Hessian's row side.
extern "C" __global__ void gptq_hadamard16_b(float* W, int M, int K, int axis) {
    long long nvec = (axis == 0) ? (long long)M * (K / 16) : (long long)(M / 16) * K;
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    float* p; long long stride;
    if (axis == 0) { long long row = v / (K / 16), b = v % (K / 16); p = W + row * K + b * 16; stride = 1; }
    else           { long long b = v / K, col = v % K; p = W + (b * 16) * K + col; stride = K; }
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = p[i * stride];
    #pragma unroll
    for (int len = 1; len < 16; len <<= 1)
        #pragma unroll
        for (int i = 0; i < 16; i += 2 * len)
            #pragma unroll
            for (int j = i; j < i + len; j++) { float a = x[j], b = x[j + len]; x[j] = a + b; x[j + len] = a - b; }
    #pragma unroll
    for (int i = 0; i < 16; i++) p[i * stride] = x[i] * 0.25f;
}

// The block sweep. c0 = first column of the block, bs = block width (multiple of 16, <= 256).
// s_tensor = 1/global_scale (the per-tensor NVFP4 scale). nclip = number of clip ratios tried per
// 16-group (1 = plain amax/6). Err [M, bs] row-major receives (w - q)/U[j][j].
extern "C" __global__ void gptq_sweep_b(float* W, unsigned char* qw, unsigned char* qs, float* Err,
                                        const float* U, int M, int K, int c0, int bs, float s_tensor, int nclip) {
    const float ratios[7] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f};
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float* w = W + (long long)r * K;
    float* e = Err + (long long)r * bs;
    float s = 1.f, inv = 0.f;
    for (int j = 0; j < bs; j++) {
        const int col = c0 + j;
        if ((col & 15) == 0) {
            // group scale from the CURRENT (error-updated) values of this row's group
            float amax = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(w[col + i]));
            unsigned char best_code = 0; float best_err = 1e30f;
            for (int c = 0; c < nclip; c++) {
                float sraw = (amax > 0.f) ? amax * ratios[c] / 6.f / s_tensor : 0.f;
                unsigned char code = f32_to_e4m3(sraw);
                float sc = e4m3_f(code) * s_tensor;
                float iv = (sc > 0.f) ? 1.f / sc : 0.f;
                float err = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; i++) { float q = gptq_e2m1_val(gptq_e2m1(w[col + i] * iv)) * sc; err += (q - w[col + i]) * (q - w[col + i]); }
                if (c == 0 || err < best_err) { best_err = err; best_code = code; }
            }
            qs[(long long)r * (K / 16) + col / 16] = best_code;
            s = e4m3_f(best_code) * s_tensor;
            inv = (s > 0.f) ? 1.f / s : 0.f;
        }
        unsigned char code = gptq_e2m1(w[col] * inv);
        float q = gptq_e2m1_val(code) * s;
        long long qi = (long long)r * (K / 2) + col / 2;
        unsigned char byte = qw[qi];
        qw[qi] = (col & 1) ? ((byte & 0x0F) | (code << 4)) : ((byte & 0xF0) | code);
        const float d = U[(long long)col * K + col];
        const float err = (w[col] - q) / d;
        e[j] = err;
        const float* urow = U + (long long)col * K;
        for (int k2 = j; k2 < bs; k2++) w[c0 + k2] -= err * urow[c0 + k2];
    }
}

// Gather n token rows (each K bf16, contiguous) by index into dst [n, K].
extern "C" __global__ void gptq_gather_rows_b(__nv_bfloat16* dst, const __nv_bfloat16* src, const int* idx, int n, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * K) return;
    int t = (int)(i / K), c = (int)(i % K);
    dst[i] = src[(long long)idx[t] * K + c];
}

// act[i, t] = silu(g[i, t]) * u[i, t] from a fused [2I, n] f32 gate|up buffer (column-major per
// token: gate rows 0..I, up rows I..2I), written bf16 [I, n].
extern "C" __global__ void gptq_silu_mul_gu_b(__nv_bfloat16* act, const float* gu, int I, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)I * n) return;
    int t = (int)(i / I), r = (int)(i % I);
    float g = gu[(long long)t * 2 * I + r], u = gu[(long long)t * 2 * I + I + r];
    act[i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
}

// The engine-side micro-rotation for MR-GPTQ artifacts: x' = H16/4 · x on every 16-block of the
// K (feature) dimension of a bf16 activation matrix [K, n] (feature-contiguous per column).
extern "C" __global__ void gptq_rotate_act_b(__nv_bfloat16* out, const __nv_bfloat16* x, long long nvec) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    const __nv_bfloat16* p = x + v * 16;
    float y[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) y[i] = __bfloat162float(p[i]);
    #pragma unroll
    for (int len = 1; len < 16; len <<= 1)
        #pragma unroll
        for (int i = 0; i < 16; i += 2 * len)
            #pragma unroll
            for (int j = i; j < i + len; j++) { float a = y[j], b = y[j + len]; y[j] = a + b; y[j + len] = a - b; }
    __nv_bfloat16* o = out + v * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = __float2bfloat16(y[i] * 0.25f);
}
extern "C" __global__ void gptq_absmax_f32_b(unsigned int* out, const float* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? fabsf(x[i]) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(v));
}

// Pick the best E4M3 local scale for one 16-value NVFP4 group. The candidates deliberately
// match gptq_sweep_b: this helper is used by the alternating tensor-scale fit and by static
// activation-order's precomputed groups, so calibration and the final sweep share one grid.
__device__ __forceinline__ unsigned char gptq_best_scale16(const float x[16], float s_tensor,
                                                            int nclip, float* best_mse) {
    const float ratios[7] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f};
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
    unsigned char best_code = 0;
    float be = 1e30f;
    for (int c = 0; c < nclip; c++) {
        float raw = (amax > 0.f && s_tensor > 0.f) ? amax * ratios[c] / 6.f / s_tensor : 0.f;
        unsigned char code = f32_to_e4m3(raw);
        float sc = e4m3_f(code) * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        float err = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float q = gptq_e2m1_val(gptq_e2m1(x[i] * inv)) * sc;
            float d = q - x[i];
            err += d * d;
        }
        if (c == 0 || err < be) { be = err; best_code = code; }
    }
    *best_mse = be;
    return best_code;
}

// Alternating global-scale sufficient statistics. With a fixed local-scale/code assignment,
// q_i = s_tensor * z_i and the least-squares tensor scale is sum(w_i*z_i)/sum(z_i^2).
// Re-evaluating this kernel at that scale reassigns the local E4M3 scales and FP4 codes. One
// block contributes three f64 atomics, avoiding the precision loss of atomics per 16-value group.
extern "C" __global__ void gptq_scale_stats_f32_b(double* stats, const float* w,
                                                   long long ngroups, float s_tensor, int nclip) {
    __shared__ double sn[256], sd[256], se[256];
    double num = 0.0, den = 0.0, mse = 0.0;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < ngroups; g += (long long)gridDim.x * blockDim.x) {
        float x[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) x[i] = w[g * 16 + i];
        float group_mse;
        unsigned char scode = gptq_best_scale16(x, s_tensor, nclip, &group_mse);
        float es = e4m3_f(scode);
        float sc = es * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float z = es * gptq_e2m1_val(gptq_e2m1(x[i] * inv));
            num += (double)x[i] * (double)z;
            den += (double)z * (double)z;
        }
        mse += (double)group_mse;
    }
    sn[threadIdx.x] = num; sd[threadIdx.x] = den; se[threadIdx.x] = mse;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x+s]; sd[threadIdx.x] += sd[threadIdx.x+s]; se[threadIdx.x] += se[threadIdx.x+s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(stats, sn[0]); atomicAdd(stats + 1, sd[0]); atomicAdd(stats + 2, se[0]); }
}

// Same statistics directly from the bf16 source. This lets a stacked MoE tensor share one
// optimized global scale without materializing every expert as f32 at once. Because MR's H16 acts
// independently on exactly these groups, the optional rotation is lossless with respect to the
// f32 working copy produced by gptq_w32.
extern "C" __global__ void gptq_scale_stats_bf16_b(double* stats, const __nv_bfloat16* w,
                                                    long long ngroups, int rotate,
                                                    float s_tensor, int nclip) {
    __shared__ double sn[256], sd[256], se[256];
    double num = 0.0, den = 0.0, mse = 0.0;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < ngroups; g += (long long)gridDim.x * blockDim.x) {
        float x[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) x[i] = __bfloat162float(w[g * 16 + i]);
        if (rotate) {
            #pragma unroll
            for (int len = 1; len < 16; len <<= 1)
                #pragma unroll
                for (int i = 0; i < 16; i += 2 * len)
                    #pragma unroll
                    for (int j = i; j < i + len; j++) { float a=x[j], b=x[j+len]; x[j]=a+b; x[j+len]=a-b; }
            #pragma unroll
            for (int i = 0; i < 16; i++) x[i] *= 0.25f;
        }
        float group_mse;
        unsigned char scode = gptq_best_scale16(x, s_tensor, nclip, &group_mse);
        float es = e4m3_f(scode);
        float sc = es * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float z = es * gptq_e2m1_val(gptq_e2m1(x[i] * inv));
            num += (double)x[i] * (double)z;
            den += (double)z * (double)z;
        }
        mse += (double)group_mse;
    }
    sn[threadIdx.x] = num; sd[threadIdx.x] = den; se[threadIdx.x] = mse;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x+s]; sd[threadIdx.x] += sd[threadIdx.x+s]; se[threadIdx.x] += se[threadIdx.x+s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(stats, sn[0]); atomicAdd(stats + 1, sd[0]); atomicAdd(stats + 2, se[0]); }
}

// Static groups: choose every local scale from the original, unpermuted rotated weight before
// GPTQ updates it. Static activation-order can then visit columns in any order while every final
// code remains reconstructible by the original block-of-16 scale layout.
extern "C" __global__ void gptq_static_scales_b(unsigned char* qs, const float* w,
                                                 long long ngroups, float s_tensor, int nclip) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = w[g * 16 + i];
    float mse;
    qs[g] = gptq_best_scale16(x, s_tensor, nclip, &mse);
}

// NVIDIA-style NVFP4 local-Hessian scale search. One CUDA block handles eight output rows
// for one original 16-column group. Each warp cooperatively evaluates all 126 positive finite
// E4M3 scales and minimizes dw^T H_block dw. The same scale bytes and
// packed FP4 layout are consumed by the existing runtime; this adds no inference overhead.
extern "C" __global__ void gptq_static_scales_hessian_b(
        unsigned char* qs, const float* w, const float* H,
        unsigned long long* fallback_count, int M, int K, float s_tensor) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = blockIdx.x;
    const int row = (int)blockIdx.y * 8 + warp;
    const int c0 = group * 16;
    __shared__ float h16[16 * 16];

    const int t = threadIdx.x;
    h16[t] = H[(long long)(c0 + t / 16) * K + c0 + t % 16];
    __syncthreads();
    if (row >= M) return;

    // The warp cooperates on one candidate at a time: lane i owns dw[i], then broadcasts it
    // to form one Hessian row. This performs the same arithmetic as four independent candidates
    // per lane while keeping register pressure and PTX size bounded.
    const unsigned mask = 0xffffffffu;
    const float wi = lane < 16 ? w[(long long)row * K + c0 + lane] : 0.f;
    float amax = lane < 16 ? fabsf(wi) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
    float best_loss = 3.402823466e+38F;
    int best_code = 0;
    #pragma unroll 1
    for (int code = 1; code <= 126; code++) {
        const float scale = e4m3_f((unsigned char)code) * s_tensor;
        const float inv = scale > 0.f ? 1.f / scale : 0.f;
        const float q = lane < 16 ? gptq_e2m1_val(gptq_e2m1(wi * inv)) * scale : 0.f;
        const float dw = lane < 16 ? wi - q : 0.f;
        float hd = 0.f;
        #pragma unroll 1
        for (int j = 0; j < 16; j++) {
            const float dwj = __shfl_sync(mask, dw, j);
            if (lane < 16) hd += h16[lane * 16 + j] * dwj;
        }
        float loss = dw * hd;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            loss += __shfl_down_sync(mask, loss, off);
        if (lane == 0 && isfinite(loss) && loss < best_loss) {
            best_loss = loss; best_code = code;
        }
    }

    if (lane == 0) {
        if (best_code == 0) {
            // A non-finite Hessian should never reach here; preserve a valid artifact with the
            // ordinary un-clipped amax scale and expose the event to the host diagnostic.
            best_code = (int)f32_to_e4m3(
                (amax > 0.f && s_tensor > 0.f) ? amax / 6.f / s_tensor : 0.f);
            atomicAdd(fallback_count, 1ULL);
        }
        qs[(long long)row * (K / 16) + group] = (unsigned char)best_code;
    }
}

// out[:,j] = in[:,perm[j]], and out[i,j] = in[perm[i],perm[j]]. The latter permutes the
// Hessian into the same activation-importance basis as the working weight.
extern "C" __global__ void gptq_permute_w_b(float* out, const float* in, const int* perm, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    int r = (int)(i / K), j = (int)(i % K);
    out[i] = in[(long long)r * K + perm[j]];
}
extern "C" __global__ void gptq_permute_h_b(float* out, const float* in, const int* perm, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)K * K) return;
    int r = (int)(i / K), c = (int)(i % K);
    out[i] = in[(long long)perm[r] * K + perm[c]];
}

// GPTQ sweep in activation-importance order with precomputed static scales. W and U are already
// permuted. Codes are written directly at their ORIGINAL columns, so neither the artifact nor the
// serving kernels need a permutation table.
extern "C" __global__ void gptq_sweep_static_b(float* W, unsigned char* qw,
                                                const unsigned char* qs, float* Err,
                                                const float* U, const int* perm,
                                                int M, int K, int c0, int bs, float s_tensor) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float* w = W + (long long)r * K;
    float* e = Err + (long long)r * bs;
    for (int j = 0; j < bs; j++) {
        int col = c0 + j;
        int orig = perm[col];
        float s = e4m3_f(qs[(long long)r * (K / 16) + orig / 16]) * s_tensor;
        float inv = s > 0.f ? 1.f / s : 0.f;
        unsigned char code = gptq_e2m1(w[col] * inv);
        float q = gptq_e2m1_val(code) * s;
        long long qi = (long long)r * (K / 2) + orig / 2;
        unsigned char byte = qw[qi];
        qw[qi] = (orig & 1) ? ((byte & 0x0F) | (code << 4)) : ((byte & 0xF0) | code);
        float d = U[(long long)col * K + col];
        float err = (w[col] - q) / d;
        e[j] = err;
        const float* urow = U + (long long)col * K;
        for (int k2 = j; k2 < bs; k2++) w[c0 + k2] -= err * urow[c0 + k2];
    }
}

// Compact activation profile for COLA/ACDM. One block owns one feature channel and reduces all
// sequence positions. stats = global sum/sum-of-squares; sketch is a deterministic CountSketch of
// the mean-pooled channel vector (a sparse random projection without materializing [K,N] on host).
extern "C" __global__ void calib_profile_b(double* stats, float* sketch,
                                             const __nv_bfloat16* x, int K, int N,
                                             int sketch_dim, unsigned int seed) {
    int c = blockIdx.x;
    if (c >= K) return;
    double sum = 0.0, sumsq = 0.0;
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        float v = __bfloat162float(x[(long long)t * K + c]);
        sum += (double)v; sumsq += (double)v * (double)v;
    }
    __shared__ double calib_prof_sum[256];
    __shared__ double calib_prof_sumsq[256];
    double* rs = calib_prof_sum; double* rq = calib_prof_sumsq;
    rs[threadIdx.x] = sum; rq[threadIdx.x] = sumsq;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) { rs[threadIdx.x] += rs[threadIdx.x + off]; rq[threadIdx.x] += rq[threadIdx.x + off]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats[0], rs[0]); atomicAdd(&stats[1], rq[0]);
        unsigned int z = ((unsigned int)c ^ seed) * 0x9e3779b9u;
        z ^= z >> 16; z *= 0x85ebca6bu; z ^= z >> 13;
        int bucket = (int)(z % (unsigned int)sketch_dim);
        float sign = (z & 0x80000000u) ? -1.0f : 1.0f;
        atomicAdd(&sketch[bucket], sign * (float)(rs[0] / (double)N));
    }
}
