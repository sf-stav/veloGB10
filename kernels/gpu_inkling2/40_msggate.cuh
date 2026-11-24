// gpu_inkling2/40_msggate.cuh — MessageGate (K1). Section of gpu_inkling2.cu.
//
// MessageGate(h, m) = (1 - tanh(F z)) * h + tanh(G z) * m,  z = down(cat(rms(h), rms(m)))  (rank 64)
//   1. ink_rms2_cat_bf16: cat(rms(h), rms(m)) as bf16 [M][2D] (D9: the bf16 reference feeds `down` a bf16
//      cast of the concat; rms eps = config eps).
//   2. ink_proj_dec / ink_proj_wide (05): z = down . cat, fp32 [M][64] (down stored B32); at decode as nzs
//      cross-CTA k-slices (partials at z + s * zsl, summed in ascending slice order here: 4 tiles alone would
//      leave the 590 KB weight on 4 CTAs).
//   3. ink_msggate_out: forget / gain in fp32 (D9: F, G are in the fp32 keep-list) from the stored bf16
//      weights [D][64], then the gate. grid (D / 256, ceil(M / 16)), block 256: one channel per thread, up
//      to 16 rows per CTA; z rows staged in shared memory. Each output sums r = 0..63 in order (fixed).

// grid (M), block 256.
extern "C" __global__ __launch_bounds__(256) void ink_rms2_cat_bf16(const float* __restrict__ h, const float* __restrict__ m,
        uint16_t* __restrict__ out, int D, float eps) {
    INK2_PDL_ENTRY();
    __shared__ float red[8 * 2];
    const int t = blockIdx.x;
    float ss[2] = {0.f, 0.f};
    for (int c = threadIdx.x; c < D; c += 256) {
        const float a = h[(size_t)t * D + c], b = m[(size_t)t * D + c];
        ss[0] = fmaf(a, a, ss[0]); ss[1] = fmaf(b, b, ss[1]);
    }
    ika_bsum<2, 8>(ss, red);
    const float ra = rsqrtf(ss[0] / (float)D + eps), rb = rsqrtf(ss[1] / (float)D + eps);
    for (int c = threadIdx.x; c < D; c += 256) {
        out[(size_t)t * 2 * D + c] = ika_f2b(h[(size_t)t * D + c] * ra);
        out[(size_t)t * 2 * D + D + c] = ika_f2b(m[(size_t)t * D + c] * rb);
    }
}

// S4: rows-per-CTA template. MT = 16 is the K1 kernel (block 256); MT = 1 / 4 serve decode steps of <= 4 rows with
// 64-channel CTAs (36 instead of 9) and only the live rows' FMAs. A row's arithmetic is the same for every MT (its z
// is summed over the slices in ascending order, its f / g chains run r = 0..63), so the outputs are bit-identical.
template <int MT>
__device__ __forceinline__ void ika_msggate_out(const float* __restrict__ z,
        const uint16_t* __restrict__ wf, const uint16_t* __restrict__ wg,
        const float* __restrict__ h, const float* __restrict__ m, float* __restrict__ out, int D, int M, int nzs, int zsl) {
    __shared__ float sz[MT][64];
    const int t0 = blockIdx.y * MT, mt = min(MT, M - t0);
    for (int i = threadIdx.x; i < MT * 64; i += blockDim.x) {
        const int r = i >> 6;
        float acc = 0.f;
        for (int sl = 0; sl < nzs; ++sl) acc += r < mt ? z[(size_t)sl * zsl + (size_t)(t0 + r) * 64 + (i & 63)] : 0.f;  // fixed order
        sz[r][i & 63] = acc;
    }
    __syncthreads();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float f[MT], g[MT];
    #pragma unroll
    for (int r = 0; r < MT; ++r) { f[r] = 0.f; g[r] = 0.f; }
    const uint4* pf = reinterpret_cast<const uint4*>(wf + (size_t)c * 64);
    const uint4* pg = reinterpret_cast<const uint4*>(wg + (size_t)c * 64);
    #pragma unroll 2
    for (int q = 0; q < 8; ++q) {
        const uint4 a = __ldg(pf + q), b = __ldg(pg + q);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float wfv = ika_b2f((uint16_t)(aw[e >> 1] >> (16 * (e & 1))));
            const float wgv = ika_b2f((uint16_t)(bw[e >> 1] >> (16 * (e & 1))));
            const int r = q * 8 + e;
            #pragma unroll
            for (int row = 0; row < MT; ++row) {
                f[row] = fmaf(sz[row][r], wfv, f[row]);
                g[row] = fmaf(sz[row][r], wgv, g[row]);
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < MT; ++row) {
        if (row >= mt) break;
        const size_t i = (size_t)(t0 + row) * D + c;
        out[i] = (1.0f - tanhf(f[row])) * h[i] + tanhf(g[row]) * m[i];
    }
}
#define INK_MSGGATE_OUT(NAME, MT)                                                                                  \
extern "C" __global__ __launch_bounds__(256) void NAME(const float* __restrict__ z,                                \
        const uint16_t* __restrict__ wf, const uint16_t* __restrict__ wg,                                           \
        const float* __restrict__ h, const float* __restrict__ m, float* __restrict__ out, int D, int M, int nzs, int zsl) { INK2_PDL_ENTRY(); \
    ika_msggate_out<MT>(z, wf, wg, h, m, out, D, M, nzs, zsl);                                                      \
}
INK_MSGGATE_OUT(ink_msggate_out, 16)
INK_MSGGATE_OUT(ink_msggate_out_m1, 1)
INK_MSGGATE_OUT(ink_msggate_out_m4, 4)

// S7: the three attention-input MessageGates of a loop layer (q / k / v streams) in ONE launch per stage: grid.z =
// the gate (0..2), each z running exactly the single-gate body on its own pointers (bit-identical outputs; 3x fewer
// launches of these latency-bound kernels at decode / verify widths).
struct InkMsg3 {
    const float* h[3]; const float* m[3]; uint16_t* cat[3];      // rms2cat
    const uint16_t* down[3]; float* z[3];                        // down GEMV (cross-CTA k-slices)
    const uint16_t* wf[3]; const uint16_t* wg[3]; float* out[3]; // gate epilogue
};
extern "C" __global__ __launch_bounds__(256) void ink_rms2_cat_bf16_x3(const InkMsg3 p, int D, float eps) {
    INK2_PDL_ENTRY();
    const int z = blockIdx.z;
    __shared__ float red[8 * 2];
    const int t = blockIdx.x;
    const float* h = p.h[z]; const float* m = p.m[z]; uint16_t* out = p.cat[z];
    float ss[2] = {0.f, 0.f};
    for (int c = threadIdx.x; c < D; c += 256) {
        const float a = h[(size_t)t * D + c], b = m[(size_t)t * D + c];
        ss[0] = fmaf(a, a, ss[0]); ss[1] = fmaf(b, b, ss[1]);
    }
    ika_bsum<2, 8>(ss, red);
    const float ra = rsqrtf(ss[0] / (float)D + eps), rb = rsqrtf(ss[1] / (float)D + eps);
    for (int c = threadIdx.x; c < D; c += 256) {
        out[(size_t)t * 2 * D + c] = ika_f2b(h[(size_t)t * D + c] * ra);
        out[(size_t)t * 2 * D + D + c] = ika_f2b(m[(size_t)t * D + c] * rb);
    }
}
extern "C" __global__ __launch_bounds__(256) void ink_msg_down_x3(const InkMsg3 p, int M, int K, int ntiles, int ysl) {
    INK2_PDL_ENTRY();
    const int z = blockIdx.z;
    const uint16_t* X = p.cat[z];
    if (M <= 16) ink_proj_dec_body<8>(p.down[z], X, X, X, 1 << 30, 1 << 30, p.z[z], 64, M, K, ntiles, ysl);
    else ink_proj_dec3_body<8>(p.down[z], X, X, X, 1 << 30, 1 << 30, p.z[z], 64, M, K, ntiles, ysl);
}
#define INK_MSGGATE_OUT_X3(NAME, MT)                                                                               \
extern "C" __global__ __launch_bounds__(256) void NAME(const InkMsg3 p, int D, int M, int nzs, int zsl) {          \
    INK2_PDL_ENTRY();                                                                                              \
    const int z = blockIdx.z;                                                                                      \
    ika_msggate_out<MT>(p.z[z], p.wf[z], p.wg[z], p.h[z], p.m[z], p.out[z], D, M, nzs, zsl);                       \
}
INK_MSGGATE_OUT_X3(ink_msggate_out_x3, 16)
INK_MSGGATE_OUT_X3(ink_msggate_out_m1_x3, 1)
INK_MSGGATE_OUT_X3(ink_msggate_out_m4_x3, 4)
