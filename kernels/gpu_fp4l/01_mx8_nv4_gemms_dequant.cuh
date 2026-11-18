// gpu_fp4l/01_mx8_nv4_gemms_dequant.cuh: FP4L weight kernels: loads/conversions, MXFP8 and NVFP4 decode and wide (tensor-core) GEMMs, dequantisers.
// Fixed k order and ascending split-K slab sums, no atomics on data (batch invariance); sm_121a only for the E2M1x2 -> F16x2 cvt.
// Section 1 of 3 of gpu_fp4l.cu: included in order by gpu_fp4l.cu, one translation unit; text is moved verbatim.
// ------------------------------------------------------------------------------------------------
// Loads and conversions
// ------------------------------------------------------------------------------------------------

// Streaming weight loads: read-only, not allocated in L1 (each weight byte is used once per launch).
__device__ __forceinline__ uint2 xf_ldw2(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 xf_ldw4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t xf_ldw1(const void* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// Activation loads: cached (every warp of the CTA, and every CTA, reads the same few rows).
__device__ __forceinline__ uint4 xf_lda4(const __half* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// Two E4M3 bytes (low byte = element 0) -> f16x2 (low half = element 0). Exact.
__device__ __forceinline__ uint32_t xf_e4m3x2(uint32_t two_bytes) {
    uint32_t r;
    const uint16_t v = (uint16_t)two_bytes;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
// One E4M3 byte -> f16x2 broadcast (both halves = the value). Exact.
__device__ __forceinline__ uint32_t xf_e4m3_bcast(uint32_t b) {
    return xf_e4m3x2((b & 0xFF) | ((b & 0xFF) << 8));
}

// Eight E2M1 nibbles (nibble i at bits 4i; low nibble = even k) -> four f16x2 (o[j] = (nib 2j, nib 2j+1)).
// CVT = true: the SM12x cvt.rn.f16x2.e2m1x2 (one instruction per byte). CVT = false: an exact
// integer path (PRMT magnitude LUT + sign bits) — the A/B twin and the fallback if a toolchain
// rejects the cvt form (selected per process by --fp4l-e2m1 cvt|int).
template <bool CVT>
__device__ __forceinline__ void xf_e2m1x8(uint32_t w, uint32_t o[4]) {
    if constexpr (CVT) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
        asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
            "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
            "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
            "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
            "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
            : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
        return;
#endif
    }
    // |e2m1| -> f16 high byte (the low byte of every e2m1 value's f16 is 0):
    // 0 0.5 1 1.5 | 2 3 4 6  ->  00 38 3C 3E | 40 42 44 46
    const uint32_t LUT_LO = 0x3E3C3800u, LUT_HI = 0x46444240u;
    uint32_t mlo, mhi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mlo) : "r"(LUT_LO), "r"(LUT_HI), "r"(w & 0x7777u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mhi) : "r"(LUT_LO), "r"(LUT_HI), "r"((w >> 16) & 0x7777u));
    const uint32_t s = w & 0x88888888u;   // nibble sign bits at 4i+3
    uint32_t r0, r1, r2, r3;
    asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(r0) : "r"(mlo));   // (mag0 << 8) | (mag1 << 24)
    asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(r1) : "r"(mlo));   // (mag2 << 8) | (mag3 << 24)
    asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(r2) : "r"(mhi));
    asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(r3) : "r"(mhi));
    o[0] = r0 | ((s << 12) & 0x00008000u) | ((s << 24) & 0x80000000u);   // bits 3, 7   -> 15, 31
    o[1] = r1 | ((s << 4)  & 0x00008000u) | ((s << 16) & 0x80000000u);   // bits 11, 15 -> 15, 31
    o[2] = r2 | ((s >> 4)  & 0x00008000u) | ((s << 8)  & 0x80000000u);   // bits 19, 23 -> 15, 31
    o[3] = r3 | ((s >> 12) & 0x00008000u) | (s & 0x80000000u);           // bits 27, 31 -> 15, 31
}

// UE8M0 byte -> 2^(b - 127) as f32 (b = 0 -> the subnormal 2^-127; 0xFF is refused at load).
__device__ __forceinline__ float xf_ue8m0(uint32_t b) {
    return __int_as_float(b ? (int)(b << 23) : (1 << 22));
}

__device__ __forceinline__ void xf_mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t xf_hmul2(uint32_t a, uint32_t b) {
    __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

// ------------------------------------------------------------------------------------------------
// Split-K epilogue shared by the dense GEMMs (m <= 16).
//   acc[h][r]: tile h (0/1) of this warp's 16 weight rows; r: (token gid, row 2t), (gid, 2t+1),
//   (gid+8, 2t), (gid+8, 2t+1).
// ks == 1      : f16 straight to Y (or fp32 to Yf when Yf != nullptr).
// ks >  1, mode SLAB  (cnt == nullptr): fp32 slab s -> ws[(s*M + t)*N + n]; the caller combines
//                (xq_ks_combine_f32 for the TP row-parallel partial).
// ks >  1, mode FIXUP (cnt != nullptr): slab as above; the last-arriving CTA of this N-tile sums
//                slabs 0..ks-1 in ascending order into Y (f16) / Yf (fp32) and re-arms cnt[tile].
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void xf_dense_epilogue(float acc[2][4], int M, int N, int n0w, bool warp_live,
                                                  int ks, float* __restrict__ ws, unsigned* __restrict__ cnt,
                                                  __half* __restrict__ Y, float* __restrict__ Yf, float osc) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    // osc: the exact power-of-two output scale of a pre-scaled f16 weight (1.0 for MXFP8): scaling every fp32
    // partial by a power of two is exact, so the fixed-order sum below is the unscaled sum, scaled.
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] *= osc;
    const int t0 = gid, t1 = gid + 8;
    if (ks == 1) {
        if (!warp_live) return;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = n0w + 8 * h + 2 * tig;
            if (n >= N) continue;
            if (Yf) {
                if (t0 < M) *reinterpret_cast<float2*>(&Yf[(size_t)t0 * N + n]) = make_float2(acc[h][0], acc[h][1]);
                if (t1 < M) *reinterpret_cast<float2*>(&Yf[(size_t)t1 * N + n]) = make_float2(acc[h][2], acc[h][3]);
            } else {
                if (t0 < M) *reinterpret_cast<__half2*>(&Y[(size_t)t0 * N + n]) = __floats2half2_rn(acc[h][0], acc[h][1]);
                if (t1 < M) *reinterpret_cast<__half2*>(&Y[(size_t)t1 * N + n]) = __floats2half2_rn(acc[h][2], acc[h][3]);
            }
        }
        return;
    }
    const int s = blockIdx.y;
    if (warp_live) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = n0w + 8 * h + 2 * tig;
            if (n >= N) continue;
            if (t0 < M) *reinterpret_cast<float2*>(&ws[((size_t)s * M + t0) * N + n]) = make_float2(acc[h][0], acc[h][1]);
            if (t1 < M) *reinterpret_cast<float2*>(&ws[((size_t)s * M + t1) * N + n]) = make_float2(acc[h][2], acc[h][3]);
        }
    }
    if (cnt == nullptr) return;
    __shared__ unsigned xf_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(&cnt[blockIdx.x], 1u);
        xf_last = (prev == (unsigned)ks - 1u) ? 1u : 0u;
    }
    __syncthreads();
    if (!xf_last) return;
    __threadfence();
    // the CTA's 128 rows x M tokens: thread i takes (token, column pair) items in a fixed map
    const int nbase = blockIdx.x * 128;
    const int items = M * 64;   // 64 column pairs per tile
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int t = it / 64, n = nbase + 2 * (it % 64);
        if (n >= N) continue;
        float a = 0.f, b = 0.f;
        for (int q = 0; q < ks; ++q) {
            const float2 v = __ldcg(reinterpret_cast<const float2*>(&ws[((size_t)q * M + t) * N + n]));
            a += v.x; b += v.y;
        }
        if (Yf) *reinterpret_cast<float2*>(&Yf[(size_t)t * N + n]) = make_float2(a, b);
        else    *reinterpret_cast<__half2*>(&Y[(size_t)t * N + n]) = __floats2half2_rn(a, b);
    }
    if (threadIdx.x == 0) cnt[blockIdx.x] = 0u;   // re-arm (stream-ordered: the next launch sees 0)
}

// ------------------------------------------------------------------------------------------------
// MXFP8 F64 fragment helpers: one 64-k block of one 16-row tile against token rows (gid, gid+8).
// w0 / w1: the lane's 16 code bytes of halves h = 0 / 1; sc: its 8 scale bytes; xa0/xa1 (xb0/xb1): the
// activations of token row gid (gid+8) at k 64b+8tig.. and 64b+32+8tig... Per 32-k group gs: two MMAs into a
// zeroed fp32 tile, then x 2^(s-127) (exact) into acc — the row-major kernel's per-group arithmetic, unchanged.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void xf_mx8_block(float acc[2][4], const uint4 w0, const uint4 w1, const uint2 sc,
                                             const uint4 xa0, const uint4 xa1, const uint4 xb0, const uint4 xb1) {
    #pragma unroll
    for (int gs = 0; gs < 2; ++gs) {
        const uint4 ax = gs ? xa1 : xa0, bx = gs ? xb1 : xb0;
        const uint32_t c0lo = gs ? w0.z : w0.x, c0hi = gs ? w0.w : w0.y;
        const uint32_t c1lo = gs ? w1.z : w1.x, c1hi = gs ? w1.w : w1.y;
        float p0[4] = {0.f, 0.f, 0.f, 0.f}, p1[4] = {0.f, 0.f, 0.f, 0.f};
        xf_mma(p0, ax.x, bx.x, ax.y, bx.y, xf_e4m3x2(c0lo), xf_e4m3x2(c0lo >> 16));
        xf_mma(p0, ax.z, bx.z, ax.w, bx.w, xf_e4m3x2(c0hi), xf_e4m3x2(c0hi >> 16));
        xf_mma(p1, ax.x, bx.x, ax.y, bx.y, xf_e4m3x2(c1lo), xf_e4m3x2(c1lo >> 16));
        xf_mma(p1, ax.z, bx.z, ax.w, bx.w, xf_e4m3x2(c1hi), xf_e4m3x2(c1hi >> 16));
        const float f00 = xf_ue8m0((sc.x >> (8 * gs)) & 0xFF), f01 = xf_ue8m0((sc.x >> (8 * (2 + gs))) & 0xFF);
        const float f10 = xf_ue8m0((sc.y >> (8 * gs)) & 0xFF), f11 = xf_ue8m0((sc.y >> (8 * (2 + gs))) & 0xFF);
        acc[0][0] = fmaf(p0[0], f00, acc[0][0]); acc[0][1] = fmaf(p0[1], f01, acc[0][1]);
        acc[0][2] = fmaf(p0[2], f00, acc[0][2]); acc[0][3] = fmaf(p0[3], f01, acc[0][3]);
        acc[1][0] = fmaf(p1[0], f10, acc[1][0]); acc[1][1] = fmaf(p1[1], f11, acc[1][1]);
        acc[1][2] = fmaf(p1[2], f10, acc[1][2]); acc[1][3] = fmaf(p1[3], f11, acc[1][3]);
    }
}

// ------------------------------------------------------------------------------------------------
// xf_mx8_gemm — MXFP8 W8A16, M <= 16 tokens (decode / verify / packed lanes), F64 layout.
//   grid (ceil(N/128), ks), block 256 (8 warps x one 16-row tile each). K % 64 == 0, (K/ks) % 64 == 0.
//   Per 64-k block a lane issues two 16-B code loads (512 contiguous bytes per warp each), one 8-B scale load and
//   four 16-B activation loads; 8 MMAs. U = 4 blocks in flight per warp.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void xf_mx8_gemm_dev(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, float* __restrict__ ws, unsigned* __restrict__ cnt,
        int M, int K, int N, int ks) {
    constexpr int U = 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    const bool live = n0w < N;   // warp-uniform
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    if (live) {
        const int KB = K >> 6;
        const int kbs = KB / ks;
        const int b0 = blockIdx.y * kbs, b1 = b0 + kbs;
        const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
        const __half* xa = X + (size_t)ta * K + 8 * tig;
        const __half* xb = X + (size_t)tb * K + 8 * tig;
        const uint8_t* wt = W + (size_t)tile * KB * 1024 + lane * 16;
        const uint8_t* st = S + (size_t)tile * KB * 32 + tig * 8;
        for (int b = b0; b < b1; b += U) {
            uint4 w0[U], w1[U], xa0[U], xa1[U], xb0[U], xb1[U];
            uint2 sc[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                if (b + u < b1) {
                    w0[u] = xf_ldw4(wt + (size_t)(b + u) * 1024);
                    w1[u] = xf_ldw4(wt + (size_t)(b + u) * 1024 + 512);
                    sc[u] = __ldg(reinterpret_cast<const uint2*>(st + (size_t)(b + u) * 32));
                    const int k = (b + u) * 64;
                    xa0[u] = xf_lda4(xa + k); xa1[u] = xf_lda4(xa + k + 32);
                    xb0[u] = xf_lda4(xb + k); xb1[u] = xf_lda4(xb + k + 32);
                }
            }
            #pragma unroll
            for (int u = 0; u < U; ++u)
                if (b + u < b1) xf_mx8_block(acc, w0[u], w1[u], sc[u], xa0[u], xa1[u], xb0[u], xb1[u]);
        }
    }
    xf_dense_epilogue(acc, M, N, n0w, live, ks, ws, cnt, Y, Yf, 1.0f);
}

extern "C" __global__ __launch_bounds__(256) void xf_mx8_gemm(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, float* __restrict__ ws, unsigned* __restrict__ cnt,
        int M, int K, int N, int ks) {
    XF_PDL_ENTRY();
    xf_mx8_gemm_dev(W, S, X, Y, Yf, ws, cnt, M, K, N, ks);
}

// ------------------------------------------------------------------------------------------------
// xf_mx8_gemm_x2 — two xf_mx8_gemm launches that read the same X with the same M, K, N and ks (the GDN in_proj_a |
//   in_proj_b pair, the shared expert's gate | up) in ONE launch: grid (ceil(N/128), ks, 2), blockIdx.z = member.
//   Each member keeps its own weights, fp32 slabs and fixup counters and runs xf_mx8_gemm's body unchanged, so each
//   output is bitwise its xf_mx8_gemm result. f16 outputs only (the fixup / ks == 1 store).
// ------------------------------------------------------------------------------------------------
struct XfPair {   // by value; the host twin is fp4l_fwd::XfPair (repr(C), the same field order)
    const uint8_t* w[2];
    const uint8_t* s[2];
    __half* y[2];
    float* ws[2];
    unsigned* cnt[2];   // nullptr when ks == 1
};
extern "C" __global__ __launch_bounds__(256) void xf_mx8_gemm_x2(
        const XfPair p, const __half* __restrict__ X, int M, int K, int N, int ks) {
    XF_PDL_ENTRY();
    // member select by ternaries: a runtime index into the by-value param arrays would copy them to local memory
    const bool z = blockIdx.z != 0;
    xf_mx8_gemm_dev(z ? p.w[1] : p.w[0], z ? p.s[1] : p.s[0], X, z ? p.y[1] : p.y[0], nullptr,
                    z ? p.ws[1] : p.ws[0], z ? p.cnt[1] : p.cnt[0], M, K, N, ks);
}

// ------------------------------------------------------------------------------------------------
// xf_f16w_gemm — plain f16 weights [N][K], M <= 16 (the FP4-Local lm_head, the bf16 MTP linears).
//   Same grid / fragment / epilogue contract as xf_mx8_gemm. K % 128 == 0.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void xf_f16w_gemm(
        const __half* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, float* __restrict__ ws, unsigned* __restrict__ cnt,
        int M, int K, int N, int ks, float osc) {
    XF_PDL_ENTRY();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int n0w = blockIdx.x * 128 + warp * 16;
    const bool live = n0w < N;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    if (live) {
        const int kslab = K / ks;
        const int kb = blockIdx.y * kslab, ke = kb + kslab;
        const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
        const __half* xa = X + (size_t)ta * K + 8 * tig;
        const __half* xb = X + (size_t)tb * K + 8 * tig;
        const __half* w0 = W + (size_t)min(n0w + gid, N - 1) * K + 8 * tig;
        const __half* w1 = W + (size_t)min(n0w + 8 + gid, N - 1) * K + 8 * tig;
        for (int k = kb; k < ke; k += 128) {
            uint4 b0[4], b1[4], xa4[4], xb4[4];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                b0[g] = xf_ldw4(w0 + k + 32 * g);
                b1[g] = xf_ldw4(w1 + k + 32 * g);
                xa4[g] = xf_lda4(xa + k + 32 * g);
                xb4[g] = xf_lda4(xb + k + 32 * g);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                xf_mma(acc[0], xa4[g].x, xb4[g].x, xa4[g].y, xb4[g].y, b0[g].x, b0[g].y);
                xf_mma(acc[0], xa4[g].z, xb4[g].z, xa4[g].w, xb4[g].w, b0[g].z, b0[g].w);
                xf_mma(acc[1], xa4[g].x, xb4[g].x, xa4[g].y, xb4[g].y, b1[g].x, b1[g].y);
                xf_mma(acc[1], xa4[g].z, xb4[g].z, xa4[g].w, xb4[g].w, b1[g].z, b1[g].w);
            }
        }
    }
    xf_dense_epilogue(acc, M, N, n0w, live, ks, ws, cnt, Y, Yf, osc);
}

// ------------------------------------------------------------------------------------------------
// xf_mx8_wide — MXFP8 W8A16 for PREFILL row counts (M > 16), EXACT like xf_mx8_gemm (each 32-k group's
// fp32 MMA partial scaled by its power of two; no f16 dequant of the weights).
//   grid (ceil(N/128), ceil(M / (16*MT))), block 256 (8 warps x 16 weight rows), MT token tiles per CTA:
//   each lane's dequantised B fragments are reused across the MT 16-row token tiles. N % 2 == 0, K % 128 == 0.
//   X [M][K] f16 -> Y [M][N] f16 (or Yf fp32 when Yf != nullptr: the TP row-parallel partial).
// ------------------------------------------------------------------------------------------------
template <int MT>
__device__ __forceinline__ void xf_mx8_wide_body(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    if (n0w >= N) return;   // no barriers in this kernel: a warp past N simply leaves
    const int t0 = blockIdx.y * 16 * MT;
    const int KB = K >> 6;
    const uint8_t* wt = W + (size_t)tile * KB * 1024 + lane * 16;
    const uint8_t* st = S + (size_t)tile * KB * 32 + tig * 8;
    const __half* xa[MT];
    const __half* xb[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        xa[mt] = X + (size_t)min(t0 + 16 * mt + gid, M - 1) * K + 8 * tig;
        xb[mt] = X + (size_t)min(t0 + 16 * mt + gid + 8, M - 1) * K + 8 * tig;
    }
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
    for (int b = 0; b < KB; ++b) {
        const uint4 w0 = xf_ldw4(wt + (size_t)b * 1024), w1 = xf_ldw4(wt + (size_t)b * 1024 + 512);
        const uint2 sc = __ldg(reinterpret_cast<const uint2*>(st + (size_t)b * 32));
        const int k = b * 64;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            xf_mx8_block(acc[mt], w0, w1, sc, xf_lda4(xa[mt] + k), xf_lda4(xa[mt] + k + 32),
                         xf_lda4(xb[mt] + k), xf_lda4(xb[mt] + k + 32));
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int ta = t0 + 16 * mt + gid, tb = ta + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = n0w + 8 * h + 2 * tig;
            if (n >= N) continue;
            if (Yf) {
                if (ta < M) *reinterpret_cast<float2*>(&Yf[(size_t)ta * N + n]) = make_float2(acc[mt][h][0], acc[mt][h][1]);
                if (tb < M) *reinterpret_cast<float2*>(&Yf[(size_t)tb * N + n]) = make_float2(acc[mt][h][2], acc[mt][h][3]);
            } else {
                if (ta < M) *reinterpret_cast<__half2*>(&Y[(size_t)ta * N + n]) = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
                if (tb < M) *reinterpret_cast<__half2*>(&Y[(size_t)tb * N + n]) = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
            }
        }
    }
}
extern "C" __global__ __launch_bounds__(256) void xf_mx8_wide(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    XF_PDL_ENTRY();
    xf_mx8_wide_body<4>(W, S, X, Y, Yf, M, K, N);
}

__device__ __forceinline__ void xf_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void xf_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void xf_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// ------------------------------------------------------------------------------------------------
// xf_mx8_tc — the exact MXFP8 (F64) x f16 GEMM at PREFILL row counts (m > 16), smem-tiled: the served
// replacement of xf_mx8_wide (bit-identical to it; 3.5-4.9x faster on GB10, microbench 2026-10-29: 61-72
// TFLOPS of the ~122 mma.sync f16 peak vs 15-21).
// CTA tile 128 tokens x 128 weight rows, 64-k steps, 2 smem stages (48 KB static), 2 CTAs / SM (the 128-register
// bound spills ~60 B of loop-invariant addressing, measured faster than 1 CTA / SM at 200 registers). 8 warps = 2 (tokens) x 4 (rows);
// warp tile 64 tokens (4 m16 tiles) x 32 weight rows (2 F64 tiles). Per 32-k group the decode kernels' arithmetic
// (p = two MMAs from zero, acc = fma(p, 2^(s-127), acc)), groups ascending: every output equals xf_mx8_wide's bits.
// Activation smem rows are 128 B (64 k) with the 16-B chunk index XOR (row & 7) (conflict-free 16-B reads).
// Grid: 1-D, CTA id -> (n block, m block) in groups of GM m-blocks (L2 reuse of the activation rows across N).
constexpr int XF_TC_GM = 8;
extern "C" __global__ __launch_bounds__(256, 2) void xf_mx8_tc(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    XF_PDL_ENTRY();
    __shared__ __align__(128) uint8_t smA[2][128 * 128];
    __shared__ __align__(128) uint8_t smB[2][8 * 1024];
    const int nb = (N + 127) >> 7, mb = (M + 127) >> 7;
    // grouped raster: GM m-blocks per group, the group's CTAs sweep N with m fastest
    const int id = blockIdx.x;
    const int per_group = XF_TC_GM * nb;
    const int g = id / per_group, first_m = g * XF_TC_GM;
    const int gm = min(mb - first_m, XF_TC_GM);
    const int m_blk = first_m + (id % per_group) % gm;
    const int n_blk = (id % per_group) / gm;
    const int m0 = m_blk * 128, n0 = n_blk * 128;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int KB = K >> 6;
    const int ntiles = (N + 15) >> 4;
    // ---- loader assignments
    // A: 1024 chunks of 16 B (128 rows x 8 chunks): thread t -> chunks t + 256 j
    // B: 512 chunks (8 tiles x 64 chunks): thread t -> chunks t, t + 256
    auto load_stage = [&](int st, int b) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tid + 256 * j, row = c >> 3, ch = c & 7;
            const int tok = min(m0 + row, M - 1);
            xf_cp16(&smA[st][row * 128 + ((ch ^ (row & 7)) << 4)], X + (size_t)tok * K + b * 64 + ch * 8);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + 256 * j, tl = c >> 6, o = (c & 63) << 4;
            const int t = min((n0 >> 4) + tl, ntiles - 1);
            xf_cp16(&smB[st][tl * 1024 + o], W + ((size_t)t * KB + b) * 1024 + o);
        }
    };
    float acc[4][2][2][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a)
        #pragma unroll
        for (int b2 = 0; b2 < 2; ++b2)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[a][b2][h][r] = 0.f;
    // scales: the warp's 2 tiles, lane tig's 8 B per tile per 64-k block
    const int t_w0 = min((n0 >> 4) + wn * 2, ntiles - 1), t_w1 = min((n0 >> 4) + wn * 2 + 1, ntiles - 1);
    const uint8_t* s0p = S + (size_t)t_w0 * KB * 32 + tig * 8;
    const uint8_t* s1p = S + (size_t)t_w1 * KB * 32 + tig * 8;
    load_stage(0, 0);
    xf_cp_commit();
    uint2 sc0 = __ldg(reinterpret_cast<const uint2*>(s0p)), sc1 = __ldg(reinterpret_cast<const uint2*>(s1p));
    for (int b = 0; b < KB; ++b) {
        const int st = b & 1;
        if (b + 1 < KB) load_stage(st ^ 1, b + 1);
        xf_cp_commit();
        uint2 nsc0 = sc0, nsc1 = sc1;
        if (b + 1 < KB) {
            nsc0 = __ldg(reinterpret_cast<const uint2*>(s0p + (size_t)(b + 1) * 32));
            nsc1 = __ldg(reinterpret_cast<const uint2*>(s1p + (size_t)(b + 1) * 32));
        }
        xf_cp_wait<1>();
        __syncthreads();
        // B fragments of the warp's two tiles (both halves), this lane
        const uint4 w00 = *reinterpret_cast<const uint4*>(&smB[st][(wn * 2) * 1024 + lane * 16]);
        const uint4 w01 = *reinterpret_cast<const uint4*>(&smB[st][(wn * 2) * 1024 + 512 + lane * 16]);
        const uint4 w10 = *reinterpret_cast<const uint4*>(&smB[st][(wn * 2 + 1) * 1024 + lane * 16]);
        const uint4 w11 = *reinterpret_cast<const uint4*>(&smB[st][(wn * 2 + 1) * 1024 + 512 + lane * 16]);
        #pragma unroll
        for (int gs = 0; gs < 2; ++gs) {
            uint32_t bf[2][2][4];
            {
                const uint4 v[2][2] = {{w00, w01}, {w10, w11}};
                #pragma unroll
                for (int jt = 0; jt < 2; ++jt)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t lo = gs ? v[jt][h].z : v[jt][h].x, hi = gs ? v[jt][h].w : v[jt][h].y;
                        bf[jt][h][0] = xf_e4m3x2(lo); bf[jt][h][1] = xf_e4m3x2(lo >> 16);
                        bf[jt][h][2] = xf_e4m3x2(hi); bf[jt][h][3] = xf_e4m3x2(hi >> 16);
                    }
            }
            float fs[2][2][2];
            {
                const uint2 scs[2] = {sc0, sc1};
                #pragma unroll
                for (int jt = 0; jt < 2; ++jt) {
                    fs[jt][0][0] = xf_ue8m0((scs[jt].x >> (8 * gs)) & 0xFF); fs[jt][0][1] = xf_ue8m0((scs[jt].x >> (8 * (2 + gs))) & 0xFF);
                    fs[jt][1][0] = xf_ue8m0((scs[jt].y >> (8 * gs)) & 0xFF); fs[jt][1][1] = xf_ue8m0((scs[jt].y >> (8 * (2 + gs))) & 0xFF);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int ra = wm * 64 + mi * 16 + gid, rb = ra + 8;
                const int ch = gs * 4 + tig;
                const uint4 xa = *reinterpret_cast<const uint4*>(&smA[st][ra * 128 + ((ch ^ (ra & 7)) << 4)]);
                const uint4 xb = *reinterpret_cast<const uint4*>(&smA[st][rb * 128 + ((ch ^ (rb & 7)) << 4)]);
                #pragma unroll
                for (int jt = 0; jt < 2; ++jt)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float p[4] = {0.f, 0.f, 0.f, 0.f};
                        xf_mma(p, xa.x, xb.x, xa.y, xb.y, bf[jt][h][0], bf[jt][h][1]);
                        xf_mma(p, xa.z, xb.z, xa.w, xb.w, bf[jt][h][2], bf[jt][h][3]);
                        float* a = acc[mi][jt][h];
                        a[0] = fmaf(p[0], fs[jt][h][0], a[0]); a[1] = fmaf(p[1], fs[jt][h][1], a[1]);
                        a[2] = fmaf(p[2], fs[jt][h][0], a[2]); a[3] = fmaf(p[3], fs[jt][h][1], a[3]);
                    }
            }
        }
        sc0 = nsc0; sc1 = nsc1;
        __syncthreads();
    }
    // epilogue
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int ta = m0 + wm * 64 + mi * 16 + gid, tb = ta + 8;
        #pragma unroll
        for (int jt = 0; jt < 2; ++jt)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = n0 + wn * 32 + jt * 16 + h * 8 + 2 * tig;
                if (n >= N) continue;
                const float* a = acc[mi][jt][h];
                if (Yf) {
                    if (ta < M) *reinterpret_cast<float2*>(&Yf[(size_t)ta * N + n]) = make_float2(a[0], a[1]);
                    if (tb < M) *reinterpret_cast<float2*>(&Yf[(size_t)tb * N + n]) = make_float2(a[2], a[3]);
                } else {
                    if (ta < M) *reinterpret_cast<__half2*>(&Y[(size_t)ta * N + n]) = __floats2half2_rn(a[0], a[1]);
                    if (tb < M) *reinterpret_cast<__half2*>(&Y[(size_t)tb * N + n]) = __floats2half2_rn(a[2], a[3]);
                }
            }
    }
}

// f16 [n] * scale -> bf16 [n] (the FP4L DFlash2 head: the pre-scaled f16 lm_head back to its stored bf16, exact).
extern "C" __global__ void xf_f16_to_bf16_scaled(uint16_t* __restrict__ out, const __half* __restrict__ in,
                                                 long long n, float scale) {
    XF_PDL_ENTRY();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float v = __half2float(in[i]) * scale;
        out[i] = (uint16_t)(__float_as_uint(v) >> 16);   // exact: v is a bf16 value (truncation == the value)
    }
}

// ------------------------------------------------------------------------------------------------
// NVFP4 expert layout "F128" (fragment-native; fp4l_fwd::nv4_frag, a pure byte permutation of the checkpoint's
// [N][K/2] nibbles / [N][K/16] E4M3 scales made at load, per expert matrix; N % 16 == 0, K % 128 == 0):
//   codes  W'[t][b][h][lane][16]: lane = gid*4 + tig, half h -> row 16t+8h+gid; bytes 0..7 = the row's nibbles of
//          k = 128b + 16tig .. +15 (64-k step 2b), bytes 8..15 = k = 128b + 64 + 16tig .. +15 (step 2b+1)
//   scales S'[t][b][lane][4]: byte 2h + s = the E4M3 scale of row 16t+8h+gid, step 2b+s, block tig
// One 128-k block of one 16-row tile = 1024 contiguous code bytes (two 16-B loads per lane, 512 B per warp
// each) + 128 contiguous scale bytes (one 4-B load per lane). Per 64-k step the arithmetic is the row-major
// kernel's, in the same order (bit-identical): 8 cvt + 8 HMUL2 per row half, 8 MMAs.
// ------------------------------------------------------------------------------------------------
template <bool CVT>
__device__ __forceinline__ void xf_nv4_step(float acc[2][4], const uint32_t n0x, const uint32_t n0y, const uint32_t n1x,
                                            const uint32_t n1y, const uint32_t sb0, const uint32_t sb1,
                                            const uint4 xa0, const uint4 xa1, const uint4 xb0, const uint4 xb1) {
    uint32_t b0[8], b1[8];
    xf_e2m1x8<CVT>(n0x, &b0[0]); xf_e2m1x8<CVT>(n0y, &b0[4]);
    xf_e2m1x8<CVT>(n1x, &b1[0]); xf_e2m1x8<CVT>(n1y, &b1[4]);
    const uint32_t f0 = xf_e4m3_bcast(sb0), f1 = xf_e4m3_bcast(sb1);
    #pragma unroll
    for (int j = 0; j < 8; ++j) { b0[j] = xf_hmul2(b0[j], f0); b1[j] = xf_hmul2(b1[j], f1); }
    // MMA j covers actual k = 16t + 4j .. 4j+3: A halves (4j, 4j+1) and (4j+2, 4j+3)
    xf_mma(acc[0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
    xf_mma(acc[0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
    xf_mma(acc[0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
    xf_mma(acc[0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
    xf_mma(acc[1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
    xf_mma(acc[1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
    xf_mma(acc[1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
    xf_mma(acc[1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
}

// ------------------------------------------------------------------------------------------------
// xf_nv4_moe — NVFP4 routed experts, M <= 16 tokens (decode / verify), the union-of-experts
// schedule of the EXL3 trunk: every ACTIVE expert slot computes all M rows once (xq_moe_route /
// the router folds built slotmap/idxmap/esel; idxmap holds the LOCAL stacked index under EP).
//   grid (N/64, emax), block 128 (4 warps x one 16-row tile). K % 128 == 0, N % 64 == 0. F128 layout.
//   W  [E] x F128(N, K), S [E] x F128 scales, GS [E][3] fp32 = {gate, up, down} multipliers.
//   gu mode (nsplit = N/2): rows < nsplit take GS[e][0], rows >= nsplit GS[e][1]; X = x [M][K]
//      (x_slot_stride 0). dn mode (nsplit = N): GS[e][2]; X = din [slot][M][K].
//   Y [slot][M][N] f16 (gu: the fused gate|up row xq_moe_gate_mul reads).
// U = 4 128-k blocks of weights in flight per warp (36 registers); activations are L1-resident.
// ------------------------------------------------------------------------------------------------
template <bool CVT>
__device__ __forceinline__ void xf_nv4_moe_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, long long x_slot_stride, __half* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, int nsplit) {
    constexpr int U = 4;
    const int slot = blockIdx.y;
    if (slot >= *esel) return;
    const int e = idxmap[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    const int n0w = tile * 16;
    const int KB = K >> 7;
    const uint8_t* wt = W + (size_t)e * N * (K >> 1) + (size_t)tile * KB * 1024 + lane * 16;
    const uint8_t* st = S + (size_t)e * N * (K >> 4) + (size_t)tile * KB * 128 + lane * 4;
    const __half* Xs = X + (size_t)slot * x_slot_stride;
    const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
    const __half* xa = Xs + (size_t)ta * K + 16 * tig;
    const __half* xb = Xs + (size_t)tb * K + 16 * tig;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    for (int b = 0; b < KB; b += U) {
        uint4 w0[U], w1[U];
        uint32_t sc[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                w0[u] = xf_ldw4(wt + (size_t)(b + u) * 1024);
                w1[u] = xf_ldw4(wt + (size_t)(b + u) * 1024 + 512);
                sc[u] = __ldg(reinterpret_cast<const uint32_t*>(st + (size_t)(b + u) * 128));
            }
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            if (b + u < KB) {
                const int k = (b + u) * 128;
                xf_nv4_step<CVT>(acc, w0[u].x, w0[u].y, w1[u].x, w1[u].y, sc[u] & 0xFF, (sc[u] >> 16) & 0xFF,
                                 xf_lda4(xa + k), xf_lda4(xa + k + 8), xf_lda4(xb + k), xf_lda4(xb + k + 8));
                xf_nv4_step<CVT>(acc, w0[u].z, w0[u].w, w1[u].z, w1[u].w, (sc[u] >> 8) & 0xFF, sc[u] >> 24,
                                 xf_lda4(xa + k + 64), xf_lda4(xa + k + 72), xf_lda4(xb + k + 64), xf_lda4(xb + k + 72));
            }
        }
    }
    const float gs = (nsplit == N) ? GS[3 * e + 2] : GS[3 * e + (n0w >= nsplit ? 1 : 0)];
    __half* Ys = Y + (size_t)slot * M * N;
    const int t0 = gid, t1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int n = n0w + 8 * h + 2 * tig;
        if (t0 < M) *reinterpret_cast<__half2*>(&Ys[(size_t)t0 * N + n]) = __floats2half2_rn(acc[h][0] * gs, acc[h][1] * gs);
        if (t1 < M) *reinterpret_cast<__half2*>(&Ys[(size_t)t1 * N + n]) = __floats2half2_rn(acc[h][2] * gs, acc[h][3] * gs);
    }
}
extern "C" __global__ __launch_bounds__(128) void xf_nv4_moe(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, long long x_slot_stride, __half* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, int nsplit) {
    XF_PDL_ENTRY();
    xf_nv4_moe_body<true>(W, S, GS, X, x_slot_stride, Y, idxmap, esel, M, K, N, nsplit);
}
extern "C" __global__ __launch_bounds__(128) void xf_nv4_moe_i(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, long long x_slot_stride, __half* __restrict__ Y,
        const int* __restrict__ idxmap, const int* __restrict__ esel, int M, int K, int N, int nsplit) {
    XF_PDL_ENTRY();
    xf_nv4_moe_body<false>(W, S, GS, X, x_slot_stride, Y, idxmap, esel, M, K, N, nsplit);
}

// ------------------------------------------------------------------------------------------------
// xf_nv4_moe_tiles — NVFP4 routed experts for PREFILL chunks: the EXL3 device-routed tile lists
// (xq_moe_pf_count / xq_moe_pf_place: tiles[2t] = expert, tiles[2t+1] = super-row; offs_row / cnt
// per expert over the COMPACT rows, expert-major).
//   grid (N/128, max_tiles), block 256 (8 warps x one F128 16-row tile). One CTA = 128 weight rows x up
//   to 16*MT compact rows of one expert. The dequantised B fragments are reused across the MT
//   16-row token tiles.
//   rowmap != nullptr: A row r of the compact block reads X[rowmap[r]] (gu: x [c][K] via row_tok);
//   rowmap == nullptr: A row r reads X[r] (dn: din [rows][K] already compact).
//   Y [rows][N] f16 (compact rows), GS as in xf_nv4_moe.
// ------------------------------------------------------------------------------------------------
//   kn = K | N << 16; nsplit_mt = nsplit | mt_list << 16 (mt_list = the tile list's 16-row tiles per
//   super-row, a multiple of MT; the CTA walks it in mt_list / MT register blocks). 12 parameters
//   (the cudarc launch tuple limit).
template <bool CVT, int MT>
__device__ __forceinline__ void xf_nv4_tiles_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles,
        const int* __restrict__ offs_row, const int* __restrict__ cnt, int kn, int nsplit_mt) {
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int K = kn & 0xFFFF, N = (int)((unsigned)kn >> 16);
    const int nsplit = nsplit_mt & 0xFFFF;
    const int mt_list = max((int)((unsigned)nsplit_mt >> 16), MT);
    const int e = tiles[2 * tile], msup = tiles[2 * tile + 1];
    const int ce = cnt[e];
    for (int sub = 0; sub < mt_list / MT; ++sub) {
    const int r0 = msup * 16 * mt_list + sub * 16 * MT;
    if (r0 >= ce) return;
    const int rows = min(16 * MT, ce - r0);
    const int base = offs_row[e] + r0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile16 = blockIdx.x * 8 + warp;
    const int n0w = tile16 * 16;
    const int KB = K >> 7;
    const uint8_t* wt = W + (size_t)e * N * (K >> 1) + (size_t)tile16 * KB * 1024 + lane * 16;
    const uint8_t* st = S + (size_t)e * N * (K >> 4) + (size_t)tile16 * KB * 128 + lane * 4;
    const __half* xa[MT];
    const __half* xb[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int ra = base + min(16 * mt + gid, rows - 1), rb = base + min(16 * mt + gid + 8, rows - 1);
        const int ia = rowmap ? rowmap[ra] : ra, ib = rowmap ? rowmap[rb] : rb;
        xa[mt] = X + (size_t)ia * K + 16 * tig;
        xb[mt] = X + (size_t)ib * K + 16 * tig;
    }
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
    for (int b = 0; b < KB; ++b) {
        const uint4 w0 = xf_ldw4(wt + (size_t)b * 1024), w1 = xf_ldw4(wt + (size_t)b * 1024 + 512);
        const uint32_t sc = __ldg(reinterpret_cast<const uint32_t*>(st + (size_t)b * 128));
        #pragma unroll 1
        for (int s2 = 0; s2 < 2; ++s2) {
            const int k = b * 128 + 64 * s2;
            uint32_t b0[8], b1[8];
            xf_e2m1x8<CVT>(s2 ? w0.z : w0.x, &b0[0]); xf_e2m1x8<CVT>(s2 ? w0.w : w0.y, &b0[4]);
            xf_e2m1x8<CVT>(s2 ? w1.z : w1.x, &b1[0]); xf_e2m1x8<CVT>(s2 ? w1.w : w1.y, &b1[4]);
            const uint32_t f0 = xf_e4m3_bcast((sc >> (8 * s2)) & 0xFF), f1 = xf_e4m3_bcast((sc >> (16 + 8 * s2)) & 0xFF);
            #pragma unroll
            for (int j = 0; j < 8; ++j) { b0[j] = xf_hmul2(b0[j], f0); b1[j] = xf_hmul2(b1[j], f1); }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint4 xa0 = xf_lda4(xa[mt] + k), xa1 = xf_lda4(xa[mt] + k + 8);
                const uint4 xb0 = xf_lda4(xb[mt] + k), xb1 = xf_lda4(xb[mt] + k + 8);
                xf_mma(acc[mt][0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
                xf_mma(acc[mt][0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
                xf_mma(acc[mt][0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
                xf_mma(acc[mt][0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
                xf_mma(acc[mt][1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
                xf_mma(acc[mt][1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
                xf_mma(acc[mt][1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
                xf_mma(acc[mt][1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
            }
        }
    }
    const float gs = (nsplit == N) ? GS[3 * e + 2] : GS[3 * e + (n0w >= nsplit ? 1 : 0)];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int t0 = 16 * mt + gid, t1 = t0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = n0w + 8 * h + 2 * tig;
            if (t0 < rows) *reinterpret_cast<__half2*>(&Y[(size_t)(base + t0) * N + n]) =
                __floats2half2_rn(acc[mt][h][0] * gs, acc[mt][h][1] * gs);
            if (t1 < rows) *reinterpret_cast<__half2*>(&Y[(size_t)(base + t1) * N + n]) =
                __floats2half2_rn(acc[mt][h][2] * gs, acc[mt][h][3] * gs);
        }
    }
    }   // sub-block
}
#define XF_TILES_ENTRY(NAME, CVT, MT)                                                                   \
extern "C" __global__ __launch_bounds__(256, 1) void NAME(                                              \
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,     \
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Y,           \
        const int* __restrict__ tiles, const int* __restrict__ ntiles,                                  \
        const int* __restrict__ offs_row, const int* __restrict__ cnt, int kn, int nsplit_mt) {        \
    XF_PDL_ENTRY();                                                                                     \
    xf_nv4_tiles_body<CVT, MT>(W, S, GS, X, rowmap, Y, tiles, ntiles, offs_row, cnt, kn, nsplit_mt);   \
}
XF_TILES_ENTRY(xf_nv4_tiles_m1, true, 1)
XF_TILES_ENTRY(xf_nv4_tiles_m2, true, 2)
XF_TILES_ENTRY(xf_nv4_tiles_m4, true, 4)
XF_TILES_ENTRY(xf_nv4_tiles_m1_i, false, 1)
XF_TILES_ENTRY(xf_nv4_tiles_m2_i, false, 2)
XF_TILES_ENTRY(xf_nv4_tiles_m4_i, false, 4)
#undef XF_TILES_ENTRY

// ------------------------------------------------------------------------------------------------
// xf_nv4_tc — the NVFP4 routed experts at PREFILL (tile-list classes mt 4 and 8), smem-tiled: the served
// replacement of xf_nv4_tiles_m4 for those classes, bit-identical to it (same per-64-k-step MMA sequence per
// output; the activation rows are staged once per CTA in shared memory instead of re-read by every warp).
//   grid (N/128, max_tiles), block 256 (8 warps x one F128 16-row tile), 64 compact rows per sub-block, 128-k
//   steps, 2 smem stages (48 KB static: activations 2 x 16 KB, codes 2 x 8 KB; the scales ride in registers).
//   Same 12 parameters as xf_nv4_tiles_m*; N % 128 == 0, K % 128 == 0; mt_list must be 4 or 8 (the host
//   keeps classes 1 and 2 on xf_nv4_tiles_m1 / m2).
// ------------------------------------------------------------------------------------------------
template <bool CVT>
__device__ __forceinline__ void xf_nv4_tc_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles,
        const int* __restrict__ offs_row, const int* __restrict__ cnt, int kn, int nsplit_mt) {
    __shared__ __align__(128) uint8_t smA[2][64 * 256];
    __shared__ __align__(128) uint8_t smB[2][8 * 1024];
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int K = kn & 0xFFFF, N = (int)((unsigned)kn >> 16);
    const int nsplit = nsplit_mt & 0xFFFF;
    const int mt_list = max((int)((unsigned)nsplit_mt >> 16), 4);
    const int e = tiles[2 * tile], msup = tiles[2 * tile + 1];
    const int ce = cnt[e];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int n0c = blockIdx.x * 8;               // the CTA's first 16-row tile
    const int n0w = (n0c + warp) * 16;
    const int KB = K >> 7;
    const uint8_t* We = W + (size_t)e * N * (K >> 1);
    const uint8_t* stp = S + (size_t)e * N * (K >> 4) + (size_t)(n0c + warp) * KB * 128 + lane * 4;
    const float gs = (nsplit == N) ? GS[3 * e + 2] : GS[3 * e + (n0w >= nsplit ? 1 : 0)];
    for (int sub = 0; sub < mt_list / 4; ++sub) {
        const int r0 = msup * 16 * mt_list + sub * 64;
        if (r0 >= ce) return;
        const int rows = min(64, ce - r0);
        const int base = offs_row[e] + r0;
        auto load = [&](int st, int b) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = tid + 256 * j, row = c >> 4, ch = c & 15;
                const int rr = base + min(row, rows - 1);
                const int src = rowmap ? rowmap[rr] : rr;
                xf_cp16(&smA[st][row * 256 + ((ch ^ (row & 7)) << 4)], X + (size_t)src * K + b * 128 + ch * 8);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = tid + 256 * j, tl = c >> 6, o = (c & 63) << 4;
                xf_cp16(&smB[st][tl * 1024 + o], We + ((size_t)(n0c + tl) * KB + b) * 1024 + o);
            }
        };
        float acc[4][2][4];
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
        load(0, 0);
        xf_cp_commit();
        uint32_t sc = __ldg(reinterpret_cast<const uint32_t*>(stp));
        for (int b = 0; b < KB; ++b) {
            const int st = b & 1;
            if (b + 1 < KB) load(st ^ 1, b + 1);
            xf_cp_commit();
            const uint32_t nsc = (b + 1 < KB) ? __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)(b + 1) * 128)) : sc;
            xf_cp_wait<1>();
            __syncthreads();
            const uint4 w0 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + lane * 16]);
            const uint4 w1 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + 512 + lane * 16]);
            #pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t b0[8], b1[8];
                xf_e2m1x8<CVT>(s2 ? w0.z : w0.x, &b0[0]); xf_e2m1x8<CVT>(s2 ? w0.w : w0.y, &b0[4]);
                xf_e2m1x8<CVT>(s2 ? w1.z : w1.x, &b1[0]); xf_e2m1x8<CVT>(s2 ? w1.w : w1.y, &b1[4]);
                const uint32_t f0 = xf_e4m3_bcast((sc >> (8 * s2)) & 0xFF), f1 = xf_e4m3_bcast((sc >> (16 + 8 * s2)) & 0xFF);
                #pragma unroll
                for (int j = 0; j < 8; ++j) { b0[j] = xf_hmul2(b0[j], f0); b1[j] = xf_hmul2(b1[j], f1); }
                const int c0 = s2 * 8 + 2 * tig;
                #pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int ra = mt * 16 + gid, rb = ra + 8;
                    const uint4 xa0 = *reinterpret_cast<const uint4*>(&smA[st][ra * 256 + ((c0 ^ (ra & 7)) << 4)]);
                    const uint4 xa1 = *reinterpret_cast<const uint4*>(&smA[st][ra * 256 + (((c0 + 1) ^ (ra & 7)) << 4)]);
                    const uint4 xb0 = *reinterpret_cast<const uint4*>(&smA[st][rb * 256 + ((c0 ^ (rb & 7)) << 4)]);
                    const uint4 xb1 = *reinterpret_cast<const uint4*>(&smA[st][rb * 256 + (((c0 + 1) ^ (rb & 7)) << 4)]);
                    xf_mma(acc[mt][0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
                    xf_mma(acc[mt][0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
                    xf_mma(acc[mt][0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
                    xf_mma(acc[mt][0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
                    xf_mma(acc[mt][1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
                    xf_mma(acc[mt][1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
                    xf_mma(acc[mt][1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
                    xf_mma(acc[mt][1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
                }
            }
            sc = nsc;
            __syncthreads();
        }
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int t0 = 16 * mt + gid, t1 = t0 + 8;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = n0w + 8 * h + 2 * tig;
                if (t0 < rows) *reinterpret_cast<__half2*>(&Y[(size_t)(base + t0) * N + n]) =
                    __floats2half2_rn(acc[mt][h][0] * gs, acc[mt][h][1] * gs);
                if (t1 < rows) *reinterpret_cast<__half2*>(&Y[(size_t)(base + t1) * N + n]) =
                    __floats2half2_rn(acc[mt][h][2] * gs, acc[mt][h][3] * gs);
            }
        }
    }
}
#define XF_NV4_TC_ENTRY(NAME, CVT)                                                                      \
extern "C" __global__ __launch_bounds__(256, 2) void NAME(                                              \
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,     \
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Y,           \
        const int* __restrict__ tiles, const int* __restrict__ ntiles,                                  \
        const int* __restrict__ offs_row, const int* __restrict__ cnt, int kn, int nsplit_mt) {        \
    XF_PDL_ENTRY();                                                                                     \
    xf_nv4_tc_body<CVT>(W, S, GS, X, rowmap, Y, tiles, ntiles, offs_row, cnt, kn, nsplit_mt);          \
}
XF_NV4_TC_ENTRY(xf_nv4_tc, true)
XF_NV4_TC_ENTRY(xf_nv4_tc_i, false)
#undef XF_NV4_TC_ENTRY

// ------------------------------------------------------------------------------------------------
// Dequantisers (prefill reconstruct for cuBLASLt; references for the GPU gates).
// ------------------------------------------------------------------------------------------------
// MXFP8 F64 -> f16 row-major [N][K]: e4m3 -> f16 (exact), then * 2^(s-127) in fp32, one RN to f16 (exact for all but
// 56,179 sub-2^-14 values of this pack — --probe-fp4l FP4L_MX8_F16). One thread = 8 consecutive k of one row (one
// lane's 8 contiguous code bytes in F64). n_elems = N * K of the REAL rows.
__device__ __forceinline__ void xf_mx8_f64_at(long long i8, int K, long long& w_off, long long& s_off) {
    const long long row = i8 / K;
    const int k = (int)(i8 - row * K);
    const int KB = K >> 6;
    const long long t = row >> 4;
    const int rr = (int)(row & 15), h = rr >> 3, gid = rr & 7;
    const int b = k >> 6, gs = (k >> 5) & 1, tig = (k >> 3) & 3;
    w_off = ((t * KB + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8;
    s_off = (t * KB + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs;
}
extern "C" __global__ void xf_mx8_dequant(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                          __half* __restrict__ out, long long n_elems, int K) {
    XF_PDL_ENTRY();
    const long long i8 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i8 >= n_elems) return;
    long long wo, so;
    xf_mx8_f64_at(i8, K, wo, so);
    const float f = xf_ue8m0(S[so]);
    const uint2 c = *reinterpret_cast<const uint2*>(W + wo);
    uint32_t h[4] = {xf_e4m3x2(c.x), xf_e4m3x2(c.x >> 16), xf_e4m3x2(c.y), xf_e4m3x2(c.y >> 16)};
    uint4 o;
    uint32_t* op = reinterpret_cast<uint32_t*>(&o);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __half22float2(*reinterpret_cast<__half2*>(&h[j]));
        __half2 r = __floats2half2_rn(v.x * f, v.y * f);
        op[j] = *reinterpret_cast<uint32_t*>(&r);
    }
    *reinterpret_cast<uint4*>(out + i8) = o;
}
// NVFP4 [N][K/2] (+ scales, gs) -> f16 [N][K]; `gs` applied in fp32 then one RN (reference only).
extern "C" __global__ void xf_nv4_dequant(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float gs,
                                          __half* __restrict__ out, long long n_elems, int K) {
    XF_PDL_ENTRY();
    const long long i8 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i8 >= n_elems) return;
    const long long row = i8 / K, col = i8 - row * K;
    const uint32_t w = *reinterpret_cast<const uint32_t*>(W + (i8 >> 1));
    const uint32_t f = xf_e4m3_bcast(S[row * (K >> 4) + (col >> 4)]);
    uint32_t h[4];
    xf_e2m1x8<true>(w, h);
    uint4 o;
    uint32_t* op = reinterpret_cast<uint32_t*>(&o);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t p = xf_hmul2(h[j], f);   // exact
        const float2 v = __half22float2(*reinterpret_cast<__half2*>(&p));
        __half2 r = __floats2half2_rn(v.x * gs, v.y * gs);
        op[j] = *reinterpret_cast<uint32_t*>(&r);
    }
    *reinterpret_cast<uint4*>(out + i8) = o;
}
// The int-path twin of the e2m1 decode, for --probe-fp4l-gpu's cvt-vs-int bit check: 8 nibbles per
// thread -> 4 f16x2 words per path into two buffers.
extern "C" __global__ void xf_e2m1_selftest(const uint32_t* __restrict__ w, uint32_t* __restrict__ o_cvt,
                                            uint32_t* __restrict__ o_int, int n) {
    XF_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t a[4], b[4];
    xf_e2m1x8<true>(w[i], a);
    xf_e2m1x8<false>(w[i], b);
    #pragma unroll
    for (int j = 0; j < 4; ++j) { o_cvt[4 * i + j] = a[j]; o_int[4 * i + j] = b[j]; }
}

