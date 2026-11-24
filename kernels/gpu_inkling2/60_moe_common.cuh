// gpu_inkling2/60_moe_common.cuh: device helpers of the MoE / LM-head band (60-79). Copied from
// kernels/gpu_fp4l/01_mx8_nv4_gemms_dequant.cuh and 02_precision_levers_router_hc.cuh (never edited there) under the
// `ik2m_` prefix, plus the Inkling-2 expert activation.
//
// Weight layouts (all made on the host at load by src/inkling2/gpu/layout.rs, pure byte permutations):
//   NVFP4 "F128" (fp4l nv4_frag): codes W'[t][b][h][lane][16], scales S'[t][b][lane][4] per 16-row tile t, 128-k block b.
//   MXFP8 "F64"  (fp4l mx8_frag): codes W'[t][b][h][lane][16], scales S'[t][b][tig][8] per 16-row tile t, 64-k block b.
//   BF16  "B32"  (fp4l bf16_frag): W'[t][g][h][lane][8] per 16-row tile t, 32-k group g.
// Inkling-2 w13 (gate|up) ROW ORDER inside a tile: the checkpoint interleaves rows (2i = gate i, 2i+1 = up i); the
// loader places, in 16-row tile t, gate rows 8t..8t+7 at rows 0..7 (fragment half h = 0) and the matching up rows
// at rows 8..15 (h = 1). A warp's two accumulator halves then hold gate AND up of the same 8 intermediate channels,
// and the softcapped SwiGLU runs in the GEMM epilogue in fp32 (no gate|up round trip through memory).

__device__ __forceinline__ uint4 ik2m_ldw4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ik2m_lda4(const __half* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// Two E4M3 bytes (low byte = element 0) -> f16x2. Exact.
__device__ __forceinline__ uint32_t ik2m_e4m3x2(uint32_t two_bytes) {
    uint32_t r;
    const uint16_t v = (uint16_t)two_bytes;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
__device__ __forceinline__ uint32_t ik2m_e4m3_bcast(uint32_t b) { return ik2m_e4m3x2((b & 0xFF) | ((b & 0xFF) << 8)); }

// Eight E2M1 nibbles (low nibble = even k) -> four f16x2, the SM12x cvt (one instruction per byte; exact).
__device__ __forceinline__ void ik2m_e2m1x8(uint32_t w, uint32_t o[4]) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || defined(__CUDA_ARCH_SPECIFIC__)
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
#else
    // exact integer path (PRMT magnitude LUT + sign bits), the fp4l twin's fallback
    const uint32_t LUT_LO = 0x3E3C3800u, LUT_HI = 0x46444240u;
    uint32_t mlo, mhi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mlo) : "r"(LUT_LO), "r"(LUT_HI), "r"(w & 0x7777u));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mhi) : "r"(LUT_LO), "r"(LUT_HI), "r"((w >> 16) & 0x7777u));
    const uint32_t s = w & 0x88888888u;
    uint32_t r0, r1, r2, r3;
    asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(r0) : "r"(mlo));
    asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(r1) : "r"(mlo));
    asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(r2) : "r"(mhi));
    asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(r3) : "r"(mhi));
    o[0] = r0 | ((s << 12) & 0x00008000u) | ((s << 24) & 0x80000000u);
    o[1] = r1 | ((s << 4)  & 0x00008000u) | ((s << 16) & 0x80000000u);
    o[2] = r2 | ((s >> 4)  & 0x00008000u) | ((s << 8)  & 0x80000000u);
    o[3] = r3 | ((s >> 12) & 0x00008000u) | (s & 0x80000000u);
#endif
}

// UE8M0 byte -> 2^(b - 127) as f32 (b = 0 -> 2^-127; 0xFF refused at load).
__device__ __forceinline__ float ik2m_ue8m0(uint32_t b) { return __int_as_float(b ? (int)(b << 23) : (1 << 22)); }

__device__ __forceinline__ void ik2m_mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ik2m_mma_bf(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ik2m_hmul2(uint32_t a, uint32_t b) {
    __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

// Softcapped SwiGLU, fp32, the reference's expression order:
//   softcap(g, 4) * sigmoid(g) * softcap(u, 25),  softcap(x, c) = tanh(x / c) * c.
__device__ __forceinline__ float ik2m_act(float g, float u) {
    const float sg = 1.0f / (1.0f + expf(-g));
    return (tanhf(g / 4.0f) * 4.0f) * sg * (tanhf(u / 25.0f) * 25.0f);
}

// One 64-k step of one NVFP4 F128 16-row tile against token rows (gid, gid+8): the fp4l xf_nv4_step, verbatim.
__device__ __forceinline__ void ik2m_nv4_step(float acc[2][4], const uint32_t n0x, const uint32_t n0y, const uint32_t n1x,
                                              const uint32_t n1y, const uint32_t sb0, const uint32_t sb1,
                                              const uint4 xa0, const uint4 xa1, const uint4 xb0, const uint4 xb1) {
    uint32_t b0[8], b1[8];
    ik2m_e2m1x8(n0x, &b0[0]); ik2m_e2m1x8(n0y, &b0[4]);
    ik2m_e2m1x8(n1x, &b1[0]); ik2m_e2m1x8(n1y, &b1[4]);
    const uint32_t f0 = ik2m_e4m3_bcast(sb0), f1 = ik2m_e4m3_bcast(sb1);
    #pragma unroll
    for (int j = 0; j < 8; ++j) { b0[j] = ik2m_hmul2(b0[j], f0); b1[j] = ik2m_hmul2(b1[j], f1); }
    ik2m_mma(acc[0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
    ik2m_mma(acc[0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
    ik2m_mma(acc[0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
    ik2m_mma(acc[0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
    ik2m_mma(acc[1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
    ik2m_mma(acc[1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
    ik2m_mma(acc[1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
    ik2m_mma(acc[1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
}

// One 64-k block of one MXFP8 F64 16-row tile against token rows (gid, gid+8): the fp4l xf_mx8_block, verbatim
// (each 32-k group's fp32 MMA partial scaled by its exact power of two, per output row).
__device__ __forceinline__ void ik2m_mx8_block(float acc[2][4], const uint4 w0, const uint4 w1, const uint2 sc,
                                               const uint4 xa0, const uint4 xa1, const uint4 xb0, const uint4 xb1) {
    #pragma unroll
    for (int gs = 0; gs < 2; ++gs) {
        const uint4 ax = gs ? xa1 : xa0, bx = gs ? xb1 : xb0;
        const uint32_t c0lo = gs ? w0.z : w0.x, c0hi = gs ? w0.w : w0.y;
        const uint32_t c1lo = gs ? w1.z : w1.x, c1hi = gs ? w1.w : w1.y;
        float p0[4] = {0.f, 0.f, 0.f, 0.f}, p1[4] = {0.f, 0.f, 0.f, 0.f};
        ik2m_mma(p0, ax.x, bx.x, ax.y, bx.y, ik2m_e4m3x2(c0lo), ik2m_e4m3x2(c0lo >> 16));
        ik2m_mma(p0, ax.z, bx.z, ax.w, bx.w, ik2m_e4m3x2(c0hi), ik2m_e4m3x2(c0hi >> 16));
        ik2m_mma(p1, ax.x, bx.x, ax.y, bx.y, ik2m_e4m3x2(c1lo), ik2m_e4m3x2(c1lo >> 16));
        ik2m_mma(p1, ax.z, bx.z, ax.w, bx.w, ik2m_e4m3x2(c1hi), ik2m_e4m3x2(c1hi >> 16));
        const float f00 = ik2m_ue8m0((sc.x >> (8 * gs)) & 0xFF), f01 = ik2m_ue8m0((sc.x >> (8 * (2 + gs))) & 0xFF);
        const float f10 = ik2m_ue8m0((sc.y >> (8 * gs)) & 0xFF), f11 = ik2m_ue8m0((sc.y >> (8 * (2 + gs))) & 0xFF);
        acc[0][0] = fmaf(p0[0], f00, acc[0][0]); acc[0][1] = fmaf(p0[1], f01, acc[0][1]);
        acc[0][2] = fmaf(p0[2], f00, acc[0][2]); acc[0][3] = fmaf(p0[3], f01, acc[0][3]);
        acc[1][0] = fmaf(p1[0], f10, acc[1][0]); acc[1][1] = fmaf(p1[1], f11, acc[1][1]);
        acc[1][2] = fmaf(p1[2], f10, acc[1][2]); acc[1][3] = fmaf(p1[3], f11, acc[1][3]);
    }
}

// The shared expert epilogue (decode and prefill, NVFP4 and MXFP8: one code path, so a row's bits do not depend on
// which kernel produced it). acc[h][r]: h = 0/1 the tile's 8-row halves; r = (token a, col 2tig), (a, 2tig+1),
// (token b, 2tig), (b, 2tig+1). gs: the expert's fp32 multiplier (NVFP4 amax/denominator; 1 for MXFP8).
//   GU: half 0 = gate, half 1 = up of intermediate channels i = 8*tile16 + 2tig (+1): act -> Din[row][NI] f16.
//   DN: the fp32 rows y = acc * gs -> Yf[row][N] (n = 16*tile16 + 8h + 2tig).
template <bool GU>
__device__ __forceinline__ void ik2m_expert_store(const float acc[2][4], float gs, int tile16, int tig,
                                                  int row_a, int row_b, bool va, bool vb, int N,
                                                  __half* __restrict__ Din, float* __restrict__ Yf) {
    if constexpr (GU) {
        const int NI = N >> 1, i = 8 * tile16 + 2 * tig;
        if (va) *reinterpret_cast<__half2*>(&Din[(size_t)row_a * NI + i]) =
            __floats2half2_rn(ik2m_act(acc[0][0] * gs, acc[1][0] * gs), ik2m_act(acc[0][1] * gs, acc[1][1] * gs));
        if (vb) *reinterpret_cast<__half2*>(&Din[(size_t)row_b * NI + i]) =
            __floats2half2_rn(ik2m_act(acc[0][2] * gs, acc[1][2] * gs), ik2m_act(acc[0][3] * gs, acc[1][3] * gs));
    } else {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = 16 * tile16 + 8 * h + 2 * tig;
            if (va) *reinterpret_cast<float2*>(&Yf[(size_t)row_a * N + n]) = make_float2(acc[h][0] * gs, acc[h][1] * gs);
            if (vb) *reinterpret_cast<float2*>(&Yf[(size_t)row_b * N + n]) = make_float2(acc[h][2] * gs, acc[h][3] * gs);
        }
    }
}
