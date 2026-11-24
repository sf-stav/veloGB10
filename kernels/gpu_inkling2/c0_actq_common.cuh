// gpu_inkling2/c0_actq_common.cuh — leg Q1: the pack-declared expert ACTIVATION quantisation (opt-in,
// --inkling2-expert-act native; default a16 = the K2 W4A16 / W8A16 kernels, untouched).
//
// The device quantisers below are statement-for-statement copies of the external reference's helpers (public
// repository SGLang, `jit/csrc/inkling2/routed_quant.cuh`, Apache-2.0): e4m3_round, e2m1_pack, quantize_part
// ("four over six", block 16), part_max, block_max_reduce, mxfp8_half_block (block 32, ue8m0) and capped_swiglu
// (ex2.approx / tanh.approx, as the reference's epilogue). Identical statements = identical bits; the CPU model in
// src/inkling2/actq/cpu.rs and the reference's own outputs (src/inkling2/actq/fixtures) gate it.
//
// Activation formats written by c1 (plain row-major, no CUTLASS swizzle):
//   NVFP4: codes [row][K/2] (low nibble = even k), e4m3 block scales [row][K/16], amax [row] f32 (outer = amax/1536).
//   MXFP8: e4m3 codes [row][K], ue8m0 exponents [row][K/32].
// Native weight layouts (src/inkling2/actq/layout.rs, byte permutations of the checkpoint):
//   NVFP4 "NA": per 16-row tile T (w13: rows 0-7 gate, 8-15 up, the K2 tile interleave), per 64-k step s:
//     codes [T][s][lane][16 B] = the m16n8k64 A fragment {a0: row g k 8t.., a1: row g+8, a2: row g k 32+8t.., a3: row g+8}
//     scales [T][s][16 rows][4 B] (the 4 e4m3 block scales of the step; SFA = lane (g, t<2) -> row g + 8t).
//   MXFP8 "MA": per tile T, per 32-k step s: codes [T][s][lane][16 B] = the m16n8k32 A fragment {a0: row g k 4t..,
//     a1: row g+8, a2: row g k 16+4t.., a3: row g+8}; exponents [T][s/4][16 rows][4 B] (byte = step s % 4).

#define INK2Q_FLT_MAX 3.4028234663852886e38f

__device__ __forceinline__ float ink2q_e4m3_round(float v, uint8_t& bits) {
    uint16_t pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
    bits = pair & 0xFF;
    uint32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(pair));
    return __half2float(__ushort_as_half(halves & 0xFFFF));
}

__device__ __forceinline__ uint32_t ink2q_e2m1_pack(float low, float high, float& rlow, float& rhigh) {
    uint32_t packed, halves;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; "
        "cvt.u32.u8 %0, b; cvt.rn.f16x2.e2m1x2 %1, b; }"
        : "=r"(packed), "=r"(halves)
        : "f"(low), "f"(high));
    rlow = __half2float(__ushort_as_half(halves & 0xFFFF));
    rhigh = __half2float(__ushort_as_half(halves >> 16));
    return packed;
}

// One 16-value block split over 16 / E consecutive lanes (fouroversix_mae).
template <int E>
__device__ __forceinline__ void ink2q_quantize_part(float const (&x)[E], float amax, float block_max, uint32_t& packed,
                                                    uint8_t& scale) {
    float encoder = fminf(__fdiv_rn(1536.0f, amax), INK2Q_FLT_MAX);
    float decoder = __fdiv_rn(1.0f, encoder);
    float outer = __fmul_rn(amax, float(1.0 / 1536.0));
    uint8_t bits6, bits4;
    float scale6 = ink2q_e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, float(1.0 / 6.0))), 448.0f), bits6);
    float scale4 = ink2q_e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, 0.25f)), 448.0f), bits4);
    float inverse6 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale6, decoder)), INK2Q_FLT_MAX);
    float inverse4 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale4, decoder)), INK2Q_FLT_MAX);
    uint32_t p6 = 0, p4 = 0;
    float e6[E], e4[E];
#pragma unroll
    for (int i = 0; i < E / 2; ++i) {
        float r0, r1;
        p6 |= ink2q_e2m1_pack(__fmul_rn(x[2 * i], inverse6), __fmul_rn(x[2 * i + 1], inverse6), r0, r1) << (8 * i);
        e6[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale6), outer), x[2 * i]));
        e6[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale6), outer), x[2 * i + 1]));
        p4 |= ink2q_e2m1_pack(__fmul_rn(x[2 * i], inverse4), __fmul_rn(x[2 * i + 1], inverse4), r0, r1) << (8 * i);
        e4[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale4), outer), x[2 * i]));
        e4[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale4), outer), x[2 * i + 1]));
    }
#pragma unroll
    for (int width = E / 2; width >= 1; width /= 2) {
#pragma unroll
        for (int i = 0; i < width; ++i) {
            e6[i] = __fadd_rn(e6[2 * i], e6[2 * i + 1]);
            e4[i] = __fadd_rn(e4[2 * i], e4[2 * i + 1]);
        }
    }
#pragma unroll
    for (int s = 1; s < 16 / E; s *= 2) {
        e6[0] = __fadd_rn(e6[0], __shfl_xor_sync(0xFFFFFFFF, e6[0], s));
        e4[0] = __fadd_rn(e4[0], __shfl_xor_sync(0xFFFFFFFF, e4[0], s));
    }
    bool choose4 = e4[0] < e6[0];
    packed = choose4 ? p4 : p6;
    scale = choose4 ? bits4 : bits6;
}

template <int E>
__device__ __forceinline__ float ink2q_part_max(float const (&x)[E]) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) m = fmaxf(m, fabsf(x[i]));
#pragma unroll
    for (int s = 1; s < 16 / E; s *= 2) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, s));
    return m;
}

template <int kThreads>
__device__ __forceinline__ float ink2q_block_max_reduce(float v, float* red) {
#pragma unroll
    for (int s = 16; s >= 1; s /= 2) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, s));
    constexpr int kWarps = kThreads / 32;
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    v = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) v = fmaxf(v, red[w]);
    return v;
}

__device__ __forceinline__ float ink2q_capped_swiglu(float gate, float up, float gamma, float gate_cap, float up_cap) {
    float sigmoid;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(sigmoid)
        : "f"(gate));
    if (gate_cap > 0) {
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(__fdiv_rn(gate, gate_cap)));
        gate = __fmul_rn(gate_cap, t);
    }
    if (up_cap > 0) {
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(__fdiv_rn(up, up_cap)));
        up = __fmul_rn(up_cap, t);
    }
    return __fmul_rn(__fmul_rn(__fmul_rn(gate, sigmoid), up), gamma);
}

// One 32-value MXFP8 block split over a lane pair (16 values each).
__device__ __forceinline__ void ink2q_mxfp8_half_block(float const (&x)[16], uint4& values, uint8_t& exponent_out) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1));
    float const min_scale = __int_as_float(0x00400000);
    float raw = __fmul_rn(amax, float(1.0 / 448.0));
    int exponent = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
    exponent = raw <= min_scale ? 0 : min(exponent, 254);
    if (raw != raw) exponent = 255;
    float scale = fmaxf(__int_as_float(exponent << 23), min_scale);
    float inverse = exponent == 0 ? __int_as_float(0x7F000000) : __frcp_rn(scale);
    if (exponent == 255) inverse = 0.0f;
    uint16_t pairs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float lo = __fmul_rn(x[2 * i], inverse), hi = __fmul_rn(x[2 * i + 1], inverse);
        lo = fminf(fmaxf(lo, -448.0f), 448.0f);
        hi = fminf(fmaxf(hi, -448.0f), 448.0f);
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pairs[i]) : "f"(hi), "f"(lo));
    }
    values = *reinterpret_cast<uint4*>(pairs);
    exponent_out = uint8_t(exponent);
}

__device__ __forceinline__ float ink2q_bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// NVFP4 lanes per 16-value block (one CTA per row): split 4 for K <= 4096.
__host__ __device__ constexpr int ink2q_split(int width) {
    return width / 16 * 4 <= 1024 ? 4 : width / 16 * 2 <= 1024 ? 2 : 1;
}
__host__ __device__ constexpr int ink2q_split_threads(int width) { return (width / 16 * ink2q_split(width) + 31) / 32 * 32; }
__host__ __device__ constexpr int ink2q_mx_threads(int width) { return (width / 16 + 31) / 32 * 32; }

// ---- block-scaled tensor-core MMAs (sm_121a) ----
// m16n8k64 e2m1 x e2m1, ue4m3 scales, 4 per row (scale_vec::4X), selectors {byte 0, thread 0}; C accumulates.
__device__ __forceinline__ void ink2q_mma_nv4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}
// m16n8k32 e4m3 x e4m3, ue8m0 scales, one per row (scale_vec::1X), byte BID of the scale registers; C accumulates.
template <int BID>
__device__ __forceinline__ void ink2q_mma_mx8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,0}, %11, {%12,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(BID));
}
__device__ __forceinline__ void ink2q_mma_mx8_v(int v, float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    switch (v) {
    case 0: ink2q_mma_mx8<0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    case 1: ink2q_mma_mx8<1>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    case 2: ink2q_mma_mx8<2>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    default: ink2q_mma_mx8<3>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    }
}

__device__ __forceinline__ uint4 ink2q_ldw4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
