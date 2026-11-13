// gpu_fp4l.cu — FP4L weight kernels: nvidia/Qwen3.8-Flash-Next-FP4-Local served on the EXL3 trunk.
//
// Module `gpu_fp4l` (sm_121a: the E2M1x2 -> F16x2 cvt is an SM12x family-specific form; every
// other instruction here is baseline sm_121). Loaded only when an FP4-Local pack is opened
// (exl3_forward/fp4l_fwd.rs). Activations are the EXL3 trunk's f16; accumulation is fp32.
// Design + contracts: PLAN/FP4L_DESIGN.md §4. Every entry opens with XF_PDL_ENTRY (the EXL3
// module's PDL contract) so a PDL-launched neighbour may overlap it.
//
// Weight formats (all row-major [N output rows][K]; K contiguous — the checkpoint layout, no repack):
//   MXFP8  : E4M3 codes + UE8M0 scales per 32, w = e4m3(c) * 2^(s-127), stored on the device in the FRAGMENT-NATIVE
//            layout "F64" (a pure byte permutation of the checkpoint's [N][K] / [N][K/32] arrays made at load,
//            fp4l_fwd::mx8_frag; N padded to a multiple of 16 with zero rows): codes W'[t][b][h][lane][16]
//            (t = N/16 tiles, b = K/64 blocks, h = 0/1 the two 8-row halves, lane = gid*4 + tig) hold row
//            16t+8h+gid, k = 64b + 32gs + 8tig + j at byte gs*8 + j; scales S'[t][b][tig][8] hold, at byte
//            h*4 + r*2 + gs, the scale of row 16t+8h+2tig+r for group 2b+gs. A warp's 16-B load is then 512
//            contiguous bytes that ARE its MMA fragments (the row-major layout's 8-B loads touched 8 rows per
//            instruction and streamed at ~130 GB/s; F64 streams at the roofline, microbench 2026-10-29).
//   NVFP4  : E2M1 nibbles [N][K/2] (low nibble = even k) + E4M3 scales [N][K/16], one fp32 global
//            multiplier per (expert, projection): w = e2m1(q) * e4m3(s) * gs
//   F16    : [N][K] f16 = f16(w * 2^k), one power-of-two shift per tensor (--fp4l-lm-head f16 / --fp4l-plain f16)
//   BF16   : [N][K] the checkpoint's bf16 bits AS STORED (--fp4l-lm-head bf16 / --fp4l-plain exact). The f16
//            activation x is split into two bf16 halves x = hi + lo (hi = bf16 RN of x, lo = x - hi: both exact in
//            bf16, since x has 11 significant bits), so W*x = W*hi + W*lo with every product exact in fp32
//            (8 x 8 significant bits): two bf16 MMAs per f16 one, the same weight bytes.
//
// Exactness. e4m3 codes and e2m1*e4m3 products are exact in f16 (|v| <= 2688, >= 2^-10). The
// MXFP8 power-of-two group scale is NOT folded into f16 (56,179 values of this pack would round):
// each 32-k group is reduced by its own pair of MMAs into a zeroed fp32 tile which is then
// multiplied by 2^(s-127) (exact) and added to the accumulator. The NVFP4 global multiplier is
// applied once to the fp32 accumulator in the epilogue.
//
// The k-permutation. Inside one m16n8k16 MMA each lane owns the k slots {2t, 2t+1, 2t+8, 2t+9}
// (t = lane & 3) in BOTH the A and the B fragment. A dot product does not care which actual k sits
// in a slot as long as A and B agree, so lane t is given CONTIGUOUS actual k (8 per 32-k group for
// MXFP8 / f16; 16 = one NVFP4 block per 64-k step for NVFP4): one wide load per row instead of four
// scattered ones, and for NVFP4 the lane's 16 values share ONE block scale. The mapping is fixed,
// so every output's reduction order is the same at every M (batch invariance, AGENTS §2.4).
//
// Batch invariance. M (tokens) is the MMA M dimension; each output row depends only on its own A
// row and the fixed k order; rows >= M read a clamped valid row and are never stored. The split-K
// factor ks is a pure function of (K, N) (host: fp4l_fwd::mx8_ks); slabs are summed in ascending
// order (the WP21 fixup contract: the last-arriving CTA of an N-tile sums slabs 0..ks-1 and re-arms
// the counter). No atomics touch data.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <stdint.h>

#ifdef KERNEL_BUILD_ID
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }
#else
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = 0; }
#endif

#define XF_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

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

// ================================================================================================
// PRECISION LEVERS (--fp4l-plain exact, --fp4l-norms f32, --fp4l-lm-head bf16; PLAN/FP4L_DESIGN.md §3a).
// The pack's own weight types served AS STORED: bf16 bits (uint16_t) and fp32 norms.
//
// * xf_bf16w_gemm / xf_bf16_wide: stored-bf16 GEMMs (decode / prefill) with the activation split
//   x = hi + lo (two exact bf16 halves of each f16 value): two bf16 MMAs per f16 one, every product
//   exact in fp32. The xf_f16w_gemm / xf_mx8_wide contracts (grid, k-permutation, split-K fixup,
//   batch invariance); K % 32 == 0.
// * The xf_*_bf / xf_*_f TWINS are line-for-line copies of the named trunk kernels (kernels/exl3_bench.cu)
//   with ONLY the weight load changed (bf16 -> f32 is a 16-bit shift; the f16 activation loads, the
//   expression order and therefore the fma contraction points are the trunk kernel's). On a weight that is
//   exactly representable in f16 a twin returns the trunk kernel's bits: the --probe-fp4l-gpu TWIN lines
//   check exactly that, on real tensors.
// * xf_sg_gate: the shared-expert gate logit from the stored bf16 gate, handed to the trunk's UNCHANGED
//   combine kernels (all six variants, TP included) bit-exactly — see the kernel's comment.
// ================================================================================================

__device__ __forceinline__ float xf_h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half xf_f2h(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float xf_silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float xf_sig(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float xf_b2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

// f16x2 -> (hi, lo) bf16x2 with hi + lo == each f16 value exactly: hi = bf16 RN of x (8 of x's <= 11
// significant bits), lo = x - hi (exact in fp32; <= 3 significant bits, so exact in bf16 too).
__device__ __forceinline__ void xf_split2(uint32_t h2, uint32_t& hi, uint32_t& lo) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    const __nv_bfloat162 bh = __floats2bfloat162_rn(f.x, f.y);
    const float2 g = __bfloat1622float2(bh);
    const __nv_bfloat162 bl = __floats2bfloat162_rn(f.x - g.x, f.y - g.y);
    hi = *reinterpret_cast<const uint32_t*>(&bh);
    lo = *reinterpret_cast<const uint32_t*>(&bl);
}
__device__ __forceinline__ void xf_split4(const uint4 x, uint4& hi, uint4& lo) {
    xf_split2(x.x, hi.x, lo.x); xf_split2(x.y, hi.y, lo.y); xf_split2(x.z, hi.z, lo.z); xf_split2(x.w, hi.w, lo.w);
}
__device__ __forceinline__ void xf_mma_bf(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 32-k group of one 16-row weight tile (w: the lane's 8 contiguous k of its B row, the
// xf_f16w_gemm k-permutation) against token rows gid (xah / xal) and gid + 8 (xbh / xbl):
// hi then lo for actual k 8t..8t+3, hi then lo for 8t+4..8t+7. A fixed sequence per output.
__device__ __forceinline__ void xf_bf_group(float acc[4], const uint4 xah, const uint4 xal, const uint4 xbh,
                                            const uint4 xbl, const uint4 w) {
    xf_mma_bf(acc, xah.x, xbh.x, xah.y, xbh.y, w.x, w.y);
    xf_mma_bf(acc, xal.x, xbl.x, xal.y, xbl.y, w.x, w.y);
    xf_mma_bf(acc, xah.z, xbh.z, xah.w, xbh.w, w.z, w.w);
    xf_mma_bf(acc, xal.z, xbl.z, xal.w, xbl.w, w.z, w.w);
}

// ------------------------------------------------------------------------------------------------
// Stored-bf16 GEMM weight layout "B32" (fragment-native; fp4l_fwd::bf16_frag, a pure u16 permutation of the
// checkpoint's [N][K] bf16 made at load, N padded to a multiple of 16 with zero rows): W'[t][g][h][lane][8]
// (t = N/16 tiles, g = K/32 groups, h = 0/1 the 8-row halves, lane = gid*4 + tig) holds row 16t+8h+gid,
// k = 32g + 8tig + j at u16 j. One 32-k group of a tile = 1 KB contiguous; a warp's 16-B load is 512 contiguous
// bytes that ARE its B fragments (the row-major loads touched 8 rows per instruction).
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// xf_bf16w_gemm — stored bf16 weights (B32), M <= 16 (decode / verify): the lm_head and the MTP
// linears (FxW::Bf16), the router (fp32 logits via Yf + fixup), the n-gram key / value projections,
// the hc mixer fallbacks. Grid (ceil(N/128), ks), block 256 (8 warps x one 16-row tile); xf_f16w_gemm's
// epilogue. K % 32 == 0, (K/ks) % 32 == 0. A 128-k main loop (4 groups' loads in flight) and a 32-k tail.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void xf_bf16w_gemm(
        const uint16_t* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, float* __restrict__ ws, unsigned* __restrict__ cnt,
        int M, int K, int N, int ks) {
    XF_PDL_ENTRY();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
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
        // B32: group g of this tile at wt + g*512 (half 0) / + 256 (half 1); k / 32 * 512 = k * 16
        const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
        int k = kb;
        for (; k + 128 <= ke; k += 128) {
            uint4 b0[4], b1[4], xa4[4], xb4[4];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                b0[g] = xf_ldw4(wt + (size_t)(k + 32 * g) * 16);
                b1[g] = xf_ldw4(wt + (size_t)(k + 32 * g) * 16 + 256);
                xa4[g] = xf_lda4(xa + k + 32 * g);
                xb4[g] = xf_lda4(xb + k + 32 * g);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint4 ah, al, bh, bl;
                xf_split4(xa4[g], ah, al);
                xf_split4(xb4[g], bh, bl);
                xf_bf_group(acc[0], ah, al, bh, bl, b0[g]);
                xf_bf_group(acc[1], ah, al, bh, bl, b1[g]);
            }
        }
        for (; k < ke; k += 32) {
            const uint4 b0 = xf_ldw4(wt + (size_t)k * 16), b1 = xf_ldw4(wt + (size_t)k * 16 + 256);
            uint4 ah, al, bh, bl;
            xf_split4(xf_lda4(xa + k), ah, al);
            xf_split4(xf_lda4(xb + k), bh, bl);
            xf_bf_group(acc[0], ah, al, bh, bl, b0);
            xf_bf_group(acc[1], ah, al, bh, bl, b1);
        }
    }
    xf_dense_epilogue(acc, M, N, n0w, live, ks, ws, cnt, Y, Yf, 1.0f);
}

// ------------------------------------------------------------------------------------------------
// xf_bf16_wide — stored bf16 weights (B32) at PREFILL row counts (any M >= 1): xf_mx8_wide's shape
// (grid (ceil(N/128), ceil(M/64)), block 256, MT = 4 token tiles per CTA reusing the B fragments).
// X [M][K] f16 -> Y [M][N] f16 (or Yf fp32 when Yf != nullptr: the prefill router logits). N even,
// K % 32 == 0. Each output row depends only on its own X row (rows are independent MMA rows): a row
// block computed alone (TP sequence-parallel prefill) has the full chunk's bits.
// ------------------------------------------------------------------------------------------------
template <int MT>
__device__ __forceinline__ void xf_bf16_wide_body(const uint16_t* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    if (n0w >= N) return;
    const int t0 = blockIdx.y * 16 * MT;
    const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
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
    for (int k = 0; k < K; k += 32) {
        const uint4 c0 = xf_ldw4(wt + (size_t)k * 16), c1 = xf_ldw4(wt + (size_t)k * 16 + 256);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint4 ah, al, bh, bl;
            xf_split4(xf_lda4(xa[mt] + k), ah, al);
            xf_split4(xf_lda4(xb[mt] + k), bh, bl);
            xf_bf_group(acc[mt][0], ah, al, bh, bl, c0);
            xf_bf_group(acc[mt][1], ah, al, bh, bl, c1);
        }
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
extern "C" __global__ __launch_bounds__(256) void xf_bf16_wide(const uint16_t* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    XF_PDL_ENTRY();
    xf_bf16_wide_body<4>(W, X, Y, Yf, M, K, N);
}

// ------------------------------------------------------------------------------------------------
// Hyper-connection twins.
// ------------------------------------------------------------------------------------------------

// == xq_hc_norm, fp32 norm weights. grid hc*B, block 1024, dynamic smem 4096.
extern "C" __global__ void xf_hc_norm_f(__half* __restrict__ out, const float* __restrict__ x,
                                        const float* __restrict__ w, int h, int hc, int B, float eps) {
    XF_PDL_ENTRY();
    int blk = blockIdx.x;
    if (blk >= hc * B) return;
    int b = blk / hc;
    int s = blk % hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    const float* xs = x + (long long)b * h * hc + (long long)s * h;
    float sum_sq = 0.0f;
    for (int i = tid; i < h; i += bs) { float v = xs[i]; sum_sq += v * v; }
    sm[tid] = sum_sq;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(sm[0] / (float)h + eps);
    for (int i = tid; i < h; i += bs)
        out[(long long)blk * h + i] = xf_f2h(xs[i] * inv * (1.0f + w[(long long)s * h + i]));
}

// == xq_hc_inj_norm, fp32 norm weights. grid hc*B, block 1024 (the tree leaves); h <= 4096.
extern "C" __global__ void __launch_bounds__(1024)
xf_hc_inj_norm_f(__half* __restrict__ out, float* __restrict__ x, const __half* __restrict__ y,
                 const float* __restrict__ inj, const float* __restrict__ w, int h, int hc, int B, float eps) {
    XF_PDL_ENTRY();
    const int blk = blockIdx.x;
    if (blk >= hc * B) return;
    const int b = blk / hc;
    const int s = blk % hc;
    __shared__ float sm[1024];
    const int tid = threadIdx.x, bs = blockDim.x;
    float* xs = x + (long long)b * h * hc + (long long)s * h;
    const __half* yr = y + (long long)b * h;
    const float g = inj[b * hc + s];
    constexpr int NV = 4;
    float v[NV];
    float sum_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int i = tid + j * bs;
        v[j] = 0.0f;
        if (i < h) {
            float t = xs[i];
            t += g * xf_h2f(yr[i]);
            xs[i] = t;
            v[j] = t;
            sum_sq += t * t;
        }
    }
    sm[tid] = sum_sq;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    const float inv = rsqrtf(sm[0] / (float)h + eps);
    #pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int i = tid + j * bs;
        if (i < h) out[(long long)blk * h + i] = xf_f2h(v[j] * inv * (1.0f + w[(long long)s * h + i]));
    }
}

// The A5-P2 prefill helpers (copies of xq_pf_unpack4 / xq_pf_pack4 / xq_pf_tree512).
__device__ __forceinline__ void xf_pf_unpack4(const uint2 v, float f[4]) {
    const __half2 a = *reinterpret_cast<const __half2*>(&v.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&v.y);
    f[0] = xf_h2f(__low2half(a)); f[1] = xf_h2f(__high2half(a));
    f[2] = xf_h2f(__low2half(b)); f[3] = xf_h2f(__high2half(b));
}
__device__ __forceinline__ void xf_pf_unpack4_bf(const uint2 v, float f[4]) {
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xFFFF0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xFFFF0000u);
}
__device__ __forceinline__ uint2 xf_pf_pack4(const float f[4]) {
    __half2 a = __halves2half2(xf_f2h(f[0]), xf_f2h(f[1]));
    __half2 b = __halves2half2(xf_f2h(f[2]), xf_f2h(f[3]));
    uint2 r;
    r.x = *reinterpret_cast<const unsigned int*>(&a);
    r.y = *reinterpret_cast<const unsigned int*>(&b);
    return r;
}
__device__ __forceinline__ float xf_pf_tree512(const float* sm, int lane) {
    float a[8];
    #pragma unroll
    for (int q = 0; q < 8; ++q) a[q] = sm[lane + 32 * q] + sm[lane + 32 * q + 256];   // level 256
    #pragma unroll
    for (int q = 0; q < 4; ++q) a[q] += a[q + 4];                                     // level 128
    #pragma unroll
    for (int q = 0; q < 2; ++q) a[q] += a[q + 2];                                     // level 64
    a[0] += a[1];                                                                     // level 32
    float v = a[0];
    #pragma unroll
    for (int s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);   // 16..1
    return v;
}

// == xq_pf_hc_inj_norm (prefill fused inject + norm, also the TP prefill-overlap hook's kernel), fp32
// norm weights. grid hc*B, block 128; h % 512 == 0, h <= 4096.
extern "C" __global__ void __launch_bounds__(128)
xf_pf_hc_inj_norm_f(__half* __restrict__ out, float* __restrict__ x, const __half* __restrict__ y,
                    const float* __restrict__ inj, const float* __restrict__ w, int h, int hc, int B, float eps) {
    XF_PDL_ENTRY();
    const int blk = blockIdx.x;
    if (blk >= hc * B) return;
    const int b = blk / hc, s = blk - b * hc;
    __shared__ float sm[512];
    __shared__ float s_inv;
    const int T = threadIdx.x;
    float* xs = x + (long long)blk * h;
    const bool doinj = (y != nullptr);
    const __half* yr = doinj ? y + (long long)b * h : y;
    const float g = doinj ? inj[b * hc + s] : 0.0f;
    const int nk = h >> 9;
    constexpr int KM = 8;
    float v[KM][4];
    float pa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, pb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float4 t4 = *reinterpret_cast<const float4*>(xs + i);
            float t[4] = {t4.x, t4.y, t4.z, t4.w};
            if (doinj) {
                float yf[4];
                xf_pf_unpack4(*reinterpret_cast<const uint2*>(yr + i), yf);
                #pragma unroll
                for (int e = 0; e < 4; ++e) t[e] += g * yf[e];
                *reinterpret_cast<float4*>(xs + i) = make_float4(t[0], t[1], t[2], t[3]);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[k][e] = t[e];
                if (k & 1) pb[e] += t[e] * t[e]; else pa[e] += t[e] * t[e];
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) sm[4 * T + e] = pa[e] + pb[e];
    __syncthreads();
    if (T < 32) {
        const float r = xf_pf_tree512(sm, T);
        if (T == 0) s_inv = rsqrtf(r / (float)h + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float* ws = w + (long long)s * h;
    __half* os = out + (long long)blk * h;
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            const float4 w4 = *reinterpret_cast<const float4*>(ws + i);
            const float wf[4] = {w4.x, w4.y, w4.z, w4.w};
            float o[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = v[k][e] * inv * (1.0f + wf[e]);
            *reinterpret_cast<uint2*>(os + i) = xf_pf_pack4(o);
        }
    }
}

// == xq_grid_barrier (self-resetting one-shot device-scope barrier; the whole grid must be co-resident).
__device__ __forceinline__ void xf_grid_barrier(cuda::atomic<int, cuda::thread_scope_device>* bar, int goal) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = bar->fetch_add(1, cuda::memory_order_release);
        if (prev == goal - 1) {
            bar->store(0, cuda::memory_order_release);
        } else {
            while (bar->load(cuda::memory_order_acquire) != 0) __nanosleep(64);
        }
    }
    __syncthreads();
}

// A grid barrier that can be used several times in one launch: bar[0] = arrival count, bar[1] = generation (only
// ever incremented). The last arriver resets the count, then publishes the next generation; waiters wait for the
// generation to move (a CTA leaving early and arriving at the NEXT barrier cannot be mistaken for this one's reset).
__device__ __forceinline__ void xf_grid_barrier_gen(cuda::atomic<int, cuda::thread_scope_device>* bar, int goal) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int gen = bar[1].load(cuda::memory_order_acquire);
        const int prev = bar[0].fetch_add(1, cuda::memory_order_acq_rel);
        if (prev == goal - 1) {
            bar[0].store(0, cuda::memory_order_relaxed);
            bar[1].fetch_add(1, cuda::memory_order_release);
        } else {
            while (bar[1].load(cuda::memory_order_acquire) == gen) __nanosleep(64);
        }
    }
    __syncthreads();
}

// == xq_dot8 with bf16 weights: 8 products, ascending fp32 accumulation (element j = word j/2, low
// half first). Word-level unpack (shift / mask) keeps xq_hc_fuse's 40-register budget.
__device__ __forceinline__ float xf_dot8_bf(const uint16_t* w, const __half* x) {
    const uint4 wv = *(const uint4*)w;
    float4 xv = *(const float4*)x;
    const __half* xh_ = (const __half*)&xv;
    float acc = 0.0f;
    acc += __uint_as_float(wv.x << 16) * __half2float(xh_[0]);
    acc += __uint_as_float(wv.x & 0xFFFF0000u) * __half2float(xh_[1]);
    acc += __uint_as_float(wv.y << 16) * __half2float(xh_[2]);
    acc += __uint_as_float(wv.y & 0xFFFF0000u) * __half2float(xh_[3]);
    acc += __uint_as_float(wv.z << 16) * __half2float(xh_[4]);
    acc += __uint_as_float(wv.z & 0xFFFF0000u) * __half2float(xh_[5]);
    acc += __uint_as_float(wv.w << 16) * __half2float(xh_[6]);
    acc += __uint_as_float(wv.w & 0xFFFF0000u) * __half2float(xh_[7]);
    return acc;
}

// == xq_hc_fuse with the stored bf16 mixers: dd = down(hn); dd = silu(dd/hcn); uu = up(dd). ONE launch.
// grid (bx, m), block 256; the launcher sizes bx from THIS kernel's measured co-residency.
extern "C" __global__ void __launch_bounds__(256, 6)
xf_hc_fuse_bf(__half* __restrict__ dd, __half* __restrict__ uu,
              const __half* __restrict__ hn, const uint16_t* __restrict__ w_down,
              const uint16_t* __restrict__ w_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
              float hcn, int lr, int rw) {
    XF_PDL_ENTRY();
    __shared__ __half sb[512];
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __half* hnr = hn + (size_t)row * rw;
    const int s8 = rw >> 8;
    for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
        const uint16_t* wr = w_down + (size_t)n * rw;
        float acc = 0.0f;
        // unroll 2 (the trunk's 4 spills 12 B here: the bf16 unpack needs more registers than the f16
        // cvt under the (256, 6) bound). Unrolling never changes the acc chain (bitwise).
        #pragma unroll 2
        for (int s = 0; s < s8; ++s) acc += xf_dot8_bf(wr + (s << 8) + (lane << 3), hnr + (s << 8) + (lane << 3));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) dd[(size_t)row * lr + n] = __float2half_rn(acc);
    }
    xf_grid_barrier(bar, gridDim.x * gridDim.y);
    for (int t = tid; t < lr; t += 256)
        sb[t] = __float2half_rn(xf_silu(__half2float(dd[(size_t)row * lr + t]) / hcn));
    __syncthreads();
    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t8 = lr >> 3;
    for (int n = gt; n < rw; n += tt) {
        const uint16_t* wr = w_up + (size_t)n * lr;
        float acc = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < t8; ++s) acc += xf_dot8_bf(wr + (s << 3), sb + (s << 3));
        uu[(size_t)row * rw + n] = __float2half_rn(acc);
    }
}

// ------------------------------------------------------------------------------------------------
// xf_bf16_tc — stored bf16 (B32) x f16 at PREFILL row counts (m > 16), smem-tiled: the served replacement of
// xf_bf16_wide, bit-identical to it (the same xf_bf_group sequence per 32-k group, groups ascending).
// CTA tile 128 tokens x 128 weight rows, 32-k steps, 3 smem stages (48 KB static), 2 CTAs / SM. 8 warps =
// 2 (tokens) x 4 (rows); warp tile 64 tokens (4 m16 tiles) x 32 rows (2 B32 tiles). 1-D grouped grid (XF_TC_GM).
// Activation smem rows are 64 B (4 chunks): a quarter-warp's 16-B reads (2 rows x 4 chunks) are conflict-free.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 2) void xf_bf16_tc(const uint16_t* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ Y, float* __restrict__ Yf, int M, int K, int N) {
    XF_PDL_ENTRY();
    constexpr int NS = 3;
    __shared__ __align__(128) uint8_t smA[NS][128 * 64];
    __shared__ __align__(128) uint8_t smB[NS][8 * 1024];
    const int nb = (N + 127) >> 7, mb = (M + 127) >> 7;
    const int id = blockIdx.x;
    const int per_group = XF_TC_GM * nb;
    const int g = id / per_group, first_m = g * XF_TC_GM;
    const int gm = min(mb - first_m, XF_TC_GM);
    const int m_blk = first_m + (id % per_group) % gm;
    const int n_blk = (id % per_group) / gm;
    const int m0 = m_blk * 128, n0 = n_blk * 128;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int G = K >> 5;
    const int ntiles = (N + 15) >> 4;
    auto load_stage = [&](int st, int gk) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {   // A: 128 rows x 4 chunks = 512 chunks
            const int c = tid + 256 * j, row = c >> 2, ch = c & 3;
            const int tok = min(m0 + row, M - 1);
            xf_cp16(&smA[st][row * 64 + ch * 16], X + (size_t)tok * K + gk * 32 + ch * 8);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {   // B: 8 tiles x 64 chunks = 512 chunks
            const int c = tid + 256 * j, tl = c >> 6, o = (c & 63) << 4;
            const int t = min((n0 >> 4) + tl, ntiles - 1);
            xf_cp16(&smB[st][tl * 1024 + o], reinterpret_cast<const uint8_t*>(W + ((size_t)t * G + gk) * 512) + o);
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
    #pragma unroll
    for (int p = 0; p < NS - 1; ++p) {
        if (p < G) load_stage(p, p);
        xf_cp_commit();
    }
    for (int gk = 0; gk < G; ++gk) {
        const int st = gk % NS;
        if (gk + NS - 1 < G) load_stage((gk + NS - 1) % NS, gk + NS - 1);
        xf_cp_commit();
        xf_cp_wait<NS - 1>();
        __syncthreads();
        uint4 wv[2][2];
        #pragma unroll
        for (int jt = 0; jt < 2; ++jt)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                wv[jt][h] = *reinterpret_cast<const uint4*>(&smB[st][(wn * 2 + jt) * 1024 + h * 512 + lane * 16]);
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int ra = wm * 64 + mi * 16 + gid, rb = ra + 8;
            uint4 ah, al, bh, bl;
            xf_split4(*reinterpret_cast<const uint4*>(&smA[st][ra * 64 + tig * 16]), ah, al);
            xf_split4(*reinterpret_cast<const uint4*>(&smA[st][rb * 64 + tig * 16]), bh, bl);
            #pragma unroll
            for (int jt = 0; jt < 2; ++jt)
                #pragma unroll
                for (int h = 0; h < 2; ++h) xf_bf_group(acc[mi][jt][h], ah, al, bh, bl, wv[jt][h]);
        }
        __syncthreads();
    }
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

// ------------------------------------------------------------------------------------------------
// FP4L draft-head screen (--fp4l-draft-head screen; DRAFT-ONLY — the verify scores with the verify head, so served
// output is unchanged; only which tokens get drafted can move). Per graphed m = 1 draft pass, three launches replace
// the full draft-head GEMV (65,536 stored-bf16 rows = 335 MB):
//   xf_dhs_screen   an int4 per-row-scaled copy of the rows (84 MB): each 128-row block's max approximate logit ->
//                   bmax[blk]; also writes -inf (f16 0xFC00) to the block's 128 logits
//   xf_dhs_select   the DHS_T blocks with the largest bmax (ties: lower index) -> cand[0..T)
//   xf_dhs_rescore  the candidate blocks' rows through xf_bf16w_gemm's exact m = 1, ks = 1 instruction sequence
//                   (B32 rows of the draft head) -> their f16 logits, bit-identical to the full GEMV's
// The pass's existing tail (argmax / WP24 rq draft / WP15 penalty mirror / DDS rowmax) reads the logits buffer as
// before: a candidate row carries the full head's exact value, every other row -inf.
// ------------------------------------------------------------------------------------------------
#define XF_DHS_T 16
extern "C" __global__ void __launch_bounds__(256) xf_dhs_screen(const uint8_t* __restrict__ Q, const float* __restrict__ SS,
        const __half* __restrict__ X, float* __restrict__ bmax, __half* __restrict__ logits, int K) {
    XF_PDL_ENTRY();
    // x as fp32, transposed by 32-element chunk so that lane c's element j sits at j * (K/32 + 1) + c (conflict-free)
    __shared__ float xs[32 * 129];
    __shared__ float wm[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nch = K >> 5, cs = nch + 1;
    for (int i = tid; i < K; i += 256) xs[(i & 31) * cs + (i >> 5)] = __half2float(X[i]);
    if (tid < 64) reinterpret_cast<uint32_t*>(logits + (size_t)blockIdx.x * 128)[tid] = 0xFC00FC00u;
    __syncthreads();
    const int K2 = K >> 1;
    float best = -INFINITY;
    #pragma unroll 1
    for (int rr = 0; rr < 16; rr += 4) {
        const int r0 = blockIdx.x * 128 + warp * 16 + rr;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int off = lane * 16; off < K2; off += 512) {
            uint4 v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = xf_ldw4(Q + (size_t)(r0 + j) * K2 + off);
            const int c = off >> 4;   // this lane's 32-element chunk
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                #pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
                    const float xv = xs[(w * 8 + nb) * cs + c];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t u = w == 0 ? v[j].x : w == 1 ? v[j].y : w == 2 ? v[j].z : v[j].w;
                        const int q = ((int)(u << (28 - 4 * nb))) >> 28;
                        acc[j] = fmaf((float)q, xv, acc[j]);
                    }
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = acc[j];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            best = fmaxf(best, a * SS[r0 + j]);
        }
    }
    if (lane == 0) wm[warp] = best;
    __syncthreads();
    if (tid == 0) {
        float m = wm[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) m = fmaxf(m, wm[w]);
        bmax[blockIdx.x] = m;
    }
}
// grid 1, block 512: the XF_DHS_T best blocks of nblk <= 512 (rank by count of strictly better, ties lower index)
extern "C" __global__ void __launch_bounds__(512) xf_dhs_select(const float* __restrict__ bmax, int* __restrict__ cand, int nblk) {
    XF_PDL_ENTRY();
    __shared__ float v[512];
    const int i = threadIdx.x;
    v[i] = i < nblk ? bmax[i] : -INFINITY;
    __syncthreads();
    if (i >= nblk) return;
    const float mine = v[i];
    int rank = 0;
    for (int j = 0; j < nblk; ++j) { const float o = v[j]; rank += (o > mine) || (o == mine && j < i); }
    if (rank < XF_DHS_T) cand[rank] = i;
}
// grid XF_DHS_T, block 256 (8 warps = the 8 B32 tiles of one 128-row candidate block): xf_bf16w_gemm at M = 1, ks = 1
extern "C" __global__ __launch_bounds__(256) void xf_dhs_rescore(const uint16_t* __restrict__ W, const __half* __restrict__ X,
        __half* __restrict__ logits, const int* __restrict__ cand, int K) {
    XF_PDL_ENTRY();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = cand[blockIdx.x] * 8 + warp;
    const int n0w = tile * 16;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    const __half* xa = X + 8 * tig;
    const __half* xb = X + 8 * tig;
    const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
    int k = 0;
    for (; k + 128 <= K; k += 128) {
        uint4 b0[4], b1[4], xa4[4], xb4[4];
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            b0[g] = xf_ldw4(wt + (size_t)(k + 32 * g) * 16);
            b1[g] = xf_ldw4(wt + (size_t)(k + 32 * g) * 16 + 256);
            xa4[g] = xf_lda4(xa + k + 32 * g);
            xb4[g] = xf_lda4(xb + k + 32 * g);
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint4 ah, al, bh, bl;
            xf_split4(xa4[g], ah, al);
            xf_split4(xb4[g], bh, bl);
            xf_bf_group(acc[0], ah, al, bh, bl, b0[g]);
            xf_bf_group(acc[1], ah, al, bh, bl, b1[g]);
        }
    }
    for (; k < K; k += 32) {
        const uint4 b0 = xf_ldw4(wt + (size_t)k * 16), b1 = xf_ldw4(wt + (size_t)k * 16 + 256);
        uint4 ah, al, bh, bl;
        xf_split4(xf_lda4(xa + k), ah, al);
        xf_split4(xf_lda4(xb + k), bh, bl);
        xf_bf_group(acc[0], ah, al, bh, bl, b0);
        xf_bf_group(acc[1], ah, al, bh, bl, b1);
    }
    if (gid == 0) {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            *reinterpret_cast<__half2*>(&logits[n0w + 8 * h + 2 * tig]) = __floats2half2_rn(acc[h][0], acc[h][1]);
    }
}

// ------------------------------------------------------------------------------------------------
// --fp4l-mtp-sidecar: an EXL3 trellis MTP head's DENSE linears served next to the FP4L trunk (draft-only). An
// FP4L-owned copy of the EXL3 trellis decode (exl3_dq_from, bit-identical arithmetic) and of the m16 x N128 dense
// GEMM body (exl3_gemm_body), so this path shares no code with — and never changes — the EXL3 kernels. Integer
// K = 2..8 (k2 = 2K). The activation Hadamards around it are the EXL3 module's exl3_had_suh / exl3_had_svh,
// launched unchanged. fp32 split-K partials -> xf_tr_combine (ascending) -> f16 y_raw.
// ------------------------------------------------------------------------------------------------
#define XF_TR_MUL1 0x83DCD12Du
__device__ __forceinline__ uint16_t xf_tr_dq(const uint32_t* __restrict__ ring, int i0m, int i1m, int sf) {
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * XF_TR_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum);
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931));
    return (uint16_t)__half_as_ushort(w16);
}
__shared__ __half xf_tr_sa[16][16];
template <int BITS>
__device__ __forceinline__ void xf_tr_gemm_body(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
        float* __restrict__ part, int M, int K, int N, int kb0, int kb1) {
    __half (&sa)[16][16] = xf_tr_sa;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;
    constexpr int WRING = 8 * BITS, W16 = 16 * BITS;
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5, i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * W16);
    const size_t ring_kstride = (size_t)nb * (8u * BITS);
    for (int kb = kb0; kb < kb1; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            sa[row][col] = (row < M) ? xh[(size_t)row * K + kb * 16 + col] : __ushort_as_half(0);
        }
        __syncthreads();
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = xf_tr_dq(ring, s_i0m[h * 4 + 0], s_i1m[h * 4 + 0], s_sf[h * 4 + 0]);
            const uint16_t r1 = xf_tr_dq(ring, s_i0m[h * 4 + 1], s_i1m[h * 4 + 1], s_sf[h * 4 + 1]);
            const uint16_t r2 = xf_tr_dq(ring, s_i0m[h * 4 + 2], s_i1m[h * 4 + 2], s_sf[h * 4 + 2]);
            const uint16_t r3 = xf_tr_dq(ring, s_i0m[h * 4 + 3], s_i1m[h * 4 + 3], s_sf[h * 4 + 3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        ring += ring_kstride;
        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];
        #pragma unroll
        for (int h = 0; h < 2; ++h) xf_mma(acc[h], a0, a1, a2, a3, bfrag[h][0], bfrag[h][1]);
        __syncthreads();
    }
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (gid < M) *reinterpret_cast<float2*>(&part[(size_t)gid * N + col]) = make_float2(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *reinterpret_cast<float2*>(&part[(size_t)(gid + 8) * N + col]) = make_float2(acc[h][2], acc[h][3]);
    }
}
// grid (N/128, ks), block 256: slab blockIdx.y of the K/16 trellis rows -> part[slab][M][N] fp32. M <= 16.
extern "C" __global__ __launch_bounds__(256) void xf_tr_gemm(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
        float* __restrict__ part, int M, int K, int N, int bits, int ks) {
    XF_PDL_ENTRY();
    const int kbs = (K >> 4) / ks, kb0 = blockIdx.y * kbs;
    float* p = part + (size_t)blockIdx.y * M * N;
    switch (bits) {
        case 2: xf_tr_gemm_body<2>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 3: xf_tr_gemm_body<3>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 4: xf_tr_gemm_body<4>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 5: xf_tr_gemm_body<5>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 6: xf_tr_gemm_body<6>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 7: xf_tr_gemm_body<7>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        case 8: xf_tr_gemm_body<8>(trellis, xh, p, M, K, N, kb0, kb0 + kbs); break;
        default: break;
    }
}
// y_raw[i] = f16(sum over slabs s ascending of part[s][i]), i < mn
extern "C" __global__ void xf_tr_combine(const float* __restrict__ part, __half* __restrict__ yraw, int mn, int ks) {
    XF_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn) return;
    float a = 0.f;
    for (int s = 0; s < ks; ++s) a += part[(size_t)s * mn + i];
    yraw[i] = __float2half_rn(a);
}

// ------------------------------------------------------------------------------------------------
// xf_router_topk — the FP4L router's softmax / top-k / renorm (the stored-bf16 router path only; EXL3 keeps
// xq_router_topk). The softmax is xq_router_topk's code verbatim (same reductions => the same bits for every prob);
// only the SELECTION changes: xq_router_topk ran k block-wide argmax passes (k x log2(256) barriers); here every
// warp first takes its contiguous slice's top-k with shuffle argmaxes (no block barrier), then warp 0 merges the
// nw * k candidates. The order is the same total order — prob descending, id ascending — so for finite logits the
// (ids, wts) rows are bit-identical to xq_router_topk's, renormalised in the same j order. (NaN / Inf logits — NaN
// probs — can pick differently among the -2 sentinels; both are degenerate.) Same launch as xq_router_topk: grid B,
// block 256, dynamic smem (256 + ne + 512) * 4 B; k * (blockDim / 32) <= blockDim.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void xf_router_topk(int* __restrict__ ids, float* __restrict__ wts,
                                          const float* __restrict__ logits, int ne, int k, int B) {
    XF_PDL_ENTRY();
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float* prob = sm + bs;
    int* cand_i = (int*)(sm + bs + ne);
    const float* lg = logits + (long long)b * ne;
    // ---- xq_router_topk's softmax, verbatim
    float mx = -INFINITY;
    for (int i = tid; i < ne; i += bs) mx = fmaxf(mx, lg[i]);
    sm[tid] = mx; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] = fmaxf(sm[tid], sm[tid + s2]); __syncthreads(); }
    float gmax = sm[0];
    __syncthreads();
    float part = 0.0f;
    for (int i = tid; i < ne; i += bs) part += __expf(lg[i] - gmax);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float denom = sm[0];
    __syncthreads();   // every thread has read denom before sm[] is reused for the candidates
    for (int i = tid; i < ne; i += bs) prob[i] = __expf(lg[i] - gmax) / denom;
    __syncthreads();
    // ---- per-warp top-k of a contiguous slice (taken entries -> -2, as xq_router_topk marks its winners)
    const int w = tid >> 5, lane = tid & 31, nw = bs >> 5;
    const int per = (ne + nw - 1) / nw, lo = w * per, hi = min(ne, lo + per);
    float* cand_v = sm;   // [nw][k] values (the reduce scratch, free now)
    for (int j = 0; j < k; ++j) {
        float bv = -2.0f; int bi = ne;
        for (int i = lo + lane; i < hi; i += 32)
            if (prob[i] > bv || (prob[i] == bv && i < bi)) { bv = prob[i]; bi = i; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            cand_v[w * k + j] = bv; cand_i[w * k + j] = bi;
            if (bi >= lo && bi < hi) prob[bi] = -2.0f;
        }
        __syncwarp();
    }
    __syncthreads();
    // ---- warp 0 merges the nw * k candidates with the same order; the renorm as xq_router_topk's thread 0
    if (w == 0) {
        const int nc = nw * k;
        for (int j = 0; j < k; ++j) {
            float bv = -2.0f; int bi = ne, bp = -1;
            for (int c = lane; c < nc; c += 32) {
                const float v = cand_v[c]; const int i = cand_i[c];
                if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; bp = c; }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                const int op = __shfl_xor_sync(0xffffffffu, bp, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bp = op; }
            }
            if (lane == 0) {
                ids[(long long)b * k + j] = (bi < 0 || bi >= ne) ? 0 : bi;
                wts[(long long)b * k + j] = bv;
                if (bp >= 0) cand_v[bp] = -2.0f;
            }
            __syncwarp();
        }
        if (lane == 0) {
            float sum = 0.0f;
            for (int j = 0; j < k; j++) sum += wts[(long long)b * k + j];
            for (int j = 0; j < k; j++) wts[(long long)b * k + j] /= sum;
        }
    }
}

// B32 -> row-major bf16 [N][K] (the DFlash2 drafter's head view, the probes' references). One thread = 8 u16.
// n_elems = N * K of the REAL rows.
extern "C" __global__ void xf_b32_to_rm(uint16_t* __restrict__ out, const uint16_t* __restrict__ W, long long n_elems, int K) {
    XF_PDL_ENTRY();
    const long long i8 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i8 >= n_elems) return;
    const long long row = i8 / K;
    const int k = (int)(i8 - row * K);
    const long long t = row >> 4;
    const int rr = (int)(row & 15), h = rr >> 3, gid = rr & 7, g = k >> 5, tig = (k >> 3) & 3;
    const long long src = ((t * (K >> 5) + g) * 2 + h) * 256 + (gid * 4 + tig) * 8;
    *reinterpret_cast<uint4*>(out + i8) = *reinterpret_cast<const uint4*>(W + src);
}

// ------------------------------------------------------------------------------------------------
// xf_hc_fuse_b32 — the exact hyper-connection mixer on the STORED bf16 down / up in the B32 layout, M <= 16 rows,
// ONE launch: dd = f16(down . hn) ; s = f16(silu(dd / hcn)) ; uu = f16(up . s). Tensor cores (the exact hi/lo
// bf16 split of xf_bf16w_gemm: every product exact, fp32 accumulation).
//   grid G (1-D, every CTA co-resident: the launcher sizes G within this kernel's measured co-residency), block 256.
//   bar: TWO ints (count, generation: xf_grid_barrier_gen), zeroed once.
//   Phase A: (16-row tile of down, k-slab) items, KS = 32 slabs of rw/32 groups each -> fp32 partials ws[KS][16][lr];
//            grid barrier; phase R: the grid's threads split the m x lr outputs, each summing its 32 slabs in
//            ascending order (one f16 dd -> dd, the silu, one f16 s -> ws row block KS*16); grid barrier; every CTA
//            loads s into shared memory (each warp's first up tile is already in flight across the barriers).
//   Phase B: one 16-row tile of up per warp item, A fragments from shared memory -> uu (f16).
// Every output's arithmetic is a fixed sequence of the row's own values: independent of m and of G (batch
// invariance: a verify row equals its decode). Contract: rw % 1024 == 0 (32 slabs of whole groups), lr % 32 == 0,
// lr <= 480, m <= 16. ws: (KS + 1) * 16 * lr fp32 (the partials, then s).
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256, 2)
xf_hc_fuse_b32(__half* __restrict__ dd, __half* __restrict__ uu, const __half* __restrict__ hn,
               const uint16_t* __restrict__ w_down, const uint16_t* __restrict__ w_up, float* __restrict__ ws,
               cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr, int rw, int m) {
    XF_PDL_ENTRY();
    constexpr int KS = 32, UA = 5, UB = 5, SR_PAD = 32;
    __shared__ __align__(16) __half sb[16 * 512];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int nw = gridDim.x * 8, gw = blockIdx.x * 8 + warp;
    const int ta = min(gid, m - 1), tb = min(gid + 8, m - 1);
    // ---- phase A: down [lr][rw]
    {
        const int TA = lr >> 4, GA = rw >> 5, gps = GA / KS;
        const __half* xa = hn + (size_t)ta * rw + 8 * tig;
        const __half* xb = hn + (size_t)tb * rw + 8 * tig;
        for (int it = gw; it < TA * KS; it += nw) {
            const int t = it / KS, slab = it - t * KS;
            const uint16_t* wt = w_down + (size_t)t * GA * 512 + lane * 8;
            float acc[2][4];
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
            const int g0 = slab * gps, g1 = g0 + gps;
            for (int g = g0; g < g1; g += UA) {
                uint4 b0[UA], b1[UA], xa4[UA], xb4[UA];
                #pragma unroll
                for (int u = 0; u < UA; ++u) {
                    if (g + u < g1) {
                        b0[u] = xf_ldw4(wt + (size_t)(g + u) * 512);
                        b1[u] = xf_ldw4(wt + (size_t)(g + u) * 512 + 256);
                        xa4[u] = xf_lda4(xa + (g + u) * 32);
                        xb4[u] = xf_lda4(xb + (g + u) * 32);
                    }
                }
                #pragma unroll
                for (int u = 0; u < UA; ++u) {
                    if (g + u < g1) {
                        uint4 ah, al, bh, bl;
                        xf_split4(xa4[u], ah, al);
                        xf_split4(xb4[u], bh, bl);
                        xf_bf_group(acc[0], ah, al, bh, bl, b0[u]);
                        xf_bf_group(acc[1], ah, al, bh, bl, b1[u]);
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = t * 16 + 8 * h + 2 * tig;
                if (gid < m) *reinterpret_cast<float2*>(&ws[((size_t)slab * 16 + gid) * lr + n]) = make_float2(acc[h][0], acc[h][1]);
                if (gid + 8 < m) *reinterpret_cast<float2*>(&ws[((size_t)slab * 16 + gid + 8) * lr + n]) = make_float2(acc[h][2], acc[h][3]);
            }
        }
    }
    // phase-B prefetch: each warp's first up tile, first UB groups (independent of phase A)
    const int TB = rw >> 4, GB = lr >> 5;
    uint4 pb0[UB], pb1[UB];
    if (gw < TB) {
        const uint16_t* wt = w_up + (size_t)gw * GB * 512 + lane * 8;
        #pragma unroll
        for (int u = 0; u < UB; ++u)
            if (u < GB) { pb0[u] = xf_ldw4(wt + (size_t)u * 512); pb1[u] = xf_ldw4(wt + (size_t)u * 512 + 256); }
    }
    __threadfence();
    xf_grid_barrier_gen(bar, gridDim.x);
    // ---- phase R: dd = f16(sum of the slabs, ascending); s = f16(silu(dd / hcn)) -> ws[KS*16*lr ..] (rows < m)
    float* sg = ws + (size_t)KS * 16 * lr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * lr; i += gridDim.x * blockDim.x) {
        const int r = i / lr, n = i - r * lr;
        float acc = 0.f;
        #pragma unroll 8
        for (int q = 0; q < KS; ++q) acc += __ldcg(&ws[((size_t)q * 16 + r) * lr + n]);
        const __half d = __float2half_rn(acc);
        dd[(size_t)r * lr + n] = d;
        sg[(size_t)r * lr + n] = __half2float(__float2half_rn(xf_silu(__half2float(d) / hcn)));
    }
    __threadfence();
    xf_grid_barrier_gen(bar, gridDim.x);
    const int SR = lr + SR_PAD;
    for (int i = threadIdx.x; i < m * lr; i += blockDim.x) {
        const int r = i / lr, n = i - r * lr;
        sb[r * SR + n] = __float2half_rn(__ldcg(&sg[(size_t)r * lr + n]));
    }
    __syncthreads();
    // ---- phase B: up [rw][lr]
    {
        const __half* sa = sb + ta * SR + 8 * tig;
        const __half* sbb = sb + tb * SR + 8 * tig;
        for (int t = gw; t < TB; t += nw) {
            const uint16_t* wt = w_up + (size_t)t * GB * 512 + lane * 8;
            float acc[2][4];
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
            for (int g = 0; g < GB; g += UB) {
                uint4 b0[UB], b1[UB];
                const bool pre = (t == gw) && (g == 0);
                #pragma unroll
                for (int u = 0; u < UB; ++u) {
                    if (g + u < GB) {
                        if (pre) { b0[u] = pb0[u]; b1[u] = pb1[u]; }
                        else {
                            b0[u] = xf_ldw4(wt + (size_t)(g + u) * 512);
                            b1[u] = xf_ldw4(wt + (size_t)(g + u) * 512 + 256);
                        }
                    }
                }
                #pragma unroll
                for (int u = 0; u < UB; ++u) {
                    if (g + u < GB) {
                        uint4 ah, al, bh, bl;
                        xf_split4(*reinterpret_cast<const uint4*>(sa + (g + u) * 32), ah, al);
                        xf_split4(*reinterpret_cast<const uint4*>(sbb + (g + u) * 32), bh, bl);
                        xf_bf_group(acc[0], ah, al, bh, bl, b0[u]);
                        xf_bf_group(acc[1], ah, al, bh, bl, b1[u]);
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = t * 16 + 8 * h + 2 * tig;
                if (gid < m) *reinterpret_cast<__half2*>(&uu[(size_t)gid * rw + n]) = __floats2half2_rn(acc[h][0], acc[h][1]);
                if (gid + 8 < m) *reinterpret_cast<__half2*>(&uu[(size_t)(gid + 8) * rw + n]) = __floats2half2_rn(acc[h][2], acc[h][3]);
            }
        }
    }
}

// == xq_hc_mix_rg with the stored bf16 block_inject weights (decode, verify and — any B — prefill).
// grid (ninj + ceil(h/1024), B), block 1024 (the tree leaves).
extern "C" __global__ void __launch_bounds__(1024)
xf_hc_mix_bf(__half* __restrict__ x, float* __restrict__ inj,
             const __half* __restrict__ hn, const __half* __restrict__ u,
             const uint16_t* __restrict__ winj, int h, int hc, int B) {
    XF_PDL_ENTRY();
    const int b = blockIdx.y;
    if (b >= B) return;
    hn += (long long)b * h * hc;
    u += (long long)b * h * hc;
    x += (long long)b * h;
    inj += (long long)b * hc;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int ninj = (winj != nullptr) ? hc : 0;
    if ((int)blockIdx.x < ninj) {
        __shared__ float sm[1024];
        const int s = blockIdx.x;
        const int n = h * hc;
        const uint16_t* wr = winj + (long long)s * n;
        float part = 0.0f;
        constexpr int J = 12;
        if (n <= J * bs) {
            float wv[J], hv[J];
            #pragma unroll
            for (int j = 0; j < J; ++j) {
                const int c = tid + j * bs;
                wv[j] = c < n ? xf_b2f(wr[c]) : 0.0f;
                hv[j] = c < n ? xf_h2f(hn[c]) : 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < J; ++j)
                if (tid + j * bs < n) part += wv[j] * hv[j];
        } else {
            for (int c = tid; c < n; c += bs) part += xf_b2f(wr[c]) * xf_h2f(hn[c]);
        }
        sm[tid] = part;
        __syncthreads();
        for (int s2 = bs / 2; s2 >= 32; s2 >>= 1) {
            if (tid < s2) sm[tid] += sm[tid + s2];
            __syncthreads();
        }
        if (tid < 32) {
            float v = sm[tid];
            #pragma unroll
            for (int s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);
            if (tid == 0) inj[s] = 2.0f * xf_sig(v / (float)hc);
        }
    } else {
        const int i = ((int)blockIdx.x - ninj) * bs + tid;
        if (i < h) {
            float acc = 0.0f;
            for (int s = 0; s < hc; s++)
                acc += xf_sig(xf_h2f(u[(long long)s * h + i])) * xf_h2f(hn[(long long)s * h + i]);
            x[i] = xf_f2h(acc / (float)hc);
        }
    }
}

// == xq_pf_hc_mix_2560x4 (prefill fused mix) with the stored bf16 block_inject weights. grid B, block 128.
template <int NK, int HC>
__device__ __forceinline__ void xf_pf_hc_mix_body(__half* __restrict__ x, float* __restrict__ inj,
                                                  const __half* __restrict__ hn, const __half* __restrict__ u,
                                                  const uint16_t* __restrict__ winj, int B) {
    constexpr int H = NK * 512, N = H * HC, KT = NK * HC;
    const int b = blockIdx.x;
    if (b >= B) return;
    hn += (long long)b * N;
    u += (long long)b * N;
    x += (long long)b * H;
    inj += (long long)b * HC;
    const int T = threadIdx.x;
    const bool dj = (winj != nullptr);
    float acc[NK][4];
    float pa[HC][4], pb[HC][4];
    #pragma unroll
    for (int m = 0; m < NK; ++m)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[m][e] = 0.0f;
    #pragma unroll
    for (int t = 0; t < HC; ++t)
        #pragma unroll
        for (int e = 0; e < 4; ++e) { pa[t][e] = 0.0f; pb[t][e] = 0.0f; }
    #pragma unroll
    for (int k = 0; k < KT; ++k) {
        const int c = 512 * k + 4 * T;
        float hf[4], uf[4];
        xf_pf_unpack4(*reinterpret_cast<const uint2*>(hn + c), hf);
        xf_pf_unpack4(*reinterpret_cast<const uint2*>(u + c), uf);
        const int m = k % NK;
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[m][e] += xf_sig(uf[e]) * hf[e];
        if (dj) {
            #pragma unroll
            for (int t = 0; t < HC; ++t) {
                float wf[4];
                xf_pf_unpack4_bf(*reinterpret_cast<const uint2*>(winj + (long long)t * N + c), wf);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (k & 1) pb[t][e] += wf[e] * hf[e]; else pa[t][e] += wf[e] * hf[e];
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < NK; ++m) {
        float o[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = acc[m][e] / (float)HC;
        *reinterpret_cast<uint2*>(x + 512 * m + 4 * T) = xf_pf_pack4(o);
    }
    if (dj) {
        __shared__ float sm[HC][512];
        #pragma unroll
        for (int t = 0; t < HC; ++t)
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm[t][4 * T + e] = pa[t][e] + pb[t][e];
        __syncthreads();
        const int wp = T >> 5, lane = T & 31;
        if (wp < HC) {
            const float r = xf_pf_tree512(sm[wp], lane);
            if (lane == 0) inj[wp] = 2.0f * xf_sig(r / (float)HC);
        }
    }
}
extern "C" __global__ void __launch_bounds__(128)
xf_pf_hc_mix_bf(__half* __restrict__ x, float* __restrict__ inj, const __half* __restrict__ hn,
                const __half* __restrict__ u, const uint16_t* __restrict__ winj, int B) {
    XF_PDL_ENTRY();
    xf_pf_hc_mix_body<5, 4>(x, inj, hn, u, winj, B);
}

// ------------------------------------------------------------------------------------------------
// Embedding, shared-expert gate
// ------------------------------------------------------------------------------------------------

// == xq_embed_resid with the stored bf16 table: resid[b][s*h+i] = f32(embed[tok[b]][i]) (exact).
extern "C" __global__ void xf_embed_resid_bf(float* __restrict__ resid, const uint16_t* __restrict__ emb,
                                             const int* __restrict__ toks, int h, int hc, int B) {
    XF_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (i >= rw * B) return;
    int b = (int)(i / rw);
    int r = (int)(i % rw);
    resid[i] = xf_b2f(emb[(long long)toks[b] * h + (r % h)]);
}

// xf_sg_gate — the shared-expert gate logit L = sum_c sg[c] * x[c] (the STORED bf16 gate, fp32, a fixed
// order) for the trunk's UNCHANGED combine kernels. Those kernels compute their own L' = sum_c sgc[c] *
// xg[c] in fp32 and use x for nothing else, so the FP4L path hands them a constant f16 vector
// sgc = (1, 2^-11, 2^-22, 2^-24, 2^-24, 2^-24, 0, ...) and per row six f16 "channels" xg[0..6) holding L's
// bits in fixed windows 2^15..2^5 | 2^4..2^-6 | 2^-7..2^-17 | 2^-18..2^-28 | 2^-29..2^-39 | 2^-40..2^-48,
// each scaled by 1/sgc[c] into f16 (every channel value is exactly representable). Every product
// sgc[c] * xg[c] is exact, and every partial sum of any reduction tree over them is a subset of L's own
// <= 24 significant bits (so exact in fp32): the combine rebuilds L bit for bit for 2^-25 <= |L| <= 65504
// (below 2^-25 the windows drop bits under 2^-48, where the fp32 sigmoid is 0.5 regardless; |L| > 65504
// is clamped to 65504, where it is exactly 0 or 1). grid M, block 256; xg [M][h] whose columns >= 6 the
// caller keeps zero (zeroed at allocation, never written).
extern "C" __global__ void xf_sg_gate(__half* __restrict__ xg, const uint16_t* __restrict__ sg,
                                      const __half* __restrict__ x, int h, int M) {
    XF_PDL_ENTRY();
    const int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float sm[256];
    const int tid = threadIdx.x;
    float part = 0.0f;
    for (int c = tid; c < h; c += 256) part += xf_b2f(sg[c]) * xf_h2f(x[(long long)m * h + c]);
    sm[tid] = part;
    __syncthreads();
    for (int s2 = 128; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    if (tid == 0) {
        const float L = sm[0];
        float a = fminf(fabsf(L), 65504.0f);
        const bool neg = L < 0.0f;
        const int lo[6] = {5, -6, -17, -28, -39, -48};   // window bottoms
        const int up[6] = {0, 11, 22, 24, 24, 24};       // f16 up-scale = -log2(sgc[c])
        __half* o = xg + (long long)m * h;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            const float v = ldexpf(floorf(ldexpf(a, -lo[i])), lo[i]);   // a's bits >= 2^lo[i] (exact)
            a -= v;                                                     // exact
            const float c = ldexpf(v, up[i]);
            o[i] = __float2half_rn(neg ? -c : c);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// n-gram (PLE) twins: fp32 norm weights / fp32 conv weights (the stored fp32 norms, the stored bf16
// conv taps widened exactly).
// ------------------------------------------------------------------------------------------------

// == xq_ple_norm_f16 (f16 in -> f32 out; key path), fp32 weights. grid (m*4), block 256.
extern "C" __global__ void xf_ple_norm_f16_f(const __half* __restrict__ x, const float* __restrict__ w,
                                             float* __restrict__ out, float eps) {
    XF_PDL_ENTRY();
    int ps = blockIdx.x;
    int s = ps & 3;
    const __half* xp = x + (long long)ps * 2560;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) { float v = xf_h2f(xp[d]); part += v * v; }
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float inv = rsqrtf(red[0] / 2560.f + eps);
    const float* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xf_h2f(xp[d]) * inv * (1.f + wp[d]);
}

// == xq_ple_norm_f32 (f32 residual in -> f32 out; query path), fp32 weights.
extern "C" __global__ void xf_ple_norm_f32_f(const float* __restrict__ x, const float* __restrict__ w,
                                             float* __restrict__ out, float eps) {
    XF_PDL_ENTRY();
    int ps = blockIdx.x;
    int s = ps & 3;
    const float* xp = x + (long long)ps * 2560;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) { float v = xp[d]; part += v * v; }
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float inv = rsqrtf(red[0] / 2560.f + eps);
    const float* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xp[d] * inv * (1.f + wp[d]);
}

// == xq_ple_norm_hh (f16 in -> f16 out; conv path), fp32 weights.
extern "C" __global__ void xf_ple_norm_hh_f(const __half* __restrict__ x, const float* __restrict__ w,
                                            __half* __restrict__ out, float eps) {
    XF_PDL_ENTRY();
    int ps = blockIdx.x;
    int s = ps & 3;
    const __half* xp = x + (long long)ps * 2560;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) { float v = xf_h2f(xp[d]); part += v * v; }
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float inv = rsqrtf(red[0] / 2560.f + eps);
    const float* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xf_f2h(xf_h2f(xp[d]) * inv * (1.f + wp[d]));
}

// == xq_ple_conv (decode / verify step), fp32 conv taps [10240][4].
extern "C" __global__ void xf_ple_conv_f(float* __restrict__ resid, const __half* __restrict__ gated,
                                         const __half* __restrict__ normed, const __half* __restrict__ state,
                                         const float* __restrict__ conv_w, int m,
                                         const int* __restrict__ slot_ids) {
    XF_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)m * 10240) return;
    int q = (int)(i / 10240);
    int d = (int)(i % 10240);
    int slot = slot_ids[q * 2];
    const __half* st = state + ((long long)slot * 10240 + d) * 9;
    float acc = conv_w[d * 4 + 0] * xf_h2f(st[0])
              + conv_w[d * 4 + 1] * xf_h2f(st[3])
              + conv_w[d * 4 + 2] * xf_h2f(st[6])
              + conv_w[d * 4 + 3] * xf_h2f(normed[i]);
    resid[i] += xf_h2f(gated[i]) + xf_silu(acc);
}

// == xq_ple_conv_chunk (prefill chunk), fp32 conv taps.
extern "C" __global__ void xf_ple_conv_chunk_f(float* __restrict__ resid, const __half* __restrict__ gated,
                                               const __half* __restrict__ normed, __half* __restrict__ state,
                                               const float* __restrict__ conv_w, int slot, int C) {
    XF_PDL_ENTRY();
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half st[9];
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    const float w0 = conv_w[d * 4 + 0];
    const float w1 = conv_w[d * 4 + 1];
    const float w2 = conv_w[d * 4 + 2];
    const float w3 = conv_w[d * 4 + 3];
    for (int t = 0; t < C; t++) {
        const long long i = (long long)t * 10240 + d;
        const float cur = xf_h2f(normed[i]);
        const float acc = w0 * xf_h2f(st[0]) + w1 * xf_h2f(st[3])
                        + w2 * xf_h2f(st[6]) + w3 * cur;
        resid[i] += xf_h2f(gated[i]) + xf_silu(acc);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xf_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// == xq_ple_conv_chunk_seg (lane-pack segmented chunk), fp32 conv taps. grid (ceil(10240/256), k_seg).
extern "C" __global__ void xf_ple_conv_chunk_seg_f(float* __restrict__ resid,
                                                   const __half* __restrict__ gated,
                                                   const __half* __restrict__ normed,
                                                   __half* __restrict__ state,
                                                   const float* __restrict__ conv_w,
                                                   const int* __restrict__ seg) {
    XF_PDL_ENTRY();
    const int s = blockIdx.y * 4;
    const int row0 = seg[s + 1], C = seg[s + 2];
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half st[9];
    __half* sbase = state + ((long long)blockIdx.y * 10240 + d) * 9;
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    const float w0 = conv_w[d * 4 + 0];
    const float w1 = conv_w[d * 4 + 1];
    const float w2 = conv_w[d * 4 + 2];
    const float w3 = conv_w[d * 4 + 3];
    for (int t = 0; t < C; t++) {
        const long long i = (long long)(row0 + t) * 10240 + d;
        const float cur = xf_h2f(normed[i]);
        const float acc = w0 * xf_h2f(st[0]) + w1 * xf_h2f(st[3])
                        + w2 * xf_h2f(st[6]) + w3 * cur;
        resid[i] += xf_h2f(gated[i]) + xf_silu(acc);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xf_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// ------------------------------------------------------------------------------------------------
// Head views for the DFlash2 drafter (draft-only): the verify head as bf16 [N][K].
// ------------------------------------------------------------------------------------------------
// MXFP8 [N][K] -> bf16 [N][K], exact: an e4m3 value has <= 4 significant bits and x 2^(s-127) stays in
// bf16's (= fp32's) exponent range.
extern "C" __global__ void xf_mx8_dequant_bf16(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                               uint16_t* __restrict__ out, long long n_elems, int K) {
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
        const __nv_bfloat162 r = __floats2bfloat162_rn(v.x * f, v.y * f);
        op[j] = *reinterpret_cast<const uint32_t*>(&r);
    }
    *reinterpret_cast<uint4*>(out + i8) = o;
}
// u16 copy (the stored bf16 head -> the DFlash2 head buffer). Grid-stride.
extern "C" __global__ void xf_copy_u16(uint16_t* __restrict__ out, const uint16_t* __restrict__ in, long long n) {
    XF_PDL_ENTRY();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = in[i];
}
