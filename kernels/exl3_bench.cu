// =============================================================================
// kernels/exl3_bench.cu — S-A3-c: EXL3 weight-path GEMM
// (trellis decode fused into fp16 HMMA), probe/bench-only module.
//
// Arch: sm_121 (fp16 mma.sync.m16n8k16 and dp4a are baseline CC 12.1 features —
// no family/architecture suffix needed; see gb10-cuda skill target rules).
// Loaded lazily ONLY by --probe-exl3-kernel / --probe-exl3-binv / --bench-exl3-gemm
// (precedent: mxfp4_bench). NOT in the serving manifest; no serving-path change.
//
// Decode math mirrors src/exl3.rs `decode_block` + `tmap` bit-exactly (that decode
// is G-A3-1-green vs ext.reconstruct over 21.3M blocks, S-A3-b):
//   ring  = the block's 16*bits int16 viewed as 8*bits LE uint32
//   t     = temporal index; window END b1 = (t+257)*bits, START b0 = b1-16
//   idx   = ((ring[i1%W] >> s) | (ring[i0%W] << (32-s))) & 0xFFFF, s = ((i1+1)*32)-b1
//           (s==0 -> ring[i1%W] & 0xFFFF; __funnelshift_r reproduces both branches)
//   w     = hfma(cvt(sum), 0x1EEE, 0xC931), sum = dp4a_u32(idx*0x83DCD12D, 0x01010101, 0x400)
//           sum = 1024 + UNSIGNED bytesum (matches the Rust LUT; fused fp16 FMA path
//           verified equal to the double-rounded f32 LUT on all 65536 entries)
//   place W[16kb+r][16nb+c] = decoded[tmap(r,c)],
//   tmap(r,c) = 8*l + (r%2) + 2*((r/8)%2) + 4*(c/8) + 32*(c%2), l = 8*((c/2)%4) + ((r%8)/2)
//
// GEMM: y_raw[m,n] = sum_k xh[m,k] * W[n,k]  (W = decoded trellis, fp32 accum)
//   CTA: 256 threads = 8 warps, tile m16 x N128; warp w owns columns [n0+16w, n0+16w+16).
//   Per k16 step each warp decodes ONE 16x16 trellis block DIRECTLY into two
//   m16n8k16 B-fragments (n-tiles h=0,1) — no weight SMEM round-trip.
//   B-fragment slot (lane l, tile h, slot s in 0..3):
//     r = 2*(l%4) + (s&1) + 8*(s>>1),  c = 8*h + (l>>2),  t = tmap(r,c).
//   A operand: xh staged in SMEM per k16 step (rows >= M zero-filled), manual
//   fragment build. Accumulators: fp32, K ascending, fixed order (batch invariance:
//   M only pads the m16 tile with zeros; the per-element k-reduction order is fixed
//   by the mma instruction and the K loop — AGENTS 2.4 / G-A3-5).
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_fp8.h>
#include "tp_doorbell.h"   // TP-G: the doorbell ABI (tp_dev_ctx, flag offsets) for the folded decode all-reduce

#define EXL3_MUL1 0x83DCD12Du

// ---------------------------------------------------------------------------
// Decode with precomputed (i0m, i1m, sf) — the hoisted-constant form used by the
// GEMM body. THE one decode device function (AGENTS 2.8: every consumer shares it).
// Returns the fp16 bit pattern of the decoded weight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t exl3_dq_from(const uint32_t* __restrict__ ring,
                                                 int i0m, int i1m, int sf) {
    // Oracle placement: the LATER word (i1) is LOW in the merged pair, the EARLIER
    // word (i0) HIGH — idx = ((ring[i1] >> s) | (ring[i0] << (32-s))) & 0xFFFF,
    // s == 0 -> ring[i1] & 0xFFFF. __funnelshift_r(lo, hi, s) = ((hi:lo) >> s) low 32,
    // so lo = ring[i1m] and hi = ring[i0m] (swapped args are invisible when i0 == i1,
    // i.e. exactly the ~50% of b=3 windows that don't cross a word boundary — the
    // second gate run's 47% mismatch fingerprint).
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    // UNSIGNED dp4a overload: value = 1024 + unsigned byte-sum s (must match Rust LUT).
    // S-A3-n (SASS diff vs the reference implementation's exl3_moe_coop_a: ZERO I2F there, one
    // I2F.F16.U32 per weight here — a reduced-rate conversion on the decode's
    // critical path): accumulate 0x6400 instead of 1024. 0x6400 is the fp16 bit
    // pattern of 1024.0 and s <= 4*255 = 1020 < 1024, so 0x6400 + s IS the fp16
    // encoding of 1024 + s (exponent 25, mantissa s; ulp = 1 on [1024, 2048)) —
    // exactly what cvt.rn.f16.u32(1024 + s) returned. Bit-identical, I2F gone.
    // (The spec §3.2 "0x6400" was this trick, not a typo.)
    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum); // fp16 bits of 1024 + s
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE),   // k_inv
                                  __ushort_as_half((unsigned short)0xC931)); // k_bias
    return (uint16_t)__half_as_ushort(w16);               // single-rounded fp16 FMA
}

// Decode by temporal index t (computes the window indices from t).
__device__ __forceinline__ uint16_t exl3_dq_t(const uint32_t* __restrict__ ring,
                                              int t, int bits) {
    const int b1 = (t + 257) * bits;      // window END (unwrapped bit)
    const int b0 = b1 - 16;               // window START
    const int i0 = b0 >> 5;
    const int i1 = (b1 - 1) >> 5;
    const int sf = ((i1 + 1) << 5) - b1;  // 0..31
    const unsigned w = (unsigned)(bits * 8);
    return exl3_dq_from(ring, (int)(i0 % w), (int)(i1 % w), sf);
}

// ---------------------------------------------------------------------------
// Kernel 1: decode gate (G-A3-1 through the kernel path).
// One warp per 16x16 trellis block; thread lane decodes t in {lane + 32*step}.
// rc_of_t[256]: inverse tmap LUT (host-built from the unit-tested Rust bijection):
// rc_of_t[t] = r*16 + c. Output: W row-major [kb*16, nb*16] fp16 bit patterns.
// ---------------------------------------------------------------------------
// PDL (A5 D14, 2026-09-26): every entry kernel starts by waiting for its programmatic predecessor
// and releasing its own dependents. On a plain launch both instructions are no-ops; with
// GB10_EXL3_PDL=1 the launcher sets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION so the
// next grid becomes resident while this one's last wave drains (launch gap + tail overlap).
// The dependent release fires only after ALL CTAs of a grid have started, so co-resident
// grid-barrier kernels (xq_hc_fuse*) can never be starved by a waiting successor.
#define XQ_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

// TUNE T0 (PLAN/AUTOTUNE_DESIGN.md finding §14.1): the stale-kernel handshake. build.rs compiles
// this module with -DKERNEL_BUILD_ID (the hash of every .cu it builds); FwdModel::load launches this
// entry from the LOADED module and refuses a mismatch — the served EXL3 path used to run an old
// exl3_bench.ptx silently whenever the kernel names still resolved.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

// =============================================================================
// TUNE T1 (PLAN/AUTOTUNE_DESIGN.md §5.4-§5.7): measurement-harness kernels. Launched ONLY by the
// `--autotune` harness (src/exl3_autotune.rs), never on a serving path.
// Target contract: baseline sm_121 (this module's): ld.global.cg, %globaltimer, nanosleep, 64-bit
// atomicAdd and warp shuffles are CC 12.1 baseline — no family/architecture suffix is needed, and
// the fatbin (CUBIN) and PTX paths get the same code. All three open with XQ_PDL_ENTRY (a PDL
// launch of a kernel without griddepcontrol.wait could read its predecessor's outputs early).
// =============================================================================

__device__ __forceinline__ unsigned long long xq_tune_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// L2 conditioning (§5.4): stream `n16` 16-byte words through L2 with ld.global.cg (L2, not L1) —
// a read-only replay of the unit's real predecessor's last reads that leaves no dirty lines (unlike
// the reference implementation's memset thrash). The loads are asm volatile (never elided or sunk under `write`);
// `sink` is written only when `write` != 0. Block 0 / thread 0 then waits until `min_ns` have
// passed since it started, so the host has submitted the timed unit before the stream reaches it
// (§5.5: the event pair then measures device time only). Grid-stride, 4 independent 16-B loads in
// flight per thread per iteration; no shared memory; no local arrays.
extern "C" __global__ void __launch_bounds__(256)
xq_l2_touch(const uint4* __restrict__ p, long long n16, unsigned int* __restrict__ sink, int write,
            long long min_ns) {
    XQ_PDL_ENTRY();
    const bool timer = (blockIdx.x == 0 && threadIdx.x == 0 && min_ns > 0);
    const unsigned long long t0 = timer ? xq_tune_gtimer() : 0ull;
    const long long stride = (long long)gridDim.x * blockDim.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0u;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        unsigned a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(p + i));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "l"(p + i + stride));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3) : "l"(p + i + 2 * stride));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "l"(p + i + 3 * stride));
        acc ^= a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3 ^ c0 ^ c1 ^ c2 ^ c3 ^ d0 ^ d1 ^ d2 ^ d3;
    }
    for (; i < n16; i += stride) {
        unsigned a0, a1, a2, a3;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(p + i));
        acc ^= a0 ^ a1 ^ a2 ^ a3;
    }
    if (write) sink[0] = acc;
    if (timer) {
        while ((long long)(xq_tune_gtimer() - t0) < min_ns) __nanosleep(1000);
    }
}

// splitmix64 finalizer
__device__ __forceinline__ unsigned long long xq_tune_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Correctness rides every sample (§5.6): out[slot] += SUM over the regions r of one digest set
// (blockIdx.y = r, the region's index WITHIN the set) and their 32-bit words w of
// mix64(mix64(r << 40 ^ w) ^ word). Integer addition mod 2^64 is associative and commutative, so the
// digest is independent of the grid, of block scheduling and of the atomic order: deterministic.
// regs = device table of (ptr, n_words) u64 pairs starting at entry reg0; every ptr is 4-byte
// aligned (host-asserted). One atomic per warp.
extern "C" __global__ void __launch_bounds__(256)
xq_digest64(const unsigned long long* __restrict__ regs, int reg0, unsigned long long* __restrict__ out,
            int slot) {
    XQ_PDL_ENTRY();
    const int r = (int)blockIdx.y;
    const unsigned* __restrict__ p = (const unsigned*)regs[2 * (reg0 + r)];
    const long long n = (long long)regs[2 * (reg0 + r) + 1];
    unsigned long long h = 0ull;
    const unsigned long long salt = (unsigned long long)r << 40;
    for (long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x; w < n;
         w += (long long)gridDim.x * blockDim.x) {
        h += xq_tune_mix64(xq_tune_mix64(salt ^ (unsigned long long)w) ^ (unsigned long long)p[w]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) h += __shfl_down_sync(0xffffffffu, h, o);
    if ((threadIdx.x & 31) == 0 && h != 0ull) atomicAdd(out + slot, h);
}

// Known-positive timing probe (§5.7 test 2): one thread waits `ns` nanoseconds on %globaltimer
// (a candidate that adds this node must be measured at +ns). Launch (1,1,1) x (32,1,1).
// `reserved` is unused (keeps the typed launcher's argument tuple at two elements).
extern "C" __global__ void xq_spin_ns(long long ns, int reserved) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned long long t0 = xq_tune_gtimer();
    while ((long long)(xq_tune_gtimer() - t0) < ns) __nanosleep(500);
}

extern "C" __global__ void exl3_decode_dump(const uint16_t* __restrict__ trellis,
                                            const uint8_t* __restrict__ rc_of_t,
                                            uint16_t* __restrict__ wout,
                                            int nb, int bits) {
    XQ_PDL_ENTRY();
    const int b = blockIdx.x;                 // trellis block = kb*nb + n
    const int lane = threadIdx.x & 31;
    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + (size_t)b * (16 * bits));
    const int kb = b / nb, nbk = b % nb;
    uint16_t* __restrict__ orow = wout + (size_t)kb * 16 * (nb * 16) + nbk * 16;

    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        const uint16_t v = exl3_dq_t(ring, t, bits);
        const int rc = rc_of_t[t];
        orow[(rc >> 4) * (nb * 16) + (rc & 15)] = v;
    }
}

// S-A3-u G: the prefill reconstruct front-end, same values, coalesced. One CTA (8 warps) per
// 16 x 128 output strip = 8 consecutive trellis blocks of one k16 row. Each warp copies its
// block's window (8*bits words) to smem and decodes with the SAME exl3_dq_t (same words => same
// bits); the tmap scatter lands in a smem tile; the strip leaves as 16-B vector stores (the old
// kernel issued 2-B stores scattered over 16 rows per instruction). Requires nb % 8 == 0.
extern "C" __global__ void __launch_bounds__(256) exl3_decode_dump8(
        const uint16_t* __restrict__ trellis, const uint8_t* __restrict__ rc_of_t,
        uint16_t* __restrict__ wout, int nb, int bits) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) uint32_t s_ring[8][64];            // bits <= 8
    __shared__ __align__(16) uint16_t s_tile[16][128 + 8];
    __shared__ uint8_t s_rc[256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb8 = nb >> 3;
    const int kb = blockIdx.x / nb8, n0 = (blockIdx.x - kb * nb8) * 8;
    s_rc[threadIdx.x] = rc_of_t[threadIdx.x];
    const int ww = bits * 8;
    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)kb * nb + n0 + warp) * (16 * bits));
    for (int i = lane; i < ww; i += 32) s_ring[warp][i] = __ldg(ring + i);
    __syncthreads();
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        const uint16_t v = exl3_dq_t(s_ring[warp], t, bits);
        const int rc = s_rc[t];
        s_tile[rc >> 4][warp * 16 + (rc & 15)] = v;
    }
    __syncthreads();
    const int row = threadIdx.x >> 4, c8 = threadIdx.x & 15;
    *(uint4*)(wout + ((size_t)kb * 16 + row) * ((size_t)nb * 16) + n0 * 16 + c8 * 8) =
        *(const uint4*)&s_tile[row][c8 * 8];
}

// ---------------------------------------------------------------------------
// Kernels 2/3: the two Hadamard passes.
//   xh = H128(x) * suh  (per 128-block along K)     -- input side
//   y  = H128(y_raw) * svh (per 128-block along N)  -- output side
// One warp per (row, 128-block). Unnormalized Walsh-Hadamard butterfly in fp32,
// fixed stage order (deterministic / batch-invariant). Normalization convention
// is an open item settled by the cross-chain script (bank, S-A3-c).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void exl3_had128(float v[4], int lane) {
    // len = 1, 2 intra-lane; len = 4..64 cross-lane (4 elems/lane -> lane dist len/4).
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int len = 1 << s;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if ((r & len) == 0) {
                const float a = v[r], b = v[r + len];
                v[r] = a + b;
                v[r + len] = a - b;
            }
        }
    }
    // Cross-lane stages: element distance 4..64 = lane distance 1..16 (4 elems/lane).
    #pragma unroll
    for (int s = 2; s < 7; ++s) {
        const unsigned mask = (unsigned)(1u << (s - 2));
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask);
            v[r] = (lane & mask) ? (t - v[r]) : (v[r] + t);
        }
    }
    // Normalization: THEIR convention (probed on .13, S-A3-c xchain bisect) is the
    // ORTHONORMAL Hadamard — ext.had_r_128 on a delta impulse returns 1/sqrt(128)
    // uniformly, RMS gain exactly 1.0, r_scale arg 1.0 in their calls. The pack's
    // suh/svh are calibrated under THIS convention, so both our had kernels
    // (had_suh, had_svh — the only two callers of this helper) normalize identically.
    const float hnorm = 0.08838834764831845f; // 1/sqrt(128)
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] *= hnorm;
}

__device__ __forceinline__ void exl3_had_suh_dev(const __half* __restrict__ x,
                                                 const __half* __restrict__ suh,
                                                 __half* __restrict__ xh,
                                                 int k) {
    const int blk = blockIdx.x;               // m * k/128
    const int kb = blk % (k >> 7);
    const int m = blk / (k >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * k + (size_t)kb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        // THEIR order (xchain bisect, S-A3-c): their had kernel pre_scale applies suh
        // BEFORE the butterfly — the weight-side order (had_l then x suh along K)
        // forces the activation side to suh-first. The spec §3.3 line
        // "x_h = H128(x) · suh" describes the post-scale form and does NOT match
        // their kernel; the reference chain is the arbiter.
        v[r] = __half2float(x[base + r]) * __half2float(suh[kb * 128 + lane * 4 + r]);
    }
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        xh[base + r] = __float2half_rn(v[r]);
    }
}
extern "C" __global__ void exl3_had_suh(const __half* __restrict__ x,
                                        const __half* __restrict__ suh,
                                        __half* __restrict__ xh,
                                        int k) {
    XQ_PDL_ENTRY();
    exl3_had_suh_dev(x, suh, xh, k);
}
// S-A3-o C1: a same-input chain PAIR in one launch (grid.z = member).
extern "C" __global__ void exl3_had_suh_x2(const __half* __restrict__ x,
                                           const __half* __restrict__ suh0, const __half* __restrict__ suh1,
                                           __half* __restrict__ xh0, __half* __restrict__ xh1, int k) {
    XQ_PDL_ENTRY();
    if (blockIdx.z == 0) exl3_had_suh_dev(x, suh0, xh0, k);
    else                 exl3_had_suh_dev(x, suh1, xh1, k);
}

__device__ __forceinline__ void exl3_had_svh_dev(const __half* __restrict__ yraw,
                                                 const __half* __restrict__ svh,
                                                 __half* __restrict__ y,
                                                 int n) {
    const int blk = blockIdx.x;               // m * n/128
    const int nb = blk % (n >> 7);
    const int m = blk / (n >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * n + (size_t)nb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __half2float(yraw[base + r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int j = lane * 4 + r;
        y[base + r] = __float2half_rn(v[r] * __half2float(svh[nb * 128 + j]));
    }
}
extern "C" __global__ void exl3_had_svh(const __half* __restrict__ yraw,
                                        const __half* __restrict__ svh,
                                        __half* __restrict__ y,
                                        int n) {
    XQ_PDL_ENTRY();
    exl3_had_svh_dev(yraw, svh, y, n);
}
extern "C" __global__ void exl3_had_svh_x2(const __half* __restrict__ yraw0, const __half* __restrict__ yraw1,
                                           const __half* __restrict__ svh0, const __half* __restrict__ svh1,
                                           __half* __restrict__ y0, __half* __restrict__ y1, int n) {
    XQ_PDL_ENTRY();
    if (blockIdx.z == 0) exl3_had_svh_dev(yraw0, svh0, y0, n);
    else                 exl3_had_svh_dev(yraw1, svh1, y1, n);
}

// ---------------------------------------------------------------------------
// Kernel 4: the fused GEMM. Template on bits (compile-time ring width).
// ---------------------------------------------------------------------------
template <int BITS>
__device__ __forceinline__ void exl3_gemm_body(const uint16_t* __restrict__ trellis,
                                               const __half* __restrict__ xh,
                                               __half* __restrict__ yraw,
                                               int M, int K, int N, int cta_tile) {
    // CTA tile: m16 x N128; 256 threads = 8 warps.
    __shared__ __half sa[16][16];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;      // this warp's trellis column
    const int KB = K >> 4;                     // k16 steps (one trellis block each)
    const int nb = N >> 4;

    // ---- hoisted per-lane slot constants (loop-invariant across the K loop).
    // slot = h*4 + s:  B-fragment (m16n8k16, .col): lane l holds
    //   k = 2*(l%4) + (s&1) + 8*(s>>1),  n = 8*h + (l>>2).
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            // tmap(r,c)
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    // Accumulators: two n8 tiles, fp32, fixed order.
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    // S-A3-f Item 3 rung (a-min): loop-carried ring pointer. The old body recomputed
    // `(size_t)kb * nb + nb_col) * (16 * BITS)` every k-step; the address advances by
    // a CONSTANT word stride per step, so carry one 64-bit pointer instead.
    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)0 * nb + nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;  // uint32 words per k16 step

    for (int kb = 0; kb < KB; ++kb) {
        // ---- stage A tile [16 x 16] (rows >= M zero) ----
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col]
                                : __ushort_as_half(0);
        }
        __syncthreads();

        // ---- decode this warp's 16x16 block into two B fragments ----
        // (ring carried across iterations — see rung (a-min) note above)

        uint32_t bfrag[2][2]; // [h][reg]: {s0,s1}, {s2,s3} as half2
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        ring += ring_kstride;  // rung (a-min): constant word stride per k16 step (after the h loop)

        // ---- A fragments (m16k16, row-major, manual from SMEM) ----
        // {a0,a1} = A[gid][2tig], {a2,a3} = A[gid+8][2tig], {a4,a5} = A[gid][2tig+8],
        // {a6,a7} = A[gid+8][2tig+8]; gid = lane>>2, tig = lane&3.
        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];

        // ---- 2x m16n8k16 ----
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    // ---- epilogue: store y_raw (fp16, RN), skip padded rows ----
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) {
            __half2 o = __floats2half2_rn(acc[h][0], acc[h][1]);
            *(__half2*)&yraw[(size_t)row0 * N + col] = o;
        }
        if (row1 < M) {
            __half2 o = __floats2half2_rn(acc[h][2], acc[h][3]);
            *(__half2*)&yraw[(size_t)row1 * N + col] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// S-A3-n F3: exl3_gemm_body with the A operand staged ONCE. The barrier body
// re-stages a 16x16 A tile every k16 step (LDG.U16 -> STS -> BAR -> ... -> BAR):
// two block barriers + a dependent global load on every step's critical path.
// Here the M live rows land in smem once (16-B vector loads, row stride K+8
// halves so the 8 fragment rows hit distinct banks) and the K loop runs with NO
// barriers. Everything else is exl3_gemm_body verbatim (N128 tile, warp map,
// ring loads, decode, mma order) => outputs bitwise identical (probe-exl3-binv
// grouped section). Dynamic smem = M*(K+8)*2 bytes (exl3_a1_fits).
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ bool exl3_a1_fits(int M, int K) {
    return M >= 1 && M <= 16 && (K & 7) == 0 && (long long)M * (K + 8) * 2 <= 48 * 1024;
}

#ifndef A1_SHFL
#define A1_SHFL 1
#endif
template <int BITS>
__device__ __forceinline__ void exl3_gemm_body_a1(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ xh,
                                                  __half* __restrict__ yraw,
                                                  int M, int K, int N, int cta_tile) {
    extern __shared__ __half a1_sa[];        // [M][K+8]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int SK = K + 8;
    {
        const int vpr = K >> 3;
        for (int i = tid; i < M * vpr; i += blockDim.x) {
            const int r = i / vpr, c = i - r * vpr;
            *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + c * 8];
        }
    }
    __syncthreads();

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;

    // S-A3-n F4: keep DRAM busy without registers — lanes 0..2 prefetch the
    // 16x16 block window (8*BITS words <= 160 B) PF_D steps ahead into L2
    // (one predicated instruction/step; no value change). The in-round expert
    // stream is DRAM-latency-bound at 24 warps/SM with one 96-B window in flight.
    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < BITS * 32;          // 3/4/5 lanes x 32 B
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;              // bytes per k16 step
    // S-A3-o O1: prologue — the first PF_D windows were never prefetched (every CTA
    // started with PF_D serial DRAM misses, all CTAs of a wave in lock-step). Issue
    // them up front; value-transparent (L2 hint only).
    if (pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    for (int kb = 0; kb < KB; ++kb) {
        if (pf_lane && kb + PF_D < KB)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
        pf_ptr += pf_step;
        uint32_t bfrag[2][2];
#if A1_SHFL
        // S-A3-n F7: ONE coalesced load per lane per step (window words
        // lane, lane+32 < 8*BITS), decode words delivered by __shfl_sync —
        // replaces 16 scalar L1 loads per lane per step. Same words => same bits.
        constexpr int WW = 8 * BITS;
        const unsigned w0 = (lane < WW) ? __ldg(ring + lane) : 0u;
        const unsigned w1 = (WW > 32 && lane + 32 < WW) ? __ldg(ring + lane + 32) : 0u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint16_t rv[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = h * 4 + q;
                unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                if (WW > 32) {
                    const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                    const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                    lo = s_i1m[j] >= 32 ? lo2 : lo;
                    hi = s_i0m[j] >= 32 ? hi2 : hi;
                }
                const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                            __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
            }
            bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
            bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
        }
#else
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
#endif
        ring += ring_kstride;
        const int col = kb * 16;
        const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }

    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
}

extern "C" __global__ void exl3_hmma_gemm(const uint16_t* __restrict__ trellis,
                                          const __half* __restrict__ xh,
                                          __half* __restrict__ yraw,
                                          int M, int K, int N, int bits) {
    XQ_PDL_ENTRY();
    switch (bits) {
        case 3: exl3_gemm_body<3>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 4: exl3_gemm_body<4>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 5: exl3_gemm_body<5>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        default: break; // loader refuses other rates before we ever get here
    }
}

// =============================================================================
// W3/LMH (2026-09-27): persistent-grid lm_head GEMM, optional fused output Hadamard.
//
// Target contract: sm_121 BASELINE (this module's arch). cp.async.cg 16B + L2::cache_hint
// (createpolicy evict_first), mma.sync.m16n8k16 f16->f32, dp4a, shf, fma.rn.f16x2 — all
// baseline CC 12.1 (no f/a suffix feature is used, none is needed).
//
// Shape class: the EXL3 lm_head at m <= 16 — K = 2560, N = 248,320 (verify / plain step /
// prefill tail) or the 65,536-id draft slice, 5-bit (3/4-bit instantiated for coverage).
// The old path (exl3_hmma_gemm = exl3_gemm_body: 1,940 CTAs x 256 threads, 4 CTAs/SM)
// re-stages the A tile every k16 step behind two block barriers and reads each warp's
// 160-B trellis window with 16 dependent scalar L1 loads per lane — one 160-B window in
// flight per warp, ~7.7 KB/SM: 79% of the 238 GB/s floor at m=6 (2.10 ms; ledger).
// This kernel (the reference implementation's exl3_gemm_kernel<5,..,16,16,512,4,3> byte schedule, minus its
// stream-K split, which would change the K-reduction order):
//   * grid = #SMs CTAs (1/SM, 8 warps). Tile = 128 columns = one col16 trellis block per
//     warp over the FULL K. Tiles interleave (t = blockIdx.x + i*gridDim.x) so the
//     co-running CTAs read one ~61 KB-contiguous trellis row span per k16 step.
//   * trellis streamed by cp.async.cg 16B (L1 bypass, L2 evict_first: 397 MB > L2, never
//     re-read; keeps the logits/xh lines) into an NST-slot smem ring of LMH_KSTEP k16 steps
//     (5,120 B/slot at 5-bit). NST-1 slots stay in flight ACROSS tile boundaries (no drain);
//     one wait_group + one __syncthreads per slot.
//   * A staged ONCE per CTA for m <= 8 ([8][K+8] halves: 16-B pad => the 8 fragment rows
//     hit distinct banks). m in 9..16 reads A fragments with ld.global.nc (L1-resident xh).
//   * decode from the smem window: 4 LDS.32 per lane per block (two ring words per 4-slot
//     quad, span 3*BITS+16 <= 31 bits) instead of 16 dependent scalar loads.
//   * fuse_had: the 128-col tile IS one Hadamard block -> exl3_had_svh's exact op sequence
//     runs in the epilogue (y_raw never round-trips DRAM; one launch fewer).
//
// BIT-IDENTITY CONTRACT (== exl3_hmma_gemm [+ exl3_had_svh] for every m in 1..16):
//   1. B fragments. exl3_gemm_body's slot (h, s) of lane l decodes tmap-derived
//      t = 8*l_tmap + (r&1) + 2*((r>>3)&1) + 4*(c>>3) + 32*(c&1), which reduces to
//      t = 8*lane + 4*h + s. Here the 16-bit window of t is cut from the two ring words
//      (I0, I2) spanning its quad h; idx equals exl3_dq_from's
//      funnelshift_r(ring[i1m], ring[i0m], sf) & 0xFFFF for every lane/slot/bits in {3,4,5}
//      (host proof: exl3_bench.rs test lmh_quad_extract_equals_dq_from — one-hot over every
//      ring bit + random rings). Then the SAME dp4a(idx*MUL1, 0x01010101, 0x6400) and fp16
//      fma(., 0x1EEE, 0xC931): hfma2 is two independent fma.rn.f16 — identical halves.
//   2. A fragments: rows < M = the same xh halves; rows >= M = 0 (predicated) — the old
//      body's zero-filled sa rows. Identical fragment registers.
//   3. Per output element: fp32 accumulators zero-initialized, one
//      mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 per (k16 step, n8 half), kb
//      STRICTLY ascending 0..KB-1 on ONE warp — never split across warps/CTAs (no split-K,
//      no stream-K). Staging (cp.async byte copies) is value-transparent.
//   4. Output: __floats2half2_rn with the old (row,col) mapping. fuse_had: the SAME fp16
//      y_raw values go through exl3_had128 + *svh + __float2half_rn (exl3_had_svh_dev).
//   => batch invariance (AGENTS §2.4) is inherited: per-row math never depends on M.
// Preconditions (host-checked, lmh_gemm_launch): M in 1..16 (MROWS=8 needs M <= 8),
// N % 128 == 0, K % (16*LMH_KSTEP) == 0, bits in {3,4,5}, smem <= the 99 KiB opt-in cap.
// =============================================================================
#define LMH_WARPS 8
#define LMH_THREADS (LMH_WARPS * 32)
#define LMH_KSTEP 4          // k16 steps per ring slot (5,120 B at 5-bit)
#define LMH_EP_LD 136        // fused-epilogue tile row stride (halves): 128 + 8 pad

__host__ __device__ __forceinline__ int lmh_smem_bytes(int bits, int mrows, int nst, int k) {
    return nst * LMH_KSTEP * LMH_WARPS * 32 * bits          // trellis ring
           + (mrows == 8 ? 8 * (k + 8) * 2 : 0)             // A-once rows (m <= 8)
           + mrows * LMH_EP_LD * 2;                         // fused-Hadamard epilogue tile
}

__device__ __forceinline__ void lmh_cp16_ef(unsigned dst, const void* src, unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void lmh_cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}

// One ring slot: LMH_KSTEP k16 rows x (8 blocks = the tile's 128 columns), 16-B chunks.
// Thread-invariant chunk geometry (c_src / c_dst) is hoisted by the caller.
template <int CPT>
__device__ __forceinline__ void lmh_issue(unsigned slot_u, const unsigned char* src_base,
                                          const long long (&c_src)[CPT], const int (&c_dst)[CPT],
                                          const bool (&c_live)[CPT], unsigned long long pol) {
    #pragma unroll
    for (int i = 0; i < CPT; ++i)
        if (c_live[i]) lmh_cp16_ef(slot_u + (unsigned)c_dst[i], src_base + c_src[i], pol);
}

template <int BITS, int MROWS, int NST>
__device__ __forceinline__ void exl3_lmh_body(const uint16_t* __restrict__ trellis,
                                              const __half* __restrict__ xh,
                                              __half* __restrict__ out,
                                              const __half* __restrict__ svh,
                                              int M, int K, int N, int fuse_had) {
    extern __shared__ __align__(16) unsigned char lmh_sm[];
    constexpr int BB   = 32 * BITS;                 // bytes per 16x16 trellis block
    constexpr int KROW = LMH_WARPS * BB;            // bytes per k16 row of one 128-col tile
    constexpr int STB  = LMH_KSTEP * KROW;          // bytes per ring slot
    constexpr int CPR  = KROW / 16;                 // 16-B chunks per k16 row
    constexpr int NCH  = STB / 16;                  // 16-B chunks per slot
    constexpr int CPT  = (NCH + LMH_THREADS - 1) / LMH_THREADS;
    constexpr int WW   = 8 * BITS;                  // ring words per block
    static_assert(NST >= 2, "ring needs >= 2 slots");
    static_assert(BB % 16 == 0 && KROW % 16 == 0, "16-B chunking");
    static_assert(MROWS == 8 || MROWS == 16, "MROWS");

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int T = N >> 7;
    const int NKS = (K >> 4) / LMH_KSTEP;           // ring slots per tile
    const int G = (int)gridDim.x;
    if ((int)blockIdx.x >= T) return;               // CTA-uniform (host sizes G <= T)
    const int ntiles = (T - (int)blockIdx.x + G - 1) / G;
    const int total = ntiles * NKS;                 // ring slots this CTA consumes
    const long long rowB = (long long)(N >> 4) * BB;  // bytes per k16 row of the trellis
    const int SK = K + 8;

    unsigned char* ring = lmh_sm;
    __half* sa = (__half*)(lmh_sm + NST * STB);
    __half* ep = (__half*)(lmh_sm + NST * STB + (MROWS == 8 ? 8 * SK * 2 : 0));
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);

    // Host/device layout tripwire (TRIPWIRE_SPEC §3.4 class): a launch with less dynamic smem
    // than this layout needs traps instead of silently overrunning shared memory.
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)lmh_smem_bytes(BITS, MROWS, NST, K)) __trap();
        if (M < 1 || M > MROWS) __trap();   // A-once holds 8 rows; epilogue tile MROWS rows
    }

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    // ---- per-thread slot chunk geometry: chunk c -> (k16 row g, byte r*16 in the row span) ----
    long long c_src[CPT];
    int c_dst[CPT];
    bool c_live[CPT];
    #pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = tid + i * LMH_THREADS;
        const int g = c / CPR, r = c - g * CPR;
        c_live[i] = c < NCH;
        c_src[i] = (long long)g * rowB + r * 16;
        c_dst[i] = g * KROW + r * 16;
    }

    // ---- prologue: A-once (m <= 8, group 0 with slot 0), then slots 0..NST-2 ----
    if (MROWS == 8) {
        const unsigned sa_u = (unsigned)__cvta_generic_to_shared(sa);
        const int vpr = K >> 3;
        for (int i = tid; i < M * vpr; i += LMH_THREADS) {
            const int r = i / vpr, c = i - r * vpr;
            lmh_cp16(sa_u + (unsigned)((r * SK + c * 8) * 2), xh + (size_t)r * K + c * 8);
        }
    }
    const unsigned char* trb = (const unsigned char*)trellis;
    int p_tile = (int)blockIdx.x, p_ks = 0, p_slot = 0, p_i = 0;
    #pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) {
        if (p_i < total) {
            lmh_issue<CPT>(ring_u + (unsigned)(p_slot * STB),
                           trb + (long long)p_ks * LMH_KSTEP * rowB + (long long)p_tile * KROW,
                           c_src, c_dst, c_live, pol);
            ++p_i;
            if (++p_ks == NKS) { p_ks = 0; p_tile += G; }
            if (++p_slot == NST) p_slot = 0;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // ---- per-lane decode constants: quad q (= n8 half h) holds t = 8*lane + 4q + s ----
    // word pair (qa, qb) spans the quad's 4 windows; per slot: shift (5 bits) | sel<<5
    // (sel: the window lies wholly in the EARLIER word qa -> funnel(qa, qa, sh-32)).
    int qa[2], qb[2];
    unsigned dsh[8];
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int t0 = 8 * lane + 4 * q;
        const int b0 = (t0 + 257) * BITS - 16;          // first window START
        const int b2 = (t0 + 3 + 257) * BITS;           // last window END
        const int I0 = b0 >> 5;
        const int I2 = (b2 - 1) >> 5;
        const int s2 = ((I2 + 1) << 5) - b2;
        qa[q] = I0 % WW;
        qb[q] = I2 % WW;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int sh = s2 + (3 - s) * BITS;         // 0..47
            dsh[q * 4 + s] = (unsigned)(sh & 31) | (sh >= 32 ? 32u : 0u);
        }
    }
    const __half2 k_inv2  = __half2half2(__ushort_as_half((unsigned short)0x1EEE));
    const __half2 k_bias2 = __half2half2(__ushort_as_half((unsigned short)0xC931));

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const bool live0 = gid < M;
    const bool live1 = (MROWS == 16) && (gid + 8 < M);
    int c_tile = (int)blockIdx.x, c_ks = 0, c_slot = 0;

    #pragma unroll 1
    for (int it = 0; it < total; ++it) {
        // slot `it` landed (<= NST-2 newer groups pending) and is visible CTA-wide; every warp
        // is past slot it-1, so its ring slot may be refilled.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        if (p_i < total) {
            lmh_issue<CPT>(ring_u + (unsigned)(p_slot * STB),
                           trb + (long long)p_ks * LMH_KSTEP * rowB + (long long)p_tile * KROW,
                           c_src, c_dst, c_live, pol);
            ++p_i;
            if (++p_ks == NKS) { p_ks = 0; p_tile += G; }
            if (++p_slot == NST) p_slot = 0;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");

        const unsigned char* slot = ring + c_slot * STB + warp * BB;
        #pragma unroll
        for (int g = 0; g < LMH_KSTEP; ++g) {
            const int kb = c_ks * LMH_KSTEP + g;
            const uint32_t* w = (const uint32_t*)(slot + g * KROW);
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const uint32_t wa = w[qa[q]];
                const uint32_t wb = w[qb[q]];
                uint32_t sum[4];
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const unsigned d = dsh[q * 4 + s];
                    const uint32_t lo = (d & 32u) ? wa : wb;
                    const unsigned idx = __funnelshift_r(lo, wa, d & 31u) & 0xFFFFu;
                    sum[s] = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                }
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const __half2 hv = __halves2half2(__ushort_as_half((unsigned short)sum[2 * p]),
                                                      __ushort_as_half((unsigned short)sum[2 * p + 1]));
                    const __half2 wv = __hfma2(hv, k_inv2, k_bias2);
                    bfrag[q][p] = *reinterpret_cast<const uint32_t*>(&wv);
                }
            }
            uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
            if (MROWS == 8) {
                if (live0) {
                    const __half* ar = sa + gid * SK + kb * 16 + 2 * tig;
                    a0 = *(const uint32_t*)ar;
                    a2 = *(const uint32_t*)(ar + 8);
                }
            } else {
                const __half* ar = xh + (size_t)gid * K + kb * 16 + 2 * tig;
                if (live0) {
                    a0 = __ldg((const unsigned int*)ar);
                    a2 = __ldg((const unsigned int*)(ar + 8));
                }
                if (live1) {
                    a1 = __ldg((const unsigned int*)(ar + (size_t)8 * K));
                    a3 = __ldg((const unsigned int*)(ar + (size_t)8 * K + 8));
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }
        if (++c_slot == NST) c_slot = 0;

        if (++c_ks == NKS) {
            // ---- tile epilogue (CTA-uniform) ----
            const int tile = c_tile;
            if (fuse_had) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = 16 * warp + 8 * h + 2 * tig;
                    if (live0) *(__half2*)&ep[gid * LMH_EP_LD + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
                    if (live1) *(__half2*)&ep[(gid + 8) * LMH_EP_LD + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
                }
                __syncthreads();
                // exl3_had_svh_dev's op sequence on the same fp16 y_raw values (one warp/row).
                for (int r = warp; r < M; r += LMH_WARPS) {
                    const __half* src = ep + r * LMH_EP_LD + lane * 4;
                    float v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) v[q] = __half2float(src[q]);
                    exl3_had128(v, lane);
                    const __half* sv = svh + (size_t)tile * 128 + lane * 4;
                    __half o[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) o[q] = __float2half_rn(v[q] * __half2float(sv[q]));
                    uint2 pk;
                    pk.x = (uint32_t)__half_as_ushort(o[0]) | ((uint32_t)__half_as_ushort(o[1]) << 16);
                    pk.y = (uint32_t)__half_as_ushort(o[2]) | ((uint32_t)__half_as_ushort(o[3]) << 16);
                    *(uint2*)&out[(size_t)r * N + (size_t)tile * 128 + lane * 4] = pk;
                }
                // ep is rewritten only at the NEXT tile end, >= 1 slot barrier later.
            } else {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const size_t col = (size_t)tile * 128 + 16 * warp + 8 * h + 2 * tig;
                    if (live0) *(__half2*)&out[(size_t)gid * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
                    if (live1) *(__half2*)&out[(size_t)(gid + 8) * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
            c_ks = 0;
            c_tile += G;
        }
    }
}

// Entries: exl3_lmh_b{BITS}_m{MROWS}_s{NST}. MROWS 8 = A-once smem (m <= 8), 16 = A via
// ld.global.nc (m in 9..16). NST = ring depth (5-bit: 4/6/8 selectable, GB10_LMH_NST).
#define LMH_ENTRY(B, MR, NS)                                                                  \
extern "C" __global__ void __launch_bounds__(LMH_THREADS, 1)                                  \
exl3_lmh_b##B##_m##MR##_s##NS(const uint16_t* __restrict__ trellis,                           \
                              const __half* __restrict__ xh, __half* __restrict__ out,        \
                              const __half* __restrict__ svh, int M, int K, int N,            \
                              int fuse_had) {                                                 \
    XQ_PDL_ENTRY();                                                                           \
    exl3_lmh_body<B, MR, NS>(trellis, xh, out, svh, M, K, N, fuse_had);                       \
}
LMH_ENTRY(5, 8, 4)
LMH_ENTRY(5, 8, 6)
LMH_ENTRY(5, 8, 8)
LMH_ENTRY(5, 16, 4)
LMH_ENTRY(5, 16, 6)
LMH_ENTRY(5, 16, 8)
LMH_ENTRY(4, 8, 6)
LMH_ENTRY(4, 16, 6)
LMH_ENTRY(3, 8, 6)
LMH_ENTRY(3, 16, 6)

// ---------------------------------------------------------------------------
// S-A3-f-b: WIDE-M GEMM (chunked prefill). Same operator, tile contract and
// fixed K-ascending reduction order as exl3_gemm_body, extended past the m16
// decode tile:
//   CTA = 256 threads (8 warps) covering (MT*16) x N128; warp w owns trellis
//   column stripe w for ALL MT m-tiles -> ONE 16x16 block decode per k16 step
//   amortizes over 2*MT HMMA ops (trellis bytes and decode ALU are M-flat).
//   grid = (ceil(M/(16*MT)), N/128): blockIdx.x = m-supertile, blockIdx.y =
//   n-tile -> co-resident CTAs share the same trellis stripes through L2.
//
// Decode addressing implements the f-a A/B#1 hoist (PLAN/S_A3_F_A_ATTRIBUTION
// §6 item 1): the per-lane ring-word addresses are formed ONCE before the K
// loop and advanced by the constant inter-k block stride inside it. Addresses
// are IDENTICAL to ring[i0m]/ring[i1m] of the m16 body (same words, same
// funnelshift argument order), so decode output is unchanged by construction;
// exl3_dq_at is the shared loader form (AGENTS 2.8: one decode function).
//
// Row-wise bitwise equivalence with the m16 kernel holds for every row: same
// B-fragment decode, same A fragment build, same mma sequence per output
// element. --probe-exl3-wide gates this empirically (bitwise vs the m16
// kernel and across MT at every tested M) before any serving use.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t exl3_dq_at(const uint32_t* w0,
                                               const uint32_t* w1, int sf) {
    // Same placement contract as exl3_dq_from: the LATER word (w1) is LOW in
    // the merged pair, the EARLIER word (w0) HIGH.
    // NO __restrict__ here: w0/w1 routinely alias (i0m == i1m in ~half the
    // windows, and the four states of a lane share ring words). Declaring them
    // restrict let ptxas hoist/pipeline loads across the k-advanced pointers
    // and produced all-wrong, stride-invariant output (caught by the wide gate,
    // 2026-09-20).
    const unsigned lo = *w1;
    const unsigned hi = *w0;
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);  // S-A3-n: fp16 bits
    const __half hv = __ushort_as_half((unsigned short)sum);                  // of 1024 + s (no I2F)
    return (uint16_t)__half_as_ushort(
        __hfma(hv, __ushort_as_half((unsigned short)0x1EEE),
                  __ushort_as_half((unsigned short)0xC931)));
}

#define EXL3_WIDE_KS 8
template <int BITS, int MT>
__device__ __forceinline__ void exl3_gemm_body_wide(
        const uint16_t* __restrict__ trellis,
        const __half* __restrict__ xh,
        __half* __restrict__ yraw,
        int M, int K, int N, int cta_tile, int m_sup, __half* __restrict__ sa_raw) {
    // S-A3-u F: A staged EXL3_WIDE_KS k16 steps at a time into the WRAPPER-owned tile
    // (one max-size copy per kernel; per-instantiation __shared__ summed across the 12 (BITS,MT)
    // instantiations): one barrier pair per KS steps instead of per step, 16-B vector loads.
    // Same values (rows >= M / cols >= K zero-filled exactly as before), same order => same bits.
    typedef __half SaRow[EXL3_WIDE_KS * 16 + 8];
    SaRow* sa = (SaRow*)sa_raw;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;      // this warp's trellis column
    const int KB = K >> 4;                     // k16 steps (one trellis block each)
    const int nb = N >> 4;
    const int row_base = m_sup * (MT * 16);    // first global row of this CTA

    // ---- hoisted per-lane slot constants (identical to exl3_gemm_body) ----
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    // ---- Accumulators: MT m-tiles x two n8 tiles, fp32, fixed order ----
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;

    for (int kb0 = 0; kb0 < KB; kb0 += EXL3_WIDE_KS) {
        // ---- stage A for KS k16 steps: [(MT*16) x (KS*16)] (rows >= M / cols >= K zero) ----
        {
            constexpr int C8 = EXL3_WIDE_KS * 2;               // uint4 chunks per staged row
            for (int i = tid; i < MT * 16 * C8; i += blockDim.x) {
                const int row = i / C8, c8 = i - row * C8;
                const int gr = row_base + row;
                const int col = kb0 * 16 + c8 * 8;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (gr < M && col < K) v = *(const uint4*)&xh[(size_t)gr * K + col];
                *(uint4*)&sa[row][c8 * 8] = v;
            }
        }
        __syncthreads();
        for (int s = 0; s < EXL3_WIDE_KS; ++s) {
        const int kb = kb0 + s;
        if (kb >= KB) break;

        // ---- decode this warp's 16x16 block ONCE into two B fragments ----
        const uint32_t* ring =
            (const uint32_t*)(trellis + ((size_t)kb * nb + nb_col) * (16 * BITS));
        uint32_t bfrag[2][2];
        if constexpr (BITS <= 4) {
            // S-A3-u C: the S-A3-n F7 LSU diet in the wide body — ONE coalesced load per lane of
            // the block's window words (8*BITS <= 32), decode words by __shfl_sync (was 16 scalar
            // L1 loads per lane per k16 step). Same words, same funnel/dp4a/hfma => same bits.
            constexpr int WW = 8 * BITS;
            const unsigned w0 = (lane < WW) ? __ldg(ring + lane) : 0u;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    const unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    const unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
        } else {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        }

        // ---- A fragments + 2*MT x m16n8k16 (one B decode, MT tiles of rows) ----
        const int gid = lane >> 2, tig = lane & 3;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const SaRow* sm = sa + mt * 16;
            const int cs = s * 16;
            const uint32_t a0 = *(const uint32_t*)&sm[gid][cs + 2 * tig];
            const uint32_t a1 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig];
            const uint32_t a2 = *(const uint32_t*)&sm[gid][cs + 2 * tig + 8];
            const uint32_t a3 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig + 8];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][h][0]), "+f"(acc[mt][h][1]),
                      "+f"(acc[mt][h][2]), "+f"(acc[mt][h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }
        }   // s
        __syncthreads();
    }

    // ---- epilogue: store y_raw (fp16, RN), skip rows >= M ----
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row0 = row_base + mt * 16 + gid;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
            if (row0 < M) {
                __half2 o = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
                *(__half2*)&yraw[(size_t)row0 * N + col] = o;
            }
            if (row1 < M) {
                __half2 o = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
                *(__half2*)&yraw[(size_t)row1 * N + col] = o;
            }
        }
    }
}

// S-A3-u D: DEVICE-side prefill MoE routing (was: sync + dtoh ids + host counting sort + ~10
// uploads per layer). Slot = expert id (identity map). count: cnt[e], exclusive offs_row[e],
// and the (expert, super-row) tile list for the grouped GEMM; place: one block per expert scans
// the ids in their original t-major order, so each expert's rows land in EXACTLY the host
// order (row_tok / row_eidx / cand identical to the host tables => bitwise-identical outputs).
extern "C" __global__ void __launch_bounds__(1024)
xq_moe_pf_count(const int* __restrict__ ids, int r, int ne, int* __restrict__ cnt,
                int* __restrict__ offs_row, int* __restrict__ tiles, int* __restrict__ ntiles, int mt16) {
    XQ_PDL_ENTRY();
    __shared__ int c_s[1024];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) c_s[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
        const int e = ids[i];
        if (e >= 0 && e < ne) atomicAdd(&c_s[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) cnt[e] = c_s[e];
    if (threadIdx.x == 0) {
        int off = 0, nt = 0;
        for (int e = 0; e < ne; ++e) {
            offs_row[e] = off;
            const int ce = c_s[e];
            for (int s = 0; s * mt16 < ce; ++s) { tiles[2 * nt] = e; tiles[2 * nt + 1] = s; ++nt; }
            off += ce;
        }
        offs_row[ne] = off;
        *ntiles = nt;
    }
}

extern "C" __global__ void __launch_bounds__(256)
xq_moe_pf_place(const int* __restrict__ ids, int r, int topk, const int* __restrict__ offs_row,
                int* __restrict__ row_tok, int* __restrict__ row_eidx, int* __restrict__ cand) {
    XQ_PDL_ENTRY();
    __shared__ int wsum[8];
    const int e = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int running = offs_row[e];
    for (int base = 0; base < r; base += 256) {
        const int i = base + threadIdx.x;
        const bool f = i < r && ids[i] == e;
        const unsigned m = __ballot_sync(0xffffffffu, f);
        const int wpre = __popc(m & ((1u << lane) - 1u));
        if (lane == 0) wsum[warp] = __popc(m);
        __syncthreads();
        int before = 0, tot = 0;
        for (int w = 0; w < 8; ++w) { if (w < warp) before += wsum[w]; tot += wsum[w]; }
        if (f) {
            const int dst = running + before + wpre;
            row_tok[dst] = i / topk;
            row_eidx[dst] = e;
            cand[i] = dst;
        }
        running += tot;
        __syncthreads();
    }
}

// S-A3-u D: exl3_hmma_gemm_wide_grouped driven by the device tile list: block (tile, n-tile);
// tile -> (expert, super-row). Same body, same (expert, n-tile, super-row) work => bitwise.
extern "C" __global__ void exl3_hmma_gemm_wide_tiles(const uint16_t* __restrict__ base,
                                                     const uint64_t* __restrict__ offs,
                                                     const __half* __restrict__ xh,
                                                     __half* __restrict__ y,
                                                     const int* __restrict__ tiles,
                                                     const int* __restrict__ ntiles,
                                                     const int* __restrict__ offs_row,
                                                     const int* __restrict__ cnt,
                                                     int K, int N, int bits, int mt) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) __half xq_wide_sa_buf[8 * 16][EXL3_WIDE_KS * 16 + 8];
    __half* xq_wide_sa = &xq_wide_sa_buf[0][0];
    // grid (N/128, max_tiles): the n-tile varies FASTEST, so all column tiles of one
    // (expert, super-row) run together and stream that expert's contiguous trellis (DRAM
    // locality, as the grouped grid's expert-major order had); tile-major was 1.46x slower.
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int e = tiles[2 * tile], m_sup = tiles[2 * tile + 1];
    const int ce = cnt[e];
    const int row0 = offs_row[e];
    const uint16_t* tr = base + (size_t)offs[e];
    const __half* xe = xh + (size_t)row0 * K;
    __half* ye = y + (size_t)row0 * N;
    const int cta_tile = (int)blockIdx.x;
#define EXL3_WT_CASE(B) case B: switch (mt) { \
        case 1: exl3_gemm_body_wide<B, 1>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 2: exl3_gemm_body_wide<B, 2>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 4: exl3_gemm_body_wide<B, 4>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 8: exl3_gemm_body_wide<B, 8>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        default: break; } break;
    switch (bits) { EXL3_WT_CASE(3) EXL3_WT_CASE(4) EXL3_WT_CASE(5) default: break; }
#undef EXL3_WT_CASE
}

// S-A3-q P1: the prefill MoE per-expert wide launches as ONE grouped launch per side.
// grid (msup, N/128, esel): blockIdx.z = compact expert slot; each block runs the UNCHANGED
// exl3_gemm_body_wide on that expert's compact row block (rows offs_row[z] .. +cnt[z]) —
// the same (expert, tile, super-row) work and instruction sequence as the old per-expert
// exl3_hmma_gemm_wide launch => bitwise identical. Slots with cnt >= rmin (rmin > 0) belong
// to the reconstruct+Lt arm and exit; cnt == 0 slots exit.
extern "C" __global__ void exl3_hmma_gemm_wide_grouped(const uint16_t* __restrict__ base,
                                                       const uint64_t* __restrict__ offs,
                                                       const __half* __restrict__ xh,
                                                       __half* __restrict__ y,
                                                       const int* __restrict__ offs_row,
                                                       const int* __restrict__ cnt,
                                                       int K, int N, int bits, int mt, int rmin) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) __half xq_wide_sa_buf[8 * 16][EXL3_WIDE_KS * 16 + 8];
    __half* xq_wide_sa = &xq_wide_sa_buf[0][0];
    const int z = blockIdx.z;
    const int ce = cnt[z];
    if (ce == 0 || (rmin > 0 && ce >= rmin)) return;
    const int m_sup = (int)blockIdx.x;
    if (m_sup * mt * 16 >= ce) return;
    const int row0 = offs_row[z];
    const uint16_t* tr = base + (size_t)offs[z];
    const __half* xe = xh + (size_t)row0 * K;
    __half* ye = y + (size_t)row0 * N;
    const int cta_tile = (int)blockIdx.y;
#define EXL3_WG_CASE(B) case B: switch (mt) { \
        case 1: exl3_gemm_body_wide<B, 1>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 2: exl3_gemm_body_wide<B, 2>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 4: exl3_gemm_body_wide<B, 4>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 8: exl3_gemm_body_wide<B, 8>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        default: break; } break;
    switch (bits) { EXL3_WG_CASE(3) EXL3_WG_CASE(4) EXL3_WG_CASE(5) default: break; }
#undef EXL3_WG_CASE
}

extern "C" __global__ void exl3_hmma_gemm_wide(const uint16_t* __restrict__ trellis,
                                               const __half* __restrict__ xh,
                                               __half* __restrict__ yraw,
                                               int M, int K, int N, int bits, int mt) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) __half xq_wide_sa_buf[8 * 16][EXL3_WIDE_KS * 16 + 8];
    __half* xq_wide_sa = &xq_wide_sa_buf[0][0];
    // S-A3-f-b param-echo bisect: mt == -1 dumps the received arguments into
    // yraw (3 pointers as u64 + M/K/N/bits/mt as i32) so the host can compare
    // against what it passed. Diagnostics only.
    if (mt == -1) {
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            long long* e = (long long*)yraw;
            e[0] = (long long)(size_t)trellis;
            e[1] = (long long)(size_t)xh;
            e[2] = (long long)(size_t)yraw;
            int* s = (int*)(e + 3);
            s[0] = M; s[1] = K; s[2] = N; s[3] = bits; s[4] = mt;
        }
        return;
    }
    const int cta_tile = (int)blockIdx.y;
    const int m_sup = (int)blockIdx.x;
#define EXL3_WIDE_CASE(B)                                                     \
    case B:                                                                   \
        switch (mt) {                                                         \
            case 1: exl3_gemm_body_wide<B, 1>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, xq_wide_sa); break; \
            case 2: exl3_gemm_body_wide<B, 2>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, xq_wide_sa); break; \
            case 4: exl3_gemm_body_wide<B, 4>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, xq_wide_sa); break; \
            case 8: exl3_gemm_body_wide<B, 8>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, xq_wide_sa); break; \
            default: break;                                                   \
        }                                                                     \
        break
    switch (bits) {
        EXL3_WIDE_CASE(3);
        EXL3_WIDE_CASE(4);
        EXL3_WIDE_CASE(5);
        default: break; // loader refuses other rates before we ever get here
    }
#undef EXL3_WIDE_CASE
}

// ---------------------------------------------------------------------------
// Kernel 5: grouped-expert GEMM — grid = E * (N/128) CTAs, CTA -> (expert, tile).
// The serving-required MoE form AND the small-N occupancy fix: N=640 experts go
// from 5 CTAs to E*5 (512*5 = 2560 CTAs at the real layer width). Same body, same
// fixed reduction order per output element (batch invariance is per-expert-column).
// ptrs: E device trellis pointers; y: [E, M, N].
// ---------------------------------------------------------------------------
extern "C" __global__ void exl3_hmma_gemm_grouped(const uint16_t* __restrict__ base,
                                                  const uint64_t* __restrict__ offs,
                                                  const __half* __restrict__ xh,
                                                  __half* __restrict__ y,
                                                  int M, int K, int N, int bits) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    const uint16_t* tr = base + (size_t)offs[expert]; // offs = ELEMENT offsets per expert
    __half* ye = y + (size_t)expert * (size_t)M * (size_t)N;
    // S-A3-n F3: A staged once when it fits (the launcher passes M*(K+8)*2 dyn smem).
    if (exl3_a1_fits(M, K)) {
        switch (bits) {
            case 3: exl3_gemm_body_a1<3>(tr, xh, ye, M, K, N, tile); break;
            case 4: exl3_gemm_body_a1<4>(tr, xh, ye, M, K, N, tile); break;
            case 5: exl3_gemm_body_a1<5>(tr, xh, ye, M, K, N, tile); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 3: exl3_gemm_body<3>(tr, xh, ye, M, K, N, tile); break;
        case 4: exl3_gemm_body<4>(tr, xh, ye, M, K, N, tile); break;
        case 5: exl3_gemm_body<5>(tr, xh, ye, M, K, N, tile); break;
        default: break;
    }
}

// =============================================================================
// S-A3-d — whole-model offline forward (probe-only, --bench-exl3-forward).
// Glue kernels port the ENGINE's qwen4_exp math (gpu_batch.cu: hc_norm_b,
// silu_div_b, hc_mix_b, hc_inject_b, conv1d_b, delta_step_b/gdn_token,
// rmsnorm_gated_sig_b) with fp16 io + fp32 math. Engine-pinned reduction orders
// (halving-tree norms with 1e-6, ascending k loops, two-pass softmax) kept.
// Residual streams fp32 (the reference capture dtype); GEMM io fp16.
// New kernels only — the S-A3-c kernels above are UNCHANGED (gates stay valid).
// NOTE exl3_had128 normalizes orthonormally INTERNALLY (1/sqrt(128)); the multi
// variants below therefore add no extra scale.
// =============================================================================

__device__ __forceinline__ float xq_h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half xq_f2h(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float xq_silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float xq_sig(float v) { return 1.0f / (1.0f + __expf(-v)); }

// ---- plain fp16 GEMM: out[m,n] = sum_k w[n,k]*x[m,k], fp32 accum, k ascending.
// grid (ceil(N/128), M), block 128; one thread per column.
extern "C" __global__ void xq_gemm_f16(__half* __restrict__ out,
                                       const __half* __restrict__ w,
                                       const __half* __restrict__ x,
                                       int M, int N, int K) {
    XQ_PDL_ENTRY();
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const __half* wr = w + (long long)n * K;
    const __half* xr = x + (long long)m * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += xq_h2f(wr[k]) * xq_h2f(xr[k]);
    out[(long long)m * N + n] = xq_f2h(acc);
}

// ---- S-A3-m: BIT-EXACT row-batched twin of xq_gemm_f16 (the PLE key/value
// projections, 52 + 13 MB fp16). xq_gemm_f16 runs a grid-y block PER ROW, so
// every verify row re-streamed the whole weight (S-A3-m budget: 2.4 ms at m=6,
// ~75 GB/s). Here one thread per output column reads its weight row ONCE (two
// 16-B loads per 16 k = full 32-B sectors) and applies it to all M rows; each
// row's accumulation is xq_gemm_f16's exact sequence (ascending k, same
// `acc += w*x` expression) => per-row bitwise identical. x loads are warp-uniform
// (broadcast). K % 16 == 0; M <= MR.
template <int MR>
__device__ __forceinline__ void xq_gemm_f16_rows_body(__half* __restrict__ out,
                                                      const __half* __restrict__ w,
                                                      const __half* __restrict__ x,
                                                      int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const uint4* wr = (const uint4*)(w + (long long)n * K);
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;
    const int s16 = K >> 4;
    for (int s = 0; s < s16; s++) {
        const uint4 wv0 = __ldg(wr + 2 * s);
        const uint4 wv1 = __ldg(wr + 2 * s + 1);
        const __half* wh0 = (const __half*)&wv0;
        const __half* wh1 = (const __half*)&wv1;
        #pragma unroll
        for (int r = 0; r < MR; r++) {
            if (r < M) {
                const uint4* xr = (const uint4*)(x + (long long)r * K) + 2 * s;
                const uint4 xv0 = __ldg(xr), xv1 = __ldg(xr + 1);
                const __half* xh0 = (const __half*)&xv0;
                const __half* xh1 = (const __half*)&xv1;
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh0[j]) * xq_h2f(xh0[j]);
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh1[j]) * xq_h2f(xh1[j]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < MR; r++)
        if (r < M) out[(long long)r * N + n] = xq_f2h(acc[r]);
}

// grid (ceil(N/128)), block 128. M in 1..16.
extern "C" __global__ void xq_gemm_f16_rows(__half* __restrict__ out,
                                            const __half* __restrict__ w,
                                            const __half* __restrict__ x,
                                            int M, int N, int K) {
    XQ_PDL_ENTRY();
    if (M <= 4)       xq_gemm_f16_rows_body<4>(out, w, x, M, N, K);
    else if (M <= 8)  xq_gemm_f16_rows_body<8>(out, w, x, M, N, K);
    else              xq_gemm_f16_rows_body<16>(out, w, x, M, N, K);
}

// ---- TP-4T1: BIT-EXACT small-M twin of xq_gemm_f16 for the prefill TAIL chunks (c < recon_min_rows, M <= 16:
// hc down/up, a_w/b_w, the PLE projections). xq_gemm_f16 is ONE THREAD PER COLUMN with 2-byte loads at a
// lane stride of K*2 bytes (20 KB for hc down): every warp load touches 32 L1 lines, ~665 us for hc down
// at ANY M (L1 wavefront bound, 20x its FMA-chain floor); xq_gemm_f16_rows is thread-per-column too (10 warps
// at N=320 -> 2.2 ms at M=11). Here the weight tile [32 cols][XQ_TG_KC k] and the x tile [XQ_TG_RB rows][KC]
// are staged through shared memory with coalesced 16-B cp.async (XQ_TG_ST-stage pipeline, one __syncthreads per
// tile), the weight tile is read back with conflict-free LDS.128 (16-B row pad), and warp r / lane c of the
// block computes output (row m0 + r, column n0 + c). EXACTNESS: per output the arithmetic is xq_gemm_f16's
// verbatim — acc = 0.0f, ascending k, the same `acc += xq_h2f(w) * xq_h2f(x)` expression (same contracted FFMA),
// the same final __float2half_rn; no split-K, no shuffle tree, no atomics, no tensor core => every output is
// BITWISE equal to xq_gemm_f16 (probe: --probe-exl3-tailgemm). grid (ceil(N/32), ceil(M/4)), block 128.
// K % 8 == 0 (16-B cp.async; the served K are 10240/320/2560), any M (rows beyond M are clamped on load, skipped
// on store). Weight rows >= N clamp to row N-1 on load (the a/b N = 48 second tile is half empty).
#define XQ_TG_RB 4
#define XQ_TG_KC 128
#define XQ_TG_ST 4
__device__ __forceinline__ void xq_tg_cp16(void* smem_dst, const void* gsrc) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(gsrc) : "memory");
}
extern "C" __global__ void __launch_bounds__(128)
xq_gemm_f16_tail(__half* __restrict__ out, const __half* __restrict__ w, const __half* __restrict__ x,
                 int M, int N, int K) {
    XQ_PDL_ENTRY();
    constexpr int KC = XQ_TG_KC, ST = XQ_TG_ST, RB = XQ_TG_RB;
    constexpr int PITCH = KC + 8;              // halves; 16-B pad => conflict-free LDS.128 across a quarter-warp
    constexpr int CH = KC / 8;                 // 16-B chunks per row per stage
    __shared__ __align__(16) __half wS[ST][32][PITCH];
    __shared__ __align__(16) __half xS[ST][RB][KC];
    const int tid = threadIdx.x, lane = tid & 31, r = tid >> 5;
    const int n0 = blockIdx.x * 32, m0 = blockIdx.y * RB;
    const int nt = (K + KC - 1) / KC;
    auto load_tile = [&](int tile, int buf) {
        const int k0 = tile * KC;
        const int kc = min(KC, K - k0);
        #pragma unroll
        for (int i = 0; i < (32 * CH) / 128; i++) {
            const int idx = tid + i * 128, row = idx / CH, ch = idx % CH;
            if (ch * 8 < kc) {
                const int nn = min(n0 + row, N - 1);
                xq_tg_cp16(&wS[buf][row][ch * 8], w + (size_t)nn * K + k0 + ch * 8);
            }
        }
        if (tid < RB * CH) {
            const int row = tid / CH, ch = tid % CH;
            if (ch * 8 < kc) {
                const int mm = min(m0 + row, M - 1);
                xq_tg_cp16(&xS[buf][row][ch * 8], x + (size_t)mm * K + k0 + ch * 8);
            }
        }
    };
    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nt) load_tile(s, s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    float acc = 0.0f;
    const bool live = (m0 + r) < M;
    for (int t = 0; t < nt; t++) {
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 2) : "memory");
        __syncthreads();
        if (t + ST - 1 < nt) load_tile(t + ST - 1, (t + ST - 1) % ST);
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (live) {
            const int buf = t % ST;
            const int kc = min(KC, K - t * KC);
            const uint4* wp = (const uint4*)&wS[buf][lane][0];
            const uint4* xp = (const uint4*)&xS[buf][r][0];
            for (int j = 0; j < kc / 8; j++) {
                const uint4 wv = wp[j], xv = xp[j];
                const __half* wh = (const __half*)&wv;
                const __half* xh = (const __half*)&xv;
                #pragma unroll
                for (int q = 0; q < 8; q++) acc += xq_h2f(wh[q]) * xq_h2f(xh[q]);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const int n = n0 + lane, m = m0 + r;
    if (live && n < N) out[(size_t)m * N + n] = xq_f2h(acc);
}

// ---- WP09: K-MAJOR twin of xq_gemm_f16_rows (the PLE key/value projections, verify + decode).
// The weights are relayouted ONCE at load to [K/16][N][16] f16: element (n, k) lives at
// ((k>>4)*N + n)*16 + (k&15). Thread n's 16-k step s is then ONE 32-B load at (s*N + n)*32 B,
// so a warp's step is a single contiguous 1-KB burst — the row-major body read 32 B from each of
// 32 rows 5 KB apart per step (74 GB/s measured, 0.886 ms/verify vs a 0.275 ms floor).
// EXACTNESS: per (row r, column n) the arithmetic is xq_gemm_f16_rows' sequence verbatim —
// acc = 0.0f, ascending s, wv0 = k 16s..16s+7 then wv1 = 16s+8..16s+15, the same
// `acc[r] += xq_h2f(w) * xq_h2f(x)` expression — so every output is BITWISE equal to
// xq_gemm_f16_rows / xq_gemm_f16 (binv probe: EXL3-F16ROWS-KM). Only the ADDRESS of each weight
// half changes. Weight loads: 256-bit ld.global.nc (assembles for plain sm_121 — baseline
// target, no f/a feature) with L1::no_allocate (single use; x stays L1-resident) and
// L2::evict_first (65.5 MB > L2, never re-read within a round). Groups of XQ_KM_U steps issue
// their weight loads back-to-back before the FFMA phase. x rows: the same
// warp-uniform __ldg broadcasts as the row-major body. Two matrices per launch (the key and value
// projections share x): blocks [0, nba) -> (out_a, wa, Na), blocks [nba, grid) -> (out_b, wb, Nb).
// K % 16 == 0; M <= 16; weight base 32-B aligned (cudaMalloc).
#define XQ_KM_U 4
__device__ __forceinline__ void xq_ld256_stream(const uint4* p, uint4& lo, uint4& hi) {
    asm("ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w), "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
        : "l"(p));
}

template <int MR>
__device__ __forceinline__ void xq_km_step(float (&acc)[MR], const uint4& wv0, const uint4& wv1,
                                           const __half* __restrict__ x, int M, int K, int s) {
    const __half* wh0 = (const __half*)&wv0;
    const __half* wh1 = (const __half*)&wv1;
    #pragma unroll
    for (int r = 0; r < MR; r++) {
        if (r < M) {
            const uint4* xr = (const uint4*)(x + (long long)r * K) + 2 * s;
            const uint4 xv0 = __ldg(xr), xv1 = __ldg(xr + 1);
            const __half* xh0 = (const __half*)&xv0;
            const __half* xh1 = (const __half*)&xv1;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh0[j]) * xq_h2f(xh0[j]);
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh1[j]) * xq_h2f(xh1[j]);
        }
    }
}

template <int MR>
__device__ __forceinline__ void xq_gemm_f16_rows_km_body(__half* __restrict__ out,
                                                         const uint4* __restrict__ w,
                                                         const __half* __restrict__ x,
                                                         int M, int N, int K, int n) {
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;
    const int s16 = K >> 4;
    const uint4* wp = w + 2 * (long long)n;        // step s of column n: wp + s*st (uint4 units)
    const long long st = 2 * (long long)N;
    const int sfull = s16 - s16 % XQ_KM_U;
    // XQ_KM_U 32-B loads issued back-to-back per group (12,800 threads x 128 B in flight at the
    // group heads ~ 2.7x the bandwidth-latency product); a register double buffer measured 190
    // regs at ptxas (2 blocks/SM -> the 100-block grid would need a second wave) — single
    // buffer = 118 regs, the whole grid co-resident.
    #pragma unroll 1
    for (int s = 0; s < sfull; s += XQ_KM_U) {
        uint4 c0[XQ_KM_U], c1[XQ_KM_U];
        #pragma unroll
        for (int u = 0; u < XQ_KM_U; u++) xq_ld256_stream(wp + (s + u) * st, c0[u], c1[u]);
        #pragma unroll
        for (int u = 0; u < XQ_KM_U; u++) xq_km_step<MR>(acc, c0[u], c1[u], x, M, K, s + u);
    }
    for (int s = sfull; s < s16; s++) {            // tail (K/16 % XQ_KM_U != 0; not the PLE shapes)
        uint4 w0, w1;
        xq_ld256_stream(wp + s * st, w0, w1);
        xq_km_step<MR>(acc, w0, w1, x, M, K, s);
    }
    #pragma unroll
    for (int r = 0; r < MR; r++)
        if (r < M) out[(long long)r * N + n] = xq_f2h(acc[r]);
}

// grid (nba + ceil(Nb/128)), block 128 (nba = ceil(Na/128)). M in 1..16. Nb = 0 -> one matrix.
extern "C" __global__ void xq_gemm_f16_rows_km(__half* __restrict__ out_a, const uint4* __restrict__ wa,
                                               int Na, int nba,
                                               __half* __restrict__ out_b, const uint4* __restrict__ wb,
                                               int Nb, const __half* __restrict__ x, int M, int K) {
    XQ_PDL_ENTRY();
    const bool second = (int)blockIdx.x >= nba;
    const int N = second ? Nb : Na;
    const int n = (second ? (int)blockIdx.x - nba : (int)blockIdx.x) * (int)blockDim.x + (int)threadIdx.x;
    if (n >= N) return;
    __half* out = second ? out_b : out_a;
    const uint4* w = second ? wb : wa;
    if (M <= 4)       xq_gemm_f16_rows_km_body<4>(out, w, x, M, N, K, n);
    else if (M <= 8)  xq_gemm_f16_rows_km_body<8>(out, w, x, M, N, K, n);
    else              xq_gemm_f16_rows_km_body<16>(out, w, x, M, N, K, n);
}

// fp32-out variant (the router: AGENTS §7 — selection in fp32).
extern "C" __global__ void xq_gemm_f16_f32(float* __restrict__ out,
                                           const __half* __restrict__ w,
                                           const __half* __restrict__ x,
                                           int M, int N, int K) {
    XQ_PDL_ENTRY();
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const __half* wr = w + (long long)n * K;
    const __half* xr = x + (long long)m * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += xq_h2f(wr[k]) * xq_h2f(xr[k]);
    out[(long long)m * N + n] = acc;
}

// S-A3-q P3: row-batched BIT-EXACT twin of xq_gemm_f16_f32 for the prefill router: thread
// per column n, 16 rows per block (grid (ceil(N/128), ceil(M/16))) — the 2.6 MB router weight
// is read once per 16 tokens instead of once per token. Per (m, n) the accumulation is the
// same ascending-k chain acc += w[n][k] * x[m][k] => identical bits (router selections safe).
extern "C" __global__ void xq_gemm_f16_f32_rows(float* __restrict__ out,
                                                const __half* __restrict__ w,
                                                const __half* __restrict__ x,
                                                int M, int N, int K) {
    XQ_PDL_ENTRY();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m0 = blockIdx.y * 16;
    if (n >= N) return;
    const __half* wr = w + (long long)n * K;
    float acc[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (int k = 0; k < K; k++) {
        const float wv = xq_h2f(wr[k]);
        #pragma unroll
        for (int r = 0; r < 16; ++r)
            if (m0 + r < M) acc[r] += wv * xq_h2f(x[(long long)(m0 + r) * K + k]);
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r)
        if (m0 + r < M) out[(long long)(m0 + r) * N + n] = acc[r];
}

// S-A3-u B: xq_gemm_f16_f32_rows with 32 rows per block and the x tile staged in shared memory
// (KT columns at a time) — the weight row is read once per 32 tokens and the per-k x reads come
// from smem broadcasts instead of 16 L1 loads. Per (m, n) the accumulation is still the single
// ascending-k chain acc += w[n][k] * x[m][k] => bitwise equal to xq_gemm_f16_f32.
#define XQ_R32_KT 256
#define XQ_R32_P 36          // transposed x-tile pitch: 32 rows + pad, 16-B aligned rows
// S-A3-u B2: x tile stored TRANSPOSED (xs[k][row]) so each k's 32 row values are 8 broadcast
// LDS.128; the weight row is read 8 halves at a time. Per (m, n): acc += w[k] * x[k], k ascending.
extern "C" __global__ void __launch_bounds__(128)
xq_gemm_f16_f32_rows32(float* __restrict__ out, const __half* __restrict__ w,
                       const __half* __restrict__ x, int M, int N, int K) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float xs[XQ_R32_KT][XQ_R32_P];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m0 = blockIdx.y * 32;
    const int rows = min(32, M - m0);
    const __half* wr = w + (long long)(n < N ? n : 0) * K;
    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) acc[r] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += XQ_R32_KT) {
        const int kt = min(XQ_R32_KT, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * XQ_R32_KT; i += blockDim.x) {
            const int r = i / XQ_R32_KT, kk = i - r * XQ_R32_KT;     // coalesced along k
            xs[kk][r] = (r < rows && kk < kt) ? xq_h2f(x[(long long)(m0 + r) * K + k0 + kk]) : 0.0f;
        }
        __syncthreads();
        if (n < N) {
            for (int kk = 0; kk < kt; kk += 8) {
                const uint4 w8 = *(const uint4*)(wr + k0 + kk);      // K % 8 == 0 (2560)
                const __half* wh = (const __half*)&w8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float wv = xq_h2f(wh[j]);
                    const float4* xr = (const float4*)&xs[kk + j][0];
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const float4 v = xr[q];
                        acc[4 * q + 0] += wv * v.x; acc[4 * q + 1] += wv * v.y;
                        acc[4 * q + 2] += wv * v.z; acc[4 * q + 3] += wv * v.w;
                    }
                }
            }
        }
    }
    if (n < N) {
        #pragma unroll
        for (int r = 0; r < 32; ++r)
            if (r < rows) out[(long long)(m0 + r) * N + n] = acc[r];
    }
}

// TP-4R1: REGISTER-TILED order-preserving prefill router GEMM, out[m][n] = sum_k w[n][k] * x[m][k] in fp32.
// WHY: rows32 gives each thread ONE column x 32 rows, so every FFMA needs one smem word (x broadcast): 256 FFMA : 64
// LDS.128 per 8 k = LSU/smem-operand bound at ~25% of the FFMA peak (7.4 TFLOP/s measured in-trace). Here a thread
// owns an (8 x 8 | 8 x 4) tile built from groups of 4 (LDS.128 fragments, interleaved group offsets so a quarter-warp
// reads 8 consecutive 16-B chunks = conflict-free): operand words per FFMA (TM+TN)/(TM*TN) = 0.25 | 0.375.
// ORDER: every output (m, n) is still ONE register chain acc = 0.0f; acc += w[n][k] * x[m][k], k = 0..K-1 ascending,
// fp16 -> fp32 exact conversions (an fp16*fp16 product is exact in fp32, so FFMA contraction cannot change a bit).
// No split-K, no tensor cores, no shuffles, no atomics: bitwise == xq_gemm_f16_f32 == rows32 == rows8.
// Staging: BK = 16 k per tile; each work item = (row-group of 4 rows, k-quad): 4 x LDG.64 (one 32-B sector per row per
// 4 lanes) -> 4x4 register transpose -> 4 x STS.128 into fp32 k-major tiles As[k][m] / Bs[k][n]; tile i+1 is loaded
// into registers while tile i is computed, smem is double-buffered, ONE __syncthreads per tile.
// Contract: N % BN == 0, K % 16 == 0, M >= 1 (x rows past M are clamped to row M-1, their outputs are never stored).
template <int BM, int BN, int GM, int GN>
__device__ __forceinline__ void xq_rtile_body(float* __restrict__ out, const __half* __restrict__ w,
                                              const __half* __restrict__ x, int M, int N, int K) {
    constexpr int BK = 16;
    constexpr int TM = 4 * GM, TN = 4 * GN;
    constexpr int TY = BM / TM, TX = BN / TN, T = TX * TY;
    constexpr int ROWS = BM + BN;
    constexpr int NQ = BK / 4;                        // k-quads per row per tile
    constexpr int RGN = ROWS / 4;                     // row groups (x first, then w)
    static_assert(ROWS % 4 == 0 && RGN % 8 == 0, "row groups multiple of 8");
    constexpr int ITEMS = RGN * NQ;
    constexpr int IPT = (ITEMS + T - 1) / T;          // staging items per thread
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];
    const int t = threadIdx.x;
    const int tx = t % TX, ty = t / TX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const __half* base[IPT];
    int kq4[IPT], rgl[IPT]; bool isx[IPT], act[IPT];
    #pragma unroll
    for (int j = 0; j < IPT; ++j) {
        const int i = t + j * T;
        act[j] = i < ITEMS;
        const int rg = (i / (8 * NQ)) * 8 + (i % 8);
        kq4[j] = ((i / 8) % NQ) * 4;
        isx[j] = (4 * rg) < BM;
        rgl[j] = isx[j] ? rg : rg - BM / 4;
        const int r0 = isx[j] ? (m0 + 4 * rgl[j]) : (n0 + 4 * rgl[j]);
        base[j] = (isx[j] ? x : w) + (long long)(act[j] ? r0 : 0) * K + kq4[j];
    }
    const int lim = M - 1;
    uint2 pre[IPT][4];
    auto gload = [&](int k0) {
        #pragma unroll
        for (int j = 0; j < IPT; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                long long roff = (long long)r * K;
                if (isx[j]) { int m = m0 + 4 * rgl[j] + r; m = m < lim ? m : lim; roff = (long long)(m - (m0 + 4 * rgl[j])) * K; }
                pre[j][r] = __ldg(reinterpret_cast<const uint2*>(base[j] + roff + k0));
            }
    };
    auto sstore = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < IPT; ++j) {
            if (!act[j]) continue;
            float f[4][4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __half2* h2 = reinterpret_cast<const __half2*>(&pre[j][r]);
                const float2 a = __half22float2(h2[0]);
                const float2 b = __half22float2(h2[1]);
                f[r][0] = a.x; f[r][1] = a.y; f[r][2] = b.x; f[r][3] = b.y;
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float* dst = isx[j] ? &As[buf][kq4[j] + e][4 * rgl[j]] : &Bs[buf][kq4[j] + e][4 * rgl[j]];
                *reinterpret_cast<float4*>(dst) = make_float4(f[0][e], f[1][e], f[2][e], f[3][e]);
            }
        }
    };
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    gload(0);
    sstore(0);
    __syncthreads();
    const int nt = K / BK;
    for (int it = 0; it < nt; ++it) {
        const int cur = it & 1;
        if (it + 1 < nt) gload((it + 1) * BK);
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int g = 0; g < GM; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&As[cur][k][g * (BM / GM) + ty * 4]);
                a[4 * g] = v.x; a[4 * g + 1] = v.y; a[4 * g + 2] = v.z; a[4 * g + 3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < GN; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[cur][k][g * (BN / GN) + tx * 4]);
                b[4 * g] = v.x; b[4 * g + 1] = v.y; b[4 * g + 2] = v.z; b[4 * g + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += b[j] * a[i];
        }
        if (it + 1 < nt) { sstore(cur ^ 1); __syncthreads(); }
    }
    #pragma unroll
    for (int gm = 0; gm < GM; ++gm)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = m0 + gm * (BM / GM) + ty * 4 + i;
            if (m < M)
                #pragma unroll
                for (int gn = 0; gn < GN; ++gn) {
                    const int n = n0 + gn * (BN / GN) + tx * 4;
                    *reinterpret_cast<float4*>(out + (long long)m * N + n) =
                        make_float4(acc[4 * gm + i][4 * gn], acc[4 * gm + i][4 * gn + 1], acc[4 * gm + i][4 * gn + 2], acc[4 * gm + i][4 * gn + 3]);
                }
        }
}
// 128 x 64 outputs per CTA, 8 x 8 per thread (grid (N/64, ceil(M/128)), block 128): chunks of >= ~1280 rows.
extern "C" __global__ void __launch_bounds__(128)
xq_gemm_f16_f32_tile128(float* __restrict__ out, const __half* __restrict__ w, const __half* __restrict__ x,
                        int M, int N, int K) {
    XQ_PDL_ENTRY();
    xq_rtile_body<128, 64, 2, 2>(out, w, x, M, N, K);
}
// 64 x 64 outputs per CTA, 8 x 4 per thread (grid (N/64, ceil(M/64)), block 128): mid chunks (more CTAs per SM wave).
extern "C" __global__ void __launch_bounds__(128)
xq_gemm_f16_f32_tile64(float* __restrict__ out, const __half* __restrict__ w, const __half* __restrict__ x,
                       int M, int N, int K) {
    XQ_PDL_ENTRY();
    xq_rtile_body<64, 64, 2, 1>(out, w, x, M, N, K);
}

// PFX1 (d): the prefill router for SMALL chunks — a BIT-EXACT twin of xq_gemm_f16_f32 /
// xq_gemm_f16_f32_rows32 sized for grids rows32 cannot fill. At c = 72 rows32 is 12 CTAs x 4
// warps (<= 1 warp per SM) walking K = 2560 with one unpipelined LDG.128 per 8 k: ~210-230
// us/layer, ~10-11 ms per short prefill against a ~0.5 ms weight-read floor (WP00 B3).
// Contract (GB10 sm_121 baseline — FP32 FFMA + LDG/LDS only, no Tensor Core: the router bits
// decide top-k selections, AGENTS §7): CTA = 64 threads = 64 columns n x R = 8 rows m0..m0+7,
// grid (N/64, ceil(M/8)); per (m, n) the SAME single ascending-k chain acc += w[n][k] * x[m][k]
// as the reference (contracted FFMA, fma(w, x, acc)) => identical bits at every M.
//  - w: each thread streams its own row as 2 x LDG.128 (32 B = one full sector) per 16-k step,
//    register-prefetched in groups of XQ_R8_PF = 4 steps (two statically indexed register
//    buffers ping-pong), plus a prefetch.global.L2 run-ahead XQ_R8_L2A = 16 steps (2 passes)
//    ahead so the register loads hit L2.
//  - x: 8 rows x XQ_R8_KT k per tile, loaded as uint4 (coalesced 64-B row segments, prefetched
//    into registers one tile ahead), converted once, stored TRANSPOSED as f32 xs[k][row] so each
//    k's 8 row values are 2 broadcast LDS.128. The 8-float pad per 8-k group makes the staging
//    stores conflict-free (bank = 8*q + r over the warp's 4 q x 8 r lanes).
// Requires K % XQ_R8_KT == 0 and 16-B aligned rows (host-checked; K = 2560 here).
#define XQ_R8_ROWS 8
#define XQ_R8_COLS 64
#define XQ_R8_KT 256
#define XQ_R8_PF 4
#define XQ_R8_L2A 16                       // L2 prefetch run-ahead, 16-k steps (a multiple of 8)
#define XQ_R8_G (8 * XQ_R8_ROWS + 8)       // floats per 8-k group (64 values + 8 pad)
static_assert((XQ_R8_KT / 16) % (2 * XQ_R8_PF) == 0, "rows8: a K tile must hold whole wA/wB group pairs");
static_assert((XQ_R8_KT / 8) * XQ_R8_ROWS % XQ_R8_COLS == 0, "rows8: x tile uint4s split evenly over the CTA");
// w group load: 16-k steps s .. s+PF-1 of this thread's row (clamped to the last step: the
// clamped duplicates past the end are never consumed).
__device__ __forceinline__ void xq_r8_wload(uint4 (&dst)[2 * XQ_R8_PF], const uint4* __restrict__ wr,
                                            int s, int nsteps) {
    #pragma unroll
    for (int p = 0; p < XQ_R8_PF; ++p) {
        const int sp = min(s + p, nsteps - 1);
        dst[2 * p] = __ldg(wr + 2 * sp);
        dst[2 * p + 1] = __ldg(wr + 2 * sp + 1);
    }
}
// One 16-k step: c0 = w[k .. k+7] against xs group g0, c1 = w[k+8 .. k+15] against g0 + G.
// Per row r: acc[r] += w[k] * x[r][k], k ascending — the reference chain (contracted FFMA).
__device__ __forceinline__ void xq_r8_step(float (&acc)[XQ_R8_ROWS], const uint4 c0, const uint4 c1,
                                           const float* g0) {
    const __half* h0 = reinterpret_cast<const __half*>(&c0);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float wv = xq_h2f(h0[e]);
        const float4 xa = *reinterpret_cast<const float4*>(g0 + e * XQ_R8_ROWS);
        const float4 xb = *reinterpret_cast<const float4*>(g0 + e * XQ_R8_ROWS + 4);
        acc[0] += wv * xa.x; acc[1] += wv * xa.y; acc[2] += wv * xa.z; acc[3] += wv * xa.w;
        acc[4] += wv * xb.x; acc[5] += wv * xb.y; acc[6] += wv * xb.z; acc[7] += wv * xb.w;
    }
    const float* g1 = g0 + XQ_R8_G;
    const __half* h1 = reinterpret_cast<const __half*>(&c1);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float wv = xq_h2f(h1[e]);
        const float4 xa = *reinterpret_cast<const float4*>(g1 + e * XQ_R8_ROWS);
        const float4 xb = *reinterpret_cast<const float4*>(g1 + e * XQ_R8_ROWS + 4);
        acc[0] += wv * xa.x; acc[1] += wv * xa.y; acc[2] += wv * xa.z; acc[3] += wv * xa.w;
        acc[4] += wv * xb.x; acc[5] += wv * xb.y; acc[6] += wv * xb.z; acc[7] += wv * xb.w;
    }
}
extern "C" __global__ void __launch_bounds__(XQ_R8_COLS)
xq_gemm_f16_f32_rows8(float* __restrict__ out, const __half* __restrict__ w,
                      const __half* __restrict__ x, int M, int N, int K) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float xs[(XQ_R8_KT / 8) * XQ_R8_G];
    const int tid = threadIdx.x;
    const int n = blockIdx.x * XQ_R8_COLS + tid;
    const int m0 = blockIdx.y * XQ_R8_ROWS;
    const int rows = min(XQ_R8_ROWS, M - m0);
    const uint4* wr = reinterpret_cast<const uint4*>(w + (long long)(n < N ? n : 0) * K);
    const int nsteps = K >> 4;                   // 16-k steps (2 uint4 of w each)
    // x staging lanes: uint4 j of this thread = row (tid & 7), 8-k group q = (tid >> 3) + 8 j.
    const int xr = tid & 7;
    const int xq0 = tid >> 3;
    const __half* xrow = x + (long long)(m0 + (xr < rows ? xr : 0)) * K;
    constexpr int XJ = (XQ_R8_KT / 8) * XQ_R8_ROWS / XQ_R8_COLS;   // uint4 of x per thread/tile
    uint4 xp[XJ];
    #pragma unroll
    for (int j = 0; j < XJ; ++j)
        xp[j] = (xr < rows) ? __ldg(reinterpret_cast<const uint4*>(xrow + (xq0 + 8 * j) * 8))
                            : make_uint4(0u, 0u, 0u, 0u);
    uint4 wA[2 * XQ_R8_PF], wB[2 * XQ_R8_PF];    // two PF-step groups of w, ping-pong
    xq_r8_wload(wA, wr, 0, nsteps);
    // L2 run-ahead: ptxas schedules the register loads only ~1-2 steps before their use, so the
    // row's 128-B lines are also prefetched into L2 XQ_R8_L2A steps (two passes) ahead of the
    // register loads — the DRAM latency is paid there, the LDGs hit L2.
    const char* wbytes = reinterpret_cast<const char*>(wr);
    #pragma unroll
    for (int s = 2 * XQ_R8_PF; s < 2 * XQ_R8_PF + XQ_R8_L2A; s += 4)   // 4 steps = one 128-B line
        if (s < nsteps) asm volatile("prefetch.global.L2 [%0];" :: "l"(wbytes + (size_t)s * 32));
    float acc[XQ_R8_ROWS];
    #pragma unroll
    for (int r = 0; r < XQ_R8_ROWS; ++r) acc[r] = 0.0f;
    int gs = 0;                                  // global 16-k step at wA's first slot
    for (int k0 = 0; k0 < K; k0 += XQ_R8_KT) {
        __syncthreads();                         // the previous tile's readers are done
        #pragma unroll
        for (int j = 0; j < XJ; ++j) {
            const __half* hv = reinterpret_cast<const __half*>(&xp[j]);
            float* g = xs + (xq0 + 8 * j) * XQ_R8_G + xr;
            #pragma unroll
            for (int e = 0; e < 8; ++e) g[e * XQ_R8_ROWS] = xq_h2f(hv[e]);
        }
        __syncthreads();
        if (k0 + XQ_R8_KT < K) {                 // next tile's x in flight during this tile
            #pragma unroll
            for (int j = 0; j < XJ; ++j)
                xp[j] = (xr < rows)
                    ? __ldg(reinterpret_cast<const uint4*>(xrow + k0 + XQ_R8_KT + (xq0 + 8 * j) * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
        }
        // Two PF-step groups per pass: group g computes from wA while group g+1 streams into wB,
        // then g+1 computes from wB while g+2 streams into wA — in program order each load leads
        // its first use by a whole group, into registers dead at that point (96 regs, 0 spill;
        // the old single ring needed old+new live and sat at 125). ptxas's SASS schedule still
        // places each LDG ~1-2 steps (~100-300 instr) before its use (cuobjdump-checked), so the
        // L2 run-ahead above/below is what takes the DRAM latency off the chain.
        for (int s0 = 0; s0 < XQ_R8_KT / 16; s0 += 2 * XQ_R8_PF) {
            {
                const int sp = gs + 2 * XQ_R8_PF + XQ_R8_L2A;          // this pass + 3 (2 lines)
                if (sp < nsteps) asm volatile("prefetch.global.L2 [%0];" :: "l"(wbytes + (size_t)sp * 32));
                if (sp + 4 < nsteps) asm volatile("prefetch.global.L2 [%0];" :: "l"(wbytes + (size_t)(sp + 4) * 32));
            }
            xq_r8_wload(wB, wr, gs + XQ_R8_PF, nsteps);
            #pragma unroll
            for (int p = 0; p < XQ_R8_PF; ++p)
                xq_r8_step(acc, wA[2 * p], wA[2 * p + 1], xs + (2 * (s0 + p)) * XQ_R8_G);
            xq_r8_wload(wA, wr, gs + 2 * XQ_R8_PF, nsteps);
            #pragma unroll
            for (int p = 0; p < XQ_R8_PF; ++p)
                xq_r8_step(acc, wB[2 * p], wB[2 * p + 1], xs + (2 * (s0 + XQ_R8_PF + p)) * XQ_R8_G);
            gs += 2 * XQ_R8_PF;
        }
    }
    if (n < N) {
        #pragma unroll
        for (int r = 0; r < XQ_R8_ROWS; ++r)
            if (r < rows) out[(long long)(m0 + r) * N + n] = acc[r];
    }
}

// ---- grouped rmsnorm over the hc streams: out_s = rmsnorm(x_s)*(1+w_s), fp16 out.
// grid hc, block 1024; mirrors hc_norm_b (fp32 sums, halving tree).
extern "C" __global__ void xq_hc_norm(__half* __restrict__ out, const float* __restrict__ x,
                                      const __half* __restrict__ w, int h, int hc, int B, float eps) {
    XQ_PDL_ENTRY();
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
        // S-A3-e FIX (latent at B>1): the output row MUST carry the lane term. The old
        // `out[s*h+i]` made every lane's blocks collide on lane 0's rows — empirically
        // proven on-device (B=4 probe: rows b>0 never written); width sweep was blind.
        out[(long long)blk * h + i] = xq_f2h(xs[i] * inv * (1.0f + xq_h2f(w[(long long)s * h + i])));
}

// x = silu(x/div) in place (fp16).
extern "C" __global__ void xq_silu_div(__half* x, float div, int total) {
    XQ_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] = xq_f2h(xq_silu(xq_h2f(x[i]) / div));
}

// ---- hc mix + inject gates, one block per token: mirrors hc_mix_b.
//   x[i]   = (1/hc) * sum_s sigmoid(u[s*h+i]) * hn[s*h+i]      (fp16 out, f32 math)
//   inj[s] = 2 * sigmoid( (sum_c winj[s,c]*hn[c]) / hc )       (fp32 scalar per stream)
extern "C" __global__ void xq_hc_mix(__half* __restrict__ x, float* __restrict__ inj,
                                     const __half* __restrict__ hn, const __half* __restrict__ u,
                                     const __half* __restrict__ winj, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    int b = blockIdx.x;
    if (b >= B) return;
    hn += (long long)b * h * hc;
    u += (long long)b * h * hc;
    x += (long long)b * h;
    inj += (long long)b * hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    for (int i = tid; i < h; i += bs) {
        float acc = 0.0f;
        for (int s = 0; s < hc; s++)
            acc += xq_sig(xq_h2f(u[(long long)s * h + i])) * xq_h2f(hn[(long long)s * h + i]);
        x[i] = xq_f2h(acc / (float)hc);
    }
    if (winj != nullptr) {
        for (int s = 0; s < hc; s++) {
            float part = 0.0f;
            for (int c = tid; c < h * hc; c += bs) part += xq_h2f(winj[(long long)s * h * hc + c]) * xq_h2f(hn[c]);
            sm[tid] = part;
            __syncthreads();
            for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
            if (tid == 0) inj[s] = 2.0f * xq_sig(sm[0] / (float)hc);
            __syncthreads();
        }
    }
}

// ---- 2026-09-26: xq_hc_mix with the hc (<= 4) inject reductions run in ONE shared tree pass.
// xq_hc_mix reduced the streams one after another (4 x 10 __syncthreads + 4 barriers, one block
// per token => one SM at decode): ~20 us per call x 97 calls per step. Here each stream keeps
// its own partials (same strided c-loop), its own smem row and the SAME halving-tree pairing —
// per-stream arithmetic is identical, so inj (and x) are bit-identical — but the tree steps are
// shared across streams. Dynamic smem = 4 * blockDim.x floats.
extern "C" __global__ void xq_hc_mix4(__half* __restrict__ x, float* __restrict__ inj,
                                      const __half* __restrict__ hn, const __half* __restrict__ u,
                                      const __half* __restrict__ winj, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    int b = blockIdx.x;
    if (b >= B) return;
    hn += (long long)b * h * hc;
    u += (long long)b * h * hc;
    x += (long long)b * h;
    inj += (long long)b * hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    for (int i = tid; i < h; i += bs) {
        float acc = 0.0f;
        for (int s = 0; s < hc; s++)
            acc += xq_sig(xq_h2f(u[(long long)s * h + i])) * xq_h2f(hn[(long long)s * h + i]);
        x[i] = xq_f2h(acc / (float)hc);
    }
    if (winj != nullptr) {
        for (int s = 0; s < hc; s++) {
            float part = 0.0f;
            for (int c = tid; c < h * hc; c += bs) part += xq_h2f(winj[(long long)s * h * hc + c]) * xq_h2f(hn[c]);
            sm[s * bs + tid] = part;
        }
        __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2)
                for (int s = 0; s < hc; s++) sm[s * bs + tid] += sm[s * bs + tid + s2];
            __syncthreads();
        }
        if (tid < hc) inj[tid] = 2.0f * xq_sig(sm[tid * bs] / (float)hc);
    }
}

// ---- WP11 R2 (2026-09-26): xq_hc_mix4 REGRIDDED, bit-identical. xq_hc_mix4 ran m blocks of
// 1024 (m SMs busy at decode/verify): each block did its row's elementwise x AND the hc inject
// trees behind serial strided-load chains (~14 us per call at m=6 against ~1 us of bytes).
// Here grid (ninj + nxb, B), block 1024 (MUST stay 1024: the tree leaves are the 1024 per-thread
// partials of the old kernels), no cross-block state (the reference implementation's hc_mix.cu shape: independent
// blocks, each re-deriving what it needs):
//   blockIdx.x <  ninj : stream s = blockIdx.x's inject tree, ALONE in its block. Same per-thread
//                        partial (c = tid + j*1024, ascending j, part += w*hn — one fma chain),
//                        with every load hoisted ahead of the chain (one latency round, not 10);
//                        the SAME halving pairing: smem levels 512..32, then levels 16..1 by
//                        shfl_down inside warp 0 (lane t gets v[t] + v[t+s2] exactly as
//                        sm[t] += sm[t+s2]; lanes >= s2 are never read again).
//   blockIdx.x >= ninj : 1024 elements of x each; per element the identical ascending-s chain.
// ninj = hc when winj != null (0 for the final mixer). Dynamic smem 0 (static 4 KB).
extern "C" __global__ void __launch_bounds__(1024)
xq_hc_mix_rg(__half* __restrict__ x, float* __restrict__ inj,
             const __half* __restrict__ hn, const __half* __restrict__ u,
             const __half* __restrict__ winj, int h, int hc, int B) {
    XQ_PDL_ENTRY();
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
        const __half* wr = winj + (long long)s * n;
        float part = 0.0f;
        constexpr int J = 12;                     // n <= 12 * 1024 covers the served 4 x 2560
        if (n <= J * bs) {
            float wv[J], hv[J];
            #pragma unroll
            for (int j = 0; j < J; ++j) {
                const int c = tid + j * bs;
                wv[j] = c < n ? xq_h2f(wr[c]) : 0.0f;
                hv[j] = c < n ? xq_h2f(hn[c]) : 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < J; ++j)
                if (tid + j * bs < n) part += wv[j] * hv[j];
        } else {
            for (int c = tid; c < n; c += bs) part += xq_h2f(wr[c]) * xq_h2f(hn[c]);
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
            if (tid == 0) inj[s] = 2.0f * xq_sig(v / (float)hc);
        }
    } else {
        const int i = ((int)blockIdx.x - ninj) * bs + tid;
        if (i < h) {
            float acc = 0.0f;
            for (int s = 0; s < hc; s++)
                acc += xq_sig(xq_h2f(u[(long long)s * h + i])) * xq_h2f(hn[(long long)s * h + i]);
            x[i] = xq_f2h(acc / (float)hc);
        }
    }
}

// resid[b][s*h+i] += inj[b][s] * out[b][i]  (fp32 resid, fp16 out, fp32 scalar per-stream inj).
// S-A3-e FIX (latent at B>1): the old kernel had no B parameter and bounds-checked against
// ONE lane's extent (h*hc) — lanes 1..m-1 never received any layer inject.
extern "C" __global__ void xq_hc_inject(float* __restrict__ resid, const __half* __restrict__ out,
                                        const float* __restrict__ inj, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)h * hc * B) return;
    int b = (int)(i / ((long long)h * hc));
    int r = (int)(i - (long long)b * h * hc);
    resid[i] += inj[b * hc + r / h] * xq_h2f(out[(long long)b * h + r % h]);
}

// ---- MTP draft-head glue (S-A3-e): streams = f32(fc_hidden_row) + f32(fc_embedding_row).
// Exact float add of two halves — matches their out_dtype=torch.float stream entry.
extern "C" __global__ void xq_add_f16_f32(float* __restrict__ out, const __half* __restrict__ a,
                                          const __half* __restrict__ b, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = xq_h2f(a[i]) + xq_h2f(b[i]);
}

// ---- S-A3-e poison: fill base[0..n) with a u16 pattern (NaN-poison stale-row probe).
extern "C" __global__ void xq_memset_u16(unsigned short* __restrict__ base, long long n, int val) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    base[i] = (unsigned short)val;
}

// ---- S-A3-e: zero base[start .. start+n) (per-slot conv/GDN state reset on admit).
extern "C" __global__ void xq_memset_f32(float* __restrict__ base, long long start, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    base[start + i] = 0.0f;
}

// ---- GDN conv1d decode step: mirrors conv1d_b (f16 io, f32 state, silu at end).
extern "C" __global__ void xq_conv1d(__half* __restrict__ x, float* __restrict__ state,
                                     const float* __restrict__ w, int conv_dim, int k, int B,
                                     const int* __restrict__ slot_ids, int row_stride) {
    XQ_PDL_ENTRY();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * conv_dim;
    if (idx >= total) return;
    int b = idx / conv_dim;
    int c = idx % conv_dim;
    // S-A3-e FIX: slotpos is INTERLEAVED [slot,pos] pairs - slot_ids[b] read the lane's
    // POSITION as the state row (latent at W>1; invisible offline where slot==row==small pos).
    int slot = slot_ids[b * 2];
    float* st = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 1; j < k; j++) st[j - 1] = st[j];
    st[k - 1] = xq_h2f(x[(long long)b * row_stride + c]);
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
    x[(long long)b * row_stride + c] = xq_f2h(xq_silu(acc));
}

// ---- GDN gated-delta-rule step (port of delta_step_b + gdn_token):
// qkv_conv [B][qkv_stride]: q at 0, k at key_dim, v at 2*key_dim + head*vd.
// state [B][nh][kd][vd] f32. beta = sigmoid(b); gt = exp(-exp(A_log)*softplus(a+dt_bias)).
// grid (B*nh*vd/32), block kd. Engine reduction orders kept.
#define XQ_GDN_C 32
#define XQ_GDN_SP (XQ_GDN_C + 1)
// 2026-09-26: coalesced state copy (8 lanes x float4 per 128-B row segment) replacing the
// thread-per-row scalar loop (a warp touched 32 rows = 32 segments per element step). The
// VALUES placed in S_sh / S are identical — only the access pattern changes (bitwise).
__device__ __forceinline__ void xq_gdn_ld_state(float* S_sh, const float* S, int kd, int vd, int bb0) {
    const int n4 = kd * (XQ_GDN_C / 4);
    for (int e = threadIdx.x; e < n4; e += blockDim.x) {
        const int r = e / (XQ_GDN_C / 4), c4 = (e % (XQ_GDN_C / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(S + (long long)r * vd + bb0 + c4);
        float* d = S_sh + r * XQ_GDN_SP + c4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
}
__device__ __forceinline__ void xq_gdn_st_state(float* S, const float* S_sh, int kd, int vd, int bb0) {
    const int n4 = kd * (XQ_GDN_C / 4);
    for (int e = threadIdx.x; e < n4; e += blockDim.x) {
        const int r = e / (XQ_GDN_C / 4), c4 = (e % (XQ_GDN_C / 4)) * 4;
        const float* d = S_sh + r * XQ_GDN_SP + c4;
        *reinterpret_cast<float4*>(S + (long long)r * vd + bb0 + c4) = make_float4(d[0], d[1], d[2], d[3]);
    }
}

__device__ __forceinline__ void xq_gdn_token(float* S_sh, int kd, int bb0,
                                             const __half* q_in, const __half* k_in, const __half* v_in,
                                             __half* coreb, int key_head, float beta, float gt,
                                             float* Srow, float* kv_mem, float* vbuf, float* delta,
                                             float* qrow, float* krow) {
    const int a = threadIdx.x;
    float qv = xq_h2f(q_in[key_head * kd + a]);
    float kv = xq_h2f(k_in[key_head * kd + a]);
    Srow[a] = qv * qv; __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float qn = rsqrtf(Srow[0] + 1e-6f); __syncthreads();
    qv *= qn;
    Srow[a] = kv * kv; __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float kn = rsqrtf(Srow[0] + 1e-6f); __syncthreads();
    kv *= kn;
    float scale = 1.0f / sqrtf((float)kd);
    qv *= scale; qrow[a] = qv; krow[a] = kv; __syncthreads();
    #pragma unroll
    for (int c = 0; c < XQ_GDN_C; c++) S_sh[a * XQ_GDN_SP + c] *= gt;
    __syncthreads();
    if (a < XQ_GDN_C) {
        float km = 0.0f;
        for (int aa = 0; aa < kd; aa++) km += S_sh[aa * XQ_GDN_SP + a] * krow[aa];
        kv_mem[a] = km;
        vbuf[a] = xq_h2f(v_in[bb0 + a]);
    }
    __syncthreads();
    if (a < XQ_GDN_C) delta[a] = (vbuf[a] - kv_mem[a]) * beta;
    __syncthreads();
    const float kk = krow[a];
    #pragma unroll
    for (int c = 0; c < XQ_GDN_C; c++) S_sh[a * XQ_GDN_SP + c] += kk * delta[c];
    __syncthreads();
    if (a < XQ_GDN_C) {
        float o = 0.0f;
        for (int aa = 0; aa < kd; aa++) o += S_sh[aa * XQ_GDN_SP + a] * qrow[aa];
        coreb[bb0 + a] = xq_f2h(o);
    }
    __syncthreads();
}

extern "C" __global__ void xq_gdn_step(__half* __restrict__ core, const __half* __restrict__ qkv,
                                       float* __restrict__ state, const __half* __restrict__ b_in,
                                       const __half* __restrict__ a_in, int nh, int n_k_heads,
                                       int kd, int vd, const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias,
                                       const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    int blk = blockIdx.x;
    int nchunk = vd / XQ_GDN_C;
    const int qkv_stride = 2 * n_k_heads * kd + nh * vd;  // conv_dim (arity limit: derive in-kernel)
    int chunk = blk % nchunk; blk /= nchunk;
    int head = blk % nh; blk /= nh;
    int b = blk;
    // S-A3-e FIX: the recurrent state is PER-SLOT (a request keeps its slot for its lifetime);
    // the batch row b is the launch position. Offline b==slot always; serving recycles lanes.
    const int slot = slot_ids[b * 2];
    int key_head = head * n_k_heads / nh;
    int key_dim = n_k_heads * kd;
    int bb0 = chunk * XQ_GDN_C;
    extern __shared__ float sh[];
    float* S_sh = sh;
    float* Srow = S_sh + kd * XQ_GDN_SP;
    float* kv_mem = Srow + kd;
    float* vbuf = kv_mem + XQ_GDN_C;
    float* delta = vbuf + XQ_GDN_C;
    float* qrow = delta + XQ_GDN_C;
    float* krow = qrow + kd;
    const __half* col = qkv + (long long)b * qkv_stride;
    float* S = state + ((long long)slot * nh + head) * kd * vd;
    float beta = 1.0f / (1.0f + __expf(-xq_h2f(b_in[(long long)b * nh + head])));
    float sp = xq_h2f(a_in[(long long)b * nh + head]) + dt_bias[head];
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    float gt = __expf(-__expf(a_log[head]) * sp);
    xq_gdn_ld_state(S_sh, S, kd, vd, bb0);
    __syncthreads();
    xq_gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                 core + (long long)b * (nh * vd) + (long long)head * vd,
                 key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// ---- GDN gated output norm (port of rmsnorm_gated_sig_b): per (b, head):
// out = x * rsqrt(mean(x^2)+eps) * w[i] * sigmoid(z). grid B*nh, block vd.
extern "C" __global__ void xq_gdn_gate(__half* __restrict__ out, const __half* __restrict__ x,
                                       const __half* __restrict__ z, const __half* __restrict__ w,
                                       int vd, int nh, int B, float eps) {
    XQ_PDL_ENTRY();
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? xq_h2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) {
        float g = xq_h2f(z[(long long)b * (nh * vd) + (long long)head * vd + tid]);
        out[base + tid] = xq_f2h(v * inv * xq_h2f(w[tid]) * xq_sig(g));
    }
}

// ---- MoE router: logits [B][ne] fp32 -> softmax -> top-k (ties -> lowest id) -> renorm.
// smem floats: [0,bs) reduce scratch, [bs, bs+ne) probs, [bs+ne, bs+ne+2*bs) candidate scratch.
extern "C" __global__ void xq_router_topk(int* __restrict__ ids, float* __restrict__ wts,
                                          const float* __restrict__ logits, int ne, int k, int B) {
    XQ_PDL_ENTRY();
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float* prob = sm + bs;
    int* cand_i = (int*)(sm + bs + ne);
    const float* lg = logits + (long long)b * ne;
    float mx = -INFINITY;
    for (int i = tid; i < ne; i += bs) mx = fmaxf(mx, lg[i]);
    sm[tid] = mx; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] = fmaxf(sm[tid], sm[tid + s2]); __syncthreads(); }
    float gmax = sm[0];
    __syncthreads();  // S-A3-e FIX (expert-confirmed): all threads must read gmax before sm is reused.
    float part = 0.0f;
    for (int i = tid; i < ne; i += bs) part += __expf(lg[i] - gmax);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float denom = sm[0];
    for (int i = tid; i < ne; i += bs) prob[i] = __expf(lg[i] - gmax) / denom;
    __syncthreads();
    float* cand_v = sm;  // reuse the reduce scratch for candidate values
    for (int j = 0; j < k; j++) {
        float bv = -2.0f; int bi = ne;
        for (int i = tid; i < ne; i += bs)
            if (prob[i] > bv || (prob[i] == bv && i < bi)) { bv = prob[i]; bi = i; }
        cand_v[tid] = bv; cand_i[tid] = bi;
        __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) {
                float v0 = cand_v[tid], v1 = cand_v[tid + s2];
                int i0 = cand_i[tid], i1 = cand_i[tid + s2];
                if (v1 > v0 || (v1 == v0 && i1 < i0)) { cand_v[tid] = v1; cand_i[tid] = i1; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            // NaN/Inf logits leave every prob comparison false and bi at its
            // `ne` sentinel; clamp so host counting stays in-bounds (S-A3-f-b).
            if (cand_i[0] < 0 || cand_i[0] >= ne) cand_i[0] = 0;
            ids[(long long)b * k + j] = cand_i[0];
            wts[(long long)b * k + j] = cand_v[0];
            prob[cand_i[0]] = -2.0f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += wts[(long long)b * k + j];
        for (int j = 0; j < k; j++) wts[(long long)b * k + j] /= sum;
    }
}

// ---- MoE gate*up: din[em,d] = silu(gate)*up; y_gu row em = [gate(640) | up(640)] interleaved.
// S-A3-f Item 2: grid padded to the topk cap; row640 = m*mi rows live.
extern "C" __global__ void xq_moe_gate_mul(__half* __restrict__ din, const __half* __restrict__ y_gu,
                                           long long row640, const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)(*esel) * row640) return;
    long long em = i / 640;          // which (slot, m) row
    long long d = i - em * 640;      // column within the 640-wide half
    float g = xq_h2f(y_gu[em * 1280 + d]);
    float u = xq_h2f(y_gu[em * 1280 + 640 + d]);
    din[i] = xq_f2h(xq_silu(g) * u);
}

// ---- MoE combine: out[m,i] = sig(sg.x)*y_sh[m,i] + sum_j wts[m,j]*y_d[slot(m,j)][m,i].
// slotmap [ne] i32: expert id -> slot in the grouped buffers this step, -1 = absent.
extern "C" __global__ void xq_moe_combine(__half* __restrict__ out, const __half* __restrict__ y_d,
                                          const __half* __restrict__ y_sh,
                                          const int* __restrict__ ids, const float* __restrict__ wts,
                                          const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                          const __half* __restrict__ x, int k, int h, int M) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float part = 0.0f;
    for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float sgv = xq_sig(sm[0]);
    for (int i = tid; i < h; i += bs) {
        float acc = sgv * xq_h2f(y_sh[(long long)m * h + i]);
        for (int j = 0; j < k; j++) {
            int e = ids[(long long)m * k + j];
            int slot = slotmap[e];
            if (slot >= 0) acc += wts[(long long)m * k + j] * xq_h2f(y_d[((long long)slot * M + m) * h + i]);
        }
        out[(long long)m * h + i] = xq_f2h(acc);
    }
}

// ---- greedy argmax over vocab (fp32 compare, lowest-id ties). One block.
extern "C" __global__ void xq_argmax(int* __restrict__ out_ids, int lane,
                                     const __half* __restrict__ logits, long long start, int V) {
    XQ_PDL_ENTRY();
    extern __shared__ float sm[];
    long long* si = (long long*)(sm + blockDim.x);
    int tid = threadIdx.x, bs = blockDim.x;
    float best = -INFINITY; long long bi = 0;
    for (int i = tid; i < V; i += bs) {
        float v = xq_h2f(logits[start + i]);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sm[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            float v0 = sm[tid], v1 = sm[tid + s2];
            long long i0 = si[tid], i1 = si[tid + s2];
            if (v1 > v0 || (v1 == v0 && i1 < i0)) { sm[tid] = v1; si[tid] = i1; }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[lane] = (int)si[0];
}

// ---- WP09: ONE-launch multi-block greedy argmax over m rows (replaces m single-block 1024-thread
// xq_argmax launches, ~21 us each). grid (NB <= 32, m), block = multiple of 32.
// EXACTNESS: every thread folds its elements with xq_argmax's comparator verbatim
// (v > best || (v == best && i < bi)) from xq_argmax's init (-inf, 0). NaN never compares true,
// so the fold is the lexicographic max over {(-inf, 0)} U {(x_i, i) : x_i not NaN} under
// (value desc by fp32 compare — +0 == -0 — then id asc). That order is total on the candidate
// set, so the winner does not depend on the partition or the reduction tree: xq_argmax's
// one-block tree and this kernel's (per-thread fold -> warp -> block -> cross-block) return the
// SAME id, including the ties, the +-0, all--inf (-> 0) and all-NaN (-> 0) rows (probe:
// EXL3-ARGMAX-ROWS). Cross-block: (best, bi) packs into an order-preserving u64 key (canonical
// +0, sign-flip map on the float bits; low word = ~id so a larger key = a lower id), each block
// publishes its key, __threadfence + per-row counter; the LAST block of row r reduces the NB keys
// (L2 reads) and writes out_ids[lane0 + r], then re-arms cnt[r] = 0 (one real zero at allocation,
// graph replays reuse the buffers — the single-stream contract, as hc_bar / done_cnt).
__device__ __forceinline__ void xq_am_take(float& best, int& bi, float v, int i) {
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
}
__device__ __forceinline__ unsigned long long xq_am_key(float v, int i) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) == 0u) u = 0u;                  // -0 == +0 under the fp32 compare
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);       // order-preserving on non-NaN floats
    return ((unsigned long long)u << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned)i);
}
__device__ __forceinline__ unsigned long long xq_am_wmax(unsigned long long k) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long ok = __shfl_xor_sync(0xffffffffu, k, o);
        k = ok > k ? ok : k;
    }
    return k;
}
extern "C" __global__ void xq_argmax_rows(int* __restrict__ out_ids, int lane0,
                                          const __half* __restrict__ logits, long long start0,
                                          long long rstride, int V,
                                          unsigned long long* __restrict__ part,
                                          unsigned int* __restrict__ cnt) {
    XQ_PDL_ENTRY();
    __shared__ unsigned long long wk[32];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const __half* row = logits + start0 + (long long)r * rstride;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;     // 8-aligned slices -> row+lo keeps row's alignment
    const int lo = min(b * chunk, V), hi = min(lo + chunk, V);
    float best = -INFINITY; int bi = 0;
    if ((reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull) {
        const int nv = (hi - lo) >> 3;
        const uint4* rv = reinterpret_cast<const uint4*>(row + lo);
        #pragma unroll 4
        for (int q = tid; q < nv; q += bs) {
            const uint4 u = __ldg(rv + q);
            const __half* hv = reinterpret_cast<const __half*>(&u);
            #pragma unroll
            for (int j = 0; j < 8; j++) xq_am_take(best, bi, xq_h2f(hv[j]), lo + q * 8 + j);
        }
        for (int i = lo + nv * 8 + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    } else {
        for (int i = lo + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    }
    unsigned long long key = xq_am_wmax(xq_am_key(best, bi));
    if ((tid & 31) == 0) wk[tid >> 5] = key;
    __syncthreads();
    if (tid < 32) {
        key = xq_am_wmax(tid < (bs >> 5) ? wk[tid] : 0ull);   // 0 < every real key (ord(-inf) > 0)
        if (tid == 0) {
            part[(long long)r * nb + b] = key;
            __threadfence();
            last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
        }
    }
    __syncthreads();
    if (!last) return;
    if (tid < 32) {
        __threadfence();
        const unsigned long long k2 = xq_am_wmax(tid < nb ? __ldcg(part + (long long)r * nb + tid) : 0ull);
        if (tid == 0) {
            out_ids[lane0 + r] = (int)(0xFFFFFFFFu - (unsigned)(k2 & 0xFFFFFFFFull));
            cnt[r] = 0u;                                      // re-arm for the next launch / replay
        }
    }
}

// ---- API-parity DDS: max logit of one fp16 row -> out[ooff] (f32). The draft head's argmax
// LOGIT is the reference implementation's dynamic-draft confidence score (exllamav3 DraftConfidenceCalibrator).
extern "C" __global__ void xq_rowmax_f16(float* __restrict__ out, long long ooff,
                                         const __half* __restrict__ x, long long start, int V) {
    XQ_PDL_ENTRY();
    __shared__ float sm[32];
    float best = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) best = fmaxf(best, __half2float(x[start + i]));
    for (int o = 16; o; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : -INFINITY;
        for (int o = 16; o; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (threadIdx.x == 0) out[ooff] = w;
    }
}

// =============================================================================
// DHEAD (w2, 2026-09-26): draft lm_head diet — DRAFT-ONLY (verification is untouched).
// The draft pass's pruned lm_head (EXL3 5-bit column slice, 65,536 x 2560 = 104.9 MB/pass) ran as
// had_suh + exl3_hmma_gemm (0.52 ms, ~85% of its byte floor) + had_svh + argmax_rows + rowmax
// + 2 copies. Replacement, three launches (target: baseline sm_121 — dp4a, mma.sync.m16n8k16 f16,
// ld.global.cs; nothing family/arch-specific is needed for a byte-bound GEMV):
//  (1) xq_dh_prep    one CTA: xh = had128(x * suh) with exl3_had_suh_dev's ops verbatim (the f16
//                    xh is BITWISE the old chain's), then a per-vector int8 quantization of xh in
//                    the screen's byte order + {sx, sum q, screen noise sigma}.
//  (2) xq_dh_screen  B-bit (1|2|4) SCREEN of the SAME rotated weights. Codes are built at load from
//                    the decoded trellis with one GLOBAL uniform quantizer (the mul1 codebook is
//                    unit-variance by construction, so group scales buy nothing). One CTA per
//                    128-col Hadamard block (512 CTAs, all co-resident at 12/SM), dp4a over
//                    streamed (ld.global.cs, single-use) 16-B code chunks, in-CTA had128 x svh
//                    epilogue -> approximate block max bmax[b]. 2-bit = 41.9 MB/pass.
//  (3) xq_dh_rescore every CTA selects the SAME candidate set (blocks whose approximate max is
//                    within delta * sigma * (bsig_b + bsig_max) of the best, at most tmax, ties and
//                    truncation by (value desc, index asc)); CTA (r, s) recomputes candidate r's
//                    128 logits EXACTLY from the EXL3 trellis over K-split s (exl3_gemm_body's
//                    decode + m16n8k16 mma per k16 step, staged through smem); the LAST CTA sums
//                    the S fp32 partials in ascending s -> f16 yraw -> had_svh's ops verbatim ->
//                    (max, lowest id) over the candidates -> argmax, dconf[i], d_dev[i], toks[0]
//                    (replacing argmax_rows + xq_rowmax_f16 + 2 x xq_copy1_i32).
// Exactness: when the old argmax's block is a candidate, the draft id and dconf equal the old
// head's up to the split-K fp32 re-association before the f16 yraw rounding (a 1-ulp tie class);
// otherwise the draft is the best in-candidate token. Greedy output never depends on the draft.
// Workspace `ws` (u32 words; mirrored by src/exl3_forward/dhead.rs ws_layout):
//   [0, K/4)            screen x (int8, permuted);  [K/4 + {0,1,2}] = sx, sum q, sigma
//   [off_bmax, +nblk)   approximate block maxima (f32)
//   [off_cnt]           rescore last-CTA counter (self re-arming; one real zero at build)
//   [off_dbg, +TCAP+4)  {ns, cand[0..ns), .., nq} (XCHECK readback)
//   [off_part, ...)     fp32 partials [TCAP][S][128]
// =============================================================================
#define DH_KPT 8     // approximate block maxima per rescore thread: nblk <= 256 * DH_KPT
#define DH_TCAP 32   // candidate cap (tmax <= DH_TCAP)
#define DH_RMAX 16   // k16 rows per rescore K-split (host: (K/16) / S <= DH_RMAX)

__device__ __forceinline__ int dh_off_bmax(int K) { return ((K >> 2) + 4 + 3) & ~3; }
__device__ __forceinline__ int dh_off_cnt(int K, int nblk) { return dh_off_bmax(K) + ((nblk + 3) & ~3); }
__device__ __forceinline__ int dh_off_dbg(int K, int nblk) { return dh_off_cnt(K, nblk) + 4; }
__device__ __forceinline__ int dh_off_part(int K, int nblk) { return dh_off_dbg(K, nblk) + DH_TCAP + 4; }

// order-preserving u32 of a non-NaN float (xq_am_key's high word) and its inverse
__device__ __forceinline__ unsigned dh_ord(float v) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) == 0u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float dh_unord(unsigned o) {
    return (o & 0x80000000u) ? __uint_as_float(o & 0x7FFFFFFFu) : __uint_as_float(~o);
}

// Load-time build: one thread per u32 code word. Word wi <-> (block nb, chunk kc, column cl,
// word u): wi = ((nb * KC + kc) * 128 + cl) * 4 + u, so a warp's 16-B loads in the screen are
// 32 consecutive columns (512 B) and a CTA's whole block is one contiguous stream. Word u of
// chunk kc covers k-group g = 4 kc + u (G = 32/B consecutive k); code(k = g G + b cpb + j) sits
// at bits [8 b + B j, +B), cpb = 8/B, so (word >> B j) & mask puts codes j of the 4 bytes in the
// byte lanes that xq_dh_prep's x word (g, j) holds. Value of code c = dq * (c - (2^B - 1) / 2).
extern "C" __global__ void __launch_bounds__(256) xq_dh_build(
        const uint16_t* __restrict__ tr, unsigned* __restrict__ codes,
        int K, int N, int bits, int B, float inv_dq, long long nwords) {
    const long long wi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= nwords) return;
    const int G = 32 / B, cpb = 8 / B, KC = K / (128 / B), L = 1 << B;
    const int u = (int)(wi & 3);
    const int cl = (int)((wi >> 2) & 127);
    const long long rest = wi >> 9;
    const int kc = (int)(rest % KC);
    const int nb = (int)(rest / KC);
    const int c = nb * 128 + cl, g = kc * 4 + u;
    const int nb16 = N >> 4, n16 = c >> 4, cc = c & 15;
    unsigned word = 0u;
    for (int kl = 0; kl < G; ++kl) {
        const int k = g * G + kl, kb = k >> 4, r = k & 15;
        const uint32_t* ring = reinterpret_cast<const uint32_t*>(tr + ((size_t)kb * nb16 + n16) * (16 * bits));
        const int l = 8 * ((cc >> 1) & 3) + ((r & 7) >> 1);                          // tmap(r, cc)
        const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (cc >> 3) + 32 * (cc & 1);
        const float w = __half2float(__ushort_as_half(exl3_dq_t(ring, t, bits)));
        int q = (int)floorf(w * inv_dq + 0.5f * (float)L);
        q = max(0, min(L - 1, q));
        const int b = kl / cpb, j = kl - b * cpb;
        word |= (unsigned)q << (8 * b + B * j);
    }
    codes[wi] = word;
}

// (1) prep: blockDim = 32 * (K/128) (one warp per 128-block of K, K <= 4096).
extern "C" __global__ void __launch_bounds__(1024) xq_dh_prep(
        const __half* __restrict__ x, const __half* __restrict__ suh, __half* __restrict__ xh,
        unsigned* __restrict__ ws, int K, int B, float mse) {
    XQ_PDL_ENTRY();
    __shared__ float s_am[32];
    __shared__ float s_ss[32];
    __shared__ int s_qs[32];
    __shared__ __align__(16) unsigned char s_q[4096];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
    const int base = warp * 128 + lane * 4;          // exl3_had_suh_dev at m = 0, kb = warp
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __half2float(x[base + r]) * __half2float(suh[base + r]);
    exl3_had128(v, lane);
    float xf[4], am = 0.f, ss = 0.f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const __half hv = __float2half_rn(v[r]);
        xh[base + r] = hv;                           // == exl3_had_suh's xh, bit for bit
        xf[r] = __half2float(hv);
        am = fmaxf(am, fabsf(xf[r]));
        ss = fmaf(xf[r], xf[r], ss);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) { s_am[warp] = am; s_ss[warp] = ss; }
    __syncthreads();
    float amax = 0.f, sst = 0.f;
    for (int w = 0; w < nw; ++w) { amax = fmaxf(amax, s_am[w]); sst += s_ss[w]; }
    const float sx = amax > 0.f ? amax * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sx;
    const int cpb = 8 / B, G = 32 / B;
    int qs = 0;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int q = max(-127, min(127, __float2int_rn(xf[r] * inv)));
        qs += q;
        const int k = base + r, g = k / G, kl = k - g * G, b = kl / cpb, j = kl - b * cpb;
        s_q[(g * cpb + j) * 4 + b] = (unsigned char)(q & 0xFF);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o);
    if (lane == 0) s_qs[warp] = qs;
    __syncthreads();
    const int KW = K >> 2;
    for (int i = tid; i < KW; i += blockDim.x) ws[i] = reinterpret_cast<const unsigned*>(s_q)[i];
    if (tid == 0) {
        int S = 0;
        for (int w = 0; w < nw; ++w) S += s_qs[w];
        ws[KW] = __float_as_uint(sx);
        ws[KW + 1] = (unsigned)S;
        // yraw noise sigma: weight-quantization MSE * |xh|^2 + x-quantization (sx^2/12 per term, K
        // unit-variance weights). The logit noise of column j is |svh_j| * sigma (had128 is orthonormal).
        ws[KW + 2] = __float_as_uint(sqrtf(mse * sst + (float)K * sx * sx * (1.0f / 12.0f)));
        ws[KW + 3] = 0u;
    }
}

// dp4a over one 16-B code chunk (4 words x 32/B codes) against its 32/B x words in smem.
template <int B>
__device__ __forceinline__ int dh_dot4(const uint4 w, const unsigned* xs, int acc) {
    constexpr int CPB = 8 / B;
    constexpr unsigned MASK = ((1u << B) - 1u) * 0x01010101u;
    const unsigned wv[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int j = 0; j < CPB; ++j)
            acc = __dp4a((int)((wv[u] >> (B * j)) & MASK), (int)xs[u * CPB + j], acc);
    }
    return acc;
}

template <int B>
__device__ __forceinline__ void dh_screen_body(const uint4* __restrict__ codes, unsigned* __restrict__ ws,
                                               const __half* __restrict__ svh, int K, float dq,
                                               unsigned* xs, float* s_y,
                                               float* __restrict__ bdst, int nb_off) {
    constexpr int XPU = 32 / B;                      // x words per 16-B code chunk (= 4 * cpb)
    // nb = the GLOBAL 128-column block: nb_off = 0 for the replicated screen, rank * nblk / world
    // for TP-H #3's sharded one. bdst[nb] = the approximate block max (ws bmax for the replicated
    // screen, the zero-padded exchange buffer for the sharded one).
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nb = nb_off + blockIdx.x;
    const int KW = K >> 2, KC = K / (128 / B);
    for (int i = tid; i < KW; i += 128) xs[i] = ws[i];
    __syncthreads();
    // 4 independent 16-B streaming loads in flight per thread (x 12 CTAs x 128 threads = 96 KB
    // per SM, far above the ~5 KB/SM the DRAM queue needs); one accumulator — the other 47
    // resident warps hide the dp4a chain. Nothing but the pointer/acc lives across the loop.
    const uint4* __restrict__ p = codes + (size_t)nb * KC * 128 + tid;
    int acc = 0;
    int kc = 0;
    for (; kc + 4 <= KC; kc += 4) {
        const uint4 w0 = __ldcs(p + (size_t)(kc + 0) * 128);
        const uint4 w1 = __ldcs(p + (size_t)(kc + 1) * 128);
        const uint4 w2 = __ldcs(p + (size_t)(kc + 2) * 128);
        const uint4 w3 = __ldcs(p + (size_t)(kc + 3) * 128);
        acc = dh_dot4<B>(w0, xs + (kc + 0) * XPU, acc);
        acc = dh_dot4<B>(w1, xs + (kc + 1) * XPU, acc);
        acc = dh_dot4<B>(w2, xs + (kc + 2) * XPU, acc);
        acc = dh_dot4<B>(w3, xs + (kc + 3) * XPU, acc);
    }
    for (; kc < KC; ++kc) acc = dh_dot4<B>(__ldcs(p + (size_t)kc * 128), xs + kc * XPU, acc);
    const float sx = __uint_as_float(ws[KW]);
    const int S = (int)ws[KW + 1];
    const float z = 0.5f * (float)((1 << B) - 1);
    s_y[tid] = dq * sx * ((float)acc - z * (float)S);   // approximate yraw of column tid
    __syncthreads();
    if (warp == 0) {
        float v[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = s_y[lane * 4 + r];
        exl3_had128(v, lane);
        float m = -INFINITY;
        #pragma unroll
        for (int r = 0; r < 4; ++r) m = fmaxf(m, v[r] * __half2float(svh[nb * 128 + lane * 4 + r]));
        #pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) bdst[nb] = m;
    }
}

// (2) screen: grid nblk = N/128, block 128. K <= 4096, K % (128/B) == 0.
extern "C" __global__ void __launch_bounds__(128, 12) xq_dh_screen(
        const uint4* __restrict__ codes, unsigned* __restrict__ ws, const __half* __restrict__ svh,
        int K, int B, float dq) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned xs[1024];
    __shared__ float s_y[128];
    float* const bdst = reinterpret_cast<float*>(ws + dh_off_bmax(K));
    switch (B) {
        case 1: dh_screen_body<1>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        case 2: dh_screen_body<2>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        case 4: dh_screen_body<4>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        default: break;                               // the loader refuses other widths
    }
}

// Exact per-split partials of one candidate block: exl3_gemm_body's slot constants, decode and
// m16n8k16 mma (A = the one live row, zero elsewhere), ring words from smem.
template <int BITS>
__device__ __forceinline__ void dh_rescore_mma(const uint32_t* __restrict__ ringw,
                                               const unsigned* __restrict__ sx2, int R, int lane,
                                               float (&acc)[2][4]) {
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const bool live = (lane >> 2) == 0;              // gid 0 = row 0, the only live A row
    const int tig = lane & 3;
    for (int row = 0; row < R; ++row) {
        const uint32_t* ring = ringw + row * (8 * BITS);
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        // A fragments: {a0,a1} = A[gid][2tig..], {a2,a3} = A[gid+8][..] (0), {a4,a5} = A[gid][2tig+8..]
        const uint32_t a0 = live ? sx2[row * 8 + tig] : 0u;
        const uint32_t a2 = live ? sx2[row * 8 + 4 + tig] : 0u;
        const uint32_t z = 0u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(z), "r"(a2), "r"(z),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }
}

// (3) rescore: grid (S, tmax) — split-fastest, so the ns * S working CTAs are the FIRST ns * S in
// dispatch order and the (tmax - ns) * S early-exit CTAs never delay them; block 256; dynamic
// smem = R * 8 * 32 * hbits bytes (<= 32 KB). geom = pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32.
// tr / svh = the EXACT pruned head (Quad lm_head_draft: N columns, nb16 = N/16 trellis stride).
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        const float* __restrict__ bsig, unsigned* __restrict__ ws,
        int* __restrict__ out_id, float* __restrict__ dconf, int* __restrict__ d_dev, int* __restrict__ toks,
        long long geom, int N, float delta) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];            // [8 warps][R rows][2 * bits]
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ unsigned s_om[8];
    __shared__ int s_nq[8];
    __shared__ int s_bs[2][8];
    __shared__ int s_wh[DH_KPT * 8];                 // per (round i, warp w) selected counts
    __shared__ int s_we[DH_KPT * 8];
    __shared__ int s_ph[DH_KPT * 8];                 // their exclusive prefixes in (i, w) order
    __shared__ int s_pe[DH_KPT * 8];
    __shared__ int s_tot[2];
    __shared__ int s_cand[DH_TCAP];
    __shared__ unsigned long long s_best[8];
    __shared__ int s_last;
    const int pass_i = (int)(geom & 0xFF), bits = (int)((geom >> 8) & 0xFF);
    const int S = (int)((geom >> 16) & 0xFF), tmax = (int)((geom >> 24) & 0xFF), K = (int)(geom >> 32);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7, KW = K >> 2;
    const float* bmax = reinterpret_cast<const float*>(ws + dh_off_bmax(K));
    unsigned* cnt = ws + dh_off_cnt(K, nblk);
    int* dbg = reinterpret_cast<int*>(ws + dh_off_dbg(K, nblk));
    float* part = reinterpret_cast<float*>(ws + dh_off_part(K, nblk));
    const float sig = __uint_as_float(ws[KW + 2]);
    const float bsmax = bsig[nblk];                   // max over blocks (host-appended)

    // ---- 1. candidate selection: identical in every CTA (pure function of bmax, sig, bsig) ----
    unsigned ob[DH_KPT];                              // ordered block maxima (0 = no block)
    unsigned om = 0u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        float v = -INFINITY;
        if (b < nblk) { v = bmax[b]; if (!(v == v)) v = -INFINITY; }
        ob[i] = b < nblk ? dh_ord(v) : 0u;
        om = max(om, ob[i]);
    }
    om = __reduce_max_sync(0xffffffffu, om);
    if (lane == 0) s_om[warp] = om;
    __syncthreads();
    unsigned go = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) go = max(go, s_om[w]);
    const float gmax = dh_unord(go);
    unsigned qm = 0u;                                 // qualified: within the noise margin of the best
    int qc = 0;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        if (b < nblk) {
            const float marg = fmaxf(delta * sig * (bsig[b] + bsmax), 0.f);   // NaN -> 0
            if (dh_unord(ob[i]) + marg >= gmax) { qm |= 1u << i; ++qc; }
        }
    }
    qc = __reduce_add_sync(0xffffffffu, qc);
    if (lane == 0) s_nq[warp] = qc;
    __syncthreads();
    int nq = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nq += s_nq[w];
    unsigned hm = qm, em = 0u;                        // ranked first (ascending id), then tie fill
    if (nq > tmax) {                                  // uniform branch
        // tau = the tmax-th largest ordered value among the qualified (32-step bisection)
        unsigned tau = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned t = tau | (1u << bit);
            int c = 0;
            #pragma unroll
            for (int i = 0; i < DH_KPT; ++i) c += (((qm >> i) & 1u) && ob[i] >= t) ? 1 : 0;
            c = __reduce_add_sync(0xffffffffu, c);
            if (lane == 0) s_bs[bit & 1][warp] = c;
            __syncthreads();
            int tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += s_bs[bit & 1][w];
            if (tot >= tmax) tau = t;
        }
        hm = 0u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            if ((qm >> i) & 1u) {
                if (ob[i] > tau) hm |= 1u << i;
                else if (ob[i] == tau) em |= 1u << i;
            }
        }
    }
    // ascending-index ranks: all of hm (count < tmax when truncating), then em fills the rest.
    // Block b = tid + 256 i, so ascending b = (i, warp, lane) order.
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
        const unsigned be = __ballot_sync(0xffffffffu, (em >> i) & 1u);
        if (lane == 0) { s_wh[i * 8 + warp] = __popc(bh); s_we[i * 8 + warp] = __popc(be); }
    }
    __syncthreads();
    if (tid < 2 * DH_KPT * 8) {                       // 64 prefix entries per mask
        const int e = tid & (DH_KPT * 8 - 1);
        const int* src = tid < DH_KPT * 8 ? s_wh : s_we;
        int pfx = 0;
        for (int f = 0; f < e; ++f) pfx += src[f];
        (tid < DH_KPT * 8 ? s_ph : s_pe)[e] = pfx;
        if (e == DH_KPT * 8 - 1) s_tot[tid < DH_KPT * 8 ? 0 : 1] = pfx + src[e];
    }
    __syncthreads();
    const int nh = s_tot[0];
    const int ns = min(min(nh + s_tot[1], tmax), DH_TCAP);
    {
        const unsigned lt = (1u << lane) - 1u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const unsigned mh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
            const unsigned me = __ballot_sync(0xffffffffu, (em >> i) & 1u);
            const int b = tid + 256 * i;
            if ((hm >> i) & 1u) {
                const int rk = s_ph[i * 8 + warp] + __popc(mh & lt);
                if (rk < ns) s_cand[rk] = b;
            }
            if ((em >> i) & 1u) {
                const int rk = nh + s_pe[i * 8 + warp] + __popc(me & lt);
                if (rk < ns) s_cand[rk] = b;
            }
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0) {         // XCHECK readback (a few words)
        if (tid < ns) dbg[1 + tid] = s_cand[tid];
        if (tid == 0) { dbg[0] = ns; dbg[DH_TCAP + 1] = nq; }
    }
    const int rc = blockIdx.y, sp = blockIdx.x;
    if (rc >= ns) return;                             // uniform per CTA
    const int cb = s_cand[rc];

    // ---- 2. exact partial of candidate cb over k16 rows [sp R, (sp + 1) R) ----
    const int R = (K >> 4) / S, kb0 = sp * R;
    const int nb16 = N >> 4;
    const int U4 = 2 * bits;                          // 16-B chunks per trellis block (32 * bits B)
    const int rowu4 = 8 * U4;                         // one k16 row of the 8 column blocks
    for (int idx = tid; idx < R * rowu4; idx += 256) {
        const int row = idx / rowu4, rem = idx - row * rowu4;
        const int w = rem / U4, q = rem - w * U4;
        const uint4* src = reinterpret_cast<const uint4*>(tr + ((size_t)(kb0 + row) * nb16 + (size_t)cb * 8) * (16 * bits));
        s_ring4[(w * R + row) * U4 + q] = __ldcs(src + rem);
    }
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    __syncthreads();
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    const uint32_t* ringw = reinterpret_cast<const uint32_t*>(s_ring4 + warp * R * U4);
    switch (bits) {
        case 2: dh_rescore_mma<2>(ringw, s_x2, R, lane, acc); break;
        case 3: dh_rescore_mma<3>(ringw, s_x2, R, lane, acc); break;
        case 4: dh_rescore_mma<4>(ringw, s_x2, R, lane, acc); break;
        case 5: dh_rescore_mma<5>(ringw, s_x2, R, lane, acc); break;
        case 6: dh_rescore_mma<6>(ringw, s_x2, R, lane, acc); break;
        case 7: dh_rescore_mma<7>(ringw, s_x2, R, lane, acc); break;
        case 8: dh_rescore_mma<8>(ringw, s_x2, R, lane, acc); break;
        default: break;
    }
    if ((lane >> 2) == 0) {                           // row 0 of the two n8 tiles: cols 16w + 8h + 2tig
        float* pp = part + ((size_t)(rc * S + sp) * 128) + warp * 16 + 2 * (lane & 3);
        pp[0] = acc[0][0]; pp[1] = acc[0][1]; pp[8] = acc[1][0]; pp[9] = acc[1][1];
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = (atomicAdd(cnt, 1u) == (unsigned)(ns * S) - 1u) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) return;

    // ---- 3. last CTA: ascending-s combine -> f16 yraw -> had_svh -> (max, lowest id) ----
    __threadfence();
    unsigned long long best = 0ull;
    for (int rr = warp; rr < ns; rr += 8) {
        const int bb = s_cand[rr];
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* pc = part + (size_t)rr * S * 128 + lane * 4 + q;
            float a = 0.f;
            for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
            v[q] = __half2float(__float2half_rn(a));   // yraw (exl3_gemm_body's f16 RN store)
        }
        exl3_had128(v, lane);                         // exl3_had_svh_dev's ops
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = bb * 128 + lane * 4 + q;
            const float y = __half2float(__float2half_rn(v[q] * __half2float(svh[col])));
            const unsigned long long kk = xq_am_key(y, col);
            best = kk > best ? kk : best;
        }
    }
    best = xq_am_wmax(best);
    if (lane == 0) s_best[warp] = best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long k = 0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) k = s_best[w] > k ? s_best[w] : k;
        const int id = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
        out_id[0] = id;                               // sc.argmax (== argmax_rows' lane 0)
        dconf[pass_i] = dh_unord((unsigned)(k >> 32)); // == xq_rowmax_f16 of the in-candidate max
        d_dev[pass_i] = id;                           // == the old xq_copy1_i32 pair
        toks[0] = id;
        cnt[0] = 0u;                                  // re-arm for the next launch / replay
    }
}

// ---- API-parity G2: device sampler for sampled (temperature > 0) lanes.
// temperature -> top-k -> top-p -> multinomial over fp16 logits; one 1024-thread block per row.
// Per-row params samp[row*8 + {0 flag, 1 T(f32 bits), 2 top_p(f32), 3 top_k(0=off), 4 seed_lo,
// 5 seed_hi, 6 counter, 7 min_p(f32, 0=off)}]. flag == 0 -> the row keeps xq_argmax's id
// untouched (greedy rows stay bit-identical). Overwrites out_ids[row] with the sample.
// Semantics: top-k keeps every logit >= the k-th largest (ties included, the HF warper rule);
// top-p keeps the smallest descending-logit prefix whose mass >= top_p (ties at the boundary
// included); min_p keeps p >= min_p * p_max. Sampling = inverse CDF over the survivors in a
// fixed (warp-chunk, index) order with a counter-hashed uniform — a pure function of
// (logits, params), so a fixed seed reproduces. Used for plain steps AND MTP verify rows:
// with a deterministic (argmax) draft, "sample the target at each verify row, accept while
// sample == draft" is distribution-exact (the reference implementation's EXL3_BATCH_VERIFY rule).
__device__ __forceinline__ unsigned int xq_skey(unsigned short b) {
    return (b & 0x8000u) ? ((~(unsigned int)b) & 0xFFFFu) : ((unsigned int)b | 0x8000u);
}
__device__ __forceinline__ float xq_skey_val(unsigned int k) {
    unsigned short b = (k & 0x8000u) ? (unsigned short)(k & 0x7FFFu) : (unsigned short)((~k) & 0xFFFFu);
    return __half2float(__ushort_as_half(b));
}
__device__ __forceinline__ unsigned long long xq_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// fixed-order block reductions (deterministic: same shuffle/tree every launch)
__device__ __forceinline__ unsigned int xq_bmax_u32(unsigned int v, unsigned int* sh) {
    for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        unsigned int w = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0u;
        for (int o = 16; o; o >>= 1) w = max(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (threadIdx.x == 0) sh[0] = w;
    }
    __syncthreads();
    unsigned int r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float xq_bsum_f32(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
        for (int o = 16; o; o >>= 1) w += __shfl_xor_sync(0xffffffffu, w, o);
        if (threadIdx.x == 0) sh[0] = w;
    }
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}
extern "C" __global__ void __launch_bounds__(1024) xq_sample_rows(
        int* __restrict__ out_ids, const __half* __restrict__ logits, int V,
        const unsigned int* __restrict__ samp) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.x;
    const unsigned int* P = samp + row * 8;
    if (P[0] == 0u) return;
    const float T = fmaxf(__uint_as_float(P[1]), 1e-6f);
    const float invT = 1.0f / T;
    const float top_p = __uint_as_float(P[2]);
    const unsigned int top_k = P[3];
    const unsigned long long seed = ((unsigned long long)P[5] << 32) | (unsigned long long)P[4];
    const unsigned int ctr = P[6];
    const float min_p = __uint_as_float(P[7]);
    const unsigned short* L = reinterpret_cast<const unsigned short*>(logits) + (long long)row * V;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;

    __shared__ unsigned int ush[32];
    __shared__ float fsh[32];
    __shared__ unsigned int hist[256];
    __shared__ float fh[256];
    __shared__ unsigned int s_u[4];
    __shared__ float s_f[2];

    // 1. max key -> lmax (softmax shift)
    unsigned int mk = 0u;
    for (int i = tid; i < V; i += bs) mk = max(mk, xq_skey(L[i]));
    mk = xq_bmax_u32(mk, ush);
    const float lmax = xq_skey_val(mk);

    // 2. top-k key threshold tau (radix select over the 16-bit key, ties included)
    unsigned int tau = 0u;
    if (top_k > 0u && top_k < (unsigned int)V) {
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) atomicAdd(&hist[xq_skey(L[i]) >> 8], 1u);
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, hb = 0u, need = 1u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= top_k) { hb = (unsigned int)b; need = top_k - cum; break; }
                cum += hist[b];
            }
            s_u[0] = hb; s_u[1] = need;
        }
        __syncthreads();
        const unsigned int hb = s_u[0], need = s_u[1];
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if ((k >> 8) == hb) atomicAdd(&hist[k & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, lb = 0u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= need) { lb = (unsigned int)b; break; }
                cum += hist[b];
            }
            s_u[2] = (hb << 8) | lb;
        }
        __syncthreads();
        tau = s_u[2];
    }
    // min_p as a logit floor: p >= min_p * p_max  <=>  l >= lmax + T * ln(min_p)
    const float mfloor = (min_p > 0.0f) ? lmax + T * __logf(min_p) : -INFINITY;

    // 3. candidate mass S (deterministic)
    float part = 0.0f;
    for (int i = tid; i < V; i += bs) {
        unsigned int k = xq_skey(L[i]);
        if (k >= tau) {
            float x = xq_skey_val(k);
            if (x >= mfloor) part += __expf((x - lmax) * invT);
        }
    }
    const float S = xq_bsum_f32(part, fsh);

    // 4. top-p key threshold theta (mass radix select; boundary ties included)
    unsigned int thr = tau;
    if (top_p < 1.0f && S > 0.0f) {
        const float target = top_p * S;
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k >> 8], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int hb = 0u; float rem = 0.0f; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= target) { hb = (unsigned int)b; rem = target - cum; found = 1; break; }
                cum += fh[b];
            }
            if (!found) { hb = (unsigned int)max(last, 0); rem = 3.0e38f; }
            s_u[0] = hb; s_f[0] = rem;
        }
        __syncthreads();
        const unsigned int hb = s_u[0];
        const float rem = s_f[0];
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau && (k >> 8) == hb) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k & 255u], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int lb = 0u; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= rem) { lb = (unsigned int)b; found = 1; break; }
                cum += fh[b];
            }
            if (!found) lb = (unsigned int)max(last, 0);
            s_u[3] = max((hb << 8) | lb, tau);
        }
        __syncthreads();
        thr = s_u[3];
    }

    // 5. inverse CDF over survivors: warp w owns [w*chunk, (w+1)*chunk), lanes stride 32.
    const int chunk = (V + nw - 1) / nw;
    const int w0 = wid * chunk, w1 = min(V, w0 + chunk);
    float lsum = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) lsum += __expf((x - lmax) * invT);
        }
    }
    for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) fsh[wid] = lsum;
    __syncthreads();
    if (tid == 0) {
        const float u = ((float)(xq_mix64(seed ^ xq_mix64((unsigned long long)ctr + 1ull)) >> 40) + 0.5f)
                        * (1.0f / 16777216.0f);
        float tot = 0.0f;
        for (int w = 0; w < nw; w++) tot += fsh[w];
        const float target = u * tot;
        float cum = 0.0f; int sel = -1, last = 0;
        for (int w = 0; w < nw; w++) {
            if (fsh[w] > 0.0f) last = w;
            if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
            cum += fsh[w];
        }
        if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
        s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
    }
    __syncthreads();
    if (wid == (int)s_u[0]) {
        const float want = s_f[0];
        float base = 0.0f;
        int pick = -1, lastidx = -1;
        for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
            const int i = i0 + lane;
            float wgt = 0.0f;
            if (i < w1) {
                unsigned int k = xq_skey(L[i]);
                if (k >= thr) {
                    float x = xq_skey_val(k);
                    if (x >= mfloor) wgt = __expf((x - lmax) * invT);
                }
            }
            float incl = wgt;
            for (int o = 1; o < 32; o <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
            unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
            if (any) lastidx = i0 + 31 - __clz(any);
            if (hit) pick = i0 + __ffs(hit) - 1;
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
        // no survivor at all (non-finite row) -> keep xq_argmax's id
        if (lane == 0 && (pick >= 0 || lastidx >= 0)) out_ids[row] = (pick >= 0) ? pick : lastidx;
    }
}

// ---- WP15: repetition / presence / frequency penalties with the REFERENCE implementation's semantics
// (exllamav3 ComboSampler: SS_RepP then SS_PresFreqP, exllamav3_ext/generator/rep_pen.cu), applied
// in place to fp16 logit rows BEFORE xq_argmax / xq_sample_rows, at every decode row and every
// verify row. Per-row params pen[row*8 + {0 flag (bit0 rep, bit1 pres/freq; 0 = the row is
// untouched), 1 rep_p (f32), 2 pres_p (f32), 3 freq_p (f32), 4 sustain, 5 decay, 6-7 0}].
// Window: past positions i with past_len - i < sustain + decay (the reference implementation's `i > past_len - S - D`),
// factor(dist) = 1 for dist <= S, else 1 - (dist - S) / D (clamped); the rep factor and the presence
// factor are the MAX over a token's occurrences (its most recent), the frequency term the SUM.
//   rep:  w = v > 0 ? v / rep_p : v * rep_p;  fr = f + 1e-30;  v = v * ((1 - fr) + 1e-30) + w * fr
//   pres/freq: v -= freq_p * sum(factor);  v -= f * pres_p
// Factors are exact integers in 1/D units (the reference implementation's float factor is k/D, exact for D = 1024),
// accumulated with integer atomics: deterministic, unlike the reference implementation's float atomicAdd; the frequency
// sum can differ from the reference implementation's by float rounding only. Only tokens seen in the window are
// rewritten — for every other token the reference implementation's formula returns v bit-exactly — so a penalized row
// costs O(window) work per 4096-id span, and an unpenalized row (flag 0) exits at once.
// Past tokens: a per-slot device ring `hist` of R = 1 << rlog2 ids indexed by position & (R-1)
// (positions < base: committed history) plus the in-flight row tokens (positions >= base).
#define XQ_PEN_SPAN 4096
// One token's penalized fp16 logit (xq_pen_span's per-token ops; DHEADP's rescore calls the SAME
// function, so a draft's penalized logit is the draft mirror's to the bit): v = the fp16 logit as
// f32, fm / fs[j] = the token's max / summed window factors in 1/D units (fm > 0), scale = D (or 1);
// fs[j] is read only when the pres/freq bit is set (xq_pen_span's load placement: its PTX is unchanged).
// MONOTONICITY (DHEADP's bound, w3 2026-09-27): with rep >= 1, pres >= 0, freq >= 0 the result is
// <= v for every v >= -65504 (rep: w <= v, and v*f1 + w*fr exceeds v by at most a few f32 ulps —
// far under half an fp16 ulp, so the fp16 RN never rounds above v; pres/freq subtract >= 0; the
// +65504 clamp only lowers). Only v = -inf rises (to -65504). rep < 1 or pres/freq < 0 break it.
__device__ __forceinline__ unsigned short xq_pen_tok(float v, unsigned int fm, const unsigned int* fs,
                                                     int j, unsigned int flag, float rep, float pres,
                                                     float freq, float scale) {
    const float f = (float)fm / scale;
    if (flag & 1u) {
        // literally the reference implementation's expression (rep_pen.cu apply_rep_pens_kernel)
        const float w = v > 0.0f ? v / rep : v * rep;
        const float fr = f + 1e-30f;
        const float f1 = (1.0f - fr) + 1e-30f;
        v = v * f1 + w * fr;
    }
    if (flag & 2u) {
        v -= freq * ((float)fs[j] / scale);
        v -= f * pres;
    }
    // keep the row finite in fp16 (an overflow to +inf would poison the sampler's softmax
    // shift); NaN stays NaN (neither comparison holds)
    if (v > 65504.0f) v = 65504.0f; else if (v < -65504.0f) v = -65504.0f;
    return __half_as_ushort(__float2half_rn(v));
}
__device__ __forceinline__ void xq_pen_span(unsigned short* __restrict__ L, int V, int span0,
                                            const unsigned int* __restrict__ P,
                                            const int* __restrict__ H, int rmask,
                                            int base, int first, const int* __restrict__ ext,
                                            int past_len, unsigned int* fmax, unsigned int* fsum) {
    const unsigned int flag = P[0];
    const float rep = __uint_as_float(P[1]);
    const float pres = __uint_as_float(P[2]);
    const float freq = __uint_as_float(P[3]);
    const int S = (int)P[4], D = (int)P[5];
    const int n = min(XQ_PEN_SPAN, V - span0);
    for (int j = threadIdx.x; j < XQ_PEN_SPAN; j += blockDim.x) { fmax[j] = 0u; fsum[j] = 0u; }
    __syncthreads();
    const int lo = max(0, past_len - S - D + 1);
    for (int i = lo + (int)threadIdx.x; i < past_len; i += blockDim.x) {
        const int t = i < base ? H[i & rmask] : (i == base ? first : ext[i - base - 1]);
        const int j = t - span0;
        if (j < 0 || j >= n) continue;
        const int dist = past_len - i;
        const unsigned int f = D > 0 ? (unsigned int)min(D, D + S - dist) : 1u;
        atomicMax(&fmax[j], f);
        atomicAdd(&fsum[j], f);
    }
    __syncthreads();
    const float scale = D > 0 ? (float)D : 1.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const unsigned int fm = fmax[j];
        if (fm == 0u) continue;
        L[span0 + j] = xq_pen_tok(__half2float(__ushort_as_half(L[span0 + j])), fm, fsum, j,
                                  flag, rep, pres, freq, scale);
    }
}

// Decode / verify rows. grid = (ceil(V / XQ_PEN_SPAN), m). Row r = (slot, pos) from the slot-pos
// table, input token toks[r] at pos. Rows of one slot at consecutive positions form a run (the
// verify: b, d0.. at p..p+k); row r's past = hist[..base) + toks[r0..=r] where r0 = the run's
// first row and base = its position — so verify row r sees exactly the drafts before it, i.e.
// the sequence a plain decode of the same prefix would see. Block x == 0 then commits toks[r]
// at pos into the ring: every reader of this launch reads positions < base <= pos, the ring holds
// >= window + verify width positions (host-checked), and a rejected draft's entry lies beyond the
// next round's base, so it is rewritten before any launch can read it.
extern "C" __global__ void __launch_bounds__(256) xq_pen_rows(
        unsigned short* __restrict__ logits, int V, const unsigned int* __restrict__ pen,
        const int* __restrict__ toks, const int* __restrict__ slots, int* __restrict__ hist, int rlog2) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.y;
    const unsigned int* P = pen + row * 8;
    if (P[0] == 0u) return;
    __shared__ unsigned int fmax[XQ_PEN_SPAN];
    __shared__ unsigned int fsum[XQ_PEN_SPAN];
    const int slot = slots[2 * row], pos = slots[2 * row + 1];
    int r0 = row;
    while (r0 > 0 && slots[2 * (r0 - 1)] == slot && slots[2 * (r0 - 1) + 1] == slots[2 * r0 + 1] - 1) r0--;
    const int base = slots[2 * r0 + 1];
    const int R = 1 << rlog2;
    int* H = hist + (long long)slot * R;
    xq_pen_span(logits + (long long)row * V, V, blockIdx.x * XQ_PEN_SPAN, P, H, R - 1,
                base, toks[r0], toks + r0 + 1, pos + 1, fmax, fsum);
    if (blockIdx.x == 0 && threadIdx.x == 0) H[pos & (R - 1)] = toks[row];
}

// Draft mirror (draft-only): the same adjustment on draft pass i's logits row (the pruned head's
// leading Vd ids). meta = {b, p, slot, 0} (sc.draft_meta); pass i predicts position p+1+i, so its
// past = hist[..p) + b (at p) + d_0..d_{i-1} (d_dev, the chain's own drafts). Params = row 0 (the
// lane). No commit: the verify commits the round's inputs.
extern "C" __global__ void __launch_bounds__(256) xq_pen_draft(
        unsigned short* __restrict__ logits, int Vd, const unsigned int* __restrict__ pen,
        const int* __restrict__ meta, const int* __restrict__ d_dev, int i,
        const int* __restrict__ hist, int rlog2) {
    XQ_PDL_ENTRY();
    if (pen[0] == 0u) return;
    __shared__ unsigned int fmax[XQ_PEN_SPAN];
    __shared__ unsigned int fsum[XQ_PEN_SPAN];
    const int b = meta[0], p = meta[1], slot = meta[2];
    const int R = 1 << rlog2;
    xq_pen_span(logits, Vd, blockIdx.x * XQ_PEN_SPAN, pen, hist + (long long)slot * R, R - 1,
                p, b, d_dev, p + 1 + i, fmax, fsum);
}

// =====================================================================
// WP24: exact speculative sampling with STOCHASTIC drafts (real-q). Opt-in
// (--spec-sampling ratio); nothing in this section runs on the default match path.
//
// Draft side (xq_rq_draft, one launch per draft pass, replaces that pass's argmax): the top-32
// of the pass's pruned-head logits (key desc, id asc — list[0] is xq_argmax_rows' id), then
// (mode 2) the proposal q' = that list truncated to the request's top-k, tempered by tau_d,
// cut to the smallest top-p prefix, renormalised: EXACTLY the distribution the pass samples
// its draft from (uniform = hash(seed, meta[3] + i, XQ_RQ_DOM_DRAFT)). (ids, logits, q') are
// stored per pass (one XQ_RQ_ROW-word row). Mode 1 = the list only (GB10_WP24_DUMP rung 0;
// the draft stays xq_argmax_rows' id).
// Verify side (xq_sample_rows_rq): the row's target p is xq_sample_rows' distribution
// (T -> top-k -> min-p -> top-p over the penalised logits, same code). Row i < nratio accepts
// d_i with probability min(1, p(d_i)/q'(d_i)); on reject it samples norm(max(0, p - q'))
// (only the listed ids are adjusted; d_i itself is 0) and the bonus row samples p. xq_accept
// runs unchanged (a = the first row whose emitted id != d_i), and
//   P(emit x) = min(p, q')(x) + (1 - sum min(p, q')) * max(0, p - q')(x) / sum max(0, p - q') = p(x)
// — distribution-exact against the plain sampler (speculative sampling, Leviathan / Chen).
// RNG domains: draft draw, accept test and residual draw are separate streams; the bonus /
// plain rows use domain 0 = xq_sample_rows' own draw (same counter -> the same id).
// w5 port (onto the DHEAD line): a real-q (mode 2) draft pass runs the FULL pruned-slice head
// (exl3_chain -> [xq_pen_draft] -> this kernel -> xq_rowmax_f16), not DHEAD's screen + rescore:
// DHEAD's candidate blocks are chosen to contain the ARGMAX (screen bound vs the best block), so
// they cannot guarantee the exact top-K. A5-L4 (spec.ratio_dhead=1, opt-in): unpenalized real-q
// passes run DHEAD + xq_rq_draft_dh instead — q' over the candidate set (exact for any q'). The dump (mode 1) keeps the served DHEAD/DHEADP draft
// and recomputes the slice logits only for its list. dconf (the DDS confidence) stays the pass's
// max draft logit (xq_rowmax_f16), never a function of the sampled draft.
// Known epsilon (float rounding only): a rejected ratio row whose residual mass R is 0 (p == q'
// to rounding) emits d, and a row with no target survivor keeps xq_argmax's id; xq_accept counts
// either as an accept. Both events have probability ~0.
// =====================================================================
#define XQ_RQ_K 32           // candidate list width (one warp)
#define XQ_RQ_NB_MAX 32      // stage-1 blocks per row
#define XQ_RQ_SLICE 8192     // max stage-1 slice (fp16 in smem, 16 KB)
#define XQ_RQ_ROW 96         // u32 words per list row: ids[32] | logits f32[32] | q' f32[32]
#define XQ_RQ_DOM_ACC 1u
#define XQ_RQ_DOM_RES 2u
#define XQ_RQ_DOM_DRAFT 3u
// uniform in (0,1) from (seed, counter, domain); dom 0 is bit-for-bit xq_sample_rows' draw.
__device__ __forceinline__ float xq_rq_u24(unsigned long long seed, unsigned int ctr, unsigned int dom) {
    return ((float)(xq_mix64(seed ^ xq_mix64(((unsigned long long)dom << 32) + (unsigned long long)ctr + 1ull)) >> 40)
            + 0.5f) * (1.0f / 16777216.0f);
}
// order key: larger = higher logit, ties -> lower id (xq_argmax_rows' comparator: -0 == +0);
// NaN -> 0 = "no candidate" (xq_argmax_rows never picks a NaN either).
__device__ __forceinline__ unsigned long long xq_rq_key(unsigned short b, int i) {
    if ((b & 0x7C00u) == 0x7C00u && (b & 0x03FFu) != 0u) return 0ull;
    if (b == 0x8000u) b = 0u;
    return ((unsigned long long)xq_skey(b) << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned)i);
}

// Warp 0 of a row's final top-32 (`top`: sorted desc, invalid = 0 trailing; lane j = candidate j)
// -> (mode 2) the proposal q' and the draft sampled from it (-> out[0], returned; counter
// meta[3] + ci), and the list row L = {ids | logits | q'}. A copy of xq_rq_draft's warp-0 tail, op
// for op (xq_rq_draft keeps its inline copy so its PTX stays byte-identical); used by xq_rq_draft_dh.
// Returns -1 when mode != 2.
__device__ __forceinline__ int xq_rq_emit(const unsigned long long* top, int lane, int mode,
        const unsigned int* __restrict__ dsamp, const int* __restrict__ meta, int ci,
        int* __restrict__ out, unsigned int* __restrict__ L) {
    int pick = -1;
    const unsigned long long kj = top[lane];
    const bool valid = kj != 0ull;
    int id = valid ? (int)(0xFFFFFFFFu - (unsigned)(kj & 0xFFFFFFFFull)) : -1;
    const float lg = valid ? xq_skey_val((unsigned)(kj >> 32)) : -INFINITY;
    float q = 0.0f;
    if (mode == 2) {
        const float tau = fmaxf(__uint_as_float(dsamp[1]), 1e-6f), invT = 1.0f / tau;
        const float top_p = __uint_as_float(dsamp[2]);
        const unsigned int top_k = dsamp[3];
        const unsigned long long seed = ((unsigned long long)dsamp[5] << 32) | (unsigned long long)dsamp[4];
        const unsigned int ctr = (unsigned int)meta[3] + (unsigned int)ci;
        const bool any = __shfl_sync(0xffffffffu, (int)valid, 0) != 0;
        pick = 0;
        if (!any) {
            // no finite-key candidate at all: the delta proposal q' = 1{0} (still exact — the
            // verify then runs the match rule's law against token 0)
            id = lane == 0 ? 0 : -1;
            q = lane == 0 ? 1.0f : 0.0f;
        } else {
            const float l0 = __shfl_sync(0xffffffffu, lg, 0);
            const bool keep = valid && (top_k == 0u || (unsigned)lane < top_k);
            float w = keep ? __expf((lg - l0) * invT) : 0.0f;
            float tot = w;
            for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (top_p < 1.0f && tot > 0.0f) {
                // smallest prefix whose mass reaches top_p * tot: keep j iff mass(0..j-1) < target
                float incl = w;
                for (int o = 1; o < 32; o <<= 1) {
                    const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += nn;
                }
                float excl = __shfl_up_sync(0xffffffffu, incl, 1);
                if (lane == 0) excl = 0.0f;
                if (!(excl < top_p * tot)) w = 0.0f;
            }
            float z = w;
            for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
            if (!(z > 0.0f && z < INFINITY)) {               // non-finite list: delta on candidate 0
                w = lane == 0 ? 1.0f : 0.0f;
                z = 1.0f;
            }
            q = w / z;
            // inverse CDF over the kept prefix in list order (warp-uniform, deterministic)
            const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_DRAFT) * z;
            float incl = w;
            for (int o = 1; o < 32; o <<= 1) {
                const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += nn;
            }
            const unsigned int hit = __ballot_sync(0xffffffffu, w > 0.0f && incl > target);
            const unsigned int pos = __ballot_sync(0xffffffffu, w > 0.0f);
            const int pl = hit ? __ffs(hit) - 1 : 31 - __clz(pos);
            pick = __shfl_sync(0xffffffffu, id, pl);
        }
        if (lane == 0) out[0] = pick;
    }
    L[lane] = (unsigned int)id;
    L[32 + lane] = __float_as_uint(lg);
    L[64 + lane] = __float_as_uint(q);
    return pick;
}

// grid (nb <= XQ_RQ_NB_MAX, rows), block 256. Row r = logits row r (stride V); its list goes to
// list + (i0 + r) * XQ_RQ_ROW and (mode 2) its sampled id to out_ids[r], counter meta[3] + i0 + r.
// Stage 1: block b stages its slice in smem and extracts its top-32 by 32 rounds of (block max
// of the per-thread best key -> the owner drops it and rescans its strided elements); stage 2:
// the LAST block of the row (threadfence + per-row counter, self re-arming like xq_argmax_rows)
// merges the nb block lists the same way — exact: keys are distinct (the id is in the low word)
// and every global top-32 key is in its block's top-32. Deterministic: the order is total.
extern "C" __global__ void __launch_bounds__(256) xq_rq_draft(
        int* __restrict__ out_ids, const unsigned short* __restrict__ logits, int V,
        unsigned long long* __restrict__ part, unsigned int* __restrict__ cnt,
        unsigned int* __restrict__ list, const unsigned int* __restrict__ dsamp,
        const int* __restrict__ meta, int i0, int mode) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned short sv[XQ_RQ_SLICE];   // stage 1 slice; stage 2 candidates (u64)
    __shared__ unsigned long long wk[2][32];
    __shared__ unsigned long long top[XQ_RQ_K];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const unsigned short* row = logits + (long long)r * V;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;     // host: chunk <= XQ_RQ_SLICE
    const int lo = min(b * chunk, V), hi = min(lo + chunk, V), n = hi - lo;
    // mode 2 needs only the first top_k candidates (q' truncates there): extract that many
    const unsigned int tk = mode == 2 ? dsamp[3] : 0u;
    const int kr = (tk > 0u && tk < (unsigned int)XQ_RQ_K) ? (int)tk : XQ_RQ_K;
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    if ((reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull) {
        const int nv = n >> 3;                              // lo is 8-aligned: row + lo stays 16-B aligned
        const uint4* rv = reinterpret_cast<const uint4*>(row + lo);
        uint4* sv4 = reinterpret_cast<uint4*>(sv);
        for (int q = tid; q < nv; q += bs) sv4[q] = __ldg(rv + q);
        for (int j = nv * 8 + tid; j < n; j += bs) sv[j] = row[lo + j];
    } else {
        for (int j = tid; j < n; j += bs) sv[j] = row[lo + j];
    }
    __syncthreads();
    unsigned long long own = 0ull;
    for (int j = tid; j < n; j += bs) { const unsigned long long x = xq_rq_key(sv[j], lo + j); own = x > own ? x : own; }
    for (int it = 0; it < kr; it++) {
        // double-buffered warp partials: one barrier per round (a thread can only rewrite
        // wk[it & 1] after every thread passed round it + 1's barrier, i.e. finished reading it)
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;                               // block-uniform (all read the same partials)
        if (tid == 0) top[it] = k;
        if (own == k) {                                     // keys are unique: exactly one owner
            sv[(int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull)) - lo] = 0xFFFFu;   // -NaN: dropped
            own = 0ull;
            for (int j = tid; j < n; j += bs) { const unsigned long long x = xq_rq_key(sv[j], lo + j); own = x > own ? x : own; }
        }
    }
    __syncthreads();
    if (tid < XQ_RQ_K) part[((long long)r * nb + b) * XQ_RQ_K + tid] = top[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
    __syncthreads();
    if (!last) return;
    __threadfence();
    unsigned long long* cand = reinterpret_cast<unsigned long long*>(sv);    // nb * 32 <= 1024 keys (8 KB)
    const int nc = nb * XQ_RQ_K;
    for (int j = tid; j < nc; j += bs) cand[j] = __ldcg(part + (long long)r * nb * XQ_RQ_K + j);
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    __syncthreads();
    own = 0ull;
    for (int j = tid; j < nc; j += bs) own = cand[j] > own ? cand[j] : own;
    for (int it = 0; it < kr; it++) {
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;
        if (tid == 0) top[it] = k;
        if (own == k) {
            own = 0ull;
            for (int j = tid; j < nc; j += bs) {
                if (cand[j] == k) cand[j] = 0ull;
                own = cand[j] > own ? cand[j] : own;
            }
        }
    }
    __syncthreads();
    if (tid == 0) cnt[r] = 0u;                              // re-arm for the next launch / replay
    if (wid != 0) return;
    // warp 0: lane j = candidate j (sorted desc; invalid entries trail)
    const unsigned long long kj = top[lane];
    const bool valid = kj != 0ull;
    int id = valid ? (int)(0xFFFFFFFFu - (unsigned)(kj & 0xFFFFFFFFull)) : -1;
    const float lg = valid ? xq_skey_val((unsigned)(kj >> 32)) : -INFINITY;
    float q = 0.0f;
    if (mode == 2) {
        const float tau = fmaxf(__uint_as_float(dsamp[1]), 1e-6f), invT = 1.0f / tau;
        const float top_p = __uint_as_float(dsamp[2]);
        const unsigned int top_k = dsamp[3];
        const unsigned long long seed = ((unsigned long long)dsamp[5] << 32) | (unsigned long long)dsamp[4];
        const unsigned int ctr = (unsigned int)meta[3] + (unsigned int)(i0 + r);
        const bool any = __shfl_sync(0xffffffffu, (int)valid, 0) != 0;
        int pick = 0;
        if (!any) {
            // no finite-key candidate at all: the delta proposal q' = 1{0} (still exact — the
            // verify then runs the match rule's law against token 0)
            id = lane == 0 ? 0 : -1;
            q = lane == 0 ? 1.0f : 0.0f;
        } else {
            const float l0 = __shfl_sync(0xffffffffu, lg, 0);
            const bool keep = valid && (top_k == 0u || (unsigned)lane < top_k);
            float w = keep ? __expf((lg - l0) * invT) : 0.0f;
            float tot = w;
            for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
            if (top_p < 1.0f && tot > 0.0f) {
                // smallest prefix whose mass reaches top_p * tot: keep j iff mass(0..j-1) < target
                float incl = w;
                for (int o = 1; o < 32; o <<= 1) {
                    const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += nn;
                }
                float excl = __shfl_up_sync(0xffffffffu, incl, 1);
                if (lane == 0) excl = 0.0f;
                if (!(excl < top_p * tot)) w = 0.0f;
            }
            float z = w;
            for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
            if (!(z > 0.0f && z < INFINITY)) {               // non-finite list: delta on candidate 0
                w = lane == 0 ? 1.0f : 0.0f;
                z = 1.0f;
            }
            q = w / z;
            // inverse CDF over the kept prefix in list order (warp-uniform, deterministic)
            const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_DRAFT) * z;
            float incl = w;
            for (int o = 1; o < 32; o <<= 1) {
                const float nn = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += nn;
            }
            const unsigned int hit = __ballot_sync(0xffffffffu, w > 0.0f && incl > target);
            const unsigned int pos = __ballot_sync(0xffffffffu, w > 0.0f);
            const int pl = hit ? __ffs(hit) - 1 : 31 - __clz(pos);
            pick = __shfl_sync(0xffffffffu, id, pl);
        }
        if (lane == 0) out_ids[r] = pick;
    }
    unsigned int* L = list + (long long)(i0 + r) * XQ_RQ_ROW;
    L[lane] = (unsigned int)id;
    L[32 + lane] = __float_as_uint(lg);
    L[64 + lane] = __float_as_uint(q);
}

// A5-L4 (spec.ratio_dhead, opt-in): a real-q (mode 2) draft pass on DHEAD instead of the full
// slice head. Launched right after xq_dh_rescore of pass i (same stream, one CTA): re-derives the
// ns x 128 exact candidate logits from the rescore's fp32 split-K partials with ITS step-3 ops
// (ascending-s sum -> f16 yraw -> had128 -> x svh -> f16 RN; the same values the rescore's argmax
// compared), takes their top-kr with xq_rq_draft's keys and order, and runs xq_rq_draft's mode-2
// tail (xq_rq_emit): q' = the request's top-k -> tau -> top-p law over DHEAD's CANDIDATE set, the
// draft ~ q' (counter meta[3] + i, domain XQ_RQ_DOM_DRAFT), list row i. Tokens outside the
// candidate blocks get q' = 0: they are never drafted, and the verify (xq_sample_rows_rq) reads
// q' only through list row i, so its residual max(0, p - q') is p there — speculative sampling
// is exact for ANY q' as long as the draft is drawn from the q' the verify reads (this row).
// Overwrites the rescore's argmax / d_dev[i] / toks[0] with the sampled draft; dconf[i] stays
// the rescore's in-candidate max (the DDS stop reads only draft-side quantities).
// geom = xq_dh_rescore's (pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32).
#define XQ_RQDH_MAX (DH_TCAP * 128)
extern "C" __global__ void __launch_bounds__(256) xq_rq_draft_dh(
        int* __restrict__ out_id, int* __restrict__ d_dev, int* __restrict__ toks,
        const __half* __restrict__ svh, const unsigned* __restrict__ ws,
        unsigned int* __restrict__ list, const unsigned int* __restrict__ dsamp,
        const int* __restrict__ meta, long long geom, int N) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned short sv[XQ_RQDH_MAX];
    __shared__ int s_cand[DH_TCAP];
    __shared__ unsigned long long wk[2][32];
    __shared__ unsigned long long top[XQ_RQ_K];
    const int pass_i = (int)(geom & 0xFF), S = (int)((geom >> 16) & 0xFF), K = (int)(geom >> 32);
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;
    const int nblk = N >> 7;
    const int* dbg = reinterpret_cast<const int*>(ws + dh_off_dbg(K, nblk));
    const float* part = reinterpret_cast<const float*>(ws + dh_off_part(K, nblk));
    const int ns = max(min(__ldcg(dbg), DH_TCAP), 0);
    if (tid < ns) s_cand[tid] = __ldcg(dbg + 1 + tid);
    if (tid < XQ_RQ_K) top[tid] = 0ull;
    __syncthreads();
    // 1. the candidate logits (xq_dh_rescore step 3, op for op) -> sv[rr * 128 + col]
    for (int rr = wid; rr < ns; rr += nw) {
        const int bb = s_cand[rr];
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* pc = part + (size_t)rr * S * 128 + lane * 4 + q;
            float a = 0.f;
            for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
            v[q] = __half2float(__float2half_rn(a));
        }
        exl3_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = bb * 128 + lane * 4 + q;
            sv[rr * 128 + lane * 4 + q] = __half_as_ushort(__float2half_rn(v[q] * __half2float(svh[col])));
        }
    }
    __syncthreads();
    // 2. top-kr (xq_rq_draft stage 1 over the candidate positions; key = (logit, lower id))
    const unsigned int tk = dsamp[3];
    const int kr = (tk > 0u && tk < (unsigned int)XQ_RQ_K) ? (int)tk : XQ_RQ_K;
    const int n = ns * 128;
    unsigned long long own = 0ull;
    int oj = -1;
    for (int j = tid; j < n; j += bs) {
        const unsigned long long x = xq_rq_key(sv[j], s_cand[j >> 7] * 128 + (j & 127));
        if (x > own) { own = x; oj = j; }
    }
    for (int it = 0; it < kr; it++) {
        unsigned long long k = xq_am_wmax(own);
        if (lane == 0) wk[it & 1][wid] = k;
        __syncthreads();
        k = 0ull;
        for (int w = 0; w < nw; w++) { const unsigned long long x = wk[it & 1][w]; k = x > k ? x : k; }
        if (k == 0ull) break;                               // block-uniform
        if (tid == 0) top[it] = k;
        if (own == k) {                                     // keys are unique: exactly one owner
            sv[oj] = 0xFFFFu;                               // -NaN: dropped
            own = 0ull; oj = -1;
            for (int j = tid; j < n; j += bs) {
                const unsigned long long x = xq_rq_key(sv[j], s_cand[j >> 7] * 128 + (j & 127));
                if (x > own) { own = x; oj = j; }
            }
        }
    }
    __syncthreads();
    if (wid != 0) return;
    // 3. q' + the draft + list row pass_i (xq_rq_draft's mode-2 tail)
    const int pick = xq_rq_emit(top, lane, 2, dsamp, meta, pass_i, out_id, list + (long long)pass_i * XQ_RQ_ROW);
    if (lane == 0) { d_dev[pass_i] = pick; toks[0] = pick; }
}

// residual weight of token i (plain weight w > 0 = a target survivor): d itself 0; a listed id
// max(0, w - q'(i) * tot) (= tot * max(0, p - q')); any other survivor w. The bloom filter
// (1024 bits over the <= 32 listed ids) keeps the list scan off almost every element.
__device__ __forceinline__ float xq_rq_resw(int i, float w, int d, float tot, const int* l_id,
                                            const float* l_q, const unsigned int* bloom) {
    if (w <= 0.0f || i == d) return 0.0f;
    if ((bloom[(i >> 5) & 31] >> (i & 31)) & 1u) {
        for (int j = 0; j < XQ_RQ_K; j++)
            if (l_id[j] == i) return fmaxf(0.0f, w - __fmul_rn(l_q[j], tot));   // no FMA: both passes agree
    }
    return w;
}

// WP24 verify sampler: xq_sample_rows (steps 1-5 below are its code verbatim) + the real-q
// rule on rows r < nratio staged with flag 2 (the draft d_r = d_vec[r], its proposal = list row
// r). Every other sampled row — the bonus row, flag-1 rows — is xq_sample_rows exactly (same
// draw, same id). Launched only by real-q verify graphs (never on the default path).
extern "C" __global__ void __launch_bounds__(1024) xq_sample_rows_rq(
        int* __restrict__ out_ids, const __half* __restrict__ logits, int V,
        const unsigned int* __restrict__ samp, const int* __restrict__ d_vec,
        const unsigned int* __restrict__ list, int nratio) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.x;
    const unsigned int* P = samp + row * 8;
    if (P[0] == 0u) return;
    const float T = fmaxf(__uint_as_float(P[1]), 1e-6f);
    const float invT = 1.0f / T;
    const float top_p = __uint_as_float(P[2]);
    const unsigned int top_k = P[3];
    const unsigned long long seed = ((unsigned long long)P[5] << 32) | (unsigned long long)P[4];
    const unsigned int ctr = P[6];
    const float min_p = __uint_as_float(P[7]);
    const unsigned short* L = reinterpret_cast<const unsigned short*>(logits) + (long long)row * V;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;

    __shared__ unsigned int ush[32];
    __shared__ float fsh[32];
    __shared__ unsigned int hist[256];
    __shared__ float fh[256];
    __shared__ unsigned int s_u[4];
    __shared__ float s_f[2];
    __shared__ int l_id[XQ_RQ_K];
    __shared__ float l_q[XQ_RQ_K];
    __shared__ unsigned int bloom[32];
    __shared__ int s_d, s_acc;
    __shared__ float s_tot;

    // 1. max key -> lmax (softmax shift)
    unsigned int mk = 0u;
    for (int i = tid; i < V; i += bs) mk = max(mk, xq_skey(L[i]));
    mk = xq_bmax_u32(mk, ush);
    const float lmax = xq_skey_val(mk);

    // 2. top-k key threshold tau (radix select over the 16-bit key, ties included)
    unsigned int tau = 0u;
    if (top_k > 0u && top_k < (unsigned int)V) {
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) atomicAdd(&hist[xq_skey(L[i]) >> 8], 1u);
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, hb = 0u, need = 1u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= top_k) { hb = (unsigned int)b; need = top_k - cum; break; }
                cum += hist[b];
            }
            s_u[0] = hb; s_u[1] = need;
        }
        __syncthreads();
        const unsigned int hb = s_u[0], need = s_u[1];
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if ((k >> 8) == hb) atomicAdd(&hist[k & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, lb = 0u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= need) { lb = (unsigned int)b; break; }
                cum += hist[b];
            }
            s_u[2] = (hb << 8) | lb;
        }
        __syncthreads();
        tau = s_u[2];
    }
    const float mfloor = (min_p > 0.0f) ? lmax + T * __logf(min_p) : -INFINITY;

    // 3. candidate mass S (deterministic)
    float part = 0.0f;
    for (int i = tid; i < V; i += bs) {
        unsigned int k = xq_skey(L[i]);
        if (k >= tau) {
            float x = xq_skey_val(k);
            if (x >= mfloor) part += __expf((x - lmax) * invT);
        }
    }
    const float S = xq_bsum_f32(part, fsh);

    // 4. top-p key threshold theta (mass radix select; boundary ties included)
    unsigned int thr = tau;
    if (top_p < 1.0f && S > 0.0f) {
        const float target = top_p * S;
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k >> 8], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int hb = 0u; float rem = 0.0f; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= target) { hb = (unsigned int)b; rem = target - cum; found = 1; break; }
                cum += fh[b];
            }
            if (!found) { hb = (unsigned int)max(last, 0); rem = 3.0e38f; }
            s_u[0] = hb; s_f[0] = rem;
        }
        __syncthreads();
        const unsigned int hb = s_u[0];
        const float rem = s_f[0];
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau && (k >> 8) == hb) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k & 255u], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int lb = 0u; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= rem) { lb = (unsigned int)b; found = 1; break; }
                cum += fh[b];
            }
            if (!found) lb = (unsigned int)max(last, 0);
            s_u[3] = max((hb << 8) | lb, tau);
        }
        __syncthreads();
        thr = s_u[3];
    }

    // 5. per-warp survivor mass of p: warp w owns [w*chunk, (w+1)*chunk), lanes stride 32.
    const int chunk = (V + nw - 1) / nw;
    const int w0 = wid * chunk, w1 = min(V, w0 + chunk);
    float lsum = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) lsum += __expf((x - lmax) * invT);
        }
    }
    for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) fsh[wid] = lsum;
    const bool ratio = P[0] == 2u && row < nratio;          // block-uniform
    if (ratio && tid < XQ_RQ_K) {
        const unsigned int* Lr = list + (long long)row * XQ_RQ_ROW;
        l_id[tid] = (int)Lr[tid];
        l_q[tid] = __uint_as_float(Lr[64 + tid]);
        bloom[tid] = 0u;
    }
    __syncthreads();
    if (!ratio) {
        // plain row (the bonus row, flag-1 rows): xq_sample_rows' inverse CDF, verbatim
        if (tid == 0) {
            const float u = ((float)(xq_mix64(seed ^ xq_mix64((unsigned long long)ctr + 1ull)) >> 40) + 0.5f)
                            * (1.0f / 16777216.0f);
            float tot = 0.0f;
            for (int w = 0; w < nw; w++) tot += fsh[w];
            const float target = u * tot;
            float cum = 0.0f; int sel = -1, last = 0;
            for (int w = 0; w < nw; w++) {
                if (fsh[w] > 0.0f) last = w;
                if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
                cum += fsh[w];
            }
            if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
            s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
        }
        __syncthreads();
        if (wid == (int)s_u[0]) {
            const float want = s_f[0];
            float base = 0.0f;
            int pick = -1, lastidx = -1;
            for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
                const int i = i0 + lane;
                float wgt = 0.0f;
                if (i < w1) {
                    unsigned int k = xq_skey(L[i]);
                    if (k >= thr) {
                        float x = xq_skey_val(k);
                        if (x >= mfloor) wgt = __expf((x - lmax) * invT);
                    }
                }
                float incl = wgt;
                for (int o = 1; o < 32; o <<= 1) {
                    float n = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += n;
                }
                unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
                unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
                if (any) lastidx = i0 + 31 - __clz(any);
                if (hit) pick = i0 + __ffs(hit) - 1;
                base += __shfl_sync(0xffffffffu, incl, 31);
            }
            if (lane == 0 && (pick >= 0 || lastidx >= 0)) out_ids[row] = (pick >= 0) ? pick : lastidx;
        }
        return;
    }

    // 6. ratio row: accept d with probability min(1, p(d) / q'(d)) <=> u * q'(d) * tot < w(d)
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < nw; w++) tot += fsh[w];
        const int d = d_vec[row];
        float qd = 0.0f;
        for (int j = 0; j < XQ_RQ_K; j++) if (l_id[j] == d && l_q[j] > 0.0f) { qd = l_q[j]; break; }
        if (!(qd > 0.0f)) {
            // d outside its pass's list (a pass always samples from its list, so only a stale /
            // foreign list lands here): the delta proposal 1{d} — exactly the match rule's law
            for (int j = 0; j < XQ_RQ_K; j++) { l_id[j] = -1; l_q[j] = 0.0f; }
            l_id[0] = d; l_q[0] = 1.0f; qd = 1.0f;
        }
        float wd = 0.0f;
        if (d >= 0 && d < V) {
            unsigned int k = xq_skey(L[d]);
            if (k >= thr) {
                float x = xq_skey_val(k);
                if (x >= mfloor) wd = __expf((x - lmax) * invT);
            }
        }
        const float ua = xq_rq_u24(seed, ctr, XQ_RQ_DOM_ACC);
        s_acc = (tot > 0.0f && ua * qd * tot < wd) ? 1 : 0;
        s_d = d; s_tot = tot;
    }
    __syncthreads();
    const float tot = s_tot;
    const int d = s_d;
    if (!(tot > 0.0f)) return;                              // no survivor at all: keep xq_argmax's id
    if (s_acc) { if (tid == 0) out_ids[row] = d; return; }
    if (tid < XQ_RQ_K && l_id[tid] >= 0) atomicOr(&bloom[(l_id[tid] >> 5) & 31], 1u << (l_id[tid] & 31));
    __syncthreads();
    // 7. reject: sample norm(max(0, p - q')) — per-warp residual mass, then the same inverse CDF
    float rs = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) rs += xq_rq_resw(i, __expf((x - lmax) * invT), d, tot, l_id, l_q, bloom);
        }
    }
    for (int o = 16; o; o >>= 1) rs += __shfl_xor_sync(0xffffffffu, rs, o);
    if (lane == 0) fsh[wid] = rs;                           // tid 0 read fsh before the last barrier
    __syncthreads();
    if (tid == 0) {
        float R = 0.0f;
        for (int w = 0; w < nw; w++) R += fsh[w];
        const float target = xq_rq_u24(seed, ctr, XQ_RQ_DOM_RES) * R;
        float cum = 0.0f; int sel = -1, last = 0;
        for (int w = 0; w < nw; w++) {
            if (fsh[w] > 0.0f) last = w;
            if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
            cum += fsh[w];
        }
        if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
        s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
        // R == 0: p == q' up to rounding, so a reject had probability ~0 — emit d (the limit law)
        s_u[1] = R > 0.0f ? 1u : 0u;
    }
    __syncthreads();
    if (s_u[1] == 0u) { if (tid == 0) out_ids[row] = d; return; }
    if (wid == (int)s_u[0]) {
        const float want = s_f[0];
        float base = 0.0f;
        int pick = -1, lastidx = -1;
        for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
            const int i = i0 + lane;
            float wgt = 0.0f;
            if (i < w1) {
                unsigned int k = xq_skey(L[i]);
                if (k >= thr) {
                    float x = xq_skey_val(k);
                    if (x >= mfloor) wgt = xq_rq_resw(i, __expf((x - lmax) * invT), d, tot, l_id, l_q, bloom);
                }
            }
            float incl = wgt;
            for (int o = 1; o < 32; o <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
            unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
            if (any) lastidx = i0 + 31 - __clz(any);
            if (hit) pick = i0 + __ffs(hit) - 1;
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) out_ids[row] = (pick >= 0) ? pick : (lastidx >= 0 ? lastidx : d);
    }
}

// =============================================================================
// DHEADP (w3, 2026-09-27): DHEAD for PENALIZED draft passes (the WP15 draft mirror) — DRAFT-ONLY
// (the verify, its penalties and every emitted token are untouched; only which token is drafted
// can change). The old penalized draft tail ran the FULL 65,536-id slice head (exl3_chain) so
// xq_pen_draft could rewrite the window's tokens before argmax_rows + xq_rowmax_f16.
//
// Bound. Under the served semantics (xq_pen_tok: rep >= 1 divides positive / multiplies negative
// logits, pres >= 0 and freq >= 0 subtract) a penalty never RAISES a logit v >= -65504 (only -inf
// rises, to -65504). So for a block b the screen's unpenalized upper bound
//     UB_b = bmax[b] + delta * sigma * max|svh_b|     (DHEAD's per-block screen-noise margin)
// also bounds every PENALIZED logit in b. Pass A rescores DHEAD's candidate set C1 exactly (the
// same selection, partials and combine as xq_dh_rescore), penalizes those logits with the draft
// mirror's per-token formula and counts, and takes their best penalized value L — a value some
// token really has, so a LOWER bound on the penalized maximum. A block outside C1 can hold the
// penalized argmax (or tie it with a lower id) only if UB_b >= L: pass B rescores exactly those
// blocks (C2), penalizes them, and folds them into the max. Blocks with UB_b < L cannot win.
// So the draft equals the old penalized head's draft whenever DHEAD's own screen-noise model holds
// for the blocks it excludes (the same assumption the unpenalized DHEAD makes; XCHECK measures it).
//
// Fallback (device-side, graph-safe): if more than `cap` blocks qualify, or the params break the
// monotonicity (rep < 1, pres < 0, freq < 0 — the host routes those requests to the old head, this
// is the belt-and-braces), or L is degenerate (NaN / <= -65504), C2 = EVERY block outside C1: the
// pass computes all N logits exactly, penalized — the old head's function, independent of the
// screen — at about the old head's cost. Counted per reason in pws (per-request log line).
// A row with pen[0] == 0 is unpenalized: C2 is empty and the result is xq_dh_rescore's, bit for bit.
// DETERMINISM: every value is a fixed-order computation (ascending-s split combine per candidate,
// integer window counts); the fold is a max over (value, id) keys — order-free.
//
// pws (u32 words; mirrored by src/exl3_forward/dhead.rs pws_layout):
#define DHP_BEST 0   // u64: running penalized best key (xq_am_key order)
#define DHP_N2 2     // pass-B candidate count (0 = C1 decided the pass)
#define DHP_DONE 3   // pass-B finished-candidate counter (self re-arming)
#define DHP_META 4   // {b, p, slot, 0}: pass A's copy of draft_meta
#define DHP_PEN 8    // row-0 penalty params (8 words): pass A's copy
#define DHP_DBG 16   // {ns1, n2, flags, L bits, nq, qualifying, K1 lo, K1 hi} (XCHECK readback)
#define DHP_STAT 24  // cumulative {passes, expanded, sum n2 expanded, fb overflow, fb non-monotone,
                     //             fb degenerate, sum n2 fallback, unpenalized-row passes}
#define DHP_HDR 32   // then cntc[nblk4] (per-candidate split counters), c2[nblk4], part[nblk][S][128]
#define DHP_F_OVER 1u
#define DHP_F_NONMONO 2u
#define DHP_F_DEGEN 4u
#define DHP_F_UNPEN 8u
#define DHP_F_FB 16u
__device__ __forceinline__ int dhp_nblk4(int nblk) { return (nblk + 3) & ~3; }

struct DhSel {                                       // dh_select's shared scratch
    unsigned om[8];
    int nq[8];
    int bs[2][8];
    int wh[DH_KPT * 8];
    int we[DH_KPT * 8];
    int ph[DH_KPT * 8];
    int pe[DH_KPT * 8];
    int tot[2];
    int cand[DH_TCAP];
};

// xq_dh_rescore's step 1, op for op (xq_dh_rescore itself is left byte-identical, so the
// unpenalized served PTX cannot move): the candidate blocks -> sm.cand[0..ns) (qualified above tau
// by ascending id, then the tau ties), returns ns; nq_out = the qualified count. 256 threads.
__device__ __forceinline__ int dh_select(const float* __restrict__ bmax, const float* __restrict__ bsig,
                                         int nblk, float sig, float delta, int tmax, DhSel& sm, int& nq_out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float bsmax = bsig[nblk];
    unsigned ob[DH_KPT];
    unsigned om = 0u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        float v = -INFINITY;
        if (b < nblk) { v = bmax[b]; if (!(v == v)) v = -INFINITY; }
        ob[i] = b < nblk ? dh_ord(v) : 0u;
        om = max(om, ob[i]);
    }
    om = __reduce_max_sync(0xffffffffu, om);
    if (lane == 0) sm.om[warp] = om;
    __syncthreads();
    unsigned go = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) go = max(go, sm.om[w]);
    const float gmax = dh_unord(go);
    unsigned qm = 0u;
    int qc = 0;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        if (b < nblk) {
            const float marg = fmaxf(delta * sig * (bsig[b] + bsmax), 0.f);   // NaN -> 0
            if (dh_unord(ob[i]) + marg >= gmax) { qm |= 1u << i; ++qc; }
        }
    }
    qc = __reduce_add_sync(0xffffffffu, qc);
    if (lane == 0) sm.nq[warp] = qc;
    __syncthreads();
    int nq = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nq += sm.nq[w];
    unsigned hm = qm, em = 0u;
    if (nq > tmax) {                                  // uniform branch
        unsigned tau = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned t = tau | (1u << bit);
            int c = 0;
            #pragma unroll
            for (int i = 0; i < DH_KPT; ++i) c += (((qm >> i) & 1u) && ob[i] >= t) ? 1 : 0;
            c = __reduce_add_sync(0xffffffffu, c);
            if (lane == 0) sm.bs[bit & 1][warp] = c;
            __syncthreads();
            int tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += sm.bs[bit & 1][w];
            if (tot >= tmax) tau = t;
        }
        hm = 0u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            if ((qm >> i) & 1u) {
                if (ob[i] > tau) hm |= 1u << i;
                else if (ob[i] == tau) em |= 1u << i;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
        const unsigned be = __ballot_sync(0xffffffffu, (em >> i) & 1u);
        if (lane == 0) { sm.wh[i * 8 + warp] = __popc(bh); sm.we[i * 8 + warp] = __popc(be); }
    }
    __syncthreads();
    if (tid < 2 * DH_KPT * 8) {
        const int e = tid & (DH_KPT * 8 - 1);
        const int* src = tid < DH_KPT * 8 ? sm.wh : sm.we;
        int pfx = 0;
        for (int f = 0; f < e; ++f) pfx += src[f];
        (tid < DH_KPT * 8 ? sm.ph : sm.pe)[e] = pfx;
        if (e == DH_KPT * 8 - 1) sm.tot[tid < DH_KPT * 8 ? 0 : 1] = pfx + src[e];
    }
    __syncthreads();
    const int nh = sm.tot[0];
    const int ns = min(min(nh + sm.tot[1], tmax), DH_TCAP);
    {
        const unsigned lt = (1u << lane) - 1u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const unsigned mh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
            const unsigned me = __ballot_sync(0xffffffffu, (em >> i) & 1u);
            const int b = tid + 256 * i;
            if ((hm >> i) & 1u) {
                const int rk = sm.ph[i * 8 + warp] + __popc(mh & lt);
                if (rk < ns) sm.cand[rk] = b;
            }
            if ((em >> i) & 1u) {
                const int rk = nh + sm.pe[i * 8 + warp] + __popc(me & lt);
                if (rk < ns) sm.cand[rk] = b;
            }
        }
    }
    __syncthreads();
    nq_out = nq;
    return ns;
}

// Ascending-id compaction of a per-thread block mask (bit i <-> block tid + 256 i): out[rank] = the
// block (global stores); returns the count. s_w / s_p: DH_KPT * 8 ints of smem; 256 threads.
__device__ __forceinline__ int dh_rank1(unsigned m, int* s_w, int* s_p, int* s_tot, unsigned* out) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bm = __ballot_sync(0xffffffffu, (m >> i) & 1u);
        if (lane == 0) s_w[i * 8 + warp] = __popc(bm);
    }
    __syncthreads();
    if (tid < DH_KPT * 8) {
        int pfx = 0;
        for (int f = 0; f < tid; ++f) pfx += s_w[f];
        s_p[tid] = pfx;
        if (tid == DH_KPT * 8 - 1) *s_tot = pfx + s_w[tid];
    }
    __syncthreads();
    const unsigned lt = (1u << lane) - 1u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bm = __ballot_sync(0xffffffffu, (m >> i) & 1u);
        if ((m >> i) & 1u) out[s_p[i * 8 + warp] + __popc(bm & lt)] = (unsigned)(tid + 256 * i);
    }
    return *s_tot;
}

// xq_dh_rescore's step 2, op for op: the exact split-K partial of candidate block cb over k16 rows
// [kb0, kb0 + R). The caller has issued the s_x2 loads (R * 8 words of xh); the __syncthreads here
// publishes them with the ring. The caller syncs before s_ring4 is reused.
__device__ __forceinline__ void dh_part(const uint16_t* __restrict__ tr, int cb, int kb0, int R, int nb16,
                                        int bits, uint4* s_ring4, const unsigned* s_x2, float (&acc)[2][4]) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int U4 = 2 * bits;
    const int rowu4 = 8 * U4;
    for (int idx = tid; idx < R * rowu4; idx += 256) {
        const int row = idx / rowu4, rem = idx - row * rowu4;
        const int w = rem / U4, q = rem - w * U4;
        const uint4* src = reinterpret_cast<const uint4*>(tr + ((size_t)(kb0 + row) * nb16 + (size_t)cb * 8) * (16 * bits));
        s_ring4[(w * R + row) * U4 + q] = __ldcs(src + rem);
    }
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    const uint32_t* ringw = reinterpret_cast<const uint32_t*>(s_ring4 + warp * R * U4);
    switch (bits) {
        case 2: dh_rescore_mma<2>(ringw, s_x2, R, lane, acc); break;
        case 3: dh_rescore_mma<3>(ringw, s_x2, R, lane, acc); break;
        case 4: dh_rescore_mma<4>(ringw, s_x2, R, lane, acc); break;
        case 5: dh_rescore_mma<5>(ringw, s_x2, R, lane, acc); break;
        case 6: dh_rescore_mma<6>(ringw, s_x2, R, lane, acc); break;
        case 7: dh_rescore_mma<7>(ringw, s_x2, R, lane, acc); break;
        case 8: dh_rescore_mma<8>(ringw, s_x2, R, lane, acc); break;
        default: break;
    }
}
// row 0 of the partial's two n8 tiles -> part_cs[128], then a device-scope fence (step 2's tail)
__device__ __forceinline__ void dh_part_store(float* __restrict__ part_cs, const float (&acc)[2][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if ((lane >> 2) == 0) {                           // cols 16w + 8h + 2tig
        float* pp = part_cs + warp * 16 + 2 * (lane & 3);
        pp[0] = acc[0][0]; pp[1] = acc[0][1]; pp[8] = acc[1][0]; pp[9] = acc[1][1];
        __threadfence();
    }
}

// The draft-mirror window of pass i (xq_pen_draft's past, xq_pen_span's factors): positions q in
// [max(0, past - S - D + 1), past), past = p + 1 + i; token(q) = H[q & rmask] (q < p), b (q == p),
// d_dev[q - p - 1] (q > p); ids outside the draft slice [0, N) are never penalized there.
// slot_of(t) = the smem counter index of t, or -1 (untracked). Integer atomics: order-free.
template <class SlotOf>
__device__ __forceinline__ void dhp_scan(const unsigned* __restrict__ P, const int* __restrict__ H, int rmask,
                                         int p, int b, const int* __restrict__ d_dev, int past, int N,
                                         SlotOf slot_of, unsigned* fm, unsigned* fs) {
    const int S = (int)P[4], D = (int)P[5];
    const int lo = max(0, past - S - D + 1);
    for (int q = lo + (int)threadIdx.x; q < past; q += blockDim.x) {
        const int t = q < p ? H[q & rmask] : (q == p ? b : d_dev[q - p - 1]);
        if (t < 0 || t >= N) continue;
        const int e = slot_of(t);
        if (e < 0) continue;
        const int dist = past - q;
        const unsigned f = D > 0 ? (unsigned)min(D, D + S - dist) : 1u;
        atomicMax(&fm[e], f);
        atomicAdd(&fs[e], f);
    }
}

// One candidate's 128 logits (xq_dh_rescore's step 3 ops: ascending-s partial sum -> f16 yraw ->
// had128 -> x svh -> f16), each penalized by xq_pen_tok where fm[c] != 0 (flag == 0: none), keyed
// (value desc, id asc). One warp; returns the warp-max key.
__device__ __forceinline__ unsigned long long dhp_combine(const float* __restrict__ part_c, int S, int bb,
                                                          const __half* __restrict__ svh,
                                                          const unsigned* fm, const unsigned* fs,
                                                          unsigned flag, float rep, float pres, float freq,
                                                          float scale, int lane) {
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float* pc = part_c + lane * 4 + q;
        float a = 0.f;
        for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
        v[q] = __half2float(__float2half_rn(a));      // yraw (exl3_gemm_body's f16 RN store)
    }
    exl3_had128(v, lane);                             // exl3_had_svh_dev's ops
    unsigned long long best = 0ull;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c = lane * 4 + q, col = bb * 128 + c;
        float y = __half2float(__float2half_rn(v[q] * __half2float(svh[col])));
        if (flag != 0u) {
            const unsigned m = fm[c];
            if (m != 0u) y = __half2float(__ushort_as_half(xq_pen_tok(y, m, fs, c, flag, rep, pres, freq, scale)));
        }
        const unsigned long long kk = xq_am_key(y, col);
        best = kk > best ? kk : best;
    }
    return xq_am_wmax(best);
}

// The pass's outputs from the final key (xq_dh_rescore's four writes, same formulas).
__device__ __forceinline__ void dhp_finish(unsigned* pws, int pass_i, int* out_id, float* dconf,
                                           int* d_dev, int* toks) {
    const unsigned long long k = atomicMax(reinterpret_cast<unsigned long long*>(pws + DHP_BEST), 0ull);
    const int id = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
    out_id[0] = id;                                   // sc.argmax
    dconf[pass_i] = dh_unord((unsigned)(k >> 32));    // == xq_rowmax_f16 of the penalized row's max
    d_dev[pass_i] = id;
    toks[0] = id;
}

// Pass A: grid (S, tmax), block 256, dynamic smem = max(R * 8 * 32 * hbits, 2 * tmax * 512) bytes.
// geomA = pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32 (16 b) | rlog2 << 48 (5 b) | cap << 53
// geomB = N | delta (f32 bits) << 32. pen = sc.pen (row 0), meta = sc.draft_meta, hist = sc.pen_hist.
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore_pa(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        const float* __restrict__ bsig, unsigned* __restrict__ ws, unsigned* __restrict__ pws,
        const unsigned* __restrict__ pen, const int* __restrict__ meta, const int* __restrict__ d_dev,
        const int* __restrict__ hist, long long geomA, long long geomB) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];            // [8 warps][R rows][2 * bits]
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ DhSel s_sel;
    __shared__ unsigned long long s_best[8];
    __shared__ unsigned char s_map[DH_KPT * 256];     // last CTA: block -> C1 rank (0xFF = not in C1)
    __shared__ int s_red[8];
    __shared__ int s_last;
    const int pass_i = (int)(geomA & 0xFF), bits = (int)((geomA >> 8) & 0xFF);
    const int S = (int)((geomA >> 16) & 0xFF), tmax = (int)((geomA >> 24) & 0xFF);
    const int K = (int)((geomA >> 32) & 0xFFFF), rlog2 = (int)((geomA >> 48) & 0x1F);
    const int cap = (int)((geomA >> 53) & 0x7FF);
    const int N = (int)(geomB & 0xFFFFFFFFll);
    const float delta = __uint_as_float((unsigned)((unsigned long long)geomB >> 32));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7, KW = K >> 2;
    const float* bmax = reinterpret_cast<const float*>(ws + dh_off_bmax(K));
    unsigned* cnt = ws + dh_off_cnt(K, nblk);
    int* dbg = reinterpret_cast<int*>(ws + dh_off_dbg(K, nblk));
    float* part = reinterpret_cast<float*>(ws + dh_off_part(K, nblk));
    const float sig = __uint_as_float(ws[KW + 2]);

    // ---- 1. C1 = DHEAD's candidate set (identical in every CTA) ----
    int nq = 0;
    const int ns = dh_select(bmax, bsig, nblk, sig, delta, tmax, s_sel, nq);
    if (blockIdx.x == 0 && blockIdx.y == 0) {         // XCHECK readback (DHEAD's dbg words)
        if (tid < ns) dbg[1 + tid] = s_sel.cand[tid];
        if (tid == 0) { dbg[0] = ns; dbg[DH_TCAP + 1] = nq; }
    }
    const int rc = blockIdx.y, sp = blockIdx.x;
    if (rc >= ns) return;                             // uniform per CTA

    // ---- 2. exact partial of candidate rc over K-split sp ----
    const int R = (K >> 4) / S, kb0 = sp * R;
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    float acc[2][4];
    dh_part(tr, s_sel.cand[rc], kb0, R, N >> 4, bits, s_ring4, s_x2, acc);
    dh_part_store(part + (size_t)(rc * S + sp) * 128, acc);
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = (atomicAdd(cnt, 1u) == (unsigned)(ns * S) - 1u) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) return;

    // ---- 3. last CTA: penalized C1 combine -> K1 (L), the bound, the pass-B list ----
    __threadfence();
    const unsigned flag = pen[0];
    const float rep = __uint_as_float(pen[1]), pres = __uint_as_float(pen[2]), freq = __uint_as_float(pen[3]);
    const int PD = (int)pen[5];
    const float scale = PD > 0 ? (float)PD : 1.0f;
    const int b0 = meta[0], p0 = meta[1], slot = meta[2];
    const int rmask = (1 << rlog2) - 1;
    unsigned* fm = reinterpret_cast<unsigned*>(s_ring4);  // [ns][128] window counts (ring is dead now)
    unsigned* fs = fm + ns * 128;
    for (int e = tid; e < ns * 128; e += 256) { fm[e] = 0u; fs[e] = 0u; }
    for (int bq = tid; bq < nblk; bq += 256) s_map[bq] = 0xFFu;
    if (tid < 8) pws[DHP_PEN + tid] = pen[tid];
    if (tid < 4) pws[DHP_META + tid] = tid < 3 ? (unsigned)meta[tid] : 0u;
    __syncthreads();
    if (tid < ns) s_map[s_sel.cand[tid]] = (unsigned char)tid;
    __syncthreads();
    if (flag != 0u) {
        const int* H = hist + (long long)slot * (rmask + 1);
        dhp_scan(pen, H, rmask, p0, b0, d_dev, p0 + 1 + pass_i, N,
                 [&](int t) { const int r = s_map[t >> 7]; return r == 0xFF ? -1 : ((r << 7) | (t & 127)); },
                 fm, fs);
        __syncthreads();
    }
    unsigned long long best = 0ull;
    for (int rr = warp; rr < ns; rr += 8) {
        const unsigned long long k = dhp_combine(part + (size_t)rr * S * 128, S, s_sel.cand[rr], svh,
                                                 fm + rr * 128, fs + rr * 128, flag, rep, pres, freq, scale, lane);
        best = k > best ? k : best;
    }
    if (lane == 0) s_best[warp] = best;
    __syncthreads();
    unsigned long long k1 = 0ull;
    #pragma unroll
    for (int w = 0; w < 8; ++w) k1 = s_best[w] > k1 ? s_best[w] : k1;
    const float L = dh_unord((unsigned)(k1 >> 32));   // the best penalized C1 logit: a lower bound
    const bool nonmono = ((flag & 1u) && !(rep >= 1.0f)) || ((flag & 2u) && !(pres >= 0.0f && freq >= 0.0f));
    const bool degen = !(L > -65504.0f);              // NaN, -inf, or the -65504 clamp floor
    unsigned mq = 0u, mall = 0u;
    int cq = 0;
    if (flag != 0u) {
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const int b = tid + 256 * i;
            if (b < nblk && s_map[b] == 0xFFu) {
                mall |= 1u << i;
                const float ub = bmax[b] + fmaxf(delta * sig * bsig[b], 0.f);
                if (!(ub < L)) { mq |= 1u << i; ++cq; }   // UB >= L (a NaN bound qualifies)
            }
        }
    }
    cq = __reduce_add_sync(0xffffffffu, cq);
    if (lane == 0) s_red[warp] = cq;
    __syncthreads();
    int nqual = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nqual += s_red[w];
    const bool over = nqual > cap;
    const bool fb = flag != 0u && (nonmono || degen || over);
    int n2 = 0;
    if (flag != 0u)                                   // uniform
        n2 = dh_rank1(fb ? mall : mq, s_sel.wh, s_sel.ph, &s_sel.tot[0], pws + DHP_HDR + dhp_nblk4(nblk));
    if (tid == 0) {
        unsigned fl = 0u;
        if (flag == 0u) fl = DHP_F_UNPEN;
        else fl = (nonmono ? DHP_F_NONMONO : 0u) | (degen ? DHP_F_DEGEN : 0u) | (over ? DHP_F_OVER : 0u)
                | (fb ? DHP_F_FB : 0u);
        *reinterpret_cast<unsigned long long*>(pws + DHP_BEST) = k1;
        pws[DHP_N2] = (unsigned)n2;
        unsigned* dg = pws + DHP_DBG;
        dg[0] = (unsigned)ns; dg[1] = (unsigned)n2; dg[2] = fl; dg[3] = __float_as_uint(L);
        dg[4] = (unsigned)nq; dg[5] = (unsigned)nqual; dg[6] = (unsigned)k1; dg[7] = (unsigned)(k1 >> 32);
        unsigned* st = pws + DHP_STAT;
        st[0] += 1u;
        if (flag == 0u) st[7] += 1u;
        else if (fb) {
            if (nonmono) st[4] += 1u; else if (degen) st[5] += 1u; else st[3] += 1u;
            st[6] += (unsigned)n2;
        } else if (n2 > 0) { st[1] += 1u; st[2] += (unsigned)n2; }
        cnt[0] = 0u;                                  // re-arm (shared with xq_dh_rescore: stream-ordered)
    }
}

// Pass B: grid (S, G), block 256, dynamic smem = R * 8 * 32 * hbits bytes; same geomA / geomB.
// CTA (sp, y) computes split sp of candidates y, y + G, ... < n2; the CTA completing a candidate's
// S splits combines + penalizes it and atomicMax-folds its key; the CTA completing the n2-th
// candidate writes the outputs. n2 == 0: CTA (0, 0) writes the outputs from pass A's key.
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore_pb(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        unsigned* __restrict__ pws, int* __restrict__ d_dev, const int* __restrict__ hist,
        int* __restrict__ out_id, float* __restrict__ dconf, int* __restrict__ toks,
        long long geomA, long long geomB) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ unsigned s_fm[128];
    __shared__ unsigned s_fs[128];
    __shared__ int s_last;
    const int pass_i = (int)(geomA & 0xFF), bits = (int)((geomA >> 8) & 0xFF);
    const int S = (int)((geomA >> 16) & 0xFF);
    const int K = (int)((geomA >> 32) & 0xFFFF), rlog2 = (int)((geomA >> 48) & 0x1F);
    const int N = (int)(geomB & 0xFFFFFFFFll);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7;
    const int n2 = (int)pws[DHP_N2];
    if (n2 == 0) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) dhp_finish(pws, pass_i, out_id, dconf, d_dev, toks);
        return;
    }
    if ((int)blockIdx.y >= n2) return;
    unsigned* cntc = pws + DHP_HDR;
    const unsigned* c2 = pws + DHP_HDR + dhp_nblk4(nblk);
    float* part = reinterpret_cast<float*>(pws + DHP_HDR + 2 * dhp_nblk4(nblk));
    const unsigned* P = pws + DHP_PEN;
    const unsigned flag = P[0];
    const float rep = __uint_as_float(P[1]), pres = __uint_as_float(P[2]), freq = __uint_as_float(P[3]);
    const int PD = (int)P[5];
    const float scale = PD > 0 ? (float)PD : 1.0f;
    const int b0 = (int)pws[DHP_META], p0 = (int)pws[DHP_META + 1], slot = (int)pws[DHP_META + 2];
    const int rmask = (1 << rlog2) - 1;
    const int* H = hist + (long long)slot * (rmask + 1);
    const int R = (K >> 4) / S, sp = blockIdx.x, kb0 = sp * R;
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    for (int rc = blockIdx.y; rc < n2; rc += gridDim.y) {
        const int cb = (int)c2[rc];
        __syncthreads();                              // the previous candidate's smem readers are done
        float acc[2][4];
        dh_part(tr, cb, kb0, R, N >> 4, bits, s_ring4, s_x2, acc);
        dh_part_store(part + (size_t)(rc * S + sp) * 128, acc);
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            s_last = (atomicAdd(cntc + rc, 1u) == (unsigned)S - 1u) ? 1 : 0;
        }
        __syncthreads();
        if (!s_last) continue;                        // uniform per CTA
        // this CTA completed candidate rc: its window counts, combine, fold
        __threadfence();
        if (tid < 128) { s_fm[tid] = 0u; s_fs[tid] = 0u; }
        __syncthreads();
        if (flag != 0u) {
            dhp_scan(P, H, rmask, p0, b0, d_dev, p0 + 1 + pass_i, N,
                     [&](int t) { return (t >> 7) == cb ? (t & 127) : -1; }, s_fm, s_fs);
            __syncthreads();
        }
        if (warp == 0) {
            const unsigned long long k = dhp_combine(part + (size_t)rc * S * 128, S, cb, svh, s_fm, s_fs,
                                                     flag, rep, pres, freq, scale, lane);
            if (lane == 0) {
                atomicMax(reinterpret_cast<unsigned long long*>(pws + DHP_BEST), k);
                cntc[rc] = 0u;                        // re-arm this candidate's split counter
                __threadfence();
                if (atomicAdd(pws + DHP_DONE, 1u) == (unsigned)n2 - 1u) {
                    __threadfence();
                    dhp_finish(pws, pass_i, out_id, dconf, d_dev, toks);
                    pws[DHP_DONE] = 0u;               // re-arm
                }
            }
        }
    }
}

// ---- embed gather + hc-stream expansion: resid[b][s*h+i] = f32(embed[tok[b]][i]).
extern "C" __global__ void xq_embed_resid(float* __restrict__ resid, const __half* __restrict__ emb,
                                          const int* __restrict__ toks, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (i >= rw * B) return;
    int b = (int)(i / rw);
    int r = (int)(i % rw);
    resid[i] = xq_h2f(emb[(long long)toks[b] * h + (r % h)]);
}

// ---- S-A3-f Item 1: RoPE table gather. cos/sin are precomputed ON HOST at load
// (bitwise-identical to the old per-step powf/cos/sin loop) into tab[max_pos][stride];
// each step copies the m active lanes' rows into the persistent per-step buffer.
// A row-copy of host-computed bits — numerics identical by construction.
extern "C" __global__ void xq_cos_gather(const float* __restrict__ tab, float* __restrict__ dst,
                                         const int* __restrict__ pos, int stride, int M) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * stride;
    if (i >= total) return;
    int b = (int)(i / stride);
    dst[i] = tab[(long long)pos[b] * stride + (int)(i - (long long)b * stride)];
}

// ---- VIS-2: RoPE row gather through a per-slot mrope map (every main-attention gather site).
// Row b is KV position p = pos[b] of slot s (slot_src ? slot_src[b * slot_stride] : slot_const).
// Inside the slot's mapped prompt (p < rplen[s]) the source row is rmap[s * max_pos + p]; past it,
// p + rdelta[s] (text after the last image runs rdelta <= 0 below its KV index, HF get_rope_index).
// Source rows >= max_pos live in the per-slot image-row table mtab (rows max_pos + s * R + j).
// A slot with rplen = rdelta = 0 — every text-only request — reads tab[p]: bit-identical to
// xq_cos_gather. An out-of-range slot (a stale id) also falls back to tab[p].
extern "C" __global__ void xq_cos_gather_v(const float* __restrict__ tab, const float* __restrict__ mtab,
                                           float* __restrict__ dst, const int* __restrict__ pos,
                                           const int* __restrict__ slot_src, int slot_stride, int slot_const,
                                           const int* __restrict__ rmap, const int* __restrict__ rplen,
                                           const int* __restrict__ rdelta, int nslots, int max_pos,
                                           int stride, int M) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * stride;
    if (i >= total) return;
    int b = (int)(i / stride);
    int p = pos[b];
    int s = slot_src ? slot_src[(long long)b * slot_stride] : slot_const;
    long long idx = p;
    if (s >= 0 && s < nslots) {
        idx = p < rplen[s] ? (long long)rmap[(long long)s * max_pos + p] : (long long)p + rdelta[s];
    }
    const float* src = idx < max_pos ? tab + idx * stride : mtab + (idx - max_pos) * stride;
    dst[i] = src[(int)(i - (long long)b * stride)];
}

// ---- VIS-2: image-embedding splice into the embedded residual of a prefill chunk.
// resid [c][hc][h] f32 (xq_embed_resid's layout); src_row[b] >= 0 = the row of img [n][h] for chunk
// row b (written into every hyper-connection stream: the reference splices BEFORE the hc repeat);
// src_row[b] < 0 leaves the token embedding.
extern "C" __global__ void xq_splice_rows(float* __restrict__ resid, const float* __restrict__ img,
                                          const int* __restrict__ src_row, int h, int hc, int B) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (i >= rw * B) return;
    int b = (int)(i / rw);
    int s = src_row[b];
    if (s < 0) return;
    resid[i] = img[(long long)s * h + (int)((i % rw) % h)];
}

// ---- per-expert input Hadamard: xh[e,m,:] = H(x[m,:] * suh_all[idxmap[e],:]).
// grid E*M*(K/128), block 32; suh PRE-butterfly (their convention, S-A3-c).
// exl3_had128 normalizes internally — no extra scale here.
extern "C" __global__ void xq_had_suh_multi(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                            const int* __restrict__ idxmap,
                                            __half* __restrict__ xh, int M, int K,
                                            const int* __restrict__ esel,
                                            int per_expert_in) {
    XQ_PDL_ENTRY();
    int nblk = K >> 7;
    int blk = blockIdx.x % nblk;
    long long em = blockIdx.x / nblk;
    int m = (int)(em % M);
    int e = (int)(em / M);
    if (e >= *esel) return;
    int lane = threadIdx.x & 31;
    long long base_in = (long long)(per_expert_in ? e : 0) * M * K + (long long)m * K
                      + (long long)blk * 128 + lane * 4;
    long long base_out = ((long long)e * M + m) * K + (long long)blk * 128 + lane * 4;
    const __half* sr = suh_all + (long long)idxmap[e] * K + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(x[base_in + r]) * xq_h2f(sr[r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = xq_f2h(v[r]);
}

// ---- S-A3-z lever 2: FUSED routing (decode / verify, M <= 8). Bit-identical to the path it
// replaces: xq_router_ks (ks = 64 K-slices) + xq_router_combine, then xq_router_topk + xq_moe_route.
// (a) xq_router_fused: block = 4 experts x 64 slices; thread (e, s) reads its 40-wide weight slice
//     ONCE and accumulates every row with xq_router_ks's exact per-(row, expert, slice) order
//     (float4 groups of 8, ascending); the 64 partials of (row, expert) are then summed ascending
//     s = 0..63 by one thread — xq_router_combine's ((p0 + p1) + p2) + ... order. Old kernel re-read
//     each weight slice once per row from uncoalesced n*K strides.
#define XQ_RF_KS 64
#define XQ_RF_EPB 4          // experts per block (256 / XQ_RF_KS)
#define XQ_RF_MAXM 8
extern "C" __global__ void __launch_bounds__(256)
xq_router_fused(float* __restrict__ out, const __half* __restrict__ w, const __half* __restrict__ x,
                int M, int N, int K) {
    XQ_PDL_ENTRY();
    __shared__ float part[XQ_RF_MAXM][XQ_RF_EPB][XQ_RF_KS];
    const int tid = threadIdx.x;
    const int el = tid / XQ_RF_KS, s = tid % XQ_RF_KS;
    const int n = blockIdx.x * XQ_RF_EPB + el;
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    float acc[XQ_RF_MAXM];
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) acc[r] = 0.0f;
    if (n < N && slice == 40) {
        // K = 2560 (this model): 5 float4 groups, fully unrolled — all weight loads issue up front.
        float4 wv5[5];
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (n < N) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int k = (s8 << 3); k < slice; ++k) {
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[k]) * __half2float(x[(size_t)r * K + k0 + k]);
        }
    }
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) if (r < M) part[r][el][s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_RF_EPB) {
        const int r = tid / XQ_RF_EPB, e = tid % XQ_RF_EPB;
        const int nn = blockIdx.x * XQ_RF_EPB + e;
        if (nn < N) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r][e][q];
            out[(size_t)r * N + nn] = a;
        }
    }
}

// (b) xq_router_topk_route: grid (B), block 256 (== xq_router_topk's launch). Softmax is
//     xq_router_topk's code verbatim (same reductions => same prob bits). Selection: exact top-k
//     by (prob desc, id asc) — per-warp top-k by shuffle argmax over the warp's strided entries,
//     then warp 0 merges the 8 x k candidates by the same order. The top-k SET and ORDER of a
//     total order are unique, so ids/wts equal xq_router_topk's; renorm = its ascending-j sum.
//     The LAST block to finish (counter, reset by that block) runs xq_moe_route's scan verbatim.
#define XQ_RTR_MAXK 16
extern "C" __global__ void __launch_bounds__(256)
xq_router_topk_route(int* __restrict__ ids, float* __restrict__ wts, const float* __restrict__ logits,
                     int ne, int k, int B, int* __restrict__ done_cnt,
                     int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                     unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                     unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    extern __shared__ float sm[];
    const int b = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, nw = bs >> 5;
    float* prob = sm + bs;
    float* cv = sm + bs + ne;                         // [nw * XQ_RTR_MAXK] candidate values
    int* ci = (int*)(cv + nw * XQ_RTR_MAXK);          // [nw * XQ_RTR_MAXK] candidate ids
    __shared__ int last;
    if (b < B) {
        const float* lg = logits + (long long)b * ne;
        // ---- softmax: xq_router_topk verbatim ----
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
        for (int i = tid; i < ne; i += bs) prob[i] = __expf(lg[i] - gmax) / denom;
        __syncthreads();
        // ---- per-warp top-k over this warp's entries {i : (i / 32) % nw == warp} ----
        for (int j = 0; j < k; j++) {
            float bv = -2.0f; int bi = ne;
            for (int i = warp * 32 + lane; i < ne; i += bs)
                if (prob[i] > bv || (prob[i] == bv && i < bi)) { bv = prob[i]; bi = i; }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                cv[warp * XQ_RTR_MAXK + j] = bv; ci[warp * XQ_RTR_MAXK + j] = bi;
                if (bi < ne) prob[bi] = -3.0f;        // below the -2 sentinel: never re-picked
            }
            __syncwarp();
        }
        __syncthreads();
        // ---- merge (warp 0): k picks over nw*k candidates, same total order ----
        if (warp == 0) {
            const int nc = nw * k;
            for (int j = 0; j < k; j++) {
                float bv = -2.0f; int bi = ne, bslot = -1;
                for (int c = lane; c < nc; c += 32) {
                    const int wq = c / k, jj = c % k;
                    const float v = cv[wq * XQ_RTR_MAXK + jj]; const int id = ci[wq * XQ_RTR_MAXK + jj];
                    if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; bslot = wq * XQ_RTR_MAXK + jj; }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                    const int os = __shfl_xor_sync(0xffffffffu, bslot, o);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bslot = os; }
                }
                if (lane == 0) {
                    int id = bi;
                    if (id < 0 || id >= ne) id = 0;   // xq_router_topk's NaN/Inf clamp
                    ids[(long long)b * k + j] = id;
                    wts[(long long)b * k + j] = bv;
                    if (bslot >= 0) cv[bslot] = -4.0f;
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
    // ---- last block: xq_moe_route verbatim ----
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the scan runs in SHARED memory: a serial thread-0 walk over global (volatile) ids + a global
    // slotmap cost ~0.4 us of L2 latency per step (24 us/launch measured) — same first-seen order.
    __shared__ int cnt;
    __shared__ int sids[XQ_RTR_MAXK * XQ_RF_MAXM];
    __shared__ int sidx[XQ_RTR_MAXK * XQ_RF_MAXM];
    int* sslot = (int*)sm;                           // ne ints (the softmax scratch is dead here)
    for (int t = tid; t < B * k; t += bs) sids[t] = __ldcg(ids + t);
    for (int e = tid; e < ne; e += bs) sslot[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < B * k; t++) {
            const int e = sids[t];
            if (e >= 0 && e < ne && sslot[e] == -1) { sslot[e] = c; sidx[c] = e; c++; }
        }
        cnt = c; esel_dev[0] = c;
        *done_cnt = 0;                               // re-arm for the next launch / graph replay
    }
    __syncthreads();
    for (int e = tid; e < ne; e += bs) slotmap[e] = sslot[e];
    for (int s = tid; s < cnt; s += bs) {
        const int e = sidx[s];
        idxmap[s] = e;
        offs_gu[s] = (unsigned long long)e * gu_words;
        offs_d[s] = (unsigned long long)e * d_words;
    }
}

// ---- WP10 (PLAN/SURPASS_PLAN_2026-09-26): ROUTING FOLD. xq_router_fused + xq_router_topk_route in
// ONE launch: grid (ne/4), block 256 — the GEMV grid of xq_router_fused; the LAST block to finish
// (self re-arming counter; publication = cooperative-groups grid-sync pattern: bar.sync -> tid0
// fence + atomic; winner: fence -> bar.sync -> __ldcg) runs softmax / top-k / renorm / first-seen
// route for all M <= 8 rows, warp r <-> row r. Bit-identity contract vs the two-kernel path (itself
// == xq_router_ks + combine + xq_router_topk + xq_moe_route):
//  logits  — every (row, expert) value is the SAME thread-local chain (40-wide slice, float4 groups
//            of 8, ascending k, acc += w*x) + the SAME one-thread ascending s = 0..63 partial sum.
//            REMAP only changes WHICH thread owns (expert, slice) and pads the smem partial layout.
//  softmax — lane l holds the virtual xq_router_topk threads tid = l + 32q (q < 8), elements
//            i = tid + 256p (ascending p, the same per-thread folds). The 256-thread halving tree
//            is replayed exactly: q += q+4 / q+2 / q+1 in registers (s2 = 128, 64, 32), then
//            shfl_down 16..1 (s2 = 16..1): same ops, same order, same (low, high) operand order =>
//            same gmax / denom bits; prob = the same __expf(lg - gmax) / denom.
//  top-k   — (prob desc, id asc) is a total order on the non-NaN probs, so the top-k sequence is
//            unique: k warp-wide shuffle argmaxes == topk_route's per-warp + merge. An all-NaN row
//            picks (-2, ne) -> id 0, w -2 at every j, exactly as topk_route.
//  renorm  — sum = ((0 + w0) + w1) + ... ascending j (topk_route's lane-0 loop), w_j / sum.
//  route   — first-seen over ids row-major: first(e) = min position (smem atomicMin: a min is
//            order-free), slot(e) = #{first positions < first(e)} (ballot / popc prefix) =>
//            identical slotmap / idxmap / esel / offs_gu / offs_d (written for s < esel only).
// REMAP (xq_router_fold): warp w owns slices 8w..8w+7 of all 4 experts (lane = el*8 + s%8), so a
// warp's x float4 loads touch 8 distinct 16-B chunks in 5 lines (was 32 in 20: the m-scaling
// L1 wavefront term), weights unchanged (4 x 640 B = 20 lines); partials padded (el stride 72,
// row stride 289) so both the partial stores and the 64-deep ascending sums are bank-conflict-free
// (the legacy [8][4][64] layout put all M*4 summing lanes on one bank per step).
// xq_router_fold_m0 = xq_router_fused's exact lane map + layout (isolates the fold; GB10_WP10_GEMV=0).
#define XQ_RFD_NEMAX 512                               // the tail holds ne/32 <= 16 probs per lane
#define XQ_RFD_PS (XQ_RF_KS + 8)                       // REMAP partial el stride (72 == 8 mod 32)
#define XQ_RFD_RS (XQ_RF_EPB * XQ_RFD_PS + 1)          // REMAP partial row stride (289 == 1 mod 32)
// A5-K2 COAL (xq_router_fold_c; ledger #4): the 80-B-per-thread weight read (5 LDG.128 at an 80-B
// lane stride: 32 sector requests per warp instruction for 16 sectors of data) streams the 2.62 MB
// router at ~146 GB/s (17.9 us standalone) where a coalesced read of the same bytes takes 11.9 us
// (221 GB/s) whatever the grid (48..640 CTAs: the CTA count is NOT the limit). COAL loads each warp's
// 4 x 640-B expert chunks with coalesced cp.async.cg (lane-contiguous 16-B pieces) into a
// warp-private smem slab (dynamic, 8 x 2560 B), __syncwarp, then every thread reads ITS five
// float4 back (LDS.128, 80-B lane stride = conflict-free per quarter-warp) and runs the unchanged
// chain; per q it issues the M x loads before the FMAs (register-only reordering of loads). Bits:
// the same (w, x) values reach the same FFMA sequence per (row, expert, slice) — identical.
// TP-G (T2 lever #5) EP: the route tail runs xq_moe_route_ep's semantics instead — first-seen over
// ids row-major restricted to the experts THIS rank holds (ep_lut[e] >= 0), idxmap / offs_* in LOCAL
// stacked indices (ep_lut[e]), slotmap[e] = -1 for a remote expert. The slot of a local first position
// = #local first positions before it (ballot/popc prefix over the local-first flags) == route_ep's
// serial counter: identical tables, one launch fewer per MoE layer. EP = false compiles it out.
template <bool REMAP, bool COAL = false, bool EP = false>
__device__ __forceinline__ void xq_router_fold_body(
        float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
        int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
        int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
        unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
        unsigned long long gu_words, unsigned long long d_words, const int* __restrict__ ep_lut = nullptr) {
    constexpr int PS = REMAP ? XQ_RFD_PS : XQ_RF_KS;
    constexpr int RS = REMAP ? XQ_RFD_RS : XQ_RF_EPB * XQ_RF_KS;
    static_assert(XQ_RF_MAXM * RS >= XQ_RFD_NEMAX + 2 * XQ_RF_MAXM * XQ_RTR_MAXK + 8, "tail smem alias");
    __shared__ float part[XQ_RF_MAXM * RS];
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int el = REMAP ? (lane >> 3) : (tid / XQ_RF_KS);
    const int s = REMAP ? ((warp << 3) | (lane & 7)) : (tid % XQ_RF_KS);
    const int n = blockIdx.x * XQ_RF_EPB + el;
    // ---- GEMV: xq_router_fused's per-thread body, verbatim ----
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    float acc[XQ_RF_MAXM];
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) acc[r] = 0.0f;
    if constexpr (COAL) {
        static_assert(REMAP, "COAL uses the REMAP lane map (warp = 4 experts x 8 slices)");
        extern __shared__ __align__(16) float4 xq_rfc_ws[];          // [8 warps][4 experts][40]
        if (slice == 40) {                                           // host: always (K == 2560)
            float4* ws = xq_rfc_ws + warp * (XQ_RF_EPB * 40);
            #pragma unroll
            for (int q = 0; q < 5; ++q) {
                const int f = (q << 5) | lane, ef = f / 40, c = f - ef * 40;
                const int nf = blockIdx.x * XQ_RF_EPB + ef;
                if (nf < N)
                    lmh_cp16((unsigned)__cvta_generic_to_shared(ws + f),
                             w + (size_t)nf * K + warp * (8 * 40) + (c << 3));
            }
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;\n" ::: "memory");
            __syncwarp();
            if (n < N) {
                float4 wv5[5];
                #pragma unroll
                for (int q = 0; q < 5; ++q) wv5[q] = ws[el * 40 + (lane & 7) * 5 + q];
                #pragma unroll
                for (int q = 0; q < 5; ++q) {
                    const __half* wh = (const __half*)&wv5[q];
                    float4 xv[XQ_RF_MAXM];
                    #pragma unroll
                    for (int r = 0; r < XQ_RF_MAXM; r++)
                        if (r < M) xv[r] = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    #pragma unroll
                    for (int r = 0; r < XQ_RF_MAXM; r++) {
                        if (r < M) {
                            const __half* xh_ = (const __half*)&xv[r];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                        }
                    }
                }
            }
        }
    } else {
    if (n < N && slice == 40) {
        float4 wv5[5];
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (n < N) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int kk = (s8 << 3); kk < slice; ++kk) {
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[kk]) * __half2float(x[(size_t)r * K + k0 + kk]);
        }
    }
    }   // !COAL
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) if (r < M) part[r * RS + el * PS + s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_RF_EPB) {
        const int r = tid / XQ_RF_EPB, e = tid % XQ_RF_EPB;
        const int nn = blockIdx.x * XQ_RF_EPB + e;
        if (nn < N) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r * RS + e * PS + q];
            out[(size_t)r * N + nn] = a;
        }
    }
    // ---- publish + elect the last block (every block's logits visible before its ticket) ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
        if (last) __threadfence();
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    // ---- tail (last block only). `part` is dead: reuse it for the route scratch ----
    int* s_first = reinterpret_cast<int*>(part);                 // [ne]  first position of expert e
    int* s_ids = s_first + XQ_RFD_NEMAX;                         // [M*k] picked ids, row-major
    int* s_slot = s_ids + XQ_RF_MAXM * XQ_RTR_MAXK;              // [M*k] slot of a first position
    int* s_wc = s_slot + XQ_RF_MAXM * XQ_RTR_MAXK;               // [8]   per-warp first counts
    for (int e = tid; e < N; e += blockDim.x) s_first[e] = 0x7fffffff;
    if (warp < M) {
        const int r = warp;
        const float* lg = out + (size_t)r * N;
        float v[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) { const int i = lane + 32 * t; v[t] = (i < N) ? __ldcg(lg + i) : 0.0f; }
        // max: virtual thread q folds i = l+32q, then l+32q+256 (ascending), then the 256-tree.
        float mq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            mq[q] = -INFINITY;
            if (lane + 32 * q < N) mq[q] = fmaxf(mq[q], v[q]);
            if (lane + 32 * q + 256 < N) mq[q] = fmaxf(mq[q], v[q + 8]);
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) mq[q] = fmaxf(mq[q], mq[q + 4]);           // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) mq[q] = fmaxf(mq[q], mq[q + 2]);           // s2 = 64
        float mx = fmaxf(mq[0], mq[1]);                                          // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
        const float gmax = __shfl_sync(0xffffffffu, mx, 0);
        // denominator: the same per-virtual-thread partials and the same halving tree.
        float ev[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) ev[t] = __expf(v[t] - gmax);
        float pq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            pq[q] = 0.0f;
            if (lane + 32 * q < N) pq[q] += ev[q];
            if (lane + 32 * q + 256 < N) pq[q] += ev[q + 8];
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) pq[q] += pq[q + 4];                          // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) pq[q] += pq[q + 2];                          // s2 = 64
        float sm_ = pq[0] + pq[1];                                               // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sm_ += __shfl_down_sync(0xffffffffu, sm_, o);
        const float denom = __shfl_sync(0xffffffffu, sm_, 0);
        float p[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) p[t] = (lane + 32 * t < N) ? ev[t] / denom : -4.0f;  // -4: never picked
        // top-k under (prob desc, id asc); lane j keeps pick j; every lane folds the same sum.
        float myw = 0.0f, sum = 0.0f;
        int myid = 0;
        for (int j = 0; j < k; ++j) {
            float bv = -2.0f; int bi = N;
            if constexpr (COAL) {
                // A5-K2: the same pick, shorter chains. (prob desc, id asc) is a total order on the
                // non-NaN probs and every candidate here is a prob in [0, 1] (ev <= 1 after the max
                // shift; NaN / picked -3 / padding -4 are never picked: they never beat the (-2, N)
                // seed). So: map non-candidates (!(p >= 0)) to -5, take the lane's best by a 4-level
                // tree (ids ascend with t, so a pair keeps its lower id unless the higher one is
                // strictly greater), then the warp's best as (max key, min id among max-key lanes)
                // with redux.sync, key = bits(p) + 1 (order-preserving for p >= 0; 0 = no
                // candidate). bv carries the exact prob bits back. == the linear scan + butterfly.
                float tv[16]; int ti[16];
                #pragma unroll
                for (int t = 0; t < 16; ++t) { tv[t] = (p[t] >= 0.0f) ? p[t] : -5.0f; ti[t] = lane + 32 * t; }
                #pragma unroll
                for (int st = 1; st < 16; st <<= 1) {
                    #pragma unroll
                    for (int t = 0; t < 16; t += 2 * st)
                        if (tv[t + st] > tv[t]) { tv[t] = tv[t + st]; ti[t] = ti[t + st]; }
                }
                const unsigned key = (tv[0] >= 0.0f) ? __float_as_uint(tv[0]) + 1u : 0u;
                const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
                const unsigned imin = __reduce_min_sync(0xffffffffu, (key == kmax) ? (unsigned)ti[0] : 0xffffffffu);
                if (kmax != 0u) { bv = __uint_as_float(kmax - 1u); bi = (int)imin; }
            } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int i = lane + 32 * t;
                if (p[t] > bv || (p[t] == bv && i < bi)) { bv = p[t]; bi = i; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            }   // !COAL
            #pragma unroll
            for (int t = 0; t < 16; ++t) if (lane + 32 * t == bi) p[t] = -3.0f;
            const int id = (bi < 0 || bi >= N) ? 0 : bi;   // topk_route's NaN/Inf clamp
            if (lane == j) { myw = bv; myid = id; }
            sum += bv;
        }
        if (lane < k) {
            ids[(size_t)r * k + lane] = myid;
            wts[(size_t)r * k + lane] = myw / sum;
            s_ids[r * k + lane] = myid;
        }
    }
    __syncthreads();
    // ---- first-seen route (xq_moe_route semantics) ----
    const int BK = M * k;                                        // host: <= 128 <= blockDim
    if (tid < BK) { const int e = s_ids[tid]; if (e >= 0 && e < N) atomicMin(&s_first[e], tid); }
    __syncthreads();
    int e_t = -1; bool first = false;
    if (tid < BK) { e_t = s_ids[tid]; first = (e_t >= 0 && e_t < N) && s_first[e_t] == tid; }
    int le_t = e_t;                                              // the stacked index idxmap/offs carry
    if constexpr (EP) {
        if (first) { le_t = __ldg(ep_lut + e_t); first = le_t >= 0; }   // a remote expert takes no slot
    }
    const unsigned bal = __ballot_sync(0xffffffffu, first);
    if (lane == 0) s_wc[warp] = __popc(bal);
    __syncthreads();
    int base = 0, total = 0;
    #pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) { const int c = s_wc[w8]; total += c; if (w8 < warp) base += c; }
    if (first) {
        const int slot = base + __popc(bal & ((1u << lane) - 1u));
        s_slot[tid] = slot;
        idxmap[slot] = le_t;
        offs_gu[slot] = (unsigned long long)le_t * gu_words;
        offs_d[slot] = (unsigned long long)le_t * d_words;
    }
    if (tid == 0) { esel_dev[0] = total; *done_cnt = 0; }   // re-arm for the next launch / graph replay
    __syncthreads();
    for (int e = tid; e < N; e += blockDim.x) {
        const int f = s_first[e];
        bool live = f != 0x7fffffff;
        if constexpr (EP) live = live && __ldg(ep_lut + e) >= 0;       // s_slot[f] is unwritten for a remote e
        slotmap[e] = live ? s_slot[f] : -1;
    }
}
extern "C" __global__ void __launch_bounds__(256, 3)   // 3 blocks/SM: all ne/4 = 128 blocks in ONE wave
xq_router_fold(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
               int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
               int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
               unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
               unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                              offs_gu, offs_d, gu_words, d_words);
}
// A5-K2: the coalesced-weight twin (dynamic smem 8 x 2560 B = XQ_RFC_SMEM; host: K == 2560).
#define XQ_RFC_SMEM (8 * XQ_RF_EPB * 40 * 16)
extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_c(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                 int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                 int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                 unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                 unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true, true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                                    offs_gu, offs_d, gu_words, d_words);
}
// TP-G (T2 lever #5): xq_router_fold_c with xq_moe_route_ep folded into its route tail (EP ranks).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_c_ep(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                    int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                    int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                    unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                    unsigned long long gu_words, unsigned long long d_words, const int* __restrict__ ep_lut) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true, true, true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                                          offs_gu, offs_d, gu_words, d_words, ep_lut);
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_m0(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                  int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                  int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                  unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                  unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<false>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                               offs_gu, offs_d, gu_words, d_words);
}

// ---- S-A3-f Item 2: device-side MoE routing. Replaces the per-layer host round-trip
// (sync + dtoh ids + host first-seen dedup + 4 uploads) with ONE kernel launch.
// Semantics replicate the old host scan EXACTLY: slots are assigned first-seen over
// ids[0..m*topk] row-major; offs_* are the compacted trellis word offsets; esel_dev[0]
// is the live-expert count the padded consumer grids early-exit on. Grid (1,1,1).
extern "C" __global__ void xq_moe_route(const int* __restrict__ ids, int m, int topk, int ne,
                                        int* __restrict__ slotmap, int* __restrict__ idxmap,
                                        int* __restrict__ esel_dev,
                                        unsigned long long* __restrict__ offs_gu,
                                        unsigned long long* __restrict__ offs_d,
                                        unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    __shared__ int cnt;
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < m * topk; t++) {
            int e = ids[t];
            if (e >= 0 && e < ne && slotmap[e] == -1) {
                slotmap[e] = c;
                idxmap[c] = e;
                c++;
            }
        }
        cnt = c;
        esel_dev[0] = c;
    }
    __syncthreads();
    for (int s = tid; s < cnt; s += blockDim.x) {
        int e = idxmap[s];
        offs_gu[s] = (unsigned long long)e * gu_words;
        offs_d[s] = (unsigned long long)e * d_words;
    }
}

// S-A3-s diagnostic: scatter per-row QSA selection lists into a position-indexed buffer
// (dst_sel[pos][sel_max], dst_psel[pos]) — grid rows, block 256; rows with pos >= cap skipped.
extern "C" __global__ void xq_dbg_scatter_sel(int* __restrict__ dst_sel, int* __restrict__ dst_psel,
                                              const int* __restrict__ src_sel, const int* __restrict__ src_psel,
                                              const int* __restrict__ pos, int sel_max, int cap) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int p = pos[r];
    if (p < 0 || p >= cap) return;
    for (int i = threadIdx.x; i < sel_max; i += blockDim.x)
        dst_sel[(long long)p * sel_max + i] = src_sel[(long long)r * sel_max + i];
    if (threadIdx.x == 0) dst_psel[p] = src_psel[r];
}

// S-A3-o diagnostic: hist[m*w + min(esel, w-1)] += 1 (GB10_EXL3_ESEL_HIST only).
extern "C" __global__ void xq_esel_hist(const int* __restrict__ esel, int* __restrict__ hist, int m, int w) {
    XQ_PDL_ENTRY();
    int e = *esel;
    atomicAdd(&hist[m * w + (e < w - 1 ? e : w - 1)], 1);
}

// TP-I #6 diagnostic (GB10_TP_EP_HIST): per-(layer, expert) routing histogram of every decode /
// verify MoE call (m <= 16). hist[e] += 1 once per call when expert e is live in the call (the
// weight-streaming unit of the expert GEMMs), hist[ne + e] += its row picks. Diagnostic only.
extern "C" __global__ void xq_expert_hist(const int* __restrict__ ids, int m, int topk, int ne, int* __restrict__ hist) {
    XQ_PDL_ENTRY();
    __shared__ unsigned char seen[1024];
    const int tid = threadIdx.x;
    for (int e = tid; e < ne && e < 1024; e += blockDim.x) seen[e] = 0;
    __syncthreads();
    for (int t = tid; t < m * topk; t += blockDim.x) {
        int e = ids[t];
        if (e >= 0 && e < ne && e < 1024) { seen[e] = 1; atomicAdd(&hist[ne + e], 1); }
    }
    __syncthreads();
    for (int e = tid; e < ne && e < 1024; e += blockDim.x) if (seen[e]) atomicAdd(&hist[e], 1);
}

// ---- per-expert output Hadamard: y[e,m,:] = H(yraw[e,m,:]) * svh_all[idxmap[e],:]).
// S-A3-f Item 2: grids are PADDED to the topk cap; blocks of absent experts exit on
// the device-routed count before touching memory.
extern "C" __global__ void xq_had_svh_multi(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                            const int* __restrict__ idxmap,
                                            __half* __restrict__ y, int M, int N,
                                            const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    int nblk = N >> 7;
    int blk = blockIdx.x % nblk;
    long long em = blockIdx.x / nblk;
    int m = (int)(em % M);
    int e = (int)(em / M);
    if (e >= *esel) return;
    int lane = threadIdx.x & 31;
    long long base = ((long long)e * M + m) * N + (long long)blk * 128 + lane * 4;
    const __half* sr = svh_all + (long long)idxmap[e] * N + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(yraw[base + r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) y[base + r] = xq_f2h(v[r] * xq_h2f(sr[r]));
}

// ---- grouped trellis GEMM with PER-EXPERT xh ([E][M][K]). Same tile contract as
// exl3_hmma_gemm_grouped; only the A-operand indexing changes.
// S-A3-f Item 2: grid padded to the topk cap; absent experts exit on the device count.
extern "C" __global__ void xq_gemm_grouped_xh(const uint16_t* __restrict__ base,
                                              const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh,
                                              __half* __restrict__ y,
                                              int M, int K, int N, int bits,
                                              const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    __half* ye = y + (long long)expert * M * N;
    // S-A3-n F3: A staged once when it fits (the launcher passes M*(K+8)*2 dyn smem).
    if (exl3_a1_fits(M, K)) {
        switch (bits) {
            case 3: exl3_gemm_body_a1<3>(tr, xe, ye, M, K, N, tile); break;
            case 4: exl3_gemm_body_a1<4>(tr, xe, ye, M, K, N, tile); break;
            case 5: exl3_gemm_body_a1<5>(tr, xe, ye, M, K, N, tile); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 3: exl3_gemm_body<3>(tr, xe, ye, M, K, N, tile); break;
        case 4: exl3_gemm_body<4>(tr, xe, ye, M, K, N, tile); break;
        case 5: exl3_gemm_body<5>(tr, xe, ye, M, K, N, tile); break;
        default: break;
    }
}

template <int BITS>
__device__ __forceinline__ void exl3_gemm_body_a1_fh(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ x_in,
                                                  const __half* __restrict__ suh,
                                                  const __half* __restrict__ svh,
                                                  __half* __restrict__ y,
                                                  int M, int K, int N, int cta_tile) {
    extern __shared__ __half a1_sa[];        // [M][K+8]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int SK = K + 8;
    {
        // fused xq_had_suh_multi: A = f16(H128(x * suh)) per (row, 128-block), the SAME loads,
        // product, exl3_had128 butterfly and fp16 rounding => the bytes xq_had_suh_multi wrote.
        const int nblk = K >> 7;
        for (int bi = warp; bi < M * nblk; bi += (blockDim.x >> 5)) {
            const int r = bi / nblk, blk = bi - r * nblk;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                v[j] = xq_h2f(x_in[(size_t)r * K + blk * 128 + lane * 4 + j]) * xq_h2f(suh[blk * 128 + lane * 4 + j]);
            exl3_had128(v, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j) a1_sa[(size_t)r * SK + blk * 128 + lane * 4 + j] = xq_f2h(v[j]);
        }
    }
    __syncthreads();

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;

    // S-A3-n F4: keep DRAM busy without registers — lanes 0..2 prefetch the
    // 16x16 block window (8*BITS words <= 160 B) PF_D steps ahead into L2
    // (one predicated instruction/step; no value change). The in-round expert
    // stream is DRAM-latency-bound at 24 warps/SM with one 96-B window in flight.
    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < BITS * 32;          // 3/4/5 lanes x 32 B
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;              // bytes per k16 step
    // S-A3-o O1: prologue — the first PF_D windows were never prefetched (every CTA
    // started with PF_D serial DRAM misses, all CTAs of a wave in lock-step). Issue
    // them up front; value-transparent (L2 hint only).
    if (pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    for (int kb = 0; kb < KB; ++kb) {
        if (pf_lane && kb + PF_D < KB)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
        pf_ptr += pf_step;
        uint32_t bfrag[2][2];
#if A1_SHFL
        // S-A3-n F7: ONE coalesced load per lane per step (window words
        // lane, lane+32 < 8*BITS), decode words delivered by __shfl_sync —
        // replaces 16 scalar L1 loads per lane per step. Same words => same bits.
        constexpr int WW = 8 * BITS;
        const unsigned w0 = (lane < WW) ? __ldg(ring + lane) : 0u;
        const unsigned w1 = (WW > 32 && lane + 32 < WW) ? __ldg(ring + lane + 32) : 0u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint16_t rv[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = h * 4 + q;
                unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                if (WW > 32) {
                    const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                    const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                    lo = s_i1m[j] >= 32 ? lo2 : lo;
                    hi = s_i0m[j] >= 32 ? hi2 : hi;
                }
                const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                            __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
            }
            bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
            bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
        }
#else
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
#endif
        ring += ring_kstride;
        const int col = kb * 16;
        const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }

    // fused xq_had_svh_multi: the fp16 GEMM tile (the exact yraw values) staged in smem, then
    // y = f16(H128(yraw) * svh) per row with the SAME butterfly and rounding.
    __syncthreads();                          // every warp is done reading A from a1_sa
    __half* yt = a1_sa;                       // M x 128 halves (<= M*(K+8))
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lc = 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) *(__half2*)&yt[row0 * 128 + lc] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (row1 < M) *(__half2*)&yt[row1 * 128 + lc] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
    __syncthreads();
    for (int r = warp; r < M; r += (blockDim.x >> 5)) {
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = xq_h2f(yt[r * 128 + lane * 4 + j]);
        exl3_had128(v, lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            y[(size_t)r * N + cta_n + lane * 4 + j] = xq_f2h(v[j] * xq_h2f(svh[cta_n + lane * 4 + j]));
    }
}

// 2026-09-26: the served expert GEMM with BOTH per-expert Hadamards fused (A5 D3, experts):
// xq_had_suh_multi -> xq_gemm_grouped_a1b3 -> xq_had_svh_multi become ONE launch per GEMM, bit
// for bit the same bytes (same load/product/butterfly/rounding; same K loop and mma sequence).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3_fh(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                        const __half* __restrict__ x_in, int per_expert_in,
                        const __half* __restrict__ suh_all, const __half* __restrict__ svh_all,
                        const int* __restrict__ idxmap, __half* __restrict__ y,
                        int M, int K, int N, const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ix = idxmap[expert];
    exl3_gemm_body_a1_fh<3>(base + (size_t)offs[expert],
                            x_in + (per_expert_in ? (long long)expert * M * K : 0LL),
                            suh_all + (long long)ix * K, svh_all + (long long)ix * N,
                            y + (long long)expert * M * N, M, K, N, tile);
}

// S-A3-o O3: the expert stream's served body (A-once, 3-bit) as its OWN entry, so
// its register budget is not the max over the barrier/b5 bodies sharing
// xq_gemm_grouped_xh (96 regs -> 2 CTAs/SM). Same body, same instruction sequence
// per output => bitwise identical to xq_gemm_grouped_xh (probe-exl3-binv).
// Preconditions (launcher-checked): exl3_a1_fits(M, K), bits == 3.
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<3>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}

// ===========================================================================
// A5 WP20 (PLAN/SURPASS_PLAN_2026-09-26.md): expert WORD DIET + MoE glue as
// LAST-ARRIVAL EPILOGUES (the reference implementation's exl3_moe_coop structure on our a1b3 body).
// Target: plain sm_121 — nothing here needs an f/a feature (fp16 mma.m16n8k16, dp4a,
// 64-bit funnel shifts, device-scope atomics/fences are baseline CC 12.1).
//
// Rung 1 (V1) — the decode's word supply. A lane's 8 B-fragment slots j = 4h+q are
// 8 CONSECUTIVE trellis indices t0+j (tmap(r,c) with r = 2(l%4)+(q&1)+8(q>>1),
// c = 8h+l/4 reduces to t = t0(l) + 4h + q). Their 16-bit windows [b1-16, b1),
// b1 = (t+257)*BITS, span 16 + 7*BITS bits starting at bit 24g+755 (b3; g = t0/8),
// i.e. start%32 in {3,11,19,27} -> ALWAYS inside the two ring words A = start>>5 and
// A+1 (mod W = 8*BITS). So each lane loads those 2 words ONCE per k16 step and
// slices every window from the 64-bit pair Q = (w[A] << 32) | w[A+1]:
//   idx_j = (Q >> (32(A+2) - b1_j)) & 0xFFFF
// which is exactly the old funnelshift_r(w[i1], w[i0], 32(i1+1)-b1) & 0xFFFF for
// every (i0, i1) in {(A,A), (A,A+1), (A+1,A+1)} — the SAME stream bits, so the same
// idx, the same dp4a/hfma, the same fragments, the same mma chain: BITWISE. It
// replaces the 16 SHFL per lane per step of the A1_SHFL block (1 LDG + 16 SHFL ->
// 2 LDG, all MIO). (Index math checked for every lane/slot at b3 and b4 against the
// old decode on random rings; the binv probe checks the entry against the barrier body.)
//
// Rung 2 (V9) — the MoE glue moves into the two expert GEMMs' epilogues:
//  gate/up (xq_moe_gu_epi): every CTA applies ITS tile's svh (== xq_had_svh_multi),
//    publishes the f16 tile to ygu, and arrives on a per-(expert, tile mod P) counter;
//    the SECOND arrival of the (gate p, up p+P) pair does silu(g)*u (== xq_moe_gate_mul)
//    and the down input transform (== xq_had_suh_multi, per-expert input) for its
//    128-col block -> xhd. Expert 0's CTAs also compute the shared-expert gate
//    sigmoid per row (== the first half of xq_moe_combine, its 1024-thread tree
//    emulated exactly) -> sgv[m].
//  down (xq_moe_dn_epi): every CTA applies its svh (== xq_had_svh_multi) -> yd and
//    arrives on a per-N-tile counter against the device esel; the LAST arrival computes
//    moe_out = f16(sgv*ysh + sum_j wts*yd) for the tile in the exact j order
//    (== xq_moe_combine; the shared expert now runs BEFORE the routed down).
//  Every rounding point is the old kernels': same f16 stores/reloads, the same
//  butterfly, and every rounding pinned with intrinsics:
//   - combine: the old PTX's own explicit mul.f32 then fma.rn chain; sigmoid dot:
//     fma.rn partials + add tree (NVVM emitted those roundings explicitly).
//   - svh / silu*mul: no product feeds an add (mul -> cvt), nothing to contract.
//   - suh (x*s products feeding the FIRST butterfly stage): the old PTX leaves
//     mul.f32/add.f32 unrounded and ptxas contracts them context-dependently. WP20-v1c:
//     the epilogue's suh is PINNED op by op to xq_had_suh_multi's served SASS (FMUL
//     q = x1*s1; FFMA x0*s0 + q; FFMA x0*s0 - q; FADD / FSEL+FADD butterfly; FMUL by
//     1/sqrt(128)) with explicit .rn intrinsics — see wp20_suh_had. (v1/v1b's runtime
//     81-convention calibration could never succeed: x and s are f16 values, so x*s is
//     EXACT in f32 and every contraction gives the same bits — all 81 matched, "no
//     unique match", rung 2 OFF.) The boot check (xq_wp20_suh_cal) now only VERIFIES
//     the pinned sequence against the JIT'd xq_had_suh_multi, else the old path runs.
//  Counters self re-arm (the last arrival zeroes its counter; ONE real zero at
//  allocation) — graph-replay safe, no memset node.
//  Top-k (WP20-v1b): the down epilogue's combine is a compile-time TOPK template (the
//  j loops unroll into registers — AGENTS §4: no runtime-indexed per-thread arrays), one
//  entry per served top-k: xq_moe_dn_epi[_sh]_k8 and _k10 (Qwen3.8-Flash-Next = top-10).
//  The launcher (FwdModel::wp20_select) picks the instance from config and falls back
//  LOUDLY for any other top-k; the kernel traps on a top-k/instance mismatch.
// ===========================================================================

// WP20-v1c: == xq_had_suh_multi's per-lane body, PINNED op by op to its SERVED SASS:
// v = H128(x * s), 4 per lane, unrounded f32 out (the callers round with xq_f2h, the old
// kernel's F2FP.F16.F32 RN). The sequence is read off `ptxas -arch=sm_121` of the shipped
// PTX (the driver JIT's SASS; cuobjdump -sass -fun xq_had_suh_multi):
//   stage len 1  : FMUL q1 = x1*s1 ; FFMA a0 = x0*s0 + q1 ; FFMA a1 = x0*s0 - q1  (pair 2,3 alike)
//   stage len 2  : FADD a0 + a2 ; FADD a1 + a3 ; FADD a0 - a2 ; FADD a1 - a3
//   len 4 .. 64  : SHFL.BFLY t ; FSEL w = (lane & mask) ? -v : v ; FADD t + w
//   normalize    : FMUL v * 0x3db504f3 (1/sqrt(128) in f32)
// Every op is an explicit .rn intrinsic (fma.rn / mul.rn / add.rn / sub.rn in the PTX), which
// ptxas may neither contract nor split, so every inlining context (xq_moe_gu_epi[_sh],
// xq_wp20_suh_cal) executes exactly this sequence — no runtime convention, nothing to
// calibrate. It is ALSO value-exact against any contraction ptxas could pick for the old
// source: x and s are f16 values, so x*s is exact in f32 (11 + 11 significand bits <= 24,
// |x*s| in [2^-48, 2^32): no rounding, no underflow, no overflow), hence
// fma(x0,s0,q) == rn(rn(x0*s0) + q) == fma(-x1,s1,rn(x0*s0)) bit for bit (incl. signed zeros).
// That exactness is also why v1/v1b's 81-convention calibration always found 81 matches and
// never a "unique" one (the WP20 FALLBACK seen at boot on preview p2).
__device__ __forceinline__ void wp20_suh_had(const float x[4], const float s[4], float v[4], int lane) {
    {
        const float q1 = __fmul_rn(x[1], s[1]);
        const float q3 = __fmul_rn(x[3], s[3]);
        const float a0 = __fmaf_rn(x[0], s[0], q1);
        const float a1 = __fmaf_rn(x[0], s[0], -q1);
        const float a2 = __fmaf_rn(x[2], s[2], q3);
        const float a3 = __fmaf_rn(x[2], s[2], -q3);
        v[0] = __fadd_rn(a0, a2);                 // exl3_had128 stage len = 2
        v[1] = __fadd_rn(a1, a3);
        v[2] = __fsub_rn(a0, a2);
        v[3] = __fsub_rn(a1, a3);
    }
    #pragma unroll
    for (int s2 = 2; s2 < 7; ++s2) {              // exl3_had128 cross-lane stages
        const unsigned mask = (unsigned)(1u << (s2 - 2));
        const bool hi = (lane & mask) != 0;       // hi: t - v, else v + t (== t + v)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask);
            v[r] = __fadd_rn(t, hi ? -v[r] : v[r]);
        }
    }
    const float hnorm = 0.08838834764831845f;     // 1/sqrt(128), exl3_had128's
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __fmul_rn(v[r], hnorm);
}

// Boot VERIFICATION of the pinned sequence (FwdModel::load via exl3_bench::wp20_calibrate):
// xq_had_suh_multi's exact signature, indexing, loads and f16 store, with its arithmetic
// replaced by wp20_suh_had. The host launches both on the same inputs and requires every
// output byte equal; any mismatch keeps rung 2 off (loud WP20 FALLBACK).
extern "C" __global__ void xq_wp20_suh_cal(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                           const int* __restrict__ idxmap,
                                           __half* __restrict__ xh, int M, int K,
                                           const int* __restrict__ esel,
                                           int per_expert_in) {
    int nblk = K >> 7;
    int blk = blockIdx.x % nblk;
    long long em = blockIdx.x / nblk;
    int m = (int)(em % M);
    int e = (int)(em / M);
    if (e >= *esel) return;
    int lane = threadIdx.x & 31;
    long long base_in = (long long)(per_expert_in ? e : 0) * M * K + (long long)m * K
                      + (long long)blk * 128 + lane * 4;
    long long base_out = ((long long)e * M + m) * K + (long long)blk * 128 + lane * 4;
    const __half* sr = suh_all + (long long)idxmap[e] * K + (long long)blk * 128 + lane * 4;
    float xs[4], ss[4], v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) { xs[r] = xq_h2f(x[base_in + r]); ss[r] = xq_h2f(sr[r]); }
    wp20_suh_had(xs, ss, v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = xq_f2h(v[r]);
}

// 4 consecutive f16 (element j in bits [16j, 16j+16) of the pair) <-> one 8-B access.
__device__ __forceinline__ void wp20_unpack4(uint2 w, float v[4]) {
    v[0] = xq_h2f(__ushort_as_half((unsigned short)(w.x & 0xFFFFu)));
    v[1] = xq_h2f(__ushort_as_half((unsigned short)(w.x >> 16)));
    v[2] = xq_h2f(__ushort_as_half((unsigned short)(w.y & 0xFFFFu)));
    v[3] = xq_h2f(__ushort_as_half((unsigned short)(w.y >> 16)));
}
__device__ __forceinline__ uint2 wp20_pack4(const __half o[4]) {
    return make_uint2((unsigned)__half_as_ushort(o[0]) | ((unsigned)__half_as_ushort(o[1]) << 16),
                      (unsigned)__half_as_ushort(o[2]) | ((unsigned)__half_as_ushort(o[3]) << 16));
}

// A5-K7 (FOLD = true, moe.gu_fold): the gate/up A operand is FORMED here instead of copied from
// xh_e — xh = x (the [M][K] hidden, shared by every expert), suh = the expert's suh row; per
// (row, 128-block) A = f16(wp20_suh_had(x, suh)): the pinned, boot-verified xq_had_suh_multi
// sequence on the same h2f'd halves, then xq_f2h (== the bytes xq_had_suh_multi wrote to xh_e,
// which the FOLD = false copy stages). The expert trellis's first PF_D k16 rows are L2-prefetched
// BEFORE the latency-bound transform (value-transparent), so the DRAM stream starts under it.
template <int BITS, bool DIET, bool FOLD = false>
__device__ __forceinline__ void wp20_a1_mainloop(const uint16_t* __restrict__ trellis,
                                                 const __half* __restrict__ xh,
                                                 __half* a1_sa,
                                                 int M, int K, int N, int cta_tile,
                                                 float (&acc)[2][4],
                                                 const __half* __restrict__ suh = nullptr) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int SK = K + 8;
    // S-A3-n F4 / S-A3-o O1 L2 prefetch constants (value-transparent)
    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < BITS * 32;
    if constexpr (FOLD) {
        const uint32_t* ring0 = (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
        if (pf_lane) {
            const char* p0 = (const char*)ring0 + lane * 32;
            const size_t st = (size_t)nb * 8u * BITS * 4;
            #pragma unroll
            for (int d = 0; d < PF_D; ++d)
                if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * st));
        }
        constexpr int FU = 4;                             // transforms in flight per warp
        const int nblk = K >> 7, nt = M * nblk;
        for (int b0 = warp; b0 < nt; b0 += 8 * FU) {      // warp-uniform (full-warp shuffles)
            float xs[FU][4], ss[FU][4];
            #pragma unroll
            for (int u = 0; u < FU; ++u) {
                const int bi = b0 + 8 * u;
                if (bi < nt) {
                    const int r = bi / nblk, blk = bi - r * nblk;
                    wp20_unpack4(__ldg((const uint2*)(xh + (size_t)r * K + blk * 128 + lane * 4)), xs[u]);
                    wp20_unpack4(__ldg((const uint2*)(suh + blk * 128 + lane * 4)), ss[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < FU; ++u) {
                const int bi = b0 + 8 * u;
                if (bi < nt) {
                    const int r = bi / nblk, blk = bi - r * nblk;
                    float v[4];
                    wp20_suh_had(xs[u], ss[u], v, lane);      // == xq_had_suh_multi (pinned SASS sequence)
                    __half o[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
                    *(uint2*)&a1_sa[(size_t)r * SK + blk * 128 + lane * 4] = wp20_pack4(o);
                }
            }
        }
    } else {
        const int vpr = K >> 3;
        for (int i = tid; i < M * vpr; i += blockDim.x) {
            const int r = i / vpr, c = i - r * vpr;
            *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + c * 8];
        }
    }
    __syncthreads();

    constexpr int W = 8 * BITS;
    int s_i0m[8], s_i1m[8], s_sf[8];      // SHFL body (DIET = false) only
    int offA = 0, offB = 0, sh7 = 0;      // diet body only
    if constexpr (DIET) {
        static_assert(BITS == 3 || BITS == 4, "word diet: 2 ring words per lane per step hold for b3/b4 only");
        // slot j = 0 (h = 0, s = 0) through the SAME tmap arithmetic as the old body
        const int r = 2 * (lane & 3), c = lane >> 2;
        const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
        const int t0 = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
        const int b10 = (t0 + 257) * BITS;        // window END of slot 0; slot j ends at b10 + j*BITS
        const int A = (b10 - 16) >> 5;            // first ring word any of the 8 windows touches
        offA = A % W;
        offB = (A + 1) % W;
        sh7 = 32 * (A + 2) - (b10 + 7 * BITS);    // >= 0: slot 7's window ends inside word A+1
    } else {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
                const int c = 8 * h + (lane >> 2);
                const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
                const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                                  + 4 * (c >> 3) + 32 * (c & 1);
                const int b1 = (t + 257) * BITS;
                const int i0 = (b1 - 16) >> 5;
                const int i1 = (b1 - 1) >> 5;
                s_i0m[h * 4 + s] = i0 % (BITS * 8);
                s_i1m[h * 4 + s] = i1 % (BITS * 8);
                s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;

    // S-A3-n F4 / S-A3-o O1 L2 prefetch, unchanged (value-transparent; FOLD issued it above).
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;
    if (!FOLD && pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    for (int kb = 0; kb < KB; ++kb) {
        if (pf_lane && kb + PF_D < KB)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
        pf_ptr += pf_step;
        uint32_t bfrag[2][2];
        if constexpr (DIET) {
            const unsigned wA = __ldg(ring + offA);
            const unsigned wB = __ldg(ring + offB);
            const unsigned long long P = ((((unsigned long long)wA) << 32) | (unsigned long long)wB) >> sh7;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    const unsigned idx = (unsigned)(P >> (BITS * (7 - j))) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
        } else {
            // the A1_SHFL block of exl3_gemm_body_a1, verbatim
            const unsigned w0 = (lane < W) ? __ldg(ring + lane) : 0u;
            const unsigned w1 = (W > 32 && lane + 32 < W) ? __ldg(ring + lane + 32) : 0u;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (W > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
        }
        ring += ring_kstride;
        const int col = kb * 16;
        const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }
}

// The CTA's f16 tile (the exact bytes the old epilogue stored as y_raw) into smem [M][128].
// Caller guarantees every warp is done reading the A operand from this smem.
__device__ __forceinline__ void wp20_stage_tile(__half* yt, const float (&acc)[2][4], int M) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lc = 16 * warp + 8 * h + 2 * tig;
        if (gid < M) *(__half2*)&yt[gid * 128 + lc] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *(__half2*)&yt[(gid + 8) * 128 + lc] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
}

// == xq_had_svh_multi on one 128-col block of one row: y = f16(H128(yraw) * svh), lane owns
// elements lane*4..+3 (yr: the f16 tile row in smem; sr: the block's svh). Returned packed.
__device__ __forceinline__ uint2 wp20_svh_row(const __half* yr, const __half* sr, int lane) {
    float v[4], s[4];
    wp20_unpack4(*(const uint2*)(yr + lane * 4), v);
    exl3_had128(v, lane);
    wp20_unpack4(__ldg((const uint2*)(sr + lane * 4)), s);
    __half o[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j] * s[j]);
    return wp20_pack4(o);
}

// Last-arrival election: returns true (CTA-uniform) in exactly ONE of the `target` CTAs that
// arrive on `cnt` in this launch — the last — after every arriving CTA's prior global stores
// are visible to it; that CTA re-arms the counter to 0 for the next launch.
__device__ __forceinline__ bool wp20_arrive(unsigned* cnt, unsigned target, int* s_flag) {
    // CUTLASS GenericBarrier's publication (split-K serial / stream-K fixup): the CTA barrier
    // orders every thread's stores before thread 0's gpu-scope acq_rel fence + counter bump;
    // the elected thread's fence after the bump is the acquire side, the second barrier hands
    // it to the CTA. Readers use ld.global.cg (L2) for the other CTAs' tiles.
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const unsigned old = atomicAdd(cnt, 1u);
        const int last = old == target - 1u;
        if (last) {
            atomicExch(cnt, 0u);                       // all `target` arrivals are in: re-arm
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        *s_flag = last;
    }
    __syncthreads();
    return *s_flag != 0;
}

// FOLD (A5-K7, entry xq_moe_gu_epi_f): A is formed in the prologue from x and suh_gu_all[ix]
// (wp20_a1_mainloop<.., FOLD>) — the xq_had_suh_multi launch before this kernel goes; xh unused.
template <bool DIET, bool FOLD = false>
__device__ __forceinline__ void wp20_moe_gu(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ xh, __half* __restrict__ ygu,
                                            int M, int K, int N, const int* __restrict__ esel,
                                            const int* __restrict__ idxmap,
                                            const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                                            __half* __restrict__ xhd, unsigned* __restrict__ cnt,
                                            const __half* __restrict__ x, const __half* __restrict__ sg,
                                            float* __restrict__ sgv,
                                            const __half* __restrict__ suh_gu_all = nullptr) {
    extern __shared__ __half a1_sa[];
    __shared__ int s_flag;
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ix = idxmap[expert];
    float acc[2][4];
    if constexpr (FOLD)
        wp20_a1_mainloop<3, DIET, true>(base + (size_t)offs[expert], x, a1_sa, M, K, N, tile, acc,
                                        suh_gu_all + (long long)ix * K);
    else
        wp20_a1_mainloop<3, DIET>(base + (size_t)offs[expert], xh + (long long)expert * M * K, a1_sa,
                                  M, K, N, tile, acc);
    __syncthreads();                                   // every warp is done reading A
    __half* yt = a1_sa;                                // [M][128]
    wp20_stage_tile(yt, acc, M);
    __syncthreads();
    // own svh (== xq_had_svh_multi, block `tile`): kept in smem, published to ygu
    const int cta_n = tile * 128;
    __half* ye = ygu + (long long)expert * M * N;
    for (int r = warp; r < M; r += 8) {
        const uint2 o = wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
        *(uint2*)(yt + r * 128 + lane * 4) = o;
        *(uint2*)(ye + (long long)r * N + cta_n + lane * 4) = o;
    }
    const int P = tiles >> 1;                          // gate tiles [0, P), up tiles [P, 2P)
    const int p = tile < P ? tile : tile - P;
    if (wp20_arrive(cnt + expert * P + p, 2u, &s_flag)) {
        // second arrival of the (gate p, up p+P) pair: silu*mul + the down input transform
        const int KD = N >> 1;                         // down K = moe_intermediate
        const int ptile = tile < P ? tile + P : tile - P;
        const bool own_gate = tile < P;
        for (int r = warp; r < M; r += 8) {
            const uint2 pw = __ldcg((const uint2*)(ye + (long long)r * N + ptile * 128 + lane * 4));
            const uint2 ow = *(const uint2*)(yt + r * 128 + lane * 4);
            float g[4], u[4], sv[4], dv[4], v[4];
            wp20_unpack4(own_gate ? ow : pw, g);
            wp20_unpack4(own_gate ? pw : ow, u);
            wp20_unpack4(__ldg((const uint2*)(suh_d_all + (long long)ix * KD + p * 128 + lane * 4)), sv);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                dv[j] = xq_h2f(xq_f2h(xq_silu(g[j]) * u[j]));   // == xq_moe_gate_mul (the f16 din)
            wp20_suh_had(dv, sv, v, lane);                        // == xq_had_suh_multi (pinned SASS sequence)
            __half o[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
            *(uint2*)(xhd + ((long long)expert * M + r) * KD + p * 128 + lane * 4) = wp20_pack4(o);
        }
    }
    // Shared-expert gate sigmoid, once per row, by expert 0's CTAs (row r -> tile r % tiles):
    // xq_moe_combine's per-row prologue with its 1024-thread shape emulated (virtual thread
    // vt = tid + 256q owns c = vt, vt+1024, ... ascending; the same halving tree), and the
    // old PTX roundings pinned (fma.rn partials, add tree, xq_sig).
    if (expert == 0) {
        float* red = (float*)a1_sa;                    // 4 KB <= the A-once smem (K = hidden)
        for (int r = tile; r < M; r += tiles) {
            __syncthreads();                           // smem reuse (tile staging / previous row)
            #pragma unroll 1
            for (int q = 0; q < 4; ++q) {
                const int vt = tid + 256 * q;
                float part = 0.0f;
                #pragma unroll 1
                for (int c = vt; c < K; c += 1024)
                    part = __fmaf_rn(xq_h2f(sg[c]), xq_h2f(x[(long long)r * K + c]), part);
                red[vt] = part;
            }
            __syncthreads();
            for (int s2 = 512; s2 > 0; s2 >>= 1) {
                for (int i = tid; i < s2; i += 256) red[i] = __fadd_rn(red[i], red[i + s2]);
                __syncthreads();
            }
            if (tid == 0) sgv[r] = xq_sig(red[0]);
        }
    }
}

// Down epilogue, templated on the combine's top-k (instances: 8, 10 — see the entries below).
// Shared memory (dynamic, the launcher passes a1_smem(M, K) = M*(K+8)*2 B, K = moe_intermediate):
//   mainloop: the A operand [M][K+8] f16;  after it (A dead): yt [M][128] f16 at 0, then the
//   combine tables s_slot [M][TOPK] i32 + s_w [M][TOPK] f32 at byte 256*M.
//   Need 256*M + 8*M*TOPK <= M*(K+8)*2  <=>  8*TOPK <= 2K - 240: at K = 640 any TOPK <= 130
//   fits for EVERY M (M = 8, TOPK = 10: 2688 B of the 10368 B); the launcher re-checks.
template <bool DIET, int TOPK>
__device__ __forceinline__ void wp20_moe_dn(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ xhd, __half* __restrict__ yd,
                                            int M, int K, int N, const int* __restrict__ esel,
                                            const int* __restrict__ idxmap, const __half* __restrict__ svh_all,
                                            unsigned* __restrict__ cnt, const __half* __restrict__ ysh,
                                            const int* __restrict__ ids, const float* __restrict__ wts,
                                            const int* __restrict__ slotmap, const float* __restrict__ sgv,
                                            __half* __restrict__ out, int topk) {
    static_assert(TOPK >= 1 && TOPK <= 16, "wp20_moe_dn: TOPK instance out of the planned range");
    extern __shared__ __half a1_sa[];
    __shared__ int s_flag;
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    const int live = *esel;
    if (expert >= live) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ix = idxmap[expert];
    float acc[2][4];
    wp20_a1_mainloop<3, DIET>(base + (size_t)offs[expert], xhd + (long long)expert * M * K, a1_sa,
                              M, K, N, tile, acc);
    __syncthreads();
    __half* yt = a1_sa;                                // [M][128]
    wp20_stage_tile(yt, acc, M);
    // The combine tables' routing rows (ids, wts: [M][TOPK], written by the router launches
    // before this grid) go to smem NOW by cp.async (baseline sm_80+ 4-B .ca form), one entry per
    // thread (M*TOPK <= 16*16 = blockDim at the a1 cap): ids -> s_slot (staging; the last arrival
    // rewrites it in place to slotmap[id]), wts -> s_w (final). No register ever holds them, so
    // nothing can stall on them before the arrival (a register prefetch did: ptxas MOVed the
    // loaded id right after the next barrier, in every CTA); each copying thread waits on its
    // own group after the svh stores (long since landed), and the arrival's barriers publish
    // the smem to the CTA. The last arrival then pays only the slotmap hop, not ids -> slotmap.
    // Region [256*M, 256*M + 8*M*TOPK) B: A is dead (the barrier above), yt is [0, 256*M).
    const int nrt = M * TOPK;
    int* s_slot = (int*)(a1_sa + M * 128);             // [M][TOPK] (fit: see the header comment)
    float* s_w = (float*)(s_slot + nrt);
    const bool rt_mine = tid < nrt;
    if (rt_mine) {
        const unsigned d_id = (unsigned)__cvta_generic_to_shared(s_slot + tid);
        const unsigned d_w = (unsigned)__cvta_generic_to_shared(s_w + tid);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d_id), "l"(ids + tid) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d_w), "l"(wts + tid) : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    __syncthreads();
    const int cta_n = tile * 128;
    for (int r = warp; r < M; r += 8)                  // == xq_had_svh_multi -> yd
        *(uint2*)(yd + ((long long)expert * M + r) * N + cta_n + lane * 4) =
            wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
    if (rt_mine) asm volatile("cp.async.wait_all;\n" ::: "memory");   // before exit / publication
    if (!wp20_arrive(cnt + tile, (unsigned)live, &s_flag)) return;
    // Last arrival for this N-tile: == xq_moe_combine on columns [cta_n, cta_n + 128).
    if (topk != TOPK) __trap();                        // launcher bug: never silently mis-combine
    if (rt_mine) s_slot[tid] = slotmap[s_slot[tid]];   // own entry: id -> slot (== slotmap[ids[i]])
    for (int i = tid + blockDim.x; i < nrt; i += blockDim.x) {   // M*TOPK > blockDim: never at the a1 cap
        s_slot[i] = slotmap[ids[i]];
        s_w[i] = wts[i];
    }
    __syncthreads();
    // 4 consecutive columns per thread (8-B loads/stores; 32 threads per row, so M <= 8 is ONE
    // pass with all TOPK L2 loads in flight at once). Per element the SAME op sequence as
    // xq_moe_combine: a = rn(sgv * ysh); for j ascending: a = fma.rn(w_j, y_j, a); f16(a).
    for (int e = tid; e < M * 32; e += blockDim.x) {
        const int r = e >> 5, i = cta_n + ((e & 31) << 2);
        const int* sl = s_slot + r * TOPK;
        const float* sw = s_w + r * TOPK;
        uint2 raw[TOPK];
        #pragma unroll
        for (int j = 0; j < TOPK; ++j) {               // unconditional (a -1 slot reads slot 0, unused)
            const int s = sl[j] < 0 ? 0 : sl[j];
            raw[j] = __ldcg((const uint2*)(yd + ((long long)s * M + r) * N + i));
        }
        float ys[4], a[4];
        wp20_unpack4(__ldg((const uint2*)(ysh + (long long)r * N + i)), ys);
        const float g = sgv[r];
        #pragma unroll
        for (int q = 0; q < 4; ++q) a[q] = __fmul_rn(g, ys[q]);
        #pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            if (sl[j] >= 0) {
                float y[4];
                wp20_unpack4(raw[j], y);
                const float w = sw[j];
                #pragma unroll
                for (int q = 0; q < 4; ++q) a[q] = __fmaf_rn(w, y[q], a[q]);
            }
        }
        __half o[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) o[q] = xq_f2h(a[q]);
        *(uint2*)(out + (long long)r * N + i) = wp20_pack4(o);
    }
}

// Rung 1 alone (GB10_WP20_EPI=0): the diet body behind xq_gemm_grouped_a1b3's exact contract.
template <bool DIET>
__device__ __forceinline__ void wp20_grouped_plain(const uint16_t* __restrict__ base,
                                                   const uint64_t* __restrict__ offs,
                                                   const __half* __restrict__ xh,
                                                   __half* __restrict__ y,
                                                   int M, int K, int N,
                                                   const int* __restrict__ esel) {
    extern __shared__ __half a1_sa[];
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    float acc[2][4];
    wp20_a1_mainloop<3, DIET>(base + (size_t)offs[expert], xh + (long long)expert * M * K, a1_sa,
                              M, K, N, tile, acc);
    __half* ye = y + (long long)expert * M * N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = tile * 128 + 16 * warp + 8 * h + 2 * tig;
        if (gid < M) *(__half2*)&ye[(size_t)gid * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (gid + 8 < M) *(__half2*)&ye[(size_t)(gid + 8) * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3_wd(const uint16_t* __restrict__ base,
                        const uint64_t* __restrict__ offs,
                        const __half* __restrict__ xh,
                        __half* __restrict__ y,
                        int M, int K, int N, int bits,
                        const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    wp20_grouped_plain<true>(base, offs, xh, y, M, K, N, esel);
}

#define WP20_GU_ENTRY(NAME, DIET)                                                                   \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,                    \
     __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,          \
     const __half* __restrict__ sg, float* __restrict__ sgv) {                                    \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_gu<DIET>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, xhd, cnt,   \
                      x, sg, sgv);                                                                 \
}
WP20_GU_ENTRY(xq_moe_gu_epi, true)
WP20_GU_ENTRY(xq_moe_gu_epi_sh, false)
// A5-K7 (moe.gu_fold): xq_moe_gu_epi with xq_had_suh_multi folded into the prologue. Same 16
// params (xh ignored) + suh_gu_all (the gate/up suh table, [experts][K]). Bitwise == the
// xq_had_suh_multi -> xq_moe_gu_epi pair (--probe-exl3-binv EXL3-MOE-EPI, "fold" rows).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_moe_gu_epi_f(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                const __half* __restrict__ xh, __half* __restrict__ ygu, int M, int K, int N,
                const int* __restrict__ esel, const int* __restrict__ idxmap,
                const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                __half* __restrict__ xhd, unsigned* __restrict__ cnt, const __half* __restrict__ x,
                const __half* __restrict__ sg, float* __restrict__ sgv, const __half* __restrict__ suh_gu_all) {
    XQ_PDL_ENTRY();
    wp20_moe_gu<true, true>(base, offs, xh, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, xhd, cnt,
                            x, sg, sgv, suh_gu_all);
}

// One entry per (decode body, top-k). Adding a top-k = one more pair here + its name in
// src/exl3_forward.rs wp20_dn_fn (the launcher logs a loud fallback for any top-k without one).
#define WP20_DN_ENTRY(NAME, DIET, TOPK)                                                             \
extern "C" __global__ void __launch_bounds__(256, 3)                                              \
NAME(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,                        \
     const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,                \
     const int* __restrict__ esel, const int* __restrict__ idxmap,                                \
     const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,                              \
     const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,  \
     const int* __restrict__ slotmap, const float* __restrict__ sgv, __half* __restrict__ out,    \
     int topk) {                                                                                   \
    XQ_PDL_ENTRY();                                                                                \
    wp20_moe_dn<DIET, TOPK>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids,  \
                            wts, slotmap, sgv, out, topk);                                         \
}
WP20_DN_ENTRY(xq_moe_dn_epi_k8, true, 8)
WP20_DN_ENTRY(xq_moe_dn_epi_sh_k8, false, 8)
WP20_DN_ENTRY(xq_moe_dn_epi_k10, true, 10)
WP20_DN_ENTRY(xq_moe_dn_epi_sh_k10, false, 10)

// ===========================================================================
// W4/MOE (decode wave 4, 2026-09-27): PERSISTENT grouped-expert kernels — the W3/LMH byte
// schedule (persistent grid, cp.async.cg 16-B ring + L2 evict_first, A staged in smem) under
// the WP20 rung-2 epilogues, plus lever (b): the gate/up input transform folded in.
//
// Target contract: sm_121 BASELINE — cp.async.cg 16 B (+ L2::cache_hint / createpolicy
// evict_first), mma.sync.m16n8k16 f16->f32, dp4a, 64-bit shifts, fma.rn.f16x2, device-scope
// atomics + fence.acq_rel.gpu. No f/a-suffix feature. 256 threads, 1..3 CTAs/SM (the host sizes
// the grid from the occupancy API with the real dynamic smem), 0 stack / 0 spill (ptxas -v).
//
// Shape class: Qwen3.8-Flash-Next routed experts — 3-bit trellis; gate/up K = hidden 2560,
// N = 2*mi = 1280 (tiles [0,P) gate, [P,2P) up); down K = mi 640, N = 2560; top-k 8 / 10
// instances; M = verify width 1..9 (the a1b3 envelope WP20 rung 2 runs in); live experts
// <= MPK_EMAX. No spin wait anywhere: the only inter-CTA protocol is WP20's non-blocking
// last-arrival counters, so any residency (a co-running graph branch, a partial wave) is safe.
//
// Why (ledger p4c gap #1): the WP20 bodies stream the expert trellis at 80-83% of 238 GB/s at
// every width — each warp reads one 96-B window per k16 step through a dependent __ldg (24
// warps/SM, L2 prefetch 8 steps ahead). W3/LMH moves the same trellis format at 244 GB/s with a
// deep cp.async ring. Here:
//  * work item = (live expert slot e, 128-col tile t) over the FULL K (no split-K), i = e*T + t,
//    T = N/128. A persistent CTA walks a CONTIGUOUS item range [b*n/G, (b+1)*n/G) (default:
//    consecutive items share the expert, so A is re-staged only when the expert changes) or the
//    interleaved sequence b, b+G, ... (MPK_IL: the LMH order — A per item). n = esel * T.
//  * the trellis streams through an NST-slot smem ring, MPK_KSTEP k16 rows x 768 B (the tile's
//    8 blocks) per slot, by cp.async.cg + evict_first; NST-1 slots stay in flight ACROSS item
//    boundaries (the producer walks the item sequence ahead of the consumer, LMH-style).
//  * A (the expert's [M][K] f16 input rows) lives in smem [M][K+8] (the a1b3 layout); a new
//    expert's copy is issued right after the old A's last reader (the item epilogue's first
//    barrier), so it overlaps the epilogue, by the 2 warps that issue no ring chunk — their wait
//    on A never drains the ring's in-flight slots. Gate/up + MPK_FOLD: the copy brings x and suh_e and
//    A = f16(H128(x * suh_e)) is formed in place — xq_had_suh_multi's bytes (that launch goes).
//  * epilogues = WP20 rung 2 per item: own-tile svh -> ygu / yd, the (gate p, up p+P) pair
//    counter -> silu*mul + down suh -> xhd; the per-N-tile last arrival -> combine -> moe_out.
//    The shared-gate sigmoid rows move from expert 0's CTAs to CTAs b < M (row r on CTA r mod G:
//    one row each, the same per-row op sequence); the down routing tables are read once per CTA.
//
// BIT-IDENTITY CONTRACT (== xq_moe_gu_epi / xq_moe_dn_epi_k<TOPK>, hence == the pre-WP20 chain):
//  1. B fragments: a ring slot is a byte copy of the trellis rows; each lane reads the SAME two
//     ring words (offA, offB — WP20 rung 1's diet) of its warp's block and cuts the same 8
//     windows -> same idx -> dp4a(idx*MUL1, 0x01010101, 0x6400) -> fma.rn.f16(., 0x1EEE,
//     0xC931) (hfma2 = two independent fma.rn.f16) -> identical fragment registers.
//  2. A fragments: smem [M][K+8] holds the SAME f16 values the a1b3 body staged (byte copies of
//     xh_e / xhd; MPK_FOLD: wp20_suh_had — the pinned, boot-verified xq_had_suh_multi sequence —
//     on the same x / suh halves, then xq_f2h); rows >= M are 0 (predicated) exactly as there.
//  3. Per output: fp32 accs zero-initialised per item, one mma.m16n8k16 pair per k16 step, kb
//     strictly ascending 0..KB-1 on ONE warp — never split across warps / CTAs / items.
//  4. Epilogues: the WP20 device functions and op sequences unchanged (wp20_stage_tile,
//     wp20_svh_row, wp20_arrive, wp20_suh_had, silu*mul, rn(g*ys) + the ascending fma.rn chain).
//  => every output byte equals the WP20 rung-2 kernels' for ANY grid G, ring depth NST and item
//     order (schedule knobs only); batch invariance is inherited (per-row math never sees M).
// ===========================================================================
#define MPK_THREADS 256
#define MPK_KSTEP 4                        // k16 rows per ring slot
#define MPK_BB 96                          // bytes of one 3-bit 16x16 trellis block
#define MPK_KROW (8 * MPK_BB)              // 768 B: one k16 row of a 128-col tile (8 blocks)
#define MPK_STB (MPK_KSTEP * MPK_KROW)     // 3072 B per ring slot
#define MPK_NCH (MPK_STB / 16)             // 192 16-B chunks per slot: threads 0..191 issue one each
#define MPK_EMAX 256                       // live experts held in the smem routing tables
#define MPK_RTMAX 256                      // M * TOPK entries of the down combine tables
#define MPK_IL 1                           // flags: interleaved item order (else contiguous ranges)
#define MPK_FOLD 2                         // flags (gate/up): A = f16(H128(x * suh_e)) in-kernel

// Dynamic smem (bytes): ring | A [M][K+8] f16 | suh [K] f16 (gate/up + fold) | epi. The epilogue
// region holds the [M][128] f16 tile; gate/up also the 1024-float sigmoid tree (>= 4 KB).
// Every term is a multiple of 16 B (K % 8 == 0). Mirrored by exl3_forward::mpk_smem_bytes.
__host__ __device__ __forceinline__ int mpk_smem_bytes(int nst, int M, int K, int gu, int fold) {
    const int epi = gu ? (M * 256 > 4096 ? M * 256 : 4096) : M * 256;
    return nst * MPK_STB + M * (K + 8) * 2 + ((gu && fold) ? K * 2 : 0) + epi;
}

// A's M rows (row stride K halves in global) into sa [M][K+8] and, when `suh` is given, the
// K-half suh row into su — issued ONLY by the A threads (tid >= MPK_NCH: warps 6-7, which issue
// no ring chunk), as one async group of theirs. Their other groups are the ring's EMPTY ones, so
// their cp.async.wait_all (mpk_wait_a) retires exactly the A copies and never drains the ring
// (commit groups retire FIFO per thread: a wait on A by a ring thread would). Plain cp.async.cg
// (default L2 policy: x / xh_e / xhd are re-read by other CTAs).
#define MPK_ATHR (MPK_THREADS - MPK_NCH)
__device__ __forceinline__ void mpk_issue_a(unsigned sa_u, const __half* __restrict__ src, int M, int K,
                                            unsigned su_u, const __half* __restrict__ suh) {
    const int ta = (int)threadIdx.x - MPK_NCH;
    if (ta < 0) return;                                   // warp-uniform (MPK_NCH = 6 warps)
    const int vpr = K >> 3, SK = K + 8;
    for (int i = ta; i < M * vpr; i += MPK_ATHR) {
        const int r = i / vpr, c = i - r * vpr;
        lmh_cp16(sa_u + (unsigned)((r * SK + c * 8) * 2), src + (size_t)r * K + c * 8);
    }
    if (suh != nullptr)
        for (int i = ta; i < vpr; i += MPK_ATHR)
            lmh_cp16(su_u + (unsigned)(i * 16), suh + (size_t)i * 8);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
// A landed and visible CTA-wide (the A threads retire their groups; then one barrier).
__device__ __forceinline__ void mpk_wait_a() {
    if ((int)threadIdx.x >= MPK_NCH) asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
}

// MPK_FOLD: A[r][blk] <- f16(H128(A[r][blk] * suh[blk])) in place (A holds x's rows) — per
// (row, 128-block) exactly xq_had_suh_multi's lane body: the same h2f loads, wp20_suh_had (the
// pinned sequence WP20 verifies against the JIT'd xq_had_suh_multi at boot), xq_f2h. Caller:
// copies landed + a barrier before; a barrier after. Warp-uniform loop (full-warp shuffles).
__device__ __forceinline__ void mpk_fold_a(__half* sa, const __half* su, int M, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = K >> 7, SK = K + 8;
    #pragma unroll 2
    for (int bi = warp; bi < M * nblk; bi += MPK_THREADS / 32) {
        const int r = bi / nblk, blk = bi - r * nblk;
        __half* ap = sa + r * SK + blk * 128 + lane * 4;
        float xs[4], ss[4], v[4];
        wp20_unpack4(*(const uint2*)ap, xs);
        wp20_unpack4(*(const uint2*)(su + blk * 128 + lane * 4), ss);
        wp20_suh_had(xs, ss, v, lane);
        __half o[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
        *(uint2*)ap = wp20_pack4(o);
    }
}

// Producer state: the item / k-slot / ring slot it issues next, the running count, and this
// thread's chunk source at k16 row 0 of the item (re-derived only when the item changes).
struct MpkProd {
    const unsigned char* src;
    int item, ks, slot, i;
};
// Chunk tid < MPK_NCH of a slot = (k16 row cg, bytes cr*16..+16) of the item's 768-B tile row:
// expert base (s_off halves) + tile t * 768 + cg * rowB + cr * 16 (c_src holds the last two).
__device__ __forceinline__ void mpk_prod_seek(MpkProd& p, const unsigned char* trb, const unsigned long long* s_off,
                                              int T, long long c_src) {
    const int e = p.item / T;
    p.src = trb + s_off[e] * 2ull + (long long)(p.item - e * T) * MPK_KROW + c_src;
}
// Issue ring slot p.slot (MPK_KSTEP k16 rows of the current item from row p.ks*MPK_KSTEP) as one
// cp.async group (an EMPTY group past the schedule, keeping the wait_group accounting aligned).
template <int NST>
__device__ __forceinline__ void mpk_prod_step(MpkProd& p, unsigned ring_u, unsigned c_dst, bool c_live,
                                              long long rowB, int NKS, int total, int i_step, int i_end,
                                              const unsigned char* trb, const unsigned long long* s_off, int T,
                                              long long c_src, unsigned long long pol) {
    if (p.i < total) {
        if (c_live)
            lmh_cp16_ef(ring_u + (unsigned)(p.slot * MPK_STB) + c_dst, p.src + (long long)p.ks * MPK_KSTEP * rowB, pol);
        ++p.i;
        if (++p.ks == NKS) {
            p.ks = 0;
            p.item += i_step;
            if (p.item < i_end) mpk_prod_seek(p, trb, s_off, T, c_src);
        }
        if (++p.slot == NST) p.slot = 0;
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Consumer: MPK_KSTEP k16 steps of this warp's 16-col block from one ring slot (`sw` = slot
// base + warp's block offset): WP20 rung 1's diet decode on the slot's words, the A fragments
// from the staged rows, the mma pair — the a1b3/WP20 per-step instruction values exactly.
__device__ __forceinline__ void mpk_consume(const unsigned char* sw, const __half* sa, int SK, int kb0,
                                            int offA, int offB, int sh7, bool l0, bool l1, bool hi_rows,
                                            int gid, int tig, float (&acc)[2][4]) {
    const __half2 k_inv2  = __half2half2(__ushort_as_half((unsigned short)0x1EEE));
    const __half2 k_bias2 = __half2half2(__ushort_as_half((unsigned short)0xC931));
    #pragma unroll
    for (int g = 0; g < MPK_KSTEP; ++g) {
        const uint32_t* w = (const uint32_t*)(sw + g * MPK_KROW);
        const unsigned wA = w[offA];
        const unsigned wB = w[offB];
        const unsigned long long P = ((((unsigned long long)wA) << 32) | (unsigned long long)wB) >> sh7;
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned sum[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = h * 4 + q;
                const unsigned idx = (unsigned)(P >> (3 * (7 - j))) & 0xFFFFu;
                sum[q] = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
            }
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const __half2 hv = __halves2half2(__ushort_as_half((unsigned short)sum[2 * p]),
                                                  __ushort_as_half((unsigned short)sum[2 * p + 1]));
                const __half2 wv = __hfma2(hv, k_inv2, k_bias2);
                bfrag[h][p] = *reinterpret_cast<const uint32_t*>(&wv);
            }
        }
        const __half* ar0 = sa + gid * SK + (kb0 + g) * 16 + 2 * tig;
        const uint32_t a0 = l0 ? *(const uint32_t*)ar0 : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar0 + 8 * SK) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar0 + 8 * SK + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }
}

// Per-lane diet constants (== wp20_a1_mainloop<3, true>): the two ring words every one of the
// lane's 8 windows lies in, and the shift aligning slot 7's window end.
__device__ __forceinline__ void mpk_diet_consts(int lane, int& offA, int& offB, int& sh7) {
    constexpr int BITS = 3, W = 8 * BITS;
    const int r = 2 * (lane & 3), c = lane >> 2;
    const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
    const int t0 = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
    const int b10 = (t0 + 257) * BITS;
    const int A = (b10 - 16) >> 5;
    offA = A % W;
    offB = (A + 1) % W;
    sh7 = 32 * (A + 2) - (b10 + 7 * BITS);
}

// The CTA's item sequence: contiguous range (default) or interleaved (MPK_IL). CTA-uniform.
__device__ __forceinline__ void mpk_items(int n, int flags, int& i_beg, int& i_step, int& i_end, int& n_items) {
    const int G = (int)gridDim.x, b = (int)blockIdx.x;
    if (flags & MPK_IL) { i_beg = b; i_step = G; i_end = n; }
    else {
        i_beg = (int)(((long long)b * n) / G);
        i_step = 1;
        i_end = (int)(((long long)(b + 1) * n) / G);
    }
    n_items = i_beg < i_end ? (i_end - i_beg + i_step - 1) / i_step : 0;
}

template <int NST>
__device__ __forceinline__ void mpk_gu_body(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ ain, const __half* __restrict__ suh_all,
                                            __half* __restrict__ ygu, int M, int K, int N,
                                            const int* __restrict__ esel, const int* __restrict__ idxmap,
                                            const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                                            __half* __restrict__ xhd, unsigned* __restrict__ cnt,
                                            const __half* __restrict__ x, const __half* __restrict__ sg,
                                            float* __restrict__ sgv, int flags) {
    static_assert(NST >= 3, "ring needs >= 3 slots");
    static_assert(MPK_NCH % 32 == 0 && MPK_NCH < MPK_THREADS, "ring issuers = whole warps, A threads left over");
    extern __shared__ __align__(16) unsigned char mpk_sm[];
    __shared__ unsigned long long s_off[MPK_EMAX];
    __shared__ int s_ix[MPK_EMAX];
    __shared__ int s_flag;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const bool fold = (flags & MPK_FOLD) != 0;
    const int T = N >> 7, NKS = (K >> 4) / MPK_KSTEP, SK = K + 8, P = T >> 1;
    unsigned char* ring = mpk_sm;
    __half* sa = (__half*)(mpk_sm + NST * MPK_STB);
    __half* su = (__half*)((unsigned char*)sa + M * SK * 2);
    __half* epi = (__half*)((unsigned char*)su + (fold ? K * 2 : 0));
    const int live = *esel;
    // Host/device layout tripwires (a short launch traps instead of overrunning smem).
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)mpk_smem_bytes(NST, M, K, 1, fold ? 1 : 0)) __trap();
        if (M < 1 || M > 16 || (N & 255) != 0 || (K & 127) != 0 || (K % (16 * MPK_KSTEP)) != 0) __trap();
        if (live < 0 || live > MPK_EMAX) __trap();
    }
    for (int e = tid; e < live && e < MPK_EMAX; e += MPK_THREADS) { s_off[e] = offs[e]; s_ix[e] = idxmap[e]; }
    int i_beg, i_step, i_end, n_items;
    mpk_items(live * T, flags, i_beg, i_step, i_end, n_items);
    const int total = n_items * NKS;
    __syncthreads();

    // ---- ring prologue (slots 0..NST-2) + the first expert's A, before any other latency ----
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const unsigned sa_u = (unsigned)__cvta_generic_to_shared(sa);
    const unsigned su_u = (unsigned)__cvta_generic_to_shared(su);
    const bool c_live = tid < MPK_NCH;
    const int cg = tid / (MPK_KROW / 16), cr = tid - cg * (MPK_KROW / 16);
    const long long rowB = (long long)(N >> 4) * MPK_BB;   // bytes per k16 row of one expert's trellis
    const long long c_src = (long long)cg * rowB + cr * 16;
    const unsigned c_dst = (unsigned)(cg * MPK_KROW + cr * 16);
    const unsigned char* trb = (const unsigned char*)base;
    MpkProd pr;
    pr.item = i_beg; pr.ks = 0; pr.slot = 0; pr.i = 0; pr.src = trb;
    bool a_pend = false;
    if (n_items > 0) {
        mpk_prod_seek(pr, trb, s_off, T, c_src);
        #pragma unroll 1
        for (int s = 0; s < NST - 1; ++s)
            mpk_prod_step<NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        const int e0 = i_beg / T;
        mpk_issue_a(sa_u, ain + (fold ? 0ll : (long long)e0 * M * K), M, K, su_u,
                    fold ? suh_all + (long long)s_ix[e0] * K : nullptr);
        a_pend = true;
    }
    // ---- shared-expert gate sigmoid, row r on CTA r (mod G): xq_moe_combine's per-row prologue
    // with its 1024-thread tree emulated (vt = tid + 256q owns c = vt, vt + 1024, ... ascending;
    // the same halving tree; fma.rn partials, add tree, xq_sig) — wp20_moe_gu's expert-0 block.
    for (int r = (int)blockIdx.x; r < M; r += (int)gridDim.x) {
        float* red = (float*)epi;                      // 4 KB (the epilogue region is idle here)
        __syncthreads();
        #pragma unroll 1
        for (int q = 0; q < 4; ++q) {
            const int vt = tid + 256 * q;
            float part = 0.0f;
            #pragma unroll 1
            for (int c = vt; c < K; c += 1024)
                part = __fmaf_rn(xq_h2f(sg[c]), xq_h2f(x[(long long)r * K + c]), part);
            red[vt] = part;
        }
        __syncthreads();
        for (int s2 = 512; s2 > 0; s2 >>= 1) {
            for (int i = tid; i < s2; i += 256) red[i] = __fadd_rn(red[i], red[i + s2]);
            __syncthreads();
        }
        if (tid == 0) sgv[r] = xq_sig(red[0]);
    }
    if (n_items == 0) return;

    int offA, offB, sh7;
    mpk_diet_consts(lane, offA, offB, sh7);
    const bool l0 = gid < M, l1 = (gid + 8) < M, hi_rows = M > 8;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
    int c_item = i_beg, c_ks = 0, c_slot = 0;

    #pragma unroll 1
    for (int it = 0; it < total; ++it) {
        if (a_pend) {                                  // CTA-uniform: a new expert's A
            mpk_wait_a();
            if (fold) { mpk_fold_a(sa, su, M, K); __syncthreads(); }
            a_pend = false;
        }
        // slot `it` landed (<= NST-2 newer groups pending) and is visible CTA-wide; every warp is
        // past slot it-1, so its ring slot may be refilled.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        mpk_prod_step<NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        mpk_consume(ring + c_slot * MPK_STB + warp * MPK_BB, sa, SK, c_ks * MPK_KSTEP, offA, offB, sh7,
                    l0, l1, hi_rows, gid, tig, acc);
        if (++c_slot == NST) c_slot = 0;
        if (++c_ks == NKS) {
            // ---- item epilogue (CTA-uniform) == wp20_moe_gu's, for (expert e, tile) ----
            const int e = c_item / T, tile = c_item - e * T, ix = s_ix[e];
            __half* yt = epi;                          // [M][128]
            wp20_stage_tile(yt, acc, M);
            __syncthreads();                           // every warp is past this item's MMAs: A is dead
            const int nx = c_item + i_step;
            if (nx < i_end && nx / T != e) {           // last-reader refill: the next expert's A
                const int e2 = nx / T;
                mpk_issue_a(sa_u, ain + (fold ? 0ll : (long long)e2 * M * K), M, K, su_u,
                            fold ? suh_all + (long long)s_ix[e2] * K : nullptr);
                a_pend = true;
            }
            const int cta_n = tile * 128;
            __half* ye = ygu + (long long)e * M * N;
            for (int r = warp; r < M; r += 8) {        // own svh (== xq_had_svh_multi, block `tile`)
                const uint2 o = wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
                *(uint2*)(yt + r * 128 + lane * 4) = o;
                *(uint2*)(ye + (long long)r * N + cta_n + lane * 4) = o;
            }
            const int p = tile < P ? tile : tile - P;
            if (wp20_arrive(cnt + e * P + p, 2u, &s_flag)) {
                // second arrival of the (gate p, up p+P) pair: silu*mul + the down input transform
                const int KD = N >> 1;
                const int ptile = tile < P ? tile + P : tile - P;
                const bool own_gate = tile < P;
                for (int r = warp; r < M; r += 8) {
                    const uint2 pw = __ldcg((const uint2*)(ye + (long long)r * N + ptile * 128 + lane * 4));
                    const uint2 ow = *(const uint2*)(yt + r * 128 + lane * 4);
                    float g[4], u[4], sv[4], dv[4], v[4];
                    wp20_unpack4(own_gate ? ow : pw, g);
                    wp20_unpack4(own_gate ? pw : ow, u);
                    wp20_unpack4(__ldg((const uint2*)(suh_d_all + (long long)ix * KD + p * 128 + lane * 4)), sv);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        dv[j] = xq_h2f(xq_f2h(xq_silu(g[j]) * u[j]));   // == xq_moe_gate_mul
                    wp20_suh_had(dv, sv, v, lane);                        // == xq_had_suh_multi (pinned)
                    __half o[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) o[j] = xq_f2h(v[j]);
                    *(uint2*)(xhd + ((long long)e * M + r) * KD + p * 128 + lane * 4) = wp20_pack4(o);
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
            c_ks = 0;
            c_item = nx;
        }
    }
}

template <int NST, int TOPK>
__device__ __forceinline__ void mpk_dn_body(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                            const __half* __restrict__ xhd, __half* __restrict__ yd,
                                            int M, int K, int N, const int* __restrict__ esel,
                                            const int* __restrict__ idxmap, const __half* __restrict__ svh_all,
                                            unsigned* __restrict__ cnt, const __half* __restrict__ ysh,
                                            const int* __restrict__ ids, const float* __restrict__ wts,
                                            const int* __restrict__ slotmap, const float* __restrict__ sgv,
                                            __half* __restrict__ out, int topk, int flags) {
    static_assert(NST >= 3, "ring needs >= 3 slots");
    static_assert(TOPK >= 1 && TOPK <= 16, "mpk_dn_body: TOPK instance out of the planned range");
    extern __shared__ __align__(16) unsigned char mpk_sm[];
    __shared__ unsigned long long s_off[MPK_EMAX];
    __shared__ int s_ix[MPK_EMAX];
    __shared__ int s_slot[MPK_RTMAX];
    __shared__ float s_w[MPK_RTMAX];
    __shared__ int s_flag;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int T = N >> 7, NKS = (K >> 4) / MPK_KSTEP, SK = K + 8;
    unsigned char* ring = mpk_sm;
    __half* sa = (__half*)(mpk_sm + NST * MPK_STB);
    __half* epi = (__half*)((unsigned char*)sa + M * SK * 2);
    const int live = *esel;
    const int nrt = M * TOPK;
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)mpk_smem_bytes(NST, M, K, 0, 0)) __trap();
        if (M < 1 || M > 16 || (N & 127) != 0 || (K & 127) != 0 || (K % (16 * MPK_KSTEP)) != 0) __trap();
        if (live < 0 || live > MPK_EMAX || nrt > MPK_RTMAX) __trap();
        if (topk != TOPK) __trap();                    // launcher bug: never silently mis-combine
    }
    for (int e = tid; e < live && e < MPK_EMAX; e += MPK_THREADS) { s_off[e] = offs[e]; s_ix[e] = idxmap[e]; }
    // The combine tables (== wp20_moe_dn's s_slot = slotmap[ids[i]], s_w = wts[i]); the router
    // wrote them before the gate/up launch — read ONCE per CTA, not per last arrival.
    for (int i = tid; i < nrt && i < MPK_RTMAX; i += MPK_THREADS) { s_slot[i] = slotmap[ids[i]]; s_w[i] = wts[i]; }
    int i_beg, i_step, i_end, n_items;
    mpk_items(live * T, flags, i_beg, i_step, i_end, n_items);
    const int total = n_items * NKS;
    __syncthreads();
    if (n_items == 0) return;

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const unsigned sa_u = (unsigned)__cvta_generic_to_shared(sa);
    const bool c_live = tid < MPK_NCH;
    const int cg = tid / (MPK_KROW / 16), cr = tid - cg * (MPK_KROW / 16);
    const long long rowB = (long long)(N >> 4) * MPK_BB;
    const long long c_src = (long long)cg * rowB + cr * 16;
    const unsigned c_dst = (unsigned)(cg * MPK_KROW + cr * 16);
    const unsigned char* trb = (const unsigned char*)base;
    MpkProd pr;
    pr.item = i_beg; pr.ks = 0; pr.slot = 0; pr.i = 0; pr.src = trb;
    mpk_prod_seek(pr, trb, s_off, T, c_src);
    #pragma unroll 1
    for (int s = 0; s < NST - 1; ++s)
        mpk_prod_step<NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
    mpk_issue_a(sa_u, xhd + (long long)(i_beg / T) * M * K, M, K, 0u, nullptr);
    bool a_pend = true;

    int offA, offB, sh7;
    mpk_diet_consts(lane, offA, offB, sh7);
    const bool l0 = gid < M, l1 = (gid + 8) < M, hi_rows = M > 8;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
    int c_item = i_beg, c_ks = 0, c_slot = 0;

    #pragma unroll 1
    for (int it = 0; it < total; ++it) {
        if (a_pend) {
            mpk_wait_a();
            a_pend = false;
        }
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        mpk_prod_step<NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        mpk_consume(ring + c_slot * MPK_STB + warp * MPK_BB, sa, SK, c_ks * MPK_KSTEP, offA, offB, sh7,
                    l0, l1, hi_rows, gid, tig, acc);
        if (++c_slot == NST) c_slot = 0;
        if (++c_ks == NKS) {
            // ---- item epilogue (CTA-uniform) == wp20_moe_dn's, for (expert e, tile) ----
            const int e = c_item / T, tile = c_item - e * T, ix = s_ix[e];
            __half* yt = epi;                          // [M][128]
            wp20_stage_tile(yt, acc, M);
            __syncthreads();                           // A is dead (every warp past its MMAs)
            const int nx = c_item + i_step;
            if (nx < i_end && nx / T != e) {
                mpk_issue_a(sa_u, xhd + (long long)(nx / T) * M * K, M, K, 0u, nullptr);
                a_pend = true;
            }
            const int cta_n = tile * 128;
            for (int r = warp; r < M; r += 8)          // == xq_had_svh_multi -> yd
                *(uint2*)(yd + ((long long)e * M + r) * N + cta_n + lane * 4) =
                    wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
            if (wp20_arrive(cnt + tile, (unsigned)live, &s_flag)) {
                // Last arrival for this N-tile: == xq_moe_combine on columns [cta_n, cta_n + 128)
                // (wp20_moe_dn's loop verbatim: a = rn(sgv * ysh); j ascending: a = fma.rn(w_j, y_j, a)).
                for (int q = tid; q < M * 32; q += MPK_THREADS) {
                    const int r = q >> 5, i = cta_n + ((q & 31) << 2);
                    const int* sl = s_slot + r * TOPK;
                    const float* sw = s_w + r * TOPK;
                    uint2 raw[TOPK];
                    #pragma unroll
                    for (int j = 0; j < TOPK; ++j) {       // unconditional (a -1 slot reads slot 0, unused)
                        const int s = sl[j] < 0 ? 0 : sl[j];
                        raw[j] = __ldcg((const uint2*)(yd + ((long long)s * M + r) * N + i));
                    }
                    float ys[4], a[4];
                    wp20_unpack4(__ldg((const uint2*)(ysh + (long long)r * N + i)), ys);
                    const float gg = sgv[r];
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) a[k4] = __fmul_rn(gg, ys[k4]);
                    #pragma unroll
                    for (int j = 0; j < TOPK; ++j) {
                        if (sl[j] >= 0) {
                            float y[4];
                            wp20_unpack4(raw[j], y);
                            const float w = sw[j];
                            #pragma unroll
                            for (int k4 = 0; k4 < 4; ++k4) a[k4] = __fmaf_rn(w, y[k4], a[k4]);
                        }
                    }
                    __half o[4];
                    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) o[k4] = xq_f2h(a[k4]);
                    *(uint2*)(out + (long long)r * N + i) = wp20_pack4(o);
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
            c_ks = 0;
            c_item = nx;
        }
    }
}

// Entries: xq_moe_gu_pk_s{NST}, xq_moe_dn_pk_k{TOPK}_s{NST}. The host (exl3_forward::mpk_plan)
// picks NST and G from the occupancy API at the real dynamic smem; flags = MPK_IL | MPK_FOLD.
#define MPK_GU_ENTRY(NS)                                                                             \
extern "C" __global__ void __launch_bounds__(MPK_THREADS, 2)                                        \
xq_moe_gu_pk_s##NS(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,            \
                   const __half* __restrict__ ain, const __half* __restrict__ suh_all,              \
                   __half* __restrict__ ygu, int M, int K, int N,                                   \
                   const int* __restrict__ esel, const int* __restrict__ idxmap,                    \
                   const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,        \
                   __half* __restrict__ xhd, unsigned* __restrict__ cnt,                            \
                   const __half* __restrict__ x, const __half* __restrict__ sg,                     \
                   float* __restrict__ sgv, int flags) {                                            \
    XQ_PDL_ENTRY();                                                                                  \
    mpk_gu_body<NS>(base, offs, ain, suh_all, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all, xhd,  \
                    cnt, x, sg, sgv, flags);                                                         \
}
MPK_GU_ENTRY(4)
MPK_GU_ENTRY(8)

#define MPK_DN_ENTRY(TK, NS)                                                                         \
extern "C" __global__ void __launch_bounds__(MPK_THREADS, 2)                                        \
xq_moe_dn_pk_k##TK##_s##NS(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,    \
                           const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K,   \
                           int N, const int* __restrict__ esel, const int* __restrict__ idxmap,     \
                           const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,          \
                           const __half* __restrict__ ysh, const int* __restrict__ ids,             \
                           const float* __restrict__ wts, const int* __restrict__ slotmap,          \
                           const float* __restrict__ sgv, __half* __restrict__ out, int topk,       \
                           int flags) {                                                             \
    XQ_PDL_ENTRY();                                                                                  \
    mpk_dn_body<NS, TK>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids, wts,    \
                        slotmap, sgv, out, topk, flags);                                             \
}
MPK_DN_ENTRY(8, 4)
MPK_DN_ENTRY(8, 8)
MPK_DN_ENTRY(10, 4)
MPK_DN_ENTRY(10, 8)

// ---------------------------------------------------------------------------
// S-A3-h item 3, rung 1: COOPERATIVE MoE expert-stream GEMM — a schedule port
// of the reference implementation's exl3_moe_coop_a/b<3,2,true> (R-M3 §4), MATH UNCHANGED.
//
// Old schedule (xq_gemm_grouped_xh)            New schedule (this kernel)
// ─ 256 threads, m16 x N128 CTA tile           ─ 512 threads, one block per
// ─ grid (expert x N/128 tiles)                  (expert-run, 32-COL GROUP) —
// ─ 8 warps compute 8 x 16-col stripes           the reference implementation's A/B work map (A =
// ─ trellis words read as scalar 4B              gate/up call site, B = down;
//   global loads inside the decode; A tile       N granularity 32 cols so tail
//   staged by direct LDG->SMEM                   groups never idle a 128 tile)
//                                                 ─ warps 0/1 compute one 16-col
//                                                   block each; the other 14
//                                                   warps are the cp.async
//                                                   staging engines
//                                                 ─ trellis words + A tiles
//                                                   staged by cp.async.cg 16B
//                                                   (the LDGSTS.E.BYPASS.128
//                                                   class) into an 8-slot SMEM
//                                                   ring, 6 committed pipeline
//                                                   groups in flight
//                                                   (commit_group/wait_group =
//                                                   the PTX form of LDGDEPBAR)
//                                                 ─ 256 threads/block x 3
//                                                   blocks/SM
//                                                   (__launch_bounds__(256,3)):
//                                                   the recipe's 512x2
//                                                   needs <=64 regs, but the
//                                                   bit-identical mma+unpack
//                                                   math floors at ~81
//                                                   (xq_gemm_grouped_xh alone
//                                                   is 66) — 64 forces stack
//                                                   spills (AGENTS §4 rule);
//                                                   256x3 keeps ZERO spill
//                                                   with the same work map,
//                                                   pipeline depth and 3
//                                                   concurrent pipelines/SM
//                                                   (50.7 KB in flight/SM vs
//                                                   the recipe's 45).
//
// BIT-IDENTITY CONTRACT (why outputs are bit-equal to xq_gemm_grouped_xh at
// every M — AGENTS 2.4, batch invariance):
//   1. The decode is a pure function of the ring words: exl3_dq_from(smem_copy)
//      == exl3_dq_from(global) because cp.async copies the exact 24-word block
//      (value-transparent staging) -> identical f16 B-fragments.
//   2. A tiles: rows < M are byte copies of xh rows; rows >= M are ZERO via
//      cp.async src-size=0 zfill (0x0000 = +0.0 f16 — the same constant the
//      old path staged with __ushort_as_half(0)). K % 16 == 0 at every call
//      site, so the old `kb*16+col < K` guard never fires in either path.
//   3. Per output: fp32 accs zero-init, one mma.sync.aligned.m16n8k16.
//      row.col.f32.f16.f16.f32 pair (h=0,1) per k16 step, kb STRICTLY
//      ascending (slot i, g ascending), on ONE warp — the reduction is never
//      split across warps/slots/stages. Identical instruction sequence and
//      operand values => identical result bits.
//   4. Epilogue: __floats2half2_rn and the same (row,col)->(acc,col) mapping.
//
// Preconditions (identical to the old path plus N % 32 == 0): M <= 16,
// K % 16 == 0, bits in {3,4,5}. Each active expert's trellis stream is read
// EXACTLY once (bytes are the schedule); the A tile is re-staged per 32-col
// group but is L2-resident (emax*m*K*2 B total — ~3.3 MB worst case).
// ---------------------------------------------------------------------------

#define MOE_COOP_THREADS 256
#define MOE_COOP_STAGES  8   // SMEM ring slots (STAGES > PIPE + 1: the reissue
                            // of a buffer targets a slot computed >= 2 iters ago)
#define MOE_COOP_PIPE    6   // committed pipeline groups kept in flight
#define MOE_COOP_GST     4   // k16 steps staged per ring slot

// cp.async.cg 16B (bypass-L1 LDGSTS class). src_size 0 => full 16B zero fill,
// no global read (row >= M predication; same +0.0 halves the old path staged).
// Explicit free functions (NOT [&]-capture lambdas — capture closures are a
// classic nvcc local-memory trigger; the hot loop must have ZERO stack).
__device__ __forceinline__ void moe_coop_cp16(void* dst, const void* src,
                                              int src_size) {
    const unsigned dst_u = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst_u), "l"(src), "r"(src_size));
}

// Issue slot `slot` (its GST k-steps) as ONE pipeline group; slots past the
// schedule commit an EMPTY group so the wait_group accounting in the caller
// stays aligned (the standard multistage empty-commit idiom).
template <int BITS>
__device__ __forceinline__ void moe_coop_issue(int slot, int tid, int M, int K,
                                               int KB, int NSLOT, int nb, int nbc0,
                                               const uint16_t* __restrict__ tr,
                                               const __half* __restrict__ xe,
                                               uint32_t* s_tr_raw, __half* s_a_raw) {
    // S-A3-h cycle-2 fix (memcheck: Invalid __shared__ write at xq_moe_coop_a+0xaa0,
    // 0x9400 OOB): these views must match xq_moe_coop_body's EXACTLY — pointer to the
    // per-stage 3-D slice. The extra [MOE_COOP_STAGES] dim made s_tr[i]/s_a[i] advance
    // a whole-buffer stride, so every stage>0 write landed OOB (ILLEGAL_ADDRESS) while
    // the reader read the right slots and found garbage (the bit-mismatch twin symptom).
    typedef uint32_t (*TrView)[MOE_COOP_GST][2][8 * BITS];
    typedef __half   (*SaView)[MOE_COOP_GST][16][16];
    TrView s_tr = (TrView)s_tr_raw;
    SaView s_a  = (SaView)s_a_raw;
    constexpr int TRC = 2 * (2 * BITS);
    constexpr int CHC = MOE_COOP_GST * (TRC + 32);
    if (slot < NSLOT && tid < CHC) {
        const int kb0 = slot * MOE_COOP_GST;
        if (tid < MOE_COOP_GST * TRC) {
            // trellis: 16B chunk c of 16x16 block b, k-step g
            const int g   = tid / TRC;
            const int rem = tid - g * TRC;
            const int b   = rem / (2 * BITS);
            const int c   = rem - b * (2 * BITS);
            const int kb  = kb0 + g;
            if (kb < KB) {
                const uint16_t* src =
                    tr + ((size_t)kb * nb + nbc0 + b) * (16 * BITS) + c * 8;
                moe_coop_cp16(&s_tr[slot % MOE_COOP_STAGES][g][b][c * 4], src, 16);
            }
        } else {
            // A tile: 16B half-row of the k-step's [16 x 16] f16 tile
            const int t2  = tid - MOE_COOP_GST * TRC;
            const int g   = t2 / 32;
            const int rem = t2 - g * 32;
            const int row = rem >> 1;
            const int half = rem & 1;
            const int kb  = kb0 + g;
            if (kb < KB) {
                const int srow = row < M ? row : 0;   // dead rows: zfill
                const __half* src = xe + (size_t)srow * K + kb * 16 + half * 8;
                moe_coop_cp16(&s_a[slot % MOE_COOP_STAGES][g][row][half * 8], src,
                              row < M ? 16 : 0);
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int BITS>
__device__ __forceinline__ void xq_moe_coop_body(const uint16_t* __restrict__ base,
                                                 const uint64_t* __restrict__ offs,
                                                 const __half* __restrict__ xh,
                                                 __half* __restrict__ y,
                                                 int M, int K, int N,
                                                 const int* __restrict__ esel,
                                                 uint32_t* s_tr_raw,
                                                 __half* s_a_raw) {
    // SMEM ring (owned by the WRAPPER — one max-size copy for all BITS
    // instantiations; per-instantiation __shared__ inside the body would
    // allocate 3 copies, 73,728 B > the 48 KiB static cap).
    typedef uint32_t (*TrView)[MOE_COOP_GST][2][8 * BITS];
    typedef __half   (*SaView)[MOE_COOP_GST][16][16];
    TrView s_tr = (TrView)s_tr_raw;
    SaView s_a  = (SaView)s_a_raw;
    // Work map: one block per (expert, 32-col group); projection is the call
    // site (A = gate/up, B = down — the reference implementation's A/B split).
    const int groups = N >> 5;
    const int expert = blockIdx.x / groups;
    const int group  = blockIdx.x - expert * groups;
    if (expert >= *esel) return;                  // padded-cap skip (S-A3-f I2)

    const uint16_t* tr = base + (size_t)offs[expert];
    const __half*  xe  = xh + (long long)expert * M * K;   // packed [E][M][K]
    __half*        ye  = y + (long long)expert * M * N;    // packed [E][M][N]

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb   = N >> 4;                      // 16-col trellis blocks/row
    const int KB   = K >> 4;                      // k16 steps
    const int nbc0 = group * 2;                   // warp 0's block; warp 1: +1

    // ---- hoisted per-lane slot constants (exl3_gemm_body's values, PACKED:
    // 24 int registers -> 8, to fit the 64-reg 2-blocks/SM budget with zero
    // spill; unpacking adds ~3 IMAD/SHF per dq_from on the compute warps).
    // Layout: pk = (i0m << 12) | (i1m << 6) | sf — i0m,i1m < 8*BITS <= 40
    // (6 bits), sf < 32 (5 bits).
    unsigned s_pk[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                          + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_pk[h * 4 + s] = ((unsigned)(i0 % (BITS * 8)) << 12)
                              | ((unsigned)(i1 % (BITS * 8)) << 6)
                              | (unsigned)(((i1 + 1) << 5) - b1);
        }
    }

    // ---- SMEM ring: per slot, GST k-steps of (2 trellis blocks + A tile) ----
    // Wrapper-owned max-size copy: 8 * 4 * (2*40*4 + 16*16*2) = 26624 B at
    // BITS=5 (viewed smaller at BITS=3/4); <= 48 KiB static cap, 2 blocks/SM
    // = 53248 B of the 100 KiB/SM budget. Views s_tr / s_a above.

    const int NSLOT = (KB + MOE_COOP_GST - 1) / MOE_COOP_GST;

    // Prologue: exactly PIPE committed groups (empty past the schedule tail).
    // Staging lives in moe_coop_issue (free fn — no capture closures).
    #pragma unroll
    for (int s = 0; s < MOE_COOP_PIPE; ++s)
        moe_coop_issue<BITS>(s, tid, M, K, KB, NSLOT, nb, nbc0, tr, xe,
                             s_tr_raw, s_a_raw);

    // ---- main pipeline ----
    // Accumulators: two n8 tiles, fp32, zero-init (identical to the old body).
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    for (int i = 0; i < NSLOT; ++i) {
        moe_coop_issue<BITS>(i + MOE_COOP_PIPE, tid, M, K, KB, NSLOT, nb, nbc0,
                             tr, xe, s_tr_raw, s_a_raw);
        // <= PIPE-1 pending => group of slot i complete (FIFO groups; at this
        // point PIPE+i+1 groups were committed, newest PIPE-1 may pend).
        asm volatile("cp.async.wait_group %0;\n" :: "n"(MOE_COOP_PIPE - 1) : "memory");
        __syncthreads();                            // cross-warp visibility

        if (warp < 2) {
            // Warps 0/1 compute their own 16-col block: decode from the SMEM
            // ring copy, mma pair per k16 step, kb ascending. The per-output
            // K-ascending chain lives ENTIRELY on this warp (never split).
            uint32_t (*strg)[2][8 * BITS] = s_tr[i % MOE_COOP_STAGES];
            __half   (*sag)[16][16]       = s_a [i % MOE_COOP_STAGES];
            const int gid = lane >> 2, tig = lane & 3;
            #pragma unroll
            for (int g = 0; g < MOE_COOP_GST; ++g) {
                const int kb = i * MOE_COOP_GST + g;
                if (kb >= KB) break;
                const uint32_t* ring = strg[g][warp];

                // ---- decode this warp's 16x16 block ONCE into two B frags --
                // (constants UNPACKED from s_pk — the integer VALUES passed
                // to exl3_dq_from are identical to exl3_gemm_body's, so the
                // decoded f16 bits are identical.)
                uint32_t bfrag[2][2];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const unsigned pk0 = s_pk[h*4+0], pk1 = s_pk[h*4+1];
                    const uint16_t r0 = exl3_dq_from(ring, (int)(pk0 >> 12), (int)((pk0 >> 6) & 63u), (int)(pk0 & 63u));
                    const uint16_t r1 = exl3_dq_from(ring, (int)(pk1 >> 12), (int)((pk1 >> 6) & 63u), (int)(pk1 & 63u));
                    bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
                    const unsigned pk2 = s_pk[h*4+2], pk3 = s_pk[h*4+3];
                    const uint16_t r2 = exl3_dq_from(ring, (int)(pk2 >> 12), (int)((pk2 >> 6) & 63u), (int)(pk2 & 63u));
                    const uint16_t r3 = exl3_dq_from(ring, (int)(pk3 >> 12), (int)((pk3 >> 6) & 63u), (int)(pk3 & 63u));
                    bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
                }

                // ---- A fragments (m16k16, manual from SMEM — same expr) ----
                const uint32_t a0 = *(const uint32_t*)&sag[g][gid][2 * tig];
                const uint32_t a1 = *(const uint32_t*)&sag[g][gid + 8][2 * tig];
                const uint32_t a2 = *(const uint32_t*)&sag[g][gid][2 * tig + 8];
                const uint32_t a3 = *(const uint32_t*)&sag[g][gid + 8][2 * tig + 8];

                // ---- 2x m16n8k16 (same asm, same operand order) ----
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                }
            }
        }
        __syncthreads();    // all warps done with slot i before its reissue
    }

    // ---- epilogue: store y_raw (fp16, RN), skip padded rows (identical
    // rounding + (row,col)->(acc,col) mapping to exl3_gemm_body) ----
    if (warp < 2) {
        const int gid = lane >> 2, tig = lane & 3;
        const int row0 = gid, row1 = gid + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = group * 32 + 16 * warp + 8 * h + 2 * tig;
            if (row0 < M) {
                __half2 o = __floats2half2_rn(acc[h][0], acc[h][1]);
                *(__half2*)&ye[(size_t)row0 * N + col] = o;
            }
            if (row1 < M) {
                __half2 o = __floats2half2_rn(acc[h][2], acc[h][3]);
                *(__half2*)&ye[(size_t)row1 * N + col] = o;
            }
        }
    }
}

// The two expert-stream entry points (A = gate/up, B = down projection). Same
// 9-arg signature/grid convention as xq_gemm_grouped_xh so the launch sites
// differ only in kernel name and grid = emax * (N/32). The SMEM ring is owned
// HERE (one max-size copy for all BITS instantiations) and passed to the body
// as raw pointers.
extern "C" __global__ void __launch_bounds__(MOE_COOP_THREADS, 3)
xq_moe_coop_a(const uint16_t* __restrict__ base,
              const uint64_t* __restrict__ offs,
              const __half* __restrict__ xh,
              __half* __restrict__ y,
              int M, int K, int N, int bits,
              const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    __shared__ uint32_t sh_tr[MOE_COOP_STAGES][MOE_COOP_GST][2][8 * 5];
    __shared__ __half   sh_a [MOE_COOP_STAGES][MOE_COOP_GST][16][16];
    switch (bits) {
        case 3: xq_moe_coop_body<3>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 4: xq_moe_coop_body<4>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 5: xq_moe_coop_body<5>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        default: break;
    }
}

extern "C" __global__ void __launch_bounds__(MOE_COOP_THREADS, 3)
xq_moe_coop_b(const uint16_t* __restrict__ base,
              const uint64_t* __restrict__ offs,
              const __half* __restrict__ xh,
              __half* __restrict__ y,
              int M, int K, int N, int bits,
              const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    __shared__ uint32_t sh_tr[MOE_COOP_STAGES][MOE_COOP_GST][2][8 * 5];
    __shared__ __half   sh_a [MOE_COOP_STAGES][MOE_COOP_GST][16][16];
    switch (bits) {
        case 3: xq_moe_coop_body<3>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 4: xq_moe_coop_body<4>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 5: xq_moe_coop_body<5>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        default: break;
    }
}

// ---- full-attention decode (dense, no QSA): grid (M*nh), block hd.
// qkv [M][13312] fused proj output: q|gate per head at 0 (12288), k at 12288, v at 12800.
// q/k norm shared weight [hd] with (1+w), partial rope (rdim, rotate_half),
// f32 KV cache [slot][nkv][max_pos][hd]. Two-pass softmax over t ascending.
// 12-arg cap (cudarc): cossin is [m][2*rdim] (cos|sin), slotpos is [m][2] (slot,pos),
// nh_packed = nh | nkv<<16, hd_packed = hd | rdim<<16.
// ---- S-A3-f-b: shared K rmsnorm+rope for the KV cache write. The sweep
// (xq_attn_kv_prefill) and decode (xq_attn_decode) must produce BIT-IDENTICAL
// cache words: separately compiled near-identical code differed in the last
// FP bit (K rel ~2e-8) which amplified to ~1e-4 attention output and O(1)
// logits over 44 layers. Same inline body in both kernels = same instruction
// sequence.
__device__ __forceinline__ void xq_k_norm_rope(float* kbuf, float* red, const __half* ksrc,
                                               const __half* knw, const float* c, const float* s,
                                               int tid, int hd, int rdim, float eps) {
    float kv = xq_h2f(ksrc[tid]);
    red[tid] = kv * kv;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    kv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(knw[tid]));
    kbuf[tid] = kv;
    __syncthreads();
    if (tid < rdim / 2) {
        float x1 = kbuf[tid], x2 = kbuf[tid + rdim / 2];
        kbuf[tid] = x1 * c[tid] - x2 * s[tid];
        kbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
}

// ---- KV cache formats (PLAN/KV_CACHE_FORMATS.md K1/K2/K3). Rows are (slot, K|V, kv-head, pos);
// row bytes: f32 4*hd, f16 2*hd, fp8 hd + 16 (hd e4m3 values, then the row's f32 scale; 16-B
// aligned), q8 hd + 16 at hd=256 (hd int8 codes, then hd/32 f16 group scales, padded to 16 B).
// The format rides in the top 4 bits of the kernels' max_pos argument (max_pos < 2^28).
// For f32 every read/write below touches the same bytes as the pre-K1 float* indexing.
#define XQ_KV_F32 0
#define XQ_KV_F16 1
#define XQ_KV_FP8 2
// WP25/K3 q8 = the reference implementation's "-cq 8" cache (exllamav3 cache/q_cache_kernels.cuh quant_block_x4 +
// triton_paged._qc_load_kt/_qc_load_v, fork 523ecd3), per 32-dim group of a K/V row:
//   v = H32(x) * (1/sqrt 32)          unnormalized butterfly, strides 1,2,4,8,16 (their H4 x H8 order)
//   s = max|v| + 1e-10                stored as ONE f16 per group (__float2half_rn)
//   q = clamp(floor(fma(v * (1/s), 128, 128)), 0, 255)          linear midpoint grid, 8-bit code
//   read: (q - 127.5) * (f16(s) / 128)                            stays in the ROTATED basis
// The rotation is folded out of attention exactly as the reference implementation does it (orthonormal and block-
// diagonal per head): q is rotated once where it is staged (q.k == H q . H k), the output is
// rotated back once (xq_h32r) before the gate, so every in-loop read is unpack x scale.
#define XQ_KV_Q8 3
// Probe-only positive control (WP25 rung 1: "a deliberately broken 4-bit arm, which must fail"):
// the q8 layout and readers with the reference implementation's 4-bit midpoint grid (code q4*16+8). Never served.
#define XQ_KV_Q4X 4
#define XQ_R32 0.17677669529663688110f   // 1/sqrt(32), the reference implementation's r32
struct XqKv { char* k; char* v; long long rb; int fmt; int hd; };
__device__ __forceinline__ int xq_kv_fmt(int mpf) { return (int)(((unsigned)mpf) >> 28); }
__device__ __forceinline__ long long xq_kv_rowb(int fmt, int hd) {
    return fmt == XQ_KV_F32 ? 4LL * hd : (fmt == XQ_KV_F16 ? 2LL * hd : (fmt == XQ_KV_FP8
        ? (long long)hd + 16 : (long long)hd + (long long)(((hd >> 4) + 15) & ~15)));
}
__device__ __forceinline__ XqKv xq_kv_make(const void* kcvc, int mpf, int hd, int nkv, int slot, int kvh) {
    XqKv r;
    r.fmt = xq_kv_fmt(mpf);
    const long long mp = (long long)(mpf & 0x0FFFFFFF);
    r.hd = hd;
    r.rb = xq_kv_rowb(r.fmt, hd);
    char* b = (char*)kcvc;
    r.k = b + ((((long long)slot * 2 + 0) * nkv + kvh) * mp) * r.rb;
    r.v = b + ((((long long)slot * 2 + 1) * nkv + kvh) * mp) * r.rb;
    return r;
}
// q8 H32 butterfly across a warp: lane j holds element j of one 32-group. Stage o pairs lanes
// j, j^o: low lane a+b, high lane a-b (= the reference implementation's sign-flip-and-add, bitwise). Unnormalized.
__device__ __forceinline__ float xq_h32(float v, int lane) {
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float p = __shfl_xor_sync(0xffffffffu, v, o);
        v = (lane & o) ? __fsub_rn(p, v) : __fadd_rn(v, p);
    }
    return v;
}
// Orthonormal H32/sqrt(32) (involutory): q into the cache basis, attention output back out.
__device__ __forceinline__ float xq_h32r(float v, int lane) { return __fmul_rn(xq_h32(v, lane), XQ_R32); }
// q8 code byte k of a word -> (code - 127.5) * sm, rounded ONCE (== the reference implementation's f32 expression):
// the byte is spliced under 2^23 (PRMT, no I2F), t = code - 128 exactly, fma(t, sm, sm/2).
__device__ __forceinline__ float xq_q8v(unsigned w, int k, float sm, float hs) {
    const float f = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7540 | k));
    return __fmaf_rn(__fsub_rn(f, 8388736.0f), sm, hs);
}
__device__ __forceinline__ float xq_q8_sm(const char* row, int hd, int g) {
    return __fmul_rn(__half2float(reinterpret_cast<const __half*>(row + hd)[g]), 0.0078125f);
}
__device__ __forceinline__ float xq_kv_ld(const char* row, int fmt, int hd, int d) {
    if (fmt == XQ_KV_F32) return reinterpret_cast<const float*>(row)[d];
    if (fmt == XQ_KV_F16) return __half2float(reinterpret_cast<const __half*>(row)[d]);
    if (fmt >= XQ_KV_Q8) {
        const float sm = xq_q8_sm(row, hd, d >> 5);
        return xq_q8v((unsigned)reinterpret_cast<const unsigned char*>(row)[d], 0, sm, 0.5f * sm);
    }
    return float(reinterpret_cast<const __nv_fp8_e4m3*>(row)[d]) * *reinterpret_cast<const float*>(row + hd);
}
__device__ __forceinline__ float xq_kv_k(const XqKv& kc, long long t, int d) { return xq_kv_ld(kc.k + t * kc.rb, kc.fmt, kc.hd, d); }
__device__ __forceinline__ float xq_kv_v(const XqKv& kc, long long t, int d) { return xq_kv_ld(kc.v + t * kc.rb, kc.fmt, kc.hd, d); }
// dims d0..d0+7 of a row (d0 % 8 == 0): f32 = the two float4 loads of the pre-K1 kernels.
// q8: one 8-byte load + the group's f16 scale (4 lanes share it) — same values as xq_kv_ld.
__device__ __forceinline__ void xq_kv_ld8(float* o, const char* row, int fmt, int hd, int d0) {
    if (fmt == XQ_KV_F32) {
        *reinterpret_cast<float4*>(o) = reinterpret_cast<const float4*>(row + 4 * d0)[0];
        *reinterpret_cast<float4*>(o + 4) = reinterpret_cast<const float4*>(row + 4 * d0)[1];
    } else if (fmt == XQ_KV_F16) {
        const uint4 u = *reinterpret_cast<const uint4*>(row + 2 * d0);
        const __half* h = reinterpret_cast<const __half*>(&u);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __half2float(h[i]);
    } else if (fmt >= XQ_KV_Q8) {
        const uint2 u = *reinterpret_cast<const uint2*>(row + d0);
        const float sm = xq_q8_sm(row, hd, d0 >> 5);
        const float hs = 0.5f * sm;
        #pragma unroll
        for (int i = 0; i < 4; i++) o[i] = xq_q8v(u.x, i, sm, hs);
        #pragma unroll
        for (int i = 0; i < 4; i++) o[4 + i] = xq_q8v(u.y, i, sm, hs);
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(row + d0);
        const __nv_fp8_e4m3* e = reinterpret_cast<const __nv_fp8_e4m3*>(&u);
        const float sc = *reinterpret_cast<const float*>(row + hd);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = float(e[i]) * sc;
    }
}
// q8 row write, warp-local (one warp = one 32-group; no block barrier). Every op is an explicit
// _rn intrinsic so the host reference (src/exl3_kvnll.rs q8_quant_row) reproduces the bytes.
__device__ __forceinline__ void xq_kv_put_q8(char* row, int fmt, int hd, int tid, float x) {
    const int lane = tid & 31;
    const float v = xq_h32r(x, lane);
    float s = fabsf(v);
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
    s = __fadd_rn(s, 1e-10f);             // == max over lanes of (|v| + 1e-10): fl(.) is monotone
    const float vs = __fmul_rn(v, __fdiv_rn(1.0f, s));
    int q;
    if (fmt == XQ_KV_Q8) {
        q = max(min(__float2int_rd(__fmaf_rn(vs, 128.0f, 128.0f)), 255), 0);
    } else {                              // XQ_KV_Q4X: 4-bit midpoint grid, centred in the q8 code
        q = max(min(__float2int_rd(__fmaf_rn(vs, 8.0f, 8.0f)), 15), 0) * 16 + 8;
    }
    reinterpret_cast<unsigned char*>(row)[tid] = (unsigned char)q;
    if (lane == 0) reinterpret_cast<__half*>(row + hd)[tid >> 5] = __float2half_rn(s);
}
// Block-wide row write: blockDim == hd threads, EVERY thread calls once, thread tid owns dim tid.
// fp8: e4m3(x / s) with the row scale s = absmax / 448 (satfinite), stored after the values.
__device__ __forceinline__ void xq_kv_put(char* row, int fmt, int hd, int tid, float x) {
    if (fmt == XQ_KV_F32) { reinterpret_cast<float*>(row)[tid] = x; return; }
    if (fmt == XQ_KV_F16) { reinterpret_cast<__half*>(row)[tid] = __float2half_rn(x); return; }
    if (fmt >= XQ_KV_Q8) { xq_kv_put_q8(row, fmt, hd, tid, x); return; }
    __shared__ float xq_kv_amax[32];
    float a = fabsf(x);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((tid & 31) == 0) xq_kv_amax[tid >> 5] = a;
    __syncthreads();
    float m = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) m = fmaxf(m, xq_kv_amax[w]);
    const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
    reinterpret_cast<__nv_fp8_e4m3*>(row)[tid] = __nv_fp8_e4m3(x / sc);
    if (tid == 0) *reinterpret_cast<float*>(row + hd) = sc;
    __syncthreads();   // xq_kv_amax is reused by the next row
}
// q8: the attention q/output basis change (identity for f32/f16/fp8 — no op is emitted there).
__device__ __forceinline__ float xq_kv_rot(float v, int fmt, int lane) {
    return fmt >= XQ_KV_Q8 ? xq_h32r(v, lane) : v;
}
// WP25 --probe-exl3-kvq: rows through the SERVED helpers, no model. Grid n rows, block hd.
// rows = n cache rows (xq_kv_rowb bytes each); deq = xq_kv_ld (cache basis), deq8 = xq_kv_ld8
// (threads 0..hd/8-1), rec = xq_kv_rot(deq) (model basis). The host checks bytes/scales/deq
// against its reference bitwise and deq8 == deq.
extern "C" __global__ void xq_kv_selftest(char* __restrict__ rows, float* __restrict__ deq,
                                          float* __restrict__ deq8, float* __restrict__ rec,
                                          const float* __restrict__ x, int hd, int fmt) {
    XQ_PDL_ENTRY();
    const int i = blockIdx.x, tid = threadIdx.x;
    char* row = rows + (long long)i * xq_kv_rowb(fmt, hd);
    xq_kv_put(row, fmt, hd, tid, x[(long long)i * hd + tid]);
    __syncthreads();
    const float d = xq_kv_ld(row, fmt, hd, tid);
    deq[(long long)i * hd + tid] = d;
    rec[(long long)i * hd + tid] = xq_kv_rot(d, fmt, tid & 31);
    if (tid < hd / 8) {
        float o[8];
        xq_kv_ld8(o, row, fmt, hd, tid * 8);
        #pragma unroll
        for (int k = 0; k < 8; k++) deq8[(long long)i * hd + tid * 8 + k] = o[k];
    }
}
template <int E>
__device__ __forceinline__ float xq_dot_tree_kv(const float* qb, const XqKv& kc, long long t, int lane) {
    const char* row = kc.k + t * kc.rb;
    float v[E];
    #pragma unroll
    for (int j = 0; j < E; j++) v[j] = __fmul_rn(qb[lane + 32 * j], xq_kv_ld(row, kc.fmt, kc.hd, lane + 32 * j));
    #pragma unroll
    for (int h = E / 2; h >= 1; h >>= 1) {
        #pragma unroll
        for (int j = 0; j < h; j++) v[j] = __fadd_rn(v[j], v[j + h]);
    }
    float x = v[0];
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, o));
    return x;  // valid in lane 0
}

// ---- S-A3-f-b: shared two-pass softmax + weighted-V accumulation for the
// attention output, used by BOTH xq_attn_decode and xq_attn_prefill_q so the
// sweep rows are bit-identical to the sequential per-token rows (those rows
// feed later layers' GDN state through the resid).
__device__ __forceinline__ void xq_attn_softmax(float* red, float qv, const XqKv& kc,
                                                int p, int tid, int hd,
                                                float& acc, float& denom) {
    const float scale = rsqrtf((float)hd);
    float mx = -INFINITY;
    for (int t = 0; t <= p; t++) {
        red[tid] = qv * xq_kv_k(kc, t, tid);
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        mx = fmaxf(mx, red[0] * scale);
        __syncthreads();  // S-A3-e race fix (kept identical)
    }
    denom = 0.0f; acc = 0.0f;
    for (int t = 0; t <= p; t++) {
        red[tid] = qv * xq_kv_k(kc, t, tid);
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        float w = __expf(red[0] * scale - mx);
        denom += w;
        acc += w * xq_kv_v(kc, t, tid);
        __syncthreads();  // S-A3-e race fix (kept identical)
    }
}

// ---- S-A3-m attention v2: the SAME per-position dot, one warp per position.
// xq_attn_softmax's halving tree pairs red[i] += red[i+s2] for s2 = hd/2 .. 1.
// With lane owning dims lane+32j (j < E = hd/32), levels s2 >= 32 pair v[j] with
// v[j+s2/32] in-register and levels 16..1 pair lanes (shfl_down) — the identical
// adds on the identical operands, so lane 0 holds red[0] bit-for-bit. __fmul_rn /
// __fadd_rn forbid FMA contraction (the smem tree materializes each product).
template <int E>
__device__ __forceinline__ float xq_dot_tree(const float* qb, const float* kt, int lane) {
    float v[E];
    #pragma unroll
    for (int j = 0; j < E; j++) v[j] = __fmul_rn(qb[lane + 32 * j], kt[lane + 32 * j]);
    #pragma unroll
    for (int h = E / 2; h >= 1; h >>= 1) {
        #pragma unroll
        for (int j = 0; j < h; j++) v[j] = __fadd_rn(v[j], v[j + h]);
    }
    float x = v[0];
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, o));
    return x;  // valid in lane 0
}

#define XQ_SCAN_T 256
// Drop-in for xq_attn_softmax (same acc/denom bits): pass 1 = warps over t,
// fmaxf-combined (exact, order-free); pass 2 = per tile the warps compute the
// weights into smem, then every thread accumulates its dim in ASCENDING t with
// xq_attn_softmax's expressions (__expf(raw*scale - mx); acc += w*v). No
// per-position block syncs: the old cost was ~2·(p+1)·log2(hd) __syncthreads.
// qb = roped q in smem (all hd dims); wbuf >= XQ_SCAN_T floats, wm >= 32 floats.
template <int E>
__device__ __forceinline__ void xq_attn_scan(const float* qb, const XqKv& kc,
                                             int p, int tid, int hd,
                                             float* wbuf, float* wm, float& acc, float& denom) {
    const float scale = rsqrtf((float)hd);
    const int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    float mxw = -INFINITY;
    for (int t = warp; t <= p; t += nw) {
        const float raw = xq_dot_tree_kv<E>(qb, kc, t, lane);
        mxw = fmaxf(mxw, raw * scale);
    }
    if (lane == 0) wm[warp] = mxw;
    __syncthreads();
    float mx = -INFINITY;
    for (int w = 0; w < nw; w++) mx = fmaxf(mx, wm[w]);
    denom = 0.0f; acc = 0.0f;
    for (int t0 = 0; t0 <= p; t0 += XQ_SCAN_T) {
        const int tn = min(XQ_SCAN_T, p + 1 - t0);
        for (int i = warp; i < tn; i += nw) {
            const float raw = xq_dot_tree_kv<E>(qb, kc, t0 + i, lane);
            if (lane == 0) wbuf[i] = __expf(raw * scale - mx);
        }
        __syncthreads();
        for (int i = 0; i < tn; i++) {
            const float w = wbuf[i];
            denom += w;
            acc += w * xq_kv_v(kc, t0 + i, tid);
        }
        __syncthreads();
    }
}

// hd -> E dispatch; returns false for head dims the warp tree does not cover
// (caller keeps xq_attn_softmax).
__device__ __forceinline__ bool xq_attn_scan_hd(const float* qb, const XqKv& kc,
                                                int p, int tid, int hd,
                                                float* wbuf, float* wm, float& acc, float& denom) {
    switch (hd) {
        case 128: xq_attn_scan<4>(qb, kc, p, tid, hd, wbuf, wm, acc, denom); return true;
        case 256: xq_attn_scan<8>(qb, kc, p, tid, hd, wbuf, wm, acc, denom); return true;
        case 512: xq_attn_scan<16>(qb, kc, p, tid, hd, wbuf, wm, acc, denom); return true;
        default: return false;
    }
}

extern "C" __global__ void xq_attn_decode(__half* __restrict__ attn,
                                          const __half* __restrict__ qg,
                                          const __half* __restrict__ kp,
                                          const __half* __restrict__ vp,
                                          float* __restrict__ kcvc,
                                          const __half* __restrict__ qnkn,
                                          const float* __restrict__ cossin,
                                          const int* __restrict__ slotpos,
                                          int nh_packed, int hd_packed, int max_pos, float eps){
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    int head = blockIdx.x % nh;
    int m = blockIdx.x / nh;
    int tid = threadIdx.x;
    int kvh = head / (nh / nkv);
    int slot = slotpos[m * 2 + 0];
    int p = slotpos[m * 2 + 1];
    qg += (long long)m * nh * hd * 2;
    kp += (long long)m * nkv * hd;
    vp += (long long)m * nkv * hd;
    const float* c = cossin + (long long)m * 2 * rdim;
    const float* s = c + rdim;
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    __shared__ float wbuf[XQ_SCAN_T];
    __shared__ float wm[32];
    // ---- q: rmsnorm (1+w), rope
    float qv = xq_h2f(qg[(long long)head * hd * 2 + tid]);
    red[tid] = qv * qv; __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
    qbuf[tid] = qv; __syncthreads();
    if (tid < rdim / 2) {
        float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
        qbuf[tid] = x1 * c[tid] - x2 * s[tid];
        qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
    qv = qbuf[tid];
    const int kvf = xq_kv_fmt(max_pos);
    if (kvf >= XQ_KV_Q8) { qv = xq_h32r(qv, tid & 31); qbuf[tid] = qv; }  // WP25 q8: rotated basis
    // ---- k: rmsnorm (1+w), rope, cache write (shared fn = bit-identical to the sweep)
    xq_k_norm_rope(kbuf, red, kp + (long long)kvh * hd, kn_w, c, s, tid, hd, rdim, eps);
    __syncthreads();
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)kvh * hd + tid]));
    __syncthreads();
    // ---- shared two-pass softmax + weighted V (bit-identical to the sweep's;
    // S-A3-m: the warp-tree scan where the head dim allows)
    float acc, denom;
    if (!xq_attn_scan_hd(qbuf, kc, p, tid, hd, wbuf, wm, acc, denom))
        xq_attn_softmax(red, qv, kc, p, tid, hd, acc, denom);
    // S-A3-e FIX (expert-confirmed): qg was ALREADY rebased by m*nh*hd*2 (the batch row);
    // the gate read added the row offset a second time -> row b read row 2b's gate. Row 0
    // was unaffected (first lane always exact); row 1 read stale row 2 -> drifted text.
    float g = xq_h2f(qg[(long long)head * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / denom, kvf, tid & 31);
    attn[(long long)m * nh * hd + (long long)head * hd + tid] =
        xq_f2h(o * xq_sig(g));
}

// din[i] = silu(g[i]) * u[i]  (shared expert: gate/up are separate dense chains).
extern "C" __global__ void xq_silu_mul(__half* __restrict__ din, const __half* __restrict__ g,
                                       const __half* __restrict__ u, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    din[i] = xq_f2h(xq_silu(xq_h2f(g[i])) * xq_h2f(u[i]));
}

// ---- PLE (S-A3-d Phase 0b): grouped RMSNorm (1+w), groups=4 over 2560.
// grid (m*4), block 256; ps = lane*4 + stream. f16 in -> f32 out (key/query path).
extern "C" __global__ void xq_ple_norm_f16(const __half* __restrict__ x, const __half* __restrict__ w,
                                           float* __restrict__ out, float eps) {
    XQ_PDL_ENTRY();
    int ps = blockIdx.x;
    int s = ps & 3;
    const __half* xp = x + (long long)ps * 2560;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) { float v = xq_h2f(xp[d]); part += v * v; }
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float inv = rsqrtf(red[0] / 2560.f + eps);
    const __half* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xq_h2f(xp[d]) * inv * (1.f + xq_h2f(wp[d]));
}

// f32 residual in (the stream stack), f32 out (query path).
extern "C" __global__ void xq_ple_norm_f32(const float* __restrict__ x, const __half* __restrict__ w,
                                           float* __restrict__ out, float eps) {
    XQ_PDL_ENTRY();
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
    const __half* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xp[d] * inv * (1.f + xq_h2f(wp[d]));
}

// f16 in -> f16 out (norm_conv path).
extern "C" __global__ void xq_ple_norm_hh(const __half* __restrict__ x, const __half* __restrict__ w,
                                          __half* __restrict__ out, float eps) {
    XQ_PDL_ENTRY();
    int ps = blockIdx.x;
    int s = ps & 3;
    const __half* xp = x + (long long)ps * 2560;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) { float v = xq_h2f(xp[d]); part += v * v; }
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float inv = rsqrtf(red[0] / 2560.f + eps);
    const __half* wp = w + (long long)s * 2560;
    for (int d = tid; d < 2560; d += 256)
        out[(long long)ps * 2560 + d] = xq_f2h(xq_h2f(xp[d]) * inv * (1.f + xq_h2f(wp[d])));
}

// per-(lane, stream) gate: dot(q,k)*scale -> sigmoid(signed_sqrt(dot)) * value broadcast.
// grid (m*4), block 256. q/k f32 [m*4, 2560]; value f16 [m, 2560]; gated f16 out [m*4, 2560].
extern "C" __global__ void xq_ple_gate(const float* __restrict__ q, const float* __restrict__ k,
                                       const __half* __restrict__ value, __half* __restrict__ gated,
                                       float scale) {
    XQ_PDL_ENTRY();
    int ps = blockIdx.x;
    int p = ps / 4;
    int tid = threadIdx.x;
    __shared__ float red[256];
    float part = 0.f;
    for (int d = tid; d < 2560; d += 256) part += q[(long long)ps * 2560 + d] * k[(long long)ps * 2560 + d];
    red[tid] = part; __syncthreads();
    for (int r = 128; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }
    float dot = red[0] * scale;
    float g = xq_sig(copysignf(sqrtf(fabsf(dot)), dot));
    const __half* v = value + (long long)p * 2560;
    for (int d = tid; d < 2560; d += 256)
        gated[(long long)ps * 2560 + d] = xq_f2h(g * xq_h2f(v[d]));
}

// dilated depthwise conv (kernel 4, dilation 3) + silu, + gated, added into the f32 residual.
// Sequential-step form: per lane, out = w0*st[0] + w1*st[3] + w2*st[6] + w3*cur  (state = last 9).
extern "C" __global__ void xq_ple_conv(float* __restrict__ resid, const __half* __restrict__ gated,
                                       const __half* __restrict__ normed, const __half* __restrict__ state,
                                       const __half* __restrict__ conv_w, int m,
                                       const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)m * 10240) return;
    int q = (int)(i / 10240);
    int d = (int)(i % 10240);
    // S-A3-e FIX: state is PER-SLOT (requests persist in slots; batch row q is just the
    // launch position). Offline q==slot always; serving mixes lanes.
    int slot = slot_ids[q * 2];
    const __half* st = state + ((long long)slot * 10240 + d) * 9;
    float acc = xq_h2f(conv_w[d * 4 + 0]) * xq_h2f(st[0])
              + xq_h2f(conv_w[d * 4 + 1]) * xq_h2f(st[3])
              + xq_h2f(conv_w[d * 4 + 2]) * xq_h2f(st[6])
              + xq_h2f(conv_w[d * 4 + 3]) * xq_h2f(normed[i]);
    resid[i] += xq_h2f(gated[i]) + xq_silu(acc);
}

// roll the conv state: new_state[q,d,i] = old[q,d,i+1] for i<8, else cur normed value.
extern "C" __global__ void xq_ple_state_shift(const __half* __restrict__ normed,
                                              __half* __restrict__ state, int m,
                                              const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)m * 10240) return;
    int q = (int)(i / 10240);
    int d = (int)(i % 10240);
    int slot = slot_ids[q * 2];  // S-A3-e FIX: per-slot state row (see xq_ple_conv).
    __half* st = state + ((long long)slot * 10240 + d) * 9;
    __half nv[9];
    #pragma unroll
    for (int a = 0; a < 9; a++) nv[a] = (a < 8) ? st[a + 1] : normed[i];
    #pragma unroll
    for (int a = 0; a < 9; a++) st[a] = nv[a];
}

// ===========================================================================
// S-A3-f-b: CHUNKED PREFILL kernels. Process C consecutive tokens of ONE slot
// in a single sweep. Weight ops (GEMMs, hc, router) run once at M=C rows via
// the wide-M path; the sequence recurrences (conv rings, GDN scan, attention
// KV+softmax, PLE ring) loop the tokens INSIDE the kernel with the per-token
// reduction orders of the decode kernels unchanged. Prefill sits OUTSIDE the
// batch-invariance contract (same rule as the NVFP4 line); the ladder gate
// compares it against sequential decode-step prefill on real chunks.
// ===========================================================================

// ---- GDN conv1d over a chunk: state ring (k-deep) lives in registers, one
// channel per thread. Per token: shift, append cur, conv, silu (the xq_conv1d
// order: shift+append BEFORE the dot). qkv is overwritten in place (conv out).
extern "C" __global__ void xq_conv1d_chunk(__half* __restrict__ x, float* __restrict__ state,
                                           const float* __restrict__ w, int conv_dim, int k,
                                           int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(x[(long long)t * row_stride + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
        x[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
    }
    for (int j = 0; j < k; j++) sbase[j] = st[j];
}

// ---- WP12: xq_conv1d_chunk for the MTP verify, NOSTORE: reads the LIVE conv ring row
// (`slot`) and never writes it back (the verify's final ring was always discarded — the
// conv commit replays the accepted raw rows from live). Same per-token ops in the same
// order as xq_conv1d_chunk, on the same values the live->shadow copy used to deliver,
// so the conv outputs are bitwise identical; kills the per-layer shadow copy.
extern "C" __global__ void xq_conv1d_chunk_ns(__half* __restrict__ x, const float* __restrict__ state,
                                              const float* __restrict__ w, int conv_dim, int k,
                                              int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    const float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(x[(long long)t * row_stride + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
        x[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
    }
}

// ---- A5 D17: the same conv as xq_conv1d_chunk, fully parallel over (token, channel). Output t
// is silu(sum_j w[c*k+j] * win_t[j]) with win_t[j] = input at time t-(k-1)+j (the ring after
// shift+append), times < 0 read from the pre-chunk state (ring index k+tt) — the identical
// ordered fp32 sum, so outputs are bit-identical. Writes to `out` (inputs must stay intact
// for the other threads); xq_conv1d_state_upd then advances the state from the intact inputs.
extern "C" __global__ void xq_conv1d_chunk_par(__half* __restrict__ out, const __half* __restrict__ x,
                                               const float* __restrict__ state, const float* __restrict__ w,
                                               int conv_dim, int k, int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)C * conv_dim) return;
    const int t = (int)(i / conv_dim), c = (int)(i % conv_dim);
    const float* sbase = state + ((long long)slot * conv_dim + c) * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        const int tt = t - (k - 1) + j;
        const float v = tt >= 0 ? xq_h2f(x[(long long)tt * row_stride + c]) : sbase[k + tt];
        acc += w[c * k + j] * v;
    }
    out[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
}
// the ring after the chunk: st[j] = input at time C-k+j (or the old state's entry for times < 0)
extern "C" __global__ void xq_conv1d_state_upd(float* __restrict__ state, const __half* __restrict__ x,
                                               int conv_dim, int k, int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    float nv[8];
    for (int j = 0; j < k; j++) {
        const int tt = C - k + j;
        nv[j] = tt >= 0 ? xq_h2f(x[(long long)tt * row_stride + c]) : sbase[k + tt];
    }
    for (int j = 0; j < k; j++) sbase[j] = nv[j];
}

// ---- GDN gated-delta-rule scan over a chunk: the xq_gdn_step body with the
// token loop INSIDE the kernel; S lives in SMEM across the chunk (loaded once,
// stored once — the value chain is identical to the per-step kernel).
// 12-arg cap: slot and C packed as slot_C = slot | C<<12 | nostore<<30.
// S-A3-m: nostore = the MTP verify pass — it reads the LIVE state row directly
// and never writes it back (the commit replays the accepted rows from live; the
// verify's final state was always discarded). Kills the per-layer shadow copy
// (3.1 MB r+w) and the discarded store (3.1 MB w) — same values read => same bits.
// S-A3-o G1: 12-arg cap — nh_nk = nh | n_k_heads<<16; `ring` (nullable) receives
// each token's rank-1 update factors for xq_gdn_commit_ring (see there).
#define XQ_GDN_RS 260   // ring floats per (token, head): delta[vd] | krow[kd] | g (vd=kd=128)
extern "C" __global__ void xq_gdn_step_chunk(__half* __restrict__ core, const __half* __restrict__ qkv,
                                             float* __restrict__ state, const __half* __restrict__ b_in,
                                             const __half* __restrict__ a_in, int nh_nk,
                                             int kd, int vd, const float* __restrict__ a_log,
                                             const float* __restrict__ dt_bias, int slot_C,
                                             float* __restrict__ ring) {
    XQ_PDL_ENTRY();
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int slot = slot_C & 0xFFF;
    const int C = (((unsigned)slot_C) >> 12) & 0x3FFFF;
    const bool nostore = (((unsigned)slot_C) >> 30) & 1u;
    int blk = blockIdx.x;
    int nchunk = vd / XQ_GDN_C;
    const int qkv_stride = 2 * n_k_heads * kd + nh * vd;
    int chunk = blk % nchunk; blk /= nchunk;
    int head = blk % nh;
    int key_head = head * n_k_heads / nh;
    int key_dim = n_k_heads * kd;
    int bb0 = chunk * XQ_GDN_C;
    extern __shared__ float sh[];
    float* S_sh = sh;
    float* Srow = S_sh + kd * XQ_GDN_SP;
    float* kv_mem = Srow + kd;
    float* vbuf = kv_mem + XQ_GDN_C;
    float* delta = vbuf + XQ_GDN_C;
    float* qrow = delta + XQ_GDN_C;
    float* krow = qrow + kd;
    float* S = state + ((long long)slot * nh + head) * kd * vd;
    xq_gdn_ld_state(S_sh, S, kd, vd, bb0);
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const __half* col = qkv + (long long)t * qkv_stride;
        float beta = 1.0f / (1.0f + __expf(-xq_h2f(b_in[(long long)t * nh + head])));
        float sp = xq_h2f(a_in[(long long)t * nh + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        float gt = __expf(-__expf(a_log[head]) * sp);
        xq_gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                     core + (long long)t * (nh * vd) + (long long)head * vd,
                     key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
        if (ring != nullptr) {
            // delta/krow still hold THIS token's values (xq_gdn_token ends synced;
            // the next token overwrites them only after the barrier below).
            float* rr = ring + ((long long)t * nh + head) * XQ_GDN_RS;
            if (threadIdx.x < XQ_GDN_C) rr[bb0 + threadIdx.x] = delta[threadIdx.x];
            if (chunk == 0) {
                rr[vd + threadIdx.x] = krow[threadIdx.x];
                if (threadIdx.x == 0) rr[vd + kd] = gt;
            }
        }
        __syncthreads();
    }
    if (nostore) return;
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// ---- WP12 (2026-09-26): register-resident GDN recurrence — the MTP verify chunk
// (xq_gdn_step_chunk_r) and the decode step (xq_gdn_step_r) run the SAME device body
// (AGENTS §2.8). Bitwise equal to xq_gdn_token token by token (the old kernels stay
// reachable: GB10_WP12_OFF=step). Target contract: plain sm_121 — FP32 FMUL/FFMA,
// warp shuffles, shared memory, __syncthreads; no f/a feature is used.
// Geometry: block = 4 warps (vd = 128 columns), one block per (row group, head); warp w
// owns state columns [32w, 32w+32), lane = column, and holds its whole kd = 128 column
// in 128 registers for the token walk (the old body: S in smem, ~25 block barriers and
// 2 dependent global-load round trips per token; this one: none).
// Exactness vs xq_gdn_token (every op pinned with an _rn intrinsic, same order):
//  * q/k L2 norms: squares, halving tree 64, 32 then 16..1 — emulated in-warp exactly as
//    xq_gdn_chunk_tc_body does (lane l holds rows l, l+32, l+64, l+96); rsqrtf(sum+1e-6f);
//    q = (q*qn)*scale, k = k*kn; scale = 1/sqrtf((float)kd) from the RUNTIME kd argument
//    (the same IEEE sqrt/div sequence as the old kernels, never constant-folded);
//  * gates (beta, g): the verbatim expressions of xq_gdn_step/_chunk (xq_gdnr_gates);
//  * S *= g as FMUL; km = FFMA chain over a ascending from +0 on the scaled S;
//    delta = (v - km)*beta (FADD, FMUL); S = FFMA(k, delta, S); o = FFMA chain over a
//    ascending from +0 on the UPDATED S; core = f16_rn(o);
//  * o_t is computed in the same pass as km_{t+1}: both read S_t (o before the FMUL by
//    g_{t+1}, km after) — two independent chains, neither reordered.
// All C tokens' q/k (normalized), v rows and gates are staged once in smem while the
// state column is in flight (the per-token dependent global loads were the binder).
// The ring (nullable) receives the SAME rank-1 factors as xq_gdn_step_chunk's.
#define XQ_GDNR_KD   128
#define XQ_GDNR_VD   128
#define XQ_GDNR_CMAX 8
__device__ __forceinline__ void xq_gdnr_gates(float bh, float ah, float dtb, float alog,
                                              float& beta, float& gt) {
    beta = 1.0f / (1.0f + __expf(-bh));
    float sp = ah + dtb;
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    gt = __expf(-__expf(alog) * sp);
}
// WITH_O: o = sum_a S[a]*q[a] (S before the g scale), then S *= g, km = sum_a S[a]*k[a].
template <bool WITH_O>
__device__ __forceinline__ void xq_gdnr_pass(float (&s)[XQ_GDNR_KD], const float* __restrict__ qr,
                                             const float* __restrict__ kr, const float gt,
                                             float& o, float& km) {
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(kr + a);
        float4 q4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (WITH_O) q4 = *reinterpret_cast<const float4*>(qr + a);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (WITH_O) o = __fmaf_rn(s[a + j], qq[j], o);
            s[a + j] = __fmul_rn(s[a + j], gt);
            km = __fmaf_rn(s[a + j], kk[j], km);
        }
    }
}
__device__ __forceinline__ void xq_gdnr_update(float (&s)[XQ_GDNR_KD], const float* __restrict__ kr,
                                               const float delta) {
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(kr + a);
        s[a + 0] = __fmaf_rn(k4.x, delta, s[a + 0]);
        s[a + 1] = __fmaf_rn(k4.y, delta, s[a + 1]);
        s[a + 2] = __fmaf_rn(k4.z, delta, s[a + 2]);
        s[a + 3] = __fmaf_rn(k4.w, delta, s[a + 3]);
    }
}
__device__ __forceinline__ float xq_gdnr_out(const float (&s)[XQ_GDNR_KD], const float* __restrict__ qr) {
    float o = 0.0f;
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 q4 = *reinterpret_cast<const float4*>(qr + a);
        o = __fmaf_rn(s[a + 0], q4.x, o);
        o = __fmaf_rn(s[a + 1], q4.y, o);
        o = __fmaf_rn(s[a + 2], q4.z, o);
        o = __fmaf_rn(s[a + 3], q4.w, o);
    }
    return o;
}
// rows: token t reads qkv/b/a row (row0 + t) and writes core row (row0 + t); the ring
// (verify only) is indexed by t. store = write the final state back (decode).
__device__ __forceinline__ void xq_gdnr_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in, int nh, int n_k_heads,
    int kd_rt, int vd, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int slot, int row0, int head, int C, bool store, float* __restrict__ ring)
{
    constexpr int KD = XQ_GDNR_KD, VD = XQ_GDNR_VD;
    __shared__ __align__(16) float q_sh[XQ_GDNR_CMAX][KD];
    __shared__ __align__(16) float k_sh[XQ_GDNR_CMAX][KD];
    __shared__ float v_sh[XQ_GDNR_CMAX][VD];
    __shared__ float beta_sh[XQ_GDNR_CMAX], g_sh[XQ_GDNR_CMAX];
    if (kd_rt != KD || vd != VD || blockDim.x != VD || C < 1 || C > XQ_GDNR_CMAX) __trap();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = threadIdx.x;                       // = warp*32 + lane
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const long long qkv_stride = 2 * key_dim + nh * VD;
    float* Sg = state + ((long long)slot * nh + head) * KD * VD + col;
    // (1) the state column -> registers (the long pole: issued first, bandwidth-bound)
    float s[KD];
    #pragma unroll
    for (int a = 0; a < KD; a++) s[a] = Sg[(long long)a * VD];
    // (2) stage every token's normalized q/k, v row and gates (warp w: t = w, w+4, ...)
    const float scale = 1.0f / sqrtf((float)kd_rt);
    const float alog = a_log[head], dtb = dt_bias[head];
    for (int t = warp; t < C; t += 4) {
        const long long r = row0 + t;
        const __half* colp = qkv + r * qkv_stride;
        const __half* qp = colp + key_head * KD;
        const __half* kp = colp + key_dim + key_head * KD;
        const __half* vp = colp + 2 * key_dim + head * VD;
        float q[4], k[4];
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            q[m] = xq_h2f(qp[lane + 32 * m]);
            k[m] = xq_h2f(kp[lane + 32 * m]);
            v_sh[t][lane + 32 * m] = xq_h2f(vp[lane + 32 * m]);
        }
        float beta, gt;
        xq_gdnr_gates(xq_h2f(b_in[r * nh + head]), xq_h2f(a_in[r * nh + head]), dtb, alog, beta, gt);
        // halving tree: s2=64 pairs (i, i+64), s2=32 pairs (i, i+32), then 16..1
        float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                             __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
        float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                             __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
            sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
        }
        const float qn = rsqrtf(__fadd_rn(__shfl_sync(0xFFFFFFFFu, sq, 0), 1e-6f));
        const float kn = rsqrtf(__fadd_rn(__shfl_sync(0xFFFFFFFFu, sk, 0), 1e-6f));
        float* rr = ring != nullptr ? ring + ((long long)t * nh + head) * XQ_GDN_RS : nullptr;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            const int i = lane + 32 * m;
            q_sh[t][i] = __fmul_rn(__fmul_rn(q[m], qn), scale);
            const float kv = __fmul_rn(k[m], kn);
            k_sh[t][i] = kv;
            if (rr != nullptr) rr[VD + i] = kv;
        }
        if (lane == 0) {
            beta_sh[t] = beta;
            g_sh[t] = gt;
            if (rr != nullptr) rr[VD + KD] = gt;
        }
    }
    __syncthreads();
    // (3) the token walk — no block barriers, no global loads
    __half* cb = core + (long long)row0 * (nh * VD) + (long long)head * VD + col;
    {
        float km = 0.0f, o_unused = 0.0f;
        xq_gdnr_pass<false>(s, nullptr, k_sh[0], g_sh[0], o_unused, km);
        // re-read k in the update (a warp fence stops the compiler keeping all 128 k
        // values of the pass live across the km chain — that was a 255-reg spill)
        __syncwarp();
        const float delta = __fmul_rn(__fsub_rn(v_sh[0][col], km), beta_sh[0]);
        if (ring != nullptr) ring[(long long)head * XQ_GDN_RS + col] = delta;
        xq_gdnr_update(s, k_sh[0], delta);
    }
    for (int t = 1; t < C; t++) {
        float km = 0.0f, o = 0.0f;
        xq_gdnr_pass<true>(s, q_sh[t - 1], k_sh[t], g_sh[t], o, km);
        cb[(long long)(t - 1) * (nh * VD)] = xq_f2h(o);
        __syncwarp();
        const float delta = __fmul_rn(__fsub_rn(v_sh[t][col], km), beta_sh[t]);
        if (ring != nullptr) ring[((long long)t * nh + head) * XQ_GDN_RS + col] = delta;
        xq_gdnr_update(s, k_sh[t], delta);
    }
    cb[(long long)(C - 1) * (nh * VD)] = xq_f2h(xq_gdnr_out(s, q_sh[C - 1]));
    if (!store) return;
    #pragma unroll
    for (int a = 0; a < KD; a++) Sg[(long long)a * VD] = s[a];
}
// verify chunk: the xq_gdn_step_chunk signature/packing (slot_C = slot | C<<12 | nostore<<30,
// nh_nk = nh | n_k_heads<<16); grid nh, block 128 (host: kd = vd = 128, C <= 8).
extern "C" __global__ void __launch_bounds__(128, 1)
xq_gdn_step_chunk_r(__half* __restrict__ core, const __half* __restrict__ qkv,
                    float* __restrict__ state, const __half* __restrict__ b_in,
                    const __half* __restrict__ a_in, int nh_nk, int kd, int vd,
                    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                    int slot_C, float* __restrict__ ring) {
    XQ_PDL_ENTRY();
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int slot = slot_C & 0xFFF;
    const int C = (((unsigned)slot_C) >> 12) & 0x3FFFF;
    const bool nostore = (((unsigned)slot_C) >> 30) & 1u;
    xq_gdnr_body(core, qkv, state, b_in, a_in, nh, n_k_heads, kd, vd, a_log, dt_bias,
                 slot, 0, blockIdx.x, C, !nostore, ring);
}
// decode step: the xq_gdn_step signature; grid B*nh (block = b*nh + head), block 128.
extern "C" __global__ void __launch_bounds__(128, 1)
xq_gdn_step_r(__half* __restrict__ core, const __half* __restrict__ qkv,
              float* __restrict__ state, const __half* __restrict__ b_in,
              const __half* __restrict__ a_in, int nh, int n_k_heads, int kd, int vd,
              const float* __restrict__ a_log, const float* __restrict__ dt_bias,
              const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    const int b = blockIdx.x / nh;
    const int head = blockIdx.x - b * nh;
    // S-A3-e: the recurrent state is PER-SLOT (slotpos is interleaved [slot, pos])
    const int slot = slot_ids[b * 2];
    xq_gdnr_body(core, qkv, state, b_in, a_in, nh, n_k_heads, kd, vd, a_log, dt_bias,
                 slot, b, head, 1, true, nullptr);
}

// ---- S-A3-v: CHUNKWISE-PARALLEL GDN prefill scan (tensor cores, split-f16 operands).
// The xq_gdn_step_chunk recurrence re-associated into 32-token chunks (the WY/UT form
// of gdn_chunk_tc_b, gpu_batch.cu — same chunk math, validated there vs the sequential
// scan): per chunk, the intra-chunk grams (A = decayed KK^T, D = decayed QK^T), the
// forward substitution U = (I + diag(beta)A)^-1 diag(beta)(V - Gamma K S), then
// O = Gamma Q S + D U and S' = gamma_C S + K_w^T U — every product on mma.m16n8k16.
// Differences from gdn_chunk_tc_b (numerics care, the prefillx state drift is the gate):
//  * reads the EXL3 packed f16 qkv / b / a rows directly (no f32 prep scratch: that is
//    ~300 MB of traffic per layer at C=2048) and writes f16 core rows;
//  * q/k L2 norms computed in-kernel in the EXACT xq_gdn_token order (squares, halving
//    tree 64,32 then 16..1, no FMA contraction) => the normalized q/k equal decode's bits;
//  * SPLIT f16 operands: x = hi + lo (hi = f16(x), lo = f16(x - hi)), each product as
//    hi*hi + hi*lo + lo*hi => ~2^-17..2^-21 relative per operand instead of f16's 2^-11
//    (S-A3-v cycle 1: plain f16 operands drifted out of class — prefillx maxd 22.2 / 15.2
//    / 18.8 vs bars 12.64 / 6.98 / 5.24; the W = V - Gamma K S difference cancels, so
//    operand rounding of K and S is amplified). Template mask SPLIT: 1 = S, 2 = Q/K,
//    4 = U/D (7 = all; the ablation lives in THINGS_TRIED);
//  * the f32 MASTER state stays in registers for the whole walk (warp w owns kd rows
//    [16w, 16w+16) x the block's VC cols); its operand copy Sf is f32 (split on load).
// Prefill-only: decode (xq_gdn_step) and verify (xq_gdn_step_chunk nostore/ring) untouched.
// kd = vd = 128 (host-checked). Grid nh * (128/XQ_GTC_VC), block 256, dynamic smem
// XQ_GTC_SMEM (> 48 KB: raw launch with the opt-in, exl3_forward.rs gdn_chunk_tc_fn).
#define XQ_GTC_C   32
#define XQ_GTC_VC  32
#define XQ_GTC_KD2 (128 + 8)
#define XQ_GTC_CD2 (XQ_GTC_C + 8)
#define XQ_GTC_VD2 (XQ_GTC_VC + 8)
#define XQ_GTC_SD2 (XQ_GTC_VC + 4)
#define XQ_GTC_THR 256
// halves: Qh,Ql,Kh,Kl [C][KD2] + Vb [C][VD2] + Dh,Dl [C][CD2] + Uh,Ul,Lh,Ll [C][VD2]
// f32:    Sf [128][SD2] + Af [C][C] + Wf [C][VC] + bt [C] + lg [C+1]
#define XQ_GTC_SMEM ((4 * XQ_GTC_C * XQ_GTC_KD2 + XQ_GTC_C * XQ_GTC_VD2 + 2 * XQ_GTC_C * XQ_GTC_CD2 \
                      + 4 * XQ_GTC_C * XQ_GTC_VD2) * 2 \
                     + (128 * XQ_GTC_SD2 + XQ_GTC_C * XQ_GTC_C + XQ_GTC_C * XQ_GTC_VC + XQ_GTC_C + XQ_GTC_C + 1) * 4)
__device__ __forceinline__ unsigned xq_gtc_h2(__half lo, __half hi) {
    __half2 v = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void xq_gtc_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// split store of an f32 value into a hi/lo tile pair
__device__ __forceinline__ void xq_gtc_st2(__half* h, __half* lo, int i, float x) {
    const __half xh = __float2half_rn(x);
    h[i] = xh; lo[i] = __float2half_rn(x - __half2float(xh));
}
// A fragment (row-major tile, the permuted-k convention of gdn_chunk_tc_b: k = cc..cc+3)
__device__ __forceinline__ void xq_gtc_lda(unsigned* a, const __half* T, int ld, int r0, int cc) {
    a[0] = *(const unsigned*)(T + r0 * ld + cc);       a[1] = *(const unsigned*)(T + (r0 + 8) * ld + cc);
    a[2] = *(const unsigned*)(T + r0 * ld + cc + 2);   a[3] = *(const unsigned*)(T + (r0 + 8) * ld + cc + 2);
}
// A fragment of the TRANSPOSE of a row-major tile T[k][row] (K^T for the S update)
__device__ __forceinline__ void xq_gtc_ldat(unsigned* a, const __half* T, int ld, int r0, int cc) {
    a[0] = xq_gtc_h2(T[cc * ld + r0], T[(cc + 1) * ld + r0]);
    a[1] = xq_gtc_h2(T[cc * ld + r0 + 8], T[(cc + 1) * ld + r0 + 8]);
    a[2] = xq_gtc_h2(T[(cc + 2) * ld + r0], T[(cc + 3) * ld + r0]);
    a[3] = xq_gtc_h2(T[(cc + 2) * ld + r0 + 8], T[(cc + 3) * ld + r0 + 8]);
}
// B fragment from a row-major [k][n] tile of halves (rows cc..cc+3, column col)
__device__ __forceinline__ void xq_gtc_ldb(unsigned* b, const __half* T, int ld, int cc, int col) {
    b[0] = xq_gtc_h2(T[cc * ld + col], T[(cc + 1) * ld + col]);
    b[1] = xq_gtc_h2(T[(cc + 2) * ld + col], T[(cc + 3) * ld + col]);
}
// B fragment pair (hi, lo) split on the fly from an f32 [k][n] tile
__device__ __forceinline__ void xq_gtc_ldb_f32(unsigned* bh, unsigned* bl, const float* T, int ld, int cc, int col) {
    float x[4] = {T[cc * ld + col], T[(cc + 1) * ld + col], T[(cc + 2) * ld + col], T[(cc + 3) * ld + col]};
    __half h[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { h[i] = __float2half_rn(x[i]); l[i] = __float2half_rn(x[i] - __half2float(h[i])); }
    bh[0] = xq_gtc_h2(h[0], h[1]); bh[1] = xq_gtc_h2(h[2], h[3]);
    bl[0] = xq_gtc_h2(l[0], l[1]); bl[1] = xq_gtc_h2(l[2], l[3]);
}
// d += A*B with optional lo terms (small terms first)
__device__ __forceinline__ void xq_gtc_mma3(float* d, const unsigned* ah, const unsigned* al, bool sa,
                                            const unsigned* bh, const unsigned* bl, bool sb) {
    if (sa) xq_gtc_mma(d, al, bh);
    if (sb) xq_gtc_mma(d, ah, bl);
    xq_gtc_mma(d, ah, bh);
}
// A5-L1: harness-only phase stamps (PROF = true twins only; PROF = false compiles to the served SASS).
// Thread 0 accumulates clock64 deltas between consecutive block barriers into pacc[phase]; the
// block writes pacc[0..7] + globaltimer start/end to prof[blockIdx.x * 10 ..].
#define XQ_GTC_STAMP(i) do { if (PROF && t == 0) { const long long _n = clock64(); pacc[i] += _n - pt; pt = _n; } } while (0)
#define XQ_GTC_PROF_BEGIN() long long pacc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
    long long pt = PROF ? clock64() : 0; unsigned long long pg0 = PROF ? xq_tune_gtimer() : 0ull
#define XQ_GTC_PROF_END() do { if (PROF && t == 0 && prof != nullptr) { \
    _Pragma("unroll") for (int _i = 0; _i < 8; _i++) prof[blockIdx.x * 10 + _i] = (unsigned long long)pacc[_i]; \
    prof[blockIdx.x * 10 + 8] = pg0; prof[blockIdx.x * 10 + 9] = xq_tune_gtimer(); } } while (0)
template <int SPLIT, bool PROF = false>
__device__ __forceinline__ void xq_gdn_chunk_tc_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int nh_nk, int slot, int N, unsigned long long* __restrict__ prof = nullptr)
{
    constexpr bool SS = SPLIT & 1, SQK = SPLIT & 2, SUD = SPLIT & 4;
    constexpr int C = XQ_GTC_C, VC = XQ_GTC_VC, KD = 128, VD = 128;
    constexpr int KD2 = XQ_GTC_KD2, CD2 = XQ_GTC_CD2, VD2 = XQ_GTC_VD2, SD2 = XQ_GTC_SD2;
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int ncb = VD / VC;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const int qkv_stride = 2 * key_dim + nh * VD;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;

    extern __shared__ __align__(16) unsigned char xq_gtc_dyn[];
    __half* Qh = (__half*)xq_gtc_dyn;       // [C][KD2]
    __half* Ql = Qh + C * KD2;
    __half* Kh = Ql + C * KD2;
    __half* Kl = Kh + C * KD2;
    __half* Vb = Kl + C * KD2;              // [C][VD2]
    __half* Dh = Vb + C * VD2;              // [C][CD2]
    __half* Dl = Dh + C * CD2;
    __half* Uh = Dl + C * CD2;              // [C][VD2] (unscaled U)
    __half* Ul = Uh + C * VD2;
    __half* Lh = Ul + C * VD2;              // [C][VD2] (lambda-scaled U)
    __half* Ll = Lh + C * VD2;
    float* Sf = (float*)(Ll + C * VD2);     // [KD][SD2]
    float* Af = Sf + KD * SD2;              // [C][C]
    float* Wf = Af + C * C;                 // [C][VC]
    float* bt = Wf + C * VC;                // [C]
    float* lg = bt + C;                     // [C+1]

    // f32 master state: warp w owns kd rows [16w, 16w+16) x VC cols (4 n-tiles x 4)
    float* Sg = state + ((long long)slot * nh + head) * KD * VD + bb0;
    float Sr[VC / 8][4];
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sr[j][e] = Sg[(long long)row * VD + col];
            Sf[row * SD2 + col] = Sr[j][e];
        }
    const float qscale = 1.0f / sqrtf((float)KD);
    const float neg_a = -__expf(a_log[head]);
    XQ_GTC_PROF_BEGIN();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage q/k (normalized in xq_gdn_token's exact order) + v + beta/log-g ----
        for (int tt = warp; tt < C; tt += XQ_GTC_THR / 32) {
            if (tt < n) {
                const __half* col = qkv + (long long)(c0 + tt) * qkv_stride;
                const __half* qp = col + key_head * KD;
                const __half* kp = col + key_dim + key_head * KD;
                float q[4], k[4];
                #pragma unroll
                for (int m = 0; m < 4; m++) { q[m] = xq_h2f(qp[lane + 32 * m]); k[m] = xq_h2f(kp[lane + 32 * m]); }
                // halving tree over squares: s2=64 pairs (i, i+64), s2=32 pairs (i, i+32), then 16..1
                float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                                     __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
                float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                                     __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
                    sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
                }
                const float qn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sq, 0) + 1e-6f);
                const float kn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sk, 0) + 1e-6f);
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, __fmul_rn(__fmul_rn(q[m], qn), qscale));
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, __fmul_rn(k[m], kn));
                }
            } else {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, 0.f);
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, 0.f);
                }
            }
        }
        for (int i = t; i < C * VC; i += XQ_GTC_THR) {
            const int tt = i / VC, c = i % VC;
            Vb[tt * VD2 + c] = (tt < n)
                ? qkv[(long long)(c0 + tt) * qkv_stride + 2 * key_dim + head * VD + bb0 + c]
                : __float2half_rn(0.f);
        }
        if (t < C) {
            if (t < n) {
                const long long r = (long long)(c0 + t) * nh + head;
                bt[t] = 1.0f / (1.0f + __expf(-xq_h2f(b_in[r])));
                float sp = xq_h2f(a_in[r]) + dt_bias[head];
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                lg[t + 1] = neg_a * sp;
            } else { bt[t] = 0.f; lg[t + 1] = 0.f; }
        }
        __syncthreads();
        XQ_GTC_STAMP(0);
        if (warp == 0) {                         // inclusive scan of log-g (one warp, C = 32)
            float v = lg[lane + 1];
            #pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                const float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if (lane >= o) v += u;
            }
            lg[lane + 1] = v;
            if (lane == 0) lg[0] = 0.f;
        }
        __syncthreads();
        XQ_GTC_STAMP(1);

        const int mt = warp >> 2, nt = warp & 3;   // 32x32 outputs: one m16n8 tile per warp
        const int r0 = mt * 16 + g;
        // ---- A / D grams (k = 128) and W = V - exp(lg[tt+1]) (K S) (k = 128) ----
        {
            float accA[4] = {0, 0, 0, 0}, accD[4] = {0, 0, 0, 0}, accW[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aKh[4], aKl[4], aQh[4], aQl[4], bKh[2], bKl[2], bSh[2], bSl[2];
                xq_gtc_lda(aKh, Kh, KD2, r0, cc); xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                if (SQK) { xq_gtc_lda(aKl, Kl, KD2, r0, cc); xq_gtc_lda(aQl, Ql, KD2, r0, cc); }
                const int s_row = nt * 8 + g;       // B = K^T: column s = row s of K
                bKh[0] = *(const unsigned*)(Kh + s_row * KD2 + cc); bKh[1] = *(const unsigned*)(Kh + s_row * KD2 + cc + 2);
                if (SQK) { bKl[0] = *(const unsigned*)(Kl + s_row * KD2 + cc); bKl[1] = *(const unsigned*)(Kl + s_row * KD2 + cc + 2); }
                xq_gtc_mma3(accA, aKh, aKl, SQK, bKh, bKl, SQK);
                xq_gtc_mma3(accD, aQh, aQl, SQK, bKh, bKl, SQK);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(accW, aKh, aKl, SQK, bSh, bSl, SS);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                // A[t,s] = (gamma_t / gamma_s)(k_t . k_s): g_t S_{t-1} carries gamma_t. gdn_chunk_tc_b
                // uses gamma_{t-1} (lg[row]) — one gate factor short per row (S-A3-v; hidden as its
                // "~2% bf16 envelope" because g ~ 1).
                Af[row * C + scol] = (scol < row) ? accA[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                const float dv = (scol <= row && scol < n) ? accD[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                if (SUD) xq_gtc_st2(Dh, Dl, row * CD2 + scol, dv); else Dh[row * CD2 + scol] = __float2half_rn(dv);
                const float v = (row < n) ? xq_h2f(Vb[row * VD2 + scol]) : 0.f;
                Wf[row * VC + scol] = v - __expf(lg[row + 1]) * accW[e];
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(2);

        // ---- U = (I + diag(beta)A)^-1 (beta . W): forward substitution, one column per thread ----
        if (t < VC) {
            for (int tt = 0; tt < n; tt++) {
                const float b = bt[tt];
                float acc = b * Wf[tt * VC + t];
                for (int s = 0; s < tt; s++) acc -= b * Af[tt * C + s] * Wf[s * VC + t];
                Wf[tt * VC + t] = acc;
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(3);
        // ---- U (unscaled) and lambda-scaled U (lambda_s = exp(lg[n] - lg[s+1])) ----
        for (int i = t; i < C * VC; i += XQ_GTC_THR) {
            const int s = i / VC, c = i % VC;
            const float u = (s < n) ? Wf[s * VC + c] : 0.f;
            const float ul = u * __expf(lg[n] - lg[s + 1]);
            if (SUD) { xq_gtc_st2(Uh, Ul, s * VD2 + c, u); xq_gtc_st2(Lh, Ll, s * VD2 + c, ul); }
            else { Uh[s * VD2 + c] = __float2half_rn(u); Lh[s * VD2 + c] = __float2half_rn(ul); }
        }
        __syncthreads();
        XQ_GTC_STAMP(4);

        // ---- O = exp(lg[tt+1]) (Q S) + D U  (QS k = 128, then DU k = C accumulates) ----
        {
            float acc[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aQh[4], aQl[4], bSh[2], bSl[2];
                xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                if (SQK) xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aQh, aQl, SQK, bSh, bSl, SS);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[e] *= __expf(lg[r0 + 8 * (e >= 2) + 1]);
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aDh[4], aDl[4], bUh[2], bUl[2];
                xq_gtc_lda(aDh, Dh, CD2, r0, cc);
                xq_gtc_ldb(bUh, Uh, VD2, cc, nt * 8 + g);
                if (SUD) { xq_gtc_lda(aDl, Dl, CD2, r0, cc); xq_gtc_ldb(bUl, Ul, VD2, cc, nt * 8 + g); }
                xq_gtc_mma3(acc, aDh, aDl, SUD, bUh, bUl, SUD);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                if (row < n) {
                    const int col = nt * 8 + 2 * tq + (e & 1);
                    core[(long long)(c0 + row) * (nh * VD) + head * VD + bb0 + col] = __float2half_rn(acc[e]);
                }
            }
        }
        __syncthreads();   // every Sf read (W, O) is done before the update rewrites it
        XQ_GTC_STAMP(5);

        // ---- S' = gamma_C S + K^T . Ulambda  (warp = 16 kd rows x VC; k = C); master stays f32 ----
        {
            const float gam = __expf(lg[n]);
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) Sr[j][e] *= gam;
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aTh[4], aTl[4];
                xq_gtc_ldat(aTh, Kh, KD2, warp * 16 + g, cc);
                if (SQK) xq_gtc_ldat(aTl, Kl, KD2, warp * 16 + g, cc);
                #pragma unroll
                for (int j = 0; j < VC / 8; j++) {
                    unsigned bLh[2], bLl[2];
                    xq_gtc_ldb(bLh, Lh, VD2, cc, j * 8 + g);
                    if (SUD) xq_gtc_ldb(bLl, Ll, VD2, cc, j * 8 + g);
                    xq_gtc_mma3(Sr[j], aTh, aTl, SQK, bLh, bLl, SUD);
                }
            }
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = warp * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sf[row * SD2 + col] = Sr[j][e];
                }
        }
        __syncthreads();
        XQ_GTC_STAMP(6);
    }
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sg[(long long)row * VD + col] = Sr[j][e];
        }
    XQ_GTC_PROF_END();
}
#define XQ_GTC_INST(NAME, SPLIT) \
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void NAME( \
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state, \
    const __half* __restrict__ b_in, const __half* __restrict__ a_in, \
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N) { \
    XQ_PDL_ENTRY(); \
    xq_gdn_chunk_tc_body<SPLIT>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N); }
XQ_GTC_INST(xq_gdn_chunk_tc, 7)
XQ_GTC_INST(xq_gdn_chunk_tc_s0, 0)
XQ_GTC_INST(xq_gdn_chunk_tc_s1, 1)
XQ_GTC_INST(xq_gdn_chunk_tc_s3, 3)

// ---- A5-L1 (2026-09-28): xq_gdn_chunk_tc2 — the served xq_gdn_chunk_tc (SPLIT = 7) with the same
// per-element arithmetic, re-scheduled. BITWISE equal (--probe-exl3-binv EXL3-GDN-TC2). Changes:
//  (1) forward substitution U = (I + diag(beta)A)^-1 diag(beta) W as a 31-step WAVEFRONT instead of one
//      thread per column walking 496 dependent smem-fed FFMAs: lane = row r, warp w owns columns
//      4w..4w+3; step k broadcasts the finished W[k] (__shfl) and every row r > k does its s = k term.
//      Each row keeps its exact chain: acc = b_r*W[r] (FMUL), then acc = fma(-(b_r*A[r][s]), W[s], acc)
//      for s = 0, 1, .., r-1 ascending — the SASS of the served loop (FMUL b*A; FFMA -t, W, acc).
//      b_r*A[r][s] (the served `b * Af`) is formed once in the gram epilogue, stored transposed (Tt[s][r])
//      so the per-step reads are conflict-free; W is stored transposed (Wt[c][r]) for the same reason.
//  (2) the U / lambda-U split stores run in the solve's registers (no Wf round trip, one barrier less);
//  (3) the log-g inclusive scan runs in warp 0 straight from the staging registers (one barrier less);
//  (4) the next 32-token sub-chunk's q/k/v/b/a global loads are issued into registers right after the
//      current sub-chunk is staged, so their latency hides behind the grams/solve/O/S phases.
// Same smem footprint (XQ_GTC_SMEM: Af -> Tt, Wf -> Wt, same sizes), same grid / block / launch.
template <bool PROF>
__device__ __forceinline__ void xq_gdn_chunk_tc2_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int nh_nk, int slot, int N, unsigned long long* __restrict__ prof)
{
    constexpr int C = XQ_GTC_C, VC = XQ_GTC_VC, KD = 128, VD = 128;
    constexpr int KD2 = XQ_GTC_KD2, CD2 = XQ_GTC_CD2, VD2 = XQ_GTC_VD2, SD2 = XQ_GTC_SD2;
    static_assert(C == 32 && VC == 32 && XQ_GTC_THR == 256, "tc2 geometry: lane = row, warp = 4 columns");
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int ncb = VD / VC;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const int qkv_stride = 2 * key_dim + nh * VD;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;

    extern __shared__ __align__(16) unsigned char xq_gtc_dyn[];
    __half* Qh = (__half*)xq_gtc_dyn;       // [C][KD2]
    __half* Ql = Qh + C * KD2;
    __half* Kh = Ql + C * KD2;
    __half* Kl = Kh + C * KD2;
    __half* Vb = Kl + C * KD2;              // [C][VD2]
    __half* Dh = Vb + C * VD2;              // [C][CD2]
    __half* Dl = Dh + C * CD2;
    __half* Uh = Dl + C * CD2;              // [C][VD2] (unscaled U)
    __half* Ul = Uh + C * VD2;
    __half* Lh = Ul + C * VD2;              // [C][VD2] (lambda-scaled U)
    __half* Ll = Lh + C * VD2;
    float* Sf = (float*)(Ll + C * VD2);     // [KD][SD2]
    float* Tt = Sf + KD * SD2;              // [C][C]  Tt[s][r] = beta_r * A[r][s]  (s < r)
    float* Wt = Tt + C * C;                 // [VC][C] Wt[c][r] = W[r][c]
    float* bt = Wt + C * VC;                // [C]
    float* lg = bt + C;                     // [C+1]

    float* Sg = state + ((long long)slot * nh + head) * KD * VD + bb0;
    float Sr[VC / 8][4];
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sr[j][e] = Sg[(long long)row * VD + col];
            Sf[row * SD2 + col] = Sr[j][e];
        }
    const float qscale = 1.0f / sqrtf((float)KD);
    const float neg_a = -__expf(a_log[head]);
    const float dtb = dt_bias[head];

    // (4) register prefetch of one sub-chunk's raw inputs: tokens tt = warp + 8i (i < 4) q/k halves
    // lane + 32m; V elements i = t + 256j (tt = i / VC, c = i % VC); b/a of token t (warp 0).
    __half pq[4][4], pk[4][4], pv[4], pb, pa;
    auto prefetch = [&](int c0n) {
        const int nn = min(C, N - c0n);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int tt = warp + 8 * i;
            const __half* col = qkv + (long long)(c0n + (tt < nn ? tt : 0)) * qkv_stride;
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                pq[i][m] = (tt < nn) ? col[key_head * KD + lane + 32 * m] : __float2half_rn(0.f);
                pk[i][m] = (tt < nn) ? col[key_dim + key_head * KD + lane + 32 * m] : __float2half_rn(0.f);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = t + XQ_GTC_THR * j, tt = i / VC, c = i % VC;
            pv[j] = (tt < nn) ? qkv[(long long)(c0n + tt) * qkv_stride + 2 * key_dim + head * VD + bb0 + c]
                              : __float2half_rn(0.f);
        }
        if (warp == 0 && lane < nn) {
            const long long r = (long long)(c0n + lane) * nh + head;
            pb = b_in[r]; pa = a_in[r];
        } else { pb = __float2half_rn(0.f); pa = __float2half_rn(0.f); }
    };
    prefetch(0);
    XQ_GTC_PROF_BEGIN();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage q/k (normalized in xq_gdn_token's exact order) + v + beta/log-g (+ its scan) ----
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int tt = warp + 8 * i;
            if (tt < n) {
                float q[4], k[4];
                #pragma unroll
                for (int m = 0; m < 4; m++) { q[m] = xq_h2f(pq[i][m]); k[m] = xq_h2f(pk[i][m]); }
                float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                                     __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
                float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                                     __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
                    sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
                }
                const float qn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sq, 0) + 1e-6f);
                const float kn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sk, 0) + 1e-6f);
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, __fmul_rn(__fmul_rn(q[m], qn), qscale));
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, __fmul_rn(k[m], kn));
                }
            } else {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, 0.f);
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, 0.f);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = t + XQ_GTC_THR * j, tt = i / VC, c = i % VC;
            Vb[tt * VD2 + c] = pv[j];
        }
        if (warp == 0) {                         // (3) beta + log-g, then its inclusive scan, in-warp
            float v = 0.f;
            if (lane < n) {
                bt[lane] = 1.0f / (1.0f + __expf(-xq_h2f(pb)));
                float sp = xq_h2f(pa) + dtb;
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                v = neg_a * sp;
            } else { bt[lane] = 0.f; }
            #pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                const float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if (lane >= o) v += u;
            }
            lg[lane + 1] = v;
            if (lane == 0) lg[0] = 0.f;
        }
        if (c0 + C < N) prefetch(c0 + C);        // (4) next sub-chunk's loads fly behind the phases below
        __syncthreads();
        XQ_GTC_STAMP(0);

        const int mt = warp >> 2, nt = warp & 3;   // 32x32 outputs: one m16n8 tile per warp
        const int r0 = mt * 16 + g;
        // ---- A / D grams (k = 128) and W = V - exp(lg[tt+1]) (K S) (k = 128) ----
        {
            float accA[4] = {0, 0, 0, 0}, accD[4] = {0, 0, 0, 0}, accW[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aKh[4], aKl[4], aQh[4], aQl[4], bKh[2], bKl[2], bSh[2], bSl[2];
                xq_gtc_lda(aKh, Kh, KD2, r0, cc); xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                xq_gtc_lda(aKl, Kl, KD2, r0, cc); xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                const int s_row = nt * 8 + g;       // B = K^T: column s = row s of K
                bKh[0] = *(const unsigned*)(Kh + s_row * KD2 + cc); bKh[1] = *(const unsigned*)(Kh + s_row * KD2 + cc + 2);
                bKl[0] = *(const unsigned*)(Kl + s_row * KD2 + cc); bKl[1] = *(const unsigned*)(Kl + s_row * KD2 + cc + 2);
                xq_gtc_mma3(accA, aKh, aKl, true, bKh, bKl, true);
                xq_gtc_mma3(accD, aQh, aQl, true, bKh, bKl, true);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(accW, aKh, aKl, true, bSh, bSl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                // served: Af = accA * exp(..) (s < row), then the solve's t = b * Af — both roundings kept
                Tt[scol * C + row] = (scol < row) ? __fmul_rn(bt[row], __fmul_rn(accA[e], __expf(lg[row + 1] - lg[scol + 1]))) : 0.f;
                const float dv = (scol <= row && scol < n) ? accD[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                xq_gtc_st2(Dh, Dl, row * CD2 + scol, dv);
                const float v = (row < n) ? xq_h2f(Vb[row * VD2 + scol]) : 0.f;
                Wt[scol * C + row] = v - __expf(lg[row + 1]) * accW[e];
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(2);

        // ---- (1)+(2) U by a 31-step wavefront (lane = row, warp = 4 columns), U / lambda-U stored ----
        {
            const int r = lane;
            const float br = bt[r];
            float acc[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) acc[j] = __fmul_rn(br, Wt[(warp * 4 + j) * C + r]);
            #pragma unroll
            for (int k = 0; k < C - 1; k++) {
                const float tk = Tt[k * C + r];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float wk = __shfl_sync(0xFFFFFFFFu, acc[j], k);
                    if (r > k) acc[j] = __fmaf_rn(-tk, wk, acc[j]);
                }
            }
            const float lam = __expf(lg[n] - lg[r + 1]);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = warp * 4 + j;
                const float u = (r < n) ? acc[j] : 0.f;
                const float ul = u * lam;
                xq_gtc_st2(Uh, Ul, r * VD2 + c, u); xq_gtc_st2(Lh, Ll, r * VD2 + c, ul);
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(3);

        // ---- O = exp(lg[tt+1]) (Q S) + D U  (QS k = 128, then DU k = C accumulates) ----
        {
            float acc[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aQh[4], aQl[4], bSh[2], bSl[2];
                xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aQh, aQl, true, bSh, bSl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[e] *= __expf(lg[r0 + 8 * (e >= 2) + 1]);
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aDh[4], aDl[4], bUh[2], bUl[2];
                xq_gtc_lda(aDh, Dh, CD2, r0, cc);
                xq_gtc_ldb(bUh, Uh, VD2, cc, nt * 8 + g);
                xq_gtc_lda(aDl, Dl, CD2, r0, cc); xq_gtc_ldb(bUl, Ul, VD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aDh, aDl, true, bUh, bUl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                if (row < n) {
                    const int col = nt * 8 + 2 * tq + (e & 1);
                    core[(long long)(c0 + row) * (nh * VD) + head * VD + bb0 + col] = __float2half_rn(acc[e]);
                }
            }
        }
        __syncthreads();   // every Sf read (W, O) is done before the update rewrites it
        XQ_GTC_STAMP(5);

        // ---- S' = gamma_C S + K^T . Ulambda  (warp = 16 kd rows x VC; k = C); master stays f32 ----
        {
            const float gam = __expf(lg[n]);
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) Sr[j][e] *= gam;
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aTh[4], aTl[4];
                xq_gtc_ldat(aTh, Kh, KD2, warp * 16 + g, cc);
                xq_gtc_ldat(aTl, Kl, KD2, warp * 16 + g, cc);
                #pragma unroll
                for (int j = 0; j < VC / 8; j++) {
                    unsigned bLh[2], bLl[2];
                    xq_gtc_ldb(bLh, Lh, VD2, cc, j * 8 + g);
                    xq_gtc_ldb(bLl, Ll, VD2, cc, j * 8 + g);
                    xq_gtc_mma3(Sr[j], aTh, aTl, true, bLh, bLl, true);
                }
            }
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = warp * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sf[row * SD2 + col] = Sr[j][e];
                }
        }
        __syncthreads();
        XQ_GTC_STAMP(6);
    }
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sg[(long long)row * VD + col] = Sr[j][e];
        }
    XQ_GTC_PROF_END();
}
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc2(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc2_body<false>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, nullptr); }
// harness-only phase-stamped twins (never launched by a serving path)
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc_prof(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N,
    unsigned long long* __restrict__ prof) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc_body<7, true>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, prof); }
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc2_prof(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N,
    unsigned long long* __restrict__ prof) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc2_body<true>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, prof); }

// ---- full-attention KV append over a chunk: per kv-head, k-norm + partial rope
// + cache write for each chunk position (pos = pos0 + t). V is cached raw.
// Layout contract = xq_attn_decode: kp/vp rows [pos][kvh][hd], cossin rows
// [pos][cos rdim/2 | pad | sin rdim/2 | pad].
// 12-arg cap: nkv_hd = nkv|hd<<16; rdim_maxpos = rdim|max_pos<<8
// (rdim needs 8 bits: Flash-Next head_dim=256 × partial_rotary 0.25 = 64 —
// the old 6-bit field truncated 64→0 and silently disabled K rope);
// slot_pos0 = slot|pos0<<12 (slot < 4096, pos0 < 2^19).
extern "C" __global__ void xq_attn_kv_prefill(__half* __restrict__ kp, __half* __restrict__ vp,
                                              float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, int nkv_hd,
                                              int rdim_maxpos, int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    // K1: bits 8..27 = max_pos, bits 28..31 = KV format -> repacked as xq_kv_make's mpf
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int kvh = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kraw[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    for (int t = 0; t < C; t++) {
        const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
        // shared fn: bit-identical K words vs xq_attn_decode's write
        xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                       cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
        const long long p = pos0 + t;
        xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
        __syncthreads();
    }
}

// ---- PQ8 item 1 (SURPASS_PLAN WP06 item 1): the chunk KV append with one CTA per (row, kv head).
// xq_attn_kv_prefill runs grid = nkv (2 CTAs on 48 SMs) and walks the C rows serially — a latency
// chain of ~11 block barriers + 3 dependent global loads per row: 2.72 ms/call at C = 2048, 13 calls
// per chunk (12 trunk layers + the MTP head fill). An 8-bit writer lengthens every row's chain (q8:
// two 5-stage H32 butterflies, two 5-stage absmax trees and an IEEE divide per row; PROBES_14
// measured the fp8 writer at 786 vs 578 ms per 32K prefill). Here grid = C*nkv: t = bid / nkv,
// kvh = bid % nkv, blockDim = hd, and the body is xq_attn_kv_prefill's loop body VERBATIM (the
// shared inline xq_k_norm_rope — blockDim.x = hd, so the same halving tree — then xq_kv_put K and
// V), so every cache byte equals the serial kernel's for every format (GB10_PQ8_XCHECK diffs the
// rows against it; GB10_PQ8_KVW=0 / GB10_PQ8_OFF=1 keep the serial kernel). Rows are independent:
// row t reads kp/vp/cossin row t and writes cache row pos0 + t only. Same 12-arg packing.
extern "C" __global__ void xq_attn_kv_prefill_rows(__half* __restrict__ kp, __half* __restrict__ vp,
                                                   float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                                   const float* __restrict__ cossin, int nkv_hd,
                                                   int rdim_maxpos, int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int t = blockIdx.x / nkv;
    const int kvh = blockIdx.x % nkv;
    if (t >= C) return;                           // block-uniform (grid is exactly C*nkv)
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
    xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                   cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
    const long long p = pos0 + t;
    xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
}

// ---- full-attention prefill: per (head), loop the chunk's query positions;
// q-norm + rope per position, two-pass softmax over t' <= pos (cache layout as
// decode, ascending t). Output row layout = decode's attn [pos][head][hd].
// 12-arg cap: nh_nkv = nkv|nh<<16; hd_rdim = hd|rdim<<16; slot_pos0 as above.
extern "C" __global__ void xq_attn_prefill_q(__half* __restrict__ attn,
                                             const __half* __restrict__ qg,
                                             const float* __restrict__ kcvc,
                                             const __half* __restrict__ qnkn,
                                             const float* __restrict__ cossin,
                                             int nh_nkv, int hd_rdim, int max_pos,
                                             int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int hd = hd_rdim & 0xFFFF;
    const int rdim = ((unsigned)hd_rdim) >> 16;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float qraw[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const float scale = rsqrtf((float)hd);
    for (int t = 0; t < C; t++) {
        const long long qrow = (long long)t * nh * hd * 2 + (long long)head * hd * 2;
        float qv = xq_h2f(qg[qrow + tid]);
        red[tid] = qv * qv;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qnkn[tid]));
        const float* c = cossin + (long long)t * 2 * rdim;
        const float* s = c + rdim;
        qraw[tid] = qv;
        __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qraw[tid], x2 = qraw[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        } else if (tid >= rdim) {
            qbuf[tid] = qraw[tid];
        }
        __syncthreads();
        qv = xq_kv_rot(qbuf[tid], kc.fmt, tid & 31);   // WP25 q8: rotated basis
        const int p = pos0 + t;
        float acc, denom;
        xq_attn_softmax(red, qv, kc, p, tid, hd, acc, denom);
        // S-A3-f-b FIX: the sweep write was missing the per-head sigmoid gate that
        // xq_attn_decode applies — sweep rows fed later layers' GDN state ungated,
        // leaving logits rel ~0.8 vs sequential. qg rows are [values|gate] pairs
        // (stride 2*hd per head); the gate is the second half.
        float g = xq_h2f(qg[qrow + hd + tid]);
        const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
        attn[(long long)t * nh * hd + (long long)head * hd + tid] =
            xq_f2h(o * xq_sig(g));
        __syncthreads();
    }
}

// ---- S-A3-f-g: flash-style chunked prefill attention (the speed directive).
// Replaces xq_attn_prefill_q's one-block-per-head serial sweep (143 ms/layer at
// C=512) with one block per (q-tile of BT rows, head). Block 512 threads = 16
// warps, one warp per q row. K/V tiles are staged through shared memory once
// per block (padded rows kill bank conflicts) and reused by every warp; each
// warp keeps the online-softmax state (m, l, acc[4]) in registers with the
// standard rescale on max growth. Math matches the sweep kernel (fp32 accum,
// f16 out, sigmoid gate after averaging); reduction order differs — the drift
// protocol applies. KV layout: kcvc[slot][k|v][kvh][pos][hd] fp32 as written
// by xq_attn_kv_prefill. qg rows are [values|gate] pairs per head.
// S-A3-f-d night FIX: the original body hardcoded the hd=128 layout (qf[.][128],
// 4 dims/lane, KPAD=132). On Flash-Next (head_dim=256) it staged HALF the q/K/V
// dims and wrote SMEM out of bounds for e >= 132 — progressively corrupting
// attention from the second chunk position on (prefillx evidence: C=16 state
// maxd 3.69 vs 1.45 flash-off; C=32/60 first-token MISMATCH; flash off → every
// width matches). The body is now a template on the head-dim layout; hd=256
// instantiates BT=8/BKV=16 so static SMEM stays < 48 KB (qf 8K + Ks/Vs 33.3K
// + P 0.5K ≈ 41 KB). Any other head_dim falls back to the sweep kernel.
//
// PQ8 item 2 (VEC = true, xq_attn_prefill_flash256v): the same kernel with two data-movement
// changes and NO arithmetic change — Ks/Vs/qf hold the same f32 values and every score / PV / gate
// op runs in the same order, so attn is bit-identical to VEC = false (GB10_PQ8_XCHECK diffs it):
//  (a) staging: the scalar loop did, per element, a runtime-hd integer divide + modulo, a runtime
//      format branch and ONE scalar load (q8: a byte load + an f16 scale load + an unpack, 32
//      dependent iterations per thread per tile). Here a thread stages 8-dim groups through
//      xq_kv_ld8 with the format a compile-time constant (--probe-exl3-kvq: ld8 == ld bitwise on
//      every format): q8 = one 8-B code load + one scale load per 8 values, f32 = two 16-B loads;
//      every item's loads issue before any store (masked rows load a valid row and are zeroed by a
//      select: the old `ok ? ld : 0.0f`, under the scalar element owner's mask — see PQ8-v2 below),
//      then two 16-B smem stores per item;
//  (b) scores: d = fma(q[e], k[e], d) over e ascending from +0 — the chain the scalar loop compiles
//      to (fma.rn.f32 from 0f00000000) — read as float4 from qf / Ks: 3 smem wavefronts per 4 dims
//      (qf broadcast + 16 lanes x 16 B at bank 4*lane+e, conflict-free) instead of 12 (scalar Ks is
//      2-way conflicted at KPAD 260 over 16 lanes). This loop was the kernel's smem bottleneck.
//      (PQ8-v2 SASS check: ptxas already emits LDS.128 + an ascending FFMA chain for flash256's
//      scalar score loop, so (b) changes the PTX only; the served-code saving is (a).)
// PQ8-v2 FIX (the GPU XCHECK found ~48% of f16 halves differing): the math above IS identical (SASS
// of the served module: same FFMA chain from RZ, FMNMX/EX2 softmax, FFMA PV, div + sigmoid gate),
// but the v1 stager wrote a DIFFERENT shared-memory image. The scalar loop hands tile element
// (jj, e) to thread e (blockDim == hd == 256: x = tid + k*256 -> jj = k, e = tid), i.e. to warp
// e >> 5, and zeroes it unless jt + jj <= THAT warp's own q-row pmax (its per-warp jn); a warp whose
// row loop has ended stages nothing (the element keeps the previous tile's value). v1 staged rows
// w and w+8 (all dims of group `lane`) under warp w's jn — on a block's diagonal tile that differs
// from the scalar image in every q row but the block's first. v2 reproduces the scalar image
// exactly: the owner of dims grp*8..grp*8+7 is warp grp >> 2; the item is stored iff
// jt <= pbase + (grp >> 2) (owner still in its loop) and zeroed iff jt + jj > pbase + (grp >> 2);
// the body runs every warp to the block's last tile (pbase + BT - 1), each warp computing only
// while jt <= its own pmax — the scalar kernel's loop, with the early-exited warps' barriers made
// explicit. Host proof: exl3_forward.rs flash256_stage_tests (cargo test --lib flash256_stage).
// NOTE — the scalar image is NOT the causal-softmax attention: dims of earlier warps are zeroed in
// the keys/values a later q row reads on its diagonal tile (row t reads its own key with dims
// [0, 32*(t%8)) zeroed when pos0 % 8 == 0). v2 is bitwise flash256 by contract; correcting it is
// a behaviour change left to the owner: CAUSAL (xq_attn_prefill_flash256c, opt-in only via
// GB10_PQ8_FLASH=causal) stages every row <= the block's last pmax in full (pw = pbase + BT - 1
// for every item), so each warp's scores / PV see exactly the causal keys / values.
template <int FMT, int HD, int BT, int BKV, int KPAD, bool CAUSAL = false>
__device__ __forceinline__ void a3fg_stage_v(float (*Ks)[KPAD], float (*Vs)[KPAD], const XqKv& kc,
                                             int jt, int pbase, int tid) {
    constexpr int G = HD / 8;                     // 8-dim groups per row
    constexpr int NTHR = BT * 32;
    constexpr int ITEMS = BKV * 2 * G;            // (row, K|V, group) items per tile
    static_assert(ITEMS % NTHR == 0, "a3fg_stage_v: items must tile the block");
    // The scalar-image owner map (element e -> thread e) needs blockDim == HD. CAUSAL staging has
    // no owner map (every row <= the block's last pmax is staged in full): any block shape works
    // (w5/PFIX: the hd=128 twin runs 512 threads over 128 dims).
    static_assert(CAUSAL || NTHR == HD, "a3fg_stage_v: the scalar owner map (element e -> thread e) needs blockDim == HD");
    constexpr int IPT = ITEMS / NTHR;
    float o[IPT][8];
    #pragma unroll
    for (int k = 0; k < IPT; k++) {
        const int it = tid + k * NTHR;
        const int grp = it % G, jj = (it / G) % BKV, isv = it / (G * BKV);
        // pmax of the scalar owner warp of dims grp*8..+7 (CAUSAL: the block's last row's pmax)
        const int pw = CAUSAL ? pbase + BT - 1 : pbase + (grp >> 2);
        const int p = min(jt + jj, pw);           // a valid row (<= the block's last pmax)
        xq_kv_ld8(o[k], (isv ? kc.v : kc.k) + (long long)p * kc.rb, FMT, HD, grp * 8);
    }
    #pragma unroll
    for (int k = 0; k < IPT; k++) {
        const int it = tid + k * NTHR;
        const int grp = it % G, jj = (it / G) % BKV, isv = it / (G * BKV);
        const int pw = CAUSAL ? pbase + BT - 1 : pbase + (grp >> 2);
        if (jt <= pw) {                           // the owner warp is still in its row loop
            const bool ok = jt + jj <= pw;        // == the owner's `jj < jn`
            float* dst = (isv ? &Vs[jj][0] : &Ks[jj][0]) + grp * 8;
            *reinterpret_cast<float4*>(dst) = make_float4(ok ? o[k][0] : 0.0f, ok ? o[k][1] : 0.0f,
                                                          ok ? o[k][2] : 0.0f, ok ? o[k][3] : 0.0f);
            *reinterpret_cast<float4*>(dst + 4) = make_float4(ok ? o[k][4] : 0.0f, ok ? o[k][5] : 0.0f,
                                                              ok ? o[k][6] : 0.0f, ok ? o[k][7] : 0.0f);
        }
    }
}
// VEC: 0 = scalar staging (flash / flash256), 1 = the flash256v twin, 2 = CAUSAL staging (flash256c,
// flash128c).
template <int HD, int BT, int BKV, int KPAD, int PL, int VEC = 0>
__device__ __forceinline__ void a3fg_flash_body(
        __half* __restrict__ attn, const __half* __restrict__ qg,
        const float* __restrict__ kcvc, const __half* __restrict__ qnkn,
        const float* __restrict__ cossin, int nh_nkv, int hd_rdim, int max_pos,
        int slot_pos0, int C, float eps) {
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int hd = hd_rdim & 0xFFFF;
    const int rdim = ((unsigned)hd_rdim) >> 16;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int head = blockIdx.y;
    const int t0 = blockIdx.x * BT;
    const int warp = threadIdx.x >> 5;   // BT warps (blockDim BT*32)
    const int lane = threadIdx.x & 31;
    const int r = warp;                  // q row within the tile
    const int t = t0 + r;                // chunk token index
    const int kvh = head / (nh / nkv);
    const float scale = rsqrtf((float)hd);

    // PQ8 (VEC): 16-B aligned for the float4 staging stores / score reads (rows are 16-B
    // multiples: HD*4 and KPAD*4 = 1040); VEC = false keeps the pre-PQ8 4-B alignment.
    __shared__ alignas(VEC ? 16 : 4) float qf[BT][HD];        // normalized+roped q rows
    __shared__ alignas(VEC ? 16 : 4) float Ks[BKV][KPAD]; // staged K tile
    __shared__ alignas(VEC ? 16 : 4) float Vs[BKV][KPAD]; // staged V tile
    __shared__ float P[BT][BKV];    // exp scores of current tile

    const long long qrow = (long long)t * nh * hd * 2 + (long long)head * hd * 2;

    // ---- normalize + rope this row's q. HD dims over 32 lanes = PL/lane.
    float qv[PL], qw[PL];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < PL; i++) {
        int d = lane + 32 * i;
        qv[i] = xq_h2f(qg[qrow + d]);
        ss += qv[i] * qv[i];
        qw[i] = 1.0f + xq_h2f(qnkn[d]);
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float inv = rsqrtf(ss / (float)hd + eps);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < PL; i++) qf[r][lane + 32 * i] = qv[i] * inv * qw[i];
    __syncwarp();
    // rope: x1 at d (< rdim/2), partner x2 at d + rdim/2 — SMEM read, warp-local.
#pragma unroll
    for (int i = 0; i < PL; i++) {
        int d = lane + 32 * i;
        if (d < rdim / 2) {
            float x1 = qf[r][d], x2 = qf[r][d + rdim / 2];
            const float* c = cossin + (long long)t * 2 * rdim;
            const float* s = c + rdim;
            qf[r][d] = x1 * c[d] - x2 * s[d];
            qf[r][d + rdim / 2] = x2 * c[d] + x1 * s[d];
        }
        // dims >= rdim: already in place (pass-through).
    }
    __syncwarp();

    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    if (kc.fmt >= XQ_KV_Q8) {   // WP25 q8: this warp's q row into the cache's rotated basis
#pragma unroll
        for (int i = 0; i < PL; i++) qf[r][lane + 32 * i] = xq_h32r(qf[r][lane + 32 * i], lane);
        __syncwarp();
    }
    const int pmax = pos0 + t; // last visible position (causal)

    float m = -INFINITY, l = 0.0f, acc[PL];
#pragma unroll
    for (int i = 0; i < PL; i++) acc[i] = 0.0f;

    const int tid = threadIdx.x;
    // PQ8-v2 (VEC): every warp walks to the block's last tile (its last row's pmax) so each scalar
    // owner's elements get staged (a3fg_stage_v); a warp computes only while jt <= its own pmax.
    const int pbase = pos0 + t0;
    const int pend = VEC ? pbase + BT - 1 : pmax;

    for (int jt = 0; jt <= pend; jt += BKV) {
        const int jn = min(BKV, pmax + 1 - jt);
        // ---- cooperative stage of K/V tile (whole block; padded SMEM rows)
        __syncthreads(); // previous tile's consumers done before overwrite
        if constexpr (VEC != 0) {
            // PQ8 (a): 8-dim groups, format a compile-time constant (hd == HD: host dispatch)
            switch (kc.fmt) {
                case XQ_KV_F32: a3fg_stage_v<XQ_KV_F32, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                case XQ_KV_F16: a3fg_stage_v<XQ_KV_F16, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                case XQ_KV_FP8: a3fg_stage_v<XQ_KV_FP8, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                default:        a3fg_stage_v<XQ_KV_Q8, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
            }
        } else {
        for (int x = tid; x < BKV * hd; x += blockDim.x) {
            int jj = x / hd, e = x % hd;
            bool ok = jj < jn;
            Ks[jj][e] = ok ? xq_kv_k(kc, jt + jj, e) : 0.0f;
            Vs[jj][e] = ok ? xq_kv_v(kc, jt + jj, e) : 0.0f;
        }
        }
        __syncthreads();
        if (VEC && jt > pmax) continue;   // warp-uniform: this row's loop is over (scalar: warp left it)
        // ---- scores: lane j owns position jt+j (serial fp32 dot over SMEM)
        float s = -INFINITY;
        if (lane < jn) {
            float d = 0.0f;
            if constexpr (VEC != 0) {
                // PQ8 (b): the same fma chain from +0, e ascending, read 4 dims at a time
                const float4* q4 = reinterpret_cast<const float4*>(&qf[r][0]);
                const float4* k4 = reinterpret_cast<const float4*>(&Ks[lane][0]);
                #pragma unroll 16
                for (int e4 = 0; e4 < HD / 4; e4++) {
                    const float4 a = q4[e4], b = k4[e4];
                    d = __fmaf_rn(a.x, b.x, d);
                    d = __fmaf_rn(a.y, b.y, d);
                    d = __fmaf_rn(a.z, b.z, d);
                    d = __fmaf_rn(a.w, b.w, d);
                }
            } else {
            for (int e = 0; e < hd; e++) d += qf[r][e] * Ks[lane][e];
            }
            s = d * scale;
        }
        float mtile = s;
        for (int off = 16; off > 0; off >>= 1)
            mtile = fmaxf(mtile, __shfl_down_sync(0xffffffff, mtile, off));
        mtile = __shfl_sync(0xffffffff, mtile, 0);
        // ---- online-softmax state update with rescale (registers per warp)
        float mnew = fmaxf(m, mtile);
        float factor = __expf(m - mnew); // first tile: exp(-inf)=0 → clean reset
        m = mnew;
        l *= factor;
#pragma unroll
        for (int i = 0; i < PL; i++) acc[i] *= factor;
        float p = (lane < jn) ? __expf(s - m) : 0.0f;
        if (lane < BKV) P[r][lane] = p; // BKV < 32: keep idle lanes in-bounds
        float lp = p;
        for (int off = 16; off > 0; off >>= 1) lp += __shfl_down_sync(0xffffffff, lp, off);
        l += __shfl_sync(0xffffffff, lp, 0);
        __syncwarp();
        // ---- accumulate V: each lane strides the tile's rows for its PL dims
        for (int jj = 0; jj < jn; jj++) {
            float pj = P[r][jj];
#pragma unroll
            for (int i = 0; i < PL; i++) acc[i] += pj * Vs[jj][lane + 32 * i];
        }
        __syncwarp();
    }
    if (t < C) {
#pragma unroll
        for (int i = 0; i < PL; i++) {
            int e = lane + 32 * i;
            float g = xq_h2f(qg[qrow + hd + e]);
            const float o = xq_kv_rot(acc[i] / l, kc.fmt, lane);
            attn[(long long)t * nh * hd + (long long)head * hd + e] =
                xq_f2h(o * xq_sig(g));
        }
    }
}

extern "C" __global__ void xq_attn_prefill_flash(__half* __restrict__ attn,
                                                 const __half* __restrict__ qg,
                                                 const float* __restrict__ kcvc,
                                                 const __half* __restrict__ qnkn,
                                                 const float* __restrict__ cossin,
                                                 int nh_nkv, int hd_rdim, int max_pos,
                                                 int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<128, 16, 32, 132, 4>(attn, qg, kcvc, qnkn, cossin,
                                         nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// hd=256 shape (Flash-Next): BT=8 rows/block, BKV=16 kv/tile — SMEM ≈ 41 KB.
extern "C" __global__ void xq_attn_prefill_flash256(__half* __restrict__ attn,
                                                    const __half* __restrict__ qg,
                                                    const float* __restrict__ kcvc,
                                                    const __half* __restrict__ qnkn,
                                                    const float* __restrict__ cossin,
                                                    int nh_nkv, int hd_rdim, int max_pos,
                                                    int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8>(attn, qg, kcvc, qnkn, cossin,
                                        nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// PQ8 item 2: the bit-identical twin of xq_attn_prefill_flash256 (vector staging + float4 scores;
// see a3fg_stage_v — PQ8-v2 stages the scalar kernel's exact smem image). Host: hd == 256 and
// blockDim 256 only (the body indexes by HD), GB10_PQ8_FLASH=0 keeps the above.
extern "C" __global__ void xq_attn_prefill_flash256v(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8, 1>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// PQ8-v2 OPT-IN (GB10_PQ8_FLASH=causal; NOT bitwise flash256 — a behaviour change): the flash256v
// body with CAUSAL staging — every key/value row a q row of the block may read is staged in full,
// so the diagonal tile no longer loses the dims of earlier warps (see the note above a3fg_stage_v).
extern "C" __global__ void xq_attn_prefill_flash256c(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8, 2>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// w5/PFIX OPT-IN (the SAME GB10_PQ8_FLASH=causal switch; output-changing): the hd=128 causal twin.
// xq_attn_prefill_flash (the hd=128 shape: BT=16 / BKV=32, blockDim 512) has flash256's diagonal-tile
// defect under a different owner map: blockDim (512) != hd (128), so its scalar stager hands tile
// element (jj, e) to thread (jj % 4) * 128 + e, i.e. warp (jj % 4) * 4 + (e >> 5), and zeroes it
// (or, once that warp's row loop has ended, leaves the previous tile's value) against THAT warp's
// pmax — whole dim slices of the diagonal-tile rows a later q row reads are wrong. Host proof:
// exl3_forward.rs flash256_stage_tests::flash128c_stage_is_causal (cargo test --lib flash256_stage).
// xq_attn_prefill_flash has had NO host caller since 2463afc (S-A3-f-d night): hd=128 dense prefill
// rows run the sweep xq_attn_prefill_q, and that default is unchanged. With GB10_PQ8_FLASH=causal the
// hd=128 dense rows (t >= 16) take this kernel instead: the flash256c body (vector staging with
// block-wide causal pw = pbase + BT - 1, float4 fma score chain) at the hd=128 shape. SMEM: qf 8 KB
// + Ks/Vs 2 x 16.5 KB + P 2 KB = ~43 KB static (< 48 KB).
extern "C" __global__ void xq_attn_prefill_flash128c(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<128, 16, 32, 132, 4, 2>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// ---- PLE conv over a chunk: 9-deep ring in registers (dilated taps at
// offsets 0/3/6 of the ring + cur), conv then ring-shift per token; the ring
// is stored back once at the end. Matches xq_ple_conv + xq_ple_state_shift.
extern "C" __global__ void xq_ple_conv_chunk(float* __restrict__ resid, const __half* __restrict__ gated,
                                             const __half* __restrict__ normed, __half* __restrict__ state,
                                             const __half* __restrict__ conv_w, int slot, int C) {
    XQ_PDL_ENTRY();
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half st[9];
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    const float w0 = xq_h2f(conv_w[d * 4 + 0]);
    const float w1 = xq_h2f(conv_w[d * 4 + 1]);
    const float w2 = xq_h2f(conv_w[d * 4 + 2]);
    const float w3 = xq_h2f(conv_w[d * 4 + 3]);
    for (int t = 0; t < C; t++) {
        const long long i = (long long)t * 10240 + d;
        const float cur = xq_h2f(normed[i]);
        const float acc = w0 * xq_h2f(st[0]) + w1 * xq_h2f(st[3])
                        + w2 * xq_h2f(st[6]) + w3 * cur;
        resid[i] += xq_h2f(gated[i]) + xq_silu(acc);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// ---- compact per-expert MoE prefill. Compact rows R = C*topk grouped by
// expert (ascending expert id, ascending token within expert — deterministic);
// row_tok[i] = source x row, row_eidx[i] = selected-expert slot (idxmap index).
extern "C" __global__ void xq_had_suh_rows(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                           const int* __restrict__ idxmap, const int* __restrict__ row_tok,
                                           const int* __restrict__ row_eidx,
                                           __half* __restrict__ xh, int R, int K) {
    XQ_PDL_ENTRY();
    int nblk = K >> 7;
    int blk = blockIdx.x % nblk;
    long long i = blockIdx.x / nblk;
    if ((int)i >= R) return;
    int lane = threadIdx.x & 31;
    const int m = row_tok[i];
    const __half* sr = suh_all + (long long)idxmap[row_eidx[i]] * K + (long long)blk * 128 + lane * 4;
    long long base = (long long)i * K + (long long)blk * 128 + lane * 4;
    long long base_in = (long long)m * K + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(x[base_in + r]) * xq_h2f(sr[r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base + r] = xq_f2h(v[r]);
}

// TP-4M1 (PLAN/TP-4M1_REPORT.md): xq_had_suh_rows_live_u{1,2,4} (U = items in flight per warp; served = u2) = xq_had_suh_rows with the EP dead rows skipped, vector
// loads/stores, and a persistent grid-stride loop. PER-ELEMENT ARITHMETIC IS THE SAME CODE: v = h2f(x) * h2f(suh)
// (f32 product, exact), exl3_had128 (the same butterfly, 4 elements per lane, 32 lanes = one 128-block), ONE f16
// RN store. Only memory behaviour differs: x / suh are read as one 8-byte uint2 per lane (4 halves), the result is
// stored as one uint2, a warp walks (row, block) items in the old blockIdx order with U items in flight.
// DEAD ROWS: under EP the compact row list holds live_end = offs_row[n_local] real rows; rows [live_end, R) were
// given a valid dummy mapping by xq_moe_pf_ep_fill purely so this kernel would not read an uninitialised index, and
// their output is never read (every consumer reads xh through the tile lists: M = cnt[e], rows >= cnt are
// zero-filled by the cp.async staging, and every tile lies below offs_row[n_local]). The live-row kernel therefore
// does not write them. live_end == nullptr processes all R rows (the old kernel's coverage).
// Requires K % 128 == 0, 8-byte aligned x / suh_all / xh bases (cudaMalloc'd slices).
template <int XQ_HSL_U>
__device__ __forceinline__ void xq_had_suh_rows_live_body(
        const __half* __restrict__ x, const __half* __restrict__ suh_all,
        const int* __restrict__ idxmap, const int* __restrict__ row_tok, const int* __restrict__ row_eidx,
        __half* __restrict__ xh, const int* __restrict__ live_end, int R, int K) {
    const int nblk = K >> 7;
    int lim = R;
    if (live_end != nullptr) { const int le = *live_end; lim = le < R ? le : R; if (lim < 0) lim = 0; }
    const long long nitem = (long long)lim * nblk;
    const int lane = threadIdx.x & 31;
    const long long gw = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const long long stride = (long long)gridDim.x * 8;
    for (long long it0 = gw; it0 < nitem; it0 += stride * XQ_HSL_U) {
        uint2 xv[XQ_HSL_U], sv[XQ_HSL_U];
        long long ob[XQ_HSL_U];
        #pragma unroll
        for (int u = 0; u < XQ_HSL_U; ++u) {
            const long long it = it0 + (long long)u * stride;
            ob[u] = -1;
            if (it < nitem) {
                const int i = (int)it / nblk;            // nitem = R * (K/128) < 2^31 (R <= ~100M rows of K = 2560)
                const int blk = (int)it - i * nblk;
                const int m = row_tok[i];
                const long long col = (long long)blk * 128 + lane * 4;
                const int e = idxmap[row_eidx[i]];
                xv[u] = *reinterpret_cast<const uint2*>(x + (long long)m * K + col);
                sv[u] = *reinterpret_cast<const uint2*>(suh_all + (long long)e * K + col);
                ob[u] = (long long)i * K + col;
            }
        }
        #pragma unroll
        for (int u = 0; u < XQ_HSL_U; ++u) {
            if (ob[u] >= 0) {
                const float2 xa = __half22float2(*reinterpret_cast<const __half2*>(&xv[u].x));
                const float2 xb = __half22float2(*reinterpret_cast<const __half2*>(&xv[u].y));
                const float2 sa = __half22float2(*reinterpret_cast<const __half2*>(&sv[u].x));
                const float2 sb = __half22float2(*reinterpret_cast<const __half2*>(&sv[u].y));
                float v[4];
                v[0] = xa.x * sa.x; v[1] = xa.y * sa.y; v[2] = xb.x * sb.x; v[3] = xb.y * sb.y;
                exl3_had128(v, lane);
                const __half2 o01 = __floats2half2_rn(v[0], v[1]);
                const __half2 o23 = __floats2half2_rn(v[2], v[3]);
                uint2 ov;
                ov.x = *reinterpret_cast<const unsigned*>(&o01);
                ov.y = *reinterpret_cast<const unsigned*>(&o23);
                *reinterpret_cast<uint2*>(xh + ob[u]) = ov;
            }
        }
    }
}
#define XQ_HSL_ENTRY(U) \
extern "C" __global__ void __launch_bounds__(256) xq_had_suh_rows_live_u##U( \
        const __half* __restrict__ x, const __half* __restrict__ suh_all, \
        const int* __restrict__ idxmap, const int* __restrict__ row_tok, const int* __restrict__ row_eidx, \
        __half* __restrict__ xh, const int* __restrict__ live_end, int R, int K) { \
    XQ_PDL_ENTRY(); \
    xq_had_suh_rows_live_body<U>(x, suh_all, idxmap, row_tok, row_eidx, xh, live_end, R, K); \
}
XQ_HSL_ENTRY(1)
XQ_HSL_ENTRY(2)
XQ_HSL_ENTRY(4)

extern "C" __global__ void xq_had_svh_rows(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                           const int* __restrict__ idxmap, const int* __restrict__ row_eidx,
                                           __half* __restrict__ y, int R, int N) {
    XQ_PDL_ENTRY();
    int nblk = N >> 7;
    int blk = blockIdx.x % nblk;
    long long i = blockIdx.x / nblk;
    if ((int)i >= R) return;
    int lane = threadIdx.x & 31;
    long long base = (long long)i * N + (long long)blk * 128 + lane * 4;
    const __half* sr = svh_all + (long long)idxmap[row_eidx[i]] * N + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(yraw[base + r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) y[base + r] = xq_f2h(v[r] * xq_h2f(sr[r]));
}

// ===========================================================================
// A5-P1 (prefill MoE routed experts; PLAN/notes_2026-09-27/P1.md).
// The served exl3_hmma_gemm_wide_tiles ran at 34% of its trellis BYTE floor and 18% of its TC
// floor (LEDGER_PREFILL_P7C §2). Its SASS (the served PTX through ptxas sm_121) shows why:
//   * ONE trellis LDG.E.CONSTANT per warp per k16 step, consumed by the SHFL decode a few
//     instructions later (the s loop unrolls by 2, no load runs ahead of its own step);
//   * A staged synchronously per KS block (LDG -> STS -> BAR, then compute);
//   * 128 registers for EVERY MT (the MT = 8 instance sets the kernel's allocation) and
//     34,816 B static smem => 2 CTAs = 16 warps per SM, i.e. <= 16 x 96 B = 1.5 KB of trellis
//     in flight per SM against the ~4 KB Little's-law need at 240 GB/s (~800 ns DRAM latency).
// xq_moe_pf_body = the SAME per-warp work (same ring words, same shfl/funnel/dp4a/hfma decode,
// same A fragments, same K-ascending m16n8k16 chain per output element, same f16 RN store =>
// bitwise == wide_tiles for every row and every MT) re-scheduled:
//   (i)   one entry per MT (registers sized per MT; __launch_bounds__ 2 CTAs/SM);
//   (ii)  dynamic smem sized per MT, A double-buffered by cp.async.cg (src-size 0 zero-fills
//         the rows >= M / cols >= K exactly as the old staging did);
//   (iii) the NEXT stage's KS trellis words issued into registers right after the stage barrier,
//         so each warp keeps KS (8) block loads in flight while it decodes the current stage;
//         one barrier per stage (was two).
// EPI 1 (knob prefill.moe_glue_fold): + xq_had_svh_rows fused in the epilogue (one CTA = one
//   128-column Hadamard block; the f16 RN y_raw tile goes through smem, then exactly the
//   per-lane had128 * svh sequence of xq_had_svh_rows) => writes y (no y_raw round trip).
// EPI 2 (gate/up, 512 threads): warps 0-7 = gate N-tile t, warps 8-15 = up N-tile t + N/256
//   sharing ONE staged A; epilogue = xq_had_svh_rows (gate and up) -> xq_moe_gate_mul ->
//   xq_had_suh_rows (down input, pinned wp20_suh_had: x*s of f16 values is exact in f32, so
//   every contraction is value-identical) on 128-block t => writes xhd directly.
// KS = 8 k16 steps per stage (4 at MT = 8: keeps the double buffer under 48 KB); KS only
// regroups the staging, the per-element mma order is unchanged.
// ===========================================================================
__device__ __forceinline__ void xq_pf_cp16(void* sdst, const void* gsrc, bool ok) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(sdst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(gsrc), "r"(ok ? 16 : 0) : "memory");
}

template <int MT> struct XqPfKs { static constexpr int v = (MT == 8) ? 4 : 8; };

// halves of dynamic smem an (MT, NW) instance needs (mirrored by moe_pf_smem_bytes() host-side)
template <int MT, int NW>
struct XqPfSmem {
    static constexpr int ROWS = MT * 16;
    static constexpr int LD = XqPfKs<MT>::v * 16 + 8;
    static constexpr int A = 2 * ROWS * LD;
    static constexpr int PR = ROWS < 64 ? ROWS : 64;
    static constexpr int T = (NW / 8) * PR * 136;
    static constexpr int halves = A > T ? A : T;
};

// The lane-constant ring-decode tables (shuffle source lanes + funnel shift of each of the lane's
// 8 decoded values). Used by xq_moe_pf2_body (A5-L2): textual copies of the served xq_moe_pf_body code,
// which stays inline so its SASS is byte-identical to A5-P1 (cuobjdump-checked).
template <int BITS>
__device__ __forceinline__ void xq_pf_tables(int lane, int (&s_i0m)[8], int (&s_i1m)[8], int (&s_sf)[8]) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
}

// One k16 step of one warp: decode its NH 16x16 trellis blocks from the lane's ring words w0[]
// (== exl3_gemm_body_wide BITS <= 4) and run the MT m16n8k16 chains on the staged A columns
// [cs, cs + 16) of `cur` (row stride LD); NH = 2 (A5-L2 v2, paired gate|up warps) shares one set
// of A fragments between the two blocks. Per output element: the served decode + the same
// K-ascending mma chain (NH = 1 is the served body's inline step).
template <int MT, int LD, int NH>
__device__ __forceinline__ void xq_pf_step(const unsigned (&w0)[NH], const __half* __restrict__ cur, const int cs,
        const int gid, const int tig, const int (&s_i0m)[8], const int (&s_i1m)[8], const int (&s_sf)[8],
        float (&acc)[NH][MT][2][4]) {
    uint32_t bfrag[NH][2][2];
    #pragma unroll
    for (int hh = 0; hh < NH; ++hh) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint16_t rv[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int jj = h * 4 + q;
                const unsigned lo = __shfl_sync(0xFFFFFFFFu, w0[hh], s_i1m[jj] & 31);
                const unsigned hi = __shfl_sync(0xFFFFFFFFu, w0[hh], s_i0m[jj] & 31);
                const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[jj]) & 0xFFFFu;
                const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                            __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
            }
            bfrag[hh][h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
            bfrag[hh][h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const __half* sm = cur + (mt * 16) * LD;
        const uint32_t a0 = *(const uint32_t*)&sm[gid * LD + cs + 2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sm[(gid + 8) * LD + cs + 2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sm[gid * LD + cs + 2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sm[(gid + 8) * LD + cs + 2 * tig + 8];
        #pragma unroll
        for (int hh = 0; hh < NH; ++hh) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[hh][mt][h][0]), "+f"(acc[hh][mt][h][1]),
                      "+f"(acc[hh][mt][h][2]), "+f"(acc[hh][mt][h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[hh][h][0]), "r"(bfrag[hh][h][1]));
            }
        }
    }
}

// The epilogues (EPI 0 raw y_raw, 1 + svh, 2 gate|up fold) == the served body's inline code. Opens with a
// block barrier per pass (EPI >= 1) before it reuses smem from offset 0.
template <int MT, int NW, int EPI, int NH>
__device__ __forceinline__ void xq_pf_epilogue(const float (&acc)[NH][MT][2][4], __half* __restrict__ y,
        __half* __restrict__ smem, const int M, const int N, const int cta_tile, const int cta_n,
        const int row_base, const int warp, const int lane, const int wl, const int hsel,
        const int gid, const int tig, const __half* __restrict__ svh_e, const __half* __restrict__ suh_de) {
    constexpr int ROWS = MT * 16;
    if constexpr (EPI == 0) {
        // ---- == exl3_gemm_body_wide's epilogue: y_raw (fp16 RN), rows >= M skipped ----
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row0 = row_base + mt * 16 + gid;
            const int row1 = row0 + 8;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = cta_n + 16 * wl + 8 * h + 2 * tig;
                if (row0 < M) *(__half2*)&y[(size_t)row0 * N + col] = __floats2half2_rn(acc[0][mt][h][0], acc[0][mt][h][1]);
                if (row1 < M) *(__half2*)&y[(size_t)row1 * N + col] = __floats2half2_rn(acc[0][mt][h][2], acc[0][mt][h][3]);
            }
        }
    } else {
        // ---- fused Hadamard epilogues: the f16 RN y_raw tile through smem, <= 64 rows a pass ----
        constexpr int PR = ROWS < 64 ? ROWS : 64;
        constexpr int NP = ROWS / PR;
        constexpr int LT = 136;
        const int nrow = M - row_base;             // live rows of this super-tile (may exceed ROWS)
        #pragma unroll
        for (int p = 0; p < NP; ++p) {
            __syncthreads();                       // A buffers / previous pass fully consumed
            #pragma unroll
            for (int hh = 0; hh < NH; ++hh) {
                const int hs = (NH == 2) ? hh : hsel;   // paired warps (A5-L2 v2) hold both halves
                #pragma unroll
                for (int mq = 0; mq < PR / 16; ++mq) {
                    const int mt = p * (PR / 16) + mq;
                    const int r0 = mq * 16 + gid;
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = 16 * wl + 8 * h + 2 * tig;
                        *(__half2*)&smem[(hs * PR + r0) * LT + col] = __floats2half2_rn(acc[hh][mt][h][0], acc[hh][mt][h][1]);
                        *(__half2*)&smem[(hs * PR + r0 + 8) * LT + col] = __floats2half2_rn(acc[hh][mt][h][2], acc[hh][mt][h][3]);
                    }
                }
            }
            __syncthreads();
            for (int r = warp; r < PR; r += NW) {
                const int grow = row_base + p * PR + r;
                if (p * PR + r >= nrow) break;     // rows ascend with r: warp-uniform exit
                if constexpr (EPI == 1) {
                    // == xq_had_svh_rows on (row grow, block cta_n / 128)
                    float v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) v[q] = xq_h2f(smem[r * LT + lane * 4 + q]);
                    exl3_had128(v, lane);
                    const __half* sr = svh_e + cta_n + lane * 4;
                    __half* yo = y + (size_t)grow * N + cta_n + lane * 4;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) yo[q] = xq_f2h(v[q] * xq_h2f(sr[q]));
                } else {
                    // gate/up block t: xq_had_svh_rows (gate cols t*128.., up cols N/2 + t*128..)
                    // -> xq_moe_gate_mul -> xq_had_suh_rows (down suh, block t) => xhd row grow
                    const int nh = N >> 1;
                    const int c0 = cta_tile * 128 + lane * 4;
                    float g[4], u[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        g[q] = xq_h2f(smem[r * LT + lane * 4 + q]);
                        u[q] = xq_h2f(smem[(PR + r) * LT + lane * 4 + q]);
                    }
                    exl3_had128(g, lane);
                    exl3_had128(u, lane);
                    float xin[4], sd[4], v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const __half gh = xq_f2h(g[q] * xq_h2f(svh_e[c0 + q]));
                        const __half uh = xq_f2h(u[q] * xq_h2f(svh_e[nh + c0 + q]));
                        const __half dh = xq_f2h(xq_silu(xq_h2f(gh)) * xq_h2f(uh));
                        xin[q] = xq_h2f(dh);
                        sd[q] = xq_h2f(suh_de[c0 + q]);
                    }
                    wp20_suh_had(xin, sd, v, lane);
                    __half* yo = y + (size_t)grow * nh + c0;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) yo[q] = xq_f2h(v[q]);
                }
            }
        }
    }
}

// A5-L2 `_prof` twins: thread 0 of every live CTA adds (stage-top wait cycles, compute cycles,
// epilogue cycles, CTA span, 1) into g_xq_pf_prof[slot * 5 ..]; PROF = slot + 1 (0 = compiled out).
// Read + cleared by xq_moe_pf_prof_take (harness only).
__device__ unsigned long long g_xq_pf_prof[24 * 5];
extern "C" __global__ void xq_moe_pf_prof_take(unsigned long long* out) {
    const int i = threadIdx.x;
    if (i < 24 * 5) { out[i] = g_xq_pf_prof[i]; g_xq_pf_prof[i] = 0ull; }
}
template <int PROF>
__device__ __forceinline__ void xq_pf_prof_put(long long w, long long c, long long e, long long span) {
    if constexpr (PROF > 0) {
        if (threadIdx.x == 0) {
            unsigned long long* p = g_xq_pf_prof + (PROF - 1) * 5;
            atomicAdd(p + 0, (unsigned long long)w); atomicAdd(p + 1, (unsigned long long)c);
            atomicAdd(p + 2, (unsigned long long)e); atomicAdd(p + 3, (unsigned long long)span);
            atomicAdd(p + 4, 1ull);
        }
    }
}
#define XQ_PF_CLK(v) do { if constexpr (PROF > 0) (v) = clock64(); } while (0)

template <int BITS, int MT, int NW, int EPI, int PROF>
__device__ __forceinline__ void xq_moe_pf_body(
        const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
        __half* __restrict__ y, int M, int K, int N, int cta_tile, int m_sup,
        __half* __restrict__ smem, const __half* __restrict__ svh_e,
        const __half* __restrict__ suh_de) {
    static_assert(BITS <= 4, "the one-load-per-lane ring decode needs 8*BITS <= 32");
    constexpr int KS = XqPfKs<MT>::v;
    constexpr int ROWS = MT * 16;
    constexpr int LD = KS * 16 + 8;
    constexpr int C8 = KS * 2;                 // 16-B chunks per staged row
    constexpr int NCH = ROWS * C8;             // 16-B chunks per stage
    constexpr int NT = NW * 32;
    constexpr int WW = 8 * BITS;
    long long pt0 = 0, pta = 0, ptb = 0, pw = 0, pc = 0;
    XQ_PF_CLK(pt0);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wl = warp & 7;
    const int hsel = (NW == 16) ? (warp >> 3) : 0;   // EPI 2: 0 = gate tile, 1 = up tile
    const int cta_n = (cta_tile + hsel * (N >> 8)) * 128;
    const int nb_col = cta_n / 16 + wl;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int row_base = m_sup * ROWS;

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;

    // lane's ring word of k16 block kb: word (kb * nb + nb_col) * 8*BITS + lane (lane < WW)
    const uint32_t* ringl = (const uint32_t*)trellis + (size_t)nb_col * (8 * BITS) + lane;
    const size_t kstr = (size_t)nb * (8 * BITS);
    const int NS = (KB + KS - 1) / KS;
    __half* buf0 = smem;
    __half* buf1 = smem + ROWS * LD;

    uint32_t wn[KS];
    // stage j -> dst: A[(ROWS) x (KS*16)] by cp.async (zero-fill outside [M) x [K))
    #define XQ_PF_STAGE(J, DST) do { \
        const int kc0_ = (J) * (KS * 16); \
        _Pragma("unroll") \
        for (int q_ = 0; q_ < (NCH + NT - 1) / NT; ++q_) { \
            const int i_ = tid + q_ * NT; \
            if ((NCH % NT) == 0 || i_ < NCH) { \
                const int row_ = i_ / C8, c8_ = i_ - row_ * C8; \
                const int gr_ = row_base + row_, col_ = kc0_ + c8_ * 8; \
                const bool ok_ = gr_ < M && col_ < K; \
                xq_pf_cp16((DST) + row_ * LD + c8_ * 8, \
                           ok_ ? (const void*)(xh + (size_t)gr_ * K + col_) : (const void*)xh, ok_); \
            } \
        } \
        asm volatile("cp.async.commit_group;\n" ::: "memory"); \
    } while (0)
    #define XQ_PF_LOADW(J) do { \
        _Pragma("unroll") \
        for (int s_ = 0; s_ < KS; ++s_) { \
            const int kb_ = (J) * KS + s_; \
            wn[s_] = (lane < WW && kb_ < KB) ? __ldg(ringl + (size_t)kb_ * kstr) : 0u; \
        } \
    } while (0)

    XQ_PF_STAGE(0, buf0);
    XQ_PF_LOADW(0);
    const int gid = lane >> 2, tig = lane & 3;
    for (int j = 0; j < NS; ++j) {
        uint32_t wc[KS];
        #pragma unroll
        for (int s = 0; s < KS; ++s) wc[s] = wn[s];
        XQ_PF_CLK(pta);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();                       // stage j visible; everyone is past stage j-1
        XQ_PF_CLK(ptb);
        const __half* cur = (j & 1) ? buf1 : buf0;
        if (j + 1 < NS) {
            XQ_PF_STAGE(j + 1, (j & 1) ? buf0 : buf1);
            XQ_PF_LOADW(j + 1);
        }
        #pragma unroll
        for (int s = 0; s < KS; ++s) {
            if (j * KS + s < KB) {
                // ---- decode this warp's 16x16 block (== exl3_gemm_body_wide BITS <= 4) ----
                const unsigned w0 = wc[s];
                uint32_t bfrag[2][2];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint16_t rv[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int jj = h * 4 + q;
                        const unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[jj] & 31);
                        const unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[jj] & 31);
                        const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[jj]) & 0xFFFFu;
                        const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                        rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                    __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                    }
                    bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                    bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
                }
                const int cs = s * 16;
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const __half* sm = cur + (mt * 16) * LD;
                    const uint32_t a0 = *(const uint32_t*)&sm[gid * LD + cs + 2 * tig];
                    const uint32_t a1 = *(const uint32_t*)&sm[(gid + 8) * LD + cs + 2 * tig];
                    const uint32_t a2 = *(const uint32_t*)&sm[gid * LD + cs + 2 * tig + 8];
                    const uint32_t a3 = *(const uint32_t*)&sm[(gid + 8) * LD + cs + 2 * tig + 8];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(acc[mt][h][0]), "+f"(acc[mt][h][1]),
                              "+f"(acc[mt][h][2]), "+f"(acc[mt][h][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                    }
                }
            }
        }
        if constexpr (PROF > 0) { const long long ptc = clock64(); pw += ptb - pta; pc += ptc - ptb; }
    }
    #undef XQ_PF_STAGE
    #undef XQ_PF_LOADW
    XQ_PF_CLK(pta);

    if constexpr (EPI == 0) {
        // ---- == exl3_gemm_body_wide's epilogue: y_raw (fp16 RN), rows >= M skipped ----
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row0 = row_base + mt * 16 + gid;
            const int row1 = row0 + 8;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = cta_n + 16 * wl + 8 * h + 2 * tig;
                if (row0 < M) *(__half2*)&y[(size_t)row0 * N + col] = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
                if (row1 < M) *(__half2*)&y[(size_t)row1 * N + col] = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
            }
        }
    } else {
        // ---- fused Hadamard epilogues: the f16 RN y_raw tile through smem, <= 64 rows a pass ----
        constexpr int PR = ROWS < 64 ? ROWS : 64;
        constexpr int NP = ROWS / PR;
        constexpr int LT = 136;
        const int nrow = M - row_base;             // live rows of this super-tile (may exceed ROWS)
        #pragma unroll
        for (int p = 0; p < NP; ++p) {
            __syncthreads();                       // A buffers / previous pass fully consumed
            #pragma unroll
            for (int mq = 0; mq < PR / 16; ++mq) {
                const int mt = p * (PR / 16) + mq;
                const int r0 = mq * 16 + gid;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = 16 * wl + 8 * h + 2 * tig;
                    *(__half2*)&smem[(hsel * PR + r0) * LT + col] = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
                    *(__half2*)&smem[(hsel * PR + r0 + 8) * LT + col] = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
                }
            }
            __syncthreads();
            for (int r = warp; r < PR; r += NW) {
                const int grow = row_base + p * PR + r;
                if (p * PR + r >= nrow) break;     // rows ascend with r: warp-uniform exit
                if constexpr (EPI == 1) {
                    // == xq_had_svh_rows on (row grow, block cta_n / 128)
                    float v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) v[q] = xq_h2f(smem[r * LT + lane * 4 + q]);
                    exl3_had128(v, lane);
                    const __half* sr = svh_e + cta_n + lane * 4;
                    __half* yo = y + (size_t)grow * N + cta_n + lane * 4;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) yo[q] = xq_f2h(v[q] * xq_h2f(sr[q]));
                } else {
                    // gate/up block t: xq_had_svh_rows (gate cols t*128.., up cols N/2 + t*128..)
                    // -> xq_moe_gate_mul -> xq_had_suh_rows (down suh, block t) => xhd row grow
                    const int nh = N >> 1;
                    const int c0 = cta_tile * 128 + lane * 4;
                    float g[4], u[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        g[q] = xq_h2f(smem[r * LT + lane * 4 + q]);
                        u[q] = xq_h2f(smem[(PR + r) * LT + lane * 4 + q]);
                    }
                    exl3_had128(g, lane);
                    exl3_had128(u, lane);
                    float xin[4], sd[4], v[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const __half gh = xq_f2h(g[q] * xq_h2f(svh_e[c0 + q]));
                        const __half uh = xq_f2h(u[q] * xq_h2f(svh_e[nh + c0 + q]));
                        const __half dh = xq_f2h(xq_silu(xq_h2f(gh)) * xq_h2f(uh));
                        xin[q] = xq_h2f(dh);
                        sd[q] = xq_h2f(suh_de[c0 + q]);
                    }
                    wp20_suh_had(xin, sd, v, lane);
                    __half* yo = y + (size_t)grow * nh + c0;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) yo[q] = xq_f2h(v[q]);
                }
            }
        }
    }
    if constexpr (PROF > 0) { const long long pte = clock64(); xq_pf_prof_put<PROF>(pw, pc, pte - pta, pte - pt0); }
}

// ===========================================================================
// A5-L2 (knob prefill.moe_pf2; PLAN/notes_2026-09-27/L2.md): xq_moe_pf2_body = xq_moe_pf_body's
// per-warp work (xq_pf_tables / xq_pf_step / xq_pf_epilogue: the same ring words, decode, A
// fragments, K-ascending m16n8k16 chain and f16 RN store => bitwise == pf_body for every row, MT
// and KS) with the trellis moved from registers to an smem ring filled by cp.async:
//   * stage j = {A rows x KS*16 cols, the CTA's KS x NSEG x 8 warps' ring blocks (contiguous
//     8*WW-word runs per (k16, half))} in one cp.async group; P stages in flight (P + 1 slots,
//     wait_group P - 1), so the loads of stage j + P overlap P stages of decode instead of one;
//   * the warp's word of step s = one LDS (was a register held across the stage), which frees
//     2*KS registers per thread: MT <= 2 at (256, 3) (24 warps/SM instead of 16).
// ===========================================================================
// NR = trellis runs per k16 (2 for the gate|up kernels, incl. the v2 paired NW = 8 ones).
template <int MT, int NW, int KS, int P, int BITS, int NR = NW / 8>
struct XqPf2Smem {
    static constexpr int ROWS = MT * 16;
    static constexpr int LD = KS * 16 + 8;
    static constexpr int ASL = ROWS * LD;                       // halves per A slot
    static constexpr int TW = KS * NR * 8 * 8 * BITS;           // words per trellis slot
    static constexpr int RING = (P + 1) * (ASL + 2 * TW);       // halves
    static constexpr int PR = ROWS < 64 ? ROWS : 64;
    static constexpr int EPIH = NR * PR * 136;
    static constexpr int halves = RING > EPIH ? RING : EPIH;
};

template <int BITS, int MT, int NW, int EPI, int KS, int P, int PROF>
__device__ __forceinline__ void xq_moe_pf2_body(
        const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
        __half* __restrict__ y, int M, int K, int N, int cta_tile, int m_sup,
        __half* __restrict__ smem, const __half* __restrict__ svh_e,
        const __half* __restrict__ suh_de) {
    static_assert(BITS <= 4, "the one-load-per-lane ring decode needs 8*BITS <= 32");
    static_assert(P >= 1, "at least one stage in flight");
    constexpr int ROWS = MT * 16;
    constexpr int LD = KS * 16 + 8;
    constexpr int C8 = KS * 2;                 // 16-B chunks per staged A row
    constexpr int NCH = ROWS * C8;             // A 16-B chunks per stage
    constexpr int NT = NW * 32;
    constexpr int WW = 8 * BITS;
    constexpr int NSEG = (EPI == 2) ? 2 : 1;   // trellis runs per k16 (EPI 2: gate, up)
    constexpr int NH = (EPI == 2 && NW == 8) ? 2 : 1;   // A5-L2 v2: 8 paired warps, each gate AND up block
    constexpr int CPS = 2 * WW;                // 16-B chunks per run (8 warps x WW words)
    constexpr int TCH = KS * NSEG * CPS;       // trellis 16-B chunks per stage
    constexpr int NSL = P + 1;
    using SM = XqPf2Smem<MT, NW, KS, P, BITS, (EPI == 2) ? 2 : 1>;
    long long pt0 = 0, pta = 0, ptb = 0, pw = 0, pc = 0;
    XQ_PF_CLK(pt0);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wl = warp & 7;
    const int hsel = (NW == 16) ? (warp >> 3) : 0;
    const int cta_n = (cta_tile + hsel * (N >> 8)) * 128;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int row_base = m_sup * ROWS;

    int s_i0m[8], s_i1m[8], s_sf[8];
    xq_pf_tables<BITS>(lane, s_i0m, s_i1m, s_sf);

    float acc[NH][MT][2][4];
    #pragma unroll
    for (int hh = 0; hh < NH; ++hh)
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[hh][mt][h][r] = 0.0f;

    const uint32_t* tr32 = (const uint32_t*)trellis;
    const int NS = (KB + KS - 1) / KS;
    __half* abuf = smem;
    uint32_t* tbuf = (uint32_t*)(smem + NSL * SM::ASL);
    const int nbc_lo = cta_tile * 8;                    // first block column of run 0 (gate / raw)
    const int nbc_hi = (cta_tile + (N >> 8)) * 8;       // run 1 (EPI 2 up tile)

    auto stage = [&](const int J, const int sl) {
        const int kc0 = J * (KS * 16);
        __half* dst = abuf + sl * SM::ASL;
        #pragma unroll
        for (int q = 0; q < (NCH + NT - 1) / NT; ++q) {
            const int i = tid + q * NT;
            if ((NCH % NT) == 0 || i < NCH) {
                const int row = i / C8, c8 = i - row * C8;
                const int gr = row_base + row, col = kc0 + c8 * 8;
                const bool ok = gr < M && col < K;
                xq_pf_cp16(dst + row * LD + c8 * 8, ok ? (const void*)(xh + (size_t)gr * K + col) : (const void*)xh, ok);
            }
        }
        uint32_t* tdst = tbuf + sl * SM::TW;
        #pragma unroll
        for (int q = 0; q < (TCH + NT - 1) / NT; ++q) {
            const int i = tid + q * NT;
            if ((TCH % NT) == 0 || i < TCH) {
                const int s = i / (NSEG * CPS);
                const int rem = i - s * (NSEG * CPS);
                const int hh = (NSEG == 2) ? (rem >= CPS) : 0;
                const int c = rem - hh * CPS;
                const int kb = J * KS + s;
                if (kb < KB)
                    xq_pf_cp16(tdst + (s * NSEG + hh) * (8 * WW) + c * 4,
                               tr32 + ((size_t)kb * nb + (hh ? nbc_hi : nbc_lo)) * WW + c * 4, true);
            }
        }
    };

    #pragma unroll
    for (int q = 0; q < P; ++q) {
        if (q < NS) stage(q, q);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    const int gid = lane >> 2, tig = lane & 3;
    int sl = 0, slw = P;
    for (int j = 0; j < NS; ++j) {
        XQ_PF_CLK(pta);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(P - 1) : "memory");
        __syncthreads();                       // stage j visible; everyone is past stage j-1
        XQ_PF_CLK(ptb);
        if (j + P < NS) stage(j + P, slw);     // into the slot stage j-1 used
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        const __half* cur = abuf + sl * SM::ASL;
        const uint32_t* tw = tbuf + sl * SM::TW + hsel * (8 * WW) + wl * WW + lane;
        #pragma unroll
        for (int s = 0; s < KS; ++s) {
            if (j * KS + s < KB) {
                unsigned w0[NH];
                #pragma unroll
                for (int hh = 0; hh < NH; ++hh) w0[hh] = lane < WW ? tw[s * (NSEG * 8 * WW) + hh * (8 * WW)] : 0u;
                xq_pf_step<MT, LD, NH>(w0, cur, s * 16, gid, tig, s_i0m, s_i1m, s_sf, acc);
            }
        }
        sl = (sl + 1 == NSL) ? 0 : sl + 1;
        slw = (slw + 1 == NSL) ? 0 : slw + 1;
        if constexpr (PROF > 0) { const long long ptc = clock64(); pw += ptb - pta; pc += ptc - ptb; }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");   // only empty groups remain; explicit anyway
    XQ_PF_CLK(pta);
    xq_pf_epilogue<MT, NW, EPI, NH>(acc, y, smem, M, N, cta_tile, cta_n, row_base, warp, lane, wl, hsel, gid, tig, svh_e, suh_de);
    if constexpr (PROF > 0) { const long long pte = clock64(); xq_pf_prof_put<PROF>(pw, pc, pte - pta, pte - pt0); }
}
#undef XQ_PF_CLK

// Tile-list entry (== exl3_hmma_gemm_wide_tiles' lookup). kn = K | (N << 16). EPI 2: y is the
// [rows x N/2] xhd buffer and the grid's x extent is N/256 (one CTA per gate|up tile pair).
// V 0 = xq_moe_pf_body (A5-P1), 1 = xq_moe_pf2_body<KS2, P2> (A5-L2).
template <int MT, int NW, int EPI, int V, int KS2, int P2, int PROF>
__device__ __forceinline__ void xq_moe_pf_entry(const uint16_t* __restrict__ base,
        const uint64_t* __restrict__ offs, const __half* __restrict__ xh, __half* __restrict__ y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles,
        const int* __restrict__ offs_row, const int* __restrict__ cnt, int kn, int bits,
        const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all) {
    extern __shared__ __align__(16) __half xq_pf_dsm[];
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int K = kn & 0xFFFF, N = kn >> 16;
    const int e = tiles[2 * tile], m_sup = tiles[2 * tile + 1];
    const int ce = cnt[e];
    const int row0 = offs_row[e];
    const uint16_t* tr = base + (size_t)offs[e];
    const __half* xe = xh + (size_t)row0 * K;
    __half* ye = y + (size_t)row0 * (EPI == 2 ? (N >> 1) : N);
    const __half* svh_e = EPI >= 1 ? svh_all + (size_t)e * N : nullptr;
    const __half* suh_de = EPI == 2 ? suh_d_all + (size_t)e * (N >> 1) : nullptr;
    const int cta_tile = (int)blockIdx.x;
    if constexpr (V == 0) {
        if (bits == 3) xq_moe_pf_body<3, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 4) xq_moe_pf_body<4, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
    } else {
        if (bits == 3) xq_moe_pf2_body<3, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 4) xq_moe_pf2_body<4, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
    }
}

#define XQ_PFV_ENTRY(NAME, MT, NW, EPI, MINB, V, KS2, P2, PROF) \
extern "C" __global__ void __launch_bounds__(NW * 32, MINB) NAME( \
        const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs, \
        const __half* __restrict__ xh, __half* __restrict__ y, const int* __restrict__ tiles, \
        const int* __restrict__ ntiles, const int* __restrict__ offs_row, const int* __restrict__ cnt, \
        int kn, int bits, const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all) { \
    XQ_PDL_ENTRY(); \
    xq_moe_pf_entry<MT, NW, EPI, V, KS2, P2, PROF>(base, offs, xh, y, tiles, ntiles, offs_row, cnt, kn, bits, svh_all, suh_d_all); \
}
#define XQ_PF_ENTRY(NAME, MT, NW, EPI, MINB) XQ_PFV_ENTRY(NAME, MT, NW, EPI, MINB, 0, 0, 0, 0)
// raw y (knob prefill.moe_kernel)
XQ_PF_ENTRY(xq_moe_pf_t1, 1, 8, 0, 2)
XQ_PF_ENTRY(xq_moe_pf_t2, 2, 8, 0, 2)
XQ_PF_ENTRY(xq_moe_pf_t4, 4, 8, 0, 2)
XQ_PF_ENTRY(xq_moe_pf_t8, 8, 8, 0, 2)
// + svh epilogue (down side under prefill.moe_glue_fold)
XQ_PF_ENTRY(xq_moe_pf_s1, 1, 8, 1, 2)
XQ_PF_ENTRY(xq_moe_pf_s2, 2, 8, 1, 2)
XQ_PF_ENTRY(xq_moe_pf_s4, 4, 8, 1, 2)
XQ_PF_ENTRY(xq_moe_pf_s8, 8, 8, 1, 2)
// gate|up pair + svh / gate_mul / down-suh epilogue (gate/up side under prefill.moe_glue_fold)
XQ_PF_ENTRY(xq_moe_pf_g1, 1, 16, 2, 1)
XQ_PF_ENTRY(xq_moe_pf_g2, 2, 16, 2, 1)
XQ_PF_ENTRY(xq_moe_pf_g4, 4, 16, 2, 1)
XQ_PF_ENTRY(xq_moe_pf_g8, 8, 16, 2, 1)
#undef XQ_PF_ENTRY

// A5-L2 (knob prefill.moe_pf2, per-MT-class mask): the smem trellis ring. (KS, P, blocks/SM) per
// class — MIRRORED by moe_pf2_cfg() host-side (EXL3-MOE-PF checks the smem mirror):
//   t/s (256 thr): MT 1, 2 -> KS 4, P 3, 3 CTAs/SM; MT 4 -> KS 4, P 2, 2; MT 8 -> KS 2, P 2, 2
//   g   (512 thr): MT 1, 2 -> KS 4, P 3;            MT 4 -> KS 4, P 2;    MT 8 -> KS 2, P 2 (1 CTA/SM)
// every instance <= 48 KB dynamic smem (the typed launcher has no opt-in).
#define XQ_PF2_T(NAME, MT, EPI, MINB, KS, P) XQ_PFV_ENTRY(NAME, MT, 8, EPI, MINB, 1, KS, P, 0)
XQ_PF2_T(xq_moe_pf2_t1, 1, 0, 3, 4, 3)
XQ_PF2_T(xq_moe_pf2_t2, 2, 0, 3, 4, 3)
XQ_PF2_T(xq_moe_pf2_t4, 4, 0, 2, 4, 2)
XQ_PF2_T(xq_moe_pf2_t8, 8, 0, 2, 2, 2)
XQ_PF2_T(xq_moe_pf2_s1, 1, 1, 3, 4, 3)
XQ_PF2_T(xq_moe_pf2_s2, 2, 1, 3, 4, 3)
XQ_PF2_T(xq_moe_pf2_s4, 4, 1, 2, 4, 2)
XQ_PF2_T(xq_moe_pf2_s8, 8, 1, 2, 2, 2)
#undef XQ_PF2_T
XQ_PFV_ENTRY(xq_moe_pf2_g1, 1, 16, 2, 1, 1, 4, 3, 0)
XQ_PFV_ENTRY(xq_moe_pf2_g2, 2, 16, 2, 1, 1, 4, 3, 0)
XQ_PFV_ENTRY(xq_moe_pf2_g4, 4, 16, 2, 1, 1, 4, 2, 0)
XQ_PFV_ENTRY(xq_moe_pf2_g8, 8, 16, 2, 1, 1, 2, 2, 0)
// A5-L2 v2 (the v1 _prof anatomy: decode/mma = 90-94 % of every CTA, stage-top wait <= 1 % with
// the ring => the loop is latency-bound at 16 warps/SM, not byte-bound; s1 at 24 warps/SM ran
// -14 %): the gate|up fold as 256-thread CTAs whose 8 warps each decode BOTH the gate and the up
// block of their column (two independent chains, one shared A-fragment load) at 2 CTAs/SM =>
// 32 decode chains per SM instead of 16. Same grid (N/256 x tiles), same smem ring, same epilogue.
XQ_PFV_ENTRY(xq_moe_pf2_h1, 1, 8, 2, 2, 1, 4, 2, 0)
XQ_PFV_ENTRY(xq_moe_pf2_h2, 2, 8, 2, 2, 1, 4, 2, 0)
// `_prof` twins (harness only): the served g/s (V 0, slots 0-7) and the pf2 g/s (V 1, slots 8-15);
// slot = 8V + 4*(g) + log2(MT); same arithmetic, clock64 stamps by thread 0.
XQ_PFV_ENTRY(xq_moe_pf_prof_s1, 1, 8, 1, 2, 0, 0, 0, 1)
XQ_PFV_ENTRY(xq_moe_pf_prof_s2, 2, 8, 1, 2, 0, 0, 0, 2)
XQ_PFV_ENTRY(xq_moe_pf_prof_s4, 4, 8, 1, 2, 0, 0, 0, 3)
XQ_PFV_ENTRY(xq_moe_pf_prof_s8, 8, 8, 1, 2, 0, 0, 0, 4)
XQ_PFV_ENTRY(xq_moe_pf_prof_g1, 1, 16, 2, 1, 0, 0, 0, 5)
XQ_PFV_ENTRY(xq_moe_pf_prof_g2, 2, 16, 2, 1, 0, 0, 0, 6)
XQ_PFV_ENTRY(xq_moe_pf_prof_g4, 4, 16, 2, 1, 0, 0, 0, 7)
XQ_PFV_ENTRY(xq_moe_pf_prof_g8, 8, 16, 2, 1, 0, 0, 0, 8)
XQ_PFV_ENTRY(xq_moe_pf2_prof_s1, 1, 8, 1, 3, 1, 4, 3, 9)
XQ_PFV_ENTRY(xq_moe_pf2_prof_s2, 2, 8, 1, 3, 1, 4, 3, 10)
XQ_PFV_ENTRY(xq_moe_pf2_prof_s4, 4, 8, 1, 2, 1, 4, 2, 11)
XQ_PFV_ENTRY(xq_moe_pf2_prof_s8, 8, 8, 1, 2, 1, 2, 2, 12)
XQ_PFV_ENTRY(xq_moe_pf2_prof_g1, 1, 16, 2, 1, 1, 4, 3, 13)
XQ_PFV_ENTRY(xq_moe_pf2_prof_g2, 2, 16, 2, 1, 1, 4, 3, 14)
XQ_PFV_ENTRY(xq_moe_pf2_prof_g4, 4, 16, 2, 1, 1, 4, 2, 15)
XQ_PFV_ENTRY(xq_moe_pf2_prof_g8, 8, 16, 2, 1, 1, 2, 2, 16)
XQ_PFV_ENTRY(xq_moe_pf2_prof_h1, 1, 8, 2, 2, 1, 4, 2, 17)
XQ_PFV_ENTRY(xq_moe_pf2_prof_h2, 2, 8, 2, 2, 1, 4, 2, 18)
#undef XQ_PFV_ENTRY

// the dynamic smem bytes of each pf2 instance (host mirror: moe_pf2_smem_bytes), bits 3 and 4:
// out[b * 8 + k] = kind k in (t1 t2 t4 t8 g1 g2 g4 g8) (s == t), b = 0 bits 3, 1 bits 4
extern "C" __global__ void xq_moe_pf2_smem_query(int* out) {
    out[0] = XqPf2Smem<1, 8, 4, 3, 3>::halves * 2;   out[1] = XqPf2Smem<2, 8, 4, 3, 3>::halves * 2;
    out[2] = XqPf2Smem<4, 8, 4, 2, 3>::halves * 2;   out[3] = XqPf2Smem<8, 8, 2, 2, 3>::halves * 2;
    out[4] = XqPf2Smem<1, 16, 4, 3, 3>::halves * 2;  out[5] = XqPf2Smem<2, 16, 4, 3, 3>::halves * 2;
    out[6] = XqPf2Smem<4, 16, 4, 2, 3>::halves * 2;  out[7] = XqPf2Smem<8, 16, 2, 2, 3>::halves * 2;
    out[8] = XqPf2Smem<1, 8, 4, 3, 4>::halves * 2;   out[9] = XqPf2Smem<2, 8, 4, 3, 4>::halves * 2;
    out[10] = XqPf2Smem<4, 8, 4, 2, 4>::halves * 2;  out[11] = XqPf2Smem<8, 8, 2, 2, 4>::halves * 2;
    out[12] = XqPf2Smem<1, 16, 4, 3, 4>::halves * 2; out[13] = XqPf2Smem<2, 16, 4, 3, 4>::halves * 2;
    out[14] = XqPf2Smem<4, 16, 4, 2, 4>::halves * 2; out[15] = XqPf2Smem<8, 16, 2, 2, 4>::halves * 2;
    // v2 paired h1 / h2 (NW 8, 2 runs), bits 3 then 4
    out[16] = XqPf2Smem<1, 8, 4, 2, 3, 2>::halves * 2;  out[17] = XqPf2Smem<2, 8, 4, 2, 3, 2>::halves * 2;
    out[18] = XqPf2Smem<1, 8, 4, 2, 4, 2>::halves * 2;  out[19] = XqPf2Smem<2, 8, 4, 2, 4, 2>::halves * 2;
}

// the dynamic smem bytes of each (MT, NW) instance (host mirror: moe_pf_smem_bytes)
extern "C" __global__ void xq_moe_pf_smem_query(int* out) {
    out[0] = XqPfSmem<1, 8>::halves * 2;  out[1] = XqPfSmem<2, 8>::halves * 2;
    out[2] = XqPfSmem<4, 8>::halves * 2;  out[3] = XqPfSmem<8, 8>::halves * 2;
    out[4] = XqPfSmem<1, 16>::halves * 2; out[5] = XqPfSmem<2, 16>::halves * 2;
    out[6] = XqPfSmem<4, 16>::halves * 2; out[7] = XqPfSmem<8, 16>::halves * 2;
}

// WP19 (knob prefill.moe_mt_per_expert): xq_moe_pf_count with the tile list split by each
// expert's OWN count: cnt >= 128 -> MT 8 (list 3), >= 33 -> MT 4 (2), >= 17 -> MT 2 (1), else
// MT 1 (0). tiles4 = 4 lists of `cap` (e, super-row) pairs; ntiles4[4]. cnt / offs_row are
// written exactly as xq_moe_pf_count writes them (the place / glue kernels read those).
extern "C" __global__ void __launch_bounds__(1024)
xq_moe_pf_count4(const int* __restrict__ ids, int r, int ne, int* __restrict__ cnt,
                 int* __restrict__ offs_row, int* __restrict__ tiles4, int* __restrict__ ntiles4, int cap) {
    XQ_PDL_ENTRY();
    __shared__ int c_s[1024];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) c_s[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
        const int e = ids[i];
        if (e >= 0 && e < ne) atomicAdd(&c_s[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) cnt[e] = c_s[e];
    if (threadIdx.x == 0) {
        int off = 0;
        int nt[4] = {0, 0, 0, 0};
        for (int e = 0; e < ne; ++e) {
            offs_row[e] = off;
            const int ce = c_s[e];
            if (ce > 0) {
                const int cls = ce >= 128 ? 3 : ce >= 33 ? 2 : ce >= 17 ? 1 : 0;
                const int mt16 = 16 << cls;
                for (int s = 0; s * mt16 < ce; ++s) {
                    int* t = tiles4 + 2 * (cls * cap + nt[cls]);
                    t[0] = e; t[1] = s;
                    ++nt[cls];
                }
            }
            off += ce;
        }
        offs_row[ne] = off;
        ntiles4[0] = nt[0]; ntiles4[1] = nt[1]; ntiles4[2] = nt[2]; ntiles4[3] = nt[3];
    }
}

// ---- grouped wide-M GEMM over per-expert compact row sets. CTA (e, tile, z):
// rows [offs_row[e] + z*16*MT, +16*MT) of expert e's compact list; rows beyond
// cnt[e] zero-pad the A tile and skip the epilogue. grid (esel, tiles, msupmax).
template <int BITS, int MT>
__device__ __forceinline__ void exl3_gemm_body_rows(
        const uint16_t* __restrict__ trellis,
        const __half* __restrict__ xh, const int* __restrict__ row_tok,
        const int* __restrict__ offs_row, const int* __restrict__ cnt,
        __half* __restrict__ yraw, int e, int K, int N, int cta_tile, int z) {
    __shared__ __half sa[MT * 16][16];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int r0 = offs_row[e] + z * (MT * 16);
    const int rem = cnt[e] - z * (MT * 16);
    const int n_live = rem > 0 ? (rem < MT * 16 ? rem : MT * 16) : 0;

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const uint32_t* ring0 = (const uint32_t*)(trellis + (size_t)nb_col * (16 * BITS));
    const size_t kstride = (size_t)nb * (8 * BITS);
    const uint32_t* p0[8];
    const uint32_t* p1[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p0[i] = ring0 + s_i0m[i];
        p1[i] = ring0 + s_i1m[i];
    }
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;

    for (int kb = 0; kb < KB; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int lr = mt * 16 + row;
                const bool ok = lr < n_live && kb * 16 + col < K;
                sa[mt * 16 + row][col] = ok ? xh[(size_t)(r0 + lr) * K + kb * 16 + col]
                                            : __ushort_as_half(0);
            }
        }
        __syncthreads();
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t q0 = exl3_dq_at(p0[h*4+0], p1[h*4+0], s_sf[h*4+0]);
            const uint16_t q1 = exl3_dq_at(p0[h*4+1], p1[h*4+1], s_sf[h*4+1]);
            const uint16_t q2 = exl3_dq_at(p0[h*4+2], p1[h*4+2], s_sf[h*4+2]);
            const uint16_t q3 = exl3_dq_at(p0[h*4+3], p1[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)q0 | ((uint32_t)q1 << 16);
            bfrag[h][1] = (uint32_t)q2 | ((uint32_t)q3 << 16);
        }
        const int gid = lane >> 2, tig = lane & 3;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const __half (*sm)[16] = sa + mt * 16;
            const uint32_t a0 = *(const uint32_t*)&sm[gid][2 * tig];
            const uint32_t a1 = *(const uint32_t*)&sm[gid + 8][2 * tig];
            const uint32_t a2 = *(const uint32_t*)&sm[gid][2 * tig + 8];
            const uint32_t a3 = *(const uint32_t*)&sm[gid + 8][2 * tig + 8];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][h][0]), "+f"(acc[mt][h][1]),
                      "+f"(acc[mt][h][2]), "+f"(acc[mt][h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) { p0[i] += kstride; p1[i] += kstride; }
        __syncthreads();
    }
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row0 = mt * 16 + gid;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
            if (row0 < n_live) {
                __half2 o = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
                *(__half2*)&yraw[(size_t)(r0 + row0) * N + col] = o;
            }
            if (row1 < n_live) {
                __half2 o = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
                *(__half2*)&yraw[(size_t)(r0 + row1) * N + col] = o;
            }
        }
    }
}

extern "C" __global__ void xq_gemm_grouped_rows(const uint16_t* __restrict__ base,
                                                const uint64_t* __restrict__ offs,
                                                const __half* __restrict__ xh,
                                                const int* __restrict__ row_tok,
                                                const int* __restrict__ offs_row,
                                                const int* __restrict__ cnt,
                                                __half* __restrict__ yraw,
                                                int K, int N, int bits, int mt) {
    XQ_PDL_ENTRY();
    const int e = blockIdx.x;
    const int cta_tile = blockIdx.y;
    const int z = blockIdx.z;
    const uint16_t* tr = base + (size_t)offs[e];
#define EXL3_ROWS_CASE(B)                                                     \
    case B:                                                                   \
        switch (mt) {                                                         \
            case 1: exl3_gemm_body_rows<B, 1>(tr, xh, row_tok, offs_row, cnt, yraw, e, K, N, cta_tile, z); break; \
            case 2: exl3_gemm_body_rows<B, 2>(tr, xh, row_tok, offs_row, cnt, yraw, e, K, N, cta_tile, z); break; \
            case 4: exl3_gemm_body_rows<B, 4>(tr, xh, row_tok, offs_row, cnt, yraw, e, K, N, cta_tile, z); break; \
            case 8: exl3_gemm_body_rows<B, 8>(tr, xh, row_tok, offs_row, cnt, yraw, e, K, N, cta_tile, z); break; \
            default: break;                                                   \
        }                                                                     \
        break
    switch (bits) {
        EXL3_ROWS_CASE(3);
        EXL3_ROWS_CASE(4);
        EXL3_ROWS_CASE(5);
        default: break;
    }
#undef EXL3_ROWS_CASE
}

// ---- compact MoE combine: out[m,:] = sig(sg·x_m)*ysh[m,:] + Sum_j wts[m,j]*yd[cand[m*k+j],:].
// j ascending (same order as the decode combine); deterministic.
extern "C" __global__ void xq_moe_combine_rows(__half* __restrict__ out, const __half* __restrict__ yd,
                                               const __half* __restrict__ ysh,
                                               const int* __restrict__ cand, const float* __restrict__ wts,
                                               const __half* __restrict__ sg,
                                               const __half* __restrict__ x, int k, int h, int M) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float part = 0.0f;
    for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float sgv = xq_sig(sm[0]);
    for (int i = tid; i < h; i += bs) {
        float acc = sgv * xq_h2f(ysh[(long long)m * h + i]);
        for (int j = 0; j < k; j++) {
            acc += wts[(long long)m * k + j] * xq_h2f(yd[(long long)cand[(long long)m * k + j] * h + i]);
        }
        out[(long long)m * h + i] = xq_f2h(acc);
    }
}

// ---- one-row device copy (chunked prefill: row c-1 of x -> the lm_head input).
extern "C" __global__ void xq_copy_row(__half* __restrict__ dst, const __half* __restrict__ src,
                                       long long src_off, int h) {
    XQ_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    dst[i] = src[src_off + i];
}

// =============================================================================
// S-A3-f-f: the dense micro-GEMV class + chain occupancy lift.
//
// f-e budget: hc down 97x grid-3 @ 9.3 GB/s (68.3 ms), a/b 72x grid-1/64t
// (10.0 ms), router 48x grid-4 (8.4 ms), quantized chains 20-96 blocks @
// 11-22 GB/s (35.9 ms). Fix families:
//   1. xq_hc_fuse         — down+silu+up of ONE hc mixer in ONE launch; phase A
//                           (K=10240 GEMV) runs warp-per-output with float4
//                           loads and a fixed XOR shuffle tree; a self-resetting
//                           device-scope grid barrier separates it from phase B
//                           (up GEMV, silu applied on the staged input). The
//                           barrier is SAFE only if the whole grid is co-resident
//                           — the launcher sizes grid.x from the occupancy API
//                           (see exl3_forward.rs hc_fuse_max_blocks) and the
//                           kernel is __launch_bounds__(256,6). Reduction order
//                           differs from the old 3-kernel path BY DESIGN
//                           (owner pre-authorized bit-drift re-baseline, brief §1);
//                           the order is FIXED per shape and m-independent
//                           (batch invariance, G-A3-5).
//   2. xq_gemm_f16_v      — warp-per-output vectorized GEMV (a/b projections).
//   3. xq_gemm_f16_f32_v  — router, BIT-EXACT vs xq_gemm_f16_f32: same
//                           thread-per-column form, same ascending scalar fp32
//                           accumulation; ONLY the loads are float4 (values
//                           identical, order identical => output bits identical).
//   4. exl3_hmma_gemm_ks  — the chain GEMM with a K-SPLIT second grid axis
//                           (trellis ring is [k16][n16]-interleaved, so a K
//                           range is directly addressable); fp32 partials +
//                           exl3_ks_combine (ascending fixed order). Order
//                           changes vs the single-block chain — re-baseline
//                           protocol applies; batch-invariant per shape.
// =============================================================================

// ---- self-resetting one-shot grid barrier (device scope).
// Precondition: the ENTIRE grid is co-resident (launcher guarantees via
// occupancy sizing + __launch_bounds__ below). Each launch leaves the counter
// at 0, so one buffer serves every replay without a memset node.
__device__ __forceinline__ void xq_grid_barrier(cuda::atomic<int, cuda::thread_scope_device>* bar,
                                                int goal) {
    __syncthreads();
    if (threadIdx.x == 0) {
        // release: publishes this block's phase-A writes
        const int prev = bar->fetch_add(1, cuda::memory_order_release);
        if (prev == goal - 1) {
            bar->store(0, cuda::memory_order_release);      // release the waiters + reset
        } else {
            while (bar->load(cuda::memory_order_acquire) != 0) __nanosleep(64);
        }
    }
    __syncthreads();   // phase-A results of ALL blocks now visible device-wide
}

// 8 half products, ascending fp32 accumulation (fixed order, drift-authorized paths).
__device__ __forceinline__ float xq_dot8(const __half* w, const __half* x) {
    float4 wv = *(const float4*)w;
    float4 xv = *(const float4*)x;
    const __half* wh = (const __half*)&wv;
    const __half* xh_ = (const __half*)&xv;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc += __half2float(wh[j]) * __half2float(xh_[j]);
    return acc;
}

// ---- fused hc mixer: dd = down(hn); dd = silu(dd/hcn); uu = up(dd). ONE launch.
// grid (bx, m), block 256. Phase A: warp-per-output over lr outputs, K = rw.
// Barrier. Phase B: thread-per-output over rw outputs, K = lr (staged in smem).
extern "C" __global__ void __launch_bounds__(256, 6)
xq_hc_fuse(__half* __restrict__ dd, __half* __restrict__ uu,
           const __half* __restrict__ hn, const __half* __restrict__ w_down,
           const __half* __restrict__ w_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
           float hcn, int lr, int rw) {
    XQ_PDL_ENTRY();
    __shared__ __half sb[512];               // silu(dd) staging (lr <= 512 asserted by launcher)
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    // ---- phase A: dd[row, n] = sum_k W_down[n,k] * hn[row,k]  (fp32, fixed order)
    // S-A3-f-f FIX: the stride MUST be gridDim.x * 8 (warps per row across
    // blocks) so every output n < lr is covered exactly once for ANY bx. The
    // original warps_x = gridDim.x*32 left outputs [bx*8, lr) UNWRITTEN whenever
    // bx*8 < lr (m>=7 => bx<=34 => stale dd tail => garbage), which probe-state
    // caught as N=7/8 DIVERGES. Coverage now: start set [0, bx*8), stride bx*8.
    const __half* hnr = hn + (size_t)row * rw;
    const int s8 = rw >> 8;                  // rw/256 float4 steps per warp-task (rw % 256 == 0)
    for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
        const __half* wr = w_down + (size_t)n * rw;
        float acc = 0.0f;
        // S-A3-p U: unrolled — loads of later steps issue early; the acc chain order is unchanged (bitwise).
        #pragma unroll 4
        for (int s = 0; s < s8; ++s) acc += xq_dot8(wr + (s << 8) + (lane << 3), hnr + (s << 8) + (lane << 3));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) dd[(size_t)row * lr + n] = __float2half_rn(acc);
    }

    xq_grid_barrier(bar, gridDim.x * gridDim.y);

    // ---- stage silu(dd/hcn) (fp16 RN, identical rounding to the old silu_div kernel)
    for (int t = tid; t < lr; t += 256)
        sb[t] = __float2half_rn(xq_silu(__half2float(dd[(size_t)row * lr + t]) / hcn));
    __syncthreads();

    // ---- phase B: uu[row, n] = sum_k W_up[n,k] * sb[k]  (thread-per-output, K = lr)
    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t8 = lr >> 3;                  // lr % 8 == 0 asserted
    for (int n = gt; n < rw; n += tt) {
        const __half* wr = w_up + (size_t)n * lr;
        float acc = 0.0f;
        // S-A3-p U: unrolled — loads of later steps issue early; the acc chain order is unchanged (bitwise).
        #pragma unroll 4
        for (int s = 0; s < t8; ++s) acc += xq_dot8(wr + (s << 3), sb + (s << 3));
        uu[(size_t)row * rw + n] = __float2half_rn(acc);
    }
}

// ---- S-A3-p: int8-weight twin of xq_hc_fuse (opt-in, --hc-int8). The reference implementation's EXL3_GR_INT8
// recipe transcribed onto our mixer layout: symmetric int8, one fp32 scale per OUTPUT row along
// the contracted dim (down: [lr][rw] -> down_s[lr]; up: [rw][lr] -> up_s[rw]). The scale is
// constant over k, so it factors out: dd[n] = s[n] * sum_k q[n,k] * hn[k]. Same grid, barrier,
// silu staging and per-output fixed reduction order as xq_hc_fuse (m-independent => batch
// invariant; plain and verify share it). Half the weight bytes.
__device__ __forceinline__ float xq_dot8_i8(const int8_t* w, const __half* x) {
    const uint2 wv = *(const uint2*)w;
    const float4 xv = *(const float4*)x;
    const int8_t* wq = (const int8_t*)&wv;
    const __half* xh_ = (const __half*)&xv;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc += (float)wq[j] * __half2float(xh_[j]);
    return acc;
}

extern "C" __global__ void __launch_bounds__(256, 6)
xq_hc_fuse_i8(__half* __restrict__ dd, __half* __restrict__ uu,
              const __half* __restrict__ hn, const int8_t* __restrict__ q_down,
              const float* __restrict__ s_down, const int8_t* __restrict__ q_up,
              const float* __restrict__ s_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
              float hcn, int lr, int rw) {
    XQ_PDL_ENTRY();
    __shared__ __half sb[512];
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __half* hnr = hn + (size_t)row * rw;
    const int s8 = rw >> 8;
    for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
        const int8_t* wr = q_down + (size_t)n * rw;
        float acc = 0.0f;
        // S-A3-p U: unrolled — loads of later steps issue early; the acc chain order is unchanged (bitwise).
        #pragma unroll 4
        for (int s = 0; s < s8; ++s) acc += xq_dot8_i8(wr + (s << 8) + (lane << 3), hnr + (s << 8) + (lane << 3));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) dd[(size_t)row * lr + n] = __float2half_rn(acc * s_down[n]);
    }

    xq_grid_barrier(bar, gridDim.x * gridDim.y);

    for (int t = tid; t < lr; t += 256)
        sb[t] = __float2half_rn(xq_silu(__half2float(dd[(size_t)row * lr + t]) / hcn));
    __syncthreads();

    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t16 = lr >> 4;                 // lr % 16 == 0 asserted by the launcher
    for (int n = gt; n < rw; n += tt) {
        const int8_t* wr = q_up + (size_t)n * lr;
        float acc = 0.0f;
        // S-A3-p U: unrolled — loads of later steps issue early; the acc chain order is unchanged (bitwise).
        #pragma unroll 4
        for (int s = 0; s < t16; ++s) {
            acc += xq_dot8_i8(wr + (s << 4), sb + (s << 4));
            acc += xq_dot8_i8(wr + (s << 4) + 8, sb + (s << 4) + 8);
        }
        uu[(size_t)row * rw + n] = __float2half_rn(acc * s_up[n]);
    }
}

// ---- 2026-09-26: ROW-BATCHED int8 mixer — xq_hc_fuse_i8 with all M rows in ONE grid row.
// xq_hc_fuse_i8 runs grid (bx, m): every row's blocks stream the full down/up weight vectors
// (6.6 MB int8 per site), so a 6-row verify read them 6x (57 us/call vs 41 at m=1; floor 27.7).
// Here a warp (phase A) / thread (phase B) loads each weight vector ONCE and runs every row's
// dot product with THAT ROW'S EXACT accumulation sequence (same per-lane ascending dot8 adds,
// same butterfly, same scale-then-round) — so each row is bit-identical to xq_hc_fuse_i8 and to
// the m=1 path (batch invariance by construction). The reference implementation's row-batched gr_mix changed the
// reduction order and was rejected for moving greedy output; this one does not.
template <int M>
__device__ __forceinline__ void xq_hc_fuse_i8_rb_body(__half* __restrict__ dd, __half* __restrict__ uu,
        const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
        const int8_t* __restrict__ q_up, const float* __restrict__ s_up,
        cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr, int rw, __half* sb) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int s8 = rw >> 8;
    for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
        const int8_t* wr = q_down + (size_t)n * rw;
        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < s8; ++s) {
            const int8_t* w = wr + (s << 8) + (lane << 3);
            #pragma unroll
            for (int r = 0; r < M; ++r)
                acc[r] += xq_dot8_i8(w, hn + (size_t)r * rw + (s << 8) + (lane << 3));
        }
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
            if (lane == 0) dd[(size_t)r * lr + n] = __float2half_rn(acc[r] * s_down[n]);
        }
    }

    xq_grid_barrier(bar, gridDim.x * gridDim.y);

    for (int t = tid; t < M * lr; t += 256) {
        const int r = t / lr, k = t - r * lr;
        sb[r * 512 + k] = __float2half_rn(xq_silu(__half2float(dd[(size_t)r * lr + k]) / hcn));
    }
    __syncthreads();

    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t16 = lr >> 4;
    for (int n = gt; n < rw; n += tt) {
        const int8_t* wr = q_up + (size_t)n * lr;
        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < t16; ++s) {
            #pragma unroll
            for (int r = 0; r < M; ++r) {
                acc[r] += xq_dot8_i8(wr + (s << 4), sb + r * 512 + (s << 4));
                acc[r] += xq_dot8_i8(wr + (s << 4) + 8, sb + r * 512 + (s << 4) + 8);
            }
        }
        #pragma unroll
        for (int r = 0; r < M; ++r) uu[(size_t)r * rw + n] = __float2half_rn(acc[r] * s_up[n]);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
xq_hc_fuse_i8_rb(__half* __restrict__ dd, __half* __restrict__ uu,
                 const __half* __restrict__ hn, const int8_t* __restrict__ q_down,
                 const float* __restrict__ s_down, const int8_t* __restrict__ q_up,
                 const float* __restrict__ s_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
                 float hcn, int lr_m, int rw) {
    XQ_PDL_ENTRY();
    __shared__ __half sb[8 * 512];           // silu(dd) rows, pitch 512 (lr <= 512, m <= 8)
    const int lr = lr_m & 0xFFFF, m = lr_m >> 16;
    switch (m) {
        case 1: xq_hc_fuse_i8_rb_body<1>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 2: xq_hc_fuse_i8_rb_body<2>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 3: xq_hc_fuse_i8_rb_body<3>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 4: xq_hc_fuse_i8_rb_body<4>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 5: xq_hc_fuse_i8_rb_body<5>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 6: xq_hc_fuse_i8_rb_body<6>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 7: xq_hc_fuse_i8_rb_body<7>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        case 8: xq_hc_fuse_i8_rb_body<8>(dd, uu, hn, q_down, s_down, q_up, s_up, bar, hcn, lr, rw, sb); break;
        default: break;
    }
}

// =============================================================================
// WP11 (2026-09-26): hc mixer rungs — BITWISE twins of xq_hc_fuse_i8 (m-grid) and
// xq_hc_fuse_i8_rb (row-batched). Every output keeps its exact old reduction sequence:
//   phase A  per lane: acc += dot8 over its 8 k of step s (dot8 = fresh 0.0f + 8 fma,
//            ascending j), s ascending; XOR butterfly 16..1; dd = f16(acc * s_down[n]).
//   phase B  per output: two 8-wide dot8 locals per k16 step, each added to acc, s ascending;
//            uu = f16(acc * s_up[n]).
// Only data movement and VALUE-EXACT conversions change:
//   R1  L2 prefetch of the block's phase-B weight pieces at kernel entry (a hint; the
//       weights are static, so it is issued BEFORE the PDL wait). mode 1 = per-thread
//       prefetch.global.L2 of exactly the 16-B pieces it will load (CCTL.E.PF2);
//       mode 2 = one cp.async.bulk.prefetch.L2 per k16 slab per block (UBLKPF.L2).
//   R3  q_up k-major [lr/16][rw][16] (host relayout at load, hc_quantize_i8): phase-B step s
//       of output n is ONE 16-B load at (s*rw + n)*16, a warp reads 512 contiguous bytes (the
//       row-major layout put 32 lanes on a 320-B stride: 32 sectors per request).
//       silu(dd) is staged as the fp32 VALUES of the same fp16-rounded numbers
//       (h2f(f2h(silu(..))) — exact), so the inner loop does no per-thread half->float.
//   R4  (row-batched only) hn staged in smem per 1024-k chunk, cp.async double-buffered and
//       shared by the block's 8 warps (was: each of 320 warps re-read all m rows, ~38 MB of
//       L1/L2 requests per call); the phase-A weight words of chunk ch+1 are loaded before
//       chunk ch is computed (each word loaded exactly once — no duplicated loads).
//   cvt int8 -> fp32 by the 2^23 magic: u = b ^ 0x80 = b + 128; f32(0x4B0000uu) = 2^23 + u;
//       minus (2^23 + 128) = b EXACTLY (Sterbenz) — PRMT + FADD on full-rate pipes instead of
//       I2F.S8 (16/clk/SM). Used where I2F binds (m <= 4 per block); I2F kept at m >= 5
//       where issue slots bind. Both produce the identical fp32 value of every code.
// Grid barrier: the unchanged xq_grid_barrier. The m-grid twin is __launch_bounds__(256,5)
// (48 regs: the k-major phase B at unroll 4 spills 24 B at the (256,6) cap of 40) with 2 KB
// static smem; the launcher clamps its grid to min(xq_hc_fuse cap, THIS kernel's measured
// occupancy) x 5/6, so co-residency is proven for the kernel actually launched (m = 1 is a
// 40-block grid either way). The rb twin keeps (256,1) with 32 KB static smem at the same
// 40-block grid (1 block/SM, 8 SMs slack), also clamped by its own measured occupancy.
// =============================================================================
__device__ __forceinline__ float xq_i8b_f32(unsigned u, unsigned sel) {
    return __int_as_float((int)__byte_perm(u, 0x4B000000u, sel)) - 8388736.0f;
}

template <bool FAST>
__device__ __forceinline__ void xq_i8x8_f32(const uint2 w, float* f) {
    if (FAST) {
        const unsigned a = w.x ^ 0x80808080u, b = w.y ^ 0x80808080u;
        f[0] = xq_i8b_f32(a, 0x7440u); f[1] = xq_i8b_f32(a, 0x7441u);
        f[2] = xq_i8b_f32(a, 0x7442u); f[3] = xq_i8b_f32(a, 0x7443u);
        f[4] = xq_i8b_f32(b, 0x7440u); f[5] = xq_i8b_f32(b, 0x7441u);
        f[6] = xq_i8b_f32(b, 0x7442u); f[7] = xq_i8b_f32(b, 0x7443u);
    } else {
        const int8_t* q = (const int8_t*)&w;
        #pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = (float)q[j];
    }
}

// dot8 against 8 fp16 activations — the xq_dot8_i8 sequence with the weights pre-converted.
__device__ __forceinline__ float xq_dot8f_h(const float* wf, const float4 xv) {
    const __half* xh_ = (const __half*)&xv;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc += wf[j] * __half2float(xh_[j]);
    return acc;
}

// dot8 against 8 fp32 activation VALUES (a = k0..3, b = k4..7) — same sequence.
__device__ __forceinline__ float xq_dot8f_f(const float* wf, const float4 a, const float4 b) {
    float acc = 0.0f;
    acc += wf[0] * a.x; acc += wf[1] * a.y; acc += wf[2] * a.z; acc += wf[3] * a.w;
    acc += wf[4] * b.x; acc += wf[5] * b.y; acc += wf[6] * b.z; acc += wf[7] * b.w;
    return acc;
}

// R1: L2 prefetch of the phase-B pieces this block will load (all sweeps n = gt + j*tt).
__device__ __forceinline__ void xq_hc_pf_upk(const int8_t* q_upk, int lr, int rw, int mode) {
    if (mode == 0) return;
    const int t16 = lr >> 4;
    const int tt = gridDim.x * 256;
    if (mode == 1) {
        for (int n = blockIdx.x * 256 + threadIdx.x; n < rw; n += tt)
            for (int s = 0; s < t16; ++s)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(q_upk + ((size_t)s * rw + n) * 16)));
    } else {
        const int nsw = (rw + tt - 1) / tt;
        for (int i = threadIdx.x; i < nsw * t16; i += blockDim.x) {
            const int j = i / t16, s = i - j * t16;
            const int n0 = blockIdx.x * 256 + j * tt;
            if (n0 < rw) {
                const unsigned nb = (unsigned)min(256, rw - n0) * 16u;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                             :: "l"(__cvta_generic_to_global(q_upk + ((size_t)s * rw + n0) * 16)), "r"(nb) : "memory");
            }
        }
    }
}

template <bool FAST>
__device__ __forceinline__ void xq_hc_fuse_i8k_body(__half* __restrict__ dd, __half* __restrict__ uu,
        const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
        const int8_t* __restrict__ q_upk, const float* __restrict__ s_up,
        cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr, int rw, float* sbf) {
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const __half* hnr = hn + (size_t)row * rw;
    const int s8 = rw >> 8;
    for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
        const int8_t* wr = q_down + (size_t)n * rw;
        float acc = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < s8; ++s) {
            float wf[8];
            xq_i8x8_f32<FAST>(*(const uint2*)(wr + (s << 8) + (lane << 3)), wf);
            acc += xq_dot8f_h(wf, *(const float4*)(hnr + (s << 8) + (lane << 3)));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) dd[(size_t)row * lr + n] = __float2half_rn(acc * s_down[n]);
    }

    xq_grid_barrier(bar, gridDim.x * gridDim.y);

    for (int t = tid; t < lr; t += 256)
        sbf[t] = __half2float(__float2half_rn(xq_silu(__half2float(dd[(size_t)row * lr + t]) / hcn)));
    __syncthreads();

    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t16 = lr >> 4;
    const uint4* wk = (const uint4*)q_upk;
    for (int n = gt; n < rw; n += tt) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < t16; ++s) {
            const uint4 w = wk[(size_t)s * rw + n];
            float wf[16];
            xq_i8x8_f32<FAST>(make_uint2(w.x, w.y), wf);
            xq_i8x8_f32<FAST>(make_uint2(w.z, w.w), wf + 8);
            const float4* xs = (const float4*)(sbf + (s << 4));
            acc += xq_dot8f_f(wf, xs[0], xs[1]);
            acc += xq_dot8f_f(wf + 8, xs[2], xs[3]);
        }
        uu[(size_t)row * rw + n] = __float2half_rn(acc * s_up[n]);
    }
}

// m-grid twin of xq_hc_fuse_i8 (grid (bx, m), block 256): m = 1 decode/draft/seam, and any
// width the row-batched path does not take. flags: bits 0-1 = R1 prefetch mode (0/1/2),
// bit 2 = force I2F conversion (A/B of the exact PRMT convert). q_upk = k-major up weights.
extern "C" __global__ void __launch_bounds__(256, 5)
xq_hc_fuse_i8k(__half* __restrict__ dd, __half* __restrict__ uu,
               const __half* __restrict__ hn, const int8_t* __restrict__ q_down,
               const float* __restrict__ s_down, const int8_t* __restrict__ q_upk,
               const float* __restrict__ s_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
               float hcn, int lr, int rw, int flags) {
    if (blockIdx.y == 0) xq_hc_pf_upk(q_upk, lr, rw, flags & 3);   // static weights: pre-PDL-wait
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float sbf[512];  // silu(dd) values (lr <= 512 asserted by launcher)
    if (flags & 4) xq_hc_fuse_i8k_body<false>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, sbf);
    else           xq_hc_fuse_i8k_body<true>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, sbf);
}

// R4 staging: chunk ch (k in [ch*1024, ch*1024+1024)) of the M hn rows -> dst[r*1024 + o],
// 16-B cp.async.cg pieces (L2 -> smem, L1 bypassed).
template <int M>
__device__ __forceinline__ void xq_hc_stage_hn(__half* dst, const __half* __restrict__ hn, int rw, int ch) {
    for (int p = threadIdx.x; p < M * 128; p += 256) {
        const int r = p >> 7, o = (p & 127) << 3;
        const __half* src = hn + (size_t)r * rw + (ch << 10) + o;
        const unsigned d = (unsigned)__cvta_generic_to_shared(dst + r * 1024 + o);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src) : "memory");
    }
}

template <int M, bool R4>
__device__ __forceinline__ void xq_hc_fuse_i8_rbk_body(__half* __restrict__ dd, __half* __restrict__ uu,
        const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
        const int8_t* __restrict__ q_upk, const float* __restrict__ s_up,
        cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr, int rw,
        unsigned char* smem) {
    constexpr bool FAST = (M <= 4);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int s8 = rw >> 8;
    if constexpr (R4) {
        __half* hs = (__half*)smem;                 // [2][M][1024] halves
        const int nch = rw >> 10;                   // rw % 1024 == 0 asserted by the launcher
        for (int base = blockIdx.x * 8; base < lr; base += gridDim.x * 8) {   // block-uniform
            const int n = base + warp;
            const bool act = n < lr;
            const int8_t* wr = q_down + (size_t)(act ? n : 0) * rw + (lane << 3);
            float acc[M];
            #pragma unroll
            for (int r = 0; r < M; ++r) acc[r] = 0.0f;
            uint2 wc[4], wn[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) wc[q] = act ? *(const uint2*)(wr + (q << 8)) : make_uint2(0u, 0u);
            xq_hc_stage_hn<M>(hs, hn, rw, 0);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            if (nch > 1) xq_hc_stage_hn<M>(hs + M * 1024, hn, rw, 1);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            for (int ch = 0; ch < nch; ++ch) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                __syncthreads();                    // chunk ch visible to every warp
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    wn[q] = (act && ch + 1 < nch) ? *(const uint2*)(wr + ((((ch + 1) << 2) + q) << 8))
                                                  : make_uint2(0u, 0u);
                if (act) {
                    const __half* hb = hs + (ch & 1) * (M * 1024) + (lane << 3);
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float wf[8];
                        xq_i8x8_f32<FAST>(wc[q], wf);
                        #pragma unroll
                        for (int r = 0; r < M; ++r)
                            acc[r] += xq_dot8f_h(wf, *(const float4*)(hb + r * 1024 + (q << 8)));
                    }
                }
                __syncthreads();                    // every warp done with buffer ch & 1
                if (ch + 2 < nch) xq_hc_stage_hn<M>(hs + (ch & 1) * (M * 1024), hn, rw, ch + 2);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                #pragma unroll
                for (int q = 0; q < 4; ++q) wc[q] = wn[q];
            }
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            if (act) {
                #pragma unroll
                for (int r = 0; r < M; ++r) {
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
                    if (lane == 0) dd[(size_t)r * lr + n] = __float2half_rn(acc[r] * s_down[n]);
                }
            }
        }
    } else {
        for (int n = blockIdx.x * 8 + warp; n < lr; n += gridDim.x * 8) {
            const int8_t* wr = q_down + (size_t)n * rw;
            float acc[M];
            #pragma unroll
            for (int r = 0; r < M; ++r) acc[r] = 0.0f;
            #pragma unroll 4
            for (int s = 0; s < s8; ++s) {
                float wf[8];
                xq_i8x8_f32<FAST>(*(const uint2*)(wr + (s << 8) + (lane << 3)), wf);
                #pragma unroll
                for (int r = 0; r < M; ++r)
                    acc[r] += xq_dot8f_h(wf, __ldg((const float4*)(hn + (size_t)r * rw + (s << 8) + (lane << 3))));
            }
            #pragma unroll
            for (int r = 0; r < M; ++r) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
                if (lane == 0) dd[(size_t)r * lr + n] = __float2half_rn(acc[r] * s_down[n]);
            }
        }
    }

    xq_grid_barrier(bar, gridDim.x * gridDim.y);   // opens with __syncthreads: hs reads are done

    float* sbf = (float*)smem;                      // [M][lr] fp32 values of the f16 silu(dd)
    for (int t = tid; t < M * lr; t += 256) {
        const int r = t / lr, k = t - r * lr;
        sbf[t] = __half2float(__float2half_rn(xq_silu(__half2float(dd[(size_t)r * lr + k]) / hcn)));
    }
    __syncthreads();

    const int gt = blockIdx.x * 256 + tid;
    const int tt = gridDim.x * 256;
    const int t16 = lr >> 4;
    const uint4* wk = (const uint4*)q_upk;
    for (int n = gt; n < rw; n += tt) {
        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.0f;
        #pragma unroll 4
        for (int s = 0; s < t16; ++s) {
            const uint4 w = __ldg(wk + (size_t)s * rw + n);
            float wf[16];
            xq_i8x8_f32<FAST>(make_uint2(w.x, w.y), wf);
            xq_i8x8_f32<FAST>(make_uint2(w.z, w.w), wf + 8);
            #pragma unroll
            for (int r = 0; r < M; ++r) {
                const float4* xs = (const float4*)(sbf + r * lr + (s << 4));
                acc[r] += xq_dot8f_f(wf, xs[0], xs[1]);
                acc[r] += xq_dot8f_f(wf + 8, xs[2], xs[3]);
            }
        }
        #pragma unroll
        for (int r = 0; r < M; ++r) uu[(size_t)r * rw + n] = __float2half_rn(acc[r] * s_up[n]);
    }
}

// Row-batched twin of xq_hc_fuse_i8_rb (grid (bx, 1), block 256, 2 <= m <= 8). lr_m = lr | m<<16;
// flags: bits 0-1 = R1 prefetch mode, bit 4 = R4 hn smem staging (needs rw % 1024 == 0).
extern "C" __global__ void __launch_bounds__(256, 1)
xq_hc_fuse_i8_rbk(__half* __restrict__ dd, __half* __restrict__ uu,
                  const __half* __restrict__ hn, const int8_t* __restrict__ q_down,
                  const float* __restrict__ s_down, const int8_t* __restrict__ q_upk,
                  const float* __restrict__ s_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
                  float hcn, int lr_m, int rw, int flags) {
    const int lr = lr_m & 0xFFFF, m = lr_m >> 16;
    xq_hc_pf_upk(q_upk, lr, rw, flags & 3);   // static weights: pre-PDL-wait
    XQ_PDL_ENTRY();
    // R4: [2][M][1024] halves (<= 32 KB at M = 8); phase B: [M][lr] floats (<= 16 KB).
    __shared__ __align__(16) unsigned char smem[32768];
    const bool r4 = (flags & 16) != 0;
#define XQ_RBK_CASE(MM) case MM: \
        if (r4) xq_hc_fuse_i8_rbk_body<MM, true>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, smem); \
        else    xq_hc_fuse_i8_rbk_body<MM, false>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, smem); \
        break;
    switch (m) {
        XQ_RBK_CASE(2) XQ_RBK_CASE(3) XQ_RBK_CASE(4) XQ_RBK_CASE(5)
        XQ_RBK_CASE(6) XQ_RBK_CASE(7) XQ_RBK_CASE(8)
        default: break;
    }
#undef XQ_RBK_CASE
}

// =============================================================================
// HC (decode wave 4, 2026-09-27): the hc mixer REGRIDDED onto >= 2 CTAs/SM with the hc mix
// (x + the inject gates) fused in, plus a fused inject+norm. BITWISE twins:
//   xq_hc_w4       == xq_hc_fuse_i8_rbk (2 <= m <= 8) / xq_hc_fuse_i8k (m = 1)  +  xq_hc_mix_rg
//   xq_hc_inj_norm == xq_hc_inject  +  xq_hc_norm
// Ledger (p4c, AnsiC m=8, 97 calls/verify): rbk 50.5 us on 40 CTAs (1/SM: 152 regs, 32 KB smem;
// 8 SMs idle) vs a 27.5 us int8 floor, +2.6 us per verify row; around it mix_rg 4.3, norm 4.5,
// inject 2.7 us — 6.0 ms/verify for a 2.67 ms floor.
//
// xq_hc_w4: block 256, __launch_bounds__(256, 3), grid G CTAs that MUST all be co-resident (grid
// barrier). The launcher takes G = clamp(96, ceil(h/32), occupancy x SMs x 5/6) with the occupancy measured on
// the very handle it launches, whose smem carveout is pinned to max (so the measured residency is
// the configured one — never a driver-chosen smaller carveout).
//   phase A  down GEMV dd[r][n] = f16(s_down[n] * sum_k q_down[n][k] hn[r][k]). The m rows are split
//            into ng = (m >= 2 ? 2 : 1) row groups: rows [0, ra) and [ra, m), ra = ceil(m/2) (<= 4
//            rows each). A UNIT = 8 consecutive outputs x one row group; units u = 0 .. ng*lr/8
//            (80 at m >= 2, 40 at m = 1) run on CTAs u, u+G, .. with the rbk R4 body restricted to
//            the group's rows: warp per output, hn chunk (1024 k) staged by cp.async and shared by the
//            CTA's 8 warps, the phase-A words of chunk ch+1 loaded while chunk ch computes. Each row
//            keeps ITS lane chain exactly (per k8 step a fresh-0 dot8 of 8 fma, added to acc, steps
//            ascending; XOR butterfly 16..1; f16(acc * s_down)) — which rows share a warp never
//            changes a bit (rbk's M-rows-per-warp was itself a per-row-identical regrouping).
//   gates    inj[r][s] = 2 sig(sum_c winj[s][c] hn[r][c] / hc), hc x m trees, need hn only: they run
//            BEFORE the barrier on the CTAs that have no phase-A unit (all CTAs when G <= #units),
//            i.e. under the phase-A weight stream. A 256-thread CTA computes mix_rg's 1024 leaves
//            (leaf l = tid + 256q: the same c = l + 1024j ascending fma chain, same guards) and runs
//            the SAME pairing (smem 512..32, warp-0 shfl_down 16..1). The reference implementation's hc_mix.cu shape: a
//            tiny L2-resident reduction redone in its own block instead of a separate launch.
//   barrier  the unchanged xq_grid_barrier.
//   phase B  up GEMV + mix, compute-bound after the barrier, so balanced: EVERY CTA owns hidden
//            columns [c*h/G, (c+1)*h/G) (26-27 at G = 96; chunks of <= 32) x the hc = 4 streams x all
//            m rows; warp w = (stream w & 3, row group w >> 2), lane = column. A thread owns ONE up
//            output n = s*h + i for its group's rows: rbk's chain (k-major 16-B word per k16 step,
//            exact int8 -> fp32, two dot8 locals added in order, f16(acc * s_up)), and parks the f16
//            value (as its exact fp32 value) in smem; then the chunk's m x ncols x values are mix_rg's
//            per-element chain: acc += sig(u_s) * hn_s over s ascending, f16(acc / hc).
// Flags: bit 0 MIX (x + gates; else uu only, the old mix follows), bit 1 write uu even with MIX
// (xcheck), bits 2-3 phase-B L2 prefetch (0 off, 1 prefetch.global.L2 per 32-B sector, 2 one
// cp.async.bulk.prefetch.L2 per contiguous (k16 step, stream) run of the CTA's columns), bits 4-6 phase-A L2 look-ahead in
// 1024-k chunks (0 off). Static smem 20 KB: phase A [2][RG][1024] halves (16 KB at RG = 4), gates
// [1024] floats, phase B [m][512] (row pitch XQ_W4HC_LRP) + [m][hc][32] floats (20 KB at m = 8).
// Launcher contract: hc == 4, h % 64 == 0, rw % 1024 == 0, lr % 16 == 0, lr <= 512, 1 <= m <= 8,
// ceil(h/32) <= G (a CTA owns <= 32 phase-B columns; violated => __trap, never a silent skip).
// =============================================================================
// TP-I3 (b): %globaltimer (ns) for the probe-only xq_hc_w4_prof twin.
__device__ __forceinline__ unsigned long long xq_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define XQ_W4HC_SMEM 20480
#define XQ_W4HC_LRP 512       // phase-B silu row pitch (floats): compile-time => immediate LDS offsets
#define XQ_W4HC_MIX 1
#define XQ_W4HC_UU 2

// RG rows (base hg = hn + r0*rw) x chunk ch (1024 k) -> dst[r*1024 + o], 16-B cp.async.cg pieces.
template <int RG>
__device__ __forceinline__ void xq_w4hc_stage(__half* dst, const __half* __restrict__ hg, int rw, int ch) {
    for (int p = threadIdx.x; p < RG * 128; p += 256) {
        const int r = p >> 7, o = (p & 127) << 3;
        const __half* src = hg + (size_t)r * rw + (ch << 10) + o;
        const unsigned d = (unsigned)__cvta_generic_to_shared(dst + r * 1024 + o);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src) : "memory");
    }
}

// Phase A, one unit: outputs base .. base+7 (warp w -> base + w) x rows r0 .. r0+RG-1.
template <int RG>
__device__ __forceinline__ void xq_w4hc_down(__half* __restrict__ dd, const __half* __restrict__ hn,
        const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
        int lr, int rw, int r0, int base, int la, __half* hs) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int n = base + warp;
    const bool act = n < lr;
    const int8_t* wrow = q_down + (size_t)(act ? n : 0) * rw;
    const int8_t* wr = wrow + (lane << 3);
    const __half* hg = hn + (size_t)r0 * rw;
    const int nch = rw >> 10;                    // rw % 1024 == 0 (launcher)
    float acc[RG];
    #pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
    uint2 wc[4], wn[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) wc[q] = act ? *(const uint2*)(wr + (q << 8)) : make_uint2(0u, 0u);
    xq_w4hc_stage<RG>(hs, hg, rw, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (nch > 1) xq_w4hc_stage<RG>(hs + RG * 1024, hg, rw, 1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    for (int ch = 0; ch < nch; ++ch) {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();                         // chunk ch visible to every warp
        if (la > 0 && act && lane < 8 && ch + 1 + la < nch)   // L2 look-ahead: 8 x 128-B lines per row chunk
            asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(wrow + ((size_t)(ch + 1 + la) << 10) + (lane << 7))));
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            wn[q] = (act && ch + 1 < nch) ? *(const uint2*)(wr + ((((ch + 1) << 2) + q) << 8)) : make_uint2(0u, 0u);
        if (act) {
            const __half* hb = hs + (ch & 1) * (RG * 1024) + (lane << 3);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float wf[8];
                xq_i8x8_f32<true>(wc[q], wf);
                #pragma unroll
                for (int r = 0; r < RG; ++r)
                    acc[r] += xq_dot8f_h(wf, *(const float4*)(hb + r * 1024 + (q << 8)));
            }
        }
        __syncthreads();                         // every warp done with buffer ch & 1
        if (ch + 2 < nch) xq_w4hc_stage<RG>(hs + (ch & 1) * (RG * 1024), hg, rw, ch + 2);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        #pragma unroll
        for (int q = 0; q < 4; ++q) wc[q] = wn[q];
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();                             // hs free (next unit / gates / phase B)
    if (act) {
        #pragma unroll
        for (int r = 0; r < RG; ++r) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
            if (lane == 0) dd[(r0 + r) * lr + n] = __float2half_rn(acc[r] * s_down[n]);   // m*lr < 2^31
        }
    }
}

// One inject gate (tree t = r*hc + s) in a 256-thread CTA: xq_hc_mix_rg's inject block, 4 leaves
// per thread. tsm: 1024 floats of smem.
__device__ __forceinline__ void xq_w4hc_gate(float* __restrict__ inj, const __half* __restrict__ hn,
        const __half* __restrict__ winj, int rw, int hc, int t, float* tsm) {
    const int r = t / hc, s = t - r * hc;
    const __half* wr = winj + (size_t)s * rw;
    const __half* hr = hn + (size_t)r * rw;
    const int tid = threadIdx.x;
    constexpr int J = 12;                        // rw <= 12 * 1024 covers the served 4 x 2560
    #pragma unroll 1
    for (int q = 0; q < 4; ++q) {
        const int l = tid + (q << 8);
        float part = 0.0f;
        if (rw <= J * 1024) {
            float wv[J], hv[J];
            #pragma unroll
            for (int j = 0; j < J; ++j) {
                const int c = l + (j << 10);
                wv[j] = c < rw ? xq_h2f(wr[c]) : 0.0f;
                hv[j] = c < rw ? xq_h2f(hr[c]) : 0.0f;
            }
            #pragma unroll
            for (int j = 0; j < J; ++j)
                if (l + (j << 10) < rw) part += wv[j] * hv[j];
        } else {
            for (int c = l; c < rw; c += 1024) part += xq_h2f(wr[c]) * xq_h2f(hr[c]);
        }
        tsm[l] = part;
    }
    __syncthreads();
    for (int s2 = 512; s2 >= 32; s2 >>= 1) {
        for (int i = tid; i < s2; i += 256) tsm[i] += tsm[i + s2];
        __syncthreads();
    }
    if (tid < 32) {
        float v = tsm[tid];
        #pragma unroll
        for (int s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);
        if (tid == 0) inj[(size_t)r * hc + s] = 2.0f * xq_sig(v / (float)hc);
    }
    __syncthreads();                             // tsm free for the next gate
}

// Phase B, one thread: up output n for rows r0 .. r0+RGB-1; the f16 result parks in ux (and uu).
template <int RGB>
__device__ __forceinline__ void xq_w4hc_up(__half* __restrict__ uu, float* ux,
        const int8_t* __restrict__ q_upk, const float* __restrict__ s_up, const float* sbf,
        int lr, int rw, int r0, int n, int uxo, int uxs, bool wuu) {
    float acc[RGB];
    #pragma unroll
    for (int r = 0; r < RGB; ++r) acc[r] = 0.0f;
    const int t16 = lr >> 4;
    const uint4* wk = (const uint4*)q_upk;
    constexpr int UNR = (RGB >= 3) ? 2 : 4;     // (256, 3) register budget: 0 stack
    #pragma unroll UNR
    for (int st = 0; st < t16; ++st) {
        const uint4 w = __ldg(wk + (size_t)st * rw + n);
        float wf[16];
        xq_i8x8_f32<true>(make_uint2(w.x, w.y), wf);
        xq_i8x8_f32<true>(make_uint2(w.z, w.w), wf + 8);
        #pragma unroll
        for (int r = 0; r < RGB; ++r) {
            const float4* xs = (const float4*)(sbf + (r0 + r) * XQ_W4HC_LRP + (st << 4));
            acc[r] += xq_dot8f_f(wf, xs[0], xs[1]);
            acc[r] += xq_dot8f_f(wf + 8, xs[2], xs[3]);
        }
    }
    const float sc = s_up[n];
    #pragma unroll
    for (int r = 0; r < RGB; ++r) {
        const __half u = __float2half_rn(acc[r] * sc);
        if (wuu) uu[(r0 + r) * rw + n] = u;       // m*rw < 2^31 (launcher: m <= 8, rw <= 12288)
        ux[(r0 + r) * uxs + uxo] = __half2float(u);
    }
}

// CTA c's phase-B hidden columns [i0, i1) = [c*h/G, (c+1)*h/G): every CTA of the grid takes an
// equal share (26-27 columns at G = 96, h = 2560) — the up GEMV + mix is compute-bound after the
// barrier, so its per-SM load must be even (the p4c rbk put 256 columns x 8 rows on each of 40 SMs).
__device__ __forceinline__ int xq_w4hc_col(int c, int h, int G) { return (int)(((long long)c * h) / G); }

// Phase B (after the barrier) for CTA blockIdx.x: its <= 32 columns (launcher contract G >=
// ceil(h / 32)); warp w = (stream s = w & 3, row group g = w >> 2: rows [0, ra) / [ra, m); at m = 1
// group 1 is empty), lane = column. hc == 4 streams (launcher contract) — compile-time here, so
// the smem offsets are immediates.
template <bool PROF>
__device__ __forceinline__ void xq_w4hc_phase_b(const __half* __restrict__ dd, __half* __restrict__ uu,
        __half* __restrict__ x, const __half* __restrict__ hn, const int8_t* __restrict__ q_upk,
        const float* __restrict__ s_up, float hcn, int lr, int h, int m, int ra, bool mix, bool wuu,
        unsigned char* smem, unsigned long long* prof) {
    constexpr int HC = 4, CW = 32, UXS = HC * CW;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rw = h * HC;
    const int i0 = xq_w4hc_col(blockIdx.x, h, gridDim.x);
    const int nc = xq_w4hc_col(blockIdx.x + 1, h, gridDim.x) - i0;
    if (nc > CW) __trap();                       // contract violation (G < ceil(h/32)): never silently skip
    if (nc <= 0) return;                         // (only when G > h) block-uniform
    float* sbf = (float*)smem;                   // [m][LRP] fp32 values of the f16 silu(dd)
    float* ux = sbf + m * XQ_W4HC_LRP;           // [m][HC][CW] fp32 values of the f16 uu
    for (int t = tid; t < m * lr; t += 256) {    // rbk's sbf[r][k] (same dd element, same rounding)
        const int r = t / lr, k = t - r * lr;
        sbf[r * XQ_W4HC_LRP + k] = __half2float(__float2half_rn(xq_silu(__half2float(__ldcg(dd + t)) / hcn)));
    }
    __syncthreads();
    if (PROF && tid == 0) prof[8] = xq_gtimer();
    const int s = warp & 3, g = warp >> 2;
    const int r0 = g ? ra : 0;
    const int rc = (m >= 2) ? (g ? m - ra : ra) : (g ? 0 : 1);
    if (lane < nc) {
        const int n = s * h + i0 + lane;
        switch (rc) {                            // warp-uniform
            case 1: xq_w4hc_up<1>(uu, ux, q_upk, s_up, sbf, lr, rw, r0, n, s * CW + lane, UXS, wuu); break;
            case 2: xq_w4hc_up<2>(uu, ux, q_upk, s_up, sbf, lr, rw, r0, n, s * CW + lane, UXS, wuu); break;
            case 3: xq_w4hc_up<3>(uu, ux, q_upk, s_up, sbf, lr, rw, r0, n, s * CW + lane, UXS, wuu); break;
            case 4: xq_w4hc_up<4>(uu, ux, q_upk, s_up, sbf, lr, rw, r0, n, s * CW + lane, UXS, wuu); break;
            default: break;
        }
    }
    if (PROF) { __syncthreads(); if (tid == 0) prof[5] = xq_gtimer(); }
    if (mix) {
        __syncthreads();                         // the CTA's uu values are parked
        for (int j = tid; j < m * nc; j += 256) {
            const int r = j / nc, jj = j - r * nc;
            const int i = i0 + jj;
            const __half* hr = hn + (r * rw + i);
            float a = 0.0f;
            #pragma unroll
            for (int s2 = 0; s2 < HC; s2++)
                a += xq_sig(ux[r * UXS + s2 * CW + jj]) * xq_h2f(__ldg(hr + s2 * h));
            x[r * h + i] = xq_f2h(a / (float)HC);
        }
    }
}

// TP-I3 (b): the kernel body, PROF = the probe-only timestamp twin (xq_hc_w4_prof: thread 0 of every CTA
// stores %globaltimer at the phase boundaries into prof[blockIdx.x * 16 + i], clock64 at entry/end; PROF =
// false compiles to the served kernel unchanged). Three bitwise phase-B variants measured with it (TP-I3
// note §(b): pre-barrier phase-B loads, silu hoisted into phase A, a pipelined phase-B weight window) all
// tied or lost and were removed.
template <bool PROF>
__device__ __forceinline__ void
xq_hc_w4_body(__half* __restrict__ dd, __half* __restrict__ uu, __half* __restrict__ x, float* __restrict__ inj,
         const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
         const int8_t* __restrict__ q_upk, const float* __restrict__ s_up, const __half* __restrict__ winj,
         cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr_m, int h_hc, int flags,
         unsigned long long* prof) {
    if (PROF) {
        prof += blockIdx.x * 16;
        if (threadIdx.x == 0) {
            unsigned smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            prof[0] = xq_gtimer();
            prof[7] = smid;
            prof[9] = clock64();
        }
    }
    const int lr = lr_m & 0xFFFF, m = lr_m >> 16;
    const int h = h_hc & 0xFFFF, hc = h_hc >> 16;
    const int rw = h * hc;
    const int G = gridDim.x, c = blockIdx.x, tid = threadIdx.x;
    const bool mix = (flags & XQ_W4HC_MIX) != 0;
    const bool wuu = !mix || (flags & XQ_W4HC_UU) != 0;
    // ---- phase-B weights of this CTA's columns into L2 (static weights: before the PDL wait).
    // Per (k16 step st, stream s) the CTA's words are ONE contiguous run of (i1 - i0) * 16 bytes.
    const int pf = (flags >> 2) & 3;
    if (pf != 0) {
        const int i0 = xq_w4hc_col(c, h, G), i1 = xq_w4hc_col(c + 1, h, G);
        const int runs = (lr >> 4) * hc;
        if (i1 > i0) {
            if (pf == 1) {
                const int len = (i1 - i0) * 16;
                const int sec = (len + 31 + 16) >> 5;       // 32-B sectors covering a run (16-B aligned start)
                for (int i = tid; i < runs * sec; i += 256) {
                    const int run = i / sec, e = i - run * sec;
                    const int st = run / hc, s = run - st * hc;
                    const int8_t* p = q_upk + ((size_t)st * rw + (size_t)s * h + i0) * 16 + min(e << 5, len - 16);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
                }
            } else {
                for (int run = tid; run < runs; run += 256) {
                    const int st = run / hc, s = run - st * hc;
                    const int8_t* p = q_upk + ((size_t)st * rw + (size_t)s * h + i0) * 16;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                 :: "l"(__cvta_generic_to_global(p)), "r"((unsigned)((i1 - i0) * 16)) : "memory");
                }
            }
        }
    }
    XQ_PDL_ENTRY();
    if (PROF && tid == 0) prof[1] = xq_gtimer();
    __shared__ __align__(16) unsigned char smem[XQ_W4HC_SMEM];

    // ---- phase A: units (row group, 8 outputs)
    const int ng = (m >= 2) ? 2 : 1;
    const int ra = (m >= 2) ? ((m + 1) >> 1) : 1;
    const int nunits = ng * ((lr + 7) >> 3);
    const int la = (flags >> 4) & 7;
    for (int u = c; u < nunits; u += G) {        // block-uniform
        const int g = u % ng, base = (u / ng) << 3;
        const int r0 = g ? ra : 0, rc = g ? m - ra : ra;
        __half* hs = (__half*)smem;
        switch (rc) {
            case 1: xq_w4hc_down<1>(dd, hn, q_down, s_down, lr, rw, r0, base, la, hs); break;
            case 2: xq_w4hc_down<2>(dd, hn, q_down, s_down, lr, rw, r0, base, la, hs); break;
            case 3: xq_w4hc_down<3>(dd, hn, q_down, s_down, lr, rw, r0, base, la, hs); break;
            case 4: xq_w4hc_down<4>(dd, hn, q_down, s_down, lr, rw, r0, base, la, hs); break;
            default: break;
        }
    }
    if (PROF) { __syncthreads(); if (tid == 0) prof[2] = xq_gtimer(); }
    // ---- inject gates (hn only): on the CTAs without a unit, else on all CTAs
    if (mix && winj != nullptr) {
        const int ntree = hc * m;
        const int tc0 = (G > nunits) ? nunits : 0;
        if (c >= tc0)
            for (int t = c - tc0; t < ntree; t += G - tc0)
                xq_w4hc_gate(inj, hn, winj, rw, hc, t, (float*)smem);
    }
    if (PROF) { __syncthreads(); if (tid == 0) prof[3] = xq_gtimer(); }

    xq_grid_barrier(bar, G);   // opens with __syncthreads: smem reads above are done
    if (PROF && tid == 0) prof[4] = xq_gtimer();

    // ---- phase B: this CTA's hidden columns x the 4 streams x all m rows
    xq_w4hc_phase_b<PROF>(dd, uu, x, hn, q_upk, s_up, hcn, lr, h, m, ra, mix, wuu, smem, prof);
    if (PROF) { __syncthreads(); if (tid == 0) { prof[6] = xq_gtimer(); prof[10] = clock64(); } }
}

extern "C" __global__ void __launch_bounds__(256, 3)
xq_hc_w4(__half* __restrict__ dd, __half* __restrict__ uu, __half* __restrict__ x, float* __restrict__ inj,
         const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
         const int8_t* __restrict__ q_upk, const float* __restrict__ s_up, const __half* __restrict__ winj,
         cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr_m, int h_hc, int flags) {
    xq_hc_w4_body<false>(dd, uu, x, inj, hn, q_down, s_down, q_upk, s_up, winj, bar, hcn, lr_m, h_hc, flags, nullptr);
}

// TP-I3 (b): probe-only timestamp twin (same body, PROF = true); prof = [grid][8] u64.
extern "C" __global__ void __launch_bounds__(256, 3)
xq_hc_w4_prof(__half* __restrict__ dd, __half* __restrict__ uu, __half* __restrict__ x, float* __restrict__ inj,
         const __half* __restrict__ hn, const int8_t* __restrict__ q_down, const float* __restrict__ s_down,
         const int8_t* __restrict__ q_upk, const float* __restrict__ s_up, const __half* __restrict__ winj,
         cuda::atomic<int, cuda::thread_scope_device>* bar, float hcn, int lr_m, int h_hc, int flags,
         unsigned long long* prof) {
    xq_hc_w4_body<true>(dd, uu, x, inj, hn, q_down, s_down, q_upk, s_up, winj, bar, hcn, lr_m, h_hc, flags, prof);
}

// Fused inject + norm (the hc_post of one sublayer + the hc_pre norm of the next), grid hc * B,
// block 1024 (MUST be 1024: the norm's tree leaves). Per (b, s) block: resid += inj[b][s] * y[b]
// (xq_hc_inject's fma, stored), the stored values' sum of squares in xq_hc_norm's per-thread order
// (i = tid, tid+1024, .. ascending) and halving tree, hn = f16(v * inv * (1 + w)) on the SAME
// values (xq_hc_norm re-reads what was stored). h <= 4096 (launcher).
extern "C" __global__ void __launch_bounds__(1024)
xq_hc_inj_norm(__half* __restrict__ out, float* __restrict__ x, const __half* __restrict__ y,
               const float* __restrict__ inj, const __half* __restrict__ w, int h, int hc, int B, float eps) {
    XQ_PDL_ENTRY();
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
            t += g * xq_h2f(yr[i]);
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
        if (i < h) out[(long long)blk * h + i] = xq_f2h(v[j] * inv * (1.0f + xq_h2f(w[(long long)s * h + i])));
    }
}

// ---- warp-per-output vectorized fp16 GEMV (a/b projections; hc fallback path).
// Fixed reduction order: per-lane ascending chunks, XOR butterfly tree. NOT
// bit-identical to xq_gemm_f16 (order changed => re-baseline protocol).
extern "C" __global__ void __launch_bounds__(256)
xq_gemm_f16_v(__half* __restrict__ out, const __half* __restrict__ w,
              const __half* __restrict__ x, int M, int N, int K) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_x = gridDim.x << 5;
    const __half* xr = x + (size_t)row * K;
    const int s8 = K >> 8;                   // full 256-wide steps (K % 256 handled below)
    for (int n = blockIdx.x * 8 + warp; n < N; n += warps_x) {
        const __half* wr = w + (size_t)n * K;
        float acc = 0.0f;
        for (int s = 0; s < s8; ++s) acc += xq_dot8(wr + (s << 8) + (lane << 3), xr + (s << 8) + (lane << 3));
        for (int k = (s8 << 8) + lane; k < K; k += 32)
            acc += __half2float(wr[k]) * __half2float(xr[k]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(size_t)row * N + n] = __float2half_rn(acc);
    }
}

// ---- WP12: the GDN a AND b projections in ONE launch (grid.x = 2 * bx: blocks [0, bx) run
// a, [bx, 2bx) run b). Each output is xq_gemm_f16_v's warp-per-output body verbatim (same
// per-lane ascending chunks, same XOR tree) on the unmodified a_w / b_w rows, so every
// value is bitwise the two-launch pair's. No weight concatenation (zero extra memory).
extern "C" __global__ void __launch_bounds__(256)
xq_gemm_f16_v_ab(__half* __restrict__ out_a, __half* __restrict__ out_b,
                 const __half* __restrict__ w_a, const __half* __restrict__ w_b,
                 const __half* __restrict__ x, int M, int N, int K) {
    XQ_PDL_ENTRY();
    const int bx = gridDim.x >> 1;
    const bool is_b = (int)blockIdx.x >= bx;
    const int bxi = is_b ? (int)blockIdx.x - bx : (int)blockIdx.x;
    __half* __restrict__ out = is_b ? out_b : out_a;
    const __half* __restrict__ w = is_b ? w_b : w_a;
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int warps_x = bx << 5;
    const __half* xr = x + (size_t)row * K;
    const int s8 = K >> 8;                   // full 256-wide steps (K % 256 handled below)
    for (int n = bxi * 8 + warp; n < N; n += warps_x) {
        const __half* wr = w + (size_t)n * K;
        float acc = 0.0f;
        for (int s = 0; s < s8; ++s) acc += xq_dot8(wr + (s << 8) + (lane << 3), xr + (s << 8) + (lane << 3));
        for (int k = (s8 << 8) + lane; k < K; k += 32)
            acc += __half2float(wr[k]) * __half2float(xr[k]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(size_t)row * N + n] = __float2half_rn(acc);
    }
}

// ---- router GEMV, fp32 out. BIT-EXACT twin of xq_gemm_f16_f32: identical
// thread-per-column form and identical ascending per-element fp32 accumulation;
// ONLY the memory ops are float4. Router bits must not move (AGENTS §7).
extern "C" __global__ void xq_gemm_f16_f32_v(float* __restrict__ out,
                                             const __half* __restrict__ w,
                                             const __half* __restrict__ x,
                                             int M, int N, int K) {
    XQ_PDL_ENTRY();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const __half* wr = w + (size_t)n * K;
    const __half* xr = x + (size_t)m * K;
    float acc = 0.0f;
    const int s8 = K >> 3;
    for (int s = 0; s < s8; ++s) {
        float4 wv = *(const float4*)(wr + (s << 3));
        float4 xv = *(const float4*)(xr + (s << 3));
        const __half* wh = (const __half*)&wv;
        const __half* xh_ = (const __half*)&xv;
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += __half2float(wh[j]) * __half2float(xh_[j]);
    }
    for (int k = (s8 << 3); k < K; ++k) acc += __half2float(wr[k]) * __half2float(xr[k]);
    out[(size_t)m * N + n] = acc;
}

// ---- S-A3-f-d step 0 lever 1: router GEMV K-split. The monolithic kernel runs
// ceil(N/256) blocks (2 for ne=512) — 2 SMs of 48, latency-bound at ~29 GB/s.
// Same per-element ascending fp32 accumulation as xq_gemm_f16_f32_v within this
// thread's contiguous K slice; xq_router_combine folds the partials in fixed
// ascending-ks order. Reduction ORDER vs the monolithic kernel differs (fp32
// partial parenthesization) — re-baseline protocol applies (brief §5);
// batch invariance holds because ks is a pure function of (K, N) and each
// row's reduction is m-independent.
extern "C" __global__ void xq_router_ks(float* __restrict__ ws,
                                        const __half* __restrict__ w,
                                        const __half* __restrict__ x,
                                        int M, int N, int K, int ks) {
    XQ_PDL_ENTRY();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;            // grid (n_blocks, M, ks): y = row, z = slice
    if (n >= N || m >= M) return;
    const int slice = K / ks;            // K % ks == 0 && slice % 8 == 0 (picker)
    const int k0 = blockIdx.z * slice;
    const __half* wr = w + (size_t)n * K + k0;
    const __half* xr = x + (size_t)m * K + k0;
    float acc = 0.0f;
    const int s8 = slice >> 3;
    for (int s = 0; s < s8; ++s) {
        float4 wv = *(const float4*)(wr + (s << 3));
        float4 xv = *(const float4*)(xr + (s << 3));
        const __half* wh = (const __half*)&wv;
        const __half* xh_ = (const __half*)&xv;
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += __half2float(wh[j]) * __half2float(xh_[j]);
    }
    for (int k = (s8 << 3); k < slice; ++k) acc += __half2float(wr[k]) * __half2float(xr[k]);
    ws[((size_t)blockIdx.z * M + m) * N + n] = acc;
}

// ---- S-A3-f-d step 0 lever 1: fixed ascending-ks combine (deterministic,
// batch-invariant): out[m][n] = ((p0 + p1) + p2) + ...
extern "C" __global__ void xq_router_combine(const float* __restrict__ ws,
                                             float* __restrict__ out,
                                             int M, int N, int ks) {
    XQ_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    const int m = i / N, n = i % N;
    float acc = 0.0f;
    for (int s = 0; s < ks; ++s) acc += ws[((size_t)s * M + m) * N + n];
    out[i] = acc;
}

// ---- split-K chain body: identical decode/staging to exl3_gemm_body, but the
// K loop runs over this block's [kb0, kb1) slice and the epilogue writes FP32
// partials to ws[ks_id][m][n] (layout [ks][M][N]). grid (tiles, KS).
template <int BITS>
__device__ __forceinline__ void exl3_gemm_body_ks(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ xh,
                                                  float* __restrict__ ws,
                                                  int M, int K, int N, int cta_tile,
                                                  int kb0, int kb1, int ks_id) {
    __shared__ __half sa[16][16];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    // hoisted per-lane slot constants (same as exl3_gemm_body)
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;

    for (int kb = kb0; kb < kb1; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col]
                                : __ushort_as_half(0);
        }
        __syncthreads();

        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
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
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    // ---- fp32 partial epilogue: ws[ks_id][row][col] (rows >= M skipped, as before)
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    float* base = ws + (size_t)ks_id * M * N;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) {
            base[(size_t)row0 * N + col]     = acc[h][0];
            base[(size_t)row0 * N + col + 1] = acc[h][1];
        }
        if (row1 < M) {
            base[(size_t)row1 * N + col]     = acc[h][2];
            base[(size_t)row1 * N + col + 1] = acc[h][3];
        }
    }
}

extern "C" __global__ void exl3_hmma_gemm_ks(const uint16_t* __restrict__ trellis,
                                             const __half* __restrict__ xh,
                                             float* __restrict__ ws,
                                             int M, int K, int N, int bits) {
    // grid.y = KS; KB must be divisible by KS (launcher asserts). Fixed chunking.
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    switch (bits) {
        case 3: exl3_gemm_body_ks<3>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 4: exl3_gemm_body_ks<4>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 5: exl3_gemm_body_ks<5>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        default: break;
    }
}
extern "C" __global__ void exl3_hmma_gemm_ks_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                                                const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                                                float* __restrict__ ws0, float* __restrict__ ws1,
                                                int M, int K, int N, int bits) {
    // exl3_hmma_gemm_ks per pair member (grid.z): identical chunking and body.
    const bool z = blockIdx.z != 0;
    const uint16_t* trellis = z ? tr1 : tr0;
    const __half* xh = z ? xh1 : xh0;
    float* ws = z ? ws1 : ws0;
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    switch (bits) {
        case 3: exl3_gemm_body_ks<3>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 4: exl3_gemm_body_ks<4>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 5: exl3_gemm_body_ks<5>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        default: break;
    }
}

// ---- WP21: split-K FIXUP epilogue — exl3_hmma_gemm_ks + exl3_ks_combine + exl3_had_svh in
// ONE launch. Every CTA writes its fp32 partials exactly as exl3_gemm_body_ks does, then
// arrives on its N-tile's counter (cnt[tile], one u32 per 128-column tile, owned by the
// Quad => distinct slices per chain in flight). The LAST-arriving CTA of the tile (no spin:
// no co-residency requirement, PDL-safe) re-arms the counter to 0 and runs the fixup:
//   combine : acc = 0; acc += ws[s][r][c] for s = 0..ks-1 ASCENDING (exl3_ks_combine's loop,
//             same fp32 add sequence incl. the 0.0f seed), yraw = f16_rn(acc)
//   svh     : warp w owns rows w, w+8 (< M); lane holds cols 4*lane..4*lane+3 of the tile =
//             exl3_had_svh_dev's (row, 128-block) warp layout; same exl3_had128 butterfly,
//             same y = f16_rn(v * svh) rounding.
// No multiply feeds an add anywhere (no FMA-contraction freedom), so y is BITWISE the
// 3-launch path's. yraw is never materialized (no caller reads it after a chain).
//
// v2 (WP21 fix/tune 1). The v1 nsys A/B on .13 had the fused GEMM at +4.1 us/call (+8.8 us
// on the ks=2 N=10240 grid) and the excess grew with the K-chunk length, so it was the MAIN
// LOOP, not the fixup tail. SASS (nvcc/ptxas 13.0, sm_121), SASS instructions per 2-iteration
// k-loop body for bits 3/4/5:
//     exl3_hmma_gemm_ks (old, unbounded, 64 regs)         250/164/240   (x2: 253/163/242 @66)
//     v1 fx  (__launch_bounds__(256,4), inline fixup)     295/181/278   (x2: 309/177/288)
//     v2 fx  (__launch_bounds__(256), isolated fixup)     243/162/236   (x2: 247/162/239 @64)
// The (256,4) / __maxnreg__(64) cap makes ptxas schedule the loop under a hard 64 target and
// rematerialize address math / decode constants inside it (the same cap gives 294/179/285 even
// with a small fixup); v1's inline fixup also CSE'd tid/tile/ks into registers live across the
// loop. This GEMM runs at ~80% of roofline with a dependent LDG -> decode -> HMMA chain per
// iteration, so +16% loop instructions was +10% kernel time. v2:
//   * __launch_bounds__(256) only (.maxntid, no min-blocks): ptxas lands at 64 regs = 4 CTAs/SM
//     on its own and knows tid < 256 (shorter address math: fewer PRMT/IMAD.WIDE than the old
//     unbounded kernel);
//   * the fixup shares NO value with the body: tid / ctaid.x / nctaid.y are re-read through
//     asm volatile (un-CSE-able) and every pointer comes from the kernel params, so nothing
//     extra is live across the k-loop;
//   * arrival: every CTA does a RELEASE-only RMW (MEMBAR + ATOMG, no CCTL.IVALL: v1's acq_rel
//     invalidated the SM's whole L1 under the co-resident CTAs once per CTA); only the elected
//     last CTA issues the acquire fence (acquire pattern = the strong RMW read + fence.acq_rel);
//     the partials are then read with ld.global.cg (L2);
//   * the combine issues up to EXL3_FX_B = 7 partial loads per lane before adding (ks 2/5 = one
//     L2 round trip, 8/10 = two, 40 = six; v1: unroll 8 + a serial remainder loop); adds stay
//     strictly ascending in s. B = 8 pushes the kernel to 78 regs (3 CTAs/SM) — do not raise;
//   * the svh load is issued before the partial loads (independent; v1 issued it after had128).
// Ordering proof (PTX memory model): every thread's partial stores -> bar.sync -> thread 0
// atom.release.gpu (release pattern, cumulative over the CTA through bar.sync). The last CTA's
// thread 0 RMW reads the value written by the ks-1 earlier RMWs (single modification order of
// cnt[tile]) -> fence.acq_rel.gpu (acquire pattern) -> bar.sync -> every fixup thread's loads
// are causally after every producer's stores. Reset: the elected CTA's relaxed store of 0 is
// the last write to cnt[tile] in this launch; the next launch is stream-ordered after it.
#define EXL3_FX_B 7

__device__ __forceinline__ unsigned xq_sreg_tid_x() {
    unsigned r; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned xq_sreg_ctaid_x() {
    unsigned r; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned xq_sreg_nctaid_y() {
    unsigned r; asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(r)); return r;
}

__device__ __forceinline__ void exl3_ks_fixup(const float* __restrict__ ws,
                                              const __half* __restrict__ svh,
                                              __half* __restrict__ y,
                                              unsigned int* __restrict__ cnt,
                                              int M, int N) {
    __shared__ int s_last;
    const unsigned tid = xq_sreg_tid_x();
    const unsigned tile = xq_sreg_ctaid_x();
    const unsigned ks = xq_sreg_nctaid_y();
    unsigned int* const ctr = cnt + tile;
    __syncthreads();                                   // this CTA's partial stores precede the arrival
    if (tid == 0) {
        unsigned prev;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(prev) : "l"(ctr) : "memory");
        const int last = (prev == ks - 1u) ? 1 : 0;
        if (last) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");          // acquire (elected CTA only)
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"          // self re-arming (graph replays)
                         :: "l"(ctr), "r"(0u) : "memory");
        }
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    const int warp = (int)(tid >> 5), lane = (int)(tid & 31u);
    const int col = (int)tile * 128 + lane * 4;
    const size_t sstride = (size_t)M * N;
    const int nks = (int)ks;
    for (int r = warp; r < M; r += 8) {                // warp-uniform: exl3_had128 shuffles full-mask
        // 8-byte svh load (col % 4 == 0, N % 128 == 0 => aligned), independent of the partials.
        const uint2 sv = __ldg((const uint2*)(svh + col));
        const float* p = ws + (size_t)r * N + col;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = 0; s0 < nks; s0 += EXL3_FX_B) {
            float4 q[EXL3_FX_B];
            #pragma unroll
            for (int j = 0; j < EXL3_FX_B; ++j)          // issue the batch's loads first ...
                if (s0 + j < nks) q[j] = __ldcg((const float4*)(p + (size_t)(s0 + j) * sstride));
            #pragma unroll
            for (int j = 0; j < EXL3_FX_B; ++j)          // ... then add strictly ascending in s
                if (s0 + j < nks) { a0 += q[j].x; a1 += q[j].y; a2 += q[j].z; a3 += q[j].w; }
        }
        float v[4];
        v[0] = __half2float(__float2half_rn(a0));
        v[1] = __half2float(__float2half_rn(a1));
        v[2] = __half2float(__float2half_rn(a2));
        v[3] = __half2float(__float2half_rn(a3));
        exl3_had128(v, lane);
        // values and roundings are exl3_had_svh_dev's per-element f16_rn(v * svh).
        const __half* sh = (const __half*)&sv;
        __half o[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * __half2float(sh[j]));
        *(uint2*)(y + (size_t)r * N + col) = *(const uint2*)o;
    }
}

// exl3_hmma_gemm_ks's chunking + body verbatim on one member's pointers, then the fixup.
#define EXL3_FX_MEMBER(TR, XH, WS, SVH, Y, CNT)                                                     \
    do {                                                                                            \
        switch (bits) {                                                                             \
            case 3: exl3_gemm_body_ks<3>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 4: exl3_gemm_body_ks<4>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 5: exl3_gemm_body_ks<5>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            default: break;                                                                         \
        }                                                                                           \
        exl3_ks_fixup(WS, SVH, Y, CNT, M, N);                                                       \
    } while (0)

extern "C" __global__ void __launch_bounds__(256) exl3_hmma_gemm_ks_fx(const uint16_t* __restrict__ trellis,
                                                const __half* __restrict__ xh,
                                                float* __restrict__ ws,
                                                const __half* __restrict__ svh,
                                                __half* __restrict__ y,
                                                unsigned int* __restrict__ cnt,
                                                int M, int K, int N, int bits) {
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    EXL3_FX_MEMBER(trellis, xh, ws, svh, y, cnt);
}
extern "C" __global__ void __launch_bounds__(256) exl3_hmma_gemm_ks_fx_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                                                   const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                                                   float* __restrict__ ws0, float* __restrict__ ws1,
                                                   const __half* __restrict__ svh0, const __half* __restrict__ svh1,
                                                   __half* __restrict__ y0, __half* __restrict__ y1,
                                                   unsigned int* __restrict__ cnt0, unsigned int* __restrict__ cnt1,
                                                   int M, int K, int N, int bits) {
    // exl3_hmma_gemm_ks_x2 per pair member (grid.z), each member with its own counter slice.
    // (A per-member branch around body + fixup compiles to the same loop quality at twice the
    // code size; the selects are kept.)
    const bool z = blockIdx.z != 0;
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    EXL3_FX_MEMBER(z ? tr1 : tr0, z ? xh1 : xh0, z ? ws1 : ws0, z ? svh1 : svh0, z ? y1 : y0, z ? cnt1 : cnt0);
}
#undef EXL3_FX_MEMBER

// ---- fixed-order combine of the split-K partials: ascending ks, fp32, one
// fp16 RN store per element (same final rounding as the single-block epilogue).
extern "C" __global__ void exl3_ks_combine(const float* __restrict__ ws,
                                           __half* __restrict__ yraw,
                                           int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}
extern "C" __global__ void exl3_ks_combine_x2(const float* __restrict__ ws0, const float* __restrict__ ws1,
                                              __half* __restrict__ yraw0, __half* __restrict__ yraw1,
                                              int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const float* ws = blockIdx.z ? ws1 : ws0;
    __half* yraw = blockIdx.z ? yraw1 : yraw0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}

// =============================================================================
// T4e / w3-PSK: PERSISTENT split-K chain GEMM — exl3_hmma_gemm_ks_p (+ _x2 pair twin).
//
// CONTRACT: every fp32 partial ws[slab][row][col] (rows < M) is BITWISE equal to what
// exl3_hmma_gemm_ks / exl3_hmma_gemm_ks_x2 write, for EVERY grid size G >= 1 and every M in
// 1..16; exl3_ks_combine(_x2) is unchanged, so y_raw (and everything downstream) is unchanged.
// Why it holds: the logical work ITEM is exactly one CTA of the old grid — (member, N128 tile,
// K slab), slab = k16 steps [s*chunk, (s+1)*chunk), chunk = (K/16)/ks, ks the SAME per-shape
// factor (chain_ks) — and each item runs the same per-element instruction sequence:
//   acc = 0.0f; for kb ascending over the slab: B = exl3_dq_from on the same ring words,
//   A = the same xh halves (rows >= M -> 0), two m16n8k16.f32.f16.f16.f32 mma (h = 0, 1);
//   then one fp32 store per element at the same address.
// G only decides WHICH CTA runs an item and WHEN, never HOW => G is a pure schedule knob
// (the autotuner's T4e axis). The partition (ks) never depends on m => batch invariance as
// before (AGENTS 2.4: fixed-order combine, no atomics).
//
// Schedule changes vs one-CTA-per-item (all value-transparent):
//   * G persistent CTAs stride the items: item = blockIdx.x + i*G, linearized tile-fastest,
//     then slab, then member (the old grid's dispatch order, so G == items reproduces the old
//     CTA->item map exactly).
//   * A: PSK_S = 8 k16 steps staged per smem block by cp.async.cg (16 B, L1 bypass), DOUBLE
//     buffered: the next block — possibly the NEXT ITEM's first — is in flight while this one
//     computes; ONE barrier per 8 steps (the old body: a dependent LDG.U16 -> STS -> BAR -> ...
//     -> BAR on EVERY step's critical path). Only rows < M are staged; fragment reads of rows
//     >= M are predicated to 0 (the old body read staged zeros: same register values).
//   * trellis: a per-warp L2 prefetch cursor pf_d windows ahead (lanes 0..BITS-1, 32 B each =
//     the 16x16 block's 32*BITS-byte window) that runs ACROSS item boundaries, so an item's
//     first windows are already requested while the previous item drains (S-A3-n F4 mechanism,
//     here without the A-once smem that cost the old F3+F4 chain trial an occupancy step).
//     The prologue is issued BEFORE the PDL wait (weights are static). pf_d = 0 disables it.
//   * the decode keeps exl3_dq_from's scalar window loads (the F8/F9 record: the shuffle
//     decode regressed b5 chains and mispredicted in-round).
// Resources: static smem 2 x 16 x (8*16+8) halves = 8704 B, owned by the ENTRY (one copy for
// all BITS instantiations); __launch_bounds__(256, 4) = the old kernel's 64-reg / 4-CTA/SM class.
// =============================================================================
#define PSK_S 8                                  // k16 steps of A per staged block
#define PSK_SROW (PSK_S * 16 + 8)                // halves per staged row (+16 B: conflict-free frags)
#define PSK_SMEM_HALVES (2 * 16 * PSK_SROW)      // two buffers x 16 rows

__device__ __forceinline__ void psk_cp16(void* smem_dst, const void* gsrc) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(__cvta_generic_to_global(gsrc)) : "memory");
}

// u16-element offset (within its member's trellis) of this warp's 16x16 block at the FIRST k16
// step of `item`; mem_out = the item's member (0/1).
__device__ __forceinline__ size_t psk_off(int item, int per_m, int tiles, int chunk, int nb, int warp,
                                          int blk_elems, bool& mem_out) {
    const bool mem = item >= per_m;
    const int r = mem ? item - per_m : item;
    const int slab = r / tiles;
    const int tile = r - slab * tiles;
    mem_out = mem;
    return ((size_t)slab * chunk * nb + tile * 8 + warp) * (size_t)blk_elems;
}

template <int BITS>
__device__ __forceinline__ void exl3_ks_p_body(const uint16_t* __restrict__ tr0,
                                               const uint16_t* __restrict__ tr1,
                                               const __half* __restrict__ xh0,
                                               const __half* __restrict__ xh1,
                                               float* __restrict__ ws0, float* __restrict__ ws1,
                                               const char* pfb0, const char* pfb1,
                                               int M, int K, int N, int ks, int members, int pf_d,
                                               __half* __restrict__ sa_raw) {
    static_assert(PSK_S == 8, "staging map assumes 16 rows x 16 x 16-B pieces = 256 threads");
    typedef __half SaRow[PSK_SROW];
    SaRow* sa = (SaRow*)sa_raw;                  // [buf*16 + row][col]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb = N >> 4;
    const int tiles = N >> 7;
    const int chunk = (K >> 4) / ks;             // launcher asserts (K/16) % ks == 0
    const int per_m = tiles * ks;                // items per member
    const int n_items = per_m * members;
    const int G = (int)gridDim.x;
    const int nblk = (chunk + PSK_S - 1) / PSK_S;
    constexpr int BLK = 16 * BITS;               // u16 elements per 16x16 trellis block
    const int kstride = nb * 8 * BITS;           // u32 words between consecutive k16 steps

    // ---- trellis L2 prefetch, one burst per staged block, pf_d blocks ahead of the compute
    // block (value-transparent hint). The cursor walks the CTA's (item, block) sequence, so it
    // crosses item boundaries: the next item's first windows are requested while this one runs.
    int pf_it = (int)blockIdx.x, pf_bk = 0;
    auto pf_burst = [&]() {
        if (pf_it < n_items) {
            if (lane < BITS) {
                bool pm;
                const size_t off = psk_off(pf_it, per_m, tiles, chunk, nb, warp, BLK, pm);
                const char* p = (pm ? pfb1 : pfb0) + off * 2u
                              + ((size_t)pf_bk * PSK_S * kstride) * 4u + lane * 32;
                const int ns = min(PSK_S, chunk - pf_bk * PSK_S);
                for (int j = 0; j < ns; ++j, p += (size_t)kstride * 4u)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
            }
            if (++pf_bk == nblk) { pf_bk = 0; pf_it += G; }
        }
    };
    for (int d = 0; d < pf_d; ++d) pf_burst();   // prologue: weights are static -> before the wait

    XQ_PDL_ENTRY();                              // xh (had_suh) / ws (previous combine) after this

    int item = (int)blockIdx.x;
    if (item >= n_items) return;                 // idle CTA (launcher clamps G <= items)

    // ---- A staging: block (it, bk) -> buffer b. Thread = (row, 16-B piece); rows < M only.
    auto stage = [&](int it, int bk, int b) {
        const bool mem = it >= per_m;
        const int r = mem ? it - per_m : it;
        const int slab = r / tiles;
        const int s0 = bk * PSK_S;
        const int ns = min(PSK_S, chunk - s0);
        const int row = tid >> 4, pc = tid & 15;
        if (row < M && pc < 2 * ns)
            psk_cp16(&sa[b * 16 + row][pc * 8],
                     (mem ? xh1 : xh0) + (size_t)row * K + ((size_t)slab * chunk + s0) * 16 + pc * 8);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    stage(item, 0, 0);

    // hoisted per-lane slot constants (identical to exl3_gemm_body_ks)
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const int gid = lane >> 2, tig = lane & 3;
    const bool l0 = gid < M, l1 = (gid + 8) < M, hi_rows = M > 8;

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    bool rm;
    size_t roff = psk_off(item, per_m, tiles, chunk, nb, warp, BLK, rm);
    const uint32_t* __restrict__ ring = (const uint32_t*)((rm ? tr1 : tr0) + roff);
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    int blk = 0, buf = 0;
    for (;;) {
        // the block after this one (the next item's first when this is the item's last)
        int n_item = item, n_blk = blk + 1;
        if (n_blk == nblk) { n_item = item + G; n_blk = 0; }
        const bool more = n_item < n_items;
        if (more) stage(n_item, n_blk, buf ^ 1); // buf^1 is free: every warp passed the last barrier
        if (pf_d > 0) pf_burst();                // trellis windows pf_d blocks ahead

        const int ns = min(PSK_S, chunk - blk * PSK_S);
        const __half* ar0 = &sa[buf * 16 + gid][2 * tig];
        const __half* ar1 = ar0 + 8 * PSK_SROW;
        #pragma unroll 1
        for (int j = 0; j < ns; ++j) {
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
                const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
                const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
                const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
                bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
                bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
            }
            ring += kstride;

            const uint32_t a0 = l0 ? *(const uint32_t*)(ar0) : 0u;
            const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + 8) : 0u;
            uint32_t a1 = 0u, a3 = 0u;
            if (hi_rows) {
                a1 = l1 ? *(const uint32_t*)(ar1) : 0u;
                a3 = l1 ? *(const uint32_t*)(ar1 + 8) : 0u;
            }
            ar0 += 16;
            ar1 += 16;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }

        if (n_blk == 0) {
            // ---- item done: fp32 partial epilogue, ws[slab][row][col] (rows >= M skipped).
            // float2 = the old two scalar stores' exact values at the same addresses.
            const bool mem = item >= per_m;
            const int r = mem ? item - per_m : item;
            const int slab = r / tiles;
            const int tile = r - slab * tiles;
            float* base = (mem ? ws1 : ws0) + (size_t)slab * M * N;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = tile * 128 + 16 * warp + 8 * h + 2 * tig;
                if (l0) *(float2*)&base[(size_t)gid * N + col] = make_float2(acc[h][0], acc[h][1]);
                if (l1) *(float2*)&base[(size_t)(gid + 8) * N + col] = make_float2(acc[h][2], acc[h][3]);
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[h][q] = 0.0f;
            }
        }
        if (!more) break;
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();                         // next block visible; this buffer free
        buf ^= 1;
        blk = n_blk;
        if (n_blk == 0) {
            item = n_item;
            roff = psk_off(item, per_m, tiles, chunk, nb, warp, BLK, rm);
            ring = (const uint32_t*)((rm ? tr1 : tr0) + roff);
        }
    }
}

// Launch word: cfg = bits | ks << 8 | pf_d << 24 (bits 3..5, ks < 65536, pf_d < 128).
// pf0/pf1 = the SAME device addresses as the trellis args, passed separately and WITHOUT
// __restrict__: the prefetch asm takes its address as an operand, which makes that pointer
// escape — derived from `trellis` it would cost the ring loads their ld.global.nc (read-only)
// form that exl3_hmma_gemm_ks has. Values are unaffected either way.
#define PSK_CFG_BITS(c) ((c) & 0xFF)
#define PSK_CFG_KS(c)   (((c) >> 8) & 0xFFFF)
#define PSK_CFG_PF(c)   (((c) >> 24) & 0x7F)

// grid (G, 1, 1), block 256.
extern "C" __global__ void __launch_bounds__(256, 4)
exl3_hmma_gemm_ks_p(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                    float* __restrict__ ws, int M, int K, int N, int cfg, const char* pf0) {
    __shared__ __align__(16) __half psk_sa[PSK_SMEM_HALVES];
    const int ks = PSK_CFG_KS(cfg), pf_d = PSK_CFG_PF(cfg);
    switch (PSK_CFG_BITS(cfg)) {
        case 3: exl3_ks_p_body<3>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        case 4: exl3_ks_p_body<4>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        case 5: exl3_ks_p_body<5>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        default: break;
    }
}
// Pair twin of exl3_hmma_gemm_ks_x2: items [0, per_m) are member 0, [per_m, 2*per_m) member 1.
extern "C" __global__ void __launch_bounds__(256, 4)
exl3_hmma_gemm_ks_p_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                       const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                       float* __restrict__ ws0, float* __restrict__ ws1,
                       int M, int K, int N, int cfg, const char* pf0, const char* pf1) {
    __shared__ __align__(16) __half psk_sa[PSK_SMEM_HALVES];
    const int ks = PSK_CFG_KS(cfg), pf_d = PSK_CFG_PF(cfg);
    switch (PSK_CFG_BITS(cfg)) {
        case 3: exl3_ks_p_body<3>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        case 4: exl3_ks_p_body<4>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        case 5: exl3_ks_p_body<5>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        default: break;
    }
}

// =============================================================================
// W4/DENSE (2026-09-27): the dense split-K chains on W3/LMH's streaming design.
// exl3_dks_b{BITS}_s{NST} replaces, per chain (or per GROUP of same-input chains), the four
// launches exl3_had_suh -> exl3_hmma_gemm_ks[_p] -> exl3_ks_combine -> exl3_had_svh with ONE.
//
// Target contract: sm_121 BASELINE — cp.async.cg 16B + L2::cache_hint (createpolicy evict_first),
// mma.sync.m16n8k16 f16->f32, dp4a, shf, fma.rn.f16x2, atom.release.gpu / fence.acq_rel.gpu,
// griddepcontrol (no-ops without PDL), __grid_constant__ params (LDC c[0x0][R+imm]). No f/a
// suffix feature is used or needed.
//
// WHY PSK v1 LOST IN-ROUND, AND WHY THIS WILL NOT (ledger p4c §3/§8/§9 #2): PSK kept
// exl3_dq_from's 16 dependent scalar loads per lane per k16 step, so every warp had exactly ONE
// 32*BITS-byte trellis window in flight: 32 warps/SM x 160 B = ~5 KB/SM. At 238 GB/s / 48 SMs =
// 5 GB/s per SM that sustains the roofline only if a window returns in <= ~1 us — the loaded
// in-round DRAM latency is longer, so the bodies ran latency-bound at 69-79% (qkv 180 GB/s,
// q 189, out/o 165, z 138 on the fork, small shapes 120-145) — within +-2% of the old
// one-CTA-per-item grid (eager-solo ks_p 9.7 ms vs old ks 9.93 at m=8: -0.25 ms delivered of
// -1.5 priced). Its L2 prefetch and cp.async A staging did not change bytes-in-flight per SM,
// the actual limiter; the persistent grid only removed CTA launch cost that was never the gap.
// LMH streams the SAME 5-bit trellis at 244 GB/s IN-ROUND (1.63 ms, 397 MB) with: a cp.async.cg
// 16-B smem RING (L1 bypass, L2 evict_first), NST-1 = 5 slots x 5 KB = 25 KB in flight per SM
// (5x PSK) that never drains across tile boundaries, decode from smem (4 LDS.32 per lane per
// block instead of 16 dependent LDG), 1 CTA/SM x 8 warps. This kernel IS that pipeline, fed with
// the split-K work items. Its in-flight bytes per SM do not depend on the item size, so the
// small-K-slab items (4 k16 steps) stream exactly like the big ones.
//
//   * item = (member, N128 tile, K slab) = exactly one CTA of exl3_hmma_gemm_ks (same chain_ks).
//     Global item g is member-major, then tile, then slab (slab fastest); CTA c runs items
//     g = c + i*G ascending. G is a multiple of ks => a CTA sees ONE slab per member.
//   * the ring walks the CTA's (item, slot) sequence (slot = DKS_KSTEP k16 rows of the tile's 8
//     trellis blocks) with no drain between items; warps 0..6 issue, warp 7 never issues (its
//     lane 0 owns the tile-counter RMW, so that RMW's release fence cannot wait on a cp.async).
//   * A (the activation after the input Hadamard) is staged ONCE per CTA in smem for every
//     (member, slab) entry the CTA touches: computed in the prologue from x with the pinned WP20
//     suh sequence (lever c: no exl3_had_suh launch; SILU: x := f16(silu(x) * x2) first, the
//     shared-expert down's xq_silu_mul folded in) or copied from a staged xh (escape).
//   * split-K fixup (lever b, WP21-v2's math): every item stores its fp32 partials and arrives on
//     its tile counter (release RMW, DEFERRED one slot so its latency hides under the next slot's
//     MMA); the LAST arrival does the ascending combine + H128 * svh -> y one slot later, and
//     re-arms the counter (graph-replay safe, one real zero at allocation). No exl3_ks_combine /
//     exl3_had_svh launch.
//   * up to DKS_MAXMEM same-input chains (same K, bits, M) share one launch (lever d): one ramp,
//     one tail, the small items fill the big ones' last round.
//
// BIT-IDENTITY CONTRACT (== exl3_had_suh [after xq_silu_mul] -> exl3_hmma_gemm_ks ->
// exl3_ks_combine -> exl3_had_svh per member, every M in 1..16, every G, NST, grouping):
//   1. xh = xq_f2h(wp20_suh_had(x, suh)) — the pinned WP20-v1c op sequence, value-exact against
//      exl3_had_suh_dev under ANY contraction ptxas picks there (x*suh of two f16 values is exact
//      in f32, so fma(x0,s0,q) == rn(rn(x0*s0)+q); the WP20 note). SILU: xq_silu_mul's
//      expression verbatim (xq_f2h(xq_silu(xq_h2f(g)) * xq_h2f(u)): no product feeds an add).
//   2. partial[s][r][c]: acc = 0.0f; for kb ascending over slab s: two m16n8k16 (n8 halves h=0,1)
//      with the same A halves (rows >= M -> 0) and B = W3/LMH's quad extract (== exl3_dq_from;
//      host test lmh_quad_extract_equals_dq_from) -> one fp32 store per element, same address.
//   3. y: acc = 0.0f; acc += partial[s] for s = 0..ks-1 ASCENDING; f16_rn; exl3_had128;
//      f16_rn(v * svh) — exl3_ks_combine + exl3_had_svh_dev's exact sequence (WP21 v2). No
//      multiply feeds an add in 3 (no contraction freedom).
//   G, NST, the grouping, the fixup CTA and the arrival order decide WHO/WHEN, never HOW. The
//   partition (ks) is the chain's own m-independent chain_ks => batch invariance as before.
// Ordering (PTX memory model, WP21's proof): partial stores (all threads) -> bar.sync ->
// atom.release.gpu (warp 7 lane 0; cumulative over the CTA through bar.sync). The last arrival's
// RMW reads the value written by the ks-1 earlier RMWs (single modification order) ->
// fence.acq_rel.gpu (acquire pattern) -> s_fix -> bar.sync -> the fixup's ld.global.cg reads.
// =============================================================================
#define DKS_KSTEP      4                          // k16 steps per ring slot (== LMH_KSTEP)
#define DKS_WARPS      8
#define DKS_THREADS    (DKS_WARPS * 32)
#define DKS_PTHREADS   (DKS_THREADS - 32)         // cp.async producers: warps 0..6
#define DKS_ARRIVE_TID (DKS_THREADS - 32)         // warp 7 lane 0: tile-counter RMW + acquire
#define DKS_MAXMEM     4
#define DKS_MAXENT     24                         // A entries per CTA (host plan checks; kernel traps)
#define DKS_FX_B       20                         // fixup partial loads in flight per lane per batch
#define DKS_AB         8                          // A-prologue tasks per warp per load batch
#define DKS_F_SUH      1                          // A from x via the fused suh (else copy staged xh)
#define DKS_F_FIX      2                          // in-launch split-K fixup -> y (else partials only)
#define DKS_F_SILU     4                          // x := f16(silu(x) * x2) before the suh

// Launch parameter (by value, __grid_constant__). Mirrored by DksMemH / DksJobH (#[repr(C)]) in
// src/exl3_forward.rs — the static_asserts below and the Rust const asserts pin the layout.
struct DksMem {
    unsigned long long tr;    // const uint16_t* trellis, [K/16][N/16] blocks of 16*BITS u16
    unsigned long long suh;   // const __half*   input scale [K]                    (DKS_F_SUH)
    unsigned long long svh;   // const __half*   output scale [N]                   (DKS_F_FIX)
    unsigned long long xh;    // const __half*   staged H(x*suh) [M][K]             (!DKS_F_SUH)
    unsigned long long ws;    // float*          partials [ks][M][N]
    unsigned long long y;     // __half*         output [M][N]                      (DKS_F_FIX)
    unsigned long long cnt;   // unsigned*       tile arrival counters [N/128], 0 at rest (DKS_F_FIX)
    int n, ks, off, rsv;      // N, split count, first global item of this member, 0
};
struct DksJob {
    DksMem mb[DKS_MAXMEM];
    unsigned long long x;     // const __half* input [M][K]                         (DKS_F_SUH)
    unsigned long long x2;    // const __half* second input [M][K]                  (DKS_F_SILU)
    int nmem, M, K, n_items, flags, rsv;
};
static_assert(sizeof(DksMem) == 72, "DksMem layout (mirrored by DksMemH in src/exl3_forward.rs)");
static_assert(sizeof(DksJob) == DKS_MAXMEM * 72 + 16 + 24, "DksJob layout (mirrored by DksJobH)");

struct DksMt  { int g0, cnt, per, ebase; };            // per member: first item, #items, A period, entry base
struct DksEnt { int a_off, c0, w, hb, t_end, j; };     // per A entry: halves offset, slab col0, width,
                                                       // hb0 | nhb << 16, cumulative task end, member
struct DksCur { int g, j, tile, slab, q, nq, chunk; }; // (item, slot) cursor

__host__ __device__ __forceinline__ int dks_gcd(int a, int b) {
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a;
}

// Member of global item g (members are contiguous item ranges, mb[0].off == 0).
__device__ __forceinline__ int dks_member(const DksJob& job, int g) {
    int j = 0;
    #pragma unroll
    for (int i = 1; i < DKS_MAXMEM; ++i)
        if (i < job.nmem && g >= job.mb[i].off) j = i;
    return j;
}

__device__ __forceinline__ void dks_seek(const DksJob& job, int KB, DksCur& c) {
    c.j = dks_member(job, c.g);
    const int ks = job.mb[c.j].ks;
    const int it = c.g - job.mb[c.j].off;
    c.tile = it / ks;
    c.slab = it - c.tile * ks;
    c.q = 0;
    c.chunk = KB / ks;
    c.nq = c.chunk / DKS_KSTEP;
}

__device__ __forceinline__ void dks_advance(const DksJob& job, int KB, int G, int n_items, DksCur& c) {
    if (++c.q == c.nq) {
        c.g += G;
        if (c.g < n_items) dks_seek(job, KB, c);
    }
}

// One ring slot of item p: DKS_KSTEP k16 rows x the tile's 8 trellis blocks (KROW contiguous bytes
// per row), 16-B chunks, L2 evict_first (the weights are streamed once per call).
template <int BITS, int CPT>
__device__ __forceinline__ void dks_issue(const DksJob& job, const DksCur& p, unsigned slot_u,
                                          const int (&c_row)[CPT], const int (&c_byte)[CPT],
                                          const bool (&c_live)[CPT], unsigned long long pol) {
    constexpr int BB = 32 * BITS;
    constexpr int KROW = DKS_WARPS * BB;
    const long long nb = job.mb[p.j].n >> 4;
    const long long rowB = nb * BB;
    const long long kb0 = (long long)p.slab * p.chunk + p.q * DKS_KSTEP;
    const unsigned char* base = (const unsigned char*)job.mb[p.j].tr + (kb0 * nb + p.tile * 8) * BB;
    #pragma unroll
    for (int i = 0; i < CPT; ++i)
        if (c_live[i]) lmh_cp16_ef(slot_u + (unsigned)(c_row[i] * KROW + c_byte[i]),
                                   base + c_row[i] * rowB + c_byte[i], pol);
}

// Tile-counter arrival (release) for item g; returns the pre-add value.
__device__ __forceinline__ unsigned dks_arrive(const DksJob& job, int g) {
    const int j = dks_member(job, g);
    const int tile = (g - job.mb[j].off) / job.mb[j].ks;
    unsigned* ctr = (unsigned*)job.mb[j].cnt + tile;
    unsigned prev;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(ctr) : "memory");
    return prev;
}

// Split-K fixup of item g's tile (called by the whole CTA once the last arrival is acquired):
// exl3_ks_combine's ascending fp32 chain from a 0.0f seed + exl3_had_svh_dev's (row, 128-block)
// warp layout, butterfly and f16_rn(v * svh). Warp w owns rows w, w+8 (< M): warp-uniform.
__device__ __forceinline__ void dks_fixup(const DksJob& job, int g, int M, int warp, int lane) {
    const int j = dks_member(job, g);
    const int N = job.mb[j].n, ks = job.mb[j].ks;
    const int tile = (g - job.mb[j].off) / ks;
    const int col = tile * 128 + lane * 4;
    const float* ws = (const float*)job.mb[j].ws;
    const __half* svh = (const __half*)job.mb[j].svh;
    __half* y = (__half*)job.mb[j].y;
    const size_t sstride = (size_t)M * N;
    for (int r = warp; r < M; r += DKS_WARPS) {
        const uint2 sv = __ldg((const uint2*)(svh + col));       // independent of the partials
        const float* p = ws + (size_t)r * N + col;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = 0; s0 < ks; s0 += DKS_FX_B) {
            // The batch's loads first (UNpredicated: an index past ks re-reads slab ks-1, whose
            // value is then not added — a predicated load + zero-init made ptxas reuse one
            // destination register and serialize the batch) ...
            float4 q[DKS_FX_B];
            #pragma unroll
            for (int b = 0; b < DKS_FX_B; ++b)
                q[b] = __ldcg((const float4*)(p + (size_t)min(s0 + b, ks - 1) * sstride));
            #pragma unroll
            for (int b = 0; b < DKS_FX_B; ++b)                    // ... then the adds, ascending s
                if (s0 + b < ks) { a0 += q[b].x; a1 += q[b].y; a2 += q[b].z; a3 += q[b].w; }
        }
        float v[4];
        v[0] = __half2float(__float2half_rn(a0));
        v[1] = __half2float(__float2half_rn(a1));
        v[2] = __half2float(__float2half_rn(a2));
        v[3] = __half2float(__float2half_rn(a3));
        exl3_had128(v, lane);
        const __half* sh = (const __half*)&sv;
        __half o[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = __float2half_rn(v[i] * __half2float(sh[i]));
        *(uint2*)(y + (size_t)r * N + col) = *(const uint2*)o;
    }
}

// A prologue, fused-suh form: task (entry e, row r, 128-block hb) = one warp computes the whole
// Hadamard block of row r (every lane: the butterfly is warp-collective) and keeps the columns
// inside the entry's slab [c0, c0 + w). DKS_AB tasks' loads are issued before any math.
__device__ __forceinline__ void dks_stage_a_suh(const DksJob& job, const DksEnt* ent, int ne, __half* A,
                                                int K, int warp, int lane) {
    const int T = ne > 0 ? ent[ne - 1].t_end : 0;
    const bool silu = (job.flags & DKS_F_SILU) != 0;
    const __half* x = (const __half*)job.x;
    const __half* x2 = (const __half*)job.x2;
    for (int t0 = warp; t0 < T; t0 += DKS_WARPS * DKS_AB) {
        uint2 xv[DKS_AB], x2v[DKS_AB], sv[DKS_AB];
        int ei[DKS_AB], rr[DKS_AB], cc[DKS_AB];
        #pragma unroll
        for (int b = 0; b < DKS_AB; ++b) {
            const int t = t0 + b * DKS_WARPS;                      // warp-uniform
            ei[b] = -1; rr[b] = 0; cc[b] = 0;
            xv[b] = make_uint2(0u, 0u); x2v[b] = make_uint2(0u, 0u); sv[b] = make_uint2(0u, 0u);
            if (t < T) {
                int e = 0;
                while (t >= ent[e].t_end) ++e;
                const int tb = e > 0 ? ent[e - 1].t_end : 0;
                const int nhb = ent[e].hb >> 16, hb0 = ent[e].hb & 0xFFFF;
                const int u = t - tb;
                const int r = u / nhb;
                const int col = (hb0 + (u - r * nhb)) * 128 + lane * 4;
                const __half* suh = (const __half*)job.mb[ent[e].j].suh;
                xv[b] = *(const uint2*)(x + (size_t)r * K + col);
                if (silu) x2v[b] = *(const uint2*)(x2 + (size_t)r * K + col);
                sv[b] = __ldg((const uint2*)(suh + col));
                ei[b] = e; rr[b] = r; cc[b] = col;
            }
        }
        #pragma unroll
        for (int b = 0; b < DKS_AB; ++b) {
            if (ei[b] < 0) continue;                               // warp-uniform
            const __half* xh4 = (const __half*)&xv[b];
            const __half* x24 = (const __half*)&x2v[b];
            const __half* sh4 = (const __half*)&sv[b];
            float xs[4], ss[4], v[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                // SILU: xq_silu_mul's element (stored f16, reloaded by exl3_had_suh) verbatim.
                xs[q] = silu ? xq_h2f(xq_f2h(xq_silu(xq_h2f(xh4[q])) * xq_h2f(x24[q]))) : xq_h2f(xh4[q]);
                ss[q] = xq_h2f(sh4[q]);
            }
            wp20_suh_had(xs, ss, v, lane);
            const DksEnt& en = ent[ei[b]];
            const int cl = cc[b] - en.c0;
            if (cl >= 0 && cl < en.w) {
                __half o[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) o[q] = xq_f2h(v[q]);
                *(uint2*)(A + en.a_off + rr[b] * (en.w + 8) + cl) = *(const uint2*)o;
            }
        }
    }
}

// A prologue, staged-xh form (GB10_W4DENSE_SUH=0): copy the slab columns of every entry.
__device__ __forceinline__ void dks_stage_a_copy(const DksJob& job, const DksEnt* ent, int ne, __half* A,
                                                 int M, int K, int tid) {
    for (int e = 0; e < ne; ++e) {
        const DksEnt en = ent[e];
        const __half* xh = (const __half*)job.mb[en.j].xh;
        const int v8 = en.w >> 3;
        for (int u = tid; u < M * v8; u += DKS_THREADS) {
            const int r = u / v8, c8 = u - r * v8;
            *(uint4*)(A + en.a_off + r * (en.w + 8) + c8 * 8) =
                *(const uint4*)(xh + (size_t)r * K + en.c0 + c8 * 8);
        }
    }
}

template <int BITS, int NST>
__device__ __forceinline__ void exl3_dks_body(const DksJob& job) {
    extern __shared__ __align__(16) unsigned char dks_sm[];
    __shared__ DksMt s_mt[DKS_MAXMEM];
    __shared__ DksEnt s_ent[DKS_MAXENT];
    __shared__ int s_nent;
    __shared__ int s_fix[2];
    constexpr int BB   = 32 * BITS;                 // bytes per 16x16 trellis block
    constexpr int KROW = DKS_WARPS * BB;            // bytes per k16 row of one 128-col tile
    constexpr int STB  = DKS_KSTEP * KROW;          // bytes per ring slot
    constexpr int CPR  = KROW / 16;                 // 16-B chunks per k16 row
    constexpr int NCH  = STB / 16;                  // 16-B chunks per slot
    constexpr int CPT  = (NCH + DKS_PTHREADS - 1) / DKS_PTHREADS;
    constexpr int WW   = 8 * BITS;                  // ring words per block
    static_assert(NST >= 2, "ring needs >= 2 slots");
    static_assert(BB % 16 == 0 && KROW % 16 == 0, "16-B chunking");

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int G = (int)gridDim.x, cta = (int)blockIdx.x;
    const int M = job.M, K = job.K, KB = K >> 4, n_items = job.n_items, flags = job.flags;
    const bool fix_on = (flags & DKS_F_FIX) != 0;
    if (cta >= n_items) {                           // CTA-uniform: no item for this CTA
        XQ_PDL_ENTRY();
        return;
    }
    unsigned char* ring = dks_sm;
    __half* A = (__half*)(dks_sm + NST * STB);
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);

    // Host/device layout tripwires (the TRIPWIRE_SPEC §3.4 class): trap instead of overrunning.
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)(NST * STB)) __trap();
        if (M < 1 || M > 16 || job.nmem < 1 || job.nmem > DKS_MAXMEM || job.mb[0].off != 0 || (K & 127) != 0) __trap();
    }

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    // ---- per-thread slot chunk geometry: chunk c -> (k16 row, byte in the row span); warps 0..6.
    int c_row[CPT], c_byte[CPT];
    bool c_live[CPT];
    #pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = tid + i * DKS_PTHREADS;
        c_live[i] = tid < DKS_PTHREADS && c < NCH;
        c_row[i] = c / CPR;
        c_byte[i] = (c - (c / CPR) * CPR) * 16;
    }

    // ---- ring prologue: slots 0..NST-2 of this CTA's sequence (static weights: before the PDL wait)
    DksCur P;
    P.g = cta;
    dks_seek(job, KB, P);
    int p_slot = 0;
    #pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) {
        if (P.g < n_items) {
            dks_issue<BITS, CPT>(job, P, ring_u + (unsigned)(p_slot * STB), c_row, c_byte, c_live, pol);
            dks_advance(job, KB, G, n_items, P);
            if (++p_slot == NST) p_slot = 0;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // ---- this CTA's member table + A entries (host mirror: dks_cta_plan, src/exl3_forward.rs)
    if (tid < job.nmem) {
        const int j = tid;
        const int off = job.mb[j].off, ks = job.mb[j].ks;
        const int end = (j + 1 < job.nmem) ? job.mb[j + 1].off : n_items;
        if (ks < 1 || (KB % ks) != 0 || ((KB / ks) % DKS_KSTEP) != 0 || (job.mb[j].n & 127) != 0 || end < off) __trap();
        const int g0 = off + (((cta - off) % G) + G) % G;      // first item of member j on this CTA
        s_mt[j].g0 = g0;
        s_mt[j].cnt = g0 < end ? (end - 1 - g0) / G + 1 : 0;
        s_mt[j].per = ks / dks_gcd(G, ks);                     // its slabs repeat with this period
    }
    __syncthreads();
    if (tid == 0) {
        int ne = 0, a_off = 0, t = 0;
        for (int j = 0; j < job.nmem; ++j) {
            const int ks = job.mb[j].ks, W = (KB / ks) * 16;
            const int E = min(s_mt[j].per, s_mt[j].cnt);
            s_mt[j].ebase = ne;
            for (int e = 0; e < E; ++e) {
                if (ne >= DKS_MAXENT) __trap();
                const int slab = (s_mt[j].g0 - job.mb[j].off + e * G) % ks;
                const int c0 = slab * W;
                const int hb0 = c0 >> 7, nhb = ((c0 + W - 1) >> 7) - hb0 + 1;
                t += M * nhb;
                s_ent[ne].a_off = a_off; s_ent[ne].c0 = c0; s_ent[ne].w = W;
                s_ent[ne].hb = hb0 | (nhb << 16); s_ent[ne].t_end = t; s_ent[ne].j = j;
                a_off += (M * (W + 8) + 7) & ~7;
                ++ne;
            }
        }
        s_nent = ne;
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)(NST * STB + a_off * 2)) __trap();
    }

    // ---- PDL: x / x2 / xh / ws / cnt / y are touched only past this point.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    int total = 0;                                  // this CTA's ring slots
    for (int j = 0; j < job.nmem; ++j) total += s_mt[j].cnt * ((KB / job.mb[j].ks) / DKS_KSTEP);
    if (flags & DKS_F_SUH) dks_stage_a_suh(job, s_ent, s_nent, A, K, warp, lane);
    else                   dks_stage_a_copy(job, s_ent, s_nent, A, M, K, tid);
    // (A visibility: the main loop's first __syncthreads)

    // ---- per-lane decode constants (W3/LMH): quad q (= n8 half h) holds t = 8*lane + 4q + s.
    int qa[2], qb[2];
    unsigned dsh[8];
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int t0 = 8 * lane + 4 * q;
        const int b0 = (t0 + 257) * BITS - 16;
        const int b2 = (t0 + 3 + 257) * BITS;
        const int I0 = b0 >> 5;
        const int I2 = (b2 - 1) >> 5;
        const int s2 = ((I2 + 1) << 5) - b2;
        qa[q] = I0 % WW;
        qb[q] = I2 % WW;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int sh = s2 + (3 - s) * BITS;
            dsh[q * 4 + s] = (unsigned)(sh & 31) | (sh >= 32 ? 32u : 0u);
        }
    }
    const __half2 k_inv2  = __half2half2(__ushort_as_half((unsigned short)0x1EEE));
    const __half2 k_bias2 = __half2half2(__ushort_as_half((unsigned short)0xC931));

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
    const bool l0 = gid < M, l1 = (gid + 8) < M;

    DksCur C;
    C.g = cta;
    dks_seek(job, KB, C);
    int c_slot = 0, srow = 8;
    const __half* ar0 = A;
    int arr_g = -1;                                 // item whose partials were stored last iteration
    int t0_g = -1;                                  // DKS_ARRIVE_TID: item whose RMW is in flight
    unsigned t0_prev = 0u;

    #pragma unroll 1
    for (int k = 0; k < total + 2; ++k) {           // + 2: arrival / fixup flush of the last items
        // slot k landed (<= NST-2 newer groups pending) and is visible CTA-wide; every warp is past
        // slot k-1, so its ring slot may be refilled.
        if (k < total) asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        const int fix = (fix_on && k > 0) ? s_fix[(k - 1) & 1] : -1;   // decided last iteration
        if (fix_on && tid == DKS_ARRIVE_TID && arr_g >= 0) {           // stores -> bar -> release
            t0_prev = dks_arrive(job, arr_g);
            t0_g = arr_g;
        }
        if (k < total) {
            if (P.g < n_items) {
                dks_issue<BITS, CPT>(job, P, ring_u + (unsigned)(p_slot * STB), c_row, c_byte, c_live, pol);
                dks_advance(job, KB, G, n_items, P);
                if (++p_slot == NST) p_slot = 0;
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        if (fix >= 0) dks_fixup(job, fix, M, warp, lane);             // CTA-uniform
        int end_g = -1;
        if (k < total) {
            if (C.q == 0) {                                            // item start: its A entry
                const DksMt mt = s_mt[C.j];
                const DksEnt en = s_ent[mt.ebase + ((C.g - mt.g0) / G) % mt.per];
                srow = en.w + 8;
                ar0 = A + en.a_off + gid * srow + 2 * tig;
            }
            const unsigned char* slot = ring + c_slot * STB + warp * BB;
            #pragma unroll
            for (int g = 0; g < DKS_KSTEP; ++g) {
                const uint32_t* w = (const uint32_t*)(slot + g * KROW);
                uint32_t bfrag[2][2];
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t wa = w[qa[q]];
                    const uint32_t wb = w[qb[q]];
                    uint32_t sum[4];
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const unsigned d = dsh[q * 4 + s];
                        const uint32_t lo = (d & 32u) ? wa : wb;
                        const unsigned idx = __funnelshift_r(lo, wa, d & 31u) & 0xFFFFu;
                        sum[s] = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    }
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const __half2 hv = __halves2half2(__ushort_as_half((unsigned short)sum[2 * p]),
                                                          __ushort_as_half((unsigned short)sum[2 * p + 1]));
                        const __half2 wv = __hfma2(hv, k_inv2, k_bias2);
                        bfrag[q][p] = *reinterpret_cast<const uint32_t*>(&wv);
                    }
                }
                const int col = (C.q * DKS_KSTEP + g) * 16;
                uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
                if (l0) {
                    a0 = *(const uint32_t*)(ar0 + col);
                    a2 = *(const uint32_t*)(ar0 + col + 8);
                }
                if (l1) {
                    a1 = *(const uint32_t*)(ar0 + 8 * srow + col);
                    a3 = *(const uint32_t*)(ar0 + 8 * srow + col + 8);
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                }
            }
            if (++c_slot == NST) c_slot = 0;
            if (C.q == C.nq - 1) {
                // ---- item done: fp32 partials ws[slab][row][col] (rows >= M skipped) — the old
                // kernel's values at its addresses (float2 = its two scalar stores).
                const int N = job.mb[C.j].n;
                float* base = (float*)job.mb[C.j].ws + (size_t)C.slab * M * N;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = C.tile * 128 + 16 * warp + 8 * h + 2 * tig;
                    if (l0) *(float2*)&base[(size_t)gid * N + col] = make_float2(acc[h][0], acc[h][1]);
                    if (l1) *(float2*)&base[(size_t)(gid + 8) * N + col] = make_float2(acc[h][2], acc[h][3]);
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
                }
                end_g = C.g;
                C.g += G;
                if (C.g < n_items) dks_seek(job, KB, C);
            } else {
                ++C.q;
            }
        }
        if (fix_on && tid == DKS_ARRIVE_TID) {
            int res = -1;
            if (t0_g >= 0) {
                const int j = dks_member(job, t0_g);
                if (t0_prev == (unsigned)(job.mb[j].ks - 1)) {         // the tile's last arrival
                    unsigned* ctr = (unsigned*)job.mb[j].cnt + (t0_g - job.mb[j].off) / job.mb[j].ks;
                    asm volatile("fence.acq_rel.gpu;" ::: "memory");            // acquire
                    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ctr), "r"(0u) : "memory");
                    res = t0_g;
                }
                t0_g = -1;
            }
            s_fix[k & 1] = res;                     // read by every thread after the next barrier
        }
        arr_g = end_g;
    }
}

// Entries exl3_dks_b{BITS}_s{NST}: 5-bit (the served dense chains) and 4-bit (MTP fc) at ring
// depth 4/6/8 (GB10_W4DENSE_NST), 3-bit at 6 for coverage. 1 CTA/SM x 8 warps (the LMH class).
#define DKS_ENTRY(B, NS)                                                                        \
extern "C" __global__ void __launch_bounds__(DKS_THREADS, 1)                                    \
exl3_dks_b##B##_s##NS(const __grid_constant__ DksJob job) {                                     \
    exl3_dks_body<B, NS>(job);                                                                  \
}
DKS_ENTRY(5, 4)
DKS_ENTRY(5, 6)
DKS_ENTRY(5, 8)
DKS_ENTRY(4, 4)
DKS_ENTRY(4, 6)
DKS_ENTRY(4, 8)
DKS_ENTRY(3, 6)

// =============================================================================
// S-A3-f-d — MTP k-chain VERIFY (width m = k+1 rows of ONE slot, positions
// p..p+m-1). Transcribed structure from the reference implementation recipe's device-resident
// draft-chain patch (0003) + pruned draft head (0002); every stateful op is a
// CHAIN variant of the decode kernel: the group's rows are processed
// SEQUENTIALLY so each row sees the previous rows' committed state, with the
// SAME device functions / reduction orders as the m=1 decode path => verified
// rows are BIT-IDENTICAL to sequential decode (the losslessness contract).
// Rollback model: the verify runs against SHADOW state buffers (copied from
// the live state before the launch); after the on-device accept computes `a`
// (accepted draft count), the commit kernels re-run rows 0..a from the LIVE
// state (bit-identical replay) so rows a+1..m-1 never commit.
// =============================================================================

// ---- chained full-attention decode: grid (nh, m) — m rides gridDim.y (the
// 12-arg cap); rows loop IN-BLOCK so row r's softmax reads rows <= r's K/V
// already written by THIS block (no cross-block race, no separate pre-pass).
// Same shared fns (xq_k_norm_rope / xq_attn_softmax) and the same per-row
// cache-write + ascending-two-pass order as xq_attn_decode.
// ---- chained verify attention (rows = one sequence's candidate tokens, one slot).
// S-A3-l: the OLD body ran xq_attn_softmax once PER ROW — the per-position
// halving-tree sync chain ran m times over nearly the same prefix. The S-A3-l
// budget priced that at +24.7 ms of the +55.6 ms m=1->m=6 overcost (30.1 ms of
// the 101.2 ms m=6 round). The BATCHED body runs the two passes ONCE with every
// row's score partials segmented in shared memory (thread tid = dim tid of EVERY
// row); the halving tree steps all segments under the same syncs.
// Bit contract: phase 1 is the old per-row body verbatim and each row's scan ops
// are xq_attn_softmax's exact per-row sequence (same tree within the segment,
// same ascending-t passes, same fmaxf/__expf ops) => row 0 stays bit-identical
// to xq_attn_decode (probe-exl3-binv). acc/denom/mx are TEMPLATE arrays (G
// compile-time): a runtime-m per-thread array lands in local memory (AGENTS §4).
__device__ void xq_attn_decode_group_loop(__half* __restrict__ attn,
                                          const __half* __restrict__ qg,
                                          const __half* __restrict__ kp,
                                          const __half* __restrict__ vp,
                                          float* __restrict__ kcvc,
                                          const __half* __restrict__ qnkn,
                                          const float* __restrict__ cossin,
                                          const int* __restrict__ slotpos,
                                          int nh_packed, int hd_packed,
                                          int max_pos, float eps) {
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int head = blockIdx.x;          // grid (nh, m)
    const int m = gridDim.y;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    const int slot = slotpos[0 * 2 + 0];  // one slot for the whole group
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    for (int r = 0; r < m; r++) {
        const int p = slotpos[r * 2 + 1];
        const float* c = cossin + (long long)r * 2 * rdim;
        const float* s = c + rdim;
        const __half* qn_w = qnkn;
        const __half* kn_w = qnkn + hd;
        // ---- q: rmsnorm (1+w), rope (shared order with xq_attn_decode)
        float qv = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + tid]);
        red[tid] = qv * qv; __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
        qbuf[tid] = qv; __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        }
        __syncthreads();
        qv = xq_kv_rot(qbuf[tid], xq_kv_fmt(max_pos), tid & 31);   // WP25 q8: rotated basis
        // ---- k: rmsnorm (1+w), rope, cache write (bit-identical K words)
        xq_k_norm_rope(kbuf, red, kp + (long long)r * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
        xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)r * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
        // ---- shared two-pass softmax over [0..p] (rows <= r written above)
        float acc, denom;
        xq_attn_softmax(red, qv, kc, p, tid, hd, acc, denom);
        float g = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + hd + tid]);
        const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
        attn[(long long)r * nh * hd + (long long)head * hd + tid] =
            xq_f2h(o * xq_sig(g));
        __syncthreads();
    }
}

// ---- S-A3-m attention v2: one block per (head, row) — grid (nh, m), blockIdx.y
// = row. (The S-A3-l body ignored blockIdx.y: every y-block recomputed ALL rows,
// m-fold redundant.) Block r writes the K/V of rows 0..r itself (identical words
// to every other block writing them — benign) so its own scan only reads cache
// words it wrote before a __syncthreads; then row r's q and ONE scan over [0..p_r]
// through xq_attn_scan (bit-identical to xq_attn_softmax — row r == the decode of
// that token, probe-exl3-binv attention section). Phase-1 order per row is the
// old body verbatim.
__device__ void xq_attn_decode_group_v2(__half* __restrict__ attn,
                                        const __half* __restrict__ qg,
                                        const __half* __restrict__ kp,
                                        const __half* __restrict__ vp,
                                        float* __restrict__ kcvc,
                                        const __half* __restrict__ qnkn,
                                        const float* __restrict__ cossin,
                                        const int* __restrict__ slotpos,
                                        int nh_packed, int hd_packed,
                                        int max_pos, float eps) {
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int head = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    const int slot = slotpos[0 * 2 + 0];  // one slot for the whole group
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    __shared__ float wbuf[XQ_SCAN_T];
    __shared__ float wm[32];
    // ---- phase 1: K/V writes for rows 0..r (xq_k_norm_rope = the shared K words)
    for (int rr = 0; rr <= r; rr++) {
        const int p = slotpos[rr * 2 + 1];
        const float* c = cossin + (long long)rr * 2 * rdim;
        const float* s = c + rdim;
        xq_k_norm_rope(kbuf, red, kp + (long long)rr * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)rr * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
    }
    // ---- row r's q: rmsnorm (1+w), rope (shared order with xq_attn_decode)
    const int p = slotpos[r * 2 + 1];
    const float* c = cossin + (long long)r * 2 * rdim;
    const float* s = c + rdim;
    float qv = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + tid]);
    red[tid] = qv * qv; __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
    qbuf[tid] = qv; __syncthreads();
    if (tid < rdim / 2) {
        float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
        qbuf[tid] = x1 * c[tid] - x2 * s[tid];
        qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
    if (kc.fmt >= XQ_KV_Q8) {   // WP25 q8: q into the rotated basis (own element; then publish)
        qbuf[tid] = xq_h32r(qbuf[tid], tid & 31);
        __syncthreads();
    }
    float acc, denom;
    xq_attn_scan_hd(qbuf, kc, p, tid, hd, wbuf, wm, acc, denom);
    float g = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
    attn[(long long)r * nh * hd + (long long)head * hd + tid] =
        xq_f2h(o * xq_sig(g));
}

extern "C" __global__ void xq_attn_decode_group(__half* __restrict__ attn,
                                                const __half* __restrict__ qg,
                                                const __half* __restrict__ kp,
                                                const __half* __restrict__ vp,
                                                float* __restrict__ kcvc,
                                                const __half* __restrict__ qnkn,
                                                const float* __restrict__ cossin,
                                                const int* __restrict__ slotpos,
                                                int nh_packed, int hd_packed,
                                                int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int hd = hd_packed & 0xFFFF;
    if (hd == 128 || hd == 256 || hd == 512) {
        xq_attn_decode_group_v2(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                                nh_packed, hd_packed, max_pos, eps);
    } else if (blockIdx.y == 0) {
        // other head dims: the verbatim per-row loop, once per head
        xq_attn_decode_group_loop(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                                  nh_packed, hd_packed, max_pos, eps);
    }
}

// Probe-only reference (probe-exl3-binv attention section): the pre-S-A3-l
// per-row loop over xq_attn_softmax — a different CLASS (smem sync-tree) from the
// v2 warp-tree scan. Launch grid (nh, m): the loop takes m = gridDim.y and every
// y-block runs all rows (identical writes).
extern "C" __global__ void xq_attn_decode_group_ref(__half* __restrict__ attn,
                                                    const __half* __restrict__ qg,
                                                    const __half* __restrict__ kp,
                                                    const __half* __restrict__ vp,
                                                    float* __restrict__ kcvc,
                                                    const __half* __restrict__ qnkn,
                                                    const float* __restrict__ cossin,
                                                    const int* __restrict__ slotpos,
                                                    int nh_packed, int hd_packed,
                                                    int max_pos, float eps) {
    XQ_PDL_ENTRY();
    xq_attn_decode_group_loop(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                              nh_packed, hd_packed, max_pos, eps);
}

// =====================================================================================
// WP13 (PLAN/SURPASS_PLAN_2026-09-26): dense-regime attention v3 — the same numbers as
// xq_attn_decode_group (v2) / xq_attn_decode, bit for bit, at a fraction of the cost.
//
// v2 runs one CTA per (q head, row) and each CTA scans [0..p] TWICE (max pass, weight
// pass), recomputing every q.k dot and re-reading every K row 12x (GQA 12) and V row
// 12x. v3 splits the same arithmetic into three launches:
//   1. xq_attn_dense_prep  grid (m*nh):  row K/V cache write (the (row, kvh) CTA with
//      head % gqa == 0; xq_k_norm_rope + xq_kv_put, the shared cache-word code) and the
//      row's roped q (v2's q expressions verbatim) staged f32 -> qstage.
//   2. xq_attn_dense_dots  grid (nchg, nkv, m): ONE K-row load feeds all 12 heads' dots;
//      raw dots stored to a score plane S[row][head][t], per-64-position chunk maxima of
//      fl(raw*scale) to pmx[row][head][chunk].
//   3. xq_attn_dense_acc{1,4} grid (hd/(16*DPT), nkv, m): mx = fmaxf of the chunk maxima,
//      w = __expf(fma(raw, scale, -mx)) per tile, then the SEQUENTIAL ascending-t
//      denom/acc chains with V tiles staged by double-buffered cp.async.
//
// Bit contract (row r == v2 row r == xq_attn_decode of that token):
//  * K/V words + q: the same source expressions (shared xq_k_norm_rope / xq_kv_put, the
//    q block copied verbatim from v2), so the same PTX FMA pattern (rope: x1*c - x2*s is
//    mul/mul/sub, fused by ptxas identically in every copy — checked in SASS).
//  * raw dot: xq_dot_tree_kv's halving tree (dims d, d+128 / d+64 / d+32 in-register, then
//    lanes L, L+16 .. L+1) with __fmul_rn/__fadd_rn. The cross-lane half is done through a
//    per-warp smem transpose (lane L owns dot L and sums a[x] += a[x+s], s = 16..1 — the
//    same pairs as shfl_down's lane-0 value), so no FMA contraction is possible anywhere.
//  * mx: fmaxf over {__fmul_rn(raw, scale)} = SASS FMUL + FMNMX of v2 pass 1 (fmaxf is
//    exact and order-free, so chunking it changes nothing).
//  * w: v2's `__expf(raw * scale - mx)` compiles to FFMA(scale, raw, -mx) -> FMUL log2e ->
//    MUFU.EX2 in every copy (SASS, sm_121). Pinned here as __expf(__fmaf_rn(raw, scale,
//    -mx)) — never a CSE'd `s = raw*scale` (that would round twice).
//  * denom += w (FADD), acc = fma(w, v, acc) (FFMA), t ascending from 0.0f, then
//    f16((acc / denom) * sig(g)) — v2's closing expression.
// Graph contract: every grid depends only on m (fixed per captured graph); the position
// range comes from slotpos on device. The score plane holds XQ_DV3_TS positions — the host
// only takes this path where the dense regime is bounded by the QSA switch (positions
// <= qsa_limit = 2051 < 2304); a longer row would be a host bug, so acc writes NaN (loud)
// instead of silently dropping positions. KV formats f32/f16/fp8 via the XqKv helpers.
// =====================================================================================
#define XQ_DV3_TS  2304                      // score-plane row stride (positions): 9 x 256
#define XQ_DV3_CH  64                        // dots chunk: 8 warps x 8 positions
#define XQ_DV3_NCH (XQ_DV3_TS / XQ_DV3_CH)   // 36 chunk-max slots per (row, head)
#define XQ_DV3_G   12                        // q heads per kv head: the W<=2 / TP=1 instantiation
#define XQ_DV3_G6  6                         // TP-4G: the W=4 / TP=4 instantiation (6 heads per kv head)
#define XQ_DV3_T   64                        // accumulate tile (positions)

// The softmax scale exactly as v2 gets it: rsqrt.approx of the RUNTIME head dim (MUFU.RSQ
// with the non-ftz guard). The v3 bodies know hd == 256 after their geometry guard, and
// ptxas then folds rsqrt.approx(256.f) to the literal 0.0625 (seen in SASS; an empty asm
// barrier does not stop ptxas) — equal to MUFU.RSQ(256) only if the SFU is exact there.
// A volatile shared round trip is a value no compiler stage may assume, so the same
// instruction runs on the same input. Block-wide: every thread of the block must call it.
__device__ __forceinline__ float xq_dv3_scale(int hd) {
    __shared__ int xq_dv3_hd;
    if (threadIdx.x == 0) xq_dv3_hd = hd;
    __syncthreads();
    const int hv = *reinterpret_cast<volatile int*>(&xq_dv3_hd);
    return rsqrtf((float)hv);
}

// ---- 1. prep: grid (m*nh), block hd. Block (row, head).
extern "C" __global__ void xq_attn_dense_prep(float* __restrict__ qstage, float* __restrict__ kcvc,
                                              const __half* __restrict__ qg, const __half* __restrict__ kp,
                                              const __half* __restrict__ vp, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, const int* __restrict__ slotpos,
                                              int nh_packed, int hd_packed, int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int row = blockIdx.x / nh;
    const int head = blockIdx.x % nh;
    const int gqa = nh / nkv;
    const int tid = threadIdx.x;
    const int slot = slotpos[row * 2 + 0];
    const int p = slotpos[row * 2 + 1];
    const float* c = cossin + (long long)row * 2 * rdim;
    const float* s = c + rdim;
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    // ---- K/V: v2 phase-1 body for this row, once per (row, kv head) (block-uniform branch)
    if (head % gqa == 0) {
        const int kvh = head / gqa;
        xq_k_norm_rope(kbuf, red, kp + (long long)row * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
        xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)row * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
    }
    // ---- q: rmsnorm (1+w), rope — v2's q block verbatim; f32 staging
    float qv = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + tid]);
    red[tid] = qv * qv; __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
    qbuf[tid] = qv; __syncthreads();
    if (tid < rdim / 2) {
        float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
        qbuf[tid] = x1 * c[tid] - x2 * s[tid];
        qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
    // q8 (WP25): q into the cache's rotated basis exactly as v2 / xq_attn_decode / xq_attn_prep
    // do it (xq_h32r of the roped element; identity — no op emitted — for f32/f16/fp8).
    qstage[((long long)row * nh + head) * hd + tid] = xq_kv_rot(qbuf[tid], xq_kv_fmt(max_pos), tid & 31);
}

// ---- 2. dots: grid (nchg, nkv, m), block 256 (8 warps). Block (chunk-stride, kv head, row);
// chunks ch = blockIdx.x, +gridDim.x, ... < ceil((p+1)/64). Warp w owns positions
// ch*64 + w*8 + [0,8): their K rows (E=8 dims per lane: lane + 32j) sit in registers and
// feed all 12 heads. Per group of 4 heads the 8x4 = 32 lane-partials go through the
// warp's [32][33] transpose tile and lane L finishes dot L = (position L>>2, head L&3).
// TP-4G: the body is templated on G (q heads per kv head; 12 = W<=2 / TP=1, 6 = W=4 / TP=4).
// Heads are processed in groups of 4 (NHG = ceil(G/4)); at G=6 the second group holds 2 valid
// heads and its lanes for the other 2 are computed on stale tile rows and discarded (never
// stored, never in the max). Every head's dot is the same in-register + cross-lane halving
// tree at any G, so a head's bits do not depend on G. At G=12 (PARTIAL = false) every guard
// below is compiled out (SASS identical to the pre-TP-4G kernel up to register naming).
template <int G>
__device__ __forceinline__ void xq_dv3_dots_body(float* __restrict__ sc, float* __restrict__ pmx,
                                                 const float* __restrict__ qstage,
                                                 const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                                                 int nh_packed, int hd_packed, int max_pos) {
    constexpr int E = 8;                         // hd 256 (host-gated)
    constexpr int NHG = (G + 3) / 4;
    constexpr bool PARTIAL = (G % 4) != 0;       // the last 4-head group is short (G=6); false at G=12
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    if (hd != 32 * E || nh != G * nkv) return;   // host-gated; never a partial write
    const int kvh = blockIdx.y;
    const int row = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = slotpos[row * 2 + 0];
    const int npos = min(slotpos[row * 2 + 1] + 1, XQ_DV3_TS);
    const int nch = (npos + XQ_DV3_CH - 1) / XQ_DV3_CH;
    if ((int)blockIdx.x >= nch) return;          // block-uniform
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const float scale = xq_dv3_scale(hd);
    __shared__ __align__(16) float qs[G * 32 * E];   // the kv group's 12 roped q rows (12 KB)
    __shared__ float tb[8][32][33];                   // per-warp transpose tiles
    __shared__ float wmx[8][G];
    const long long hbase = (long long)row * nh + (long long)kvh * G;   // (row, first head of group)
    {
        const float4* qsrc = reinterpret_cast<const float4*>(qstage + hbase * hd);
        for (int i = threadIdx.x; i < G * hd / 4; i += blockDim.x)
            reinterpret_cast<float4*>(qs)[i] = qsrc[i];
    }
    __syncthreads();
    float* srow = sc + hbase * XQ_DV3_TS;
    for (int ch = blockIdx.x; ch < nch; ch += gridDim.x) {
        const int t0 = ch * XQ_DV3_CH + warp * 8;
        float kr[8][E];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int t = t0 + i;
            if (t < npos) {
                const char* krow = kc.k + (long long)t * kc.rb;
                #pragma unroll
                for (int j = 0; j < E; j++) kr[i][j] = xq_kv_ld(krow, kc.fmt, hd, lane + 32 * j);
            } else {
                #pragma unroll
                for (int j = 0; j < E; j++) kr[i][j] = 0.0f;
            }
        }
        #pragma unroll
        for (int hg = 0; hg < NHG; hg++) {
            #pragma unroll
            for (int hh = 0; hh < 4; hh++) {
                if (PARTIAL && hg * 4 + hh >= G) continue;   // skipped rows are never stored / maxed
                const float* qh = qs + (hg * 4 + hh) * (32 * E);
                float qv[E];
                #pragma unroll
                for (int j = 0; j < E; j++) qv[j] = qh[lane + 32 * j];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v[E];
                    #pragma unroll
                    for (int j = 0; j < E; j++) v[j] = __fmul_rn(qv[j], kr[i][j]);
                    #pragma unroll
                    for (int h2 = E / 2; h2 >= 1; h2 >>= 1) {
                        #pragma unroll
                        for (int j = 0; j < h2; j++) v[j] = __fadd_rn(v[j], v[j + h2]);
                    }
                    tb[warp][i * 4 + hh][lane] = v[0];
                }
            }
            __syncwarp();
            // lane L owns dot L: the cross-lane halving tree over the 32 lane partials
            float a[32];
            #pragma unroll
            for (int x = 0; x < 32; x++) a[x] = tb[warp][lane][x];
            #pragma unroll
            for (int s2 = 16; s2 >= 1; s2 >>= 1) {
                #pragma unroll
                for (int x = 0; x < s2; x++) a[x] = __fadd_rn(a[x], a[x + s2]);
            }
            const int t = t0 + (lane >> 2);
            const int h = hg * 4 + (lane & 3);
            float mv = -INFINITY;
            if (t < npos && (!PARTIAL || h < G)) {
                srow[(long long)h * XQ_DV3_TS + t] = a[0];
                mv = __fmul_rn(a[0], scale);
            }
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 4));
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 8));
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 16));
            if (lane < 4 && (!PARTIAL || hg * 4 + lane < G)) wmx[warp][hg * 4 + lane] = mv;
            __syncwarp();                         // tb is rewritten by the next head group
        }
        __syncthreads();
        if (threadIdx.x < G) {
            float mm = -INFINITY;
            #pragma unroll
            for (int w = 0; w < 8; w++) mm = fmaxf(mm, wmx[w][threadIdx.x]);
            pmx[(hbase + threadIdx.x) * XQ_DV3_NCH + ch] = mm;
        }
        __syncthreads();                          // wmx reuse by the next chunk
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
xq_attn_dense_dots(float* __restrict__ sc, float* __restrict__ pmx, const float* __restrict__ qstage,
                   const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                   int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_dots_body<XQ_DV3_G>(sc, pmx, qstage, kcvc, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(256, 2)
xq_attn_dense_dots_g6(float* __restrict__ sc, float* __restrict__ pmx, const float* __restrict__ qstage,
                      const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                      int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_dots_body<XQ_DV3_G6>(sc, pmx, qstage, kcvc, slotpos, nh_packed, hd_packed, max_pos);
}

// Stage V rows [t0, t0+T) of this CTA's dim slice (raw cache bytes, SD*eb per row; fp8 also
// the 16-B chunk holding the row scale; q8 the 16-B chunk holding the row's hd/32 f16 group
// scales) into one smem buffer as ONE cp.async group. RS = smem row stride (bytes).
template <int SD, int N16, bool FP8, int RS = SD * 4>
__device__ __forceinline__ void xq_dv3_stage_v(char* vbt, float* vsct, const XqKv& kc, int t0,
                                               int npos, int d0b, int tid) {
    constexpr int NPER = N16 + (FP8 ? 1 : 0);
    for (int i = tid; i < XQ_DV3_T * NPER; i += blockDim.x) {
        const int tr = i / NPER;
        const int c = i - tr * NPER;
        const int t = t0 + tr;
        if (t < npos) {
            const char* src = kc.v + (long long)t * kc.rb;
            if (c < N16) moe_coop_cp16(vbt + tr * RS + c * 16, src + d0b + c * 16, 16);
            else         moe_coop_cp16(vsct + tr * 4, src + kc.hd, 16);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// V elements d0..d0+DPT-1 (thread dim-group dgl) at smem row tr, format FMT: the same values
// xq_kv_ld returns (f16 -> f32 exact; fp8 float(e4m3) * row scale, one FMUL).
template <int FMT, int DPT, int SD>
__device__ __forceinline__ void xq_dv3_ldv(float* v, const char* vbt, const float* vsct, int tr, int dgl) {
    const char* r = vbt + tr * (SD * 4);
    if constexpr (FMT == XQ_KV_F32) {
        if constexpr (DPT == 4) {
            const float4 f = *reinterpret_cast<const float4*>(r + dgl * 16);
            v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = reinterpret_cast<const float*>(r)[dgl * DPT + e];
        }
    } else if constexpr (FMT == XQ_KV_F16) {
        if constexpr (DPT == 4) {
            const uint2 u = *reinterpret_cast<const uint2*>(r + dgl * 8);
            const __half* hh = reinterpret_cast<const __half*>(&u);
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = __half2float(hh[e]);
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = __half2float(reinterpret_cast<const __half*>(r)[dgl * DPT + e]);
        }
    } else {
        const float scv = vsct[tr * 4];
        if constexpr (DPT == 4) {
            const unsigned u = *reinterpret_cast<const unsigned*>(r + dgl * 4);
            const __nv_fp8_e4m3* e8 = reinterpret_cast<const __nv_fp8_e4m3*>(&u);
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = float(e8[e]) * scv;
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = float(reinterpret_cast<const __nv_fp8_e4m3*>(r)[dgl * DPT + e]) * scv;
        }
    }
}

// q8 (WP25) V elements d0..d0+DPT-1 at smem row tr (codes packed RS bytes per row, the row's
// 16-B group-scale chunk at vsct + tr*4): the SAME values xq_kv_ld / xq_kv_ld8 return —
// sm = f16 scale of group g * 2^-7 (xq_q8_sm), hs = 0.5*sm, v = xq_q8v(code). DPT in {2, 4}
// (never crosses a 32-group: d0 % DPT == 0).
template <int DPT, int RS>
__device__ __forceinline__ void xq_dv3_ldv_q8(float* v, const char* vbt, const float* vsct, int tr, int dgl, int g) {
    static_assert(DPT == 2 || DPT == 4, "q8 dense v3 slices hold whole 32-groups");
    const char* r = vbt + tr * RS;
    const float sm = __fmul_rn(__half2float(reinterpret_cast<const __half*>(vsct + tr * 4)[g]), 0.0078125f);
    const float hs = 0.5f * sm;
    unsigned u;
    if constexpr (DPT == 4) u = *reinterpret_cast<const unsigned*>(r + dgl * 4);
    else                    u = (unsigned)*reinterpret_cast<const unsigned short*>(r + dgl * 2);
    #pragma unroll
    for (int e = 0; e < DPT; e++) v[e] = xq_q8v(u, e, sm, hs);
}

// ---- 3. accumulate: grid (hd/SD, nkv, m), block 192 = 12 heads x 16 dim-groups of DPT dims.
// Thread (h = tid>>4, dgl = tid&15) owns dims d0..d0+DPT-1 of head kvh*12+h.
template <int G, int FMT, int DPT>
__device__ __forceinline__ void xq_dv3_acc_body(__half* __restrict__ attn, const float* __restrict__ sc,
                                                const float* __restrict__ pmx, const XqKv& kc,
                                                const __half* __restrict__ qg, int nh, int hd,
                                                int kvh, int row, int npos, char* vb, float* vsc, float* wb) {
    constexpr int T = XQ_DV3_T, SD = 16 * DPT, WST = XQ_DV3_T + 4;
    constexpr int EB = FMT == XQ_KV_F32 ? 4 : (FMT == XQ_KV_F16 ? 2 : 1);
    constexpr int N16 = SD * EB / 16;
    // q8 (WP25): codes packed SD bytes per smem row (so the DPT=2 body fits acc1's buffer);
    // f32/f16/fp8 keep the SD*4-byte stride (their code is unchanged). SCH = the row carries a
    // 16-B scale chunk (fp8 row scale | q8 hd/32 f16 group scales, hd == 256).
    constexpr bool Q8 = FMT >= XQ_KV_Q8;
    constexpr int RS = Q8 ? SD : SD * 4;
    constexpr bool SCH = FMT == XQ_KV_FP8 || Q8;
    static_assert(!Q8 || DPT >= 2, "q8: a thread's 16-dgl slice must hold whole 32-groups");
    const int tid = threadIdx.x;
    const int h = tid >> 4;
    const int dgl = tid & 15;
    const int d0 = blockIdx.x * SD + dgl * DPT;
    const long long hrow = (long long)row * nh + (long long)kvh * G + h;
    const float* srow = sc + hrow * XQ_DV3_TS;
    const float scale = xq_dv3_scale(hd);
    const int ntile = (npos + T - 1) / T;
    const int d0b = blockIdx.x * SD * EB;
    // tile 0's V copies and raw scores go out first so their latency overlaps the mx loads
    xq_dv3_stage_v<SD, N16, SCH, RS>(vb, vsc, kc, 0, npos, d0b, tid);
    float4 rw = *reinterpret_cast<const float4*>(srow + dgl * 4);
    // mx: fmaxf over the row's chunk maxima (exactly the set {fl(raw*scale) : t < npos})
    const int nch = (npos + XQ_DV3_CH - 1) / XQ_DV3_CH;
    float mx = -INFINITY;
    for (int c = dgl; c < nch; c += 16) mx = fmaxf(mx, pmx[hrow * XQ_DV3_NCH + c]);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    float acc[DPT];
    #pragma unroll
    for (int e = 0; e < DPT; e++) acc[e] = 0.0f;
    float denom = 0.0f;
    for (int ti = 0; ti < ntile; ti++) {
        const int t0 = ti * T;
        // weights of tile ti (positions t0 + dgl*4 + [0,4) of head h); t >= npos never read
        {
            float4 w4;
            w4.x = __expf(__fmaf_rn(rw.x, scale, -mx));
            w4.y = __expf(__fmaf_rn(rw.y, scale, -mx));
            w4.z = __expf(__fmaf_rn(rw.z, scale, -mx));
            w4.w = __expf(__fmaf_rn(rw.w, scale, -mx));
            *reinterpret_cast<float4*>(wb + h * WST + dgl * 4) = w4;
        }
        char* vcur = vb + (ti & 1) * (T * RS);
        float* scur = vsc + (ti & 1) * (T * 4);
        if (ti + 1 < ntile) {
            xq_dv3_stage_v<SD, N16, SCH, RS>(vb + ((ti + 1) & 1) * (T * RS),
                                                      vsc + ((ti + 1) & 1) * (T * 4),
                                                      kc, t0 + T, npos, d0b, tid);
            rw = *reinterpret_cast<const float4*>(srow + t0 + T + dgl * 4);
        } else {
            asm volatile("cp.async.commit_group;\n" ::: "memory");   // keep the group count aligned
        }
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");       // tile ti's copies (this thread)
        __syncthreads();                                              // ... and everyone's, + wb
        const int tn = min(T, npos - t0);
        const float* wh = wb + h * WST;
        int t = 0;
        for (; t + 4 <= tn; t += 4) {
            const float4 w4 = *reinterpret_cast<const float4*>(wh + t);
            const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                float v[DPT];
                if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t + u, dgl, d0 >> 5);
                else              xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t + u, dgl);
                denom = __fadd_rn(denom, ws[u]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(ws[u], v[e], acc[e]);
            }
        }
        for (; t < tn; t++) {
            const float w = wh[t];
            float v[DPT];
            if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t, dgl, d0 >> 5);
            else              xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t, dgl);
            denom = __fadd_rn(denom, w);
            #pragma unroll
            for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(w, v[e], acc[e]);
        }
        __syncthreads();                                              // vcur / wb reuse
    }
    const int head = kvh * G + h;
    if constexpr (Q8) {
        // q8: the output leaves the rotated basis exactly as v2's xq_kv_rot(acc / denom) does:
        // xq_h32's butterfly on the 32-group (element j = (d0 & 31) + e), stage s pairs j, j^s
        // (low a+b, high partner-minus-self; __fadd_rn/__fsub_rn), then one __fmul_rn by
        // 1/sqrt(32). Stages s < DPT are in-register; s >= DPT pair thread dgl with
        // dgl ^ (s/DPT) - same warp, same head (DPT >= 2 keeps s/DPT <= 8 < 16).
        float o[DPT];
        #pragma unroll
        for (int e = 0; e < DPT; e++) o[e] = acc[e] / denom;
        #pragma unroll
        for (int st = 1; st < 32; st <<= 1) {
            if (st < DPT) {
                float n[DPT];
                #pragma unroll
                for (int e = 0; e < DPT; e++) n[e] = (e & st) ? __fsub_rn(o[e ^ st], o[e]) : __fadd_rn(o[e], o[e ^ st]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) o[e] = n[e];
            } else {
                const int tm = st / DPT;
                const bool hi = (dgl & tm) != 0;
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    const float pv = __shfl_xor_sync(0xffffffffu, o[e], tm);
                    o[e] = hi ? __fsub_rn(pv, o[e]) : __fadd_rn(o[e], pv);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            const int d = d0 + e;
            const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
            attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h(__fmul_rn(o[e], XQ_R32) * xq_sig(g));
        }
    } else {
    #pragma unroll
    for (int e = 0; e < DPT; e++) {
        const int d = d0 + e;
        const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
        attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h((acc[e] / denom) * xq_sig(g));
    }
    }
}

template <int G, int DPT>
__device__ __forceinline__ void xq_dv3_acc_entry(__half* __restrict__ attn, const float* __restrict__ sc,
                                                 const float* __restrict__ pmx, const float* __restrict__ kcvc,
                                                 const __half* __restrict__ qg, const int* __restrict__ slotpos,
                                                 int nh_packed, int hd_packed, int max_pos) {
    constexpr int SD = 16 * DPT;
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    if (hd != 256 || nh != G * nkv) return;   // host-gated
    const int kvh = blockIdx.y;
    const int row = blockIdx.z;
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;
    __shared__ __align__(16) char vb[2 * XQ_DV3_T * SD * 4];
    __shared__ __align__(16) float vsc[2 * XQ_DV3_T * 4];
    __shared__ __align__(16) float wb[G * (XQ_DV3_T + 4)];
    if (npos > XQ_DV3_TS) {
        // host bug tripwire: the score plane cannot hold this row — loud, never truncated
        const int h = threadIdx.x >> 4, dgl = threadIdx.x & 15;
        #pragma unroll
        for (int e = 0; e < DPT; e++)
            attn[(long long)row * nh * hd + (long long)(kvh * G + h) * hd + blockIdx.x * SD + dgl * DPT + e] =
                __float2half(__int_as_float(0x7fc00000));
        return;
    }
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    switch (kc.fmt) {
        case XQ_KV_F32: xq_dv3_acc_body<G, XQ_KV_F32, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_F16: xq_dv3_acc_body<G, XQ_KV_F16, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_FP8: xq_dv3_acc_body<G, XQ_KV_FP8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        default:
            // q8 / probe-only q4x (same layout + reader). The output un-rotation needs a whole
            // 32-group per thread-16 slice, so the single-row acc1 (DPT=1, 16-dim slices) runs
            // the DPT=2 body on its first hd/32 CTAs (32-dim slices; the rest exit) - the
            // grid (and so every captured graph) is unchanged. RS = SD bytes keeps the DPT=2
            // body inside acc1's static vb buffer (2*64*32 B <= 2*64*16*4 B).
            if constexpr (DPT >= 2) {
                xq_dv3_acc_body<G, XQ_KV_Q8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            } else {
                if ((int)blockIdx.x < hd / 32)   // block-uniform
                    xq_dv3_acc_body<G, XQ_KV_Q8, 2>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            }
            break;
    }
}

// DPT=4 (64-dim slices, grid (4, nkv, m)) for verify widths; DPT=1 (16-dim slices, grid
// (16, nkv, m)) for the single-row decode step, where 4x the CTAs beats 4x the chains.
extern "C" __global__ void __launch_bounds__(192)
xq_attn_dense_acc4(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                   const float* __restrict__ kcvc, const __half* __restrict__ qg,
                   const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(192)
xq_attn_dense_acc1(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                   const float* __restrict__ kcvc, const __half* __restrict__ qg,
                   const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
// TP-4G: G=6 twins (block 16*6 = 96). Same body, same per-thread chains; only the head count of
// the CTA differs, so each (head, dim) thread runs the identical ascending-t fma chain.
extern "C" __global__ void __launch_bounds__(96)
xq_attn_dense_acc4_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G6, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(96)
xq_attn_dense_acc1_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G6, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}

// WP13 XCHECK (GB10_EXL3_DENSE_XCHECK=1): bitwise diff of the v3 output vs the old kernel's
// output (re-run into a side buffer). One block of 256. st[0] = checks, st[1] = total
// mismatching halves. Prints every mismatch event and a heartbeat every 1024 checks.
extern "C" __global__ void xq_dv3_xcheck(const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
                                         int n, int* __restrict__ st, int tag) {
    XQ_PDL_ENTRY();
    __shared__ int cnt, first;
    if (threadIdx.x == 0) { cnt = 0; first = 0x7fffffff; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (a[i] != b[i]) { atomicAdd(&cnt, 1); atomicMin(&first, i); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int calls = atomicAdd(&st[0], 1);
        if (cnt) {
            const int tot = atomicAdd(&st[1], cnt) + cnt;
            printf("[dense-v3-xcheck] MISMATCH tag %d (layer %d, m %d, %s): %d of %d halves differ, "
                   "first idx %d new %04x old %04x (total %d)\n", tag, tag & 0xFF, (tag >> 8) & 0xFF,
                   (tag >> 16) ? "verify" : "step", cnt, n, first, (unsigned)a[first], (unsigned)b[first], tot);
        } else if ((calls & 1023) == 0) {
            printf("[dense-v3-xcheck] %d checks, %d mismatching halves so far\n", calls + 1, st[1]);
        }
    }
}

// ---- chained MoE-router-free? no — router/hc/MoE are row-independent (m-batched
// already). The remaining stateful ops: GDN conv ring + recurrent state, the PLE
// conv ring. Verify runs those against SHADOW buffers (xq_conv1d_chunk /
// xq_gdn_step_chunk / xq_ple_conv_chunk with slot=0). Commit = bit-identical
// replay of rows 0..a on the LIVE state, with C read from the device accept
// record slotacc[2] = {slot, a} (C = a+1). When a == m-1 the replay reproduces
// the shadow result exactly — the commit is then redundant but harmless; the
// launcher skips it in that case only if it knows a (it does not — a is
// device-side), so the replay always runs: <= 4 rows, sub-ms.

// GDN commit: xq_gdn_step_chunk with C = slotacc[1] + 1 (device-read).
extern "C" __global__ void xq_gdn_commit(__half* __restrict__ core,
                                         const __half* __restrict__ qkv,
                                         float* __restrict__ state,
                                         const __half* __restrict__ b_in,
                                         const __half* __restrict__ a_in,
                                         int nh, int n_k_heads, int kd, int vd,
                                         const float* __restrict__ a_log,
                                         const float* __restrict__ dt_bias,
                                         const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    int blk = blockIdx.x;
    int nchunk = vd / XQ_GDN_C;
    const int qkv_stride = 2 * n_k_heads * kd + nh * vd;
    int chunk = blk % nchunk; blk /= nchunk;
    int head = blk % nh;
    int key_head = head * n_k_heads / nh;
    int key_dim = n_k_heads * kd;
    int bb0 = chunk * XQ_GDN_C;
    extern __shared__ float sh[];
    float* S_sh = sh;
    float* Srow = S_sh + kd * XQ_GDN_SP;
    float* kv_mem = Srow + kd;
    float* vbuf = kv_mem + XQ_GDN_C;
    float* delta = vbuf + XQ_GDN_C;
    float* qrow = delta + XQ_GDN_C;
    float* krow = qrow + kd;
    float* S = state + ((long long)slot * nh + head) * kd * vd;
    xq_gdn_ld_state(S_sh, S, kd, vd, bb0);
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const __half* col = qkv + (long long)t * qkv_stride;
        float beta = 1.0f / (1.0f + __expf(-xq_h2f(b_in[(long long)t * nh + head])));
        float sp = xq_h2f(a_in[(long long)t * nh + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        float gt = __expf(-__expf(a_log[head]) * sp);
        xq_gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                     core + (long long)t * (nh * vd) + (long long)head * vd,
                     key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
        __syncthreads();
    }
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// S-A3-o G1: GDN commit from the verify's rank-1 ring. xq_gdn_token's state update
// per token is  S *= g;  S += krow ⊗ delta  (delta = (v - Sᵀk)·β uses the PRE-update
// state — computed once by the verify, which runs the same device function on the
// same live state and inputs). Replaying it only needs (g, krow, delta) per token, so
// the commit skips both norm reductions, the km matvec and the q readout, and holds
// each state row in registers. Per element the op sequence is identical to
// xq_gdn_token's (FMUL by g, then FFMA kk·delta + S), so the committed state is
// bitwise the replay's. grid (nh * vd/XQ_GDN_C), block kd; thread = one state row.
#define XQ_GDN_RING_CMAX 8   // WP23: mirrors GDN_RING_CMAX in exl3_forward.rs
extern "C" __global__ void __launch_bounds__(128)
xq_gdn_commit_ring(float* __restrict__ state, const float* __restrict__ ring,
                   int nh, int kd, int vd, const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    // S-A3-o G2: lane = column (warp loads/stores are 128 B coalesced), warp w owns
    // rows w, w+4, ... (kd/4 per thread, in registers); krow/g of all C tokens staged
    // in smem once. Per element the ops are unchanged (FMUL g, FFMA kk·delta + s).
    // WP23: extent 6 -> XQ_GDN_RING_CMAX = 8 rows (draft depth 7: C = a+1 <= 8; host gate
    // MTP_MAX_K+1 <= GDN_RING_CMAX). C only bounds the loops, so C <= 6 runs the same ops.
    __shared__ float kr_sh[XQ_GDN_RING_CMAX][128];
    __shared__ float g_sh[XQ_GDN_RING_CMAX];
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    // device guard: an accept count past the extent would overrun kr_sh — fail loudly, never
    // commit a truncated state (block-uniform: C is one device value for the whole launch)
    if (C < 1 || C > XQ_GDN_RING_CMAX) __trap();
    const int nchunk = vd / XQ_GDN_C;
    const int chunk = blockIdx.x % nchunk;
    const int head = blockIdx.x / nchunk;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = chunk * XQ_GDN_C + lane;
    for (int i = threadIdx.x; i < C * kd; i += blockDim.x) {
        const int t = i / kd, r = i - t * kd;
        kr_sh[t][r] = ring[((long long)t * nh + head) * XQ_GDN_RS + vd + r];
    }
    if (threadIdx.x < C) g_sh[threadIdx.x] = ring[((long long)threadIdx.x * nh + head) * XQ_GDN_RS + vd + kd];
    constexpr int RPT = 128 / 4;                 // rows per thread (kd = 128, 4 warps)
    float* S = state + ((long long)slot * nh + head) * kd * vd + col;
    float s[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) s[i] = S[(long long)(warp + 4 * i) * vd];
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const float gt = g_sh[t];
        const float d = __ldg(ring + ((long long)t * nh + head) * XQ_GDN_RS + col);
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] *= gt;
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = __fmaf_rn(kr_sh[t][warp + 4 * i], d, s[i]);
    }
    #pragma unroll
    for (int i = 0; i < RPT; i++) S[(long long)(warp + 4 * i) * vd] = s[i];
}

// GDN conv1d commit: replay the state window over rows 0..C-1 using the RAW
// (pre-conv) projections saved by the verify chunk (xq_conv1d_chunk overwrites
// qkv in place, so the post-conv silu values must NOT be shifted in). C =
// slotacc[1] + 1 (device-read). x rows are left untouched — they already hold
// the conv outputs the GDN commit consumes.
extern "C" __global__ void xq_conv_commit(float* __restrict__ state,
                                          const float* __restrict__ w, int conv_dim, int k,
                                          const __half* __restrict__ raw, int raw_off,
                                          const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(raw[raw_off + (long long)t * conv_dim + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
    }
    for (int j = 0; j < k; j++) sbase[j] = st[j];
}

// PLE ring commit: roll-only replay (xq_ple_conv_chunk's ring half; the conv
// add is NOT redone — resid rows 0..a already carry it). C = slotacc[1] + 1.
extern "C" __global__ void xq_ple_ring_commit(const __half* __restrict__ normed,
                                              __half* __restrict__ state,
                                              const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    __half st[9];
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(normed[(long long)t * 10240 + d]);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// On-device greedy accept: d_vec[k] drafted ids, e_vec[m=k+1] verify argmaxes.
// a = longest prefix with d_i == e_i (0..k). out[0..k]: emit[i] = d_i for i < a;
// emit[a] = e_a (the bonus); rest = -1. acc2 = {slot, a} (device commit key —
// the commit kernels read it; the host's single readback is `out` only).
extern "C" __global__ void xq_accept(const int* __restrict__ d_vec,
                                     const int* __restrict__ e_vec,
                                     int* __restrict__ out,
                                     int* __restrict__ acc2, int k, int slot) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a < k && d_vec[a] == e_vec[a]) a++;
    for (int i = 0; i <= k; i++) {
        if (i < a) out[i] = d_vec[i];
        else if (i == a) out[i] = e_vec[a];
        else out[i] = -1;
    }
    acc2[0] = slot;
    acc2[1] = a;
    // S-A3-f-d Item 1b FIX-3: the host's single readback is `out` and mtp_round
    // reads `a` from out[k+2] — without these stores it reads STALE memory while
    // the commit kernels use acc2's true a, so accepted rows were double-consumed
    // (state ran ahead of the token/position bookkeeping -> long-horizon drift).
    out[k + 1] = -1;
    out[k + 2] = a;
}

// 1-element i32 device copy with dst offset (compute-stream D2D; AGENTS 2.3 —
// never a sync dtod). dst[doff] = src[0].
// S-A3-p: draft-chain pass setup from the per-round meta {b, p, slot, 0}: row 0 of the
// staging buffers = what head_forward's host uploads wrote (pos p+i, slot row), and the
// bonus token for pass 0 (later passes' toks[0] is the previous argmax, copied on device).
extern "C" __global__ void xq_draft_setup(int* __restrict__ toks, int* __restrict__ pos,
                                          int* __restrict__ slots, const int* __restrict__ meta, int i) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    const int p = meta[1] + i;
    pos[0] = p;
    slots[0] = meta[2];
    slots[1] = p;
    if (i == 0) toks[0] = meta[0];
}

// ---- REPRIME (w2, SURPASS WP22 rung 1 + graphed re-prime): the head re-prime at a FIXED row
// count R from device state only (graph-capturable, no host round trip). Target: baseline sm_121
// (plain CUDA: integer indexing, copies and the SAME inline K norm/rope + KV put as
// xq_attn_kv_prefill — no family/arch-specific instruction; bitwise equality with the eager
// kernels needs the same TU, target and flags, so these live next to them).
//
// Stage: a = acc2[1] (xq_accept's device commit key, clamped to [0, R]); p, slot, b from the
// round's draft_meta {b, p, slot, 0}. Row r < a = the eager re-prime's row r exactly: tap row r
// (taps_keep + soff), token emit[r] (= accepted draft d_r), position p+1+r. Rows r >= a are
// "masked": their inputs DUPLICATE row a-1 (a = 0: tap row 0, token b at position p), so every
// row-local kernel sees finite, in-range inputs (embedding index, cos_tab row), and their
// persistent writes are either suppressed (key rows, KV rows: meta[2] = a) or an identical-bits
// refresh of block(p+a)'s pooled key (see qsa_keys_append_dense_masked). meta = {slot, p+1, a, 0}.
extern "C" __global__ void xq_reprime_stage(float* __restrict__ resid, int* __restrict__ toks,
                                            int* __restrict__ pos, int* __restrict__ meta,
                                            const float* __restrict__ taps, const int* __restrict__ dmeta,
                                            const int* __restrict__ acc2, const int* __restrict__ emit,
                                            int R, long long he, long long soff) {
    XQ_PDL_ENTRY();
    int a = acc2[1];
    a = a < 0 ? 0 : (a > R ? R : a);
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)R * he) {
        const int r = (int)(i / he);
        const int src = r < a ? r : (a > 0 ? a - 1 : 0);
        resid[i] = taps[soff + (long long)src * he + (i - (long long)r * he)];
    }
    if (i < R) {
        const int r = (int)i;
        const int q = r < a ? r : a - 1;          // masked rows: row a-1 (a = 0: -1 = b at p)
        toks[r] = q >= 0 ? emit[q] : dmeta[0];
        pos[r] = dmeta[1] + 1 + q;
    }
    if (i == 0) {
        meta[0] = dmeta[2];
        meta[1] = dmeta[1] + 1;
        meta[2] = a;
        meta[3] = 0;
    }
}

// REPRIME: qsa_key_write_b (gpu_batch.cu) with the row count read on device — rows b >= meta[2]
// write nothing. Pure bit copy (bf16 as u16), so the written words equal qsa_key_write_b's.
extern "C" __global__ void xq_qsa_key_write_dn(unsigned short* __restrict__ keys,
                                               const unsigned short* __restrict__ qk,
                                               const int* __restrict__ pos, const int* __restrict__ slot_ids,
                                               int stride, int pitch, int k_off, int hd, int B,
                                               const int* __restrict__ meta) {
    XQ_PDL_ENTRY();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * hd) return;
    const int b = (int)(idx / hd), d = (int)(idx % hd);
    if (b >= meta[2]) return;
    // WP02 (p2 merge): qsa_key_write_b's window guard — a row past the slot's plane would land in
    // the next slot (or past the allocation). In-range writes are unchanged.
    const int col = pos[b];
    if (col < 0 || col >= stride) return;
    keys[((long long)slot_ids[b] * stride + col) * hd + d] = qk[(long long)b * pitch + k_off + d];
}

// REPRIME: xq_attn_kv_prefill with {slot, pos0, n} read on device (meta) — rows t < min(C, n)
// only. Loop body = xq_attn_kv_prefill's verbatim (shared inline K norm+rope and xq_kv_put), so
// every cache word it writes equals the eager kernel's for the same row. n is block-uniform, so
// the per-row __syncthreads stay convergent.
extern "C" __global__ void xq_attn_kv_rows_dn(__half* __restrict__ kp, __half* __restrict__ vp,
                                              float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, int nkv_hd,
                                              int rdim_maxpos, const int* __restrict__ meta, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = meta[0];
    const int pos0 = meta[1];
    const int n = min(C, meta[2]);
    const int kvh = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    for (int t = 0; t < n; t++) {
        const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
        xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                       cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
        const long long p = pos0 + t;
        xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
        __syncthreads();
    }
}

extern "C" __global__ void xq_copy1_i32(int* __restrict__ dst, long long doff,
                                        const int* __restrict__ src) {
    XQ_PDL_ENTRY();
    if (blockIdx.x == 0 && threadIdx.x == 0) dst[doff] = src[0];
}

// HOST / RO-5(a): save (dir 0: rows -> buf) / restore (dir 1: buf -> rows) the draft head's
// PERSISTENT rows at one position around a SPECULATIVE draft pass (mtp_round_adaptive): block
// r < nkvrows = the K|V cache row r (kv_row0 + r*kv_stride bytes, rb bytes), block nkvrows =
// the raw indexer-key row, block nkvrows+1 = the pooled-plane block row (0 bytes / null = absent).
// buf = [nkvrows*rb | key_bytes | plane_bytes]. A pure 4-byte-word copy (every size and offset
// is a multiple of 4 — asserted on the host), so a restore returns the exact pre-speculation bits.
// ADDRESSING (review fix): the K|V rows are the lane's slot (the pass's xq_attn_prep/_decode write
// through sc.slots = draft_meta's slot, baked into kv_row0 on the host), but the raw-key row and
// the pooled block are written by qsa_key_write_b / qsa_pool_update_b at slot = slot_ids[0] — the
// DEVICE slot table, which the DDS round stages only in the QSA-live regime (in the dense regime
// it holds whatever the previous launch left, possibly another lane's slot). So those two rows
// are addressed HERE from the same slot_ids[0] the pass reads: key_row0 / plane_row0 = the slot-0
// row of the position, sstrides = key slot stride (bytes, low 32) | plane slot stride (bytes,
// high 32), kp_bytes = key_bytes | plane_bytes << 16, nslots_dir = nslots | dir << 16. slot_ids is
// not rewritten between the save and the restore (only the speculative pass runs in between), so
// both resolve to the same rows. An out-of-range slot (never staged) skips both rows consistently.
extern "C" __global__ void xq_spec_rows(unsigned int* __restrict__ buf, unsigned long long kv_row0,
                                        long long kv_stride, int rb, int nkvrows,
                                        unsigned long long key_row0, unsigned long long plane_row0,
                                        long long sstrides, int kp_bytes,
                                        const int* __restrict__ slot_ids, int nslots_dir) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int key_bytes = kp_bytes & 0xFFFF, plane_bytes = (kp_bytes >> 16) & 0xFFFF;
    const int nslots = nslots_dir & 0xFFFF, dir = (nslots_dir >> 16) & 0xFFFF;
    unsigned long long row;
    long long boff;
    int nb;
    if (r < nkvrows) { row = kv_row0 + (unsigned long long)((long long)r * kv_stride); boff = (long long)r * rb; nb = rb; }
    else {
        const int s = slot_ids[0];
        if (s < 0 || s >= nslots) return;
        const unsigned long long ss = (unsigned long long)sstrides;
        if (r == nkvrows) {
            row = key_row0 ? key_row0 + (unsigned long long)s * (ss & 0xFFFFFFFFull) : 0ull;
            boff = (long long)nkvrows * rb; nb = key_bytes;
        } else {
            row = plane_row0 ? plane_row0 + (unsigned long long)s * (ss >> 32) : 0ull;
            boff = (long long)nkvrows * rb + key_bytes; nb = plane_bytes;
        }
    }
    if (nb <= 0 || row == 0ull) return;
    unsigned int* g = reinterpret_cast<unsigned int*>(row);
    unsigned int* b = buf + boff / 4;
    const int nw = nb >> 2;
    if (dir == 0) { for (int i = threadIdx.x; i < nw; i += blockDim.x) b[i] = g[i]; }
    else          { for (int i = threadIdx.x; i < nw; i += blockDim.x) g[i] = b[i]; }
}

// block f32 copy (shadow-state staging + tap-row restore; compute-stream).
extern "C" __global__ void xq_copy_f32(float* __restrict__ dst, const float* __restrict__ src,
                                       long long n, long long doff, long long soff) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[doff + i] = src[soff + i];
}

// block u16 copy (PLE ring shadow staging + verify per-layer saves).
// S-A3-f-d Item 1b: 5-arg (doff, soff) mirroring xq_copy_f32 — the previous
// 3-arg version SILENTLY IGNORED the offsets its call sites passed (the
// driver drops surplus args), corrupting the verify commit's raw saves.
extern "C" __global__ void xq_copy_u16(unsigned short* __restrict__ dst,
                                       const unsigned short* __restrict__ src,
                                       long long n, long long doff, long long soff) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[doff + i] = src[soff + i];
}

// Pruned draft lm_head (reference patch 0002, EXL3_MTP_HEAD_N): keep the first
// nb_keep 16-column trellis blocks of every k-block. bw = words per (kb, nb)
// block = 16*BITS. One thread per word.
extern "C" __global__ void xq_slice_trellis_u16(unsigned short* __restrict__ dst,
                                                const unsigned short* __restrict__ src,
                                                int nb_full, int nb_keep, long long bw) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_kb = (long long)nb_keep * bw;
    long long kb = i / per_kb;
    long long off = i % per_kb;
    long long nb2 = off / bw, w = off % bw;
    dst[i] = src[kb * (long long)nb_full * bw + nb2 * bw + w];
}

// TP-H #4 (vocab-parallel lm_head): the same column slice as xq_slice_trellis_u16 but starting at
// 16-column block nb_off (rank * nb_keep) instead of 0. The lmh GEMM's columns are bit-independent
// of the tile partition (each 128-column tile is computed alone), so a shard with a 128-multiple
// column offset produces bitwise the replicated head's columns [nb_off*16, (nb_off+nb_keep)*16).
extern "C" __global__ void xq_slice_trellis_u16_off(unsigned short* __restrict__ dst,
                                                    const unsigned short* __restrict__ src,
                                                    int nb_full, int nb_keep, long long bw, int nb_off) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_kb = (long long)nb_keep * bw;
    long long kb = i / per_kb;
    long long off = i % per_kb;
    long long nb2 = off / bw, w = off % bw;
    dst[i] = src[kb * (long long)nb_full * bw + (nb2 + nb_off) * bw + w];
}

// =====================================================================
// S-A3-i: QSA sparse attention for the EXL3 path (see PLAN/S_A3_I_ATTENTION_DECODE.md).
//
// Native QSA sparsity for the xq decode path (f16 projections, f32 KV cache):
// a per-query selection list of token positions (ascending, -1 padded) is
// produced elsewhere; these kernels cover (a) f16->bf16 indexer staging,
// (b) indexer-q norm+rope, (c) radix top-k selection with ASCENDING identity
// emission, (d) K/V cache write + roped-q f32 staging, (e) gathered
// flash-decode split-K online softmax over the selection list, (f) split
// merge + gate. Math transcribed from gpu_batch.cu's bf16 QSA section
// (QsaParams / qsa_topk_b / rmsnorm_rope_b) and the xq_attn_decode family
// (shared xq_k_norm_rope, softmax scale, final gate expression). exl3_bench.cu
// is a separate translation unit: QsaParamsX and the qsa helpers are
// file-local copies (no cross-TU static references), xq_-prefixed.
// =====================================================================

#include <cuda_bf16.h>

// ---- file-local f32 <-> bf16 converters (this TU otherwise has no bf16):
// bit manipulation, round-to-nearest-even; NaN quieted (sign kept), inf and
// denormals pass through the truncation unchanged.
__device__ __forceinline__ __nv_bfloat16 xq_f2b(float v) {
    unsigned u = __float_as_uint(v);
    if ((u << 1) > 0xFF000000u)                  // NaN: exp all ones + mantissa != 0
        u = (u & 0x80000000u) | 0x7FC00000u;     // canonical quiet NaN, sign kept
    else
        u += 0x7FFFu + ((u >> 16) & 1u);         // RNE on the truncated low 16 bits
    __nv_bfloat16_raw r;
    r.x = (unsigned short)(u >> 16);
    return __nv_bfloat16(r);
}
__device__ __forceinline__ float xq_b2f(__nv_bfloat16 v) {
    __nv_bfloat16_raw r(v);
    return __uint_as_float((unsigned)r.x << 16);
}

// ---- 1. f16 -> bf16 elementwise staging copy. out[i] = bf16(f32(in[i]));
// grid-stride, block 256.
extern "C" __global__ void xq_qsa_h2b(__nv_bfloat16* __restrict__ out, const __half* __restrict__ in,
                                      long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long gs = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += gs) out[i] = xq_f2b(xq_h2f(in[i]));
}

// ---- 2. indexer-q fused per-head RMSNorm + RoPE, rmsnorm_rope_b's math on
// the bf16 indexer tensor (IN PLACE). qk rows [B][heads+1][hdx] (heads q rows
// + 1 k row per token); w = q_layernorm [hdx] f32. UNLIKE rmsnorm_rope_b's
// separate cos/sin tables, cossin rows are [B][2*rdim] INTERLEAVED cos|sin
// (same layout the xq_attn_decode family uses). Grid (B*heads), block hdx
// (=128; static smem sized to the indexer head-dim max).
extern "C" __global__ void xq_qsa_qnorm_rope(__nv_bfloat16* __restrict__ qk, const float* __restrict__ w,
                                             const float* __restrict__ cossin, int heads, int hdx,
                                             int rdim, int B, float eps) {
    XQ_PDL_ENTRY();
    (void)B;                                     // grid is B*heads; b derives from blockIdx
    const int b = blockIdx.x / heads;
    const int h = blockIdx.x % heads;
    const int tid = threadIdx.x;
    const long long base = (long long)b * (heads + 1) * hdx + (long long)h * hdx;
    __shared__ float s[128];
    float v = (tid < hdx) ? xq_b2f(qk[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    const float inv = rsqrtf(s[0] / (float)hdx + eps);
    if (tid < hdx) qk[base + tid] = xq_f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    const int half = rdim / 2;
    if (tid < half) {
        const long long cb = (long long)b * 2 * rdim + tid;
        const float x1 = xq_b2f(qk[base + tid]);
        const float x2 = xq_b2f(qk[base + tid + half]);
        const float c = cossin[cb], sn = cossin[cb + rdim];
        qk[base + tid] = xq_f2b(x1 * c - x2 * sn);
        qk[base + tid + half] = xq_f2b(x2 * c + x1 * sn);
    }
}

// ---- QsaParams (gpu_batch.cu) mirrored field-for-field as the file-local
// QsaParamsX: the 64-B device struct built once per indexer at load, carrying
// what would not fit the 12-argument launch cap.
struct QsaParamsX {
    const float* cos_t;     // [max_pos][rdim] (duplicated halves, see build_rope_tables)
    const float* sin_t;
    const float* kw;        // k_layernorm weight [hd] (gemma style: scale = 1 + w)
    const float* qw;        // q_layernorm weight [hd] (unused on device; the q norm runs kernel 2)
    float eps;
    int hd;                 // indexer head dim (128; 32 on the tiny model)
    int heads;              // indexer query heads (4)
    int rdim;               // rotary dims (64; 16 on the tiny model)
    int ratio;              // compress ratio (4)
    int topk;               // block budget = indexer_budget / ratio (512)
    int sel_max;            // indexer_budget + ratio - 1 (2051): row pitch of the selection lists
    int pad;
    const void* mr;         // VIS-3: QsaParams.mr (per-slot indexer mrope; read only by gpu_batch.cu)
};

#define XQ_QSA_MAX_VERIFY 16   // mirrors gpu_batch.cu MAX_VERIFY (path table row stride)

// rank -> cache column, qsa_col's rule, file-local copy. path == nullptr =>
// identity: the S-A3-i list is raw token positions, no verify tree.
__device__ __forceinline__ int xq_qsa_col(int rank, int pos_start, const unsigned char* path, int b) {
    if (!path || rank < pos_start) return rank;
    return pos_start + (int)path[b * XQ_QSA_MAX_VERIFY + (rank - pos_start)];
}

// Block-wide exclusive scan of two int flags (1024 threads = 32 warps). Returns
// the exclusive prefix of each and the block totals. qsa_exscan2 verbatim.
__device__ __forceinline__ void xq_qsa_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc,
                                               int* wa, int* wc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ia = a, ic = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int xa = __shfl_up_sync(0xffffffffu, ia, off), xc = __shfl_up_sync(0xffffffffu, ic, off);
        if (lane >= off) { ia += xa; ic += xc; }
    }
    if (lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    int ba = 0, bc = 0, sa = 0, sc = 0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; w++) { if (w < warp) { ba += wa[w]; bc += wc[w]; } sa += wa[w]; sc += wc[w]; }
    pa = ba + ia - a; pc = bc + ic - c; ta = sa; tc = sc;
    __syncthreads();
}

// ---- 3. top-k block select + ASCENDING selection list. One 1024-thread block
// per row b; pc = pos[b]+1, nb = pc/ratio blocks, k = min(topk, nb). Radix
// select (4 passes, MSB first, score bits) — qsa_topk_b's logic verbatim:
// score DESCENDING, ties -> LOWEST block index, so the kept set is a pure
// function of the scores. Emission is SIMPLER than qsa_topk_b: an ascending
// sweep over j emits each selected block's ratio token columns with the
// IDENTITY mapping (no path/cps re-ranking), so the list is plain ascending
// token positions; the tail ranks [nb*ratio, pc) ride after. select_all != 0:
// every complete block is selected. sel rows [B][sel_max] are PRE-FILLED with
// -1 by the caller — only emitted entries are written here. pos_sel[b] =
// out-1 (the splitk "pos" of the selection, as in qsa_topk_b).
extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    // Score row read IN PLACE from global (qsa_topk_b's pattern): staging it in
    // __shared__ float sc[8192] claimed "nb <= nblk_stride <= 8192" — false:
    // nblk_stride = max_pos/ratio = 65536, so any pc > 32768 overflowed shared.
    // The row is L2-resident across the radix passes + emission sweep.
    const float* sc = scores + (long long)b * nblk_stride;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    // S-A3-i review fix: full selection only when it fits the list (a >sel_max
    // probe would overflow the sel row); the Rust side passes 0 unconditionally.
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K) {
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int j = tid; j < nb; j += blockDim.x) {
                const unsigned key = __float_as_uint(sc[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0, bin = 0, rem = kp;
                for (int i = 255; i >= 0; i--) {
                    const int c = hist[i];
                    if (cum + c >= kp) { bin = i; rem = kp - cum; break; }
                    cum += c;
                }
                s_bin = bin; s_kp = rem;
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            __syncthreads();
        }
        T = prefix;
    }
    // ---- ascending emission sweep: selected j (in block order) emit their
    // ratio columns at the running selection rank — ascending by construction.
    int run_sel = 0, run_eq = 0;
    int* srow = sel + (long long)b * sel_max;
    for (int base = 0; base < nb; base += blockDim.x) {
        const int j = base + tid;
        const bool valid = j < nb;
        const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
        const bool gt = valid && key > T;
        const bool eq = valid && key == T;
        int p_sel0, p_eq, t_sel, t_eq;
        // first scan: equal-rank (needed to decide selection), second: selection rank
        xq_qsa_exscan2(eq ? 1 : 0, 0, p_eq, p_sel0, t_eq, t_sel, wa, wc);
        const bool selected = select_all ? valid : (gt || (eq && (run_eq + p_eq) < kp));
        int p_s, p_d, t_s, t_d;
        xq_qsa_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) {
            const int r = (run_sel + p_s) * ratio;
            for (int i = 0; i < ratio; i++) srow[r + i] = xq_qsa_col(j * ratio + i, 0, nullptr, b);
        }
        run_sel += t_s; run_eq += t_eq;
    }
    // ---- tail: the pc % ratio most recent tokens (identity, ascending)
    const int nsel = run_sel * ratio;
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += blockDim.x) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3b. WP14 stage 1 (PLAN/SURPASS_PLAN_2026-09-26.md): xq_qsa_topk_asc with the same output
// bits and cheaper steps. The kept set is a pure function of the scores (descending, ties -> lowest
// block index) and the list is emitted ascending, so ANY exact implementation writes the same sel
// row and pos_sel (GB10_EXL3_TOPK_XCHECK=1 diffs them against 3.):
//  (a) histogram increments are warp-aggregated: the MSB passes pile nearly every block into one
//      or two bins (all scores share sign/exponent), i.e. up to 1024-way shared-atomic conflicts.
//      A warp whose predicated lanes share one bin adds popc(ballot) once; otherwise
//      __match_any_sync groups equal bins and each group's leader adds its popc. Same counts.
//  (b) the bucket search is warp-parallel: lane L owns bins 255-8L..248-8L (descending), a shfl
//      scan gives each lane the count above its bins, the one lane holding the crossing walks its
//      8 bins — the serial tid-0 walk's bin and remainder (integer arithmetic, exact).
//  (c) the emission sweep does ONE ballot/popc scan per 1024-block chunk instead of two block
//      exscans: equal-key blocks are selected in ascending j order until kp is used up, so with
//      rem = max(0, kp - run_eq) a block's rank is run_sel + #gt-before + min(#eq-before, rem).
//      Warp totals are double-buffered by chunk parity -> one __syncthreads per chunk.
// Same launch contract as 3. (grid m, block 1024).
extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc2(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                 const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                 int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wgt[2][32], weq[2][32];
    const unsigned FULL = 0xffffffffu;
    const int b = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    const float* sc = scores + (long long)b * nblk_stride;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K && !select_all) {                         // select_all ignores T/kp (as in 3.)
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int base = warp * 32; base < nb; base += blockDim.x) {   // warp-uniform trip count
                const int j = base + lane;
                const bool valid = j < nb;
                const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
                const bool pred = valid && ((key & mask) == prefix);
                const unsigned bin = (key >> shift) & 255u;
                const unsigned pm = __ballot_sync(FULL, pred);
                if (pm == 0u) continue;
                const int src = __ffs(pm) - 1;
                const unsigned b0 = __shfl_sync(FULL, bin, src);
                if (__all_sync(FULL, !pred || bin == b0)) {
                    if (lane == src) atomicAdd(&hist[b0], __popc(pm));
                } else {
                    const unsigned peers = __match_any_sync(FULL, pred ? bin : 0x100u);
                    if (pred && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
                }
            }
            __syncthreads();
            if (warp == 0) {
                int c[8], lsum = 0;
                #pragma unroll
                for (int t = 0; t < 8; t++) { c[t] = hist[255 - 8 * lane - t]; lsum += c[t]; }
                int incl = lsum;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, incl, off);
                    if (lane >= off) incl += x;
                }
                const int excl = incl - lsum;
                const bool here = excl < kp && incl >= kp;
                const unsigned hb = __ballot_sync(FULL, here);
                if (here) {
                    int cum = excl, bin = 0, rem = kp;
                    #pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (cum + c[t] >= kp) { bin = 255 - 8 * lane - t; rem = kp - cum; break; }
                        cum += c[t];
                    }
                    s_bin = bin; s_kp = rem;
                }
                if (hb == 0u && lane == 0) { s_bin = 0; s_kp = kp; }   // the serial walk's defaults
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            // s_bin/s_kp/hist are next written after the next pass's first barrier, which every
            // thread reaches only after these reads — no third barrier needed.
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0, par = 0;
    int* srow = sel + (long long)b * sel_max;
    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < nb; base += blockDim.x, par ^= 1) {
        const int j = base + tid;
        const bool valid = j < nb;
        const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
        const bool gt = select_all ? valid : (valid && key > T);
        const bool eq = !select_all && valid && key == T;
        const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
        if (lane == 0) { wgt[par][warp] = __popc(bg); weq[par][warp] = __popc(be); }
        __syncthreads();
        int g = (lane < nwarps) ? wgt[par][lane] : 0;
        int e = (lane < nwarps) ? weq[par][lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int wg = __shfl_sync(FULL, g, (warp + 31) & 31), we = __shfl_sync(FULL, e, (warp + 31) & 31);
        if (warp == 0) { wg = 0; we = 0; }
        const int pg = wg + __popc(bg & lt), pe = we + __popc(be & lt);
        const int rem = max(0, kp - run_eq);
        if (gt || (eq && pe < rem)) {
            const int r = (run_sel + pg + min(pe, rem)) * ratio;
            for (int i = 0; i < ratio; i++) srow[r + i] = j * ratio + i;
        }
        run_sel += tg + min(te, rem);
        run_eq += te;
    }
    const int nsel = run_sel * ratio;
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += blockDim.x) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3b2. A5-K3 (attn.qsa_select, PLAN/notes_2026-09-27/K3.md): xq_qsa_topk_asc2's output bits with
// a latency-lean schedule. Same launch contract (grid m, block 1024, no dynamic smem, sel rows
// pre-filled with -1 by the caller, only entries [0, pos_sel] + pos_sel written).
// Why asc2 is slow at long context: 4 radix passes + an emission sweep over the row, each element
// paying warp-collective ballots/shuffles/syncs, on ONE SM per row (55-67 us/call at 128K in-round,
// p6d ledger #5a). The kept set is a pure function of the scores under the strict order (key desc,
// block index asc) with key = the score's f32 BITS read as u32 (asc2's order for ANY bits: -0,
// negatives and NaNs rank by their u32 value exactly as there), and the list is emitted ascending —
// so (T, kp) = (the K-th largest key, K - #keys > T) is unique and any exact select writes the same
// entries and pos_sel. The schedule (per row, 16-B loads, per-thread work, no per-element ballots):
//  (0) all blocks kept (select_all, nb <= K) or every key equal (min == max: the K lowest indices)
//      -> the list is the identity prefix; no histogram.
//  (a) min/max of the row: the bits above the highest bit where they differ are common to every
//      key, so the first digit starts right below them.
//  (b) 11-bit digits MSB first; per-thread shared atomics (a quad whose 4 keys share a bin adds 4
//      once); the bucket search is a block scan over the bins in DESCENDING order whose crossing
//      thread returns the serial walk's bin and remainder (integer arithmetic).
//  (c) as soon as the crossing bin holds <= XQ_TK3_CAP keys, ONE more pass compacts every key >= the
//      bin's low edge (= the bin + the < K keys above it) with its block index into shared memory;
//      the remaining digits and the whole emission then run on that list: keys > T and == T are
//      marked in two block bitmaps, equal keys are taken in ASCENDING index until kp is used up
//      (asc2's tie rule), and a block scan over the selected bitmap words emits the blocks in
//      ascending order at their ranks. The list order is not deterministic; nothing downstream of it
//      depends on order (counts and bitmaps only).
//  (d) fallback (a bin never fits — adversarial ties — or K > XQ_TK3_KMAX, or nb > 64 x blockDim):
//      the remaining digits run over the global row and the emission is a two-pass warp-segment
//      scan (rank = #gt-before + min(#eq-before, kp)). Exact, just slower.
#define XQ_TK3_NBIN 2048
#define XQ_TK3_CAP  4096
#define XQ_TK3_KMAX 512
#define XQ_TK3_LIST (XQ_TK3_KMAX + XQ_TK3_CAP)
#define XQ_TK3_U    4

// block-wide inclusive scan (blockDim.x = 32 * nw, nw <= 32); total -> every thread
__device__ __forceinline__ int xq_tk3_bscan(int v, int* wsum, int& total) {
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int incl = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, incl, off); if (lane >= off) incl += x; }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    const int w = lane < nw ? wsum[lane] : 0;
    int wi = w;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, wi, off); if (lane >= off) wi += x; }
    total = __shfl_sync(FULL, wi, 31);
    const int wex = __shfl_sync(FULL, wi - w, warp);
    __syncthreads();                                             // wsum is reused by the next scan
    return incl + wex;
}

// the 4 keys of quad q (blocks 4q..4q+3); lanes past nb read 0 (callers mask by index)
__device__ __forceinline__ uint4 xq_tk3_ld4(const unsigned* __restrict__ sc, int q, int nb, bool vec) {
    const int j = 4 * q;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (vec) {
        if (j < nb) r = __ldg(reinterpret_cast<const uint4*>(sc) + q);
    } else {
        if (j < nb) r.x = __ldg(sc + j);
        if (j + 1 < nb) r.y = __ldg(sc + j + 1);
        if (j + 2 < nb) r.z = __ldg(sc + j + 2);
        if (j + 3 < nb) r.w = __ldg(sc + j + 3);
    }
    return r;
}
__device__ __forceinline__ unsigned xq_tk3_c(const uint4& v, int c) {
    return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}
// the lowest `need` set bits of x (need consumed)
__device__ __forceinline__ unsigned xq_tk3_take(unsigned x, int& need) {
    if (need <= 0) return 0u;
    const int c = __popc(x);
    if (need >= c) { need -= c; return x; }
    unsigned r = 0u;
    for (int k = 0; k < need; k++) { const unsigned lsb = x & (0u - x); r |= lsb; x ^= lsb; }
    need = 0;
    return r;
}

extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc3(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                 const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                 int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned smap[2 * XQ_TK3_NBIN];    // hist | (eq bitmap, sel bitmap)
    __shared__ unsigned lkey[XQ_TK3_LIST];
    __shared__ unsigned short lidx[XQ_TK3_LIST];
    __shared__ int wsum[32], wgt[32], weq[32];
    __shared__ unsigned wlo[32], whi[32];
    __shared__ int s_bin, s_kp, s_cnt, s_n;
    int* hist = reinterpret_cast<int*>(smap);
    const unsigned FULL = 0xffffffffu;
    const int b = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, nt = blockDim.x, nw = nt >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    const float* rowf = scores + (long long)b * nblk_stride;
    const unsigned* __restrict__ sc = reinterpret_cast<const unsigned*>(rowf);
    const bool vec = (reinterpret_cast<unsigned long long>(rowf) & 15ull) == 0ull;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
    int* srow = sel + (long long)b * sel_max;
    int mode = 0, nall = nb, nl = -1;             // 0 identity prefix of nall blocks | 1 list | 2 global
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K && !select_all) {
        // ---- (a) row min / max
        unsigned lo = 0xffffffffu, hi = 0u;
        for (int q0 = tid; 4 * q0 < nb; q0 += XQ_TK3_U * nt) {
            uint4 v[XQ_TK3_U];
            #pragma unroll
            for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, q0 + u * nt, nb, vec);
            #pragma unroll
            for (int u = 0; u < XQ_TK3_U; u++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (4 * (q0 + u * nt) + c < nb) { const unsigned k = xq_tk3_c(v[u], c); lo = min(lo, k); hi = max(hi, k); }
                }
            }
        }
        lo = __reduce_min_sync(FULL, lo); hi = __reduce_max_sync(FULL, hi);
        if (lane == 0) { wlo[warp] = lo; whi[warp] = hi; }
        __syncthreads();
        lo = __reduce_min_sync(FULL, lane < nw ? wlo[lane] : 0xffffffffu);
        hi = __reduce_max_sync(FULL, lane < nw ? whi[lane] : 0u);
        kp = K;
        if (lo == hi) {
            T = hi; nall = K;                        // every key equal: the K lowest indices, ascending
        } else {
            const bool can_list = K <= XQ_TK3_KMAX && nb <= 64 * nt;
            int rb = 32 - __clz(lo ^ hi);            // low bits that still vary (1..32)
            unsigned M = rb >= 32 ? 0u : (0xffffffffu << rb);
            unsigned P = hi & M;
            while (true) {
                const int db = rb < 11 ? rb : 11, shift = rb - db, nbin = 1 << db;
                const unsigned dm = (unsigned)(nbin - 1);
                for (int i = tid; i < nbin; i += nt) hist[i] = 0;
                __syncthreads();                     // every thread is past its last s_* read
                if (tid == 0) { s_bin = 0; s_kp = kp; s_cnt = 0; }   // the serial walk's defaults
                if (nl < 0) {
                    for (int q0 = tid; 4 * q0 < nb; q0 += XQ_TK3_U * nt) {
                        uint4 v[XQ_TK3_U];
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, q0 + u * nt, nb, vec);
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) {
                            const int j0 = 4 * (q0 + u * nt);
                            unsigned bn[4]; bool pr[4];
                            #pragma unroll
                            for (int c = 0; c < 4; c++) {
                                const unsigned k = xq_tk3_c(v[u], c);
                                pr[c] = j0 + c < nb && (k & M) == P;
                                bn[c] = (k >> shift) & dm;
                            }
                            if (pr[0] && pr[1] && pr[2] && pr[3] && bn[0] == bn[1] && bn[0] == bn[2] && bn[0] == bn[3]) {
                                atomicAdd(&hist[bn[0]], 4);
                            } else {
                                #pragma unroll
                                for (int c = 0; c < 4; c++) if (pr[c]) atomicAdd(&hist[bn[c]], 1);
                            }
                        }
                    }
                } else {
                    for (int i = tid; i < nl; i += nt) {
                        const unsigned k = lkey[i];
                        if ((k & M) == P) atomicAdd(&hist[(k >> shift) & dm], 1);
                    }
                }
                __syncthreads();
                {   // thread t owns bins nbin-1-2t, nbin-2-2t (descending); nbin <= 2 * blockDim.x
                    const int b0 = nbin - 1 - 2 * tid, b1 = b0 - 1;
                    const int c0 = b0 >= 0 ? hist[b0] : 0, c1 = b1 >= 0 ? hist[b1] : 0;
                    int tot;
                    const int incl = xq_tk3_bscan(c0 + c1, wsum, tot);
                    const int excl = incl - c0 - c1;
                    if (excl < kp && incl >= kp) {
                        if (excl + c0 >= kp) { s_bin = b0; s_kp = kp - excl; s_cnt = c0; }
                        else                 { s_bin = b1; s_kp = kp - excl - c0; s_cnt = c1; }
                    }
                }
                __syncthreads();
                const int bin = s_bin, cnt = s_cnt;
                kp = s_kp;
                P |= (unsigned)bin << shift;
                M |= dm << shift;
                rb = shift;
                if (can_list && nl < 0 && cnt <= XQ_TK3_CAP) {
                    // ---- (c) compact every key >= P (the crossing bin + the K - kp keys above it)
                    __syncthreads();                 // the s_* reads above precede s_n's reset
                    if (tid == 0) s_n = 0;
                    __syncthreads();
                    for (int base = warp * 32; 4 * base < nb; base += XQ_TK3_U * nt) {   // warp-uniform trips
                        uint4 v[XQ_TK3_U];
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, base + lane + u * nt, nb, vec);
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) {
                            const int j0 = 4 * (base + lane + u * nt);
                            int c = 0;
                            #pragma unroll
                            for (int e = 0; e < 4; e++) c += (j0 + e < nb && xq_tk3_c(v[u], e) >= P) ? 1 : 0;
                            if (!__any_sync(FULL, c > 0)) continue;
                            int incl = c;
                            #pragma unroll
                            for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, incl, off); if (lane >= off) incl += x; }
                            int o = 0;
                            if (lane == 31) o = atomicAdd(&s_n, incl);
                            o = __shfl_sync(FULL, o, 31) + incl - c;
                            #pragma unroll
                            for (int e = 0; e < 4; e++) {
                                const unsigned k = xq_tk3_c(v[u], e);
                                if (j0 + e < nb && k >= P && o < XQ_TK3_LIST) { lkey[o] = k; lidx[o] = (unsigned short)(j0 + e); o++; }
                            }
                        }
                    }
                    __syncthreads();
                    nl = min(s_n, XQ_TK3_LIST);
                }
                if (rb == 0) break;
            }
            T = P;
            mode = nl >= 0 ? 1 : 2;
        }
    }
    int nsel;
    if (mode == 0) {
        for (int t = tid; t < nall * ratio; t += nt) srow[t] = t;
        nsel = nall * ratio;
    } else if (mode == 1) {
        // ---- (c) bitmap selection + ascending emission from the list
        unsigned* eqm = smap;
        unsigned* selm = smap + XQ_TK3_NBIN;
        const int W = (nb + 31) >> 5;
        __syncthreads();                             // hist / s_* reads are done
        for (int w = tid; w < W; w += nt) { eqm[w] = 0u; selm[w] = 0u; }
        __syncthreads();
        for (int i = tid; i < nl; i += nt) {
            const unsigned k = lkey[i];
            const int j = lidx[i];
            if (k > T) atomicOr(&selm[j >> 5], 1u << (j & 31));
            else if (k == T) atomicOr(&eqm[j >> 5], 1u << (j & 31));
        }
        __syncthreads();
        const int w0 = 2 * tid, w1 = w0 + 1;
        const unsigned e0 = w0 < W ? eqm[w0] : 0u, e1 = w1 < W ? eqm[w1] : 0u;
        int etot;
        const int ei = xq_tk3_bscan(__popc(e0) + __popc(e1), wsum, etot);
        int need = kp - (ei - __popc(e0) - __popc(e1));          // equal keys still to take before my words
        const unsigned s0 = (w0 < W ? selm[w0] : 0u) | xq_tk3_take(e0, need);
        const unsigned s1 = (w1 < W ? selm[w1] : 0u) | xq_tk3_take(e1, need);
        int stot;
        const int si = xq_tk3_bscan(__popc(s0) + __popc(s1), wsum, stot);
        int r = si - __popc(s0) - __popc(s1);
        for (unsigned x = s0; x; x &= x - 1u) {
            const int j = w0 * 32 + __ffs(x) - 1;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = j * ratio + i;
            r++;
        }
        for (unsigned x = s1; x; x &= x - 1u) {
            const int j = w1 * 32 + __ffs(x) - 1;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = j * ratio + i;
            r++;
        }
        nsel = stot * ratio;
    } else {
        // ---- (d) fallback emission over warp segments (scalar loads; any alignment)
        const int S = (((nb + nw - 1) / nw) + 31) & ~31;
        const int g0s = min(nb, warp * S), g1s = min(nb, g0s + S);
        int cg = 0, ce = 0;
        for (int base = g0s; base < g1s; base += 32) {
            const int j = base + lane;
            const bool valid = j < g1s;
            const unsigned k = valid ? __ldg(sc + j) : 0u;
            cg += __popc(__ballot_sync(FULL, valid && k > T));
            ce += __popc(__ballot_sync(FULL, valid && k == T));
        }
        if (lane == 0) { wgt[warp] = cg; weq[warp] = ce; }
        __syncthreads();
        int g = lane < nw ? wgt[lane] : 0, e = lane < nw ? weq[lane] : 0;
        const int gs = g, es = e;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int rg = __shfl_sync(FULL, g - gs, warp), re = __shfl_sync(FULL, e - es, warp);
        const unsigned lt = (1u << lane) - 1u;
        for (int base = g0s; base < g1s; base += 32) {
            const int j = base + lane;
            const bool valid = j < g1s;
            const unsigned k = valid ? __ldg(sc + j) : 0u;
            const bool gt = valid && k > T, eq = valid && k == T;
            const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
            const int pg = rg + __popc(bg & lt), pe = re + __popc(be & lt);
            if (gt || (eq && pe < kp)) {
                const int r = (pg + min(pe, kp)) * ratio;
                for (int i = 0; i < ratio; i++) srow[r + i] = j * ratio + i;
            }
            rg += __popc(bg); re += __popc(be);
        }
        nsel = (tg + min(te, kp)) * ratio;
    }
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += nt) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3c. WP18 rung A (PLAN/SURPASS_PLAN_2026-09-26.md): bounded PREFILL selection, bitwise.
// The served prefill scored every (row, block) of a chunk into a [c][max_pos/ratio] f32 matrix
// (0.54 GB at a 262K window) with gpu_batch.cu's qsa_score_prefill_b (one block per thread, 8
// rows x 4 heads per thread, q float4 re-read from smem for every 4 FFMAs: ~13% of FMA peak), then
// ran xq_qsa_topk_asc over each sparse row. WP18 streams the blocks in tiles of WP18_T and never
// holds more than a [rows][WP18_T] slab:
//   xq_wp18_score  — the SAME per-(row, head, block) arithmetic (a),
//   xq_wp18_merge  — per row: carried top-K list  U  the tile  ->  new top-K (b); on the row's
//                    last tile it emits the ascending selection list + tail + pos_sel (c).
// (a) Scores. qsa_score_prefill_b's PTX, per accumulator and 4-dim group d (ascending):
//       t = k[d+1]*q[d+1]; t = fma(k[d], q[d], t); t = fma(k[d+2], q[d+2], t);
//       t = fma(k[d+3], q[d+3], t); acc = acc + t                       (mul/fma.rn/add, acc from +0)
//     then s = (((0 + max(a0,0)) + max(a1,0)) + max(a2,0)) + max(a3,0); s = s / sqrt(128); -0 -> +0.
//     Here every one of those steps is an explicit round-to-nearest intrinsic (nothing left for the
//     compiler or the driver JIT to contract or reorder), on the same bf16 -> f32 exact inputs, so
//     each score is bit-identical; the register tiling only changes WHICH thread owns an
//     accumulator, never its sequence.
// (b) Selection. qsa_topk_asc keeps the top K = topk blocks of a row under the strict total order
//     (key desc, block index asc) — key = the score's f32 bits (all scores are >= +0). For such an
//     order top-K(A u B) = top-K(top-K(A) u B), so folding tiles in ascending block order with a
//     carried list is exact. And once the list holds K entries with minimum key thr, a tile block
//     with key <= thr is below all K of them (equal key -> the carried block has the lower index),
//     so only keys > thr can enter — the filter is exact too. The merge is asc2's radix select
//     (warp-aggregated histogram, warp-parallel bucket search) and its one-scan ascending emission
//     over the virtual array [carried list (ascending index) | filtered tile (ascending index)].
// (c) Emission: carried list ascending -> each block's `ratio` token columns, then the tail
//     tokens [nb*ratio, pos], pos_sel = count-1 — xq_qsa_topk_asc's exact writes (entries past
//     pos_sel are never touched, as before).
// Shape contract (host-gated): indexer hd 128, 4 heads, topk <= WP18_KMAX <= WP18_T.
#define WP18_T     8192        // blocks per tile = slab row pitch (floats)
#define WP18_TM    32          // query rows per scorer CTA
#define WP18_TN    64          // blocks per scorer CTA
#define WP18_DK    32          // head dims per smem stage
#define WP18_HD    128
#define WP18_NH    4
#define WP18_KMAX  512         // carried-list capacity (topk = budget / ratio = 512 served)
#define WP18_MT    512         // merge threads per row

// 8 bf16 (one 16-B load, little-endian pairs) -> 8 f32: exact (b2f is the same bit shift).
__device__ __forceinline__ void wp18_bf8(float4& lo, float4& hi, const uint4 v) {
    lo.x = __uint_as_float(v.x << 16); lo.y = __uint_as_float(v.x & 0xFFFF0000u);
    lo.z = __uint_as_float(v.y << 16); lo.w = __uint_as_float(v.y & 0xFFFF0000u);
    hi.x = __uint_as_float(v.z << 16); hi.y = __uint_as_float(v.z & 0xFFFF0000u);
    hi.z = __uint_as_float(v.w << 16); hi.w = __uint_as_float(v.w & 0xFFFF0000u);
}
// One 4-dim group of one accumulator: qsa_score_prefill_b's contracted expression, pinned.
__device__ __forceinline__ float wp18_step(float acc, const float4 k, const float4 q) {
    float t = __fmul_rn(k.y, q.y);
    t = __fmaf_rn(k.x, q.x, t);
    t = __fmaf_rn(k.z, q.z, t);
    t = __fmaf_rn(k.w, q.w, t);
    return __fadd_rn(acc, t);
}

// Scorer. Grid (ceil(tile blocks / WP18_TN), ceil(nrows / WP18_TM)), block 256, static smem 48 KB
// (2 stages x (q 32 rows x 4 heads x 32 dims + k 64 blocks x 32 dims) f32). Thread (ty, tx) of a
// 16 x 16 grid owns rows ty, ty+16 x blocks tx+16j (j < 4) x all 4 heads = 32 accumulators. The
// 128 head dims stream in 4 stages of 32 through a register prefetch + double-buffered smem (one
// barrier per stage); bf16 -> f32 once at staging. float4 columns are XOR-swizzled (q by row&1,
// k by block&7) so the per-d-group LDS.128 reads are conflict-free (1 / 2 wavefronts).
// slab[r][jj] = score(row r, block kt0 + jj) for block < nb_r = (pos_first + r + 1) / ratio;
// other slab entries are left untouched. q: row r at q + r * q_pitch (bf16, 4 heads x 128 first).
extern "C" __global__ void __launch_bounds__(256, 2)
xq_wp18_score(float* __restrict__ slab, const __nv_bfloat16* __restrict__ q,
              const __nv_bfloat16* __restrict__ plane, const struct QsaParamsX* __restrict__ p,
              int pos_first, int nrows, int kt0, int q_pitch) {
    XQ_PDL_ENTRY();
    constexpr int QROW = WP18_NH * WP18_DK;                // floats per q row per stage (128)
    __shared__ __align__(16) float sq[2][WP18_TM * QROW];  // 2 x 16 KB
    __shared__ __align__(16) float sk[2][WP18_TN * WP18_DK]; // 2 x 8 KB
    const int ratio = p->ratio;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.y * WP18_TM;
    const int jj0 = blockIdx.x * WP18_TN;
    const int blk0 = kt0 + jj0;
    const int rlast = min(row0 + WP18_TM, nrows) - 1;
    const int nb_last = (pos_first + rlast + 1) / ratio;   // the tile's widest row
    if (rlast < row0 || blk0 >= nb_last) return;           // CTA-uniform: nothing visible
    // staging assignment: q chunks c = tid, tid+256 (row c>>4, head (c>>2)&3, 8-dim part c&3);
    // k chunk tid (block tid>>2, part tid&3).
    // (chunk tid+256 is row +16 of chunk tid: same head/part/parity -> same column, +16 rows.)
    const int qr = tid >> 4, qh = (tid >> 2) & 3, qpart = tid & 3;
    const bool qok0 = row0 + qr < nrows, qok1 = row0 + qr + 16 < nrows;
    const __nv_bfloat16* qsrc = q + (long long)(row0 + qr) * q_pitch + qh * WP18_HD + qpart * 8;
    const long long qstep = 16LL * q_pitch;
    const int qdst = qr * QROW + (((qh * 8 + qpart * 2) ^ (qr & 1)) << 2);   // swizzled float4 pair (lo; hi = ^4)
    const int kb = tid >> 2, kpart = tid & 3;
    const bool kok = blk0 + kb < nb_last;
    const __nv_bfloat16* ksrc = plane + (long long)(blk0 + kb) * WP18_HD + kpart * 8;
    const int kdst = kb * WP18_DK + (((kpart * 2) ^ (kb & 7)) << 2);          // hi = ^4
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    uint4 pq0, pq1, pk;
    // stage 0 -> buffer 0
    pq0 = qok0 ? *reinterpret_cast<const uint4*>(qsrc) : z4;
    pq1 = qok1 ? *reinterpret_cast<const uint4*>(qsrc + qstep) : z4;
    pk = kok ? *reinterpret_cast<const uint4*>(ksrc) : z4;
    {
        float4 lo, hi;
        wp18_bf8(lo, hi, pq0);
        *reinterpret_cast<float4*>(&sq[0][qdst]) = lo;
        *reinterpret_cast<float4*>(&sq[0][qdst ^ 4]) = hi;
        wp18_bf8(lo, hi, pq1);
        *reinterpret_cast<float4*>(&sq[0][qdst + 16 * QROW]) = lo;
        *reinterpret_cast<float4*>(&sq[0][(qdst ^ 4) + 16 * QROW]) = hi;
        wp18_bf8(lo, hi, pk);
        *reinterpret_cast<float4*>(&sk[0][kdst]) = lo;
        *reinterpret_cast<float4*>(&sk[0][kdst ^ 4]) = hi;
    }
    __syncthreads();
    float acc[2][4][WP18_NH];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < WP18_NH; h++) acc[i][j][h] = 0.0f;
    const int swq = ty & 1;                                 // (ty + 16) & 1 == ty & 1
    const int swk = tx & 7;                                 // (tx + 16j) & 7 == tx & 7
    constexpr int NST = WP18_HD / WP18_DK;
    #pragma unroll 1
    for (int s = 0; s < NST; s++) {
        const int buf = s & 1;
        if (s + 1 < NST) {                                  // prefetch stage s+1 into registers
            const int off = (s + 1) * WP18_DK;
            pq0 = qok0 ? *reinterpret_cast<const uint4*>(qsrc + off) : z4;
            pq1 = qok1 ? *reinterpret_cast<const uint4*>(qsrc + qstep + off) : z4;
            pk = kok ? *reinterpret_cast<const uint4*>(ksrc + off) : z4;
        }
        const float* sqb = sq[buf];
        const float* skb = sk[buf];
        #pragma unroll 2                                    // full unroll: 128 regs + spills; 2: 119, 0 stack
        for (int g = 0; g < WP18_DK / 4; g++) {             // 4-dim groups, ascending d
            float4 kv[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                kv[j] = *reinterpret_cast<const float4*>(&skb[(tx + 16 * j) * WP18_DK + ((g ^ swk) << 2)]);
            #pragma unroll
            for (int h = 0; h < WP18_NH; h++) {
                const int c4 = (h * 8 + g) ^ swq;
                const float4 q0 = *reinterpret_cast<const float4*>(&sqb[ty * QROW + (c4 << 2)]);
                const float4 q1 = *reinterpret_cast<const float4*>(&sqb[(ty + 16) * QROW + (c4 << 2)]);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[0][j][h] = wp18_step(acc[0][j][h], kv[j], q0);
                    acc[1][j][h] = wp18_step(acc[1][j][h], kv[j], q1);
                }
            }
        }
        if (s + 1 < NST) {                                  // the other buffer: last read at s-1
            float* sqn = sq[buf ^ 1];
            float* skn = sk[buf ^ 1];
            float4 lo, hi;
            wp18_bf8(lo, hi, pq0);
            *reinterpret_cast<float4*>(&sqn[qdst]) = lo;
            *reinterpret_cast<float4*>(&sqn[qdst ^ 4]) = hi;
            wp18_bf8(lo, hi, pq1);
            *reinterpret_cast<float4*>(&sqn[qdst + 16 * QROW]) = lo;
            *reinterpret_cast<float4*>(&sqn[(qdst ^ 4) + 16 * QROW]) = hi;
            wp18_bf8(lo, hi, pk);
            *reinterpret_cast<float4*>(&skn[kdst]) = lo;
            *reinterpret_cast<float4*>(&skn[kdst ^ 4]) = hi;
        }
        __syncthreads();
    }
    const float rs = __fsqrt_rn((float)WP18_HD);
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = row0 + ty + 16 * i;
        if (r >= nrows) continue;
        const int nb = (pos_first + r + 1) / ratio;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int jj = jj0 + tx + 16 * j;
            if (kt0 + jj >= nb) continue;
            float sc = __fadd_rn(0.0f, fmaxf(acc[i][j][0], 0.0f));
            #pragma unroll
            for (int h = 1; h < WP18_NH; h++) sc = __fadd_rn(sc, fmaxf(acc[i][j][h], 0.0f));
            sc = __fdiv_rn(sc, rs);
            if (sc == 0.0f) sc = 0.0f;                      // never -0.0 (the select keys on the bits)
            slab[(long long)r * WP18_T + jj] = sc;
        }
    }
}

// U = [carried list (m entries, ascending block index) | tile (nk entries, filtered)]: element u.
__device__ __forceinline__ void wp18_elem(int u, int m, int nk, bool filt, unsigned thr,
                                          const unsigned* sCk, const float* sS, unsigned& key, bool& valid) {
    if (u < m) { key = sCk[u]; valid = true; return; }
    const int jj = u - m;
    valid = jj < nk;
    key = valid ? __float_as_uint(sS[jj]) : 0u;
    if (filt) valid = valid && key > thr;
}

// Merge. Grid (rows of the group), block WP18_MT, static smem ~37 KB. Row r = pos_first + r;
// tile kt covers blocks [kt*T, kt*T + T). cand_k/cand_i: [rows][topk] carried list (keys = score
// bits, ascending block index), valid for min(topk, kt*T) entries when kt > 0. Rows whose blocks
// end before this tile return (kt == 0 with nb == 0: tail-only list). On the row's LAST tile the
// list is emitted into sel/pos_sel (row pitch sel_max) exactly as xq_qsa_topk_asc writes it.
extern "C" __global__ void __launch_bounds__(WP18_MT)
xq_wp18_merge(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ slab,
              unsigned* __restrict__ cand_k, int* __restrict__ cand_i,
              const struct QsaParamsX* __restrict__ p, int pos_first, int kt) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float sS[WP18_T];
    __shared__ unsigned sCk[WP18_KMAX];
    __shared__ int sCi[WP18_KMAX];
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wgt[2][32], weq[2][32];
    __shared__ unsigned s_red[32];
    __shared__ int s_cnt[32];
    const unsigned FULL = 0xffffffffu;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pos = pos_first + r;
    const int pc = pos + 1;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    const int kT = kt * WP18_T;
    int* srow = sel + (long long)r * sel_max;
    if (nb <= kT) {
        if (kt == 0) {                                     // nb == 0: the tail-only list
            for (int i = tid; i < ratio; i += blockDim.x) if (i <= pos) srow[i] = i;
            if (tid == 0) pos_sel[r] = tail - 1;
        }
        return;
    }
    const int nk = min(nb - kT, WP18_T);
    const bool last = nb <= kT + WP18_T;
    const int m = min(K, kT);                              // carried entries (K whenever kt > 0)
    unsigned* ckr = cand_k + (long long)r * K;
    int* cir = cand_i + (long long)r * K;
    for (int i = tid; i < m; i += blockDim.x) { sCk[i] = ckr[i]; sCi[i] = cir[i]; }
    const float* srcS = slab + (long long)r * WP18_T;
    for (int i = tid * 4; i < nk; i += blockDim.x * 4)     // i + 4 <= WP18_T: whole float4s
        *reinterpret_cast<float4*>(&sS[i]) = *reinterpret_cast<const float4*>(srcS + i);
    __syncthreads();
    // the carried list's K-th largest key (its minimum) — only keys above it can enter
    const bool filt = m == K;
    unsigned thr = 0u;
    if (filt) {                                            // block-uniform
        unsigned mn = 0xFFFFFFFFu;
        for (int i = tid; i < m; i += blockDim.x) mn = min(mn, sCk[i]);
        mn = __reduce_min_sync(FULL, mn);
        if (lane == 0) s_red[warp] = mn;
        __syncthreads();
        for (int w = 0; w < nwarps; w++) thr = w ? min(thr, s_red[w]) : s_red[0];
    }
    int cl = 0;                                            // tile entries that can enter
    for (int i = tid; i < nk; i += blockDim.x) cl += (!filt || __float_as_uint(sS[i]) > thr) ? 1 : 0;
    cl = __reduce_add_sync(FULL, cl);
    if (lane == 0) s_cnt[warp] = cl;
    __syncthreads();
    int cnt = 0;
    for (int w = 0; w < nwarps; w++) cnt += s_cnt[w];
    const int N = m + cnt;                                 // |U| (valid entries)
    if (filt && cnt == 0 && !last) return;                 // carried list unchanged (block-uniform)
    const int U = m + nk;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (N > K) {                                           // asc2's radix select over U
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int base = warp * 32; base < U; base += blockDim.x) {   // warp-uniform trip count
                const int u = base + lane;
                unsigned key; bool valid;
                wp18_elem(u, m, nk, filt, thr, sCk, sS, key, valid);
                const bool pred = valid && ((key & mask) == prefix);
                const unsigned bin = (key >> shift) & 255u;
                const unsigned pm = __ballot_sync(FULL, pred);
                if (pm == 0u) continue;
                const int src = __ffs(pm) - 1;
                const unsigned b0 = __shfl_sync(FULL, bin, src);
                if (__all_sync(FULL, !pred || bin == b0)) {
                    if (lane == src) atomicAdd(&hist[b0], __popc(pm));
                } else {
                    const unsigned peers = __match_any_sync(FULL, pred ? bin : 0x100u);
                    if (pred && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
                }
            }
            __syncthreads();
            if (warp == 0) {
                int c[8], lsum = 0;
                #pragma unroll
                for (int t = 0; t < 8; t++) { c[t] = hist[255 - 8 * lane - t]; lsum += c[t]; }
                int incl = lsum;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, incl, off);
                    if (lane >= off) incl += x;
                }
                const int excl = incl - lsum;
                const bool here = excl < kp && incl >= kp;
                const unsigned hb = __ballot_sync(FULL, here);
                if (here) {
                    int cum = excl, bin = 0, rem = kp;
                    #pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (cum + c[t] >= kp) { bin = 255 - 8 * lane - t; rem = kp - cum; break; }
                        cum += c[t];
                    }
                    s_bin = bin; s_kp = rem;
                }
                if (hb == 0u && lane == 0) { s_bin = 0; s_kp = kp; }
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
        }
        T = prefix;
    }
    // ascending emission over U (asc2's one-scan form): carried entries first (lower indices)
    int run_sel = 0, run_eq = 0, par = 0;
    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < U; base += blockDim.x, par ^= 1) {
        const int u = base + tid;
        unsigned key; bool valid;
        wp18_elem(u, m, nk, filt, thr, sCk, sS, key, valid);
        const bool gt = valid && key > T;
        const bool eq = valid && key == T;
        const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
        if (lane == 0) { wgt[par][warp] = __popc(bg); weq[par][warp] = __popc(be); }
        __syncthreads();
        int g = (lane < nwarps) ? wgt[par][lane] : 0;
        int e = (lane < nwarps) ? weq[par][lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int wg = __shfl_sync(FULL, g, (warp + 31) & 31), we = __shfl_sync(FULL, e, (warp + 31) & 31);
        if (warp == 0) { wg = 0; we = 0; }
        const int pg = wg + __popc(bg & lt), pe = we + __popc(be & lt);
        const int rem = max(0, kp - run_eq);
        if (gt || (eq && pe < rem)) {
            const int rk = run_sel + pg + min(pe, rem);
            const int idx = (u < m) ? sCi[u] : kT + (u - m);
            if (last) {
                for (int i = 0; i < ratio; i++) srow[rk * ratio + i] = idx * ratio + i;
            } else {
                ckr[rk] = key; cir[rk] = idx;
            }
        }
        run_sel += tg + min(te, rem);
        run_eq += te;
    }
    if (last) {
        const int nsel = run_sel * ratio;
        const int t0 = nb * ratio;
        for (int i = tid; i < ratio; i += blockDim.x) {
            const int t = t0 + i;
            if (t <= pos) srow[nsel + i] = t;
        }
        if (tid == 0) pos_sel[r] = nsel + tail - 1;
    }
}

// WP18 XCHECK (eager, diagnostics only): bitwise diff of a scored slab tile against the old full
// matrix (qsa_score_prefill_b's output, row pitch nblk_stride; slab row r = chunk row t_first + r).
// cnt[0] += differing entries, cnt[1] = min packed (chunk row << 16 | block) of a difference,
// cnt[2] += compared entries. Grid (rows), block 256.
extern "C" __global__ void xq_wp18_cmp(int* __restrict__ cnt, const float* __restrict__ slab,
                                       const float* __restrict__ full, const struct QsaParamsX* __restrict__ p,
                                       int pos_first, int t_first, int kt, int nblk_stride) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int nb = (pos_first + r + 1) / p->ratio;
    const int kT = kt * WP18_T;
    const int nk = min(max(nb - kT, 0), WP18_T);
    const float* a = slab + (long long)r * WP18_T;
    const float* b = full + (long long)(t_first + r) * nblk_stride + kT;
    int nd = 0;
    for (int jj = threadIdx.x; jj < nk; jj += blockDim.x) {
        if (__float_as_uint(a[jj]) != __float_as_uint(b[jj])) {
            nd++;
            atomicMin(&cnt[1], ((t_first + r) << 16) | ((kT + jj) & 0xFFFF));
        }
    }
    if (nd) atomicAdd(&cnt[0], nd);
    if (threadIdx.x == 0 && nk > 0) atomicAdd(&cnt[2], nk);
}

// ---- 4. K/V cache write + roped-q staging for the gathered decode. Grid
// (m*nkv) 1-D: row = blockIdx.x/nkv, kvh = blockIdx.x%nkv; block hd (=256).
// (a) K: the EXISTING shared xq_k_norm_rope (bit-identical cache words vs
// xq_attn_decode / xq_attn_decode_group / xq_attn_kv_prefill) then the cache
// write with xq_attn_decode's exact slotbase/kcbase/vcbase arithmetic; V: raw
// f16 -> f32. (b) q: each q head of the kv group normed+roped EXACTLY as
// xq_attn_decode's q section (v^2 tree, (1+w) scale, partial rotate-half with
// c/s; sequential heads over the shared qbuf), staged as f32 to qstage rows
// [row*nh+h][hd] for the splitk kernel. qnkn = q norm [hd] | k norm [hd] f16.
extern "C" __global__ void xq_attn_prep(float* __restrict__ qstage, float* __restrict__ kcvc,
                                        const __half* __restrict__ qg, const __half* __restrict__ kp,
                                        const __half* __restrict__ vp, const __half* __restrict__ qnkn,
                                        const float* __restrict__ cossin, const int* __restrict__ slotpos,
                                        int nh_packed, int hd_packed, int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int row = blockIdx.x / nkv;
    const int kvh = blockIdx.x % nkv;
    const int gqa = nh / nkv;
    const int tid = threadIdx.x;
    const int slot = slotpos[row * 2 + 0];
    const int p = slotpos[row * 2 + 1];
    const float* c = cossin + (long long)row * 2 * rdim;
    const float* s = c + rdim;
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    // ---- (a) K: shared fn = bit-identical cache words; V: raw f16 -> f32
    xq_k_norm_rope(kbuf, red, kp + (long long)row * nkv * hd + (long long)kvh * hd,
                   kn_w, c, s, tid, hd, rdim, eps);
    __syncthreads();
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)row * nkv * hd + (long long)kvh * hd + tid]));
    __syncthreads();
    // ---- (b) q: rmsnorm (1+w), rope, f32 staging — xq_attn_decode's q order,
    // sequential heads over the shared qbuf (syncthreads as in the original).
    for (int h = kvh * gqa; h < kvh * gqa + gqa; h++) {
        float qv = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)h * hd * 2 + tid]);
        red[tid] = qv * qv; __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
        qbuf[tid] = qv; __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        }
        __syncthreads();
        qstage[((long long)row * nh + h) * hd + tid] = xq_kv_rot(qbuf[tid], kc.fmt, tid & 31);
        __syncthreads();
    }
}

// ---- 5. gathered flash-decode split-K: xq_attn_decode's two-pass softmax
// replaced by an ONLINE (streaming) softmax over the row's selection list —
// rank r -> cache column sel[row*sel_max+r], warp-uniform load; -1 padding
// entries skipped. Grid (m*nkv*nsplits) 1-D: sp = blockIdx.x % nsplits,
// kvh = (blockIdx.x/nsplits) % nkv, row = blockIdx.x/(nsplits*nkv); block
// 256 = 8 warps; warp w owns head kvh*gqa + w and, when w+8 < gqa, ALSO head
// kvh*gqa + w + 8 (sequential, separate register sets; this model: gqa 12,
// warps 8 -> warps 0..3 take two heads). Lane owns V dims [lane*8, lane*8+8)
// (hd = 256 = blockDim.x); logit = dot * rsqrtf(hd) exactly matching
// xq_attn_softmax's `red[0] * scale`; all state f32, __expf only, entries
// processed r ASCENDING. Empty split (begin >= nsel): pm = -INFINITY, pl = 0,
// pacc = 0 — the combine skips those splits. Partials per (row, head, split):
// pm/pl [hg*nsplits + sp], pacc [(hg*nsplits + sp)*hd + d], hg = row*nh + h.
extern "C" __global__ void xq_attn_sel_splitk(float* __restrict__ pm, float* __restrict__ pl,
                                              float* __restrict__ pacc, const float* __restrict__ qstage,
                                              const float* __restrict__ kcvc, const int* __restrict__ sel,
                                              const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                              int nh_packed, long long geom, int split_len) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    for (int hh = 0; hh < 2; hh++) {             // head slot 0: warp; slot 1: warp+8
        const int w = warp + hh * 8;
        if (w >= gqa) break;                     // second slot only when warp+8 < gqa
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int r = begin; r < end; r++) {
            const int t = sel[(long long)row * sel_max + r];   // warp-uniform (-1 padded)
            if (t < 0) continue;
            float krow8[8];
            xq_kv_ld8(krow8, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qrow[lane * 8 + i] * krow8[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            const float s = dot * rsqrtf((float)hd);
            const float m_new = fmaxf(m, s);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(s - m_new);
            l = l * scale_f + wt;
            float vrow8[8];
            xq_kv_ld8(vrow8, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vrow8[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- 5a. S-A3-v: xq_attn_sel_splitk with 128-bit loads (prefill-only twin; decode keeps
// splitk). splitk's scalar lane*8+i loads cost 8 L1 wavefronts EACH (24 per entry per head):
// the prefill gather was LSU-bound (133 ms/call @8.5K), not DRAM-bound — a tiled 4-row walk
// with smem-staged K/V (cross-row reuse) measured 142 ms, and with float4 loads 47 ms, vs
// this kernel's high-occupancy one-row layout at ~28 ms. Same values, same expression order:
// bit-identical partials to splitk (GB10_EXL3_SEL_XCHECK, all 48 calls @8.5K).
extern "C" __global__ void xq_attn_sel_splitk4(float* __restrict__ pm, float* __restrict__ pl,
                                              float* __restrict__ pacc, const float* __restrict__ qstage,
                                              const float* __restrict__ kcvc, const int* __restrict__ sel,
                                              const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                              int nh_packed, long long geom, int split_len) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    for (int hh = 0; hh < 2; hh++) {             // head slot 0: warp; slot 1: warp+8
        const int w = warp + hh * 8;
        if (w >= gqa) break;                     // second slot only when warp+8 < gqa
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float qv[8];
        *reinterpret_cast<float4*>(qv) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int r = begin; r < end; r++) {
            const int t = sel[(long long)row * sel_max + r];   // warp-uniform (-1 padded)
            if (t < 0) continue;
            float kv[8], vv[8];
            xq_kv_ld8(kv, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qv[i] * kv[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            const float s = dot * rsqrtf((float)hd);
            const float m_new = fmaxf(m, s);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(s - m_new);
            l = l * scale_f + wt;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vv[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- 5b. WP14 rungs 1/2 (PLAN/SURPASS_PLAN_2026-09-26.md): head-grouped gathered split-K,
// BIT-IDENTICAL partials to 5./5a. (GB10_EXL3_GATHER_XCHECK=1 diffs them). 5a's CTA is 8 warps
// over gqa=12 heads, so warps 0..3 walk their split TWICE (heads w and w+8), and every warp of
// every walk issues its own dependent sel -> K/V row loads (~65 serial DRAM round trips per walk,
// 12x redundant row requests per CTA). Here:
//  - warp w owns heads w, w+nwarps, ... (HPW per warp; blockDim = 32*ceil(gqa/HPW)): ONE walk;
//  - the split's selection entries are staged once in smem (no sel load on the critical path);
//  - the CTA stages each selected K and V row ONCE into a cp.async ring (XQ_HG_STAGES entries,
//    16-B copies, bypass-L1) and every warp reads its lane's 8 dims from smem, so up to
//    XQ_HG_STAGES-1 rows are in flight while the current one is consumed (one barrier per entry).
// The same grid / splits / split_len as 5a (identical partial grouping), entries ascending, -1
// entries skipped, and per head per entry 5a's arithmetic pinned against contraction to the SASS
// 5a compiles to: dot = FFMA chain from +0 over dims ascending, xor tree 16..1 (FADD), broadcast of
// lane 0, s = FMUL(dot, rsqrtf(hd)) with hd a RUNTIME value (5a: blockDim.x — never constant
// folded), m_new = fmaxf, __expf both ways, l = FFMA(l, scale, wt), acc = FFMA(acc, scale,
// FMUL(wt, v)). K/V values are the same bytes read through the same xq_kv_ld8 decode. hd must be
// 256 (lane owns 8 dims) and a row must fit XQ_HG_ROWB; the host dispatch checks both.
// split_hd = split_len | hd << 16.
#define XQ_HG_STAGES 8           // 7 selected rows in flight per CTA (16 KiB of f32 K+V)
#define XQ_HG_CHUNK 256
#define XQ_HG_ROWB 1024
// Stage entry e of the current chunk into ring slot e % XQ_HG_STAGES (whole CTA, 16 B per copy),
// then commit ONE group (empty for e >= n or t < 0: keeps the wait_group accounting aligned).
__device__ __forceinline__ void xq_hg_issue(char (*ring)[2][XQ_HG_ROWB], const int* s_t, int e, int n,
                                            const XqKv& kc) {
    if (e < n) {
        const int t = s_t[e];
        if (t >= 0) {
            const int st = e % XQ_HG_STAGES;
            const int c16 = (int)(kc.rb >> 4);
            for (int c = threadIdx.x; c < 2 * c16; c += blockDim.x) {
                const int isv = c >= c16;
                const int off = (c - isv * c16) << 4;
                const char* src = (isv ? kc.v : kc.k) + (long long)t * kc.rb + off;
                moe_coop_cp16(&ring[st][isv][off], src, 16);
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int HPW>
__device__ __forceinline__ void xq_attn_sel_hg_body(float* __restrict__ pm, float* __restrict__ pl,
                                                    float* __restrict__ pacc, const float* __restrict__ qstage,
                                                    const float* __restrict__ kcvc, const int* __restrict__ sel,
                                                    const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                                                    int nh_packed, long long geom, int split_hd) {
    __shared__ __align__(16) char ring[XQ_HG_STAGES][2][XQ_HG_ROWB];
    __shared__ int s_t[XQ_HG_CHUNK];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int split_len = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int gqa = nh / nkv;
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    const float rsq = rsqrtf((float)hd);
    float qv[HPW][8], acc[HPW][8], mx[HPW], lx[HPW];
    #pragma unroll
    for (int g = 0; g < HPW; g++) {
        mx[g] = -INFINITY; lx[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[g][i] = 0.0f; qv[g][i] = 0.0f; }
        const int w = warp + g * nwarps;
        if (w < gqa) {
            const float* qrow = qstage + ((long long)row * nh + kvh * gqa + w) * hd;
            *reinterpret_cast<float4*>(qv[g]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
            *reinterpret_cast<float4*>(qv[g] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        }
    }
    for (int cb = begin; cb < end; cb += XQ_HG_CHUNK) {
        const int n = min(XQ_HG_CHUNK, end - cb);
        __syncthreads();                                 // the previous chunk's ring/s_t readers are done
        for (int i = threadIdx.x; i < n; i += blockDim.x) s_t[i] = sel[(long long)row * sel_max + cb + i];
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < XQ_HG_STAGES - 1; e++) xq_hg_issue(ring, s_t, e, n, kc);
        for (int r = 0; r < n; r++) {
            // groups committed: (STAGES-1) + r; entry r is group r -> at most STAGES-2 may pend
            asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_HG_STAGES - 2) : "memory");
            __syncthreads();                             // entry r visible; slot (r-1)%S free
            xq_hg_issue(ring, s_t, r + XQ_HG_STAGES - 1, n, kc);
            const int t = s_t[r];                        // block-uniform
            if (t < 0) continue;
            const int st = r % XQ_HG_STAGES;
            float kv[8], vv[8];
            xq_kv_ld8(kv, ring[st][0], kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, ring[st][1], kc.fmt, hd, lane * 8);
            #pragma unroll
            for (int g = 0; g < HPW; g++) {
                if (warp + g * nwarps >= gqa) continue;  // warp-uniform
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) dot = __fmaf_rn(qv[g][i], kv[i], dot);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
                dot = __shfl_sync(0xffffffffu, dot, 0);
                const float s = __fmul_rn(dot, rsq);
                const float m_new = fmaxf(mx[g], s);
                const float scale_f = __expf(mx[g] - m_new);
                const float wt = __expf(s - m_new);
                lx[g] = __fmaf_rn(lx[g], scale_f, wt);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = __fmaf_rn(acc[g][i], scale_f, __fmul_rn(wt, vv[i]));
                mx[g] = m_new;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // only empty groups can be pending
    #pragma unroll
    for (int g = 0; g < HPW; g++) {
        const int w = warp + g * nwarps;
        if (w >= gqa) continue;
        const long long hg = (long long)row * nh + kvh * gqa + w;
        if (lane == 0) {
            pm[hg * nsplits + sp] = mx[g];
            pl[hg * nsplits + sp] = lx[g];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[g][i];
    }
}
// Rung 1: one head per warp (block 32*gqa = 384 at gqa 12; served at m = 1). Rung 2: three heads
// per warp (block 32*ceil(gqa/3) = 128; served at m >= 2) — one smem K/V read feeds three heads.
extern "C" __global__ void __launch_bounds__(384)
xq_attn_sel_hg1(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    xq_attn_sel_hg_body<1>(pm, pl, pacc, qstage, kcvc, sel, pos_sel, slotpos, nh_packed, geom, split_hd);
}
extern "C" __global__ void __launch_bounds__(128)
xq_attn_sel_hg3(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    xq_attn_sel_hg_body<3>(pm, pl, pacc, qstage, kcvc, sel, pos_sel, slotpos, nh_packed, geom, split_hd);
}

// ---- 5d. W4/QSA (decode wave 4, gap #4): the ROW-SHARED union gather for multi-row verifies.
// BIT-IDENTICAL partials to 5./5a./5b. (GB10_W4QSA_XCHECK=1 diffs them against the old dispatch).
// 5b's grid is (row, kv head, split): at a 32K verify (m = 8) the 8 rows' CTAs for one (kv head,
// split) each walk their own ~65-entry sub-list and each stage every selected K/V row again,
// although the rows sit at consecutive positions and select heavily overlapping blocks. Here:
//  - grid (kv head, split, head group of XQ_QU_HPW = 4 heads), grp fastest so the 3 CTAs sharing
//    one (kv head, split) run side by side (their K/V reads coalesce in L2); block 32*m:
//    warp r = verify ROW r, owning the group's 4 heads of that row (4 register chains);
//  - the CTA builds the UNION of the m rows' split sub-lists (per slot group; rows of different
//    slots never share an entry) in ascending token order with a per-entry row mask — a
//    windowed token bitmap (XQ_QU_W tokens per window) + popc prefix: O(1) per element;
//  - each union entry's K and V rows are staged ONCE (cp.async ring, XQ_QU_RC chunks of XQ_QU_S
//    entries, one barrier per chunk; f32 rows land in a lo/hi dim layout so every LDS.128 is
//    conflict-free; other formats are decoded ONCE per CTA by xq_kv_ld8 — the served reader —
//    into an f32 lo/hi tile) and every row whose mask bit is set consumes it.
// Exactness: row r visits exactly its own sub-list entries [sp*L, min((sp+1)*L, nsel)), in list
// order (the union is ascending and each row's valid list is strictly ascending, so its masked
// entries ARE its list), with the same split/partial grouping as 5b. Per head per entry the
// arithmetic is 5b's pinned sequence: dot = FFMA chain from +0 over the lane's 8 dims ascending,
// the xor tree 16..1 (FADD own + partner), s = FMUL(dot, rsqrtf(hd)) with hd RUNTIME, fmaxf,
// __expf both ways, l = FFMA(l, scale, wt), acc = FFMA(acc, scale, FMUL(wt, v)). The 4 chains of a
// warp share ONE transposed tree: level 16 exchanges halves (lanes < 16 keep chains 0,1, lanes
// >= 16 keep 2,3), level 8 keeps one chain per lane (lane>>3), levels 4,2,1 are the plain
// butterfly — every level adds exactly the (own, partner) pair 5b adds for that chain, so the
// sum is the same value, bit for bit (a butterfly leaves every lane with the same bits: FADD is
// commutative), and lanes 8c..8c+7 hold chain c's dot. The softmax step then runs once per chain
// (one lane group) and scale/wt are broadcast for the accumulate. Any row sub-list that is NOT
// strictly ascending / contains a -1 (never produced by xq_qsa_topk_asc*) switches the CTA to an
// exact fallback: each row alone, its list verbatim (order, duplicates, -1 skips as in 5b).
// Contract (host-checked): hd == 256, gqa % 4 == 0, 1 <= m <= XQ_QU_MAXR, split_len <= XQ_QU_MAXL,
// K/V row bytes <= 1024, dynamic smem = xq_qu_smem_bytes below (mirrored by the host).
#define XQ_QU_HPW 4
#define XQ_QU_S 2                        // union entries per chunk (one barrier per chunk)
#define XQ_QU_RC 4                       // chunks in the ring: RC-1 in flight while one is consumed
#define XQ_QU_NE (XQ_QU_S * XQ_QU_RC)
#define XQ_QU_MAXR 16
#define XQ_QU_MAXL 128
#define XQ_QU_W 8192                     // bitmap window (tokens)
#define XQ_QU_WW (XQ_QU_W / 32)
// smem: ring NE*2*rb | [non-f32] f32 tile S*2*hd*4 | sel m*L | u m*L | msk m*L | bm WW | wp WW
//       | n MAXR | slot MAXR | red 32 (ints)
// Stage chunk c (entries c*S .. c*S+S-1 of the union) into its ring slots, then commit ONE group
// (empty past the union / for -1 entries: keeps the wait_group accounting aligned). f32 rows are
// permuted to the lo/hi layout (16-B chunk k -> float4 slot (k&1)*32 + (k>>1)); others are raw.
__device__ __forceinline__ void xq_qu_issue(char* ring, const int* s_u, const unsigned* s_msk,
                                            const int* s_slot, int c, int N, const char* kcvc,
                                            long long rb, long long mp, int nkv, int kvh, int fmt) {
    const int e0 = c * XQ_QU_S;
    const int c16 = (int)(rb >> 4);
    const int per_e = 2 * c16;
    const int tot = min(XQ_QU_S, N - e0) * per_e;          // <= 0 past the union
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int j = q / per_e;
        const int e = e0 + j;
        const int t = s_u[e];
        if (t < 0) continue;
        const int k = q - j * per_e;
        const int isv = k >= c16;
        const int kk = k - isv * c16;                      // 16-B chunk of the row
        const int slot = s_slot[__ffs(s_msk[e]) - 1];      // every row of an entry shares the slot
        const char* src = kcvc + ((((long long)slot * 2 + isv) * nkv + kvh) * mp + t) * rb + ((long long)kk << 4);
        const int dk = fmt == XQ_KV_F32 ? ((kk & 1) * 32 + (kk >> 1)) : kk;
        moe_coop_cp16(ring + (long long)(e % XQ_QU_NE) * 2 * rb + isv * rb + (dk << 4), src, 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
extern "C" __global__ void __launch_bounds__(XQ_QU_MAXR * 32)
xq_attn_sel_union(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                  const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                  const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                  int nh_packed, long long geom, int split_hd, int m) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) char xq_qu_sm[];
    const unsigned FULL = 0xffffffffu;
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int mpf = (int)(geom >> 32);
    const int L = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int gqa = nh / nkv;
    const int ngrp = gqa / XQ_QU_HPW;
    const int grp = blockIdx.x % ngrp;
    const int sp = (blockIdx.x / ngrp) % nsplits;
    const int kvh = blockIdx.x / (ngrp * nsplits);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nthr = blockDim.x, nwarps = nthr >> 5;
    const int fmt = xq_kv_fmt(mpf);
    const long long mp = (long long)(mpf & 0x0FFFFFFF);
    const long long rb = xq_kv_rowb(fmt, hd);
    const int ml = m * L;
    const bool is_f32 = fmt == XQ_KV_F32;
    const int slot_b = 2 * (int)rb;                                // one ring entry: K row | V row
    const int tile_o = XQ_QU_NE * slot_b;                          // non-f32 only: f32 lo/hi tile
    char* ring = xq_qu_sm;
    int* s_sel = reinterpret_cast<int*>(xq_qu_sm + tile_o + (is_f32 ? 0 : XQ_QU_S * 2 * hd * 4));
    int* s_u = s_sel + ml;
    unsigned* s_msk = reinterpret_cast<unsigned*>(s_u + ml);
    unsigned* s_bm = s_msk + ml;
    int* s_wp = reinterpret_cast<int*>(s_bm + XQ_QU_WW);
    int* s_n = s_wp + XQ_QU_WW;
    int* s_slot = s_n + XQ_QU_MAXR;
    int* s_red = s_slot + XQ_QU_MAXR;
    const int begin = sp * L;

    // ---- the rows' split sub-lists (5b's begin/end), slots, masks cleared
    if (tid < m) {
        const int nsel = pos_sel[tid] + 1;
        s_n[tid] = max(0, min(begin + L, nsel) - begin);
        s_slot[tid] = slotpos[tid * 2 + 0];
    }
    for (int i = tid; i < ml; i += nthr) s_msk[i] = 0u;
    __syncthreads();
    for (int idx = tid; idx < ml; idx += nthr) {
        const int r = idx / L, i = idx - r * L;
        s_sel[idx] = i < s_n[r] ? sel[(long long)r * sel_max + begin + i] : 0x7fffffff;
    }
    __syncthreads();
    int bad = 0;
    for (int idx = tid; idx < ml; idx += nthr) {
        const int r = idx / L, i = idx - r * L;
        if (i < s_n[r]) {
            const int v = s_sel[idx];
            bad |= (v < 0) | (i > 0 && v <= s_sel[idx - 1]);
        }
    }
    bad = __syncthreads_or(bad);

    // ---- union (block-uniform N: every thread derives it from the same smem)
    int N = 0;
    if (bad) {
        // exact fallback: each row alone, its list verbatim
        for (int r = 0; r < m; r++) {
            const int n = s_n[r];
            for (int i = tid; i < n; i += nthr) { s_u[N + i] = s_sel[r * L + i]; s_msk[N + i] = 1u << r; }
            N += n;
        }
    } else {
        for (int g = 0; g < m; g++) {
            const int sg = s_slot[g];
            bool lead = true;
            for (int r = 0; r < g; r++) lead = lead && (s_slot[r] != sg);
            if (!lead) continue;                                   // group already done
            int lo = 0x7fffffff, hi_all = -1;
            for (int r = g; r < m; r++) {
                if (s_slot[r] == sg && s_n[r] > 0) {
                    lo = min(lo, s_sel[r * L]);
                    hi_all = max(hi_all, s_sel[r * L + s_n[r] - 1]);
                }
            }
            while (lo <= hi_all) {                                 // one token window [lo, lo+W)
                const int hi = lo + XQ_QU_W;
                for (int w = tid; w < XQ_QU_WW; w += nthr) s_bm[w] = 0u;
                __syncthreads();
                for (int idx = tid; idx < ml; idx += nthr) {
                    const int r = idx / L, i = idx - r * L;
                    const int v = s_sel[idx];
                    if (i < s_n[r] && s_slot[r] == sg && v >= lo && v < hi)
                        atomicOr(&s_bm[(v - lo) >> 5], 1u << ((v - lo) & 31));
                }
                __syncthreads();
                // exclusive popc prefix over the window's words, offset by N (contiguous per thread)
                const int per = (XQ_QU_WW + nthr - 1) / nthr;
                const int w0 = tid * per;
                int loc = 0;
                for (int k = 0; k < per; k++) if (w0 + k < XQ_QU_WW) loc += __popc(s_bm[w0 + k]);
                int inc = loc;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, inc, off);
                    if (lane >= off) inc += x;
                }
                if (lane == 31) s_red[warp] = inc;
                __syncthreads();
                int wbase = 0, tot = 0;
                for (int w = 0; w < nwarps; w++) { const int x = s_red[w]; if (w < warp) wbase += x; tot += x; }
                int run = N + wbase + inc - loc;
                for (int k = 0; k < per; k++) {
                    if (w0 + k < XQ_QU_WW) { s_wp[w0 + k] = run; run += __popc(s_bm[w0 + k]); }
                }
                __syncthreads();
                for (int idx = tid; idx < ml; idx += nthr) {
                    const int r = idx / L, i = idx - r * L;
                    const int v = s_sel[idx];
                    if (i < s_n[r] && s_slot[r] == sg && v >= lo && v < hi) {
                        const int d = v - lo;
                        const int u = s_wp[d >> 5] + __popc(s_bm[d >> 5] & ((1u << (d & 31)) - 1u));
                        s_u[u] = v;                                // equal writers write equal values
                        atomicOr(&s_msk[u], 1u << r);
                    }
                }
                N += tot;
                // next window: the group's smallest element >= hi (rows are strictly ascending);
                // none when the window reached the group's largest element (the common case)
                int nlo = 0x7fffffff;
                if (hi_all >= hi) for (int r = g; r < m; r++) {
                    if (s_slot[r] != sg) continue;
                    const int n = s_n[r];
                    int pos = 0;
                    #pragma unroll
                    for (int step = 128; step > 0; step >>= 1)
                        if (pos + step <= n && s_sel[r * L + pos + step - 1] < hi) pos += step;
                    if (pos < n) nlo = min(nlo, s_sel[r * L + pos]);
                }
                lo = nlo;
                __syncthreads();                                   // s_bm / s_wp / s_red reuse
            }
        }
    }
    __syncthreads();

    // ---- q and state: warp = row, 4 heads (chains) of this CTA's group
    const int r = warp;
    const int h0 = kvh * gqa + grp * XQ_QU_HPW;
    const float rsq = rsqrtf((float)hd);
    float qv[XQ_QU_HPW][8], acc[XQ_QU_HPW][8];
    #pragma unroll
    for (int c = 0; c < XQ_QU_HPW; c++) {
        const float* qrow = qstage + ((long long)r * nh + h0 + c) * hd;
        *reinterpret_cast<float4*>(qv[c]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv[c] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[c][i] = 0.0f;
    }
    float mx = -INFINITY, lx = 0.0f;                               // chain lane>>3 (lanes 8c..8c+7)
    const unsigned rbit = 1u << r;
    const bool b16 = (lane & 16) != 0, b8 = (lane & 8) != 0;

    const int nchunk = (N + XQ_QU_S - 1) / XQ_QU_S;
    #pragma unroll
    for (int c = 0; c < XQ_QU_RC - 1; c++) xq_qu_issue(ring, s_u, s_msk, s_slot, c, N, (const char*)kcvc, rb, mp, nkv, kvh, fmt);
    for (int c = 0; c < nchunk; c++) {
        // groups committed: (RC-1) + c; chunk c is group c -> at most RC-2 may pend
        asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_QU_RC - 2) : "memory");
        __syncthreads();                                           // chunk c visible; chunk c-1's slots free
        xq_qu_issue(ring, s_u, s_msk, s_slot, c + XQ_QU_RC - 1, N, (const char*)kcvc, rb, mp, nkv, kvh, fmt);
        if (!is_f32) {
            // decode chunk c ONCE per CTA (xq_kv_ld8 = the served reader) into the f32 lo/hi tile;
            // the previous chunk's tile readers finished before the barrier above
            for (int it = tid; it < XQ_QU_S * 64; it += nthr) {
                const int j = it >> 6, isv = (it >> 5) & 1, gl = it & 31;
                const int e = c * XQ_QU_S + j;
                if (e < N && s_u[e] >= 0) {
                    float o[8];
                    xq_kv_ld8(o, xq_qu_sm + (e % XQ_QU_NE) * slot_b + isv * (int)rb, fmt, hd, gl * 8);
                    float4* tt = reinterpret_cast<float4*>(xq_qu_sm + tile_o) + (j * 2 + isv) * 64;
                    tt[gl] = make_float4(o[0], o[1], o[2], o[3]);
                    tt[32 + gl] = make_float4(o[4], o[5], o[6], o[7]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < XQ_QU_S; j++) {
            const int e = c * XQ_QU_S + j;
            if (e >= N) break;                                     // block-uniform
            if (!(s_msk[e] & rbit)) continue;                      // warp-uniform: not this row's entry
            if (s_u[e] < 0) continue;                              // fallback -1 entry (5b skips it)
            const float4* kt = reinterpret_cast<const float4*>(
                xq_qu_sm + (is_f32 ? (e % XQ_QU_NE) * slot_b : tile_o + j * 2 * hd * 4));
            float kv[8];
            {
                const float4 a = kt[lane], b = kt[32 + lane];
                kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w;
                kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
            }
            float d[XQ_QU_HPW];
            #pragma unroll
            for (int ch = 0; ch < XQ_QU_HPW; ch++) {
                d[ch] = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) d[ch] = __fmaf_rn(qv[ch][i], kv[i], d[ch]);
            }
            // transposed xor tree (see the header): lane holds chain lane>>3's dot at the end
            const float k0 = b16 ? d[2] : d[0], k1 = b16 ? d[3] : d[1];
            const float s0 = b16 ? d[0] : d[2], s1 = b16 ? d[1] : d[3];
            const float e0 = __fadd_rn(k0, __shfl_xor_sync(FULL, s0, 16));
            const float e1 = __fadd_rn(k1, __shfl_xor_sync(FULL, s1, 16));
            float x = __fadd_rn(b8 ? e1 : e0, __shfl_xor_sync(FULL, b8 ? e0 : e1, 8));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 4));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 2));
            x = __fadd_rn(x, __shfl_xor_sync(FULL, x, 1));
            const float s = __fmul_rn(x, rsq);
            const float m_new = fmaxf(mx, s);
            const float scale_f = __expf(mx - m_new);
            const float wt = __expf(s - m_new);
            lx = __fmaf_rn(lx, scale_f, wt);
            mx = m_new;
            float vv[8];
            {
                const float4 a = kt[64 + lane], b = kt[96 + lane];
                vv[0] = a.x; vv[1] = a.y; vv[2] = a.z; vv[3] = a.w;
                vv[4] = b.x; vv[5] = b.y; vv[6] = b.z; vv[7] = b.w;
            }
            #pragma unroll
            for (int ch = 0; ch < XQ_QU_HPW; ch++) {
                const float sc = __shfl_sync(FULL, scale_f, ch * 8);
                const float w = __shfl_sync(FULL, wt, ch * 8);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[ch][i] = __fmaf_rn(acc[ch][i], sc, __fmul_rn(w, vv[i]));
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");          // only empty groups can pend
    #pragma unroll
    for (int ch = 0; ch < XQ_QU_HPW; ch++) {
        const long long hg = (long long)r * nh + h0 + ch;
        if (lane == ch * 8) {
            pm[hg * nsplits + sp] = mx;
            pl[hg * nsplits + sp] = lx;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[ch][i];
    }
}
// W4/QSA probe-only (--probe-exl3-binv sparse section): fill positions [p0, p0 + gridDim.x) of
// every (slot, K|V, kv head) plane (blockIdx.y = (slot*2 + kv)*nkv + h) with hashed pseudo-random
// values through the SERVED row writer xq_kv_put (every KV format; blockDim == hd), so a 128K
// synthetic context costs one launch and every position holds distinct content.
extern "C" __global__ void xq_qu_synth_kv(float* __restrict__ kcvc, int mpf, int hd, int nkv, int p0,
                                          unsigned seed) {
    XQ_PDL_ENTRY();
    const int t = p0 + (int)blockIdx.x;
    const int plane = (int)blockIdx.y;
    const int h = plane % nkv, kv = (plane / nkv) & 1, slot = plane / (2 * nkv);
    const XqKv kc = xq_kv_make(kcvc, mpf, hd, nkv, slot, h);
    unsigned x = seed ^ ((unsigned)plane * 0x9E3779B1u) ^ ((unsigned)t * 0x85EBCA6Bu) ^ (threadIdx.x * 0xC2B2AE35u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    const float v = ((float)(x & 0xFFFFFFu) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * (kv == 0 ? 1.5f : 1.0f);
    xq_kv_put((kv ? kc.v : kc.k) + (long long)t * kc.rb, kc.fmt, hd, threadIdx.x, v);
}

// ---- 5c. PQ8 item 3: the PREFILL gather for the 8-bit KV layouts (q8; the probe-only q4x shares its
// layout and reader). BIT-IDENTICAL partials to xq_attn_sel_splitk4 (GB10_PQ8_XCHECK diffs them):
// the same grid (rows*nkv*nsplits: sp = bid % nsplits, kvh = (bid/nsplits) % nkv, row =
// bid/(nsplits*nkv)), split_len and partial layout; entries ascending, -1 entries skipped; per head
// per entry splitk4's arithmetic as its PTX reads: dot = fma.rn chain from +0 over the lane's 8 dims
// ascending, xor tree 16..1 (dot + shfl), lane-0 broadcast, s = dot * rsqrtf(hd) with hd RUNTIME
// (MUFU.RSQ, as splitk4's blockDim.x), m_new = fmaxf(m, s), __expf both ways, l = fma(l, scale,
// wt), acc = fma(acc, scale, wt * v). K/V values: xq_kv_ld8's q8 expression on the same bytes.
// What changes is where the values come from. splitk4 (8 warps over gqa 12 heads, warps 0..3 walk
// their split twice) makes EVERY head-walk re-load its lane's 8 codes + scale and re-run the unpack
// (12x per entry per CTA), behind a per-entry dependent sel -> K/V load chain: PROBES_14 found the
// prefill gather latency/issue-bound (21% of 32K prefill kernel time; an 8-bit cache made it
// slower, fp8 +13.6%). Here:
//  - warp w owns heads w, w+4, w+8 (XQ_PQ8_HPW = 3 x 4 warps: gqa == 12 exactly, host-checked):
//    one walk, three independent head chains per warp (ILP), nothing walked twice;
//  - the split's selection entries are staged once in smem;
//  - the CTA unpacks each selected K and V row ONCE into an f32 smem tile of XQ_PQ8_TE entries,
//    stored lo/hi (dims 8L..8L+3 at float4 [L], 8L+4..8L+7 at float4 [32+L]) so each warp's
//    4 LDS.128 per entry are conflict-free;
//  - the NEXT tile's raw bytes (8 codes + the group's f16 scale per thread-item) are loaded into
//    registers before the current tile's math, so global latency hides behind a tile of compute.
// hd must be 256 (lane owns 8 dims; host-checked). split_hd = split_len | hd << 16.
#define XQ_PQ8_HPW 3
#define XQ_PQ8_NW 4
#define XQ_PQ8_TE 8
#define XQ_PQ8_CH 1024
#define XQ_PQ8_IPT ((XQ_PQ8_TE * 2 * 32) / (XQ_PQ8_NW * 32))   // raw items per thread per tile (4)
// q8 dims d0..d0+7 as raw words: the 8 codes + the 32-group's f16 scale bits (d0 % 8 == 0).
struct XqQ8Raw { uint2 u; unsigned short h; };
__device__ __forceinline__ void xq_q8_raw(XqQ8Raw& r, const char* row, int hd, int d0) {
    r.u = *reinterpret_cast<const uint2*>(row + d0);
    r.h = reinterpret_cast<const unsigned short*>(row + hd)[d0 >> 5];
}
// == xq_kv_ld8's q8 branch on the same bytes: sm = fl(f16(scale) * 2^-7), hs = 0.5 * sm, o = xq_q8v.
__device__ __forceinline__ void xq_q8_deq8(float* o, const XqQ8Raw& r) {
    const float sm = __fmul_rn(__half2float(__ushort_as_half(r.h)), 0.0078125f);
    const float hs = 0.5f * sm;
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = xq_q8v(r.u.x, i, sm, hs);
    #pragma unroll
    for (int i = 0; i < 4; i++) o[4 + i] = xq_q8v(r.u.y, i, sm, hs);
}
extern "C" __global__ void __launch_bounds__(XQ_PQ8_NW * 32, 4)
xq_attn_sel_pq8(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float4 tile[XQ_PQ8_TE][2][64];   // [entry][K|V][lo 0..31 | hi 32..63]
    __shared__ int s_t[XQ_PQ8_CH];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int split_len = split_hd & 0xFFFF;
    const int hd = (int)((unsigned)split_hd >> 16);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gqa = nh / nkv;
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    const float rsq = rsqrtf((float)hd);
    float qv[XQ_PQ8_HPW][8], acc[XQ_PQ8_HPW][8], mx[XQ_PQ8_HPW], lx[XQ_PQ8_HPW];
    #pragma unroll
    for (int g = 0; g < XQ_PQ8_HPW; g++) {
        mx[g] = -INFINITY; lx[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[g][i] = 0.0f; qv[g][i] = 0.0f; }
        const int w = warp + g * XQ_PQ8_NW;
        const float* qrow = qstage + ((long long)row * nh + kvh * gqa + w) * hd;
        *reinterpret_cast<float4*>(qv[g]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv[g] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    // thread item k of a tile: row-slot rs = warp + NW*k -> entry rs >> 1, K|V = rs & 1, dims lane*8..+7
    // (a warp reads one row's 256 contiguous code bytes + its 16 scale bytes per item: coalesced).
    XqQ8Raw raw[XQ_PQ8_IPT];
    for (int cb = begin; cb < end; cb += XQ_PQ8_CH) {
        const int n = min(XQ_PQ8_CH, end - cb);
        __syncthreads();                                   // previous chunk's s_t / tile readers done
        for (int i = tid; i < n; i += XQ_PQ8_NW * 32) s_t[i] = sel[(long long)row * sel_max + cb + i];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < XQ_PQ8_IPT; k++) {             // prefetch tile 0 (warp-uniform branches)
            const int rs = warp + XQ_PQ8_NW * k, e = rs >> 1;
            const int t = e < n ? s_t[e] : -1;
            if (t >= 0) xq_q8_raw(raw[k], ((rs & 1) ? kc.v : kc.k) + (long long)t * kc.rb, hd, lane * 8);
        }
        for (int e0 = 0; e0 < n; e0 += XQ_PQ8_TE) {
            const int ne = min(XQ_PQ8_TE, n - e0);
            __syncthreads();                               // previous tile's readers done
            #pragma unroll
            for (int k = 0; k < XQ_PQ8_IPT; k++) {         // unpack ONCE per CTA into the f32 tile
                const int rs = warp + XQ_PQ8_NW * k, e = rs >> 1;
                if (e < ne && s_t[e0 + e] >= 0) {
                    float o[8];
                    xq_q8_deq8(o, raw[k]);
                    tile[e][rs & 1][lane] = make_float4(o[0], o[1], o[2], o[3]);
                    tile[e][rs & 1][32 + lane] = make_float4(o[4], o[5], o[6], o[7]);
                }
            }
            __syncthreads();
            if (e0 + XQ_PQ8_TE < n) {                      // next tile's bytes in flight during the math
                #pragma unroll
                for (int k = 0; k < XQ_PQ8_IPT; k++) {
                    const int rs = warp + XQ_PQ8_NW * k, e = e0 + XQ_PQ8_TE + (rs >> 1);
                    const int t = e < n ? s_t[e] : -1;
                    if (t >= 0) xq_q8_raw(raw[k], ((rs & 1) ? kc.v : kc.k) + (long long)t * kc.rb, hd, lane * 8);
                }
            }
            for (int e = 0; e < ne; e++) {
                if (s_t[e0 + e] < 0) continue;             // block-uniform (-1 padding)
                float kv[8], vv[8];
                {
                    const float4 a = tile[e][0][lane], b = tile[e][0][32 + lane];
                    kv[0] = a.x; kv[1] = a.y; kv[2] = a.z; kv[3] = a.w;
                    kv[4] = b.x; kv[5] = b.y; kv[6] = b.z; kv[7] = b.w;
                    const float4 c = tile[e][1][lane], d = tile[e][1][32 + lane];
                    vv[0] = c.x; vv[1] = c.y; vv[2] = c.z; vv[3] = c.w;
                    vv[4] = d.x; vv[5] = d.y; vv[6] = d.z; vv[7] = d.w;
                }
                // the three heads' chains stage by stage (independent: ILP 3); each head's own op
                // sequence is splitk4's, in splitk4's order. gqa == XQ_PQ8_NW*XQ_PQ8_HPW (host).
                float dot[XQ_PQ8_HPW];
                #pragma unroll
                for (int g = 0; g < XQ_PQ8_HPW; g++) {
                    dot[g] = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) dot[g] = __fmaf_rn(qv[g][i], kv[i], dot[g]);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int g = 0; g < XQ_PQ8_HPW; g++)
                        dot[g] = __fadd_rn(dot[g], __shfl_xor_sync(0xffffffffu, dot[g], off));
                }
                #pragma unroll
                for (int g = 0; g < XQ_PQ8_HPW; g++) {
                    const float s = __fmul_rn(__shfl_sync(0xffffffffu, dot[g], 0), rsq);
                    const float m_new = fmaxf(mx[g], s);
                    const float scale_f = __expf(mx[g] - m_new);
                    const float wt = __expf(s - m_new);
                    lx[g] = __fmaf_rn(lx[g], scale_f, wt);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) acc[g][i] = __fmaf_rn(acc[g][i], scale_f, __fmul_rn(wt, vv[i]));
                    mx[g] = m_new;
                }
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < XQ_PQ8_HPW; g++) {
        const int w = warp + g * XQ_PQ8_NW;
        const long long hg = (long long)row * nh + kvh * gqa + w;
        if (lane == 0) {
            pm[hg * nsplits + sp] = mx[g];
            pl[hg * nsplits + sp] = lx[g];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[g][i];
    }
}

// ---- A5 D5: dense split-K decode attention for the MTP draft head (was xq_attn_decode:
// one block per QUERY head, each serially streaming its kv head's whole f32 K/V — at 32K
// context 12x redundant reads (GQA 12), 6.6 ms of a 7.65 ms draft pass). One block per
// (row, kv head, split); warps own query heads (w, w+8) and share every K/V row read, so a
// pass reads K/V ONCE. The split range is derived ON DEVICE from the row's position
// (split_len = ceil((p+1)/nsplits)), so a graph captured at any context replays at every
// context. Partials feed xq_attn_sel_combine (non-empty splits, ascending). Draft-only:
// the verify decides every emitted token (acceptance, not losslessness, is the gate).
extern "C" __global__ void xq_attn_dense_splitk4(float* __restrict__ pm, float* __restrict__ pl,
                                                float* __restrict__ pacc, const float* __restrict__ qstage,
                                                const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                                                int nh_packed, long long geom) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;   // keys 0..=p (p's own K/V written by xq_attn_prep)
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int split_len = (npos + nsplits - 1) / nsplits;
    const int begin = sp * split_len;
    const int end = min(begin + split_len, npos);
    for (int hh = 0; hh < 2; hh++) {
        const int w = warp + hh * 8;
        if (w >= gqa) break;
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float qv[8];
        *reinterpret_cast<float4*>(qv) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        const float qs = rsqrtf((float)hd);
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int t = begin; t < end; t++) {
            float kv[8], vv[8];
            xq_kv_ld8(kv, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qv[i] * kv[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            const float sc = dot * qs;
            const float m_new = fmaxf(m, sc);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(sc - m_new);
            l = l * scale_f + wt;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vv[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- 6. split merge + gate — xq_attn_decode's final expression order.
// Grid (rows*nh), block hd. m_g = max over split maxes; l_g = sum of
// pl*exp(pm-m_g) and acc[tid] = sum of pacc*exp(pm-m_g) over NON-EMPTY splits
// (pm > -INFINITY), s ASCENDING; then the per-head gate at qg's +hd offset
// and attn = f16((acc/l_g) * sig(g)) — exactly xq_attn_decode's closing lines.
// WP25: bits 28..31 of `rows` carry the KV format (q8: the split-K partials accumulate in the
// cache's rotated basis; the output rotates back here, once, before the gate — the reference implementation's merge).
extern "C" __global__ void xq_attn_sel_combine(__half* __restrict__ attn, const float* __restrict__ pm,
                                               const float* __restrict__ pl, const float* __restrict__ pacc,
                                               const __half* __restrict__ qg, int nh_packed,
                                               int nsplits, int rows) {
    XQ_PDL_ENTRY();
    const int kvf = xq_kv_fmt(rows);             // grid is rows*nh; b derives from blockIdx
    const int nh = nh_packed & 0xFFFF;
    const int hd = blockDim.x;
    const int b = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int tid = threadIdx.x;
    const long long hg = (long long)b * nh + h;
    float m_g = -INFINITY;
    for (int s = 0; s < nsplits; s++) m_g = fmaxf(m_g, pm[hg * nsplits + s]);
    float l_g = 0.0f;
    for (int s = 0; s < nsplits; s++)
        if (pm[hg * nsplits + s] > -INFINITY) l_g += pl[hg * nsplits + s] * __expf(pm[hg * nsplits + s] - m_g);
    float acc = 0.0f;
    for (int s = 0; s < nsplits; s++) {
        if (pm[hg * nsplits + s] > -INFINITY) {
            const float w = __expf(pm[hg * nsplits + s] - m_g);
            acc += pacc[(hg * nsplits + s) * hd + tid] * w;
        }
    }
    const float g = xq_h2f(qg[(long long)b * nh * hd * 2 + (long long)h * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / l_g, kvf, tid & 31);
    attn[(long long)b * nh * hd + (long long)h * hd + tid] = xq_f2h(o * xq_sig(g));
}

// =====================================================================================
// W4/SMALL (decode wave 4, package SMALL): five latency-bound small-kernel levers. Every kernel
// below is a BITWISE twin of the kernel it replaces: the same per-output operation sequence and
// rounding points; only where an operand lives (smem instead of a serialized global load), which
// CTA computes an output, how many launches carry the work, or how early a load is issued
// changes. Host escapes: GB10_W4S_OFF=1|all or a comma list router,gdn,dattn,ple,draft (the old
// kernels, exactly); eager per-call checks GB10_W4S_XCHECK=1 (+GB10_EXL3_NO_GRAPH=1): new vs
// the old twin, bitwise. Target contract: plain sm_121 — FFMA/FMUL/FADD, MUFU via __expf, warp
// shuffles, shared memory, cp.async.cg (baseline sm_80+ class), bar.sync; no f/a feature.
// =====================================================================================

// ---- (5) ROUTER FOLD, persistent (xq_router_fold_w4). The p4c ledger shows xq_router_fold at
// 22.3 + 0.95*m us per call. After its 5 weight loads, the served SASS issues each (q, row) x
// float4 load right before its 8 FFMAs, reusing ONE register quad, so a thread walks 5*M
// SERIALIZED L1/L2 round trips (the m-slope), and every one of the 128 CTAs re-reads all of x
// (5.2 MB of L2->SM traffic at m=8). Here grid = #SMs (host: 48), 1 CTA/SM x 768 threads: CTA c
// owns experts [c*N/G, (c+1)*N/G) (<= 12), x is staged ONCE per CTA into smem (coalesced 16-B
// vectors, overlapping the weight loads), the GEMV reads x from smem (LDS, ~30 cycles).
// BIT CONTRACT (== xq_router_fold, itself == the S-A3-z pair): per (row, expert) the SAME
// thread-local chain — 40-wide slice (5 float4 groups of 8), ascending k, `acc += w*x` (the same
// source expression -> FFMA(w, x, acc)), the same REMAP lane map (warp = 4 experts x 8 slices),
// the same padded partial layout (el stride 72), the same one-thread ascending s = 0..63 sum from
// 0.0f; the tail (softmax / top-k / renorm / first-seen route) is xq_router_fold's verbatim (its
// per-warp first-count table only takes warps < 8: route threads are tid < M*k <= 128).
#define XQ_W4R_EPC 12
#define XQ_W4R_NT (XQ_W4R_EPC * XQ_RF_KS)         // 768 threads
#define XQ_W4R_RS (XQ_W4R_EPC * XQ_RFD_PS + 1)    // partial row stride (865 == 1 mod 32)
extern "C" __global__ void __launch_bounds__(XQ_W4R_NT, 1)
xq_router_fold_w4(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                  int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                  int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                  unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                  unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    // dynamic smem (host): max(M*K*2 [x], M*RS*4 [partials], tail scratch) bytes
    extern __shared__ __align__(16) float w4r_sm[];
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int e_lo = (int)(((long long)blockIdx.x * N) / (long long)gridDim.x);
    const int e_hi = (int)(((long long)(blockIdx.x + 1) * N) / (long long)gridDim.x);
    const int ne_c = e_hi - e_lo;                                // host: <= XQ_W4R_EPC
    const int el = ((warp >> 3) << 2) | (lane >> 3);             // REMAP: 4 experts x 8 slices per warp
    const int s = ((warp & 7) << 3) | (lane & 7);
    const int n = e_lo + el;
    const bool act = el < ne_c;
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    __half* xs = reinterpret_cast<__half*>(w4r_sm);
    // the weights first (the DRAM long pole), then x -> smem while they are in flight
    float4 wv5[5];
    if (act && slice == 40) {
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
    }
    {
        const int nv = (M * K) >> 3;                             // host: M*K % 8 == 0
        const uint4* src = reinterpret_cast<const uint4*>(x);
        uint4* dst = reinterpret_cast<uint4*>(xs);
        for (int i = tid; i < nv; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    // ---- GEMV: xq_router_fold's per-thread body, verbatim, x read from smem ----
    float acc[XQ_RF_MAXM];
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) acc[r] = 0.0f;
    if (act && slice == 40) {
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(xs + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (act) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(xs + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int kk = (s8 << 3); kk < slice; ++kk) {
            #pragma unroll
            for (int r = 0; r < XQ_RF_MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[kk]) * __half2float(xs[(size_t)r * K + k0 + kk]);
        }
    }
    __syncthreads();                                             // every x read done: part aliases xs
    float* part = w4r_sm;
    #pragma unroll
    for (int r = 0; r < XQ_RF_MAXM; r++) if (r < M) part[r * XQ_W4R_RS + el * XQ_RFD_PS + s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_W4R_EPC) {
        const int r = tid / XQ_W4R_EPC, e = tid % XQ_W4R_EPC;
        if (e < ne_c) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r * XQ_W4R_RS + e * XQ_RFD_PS + q];
            out[(size_t)r * N + e_lo + e] = a;
        }
    }
    // ---- publish + elect the last block (every block's logits visible before its ticket) ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
        if (last) __threadfence();
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    // ---- tail (last block only): xq_router_fold's, verbatim. `part` is dead: route scratch ----
    int* s_first = reinterpret_cast<int*>(part);                 // [ne]  first position of expert e
    int* s_ids = s_first + XQ_RFD_NEMAX;                         // [M*k] picked ids, row-major
    int* s_slot = s_ids + XQ_RF_MAXM * XQ_RTR_MAXK;              // [M*k] slot of a first position
    int* s_wc = s_slot + XQ_RF_MAXM * XQ_RTR_MAXK;               // [8]   per-warp first counts
    for (int e = tid; e < N; e += blockDim.x) s_first[e] = 0x7fffffff;
    if (warp < M) {
        const int r = warp;
        const float* lg = out + (size_t)r * N;
        float v[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) { const int i = lane + 32 * t; v[t] = (i < N) ? __ldcg(lg + i) : 0.0f; }
        float mq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            mq[q] = -INFINITY;
            if (lane + 32 * q < N) mq[q] = fmaxf(mq[q], v[q]);
            if (lane + 32 * q + 256 < N) mq[q] = fmaxf(mq[q], v[q + 8]);
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) mq[q] = fmaxf(mq[q], mq[q + 4]);           // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) mq[q] = fmaxf(mq[q], mq[q + 2]);           // s2 = 64
        float mx = fmaxf(mq[0], mq[1]);                                          // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
        const float gmax = __shfl_sync(0xffffffffu, mx, 0);
        float ev[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) ev[t] = __expf(v[t] - gmax);
        float pq[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            pq[q] = 0.0f;
            if (lane + 32 * q < N) pq[q] += ev[q];
            if (lane + 32 * q + 256 < N) pq[q] += ev[q + 8];
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) pq[q] += pq[q + 4];                          // s2 = 128
        #pragma unroll
        for (int q = 0; q < 2; ++q) pq[q] += pq[q + 2];                          // s2 = 64
        float sm_ = pq[0] + pq[1];                                               // s2 = 32
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sm_ += __shfl_down_sync(0xffffffffu, sm_, o);
        const float denom = __shfl_sync(0xffffffffu, sm_, 0);
        float p[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) p[t] = (lane + 32 * t < N) ? ev[t] / denom : -4.0f;  // -4: never picked
        float myw = 0.0f, sum = 0.0f;
        int myid = 0;
        for (int j = 0; j < k; ++j) {
            float bv = -2.0f; int bi = N;
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int i = lane + 32 * t;
                if (p[t] > bv || (p[t] == bv && i < bi)) { bv = p[t]; bi = i; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            #pragma unroll
            for (int t = 0; t < 16; ++t) if (lane + 32 * t == bi) p[t] = -3.0f;
            const int id = (bi < 0 || bi >= N) ? 0 : bi;   // topk_route's NaN/Inf clamp
            if (lane == j) { myw = bv; myid = id; }
            sum += bv;
        }
        if (lane < k) {
            ids[(size_t)r * k + lane] = myid;
            wts[(size_t)r * k + lane] = myw / sum;
            s_ids[r * k + lane] = myid;
        }
    }
    __syncthreads();
    const int BK = M * k;                                        // host: <= 128 <= blockDim
    if (tid < BK) { const int e = s_ids[tid]; if (e >= 0 && e < N) atomicMin(&s_first[e], tid); }
    __syncthreads();
    int e_t = -1; bool first = false;
    if (tid < BK) { e_t = s_ids[tid]; first = (e_t >= 0 && e_t < N) && s_first[e_t] == tid; }
    const unsigned bal = __ballot_sync(0xffffffffu, first);
    if (lane == 0 && warp < 8) s_wc[warp] = __popc(bal);
    __syncthreads();
    int base = 0, total = 0;
    #pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) { const int c = s_wc[w8]; total += c; if (w8 < warp) base += c; }
    if (first) {
        const int slot = base + __popc(bal & ((1u << lane) - 1u));
        s_slot[tid] = slot;
        idxmap[slot] = e_t;
        offs_gu[slot] = (unsigned long long)e_t * gu_words;
        offs_d[slot] = (unsigned long long)e_t * d_words;
    }
    if (tid == 0) { esel_dev[0] = total; *done_cnt = 0; }   // re-arm for the next launch / graph replay
    __syncthreads();
    for (int e = tid; e < N; e += blockDim.x) {
        const int f = s_first[e];
        slotmap[e] = (f == 0x7fffffff) ? -1 : s_slot[f];
    }
}
// ---- (10) PLE k-major GEMV with x in smem (xq_gemm_f16_rows_km_w4). The p4c trace's per-width
// durations of xq_gemm_f16_rows_km grow +63 us per row (MR=4 body: 307/370/434/500 us at m=1..4;
// MR=8: 275/335/405/466 at m=5..8) over a 275 us byte floor: each (step, row) x load sits behind
// a runtime `r < M` branch and is issued right before its 16 FFMAs — 160*M serialized x round
// trips per thread, queued behind the in-flight 256-bit weight stream. Here x [M][K] is staged
// ONCE per CTA into smem (dynamic, M*K*2 B) and every (step, row) reads it with two LDS.128 (a
// warp-uniform broadcast); the body is templated on the EXACT M (no per-row branch).
// BIT CONTRACT (== xq_gemm_f16_rows_km == xq_gemm_f16_rows == xq_gemm_f16): per (row, column)
// acc = 0.0f, ascending 16-k steps, wv0 (k 16s..16s+7) then wv1 (16s+8..16s+15), the same
// `acc[r] += xq_h2f(w) * xq_h2f(x)` expression; the weight stream (256-bit no_allocate /
// evict_first loads, XQ_KM_U groups) is xq_gemm_f16_rows_km's. Grid (nba + nbb) blocks of 160
// (the PLE shapes: 64 + 16 CTAs, all co-resident at 2/SM with 40 KB of x at M = 8).
template <int MR>
__device__ __forceinline__ void xq_w4p_step(float (&acc)[MR], const uint4& wv0, const uint4& wv1,
                                            const __half* xs, int K, int s) {
    const __half* wh0 = (const __half*)&wv0;
    const __half* wh1 = (const __half*)&wv1;
    #pragma unroll
    for (int r = 0; r < MR; r++) {
        const uint4* xr = reinterpret_cast<const uint4*>(xs + (long long)r * K) + 2 * s;
        const uint4 xv0 = xr[0], xv1 = xr[1];
        const __half* xh0 = (const __half*)&xv0;
        const __half* xh1 = (const __half*)&xv1;
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh0[j]) * xq_h2f(xh0[j]);
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[r] += xq_h2f(wh1[j]) * xq_h2f(xh1[j]);
    }
}
template <int MR>
__device__ __forceinline__ void xq_w4p_body(__half* __restrict__ out, const uint4* __restrict__ w,
                                            const __half* xs, int N, int K, int n) {
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;
    const int s16 = K >> 4;
    const uint4* wp = w + 2 * (long long)n;
    const long long st = 2 * (long long)N;
    const int sfull = s16 - s16 % XQ_KM_U;
    #pragma unroll 1
    for (int s = 0; s < sfull; s += XQ_KM_U) {
        uint4 c0[XQ_KM_U], c1[XQ_KM_U];
        #pragma unroll
        for (int u = 0; u < XQ_KM_U; u++) xq_ld256_stream(wp + (s + u) * st, c0[u], c1[u]);
        #pragma unroll
        for (int u = 0; u < XQ_KM_U; u++) xq_w4p_step<MR>(acc, c0[u], c1[u], xs, K, s + u);
    }
    for (int s = sfull; s < s16; s++) {
        uint4 w0, w1;
        xq_ld256_stream(wp + s * st, w0, w1);
        xq_w4p_step<MR>(acc, w0, w1, xs, K, s);
    }
    #pragma unroll
    for (int r = 0; r < MR; r++) out[(long long)r * N + n] = xq_f2h(acc[r]);
}
// one entry per exact M (xq_gemm_f16_rows_km_w4m<M>, host picks by M; each gets its own register
// allocation — one switch-dispatched entry spilled the loop-invariant pointers across its cases)
template <int MR>
__device__ __forceinline__ void xq_w4p_entry(__half* __restrict__ out_a, const uint4* __restrict__ wa, int Na, int nba,
                                             __half* __restrict__ out_b, const uint4* __restrict__ wb, int Nb,
                                             const __half* __restrict__ x, int K) {
    extern __shared__ __align__(16) __half w4p_xs[];
    {
        const int nv = (MR * K) >> 3;                            // host: K % 16 == 0
        const uint4* src = reinterpret_cast<const uint4*>(x);
        uint4* dst = reinterpret_cast<uint4*>(w4p_xs);
        for (int i = threadIdx.x; i < nv; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    const bool second = (int)blockIdx.x >= nba;
    const int N = second ? Nb : Na;
    const int n = (second ? (int)blockIdx.x - nba : (int)blockIdx.x) * (int)blockDim.x + (int)threadIdx.x;
    if (n >= N) return;
    xq_w4p_body<MR>(second ? out_b : out_a, second ? wb : wa, w4p_xs, N, K, n);
}
#define XQ_W4P_ENTRY(MR)                                                                              \
extern "C" __global__ void __launch_bounds__(160, 1)                                                  \
xq_gemm_f16_rows_km_w4m##MR(__half* __restrict__ out_a, const uint4* __restrict__ wa, int Na, int nba,\
                            __half* __restrict__ out_b, const uint4* __restrict__ wb, int Nb,         \
                            const __half* __restrict__ x, int M, int K) {                             \
    XQ_PDL_ENTRY();                                                                                   \
    xq_w4p_entry<MR>(out_a, wa, Na, nba, out_b, wb, Nb, x, K);                                        \
}
XQ_W4P_ENTRY(1)
XQ_W4P_ENTRY(2)
XQ_W4P_ENTRY(3)
XQ_W4P_ENTRY(4)
XQ_W4P_ENTRY(5)
XQ_W4P_ENTRY(6)
XQ_W4P_ENTRY(7)
XQ_W4P_ENTRY(8)

// ---- (6) ONE-LAUNCH VERIFY COMMIT (xq_gdn_commit_all_w4). verify_commit ran, per GDN layer,
// xq_conv_commit (40 CTAs, ~7.5 us, latency-bound) then xq_gdn_commit_ring (192 CTAs, ~32 us),
// strictly serial: 36 x ~41 us incl. node gaps (~1.5 ms per round in the p4c trace), then
// xq_ple_ring_commit. Every layer's commit is independent of every other layer's (disjoint live
// state rows, disjoint ring / raw planes), so ONE grid does all of them: grid.y = layer (+1 row
// for the PLE ring when ple_state != null), grid.x = nh*vd/32 ring blocks + ceil(conv_dim/128)
// conv blocks, block 128 (= kd). BIT CONTRACT: each block runs its old kernel's body verbatim —
// the ring block the rank-1 replay (FMUL g, FFMA kk*delta + s, t ascending), the conv block the
// raw-row ring shift, the PLE block the f16 roll — on the same pointers the per-layer launches
// were given (per-layer state pointers from the device table lptr = [S_0..S_{L-1}, conv_0..]).
// nh_kd = nh | kd<<16; vd_conv = vd | conv_dim<<16. The folded conv commit is pure data
// movement, the verify side's conv (xq_conv1d_chunk_ns) is unchanged.
extern "C" __global__ void __launch_bounds__(128)
xq_gdn_commit_all_w4(const unsigned long long* __restrict__ lptr, int L,
                     const float* __restrict__ ring_all, long long ring_ls,
                     const __half* __restrict__ raw_all, long long raw_ls,
                     int nh_kd, int vd_conv, int ck, const int* __restrict__ slotacc,
                     const __half* __restrict__ ple_normed, __half* __restrict__ ple_state) {
    XQ_PDL_ENTRY();
    const int nh = nh_kd & 0xFFFF, kd = nh_kd >> 16;
    const int vd = vd_conv & 0xFFFF, conv_dim = vd_conv >> 16;
    const int layer = blockIdx.y;
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    if (layer >= L) {
        // ---- xq_ple_ring_commit's body (roll-only replay; one thread per PLE channel)
        if (ple_state == nullptr) return;
        const int d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d >= 10240) return;
        __half* sbase = ple_state + ((long long)slot * 10240 + d) * 9;
        __half st[9];
        for (int a = 0; a < 9; a++) st[a] = sbase[a];
        for (int t = 0; t < C; t++) {
            const float cur = xq_h2f(ple_normed[(long long)t * 10240 + d]);
            for (int a = 0; a < 8; a++) st[a] = st[a + 1];
            st[8] = xq_f2h(cur);
        }
        for (int a = 0; a < 9; a++) sbase[a] = st[a];
        return;
    }
    const int nchunk = vd / XQ_GDN_C;
    const int nring = nh * nchunk;
    if ((int)blockIdx.x < nring) {
        // ---- xq_gdn_commit_ring's body (state = this layer's S plane, ring = its ring plane)
        float* state = reinterpret_cast<float*>(lptr[layer]);
        const float* ring = ring_all + (long long)layer * ring_ls;
        __shared__ float kr_sh[XQ_GDN_RING_CMAX][128];
        __shared__ float g_sh[XQ_GDN_RING_CMAX];
        if (C < 1 || C > XQ_GDN_RING_CMAX) __trap();
        const int chunk = blockIdx.x % nchunk;
        const int head = blockIdx.x / nchunk;
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int col = chunk * XQ_GDN_C + lane;
        for (int i = threadIdx.x; i < C * kd; i += blockDim.x) {
            const int t = i / kd, r = i - t * kd;
            kr_sh[t][r] = ring[((long long)t * nh + head) * XQ_GDN_RS + vd + r];
        }
        if (threadIdx.x < C) g_sh[threadIdx.x] = ring[((long long)threadIdx.x * nh + head) * XQ_GDN_RS + vd + kd];
        constexpr int RPT = 128 / 4;
        float* S = state + ((long long)slot * nh + head) * kd * vd + col;
        float s[RPT];
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = S[(long long)(warp + 4 * i) * vd];
        __syncthreads();
        for (int t = 0; t < C; t++) {
            const float gt = g_sh[t];
            const float d = __ldg(ring + ((long long)t * nh + head) * XQ_GDN_RS + col);
            #pragma unroll
            for (int i = 0; i < RPT; i++) s[i] *= gt;
            #pragma unroll
            for (int i = 0; i < RPT; i++) s[i] = __fmaf_rn(kr_sh[t][warp + 4 * i], d, s[i]);
        }
        #pragma unroll
        for (int i = 0; i < RPT; i++) S[(long long)(warp + 4 * i) * vd] = s[i];
        return;
    }
    // ---- xq_conv_commit's body (state = this layer's conv ring, raw = its saved raw rows)
    float* cstate = reinterpret_cast<float*>(lptr[L + layer]);
    const __half* raw = raw_all + (long long)layer * raw_ls;
    const int c = ((int)blockIdx.x - nring) * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    // host-gated to the served conv width ck == 4 (register-resident: xq_conv_commit's runtime-k
    // st[8] sat in local memory); the same moves — st[j-1] = st[j], st[3] = cur, per row
    if (ck != 4) __trap();
    float* sb = cstate + ((long long)slot * conv_dim + c) * 4;
    float a0 = sb[0], a1 = sb[1], a2 = sb[2], a3 = sb[3];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(raw[(long long)t * conv_dim + c]);
        a0 = a1; a1 = a2; a2 = a3; a3 = cur;
    }
    sb[0] = a0; sb[1] = a1; sb[2] = a2; sb[3] = a3;
}
// ---- (9) DENSE v3 ACCUMULATE, deep pipeline (xq_attn_dense_acc4_w4 / _acc1_w4). The p4c eager
// trace shows xq_attn_dense_acc4 (m=8) growing 7.5 -> 34.6 us across an 1100-token request,
// ~1.9 us per 64-position tile: its V ring is double-buffered (one tile in flight while one is
// consumed), so every tile pays most of an L2/DRAM round trip. Here the ring holds NST = 5 tiles
// of T = 32 positions (4 tiles = 128 positions in flight; 44 KB static at f32 / DPT 4, still
// 2 CTAs/SM) — same grid, same threads, same per-thread work.
// BIT CONTRACT (== xq_attn_dense_acc4 / _acc1 == v2): mx = the same fmaxf over the row's chunk
// maxima; w = __expf(__fmaf_rn(raw, scale, -mx)) per (head, position) (a tile only regroups
// WHICH thread evaluates it); denom = __fadd_rn(denom, w) and acc = __fmaf_rn(w, v, acc) over t
// ascending from 0, through the same V readers (xq_dv3_ldv / xq_dv3_ldv_q8) on the same staged
// raw bytes; the closing expressions (f32/f16/fp8: f16((acc/denom)*sig(g)); q8: the in-warp H32
// un-rotation) are xq_dv3_acc_body's verbatim.
#define XQ_W4A_T   32
#define XQ_W4A_NST 5
template <int SD, int N16, bool SCH, int RS>
__device__ __forceinline__ void xq_w4a_stage_v(char* vbt, float* vsct, const XqKv& kc, int t0,
                                               int npos, int d0b, int tid) {
    constexpr int NPER = N16 + (SCH ? 1 : 0);
    for (int i = tid; i < XQ_W4A_T * NPER; i += blockDim.x) {
        const int tr = i / NPER;
        const int c = i - tr * NPER;
        const int t = t0 + tr;
        if (t < npos) {
            const char* src = kc.v + (long long)t * kc.rb;
            if (c < N16) moe_coop_cp16(vbt + tr * RS + c * 16, src + d0b + c * 16, 16);
            else         moe_coop_cp16(vsct + tr * 4, src + kc.hd, 16);
        }
    }
}
template <int G, int FMT, int DPT>
__device__ __forceinline__ void xq_w4a_body(__half* __restrict__ attn, const float* __restrict__ sc,
                                            const float* __restrict__ pmx, const XqKv& kc,
                                            const __half* __restrict__ qg, int nh, int hd,
                                            int kvh, int row, int npos, char* vb, float* vsc, float* wb) {
    constexpr int T = XQ_W4A_T, NST = XQ_W4A_NST, SD = 16 * DPT, WST = XQ_W4A_T + 4;
    constexpr int EB = FMT == XQ_KV_F32 ? 4 : (FMT == XQ_KV_F16 ? 2 : 1);
    constexpr int N16 = SD * EB / 16;
    constexpr bool Q8 = FMT >= XQ_KV_Q8;
    constexpr int RS = Q8 ? SD : SD * 4;
    constexpr bool SCH = FMT == XQ_KV_FP8 || Q8;
    static_assert(!Q8 || DPT >= 2, "q8: a thread's 16-dgl slice must hold whole 32-groups");
    static_assert(T == 32, "the weight tile below maps 16 threads x 2 positions");
    const int tid = threadIdx.x;
    const int h = tid >> 4;
    const int dgl = tid & 15;
    const int d0 = blockIdx.x * SD + dgl * DPT;
    const long long hrow = (long long)row * nh + (long long)kvh * G + h;
    const float* srow = sc + hrow * XQ_DV3_TS;
    const float scale = xq_dv3_scale(hd);
    const int ntile = (npos + T - 1) / T;
    const int d0b = blockIdx.x * SD * EB;
    // prologue: tiles 0..NST-2, one cp.async group each (empty groups keep the count aligned)
    #pragma unroll
    for (int p = 0; p < NST - 1; p++) {
        if (p < ntile) xq_w4a_stage_v<SD, N16, SCH, RS>(vb + p * (T * RS), vsc + p * (T * 4), kc, p * T, npos, d0b, tid);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    float2 rw = *reinterpret_cast<const float2*>(srow + dgl * 2);
    // mx: fmaxf over the row's chunk maxima (exactly the set {fl(raw*scale) : t < npos})
    const int nch = (npos + XQ_DV3_CH - 1) / XQ_DV3_CH;
    float mx = -INFINITY;
    for (int c = dgl; c < nch; c += 16) mx = fmaxf(mx, pmx[hrow * XQ_DV3_NCH + c]);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    float acc[DPT];
    #pragma unroll
    for (int e = 0; e < DPT; e++) acc[e] = 0.0f;
    float denom = 0.0f;
    for (int ti = 0; ti < ntile; ti++) {
        const int t0 = ti * T;
        // weights of tile ti (positions t0 + dgl*2 + {0,1} of head h); t >= npos never read
        {
            float2 w2;
            w2.x = __expf(__fmaf_rn(rw.x, scale, -mx));
            w2.y = __expf(__fmaf_rn(rw.y, scale, -mx));
            *reinterpret_cast<float2*>(wb + h * WST + dgl * 2) = w2;
        }
        // prefetch tile ti+NST-1 into the slot tile ti-1 left (freed by the loop-end barrier)
        const int pf = ti + NST - 1;
        if (pf < ntile)
            xq_w4a_stage_v<SD, N16, SCH, RS>(vb + (pf % NST) * (T * RS), vsc + (pf % NST) * (T * 4),
                                             kc, pf * T, npos, d0b, tid);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (ti + 1 < ntile) rw = *reinterpret_cast<const float2*>(srow + t0 + T + dgl * 2);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_W4A_NST - 1) : "memory");   // tile ti (this thread)
        __syncthreads();                                              // ... and everyone's, + wb
        const char* vcur = vb + (ti % NST) * (T * RS);
        const float* scur = vsc + (ti % NST) * (T * 4);
        const int tn = min(T, npos - t0);
        const float* wh = wb + h * WST;
        int t = 0;
        for (; t + 4 <= tn; t += 4) {
            const float4 w4 = *reinterpret_cast<const float4*>(wh + t);
            const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                float v[DPT];
                if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t + u, dgl, d0 >> 5);
                else              xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t + u, dgl);
                denom = __fadd_rn(denom, ws[u]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(ws[u], v[e], acc[e]);
            }
        }
        for (; t < tn; t++) {
            const float w = wh[t];
            float v[DPT];
            if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t, dgl, d0 >> 5);
            else              xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t, dgl);
            denom = __fadd_rn(denom, w);
            #pragma unroll
            for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(w, v[e], acc[e]);
        }
        __syncthreads();                                              // slot ti / wb reuse
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");           // no copy outlives the CTA
    const int head = kvh * G + h;
    if constexpr (Q8) {
        float o[DPT];
        #pragma unroll
        for (int e = 0; e < DPT; e++) o[e] = acc[e] / denom;
        #pragma unroll
        for (int st = 1; st < 32; st <<= 1) {
            if (st < DPT) {
                float nn[DPT];
                #pragma unroll
                for (int e = 0; e < DPT; e++) nn[e] = (e & st) ? __fsub_rn(o[e ^ st], o[e]) : __fadd_rn(o[e], o[e ^ st]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) o[e] = nn[e];
            } else {
                const int tm = st / DPT;
                const bool hi = (dgl & tm) != 0;
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    const float pv = __shfl_xor_sync(0xffffffffu, o[e], tm);
                    o[e] = hi ? __fsub_rn(pv, o[e]) : __fadd_rn(o[e], pv);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            const int d = d0 + e;
            const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
            attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h(__fmul_rn(o[e], XQ_R32) * xq_sig(g));
        }
    } else {
    #pragma unroll
    for (int e = 0; e < DPT; e++) {
        const int d = d0 + e;
        const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
        attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h((acc[e] / denom) * xq_sig(g));
    }
    }
}
template <int G, int DPT>
__device__ __forceinline__ void xq_w4a_entry(__half* __restrict__ attn, const float* __restrict__ sc,
                                             const float* __restrict__ pmx, const float* __restrict__ kcvc,
                                             const __half* __restrict__ qg, const int* __restrict__ slotpos,
                                             int nh_packed, int hd_packed, int max_pos) {
    constexpr int SD = 16 * DPT;
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    if (hd != 256 || nh != G * nkv) return;   // host-gated
    const int kvh = blockIdx.y;
    const int row = blockIdx.z;
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;
    __shared__ __align__(16) char vb[XQ_W4A_NST * XQ_W4A_T * SD * 4];
    __shared__ __align__(16) float vsc[XQ_W4A_NST * XQ_W4A_T * 4];
    __shared__ __align__(16) float wb[G * (XQ_W4A_T + 4)];
    if (npos > XQ_DV3_TS) {
        // host bug tripwire: the score plane cannot hold this row — loud, never truncated
        const int h = threadIdx.x >> 4, dgl = threadIdx.x & 15;
        #pragma unroll
        for (int e = 0; e < DPT; e++)
            attn[(long long)row * nh * hd + (long long)(kvh * G + h) * hd + blockIdx.x * SD + dgl * DPT + e] =
                __float2half(__int_as_float(0x7fc00000));
        return;
    }
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    switch (kc.fmt) {
        case XQ_KV_F32: xq_w4a_body<G, XQ_KV_F32, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_F16: xq_w4a_body<G, XQ_KV_F16, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_FP8: xq_w4a_body<G, XQ_KV_FP8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        default:
            // q8 / probe-only q4x: xq_dv3_acc_entry's rule (the single-row acc1 runs the DPT=2 body on
            // its first hd/32 CTAs; RS = SD bytes keeps it inside acc1's vb)
            if constexpr (DPT >= 2) {
                xq_w4a_body<G, XQ_KV_Q8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            } else {
                if ((int)blockIdx.x < hd / 32)   // block-uniform
                    xq_w4a_body<G, XQ_KV_Q8, 2>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            }
            break;
    }
}
extern "C" __global__ void __launch_bounds__(192, 2)
xq_attn_dense_acc4_w4(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_w4a_entry<XQ_DV3_G, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(192)
xq_attn_dense_acc1_w4(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_w4a_entry<XQ_DV3_G, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
// TP-4G: G=6 twins of the 5-stage-ring kernels (block 96).
extern "C" __global__ void __launch_bounds__(96, 2)
xq_attn_dense_acc4_w4_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                         const float* __restrict__ kcvc, const __half* __restrict__ qg,
                         const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_w4a_entry<XQ_DV3_G6, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(96)
xq_attn_dense_acc1_w4_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                         const float* __restrict__ kcvc, const __half* __restrict__ qg,
                         const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_w4a_entry<XQ_DV3_G6, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}

// ---- (8) DRAFT-PASS ATTENTION (m = 1 MTP head). Three latency-bound launches per pass
// (p4c: prep 0.011 + splitk4 0.021 + combine 0.009 ms):
//  * prep: xq_attn_prep runs ONE CTA per kv head (grid 2), each walking its 12 q heads through a
//    block tree with 11 barriers per head (11 us). The host now launches xq_attn_dense_prep (grid
//    m*nh = 24, one CTA per q head): the same K/V cache-word code (xq_k_norm_rope + xq_kv_put, run
//    by the gqa-first CTA of each kv head) and xq_attn_prep's q expressions verbatim (same
//    staged qstage words; xq_kv_fmt(max_pos) == kc.fmt) — no new kernel.
//  * xq_attn_dense_splitk4_w4: split-K online softmax. The old loop loads the K|V row of
//    position t right before using it (one L2/DRAM round trip per position, every pass re-reads
//    the head KV from DRAM — the pass streams ~100 MB of weights between reads) and warps 0..3
//    walk their split TWICE (heads w and w+8 sequentially). Here the CTA stages its split's K|V
//    rows through an 8-row smem ring with cp.async (groups of 4, one group in flight while one
//    is consumed) and each warp carries both of its heads in ONE walk (two independent register
//    sets). BIT CONTRACT: per (head, split) the old op sequence pinned from its SASS — dot =
//    FFMA(q, k) chain from +0 (dims lane*8 + i ascending), shfl_xor 16..1 FADDs, sc = FMUL(dot,
//    rsqrt(hd)), m_new = fmaxf, scale_f = __expf(m - m_new), wt = __expf(sc - m_new),
//    l = FFMA(l, scale_f, wt), acc = FFMA(acc, scale_f, FMUL(wt, v)) — positions ascending; the
//    K/V values come from the same xq_kv_ld8 reader on the same raw row bytes (staged verbatim).
//  * xq_attn_sel_combine_w4: the split merge with pm/pl staged once in smem and each split's
//    weight w_s = __expf(pm_s - m_g) evaluated ONCE (the old kernel evaluates the same expression
//    in both of its loops) — then l_g = FFMA(pl, w, l_g) and acc = FFMA(pacc, w, acc) over the
//    non-empty splits ascending, the same closing expression.
#define XQ_W4D_NR 8
__device__ __forceinline__ void xq_w4d_stage(char (*ring)[2][1024], const XqKv& kc, int t0, int end,
                                             int slot0, int tid) {
    const int nc = (int)(kc.rb >> 4);                // 16-B chunks per row (host: rb % 16 == 0, <= 1024)
    const int per = 2 * nc;                          // K row + V row
    for (int i = tid; i < 4 * per; i += blockDim.x) {
        const int j = i / per, r = i - j * per;
        const int t = t0 + j;
        if (t < end) {
            const int kvsel = r >= nc ? 1 : 0;
            const int c = r - kvsel * nc;
            const char* src = (kvsel ? kc.v : kc.k) + (long long)t * kc.rb + c * 16;
            moe_coop_cp16(&ring[slot0 + j][kvsel][c * 16], src, 16);
        }
    }
}
__device__ __forceinline__ void xq_w4d_step(const float (&qv)[8], const float (&kv)[8], const float (&vv)[8],
                                            float qs, float& m, float& l, float (&acc)[8]) {
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) dot = __fmaf_rn(qv[i], kv[i], dot);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
    const float sc = __fmul_rn(dot, qs);
    const float m_new = fmaxf(m, sc);
    const float scale_f = __expf(m - m_new);
    const float wt = __expf(sc - m_new);
    l = __fmaf_rn(l, scale_f, wt);
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = __fmaf_rn(acc[i], scale_f, __fmul_rn(wt, vv[i]));
    m = m_new;
}
extern "C" __global__ void __launch_bounds__(256)
xq_attn_dense_splitk4_w4(float* __restrict__ pm, float* __restrict__ pl,
                         float* __restrict__ pacc, const float* __restrict__ qstage,
                         const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                         int nh_packed, long long geom) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) char ring[XQ_W4D_NR][2][1024];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;                    // host: gqa <= 16 (warps 0..7 take heads w, w+8)
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;   // keys 0..=p (p's own K/V written by the prep)
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int split_len = (npos + nsplits - 1) / nsplits;
    const int begin = sp * split_len;
    const int end = min(begin + split_len, npos);
    const bool hasA = warp < gqa, hasB = warp + 8 < gqa;
    const int hA = kvh * gqa + warp, hB = kvh * gqa + warp + 8;
    float qA[8], qB[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { qA[i] = 0.0f; qB[i] = 0.0f; }
    if (hasA) {
        const float* qrow = qstage + ((long long)row * nh + hA) * hd;
        *reinterpret_cast<float4*>(qA) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qA + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    if (hasB) {
        const float* qrow = qstage + ((long long)row * nh + hB) * hd;
        *reinterpret_cast<float4*>(qB) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qB + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    const float qs = rsqrtf((float)hd);
    float mA = -INFINITY, lA = 0.0f, mB = -INFINITY, lB = 0.0f;
    float accA[8], accB[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { accA[i] = 0.0f; accB[i] = 0.0f; }
    const int n = end > begin ? end - begin : 0;
    const int ng = (n + 3) >> 2;
    xq_w4d_stage(ring, kc, begin, end, 0, threadIdx.x);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (ng > 1) xq_w4d_stage(ring, kc, begin + 4, end, 4, threadIdx.x);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    for (int g = 0; g < ng; g++) {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");       // group g (this thread)
        __syncthreads();                                              // ... and everyone's
        const int t0 = begin + 4 * g;
        const int tn = min(4, end - t0);
        const int s0 = (g & 1) * 4;
        for (int j = 0; j < tn; j++) {
            float kv[8], vv[8];
            xq_kv_ld8(kv, ring[s0 + j][0], kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, ring[s0 + j][1], kc.fmt, hd, lane * 8);
            if (hasA) xq_w4d_step(qA, kv, vv, qs, mA, lA, accA);
            if (hasB) xq_w4d_step(qB, kv, vv, qs, mB, lB, accB);
        }
        __syncthreads();                                              // slots s0.. free
        if (g + 2 < ng) xq_w4d_stage(ring, kc, begin + 4 * (g + 2), end, s0, threadIdx.x);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    if (hasA) {
        const long long hg = (long long)row * nh + hA;
        if (lane == 0) { pm[hg * nsplits + sp] = mA; pl[hg * nsplits + sp] = lA; }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = accA[i];
    }
    if (hasB) {
        const long long hg = (long long)row * nh + hB;
        if (lane == 0) { pm[hg * nsplits + sp] = mB; pl[hg * nsplits + sp] = lB; }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = accB[i];
    }
}
#define XQ_W4C_MAXS 256
extern "C" __global__ void __launch_bounds__(256)
xq_attn_sel_combine_w4(__half* __restrict__ attn, const float* __restrict__ pm,
                       const float* __restrict__ pl, const float* __restrict__ pacc,
                       const __half* __restrict__ qg, int nh_packed,
                       int nsplits, int rows) {
    XQ_PDL_ENTRY();
    __shared__ float s_pm[XQ_W4C_MAXS], s_pl[XQ_W4C_MAXS], s_w[XQ_W4C_MAXS];
    const int kvf = xq_kv_fmt(rows);
    const int nh = nh_packed & 0xFFFF;
    const int hd = blockDim.x;
    const int b = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int tid = threadIdx.x;
    const long long hg = (long long)b * nh + h;
    for (int s = tid; s < nsplits; s += blockDim.x) { s_pm[s] = pm[hg * nsplits + s]; s_pl[s] = pl[hg * nsplits + s]; }
    __syncthreads();
    float m_g = -INFINITY;
    for (int s = 0; s < nsplits; s++) m_g = fmaxf(m_g, s_pm[s]);
    for (int s = tid; s < nsplits; s += blockDim.x) s_w[s] = __expf(s_pm[s] - m_g);   // read only where pm > -inf
    __syncthreads();
    float l_g = 0.0f;
    for (int s = 0; s < nsplits; s++)
        if (s_pm[s] > -INFINITY) l_g = __fmaf_rn(s_pl[s], s_w[s], l_g);
    float acc = 0.0f;
    #pragma unroll 16
    for (int s = 0; s < nsplits; s++) {
        if (s_pm[s] > -INFINITY) acc = __fmaf_rn(pacc[(hg * nsplits + s) * hd + tid], s_w[s], acc);
    }
    const float g = xq_h2f(qg[(long long)b * nh * hd * 2 + (long long)h * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / l_g, kvf, tid & 31);
    attn[(long long)b * nh * hd + (long long)h * hd + tid] = xq_f2h(o * xq_sig(g));
}

// =============================================================================
// A5-P2 (prefill hc mixers; PLAN/notes_2026-09-27/P2.md). The prefill hc elementwise chain —
// xq_hc_inject (hc_post) + xq_hc_norm (next hc_pre) + xq_hc_mix4 — ran as 1024-thread one-block-
// per-(row, stream) / one-block-per-row kernels (1 CTA/SM, 10-level smem trees): norm at ~58 % and
// mix4 at ~56 % of DRAM bandwidth, plus a full extra read of the 84 MB f32 stream state per
// sublayer for the separate inject. BITWISE twins, 128-thread blocks (up to 12 CTAs/SM):
//   xq_pf_hc_inj_norm == xq_hc_inject (optional; y == null skips it) + xq_hc_norm
//   xq_pf_hc_mix_2560x4 == xq_hc_mix4 (== xq_hc_mix) at h = 2560, hc = 4
// The contract is the 1024-leaf reduction of the old kernels: leaf vt (= old threadIdx.x) sums its
// elements i = vt + 1024 j in ascending j with the same `acc += a * b` (one fma chain), then the
// same halving tree (level s2: leaf t += leaf t + s2). Here thread T (0..127) OWNS leaves
// A = 4T..4T+3 and B = 512+4T..512+4T+3, i.e. every element i = 512 k + 4T + e (k even -> A,
// j = k/2; k odd -> B, j = (k-1)/2): so the leaf chains stay whole in one thread (vector loads),
// and tree level 512 (A + B) is in-thread; 256 via smem, 128/64/32 in-lane (lane l holds positions
// l + 32 q), 16..1 by shfl_down in one warp (lane t gets v[t] + v[t+s2], exactly sm[t] += sm[t+s2]).
// Every per-element expression is written as in the old kernels (same fma contraction points).
// Requires h % 512 == 0 and h <= 4096 (launcher-checked; else the old kernels run).

__device__ __forceinline__ void xq_pf_unpack4(const uint2 v, float f[4]) {
    const __half2 a = *reinterpret_cast<const __half2*>(&v.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&v.y);
    f[0] = xq_h2f(__low2half(a)); f[1] = xq_h2f(__high2half(a));
    f[2] = xq_h2f(__low2half(b)); f[3] = xq_h2f(__high2half(b));
}
__device__ __forceinline__ uint2 xq_pf_pack4(const float f[4]) {
    __half2 a = __halves2half2(xq_f2h(f[0]), xq_f2h(f[1]));
    __half2 b = __halves2half2(xq_f2h(f[2]), xq_f2h(f[3]));
    uint2 r;
    r.x = *reinterpret_cast<const unsigned int*>(&a);
    r.y = *reinterpret_cast<const unsigned int*>(&b);
    return r;
}

// One warp finishes a 1024-leaf halving tree whose level 512 is already in sm[0..512)
// (sm[t] = leaf t + leaf t+512). Returns the root in lane 0.
__device__ __forceinline__ float xq_pf_tree512(const float* sm, int lane) {
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

// Fused inject + norm over one (row b, stream s) slab per CTA; grid hc * B, block 128.
// out = hn (f16), x = resid (f32, updated in place when y != null), y = the sublayer output (f16
// [B][h]) or null, inj = [B][hc] gates, w = hc_norm weight [hc][h].
extern "C" __global__ void __launch_bounds__(128)
xq_pf_hc_inj_norm(__half* __restrict__ out, float* __restrict__ x, const __half* __restrict__ y,
                  const float* __restrict__ inj, const __half* __restrict__ w, int h, int hc, int B, float eps) {
    XQ_PDL_ENTRY();
    const int blk = blockIdx.x;
    if (blk >= hc * B) return;
    const int b = blk / hc, s = blk - b * hc;
    __shared__ float sm[512];
    __shared__ float s_inv;
    const int T = threadIdx.x;
    float* xs = x + (long long)blk * h;                  // == b*h*hc + s*h
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
                xq_pf_unpack4(*reinterpret_cast<const uint2*>(yr + i), yf);
                #pragma unroll
                for (int e = 0; e < 4; ++e) t[e] += g * yf[e];          // xq_hc_inject's fma
                *reinterpret_cast<float4*>(xs + i) = make_float4(t[0], t[1], t[2], t[3]);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[k][e] = t[e];
                if (k & 1) pb[e] += t[e] * t[e]; else pa[e] += t[e] * t[e];   // xq_hc_norm's sum_sq
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) sm[4 * T + e] = pa[e] + pb[e];              // level 512
    __syncthreads();
    if (T < 32) {
        const float r = xq_pf_tree512(sm, T);
        if (T == 0) s_inv = rsqrtf(r / (float)h + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const __half* ws = w + (long long)s * h;
    __half* os = out + (long long)blk * h;
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float wf[4], o[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ws + i), wf);
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = v[k][e] * inv * (1.0f + wf[e]);
            *reinterpret_cast<uint2*>(os + i) = xq_pf_pack4(o);
        }
    }
}

// hc mix + inject gates, one row per CTA; grid B, block 128 (4 warps = the HC tree warps).
//   x[i]   = f16((sum_s sig(u[s*H+i]) * hn[s*H+i]) / HC)            (ascending s: k = s*NK + m)
//   inj[t] = 2 * sig((sum_c winj[t,c] * hn[c]) / HC)                 (1024-leaf tree, as xq_hc_mix4)
template <int NK, int HC>
__device__ __forceinline__ void xq_pf_hc_mix_body(__half* __restrict__ x, float* __restrict__ inj,
                                                  const __half* __restrict__ hn, const __half* __restrict__ u,
                                                  const __half* __restrict__ winj, int B) {
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
        xq_pf_unpack4(*reinterpret_cast<const uint2*>(hn + c), hf);
        xq_pf_unpack4(*reinterpret_cast<const uint2*>(u + c), uf);
        const int m = k % NK;
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[m][e] += xq_sig(uf[e]) * hf[e];
        if (dj) {
            #pragma unroll
            for (int t = 0; t < HC; ++t) {
                float wf[4];
                xq_pf_unpack4(*reinterpret_cast<const uint2*>(winj + (long long)t * N + c), wf);
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
        *reinterpret_cast<uint2*>(x + 512 * m + 4 * T) = xq_pf_pack4(o);
    }
    if (dj) {
        __shared__ float sm[HC][512];
        #pragma unroll
        for (int t = 0; t < HC; ++t)
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm[t][4 * T + e] = pa[t][e] + pb[t][e];   // level 512
        __syncthreads();
        const int wp = T >> 5, lane = T & 31;
        if (wp < HC) {
            const float r = xq_pf_tree512(sm[wp], lane);
            if (lane == 0) inj[wp] = 2.0f * xq_sig(r / (float)HC);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
xq_pf_hc_mix_2560x4(__half* __restrict__ x, float* __restrict__ inj, const __half* __restrict__ hn,
                    const __half* __restrict__ u, const __half* __restrict__ winj, int B) {
    XQ_PDL_ENTRY();
    xq_pf_hc_mix_body<5, 4>(x, inj, hn, u, winj, B);
}

// ===========================================================================
// TP-A (EXL3 TP=2 bring-up, rung 3 — PLAN/TP2_DESIGN_EXL3_2026-09-28.md §1.3/§2): expert
// parallelism. Each rank holds a STATIC subset of every trunk MoE layer's routed experts (the
// deal is a load-time table), the router/top-k stay replicated (every rank computes the same
// global ids/wts), and the rank's routed contribution is combined into an FP32 partial that one
// all-reduce per MoE layer sums (canonical rank0 + rank1 order, K2 mode 2) before ONE f16
// rounding. The kernels below are the only EP-specific device code; everything between them is
// the served expert path running on the rank's local tables.
// ===========================================================================

// == xq_moe_route restricted to the experts this rank holds. ep_lut[e] = local stacked index of
// global expert e on this rank, -1 = held by another rank. slotmap[e] = slot (first-seen over ids
// row-major among the LOCAL experts) or -1; idxmap[slot] = the LOCAL index (the rank's suh/svh
// tables are its stacked subset); offs_* = local word offsets into the rank's stacked trellis.
// ids/wts stay GLOBAL (the combine reads them through slotmap, which is -1 for remote experts).
extern "C" __global__ void xq_moe_route_ep(const int* __restrict__ ids, int m, int topk, int ne,
                                           const int* __restrict__ ep_lut,
                                           int* __restrict__ slotmap, int* __restrict__ idxmap,
                                           int* __restrict__ esel_dev,
                                           unsigned long long* __restrict__ offs_gu,
                                           unsigned long long* __restrict__ offs_d,
                                           unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    __shared__ int cnt;
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < m * topk; t++) {
            int e = ids[t];
            if (e >= 0 && e < ne && ep_lut[e] >= 0 && slotmap[e] == -1) {
                slotmap[e] = c;
                idxmap[c] = ep_lut[e];
                c++;
            }
        }
        cnt = c;
        esel_dev[0] = c;
    }
    __syncthreads();
    for (int s = tid; s < cnt; s += blockDim.x) {
        int le = idxmap[s];
        offs_gu[s] = (unsigned long long)le * gu_words;
        offs_d[s] = (unsigned long long)le * d_words;
    }
}

// == xq_moe_combine (same sgv tree, same j order, the WP20 combine's explicit roundings) with an
// FP32 output = this rank's PARTIAL. add_shared = 1 on exactly one rank (rank 0): the replicated
// shared expert enters the sum once. A rank with no local expert for a row writes that row's
// shared term (rank 0) or exact +0.0 (other ranks) — never leaves the row unwritten (the WP20
// down epilogue writes moe_out only from a last arrival, which a rank with esel == 0 never has).
extern "C" __global__ void xq_moe_combine_ep(float* __restrict__ out, const __half* __restrict__ y_d,
                                             const __half* __restrict__ y_sh,
                                             const int* __restrict__ ids, const float* __restrict__ wts,
                                             const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                             const __half* __restrict__ x, int k, int h, int M,
                                             int add_shared) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(y_sh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            int e = ids[(long long)m * k + j];
            int slot = slotmap[e];
            if (slot >= 0)
                acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(y_d[((long long)slot * M + m) * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
}

// f32 -> f16 (round-to-nearest-even, xq_f2h) — the single rounding after the EP all-reduce.
extern "C" __global__ void xq_cvt_f32_f16(__half* __restrict__ out, const float* __restrict__ in, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = xq_f2h(in[i]);
}

// =============================================================================
// TP-B (EXL3 TP=2 rung 4 — PLAN/TP2_DESIGN_EXL3_2026-09-28.md §1.3/§2): head-sharded
// attention/GDN + row-parallel dense (o_proj / out_proj / shared down) + expert-parallel
// PREFILL. Row-parallel chains reduce the PRE-output-Hadamard y_raw in FP32 (the output H128 and
// svh are linear and run once, after the all-reduce, on the f16-rounded sum — the same single
// rounding point as the TP=1 chain), so TP=2 differs from TP=1 only in the reassociation of the
// K split.
// =============================================================================

// Split-K combine into FP32: the SAME ascending slab sum as exl3_ks_combine, no f16 rounding —
// this rank's partial of y_raw for the row-parallel all-reduce.
extern "C" __global__ void xq_ks_combine_f32(const float* __restrict__ ws, float* __restrict__ out,
                                             int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    out[(size_t)m * N + n] = acc;
}

// f16 -> f32 widening (exact): a wide-M (prefill) row-parallel chain's f16 y_raw partial.
extern "C" __global__ void xq_cvt_f16_f32(float* __restrict__ out, const __half* __restrict__ in, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// EP prefill routing: global expert ids -> this rank's LOCAL stacked ids (-1 = held by another
// rank), and cand[i] = -1 for every pick (xq_moe_pf_place then writes the local picks' rows; a
// remote pick keeps -1 and the EP combine skips it).
extern "C" __global__ void xq_moe_ids_ep(const int* __restrict__ ids, int* __restrict__ ids_l,
                                         int* __restrict__ cand, const int* __restrict__ ep_lut,
                                         int r, int ne) {
    XQ_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r) return;
    const int e = ids[i];
    ids_l[i] = (e >= 0 && e < ne) ? ep_lut[e] : -1;
    cand[i] = -1;
}

// EP prefill: the compact row list holds offs_row[nl] (<= r) local rows; the row-wise glue
// kernels (had_suh_rows / had_svh_rows / gate_mul) are sized by r. Give rows [offs_row[nl], r)
// a VALID (token 0, local expert 0) mapping — their values are computed and never read (no
// tile, no cand points at them) — so no kernel reads an uninitialized index.
extern "C" __global__ void xq_moe_pf_ep_fill(const int* __restrict__ offs_row, int nl, int r,
                                             int* __restrict__ row_tok, int* __restrict__ row_eidx) {
    XQ_PDL_ENTRY();
    const int r0 = offs_row[nl];
    for (int i = r0 + blockIdx.x * blockDim.x + threadIdx.x; i < r; i += gridDim.x * blockDim.x) {
        row_tok[i] = 0;
        row_eidx[i] = 0;
    }
}

// EP prefill combine (xq_moe_combine_rows' sgv tree and j order) into an FP32 PARTIAL:
// add_shared = the shared-expert term enters on this rank (rank 0 when the shared expert is
// replicated; every rank when it is row-sharded — each adds sgv * its own partial); a pick with
// cand < 0 (remote expert) contributes nothing.
extern "C" __global__ void xq_moe_combine_rows_ep(float* __restrict__ out, const __half* __restrict__ yd,
                                                  const __half* __restrict__ ysh,
                                                  const int* __restrict__ cand, const float* __restrict__ wts,
                                                  const __half* __restrict__ sg,
                                                  const __half* __restrict__ x, int k, int h, int M,
                                                  int add_shared) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(ysh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            const int cr = cand[(long long)m * k + j];
            if (cr >= 0) acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(yd[(long long)cr * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
}

// TP-4M1: xq_moe_combine_rows_ep2<KM> = xq_moe_combine_rows_ep with every memory access vectorised and all the
// row's gathers issued up front. Per output element the arithmetic is the old kernel's:
//   acc = add_shared ? __fmul_rn(sgv, h2f(ysh)) : 0;  for j ascending, cand >= 0: acc = __fmaf_rn(wts_j, h2f(yd_j), acc)
// and sgv = xq_sig(tree) where tree is the old kernel's 1024-lane halving tree over the per-lane partials
//   part_v = ((0 + sg[v]*x[v]) + sg[v+1024]*x[v+1024]) + sg[v+2048]*x[v+2048]   (ascending, c < h only).
// The tree is reproduced EXACTLY (same pairs, same levels) but evaluated in one warp: lane l loads the 32 values
// sm[l + 32 j] and runs the levels s2 = 512 .. 32 in registers (pair (j, j + s2/32) per level), then s2 = 16 .. 1
// by shuffles — each add has the same two operands the old smem tree gave it, so sm[0] is bit-identical.
// One block = one row, h/8 threads, each owns 8 consecutive output columns (16-B yd / ysh loads, 2 x float4 store).
// Requires h == 2560 (320 threads), k <= KM, 16-B aligned yd / ysh / out / x / sg bases.
__device__ __forceinline__ void xq_u4_f8(const uint4& u, float* f) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
    const float2 d = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y; f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}
template <int KM>
__device__ __forceinline__ void xq_moe_combine_ep2_body(float* __restrict__ out, const __half* __restrict__ yd,
        const __half* __restrict__ ysh, const int* __restrict__ cand, const float* __restrict__ wts,
        const __half* __restrict__ sg, const __half* __restrict__ x, int k, int h, int M, int add_shared) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float sm[1024];
    __shared__ float sgv_s;
    const int tid = threadIdx.x;
    int cr[KM];
    float wv[KM];
    #pragma unroll
    for (int j = 0; j < KM; ++j) {
        cr[j] = -1; wv[j] = 0.0f;
        if (j < k) { cr[j] = cand[(long long)m * k + j]; wv[j] = wts[(long long)m * k + j]; }
    }
    uint4 yv[KM];
    #pragma unroll
    for (int j = 0; j < KM; ++j)
        if (cr[j] >= 0) yv[j] = *reinterpret_cast<const uint4*>(yd + (long long)cr[j] * h + tid * 8);
    uint4 sv = make_uint4(0, 0, 0, 0);
    if (add_shared) sv = *reinterpret_cast<const uint4*>(ysh + (long long)m * h + tid * 8);
    float sgv = 0.0f;
    if (add_shared) {
        if (tid < 128) {
            float part[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) part[u] = 0.0f;
            for (int c0 = tid * 8; c0 < h; c0 += 1024) {
                float xf[8], gf[8];
                xq_u4_f8(*reinterpret_cast<const uint4*>(x + (long long)m * h + c0), xf);
                xq_u4_f8(*reinterpret_cast<const uint4*>(sg + c0), gf);
                #pragma unroll
                for (int u = 0; u < 8; ++u) part[u] += gf[u] * xf[u];
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) sm[tid * 8 + u] = part[u];
        }
        __syncthreads();
        if (tid < 32) {
            float a[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) a[j] = sm[tid + 32 * j];
            #pragma unroll
            for (int j = 0; j < 16; ++j) a[j] = a[j] + a[j + 16];   // s2 = 512
            #pragma unroll
            for (int j = 0; j < 8; ++j) a[j] = a[j] + a[j + 8];     // s2 = 256
            #pragma unroll
            for (int j = 0; j < 4; ++j) a[j] = a[j] + a[j + 4];     // s2 = 128
            #pragma unroll
            for (int j = 0; j < 2; ++j) a[j] = a[j] + a[j + 2];     // s2 = 64
            float val = a[0] + a[1];                                // s2 = 32
            #pragma unroll
            for (int s2 = 16; s2 > 0; s2 >>= 1) val = val + __shfl_down_sync(0xFFFFFFFFu, val, s2);
            if (tid == 0) sgv_s = xq_sig(val);
        }
        __syncthreads();
        sgv = sgv_s;
    }
    float acc[8];
    if (add_shared) {
        float sf[8];
        xq_u4_f8(sv, sf);
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc[u] = __fmul_rn(sgv, sf[u]);
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc[u] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < KM; ++j) {
        if (cr[j] >= 0) {
            float yf[8];
            xq_u4_f8(yv[j], yf);
            #pragma unroll
            for (int u = 0; u < 8; ++u) acc[u] = __fmaf_rn(wv[j], yf[u], acc[u]);
        }
    }
    float4* o = reinterpret_cast<float4*>(out + (long long)m * h + tid * 8);
    o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}
extern "C" __global__ void __launch_bounds__(320) xq_moe_combine_rows_ep2_k10(float* __restrict__ out,
        const __half* __restrict__ yd, const __half* __restrict__ ysh, const int* __restrict__ cand,
        const float* __restrict__ wts, const __half* __restrict__ sg, const __half* __restrict__ x,
        int k, int h, int M, int add_shared) {
    XQ_PDL_ENTRY();
    xq_moe_combine_ep2_body<10>(out, yd, ysh, cand, wts, sg, x, k, h, M, add_shared);
}
extern "C" __global__ void __launch_bounds__(320) xq_moe_combine_rows_ep2_k16(float* __restrict__ out,
        const __half* __restrict__ yd, const __half* __restrict__ ysh, const int* __restrict__ cand,
        const float* __restrict__ wts, const __half* __restrict__ sg, const __half* __restrict__ x,
        int k, int h, int M, int add_shared) {
    XQ_PDL_ENTRY();
    xq_moe_combine_ep2_body<16>(out, yd, ysh, cand, wts, sg, x, k, h, M, add_shared);
}

// =============================================================================
// TP-G (T2 lever #2, PLAN/notes_2026-09-28/TP-G.md): the DECODE all-reduce (<= 16 rows) with K1
// FOLDED into the producing kernel's epilogue and a multi-block K2 with an optional GPU-side
// receive. Same doorbell protocol as gpu_batch.cu's K1/K2 (native/tp_doorbell.h I1-I9: same rings,
// same len/tail tags, same proxy, same wire bytes, fp32 payload) — only WHO runs each step moves:
//   producer+K1 : every block computes its share of the fp32 partial EXACTLY as the unfused producer
//                 (xq_ks_combine_f32 / xq_moe_combine_ep / xq_cvt_f16_f32, same expressions) into the
//                 device-memory local partial and arrives on a counter; the LAST block runs K1: the I3
//                 reuse gate (plain loads, I6), the partial -> send slot copy, fence.sc.sys, tail epoch +
//                 len tag, c->epoch, RELEASE gpu_ready — the single-block K1's publish sequence.
//   K2 (dec)    : every block gates itself: the proxy's validated release (cpu_done >= e, I5) OR, with
//                 io bit 3, the GPU-side placement proof tp_wait_add_g uses (peer_committed >= e hint +
//                 the slot's generation-tagged tail == e, then fence.acquire.sys); publishes RX_DONE = e
//                 (the proxy then skips validating epochs the GPU already proved — the slot may hold a
//                 later generation by the time the proxy looks). Sum = tp_reduce_body mode 2's canonical
//                 lower + upper fp32 add, rounded ONCE to f16 with __float2half_rn (= xq_cvt_f32_f16)
//                 when io bit 1 — the separate cvt launch is gone. Bitwise equal to K1 + K2 + cvt.
// World-general gates (world>2: partner(e-R)'s per-QP tx_retired, partner(e)'s per-peer cpu_done +
// the R9 tail equality, per-round recv rings, the round-k canonical order); the host takes this path
// at world 2 only (f16 out would round a world>2 intermediate round sum — TP-F.md §7 item 1).
// =============================================================================
__device__ __forceinline__ unsigned long long xtp_ld(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void xtp_st_rel(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void xtp_fence_acq() { asm volatile("fence.acquire.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long xtp_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned long long* xtp_flag(tp_dev_ctx* c, int off) {
    return (unsigned long long*)((char*)c->flags + off);
}
__device__ __forceinline__ int xtp_partner(tp_dev_ctx* c, unsigned long long e) {
    return (int)c->rank ^ (1 << (int)(e % (unsigned long long)c->rounds));
}
__device__ __forceinline__ unsigned long long* xtp_nway(tp_dev_ctx* c, int peer, size_t sub) {
    return (unsigned long long*)((char*)c->nway_flags + (size_t)peer * TP_CL + sub);
}
__device__ __forceinline__ unsigned char* xtp_send_slot(tp_dev_ctx* c, unsigned long long e) {
    return c->send_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
__device__ __forceinline__ const unsigned char* xtp_recv_slot(tp_dev_ctx* c, unsigned long long e) {
    if (c->world > 2) {
        const unsigned long long round = e % (unsigned long long)c->rounds;
        return c->nway_recv + (round * (unsigned long long)TP_RING_SLOTS + (e & (TP_RING_SLOTS - 1)))
                              * (unsigned long long)c->slot_stride;
    }
    return c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}

// TP-4F2 (--tp-reduce single): the I3 reuse gate of the uniform one-shot ctx. Every epoch of that ctx is pushed
// to ALL THREE peers, so the send slot's previous owner (epoch e-R) is free only when EVERY peer QP retired it:
// warp 0, lane l < 3 polls tx_retired[rank ^ (l+1)] >= e-R (three independent host-memory loads in flight at
// once, not three serial round trips on the critical path), the warp votes, same bounded deadline / abort
// word / status 11 as the tree gate. Reached only when c->oneshot (the dedicated ctx; rails 1 and 2 have 0).
// World 4 only (the host refuses the flag elsewhere). All threads call; returns 1 = aborted.
__device__ __forceinline__ int xtp_k1_gate_os(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab_os;
    if (threadIdx.x < 32) {
        const int lane = (int)threadIdx.x;
        if (lane == 0) s_ab_os = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = (lane < 3) ? xtp_nway(c, (int)c->rank ^ (lane + 1), TP_NWAY_TX_OFF) : nullptr;
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            auto retired = [&]() -> bool { return ret == nullptr || xtp_ld(ret) >= tgt; };
            bool ok = __all_sync(0xffffffffu, retired());
            if (!ok) {
                if (lane == 0) c->gate_waits += 1;
                const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                for (;;) {
                    if (__all_sync(0xffffffffu, retired())) break;
                    int stop = 0;
                    if (lane == 0) {
                        if (xtp_ld(ab)) stop = 1;
                        else if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); stop = 1; }
                    }
                    stop = __shfl_sync(0xffffffffu, stop, 0);
                    if (stop) { if (lane == 0) s_ab_os = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            xtp_fence_acq();                        // every lane (no lane reads lane 0's shared flag mid-warp)
        }
    }
    __syncthreads();
    return s_ab_os;
}

// The I3 reuse gate (tp_gate_copy_signal_mb's, verbatim): thread 0 waits tx_retired >= e-R, bounded
// (K1_GATE_WAIT_NS -> device status 11, I9). All threads call; returns 1 = aborted (store nothing).
__device__ __forceinline__ int xtp_k1_gate(tp_dev_ctx* c, unsigned long long e) {
    if (c->oneshot) return xtp_k1_gate_os(c, e);        // TP-4F2: the uniform one-shot ctx only (rails 1/2: 0)
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = (c->world > 2)
                ? xtp_nway(c, xtp_partner(c, tgt), TP_NWAY_TX_OFF) : xtp_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            // TP-I (audit C7): the only caller is xtp_k1_last, where exactly ONE block (the last to
            // arrive, an arbitrary blockIdx) runs the gate, so count on its thread 0. The old
            // `blockIdx.x == 0` guard (inherited from the multi-block K1m, gpu_batch.cu, where every
            // block gates) made the folded decode path under-report binds at grid > 1. Diagnostic only.
            if (xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) {
                    // device status 11 (I9; no printf: it would put a stack frame on the producer —
                    // the host reports the abort loudly at its next agree / sync)
                    xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11);
                    s_ab = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

// TP-G v2 fold (the "last-block" K1): the producer blocks store ONLY their device-memory partial
// (no system-scope traffic per block: the v1 fold, where every block wrote the host-memory send slot
// and ran fence.sc.sys, cost ~+11 us per 80 KiB barrier on the transport bench), arrive on the counter
// (fence.gpu), and the LAST block runs K1 itself: the I3 gate, the whole partial -> send slot copy
// (16 B vectors, L2-hot), fence.sc.sys, tail + len tag, release gpu_ready. One launch fewer than
// producer + K1, same bytes on the wire, same publish sequence. nbytes % 16 == 0 (host).
__device__ __forceinline__ void xtp_k1_last(tp_dev_ctx* c, unsigned* arrive, const float* part, unsigned nbytes) {
    __shared__ int s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                                // this block's partial stores, device scope
        const unsigned prev = atomicAdd(arrive, 1u);
        s_last = (prev == gridDim.x * gridDim.y * gridDim.z - 1u);
        if (s_last) __threadfence();                    // acquire side of the last-block pattern
    }
    __syncthreads();
    if (!s_last) return;
    const unsigned long long e = c->epoch + 1;          // the pre-kernel counter (only the last block reads it)
    const int ab = xtp_k1_gate(c, e);
    if (!ab) {
        uint4* dst = (uint4*)xtp_send_slot(c, e);
        const uint4* s4 = (const uint4*)part;
        // TP-4Z16a: the copy is latency-bound (ONE block, one load + one store per thread per trip): four loads in flight per
        // trip halve it (80 KiB: 4.0 -> 2.0 us, 160 KiB: 7.5 -> 3.3 us, kernel-only microbenchmark). Same bytes, same slot.
        const unsigned n16 = nbytes >> 4;
        unsigned i = threadIdx.x;
        for (; i + 3u * blockDim.x < n16; i += 4u * blockDim.x) {
            const uint4 v0 = __ldcg(s4 + i), v1 = __ldcg(s4 + i + blockDim.x),
                        v2 = __ldcg(s4 + i + 2u * blockDim.x), v3 = __ldcg(s4 + i + 3u * blockDim.x);
            dst[i] = v0; dst[i + blockDim.x] = v1; dst[i + 2u * blockDim.x] = v2; dst[i + 3u * blockDim.x] = v3;
        }
        for (; i < n16; i += blockDim.x) dst[i] = __ldcg(s4 + i);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // the slot stores, system scope, before the tail
        *arrive = 0u;
        c->epoch = e;
        if (c->world > 2 && c->qp_mask)
            c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << xtp_partner(c, e);
        if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
            unsigned char* slot = xtp_send_slot(c, e);
            const unsigned wire = (nbytes + 7u) & ~7u;
            *(unsigned long long*)(slot + wire) = e;
            c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
            xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
        }
    }
}

extern "C" __global__ void xq_ks_combine_f32_k1l(tp_dev_ctx* c, const float* __restrict__ ws, float* __restrict__ out,
                                                 int M, int N, int KS, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        const int m = idx / N;
        const int n = idx - m * N;
        float acc = 0.0f;
        for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
        out[idx] = acc;
    }
    xtp_k1_last(c, arrive, out, (unsigned)(M * N) * 4u);
}

extern "C" __global__ void xq_moe_combine_ep_k1l(tp_dev_ctx* c, float* __restrict__ out, const __half* __restrict__ y_d,
                                                 const __half* __restrict__ y_sh,
                                                 const int* __restrict__ ids, const float* __restrict__ wts,
                                                 const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                                 const __half* __restrict__ x, int k, int h, int M,
                                                 int add_shared, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int m = blockIdx.x;                           // host: grid == M
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int cc = tid; cc < h; cc += bs) part += xq_h2f(sg[cc]) * xq_h2f(x[(long long)m * h + cc]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(y_sh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            int ee = ids[(long long)m * k + j];
            int slot = slotmap[ee];
            if (slot >= 0)
                acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(y_d[((long long)slot * M + m) * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
    xtp_k1_last(c, arrive, out, (unsigned)(M * h) * 4u);
}

// The ks == 1 decode chain: its partial is the GEMM's f16 y_raw — widen it (exact, xq_cvt_f16_f32)
// into the fp32 local partial, then the last block runs K1.
extern "C" __global__ void xq_cvt_f16_f32_k1l(tp_dev_ctx* c, float* __restrict__ out, const __half* __restrict__ in,
                                              int n, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
    xtp_k1_last(c, arrive, out, (unsigned)n * 4u);
}

// TP-4Z2: L2 run-ahead of the NEXT kernel's weights from the otherwise idle threads of a K2 block. A K2 spends tens of
// microseconds waiting for the peer's payload with the DRAM bus idle, and the kernel behind it (the hc mixer, 6.5 MB of int8
// weights) is weight-bound and streams cold from DRAM. The weights never depend on this reduce's output, so they can be
// pulled into L2 now. Pure prefetch: no data path, no output byte, and a 0 byte count (the default) is a no-op. The threads
// below `skip` (the spinners) never take part, so the wait is not delayed; the rest of the grid stripes the two ranges by
// 128-byte line.
__device__ __forceinline__ void xq_l2_prefetch2(unsigned long long p0, unsigned n0, unsigned long long p1, unsigned n1,
                                                unsigned skip) {
    if (n0 == 0u && n1 == 0u) return;
    if (threadIdx.x < skip) return;
    const unsigned per = blockDim.x - skip;
    const unsigned t = blockIdx.x * per + (threadIdx.x - skip);
    const unsigned nt = gridDim.x * per;
    for (unsigned off = t * 128u; off < n0; off += nt * 128u) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + off));
    for (unsigned off = t * 128u; off < n1; off += nt * 128u) asm volatile("prefetch.global.L2 [%0];" :: "l"(p1 + off));
}

// K2 (decode): gate per block (see the section header), then out = lower + upper (fp32, the canonical
// order), f16 (__float2half_rn) when io bit 1. io bit 0: `local` is f16 (widened exactly); bit 3: the
// GPU-side receive. n % 4 == 0 (host). out may alias local (same-thread read-then-write).
extern "C" __global__ void xq_tp_wait_add_dec(tp_dev_ctx* c, void* out, const void* local, int n, int io,
                                              unsigned long long pf0, unsigned pfn0, unsigned long long pf1, unsigned pfn1) {
    XQ_PDL_ENTRY();                                     // the producer+K1 wrote c->epoch: wait for it
    xq_l2_prefetch2(pf0, pfn0, pf1, pfn1, 1u);          // TP-4Z2: thread 0 spins; every other thread run-ahead-prefetches
    const unsigned long long e = c->epoch;
    const unsigned char* slot = xtp_recv_slot(c, e);
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool nw = c->world > 2, g = (io & 8) != 0;
        const unsigned long long* done = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF) : xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF) : xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* tail = (const unsigned long long*)(slot + (((unsigned)n * 4u + 7u) & ~7u));
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            // v1: the proxy's validated release (world>2 adds R9's tail equality: cpu_done is a watermark)
            if (xtp_ld(done) >= e && (!nw || xtp_ld(tail) == e)) break;
            // GPU-side receive: the commit hint arms, the payload's own generation-tagged tail proves
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }        // the proxy's tail/len guards abort a lost payload
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    if (s_ab) return;
    const int lower = (c->world > 2) ? ((c->rank & (1 << (int)(e % (unsigned long long)c->rounds))) == 0) : (c->rank == 0);
    const float4* pe = (const float4*)slot;
    const int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 l4;
        if (io & 1) {
            const uint2 u = ((const uint2*)local)[i];
            const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            l4 = make_float4(f01.x, f01.y, f23.x, f23.y);
        } else {
            l4 = ((const float4*)local)[i];
        }
        const float4 p4 = pe[i];
        const float4 a = lower ? l4 : p4;
        const float4 b = lower ? p4 : l4;
        float4 r;
        r.x = a.x + b.x; r.y = a.y + b.y; r.z = a.z + b.z; r.w = a.w + b.w;
        if (io & 2) {
            __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
            uint2 u;
            u.x = *reinterpret_cast<unsigned*>(&h01);
            u.y = *reinterpret_cast<unsigned*>(&h23);
            ((uint2*)out)[i] = u;
        } else {
            ((float4*)out)[i] = r;
        }
    }
}

// TP-4F2 K2 (--tp-reduce single): the ONE-STAGE world-4 decode/verify reduce. The dedicated uniform one-shot
// ctx carries ONE epoch per logical reduce (the producer's folded K1 published this rank's fp32 partial to all
// three peers); here every block gates on all three peers (warp 0, lane l < 3 = peer rank ^ (l+1): the proxy's
// validated release cpu_done[p] >= e AND the sender-indexed slot's generation-tagged tail == e, R9 — the same two
// conditions the rd K2 gates on, per peer), then forms
//     out = f16( (p0 + p2) + (p1 + p3) )          p_r = rank r's fp32 partial (own = `local`)
// which is EXACTLY the rd two-stage association (stage 0 pairs {0,2},{1,3} — epoch parity, PLAN/TP-4F2_REPORT.md
// section 1 — stage 1 adds the pair sums; fp32 add is commutative so the grouping is rank-independent), rounded
// ONCE with __floats2half2_rn like the rd last stage. n % 4 == 0; out must not alias local (the host passes
// y_raw / moe_out vs the fp32 `part`).
extern "C" __global__ void xq_tp_wait_add_dec_single(tp_dev_ctx* c, __half* out, const float* local, int n,
                                                     unsigned long long pf0, unsigned pfn0, unsigned long long pf1, unsigned pfn1) {
    XQ_PDL_ENTRY();                                     // the producer+K1 wrote c->epoch: wait for it
    xq_l2_prefetch2(pf0, pfn0, pf1, pfn1, 32u);         // TP-4Z2: warp 0 spins; the other warps run-ahead-prefetch
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    const int rank = (int)c->rank;
    const size_t ring_e = (size_t)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x < 32) {
        const int lane = (int)threadIdx.x;
        if (lane == 0) s_ab = 0;
        const int peer = (lane < 3) ? (rank ^ (lane + 1)) : -1;
        const unsigned long long* done = nullptr;
        const unsigned long long* tail = nullptr;
        if (peer >= 0) {
            done = xtp_nway(c, peer, TP_NWAY_CPU_OFF);
            tail = (const unsigned long long*)(c->oneshot_recv
                   + ((size_t)peer * TP_RING_SLOTS + ring_e) * (size_t)c->slot_stride + wire);
        }
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            const bool mine = (peer < 0) || (xtp_ld(done) >= e && xtp_ld(tail) == e);
            if (__all_sync(0xffffffffu, mine)) break;
            int stop = (lane == 0) ? (int)(xtp_ld(ab) != 0) : 0;   // the proxy's guards abort a lost payload
            stop = __shfl_sync(0xffffffffu, stop, 0);
            if (stop) { if (lane == 0) s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        xtp_fence_acq();                                // every lane, then the block barrier publishes it
    }
    __syncthreads();
    if (s_ab) return;
    const float4* pr[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q)
        pr[q] = (const float4*)(c->oneshot_recv + ((size_t)q * TP_RING_SLOTS + ring_e) * (size_t)c->slot_stride);
    const float4* lo = (const float4*)local;
    const int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = (q == rank) ? lo[i] : pr[q][i];
        float4 r;
        r.x = (v[0].x + v[2].x) + (v[1].x + v[3].x);
        r.y = (v[0].y + v[2].y) + (v[1].y + v[3].y);
        r.z = (v[0].z + v[2].z) + (v[1].z + v[3].z);
        r.w = (v[0].w + v[2].w) + (v[1].w + v[3].w);
        __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
        uint2 u;
        u.x = *reinterpret_cast<unsigned*>(&h01);
        u.y = *reinterpret_cast<unsigned*>(&h23);
        ((uint2*)out)[i] = u;
    }
}

// =============================================================================
// TP-I3 (c1): K2m FOLDED into the prefill consumers (world 2, pipelined transport, row-aligned
// epochs: an epoch carries whole rows [r0, r1) of the chunk). Each kernel takes K2m's place in the
// schedule (same stream position, same `lag`, so e = c->epoch - lag is K2m's epoch), gates like K2m
// (world 2: the proxy's validated cpu_done >= e, abort-aware, the critical-spin shape of
// tp_spin_until_ge), reads the peer's partial straight from the recv slot and the local partial, forms
// the canonical lower + upper fp32 sum and the single __float2half_rn — K2m's exact element ops — and
// hands that f16 value to the consumer body instead of writing it to memory. Bitwise the K2m +
// consumer pair; the f16 reduce output is never written or re-read. K2m publishes no flag in prefill
// (the recv-slot rule is stream order, R >= 2G), so neither do these.
// =============================================================================
__device__ __forceinline__ int xtp_k2m_gate(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const unsigned long long* done = xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        while (xtp_ld(done) < e) {
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) xtp_fence_acq();
    }
    __syncthreads();
    return s_ab;
}

// MoE sites: K2m (fp32 local partial, fp32 wire, f16 out) + xq_pf_hc_inj_norm over the epoch's rows.
// grid = rows of the epoch, block 512 = 4 groups of 128 threads; group s is xq_pf_hc_inj_norm's
// (row, stream s) CTA verbatim (same thread -> element map, same fma chain, same trees) reading the
// row's f16 y from shared memory, where the whole CTA staged it once (K2m's element ops). Pointers are
// offset to the epoch's first row by the host: local [rows][h] f32, out/x [rows][hc][h], inj [rows][hc].
// hc == 4 (the served model; host-checked), h % 512 == 0, h <= 4096.
extern "C" __global__ void __launch_bounds__(512)
xq_pf_hc_inj_norm_k2(tp_dev_ctx* c, unsigned lag, const float* __restrict__ local, __half* __restrict__ out,
                     float* __restrict__ x, const float* __restrict__ inj, const __half* __restrict__ w,
                     int h, float eps) {
    XQ_PDL_ENTRY();
    constexpr int HC = 4;
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    if (xtp_k2m_gate(c, e)) return;
    const int b = blockIdx.x;
    const float* peer = (const float*)xtp_recv_slot(c, e) + (long long)b * h;
    const float* loc = local + (long long)b * h;
    const bool lower = (c->rank == 0);
    __shared__ __align__(16) __half ys[4096];
    for (int i4 = threadIdx.x; i4 < (h >> 2); i4 += 512) {
        const float4 l4 = ((const float4*)loc)[i4];
        const float4 p4 = ((const float4*)peer)[i4];
        const float4 a = lower ? l4 : p4;
        const float4 bb = lower ? p4 : l4;
        float4 r;
        r.x = a.x + bb.x; r.y = a.y + bb.y; r.z = a.z + bb.z; r.w = a.w + bb.w;
        const __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&h01);
        u.y = *reinterpret_cast<const unsigned*>(&h23);
        *reinterpret_cast<uint2*>(ys + 4 * i4) = u;
    }
    __syncthreads();
    // ---- xq_pf_hc_inj_norm's body for (row b, stream s = group)
    const int s = threadIdx.x >> 7;
    const int blk = b * HC + s;
    __shared__ float smg[HC][512];
    __shared__ float s_invg[HC];
    float* sm = smg[s];
    const int T = threadIdx.x & 127;
    float* xs = x + (long long)blk * h;
    const float g = inj[b * HC + s];
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
            float yf[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ys + i), yf);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) t[e2] += g * yf[e2];          // xq_hc_inject's fma
            *reinterpret_cast<float4*>(xs + i) = make_float4(t[0], t[1], t[2], t[3]);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                v[k][e2] = t[e2];
                if (k & 1) pb[e2] += t[e2] * t[e2]; else pa[e2] += t[e2] * t[e2];   // xq_hc_norm's sum_sq
            }
        }
    }
    #pragma unroll
    for (int e2 = 0; e2 < 4; ++e2) sm[4 * T + e2] = pa[e2] + pb[e2];              // level 512
    __syncthreads();
    if (T < 32) {
        const float r = xq_pf_tree512(sm, T);
        if (T == 0) s_invg[s] = rsqrtf(r / (float)h + eps);
    }
    __syncthreads();
    const float inv = s_invg[s];
    const __half* ws = w + (long long)s * h;
    __half* os = out + (long long)blk * h;
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float wf[4], o[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ws + i), wf);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) o[e2] = v[k][e2] * inv * (1.0f + wf[e2]);
            *reinterpret_cast<uint2*>(os + i) = xq_pf_pack4(o);
        }
    }
}

// Mixer sites: K2m (f16 local partial = the chain's y_raw, f16 wire, f16 out) + exl3_had_svh over the
// epoch's rows. A persistent grid (block 256 = 8 warps; grid <= 96): ONE gate per CTA (v1 gave every
// (row, 128-block) its own 32-thread CTA, i.e. ~4,000 CTAs per epoch each spinning on the system-scope
// cpu_done line and fencing — 839 us per epoch in the trace), then each warp runs exl3_had_svh's warp
// body on tasks (row m, block nb) t = cta*8 + warp, += grid*8: per lane 4 elements formed as K2m io 7
// (widen both f16 partials exactly, canonical fp32 add, __floats2half2_rn) — then exl3_had_svh_dev's
// arithmetic on those values. local / y are offset to the epoch's first row by the host.
extern "C" __global__ void __launch_bounds__(256)
exl3_had_svh_k2(tp_dev_ctx* c, unsigned lag, const __half* __restrict__ local,
                const __half* __restrict__ svh, __half* __restrict__ y, int n, int rows) {
    XQ_PDL_ENTRY();
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    if (xtp_k2m_gate(c, e)) return;
    const __half* peer = (const __half*)xtp_recv_slot(c, e);
    const bool lower = (c->rank == 0);
    const int nbk = n >> 7;
    const int tasks = rows * nbk;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int t = blockIdx.x * 8 + warp; t < tasks; t += gridDim.x * 8) {     // warp-uniform
        const int nb = t % nbk;
        const int m = t / nbk;
        const size_t base = (size_t)m * n + (size_t)nb * 128 + lane * 4;
        float v[4];
        {
            const uint2 ul = *reinterpret_cast<const uint2*>(local + base);
            const uint2 up = *reinterpret_cast<const uint2*>(peer + base);
            const float2 l01 = __half22float2(*reinterpret_cast<const __half2*>(&ul.x));
            const float2 l23 = __half22float2(*reinterpret_cast<const __half2*>(&ul.y));
            const float2 p01 = __half22float2(*reinterpret_cast<const __half2*>(&up.x));
            const float2 p23 = __half22float2(*reinterpret_cast<const __half2*>(&up.y));
            const float4 l4 = make_float4(l01.x, l01.y, l23.x, l23.y);
            const float4 p4 = make_float4(p01.x, p01.y, p23.x, p23.y);
            const float4 a = lower ? l4 : p4;
            const float4 bb = lower ? p4 : l4;
            float4 r;
            r.x = a.x + bb.x; r.y = a.y + bb.y; r.z = a.z + bb.z; r.w = a.w + bb.w;
            const __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
            v[0] = __half2float(__low2half(h01)); v[1] = __half2float(__high2half(h01));
            v[2] = __half2float(__low2half(h23)); v[3] = __half2float(__high2half(h23));
        }
        exl3_had128(v, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int j = lane * 4 + r;
            y[base + r] = __float2half_rn(v[r] * __half2float(svh[nb * 128 + j]));
        }
    }
}

// =============================================================================
// TP-H #4: vocab-parallel greedy lm_head tail (world 2). Each rank computes ONLY its 124160-column
// shard of the logits (lmh GEMM on the shard Quad, fused Hadamard: bitwise the replicated head's
// columns), reduces each row to ONE u64 key (xq_am_key: value desc, then lowest id — a TOTAL
// order), and exchanges the m keys through the doorbell transport; both ranks then take the
// max-merge of the two keys per row. Because the key order is total, max(shard keys with GLOBAL
// ids) == the replicated xq_argmax_rows result bit for bit (NaN loses in both; the degenerate
// all(-inf/NaN) row resolves to id 0 in both, since rank 0's sentinel outranks rank 1's).
//   xq_argmax_rows_vp : xq_argmax_rows' body over the shard (NO early return — every block must
//                       reach the fold), the row-last block writes keys[r] with the GLOBAL id, then
//                       xtp_k1_last (the TP-G v2 last-block K1) over the whole grid publishes keys.
//   xq_tp_wait_keys   : K2 — the shared GPU-side gate, then out_ids[r] = id(max(own, peer)).
// nbytes = 8 * roundup_even(m) (a multiple of 16: xtp_k1_last copies uint4). The padding keys are
// never written (zero on both ranks).
// =============================================================================
extern "C" __global__ void xq_argmax_rows_vp(tp_dev_ctx* c, unsigned long long* __restrict__ keys,
                                             const __half* __restrict__ logits, long long rstride,
                                             int V, int id_base,
                                             unsigned long long* __restrict__ part,
                                             unsigned int* __restrict__ cnt,
                                             unsigned* __restrict__ arrive, unsigned nbytes) {
    XQ_PDL_ENTRY();
    __shared__ unsigned long long wk[32];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const __half* row = logits + (long long)r * rstride;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;
    const int lo = min(b * chunk, V), hi = min(lo + chunk, V);
    float best = -INFINITY; int bi = 0;
    if ((reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull) {
        const int nv = (hi - lo) >> 3;
        const uint4* rv = reinterpret_cast<const uint4*>(row + lo);
        #pragma unroll 4
        for (int q = tid; q < nv; q += bs) {
            const uint4 u = __ldg(rv + q);
            const __half* hv = reinterpret_cast<const __half*>(&u);
            #pragma unroll
            for (int j = 0; j < 8; j++) xq_am_take(best, bi, xq_h2f(hv[j]), lo + q * 8 + j);
        }
        for (int i = lo + nv * 8 + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    } else {
        for (int i = lo + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    }
    unsigned long long key = xq_am_wmax(xq_am_key(best, bi));
    if ((tid & 31) == 0) wk[tid >> 5] = key;
    __syncthreads();
    if (tid < 32) {
        key = xq_am_wmax(tid < (bs >> 5) ? wk[tid] : 0ull);
        if (tid == 0) {
            part[(long long)r * nb + b] = key;
            __threadfence();
            last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
        }
    }
    __syncthreads();
    if (last && tid < 32) {
        __threadfence();
        const unsigned long long k2 = xq_am_wmax(tid < nb ? __ldcg(part + (long long)r * nb + tid) : 0ull);
        if (tid == 0) {
            const unsigned lid = 0xFFFFFFFFu - (unsigned)(k2 & 0xFFFFFFFFull);
            keys[r] = (k2 & 0xFFFFFFFF00000000ull) | (unsigned long long)(0xFFFFFFFFu - (lid + (unsigned)id_base));
            cnt[r] = 0u;                                      // re-arm for the next launch / replay
        }
    }
    xtp_k1_last(c, arrive, reinterpret_cast<const float*>(keys), nbytes);
}

// The K2 gate of xq_tp_wait_add_dec (same three release conditions, same fence / RX_DONE), as a
// single-block function: returns 1 = aborted (store nothing).
__device__ __forceinline__ int xtp_k2_gate1(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool nw = c->world > 2, g = (io & 8) != 0;
        const unsigned long long* done = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF) : xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF) : xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e && (!nw || xtp_ld(tail) == e)) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void xq_tp_wait_keys(tp_dev_ctx* c, unsigned long long* keys, int* out_ids,
                                           int m, int nbytes, int io) {
    XQ_PDL_ENTRY();                                     // the argmax+K1 wrote c->epoch: wait for it
    const unsigned long long e = c->epoch;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const unsigned long long* tail = (const unsigned long long*)(slot + (((unsigned)nbytes + 7u) & ~7u));
    if (xtp_k2_gate1(c, e, tail, io)) return;
    const unsigned long long* pk = (const unsigned long long*)slot;
    for (int r = threadIdx.x; r < m; r += blockDim.x) {
        const unsigned long long a = keys[r], b = pk[r];
        const unsigned long long k = a > b ? a : b;     // total order: rank-order independent
        keys[r] = k;
        out_ids[r] = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
    }
}

// =============================================================================
// TP-H2: vocab-parallel lm_head for SAMPLED / PENALIZED / RATIO-RULE rows (world 2; --tp-vp-sampled).
// The sampler needs the full logits row, so each rank's shard rows (the TP-H #4 shard GEMM: every column
// bitwise the replicated head's, ks = 1) are ALL-GATHERED into sc.logits [m][V] through the decode
// doorbell, and the UNCHANGED xq_pen_rows / xq_argmax_rows / xq_sample_rows(_rq) run on the reassembled
// rows: bitwise the replicated head's buffer by construction (the gather moves f16 bits unchanged).
// One epoch carries rows [r0, r0 + nr) (nr <= 4: 4 x 248,320 B fits one 1 MiB ring slot); the host
// launches the K1s of every epoch, then the K2s (lag = epochs - 1 - j; G <= 4 = R/2: the TP-F recv-slot
// rule). Payload layout in the slot: [nr][n_sh] f16, row-contiguous.
//   xq_vp_gather_k1 : K1m's shape (tp_gate_copy_signal_mb): every block waits the I3 reuse gate itself,
//                     copies its grid-stride share of the shard rows to BOTH the send slot and this rank's
//                     columns [col_own, col_own + n_sh) of the logits rows, fence.sc.sys, arrives; the LAST
//                     block publishes (c->epoch, tail epoch, len tag, gpu_ready release) — K1m's sequence.
//   xq_vp_gather_k2 : the receive gate of xq_tp_wait_add_dec per block (cpu_done, or with io bit 3 the
//                     commit hint + the payload's generation-tagged tail; RX_DONE from block 0), then the
//                     peer's rows -> its columns [col_peer, col_peer + n_sh). No arithmetic anywhere.
// All row / column offsets are 16 B aligned (n_sh % 8 == 0, V % 8 == 0: host-checked). World 2 only.
// =============================================================================
// The I3 reuse gate of K1m (every block gates itself; the diagnostic count on block 0 only).
__device__ __forceinline__ int xtp_k1_gate_mb(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = xtp_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather_k1(
        tp_dev_ctx* c, const __half* __restrict__ shard, __half* __restrict__ logits, long long ldl,
        int n_sh, int col_own, int r0, int nr, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    // every block reads the PRE-kernel counter (the last block stores it only after all arrived)
    const unsigned long long e = c->epoch + 1;
    const int ab = xtp_k1_gate_mb(c, e);
    unsigned char* slot = xtp_send_slot(c, e);
    const unsigned nbytes = (unsigned)nr * (unsigned)n_sh * 2u;
    if (!ab) {
        const int rq = n_sh >> 3;                                   // uint4 per shard row
        const int tot = nr * rq;
        const uint4* src = reinterpret_cast<const uint4*>(shard) + (long long)r0 * rq;
        uint4* dsl = reinterpret_cast<uint4*>(slot);
        const long long lq = ldl >> 3;                              // uint4 per logits row
        uint4* dlg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq + (col_own >> 3);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
            const int r = i / rq;
            const int j = i - r * rq;
            const uint4 v = __ldcg(src + i);
            dsl[i] = v;
            dlg[(long long)r * lq + j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's slot stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // re-armed for the next launch / replay
            c->epoch = e;                               // advances even on abort (I9 no-op), like K1m
            if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e; // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

// K2 gate, multi-block: xq_tp_wait_add_dec's three release conditions (cpu_done; with io bit 3 the commit
// hint + the payload tail; abort) per block, RX_DONE published by block 0 only. Returns 1 = aborted.
__device__ __forceinline__ int xtp_k2_gate_mb(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool g = (io & 8) != 0;
        const unsigned long long* done = xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather_k2(
        tp_dev_ctx* c, __half* __restrict__ logits, long long ldl, int n_sh, int col_peer,
        int r0, int nr, unsigned lag, int io) {
    XQ_PDL_ENTRY();                                     // the K1s wrote c->epoch: wait for them
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const unsigned nbytes = (unsigned)nr * (unsigned)n_sh * 2u;
    const unsigned long long* tail = (const unsigned long long*)(slot + ((nbytes + 7u) & ~7u));
    if (xtp_k2_gate_mb(c, e, tail, io)) return;
    const int rq = n_sh >> 3;
    const int tot = nr * rq;
    const uint4* src = reinterpret_cast<const uint4*>(slot);
    const long long lq = ldl >> 3;
    uint4* dlg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq + (col_peer >> 3);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
        const int r = i / rq;
        const int j = i - r * rq;
        dlg[(long long)r * lq + j] = src[i];
    }
}

// =============================================================================
// TP-4H2: WORLD-4 vocab-parallel row all-gather (--tp-vp-sampled at world 4). ADDITIVE: nothing above
// (xq_vp_gather_k1/k2, xtp_k1_gate_mb, xtp_k2_gate_mb — the world-2 path) is touched; the host takes
// these kernels at world 4 only (`vp_gather_rows_w4`, src/exl3_forward/xtp.rs). The schedule, launch
// tables and the index math below are mirrored in src/tp_vpgather.rs and pinned by its CPU tests.
//
// Recursive doubling over the decode doorbell, TWO epochs per group of nr <= 4 rows (partner(e) = rank ^
// (1 << (e % rounds)), rounds = 2; shard s = logits columns [s * n_sh, (s + 1) * n_sh)):
//   stage 0 (epoch e)   : payload = the OWN shard rows [nr][n_sh] f16 (from the shard GEMM output); K1 also
//                         copies them into this rank's columns of the logits rows. K2 writes the partner's
//                         shard (partner = rank ^ m0, m0 = 1 << (e % rounds)) into ITS columns.
//   stage 1 (epoch e+1) : payload = TWO shards, [blk 0 = shard rank][blk 1 = shard rank ^ m0] (m0 = the
//                         PREVIOUS epoch's mask: 1 << ((e + rounds - 1) % rounds)), each [nr][n_sh], read
//                         from the logits rows stage 0 assembled. K2 writes partner p's two shards
//                         (p, p ^ m0) — exactly the two this rank lacks. Not a contiguous column block
//                         when m0 = 2 (the odd-first-epoch phase): shards are indexed, never assumed adjacent.
// Payload unit i (uint4 = 8 f16) -> (blk, row, j): i = (blk * nr + row) * rq + j, rq = n_sh / 8. Payload
// bytes = nr * n_sh * 2 * (stage + 1) (<= 993,280 at nr = 4, stage 1), the 8 B generation tail at align8.
// The host launches per group K1(0) K2(0) K1(1) K2(1), every K2 with lag 0 (stage 1's K1 reads columns
// stage 0's K2 wrote; serial lookahead 1 = the proven recv-slot rule). No arithmetic anywhere: bit copies.
// All offsets are 16 B aligned (n_sh % 8 == 0, V % 8 == 0: host-checked). World 4 only (host-checked).
// =============================================================================
// The I3 reuse gate, multi-block, world > 2: every block waits partner(e - R)'s per-QP tx_retired (the
// same peer as partner(e): R is even). xtp_k1_gate's body with xtp_k1_gate_mb's per-block shape.
__device__ __forceinline__ int xtp_k1_gate_mb_nw(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = xtp_nway(c, xtp_partner(c, tgt), TP_NWAY_TX_OFF);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather4_k1(
        tp_dev_ctx* c, const __half* __restrict__ shard, __half* __restrict__ logits, long long ldl,
        int n_sh, int r0, int nr, int stage, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    // every block reads the PRE-kernel counter (the last block stores it only after all arrived)
    const unsigned long long e = c->epoch + 1;
    const int ab = xtp_k1_gate_mb_nw(c, e);
    unsigned char* slot = xtp_send_slot(c, e);
    const int rq = n_sh >> 3;                                       // uint4 per shard row
    const int per = nr * rq;                                        // uint4 per shard block
    const int tot = per * (stage + 1);
    const unsigned nbytes = (unsigned)tot * 16u;
    if (!ab) {
        const int rank = (int)c->rank;
        const long long lq = ldl >> 3;                              // uint4 per logits row
        uint4* dsl = reinterpret_cast<uint4*>(slot);
        uint4* lg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq;
        if (stage == 0) {
            const uint4* src = reinterpret_cast<const uint4*>(shard) + (long long)r0 * rq;
            uint4* own = lg + (long long)rank * rq;                 // this rank's columns
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
                const int r = i / rq;
                const int j = i - r * rq;
                const uint4 v = __ldcg(src + i);
                dsl[i] = v;
                own[(long long)r * lq + j] = v;
            }
        } else {
            // the two shards held after stage 0: own + the stage-0 partner's (mask of the previous epoch)
            const int m0 = 1 << (int)((e + (unsigned long long)c->rounds - 1ull) % (unsigned long long)c->rounds);
            const int s1 = rank ^ m0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
                const int blk = i / per;
                const int rem = i - blk * per;
                const int r = rem / rq;
                const int j = rem - r * rq;
                const int s = blk ? s1 : rank;
                dsl[i] = __ldcg(lg + (long long)r * lq + (long long)s * rq + j);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's slot stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // re-armed for the next launch / replay
            c->epoch = e;                               // advances even on abort (I9 no-op), like K1m
            if (c->qp_mask) c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << xtp_partner(c, e);
            if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e; // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

// K2 gate, multi-block, world > 2: xtp_k2_gate1's conditions (partner(e)'s per-peer cpu_done AND the R9
// tail equality; with io bit 3 the commit hint + the tail; abort) per block, RX_DONE by block 0 only.
__device__ __forceinline__ int xtp_k2_gate_mb_nw(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool g = (io & 8) != 0;
        const unsigned long long* done = xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF);
        const unsigned long long* hint = xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e && xtp_ld(tail) == e) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather4_k2(
        tp_dev_ctx* c, __half* __restrict__ logits, long long ldl, int n_sh,
        int r0, int nr, int stage, unsigned lag, int io) {
    XQ_PDL_ENTRY();                                     // the K1s wrote c->epoch: wait for them
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const int rq = n_sh >> 3;
    const int per = nr * rq;
    const int tot = per * (stage + 1);
    const unsigned nbytes = (unsigned)tot * 16u;
    const unsigned long long* tail = (const unsigned long long*)(slot + ((nbytes + 7u) & ~7u));
    if (xtp_k2_gate_mb_nw(c, e, tail, io)) return;
    // the partner's payload: stage 0 = its own shard; stage 1 = {partner, partner ^ (the previous epoch's mask)}
    const int p0 = (int)c->rank ^ (1 << (int)(e % (unsigned long long)c->rounds));
    const int p1 = p0 ^ (1 << (int)((e + (unsigned long long)c->rounds - 1ull) % (unsigned long long)c->rounds));
    const uint4* src = reinterpret_cast<const uint4*>(slot);
    const long long lq = ldl >> 3;
    uint4* lg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
        const int blk = i / per;
        const int rem = i - blk * per;
        const int r = rem / rq;
        const int j = rem - r * rq;
        const int s = blk ? p1 : p0;
        lg[(long long)r * lq + (long long)s * rq + j] = src[i];
    }
}

// =============================================================================
// TP-H #3: sharded DHEAD screen (world 2). The draft head's 2-bit screen is 42 MB/pass — the biggest
// byte-bound item of a draft pass. Each rank streams ONLY its half of the 512 128-column blocks
// (grid = nblk / world, nb_off = rank * grid; the codes buffer stays whole, indexed by the GLOBAL
// block), writes the approximate block maxima of its blocks into the ZERO-PADDED exchange buffer
// `part` [nblk] f32 (the peer's half stays zero forever: only the owned half is ever written), and
// the last block publishes the whole buffer (xtp_k1_last, nbytes = 4 * nblk, a multiple of 16).
// K2 is the existing xq_tp_wait_add_dec on fp32 (out = ws bmax): lower + upper of a value and an
// exact 0.0 is the value, so every rank ends up with the replicated screen's bmax array BIT FOR
// BIT (a -0 max canonicalizes in dh_ord), and the replicated candidate selection + exact rescore
// then produce the replicated head's draft, dconf, d_dev and toks unchanged.
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 12) xq_dh_screen_k1(
        tp_dev_ctx* c, const uint4* __restrict__ codes, unsigned* __restrict__ ws,
        const __half* __restrict__ svh, float* __restrict__ part,
        int K, int B, float dq, int nb_off, unsigned* __restrict__ arrive, unsigned nbytes) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned xs[1024];
    __shared__ float s_y[128];
    switch (B) {
        case 1: dh_screen_body<1>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        case 2: dh_screen_body<2>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        case 4: dh_screen_body<4>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        default: break;
    }
    xtp_k1_last(c, arrive, part, nbytes);
}
