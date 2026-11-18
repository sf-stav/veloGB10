// exl3_bench/01_tune_lmh_gemm.cuh: autotune measurement-harness kernels, EXL3 trellis decode + HMMA GEMM bodies, persistent-grid lm_head (W3/LMH).
// Harness kernels launch only from --autotune, never on a serving path.
// Section 1 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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
        (const uint32_t*)(trellis + (size_t)b * (8 * bits));   // bits arg = k2
    const int kb = b / nb, nbk = b % nb;
    uint16_t* __restrict__ orow = wout + (size_t)kb * 16 * (nb * 16) + nbk * 16;

    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        const uint16_t v = exl3_dq_k2(ring, t, bits);
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
    const int ww = 4 * bits;   // bits arg = k2
    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)kb * nb + n0 + warp) * (8 * bits));
    for (int i = lane; i < ww; i += 32) s_ring[warp][i] = __ldg(ring + i);
    __syncthreads();
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        const uint16_t v = exl3_dq_k2(s_ring[warp], t, bits);
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
// B3 leg 2: file-scope so every exl3_gemm_body<BITS> instantiation inside one kernel
// aliases a single 512 B block buffer (switch cases never run simultaneously). In-function
// __shared__ gave one 512 B per instantiation: 6 widths = 3072 B static, which cut the
// no-opt-in dynamic cap to 46080 B and rejected the m=9 gu launch (46224 B) outright
// (cuLaunchKernel CUDA_ERROR_INVALID_VALUE; EPI-ATTR receipt 2026-10-29).
__shared__ __half exl3_gemm_body_sa[16][16];
template <int BITS, bool HALF = false>
__device__ __forceinline__ void exl3_gemm_body(const uint16_t* __restrict__ trellis,
                                               const __half* __restrict__ xh,
                                               __half* __restrict__ yraw,
                                               int M, int K, int N, int cta_tile) {
    // CTA tile: m16 x N128; 256 threads = 8 warps.
    __half (&sa)[16][16] = exl3_gemm_body_sa;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;      // this warp's trellis column
    const int KB = K >> 4;                     // k16 steps (one trellis block each)
    const int nb = N >> 4;
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);   // ring words per 16x16 tile
    constexpr int W16 = HALF ? (8 * (2 * BITS + 1)) : (16 * BITS);    // uint16 words per tile

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
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
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
        (const uint32_t*)(trellis + ((size_t)0 * nb + nb_col) * W16);
    const size_t ring_kstride = (size_t)nb * (HALF ? (4u * (2 * BITS + 1)) : (8u * BITS));  // uint32 words per k16 step

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
// K35: HALF = the fractional width K = BITS + 0.5 (window math exl3_half_b1<BITS>, ring WRING = 4*(2*BITS+1)
// words per tile = 12 / 20 / 28 for KA 1 / 2 / 3, all <= 32 = one SHFL source word). HALF = false is the
// integer body, instruction for instruction (WRING = 8*BITS, b1 = (t+257)*BITS): the a1bX entries are unchanged.
template <int BITS, bool HALF = false>
__device__ __forceinline__ void exl3_gemm_body_a1(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ xh,
                                                  __half* __restrict__ yraw,
                                                  int M, int K, int N, int cta_tile) {
    extern __shared__ __half a1_sa[];        // [M][K+8]
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);   // ring words per 16x16 tile
    constexpr int W16 = 2 * WRING;                                     // uint16 words per tile
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
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * W16);
    const size_t ring_kstride = (size_t)nb * (size_t)WRING;
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
    const bool pf_lane = lane * 32 < WRING * 4;          // 3/4/5 lanes x 32 B (window = WRING*4 bytes)
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
        constexpr int WW = WRING;
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
        case 3: exl3_gemm_body<1, true>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;  // K=1.5
        case 4: exl3_gemm_body<2>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;  // K=2.0
        case 5: exl3_gemm_body<2, true>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;  // K=2.5 (K35: probe reference)
        case 7: exl3_gemm_body<3, true>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;  // K=3.5 (K35: probe reference)
        case 6: exl3_gemm_body<3>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 8: exl3_gemm_body<4>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 10: exl3_gemm_body<5>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 12: exl3_gemm_body<6>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break;
        case 14: exl3_gemm_body<7>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break; // P7 B5 (probe reference; K=7)
        case 16: exl3_gemm_body<8>(trellis, xh, yraw, M, K, N, (int)blockIdx.x); break; // PR#27-D2: K=8 dense serve (compatibility cell)
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
           + mrows * LMH_EP_LD * 2;                         // fused-Hadamard epilogue tile
    // W9b: the m8 A-once block is GONE — at G==T each CTA walks A once, so __ldg from global
    // (the m16 path) is strictly cheaper than 8*(k+8)*2 bytes of staging that pushed b6_m8_s6
    // over the 101376 B opt-in cap and forced the WP20 fallback on every verify round.
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
    __half* ep = (__half*)(lmh_sm + NST * STB);
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

    // ---- prologue: trellis slots 0..NST-2 (W9b: A-once staging removed) ----
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
            {   // W9b: A from global for BOTH classes (live1 covers the m16 second half)
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
// C4STEP1: 6-bit lm_head (head_bits=6 packs, e.g. EXL3 4.05bpw Flash-Next) previously had no
// persistent entry and kept exl3_hmma_gemm; same body template, ring depth 6 like 3/4-bit.
LMH_ENTRY(6, 8, 6)
LMH_ENTRY(6, 16, 6)

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
// K35: HALF = the fractional width K = BITS + 0.5 (exl3_half_b1<BITS> windows, WRING = 4*(2*BITS+1) ring words
// per tile, <= 28 so the one-word SHFL diet applies). HALF = false is the integer body instruction for instruction.
template <int BITS, int MT, bool HALF = false>
__device__ __forceinline__ void exl3_gemm_body_wide(
        const uint16_t* __restrict__ trellis,
        const __half* __restrict__ xh,
        __half* __restrict__ yraw,
        int M, int K, int N, int cta_tile, int m_sup, __half* __restrict__ sa_raw) {
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);   // ring words per 16x16 tile
    constexpr int W16 = 2 * WRING;                                     // uint16 words per tile
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
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
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
            (const uint32_t*)(trellis + ((size_t)kb * nb + nb_col) * W16);
        uint32_t bfrag[2][2];
        if constexpr (WRING <= 32) {
            // S-A3-u C: the S-A3-n F7 LSU diet in the wide body — ONE coalesced load per lane of
            // the block's window words (WRING <= 32: integer BITS <= 4 and every HALF width), decode
            // words by __shfl_sync (was 16 scalar L1 loads per lane per k16 step). Same words, same
            // funnel/dp4a/hfma => same bits.
            constexpr int WW = WRING;
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

