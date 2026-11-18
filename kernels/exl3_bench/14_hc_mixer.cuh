// exl3_bench/14_hc_mixer.cuh: WP11 hc mixer rungs, HC regrid onto >= 2 CTAs/SM, router ks/combine.
// BITWISE twins of xq_hc_fuse_i8 / _rb: do not change reduction orders.
// Section 14 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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


// S-B9-CF-R16: the row-batched HC mixer (xq_hc_fuse_i8_rbk) at m = 9..16 rows.
// xq_hc_fuse_i8_rbk_r16 == xq_hc_fuse_i8_rbk with (a) M = 9..16 template cases
// and (b) the smem carve-out made DYNAMIC (R4 staging [2][M][1024] halves =
// m*4096 B, phase B [M][lr] f32; the host opts in past the 48 KB default — a
// STATIC > 48 KB array compiles but is never resident on this arch).
// BIT-IDENTITY to xq_hc_fuse_i8_rbk and to xq_hc_fuse_i8k at the same width:
// every per-row chain is the template body verbatim (down: per-k8 fresh-0 dot8
// added to acc, steps ascending, XOR butterfly 16..1, f16(acc * s_down); phase B:
// k-major 16-B word per k16 step, two dot8 locals added in order,
// f16(acc * s_up)), and the CTA count only grid-strides the outputs.
// m <= 8 dispatches to the untouched xq_hc_fuse_i8_rbk; this entry is only ever
// launched for m > 8 (host gate; --exl3-r16=0 restores today's dispatch).
// Dynamic smem (m*4096 B with R4, m*lr*4 B without): the host opts the function
// in once and takes the grid from its measured co-residency AT that carve-out.
extern "C" __global__ void __launch_bounds__(256, 1)
xq_hc_fuse_i8_rbk_r16(__half* __restrict__ dd, __half* __restrict__ uu,
                      const __half* __restrict__ hn, const int8_t* __restrict__ q_down,
                      const float* __restrict__ s_down, const int8_t* __restrict__ q_upk,
                      const float* __restrict__ s_up, cuda::atomic<int, cuda::thread_scope_device>* bar,
                      float hcn, int lr_m, int rw, int flags) {
    const int lr = lr_m & 0xFFFF, m = lr_m >> 16;
    xq_hc_pf_upk(q_upk, lr, rw, flags & 3);   // static weights: pre-PDL-wait
    XQ_PDL_ENTRY();
    // R4: [2][M][1024] halves (<= 64 KB at M = 16); phase B: [M][lr] floats (<= 32 KB).
    extern __shared__ __align__(16) unsigned char smem[];
    const bool r4 = (flags & 16) != 0;
#define XQ_RBK16_CASE(MM) case MM: \
        if (r4) xq_hc_fuse_i8_rbk_body<MM, true>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, smem); \
        else    xq_hc_fuse_i8_rbk_body<MM, false>(dd, uu, hn, q_down, s_down, q_upk, s_up, bar, hcn, lr, rw, smem); \
        break;
    switch (m) {
        XQ_RBK16_CASE(9) XQ_RBK16_CASE(10) XQ_RBK16_CASE(11) XQ_RBK16_CASE(12)
        XQ_RBK16_CASE(13) XQ_RBK16_CASE(14) XQ_RBK16_CASE(15) XQ_RBK16_CASE(16)
        default: break;
    }
#undef XQ_RBK16_CASE
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

