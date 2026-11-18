// gpu_fp4l/02_precision_levers_router_hc.cuh: Precision levers (stored bf16/fp32 weights), B32 GEMMs, MTP sidecar, router top-k and the B32 hyper-connection mixer.
// The _bf/_f twins are line-for-line copies of the trunk kernels with only the weight load changed; keep them in step.
// Section 2 of 3 of gpu_fp4l.cu: included in order by gpu_fp4l.cu, one translation unit; text is moved verbatim.
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

