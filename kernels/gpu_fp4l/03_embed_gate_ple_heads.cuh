// gpu_fp4l/03_embed_gate_ple_heads.cuh: Embedding and shared-expert gate, PLE twins with fp32 weights, bf16 head views for the DFlash2 drafter.
// Drafter-only head views are draft-side; the trunk kernels they twin are unchanged.
// Section 3 of 3 of gpu_fp4l.cu: included in order by gpu_fp4l.cu, one translation unit; text is moved verbatim.
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
