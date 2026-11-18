// exl3_bench/21_small_levers.cuh: W4/SMALL five latency-bound small-kernel levers.
// Every kernel is a BITWISE twin of the kernel it replaces: same per-output operation sequence and rounding points.
// Section 21 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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
// ascending from 0, through the same V readers (xq_dv3_ldv / xq_dv3_ldv_q8 / xq_dv3_ldv_q4k) on the
// same staged raw bytes; the closing expressions (f32/f16/fp8: f16((acc/denom)*sig(g)); q8 and q4k:
// the in-warp H32 un-rotation) are xq_dv3_acc_body's verbatim.
#define XQ_W4A_T   32
#define XQ_W4A_NST 5
template <int SD, int N16, bool SCH, int RS>
__device__ __forceinline__ void xq_w4a_stage_v(char* vbt, float* vsct, const XqKv& kc, int t0,
                                               int npos, int d0b, int scoff, int tid) {
    constexpr int NPER = N16 + (SCH ? 1 : 0);
    for (int i = tid; i < XQ_W4A_T * NPER; i += blockDim.x) {
        const int tr = i / NPER;
        const int c = i - tr * NPER;
        const int t = t0 + tr;
        if (t < npos) {
            const char* src = kc.v + (long long)t * kc.rb;
            if (c < N16) moe_coop_cp16(vbt + tr * RS + c * 16, src + d0b + c * 16, 16);
            else         moe_coop_cp16(vsct + tr * 4, src + scoff, 16);
        }
    }
}
template <int G, int FMT, int DPT>
__device__ __forceinline__ void xq_w4a_body(__half* __restrict__ attn, const float* __restrict__ sc,
                                            const float* __restrict__ pmx, const XqKv& kc,
                                            const __half* __restrict__ qg, int nh, int hd,
                                            int kvh, int row, int npos, char* vb, float* vsc, float* wb) {
    constexpr int T = XQ_W4A_T, NST = XQ_W4A_NST, SD = 16 * DPT, WST = XQ_W4A_T + 4;
    constexpr bool P4 = FMT == XQ_KV_Q4K;
    constexpr int EB = P4 ? 1 : (FMT == XQ_KV_F32 ? 4 : (FMT == XQ_KV_F16 ? 2 : 1));
    // q4k: SD/2 code bytes per smem row (2 dims per byte); q8: SD; f32/f16/fp8: SD*4.
    constexpr int N16 = P4 ? SD / 32 : SD * EB / 16;
    // The closing un-rotation runs the q8 branch for q4k too (same H32, same rotated basis).
    constexpr bool Q8 = FMT >= XQ_KV_Q8;
    constexpr int RS = P4 ? SD / 2 : (Q8 ? SD : SD * 4);
    constexpr bool SCH = FMT == XQ_KV_FP8 || Q8;
    static_assert(!Q8 || DPT >= 2, "q8/q4k: a thread's 16-dgl slice must hold whole 32-groups");
    static_assert(T == 32, "the weight tile below maps 16 threads x 2 positions");
    const int tid = threadIdx.x;
    const int h = tid >> 4;
    const int dgl = tid & 15;
    const int d0 = blockIdx.x * SD + dgl * DPT;
    const long long hrow = (long long)row * nh + (long long)kvh * G + h;
    const float* srow = sc + hrow * XQ_DV3_TS;
    const float scale = xq_dv3_scale(hd);
    const int ntile = (npos + T - 1) / T;
    const int d0b = blockIdx.x * (P4 ? SD / 2 : SD * EB);
    const int scoff = P4 ? hd / 2 : hd;
    // prologue: tiles 0..NST-2, one cp.async group each (empty groups keep the count aligned)
    #pragma unroll
    for (int p = 0; p < NST - 1; p++) {
        if (p < ntile) xq_w4a_stage_v<SD, N16, SCH, RS>(vb + p * (T * RS), vsc + p * (T * 4), kc, p * T, npos, d0b, scoff, tid);
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
                                             kc, pf * T, npos, d0b, scoff, tid);
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
                if constexpr (P4)      xq_dv3_ldv_q4k<DPT, RS>(v, vcur, scur, t + u, dgl, d0);
                else if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t + u, dgl, d0 >> 5);
                else                   xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t + u, dgl);
                denom = __fadd_rn(denom, ws[u]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(ws[u], v[e], acc[e]);
            }
        }
        for (; t < tn; t++) {
            const float w = wh[t];
            float v[DPT];
            if constexpr (P4)      xq_dv3_ldv_q4k<DPT, RS>(v, vcur, scur, t, dgl, d0);
            else if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t, dgl, d0 >> 5);
            else                   xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t, dgl);
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
        case XQ_KV_Q4K:
            // q4k: xq_dv3_acc_entry's rule (the single-row acc1_w4 runs the DPT=2 body on its
            // first hd/32 CTAs; RS = SD/2 keeps it inside acc1_w4's vb)
            if constexpr (DPT >= 2) {
                xq_w4a_body<G, XQ_KV_Q4K, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            } else {
                if ((int)blockIdx.x < hd / 32)   // block-uniform
                    xq_w4a_body<G, XQ_KV_Q4K, 2>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            }
            break;
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

