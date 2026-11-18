// exl3_bench/02_wide_grouped_forward.cuh: wide/grouped HMMA GEMM entries, whole-model offline forward glue (S-A3-d), conv1d.
// Grouped entries run the UNCHANGED wide body (same work per (expert, n-tile, super-row) => bitwise).
// Section 2 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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
    // K35: the fractional widths (EXL3_FRAC_BITS code) on the HALF twin — the device-routed prefill arm's body.
#define EXL3_WT_CASE_H(KA) case EXL3_FRAC_BITS(KA): switch (mt) { \
        case 1: exl3_gemm_body_wide<KA, 1, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 2: exl3_gemm_body_wide<KA, 2, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 4: exl3_gemm_body_wide<KA, 4, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        case 8: exl3_gemm_body_wide<KA, 8, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_wide_sa); break; \
        default: break; } break;
    switch (bits) { EXL3_WT_CASE(3) EXL3_WT_CASE(4) EXL3_WT_CASE(5) EXL3_WT_CASE(6) EXL3_WT_CASE(7) EXL3_WT_CASE(8) EXL3_WT_CASE_H(1) EXL3_WT_CASE_H(2) EXL3_WT_CASE_H(3) default: break; } // PR#27-D2: 8 joins; K35: 1.5/2.5/3.5
#undef EXL3_WT_CASE_H
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
    switch (bits) { EXL3_WG_CASE(3) EXL3_WG_CASE(4) EXL3_WG_CASE(5) EXL3_WG_CASE(6) EXL3_WG_CASE(7) EXL3_WG_CASE(8) default: break; } // PR#27-D2: 8 joins
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
        EXL3_WIDE_CASE(6);
        EXL3_WIDE_CASE(7);
        EXL3_WIDE_CASE(8); // PR#27-D2: K=8 dense/wide serve
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
            case 2: exl3_gemm_body_a1<2>(tr, xh, ye, M, K, N, tile); break;
            case 3: exl3_gemm_body_a1<3>(tr, xh, ye, M, K, N, tile); break;
            case 4: exl3_gemm_body_a1<4>(tr, xh, ye, M, K, N, tile); break;
            case 5: exl3_gemm_body_a1<5>(tr, xh, ye, M, K, N, tile); break;
            case 6: exl3_gemm_body_a1<6>(tr, xh, ye, M, K, N, tile); break;
            case 7: exl3_gemm_body_a1<7>(tr, xh, ye, M, K, N, tile); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 3: exl3_gemm_body<3>(tr, xh, ye, M, K, N, tile); break;
        case 4: exl3_gemm_body<4>(tr, xh, ye, M, K, N, tile); break;
        case 5: exl3_gemm_body<5>(tr, xh, ye, M, K, N, tile); break;
        case 6: exl3_gemm_body<6>(tr, xh, ye, M, K, N, tile); break;
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

