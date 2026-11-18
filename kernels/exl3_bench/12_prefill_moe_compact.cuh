// exl3_bench/12_prefill_moe_compact.cuh: compact per-expert MoE prefill (A5-P1): grouped rows, had_suh/svh rows, prof take.
// Deterministic: ascending expert id, ascending token within expert.
// Section 12 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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

// PR #32 (andersc), generalized to every served width (BITS 2..7): the 8 trellis windows a lane decodes
// sit at 8 consecutive trellis positions, so their 16 (lo, hi) ring words are N consecutive ring words
// starting at window 0's hi word: N = 2 for BITS <= 4, 3 for BITS 5..7 (7 * BITS + 16 bits from a start
// bit fixed by the lane; pinned for every lane and width by the host test xq_pf_ring_sources). Each of
// the N words is fetched with one __shfl_sync (two for BITS > 4: the word may sit in the second ring
// load) and every window selects its lo / hi among them: 2 / 6 shuffles per 16x16 block instead of
// 16 / 32. The words are the same bits, so the decode and every output are bitwise unchanged.
template <int BITS> struct XqPfSel {
    static constexpr int N = (BITS <= 4) ? 2 : 3;
    int src[N];             // ring word index (0 .. 8*BITS-1) of source k = (window 0's hi word + k) mod ring
    unsigned mlo, mhi;      // 2 bits per window jj: the source holding its lo (i1) / hi (i0) word
};
template <int BITS>
__device__ __forceinline__ XqPfSel<BITS> xq_pf_sel(const int (&s_i0m)[8], const int (&s_i1m)[8]) {
    constexpr int RING = 8 * BITS;
    XqPfSel<BITS> q;
    const int base = s_i0m[0];
    #pragma unroll
    for (int k = 0; k < XqPfSel<BITS>::N; ++k) q.src[k] = (base + k) % RING;
    q.mlo = 0u; q.mhi = 0u;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        q.mlo |= (unsigned)((s_i1m[j] - base + RING) % RING) << (2 * j);
        q.mhi |= (unsigned)((s_i0m[j] - base + RING) % RING) << (2 * j);
    }
    return q;
}
// The N source words of one 16x16 block from the lane's ring words (w1 = ring words 32.., BITS > 4).
template <int BITS>
__device__ __forceinline__ void xq_pf_fetch(const XqPfSel<BITS>& q, const unsigned w0, const unsigned w1,
                                            unsigned (&v)[XqPfSel<BITS>::N]) {
    #pragma unroll
    for (int k = 0; k < XqPfSel<BITS>::N; ++k) {
        unsigned a = __shfl_sync(0xFFFFFFFFu, w0, q.src[k] & 31);
        if constexpr (8 * BITS > 32) {
            const unsigned b = __shfl_sync(0xFFFFFFFFu, w1, q.src[k] & 31);
            a = q.src[k] >= 32 ? b : a;
        }
        v[k] = a;
    }
}
// The pre-PR #32 per-window pair of shuffles (two more for BITS > 4: the word may sit in the second ring load).
template <int BITS>
__device__ __forceinline__ void xq_pf_win(const unsigned w0, const unsigned w1, const int i0, const int i1,
                                          unsigned& lo, unsigned& hi) {
    lo = __shfl_sync(0xFFFFFFFFu, w0, i1 & 31);
    hi = __shfl_sync(0xFFFFFFFFu, w0, i0 & 31);
    if constexpr (8 * BITS > 32) {
        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, i1 & 31);
        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, i0 & 31);
        lo = i1 >= 32 ? lo2 : lo;
        hi = i0 >= 32 ? hi2 : hi;
    }
}
// Where the source-word selection pays (measured 2026-11-02, .14, --probe-exl3-bitwidths class rows, best of 3
// alternating runs): BITS 2..4 and 6..7 at every MT, BITS 5 at MT >= 4 (-5..-13 %). BITS 5 at MT 1 / 2 (decode-bound
// tiles) measured +4..+18 % with it, so those keep the per-window shuffles. Either way the same words: bitwise.
template <int BITS, int MT> struct XqPfUseSel { static constexpr bool v = !(BITS == 5 && MT <= 2); };
template <int N>
__device__ __forceinline__ unsigned xq_pf_pick(const unsigned (&v)[N], const unsigned m, const int jj) {
    const unsigned c = (m >> (2 * jj)) & 3u;
    if constexpr (N == 2) return c ? v[1] : v[0];
    else return c == 0u ? v[0] : (c == 1u ? v[1] : v[2]);
}
// PR #32 (andersc): one 16x16 A fragment (a0..a3 of m16n8k16, row-major f16 at `base`, row stride LD halves,
// k16 column offset cs) with one ldmatrix.x4 instead of four 32-bit shared loads. Matrix m of the x4 =
// rows (m & 1) * 8.., k (m >> 1) * 8..: the same four registers the scalar loads produced (bitwise).
// LD = 40 / 72 / 136 halves: 16-byte rows, and 8 consecutive rows hit 8 disjoint 4-bank groups.
template <int LD>
__device__ __forceinline__ void xq_pf_ldsm_a(uint32_t (&a)[4], const __half* __restrict__ base, const int cs, const int lane) {
    static_assert(LD % 8 == 0, "ldmatrix row addresses must be 16-byte aligned");
    const int m = lane >> 3, r = lane & 7;
    const __half* p = base + ((m & 1) * 8 + r) * LD + cs + (m >> 1) * 8;
    const uint32_t ad = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(ad));
}

// One k16 step of one warp: decode its NH 16x16 trellis blocks from the lane's ring words w0[]
// (== exl3_gemm_body_wide BITS <= 4) and run the MT m16n8k16 chains on the staged A columns
// [cs, cs + 16) of `cur` (row stride LD); NH = 2 (A5-L2 v2, paired gate|up warps) shares one set
// of A fragments between the two blocks. Per output element: the served decode + the same
// K-ascending mma chain (NH = 1 is the served body's inline step).
template <int MT, int LD, int NH, int BITS = 4>
__device__ __forceinline__ void xq_pf_step(const unsigned (&w0)[NH], const unsigned (&w1)[NH], const __half* __restrict__ cur, const int cs,
        const int lane, const XqPfSel<BITS>& sel, const int (&s_i0m)[8], const int (&s_i1m)[8], const int (&s_sf)[8],
        float (&acc)[NH][MT][2][4]) {
    constexpr bool SEL = XqPfUseSel<BITS, MT>::v;
    uint32_t bfrag[NH][2][2];
    #pragma unroll
    for (int hh = 0; hh < NH; ++hh) {
        unsigned v[XqPfSel<BITS>::N];
        if constexpr (SEL) xq_pf_fetch<BITS>(sel, w0[hh], w1[hh], v);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint16_t rv[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int jj = h * 4 + q;
                unsigned lo, hi;
                if constexpr (SEL) { lo = xq_pf_pick(v, sel.mlo, jj); hi = xq_pf_pick(v, sel.mhi, jj); }
                else xq_pf_win<BITS>(w0[hh], w1[hh], s_i0m[jj], s_i1m[jj], lo, hi);
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
        uint32_t af[4];
        xq_pf_ldsm_a<LD>(af, cur + (mt * 16) * LD, cs, lane);
        const uint32_t a0 = af[0], a1 = af[1], a2 = af[2], a3 = af[3];
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

// PFHALF (2026-11-06): HALF = the fractional width K = BITS + 0.5 (BITS = KA 1..3; the exl3_gemm_body_wide HALF twin's
// window math exl3_half_b1<KA> and ring WRING = 4*(2*KA+1) = 12 / 20 / 28 words per 16x16 tile, one ring word per
// lane). The HALF path decodes with the per-window shuffles (the PR #32 source-word selection is built on the
// integer 8*BITS ring); HALF = false is the integer body unchanged.
template <int BITS, int MT, int NW, int EPI, int PROF, bool HALF = false>
__device__ __forceinline__ void xq_moe_pf_body(
        const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
        __half* __restrict__ y, int M, int K, int N, int cta_tile, int m_sup,
        __half* __restrict__ smem, const __half* __restrict__ svh_e,
        const __half* __restrict__ suh_de) {
    static_assert(BITS <= 7, "P7 B4: K <= 7 loads a second ring word (lane + 32; index < 8*BITS <= 56 < 64), the B3 SHFL pattern");
    static_assert(!HALF || (BITS >= 1 && BITS <= 3), "PFHALF: fractional K = 1.5 / 2.5 / 3.5 only");
    constexpr int KS = XqPfKs<MT>::v;
    constexpr int ROWS = MT * 16;
    constexpr int LD = KS * 16 + 8;
    constexpr int C8 = KS * 2;                 // 16-B chunks per staged row
    constexpr int NCH = ROWS * C8;             // 16-B chunks per stage
    constexpr int NT = NW * 32;
    constexpr int WW = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);   // ring words per 16x16 tile
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
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WW;
            s_i1m[h * 4 + s] = i1 % WW;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const XqPfSel<BITS> sel = xq_pf_sel<BITS>(s_i0m, s_i1m);   // PR #32: the block's source ring words (unused when HALF)

    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;

    // lane's ring word of k16 block kb: word (kb * nb + nb_col) * WW + lane (lane < WW)
    const uint32_t* ringl = (const uint32_t*)trellis + (size_t)nb_col * WW + lane;
    const size_t kstr = (size_t)nb * WW;
    const int NS = (KB + KS - 1) / KS;
    __half* buf0 = smem;
    __half* buf1 = smem + ROWS * LD;

    uint32_t wn[KS]; uint32_t wn2[KS];   // P7 B4: ring words 32..WW-1 (K = 5)
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
            wn2[s_] = (WW > 32 && lane + 32 < WW && kb_ < KB) ? __ldg(ringl + (size_t)kb_ * kstr + 32) : 0u; \
        } \
    } while (0)

    XQ_PF_STAGE(0, buf0);
    XQ_PF_LOADW(0);
    const int gid = lane >> 2, tig = lane & 3;
    for (int j = 0; j < NS; ++j) {
        uint32_t wc[KS]; uint32_t wc2[KS];
        #pragma unroll
        for (int s = 0; s < KS; ++s) { wc[s] = wn[s]; wc2[s] = wn2[s]; }
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
                const unsigned w0 = wc[s]; const unsigned w1 = wc2[s];
                uint32_t bfrag[2][2];
                constexpr bool SEL = !HALF && XqPfUseSel<BITS, MT>::v;   // HALF: per-window shuffles (WW <= 28: w0 only)
                unsigned v[XqPfSel<BITS>::N];
                if constexpr (SEL) xq_pf_fetch<BITS>(sel, w0, w1, v);   // P7 B4 (K >= 5): w1 = the second ring load
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint16_t rv[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int jj = h * 4 + q;
                        unsigned lo, hi;
                        if constexpr (SEL) { lo = xq_pf_pick(v, sel.mlo, jj); hi = xq_pf_pick(v, sel.mhi, jj); }
                        else xq_pf_win<BITS>(w0, w1, s_i0m[jj], s_i1m[jj], lo, hi);
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
                    uint32_t af[4];
                    xq_pf_ldsm_a<LD>(af, cur + (mt * 16) * LD, cs, lane);
                    const uint32_t a0 = af[0], a1 = af[1], a2 = af[2], a3 = af[3];
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

