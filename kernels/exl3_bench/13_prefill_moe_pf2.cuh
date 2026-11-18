// exl3_bench/13_prefill_moe_pf2.cuh: A5-L2 xq_moe_pf2 (trellis in an smem ring via cp.async), rows GEMM and combine.
// pf2 is bitwise == pf_body for every row, MT and KS.
// Section 13 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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
    static_assert(BITS <= 7, "P7 B4: K <= 7 loads a second ring word (lane + 32; index < 8*BITS <= 56 < 64), the B3 SHFL pattern");
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
    const XqPfSel<BITS> sel = xq_pf_sel<BITS>(s_i0m, s_i1m);   // PR #32: the block's source ring words

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
                unsigned w0[NH], w1[NH];
                #pragma unroll
                for (int hh = 0; hh < NH; ++hh) {
                    w0[hh] = lane < WW ? tw[s * (NSEG * 8 * WW) + hh * (8 * WW)] : 0u;
                    w1[hh] = 8 * BITS > 32 && lane + 32 < WW ? tw[s * (NSEG * 8 * WW) + hh * (8 * WW) + 32] : 0u;
                }
                xq_pf_step<MT, LD, NH, BITS>(w0, w1, cur, s * 16, lane, sel, s_i0m, s_i1m, s_sf, acc);
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
        else if (bits == 5) xq_moe_pf_body<5, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 6) xq_moe_pf_body<6, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 7) xq_moe_pf_body<7, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 2) xq_moe_pf_body<2, MT, NW, EPI, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        // PFHALF: the fractional widths (EXL3_FRAC_BITS code; the host routes them to V 0 only — pf2 has no HALF body)
        else if (bits == EXL3_FRAC_BITS(1)) xq_moe_pf_body<1, MT, NW, EPI, PROF, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == EXL3_FRAC_BITS(2)) xq_moe_pf_body<2, MT, NW, EPI, PROF, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == EXL3_FRAC_BITS(3)) xq_moe_pf_body<3, MT, NW, EPI, PROF, true>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
    } else {
        if (bits == 3) xq_moe_pf2_body<3, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 4) xq_moe_pf2_body<4, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 5) xq_moe_pf2_body<5, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 6) xq_moe_pf2_body<6, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 7) xq_moe_pf2_body<7, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
        else if (bits == 2) xq_moe_pf2_body<2, MT, NW, EPI, KS2, P2, PROF>(tr, xe, ye, ce, K, N, cta_tile, m_sup, xq_pf_dsm, svh_e, suh_de);
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
    // P7 B4: bits 5 (h paired mt 1/2 at 20/21, then the 8 t/s/g instances in the bits-3 order)
    out[20] = XqPf2Smem<1, 8, 4, 2, 5, 2>::halves * 2;  out[21] = XqPf2Smem<2, 8, 4, 2, 5, 2>::halves * 2;
    out[22] = XqPf2Smem<1, 8, 4, 3, 5>::halves * 2;   out[23] = XqPf2Smem<2, 8, 4, 3, 5>::halves * 2;
    out[24] = XqPf2Smem<4, 8, 4, 2, 5>::halves * 2;   out[25] = XqPf2Smem<8, 8, 2, 2, 5>::halves * 2;
    out[26] = XqPf2Smem<1, 16, 4, 3, 5>::halves * 2;  out[27] = XqPf2Smem<2, 16, 4, 3, 5>::halves * 2;
    out[28] = XqPf2Smem<4, 16, 4, 2, 5>::halves * 2;  out[29] = XqPf2Smem<8, 16, 2, 2, 5>::halves * 2;
    // P7 B4 leg 2: bits 2/6/7 blocks of 10 (t/s, g, h-paired); host mirror reads 30..59
    out[30] = XqPf2Smem<1, 8, 4, 3, 2>::halves * 2;   out[31] = XqPf2Smem<2, 8, 4, 3, 2>::halves * 2;
    out[32] = XqPf2Smem<4, 8, 4, 2, 2>::halves * 2;   out[33] = XqPf2Smem<8, 8, 2, 2, 2>::halves * 2;
    out[34] = XqPf2Smem<1, 16, 4, 3, 2>::halves * 2;   out[35] = XqPf2Smem<2, 16, 4, 3, 2>::halves * 2;
    out[36] = XqPf2Smem<4, 16, 4, 2, 2>::halves * 2;   out[37] = XqPf2Smem<8, 16, 2, 2, 2>::halves * 2;
    out[38] = XqPf2Smem<1, 8, 4, 2, 2, 2>::halves * 2;  out[39] = XqPf2Smem<2, 8, 4, 2, 2, 2>::halves * 2;
    out[40] = XqPf2Smem<1, 8, 4, 3, 6>::halves * 2;   out[41] = XqPf2Smem<2, 8, 4, 3, 6>::halves * 2;
    out[42] = XqPf2Smem<4, 8, 4, 2, 6>::halves * 2;   out[43] = XqPf2Smem<8, 8, 2, 2, 6>::halves * 2;
    out[44] = XqPf2Smem<1, 16, 4, 3, 6>::halves * 2;   out[45] = XqPf2Smem<2, 16, 4, 3, 6>::halves * 2;
    out[46] = XqPf2Smem<4, 16, 4, 2, 6>::halves * 2;   out[47] = XqPf2Smem<8, 16, 2, 2, 6>::halves * 2;
    out[48] = XqPf2Smem<1, 8, 4, 2, 6, 2>::halves * 2;  out[49] = XqPf2Smem<2, 8, 4, 2, 6, 2>::halves * 2;
    out[50] = XqPf2Smem<1, 8, 4, 3, 7>::halves * 2;   out[51] = XqPf2Smem<2, 8, 4, 3, 7>::halves * 2;
    out[52] = XqPf2Smem<4, 8, 4, 2, 7>::halves * 2;   out[53] = XqPf2Smem<8, 8, 2, 2, 7>::halves * 2;
    out[54] = XqPf2Smem<1, 16, 4, 3, 7>::halves * 2;   out[55] = XqPf2Smem<2, 16, 4, 3, 7>::halves * 2;
    out[56] = XqPf2Smem<4, 16, 4, 2, 7>::halves * 2;   out[57] = XqPf2Smem<8, 16, 2, 2, 7>::halves * 2;
    out[58] = XqPf2Smem<1, 8, 4, 2, 7, 2>::halves * 2;  out[59] = XqPf2Smem<2, 8, 4, 2, 7, 2>::halves * 2;
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

