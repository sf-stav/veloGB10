// exl3_bench/08_moe_persistent.cuh: W4/MOE persistent grouped-expert kernels (cp.async ring, L2 evict_first, WP20 epilogues).
// Target contract: sm_121 BASELINE.
// Section 8 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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

// ---- B10 (width closure): width-generic persistent grouped experts, BITS in {2,4,5,6,7}.
// K=3 keeps the diet entries above BYTE-IDENTICAL (mpk_consume's 2-word window decode is
// b3/b4-only by its own static_assert, wp20_a1_mainloop:6315); the generic bodies below are
// adapted copies of mpk_gu_body / mpk_dn_body with (a) geometry per width (a 16x16 trellis
// block is 32*BITS B: BB, KROW = 8*BB, STB per ring slot, NCH = 64*BITS issuer threads),
// (b) the consumer = wp20_a1_mainloop's SHFL window-table decode (the all-widths reference
// body: per-window (i0,i1,sf) tables + funnelshift), (c) THREADS = 256 at BITS<=3 else 512
// (NCH must stay whole-warp and leave >= 64 A threads; 1024 % THREADS == 0 for the sigmoid
// tree), and the >= 8-warp loops (stage_tile, svh epilogues, combine) pinned to warps 0..7.
template <int BITS> struct MpkGeom {
    static constexpr int BB = 32 * BITS, KROW = 8 * BB, STB = MPK_KSTEP * KROW,
                         NCH = STB / 16, THREADS = (BITS <= 3) ? 256 : 512,
                         ATHR = THREADS - NCH;
    static_assert(BITS >= 1 && BITS <= 8, "MpkGeom: width out of range");
    static_assert(NCH % 32 == 0 && NCH < THREADS, "MpkGeom: issuers must be whole warps below the block");
};

// Host-side smem mirror at width `bits` (MPK_STB == MpkGeom<3>::STB; bits 3 reproduces the
// legacy numbers exactly). Mirrored by exl3_forward::mpk_smem_bytes(bits).
__host__ __device__ __forceinline__ int mpk_smem_bytes_b(int nst, int M, int K, int gu, int fold, int bits) {
    const int epi = gu ? (M * 256 > 4096 ? M * 256 : 4096) : M * 256;
    return nst * (1024 * bits) + M * (K + 8) * 2 + ((gu && fold) ? K * 2 : 0) + epi;
}

// wp20_a1_mainloop's SHFL window tables, verbatim (W = 8*BITS words per k16 row).
template <int BITS>
__device__ __forceinline__ void mpk_shfl_consts(int lane, int (&s_i0m)[8], int (&s_i1m)[8], int (&s_sf)[8]) {
    constexpr int W = 8 * BITS;
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % W;
            s_i1m[h * 4 + s] = i1 % W;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
}

// Consumer, generic widths: the SHFL decode (wp20_a1_mainloop's A1_SHFL block, funnelshift
// form) on one ring slot's words; bfrag/A-frag/mma exactly as mpk_consume's.
template <int BITS>
__device__ __forceinline__ void mpk_consume_g(const unsigned char* sw, const __half* sa, int SK, int kb0,
                                              int lane, const int* s_i0m, const int* s_i1m, const int* s_sf,
                                              int gid, int tig, bool l0, bool l1, bool hi_rows,
                                              float (&acc)[2][4]) {
    constexpr int W = 8 * BITS;
    #pragma unroll
    for (int g = 0; g < MPK_KSTEP; ++g) {
        const uint32_t* wr = (const uint32_t*)(sw + g * MpkGeom<BITS>::KROW);
        const unsigned w0 = (lane < W) ? wr[lane] : 0u;
        const unsigned w1 = (W > 32 && lane + 32 < W) ? wr[lane + 32] : 0u;
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
            uint32_t bfrag[2];
            bfrag[0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
            bfrag[1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            const __half* ar0 = sa + gid * SK + (kb0 + g) * 16 + 2 * tig;
            const uint32_t a0 = l0 ? *(const uint32_t*)ar0 : 0u;
            const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + 8) : 0u;
            uint32_t a1 = 0u, a3 = 0u;
            if (hi_rows) {
                a1 = l1 ? *(const uint32_t*)(ar0 + 8 * SK) : 0u;
                a3 = l1 ? *(const uint32_t*)(ar0 + 8 * SK + 8) : 0u;
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[0]), "r"(bfrag[1]));
        }
    }
}

// mpk_issue_a / mpk_wait_a / mpk_prod_seek / mpk_prod_step / mpk_fold_a, geometry-templated
// (BITS=3 + THREADS=256 reproduces the helpers above exactly).
template <int BITS>
__device__ __forceinline__ void mpk_issue_a_g(unsigned sa_u, const __half* __restrict__ src, int M, int K,
                                              unsigned su_u, const __half* __restrict__ suh) {
    const int ta = (int)threadIdx.x - MpkGeom<BITS>::NCH;
    if (ta < 0) return;                                   // warp-uniform (NCH = whole warps)
    const int vpr = K >> 3, SK = K + 8;
    for (int i = ta; i < M * vpr; i += MpkGeom<BITS>::ATHR) {
        const int r = i / vpr, c = i - r * vpr;
        lmh_cp16(sa_u + (unsigned)((r * SK + c * 8) * 2), src + (size_t)r * K + c * 8);
    }
    if (suh != nullptr)
        for (int i = ta; i < vpr; i += MpkGeom<BITS>::ATHR)
            lmh_cp16(su_u + (unsigned)(i * 16), suh + (size_t)i * 8);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int BITS>
__device__ __forceinline__ void mpk_wait_a_g() {
    if ((int)threadIdx.x >= MpkGeom<BITS>::NCH) asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
}
template <int BITS>
__device__ __forceinline__ void mpk_fold_a_g(__half* sa, const __half* su, int M, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = K >> 7, SK = K + 8;
    #pragma unroll 2
    for (int bi = warp; bi < M * nblk; bi += MpkGeom<BITS>::THREADS / 32) {
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
struct MpkProdG { const unsigned char* src; int item, ks, slot, i; };
template <int BITS>
__device__ __forceinline__ void mpk_prod_seek_g(MpkProdG& p, const unsigned char* trb,
                                                const unsigned long long* s_off, int T, long long c_src) {
    const int e = p.item / T;
    p.src = trb + s_off[e] * 2ull + (long long)(p.item - e * T) * MpkGeom<BITS>::KROW + c_src;
}
template <int BITS, int NST>
__device__ __forceinline__ void mpk_prod_step_g(MpkProdG& p, unsigned ring_u, unsigned c_dst, bool c_live,
                                                long long rowB, int NKS, int total, int i_step, int i_end,
                                                const unsigned char* trb, const unsigned long long* s_off,
                                                int T, long long c_src, unsigned long long pol) {
    if (p.i < total) {
        if (c_live)
            lmh_cp16_ef(ring_u + (unsigned)(p.slot * MpkGeom<BITS>::STB) + c_dst,
                        p.src + (long long)p.ks * MPK_KSTEP * rowB, pol);
        ++p.i;
        if (++p.ks == NKS) {
            p.ks = 0;
            p.item += i_step;
            if (p.item < i_end) mpk_prod_seek_g<BITS>(p, trb, s_off, T, c_src);
        }
        if (++p.slot == NST) p.slot = 0;
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
// wp20_stage_tile assumes 8 warps (16-col blocks per warp); at 512 threads only warps 0..7
// own columns — the guarded wrapper.
__device__ __forceinline__ void mpk_stage_tile8(__half* yt, const float (&acc)[2][4], int M) {
    if ((threadIdx.x >> 5) >= 8) return;
    wp20_stage_tile(yt, acc, M);
}

// Generic gate/up body (adapted from mpk_gu_body; see the header comment).
template <int BITS, int NST>
__device__ __forceinline__ void mpk_gu_body_g(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ ain, const __half* __restrict__ suh_all,
                                              __half* __restrict__ ygu, int M, int K, int N,
                                              const int* __restrict__ esel, const int* __restrict__ idxmap,
                                              const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,
                                              __half* __restrict__ xhd, unsigned* __restrict__ cnt,
                                              const __half* __restrict__ x, const __half* __restrict__ sg,
                                              float* __restrict__ sgv, int flags) {
    constexpr int THR = MpkGeom<BITS>::THREADS;
    static_assert(NST >= 3, "ring needs >= 3 slots");
    extern __shared__ __align__(16) unsigned char mpk_sm[];
    __shared__ unsigned long long s_off[MPK_EMAX];
    __shared__ int s_ix[MPK_EMAX];
    __shared__ int s_flag;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const bool fold = (flags & MPK_FOLD) != 0;
    const int T = N >> 7, NKS = (K >> 4) / MPK_KSTEP, SK = K + 8, P = T >> 1;
    unsigned char* ring = mpk_sm;
    __half* sa = (__half*)(mpk_sm + NST * MpkGeom<BITS>::STB);
    __half* su = (__half*)((unsigned char*)sa + M * SK * 2);
    __half* epi = (__half*)((unsigned char*)su + (fold ? K * 2 : 0));
    const int live = *esel;
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)mpk_smem_bytes_b(NST, M, K, 1, fold ? 1 : 0, BITS)) __trap();
        if (M < 1 || M > 16 || (N & 255) != 0 || (K & 127) != 0 || (K % (16 * MPK_KSTEP)) != 0) __trap();
        if (live < 0 || live > MPK_EMAX) __trap();
    }
    for (int e = tid; e < live && e < MPK_EMAX; e += THR) { s_off[e] = offs[e]; s_ix[e] = idxmap[e]; }
    int i_beg, i_step, i_end, n_items;
    mpk_items(live * T, flags, i_beg, i_step, i_end, n_items);
    const int total = n_items * NKS;
    __syncthreads();

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const unsigned sa_u = (unsigned)__cvta_generic_to_shared(sa);
    const unsigned su_u = (unsigned)__cvta_generic_to_shared(su);
    const bool c_live = tid < MpkGeom<BITS>::NCH;
    const int cg = tid / (MpkGeom<BITS>::KROW / 16), cr = tid - cg * (MpkGeom<BITS>::KROW / 16);
    const long long rowB = (long long)(N >> 4) * MpkGeom<BITS>::BB;
    const long long c_src = (long long)cg * rowB + cr * 16;
    const unsigned c_dst = (unsigned)(cg * MpkGeom<BITS>::KROW + cr * 16);
    const unsigned char* trb = (const unsigned char*)base;
    MpkProdG pr;
    pr.item = i_beg; pr.ks = 0; pr.slot = 0; pr.i = 0; pr.src = trb;
    bool a_pend = false;
    if (n_items > 0) {
        mpk_prod_seek_g<BITS>(pr, trb, s_off, T, c_src);
        #pragma unroll 1
        for (int s = 0; s < NST - 1; ++s)
            mpk_prod_step_g<BITS, NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        const int e0 = i_beg / T;
        mpk_issue_a_g<BITS>(sa_u, ain + (fold ? 0ll : (long long)e0 * M * K), M, K, su_u,
                            fold ? suh_all + (long long)s_ix[e0] * K : nullptr);
        a_pend = true;
    }
    // shared-expert gate sigmoid (mpk_gu_body's): the EXACT original 256-thread sequence —
    // partials and halving tree written only by threads 0..255 (threads 256..511 idle through
    // the tree; the B10 b5 bug was the unguarded (tid & 255) tree letting both halves of the
    // CTA duplicate-write red[] concurrently — values identical, but empirically nondeterministic
    // at 512 threads: sgv readback varied run-to-run while every input was fixed).
    for (int r = (int)blockIdx.x; r < M; r += (int)gridDim.x) {
        float* red = (float*)epi;
        __syncthreads();
        #pragma unroll 1
        for (int q = 0; q < 4; ++q) {
            const int vt = (tid & 255) + 256 * q;
            float part = 0.0f;
            #pragma unroll 1
            for (int c = vt; c < K; c += 1024)
                part = __fmaf_rn(xq_h2f(sg[c]), xq_h2f(x[(long long)r * K + c]), part);
            if (tid < 256) red[vt] = part;
        }
        __syncthreads();
        for (int s2 = 512; s2 > 0; s2 >>= 1) {
            if (tid < 256)
                for (int i = tid; i < s2; i += 256) red[i] = __fadd_rn(red[i], red[i + s2]);
            __syncthreads();
        }
        if (tid == 0) sgv[r] = xq_sig(red[0]);
    }
    if (n_items == 0) return;

    int s_i0m[8], s_i1m[8], s_sf[8];
    mpk_shfl_consts<BITS>(lane, s_i0m, s_i1m, s_sf);
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
            mpk_wait_a_g<BITS>();
            if (fold) { mpk_fold_a_g<BITS>(sa, su, M, K); __syncthreads(); }
            a_pend = false;
        }
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        mpk_prod_step_g<BITS, NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        if (warp < 8)
            mpk_consume_g<BITS>(ring + c_slot * MpkGeom<BITS>::STB + warp * MpkGeom<BITS>::BB, sa, SK,
                                c_ks * MPK_KSTEP, lane, s_i0m, s_i1m, s_sf, gid, tig, l0, l1, hi_rows, acc);
        if (++c_slot == NST) c_slot = 0;
        if (++c_ks == NKS) {
            const int e = c_item / T, tile = c_item - e * T, ix = s_ix[e];
            __half* yt = epi;
            mpk_stage_tile8(yt, acc, M);
            __syncthreads();
            const int nx = c_item + i_step;
            if (nx < i_end && nx / T != e) {
                const int e2 = nx / T;
                mpk_issue_a_g<BITS>(sa_u, ain + (fold ? 0ll : (long long)e2 * M * K), M, K, su_u,
                                    fold ? suh_all + (long long)s_ix[e2] * K : nullptr);
                a_pend = true;
            }
            const int cta_n = tile * 128;
            __half* ye = ygu + (long long)e * M * N;
            if (warp < 8)                               // single-writer, exact original
                for (int r = warp; r < M; r += 8) {       // == xq_had_svh_multi
                const uint2 o = wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
                *(uint2*)(yt + r * 128 + lane * 4) = o;
                *(uint2*)(ye + (long long)r * N + cta_n + lane * 4) = o;
            }
            const int p = tile < P ? tile : tile - P;
            if (wp20_arrive(cnt + e * P + p, 2u, &s_flag)) {
                const int KD = N >> 1;
                const int ptile = tile < P ? tile + P : tile - P;
                const bool own_gate = tile < P;
                if (warp < 8)                             // single-writer, exact original
                    for (int r = warp; r < M; r += 8) {
                    const uint2 pw = __ldcg((const uint2*)(ye + (long long)r * N + ptile * 128 + lane * 4));
                    const uint2 ow = *(const uint2*)(yt + r * 128 + lane * 4);
                    float g[4], u[4], sv[4], dv[4], v[4];
                    wp20_unpack4(own_gate ? ow : pw, g);
                    wp20_unpack4(own_gate ? pw : ow, u);
                    wp20_unpack4(__ldg((const uint2*)(suh_d_all + (long long)ix * KD + p * 128 + lane * 4)), sv);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        dv[j] = xq_h2f(xq_f2h(xq_silu(g[j]) * u[j]));
                    wp20_suh_had(dv, sv, v, lane);
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

// Generic down body (adapted from mpk_dn_body).
template <int BITS, int NST, int TOPK>
__device__ __forceinline__ void mpk_dn_body_g(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xhd, __half* __restrict__ yd,
                                              int M, int K, int N, const int* __restrict__ esel,
                                              const int* __restrict__ idxmap, const __half* __restrict__ svh_all,
                                              unsigned* __restrict__ cnt, const __half* __restrict__ ysh,
                                              const int* __restrict__ ids, const float* __restrict__ wts,
                                              const int* __restrict__ slotmap, const float* __restrict__ sgv,
                                              __half* __restrict__ out, int topk, int flags) {
    constexpr int THR = MpkGeom<BITS>::THREADS;
    static_assert(NST >= 3, "ring needs >= 3 slots");
    static_assert(TOPK >= 1 && TOPK <= 16, "mpk_dn_body_g: TOPK instance out of the planned range");
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
    __half* sa = (__half*)(mpk_sm + NST * MpkGeom<BITS>::STB);
    __half* epi = (__half*)((unsigned char*)sa + M * SK * 2);
    const int live = *esel;
    const int nrt = M * TOPK;
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)mpk_smem_bytes_b(NST, M, K, 0, 0, BITS)) __trap();
        if (M < 1 || M > 16 || (N & 127) != 0 || (K & 127) != 0 || (K % (16 * MPK_KSTEP)) != 0) __trap();
        if (live < 0 || live > MPK_EMAX || nrt > MPK_RTMAX) __trap();
        if (topk != TOPK) __trap();
    }
    for (int e = tid; e < live && e < MPK_EMAX; e += THR) { s_off[e] = offs[e]; s_ix[e] = idxmap[e]; }
    for (int i = tid; i < nrt && i < MPK_RTMAX; i += THR) { s_slot[i] = slotmap[ids[i]]; s_w[i] = wts[i]; }
    int i_beg, i_step, i_end, n_items;
    mpk_items(live * T, flags, i_beg, i_step, i_end, n_items);
    const int total = n_items * NKS;
    __syncthreads();
    if (n_items == 0) return;

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const unsigned sa_u = (unsigned)__cvta_generic_to_shared(sa);
    const bool c_live = tid < MpkGeom<BITS>::NCH;
    const int cg = tid / (MpkGeom<BITS>::KROW / 16), cr = tid - cg * (MpkGeom<BITS>::KROW / 16);
    const long long rowB = (long long)(N >> 4) * MpkGeom<BITS>::BB;
    const long long c_src = (long long)cg * rowB + cr * 16;
    const unsigned c_dst = (unsigned)(cg * MpkGeom<BITS>::KROW + cr * 16);
    const unsigned char* trb = (const unsigned char*)base;
    MpkProdG pr;
    pr.item = i_beg; pr.ks = 0; pr.slot = 0; pr.i = 0; pr.src = trb;
    mpk_prod_seek_g<BITS>(pr, trb, s_off, T, c_src);
    #pragma unroll 1
    for (int s = 0; s < NST - 1; ++s)
        mpk_prod_step_g<BITS, NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
    mpk_issue_a_g<BITS>(sa_u, xhd + (long long)(i_beg / T) * M * K, M, K, 0u, nullptr);
    bool a_pend = true;

    int s_i0m[8], s_i1m[8], s_sf[8];
    mpk_shfl_consts<BITS>(lane, s_i0m, s_i1m, s_sf);
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
            mpk_wait_a_g<BITS>();
            a_pend = false;
        }
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        mpk_prod_step_g<BITS, NST>(pr, ring_u, c_dst, c_live, rowB, NKS, total, i_step, i_end, trb, s_off, T, c_src, pol);
        if (warp < 8)
            mpk_consume_g<BITS>(ring + c_slot * MpkGeom<BITS>::STB + warp * MpkGeom<BITS>::BB, sa, SK,
                                c_ks * MPK_KSTEP, lane, s_i0m, s_i1m, s_sf, gid, tig, l0, l1, hi_rows, acc);
        if (++c_slot == NST) c_slot = 0;
        if (++c_ks == NKS) {
            const int e = c_item / T, tile = c_item - e * T, ix = s_ix[e];
            __half* yt = epi;
            mpk_stage_tile8(yt, acc, M);
            __syncthreads();
            const int nx = c_item + i_step;
            if (nx < i_end && nx / T != e) {
                mpk_issue_a_g<BITS>(sa_u, xhd + (long long)(nx / T) * M * K, M, K, 0u, nullptr);
                a_pend = true;
            }
            const int cta_n = tile * 128;
            if (warp < 8)                               // single-writer, exact original
                for (int r = warp; r < M; r += 8)
                    *(uint2*)(yd + ((long long)e * M + r) * N + cta_n + lane * 4) =
                        wp20_svh_row(yt + r * 128, svh_all + (long long)ix * N + cta_n, lane);
            if (wp20_arrive(cnt + tile, (unsigned)live, &s_flag)) {
                if (tid < 256)                           // single-writer, exact original
                    for (int q = tid; q < M * 32; q += 256) {
                    const int r = q >> 5, i = cta_n + ((q & 31) << 2);
                    const int* sl = s_slot + r * TOPK;
                    const float* sw = s_w + r * TOPK;
                    uint2 raw[TOPK];
                    #pragma unroll
                    for (int j = 0; j < TOPK; ++j) {
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

