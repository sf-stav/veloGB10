// exl3_bench/09_moe_pk_widths_ple.cuh: persistent MoE entries (xq_moe_gu_pk / xq_moe_dn_pk, width-generic B10) and the PLE gate/conv/state kernels.
// The host (exl3_forward::mpk_plan) picks NST/G/THREADS from the occupancy API.
// Section 9 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
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
// B10: width-generic entries (BITS 2,4,5,6,7 — K=3 keeps the diet entries above). The host
// (exl3_forward::mpk_plan) picks THREADS by width; smem geometry via mpk_smem_bytes_b.
#define MPK_GU_ENTRY_B(B, NS)                                                                        \
extern "C" __global__ void __launch_bounds__((B <= 3 ? 256 : 512), 1)                                \
xq_moe_gu_pk_b##B##_s##NS(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,      \
                   const __half* __restrict__ ain, const __half* __restrict__ suh_all,              \
                   __half* __restrict__ ygu, int M, int K, int N,                                   \
                   const int* __restrict__ esel, const int* __restrict__ idxmap,                    \
                   const __half* __restrict__ svh_all, const __half* __restrict__ suh_d_all,        \
                   __half* __restrict__ xhd, unsigned* __restrict__ cnt,                            \
                   const __half* __restrict__ x, const __half* __restrict__ sg,                     \
                   float* __restrict__ sgv, int flags) {                                            \
    XQ_PDL_ENTRY();                                                                                 \
    mpk_gu_body_g<B, NS>(base, offs, ain, suh_all, ygu, M, K, N, esel, idxmap, svh_all, suh_d_all,   \
                    xhd, cnt, x, sg, sgv, flags);                                                   \
}
#define MPK_DN_ENTRY_B(B, TK, NS)                                                                    \
extern "C" __global__ void __launch_bounds__((B <= 3 ? 256 : 512), 1)                                \
xq_moe_dn_pk_b##B##_k##TK##_s##NS(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs, \
          const __half* __restrict__ xhd, __half* __restrict__ yd, int M, int K, int N,             \
          const int* __restrict__ esel, const int* __restrict__ idxmap,                             \
          const __half* __restrict__ svh_all, unsigned* __restrict__ cnt,                           \
          const __half* __restrict__ ysh, const int* __restrict__ ids, const float* __restrict__ wts,\
          const int* __restrict__ slotmap, const float* __restrict__ sgv,                           \
          __half* __restrict__ out, int topk, int flags) {                                          \
    XQ_PDL_ENTRY();                                                                                 \
    mpk_dn_body_g<B, NS, TK>(base, offs, xhd, yd, M, K, N, esel, idxmap, svh_all, cnt, ysh, ids, wts,\
                    slotmap, sgv, out, topk, flags);                                                \
}
MPK_GU_ENTRY_B(2, 4)
MPK_GU_ENTRY_B(2, 8)
MPK_GU_ENTRY_B(4, 4)
MPK_GU_ENTRY_B(4, 8)
MPK_GU_ENTRY_B(5, 4)
MPK_GU_ENTRY_B(5, 8)
MPK_GU_ENTRY_B(6, 4)
MPK_GU_ENTRY_B(6, 8)
MPK_GU_ENTRY_B(7, 4)
MPK_GU_ENTRY_B(7, 8)
MPK_DN_ENTRY_B(2, 8, 4)
MPK_DN_ENTRY_B(2, 8, 8)
MPK_DN_ENTRY_B(2, 10, 4)
MPK_DN_ENTRY_B(2, 10, 8)
MPK_DN_ENTRY_B(4, 8, 4)
MPK_DN_ENTRY_B(4, 8, 8)
MPK_DN_ENTRY_B(4, 10, 4)
MPK_DN_ENTRY_B(4, 10, 8)
MPK_DN_ENTRY_B(5, 8, 4)
MPK_DN_ENTRY_B(5, 8, 8)
MPK_DN_ENTRY_B(5, 10, 4)
MPK_DN_ENTRY_B(5, 10, 8)
MPK_DN_ENTRY_B(6, 8, 4)
MPK_DN_ENTRY_B(6, 8, 8)
MPK_DN_ENTRY_B(6, 10, 4)
MPK_DN_ENTRY_B(6, 10, 8)
MPK_DN_ENTRY_B(7, 8, 4)
MPK_DN_ENTRY_B(7, 8, 8)
MPK_DN_ENTRY_B(7, 10, 4)
MPK_DN_ENTRY_B(7, 10, 8)

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
// PR#27-D1 (2026-10-30): the wrapper-owned staging ring must be sized for the WIDEST
// case the switch below can dispatch (case 6), not the historical 5 -- the body casts
// this storage to [2][8*BITS] views and the trellis stager writes 8*BITS words/slot,
// so [8*5] + --moe-coop=1 on 6-bit experts overran shared memory by 8 words/slot.
#define MOE_COOP_MAX_BITS 6

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
    // allocate one copy PER template instantiation (4 of them: BITS 3..6),
    // 4 x 28,672 B = 114,688 B > the 48 KiB static cap — the single wrapper
    // copy at MOE_COOP_MAX_BITS is sh_tr 12,288 + sh_a 16,384 = 28,672 B).
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
    __shared__ uint32_t sh_tr[MOE_COOP_STAGES][MOE_COOP_GST][2][8 * MOE_COOP_MAX_BITS]; // PR#27-D1: was [8*5]
    __shared__ __half   sh_a [MOE_COOP_STAGES][MOE_COOP_GST][16][16];
    switch (bits) {
        case 2: xq_moe_coop_body<2>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;  // PCKG-ALL: K=2 experts (1.66/2.05/2.18 packs)
        case 3: xq_moe_coop_body<3>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 4: xq_moe_coop_body<4>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 5: xq_moe_coop_body<5>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 6: xq_moe_coop_body<6>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
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
    __shared__ uint32_t sh_tr[MOE_COOP_STAGES][MOE_COOP_GST][2][8 * MOE_COOP_MAX_BITS]; // PR#27-D1: was [8*5]
    __shared__ __half   sh_a [MOE_COOP_STAGES][MOE_COOP_GST][16][16];
    switch (bits) {
        case 2: xq_moe_coop_body<2>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;  // PCKG-ALL: K=2 experts (1.66/2.05/2.18 packs)
        case 3: xq_moe_coop_body<3>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 4: xq_moe_coop_body<4>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 5: xq_moe_coop_body<5>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
        case 6: xq_moe_coop_body<6>(base, offs, xh, y, M, K, N, esel, &sh_tr[0][0][0][0], &sh_a[0][0][0][0]); break;
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
// q4k: 4-bit K/V (docs/KV_CACHE_Q4K.md) — the q8 H32 basis and per-32 f16 scale, 4-bit codes two
// dims per byte (144 B per row at hd=256).
#define XQ_KV_Q4K 5
#define XQ_R32 0.17677669529663688110f   // 1/sqrt(32), the reference implementation's r32
struct XqKv { char* k; char* v; long long rb; int fmt; int hd; };
__device__ __forceinline__ int xq_kv_fmt(int mpf) { return (int)(((unsigned)mpf) >> 28); }
__device__ __forceinline__ long long xq_kv_rowb(int fmt, int hd) {
    if (fmt == XQ_KV_Q4K) return (long long)(((hd / 2 + hd / 16) + 15) & ~15);
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
    if (fmt == XQ_KV_Q4K) {
        const float sm = __fmul_rn(__half2float(reinterpret_cast<const __half*>(row + hd / 2)[d >> 5]), 0.125f);
        const int c = (reinterpret_cast<const unsigned char*>(row)[d >> 1] >> ((d & 1) * 4)) & 0xF;
        return __fmaf_rn((float)(c - 8), sm, 0.5f * sm);
    }
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
    } else if (fmt == XQ_KV_Q4K) {
        const unsigned u = *reinterpret_cast<const unsigned*>(row + (d0 >> 1));
        const float sm = __fmul_rn(__half2float(reinterpret_cast<const __half*>(row + hd / 2)[d0 >> 5]), 0.125f);
        const float hs = 0.5f * sm;
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __fmaf_rn((float)((int)((u >> (4 * i)) & 0xF) - 8), sm, hs);
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
    if (fmt == XQ_KV_Q4K) {
        const int q = max(min(__float2int_rd(__fmaf_rn(vs, 8.0f, 8.0f)), 15), 0);
        const unsigned p = __shfl_xor_sync(0xffffffffu, (unsigned)q, 1);
        if ((lane & 1) == 0) reinterpret_cast<unsigned char*>(row)[tid >> 1] = (unsigned char)(q | (p << 4));
        if (lane == 0) reinterpret_cast<__half*>(row + hd / 2)[tid >> 5] = __float2half_rn(s);
        return;
    }
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

