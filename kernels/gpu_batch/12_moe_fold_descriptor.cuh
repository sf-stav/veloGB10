// gpu_batch/12_moe_fold_descriptor.cuh: shared-slot descriptor, E12/E8/E9 fold kernels, rope/write_kv fold, E11 expert-GEMM tuning variants.
// E11 variants carry a BITWISE CONTRACT with gemm_moe_mma_fp4_fold.
// Section 12 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================================================================================================
// THE SHARED-SLOT DESCRIPTOR (device i64[16], written once at load; the fold GEMMs take its
// POINTER as their single extra arg — cudarc launch tuples cap at 12 elements):
//   [0..4)  gate_up { qweight, scales, gs, (SK<<32)|sk_off }   SK = h, sk_off = 0 (col-shard: M)
//   [4..8)  down    { qweight, scales, gs, (sdn_k<<32)|rank·sdn_k }
//   [8]     the shared slot's gu M (2·si, or 2·si_local = si under the shard)
//   [12]    the down's M (= h, uniform no-op; keeps the uniform desc[8] read in-bounds)
// A launch passes the pointer of the descriptor it reads: the GU GEMMs pass the BASE (their own
// desc at [0..4), the down's sk at [7] for the rank's band offset), the DOWN GEMMs pass base+32
// (their own desc at [0..4) with sk_off = rank·sdn_k, the down's M at [8]).
// ===================================================================================================

// The slot-path fold GEMM: gemm_moe_mma_fp4 with Kslots = k+1 columns (the last is the shared
// slot). Routed slots are byte-identical to gemm_moe_mma_fp4 (same per-element k-chain, same
// N=1 epilogue). The shared slot reads its weight geometry from the descriptor: kk = desc[3]>>32,
// X k-offset = desc[3]&0xffffffff, its own M = desc[8] (== the launch M -> identity C rows; the
// sharded gu half -> the C rows map into the rank's bands of the full-width slot: gate rows at
// [rank_off, rank_off+sl), up rows at [2·sl+rank_off, 2·sl+rank_off+sl) with sl = desc[8]/2 and
// rank_off = the down desc's sk_off — the weight row m' lands at C row m' + rank_off for the gate
// band and m' + sl + rank_off for the up band, which is exactly the paired-ColSegs layout).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        // Sharded-GU epilogue: the C row is the launch tile's row; the weight row = m + m_shift
        // (the rank's band). The whole block shares one band, so the per-tile scale is gs_w[mt_s].
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        // gs_w is ALREADY the per-expert base (gs + e·(M>>4), or the shared tensor's own gs); mt_g
        // is the full tile index used for the WEIGHT reads and must NOT be added again — the
        // epilogue indexes gs_w[mt] itself. (The old gs_w + mt_g read the scale at
        // 2·(e·(M>>4) + mt) — double the expert AND tile offsets — garbage/OOB for e>0, mt>0.)
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ===================================================================================================
// E11 — expert-GEMM tuning variants of gemm_moe_mma_fp4_fold (branch work/e11). BITWISE CONTRACT:
// every variant preserves the exact per-element k-chain — same pair partition (p = warp + 8k
// ascending), same per-pair block order (2p then 2p+1), same per-warp fp32 accumulation, same
// warp-order epilogue reduction. Variants differ ONLY in block→tile mapping or instruction
// scheduling, so col-0 batch-invariance and the oracle-identical MoE gate are untouched
// (gates: --probe-binv + bench_mtp LOSSLESS + acceptance).

// ---- gemm_moe_mma_fp4_fold_u4 — the u4 burst-load structure (gemm_moe_mma_fp4_u4's winning
// load schedule) applied to the fold kernel: FOUR pair-iterations' global loads hoisted per loop
// step, then the MMAs execute in locked ascending order (the acc dependency chain serializes
// them). Same pair ownership as the plain fold (p = warp + 8k ascending), so bit-identical.
// npair % 4 == 0 is NOT required — a tail guard runs the leftover pairs in the plain ascending
// loop (same order). launch_bounds(256,3) mirrors the u4 probe kernel's occupancy.
extern "C" __global__ __launch_bounds__(256, 3) void gemm_moe_mma_fp4_fold_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    int p0;
    for (p0 = warp; p0 + 3 * MMA_NW < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2];
        const uint8_t* sctv[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + MMA_NW * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sctv[u] = sct + tile * 16;
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sctv[u][g]),      s0hi = e4m3_f(sctv[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            const int k0 = (p0 + MMA_NW * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
            uint32_t r0[2] = { Xl[t], Xl[t + 4] };
            mma_m16n8k16(acc[0], ra, r0);                            // block 2p,   columns 0..7
            const float s1lo = e4m3_f(sctv[u][g + 16]), s1hi = e4m3_f(sctv[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { Xl[t + 8], Xl[t + 12] };
            mma_m16n8k16(acc[0], ra, r2);                            // block 2p+1, columns 0..7
        }
    }
    // Tail (npair not a multiple of 32): the leftover pairs in the SAME ascending order.
    for (int p = p0; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_x2 — the x2 structure (TWO 16-row weight tiles per CTA share the X
// fragments and interleave two INDEPENDENT K-chains in one loop body) applied to the fold kernel.
// Per-tile k-chains are UNTOUCHED (each tile's pairs in the same ascending order, epilogue reduce
// in the same warp order) — only which block computes which mt PAIR changes. M % 32 == 0 required
// (hy3: gate_up M=3072, down M=4096 — both even). The band (sharded-GU) mapping is computed per
// tile. launch_bounds(256,4) mirrors the x2 probe kernel.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_fold_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g0, mt_g1;
    bool band0 = false, band1 = false;
    int m_shift0 = 0, m_shift1 = 0, mt_s0 = mt0, mt_s1 = mt1, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) {
                Cb[mt0 * 16 + threadIdx.x] = f2b(0.f);
                Cb[mt1 * 16 + threadIdx.x] = f2b(0.f);
            }
            return;
        }
        mt_g0 = (long long)e * (M >> 4) + mt0;
        mt_g1 = mt_g0 + 1;                                           // adjacent tile of the same expert
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cb0 = mt0 << 4;
            if (cb0 < sl + rank_off) { mt_s0 = (cb0 - rank_off) >> 4;  m_shift0 = -rank_off;      band0 = true; }
            else                     { mt_s0 = (cb0 - sl - rank_off) >> 4; m_shift0 = -(sl + rank_off); band0 = true; }
            const int cb1 = mt1 << 4;
            if (cb1 < sl + rank_off) { mt_s1 = (cb1 - rank_off) >> 4;  m_shift1 = -rank_off;      band1 = true; }
            else                     { mt_s1 = (cb1 - sl - rank_off) >> 4; m_shift1 = -(sl + rank_off); band1 = true; }
        } else { mt_s0 = mt0; mt_s1 = mt1; }
        mt_g0 = mt_s0; mt_g1 = mt_s1;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mt_g0 * nblk + (p << 1);
        const long long tileB = mt_g1 * nblk + (p << 1);
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = sct + tileA * 16;
        const uint8_t* sctB = sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                             // A block 2p,   columns 0..7
        mma_m16n8k16(accB[0], raB, rb0);                             // B block 2p,   columns 0..7
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                             // A block 2p+1, columns 0..7
        mma_m16n8k16(accB[0], raB, rb2);                             // B block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band0) {
        const float v = mma_warp_reduce(sh, accA);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt0 * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift0;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s0]);
    } else {
        mma_epilogue(sh, accA, Cb, nullptr, gs_w, mt_s0, M, 1);
    }
    __syncthreads();
    if (band1) {
        const float v = mma_warp_reduce(sh, accB);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt1 * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift1;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s1]);
    } else {
        mma_epilogue(sh, accB, Cb, nullptr, gs_w, mt_s1, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_rast — the rasterization candidate: the SAME fold kernel with the
// grid axes swapped (blockIdx.x = bslot, blockIdx.y = mt; grid = (bkf, M/16)). Pure scheduling —
// the block→tile map changes which (mt, slot) blocks are co-resident, the per-element math is
// untouched.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold_rast(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int bslot = blockIdx.x, mt = blockIdx.y;                   // AXES SWAPPED vs the fold
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- occupancy twins of the plain fold (launch_bounds 256,5 and 256,4): more registers per
// thread for the scheduler at lower co-resident-block counts — the launch_bounds knob the E11
// "safe cluster/PDL/occupancy" candidate list wants measured at Hy3's shapes. Identical body.
extern "C" __global__ __launch_bounds__(256, 5) void gemm_moe_mma_fp4_fold_lb5(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_fold_lb4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_pdl — the E9 PDL (programmatic dependent launch) twin of the fold
// kernel. pdl==1: publish the launch-completion edge, stream THIS BLOCK's weight tiles into L2
// (read-only — the loads are the same tile range the k-loop performs; kept alive by a never-true
// sink store), then griddepcontrol.wait gates the X reads. The weight SELECTION (ids for routed
// slots, sdesc for the shared slot) is final by the time this kernel launches (the up GEMM and
// silu — which consumed ids — run between the router and the down GEMM; the shared slot's
// descriptor is a kernel argument), so the prologue is dependency-safe: it must only wait before
// touching X. pdl==0 skips the preamble — byte-identical to the plain fold. This is the E11
// "safe PDL" candidate: the down GEMM's weight stream overlaps the silu (+ the router tail), and
// the shared slot's blocks overlap the routed up GEMM. Numerics are untouched (the sink store is
// never executed; the loads are the same values the compute reads).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold_pdl(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc, int pdl)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;

    if (pdl) {
        // E9 prologue: publish, prefetch this block's tile, wait. The prefetch issues exactly the
        // weight/scale loads the k-loop performs (same tile, same pairs, same lanes) — the compute
        // then hits L2. The shared slot's descriptor is a kernel argument (constant); the routed
        // slot's ids were consumed by the up GEMM before this down GEMM launched, so both are safe
        // to read before the wait. The never-true store keeps the loads alive.
        asm volatile("griddepcontrol.launch_dependents;");
        const uint32_t* Wt32p = reinterpret_cast<const uint32_t*>(wt);
        unsigned sink = 0u;
        for (int p = warp; p < npair; p += MMA_NW) {
            const long long tile = mt_g * nblk + (p << 1);
            sink ^= Wt32p[tile * 32 + lane];
            sink ^= Wt32p[tile * 32 + 32 + lane];
            sink ^= (unsigned)sct[tile * 16 + g] << 8;
            sink ^= (unsigned)sct[tile * 16 + g + 8] << 16;
        }
        if (sink == 0xDEADBEEFu) C[threadIdx.x] = f2b(0.f);   // never true — keeps the loads alive
        asm volatile("griddepcontrol.wait;");
    }

    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- E11 probe fill kernel: writes a deterministic pattern over `n` uint32s (probe-only — stands
// in for the silu/router as the PDL predecessor so the overlap mechanism can be measured in
// isolation). Content-irrelevant; the fold GEMM reads X AFTER griddepcontrol.wait in the pdl arm.
extern "C" __global__ void e11_fill_x_b(unsigned int* X, int n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    X[idx] = (unsigned int)(idx * 2654435761u);
}

// The grouped-path fold GEMM: gemm_moe_grouped_mma_fp4 with the early-exit bound poff[nef]
// (nef = ne+1 — the routed band PLUS the shared region) and the descriptor-driven shared region
// [poff[nef-1], poff[nef]). Routed groups are byte-identical to gemm_moe_grouped_mma_fp4 (same
// weight-reuse structure, same per-element k-chain); a shared-region group computes the shared
// weight from the descriptor — the GU writes the rank's bands (C rows per the paired layout), the
// down writes its full h rows reading the rank's act half at the k-offset (bit-identical to the
// old act_half slice feeding the down GEMM).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_grouped_mma_fp4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int nef,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[nef]) return;   // the fold bound = poff[ne+1] (routed + shared total)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, N = 16;
    const bool is_shared = (nt * 16 >= poff[nef - 1]);

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (!is_shared) {
        const int e = tile_e[nt * 2] - expert_base;
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K + sk_off;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        uint32_t ra[4];
        ra[0]=fp4_pair_bf16(wq0,s0lo); ra[1]=fp4_pair_bf16(wq0>>8,s0hi); ra[2]=fp4_pair_bf16(wq0>>16,s0lo); ra[3]=fp4_pair_bf16(wq0>>24,s0hi);
        uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
        mma_m16n8k16(acc[0], ra, rb0); mma_m16n8k16(acc[1], ra, rb1);
        ra[0]=fp4_pair_bf16(wq1,s1lo); ra[1]=fp4_pair_bf16(wq1>>8,s1hi); ra[2]=fp4_pair_bf16(wq1>>16,s1lo); ra[3]=fp4_pair_bf16(wq1>>24,s1hi);
        uint32_t rb2[2]={Xl[t+8],Xl[t+12]}, rb3[2]={Xh[t+8],Xh[t+12]};
        mma_m16n8k16(acc[0], ra, rb2); mma_m16n8k16(acc[1], ra, rb3);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        // gs_w is the per-expert base; mt_g already carries the expert offset (weight reads) and
        // must not be re-added — the epilogue indexes gs_w[mt_s] itself (see the slot-fold kernel).
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, N);
    }
}

// ---- gemm_moe_grouped_mma_fp4_x4 — the LOST 32-token wide grouped kernel (the E23 wide prefill
// path's default). ONE block per 32-token group; the block treats the group as TWO 16-token halves
// (moe_offsets_b pads experts to 16, not 32, so a 32-token group may straddle two experts) and
// resolves each half's expert independently — when the halves share an expert, the weight loads hit
// the same addresses (L2-cached), halving the expert-weight DRAM re-reads vs the 16-token kernel;
// the X fragments are shared by all four 8-token n-tiles. Per-element contract is identical to
// gemm_moe_grouped_mma_fp4: same pair partition (p = warp + 8k ascending), same per-pair block
// order (2p then 2p+1), same per-element fp32 accumulation, same fixed-order warp epilogue per
// 16-token half — only which block covers which group and the instruction scheduling change.
// grid = (M/16, ppad/32); the early-exit bound is poff[ne] read ON DEVICE (E16 device-only form).
// A half whose rows fall past the real padded total (the last group's tail) clamps its expert to
// the other half's — its outputs are padding rows (never combined), and the read stays in-bounds.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 32 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 32;
    // Per-half experts (8-tile index nt*4 + 2*h). The last group's tail half clamps to the first.
    const int ntiles = poff[ne] >> 3;
    const int e0 = tile_e[nt * 4] - expert_base;
    const int e1 = (nt * 4 + 2 < ntiles) ? (tile_e[nt * 4 + 2] - expert_base) : e0;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 32) * K;  // this group's 32 tokens
    const long long mtA = (long long)e0 * (M >> 4) + mt;
    const long long mtB = (long long)e1 * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 32) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = mtB * nblk + (p << 1);
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = Sct + tileA * 16;
        const uint8_t* sctB = Sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        const uint32_t* Xl2 = reinterpret_cast<const uint32_t*>(X + xr0 + 16 * K + k0);
        const uint32_t* Xh2 = reinterpret_cast<const uint32_t*>(X + xr1 + 16 * K + k0);
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
        uint32_t rb2[2]={Xl2[t],Xl2[t+4]}, rb3[2]={Xh2[t],Xh2[t+4]};
        mma_m16n8k16(accA[0], raA, rb0); mma_m16n8k16(accA[1], raA, rb1); // half 0, block 2p
        mma_m16n8k16(accB[0], raB, rb2); mma_m16n8k16(accB[1], raB, rb3); // half 1, block 2p
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb4[2]={Xl[t+8],Xl[t+12]}, rb5[2]={Xh[t+8],Xh[t+12]};
        uint32_t rb6[2]={Xl2[t+8],Xl2[t+12]}, rb7[2]={Xh2[t+8],Xh2[t+12]};
        mma_m16n8k16(accA[0], raA, rb4); mma_m16n8k16(accA[1], raA, rb5); // half 0, block 2p+1
        mma_m16n8k16(accB[0], raB, rb6); mma_m16n8k16(accB[1], raB, rb7); // half 1, block 2p+1
    }
    __shared__ float sh[MMA_SMEM];
    const float* gsA = gs + (long long)e0 * (M >> 4);
    const float* gsB = gs + (long long)e1 * (M >> 4);
    mma_epilogue(sh, accA, Cb, nullptr, gsA, mt, M, 16);
    __syncthreads();                                                    // sh read->next-write hazard
    mma_epilogue(sh, accB, Cb + (long long)16 * M, nullptr, gsB, mt, M, 16);
}

// ---- gemm_moe_grouped_mma_fp4_x4_fold — the wide-prefill fold twin of the x4 kernel: 32-token
// groups, the fold bound poff[nef] = poff[ne+1], and the descriptor-driven shared region. Each
// 16-token half is resolved independently: routed iff its base < poff[nef-1] (= poff[ne], 16-
// aligned so a half never straddles the routed/shared boundary); the shared half reads the shared
// weight from the descriptor (kk/sk_off at desc[3], M at desc[8]) with the band/identity C-row
// mapping — the GU writes the rank's bands, the down writes its full h rows reading the rank's act
// half at the k-offset. A tail half past the real total (never in the fold: the shared region is
// pad32(ns) and the routed band 16-aligned, but the group-level exit covers poff[nef] mod 32)
// clamps its expert to the other half's; its rows are padding (never combined).
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int nef,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 32 >= poff[nef]) return;   // the fold bound = poff[ne+1]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, N = 32;
    const int shared_start = poff[nef - 1];
    const int ntiles = poff[nef] >> 3;

    // Per-half geometry: {weight ptrs, gs base, kk, sk_off, mt_g, band, m_shift, mt_s, m_local}.
    const uint8_t* wt0 = Wt; const uint8_t* sct0 = Sct; const float* gs0 = gs;
    const uint8_t* wt1 = Wt; const uint8_t* sct1 = Sct; const float* gs1 = gs;
    int kk0 = K, kk1 = K, so0 = 0, so1 = 0;
    long long mg0, mg1;
    bool band0 = false, band1 = false;
    int ms0 = mt, ms1 = mt, msh0 = 0, msh1 = 0, ml0 = M, ml1 = M;

    const bool sh0 = (nt * 32     >= shared_start);
    const bool sh1 = (nt * 32 + 16 >= shared_start);
    // Route each half: shared -> descriptor; routed -> tile_e (clamp a tail half to the other's).
    if (sh0) {
        const long long sk = sdesc[3];
        kk0 = (int)(sk >> 32); so0 = (int)(sk & 0xffffffffu);
        ml0 = (int)sdesc[8];
        wt0 = (const uint8_t*)(uintptr_t)sdesc[0];
        sct0 = (const uint8_t*)(uintptr_t)sdesc[1];
        gs0 = (const float*)(uintptr_t)sdesc[2];
        if (ml0 != M) {
            const int sl = ml0 >> 1, rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { ms0 = (cbase - rank_off) >> 4; msh0 = -rank_off; band0 = true; }
            else                       { ms0 = (cbase - sl - rank_off) >> 4; msh0 = -(sl + rank_off); band0 = true; }
        } else ms0 = mt;
        mg0 = ms0;
    } else {
        const int e = (nt * 4 < ntiles) ? (tile_e[nt * 4] - expert_base) : 0;
        mg0 = (long long)e * (M >> 4) + mt;
        gs0 = gs + (long long)e * (M >> 4);
    }
    if (sh1) {
        const long long sk = sdesc[3];
        kk1 = (int)(sk >> 32); so1 = (int)(sk & 0xffffffffu);
        ml1 = (int)sdesc[8];
        wt1 = (const uint8_t*)(uintptr_t)sdesc[0];
        sct1 = (const uint8_t*)(uintptr_t)sdesc[1];
        gs1 = (const float*)(uintptr_t)sdesc[2];
        if (ml1 != M) {
            const int sl = ml1 >> 1, rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { ms1 = (cbase - rank_off) >> 4; msh1 = -rank_off; band1 = true; }
            else                       { ms1 = (cbase - sl - rank_off) >> 4; msh1 = -(sl + rank_off); band1 = true; }
        } else ms1 = mt;
        mg1 = ms1;
    } else {
        const int e = (nt * 4 + 2 < ntiles) ? (tile_e[nt * 4 + 2] - expert_base) : (tile_e[nt * 4] - expert_base);
        mg1 = (long long)e * (M >> 4) + mt;
        gs1 = gs + (long long)e * (M >> 4);
    }

    const int nb0 = kk0 >> 4, nb1 = kk1 >> 4, np0 = nb0 >> 1, np1 = nb1 >> 1;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 32) * K;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    __nv_bfloat16* Cb = C + (long long)(nt * 32) * M;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* W0 = reinterpret_cast<const uint32_t*>(wt0);
    const uint32_t* W1 = reinterpret_cast<const uint32_t*>(wt1);
    // The k-loop: both halves take the same pairs p = warp + 8k ascending; half 0 reads its own
    // weight, half 1 its own (the same addresses when both are routed with the same expert).
    const int npair = max(np0, np1);
    for (int p = warp; p < npair; p += MMA_NW) {
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + so0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + so0 + k0);
        const uint32_t* Xl2 = reinterpret_cast<const uint32_t*>(X + xr0 + so1 + 16 * K + k0);
        const uint32_t* Xh2 = reinterpret_cast<const uint32_t*>(X + xr1 + so1 + 16 * K + k0);
        if (p < np0) {
            const long long tileA = mg0 * nb0 + (p << 1);
            const uint32_t wA0 = W0[tileA * 32 + lane];
            const uint32_t wA1 = W0[tileA * 32 + 32 + lane];
            const uint8_t* sctA = sct0 + tileA * 16;
            const float a0lo = e4m3_f(sctA[g]), a0hi = e4m3_f(sctA[g + 8]);
            const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
            uint32_t raA[4];
            raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
            uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
            mma_m16n8k16(accA[0], raA, rb0); mma_m16n8k16(accA[1], raA, rb1);
            raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
            uint32_t rb4[2]={Xl[t+8],Xl[t+12]}, rb5[2]={Xh[t+8],Xh[t+12]};
            mma_m16n8k16(accA[0], raA, rb4); mma_m16n8k16(accA[1], raA, rb5);
        }
        if (p < np1) {
            const long long tileB = mg1 * nb1 + (p << 1);
            const uint32_t wB0 = W1[tileB * 32 + lane];
            const uint32_t wB1 = W1[tileB * 32 + 32 + lane];
            const uint8_t* sctB = sct1 + tileB * 16;
            const float b0lo = e4m3_f(sctB[g]), b0hi = e4m3_f(sctB[g + 8]);
            const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
            uint32_t raB[4];
            raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
            uint32_t rb2[2]={Xl2[t],Xl2[t+4]}, rb3[2]={Xh2[t],Xh2[t+4]};
            mma_m16n8k16(accB[0], raB, rb2); mma_m16n8k16(accB[1], raB, rb3);
            raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
            uint32_t rb6[2]={Xl2[t+8],Xl2[t+12]}, rb7[2]={Xh2[t+8],Xh2[t+12]};
            mma_m16n8k16(accB[0], raB, rb6); mma_m16n8k16(accB[1], raB, rb7);
        }
    }
    __shared__ float sh[MMA_SMEM];
    // Epilogue per 16-token half: band mapping (sharded GU) or the identity mma_epilogue. gs0/gs1
    // are the per-expert bases (mg0/mg1 already carry the expert offset for the weight reads and
    // must not be re-added — the epilogue indexes gs[mt] itself).
    const float* gse0 = gs0;
    const float* gse1 = gs1;
    if (band0) {
        const float v = mma_warp_reduce(sh, accA);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + msh0;
        if (mw >= 0 && mw < ml0) Cb[m] = f2b(v * gs0[ms0]);
    } else {
        mma_epilogue(sh, accA, Cb, nullptr, gse0, mt, M, 16);
    }
    __syncthreads();                                                    // sh read->next-write hazard
    __nv_bfloat16* Cb1 = Cb + (long long)16 * M;
    if (band1) {
        const float v = mma_warp_reduce(sh, accB);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + msh1;
        if (mw >= 0 && mw < ml1) Cb1[m] = f2b(v * gs1[ms1]);
    } else {
        mma_epilogue(sh, accB, Cb1, nullptr, gse1, mt, M, 16);
    }
}

