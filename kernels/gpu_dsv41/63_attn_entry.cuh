// =============================================================================
// kernels/gpu_dsv41/63_attn_entry.cuh — band 60 (KA): the entry kernels.
//
//   d41_attn_decode   decode and verify rows (<= 24 rows typical): grid = rows x groups x nsplit_max
//                     CTAs, one split each; the last CTA of a (row, head group) to arrive merges the
//                     partials in split order and runs the epilogue (one launch, no second kernel).
//                     Needs args.part / args.cnt (cnt zeroed once; it resets itself).
//   d41_attn_prefill  prefill rows (any count): grid = rows x groups CTAs, every split of the row in
//                     one CTA, merged in registers in the same order.
// Both are 512 threads, D41_ATTN_SMEM bytes of dynamic shared memory (opt-in above 48 KB on the
// handle that launches). groups = ceil(heads / 16); grid.x = rows * groups (* nsplit_max).
// =============================================================================
#pragma once

// Block idx of the row's block list (nw window blocks, nc compressed blocks, then the extra keys).
__device__ __forceinline__ void d41a_blk(int nw, int nc, int idx, int& kind, int& base) {
    if (idx < nw) { kind = 0; base = 64 * (idx + 2 - nw); }
    else if (idx < nw + nc) { kind = 1; base = 64 * (idx - nw); }
    else { kind = 2; base = 0; }
}

// leg PA1 (D41_ATTN_X_WPF, theirs: `_sparse_attn_part` CCTL.E.PF2 of the split's window block before ACQBULK): the CTA that
// owns split 0 of a (row, head group) (it holds the window blocks: SPLIT_BLOCKS = 2) prefetches the row's 128 window keys
// (the ring rows of positions win_end - 127 .. win_end: 4 x 128 B of E4M3 codes + the 16 E8M0 bytes) into L2. It runs BEFORE
// griddepcontrol.wait, so pos / win_end / lane may be the previous round's values: they only choose addresses inside the
// lane's ring (slot in [0, slots)), and a prefetch moves no data into registers, so nothing computed can change.
__device__ __forceinline__ void d41a_wpf(const d41_attn_args& a) {
    const int slots = a.nsplit_max & 0xFFFF;
    const int G = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slots;
    int bid = blockIdx.x;
    if (bid % G != 0) return;           // CTA 0 of the (row, group): split 0 = the window blocks
    bid /= G;
    const int row = bid / a.groups;
    if (row >= a.rows) return;
    const int pos = __ldg(a.pos + row);
    const int wend = a.win_end ? __ldg(a.win_end + row) : pos;
    int ln = a.lane ? __ldg(a.lane + row) : 0;
    ln = ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln);
    const d41_attn_lane& L = a.lanes[ln];
    if (L.swa_q == nullptr || L.slots <= 0) return;
    const int t = threadIdx.x, k = t >> 2, part = t & 3;     // 128 keys x 4 lines of 128 B
    const long long p = (long long)wend - (D41_ATTN_WIN - 1) + k;
    if (p < 0 || p >= L.lin_base) return;
    const long long slot = (long long)((unsigned)p % (unsigned)L.slots);   // 32-bit, as d41a_fetch
    asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_q + slot * D41_ATTN_D + part * 128));
    if (part == 0 && L.swa_s != nullptr) asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_s + slot * 16));
}

// Blocks [b0, b1) of a row (block b0 already fetched into raw), each next block's keys fetched one block
// ahead. FULL: T <- merge(T, A) after every D41_ATTN_SPLIT_BLOCKS blocks (and after the last); one split
// (decode): T is left untouched and A holds the split.
template <bool FULL>
__device__ __forceinline__ void d41a_splits(unsigned char* sm, const d41a_row& r, int nw, int nc, int b0, int b1, d41a_raw& raw,
                                            float scale, d41a_acc& A, d41a_acc& T, int* err) {
    d41a_acc_reset(A);
    for (int idx = b0; idx < b1; ++idx) {
        const bool more = idx + 1 < b1;
        int kind2 = 0, base2 = 0;
        if (more) d41a_blk(nw, nc, idx + 1, kind2, base2);
        d41a_block(sm, r, raw, more, kind2, base2, scale, A, err);
        if (FULL && ((idx + 1) % D41_ATTN_SPLIT_BLOCKS == 0 || !more)) { d41a_acc_load_ml(sm, A); d41a_merge(T, A); d41a_acc_reset(A); }
    }
}

// Blocks [b0, b1) of a row covering whole splits [s_lo, s_hi) of a split group's NON-PREFIX CTA: each split's state is
// written as its own partial (slot = split index), block b0 already fetched into raw, every next block fetched one block
// ahead, across split boundaries too (PV1).
__device__ __forceinline__ void d41a_split_parts(unsigned char* sm, const d41_attn_args& a, const d41a_row& r, int nw, int nc, int b0, int b1,
                                                 d41a_raw& raw, float* pbase, int* err) {
    d41a_acc A;
    d41a_acc_reset(A);
    for (int idx = b0; idx < b1; ++idx) {
        const bool more = idx + 1 < b1;
        int kind2 = 0, base2 = 0;
        if (more) d41a_blk(nw, nc, idx + 1, kind2, base2);
        d41a_block(sm, r, raw, more, kind2, base2, a.scale, A, err);
        if ((idx + 1) % D41_ATTN_SPLIT_BLOCKS == 0 || !more) {
            d41a_acc_load_ml(sm, A);
            const int split = idx / D41_ATTN_SPLIT_BLOCKS;
            float* pp = pbase + (long long)split * D41_ATTN_PART_FLOATS;
            float4* po = reinterpret_cast<float4*>(pp) + threadIdx.x;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) po[nt * D41_ATTN_THREADS] = make_float4(A.o[nt][0], A.o[nt][1], A.o[nt][2], A.o[nt][3]);
            const int lane = threadIdx.x & 31, r0 = lane >> 2;
            if ((threadIdx.x >> 5) == 0 && (lane & 3) == 0) {
                float2* pm = reinterpret_cast<float2*>(pp + D41_ATTN_THREADS * 16);
                pm[r0] = make_float2(A.m0, A.l0);
                pm[r0 + 8] = make_float2(A.m1, A.l1);
            }
            d41a_acc_reset(A);
        }
    }
}

// The decode / verify entry with SPLIT GROUPS (PV1). `nsplit_max` = slots | (G << 16): `slots` partial slots per (row, head
// group) (>= the row's splits), G CTAs per (row, head group) (0: G = slots, KA's one split per CTA). A row with nS splits
// uses Ge = min(G, nS) CTAs; CTA c < Ge owns splits [c nS / Ge, (c + 1) nS / Ge):
//   CTA 0 (the PREFIX) merges its splits in registers in split order from the reset state (the prefill entry's in-CTA
//     merge), then, when Ge == 1, runs the epilogue (the prefill path exactly); else it writes that merged state as slot 0;
//   CTAs c > 0 write each of their splits' states as its own slot (slot = split index);
//   the last of the Ge CTAs to arrive loads slot 0 and merges slots nS / Ge .. nS - 1 in order, then the epilogue.
// The merges are the decode merger's (KA) in the same order: merge(reset, S0) is the prefix's first step, then S1, S2, ...,
// so the bits equal KA's decode (Ge = nS) and prefill (Ge = 1) entries for any G: the host picks G by the call's shape
// (src/dsv41/gpu/attn.rs `split_group`), never the arithmetic.
template <bool SPLIT>
__device__ __forceinline__ void d41a_run(const d41_attn_args& a) {
    extern __shared__ __align__(16) unsigned char d41a_sm[];
    unsigned char* sm = d41a_sm;
    const int slots = SPLIT ? (a.nsplit_max & 0xFFFF) : 1;
    const int G = SPLIT ? ((a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slots) : 1;
    int bid = blockIdx.x;
    int cta = 0;
    if (SPLIT) { cta = bid % G; bid /= G; }
    const int g = bid % a.groups, row = bid / a.groups;
    if (row >= a.rows) return;
    d41a_row r;
    r.row = row;
    r.pos = __ldg(a.pos + row);
    r.wend = a.win_end ? __ldg(a.win_end + row) : r.pos;
    r.wlo = a.win_lo > 0 ? a.win_lo : 0;
    const int ln = a.lane ? __ldg(a.lane + row) : 0;
    r.L = &a.lanes[ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln)];
    int nsel = 0;
    if (a.sel != nullptr && r.L->kv != nullptr) {
        nsel = a.n_sel ? __ldg(a.n_sel + row) : a.sel_cap;
        nsel = max(0, min(nsel, min(a.sel_cap, D41_ATTN_SEL_MAX)));
    }
    r.nsel = nsel;
    r.sel = a.sel ? a.sel + (long long)row * a.sel_stride : nullptr;
    r.n_ext = r.L->ext_q != nullptr ? max(0, min(r.L->n_ext, D41_ATTN_EXT_MAX)) : 0;
    const int nw = r.wend >= 64 ? 2 : 1;                 // window block 0 has a key only from position 64 on
    const int nc = (nsel + 63) >> 6;
    const int nb = nw + nc + (r.n_ext > 0 ? 1 : 0);
    const int nS = (nb + D41_ATTN_SPLIT_BLOCKS - 1) / D41_ATTN_SPLIT_BLOCKS;
    if (SPLIT && nS > slots) {               // the host sized the partial slots for fewer splits: report, write nothing
        if (cta == 0 && threadIdx.x == 0 && a.err != nullptr) atomicCAS(a.err, 0, D41_ATTN_ERR_ARGS);
        return;
    }
    const int Ge = SPLIT ? min(G, nS) : 1;
    if (cta >= Ge) return;
    const int s_lo = cta * nS / Ge, s_hi = (cta + 1) * nS / Ge;
    // the first block's keys and the queries are fetched together (two independent global latencies)
    const int b0 = s_lo * D41_ATTN_SPLIT_BLOCKS;
    const int b1 = min(nb, s_hi * D41_ATTN_SPLIT_BLOCKS);
    int kind0 = 0, base0 = 0;
    d41a_blk(nw, nc, b0, kind0, base0);
    d41a_raw raw;
    d41a_fetch(r, kind0, base0, raw, a.err);
    if (SPLIT && (a.xflags & D41_ATTN_X_QROPE)) d41a_load_q_rope(sm, a.q, a.q_stride, row, g, a.heads, r.pos, a.rope_freq);   // PA1
    else d41a_load_q(sm, a.q, a.q_stride, row, g, a.heads);   // visible after the first block's barrier
    d41a_sinks(sm, a, g);

    float* pbase = SPLIT ? a.part + (long long)(row * a.groups + g) * slots * D41_ATTN_PART_FLOATS : nullptr;
    d41a_acc T;
    if (cta == 0) {
        d41a_acc A;
        d41a_acc_reset(T);
        d41a_splits<true>(sm, r, nw, nc, b0, b1, raw, a.scale, A, T, a.err);   // the prefix: splits merged in order
    }
    const bool defer = SPLIT && (a.xflags & D41_ATTN_X_DEFER) != 0;   // PA1: d41_attn_merge merges and runs the epilogue
    if (Ge > 1 || defer) {
    if (cta == 0) {
        float4* po = reinterpret_cast<float4*>(pbase) + threadIdx.x;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) po[nt * D41_ATTN_THREADS] = make_float4(T.o[nt][0], T.o[nt][1], T.o[nt][2], T.o[nt][3]);
        const int lane = threadIdx.x & 31, r0 = lane >> 2;
        if ((threadIdx.x >> 5) == 0 && (lane & 3) == 0) {
            float2* pm = reinterpret_cast<float2*>(pbase + D41_ATTN_THREADS * 16);
            pm[r0] = make_float2(T.m0, T.l0);
            pm[r0 + 8] = make_float2(T.m1, T.l1);
        }
    } else {
        d41a_split_parts(sm, a, r, nw, nc, b0, b1, raw, pbase, a.err);
    }
    if (defer) return;                  // the slots are complete at this grid's end (the merge kernel's wait covers them)
    unsigned int* cnt = a.cnt + (row * a.groups + g);
    // publish: every writer fences, the barrier orders the CTA, one thread counts
    __threadfence();
    __syncthreads();
    volatile int* fl = reinterpret_cast<volatile int*>(sm + D41A_SM_FL);
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(cnt, 1u);
        fl[0] = prev == (unsigned)(Ge - 1);
    }
    __syncthreads();
    if (!fl[0]) return;
    __threadfence();
    // the prefix state (slot 0), then slots nS / Ge .. nS - 1 merged in split order; the loads of two slots are issued
    // before their merges (one L2 round trip per two slots; three would spill at the 128-register cap of 512 threads)
    const int r0 = (threadIdx.x & 31) >> 2;
    {
        const float4* q4 = reinterpret_cast<const float4*>(pbase) + threadIdx.x;
        float4 po[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) po[nt] = __ldcg(q4 + nt * D41_ATTN_THREADS);
        const float2* q2 = reinterpret_cast<const float2*>(pbase + D41_ATTN_THREADS * 16);
        const float2 m0 = __ldcg(q2 + r0), m1 = __ldcg(q2 + r0 + 8);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) { T.o[nt][0] = po[nt].x; T.o[nt][1] = po[nt].y; T.o[nt][2] = po[nt].z; T.o[nt][3] = po[nt].w; }
        T.m0 = m0.x; T.l0 = m0.y; T.m1 = m1.x; T.l1 = m1.y;
    }
    for (int s0 = nS / Ge; s0 < nS; s0 += 2) {
        float4 po[2][4];
        float2 pm[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (s0 + j < nS) {
                const float* pp = pbase + (long long)(s0 + j) * D41_ATTN_PART_FLOATS;
                const float4* q4 = reinterpret_cast<const float4*>(pp) + threadIdx.x;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) po[j][nt] = __ldcg(q4 + nt * D41_ATTN_THREADS);
                const float2* q2 = reinterpret_cast<const float2*>(pp + D41_ATTN_THREADS * 16);
                pm[j][0] = __ldcg(q2 + r0);
                pm[j][1] = __ldcg(q2 + r0 + 8);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (s0 + j < nS) {
                d41a_acc S;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) { S.o[nt][0] = po[j][nt].x; S.o[nt][1] = po[j][nt].y; S.o[nt][2] = po[j][nt].z; S.o[nt][3] = po[j][nt].w; }
                S.m0 = pm[j][0].x; S.l0 = pm[j][0].y; S.m1 = pm[j][1].x; S.l1 = pm[j][1].y;
                d41a_merge(T, S);
            }
        }
    }
    if (threadIdx.x == 0) *cnt = 0u;   // ready for the next launch (kernel boundaries order it)
    }                                   // Ge > 1 || defer
    {   // row / head group / position recomputed from the launch (three registers fewer through the block loop: no spill)
        const int Gx = SPLIT ? (((a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : (a.nsplit_max & 0xFFFF))) : 1;
        const int bx = blockIdx.x / Gx, rowx = bx / a.groups, gx = bx - rowx * a.groups;
        d41a_epilogue(sm, a, rowx, gx, __ldg(a.pos + rowx), T);
    }
}

extern "C" __global__ void __launch_bounds__(D41_ATTN_THREADS, 1)
d41_attn_decode(const __grid_constant__ d41_attn_args a) {
    if (a.xflags & D41_ATTN_X_WPF) d41a_wpf(a);   // PA1: the window keys into L2 while the predecessor finishes
    D41_PDL_ENTRY();
    d41a_run<true>(a);
}

extern "C" __global__ void __launch_bounds__(D41_ATTN_THREADS, 1)
d41_attn_prefill(const __grid_constant__ d41_attn_args a) {
    D41_PDL_ENTRY();
    d41a_run<false>(a);
}

// The host mirror check (src/dsv41/gpu/attn.rs): struct sizes / offsets and the band's constants.
extern "C" __global__ void d41_attn_abi(unsigned* out) {
    D41_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    d41_attn_args a;
    const char* b = reinterpret_cast<const char*>(&a);
    out[0] = (unsigned)sizeof(d41_attn_lane);
    out[1] = (unsigned)sizeof(d41_attn_args);
    out[2] = (unsigned)(reinterpret_cast<const char*>(&a.q) - b);
    out[3] = (unsigned)(reinterpret_cast<const char*>(&a.out) - b);
    out[4] = (unsigned)(reinterpret_cast<const char*>(&a.rows) - b);
    out[5] = (unsigned)(reinterpret_cast<const char*>(&a.rope_freq[0]) - b);
    out[6] = D41_ATTN_SMEM;
    out[7] = D41_ATTN_THREADS;
    out[8] = D41_ATTN_PART_FLOATS;
    out[9] = D41_ATTN_MAX_SPLITS;
    out[10] = D41_ATTN_MAX_LANES;
    out[11] = (unsigned)(reinterpret_cast<const char*>(&a.lanes[1]) - b);
    out[12] = D41_ATTN_SPLIT_BLOCKS;
}
