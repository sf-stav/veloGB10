// gpu_batch/11_moe_fp4_warp_tc.cuh: optimized MoE NVFP4 expert MLP: warp GEMV, tensor-core grouped GEMV, token-gather, burst/x2 variants.
// Same math as moe_experts_fp4_b (validated by the same oracle).
// Section 11 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================================================================================================
// MoE NVFP4 expert MLP — OPTIMIZED (warp-per-output-row GEMV). Fixes the scalar kernel's two problems:
// (1) occupancy — grid is B*K*2I / B*H warps (fills all SMs, vs 1 block/token); (2) coalescing — a warp's
// 32 lanes read consecutive bytes of one weight row. Raw NVFP4 layout: gate_up_q [E*2I, H/2] + [E*2I,H/16]
// E4M3; down_q [E*H, I/2] + [E*H, I/16]. Same math as moe_experts_fp4_b → validated by the same oracle.
// Intermediate gate_up written to a [B*K, 2I] float scratch (silu barrier splits gate_up from down).
// ===================================================================================================

// gate_up: gu_out[(b*K+slot), r] = Σ_c dequant(gate_up[e·2I+r, c]) · x[b, c].  One warp per (b, slot, r).
extern "C" __global__ void moe_gate_up_fp4_warp(
        float* gu_out, const __nv_bfloat16* x, const int* ids,
        const uint8_t* gu_q, const uint8_t* gu_s, float gs, int H, int I, int K, int B) {
    int wid = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (wid >= B * K * (2 * I)) return;
    int r = wid % (2 * I); int t = wid / (2 * I); int slot = t % K; int b = t / K;
    int e = ids[b * K + slot];
    long ROW = (long)e * (2 * I) + r;
    int Hb = H >> 1;
    const uint32_t* q = reinterpret_cast<const uint32_t*>(gu_q + ROW * Hb);   // 4 bytes = 8 nibbles
    const uint8_t* s = gu_s + ROW * (H >> 4);
    const __nv_bfloat16* xb = x + (long)b * H;
    float acc = 0.f;
    for (int u = lane; u < (Hb >> 2); u += 32) {          // coalesced 128 B/warp; 8 K-elements/lane
        uint32_t w = q[u]; int c0 = u << 3;
        float sc = e4m3_f(s[c0 >> 4]) * gs;               // one scale decode per 8 elements (same block)
        #pragma unroll
        for (int n = 0; n < 8; n++)
            acc += e2m1_f((w >> (4 * n)) & 0xF) * sc * __bfloat162float(xb[c0 + n]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) gu_out[(long)(b * K + slot) * (2 * I) + r] = acc;
}

// silu: h[(bk), r] = silu(gu[bk, r]) * gu[bk, I+r].   idx over B*K*I.
extern "C" __global__ void moe_silu_b(float* h_out, const float* gu, int I, int BK) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    float g = gu[(long)bk * 2 * I + r], u = gu[(long)bk * 2 * I + I + r];
    h_out[idx] = (g / (1.f + __expf(-g))) * u;
}

// down: out[b, r] = Σ_slot w · Σ_c dequant(down[e·H+r, c]) · h[(b*K+slot), c].  One warp per (b, r),
// looping the K slots (each output row written once → no atomics).
extern "C" __global__ void moe_down_fp4_warp(
        __nv_bfloat16* out, const float* h, const int* ids, const float* wts,
        const uint8_t* dn_q, const uint8_t* dn_s, float gs, int H, int I, int K, int B) {
    int wid = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (wid >= B * H) return;
    int r = wid % H, b = wid / H;
    int Ib = I >> 1;
    // Accumulate the K experts' weighted partials PER LANE, then reduce ONCE (vs one reduce/expert).
    float lane_acc = 0.f;
    for (int slot = 0; slot < K; slot++) {
        int e = ids[b * K + slot]; float w = wts[b * K + slot];
        long ROW = (long)e * H + r;
        const uint32_t* q = reinterpret_cast<const uint32_t*>(dn_q + ROW * Ib);
        const uint8_t* s = dn_s + ROW * (I >> 4);
        const float* hb = h + (long)(b * K + slot) * I;
        float p = 0.f;
        for (int u = lane; u < (Ib >> 2); u += 32) {
            uint32_t wv = q[u]; int c0 = u << 3;
            float sc = e4m3_f(s[c0 >> 4]) * gs;
            #pragma unroll
            for (int n = 0; n < 8; n++)
                p += e2m1_f((wv >> (4 * n)) & 0xF) * sc * hb[c0 + n];
        }
        lane_acc += w * p;
    }
    for (int o = 16; o > 0; o >>= 1) lane_acc += __shfl_down_sync(0xffffffff, lane_acc, o);
    if (lane == 0) out[(long)b * H + r] = __float2bfloat16(lane_acc);
}

// ===================================================================================================
// TENSOR-CORE grouped MoE GEMV (the perf lever, ~marlin-style). Reuses gemm_mma_fp4_b's tuned inner
// loop + epilogue verbatim, but the weight tile is chosen by the ON-DEVICE routing (expert = ids[bslot])
// and N=1 (one token per slot). fp32 MMA accumulate → quality preserved. Weights are the REPACKED
// (MMA-layout) stacked experts. Used for gate_up (M=2I, K=H, x_by_slot=0 → X row=token b) and down
// (M=H, K=I, x_by_slot=1 → X row=the per-slot h). Output → per-slot bf16 scratch C[bslot, M].
// grid = (M/16 tiles, B*Kslots), block = 256 (8 warps). The stacked experts are one NVFP4 segment, so
// the global scale is uniform → the epilogue's per-tile gs lookup stays correct with the local tile.
// TP=2 expert-parallel: Wt/Sct/gs hold only this rank's expert band, so the router's GLOBAL id is
// rebased by expert_base. A (token,slot) whose expert is remote (local id outside [0,e_span)) gets
// EXPLICIT zero output rows — C is pool scratch (stale, not zeroed) and moe_combine_experts_b sums
// all K slots unconditionally. Owned-expert math is byte-identical to the unsharded path (base=0, span=ne).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = __float2bfloat16(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;                // N=1: every fragment reads this row
    const long long mt_g = (long long)e * (M >> 4) + mt;               // expert e's (local) weight tile

    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sct = Sct + tile * 16;
        const float s0lo = e4m3_f(sct[g]),      s0hi = e4m3_f(sct[g + 8]);
        const float s1lo = e4m3_f(sct[g + 16]), s1hi = e4m3_f(sct[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);   // N=1 → Xh == Xl
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        // acc[1] holds the n=8..15 output columns; for this N=1 GEMV those are never written by the
        // epilogue (n < N=1), so the second MMA per K-step is pure dead compute — dropped. This roughly
        // halves the tensor-core issue in the hot loop (73%→ closer to the fp8 GEMV's 87% roofline).
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, 1);
}

// Combine the K experts' per-slot down outputs: out[b, r] = Σ_slot wts[b*K+slot] · down_s[(b*K+slot), r].
extern "C" __global__ void moe_combine_experts_b(__nv_bfloat16* out, const __nv_bfloat16* down_s,
                                                 const float* wts, int H, int K, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < K; slot++)
        acc += wts[b * K + slot] * __bfloat162float(down_s[(long)(b * K + slot) * H + r]);
    out[idx] = __float2bfloat16(acc);
}

// silu for the MMA path: h[bk, r] = silu(gu[bk, r]) * gu[bk, I+r], gu bf16 interleaved [B*K, 2I].
// BS(b): the grouped arm passes its poff[ne] bound so rows past the real padded total early-exit
// (the GEMM downstream exits on the same device bound); the slot arm passes poff=NULL (all rows
// real there).
extern "C" __global__ void moe_silu_bf16_b(__nv_bfloat16* h_out, const __nv_bfloat16* gu, int I, int BK,
                                           const int* poff, int ne) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    if (poff && bk >= poff[ne]) return;
    float g = __bfloat162float(gu[(long)bk * 2 * I + r]), u = __bfloat162float(gu[(long)bk * 2 * I + I + r]);
    h_out[idx] = __float2bfloat16((g / (1.f + __expf(-g))) * u);
}

// ===================================================================================================
// TOKEN-GATHER grouped MoE (marlin-style) for batch>1 (prefill/verify). The N=1 kernel re-reads each
// expert's weights once PER TOKEN → catastrophic for prefill (~64 tokens/expert → 64× redundant reads).
// Here: counting-sort the (token,slot) pairs by expert, permute activations so an expert's tokens are
// contiguous, then ONE grouped GEMM reads each expert's weight ONCE for all its tokens (N>1). Pairs are
// padded per expert to a multiple of 8 (one MMA n-tile) so a tile never straddles two experts.
// P = batch*k pairs; ids[p] = expert of pair p (p = token*k + slot).

// TP=2: [e_lo, e_hi) is this rank's expert band; remote pairs are not counted, so no padded group is
// ever allocated for a remote expert (whose weight tiles this rank does not hold).
extern "C" __global__ void moe_count_b(int* count, const int* ids, int P, int e_lo, int e_hi) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    int e = ids[p];
    if (e < e_lo || e >= e_hi) return;
    atomicAdd(&count[e], 1);
}
// Padded prefix offsets (single block): poff[e] = start row of expert e; poff[ne] = total padded rows.
// Also seeds cursor[e]=poff[e] for the scatter (avoids a separate copy).
extern "C" __global__ void moe_offsets_b(int* poff, int* cursor, const int* count, int ne) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    // Pad each expert to a multiple of 16 (was 8) so the grouped GEMM's 16-token blocks never straddle
    // two experts (see gemm_moe_grouped_mma_fp4's weight-reuse fix).
    for (int e = 0; e < ne; e++) { poff[e] = acc; cursor[e] = acc; acc += ((count[e] + 15) / 16) * 16; }
    poff[ne] = acc;
}
// Scatter each pair to its expert's contiguous block. cursor[] starts = poff[] (copied host-side).
// perm_tok[pos]=token, perm_wt[pos]=weight, inv_pos[p]=pos (for the no-atomics combine).
// TP=2: a pair whose expert is remote ([e_lo,e_hi) misses) is NOT enqueued — it contributes an exact
// zero. inv_pos[p] = -1 marks it; moe_combine_grouped_b skips -1 explicitly (its down_perm row would
// be indeterminate — no group was allocated for the remote expert).
extern "C" __global__ void moe_scatter_b(int* perm_tok, float* perm_wt, int* inv_pos, int* cursor,
                                         const int* ids, const float* wts, int P, int k,
                                         int e_lo, int e_hi) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    int e = ids[p];
    if (e < e_lo || e >= e_hi) { inv_pos[p] = -1; return; }
    int pos = atomicAdd(&cursor[e], 1);
    perm_tok[pos] = p / k;         // token index
    perm_wt[pos]  = wts[p];
    inv_pos[p]    = pos;
}
// Per-n-tile expert id: tiles [poff[e]/8, poff[e+1]/8) belong to expert e.
extern "C" __global__ void moe_tilemap_b(int* tile_e, const int* poff, int ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= ne) return;
    for (int nt = poff[e] >> 3; nt < poff[e + 1] >> 3; nt++) tile_e[nt] = e;
}
// Gather activations into permuted order: x_perm[pos, :] = x[:, perm_tok[pos]] (0 for padding pos=-1).
extern "C" __global__ void moe_gather_x_b(__nv_bfloat16* x_perm, const __nv_bfloat16* x,
                                          const int* perm_tok, int H, int Ppad,
                                          const int* poff, int ne) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)Ppad * H) return;
    int pos = idx / H, c = idx % H;
    // BS(b) group-bound exit (EXPERT_BATCH_SCALING §5b): rows past the real padded total are never
    // consumed — the grouped GEMM / quant / silu early-exit on the SAME poff[ne] device bound — so
    // skip writing them (they hold stale pool data; nothing reads past poff[ne]). The fold arm
    // passes ne+1 (its GEMMs bound on poff[ne+1] — the routed+shared total).
    if (poff && pos >= poff[ne]) return;
    int t = perm_tok[pos];
    x_perm[idx] = (t >= 0) ? x[(long)t * H + c] : __float2bfloat16(0.f);
}
// Grouped MMA: same tuned loop as gemm_mma_fp4_b, but the weight tile is expert tile_e[nt] and the 8
// columns are this n-tile's 8 permuted tokens. C[bslot-block, M] = C_perm. N=16 (padding tokens masked
// out downstream by inv_pos). X_perm [Ppad, K], C_perm [Ppad, M], both row-major.
// E16 device-only form: the launch grid is a STATIC host-known upper bound (verify: ppad_max/16;
// prefill: (p + e_span*16)/16 groups) — there is NO host readback of the padded total. Each block
// early-exits against poff[ne] read ON DEVICE (both call sites now launch the bound, so the exit
// fires for every 16-token group past the real padded total).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_grouped_mma_fp4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    // WEIGHT-REUSE FIX: each block now covers 16 tokens (two 8-token n-tiles), reading the expert weight
    // ONCE and MMA-ing it into acc[0] (tokens 0-7) AND acc[1] (tokens 8-15). Previously N=8 fed acc[1]
    // clamped/dead data, so the weight was re-read once per 8 tokens; for prefill (~256 tok/expert) that
    // meant ~32× weight re-reads from HBM (~28.8 GB/layer, the 59%-of-prefill bottleneck). At N=16 the
    // re-reads halve. Requires experts padded to a multiple of 16 (moe_offsets_b) so a block never spans
    // two experts; blockIdx.y now indexes 16-token GROUPS (2 of the per-8-tile tile_e entries).
    // TP=2: tile_e holds GLOBAL ids but only OWNED experts emit tiles (remote counts are zero), so
    // e = tile_e - expert_base always lands inside this rank's band — weight reads stay in bounds.
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mt_g = (long long)e * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sct = Sct + tile * 16;
        const float s0lo = e4m3_f(sct[g]),      s0hi = e4m3_f(sct[g + 8]);
        const float s1lo = e4m3_f(sct[g + 16]), s1hi = e4m3_f(sct[g + 24]);
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
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, N);
}
// ===================================================================================================
// u4 burst-load variants (queue #3 probe, 2026-07-31): the fp8 family's winning load structure —
// FOUR pair-iterations' worth of global loads hoisted before the mmas consume them (the
// gemm_dsv4_fp8_bsb1q inner `#pragma unroll` pattern), applied to the fp4 MoE loop. The per-element
// K chains are UNTOUCHED: the warp still owns pairs p = warp + 8k ascending; the unrolled body just
// issues the loads of k, k+1, k+2, k+3 together and executes the mmas in locked ascending order
// (the acc dependency chain serializes them). DSV4 K ∈ {4096, 2048} → npair % 32 == 0, so the
// 4-pair groups are exact (no remainder order change).
extern "C" __global__ __launch_bounds__(256, 3) void gemm_moe_mma_fp4_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = __float2bfloat16(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;
    const long long mt_g = (long long)e * (M >> 4) + mt;

    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p0 = warp; p0 < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2];
        const uint8_t* sct[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + 8 * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sct[u] = Sct + tile * 16;
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sct[u][g]),      s0hi = e4m3_f(sct[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            const int k0 = (p0 + 8 * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
            uint32_t r0[2] = { Xl[t], Xl[t + 4] };
            mma_m16n8k16(acc[0], ra, r0);                              // block 2p,   columns 0..7
            const float s1lo = e4m3_f(sct[u][g + 16]), s1hi = e4m3_f(sct[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { Xl[t + 8], Xl[t + 12] };
            mma_m16n8k16(acc[0], ra, r2);                              // block 2p+1, columns 0..7
        }
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, 1);
}

// ===================================================================================================
// x2-tile streaming variants (queue #3, 2026-07-31): TWO 16-row weight tiles per CTA share the X
// fragments and interleave two INDEPENDENT K-chains in one loop body, doubling the global loads in
// flight per warp at the same occupancy (the pf2-class schedule from the fp8 family, bsb2's decode
// trick). The bind this attacks is the measured one: both single-tile kernels are long-scoreboard
// (global-load-latency) bound at ~62-72% of the 255 GB/s roofline (ncu: stalled_long_scoreboard 8.8-
// 12.6 per issue-active, issue 49-63%, occupancy 94-98%, zero stack/spills).
// BITWISE CONTRACT: each output element's K chain is UNTOUCHED — same pair partition (p = warp + 8k,
// ascending), same per-pair block order (2p then 2p+1), same per-warp fp32 accumulation, same
// warp-order epilogue reduction (mma_epilogue). The x2 kernels differ from the single-tile kernels
// ONLY in which block computes which mt pair and in instruction scheduling, so every existing gate
// (N=1 vs grouped at every N in 2..=16, wide-vs-chunked, serving hashes) remains the contract.
// grid.x = M/32 (was M/16); launch_bounds(256,4) keeps ptxas at <=64 regs (4 CTAs/SM).
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) {
            Cb[mt0 * 16 + threadIdx.x] = __float2bfloat16(0.f);
            Cb[mt1 * 16 + threadIdx.x] = __float2bfloat16(0.f);
        }
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;                // N=1: every fragment reads this row
    const long long mtA = (long long)e * (M >> 4) + mt0;
    const long long mtB = mtA + 1;                                     // adjacent tile of the same expert

    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = tileA + nblk;                          // +1 mt tile, same k-blocks
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
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);   // shared by both tiles
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                                // A block 2p,   columns 0..7
        mma_m16n8k16(accB[0], raB, rb0);                                // B block 2p,   columns 0..7
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                                // A block 2p+1, columns 0..7
        mma_m16n8k16(accB[0], raB, rb2);                                // B block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, accA, Cb, nullptr, gs + (long long)e * (M >> 4), mt0, M, 1);
    __syncthreads();                                                    // sh read->next-write hazard (the dense kernel's documented barrier)
    mma_epilogue(sh, accB, Cb, nullptr, gs + (long long)e * (M >> 4), mt1, M, 1);
}

extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mtA = (long long)e * (M >> 4) + mt0;
    const long long mtB = mtA + 1;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = tileA + nblk;                          // +1 mt tile, same k-blocks
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
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                                // A block 2p,   columns 0..7
        mma_m16n8k16(accA[1], raA, rb1);                                // A block 2p,   columns 8..15
        mma_m16n8k16(accB[0], raB, rb0);                                // B block 2p,   columns 0..7
        mma_m16n8k16(accB[1], raB, rb1);                                // B block 2p,   columns 8..15
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        uint32_t rb3[2] = { Xh[t + 8], Xh[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                                // A block 2p+1, columns 0..7
        mma_m16n8k16(accA[1], raA, rb3);                                // A block 2p+1, columns 8..15
        mma_m16n8k16(accB[0], raB, rb2);                                // B block 2p+1, columns 0..7
        mma_m16n8k16(accB[1], raB, rb3);                                // B block 2p+1, columns 8..15
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, accA, Cb, nullptr, gs + (long long)e * (M >> 4), mt0, M, N);
    __syncthreads();                                                    // sh read->next-write hazard (the dense kernel's documented barrier)
    mma_epilogue(sh, accB, Cb, nullptr, gs + (long long)e * (M >> 4), mt1, M, N);
}

// ---- gemm_moe_grouped_mma_fp4_u4 — the same burst-load structure for the grouped kernel: FOUR
// pair-iterations' loads hoisted (weights, scales, both n-tile X fragments) before the mmas, which
// execute in the locked per-element order (per pair: block 2p cols 0..7, 2p cols 8..15, 2p+1 cols
// 0..7, 2p+1 cols 8..15, ascending pairs). DSV4 K % 1024 == 0 → npair % 32 == 0, groups exact.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mt_g = (long long)e * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p0 = warp; p0 < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2], rb[4][4], rh[4][4];
        const uint8_t* sct[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + 8 * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sct[u] = Sct + tile * 16;
            const int k0 = (p0 + 8 * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
            const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
            rb[u][0] = Xl[t];      rb[u][1] = Xl[t + 4];
            rb[u][2] = Xl[t + 8];  rb[u][3] = Xl[t + 12];
            rh[u][0] = Xh[t];      rh[u][1] = Xh[t + 4];
            rh[u][2] = Xh[t + 8];  rh[u][3] = Xh[t + 12];
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sct[u][g]),      s0hi = e4m3_f(sct[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            uint32_t r0[2] = { rb[u][0], rb[u][1] };
            uint32_t r1[2] = { rh[u][0], rh[u][1] };
            mma_m16n8k16(acc[0], ra, r0);                              // block 2p,   columns 0..7
            mma_m16n8k16(acc[1], ra, r1);                              // block 2p,   columns 8..15
            const float s1lo = e4m3_f(sct[u][g + 16]), s1hi = e4m3_f(sct[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { rb[u][2], rb[u][3] };
            uint32_t r3[2] = { rh[u][2], rh[u][3] };
            mma_m16n8k16(acc[0], ra, r2);                              // block 2p+1, columns 0..7
            mma_m16n8k16(acc[1], ra, r3);                              // block 2p+1, columns 8..15
        }
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, N);
}

// Combine: out[:, t] = Σ_slot perm_wt[inv_pos[t*k+slot]] · down_perm[inv_pos[t*k+slot], :].  No atomics.
// TP=2: inv_pos == -1 marks a remote-expert pair (moe_scatter_b) — skip it; it contributes an exact
// zero (skipping vs adding 0.0f is the same fp32 sum, slot order preserved).
extern "C" __global__ void moe_combine_grouped_b(__nv_bfloat16* out, const __nv_bfloat16* down_perm,
                                                 const float* perm_wt, const int* inv_pos,
                                                 int H, int k, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++) {
        int pos = inv_pos[b * k + slot];
        if (pos < 0) continue;
        acc += perm_wt[pos] * __bfloat162float(down_perm[(long)pos * H + r]);
    }
    out[idx] = __float2bfloat16(acc);
}

// verify_chain_params_b — E13: expand the chain-identity verify inputs ON DEVICE from 3 scalars
// (pos_start, slot, n), so a captured verify graph needs only a tiny params write per replay instead
// of six host->device array uploads + a pipeline sync. Fills exactly the arrays the eager host path
// builds in verify_forward_core_topo (chain identity: KV pos == RoPE pos == pos_start+t; parent[t] =
// t-1 packed with the lane slot; winsrc the width-ck stencil; path[b][d] = d).
extern "C" __global__ void verify_chain_params_b(
    const int* params,   // [pos_start, slot, n, ck]
    int* pos, int* rope, int* slot_ids, int* winsrc, int* parent, unsigned char* path,
    int max_verify) {
    const int pos_start = params[0];
    const int slot = params[1];
    const int n = params[2];
    const int ck = params[3];
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    pos[t] = pos_start + t;
    rope[t] = pos_start + t;
    slot_ids[t] = slot;
    for (int j = 0; j < ck; j++) winsrc[t * ck + j] = t + j - (ck - 1);
    parent[t] = ((slot & 0xFFFF) << 16) | ((t - 1) & 0xFFFF);
    for (int d = 0; d < max_verify; d++) path[t * max_verify + d] = (unsigned char)d;
}

// ---- R3.2 (L2): vocab-parallel LM head under TP=2 — the 8 B maxloc exchange ----
// dsv4_argmax_pair_b: single-block (val, idx) argmax over n fp32 logits, total order (val
// desc, idx asc) — the same order as the host dsv4_argmax (sequential ascending, strict >) and
// the engine's argmax_pass1/2 (halving tree keeps the LOWER index on ties). Each thread strides
// n tracking its local best; a fixed halving tree reduces the 1024 partials. Writes
// pair = {val f32 bits, GLOBAL idx} (idx_base = the rank's vocab-half offset). Grid (1,1,1),
// block (1024,1,1) — one launch per token.
extern "C" __global__ void dsv4_argmax_pair_b(const float* __restrict__ logits, int n,
                                              unsigned int* __restrict__ pair, int idx_base) {
    const int tid = threadIdx.x;
    float bv = -1e30f; unsigned bi = 0xffffffffu;
    for (int i = tid; i < n; i += 1024) {
        float v = logits[i];
        if (v > bv || (v == bv && (unsigned)i < bi)) { bv = v; bi = (unsigned)i; }
    }
    __shared__ float sv[1024];
    __shared__ unsigned si[1024];
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s2 = 512; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            float ov = sv[tid + s2]; unsigned oi = si[tid + s2];
            if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { pair[0] = __float_as_uint(sv[0]); pair[1] = (unsigned)idx_base + si[0]; }
}

// tp_wait_maxloc_b — the K2 of the maxloc barrier: each rank shipped its 8 B (val, idx) local
// winner with K1 (tp_gate_copy_signal, nbytes=8); wait for the peer's, then write the GLOBAL
// winner's index. Total order (val desc, idx asc) — deterministic and identical on both ranks,
// so SPMD lockstep holds without a token broadcast. NaN loses every comparison (matches the
// host argmax's strict-> skip).
extern "C" __global__ void tp_wait_maxloc_b(tp_dev_ctx* c, const unsigned int* local_pair, int* out_idx) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);
    if (tp_spin_until_ge(c, tp_flag(c, TP_F_CPU_DONE), e, 1)) return;   // I5 gate; abort => no-op
    tp_stamp(c, e, TP_GTS_K2_GO);
    if (threadIdx.x == 0) {
        const unsigned int* peer = (const unsigned int*)(c->recv_ring + (size_t)s * c->slot_stride);
        const float lv = __uint_as_float(local_pair[0]);
        const float pv = __uint_as_float(peer[0]);
        const unsigned li = local_pair[1], pi = peer[1];
        const bool local_wins = (lv > pv) || (lv == pv && li < pi);
        out_idx[0] = (int)(local_wins ? li : pi);
    }
}

// v2 twin of tp_wait_maxloc_b: the same two-stage gate as tp_wait_add_g (hint, then the 8 B
// payload's tail at slot+8 with the deadline -> status 11), winner logic byte-identical.
extern "C" __global__ void tp_wait_maxloc_g(tp_dev_ctx* c, const unsigned int* local_pair, int* out_idx) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);
    asm volatile("griddepcontrol.launch_dependents;");
    if (tp_spin_until_ge(c, tp_flag(c, TP_F_PEER_COMMITTED), e, 1)) return;
    const unsigned char* peer = c->recv_ring + (size_t)s * c->slot_stride;
    __shared__ int s_timeout;
    if (threadIdx.x == 0) {
        s_timeout = 0;
        const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
        const unsigned long long* tail =
            (const unsigned long long*)(peer + 8);   // 8 B payload, tail at slot+8
        // maxloc always ships exactly 8 B (never a slot-filling "wide" barrier), so the short
        // stage-B deadline is correct here — no adaptive bound needed (B8 §1.5-3).
        unsigned long long deadline = tp_globaltimer() + TP_TAIL_WAIT_NS;
        unsigned ns = 64, cap = 512u;
        while (tp_ld_relaxed(tail) != e) {
            if (tp_ld_relaxed(ab)) { s_timeout = 1; break; }
            if (tp_globaltimer() >= deadline) {
                tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                printf("[maxloc-stageB-timeout] rank=%d epoch=%llu\n", c->rank, e);
                s_timeout = 1;
                break;
            }
            __nanosleep(ns);
            if (ns < cap) ns <<= 1;
        }
        if (!s_timeout) tp_fence_acquire();
    }
    __syncthreads();
    if (s_timeout) return;
    tp_stamp(c, e, TP_GTS_K2_GO);
    if (threadIdx.x == 0) {
        const unsigned int* p = (const unsigned int*)peer;
        const float lv = __uint_as_float(local_pair[0]);
        const float pv = __uint_as_float(p[0]);
        const unsigned li = local_pair[1], pi = p[1];
        const bool local_wins = (lv > pv) || (lv == pv && li < pi);
        out_idx[0] = (int)(local_wins ? li : pi);
    }
    __syncthreads();
    if (threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);
}

// ===================================================================================================
// E12/E8/E9 re-implementation (2026-08-11) — the fold kernels + the E8 fp32 tail + the fused
// norm+rope WIP kernel. Everything below matches the surviving orchestration (gpu.rs's moe_batch
// fold paths + E8 tail + the rmsnorm_rope_b call sites) exactly — the launch args, smem, grids.
// ===================================================================================================

// ---- fused per-head RMSNorm + RoPE (the lost WIP kernel; replaces the split rmsnorm_b + rope_b
// pair at the full_attn_batch / prefill per-head sites). One block per (seq, head), IN-PLACE on
// `out` (x == out — the call sites pass 11 args: out, w, cos, sin, nh, hd, rdim, B, eps, off,
// row_stride). The norm math is rmsnorm_perhead_b's (v = out[base+tid], tree-reduce s[tid] = v^2,
// inv = rsqrtf(s[0]/hd + eps), out = v*inv*(1+w[tid])), then the rope is rope_b's per-head rotary
// pair on the normed row — byte-identical to running rmsnorm_perhead_b then rope_b on the same row.
// F0: `off`/`row_stride` give the row pitch of the buffer — packed buffers pass (0, nh*hd) /
// (0, nkv*hd), the fused qkv VIEW passes (0, mtot) for q / (qg_dim, mtot) for k — so the SAME
// kernel serves both layouts with identical arithmetic (base = off + b*row_stride + head*hd).
extern "C" __global__ void rmsnorm_rope_b(__nv_bfloat16* out, const float* w,
                                          const float* cos, const float* sin, int nh, int hd,
                                          int rdim, int B, float eps, int off, int row_stride) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * row_stride + off + (long long)head * hd;
    float v = (tid < hd) ? b2f(out[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) out[base + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(out[base + tid]);
        float x2 = b2f(out[base + tid + half]);
        float c = cos[cb], sn = sin[cb];
        out[base + tid] = f2b(x1 * c - x2 * sn);
        out[base + tid + half] = f2b(x2 * c + x1 * sn);
    }
}

// ===================================================================================================
// F4 (EXPERT_FUSION_PASSES §3.4): fold write_kv_b into the k-side rope kernel, with the raw v copy
// riding the same launch. Grid (b·nkv,1,1), block 256 (pad for the v-copy phase), smem hd fp32.
// Phase 1: norm+rope the k row IN PLACE exactly as rmsnorm_rope_b (same v^2 tree — with blockDim
// 256 the threads tid >= hd contribute exact +0.0f leaves, so s[0] and every f2b round are
// bit-identical to a 128-thread launch — same rsqrtf, rope reads the ROUNDED values). Phase 2:
// write the roped k row AND the raw v row to the cache at (slot_ids[b], kvh, pos[b]) with
// write_kv_b's indexing. Bit-exact by construction: norm/rope math untouched, the write is a copy.
// F0: the k/v rows are VIEWS into the fused qkv GEMM output [token, [q|k|v]] — k at
// k_off + b*mtot + head*hd, v at k_off + nkv*hd + b*mtot + head*hd, mtot = k_off + 2*nkv*hd.
// FUSED-VIEW LAYOUT ONLY: the GB10_NO_QKV_VIEW / GB10_NO_FUSED_PERHEAD_ROPE escapes restore the
// un-fused write path (write_kv_b / write_kv_b_q4 on packed buffers). 12-arg launch (packed):
//   stride_nkv = nkv<<19 | stride        (stride = cache positions per slot, 19 bits)
//   hd_rdim    = hd<<16  | rdim
//   k_off      = the q|gate segment width (qg_dim) — derives mtot and the v segment offset.
extern "C" __global__ void rmsnorm_rope_kvwrite_b(
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    __nv_bfloat16* qkv, const float* w,
    const float* cos, const float* sin,
    const int* pos, const int* slot_ids,
    int stride_nkv, int hd_rdim, int k_off, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    const int stride = stride_nkv & 0x7FFFF;
    const int nkv = (unsigned)stride_nkv >> 19;
    const int hd = (unsigned)hd_rdim >> 16;
    const int rdim = hd_rdim & 0xFFFF;
    int blk = blockIdx.x;
    int b = blk / nkv;
    int head = blk % nkv;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const long long row_stride = (long long)k_off + 2LL * nkv * hd;
    long long kbase = (long long)b * row_stride + k_off + (long long)head * hd;
    // Phase 1: norm+rope k in place — rmsnorm_rope_b's math verbatim (tree over hd, exact-zero
    // padding for tid >= hd; the tree level hd/2..1 is identical to a blockDim==hd launch).
    float v = (tid < hd) ? b2f(qkv[kbase + tid]) : 0.0f;
    if (tid < hd) s[tid] = v * v;
    __syncthreads();
    for (int s2 = hd / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) qkv[kbase + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(qkv[kbase + tid]);
        float x2 = b2f(qkv[kbase + tid + half]);
        float c = cos[cb], sn = sin[cb];
        qkv[kbase + tid] = f2b(x1 * c - x2 * sn);
        qkv[kbase + tid + half] = f2b(x2 * c + x1 * sn);
    }
    __syncthreads();   // phase 2 reads the roped row
    // Phase 2: write the roped k row + the RAW v row (write_kv_b indexing: (slot*nkv+h)*stride+pos).
    const long long vbase = kbase + (long long)nkv * hd;
    const int slot = slot_ids[b];
    const long long coff = ((long long)slot * nkv + head) * (long long)stride + pos[b];
    if (tid < hd) {
        k_cache[coff * hd + tid] = qkv[kbase + tid];
        v_cache[coff * hd + tid] = qkv[vbase + tid];
    }
}

// ---- F4 q4 twin: identical phase 1; phase 2 packs the roped k row + the raw v row with
// kvq16_pack into the q4 cache — write_kv_b_q4's quantize-at-write op order preserved byte-for-byte
// (per-16 amax ascending, s = amax/7 e4m3, lrintf clamp ±7, kvq16_pack byte order): the pack is a
// pure function of the 16 input values, and those values are exactly what write_kv_b_q4 reads
// after rmsnorm_rope_b — same inputs, same kvq16_pack => same 9 B per block.
extern "C" __global__ void rmsnorm_rope_kvwrite_q4_b(
    unsigned char* k_cache, unsigned char* v_cache,
    __nv_bfloat16* qkv, const float* w,
    const float* cos, const float* sin,
    const int* pos, const int* slot_ids,
    int stride_nkv, int hd_rdim, int k_off, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    const int stride = stride_nkv & 0x7FFFF;
    const int nkv = (unsigned)stride_nkv >> 19;
    const int hd = (unsigned)hd_rdim >> 16;
    const int rdim = hd_rdim & 0xFFFF;
    int blk = blockIdx.x;
    int b = blk / nkv;
    int head = blk % nkv;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const long long row_stride = (long long)k_off + 2LL * nkv * hd;
    long long kbase = (long long)b * row_stride + k_off + (long long)head * hd;
    float v = (tid < hd) ? b2f(qkv[kbase + tid]) : 0.0f;
    if (tid < hd) s[tid] = v * v;
    __syncthreads();
    for (int s2 = hd / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) qkv[kbase + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(qkv[kbase + tid]);
        float x2 = b2f(qkv[kbase + tid + half]);
        float c = cos[cb], sn = sin[cb];
        qkv[kbase + tid] = f2b(x1 * c - x2 * sn);
        qkv[kbase + tid + half] = f2b(x2 * c + x1 * sn);
    }
    __syncthreads();
    // Phase 2: pack the roped k row + the raw v row (write_kv_b_q4 indexing: crow =
    // (slot*nkv+h)*stride + pos, coff = crow*KVQ_ROW_BYTES(hd) + blk*12).
    const long long vbase = kbase + (long long)nkv * hd;
    const int nb = hd / KVQ_BLK;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + head) * (long long)stride + pos[b];
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd);
    if (tid < nb) {
        kvq16_pack(qkv + kbase + (long long)tid * KVQ_BLK, k_cache + coff + (long long)tid * 12);
        kvq16_pack(qkv + vbase + (long long)tid * KVQ_BLK, v_cache + coff + (long long)tid * 12);
    }
}

// ---- E8 fp32-partial tail: accumulate the rank's bf16 routed partial into the fp32 shared partial
// (exact in fp32 — the routed sum is the bf16 `out`, whose value in fp32 is exact). The fp32 sum
// then crosses the wire and rounds to bf16 exactly ONCE in tp_wait_add mode 1 (tp_all_reduce_fp32).
extern "C" __global__ void add_f32_bf16_b(float* acc, const __nv_bfloat16* routed, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) acc[i] += b2f(routed[i]);
}

// ---- moe_offsets_fold_b: moe_offsets_b + the E12 shared-expert region. poff grows to ne+2:
// poff[ne] = the routed padded total, poff[ne+1] = poff[ne] + pad32(ns) (the shared region holds
// pad32(ns) rows — one per token, slot k of every token, in FIXED token order). Seeds cursor[e] =
// poff[e] for the scatter, like the plain kernel.
extern "C" __global__ void moe_offsets_fold_b(int* poff, int* cursor, const int* count, int ne, int ns) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < ne; e++) { poff[e] = acc; cursor[e] = acc; acc += ((count[e] + 15) / 16) * 16; }
    poff[ne] = acc;
    poff[ne + 1] = acc + ((ns + 31) / 32) * 32;
}

// ---- moe_tilemap_fold_b: moe_tilemap_b over ne+1 blocks — the routed experts' tiles plus the
// shared region's tiles ([poff[ne]>>3, poff[ne+1]>>3), marked with the pseudo-expert id ne — the
// fold GEMMs branch on the ROW range, so the value only distinguishes the region).
extern "C" __global__ void moe_tilemap_fold_b(int* tile_e, const int* poff, int ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x; if (e > ne) return;
    if (e < ne) {
        for (int nt = poff[e] >> 3; nt < poff[e + 1] >> 3; nt++) tile_e[nt] = e;
    } else {
        for (int nt = poff[ne] >> 3; nt < poff[ne + 1] >> 3; nt++) tile_e[nt] = ne;
    }
}

// ---- moe_shared_scatter_b: the shared pairs (slot k of every token) land in the dedicated region
// in FIXED token order — position of token b is poff[ne]+b, so the combine needs no per-pair
// bookkeeping and the fp32 add order is deterministic. perm_wt = sigmoid(sgate[b]) (the old
// moe_shared_combine_b weight) or exactly 1.0 for hy_v3's ungated shared expert.
extern "C" __global__ void moe_shared_scatter_b(int* perm_tok, float* perm_wt, const int* poff,
                                                const __nv_bfloat16* sgate, int ne, int ns, int gated) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= ns) return;
    int pos = poff[ne] + b;
    perm_tok[pos] = b;
    perm_wt[pos] = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
}

// ---- moe_silu_fold_b: the E12/E8 fold-aware silu. Routed rows (bk in the routed band, or
// bk%(k+1) != k in the slot path) are byte-identical to moe_silu_bf16_b. The shared rows write the
// RANK's half of the act row: sk_off = sdesc[3]&0xffffffff (rank·sdn_k), sl = sdesc[3]>>32 (sdn_k —
// the gu half width; the replicated path has sk_off = 0, sl = si, i.e. the full row). The gu reads
// are the rank's bands of the slot's gu rows (the fold GU GEMM wrote them there).
// Slot path: poff = NULL, ne = k -> the shared slot is row % (k+1) == k.
// Grouped path: poff = poff, ne = ne -> the shared region is [poff[ne], poff[ne+1]).
extern "C" __global__ void moe_silu_fold_b(__nv_bfloat16* h_out, const __nv_bfloat16* gu,
                                           int I, int BK, const int* poff, int ne,
                                           const long long* sdesc) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    // E13 fold-race fix: the GU GEMMs write only rows [0, poff[ne+1]) (device early-exit on the
    // fold bound) while the host grid bound BK (ppad_cap/ppad_max) exceeds it — the tail rows
    // held STALE pool memory and this kernel READ them, making the wide-fold prefill
    // nondeterministic run-to-run. Exit at the device-computed real total (slot path: BK is
    // exact, no poff). Rows [poff[ne+1], ceil32) that the GEMM wrote are padding — never read
    // downstream (down GEMM + combine both exit at poff[ne+1]).
    if (poff) {
        const int real = poff[ne + 1];
        // E13: the down GEMM's grid covers rows < ceil32(real) (its tail half reads h_p there) —
        // WRITE ZEROS through that range so the tail reads are deterministic (its outputs are
        // padding, never combined); rows >= ceil32(real) are never read by any kernel.
        const int tail_end = (real + 31) & ~31;
        if (bk >= real) {
            if (bk < tail_end) h_out[idx] = f2b(0.f);
            return;
        }
    }
    const int sk_off = (int)(sdesc[3] & 0xffffffffu);
    const int sl = (int)(sdesc[3] >> 32);
    bool shared;
    if (poff) shared = (bk >= poff[ne] && bk < poff[ne + 1]);
    else      shared = (bk % (ne + 1) == ne);
    int rp = r - sk_off;
    if (!shared || (rp >= 0 && rp < sl)) {
        float g = b2f(gu[(long)bk * 2 * I + r]), u = b2f(gu[(long)bk * 2 * I + I + r]);
        h_out[idx] = f2b(silu_f(g) * u);
    }
}

// ---- moe_combine_experts_fold_b: the slot-path fold combine. Routed slots via wts (slot order
// unchanged — routed slots first), then the shared slot (row (b·(k+1)+k) of down_s) with the
// sigmoid gate weight. fold_shared==1 (wide prefill, E8 shard): the shared slot joins the SAME
// fp32 sum — one rounding (prefill reassociation accepted). fold_shared==0 (the decode/verify
// regime): routed-only, bf16 round here, then the shared add rounds again in
// moe_shared_slot_combine_b post-reduce — the old two-stage rounding, bit-identical to the
// separate-launch path.
extern "C" __global__ void moe_combine_experts_fold_b(__nv_bfloat16* out, const __nv_bfloat16* down_s,
                                                      const float* wts, const __nv_bfloat16* sgate,
                                                      int H, int k, int B, int fold_shared, int gated) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++)
        acc += wts[b * k + slot] * b2f(down_s[(long)(b * (k + 1) + slot) * H + r]);
    if (fold_shared) {
        float w = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
        acc += w * b2f(down_s[(long)(b * (k + 1) + k) * H + r]);
    }
    out[idx] = f2b(acc);
}

// ---- moe_combine_grouped_fold_b: the grouped-path fold combine. Routed slots via inv_pos (skip
// -1 — remote experts under TP, exactly moe_combine_grouped_b), then — fold_shared only — the
// shared slot at poff[ne]+b (fixed token order, perm_wt pre-filled by moe_shared_scatter_b). With
// fold_shared==0 the shared add happens post-reduce in moe_shared_slot_combine_b (two-stage
// rounding); the reduce placement mirrors the old path (per-sub-batch reduces are elementwise and
// token-independent, so sub-batching is combine-invariant).
extern "C" __global__ void moe_combine_grouped_fold_b(__nv_bfloat16* out, const __nv_bfloat16* down_perm,
                                                      const float* perm_wt, const int* inv_pos,
                                                      const int* poff, int H, int k, int B, int ne,
                                                      int fold_shared) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++) {
        int pos = inv_pos[b * k + slot];
        if (pos < 0) continue;
        acc += perm_wt[pos] * b2f(down_perm[(long)pos * H + r]);
    }
    if (fold_shared) {
        int pos = poff[ne] + b;
        acc += perm_wt[pos] * b2f(down_perm[(long)pos * H + r]);
    }
    out[idx] = f2b(acc);
}

// ---- moe_shared_slot_combine_b: the replicated-shared post-reduce add — out[b,r] =
// f2b(b2f(out[b,r]) + w·b2f(down[row(b),r])), the SECOND stage of the two-stage rounding (the
// routed sum was already bf16). mode==0 (slot path): row = b·stride + slot; mode==1 (grouped
// path): row = poff[ne] + b (the fixed-token-order shared region). gated: w = sigmoid(sgate[b]),
// else exactly 1.0.
extern "C" __global__ void moe_shared_slot_combine_b(__nv_bfloat16* out, const __nv_bfloat16* down,
                                                     const int* poff, const __nv_bfloat16* sgate,
                                                     int H, int stride, int slot, int B, int ne,
                                                     int mode, int gated) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    int row = mode ? (poff[ne] + b) : (b * stride + slot);
    float w = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
    out[idx] = f2b(b2f(out[idx]) + w * b2f(down[(long)row * H + r]));
}

