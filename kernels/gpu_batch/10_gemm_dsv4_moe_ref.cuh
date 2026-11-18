// gpu_batch/10_gemm_dsv4_moe_ref.cuh: gemm_dsv4_fp8 pair kernels, dequant_fp4_tiled_b, reference MoE expert kernels.
// Correctness-first MoE path alongside the optimized warp kernels.
// Section 10 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ---- gemm_dsv4_fp8_bsb1q — TWO-OP pair, ONE 16-row tile per CTA (Tier-1 item 1.4, RUN 16).
// Same two-independent-ops-sharing-the-activation fusion as gemm_dsv4_fp8_bsb2q (production:
// wq_a + wkv per layer), but with the single-tile bsb body per CTA instead of two tiles.
// Motivation (measured, RUN 16): at the decode pair shapes (M0=1024, M1=512, K=4096) the
// two-tile fused launch fields only 48 CTAs (~1 CTA/SM, 8/64 warps — latency-bound ramp);
// single-tile bsb BEATS bsb2 at these shapes in isolation (wq_a 26.7 vs 36.6 µs; wkv 14.4 vs
// 25.4 µs — the opposite of the big shapes where bsb2 won). This kernel keeps the one-launch
// fusion (one ramp for both ops) at 96 CTAs. PER-ELEMENT CONTRACT IS IDENTICAL TO
// gemm_dsv4_fp8_bsb on each op's weight (same K-partition, same per-block 8 ascending-k mma
// into zero-inited raw, same per-block sa*sb promotion, same fixed ascending-warp epilogue
// reduce, same clamp construction) — the op select only routes tile indices; the gate
// compares against the separate bsb launches bitwise. Host contract: M0,M1 % 128 == 0,
// K % 128 == 0, 1 <= N <= 16; grid = M0/16 + M1/16 CTAs.
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb1q(
    __nv_bfloat16* C0, const uint8_t* __restrict__ Wt0, const uint8_t* __restrict__ Sb0,
    __nv_bfloat16* C1, const uint8_t* __restrict__ Wt1, const uint8_t* __restrict__ Sb1,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    long long m01, long long kn, float* Cf)
{
    const int M0 = (int)(m01 & 0xffffffff);
    const int tiles0 = M0 >> 4;
    const int op = (blockIdx.x >= tiles0) ? 1 : 0;
    const int mt = op ? (int)blockIdx.x - tiles0 : (int)blockIdx.x;
    const int M = op ? (int)(m01 >> 32) : M0;
    __nv_bfloat16* C = op ? C1 : C0;
    const uint8_t* Wt = op ? Wt1 : Wt0;
    const uint8_t* Sb = op ? Sb1 : Sb0;
    const int K = (int)(kn & 0xffffffff), N = (int)(kn >> 32);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns —
    // verbatim from bsb (activation path is unchanged).
    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
            const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
            const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
            ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
            ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
            ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
            uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
            mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

            ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
            ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
            ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
            ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
            uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
            mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
        }

        // Verbatim bsb promotion: sb per (128-row, 128-K) block, sa per token column, pow2.
        const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
        const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
        const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
        acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
        acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
        acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
        acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb_pf — PREFILL weight-stationary variant (R3A.4 P1).
// The serving path used to chunk prefill into <=16-row LAUNCHES (the G2 N<=16 regime), so at
// s=2048 every projection's full weight matrix was re-read from DRAM 128x (nsys: 2.5 s of a
// 21 s prefill). Here the token-chunk loop moves INSIDE the kernel: one launch per
// projection, each CTA's 16-row weight tile is read from DRAM once and stays L2-hot across
// the chunk loop. PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb: for every output
// (row m, token n) the chain is the same — warp w owns 128-K blocks w, w+8, ... ascending;
// per block the same 8 ascending-k mma into a zero-inited raw; promote by sa*sb per block in
// ascending owned-kb order; the same fixed ascending-warp epilogue reduce; the same clamps
// for the partial last chunk. The chunk loop only sequences INDEPENDENT accumulator sets
// (one per chunk) — the gate compares this kernel against the <=16-row decomposition bitwise.
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    // grid (M/16, G): blockIdx.y selects a GROUP of up to PF_G chunks — the chunk loop is
    // split across G CTA groups so chunks run in parallel again (a single CTA serially
    // walking all chunks serialized the latency chain; each weight tile is re-read at most
    // G times, and stays L2-hot inside each group). Per-element chains are untouched: every
    // (row, token) still gets the same ascending-K chain as the <=16-row decomposition.
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + (long long)nc * 16 * K;
        const uint8_t* Sac = Sa + (long long)nc * 16 * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
        const uint8_t* sa0 = Sac + n0 * nkb; const uint8_t* sa1 = Sac + n1 * nkb;
        const uint8_t* sa2 = Sac + n2 * nkb; const uint8_t* sa3 = Sac + n3 * nkb;

        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
                const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
                const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
                ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
                ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
                ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
                uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
                mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

                ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
                ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
                ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
                ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
                uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
                mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
                mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
            }

            // Promote the warp-local raw partial: sb is per (128-row, 128-K) block — constant over
            // this whole tile — and sa is per token column. All pow2, so every product is exact.
            const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
            const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
            const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
            acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
            acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
            acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
            acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;
        }

        mma_epilogue_prescaled(sh, acc, C + (long long)nc * 16 * M, mt, M, N,
                               Cf ? Cf + (long long)nc * 16 * M : nullptr);
        __syncthreads();             // sh reuse barrier between chunks
    }
}

// ---- gemm_dsv4_fp8_bsb_pf4 — PREFILL width variant: FOUR 16-token sub-chunks per pass
// (Tier-2 item 2.2, session 6). pf loops 16-token chunks; at width every (kb, q) weight
// fragment is re-loaded from L2 once per 16-token chunk and each warp's mma stream has
// only 2 independent accumulator chains (raw[0]/raw[1]) — issue-starved: ~4–8% of fp8
// peak in the session-5 decomposition (18.0% of prefill busy). Here the CTA walks
// 64-token GROUPS: the weight fragments for a (kb, q) are loaded ONCE and feed four
// sub-chunks' mma — 4 independent raw chains per warp (the bsb2 ILP trick applied
// across token chunks), quartering the weight L2 request rate per output.
//
// PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb_pf (and hence to the <=16-row
// bsb decomposition): for every output (row m, token n) — warp w owns 128-K blocks
// w, w+8, ... ascending; per block the same 8 ascending-k mma into a zero-inited raw
// (within a q step the c-inner loop issues sub-chunk c's block-2q mma pair then its
// block-2q+1 pair — the same ascending-k sequence bsb issues, with the four
// sub-chunks' INDEPENDENT chains interleaved in the instruction stream, the
// contract-sanctioned ILP argument of bsb2); per-block promotion by sa*sb in ascending
// owned-kb order; the same fixed ascending-warp epilogue reduce per sub-chunk; the same
// clamps for partial sub-chunks. Sub-chunks entirely past S are skipped wholesale —
// their outputs do not exist and no live chain reads their bytes. Gate: bitwise vs the
// <=16-row bsb decomposition at S in {17, 63, 64, 65, 130, 2048} (tests/dsv4_fp8_bsb_test.rs).
//
// Grid: (M/16, G) — blockIdx.y walks 64-token groups (~2 groups = 8 chunks per CTA,
// the same grouping target as pf).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int ngroups = (S + 63) >> 6;
    const int per_g = (ngroups + gridDim.y - 1) / gridDim.y;
    const int ng0 = blockIdx.y * per_g, ng1 = min(ngroups, ng0 + per_g);

    for (int ng = ng0; ng < ng1; ng++) {
        const int base = ng << 6;
        const int nlive = min(4, (S - base + 15) >> 4);   // live 16-token sub-chunks
        // Per-sub-chunk clamps — verbatim pf. Dead sub-chunks clamp onto row 0/1; their
        // loads, mma, promotions and epilogues are all skipped by the c < nlive guards.
        int Nn[4]; long long xr0[4], xr1[4];
        const uint8_t *sa0[4], *sa1[4], *sa2[4], *sa3[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int N = max(1, min(16, S - base - 16 * c));
            const uint8_t* Sac = Sa + (long long)(base + 16 * c) * nkb;
            const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
            const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
            Nn[c] = N;
            xr0[c] = (long long)(g     < N ? g     : N - 1) * K;
            xr1[c] = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
            sa0[c] = Sac + n0 * nkb; sa1[c] = Sac + n1 * nkb;
            sa2[c] = Sac + n2 * nkb; sa3[c] = Sac + n3 * nkb;
        }

        float acc[4][2][4];
        #pragma unroll
        for (int c = 0; c < 4; c++)
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[c][i][j] = 0.f;

        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw[4][2][4];
            #pragma unroll
            for (int c = 0; c < 4; c++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) raw[c][i][j] = 0.f;

            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
                const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
                const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back
                uint32_t ra0[4], ra1[4];
                ra0[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
                ra0[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
                ra0[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
                ra0[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
                ra1[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
                ra1[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
                ra1[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
                ra1[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (c >= nlive) continue;
                    const uint16_t* Xl = reinterpret_cast<const uint16_t*>(
                        X + (long long)(base + 16 * c) * K + xr0[c] + k0);
                    const uint16_t* Xh = reinterpret_cast<const uint16_t*>(
                        X + (long long)(base + 16 * c) * K + xr1[c] + k0);
                    const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                    const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                    const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                    const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                    const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                    const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
                    mma_m16n8k16(raw[c][0], ra0, rb0);      // block 2q,   token columns 0..7
                    mma_m16n8k16(raw[c][1], ra0, rb1);      // block 2q,   token columns 8..15
                    mma_m16n8k16(raw[c][0], ra1, rb2);      // block 2q+1, token columns 0..7
                    mma_m16n8k16(raw[c][1], ra1, rb3);      // block 2q+1, token columns 8..15
                }
            }

            // Promote each sub-chunk's raw partial by its own (sa*sb) — the same two-operand
            // products in the same ascending owned-block order as pf (all pow2, exact).
            const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c >= nlive) continue;
                const float s0 = e8m0_f(sa0[c][kb]) * sb, s1 = e8m0_f(sa1[c][kb]) * sb;
                const float s2 = e8m0_f(sa2[c][kb]) * sb, s3 = e8m0_f(sa3[c][kb]) * sb;
                acc[c][0][0] += raw[c][0][0] * s0;  acc[c][0][1] += raw[c][0][1] * s1;
                acc[c][0][2] += raw[c][0][2] * s0;  acc[c][0][3] += raw[c][0][3] * s1;
                acc[c][1][0] += raw[c][1][0] * s2;  acc[c][1][1] += raw[c][1][1] * s3;
                acc[c][1][2] += raw[c][1][2] * s2;  acc[c][1][3] += raw[c][1][3] * s3;
            }
        }

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (c >= nlive) continue;
            const long long off = (long long)(base + 16 * c) * M;
            mma_epilogue_prescaled(sh, acc[c], C + off, mt, M, Nn[c],
                                   Cf ? Cf + off : nullptr);
            __syncthreads();             // sh reuse barrier between sub-chunks
        }
    }
}

// ---- gemm_dsv4_fp8_bsb_pf2 — PREFILL width variant: TWO 16-row weight tiles per CTA
// (Tier-2 item 2.2, session 6). pf walks 16-token chunks with ONE tile per CTA; at
// width the X fragment loads (8 LDG per (kb, q)) and their e4m3->bf16 conversions feed
// only 4 mma, and every warp's mma stream has only 2 independent accumulator chains.
// Here each CTA computes TWO adjacent 16-row tiles (bsb2's decode trick moved to the
// prefill chunk loop): the X fragments for a (chunk, kb, q) are loaded and converted
// ONCE and feed both tiles' mma — halving the activation LDG + conversion cost per
// mma and giving each warp 4 independent raw chains, at ~pf's register footprint.
//
// PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb_pf (and hence to the <=16-row
// bsb decomposition): for every output (row m, token n) — warp w owns 128-K blocks
// w, w+8, ... ascending; per block the same 8 ascending-k mma into a zero-inited raw
// (the two tiles' INDEPENDENT chains interleave in the instruction stream — the
// contract-sanctioned ILP argument of bsb2); per-block promotion by sa*sb in ascending
// owned-kb order (each tile by its OWN sb — (mt0>>3) vs (mt1>>3) differ across a
// 128-row boundary, exactly as bsb2); the same fixed ascending-warp epilogue reduce
// per tile; the same clamps for the partial last chunk. M % 128 == 0 in production so
// the has1 guard never fires (kept for generality — a lone-tile CTA's chain is
// unchanged). Gate: bitwise vs the <=16-row bsb decomposition (tests/dsv4_fp8_bsb_test.rs).
//
// Grid: ((M+31)/32, G) — blockIdx.y walks 16-token chunk groups (~8 chunks per CTA,
// the same grouping target as pf).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    const int mt0 = 2 * blockIdx.x, mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < M;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + (long long)nc * 16 * K;
        const uint8_t* Sac = Sa + (long long)nc * 16 * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
        const uint8_t* sa0 = Sac + n0 * nkb; const uint8_t* sa1 = Sac + n1 * nkb;
        const uint8_t* sa2 = Sac + n2 * nkb; const uint8_t* sa3 = Sac + n3 * nkb;

        float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
                const uint2 w00 = Wt64[tile0 * 32 + lane];
                const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
                ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
                ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
                ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
                ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
                ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
                ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
                ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

                if (has1) {
                    const long long tile1 = (long long)mt1 * nblk + (kb << 3) + 2 * q;
                    const uint2 w10 = Wt64[tile1 * 32 + lane];
                    const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                    ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                    ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                    ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                    ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                    mma_m16n8k16(raw1[1], ra, rb1);
                    ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                    ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                    ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                    ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                    mma_m16n8k16(raw1[1], ra, rb3);
                }
            }

            // Promote each tile's raw partial by its own (sa*sb) — the same two-operand
            // products in the same ascending owned-block order as pf (all pow2, exact).
            const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
            const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
            {
                const float sb = e8m0_f(Sb[(long long)(mt0 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
                acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
                acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
                acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
            }
            if (has1) {
                const float sb = e8m0_f(Sb[(long long)(mt1 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
                acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
                acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
                acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
            }
        }

        const long long off = (long long)nc * 16 * M;
        float* Cfc = Cf ? Cf + off : nullptr;
        mma_epilogue_prescaled(sh, acc0, C + off, mt0, M, N, Cfc);
        __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
        if (has1) mma_epilogue_prescaled(sh, acc1, C + off, mt1, M, N, Cfc);
        __syncthreads();                 // sh reuse barrier between chunks
    }
}

// ---- dsv4_olo_einsum_fp8_b — item 2.5: wo_a as an fp8 einsum-class per-group BMM.
// out[t, grp*R + c] = Σ_k o[t, grp*K + k] · wo_aq[grp*R + c, k] for t < S — the DeepGEMM
// fp8_einsum "bhr,hdr->bhd" class (n_groups = G local, group = 8 heads × 512, o_lora_rank R),
// with fp8 e4m3 + UE8M0 128-block scales on BOTH operands. The wo_a_q quantizer lays the
// groups as contiguous R-row bands (per-head-group tiles): the group offset enters the
// weight-tile index, the scale index, the X/Sa base pointers, and the output column base.
//
// TOLERANCE-CLASS by contract (item 2.5 / §6-a): reduction order is scheduler-chosen,
// K-split/atomics allowed, no N-invariance requirement — the ONLY bitwise obligations are
// the caller's (quantizers must be deterministic). The per-element instruction chain is the
// bsb-family one (fp8_pair_bf16 → mma m16n8k16 → per-128-block sa·sb promotion in ascending
// owned-kb order → fixed-order cross-warp reduce), so the OUTPUT is a bounded-rel-L2
// approximation of dsv4_olo_proj_tc_b/tc4_b (fp8 inputs vs bf16) — gated, not bitwise.
//
// Grid: ((R/32)*G, GY) — blockIdx.x = (group, 16-row tile pair), blockIdx.y walks 16-token
// chunk groups (pf2's ~8-chunks-per-CTA grouping + the 2-wave fill floor). R % 128 == 0 and
// K % 128 == 0 asserted host-side (production R=1024, K=4096). Two adjacent tiles per CTA
// share each chunk's X fragments (pf2's schedule: halved activation LDG + 4 independent
// accumulator chains per warp, ~80 regs, 3 CTA/SM).
// Epilogue for dsv4_olo_einsum_fp8_b: the bsb fixed-order reduce + fragment map, but the
// output tile's columns are offset by the group's R-row band (col = col_base + mt*16 + ...)
// inside a [S, G*R] row-major buffer (stride = G*R). Prescaled (the sa·sb promotion already
// happened in the k-loop). bf16 RNE round on store, same as mma_epilogue_prescaled.
__device__ __forceinline__ void mma_epilogue_olo(float* sh, float acc[2][4], __nv_bfloat16* C,
                                                 int mt, int col_base, int stride, int N) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order

    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = col_base + mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N) C[(long long)n * stride + m] = f2b(v);
}
extern "C" __global__ __launch_bounds__(256) void dsv4_olo_einsum_fp8_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int R, int K, int S, int G, float* Cf)
{
    const int gx_pairs = R / 32;
    const int grp = blockIdx.x / gx_pairs;
    const int mt0 = 2 * (blockIdx.x % gx_pairs), mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int stride = G * R;
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);
    const long long wt_grp = (long long)grp * (R / 16) * nblk;
    const int sb_grp = grp * (R / 128) * nkb;

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + ((long long)nc * 16 * G + grp) * K;
        const uint8_t* Sac = Sa + ((long long)nc * 16 * G + grp) * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        // NB: the X token ROW stride is G*K (all groups' columns) and the per-token sa row
        // is G*nkb wide (one nkb-scale run per group) — stride both by the FULL row or
        // tokens > 0 read the wrong group's columns/scales (bitwise-broken at N>1).
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * G * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * G * K;
        const uint8_t* sa0 = Sac + n0 * (G * nkb); const uint8_t* sa1 = Sac + n1 * (G * nkb);
        const uint8_t* sa2 = Sac + n2 * (G * nkb); const uint8_t* sa3 = Sac + n3 * (G * nkb);

        float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile0 = wt_grp + (long long)mt0 * nblk + (kb << 3) + 2 * q;
                const uint2 w00 = Wt64[tile0 * 32 + lane];
                const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
                ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
                ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
                ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
                ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
                ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
                ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
                ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

                if (has1) {
                    const long long tile1 = wt_grp + (long long)mt1 * nblk + (kb << 3) + 2 * q;
                    const uint2 w10 = Wt64[tile1 * 32 + lane];
                    const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                    ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                    ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                    ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                    ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                    mma_m16n8k16(raw1[1], ra, rb1);
                    ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                    ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                    ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                    ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                    mma_m16n8k16(raw1[1], ra, rb3);
                }
            }

            // Promote each tile's raw partial by its own (sa*sb) in ascending owned-block
            // order — same two-operand pow2 products as the bsb family (exact).
            const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
            const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
            {
                const float sb = e8m0_f(Sb[sb_grp + (mt0 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
                acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
                acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
                acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
            }
            if (has1) {
                const float sb = e8m0_f(Sb[sb_grp + (mt1 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
                acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
                acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
                acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
            }
        }

        __nv_bfloat16* Cc = C + (long long)nc * 16 * stride;
        mma_epilogue_olo(sh, acc0, Cc, mt0, grp * R, stride, N);
        __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
        if (has1) mma_epilogue_olo(sh, acc1, Cc, mt1, grp * R, stride, N);
        __syncthreads();                 // sh reuse barrier between chunks
    }
}

// ---- The tiled layout is now the ONLY layout a quantized weight is stored in, so the two consumers
// that read weights element-wise (prefill dequant, embedding gather) must invert the permutation.
// This mirrors `fp4_tile_slot` / `fp8_tile_slot` in quant.rs — the Rust unit test proves the map is a
// bijection; these must agree with it or the model is quietly, subtly wrong.
__device__ __forceinline__ float fp4_tiled_at(const uint8_t* Wt, const uint8_t* Sct, const float* gs,
                                              int nblk, int row, int c) {
    const int r = row & 15, cc = c & 15;
    const int lane = (r & 7) * 4 + ((cc & 7) >> 1);
    const int j    = (r >> 3) | ((cc >> 3) << 1);
    const long long tile = (long long)(row >> 4) * nblk + (c >> 4);
    const uint8_t byte = Wt[tile * 128 + lane * 4 + j];
    const uint8_t nib  = (cc & 1) ? (byte >> 4) : (byte & 0x0F);
    return e2m1_f(nib) * e4m3_f(Sct[tile * 16 + r]) * gs[row >> 4];
}
__device__ __forceinline__ float fp8_tiled_at(const uint8_t* Wt, int nblk, int row, int c) {
    const int r = row & 15, cc = c & 15;
    const int lane = (r & 7) * 4 + ((cc & 7) >> 1);
    const int j    = (cc & 1) | ((r >> 3) << 1) | ((cc >> 3) << 2);
    const long long tile = (long long)(row >> 4) * nblk + (c >> 4);
    return e4m3_f(Wt[tile * 256 + lane * 8 + j]);
}

extern "C" __global__ void dequant_fp4_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                               const uint8_t* Sct, const float* gs, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    out[i] = f2b(fp4_tiled_at(Wt, Sct, gs, K >> 4, (int)(i / K), (int)(i % K)));
}
extern "C" __global__ void dequant_fp8_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                               const float* RowScale, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    int row = (int)(i / K);
    out[i] = f2b(fp8_tiled_at(Wt, K >> 4, row, (int)(i % K)) * RowScale[row]);
}
// Qwen fine-grained FP8 prefill dequant: same tiled layout, scale is per-128x128 block
// `BlkScale[(row/128)*nb_k + (col/128)]` (exact — one multiply, no requantization).
extern "C" __global__ void dequant_fp8_blk_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                                   const float* BlkScale, int nb_k, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    int row = (int)(i / K), col = (int)(i % K);
    out[i] = f2b(fp8_tiled_at(Wt, K >> 4, row, col) * BlkScale[(row >> 7) * nb_k + (col >> 7)]);
}
extern "C" __global__ void embed_gather_fp4_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                                    const uint8_t* Sct, const float* gs,
                                                    const int* tokens, int h, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * batch) return;
    out[i] = f2b(fp4_tiled_at(Wt, Sct, gs, h >> 4, tokens[i / h], i % h));
}
extern "C" __global__ void embed_gather_fp8_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                                    const float* RowScale, const int* tokens,
                                                    int h, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * batch) return;
    int row = tokens[i / h];
    out[i] = f2b(fp8_tiled_at(Wt, h >> 4, row, i % h) * RowScale[row]);
}

// ---- STREAM-style read-bandwidth probe. Settles "what is the roofline, actually?".
//
// Two of our own documents disagreed (248 GB/s "measured sustained" vs 216 GB/s observed by the best
// kernel), and that 15% decides whether the mid-size GEMMs have 10% left in them or 25%. It also
// decides whether a competitor's claimed tok/s is physically possible on this part. So: measure it,
// with the simplest possible pure-read kernel — 16-byte vectorized loads, grid-stride, no writes
// except one guarded sink the compiler cannot fold away.
extern "C" __global__ void bw_read_b(float* sink, const uint4* __restrict__ src, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0xFFFFFFFFu) sink[0] = 1.0f;   // never taken; keeps the loads live
}

// ================== SPLIT the fused projections back into their consumers ==================
//
// The fused GEMM writes [M_tot, N] with M contiguous within a column (C[n*M + m]). Its consumers
// (conv1d, the GDN scan, rope, attention) each want their own [m_i, N] buffer. Rather than thread a
// column stride through every one of them -- including the GDN scan, which carries the bitwise
// losslessness guarantee -- scatter once here. It is pure activation traffic: ~200 KB per GDN layer,
// under 0.2% of a decode step, against a GEMM win of 4.7%.

/// GDN: fused [conv_dim + value_dim + nh + nh, N] -> qkv, z, b, a.
// `nh_src` is the number of b/a rows PRESENT in the fused tensor, `nh` the number this rank consumes,
// and `h0` the first head it owns. Under TP=2 GDN sharding the b/a segments stay REPLICATED at full
// width: they are one row per value head (48), and NVFP4 packs output rows in 16-row tiles, so a 48-row
// segment cannot be halved at tile granularity. They are 0.6 % of in_proj, so replicating the bytes is
// free and the slice happens here instead. Unsharded callers pass nh_src == nh and h0 == 0.
extern "C" __global__ void split_gdn_b(__nv_bfloat16* qkv, __nv_bfloat16* z, __nv_bfloat16* bb,
                                       __nv_bfloat16* aa, const __nv_bfloat16* fused,
                                       int conv_dim, int value_dim, int nh, int batch,
                                       int nh_src, int h0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int mtot = conv_dim + value_dim + 2 * nh_src;
    if (i >= mtot * batch) return;
    const int n = i / mtot, r = i - n * mtot;
    const __nv_bfloat16 v = fused[(long long)n * mtot + r];
    if (r < conv_dim)                  { qkv[(long long)n * conv_dim + r] = v; return; }
    if (r < conv_dim + value_dim)      { z[(long long)n * value_dim + (r - conv_dim)] = v; return; }
    // b/a rows: keep only this rank's head range [h0, h0+nh)
    if (r < conv_dim + value_dim + nh_src) {
        const int hsrc = r - conv_dim - value_dim;
        if (hsrc >= h0 && hsrc < h0 + nh) bb[(long long)n * nh + (hsrc - h0)] = v;
    } else {
        const int hsrc = r - conv_dim - value_dim - nh_src;
        if (hsrc >= h0 && hsrc < h0 + nh) aa[(long long)n * nh + (hsrc - h0)] = v;
    }
}

/// Attention: fused [qg_dim + kv_dim + kv_dim, N] -> qg (q|gate, split later), k, v.
extern "C" __global__ void split_qkv_b(__nv_bfloat16* qg, __nv_bfloat16* k, __nv_bfloat16* v,
                                       const __nv_bfloat16* fused, int qg_dim, int kv_dim, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int mtot = qg_dim + 2 * kv_dim;
    if (i >= mtot * batch) return;
    const int n = i / mtot, r = i - n * mtot;
    const __nv_bfloat16 val = fused[(long long)n * mtot + r];
    if (r < qg_dim)                qg[(long long)n * qg_dim + r] = val;
    else if (r < qg_dim + kv_dim)  k[(long long)n * kv_dim + (r - qg_dim)] = val;
    else                           v[(long long)n * kv_dim + (r - qg_dim - kv_dim)] = val;
}

// ===================================================================================================
// MoE (qwen3_5_moe). Correctness-first bf16 path. Router (softmax→top-k→renorm), a per-token grouped
// expert MLP over the STACKED fused weights, and the sigmoid-gated shared expert. Not yet fused-dequant
// or tuned — this is the oracle-correct reference the NVFP4 grouped kernel will be validated against.
// ===================================================================================================

// Router: logits [E, B] col-major (logit[e + b*E]). Per token: softmax(fp32) over E → top-K → RENORM the
// K probs. Emits ids [K, B] (int) and wts [K, B] (float), col-major. One block per token; smem = E floats.
extern "C" __global__ void moe_router_topk_b(int* ids, float* wts, const __nv_bfloat16* logits,
                                             int E, int K, int B) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E]
    for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = __bfloat162float(logits[e + (long)b * E]);
    __syncthreads();
    // WARP 0 does a parallel top-K: 32 lanes scan E/32 experts each, K rounds of warp-argmax+remove.
    // softmax is monotonic → top-K of probs == top-K of logits, and the full-softmax denominator Z
    // CANCELS in the renorm, so only the K winners need exp().
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            float mx = __bfloat162float(logits[ids[b * K] + (long)b * E]);  // top-1 logit = global max
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = __bfloat162float(logits[ids[j + b * K] + (long)b * E]);
                ev[j] = __expf(l - mx); wsum += ev[j];
            }
            float winv = 1.f / wsum; for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv;
        }
    }
}

// Router (hy_v3): logits [E, B] col-major, bias [E] fp32. Per token: score_e = sigmoid(logit_e) in
// fp32; top-K selected by score_e + bias_e (the learned noaux balancing bias — SELECTION ONLY); the
// combine weights are the UN-biased sigmoid scores at the selected indices, renormalized
// (route_norm) and multiplied by `scaling` (router_scaling_factor = 2.826). Mirrors the reference
// (HYV3TopKRouter: routing_weights.gather(top_k_index) / (sum + 1e-20) * router_scaling_factor).
// Emits ids [K, B] (int) and wts [K, B] (float), col-major — the SAME contract as moe_router_topk_b,
// so every downstream expert path (bf16 moe_experts_b, NVFP4 gemm_moe_mma_fp4/grouped) is shared.
// One block per token; smem = E floats.
extern "C" __global__ void moe_router_topk_sigmoid_b(int* ids, float* wts, const __nv_bfloat16* logits,
                                                     const float* bias, int E, int K, int B,
                                                     int route_norm, float scaling) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E] selection scores
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float l = __bfloat162float(logits[e + (long)b * E]);
        s[e] = 1.f / (1.f + __expf(-l)) + bias[e];     // sigmoid + bias, for selection ONLY
    }
    __syncthreads();
    // WARP 0 does a parallel top-K: 32 lanes scan E/32 experts each, K rounds of warp-argmax+remove.
    // Same scan structure as moe_router_topk_b (deterministic tie-break toward the lower stride slot).
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            // Weights: the UN-biased sigmoid scores, recomputed from the logits so the bias and the
            // in-place removals above never touch them. Same sigmoid formula => same bits as s[] had.
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = __bfloat162float(logits[ids[j + b * K] + (long)b * E]);
                ev[j] = 1.f / (1.f + __expf(-l));
                wsum += ev[j];
            }
            if (route_norm) {
                float winv = 1.f / (wsum + 1e-20f);      // the reference's +1e-20 guard
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv * scaling;
            } else {
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * scaling;
            }
        }
    }
}

// FP32-logits twin of moe_router_topk_sigmoid_b. The hy_v3 reference computes router logits in
// FP32 (F.linear(hidden.float(), gate.float())) and selects on them; rounding the logits to bf16
// first perturbs Hy3's tightly-clustered sigmoid scores by ~1e-3 and flips near-tie top-8
// selections (~10-20% of tokens) — an O(1) output change per flipped token. Identical
// selection/weight math otherwise.
extern "C" __global__ void moe_router_topk_sigmoid_f32_b(int* ids, float* wts, const float* logits,
                                                         const float* bias, int E, int K, int B,
                                                         int route_norm, float scaling) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E] selection scores
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float l = logits[e + (long)b * E];
        s[e] = 1.f / (1.f + __expf(-l)) + bias[e];     // sigmoid + bias, for selection ONLY
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = logits[ids[j + b * K] + (long)b * E];
                ev[j] = 1.f / (1.f + __expf(-l));
                wsum += ev[j];
            }
            if (route_norm) {
                float winv = 1.f / (wsum + 1e-20f);
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv * scaling;
            } else {
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * scaling;
            }
        }
    }
}


// Stacked: gate_up [E, 2I, H] (row-major per expert; rows 0..I=gate, I..2I=up), down [E, H, I]. x,out [H,B]
// col-major. One block per token; smem = (2I + I + H) floats. Correctness-first (scalar dots).
extern "C" __global__ void moe_experts_b(__nv_bfloat16* out, const __nv_bfloat16* x,
                                         const int* ids, const float* wts,
                                         const __nv_bfloat16* gate_up, const __nv_bfloat16* down,
                                         int H, int I, int K, int B) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float sm[];
    float* gu  = sm;              // [2I]
    float* hh  = sm + 2 * I;      // [I]
    float* acc = sm + 3 * I;      // [H]
    for (int i = threadIdx.x; i < H; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();
    const __nv_bfloat16* xb = x + (long)b * H;
    for (int j = 0; j < K; j++) {
        int e = ids[j + b * K]; float w = wts[j + b * K];
        const __nv_bfloat16* gW = gate_up + (long)e * (2 * I) * H;   // [2I, H]
        const __nv_bfloat16* dW = down    + (long)e * H * I;         // [H, I]
        for (int r = threadIdx.x; r < 2 * I; r += blockDim.x) {
            float acc0 = 0.f; const __nv_bfloat16* wr = gW + (long)r * H;
            for (int c = 0; c < H; c++) acc0 += __bfloat162float(wr[c]) * __bfloat162float(xb[c]);
            gu[r] = acc0;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < I; r += blockDim.x) {
            float g = gu[r], u = gu[I + r]; hh[r] = (g / (1.f + __expf(-g))) * u;   // silu(gate)*up
        }
        __syncthreads();
        for (int r = threadIdx.x; r < H; r += blockDim.x) {
            float acc0 = 0.f; const __nv_bfloat16* wr = dW + (long)r * I;
            for (int c = 0; c < I; c++) acc0 += __bfloat162float(wr[c]) * hh[c];
            acc[r] += w * acc0;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[i + (long)b * H] = __float2bfloat16(acc[i]);
}

// Shared-expert combine: out[h,b] += sigmoid(gate[b]) * shared[h,b].  gate [1,B], shared/out [H,B].
extern "C" __global__ void moe_shared_combine_b(__nv_bfloat16* out, const __nv_bfloat16* shared,
                                                const __nv_bfloat16* gate, int H, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)H * B) return;
    int b = idx / H;
    float g = __bfloat162float(gate[b]); float sig = 1.f / (1.f + __expf(-g));
    out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + sig * __bfloat162float(shared[idx]));
}

// NVFP4 fused-dequant grouped expert MLP — same structure as moe_experts_b, but the stacked expert
// weights are RAW NVFP4: gate_up_q [E*2I, H/2] nibbles + gate_up_s [E*2I, H/16] E4M3; down_q [E*H, I/2]
// + down_s [E*H, I/16]. Dequant per element: e2m1_f(nib)*e4m3_f(blockscale)*gs (gs = 1/global_scale).
// 4-bit reads = ~4x less weight bandwidth than the bf16 kernel. Uses e2m1_f/e4m3_f defined above.
extern "C" __global__ void moe_experts_fp4_b(
        __nv_bfloat16* out, const __nv_bfloat16* x, const int* ids, const float* wts,
        const uint8_t* gu_q, const uint8_t* gu_s, float gu_gs,
        const uint8_t* dn_q, const uint8_t* dn_s, float dn_gs,
        int hi, int kb) {                                  // packed (H<<16|I), (K<<16|B) — cudarc arity
    const int H = hi >> 16, I = hi & 0xffff, K = kb >> 16, B = kb & 0xffff;
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float sm[];
    float* gu  = sm;              // [2I]
    float* hh  = sm + 2 * I;      // [I]
    float* acc = sm + 3 * I;      // [H]
    const int Hb = H >> 1, Hs = H >> 4, Ib = I >> 1, Is = I >> 4;   // bytes/scales per row
    for (int i = threadIdx.x; i < H; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();
    const __nv_bfloat16* xb = x + (long)b * H;
    for (int j = 0; j < K; j++) {
        int e = ids[j + b * K]; float w = wts[j + b * K];
        for (int r = threadIdx.x; r < 2 * I; r += blockDim.x) {     // gate_up: row e*2I + r, len H
            long grow = (long)e * (2 * I) + r;
            const uint8_t* q = gu_q + grow * Hb;
            const uint8_t* s = gu_s + grow * Hs;
            float acc0 = 0.f;
            for (int c = 0; c < H; c++) {
                uint8_t byte = q[c >> 1];
                uint8_t nib  = (c & 1) ? (byte >> 4) : (byte & 0x0F);
                acc0 += (e2m1_f(nib) * e4m3_f(s[c >> 4]) * gu_gs) * __bfloat162float(xb[c]);
            }
            gu[r] = acc0;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < I; r += blockDim.x) {
            float g = gu[r], u = gu[I + r]; hh[r] = (g / (1.f + __expf(-g))) * u;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < H; r += blockDim.x) {          // down: row e*H + r, len I
            long grow = (long)e * H + r;
            const uint8_t* q = dn_q + grow * Ib;
            const uint8_t* s = dn_s + grow * Is;
            float acc0 = 0.f;
            for (int c = 0; c < I; c++) {
                uint8_t byte = q[c >> 1];
                uint8_t nib  = (c & 1) ? (byte >> 4) : (byte & 0x0F);
                acc0 += (e2m1_f(nib) * e4m3_f(s[c >> 4]) * dn_gs) * hh[c];
            }
            acc[r] += w * acc0;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[i + (long)b * H] = __float2bfloat16(acc[i]);
}

