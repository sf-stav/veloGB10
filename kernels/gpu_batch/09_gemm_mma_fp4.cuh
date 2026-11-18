// gpu_batch/09_gemm_mma_fp4.cuh: tensor-core quantized GEMM (gemm_mma_fp4_b family) and split-K reduce.
// The only GEMM the quantized serving path uses; batch-invariant: do not change reduction orders (gate: --probe-binv).
// Section 9 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ================== TENSOR-CORE QUANTIZED GEMM — one fixed shape, flat in N ==================
//
// The ONLY GEMM the quantized serving path uses, at every decode/verify width. The SIMT dequant-GEMV
// it replaced is deleted, not kept as a fallback: it needed a different weight layout (row-major),
// which cannot carry a fused multi-tensor weight, and a second layout is exactly the kind of trap
// that has bitten this project repeatedly. The problem it solves, measured on 9B:
//
//     9B         decode (N=1)   verify (N=4)
//     bf16        73.3 ms        83.9 ms      <- cuBLAS at N=4: tensor cores, flat in N
//     NVFP4       31.6 ms        84.4 ms      <- our SIMT GEMV: 2.7x, and lands on bf16's time
//
// Quantization bought NOTHING at the width speculation actually needs. Per 16-element K-block a SIMT
// thread pays ~16 weight-decode ops (constant in N) + N*16 FMAs (linear in N). At N=1 the kernel sat
// at 83% of the bandwidth roofline — i.e. with almost no compute headroom — so the linear-in-N FMA
// term tipped it compute-bound almost immediately. Landing exactly on bf16's tensor-core time is what
// "the SIMT FMA pipe became the roofline" looks like. The fix is not a better GEMV. It is to move the
// FMAs off the SIMT pipe entirely.
//
// THE DESIGN (Marlin's, in essence):
//
//   * Weights stay PACKED in VRAM and are permuted offline into mma-fragment order (quant.rs).
//     One contiguous aligned load per lane per k-step; a warp's tile is one contiguous byte run.
//   * Dequantize in REGISTERS to bf16 fragments. This cost is per-WEIGHT, so it is constant in N.
//     It is the term that must not scale, and it doesn't.
//   * Feed `mma.sync.m16n8k16`. The N*K products now cost ~nothing: one HMMA covers 8 columns of N.
//     The kernel goes back to being bound by the packed-weight bytes — flat in N until activation
//     traffic matters, which at N<=16 it does not.
//
// WHY THIS IS STILL BITWISE BATCH-INVARIANT (the guarantee greedy-MTP losslessness rests on):
//
// The instinct is that tensor cores must cost us determinism, because cuBLAS does. But cuBLAS breaks
// invariance by SELECTING DIFFERENT KERNELS AND TILINGS PER SHAPE, not because `mma` is inherently
// unstable — an mma instruction is a deterministic function of its inputs. So:
//
//   1. Inside one mma: hardware reduction order is fixed for a given instruction shape, and the 8
//      output columns are INDEPENDENT dot products. Columns 1..15 cannot perturb column 0.
//   2. Across k-slices: the K loop is a fixed stride with no dependence on N.
//   3. Across the 8 warps' split-K partials: summed in warp-index order in shared memory. Fixed.
//      (Not atomicAdd — note vLLM's Marlin path sets VLLM_MARLIN_USE_ATOMIC_ADD=1, i.e. the
//      competition's reduction order is scheduler-dependent. That is a choice, not a necessity.)
//
// Then the move that makes invariance TRIVIAL rather than merely argued: N IS ALWAYS PADDED TO 16.
// Decode (N=1) and verify (N=2..16) execute the identical instruction sequence; the padded columns
// are separate accumulators that are computed and thrown away. Column 0 is bit-identical at every N
// BY CONSTRUCTION. The padding is free where it counts — the kernel is bound by packed-weight bytes,
// which are the same at N=1 and N=16, and the wasted HMMA slots ride a pipe the GEMV left idle.
//
// This also collapses the engineering surface: ONE kernel, no N-dispatch boundary. The old
// bf16-falls-to-cuBLAS-above-N=2 split was both a perf cliff and, historically, how invariance broke.

#define MMA_NW 8                       // warps per block; they split K and reduce in fixed order
#define MMA_SMEM (MMA_NW * 32 * 8)     // [8 acc slots][8 warps][32 lanes] f32

// D[16x8] += A[16x16] * B[16x8], bf16 in, f32 accumulate. A row-major, B col-major (= our X layout:
// X[n][k] with k contiguous IS "col" for B, so no transpose anywhere).
__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two E2M1 nibbles -> a bf16x2 register, each PRE-SCALED by the block's E4M3 scale.
//
// Folding the scale in here is exact, and that is not a lucky accident: E2M1 magnitudes carry 1
// mantissa bit, E4M3 carries 3, so the product needs at most 5 — and bf16 has 7. Every scaled weight
// is therefore representable with ZERO rounding, and the mma's f32 accumulate is at least as accurate
// as the SIMT kernel's. The f32 TENSOR scale (`inv_gs`, arbitrary bits) is NOT folded here — it is
// constant over the whole tensor and is applied once to the f32 accumulator at the end.
__device__ __forceinline__ uint32_t fp4_pair_bf16(uint32_t byte, float s) {
    __nv_bfloat162 v = __floats2bfloat162_rn(e2m1_f(byte & 0x0F) * s, e2m1_f((byte >> 4) & 0x0F) * s);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t fp8_pair_bf16(uint32_t lo, uint32_t hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(e4m3_f((uint8_t)lo), e4m3_f((uint8_t)hi));
    return *reinterpret_cast<uint32_t*>(&v);
}

// Merge the 8 warps' fragment accumulators in FIXED warp order (shared memory, no atomics) and
// return this thread's element of the 16x16 tile: thread tid owns (rlane, rslot) = (tid&31, tid>>5).
// This is the whole cross-warp reduction of both the plain epilogue and the split-K partial; the
// caller owns the (m, n) fragment map and whatever it does with the value.
__device__ __forceinline__ float mma_warp_reduce(float* sh, float acc[2][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    // Re-slice the block: 256 threads, 256 (lane, acc-slot) pairs, one each. Keeping `lane` in the
    // low bits keeps both the store above and the load below bank-conflict-free.
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order
    return v;
}

// Reduce the 8 warps' fragment accumulators in fixed warp order, scale, and scatter to C[n][m].
// `rs` is the FP8 per-row scale (nullptr for NVFP4, where `gs` carries the tensor scale instead).
// `Cf` (optional): write the FP32 accumulator instead of rounding to bf16. This is the whole of the
// FP32-preserving TP=2 reduction — the row-parallel partial leaves the GEMM UNROUNDED, crosses the wire
// in FP32, is summed in FP32 on both ranks, and is rounded to bf16 exactly ONCE in tp_wait_add. The only
// remaining difference from a single-node full-K accumulation is FP32 addition association, which is
// reassociation-class, not a precision loss.
__device__ __forceinline__ void mma_epilogue(float* sh, float acc[2][4], __nv_bfloat16* C,
                                             const float* rs, const float* gs, int mt, int M, int N,
                                             float* Cf = nullptr) {
    const float v = mma_warp_reduce(sh, acc);

    // Invert the mma C-fragment map: c_i is row (g + 8*(i>=2)), col (2t + (i&1)) of the 16x8 subtile.
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    // FUSED weights hold several source tensors stacked along M, each with its own NVFP4 tensor
    // scale. Every segment boundary is 16-aligned, so a tile lies wholly inside one segment and the
    // scale is a per-TILE lookup — read once per block, no requantization, no precision loss.
    if (n < N && m < M) {
        const float o = v * (rs ? rs[m] : gs[mt]);
        if (Cf) Cf[(long long)n * M + m] = o;          // FP32-preserving partial (no round here)
        else    C[(long long)n * M + m] = f2b(o);
    }
}

// mma_epilogue with NO output-side scale: the DSV4 block-scale kernel promotes sa·sb into the
// accumulator inside the k-loop, so the fixed-order cross-warp reduce is written out as-is.
__device__ __forceinline__ void mma_epilogue_prescaled(float* sh, float acc[2][4], __nv_bfloat16* C,
                                                       int mt, int M, int N, float* Cf = nullptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order

    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N && m < M) {
        if (Cf) Cf[(long long)n * M + m] = v;
        else    C[(long long)n * M + m] = f2b(v);
    }
}

// One warp's pass over k-block PAIRS p in [p_lo, p_hi) stepping MMA_NW, accumulating into acc —
// the exact body of gemm_mma_fp4_b's k-loop, shared verbatim with the split-K partial kernel
// (gemm_mma_fp4_splitk_b) so the two can never drift. xr0/xr1 are this lane's two X rows (already
// N-clamped by the caller); the rest of the geometry is the caller's.
//
// A warp takes an ADJACENT PAIR of k-blocks per iteration. The pairing is not for unrolling —
// it is to fix a DRAM sector waste that ncu found and that nothing else could see.
//
// The scale array holds 16 E4M3 bytes per 16x16 tile. Reading them the obvious way,
//
//     const uint8_t* sct = Sct + tile*16;   s_lo = sct[g];   s_hi = sct[g+8];   // g = lane>>2
//
// makes 32 lanes ask for only 16 DISTINCT BYTES -- half of a 32-byte DRAM sector. The weight load
// sitting next to it is perfect (128 contiguous bytes, 4 full sectors), but every scale fetch threw
// away half its sector. Measured: **18.7 of 32 bytes utilized** across the 64.6% of sectors that
// miss L2, on a kernel that ncu confirms is latency-starved (14 active warps per scheduler but only
// 0.82 ELIGIBLE -- they are all parked on `long_scoreboard`, waiting for memory).
//
// Consecutive tiles' scales are contiguous, so a warp that takes tiles 2p and 2p+1 reads
// Sct[..2p*16 .. 2p*16+32) -- 32 contiguous bytes, ONE FULL SECTOR, 32/32 utilized.
//
// NB: an earlier attempt at this failed and taught the lesson. It paired kb with kb+MMA_NW (a
// STRIDED pair), which gives the same instruction-level parallelism but leaves the two scale reads
// on DIFFERENT sectors -- still 50% each. It measured no faster. Adjacency is the entire point;
// "unroll by 2" is not.
//
// The k-visit order changes (2w, 2w+1, 2w+16, ... instead of w, w+8, ...) but it is still FIXED and
// N-INDEPENDENT, so column 0 is bit-identical at every N and batch-invariance is untouched. Gate:
// --probe-binv. K % 32 == 0 for every tensor in this family (asserted host-side).
__device__ __forceinline__ void mma_fp4_pairs(float acc[2][4], int p_lo, int p_hi,
                                              const uint32_t* Wt32, const uint8_t* Sct,
                                              const __nv_bfloat16* X, long long xr0, long long xr1,
                                              int mt, int nblk, int g, int t) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // S9F L1 (the ~15 ms verify lever): cp.async double-buffering of the weight pair + the scale
    // sector, so the k-loop overlaps the (L2/DRAM) load latency with the mma stream — ncu showed
    // the M=8 kernel latency-starved (0.82 eligible warps parked on `long_scoreboard`). Each lane
    // stages its OWN 3 u32s (wq0, wq1, and one 4-B scale word) into per-lane smem slots; the scale
    // bytes come back via __shfl (the lane holding the word extracts its byte). The k-visit order
    // and EVERY arithmetic op are unchanged — only the memory path differs (async global->smem
    // instead of a synchronous global read), so column 0 stays bit-identical at every N and
    // batch-invariance is untouched. Gate: --probe-binv. The adjacent-pair sector trick (2p, 2p+1)
    // and the fixed k-visit order are preserved verbatim.
    //
    // Layout: wstage[2][MMA_NW*32*3] — [buffer][warp*32+lane][wq0 | wq1 | scale-word]. Each lane's
    // scale word is the 4-B ALIGNED fetch Sct[tile*16 + (lane&~3)] (cp.async needs naturally
    // aligned sources); the warp's 8 distinct words cover the whole 32-B scale sector (one
    // DRAM sector, 32/32 utilized — the scale sector is never split, exactly like the direct-load
    // pairing). Byte g of the sector lives in lane (g>>2)*4's word at byte (g&3). A/B measured:
    // the full staging (weights + scales) beat both the reference and a weights-only variant
    // (verify 96.2 vs 98.4/98.9 ms) — the scale path replaced 4 strided global byte-loads per
    // lane with 1 cp.async + 4 shuffles, fewer memory instructions on the dequant chain.
    __shared__ uint32_t wstage[2][MMA_NW * 32 * 3];
    const int sid = (warp * 32 + lane) * 3;

    auto stage = [&](int p, int buf) {
        const long long tile = (long long)mt * nblk + (p << 1);
        const uint32_t* wsrc = Wt32 + tile * 32;
        const uint8_t* ssrc = Sct + tile * 16;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 0])), "l"(wsrc + lane));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 1])), "l"(wsrc + 32 + lane));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 2])),
                        "l"(ssrc + (lane & ~3)));
        asm volatile("cp.async.commit_group;");
    };

    // Preload the first pair, then pipeline: issue pair p+MMA_NW while computing pair p. wait_group
    // 1 after each issue leaves exactly the just-issued prefetch outstanding — the current pair's
    // group (committed earlier) has completed. The per-thread staging needs NO barrier: every lane
    // reads only its own slots (cp.async completion is per-thread).
    stage(p_lo, 0);
    int i = 0;
    for (int p = p_lo; p < p_hi; p += MMA_NW, i++) {
        const int cur = i & 1;
        const bool more = (p + MMA_NW) < p_hi;
        if (more) { stage(p + MMA_NW, cur ^ 1); }
        if (more) { asm volatile("cp.async.wait_group 1;"); }
        else      { asm volatile("cp.async.wait_group 0;"); }

        const uint32_t wq0 = wstage[cur][sid + 0];              // this lane's 4 B of tile 2p's weight row
        const uint32_t wq1 = wstage[cur][sid + 1];              // tile 2p+1's
        const uint32_t sw  = wstage[cur][sid + 2];              // this lane's 4-B scale-sector word
        // The scale bytes via shuffles (same bytes the direct-load path read — bit-identical).
        // Byte b of the sector sits in lane (b>>2)*4's word at byte (b&3).
        const uint32_t wb = __shfl_sync(0xffffffffu, sw, (g >> 2) * 4);
        const uint32_t wb8 = __shfl_sync(0xffffffffu, sw, ((g + 8) >> 2) * 4);
        const uint32_t wb16 = __shfl_sync(0xffffffffu, sw, ((g + 16) >> 2) * 4);
        const uint32_t wb24 = __shfl_sync(0xffffffffu, sw, ((g + 24) >> 2) * 4);
        const float s0lo = e4m3_f((uint8_t)((wb   >> ((g & 3) * 8)) & 0xFF));    // sct[g]
        const float s0hi = e4m3_f((uint8_t)((wb8  >> (((g + 8) & 3) * 8)) & 0xFF));   // sct[g+8]
        const float s1lo = e4m3_f((uint8_t)((wb16 >> (((g + 16) & 3) * 8)) & 0xFF));  // sct[g+16]
        const float s1hi = e4m3_f((uint8_t)((wb24 >> (((g + 24) & 3) * 8)) & 0xFF));  // sct[g+24]

        const int k0 = (p << 5);                                  // 2 k-blocks = 32 elements of K
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);

        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);        // block 2p,   columns 0..7
        mma_m16n8k16(acc[1], ra, rb1);        // block 2p,   columns 8..15

        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        uint32_t rb3[2] = { Xh[t + 8], Xh[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);        // block 2p+1, columns 0..7
        mma_m16n8k16(acc[1], ra, rb3);        // block 2p+1, columns 8..15
    }
}

// ---- NVFP4. One mma k-step consumes exactly one 16-element scale block, so the block scale is
// constant over the step and folds into the A-fragment for free. That alignment is the whole trick.
// Cf != nullptr => write the FP32 accumulator to Cf and leave C untouched (TP=2 FP32-preserving path).
//
// E9: the LAST argument is the PDL (programmatic dependent launch) flag. pdl==1 means this grid was
// launched as the programmatic-stream-serialization SECONDARY of a TP barrier chain: it publishes
// its own launch-completion edge, streams this block's weight bytes into L2 (read-only — no
// numerics change), and only then griddepcontrol.wait gates the (barrier-reduced) activation. pdl==0
// (the plain path, and every non-E9 caller) skips all three — byte-identical to the pre-E9 kernel.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp4_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, int M, int K, int N,
    float* Cf, int pdl)
{
    const int ntm = M >> 4, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    if (pdl) {
        // E9 prologue: publish, prefetch this block's tiles' weights, then wait. The prefetch
        // issues exactly the weight/scale loads the k-loop below will perform (same tile range,
        // same lanes), so the compute hits L2; the loads are kept alive by the sink checksum and
        // the never-true store (a store under a false condition cannot affect C).
        asm volatile("griddepcontrol.launch_dependents;");
        const uint32_t* Wt32p = reinterpret_cast<const uint32_t*>(Wt);
        const int npairp = nblk >> 1;
        unsigned sink = 0u;
        for (int mtp = blockIdx.x; mtp < ntm; mtp += gridDim.x) {
            for (int p = warp; p < npairp; p += MMA_NW) {
                const long long tile = (long long)mtp * nblk + (p << 1);
                sink ^= Wt32p[tile * 32 + lane];
                sink ^= Wt32p[tile * 32 + 32 + lane];
                sink ^= (unsigned)Sct[tile * 16 + g] << 8;
                sink ^= (unsigned)Sct[tile * 16 + g + 8] << 16;
            }
        }
        if (sink == 0xDEADBEEFu) C[threadIdx.x] = f2b(0.f);   // never true — keeps the loads alive
        asm volatile("griddepcontrol.wait;");
    }

    // The two X rows this lane's B-fragments read. Columns >= N are padding: clamp them onto a valid
    // row so the load stays in bounds. They feed independent accumulators that are never written, so
    // the garbage cannot reach column 0 — see the invariance argument above.
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;

    __shared__ float sh[MMA_SMEM];
    // Persistent output tiles: the grid is capped at the resident-block capacity (48 SMs x the 6
    // blocks/SM __launch_bounds__ asserts) and block b computes tiles b, b+gridDim.x, ... . Every
    // tile's arithmetic is self-contained — the k-loop visits k-blocks in the same fixed order and
    // mma_epilogue reduces the warps in the same fixed order — so WHICH block runs a tile, and in
    // what order tiles are consumed, cannot change any tile's result. The tile->block map is a
    // function of the weight shape (ntm, gridDim.x) only, never of N: column 0 stays bit-identical
    // at every N and batch-invariance is untouched. Gate: --probe-binv.
    for (int mt = blockIdx.x; mt < ntm; mt += gridDim.x) {
        // `sh` is reused across tiles: mma_epilogue syncs write->read but NOT read->next-write, so
        // barrier here or a fast warp could overwrite sh while a slow one still reads the last tile.
        __syncthreads();
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        mma_fp4_pairs(acc, warp, npair, Wt32, Sct, X, xr0, xr1, mt, nblk, g, t);

        mma_epilogue(sh, acc, C, nullptr, gs, mt, M, N, Cf);
    }
}

// ---- SPLIT-K VARIANT for the shapes the persistent grid cannot fill: long-K, small-M (the FFN
// down-proj, M=5120 K=17408). There ntm=320 tiles is 1.11 waves of the 288 resident blocks AND each
// warp sits on a ~136-block serial k-chain, so the machine is both underfilled and latency-starved.
// Split the npair k-block-pair range into `nsplit` CONTIGUOUS chunks: work item (mt, s) accumulates
// chunk s of tile mt exactly as gemm_mma_fp4_b would, warp-reduces in the same fixed order, and
// writes the UNSCALED fp32 partial to P[(mt*nsplit+s)*256 + tid]. gemm_mma_splitk_reduce_b then
// sums the nsplit partials in FIXED s order and applies the epilogue.
//
// BATCH-INVARIANCE: nsplit and the chunk boundaries are functions of the weight shape (K, and M via
// the host-side tile cap) ONLY — never of N — and every reduction is fixed-order (no atomics, ever;
// see the Marlin note above). A decode (N=1) and a verify (N<=16) see identical split geometry and
// identical per-column arithmetic, so column 0 is bit-identical at every N. Gate: --probe-binv;
// force the path on small models with GB10_GEMM_SPLITK=<S>.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp4_splitk_b(
    float* P, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const __nv_bfloat16* __restrict__ X, int M, int K, int N, int nsplit)
{
    const int ntm = M >> 4, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    const int chunk = (npair + nsplit - 1) / nsplit;   // pair-range per split; K-only

    __shared__ float sh[MMA_SMEM];
    for (int work = blockIdx.x; work < ntm * nsplit; work += gridDim.x) {
        // `sh` is reused across work items — barrier here, same reason as the unsplit tile loop.
        __syncthreads();
        const int mt = work / nsplit, s = work - mt * nsplit;
        const int p0 = s * chunk;
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        mma_fp4_pairs(acc, p0 + warp, min(p0 + chunk, npair), Wt32, Sct, X, xr0, xr1, mt, nblk, g, t);

        P[(long long)work * 256 + threadIdx.x] = mma_warp_reduce(sh, acc);   // UNSCALED fp32 partial
    }
}

// The second pass: one block per output tile, 256 threads = the 256 (m, n) fragment slots. Sum the
// nsplit partials in FIXED s order, then the usual epilogue — per-tile gs scale, bf16 store, or the
// unrounded fp32 store to Cf for the FP32-preserving TP=2 path (exactly mma_epilogue's semantics).
extern "C" __global__ void gemm_mma_splitk_reduce_b(
    __nv_bfloat16* C, const float* __restrict__ P, const float* __restrict__ gs,
    int M, int N, int nsplit, float* Cf)
{
    const int mt = blockIdx.x;
    const long long base = (long long)mt * nsplit * 256 + threadIdx.x;
    float v = 0.0f;
    for (int s = 0; s < nsplit; s++) v += P[base + (long long)s * 256];   // FIXED order s=0..S-1

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N && m < M) {
        const float o = v * gs[mt];
        if (Cf) Cf[(long long)n * M + m] = o;          // FP32-preserving partial (no round here)
        else    C[(long long)n * M + m] = f2b(o);
    }
}

// ---- FP8 E4M3. Scales are per output ROW, constant over K, so nothing folds into the fragment and
// the row scale is applied once to the f32 accumulator in the epilogue.
extern "C" __global__ __launch_bounds__(256) void gemm_mma_fp8_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const float* __restrict__ RowScale,
    const __nv_bfloat16* __restrict__ X, int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nblk; kb += MMA_NW) {
        const long long tile = (long long)mt * nblk + kb;
        const uint2 w8 = Wt64[tile * 32 + lane];            // ONE 8-byte load = the whole A-fragment

        uint32_t ra[4];
        ra[0] = fp8_pair_bf16(w8.x,       w8.x >>  8);      // row g,   cols 2t, 2t+1
        ra[1] = fp8_pair_bf16(w8.x >> 16, w8.x >> 24);      // row g+8, cols 2t, 2t+1
        ra[2] = fp8_pair_bf16(w8.y,       w8.y >>  8);      // row g,   cols 2t+8, 2t+9
        ra[3] = fp8_pair_bf16(w8.y >> 16, w8.y >> 24);      // row g+8, cols 2t+8, 2t+9

        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + (kb << 4));
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + (kb << 4));
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };

        mma_m16n8k16(acc[0], ra, rb0);
        mma_m16n8k16(acc[1], ra, rb1);
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, C, RowScale, nullptr, mt, M, N, Cf);
}

// ================== QWEN FINE-GRAINED FP8 (block-128) DECODE GEMV ==================
//
// C[n,m] = Σ over K/128 blocks kbb of ( Σ over that 128-K span of code·x ) · scale_inv[m/128, kbb]
//
// Qwen3.8-27B-FP8 layout (quant_method "fp8", weight_block_size [128,128]): weights are RAW
// row-major E4M3 (MMA-repacked at ingest by the loader — same tile order as gemm_mma_fp8_b),
// scales are BF16 weight_scale_inv [M/128, K/128] upcast to f32 on upload. Dequant direction
// verified numerically vs the BF16 source: w = code · scale_inv (rel-L2 2.65e-2, the E4M3 class).
//
// Every 16-row output tile lies wholly inside one 128-row scale band (16 | 128), so the band index
// is a per-block constant `mt>>3`. The k-loop accumulates each 128-K span into a private
// sub-accumulator and folds it × s into the main accumulator at span boundaries — a FIXED
// accumulation order independent of N, so batch-invariance (col 0 == a decode) is preserved the
// same way gemm_mma_fp4_b's is. Requires K % 128 == 0 (asserted at load: every Qwen tensor is).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp8_blk_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const float* __restrict__ BlkScale,
    const __nv_bfloat16* __restrict__ X, int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkbb = K >> 7;

    const int band = mt >> 3;                       // 128-row scale band of this output tile
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    float acc[2][4]  = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float accb[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    int kbb = 0; float s = BlkScale[band * nkbb];   // span 0's scale
    for (int kb = warp; kb < nblk; kb += MMA_NW) {
        if ((kb >> 3) != kbb) {                     // crossed a 128-K boundary: fold, reset, re-read
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i >> 2][i & 3] += accb[i >> 2][i & 3] * s;
            #pragma unroll
            for (int i = 0; i < 8; i++) accb[i >> 2][i & 3] = 0.f;
            kbb = kb >> 3;
            s = BlkScale[band * nkbb + kbb];
        }
        const long long tile = (long long)mt * nblk + kb;
        const uint2 w8 = Wt64[tile * 32 + lane];            // ONE 8-byte load = the whole A-fragment

        uint32_t ra[4];
        ra[0] = fp8_pair_bf16(w8.x,       w8.x >>  8);      // row g,   cols 2t, 2t+1
        ra[1] = fp8_pair_bf16(w8.x >> 16, w8.x >> 24);      // row g+8, cols 2t, 2t+1
        ra[2] = fp8_pair_bf16(w8.y,       w8.y >>  8);      // row g,   cols 2t+8, 2t+9
        ra[3] = fp8_pair_bf16(w8.y >> 16, w8.y >> 24);      // row g+8, cols 2t+8, 2t+9

        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + (kb << 4));
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + (kb << 4));
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };

        mma_m16n8k16(accb[0], ra, rb0);
        mma_m16n8k16(accb[1], ra, rb1);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i >> 2][i & 3] += accb[i >> 2][i & 3] * s;   // final span

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ================== DSV4 FP8 BLOCK-SCALE EPILOGUE GEMM (§12.A.2, §C.3) ==================
//
// C[n,m] (bf16, or f32 into Cf for the TP path) =
//     Σ over K-128 blocks kb of ( raw fp32 block GEMM over codes ) · sa[n,kb] · sb[m/128,kb]
//
// DeepSeek-V4's FP8 weights are NOT our per-row-scaled FP8: the scales are per-128×128-block
// UE8M0 (sb), and the activations arrive pre-quantized as e4m3 codes [N,K] + per-128 UE8M0
// scales sa [N,K/128] (activation quant is a different kernel's job). The weight bytes are the
// SAME MMA-repacked 16x16 tiles as gemm_mma_fp8_b (quant.rs::repack_fp8_mma, 256 B/tile).
//
// EXACTNESS (why the only CPU-reference divergence is f32 addition ORDER):
//   * e4m3 -> bf16 is EXACT for both operands (4 significand bits <= bf16's 8), so the mma's
//     raw block GEMM is the reference's raw code GEMM bit-for-bit (fixed hw add order).
//   * UE8M0 scales are decoded IN-KERNEL as bare powers of two (bit shift, exact — no host
//     pre-decode pass, 4x less scale traffic, and the sa·sb product stays exact). b=0 is
//     2^-127, an f32 SUBNORMAL — the naive `b<<23` would give 0.0; the one-instruction fixup
//     keeps the decode equal to dsv4_load::e8m0_to_f32 on every byte.
//   * Promotion multiplies the raw block partial by sa·sb (pow2 x pow2 = exact in f32), so it
//     introduces NO rounding of its own; pow2 scaling commutes exactly with the f32 sums.
//
// THE WARP-OWNERSHIP RESTRUCTURE (the reason this is not just gemm_mma_fp8_b + a scale):
// the per-128-K promotion must happen on a whole 128-K block's raw partial, so a warp must
// OWN whole 128-K blocks. gemm_mma_fp8_b strides single 16-K blocks across the 8 warps and
// gemm_mma_fp4_b strides PAIRS — under either, one 128-K block (8 blocks = 4 pairs) is split
// across 4 warps and no warp can promote. Here warp w takes 128-blocks w, w+8, w+16, ... and
// walks the block's 4 ADJACENT k-block pairs in order. The pair step keeps fp4_b's DRAM-sector
// argument (:3128-3147): two adjacent 256 B tiles = 512 contiguous bytes per warp per step,
// every load a full sector run. The visit order (block-stride outside, ascending pairs inside)
// is FIXED and N-independent, each warp promotes its own partial in ascending owned-block
// order, and the cross-warp reduce is the existing fixed-order smem one — column 0 is
// bit-identical at every N (N padded to 16 by the usual row-clamps; no atomics anywhere).
//
// Host contract: M % 128 == 0 && K % 128 == 0 && 1 <= N <= 16 (asserted host-side).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns — the
    // padding feeds independent accumulators that are never written (invariance: :3120-3124).
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

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb2 — TWO adjacent 16-row tiles per CTA (R3A.1 E1b).
// Identical per-element contract to gemm_dsv4_fp8_bsb (above): same K-partition (warp w owns
// 128-K blocks w, w+8, ...), same per-block 8 ascending-k mma into a zero-inited raw, same
// per-block promotion by sa*sb (pow2, exact), same fixed ascending-warp epilogue reduce, same
// clamp construction. The two tiles' chains are INDEPENDENT accumulators interleaved in the
// instruction stream (contract-sanctioned ILP); the X fragments are loaded once and shared
// (same values, same columns) — activation reads halve and each warp has two independent
// weight streams in flight, which is the whole point: the single-tile kernel's per-warp
// stream is too short to cover LPDDR latency at the decode shapes (143-165 GB/s ncu).
// Grid: one block per PAIR of 16-row tiles, (M+31)/32 CTAs; M % 128 == 0 in production so the
// has1 guard never fires (kept for generality — a lone-tile CTA's chain is unchanged).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt0 = 2 * blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < M;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
            const uint2 w00 = Wt64[tile0 * 32 + lane];
            const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
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

        // Promote each tile's raw partial by its own (sa*sb) — the same two-operand products
        // in the same ascending owned-block order as the single-tile kernel (all pow2, exact).
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

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc0, C, mt0, M, N, Cf);
    __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
    if (has1) mma_epilogue_prescaled(sh, acc1, C, mt1, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb_tma — Tier-1 item 1.3 order-faithfulness PROBE: DeepGEMM-class
// (1d1d) memory structure — one cp.async.bulk (TMA) per 16 KB weight sweep into an
// FP8_TMA_NS-stage smem pipeline with mbarrier completion — hosting our LOCKED reduction
// order. PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb: same warp ownership of
// whole 128-K blocks (warp w owns kb = w, w+8, ...), same per-block 4 ascending adjacent
// k-block pairs of mma into a zeroed raw, same per-block sa*sb promotion in ascending
// owned-block order, same clamp construction, same fixed-order cross-warp epilogue. The
// ONLY change is the weight data path: the tile's repacked weight region is consumed in
// 16 KB sweeps (8 128-K blocks = kb 8j..8j+7, CONTIGUOUS in the repacked layout), staged
// by bulk TMA; the fragment loads then read the same bytes from smem that bsb reads from
// global. Values, visit order, promotion order: untouched. Bitwise gate = bsb (the probe
// verdict in DSV4_SESSION_OPT_FINAL_LIST.md 1.3 decides port vs capped).
//
// Host contract: same as bsb PLUS K % 1024 == 0 (whole sweeps), dynamic smem =
// FP8_TMA_NS * (16384 + 8) bytes, grid = M/16. NS sweep (RUN 15): NS=5 → 1 CTA/SM
// (GB10 smem opt-in cap ~99 KB incl. the 8 KB static epilogue buffer) → 149–164 GB/s;
// NS=2 → 2 CTAs/SM → 191–230 GB/s ≈ parity with bsb2, never above — hence CAPPED at
// probe status, NOT wired into production (verdict: DSV4_SESSION_OPT_FINAL_LIST.md 1.3).
#define FP8_TMA_NS 2
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_tma(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nkb = K >> 7;
    const int sweeps = nkb >> 3;             // 8 128-K blocks (16 KB) per sweep

    extern __shared__ __align__(128) uint8_t tsmem[];   // NS x 16 KB stages, then NS barriers
    uint64_t* bars = reinterpret_cast<uint64_t*>(tsmem + FP8_TMA_NS * 16384);

    if (threadIdx.x < FP8_TMA_NS) {
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[threadIdx.x]);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(baddr));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    // TMA issue for sweep j into stage j%NS (one thread; the barrier's single arrival is
    // the issuing thread's arrive.expect_tx; completion needs the 16384 tx bytes).
    const long long region = (long long)mt * nkb * 2048;    // this tile's weight region base
    auto issue = [&](int j) {
        const int s = j % FP8_TMA_NS;
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[s]);
        const uint32_t saddr = (uint32_t)__cvta_generic_to_shared(tsmem + s * 16384);
        const uint8_t* src = Wt + region + (long long)j * 16384;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(baddr), "r"(16384));
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(saddr), "l"(src), "r"(16384), "r"(baddr) : "memory");
    };
    if (threadIdx.x == 0) {
        const int pre = sweeps < FP8_TMA_NS ? sweeps : FP8_TMA_NS;
        for (int j = 0; j < pre; j++) issue(j);
    }

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns —
    // verbatim from bsb (activation path is unchanged).
    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int j = 0; j < sweeps; j++) {
        const int s = j % FP8_TMA_NS;
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[s]);
        const int phase = (j / FP8_TMA_NS) & 1;
        uint32_t done = 0;
        while (!done) {
            asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                         " selp.u32 %0, 1, 0, p;}"
                         : "=r"(done) : "r"(baddr), "r"(phase));
        }

        // warp's 128-K block this sweep is kb = warp + 8j — the warp's 2 KB run inside the
        // staged 16 KB (offset warp*2048), then the verbatim bsb fragment walk.
        const int kb = warp + 8 * j;
        const uint2* Wt64 = reinterpret_cast<const uint2*>(tsmem + s * 16384 + warp * 2048);
        float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const uint2 w0 = Wt64[q * 64 + lane];           // tile 2q   (256 B, 8 B/lane)
            const uint2 w1 = Wt64[q * 64 + 32 + lane];      // tile 2q+1, back-to-back

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

        __syncthreads();        // stage fully consumed before it is reissued
        if (threadIdx.x == 0 && j + FP8_TMA_NS < sweeps) issue(j + FP8_TMA_NS);
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb2q — TWO-OP pair (R3A.1 E2 first rung): ONE launch computes two
// INDEPENDENT projections that share the activation (production: wq_a + wkv per layer, both
// read the same x codes/scales). CTAs [0, packs0) compute op0's tiles (2 per CTA), CTAs
// [packs0, packs0+packs1) compute op1's. Per-element contract per tile is IDENTICAL to
// gemm_dsv4_fp8_bsb2 on that op's weight (same K-partition, same per-block 8 ascending-k mma
// into zero-inited raw, same per-block sa*sb promotion, same fixed ascending-warp epilogue
// reduce) — the op select only routes tile indices; the gate compares against the two
// separate bsb2 launches bitwise. M0 % 128 == 0 && M1 % 128 == 0 in production (guard kept).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb2q(
    __nv_bfloat16* C0, const uint8_t* __restrict__ Wt0, const uint8_t* __restrict__ Sb0,
    __nv_bfloat16* C1, const uint8_t* __restrict__ Wt1, const uint8_t* __restrict__ Sb1,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    long long m01, long long kn, float* Cf)
{
    const int M0 = (int)(m01 & 0xffffffff), M1 = (int)(m01 >> 32);
    const int K = (int)(kn & 0xffffffff), N = (int)(kn >> 32);
    const int packs0 = (M0 + 31) / 32;
    const int op = (blockIdx.x >= packs0) ? 1 : 0;
    const int b = op ? (int)blockIdx.x - packs0 : (int)blockIdx.x;
    const int mt0 = 2 * b;
    const int M = op ? M1 : M0;
    const bool has1 = (mt0 + 1) * 16 < M;
    __nv_bfloat16* C = op ? C1 : C0;
    const uint8_t* Wt = op ? Wt1 : Wt0;
    const uint8_t* Sb = op ? Sb1 : Sb0;
    const int mt1 = mt0 + (has1 ? 1 : 0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
            const uint2 w00 = Wt64[tile0 * 32 + lane];
            const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
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

        // Promote each tile's raw partial by its own (sa*sb) — the same two-operand products
        // in the same ascending owned-block order as the single-tile kernel (all pow2, exact).
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

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc0, C, mt0, M, N, Cf);
    __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
    if (has1) mma_epilogue_prescaled(sh, acc1, C, mt1, M, N, Cf);
}

