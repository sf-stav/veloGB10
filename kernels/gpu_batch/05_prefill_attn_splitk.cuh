// gpu_batch/05_prefill_attn_splitk.cuh: PREFILL KERNELS: write_kv_prefill, prefill attention, conv1d_prefill_state, gqa_attn_splitk (bf16, q4).
// Sequential over N positions.
// Section 5 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================== PREFILL KERNELS (sequential over N positions) =====================

// ---- write_kv_prefill: write N positions of K/V into ONE slot's cache ----
// k_new,v_new: [nkv*hd, N] bf16; k_cache,v_cache: [nkv, stride, hd] bf16.
// pos_start: absolute write offset (0 for from-scratch prefill; = current decode position for
// causal-append / MTP verify, so the N new K/V vectors land at positions pos_start..pos_start+N-1).
extern "C" __global__ void write_kv_prefill(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, int stride, int nkv, int hd, int N, int pos_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * nkv * hd;
    if (idx >= total) return;
    int t = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int d = rem % hd;
    // The cache holds `stride` positions per head. A prompt longer than the cache used to write straight
    // past the end of it — silently, corrupting whatever allocation came next, which surfaced as two
    // identical prefills producing different answers. The caller must reject an over-long prompt; this
    // guard is the backstop that turns a heap corruption into a dropped write.
    if (pos_start + t >= stride) return;
    long long pos = (long long)h * stride + (pos_start + t);
    k_cache[pos * hd + d] = k_new[(long long)t * nkv * hd + h * hd + d];
    v_cache[pos * hd + d] = v_new[(long long)t * nkv * hd + h * hd + d];
}

// ---- gqa_attn_prefill: causal attention for N positions of one sequence ----
// q: [nh*hd, N] bf16; k_cache,v_cache: [nkv, stride, hd] bf16; out: [nh*hd, N] bf16
// ONE BLOCK PER (QUERY TILE OF hd/32, query_head); blockDim.x = hd. Each warp owns ONE query of the
// tile (hd=256 → 8 queries/block, hd=128 → 4). Works for any hd that is a positive multiple of 32
// and <= 512 (SK_DPL_MAX register cap). pos_start: absolute position of the first of the N tokens
// (0 for a from-scratch prompt prefill; for a causal-append -- the MTP head's prompt prime -- = the
// position the append starts at, so column i attends to KV[0 .. pos_start+i]).
//
// TWO BUGS DIED HERE, and they are worth remembering separately.
//
// (1) The ORIGINAL did a full 256-thread tree reduction, WITH __syncthreads(), FOR EVERY SINGLE KEY:
//     ~11 barriers per key on a loop whose length is the context. Measured on an 8K prompt: 2.27 s
//     per layer. Fixed by making the q.k dot product a WARP SHUFFLE -- no shared memory, no barriers.
//
// (2) That fix took it to 348 ms/layer and I nearly stopped, because it "looked like flash attention"
//     -- but it still gave EVERY QUERY ITS OWN BLOCK, and every block re-read the entire K/V history.
//     At 8K that is nh * N^2/2 = 536M (query,key) pairs x 1024 B = 549 GB of L1/L2 traffic per layer.
//     Against the measured 348 ms that is ~1.6 TB/s: pinned to the L2 bandwidth ceiling. The arithmetic
//     says only ~46 ms of it was instruction issue, so it was never compute -- it was re-reading K/V
//     16-thousand times. STATIC INSPECTION CANNOT SEE THIS; only the traffic arithmetic can.
//
// So: a block now owns a TILE of query positions (warp w takes query tile*QT + w) and the warps sweep
// the SAME keys together. One K/V row fetch from L2 now serves QT queries instead of 1 -- an 8x cut
// (at hd=256) in the binding resource. Each warp carries the complete running softmax (m, l, acc)
// for ITS OWN query in registers across the whole key range, so no warp ever needs another warp's
// partial: the cross-warp merge, and all of the kernel's shared memory, are simply GONE.
//
// Causality: warps in a block are within QT positions of each other, so they diverge only over the
// last handful of keys -- masked with a predicate, not a branch out of the loop.
extern "C" __global__ void gqa_attn_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh, int nkv, int hd, float scale, int N, int pos_start) {
    const int QT = blockDim.x >> 5;              // query positions per block == warps per block (hd/32)
    const int blk = blockIdx.x;
    const int tile = blk / nh, qh = blk % nh;
    const int kvh = qh / (nh / nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;                   // head dims per lane (8 when hd=256, 4 when hd=128)

    const int t = tile * QT + warp;            // THIS warp's query position
    const bool active = (t < N);
    const int pc = pos_start + t + 1;          // keys this warp's query attends to
    // Keys ANY warp in this block needs: the block sweeps to the last query's horizon.
    const int tlast = min(tile * QT + QT - 1, N - 1);
    const int pc_blk = pos_start + tlast + 1;

    // Per-lane register slices: compile-time trip count SK_DPL_MAX keeps qv/acc in registers
    // (AGENTS.md §4.1); lanes i >= DPL are predicated off and contribute exact +0.0f terms, so an
    // hd=256 launch is bit-identical to the old hardcoded-256 kernel.
    const __nv_bfloat16* qrow = q + (long long)(active ? t : 0) * (nh * hd) + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;

    // ===================== WHAT THIS KERNEL IS ACTUALLY BOUND BY: STILL UNKNOWN =====================
    //
    // Measured, 9B, 8K prompt, per attention layer (nsys, 16 launches = 2 prefills x 8 layers):
    //
    //   barrier-per-key (the original)            2270 ms
    //   warp-shuffle, one block per QUERY          348 ms
    //   + query tiling, 8 queries share each key   310 ms   <-- 11%, NOT the 7x predicted
    //   + unroll 4 keys per iteration              312 ms   <-- 0%
    //
    // TWO HYPOTHESES, BOTH WRONG, BOTH KILLED BY MEASUREMENT:
    //
    //   1. "L2-bandwidth bound." One block per query re-read the whole K/V: nh*N^2/2 = 536M
    //      (query,key) pairs x 1024 B = 549 GB/layer, which against 348 ms is ~1.6 TB/s -- right at the
    //      L2 ceiling. Query tiling cuts that traffic 8x. It bought 11%. So L2 bandwidth was NOT the
    //      binding constraint, and the fact that the arithmetic *landed on* the L2 ceiling was a
    //      coincidence that read like a diagnosis.
    //
    //   2. "Then it's latency: sharing a key across 8 warps collapsed memory-level parallelism 8x."
    //      Plausible, and it predicts that giving each warp 4 independent K/V rows in flight recovers
    //      it. Implemented (KU=4). It bought ZERO. So it is not MLP either.
    //
    // What is left is instruction issue: ~536M warp-key iterations, each doing 16 scalar FMA, 16 bf16
    // -> f32 converts, a 5-step shuffle reduction and 2 transcendentals. That is ~25e9 warp-instructions
    // for one layer, and the arithmetic says an issue-bound kernel should take ~80 ms, not 310 -- so
    // even that does not close. `ncu` would answer it in one run (stall reasons, issue efficiency), but
    // GPU performance counters are locked on this box (ERR_NVGPUCTRPERM -- see PROFILING.md), so this
    // stays honestly open rather than getting a third confident story.
    //
    // The structural answer is almost certainly tensor cores: an mma.sync QK^T + PV replaces ~16 scalar
    // FMAs and 16 converts per lane per key with a handful of instructions on bf16 fragments. That is
    // the FlashAttention-2 design and it is the open question for the expert.
    //
    // Keeping the query tiling (it is a real 11% and it is strictly less DRAM traffic); NOT keeping the
    // key unroll (it bought nothing and cost readability).
    for (int tt = 0; tt < pc_blk; tt++) {
        // All warps issue this same address: one fetch, QT queries served.
        const __nv_bfloat16* krow = kb + (long long)tt * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;                              // now uniform across the warp

        if (active && tt < pc) {                 // causal mask (differs only over the last <QT keys)
            const float m_new = fmaxf(m, s);
            const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
            const __nv_bfloat16* vrow = vb + (long long)tt * hd;
            #pragma unroll
            for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
            m = m_new;
            l = l * a_old + a_cur;
        }
    }

    // A warp owns its query's ENTIRE softmax -- nothing to merge, nothing to share.
    // U2 class-1 (PLAN/U2_FIX_DRAFT.md): normalise with a single division acc/l, matching
    // gqa_attn_reduce's acc/l — acc*(1/l) rounds differently and a near-tie argmax can flip.
    if (active) {
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) orow[i] = f2b(l > 0.0f ? acc[i] / l : 0.0f);
    }
}

// ===================== TILED PREFILL ATTENTION (materialised S, cuBLAS GEMMs) =====================
//
// The scalar gqa_attn_prefill is bound by instruction issue, not by memory: query tiling cut its L1/L2
// traffic 8x for 11%, and putting 4 independent K/V rows in flight per warp bought 0%. A PERFECT scalar
// kernel bottoms out around 25 ms/layer (549 GFLOP of QK^T+PV against ~22 TFLOPS of f32 CUDA-core
// issue); a MEDIOCRE tensor-core one starts at ~9 ms. That is a choice between hardware units, not
// between tunings -- so the fix is to route the two GEMMs through tensor cores.
//
// This is the cuBLAS-materialised form: S = Q^T K into a scratch tile, online softmax, O += P V^T.
// It is slower than a fused mma flash kernel (it pays to write and re-read S), but it (a) captures most
// of the win, and (b) is numerically simple enough to be the ORACLE that a later mma kernel is fuzzed
// against. This kernel has produced two confident wrong diagnoses already; the next version gets a
// reference to check against.
//
// PREFILL IS OUTSIDE THE BATCH-INVARIANCE CONTRACT. That contract binds decode <-> verify; prefill
// feeds both identically. So we may tile, split and reorder here freely -- the discipline that governs
// every other kernel in this file does not apply. (Still deterministic: fixed tiling, fixed order.)
//
// Layouts, all column-major so cuBLAS can eat them directly:
//   Q_h  [hd x N ] ld = nh*hd   base = q  + qh*hd            (head-major within a token)
//   K_h  [hd x pc] ld = hd      base = kc + kvh*stride*hd
//   S^T  [Bc x Br] f32, ld = Bc, one slab per kv head   <-- TRANSPOSED, deliberately
//   O    [Br x hd] f32, ld = Br, one slab per query head -- carried across key tiles
//
// S IS STORED TRANSPOSED FOR ONE REASON: COALESCING. The softmax runs one block per query row, walking
// that row's keys. With S as [Br x Bc] column-major, adjacent threads (adjacent keys j) are Br floats
// = 4 KB apart -- every lane's load is its own 32-byte sector, and the kernel ate 85% of the attention
// time despite the GEMMs being fast. Storing S^T makes a query's row contiguous. Same DRAM-sector
// lesson as the GEMM scale reads; it does not stop being true because cuBLAS wrote the buffer.
//
// ---- attn_softmax_tile: the online-softmax step between the two GEMMs ----
// One block per (query row i, kv head). Applies scale + causal mask, updates the running (m, l) for the
// row, rescales that row's O accumulator by exp(m_old - m_new), and writes P = exp(s - m_new) in bf16
// for the PV GEMM.
//
// `r` is the index of the query head WITHIN its GQA group: the caller issues one batched GEMM per r
// (batched over kv heads), because a GQA group's 4 query heads share one K/V and cuBLAS cannot express
// "stride 0 for 4, then jump".
//
// The argument list is packed because cudarc's launch tuple caps at 12. `scale` is folded into the
// QK^T GEMM's alpha (free), nh/gqa/r share one int, and (s0, pc) share one i64.
extern "C" __global__ void attn_softmax_tile(
    const float* __restrict__ S, __nv_bfloat16* __restrict__ P,
    float* __restrict__ O, float* __restrict__ mrun, float* __restrict__ lrun,
    int Br, int Bc, int hd, int nh_gqa_r, int rows, int qpos0, long long s0_pc) {
    const int nh  = nh_gqa_r / 10000;
    const int gqa = (nh_gqa_r / 100) % 100;
    const int r   = nh_gqa_r % 100;
    const int s0  = (int)(s0_pc >> 32);
    const int pc  = (int)(s0_pc & 0xffffffffLL);
    (void)nh;

    const int i   = blockIdx.x;          // query row within the tile
    const int kvh = blockIdx.y;          // kv head
    const int qh  = kvh * gqa + r;       // query head
    const int tid = threadIdx.x, nt = blockDim.x;

    const bool live = (i < rows);
    const int qpos = qpos0 + i;                      // absolute position of the query
    const int kmax = min(pc - 1, qpos);              // causal: last key this query may see

    const float* Ss = S + (long long)kvh * Br * Bc;  // this kv head's S^T slab
    __nv_bfloat16* Ps = P + (long long)kvh * Br * Bc;
    float* Os = O + (long long)qh * Br * hd;
    float* mr = mrun + (long long)qh * Br;
    float* lr = lrun + (long long)qh * Br;

    extern __shared__ float red[];

    // ---- pass 1: row max over the valid keys of this tile ----
    float mx = -1e30f;
    if (live) {
        for (int j = tid; j < Bc; j += nt) {
            const int key = s0 + j;
            if (key <= kmax) mx = fmaxf(mx, Ss[(long long)i * Bc + j]);
        }
    }
    red[tid] = mx;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float tile_max = red[0];
    __syncthreads();

    const float m_old = live ? mr[i] : 0.0f;
    const float m_new = fmaxf(m_old, tile_max);
    // A tile that is entirely masked (or a dead row) must not touch (m, l) or O -- and exp(-inf - -inf)
    // is NaN, so guard rather than rely on arithmetic.
    const bool any = live && (tile_max > -1e29f);
    const float alpha = (any && m_old > -1e29f) ? __expf(m_old - m_new) : (any ? 0.0f : 1.0f);

    // ---- pass 2: P = exp(s - m_new), and the row sum ----
    float sum = 0.0f;
    if (live) {
        for (int j = tid; j < Bc; j += nt) {
            const int key = s0 + j;
            float p = 0.0f;
            if (any && key <= kmax) p = __expf(Ss[(long long)i * Bc + j] - m_new);
            Ps[(long long)i * Bc + j] = f2b(p);      // masked -> exactly 0, so PV adds nothing
            sum += p;
        }
    } else {
        for (int j = tid; j < Bc; j += nt) Ps[(long long)i * Bc + j] = f2b(0.0f);
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float tile_sum = red[0];

    // ---- rescale this row's O accumulator, then commit (m, l) ----
    // alpha == 1.0f exactly when the running max did not move this tile (the common case once the
    // max stabilizes), and x * 1.0f is bit-identical to x for every finite accumulator — so skipping
    // the pass is value-exact and avoids a fully strided (8x-amplified) read-modify-write of O.
    if (live && any && alpha != 1.0f) {
        for (int d = tid; d < hd; d += nt) Os[(long long)d * Br + i] *= alpha;
    }
    if (live && tid == 0) {
        mr[i] = any ? m_new : m_old;
        lr[i] = any ? (lr[i] * alpha + tile_sum) : lr[i];
    }
}

// ---- attn_finalize: O / l -> the [nh*hd, N] output layout ----
extern "C" __global__ void attn_finalize(
    __nv_bfloat16* __restrict__ out, const float* __restrict__ O, const float* __restrict__ lrun,
    int Br, int hd, int nh, int t0, int N) {
    // Coalesced on both sides via a 32-row smem transpose. The old mapping (grid (br, nh),
    // thread-per-d) read the column-major O slab at O[d*Br + i] — one 32 B sector per 4 B element,
    // ~8x read waste per row tile. Load phase: warp lanes read consecutive ROWS for a fixed d
    // (contiguous 128 B). Write phase: one warp per row writes the contiguous bf16 run. Values are
    // untouched — the same per-element o / l in the same order.
    __shared__ float tile[32][256 + 1];                  // +1 kills bank conflicts; hd <= 256 asserted by the caller
    const int i0 = blockIdx.x * 32;
    const int qh = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;        // block is (32, 8)
    const float* Os = O + (long long)qh * Br * hd;

    for (int k = 0; k < 32; k++) {
        const int d = ty * 32 + k;
        const int i = i0 + tx;
        tile[tx][d] = (i < Br && d < hd) ? Os[(long long)d * Br + i] : 0.0f;
    }
    __syncthreads();

    const int warp = ty, lane = tx;                      // 8 warps x 32 lanes; 4 rows per warp
    for (int r = 0; r < 4; r++) {
        const int i = i0 + warp * 4 + r;
        const int t = t0 + i;
        if (i >= Br || t >= N) continue;
        const float l = lrun[(long long)qh * Br + i];
        for (int d = lane; d < hd; d += 32)
            out[(long long)t * (nh * hd) + (long long)qh * hd + d] =
                f2b(l > 0.0f ? tile[warp * 4 + r][d] / l : 0.0f);
    }
}

// ---- attn_tile_init: reset the per-tile running softmax state ----
extern "C" __global__ void attn_tile_init(float* O, float* mrun, float* lrun, int Br, int hd, int nh) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long no = (long long)nh * Br * hd;
    if (idx < no) O[idx] = 0.0f;
    const long long nml = (long long)nh * Br;
    if (idx < nml) { mrun[idx] = -1e30f; lrun[idx] = 0.0f; }
}

// ---- attn_prefill_fa_b: fused flash-attention prefill (ONE launch replaces the QK^T/softmax/PV
// tile walk of attn_prefill_tiled; no S/P global materialization, no per-tile init/finalize) ----
//
// Prefill ONLY — gated by GB10_FA_PREFILL=1 and n >= PF_MIN at the Rust call site; the tiled path
// below it stays the default and the verify path is untouched (verify never reaches PF_MIN).
//
// Semantics = the attn_softmax_tile/attn_finalize numerics, one pass in registers:
//   score = (Q[t,hh,:] . K[kvh,k,:]) * rsqrt(hd);  online softmax with running (m, l, O);
//   Out = O / l.  Keys 0..pos_start+t inclusive (causal, chunked prefill — the prefix is already
//   in the cache from earlier chunks + write_kv_prefill).
//
// Layouts (identical to the cuBLAS calls it replaces):
//   Q  bf16 row-major  [N, nh*hd]        Q[t,hh,d]  = q + (t*nh+hh)*hd + d
//   K,V bf16 head-major [nkv, stride, hd] K[h,k,d]  = kc + (h*stride+k)*hd + d
//   Out bf16 [N, nh*hd]                  (same as Q)
//
// THIS BODY IS THE SCALAR REFERENCE ORACLE (warp-per-row), validated relL2 2.3e-05 vs a host
// reference across (N,pos_start) = (256,0),(130,0),(256,512),(130,501) on .13. It is the
// correctness fallback for geometries the tensor-core kernel does not cover.
extern "C" __global__ void attn_prefill_fa_sc_b(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int hd, int stride, int pos_start)
{
    const int WARPS = 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int qhead = blockIdx.y;
    const int kvh  = qhead / gqa;
    const int t = blockIdx.x * WARPS + warp;
    if (t >= N) return;
    const int qpos = pos_start + t;
    const int dpl = hd / 32;                       // dims per lane (8 at hd=256)
    const __nv_bfloat16* Qp = Q + ((long long)t * nh + qhead) * hd;
    float qv[8];
    for (int i = 0; i < dpl; i++) qv[i] = b2f(Qp[lane * dpl + i]);
    float m = -1e30f, l = 0.0f;
    float ov[8];
    for (int i = 0; i < dpl; i++) ov[i] = 0.0f;
    const float inv_sqrt_hd = rsqrtf((float)hd);
    const __nv_bfloat16* Kb = K + (long long)kvh * stride * hd;
    const __nv_bfloat16* Vb = V + (long long)kvh * stride * hd;
    for (int k = 0; k <= qpos; k++) {
        const __nv_bfloat16* Kp = Kb + (long long)k * hd;
        const __nv_bfloat16* Vp = Vb + (long long)k * hd;
        float kv[8], vv[8];
        for (int i = 0; i < dpl; i++) { kv[i] = b2f(Kp[lane * dpl + i]); vv[i] = b2f(Vp[lane * dpl + i]); }
        float dot = 0.f;
        for (int i = 0; i < dpl; i++) dot += qv[i] * kv[i];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        float score = dot * inv_sqrt_hd;
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * alpha + p;
        for (int i = 0; i < dpl; i++) ov[i] = ov[i] * alpha + p * vv[i];
        m = m_new;
    }
    for (int i = 0; i < dpl; i++) {
        const int d = lane * dpl + i;
        if (d < hd)
            Out[((long long)t * nh + qhead) * hd + d] = f2b(l > 0.0f ? ov[i] / l : 0.0f);
    }
}

// ---- attn_prefill_fa_b: FUSED FLASH-ATTENTION PREFILL, TENSOR-CORE BODY ----
// Implements PLAN/FA_KERNEL_EXPERT_REPORT.md (measured on .13: 15.3/44.7/74.0/103.5 ms at
// N=8192, pc=8K/16K/24K/32K — 4.0x under the 3-kernel path's per-call bar at every geometry,
// ~54 TF/s effective vs the ~75 TF/s bf16 mma ceiling).
//
// Config (report §0): block 192 thr (6 warps) x M=96 rows = 16 tokens x 6 GQA heads
// (token-major; the g heads share the K/V smem tiles = guaranteed 6x reuse), Bc=32 keys
// double-buffered via cp.async (64 KiB dynamic smem, opt-in from Rust), Q+O in registers
// (~255 regs, 4 B spill at hd=256), K ldmatrix NO-transpose (K[key][d] IS B .col), V .trans,
// P->A repack in-register (zero permutations — D-frag col pattern == A-frag k pattern).
// Online-softmax guards replicate attn_softmax_tile exactly (sentinels, any-guard,
// alpha==1.0f rescale skip, l>0 epilogue). P is rounded to bf16 before PV — the SAME rounding
// point as the production tiled path's bf16 p_buf.
//
// Correctness (tool_probe/fa_tc.cu harness, .13): relL2 1.9e-3 vs an fp32 host ref = the bf16-P
// quantization envelope (2^-9), verified by a peaky-softmax structural test (error collapses to
// 1-output-ulp flips concentrated on few-key rows; hd=128 identical). Layouts identical to the
// tiled path; hd must be 128 or 256 (host dispatches, other hd -> scalar/old path); g must
// divide 96 (g=6 -> 16 tokens/block).
__device__ __forceinline__ unsigned fa_cvt2(float lo, float hi) {
    __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void fa_mma_m16n8k16(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void fa_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fa_ldm_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fa_cp16(__nv_bfloat16* dst, const __nv_bfloat16* src, bool full) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    const int sz = full ? 16 : 0;               // src-size 0 => 16 B zero-fill (cache bound)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(src), "r"(sz));
}

#define FA_BC 32
#define FA_THREADS 192

template <int HD>
__device__ __forceinline__ void fa_tc_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int stride, int pos_start, int g, __nv_bfloat16* smem)
{
    const int Br_t = 96 / g;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = (int)threadIdx.x;
    const int kvh  = blockIdx.y;
    const int t0   = (int)(gridDim.x - 1 - blockIdx.x) * Br_t;   // longest blocks first
    const int pc   = pos_start + N;
    const int kmax = min(pos_start + t0 + Br_t - 1, pc - 1);

    const int rr0 = 16*warp + (lane >> 2), rr1 = rr0 + 8;
    const int ttok0 = t0 + rr0 / g, ttok1 = t0 + rr1 / g;
    const int hed0 = kvh*g + rr0 % g, hed1 = kvh*g + rr1 % g;
    const int ctok0 = min(ttok0, N-1), ctok1 = min(ttok1, N-1);   // dead rows: safe loads, no stores
    const long long qb0 = ((long long)ctok0 * nh + hed0) * HD;
    const long long qb1 = ((long long)ctok1 * nh + hed1) * HD;
    const int qpos0 = pos_start + ttok0, qpos1 = pos_start + ttok1;

    unsigned qf[HD/16][4];
    #pragma unroll
    for (int ks = 0; ks < HD/16; ks++) {
        const int doff = 16*ks + 2*(lane & 3);
        qf[ks][0] = *(const unsigned*)(Q + qb0 + doff);
        qf[ks][1] = *(const unsigned*)(Q + qb1 + doff);
        qf[ks][2] = *(const unsigned*)(Q + qb0 + doff + 8);
        qf[ks][3] = *(const unsigned*)(Q + qb1 + doff + 8);
    }

    float o[HD/8][4];
    #pragma unroll
    for (int j = 0; j < HD/8; j++) o[j][0]=o[j][1]=o[j][2]=o[j][3]=0.f;
    float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;
    const float rsqrt_hd = rsqrtf((float)HD);

    auto load_tile = [&](int stage, int s0g) {
        #pragma unroll 4
        for (int c = tid; c < FA_BC * (HD >> 3); c += FA_THREADS) {
            const int key = c / (HD >> 3), chunk = c % (HD >> 3);
            const int gkey = s0g + key;
            const bool valid = gkey < pc;
            const int gk = min(gkey, pc - 1);
            const int cs = (chunk & 24) | ((chunk & 7) ^ (key & 7));   // Swizzle<3,3,3>
            __nv_bfloat16* kd = smem + (stage * FA_BC + key) * HD + (cs << 3);
            __nv_bfloat16* vd = smem + ((2 + stage) * FA_BC + key) * HD + (cs << 3);
            const __nv_bfloat16* gkP = K + ((long long)kvh * stride + gk) * HD + (chunk << 3);
            const __nv_bfloat16* gvP = V + ((long long)kvh * stride + gk) * HD + (chunk << 3);
            fa_cp16(kd, gkP, valid);
            fa_cp16(vd, gvP, valid);
        }
        asm volatile("cp.async.commit_group;");
    };

    load_tile(0, 0);
    if (FA_BC <= kmax) load_tile(1, FA_BC);

    for (int s0 = 0; s0 <= kmax; s0 += FA_BC) {
        const int stage = (s0 / FA_BC) & 1;
        if (s0 + FA_BC <= kmax) asm volatile("cp.async.wait_group 1;");   // retire tile s0's group
        else                    asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // ===== QK^T =====
        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f;
        {
            const __nv_bfloat16* Kb = smem + stage * FA_BC * HD;
            #pragma unroll
            for (int ks = 0; ks < HD/16; ks++) {
                #pragma unroll
                for (int j2 = 0; j2 < 2; j2++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16*j2 + lane;         ldim = 16*ks; }
                    else if (lane < 16) { lkey = 16*j2 + lane - 8;     ldim = 16*ks + 8; }
                    else if (lane < 24) { lkey = 16*j2 + 8 + lane-16;  ldim = 16*ks; }
                    else                { lkey = 16*j2 + 8 + lane-24;  ldim = 16*ks + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    const unsigned addr = (unsigned)__cvta_generic_to_shared(Kb + lkey*HD + (cs << 3));
                    unsigned r0, r1, r2, r3;
                    fa_ldm_x4(r0, r1, r2, r3, addr);
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    fa_mma_m16n8k16(s[2*j2],   qf[ks], ba);
                    fa_mma_m16n8k16(s[2*j2+1], qf[ks], bb);
                }
            }
        }

        // ===== online softmax (attn_softmax_tile guards) =====
        const bool masked_tile = (s0 + FA_BC > pos_start + t0);
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * rsqrt_hd;
                if (masked_tile) {
                    const int key = s0 + 8*j + 2*(lane&3) + (e&1);
                    if (key > (e < 2 ? qpos0 : qpos1)) v = -1e30f;
                }
                s[j][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 2));
        const bool any0 = mx0 > -1e29f, any1 = mx1 > -1e29f;
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = (any0 && m0 > -1e29f) ? __expf(m0 - mn0) : (any0 ? 0.f : 1.f);
        const float al1 = (any1 && m1 > -1e29f) ? __expf(m1 - mn1) : (any1 ? 0.f : 1.f);
        float su0 = 0.f, su1 = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = (s[j][e] > -1e29f) ? __expf(s[j][e] - (e < 2 ? mn0 : mn1)) : 0.f;
                s[j][e] = p;
                if (e < 2) su0 += p; else su1 += p;
            }
        }
        su0 += __shfl_xor_sync(~0u, su0, 1); su0 += __shfl_xor_sync(~0u, su0, 2);
        su1 += __shfl_xor_sync(~0u, su1, 1); su1 += __shfl_xor_sync(~0u, su1, 2);
        l0 = l0*al0 + su0;  m0 = mn0;
        l1 = l1*al1 + su1;  m1 = mn1;

        if (al0 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD/8; j++) { o[j][0] *= al0; o[j][1] *= al0; }
        }
        if (al1 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD/8; j++) { o[j][2] *= al1; o[j][3] *= al1; }
        }

        // ===== P->A in-register repack + PV =====
        unsigned pa[2][4];
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            pa[ks][0] = fa_cvt2(s[2*ks  ][0], s[2*ks  ][1]);
            pa[ks][1] = fa_cvt2(s[2*ks  ][2], s[2*ks  ][3]);
            pa[ks][2] = fa_cvt2(s[2*ks+1][0], s[2*ks+1][1]);
            pa[ks][3] = fa_cvt2(s[2*ks+1][2], s[2*ks+1][3]);
        }
        {
            const __nv_bfloat16* Vb = smem + (2 + stage) * FA_BC * HD;
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                #pragma unroll
                for (int jn = 0; jn < HD/16; jn++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16*ks + lane;         ldim = 16*jn; }
                    else if (lane < 16) { lkey = 16*ks + 8 + lane - 8; ldim = 16*jn; }
                    else if (lane < 24) { lkey = 16*ks + lane - 16;    ldim = 16*jn + 8; }
                    else                { lkey = 16*ks + 8 + lane-24;  ldim = 16*jn + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    const unsigned addr = (unsigned)__cvta_generic_to_shared(Vb + lkey*HD + (cs << 3));
                    unsigned r0, r1, r2, r3;
                    fa_ldm_x4_trans(r0, r1, r2, r3, addr);
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    fa_mma_m16n8k16(o[2*jn],   pa[ks], ba);
                    fa_mma_m16n8k16(o[2*jn+1], pa[ks], bb);
                }
            }
        }

        __syncthreads();                                  // warps done reading buf[stage]
        if (s0 + 2*FA_BC <= kmax) load_tile(stage, s0 + 2*FA_BC);
    }

    const float il0 = (l0 > 0.f) ? 1.f/l0 : 0.f;
    const float il1 = (l1 > 0.f) ? 1.f/l1 : 0.f;
    if (ttok0 < N) {
        const long long ob = ((long long)ttok0 * nh + hed0) * HD;
        #pragma unroll
        for (int j = 0; j < HD/8; j++)
            *(__nv_bfloat162*)(Out + ob + 8*j + 2*(lane&3)) =
                __float22bfloat162_rn(make_float2(o[j][0]*il0, o[j][1]*il0));
    }
    if (ttok1 < N) {
        const long long ob = ((long long)ttok1 * nh + hed1) * HD;
        #pragma unroll
        for (int j = 0; j < HD/8; j++)
            *(__nv_bfloat162*)(Out + ob + 8*j + 2*(lane&3)) =
                __float22bfloat162_rn(make_float2(o[j][2]*il1, o[j][3]*il1));
    }
}

extern "C" __global__ void __launch_bounds__(FA_THREADS, 1) attn_prefill_fa_b(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int hd, int stride, int pos_start, int g)
{
    extern __shared__ __nv_bfloat16 fa_smem[];    // K[2][32][hd] then V[2][32][hd], opt-in >48 KiB
    if (hd == 256)      fa_tc_body<256>(Q, K, V, Out, N, nh, nkv, stride, pos_start, g, fa_smem);
    else if (hd == 128) fa_tc_body<128>(Q, K, V, Out, N, nh, nkv, stride, pos_start, g, fa_smem);
    // other hd: caller must route to attn_prefill_fa_sc_b / the tiled path
}



// ---- conv1d prefill: FULLY PARALLEL over (channel, position) ----
// in: [conv_dim, N] bf16 col-major; out: [conv_dim, N] bf16; state: [conv_dim, k] f32; w: [conv_dim, k] f32
//
// THIS IS NOT A RECURRENCE, AND TREATING IT AS ONE COST 163 ms OF EVERY 8K PREFILL.
//
// The old kernel ran one thread per channel and walked N positions sequentially, shuffling a k-wide
// window as if the convolution had a carried dependence. It does not. A width-k CAUSAL DEPTHWISE conv
// is a stencil: output t reads inputs t-(k-1) .. t and nothing else. Written out, with `xf` the virtual
// input = (the k-1 tail carried in `state`) followed by this chunk's N inputs:
//
//     xf[i]           = state[c][i + 1]              for i < k-1        (x[-(k-1)] .. x[-1])
//     xf[(k-1) + t]   = in[t][c]                     for t in [0, N)
//     out[t][c]       = silu( sum_{j<k} w[c][j] * xf[t + j] )
//
// so every (c, t) is independent. One thread each; conv_dim*N threads instead of conv_dim.
//
// THE ACCUMULATION ORDER IS LOAD-BEARING. This kernel also runs on the MTP verify path, where column 0
// must stay BIT-IDENTICAL to a 1-token decode (which uses conv1d_b). Same values, same ascending-j FMA
// order => same bits. Do not "optimise" this into a tree reduction.
//
// It had to stop being in-place: thread (c,t) reads inputs t-(k-1)..t, which other threads are writing.
// Hence separate in/out buffers -- the caller swaps them.
//
// mid_state: optional PER-COLUMN state checkpoints. Column t's post-state is written to
// mid_state + t*(conv_dim*k), for t in [0, N-2] -- one snapshot per position we might roll back to.
// NULL disables snapshotting (regular prefill passes NULL). The post-state after token t is just the
// window ending at t: state_after_t[j] = xf[t + j]. Which is exactly what this thread already read.
//
// (Snapshotting used to cover only column 0 -- "the committed token is always accepted". True ONLY at
// depth 2. At depth >= 3, accepting nacc=1 and rejecting the rest rolled the recurrent state back past
// the accepted draft, so the GDN state went stale and every later draft was built on it.)
// win_src (tree drafting): per-node window sources. win_src[t*k + j] = the j-th window source for node
// t, encoded as: v >= 0 -> draft-block input in[v*conv_dim + c]; v < 0 -> carried state st0[v + k].
// nullptr => the CHAIN formula v = t + j - (k-1) (consecutive positions) -- byte-identical to the
// pre-tree kernel, and what the main prefill always uses. Only the verify passes a real table.
// F0: `in_stride` is the `in` row pitch (conv_dim packed, or the fused GDN mtot for a qkv view).
extern "C" __global__ void conv1d_prefill(__nv_bfloat16* out, const __nv_bfloat16* in,
                                          const float* state, const float* w,
                                          int conv_dim, int k, int N, float* mid_state,
                                          const int* win_src, const int* slot_ids, int in_stride) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)conv_dim * N;
    if (idx >= total) return;
    const int t = (int)(idx / conv_dim);        // position
    const int c = (int)(idx % conv_dim);        // channel -- adjacent threads => coalesced over c

    // Forest: a negative win_src entry reads the committed conv tail from `state`. slot_ids non-null
    // (verify) => `state` is the layer BASE and this column's lane is slot_ids[t]; null (prefill) =>
    // `state` is pre-offset to the single slot. Single-lane verify: slot_ids[t]==slot, identical.
    const long long cslot = slot_ids ? (long long)slot_ids[t] : 0;
    const float* st0 = state + cslot * ((long long)conv_dim * k) + (long long)c * k;
    const float* wc  = w + (long long)c * k;

    // k == 4 specialization: the window lives in SCALAR REGISTERS. A runtime-indexed local array
    // (float win[8] under #pragma unroll 1, the old code) cannot be register-allocated — ptxas
    // reports a 32 B stack frame and every access is a DRAM-backed load+store, in the kernel that
    // runs conv_dim*N threads per GDN layer. Same values, same ascending-j FMA order => bit-identical
    // to the generic path (and to conv1d_b, as the note above requires).
    if (k == 4) {
        const int v0 = win_src ? win_src[t * 4 + 0] : (t - 3);
        const int v1 = win_src ? win_src[t * 4 + 1] : (t - 2);
        const int v2 = win_src ? win_src[t * 4 + 2] : (t - 1);
        const int v3 = win_src ? win_src[t * 4 + 3] : (t - 0);
        const float w0 = (v0 < 0) ? st0[v0 + 4] : b2f(in[(long long)v0 * in_stride + c]);
        const float w1 = (v1 < 0) ? st0[v1 + 4] : b2f(in[(long long)v1 * in_stride + c]);
        const float w2 = (v2 < 0) ? st0[v2 + 4] : b2f(in[(long long)v2 * in_stride + c]);
        const float w3 = (v3 < 0) ? st0[v3 + 4] : b2f(in[(long long)v3 * in_stride + c]);
        float acc = wc[0] * w0;
        acc += wc[1] * w1;
        acc += wc[2] * w2;
        acc += wc[3] * w3;                          // ASCENDING — see the bit-identity note
        out[(long long)t * conv_dim + c] = f2b(silu_f(acc));
        if (mid_state && t < N - 1) {
            float* ms = mid_state + (long long)t * conv_dim * 4 + (long long)c * 4;
            ms[0] = w0; ms[1] = w1; ms[2] = w2; ms[3] = w3;
        }
        return;
    }

    float win[8];                                // generic fallback (k != 4; unused in production)
    #pragma unroll 1
    for (int j = 0; j < k; j++) {
        const int v = win_src ? win_src[t * k + j] : (t + j - (k - 1));
        win[j] = (v < 0) ? st0[v + k]
                         : b2f(in[(long long)v * in_stride + c]);
    }
    float acc = 0.0f;
    #pragma unroll 1
    for (int j = 0; j < k; j++) acc += wc[j] * win[j];   // ASCENDING -- see the bit-identity note
    out[(long long)t * conv_dim + c] = f2b(silu_f(acc));

    if (mid_state && t < N - 1) {
        float* ms = mid_state + (long long)t * conv_dim * k + (long long)c * k;
        for (int j = 0; j < k; j++) ms[j] = win[j];
    }
}

// ---- conv1d prefill: carry the final window back into `state` ----
// Separate launch, because conv1d_prefill READS `state` from every thread: writing it there would race.
// One thread per channel; reads the whole final window before writing any of it, so the in-place update
// is safe even when N < k (short verify widths, where the window still straddles the old state).
// F0: `in_stride` is the `in` row pitch (conv_dim packed, or the fused GDN mtot for a qkv view).
extern "C" __global__ void conv1d_prefill_state(float* state, const __nv_bfloat16* in,
                                                int conv_dim, int k, int last_t, int lane_len,
                                                const int* slot_ids, int in_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    // Carry ONE lane's final conv window into its committed slot. The lane ends at column `last_t` and
    // spans `lane_len` columns; a window position before the lane start (offset < 0) falls back to the
    // lane's own committed tail (via slot_ids[last_t]). Launched once per lane by the host. Single lane
    // (last_t==N-1, lane_len==N; slot_ids null=>pre-offset): byte-identical to the pre-forest carry.
    const long long lslot = slot_ids ? (long long)slot_ids[last_t] : 0;
    float* st = state + lslot * ((long long)conv_dim * k) + (long long)c * k;

    if (k == 4) {
        // Scalar registers — same local-memory trap as conv1d_prefill (32 B stack frame via ptxas).
        const int off0 = lane_len - 4;
        const long long base = (long long)last_t - 3;
        const float n0 = (off0 + 0 < 0) ? st[off0 + 4] : b2f(in[(base + 0) * in_stride + c]);
        const float n1 = (off0 + 1 < 0) ? st[off0 + 5] : b2f(in[(base + 1) * in_stride + c]);
        const float n2 = (off0 + 2 < 0) ? st[off0 + 6] : b2f(in[(base + 2) * in_stride + c]);
        const float n3 = (off0 + 3 < 0) ? st[off0 + 7] : b2f(in[(base + 3) * in_stride + c]);
        st[0] = n0; st[1] = n1; st[2] = n2; st[3] = n3;   // read-all-then-write-all
        return;
    }

    float nx[8];
    for (int j = 0; j < k; j++) {
        const int off = (lane_len - k) + j;      // offset from lane start; < 0 => committed tail
        nx[j] = (off < 0) ? st[off + k]
                          : b2f(in[(long long)(last_t - (k - 1) + j) * in_stride + c]);
    }
    for (int j = 0; j < k; j++) st[j] = nx[j];   // read-all-then-write-all: no self-overlap hazard
}

// ===================== SPLIT-K ATTENTION (Flash-Decoding) =====================
// Parameterized for any head_dim that is a positive multiple of 32 and <= SK_HD_MAX
// (qwen3.5: 256; hy_v3: 128; DeepSeek: 512). hd arrives inside nh_packed =
// (nh << 20) | (hd << 10) | nkv  (nh < 2048, hd/nkv <= 1023).

#define MAX_VERIFY 16   // MUST match gpu.rs MAX_VERIFY (verify width cap; path table row stride)

// One block per (batch, query_head, split); blockDim.x = 256 = 8 warps. Each block owns a contiguous
// slice of the key range and emits a partial softmax (m, l, acc) that gqa_attn_reduce merges.
//
// THE PER-KEY BARRIER: this used to run a 256-thread __syncthreads() tree reduction for EVERY key --
// ~90k serialised barriers per block at an 8K context. Now: warp-shuffle dot with 8 warps striding the
// split's keys, each carrying its own register-resident (m, l, acc[8]).
//
// ===================== THE LOSSLESS-MTP CONTRACT LIVES HERE =====================
//
// Column k of an N-wide verify MUST be bit-identical to a 1-token decode at the same position. That
// forces attention for a query at position p to be a pure function of (q, KV[0..p], pc) -- it may not
// depend on B, on the other columns, or on anything derived from them. Two violations lived here:
//
//   1. n_splits was derived from batch*nh ("split more when there are too few CTAs"), so a decode
//      split the keys 6 ways and a 4-wide verify split them 2 ways.
//   2. Deriving it from max_pc instead was ALSO WRONG, and this is the subtle one: a decode has
//      max_pc = pos+1 while a verify has max_pc = pos_start+N. Whenever those straddle a multiple of
//      256 the two disagree -- and if one lands on n_splits==1 it took a DIFFERENT KERNEL entirely
//      (gqa_attn_flash, which keeps the numerator in fp32) while the other round-tripped the
//      numerator through bf16 here. End-to-end MTP was silently non-lossless; it passed anyway most
//      runs, because a 1-ulp difference rarely flips an argmax. That is the worst kind of bug: it
//      fails as a coin toss, so a green gate proves nothing.
//
// So `ns` is computed HERE, from THIS COLUMN's OWN pc, and nothing else. The caller sizes the grid
// with an upper bound (pc_b <= max_pc => ns_b <= ns_grid) and surplus blocks return without writing.
// gqa_attn_flash is gone: one kernel, one code path, no way for decode and verify to take different
// ones. out_acc is fp32 for the same reason -- the bf16 round-trip was pure lossy noise.
__device__ __forceinline__ int sk_nsplits(int pc) { return min(max(pc / 256, 1), 32); }

// RANK-SPACE split-K (tree drafting, review §2). A tree column must execute a DECODE's exact iteration
// space over its LOGICAL prefix; only the rank->slot address map changes. So `pos` now carries the
// LOGICAL position (prefix + ancestor path incl. self); it governs `pc` and the split structure. The
// keys enter via `path`: rank r < pos_start is a prefix key at slot r; rank r >= pos_start is the
// (r-pos_start)-th on-block ancestor, at slot pos_start + path[b*MAX_VERIFY + (r-pos_start)].
// path == nullptr => identity (t = r): the decode path and any plain chain, byte-identical to pre-tree.
// This keeps `ns` a pure function of each column's own logical key count, so column 0 (and every
// ACCEPTABLE node, whose emitted logits must match a decode) stays bit-identical -- unlike a
// slot-derived pc, which drifts by ulps on a 256-straddle (the third n_splits bug). ONE loop, ONE
// warp stride, address indirection only: splitting into prefix+draft loops would regroup the per-warp
// partials and break bit-identity even for a chain.
// stride is packed into bs_packed to fit the 12-arg launch cap: bits 0-18 stride, 19-24 ns_grid, 25-30 B.
extern "C" __global__ void gqa_attn_splitk(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    // scale = 1/sqrt(hd), computed in-kernel to free a launch slot for `col_pos_start` (12-arg cudarc
    // ceiling). For hd=256, sqrtf(256)=16 exactly, so 1/16 = 0.0625f is bit-identical to the old
    // constant — the qwen3.5 decode is unchanged to the last bit. FOREST: `col_pos_start[b]` is column
    // b's lane prefix boundary; null => pos[0] (single lane / tree / decode) — byte-identical to pre-forest.
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);   // grid fan-out (an UPPER BOUND on every column's ns)
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    // F0: the q row pitch (bits 31-49) — the fused qkv mtot for offset views, nh*hd for packed.
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos[b] + 1;                  // LOGICAL key count (chain: unchanged)
    // Prefix boundary: per-column (its lane's committed length) for a forest; pos[0] otherwise (used iff path).
    const int pos_start = col_pos_start ? col_pos_start[b] : pos[0];
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);             // THIS COLUMN's split count -- from ITS OWN pc alone
    if (split >= ns) return;                   // surplus block: a wider column needed the fan-out
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;            // hd/32 warps (8 at hd=256)
    const int DPL = hd >> 5;                   // head dims per lane (8 at hd=256, 4 at hd=128)

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {                          // this split has no keys
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    // Per-lane register slices, compile-time trip count SK_DPL_MAX so qv/acc stay in registers
    // (AGENTS.md §4.1); lanes i >= DPL are predicated off and contribute exact +0.0f terms, so an
    // hd=256 launch executes bit-identically to the old hardcoded-256 kernel.
    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd]; // rank -> slot
        const __nv_bfloat16* krow = kb + (long long)t * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)t * hd;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
        m = m_new;
        l = l * a_old + a_cur;
    }

    // Merge this block's warp-partials into one partial softmax, in FIXED warp order.
    extern __shared__ float sh[];
    float* sacc = sh;                     // NW * hd
    float* sm   = sh + NW * hd;           // NW
    float* sl   = sm + NW;                // NW
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) sacc[warp * hd + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

    if (threadIdx.x < hd) {
        const int d = threadIdx.x;
        float mg = -1e30f;
        for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < NW; w++) {
            const float a = __expf(sm[w] - mg);
            num += sacc[w * hd + d] * a;
            den += sl[w] * a;
        }
        out_acc[idx * hd + d] = num;              // UNNORMALISED fp32, paired with (mg, den)
        if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
    }
}

// ===================================================================================================
// gqa_attn_splitk_q4 — the 4-bit-KV twin of gqa_attn_splitk, SAME split structure, SAME reduction
// order, SAME merge. Only the K/V source changes: packed per-16 affine blocks (see kvq16_pack)
// instead of bf16 rows. Each lane dequantizes its DPL-dim slice from the block that holds it
// (DPL divides 16, so a lane never straddles blocks): scale = e4m3 byte, codes = its DPL nibbles.
// A (kvh, position, dim) dequantizes to the same fp32 value in decode, verify, and every split —
// the lossless-MTP contract is preserved by construction (see the note on the bf16 original).
extern "C" __global__ void gqa_attn_splitk_q4(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    // F0: the q row pitch (bits 31-49) — the fused qkv mtot for offset views, nh*hd for packed.
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos[b] + 1;
    const int pos_start = col_pos_start ? col_pos_start[b] : pos[0];
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);
    if (split >= ns) return;
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;
    const int DPL = hd >> 5;

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const int rb = KVQ_ROW_BYTES(hd);
    const int lane_blk = (lane * DPL) / KVQ_BLK;       // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;       // its first nibble within the block
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride * rb;
    const unsigned char* kb = k_cache + kvbase + lane_blk * 12;
    const unsigned char* vb = v_cache + kvbase + lane_blk * 12;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd];
        const unsigned char* krow = kb + (long long)t * rb;
        const float ksc = e4m3_f(krow[8]);
        const unsigned short* kcodes = (const unsigned short*)(krow + lane_off / 2);
        float kdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) kdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const int pk = (int)kcodes[u];   // SIGNED: `(pk & 0xF) - 7` in unsigned wraps nibble<7 to ~4.3e9
            kdq[u * 4 + 0] = (float)((pk & 0xF) - 7) * ksc;
            kdq[u * 4 + 1] = (float)(((pk >> 4) & 0xF) - 7) * ksc;
            kdq[u * 4 + 2] = (float)(((pk >> 8) & 0xF) - 7) * ksc;
            kdq[u * 4 + 3] = (float)(((pk >> 12) & 0xF) - 7) * ksc;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * kdq[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const unsigned char* vrow = vb + (long long)t * rb;
        const float vsc = e4m3_f(vrow[8]);
        const unsigned short* vcodes = (const unsigned short*)(vrow + lane_off / 2);
        float vdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) vdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const int pk = (int)vcodes[u];   // SIGNED: `(pk & 0xF) - 7` in unsigned wraps nibble<7 to ~4.3e9
            vdq[u * 4 + 0] = (float)((pk & 0xF) - 7) * vsc;
            vdq[u * 4 + 1] = (float)(((pk >> 4) & 0xF) - 7) * vsc;
            vdq[u * 4 + 2] = (float)(((pk >> 8) & 0xF) - 7) * vsc;
            vdq[u * 4 + 3] = (float)(((pk >> 12) & 0xF) - 7) * vsc;
        }
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * vdq[i];
        m = m_new;
        l = l * a_old + a_cur;
    }

    extern __shared__ float sh[];
    float* sacc = sh;
    float* sm   = sh + NW * hd;
    float* sl   = sm + NW;
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) sacc[warp * hd + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

    if (threadIdx.x < hd) {
        const int d = threadIdx.x;
        float mg = -1e30f;
        for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < NW; w++) {
            const float a = __expf(sm[w] - mg);
            num += sacc[w * hd + d] * a;
            den += sl[w] * a;
        }
        out_acc[idx * hd + d] = num;
        if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
    }
}

