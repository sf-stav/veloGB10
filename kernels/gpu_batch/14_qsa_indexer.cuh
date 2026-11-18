// gpu_batch/14_qsa_indexer.cuh: qwen4_exp QSA sparse-attention indexer: key write, block keys, pooling, scoring, compaction.
// Reference: modeling_qwen4_exp.py.
// Section 14 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================== qwen4_exp QSA sparse-attention indexer (Qwen4ExpTextQSAIndexer) =====================
//
// Reference (modeling_qwen4_exp.py): per full-attention layer, index_qk_proj(hidden) → q (heads×hd,
// q_layernorm (1+w), RoPE on the first rdim dims at the query position) and one RAW key (hd, cached per
// token pre-norm). The query's visible ranks [0, pc) are cut into complete blocks of `ratio` consecutive
// ranks; a block's key = k_layernorm(mean of its raw keys) with RoPE at the block-start rank; the score
// is Σ_heads relu(q_h·k_j)/√hd; the top-min(budget/ratio, nblocks) blocks plus the tail ranks
// (pc mod ratio) are the ONLY keys the layer's attention sees. Below budget+ratio-1 visible tokens every
// block is selected (dense attention is exact there — the engine keeps its dense kernels for that case).
//
// Engine mapping. Raw keys live in a per-layer `[slot][stride][hd]` bf16 cache addressed EXACTLY like
// the KV cache (write column = pos[b], slot = slot_ids[b]); block keys are recomputed on the fly from
// raw keys (four 256-B reads per block), so there is no derived cache to invalidate on an MTP rollback,
// a prefix-cache clamp or a tree compaction — whatever holds for the KV cache holds here. Selection
// runs in the SAME rank space as gqa_attn_splitk (rank r < pos_start is prefix column r; rank r >=
// pos_start is on-block ancestor `path[...]`), so a verify column's selection is a pure function of
// (its q, the committed raw keys, its ancestors) — identical to the decode at that position, which
// is what the lossless-MTP contract needs. The top-k is a deterministic radix select (ties → lowest
// block index) and the selected ranks are emitted ascending, mapped to cache columns, so the attention
// kernels below are the dense kernels with one address indirection (an identity list is bit-identical
// to the dense kernel by construction).
//
// `QsaParams` (a 64-B device struct built once per indexer at load) carries what would not fit the
// 12-argument launch cap: the rope tables, the two norm weights, eps and the geometry.
// VIS-3: per-slot indexer mrope (image requests on the EXL3 engine). A pooled block key starting at KV
// position p of slot s is roped at row rmap[s * max_pos + p] inside the slot's prompt (p < rplen[s]),
// p + rdelta[s] past it; rows >= max_pos index the slot's image-row tables mcos/msin ([rows][rdim],
// QsaParams.cos_t's layout). QsaParams.mr == nullptr (the NVFP4 engine; any text-only build) = off.
struct QsaMrope {
    const int* rmap;
    const int* rplen;
    const int* rdelta;
    const float* mcos;
    const float* msin;
    int nslots;
    int max_pos;
};
struct QsaParams {
    const float* cos_t;     // [max_pos][rdim] (duplicated halves, see build_rope_tables)
    const float* sin_t;
    const float* kw;        // k_layernorm weight [hd] (gemma style: scale = 1 + w)
    const float* qw;        // q_layernorm weight [hd] (unused on device; the q norm runs rmsnorm_rope_b)
    float eps;
    int hd;                 // indexer head dim (128; 32 on the tiny model)
    int heads;              // indexer query heads (4)
    int rdim;               // rotary dims (64; 16 on the tiny model)
    int ratio;              // compress ratio (4)
    int topk;               // block budget = indexer_budget / ratio (512)
    int sel_max;            // indexer_budget + ratio - 1 (2051): row pitch of the selection lists
    int pad;
    const struct QsaMrope* mr; // VIS-3: per-slot mrope (nullptr = off)
};
#define QSA_HD_MAX 128
#define QSA_HEADS_MAX 8
#define QSA_PF_QT 8            // queries per block in qsa_score_prefill_b

__device__ __forceinline__ float qsa_bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

// rank -> cache column, gqa_attn_splitk's rule (path == nullptr => identity).
__device__ __forceinline__ int qsa_col(int rank, int pos_start, const unsigned char* path, int b) {
    if (!path || rank < pos_start) return rank;
    return pos_start + (int)path[b * MAX_VERIFY + (rank - pos_start)];
}

// 1. raw key write. keys[(slot*stride + col)*hd + d] = qk[b*pitch + k_off + d].
//    pos == nullptr: col = pos_start + b and slot 0 (prefill chain into a slot base); else col = pos[b],
//    slot = slot_ids[b] (decode / verify, base pointer = slot 0).
extern "C" __global__ void qsa_key_write_b(__nv_bfloat16* keys, const __nv_bfloat16* qk, const int* pos,
                                           const int* slot_ids, int stride, int pos_start, int pitch, int k_off,
                                           int hd, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * hd) return;
    int b = (int)(idx / hd), d = (int)(idx % hd);
    int col = pos ? pos[b] : pos_start + b;
    // WP02: a slot plane holds `stride` rows — a row past it would land in the NEXT slot's plane
    // (or past the allocation for the last slot). In-range writes are unchanged.
    if (col < 0 || col >= stride) return;
    int slot = (pos && slot_ids) ? slot_ids[b] : 0;
    keys[((long long)slot * stride + col) * hd + d] = qk[(long long)b * pitch + k_off + d];
}

// Block key (warp-cooperative): mean of the `ratio` raw keys at cache rows `rows[]` (f32 sum / ratio →
// bf16, HF's `.float().mean().to(bf16)`), k_layernorm (f32 from the bf16 mean, ×(1+w), → bf16), RoPE at
// rank `pos_rope` on dims [0, rdim) (f32, → bf16). Lane owns dims [lane*DPL, lane*DPL+DPL); `sm` is a
// per-warp hd-float scratch used for the rotate-half pairing.
__device__ __forceinline__ void qsa_block_key(float* v, const __nv_bfloat16* keys, const long long* rows,
                                              const QsaParams* p, int pos_rope, float* sm, int lane, int slot) {
    const int hd = p->hd, DPL = hd >> 5, rdim = p->rdim, half = rdim >> 1;
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = 0.0f;
    for (int r = 0; r < p->ratio; r++) {
        const __nv_bfloat16* row = keys + rows[r] * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < DPL) v[i] += b2f(row[i]);
    }
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) { v[i] = qsa_bf16r(v[i] / (float)p->ratio); ss += v[i] * v[i]; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float inv = rsqrtf(ss / (float)hd + p->eps);
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) {
        v[i] = qsa_bf16r(v[i] * inv * (1.0f + p->kw[lane * DPL + i]));
        sm[lane * DPL + i] = v[i];
    }
    __syncwarp();
    const float* ct = p->cos_t + (long long)pos_rope * rdim;
    const float* st = p->sin_t + (long long)pos_rope * rdim;
    // VIS-3: the slot's mrope row for this block start (text-only slot: rplen = rdelta = 0 -> unchanged)
    const QsaMrope* mr = p->mr;
    if (mr != nullptr && slot >= 0 && slot < mr->nslots) {
        const long long idx = pos_rope < mr->rplen[slot]
            ? (long long)mr->rmap[(long long)slot * mr->max_pos + pos_rope]
            : (long long)pos_rope + mr->rdelta[slot];
        if (idx < mr->max_pos) { ct = p->cos_t + idx * rdim; st = p->sin_t + idx * rdim; }
        else { ct = mr->mcos + (idx - mr->max_pos) * rdim; st = mr->msin + (idx - mr->max_pos) * rdim; }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) {
        const int d = lane * DPL + i;
        if (d < half)       v[i] = qsa_bf16r(sm[d] * ct[d] - sm[d + half] * st[d]);
        else if (d < rdim)  v[i] = qsa_bf16r(sm[d] * ct[d] + sm[d - half] * st[d]);
    }
    __syncwarp();
}

// 2. decode / verify scores: one warp per (column b = blockIdx.y, block j). scores[b*nblk_stride + j] =
//    Σ_h relu(q_h·K_j)/√hd for j < pc/ratio (pc = pos[b]+1, `pos` = LOGICAL positions). q rows are the
//    normed+roped queries at q + b*q_pitch (heads*hd contiguous). geom = stride | q_pitch<<20 | nblk_stride<<40.
extern "C" __global__ void qsa_score_b(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* keys,
                                       const int* pos, const int* slot_ids, const unsigned char* path,
                                       const int* cps, const QsaParams* p, long long geom) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int stride = (int)(geom & 0xFFFFF);
    const int q_pitch = (int)((geom >> 20) & 0xFFFFF);
    const int nblk_stride = (int)((geom >> 40) & 0xFFFFF);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    const int pc = pos[b] + 1;
    const int ratio = p->ratio;
    const int nb = pc / ratio;
    if (j >= nb) return;
    const int pos_start = cps ? cps[b] : pos[0];
    const long long slot_base = (long long)slot_ids[b] * stride;
    const int qslot = slot_ids[b];
    long long rows[8];
    for (int r = 0; r < ratio; r++) rows[r] = slot_base + qsa_col(j * ratio + r, pos_start, path, b);
    float v[4];
    qsa_block_key(v, keys, rows, p, j * ratio, sm + warp * QSA_HD_MAX, lane, qslot);
    const int hd = p->hd, DPL = hd >> 5;
    float s = 0.0f;
    for (int h = 0; h < p->heads; h++) {
        const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)h * hd + lane * DPL;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < DPL) dot += v[i] * b2f(qrow[i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        s += fmaxf(dot, 0.0f);
    }
    if (lane == 0) {
        s = s / sqrtf((float)hd);
        if (s == 0.0f) s = 0.0f;              // never -0.0: the radix select keys on the float bits
        scores[(long long)b * nblk_stride + j] = s;
    }
}

// 3. prefill block keys: blocks[j] for j < nblk from the slot's committed raw keys (identity ranks).
extern "C" __global__ void qsa_block_keys_b(__nv_bfloat16* blocks, const __nv_bfloat16* keys, const QsaParams* p, int nblk,
                                            int slot /* VIS-3: mrope slot (-1 = none) */) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= nblk) return;
    long long rows[8];
    for (int r = 0; r < p->ratio; r++) rows[r] = (long long)j * p->ratio + r;
    float v[4];
    qsa_block_key(v, keys, rows, p, j * p->ratio, sm + warp * QSA_HD_MAX, lane, slot);
    const int DPL = p->hd >> 5;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) blocks[(long long)j * p->hd + lane * DPL + i] = f2b(v[i]);
}

// Pooled-key plane (2026-09-26, exl3 long context): plane[slot][j][hd] bf16 = qsa_block_key over block j's
// raw keys. qsa_block_key bf16-rounds every output element (qsa_bf16r), so the bf16 store is EXACT and a
// score read from the plane equals qsa_score_b's on-the-fly recompute bit for bit. The plane is maintained
// incrementally (only blocks the step touches); scoring then reads 1 key per block instead of re-pooling
// ratio raw keys (mean, norm, rope) per block, per row, per step — O(context) work that was 1.8 ms per
// verify at 32K and ~7 ms at 128K.
// (a) decode/verify: one warp per row b refreshes the block containing pos[b] (rows sharing a block write
//     identical values). geom = stride(max_pos) | nblk_stride<<40 (qsa_score_b's packing).
extern "C" __global__ void qsa_pool_update_b(__nv_bfloat16* plane, const __nv_bfloat16* keys, const int* pos,
                                              const int* slot_ids, const QsaParams* p, long long geom, int m) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * (blockDim.x >> 5) + warp;
    if (b >= m) return;
    const int stride = (int)(geom & 0xFFFFF);
    const int nblk_stride = (int)((geom >> 40) & 0xFFFFF);
    const int ratio = p->ratio;
    const int j = pos[b] / ratio;
    if (j >= nblk_stride) return;
    const long long slot = slot_ids[b];
    long long rows[8];
    for (int r = 0; r < ratio; r++) rows[r] = slot * stride + (long long)j * ratio + r;
    float v[4];
    qsa_block_key(v, keys, rows, p, j * ratio, sm + warp * QSA_HD_MAX, lane, (int)slot);
    const int DPL = p->hd >> 5;
    __nv_bfloat16* dst = plane + ((long long)slot * nblk_stride + j) * p->hd + lane * DPL;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) dst[i] = f2b(v[i]);
}
// (b) prefill: blocks [j0, j1) of ONE slot (plane_slot/keys_slot are that slot's bases) — qsa_block_keys_b's
//     body over a block range.
extern "C" __global__ void qsa_pool_range_b(__nv_bfloat16* plane_slot, const __nv_bfloat16* keys_slot,
                                             const QsaParams* p, int j0, int j1,
                                             int slot /* VIS-3: mrope slot (-1 = none) */) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = j0 + blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= j1) return;
    long long rows[8];
    for (int r = 0; r < p->ratio; r++) rows[r] = (long long)j * p->ratio + r;
    float v[4];
    qsa_block_key(v, keys_slot, rows, p, j * p->ratio, sm + warp * QSA_HD_MAX, lane, slot);
    const int DPL = p->hd >> 5;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) plane_slot[(long long)j * p->hd + lane * DPL + i] = f2b(v[i]);
}
// (c) decode/verify scores from the plane: qsa_score_b with the block key READ, not recomputed.
extern "C" __global__ void qsa_score_pooled_b(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* plane,
                                               const int* pos, const int* slot_ids, const QsaParams* p, long long geom) {
    const int q_pitch = (int)((geom >> 20) & 0xFFFFF);
    const int nblk_stride = (int)((geom >> 40) & 0xFFFFF);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    const int pc = pos[b] + 1;
    const int nb = pc / p->ratio;
    if (j >= nb) return;
    const int hd = p->hd, DPL = hd >> 5;
    const __nv_bfloat16* kb = plane + ((long long)slot_ids[b] * nblk_stride + j) * hd + lane * DPL;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = (i < DPL) ? b2f(kb[i]) : 0.0f;
    float s = 0.0f;
    for (int h = 0; h < p->heads; h++) {
        const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)h * hd + lane * DPL;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < DPL) dot += v[i] * b2f(qrow[i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        s += fmaxf(dot, 0.0f);
    }
    if (lane == 0) {
        s = s / sqrtf((float)hd);
        if (s == 0.0f) s = 0.0f;
        scores[(long long)b * nblk_stride + j] = s;
    }
}
// (d) WP04 (PLAN/SURPASS_PLAN_2026-09-26.md LC2): bucket-free persistent twin of (c). (c) ran grid
//     (bucket/8, m) with the bucket = next_pow2 >= 1024 of the step's largest position, so every
//     power-of-two context bucket was a separate CUDA graph and half the CTAs of a bucket exited at
//     once; each of its m*nb warps re-read its row's q (4 heads x 256 B) and its block key. Here the
//     grid is FIXED (gridDim.x = 4 CTAs/SM (v2; was 2), gridDim.y = ceil(m / QSA_MR_ROWS)), the per-row block
//     bound nb_b = (pos[b]+1)/ratio is read ON DEVICE (a graph captured at any context replays at
//     every context), the rows' q are staged once per CTA in smem as f32 (b2f is exact), and each
//     warp loads a 256-B pooled key ONCE for all rows of the group that share its slot (with a
//     one-ahead prefetch of its next block's key). Per (row, block) the arithmetic is (c)'s, in (c)'s
//     order, pinned against contraction: lane owns dims [lane*DPL, +DPL), the DPL-step FMA partial
//     from +0, the xor tree 16..1, s += fmaxf(dot, 0) over heads ascending, s / sqrtf(hd), the -0
//     fix — so every written score is BIT-IDENTICAL to (c)'s (GB10_EXL3_SCORE_XCHECK=1 diffs them).
//     Rows write only j < nb_b, exactly the entries (c) writes. dyn smem = rows_in_group*heads*hd*4.
#define QSA_MR_ROWS 8
__device__ __forceinline__ void qsa_mr_ldkey(float* v, const __nv_bfloat16* plane, long long slot, int j,
                                             int nblk_stride, int hd, int DPL, int lane) {
    const __nv_bfloat16* kb = plane + (slot * nblk_stride + j) * hd + lane * DPL;
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = (i < DPL) ? b2f(kb[i]) : 0.0f;
}
// (d') WP04-v2 fast body for the served indexer geometry (hd = 128 -> DPL = 4, heads = 4). The
//      generic body below measured 43 us/call at m=6 @32K for a 2 MB plane read (issue/latency-
//      bound, not byte-bound) and was 37% SLOWER than (c) at m=1: per (row, block) it runs the 4
//      heads SEQUENTIALLY, each a 5-shuffle xor tree on one dependent chain (20 SHFL), plus 4
//      LDS.128 of q. Here a warp iteration takes a PAIR of blocks (j0, j0+1):
//       - key: one 8-B vector load per lane per block (4 bf16 -> f32 through the same b2f), the
//         warp's next pair prefetched;
//       - the 4 heads' FFMA partials are independent chains, and the 4 xor trees fold into ONE
//         6-shuffle butterfly (qsa_mr_red4): each stage still adds, per head, exactly the lane^off
//         partner's partial of THAT head, so each head's total is bit-identical to its own 16..1
//         tree — it just ends in the 8-lane group of that head (lanes of a xor tree all end on the
//         same value: IEEE add is commutative);
//       - 3 shuffles gather the 4 per-head maxima of j0 into lane 0 and of j0+1 into lane 1, which
//         run (c)'s head sum (from +0, heads ascending), IEEE s / sqrtf(hd), the -0 fix and the
//         store — one division stream for two blocks. q (4 x LDS.128 per row) is amortised over
//         the pair. SHFL per (row, block): 7.5 (was 20).
//      Every written score stays BIT-IDENTICAL to (c) (GB10_EXL3_SCORE_XCHECK=1 diffs them); set
//      GB10_EXL3_SCORE_MR=generic to run the generic body (geom bit 62) for A/B.
__device__ __forceinline__ void qsa_mr_ld4v(float (&v)[4], const __nv_bfloat16* plane, long long slot, int j,
                                            int nblk_stride, int lane) {
    // hd = 128: the lane's 4 bf16 start at element (slot*nblk_stride + j)*128 + lane*4 -> 8-B aligned
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(plane + (slot * nblk_stride + j) * 128 + lane * 4));
    v[0] = b2f(__ushort_as_bfloat16((unsigned short)(w.x & 0xFFFFu)));
    v[1] = b2f(__ushort_as_bfloat16((unsigned short)(w.x >> 16)));
    v[2] = b2f(__ushort_as_bfloat16((unsigned short)(w.y & 0xFFFFu)));
    v[3] = b2f(__ushort_as_bfloat16((unsigned short)(w.y >> 16)));
}
// (c)'s per-head lane partial: the FFMA chain from +0 over the lane's 4 dims, ascending.
__device__ __forceinline__ float qsa_mr_dot4(const float (&v)[4], const float4 q) {
    float d = __fmaf_rn(v[0], q.x, 0.0f);
    d = __fmaf_rn(v[1], q.y, d);
    d = __fmaf_rn(v[2], q.z, d);
    return __fmaf_rn(v[3], q.w, d);
}
// Four xor-tree totals in 6 shuffles. Stage 16: a lane keeps heads {2hi, 2hi+1} and adds the
// lane^16 partner's partials of THOSE heads (the partner keeps the other two and sends these);
// stage 8: it keeps head 2hi+b3 the same way; stages 4/2/1 are the plain tree inside the 8-lane
// group. Returns head ((lane >> 3) & 3)'s total.
__device__ __forceinline__ float qsa_mr_red4(float d0, float d1, float d2, float d3, bool hi, bool b3) {
    const float a = __fadd_rn(hi ? d2 : d0, __shfl_xor_sync(0xffffffffu, hi ? d0 : d2, 16));
    const float b = __fadd_rn(hi ? d3 : d1, __shfl_xor_sync(0xffffffffu, hi ? d1 : d3, 16));
    float x = __fadd_rn(b3 ? b : a, __shfl_xor_sync(0xffffffffu, b3 ? a : b, 8));
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 4));
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    return x;
}
__device__ __forceinline__ void qsa_mr_fast4(float* scores, const __nv_bfloat16* plane, const float* sq,
                                             const int* s_nb, const int* s_slot, int nr, int nbmax, int r0,
                                             int nblk_stride, float sqhd) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int pstride = gridDim.x * nwarps;
    const int npairs = (nbmax + 1) >> 1;
    const bool hi = (lane & 16) != 0, b3 = (lane & 8) != 0;
    const int odd = lane & 1;                            // lane 0 finishes block j0, lane 1 block j0+1
    const long long slot0 = s_slot[0];
    int pj = blockIdx.x * nwarps + warp;
    float n0[4], n1[4];                                  // the warp's next pair (row 0's slot)
    #pragma unroll
    for (int i = 0; i < 4; i++) { n0[i] = 0.0f; n1[i] = 0.0f; }
    if (pj < npairs) {
        qsa_mr_ld4v(n0, plane, slot0, 2 * pj, nblk_stride, lane);
        if (2 * pj + 1 < nbmax) qsa_mr_ld4v(n1, plane, slot0, 2 * pj + 1, nblk_stride, lane);
    }
    for (; pj < npairs; pj += pstride) {
        const int j0 = 2 * pj;
        float k0[4], k1[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) { k0[i] = n0[i]; k1[i] = n1[i]; }
        const int pn = pj + pstride;                     // prefetch the next pair of row 0's slot
        if (pn < npairs) {
            qsa_mr_ld4v(n0, plane, slot0, 2 * pn, nblk_stride, lane);
            if (2 * pn + 1 < nbmax) qsa_mr_ld4v(n1, plane, slot0, 2 * pn + 1, nblk_stride, lane);
            else {
                #pragma unroll
                for (int i = 0; i < 4; i++) n1[i] = 0.0f;
            }
        }
        long long cur = slot0;
        for (int r = 0; r < nr; r++) {
            const int nbr = s_nb[r];
            if (j0 >= nbr) continue;                     // warp-uniform: (c)'s `j >= nb` exit
            const long long sl = s_slot[r];
            if (sl != cur) {
                qsa_mr_ld4v(k0, plane, sl, j0, nblk_stride, lane);
                if (j0 + 1 < nbmax) qsa_mr_ld4v(k1, plane, sl, j0 + 1, nblk_stride, lane);
                cur = sl;
            }
            const float* qr = sq + r * 512 + lane * 4;  // [row][head][128] f32
            const float4 qa = *reinterpret_cast<const float4*>(qr);
            const float4 qb = *reinterpret_cast<const float4*>(qr + 128);
            const float4 qc = *reinterpret_cast<const float4*>(qr + 256);
            const float4 qd = *reinterpret_cast<const float4*>(qr + 384);
            const float t0 = qsa_mr_red4(qsa_mr_dot4(k0, qa), qsa_mr_dot4(k0, qb),
                                         qsa_mr_dot4(k0, qc), qsa_mr_dot4(k0, qd), hi, b3);
            const float t1 = qsa_mr_red4(qsa_mr_dot4(k1, qa), qsa_mr_dot4(k1, qb),
                                         qsa_mr_dot4(k1, qc), qsa_mr_dot4(k1, qd), hi, b3);
            const float snd = odd ? fmaxf(t1, 0.0f) : fmaxf(t0, 0.0f);
            const float g0 = __shfl_sync(0xffffffffu, snd, odd);        // head 0 (lanes 0-7)
            const float g1 = __shfl_sync(0xffffffffu, snd, 8 + odd);    // head 1 (lanes 8-15)
            const float g2 = __shfl_sync(0xffffffffu, snd, 16 + odd);   // head 2
            const float g3 = __shfl_sync(0xffffffffu, snd, 24 + odd);   // head 3
            const int j = j0 + odd;
            if (lane < 2 && j < nbr) {
                float s = __fadd_rn(0.0f, g0);
                s = __fadd_rn(s, g1);
                s = __fadd_rn(s, g2);
                s = __fadd_rn(s, g3);
                s = s / sqhd;
                if (s == 0.0f) s = 0.0f;                 // never -0.0 (the radix select keys on the bits)
                scores[(long long)(r0 + r) * nblk_stride + j] = s;
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(256, 4)
qsa_score_pooled_mr(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* plane,
                    const int* pos, const int* slot_ids, const QsaParams* p, long long geom, int m) {
    extern __shared__ float qsa_mr_sq[];                 // [nr][heads][hd] f32
    __shared__ int s_nb[QSA_MR_ROWS], s_slot[QSA_MR_ROWS];
    const int q_pitch = (int)((geom >> 20) & 0xFFFFF);
    const int nblk_stride = (int)((geom >> 40) & 0xFFFFF);
    const int hd = p->hd, heads = p->heads, ratio = p->ratio, DPL = hd >> 5;
    const int r0 = blockIdx.y * QSA_MR_ROWS;
    const int nr = min(QSA_MR_ROWS, m - r0);
    if (nr <= 0) return;                                 // block-uniform
    const int qd = heads * hd;
    for (int i = threadIdx.x; i < nr * qd; i += blockDim.x) {
        const int r = i / qd, e = i - r * qd;
        qsa_mr_sq[i] = b2f(q[(long long)(r0 + r) * q_pitch + e]);
    }
    if (threadIdx.x < nr) {
        s_nb[threadIdx.x] = (pos[r0 + threadIdx.x] + 1) / ratio;
        s_slot[threadIdx.x] = slot_ids[r0 + threadIdx.x];
    }
    __syncthreads();
    int nbmax = 0;
    for (int r = 0; r < nr; r++) nbmax = max(nbmax, s_nb[r]);
    if (hd == 128 && heads == 4 && !((geom >> 62) & 1)) {
        qsa_mr_fast4(scores, plane, qsa_mr_sq, s_nb, s_slot, nr, nbmax, r0, nblk_stride, sqrtf((float)hd));
        return;
    }
    // generic body (any hd <= 128 / heads; GB10_EXL3_SCORE_MR=generic forces it)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wstride = gridDim.x * (blockDim.x >> 5);
    const long long slot0 = s_slot[0];
    int j =blockIdx.x * (blockDim.x >> 5) + warp;
    float vpre[4];
    if (j < nbmax) qsa_mr_ldkey(vpre, plane, slot0, j, nblk_stride, hd, DPL, lane);
    for (; j < nbmax; j += wstride) {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) v[i] = vpre[i];
        const int jn = j + wstride;                      // prefetch the next block of row 0's slot
        if (jn < nbmax) qsa_mr_ldkey(vpre, plane, slot0, jn, nblk_stride, hd, DPL, lane);
        long long cur = slot0;
        for (int r = 0; r < nr; r++) {
            if (j >= s_nb[r]) continue;                  // warp-uniform: (c)'s `j >= nb` exit
            const long long sl = s_slot[r];
            if (sl != cur) { qsa_mr_ldkey(v, plane, sl, j, nblk_stride, hd, DPL, lane); cur = sl; }
            const float* qr = qsa_mr_sq + r * qd + lane * DPL;
            float s = 0.0f;
            for (int h = 0; h < heads; h++) {
                float qv[4];
                if (DPL == 4) {
                    const float4 q4 = *reinterpret_cast<const float4*>(qr + h * hd);
                    qv[0] = q4.x; qv[1] = q4.y; qv[2] = q4.z; qv[3] = q4.w;
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) qv[i] = (i < DPL) ? qr[h * hd + i] : 0.0f;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++) if (i < DPL) dot = __fmaf_rn(v[i], qv[i], dot);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
                s = __fadd_rn(s, fmaxf(dot, 0.0f));
            }
            if (lane == 0) {
                s = s / sqrtf((float)hd);
                if (s == 0.0f) s = 0.0f;                 // never -0.0 (the radix select keys on the bits)
                scores[(long long)(r0 + r) * nblk_stride + j] = s;
            }
        }
    }
}

// 4. prefill scores: QSA_PF_QT queries (positions pos_start+t, their normed+roped q in smem as f32)
//    against one block per thread; scores[t*nblk_stride + j] for j < (pos_start+t+1)/ratio.
//    grid (ceil(nblk/256), ceil(n/QT)), block 256, smem QT*heads*hd floats.
extern "C" __global__ void qsa_score_prefill_b(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* blocks,
                                               const QsaParams* p, int pos_start, int n, int nblk_stride, int q_pitch) {
    extern __shared__ float sq[];
    const int hd = p->hd, heads = p->heads, ratio = p->ratio;
    const int qd = heads * hd;
    const int t0 = blockIdx.y * QSA_PF_QT;
    for (int i = threadIdx.x; i < QSA_PF_QT * qd; i += blockDim.x) {
        const int qq = i / qd, e = i % qd;
        const int t = t0 + qq;
        sq[i] = (t < n) ? b2f(q[(long long)t * q_pitch + e]) : 0.0f;
    }
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nblk_stride) return;
    // Only blocks some query of this tile can see.
    const int tlast = min(t0 + QSA_PF_QT - 1, n - 1);
    if (j >= (pos_start + tlast + 1) / ratio) return;
    float acc[QSA_PF_QT][QSA_HEADS_MAX];
    #pragma unroll
    for (int qq = 0; qq < QSA_PF_QT; qq++)
        #pragma unroll
        for (int h = 0; h < QSA_HEADS_MAX; h++) acc[qq][h] = 0.0f;
    const __nv_bfloat16* krow = blocks + (long long)j * hd;
    for (int d = 0; d < hd; d += 4) {
        const float k0 = b2f(krow[d]), k1 = b2f(krow[d + 1]), k2 = b2f(krow[d + 2]), k3 = b2f(krow[d + 3]);
        #pragma unroll
        for (int qq = 0; qq < QSA_PF_QT; qq++) {
            #pragma unroll
            for (int h = 0; h < QSA_HEADS_MAX; h++) if (h < heads) {   // compile-time indices: acc stays in registers
                const float4 qv = *reinterpret_cast<const float4*>(sq + (qq * heads + h) * hd + d);
                acc[qq][h] += k0 * qv.x + k1 * qv.y + k2 * qv.z + k3 * qv.w;
            }
        }
    }
    #pragma unroll
    for (int qq = 0; qq < QSA_PF_QT; qq++) {
        const int t = t0 + qq;
        if (t >= n) break;
        if (j >= (pos_start + t + 1) / ratio) continue;
        float s = 0.0f;
        #pragma unroll
        for (int h = 0; h < QSA_HEADS_MAX; h++) if (h < heads) s += fmaxf(acc[qq][h], 0.0f);
        s = s / sqrtf((float)hd);
        if (s == 0.0f) s = 0.0f;
        scores[(long long)t * nblk_stride + j] = s;
    }
}

// Block-wide exclusive scan of two int flags (1024 threads = 32 warps). Returns the exclusive prefix
// of each and the block totals.
__device__ __forceinline__ void qsa_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc, int* wa, int* wc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ia = a, ic = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int xa = __shfl_up_sync(0xffffffffu, ia, off), xc = __shfl_up_sync(0xffffffffu, ic, off);
        if (lane >= off) { ia += xa; ic += xc; }
    }
    if (lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    int ba = 0, bc = 0, sa = 0, sc = 0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; w++) { if (w < warp) { ba += wa[w]; bc += wc[w]; } sa += wa[w]; sc += wc[w]; }
    pa = ba + ia - a; pc = bc + ic - c; ta = sa; tc = sc;
    __syncthreads();
}

// 5. top-k + selection list. One 1024-thread block per column b. pc = pos ? pos[b]+1 : pos_start+b+1;
//    nb = pc/ratio blocks, k = min(topk, nb). Radix select (4 passes, MSB first) over the non-negative
//    score bits finds the k-th largest key T and how many key==T blocks to keep (`kp`); the stable
//    compaction keeps every block > T and the first kp blocks == T in block order, emitting their
//    ratio ranks (→ cache columns via qsa_col) ascending, then the tail ranks [nb*ratio, pc).
//    sel row pitch = p->sel_max; pos_sel[b] = nsel-1 (the splitk/reduce "pos" of the selection).
extern "C" __global__ void __launch_bounds__(1024)
qsa_topk_b(int* sel, int* pos_sel, const float* scores, const int* pos, const unsigned char* path,
           const int* cps, const QsaParams* p, int nblk_stride, int pos_start_chain) {
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos ? pos[b] + 1 : pos_start_chain + b + 1;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    const int pos_start = pos ? (cps ? cps[b] : pos[0]) : 0;
    const float* sc = scores + (long long)b * nblk_stride;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K) {
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int j = tid; j < nb; j += blockDim.x) {
                const unsigned key = __float_as_uint(sc[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0, bin = 0, rem = kp;
                for (int i = 255; i >= 0; i--) {
                    const int c = hist[i];
                    if (cum + c >= kp) { bin = i; rem = kp - cum; break; }
                    cum += c;
                }
                s_bin = bin; s_kp = rem;
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            __syncthreads();
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0;
    int* srow = sel + (long long)b * sel_max;
    for (int base = 0; base < nb; base += blockDim.x) {
        const int j = base + tid;
        const bool valid = j < nb;
        const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
        const bool gt = valid && key > T;
        const bool eq = valid && key == T;
        int p_sel0, p_eq, t_sel, t_eq;
        // first scan: equal-rank (needed to decide selection), second: selection rank
        qsa_exscan2(eq ? 1 : 0, 0, p_eq, p_sel0, t_eq, t_sel, wa, wc);
        const bool selected = gt || (eq && (run_eq + p_eq) < kp);
        int p_s, p_d, t_s, t_d;
        qsa_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) {
            const int r = run_sel + p_s;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = qsa_col(j * ratio + i, pos_start, path, b);
        }
        run_sel += t_s; run_eq += t_eq;
    }
    const int nsel = run_sel * ratio;
    for (int i = tid; i < tail; i += blockDim.x) srow[nsel + i] = qsa_col(nb * ratio + i, pos_start, path, b);
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// 6. gqa_attn_sel_splitk — gqa_attn_splitk with the key set replaced by the column's selection list:
//    pc = pos_sel[b]+1 keys, rank r → cache column sel[b*sel_max + r]. Split structure, warp stride,
//    reduction order and merge are the dense kernel's, verbatim: an identity list is bit-identical to
//    gqa_attn_splitk, and gqa_attn_reduce (with pos = pos_sel) finishes it unchanged.
extern "C" __global__ void gqa_attn_sel_splitk(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos_sel, long long bs_packed, int nh_packed, const int* slot_ids,
    const int* sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos_sel[b] + 1;
    const int slot = slot_ids[b];
    const int* srow = sel + (long long)b * sel_max;

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

    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r = start + warp; r < end; r += NW) {
        const int t = srow[r];
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

// 6b. gqa_attn_sel_splitk_k8v4 — gqa_attn_sel_splitk over the k8v4 packed cache: K rows of 20 B/16
//     (int8 codes + fp16 block scale), V rows of 12 B/16 (q4 nibbles + e4m3 block scale). The row
//     dequant is gqa_attn_splitk_k8v4's, the key set is the column's selection list; split structure,
//     warp stride, reduction order and merge are the bf16 kernel's verbatim, so gqa_attn_reduce (with
//     pos = pos_sel) finishes it unchanged and decode == verify col-0 holds as for the dense pair.
extern "C" __global__ void gqa_attn_sel_splitk_k8v4(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos_sel, long long bs_packed, int nh_packed, const int* slot_ids,
    const int* sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos_sel[b] + 1;
    const int slot = slot_ids[b];
    const int* srow = sel + (long long)b * sel_max;

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

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows: 20 B/16
    const int v_rb = KVQ_ROW_BYTES(hd);              // V rows: 12 B/16
    const int lane_blk = (lane * DPL) / KVQ_BLK;     // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;     // its first code within the block (4-aligned)
    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 12;
    for (int r = start + warp; r < end; r += NW) {
        const int t = srow[r];
        const unsigned char* krow = kb + (long long)t * k_rb;
        const float ksc = __half2float(__ushort_as_half(*(const unsigned short*)(krow + KVQ_BLK)));
        const uint32_t* kcodes = (const uint32_t*)(krow + lane_off);   // 4 aligned int8 codes
        float kdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) kdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const uint32_t codes = kcodes[u];
            kdq[u * 4 + 0] = (float)((int8_t)(codes >> 0)) * ksc;
            kdq[u * 4 + 1] = (float)((int8_t)(codes >> 8)) * ksc;
            kdq[u * 4 + 2] = (float)((int8_t)(codes >> 16)) * ksc;
            kdq[u * 4 + 3] = (float)((int8_t)(codes >> 24)) * ksc;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * kdq[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const unsigned char* vrow = vb + (long long)t * v_rb;
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

// 7. gqa_attn_sel_prefill — causal prefill attention over per-query selection lists (one warp per
//    query, its whole softmax in registers, gqa_attn_prefill's arithmetic). Prefill is outside the
//    batch-invariance contract, so no split structure is needed. k/v pointers are the SLOT base.
//    grid (ceil(N/8) * nh), block 256.
extern "C" __global__ void gqa_attn_sel_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh_packed, float scale,
    int N, const int* sel, const int* pos_sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int QT = blockDim.x >> 5;
    const int blk = blockIdx.x;
    const int tile = blk / nh, qh = blk % nh;
    const int kvh = qh / (nh / nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;
    const int t = tile * QT + warp;
    if (t >= N) return;
    const int pc = pos_sel[t] + 1;
    const int* srow = sel + (long long)t * sel_max;

    const __nv_bfloat16* qrow = q + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
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
    for (int r = 0; r < pc; r++) {
        const int tt = srow[r];
        const __nv_bfloat16* krow = kb + (long long)tt * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)tt * hd;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
        m = m_new;
        l = l * a_old + a_cur;
    }
    __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) orow[i] = f2b(l > 0.0f ? acc[i] / l : 0.0f);
}

// 8. raw-key compaction twin of compact_kv_b (an accepted tree path moved to contiguous columns).
extern "C" __global__ void qsa_compact_b(__nv_bfloat16* keys, __nv_bfloat16* scratch, const int* src_pos,
                                         int len, int pos_start, int slot, int stride, int hd, int dir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len * hd) return;
    int k = idx / hd, dv = idx % hd;
    long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    long long coff = ((long long)slot * stride + cache_pos) * hd + dv;
    long long soff = (long long)k * hd + dv;
    if (dir == 0) scratch[soff] = keys[coff]; else keys[coff] = scratch[soff];
}

// 4b. prefill score combine after the cuBLAS head GEMMs: G[h][t*nblk + j] = (q_{t,h} · K_j)/√hd
//     (alpha folded in), scores[t*nblk + j] = Σ_h relu(G) for j < (pos_start+t+1)/ratio.
extern "C" __global__ void qsa_score_combine_b(float* scores, const float* G, const QsaParams* p,
                                               int pos_start, int nch, int nblk) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nch * nblk) return;
    const int t = (int)(idx / nblk), j = (int)(idx % nblk);
    if (j >= (pos_start + t + 1) / p->ratio) return;
    const long long hs = (long long)nch * nblk;
    float s = 0.0f;
    for (int h = 0; h < p->heads; h++) s += fmaxf(G[h * hs + idx], 0.0f);
    if (s == 0.0f) s = 0.0f;
    scores[idx] = s;
}

// 7b. gqa_attn_sel_prefill2 — the selected-list prefill attention, restructured for gathered rows:
//     one warp per (query, kv head, group of SEL_GS query heads) so a K/V row is fetched once per
//     group instead of once per head, and SEL_KU keys in flight per iteration (the selected rows have
//     no locality — the one-key loop of gqa_attn_sel_prefill was latency-bound: 250 ms/layer on a
//     7.3K prompt). Per-query arithmetic and order are unchanged (keys ascending, same lane dot /
//     xor-tree / online softmax), so the output is bit-identical to gqa_attn_sel_prefill.
//     hd <= 256. grid (ceil(N/8) * nkv * ngroups), block 256.
#define SEL_GS 4
#define SEL_KU 4
extern "C" __global__ void __launch_bounds__(256) gqa_attn_sel_prefill2(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh_packed, float scale,
    int N, const int* sel, const int* pos_sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int G = nh / nkv;
    const int ngroups = (G + SEL_GS - 1) / SEL_GS;
    const int QT = blockDim.x >> 5;
    const int blk = blockIdx.x;
    const int tile = blk / (nkv * ngroups);
    const int rem = blk % (nkv * ngroups);
    const int kvh = rem / ngroups, grp = rem % ngroups;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;                    // <= 8
    const int t = tile * QT + warp;
    if (t >= N) return;
    const int pc = pos_sel[t] + 1;
    const int* srow = sel + (long long)t * sel_max;
    const int h0 = grp * SEL_GS;
    const int nhg = min(SEL_GS, G - h0);

    float qv[SEL_GS][8];
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) {
        const int qh = kvh * G + h0 + g;
        const __nv_bfloat16* qrow = q + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 8; i++) qv[g][i] = (g < nhg && i < DPL) ? b2f(qrow[i]) : 0.0f;
    }
    float m[SEL_GS], l[SEL_GS], acc[SEL_GS][8];
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) { m[g] = -1e30f; l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[g][i] = 0.0f; }

    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r0 = 0; r0 < pc; r0 += SEL_KU) {
        float kr[SEL_KU][8], vr[SEL_KU][8];
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++) {
            const int r = min(r0 + u, pc - 1);          // clamp: surplus lanes recompute the last key, masked below
            const int tt = srow[r];
            const __nv_bfloat16* krow = kb + (long long)tt * hd;
            const __nv_bfloat16* vrow = vb + (long long)tt * hd;
            #pragma unroll
            for (int i = 0; i < 8; i++) { kr[u][i] = (i < DPL) ? b2f(krow[i]) : 0.0f; vr[u][i] = (i < DPL) ? b2f(vrow[i]) : 0.0f; }
        }
        float s[SEL_KU][SEL_GS];
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++)
            #pragma unroll
            for (int g = 0; g < SEL_GS; g++) {
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) d += qv[g][i] * kr[u][i];
                s[u][g] = d;
            }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            #pragma unroll
            for (int u = 0; u < SEL_KU; u++)
                #pragma unroll
                for (int g = 0; g < SEL_GS; g++) s[u][g] += __shfl_xor_sync(0xffffffffu, s[u][g], off);
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++) {
            if (r0 + u >= pc) break;
            #pragma unroll
            for (int g = 0; g < SEL_GS; g++) {
                const float sc = s[u][g] * scale;
                const float m_new = fmaxf(m[g], sc);
                const float a_old = __expf(m[g] - m_new), a_cur = __expf(sc - m_new);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = acc[g][i] * a_old + a_cur * vr[u][i];
                m[g] = m_new;
                l[g] = l[g] * a_old + a_cur;
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) {
        if (g >= nhg) break;
        const int qh = kvh * G + h0 + g;
        __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 8; i++) if (i < DPL) orow[i] = f2b(l[g] > 0.0f ? acc[g][i] / l[g] : 0.0f);
    }
}

