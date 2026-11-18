// exl3_bench/18_draft_slice_qsa.cuh: pruned draft lm_head slices, DF2 tap conversion, QSA sparse-attention prep/score/select kernels.
// Sparse attention for the EXL3 path (PLAN/S_A3_I_ATTENTION_DECODE.md).
// Section 18 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// Pruned draft lm_head (reference patch 0002, EXL3_MTP_HEAD_N): keep the first
// nb_keep 16-column trellis blocks of every k-block. bw = words per (kb, nb)
// block = 16*BITS. One thread per word.
extern "C" __global__ void xq_slice_trellis_u16(unsigned short* __restrict__ dst,
                                                const unsigned short* __restrict__ src,
                                                int nb_full, int nb_keep, long long bw) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_kb = (long long)nb_keep * bw;
    long long kb = i / per_kb;
    long long off = i % per_kb;
    long long nb2 = off / bw, w = off % bw;
    dst[i] = src[kb * (long long)nb_full * bw + nb2 * bw + w];
}

// TP-H #4 (vocab-parallel lm_head): the same column slice as xq_slice_trellis_u16 but starting at
// 16-column block nb_off (rank * nb_keep) instead of 0. The lmh GEMM's columns are bit-independent
// of the tile partition (each 128-column tile is computed alone), so a shard with a 128-multiple
// column offset produces bitwise the replicated head's columns [nb_off*16, (nb_off+nb_keep)*16).
extern "C" __global__ void xq_slice_trellis_u16_off(unsigned short* __restrict__ dst,
                                                    const unsigned short* __restrict__ src,
                                                    int nb_full, int nb_keep, long long bw, int nb_off) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_kb = (long long)nb_keep * bw;
    long long kb = i / per_kb;
    long long off = i % per_kb;
    long long nb2 = off / bw, w = off % bw;
    dst[i] = src[kb * (long long)nb_full * bw + (nb2 + nb_off) * bw + w];
}

// =====================================================================
// S-A3-i: QSA sparse attention for the EXL3 path (see PLAN/S_A3_I_ATTENTION_DECODE.md).
//
// Native QSA sparsity for the xq decode path (f16 projections, f32 KV cache):
// a per-query selection list of token positions (ascending, -1 padded) is
// produced elsewhere; these kernels cover (a) f16->bf16 indexer staging,
// (b) indexer-q norm+rope, (c) radix top-k selection with ASCENDING identity
// emission, (d) K/V cache write + roped-q f32 staging, (e) gathered
// flash-decode split-K online softmax over the selection list, (f) split
// merge + gate. Math transcribed from gpu_batch.cu's bf16 QSA section
// (QsaParams / qsa_topk_b / rmsnorm_rope_b) and the xq_attn_decode family
// (shared xq_k_norm_rope, softmax scale, final gate expression). exl3_bench.cu
// is a separate translation unit: QsaParamsX and the qsa helpers are
// file-local copies (no cross-TU static references), xq_-prefixed.
// =====================================================================

#include <cuda_bf16.h>

// ---- file-local f32 <-> bf16 converters (this TU otherwise has no bf16):
// bit manipulation, round-to-nearest-even; NaN quieted (sign kept), inf and
// denormals pass through the truncation unchanged.
__device__ __forceinline__ __nv_bfloat16 xq_f2b(float v) {
    unsigned u = __float_as_uint(v);
    if ((u << 1) > 0xFF000000u)                  // NaN: exp all ones + mantissa != 0
        u = (u & 0x80000000u) | 0x7FC00000u;     // canonical quiet NaN, sign kept
    else
        u += 0x7FFFu + ((u >> 16) & 1u);         // RNE on the truncated low 16 bits
    __nv_bfloat16_raw r;
    r.x = (unsigned short)(u >> 16);
    return __nv_bfloat16(r);
}
__device__ __forceinline__ float xq_b2f(__nv_bfloat16 v) {
    __nv_bfloat16_raw r(v);
    return __uint_as_float((unsigned)r.x << 16);
}

// ---- 1. f16 -> bf16 elementwise staging copy. out[i] = bf16(f32(in[i]));
// grid-stride, block 256.
extern "C" __global__ void xq_qsa_h2b(__nv_bfloat16* __restrict__ out, const __half* __restrict__ in,
                                      long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long gs = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += gs) out[i] = xq_f2b(xq_h2f(in[i]));
}

// ---- 1b. EXL27C-DF2 (fwd): the DFlash2 tap capture for the EXL3 f32 residual staging.
//
// The NVFP4 trunk taps hidden_states[l+1] with a plain bf16->bf16 2D D2D
// (dflash2::capture::capture_cols_into). The EXL3 forward stages its residual as
// f32, token-major [rows][pitch] with pitch = hcn*h (xq_hc_norm: stream s of token
// b lives at b*hcn*h + s*h; xq_hc_inject writes every stream), so the capture must
// BOTH down-cast f32->bf16 AND stride over the hcn*h row pitch. cuMemcpy2D cannot
// convert element types, so the capture is this kernel instead (the documented
// choice; see the exl27c_fwd report).
//
// FLASHNEXT-DF2-B (2026-10-23): the tap is the MEAN OVER THE hc STREAMS of the token's
// row — elements [r*pitch + s*h + c] for s in [0, hc) — not stream 0. The DFlash2
// drafters are trained on "completed, zero-based layer outputs, averaged over the four
// hyper-connection streams" (the artifact's own SGLang patch, qwen38-dflash2-capture.patch:
// hidden_states.unflatten(-1, (hc_count, hidden)).mean(dim=-2)). On the hc_count=1 packs
// (27B, Ornith) the mean of one stream IS the stream (acc/1.0f is exact) — bit-identical
// to the old stream-0 read; on the hc_count=4 qwen4_exp trunk (Flash-Next) stream 0 alone
// was a different feature from the one the drafter learned (acceptance 1.41/round).
//
// dst is the sink staging, row-major [cols][TAP_CONCAT_DIM] (the same bytes
// capture_cols_into produces): token r's tap-li slice lands at
// dst[r*dst_pitch + tap_li*h + c] = bf16(mean_s src[r*pitch + s*h + c]).
//
// Grid-stride over n*h elements, block 256. One launch per tap layer per round,
// stream-ordered on the compute stream (AGENTS §2.3), fixed addresses (the sink's
// staging is allocated once per process) — CUDA-graph-safe. Zero cost when the sink
// is unarmed (the caller never launches).
extern "C" __global__ void xq_df2_tap_cvt(__nv_bfloat16* __restrict__ dst,
                                          const float* __restrict__ src,
                                          long long n_elems, int h, long long pitch,
                                          int tap_li, long long dst_pitch, int hc) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long gs = (long long)gridDim.x * blockDim.x;
    for (; i < n_elems; i += gs) {
        const long long r = i / h;
        const int c = (int)(i % h);
        const float* row = src + r * pitch + c;
        // ascending stream order, f32 accumulate, one bf16 rounding (torch mean semantics)
        float acc = row[0];
        for (int s = 1; s < hc; ++s) acc += row[(long long)s * h];
        dst[r * dst_pitch + (long long)tap_li * h + c] = xq_f2b(acc / (float)hc);
    }
}

// ---- 1c. EXL27C-DF2 (fwd): boot-time tiled f16 transpose [K, N] -> [N, K].
//
// One step of the one-shot df2 head materialization (FwdModel::df2_boot_ptrs): the
// EXL3 reconstruct (exl3_decode_dump) yields W_dec [K=hidden, N=vocab] f16 row-major;
// the DF2 round's bf16 head contract is [vocab, hidden] row-major. 32x32
// shared-memory tiles (padded +1 to avoid bank conflicts), guard-remnant edges.
// BOOT-ONLY (once per process, ~2x2.4 GB traffic); never on a serving round.
#define XQ_DF2_TT 32
extern "C" __global__ void xq_df2_transpose_f16(__half* __restrict__ out, const __half* __restrict__ in,
                                                int k, int n) {
    XQ_PDL_ENTRY();
    __shared__ __half tile[XQ_DF2_TT][XQ_DF2_TT + 1];
    const int k0 = blockIdx.x * XQ_DF2_TT;
    const int n0 = blockIdx.y * XQ_DF2_TT;
    const int tx = threadIdx.x, ty = threadIdx.y;
    if (k0 + ty < k && n0 + tx < n) {
        // coalesced read along n: thread (tx, ty) reads in[k0+ty][n0+tx]
        tile[ty][tx] = in[(size_t)(k0 + ty) * n + (n0 + tx)];
    }
    __syncthreads();
    if (n0 + ty < n && k0 + tx < k) {
        // coalesced write along k: thread (tx, ty) writes out[n0+ty][k0+tx] = tile[tx][ty]
        // (out row n0+ty, col k0+tx — consecutive tx write consecutive k-columns).
        out[(size_t)(n0 + ty) * k + (k0 + tx)] = tile[tx][ty];
    }
}

// ---- 2. indexer-q fused per-head RMSNorm + RoPE, rmsnorm_rope_b's math on
// the bf16 indexer tensor (IN PLACE). qk rows [B][heads+1][hdx] (heads q rows
// + 1 k row per token); w = q_layernorm [hdx] f32. UNLIKE rmsnorm_rope_b's
// separate cos/sin tables, cossin rows are [B][2*rdim] INTERLEAVED cos|sin
// (same layout the xq_attn_decode family uses). Grid (B*heads), block hdx
// (=128; static smem sized to the indexer head-dim max).
extern "C" __global__ void xq_qsa_qnorm_rope(__nv_bfloat16* __restrict__ qk, const float* __restrict__ w,
                                             const float* __restrict__ cossin, int heads, int hdx,
                                             int rdim, int B, float eps) {
    XQ_PDL_ENTRY();
    (void)B;                                     // grid is B*heads; b derives from blockIdx
    const int b = blockIdx.x / heads;
    const int h = blockIdx.x % heads;
    const int tid = threadIdx.x;
    const long long base = (long long)b * (heads + 1) * hdx + (long long)h * hdx;
    __shared__ float s[128];
    float v = (tid < hdx) ? xq_b2f(qk[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    const float inv = rsqrtf(s[0] / (float)hdx + eps);
    if (tid < hdx) qk[base + tid] = xq_f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    const int half = rdim / 2;
    if (tid < half) {
        const long long cb = (long long)b * 2 * rdim + tid;
        const float x1 = xq_b2f(qk[base + tid]);
        const float x2 = xq_b2f(qk[base + tid + half]);
        const float c = cossin[cb], sn = cossin[cb + rdim];
        qk[base + tid] = xq_f2b(x1 * c - x2 * sn);
        qk[base + tid + half] = xq_f2b(x2 * c + x1 * sn);
    }
}

// ---- QsaParams (gpu_batch.cu) mirrored field-for-field as the file-local
// QsaParamsX: the 64-B device struct built once per indexer at load, carrying
// what would not fit the 12-argument launch cap.
struct QsaParamsX {
    const float* cos_t;     // [max_pos][rdim] (duplicated halves, see build_rope_tables)
    const float* sin_t;
    const float* kw;        // k_layernorm weight [hd] (gemma style: scale = 1 + w)
    const float* qw;        // q_layernorm weight [hd] (unused on device; the q norm runs kernel 2)
    float eps;
    int hd;                 // indexer head dim (128; 32 on the tiny model)
    int heads;              // indexer query heads (4)
    int rdim;               // rotary dims (64; 16 on the tiny model)
    int ratio;              // compress ratio (4)
    int topk;               // block budget = indexer_budget / ratio (512)
    int sel_max;            // indexer_budget + ratio - 1 (2051): row pitch of the selection lists
    int pad;
    const void* mr;         // VIS-3: QsaParams.mr (per-slot indexer mrope; read only by gpu_batch.cu)
};

#define XQ_QSA_MAX_VERIFY 16   // mirrors gpu_batch.cu MAX_VERIFY (path table row stride)

// rank -> cache column, qsa_col's rule, file-local copy. path == nullptr =>
// identity: the S-A3-i list is raw token positions, no verify tree.
__device__ __forceinline__ int xq_qsa_col(int rank, int pos_start, const unsigned char* path, int b) {
    if (!path || rank < pos_start) return rank;
    return pos_start + (int)path[b * XQ_QSA_MAX_VERIFY + (rank - pos_start)];
}

// Block-wide exclusive scan of two int flags (1024 threads = 32 warps). Returns
// the exclusive prefix of each and the block totals. qsa_exscan2 verbatim.
__device__ __forceinline__ void xq_qsa_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc,
                                               int* wa, int* wc) {
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

// ---- 3. top-k block select + ASCENDING selection list. One 1024-thread block
// per row b; pc = pos[b]+1, nb = pc/ratio blocks, k = min(topk, nb). Radix
// select (4 passes, MSB first, score bits) — qsa_topk_b's logic verbatim:
// score DESCENDING, ties -> LOWEST block index, so the kept set is a pure
// function of the scores. Emission is SIMPLER than qsa_topk_b: an ascending
// sweep over j emits each selected block's ratio token columns with the
// IDENTITY mapping (no path/cps re-ranking), so the list is plain ascending
// token positions; the tail ranks [nb*ratio, pc) ride after. select_all != 0:
// every complete block is selected. sel rows [B][sel_max] are PRE-FILLED with
// -1 by the caller — only emitted entries are written here. pos_sel[b] =
// out-1 (the splitk "pos" of the selection, as in qsa_topk_b).
extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    // Score row read IN PLACE from global (qsa_topk_b's pattern): staging it in
    // __shared__ float sc[8192] claimed "nb <= nblk_stride <= 8192" — false:
    // nblk_stride = max_pos/ratio = 65536, so any pc > 32768 overflowed shared.
    // The row is L2-resident across the radix passes + emission sweep.
    const float* sc = scores + (long long)b * nblk_stride;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    // S-A3-i review fix: full selection only when it fits the list (a >sel_max
    // probe would overflow the sel row); the Rust side passes 0 unconditionally.
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
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
    // ---- ascending emission sweep: selected j (in block order) emit their
    // ratio columns at the running selection rank — ascending by construction.
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
        xq_qsa_exscan2(eq ? 1 : 0, 0, p_eq, p_sel0, t_eq, t_sel, wa, wc);
        const bool selected = select_all ? valid : (gt || (eq && (run_eq + p_eq) < kp));
        int p_s, p_d, t_s, t_d;
        xq_qsa_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) {
            const int r = (run_sel + p_s) * ratio;
            for (int i = 0; i < ratio; i++) srow[r + i] = xq_qsa_col(j * ratio + i, 0, nullptr, b);
        }
        run_sel += t_s; run_eq += t_eq;
    }
    // ---- tail: the pc % ratio most recent tokens (identity, ascending)
    const int nsel = run_sel * ratio;
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += blockDim.x) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3b. WP14 stage 1 (PLAN/SURPASS_PLAN_2026-09-26.md): xq_qsa_topk_asc with the same output
// bits and cheaper steps. The kept set is a pure function of the scores (descending, ties -> lowest
// block index) and the list is emitted ascending, so ANY exact implementation writes the same sel
// row and pos_sel (GB10_EXL3_TOPK_XCHECK=1 diffs them against 3.):
//  (a) histogram increments are warp-aggregated: the MSB passes pile nearly every block into one
//      or two bins (all scores share sign/exponent), i.e. up to 1024-way shared-atomic conflicts.
//      A warp whose predicated lanes share one bin adds popc(ballot) once; otherwise
//      __match_any_sync groups equal bins and each group's leader adds its popc. Same counts.
//  (b) the bucket search is warp-parallel: lane L owns bins 255-8L..248-8L (descending), a shfl
//      scan gives each lane the count above its bins, the one lane holding the crossing walks its
//      8 bins — the serial tid-0 walk's bin and remainder (integer arithmetic, exact).
//  (c) the emission sweep does ONE ballot/popc scan per 1024-block chunk instead of two block
//      exscans: equal-key blocks are selected in ascending j order until kp is used up, so with
//      rem = max(0, kp - run_eq) a block's rank is run_sel + #gt-before + min(#eq-before, rem).
//      Warp totals are double-buffered by chunk parity -> one __syncthreads per chunk.
// Same launch contract as 3. (grid m, block 1024).
extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc2(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                 const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                 int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wgt[2][32], weq[2][32];
    const unsigned FULL = 0xffffffffu;
    const int b = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    const float* sc = scores + (long long)b * nblk_stride;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K && !select_all) {                         // select_all ignores T/kp (as in 3.)
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int base = warp * 32; base < nb; base += blockDim.x) {   // warp-uniform trip count
                const int j = base + lane;
                const bool valid = j < nb;
                const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
                const bool pred = valid && ((key & mask) == prefix);
                const unsigned bin = (key >> shift) & 255u;
                const unsigned pm = __ballot_sync(FULL, pred);
                if (pm == 0u) continue;
                const int src = __ffs(pm) - 1;
                const unsigned b0 = __shfl_sync(FULL, bin, src);
                if (__all_sync(FULL, !pred || bin == b0)) {
                    if (lane == src) atomicAdd(&hist[b0], __popc(pm));
                } else {
                    const unsigned peers = __match_any_sync(FULL, pred ? bin : 0x100u);
                    if (pred && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
                }
            }
            __syncthreads();
            if (warp == 0) {
                int c[8], lsum = 0;
                #pragma unroll
                for (int t = 0; t < 8; t++) { c[t] = hist[255 - 8 * lane - t]; lsum += c[t]; }
                int incl = lsum;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, incl, off);
                    if (lane >= off) incl += x;
                }
                const int excl = incl - lsum;
                const bool here = excl < kp && incl >= kp;
                const unsigned hb = __ballot_sync(FULL, here);
                if (here) {
                    int cum = excl, bin = 0, rem = kp;
                    #pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (cum + c[t] >= kp) { bin = 255 - 8 * lane - t; rem = kp - cum; break; }
                        cum += c[t];
                    }
                    s_bin = bin; s_kp = rem;
                }
                if (hb == 0u && lane == 0) { s_bin = 0; s_kp = kp; }   // the serial walk's defaults
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            // s_bin/s_kp/hist are next written after the next pass's first barrier, which every
            // thread reaches only after these reads — no third barrier needed.
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0, par = 0;
    int* srow = sel + (long long)b * sel_max;
    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < nb; base += blockDim.x, par ^= 1) {
        const int j = base + tid;
        const bool valid = j < nb;
        const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
        const bool gt = select_all ? valid : (valid && key > T);
        const bool eq = !select_all && valid && key == T;
        const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
        if (lane == 0) { wgt[par][warp] = __popc(bg); weq[par][warp] = __popc(be); }
        __syncthreads();
        int g = (lane < nwarps) ? wgt[par][lane] : 0;
        int e = (lane < nwarps) ? weq[par][lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int wg = __shfl_sync(FULL, g, (warp + 31) & 31), we = __shfl_sync(FULL, e, (warp + 31) & 31);
        if (warp == 0) { wg = 0; we = 0; }
        const int pg = wg + __popc(bg & lt), pe = we + __popc(be & lt);
        const int rem = max(0, kp - run_eq);
        if (gt || (eq && pe < rem)) {
            const int r = (run_sel + pg + min(pe, rem)) * ratio;
            for (int i = 0; i < ratio; i++) srow[r + i] = j * ratio + i;
        }
        run_sel += tg + min(te, rem);
        run_eq += te;
    }
    const int nsel = run_sel * ratio;
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += blockDim.x) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3b2. A5-K3 (attn.qsa_select, PLAN/notes_2026-09-27/K3.md): xq_qsa_topk_asc2's output bits with
// a latency-lean schedule. Same launch contract (grid m, block 1024, no dynamic smem, sel rows
// pre-filled with -1 by the caller, only entries [0, pos_sel] + pos_sel written).
// Why asc2 is slow at long context: 4 radix passes + an emission sweep over the row, each element
// paying warp-collective ballots/shuffles/syncs, on ONE SM per row (55-67 us/call at 128K in-round,
// p6d ledger #5a). The kept set is a pure function of the scores under the strict order (key desc,
// block index asc) with key = the score's f32 BITS read as u32 (asc2's order for ANY bits: -0,
// negatives and NaNs rank by their u32 value exactly as there), and the list is emitted ascending —
// so (T, kp) = (the K-th largest key, K - #keys > T) is unique and any exact select writes the same
// entries and pos_sel. The schedule (per row, 16-B loads, per-thread work, no per-element ballots):
//  (0) all blocks kept (select_all, nb <= K) or every key equal (min == max: the K lowest indices)
//      -> the list is the identity prefix; no histogram.
//  (a) min/max of the row: the bits above the highest bit where they differ are common to every
//      key, so the first digit starts right below them.
//  (b) 11-bit digits MSB first; per-thread shared atomics (a quad whose 4 keys share a bin adds 4
//      once); the bucket search is a block scan over the bins in DESCENDING order whose crossing
//      thread returns the serial walk's bin and remainder (integer arithmetic).
//  (c) as soon as the crossing bin holds <= XQ_TK3_CAP keys, ONE more pass compacts every key >= the
//      bin's low edge (= the bin + the < K keys above it) with its block index into shared memory;
//      the remaining digits and the whole emission then run on that list: keys > T and == T are
//      marked in two block bitmaps, equal keys are taken in ASCENDING index until kp is used up
//      (asc2's tie rule), and a block scan over the selected bitmap words emits the blocks in
//      ascending order at their ranks. The list order is not deterministic; nothing downstream of it
//      depends on order (counts and bitmaps only).
//  (d) fallback (a bin never fits — adversarial ties — or K > XQ_TK3_KMAX, or nb > 64 x blockDim):
//      the remaining digits run over the global row and the emission is a two-pass warp-segment
//      scan (rank = #gt-before + min(#eq-before, kp)). Exact, just slower.
#define XQ_TK3_NBIN 2048
#define XQ_TK3_CAP  4096
#define XQ_TK3_KMAX 512
#define XQ_TK3_LIST (XQ_TK3_KMAX + XQ_TK3_CAP)
#define XQ_TK3_U    4

// block-wide inclusive scan (blockDim.x = 32 * nw, nw <= 32); total -> every thread
__device__ __forceinline__ int xq_tk3_bscan(int v, int* wsum, int& total) {
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int incl = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, incl, off); if (lane >= off) incl += x; }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    const int w = lane < nw ? wsum[lane] : 0;
    int wi = w;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, wi, off); if (lane >= off) wi += x; }
    total = __shfl_sync(FULL, wi, 31);
    const int wex = __shfl_sync(FULL, wi - w, warp);
    __syncthreads();                                             // wsum is reused by the next scan
    return incl + wex;
}

// the 4 keys of quad q (blocks 4q..4q+3); lanes past nb read 0 (callers mask by index)
__device__ __forceinline__ uint4 xq_tk3_ld4(const unsigned* __restrict__ sc, int q, int nb, bool vec) {
    const int j = 4 * q;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (vec) {
        if (j < nb) r = __ldg(reinterpret_cast<const uint4*>(sc) + q);
    } else {
        if (j < nb) r.x = __ldg(sc + j);
        if (j + 1 < nb) r.y = __ldg(sc + j + 1);
        if (j + 2 < nb) r.z = __ldg(sc + j + 2);
        if (j + 3 < nb) r.w = __ldg(sc + j + 3);
    }
    return r;
}
__device__ __forceinline__ unsigned xq_tk3_c(const uint4& v, int c) {
    return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}
// the lowest `need` set bits of x (need consumed)
__device__ __forceinline__ unsigned xq_tk3_take(unsigned x, int& need) {
    if (need <= 0) return 0u;
    const int c = __popc(x);
    if (need >= c) { need -= c; return x; }
    unsigned r = 0u;
    for (int k = 0; k < need; k++) { const unsigned lsb = x & (0u - x); r |= lsb; x ^= lsb; }
    need = 0;
    return r;
}

extern "C" __global__ void __launch_bounds__(1024)
xq_qsa_topk_asc3(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ scores,
                 const int* __restrict__ pos, const struct QsaParamsX* __restrict__ p,
                 int nblk_stride, int select_all) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned smap[2 * XQ_TK3_NBIN];    // hist | (eq bitmap, sel bitmap)
    __shared__ unsigned lkey[XQ_TK3_LIST];
    __shared__ unsigned short lidx[XQ_TK3_LIST];
    __shared__ int wsum[32], wgt[32], weq[32];
    __shared__ unsigned wlo[32], whi[32];
    __shared__ int s_bin, s_kp, s_cnt, s_n;
    int* hist = reinterpret_cast<int*>(smap);
    const unsigned FULL = 0xffffffffu;
    const int b = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, nt = blockDim.x, nw = nt >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos[b] + 1;
    const float* rowf = scores + (long long)b * nblk_stride;
    const unsigned* __restrict__ sc = reinterpret_cast<const unsigned*>(rowf);
    const bool vec = (reinterpret_cast<unsigned long long>(rowf) & 15ull) == 0ull;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    if (select_all && nb * ratio + tail > sel_max) select_all = 0;
    int* srow = sel + (long long)b * sel_max;
    int mode = 0, nall = nb, nl = -1;             // 0 identity prefix of nall blocks | 1 list | 2 global
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K && !select_all) {
        // ---- (a) row min / max
        unsigned lo = 0xffffffffu, hi = 0u;
        for (int q0 = tid; 4 * q0 < nb; q0 += XQ_TK3_U * nt) {
            uint4 v[XQ_TK3_U];
            #pragma unroll
            for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, q0 + u * nt, nb, vec);
            #pragma unroll
            for (int u = 0; u < XQ_TK3_U; u++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (4 * (q0 + u * nt) + c < nb) { const unsigned k = xq_tk3_c(v[u], c); lo = min(lo, k); hi = max(hi, k); }
                }
            }
        }
        lo = __reduce_min_sync(FULL, lo); hi = __reduce_max_sync(FULL, hi);
        if (lane == 0) { wlo[warp] = lo; whi[warp] = hi; }
        __syncthreads();
        lo = __reduce_min_sync(FULL, lane < nw ? wlo[lane] : 0xffffffffu);
        hi = __reduce_max_sync(FULL, lane < nw ? whi[lane] : 0u);
        kp = K;
        if (lo == hi) {
            T = hi; nall = K;                        // every key equal: the K lowest indices, ascending
        } else {
            const bool can_list = K <= XQ_TK3_KMAX && nb <= 64 * nt;
            int rb = 32 - __clz(lo ^ hi);            // low bits that still vary (1..32)
            unsigned M = rb >= 32 ? 0u : (0xffffffffu << rb);
            unsigned P = hi & M;
            while (true) {
                const int db = rb < 11 ? rb : 11, shift = rb - db, nbin = 1 << db;
                const unsigned dm = (unsigned)(nbin - 1);
                for (int i = tid; i < nbin; i += nt) hist[i] = 0;
                __syncthreads();                     // every thread is past its last s_* read
                if (tid == 0) { s_bin = 0; s_kp = kp; s_cnt = 0; }   // the serial walk's defaults
                if (nl < 0) {
                    for (int q0 = tid; 4 * q0 < nb; q0 += XQ_TK3_U * nt) {
                        uint4 v[XQ_TK3_U];
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, q0 + u * nt, nb, vec);
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) {
                            const int j0 = 4 * (q0 + u * nt);
                            unsigned bn[4]; bool pr[4];
                            #pragma unroll
                            for (int c = 0; c < 4; c++) {
                                const unsigned k = xq_tk3_c(v[u], c);
                                pr[c] = j0 + c < nb && (k & M) == P;
                                bn[c] = (k >> shift) & dm;
                            }
                            if (pr[0] && pr[1] && pr[2] && pr[3] && bn[0] == bn[1] && bn[0] == bn[2] && bn[0] == bn[3]) {
                                atomicAdd(&hist[bn[0]], 4);
                            } else {
                                #pragma unroll
                                for (int c = 0; c < 4; c++) if (pr[c]) atomicAdd(&hist[bn[c]], 1);
                            }
                        }
                    }
                } else {
                    for (int i = tid; i < nl; i += nt) {
                        const unsigned k = lkey[i];
                        if ((k & M) == P) atomicAdd(&hist[(k >> shift) & dm], 1);
                    }
                }
                __syncthreads();
                {   // thread t owns bins nbin-1-2t, nbin-2-2t (descending); nbin <= 2 * blockDim.x
                    const int b0 = nbin - 1 - 2 * tid, b1 = b0 - 1;
                    const int c0 = b0 >= 0 ? hist[b0] : 0, c1 = b1 >= 0 ? hist[b1] : 0;
                    int tot;
                    const int incl = xq_tk3_bscan(c0 + c1, wsum, tot);
                    const int excl = incl - c0 - c1;
                    if (excl < kp && incl >= kp) {
                        if (excl + c0 >= kp) { s_bin = b0; s_kp = kp - excl; s_cnt = c0; }
                        else                 { s_bin = b1; s_kp = kp - excl - c0; s_cnt = c1; }
                    }
                }
                __syncthreads();
                const int bin = s_bin, cnt = s_cnt;
                kp = s_kp;
                P |= (unsigned)bin << shift;
                M |= dm << shift;
                rb = shift;
                if (can_list && nl < 0 && cnt <= XQ_TK3_CAP) {
                    // ---- (c) compact every key >= P (the crossing bin + the K - kp keys above it)
                    __syncthreads();                 // the s_* reads above precede s_n's reset
                    if (tid == 0) s_n = 0;
                    __syncthreads();
                    for (int base = warp * 32; 4 * base < nb; base += XQ_TK3_U * nt) {   // warp-uniform trips
                        uint4 v[XQ_TK3_U];
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) v[u] = xq_tk3_ld4(sc, base + lane + u * nt, nb, vec);
                        #pragma unroll
                        for (int u = 0; u < XQ_TK3_U; u++) {
                            const int j0 = 4 * (base + lane + u * nt);
                            int c = 0;
                            #pragma unroll
                            for (int e = 0; e < 4; e++) c += (j0 + e < nb && xq_tk3_c(v[u], e) >= P) ? 1 : 0;
                            if (!__any_sync(FULL, c > 0)) continue;
                            int incl = c;
                            #pragma unroll
                            for (int off = 1; off < 32; off <<= 1) { const int x = __shfl_up_sync(FULL, incl, off); if (lane >= off) incl += x; }
                            int o = 0;
                            if (lane == 31) o = atomicAdd(&s_n, incl);
                            o = __shfl_sync(FULL, o, 31) + incl - c;
                            #pragma unroll
                            for (int e = 0; e < 4; e++) {
                                const unsigned k = xq_tk3_c(v[u], e);
                                if (j0 + e < nb && k >= P && o < XQ_TK3_LIST) { lkey[o] = k; lidx[o] = (unsigned short)(j0 + e); o++; }
                            }
                        }
                    }
                    __syncthreads();
                    nl = min(s_n, XQ_TK3_LIST);
                }
                if (rb == 0) break;
            }
            T = P;
            mode = nl >= 0 ? 1 : 2;
        }
    }
    int nsel;
    if (mode == 0) {
        for (int t = tid; t < nall * ratio; t += nt) srow[t] = t;
        nsel = nall * ratio;
    } else if (mode == 1) {
        // ---- (c) bitmap selection + ascending emission from the list
        unsigned* eqm = smap;
        unsigned* selm = smap + XQ_TK3_NBIN;
        const int W = (nb + 31) >> 5;
        __syncthreads();                             // hist / s_* reads are done
        for (int w = tid; w < W; w += nt) { eqm[w] = 0u; selm[w] = 0u; }
        __syncthreads();
        for (int i = tid; i < nl; i += nt) {
            const unsigned k = lkey[i];
            const int j = lidx[i];
            if (k > T) atomicOr(&selm[j >> 5], 1u << (j & 31));
            else if (k == T) atomicOr(&eqm[j >> 5], 1u << (j & 31));
        }
        __syncthreads();
        const int w0 = 2 * tid, w1 = w0 + 1;
        const unsigned e0 = w0 < W ? eqm[w0] : 0u, e1 = w1 < W ? eqm[w1] : 0u;
        int etot;
        const int ei = xq_tk3_bscan(__popc(e0) + __popc(e1), wsum, etot);
        int need = kp - (ei - __popc(e0) - __popc(e1));          // equal keys still to take before my words
        const unsigned s0 = (w0 < W ? selm[w0] : 0u) | xq_tk3_take(e0, need);
        const unsigned s1 = (w1 < W ? selm[w1] : 0u) | xq_tk3_take(e1, need);
        int stot;
        const int si = xq_tk3_bscan(__popc(s0) + __popc(s1), wsum, stot);
        int r = si - __popc(s0) - __popc(s1);
        for (unsigned x = s0; x; x &= x - 1u) {
            const int j = w0 * 32 + __ffs(x) - 1;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = j * ratio + i;
            r++;
        }
        for (unsigned x = s1; x; x &= x - 1u) {
            const int j = w1 * 32 + __ffs(x) - 1;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = j * ratio + i;
            r++;
        }
        nsel = stot * ratio;
    } else {
        // ---- (d) fallback emission over warp segments (scalar loads; any alignment)
        const int S = (((nb + nw - 1) / nw) + 31) & ~31;
        const int g0s = min(nb, warp * S), g1s = min(nb, g0s + S);
        int cg = 0, ce = 0;
        for (int base = g0s; base < g1s; base += 32) {
            const int j = base + lane;
            const bool valid = j < g1s;
            const unsigned k = valid ? __ldg(sc + j) : 0u;
            cg += __popc(__ballot_sync(FULL, valid && k > T));
            ce += __popc(__ballot_sync(FULL, valid && k == T));
        }
        if (lane == 0) { wgt[warp] = cg; weq[warp] = ce; }
        __syncthreads();
        int g = lane < nw ? wgt[lane] : 0, e = lane < nw ? weq[lane] : 0;
        const int gs = g, es = e;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int rg = __shfl_sync(FULL, g - gs, warp), re = __shfl_sync(FULL, e - es, warp);
        const unsigned lt = (1u << lane) - 1u;
        for (int base = g0s; base < g1s; base += 32) {
            const int j = base + lane;
            const bool valid = j < g1s;
            const unsigned k = valid ? __ldg(sc + j) : 0u;
            const bool gt = valid && k > T, eq = valid && k == T;
            const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
            const int pg = rg + __popc(bg & lt), pe = re + __popc(be & lt);
            if (gt || (eq && pe < kp)) {
                const int r = (pg + min(pe, kp)) * ratio;
                for (int i = 0; i < ratio; i++) srow[r + i] = j * ratio + i;
            }
            rg += __popc(bg); re += __popc(be);
        }
        nsel = (tg + min(te, kp)) * ratio;
    }
    const int t0 = nb * ratio;
    for (int i = tid; i < ratio; i += nt) {
        const int t = t0 + i;
        if (t <= pos[b]) srow[nsel + i] = t;
    }
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// ---- 3c. WP18 rung A (PLAN/SURPASS_PLAN_2026-09-26.md): bounded PREFILL selection, bitwise.
// The served prefill scored every (row, block) of a chunk into a [c][max_pos/ratio] f32 matrix
// (0.54 GB at a 262K window) with gpu_batch.cu's qsa_score_prefill_b (one block per thread, 8
// rows x 4 heads per thread, q float4 re-read from smem for every 4 FFMAs: ~13% of FMA peak), then
// ran xq_qsa_topk_asc over each sparse row. WP18 streams the blocks in tiles of WP18_T and never
// holds more than a [rows][WP18_T] slab:
//   xq_wp18_score  — the SAME per-(row, head, block) arithmetic (a),
//   xq_wp18_merge  — per row: carried top-K list  U  the tile  ->  new top-K (b); on the row's
//                    last tile it emits the ascending selection list + tail + pos_sel (c).
// (a) Scores. qsa_score_prefill_b's PTX, per accumulator and 4-dim group d (ascending):
//       t = k[d+1]*q[d+1]; t = fma(k[d], q[d], t); t = fma(k[d+2], q[d+2], t);
//       t = fma(k[d+3], q[d+3], t); acc = acc + t                       (mul/fma.rn/add, acc from +0)
//     then s = (((0 + max(a0,0)) + max(a1,0)) + max(a2,0)) + max(a3,0); s = s / sqrt(128); -0 -> +0.
//     Here every one of those steps is an explicit round-to-nearest intrinsic (nothing left for the
//     compiler or the driver JIT to contract or reorder), on the same bf16 -> f32 exact inputs, so
//     each score is bit-identical; the register tiling only changes WHICH thread owns an
//     accumulator, never its sequence.
// (b) Selection. qsa_topk_asc keeps the top K = topk blocks of a row under the strict total order
//     (key desc, block index asc) — key = the score's f32 bits (all scores are >= +0). For such an
//     order top-K(A u B) = top-K(top-K(A) u B), so folding tiles in ascending block order with a
//     carried list is exact. And once the list holds K entries with minimum key thr, a tile block
//     with key <= thr is below all K of them (equal key -> the carried block has the lower index),
//     so only keys > thr can enter — the filter is exact too. The merge is asc2's radix select
//     (warp-aggregated histogram, warp-parallel bucket search) and its one-scan ascending emission
//     over the virtual array [carried list (ascending index) | filtered tile (ascending index)].
// (c) Emission: carried list ascending -> each block's `ratio` token columns, then the tail
//     tokens [nb*ratio, pos], pos_sel = count-1 — xq_qsa_topk_asc's exact writes (entries past
//     pos_sel are never touched, as before).
// Shape contract (host-gated): indexer hd 128, 4 heads, topk <= WP18_KMAX <= WP18_T.
#define WP18_T     8192        // blocks per tile = slab row pitch (floats)
#define WP18_TM    32          // query rows per scorer CTA
#define WP18_TN    64          // blocks per scorer CTA
#define WP18_DK    32          // head dims per smem stage
#define WP18_HD    128
#define WP18_NH    4
#define WP18_KMAX  512         // carried-list capacity (topk = budget / ratio = 512 served)
#define WP18_MT    512         // merge threads per row

// 8 bf16 (one 16-B load, little-endian pairs) -> 8 f32: exact (b2f is the same bit shift).
__device__ __forceinline__ void wp18_bf8(float4& lo, float4& hi, const uint4 v) {
    lo.x = __uint_as_float(v.x << 16); lo.y = __uint_as_float(v.x & 0xFFFF0000u);
    lo.z = __uint_as_float(v.y << 16); lo.w = __uint_as_float(v.y & 0xFFFF0000u);
    hi.x = __uint_as_float(v.z << 16); hi.y = __uint_as_float(v.z & 0xFFFF0000u);
    hi.z = __uint_as_float(v.w << 16); hi.w = __uint_as_float(v.w & 0xFFFF0000u);
}
// One 4-dim group of one accumulator: qsa_score_prefill_b's contracted expression, pinned.
__device__ __forceinline__ float wp18_step(float acc, const float4 k, const float4 q) {
    float t = __fmul_rn(k.y, q.y);
    t = __fmaf_rn(k.x, q.x, t);
    t = __fmaf_rn(k.z, q.z, t);
    t = __fmaf_rn(k.w, q.w, t);
    return __fadd_rn(acc, t);
}

// Scorer. Grid (ceil(tile blocks / WP18_TN), ceil(nrows / WP18_TM)), block 256, static smem 48 KB
// (2 stages x (q 32 rows x 4 heads x 32 dims + k 64 blocks x 32 dims) f32). Thread (ty, tx) of a
// 16 x 16 grid owns rows ty, ty+16 x blocks tx+16j (j < 4) x all 4 heads = 32 accumulators. The
// 128 head dims stream in 4 stages of 32 through a register prefetch + double-buffered smem (one
// barrier per stage); bf16 -> f32 once at staging. float4 columns are XOR-swizzled (q by row&1,
// k by block&7) so the per-d-group LDS.128 reads are conflict-free (1 / 2 wavefronts).
// slab[r][jj] = score(row r, block kt0 + jj) for block < nb_r = (pos_first + r + 1) / ratio;
// other slab entries are left untouched. q: row r at q + r * q_pitch (bf16, 4 heads x 128 first).
extern "C" __global__ void __launch_bounds__(256, 2)
xq_wp18_score(float* __restrict__ slab, const __nv_bfloat16* __restrict__ q,
              const __nv_bfloat16* __restrict__ plane, const struct QsaParamsX* __restrict__ p,
              int pos_first, int nrows, int kt0, int q_pitch) {
    XQ_PDL_ENTRY();
    constexpr int QROW = WP18_NH * WP18_DK;                // floats per q row per stage (128)
    __shared__ __align__(16) float sq[2][WP18_TM * QROW];  // 2 x 16 KB
    __shared__ __align__(16) float sk[2][WP18_TN * WP18_DK]; // 2 x 8 KB
    const int ratio = p->ratio;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.y * WP18_TM;
    const int jj0 = blockIdx.x * WP18_TN;
    const int blk0 = kt0 + jj0;
    const int rlast = min(row0 + WP18_TM, nrows) - 1;
    const int nb_last = (pos_first + rlast + 1) / ratio;   // the tile's widest row
    if (rlast < row0 || blk0 >= nb_last) return;           // CTA-uniform: nothing visible
    // staging assignment: q chunks c = tid, tid+256 (row c>>4, head (c>>2)&3, 8-dim part c&3);
    // k chunk tid (block tid>>2, part tid&3).
    // (chunk tid+256 is row +16 of chunk tid: same head/part/parity -> same column, +16 rows.)
    const int qr = tid >> 4, qh = (tid >> 2) & 3, qpart = tid & 3;
    const bool qok0 = row0 + qr < nrows, qok1 = row0 + qr + 16 < nrows;
    const __nv_bfloat16* qsrc = q + (long long)(row0 + qr) * q_pitch + qh * WP18_HD + qpart * 8;
    const long long qstep = 16LL * q_pitch;
    const int qdst = qr * QROW + (((qh * 8 + qpart * 2) ^ (qr & 1)) << 2);   // swizzled float4 pair (lo; hi = ^4)
    const int kb = tid >> 2, kpart = tid & 3;
    const bool kok = blk0 + kb < nb_last;
    const __nv_bfloat16* ksrc = plane + (long long)(blk0 + kb) * WP18_HD + kpart * 8;
    const int kdst = kb * WP18_DK + (((kpart * 2) ^ (kb & 7)) << 2);          // hi = ^4
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    uint4 pq0, pq1, pk;
    // stage 0 -> buffer 0
    pq0 = qok0 ? *reinterpret_cast<const uint4*>(qsrc) : z4;
    pq1 = qok1 ? *reinterpret_cast<const uint4*>(qsrc + qstep) : z4;
    pk = kok ? *reinterpret_cast<const uint4*>(ksrc) : z4;
    {
        float4 lo, hi;
        wp18_bf8(lo, hi, pq0);
        *reinterpret_cast<float4*>(&sq[0][qdst]) = lo;
        *reinterpret_cast<float4*>(&sq[0][qdst ^ 4]) = hi;
        wp18_bf8(lo, hi, pq1);
        *reinterpret_cast<float4*>(&sq[0][qdst + 16 * QROW]) = lo;
        *reinterpret_cast<float4*>(&sq[0][(qdst ^ 4) + 16 * QROW]) = hi;
        wp18_bf8(lo, hi, pk);
        *reinterpret_cast<float4*>(&sk[0][kdst]) = lo;
        *reinterpret_cast<float4*>(&sk[0][kdst ^ 4]) = hi;
    }
    __syncthreads();
    float acc[2][4][WP18_NH];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int h = 0; h < WP18_NH; h++) acc[i][j][h] = 0.0f;
    const int swq = ty & 1;                                 // (ty + 16) & 1 == ty & 1
    const int swk = tx & 7;                                 // (tx + 16j) & 7 == tx & 7
    constexpr int NST = WP18_HD / WP18_DK;
    #pragma unroll 1
    for (int s = 0; s < NST; s++) {
        const int buf = s & 1;
        if (s + 1 < NST) {                                  // prefetch stage s+1 into registers
            const int off = (s + 1) * WP18_DK;
            pq0 = qok0 ? *reinterpret_cast<const uint4*>(qsrc + off) : z4;
            pq1 = qok1 ? *reinterpret_cast<const uint4*>(qsrc + qstep + off) : z4;
            pk = kok ? *reinterpret_cast<const uint4*>(ksrc + off) : z4;
        }
        const float* sqb = sq[buf];
        const float* skb = sk[buf];
        #pragma unroll 2                                    // full unroll: 128 regs + spills; 2: 119, 0 stack
        for (int g = 0; g < WP18_DK / 4; g++) {             // 4-dim groups, ascending d
            float4 kv[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                kv[j] = *reinterpret_cast<const float4*>(&skb[(tx + 16 * j) * WP18_DK + ((g ^ swk) << 2)]);
            #pragma unroll
            for (int h = 0; h < WP18_NH; h++) {
                const int c4 = (h * 8 + g) ^ swq;
                const float4 q0 = *reinterpret_cast<const float4*>(&sqb[ty * QROW + (c4 << 2)]);
                const float4 q1 = *reinterpret_cast<const float4*>(&sqb[(ty + 16) * QROW + (c4 << 2)]);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[0][j][h] = wp18_step(acc[0][j][h], kv[j], q0);
                    acc[1][j][h] = wp18_step(acc[1][j][h], kv[j], q1);
                }
            }
        }
        if (s + 1 < NST) {                                  // the other buffer: last read at s-1
            float* sqn = sq[buf ^ 1];
            float* skn = sk[buf ^ 1];
            float4 lo, hi;
            wp18_bf8(lo, hi, pq0);
            *reinterpret_cast<float4*>(&sqn[qdst]) = lo;
            *reinterpret_cast<float4*>(&sqn[qdst ^ 4]) = hi;
            wp18_bf8(lo, hi, pq1);
            *reinterpret_cast<float4*>(&sqn[qdst + 16 * QROW]) = lo;
            *reinterpret_cast<float4*>(&sqn[(qdst ^ 4) + 16 * QROW]) = hi;
            wp18_bf8(lo, hi, pk);
            *reinterpret_cast<float4*>(&skn[kdst]) = lo;
            *reinterpret_cast<float4*>(&skn[kdst ^ 4]) = hi;
        }
        __syncthreads();
    }
    const float rs = __fsqrt_rn((float)WP18_HD);
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = row0 + ty + 16 * i;
        if (r >= nrows) continue;
        const int nb = (pos_first + r + 1) / ratio;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int jj = jj0 + tx + 16 * j;
            if (kt0 + jj >= nb) continue;
            float sc = __fadd_rn(0.0f, fmaxf(acc[i][j][0], 0.0f));
            #pragma unroll
            for (int h = 1; h < WP18_NH; h++) sc = __fadd_rn(sc, fmaxf(acc[i][j][h], 0.0f));
            sc = __fdiv_rn(sc, rs);
            if (sc == 0.0f) sc = 0.0f;                      // never -0.0 (the select keys on the bits)
            slab[(long long)r * WP18_T + jj] = sc;
        }
    }
}

// U = [carried list (m entries, ascending block index) | tile (nk entries, filtered)]: element u.
__device__ __forceinline__ void wp18_elem(int u, int m, int nk, bool filt, unsigned thr,
                                          const unsigned* sCk, const float* sS, unsigned& key, bool& valid) {
    if (u < m) { key = sCk[u]; valid = true; return; }
    const int jj = u - m;
    valid = jj < nk;
    key = valid ? __float_as_uint(sS[jj]) : 0u;
    if (filt) valid = valid && key > thr;
}

// Merge. Grid (rows of the group), block WP18_MT, static smem ~37 KB. Row r = pos_first + r;
// tile kt covers blocks [kt*T, kt*T + T). cand_k/cand_i: [rows][topk] carried list (keys = score
// bits, ascending block index), valid for min(topk, kt*T) entries when kt > 0. Rows whose blocks
// end before this tile return (kt == 0 with nb == 0: tail-only list). On the row's LAST tile the
// list is emitted into sel/pos_sel (row pitch sel_max) exactly as xq_qsa_topk_asc writes it.
extern "C" __global__ void __launch_bounds__(WP18_MT)
xq_wp18_merge(int* __restrict__ sel, int* __restrict__ pos_sel, const float* __restrict__ slab,
              unsigned* __restrict__ cand_k, int* __restrict__ cand_i,
              const struct QsaParamsX* __restrict__ p, int pos_first, int kt) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) float sS[WP18_T];
    __shared__ unsigned sCk[WP18_KMAX];
    __shared__ int sCi[WP18_KMAX];
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wgt[2][32], weq[2][32];
    __shared__ unsigned s_red[32];
    __shared__ int s_cnt[32];
    const unsigned FULL = 0xffffffffu;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pos = pos_first + r;
    const int pc = pos + 1;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    const int kT = kt * WP18_T;
    int* srow = sel + (long long)r * sel_max;
    if (nb <= kT) {
        if (kt == 0) {                                     // nb == 0: the tail-only list
            for (int i = tid; i < ratio; i += blockDim.x) if (i <= pos) srow[i] = i;
            if (tid == 0) pos_sel[r] = tail - 1;
        }
        return;
    }
    const int nk = min(nb - kT, WP18_T);
    const bool last = nb <= kT + WP18_T;
    const int m = min(K, kT);                              // carried entries (K whenever kt > 0)
    unsigned* ckr = cand_k + (long long)r * K;
    int* cir = cand_i + (long long)r * K;
    for (int i = tid; i < m; i += blockDim.x) { sCk[i] = ckr[i]; sCi[i] = cir[i]; }
    const float* srcS = slab + (long long)r * WP18_T;
    for (int i = tid * 4; i < nk; i += blockDim.x * 4)     // i + 4 <= WP18_T: whole float4s
        *reinterpret_cast<float4*>(&sS[i]) = *reinterpret_cast<const float4*>(srcS + i);
    __syncthreads();
    // the carried list's K-th largest key (its minimum) — only keys above it can enter
    const bool filt = m == K;
    unsigned thr = 0u;
    if (filt) {                                            // block-uniform
        unsigned mn = 0xFFFFFFFFu;
        for (int i = tid; i < m; i += blockDim.x) mn = min(mn, sCk[i]);
        mn = __reduce_min_sync(FULL, mn);
        if (lane == 0) s_red[warp] = mn;
        __syncthreads();
        for (int w = 0; w < nwarps; w++) thr = w ? min(thr, s_red[w]) : s_red[0];
    }
    int cl = 0;                                            // tile entries that can enter
    for (int i = tid; i < nk; i += blockDim.x) cl += (!filt || __float_as_uint(sS[i]) > thr) ? 1 : 0;
    cl = __reduce_add_sync(FULL, cl);
    if (lane == 0) s_cnt[warp] = cl;
    __syncthreads();
    int cnt = 0;
    for (int w = 0; w < nwarps; w++) cnt += s_cnt[w];
    const int N = m + cnt;                                 // |U| (valid entries)
    if (filt && cnt == 0 && !last) return;                 // carried list unchanged (block-uniform)
    const int U = m + nk;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (N > K) {                                           // asc2's radix select over U
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int base = warp * 32; base < U; base += blockDim.x) {   // warp-uniform trip count
                const int u = base + lane;
                unsigned key; bool valid;
                wp18_elem(u, m, nk, filt, thr, sCk, sS, key, valid);
                const bool pred = valid && ((key & mask) == prefix);
                const unsigned bin = (key >> shift) & 255u;
                const unsigned pm = __ballot_sync(FULL, pred);
                if (pm == 0u) continue;
                const int src = __ffs(pm) - 1;
                const unsigned b0 = __shfl_sync(FULL, bin, src);
                if (__all_sync(FULL, !pred || bin == b0)) {
                    if (lane == src) atomicAdd(&hist[b0], __popc(pm));
                } else {
                    const unsigned peers = __match_any_sync(FULL, pred ? bin : 0x100u);
                    if (pred && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
                }
            }
            __syncthreads();
            if (warp == 0) {
                int c[8], lsum = 0;
                #pragma unroll
                for (int t = 0; t < 8; t++) { c[t] = hist[255 - 8 * lane - t]; lsum += c[t]; }
                int incl = lsum;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const int x = __shfl_up_sync(FULL, incl, off);
                    if (lane >= off) incl += x;
                }
                const int excl = incl - lsum;
                const bool here = excl < kp && incl >= kp;
                const unsigned hb = __ballot_sync(FULL, here);
                if (here) {
                    int cum = excl, bin = 0, rem = kp;
                    #pragma unroll
                    for (int t = 0; t < 8; t++) {
                        if (cum + c[t] >= kp) { bin = 255 - 8 * lane - t; rem = kp - cum; break; }
                        cum += c[t];
                    }
                    s_bin = bin; s_kp = rem;
                }
                if (hb == 0u && lane == 0) { s_bin = 0; s_kp = kp; }
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
        }
        T = prefix;
    }
    // ascending emission over U (asc2's one-scan form): carried entries first (lower indices)
    int run_sel = 0, run_eq = 0, par = 0;
    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < U; base += blockDim.x, par ^= 1) {
        const int u = base + tid;
        unsigned key; bool valid;
        wp18_elem(u, m, nk, filt, thr, sCk, sS, key, valid);
        const bool gt = valid && key > T;
        const bool eq = valid && key == T;
        const unsigned bg = __ballot_sync(FULL, gt), be = __ballot_sync(FULL, eq);
        if (lane == 0) { wgt[par][warp] = __popc(bg); weq[par][warp] = __popc(be); }
        __syncthreads();
        int g = (lane < nwarps) ? wgt[par][lane] : 0;
        int e = (lane < nwarps) ? weq[par][lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int xg = __shfl_up_sync(FULL, g, off), xe = __shfl_up_sync(FULL, e, off);
            if (lane >= off) { g += xg; e += xe; }
        }
        const int tg = __shfl_sync(FULL, g, 31), te = __shfl_sync(FULL, e, 31);
        int wg = __shfl_sync(FULL, g, (warp + 31) & 31), we = __shfl_sync(FULL, e, (warp + 31) & 31);
        if (warp == 0) { wg = 0; we = 0; }
        const int pg = wg + __popc(bg & lt), pe = we + __popc(be & lt);
        const int rem = max(0, kp - run_eq);
        if (gt || (eq && pe < rem)) {
            const int rk = run_sel + pg + min(pe, rem);
            const int idx = (u < m) ? sCi[u] : kT + (u - m);
            if (last) {
                for (int i = 0; i < ratio; i++) srow[rk * ratio + i] = idx * ratio + i;
            } else {
                ckr[rk] = key; cir[rk] = idx;
            }
        }
        run_sel += tg + min(te, rem);
        run_eq += te;
    }
    if (last) {
        const int nsel = run_sel * ratio;
        const int t0 = nb * ratio;
        for (int i = tid; i < ratio; i += blockDim.x) {
            const int t = t0 + i;
            if (t <= pos) srow[nsel + i] = t;
        }
        if (tid == 0) pos_sel[r] = nsel + tail - 1;
    }
}

