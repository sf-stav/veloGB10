// =============================================================================
// kernels/gpu_dsv41/51_idx_common.cuh — leg KI: shared pieces of the indexer kernels (band 50).
//
// The indexer of DeepSeek-V4.1 (inference/model.py `Indexer`, PLAN/DSV41_REF_MAP.md §3.5, §4):
//   q   = rope(wq_b(qr))[32 heads x 128] (RoPE on dims 64..127, YaRN theta 160000), then
//         fp4_act_quant(q, 32, ue8m0): E2M1 codes + a power-of-two scale per 32
//   k   = the index-K cache of the layer's KV source (FP4 E2M1 + E8M0 per 32, 64 + 4 B / row)
//   w   = bf16( bf16(weights_proj(x)) * 2^-6 )            (128^-0.5 * 32^-0.5 = 1/64, exact)
//   s_t = bf16( sum_h  bf16( relu(bf16(q_h . k_t)) * w_h ) )     (fp32 head sum)
//   top-512 of s over the visible positions t < (pos + 1) / ratio, re-sorted into position order.
// Ties: the highest score first, then the LOWEST position (REFb measured torch so; DeepInfer's
// selection is deterministic; X-VL item 1.11). One total order (score desc, position asc) makes a
// key unique, so a chunked / striped selection merged later is bit-identical to the unsplit one.
//
// Numerics of the dot (measured on GB10, PLAN/DSV41_KI_REPORT.md §3): the block-scaled MMA
// mma.m16n8k64.kind::mxf4.block_scale.scale_vec::2X returns RZ(C + exact sum of its two 32-groups);
// with the q / k scale spreads of real data the sums are exact, so the dot equals the reference's
// fp32 einsum; where a sum is not representable the MMA truncates (1 fp32 ulp), which the host
// reference (src/dsv41/gpu/idx_ref.rs) emulates exactly.
//
// The head sum order is fixed (batch invariant): lane t of a quad sums its 8 heads
// (nt*8 + 2t, nt*8 + 2t + 1 for nt = 0..3, in that order), then (t0 + t1) + (t2 + t3).
// =============================================================================
#pragma once

#define D41_IDX_H 32          // index heads
#define D41_IDX_D 128         // index head dim
#define D41_IDX_ROPE_OFF 64   // RoPE on dims 64..127 (the last rope_head_dim = 64)
#define D41_IDX_TOPK 512
#define D41_IDX_CAND_BLOCKS 2048
#define D41_IDX_CAND_BLOCK 8
#define D41_IDX_TILE 128      // keys per tile (16 blocks of 8)
#define D41_IDX_KW 4          // key warps: warp w owns keys [32 (w % 4), + 32) of a tile; w / 4 = row group
#define D41_IDX_CAP 1024      // per-row candidate buffer (u64 keys), power of two

// One row block: rows [row0, row0 + nrows) of the call share one key cache (one lane).
struct d41_idx_rb {
    unsigned long long kq;    // index-K code plane [n_keys_cap][64 B]
    unsigned long long ks;    // index-K scale plane [n_keys_cap][4 B]
    int row0, nrows, pad0, pad1;
};

// The argument block of every indexer entry (passed by value; Rust mirror: gpu/indexer.rs IdxArgs).
struct d41_idx_args {
    unsigned long long iq;        // [rows][iq_ld] bf16 or f32: indexer wq_b output (pre-RoPE)
    unsigned long long w;         // [rows][w_ld] bf16 or f32: weights_proj output (unscaled)
    unsigned long long pos;       // int [rows]: absolute token position of each row
    unsigned long long rbs;       // d41_idx_rb [n_rb]
    unsigned long long sel;       // int [rows][512]: selected compressed positions, ascending, -1 padded
    unsigned long long sel_cnt;   // int [rows]: number of valid entries (= min(512, visible))
    unsigned long long part;      // u64 [n_rb][R][G][512]: per-chunk local top-512 (sorted desc)
    unsigned long long pcnt;      // int [n_rb][R][G]
    unsigned long long bmax;      // u16 [rows][nb_cap]: per-8-block max sortable score (candidate source)
    unsigned long long cand;      // int [rows][2048]: selected blocks, ascending (cand kernel output)
    unsigned long long cand_cnt;  // int [rows]
    unsigned long long bm;        // u32 [rows][nbw_cap]: selected-block bitmap
    unsigned long long uni;       // u32 [n_rb][ucap]: union of the rb's selected blocks: block << 8 | row mask
    unsigned long long ucnt;      // int [n_rb]
    unsigned long long dbg;       // u16 [rows][dbg_cap] sortable scores (tests), 0 = off
    unsigned long long err;       // int*: contract violations (0 = fine)
    unsigned long long dbg_q;     // u8 [rows][32*64 codes + 32*4 scales] prepared queries (tests), 0 = off
    unsigned long long tick;      // int [n_rb], zeroed once: the score kernel's last chunk CTA of a row
                                  // block merges the chunks itself and re-zeroes it; 0 = merge kernel
    int n_rb, rows_total, iq_ld, w_ld;
    int in_f32, ratio, mode, write_bmax;   // mode 0: all visible keys; 1: the union candidate blocks
    int n_keys_cap, nb_cap, nbw_cap, G;    // G = chunks per row block (grid.x of the score kernel)
    int min_tiles, dbg_cap, ucap, R;       // R = row-block width the score kernel was launched with
    int stripe_lo, stripe_hi;              // key-stripe mode (TP, T0 §3.8): only keys [lo, hi), multiples of
                                           // 128, partial lists always; stripe_hi = 0: off
    int list_off, lists;                   // partial lists of row (b, r) at [(b*R + r) * lists + list_off + c];
                                           // lists = 0: G (one GPU). The merge reads `lists` lists when > 0.
    float inv_freq[32];                    // RoPE inverse frequencies (YaRN, torch fp32 semantics)
};

#define D41_IDX_ERR_POS 1      // a row's visible count exceeds the cache capacity
#define D41_IDX_ERR_UCAP 2     // the union list overflowed its capacity

__device__ __forceinline__ void d41_idx_err(const d41_idx_args& a, int code) {
    if (a.err) atomicOr((int*)a.err, code);
}

// ---- block-scaled FP4 MMA (verified layout: probe in PLAN/DSV41_KI_REPORT.md §3) ----------------
// A row-major 16x64 E2M1 (a0: row g k 8t.., a1: row g+8, a2: row g k 32+8t.., a3: row g+8);
// B col-major 64x8 (b0: col g k 8t.., b1: col g k 32+8t..); sfa: lanes t<=1, bytes 0,1 = the two
// 32-group scales of row g+8t; sfb: lanes t==0, bytes 0,1 = col g's.
__device__ __forceinline__ void d41_mma_mxf4(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1, unsigned sfa, unsigned sfb,
                                             float& d0, float& d1, float& d2, float& d3) {
    const unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// Two f32 -> one byte of two E2M1 nibbles (low nibble = lo), round-to-nearest-even, satfinite
// (REFb: bitwise equal to DeepSeek's TileLang cast on GB10).
__device__ __forceinline__ unsigned d41_cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile("{\n.reg .b8 b0, b1, b2, b3;\ncvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\nmov.b32 %0, {b0, b1, b2, b3};\n}"
                 : "=r"(tmp) : "f"(lo), "f"(hi));
    return tmp & 0xFFu;
}

__device__ __forceinline__ float d41_bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// bf16 bits -> 16-bit key that orders like the value (-0 folded onto +0: torch compares values).
__device__ __forceinline__ unsigned d41_idx_sortable(unsigned short b) {
    unsigned u = b;
    if (u == 0x8000u) u = 0;
    return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}

// The total order key: higher score first, then lower position. Unique per position.
__device__ __forceinline__ unsigned long long d41_idx_key(unsigned s16, unsigned p) {
    return ((unsigned long long)s16 << 32) | (unsigned long long)(0xFFFFFFFFu - p);
}
__device__ __forceinline__ unsigned d41_idx_key_pos(unsigned long long k) { return 0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull); }

// Visible compressed positions of a row at absolute position p (reference: (t + 1) // ratio).
__device__ __forceinline__ int d41_idx_nvis(int p, int ratio) { return p < 0 ? 0 : (p + 1) / ratio; }

// Effective chunk count of a row block: as many chunks of >= min_tiles tiles as the grid allows.
// The score kernel and the merge kernel both call this, from the same inputs.
__device__ __forceinline__ int d41_idx_geff(int n_tiles, int G, int min_tiles) {
    if (n_tiles <= 0) return 1;
    int want = (n_tiles + min_tiles - 1) / min_tiles;
    return want < G ? want : G;
}

// A thread group running one of the selection helpers below: the whole CTA (bar 0 = __syncthreads), or a warp-aligned
// slice of it with its own named barrier (PK1: the finish selects the rows of a row block in parallel, one group per row).
struct d41_idx_grp { int tid; int nt; int bar; };
__device__ __forceinline__ d41_idx_grp d41_idx_cta() { return d41_idx_grp{(int)threadIdx.x, (int)blockDim.x, 0}; }
__device__ __forceinline__ void d41_idx_gsync(const d41_idx_grp& g) {
    if (g.bar == 0) __syncthreads();
    else asm volatile("bar.sync %0, %1;" :: "r"(g.bar), "r"(g.nt) : "memory");
}

// Bitonic sort of N (power of two) u64 keys in shared memory, DESCENDING, by a group (default the whole CTA).
template <int N>
__device__ __forceinline__ void d41_idx_sort_desc(unsigned long long* v, const d41_idx_grp g = d41_idx_cta()) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = g.tid; i < N / 2; i += g.nt) {
                int lo = ((i & ~(j - 1)) << 1) | (i & (j - 1)), hi = lo | j;
                unsigned long long a = v[lo], b = v[hi];
                bool desc = (lo & k) == 0;
                if ((a < b) == desc) { v[lo] = b; v[hi] = a; }
            }
            d41_idx_gsync(g);
        }
    }
}
// Same, u32, ASCENDING (final position order).
template <int N>
__device__ __forceinline__ void d41_idx_sort_asc_u32(unsigned* v, const d41_idx_grp g = d41_idx_cta()) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = g.tid; i < N / 2; i += g.nt) {
                int lo = ((i & ~(j - 1)) << 1) | (i & (j - 1)), hi = lo | j;
                unsigned a = v[lo], b = v[hi];
                bool asc = (lo & k) == 0;
                if ((a > b) == asc) { v[lo] = b; v[hi] = a; }
            }
            d41_idx_gsync(g);
        }
    }
}

// k-th largest (k >= 1, k <= n) of n UNIQUE u64 keys in shared memory, by an MSB-first radix select on
// the 48 used bits (score 16 | inverted position 32), 8 bits a level; stops as soon as the bucket
// holding the k-th key is taken whole (its minimum is the answer). `sc` = 256 ints + 8 u64 of shared
// scratch (one per group). A group (default the whole CTA; warp-aligned); returns the key to every thread of it.
// PACKED = false: `sc` = 256 ints of bins + 8 u64 (1,088 B); true: 256 16-bit bins packed in 128 ints + 8 u64 (576 B; n < 65,536),
// the same counts, so the same answer (PK1: small per-group scratch for the in-loop compactions).
template <bool PACKED = false>
__device__ __forceinline__ unsigned long long d41_idx_kth(const unsigned long long* v, int n, int k, int* sc, const d41_idx_grp g = d41_idx_cta()) {
    int* hist = sc;
    unsigned long long* st = (unsigned long long*)(sc + (PACKED ? 128 : 256));   // [0] prefix, [1] mask, [2] need, [3] result, [4] done
    const int tid = g.tid, nt = g.nt;
    if (tid == 0) { st[0] = 0ull; st[1] = 0ull; st[2] = (unsigned long long)k; st[3] = ~0ull; st[4] = 0ull; }
    for (int shift = 40; shift >= 0; shift -= 8) {
        for (int i = tid; i < (PACKED ? 128 : 256); i += nt) hist[i] = 0;
        d41_idx_gsync(g);
        const unsigned long long pre = st[0], msk = st[1];
        for (int i = tid; i < n; i += nt) {
            const unsigned long long x = v[i];
            if ((x & msk) == pre) {
                const int d = (int)((x >> shift) & 255);
                if (PACKED) atomicAdd((unsigned*)&hist[d >> 1], 1u << ((d & 1) * 16)); else atomicAdd(&hist[d], 1);
            }
        }
        d41_idx_gsync(g);
        if (tid < 32) {   // the group's first warp: suffix scan of the 256 bins (8 per lane), find the bucket of the need-th key
            int loc[8], tot = 0;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int bin = 255 - (tid * 8 + j);
                loc[j] = PACKED ? (int)((((const unsigned*)hist)[bin >> 1] >> ((bin & 1) * 16)) & 0xFFFFu) : hist[bin];
                tot += loc[j];
            }
            int inc = tot;   // inclusive prefix over lanes (descending digits)
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xFFFFFFFFu, inc, o); if (tid >= o) inc += y; }
            const int need = (int)st[2], before = inc - tot;
            if (before < need && inc >= need) {   // exactly one lane holds the bucket
                int cum = before, d = 0, cnt = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) { if (cnt == 0 && cum + loc[j] >= need) { d = 255 - (tid * 8 + j); cnt = loc[j]; } else if (cnt == 0) cum += loc[j]; }
                const int rem = need - cum;
                st[0] = pre | ((unsigned long long)d << shift);
                st[1] = msk | (255ull << shift);
                st[2] = (unsigned long long)rem;
                st[4] = (rem == cnt || shift == 0) ? 1ull : 0ull;
            }
        }
        d41_idx_gsync(g);
        if (st[4]) break;
    }
    // the k-th key is the smallest key of the final bucket
    const unsigned long long pre = st[0], msk = st[1];
    unsigned long long mn = ~0ull;
    for (int i = tid; i < n; i += nt) { const unsigned long long x = v[i]; if ((x & msk) == pre && x < mn) mn = x; }
    for (int o = 16; o > 0; o >>= 1) { const unsigned long long y = __shfl_xor_sync(0xFFFFFFFFu, mn, o); mn = y < mn ? y : mn; }
    d41_idx_gsync(g);
    if ((tid & 31) == 0) atomicMin(&st[3], mn);
    d41_idx_gsync(g);
    const unsigned long long r = st[3];
    d41_idx_gsync(g);
    return r;
}

// Keep the keys >= kmin of v[0..n) (unsorted, in place) and return how many were kept. A group (default the whole
// CTA; EPT >= n / group threads); `sc` as for d41_idx_kth (its first int is the counter). Reads everything before writing.
// (`sc[0]` is the counter: the first word of either kth scratch layout.)
template <int EPT>
__device__ __forceinline__ int d41_idx_keep_ge(unsigned long long* v, int n, unsigned long long kmin, int* sc, const d41_idx_grp g = d41_idx_cta()) {
    unsigned long long x[EPT];
    const int tid = g.tid, nt = g.nt;
#pragma unroll
    for (int j = 0; j < EPT; j++) { const int i = tid + j * nt; x[j] = i < n ? v[i] : 0ull; }
    if (tid == 0) sc[0] = 0;
    d41_idx_gsync(g);
#pragma unroll
    for (int j = 0; j < EPT; j++) { const int i = tid + j * nt; if (i < n && x[j] >= kmin) v[atomicAdd(&sc[0], 1)] = x[j]; }
    d41_idx_gsync(g);
    const int kept = sc[0];
    d41_idx_gsync(g);
    return kept;
}

// Final output of one row: the n (<= 512) selected keys in `keys` (any order) -> positions ascending
// into sel[row][0..n), -1 to 512, and sel_cnt[row] = n. `tmp` = `tmp_words` (>= 512 + 16) u32 of shared
// scratch. When the visible range fits a bitmap in `tmp`, positions are ordered by an ordered
// compaction of that bitmap (no sort); otherwise by a bitonic sort (the same positions in the same order).
// A group (default the whole CTA; warp-aligned, <= 16 warps: the per-warp sums take the last 16 words of tmp).
__device__ __forceinline__ void d41_idx_write_final(const d41_idx_args& a, int row, const unsigned long long* keys, int n,
                                                    unsigned* tmp, int tmp_words, int nvis, const d41_idx_grp g = d41_idx_cta()) {
    const int tid = g.tid, nt = g.nt;
    int* sel = (int*)a.sel + (long long)row * D41_IDX_TOPK;
    {
        const int nw = (nvis + 31) >> 5;
        unsigned* bmw = tmp;
        int* wsum = (int*)(tmp + tmp_words - 16);   // never overlaps a used word: nw <= tmp_words - 16
        if (nw <= tmp_words - 16) {
            for (int i = tid; i < nw; i += nt) bmw[i] = 0u;
            d41_idx_gsync(g);
            for (int i = tid; i < n; i += nt) { const unsigned p = d41_idx_key_pos(keys[i]); atomicOr(&bmw[p >> 5], 1u << (p & 31)); }
            d41_idx_gsync(g);
            // each thread a contiguous run of words; exclusive scan of the per-thread popcounts
            const int per = (nw + nt - 1) / nt, w0 = tid * per, w1 = min(nw, w0 + per);
            int c = 0;
            for (int i = w0; i < w1; i++) c += __popc(bmw[i]);
            int inc = c;
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xFFFFFFFFu, inc, o); if ((tid & 31) >= o) inc += y; }
            if ((tid & 31) == 31) wsum[tid >> 5] = inc;
            d41_idx_gsync(g);
            int base = 0;
            for (int ww = 0; ww < (tid >> 5); ww++) base += wsum[ww];
            int o = base + inc - c;
            for (int i = w0; i < w1; i++) { unsigned u = bmw[i]; while (u) { const int j = __ffs(u) - 1; u &= u - 1; sel[o++] = i * 32 + j; } }
            for (int i = n + tid; i < D41_IDX_TOPK; i += nt) sel[i] = -1;
            if (tid == 0) ((int*)a.sel_cnt)[row] = n;
            d41_idx_gsync(g);
            return;
        }
    }
    for (int i = tid; i < D41_IDX_TOPK; i += nt) tmp[i] = i < n ? d41_idx_key_pos(keys[i]) : 0xFFFFFFFFu;
    d41_idx_gsync(g);
    d41_idx_sort_asc_u32<D41_IDX_TOPK>(tmp, g);
    for (int i = tid; i < D41_IDX_TOPK; i += nt) sel[i] = i < n ? (int)tmp[i] : -1;
    if (tid == 0) ((int*)a.sel_cnt)[row] = n;
    d41_idx_gsync(g);
}
