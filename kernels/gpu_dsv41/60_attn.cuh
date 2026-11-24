// =============================================================================
// kernels/gpu_dsv41/60_attn.cuh — band 60 (leg KA): sparse attention core of DeepSeek-V4.1-Flash.
// Head file of the band (included by gpu_dsv41.cu through __has_include); it includes the band's
// other files. Host side: src/dsv41/gpu/attn.rs (launch wrappers, CPU references, probe).
//
// WHAT IT COMPUTES (inference/model.py Attention.forward 766-791 + kernel.py sparse_attn 311-403):
//   for one query row (position p, lane l) and the rank's heads h:
//     keys   = the SWA window (positions win_end-127 .. win_end, FP8 E4M3 + E8M0/32, ring or linear rows)
//            + the selected compressed rows (FP4 E2M1 + E4M3/16 records of the layer's KV source)
//            + (DSpark stages) the block's extra keys (FP8 planes)
//     s      = (q_h . k) * 512^-0.5          bf16 operands, fp32 accumulate (MQA: one KV for all heads)
//     softmax over the keys + the per-head sink (denominator only), P cast to bf16 before P.V
//     o_h    = bf16( sum bf16(p) v / sum p )  then, per epilogue flags, the inverse RoPE on the last
//              64 dims and act_quant (E4M3 + pow2 E8M0 per 32, the WO-A input).
//
// THE CANONICAL PER-ROW ARITHMETIC (what every variant computes, bit for bit):
//   * the key list of a row: list index 0..127 = window position win_end-127+i (masked if < 0),
//     then the compressed entries sel[0..n_sel) (-1 / out-of-range = masked), then the extra keys;
//   * BLOCKS of 64 consecutive list entries; within a block S = ((S0 + S1) + S2) + S3, Si = the dim
//     quarter i (8 chained m16n8k16 MMAs, k ascending), times the scale; the block max and sum are
//     32-lane xor-butterfly reductions (two keys per lane, keys 2L, 2L+1), p = exp(s - m), bf16(p), P.V;
//   * the row's BLOCK LIST: the window blocks (list entries 0..63 only when win_end >= 64, then 64..127),
//     the compressed blocks (sel entries 64j..64j+63, j < ceil(n_sel / 64)), then the extra keys (one
//     block); SPLITS = D41_ATTN_SPLIT_BLOCKS consecutive blocks of that list (1: every block is a split).
//     Inside a split: the reference's online softmax (running max from -1e30, sum = fma(sum, corr, bsum),
//     acc *= corr). Splits are merged IN ORDER:
//        Mn = max(M, m_j); c0 = exp(M - Mn); c1 = exp(m_j - Mn);
//        L = fma(L, c0, l_j * c1); O = fma(O, c0, a_j * c1); M = Mn       (from M=-1e30, L=0, O=0)
//     then L += exp(sink - M), out = O * rcp(L) with one FMA correction (d41a_div: the IEEE quotient in
//     normal ranges), bf16 RNE. Split size 2 measured faster than 1 for decode (the per-CTA prologue and
//     the merge reads cost more than a second block).
//   A fully masked block or split is an exact identity in both steps, so skipping one is bit-
//   identical. The decode / verify entry runs one split per CTA and the last-arriving CTA merges the
//   partials in split order; the prefill entry runs every split of a row in one CTA and merges in
//   registers: the same operations in the same order, so a row's output is identical in decode,
//   verify and prefill, whatever the row count, the chunking or the rank's head range (TP).
//   The reference's own block layout (window list order, one running max over all blocks) differs
//   only in where bf16(p) is rounded: bf16-P noise, measured against REFb's emulation (KA report:
//   rel-L2 ~5e-4..1.5e-3 in o, the same size as the reference's own distance to an exact softmax).
//
// FILES OF THE BAND:
//   60_attn.cuh        this file: constants, the argument structs
//   61_attn_tile.cuh   smem layout, swizzle, ldmatrix / mma, key resolution, FP8 / FP4 dequant
//   62_attn_core.cuh   one block (QK, online softmax, PV), the split merge, the epilogue
//   63_attn_entry.cuh  the entry kernels d41_attn_decode / d41_attn_prefill
// =============================================================================
#pragma once

#define D41_ATTN_D 512            // head_dim (latent = K = V)
#define D41_ATTN_ROPE 64          // rope tail (last 64 dims)
#define D41_ATTN_HG 16            // heads per CTA (one m16 tile)
#define D41_ATTN_BLK 64           // keys per online-softmax block
#define D41_ATTN_WIN 128          // window list entries (window_size)
#define D41_ATTN_SEL_MAX 512      // index_topk
#define D41_ATTN_EXT_MAX 64       // extra keys (DSpark block = 5)
#define D41_ATTN_SPLIT_BLOCKS 2   // 64-key blocks per split (the canonical arithmetic; see below)
#define D41_ATTN_MAX_BLOCKS 11    // window 2 + compressed 8 + extra 1
#define D41_ATTN_MAX_SPLITS ((D41_ATTN_MAX_BLOCKS + D41_ATTN_SPLIT_BLOCKS - 1) / D41_ATTN_SPLIT_BLOCKS)
#define D41_ATTN_MAX_LANES 8
#define D41_ATTN_THREADS 512      // 16 warps (one per head of the CTA for the softmax)
#define D41_ATTN_EPI_ROPE 1u      // epilogue: inverse RoPE on dims 448..511
#define D41_ATTN_EPI_AQ 2u        // epilogue: act_quant (E4M3 + E8M0 per 32) -> bf16 dequantised values
#define D41_ATTN_PART_FLOATS (D41_ATTN_HG * D41_ATTN_D + 2 * D41_ATTN_HG)   // one split partial: O (per thread), (m, l) per head

// error codes written to args.err (first one wins; 0 = fine)
#define D41_ATTN_ERR_SEL_RANGE 1  // a selected compressed position >= kv_rows (treated as masked)
#define D41_ATTN_ERR_ARGS 2       // bad arguments (heads / groups / alignment)

// One lane's caches for THIS layer (device pointers; nullptr = the source is absent).
struct __align__(8) d41_attn_lane {
    const unsigned char* swa_q;   // SWA ring plane q [slots][512] E4M3 (slot = position % slots)
    const unsigned char* swa_s;   // SWA ring plane s [slots][16] E8M0 (one per 32 values)
    const unsigned char* lin_q;   // linear window rows for positions >= lin_base (prefill chunk), [n][512]
    const unsigned char* lin_s;   // [n][16]
    const unsigned char* kv;      // main compressed KV records [kv_rows][288] (d41_kv_main_row) of the layer's KV source
    const unsigned char* ext_q;   // extra keys [n_ext][512] (DSpark block KV), E4M3
    const unsigned char* ext_s;   // [n_ext][16] E8M0
    long long lin_base;           // first position held by lin_q (>= every position when there is none)
    int slots;                    // ring slots (D41_SWA_SLOTS)
    int kv_rows;                  // rows in kv (selected positions must be below)
    int n_ext;                    // extra keys, 0..64
    int pad_;
};

// Kernel arguments (passed by value, __grid_constant__). Rows are query rows; row r belongs to
// lane lane[r] and uses q row r, sel row r and out row r.
struct __align__(8) d41_attn_args {
    d41_attn_lane lanes[D41_ATTN_MAX_LANES];
    const __nv_bfloat16* q;       // [rows][q_stride] bf16, heads contiguous: head h at h * 512 (post-RoPE)
    long long q_stride;           // elements; multiple of 8
    const float* sink;            // [heads] fp32 attn_sink of the rank's heads
    const int* pos;               // [rows] absolute position (inverse RoPE)
    const int* win_end;           // [rows] last window position (nullptr: = pos; DSpark rows: the main position)
    const int* lane;              // [rows] lane index (nullptr: 0)
    const int* sel;               // [rows][sel_stride] compressed positions, -1 = none (nullptr: no compressed part)
    const int* n_sel;             // [rows] entries of sel to read (nullptr: sel_cap)
    long long sel_stride;
    int sel_cap;                  // <= 512
    int heads;                    // heads in q / sink / out (1..64)
    __nv_bfloat16* out;           // [rows][out_stride] bf16: o after the enabled epilogue steps
    long long out_stride;         // elements; multiple of 2
    unsigned char* out_q;         // optional [rows][outq_stride] E4M3 codes (EPI_AQ)
    long long outq_stride;
    unsigned char* out_s;         // optional [rows][outs_stride] E8M0 scale bytes, one per 32 (EPI_AQ)
    long long outs_stride;
    float* part;                  // decode: [rows][groups][nsplit_max][D41_ATTN_PART_FLOATS] split partials
    unsigned int* cnt;            // decode: [rows][groups] arrival counters, zero before the first launch (self-resetting)
    int* err;                     // optional error word
    int rows;
    int groups;                   // ceil(heads / 16)
    int nsplit_max;               // decode grid splits (<= D41_ATTN_MAX_SPLITS)
    unsigned int epi;             // D41_ATTN_EPI_* flags
    float scale;                  // softmax scale (fp32(512^-0.5))
    int win_lo;                   // window keys below this position are masked (0 = none; CE's truncated replay)
    float rope_freq[D41_ATTN_ROPE / 2];   // inverse frequencies of this layer's RoPE class
    // PD1 (optional, decode entry): the epilogue also writes wo_a's ROTATED input rows (the producer-side rotation of the
    // dense decode class: wo_a then runs without d41x_dd_prep): rot_xh[row][h * 512 + d] = fp16(H128(o * suh)) per
    // 128-block, o = the bf16 value stored in out, suh = rot_suh[h / rot_hpg] at column (h % rot_hpg) * 512 + d (wo_a's
    // group of head h). rot_xh == nullptr: off.
    const __half* rot_suh[8];
    __half* rot_xh;
    long long rot_ld;             // elements
    int rot_hpg;                  // heads per wo_a group
    // leg PA1 (decode entry only; 0 = KA / PV1 / PD1's behaviour): D41_ATTN_X_* flags
    int xflags;
};
// leg PA1 (PLAN/DSV41_RX6_REPORT.md 2.1, the parts that keep our bits):
#define D41_ATTN_X_WPF 1          // L2-prefetch the row's window keys BEFORE griddepcontrol.wait (no data into registers)
#define D41_ATTN_X_DEFER 2        // every CTA writes its split state (CTA 0 its in-order prefix) as a slot; d41_attn_merge
                                  // (one CTA per (row, head), a second launch) merges in split order and runs the epilogue
#define D41_ATTN_X_QROPE 4        // q's forward RoPE (last 64 dims, rope_freq, pos) applied while loading q (the separate
                                  // d41f_rope_rows launch's per-element sequence: bf16 -> f32 rotate -> bf16 RNE)
#define D41_ATTN_X_2CTA 16        // leg P7ATTN: 256-thread 8-head 2-CTA/SM decode entry (`d41_attn_decode_2cta`), contiguous slot layout

#include "61_attn_tile.cuh"
#include "62_attn_core.cuh"
#include "63_attn_entry.cuh"
#include "64_attn_merge.cuh"
#include "65_attn_recipe.cuh"
#include "66_attn_2cta.cuh"
