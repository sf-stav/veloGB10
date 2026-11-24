// =============================================================================
// kernels/gpu_dsv41/90_dspark.cuh — band 90-99 (leg DS): the DSpark round glue on the GPU.
//
//   90_dspark.cuh   this head: the shared structs (Rust twins in src/dsv41/gpu/dspark.rs), the counter RNG,
//                   the vocab-record merge
//   91_ds_vocab.cuh the vocab passes over a rank's vocab shard: per-row statistics (max / argmax, log-sum-exp of
//                   the processed distribution, a keyed Gumbel-max sample, a probe token's value), the residual
//                   pass, the per-rank reduce, the cross-rank merge with its epilogues (draft pick, verify
//                   statistics, residual + finish) and the per-lane accept kernel (DeepSeek's rule)
//   92_ds_draft.cuh the drafter's glue: KV rows with device positions (ring / block keys), the block setup, the
//                   hc copy, the main-row setup, the round commit
//   93_ds_chain.cuh (leg PK5) a Markov chain step in three launches (fused gather + GEMV + Gumbel scores, the
//                   scan, the merge + pick), the confidence head, a one-pass plain GEMV for up to 32 rows
//
// Semantics (PLAN/DSV41_XREF_DEEPINFER.md §4-5, plan Revision 4):
// * processed distribution of a row = softmax(a / T) over the kept set (a prefix of the (a desc, id asc) order,
//   given by a per-row cut (cut_a, cut_id); no cut = the whole vocabulary). Greedy rows (inv_t == 0) use the exact
//   argmax of the raw logits, lowest id on ties.
// * draws are keyed by (seed, position * 3 + stream): stream 0 = the token draw (the bonus row; drafts in the draft
//   namespace seed + 1,000,000,007), 1 = the residual draw after a rejection, 2 = the acceptance uniform.
//   A token draw is a Gumbel maximum: argmax over the kept set of x_v + g_v, g_v = -log(-log1p(-u_v)) with u_v from
//   splitmix64 of (row key + v * gamma) converted at full float range (no truncated tail at 2^-24); it is exactly a
//   sample of the processed distribution and it splits over vocab shards (the maximum of the ranks' maxima).
// * acceptance of draft d at a verify row: u < exp(min(log q(d) - log p(d), 0)) with u = SV's 53-bit uniform of
//   (seed, pos * 3 + 2); first rejection: a draw from max(q - alpha p, 0), alpha = exp(-5e-5); zero residual mass:
//   the target's first maximum logit. All accepted: the bonus row's draw. Greedy: d == argmax.
// Deterministic: every reduction runs in a fixed order (threads strided ascending, a fixed butterfly, blocks and
// ranks ascending), no atomics on values; a row's result never depends on the other rows of the call.
// =============================================================================
#pragma once

#define D41_DS_THREADS 256
#define D41_DS_BLOCK 5                    // DSpark block rows per lane (the trained block size)
#define D41_DS_IDS_ROWS (D41_DS_BLOCK + 1)  // DSpark draft ids table rows: [0] = the block token, [1..5] = drafts
#define D41_DS_WIDE_RPL 16                // leg P7COPY (WIDEVERIFY1): [0] = pending + up to 15 copied tokens
#define D41_DS_OUT_TOK 16                 // emitted tokens slot per lane (<= 6 DSpark, <= 16 wide copy verify)
#define D41_DS_NS_DRAFT 1000000007ull     // the draft draws' seed offset (DeepSeek's namespace)

// one (row, part) of a vocab pass; a part = a block of a rank's shard, a rank, or the whole row. 64 B.
struct d41_ds_vrec {
    float amax; int aid;        // raw-logit maximum over ALL ids, lowest id on ties (greedy, zero-mass fallback)
    float m; float s;           // log-sum-exp of x = a * inv_t over the KEPT set: max m, sum s of exp(x - m)
    float gmax; int gid;        // Gumbel maximum x + g over the kept set and its id (sampled rows)
    float gx;                   // x of gid
    float px; int pheld;        // probe token: x if kept (-inf if cut), pheld = 1 when this part holds it
    float rsum;                 // residual pass: sum of r_v = max(q_v - alpha p_v, 0)
    float rmax; int rid;        // residual pass: maximum of log r_v + g_v and its id
    int pad0, pad1, pad2, pad3;
};

// per-lane round state, device resident (written by the host at lane setup, then by the round's kernels). 72 B.
struct d41_ds_lane {
    unsigned long long seed;    // the request's sampling seed
    float inv_t;                // 1 / temperature; 0 = greedy
    int vbase;                  // position of the verify block's first row (its input token)
    int k;                      // drafts verified this round (0 = a plain decode row)
    int wend;                   // the drafter's window end (last fed, accepted position)
    int tok;                    // the drafter's block token (the emitted bonus / correction)
    int n_acc;                  // drafts accepted this round
    int n_emit;                 // tokens emitted this round (n_acc + 1)
    int need_res;               // a rejection needs the residual draw
    unsigned rounds;            // rounds finished (counters, read by the host)
    unsigned drafted;           // drafts verified
    unsigned accepted;          // drafts accepted
    int active;                 // 0 = the lane takes no part in this call
    int zmass;                  // this round's rejection found an empty residual (the first max logit was taken)
    int copy;                   // leg CD1: this round's drafts are a copy proposal (accept: draft == the target's own draw, no residual)
    int suppress_eos;           // leg P7.2: rows 0..suppress_eos mask stop_ids to -inf (0 = none)
    int trow0_p1;               // leg P7.4: 1 + packed target row0 in tlog (0 = strided layout)
};
static_assert(sizeof(d41_ds_lane) == 72, "d41_ds_lane layout");

__device__ __forceinline__ unsigned long long d41_ds_mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// The draft namespace's seed of a lane's CURRENT round (leg PS1): the request seed + DeepSeek's namespace offset + the
// lane's round counter (x the golden ratio), so a draft's noise is never reused by a later round. A draft-length rule
// that reads the drafts (the conf rule: draft i's confidence depends on draft i - 1) would otherwise bias the next
// round's drafts at the positions it re-drafts (position-keyed noise conditioned on the choice); with fresh noise per
// round every draft is an exact draw from q whatever the rule saw. Round 0 keeps the position-only key. Host twin:
// `vocab::draft_seed`.
__device__ __forceinline__ unsigned long long d41_ds_draft_seed(const struct d41_ds_lane& L) {
    return L.seed + D41_DS_NS_DRAFT + (unsigned long long)L.rounds * 0x9E3779B97F4A7C15ull;
}
// the row key of (seed, counter): SV's uniform() word (src/dsv41/serve/sampler.rs)
__device__ __forceinline__ unsigned long long d41_ds_key(unsigned long long seed, unsigned long long ctr) {
    return d41_ds_mix(d41_ds_mix(seed) ^ (ctr * 0xD6E8FEB86659FD93ull));
}
// SV's 53-bit uniform in [0, 1)
__device__ __forceinline__ double d41_ds_uniform(unsigned long long seed, unsigned long long ctr) {
    return (double)(d41_ds_key(seed, ctr) >> 11) * (1.0 / 9007199254740992.0);
}
// Gumbel noise of token `id` under row key `key`: u = r * 2^-64 at full float range (r = 0 -> the smallest normal)
__device__ __forceinline__ float d41_ds_gumbel(unsigned long long key, int id) {
    const unsigned long long r = d41_ds_mix(key + (unsigned long long)(unsigned)id * 0x9E3779B97F4A7C15ull);
    float u = __fmul_rn((float)r, 5.421010862427522e-20f);   // 2^-64
    u = fmaxf(u, 1.1754943508222875e-38f);
    const float e = -log1pf(-u);          // Exp(1); u -> 1 rounds to +inf: g = -inf, never wins
    return -logf(e);
}

__device__ __forceinline__ void d41_ds_rec_empty(d41_ds_vrec& r) {
    r.amax = -INFINITY; r.aid = 0x7FFFFFFF; r.m = -INFINITY; r.s = 0.0f; r.gmax = -INFINITY; r.gid = 0x7FFFFFFF; r.gx = -INFINITY;
    r.px = -INFINITY; r.pheld = 0; r.rsum = 0.0f; r.rmax = -INFINITY; r.rid = 0x7FFFFFFF; r.pad0 = r.pad1 = r.pad2 = r.pad3 = 0;
}
// a <- a (+) b, every field with an explicit tie order (lowest id), sums in the order a then b
__device__ __forceinline__ void d41_ds_rec_merge(d41_ds_vrec& a, const d41_ds_vrec& b) {
    if (b.amax > a.amax || (b.amax == a.amax && b.aid < a.aid)) { a.amax = b.amax; a.aid = b.aid; }
    const float m = fmaxf(a.m, b.m);
    if (m != -INFINITY) {
        const float sa = a.m == -INFINITY ? 0.0f : __fmul_rn(a.s, expf(a.m - m));
        const float sb = b.m == -INFINITY ? 0.0f : __fmul_rn(b.s, expf(b.m - m));
        a.s = __fadd_rn(sa, sb);
        a.m = m;
    }
    if (b.gmax > a.gmax || (b.gmax == a.gmax && b.gid < a.gid)) { a.gmax = b.gmax; a.gid = b.gid; a.gx = b.gx; }
    if (b.pheld) { a.px = b.px; a.pheld = 1; }
    a.rsum = __fadd_rn(a.rsum, b.rsum);
    if (b.rmax > a.rmax || (b.rmax == a.rmax && b.rid < a.rid)) { a.rmax = b.rmax; a.rid = b.rid; }
}
__device__ __forceinline__ float d41_ds_lse(const d41_ds_vrec& r) { return r.m == -INFINITY ? -INFINITY : __fadd_rn(r.m, logf(r.s)); }

// warp-level merge (fixed xor butterfly) through shuffles of every field
__device__ __forceinline__ void d41_ds_rec_shfl_merge(d41_ds_vrec& r, int o) {
    d41_ds_vrec b;
    b.amax = __shfl_xor_sync(0xFFFFFFFFu, r.amax, o); b.aid = __shfl_xor_sync(0xFFFFFFFFu, r.aid, o);
    b.m = __shfl_xor_sync(0xFFFFFFFFu, r.m, o); b.s = __shfl_xor_sync(0xFFFFFFFFu, r.s, o);
    b.gmax = __shfl_xor_sync(0xFFFFFFFFu, r.gmax, o); b.gid = __shfl_xor_sync(0xFFFFFFFFu, r.gid, o); b.gx = __shfl_xor_sync(0xFFFFFFFFu, r.gx, o);
    b.px = __shfl_xor_sync(0xFFFFFFFFu, r.px, o); b.pheld = __shfl_xor_sync(0xFFFFFFFFu, r.pheld, o);
    b.rsum = __shfl_xor_sync(0xFFFFFFFFu, r.rsum, o); b.rmax = __shfl_xor_sync(0xFFFFFFFFu, r.rmax, o); b.rid = __shfl_xor_sync(0xFFFFFFFFu, r.rid, o);
    // the lower lane of each pair is "a" so both lanes compute the same merged value (sums commute in fp32
    // only bit-exactly when the operand order is the same: lane i and lane i^o both use (lower, upper))
    const bool lower = (threadIdx.x & o) == 0;
    d41_ds_vrec lo = lower ? r : b, hi = lower ? b : r;
    d41_ds_rec_merge(lo, hi);
    r = lo;
}

#include "91_ds_vocab.cuh"
#include "92_ds_draft.cuh"
#include "93_ds_chain.cuh"
#include "94_ds_mcache.cuh"
#if __has_include("95_vision.cuh")
#include "95_vision.cuh"
#endif

// ABI twin of src/dsv41/gpu/dspark.rs: sizes of the shared structs and the argument structs.
extern "C" __global__ void d41_ds_sizes(unsigned* out) {
    D41_PDL_ENTRY();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = (unsigned)sizeof(d41_ds_vrec);
        out[1] = (unsigned)sizeof(d41_ds_lane);
        out[2] = (unsigned)sizeof(d41_ds_vpart_args);
        out[3] = (unsigned)sizeof(d41_ds_vrank_args);
        out[4] = (unsigned)sizeof(d41_ds_vmerge_args);
        out[5] = (unsigned)sizeof(d41_ds_accept_args);
        out[6] = (unsigned)sizeof(d41_ds_kv_args);
        out[7] = (unsigned)sizeof(d41_ds_setup_args);
        out[8] = (unsigned)sizeof(d41_ds_rep_args);
        out[9] = (unsigned)sizeof(d41_ds_main_args);
        out[10] = (unsigned)sizeof(d41_ds_mstep_args);
        out[11] = (unsigned)sizeof(d41_ds_conf_args);
        out[12] = (unsigned)sizeof(d41_ds_plain_args);
        out[13] = (unsigned)sizeof(d41_ds_vscan_args);
        out[14] = (unsigned)sizeof(d41_ds_vfin_args);
        out[15] = (unsigned)sizeof(d41_ds_mbias_args);
    }
}
