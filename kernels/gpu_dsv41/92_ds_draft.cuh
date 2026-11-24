// d41_ds_kv_rows / d41_ds_setup / d41_ds_rep / d41_ds_main (leg DS): the drafter's glue with every per-round
// value read from DEVICE memory (positions, tokens, lanes), so a draft step is graph-capturable as a whole.
//
// d41_ds_kv_rows — model.py DSparkAttention 1041-1046 / 1061-1064: kv = kv_norm(wkv(x)) (the bf16 GEMM output is
//   the input here), RoPE on the last 64 of 512 channels at the row's position (theta 10,000, no YaRN: DSpark
//   stages are ratio 0), bf16 round, act_quant(32, ue8m0), stored as two planes (q [row][512 B], s [row][16 B]):
//   to the lane's ring at slot pos % 134 (main rows: the drafter's window keys) and / or to a per-call buffer
//   (block rows: the extra keys of KA's attention). The arithmetic is KC's d41_kc_swa_write (43_kv_index_swa.cuh)
//   line for line; only the position source (a device array instead of p0 + i) and the per-lane ring differ.
// d41_ds_setup — the block of each lane: embedding ids [token, noise x 4], positions wend + 1 .. wend + 5, the
//   window end, the lane index, the ids table's row 0 and the identity pre-mix (1, 0, 0, 0) of the first stage.
// d41_ds_rep — the block stream: the gathered embedding row copied to the 4 hc copies (forward_embed 1137).
// d41_ds_main — main-row positions of a round (vbase + j per (lane, j)), or (mode 1) the round's commit:
//   the next verify block starts at the drafter's block token, vbase = wend + 1.
#pragma once

struct d41_ds_kv_args {
    const __nv_bfloat16* kv_raw;  // [rows][512] bf16 (wkv output)
    const void* norm_w;           // [512]
    const float* inv_freq;        // [32]
    const int* pos;               // [rows] absolute positions
    const int* lane;              // [rows] lane of each row (nullable: lane 0)
    unsigned char* ring_q;        // lane 0's ring [134][512 B]; lane l's at + l * ring_lq (nullable: no ring write)
    unsigned char* ring_s;        // [134][16 B], lane l at + l * ring_ls
    long long ring_lq, ring_ls;
    unsigned char* chunk_q;       // [rows][512 B] (nullable)
    unsigned char* chunk_s;       // [rows][16 B]
    int rows;
    unsigned norm_fmt;
    float eps;
    int pad;
};
__device__ __forceinline__ void d41_ds_kv_row_one(
    const __nv_bfloat16* kv_raw, const void* norm_w, const float* inv_freq, const int* pos_arr, const int* lane_arr,
    unsigned char* ring_q, unsigned char* ring_s, long long ring_lq, long long ring_ls,
    unsigned char* chunk_q, unsigned char* chunk_s, int rows, unsigned norm_fmt, float eps, int i) {
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    if (i >= rows) return;
    const long long pos = (long long)__ldg(pos_arr + i);
    __shared__ float s_red[4];
    float y[4];
    const __nv_bfloat16* p = kv_raw + (size_t)i * D41_KC_DIM + 4 * t;
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) { y[c] = d41_bf(p[c]); ss = __fadd_rn(ss, __fmul_rn(y[c], y[c])); }
    ss = d41_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(s_red[0], s_red[1]), __fadd_rn(s_red[2], s_red[3]));
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)D41_KC_DIM), eps));
#pragma unroll
    for (int c = 0; c < 4; ++c) y[c] = d41_rbf(__fmul_rn(d41_load_w(norm_w, norm_fmt, 4 * t + c), __fmul_rn(y[c], rs)));
    if (t >= (D41_KC_DIM - D41_KC_ROPE) / 4) {
        const int i0 = (4 * t - (D41_KC_DIM - D41_KC_ROPE)) >> 1;
        float c0, s0, c1, s1;
        d41_kc_rope_cs(inv_freq, pos, i0, c0, s0);
        d41_kc_rope_cs(inv_freq, pos, i0 + 1, c1, s1);
        d41_kc_rope_pair(y[0], y[1], c0, s0); d41_kc_rope_pair(y[2], y[3], c1, s1);
#pragma unroll
        for (int c = 0; c < 4; ++c) y[c] = d41_rbf(y[c]);
    }
    float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    am = d41_group_max<8>(am);
    am = fmaxf(am, 1e-4f);
    const unsigned sb = d41_pow2ceil_byte(__fmul_rn(am, 1.0f / 448.0f));
    const float sc = d41_pow2_of_byte(sb);
    unsigned w = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) w |= d41_e4m3_byte(fminf(fmaxf(__fdiv_rn(y[c], sc), -448.0f), 448.0f)) << (8 * c);
    if (ring_q) {
        const int ln = lane_arr ? __ldg(lane_arr + i) : 0;
        const int slot = (int)((unsigned long long)pos % (unsigned long long)D41_SWA_SLOTS);
        unsigned char* rq = ring_q + (long long)ln * ring_lq;
        unsigned char* rsb = ring_s + (long long)ln * ring_ls;
        *(unsigned*)(rq + (size_t)slot * sizeof(d41_swa_q) + 4 * t) = w;
        if ((t & 7) == 0) rsb[(size_t)slot * sizeof(d41_swa_s) + (t >> 3)] = (unsigned char)sb;
    }
    if (chunk_q) {
        *(unsigned*)(chunk_q + (size_t)i * sizeof(d41_swa_q) + 4 * t) = w;
        if ((t & 7) == 0) chunk_s[(size_t)i * sizeof(d41_swa_s) + (t >> 3)] = (unsigned char)sb;
    }
}

// grid.x = rows, D41_KC_THREADS (128) threads, 4 channels per thread.
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41_ds_kv_rows(d41_ds_kv_args a) {
    D41_PDL_ENTRY();
    d41_ds_kv_row_one(a.kv_raw, a.norm_w, a.inv_freq, a.pos, a.lane, a.ring_q, a.ring_s, a.ring_lq, a.ring_ls, a.chunk_q, a.chunk_s, a.rows, a.norm_fmt, a.eps, blockIdx.x);
}

// leg P7.4: all 3 DSpark stages' window ring writes in one launch (grid = (rows, 3, 1), 128 threads).
struct d41_ds_kv3_args {
    const __nv_bfloat16* kv_raw;      // [3][rows][512] bf16 (stage s at + s * rows * 512)
    unsigned long long norm_w[3];     // [3] stage kv_norm weight ptrs
    const float* inv_freq;            // [32]
    const int* pos;                   // [rows]
    const int* lane;                  // [rows]
    unsigned char* ring_q;            // stage 0 lane 0; stage s lane l at + (s * cap + l) * ring_lq
    unsigned char* ring_s;            // stage 0 lane 0; stage s lane l at + (s * cap + l) * ring_ls
    long long ring_lq, ring_ls;
    int rows, cap;
    unsigned norm_fmt[3];
    float eps;
};
static_assert(sizeof(d41_ds_kv3_args) == 11 * 8 + 6 * 4, "d41_ds_kv3_args layout");

extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41_ds_kv_rows3(d41_ds_kv3_args a) {
    D41_PDL_ENTRY();
    const int s = blockIdx.y;
    if (s >= 3) return;
    const void* nw = (const void*)(s == 0 ? a.norm_w[0] : (s == 1 ? a.norm_w[1] : a.norm_w[2]));
    const unsigned nf = s == 0 ? a.norm_fmt[0] : (s == 1 ? a.norm_fmt[1] : a.norm_fmt[2]);
    d41_ds_kv_row_one(
        a.kv_raw + (size_t)s * (size_t)a.rows * D41_KC_DIM, nw, a.inv_freq, a.pos, a.lane,
        a.ring_q + (long long)s * (long long)a.cap * a.ring_lq,
        a.ring_s + (long long)s * (long long)a.cap * a.ring_ls,
        a.ring_lq, a.ring_ls, nullptr, nullptr, a.rows, nf, a.eps, blockIdx.x);
}

struct d41_ds_setup_args {
    const d41_ds_lane* lanes;
    int* eids;                    // [nlanes * 5] embedding ids: [tok, noise, noise, noise, noise] per lane
    int* bpos;                    // [nlanes * 5] positions wend + 1 + j
    int* bwend;                   // [nlanes * 5] the window end (wend)
    int* blane;                   // [nlanes * 5] the lane index
    int* ids;                     // ids table [6][nlanes]: row 0 = tok
    float* pre;                   // [nlanes * 5][4] the first stage's pre-mix: identity (1, 0, 0, 0)
    int nlanes, noise;
};
// grid (ceil(nlanes * 5 / 32)), 32 threads.
extern "C" __global__ void __launch_bounds__(32) d41_ds_setup(d41_ds_setup_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x * 32 + threadIdx.x;
    if (r >= a.nlanes * D41_DS_BLOCK) return;
    const int l = r / D41_DS_BLOCK, j = r % D41_DS_BLOCK;
    const d41_ds_lane& L = a.lanes[l];
    a.eids[r] = j == 0 ? L.tok : a.noise;
    a.bpos[r] = L.wend + 1 + j;
    a.bwend[r] = L.wend;
    a.blane[r] = l;
    if (j == 0) a.ids[l] = L.tok;
#pragma unroll
    for (int c = 0; c < 4; ++c) a.pre[r * 4 + c] = c == 0 ? 1.0f : 0.0f;
}

struct d41_ds_rep_args {
    const __nv_bfloat16* e;       // [rows][dim]
    __nv_bfloat16* x;             // [rows][4][dim]
    int rows, dim;
};
// grid (ceil(dim / 256), rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) d41_ds_rep(d41_ds_rep_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (r >= a.rows || c >= a.dim) return;
    const __nv_bfloat16 v = a.e[(size_t)r * a.dim + c];
#pragma unroll
    for (int h = 0; h < 4; ++h) a.x[((size_t)r * 4 + h) * a.dim + c] = v;
}

struct d41_ds_main_args {
    d41_ds_lane* lanes;
    const int* rlane;             // [rows] lane of each main row (mode 0)
    const int* rj;                // [rows] the row's index in its lane's verify block (mode 0)
    int* mpos;                    // [rows] out: vbase + j
    int* mlane;                   // [rows] out: the lane
    int rows, mode;               // mode 0: main-row setup over `rows`; mode 1: commit over `rows` = nlanes lanes
};
// grid (ceil(rows / 32)), 32 threads.
extern "C" __global__ void __launch_bounds__(32) d41_ds_main(d41_ds_main_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x * 32 + threadIdx.x;
    if (r >= a.rows) return;
    if (a.mode == 1) {
        d41_ds_lane& L = a.lanes[r];
        if (L.active) L.vbase = L.wend + 1;
        return;
    }
    const int l = a.rlane[r];
    a.mpos[r] = a.lanes[l].vbase + a.rj[r];
    a.mlane[r] = l;
}
