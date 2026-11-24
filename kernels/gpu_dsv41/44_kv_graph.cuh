// =============================================================================
// kernels/gpu_dsv41/44_kv_graph.cuh — band 40 (leg FW, included from 40_kv.cuh after KC's kernels it copies):
// graph-safe twins of KC's window / compressor kernels for the TP decode / verify calls (host side
// src/dsv41/tp/tpk.rs, launched under the TP glue's module name).
#pragma once

// Graph-safe decode / verify variants (leg FW): the same arithmetic as leg KC's d41_kc_swa_write, d41_kc_compress,
// d41_kc_stash and d41_kc_index_k (line for line), with the positions read from DEVICE memory instead of the by-value
// p0, so one captured decode / verify call replays at every context:
//   * d41fw_swa_rows      the window K/V of every row of the call into its lane's ring (slot = pos % 134; per-row
//                         ring pointers from a table: rows of different lanes in one launch). Decode / verify only:
//                         every row goes to the ring (a call's rows are <= 6 per lane, the ring holds 128 + 6).
//   * d41fw_compress_seg  one lane segment's compressor: p0 = pos[0] of the segment (device), the groups the
//                         segment completes [p0 / r, (p0 + rows) / r) computed on the device; grid = the most groups
//                         the segment can complete (extra blocks return).
//   * d41fw_stash_seg     the frontier ring stash with the device p0.
//   * d41fw_index_k_seg   the index-K planes of the groups the segment completes (k_raw rows = latent rows from 0).
// =============================================================================
struct d41fw_swa_rows_args {
    const __nv_bfloat16* kv_raw;   // [rows][512] bf16 (wkv output)
    const void* norm_w;            // [512]
    const float* inv_freq;         // [32]
    const int* pos;                // [rows] absolute positions
    const unsigned long long* ring_q;   // [rows] the row's lane ring (q plane [134][512 B])
    const unsigned long long* ring_s;   // [rows] (s plane [134][16 B])
    int rows;
    unsigned norm_fmt;
    float eps;
    int pad;
};
// grid.x = rows, 128 threads.
// NB (leg PA1): the row's 128 threads are threads 0..127 of a wider CTA (d41_q_kv_norm's window role): the one barrier is
// the named barrier 1 over those 128 threads instead of __syncthreads. Same arithmetic.
template <bool NB>
__device__ __forceinline__ void d41fw_swa_row(const d41fw_swa_rows_args& a, int i) {
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    if (i >= a.rows) return;
    const long long pos = (long long)__ldg(a.pos + i);
    __shared__ float s_red[4];
    float y[4];
    const __nv_bfloat16* p = a.kv_raw + (size_t)i * D41_KC_DIM + 4 * t;
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) { y[c] = d41_bf(p[c]); ss = __fadd_rn(ss, __fmul_rn(y[c], y[c])); }
    ss = d41_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    if constexpr (NB) asm volatile("bar.sync 1, 128;" ::: "memory"); else __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(s_red[0], s_red[1]), __fadd_rn(s_red[2], s_red[3]));
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)D41_KC_DIM), a.eps));
#pragma unroll
    for (int c = 0; c < 4; ++c) y[c] = d41_rbf(__fmul_rn(d41_load_w(a.norm_w, a.norm_fmt, 4 * t + c), __fmul_rn(y[c], rs)));
    if (t >= (D41_KC_DIM - D41_KC_ROPE) / 4) {
        const int i0 = (4 * t - (D41_KC_DIM - D41_KC_ROPE)) >> 1;
        float c0, s0, c1, s1;
        d41_kc_rope_cs(a.inv_freq, pos, i0, c0, s0);
        d41_kc_rope_cs(a.inv_freq, pos, i0 + 1, c1, s1);
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
    unsigned char* rq = (unsigned char*)a.ring_q[i];
    unsigned char* rsb = (unsigned char*)a.ring_s[i];
    const int slot = (int)(pos % D41_SWA_SLOTS);
    *(unsigned*)(rq + (size_t)slot * sizeof(d41_swa_q) + 4 * t) = w;
    if ((t & 7) == 0) rsb[(size_t)slot * sizeof(d41_swa_s) + (t >> 3)] = (unsigned char)sb;
}
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41fw_swa_rows(d41fw_swa_rows_args a) {
    D41_PDL_ENTRY();
    d41fw_swa_row<false>(a, blockIdx.x);
}

// One lane segment's compressor with the device p0 (d41_kc_compress's body; `a.p0`, `a.g_first`, `a.n_groups` are
// derived here from pos0[0] and a.rows, a.ratio).
__device__ __forceinline__ void d41fw_compress_body(d41_kc_compress_args a, const int* pos0) {
    const long long p0 = (long long)__ldg(pos0);
    const int r0 = a.ratio;
    a.p0 = p0;
    a.g_first = (int)(p0 / r0);
    a.n_groups = (int)((p0 + a.rows) / r0) - a.g_first;
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int j = blockIdx.x;
    if (j >= a.n_groups) return;
    const int g = a.g_first + j;
    const int r = a.ratio;
    __shared__ float s_red[4];
    float v[4];
    if (r >= 2) {
        if (a.ring_rows <= 0 || r > 16) { if (t == 0 && a.err) *a.err = 2; return; }
        const long long q0 = (long long)g * r;
        float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, den[4] = {0.f, 0.f, 0.f, 0.f};
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float* sp;
            if (q >= a.p0) sp = a.score + (size_t)(q - a.p0) * D41_KC_DIM;
            else { if (!a.ring_score) { if (t == 0 && a.err) *a.err = 1; return; } sp = a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM; }
            const float4 x = *(const float4*)(sp + 4 * t);
            mx[0] = fmaxf(mx[0], x.x); mx[1] = fmaxf(mx[1], x.y); mx[2] = fmaxf(mx[2], x.z); mx[3] = fmaxf(mx[3], x.w);
        }
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float* sp = q >= a.p0 ? a.score + (size_t)(q - a.p0) * D41_KC_DIM : a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM;
            const float4 x = *(const float4*)(sp + 4 * t);
            den[0] = __fadd_rn(den[0], expf(__fsub_rn(x.x, mx[0]))); den[1] = __fadd_rn(den[1], expf(__fsub_rn(x.y, mx[1])));
            den[2] = __fadd_rn(den[2], expf(__fsub_rn(x.z, mx[2]))); den[3] = __fadd_rn(den[3], expf(__fsub_rn(x.w, mx[3])));
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int m = 0; m < r; ++m) {
            const long long q = q0 + m;
            const float *sp, *kp;
            if (q >= a.p0) { sp = a.score + (size_t)(q - a.p0) * D41_KC_DIM; kp = a.kv + (size_t)(q - a.p0) * D41_KC_DIM; }
            else { sp = a.ring_score + (size_t)(q % a.ring_rows) * D41_KC_DIM; kp = a.ring_kv + (size_t)(q % a.ring_rows) * D41_KC_DIM; }
            const float4 x = *(const float4*)(sp + 4 * t);
            const float4 k = *(const float4*)(kp + 4 * t);
            d41_kc_pool_acc(acc[0], k.x, x.x, mx[0], den[0]); d41_kc_pool_acc(acc[1], k.y, x.y, mx[1], den[1]);
            d41_kc_pool_acc(acc[2], k.z, x.z, mx[2], den[2]); d41_kc_pool_acc(acc[3], k.w, x.w, mx[3], den[3]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = d41_rbf(acc[c]);
    } else {
        if (g < a.p0 || g >= a.p0 + a.rows) { if (t == 0 && a.err) *a.err = 2; return; }
        const __nv_bfloat16* p = a.kv16 + (size_t)(g - a.p0) * D41_KC_DIM + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = d41_bf(p[c]);
    }
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) ss = __fadd_rn(ss, __fmul_rn(v[c], v[c]));
    ss = d41_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
    const float tot = __fadd_rn(__fadd_rn(s_red[0], s_red[1]), __fadd_rn(s_red[2], s_red[3]));
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)D41_KC_DIM), a.eps));
    float y[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) y[c] = d41_rbf(__fmul_rn(d41_load_w(a.norm_w, a.norm_fmt, 4 * t + c), __fmul_rn(v[c], rs)));
    if (a.latent_out) {
        __nv_bfloat16* o = a.latent_out + (size_t)j * D41_KC_DIM + 4 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) o[c] = __float2bfloat16_rn(y[c]);
    }
    if (!a.kv_main) return;
    if (t >= (D41_KC_DIM - D41_KC_ROPE) / 4) {
        const int i0 = (4 * t - (D41_KC_DIM - D41_KC_ROPE)) >> 1;
        float c0, s0, c1, s1;
        d41_kc_rope_cs(a.inv_freq, (long long)g * r, i0, c0, s0);
        d41_kc_rope_cs(a.inv_freq, (long long)g * r, i0 + 1, c1, s1);
        d41_kc_rope_pair(y[0], y[1], c0, s0); d41_kc_rope_pair(y[2], y[3], c1, s1);
#pragma unroll
        for (int c = 0; c < 4; ++c) y[c] = d41_rbf(y[c]);
    }
    float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    am = d41_group_max<4>(am);
    am = fmaxf(am, 6.0f * 0.001953125f);
    const unsigned sb = d41_e4m3_byte(__fdiv_rn(am, 6.0f));
    const float sc = d41_e4m3(sb);
    unsigned code[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) code[c] = d41_e2m1_code(d41_clamp6(__fdiv_rn(y[c], sc)));
    unsigned char* rec = a.kv_main + (size_t)g * sizeof(d41_kv_main_row);
    rec[2 * t] = (unsigned char)(code[0] | (code[1] << 4));
    rec[2 * t + 1] = (unsigned char)(code[2] | (code[3] << 4));
    if ((t & 3) == 0) rec[256 + (t >> 2)] = (unsigned char)sb;
}
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41fw_compress_seg(d41_kc_compress_args a, const int* pos0) {
    D41_PDL_ENTRY();
    d41fw_compress_body(a, pos0);
}

// d41_kc_stash with the device p0 (grid.x = min(rows, R)).
__device__ __forceinline__ void d41fw_stash_body(d41_kc_stash_args a, const int* pos0) {
    a.p0 = (long long)__ldg(pos0);
    const int n = a.rows < a.ring_rows ? a.rows : a.ring_rows;
    if ((int)blockIdx.x >= n) return;
    const int row = a.rows - n + blockIdx.x;
    const size_t dst = (size_t)((a.p0 + row) % a.ring_rows) * D41_KC_DIM, src = (size_t)row * D41_KC_DIM;
    *(float4*)(a.ring_kv + dst + 4 * threadIdx.x) = *(const float4*)(a.kv + src + 4 * threadIdx.x);
    *(float4*)(a.ring_score + dst + 4 * threadIdx.x) = *(const float4*)(a.score + src + 4 * threadIdx.x);
}
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41fw_stash_seg(d41_kc_stash_args a, const int* pos0) {
    D41_PDL_ENTRY();
    d41fw_stash_body(a, pos0);
}

// d41_kc_index_k for the groups a segment of `rows` rows at the device p0 completes (block (32, 4), grid.x =
// ceil(max groups / 4)); `a.g_first` / `a.n_groups` are derived here.
__device__ __forceinline__ void d41fw_index_k_body(d41_kc_index_args a, const int* pos0, int rows) {
    const long long p0 = (long long)__ldg(pos0);
    a.g_first = (int)(p0 / a.ratio);
    a.n_groups = (int)((p0 + rows) / a.ratio) - a.g_first;
    const int lane = threadIdx.x, j = blockIdx.x * blockDim.y + threadIdx.y;
    if (j >= a.n_groups) return;
    const int g = a.g_first + j;
    float y[4];
    {
        const __nv_bfloat16* p = a.k_raw + (size_t)j * 128 + 4 * lane;
        float ss = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) { y[c] = d41_bf(p[c]); ss = __fadd_rn(ss, __fmul_rn(y[c], y[c])); }
        ss = d41_warp_sum(ss);
        const float rs = rsqrtf(__fadd_rn(__fdiv_rn(ss, 128.0f), a.eps));
#pragma unroll
        for (int c = 0; c < 4; ++c) y[c] = d41_rbf(__fmul_rn(d41_load_w(a.norm_w, a.norm_fmt, 4 * lane + c), __fmul_rn(y[c], rs)));
    }
    if (lane >= 16) {
        const int i0 = (4 * lane - 64) >> 1;
        float c0, s0, c1, s1;
        d41_kc_rope_cs(a.inv_freq, (long long)g * a.ratio, i0, c0, s0);
        d41_kc_rope_cs(a.inv_freq, (long long)g * a.ratio, i0 + 1, c1, s1);
        d41_kc_rope_pair(y[0], y[1], c0, s0); d41_kc_rope_pair(y[2], y[3], c1, s1);
#pragma unroll
        for (int c = 0; c < 4; ++c) y[c] = d41_rbf(y[c]);
    }
    float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    am = d41_group_max<8>(am);
    am = fmaxf(am, 6.0f * __uint_as_float(0x00800000u));
    const unsigned sb = d41_pow2ceil_byte(__fmul_rn(am, 1.0f / 6.0f));
    const float sc = d41_pow2_of_byte(sb);
    unsigned code[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) code[c] = d41_e2m1_code(d41_clamp6(__fdiv_rn(y[c], sc)));
    unsigned char* q = a.q_plane + (size_t)g * sizeof(d41_index_k_q);
    q[2 * lane] = (unsigned char)(code[0] | (code[1] << 4));
    q[2 * lane + 1] = (unsigned char)(code[2] | (code[3] << 4));
    if ((lane & 7) == 0) a.s_plane[(size_t)g * sizeof(d41_index_k_s) + (lane >> 3)] = (unsigned char)sb;
}
extern "C" __global__ void __launch_bounds__(128) d41fw_index_k_seg(d41_kc_index_args a, const int* pos0, int rows) {
    D41_PDL_ENTRY();
    d41fw_index_k_body(a, pos0, rows);
}

// ---- leg PG1: the same three per-lane steps with the lane's segment read from DEVICE memory, so a captured decode /
// verify call does not depend on how its rows are split between lanes (the graph is keyed on the total rows).
// seg = i32 [lanes][2]: (row0, rows) of lane `slot` in the call (rows 0: the lane has no row in this call; the step
// returns). The row-indexed operands (kv / score / kv16 rows of the call) are offset by row0 here; the bodies are the
// seg variants' bodies above, line for line (the same arithmetic per row and group).
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41fw_compress_tab(d41_kc_compress_args a, const int* pos, const int* seg, int slot) {
    D41_PDL_ENTRY();
    const int row0 = __ldg(seg + 2 * slot), n = __ldg(seg + 2 * slot + 1);
    if (n <= 0) return;
    a.rows = n;
    if (a.kv) a.kv += (size_t)row0 * D41_KC_DIM;
    if (a.score) a.score += (size_t)row0 * D41_KC_DIM;
    if (a.kv16) a.kv16 += (size_t)row0 * D41_KC_DIM;
    d41fw_compress_body(a, pos + row0);
}
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41fw_stash_tab(d41_kc_stash_args a, const int* pos, const int* seg, int slot) {
    D41_PDL_ENTRY();
    const int row0 = __ldg(seg + 2 * slot), n = __ldg(seg + 2 * slot + 1);
    if (n <= 0) return;
    a.rows = n;
    a.kv += (size_t)row0 * D41_KC_DIM;
    a.score += (size_t)row0 * D41_KC_DIM;
    d41fw_stash_body(a, pos + row0);
}
extern "C" __global__ void __launch_bounds__(128) d41fw_index_k_tab(d41_kc_index_args a, const int* pos, const int* seg, int slot) {
    D41_PDL_ENTRY();
    const int row0 = __ldg(seg + 2 * slot), n = __ldg(seg + 2 * slot + 1);
    if (n <= 0) return;
    d41fw_index_k_body(a, pos + row0, n);
}
