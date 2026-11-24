// d41_kc_index_k / d41_kc_swa_write (leg KC).
//
// d41_kc_index_k — model.py Indexer.forward 540-548: k = k_norm(wk(latent)) (the bf16 GEMM output is
//   the input here), RoPE on the last 64 of 128 channels at position g * ratio, bf16 round,
//   fp4_act_quant(32, e8m0 scales), stored as two planes: q [row][64 B], s [row][4 B].
// d41_kc_swa_write — model.py _window_kv 701-721: kv = kv_norm(wkv(x)) (bf16 GEMM output in),
//   RoPE on the last 64 of 512 channels at position p0 + i, bf16 round, act_quant(32, ue8m0)
//   (amax floor 1e-4, scale = pow2ceil(amax * fp32(1/448))), stored to the ring slot (pos % 134) for
//   the LAST 134 rows of the call, and, if asked, to a per-call chunk buffer [rows] (prefill attention
//   reads the chunk's own quantised K/V from it). Two planes each: q [row][512 B], s [row][16 B].
#pragma once

struct d41_kc_index_args {
    const __nv_bfloat16* k_raw;  // [n_groups][128] bf16 (wk output)
    const void* norm_w;          // [128]
    const float* inv_freq;       // [32]
    unsigned char* q_plane;      // [rows][64 B] indexed by the absolute group
    unsigned char* s_plane;      // [rows][4 B]
    int ratio, g_first, n_groups;
    unsigned norm_fmt;
    float eps;
};
// block (32, 4): one warp per group, 4 groups per block; grid.x = ceil(n_groups / 4).
extern "C" __global__ void __launch_bounds__(128) d41_kc_index_k(d41_kc_index_args a) {
    D41_PDL_ENTRY();
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
    if (lane >= 16) {  // channels 64..127
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
    am = fmaxf(am, 6.0f * __uint_as_float(0x00800000u));       // 6 * 2^-126
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

struct d41_kc_swa_args {
    const __nv_bfloat16* kv_raw;  // [rows][512] bf16 (wkv output)
    const void* norm_w;           // [512]
    const float* inv_freq;        // [32]
    unsigned char* ring_q;        // [134][512 B] (nullable)
    unsigned char* ring_s;        // [134][16 B]
    unsigned char* chunk_q;       // [rows][512 B] (nullable)
    unsigned char* chunk_s;       // [rows][16 B]
    long long p0;
    int rows;
    unsigned norm_fmt;
    float eps;
};
// grid.x = rows, 128 threads.
extern "C" __global__ void __launch_bounds__(D41_KC_THREADS) d41_kc_swa_write(d41_kc_swa_args a) {
    D41_PDL_ENTRY();
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5, i = blockIdx.x;
    if (i >= a.rows) return;
    const long long pos = a.p0 + i;
    const bool to_ring = a.ring_q && i >= a.rows - D41_SWA_SLOTS;
    if (!to_ring && !a.chunk_q) return;
    __shared__ float s_red[4];
    float y[4];
    const __nv_bfloat16* p = a.kv_raw + (size_t)i * D41_KC_DIM + 4 * t;
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) { y[c] = d41_bf(p[c]); ss = __fadd_rn(ss, __fmul_rn(y[c], y[c])); }
    ss = d41_warp_sum(ss);
    if (lane == 0) s_red[wid] = ss;
    __syncthreads();
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
    // act_quant(32, ue8m0): 8 threads per group
    float am = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
    am = d41_group_max<8>(am);
    am = fmaxf(am, 1e-4f);
    const unsigned sb = d41_pow2ceil_byte(__fmul_rn(am, 1.0f / 448.0f));
    const float sc = d41_pow2_of_byte(sb);
    unsigned w = 0;
#pragma unroll
    for (int c = 0; c < 4; ++c) w |= d41_e4m3_byte(fminf(fmaxf(__fdiv_rn(y[c], sc), -448.0f), 448.0f)) << (8 * c);
    if (to_ring) {
        const int slot = (int)(pos % D41_SWA_SLOTS);
        *(unsigned*)(a.ring_q + (size_t)slot * sizeof(d41_swa_q) + 4 * t) = w;
        if ((t & 7) == 0) a.ring_s[(size_t)slot * sizeof(d41_swa_s) + (t >> 3)] = (unsigned char)sb;
    }
    if (a.chunk_q) {
        *(unsigned*)(a.chunk_q + (size_t)i * sizeof(d41_swa_q) + 4 * t) = w;
        if ((t & 7) == 0) a.chunk_s[(size_t)i * sizeof(d41_swa_s) + (t >> 3)] = (unsigned char)sb;
    }
}

// ABI twin of src/dsv41/gpu/kv.rs: out[0..4] = sizeof of the four argument structs, out[4] = offsetof(p0), out[5] = offsetof(err) of
// the compress args. The probe compares them with the Rust structs.
extern "C" __global__ void d41_kc_sizes(unsigned* out) {
    D41_PDL_ENTRY();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = (unsigned)sizeof(d41_kc_compress_args);
        out[1] = (unsigned)sizeof(d41_kc_stash_args);
        out[2] = (unsigned)sizeof(d41_kc_index_args);
        out[3] = (unsigned)sizeof(d41_kc_swa_args);
        out[4] = (unsigned)offsetof(d41_kc_compress_args, p0);
        out[5] = (unsigned)offsetof(d41_kc_compress_args, err);
    }
}
