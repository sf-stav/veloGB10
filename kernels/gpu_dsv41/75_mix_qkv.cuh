// 75_mix_qkv.cuh — band 70, leg PA1: the attention's two helper launches in one (the external reference's `_q_kv_norm`
// structure, PLAN/DSV41_RX6_REPORT.md 2.3; our arithmetic): blockIdx.y = the role, each role runs today's device function
// unchanged:
//   role 0: the q RMSNorm + wq_b's (and the index layer's idx wq_b's) rotated rows (d41x_dd_normrot_body, 256 threads);
//   role 1: the window K/V of the row into its lane's ring (d41fw_swa_row, threads 0..127, named barrier 1).
// Before the wait each role prefetches its constants into L2 (theirs: CCTL.E.PF2 of the norm weight and the suh lines
// before ACQBULK): role 0 the q-norm weight and the targets' suh, role 1 the kv-norm weight. (The q RoPE after wq_b is
// folded into the attention's q load instead: D41_ATTN_X_QROPE.)
#pragma once

struct d41_q_kv_norm_args { D41xNormRotArgs nr; d41fw_swa_rows_args sw; };

__device__ __forceinline__ void d41_l2pf(const void* p, long long bytes, int t, int nt) {
    if (p == nullptr) return;
    const char* c = reinterpret_cast<const char*>(p);
    for (long long o = (long long)t * 128; o < bytes; o += (long long)nt * 128) asm volatile("prefetch.global.L2 [%0];" :: "l"(c + o));
}
__device__ __forceinline__ int d41_fmt_bytes(unsigned f) { return f == D41_FMT_F32 ? 4 : 2; }

// grid (rows, 2), 256 threads
extern "C" __global__ void __launch_bounds__(256) d41_q_kv_norm(const __grid_constant__ d41_q_kv_norm_args a) {
    const int t = threadIdx.x;
    if (blockIdx.y == 0) {
        d41_l2pf(a.nr.w, (long long)a.nr.dim * d41_fmt_bytes(a.nr.w_fmt), t, 256);
        for (int i = 0; i < a.nr.nt && i < 2; ++i) d41_l2pf(a.nr.suh[i], (long long)a.nr.dim * 2, t, 256);
        D41_PDL_ENTRY();
        if ((int)blockIdx.x >= a.nr.rows) return;
        d41x_dd_normrot_body(a.nr);
    } else {
        d41_l2pf(a.sw.norm_w, (long long)D41_KC_DIM * d41_fmt_bytes(a.sw.norm_fmt), t, 256);
        D41_PDL_ENTRY();
        if (t >= D41_KC_THREADS) return;     // whole warps 4..7: the row's barrier is the named barrier over warps 0..3
        d41fw_swa_row<true>(a.sw, blockIdx.x);
    }
}
