// =============================================================================
// kernels/gpu_dsv41/05_tp_fwd.cuh — band 05 (leg TP): the glue of the TP forward (src/dsv41/tp/fwd.rs)
// that no other band does. Included from 05_tp.cuh; loaded under the TP leg's own module name
// (src/dsv41/tp/tpk.rs), so the I0 loader's entry list is untouched.
//   * d41tpf_bf16_f32     bf16 rows -> f32 rows (the embedding rows rank 0 broadcasts through an fp32 sum
//                         all-reduce: every other rank contributes zeros, x + 0 is exact).
//   * d41tpf_expand_hc    f32 [rows][dim] -> bf16 [rows][hc][dim] (the reference's `h.unsqueeze(2).repeat(hc)`
//                         of the bf16 embedding; the f32 values are bf16 already, the cast is exact).
//   * d41tpf_fill_pre     the identity pre-mix [rows][stride] f32 (1, 0, 0, 0): make_identity_pre_mix.
//   * d41tpf_ring_lin     copy SWA-ring slots into linear window rows (positions [p_lo, p_hi) of the lane:
//                         slot = pos % slots), so a prefill-shaped call (rows > the backtrack) reads its window
//                         rows below p0 from linear rows the ring overwrite of the same call cannot touch.
// Elementwise, 256 threads, no runtime-indexed per-thread arrays.
// =============================================================================
#pragma once

struct d41tpf_cvt_args { const __nv_bfloat16* in; float* out; long long n; };
extern "C" __global__ void __launch_bounds__(256) d41tpf_bf16_f32(d41tpf_cvt_args a) {
    D41_PDL_ENTRY();
    for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < a.n; i += (long long)gridDim.x * 256)
        a.out[i] = __bfloat162float(a.in[i]);
}

struct d41tpf_expand_args { const float* in; __nv_bfloat16* out; long long in_ld, out_ld; int rows, dim, hc, pad; };
// grid (ceil(dim / 256), rows)
extern "C" __global__ void __launch_bounds__(256) d41tpf_expand_hc(d41tpf_expand_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (r >= a.rows || d >= a.dim) return;
    const __nv_bfloat16 v = __float2bfloat16_rn(a.in[(long long)r * a.in_ld + d]);
    for (int c = 0; c < a.hc; ++c) a.out[(long long)r * a.out_ld + (long long)c * a.dim + d] = v;
}

struct d41tpf_pre_args { float* out; long long stride; int rows, hc; };
extern "C" __global__ void __launch_bounds__(256) d41tpf_fill_pre(d41tpf_pre_args a) {
    D41_PDL_ENTRY();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= a.rows * a.hc) return;
    const int r = i / a.hc, c = i - r * a.hc;
    a.out[(long long)r * a.stride + c] = c == 0 ? 1.0f : 0.0f;
}

// One block per linear row: row i <- ring slot (p_lo + i) % slots, q bytes (q_bytes per row) and s bytes.
struct d41tpf_ring_args { const unsigned char* ring_q; const unsigned char* ring_s; unsigned char* lin_q; unsigned char* lin_s;
                          long long p_lo; int n, slots, q_bytes, s_bytes; };
extern "C" __global__ void __launch_bounds__(256) d41tpf_ring_lin(d41tpf_ring_args a) {
    D41_PDL_ENTRY();
    const int i = blockIdx.x;
    if (i >= a.n) return;
    const long long slot = (a.p_lo + i) % a.slots;
    for (int b = threadIdx.x; b < a.q_bytes; b += 256) a.lin_q[(long long)i * a.q_bytes + b] = a.ring_q[slot * a.q_bytes + b];
    for (int b = threadIdx.x; b < a.s_bytes; b += 256) a.lin_s[(long long)i * a.s_bytes + b] = a.ring_s[slot * a.s_bytes + b];
}
