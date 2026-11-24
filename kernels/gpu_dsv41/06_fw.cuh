// =============================================================================
// kernels/gpu_dsv41/06_fw.cuh — band 05/06 (leg FW): the serving engine's glue on top of the TP forward
// (src/dsv41/tp/{fwd,spec}.rs). Included from 05_tp.cuh; loaded under the TP glue's module name
// (src/dsv41/tp/tpk.rs), so the I0 loader's entry list is untouched.
//   * d41fw_tap      the DSpark tap of call rows [row0, row0 + n): the mean over the hc copies of the stream
//                    (bf16 [rows][hc][dim]) at the input of a target layer, as model.py `h.mean(dim=2)` and leg OD's
//                    oracle pin it: an exact f64 sum of the hc bf16 values, / hc, rounded to f32 then to bf16
//                    (OD: `bf16r((s / hc as f64) as f32)`). Written to dst rows dst0.. at column `col`.
//   * d41fw_xenc     the byte-exact all-gather over an fp32 sum all-reduce (leg DS's 64-B vocab records, any int
//                    bits): every 32-bit word becomes two exact floats (its 16-bit halves, 0..65535) in this rank's
//                    slot of [world][2 * nwords]; every other slot is 0 (x + 0 is exact for these values, and no
//                    denormal / NaN bit pattern ever enters the sum).
//   * d41fw_xdec     the reverse: [world][2 * nwords] floats -> [world][nwords] words (rank order).
// Elementwise, 256 threads, no runtime-indexed per-thread arrays.
// =============================================================================
#pragma once

struct d41fw_tap_args { const __nv_bfloat16* x; __nv_bfloat16* dst; long long dst_ld; int hc, dim, row0, n, dst0, col; };
// grid (ceil(dim / 256), n)
extern "C" __global__ void __launch_bounds__(256) d41fw_tap(d41fw_tap_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y, e = blockIdx.x * 256 + threadIdx.x;
    if (r >= a.n || e >= a.dim) return;
    const __nv_bfloat16* row = a.x + (long long)(a.row0 + r) * a.hc * a.dim;
    double s = 0.0;
    for (int j = 0; j < a.hc; ++j) s += (double)__bfloat162float(row[(long long)j * a.dim + e]);
    const float f = (float)(s / (double)a.hc);
    a.dst[(long long)(a.dst0 + r) * a.dst_ld + a.col + e] = __float2bfloat16_rn(f);
}

struct d41fw_xenc_args { const unsigned* in; float* out; int nwords, rank, world, pad; };
// grid ceil(world * 2 * nwords / 256)
extern "C" __global__ void __launch_bounds__(256) d41fw_xenc(d41fw_xenc_args a) {
    D41_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const long long per = 2ll * a.nwords;
    if (i >= per * a.world) return;
    const int r = (int)(i / per);
    const long long k = i - (long long)r * per;
    float v = 0.0f;
    if (r == a.rank) { const unsigned w = a.in[k >> 1]; v = (float)((k & 1) ? (w >> 16) : (w & 0xFFFFu)); }
    a.out[i] = v;
}

struct d41fw_xdec_args { const float* in; unsigned* out; int nwords, world; };
// grid ceil(world * nwords / 256)
extern "C" __global__ void __launch_bounds__(256) d41fw_xdec(d41fw_xdec_args a) {
    D41_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)a.nwords * a.world) return;
    const unsigned lo = (unsigned)a.in[2 * i], hi = (unsigned)a.in[2 * i + 1];
    a.out[i] = lo | (hi << 16);
}

// ---------------------------------------------------------------------------------------------------------------------
// leg SE: presence / frequency / repetition penalties on one row of target logits, on this rank's vocabulary slice.
//   * d41se_pen   logits [vlocal] (global ids v0 + i): c = counts[id] (the lane's generated tokens so far, u16 saturating)
//                 + the multiplicity of id among the row's chain tokens e0..e4 (the drafts before it); when c > 0:
//                 repetition (x > 0 ? x / r : x * r), then x -= freq * c + pres: the host sampler's `apply_penalties` exactly.
// grid ceil(vlocal / 256)
// ---------------------------------------------------------------------------------------------------------------------
struct d41se_pen_args { float* logits; const unsigned short* counts; long long v0; int vlocal, n_extra; int e0, e1, e2, e3; int e4, pad0; float pres, freq, rep, pad1; };
extern "C" __global__ void __launch_bounds__(256) d41se_pen(d41se_pen_args a) {
    D41_PDL_ENTRY();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= a.vlocal) return;
    const long long id = a.v0 + i;
    int c = (int)a.counts[id];
    if (a.n_extra > 0 && a.e0 == id) ++c;
    if (a.n_extra > 1 && a.e1 == id) ++c;
    if (a.n_extra > 2 && a.e2 == id) ++c;
    if (a.n_extra > 3 && a.e3 == id) ++c;
    if (a.n_extra > 4 && a.e4 == id) ++c;
    if (c == 0) return;
    float x = a.logits[i];
    if (a.rep != 1.0f) x = x > 0.0f ? x / a.rep : x * a.rep;
    x -= a.freq * (float)c + a.pres;
    a.logits[i] = x;
}
