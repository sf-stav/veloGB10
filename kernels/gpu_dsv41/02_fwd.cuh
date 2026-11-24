// =============================================================================
// kernels/gpu_dsv41/02_fwd.cuh — band 02-04, leg SL: the glue of the per-layer forward (src/dsv41/fwd/).
// Everything the other bands do not do between two of their entries:
//   * d41f_rope_rows   RoPE on the last 64 dims of every head of bf16 rows (q of the attention / DSpark),
//                      forward or inverse, at per-row positions; torch-on-CUDA arithmetic (REFb's): the
//                      angle fp32(pos) * freq, sincos of it, the complex multiply as torch's CUDA build
//                      contracts it (re = fma(a, c, -(b s)), im = fma(b, c, a s)), bf16 round.
//   * d41f_swiglu      shared expert: bf16(silu(min(g, L)) * clamp(u, -L, L)) of the bf16 w1 / w3 outputs,
//                      silu = x / (1 + expf(-x)) as torch's CUDA kernel, the product a separate op.
//   * d41f_rope_table  the RoPE inverse-frequency table as torch-on-CUDA builds it (src/dsv41/rope.rs)
//   * d41f_add_round   MoE output: bf16(y_f32 + f32(s_bf16)) (the reference's `y += shared(x)`, then
//                      `type_as(x)`); s may be null (a plain f32 -> bf16 cast).
// One by-value argument struct per entry (mirrored in src/dsv41/fwd/glue.rs). No runtime-indexed
// per-thread arrays; elementwise kernels, 256 threads.
// =============================================================================
#pragma once

struct d41f_rope_args {
    __nv_bfloat16* x;        // [rows][ld] bf16, rotated in place
    const int* pos;          // [rows] absolute positions
    long long ld;            // row stride (elements)
    int rows, heads, head_dim, inverse;   // the last 64 dims of each head [h * head_dim, (h + 1) * head_dim)
    float freq[32];          // the RoPE class's inverse frequencies
};

// grid (rows, ceil(heads * 32 / 256)), 256 threads: one thread per (head, rope pair).
extern "C" __global__ void __launch_bounds__(256) d41f_rope_rows(d41f_rope_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x;
    const int t = blockIdx.y * 256 + threadIdx.x;
    if (r >= a.rows || t >= a.heads * 32) return;
    const int h = t >> 5, i = t & 31;
    __nv_bfloat16* p = a.x + (long long)r * a.ld + (long long)h * a.head_dim + (a.head_dim - 64) + 2 * i;
    const float x0 = __bfloat162float(p[0]), x1 = __bfloat162float(p[1]);
    float c, s, re = x0, im = x1;
    d41_rope_cs(a.pos[r], a.freq[i], c, s);
    if (a.inverse) d41_rope_inv(re, im, c, s); else d41_rope_fwd(re, im, c, s);
    p[0] = __float2bfloat16_rn(re);
    p[1] = __float2bfloat16_rn(im);
}

// The RoPE table of src/dsv41/rope.rs, computed as torch computes it ON CUDA: 1 / powf(base, i * 2 / dim) (CUDA
// powf, IEEE reciprocal), YaRN with the reciprocal-multiply ramp and separate multiply / add (no FMA). One warp.
struct d41f_rope_table_args { float* out; float base, inv_dim, low, inv_den, inv_factor; int yarn, half, pad; };
extern "C" __global__ void __launch_bounds__(32) d41f_rope_table(d41f_rope_table_args a) {
    D41_PDL_ENTRY();
    const int i = threadIdx.x;
    if (i >= a.half) return;
    const float e = __fmul_rn((float)(2 * i), a.inv_dim);
    float f = __fdiv_rn(1.0f, powf(a.base, e));
    if (a.yarn) {
        const float ramp = fminf(fmaxf(__fmul_rn(__fsub_rn((float)i, a.low), a.inv_den), 0.0f), 1.0f);
        const float smooth = __fsub_rn(1.0f, ramp);
        f = __fadd_rn(__fmul_rn(__fmul_rn(f, a.inv_factor), __fsub_rn(1.0f, smooth)), __fmul_rn(f, smooth));
    }
    a.out[i] = f;
}

struct d41f_swiglu_args {
    const __nv_bfloat16* g;  // [rows][ldg] w1 output
    const __nv_bfloat16* u;  // [rows][ldu] w3 output
    __nv_bfloat16* out;      // [rows][ldo]
    long long ldg, ldu, ldo;
    int rows, n;
    float limit;             // swiglu_limit (0 = no clamp)
};

// grid (ceil(n / 256), rows)
extern "C" __global__ void __launch_bounds__(256) d41f_swiglu(d41f_swiglu_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (r >= a.rows || c >= a.n) return;
    float gate = __bfloat162float(a.g[(long long)r * a.ldg + c]);
    float up = __bfloat162float(a.u[(long long)r * a.ldu + c]);
    if (a.limit > 0.f) { up = fminf(fmaxf(up, -a.limit), a.limit); gate = fminf(gate, a.limit); }
    const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, expf(-gate)));
    a.out[(long long)r * a.ldo + c] = __float2bfloat16_rn(__fmul_rn(silu, up));
}

struct d41f_add_args {
    const float* y;          // [rows][ldy] f32
    const __nv_bfloat16* s;  // [rows][lds] bf16, nullable
    __nv_bfloat16* out;      // [rows][ldo]
    long long ldy, lds, ldo;
    int rows, n;
};

// grid (ceil(n / 256), rows)
extern "C" __global__ void __launch_bounds__(256) d41f_add_round(d41f_add_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (r >= a.rows || c >= a.n) return;
    float v = a.y[(long long)r * a.ldy + c];
    if (a.s) v = __fadd_rn(v, __bfloat162float(a.s[(long long)r * a.lds + c]));
    a.out[(long long)r * a.ldo + c] = __float2bfloat16_rn(v);
}
