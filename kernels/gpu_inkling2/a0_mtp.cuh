// gpu_inkling2/a0_mtp.cuh — the bundled DSpark / MTP drafter (`mtp.safetensors`) and the exact speculative accept
// (leg S6). Section of gpu_inkling2.cu, band a0-af, kernels `ink6_*`. Host side: src/inkling2/mtp/gpu.rs.
// Reference: the external SGLang `inkling2_mtp.py` (+ DFlash/DSpark draft model, FA4 block-anchored windows); the
// CPU oracle src/inkling2/mtp/cpu.rs is op for op.
//
// Drafter shapes: hidden 2304, 3 layers (24 q / 8 kv heads x 128, SwiGLU 9216), layers 0 and 2 see the last 1024
// context positions, layer 1 all of them; the 8 block rows (anchor + 7 mask rows) attend NON-causally over
// context ++ block. Context K / V come from the target's tap features (written at commit), the block's own K / V live
// in a per-call scratch (never in the cache: the cache only ever holds committed context rows).
//
//   ink6_tap_store   f32 rows -> bf16 tap slot t of [rows][4][D] (the target's tap capture, state ENTERING a visit)
//   ink6_mix_norm    rmsnorm(sum_t alpha[l][t] * tap[t]) for the drafter layers l (+ the Markov row 3), bf16/f32 out
//   ink6_ctx_kv      context K (k_norm_ctx + RoPE) / V -> the drafter cache, row = pos & cap_mask (bf16)
//   ink6_block_init  block rows 1..7 = mask_embed (row 0 = embed_norm(anchor), written by ink_embed_norm)
//   ink6_qk_rope     block q / k head norms + RoPE, v copy (q fp32, block k / v bf16 scratch)
//   ink6_dattn       split-K non-causal attention (8 block rows x 3 q heads per kv head) -> partials
//   ink6_dattn_comb  merge the splits -> bf16 [8][3072]
//   ink6_add_norm    h += delta (optional), y = rmsnorm(h) (bf16 / f32 out)
//   ink6_silu_mul    SwiGLU of the fused gate|up projection -> bf16
//   ink6_rnn_in      Markov step input [state | markov_w1[prev] | hidden | f_rnn] (bf16)
//   ink6_rnn_gates   (state, latent) = gates(rnn_proj + bias, state), every op rounded to bf16 like the reference
//   ink6_fold        step logits = softcap(base + fold_bias + latent . fold_w) + per-CTA (max, sumexp) partials
//   ink6_lse         per-CTA (max, sumexp) partials of rows x * it
//   ink6_lse_fin     -> lse per row (+ the top probability = the drafter's confidence)
//   ink6_accept      sequential exact accept (greedy: equality with the target argmax; sampled: u < p(d) / q(d))
//   ink6_final_part  the emitted token after the accepted drafts: residual (p - q)+ draw / plain p draw (Gumbel)

// ---------------------------------------------------------------------------------------------------------- taps
extern "C" __global__ __launch_bounds__(256) void ink6_tap_store(const float* __restrict__ x, uint16_t* __restrict__ taps,
        int D, int slot) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.x;
    const float* xr = x + (size_t)r * D;
    uint16_t* o = taps + ((size_t)r * 4 + slot) * D;
    for (int c = threadIdx.x; c < D; c += 256) o[c] = ika_f2b(xr[c]);
}

// grid (rows, nl), block 256. alpha [4][4] fp32 (bf16-valued, the reference rounds softmax(fuse_weight) to bf16);
// w = the dflare_norm weight of output l (lw[l]); out l at ob + l * ldo (bf16) and / or of + l * ldo (fp32).
// The reference (mix_rmsnorm triton): mixed = sum_t x_t * alpha_t (fp32), y = mixed * rsqrt(mean + eps) * w.
struct MixArgs { const float* w[4]; int al[4]; };
extern "C" __global__ __launch_bounds__(256) void ink6_mix_norm(const uint16_t* __restrict__ taps, const float* __restrict__ alpha,
        MixArgs a, uint16_t* __restrict__ ob, float* __restrict__ of, long long ldo, int D, float eps) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float red[8];
    const int r = blockIdx.x, l = blockIdx.y;
    const int ar = a.al[l];
    const float al0 = alpha[ar * 4 + 0], al1 = alpha[ar * 4 + 1], al2 = alpha[ar * 4 + 2], al3 = alpha[ar * 4 + 3];
    const uint16_t* t = taps + (size_t)r * 4 * D;
    float v[9];
    float ss[1] = {0.f};
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int c = threadIdx.x + 256 * i;
        float m = 0.f;
        if (c < D) {
            m = ika_b2f(t[c]) * al0;
            m = fmaf(ika_b2f(t[D + c]), al1, m);
            m = fmaf(ika_b2f(t[2 * D + c]), al2, m);
            m = fmaf(ika_b2f(t[3 * D + c]), al3, m);
            ss[0] = fmaf(m, m, ss[0]);
        }
        v[i] = m;
    }
    ika_bsum<1, 8>(ss, red);
    const float rs = rsqrtf(ss[0] / (float)D + eps);
    const float* w = a.w[l];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int c = threadIdx.x + 256 * i;
        if (c < D) {
            const float y = v[i] * rs * w[c];
            if (ob) ob[l * ldo + (size_t)r * D + c] = ika_f2b(y);
            if (of) of[l * ldo + (size_t)r * D + c] = y;
        }
    }
}

// RoPE (neox halves, rotary 128, fp32 angle p * inv_freq like the reference's fp32 cos/sin cache).
__device__ __forceinline__ void ink6_rope_pair(float& a, float& b, int pos, int i, float theta) {
    const float inv = 1.0f / powf(theta, (float)(2 * i) / 128.0f);
    const float ang = (float)pos * inv;
    float s, c;
    sincosf(ang, &s, &c);
    const float x1 = a, x2 = b;
    a = x1 * c - x2 * s;
    b = x2 * c + x1 * s;
}

// One head vector (128) per warp: rmsnorm(w) (+ optional RoPE). Lane holds dims lane and lane + 64 (+32, +96).
__device__ __forceinline__ void ink6_head_norm_rope(float (&x)[4], const float* __restrict__ w, float eps, int pos,
        float theta, bool rope) {
    const int lane = threadIdx.x & 31;
    float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    ss = ika_wsum(ss);
    const float r = rsqrtf(ss / 128.0f + eps);
    // dims: x[0] = lane, x[1] = lane + 32, x[2] = lane + 64, x[3] = lane + 96
    x[0] = ika_b2f(ika_f2b(x[0] * r * w[lane]));
    x[1] = ika_b2f(ika_f2b(x[1] * r * w[lane + 32]));
    x[2] = ika_b2f(ika_f2b(x[2] * r * w[lane + 64]));
    x[3] = ika_b2f(ika_f2b(x[3] * r * w[lane + 96]));
    if (rope) {
        ink6_rope_pair(x[0], x[2], pos, lane, theta);
        ink6_rope_pair(x[1], x[3], pos, lane + 32, theta);
    }
}

// Context K / V: y = [rows][ldy] fp32 (k at col 0, v at col 1024 of each layer's projection), positions p0 + r.
// grid (rows, 8 kv heads / 8 warps = 1, nl) -> one warp per kv head. Cache [cap][8][128] bf16, row = pos & cap_mask.
struct CtxArgs { const float* y[3]; const float* kn[3]; uint16_t* kc[3]; uint16_t* vc[3]; unsigned cap_mask[3]; };
extern "C" __global__ __launch_bounds__(256) void ink6_ctx_kv(CtxArgs a, int ldy, int p0, float eps, float theta) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.x, l = blockIdx.y, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = p0 + r;
    const float* yr = a.y[l] + (size_t)r * ldy;
    float k[4], v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        k[i] = ika_b2f(ika_f2b(yr[h * 128 + lane + 32 * i]));          // the reference's bf16 projection output
        v[i] = yr[1024 + h * 128 + lane + 32 * i];
    }
    ink6_head_norm_rope(k, a.kn[l], eps, pos, theta, true);
    const size_t row = (size_t)((unsigned)pos & a.cap_mask[l]) * 1024 + h * 128;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        a.kc[l][row + lane + 32 * i] = ika_f2b(k[i]);
        a.vc[l][row + lane + 32 * i] = ika_f2b(v[i]);
    }
}

// h rows 1..B-1 = mask_embed (bf16 values); row 0 is written by ink_embed_norm. grid (B - 1), block 256.
extern "C" __global__ __launch_bounds__(256) void ink6_block_init(const float* __restrict__ mask, float* __restrict__ h, int D) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    float* o = h + (size_t)(blockIdx.x + 1) * D;
    for (int c = threadIdx.x; c < D; c += 256) o[c] = mask[c];
}

// Block q / k / v: y [B][ldy] fp32 = [q 3072 | k 1024 | v 1024]. grid (B), block 32 * 8: warp w handles q heads w,
// w + 8, w + 16 and kv head w. q -> qf [B][24][128] fp32 (bf16-valued), k / v -> bk / bv [B][8][128] bf16.
extern "C" __global__ __launch_bounds__(256) void ink6_qk_rope(const float* __restrict__ y, int ldy, const float* __restrict__ qn,
        const float* __restrict__ kn, float* __restrict__ qf, uint16_t* __restrict__ bk, uint16_t* __restrict__ bv, int p0,
        float eps, float theta) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = p0 + r;
    const float* yr = y + (size_t)r * ldy;
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int h = w + 8 * j;
        float q[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = ika_b2f(ika_f2b(yr[h * 128 + lane + 32 * i]));
        ink6_head_norm_rope(q, qn, eps, pos, theta, true);
        #pragma unroll
        for (int i = 0; i < 4; ++i) qf[((size_t)r * 24 + h) * 128 + lane + 32 * i] = ika_b2f(ika_f2b(q[i]));
    }
    float k[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) k[i] = ika_b2f(ika_f2b(yr[3072 + w * 128 + lane + 32 * i]));
    ink6_head_norm_rope(k, kn, eps, pos, theta, true);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        bk[((size_t)r * 8 + w) * 128 + lane + 32 * i] = ika_f2b(k[i]);
        bv[((size_t)r * 8 + w) * 128 + lane + 32 * i] = ika_f2b(yr[4096 + w * 128 + lane + 32 * i]);
    }
}

// Split-K block attention. Keys: context rows lo..P-1 (cache row = pos & cap_mask), then the 8 block rows. grid
// (8 kv heads, S splits), block 256 (8 warps): warp w = block row w, its 3 q heads 3 kvh .. 3 kvh + 2.
// Per 32-key tile: lane k scores key k (transpose-reduced partial dots), online softmax in exp2, V accumulated
// 4 dims per lane. Partials: pm / pl [S][8 rows][24 heads], pacc [S][8][24][128].
extern "C" __global__ __launch_bounds__(256) void ink6_dattn(const float* __restrict__ qf, const uint16_t* __restrict__ kc,
        const uint16_t* __restrict__ vc, unsigned cap_mask, const uint16_t* __restrict__ bk, const uint16_t* __restrict__ bv,
        int lo, int P, int per, float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ __align__(16) uint16_t sk[32][128];
    __shared__ __align__(16) uint16_t sv[32][128];
    __shared__ float sp[8][3][32];
    const int kvh = blockIdx.x, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nctx = P - lo, ntot = nctx + 8;
    const int t0 = s * per, t1 = min(ntot, t0 + per);
    const float sc = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
    float q[3][4];
    #pragma unroll
    for (int j = 0; j < 3; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i) q[j][i] = qf[((size_t)w * 24 + kvh * 3 + j) * 128 + lane * 4 + i] * sc;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, lsum[3] = {0.f, 0.f, 0.f};
    float acc[3][4];
    #pragma unroll
    for (int j = 0; j < 3; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;
    for (int tb = t0; tb < t1; tb += 32) {
        const int nk = min(32, t1 - tb);
        __syncthreads();
        // 32 keys x 128 dims bf16 = 8 KB each; 256 threads x 16 B = 4 KB per pass
        for (int e = threadIdx.x; e < 32 * 16; e += 256) {
            const int kk = e >> 4, ch = e & 15;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kk < nk) {
                const int t = tb + kk;
                const uint16_t *kp, *vp;
                if (t < nctx) {
                    const size_t row = (size_t)((unsigned)(lo + t) & cap_mask) * 1024 + kvh * 128;
                    kp = kc + row; vp = vc + row;
                } else {
                    const size_t row = ((size_t)(t - nctx) * 8 + kvh) * 128;
                    kp = bk + row; vp = bv + row;
                }
                kv = *reinterpret_cast<const uint4*>(kp + ch * 8);
                vv = *reinterpret_cast<const uint4*>(vp + ch * 8);
            }
            *reinterpret_cast<uint4*>(&sk[kk][ch * 8]) = kv;
            *reinterpret_cast<uint4*>(&sv[kk][ch * 8]) = vv;
        }
        __syncthreads();
        // partial dots: lane's 4 dims (lane*4 .. +3) against every key, 3 queries
        float d[3][32];
        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            const uint2 kr = *reinterpret_cast<const uint2*>(&sk[kk][lane * 4]);
            const float k0 = ika_b2f(kr.x & 0xffff), k1 = ika_b2f(kr.x >> 16), k2 = ika_b2f(kr.y & 0xffff), k3 = ika_b2f(kr.y >> 16);
            #pragma unroll
            for (int j = 0; j < 3; ++j) d[j][kk] = fmaf(q[j][0], k0, fmaf(q[j][1], k1, fmaf(q[j][2], k2, q[j][3] * k3)));
        }
        // transpose reduction: lane L ends with the full dot of key L
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            #pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                const bool up = (lane & o) != 0;
                #pragma unroll
                for (int kk = 0; kk < o; ++kk) {
                    const float send = up ? d[j][kk] : d[j][kk + o];
                    const float keep = up ? d[j][kk + o] : d[j][kk];
                    d[j][kk] = keep + __shfl_xor_sync(0xffffffffu, send, o);
                }
            }
        }
        // after the tree lane L holds key index L in d[j][0]
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float sv_ = lane < nk ? d[j][0] : -INFINITY;
            const float mt = ika_wmax(sv_);
            const float mn = fmaxf(m[j], mt);
            const float p = lane < nk ? exp2f(sv_ - mn) : 0.f;
            const float corr = exp2f(m[j] - mn);
            lsum[j] = lsum[j] * corr + ika_wsum(p);
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][i] *= corr;
            m[j] = mn;
            sp[w][j][lane] = p;
        }
        __syncwarp();
        for (int kk = 0; kk < nk; ++kk) {
            const uint2 vr = *reinterpret_cast<const uint2*>(&sv[kk][lane * 4]);
            const float v0 = ika_b2f(vr.x & 0xffff), v1 = ika_b2f(vr.x >> 16), v2 = ika_b2f(vr.y & 0xffff), v3 = ika_b2f(vr.y >> 16);
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float p = sp[w][j][kk];
                acc[j][0] = fmaf(p, v0, acc[j][0]); acc[j][1] = fmaf(p, v1, acc[j][1]);
                acc[j][2] = fmaf(p, v2, acc[j][2]); acc[j][3] = fmaf(p, v3, acc[j][3]);
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int h = kvh * 3 + j;   // GQA: q heads 3 kvh .. 3 kvh + 2 share kv head kvh
        const size_t qi = ((size_t)s * 8 + w) * 24 + h;
        if (lane == 0) { pm[qi] = m[j]; pl[qi] = lsum[j]; }
        #pragma unroll
        for (int i = 0; i < 4; ++i) pacc[qi * 128 + lane * 4 + i] = acc[j][i];
    }
}

// grid (24 heads, 8 rows), block 128: merge S splits -> out bf16 [8][24 * 128].
extern "C" __global__ __launch_bounds__(128) void ink6_dattn_comb(const float* __restrict__ pm, const float* __restrict__ pl,
        const float* __restrict__ pacc, int S, uint16_t* __restrict__ out) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < S; ++s) M = fmaxf(M, pm[((size_t)s * 8 + r) * 24 + h]);
    float L = 0.f, a = 0.f;
    for (int s = 0; s < S; ++s) {
        const size_t qi = ((size_t)s * 8 + r) * 24 + h;
        const float mm = pm[qi];
        if (mm == -INFINITY) continue;
        const float c = exp2f(mm - M);
        L = fmaf(pl[qi], c, L);
        a = fmaf(pacc[qi * 128 + d], c, a);
    }
    out[(size_t)r * 3072 + h * 128 + d] = ika_f2b(a / L);
}

// h += delta (delta nullable), y = w * rmsnorm(h). grid (rows), block 256.
extern "C" __global__ __launch_bounds__(256) void ink6_add_norm(float* __restrict__ h, const float* __restrict__ delta,
        const float* __restrict__ w, uint16_t* __restrict__ ob, float* __restrict__ of, int D, float eps) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float red[8];
    const int r = blockIdx.x;
    float* hr = h + (size_t)r * D;
    float v[9];
    float ss[1] = {0.f};
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int c = threadIdx.x + 256 * i;
        float x = 0.f;
        if (c < D) {
            x = hr[c];
            if (delta) { x += delta[(size_t)r * D + c]; hr[c] = x; }
            ss[0] = fmaf(x, x, ss[0]);
        }
        v[i] = x;
    }
    ika_bsum<1, 8>(ss, red);
    const float rs = rsqrtf(ss[0] / (float)D + eps);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int c = threadIdx.x + 256 * i;
        if (c < D) {
            const float y = w[c] * (v[i] * rs);
            if (ob) ob[(size_t)r * D + c] = ika_f2b(y);
            if (of) of[(size_t)r * D + c] = y;
        }
    }
}

// SwiGLU: y [rows][ldy] = [gate (I) | up (I)] fp32 -> bf16 silu(bf16 g) * bf16 u. grid (ceil(I/256), rows).
extern "C" __global__ __launch_bounds__(256) void ink6_silu_mul(const float* __restrict__ y, int ldy, int I, uint16_t* __restrict__ out) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= I) return;
    const float g = ika_b2f(ika_f2b(y[(size_t)r * ldy + c]));
    const float u = ika_b2f(ika_f2b(y[(size_t)r * ldy + I + c]));
    const float sg = ika_b2f(ika_f2b(g / (1.0f + expf(-g))));
    out[(size_t)r * I + c] = ika_f2b(sg * u);
}

// Markov input row (bf16 [5120]) for step j: [state | markov_w1[tok[j]] | hidden (bf16 row) | f_rnn (bf16)].
extern "C" __global__ __launch_bounds__(256) void ink6_rnn_in(const uint16_t* __restrict__ state, const int* __restrict__ tok, int j,
        const uint16_t* __restrict__ w1, const uint16_t* __restrict__ hid, const uint16_t* __restrict__ frnn, uint16_t* __restrict__ out,
        int R, int D) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int t = tok[j];
    for (int c = threadIdx.x + blockIdx.x * 256; c < 2 * R + 2 * D; c += 256 * gridDim.x) {
        uint16_t v;
        if (c < R) v = state[c];
        else if (c < 2 * R) v = w1[(size_t)t * R + c - R];
        else if (c < 2 * R + D) v = hid[c - 2 * R];
        else v = frnn[c - 2 * R - D];
        out[c] = v;
    }
}

// The reference's _rnn_gates (each op rounded to bf16): proj = y + bias rounded to bf16 (Linear output bf16).
extern "C" __global__ __launch_bounds__(256) void ink6_rnn_gates(const float* __restrict__ y, const float* __restrict__ bias,
        uint16_t* __restrict__ state, uint16_t* __restrict__ latent, int R) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int c = threadIdx.x;
    if (c >= R) return;
    const float g_raw = ika_b2f(ika_f2b(y[c] + bias[c]));
    const float c_raw = ika_b2f(ika_f2b(y[R + c] + bias[R + c]));
    const float o_raw = ika_b2f(ika_f2b(y[2 * R + c] + bias[2 * R + c]));
    const float st = ika_b2f(state[c]);
    const float gate = ika_b2f(ika_f2b(__frcp_rn(1.0f + expf(-g_raw))));
    const float kept = ika_b2f(ika_f2b(gate * st));
    const float blend = ika_b2f(ika_f2b(ika_b2f(ika_f2b(1.0f - gate)) * ika_b2f(ika_f2b(tanhf(c_raw)))));
    state[c] = ika_f2b(kept + blend);
    latent[c] = ika_f2b(tanhf(o_raw));
}

// Step logits: out[i] = cap * tanh((base[i] + fb[i] + latent . fw[i]) / cap), i < N; fw [N][256] bf16 rows, fb fp32.
// One warp per vocab row (lane = 8 consecutive k), 4 rows in flight. Partials per CTA of (max, sumexp) of out * it
// (it = inverse temperature, or 1 for a greedy row: the confidence is always read at the sampling temperature).
extern "C" __global__ __launch_bounds__(256) void ink6_fold(const uint16_t* __restrict__ fw, const float* __restrict__ fb,
        const float* __restrict__ base, const uint16_t* __restrict__ latent, float* __restrict__ out, int N, float cap,
        float it, float* __restrict__ pmax, float* __restrict__ psum) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float sm[8], ss_[8];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint4 lr = *reinterpret_cast<const uint4*>(latent + lane * 8);
    float lv[8];
    lv[0] = ika_b2f(lr.x & 0xffff); lv[1] = ika_b2f(lr.x >> 16); lv[2] = ika_b2f(lr.y & 0xffff); lv[3] = ika_b2f(lr.y >> 16);
    lv[4] = ika_b2f(lr.z & 0xffff); lv[5] = ika_b2f(lr.z >> 16); lv[6] = ika_b2f(lr.w & 0xffff); lv[7] = ika_b2f(lr.w >> 16);
    const int nw = gridDim.x * 8;
    float mx = -INFINITY, sum = 0.f;
    for (int i0 = (blockIdx.x * 8 + w) * 4; i0 < N; i0 += nw * 4) {
        uint4 wr[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = i0 + u;
            wr[u] = i < N ? ika_ldw4(fw + (size_t)i * 256 + lane * 8) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 q = wr[u];
            float d = ika_b2f(q.x & 0xffff) * lv[0];
            d = fmaf(ika_b2f(q.x >> 16), lv[1], d); d = fmaf(ika_b2f(q.y & 0xffff), lv[2], d); d = fmaf(ika_b2f(q.y >> 16), lv[3], d);
            d = fmaf(ika_b2f(q.z & 0xffff), lv[4], d); d = fmaf(ika_b2f(q.z >> 16), lv[5], d); d = fmaf(ika_b2f(q.w & 0xffff), lv[6], d);
            d = fmaf(ika_b2f(q.w >> 16), lv[7], d);
            d = ika_wsum(d);
            const int i = i0 + u;
            if (i < N) {
                const float v = tanhf((base[i] + fb[i] + d) / cap) * cap;
                if (lane == 0) out[i] = v;
                const float x = v * it;
                if (x > mx) { sum = sum * __expf(mx - x) + 1.f; mx = x; } else { sum += __expf(x - mx); }
            }
        }
    }
    // lane 0 of each warp holds the warp's stats (every lane computed the same values)
    if (lane == 0) { sm[w] = mx; ss_[w] = sum; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int k = 0; k < 8; ++k) M = fmaxf(M, sm[k]);
        float S = 0.f;
        for (int k = 0; k < 8; ++k) if (sm[k] > -INFINITY) S += ss_[k] * __expf(sm[k] - M);
        pmax[blockIdx.x] = M; psum[blockIdx.x] = S;
    }
}

// Partials of (max, sumexp) of x * it over rows. grid (G, rows), block 256; it per row (<= 0 -> 1).
extern "C" __global__ __launch_bounds__(256) void ink6_lse(const float* __restrict__ x, int ldx, int n, const float* __restrict__ inv_t,
        float* __restrict__ pmax, float* __restrict__ psum) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float sm[256], ss_[256];
    const int row = blockIdx.y, G = gridDim.x;
    const int per = (n + G - 1) / G;
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    const float it = inv_t[row] > 0.f ? inv_t[row] : 1.f;
    const float* xr = x + (size_t)row * ldx;
    float mx = -INFINITY, sum = 0.f;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {
        const float v = xr[i] * it;
        if (v > mx) { sum = sum * __expf(mx - v) + 1.f; mx = v; } else { sum += __expf(v - mx); }
    }
    sm[threadIdx.x] = mx; ss_[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float a = sm[threadIdx.x], b = sm[threadIdx.x + s];
            const float M = fmaxf(a, b);
            float S = 0.f;
            if (a > -INFINITY) S += ss_[threadIdx.x] * __expf(a - M);
            if (b > -INFINITY) S += ss_[threadIdx.x + s] * __expf(b - M);
            sm[threadIdx.x] = M; ss_[threadIdx.x] = S;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmax[row * G + blockIdx.x] = sm[0]; psum[row * G + blockIdx.x] = ss_[0]; }
}

// lse[row] = M + log(S) over G partials; conf[row] = top probability = 1 / S'. grid (rows), block 32.
extern "C" __global__ __launch_bounds__(32) void ink6_lse_fin(const float* __restrict__ pmax, const float* __restrict__ psum, int G,
        float* __restrict__ lse, float* __restrict__ conf) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int row = blockIdx.x;
    if (threadIdx.x) return;
    float M = -INFINITY;
    for (int g = 0; g < G; ++g) M = fmaxf(M, pmax[row * G + g]);
    double S = 0.0;
    for (int g = 0; g < G; ++g) { const float m = pmax[row * G + g]; if (m > -INFINITY) S += (double)psum[row * G + g] * exp((double)(m - M)); }
    lse[row] = M + (float)log(S);
    if (conf) conf[row] = (float)(1.0 / S);
}

__device__ __forceinline__ double ink6_u01(unsigned long long seed, unsigned long long k) {
    const unsigned long long h = ink3_mix(seed ^ ink3_mix(k ^ 0x5bd1e9955bd1e995ULL));
    return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Sequential accept of c drafts (tok[1..c]) against the target rows 0..c-1 (tl, ldl) and the drafter rows (dl [j][V]).
// greedy (it <= 0): accept while tok[j + 1] == tgt[j]. sampled: accept draft j + 1 iff u_j < p_j(d) / q_j(d), with
// p_j = softmax(tl_j * it) (lse_p[j]) and q_j = softmax(dl_j * it) (lse_q[j]) — the drafter drew d from q_j.
// res[0] = accepted count a. One thread.
extern "C" __global__ void ink6_accept(const int* __restrict__ tok, int c, const int* __restrict__ tgt, const float* __restrict__ tl,
        int ldl, const float* __restrict__ dl, int ldd, const float* __restrict__ lse_p, const float* __restrict__ lse_q, float it,
        unsigned long long seed, int* __restrict__ res) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    if (threadIdx.x || blockIdx.x) return;
    int a = 0;
    for (int j = 0; j < c; ++j) {
        const int d = tok[j + 1];
        bool ok;
        if (it <= 0.f) ok = d == tgt[j];
        else {
            const double lp = (double)(tl[(size_t)j * ldl + d] * it) - (double)lse_p[j];
            const double lq = (double)(dl[(size_t)j * ldd + d] * it) - (double)lse_q[j];
            const double u = ink6_u01(seed, (unsigned long long)j);
            ok = log(u) < lp - lq;
        }
        if (!ok) break;
        ++a;
    }
    res[0] = a;
}

// The emitted token after a = res[0] accepted drafts: a < c -> a draw from (p_a - q_a)+, a == c -> a draw from p_c.
// Gumbel-max over log w_i (w = 0 -> excluded). grid (G), block 256 -> (pmax, pidx) partials for ink3_gumbel_pick.
extern "C" __global__ __launch_bounds__(256) void ink6_final_part(const float* __restrict__ tl, int ldl, const float* __restrict__ dl,
        int ldd, int n, const float* __restrict__ lse_p, const float* __restrict__ lse_q, float it, const int* __restrict__ res, int c,
        unsigned long long seed, float* __restrict__ pmax, int* __restrict__ pidx) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float sv[256];
    __shared__ int si[256];
    const int a = res[0];
    const bool resid = a < c;
    const int G = gridDim.x;
    const int per = (n + G - 1) / G;
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    const float* tr = tl + (size_t)a * ldl;
    const float* dr = dl + (size_t)a * ldd;
    const float lp0 = lse_p[a], lq0 = resid ? lse_q[a] : 0.f;
    const unsigned long long sd = ink3_mix(seed ^ (unsigned long long)(a + 1) * 0x9e3779b97f4a7c15ULL);
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {
        float lw;
        const float lp = tr[i] * it - lp0;
        if (resid) {
            const float lq = dr[i] * it - lq0;
            if (lp <= lq) continue;
            // log(p - q) = lp + log1p(-exp(lq - lp))
            lw = lp + log1pf(-expf(lq - lp));
        } else lw = lp;
        const unsigned long long h = ink3_mix(sd ^ ink3_mix((unsigned long long)i));
        const double wv = ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        const double e = -log1p(-wv);
        const float v = lw - (float)log(e);
        if (v > best) { best = v; bi = i; }
    }
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float x = sv[threadIdx.x], y = sv[threadIdx.x + s];
            const int ix = si[threadIdx.x], iy = si[threadIdx.x + s];
            if (y > x || (y == x && iy < ix)) { sv[threadIdx.x] = y; si[threadIdx.x] = iy; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}

// Load-time fold: y [N][ldy] fp32 = unembed . [hidden_out.weight^T | bias] -> fw [N][256] bf16, fb [N] fp32.
extern "C" __global__ __launch_bounds__(256) void ink6_fold_pack(const float* __restrict__ y, int ldy, uint16_t* __restrict__ fw,
        float* __restrict__ fb, int N) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int i = blockIdx.x;
    if (i >= N) return;
    fw[(size_t)i * 256 + threadIdx.x] = ika_f2b(y[(size_t)i * ldy + threadIdx.x]);
    if (threadIdx.x == 0) fb[i] = ika_b2f(ika_f2b(y[(size_t)i * ldy + 256]));
}

// ---------------------------------------------------------------------------------------------- multi-lane draft
// The drafter's base logits for up to 32 rows (4 lanes x 7 block rows) in ONE pass over the bf16 head (the B32
// layout of ink2_lm_head): two 16-row token tiles per warp, each fp32 row split hi + lo bf16 (as ink2_lm_head), no
// softcap (the cap comes after the Markov bias). logits [t][ldo]. grid ceil(N/128), block 256.
extern "C" __global__ __launch_bounds__(256) void ink6_head32(const uint16_t* __restrict__ W, const float* __restrict__ X,
        float* __restrict__ logits, int M, int K, int N, int ldo) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    if (n0w >= N) return;
    float acc[2][2][4];   // [token tile][n half][4]
    #pragma unroll
    for (int q = 0; q < 2; ++q)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[q][h][r] = 0.f;
    const int nq = M > 16 ? 2 : 1;
    const float* xa[2]; const float* xb[2];
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int ta = min(16 * q + gid, M - 1), tb = min(16 * q + gid + 8, M - 1);
        xa[q] = X + (size_t)ta * K + 8 * tig;
        xb[q] = X + (size_t)tb * K + 8 * tig;
    }
    const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
    for (int k = 0; k < K; k += 32) {
        const uint4 b0 = ik2m_ldw4(wt + (size_t)k * 16), b1 = ik2m_ldw4(wt + (size_t)k * 16 + 256);
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (q >= nq) break;
            uint4 ah, al, bh, bl;
            ik2m_split8(__ldg(reinterpret_cast<const float4*>(xa[q] + k)), __ldg(reinterpret_cast<const float4*>(xa[q] + k + 4)), ah, al);
            ik2m_split8(__ldg(reinterpret_cast<const float4*>(xb[q] + k)), __ldg(reinterpret_cast<const float4*>(xb[q] + k + 4)), bh, bl);
            ik2m_bf_group(acc[q][0], ah, al, bh, bl, b0);
            ik2m_bf_group(acc[q][1], ah, al, bh, bl, b1);
        }
    }
    #pragma unroll
    for (int q = 0; q < 2; ++q)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int n = n0w + 8 * h + 2 * tig + c;
                if (n >= N) continue;
                #pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const int t = 16 * q + gid + 8 * s;
                    if (t < M) logits[(size_t)t * ldo + n] = acc[q][h][2 * s + c];
                }
            }
}

// One Markov step for up to 4 lanes in one pass over the folded head: out_l[i] = cap tanh((base_l[i] + fb[i] +
// latent_l . fw[i]) / cap), per-lane (max, sumexp) partials of out * it_l. latent [L][256] bf16. grid FOLD_G, block 256.
struct FoldArgs { const float* base[4]; float* out[4]; float it[4]; float* pmax[4]; float* psum[4]; };
extern "C" __global__ __launch_bounds__(256) void ink6_fold4(const uint16_t* __restrict__ fw, const float* __restrict__ fb,
        const uint16_t* __restrict__ latent, FoldArgs a, int L, int N, float cap) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float sm[4][8], ss_[4][8];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float lv[4][8];
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        if (l < L) {
            const uint4 lr = *reinterpret_cast<const uint4*>(latent + l * 256 + lane * 8);
            lv[l][0] = ika_b2f(lr.x & 0xffff); lv[l][1] = ika_b2f(lr.x >> 16); lv[l][2] = ika_b2f(lr.y & 0xffff); lv[l][3] = ika_b2f(lr.y >> 16);
            lv[l][4] = ika_b2f(lr.z & 0xffff); lv[l][5] = ika_b2f(lr.z >> 16); lv[l][6] = ika_b2f(lr.w & 0xffff); lv[l][7] = ika_b2f(lr.w >> 16);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) lv[l][i] = 0.f;
        }
    }
    const int nw = gridDim.x * 8;
    float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, sum[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i0 = (blockIdx.x * 8 + w) * 4; i0 < N; i0 += nw * 4) {
        uint4 wr[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = i0 + u;
            wr[u] = i < N ? ika_ldw4(fw + (size_t)i * 256 + lane * 8) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 q = wr[u];
            const float f[8] = {ika_b2f(q.x & 0xffff), ika_b2f(q.x >> 16), ika_b2f(q.y & 0xffff), ika_b2f(q.y >> 16),
                                ika_b2f(q.z & 0xffff), ika_b2f(q.z >> 16), ika_b2f(q.w & 0xffff), ika_b2f(q.w >> 16)};
            const int i = i0 + u;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                if (l >= L) break;
                float d = f[0] * lv[l][0];
                #pragma unroll
                for (int e = 1; e < 8; ++e) d = fmaf(f[e], lv[l][e], d);
                d = ika_wsum(d);
                if (i < N) {
                    const float v = tanhf((a.base[l][i] + fb[i] + d) / cap) * cap;
                    if (lane == l) a.out[l][i] = v;
                    const float x = v * a.it[l];
                    if (x > mx[l]) { sum[l] = sum[l] * __expf(mx[l] - x) + 1.f; mx[l] = x; } else { sum[l] += __expf(x - mx[l]); }
                }
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) { sm[l][w] = mx[l]; ss_[l][w] = sum[l]; }
    }
    __syncthreads();
    if (threadIdx.x < L) {
        const int l = threadIdx.x;
        float M = -INFINITY;
        for (int k = 0; k < 8; ++k) M = fmaxf(M, sm[l][k]);
        float S = 0.f;
        for (int k = 0; k < 8; ++k) if (sm[l][k] > -INFINITY) S += ss_[l][k] * __expf(sm[l][k] - M);
        a.pmax[l][blockIdx.x] = M; a.psum[l][blockIdx.x] = S;
    }
}
