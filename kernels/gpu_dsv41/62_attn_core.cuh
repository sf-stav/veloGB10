// =============================================================================
// kernels/gpu_dsv41/62_attn_core.cuh — band 60 (KA): one 64-key block (QK^T, the reference's online
// softmax, P.V), the in-order split merge and the epilogue (sink, divide, inverse RoPE, act_quant).
//
// Warp layout (16 warps, 16 heads):
//   QK^T     warp w = (dq = w % 4, kq = w / 4) computes the partial scores of keys 16kq..16kq+15 (two n8
//            tiles, two independent chains of eight k16 MMAs in k order) over dims 128dq..128dq+127; the
//            partials go to SP.
//   softmax  warp h owns head h: lane L owns keys 2L, 2L+1; s = ((q0 + q1) + q2) + q3 over the four dim
//            quarters, the block max / sum are xor-butterfly reductions over the warp (every lane ends with
//            the same value), the head's running (m, l) live in the owner warp and are published to HS.
//   P.V      warp w owns dims 32w..32w+31 (four n8 tiles, four k16 steps over the 64 keys); its threads
//            hold heads r0 = lane / 4 and r0 + 8 and columns 32w + 8nt + 2(lane % 4) + {0, 1}.
// =============================================================================
#pragma once

#define D41A_FULL 0xffffffffu

// Per-thread softmax state of one split (or of the merged total).
struct d41a_acc {
    float o[4][4];      // P.V accumulators [n-tile][d0 d1 (head r0) d2 d3 (head r0 + 8)]
    float m0, m1, l0, l1;   // max / sum of heads r0, r0 + 8 (copies of the owners' values, taken at a split's end)
    float om, ol;           // the owner warp's running max / sum of head (warp index)
};

__device__ __forceinline__ void d41a_acc_reset(d41a_acc& A) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) { A.o[nt][0] = 0.f; A.o[nt][1] = 0.f; A.o[nt][2] = 0.f; A.o[nt][3] = 0.f; }
    A.m0 = -1e30f; A.m1 = -1e30f; A.l0 = 0.f; A.l1 = 0.f;
    A.om = -1e30f; A.ol = 0.f;
}

// One block: store the fetched keys into T, prefetch the next block's keys (in flight during the
// math), then the reference's per-block step
//   s = (q.k) * scale (masked: -inf); m' = max(m, rowmax s); corr = exp(m - m'); p = exp(s - m');
//   l = fma(l, corr, rowsum p); acc = acc * corr + bf16(p) . V
// Ends with a barrier (T, SP, P, HS are reused by the next block).
__device__ __forceinline__ void d41a_block(unsigned char* sm, const d41a_row& r, d41a_raw& raw, bool more, int nkind, int nbase,
                                           float scale, d41a_acc& A, int* err) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    d41a_store(sm, raw);
    __syncthreads();
    if (more) d41a_fetch(r, nkind, nbase, raw, err);
    const unsigned smb = (unsigned)__cvta_generic_to_shared(sm);
    const int r0 = lane >> 2, cq = lane & 3;

    // ---- partial S = Q K^T over this warp's dim quarter, keys 16 kq .. 16 kq + 15 ----
    {
        const int dq = warp & 3, kq = warp >> 2;
        float sp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const int qrow = (lane & 7) + ((lane >> 3) & 1) * 8, qch = lane >> 4, kch = lane >> 3;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            unsigned q0[4], q1[4];
            d41a_ldsm_x4(q0, smb + D41A_SM_Q + d41a_swz(qrow, 2 * (8 * dq + 2 * k2) + qch));
            d41a_ldsm_x4(q1, smb + D41A_SM_Q + d41a_swz(qrow, 2 * (8 * dq + 2 * k2 + 1) + qch));
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                unsigned b[4];
                d41a_ldsm_x4(b, smb + D41A_SM_T + d41a_swz(16 * kq + 8 * j + (lane & 7), 2 * (8 * dq + 2 * k2) + kch));
                d41a_mma(sp[j], q0, b[0], b[1]);
                d41a_mma(sp[j], q1, b[2], b[3]);
            }
        }
        float* spp = reinterpret_cast<float*>(sm + D41A_SM_SP) + dq * (16 * D41A_SP_STRIDE);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int key = 16 * kq + 8 * j + 2 * cq;
            *reinterpret_cast<float2*>(spp + r0 * D41A_SP_STRIDE + key) = make_float2(sp[j][0], sp[j][1]);
            *reinterpret_cast<float2*>(spp + (r0 + 8) * D41A_SP_STRIDE + key) = make_float2(sp[j][2], sp[j][3]);
        }
    }
    __syncthreads();

    // ---- softmax of head h = warp over the 64 keys; lane owns keys 2 lane, 2 lane + 1 ----
    volatile float* hs = reinterpret_cast<volatile float*>(sm + D41A_SM_HS);
    {
        const int h = warp, k0 = 2 * lane;
        const volatile float* sv = reinterpret_cast<const volatile float*>(sm + D41A_SM_SP) + h * D41A_SP_STRIDE + k0;
        float a0 = sv[0], a1 = sv[1];
#pragma unroll
        for (int dq = 1; dq < 4; ++dq) {
            a0 = __fadd_rn(a0, sv[dq * 16 * D41A_SP_STRIDE]);
            a1 = __fadd_rn(a1, sv[dq * 16 * D41A_SP_STRIDE + 1]);
        }
        const volatile int* kvf = reinterpret_cast<const volatile int*>(sm + D41A_SM_KV);
        const float s0 = kvf[k0] != 0 ? __fmul_rn(a0, scale) : -INFINITY;
        const float s1 = kvf[k0 + 1] != 0 ? __fmul_rn(a1, scale) : -INFINITY;
        float bm = fmaxf(s0, s1);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(D41A_FULL, bm, o));
        const float mn = fmaxf(A.om, bm);
        const float corr = expf(A.om - mn);
        const float p0 = expf(s0 - mn), p1 = expf(s1 - mn);
        float bs = __fadd_rn(p0, p1);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) bs = __fadd_rn(bs, __shfl_xor_sync(D41A_FULL, bs, o));
        A.ol = __fmaf_rn(A.ol, corr, bs);
        A.om = mn;
        // P row h (bf16) over the first 128 B of SP quarter 0 row h (read above by this warp only)
        *reinterpret_cast<volatile unsigned*>(sm + d41a_swz_p(h, lane >> 2) + 4 * (lane & 3)) = d41a_pack_bf16x2(p0, p1);
        if (lane == 0) { hs[h] = corr; hs[16 + h] = mn; hs[32 + h] = A.ol; }
    }
    __syncthreads();
    const float c0 = hs[r0], c1 = hs[r0 + 8];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        A.o[nt][0] = __fmul_rn(A.o[nt][0], c0); A.o[nt][1] = __fmul_rn(A.o[nt][1], c0);
        A.o[nt][2] = __fmul_rn(A.o[nt][2], c1); A.o[nt][3] = __fmul_rn(A.o[nt][3], c1);
    }

    // ---- O += P V, dims 32 warp .. 32 warp + 31 ----
    {
        const int prow = (lane & 7) + ((lane >> 3) & 1) * 8, pch = lane >> 4;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned pa[4];
            d41a_ldsm_x4(pa, smb + d41a_swz_p(prow, 2 * kk + pch));
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned vb[4];
                d41a_ldsm_x4_t(vb, smb + D41A_SM_T + d41a_swz(16 * kk + prow, warp * 4 + 2 * np + pch));
                d41a_mma(A.o[2 * np], pa, vb[0], vb[1]);
                d41a_mma(A.o[2 * np + 1], pa, vb[2], vb[3]);
            }
        }
    }
    __syncthreads();
}

// x / y for the epilogue's normal-range operands (y = the softmax denominator, >= 1 whenever a key is valid;
// x the accumulator): the correctly rounded reciprocal, one product, one FMA correction (the fast path of
// IEEE division without its out-of-line slow path, whose call would cost the kernel a stack frame).
// Equal to the IEEE quotient except in rare last-ulp cases, before the bf16 rounding that follows.
__device__ __forceinline__ float d41a_div(float x, float y) {
    const float r = __frcp_rn(y);
    const float q = __fmul_rn(x, r);
    return __fmaf_rn(__fmaf_rn(-q, y, x), r, q);
}

// The (m, l) of heads r0, r0 + 8 after the latest block, from HS (the owners publish them every block; the
// copies are taken only where a split ends, which keeps four registers free during the block loop).
__device__ __forceinline__ void d41a_acc_load_ml(const unsigned char* sm, d41a_acc& A) {
    const volatile float* hs = reinterpret_cast<const volatile float*>(sm + D41A_SM_HS);
    const int r0 = (threadIdx.x & 31) >> 2;
    A.m0 = hs[16 + r0]; A.m1 = hs[16 + r0 + 8];
    A.l0 = hs[32 + r0]; A.l1 = hs[32 + r0 + 8];
}

// The split merge (the canonical order): T <- merge(T, S), on the P.V accumulators and the (m, l) copies.
//   Mn = max(M, m); e = exp(M - Mn); f = exp(m - Mn); L = fma(L, e, l * f); O = fma(O, e, a * f); M = Mn
__device__ __forceinline__ void d41a_merge(d41a_acc& T, const d41a_acc& S) {
    const float n0 = fmaxf(T.m0, S.m0), n1 = fmaxf(T.m1, S.m1);
    const float e0 = expf(T.m0 - n0), f0 = expf(S.m0 - n0);
    const float e1 = expf(T.m1 - n1), f1 = expf(S.m1 - n1);
    T.l0 = __fmaf_rn(T.l0, e0, __fmul_rn(S.l0, f0));
    T.l1 = __fmaf_rn(T.l1, e1, __fmul_rn(S.l1, f1));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        T.o[nt][0] = __fmaf_rn(T.o[nt][0], e0, __fmul_rn(S.o[nt][0], f0));
        T.o[nt][1] = __fmaf_rn(T.o[nt][1], e0, __fmul_rn(S.o[nt][1], f0));
        T.o[nt][2] = __fmaf_rn(T.o[nt][2], e1, __fmul_rn(S.o[nt][2], f1));
        T.o[nt][3] = __fmaf_rn(T.o[nt][3], e1, __fmul_rn(S.o[nt][3], f1));
    }
    T.m0 = n0;
    T.m1 = n1;
}

// sin / cos of the epilogue: d41_sincos of 00_abi.cuh (moved there from this file by the SL RoPE unification).

// The CTA's 16 sinks into SK (0 for heads past `heads`); read by the epilogue, after the first barrier.
__device__ __forceinline__ void d41a_sinks(unsigned char* sm, const d41_attn_args& a, int g) {
    const int t = threadIdx.x, h = g * D41_ATTN_HG + t;
    if (t < D41_ATTN_HG) reinterpret_cast<float*>(sm + D41A_SM_SK)[t] = h < a.heads ? __ldg(a.sink + h) : 0.f;
}

// pow2ceil(amax * fp32(1/448)) by exponent bits (kernel.py fast_round_scale); returns the exponent
__device__ __forceinline__ int d41a_aq_exp(float amax) {
    const float v = __fmul_rn(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
    const unsigned b = __float_as_uint(v);
    return (int)((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

// The epilogue of the merged state T: L += exp(sink - M); o = bf16(O / L) (d41a_div); then the inverse RoPE on the
// last 64 dims (bf16 -> fp32 rotate -> bf16; warps 14 and 15 own those dims) and act_quant over the warp's
// 32 dims (E4M3 RNE satfinite x pow2 scale, the WO-A input), as enabled. Stores bf16 o (+ the E4M3 codes
// and E8M0 bytes when the caller passes buffers).
// PD1: wo_a's rotated input rows from the staged bf16 o of the CTA's 16 heads (D41A_SM_T, free once every tile is
// consumed): one warp per (head, 128-block), d41x_dd_prep's math (x . suh -> H128 -> fp16) on the stored bits.
__device__ __forceinline__ void d41a_rot_out(const unsigned char* sm, const d41_attn_args& a, int row, int g) {
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* st = reinterpret_cast<const float*>(sm + D41A_SM_T);
    for (int job = warp; job < D41_ATTN_HG * (D41_ATTN_D / 128); job += D41_ATTN_THREADS / 32) {
        const int h = job / (D41_ATTN_D / 128), b = job % (D41_ATTN_D / 128), hg = g * D41_ATTN_HG + h;
        if (hg >= a.heads) continue;
        float v[4], sv[4];
        const __half* su = a.rot_suh[hg / a.rot_hpg] + (hg % a.rot_hpg) * D41_ATTN_D + b * 128 + lane * 4;
#pragma unroll
        for (int r = 0; r < 4; ++r) { v[r] = st[h * D41_ATTN_D + b * 128 + lane * 4 + r]; sv[r] = __half2float(su[r]); }
        d41x_prep4(v, sv, lane);
        __half o[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
        *reinterpret_cast<uint2*>(a.rot_xh + (long long)row * a.rot_ld + (long long)hg * D41_ATTN_D + b * 128 + lane * 4) = *reinterpret_cast<const uint2*>(o);
    }
}

__device__ __forceinline__ void d41a_epilogue(const unsigned char* sm, const d41_attn_args& a, int row, int g, int pos, const d41a_acc& T) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, r0 = lane >> 2, cq = lane & 3;
    if (a.rot_xh != nullptr) __syncthreads();          // PD1: every warp is done with the tiles before D41A_SM_T is staged into
    const bool rope = (a.epi & D41_ATTN_EPI_ROPE) && warp >= 14;
    const bool aq = (a.epi & D41_ATTN_EPI_AQ) != 0;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int hg = g * D41_ATTN_HG + r0 + 8 * hh;
        const bool hv = hg < a.heads;                  // uniform over the 4 lanes of a head row
        const float M = hh ? T.m1 : T.m0;
        const float Ls = hh ? T.l1 : T.l0;
        const float sink = reinterpret_cast<const volatile float*>(sm + D41A_SM_SK)[r0 + 8 * hh];
        const float Lf = __fadd_rn(Ls, expf(sink - M));
        float x[4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            x[nt][0] = __bfloat162float(__float2bfloat16_rn(d41a_div(T.o[nt][2 * hh], Lf)));
            x[nt][1] = __bfloat162float(__float2bfloat16_rn(d41a_div(T.o[nt][2 * hh + 1], Lf)));
        }
        if (rope) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float f = a.rope_freq[16 * (warp - 14) + 4 * nt + cq];
                float sn, cs;
                d41_rope_cs(pos, f, cs, sn);
                float y0 = x[nt][0], y1 = x[nt][1];
                d41_rope_inv(y0, y1, cs, sn);   // torch's conj complex multiply (two FMAs; SL RoPE unification)
                x[nt][0] = __bfloat162float(__float2bfloat16_rn(y0));
                x[nt][1] = __bfloat162float(__float2bfloat16_rn(y1));
            }
        }
        __nv_bfloat16* op = a.out + (long long)row * a.out_stride + (long long)hg * D41_ATTN_D + warp * 32 + 2 * cq;
        float sc = 1.f, isc = 1.f;
        if (aq) {                                   // act_quant group = this warp's 32 dims
            float am = 0.f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) am = fmaxf(am, fmaxf(fabsf(x[nt][0]), fabsf(x[nt][1])));
            am = fmaxf(am, __shfl_xor_sync(D41A_FULL, am, 1));
            am = fmaxf(am, __shfl_xor_sync(D41A_FULL, am, 2));
            const int e = d41a_aq_exp(am);
            sc = __uint_as_float((unsigned)(e + 127) << 23);
            isc = __uint_as_float((unsigned)(127 - e) << 23);   // 1 / sc, exact (a power of two in range)
            if (hv && cq == 0 && a.out_s != nullptr)
                a.out_s[(long long)row * a.outs_stride + (long long)hg * (D41_ATTN_D / 32) + warp] = (unsigned char)(e + 127);
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float y0 = x[nt][0], y1 = x[nt][1];
            if (aq) {
                const __nv_fp8_storage_t q0 = __nv_cvt_float_to_fp8(__fmul_rn(y0, isc), __NV_SATFINITE, __NV_E4M3);   // = y0 / sc exactly
                const __nv_fp8_storage_t q1 = __nv_cvt_float_to_fp8(__fmul_rn(y1, isc), __NV_SATFINITE, __NV_E4M3);
                y0 = __fmul_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(q0, __NV_E4M3))), sc);
                y1 = __fmul_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(q1, __NV_E4M3))), sc);
                if (hv && a.out_q != nullptr)
                    *reinterpret_cast<unsigned short*>(a.out_q + (long long)row * a.outq_stride + (long long)hg * D41_ATTN_D + warp * 32 + 8 * nt + 2 * cq) =
                        (unsigned short)(q0 | (q1 << 8));
            }
            if (hv) *reinterpret_cast<unsigned*>(op + 8 * nt) = d41a_pack_bf16x2(y0, y1);
            if (a.rot_xh != nullptr) {                  // PD1: stage the stored bf16 values for wo_a's input rotation
                float* st = reinterpret_cast<float*>(const_cast<unsigned char*>(sm) + D41A_SM_T) + (r0 + 8 * hh) * D41_ATTN_D + warp * 32 + 8 * nt + 2 * cq;
                st[0] = __bfloat162float(__float2bfloat16_rn(y0));
                st[1] = __bfloat162float(__float2bfloat16_rn(y1));
            }
        }
    }
    if (a.rot_xh != nullptr) d41a_rot_out(sm, a, row, g);
}
