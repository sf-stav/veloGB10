// =============================================================================
// kernels/gpu_dsv41/66_attn_2cta.cuh — band 60 (leg P7ATTN): 256-thread (8-warp), 8-head,
// 2-CTA/SM decode / verify sparse attention (`d41_attn_decode_2cta`).
//
// Preserves 100% bit-for-bit identity with `d41_attn_decode` (512 threads, 16 heads, 1 CTA/SM):
//   * Each CTA owns one 8-head half (`h_sub` in {0, 1}, `h_base = g * 16 + h_sub * 8`) of the 16-head
//     group `g` (active sub-groups = ceil(heads / 8), adjacent in blockIdx.x so all sub-groups of a
//     (row, split) co-issue and share hot L2 KV cache lines).
//   * Within every 64-key online-softmax block, T [64][256] bf16 (32 KB) tiles the two 256-dim halves:
//     - Half 0 (dims 0..255) -> K * Q^T for quarters dq = 0, 1 -> s01 = q0 + q1 in registers;
//     - Half 1 (dims 256..511) -> K * Q^T for quarters dq = 2, 3 -> s = (((q0 + q1) + q2) + q3) * scale;
//     - Online softmax over the 64 keys (same 32-lane butterfly max/sum, mn, corr, bf16(p), (m, l));
//     - P.V Half 1 immediately while Half 1 is resident in T (V^T * P^T, k ascending);
//     - Restore Half 0 to T from `raw0` registers (zero global reload), prefetch next block Half 0,
//       and compute P.V Half 0 (V^T * P^T, k ascending).
//   * Split partials in `args.part` use the same D41_ATTN_PART_FLOATS stride per (row, 16-head group,
//     split): O [16][512] contiguous per head + float2 ml[16] at +8192, compatible with both in-CTA
//     merge (two 16-bit arrival counters packed in `cnt[row * groups + g]`) and `d41_attn_merge`.
//   * Dynamic SMEM = 46,736 B (<= 48 KB default cap, zero opt-in required), <= 128 regs/thread ->
//     2 CTAs/SM (96 concurrent CTAs across 48 GB10 SMs).
// =============================================================================
#pragma once

#define D41_ATTN_2CTA_THREADS 256
#define D41_ATTN_2CTA_HG 8

#define D41A2_SM_T 0              // [64][256] bf16 = 32,768 B (reused at split end as [8][516] f32 = 16,512 B)
#define D41A2_SM_Q 32768          // [8][512] bf16  =  8,192 B
#define D41A2_SM_SP 40960         // [2][8][68] f32 =  4,352 B
#define D41A2_SP_STRIDE 68
#define D41A2_SM_P 45312          // [8][64] bf16   =  1,024 B
#define D41A2_SM_HS 46336         // [3][8] f32     =     96 B
#define D41A2_SM_KV 46432         // [64] i32       =    256 B
#define D41A2_SM_SK 46688         // [8] f32        =     32 B
#define D41A2_SM_FL 46720         // [4] i32        =     16 B
#define D41_ATTN_2CTA_SMEM 46736
#define D41A2_O_STRIDE 516        // 512 + 4 floats: 16B aligned and 32-bank conflict-free on 2*cq writes

__device__ __forceinline__ unsigned d41a2_swz_t(int row, int chunk32) {
    return (unsigned)(row * 512 + ((chunk32 ^ (row & 7)) << 4));
}
__device__ __forceinline__ unsigned d41a2_swz_p(int h, int ch8) {
    return (unsigned)(D41A2_SM_P + h * 128 + ((ch8 ^ h) << 4));
}
__device__ __forceinline__ void d41a2_ldsm_x2(unsigned (&r)[2], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}

struct d41a2_raw0 {
    uint4 v[4];
    uint4 sc;                   // all scale bytes of both halves (FP8: 16 E8M0 bytes; FP4: sc.x = half0, sc.y = half1)
    const unsigned char* dq;    // resolved key base for Half 1's v load
    int kind;                   // 0: masked, 1: FP8, 2: FP4
};

// Resolve key kk = t / 4 (part p = t % 4) of a 64-key block and fetch Half 0 (dims 0..255) + scales.
__device__ __forceinline__ void d41a2_fetch0(const d41a_row& r, int kind, int base, d41a2_raw0& raw, int* err) {
    const int t = threadIdx.x, kk = t >> 2, p = t & 3;
    const d41_attn_lane& L = *r.L;
    const unsigned char* dq = nullptr;
    const unsigned char* ds = nullptr;
    bool fp4 = false;
    if (kind == 0) {
        const long long w = (long long)r.wend - (D41_ATTN_WIN - 1) + base + kk;
        if (w >= r.wlo) {
            if (w >= L.lin_base) {
                const long long lr = w - L.lin_base;
                dq = L.lin_q + lr * D41_ATTN_D;
                ds = L.lin_s + lr * (D41_ATTN_D / 32);
            } else {
                const long long slot = (long long)((unsigned)w % (unsigned)L.slots);
                dq = L.swa_q + slot * D41_ATTN_D;
                ds = L.swa_s + slot * (D41_ATTN_D / 32);
            }
        }
    } else if (kind == 1) {
        const int e = base + kk;
        if (e < r.nsel) {
            const int idx = __ldg(r.sel + e);
            if (idx >= 0) {
                if (idx < L.kv_rows) { dq = L.kv + (long long)idx * 288; fp4 = true; }
                else if (err != nullptr && p == 0) atomicCAS(err, 0, D41_ATTN_ERR_SEL_RANGE);
            }
        }
    } else {
        const int e = base + kk;
        if (e < r.n_ext) { dq = L.ext_q + (long long)e * D41_ATTN_D; ds = L.ext_s + (long long)e * (D41_ATTN_D / 32); }
    }
    raw.dq = dq;
    raw.kind = dq == nullptr ? 0 : (fp4 ? 2 : 1);
    raw.sc = make_uint4(0u, 0u, 0u, 0u);
    if (raw.kind == 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) raw.v[i] = __ldg(reinterpret_cast<const uint4*>(dq) + (p + 4 * i));
        raw.sc = __ldg(reinterpret_cast<const uint4*>(ds));
    } else if (raw.kind == 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) raw.v[i] = __ldg(reinterpret_cast<const uint4*>(dq) + (p + 4 * i));
        const unsigned s00 = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 2 * p));
        const unsigned s01 = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 2 * (p + 4)));
        const unsigned s10 = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 16 + 2 * p));
        const unsigned s11 = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 16 + 2 * (p + 4)));
        raw.sc.x = s00 | (s01 << 16);
        raw.sc.y = s10 | (s11 << 16);
    }
}

// Fetch Half 1 (dims 256..511) values using `raw0.dq` resolved by d41a2_fetch0.
__device__ __forceinline__ void d41a2_fetch1(const d41a2_raw0& raw0, uint4 (&v1)[4]) {
    const int p = threadIdx.x & 3;
    if (raw0.kind == 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v1[i] = __ldg(reinterpret_cast<const uint4*>(raw0.dq + 256) + (p + 4 * i));
    } else if (raw0.kind == 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) v1[i] = __ldg(reinterpret_cast<const uint4*>(raw0.dq + 128) + (p + 4 * i));
    }
}

// Dequantise one 256-dim half into T [64][256] bf16 (exact match to d41a_store).
template <bool SET_KV>
__device__ __forceinline__ void d41a2_store_half(unsigned char* sm, int kind, const uint4 (&v)[4], unsigned sc0, unsigned sc1) {
    const int t = threadIdx.x, kk = t >> 2, p = t & 3;
    unsigned char* tp = sm + D41A2_SM_T;
    if (SET_KV && p == 0) reinterpret_cast<int*>(sm + D41A2_SM_KV)[kk] = kind != 0;
    if (kind == 0) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(tp + d41a2_swz_t(kk, p + 4 * i)) = z;
    } else if (kind == 1) {
        const unsigned scw[2] = {sc0, sc1};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = p + 4 * i;
            const unsigned eb = (scw[i >> 1] >> (8 * ((p >> 1) + 2 * (i & 1)))) & 0xFFu;
            const float s = __int_as_float((int)(eb << 23));
            const unsigned w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
            unsigned o[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[2 * j] = d41a_f16x2_scale_bf16x2(d41a_e4m3x2_f16x2((unsigned short)(w[j] & 0xFFFFu)), s);
                o[2 * j + 1] = d41a_f16x2_scale_bf16x2(d41a_e4m3x2_f16x2((unsigned short)(w[j] >> 16)), s);
            }
            *reinterpret_cast<uint4*>(tp + d41a2_swz_t(kk, 2 * c)) = make_uint4(o[0], o[1], o[2], o[3]);
            *reinterpret_cast<uint4*>(tp + d41a2_swz_t(kk, 2 * c + 1)) = make_uint4(o[4], o[5], o[6], o[7]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = p + 4 * i;
            const unsigned short scp = (unsigned short)(i == 0 ? (sc0 & 0xFFFFu) : (sc0 >> 16));
            const unsigned s2 = d41a_e4m3x2_f16x2(scp);
            const __half2 slo = __low2half2(*reinterpret_cast<const __half2*>(&s2));
            const __half2 shi = __high2half2(*reinterpret_cast<const __half2*>(&s2));
            const unsigned w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned h[4];
                d41a_e2m1x8_f16x2(w[j], h);
                const __half2 sj = (j < 2) ? slo : shi;
                *reinterpret_cast<uint4*>(tp + d41a2_swz_t(kk, 4 * c + j)) =
                    make_uint4(d41a_f16x2_mul_bf16x2(h[0], sj), d41a_f16x2_mul_bf16x2(h[1], sj),
                               d41a_f16x2_mul_bf16x2(h[2], sj), d41a_f16x2_mul_bf16x2(h[3], sj));
            }
        }
    }
}

template <bool QROPE>
__device__ __forceinline__ void d41a2_load_q(unsigned char* sm, const __nv_bfloat16* q, long long q_stride,
                                             int row, int h_base, int heads, int pos, const float* freq) {
    const int t = threadIdx.x;
    unsigned char* qp = sm + D41A2_SM_Q;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = t + D41_ATTN_2CTA_THREADS * i;
        const int h = c >> 6, ch = c & 63, hg = h_base + h;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (hg < heads) {
            v = __ldg(reinterpret_cast<const uint4*>(q + (long long)row * q_stride + (long long)hg * D41_ATTN_D) + ch);
            if (QROPE && ch >= (D41_ATTN_D - D41_ATTN_ROPE) / 8) {
                unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float re = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] & 0xFFFFu)));
                    float im = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] >> 16)));
                    float cs, sn;
                    d41_rope_cs(pos, freq[4 * (ch - (D41_ATTN_D - D41_ATTN_ROPE) / 8) + j], cs, sn);
                    d41_rope_fwd(re, im, cs, sn);
                    w[j] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(re))
                         | ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(im)) << 16);
                }
                v = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        *reinterpret_cast<uint4*>(qp + d41a_swz(h, ch)) = v;
    }
}

__device__ __forceinline__ void d41a2_sinks(unsigned char* sm, const d41_attn_args& a, int h_base) {
    const int t = threadIdx.x, hg = h_base + t;
    if (t < D41_ATTN_2CTA_HG) reinterpret_cast<float*>(sm + D41A2_SM_SK)[t] = hg < a.heads ? __ldg(a.sink + hg) : 0.f;
}

// Per-thread state of an 8-head split: o[2*hf + mt][0..3] holds (d0,h0), (d0,h1), (d0+8,h0), (d0+8,h1)
// where h0 = 2 * (lane & 3), h1 = h0 + 1, d0 = 256 * hf + 32 * warp + 16 * mt + (lane >> 2).
// Per-head (m, l) live in D41A2_SM_HS[8..23] published by owner warp `h = warp`.
struct d41a2_acc {
    float o[4][4];
    float om, ol;
};

__device__ __forceinline__ void d41a2_acc_reset(d41a2_acc& A) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) { A.o[nt][0] = 0.f; A.o[nt][1] = 0.f; A.o[nt][2] = 0.f; A.o[nt][3] = 0.f; }
    A.om = -1e30f; A.ol = 0.f;
}

// K [64, 256] * Q^T [256, 8] -> SP [2][8][68] for dim half `hf` (quarters 2*hf and 2*hf + 1).
__device__ __forceinline__ void d41a2_qk_half(unsigned char* sm, unsigned smb, int warp, int lane, int hf) {
    const int dq_loc = warp & 1, kq = warp >> 1;
    const int dq = 2 * hf + dq_loc;
    const int r0 = lane >> 2, cq = lane & 3;
    const int krow = 16 * kq + (lane & 7) + ((lane >> 3) & 1) * 8, kch = lane >> 4;
    const int qhead = lane & 7, qch = lane >> 3;
    float sp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) {
        unsigned k0[4], k1[4], qb[4];
        d41a_ldsm_x4(k0, smb + D41A2_SM_T + d41a2_swz_t(krow, 16 * dq_loc + 4 * k2 + kch));
        d41a_ldsm_x4(k1, smb + D41A2_SM_T + d41a2_swz_t(krow, 16 * dq_loc + 4 * k2 + 2 + kch));
        d41a_ldsm_x4(qb, smb + D41A2_SM_Q + d41a_swz(qhead, 16 * dq + 4 * k2 + qch));
        d41a_mma(sp, k0, qb[0], qb[1]);
        d41a_mma(sp, k1, qb[2], qb[3]);
    }
    float* spp = reinterpret_cast<float*>(sm + D41A2_SM_SP) + dq_loc * (D41_ATTN_2CTA_HG * D41A2_SP_STRIDE);
    spp[(2 * cq) * D41A2_SP_STRIDE + 16 * kq + r0] = sp[0];
    spp[(2 * cq + 1) * D41A2_SP_STRIDE + 16 * kq + r0] = sp[1];
    spp[(2 * cq) * D41A2_SP_STRIDE + 16 * kq + r0 + 8] = sp[2];
    spp[(2 * cq + 1) * D41A2_SP_STRIDE + 16 * kq + r0 + 8] = sp[3];
}

// V^T [256, 64] * P^T [64, 8] -> A.o[2*hf .. 2*hf+1] for dim half `hf`.
__device__ __forceinline__ void d41a2_pv_half(unsigned smb, int warp, int lane, int hf, d41a2_acc& A) {
    const int v_koff = (lane >> 4) * 8 + (lane & 7);
    const int d_off = (lane >> 3) & 1;
    const int ph = lane & 7;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        unsigned pb[2];
        d41a2_ldsm_x2(pb, smb + d41a2_swz_p(ph, 2 * kk + d_off));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            unsigned va[4];
            d41a_ldsm_x4_t(va, smb + D41A2_SM_T + d41a2_swz_t(16 * kk + v_koff, 4 * warp + 2 * mt + d_off));
            d41a_mma(A.o[2 * hf + mt], va, pb[0], pb[1]);
        }
    }
}

__device__ __forceinline__ void d41a2_block(unsigned char* sm, const d41a_row& r, d41a2_raw0& raw0, bool more, int nkind, int nbase,
                                            float scale, d41a2_acc& A, int* err) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, cq = lane & 3;
    const unsigned smb = (unsigned)__cvta_generic_to_shared(sm);

    // Phase 1: store Half 0 into T, prefetch Half 1, compute QK Half 0 (dq = 0, 1)
    d41a2_store_half<true>(sm, raw0.kind, raw0.v, raw0.sc.x, raw0.sc.y);
    __syncthreads();
    uint4 v1[4];
    d41a2_fetch1(raw0, v1);
    d41a2_qk_half(sm, smb, warp, lane, 0);
    __syncthreads();

    // Phase 2: accumulate s01 = q0 + q1 in regs while storing Half 1 into T, then compute QK Half 1 (dq = 2, 3)
    const int k0 = 2 * lane;
    const float* sp0 = reinterpret_cast<const float*>(sm + D41A2_SM_SP) + warp * D41A2_SP_STRIDE + k0;
    const float s01_0 = __fadd_rn(sp0[0], sp0[D41_ATTN_2CTA_HG * D41A2_SP_STRIDE]);
    const float s01_1 = __fadd_rn(sp0[1], sp0[D41_ATTN_2CTA_HG * D41A2_SP_STRIDE + 1]);
    d41a2_store_half<false>(sm, raw0.kind, v1, raw0.kind == 1 ? raw0.sc.z : raw0.sc.y, raw0.sc.w);
    __syncthreads();
    d41a2_qk_half(sm, smb, warp, lane, 1);
    __syncthreads();

    // Phase 3: online softmax over the 64 keys, rescale O, compute PV Half 1 while Half 1 is still in T
    volatile float* hs = reinterpret_cast<volatile float*>(sm + D41A2_SM_HS);
    {
        const volatile float* sv = reinterpret_cast<const volatile float*>(sm + D41A2_SM_SP) + warp * D41A2_SP_STRIDE + k0;
        const float a0 = __fadd_rn(__fadd_rn(s01_0, sv[0]), sv[D41_ATTN_2CTA_HG * D41A2_SP_STRIDE]);
        const float a1 = __fadd_rn(__fadd_rn(s01_1, sv[1]), sv[D41_ATTN_2CTA_HG * D41A2_SP_STRIDE + 1]);
        const volatile int* kvf = reinterpret_cast<const volatile int*>(sm + D41A2_SM_KV);
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
        *reinterpret_cast<volatile unsigned*>(sm + d41a2_swz_p(warp, lane >> 2) + 4 * (lane & 3)) = d41a_pack_bf16x2(p0, p1);
        if (lane == 0) { hs[warp] = corr; hs[8 + warp] = mn; hs[16 + warp] = A.ol; }
    }
    __syncthreads();
    {
        const float c0 = hs[2 * cq], c1 = hs[2 * cq + 1];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            A.o[nt][0] = __fmul_rn(A.o[nt][0], c0); A.o[nt][1] = __fmul_rn(A.o[nt][1], c1);
            A.o[nt][2] = __fmul_rn(A.o[nt][2], c0); A.o[nt][3] = __fmul_rn(A.o[nt][3], c1);
        }
    }
    d41a2_pv_half(smb, warp, lane, 1, A);
    __syncthreads();

    // Phase 4: restore Half 0 into T from raw0 registers, prefetch next block Half 0, compute PV Half 0
    d41a2_store_half<false>(sm, raw0.kind, raw0.v, raw0.sc.x, raw0.sc.y);
    __syncthreads();
    if (more) d41a2_fetch0(r, nkind, nbase, raw0, err);
    d41a2_pv_half(smb, warp, lane, 0, A);
    __syncthreads();
}

// Stage X.o [8 heads][512 dims] into SMEM T (stride D41A2_O_STRIDE = 516 floats: conflict-free, 16B aligned).
__device__ __forceinline__ void d41a2_stage_o(unsigned char* sm, const d41a2_acc& X) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r0 = lane >> 2, cq = lane & 3;
    float* st = reinterpret_cast<float*>(sm + D41A2_SM_T);
#pragma unroll
    for (int idx = 0; idx < 4; ++idx) {
        const int hf = idx >> 1, mt = idx & 1;
        const int d = 256 * hf + 32 * warp + 16 * mt + r0;
        st[(2 * cq) * D41A2_O_STRIDE + d] = X.o[idx][0];
        st[(2 * cq + 1) * D41A2_O_STRIDE + d] = X.o[idx][1];
        st[(2 * cq) * D41A2_O_STRIDE + d + 8] = X.o[idx][2];
        st[(2 * cq + 1) * D41A2_O_STRIDE + d + 8] = X.o[idx][3];
    }
    __syncthreads();
}

// Write or merge one split partial for 8 heads into `pp` (contiguous [head][512] + float2 ml at +8192).
// mode 0: raw split write (non-prefix CTA); mode 1: prefix split 0 (canonical merge(reset, A)); mode 2: prefix split > 0 (merge(slot0, A)).
__device__ __forceinline__ void d41a2_commit_slot(unsigned char* sm, float* pp, int h_sub, int mode, const d41a2_acc& X) {
    d41a2_stage_o(sm, X);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hl = h_sub * 8 + warp;
    const volatile float* hs = reinterpret_cast<const volatile float*>(sm + D41A2_SM_HS);
    const float sm_m = hs[8 + warp], sm_l = hs[16 + warp];
    const float* st = reinterpret_cast<const float*>(sm + D41A2_SM_T) + warp * D41A2_O_STRIDE;
    float* dst = pp + hl * D41_ATTN_D;
    float2* pm = reinterpret_cast<float2*>(pp + D41_ATTN_HG * D41_ATTN_D) + hl;

    if (mode == 0) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int d0 = 128 * b + 4 * lane;
            *reinterpret_cast<float4*>(dst + d0) = *reinterpret_cast<const float4*>(st + d0);
        }
        if (lane == 0) *pm = make_float2(sm_m, sm_l);
    } else {
        float M = -1e30f, Ls = 0.f;
        if (mode == 2) {
            const float2 ml0 = *pm;
            M = ml0.x; Ls = ml0.y;
        }
        const float n = fmaxf(M, sm_m);
        const float e = expf(M - n), f = expf(sm_m - n);
        const float Ln = __fmaf_rn(Ls, e, __fmul_rn(sm_l, f));
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int d0 = 128 * b + 4 * lane;
            const float4 sa = *reinterpret_cast<const float4*>(st + d0);
            const float4 s0 = (mode == 2) ? *reinterpret_cast<const float4*>(dst + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(dst + d0) = make_float4(
                __fmaf_rn(s0.x, e, __fmul_rn(sa.x, f)),
                __fmaf_rn(s0.y, e, __fmul_rn(sa.y, f)),
                __fmaf_rn(s0.z, e, __fmul_rn(sa.z, f)),
                __fmaf_rn(s0.w, e, __fmul_rn(sa.w, f)));
        }
        if (lane == 0) *pm = make_float2(n, Ln);
    }
}

__device__ __forceinline__ void d41a2_wpf(const d41_attn_args& a) {
    const int sub_groups = (a.heads + 7) >> 3;
    const int slots = a.nsplit_max & 0xFFFF;
    const int G = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slots;
    int bid = blockIdx.x;
    const int sg = bid % sub_groups; bid /= sub_groups;
    const int cta = bid % G;
    const int row = bid / G;
    if (cta != 0 || (sg & 1) != 0 || row >= a.rows) return;
    const int pos = __ldg(a.pos + row);
    const int wend = a.win_end ? __ldg(a.win_end + row) : pos;
    int ln = a.lane ? __ldg(a.lane + row) : 0;
    ln = ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln);
    const d41_attn_lane& L = a.lanes[ln];
    if (L.swa_q == nullptr || L.slots <= 0) return;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = threadIdx.x + D41_ATTN_2CTA_THREADS * i;
        const int k = idx >> 2, part = idx & 3;
        const long long p = (long long)wend - (D41_ATTN_WIN - 1) + k;
        if (p >= 0 && p < L.lin_base) {
            const long long slot = (long long)((unsigned)p % (unsigned)L.slots);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_q + slot * D41_ATTN_D + part * 128));
            if (part == 0 && L.swa_s != nullptr) asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_s + slot * 16));
        }
    }
}

extern "C" __global__ void __launch_bounds__(D41_ATTN_2CTA_THREADS, 2)
d41_attn_decode_2cta(const __grid_constant__ d41_attn_args a) {
    if (a.xflags & D41_ATTN_X_WPF) d41a2_wpf(a);
    D41_PDL_ENTRY();

    extern __shared__ __align__(16) unsigned char d41a2_sm[];
    unsigned char* sm = d41a2_sm;
    const int sub_groups = (a.heads + 7) >> 3;
    const int slots = a.nsplit_max & 0xFFFF;
    const int G = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slots;
    int bid = blockIdx.x;
    const int sg = bid % sub_groups; bid /= sub_groups;
    const int cta = bid % G;
    const int row = bid / G;
    const int h_base = sg * D41_ATTN_2CTA_HG;
    if (row >= a.rows || h_base >= a.heads) return;

    d41a_row r;
    r.row = row;
    r.pos = __ldg(a.pos + row);
    r.wend = a.win_end ? __ldg(a.win_end + row) : r.pos;
    r.wlo = a.win_lo > 0 ? a.win_lo : 0;
    const int ln = a.lane ? __ldg(a.lane + row) : 0;
    r.L = &a.lanes[ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln)];
    int nsel = 0;
    if (a.sel != nullptr && r.L->kv != nullptr) {
        nsel = a.n_sel ? __ldg(a.n_sel + row) : a.sel_cap;
        nsel = max(0, min(nsel, min(a.sel_cap, D41_ATTN_SEL_MAX)));
    }
    r.nsel = nsel;
    r.sel = a.sel ? a.sel + (long long)row * a.sel_stride : nullptr;
    r.n_ext = r.L->ext_q != nullptr ? max(0, min(r.L->n_ext, D41_ATTN_EXT_MAX)) : 0;

    const int nw = r.wend >= 64 ? 2 : 1;
    const int nc = (nsel + 63) >> 6;
    const int nb = nw + nc + (r.n_ext > 0 ? 1 : 0);
    const int nS = (nb + D41_ATTN_SPLIT_BLOCKS - 1) / D41_ATTN_SPLIT_BLOCKS;
    if (nS > slots) {
        if (cta == 0 && sg == 0 && threadIdx.x == 0 && a.err != nullptr) atomicCAS(a.err, 0, D41_ATTN_ERR_ARGS);
        return;
    }
    const int Ge = min(G, nS);
    if (cta >= Ge) return;

    const int s_lo = cta * nS / Ge, s_hi = (cta + 1) * nS / Ge;
    const int b0 = s_lo * D41_ATTN_SPLIT_BLOCKS;
    const int b1 = min(nb, s_hi * D41_ATTN_SPLIT_BLOCKS);
    int kind0 = 0, base0 = 0;
    d41a_blk(nw, nc, b0, kind0, base0);
    d41a2_raw0 raw0;
    d41a2_fetch0(r, kind0, base0, raw0, a.err);
    if (a.xflags & D41_ATTN_X_QROPE) d41a2_load_q<true>(sm, a.q, a.q_stride, row, h_base, a.heads, r.pos, a.rope_freq);
    else d41a2_load_q<false>(sm, a.q, a.q_stride, row, h_base, a.heads, r.pos, a.rope_freq);
    d41a2_sinks(sm, a, h_base);

    d41a2_acc A;
    d41a2_acc_reset(A);
    for (int idx = b0; idx < b1; ++idx) {
        const bool more = idx + 1 < b1;
        int kind2 = 0, base2 = 0;
        if (more) d41a_blk(nw, nc, idx + 1, kind2, base2);
        d41a2_block(sm, r, raw0, more, kind2, base2, a.scale, A, a.err);
        if ((idx + 1) % D41_ATTN_SPLIT_BLOCKS == 0 || !more) {
            const int split = idx / D41_ATTN_SPLIT_BLOCKS;
            const int sub_g = (a.heads + 7) >> 3;
            const int sgx = blockIdx.x % sub_g, bx = blockIdx.x / sub_g;
            const int slx = a.nsplit_max & 0xFFFF, Gx = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slx;
            const int ctax = bx % Gx, rx = bx / Gx, gx = sgx >> 1, hsx = sgx & 1, hbx = sgx * D41_ATTN_2CTA_HG;
            const bool defx = (a.xflags & D41_ATTN_X_DEFER) != 0;
            float* pbx = a.part + (long long)(rx * a.groups + gx) * slx * D41_ATTN_PART_FLOATS;

            if (ctax == 0 && Ge == 1 && !defx && !more) {
                // Single-CTA row (Ge == 1, !defer): merge with slot 0 (or reset if split == 0) and run epilogue directly.
                d41a2_stage_o(sm, A);
                const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
                const int hg = hbx + warp, hl = hsx * 8 + warp;
                if (hg < a.heads) {
                    const volatile float* hs = reinterpret_cast<const volatile float*>(sm + D41A2_SM_HS);
                    const float sm_m = hs[8 + warp], sm_l = hs[16 + warp];
                    float M = -1e30f, Ls = 0.f;
                    if (split > 0) {
                        const float2 ml0 = *(reinterpret_cast<const float2*>(pbx + D41_ATTN_HG * D41_ATTN_D) + hl);
                        M = ml0.x; Ls = ml0.y;
                    }
                    const float n = fmaxf(M, sm_m);
                    const float e = expf(M - n), f = expf(sm_m - n);
                    const float Ln = __fmaf_rn(Ls, e, __fmul_rn(sm_l, f));
                    const float sink = reinterpret_cast<const float*>(sm + D41A2_SM_SK)[warp];
                    const float Lf = __fadd_rn(Ln, expf(sink - n));
                    const float* st = reinterpret_cast<const float*>(sm + D41A2_SM_T) + warp * D41A2_O_STRIDE;
                    const float* dst0 = pbx + hl * D41_ATTN_D;
                    const int px = __ldg(a.pos + rx);
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int d0 = 128 * b + 4 * lane;
                        const float4 sa = *reinterpret_cast<const float4*>(st + d0);
                        const float4 s0 = (split > 0) ? *reinterpret_cast<const float4*>(dst0 + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
                        const float4 o4 = make_float4(
                            __fmaf_rn(s0.x, e, __fmul_rn(sa.x, f)),
                            __fmaf_rn(s0.y, e, __fmul_rn(sa.y, f)),
                            __fmaf_rn(s0.z, e, __fmul_rn(sa.z, f)),
                            __fmaf_rn(s0.w, e, __fmul_rn(sa.w, f)));
                        float x[4] = {
                            __bfloat162float(__float2bfloat16_rn(d41a_div(o4.x, Lf))),
                            __bfloat162float(__float2bfloat16_rn(d41a_div(o4.y, Lf))),
                            __bfloat162float(__float2bfloat16_rn(d41a_div(o4.z, Lf))),
                            __bfloat162float(__float2bfloat16_rn(d41a_div(o4.w, Lf))),
                        };
                        d41a_epi4(a, rx, hg, px, d0, lane, x);
                    }
                }
                return;
            }
            if (ctax > 0) d41a2_commit_slot(sm, pbx + (long long)split * D41_ATTN_PART_FLOATS, hsx, 0, A);
            else          d41a2_commit_slot(sm, pbx, hsx, split == 0 ? 1 : 2, A);
            if (more) __syncthreads();
            d41a2_acc_reset(A);
        }
    }

    if (a.xflags & D41_ATTN_X_DEFER) return;

    const int sgx = blockIdx.x % ((a.heads + 7) >> 3), bx = blockIdx.x / ((a.heads + 7) >> 3);
    const int slx = a.nsplit_max & 0xFFFF, Gx = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slx;
    const int rx = bx / Gx, gx = sgx >> 1, hsx = sgx & 1, hbx = sgx * D41_ATTN_2CTA_HG;
    float* pbase = a.part + (long long)(rx * a.groups + gx) * slx * D41_ATTN_PART_FLOATS;
    unsigned int* cnt = a.cnt + (rx * a.groups + gx);
    __threadfence();
    __syncthreads();
    volatile int* fl = reinterpret_cast<volatile int*>(sm + D41A2_SM_FL);
    if (threadIdx.x == 0) {
        const unsigned add = hsx ? 0x00010000u : 1u;
        const unsigned prev_w = atomicAdd(cnt, add);
        const unsigned prev = hsx ? (prev_w >> 16) : (prev_w & 0xFFFFu);
        fl[0] = prev == (unsigned)(Ge - 1);
    }
    __syncthreads();
    if (!fl[0]) return;
    __threadfence();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hl = hsx * 8 + warp, hg = hbx + warp;
    if (hg < a.heads) {
        float4 o[4];
#pragma unroll
        for (int b = 0; b < 4; ++b)
            o[b] = __ldcg(reinterpret_cast<const float4*>(pbase + hl * D41_ATTN_D + 128 * b + 4 * lane));
        const float2 ml0 = __ldcg(reinterpret_cast<const float2*>(pbase + D41_ATTN_HG * D41_ATTN_D) + hl);
        float M = ml0.x, Ls = ml0.y;
        for (int s = nS / Ge; s < nS; ++s) {
            const float* pp = pbase + (long long)s * D41_ATTN_PART_FLOATS;
            const float2 mls = __ldcg(reinterpret_cast<const float2*>(pp + D41_ATTN_HG * D41_ATTN_D) + hl);
            const float n = fmaxf(M, mls.x);
            const float e = expf(M - n), f = expf(mls.x - n);
            Ls = __fmaf_rn(Ls, e, __fmul_rn(mls.y, f));
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float4 so = __ldcg(reinterpret_cast<const float4*>(pp + hl * D41_ATTN_D + 128 * b + 4 * lane));
                o[b].x = __fmaf_rn(o[b].x, e, __fmul_rn(so.x, f));
                o[b].y = __fmaf_rn(o[b].y, e, __fmul_rn(so.y, f));
                o[b].z = __fmaf_rn(o[b].z, e, __fmul_rn(so.z, f));
                o[b].w = __fmaf_rn(o[b].w, e, __fmul_rn(so.w, f));
            }
            M = n;
        }
        const float sink = reinterpret_cast<const float*>(sm + D41A2_SM_SK)[warp];
        const float Lf = __fadd_rn(Ls, expf(sink - M));
        const int px = __ldg(a.pos + rx);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int d0 = 128 * b + 4 * lane;
            float x[4] = {
                __bfloat162float(__float2bfloat16_rn(d41a_div(o[b].x, Lf))),
                __bfloat162float(__float2bfloat16_rn(d41a_div(o[b].y, Lf))),
                __bfloat162float(__float2bfloat16_rn(d41a_div(o[b].z, Lf))),
                __bfloat162float(__float2bfloat16_rn(d41a_div(o[b].w, Lf))),
            };
            d41a_epi4(a, rx, hg, px, d0, lane, x);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAnd(cnt, hsx ? 0x0000FFFFu : 0xFFFF0000u);
}
