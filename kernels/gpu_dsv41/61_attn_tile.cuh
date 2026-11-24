// =============================================================================
// kernels/gpu_dsv41/61_attn_tile.cuh — band 60 (KA): shared-memory layout, swizzle, ldmatrix / mma
// wrappers, key resolution and the in-kernel FP8 / FP4 dequant of one 64-key tile.
//
// Shared memory (dynamic, 100,880 B of the 101,376 B opt-in cap; one CTA of 512 threads per SM):
//   T  [64 keys][512] bf16   65,536 B  the dequantised KV tile (K for QK^T, V for P.V: MQA, K = V)
//   Q  [16 heads][512] bf16  16,384 B  the CTA's queries
//   SP [4][16][72] f32       18,432 B  QK^T partial sums of the four dim quarters [quarter][head][key]
//                                      (row stride 72 floats: conflict-free float2 stores per half-warp).
//                                      P [16][64] bf16 lives INSIDE it: row h of P is the first 128 B of
//                                      quarter 0's row h, written by warp h after it has read that row
//                                      (no other warp reads head h's rows).
//   HS [3][16] f32              192 B  per-head softmax state published by the owner warp: corr, m, l
//   KV [64] i32                 256 B  key valid flags of the current block
//   FL [4] i32                   16 B  last-arriver flag (decode)
//   SK [16] f32                  64 B  the CTA's sinks (read at the start, used by the epilogue)
// Rows of T and Q are 1,024 B = 64 chunks of 16 B; chunk c of row r lives at chunk (c ^ (r & 7)) so the
// 8 row addresses of every ldmatrix phase hit 8 different bank groups. P rows are 288 B apart; chunk c
// of P row h lives at chunk c ^ ((h >> 2) & 1), which again spreads 8 rows over 8 bank groups.
// =============================================================================
#pragma once

#define D41A_SM_T 0
#define D41A_SM_Q 65536
#define D41A_SM_SP 81920
#define D41A_SP_STRIDE 72
#define D41A_SM_HS 100352
#define D41A_SM_KV 100544
#define D41A_SM_FL 100800
#define D41A_SM_SK 100816
#define D41_ATTN_SMEM 100880

__device__ __forceinline__ unsigned d41a_swz(int row, int chunk) {   // byte offset in a 1,024-B-row tile
    return (unsigned)(row * 1024 + ((chunk ^ (row & 7)) << 4));
}
__device__ __forceinline__ unsigned d41a_swz_p(int row, int chunk) { // byte offset of P (inside SP)
    return (unsigned)(D41A_SM_SP + row * (D41A_SP_STRIDE * 4) + ((chunk ^ ((row >> 2) & 1)) << 4));
}

__device__ __forceinline__ void d41a_ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void d41a_ldsm_x4_t(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
// D = A (16x16 bf16, row) x B (16x8 bf16, col) + D, fp32 accumulate.
__device__ __forceinline__ void d41a_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// two E4M3 codes (low byte = first) -> f16x2, exact
__device__ __forceinline__ unsigned d41a_e4m3x2_f16x2(unsigned short v) {
    unsigned r;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(v));
    return r;
}
// four E2M1 pairs of one 32-bit word (byte j = values 2j (low nibble), 2j+1) -> four f16x2, exact
__device__ __forceinline__ void d41a_e2m1x8_f16x2(unsigned w, unsigned (&h)[4]) {
    asm volatile("{ .reg .b8 b0, b1, b2, b3;\n"
                 "  mov.b32 {b0, b1, b2, b3}, %4;\n"
                 "  cvt.rn.f16x2.e2m1x2 %0, b0;\n"
                 "  cvt.rn.f16x2.e2m1x2 %1, b1;\n"
                 "  cvt.rn.f16x2.e2m1x2 %2, b2;\n"
                 "  cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
                 : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]) : "r"(w));
}
__device__ __forceinline__ unsigned d41a_pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);   // .x (low 16 bits) = lo, round to nearest even
    return *reinterpret_cast<unsigned*>(&b);
}
// f16x2 x f32 scale -> bf16x2 (the product of an E4M3 value and a power of two is exact in bf16)
__device__ __forceinline__ unsigned d41a_f16x2_scale_bf16x2(unsigned h, float s) {
    __half2 hh = *reinterpret_cast<__half2*>(&h);
    float2 f = __half22float2(hh);
    return d41a_pack_bf16x2(__fmul_rn(f.x, s), __fmul_rn(f.y, s));
}
// f16x2 x f16x2 scale (exact: E2M1 x E4M3 has <= 6 significant bits, |x| in [2^-10, 2688]) -> bf16x2 (exact)
__device__ __forceinline__ unsigned d41a_f16x2_mul_bf16x2(unsigned h, __half2 s2) {
    __half2 hh = __hmul2(*reinterpret_cast<__half2*>(&h), s2);
    float2 f = __half22float2(hh);
    return d41a_pack_bf16x2(f.x, f.y);
}

// What a CTA knows about its query row.
struct d41a_row {
    int row, pos, wend, nsel, n_ext;
    int wlo;          // window floor (args.win_lo)
    const d41_attn_lane* L;
    const int* sel;   // this row's selection (sel + row * sel_stride)
};

// The raw bytes one thread brings for one block (the register prefetch of the next block). Thread t
// handles key t / 8, part p = t % 8; the 8 threads of a key read one contiguous 128-byte stretch per load
// instruction:
//   FP8 key (window / extra): v[i] = 16-byte chunk c = p + 8i (i < 4) of the 512 E4M3 codes (dims
//                             16c..16c+15); sc = the key's 16 E8M0 bytes
//   FP4 key (compressed):     v[i] = 16-byte chunk c = p + 8i (i < 2) of the 256 E2M1 bytes (dims
//                             32c..32c+31); sc.x / sc.y = the E4M3 scale byte pairs (record bytes 256 + 2c, 257 + 2c)
struct d41a_raw {
    uint4 v[4];
    uint4 sc;
    int kind;        // 0: masked (zero row), 1: FP8, 2: FP4
};

// Resolve this thread's key of a block and issue its loads. kind 0: window (list entries base..base+63,
// base = 0 or 64), 1: compressed (sel entries base..), 2: extra keys (base = 0).
__device__ __forceinline__ void d41a_fetch(const d41a_row& r, int kind, int base, d41a_raw& raw, int* err) {
    const int t = threadIdx.x, kk = t >> 3, p = t & 7;
    const d41_attn_lane& L = *r.L;
    const unsigned char* dq = nullptr;
    const unsigned char* ds = nullptr;
    bool fp4 = false;
    if (kind == 0) {
        const long long w = (long long)r.wend - (D41_ATTN_WIN - 1) + base + kk;
        if (w >= r.wlo) {   // wlo = 0 unless a truncated replay masks the window below its segment
            if (w >= L.lin_base) {
                const long long lr = w - L.lin_base;
                dq = L.lin_q + lr * D41_ATTN_D;
                ds = L.lin_s + lr * (D41_ATTN_D / 32);
            } else {
                const long long slot = (long long)((unsigned)w % (unsigned)L.slots);   // 32-bit: no 64-bit division call
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
    raw.kind = dq == nullptr ? 0 : (fp4 ? 2 : 1);
    if (raw.kind == 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) raw.v[i] = __ldg(reinterpret_cast<const uint4*>(dq + 16 * (p + 8 * i)));
        raw.sc = __ldg(reinterpret_cast<const uint4*>(ds));
    } else if (raw.kind == 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) raw.v[i] = __ldg(reinterpret_cast<const uint4*>(dq + 16 * (p + 8 * i)));
        raw.sc.x = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 2 * p));
        raw.sc.y = __ldg(reinterpret_cast<const unsigned short*>(dq + 256 + 2 * (p + 8)));
    }
}

// Dequantise this thread's part of the fetched key into T (bf16, exact) and set the key's valid flag.
// Masked keys are zeroed (their p is 0, and 0 * NaN garbage would not be). Caller syncs before reading.
__device__ __forceinline__ void d41a_store(unsigned char* sm, const d41a_raw& raw) {
    const int t = threadIdx.x, kk = t >> 3, p = t & 7;
    unsigned char* tp = sm + D41A_SM_T;
    if (p == 0) reinterpret_cast<int*>(sm + D41A_SM_KV)[kk] = raw.kind != 0;
    if (raw.kind == 0) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(tp + d41a_swz(kk, p + 8 * i)) = z;
    } else if (raw.kind == 1) {
        // chunk c = p + 8i: dims 16c..16c+15 -> bf16 chunks 2c, 2c+1; its scale byte c / 2 = (p / 2) + 4i is
        // byte p / 2 of scale word i
        const unsigned scw[4] = {raw.sc.x, raw.sc.y, raw.sc.z, raw.sc.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = p + 8 * i;
            const unsigned eb = (scw[i] >> (8 * (p >> 1))) & 0xFFu;
            const float s = __int_as_float((int)(eb << 23));   // 2^(eb - 127); act_quant never writes 0 / 255
            const unsigned w[4] = {raw.v[i].x, raw.v[i].y, raw.v[i].z, raw.v[i].w};
            unsigned o[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[2 * j] = d41a_f16x2_scale_bf16x2(d41a_e4m3x2_f16x2((unsigned short)(w[j] & 0xFFFFu)), s);
                o[2 * j + 1] = d41a_f16x2_scale_bf16x2(d41a_e4m3x2_f16x2((unsigned short)(w[j] >> 16)), s);
            }
            *reinterpret_cast<uint4*>(tp + d41a_swz(kk, 2 * c)) = make_uint4(o[0], o[1], o[2], o[3]);
            *reinterpret_cast<uint4*>(tp + d41a_swz(kk, 2 * c + 1)) = make_uint4(o[4], o[5], o[6], o[7]);
        }
    } else {
        // chunk c = p + 8i: dims 32c..32c+31 -> bf16 chunks 4c..4c+3; word j = dims 32c + 8j..+7;
        // scale lo (dims 32c..+15) for words 0-1, hi for words 2-3
        const unsigned scp[2] = {raw.sc.x, raw.sc.y};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = p + 8 * i;
            const unsigned s2 = d41a_e4m3x2_f16x2((unsigned short)scp[i]);
            const __half2 slo = __low2half2(*reinterpret_cast<const __half2*>(&s2));
            const __half2 shi = __high2half2(*reinterpret_cast<const __half2*>(&s2));
            const unsigned w[4] = {raw.v[i].x, raw.v[i].y, raw.v[i].z, raw.v[i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned h[4];
                d41a_e2m1x8_f16x2(w[j], h);
                const __half2 sj = (j < 2) ? slo : shi;
                *reinterpret_cast<uint4*>(tp + d41a_swz(kk, 4 * c + j)) =
                    make_uint4(d41a_f16x2_mul_bf16x2(h[0], sj), d41a_f16x2_mul_bf16x2(h[1], sj),
                               d41a_f16x2_mul_bf16x2(h[2], sj), d41a_f16x2_mul_bf16x2(h[3], sj));
            }
        }
    }
}

// Load the CTA's 16 heads of q (heads >= heads are zero) into Q.
__device__ __forceinline__ void d41a_load_q(unsigned char* sm, const __nv_bfloat16* q, long long q_stride, int row, int g, int heads) {
    const int t = threadIdx.x;
    unsigned char* qp = sm + D41A_SM_Q;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = t + D41_ATTN_THREADS * i;      // 0..1023: head c / 64, chunk c % 64
        const int h = c >> 6, ch = c & 63;
        const int hg = g * D41_ATTN_HG + h;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (hg < heads) v = __ldg(reinterpret_cast<const uint4*>(q + (long long)row * q_stride + (long long)hg * D41_ATTN_D) + ch);
        *reinterpret_cast<uint4*>(qp + d41a_swz(h, ch)) = v;
    }
}

// leg PA1 (D41_ATTN_X_QROPE): d41a_load_q with q's forward RoPE applied on the way (the chunks of dims 448..511): per pair
// (x0, x1) of dims (d, d + 1), i = (d - 448) / 2: d41_rope_cs(pos, freq[i]); d41_rope_fwd; bf16 RNE each, exactly
// d41f_rope_rows' sequence on the same bf16 inputs. The q rows in global memory stay un-rotated (nothing else reads them).
__device__ __forceinline__ void d41a_load_q_rope(unsigned char* sm, const __nv_bfloat16* q, long long q_stride, int row, int g, int heads,
                                                 int pos, const float* freq) {
    const int t = threadIdx.x;
    unsigned char* qp = sm + D41A_SM_Q;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = t + D41_ATTN_THREADS * i;
        const int h = c >> 6, ch = c & 63;
        const int hg = g * D41_ATTN_HG + h;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (hg < heads) {
            v = __ldg(reinterpret_cast<const uint4*>(q + (long long)row * q_stride + (long long)hg * D41_ATTN_D) + ch);
            if (ch >= (D41_ATTN_D - D41_ATTN_ROPE) / 8) {
                unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float re = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] & 0xFFFFu)));
                    float im = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] >> 16)));
                    float cs, sn;
                    d41_rope_cs(pos, freq[4 * (ch - (D41_ATTN_D - D41_ATTN_ROPE) / 8) + j], cs, sn);
                    d41_rope_fwd(re, im, cs, sn);
                    w[j] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(re)) | ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(im)) << 16);
                }
                v = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        *reinterpret_cast<uint4*>(qp + d41a_swz(h, ch)) = v;
    }
}
