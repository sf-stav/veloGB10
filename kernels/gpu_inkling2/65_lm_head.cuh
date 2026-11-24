// gpu_inkling2/65_lm_head.cuh: the LM head — stored bf16 unembed (B32, the first unpadded_vocab rows; a TP rank
// passes its own row shard) against the fp32 final hidden, fp32 logits, final softcap 30.
//   x is split exactly-enough into two bf16 halves x = hi + lo (hi = bf16 RN(x), lo = bf16 RN(x - hi): 16 of the 24
//   significand bits, |x - hi - lo| <= 2^-17 |x|), two bf16 MMAs per weight fragment, every product exact in fp32:
//   the logits are the fp32 GEMM of the stored weights to ~1e-6 relative (the reference: fp32 logits).
//   logit = tanh(acc / cap) * cap  (the reference's softcap expression order).
// Fused sampler hook: per CTA and token the (max logit, lowest row id among equals) partial -> pmax / pidx
// [M][gridDim.x]; ink2_head_argmax reduces them (greedy / the drafter's anchor). Row ids are GLOBAL (row_base + n).
// grid ceil(N/128), block 256 (8 warps x one 16-row tile), M <= 16. K % 32 == 0. Rows are independent MMA rows:
// row 0 of an M-row launch is bit-identical to M = 1 (AGENTS §2.4).

__device__ __forceinline__ void ik2m_split8(const float4 a, const float4 b, uint4& hi, uint4& lo) {
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    uint32_t h[4], l[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 bh = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
        const float2 g = __bfloat1622float2(bh);
        const __nv_bfloat162 bl = __floats2bfloat162_rn(v[2 * j] - g.x, v[2 * j + 1] - g.y);
        h[j] = *reinterpret_cast<const uint32_t*>(&bh);
        l[j] = *reinterpret_cast<const uint32_t*>(&bl);
    }
    hi = make_uint4(h[0], h[1], h[2], h[3]);
    lo = make_uint4(l[0], l[1], l[2], l[3]);
}
__device__ __forceinline__ void ik2m_bf_group(float acc[4], const uint4 xah, const uint4 xal, const uint4 xbh,
                                              const uint4 xbl, const uint4 w) {
    ik2m_mma_bf(acc, xah.x, xbh.x, xah.y, xbh.y, w.x, w.y);
    ik2m_mma_bf(acc, xal.x, xbl.x, xal.y, xbl.y, w.x, w.y);
    ik2m_mma_bf(acc, xah.z, xbh.z, xah.w, xbh.w, w.z, w.w);
    ik2m_mma_bf(acc, xal.z, xbl.z, xal.w, xbl.w, w.z, w.w);
}

extern "C" __global__ __launch_bounds__(256) void ink2_lm_head(
        const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ logits,
        float* __restrict__ pmax, int* __restrict__ pidx, int M, int K, int N, int ldo, int row_base, float cap) {
    // S7: prefetch the first part of this CTA's 8 contiguous weight tiles while the predecessor runs (PDL)
    INK2_PDL_ENTRY_PREFETCH(W + (size_t)blockIdx.x * 8 * (K >> 5) * 512,
                            min(INK2_PF_BYTES, (unsigned)max(0, min(8, (N + 15) / 16 - (int)blockIdx.x * 8)) * (unsigned)(K >> 5) * 1024u));
    constexpr int G = 4;   // 32-k groups in flight per iteration
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    const bool live = n0w < N;
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    if (live) {
        const int ta = min(gid, M - 1), tb = min(gid + 8, M - 1);
        const float* xa = X + (size_t)ta * K + 8 * tig;
        const float* xb = X + (size_t)tb * K + 8 * tig;
        const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
        int k = 0;
        for (; k + 32 * G <= K; k += 32 * G) {
            uint4 b0[G], b1[G];
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                b0[g] = ik2m_ldw4(wt + (size_t)(k + 32 * g) * 16);
                b1[g] = ik2m_ldw4(wt + (size_t)(k + 32 * g) * 16 + 256);
            }
            #pragma unroll
            for (int g = 0; g < G; ++g) {
                uint4 ah, al, bh, bl;
                const int kk = k + 32 * g;
                ik2m_split8(__ldg(reinterpret_cast<const float4*>(xa + kk)), __ldg(reinterpret_cast<const float4*>(xa + kk + 4)), ah, al);
                ik2m_split8(__ldg(reinterpret_cast<const float4*>(xb + kk)), __ldg(reinterpret_cast<const float4*>(xb + kk + 4)), bh, bl);
                ik2m_bf_group(acc[0], ah, al, bh, bl, b0[g]);
                ik2m_bf_group(acc[1], ah, al, bh, bl, b1[g]);
            }
        }
        for (; k < K; k += 32) {
            const uint4 b0 = ik2m_ldw4(wt + (size_t)k * 16), b1 = ik2m_ldw4(wt + (size_t)k * 16 + 256);
            uint4 ah, al, bh, bl;
            ik2m_split8(__ldg(reinterpret_cast<const float4*>(xa + k)), __ldg(reinterpret_cast<const float4*>(xa + k + 4)), ah, al);
            ik2m_split8(__ldg(reinterpret_cast<const float4*>(xb + k)), __ldg(reinterpret_cast<const float4*>(xb + k + 4)), bh, bl);
            ik2m_bf_group(acc[0], ah, al, bh, bl, b0);
            ik2m_bf_group(acc[1], ah, al, bh, bl, b1);
        }
    }
    // epilogue: softcap, store, per-token (max, lowest id) of this lane's columns
    float bv[2] = {-INFINITY, -INFINITY};
    int bi[2] = {0x7fffffff, 0x7fffffff};
    if (live) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int n = n0w + 8 * h + 2 * tig + c;
                if (n >= N) continue;
                #pragma unroll
                for (int s = 0; s < 2; ++s) {   // s = 0: token gid (r = c), s = 1: token gid + 8 (r = 2 + c)
                    const int t = gid + 8 * s;
                    if (t >= M) continue;
                    const float v = tanhf(acc[h][2 * s + c] / cap) * cap;
                    logits[(size_t)t * ldo + n] = v;
                    const int id = row_base + n;
                    if (v > bv[s] || (v == bv[s] && id < bi[s])) { bv[s] = v; bi[s] = id; }
                }
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv[s], off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi[s], off);
            if (ov > bv[s] || (ov == bv[s] && oi < bi[s])) { bv[s] = ov; bi[s] = oi; }
        }
    }
    __shared__ float s_v[8][16];
    __shared__ int s_i[8][16];
    if (tig == 0) { s_v[warp][gid] = bv[0]; s_i[warp][gid] = bi[0]; s_v[warp][gid + 8] = bv[1]; s_i[warp][gid + 8] = bi[1]; }
    __syncthreads();
    if (threadIdx.x < M) {
        const int t = threadIdx.x;
        float v = s_v[0][t];
        int id = s_i[0][t];
        for (int w = 1; w < 8; ++w)
            if (s_v[w][t] > v || (s_v[w][t] == v && s_i[w][t] < id)) { v = s_v[w][t]; id = s_i[w][t]; }
        pmax[(size_t)t * gridDim.x + blockIdx.x] = v;
        pidx[(size_t)t * gridDim.x + blockIdx.x] = id;
    }
}

// The final argmax over ink2_lm_head's per-CTA partials: grid M, block 256 -> out_idx[t], out_val[t].
// (A TP rank's partials cover its row shard; the cross-rank argmax is the same reduction over the ranks' pairs.)
extern "C" __global__ __launch_bounds__(256) void ink2_head_argmax(
        const float* __restrict__ pmax, const int* __restrict__ pidx, int nparts, int* __restrict__ out_idx,
        float* __restrict__ out_val) {
    INK2_PDL_ENTRY();
    const int t = blockIdx.x;
    float v = -INFINITY;
    int id = 0x7fffffff;
    for (int p = threadIdx.x; p < nparts; p += blockDim.x) {
        const float pv = pmax[(size_t)t * nparts + p];
        const int pi = pidx[(size_t)t * nparts + p];
        if (pv > v || (pv == v && pi < id)) { v = pv; id = pi; }
    }
    __shared__ float s_v[256];
    __shared__ int s_i[256];
    s_v[threadIdx.x] = v; s_i[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = s_v[threadIdx.x + s];
            const int oi = s_i[threadIdx.x + s];
            if (ov > s_v[threadIdx.x] || (ov == s_v[threadIdx.x] && oi < s_i[threadIdx.x])) { s_v[threadIdx.x] = ov; s_i[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out_idx[t] = s_i[0]; out_val[t] = s_v[0]; }
}

// S7: the LM head over 17..48 rows (a verify) — ink2_lm_head's arithmetic per row (the same split8 + bf16 MMA
// sequence in the same k order, the same softcap), with up to three 16-row groups sharing every weight fragment, so
// the 0.92 GB of unembed rows are read ONCE for all rows (the 16-row kernel would read them once per 16 rows).
// No argmax partials (verify reads the logits). grid ceil(N/128), block 256, 17 <= M <= 48.
extern "C" __global__ __launch_bounds__(256) void ink2_lm_head_g(
        const uint16_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ logits,
        int M, int K, int N, int ldo, float cap) {
    INK2_PDL_ENTRY();
    constexpr int G = 4, NG = 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int tile = blockIdx.x * 8 + warp;
    const int n0w = tile * 16;
    if (n0w >= N) return;
    const int ng = (M + 15) >> 4;
    float acc[NG][2][4];
    const float* xa[NG];
    const float* xb[NG];
    #pragma unroll
    for (int q = 0; q < NG; ++q) {
        xa[q] = X + (size_t)min(16 * q + gid, M - 1) * K + 8 * tig;
        xb[q] = X + (size_t)min(16 * q + gid + 8, M - 1) * K + 8 * tig;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[q][h][r] = 0.f;
    }
    const uint16_t* wt = W + (size_t)tile * (K >> 5) * 512 + lane * 8;
    int k = 0;
    for (; k + 32 * G <= K; k += 32 * G) {
        uint4 b0[G], b1[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            b0[g] = ik2m_ldw4(wt + (size_t)(k + 32 * g) * 16);
            b1[g] = ik2m_ldw4(wt + (size_t)(k + 32 * g) * 16 + 256);
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            const int kk = k + 32 * g;
            #pragma unroll
            for (int q = 0; q < NG; ++q) {
                if (q < ng) {
                    uint4 ah, al, bh, bl;
                    ik2m_split8(__ldg(reinterpret_cast<const float4*>(xa[q] + kk)), __ldg(reinterpret_cast<const float4*>(xa[q] + kk + 4)), ah, al);
                    ik2m_split8(__ldg(reinterpret_cast<const float4*>(xb[q] + kk)), __ldg(reinterpret_cast<const float4*>(xb[q] + kk + 4)), bh, bl);
                    ik2m_bf_group(acc[q][0], ah, al, bh, bl, b0[g]);
                    ik2m_bf_group(acc[q][1], ah, al, bh, bl, b1[g]);
                }
            }
        }
    }
    for (; k < K; k += 32) {
        const uint4 b0 = ik2m_ldw4(wt + (size_t)k * 16), b1 = ik2m_ldw4(wt + (size_t)k * 16 + 256);
        #pragma unroll
        for (int q = 0; q < NG; ++q) {
            if (q < ng) {
                uint4 ah, al, bh, bl;
                ik2m_split8(__ldg(reinterpret_cast<const float4*>(xa[q] + k)), __ldg(reinterpret_cast<const float4*>(xa[q] + k + 4)), ah, al);
                ik2m_split8(__ldg(reinterpret_cast<const float4*>(xb[q] + k)), __ldg(reinterpret_cast<const float4*>(xb[q] + k + 4)), bh, bl);
                ik2m_bf_group(acc[q][0], ah, al, bh, bl, b0);
                ik2m_bf_group(acc[q][1], ah, al, bh, bl, b1);
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < NG; ++q) {
        if (q >= ng) continue;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int n = n0w + 8 * h + 2 * tig + c;
                if (n >= N) continue;
                #pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const int t = 16 * q + gid + 8 * s;
                    if (t >= M) continue;
                    logits[(size_t)t * ldo + n] = tanhf(acc[q][h][2 * s + c] / cap) * cap;
                }
            }
        }
    }
}
