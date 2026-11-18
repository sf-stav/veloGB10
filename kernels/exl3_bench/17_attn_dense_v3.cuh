// exl3_bench/17_attn_dense_v3.cuh: WP13 dense-regime attention v3 and spec/copy helper kernels.
// Same numbers as xq_attn_decode_group (v2) / xq_attn_decode, bit for bit.
// Section 17 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// =====================================================================================
// WP13 (PLAN/SURPASS_PLAN_2026-09-26): dense-regime attention v3 — the same numbers as
// xq_attn_decode_group (v2) / xq_attn_decode, bit for bit, at a fraction of the cost.
//
// v2 runs one CTA per (q head, row) and each CTA scans [0..p] TWICE (max pass, weight
// pass), recomputing every q.k dot and re-reading every K row 12x (GQA 12) and V row
// 12x. v3 splits the same arithmetic into three launches:
//   1. xq_attn_dense_prep  grid (m*nh):  row K/V cache write (the (row, kvh) CTA with
//      head % gqa == 0; xq_k_norm_rope + xq_kv_put, the shared cache-word code) and the
//      row's roped q (v2's q expressions verbatim) staged f32 -> qstage.
//   2. xq_attn_dense_dots  grid (nchg, nkv, m): ONE K-row load feeds all 12 heads' dots;
//      raw dots stored to a score plane S[row][head][t], per-64-position chunk maxima of
//      fl(raw*scale) to pmx[row][head][chunk].
//   3. xq_attn_dense_acc{1,4} grid (hd/(16*DPT), nkv, m): mx = fmaxf of the chunk maxima,
//      w = __expf(fma(raw, scale, -mx)) per tile, then the SEQUENTIAL ascending-t
//      denom/acc chains with V tiles staged by double-buffered cp.async.
//
// Bit contract (row r == v2 row r == xq_attn_decode of that token):
//  * K/V words + q: the same source expressions (shared xq_k_norm_rope / xq_kv_put, the
//    q block copied verbatim from v2), so the same PTX FMA pattern (rope: x1*c - x2*s is
//    mul/mul/sub, fused by ptxas identically in every copy — checked in SASS).
//  * raw dot: xq_dot_tree_kv's halving tree (dims d, d+128 / d+64 / d+32 in-register, then
//    lanes L, L+16 .. L+1) with __fmul_rn/__fadd_rn. The cross-lane half is done through a
//    per-warp smem transpose (lane L owns dot L and sums a[x] += a[x+s], s = 16..1 — the
//    same pairs as shfl_down's lane-0 value), so no FMA contraction is possible anywhere.
//  * mx: fmaxf over {__fmul_rn(raw, scale)} = SASS FMUL + FMNMX of v2 pass 1 (fmaxf is
//    exact and order-free, so chunking it changes nothing).
//  * w: v2's `__expf(raw * scale - mx)` compiles to FFMA(scale, raw, -mx) -> FMUL log2e ->
//    MUFU.EX2 in every copy (SASS, sm_121). Pinned here as __expf(__fmaf_rn(raw, scale,
//    -mx)) — never a CSE'd `s = raw*scale` (that would round twice).
//  * denom += w (FADD), acc = fma(w, v, acc) (FFMA), t ascending from 0.0f, then
//    f16((acc / denom) * sig(g)) — v2's closing expression.
// Graph contract: every grid depends only on m (fixed per captured graph); the position
// range comes from slotpos on device. The score plane holds XQ_DV3_TS positions — the host
// only takes this path where the dense regime is bounded by the QSA switch (positions
// <= qsa_limit = 2051 < 2304); a longer row would be a host bug, so acc writes NaN (loud)
// instead of silently dropping positions. KV formats f32/f16/fp8 via the XqKv helpers.
// =====================================================================================
#define XQ_DV3_TS  2304                      // score-plane row stride (positions): 9 x 256
#define XQ_DV3_CH  64                        // dots chunk: 8 warps x 8 positions
#define XQ_DV3_NCH (XQ_DV3_TS / XQ_DV3_CH)   // 36 chunk-max slots per (row, head)
#define XQ_DV3_G   12                        // q heads per kv head: the W<=2 / TP=1 instantiation
#define XQ_DV3_G6  6                         // TP-4G: the W=4 / TP=4 instantiation (6 heads per kv head)
#define XQ_DV3_T   64                        // accumulate tile (positions)

// The softmax scale exactly as v2 gets it: rsqrt.approx of the RUNTIME head dim (MUFU.RSQ
// with the non-ftz guard). The v3 bodies know hd == 256 after their geometry guard, and
// ptxas then folds rsqrt.approx(256.f) to the literal 0.0625 (seen in SASS; an empty asm
// barrier does not stop ptxas) — equal to MUFU.RSQ(256) only if the SFU is exact there.
// A volatile shared round trip is a value no compiler stage may assume, so the same
// instruction runs on the same input. Block-wide: every thread of the block must call it.
__device__ __forceinline__ float xq_dv3_scale(int hd) {
    __shared__ int xq_dv3_hd;
    if (threadIdx.x == 0) xq_dv3_hd = hd;
    __syncthreads();
    const int hv = *reinterpret_cast<volatile int*>(&xq_dv3_hd);
    return rsqrtf((float)hv);
}

// ---- 1. prep: grid (m*nh), block hd. Block (row, head).
extern "C" __global__ void xq_attn_dense_prep(float* __restrict__ qstage, float* __restrict__ kcvc,
                                              const __half* __restrict__ qg, const __half* __restrict__ kp,
                                              const __half* __restrict__ vp, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, const int* __restrict__ slotpos,
                                              int nh_packed, int hd_packed, int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int row = blockIdx.x / nh;
    const int head = blockIdx.x % nh;
    const int gqa = nh / nkv;
    const int tid = threadIdx.x;
    const int slot = slotpos[row * 2 + 0];
    const int p = slotpos[row * 2 + 1];
    const float* c = cossin + (long long)row * 2 * rdim;
    const float* s = c + rdim;
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    // ---- K/V: v2 phase-1 body for this row, once per (row, kv head) (block-uniform branch)
    if (head % gqa == 0) {
        const int kvh = head / gqa;
        xq_k_norm_rope(kbuf, red, kp + (long long)row * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
        xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)row * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
    }
    // ---- q: rmsnorm (1+w), rope — v2's q block verbatim; f32 staging
    float qv = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + tid]);
    red[tid] = qv * qv; __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
    qbuf[tid] = qv; __syncthreads();
    if (tid < rdim / 2) {
        float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
        qbuf[tid] = x1 * c[tid] - x2 * s[tid];
        qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
    // q8 (WP25): q into the cache's rotated basis exactly as v2 / xq_attn_decode / xq_attn_prep
    // do it (xq_h32r of the roped element; identity — no op emitted — for f32/f16/fp8).
    qstage[((long long)row * nh + head) * hd + tid] = xq_kv_rot(qbuf[tid], xq_kv_fmt(max_pos), tid & 31);
}

// ---- 2. dots: grid (nchg, nkv, m), block 256 (8 warps). Block (chunk-stride, kv head, row);
// chunks ch = blockIdx.x, +gridDim.x, ... < ceil((p+1)/64). Warp w owns positions
// ch*64 + w*8 + [0,8): their K rows (E=8 dims per lane: lane + 32j) sit in registers and
// feed all 12 heads. Per group of 4 heads the 8x4 = 32 lane-partials go through the
// warp's [32][33] transpose tile and lane L finishes dot L = (position L>>2, head L&3).
// TP-4G: the body is templated on G (q heads per kv head; 12 = W<=2 / TP=1, 6 = W=4 / TP=4).
// Heads are processed in groups of 4 (NHG = ceil(G/4)); at G=6 the second group holds 2 valid
// heads and its lanes for the other 2 are computed on stale tile rows and discarded (never
// stored, never in the max). Every head's dot is the same in-register + cross-lane halving
// tree at any G, so a head's bits do not depend on G. At G=12 (PARTIAL = false) every guard
// below is compiled out (SASS identical to the pre-TP-4G kernel up to register naming).
template <int G>
__device__ __forceinline__ void xq_dv3_dots_body(float* __restrict__ sc, float* __restrict__ pmx,
                                                 const float* __restrict__ qstage,
                                                 const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                                                 int nh_packed, int hd_packed, int max_pos) {
    constexpr int E = 8;                         // hd 256 (host-gated)
    constexpr int NHG = (G + 3) / 4;
    constexpr bool PARTIAL = (G % 4) != 0;       // the last 4-head group is short (G=6); false at G=12
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    if (hd != 32 * E || nh != G * nkv) return;   // host-gated; never a partial write
    const int kvh = blockIdx.y;
    const int row = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = slotpos[row * 2 + 0];
    const int npos = min(slotpos[row * 2 + 1] + 1, XQ_DV3_TS);
    const int nch = (npos + XQ_DV3_CH - 1) / XQ_DV3_CH;
    if ((int)blockIdx.x >= nch) return;          // block-uniform
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const float scale = xq_dv3_scale(hd);
    __shared__ __align__(16) float qs[G * 32 * E];   // the kv group's 12 roped q rows (12 KB)
    __shared__ float tb[8][32][33];                   // per-warp transpose tiles
    __shared__ float wmx[8][G];
    const long long hbase = (long long)row * nh + (long long)kvh * G;   // (row, first head of group)
    {
        const float4* qsrc = reinterpret_cast<const float4*>(qstage + hbase * hd);
        for (int i = threadIdx.x; i < G * hd / 4; i += blockDim.x)
            reinterpret_cast<float4*>(qs)[i] = qsrc[i];
    }
    __syncthreads();
    float* srow = sc + hbase * XQ_DV3_TS;
    for (int ch = blockIdx.x; ch < nch; ch += gridDim.x) {
        const int t0 = ch * XQ_DV3_CH + warp * 8;
        float kr[8][E];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int t = t0 + i;
            if (t < npos) {
                const char* krow = kc.k + (long long)t * kc.rb;
                #pragma unroll
                for (int j = 0; j < E; j++) kr[i][j] = xq_kv_ld(krow, kc.fmt, hd, lane + 32 * j);
            } else {
                #pragma unroll
                for (int j = 0; j < E; j++) kr[i][j] = 0.0f;
            }
        }
        #pragma unroll
        for (int hg = 0; hg < NHG; hg++) {
            #pragma unroll
            for (int hh = 0; hh < 4; hh++) {
                if (PARTIAL && hg * 4 + hh >= G) continue;   // skipped rows are never stored / maxed
                const float* qh = qs + (hg * 4 + hh) * (32 * E);
                float qv[E];
                #pragma unroll
                for (int j = 0; j < E; j++) qv[j] = qh[lane + 32 * j];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v[E];
                    #pragma unroll
                    for (int j = 0; j < E; j++) v[j] = __fmul_rn(qv[j], kr[i][j]);
                    #pragma unroll
                    for (int h2 = E / 2; h2 >= 1; h2 >>= 1) {
                        #pragma unroll
                        for (int j = 0; j < h2; j++) v[j] = __fadd_rn(v[j], v[j + h2]);
                    }
                    tb[warp][i * 4 + hh][lane] = v[0];
                }
            }
            __syncwarp();
            // lane L owns dot L: the cross-lane halving tree over the 32 lane partials
            float a[32];
            #pragma unroll
            for (int x = 0; x < 32; x++) a[x] = tb[warp][lane][x];
            #pragma unroll
            for (int s2 = 16; s2 >= 1; s2 >>= 1) {
                #pragma unroll
                for (int x = 0; x < s2; x++) a[x] = __fadd_rn(a[x], a[x + s2]);
            }
            const int t = t0 + (lane >> 2);
            const int h = hg * 4 + (lane & 3);
            float mv = -INFINITY;
            if (t < npos && (!PARTIAL || h < G)) {
                srow[(long long)h * XQ_DV3_TS + t] = a[0];
                mv = __fmul_rn(a[0], scale);
            }
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 4));
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 8));
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 16));
            if (lane < 4 && (!PARTIAL || hg * 4 + lane < G)) wmx[warp][hg * 4 + lane] = mv;
            __syncwarp();                         // tb is rewritten by the next head group
        }
        __syncthreads();
        if (threadIdx.x < G) {
            float mm = -INFINITY;
            #pragma unroll
            for (int w = 0; w < 8; w++) mm = fmaxf(mm, wmx[w][threadIdx.x]);
            pmx[(hbase + threadIdx.x) * XQ_DV3_NCH + ch] = mm;
        }
        __syncthreads();                          // wmx reuse by the next chunk
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
xq_attn_dense_dots(float* __restrict__ sc, float* __restrict__ pmx, const float* __restrict__ qstage,
                   const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                   int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_dots_body<XQ_DV3_G>(sc, pmx, qstage, kcvc, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(256, 2)
xq_attn_dense_dots_g6(float* __restrict__ sc, float* __restrict__ pmx, const float* __restrict__ qstage,
                      const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                      int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_dots_body<XQ_DV3_G6>(sc, pmx, qstage, kcvc, slotpos, nh_packed, hd_packed, max_pos);
}

// Stage V rows [t0, t0+T) of this CTA's dim slice (raw cache bytes, SD*eb per row; fp8 also
// the 16-B chunk holding the row scale; q8 the 16-B chunk holding the row's hd/32 f16 group
// scales) into one smem buffer as ONE cp.async group. RS = smem row stride (bytes).
template <int SD, int N16, bool FP8, int RS = SD * 4>
__device__ __forceinline__ void xq_dv3_stage_v(char* vbt, float* vsct, const XqKv& kc, int t0,
                                               int npos, int d0b, int scoff, int tid) {
    constexpr int NPER = N16 + (FP8 ? 1 : 0);
    for (int i = tid; i < XQ_DV3_T * NPER; i += blockDim.x) {
        const int tr = i / NPER;
        const int c = i - tr * NPER;
        const int t = t0 + tr;
        if (t < npos) {
            const char* src = kc.v + (long long)t * kc.rb;
            if (c < N16) moe_coop_cp16(vbt + tr * RS + c * 16, src + d0b + c * 16, 16);
            else         moe_coop_cp16(vsct + tr * 4, src + scoff, 16);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// V elements d0..d0+DPT-1 (thread dim-group dgl) at smem row tr, format FMT: the same values
// xq_kv_ld returns (f16 -> f32 exact; fp8 float(e4m3) * row scale, one FMUL).
template <int FMT, int DPT, int SD>
__device__ __forceinline__ void xq_dv3_ldv(float* v, const char* vbt, const float* vsct, int tr, int dgl) {
    const char* r = vbt + tr * (SD * 4);
    if constexpr (FMT == XQ_KV_F32) {
        if constexpr (DPT == 4) {
            const float4 f = *reinterpret_cast<const float4*>(r + dgl * 16);
            v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = reinterpret_cast<const float*>(r)[dgl * DPT + e];
        }
    } else if constexpr (FMT == XQ_KV_F16) {
        if constexpr (DPT == 4) {
            const uint2 u = *reinterpret_cast<const uint2*>(r + dgl * 8);
            const __half* hh = reinterpret_cast<const __half*>(&u);
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = __half2float(hh[e]);
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = __half2float(reinterpret_cast<const __half*>(r)[dgl * DPT + e]);
        }
    } else {
        const float scv = vsct[tr * 4];
        if constexpr (DPT == 4) {
            const unsigned u = *reinterpret_cast<const unsigned*>(r + dgl * 4);
            const __nv_fp8_e4m3* e8 = reinterpret_cast<const __nv_fp8_e4m3*>(&u);
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = float(e8[e]) * scv;
        } else {
            #pragma unroll
            for (int e = 0; e < DPT; e++) v[e] = float(reinterpret_cast<const __nv_fp8_e4m3*>(r)[dgl * DPT + e]) * scv;
        }
    }
}

// q8 (WP25) V elements d0..d0+DPT-1 at smem row tr (codes packed RS bytes per row, the row's
// 16-B group-scale chunk at vsct + tr*4): the SAME values xq_kv_ld / xq_kv_ld8 return —
// sm = f16 scale of group g * 2^-7 (xq_q8_sm), hs = 0.5*sm, v = xq_q8v(code). DPT in {2, 4}
// (never crosses a 32-group: d0 % DPT == 0).
template <int DPT, int RS>
__device__ __forceinline__ void xq_dv3_ldv_q8(float* v, const char* vbt, const float* vsct, int tr, int dgl, int g) {
    static_assert(DPT == 2 || DPT == 4, "q8 dense v3 slices hold whole 32-groups");
    const char* r = vbt + tr * RS;
    const float sm = __fmul_rn(__half2float(reinterpret_cast<const __half*>(vsct + tr * 4)[g]), 0.0078125f);
    const float hs = 0.5f * sm;
    unsigned u;
    if constexpr (DPT == 4) u = *reinterpret_cast<const unsigned*>(r + dgl * 4);
    else                    u = (unsigned)*reinterpret_cast<const unsigned short*>(r + dgl * 2);
    #pragma unroll
    for (int e = 0; e < DPT; e++) v[e] = xq_q8v(u, e, sm, hs);
}

// q4k V elements d0..d0+DPT-1 at smem row tr (codes packed RS = SD/2 bytes per row, the row's
// 16-B group-scale chunk at vsct + tr*4): the SAME values xq_kv_ld returns — sm = f16 scale of
// group d0>>5 * 1/8, hs = 0.5*sm, v = fma(code - 8, sm, hs).
template <int DPT, int RS>
__device__ __forceinline__ void xq_dv3_ldv_q4k(float* v, const char* vbt, const float* vsct, int tr, int dgl, int d0) {
    static_assert(DPT == 2 || DPT == 4, "q4k dense v3 slices hold whole 32-groups");
    const char* r = vbt + tr * RS;
    const float sm = __fmul_rn(__half2float(reinterpret_cast<const __half*>(vsct + tr * 4)[d0 >> 5]), 0.125f);
    const float hs = 0.5f * sm;
    unsigned u;
    if constexpr (DPT == 4) u = (unsigned)*reinterpret_cast<const unsigned short*>(r + dgl * 2);
    else                    u = (unsigned)*reinterpret_cast<const unsigned char*>(r + dgl);
    #pragma unroll
    for (int e = 0; e < DPT; e++)
        v[e] = __fmaf_rn((float)((int)((u >> (4 * e)) & 0xF) - 8), sm, hs);
}

// ---- 3. accumulate: grid (hd/SD, nkv, m), block 192 = 12 heads x 16 dim-groups of DPT dims.
// Thread (h = tid>>4, dgl = tid&15) owns dims d0..d0+DPT-1 of head kvh*12+h.
template <int G, int FMT, int DPT>
__device__ __forceinline__ void xq_dv3_acc_body(__half* __restrict__ attn, const float* __restrict__ sc,
                                                const float* __restrict__ pmx, const XqKv& kc,
                                                const __half* __restrict__ qg, int nh, int hd,
                                                int kvh, int row, int npos, char* vb, float* vsc, float* wb) {
    constexpr int T = XQ_DV3_T, SD = 16 * DPT, WST = XQ_DV3_T + 4;
    constexpr bool P4 = FMT == XQ_KV_Q4K;
    constexpr int EB = P4 ? 1 : (FMT == XQ_KV_F32 ? 4 : (FMT == XQ_KV_F16 ? 2 : 1));
    // q4k: SD/2 code bytes per row (2 dims per byte); q8: SD; f32/f16/fp8: SD*4.
    constexpr int N16 = P4 ? SD / 32 : SD * EB / 16;
    // q8 (WP25): codes packed SD bytes per smem row (so the DPT=2 body fits acc1's buffer);
    // f32/f16/fp8 keep the SD*4-byte stride (their code is unchanged). SCH = the row carries a
    // 16-B scale chunk (fp8 row scale | q8/q4k group scales, hd == 256).
    constexpr bool Q8 = FMT >= XQ_KV_Q8;
    constexpr int RS = P4 ? SD / 2 : (Q8 ? SD : SD * 4);
    constexpr bool SCH = FMT == XQ_KV_FP8 || Q8;
    static_assert(!Q8 || DPT >= 2, "q8: a thread's 16-dgl slice must hold whole 32-groups");
    const int tid = threadIdx.x;
    const int h = tid >> 4;
    const int dgl = tid & 15;
    const int d0 = blockIdx.x * SD + dgl * DPT;
    const long long hrow = (long long)row * nh + (long long)kvh * G + h;
    const float* srow = sc + hrow * XQ_DV3_TS;
    const float scale = xq_dv3_scale(hd);
    const int ntile = (npos + T - 1) / T;
    const int d0b = blockIdx.x * (P4 ? SD / 2 : SD * EB);
    const int scoff = P4 ? hd / 2 : hd;
    // tile 0's V copies and raw scores go out first so their latency overlaps the mx loads
    xq_dv3_stage_v<SD, N16, SCH, RS>(vb, vsc, kc, 0, npos, d0b, scoff, tid);
    float4 rw = *reinterpret_cast<const float4*>(srow + dgl * 4);
    // mx: fmaxf over the row's chunk maxima (exactly the set {fl(raw*scale) : t < npos})
    const int nch = (npos + XQ_DV3_CH - 1) / XQ_DV3_CH;
    float mx = -INFINITY;
    for (int c = dgl; c < nch; c += 16) mx = fmaxf(mx, pmx[hrow * XQ_DV3_NCH + c]);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    float acc[DPT];
    #pragma unroll
    for (int e = 0; e < DPT; e++) acc[e] = 0.0f;
    float denom = 0.0f;
    for (int ti = 0; ti < ntile; ti++) {
        const int t0 = ti * T;
        // weights of tile ti (positions t0 + dgl*4 + [0,4) of head h); t >= npos never read
        {
            float4 w4;
            w4.x = __expf(__fmaf_rn(rw.x, scale, -mx));
            w4.y = __expf(__fmaf_rn(rw.y, scale, -mx));
            w4.z = __expf(__fmaf_rn(rw.z, scale, -mx));
            w4.w = __expf(__fmaf_rn(rw.w, scale, -mx));
            *reinterpret_cast<float4*>(wb + h * WST + dgl * 4) = w4;
        }
        char* vcur = vb + (ti & 1) * (T * RS);
        float* scur = vsc + (ti & 1) * (T * 4);
        if (ti + 1 < ntile) {
            xq_dv3_stage_v<SD, N16, SCH, RS>(vb + ((ti + 1) & 1) * (T * RS),
                                                      vsc + ((ti + 1) & 1) * (T * 4),
                                                      kc, t0 + T, npos, d0b, scoff, tid);
            rw = *reinterpret_cast<const float4*>(srow + t0 + T + dgl * 4);
        } else {
            asm volatile("cp.async.commit_group;\n" ::: "memory");   // keep the group count aligned
        }
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");       // tile ti's copies (this thread)
        __syncthreads();                                              // ... and everyone's, + wb
        const int tn = min(T, npos - t0);
        const float* wh = wb + h * WST;
        int t = 0;
        for (; t + 4 <= tn; t += 4) {
            const float4 w4 = *reinterpret_cast<const float4*>(wh + t);
            const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                float v[DPT];
                if constexpr (P4)      xq_dv3_ldv_q4k<DPT, RS>(v, vcur, scur, t + u, dgl, d0);
                else if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t + u, dgl, d0 >> 5);
                else                   xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t + u, dgl);
                denom = __fadd_rn(denom, ws[u]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(ws[u], v[e], acc[e]);
            }
        }
        for (; t < tn; t++) {
            const float w = wh[t];
            float v[DPT];
            if constexpr (P4)      xq_dv3_ldv_q4k<DPT, RS>(v, vcur, scur, t, dgl, d0);
            else if constexpr (Q8) xq_dv3_ldv_q8<DPT, RS>(v, vcur, scur, t, dgl, d0 >> 5);
            else                   xq_dv3_ldv<FMT, DPT, SD>(v, vcur, scur, t, dgl);
            denom = __fadd_rn(denom, w);
            #pragma unroll
            for (int e = 0; e < DPT; e++) acc[e] = __fmaf_rn(w, v[e], acc[e]);
        }
        __syncthreads();                                              // vcur / wb reuse
    }
    const int head = kvh * G + h;
    if constexpr (Q8) {
        // q8: the output leaves the rotated basis exactly as v2's xq_kv_rot(acc / denom) does:
        // xq_h32's butterfly on the 32-group (element j = (d0 & 31) + e), stage s pairs j, j^s
        // (low a+b, high partner-minus-self; __fadd_rn/__fsub_rn), then one __fmul_rn by
        // 1/sqrt(32). Stages s < DPT are in-register; s >= DPT pair thread dgl with
        // dgl ^ (s/DPT) - same warp, same head (DPT >= 2 keeps s/DPT <= 8 < 16).
        float o[DPT];
        #pragma unroll
        for (int e = 0; e < DPT; e++) o[e] = acc[e] / denom;
        #pragma unroll
        for (int st = 1; st < 32; st <<= 1) {
            if (st < DPT) {
                float n[DPT];
                #pragma unroll
                for (int e = 0; e < DPT; e++) n[e] = (e & st) ? __fsub_rn(o[e ^ st], o[e]) : __fadd_rn(o[e], o[e ^ st]);
                #pragma unroll
                for (int e = 0; e < DPT; e++) o[e] = n[e];
            } else {
                const int tm = st / DPT;
                const bool hi = (dgl & tm) != 0;
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    const float pv = __shfl_xor_sync(0xffffffffu, o[e], tm);
                    o[e] = hi ? __fsub_rn(pv, o[e]) : __fadd_rn(o[e], pv);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            const int d = d0 + e;
            const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
            attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h(__fmul_rn(o[e], XQ_R32) * xq_sig(g));
        }
    } else {
    #pragma unroll
    for (int e = 0; e < DPT; e++) {
        const int d = d0 + e;
        const float g = xq_h2f(qg[(long long)row * nh * hd * 2 + (long long)head * hd * 2 + hd + d]);
        attn[(long long)row * nh * hd + (long long)head * hd + d] = xq_f2h((acc[e] / denom) * xq_sig(g));
    }
    }
}

template <int G, int DPT>
__device__ __forceinline__ void xq_dv3_acc_entry(__half* __restrict__ attn, const float* __restrict__ sc,
                                                 const float* __restrict__ pmx, const float* __restrict__ kcvc,
                                                 const __half* __restrict__ qg, const int* __restrict__ slotpos,
                                                 int nh_packed, int hd_packed, int max_pos) {
    constexpr int SD = 16 * DPT;
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    if (hd != 256 || nh != G * nkv) return;   // host-gated
    const int kvh = blockIdx.y;
    const int row = blockIdx.z;
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;
    __shared__ __align__(16) char vb[2 * XQ_DV3_T * SD * 4];
    __shared__ __align__(16) float vsc[2 * XQ_DV3_T * 4];
    __shared__ __align__(16) float wb[G * (XQ_DV3_T + 4)];
    if (npos > XQ_DV3_TS) {
        // host bug tripwire: the score plane cannot hold this row — loud, never truncated
        const int h = threadIdx.x >> 4, dgl = threadIdx.x & 15;
        #pragma unroll
        for (int e = 0; e < DPT; e++)
            attn[(long long)row * nh * hd + (long long)(kvh * G + h) * hd + blockIdx.x * SD + dgl * DPT + e] =
                __float2half(__int_as_float(0x7fc00000));
        return;
    }
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    switch (kc.fmt) {
        case XQ_KV_F32: xq_dv3_acc_body<G, XQ_KV_F32, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_F16: xq_dv3_acc_body<G, XQ_KV_F16, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_FP8: xq_dv3_acc_body<G, XQ_KV_FP8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb); break;
        case XQ_KV_Q4K:
            if constexpr (DPT >= 2) {
                xq_dv3_acc_body<G, XQ_KV_Q4K, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            } else {
                if ((int)blockIdx.x < hd / 32)   // block-uniform
                    xq_dv3_acc_body<G, XQ_KV_Q4K, 2>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            }
            break;
        default:
            // q8 / probe-only q4x (same layout + reader). The output un-rotation needs a whole
            // 32-group per thread-16 slice, so the single-row acc1 (DPT=1, 16-dim slices) runs
            // the DPT=2 body on its first hd/32 CTAs (32-dim slices; the rest exit) - the
            // grid (and so every captured graph) is unchanged. RS = SD bytes keeps the DPT=2
            // body inside acc1's static vb buffer (2*64*32 B <= 2*64*16*4 B).
            if constexpr (DPT >= 2) {
                xq_dv3_acc_body<G, XQ_KV_Q8, DPT>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            } else {
                if ((int)blockIdx.x < hd / 32)   // block-uniform
                    xq_dv3_acc_body<G, XQ_KV_Q8, 2>(attn, sc, pmx, kc, qg, nh, hd, kvh, row, npos, vb, vsc, wb);
            }
            break;
    }
}

// DPT=4 (64-dim slices, grid (4, nkv, m)) for verify widths; DPT=1 (16-dim slices, grid
// (16, nkv, m)) for the single-row decode step, where 4x the CTAs beats 4x the chains.
extern "C" __global__ void __launch_bounds__(192)
xq_attn_dense_acc4(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                   const float* __restrict__ kcvc, const __half* __restrict__ qg,
                   const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(192)
xq_attn_dense_acc1(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                   const float* __restrict__ kcvc, const __half* __restrict__ qg,
                   const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
// TP-4G: G=6 twins (block 16*6 = 96). Same body, same per-thread chains; only the head count of
// the CTA differs, so each (head, dim) thread runs the identical ascending-t fma chain.
extern "C" __global__ void __launch_bounds__(96)
xq_attn_dense_acc4_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G6, 4>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}
extern "C" __global__ void __launch_bounds__(96)
xq_attn_dense_acc1_g6(__half* __restrict__ attn, const float* __restrict__ sc, const float* __restrict__ pmx,
                      const float* __restrict__ kcvc, const __half* __restrict__ qg,
                      const int* __restrict__ slotpos, int nh_packed, int hd_packed, int max_pos) {
    XQ_PDL_ENTRY();
    xq_dv3_acc_entry<XQ_DV3_G6, 1>(attn, sc, pmx, kcvc, qg, slotpos, nh_packed, hd_packed, max_pos);
}

// WP13 XCHECK (GB10_EXL3_DENSE_XCHECK=1): bitwise diff of the v3 output vs the old kernel's
// output (re-run into a side buffer). One block of 256. st[0] = checks, st[1] = total
// mismatching halves. Prints every mismatch event and a heartbeat every 1024 checks.
extern "C" __global__ void xq_dv3_xcheck(const unsigned short* __restrict__ a, const unsigned short* __restrict__ b,
                                         int n, int* __restrict__ st, int tag) {
    XQ_PDL_ENTRY();
    __shared__ int cnt, first;
    if (threadIdx.x == 0) { cnt = 0; first = 0x7fffffff; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (a[i] != b[i]) { atomicAdd(&cnt, 1); atomicMin(&first, i); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int calls = atomicAdd(&st[0], 1);
        if (cnt) {
            const int tot = atomicAdd(&st[1], cnt) + cnt;
            printf("[dense-v3-xcheck] MISMATCH tag %d (layer %d, m %d, %s): %d of %d halves differ, "
                   "first idx %d new %04x old %04x (total %d)\n", tag, tag & 0xFF, (tag >> 8) & 0xFF,
                   (tag >> 16) ? "verify" : "step", cnt, n, first, (unsigned)a[first], (unsigned)b[first], tot);
        } else if ((calls & 1023) == 0) {
            printf("[dense-v3-xcheck] %d checks, %d mismatching halves so far\n", calls + 1, st[1]);
        }
    }
}

// ---- chained MoE-router-free? no — router/hc/MoE are row-independent (m-batched
// already). The remaining stateful ops: GDN conv ring + recurrent state, the PLE
// conv ring. Verify runs those against SHADOW buffers (xq_conv1d_chunk /
// xq_gdn_step_chunk / xq_ple_conv_chunk with slot=0). Commit = bit-identical
// replay of rows 0..a on the LIVE state, with C read from the device accept
// record slotacc[2] = {slot, a} (C = a+1). When a == m-1 the replay reproduces
// the shadow result exactly — the commit is then redundant but harmless; the
// launcher skips it in that case only if it knows a (it does not — a is
// device-side), so the replay always runs: <= 4 rows, sub-ms.

// GDN commit: xq_gdn_step_chunk with C = slotacc[1] + 1 (device-read).
extern "C" __global__ void xq_gdn_commit(__half* __restrict__ core,
                                         const __half* __restrict__ qkv,
                                         float* __restrict__ state,
                                         const __half* __restrict__ b_in,
                                         const __half* __restrict__ a_in,
                                         int nh, int n_k_heads, int kd, int vd,
                                         const float* __restrict__ a_log,
                                         const float* __restrict__ dt_bias,
                                         const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    int blk = blockIdx.x;
    int nchunk = vd / XQ_GDN_C;
    const int qkv_stride = 2 * n_k_heads * kd + nh * vd;
    int chunk = blk % nchunk; blk /= nchunk;
    int head = blk % nh;
    int key_head = head * n_k_heads / nh;
    int key_dim = n_k_heads * kd;
    int bb0 = chunk * XQ_GDN_C;
    extern __shared__ float sh[];
    float* S_sh = sh;
    float* Srow = S_sh + kd * XQ_GDN_SP;
    float* kv_mem = Srow + kd;
    float* vbuf = kv_mem + XQ_GDN_C;
    float* delta = vbuf + XQ_GDN_C;
    float* qrow = delta + XQ_GDN_C;
    float* krow = qrow + kd;
    float* S = state + ((long long)slot * nh + head) * kd * vd;
    xq_gdn_ld_state(S_sh, S, kd, vd, bb0);
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const __half* col = qkv + (long long)t * qkv_stride;
        float beta = 1.0f / (1.0f + __expf(-xq_h2f(b_in[(long long)t * nh + head])));
        float sp = xq_h2f(a_in[(long long)t * nh + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        float gt = __expf(-__expf(a_log[head]) * sp);
        xq_gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                     core + (long long)t * (nh * vd) + (long long)head * vd,
                     key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
        __syncthreads();
    }
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// S-A3-o G1: GDN commit from the verify's rank-1 ring. xq_gdn_token's state update
// per token is  S *= g;  S += krow ⊗ delta  (delta = (v - Sᵀk)·β uses the PRE-update
// state — computed once by the verify, which runs the same device function on the
// same live state and inputs). Replaying it only needs (g, krow, delta) per token, so
// the commit skips both norm reductions, the km matvec and the q readout, and holds
// each state row in registers. Per element the op sequence is identical to
// xq_gdn_token's (FMUL by g, then FFMA kk·delta + S), so the committed state is
// bitwise the replay's. grid (nh * vd/XQ_GDN_C), block kd; thread = one state row.
// WIDEVERIFY1 refinement 2: 11 -> 16 (a wide COPY round commits up to C = 16 rows). The extent
// sizes kr_sh / g_sh only; C bounds every loop, so a C <= 11 commit runs the same operations.
#define XQ_GDN_RING_CMAX 16   // WP23: mirrors GDN_RING_CMAX in exl3_forward.rs
extern "C" __global__ void __launch_bounds__(128)
xq_gdn_commit_ring(float* __restrict__ state, const float* __restrict__ ring,
                   int nh, int kd, int vd, const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    // S-A3-o G2: lane = column (warp loads/stores are 128 B coalesced), warp w owns
    // rows w, w+4, ... (kd/4 per thread, in registers); krow/g of all C tokens staged
    // in smem once. Per element the ops are unchanged (FMUL g, FFMA kk·delta + s).
    // WP23: extent 6 -> XQ_GDN_RING_CMAX = 8 rows (draft depth 7: C = a+1 <= 8; host gate
    // MTP_MAX_K+1 <= GDN_RING_CMAX). C only bounds the loops, so C <= 6 runs the same ops.
    __shared__ float kr_sh[XQ_GDN_RING_CMAX][128];
    __shared__ float g_sh[XQ_GDN_RING_CMAX];
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    // device guard: an accept count past the extent would overrun kr_sh — fail loudly, never
    // commit a truncated state (block-uniform: C is one device value for the whole launch)
    if (C < 1 || C > XQ_GDN_RING_CMAX) __trap();
    const int nchunk = vd / XQ_GDN_C;
    const int chunk = blockIdx.x % nchunk;
    const int head = blockIdx.x / nchunk;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = chunk * XQ_GDN_C + lane;
    for (int i = threadIdx.x; i < C * kd; i += blockDim.x) {
        const int t = i / kd, r = i - t * kd;
        kr_sh[t][r] = ring[((long long)t * nh + head) * XQ_GDN_RS + vd + r];
    }
    if (threadIdx.x < C) g_sh[threadIdx.x] = ring[((long long)threadIdx.x * nh + head) * XQ_GDN_RS + vd + kd];
    constexpr int RPT = 128 / 4;                 // rows per thread (kd = 128, 4 warps)
    float* S = state + ((long long)slot * nh + head) * kd * vd + col;
    float s[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) s[i] = S[(long long)(warp + 4 * i) * vd];
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const float gt = g_sh[t];
        const float d = __ldg(ring + ((long long)t * nh + head) * XQ_GDN_RS + col);
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] *= gt;
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = __fmaf_rn(kr_sh[t][warp + 4 * i], d, s[i]);
    }
    #pragma unroll
    for (int i = 0; i < RPT; i++) S[(long long)(warp + 4 * i) * vd] = s[i];
}

// GDN conv1d commit: replay the state window over rows 0..C-1 using the RAW
// (pre-conv) projections saved by the verify chunk (xq_conv1d_chunk overwrites
// qkv in place, so the post-conv silu values must NOT be shifted in). C =
// slotacc[1] + 1 (device-read). x rows are left untouched — they already hold
// the conv outputs the GDN commit consumes.
extern "C" __global__ void xq_conv_commit(float* __restrict__ state,
                                          const float* __restrict__ w, int conv_dim, int k,
                                          const __half* __restrict__ raw, int raw_off,
                                          const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(raw[raw_off + (long long)t * conv_dim + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
    }
    for (int j = 0; j < k; j++) sbase[j] = st[j];
}

// PLE ring commit: roll-only replay (xq_ple_conv_chunk's ring half; the conv
// add is NOT redone — resid rows 0..a already carry it). C = slotacc[1] + 1.
extern "C" __global__ void xq_ple_ring_commit(const __half* __restrict__ normed,
                                              __half* __restrict__ state,
                                              const int* __restrict__ slotacc) {
    XQ_PDL_ENTRY();
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    const int slot = slotacc[0];
    const int C = slotacc[1] + 1;
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    __half st[9];
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(normed[(long long)t * 10240 + d]);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// On-device greedy accept: d_vec[k] drafted ids, e_vec[m=k+1] verify argmaxes.
// a = longest prefix with d_i == e_i (0..k). out[0..k]: emit[i] = d_i for i < a;
// emit[a] = e_a (the bonus); rest = -1. acc2 = {slot, a} (device commit key —
// the commit kernels read it; the host's single readback is `out` only).
extern "C" __global__ void xq_accept(const int* __restrict__ d_vec,
                                     const int* __restrict__ e_vec,
                                     int* __restrict__ out,
                                     int* __restrict__ acc2, int k, int slot) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    int a = 0;
    while (a < k && d_vec[a] == e_vec[a]) a++;
    for (int i = 0; i <= k; i++) {
        if (i < a) out[i] = d_vec[i];
        else if (i == a) out[i] = e_vec[a];
        else out[i] = -1;
    }
    acc2[0] = slot;
    acc2[1] = a;
    // S-A3-f-d Item 1b FIX-3: the host's single readback is `out` and mtp_round
    // reads `a` from out[k+2] — without these stores it reads STALE memory while
    // the commit kernels use acc2's true a, so accepted rows were double-consumed
    // (state ran ahead of the token/position bookkeeping -> long-horizon drift).
    out[k + 1] = -1;
    out[k + 2] = a;
}

// PACK1 LEG 1: per-segment accept — grid (k_seg), one thread. Segment j's draft/argmax
// rows live at seg.row_start with len seg.len; the emitted layout per segment is the
// lone one shifted to row_start (d before a, e_vec[a] at a, -1 after), and the commit
// key lands in acc[j*2] = {slot_j, a_j} (LEG 2's segmented commits read it on device;
// LEG 1's probe reads it on host). No out[k+2] header: the host knows row_start/len.
extern "C" __global__ void xq_accept_seg(const int* __restrict__ d_vec,
                                         const int* __restrict__ e_vec,
                                         int* __restrict__ out, int* __restrict__ acc,
                                         const int* __restrict__ seg) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    const int s = blockIdx.x * 4;
    const int slot = seg[s], row0 = seg[s + 1], kj = seg[s + 2];
    const int* d = d_vec + row0;
    const int* e = e_vec + row0;
    int a = 0;
    // K2 (REL v0.7.3 review): bound on the DRAFTS (kj - 1 = k), like the lone xq_accept's
    // `a < k`. The e/d rows run to kj, but a must never reach kj: a zero/stale tail draft
    // slot equal to the last argmax made a = k + 1, the commit then replayed k + 2 rows
    // (reading the NEXT segment) and at k = 7 C > XQ_GDN_RING_CMAX trapped.
    while (a < kj - 1 && d[a] == e[a]) a++;
    // kj slots per segment (the lone path's extra out[k] header slot lives in acc here):
    // emitted at i <= a, -1 after — never writes past out[row0 + kj - 1].
    for (int i = 0; i < kj; i++) {
        if (i < a) out[row0 + i] = d[i];
        else if (i == a) out[row0 + i] = e[a];
        else out[row0 + i] = -1;
    }
    acc[blockIdx.x * 2] = slot;
    acc[blockIdx.x * 2 + 1] = a;
}

// ---- PACK2 LEG 2 (S-B9-CF-PACK2): per-segment DEVICE-KEYED commit kernels. ----
// Each calls the same device body as its served twin (xq_conv_commit /
// xq_gdn_commit_ring / xq_ple_ring_commit) with the commit key (slot, a) read
// ON DEVICE from acc_seg[j*2..+2] (xq_accept_seg's output — never the host) and
// the row range taken from the segment table seg[j*4+1] = row0: the packed
// verify's saves (qkv_save rows, gdn_ring_p ring rows, ple_normed rows) are
// ABSOLUTE rows of the R_pad-wide staging, so the replay reads rows
// row0 + t for t < C = a_j + 1. Grids put the segment on blockIdx.y.
// (The served commit on this model is the fused xq_gdn_commit_all_w4 —
// bitwise-equal to these per-layer bodies per its own gate's contract — but
// that kernel reads the served 5-row ring layout and a single acc2 key; a
// packed fused variant would exceed the 12-arg LaunchAsync cap. The packed
// commit runs the per-layer launches; commit cost is not what this leg times.)

// Per-segment conv commit: same walk as xq_conv_commit; raw rows at absolute
// rows of this layer's qkv_save plane (the host passes the plane base pointer,
// exactly like the served per-layer qkv_post/b_save/a_save offsets).
extern "C" __global__ void xq_conv_commit_seg(float* __restrict__ state,
                                              const float* __restrict__ w, int conv_dim, int k,
                                              const __half* __restrict__ raw,
                                              const int* __restrict__ seg,
                                              const int* __restrict__ acc_seg) {
    XQ_PDL_ENTRY();
    const int j = blockIdx.y;
    const int row0 = seg[j * 4 + 1];
    const int slot = acc_seg[j * 2];
    const int C = acc_seg[j * 2 + 1] + 1;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j2 = 0; j2 < k; j2++) st[j2] = sbase[j2];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(raw[(long long)(row0 + t) * conv_dim + c]);
        for (int j2 = 1; j2 < k; j2++) st[j2 - 1] = st[j2];
        st[k - 1] = cur;
    }
    for (int j2 = 0; j2 < k; j2++) sbase[j2] = st[j2];
}

// Per-segment GDN ring commit: same body as xq_gdn_commit_ring; ring rows at
// ABSOLUTE rows (xq_gdn_step_chunk_seg wrote its ring at row0 + t of the packed
// 16-row planes; the host passes this layer's plane base).
extern "C" __global__ void xq_gdn_commit_ring_seg(float* __restrict__ state,
                                                  const float* __restrict__ ring,
                                                  int nh, int kd, int vd,
                                                  const int* __restrict__ seg,
                                                  const int* __restrict__ acc_seg) {
    XQ_PDL_ENTRY();
    const int j = blockIdx.y;
    const int row0 = seg[j * 4 + 1];
    const int slot = acc_seg[j * 2];
    const int C = acc_seg[j * 2 + 1] + 1;
    // device guard: an accept count past the extent would overrun kr_sh — fail
    // loudly, never commit a truncated state (same contract as the served twin).
    if (C < 1 || C > XQ_GDN_RING_CMAX) __trap();
    const int nchunk = vd / XQ_GDN_C;
    const int chunk = blockIdx.x % nchunk;
    const int head = blockIdx.x / nchunk;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = chunk * XQ_GDN_C + lane;
    __shared__ float kr_sh[XQ_GDN_RING_CMAX][128];
    __shared__ float g_sh[XQ_GDN_RING_CMAX];
    for (int i = threadIdx.x; i < C * kd; i += blockDim.x) {
        const int t = i / kd, r = i - t * kd;
        kr_sh[t][r] = ring[((long long)(row0 + t) * nh + head) * XQ_GDN_RS + vd + r];
    }
    if (threadIdx.x < C) g_sh[threadIdx.x] = ring[((long long)(row0 + threadIdx.x) * nh + head) * XQ_GDN_RS + vd + kd];
    constexpr int RPT = 128 / 4;                 // rows per thread (kd = 128, 4 warps)
    float* S = state + ((long long)slot * nh + head) * kd * vd + col;
    float s[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) s[i] = S[(long long)(warp + 4 * i) * vd];
    __syncthreads();
    for (int t = 0; t < C; t++) {
        const float gt = g_sh[t];
        const float d = __ldg(ring + ((long long)(row0 + t) * nh + head) * XQ_GDN_RS + col);
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] *= gt;
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = __fmaf_rn(kr_sh[t][warp + 4 * i], d, s[i]);
    }
    #pragma unroll
    for (int i = 0; i < RPT; i++) S[(long long)(warp + 4 * i) * vd] = s[i];
}

// Per-segment PLE ring commit: same body as xq_ple_ring_commit; normed rows at
// ABSOLUTE rows of the packed ple_normed staging.
extern "C" __global__ void xq_ple_ring_commit_seg(const __half* __restrict__ normed,
                                                  __half* __restrict__ state,
                                                  const int* __restrict__ seg,
                                                  const int* __restrict__ acc_seg) {
    XQ_PDL_ENTRY();
    const int j = blockIdx.y;
    const int row0 = seg[j * 4 + 1];
    const int slot = acc_seg[j * 2];
    const int C = acc_seg[j * 2 + 1] + 1;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    __half st[9];
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(normed[(long long)(row0 + t) * 10240 + d]);
        for (int a2 = 0; a2 < 8; a2++) st[a2] = st[a2 + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a2 = 0; a2 < 9; a2++) sbase[a2] = st[a2];
}

// 1-element i32 device copy with dst offset (compute-stream D2D; AGENTS 2.3 —
// never a sync dtod). dst[doff] = src[0].
// S-A3-p: draft-chain pass setup from the per-round meta {b, p, slot, 0}: row 0 of the
// staging buffers = what head_forward's host uploads wrote (pos p+i, slot row), and the
// bonus token for pass 0 (later passes' toks[0] is the previous argmax, copied on device).
extern "C" __global__ void xq_draft_setup(int* __restrict__ toks, int* __restrict__ pos,
                                          int* __restrict__ slots, const int* __restrict__ meta, int i) {
    XQ_PDL_ENTRY();
    if (threadIdx.x != 0) return;
    const int p = meta[1] + i;
    pos[0] = p;
    slots[0] = meta[2];
    slots[1] = p;
    if (i == 0) toks[0] = meta[0];
}

// ---- REPRIME (w2, SURPASS WP22 rung 1 + graphed re-prime): the head re-prime at a FIXED row
// count R from device state only (graph-capturable, no host round trip). Target: baseline sm_121
// (plain CUDA: integer indexing, copies and the SAME inline K norm/rope + KV put as
// xq_attn_kv_prefill — no family/arch-specific instruction; bitwise equality with the eager
// kernels needs the same TU, target and flags, so these live next to them).
//
// Stage: a = acc2[1] (xq_accept's device commit key, clamped to [0, R]); p, slot, b from the
// round's draft_meta {b, p, slot, 0}. Row r < a = the eager re-prime's row r exactly: tap row r
// (taps_keep + soff), token emit[r] (= accepted draft d_r), position p+1+r. Rows r >= a are
// "masked": their inputs DUPLICATE row a-1 (a = 0: tap row 0, token b at position p), so every
// row-local kernel sees finite, in-range inputs (embedding index, cos_tab row), and their
// persistent writes are either suppressed (key rows, KV rows: meta[2] = a) or an identical-bits
// refresh of block(p+a)'s pooled key (see qsa_keys_append_dense_masked). meta = {slot, p+1, a, 0}.
extern "C" __global__ void xq_reprime_stage(float* __restrict__ resid, int* __restrict__ toks,
                                            int* __restrict__ pos, int* __restrict__ meta,
                                            const float* __restrict__ taps, const int* __restrict__ dmeta,
                                            const int* __restrict__ acc2, const int* __restrict__ emit,
                                            int R, long long he, long long soff) {
    XQ_PDL_ENTRY();
    int a = acc2[1];
    a = a < 0 ? 0 : (a > R ? R : a);
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)R * he) {
        const int r = (int)(i / he);
        const int src = r < a ? r : (a > 0 ? a - 1 : 0);
        resid[i] = taps[soff + (long long)src * he + (i - (long long)r * he)];
    }
    if (i < R) {
        const int r = (int)i;
        const int q = r < a ? r : a - 1;          // masked rows: row a-1 (a = 0: -1 = b at p)
        toks[r] = q >= 0 ? emit[q] : dmeta[0];
        pos[r] = dmeta[1] + 1 + q;
    }
    if (i == 0) {
        meta[0] = dmeta[2];
        meta[1] = dmeta[1] + 1;
        meta[2] = a;
        meta[3] = 0;
    }
}

// REPRIME: qsa_key_write_b (gpu_batch.cu) with the row count read on device — rows b >= meta[2]
// write nothing. Pure bit copy (bf16 as u16), so the written words equal qsa_key_write_b's.
extern "C" __global__ void xq_qsa_key_write_dn(unsigned short* __restrict__ keys,
                                               const unsigned short* __restrict__ qk,
                                               const int* __restrict__ pos, const int* __restrict__ slot_ids,
                                               int stride, int pitch, int k_off, int hd, int B,
                                               const int* __restrict__ meta) {
    XQ_PDL_ENTRY();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * hd) return;
    const int b = (int)(idx / hd), d = (int)(idx % hd);
    if (b >= meta[2]) return;
    // WP02 (p2 merge): qsa_key_write_b's window guard — a row past the slot's plane would land in
    // the next slot (or past the allocation). In-range writes are unchanged.
    const int col = pos[b];
    if (col < 0 || col >= stride) return;
    keys[((long long)slot_ids[b] * stride + col) * hd + d] = qk[(long long)b * pitch + k_off + d];
}

// REPRIME: xq_attn_kv_prefill with {slot, pos0, n} read on device (meta) — rows t < min(C, n)
// only. Loop body = xq_attn_kv_prefill's verbatim (shared inline K norm+rope and xq_kv_put), so
// every cache word it writes equals the eager kernel's for the same row. n is block-uniform, so
// the per-row __syncthreads stay convergent.
extern "C" __global__ void xq_attn_kv_rows_dn(__half* __restrict__ kp, __half* __restrict__ vp,
                                              float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, int nkv_hd,
                                              int rdim_maxpos, const int* __restrict__ meta, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = meta[0];
    const int pos0 = meta[1];
    const int n = min(C, meta[2]);
    const int kvh = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    for (int t = 0; t < n; t++) {
        const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
        xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                       cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
        const long long p = pos0 + t;
        xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
        __syncthreads();
    }
}

extern "C" __global__ void xq_copy1_i32(int* __restrict__ dst, long long doff,
                                        const int* __restrict__ src) {
    XQ_PDL_ENTRY();
    if (blockIdx.x == 0 && threadIdx.x == 0) dst[doff] = src[0];
}

// HOST / RO-5(a): save (dir 0: rows -> buf) / restore (dir 1: buf -> rows) the draft head's
// PERSISTENT rows at one position around a SPECULATIVE draft pass (mtp_round_adaptive): block
// r < nkvrows = the K|V cache row r (kv_row0 + r*kv_stride bytes, rb bytes), block nkvrows =
// the raw indexer-key row, block nkvrows+1 = the pooled-plane block row (0 bytes / null = absent).
// buf = [nkvrows*rb | key_bytes | plane_bytes]. A pure 4-byte-word copy (every size and offset
// is a multiple of 4 — asserted on the host), so a restore returns the exact pre-speculation bits.
// ADDRESSING (review fix): the K|V rows are the lane's slot (the pass's xq_attn_prep/_decode write
// through sc.slots = draft_meta's slot, baked into kv_row0 on the host), but the raw-key row and
// the pooled block are written by qsa_key_write_b / qsa_pool_update_b at slot = slot_ids[0] — the
// DEVICE slot table, which the DDS round stages only in the QSA-live regime (in the dense regime
// it holds whatever the previous launch left, possibly another lane's slot). So those two rows
// are addressed HERE from the same slot_ids[0] the pass reads: key_row0 / plane_row0 = the slot-0
// row of the position, sstrides = key slot stride (bytes, low 32) | plane slot stride (bytes,
// high 32), kp_bytes = key_bytes | plane_bytes << 16, nslots_dir = nslots | dir << 16. slot_ids is
// not rewritten between the save and the restore (only the speculative pass runs in between), so
// both resolve to the same rows. An out-of-range slot (never staged) skips both rows consistently.
extern "C" __global__ void xq_spec_rows(unsigned int* __restrict__ buf, unsigned long long kv_row0,
                                        long long kv_stride, int rb, int nkvrows,
                                        unsigned long long key_row0, unsigned long long plane_row0,
                                        long long sstrides, int kp_bytes,
                                        const int* __restrict__ slot_ids, int nslots_dir) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int key_bytes = kp_bytes & 0xFFFF, plane_bytes = (kp_bytes >> 16) & 0xFFFF;
    const int nslots = nslots_dir & 0xFFFF, dir = (nslots_dir >> 16) & 0xFFFF;
    unsigned long long row;
    long long boff;
    int nb;
    if (r < nkvrows) { row = kv_row0 + (unsigned long long)((long long)r * kv_stride); boff = (long long)r * rb; nb = rb; }
    else {
        const int s = slot_ids[0];
        if (s < 0 || s >= nslots) return;
        const unsigned long long ss = (unsigned long long)sstrides;
        if (r == nkvrows) {
            row = key_row0 ? key_row0 + (unsigned long long)s * (ss & 0xFFFFFFFFull) : 0ull;
            boff = (long long)nkvrows * rb; nb = key_bytes;
        } else {
            row = plane_row0 ? plane_row0 + (unsigned long long)s * (ss >> 32) : 0ull;
            boff = (long long)nkvrows * rb + key_bytes; nb = plane_bytes;
        }
    }
    if (nb <= 0 || row == 0ull) return;
    unsigned int* g = reinterpret_cast<unsigned int*>(row);
    unsigned int* b = buf + boff / 4;
    const int nw = nb >> 2;
    if (dir == 0) { for (int i = threadIdx.x; i < nw; i += blockDim.x) b[i] = g[i]; }
    else          { for (int i = threadIdx.x; i < nw; i += blockDim.x) g[i] = b[i]; }
}

// block f32 copy (shadow-state staging + tap-row restore; compute-stream).
extern "C" __global__ void xq_copy_f32(float* __restrict__ dst, const float* __restrict__ src,
                                       long long n, long long doff, long long soff) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[doff + i] = src[soff + i];
}

// block u16 copy (PLE ring shadow staging + verify per-layer saves).
// S-A3-f-d Item 1b: 5-arg (doff, soff) mirroring xq_copy_f32 — the previous
// 3-arg version SILENTLY IGNORED the offsets its call sites passed (the
// driver drops surplus args), corrupting the verify commit's raw saves.
extern "C" __global__ void xq_copy_u16(unsigned short* __restrict__ dst,
                                       const unsigned short* __restrict__ src,
                                       long long n, long long doff, long long soff) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[doff + i] = src[soff + i];
}

