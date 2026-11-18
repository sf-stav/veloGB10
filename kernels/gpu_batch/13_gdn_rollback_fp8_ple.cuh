// gpu_batch/13_gdn_rollback_fp8_ple.cuh: E0 fused GDN slot restore (rollback), FP8 pack/quant kernels, PLE and hc helper kernels.
// F8 parity lane: W8A8 FP8 prefill GEMM.
// Section 13 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ---- E0: fused GDN slot restore (rollback). Replaces the host loop in GpuModel::copy_gdn_slot that
// issued 2*n_gdn cudaMemcpyDtoDAsync calls per rollback (48 layers x {conv_state, s_state} on the
// 27B = 96 driver calls, ~25% of the rollback span idle in issue gaps — E1's nsys inter-graph
// budget). `table` is [n_gdn] conv_state base pointers followed by [n_gdn] s_state base pointers,
// both rank-local (TP-aware strides are the caller's, exactly as the old per-layer dtod math).
// gridDim.y partitions each layer's flat byte range [0, cb+sb) so the copy spreads over ~4 blocks
// per layer (192 blocks at n_gdn=48 — the one-block-per-layer shape leaves the GPU ~17% occupied
// and cannot hold the 238 GB/s roofline). PURE BYTE COPY: no arithmetic, so it is bit-identical to
// the dtod memcpys by construction. 16B vector path when both strides are 16B multiples (every
// current model: conv 40,960 B, s 786,432 B at rank-local 12 v-heads), u32 path otherwise (f32
// tensors are always 4B aligned).
extern "C" __global__ void gdn_rollback_b(const unsigned long long* __restrict__ table,
                                          int n_gdn, unsigned long long cb, unsigned long long sb,
                                          int src, int dst)
{
    const int li = blockIdx.x;
    if (li >= n_gdn) return;
    const unsigned char* conv_src = (const unsigned char*)table[li]            + (unsigned long long)src * cb;
    unsigned char*       conv_dst = (unsigned char*)table[li]            + (unsigned long long)dst * cb;
    const unsigned char* s_src    = (const unsigned char*)table[n_gdn + li] + (unsigned long long)src * sb;
    unsigned char*       s_dst    = (unsigned char*)table[n_gdn + li] + (unsigned long long)dst * sb;
    // Independent per-region partitions across gridDim.y — a single flat split could span the
    // conv/s_state boundary and copy neighbor-slot bytes; two ranges cannot. Partition widths
    // round up to 16B so the uint4 path's addresses stay aligned (both strides are 16B multiples
    // in that branch, hence every boundary too).
    unsigned long long cc = (cb + gridDim.y - 1) / gridDim.y;
    unsigned long long cs = (sb + gridDim.y - 1) / gridDim.y;
    cc = (cc + 15ULL) & ~15ULL;
    cs = (cs + 15ULL) & ~15ULL;
    const unsigned long long clo = (unsigned long long)blockIdx.y * cc;
    const unsigned long long slo = (unsigned long long)blockIdx.y * cs;
    const long long cn = (clo < cb) ? (long long)min(cb - clo, cc) : 0LL;
    const long long sn = (slo < sb) ? (long long)min(sb - slo, cs) : 0LL;
    const bool vec = (cb % 16 == 0) && (sb % 16 == 0);
    #define GDN_CPY(T, sp, dp, n) do {                                     \
        T const* s_ = (T const*)((sp)); T* d_ = (T*)(dp);                  \
        for (long long i = threadIdx.x; i < (n); i += blockDim.x) d_[i] = s_[i]; \
    } while (0)
    if (cn > 0) {
        if (vec) GDN_CPY(uint4,  conv_src + clo, conv_dst + clo, cn >> 4);
        else     GDN_CPY(unsigned int, conv_src + clo, conv_dst + clo, cn >> 2);
    }
    if (sn > 0) {
        if (vec) GDN_CPY(uint4,  s_src + slo, s_dst + slo, sn >> 4);
        else     GDN_CPY(unsigned int, s_src + slo, s_dst + slo, sn >> 2);
    }
    #undef GDN_CPY
}

// ===========================================================================
// F8 PARITY LANE (2026-09-07): W8A8 FP8 prefill GEMM — e4m3 m16n8k32 tensor
// cores for the PREFILL regime (the decode regime stays on gemm_mma_fp8_blk_b;
// measured dead-end note: at M=16 the dequant+bf16 path is at the DRAM
// roofline, at M=8192 the bf16 path is compute-bound at ~73% of the BF16 TC
// peak and e4m3 doubles the ceiling — the recipe's ~2x prefill).
//
// Pipeline: fp8_repack_e_b (one-time, on-device: k16-frag codes -> k32 e4m3
// A-frag panels), fp8_blk_scales_b + fp8_pack_b (per GEMM call: X bf16 ->
// per-(token,128k) f32 scales + B-frag e4m3 panels), gemm_fp8_prefill_e_b
// (128x64 tile, 8 warps of 32x32, 4-stage cp.async, raw accumulators folded
// by w_s(block,128x128 band) x x_s(token,128k) every 128-K span).
// Oracle: e4m3 fragment table validated bit-exact (tool_probe/e4m3_mma_probe.cu).
// ===========================================================================

__device__ __forceinline__ void mma_e4m3_k32(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp16_b(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp8_b(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit_b() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait_b() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// One-time weight repack: k16-bf16-frag u64 tiles -> e4m3 k32 A-frag panels.
// Old tile (mt, kb16), lane l=g*4+t, u64 bytes:
//   [g:2t][g:2t+1][g+8:2t][g+8:2t+1][g:2t+8][g:2t+9][g+8:2t+8][g+8:2t+9]
// New panel (mt, kb32), lane l, 4xu32:
//   a0={g, k4t..+3} a1={g+8, k4t..+3} a2={g, k4t+16..+3} a3={g+8, +16..+3}
// Byte gather crosses the t-field (target k 4t..4t+3 spans two source lanes);
// done warp-internally with shfl + prmt. Grid-stride over panels x 32 lanes.
extern "C" __global__ void fp8_repack_e_b(
    const unsigned long long* __restrict__ Wt,   // [(M/16)*(K/16)] old tiles
    unsigned* __restrict__ We,                   // [(M/16)*(K/32)][128] new panels
    int m, int k)
{
    const int nk32 = k >> 5, nk16 = k >> 4;
    const long long npan = (long long)(m >> 4) * nk32;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < npan * 32;
         i += (long long)gridDim.x * blockDim.x) {
        const int lane = (int)(i & 31);
        const long long pan = i >> 5;
        const int mt = (int)(pan / nk32), kb32 = (int)(pan % nk32);
        const int g = lane >> 2, t = lane & 3;
        const long long base = (long long)mt * nk16;
        const unsigned long long wa = Wt[(base + 2 * kb32) * 32 + lane];
        const unsigned long long wb = Wt[(base + 2 * kb32 + 1) * 32 + lane];
        const unsigned wa_lo = (unsigned)wa, wa_hi = (unsigned)(wa >> 32);
        const unsigned wb_lo = (unsigned)wb, wb_hi = (unsigned)(wb >> 32);
        // target k 4t..4t+3 lives in source lanes 2t and 2t+1 (t<2: low bytes
        // 0,1 / 2,3; t>=2: their k is in the +8 half, bytes 4,5 / 6,7)
        const int d = (t < 2) ? t : (t - 4);
        const int L0 = lane + d, L1 = lane + d + 1;
        const unsigned msk = 0xFFFFFFFFu;
        unsigned s[8];
        s[0] = __shfl_sync(msk, wa_lo, L0); s[1] = __shfl_sync(msk, wa_lo, L1);
        s[2] = __shfl_sync(msk, wa_hi, L0); s[3] = __shfl_sync(msk, wa_hi, L1);
        s[4] = __shfl_sync(msk, wb_lo, L0); s[5] = __shfl_sync(msk, wb_lo, L1);
        s[6] = __shfl_sync(msk, wb_hi, L0); s[7] = __shfl_sync(msk, wb_hi, L1);
        unsigned a[4];
        if (t < 2) {
            a[0] = __byte_perm(s[0], s[1], 0x5410);   // g  : bytes 0,1 | 0,1
            a[1] = __byte_perm(s[0], s[1], 0x7632);   // g+8: bytes 2,3 | 2,3
            a[2] = __byte_perm(s[4], s[5], 0x5410);
            a[3] = __byte_perm(s[4], s[5], 0x7632);
        } else {
            a[0] = __byte_perm(s[2], s[3], 0x5410);   // +8k half: bytes 4,5 / 6,7
            a[1] = __byte_perm(s[2], s[3], 0x7632);
            a[2] = __byte_perm(s[6], s[7], 0x5410);
            a[3] = __byte_perm(s[6], s[7], 0x7632);
        }
        uint4* dst = (uint4*)(We + pan * 128 + (unsigned)lane * 4);
        *dst = make_uint4(a[0], a[1], a[2], a[3]);
    }
}

// Per-(token, 128-K span) amax scale for the activation side: Xs[row][kspan] = amax/448.
extern "C" __global__ void fp8_blk_scales_b(
    const __nv_bfloat16* __restrict__ X, float* __restrict__ Xs, int rows, int k, int kspans)
{
    // Launched over the PADDED row count (tile multiple): rows >= real tokens write the
    // identity scale and never touch X (the activation buffer only holds real rows).
    const int row = blockIdx.x;
    if (row >= rows) {
        for (int s = threadIdx.x; s < kspans; s += blockDim.x)
            Xs[(long long)row * kspans + s] = 1.0f;
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int s = warp; s < kspans; s += nw) {
        const __nv_bfloat16* xr = X + (long long)row * k + s * 128;
        float am = 0.f;
        {   // 32 lanes x 4 elems = the full 128-elem span (uint2 = 2 x bf16x2)
            const uint2 u = *(const uint2*)(xr + lane * 4);
            const __nv_bfloat162 p0 = *(__nv_bfloat162*)&u.x, p1 = *(__nv_bfloat162*)&u.y;
            am = fmaxf(am, fmaxf(fabsf(__bfloat162float(p0.x)), fabsf(__bfloat162float(p0.y))));
            am = fmaxf(am, fmaxf(fabsf(__bfloat162float(p1.x)), fabsf(__bfloat162float(p1.y))));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
        if (lane == 0) Xs[(long long)row * kspans + s] = (am > 0.f) ? am / 448.0f : 1.0f;
    }
}

// Activation pack: X bf16 + Xs -> Bq e4m3 B-frag panels [(T8)*nk32][64 u32]
// (lane l=(g,t): b0 = codes[tok g][k 4t..+3], b1 = codes[tok g][k 4t+16..+3]).
// One warp per (8-token group, 128-K span) writes its 4 k32 panels.
extern "C" __global__ void fp8_pack_b(
    const __nv_bfloat16* __restrict__ X, const float* __restrict__ Xs,
    unsigned* __restrict__ Bq, int rows, int k, int kspans)
{
    const long long job = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = (int)(job & 31);
    const long long t8s = job >> 5;                 // (t8 * kspans + kspan)
    const int t8 = (int)(t8s / kspans), kspan = (int)(t8s % kspans);
    const int nk32 = k >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int row = t8 * 8 + g;
    // Pad tokens (row >= rows) contribute zero codes: they are never read back by
    // consumers, and zeros keep the tile arithmetic free of garbage NaN patterns.
    const float inv = (row < rows) ? 1.0f / Xs[(long long)row * kspans + kspan] : 0.0f;
    const __nv_bfloat16* xr = X + ((long long)(row < rows ? row : 0)) * k + (long long)kspan * 128;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {                   // 4 k32 panels per 128-K span
        // uint2 = 4 bf16 = 4 e4m3 codes: a full 4-k byte group per u32 fragment register.
        const uint2 lo = *(const uint2*)(xr + j * 32 + 4 * t);
        const uint2 hi = *(const uint2*)(xr + j * 32 + 16 + 4 * t);
        unsigned b[2];
        {
            const __nv_bfloat162 p0 = *(__nv_bfloat162*)&lo.x;   // k 4t, 4t+1
            const __nv_bfloat162 p1 = *(__nv_bfloat162*)&lo.y;   // k 4t+2, 4t+3
            b[0] = (unsigned)f32_to_e4m3(__bfloat162float(p0.x) * inv)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p0.y) * inv) << 8)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p1.x) * inv) << 16)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p1.y) * inv) << 24);
        }
        {
            const __nv_bfloat162 p0 = *(__nv_bfloat162*)&hi.x;   // k 4t+16, 4t+17
            const __nv_bfloat162 p1 = *(__nv_bfloat162*)&hi.y;   // k 4t+18, 4t+19
            b[1] = (unsigned)f32_to_e4m3(__bfloat162float(p0.x) * inv)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p0.y) * inv) << 8)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p1.x) * inv) << 16)
                 | ((unsigned)f32_to_e4m3(__bfloat162float(p1.y) * inv) << 24);
        }
        uint2* dst = (uint2*)(Bq + ((long long)t8 * nk32 + (kspan << 2) + j) * 64 + (unsigned)lane * 2);
        *dst = make_uint2(b[0], b[1]);
    }
}

// The GEMM: Out[tok][row] bf16 = (W[e4m3] x diag(w_s) ) x (diag(x_s) x X[e4m3])
// with raw per-128-K accumulators folded at span boundaries.
//   tile 128 rows x 64 tokens, 8 warps (wm 0..3 x wn 0..1), warp tile 32x32
//   (A 2x16-row frags, B 4x8-token frags, 8 mma/panel), 4-stage cp.async,
//   stage = {A 8x512B | B 8x256B} = 6144 B, epilogue via smem transpose.
// Requires: M%128==0, K%128==0, T%64==0 (host pads at pack time).
// Fused activation quant (2026-09-07, expert-review item): ONE pass over X —
// stage the 128-elem span in smem, warp-reduce amax, encode e4m3 from smem, write
// both the scale and the 4 k32 B-frag panels. Pad rows (row >= rows): identity
// scale + zero codes, never touching X.
extern "C" __global__ void fp8_quant_pack_b(
    const __nv_bfloat16* __restrict__ X, unsigned* __restrict__ Bq, float* __restrict__ Xs,
    int rows, int k, int kspans)
{
    // One warp per (8-token group, 128-K span). For each of the 8 rows: every lane loads
    // that row's [lane*4, lane*4+4) span (all 32 lanes cover the full 128), warp-reduces
    // amax, encodes 4 codes -> one u32 per lane (k-chunk lane*4..+3). The B-frag for lane
    // (g,t) needs row g's code-chunks 8j+t (b0) and 8j+t+4 (b1) — exactly one u32 each —
    // so the (r,j) shuffle broadcasts whole chunks with no byte surgery. Pad rows
    // (row >= rows): identity scale, zero codes, X never read.
    const long long job = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = (int)(job & 31);
    const long long t8s = job >> 5;                 // (t8 * kspans + kspan)
    const int t8 = (int)(t8s / kspans), kspan = (int)(t8s % kspans);
    const int nk32 = k >> 5;
    const int g = lane >> 2, t = lane & 3;

    unsigned b[4][2];
    #pragma unroll
    for (int j = 0; j < 4; ++j) b[j][0] = 0u, b[j][1] = 0u;

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = t8 * 8 + r;
        float am = 0.f;
        uint2 u = make_uint2(0u, 0u);
        if (row < rows) {
            const __nv_bfloat16* xr = X + (long long)row * k + (long long)kspan * 128;
            u = *(const uint2*)(xr + lane * 4);
            const __nv_bfloat162 p0 = *(__nv_bfloat162*)&u.x, p1 = *(__nv_bfloat162*)&u.y;
            am = fmaxf(am, fmaxf(fabsf(__bfloat162float(p0.x)), fabsf(__bfloat162float(p0.y))));
            am = fmaxf(am, fmaxf(fabsf(__bfloat162float(p1.x)), fabsf(__bfloat162float(p1.y))));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, o));
        const float sc = (am > 0.f) ? am / 448.0f : 1.0f;
        if (lane == 0) Xs[(long long)row * kspans + kspan] = sc;
        const float inv = 1.0f / sc;
        unsigned cr = 0;
        {
            const __nv_bfloat162 p0 = *(__nv_bfloat162*)&u.x, p1 = *(__nv_bfloat162*)&u.y;
            cr = (unsigned)f32_to_e4m3(__bfloat162float(p0.x) * inv)
               | ((unsigned)f32_to_e4m3(__bfloat162float(p0.y) * inv) << 8)
               | ((unsigned)f32_to_e4m3(__bfloat162float(p1.x) * inv) << 16)
               | ((unsigned)f32_to_e4m3(__bfloat162float(p1.y) * inv) << 24);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v0 = __shfl_sync(0xFFFFFFFFu, cr, 8 * j + t);
            const unsigned v1 = __shfl_sync(0xFFFFFFFFu, cr, 8 * j + t + 4);
            if (g == r) { b[j][0] = v0; b[j][1] = v1; }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint2* dst = (uint2*)(Bq + ((long long)t8 * nk32 + (kspan << 2) + j) * 64 + (unsigned)lane * 2);
        *dst = make_uint2(b[j][0], b[j][1]);
    }
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_fp8_prefill_e_b(
    const unsigned* __restrict__ We,      // [(M/16)*(K/32)][128]
    const unsigned* __restrict__ Bq,      // [(T/8)*(K/32)][64]
    const float* __restrict__ Xs,         // [T][K/128]
    const float* __restrict__ Ws,         // [(M/128)*(K/128)] block scales
    __nv_bfloat16* __restrict__ Out,      // [T][M] token-major
    int M, int T, int K)
{
    constexpr int BM = 128, BN = 64, STAGE_A = 4096, STAGE_B = 2048;
    constexpr int STAGE = STAGE_A + STAGE_B, NSTAGE = 4;
    const int nk32 = K >> 5, nks = K >> 7;
    const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int warp = tid >> 5, wm = warp >> 1, wn = warp & 1;

    // token-fastest raster, group <= 8: co-resident blocks share the A tile in L2.
    // gw CLAMPS to tn: with tn < 8 (small tail chunks) the unclamped form mapped blocks
    // to out-of-range bn tiles and left high-M tiles uncomputed (oracle: M=256/T=64 was
    // 0.71 rel-L2 with half of Out never written). Clamping keeps every (bm,bn) covered
    // exactly once for any tn; the L2 grouping just degrades on narrow T.
    const int tm = M / BM, tn = T / BN;
    // 2-D locality groups (expert review): 8 token-tiles x 4 out-tiles run as one
    // wave so BOTH operands' panels stay hot in L2 (token-only grouping streamed the
    // full 44 MB W per token-group). Group edges clamp to the tile grid.
    const int gt = (tn < 8) ? tn : 8, go = (tm < 4) ? tm : 4;
    const int gtm = (tm + go - 1) / go, gtn = (tn + gt - 1) / gt;   // groups along each axis
    const int per_g = gt * go;
    const int gid = blockIdx.x / per_g, rem = blockIdx.x % per_g;
    const int g_o = gid % gtm, g_t = gid / gtm;
    const int o0 = g_o * go, t0 = g_t * gt;                          // group origin (tiles)
    const int n_o = ((g_o + 1) * go < tm) ? go : tm - o0;            // clamped group sizes
    const int n_t = ((g_t + 1) * gt < tn) ? gt : tn - t0;
    const int li = rem / n_t, lj = rem % n_t;                        // out-fastest inside a group
    const int bm = (o0 + li) * BM, bn = (t0 + lj) * BN;
    const int mt0 = bm >> 4, bt0 = bn >> 3;

    extern __shared__ unsigned char psmem[];
    auto load_stage = [&](int s, int p) {
        unsigned char* st = psmem + s * STAGE;
        // A: 8 panels of 512 B (thread i -> panel i>>5, 16 B at lane i&31)
        {
            const int pi = tid >> 5, pl = tid & 31;
            if ((mt0 + pi) * 16 < M) {
                const unsigned* src = We + (((long long)(mt0 + pi) * nk32) + p) * 128 + (unsigned)pl * 4;
                cp16_b(st + pi * 512 + pl * 16, src);
            } else {
                *(uint4*)(st + pi * 512 + pl * 16) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        // B: 8 panels of 256 B (thread i -> panel i>>5, 8 B at lane i&31)
        {
            const int pi = tid >> 5, pl = tid & 31;
            if ((bt0 + pi) * 8 < T) {
                const unsigned* src = Bq + (((long long)(bt0 + pi) * nk32) + p) * 64 + (unsigned)pl * 2;
                cp8_b(st + STAGE_A + pi * 256 + pl * 8, src);
            } else {
                *(uint2*)(st + STAGE_A + pi * 256 + pl * 8) = make_uint2(0u, 0u);
            }
        }
        cp_commit_b();
    };

    load_stage(0, 0);
    if (nk32 > 1) load_stage(1, 1);
    if (nk32 > 2) load_stage(2, 2);
    if (nk32 > 3) load_stage(3, 3);

    float raw[2][4][4], acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int c = 0; c < 4; ++c) { raw[i][j][c] = 0.f; acc[i][j][c] = 0.f; }

    for (int p = 0; p < nk32; ++p) {
        const int s = p & (NSTAGE - 1);
        const int committed = (p + 3 < nk32) ? (p + 3) : nk32;
        const int need = committed - p - 1;
        if (need <= 0) cp_wait_b<0>();
        else if (need == 1) cp_wait_b<1>();
        else cp_wait_b<2>();
        __syncthreads();
        if (p + 3 < nk32) load_stage((p + 3) & (NSTAGE - 1), p + 3);

        const unsigned char* st = psmem + s * STAGE;
        unsigned a[2][4]; unsigned b[4][2];
        #pragma unroll
        for (int ma = 0; ma < 2; ++ma) {
            const uint4* q = (const uint4*)(st + ((wm * 2 + ma) * 128 + lane * 4) * 4);
            a[ma][0] = q->x; a[ma][1] = q->y; a[ma][2] = q->z; a[ma][3] = q->w;
        }
        #pragma unroll
        for (int na = 0; na < 4; ++na) {
            const uint2* q = (const uint2*)(st + STAGE_A + ((wn * 4 + na) * 32 + lane) * 8);
            b[na][0] = q->x; b[na][1] = q->y;
        }
        #pragma unroll
        for (int ma = 0; ma < 2; ++ma)
            #pragma unroll
            for (int na = 0; na < 4; ++na)
                mma_e4m3_k32(raw[ma][na], a[ma], b[na]);

        if ((p & 3) == 3) {                       // fold at the 128-K span boundary
            const int kspan = p >> 2;
            const float w_s = Ws[(long long)(bm >> 7) * nks + kspan];
            #pragma unroll
            for (int na = 0; na < 4; ++na) {
                const int tk0 = bn + wn * 32 + na * 8 + 2 * t;
                const float xs0 = Xs[(long long)tk0 * nks + kspan];
                const float xs1 = Xs[(long long)(tk0 + 1) * nks + kspan];
                #pragma unroll
                for (int ma = 0; ma < 2; ++ma)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const float xs = (c & 1) ? xs1 : xs0;
                        acc[ma][na][c] = fmaf(raw[ma][na][c], w_s * xs, acc[ma][na][c]);
                        raw[ma][na][c] = 0.f;
                    }
            }
        }
    }

    // smem-transpose epilogue (token-major Out, 8-row pad): reuse the dead stages.
    __syncthreads();
    __nv_bfloat16* tile = (__nv_bfloat16*)psmem;      // [BN=64][BM+8=136]
    constexpr int TS = 136;
    #pragma unroll
    for (int ma = 0; ma < 2; ++ma)
        #pragma unroll
        for (int na = 0; na < 4; ++na)
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int row = bm + wm * 32 + ma * 16 + g + 8 * (c >= 2);
                const int tok = bn + wn * 32 + na * 8 + 2 * t + (c & 1);
                tile[(tok - bn) * TS + (row - bm)] = f2b(acc[ma][na][c]);
            }
    __syncthreads();
    // [64 tok][128+8 pad] tile -> coalesced 16-B stores: 256 threads x 4 stores cover
    // (2 row-halves x 2 col-halves) — the ORIGINAL single-store version wrote only 1/4
    // of the tile (r3 0..31, c8 0..56), leaving 3/4 of Out stale. Caught by the
    // standalone oracle (tool_probe lineage) after repack/scales/pack all verified exact.
    const int r3 = tid >> 3, c8 = (tid & 7) * 8;
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            const int tr = r3 + rr * 32, tc = c8 + cc * 64;
            const int tok = bn + tr;
            if (tok < T) {
                const uint4* src = (const uint4*)(tile + tr * TS + tc);
                uint4* dst = (uint4*)(Out + (long long)tok * M + bm + tc);
                *dst = *src;
            }
        }
}


// =====================================================================================================
// ===================== qwen4_exp (Qwen3.8-Flash-Next): hyper-connections + PLE =====================
// =====================================================================================================
//
// The residual stream is `hc` copies of the hidden ("hyper-connections", rw = hc*h): every sublayer
// reads a gated MIX of the streams and writes back a per-stream-weighted INJECTION of its output.
// Reference: Qwen4ExpTextGatedResidual (modeling_qwen4_exp.py). Layout stays column-major [feat, B].
// The PLE layer (Qwen4ExpTextPLELayer) adds hashed n-gram embeddings into every stream once, at one
// trunk layer. All norms here are the qwen (1+w) RMSNorm, GROUPED per stream where noted.

// out[s*h+i, b] = in[i, b] for every stream s (the embedding replicated into the hc streams).
extern "C" __global__ void hc_expand_b(__nv_bfloat16* out, const __nv_bfloat16* in, int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int r = (int)(idx % rw);
    out[idx] = in[(long long)b * h + (r % h)];
}

// Grouped RMSNorm: one block per (b, s); rmsnorm over the s-th group of h, times (1 + w[s*h+i]).
extern "C" __global__ void hc_norm_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w,
                                     int h, int hc, int B, float eps) {
    int blk = blockIdx.x;
    if (blk >= B * hc) return;
    int b = blk / hc, s = blk % hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long off = (long long)b * ((long long)h * hc) + (long long)s * h;
    float sum_sq = 0.0f;
    for (int i = tid; i < h; i += bs) { float v = b2f(x[off + i]); sum_sq += v * v; }
    sm[tid] = sum_sq;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(sm[0] / (float)h + eps);
    for (int i = tid; i < h; i += bs) {
        float v = b2f(x[off + i]);
        out[off + i] = f2b(v * inv * (1.0f + w[s * h + i]));
    }
}

// x = silu(x / div), in place (the hc lowrank activation: silu(down(hn) / hc)).
extern "C" __global__ void silu_div_b(__nv_bfloat16* x, float div, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = f2b(silu_f(b2f(x[idx]) / div));
}

// The stream MIX + the injection weights, one block per column b:
//   x[i, b]   = (1/hc) * sum_s sigmoid(u[s*h+i, b]) * hn[s*h+i, b]
//   inj[s, b] = 2 * sigmoid( (sum_c winj[s, c] * hn[c, b]) / hc )     (winj bf16 [hc, rw]; skipped if null)
extern "C" __global__ void hc_mix_b(__nv_bfloat16* x, float* inj, const __nv_bfloat16* hn, const __nv_bfloat16* u,
                                    const __nv_bfloat16* winj, int h, int hc, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long rw = (long long)h * hc;
    const __nv_bfloat16* hb = hn + (long long)b * rw;
    const __nv_bfloat16* ub = u + (long long)b * rw;
    for (int i = tid; i < h; i += bs) {
        float acc = 0.0f;
        for (int s = 0; s < hc; s++) {
            long long o = (long long)s * h + i;
            float g = 1.0f / (1.0f + __expf(-b2f(ub[o])));
            acc += g * b2f(hb[o]);
        }
        x[(long long)b * h + i] = f2b(acc / (float)hc);
    }
    if (winj == nullptr || inj == nullptr) return;
    // injection logits: hc dot products over rw, block-reduced.
    for (int s = 0; s < hc; s++) {
        const __nv_bfloat16* ws = winj + (long long)s * rw;
        float part = 0.0f;
        for (long long c = tid; c < rw; c += bs) part += b2f(ws[c]) * b2f(hb[c]);
        sm[tid] = part;
        __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        if (tid == 0) inj[(long long)b * hc + s] = 2.0f / (1.0f + __expf(-(sm[0] / (float)hc)));
        __syncthreads();
    }
}

// resid[s*h+i, b] += out[i, b] * inj[s, b]
extern "C" __global__ void hc_inject_b(__nv_bfloat16* resid, const __nv_bfloat16* out, const float* inj,
                                       int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int r = (int)(idx % rw);
    int s = r / h, i = r % h;
    float v = b2f(resid[idx]) + b2f(out[(long long)b * h + i]) * inj[(long long)b * hc + s];
    resid[idx] = f2b(v);
}

// GDN output norm with a SIGMOID gate (qwen4_exp `output_gate_type: sigmoid`); twin of rmsnorm_gated_b.
extern "C" __global__ void rmsnorm_gated_sig_b(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z,
                                               const float* w, int vd, int nh, int B, float eps, int z_off_z_stride) {
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const int z_off = z_off_z_stride & 0x7FFF;
    const int z_stride = (unsigned)z_off_z_stride >> 15;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? b2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) {
        float g = b2f(z[z_off + (long long)b * z_stride + (long long)head * vd + tid]);
        out[base + tid] = f2b(v * inv * w[tid] * (1.0f / (1.0f + __expf(-g))));
    }
}

// ---- PLE: ancestry helpers ----
// Column t's k-th ancestor in a verify tree (`parent` packed: low 16 bits = DFS parent, 0xFFFF = root;
// null parent = chain t-1). Returns the column index, or -(m) where m >= 1 is how many steps
// remain past the root (those come from the slot's committed state / token ring).
// `chain`: with a null `parent`, 1 = columns form a chain (t-1), 0 = every column is its own root
// (a decode batch: independent lanes, each continuing from its own slot's state).
__device__ __forceinline__ int ple_ancestor(const int* parent, int chain, int t, int k) {
    int c = t;
    for (int i = 0; i < k; i++) {
        int p;
        if (parent) { int v = parent[c] & 0xFFFF; p = (v == 0xFFFF) ? -1 : v; }
        else p = chain ? (c - 1) : -1;
        if (p < 0) return -(k - i);
        c = p;
    }
    return c;
}
__device__ __forceinline__ int ple_slot_of(const int* slot_ids, int slot0, int t) { return slot_ids ? slot_ids[t] : slot0; }

// ---- PLE hash: row ids [heads, n] (row-major per column: ids[t*heads + j]) ----
// tab = [mult[ngram] | head_vocab[heads] | head_offset[heads]] (i64). ring = [slots, ngram-1] tokens
// oldest first (the committed history of the slot). EOS rule: the token p back is EOS if any token
// in the p positions before the current one is EOS (Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos).
// geom = ngram | (hpn << 8) | (heads << 16) | (chain << 24)   (cudarc launch tuples cap at 12 args)
extern "C" __global__ void ple_hash_b(long long* ids, const int* tokens, const int* ring, const int* slot_ids, int slot0,
                                      const int* parent, const long long* tab, int geom, int eos, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int ngram = geom & 0xFF, hpn = (geom >> 8) & 0xFF, heads = (geom >> 16) & 0xFF, chain = (geom >> 24) & 1;
    long long shifted[8];
    int slot = ple_slot_of(slot_ids, slot0, t);
    shifted[0] = tokens[t];
    bool blocked = false;
    for (int p = 1; p < ngram; p++) {
        int a = ple_ancestor(parent, chain, t, p);
        int tok;
        if (a >= 0) tok = tokens[a];
        else { int m = -a; tok = ring[(long long)slot * (ngram - 1) + (ngram - 1 - m)]; }   // m=1 → newest ring entry
        if (tok == eos) blocked = true;
        shifted[p] = blocked ? (long long)eos : (long long)tok;
    }
    const long long* mult = tab;
    const long long* hv = tab + ngram;
    const long long* ho = tab + ngram + heads;
    for (int ng = 2; ng <= ngram; ng++) {
        int start = (ng - 2) * hpn;
        long long mixed = shifted[0] * mult[0];
        for (int p = 1; p < ng; p++) mixed ^= shifted[p] * mult[p];
        for (int j = 0; j < hpn; j++) {
            int hh = start + j;
            long long r = mixed % hv[hh]; if (r < 0) r += hv[hh];
            ids[(long long)t * heads + hh] = r + ho[hh];
        }
    }
}

// Commit the token ring for slot(s): dst_ring[slot(dst)] = the last (ngram-1) tokens along column t's
// path (older ones from the source ring). tsel >= 0: only column tsel → dst slot dst_slot0; tsel < 0:
// every column t → dst slot dst_slot0 + t (per-column checkpoints). dst may alias src only when the
// written slot differs from every read slot (the caller guarantees it: main-slot commits go through
// a scratch slot).
// tsel >= 0: column tsel → dst slot dst_slot0. tsel == -1: column t → dst slot dst_slot0 + t (per-column
// checkpoints). tsel == -2: column t → ITS OWN slot (decode lanes, in place: read-all-then-write).
// One block per column, blockDim >= ngram-1.
extern "C" __global__ void ple_ring_commit_b(int* dst_ring, int dst_slot0, const int* src_ring, const int* tokens,
                                             const int* slot_ids, int slot0, const int* parent, int chain,
                                             int ngram, int n, int tsel) {
    int col = blockIdx.x;
    int t = (tsel >= 0) ? tsel : col;
    if (tsel < 0 && col >= n) return;
    if (tsel >= 0 && col > 0) return;
    int L = ngram - 1;
    int src_slot = ple_slot_of(slot_ids, slot0, t);
    int dst_slot = (tsel >= 0) ? dst_slot0 : (tsel == -1 ? dst_slot0 + t : src_slot);
    int l = threadIdx.x;
    int tok = 0;
    if (l < L) {
        int back = L - 1 - l;        // l = L-1 → the token itself (0 back)
        int a = ple_ancestor(parent, chain, t, back);
        tok = (a >= 0) ? tokens[a] : src_ring[(long long)src_slot * L + (L - (-a))];
    }
    __syncthreads();
    if (l < L) dst_ring[(long long)dst_slot * L + l] = tok;
}

// ---- PLE table records: gather (device-resident table) and dequant ----
// stage[r*96 ..] = table[ids[r]*96 ..], 24 u32 words per record.
extern "C" __global__ void ple_gather_rows_b(unsigned char* stage, const unsigned char* table, const long long* ids, int nrec) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nrec * 24) return;
    long long r = idx / 24; int w = (int)(idx % 24);
    const unsigned int* src = (const unsigned int*)(table + ids[r] * 96);
    unsigned int* dst = (unsigned int*)(stage + r * 96);
    dst[w] = src[w];
}

__device__ __forceinline__ float ple_e4m3(unsigned char b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0x0F;
    float man = (float)(b & 0x07);
    if (exp == 0) return sign * (man / 8.0f) * 0.015625f;          // 2^-6
    return sign * (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
}
__constant__ float PLE_E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// emb[(head*160 + blk*16 + i), col] for record r = col*heads + head; one thread per (record, 16-block).
// gs[shard] is the RECIPROCAL global scale of the record's source shard: w = e2m1 * e4m3 / gs.
extern "C" __global__ void ple_dequant_rows_b(__nv_bfloat16* emb, const unsigned char* stage, const long long* ids,
                                              const float* gs, int rows_per_shard, int heads, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nrec = (long long)n * heads;
    if (idx >= nrec * 10) return;
    long long r = idx / 10; int blk = (int)(idx % 10);
    int col = (int)(r / heads), head = (int)(r % heads);
    const unsigned char* rec = stage + r * 96;
    float st = 1.0f / gs[(int)(ids[r] / rows_per_shard)];
    float s = ple_e4m3(rec[80 + blk]) * st;
    __nv_bfloat16* o = emb + (long long)col * (heads * 160) + (long long)head * 160 + blk * 16;
    for (int i = 0; i < 16; i++) {
        int e = blk * 16 + i;
        unsigned char byte = rec[e >> 1];
        unsigned char code = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        float v = PLE_E2M1[code & 7];
        if (code & 8) v = -v;
        o[i] = f2b(v * s);
    }
}

// ---- PLE gate: gv[s*h+i, b] = sigmoid(gate[s,b]) * value[i, b] ----
//   gate = (kn[s] . qn[s]) / sqrt(h);  gate = sign(gate) * sqrt(max(|gate|, 1e-6))
// one block per (b, s).
extern "C" __global__ void ple_gate_b(__nv_bfloat16* gv, const __nv_bfloat16* kn, const __nv_bfloat16* qn,
                                      const __nv_bfloat16* value, int h, int hc, int B) {
    int blk = blockIdx.x;
    if (blk >= B * hc) return;
    int b = blk / hc, s = blk % hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long off = (long long)b * ((long long)h * hc) + (long long)s * h;
    float part = 0.0f;
    for (int i = tid; i < h; i += bs) part += b2f(kn[off + i]) * b2f(qn[off + i]);
    sm[tid] = part;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float g = sm[0] * rsqrtf((float)h);
    float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    g = (g < 0.0f) ? -mag : mag;
    float sg = 1.0f / (1.0f + __expf(-g));
    for (int i = tid; i < h; i += bs) gv[off + i] = f2b(sg * b2f(value[(long long)b * h + i]));
}

// ---- PLE dilated depthwise causal conv (kernel K, dilation dil, state length L = (K-1)*dil) ----
// The conv input is gvn (the norm_conv output); its output is silu(conv) ADDED to gv, and the sum is
// the PLE output ADDED to the residual: resid += gv + silu(conv(gvn)).
// state[slot][l][c], l = 0 oldest ... L-1 newest (the last L conv inputs of the slot).
// DECODE step (one token per column, its own slot): taps come from the state; then the state shifts
// and takes gvn as its newest row.
extern "C" __global__ void ple_dconv_decode_b(__nv_bfloat16* resid, const __nv_bfloat16* gv, const __nv_bfloat16* gvn,
                                              float* state, const float* w, const int* slot_ids,
                                              int rw, int L, int K, int dil, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * B) return;
    int b = (int)(idx / rw);
    int c = (int)(idx % rw);
    int slot = slot_ids[b];
    float* st = state + (long long)slot * L * rw;
    float x = b2f(gvn[idx]);
    float acc = w[c * K + (K - 1)] * x;
    for (int j = 0; j < K - 1; j++) {
        int back = (K - 1 - j) * dil;          // positions back
        acc += w[c * K + j] * st[(long long)(L - back) * rw + c];
    }
    for (int l = 1; l < L; l++) st[(long long)(l - 1) * rw + c] = st[(long long)l * rw + c];
    st[(long long)(L - 1) * rw + c] = x;
    float v = b2f(resid[idx]) + b2f(gv[idx]) + silu_f(acc);
    resid[idx] = f2b(v);
}

// PREFILL / VERIFY (n columns along a chain or tree): taps at `back` positions come from the
// ancestor columns, or from the slot's committed state past the root. Pure read of `state` — the
// commit is a separate launch (ple_dconv_state_b) so nothing races.
// geom = L | (K << 8) | (dil << 16) | (chain << 24)
extern "C" __global__ void ple_dconv_prefill_b(__nv_bfloat16* resid, const __nv_bfloat16* gv, const __nv_bfloat16* gvn,
                                               const float* state, const int* slot_ids, int slot0, const int* parent,
                                               const float* w, int rw, int geom, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * n) return;
    const int L = geom & 0xFF, K = (geom >> 8) & 0xFF, dil = (geom >> 16) & 0xFF, chain = (geom >> 24) & 1;
    int t = (int)(idx / rw);
    int c = (int)(idx % rw);
    int slot = ple_slot_of(slot_ids, slot0, t);
    const float* st = state + (long long)slot * L * rw;
    float acc = w[c * K + (K - 1)] * b2f(gvn[idx]);
    for (int j = 0; j < K - 1; j++) {
        int back = (K - 1 - j) * dil;
        int a = ple_ancestor(parent, chain, t, back);
        float xv = (a >= 0) ? b2f(gvn[(long long)a * rw + c]) : st[(long long)(L - (-a)) * rw + c];
        acc += w[c * K + j] * xv;
    }
    float v = b2f(resid[idx]) + b2f(gv[idx]) + silu_f(acc);
    resid[idx] = f2b(v);
}

// State after column t: rows l = 0..L-1 hold the conv input (L-1-l) positions back from t (ancestor
// columns, else the SOURCE slot's state). tsel >= 0: column tsel → dst slot dst_slot0 (blockIdx.y
// unused); tsel < 0: column t = blockIdx.y → dst slot dst_slot0 + t (per-column checkpoints).
// dst must not alias the source rows being read (main-slot commits go through a scratch slot).
extern "C" __global__ void ple_dconv_state_b(float* dst, int dst_slot0, const float* src, const int* slot_ids, int slot0,
                                             const int* parent, int chain, const __nv_bfloat16* gvn, int rw, int L, int n, int tsel) {
    int t = (tsel >= 0) ? tsel : (int)blockIdx.y;
    if (t >= n) return;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * L) return;
    int l = (int)(idx / rw);
    int c = (int)(idx % rw);
    int dst_slot = (tsel >= 0) ? dst_slot0 : dst_slot0 + t;
    int src_slot = ple_slot_of(slot_ids, slot0, t);
    int back = L - 1 - l;
    int a = ple_ancestor(parent, chain, t, back);
    float v = (a >= 0) ? b2f(gvn[(long long)a * rw + c]) : src[((long long)src_slot * L + (L - (-a))) * rw + c];
    dst[((long long)dst_slot * L + l) * rw + c] = v;
}

// Byte copy of one PLE state slot + ring entry (rollback / snapshot), same contract as gdn_rollback_b.
// dst_ids != null: the destination slot is dst_ids[dst_idx] (read on device — capture-legal).
extern "C" __global__ void ple_slot_copy_b(float* state, int* ring, unsigned long long state_floats, int ring_ints, int src, int dst,
                                           const int* dst_ids, int dst_idx) {
    if (dst_ids) dst = dst_ids[dst_idx];
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (long long)state_floats) {
        state[(long long)dst * state_floats + idx] = state[(long long)src * state_floats + idx];
    }
    if (idx < ring_ints) ring[(long long)dst * ring_ints + idx] = ring[(long long)src * ring_ints + idx];
}

// qwen4_exp MTP input fusion: streams[s*h+i, b] += x[i, b] (the fc_embedding term broadcast to every stream).
extern "C" __global__ void hc_add_bcast_b(__nv_bfloat16* streams, const __nv_bfloat16* x, int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int i = (int)((idx % rw) % h);
    streams[idx] = f2b(b2f(streams[idx]) + b2f(x[(long long)b * h + i]));
}

