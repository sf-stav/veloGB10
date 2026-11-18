// exl3_bench/22_dsplitk_ep.cuh: DSPLITK round 3 q8-cache split-K attention and the expert-parallel MoE route/combine kernels.
// K/V rows are decoded once per block with xq_kv_ld8 (the served reader).
// Section 22 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- DSPLITK round 3 (owner 2026-11-08): the q8 cache (the served default) was decoded by EVERY warp of a block —
// gqa (6 on the 27B, 8 on the 35B) x the same unpack-and-scale per K/V value — which made the trunk's split-K
// issue-bound at ~2x its memory floor (27B q8 at 48K: d1 +26 ms/round over short vs a ~14 ms two-row floor).
// Here a block decodes each staged K/V row ONCE with xq_kv_ld8 (the served reader, so the values are the same bits)
// into an f32 tile, [row j][K|V][half][lane]: dims lane*8+0..3 in half 0 and lane*8+4..7 in half 1, so both the
// writes (one warp per (row, K|V)) and every warp's two lds.128 per row are conflict-free. The step after it is
// xq_w4d_step unchanged -> per (row, head, split) the same keys ascending, the same values and the same ops as
// xq_attn_dense_splitk4_w4 -> bitwise. Needs blockDim == hd == 256 (32 lanes x 8 dims; the host geometry check).
__device__ __forceinline__ void xq_dskt_decode(float4 (*xt)[2][2][32], char (*ring)[2][1024], int s0, int tn,
                                               int fmt, int hd, int tid) {
    const int j = tid >> 6, kvsel = (tid >> 5) & 1, c = tid & 31;
    if (j < tn) {
        float o[8];
        xq_kv_ld8(o, ring[s0 + j][kvsel], fmt, hd, c * 8);
        xt[j][kvsel][0][c] = make_float4(o[0], o[1], o[2], o[3]);
        xt[j][kvsel][1][c] = make_float4(o[4], o[5], o[6], o[7]);
    }
}
__device__ __forceinline__ void xq_dskt_read(float (&kv)[8], float (&vv)[8], float4 (*xt)[2][2][32], int j, int lane) {
    const float4 k0 = xt[j][0][0][lane], k1 = xt[j][0][1][lane];
    const float4 v0 = xt[j][1][0][lane], v1 = xt[j][1][1][lane];
    kv[0] = k0.x; kv[1] = k0.y; kv[2] = k0.z; kv[3] = k0.w; kv[4] = k1.x; kv[5] = k1.y; kv[6] = k1.z; kv[7] = k1.w;
    vv[0] = v0.x; vv[1] = v0.y; vv[2] = v0.z; vv[3] = v0.w; vv[4] = v1.x; vv[5] = v1.y; vv[6] = v1.z; vv[7] = v1.w;
}
// The trunk-only twin of xq_attn_dense_splitk4_w4 (that kernel also serves the draft head on every EXL3 pack,
// Flash-Next included, and stays byte-for-byte as it is), for q8 / fp8 caches (row = 272 B at hd 256):
//  * the group (4 keys) is decoded ONCE per block into the f32 tile (xq_dskt_decode / xq_dskt_read);
//  * a DEEP ring: XQ_T4_NG groups of raw 272-B rows (13 KB), so XQ_T4_NG - 1..XQ_T4_NG groups are in flight while one
//    computes (the 2-group ring of _w4 kept ONE group in flight: a 35B q8 48K split-K call ran at ~1.5x its DRAM floor
//    and the 2-row shared-read verify at ~3.7x, the nsys receipt in PLAN/DSPLITK round 3);
//  * the grid is ROW-FASTEST (blockIdx = ((split * nkv) + kvh) * rows + row): the rows of one verify walk the same
//    split's keys at the same time, so the second..m-th reader of each K/V line is an L2 hit, without the register-
//    heavy shared-read body. A decode (rows = 1) is the same order as before.
// Per (row, head, split): the same keys ascending, the same values, the same xq_w4d_step ops -> bitwise == _w4.
#define XQ_T4_NG 6
#define XQ_T4_RB 272
__device__ __forceinline__ void xq_t4_stage(char (*ring)[4][2][XQ_T4_RB], int slot, const XqKv& kc, int t0, int end, int tid) {
    constexpr int NC = XQ_T4_RB / 16;                // 17 chunks per row
    for (int i = tid; i < 4 * 2 * NC; i += blockDim.x) {
        const int j = i / (2 * NC), r = i - j * (2 * NC);
        const int t = t0 + j;
        if (t < end) {
            const int kvsel = r >= NC ? 1 : 0;
            const int c = r - kvsel * NC;
            const char* src = (kvsel ? kc.v : kc.k) + (long long)t * XQ_T4_RB + c * 16;
            moe_coop_cp16(&ring[slot][j][kvsel][c * 16], src, 16);
        }
    }
}
__device__ __forceinline__ void xq_t4_decode(float4 (*xt)[2][2][32], char (*grp)[2][XQ_T4_RB], int tn, int fmt, int hd, int tid) {
    const int j = tid >> 6, kvsel = (tid >> 5) & 1, c = tid & 31;
    if (j < tn) {
        float o[8];
        xq_kv_ld8(o, grp[j][kvsel], fmt, hd, c * 8);
        xt[j][kvsel][0][c] = make_float4(o[0], o[1], o[2], o[3]);
        xt[j][kvsel][1][c] = make_float4(o[4], o[5], o[6], o[7]);
    }
}
// A full group of 4 keys for one head: the 4 dots (each xq_w4d_step's FFMA chain from +0, dims lane*8+i ascending)
// and their shfl_xor 16..1 FADD trees INTERLEAVED (one position's reduction no longer waits on the previous one's),
// then the 4 online-softmax updates IN KEY ORDER with xq_w4d_step's exact ops -> bitwise == 4 xq_w4d_step calls.
__device__ __forceinline__ void xq_t4_step4(const float (&q)[8], float4 (*xt)[2][2][32], int lane, float qs,
                                            float& m, float& l, float (&acc)[8]) {
    float d[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float4 k0 = xt[j][0][0][lane], k1 = xt[j][0][1][lane];
        const float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
        float x = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) x = __fmaf_rn(q[i], kv[i], x);
        d[j] = x;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int j = 0; j < 4; j++) d[j] = __fadd_rn(d[j], __shfl_xor_sync(0xffffffffu, d[j], off));
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float4 v0 = xt[j][1][0][lane], v1 = xt[j][1][1][lane];
        const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        const float sc = __fmul_rn(d[j], qs);
        const float m_new = fmaxf(m, sc);
        const float scale_f = __expf(m - m_new);
        const float wt = __expf(sc - m_new);
        l = __fmaf_rn(l, scale_f, wt);
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = __fmaf_rn(acc[i], scale_f, __fmul_rn(wt, vv[i]));
        m = m_new;
    }
}
extern "C" __global__ void __launch_bounds__(256, 3)
xq_attn_dense_splitk4_t4(float* __restrict__ pm, float* __restrict__ pl,
                         float* __restrict__ pacc, const float* __restrict__ qstage,
                         const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                         int nh_packed, long long geom, int rows) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) char ring[XQ_T4_NG][4][2][XQ_T4_RB];
    __shared__ float4 xt[4][2][2][32];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int row = blockIdx.x % rows;                               // row-fastest
    const int kvh = (blockIdx.x / rows) % nkv;
    const int sp = blockIdx.x / (rows * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int split_len = (npos + nsplits - 1) / nsplits;
    const int begin = sp * split_len;
    const int end = min(begin + split_len, npos);
    const bool hasA = warp < gqa, hasB = warp + 8 < gqa;
    const int hA = kvh * gqa + warp, hB = kvh * gqa + warp + 8;
    float qA[8], qB[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { qA[i] = 0.0f; qB[i] = 0.0f; }
    if (hasA) {
        const float* qrow = qstage + ((long long)row * nh + hA) * hd;
        *reinterpret_cast<float4*>(qA) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qA + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    if (hasB) {
        const float* qrow = qstage + ((long long)row * nh + hB) * hd;
        *reinterpret_cast<float4*>(qB) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qB + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
    }
    const float qs = rsqrtf((float)hd);
    float mA = -INFINITY, lA = 0.0f, mB = -INFINITY, lB = 0.0f;
    float accA[8], accB[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { accA[i] = 0.0f; accB[i] = 0.0f; }
    const int n = end > begin ? end - begin : 0;
    const int ng = (n + 3) >> 2;
    #pragma unroll
    for (int q = 0; q < XQ_T4_NG; q++) {                             // prologue: XQ_T4_NG groups in flight
        if (q < ng) xq_t4_stage(ring, q, kc, begin + 4 * q, end, threadIdx.x);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    for (int g = 0; g < ng; g++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(XQ_T4_NG - 1) : "memory");   // group g (this thread)
        __syncthreads();                                              // ... and everyone's; the tile's readers are done
        const int t0 = begin + 4 * g;
        const int tn = min(4, end - t0);
        const int sl = g % XQ_T4_NG;
        xq_t4_decode(xt, ring[sl], tn, kc.fmt, hd, threadIdx.x);
        __syncthreads();                                              // tile ready; ring slot sl free
        if (g + XQ_T4_NG < ng) xq_t4_stage(ring, sl, kc, begin + 4 * (g + XQ_T4_NG), end, threadIdx.x);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (tn == 4) {                                                // a full group: 4 positions in flight
            if (hasA) xq_t4_step4(qA, xt, lane, qs, mA, lA, accA);
            if (hasB) xq_t4_step4(qB, xt, lane, qs, mB, lB, accB);
        } else {
            for (int j = 0; j < tn; j++) {
                float kv[8], vv[8];
                xq_dskt_read(kv, vv, xt, j, lane);
                if (hasA) xq_w4d_step(qA, kv, vv, qs, mA, lA, accA);
                if (hasB) xq_w4d_step(qB, kv, vv, qs, mB, lB, accB);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    if (hasA) {
        const long long hg = (long long)row * nh + hA;
        if (lane == 0) { pm[hg * nsplits + sp] = mA; pl[hg * nsplits + sp] = lA; }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = accA[i];
    }
    if (hasB) {
        const long long hg = (long long)row * nh + hB;
        if (lane == 0) { pm[hg * nsplits + sp] = mB; pl[hg * nsplits + sp] = lB; }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = accB[i];
    }
}
// ---- DSPLITK round 2b (owner 2026-11-07): xq_attn_dense_splitk4_vr on xq_attn_dense_splitk4_w4's machinery —
// the rows of ONE verify (XQ_VR_ROWS per block) walk the UNION of their slice ranges once through the cp.async
// smem ring (xq_w4d_stage, groups of 4 keys), and each staged K/V row is folded into every row whose own range
// holds it with xq_w4d_step (the pinned op sequence of xq_attn_dense_splitk4). Per (row, head, split): the same
// keys ascending and the same ops as _w4 / the base kernel -> bitwise. Heads w and w+8 (gqa > 8) take a second
// walk. Grid ceil(m / XQ_VR_ROWS) x nkv x nsplits, block 256 (host: rb % 16 == 0, rb <= 1024).
extern "C" __global__ void __launch_bounds__(256)
xq_attn_dense_splitk4_vr_w4(float* __restrict__ pm, float* __restrict__ pl,
                            float* __restrict__ pacc, const float* __restrict__ qstage,
                            const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                            int nh_packed, long long geom, int m) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) char ring[XQ_W4D_NR][2][1024];
    __shared__ float4 xt[4][2][2][32];                        // round 3: the decoded group (q8 / fp8 only)
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int r0 = (blockIdx.x / (nsplits * nkv)) * XQ_VR_ROWS;
    const int nrows = min(XQ_VR_ROWS, m - r0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;
    const int slot = slotpos[r0 * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const bool tile = kc.fmt != XQ_KV_F16 && kc.fmt != XQ_KV_F32;   // block-uniform
    int rb[XQ_VR_ROWS], re[XQ_VR_ROWS];
    int ub = 0x7fffffff, ue = 0;
    #pragma unroll
    for (int r = 0; r < XQ_VR_ROWS; r++) {
        rb[r] = 0; re[r] = 0;
        if (r < nrows) {
            const int npos = slotpos[(r0 + r) * 2 + 1] + 1;
            const int split_len = (npos + nsplits - 1) / nsplits;
            rb[r] = sp * split_len;
            re[r] = min(rb[r] + split_len, npos);
            if (rb[r] < re[r]) { ub = min(ub, rb[r]); ue = max(ue, re[r]); }
        }
    }
    if (ue <= ub) { ub = 0; ue = 0; }   // no row has a key in this slice: an EMPTY walk (the sentinel must never be staged:
                                        // xq_w4d_stage's t = ub + j would overflow INT_MAX into a negative key index)
    const float qs = rsqrtf((float)hd);
    const int npass = gqa > 8 ? 2 : 1;
    for (int pass = 0; pass < npass; pass++) {
        const int w = warp + pass * 8;
        const bool has = w < gqa;
        const int h = kvh * gqa + w;
        float qv[XQ_VR_ROWS][8];
        float mr[XQ_VR_ROWS], lr[XQ_VR_ROWS], acc[XQ_VR_ROWS][8];
        #pragma unroll
        for (int r = 0; r < XQ_VR_ROWS; r++) {
            mr[r] = -INFINITY; lr[r] = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) { acc[r][i] = 0.0f; qv[r][i] = 0.0f; }
            if (has) {
                const float* qrow = qstage + ((long long)(r0 + min(r, nrows - 1)) * nh + h) * hd;
                *reinterpret_cast<float4*>(qv[r]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
                *reinterpret_cast<float4*>(qv[r] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
            }
        }
        const int n = ue > ub ? ue - ub : 0;
        const int ng = (n + 3) >> 2;
        __syncthreads();                                              // the ring is free (previous pass)
        xq_w4d_stage(ring, kc, ub, ue, 0, threadIdx.x);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (ng > 1) xq_w4d_stage(ring, kc, ub + 4, ue, 4, threadIdx.x);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        for (int g = 0; g < ng; g++) {
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            __syncthreads();                                          // group g landed; the tile's readers are done
            const int t0 = ub + 4 * g;
            const int tn = min(4, ue - t0);
            const int s0 = (g & 1) * 4;
            if (tile) {                                               // round 3: decode the group ONCE per block
                xq_dskt_decode(xt, ring, s0, tn, kc.fmt, hd, threadIdx.x);
                __syncthreads();                                      // tile ready; ring slots s0.. free
                if (g + 2 < ng) xq_w4d_stage(ring, kc, ub + 4 * (g + 2), ue, s0, threadIdx.x);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            bool all4 = tile && tn == 4;                              // a full group inside every live row's range:
            #pragma unroll                                            // xq_t4_step4 per row (bitwise == 4 steps)
            for (int r = 0; r < XQ_VR_ROWS; r++)
                if (r < nrows && !(t0 >= rb[r] && t0 + 3 < re[r])) all4 = false;
            if (all4) {
                if (has) {
                    #pragma unroll
                    for (int r = 0; r < XQ_VR_ROWS; r++)
                        if (r < nrows) xq_t4_step4(qv[r], xt, lane, qs, mr[r], lr[r], acc[r]);
                }
            } else
            for (int j = 0; j < tn; j++) {
                const int t = t0 + j;
                float kv[8], vv[8];
                if (tile) {
                    xq_dskt_read(kv, vv, xt, j, lane);
                } else {
                    xq_kv_ld8(kv, ring[s0 + j][0], kc.fmt, hd, lane * 8);
                    xq_kv_ld8(vv, ring[s0 + j][1], kc.fmt, hd, lane * 8);
                }
                if (has) {
                    #pragma unroll
                    for (int r = 0; r < XQ_VR_ROWS; r++)
                        if (t >= rb[r] && t < re[r]) xq_w4d_step(qv[r], kv, vv, qs, mr[r], lr[r], acc[r]);
                }
            }
            if (!tile) {
                __syncthreads();
                if (g + 2 < ng) xq_w4d_stage(ring, kc, ub + 4 * (g + 2), ue, s0, threadIdx.x);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        if (has) {
            #pragma unroll
            for (int r = 0; r < XQ_VR_ROWS; r++) {
                if (r >= nrows) break;
                const long long hg = (long long)(r0 + r) * nh + h;
                if (lane == 0) { pm[hg * nsplits + sp] = mr[r]; pl[hg * nsplits + sp] = lr[r]; }
                #pragma unroll
                for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[r][i];
            }
        }
    }
}
#define XQ_W4C_MAXS 256
extern "C" __global__ void __launch_bounds__(256)
xq_attn_sel_combine_w4(__half* __restrict__ attn, const float* __restrict__ pm,
                       const float* __restrict__ pl, const float* __restrict__ pacc,
                       const __half* __restrict__ qg, int nh_packed,
                       int nsplits, int rows) {
    XQ_PDL_ENTRY();
    __shared__ float s_pm[XQ_W4C_MAXS], s_pl[XQ_W4C_MAXS], s_w[XQ_W4C_MAXS];
    const int kvf = xq_kv_fmt(rows);
    const int nh = nh_packed & 0xFFFF;
    const int hd = blockDim.x;
    const int b = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int tid = threadIdx.x;
    const long long hg = (long long)b * nh + h;
    for (int s = tid; s < nsplits; s += blockDim.x) { s_pm[s] = pm[hg * nsplits + s]; s_pl[s] = pl[hg * nsplits + s]; }
    __syncthreads();
    float m_g = -INFINITY;
    for (int s = 0; s < nsplits; s++) m_g = fmaxf(m_g, s_pm[s]);
    for (int s = tid; s < nsplits; s += blockDim.x) s_w[s] = __expf(s_pm[s] - m_g);   // read only where pm > -inf
    __syncthreads();
    float l_g = 0.0f;
    for (int s = 0; s < nsplits; s++)
        if (s_pm[s] > -INFINITY) l_g = __fmaf_rn(s_pl[s], s_w[s], l_g);
    float acc = 0.0f;
    #pragma unroll 16
    for (int s = 0; s < nsplits; s++) {
        if (s_pm[s] > -INFINITY) acc = __fmaf_rn(pacc[(hg * nsplits + s) * hd + tid], s_w[s], acc);
    }
    const float g = xq_h2f(qg[(long long)b * nh * hd * 2 + (long long)h * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / l_g, kvf, tid & 31);
    attn[(long long)b * nh * hd + (long long)h * hd + tid] = xq_f2h(o * xq_sig(g));
}

// =============================================================================
// A5-P2 (prefill hc mixers; PLAN/notes_2026-09-27/P2.md). The prefill hc elementwise chain —
// xq_hc_inject (hc_post) + xq_hc_norm (next hc_pre) + xq_hc_mix4 — ran as 1024-thread one-block-
// per-(row, stream) / one-block-per-row kernels (1 CTA/SM, 10-level smem trees): norm at ~58 % and
// mix4 at ~56 % of DRAM bandwidth, plus a full extra read of the 84 MB f32 stream state per
// sublayer for the separate inject. BITWISE twins, 128-thread blocks (up to 12 CTAs/SM):
//   xq_pf_hc_inj_norm == xq_hc_inject (optional; y == null skips it) + xq_hc_norm
//   xq_pf_hc_mix_2560x4 == xq_hc_mix4 (== xq_hc_mix) at h = 2560, hc = 4
// The contract is the 1024-leaf reduction of the old kernels: leaf vt (= old threadIdx.x) sums its
// elements i = vt + 1024 j in ascending j with the same `acc += a * b` (one fma chain), then the
// same halving tree (level s2: leaf t += leaf t + s2). Here thread T (0..127) OWNS leaves
// A = 4T..4T+3 and B = 512+4T..512+4T+3, i.e. every element i = 512 k + 4T + e (k even -> A,
// j = k/2; k odd -> B, j = (k-1)/2): so the leaf chains stay whole in one thread (vector loads),
// and tree level 512 (A + B) is in-thread; 256 via smem, 128/64/32 in-lane (lane l holds positions
// l + 32 q), 16..1 by shfl_down in one warp (lane t gets v[t] + v[t+s2], exactly sm[t] += sm[t+s2]).
// Every per-element expression is written as in the old kernels (same fma contraction points).
// Requires h % 512 == 0 and h <= 4096 (launcher-checked; else the old kernels run).

__device__ __forceinline__ void xq_pf_unpack4(const uint2 v, float f[4]) {
    const __half2 a = *reinterpret_cast<const __half2*>(&v.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&v.y);
    f[0] = xq_h2f(__low2half(a)); f[1] = xq_h2f(__high2half(a));
    f[2] = xq_h2f(__low2half(b)); f[3] = xq_h2f(__high2half(b));
}
__device__ __forceinline__ uint2 xq_pf_pack4(const float f[4]) {
    __half2 a = __halves2half2(xq_f2h(f[0]), xq_f2h(f[1]));
    __half2 b = __halves2half2(xq_f2h(f[2]), xq_f2h(f[3]));
    uint2 r;
    r.x = *reinterpret_cast<const unsigned int*>(&a);
    r.y = *reinterpret_cast<const unsigned int*>(&b);
    return r;
}

// One warp finishes a 1024-leaf halving tree whose level 512 is already in sm[0..512)
// (sm[t] = leaf t + leaf t+512). Returns the root in lane 0.
__device__ __forceinline__ float xq_pf_tree512(const float* sm, int lane) {
    float a[8];
    #pragma unroll
    for (int q = 0; q < 8; ++q) a[q] = sm[lane + 32 * q] + sm[lane + 32 * q + 256];   // level 256
    #pragma unroll
    for (int q = 0; q < 4; ++q) a[q] += a[q + 4];                                     // level 128
    #pragma unroll
    for (int q = 0; q < 2; ++q) a[q] += a[q + 2];                                     // level 64
    a[0] += a[1];                                                                     // level 32
    float v = a[0];
    #pragma unroll
    for (int s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);   // 16..1
    return v;
}

// Fused inject + norm over one (row b, stream s) slab per CTA; grid hc * B, block 128.
// out = hn (f16), x = resid (f32, updated in place when y != null), y = the sublayer output (f16
// [B][h]) or null, inj = [B][hc] gates, w = hc_norm weight [hc][h].
extern "C" __global__ void __launch_bounds__(128)
xq_pf_hc_inj_norm(__half* __restrict__ out, float* __restrict__ x, const __half* __restrict__ y,
                  const float* __restrict__ inj, const __half* __restrict__ w, int h, int hc, int B, float eps) {
    XQ_PDL_ENTRY();
    const int blk = blockIdx.x;
    if (blk >= hc * B) return;
    const int b = blk / hc, s = blk - b * hc;
    __shared__ float sm[512];
    __shared__ float s_inv;
    const int T = threadIdx.x;
    float* xs = x + (long long)blk * h;                  // == b*h*hc + s*h
    const bool doinj = (y != nullptr);
    const __half* yr = doinj ? y + (long long)b * h : y;
    const float g = doinj ? inj[b * hc + s] : 0.0f;
    const int nk = h >> 9;
    constexpr int KM = 8;
    float v[KM][4];
    float pa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, pb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float4 t4 = *reinterpret_cast<const float4*>(xs + i);
            float t[4] = {t4.x, t4.y, t4.z, t4.w};
            if (doinj) {
                float yf[4];
                xq_pf_unpack4(*reinterpret_cast<const uint2*>(yr + i), yf);
                #pragma unroll
                for (int e = 0; e < 4; ++e) t[e] += g * yf[e];          // xq_hc_inject's fma
                *reinterpret_cast<float4*>(xs + i) = make_float4(t[0], t[1], t[2], t[3]);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[k][e] = t[e];
                if (k & 1) pb[e] += t[e] * t[e]; else pa[e] += t[e] * t[e];   // xq_hc_norm's sum_sq
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < 4; ++e) sm[4 * T + e] = pa[e] + pb[e];              // level 512
    __syncthreads();
    if (T < 32) {
        const float r = xq_pf_tree512(sm, T);
        if (T == 0) s_inv = rsqrtf(r / (float)h + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const __half* ws = w + (long long)s * h;
    __half* os = out + (long long)blk * h;
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float wf[4], o[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ws + i), wf);
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = v[k][e] * inv * (1.0f + wf[e]);
            *reinterpret_cast<uint2*>(os + i) = xq_pf_pack4(o);
        }
    }
}

// hc mix + inject gates, one row per CTA; grid B, block 128 (4 warps = the HC tree warps).
//   x[i]   = f16((sum_s sig(u[s*H+i]) * hn[s*H+i]) / HC)            (ascending s: k = s*NK + m)
//   inj[t] = 2 * sig((sum_c winj[t,c] * hn[c]) / HC)                 (1024-leaf tree, as xq_hc_mix4)
template <int NK, int HC>
__device__ __forceinline__ void xq_pf_hc_mix_body(__half* __restrict__ x, float* __restrict__ inj,
                                                  const __half* __restrict__ hn, const __half* __restrict__ u,
                                                  const __half* __restrict__ winj, int B) {
    constexpr int H = NK * 512, N = H * HC, KT = NK * HC;
    const int b = blockIdx.x;
    if (b >= B) return;
    hn += (long long)b * N;
    u += (long long)b * N;
    x += (long long)b * H;
    inj += (long long)b * HC;
    const int T = threadIdx.x;
    const bool dj = (winj != nullptr);
    float acc[NK][4];
    float pa[HC][4], pb[HC][4];
    #pragma unroll
    for (int m = 0; m < NK; ++m)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[m][e] = 0.0f;
    #pragma unroll
    for (int t = 0; t < HC; ++t)
        #pragma unroll
        for (int e = 0; e < 4; ++e) { pa[t][e] = 0.0f; pb[t][e] = 0.0f; }
    #pragma unroll
    for (int k = 0; k < KT; ++k) {
        const int c = 512 * k + 4 * T;
        float hf[4], uf[4];
        xq_pf_unpack4(*reinterpret_cast<const uint2*>(hn + c), hf);
        xq_pf_unpack4(*reinterpret_cast<const uint2*>(u + c), uf);
        const int m = k % NK;
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[m][e] += xq_sig(uf[e]) * hf[e];
        if (dj) {
            #pragma unroll
            for (int t = 0; t < HC; ++t) {
                float wf[4];
                xq_pf_unpack4(*reinterpret_cast<const uint2*>(winj + (long long)t * N + c), wf);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (k & 1) pb[t][e] += wf[e] * hf[e]; else pa[t][e] += wf[e] * hf[e];
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < NK; ++m) {
        float o[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = acc[m][e] / (float)HC;
        *reinterpret_cast<uint2*>(x + 512 * m + 4 * T) = xq_pf_pack4(o);
    }
    if (dj) {
        __shared__ float sm[HC][512];
        #pragma unroll
        for (int t = 0; t < HC; ++t)
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm[t][4 * T + e] = pa[t][e] + pb[t][e];   // level 512
        __syncthreads();
        const int wp = T >> 5, lane = T & 31;
        if (wp < HC) {
            const float r = xq_pf_tree512(sm[wp], lane);
            if (lane == 0) inj[wp] = 2.0f * xq_sig(r / (float)HC);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
xq_pf_hc_mix_2560x4(__half* __restrict__ x, float* __restrict__ inj, const __half* __restrict__ hn,
                    const __half* __restrict__ u, const __half* __restrict__ winj, int B) {
    XQ_PDL_ENTRY();
    xq_pf_hc_mix_body<5, 4>(x, inj, hn, u, winj, B);
}

// ===========================================================================
// TP-A (EXL3 TP=2 bring-up, rung 3 — PLAN/TP2_DESIGN_EXL3_2026-09-28.md §1.3/§2): expert
// parallelism. Each rank holds a STATIC subset of every trunk MoE layer's routed experts (the
// deal is a load-time table), the router/top-k stay replicated (every rank computes the same
// global ids/wts), and the rank's routed contribution is combined into an FP32 partial that one
// all-reduce per MoE layer sums (canonical rank0 + rank1 order, K2 mode 2) before ONE f16
// rounding. The kernels below are the only EP-specific device code; everything between them is
// the served expert path running on the rank's local tables.
// ===========================================================================

// == xq_moe_route restricted to the experts this rank holds. ep_lut[e] = local stacked index of
// global expert e on this rank, -1 = held by another rank. slotmap[e] = slot (first-seen over ids
// row-major among the LOCAL experts) or -1; idxmap[slot] = the LOCAL index (the rank's suh/svh
// tables are its stacked subset); offs_* = local word offsets into the rank's stacked trellis.
// ids/wts stay GLOBAL (the combine reads them through slotmap, which is -1 for remote experts).
extern "C" __global__ void xq_moe_route_ep(const int* __restrict__ ids, int m, int topk, int ne,
                                           const int* __restrict__ ep_lut,
                                           int* __restrict__ slotmap, int* __restrict__ idxmap,
                                           int* __restrict__ esel_dev,
                                           unsigned long long* __restrict__ offs_gu,
                                           unsigned long long* __restrict__ offs_d,
                                           unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    __shared__ int cnt;
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < m * topk; t++) {
            int e = ids[t];
            if (e >= 0 && e < ne && ep_lut[e] >= 0 && slotmap[e] == -1) {
                slotmap[e] = c;
                idxmap[c] = ep_lut[e];
                c++;
            }
        }
        cnt = c;
        esel_dev[0] = c;
    }
    __syncthreads();
    for (int s = tid; s < cnt; s += blockDim.x) {
        int le = idxmap[s];
        offs_gu[s] = (unsigned long long)le * gu_words;
        offs_d[s] = (unsigned long long)le * d_words;
    }
}

// PACK1 LEG 1: pad-row route fixup — rows >= r_live duplicate live row r_live-1's expert
// ids/weights so the per-row route tables never reference a pad row's arbitrary (though
// in-range) top-k. Layout matches sc.ids/sc.wts ([row*topk + j]; cf. xq_moe_route_ep's
// t in 0..m*topk scan). LEG 1 runs at exact R (5+5, no pad rows); this entry ships beside
// the router fold's route tail per design section 4 and is validated synthetically by
// --probe-exl3-pack2. NOTE for the k=3 leg: the fold's route tail compacts BEFORE this
// copy, so pad rows enter the compacted GEMM with their staged ids and only the per-row
// id/weight tables are aligned here — pad rows must be staged as duplicates of a live row
// (not zeros) unless this moves before the fold in that leg.
extern "C" __global__ void xq_pad_route(int* __restrict__ ids, float* __restrict__ wts,
                                        int topk, int r_live, int npad) {
    XQ_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npad * topk) return;
    const int dst_row = r_live + i / topk;
    const int j = i - (dst_row - r_live) * topk;
    ids[dst_row * topk + j] = ids[(r_live - 1) * topk + j];
    if (wts != nullptr) wts[dst_row * topk + j] = wts[(r_live - 1) * topk + j];
}

// == xq_moe_combine (same sgv tree, same j order, the WP20 combine's explicit roundings) with an
// FP32 output = this rank's PARTIAL. add_shared = 1 on exactly one rank (rank 0): the replicated
// shared expert enters the sum once. A rank with no local expert for a row writes that row's
// shared term (rank 0) or exact +0.0 (other ranks) — never leaves the row unwritten (the WP20
// down epilogue writes moe_out only from a last arrival, which a rank with esel == 0 never has).
extern "C" __global__ void xq_moe_combine_ep(float* __restrict__ out, const __half* __restrict__ y_d,
                                             const __half* __restrict__ y_sh,
                                             const int* __restrict__ ids, const float* __restrict__ wts,
                                             const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                             const __half* __restrict__ x, int k, int h, int M,
                                             int add_shared) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(y_sh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            int e = ids[(long long)m * k + j];
            int slot = slotmap[e];
            if (slot >= 0)
                acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(y_d[((long long)slot * M + m) * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
}

// f32 -> f16 (round-to-nearest-even, xq_f2h) — the single rounding after the EP all-reduce.
extern "C" __global__ void xq_cvt_f32_f16(__half* __restrict__ out, const float* __restrict__ in, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = xq_f2h(in[i]);
}

// =============================================================================
// TP-B (EXL3 TP=2 rung 4 — PLAN/TP2_DESIGN_EXL3_2026-09-28.md §1.3/§2): head-sharded
// attention/GDN + row-parallel dense (o_proj / out_proj / shared down) + expert-parallel
// PREFILL. Row-parallel chains reduce the PRE-output-Hadamard y_raw in FP32 (the output H128 and
// svh are linear and run once, after the all-reduce, on the f16-rounded sum — the same single
// rounding point as the TP=1 chain), so TP=2 differs from TP=1 only in the reassociation of the
// K split.
// =============================================================================

// Split-K combine into FP32: the SAME ascending slab sum as exl3_ks_combine, no f16 rounding —
// this rank's partial of y_raw for the row-parallel all-reduce.
extern "C" __global__ void xq_ks_combine_f32(const float* __restrict__ ws, float* __restrict__ out,
                                             int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    out[(size_t)m * N + n] = acc;
}

// f16 -> f32 widening (exact): a wide-M (prefill) row-parallel chain's f16 y_raw partial.
extern "C" __global__ void xq_cvt_f16_f32(float* __restrict__ out, const __half* __restrict__ in, long long n) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// EP prefill routing: global expert ids -> this rank's LOCAL stacked ids (-1 = held by another
// rank), and cand[i] = -1 for every pick (xq_moe_pf_place then writes the local picks' rows; a
// remote pick keeps -1 and the EP combine skips it).
extern "C" __global__ void xq_moe_ids_ep(const int* __restrict__ ids, int* __restrict__ ids_l,
                                         int* __restrict__ cand, const int* __restrict__ ep_lut,
                                         int r, int ne) {
    XQ_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r) return;
    const int e = ids[i];
    ids_l[i] = (e >= 0 && e < ne) ? ep_lut[e] : -1;
    cand[i] = -1;
}

// EP prefill: the compact row list holds offs_row[nl] (<= r) local rows; the row-wise glue
// kernels (had_suh_rows / had_svh_rows / gate_mul) are sized by r. Give rows [offs_row[nl], r)
// a VALID (token 0, local expert 0) mapping — their values are computed and never read (no
// tile, no cand points at them) — so no kernel reads an uninitialized index.
extern "C" __global__ void xq_moe_pf_ep_fill(const int* __restrict__ offs_row, int nl, int r,
                                             int* __restrict__ row_tok, int* __restrict__ row_eidx) {
    XQ_PDL_ENTRY();
    const int r0 = offs_row[nl];
    for (int i = r0 + blockIdx.x * blockDim.x + threadIdx.x; i < r; i += gridDim.x * blockDim.x) {
        row_tok[i] = 0;
        row_eidx[i] = 0;
    }
}

// EP prefill combine (xq_moe_combine_rows' sgv tree and j order) into an FP32 PARTIAL:
// add_shared = the shared-expert term enters on this rank (rank 0 when the shared expert is
// replicated; every rank when it is row-sharded — each adds sgv * its own partial); a pick with
// cand < 0 (remote expert) contributes nothing.
extern "C" __global__ void xq_moe_combine_rows_ep(float* __restrict__ out, const __half* __restrict__ yd,
                                                  const __half* __restrict__ ysh,
                                                  const int* __restrict__ cand, const float* __restrict__ wts,
                                                  const __half* __restrict__ sg,
                                                  const __half* __restrict__ x, int k, int h, int M,
                                                  int add_shared) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(ysh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            const int cr = cand[(long long)m * k + j];
            if (cr >= 0) acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(yd[(long long)cr * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
}

// TP-4M1: xq_moe_combine_rows_ep2<KM> = xq_moe_combine_rows_ep with every memory access vectorised and all the
// row's gathers issued up front. Per output element the arithmetic is the old kernel's:
//   acc = add_shared ? __fmul_rn(sgv, h2f(ysh)) : 0;  for j ascending, cand >= 0: acc = __fmaf_rn(wts_j, h2f(yd_j), acc)
// and sgv = xq_sig(tree) where tree is the old kernel's 1024-lane halving tree over the per-lane partials
//   part_v = ((0 + sg[v]*x[v]) + sg[v+1024]*x[v+1024]) + sg[v+2048]*x[v+2048]   (ascending, c < h only).
// The tree is reproduced EXACTLY (same pairs, same levels) but evaluated in one warp: lane l loads the 32 values
// sm[l + 32 j] and runs the levels s2 = 512 .. 32 in registers (pair (j, j + s2/32) per level), then s2 = 16 .. 1
// by shuffles — each add has the same two operands the old smem tree gave it, so sm[0] is bit-identical.
// One block = one row, h/8 threads, each owns 8 consecutive output columns (16-B yd / ysh loads, 2 x float4 store).
// Requires h == 2560 (320 threads), k <= KM, 16-B aligned yd / ysh / out / x / sg bases.
__device__ __forceinline__ void xq_u4_f8(const uint4& u, float* f) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
    const float2 d = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y; f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}
template <int KM>
__device__ __forceinline__ void xq_moe_combine_ep2_body(float* __restrict__ out, const __half* __restrict__ yd,
        const __half* __restrict__ ysh, const int* __restrict__ cand, const float* __restrict__ wts,
        const __half* __restrict__ sg, const __half* __restrict__ x, int k, int h, int M, int add_shared) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float sm[1024];
    __shared__ float sgv_s;
    const int tid = threadIdx.x;
    int cr[KM];
    float wv[KM];
    #pragma unroll
    for (int j = 0; j < KM; ++j) {
        cr[j] = -1; wv[j] = 0.0f;
        if (j < k) { cr[j] = cand[(long long)m * k + j]; wv[j] = wts[(long long)m * k + j]; }
    }
    uint4 yv[KM];
    #pragma unroll
    for (int j = 0; j < KM; ++j)
        if (cr[j] >= 0) yv[j] = *reinterpret_cast<const uint4*>(yd + (long long)cr[j] * h + tid * 8);
    uint4 sv = make_uint4(0, 0, 0, 0);
    if (add_shared) sv = *reinterpret_cast<const uint4*>(ysh + (long long)m * h + tid * 8);
    float sgv = 0.0f;
    if (add_shared) {
        if (tid < 128) {
            float part[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) part[u] = 0.0f;
            for (int c0 = tid * 8; c0 < h; c0 += 1024) {
                float xf[8], gf[8];
                xq_u4_f8(*reinterpret_cast<const uint4*>(x + (long long)m * h + c0), xf);
                xq_u4_f8(*reinterpret_cast<const uint4*>(sg + c0), gf);
                #pragma unroll
                for (int u = 0; u < 8; ++u) part[u] += gf[u] * xf[u];
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) sm[tid * 8 + u] = part[u];
        }
        __syncthreads();
        if (tid < 32) {
            float a[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) a[j] = sm[tid + 32 * j];
            #pragma unroll
            for (int j = 0; j < 16; ++j) a[j] = a[j] + a[j + 16];   // s2 = 512
            #pragma unroll
            for (int j = 0; j < 8; ++j) a[j] = a[j] + a[j + 8];     // s2 = 256
            #pragma unroll
            for (int j = 0; j < 4; ++j) a[j] = a[j] + a[j + 4];     // s2 = 128
            #pragma unroll
            for (int j = 0; j < 2; ++j) a[j] = a[j] + a[j + 2];     // s2 = 64
            float val = a[0] + a[1];                                // s2 = 32
            #pragma unroll
            for (int s2 = 16; s2 > 0; s2 >>= 1) val = val + __shfl_down_sync(0xFFFFFFFFu, val, s2);
            if (tid == 0) sgv_s = xq_sig(val);
        }
        __syncthreads();
        sgv = sgv_s;
    }
    float acc[8];
    if (add_shared) {
        float sf[8];
        xq_u4_f8(sv, sf);
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc[u] = __fmul_rn(sgv, sf[u]);
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) acc[u] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < KM; ++j) {
        if (cr[j] >= 0) {
            float yf[8];
            xq_u4_f8(yv[j], yf);
            #pragma unroll
            for (int u = 0; u < 8; ++u) acc[u] = __fmaf_rn(wv[j], yf[u], acc[u]);
        }
    }
    float4* o = reinterpret_cast<float4*>(out + (long long)m * h + tid * 8);
    o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}
extern "C" __global__ void __launch_bounds__(320) xq_moe_combine_rows_ep2_k10(float* __restrict__ out,
        const __half* __restrict__ yd, const __half* __restrict__ ysh, const int* __restrict__ cand,
        const float* __restrict__ wts, const __half* __restrict__ sg, const __half* __restrict__ x,
        int k, int h, int M, int add_shared) {
    XQ_PDL_ENTRY();
    xq_moe_combine_ep2_body<10>(out, yd, ysh, cand, wts, sg, x, k, h, M, add_shared);
}
extern "C" __global__ void __launch_bounds__(320) xq_moe_combine_rows_ep2_k16(float* __restrict__ out,
        const __half* __restrict__ yd, const __half* __restrict__ ysh, const int* __restrict__ cand,
        const float* __restrict__ wts, const __half* __restrict__ sg, const __half* __restrict__ x,
        int k, int h, int M, int add_shared) {
    XQ_PDL_ENTRY();
    xq_moe_combine_ep2_body<16>(out, yd, ysh, cand, wts, sg, x, k, h, M, add_shared);
}

