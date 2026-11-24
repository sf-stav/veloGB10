// gpu_inkling2/85_lanes.cuh — lane-batched decode entries with DEVICE-SIDE positions (leg S4). Section of
// gpu_inkling2.cu.
//
// A decode step carries one row per active lane. The per-sequence ops (attention prep, attention, its combine,
// the two short convs) used to loop over the lanes on the host with each lane's position passed by value, so a
// step could not be one CUDA graph and every lane cost its own launches. These entries do all rows in ONE launch
// and read everything that changes from step to step from device memory:
//   rinfo [4][IKL_RI] i32: ids (row r), positions p_r, lane index of row r, and t_r = the row's index inside its
//         lane's run of consecutive rows (S7: a lane may carry up to 16 consecutive rows, e.g. a speculative
//         verify; t_r = 0 for a plain decode row). One upload per step.
//   tab   [lanes][slots] InkLaneSlot: each lane's K / V caches, conv rings and cache mask (written once at boot).
// Each row runs the SAME device body as the per-lane kernels (10_attn.cuh / 20_conv.cuh). The convs (attention
// prep, sconv) run the M-row body with t = t_r on arguments rebased to the lane's first row (so a row reads the
// earlier rows of its own run from the call's inputs, exactly like the per-segment kernels); the attention runs
// every row as its own M = 1 call at p_r (split computed from p_r exactly as k1.rs `attn_split`), so EVERY row of
// a multi-row run is bit-identical to a 1-row decode of that token at that position (S7: the verify rows).

struct InkLaneSlot {
    __half* kc; __half* vc;
    float* ring[4];          // k-in, v-in, attn-out, mlp-out conv input rings [32][C]
    unsigned cap_mask; unsigned pad;
    unsigned long long pad2;  // 64 B = k1.rs LANE_SLOT_U64 x 8
};
static_assert(sizeof(InkLaneSlot) == 64, "InkLaneSlot must match LANE_SLOT_U64");

#define IKL_MAXS 18          // max attention splits of one row (chunk >= span / 16: <= 17)
#define IKL_PSTR (24 * 64)   // per-row stride of the split partials (pm / pl: floats; pacc: x 128)
#define IKL_RI 48            // rinfo row capacity (k1.rs LANES_MAXR)
#define IKL_POS(ri, r) ((ri)[IKL_RI + (r)])
#define IKL_LANE(ri, r) ((ri)[2 * IKL_RI + (r)])
#define IKL_T(ri, r) ((ri)[3 * IKL_RI + (r)])

__device__ __forceinline__ int ikl_np2(int x) { return x <= 1 ? 1 : 1 << (32 - __clz(x - 1)); }
// k1.rs attn_split(p0, m = 1, window): chunk, c0, nsplit
__device__ __forceinline__ void ikl_split(int p0, int window, int& chunk, int& c0, int& nsplit) {
    const int kmin = window > 0 ? max(0, p0 + 1 - window) : 0;
    const int span = p0 + 16 - kmin;
    chunk = max(64, ikl_np2((span + 15) / 16));
    c0 = kmin / chunk;
    nsplit = p0 / chunk - c0 + 1;
}

// S7: the multi-row (verify) twin of ika_attn_body for ONE lane's run of M <= 16 consecutive rows that share the
// split chunk size: the CTA covers ABSOLUTE chunk c0(row 0) + bx, the run's rows share every K / V subtile load, and
// each row keeps its own decode split (c0_r, nsplit_r from its own position, k1.rs attn_split): it only writes
// for chunks inside its own range, at its own slot (or finalizes in place when its nsplit is 1). Visible keys,
// subtile order and arithmetic per row are those of a 1-row decode at that position (a key the CTA range adds is
// outside the row's window / causal range or chunk: masked, and an all-masked subtile is skipped), so every row is
// bit-identical to the 1-row decode. (Same body text as ika_attn_body except the epilogue's row gating / slot.)
__device__ __forceinline__ void ika_attn_mr_body(const InkAttnArgs& a, const int bx) {
    const int bz = 0;
    extern __shared__ __align__(16) unsigned char smraw[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = blockIdx.y;
    const int t0 = bz * 16;
    const int mt = min(16, a.M - t0);
    const int nq = 3 * mt;
    const int qcap = 3 * min(16, a.M);                                // same for every CTA of the launch
    float* sQ = reinterpret_cast<float*>(smraw);                     // [qcap][128]
    float* sR = sQ + qcap * IKA_HD;                                   // [qcap][16]
    float* sP = sR + qcap * IKA_DREL;                                 // [16][80] proj columns
    __half* sK = reinterpret_cast<__half*>(sP + IKA_DREL * 80);       // [64][130]
    __half* sV = sK + IKA_SUB * IKA_KSTR;                             // [64][128]

    for (int i = threadIdx.x; i < nq * IKA_HD; i += blockDim.x) {
        const int qi = i / IKA_HD, d = i % IKA_HD;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        sQ[i] = a.Qf[((size_t)t * 24 + h) * IKA_HD + d];
    }
    for (int i = threadIdx.x; i < nq * IKA_DREL; i += blockDim.x) {
        const int qi = i / IKA_DREL, d = i % IKA_DREL;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        sR[i] = a.Rf[((size_t)t * 24 + h) * IKA_DREL + d];
    }
    const int P0 = a.p0 + t0, P1 = P0 + mt - 1;
    const int kmin = a.window ? max(0, P0 - a.window + 1) : 0;
    const long long cs = (long long)(a.c0 + bx) * a.chunk;
    const int lo = (int)max((long long)kmin, cs);
    const int hi = (int)min((long long)P1 + 1, cs + a.chunk);       // exclusive
    const float scale = rsqrtf((float)IKA_HD);

    float m_run[IKA_QPW], l_run[IKA_QPW], acc[IKA_QPW][4];
    #pragma unroll
    for (int j = 0; j < IKA_QPW; ++j) {
        m_run[j] = -INFINITY; l_run[j] = 0.f;
        #pragma unroll
        for (int d = 0; d < 4; ++d) acc[j][d] = 0.f;
    }
    for (int kb = (lo / IKA_SUB) * IKA_SUB; kb < hi; kb += IKA_SUB) {
        __syncthreads();
        // K / V subtile (rows outside [lo, hi) are zero: never read uninitialised cache)
        for (int i = threadIdx.x; i < IKA_SUB * 64; i += blockDim.x) {
            const int r = i >> 6, w2 = i & 63;     // 64 half2 per row
            const int kp = kb + r;
            __half2 kv2 = __float2half2_rn(0.f), vv2 = __float2half2_rn(0.f);
            if (kp >= lo && kp < hi) {
                const size_t base = (size_t)(kp & a.cap_mask) * 1024 + kvh * IKA_HD + 2 * w2;
                kv2 = *reinterpret_cast<const __half2*>(a.kc + base);
                vv2 = *reinterpret_cast<const __half2*>(a.vc + base);
            }
            *reinterpret_cast<__half2*>(sK + r * IKA_KSTR + 2 * w2) = kv2;
            *reinterpret_cast<__half2*>(sV + r * IKA_HD + 2 * w2) = vv2;
        }
        // proj columns for distances dmin .. dmin+79 (dmin = P0 - (kb + 63))
        const int dmin = P0 - (kb + IKA_SUB - 1);
        for (int i = threadIdx.x; i < IKA_DREL * 80; i += blockDim.x) {
            const int dd = i / 80, c = i % 80;
            const int dist = dmin + c;
            sP[i] = (dist >= 0 && dist < a.ext) ? a.proj[(size_t)dd * a.ext + dist] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < IKA_QPW; ++j) {
            const int qi = warp + 8 * j;
            if (qi >= nq) break;
            const int P = P0 + qi / 3;
            const float* q = sQ + qi * IKA_HD;
            const float* rr = sR + qi * IKA_DREL;
            float s[2];
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int r = lane + 32 * u, kp = kb + r;
                const __half2* kr = reinterpret_cast<const __half2*>(sK + r * IKA_KSTR);
                float dot = 0.f;
                #pragma unroll 16
                for (int d2 = 0; d2 < IKA_HD / 2; ++d2) {
                    const float2 kf = __half22float2(kr[d2]);
                    dot = fmaf(q[2 * d2], kf.x, dot);
                    dot = fmaf(q[2 * d2 + 1], kf.y, dot);
                }
                const int dist = P - kp;
                float b = 0.f;
                const int col = dist - dmin;
                #pragma unroll
                for (int dd = 0; dd < IKA_DREL; ++dd) b = fmaf(rr[dd], sP[dd * 80 + col], b);
                const bool ok = kp >= lo && kp < hi && dist >= 0 && (a.window == 0 || dist < a.window);
                s[u] = ok ? fmaf(dot, scale, b) : -INFINITY;
            }
            const float msub = ika_wmax(fmaxf(s[0], s[1]));
            if (msub == -INFINITY) continue;          // nothing visible here: state untouched
            const float mnew = fmaxf(m_run[j], msub);
            const float fold = __expf(m_run[j] - mnew); // m_run = -inf -> 0
            const float e0 = __expf(s[0] - mnew), e1 = __expf(s[1] - mnew);
            l_run[j] = l_run[j] * fold + ika_wsum(e0 + e1);
            m_run[j] = mnew;
            float o0 = acc[j][0] * fold, o1 = acc[j][1] * fold, o2 = acc[j][2] * fold, o3 = acc[j][3] * fold;
            #pragma unroll 8
            for (int r = 0; r < IKA_SUB; ++r) {
                const float pr = __shfl_sync(0xffffffffu, r < 32 ? e0 : e1, r & 31);
                const uint2 vv = *reinterpret_cast<const uint2*>(sV + r * IKA_HD + 4 * lane);
                const float2 va = __half22float2(*reinterpret_cast<const __half2*>(&vv.x));
                const float2 vb = __half22float2(*reinterpret_cast<const __half2*>(&vv.y));
                o0 = fmaf(pr, va.x, o0); o1 = fmaf(pr, va.y, o1); o2 = fmaf(pr, vb.x, o2); o3 = fmaf(pr, vb.y, o3);
            }
            acc[j][0] = o0; acc[j][1] = o1; acc[j][2] = o2; acc[j][3] = o3;
        }
    }
    #pragma unroll
    for (int j = 0; j < IKA_QPW; ++j) {
        const int qi = warp + 8 * j;
        if (qi >= nq) break;
        const int t = t0 + qi / 3, h = 3 * kvh + qi % 3;
        int rch, rc0, nsplit;
        ikl_split(a.p0 + t, a.window, rch, rc0, nsplit);
        const int slot = a.c0 + bx - rc0;
        if (slot < 0 || slot >= nsplit) continue;            // not one of this row's chunks: the decode has no such CTA
        if (nsplit == 1) {
            const float* y = a.Y + (size_t)t * a.ldy;
            const float sink = y[a.os + h];
            const float m = fmaxf(m_run[j], sink);
            const float fold = __expf(m_run[j] - m);
            const float inv = 1.0f / (l_run[j] * fold + __expf(sink - m));
            const float4 gt = *reinterpret_cast<const float4*>(y + a.og + h * IKA_HD + 4 * lane);
            uint16_t* o = a.out + (size_t)t * 24 * IKA_HD + h * IKA_HD + 4 * lane;
            const float w = fold * inv;
            uint2 pk;
            pk.x = (uint32_t)ika_f2b(acc[j][0] * w * ika_sig(gt.x)) | ((uint32_t)ika_f2b(acc[j][1] * w * ika_sig(gt.y)) << 16);
            pk.y = (uint32_t)ika_f2b(acc[j][2] * w * ika_sig(gt.z)) | ((uint32_t)ika_f2b(acc[j][3] * w * ika_sig(gt.w)) << 16);
            *reinterpret_cast<uint2*>(o) = pk;
        } else {
            const size_t pi = (size_t)t * IKL_PSTR + (size_t)h * nsplit + slot;   // the lanes partial layout
            if (lane == 0) { a.pm[pi] = m_run[j]; a.pl[pi] = l_run[j]; }
            *reinterpret_cast<float4*>(a.pacc + pi * IKA_HD + 4 * lane) = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
    }
}
// S7: the run of consecutive rows that row `row` belongs to (start s, length L <= 16) and whether the multi-row
// kernel takes it (L >= 2 and every row of the run has the same split chunk); otherwise each row is its own decode.
__device__ __forceinline__ bool ikl_run_mr(const int* __restrict__ rinfo, int row, int n, int window, int& s, int& L) {
    s = row - IKL_T(rinfo, row);
    L = 1;
    while (s + L < n && IKL_T(rinfo, s + L) == L) ++L;
    if (L < 2) return false;
    int ch0, c, ns;
    ikl_split(IKL_POS(rinfo, s), window, ch0, c, ns);
    for (int r = 1; r < L; ++r) { int ch; ikl_split(IKL_POS(rinfo, s + r), window, ch, c, ns); if (ch != ch0) return false; }
    return true;
}

// grid (n rows), block 1024. `a` carries the layer's constants; Y / Qf / Rf are row 0 of the step.
extern "C" __global__ __launch_bounds__(1024) void ink_attn_prep_lanes(const InkPrepArgs a, const InkLaneSlot* __restrict__ tab,
        int slot, int nslots, const int* __restrict__ rinfo) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.x, t = IKL_T(rinfo, row), r0 = row - t;
    const InkLaneSlot s = tab[IKL_LANE(rinfo, row) * nslots + slot];
    InkPrepArgs b = a;
    b.Y = a.Y + (size_t)r0 * a.ldy;
    b.Qf = a.Qf + (size_t)r0 * 24 * IKA_HD;
    b.Rf = a.Rf + (size_t)r0 * 24 * IKA_DREL;
    b.ring_k = s.ring[0]; b.ring_v = s.ring[1]; b.ring_write = 1;
    b.kc = s.kc; b.vc = s.vc; b.cap_mask = s.cap_mask;
    b.p0 = IKL_POS(rinfo, row) - t; b.M = t + 1;
    ika_attn_prep_body(b, t);
}

// grid (IKL_MAXS, 8 kv heads, n rows), block 256, smem attn_smem(1). CTAs past the row's split count exit.
extern "C" __global__ __launch_bounds__(256) void ink_attn_lanes(const InkAttnArgs a, const InkLaneSlot* __restrict__ tab,
        int slot, int nslots, const int* __restrict__ rinfo) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.z;
    const int p = IKL_POS(rinfo, row);
    int chunk, c0, nsplit;
    ikl_split(p, a.window, chunk, c0, nsplit);
    if (nsplit > IKL_MAXS) __trap();
    if ((int)blockIdx.x >= nsplit) return;
    { int rs, rl; if (a.c0 == 0 && ikl_run_mr(rinfo, row, gridDim.z, a.window, rs, rl)) return; }  // S7: the run is ink_attn_lanes_mr's (a.c0 = 1: mr off)
    const InkLaneSlot s = tab[IKL_LANE(rinfo, row) * nslots + slot];
    InkAttnArgs b = a;
    b.Qf = a.Qf + (size_t)row * 24 * IKA_HD;
    b.Rf = a.Rf + (size_t)row * 24 * IKA_DREL;
    b.kc = s.kc; b.vc = s.vc; b.cap_mask = s.cap_mask;
    b.p0 = p; b.M = 1; b.chunk = chunk; b.c0 = c0;
    b.Y = a.Y + (size_t)row * a.ldy;
    b.out = a.out + (size_t)row * 24 * IKA_HD;
    b.pm = a.pm + (size_t)row * IKL_PSTR; b.pl = a.pl + (size_t)row * IKL_PSTR; b.pacc = a.pacc + (size_t)row * IKL_PSTR * IKA_HD;
    ika_attn_body(b, nsplit, blockIdx.x, 0);
}

// grid (n rows, 24), block 128: merge the split partials of rows that used more than one split.
extern "C" __global__ __launch_bounds__(128) void ink_attn_combine_lanes(const InkAttnArgs a, const int* __restrict__ rinfo) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.x;
    const int p = IKL_POS(rinfo, row);
    int chunk, c0, nsplit;
    ikl_split(p, a.window, chunk, c0, nsplit);
    if (nsplit <= 1) return;
    InkAttnArgs b = a;
    b.Y = a.Y + (size_t)row * a.ldy;
    b.out = a.out + (size_t)row * 24 * IKA_HD;
    b.pm = a.pm + (size_t)row * IKL_PSTR; b.pl = a.pl + (size_t)row * IKL_PSTR; b.pacc = a.pacc + (size_t)row * IKL_PSTR * IKA_HD;
    ika_attn_combine_body(b, nsplit, 0);
}

// grid (ceil(C / 256), n rows), block 256: static conv + residual of each row on its lane's ring `ri`.
extern "C" __global__ __launch_bounds__(256) void ink_sconv_lanes(const float* __restrict__ x, int ldx,
        const InkLaneSlot* __restrict__ tab, int slot, int nslots, int ri, const float* __restrict__ w,
        const float* __restrict__ base, float* __restrict__ out, int C, const int* __restrict__ rinfo, float div) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.y, t = IKL_T(rinfo, row), r0 = row - t;
    float* ring = tab[IKL_LANE(rinfo, row) * nslots + slot].ring[ri];
    ika_sconv_body(x + (size_t)r0 * ldx, ldx, ring, 31, 1, w, base ? base + (size_t)r0 * C : nullptr, out + (size_t)r0 * C,
                   C, IKL_POS(rinfo, row) - t, div, t);
}

// S7: grid (IKL_MAXS + 1, 8 kv heads, n rows), block 256, smem attn_smem(16). The CTA whose z is the FIRST row of a
// multi-row run (ikl_run_mr) serves the whole run with ika_attn_mr_body (absolute chunk c0(first row) + x); every
// other CTA exits. The per-row ink_attn_lanes skips exactly these runs; ink_attn_combine_lanes merges per row.
extern "C" __global__ __launch_bounds__(256) void ink_attn_lanes_mr(const InkAttnArgs a, const InkLaneSlot* __restrict__ tab,
        int slot, int nslots, const int* __restrict__ rinfo) {
    INK2_PDL_ENTRY();
    const int row = blockIdx.z;
    if (IKL_T(rinfo, row) != 0) return;
    int s, L;
    if (!ikl_run_mr(rinfo, row, gridDim.z, a.window, s, L)) return;
    const int p = IKL_POS(rinfo, row);
    int chunk, c0, nsplit;
    ikl_split(p, a.window, chunk, c0, nsplit);
    int cl, c0l, nsl;
    ikl_split(p + L - 1, a.window, cl, c0l, nsl);
    if (c0l + nsl - c0 > IKL_MAXS + 1) __trap();
    if ((int)blockIdx.x >= c0l + nsl - c0) return;
    const InkLaneSlot sl = tab[IKL_LANE(rinfo, row) * nslots + slot];
    InkAttnArgs b = a;
    b.Qf = a.Qf + (size_t)row * 24 * IKA_HD;
    b.Rf = a.Rf + (size_t)row * 24 * IKA_DREL;
    b.kc = sl.kc; b.vc = sl.vc; b.cap_mask = sl.cap_mask;
    b.p0 = p; b.M = L; b.chunk = chunk; b.c0 = c0;
    b.Y = a.Y + (size_t)row * a.ldy;
    b.out = a.out + (size_t)row * 24 * IKA_HD;
    b.pm = a.pm + (size_t)row * IKL_PSTR; b.pl = a.pl + (size_t)row * IKL_PSTR; b.pacc = a.pacc + (size_t)row * IKL_PSTR * IKA_HD;
    ika_attn_mr_body(b, blockIdx.x);
}
