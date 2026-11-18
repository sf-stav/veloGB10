// exl3_bench/10_prefill_chunk.cuh: S-A3-f-b chunked prefill: conv rings, GDN scan, per-token recurrences inside the kernel.
// Weight ops run once at M=C rows; sequence recurrences loop the tokens in-kernel.
// Section 10 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ===========================================================================
// S-A3-f-b: CHUNKED PREFILL kernels. Process C consecutive tokens of ONE slot
// in a single sweep. Weight ops (GEMMs, hc, router) run once at M=C rows via
// the wide-M path; the sequence recurrences (conv rings, GDN scan, attention
// KV+softmax, PLE ring) loop the tokens INSIDE the kernel with the per-token
// reduction orders of the decode kernels unchanged. Prefill sits OUTSIDE the
// batch-invariance contract (same rule as the NVFP4 line); the ladder gate
// compares it against sequential decode-step prefill on real chunks.
// ===========================================================================

// ---- GDN conv1d over a chunk: state ring (k-deep) lives in registers, one
// channel per thread. Per token: shift, append cur, conv, silu (the xq_conv1d
// order: shift+append BEFORE the dot). qkv is overwritten in place (conv out).
extern "C" __global__ void xq_conv1d_chunk(__half* __restrict__ x, float* __restrict__ state,
                                           const float* __restrict__ w, int conv_dim, int k,
                                           int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(x[(long long)t * row_stride + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
        x[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
    }
    for (int j = 0; j < k; j++) sbase[j] = st[j];
}

// ---- WP12: xq_conv1d_chunk for the MTP verify, NOSTORE: reads the LIVE conv ring row
// (`slot`) and never writes it back (the verify's final ring was always discarded — the
// conv commit replays the accepted raw rows from live). Same per-token ops in the same
// order as xq_conv1d_chunk, on the same values the live->shadow copy used to deliver,
// so the conv outputs are bitwise identical; kills the per-layer shadow copy.
extern "C" __global__ void xq_conv1d_chunk_ns(__half* __restrict__ x, const float* __restrict__ state,
                                              const float* __restrict__ w, int conv_dim, int k,
                                              int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    const float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(x[(long long)t * row_stride + c]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
        x[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
    }
}

// PACK1 LEG 1: segmented WP12 conv chunk — grid (ceil(conv_dim/256), k_seg), block 256.
// Per segment: the SAME nostore walk (live ring row read, never written back) on the
// segment's rows (x based at row_start * row_stride) and the segment's slot state row.
// Per-token op order identical to xq_conv1d_chunk_ns, so per-segment bytes match the
// lone chunk bit for bit.
extern "C" __global__ void xq_conv1d_chunk_seg_ns(__half* __restrict__ x,
                                                  const float* __restrict__ state,
                                                  const float* __restrict__ w, int conv_dim,
                                                  int k, const int* __restrict__ seg,
                                                  int row_stride) {
    XQ_PDL_ENTRY();
    const int s = blockIdx.y * 4;
    const int slot = seg[s], row0 = seg[s + 1], C = seg[s + 2];
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float st[8];
    const float* sbase = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 0; j < k; j++) st[j] = sbase[j];
    __half* xc = x + (long long)row0 * row_stride + c;
    for (int t = 0; t < C; t++) {
        const float cur = xq_h2f(xc[(long long)t * row_stride]);
        for (int j = 1; j < k; j++) st[j - 1] = st[j];
        st[k - 1] = cur;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
        xc[(long long)t * row_stride] = xq_f2h(xq_silu(acc));
    }
}

// ---- A5 D17: the same conv as xq_conv1d_chunk, fully parallel over (token, channel). Output t
// is silu(sum_j w[c*k+j] * win_t[j]) with win_t[j] = input at time t-(k-1)+j (the ring after
// shift+append), times < 0 read from the pre-chunk state (ring index k+tt) — the identical
// ordered fp32 sum, so outputs are bit-identical. Writes to `out` (inputs must stay intact
// for the other threads); xq_conv1d_state_upd then advances the state from the intact inputs.
extern "C" __global__ void xq_conv1d_chunk_par(__half* __restrict__ out, const __half* __restrict__ x,
                                               const float* __restrict__ state, const float* __restrict__ w,
                                               int conv_dim, int k, int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)C * conv_dim) return;
    const int t = (int)(i / conv_dim), c = (int)(i % conv_dim);
    const float* sbase = state + ((long long)slot * conv_dim + c) * k;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        const int tt = t - (k - 1) + j;
        const float v = tt >= 0 ? xq_h2f(x[(long long)tt * row_stride + c]) : sbase[k + tt];
        acc += w[c * k + j] * v;
    }
    out[(long long)t * row_stride + c] = xq_f2h(xq_silu(acc));
}
// the ring after the chunk: st[j] = input at time C-k+j (or the old state's entry for times < 0)
extern "C" __global__ void xq_conv1d_state_upd(float* __restrict__ state, const __half* __restrict__ x,
                                               int conv_dim, int k, int C, int slot, int row_stride) {
    XQ_PDL_ENTRY();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float* sbase = state + ((long long)slot * conv_dim + c) * k;
    float nv[8];
    for (int j = 0; j < k; j++) {
        const int tt = C - k + j;
        nv[j] = tt >= 0 ? xq_h2f(x[(long long)tt * row_stride + c]) : sbase[k + tt];
    }
    for (int j = 0; j < k; j++) sbase[j] = nv[j];
}

// ---- GDN gated-delta-rule scan over a chunk: the xq_gdn_step body with the
// token loop INSIDE the kernel; S lives in SMEM across the chunk (loaded once,
// stored once — the value chain is identical to the per-step kernel).
// 12-arg cap: slot and C packed as slot_C = slot | C<<12 | nostore<<30.
// S-A3-m: nostore = the MTP verify pass — it reads the LIVE state row directly
// and never writes it back (the commit replays the accepted rows from live; the
// verify's final state was always discarded). Kills the per-layer shadow copy
// (3.1 MB r+w) and the discarded store (3.1 MB w) — same values read => same bits.
// S-A3-o G1: 12-arg cap — nh_nk = nh | n_k_heads<<16; `ring` (nullable) receives
// each token's rank-1 update factors for xq_gdn_commit_ring (see there).
#define XQ_GDN_RS 260   // ring floats per (token, head): delta[vd] | krow[kd] | g (vd=kd=128)
extern "C" __global__ void xq_gdn_step_chunk(__half* __restrict__ core, const __half* __restrict__ qkv,
                                             float* __restrict__ state, const __half* __restrict__ b_in,
                                             const __half* __restrict__ a_in, int nh_nk,
                                             int kd, int vd, const float* __restrict__ a_log,
                                             const float* __restrict__ dt_bias, int slot_C,
                                             float* __restrict__ ring) {
    XQ_PDL_ENTRY();
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int slot = slot_C & 0xFFF;
    const int C = (((unsigned)slot_C) >> 12) & 0x3FFFF;
    const bool nostore = (((unsigned)slot_C) >> 30) & 1u;
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
        if (ring != nullptr) {
            // delta/krow still hold THIS token's values (xq_gdn_token ends synced;
            // the next token overwrites them only after the barrier below).
            float* rr = ring + ((long long)t * nh + head) * XQ_GDN_RS;
            if (threadIdx.x < XQ_GDN_C) rr[bb0 + threadIdx.x] = delta[threadIdx.x];
            if (chunk == 0) {
                rr[vd + threadIdx.x] = krow[threadIdx.x];
                if (threadIdx.x == 0) rr[vd + kd] = gt;
            }
        }
        __syncthreads();
    }
    if (nostore) return;
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// ---- WP12 (2026-09-26): register-resident GDN recurrence — the MTP verify chunk
// (xq_gdn_step_chunk_r) and the decode step (xq_gdn_step_r) run the SAME device body
// (AGENTS §2.8). Bitwise equal to xq_gdn_token token by token (the old kernels stay
// reachable: GB10_WP12_OFF=step). Target contract: plain sm_121 — FP32 FMUL/FFMA,
// warp shuffles, shared memory, __syncthreads; no f/a feature is used.
// Geometry: block = 4 warps (vd = 128 columns), one block per (row group, head); warp w
// owns state columns [32w, 32w+32), lane = column, and holds its whole kd = 128 column
// in 128 registers for the token walk (the old body: S in smem, ~25 block barriers and
// 2 dependent global-load round trips per token; this one: none).
// Exactness vs xq_gdn_token (every op pinned with an _rn intrinsic, same order):
//  * q/k L2 norms: squares, halving tree 64, 32 then 16..1 — emulated in-warp exactly as
//    xq_gdn_chunk_tc_body does (lane l holds rows l, l+32, l+64, l+96); rsqrtf(sum+1e-6f);
//    q = (q*qn)*scale, k = k*kn; scale = 1/sqrtf((float)kd) from the RUNTIME kd argument
//    (the same IEEE sqrt/div sequence as the old kernels, never constant-folded);
//  * gates (beta, g): the verbatim expressions of xq_gdn_step/_chunk (xq_gdnr_gates);
//  * S *= g as FMUL; km = FFMA chain over a ascending from +0 on the scaled S;
//    delta = (v - km)*beta (FADD, FMUL); S = FFMA(k, delta, S); o = FFMA chain over a
//    ascending from +0 on the UPDATED S; core = f16_rn(o);
//  * o_t is computed in the same pass as km_{t+1}: both read S_t (o before the FMUL by
//    g_{t+1}, km after) — two independent chains, neither reordered.
// All C tokens' q/k (normalized), v rows and gates are staged once in smem while the
// state column is in flight (the per-token dependent global loads were the binder).
// The ring (nullable) receives the SAME rank-1 factors as xq_gdn_step_chunk's.
#define XQ_GDNR_KD   128
#define XQ_GDNR_VD   128
#define XQ_GDNR_CMAX 11
__device__ __forceinline__ void xq_gdnr_gates(float bh, float ah, float dtb, float alog,
                                              float& beta, float& gt) {
    beta = 1.0f / (1.0f + __expf(-bh));
    float sp = ah + dtb;
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    gt = __expf(-__expf(alog) * sp);
}
// WITH_O: o = sum_a S[a]*q[a] (S before the g scale), then S *= g, km = sum_a S[a]*k[a].
template <bool WITH_O>
__device__ __forceinline__ void xq_gdnr_pass(float (&s)[XQ_GDNR_KD], const float* __restrict__ qr,
                                             const float* __restrict__ kr, const float gt,
                                             float& o, float& km) {
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(kr + a);
        float4 q4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (WITH_O) q4 = *reinterpret_cast<const float4*>(qr + a);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (WITH_O) o = __fmaf_rn(s[a + j], qq[j], o);
            s[a + j] = __fmul_rn(s[a + j], gt);
            km = __fmaf_rn(s[a + j], kk[j], km);
        }
    }
}
__device__ __forceinline__ void xq_gdnr_update(float (&s)[XQ_GDNR_KD], const float* __restrict__ kr,
                                               const float delta) {
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 k4 = *reinterpret_cast<const float4*>(kr + a);
        s[a + 0] = __fmaf_rn(k4.x, delta, s[a + 0]);
        s[a + 1] = __fmaf_rn(k4.y, delta, s[a + 1]);
        s[a + 2] = __fmaf_rn(k4.z, delta, s[a + 2]);
        s[a + 3] = __fmaf_rn(k4.w, delta, s[a + 3]);
    }
}
__device__ __forceinline__ float xq_gdnr_out(const float (&s)[XQ_GDNR_KD], const float* __restrict__ qr) {
    float o = 0.0f;
    #pragma unroll
    for (int a = 0; a < XQ_GDNR_KD; a += 4) {
        const float4 q4 = *reinterpret_cast<const float4*>(qr + a);
        o = __fmaf_rn(s[a + 0], q4.x, o);
        o = __fmaf_rn(s[a + 1], q4.y, o);
        o = __fmaf_rn(s[a + 2], q4.z, o);
        o = __fmaf_rn(s[a + 3], q4.w, o);
    }
    return o;
}
// rows: token t reads qkv/b/a row (row0 + t) and writes core row (row0 + t); the ring
// (verify only) is indexed by t. store = write the final state back (decode).
__device__ __forceinline__ void xq_gdnr_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in, int nh, int n_k_heads,
    int kd_rt, int vd, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int slot, int row0, int head, int C, bool store, float* __restrict__ ring)
{
    constexpr int KD = XQ_GDNR_KD, VD = XQ_GDNR_VD;
    __shared__ __align__(16) float q_sh[XQ_GDNR_CMAX][KD];
    __shared__ __align__(16) float k_sh[XQ_GDNR_CMAX][KD];
    __shared__ float v_sh[XQ_GDNR_CMAX][VD];
    __shared__ float beta_sh[XQ_GDNR_CMAX], g_sh[XQ_GDNR_CMAX];
    if (kd_rt != KD || vd != VD || blockDim.x != VD || C < 1 || C > XQ_GDNR_CMAX) __trap();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = threadIdx.x;                       // = warp*32 + lane
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const long long qkv_stride = 2 * key_dim + nh * VD;
    float* Sg = state + ((long long)slot * nh + head) * KD * VD + col;
    // (1) the state column -> registers (the long pole: issued first, bandwidth-bound)
    float s[KD];
    #pragma unroll
    for (int a = 0; a < KD; a++) s[a] = Sg[(long long)a * VD];
    // (2) stage every token's normalized q/k, v row and gates (warp w: t = w, w+4, ...)
    const float scale = 1.0f / sqrtf((float)kd_rt);
    const float alog = a_log[head], dtb = dt_bias[head];
    for (int t = warp; t < C; t += 4) {
        const long long r = row0 + t;
        const __half* colp = qkv + r * qkv_stride;
        const __half* qp = colp + key_head * KD;
        const __half* kp = colp + key_dim + key_head * KD;
        const __half* vp = colp + 2 * key_dim + head * VD;
        float q[4], k[4];
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            q[m] = xq_h2f(qp[lane + 32 * m]);
            k[m] = xq_h2f(kp[lane + 32 * m]);
            v_sh[t][lane + 32 * m] = xq_h2f(vp[lane + 32 * m]);
        }
        float beta, gt;
        xq_gdnr_gates(xq_h2f(b_in[r * nh + head]), xq_h2f(a_in[r * nh + head]), dtb, alog, beta, gt);
        // halving tree: s2=64 pairs (i, i+64), s2=32 pairs (i, i+32), then 16..1
        float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                             __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
        float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                             __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
            sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
        }
        const float qn = rsqrtf(__fadd_rn(__shfl_sync(0xFFFFFFFFu, sq, 0), 1e-6f));
        const float kn = rsqrtf(__fadd_rn(__shfl_sync(0xFFFFFFFFu, sk, 0), 1e-6f));
        float* rr = ring != nullptr ? ring + ((long long)t * nh + head) * XQ_GDN_RS : nullptr;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            const int i = lane + 32 * m;
            q_sh[t][i] = __fmul_rn(__fmul_rn(q[m], qn), scale);
            const float kv = __fmul_rn(k[m], kn);
            k_sh[t][i] = kv;
            if (rr != nullptr) rr[VD + i] = kv;
        }
        if (lane == 0) {
            beta_sh[t] = beta;
            g_sh[t] = gt;
            if (rr != nullptr) rr[VD + KD] = gt;
        }
    }
    __syncthreads();
    // (3) the token walk — no block barriers, no global loads
    __half* cb = core + (long long)row0 * (nh * VD) + (long long)head * VD + col;
    {
        float km = 0.0f, o_unused = 0.0f;
        xq_gdnr_pass<false>(s, nullptr, k_sh[0], g_sh[0], o_unused, km);
        // re-read k in the update (a warp fence stops the compiler keeping all 128 k
        // values of the pass live across the km chain — that was a 255-reg spill)
        __syncwarp();
        const float delta = __fmul_rn(__fsub_rn(v_sh[0][col], km), beta_sh[0]);
        if (ring != nullptr) ring[(long long)head * XQ_GDN_RS + col] = delta;
        xq_gdnr_update(s, k_sh[0], delta);
    }
    for (int t = 1; t < C; t++) {
        float km = 0.0f, o = 0.0f;
        xq_gdnr_pass<true>(s, q_sh[t - 1], k_sh[t], g_sh[t], o, km);
        cb[(long long)(t - 1) * (nh * VD)] = xq_f2h(o);
        __syncwarp();
        const float delta = __fmul_rn(__fsub_rn(v_sh[t][col], km), beta_sh[t]);
        if (ring != nullptr) ring[((long long)t * nh + head) * XQ_GDN_RS + col] = delta;
        xq_gdnr_update(s, k_sh[t], delta);
    }
    cb[(long long)(C - 1) * (nh * VD)] = xq_f2h(xq_gdnr_out(s, q_sh[C - 1]));
    if (!store) return;
    #pragma unroll
    for (int a = 0; a < KD; a++) Sg[(long long)a * VD] = s[a];
}
// verify chunk: the xq_gdn_step_chunk signature/packing (slot_C = slot | C<<12 | nostore<<30,
// nh_nk = nh | n_k_heads<<16); grid nh, block 128 (host: kd = vd = 128, C <= 8).
extern "C" __global__ void __launch_bounds__(128, 1)
xq_gdn_step_chunk_r(__half* __restrict__ core, const __half* __restrict__ qkv,
                    float* __restrict__ state, const __half* __restrict__ b_in,
                    const __half* __restrict__ a_in, int nh_nk, int kd, int vd,
                    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                    int slot_C, float* __restrict__ ring) {
    XQ_PDL_ENTRY();
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int slot = slot_C & 0xFFF;
    const int C = (((unsigned)slot_C) >> 12) & 0x3FFFF;
    const bool nostore = (((unsigned)slot_C) >> 30) & 1u;
    xq_gdnr_body(core, qkv, state, b_in, a_in, nh, n_k_heads, kd, vd, a_log, dt_bias,
                 slot, 0, blockIdx.x, C, !nostore, ring);
}

// PACK1 LEG 1 (S-B9-CF-PACK1): segmented verify chunk — grid (nh, k_seg), block 128.
// blockIdx.y picks the segment row of `seg` = k_seg x {slot, row_start, len, flags};
// the SAME xq_gdnr_body runs with slot = seg.slot (state plane), row0 = seg.row_start
// (qkv/b/a/core rows), C = seg.len (<= XQ_GDNR_CMAX: the shared stages are CMAX-deep)
// and store = !nostore (the packed verify is nostore; the commit replays from live).
// The ring base is offset by row_start so the body's t-indexed ring writes land at the
// segment's absolute rows — no body edit, no reassociation: per-segment bytes equal the
// lone chunk's by construction (AGENTS 2.8).
extern "C" __global__ void __launch_bounds__(128, 1)
xq_gdn_step_chunk_seg(__half* __restrict__ core, const __half* __restrict__ qkv,
                      float* __restrict__ state, const __half* __restrict__ b_in,
                      const __half* __restrict__ a_in, int nh_nk, int kd, int vd,
                      const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                      const int* __restrict__ seg, float* __restrict__ ring) {
    XQ_PDL_ENTRY();
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int s = blockIdx.y * 4;
    const int slot = seg[s], row0 = seg[s + 1], C = seg[s + 2];
    // seg[s+3] flags: bit0 = nostore (the packed verify never writes live state;
    // the LEG 2 commit replays from live). Host staging sets flags = 1.
    const bool nostore = (seg[s + 3] & 1) != 0;
    if (C < 1 || C > XQ_GDNR_CMAX) __trap();
    float* ring_seg = ring != nullptr ? ring + (long long)row0 * nh * XQ_GDN_RS : nullptr;
    xq_gdnr_body(core, qkv, state, b_in, a_in, nh, n_k_heads, kd, vd, a_log, dt_bias,
                 slot, row0, blockIdx.x, C, !nostore, ring_seg);
}
// decode step: the xq_gdn_step signature; grid B*nh (block = b*nh + head), block 128.
extern "C" __global__ void __launch_bounds__(128, 1)
xq_gdn_step_r(__half* __restrict__ core, const __half* __restrict__ qkv,
              float* __restrict__ state, const __half* __restrict__ b_in,
              const __half* __restrict__ a_in, int nh, int n_k_heads, int kd, int vd,
              const float* __restrict__ a_log, const float* __restrict__ dt_bias,
              const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    const int b = blockIdx.x / nh;
    const int head = blockIdx.x - b * nh;
    // S-A3-e: the recurrent state is PER-SLOT (slotpos is interleaved [slot, pos])
    const int slot = slot_ids[b * 2];
    xq_gdnr_body(core, qkv, state, b_in, a_in, nh, n_k_heads, kd, vd, a_log, dt_bias,
                 slot, b, head, 1, true, nullptr);
}

// ---- S-A3-v: CHUNKWISE-PARALLEL GDN prefill scan (tensor cores, split-f16 operands).
// The xq_gdn_step_chunk recurrence re-associated into 32-token chunks (the WY/UT form
// of gdn_chunk_tc_b, gpu_batch.cu — same chunk math, validated there vs the sequential
// scan): per chunk, the intra-chunk grams (A = decayed KK^T, D = decayed QK^T), the
// forward substitution U = (I + diag(beta)A)^-1 diag(beta)(V - Gamma K S), then
// O = Gamma Q S + D U and S' = gamma_C S + K_w^T U — every product on mma.m16n8k16.
// Differences from gdn_chunk_tc_b (numerics care, the prefillx state drift is the gate):
//  * reads the EXL3 packed f16 qkv / b / a rows directly (no f32 prep scratch: that is
//    ~300 MB of traffic per layer at C=2048) and writes f16 core rows;
//  * q/k L2 norms computed in-kernel in the EXACT xq_gdn_token order (squares, halving
//    tree 64,32 then 16..1, no FMA contraction) => the normalized q/k equal decode's bits;
//  * SPLIT f16 operands: x = hi + lo (hi = f16(x), lo = f16(x - hi)), each product as
//    hi*hi + hi*lo + lo*hi => ~2^-17..2^-21 relative per operand instead of f16's 2^-11
//    (S-A3-v cycle 1: plain f16 operands drifted out of class — prefillx maxd 22.2 / 15.2
//    / 18.8 vs bars 12.64 / 6.98 / 5.24; the W = V - Gamma K S difference cancels, so
//    operand rounding of K and S is amplified). Template mask SPLIT: 1 = S, 2 = Q/K,
//    4 = U/D (7 = all; the ablation lives in THINGS_TRIED);
//  * the f32 MASTER state stays in registers for the whole walk (warp w owns kd rows
//    [16w, 16w+16) x the block's VC cols); its operand copy Sf is f32 (split on load).
// Prefill-only: decode (xq_gdn_step) and verify (xq_gdn_step_chunk nostore/ring) untouched.
// kd = vd = 128 (host-checked). Grid nh * (128/XQ_GTC_VC), block 256, dynamic smem
// XQ_GTC_SMEM (> 48 KB: raw launch with the opt-in, exl3_forward.rs gdn_chunk_tc_fn).
#define XQ_GTC_C   32
#define XQ_GTC_VC  32
#define XQ_GTC_KD2 (128 + 8)
#define XQ_GTC_CD2 (XQ_GTC_C + 8)
#define XQ_GTC_VD2 (XQ_GTC_VC + 8)
#define XQ_GTC_SD2 (XQ_GTC_VC + 4)
#define XQ_GTC_THR 256
// halves: Qh,Ql,Kh,Kl [C][KD2] + Vb [C][VD2] + Dh,Dl [C][CD2] + Uh,Ul,Lh,Ll [C][VD2]
// f32:    Sf [128][SD2] + Af [C][C] + Wf [C][VC] + bt [C] + lg [C+1]
#define XQ_GTC_SMEM ((4 * XQ_GTC_C * XQ_GTC_KD2 + XQ_GTC_C * XQ_GTC_VD2 + 2 * XQ_GTC_C * XQ_GTC_CD2 \
                      + 4 * XQ_GTC_C * XQ_GTC_VD2) * 2 \
                     + (128 * XQ_GTC_SD2 + XQ_GTC_C * XQ_GTC_C + XQ_GTC_C * XQ_GTC_VC + XQ_GTC_C + XQ_GTC_C + 1) * 4)
__device__ __forceinline__ unsigned xq_gtc_h2(__half lo, __half hi) {
    __half2 v = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void xq_gtc_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// split store of an f32 value into a hi/lo tile pair
__device__ __forceinline__ void xq_gtc_st2(__half* h, __half* lo, int i, float x) {
    const __half xh = __float2half_rn(x);
    h[i] = xh; lo[i] = __float2half_rn(x - __half2float(xh));
}
// A fragment (row-major tile, the permuted-k convention of gdn_chunk_tc_b: k = cc..cc+3)
__device__ __forceinline__ void xq_gtc_lda(unsigned* a, const __half* T, int ld, int r0, int cc) {
    a[0] = *(const unsigned*)(T + r0 * ld + cc);       a[1] = *(const unsigned*)(T + (r0 + 8) * ld + cc);
    a[2] = *(const unsigned*)(T + r0 * ld + cc + 2);   a[3] = *(const unsigned*)(T + (r0 + 8) * ld + cc + 2);
}
// A fragment of the TRANSPOSE of a row-major tile T[k][row] (K^T for the S update)
__device__ __forceinline__ void xq_gtc_ldat(unsigned* a, const __half* T, int ld, int r0, int cc) {
    a[0] = xq_gtc_h2(T[cc * ld + r0], T[(cc + 1) * ld + r0]);
    a[1] = xq_gtc_h2(T[cc * ld + r0 + 8], T[(cc + 1) * ld + r0 + 8]);
    a[2] = xq_gtc_h2(T[(cc + 2) * ld + r0], T[(cc + 3) * ld + r0]);
    a[3] = xq_gtc_h2(T[(cc + 2) * ld + r0 + 8], T[(cc + 3) * ld + r0 + 8]);
}
// B fragment from a row-major [k][n] tile of halves (rows cc..cc+3, column col)
__device__ __forceinline__ void xq_gtc_ldb(unsigned* b, const __half* T, int ld, int cc, int col) {
    b[0] = xq_gtc_h2(T[cc * ld + col], T[(cc + 1) * ld + col]);
    b[1] = xq_gtc_h2(T[(cc + 2) * ld + col], T[(cc + 3) * ld + col]);
}
// B fragment pair (hi, lo) split on the fly from an f32 [k][n] tile
__device__ __forceinline__ void xq_gtc_ldb_f32(unsigned* bh, unsigned* bl, const float* T, int ld, int cc, int col) {
    float x[4] = {T[cc * ld + col], T[(cc + 1) * ld + col], T[(cc + 2) * ld + col], T[(cc + 3) * ld + col]};
    __half h[4], l[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { h[i] = __float2half_rn(x[i]); l[i] = __float2half_rn(x[i] - __half2float(h[i])); }
    bh[0] = xq_gtc_h2(h[0], h[1]); bh[1] = xq_gtc_h2(h[2], h[3]);
    bl[0] = xq_gtc_h2(l[0], l[1]); bl[1] = xq_gtc_h2(l[2], l[3]);
}
// d += A*B with optional lo terms (small terms first)
__device__ __forceinline__ void xq_gtc_mma3(float* d, const unsigned* ah, const unsigned* al, bool sa,
                                            const unsigned* bh, const unsigned* bl, bool sb) {
    if (sa) xq_gtc_mma(d, al, bh);
    if (sb) xq_gtc_mma(d, ah, bl);
    xq_gtc_mma(d, ah, bh);
}
// A5-L1: harness-only phase stamps (PROF = true twins only; PROF = false compiles to the served SASS).
// Thread 0 accumulates clock64 deltas between consecutive block barriers into pacc[phase]; the
// block writes pacc[0..7] + globaltimer start/end to prof[blockIdx.x * 10 ..].
#define XQ_GTC_STAMP(i) do { if (PROF && t == 0) { const long long _n = clock64(); pacc[i] += _n - pt; pt = _n; } } while (0)
#define XQ_GTC_PROF_BEGIN() long long pacc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
    long long pt = PROF ? clock64() : 0; unsigned long long pg0 = PROF ? xq_tune_gtimer() : 0ull
#define XQ_GTC_PROF_END() do { if (PROF && t == 0 && prof != nullptr) { \
    _Pragma("unroll") for (int _i = 0; _i < 8; _i++) prof[blockIdx.x * 10 + _i] = (unsigned long long)pacc[_i]; \
    prof[blockIdx.x * 10 + 8] = pg0; prof[blockIdx.x * 10 + 9] = xq_tune_gtimer(); } } while (0)
template <int SPLIT, bool PROF = false>
__device__ __forceinline__ void xq_gdn_chunk_tc_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int nh_nk, int slot, int N, unsigned long long* __restrict__ prof = nullptr)
{
    constexpr bool SS = SPLIT & 1, SQK = SPLIT & 2, SUD = SPLIT & 4;
    constexpr int C = XQ_GTC_C, VC = XQ_GTC_VC, KD = 128, VD = 128;
    constexpr int KD2 = XQ_GTC_KD2, CD2 = XQ_GTC_CD2, VD2 = XQ_GTC_VD2, SD2 = XQ_GTC_SD2;
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int ncb = VD / VC;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const int qkv_stride = 2 * key_dim + nh * VD;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;

    extern __shared__ __align__(16) unsigned char xq_gtc_dyn[];
    __half* Qh = (__half*)xq_gtc_dyn;       // [C][KD2]
    __half* Ql = Qh + C * KD2;
    __half* Kh = Ql + C * KD2;
    __half* Kl = Kh + C * KD2;
    __half* Vb = Kl + C * KD2;              // [C][VD2]
    __half* Dh = Vb + C * VD2;              // [C][CD2]
    __half* Dl = Dh + C * CD2;
    __half* Uh = Dl + C * CD2;              // [C][VD2] (unscaled U)
    __half* Ul = Uh + C * VD2;
    __half* Lh = Ul + C * VD2;              // [C][VD2] (lambda-scaled U)
    __half* Ll = Lh + C * VD2;
    float* Sf = (float*)(Ll + C * VD2);     // [KD][SD2]
    float* Af = Sf + KD * SD2;              // [C][C]
    float* Wf = Af + C * C;                 // [C][VC]
    float* bt = Wf + C * VC;                // [C]
    float* lg = bt + C;                     // [C+1]

    // f32 master state: warp w owns kd rows [16w, 16w+16) x VC cols (4 n-tiles x 4)
    float* Sg = state + ((long long)slot * nh + head) * KD * VD + bb0;
    float Sr[VC / 8][4];
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sr[j][e] = Sg[(long long)row * VD + col];
            Sf[row * SD2 + col] = Sr[j][e];
        }
    const float qscale = 1.0f / sqrtf((float)KD);
    const float neg_a = -__expf(a_log[head]);
    XQ_GTC_PROF_BEGIN();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage q/k (normalized in xq_gdn_token's exact order) + v + beta/log-g ----
        for (int tt = warp; tt < C; tt += XQ_GTC_THR / 32) {
            if (tt < n) {
                const __half* col = qkv + (long long)(c0 + tt) * qkv_stride;
                const __half* qp = col + key_head * KD;
                const __half* kp = col + key_dim + key_head * KD;
                float q[4], k[4];
                #pragma unroll
                for (int m = 0; m < 4; m++) { q[m] = xq_h2f(qp[lane + 32 * m]); k[m] = xq_h2f(kp[lane + 32 * m]); }
                // halving tree over squares: s2=64 pairs (i, i+64), s2=32 pairs (i, i+32), then 16..1
                float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                                     __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
                float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                                     __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
                    sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
                }
                const float qn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sq, 0) + 1e-6f);
                const float kn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sk, 0) + 1e-6f);
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, __fmul_rn(__fmul_rn(q[m], qn), qscale));
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, __fmul_rn(k[m], kn));
                }
            } else {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, 0.f);
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, 0.f);
                }
            }
        }
        for (int i = t; i < C * VC; i += XQ_GTC_THR) {
            const int tt = i / VC, c = i % VC;
            Vb[tt * VD2 + c] = (tt < n)
                ? qkv[(long long)(c0 + tt) * qkv_stride + 2 * key_dim + head * VD + bb0 + c]
                : __float2half_rn(0.f);
        }
        if (t < C) {
            if (t < n) {
                const long long r = (long long)(c0 + t) * nh + head;
                bt[t] = 1.0f / (1.0f + __expf(-xq_h2f(b_in[r])));
                float sp = xq_h2f(a_in[r]) + dt_bias[head];
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                lg[t + 1] = neg_a * sp;
            } else { bt[t] = 0.f; lg[t + 1] = 0.f; }
        }
        __syncthreads();
        XQ_GTC_STAMP(0);
        if (warp == 0) {                         // inclusive scan of log-g (one warp, C = 32)
            float v = lg[lane + 1];
            #pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                const float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if (lane >= o) v += u;
            }
            lg[lane + 1] = v;
            if (lane == 0) lg[0] = 0.f;
        }
        __syncthreads();
        XQ_GTC_STAMP(1);

        const int mt = warp >> 2, nt = warp & 3;   // 32x32 outputs: one m16n8 tile per warp
        const int r0 = mt * 16 + g;
        // ---- A / D grams (k = 128) and W = V - exp(lg[tt+1]) (K S) (k = 128) ----
        {
            float accA[4] = {0, 0, 0, 0}, accD[4] = {0, 0, 0, 0}, accW[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aKh[4], aKl[4], aQh[4], aQl[4], bKh[2], bKl[2], bSh[2], bSl[2];
                xq_gtc_lda(aKh, Kh, KD2, r0, cc); xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                if (SQK) { xq_gtc_lda(aKl, Kl, KD2, r0, cc); xq_gtc_lda(aQl, Ql, KD2, r0, cc); }
                const int s_row = nt * 8 + g;       // B = K^T: column s = row s of K
                bKh[0] = *(const unsigned*)(Kh + s_row * KD2 + cc); bKh[1] = *(const unsigned*)(Kh + s_row * KD2 + cc + 2);
                if (SQK) { bKl[0] = *(const unsigned*)(Kl + s_row * KD2 + cc); bKl[1] = *(const unsigned*)(Kl + s_row * KD2 + cc + 2); }
                xq_gtc_mma3(accA, aKh, aKl, SQK, bKh, bKl, SQK);
                xq_gtc_mma3(accD, aQh, aQl, SQK, bKh, bKl, SQK);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(accW, aKh, aKl, SQK, bSh, bSl, SS);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                // A[t,s] = (gamma_t / gamma_s)(k_t . k_s): g_t S_{t-1} carries gamma_t. gdn_chunk_tc_b
                // uses gamma_{t-1} (lg[row]) — one gate factor short per row (S-A3-v; hidden as its
                // "~2% bf16 envelope" because g ~ 1).
                Af[row * C + scol] = (scol < row) ? accA[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                const float dv = (scol <= row && scol < n) ? accD[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                if (SUD) xq_gtc_st2(Dh, Dl, row * CD2 + scol, dv); else Dh[row * CD2 + scol] = __float2half_rn(dv);
                const float v = (row < n) ? xq_h2f(Vb[row * VD2 + scol]) : 0.f;
                Wf[row * VC + scol] = v - __expf(lg[row + 1]) * accW[e];
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(2);

        // ---- U = (I + diag(beta)A)^-1 (beta . W): forward substitution, one column per thread ----
        if (t < VC) {
            for (int tt = 0; tt < n; tt++) {
                const float b = bt[tt];
                float acc = b * Wf[tt * VC + t];
                for (int s = 0; s < tt; s++) acc -= b * Af[tt * C + s] * Wf[s * VC + t];
                Wf[tt * VC + t] = acc;
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(3);
        // ---- U (unscaled) and lambda-scaled U (lambda_s = exp(lg[n] - lg[s+1])) ----
        for (int i = t; i < C * VC; i += XQ_GTC_THR) {
            const int s = i / VC, c = i % VC;
            const float u = (s < n) ? Wf[s * VC + c] : 0.f;
            const float ul = u * __expf(lg[n] - lg[s + 1]);
            if (SUD) { xq_gtc_st2(Uh, Ul, s * VD2 + c, u); xq_gtc_st2(Lh, Ll, s * VD2 + c, ul); }
            else { Uh[s * VD2 + c] = __float2half_rn(u); Lh[s * VD2 + c] = __float2half_rn(ul); }
        }
        __syncthreads();
        XQ_GTC_STAMP(4);

        // ---- O = exp(lg[tt+1]) (Q S) + D U  (QS k = 128, then DU k = C accumulates) ----
        {
            float acc[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aQh[4], aQl[4], bSh[2], bSl[2];
                xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                if (SQK) xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aQh, aQl, SQK, bSh, bSl, SS);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[e] *= __expf(lg[r0 + 8 * (e >= 2) + 1]);
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aDh[4], aDl[4], bUh[2], bUl[2];
                xq_gtc_lda(aDh, Dh, CD2, r0, cc);
                xq_gtc_ldb(bUh, Uh, VD2, cc, nt * 8 + g);
                if (SUD) { xq_gtc_lda(aDl, Dl, CD2, r0, cc); xq_gtc_ldb(bUl, Ul, VD2, cc, nt * 8 + g); }
                xq_gtc_mma3(acc, aDh, aDl, SUD, bUh, bUl, SUD);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                if (row < n) {
                    const int col = nt * 8 + 2 * tq + (e & 1);
                    core[(long long)(c0 + row) * (nh * VD) + head * VD + bb0 + col] = __float2half_rn(acc[e]);
                }
            }
        }
        __syncthreads();   // every Sf read (W, O) is done before the update rewrites it
        XQ_GTC_STAMP(5);

        // ---- S' = gamma_C S + K^T . Ulambda  (warp = 16 kd rows x VC; k = C); master stays f32 ----
        {
            const float gam = __expf(lg[n]);
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) Sr[j][e] *= gam;
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aTh[4], aTl[4];
                xq_gtc_ldat(aTh, Kh, KD2, warp * 16 + g, cc);
                if (SQK) xq_gtc_ldat(aTl, Kl, KD2, warp * 16 + g, cc);
                #pragma unroll
                for (int j = 0; j < VC / 8; j++) {
                    unsigned bLh[2], bLl[2];
                    xq_gtc_ldb(bLh, Lh, VD2, cc, j * 8 + g);
                    if (SUD) xq_gtc_ldb(bLl, Ll, VD2, cc, j * 8 + g);
                    xq_gtc_mma3(Sr[j], aTh, aTl, SQK, bLh, bLl, SUD);
                }
            }
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = warp * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sf[row * SD2 + col] = Sr[j][e];
                }
        }
        __syncthreads();
        XQ_GTC_STAMP(6);
    }
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sg[(long long)row * VD + col] = Sr[j][e];
        }
    XQ_GTC_PROF_END();
}
#define XQ_GTC_INST(NAME, SPLIT) \
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void NAME( \
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state, \
    const __half* __restrict__ b_in, const __half* __restrict__ a_in, \
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N) { \
    XQ_PDL_ENTRY(); \
    xq_gdn_chunk_tc_body<SPLIT>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N); }
XQ_GTC_INST(xq_gdn_chunk_tc, 7)
XQ_GTC_INST(xq_gdn_chunk_tc_s0, 0)
XQ_GTC_INST(xq_gdn_chunk_tc_s1, 1)
XQ_GTC_INST(xq_gdn_chunk_tc_s3, 3)

// ---- A5-L1 (2026-09-28): xq_gdn_chunk_tc2 — the served xq_gdn_chunk_tc (SPLIT = 7) with the same
// per-element arithmetic, re-scheduled. BITWISE equal (--probe-exl3-binv EXL3-GDN-TC2). Changes:
//  (1) forward substitution U = (I + diag(beta)A)^-1 diag(beta) W as a 31-step WAVEFRONT instead of one
//      thread per column walking 496 dependent smem-fed FFMAs: lane = row r, warp w owns columns
//      4w..4w+3; step k broadcasts the finished W[k] (__shfl) and every row r > k does its s = k term.
//      Each row keeps its exact chain: acc = b_r*W[r] (FMUL), then acc = fma(-(b_r*A[r][s]), W[s], acc)
//      for s = 0, 1, .., r-1 ascending — the SASS of the served loop (FMUL b*A; FFMA -t, W, acc).
//      b_r*A[r][s] (the served `b * Af`) is formed once in the gram epilogue, stored transposed (Tt[s][r])
//      so the per-step reads are conflict-free; W is stored transposed (Wt[c][r]) for the same reason.
//  (2) the U / lambda-U split stores run in the solve's registers (no Wf round trip, one barrier less);
//  (3) the log-g inclusive scan runs in warp 0 straight from the staging registers (one barrier less);
//  (4) the next 32-token sub-chunk's q/k/v/b/a global loads are issued into registers right after the
//      current sub-chunk is staged, so their latency hides behind the grams/solve/O/S phases.
// Same smem footprint (XQ_GTC_SMEM: Af -> Tt, Wf -> Wt, same sizes), same grid / block / launch.
template <bool PROF>
__device__ __forceinline__ void xq_gdn_chunk_tc2_body(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int nh_nk, int slot, int N, unsigned long long* __restrict__ prof)
{
    constexpr int C = XQ_GTC_C, VC = XQ_GTC_VC, KD = 128, VD = 128;
    constexpr int KD2 = XQ_GTC_KD2, CD2 = XQ_GTC_CD2, VD2 = XQ_GTC_VD2, SD2 = XQ_GTC_SD2;
    static_assert(C == 32 && VC == 32 && XQ_GTC_THR == 256, "tc2 geometry: lane = row, warp = 4 columns");
    const int nh = nh_nk & 0xFFFF;
    const int n_k_heads = nh_nk >> 16;
    const int ncb = VD / VC;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * KD;
    const int qkv_stride = 2 * key_dim + nh * VD;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;

    extern __shared__ __align__(16) unsigned char xq_gtc_dyn[];
    __half* Qh = (__half*)xq_gtc_dyn;       // [C][KD2]
    __half* Ql = Qh + C * KD2;
    __half* Kh = Ql + C * KD2;
    __half* Kl = Kh + C * KD2;
    __half* Vb = Kl + C * KD2;              // [C][VD2]
    __half* Dh = Vb + C * VD2;              // [C][CD2]
    __half* Dl = Dh + C * CD2;
    __half* Uh = Dl + C * CD2;              // [C][VD2] (unscaled U)
    __half* Ul = Uh + C * VD2;
    __half* Lh = Ul + C * VD2;              // [C][VD2] (lambda-scaled U)
    __half* Ll = Lh + C * VD2;
    float* Sf = (float*)(Ll + C * VD2);     // [KD][SD2]
    float* Tt = Sf + KD * SD2;              // [C][C]  Tt[s][r] = beta_r * A[r][s]  (s < r)
    float* Wt = Tt + C * C;                 // [VC][C] Wt[c][r] = W[r][c]
    float* bt = Wt + C * VC;                // [C]
    float* lg = bt + C;                     // [C+1]

    float* Sg = state + ((long long)slot * nh + head) * KD * VD + bb0;
    float Sr[VC / 8][4];
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sr[j][e] = Sg[(long long)row * VD + col];
            Sf[row * SD2 + col] = Sr[j][e];
        }
    const float qscale = 1.0f / sqrtf((float)KD);
    const float neg_a = -__expf(a_log[head]);
    const float dtb = dt_bias[head];

    // (4) register prefetch of one sub-chunk's raw inputs: tokens tt = warp + 8i (i < 4) q/k halves
    // lane + 32m; V elements i = t + 256j (tt = i / VC, c = i % VC); b/a of token t (warp 0).
    __half pq[4][4], pk[4][4], pv[4], pb, pa;
    auto prefetch = [&](int c0n) {
        const int nn = min(C, N - c0n);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int tt = warp + 8 * i;
            const __half* col = qkv + (long long)(c0n + (tt < nn ? tt : 0)) * qkv_stride;
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                pq[i][m] = (tt < nn) ? col[key_head * KD + lane + 32 * m] : __float2half_rn(0.f);
                pk[i][m] = (tt < nn) ? col[key_dim + key_head * KD + lane + 32 * m] : __float2half_rn(0.f);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = t + XQ_GTC_THR * j, tt = i / VC, c = i % VC;
            pv[j] = (tt < nn) ? qkv[(long long)(c0n + tt) * qkv_stride + 2 * key_dim + head * VD + bb0 + c]
                              : __float2half_rn(0.f);
        }
        if (warp == 0 && lane < nn) {
            const long long r = (long long)(c0n + lane) * nh + head;
            pb = b_in[r]; pa = a_in[r];
        } else { pb = __float2half_rn(0.f); pa = __float2half_rn(0.f); }
    };
    prefetch(0);
    XQ_GTC_PROF_BEGIN();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage q/k (normalized in xq_gdn_token's exact order) + v + beta/log-g (+ its scan) ----
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int tt = warp + 8 * i;
            if (tt < n) {
                float q[4], k[4];
                #pragma unroll
                for (int m = 0; m < 4; m++) { q[m] = xq_h2f(pq[i][m]); k[m] = xq_h2f(pk[i][m]); }
                float sq = __fadd_rn(__fadd_rn(__fmul_rn(q[0], q[0]), __fmul_rn(q[2], q[2])),
                                     __fadd_rn(__fmul_rn(q[1], q[1]), __fmul_rn(q[3], q[3])));
                float sk = __fadd_rn(__fadd_rn(__fmul_rn(k[0], k[0]), __fmul_rn(k[2], k[2])),
                                     __fadd_rn(__fmul_rn(k[1], k[1]), __fmul_rn(k[3], k[3])));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sq = __fadd_rn(sq, __shfl_down_sync(0xFFFFFFFFu, sq, o));
                    sk = __fadd_rn(sk, __shfl_down_sync(0xFFFFFFFFu, sk, o));
                }
                const float qn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sq, 0) + 1e-6f);
                const float kn = rsqrtf(__shfl_sync(0xFFFFFFFFu, sk, 0) + 1e-6f);
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, __fmul_rn(__fmul_rn(q[m], qn), qscale));
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, __fmul_rn(k[m], kn));
                }
            } else {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int r = lane + 32 * m;
                    xq_gtc_st2(Qh, Ql, tt * KD2 + r, 0.f);
                    xq_gtc_st2(Kh, Kl, tt * KD2 + r, 0.f);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = t + XQ_GTC_THR * j, tt = i / VC, c = i % VC;
            Vb[tt * VD2 + c] = pv[j];
        }
        if (warp == 0) {                         // (3) beta + log-g, then its inclusive scan, in-warp
            float v = 0.f;
            if (lane < n) {
                bt[lane] = 1.0f / (1.0f + __expf(-xq_h2f(pb)));
                float sp = xq_h2f(pa) + dtb;
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                v = neg_a * sp;
            } else { bt[lane] = 0.f; }
            #pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                const float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if (lane >= o) v += u;
            }
            lg[lane + 1] = v;
            if (lane == 0) lg[0] = 0.f;
        }
        if (c0 + C < N) prefetch(c0 + C);        // (4) next sub-chunk's loads fly behind the phases below
        __syncthreads();
        XQ_GTC_STAMP(0);

        const int mt = warp >> 2, nt = warp & 3;   // 32x32 outputs: one m16n8 tile per warp
        const int r0 = mt * 16 + g;
        // ---- A / D grams (k = 128) and W = V - exp(lg[tt+1]) (K S) (k = 128) ----
        {
            float accA[4] = {0, 0, 0, 0}, accD[4] = {0, 0, 0, 0}, accW[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aKh[4], aKl[4], aQh[4], aQl[4], bKh[2], bKl[2], bSh[2], bSl[2];
                xq_gtc_lda(aKh, Kh, KD2, r0, cc); xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                xq_gtc_lda(aKl, Kl, KD2, r0, cc); xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                const int s_row = nt * 8 + g;       // B = K^T: column s = row s of K
                bKh[0] = *(const unsigned*)(Kh + s_row * KD2 + cc); bKh[1] = *(const unsigned*)(Kh + s_row * KD2 + cc + 2);
                bKl[0] = *(const unsigned*)(Kl + s_row * KD2 + cc); bKl[1] = *(const unsigned*)(Kl + s_row * KD2 + cc + 2);
                xq_gtc_mma3(accA, aKh, aKl, true, bKh, bKl, true);
                xq_gtc_mma3(accD, aQh, aQl, true, bKh, bKl, true);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(accW, aKh, aKl, true, bSh, bSl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                // served: Af = accA * exp(..) (s < row), then the solve's t = b * Af — both roundings kept
                Tt[scol * C + row] = (scol < row) ? __fmul_rn(bt[row], __fmul_rn(accA[e], __expf(lg[row + 1] - lg[scol + 1]))) : 0.f;
                const float dv = (scol <= row && scol < n) ? accD[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                xq_gtc_st2(Dh, Dl, row * CD2 + scol, dv);
                const float v = (row < n) ? xq_h2f(Vb[row * VD2 + scol]) : 0.f;
                Wt[scol * C + row] = v - __expf(lg[row + 1]) * accW[e];
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(2);

        // ---- (1)+(2) U by a 31-step wavefront (lane = row, warp = 4 columns), U / lambda-U stored ----
        {
            const int r = lane;
            const float br = bt[r];
            float acc[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) acc[j] = __fmul_rn(br, Wt[(warp * 4 + j) * C + r]);
            #pragma unroll
            for (int k = 0; k < C - 1; k++) {
                const float tk = Tt[k * C + r];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float wk = __shfl_sync(0xFFFFFFFFu, acc[j], k);
                    if (r > k) acc[j] = __fmaf_rn(-tk, wk, acc[j]);
                }
            }
            const float lam = __expf(lg[n] - lg[r + 1]);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = warp * 4 + j;
                const float u = (r < n) ? acc[j] : 0.f;
                const float ul = u * lam;
                xq_gtc_st2(Uh, Ul, r * VD2 + c, u); xq_gtc_st2(Lh, Ll, r * VD2 + c, ul);
            }
        }
        __syncthreads();
        XQ_GTC_STAMP(3);

        // ---- O = exp(lg[tt+1]) (Q S) + D U  (QS k = 128, then DU k = C accumulates) ----
        {
            float acc[4] = {0, 0, 0, 0};
            #pragma unroll 2
            for (int ks = 0; ks < KD; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aQh[4], aQl[4], bSh[2], bSl[2];
                xq_gtc_lda(aQh, Qh, KD2, r0, cc);
                xq_gtc_lda(aQl, Ql, KD2, r0, cc);
                xq_gtc_ldb_f32(bSh, bSl, Sf, SD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aQh, aQl, true, bSh, bSl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[e] *= __expf(lg[r0 + 8 * (e >= 2) + 1]);
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aDh[4], aDl[4], bUh[2], bUl[2];
                xq_gtc_lda(aDh, Dh, CD2, r0, cc);
                xq_gtc_ldb(bUh, Uh, VD2, cc, nt * 8 + g);
                xq_gtc_lda(aDl, Dl, CD2, r0, cc); xq_gtc_ldb(bUl, Ul, VD2, cc, nt * 8 + g);
                xq_gtc_mma3(acc, aDh, aDl, true, bUh, bUl, true);
            }
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = r0 + 8 * (e >= 2);
                if (row < n) {
                    const int col = nt * 8 + 2 * tq + (e & 1);
                    core[(long long)(c0 + row) * (nh * VD) + head * VD + bb0 + col] = __float2half_rn(acc[e]);
                }
            }
        }
        __syncthreads();   // every Sf read (W, O) is done before the update rewrites it
        XQ_GTC_STAMP(5);

        // ---- S' = gamma_C S + K^T . Ulambda  (warp = 16 kd rows x VC; k = C); master stays f32 ----
        {
            const float gam = __expf(lg[n]);
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) Sr[j][e] *= gam;
            #pragma unroll
            for (int ks = 0; ks < C; ks += 16) {
                const int cc = ks + 4 * tq;
                unsigned aTh[4], aTl[4];
                xq_gtc_ldat(aTh, Kh, KD2, warp * 16 + g, cc);
                xq_gtc_ldat(aTl, Kl, KD2, warp * 16 + g, cc);
                #pragma unroll
                for (int j = 0; j < VC / 8; j++) {
                    unsigned bLh[2], bLl[2];
                    xq_gtc_ldb(bLh, Lh, VD2, cc, j * 8 + g);
                    xq_gtc_ldb(bLl, Ll, VD2, cc, j * 8 + g);
                    xq_gtc_mma3(Sr[j], aTh, aTl, true, bLh, bLl, true);
                }
            }
            #pragma unroll
            for (int j = 0; j < VC / 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = warp * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sf[row * SD2 + col] = Sr[j][e];
                }
        }
        __syncthreads();
        XQ_GTC_STAMP(6);
    }
    #pragma unroll
    for (int j = 0; j < VC / 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = warp * 16 + g + 8 * (e >= 2);
            const int col = j * 8 + 2 * tq + (e & 1);
            Sg[(long long)row * VD + col] = Sr[j][e];
        }
    XQ_GTC_PROF_END();
}
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc2(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc2_body<false>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, nullptr); }
// harness-only phase-stamped twins (never launched by a serving path)
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc_prof(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N,
    unsigned long long* __restrict__ prof) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc_body<7, true>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, prof); }
extern "C" __global__ __launch_bounds__(XQ_GTC_THR, 1) void xq_gdn_chunk_tc2_prof(
    __half* __restrict__ core, const __half* __restrict__ qkv, float* __restrict__ state,
    const __half* __restrict__ b_in, const __half* __restrict__ a_in,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, int nh_nk, int slot, int N,
    unsigned long long* __restrict__ prof) {
    XQ_PDL_ENTRY();
    xq_gdn_chunk_tc2_body<true>(core, qkv, state, b_in, a_in, a_log, dt_bias, nh_nk, slot, N, prof); }

