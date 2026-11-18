// gpu_batch/03_gdn_prefill_kv_q4.cuh: P4 B3 chunked GDN prefill scan (WY/UT form), gdn_prep_b, 4-bit KV write/compact/dequant.
// GDN prefill-ONLY; decode and verify call the same device function with the same reduction order.
// Section 3 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===========================================================================
// P0 (F-report step 3): one coalesced, full-occupancy prep pass per GDN layer.
// Grid: (N); block: 128 threads == kd for one head... expanded: blockIdx.y = head.
// blockDim.x = kd; grid (N, nh). Each block: halving-tree norms for its (t, head),
// reads the packed qkv ONCE coalesced (per head), writes DENSE scratch:
//   Qs [N][nh][KDS] f32 (normalized, *1/sqrt(kd); KDS=kd+4 pad keeps the scan's
//      smem loads conflict-free when copied 1:1),
//   Ks [N][nh][KDS] f32 (normalized, NO extra scale — gdn_token convention),
//   Vs [N][nh][vd] f32,
//   Ps [N][nh][2]  f32 (beta, log_g).
// The scan kernel then stages by pure coalesced f32 copies — no norms, no pitched bf16.
// Numerics: same formulas as the in-kernel staging it replaces (which themselves mirror
// gdn_token; the reduction order here is a halving tree like gdn_token's).
// ===========================================================================
extern "C" __global__ void gdn_prep_b(const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ b_in, const __nv_bfloat16* __restrict__ a_in,
    int qkv_ba_stride, int kd_vd, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, int N_nkh,
    float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs,
    float* __restrict__ Ps)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int nh = (int)gridDim.y;
    const int t = blockIdx.x;                 // token
    const int head = blockIdx.y;              // value head
    if (t >= N) return;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int a = threadIdx.x;                // 0..kd-1
    const int KDS = kd + 4;

    const __nv_bfloat16* col = qkv + (long long)t * qkv_stride;
    const float qv = b2f(col[key_head * kd + a]);
    const float kv = b2f(col[key_dim + key_head * kd + a]);

    // halving-tree norms (gdn_token's pattern; thread a of kd)
    __shared__ float nrm[2][128];
    nrm[0][a] = qv * qv; nrm[1][a] = kv * kv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) {
        if (a < s2) { nrm[0][a] += nrm[0][a + s2]; nrm[1][a] += nrm[1][a + s2]; }
        __syncthreads();
    }
    const float qn = rsqrtf(nrm[0][0] + 1e-6f) * rsqrtf((float)kd);   // q: L2 * 1/sqrt(kd)
    const float kn = rsqrtf(nrm[1][0] + 1e-6f);                       // k: L2 ONLY
    __syncthreads();

    const long long base = ((long long)t * nh + head);
    Qs[base * KDS + a] = qv * qn;
    Ks[base * KDS + a] = kv * kn;

    // v: threads beyond kd copy columns (vd=128=kd here; general: strided)
    if (a < vd) Vs[base * vd + a] = b2f(col[2 * key_dim + head * vd + a]);

    if (a == 0) {
        float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)t * ba_stride + head])));
        float sp = b2f(a_in[(long long)t * ba_stride + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        Ps[base * 2 + 0] = beta;
        Ps[base * 2 + 1] = -__expf(a_log[head]) * sp;   // raw log-g (scan cumsums)
    }
}

// GB10_GDN_CHUNK2 (2026-08-26): tensor-core chunked GDN prefill scan — the chunk math of
// gdn_chunk_prefill_b (o/S rel-L2 7.6e-5 scalar) with every GEMM phase on mma.m16n8k16
// bf16 (probe: tool_probe/gdn_chunk_tc.cu, 9.4 ms/layer at N=8192 vs 198 sequential = 21x;
// o/S rel-L2 ~2.2e-2 vs the f32 seq oracle = the bf16-operand envelope, non-compounding
// over N=32..8192). Consumes the SAME gdn_prep_b P0 scratch; S carried in bf16 across
// chunks (FLA precedent). Host dispatches only for kd==128 && vd==128 && n>=256.
__device__ __forceinline__ unsigned gtc_pack2(float lo, float hi) {
    __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void gtc_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#define GTC_C 32
#define GTC_VC 64
#define GTC_KD2 (128 + 8)
#define GTC_CD2 (GTC_C + 8)
#define GTC_VD2 (GTC_VC + 8)
#define GTC_THR 256
extern "C" __global__ __launch_bounds__(GTC_THR, 1) void gdn_chunk_tc_b(
    __nv_bfloat16* __restrict__ core, float* __restrict__ state,
    const float* __restrict__ Qs, const float* __restrict__ Ks,
    const float* __restrict__ Vs, const float* __restrict__ Ps,
    int kd_vd, int N_nkh)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    (void)kd; (void)vd;              // host-guaranteed 128/128; tiles below are compile-time
    const int N = N_nkh & 0xFFFFFF;
    const int ncb = GTC_VC == 64 ? 2 : 1;
    const int NH_ = (int)(gridDim.x / ncb);
    const int C = GTC_C, VC = GTC_VC, KD = 128, VD = 128, KD2 = GTC_KD2, CD2 = GTC_CD2, VD2 = GTC_VD2;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;      // fragment coords

    extern __shared__ unsigned char gtc_dyn[];
    __nv_bfloat16* Qb = (__nv_bfloat16*)gtc_dyn;                        // [C][KD2]
    __nv_bfloat16* Kb = Qb + C * KD2;                      // [C][KD2]
    __nv_bfloat16* Kt = Kb + C * KD2;                      // [KD][CD2] (transposed K)
    __nv_bfloat16* Vb = Kt + KD * CD2;                     // [C][VD2]
    __nv_bfloat16* Db = Vb + C * VD2;                      // [C][CD2]
    __nv_bfloat16* Ub = Db + C * CD2;                      // [C][VD2]  (unscaled U)
    __nv_bfloat16* Ulb = Ub + C * VD2;                     // [C][VD2]  (lambda-scaled U)
    __nv_bfloat16* Sb = Ulb + C * VD2;                     // [KD][VD2]
    float* Af = (float*)(Sb + KD * VD2);          // [C][C]
    float* Wf = Af + C * C;                       // [C][VC]
    float* Of = Wf + C * VC;                      // [C][VC]
    float* bt = Of + C * VC;                      // [C]
    float* lg = bt + C;                           // [C+1]

    const float* S_in = state + ((long long)head * KD) * VD + bb0;
    for (int i = t; i < KD * VC; i += GTC_THR) {
        int r = i / VC, c = i % VC;
        Sb[r * VD2 + c] = f2b(S_in[(long long)r * VD + c]);
    }
    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage: prep f32 -> bf16 tiles; pads zero ----
        for (int i = t; i < n * KD; i += GTC_THR) {
            int tt = i / KD, r = i % KD;
            const long long base = ((long long)(c0 + tt) * NH_ + head);
            Qb[tt * KD2 + r] = f2b(Qs[base * 132 + r]);
            const float kv = Ks[base * 132 + r];
            Kb[tt * KD2 + r] = f2b(kv);
            Kt[r * CD2 + tt] = f2b(kv);
        }
        for (int i = t; i < n * VC; i += GTC_THR) {
            int tt = i / VC, c = i % VC;
            const long long base = ((long long)(c0 + tt) * NH_ + head);
            Vb[tt * VD2 + c] = f2b(Vs[base * 128 + bb0 + c]);
        }
        __syncthreads();
        for (int i = t; i < (C - n) * KD; i += GTC_THR) {
            int tt = n + i / KD, r = i % KD;
            Qb[tt * KD2 + r] = f2b(0.f); Kb[tt * KD2 + r] = f2b(0.f); Kt[r * CD2 + tt] = f2b(0.f);
        }
        for (int i = t; i < (C - n) * VC; i += GTC_THR) {
            Vb[(n + i / VC) * VD2 + (i % VC)] = f2b(0.f);
        }
        __syncthreads();
        for (int i = t; i < C; i += GTC_THR) {
            if (i < n) {
                const long long base = ((long long)(c0 + i) * NH_ + head);
                bt[i] = Ps[base * 2 + 0]; lg[i + 1] = Ps[base * 2 + 1];
            } else { bt[i] = 0.f; lg[i + 1] = 0.f; }
        }
        __syncthreads();
        if (warp == 0 && lane < C) {              // inclusive scan (one warp)
            float v = lg[lane + 1];
#pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if ((int)lane >= o) v += u;
            }
            lg[lane + 1] = v;
        }
        if (t == 0) lg[0] = 0.f;
        __syncthreads();

        // ---- A / D grams (mma, out 32x32 = 8 tiles, one per warp; k=128) ----
        {
            const int mt = warp >> 2, nt = warp & 3;
            float accA[4] = {0,0,0,0}, accD[4] = {0,0,0,0};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq, s_row = nt * 8 + g;
                unsigned aK[4] = {
                    *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                    *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
                unsigned aQ[4] = {
                    *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                    *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
                unsigned bK[2] = {
                    gtc_pack2(b2f(Kb[s_row * KD2 + cc]), b2f(Kb[s_row * KD2 + cc + 1])),
                    gtc_pack2(b2f(Kb[s_row * KD2 + cc + 2]), b2f(Kb[s_row * KD2 + cc + 3]))};
                gtc_mma(accA, aK, bK);
                gtc_mma(accD, aQ, bK);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = mt * 16 + g + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                // S-A3-w: A[t,s] = (gamma_t / gamma_s)(k_t . k_s) — g_t S_{t-1} carries gamma_t, so the
                // exponent is lg[row + 1] (was lg[row] = gamma_{t-1}: one gate factor short per row; the
                // recorded "~2.2e-2 bf16 envelope" was mostly this — S-A3-v found it on the EXL3 twin).
                Af[row * C + scol] = (scol < row) ? accA[e] * __expf(lg[row + 1] - lg[scol + 1]) : 0.f;
                Db[row * CD2 + scol] =
                    (scol <= row && scol < n) ? f2b(accD[e] * __expf(lg[row + 1] - lg[scol + 1])) : f2b(0.f);
            }
        }
        __syncthreads();

        // ---- W = V - exp(lg[tt+1]) * (K S) (mma, out 32x64 = 2 tiles/warp; k=128) ----
        {
            const int mt = warp >> 2, nt0 = (warp & 3) * 2;
            float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aK[4] = {
                    *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                    *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bS[2] = {
                        gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aK, bS);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                    const float v = (row < n) ? b2f(Vb[row * VD2 + col]) : 0.f;
                    Wf[row * VC + col] = v - __expf(lg[row + 1]) * acc[j][e];
                }
        }
        __syncthreads();

        // ---- U = (I + diag(beta)A)^-1 (beta . W): fwd-subst in place, parallel cols ----
        for (int c = t; c < VC; c += GTC_THR) {
            for (int tt = 0; tt < n; tt++) {
                const float b = bt[tt];
                float acc = b * Wf[tt * VC + c];
                for (int s = 0; s < tt; s++) acc -= b * Af[tt * C + s] * Wf[s * VC + c];
                Wf[tt * VC + c] = acc;
            }
        }
        __syncthreads();

        // ---- materialize Ub (unscaled) and Ulb (lambda_s = exp(lg[n]-lg[s+1])) ----
        {
            const float gn = __expf(lg[n]);
            for (int i = t; i < C * VC; i += GTC_THR) {
                int s = i / VC, c = i % VC;
                const float u = (s < n) ? Wf[s * VC + c] : 0.f;
                Ub[s * VD2 + c] = f2b(u);
                Ulb[s * VD2 + c] = f2b(u * __expf(lg[n] - lg[s + 1]));
            }
            (void)gn;
        }
        __syncthreads();

        // ---- O = exp(lg[tt+1]) * (Q S) + (D U) (mma; QS k=128 then DU k=C accumulates) ----
        {
            const int mt = warp >> 2, nt0 = (warp & 3) * 2;
            float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aQ[4] = {
                    *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                    *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bS[2] = {
                        gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aQ, bS);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                    acc[j][e] = __expf(lg[row + 1]) * acc[j][e];
                }
            for (int ks = 0; ks < C; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aD[4] = {
                    *(const unsigned*)(Db + r0 * CD2 + cc), *(const unsigned*)(Db + r1 * CD2 + cc),
                    *(const unsigned*)(Db + r0 * CD2 + cc + 2), *(const unsigned*)(Db + r1 * CD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bU[2] = {
                        gtc_pack2(b2f(Ub[(ks + 4 * tq) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Ub[(ks + 4 * tq + 2) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aD, bU);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    if (row < n) {
                        const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                        core[(long long)(c0 + row) * (NH_ * 128) + head * 128 + bb0 + col] = f2b(acc[j][e]);
                    }
                }
        }
        __syncthreads();

        // ---- S' = gamma S + Kt . Ulb  (mma out KD x VC = 8 tiles/warp; k=C) ----
        {
            const int mt = warp;                   // 8 m-tiles of 16 kd-rows
            const float gam = __expf(lg[n]);
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    acc[j][e] = gam * b2f(Sb[row * VD2 + col]);
                }
            for (int ks = 0; ks < C; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aKt[4] = {
                    *(const unsigned*)(Kt + r0 * CD2 + cc), *(const unsigned*)(Kt + r1 * CD2 + cc),
                    *(const unsigned*)(Kt + r0 * CD2 + cc + 2), *(const unsigned*)(Kt + r1 * CD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int col = j * 8 + g;
                    unsigned bU[2] = {
                        gtc_pack2(b2f(Ulb[(ks + 4 * tq) * VD2 + col]), b2f(Ulb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Ulb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Ulb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aKt, bU);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sb[row * VD2 + col] = f2b(acc[j][e]);
                }
        }
        __syncthreads();
    }

    // writeback final state (bf16 -> f32)
    float* S_out = state + ((long long)head * KD) * VD + bb0;
    for (int i = t; i < KD * VC; i += GTC_THR) {
        int r = i / VC, c = i % VC;
        S_out[(long long)r * VD + c] = b2f(Sb[r * VD2 + c]);
    }
}


// ================== SPLIT CHUNKED GDN PREFILL (F8 P2c', 2026-09-06) ==================
// The fused gdn_chunk_tc_b serializes C=32-token chunk steps inside ONE block per (head,
// v-chunk): the heavy per-chunk WY/gram/subst work runs 333 iterations deep in a 48-block grid,
// so the GPU idles at ~13% occupancy for the whole scan. FlashInfer's decomposition (the
// recipe's 6.1 ms/layer vs our 37) splits the SERIAL part (the state recurrence — which the WY
// identity reduces to S' = gamma*S + K'^T V with K' row-scaled by lambda_s*beta_s, NO U/W
// dependency in exact math) from the per-chunk HEAVY work (grams/W/subst/O — embarrassingly
// parallel over chunks once every chunk's ENTRY state is materialized).
//
//   gdn_chunk_states_b  — kernel H, grid (nh * vchunks): serial walk, ONE lean mma stage per
//                         chunk (S' = gamma*S + Kt'·V, k=C=32), writing every chunk ENTRY state
//                         (bf16 — the SAME rounding the fused kernel's SMEM Sb applies each
//                         chunk boundary) to the scratch [chunk][head][KD][VD].
//   gdn_chunk_wuo_b     — kernel WUO, grid (chunks, nh, vchunks): the fused kernel's
//                         staging/A-D-grams/W/subst/Ub-Ulb/O sections VERBATIM, with Sb loaded
//                         from the scratch instead of carried. No state update, no writeback.
//
// Numerics: the state route changes rounding (direct lambda-beta-V vs the WY route through
// W->subst->U), prefill-only drift of the same class as chunked-vs-seq (~1e-4, tolerance 1e-2);
// decode/verify keep delta_step_prefill (bit-exact contract). Gate GB10_GDN_SPLIT (=0 restores
// the fused gdn_chunk_tc_b path).

#define GSC_C 32                    // chunk length — MUST match GTC_C (staging constants shared)
#define GSC_KD2 (128 + 8)
#define GSC_CD2 (GSC_C + 8)
#define GSC_VD2 (GTC_VC + 8)
#define GSC_THR 256

// --- H: serial state recurrence + per-chunk entry-state materialization ---
extern "C" __global__ __launch_bounds__(GSC_THR, 1) void gdn_chunk_states_b(
    float* state, const float* __restrict__ Ks, const float* __restrict__ Vs,
    const float* __restrict__ Ps, __nv_bfloat16* __restrict__ states_scratch,
    int kd_vd, int N_nkh)
{
    (void)kd_vd;                     // host-guaranteed 128/128; tiles compile-time
    const int N = N_nkh & 0xFFFFFF;
    const int ncb = GTC_VC == 64 ? 2 : 1;
    const int NH_ = (int)(gridDim.x / ncb);
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * GTC_VC;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;

    extern __shared__ unsigned char gsc_dyn[];
    __nv_bfloat16* Sb = (__nv_bfloat16*)gsc_dyn;                 // [KD][VD2] carried state
    __nv_bfloat16* Kt = Sb + 128 * GSC_VD2;                      // [KD][CD2] scaled K^T
    __nv_bfloat16* Vb = Kt + 128 * GSC_CD2;                      // [C][VD2]
    float* lam = (float*)(Vb + GSC_C * GSC_VD2);                 // [C]   lambda_s * beta_s
    float* bt = lam + GSC_C;                                     // [C]
    float* lg = bt + GSC_C;                                      // [C+1] log-decay prefix

    // load entry state of chunk 0 (bf16 rounding matches the fused kernel's SMEM carry)
    const float* S_in = state + ((long long)head * 128) * 128 + bb0;
    for (int i = t; i < 128 * GTC_VC; i += GSC_THR) {
        int r = i / GTC_VC, c = i % GTC_VC;
        Sb[r * GSC_VD2 + c] = f2b(S_in[(long long)r * 128 + c]);
    }

    const int nchunks = (N + GSC_C - 1) / GSC_C;
    for (int c0 = 0; c0 < N; c0 += GSC_C) {
        const int ci = c0 / GSC_C;
        const int n = min(GSC_C, N - c0);
        // ---- stage: scaled K^T, V, beta, log-decays ----
        for (int i = t; i < n * 128; i += GSC_THR) {
            int s = i / 128, r = i % 128;
            const long long base = ((long long)(c0 + s) * NH_ + head);
            Kt[r * GSC_CD2 + s] = f2b(Ks[base * 132 + r]);
        }
        for (int i = t; i < n * GTC_VC; i += GSC_THR) {
            int s = i / GTC_VC, c = i % GTC_VC;
            const long long base = ((long long)(c0 + s) * NH_ + head);
            Vb[s * GSC_VD2 + c] = f2b(Vs[base * 128 + bb0 + c]);
        }
        __syncthreads();
        for (int i = t; i < GSC_C; i += GSC_THR) {
            if (i < n) {
                const long long base = ((long long)(c0 + i) * NH_ + head);
                bt[i] = Ps[base * 2 + 0]; lg[i + 1] = Ps[base * 2 + 1];
            } else { bt[i] = 0.f; lg[i + 1] = 0.f; }
        }
        __syncthreads();
        if (warp == 0 && lane < GSC_C) {          // inclusive scan (same as the fused kernel)
            float v = lg[lane + 1];
#pragma unroll
            for (int o = 1; o < GSC_C; o <<= 1) {
                float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if ((int)lane >= o) v += u;
            }
            lg[lane + 1] = v;
        }
        if (t == 0) lg[0] = 0.f;
        __syncthreads();
        const float gn = __expf(lg[n]);
        for (int i = t; i < GSC_C; i += GSC_THR)
            lam[i] = (i < n) ? __expf(gn - lg[i + 1]) * bt[i] : 0.f;
        // scale K^T rows by lam (the direct WY-form recurrence operand)
        for (int i = t; i < n * 128; i += GSC_THR) {
            int s = i / 128, r = i % 128;
            Kt[r * GSC_CD2 + s] = f2b(b2f(Kt[r * GSC_CD2 + s]) * lam[s]);
        }
        // zero-pad the tail
        for (int i = t; i < (GSC_C - n) * 128; i += GSC_THR) {
            int s = n + i / 128, r = i % 128;
            Kt[r * GSC_CD2 + s] = f2b(0.f);
        }
        for (int i = t; i < (GSC_C - n) * GTC_VC; i += GSC_THR)
            Vb[(n + i / GTC_VC) * GSC_VD2 + (i % GTC_VC)] = f2b(0.f);
        __syncthreads();

        // ---- materialize the ENTRY state of this chunk ----
        __nv_bfloat16* S_c = states_scratch + ((long long)ci * NH_ + head) * (128 * 128) + bb0;
        for (int i = t; i < 128 * GTC_VC; i += GSC_THR) {
            int r = i / GTC_VC, c = i % GTC_VC;
            S_c[(long long)r * 128 + c] = Sb[r * GSC_VD2 + c];
        }

        // ---- S' = gamma*S + Kt'·V  (mma 128x64, k=C) — the ONLY math in the serial walk ----
        {
            const int mt = warp;                   // 8 m-tiles of 16 kd-rows
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    acc[j][e] = gn * b2f(Sb[row * GSC_VD2 + col]);
                }
            for (int ks = 0; ks < GSC_C; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aKt[4] = {
                    *(const unsigned*)(Kt + r0 * GSC_CD2 + cc), *(const unsigned*)(Kt + r1 * GSC_CD2 + cc),
                    *(const unsigned*)(Kt + r0 * GSC_CD2 + cc + 2), *(const unsigned*)(Kt + r1 * GSC_CD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int col = j * 8 + g;
                    unsigned bV[2] = {
                        gtc_pack2(b2f(Vb[(ks + 4 * tq) * GSC_VD2 + col]), b2f(Vb[(ks + 4 * tq + 1) * GSC_VD2 + col])),
                        gtc_pack2(b2f(Vb[(ks + 4 * tq + 2) * GSC_VD2 + col]), b2f(Vb[(ks + 4 * tq + 3) * GSC_VD2 + col]))};
                    gtc_mma(acc[j], aKt, bV);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sb[row * GSC_VD2 + col] = f2b(acc[j][e]);
                }
        }
        __syncthreads();
    }

    // final state writeback (bf16 -> f32), same as the fused kernel
    float* S_out = state + ((long long)head * 128) * 128 + bb0;
    for (int i = t; i < 128 * GTC_VC; i += GSC_THR) {
        int r = i / GTC_VC, c = i % GTC_VC;
        S_out[(long long)r * 128 + c] = b2f(Sb[r * GSC_VD2 + c]);
    }
}

// --- WUO: per-chunk grams/W/subst/O in PARALLEL over (chunk, head, vchunk) ---
extern "C" __global__ __launch_bounds__(GTC_THR, 1) void gdn_chunk_wuo_b(
    __nv_bfloat16* core, const float* __restrict__ Qs, const float* __restrict__ Ks,
    const float* __restrict__ Vs, const float* __restrict__ Ps,
    const __nv_bfloat16* __restrict__ states_scratch, int N_nkh)
{
    const int N = N_nkh & 0xFFFFFF;
    const int ncb = GTC_VC == 64 ? 2 : 1;
    const int NH_ = (int)gridDim.y;          // heads (gridDim.y = lin_nh; z = v-chunk)
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int bb0 = blockIdx.z * GTC_VC;
    const int c0 = chunk * GSC_C;
    const int n = min(GSC_C, N - c0);
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;
    const int C = GSC_C, VC = GTC_VC, KD2 = GSC_KD2, CD2 = GSC_CD2, VD2 = GSC_VD2;

    extern __shared__ unsigned char gwo_dyn[];
    __nv_bfloat16* Qb = (__nv_bfloat16*)gwo_dyn;                 // [C][KD2]
    __nv_bfloat16* Kb = Qb + C * KD2;                            // [C][KD2]
    __nv_bfloat16* Kt = Kb + C * KD2;                            // [KD][CD2]
    __nv_bfloat16* Vb = Kt + 128 * CD2;                          // [C][VD2]
    __nv_bfloat16* Db = Vb + C * VD2;                            // [C][CD2]
    __nv_bfloat16* Ub = Db + C * CD2;                            // [C][VD2]
    __nv_bfloat16* Ulb = Ub + C * VD2;                           // [C][VD2]
    __nv_bfloat16* Sb = Ulb + C * VD2;                           // [KD][VD2]  (from scratch)
    float* Af = (float*)(Sb + 128 * VD2);                        // [C][C]
    float* Wf = Af + C * C;                                      // [C][VC]
    float* Of = Wf + C * VC;                                     // [C][VC]  (unused, layout kept)
    float* bt = Of + C * VC;                                     // [C]
    float* lg = bt + C;                                          // [C+1]

    // ---- staging (verbatim from the fused kernel, chunk fixed) ----
    for (int i = t; i < n * 128; i += GTC_THR) {
        int tt = i / 128, r = i % 128;
        const long long base = ((long long)(c0 + tt) * NH_ + head);
        Qb[tt * KD2 + r] = f2b(Qs[base * 132 + r]);
        const float kv = Ks[base * 132 + r];
        Kb[tt * KD2 + r] = f2b(kv);
        Kt[r * CD2 + tt] = f2b(kv);
    }
    for (int i = t; i < n * VC; i += GTC_THR) {
        int tt = i / VC, c = i % VC;
        const long long base = ((long long)(c0 + tt) * NH_ + head);
        Vb[tt * VD2 + c] = f2b(Vs[base * 128 + bb0 + c]);
    }
    __syncthreads();
    for (int i = t; i < (C - n) * 128; i += GTC_THR) {
        int tt = n + i / 128, r = i % 128;
        Qb[tt * KD2 + r] = f2b(0.f); Kb[tt * KD2 + r] = f2b(0.f); Kt[r * CD2 + tt] = f2b(0.f);
    }
    for (int i = t; i < (C - n) * VC; i += GTC_THR)
        Vb[(n + i / VC) * VD2 + (i % VC)] = f2b(0.f);
    __syncthreads();
    for (int i = t; i < C; i += GTC_THR) {
        if (i < n) {
            const long long base = ((long long)(c0 + i) * NH_ + head);
            bt[i] = Ps[base * 2 + 0]; lg[i + 1] = Ps[base * 2 + 1];
        } else { bt[i] = 0.f; lg[i + 1] = 0.f; }
    }
    __syncthreads();
    if (warp == 0 && lane < C) {
        float v = lg[lane + 1];
#pragma unroll
        for (int o = 1; o < C; o <<= 1) {
            float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
            if ((int)lane >= o) v += u;
        }
        lg[lane + 1] = v;
    }
    if (t == 0) lg[0] = 0.f;
    __syncthreads();

    // ---- chunk ENTRY state from the H kernel's scratch ----
    const __nv_bfloat16* S_c = states_scratch + ((long long)chunk * NH_ + head) * (128 * 128) + bb0;
    for (int i = t; i < 128 * VC; i += GTC_THR) {
        int r = i / VC, c = i % VC;
        Sb[r * VD2 + c] = S_c[(long long)r * 128 + c];
    }
    __syncthreads();

    // ---- A / D grams (verbatim) ----
    {
        const int mt = warp >> 2, nt = warp & 3;
        float accA[4] = {0,0,0,0}, accD[4] = {0,0,0,0};
        for (int ks = 0; ks < 128; ks += 16) {
            const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq, s_row = nt * 8 + g;
            unsigned aK[4] = {
                *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
            unsigned aQ[4] = {
                *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
            unsigned bK[2] = {
                gtc_pack2(b2f(Kb[s_row * KD2 + cc]), b2f(Kb[s_row * KD2 + cc + 1])),
                gtc_pack2(b2f(Kb[s_row * KD2 + cc + 2]), b2f(Kb[s_row * KD2 + cc + 3]))};
            gtc_mma(accA, aK, bK);
            gtc_mma(accD, aQ, bK);
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = mt * 16 + g + 8 * (e >= 2);
            const int scol = nt * 8 + 2 * tq + (e & 1);
            // S-A3-w: this is the gamma_{t-1} off-by-one fixed in gdn_chunk_tc_b / gdn_chunk_prefill_b
            // (the exponent must be lg[row + 1]). NOT changed here: this split arm (GB10_GDN_SPLIT,
            // default off) is independently BROKEN on qwen4_exp — NaN state, "!!!!" output on both the
            // pre- and post-fix binaries — so an edit here could not be proven. Parked with the arm.
            Af[row * C + scol] = (scol < row) ? accA[e] * __expf(lg[row] - lg[scol + 1]) : 0.f;
            Db[row * CD2 + scol] =
                (scol <= row && scol < n) ? f2b(accD[e] * __expf(lg[row + 1] - lg[scol + 1])) : f2b(0.f);
        }
    }
    __syncthreads();

    // ---- W = V - exp(lg[tt+1]) * (K S) (verbatim; Sb from scratch) ----
    {
        const int mt = warp >> 2, nt0 = (warp & 3) * 2;
        float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int ks = 0; ks < 128; ks += 16) {
            const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
            unsigned aK[4] = {
                *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = (nt0 + j) * 8 + g;
                unsigned bS[2] = {
                    gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                    gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                gtc_mma(acc[j], aK, bS);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = mt * 16 + g + 8 * (e >= 2);
                const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                const float v = (row < n) ? b2f(Vb[row * VD2 + col]) : 0.f;
                Wf[row * VC + col] = v - __expf(lg[row + 1]) * acc[j][e];
            }
    }
    __syncthreads();

    // ---- U fwd-substitution (verbatim) ----
    for (int c = t; c < VC; c += GTC_THR) {
        for (int tt = 0; tt < n; tt++) {
            const float b = bt[tt];
            float acc = b * Wf[tt * VC + c];
            for (int s = 0; s < tt; s++) acc -= b * Af[tt * C + s] * Wf[s * VC + c];
            Wf[tt * VC + c] = acc;
        }
    }
    __syncthreads();

    // ---- Ub / Ulb (verbatim) ----
    {
        for (int i = t; i < C * VC; i += GTC_THR) {
            int s = i / VC, c = i % VC;
            const float u = (s < n) ? Wf[s * VC + c] : 0.f;
            Ub[s * VD2 + c] = f2b(u);
            Ulb[s * VD2 + c] = f2b(u * __expf(lg[n] - lg[s + 1]));
        }
    }
    __syncthreads();

    // ---- O = exp(lg[tt+1]) * (Q S) + (D U) (verbatim) ----
    {
        const int mt = warp >> 2, nt0 = (warp & 3) * 2;
        float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int ks = 0; ks < 128; ks += 16) {
            const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
            unsigned aQ[4] = {
                *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = (nt0 + j) * 8 + g;
                unsigned bS[2] = {
                    gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                    gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                gtc_mma(acc[j], aQ, bS);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = mt * 16 + g + 8 * (e >= 2);
                const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                acc[j][e] = __expf(lg[row + 1]) * acc[j][e];
            }
        for (int ks = 0; ks < C; ks += 16) {
            const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
            unsigned aD[4] = {
                *(const unsigned*)(Db + r0 * CD2 + cc), *(const unsigned*)(Db + r1 * CD2 + cc),
                *(const unsigned*)(Db + r0 * CD2 + cc + 2), *(const unsigned*)(Db + r1 * CD2 + cc + 2)};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int col = (nt0 + j) * 8 + g;
                unsigned bU[2] = {
                    gtc_pack2(b2f(Ub[(ks + 4 * tq) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 1) * VD2 + col])),
                    gtc_pack2(b2f(Ub[(ks + 4 * tq + 2) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 3) * VD2 + col]))};
                gtc_mma(acc[j], aD, bU);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = mt * 16 + g + 8 * (e >= 2);
                if (row < n) {
                    const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                    core[(long long)(c0 + row) * (NH_ * 128) + head * 128 + bb0 + col] = f2b(acc[j][e]);
                }
            }
    }
}

// Template body: OLDSTAGE=false is the serving path (P0 dense-scratch staging from gdn_prep_b);
// OLDSTAGE=true is a DIAGNOSTIC variant with the pre-P0 in-kernel staging (normalize + serial
// dots from qkv/b_in/a_in, scratch args ignored) — selected by GB10_GDN_OLDSTAGE at load, used
// with GB10_GDN_XCHECK to attribute chunked-vs-seq divergence to the staging rework vs the chunk
// algorithm itself. Dead branch compiles out; the serving kernel is unchanged.
template <bool OLDSTAGE>
__device__ __forceinline__ void gdn_chunk_prefill_body(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int ncb = vd / GDN_VCB;                    // value-column blocks (2 at vd=128)
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * GDN_VCB;
    const int VC = GDN_VCB;
    const int nh = (int)(gridDim.x / ncb);
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int t = threadIdx.x;
    const float kds = rsqrtf((float)kd);

    // F-padded row stride for Kt/Qt: kd(=128) is ~0 mod 32 banks => 32-way conflicts in the
    // A/D Gram loops. kd+4 padding: measured -35% kernel time, bit-identical output (F report).
    const int KDS = kd + 4;
    extern __shared__ float dyn[];
    float* Ssh = dyn;                                // kd*VC
    float* Kt  = Ssh + kd * VC;                      // C*KDS (normed, scaled)
    float* Qt  = Kt + GDN_CHUNK * KDS;               // C*KDS
    float* Vt  = Qt + GDN_CHUNK * KDS;               // C*VC
    float* Wsh = Vt + GDN_CHUNK * VC;                // C*VC (W then U in place)
    float* Ash = Wsh + GDN_CHUNK * VC;               // C*C
    float* Dsh = Ash + GDN_CHUNK * GDN_CHUNK;        // C*C
    float* bt  = Dsh + GDN_CHUNK * GDN_CHUNK;        // C   (beta)
    float* lg  = bt + GDN_CHUNK;                     // C+1 (log gamma cumsum)

    const long long head_off = (long long)head * kd * vd;
    float* S_ = state + head_off;                    // pre-offset (prefill: slot 0)

    // load this block's state columns
    if (t == 0) lg[0] = 0.f;   // log-gamma cumsum base (uninitialized smem = NaN otherwise)
    for (int i = t; i < kd * VC; i += blockDim.x) {
        int r = i / VC, c = i % VC;
        Ssh[i] = S_[(long long)r * vd + bb0 + c];
    }
    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += GDN_CHUNK) {
        const int n = min(GDN_CHUNK, N - c0);
        if (OLDSTAGE) {
            // pre-P0 staging (diagnostic reference): in-kernel normalize, serial dots
            for (int tt = t; tt < n; tt += blockDim.x) {
                const __nv_bfloat16* col = qkv + (long long)(c0 + tt) * qkv_stride;
                const __nv_bfloat16* q_src = col + key_head * kd;
                const __nv_bfloat16* k_src = col + key_dim + key_head * kd;
                const __nv_bfloat16* v_src = col + 2 * key_dim + head * vd + bb0;
                float qs2 = 0.f, ks2 = 0.f;
                for (int r = 0; r < kd; r++) {
                    float qv = b2f(q_src[r]), kv = b2f(k_src[r]);
                    Qt[tt * KDS + r] = qv; Kt[tt * KDS + r] = kv;
                    qs2 += qv * qv; ks2 += kv * kv;
                }
                float qn = rsqrtf(qs2 + 1e-6f) * kds, kn = rsqrtf(ks2 + 1e-6f);
                for (int r = 0; r < kd; r++) {
                    Qt[tt * KDS + r] *= qn; Kt[tt * KDS + r] *= kn;
                }
                for (int c = 0; c < VC; c++) Vt[tt * VC + c] = b2f(v_src[c]);
                float sp = b2f(a_in[(long long)(c0 + tt) * ba_stride + head]) + dt_bias[head];
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                bt[tt] = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)(c0 + tt) * ba_stride + head])));
                lg[tt + 1] = -__expf(a_log[head]) * sp;   // raw log-g (cumsum below — SERIAL dep)
            }
        } else {
        // P0 staging: dense scratch from gdn_prep_b — coalesced f32 copies, no math.
        for (int i = t; i < n * kd; i += blockDim.x) {
            int tt = i / kd, r = i % kd;
            long long base = ((long long)(c0 + tt) * nh + head);
            Qt[tt * KDS + r] = Qs[base * KDS + r];
            Kt[tt * KDS + r] = Ks[base * KDS + r];
        }
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            long long base = ((long long)(c0 + tt) * nh + head);
            Vt[tt * VC + c] = Vs[base * vd + bb0 + c];
            if (c == 0) {
                bt[tt]       = Ps[base * 2 + 0];
                lg[tt + 1]   = Ps[base * 2 + 1];        // raw log-g (cumsum below)
            }
        }
        }
        __syncthreads();
        if (t == 0) {                                   // exclusive cumsum, single thread
            for (int i = 0; i < n; i++) lg[i + 1] += lg[i];
        }
        __syncthreads();
        // A = decay-weighted (k.k) Gram; D = decay-weighted (q.k) Gram — DIFFERENT left operands
        // (B3 port bug found by expert review: D must use Qt, not Kt).
        for (int i = t; i < n * n; i += blockDim.x) {
            int tt = i / n, s = i % n;
            float a = 0.f, d = 0.f;
            if (s < tt) {
                for (int r = 0; r < kd; r++) a += Kt[tt * KDS + r] * Kt[s * KDS + r];
                for (int r = 0; r < kd; r++) d += Qt[tt * KDS + r] * Kt[s * KDS + r];
                a *= __expf(lg[tt + 1] - lg[s + 1]);   // S-A3-w: gamma_t / gamma_s (was gamma_{t-1})
            } else if (s == tt) {
                for (int r = 0; r < kd; r++) d += Qt[tt * KDS + r] * Kt[s * KDS + r];
            }
            Ash[tt * GDN_CHUNK + s] = a;
            if (s <= tt) Dsh[tt * GDN_CHUNK + s] = d * __expf(lg[tt + 1] - lg[s + 1]);
            else        Dsh[tt * GDN_CHUNK + s] = 0.f;
        }
        __syncthreads();
        // W = V - Gamma K S_in (parallel over (t,c))
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            float ks = 0.f;
            for (int r = 0; r < kd; r++) ks += Kt[tt * KDS + r] * Ssh[r * VC + c];
            Wsh[i] = Vt[tt * VC + c] - __expf(lg[tt + 1]) * ks;
        }
        __syncthreads();
        // U = (I + diag(beta)A)^-1 diag(beta) W (fwd-subst, parallel over c)
        for (int c = t; c < VC; c += blockDim.x) {
            for (int tt = 0; tt < n; tt++) {
                float b = bt[tt];
                float acc = b * Wsh[tt * VC + c];
                for (int s = 0; s < tt; s++) acc -= Ash[tt * GDN_CHUNK + s] * b * Wsh[s * VC + c];
                Wsh[tt * VC + c] = acc;
            }
        }
        __syncthreads();
        // O = D U + Gamma (Q S_in)
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            float qs = 0.f;
            for (int r = 0; r < kd; r++) qs += Qt[tt * KDS + r] * Ssh[r * VC + c];
            float acc = __expf(lg[tt + 1]) * qs;
            for (int s = 0; s <= tt; s++) acc += Dsh[tt * GDN_CHUNK + s] * Wsh[s * VC + c];
            core[(long long)(c0 + tt) * (nh * vd) + (long long)head * vd + bb0 + c] = f2b(acc);
        }
        __syncthreads();
        // S = gamma_C S + sum_s K_w[s] U[s]^T
        {
            float gC = __expf(lg[n]);
            for (int i = t; i < kd * VC; i += blockDim.x) {
                int r = i / VC, c = i % VC;
                float acc = gC * Ssh[i];
                for (int s = 0; s < n; s++)
                    acc += __expf(lg[n] - lg[s + 1]) * Kt[s * KDS + r] * Wsh[s * VC + c];
                Ssh[i] = acc;
            }
            __syncthreads();
        }
    }
    for (int i = t; i < kd * VC; i += blockDim.x) {
        int r = i / VC, c = i % VC;
        S_[(long long)r * vd + bb0 + c] = Ssh[i];
    }
}

extern "C" __global__ void gdn_chunk_prefill_b(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    gdn_chunk_prefill_body<false>(core, qkv, state, b_in, a_in, qkv_ba_stride, kd_vd,
                                  a_log, dt_bias, N_nkh, Qs, Ks, Vs, Ps);
}

// Diagnostic twin (same signature/launch config; scratch args ignored): pre-P0 staging.
extern "C" __global__ void gdn_chunk_prefill_b_oldstage(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    gdn_chunk_prefill_body<true>(core, qkv, state, b_in, a_in, qkv_ba_stride, kd_vd,
                                 a_log, dt_bias, N_nkh, Qs, Ks, Vs, Ps);
}

extern "C" __global__ void compact_kv_b(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    __nv_bfloat16* ks, __nv_bfloat16* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = len * nkv * hd;
    if (idx >= total) return;
    int k = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int dv = rem % hd;
    long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * hd + dv;
    long long soff = ((long long)k * nkv + h) * hd + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

// ===================================================================================================
// 4-bit KV cache (GB10_KV_QUANT=1) — per-16-element affine quantization, deterministic per position.
//
// Layout per (slot, kvh, position): hd/16 blocks × 9 B = [8 B codes (16×4b)] [1 B e4m3 scale].
// The SAME e4m3 codec the NVFP4 weights use (round to nearest, low nibble = element 2j, high = 2j+1).
// Per-block scale = e4m3(amax/7), codes = round(x / e4m3(amax/7)) clamped to [-7, 7]. Deterministic and
// position-local: the packed form of position p depends only on (kvh, p), never on batch or the
// reduction structure — so decode, verify, and the prefill scratch all dequantize to the SAME
// values and the lossless-MTP contract (bitwise verify==decode) is preserved by construction.
//
// q = round(x * 7/amax); x' = q * amax/7. Error ~ amax/14 RMS per block — the P7 trade: 3.56x
// fewer KV bytes for a small attention-input perturbation (gated vs bf16 KV; see HY3 docs).
// ===================================================================================================
#define KVQ_BLK 16
#define KVQ_ROW_BYTES(hd) (((hd) / KVQ_BLK) * 12)   // [8 B codes][1 B e4m3 scale][3 B pad] — keeps u16/u32 alignment

// e4m3 codec (defined with the other quantization helpers further down; the KV pack/unpack
// and the 4-bit attention reader need it here).
__device__ __forceinline__ float e4m3_f(uint8_t b);
__device__ __forceinline__ uint8_t f32_to_e4m3(float f);

__device__ __forceinline__ void kvq16_pack(const __nv_bfloat16* x, unsigned char* out) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) amax = fmaxf(amax, fabsf(b2f(x[i])));
    // Round the scale to e4m3 FIRST and code with the rounded value (the weight quantizer's
    // convention, quant.rs). Coding with the exact amax/7 while the readers dequantize with the
    // e4m3-rounded byte gave every block a common-mode gain error of up to +-6.25% (3 mantissa
    // bits) on top of the +-0.5 LSB code error: +2..6% relL2 for nothing. f32_to_e4m3 is the exact
    // inverse of e4m3_f on representable values, so the stored byte is the scale the codes used.
    const float s = amax > 0.f ? e4m3_f(f32_to_e4m3(amax / 7.0f)) : 0.f;
    const float inv_s = s > 0.f ? 1.0f / s : 0.f;
    unsigned char c[KVQ_BLK / 2];
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) {
        int lo = (int)lrintf(b2f(x[2 * j]) * inv_s);
        int hi = (int)lrintf(b2f(x[2 * j + 1]) * inv_s);
        lo = max(-7, min(7, lo)) + 7;
        hi = max(-7, min(7, hi)) + 7;
        c[j] = (unsigned char)(lo | (hi << 4));
    }
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) out[j] = c[j];
    out[KVQ_BLK / 2] = f32_to_e4m3(s);
}

__device__ __forceinline__ void kvq16_unpack(const unsigned char* in, float* x) {
    const float s = e4m3_f(in[KVQ_BLK / 2]);
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) {
        const int lo = (in[j] & 0xF) - 7;
        const int hi = (in[j] >> 4) - 7;
        x[2 * j] = (float)lo * s;
        x[2 * j + 1] = (float)hi * s;
    }
}

// write_kv_b_q4 — quantize + write one token's K/V (decode/verify path).
// One block per (b, kvh, block16): reads its 16 values, packs, writes 9 B. grid(B*nkv*(hd/16)).
extern "C" __global__ void write_kv_b_q4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = B * nkv * nb;
    if (idx >= total) return;
    const int b = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, k_cache + coff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + coff);
}

// write_kv_prefill_q4 — quantize + append N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_q4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        int stride, int nkv, int hd, int N, int pos_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = N * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, k_cache + coff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + coff);
}

// compact_kv_q4 — verbatim copy of packed rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; the packed form is position-local, so a byte copy suffices).
extern "C" __global__ void compact_kv_q4(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    const int rb = KVQ_ROW_BYTES(hd);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * rb;
    if (idx >= total) return;
    const int k = idx / (nkv * rb);
    const int rem = idx % (nkv * rb);
    const int h = rem / rb;
    const int dv = rem % rb;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * rb + dv;
    const long long soff = ((long long)k * nkv + h) * rb + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

// dequant_kv_q4 — expand packed KV to bf16 scratch for the prefill paths, in CACHE layout
// ([kvh][pos][hd], the layout attn_prefill_tiled / gqa_attn_prefill index). The formula is
// IDENTICAL to the splitk reader's, so prefill and decode see the same values.
// `out_stride` is the OUTPUT head stride: n_pos for a one-shot scratch (the old behavior — every
// call packs its own [0, n_pos) range), or the persistent mirror's position budget for the
// incremental path (E2 Fix 2), where the host shifts `out` by pos_start*hd so rows land at
// (h*out_stride + pos_start + t) — the shift is head-uniform, so a base shift suffices.
extern "C" __global__ void dequant_kv_q4(__nv_bfloat16* out, const unsigned char* cache,
        int nkv, int stride, int hd, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = n_pos * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const unsigned char* in = cache + crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    float x[KVQ_BLK];
    kvq16_unpack(in, x);
    __nv_bfloat16* o = out + ((long long)h * out_stride + t) * hd + blk * KVQ_BLK;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) o[i] = f2b(x[i]);
}

