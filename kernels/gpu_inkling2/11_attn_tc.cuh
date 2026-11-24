// gpu_inkling2/11_attn_tc.cuh — tensor-core attention (K1). Section of gpu_inkling2.cu; same contract as
// ink_attn (10_attn.cuh): same args, grid (nsplit, 8 kv heads, ceil(M / 16)), same split / finalize / partial
// outputs (ink_attn_combine merges), same masking and skip rule. Block 96 threads = 3 warps; warp w owns the
// query rows 16w..16w+15 of the CTA's 48 = 16 tokens x 3 heads (row r = token r / 3, head 3 kvh + r % 3).
//
// Per 64-key subtile (K / V f16 in shared memory, rows outside [lo, hi) zero-filled by cp.async):
//   S    = Q K^T with mma.m16n8k16 f16, Q split EXACTLY into f16 hi + lo (two MMAs: Q keeps ~22 bits, the
//          logits reach |s| ~ 45 after the 22.6 clip, so a single f16 Q would cost ~1e-3 in the logits);
//   bias = R . proj[:, dist] for the 80 distances the CTA's rows can see (bf16 MMA, R split hi + lo, proj
//          stored bf16 = exact), staged per warp in shared memory and gathered per score (1 LDS / score);
//   online softmax per row in fp32 (quad shuffles, fixed order); a row with no visible key in the subtile
//   keeps its state untouched (the batch-invariance skip rule of ink_attn);
//   O   += P V with P split f16 hi + lo (P in [0, 1]).
// Every row's arithmetic depends only on its own query / position and the absolute subtile grid, so row 0 of
// an M-row call is bit-identical to the M = 1 call.

#define IKT_KS 136           // K / V smem row stride (halves): 272 B, conflict-free ldmatrix
#define IKT_PJS 24           // proj smem row stride (halves)
#define IKT_NB 80            // bias columns per subtile (64 + 15 + 1)

__device__ __forceinline__ uint32_t ika_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ika_ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ika_smem_u32(p)));
}
__device__ __forceinline__ void ika_ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ika_smem_u32(p)));
}
__device__ __forceinline__ void ika_cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(ika_smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ika_mma_h(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// fp32 pair -> exact f16 hi / lo halves (hi = RN(x), lo = RN(x - hi))
__device__ __forceinline__ void ika_split_h(float x, float y, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x, y);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(x - hf.x, y - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
__device__ __forceinline__ void ika_split_b(float x, float y, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    const float2 hf = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(x - hf.x, y - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}

// dynamic smem: K + V (64 x 136 halves each) + proj (80 x 24 halves) + bias (3 warps x 16 x 80 fp32)
// + the Q lo halves (3 warps x 16 x 136 halves; Q hi stays in registers: 255-register budget)
#define IKT_SMEM (2 * 64 * IKT_KS * 2 + IKT_NB * IKT_PJS * 2 + 3 * 16 * IKT_NB * 4 + 3 * 16 * IKT_KS * 2)

extern "C" __global__ __launch_bounds__(96) void ink_attn_tc(const InkAttnArgs a) {
    INK2_PDL_ENTRY();
    extern __shared__ __align__(16) unsigned char smraw[];
    __half* sK = reinterpret_cast<__half*>(smraw);
    __half* sV = sK + 64 * IKT_KS;
    uint16_t* sPJ = reinterpret_cast<uint16_t*>(sV + 64 * IKT_KS);
    float* sB = reinterpret_cast<float*>(sPJ + IKT_NB * IKT_PJS);
    __half* sQl = reinterpret_cast<__half*>(sB + 3 * 16 * IKT_NB);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int nsplit = gridDim.x, kvh = blockIdx.y;
    const int t0 = blockIdx.z * 16;
    const int mt = min(16, a.M - t0);
    const int nq = 3 * mt;
    const int P0 = a.p0 + t0, P1 = P0 + mt - 1;
    const int kmin = a.window ? max(0, P0 - a.window + 1) : 0;
    const long long cs = (long long)(a.c0 + blockIdx.x) * a.chunk;
    const int lo = (int)max((long long)kmin, cs);
    const int hi = (int)min((long long)P1 + 1, cs + a.chunk);
    const float scale = rsqrtf((float)IKA_HD);
    float* myB = sB + warp * 16 * IKT_NB;
    const bool wlive = 16 * warp < nq;

    // this thread's two rows (gid, gid + 8) of the warp's 16: query index, position
    const int ra = 16 * warp + gid, rb = ra + 8;
    const int qa = min(ra, nq - 1), qb = min(rb, nq - 1);
    const int Pa = P0 + ra / 3, Pb = P0 + rb / 3;
    // Q fragments (hi / lo) for the 8 k-steps, R fragments (hi / lo) for the bias MMA
    uint32_t qh[8][4], rh[4], rl[4];
    __half* myQl = sQl + warp * 16 * IKT_KS;
    {
        const float* qra = a.Qf + ((size_t)(t0 + qa / 3) * 24 + 3 * kvh + qa % 3) * IKA_HD;
        const float* qrb = a.Qf + ((size_t)(t0 + qb / 3) * 24 + 3 * kvh + qb % 3) * IKA_HD;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(qra + 16 * kk + 2 * tig);
            const float2 x1 = *reinterpret_cast<const float2*>(qrb + 16 * kk + 2 * tig);
            const float2 x2 = *reinterpret_cast<const float2*>(qra + 16 * kk + 8 + 2 * tig);
            const float2 x3 = *reinterpret_cast<const float2*>(qrb + 16 * kk + 8 + 2 * tig);
            uint32_t l0, l1, l2, l3;
            ika_split_h(x0.x, x0.y, qh[kk][0], l0);
            ika_split_h(x1.x, x1.y, qh[kk][1], l1);
            ika_split_h(x2.x, x2.y, qh[kk][2], l2);
            ika_split_h(x3.x, x3.y, qh[kk][3], l3);
            *reinterpret_cast<uint32_t*>(myQl + gid * IKT_KS + 16 * kk + 2 * tig) = l0;
            *reinterpret_cast<uint32_t*>(myQl + (gid + 8) * IKT_KS + 16 * kk + 2 * tig) = l1;
            *reinterpret_cast<uint32_t*>(myQl + gid * IKT_KS + 16 * kk + 8 + 2 * tig) = l2;
            *reinterpret_cast<uint32_t*>(myQl + (gid + 8) * IKT_KS + 16 * kk + 8 + 2 * tig) = l3;
        }
        const float* rra = a.Rf + ((size_t)(t0 + qa / 3) * 24 + 3 * kvh + qa % 3) * IKA_DREL;
        const float* rrb = a.Rf + ((size_t)(t0 + qb / 3) * 24 + 3 * kvh + qb % 3) * IKA_DREL;
        const float2 y0 = *reinterpret_cast<const float2*>(rra + 2 * tig);
        const float2 y1 = *reinterpret_cast<const float2*>(rrb + 2 * tig);
        const float2 y2 = *reinterpret_cast<const float2*>(rra + 8 + 2 * tig);
        const float2 y3 = *reinterpret_cast<const float2*>(rrb + 8 + 2 * tig);
        ika_split_b(y0.x, y0.y, rh[0], rl[0]);
        ika_split_b(y1.x, y1.y, rh[1], rl[1]);
        ika_split_b(y2.x, y2.y, rh[2], rl[2]);
        ika_split_b(y3.x, y3.y, rh[3], rl[3]);
    }
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; ++n)
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[n][r] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

    for (int kb = (lo / IKA_SUB) * IKA_SUB; kb < hi; kb += IKA_SUB) {
        __syncthreads();
        // K / V rows kb..kb+63 (16-B chunks; invalid rows zero-filled)
        for (int i = threadIdx.x; i < 64 * 16; i += 96) {
            const int r = i >> 4, ch = i & 15, kp = kb + r;
            const bool ok = kp >= lo && kp < hi;
            const size_t g = (size_t)((ok ? kp : lo) & a.cap_mask) * 1024 + kvh * IKA_HD + 8 * ch;
            ika_cp16(sK + r * IKT_KS + 8 * ch, a.kc + g, ok);
            ika_cp16(sV + r * IKT_KS + 8 * ch, a.vc + g, ok);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        const int dmin = P0 - (kb + IKA_SUB - 1);
        for (int i = threadIdx.x; i < IKT_NB * IKA_DREL; i += 96) {
            const int c = i >> 4, dd = i & 15, dist = dmin + c;
            const float v = (dist >= 0 && dist < a.ext) ? a.proj[(size_t)dd * a.ext + dist] : 0.f;
            sPJ[c * IKT_PJS + dd] = (uint16_t)(__float_as_uint(v) >> 16);   // proj is bf16-valued: exact
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (!wlive) continue;
        // bias tile [16 rows][80 distances] -> warp-private smem
        #pragma unroll
        for (int n = 0; n < IKT_NB / 8; n += 2) {
            uint32_t b0, b1, b2, b3;
            ika_ldsm4(b0, b1, b2, b3, sPJ + (8 * n + (lane >> 4) * 8 + (lane & 7)) * IKT_PJS + ((lane >> 3) & 1) * 8);
            float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
            ika_mma_bf(c0, rh[0], rh[1], rh[2], rh[3], b0, b1);
            ika_mma_bf(c0, rl[0], rl[1], rl[2], rl[3], b0, b1);
            ika_mma_bf(c1, rh[0], rh[1], rh[2], rh[3], b2, b3);
            ika_mma_bf(c1, rl[0], rl[1], rl[2], rl[3], b2, b3);
            *reinterpret_cast<float2*>(myB + gid * IKT_NB + 8 * n + 2 * tig) = make_float2(c0[0], c0[1]);
            *reinterpret_cast<float2*>(myB + (gid + 8) * IKT_NB + 8 * n + 2 * tig) = make_float2(c0[2], c0[3]);
            *reinterpret_cast<float2*>(myB + gid * IKT_NB + 8 * n + 8 + 2 * tig) = make_float2(c1[0], c1[1]);
            *reinterpret_cast<float2*>(myB + (gid + 8) * IKT_NB + 8 * n + 8 + 2 * tig) = make_float2(c1[2], c1[3]);
        }
        __syncwarp();
        // S = Q K^T
        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int r = 0; r < 4; ++r) s[n][r] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t ql[4];
            ika_ldsm4(ql[0], ql[1], ql[2], ql[3], myQl + ((lane & 7) + ((lane >> 3) & 1) * 8) * IKT_KS + 16 * kk + (lane >> 4) * 8);
            #pragma unroll
            for (int n = 0; n < 8; n += 2) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4(b0, b1, b2, b3, sK + (8 * n + (lane >> 4) * 8 + (lane & 7)) * IKT_KS + 16 * kk + ((lane >> 3) & 1) * 8);
                ika_mma_h(s[n], qh[kk][0], qh[kk][1], qh[kk][2], qh[kk][3], b0, b1);
                ika_mma_h(s[n], ql[0], ql[1], ql[2], ql[3], b0, b1);
                ika_mma_h(s[n + 1], qh[kk][0], qh[kk][1], qh[kk][2], qh[kk][3], b2, b3);
                ika_mma_h(s[n + 1], ql[0], ql[1], ql[2], ql[3], b2, b3);
            }
        }
        // scale + bias + mask; row maxima
        float mxa = -INFINITY, mxb = -INFINITY;
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int kp = kb + 8 * n + 2 * tig + e;
                const bool kin = kp >= lo && kp < hi;
                const int da = Pa - kp, db = Pb - kp;
                const bool oka = kin && da >= 0 && (a.window == 0 || da < a.window);
                const bool okb = kin && db >= 0 && (a.window == 0 || db < a.window);
                s[n][e] = oka ? fmaf(s[n][e], scale, myB[gid * IKT_NB + (da - dmin)]) : -INFINITY;
                s[n][2 + e] = okb ? fmaf(s[n][2 + e], scale, myB[(gid + 8) * IKT_NB + (db - dmin)]) : -INFINITY;
                mxa = fmaxf(mxa, s[n][e]);
                mxb = fmaxf(mxb, s[n][2 + e]);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        // a row with nothing visible keeps its state (fold 1, p 0)
        const float mna = mxa == -INFINITY ? m_a : fmaxf(m_a, mxa);
        const float mnb = mxb == -INFINITY ? m_b : fmaxf(m_b, mxb);
        const float fa = mxa == -INFINITY ? 1.f : __expf(m_a - mna);
        const float fb = mxb == -INFINITY ? 1.f : __expf(m_b - mnb);
        float sa = 0.f, sb = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {   // (a row with nothing visible: p = 0, not exp(-inf - -inf))
            s[n][0] = mxa == -INFINITY ? 0.f : __expf(s[n][0] - mna); s[n][1] = mxa == -INFINITY ? 0.f : __expf(s[n][1] - mna);
            s[n][2] = mxb == -INFINITY ? 0.f : __expf(s[n][2] - mnb); s[n][3] = mxb == -INFINITY ? 0.f : __expf(s[n][3] - mnb);
            sa += s[n][0] + s[n][1];
            sb += s[n][2] + s[n][3];
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        if (mxa != -INFINITY) { l_a = l_a * fa + sa; m_a = mna; }
        if (mxb != -INFINITY) { l_b = l_b * fb + sb; m_b = mnb; }
        #pragma unroll
        for (int n = 0; n < 16; ++n) { o[n][0] *= fa; o[n][1] *= fa; o[n][2] *= fb; o[n][3] *= fb; }
        // O += P V   (P hi / lo f16; masked entries are exp(-inf) = 0)
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t ph[4], pl[4];
            ika_split_h(s[2 * kk][0], s[2 * kk][1], ph[0], pl[0]);
            ika_split_h(s[2 * kk][2], s[2 * kk][3], ph[1], pl[1]);
            ika_split_h(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[2], pl[2]);
            ika_split_h(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[3], pl[3]);
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4t(b0, b1, b2, b3, sV + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * IKT_KS + 8 * n + (lane >> 4) * 8);
                ika_mma_h(o[n], ph[0], ph[1], ph[2], ph[3], b0, b1);
                ika_mma_h(o[n], pl[0], pl[1], pl[2], pl[3], b0, b1);
                ika_mma_h(o[n + 1], ph[0], ph[1], ph[2], ph[3], b2, b3);
                ika_mma_h(o[n + 1], pl[0], pl[1], pl[2], pl[3], b2, b3);
            }
        }
    }
    if (!wlive) return;
    // epilogue: rows ra / rb (< nq)
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = half ? rb : ra;
        if (r >= nq) continue;
        const int t = t0 + r / 3, h = 3 * kvh + r % 3;
        const float mr = half ? m_b : m_a, lr = half ? l_b : l_a;
        if (nsplit == 1) {
            const float* y = a.Y + (size_t)t * a.ldy;
            const float sink = y[a.os + h];
            const float m = fmaxf(mr, sink);
            const float fold = __expf(mr - m);
            const float w = fold / (lr * fold + __expf(sink - m));
            if (a.og < 0) {
                float* fp = reinterpret_cast<float*>(a.out) + (size_t)t * 24 * IKA_HD + h * IKA_HD;
                #pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const int dcol = 8 * n + 2 * tig;
                    *reinterpret_cast<float2*>(fp + dcol) = make_float2(o[n][2 * half] * w, o[n][2 * half + 1] * w);
                }
            } else {
                uint16_t* op = a.out + (size_t)t * 24 * IKA_HD + h * IKA_HD;
                #pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const int dcol = 8 * n + 2 * tig;
                    const float2 gt = *reinterpret_cast<const float2*>(y + a.og + h * IKA_HD + dcol);
                    const float v0 = o[n][2 * half] * w * ika_sig(gt.x), v1 = o[n][2 * half + 1] * w * ika_sig(gt.y);
                    *reinterpret_cast<uint32_t*>(op + dcol) = (uint32_t)ika_f2b(v0) | ((uint32_t)ika_f2b(v1) << 16);
                }
            }
        } else {
            const size_t pi = ((size_t)t * 24 + h) * nsplit + blockIdx.x;
            if (tig == 0) { a.pm[pi] = mr; a.pl[pi] = lr; }
            #pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2*>(a.pacc + pi * IKA_HD + 8 * n + 2 * tig) = make_float2(o[n][2 * half], o[n][2 * half + 1]);
        }
    }
}

// S4: the prefill variant — TWO 16-token tiles per CTA (6 warps, 192 threads) sharing each K / V subtile (half the
// K / V traffic, twice the warps per SM: the 1-tile CTA's 67 KB of smem left 3 warps on an SM). grid (1, 8,
// ceil(M / 32)), smem IKT2_SMEM. Each row's arithmetic is the 1-tile kernel's: its visible keys, the absolute 64-key
// subtile grid and its bias columns are unchanged; the subtiles the wider CTA range adds hold no key the row can see,
// and such a subtile leaves a row's state untouched (fold 1, p 0), so the outputs are bit-identical (prefill only:
// nsplit 1).
#define IKT2_NB 96           // bias columns staged per subtile for 32 tokens (64 + 31 + 1)
#define IKT2_SMEM (2 * 64 * IKT_KS * 2 + IKT2_NB * IKT_PJS * 2 + 6 * 16 * IKT_NB * 4 + 6 * 16 * IKT_KS * 2)
// S7: the body is templated on SP. SP = false is ink_attn_tc2 (exact: Q and P split f16 hi + lo). SP = true is the
// OPT-IN single-pass variant ink_attn_tc2_sp (--inkling2-attn-single-pass): Q and P rounded ONCE to f16 (half the S
// and PV MMAs, no Q-lo staging); bias, softmax, masking and the epilogue unchanged. It changes the outputs (f16
// rounding of Q / P, ~1e-3 relative on the logits of a full-attention visit); the S7 probe reports its KL / top-1
// agreement against the exact path. Never the default.
template <bool SP>
__device__ __forceinline__ void ika_attn_tc2_body(const InkAttnArgs& a) {
    extern __shared__ __align__(16) unsigned char smraw[];
    __half* sK = reinterpret_cast<__half*>(smraw);
    __half* sV = sK + 64 * IKT_KS;
    uint16_t* sPJ = reinterpret_cast<uint16_t*>(sV + 64 * IKT_KS);
    float* sB = reinterpret_cast<float*>(sPJ + IKT2_NB * IKT_PJS);
    __half* sQl = reinterpret_cast<__half*>(sB + 6 * 16 * IKT_NB);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int nsplit = gridDim.x, kvh = blockIdx.y;
    const int tt = warp / 3;                                   // this warp's token tile (0 / 1)
    const int tc0 = blockIdx.z * 32;                            // the CTA's first token
    const int mtc = min(32, a.M - tc0);                         // the CTA's tokens
    const int t0 = tc0 + 16 * tt;
    const int mt = min(16, a.M - t0);                           // <= 0: an empty second tile (its warps only stage)
    const int nq = max(0, 3 * mt);
    const int P0c = a.p0 + tc0, P1 = P0c + mtc - 1;
    const int P0 = a.p0 + t0;
    const int kmin = a.window ? max(0, P0c - a.window + 1) : 0;
    const long long cs = (long long)(a.c0 + blockIdx.x) * a.chunk;
    const int lo = (int)max((long long)kmin, cs);
    const int hi = (int)min((long long)P1 + 1, cs + a.chunk);
    const float scale = rsqrtf((float)IKA_HD);
    float* myB = sB + warp * 16 * IKT_NB;
    const int wq = warp % 3;                                    // the warp's 16 rows of its tile's 48
    const bool wlive = 16 * wq < nq;

    // this thread's two rows (gid, gid + 8) of the warp's 16: query index, position
    const int ra = 16 * wq + gid, rb = ra + 8;
    const int qa = max(0, min(ra, nq - 1)), qb = max(0, min(rb, nq - 1));
    const int tl = mt > 0 ? t0 : tc0;                           // load rows of a live tile (an empty tile reads tile 0)
    const int Pa = P0 + ra / 3, Pb = P0 + rb / 3;
    // Q fragments (hi / lo) for the 8 k-steps, R fragments (hi / lo) for the bias MMA
    uint32_t qh[8][4], rh[4], rl[4];
    __half* myQl = sQl + warp * 16 * IKT_KS;
    {
        const float* qra = a.Qf + ((size_t)(tl + qa / 3) * 24 + 3 * kvh + qa % 3) * IKA_HD;
        const float* qrb = a.Qf + ((size_t)(tl + qb / 3) * 24 + 3 * kvh + qb % 3) * IKA_HD;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(qra + 16 * kk + 2 * tig);
            const float2 x1 = *reinterpret_cast<const float2*>(qrb + 16 * kk + 2 * tig);
            const float2 x2 = *reinterpret_cast<const float2*>(qra + 16 * kk + 8 + 2 * tig);
            const float2 x3 = *reinterpret_cast<const float2*>(qrb + 16 * kk + 8 + 2 * tig);
            uint32_t l0, l1, l2, l3;
            ika_split_h(x0.x, x0.y, qh[kk][0], l0);
            ika_split_h(x1.x, x1.y, qh[kk][1], l1);
            ika_split_h(x2.x, x2.y, qh[kk][2], l2);
            ika_split_h(x3.x, x3.y, qh[kk][3], l3);
            if constexpr (!SP) {
                *reinterpret_cast<uint32_t*>(myQl + gid * IKT_KS + 16 * kk + 2 * tig) = l0;
                *reinterpret_cast<uint32_t*>(myQl + (gid + 8) * IKT_KS + 16 * kk + 2 * tig) = l1;
                *reinterpret_cast<uint32_t*>(myQl + gid * IKT_KS + 16 * kk + 8 + 2 * tig) = l2;
                *reinterpret_cast<uint32_t*>(myQl + (gid + 8) * IKT_KS + 16 * kk + 8 + 2 * tig) = l3;
            }
        }
        const float* rra = a.Rf + ((size_t)(tl + qa / 3) * 24 + 3 * kvh + qa % 3) * IKA_DREL;
        const float* rrb = a.Rf + ((size_t)(tl + qb / 3) * 24 + 3 * kvh + qb % 3) * IKA_DREL;
        const float2 y0 = *reinterpret_cast<const float2*>(rra + 2 * tig);
        const float2 y1 = *reinterpret_cast<const float2*>(rrb + 2 * tig);
        const float2 y2 = *reinterpret_cast<const float2*>(rra + 8 + 2 * tig);
        const float2 y3 = *reinterpret_cast<const float2*>(rrb + 8 + 2 * tig);
        ika_split_b(y0.x, y0.y, rh[0], rl[0]);
        ika_split_b(y1.x, y1.y, rh[1], rl[1]);
        ika_split_b(y2.x, y2.y, rh[2], rl[2]);
        ika_split_b(y3.x, y3.y, rh[3], rl[3]);
    }
    float o[16][4];
    #pragma unroll
    for (int n = 0; n < 16; ++n)
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[n][r] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

    for (int kb = (lo / IKA_SUB) * IKA_SUB; kb < hi; kb += IKA_SUB) {
        __syncthreads();
        // K / V rows kb..kb+63 (16-B chunks; invalid rows zero-filled)
        for (int i = threadIdx.x; i < 64 * 16; i += 192) {
            const int r = i >> 4, ch = i & 15, kp = kb + r;
            const bool ok = kp >= lo && kp < hi;
            const size_t g = (size_t)((ok ? kp : lo) & a.cap_mask) * 1024 + kvh * IKA_HD + 8 * ch;
            ika_cp16(sK + r * IKT_KS + 8 * ch, a.kc + g, ok);
            ika_cp16(sV + r * IKT_KS + 8 * ch, a.vc + g, ok);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        const int dminc = P0c - (kb + IKA_SUB - 1);           // the CTA's columns: distances dminc .. dminc + 95
        const int dmin = dminc + 16 * tt;                       // this warp's tile: its 80 columns start 16 tt later
        for (int i = threadIdx.x; i < IKT2_NB * IKA_DREL; i += 192) {
            const int c = i >> 4, dd = i & 15, dist = dminc + c;
            const float v = (dist >= 0 && dist < a.ext) ? a.proj[(size_t)dd * a.ext + dist] : 0.f;
            sPJ[c * IKT_PJS + dd] = (uint16_t)(__float_as_uint(v) >> 16);   // proj is bf16-valued: exact
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (!wlive) continue;
        // bias tile [16 rows][80 distances] -> warp-private smem
        #pragma unroll
        for (int n = 0; n < IKT_NB / 8; n += 2) {
            uint32_t b0, b1, b2, b3;
            ika_ldsm4(b0, b1, b2, b3, sPJ + (16 * tt + 8 * n + (lane >> 4) * 8 + (lane & 7)) * IKT_PJS + ((lane >> 3) & 1) * 8);
            float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
            ika_mma_bf(c0, rh[0], rh[1], rh[2], rh[3], b0, b1);
            ika_mma_bf(c0, rl[0], rl[1], rl[2], rl[3], b0, b1);
            ika_mma_bf(c1, rh[0], rh[1], rh[2], rh[3], b2, b3);
            ika_mma_bf(c1, rl[0], rl[1], rl[2], rl[3], b2, b3);
            *reinterpret_cast<float2*>(myB + gid * IKT_NB + 8 * n + 2 * tig) = make_float2(c0[0], c0[1]);
            *reinterpret_cast<float2*>(myB + (gid + 8) * IKT_NB + 8 * n + 2 * tig) = make_float2(c0[2], c0[3]);
            *reinterpret_cast<float2*>(myB + gid * IKT_NB + 8 * n + 8 + 2 * tig) = make_float2(c1[0], c1[1]);
            *reinterpret_cast<float2*>(myB + (gid + 8) * IKT_NB + 8 * n + 8 + 2 * tig) = make_float2(c1[2], c1[3]);
        }
        __syncwarp();
        // S = Q K^T
        float s[8][4];
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int r = 0; r < 4; ++r) s[n][r] = 0.f;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t ql[4] = {0u, 0u, 0u, 0u};
            if constexpr (!SP)
                ika_ldsm4(ql[0], ql[1], ql[2], ql[3], myQl + ((lane & 7) + ((lane >> 3) & 1) * 8) * IKT_KS + 16 * kk + (lane >> 4) * 8);
            #pragma unroll
            for (int n = 0; n < 8; n += 2) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4(b0, b1, b2, b3, sK + (8 * n + (lane >> 4) * 8 + (lane & 7)) * IKT_KS + 16 * kk + ((lane >> 3) & 1) * 8);
                ika_mma_h(s[n], qh[kk][0], qh[kk][1], qh[kk][2], qh[kk][3], b0, b1);
                if constexpr (!SP) ika_mma_h(s[n], ql[0], ql[1], ql[2], ql[3], b0, b1);
                ika_mma_h(s[n + 1], qh[kk][0], qh[kk][1], qh[kk][2], qh[kk][3], b2, b3);
                if constexpr (!SP) ika_mma_h(s[n + 1], ql[0], ql[1], ql[2], ql[3], b2, b3);
            }
        }
        // scale + bias + mask; row maxima
        float mxa = -INFINITY, mxb = -INFINITY;
        #pragma unroll
        for (int n = 0; n < 8; ++n)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int kp = kb + 8 * n + 2 * tig + e;
                const bool kin = kp >= lo && kp < hi;
                const int da = Pa - kp, db = Pb - kp;
                const bool oka = kin && da >= 0 && (a.window == 0 || da < a.window);
                const bool okb = kin && db >= 0 && (a.window == 0 || db < a.window);
                s[n][e] = oka ? fmaf(s[n][e], scale, myB[gid * IKT_NB + (da - dmin)]) : -INFINITY;
                s[n][2 + e] = okb ? fmaf(s[n][2 + e], scale, myB[(gid + 8) * IKT_NB + (db - dmin)]) : -INFINITY;
                mxa = fmaxf(mxa, s[n][e]);
                mxb = fmaxf(mxb, s[n][2 + e]);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        // a row with nothing visible keeps its state (fold 1, p 0)
        const float mna = mxa == -INFINITY ? m_a : fmaxf(m_a, mxa);
        const float mnb = mxb == -INFINITY ? m_b : fmaxf(m_b, mxb);
        const float fa = mxa == -INFINITY ? 1.f : __expf(m_a - mna);
        const float fb = mxb == -INFINITY ? 1.f : __expf(m_b - mnb);
        float sa = 0.f, sb = 0.f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {   // (a row with nothing visible: p = 0, not exp(-inf - -inf))
            s[n][0] = mxa == -INFINITY ? 0.f : __expf(s[n][0] - mna); s[n][1] = mxa == -INFINITY ? 0.f : __expf(s[n][1] - mna);
            s[n][2] = mxb == -INFINITY ? 0.f : __expf(s[n][2] - mnb); s[n][3] = mxb == -INFINITY ? 0.f : __expf(s[n][3] - mnb);
            sa += s[n][0] + s[n][1];
            sb += s[n][2] + s[n][3];
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        if (mxa != -INFINITY) { l_a = l_a * fa + sa; m_a = mna; }
        if (mxb != -INFINITY) { l_b = l_b * fb + sb; m_b = mnb; }
        #pragma unroll
        for (int n = 0; n < 16; ++n) { o[n][0] *= fa; o[n][1] *= fa; o[n][2] *= fb; o[n][3] *= fb; }
        // O += P V   (P hi / lo f16; masked entries are exp(-inf) = 0)
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t ph[4], pl[4];
            ika_split_h(s[2 * kk][0], s[2 * kk][1], ph[0], pl[0]);
            ika_split_h(s[2 * kk][2], s[2 * kk][3], ph[1], pl[1]);
            ika_split_h(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[2], pl[2]);
            ika_split_h(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[3], pl[3]);
            #pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4t(b0, b1, b2, b3, sV + (16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * IKT_KS + 8 * n + (lane >> 4) * 8);
                ika_mma_h(o[n], ph[0], ph[1], ph[2], ph[3], b0, b1);
                if constexpr (!SP) ika_mma_h(o[n], pl[0], pl[1], pl[2], pl[3], b0, b1);
                ika_mma_h(o[n + 1], ph[0], ph[1], ph[2], ph[3], b2, b3);
                if constexpr (!SP) ika_mma_h(o[n + 1], pl[0], pl[1], pl[2], pl[3], b2, b3);
            }
        }
    }
    if (!wlive) return;
    // epilogue: rows ra / rb (< nq)
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = half ? rb : ra;
        if (r >= nq) continue;
        const int t = t0 + r / 3, h = 3 * kvh + r % 3;
        const float mr = half ? m_b : m_a, lr = half ? l_b : l_a;
        if (nsplit == 1) {
            const float* y = a.Y + (size_t)t * a.ldy;
            const float sink = y[a.os + h];
            const float m = fmaxf(mr, sink);
            const float fold = __expf(mr - m);
            const float w = fold / (lr * fold + __expf(sink - m));
            if (a.og < 0) {
                float* fp = reinterpret_cast<float*>(a.out) + (size_t)t * 24 * IKA_HD + h * IKA_HD;
                #pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const int dcol = 8 * n + 2 * tig;
                    *reinterpret_cast<float2*>(fp + dcol) = make_float2(o[n][2 * half] * w, o[n][2 * half + 1] * w);
                }
            } else {
                uint16_t* op = a.out + (size_t)t * 24 * IKA_HD + h * IKA_HD;
                #pragma unroll
                for (int n = 0; n < 16; ++n) {
                    const int dcol = 8 * n + 2 * tig;
                    const float2 gt = *reinterpret_cast<const float2*>(y + a.og + h * IKA_HD + dcol);
                    const float v0 = o[n][2 * half] * w * ika_sig(gt.x), v1 = o[n][2 * half + 1] * w * ika_sig(gt.y);
                    *reinterpret_cast<uint32_t*>(op + dcol) = (uint32_t)ika_f2b(v0) | ((uint32_t)ika_f2b(v1) << 16);
                }
            }
        } else {
            const size_t pi = ((size_t)t * 24 + h) * nsplit + blockIdx.x;
            if (tig == 0) { a.pm[pi] = mr; a.pl[pi] = lr; }
            #pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2*>(a.pacc + pi * IKA_HD + 8 * n + 2 * tig) = make_float2(o[n][2 * half], o[n][2 * half + 1]);
        }
    }
}
extern "C" __global__ __launch_bounds__(192) void ink_attn_tc2(const InkAttnArgs a) { INK2_PDL_ENTRY(); ika_attn_tc2_body<false>(a); }
extern "C" __global__ __launch_bounds__(192) void ink_attn_tc2_sp(const InkAttnArgs a) { INK2_PDL_ENTRY(); ika_attn_tc2_body<true>(a); }
