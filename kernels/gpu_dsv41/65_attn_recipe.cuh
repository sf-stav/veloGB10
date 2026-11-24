// =============================================================================
// kernels/gpu_dsv41/65_attn_recipe.cuh — band 60, leg PA1: the external reference recipe's split rule for the sparse
// attention (PLAN/DSV41_RX6_REPORT.md 2.1), OPT-IN (`--dsv41-attn-split recipe`; default `exact` = KA's canonical rule).
// IT CHANGES THE OUTPUT BITS against the exact rule (bf16(p) is rounded against each split's own running max, and the merge
// form differs); it is deterministic, and a row's bits do not depend on the other rows of a call of the same class.
//
// The rule, as specified there:
//   * BN = 32 keys a block; NBLK = 4 window blocks (list entries 32 b .. 32 b + 31 = positions win_end - 127 + ..) +
//     ceil(n_sel / 32) pick blocks (sel entries 32 (b - 4) ..);
//   * SPLITS = 8 (split s takes blocks s, s + 8, s + 16, ascending). Theirs uses 8 for decode rounds (<= 16 rows) and ONE
//     pass for prompt chunks (other bits); OUR flag runs 8 splits for every row of every call (the host launches pieces of
//     <= 16 rows), so batch invariance and decode == prefill hold under it. The kernel's one-split path (nsplit_max = 1)
//     is kept for measurement only: no served call uses it;
//   * per split: s = (dot(q[0:256], k[0:256]) + dot(q[256:512], k[256:512])) * scale (masked keys -inf); m from -1e30;
//     m' = max(m, rowmax s); a = exp(m - m'); p = exp(s - m'); l = l a + sum p; P = bf16(p); acc = acc a + P K (V = K);
//   * merge (one CTA per (row, head)): m = max_s m_s; l = sum_s l_s exp(m_s - m), acc = sum_s po_s exp(m_s - m), in split
//     order; l += exp(sink - m); y = bf16(acc / l); then OUR epilogue (inverse RoPE, act_quant, wo_a's rotated rows:
//     d41a_epi4), the same steps the exact rule runs after its division.
// What is ours, not theirs (their bits are NOT claimed): the key dequantisation (d41a_fetch / the same E4M3 / E2M1 values as
// the exact rule), the MMA tile order inside a dot, the 32-lane tree of sum p, IEEE division, our act_quant epilogue.
// Not covered by their rule -> the caller keeps the exact rule: DSpark extra keys, a window floor (win_lo).
//
// d41_attn_recipe: grid (rows x groups, SPLITS), 256 threads (8 warps), D41R_SMEM bytes of dynamic shared memory (no opt-in; 255 registers a thread:
// ONE CTA an SM, theirs is 2 at 128 threads x 255 registers: a deviation, see the PA1 report). Warp w: QK of key tile w % 4 over dim half w / 4; the softmax of heads w and w + 8; P.V over dims
// 64 w .. 64 w + 63. q's A fragments stay in registers for the whole row.
// d41_attn_merge_recipe: grid rows x groups x 16, 128 threads (d41_attn_merge's map).
// =============================================================================
#pragma once

#define D41R_BN 32
#define D41R_SM_SP 32768          // S partials [2 halves][16 heads][D41R_SP_LD] f32
#define D41R_SP_LD 36
#define D41R_SM_P 37376           // P [16 heads][D41R_P_LD] bf16
#define D41R_P_LD 40
#define D41R_SM_AL 38656          // per-head rescale of the block [16] f32
#define D41R_SM_ML 38720          // the split's (m, l) per head [32] f32
#define D41R_SM_KV 38848          // key valid flags [32]
#define D41R_SMEM 38976

// d41a_store for a 32-key block (thread t = key t / 8, part t % 8) with the flags at D41R_SM_KV.
__device__ __forceinline__ void d41r_store(unsigned char* sm, const d41a_raw& raw) {
    const int t = threadIdx.x, kk = t >> 3, p = t & 7;
    unsigned char* tp = sm;
    if (p == 0) reinterpret_cast<int*>(sm + D41R_SM_KV)[kk] = raw.kind != 0;
    if (raw.kind == 0) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(tp + d41a_swz(kk, p + 8 * i)) = z;
    } else if (raw.kind == 1) {
        const unsigned scw[4] = {raw.sc.x, raw.sc.y, raw.sc.z, raw.sc.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = p + 8 * i;
            const unsigned eb = (scw[i] >> (8 * (p >> 1))) & 0xFFu;
            const float s = __int_as_float((int)(eb << 23));
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

// Block b of a row: kind / base for d41a_fetch (window entries 32 b .., or sel entries 32 (b - 4) ..).
__device__ __forceinline__ void d41r_blk(int b, int& kind, int& base) { if (b < 4) { kind = 0; base = 32 * b; } else { kind = 1; base = 32 * (b - 4); } }

extern "C" __global__ void __launch_bounds__(256, 1) d41_attn_recipe(const __grid_constant__ d41_attn_args a) {
    extern __shared__ __align__(16) unsigned char sm[];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int nspl = a.nsplit_max, split = blockIdx.y;
    const int g = blockIdx.x % a.groups, row = blockIdx.x / a.groups;
    if (row >= a.rows) return;
    if (a.xflags & D41_ATTN_X_WPF) {        // before the wait: this split's window block(s) into L2 (theirs: CCTL.E.PF2)
        const int pos = __ldg(a.pos + row);
        const int wend = a.win_end ? __ldg(a.win_end + row) : pos;
        int ln = a.lane ? __ldg(a.lane + row) : 0;
        ln = ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln);
        const d41_attn_lane& L = a.lanes[ln];
        const int part = t & 7, kk = t >> 3;
        for (int b = split; b < 4 && L.swa_q != nullptr && L.slots > 0; b += nspl) {
            const long long p = (long long)wend - (D41_ATTN_WIN - 1) + 32 * b + kk;
            if (p < 0 || p >= L.lin_base) continue;
            const long long slot = (long long)((unsigned)p % (unsigned)L.slots);
            if (part < 4) asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_q + slot * D41_ATTN_D + part * 128));
            else if (part == 4 && L.swa_s != nullptr) asm volatile("prefetch.global.L2 [%0];" :: "l"(L.swa_s + slot * 16));
        }
    }
    D41_PDL_ENTRY();
    d41a_row r;
    r.row = row;
    r.pos = __ldg(a.pos + row);
    r.wend = a.win_end ? __ldg(a.win_end + row) : r.pos;
    r.wlo = 0;
    const int ln = a.lane ? __ldg(a.lane + row) : 0;
    r.L = &a.lanes[ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln)];
    int nsel = 0;
    if (a.sel != nullptr && r.L->kv != nullptr) {
        nsel = a.n_sel ? __ldg(a.n_sel + row) : a.sel_cap;
        nsel = max(0, min(nsel, min(a.sel_cap, D41_ATTN_SEL_MAX)));
    }
    r.nsel = nsel;
    r.sel = a.sel ? a.sel + (long long)row * a.sel_stride : nullptr;
    r.n_ext = 0;
    const int nblk = 4 + (nsel + D41R_BN - 1) / D41R_BN;
    const unsigned smb = (unsigned)__cvta_generic_to_shared(sm);
    const int g8 = lane >> 2, cq = lane & 3;

    // q of the CTA's 16 heads into the tile rows 0..15 (q's RoPE on the way when asked), then this warp's A fragments
    for (int c = t; c < D41_ATTN_HG * 64; c += 256) {
        const int h = c >> 6, ch = c & 63, hg = g * D41_ATTN_HG + h;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (hg < a.heads) {
            v = __ldg(reinterpret_cast<const uint4*>(a.q + (long long)row * a.q_stride + (long long)hg * D41_ATTN_D) + ch);
            if ((a.xflags & D41_ATTN_X_QROPE) && ch >= (D41_ATTN_D - D41_ATTN_ROPE) / 8) {
                unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float re = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] & 0xFFFFu)));
                    float im = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w[j] >> 16)));
                    float cs, sn;
                    d41_rope_cs(r.pos, a.rope_freq[4 * (ch - (D41_ATTN_D - D41_ATTN_ROPE) / 8) + j], cs, sn);
                    d41_rope_fwd(re, im, cs, sn);
                    w[j] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(re)) | ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(im)) << 16);
                }
                v = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        *reinterpret_cast<uint4*>(sm + d41a_swz(h, ch)) = v;
    }
    __syncthreads();
    const int kt = warp & 3, hf = warp >> 2;
    unsigned qf[8][2][4];
    {
        const int qrow = (lane & 7) + ((lane >> 3) & 1) * 8, qch = lane >> 4;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
            d41a_ldsm_x4(qf[k2][0], smb + d41a_swz(qrow, 32 * hf + 4 * k2 + qch));
            d41a_ldsm_x4(qf[k2][1], smb + d41a_swz(qrow, 32 * hf + 4 * k2 + 2 + qch));
        }
    }
    __syncthreads();                     // the tile is free for the keys

    float o[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { o[nt][0] = 0.f; o[nt][1] = 0.f; o[nt][2] = 0.f; o[nt][3] = 0.f; }
    float mh[2] = {-1e30f, -1e30f}, lh[2] = {0.f, 0.f};
    d41a_raw raw;
    raw.kind = 0;
    if (split < nblk) { int kd, bs; d41r_blk(split, kd, bs); d41a_fetch(r, kd, bs, raw, a.err); }
    float* sp = reinterpret_cast<float*>(sm + D41R_SM_SP);
    __nv_bfloat16* P = reinterpret_cast<__nv_bfloat16*>(sm + D41R_SM_P);
    volatile float* al = reinterpret_cast<volatile float*>(sm + D41R_SM_AL);
    const volatile int* kvf = reinterpret_cast<const volatile int*>(sm + D41R_SM_KV);
    for (int b = split; b < nblk; b += nspl) {
        d41r_store(sm, raw);
        __syncthreads();
        if (b + nspl < nblk) { int kd, bs; d41r_blk(b + nspl, kd, bs); d41a_fetch(r, kd, bs, raw, a.err); }
        {   // S half hf of key tile kt: 16 k16 steps in k order
            float s4[4] = {0.f, 0.f, 0.f, 0.f};
            const int kch = lane >> 3;
#pragma unroll
            for (int k2 = 0; k2 < 8; ++k2) {
                unsigned bb[4];
                d41a_ldsm_x4(bb, smb + d41a_swz(8 * kt + (lane & 7), 32 * hf + 4 * k2 + kch));
                d41a_mma(s4, qf[k2][0], bb[0], bb[1]);
                d41a_mma(s4, qf[k2][1], bb[2], bb[3]);
            }
            float* q0 = sp + (hf * 16 + g8) * D41R_SP_LD + 8 * kt + 2 * cq;
            float* q1 = sp + (hf * 16 + g8 + 8) * D41R_SP_LD + 8 * kt + 2 * cq;
            q0[0] = s4[0]; q0[1] = s4[1]; q1[0] = s4[2]; q1[1] = s4[3];
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {     // the softmax of head h = warp + 8 hh; lane = key
            const int h = warp + 8 * hh;
            const float s = kvf[lane] != 0 ? __fmul_rn(__fadd_rn(sp[h * D41R_SP_LD + lane], sp[(16 + h) * D41R_SP_LD + lane]), a.scale) : -INFINITY;
            float bm = s;
#pragma unroll
            for (int x = 16; x >= 1; x >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, x));
            const float mn = fmaxf(mh[hh], bm);
            const float alpha = expf(mh[hh] - mn);
            const float p = expf(s - mn);
            float ps = p;
#pragma unroll
            for (int x = 16; x >= 1; x >>= 1) ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffffu, ps, x));
            lh[hh] = __fadd_rn(__fmul_rn(lh[hh], alpha), ps);
            mh[hh] = mn;
            P[h * D41R_P_LD + lane] = __float2bfloat16_rn(p);
            if (lane == 0) al[h] = alpha;
        }
        __syncthreads();
        {   // acc = acc a + P K over dims 64 warp ..
            const float a0 = al[g8], a1 = al[g8 + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                o[nt][0] = __fmul_rn(o[nt][0], a0); o[nt][1] = __fmul_rn(o[nt][1], a0);
                o[nt][2] = __fmul_rn(o[nt][2], a1); o[nt][3] = __fmul_rn(o[nt][3], a1);
            }
            const int prow = (lane & 7) + ((lane >> 3) & 1) * 8, pch = lane >> 4;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned pa[4];
                pa[0] = *reinterpret_cast<const unsigned*>(P + g8 * D41R_P_LD + 16 * kk + 2 * cq);
                pa[1] = *reinterpret_cast<const unsigned*>(P + (g8 + 8) * D41R_P_LD + 16 * kk + 2 * cq);
                pa[2] = *reinterpret_cast<const unsigned*>(P + g8 * D41R_P_LD + 16 * kk + 2 * cq + 8);
                pa[3] = *reinterpret_cast<const unsigned*>(P + (g8 + 8) * D41R_P_LD + 16 * kk + 2 * cq + 8);
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    unsigned vb[4];
                    d41a_ldsm_x4_t(vb, smb + d41a_swz(16 * kk + prow, 8 * warp + 2 * np + pch));
                    d41a_mma(o[2 * np], pa, vb[0], vb[1]);
                    d41a_mma(o[2 * np + 1], pa, vb[2], vb[3]);
                }
            }
        }
        __syncthreads();
    }
    float* ml = reinterpret_cast<float*>(sm + D41R_SM_ML);
    if (lane == 0) { ml[warp] = mh[0]; ml[warp + 8] = mh[1]; ml[16 + warp] = lh[0]; ml[16 + warp + 8] = lh[1]; }
    if (nspl > 1) {
        float* pp = a.part + ((long long)(row * a.groups + g) * nspl + split) * D41_ATTN_PART_FLOATS;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int d = 64 * warp + 8 * nt + 2 * cq;
            *reinterpret_cast<float2*>(pp + g8 * D41_ATTN_D + d) = make_float2(o[nt][0], o[nt][1]);
            *reinterpret_cast<float2*>(pp + (g8 + 8) * D41_ATTN_D + d) = make_float2(o[nt][2], o[nt][3]);
        }
        __syncthreads();
        if (t < 32) pp[D41_ATTN_HG * D41_ATTN_D + t] = ml[t];    // m [16], l [16]
        return;
    }
    // one split (calls of > 16 rows): the merge of one split is exact (m = m0, l = l0, acc = po), so finish here
    float* st = reinterpret_cast<float*>(sm);                     // the key tile is free: O [16][512] f32
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int d = 64 * warp + 8 * nt + 2 * cq;
        st[g8 * D41_ATTN_D + d] = o[nt][0]; st[g8 * D41_ATTN_D + d + 1] = o[nt][1];
        st[(g8 + 8) * D41_ATTN_D + d] = o[nt][2]; st[(g8 + 8) * D41_ATTN_D + d + 1] = o[nt][3];
    }
    __syncthreads();
    const int half = t >> 7, d0 = 4 * (t & 127);
    for (int hp = 0; hp < D41_ATTN_HG / 2; ++hp) {
        const int h = 2 * hp + half, hg = g * D41_ATTN_HG + h;
        if (hg >= a.heads) continue;     // uniform over the 128 threads of a half
        const float Lf = __fadd_rn(ml[16 + h], expf(__ldg(a.sink + hg) - ml[h]));
        float x[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) x[k] = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(st[h * D41_ATTN_D + d0 + k], Lf)));
        d41a_epi4(a, row, hg, r.pos, d0, lane, x);
    }
}

extern "C" __global__ void __launch_bounds__(128) d41_attn_merge_recipe(const __grid_constant__ d41_attn_args a) {
    const int t = threadIdx.x, lane = t & 31;
    const int hl = blockIdx.x % D41_ATTN_HG, rg = blockIdx.x / D41_ATTN_HG;
    const int g = rg % a.groups, row = rg / a.groups;
    const int hg = g * D41_ATTN_HG + hl;
    if (row >= a.rows || hg >= a.heads) return;
    if (t == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.sink + hg));
    if (a.rot_xh != nullptr && t < 8)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(a.rot_suh[hg / a.rot_hpg] + (long long)(hg % a.rot_hpg) * D41_ATTN_D + t * 64));
    D41_PDL_ENTRY();
    const int nspl = a.nsplit_max;
    const float* pb = a.part + (long long)(row * a.groups + g) * nspl * D41_ATTN_PART_FLOATS;
    const int d0 = 4 * t;
    float m = -INFINITY;
    for (int s = 0; s < nspl; ++s) m = fmaxf(m, __ldcg(pb + (long long)s * D41_ATTN_PART_FLOATS + D41_ATTN_HG * D41_ATTN_D + hl));
    float l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < nspl; ++s) {
        const float* pp = pb + (long long)s * D41_ATTN_PART_FLOATS;
        const float e = expf(__ldcg(pp + D41_ATTN_HG * D41_ATTN_D + hl) - m);
        l = __fmaf_rn(__ldcg(pp + D41_ATTN_HG * D41_ATTN_D + 16 + hl), e, l);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(pp + hl * D41_ATTN_D + d0));
        acc[0] = __fmaf_rn(v.x, e, acc[0]); acc[1] = __fmaf_rn(v.y, e, acc[1]); acc[2] = __fmaf_rn(v.z, e, acc[2]); acc[3] = __fmaf_rn(v.w, e, acc[3]);
    }
    l = __fadd_rn(l, expf(__ldg(a.sink + hg) - m));
    float x[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) x[k] = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(acc[k], l)));
    d41a_epi4(a, row, hg, __ldg(a.pos + row), d0, lane, x);
}
