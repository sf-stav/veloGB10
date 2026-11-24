// =============================================================================
// kernels/gpu_dsv41/64_attn_merge.cuh — band 60, leg PA1: the decode attention's split merge + epilogue as its own
// launch (D41_ATTN_X_DEFER), the structure of the external reference's `_sparse_attn_merge_rot2` (one CTA per (row,
// head), the inverse RoPE and wo_a's rotated rows in the merge) with OUR arithmetic (PLAN/DSV41_RX6_REPORT.md 2.1 (ii)):
//   T = slot 0 (CTA 0's in-order prefix), then T <- merge(T, slot s) for s = nS / Ge .. nS - 1 in order (d41a_merge's
//   operations per element), then d41a_epilogue's per-element sequence: L += exp(sink - M); o = bf16(d41a_div(O, L));
//   inverse RoPE on dims 448..511; act_quant per 32 dims; bf16 store; wo_a's rotated rows (d41x_prep4 per 128-block).
// The values are the decode entry's own (the same slots, the same order, the same operations), so the output bits equal
// the in-kernel last-arriver merge for every row count, head range and split group G.
//
// d41_attn_merge: grid = rows x groups x 16 CTAs (head hl of group g of row r at ((r * groups + g) * 16 + hl)), 128
// threads; thread t holds dims 4t .. 4t + 3 of its head (warp w = the 128-block w, lane L = dims 128w + 4L ..: the rotation's
// lane map). The partial layout is the decode CTA's fragment layout (63_attn_entry.cuh): element (head h, dim d) of a slot
// is float ((nt * 512 + tt) * 4 + 2 (h >> 3) + (d & 1)), tt = 32 (d >> 5) + 4 (h & 7) + ((d >> 1) & 3), nt = (d >> 3) & 3;
// (m, l) of head h is float2 h at float 512 * 16.
// =============================================================================
#pragma once

// The row's split count and the CTAs per (row, group) the decode entry used (d41a_run's own computation). False when the
// decode entry wrote nothing for the row (more splits than slots: it reported D41_ATTN_ERR_ARGS).
__device__ __forceinline__ bool d41a_row_splits(const d41_attn_args& a, int row, int& nS, int& Ge, int& pos) {
    pos = __ldg(a.pos + row);
    const int wend = a.win_end ? __ldg(a.win_end + row) : pos;
    const int ln = a.lane ? __ldg(a.lane + row) : 0;
    const d41_attn_lane* L = &a.lanes[ln < 0 ? 0 : (ln >= D41_ATTN_MAX_LANES ? D41_ATTN_MAX_LANES - 1 : ln)];
    int nsel = 0;
    if (a.sel != nullptr && L->kv != nullptr) {
        nsel = a.n_sel ? __ldg(a.n_sel + row) : a.sel_cap;
        nsel = max(0, min(nsel, min(a.sel_cap, D41_ATTN_SEL_MAX)));
    }
    const int n_ext = L->ext_q != nullptr ? max(0, min(L->n_ext, D41_ATTN_EXT_MAX)) : 0;
    const int nw = wend >= 64 ? 2 : 1;
    const int nb = nw + ((nsel + 63) >> 6) + (n_ext > 0 ? 1 : 0);
    nS = (nb + D41_ATTN_SPLIT_BLOCKS - 1) / D41_ATTN_SPLIT_BLOCKS;
    const int slots = a.nsplit_max & 0xFFFF;
    const int G = (a.nsplit_max >> 16) ? (a.nsplit_max >> 16) : slots;
    if (nS > slots) return false;
    Ge = min(G, nS);
    return true;
}

__device__ __forceinline__ void d41a_ld_slot(const float* pp, long long i0, int hl, float (&o)[4], float& m, float& l) {
    const float2 a0 = __ldcg(reinterpret_cast<const float2*>(pp + i0));
    const float2 a1 = __ldcg(reinterpret_cast<const float2*>(pp + i0 + 4));
    const float2 ml = __ldcg(reinterpret_cast<const float2*>(pp + D41_ATTN_THREADS * 16) + hl);
    o[0] = a0.x; o[1] = a0.y; o[2] = a1.x; o[3] = a1.y; m = ml.x; l = ml.y;
}

// d41a_epilogue's per-element sequence after the division, for 4 consecutive dims d0..d0+3 of head hg (x = the bf16-rounded
// O / L values): inverse RoPE, act_quant (group = the 8 lanes 8k .. 8k + 7 of a warp), the bf16 store and PD1's wo_a rotation
// (the warp = one 128-block: lane L holds dims 128 w + 4 L). Shared by d41_attn_merge and the recipe rule (65_attn_recipe.cuh).
__device__ __forceinline__ void d41a_epi4(const d41_attn_args& a, int row, int hg, int pos, int d0, int lane, float (&x)[4]) {
    if ((a.epi & D41_ATTN_EPI_ROPE) && d0 >= D41_ATTN_D - D41_ATTN_ROPE) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float sn, cs;
            d41_rope_cs(pos, a.rope_freq[(d0 + 2 * j - (D41_ATTN_D - D41_ATTN_ROPE)) >> 1], cs, sn);
            float y0 = x[2 * j], y1 = x[2 * j + 1];
            d41_rope_inv(y0, y1, cs, sn);
            x[2 * j] = __bfloat162float(__float2bfloat16_rn(y0));
            x[2 * j + 1] = __bfloat162float(__float2bfloat16_rn(y1));
        }
    }
    if (a.epi & D41_ATTN_EPI_AQ) {                  // act_quant group = 32 dims = the 8 lanes 8k .. 8k + 7
        float am = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        const int e = d41a_aq_exp(am);
        const float sc = __uint_as_float((unsigned)(e + 127) << 23);
        const float isc = __uint_as_float((unsigned)(127 - e) << 23);
        if ((lane & 7) == 0 && a.out_s != nullptr)
            a.out_s[(long long)row * a.outs_stride + (long long)hg * (D41_ATTN_D / 32) + (d0 >> 5)] = (unsigned char)(e + 127);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const __nv_fp8_storage_t q0 = __nv_cvt_float_to_fp8(__fmul_rn(x[2 * j], isc), __NV_SATFINITE, __NV_E4M3);
            const __nv_fp8_storage_t q1 = __nv_cvt_float_to_fp8(__fmul_rn(x[2 * j + 1], isc), __NV_SATFINITE, __NV_E4M3);
            x[2 * j] = __fmul_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(q0, __NV_E4M3))), sc);
            x[2 * j + 1] = __fmul_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(q1, __NV_E4M3))), sc);
            if (a.out_q != nullptr)
                *reinterpret_cast<unsigned short*>(a.out_q + (long long)row * a.outq_stride + (long long)hg * D41_ATTN_D + d0 + 2 * j) = (unsigned short)(q0 | (q1 << 8));
        }
    }
    unsigned* op = reinterpret_cast<unsigned*>(a.out + (long long)row * a.out_stride + (long long)hg * D41_ATTN_D + d0);
    op[0] = d41a_pack_bf16x2(x[0], x[1]);
    op[1] = d41a_pack_bf16x2(x[2], x[3]);
    if (a.rot_xh != nullptr) {                      // PD1's wo_a input rotation on the stored bf16 values
        float v[4], sv[4];
        const __half* su = a.rot_suh[hg / a.rot_hpg] + (hg % a.rot_hpg) * D41_ATTN_D + d0;
#pragma unroll
        for (int r = 0; r < 4; ++r) { v[r] = __bfloat162float(__float2bfloat16_rn(x[r])); sv[r] = __half2float(su[r]); }
        d41x_prep4(v, sv, lane);
        __half h[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) h[r] = __float2half_rn(v[r]);
        *reinterpret_cast<uint2*>(a.rot_xh + (long long)row * a.rot_ld + (long long)hg * D41_ATTN_D + d0) = *reinterpret_cast<const uint2*>(h);
    }
}

extern "C" __global__ void __launch_bounds__(128) d41_attn_merge(const __grid_constant__ d41_attn_args a) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int hl = blockIdx.x % D41_ATTN_HG, rg = blockIdx.x / D41_ATTN_HG;
    const int g = rg % a.groups, row = rg / a.groups;
    const int hg = g * D41_ATTN_HG + hl;
    if (row >= a.rows || hg >= a.heads) return;
    // before the wait: the epilogue's constants into L2 (the sink, wo_a's suh row of this head: 8 lines of 128 B)
    if (t == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.sink + hg));
    if (a.rot_xh != nullptr && t < 8)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(a.rot_suh[hg / a.rot_hpg] + (long long)(hg % a.rot_hpg) * D41_ATTN_D + t * 64));
    D41_PDL_ENTRY();
    int nS = 0, Ge = 1, pos = 0;
    if (!d41a_row_splits(a, row, nS, Ge, pos)) return;
    const int slots = a.nsplit_max & 0xFFFF;
    const float* pbase = a.part + (long long)(row * a.groups + g) * slots * D41_ATTN_PART_FLOATS;
    const int d0 = 4 * t;
    const int tt0 = 32 * (d0 >> 5) + 4 * (hl & 7) + ((d0 >> 1) & 3);
    const long long i0 = ((long long)((d0 >> 3) & 3) * D41_ATTN_THREADS + tt0) * 4 + 2 * (hl >> 3);
    float o[4], M, Ls;
    if (a.xflags & D41_ATTN_X_2CTA) {
        const float4 v0 = __ldcg(reinterpret_cast<const float4*>(pbase + hl * D41_ATTN_D + d0));
        const float2 ml0 = __ldcg(reinterpret_cast<const float2*>(pbase + D41_ATTN_HG * D41_ATTN_D) + hl);
        o[0] = v0.x; o[1] = v0.y; o[2] = v0.z; o[3] = v0.w; M = ml0.x; Ls = ml0.y;
        for (int s = nS / Ge; s < nS; ++s) {            // d41a_merge, in split order
            const float* pp = pbase + (long long)s * D41_ATTN_PART_FLOATS;
            const float4 vs = __ldcg(reinterpret_cast<const float4*>(pp + hl * D41_ATTN_D + d0));
            const float2 mls = __ldcg(reinterpret_cast<const float2*>(pp + D41_ATTN_HG * D41_ATTN_D) + hl);
            const float n = fmaxf(M, mls.x);
            const float e = expf(M - n), f = expf(mls.x - n);
            Ls = __fmaf_rn(Ls, e, __fmul_rn(mls.y, f));
            o[0] = __fmaf_rn(o[0], e, __fmul_rn(vs.x, f));
            o[1] = __fmaf_rn(o[1], e, __fmul_rn(vs.y, f));
            o[2] = __fmaf_rn(o[2], e, __fmul_rn(vs.z, f));
            o[3] = __fmaf_rn(o[3], e, __fmul_rn(vs.w, f));
            M = n;
        }
    } else {
        d41a_ld_slot(pbase, i0, hl, o, M, Ls);
        for (int s = nS / Ge; s < nS; ++s) {            // d41a_merge, in split order
            float so[4], sm, sl;
            d41a_ld_slot(pbase + (long long)s * D41_ATTN_PART_FLOATS, i0, hl, so, sm, sl);
            const float n = fmaxf(M, sm);
            const float e = expf(M - n), f = expf(sm - n);
            Ls = __fmaf_rn(Ls, e, __fmul_rn(sl, f));
#pragma unroll
            for (int k = 0; k < 4; ++k) o[k] = __fmaf_rn(o[k], e, __fmul_rn(so[k], f));
            M = n;
        }
    }
    // d41a_epilogue's per-element sequence
    const float Lf = __fadd_rn(Ls, expf(__ldg(a.sink + hg) - M));
    float x[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) x[k] = __bfloat162float(__float2bfloat16_rn(d41a_div(o[k], Lf)));
    d41a_epi4(a, row, hg, pos, d0, lane, x);
    (void)warp;
}
