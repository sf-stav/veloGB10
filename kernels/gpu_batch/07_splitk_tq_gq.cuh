// gpu_batch/07_splitk_tq_gq.cuh: TurboQuant GQA-packed split-K attention and the bf16 GQA-packed twin.
// Per-head bit-identical to the per-head kernels (same splits, lane slices, shfl trees).
// Section 7 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================================================================================================
// gqa_attn_splitk_tq_gq — GQA-PACKED twin of gqa_attn_splitk_tq: one block per (kv head, split)
// reads the K/V chunk ONCE and applies the whole GQA group (the E1 fix — without it the KV-read
// win is re-read gqa_ratio times at Hy3's 8:1 GQA). Per-head bit-identical to the per-head kernel
// by the same construction as the q4 pair (same splits, same lane slices, same shfl trees, same
// merge, same epilogue). hd == 128 only (the TQ layout); gqa_ratio <= SK_GQA_MAX.
template<int DPL_T>
__device__ __forceinline__ void gqa_splitk_tq_gq_impl(
    float* out_m, float* out_l, float* out_acc,
    const float* qrqs, const unsigned char* k_cache, const unsigned char* v_cache,
    const float* tables, const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int hd  = TQ_HD;                   // == DPL_T*32 by launch contract
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (kvh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nkv*ns_grid*B grid.
    const int kvh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int pc = pos[b] + 1;
    const int pos_start = pos[0];
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);
    if (split >= ns) return;
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;

    if (start >= pc) {
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            if (g >= gqa_ratio) break;
            const long long idx = ((long long)b * nh + (kvh * gqa_ratio + g)) * ns_grid + split;
            if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
            if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        }
        return;
    }

    // Group qr/qs staged in smem (rotated once per (token, head) by rotate_q_tq).
    extern __shared__ float sh[];
    float* sqr = sh;                             // [SK_GQA_MAX][hd]
    float* sqs = sh + SK_GQA_MAX * hd;           // [SK_GQA_MAX][hd]
    float* sacc = sh + 2 * SK_GQA_MAX * hd;      // NW * hd
    float* sm   = sacc + NW * hd;                // NW
    float* sl   = sm + NW;                       // NW
    float* tacc = sl + NW;                       // hd
    {
        const float* qbase = qrqs + (long long)b * (nh * 2 * hd) + (kvh * gqa_ratio) * 2 * hd + 2 * lane * DPL_T;  // interleaved: coord j at floats 2j/2j+1
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            const int gs = min(g, gqa_ratio - 1);
            const float* qrow = qbase + gs * 2 * hd;
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) {
                sqr[g * hd + lane * DPL_T + i] = (g < gqa_ratio) ? qrow[2 * i] : 0.0f;
                sqs[g * hd + lane * DPL_T + i] = (g < gqa_ratio) ? qrow[2 * i + 1] : 0.0f;
            }
        }
    }
    __syncthreads();

    float m[SK_GQA_MAX], l[SK_GQA_MAX];
    float acc[SK_GQA_MAX][DPL_T];
    #pragma unroll
    for (int g = 0; g < SK_GQA_MAX; g++) {
        m[g] = -1e30f; l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) acc[g][i] = 0.0f;
    }

    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    const float qjl_scale = tables[TQ_TAB_SCALE];
    const int rb = TQ_ROW_BYTES;
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride * rb;
    const unsigned char* kb = k_cache + kvbase;
    const unsigned char* vb = v_cache + kvbase;
#if TQ_K_BITS == 3
    const int kc_byte = ((lane * DPL_T) * 3) >> 3;
    const int kc_shift0 = (lane * DPL_T) * 3 - 8 * kc_byte;   // {0,4}: the lane's first K coord in the u16
#else
    const int kc_byte = (lane * DPL_T) >> 2;
#endif
    const int sc_byte = TQ_SIGN_OFF + ((lane * DPL_T) >> 3);
    const int sc_shift = (lane * DPL_T) & 7;
    const int vc_byte = ((lane * DPL_T) * 3) >> 3;
    const int vc_shift0 = (lane * DPL_T) * 3 - 8 * vc_byte;   // {0,4}: the lane's first coord in the u16
    // E1b 2-row software pipeline: all loads up front (one memory round-trip per pair), then the
    // two online-softmax updates sequentially in ascending order — bit-identical to unpipelined.
    for (int r = start + warp; r < end; r += 2 * NW) {
        const int r2 = r + NW;
        const bool has2 = r2 < end;
        const int dd = r - pos_start;
        const int t  = (!path || dd < 0) ? r  : pos_start + (int)path[b * MAX_VERIFY + dd];
        const int t2 = has2 ? ((!path || (r2 - pos_start) < 0) ? r2 : pos_start + (int)path[b * MAX_VERIFY + (r2 - pos_start)]) : 0;
        const unsigned char* krow  = kb + (long long)t * rb;
        const unsigned char* krow2 = kb + (long long)t2 * rb;
        const unsigned char* vrow  = vb + (long long)t * rb;
        const unsigned char* vrow2 = vb + (long long)t2 * rb;
        const float rn  = tq_half_at(krow, TQ_RN_OFF),  kn  = tq_half_at(krow, TQ_KN_OFF);
        const float vn  = tq_half_at(vrow, 48);
        const float rn2 = has2 ? tq_half_at(krow2, TQ_RN_OFF) : 0.0f, kn2 = has2 ? tq_half_at(krow2, TQ_KN_OFF) : 0.0f;
        const float vn2 = has2 ? tq_half_at(vrow2, 48) : 0.0f;
#if TQ_K_BITS == 3
        const unsigned short kw  = (unsigned short)krow[kc_byte] | ((unsigned short)krow[kc_byte + 1] << 8);
        const unsigned short kw2 = has2 ? ((unsigned short)krow2[kc_byte] | ((unsigned short)krow2[kc_byte + 1] << 8)) : 0;
#else
        const unsigned char kcb  = krow[kc_byte];
        const unsigned char kcb2 = has2 ? krow2[kc_byte] : 0;
#endif
        const unsigned char scb  = krow[sc_byte];
        const unsigned char scb2 = has2 ? krow2[sc_byte] : 0;
        const unsigned short vw  = (unsigned short)vrow[vc_byte] | ((unsigned short)vrow[vc_byte + 1] << 8);
        const unsigned short vw2 = has2 ? ((unsigned short)vrow2[vc_byte] | ((unsigned short)vrow2[vc_byte + 1] << 8)) : 0;
        float kdq[DPL_T], kdq2[DPL_T], vdq[DPL_T], vdq2[DPL_T], sg[DPL_T], sg2[DPL_T];
        #pragma unroll
        for (int u = 0; u < DPL_T; u++) {
#if TQ_K_BITS == 3
            kdq[u]  = cb3[(kw  >> (kc_shift0 + 3 * u)) & 7];
            kdq2[u] = has2 ? cb3[(kw2 >> (kc_shift0 + 3 * u)) & 7] : 0.0f;
#else
            kdq[u]  = cb2[(kcb  >> (2 * u)) & 3];
            kdq2[u] = has2 ? cb2[(kcb2 >> (2 * u)) & 3] : 0.0f;
#endif
            sg[u]   = ((scb  >> (sc_shift + u)) & 1) ? 1.0f : -1.0f;
            sg2[u]  = has2 ? (((scb2 >> (sc_shift + u)) & 1) ? 1.0f : -1.0f) : 0.0f;
            vdq[u]  = cb3[(vw  >> (vc_shift0 + 3 * u)) & 7];
            vdq2[u] = has2 ? cb3[(vw2 >> (vc_shift0 + 3 * u)) & 7] : 0.0f;
        }
        // dual dots for both rows, all heads (independent of the running softmax state)
        float s[SK_GQA_MAX], s2[SK_GQA_MAX];
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            float sc_ = 0.0f, sq_ = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) {
                sc_ += sqr[g * hd + lane * DPL_T + i] * kdq[i];
                sq_ += sqs[g * hd + lane * DPL_T + i] * sg[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sc_ += __shfl_xor_sync(0xffffffffu, sc_, off);
                sq_ += __shfl_xor_sync(0xffffffffu, sq_, off);
            }
            s[g] = kn * (sc_ + qjl_scale * rn * sq_);
        }
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            float sc_ = 0.0f, sq_ = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) {
                sc_ += sqr[g * hd + lane * DPL_T + i] * kdq2[i];
                sq_ += sqs[g * hd + lane * DPL_T + i] * sg2[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sc_ += __shfl_xor_sync(0xffffffffu, sc_, off);
                sq_ += __shfl_xor_sync(0xffffffffu, sq_, off);
            }
            s2[g] = has2 ? kn2 * (sc_ + qjl_scale * rn2 * sq_) : 0.0f;
        }
        // the two state updates, ascending — identical sequence to the unpipelined loop
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            if (g >= gqa_ratio) break;
            const float m_new = fmaxf(m[g], s[g]);
            const float a_old = __expf(m[g] - m_new), a_cur = __expf(s[g] - m_new);
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) acc[g][i] = acc[g][i] * a_old + a_cur * (vn * vdq[i]);
            m[g] = m_new;
            l[g] = l[g] * a_old + a_cur;
        }
        if (has2) {
            #pragma unroll
            for (int g = 0; g < SK_GQA_MAX; g++) {
                if (g >= gqa_ratio) break;
                const float m_new = fmaxf(m[g], s2[g]);
                const float a_old = __expf(m[g] - m_new), a_cur = __expf(s2[g] - m_new);
                #pragma unroll
                for (int i = 0; i < DPL_T; i++) acc[g][i] = acc[g][i] * a_old + a_cur * (vn2 * vdq2[i]);
                m[g] = m_new;
                l[g] = l[g] * a_old + a_cur;
            }
        }
    }

    // Merge per head sequentially (the per-head kernel's exact fixed-warp-order merge) + Pi^T
    // epilogue once per head.
    #pragma unroll
    for (int g = 0; g < SK_GQA_MAX; g++) {
        if (g >= gqa_ratio) break;
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) sacc[warp * hd + lane * DPL_T + i] = acc[g][i];
        if (lane == 0) { sm[warp] = m[g]; sl[warp] = l[g]; }
        __syncthreads();
        const long long idx = ((long long)b * nh + (kvh * gqa_ratio + g)) * ns_grid + split;
        if (threadIdx.x < hd) {
            const int d = threadIdx.x;
            float mg = -1e30f;
            for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w]);
            float num = 0.0f, den = 0.0f;
            for (int w = 0; w < NW; w++) {
                const float a = __expf(sm[w] - mg);
                num += sacc[w * hd + d] * a;
                den += sl[w] * a;
            }
            tacc[d] = num;
            if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
        }
        __syncthreads();
        if (threadIdx.x < hd) {
            const int d = threadIdx.x;
            const float* pi = tables + TQ_TAB_PI + d;
            float o = 0.0f;
            #pragma unroll
            for (int i = 0; i < TQ_HD; i++) o += pi[i * TQ_HD] * tacc[i];
            out_acc[idx * hd + d] = o;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gqa_attn_splitk_tq_gq(
    float* out_m, float* out_l, float* out_acc,
    const float* qrqs, const unsigned char* k_cache, const unsigned char* v_cache,
    const float* tables, const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path) {
    gqa_splitk_tq_gq_impl<4>(out_m, out_l, out_acc, qrqs, k_cache, v_cache, tables,
                             pos, bs_packed, nh_packed, slot_ids, path);
}

// Probe-only: dump the RAW dual-dot scores (the splitk score path, pre-softmax) for every
// (b, qh) row — scores[(b*nh+qh)*pc + r] = kn_r·(<Pi q, cb2[idx_r]> + qjl·rn_r·<S q, sign_r>),
// pc = pos[b]+1. Same row math as gqa_attn_splitk_tq; validated against the goldens' scores_tq.bin.
extern "C" __global__ void gqa_attn_splitk_tq_dbg_scores(
    float* scores, const float* qrqs, const unsigned char* k_cache,
    const float* tables, const int* pos, long long bs_packed, int nh_packed, const int* slot_ids) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int hd  = TQ_HD;
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const int blk = blockIdx.x;
    if (blk >= B * nh) return;
    const int b = blk / nh;
    const int qh = blk % nh;
    const int kvh = qh / gqa_ratio;
    const int pc = pos[b] + 1;
    const int slot = slot_ids[b];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;
    const int DPL = hd >> 5;
    const float* qr = qrqs + ((long long)b * nh + qh) * 2 * hd + 2 * lane * DPL;  // interleaved: coord j at floats 2j/2j+1
    float qrv[SK_DPL_MAX], qsv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) {
        qrv[i] = (i < DPL) ? qr[2 * i] : 0.0f;
        qsv[i] = (i < DPL) ? qr[2 * i + 1] : 0.0f;
    }
    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    const float qjl_scale = tables[TQ_TAB_SCALE];
    const int rb = TQ_ROW_BYTES;
    const unsigned char* kb = k_cache + (((long long)slot * nkv + kvh) * (long long)stride) * rb;
#if TQ_K_BITS == 3
    const int kc_byte = ((lane * DPL) * 3) >> 3;
    const int kc_shift0 = (lane * DPL) * 3 - 8 * kc_byte;   // {0,4}: the lane's first K coord in the u16
#else
    const int kc_byte = (lane * DPL) >> 2;
#endif
    const int sc_byte = TQ_SIGN_OFF + ((lane * DPL) >> 3);
    const int sc_shift = (lane * DPL) & 7;
    for (int r = warp; r < pc; r += NW) {
        const unsigned char* krow = kb + (long long)r * rb;
        const float rn = tq_half_at(krow, TQ_RN_OFF), kn = tq_half_at(krow, TQ_KN_OFF);
#if TQ_K_BITS == 3
        const unsigned short kw = (unsigned short)krow[kc_byte] | ((unsigned short)krow[kc_byte + 1] << 8);
#else
        const unsigned char kcb = krow[kc_byte];
#endif
        const unsigned char scb = krow[sc_byte];
        float s_code = 0.0f, s_qjl = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) {
            if (i >= DPL) break;
#if TQ_K_BITS == 3
            s_code += qrv[i] * cb3[(kw >> (kc_shift0 + 3 * i)) & 7];
#else
            s_code += qrv[i] * cb2[(kcb >> (2 * i)) & 3];
#endif
            s_qjl  += qsv[i] * (((scb >> (sc_shift + i)) & 1) ? 1.0f : -1.0f);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s_code += __shfl_xor_sync(0xffffffffu, s_code, off);
            s_qjl  += __shfl_xor_sync(0xffffffffu, s_qjl, off);
        }
        const float s = kn * (s_code + qjl_scale * rn * s_qjl);
        if (lane == 0) scores[((long long)b * nh + qh) * pc + r] = s;
    }
}

// ===================================================================================================
// gqa_attn_splitk_gq — GQA-PACKED twin of the bf16 gqa_attn_splitk (the E1 fix generalized, E1c).
// The per-head bf16 kernel has the same re-read pathology as the q4 one: one block per QUERY head
// reads its GQA group's KV chunk gqa_ratio times (4× at qwen's 16Q/4KV, 8× at 122B's 16Q/2KV —
// at 32K+ ctx that is the whole "32K anomaly" again, in bf16). One block per (kv head, split)
// reads the chunk ONCE for the whole group. Per-head bit-identical by the same construction as
// the q4 packed kernel (same splits, same lane slices, same shfl trees, same merge), including
// the 2-row latency pipeline. hd ∈ {128, 256}, gqa_ratio ≤ SK_GQA_MAX.
template<int DPL_T>
__device__ __forceinline__ void gqa_splitk_gq_impl(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;   // == DPL_T*32 by launch contract
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    // F0: the q row pitch (bits 31-49) — the fused qkv mtot for offset views, nh*hd for packed.
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (kvh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nkv*ns_grid*B grid.
    const int kvh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int pc = pos[b] + 1;
    const int pos_start = col_pos_start ? col_pos_start[b] : pos[0];
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);
    if (split >= ns) return;
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;

    if (start >= pc) {                          // empty split: zero partials for the whole group
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            if (g >= gqa_ratio) break;
            const long long idx = ((long long)b * nh + (kvh * gqa_ratio + g)) * ns_grid + split;
            if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
            if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        }
        return;
    }

    float qv[SK_GQA_MAX][DPL_T];
    #pragma unroll
    for (int g = 0; g < SK_GQA_MAX; g++) {
        const int gs = min(g, gqa_ratio - 1);
        const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)(kvh * gqa_ratio + gs) * hd + lane * DPL_T;
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) qv[g][i] = (g < gqa_ratio) ? b2f(qrow[i]) : 0.0f;
    }

    float m[SK_GQA_MAX], l[SK_GQA_MAX];
    float acc[SK_GQA_MAX][DPL_T];
    #pragma unroll
    for (int g = 0; g < SK_GQA_MAX; g++) {
        m[g] = -1e30f; l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) acc[g][i] = 0.0f;
    }

    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL_T;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL_T;
    for (int r = start + warp; r < end; r += 2 * NW) {
        const int r2 = r + NW;
        const bool has2 = r2 < end;
        const int dd = r - pos_start;
        const int t  = (!path || dd < 0) ? r  : pos_start + (int)path[b * MAX_VERIFY + dd];
        const int t2 = has2 ? ((!path || (r2 - pos_start) < 0) ? r2 : pos_start + (int)path[b * MAX_VERIFY + (r2 - pos_start)]) : 0;
        const __nv_bfloat16* krow  = kb + (long long)t * hd;
        const __nv_bfloat16* krow2 = kb + (long long)t2 * hd;
        const __nv_bfloat16* vrow  = vb + (long long)t * hd;
        const __nv_bfloat16* vrow2 = vb + (long long)t2 * hd;
        float kdq[DPL_T], kdq2[DPL_T], vdq[DPL_T], vdq2[DPL_T];
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) {
            kdq[i]  = b2f(krow[i]);
            kdq2[i] = has2 ? b2f(krow2[i]) : 0.0f;
            vdq[i]  = b2f(vrow[i]);
            vdq2[i] = has2 ? b2f(vrow2[i]) : 0.0f;
        }
        float s[SK_GQA_MAX], s2[SK_GQA_MAX];
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            s[g] = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) s[g] += qv[g][i] * kdq[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s[g] += __shfl_xor_sync(0xffffffffu, s[g], off);
            s[g] *= scale;
        }
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            s2[g] = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) s2[g] += qv[g][i] * kdq2[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) s2[g] += __shfl_xor_sync(0xffffffffu, s2[g], off);
            s2[g] *= scale;
        }
        #pragma unroll
        for (int g = 0; g < SK_GQA_MAX; g++) {
            if (g >= gqa_ratio) break;
            const float m_new = fmaxf(m[g], s[g]);
            const float a_old = __expf(m[g] - m_new), a_cur = __expf(s[g] - m_new);
            #pragma unroll
            for (int i = 0; i < DPL_T; i++) acc[g][i] = acc[g][i] * a_old + a_cur * vdq[i];
            m[g] = m_new;
            l[g] = l[g] * a_old + a_cur;
        }
        if (has2) {
            #pragma unroll
            for (int g = 0; g < SK_GQA_MAX; g++) {
                if (g >= gqa_ratio) break;
                const float m_new = fmaxf(m[g], s2[g]);
                const float a_old = __expf(m[g] - m_new), a_cur = __expf(s2[g] - m_new);
                #pragma unroll
                for (int i = 0; i < DPL_T; i++) acc[g][i] = acc[g][i] * a_old + a_cur * vdq2[i];
                m[g] = m_new;
                l[g] = l[g] * a_old + a_cur;
            }
        }
    }

    extern __shared__ float sh[];
    float* sacc = sh;                     // NW * hd
    float* sm   = sh + NW * hd;           // NW
    float* sl   = sm + NW;                // NW
    #pragma unroll
    for (int g = 0; g < SK_GQA_MAX; g++) {
        if (g >= gqa_ratio) break;
        #pragma unroll
        for (int i = 0; i < DPL_T; i++) sacc[warp * hd + lane * DPL_T + i] = acc[g][i];
        if (lane == 0) { sm[warp] = m[g]; sl[warp] = l[g]; }
        __syncthreads();
        const long long idx = ((long long)b * nh + (kvh * gqa_ratio + g)) * ns_grid + split;
        if (threadIdx.x < hd) {
            const int d = threadIdx.x;
            float mg = -1e30f;
            for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w]);
            float num = 0.0f, den = 0.0f;
            for (int w = 0; w < NW; w++) {
                const float a = __expf(sm[w] - mg);
                num += sacc[w * hd + d] * a;
                den += sl[w] * a;
            }
            out_acc[idx * hd + d] = num;
            if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
        }
        __syncthreads();
    }
}

extern "C" __global__ void gqa_attn_splitk_gq(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128) {
        gqa_splitk_gq_impl<4>(out_m, out_l, out_acc, q, k_cache, v_cache,
                              pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    } else if (hd == 256) {
        gqa_splitk_gq_impl<8>(out_m, out_l, out_acc, q, k_cache, v_cache,
                              pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    }
    // other hd: never launched — attn_dispatch falls back to the per-head kernel.
}

// ================== gqa_attn_verify_e (F8/C: trunk GQA verify attention on tensor cores) =====
// F9: keep in sync with gpu.rs MAX_VERIFY (attn_dispatch debug_asserts the match).
#define GVE_MAX_VERIFY 16
//
// The bf16 splitk (gqa_attn_splitk_gq) computes q·k and p·v on SCALAR cores: at 30K ctx it ran
// 2x over the KV roofline (17.6 ms/step TP2; ~580 ms at 290K), ALU-bound on the per-lane dot.
// This port of the dspark ring_e pattern runs BOTH matmuls on
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32: one block per (kvh, SEG-slice) covers
// ALL B<=8 verify columns as query rows (B*QG <= 48 = 6 warps x 8 real rows; each mma m16
// tile carries 8 REAL rows + 8 parked — the row-half is masked to zero and its C-frag never
// emitted, trading half the tensor-core row throughput for half the register file, which is
// the right trade on a bandwidth-bound kernel). Q row-fragments load ONCE from global (the
// operand is ~24 KB, L2-resident; no smem staging), softmax folds ONCE per 32-key tile (raw
// scores parked in registers; per-group folds park P in stale bases), P round-trips a per-warp
// bf16 smem tile, V is staged TRANSPOSED for the PV B-fragments.
//
// CONTRACT (host-gated; falls back to splitk otherwise):
//   * hd == 128 or 256, B <= 8, B*QG <= 48, ONE slot for all columns (single-lane decode or
//     chain verify), KV path remap IDENTITY (tree paths cannot share K tiles across columns).
//   * Partials land in the SPLITK layout so the same reduce shape merges them: m/l at
//     [(b*nh + qh)*ns_grid + seg], acc fp32 at [same * hd + d]; ns_grid == SEG == gridDim.y.
//   * Columns whose pc <= jbegin write the empty-split zeros (-1e30 / 0 / 0) the merge expects.
// The 0*NaN fix from ring_e applies: masked keys park P=0 but the PV mma still multiplies
// stale smem V, so invalid key COLUMNS of the V tile are zeroed (K needs none: masked scores
// are REPLACED by -1e30, not multiplied away).
#define GVE_PT_PITCH 40            // 32 keys + 8 pad (bank-conflict free)

__device__ __forceinline__ unsigned gve_bf162_b32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ void gve_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int HDT>                                   // hd = 128 | 256
__device__ __forceinline__ void gve_impl(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ pos,
    long long bs_packed, int nh_packed, const int* __restrict__ slot_ids) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;
    const float scale = 1.0f / sqrtf((float)HDT);
    const int QG = nh / nkv, NQ = B * QG;           // row groups: gridDim.z = ceil(NQ/48)
    const int KSTEPS = HDT / 16;                    // 8 | 16
    const int NTILES = HDT / 8;                     // 16 | 32
    const int kvh = blockIdx.x, seg = blockIdx.y, SEG = gridDim.y;

    int max_pc = 0;
    for (int b = 0; b < B; ++b) max_pc = max(max_pc, pos[b] + 1);
    // F9 (batch invariance): the key-range partition must be a function of the REQUEST, never of
    // the batch width. Partitioning from this batch's max_pc gave a decode (pc=p+1) and a verify
    // (pc=p+N) different seg_size -> different softmax fold boundaries for column 0 -> ulp drift
    // -> argmax flips at near-ties (the F9 ladder: 10/16 clean). The bound is pos[0] +
    // GVE_MAX_VERIFY: pos[0] is the first column's position (identical for a decode and a verify
    // at the same position, and >= this batch's max_pc for every width <= GVE_MAX_VERIFY). It is
    // read from pos[] (rewritten per call AND per graph replay), never from a baked launch arg.
    // `jend` still stops at this batch's max_pc: tiles wholly beyond a column's own rbnd are exact
    // no-ops (scores -1e30 -> P=0; the fold's rescale is exp(0)=1).
    // `max(max_pc, ...)` is belt-and-braces: the _e lane is host-gated to single-request
    // forwards (decode b=1 / chain verify), where pos[] is contiguous and max_pc <=
    // pos[0]+16, so the max is exactly pos[0]+16; if a multi-request batch ever reaches
    // this lane, the bound degrades to the old max_pc (correct, just not width-invariant).
    // F9 (phase-8, the depth-2 residue): pos[0]+GVE_MAX_VERIFY aligns a verify col 0 with its
    // decode, but a verify col j>=1 is compared against a decode at pos[0]+j — whose pos[0]
    // differs, whose pc_bound differs, whose seg_size differs => different softmax fold
    // boundaries => 1-ulp drift on EVERY col>=1 (argmax flips at near-ties ~1/512 tokens;
    // single-proc repro: bench-mtp d2 512tok, MISMATCH at 442). The partition must key on a
    // per-REQUEST constant so decode and verify at ANY width resolve identical boundaries:
    // `stride` (the KV row stride) is exactly that — identical launch-to-launch for the whole
    // life of the request, and the segments beyond max_pc exit as exact no-ops via jend.
    const int pc_bound = (int)stride;
    const int seg_size = (pc_bound + SEG - 1) / SEG;
    const int jbegin = seg * seg_size;
    const int jend = min(max_pc, jbegin + seg_size);

    __shared__ __nv_bfloat16 kt[32 * HDT];
    __shared__ __nv_bfloat16 vt2[HDT * GVE_PT_PITCH];
    __shared__ __nv_bfloat16 pt[6 * 16 * GVE_PT_PITCH];   // 6 warps x 16 rows (8 real + 8 parked)

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;              // 0..5 — owns REAL rows w*8..w*8+7
    const int g    = lane >> 2, tq = lane & 3;      // mma frag coords
    const int qrow0 = w * 8 + blockIdx.z * 48;      // this lane's real row; grid.z = row group

    // This row's causal bound (pos[b]+1); rows >= NQ park everything.
    const int rbnd = (qrow0 + g < NQ) ? pos[(qrow0 + g) / QG] + 1 : 0;

    // Q row-fragments from GLOBAL (once; ~24 KB total, L2-resident): row (qrow0+g) of this
    // kvh's group, k-chunk ks16. A-frag layout: a[0],a[2] = row g cols 4tq..4tq+3; a[1],a[3]
    // = row g+8 (PARKED rows: zero — their scores stay -1e30, never emitted).
    unsigned qa[KSTEPS][4];
    #pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        qa[ks][0] = qa[ks][2] = 0u;
        qa[ks][1] = qa[ks][3] = 0u;
        if (qrow0 + g < NQ) {
            const __nv_bfloat16* q0 = q + (long long)((qrow0 + g) / QG) * q_pitch
                                     + (long long)(kvh * QG + ((qrow0 + g) % QG)) * HDT
                                     + ks * 16 + 4 * tq;
            qa[ks][0] = *(const unsigned*)q0;
            qa[ks][2] = *(const unsigned*)(q0 + 2);
        }
    }
    float m = -1e30f, l = 0.f;
    float o[NTILES][2];                             // real row g only (cols 2tq, 2tq+1)
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] = 0.f; o[nt][1] = 0.f; }

    const long long kvbase = ((long long)slot_ids[0] * nkv + kvh) * (long long)stride;
    const __nv_bfloat16* kbase = k_cache + kvbase * HDT;
    const __nv_bfloat16* vbase = v_cache + kvbase * HDT;

    for (int t0 = jbegin; t0 < jend; t0 += 32) {
        const int tv = min(32, jend - t0);
        for (int i = threadIdx.x * 8; i < tv * HDT; i += blockDim.x * 8) {
            const int row = i / HDT, col = i % HDT;
            *(uint4*)&kt[row * HDT + col] = *(const uint4*)&kbase[(long long)(t0 + row) * HDT + col];
            const __nv_bfloat16* v8 = &vbase[(long long)(t0 + row) * HDT + col];
            #pragma unroll
            for (int k = 0; k < 8; ++k) vt2[(col + k) * GVE_PT_PITCH + row] = v8[k];
        }
        if (tv < 32) {                              // 0*NaN guard (see header)
            for (int i = threadIdx.x; i < (32 - tv) * HDT; i += blockDim.x) {
                const int key = tv + i / HDT, col = i % HDT;
                vt2[col * GVE_PT_PITCH + key] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();
        __nv_bfloat16* mypt = pt + w * 16 * GVE_PT_PITCH;
        // ---- scores + single-fold-per-tile softmax (4 key-groups of 8) ----
        float sraw[4][2];
        float rmx = m;
        #pragma unroll
        for (int jg = 0; jg < 4; ++jg) {
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            #pragma unroll
            for (int ks = 0; ks < KSTEPS; ++ks) {
                const __nv_bfloat16* krow = kt + (jg * 8 + g) * HDT + ks * 16 + 4 * tq;
                unsigned bK[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)krow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(krow + 2)) };
                gve_mma(s, qa[ks], bK);
            }
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = jg * 8 + 2 * tq + e;
                sraw[jg][e] = (key < tv && (t0 + key) < rbnd) ? s[e] * scale : -1e30f;
            }
            float rh = fmaxf(sraw[jg][0], sraw[jg][1]);
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 1));
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 2));
            rmx = fmaxf(rmx, rh);
        }
        {                                           // one rescale to the tile-final basis
            const float rs = __expf(m - rmx);       // m=-1e30 first -> 0, no NaN
            l *= rs; m = rmx;
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] *= rs; o[nt][1] *= rs; }
        }
        float pacc = 0.f;
        #pragma unroll
        for (int jg = 0; jg < 4; ++jg) {
            const float p0 = __expf(sraw[jg][0] - rmx);
            const float p1 = __expf(sraw[jg][1] - rmx);
            pacc += p0 + p1;
            mypt[g * GVE_PT_PITCH + jg * 8 + 2 * tq]     = __float2bfloat16(p0);
            mypt[g * GVE_PT_PITCH + jg * 8 + 2 * tq + 1] = __float2bfloat16(p1);
        }
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 1);
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 2);
        l += pacc;
        // NOTE: rows 8..15 of this warp's P tile are never written — the PV A-frag's second
        // row-half reads that stale garbage, but its products land ONLY in the C-frag halves
        // this kernel discards (o keeps row g's columns); a NaN there cannot cross rows.
        #pragma unroll
        for (int kp = 0; kp < 32; kp += 16) {
            const __nv_bfloat16* prow = mypt + g * GVE_PT_PITCH + kp + 4 * tq;
            unsigned aP[4] = {
                *(const unsigned*)prow,
                *(const unsigned*)(prow + 8 * GVE_PT_PITCH),
                *(const unsigned*)(prow + 2),
                *(const unsigned*)(prow + 8 * GVE_PT_PITCH + 2) };
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const __nv_bfloat16* vrow = vt2 + (nt * 8 + g) * GVE_PT_PITCH + kp + 4 * tq;
                unsigned bV[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)vrow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(vrow + 2)) };
                float dd[4] = { 0.f, 0.f, 0.f, 0.f };
                gve_mma(dd, aP, bV);
                o[nt][0] += dd[0]; o[nt][1] += dd[1];   // keep row g's cols only
            }
        }
        __syncwarp();
        __syncthreads();
    }
    // Emit partials in the SPLITK layout for gqa_attn_reduce_e.
    const int qi = qrow0 + g;
    if (qi < NQ) {
        const int b = qi / QG, qh = kvh * QG + (qi % QG);
        const long long base = ((long long)b * nh + qh) * ns_grid + seg;
        if (tq == 0) { out_m[base] = m; out_l[base] = l; }
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            out_acc[base * HDT + nt * 8 + 2 * tq]     = o[nt][0];
            out_acc[base * HDT + nt * 8 + 2 * tq + 1] = o[nt][1];
        }
    }
}

// gve_k8v8_impl — p3 core on the k8v8 packed cache: 16-key tiles, register-staged pipeline;
// stage phase dequantizes inline int8+fp16-block rows to the same bf16 kt/vt2 the p3 mma path
// consumes (deterministic per position — same values every reader sees). Block mapping: each
// thread owns ONE 16-dim block (20 B) of ONE key per pipeline slot; a 16-key tile has
// 16*(HDT/16) blocks (256 @ hd256) over 192 threads => r in {0,1}.
template <int HDT>
__device__ __forceinline__ void gve_k8v8_impl(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache, const int* __restrict__ pos,
    long long bs_packed, int nh_packed, const int* __restrict__ slot_ids) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;
    const float scale = 1.0f / sqrtf((float)HDT);
    const int QG = nh / nkv, NQ = B * QG;           // row groups: gridDim.z = ceil(NQ/48)
    const int KSTEPS = HDT / 16;                    // 8 | 16
    const int NTILES = HDT / 8;                     // 16 | 32
    const int kvh = blockIdx.x, seg = blockIdx.y, SEG = gridDim.y;

    int max_pc = 0;
    for (int b = 0; b < B; ++b) max_pc = max(max_pc, pos[b] + 1);
    // F9 (batch invariance): the key-range partition must be a function of the REQUEST, never of
    // the batch width. Partitioning from this batch's max_pc gave a decode (pc=p+1) and a verify
    // (pc=p+N) different seg_size -> different softmax fold boundaries for column 0 -> ulp drift
    // -> argmax flips at near-ties (the F9 ladder: 10/16 clean). The bound is pos[0] +
    // GVE_MAX_VERIFY: pos[0] is the first column's position (identical for a decode and a verify
    // at the same position, and >= this batch's max_pc for every width <= GVE_MAX_VERIFY). It is
    // read from pos[] (rewritten per call AND per graph replay), never from a baked launch arg.
    // `jend` still stops at this batch's max_pc: tiles wholly beyond a column's own rbnd are exact
    // no-ops (scores -1e30 -> P=0; the fold's rescale is exp(0)=1).
    // `max(max_pc, ...)` is belt-and-braces: the _e lane is host-gated to single-request
    // forwards (decode b=1 / chain verify), where pos[] is contiguous and max_pc <=
    // pos[0]+16, so the max is exactly pos[0]+16; if a multi-request batch ever reaches
    // this lane, the bound degrades to the old max_pc (correct, just not width-invariant).
    // F9 (phase-8, the depth-2 residue): pos[0]+GVE_MAX_VERIFY aligns a verify col 0 with its
    // decode, but a verify col j>=1 is compared against a decode at pos[0]+j — whose pos[0]
    // differs, whose pc_bound differs, whose seg_size differs => different softmax fold
    // boundaries => 1-ulp drift on EVERY col>=1 (argmax flips at near-ties ~1/512 tokens;
    // single-proc repro: bench-mtp d2 512tok, MISMATCH at 442). The partition must key on a
    // per-REQUEST constant so decode and verify at ANY width resolve identical boundaries:
    // `stride` (the KV row stride) is exactly that — identical launch-to-launch for the whole
    // life of the request, and the segments beyond max_pc exit as exact no-ops via jend.
    const int pc_bound = (int)stride;
    const int seg_size = (pc_bound + SEG - 1) / SEG;
    const int jbegin = seg * seg_size;
    const int jend = min(max_pc, jbegin + seg_size);

    __shared__ __nv_bfloat16 kt[16 * HDT];
    __shared__ __nv_bfloat16 vt2[HDT * 24];
    __shared__ __nv_bfloat16 pt[6 * 16 * 24];   // p3: 16 key cols + pad

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;              // 0..5 — owns REAL rows w*8..w*8+7
    const int g    = lane >> 2, tq = lane & 3;      // mma frag coords
    const int qrow0 = w * 8 + blockIdx.z * 48;      // this lane's real row; grid.z = row group

    // This row's causal bound (pos[b]+1); rows >= NQ park everything.
    const int rbnd = (qrow0 + g < NQ) ? pos[(qrow0 + g) / QG] + 1 : 0;

    // Q row-fragments from GLOBAL (once; ~24 KB total, L2-resident): row (qrow0+g) of this
    // kvh's group, k-chunk ks16. A-frag layout: a[0],a[2] = row g cols 4tq..4tq+3; a[1],a[3]
    // = row g+8 (PARKED rows: zero — their scores stay -1e30, never emitted).
    unsigned qa[KSTEPS][4];
    #pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        qa[ks][0] = qa[ks][2] = 0u;
        qa[ks][1] = qa[ks][3] = 0u;
        if (qrow0 + g < NQ) {
            const __nv_bfloat16* q0 = q + (long long)((qrow0 + g) / QG) * q_pitch
                                     + (long long)(kvh * QG + ((qrow0 + g) % QG)) * HDT
                                     + ks * 16 + 4 * tq;
            qa[ks][0] = *(const unsigned*)q0;
            qa[ks][2] = *(const unsigned*)(q0 + 2);
        }
    }
    float m = -1e30f, l = 0.f;
    float o[NTILES][2];                             // real row g only (cols 2tq, 2tq+1)
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] = 0.f; o[nt][1] = 0.f; }

    const long long kvbase = ((long long)slot_ids[0] * nkv + kvh) * (long long)stride;
    const uint8_t* kbase = k_cache + kvbase * (long long)KV8_ROW_BYTES(HDT);
    const uint8_t* vbase = v_cache + kvbase * (long long)KV8_ROW_BYTES(HDT);

    uint4 pkc[2], pvc[2]; unsigned short pks[2], pvs[2]; int pkey[2], pblk[2];
    const int NBLK = HDT / 16;
    // initial prefetch of tile jbegin (16 keys x NBLK blocks)
    {
        const int tv0 = min(16, max(0, jend - jbegin));
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int i = threadIdx.x + r * blockDim.x;
            pkey[r] = i / NBLK; pblk[r] = i % NBLK;
            if (i < tv0 * NBLK) {
                const unsigned char* kr = kbase + (long long)(jbegin + pkey[r]) * KV8_ROW_BYTES(HDT) + pblk[r] * 20;
                const uint* kru = (const uint*)kr;                 // 4-B loads: blk*20 is NOT 16-B aligned
                pkc[r] = make_uint4(kru[0], kru[1], kru[2], kru[3]);
                pks[r] = *(const unsigned short*)(kr + 16);
                const unsigned char* vr = vbase + (long long)(jbegin + pkey[r]) * KV8_ROW_BYTES(HDT) + pblk[r] * 20;
                const uint* vru = (const uint*)vr;
                pvc[r] = make_uint4(vru[0], vru[1], vru[2], vru[3]);
                pvs[r] = *(const unsigned short*)(vr + 16);
            }
        }
    }
    for (int t0 = jbegin; t0 < jend; t0 += 16) {
        const int tv = min(16, jend - t0);
        __syncthreads();                            // tile t-1 consumed
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int key = pkey[r], blk = pblk[r];
            if (key < tv) {
                const float ks = __half2float(__ushort_as_half(pks[r]));
                const float vs = __half2float(__ushort_as_half(pvs[r]));
                const signed char* kc = (const signed char*)&pkc[r];
                const signed char* vc = (const signed char*)&pvc[r];
                #pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    *(__nv_bfloat162*)&kt[key * HDT + blk * 16 + i] =
                        __floats2bfloat162_rn(kc[i] * ks, kc[i + 1] * ks);
                    vt2[(blk * 16 + i) * 24 + key]     = __float2bfloat16_rn(vc[i] * vs);
                    vt2[(blk * 16 + i + 1) * 24 + key] = __float2bfloat16_rn(vc[i + 1] * vs);
                }
            }
        }
        if (tv < 16) {                               // 0*NaN guard (see header)
            for (int i = threadIdx.x; i < (16 - tv) * HDT; i += blockDim.x) {
                const int key = tv + i / HDT, col = i % HDT;
                vt2[col * 24 + key] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();                             // tile t staged
        {
            const int t1 = t0 + 16;
            const int tv1 = min(16, max(0, jend - t1));
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int i = threadIdx.x + r * blockDim.x;
                pkey[r] = i / NBLK; pblk[r] = i % NBLK;
                if (i < tv1 * NBLK) {
                    const unsigned char* kr = kbase + (long long)(t1 + pkey[r]) * KV8_ROW_BYTES(HDT) + pblk[r] * 20;
                    const uint* kru = (const uint*)kr;
                    pkc[r] = make_uint4(kru[0], kru[1], kru[2], kru[3]);
                    pks[r] = *(const unsigned short*)(kr + 16);
                    const unsigned char* vr = vbase + (long long)(t1 + pkey[r]) * KV8_ROW_BYTES(HDT) + pblk[r] * 20;
                    const uint* vru = (const uint*)vr;
                    pvc[r] = make_uint4(vru[0], vru[1], vru[2], vru[3]);
                    pvs[r] = *(const unsigned short*)(vr + 16);
                }
            }
        }
        __nv_bfloat16* mypt = pt + w * 16 * 24;
        // ---- scores + single-fold-per-tile softmax (4 key-groups of 8) ----
        float sraw[2][2];
        float rmx = m;
        #pragma unroll
        for (int jg = 0; jg < 2; ++jg) {
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            #pragma unroll
            for (int ks = 0; ks < KSTEPS; ++ks) {
                const __nv_bfloat16* krow = kt + (jg * 8 + g) * HDT + ks * 16 + 4 * tq;
                unsigned bK[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)krow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(krow + 2)) };
                gve_mma(s, qa[ks], bK);
            }
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = jg * 8 + 2 * tq + e;
                sraw[jg][e] = (key < tv && (t0 + key) < rbnd) ? s[e] * scale : -1e30f;
            }
            float rh = fmaxf(sraw[jg][0], sraw[jg][1]);
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 1));
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 2));
            rmx = fmaxf(rmx, rh);
        }
        {                                           // one rescale to the tile-final basis
            const float rs = __expf(m - rmx);       // m=-1e30 first -> 0, no NaN
            l *= rs; m = rmx;
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] *= rs; o[nt][1] *= rs; }
        }
        float pacc = 0.f;
        #pragma unroll
        for (int jg = 0; jg < 2; ++jg) {
            const float p0 = __expf(sraw[jg][0] - rmx);
            const float p1 = __expf(sraw[jg][1] - rmx);
            pacc += p0 + p1;
            mypt[g * 24 + jg * 8 + 2 * tq]     = __float2bfloat16(p0);
            mypt[g * 24 + jg * 8 + 2 * tq + 1] = __float2bfloat16(p1);
        }
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 1);
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 2);
        l += pacc;
        // NOTE: rows 8..15 of this warp's P tile are never written — the PV A-frag's second
        // row-half reads that stale garbage, but its products land ONLY in the C-frag halves
        // this kernel discards (o keeps row g's columns); a NaN there cannot cross rows.
        #pragma unroll
        for (int kp = 0; kp < 16; kp += 16) {
            const __nv_bfloat16* prow = mypt + g * 24 + kp + 4 * tq;
            unsigned aP[4] = {
                *(const unsigned*)prow,
                *(const unsigned*)(prow + 8 * 24),
                *(const unsigned*)(prow + 2),
                *(const unsigned*)(prow + 8 * 24 + 2) };
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const __nv_bfloat16* vrow = vt2 + (nt * 8 + g) * 24 + kp + 4 * tq;
                unsigned bV[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)vrow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(vrow + 2)) };
                float dd[4] = { 0.f, 0.f, 0.f, 0.f };
                gve_mma(dd, aP, bV);
                o[nt][0] += dd[0]; o[nt][1] += dd[1];   // keep row g's cols only
            }
        }
        __syncwarp();
    }
    // Emit partials in the SPLITK layout for gqa_attn_reduce_e.
    const int qi = qrow0 + g;
    if (qi < NQ) {
        const int b = qi / QG, qh = kvh * QG + (qi % QG);
        const long long base = ((long long)b * nh + qh) * ns_grid + seg;
        if (tq == 0) { out_m[base] = m; out_l[base] = l; }
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            out_acc[base * HDT + nt * 8 + 2 * tq]     = o[nt][0];
            out_acc[base * HDT + nt * 8 + 2 * tq + 1] = o[nt][1];
        }
    }
}





// k8v8 direct-read verify attention (p8b): the p3 mma/softmax/PV core on the packed int8
// cache. hd 128/256. Not yet dispatched by Rust (write path + arms land with the mode enum).
extern "C" __global__ void __launch_bounds__(192) gqa_attn_verify_e_k8v8(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ pos, long long bs_packed, int nh_packed,
    const int* __restrict__ slot_ids, const int* __restrict__ col_pos_start) {
    (void)col_pos_start;
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128)      gve_k8v8_impl<128>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
    else if (hd == 256) gve_k8v8_impl<256>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
}

template <int HDT>                                   // hd = 128 | 256
__device__ __forceinline__ void gve_p3_impl(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ pos,
    long long bs_packed, int nh_packed, const int* __restrict__ slot_ids) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;
    const float scale = 1.0f / sqrtf((float)HDT);
    const int QG = nh / nkv, NQ = B * QG;           // row groups: gridDim.z = ceil(NQ/48)
    const int KSTEPS = HDT / 16;                    // 8 | 16
    const int NTILES = HDT / 8;                     // 16 | 32
    const int kvh = blockIdx.x, seg = blockIdx.y, SEG = gridDim.y;

    int max_pc = 0;
    for (int b = 0; b < B; ++b) max_pc = max(max_pc, pos[b] + 1);
    // F9 (batch invariance): the key-range partition must be a function of the REQUEST, never of
    // the batch width. Partitioning from this batch's max_pc gave a decode (pc=p+1) and a verify
    // (pc=p+N) different seg_size -> different softmax fold boundaries for column 0 -> ulp drift
    // -> argmax flips at near-ties (the F9 ladder: 10/16 clean). The bound is pos[0] +
    // GVE_MAX_VERIFY: pos[0] is the first column's position (identical for a decode and a verify
    // at the same position, and >= this batch's max_pc for every width <= GVE_MAX_VERIFY). It is
    // read from pos[] (rewritten per call AND per graph replay), never from a baked launch arg.
    // `jend` still stops at this batch's max_pc: tiles wholly beyond a column's own rbnd are exact
    // no-ops (scores -1e30 -> P=0; the fold's rescale is exp(0)=1).
    // `max(max_pc, ...)` is belt-and-braces: the _e lane is host-gated to single-request
    // forwards (decode b=1 / chain verify), where pos[] is contiguous and max_pc <=
    // pos[0]+16, so the max is exactly pos[0]+16; if a multi-request batch ever reaches
    // this lane, the bound degrades to the old max_pc (correct, just not width-invariant).
    // F9 (phase-8, the depth-2 residue): pos[0]+GVE_MAX_VERIFY aligns a verify col 0 with its
    // decode, but a verify col j>=1 is compared against a decode at pos[0]+j — whose pos[0]
    // differs, whose pc_bound differs, whose seg_size differs => different softmax fold
    // boundaries => 1-ulp drift on EVERY col>=1 (argmax flips at near-ties ~1/512 tokens;
    // single-proc repro: bench-mtp d2 512tok, MISMATCH at 442). The partition must key on a
    // per-REQUEST constant so decode and verify at ANY width resolve identical boundaries:
    // `stride` (the KV row stride) is exactly that — identical launch-to-launch for the whole
    // life of the request, and the segments beyond max_pc exit as exact no-ops via jend.
    const int pc_bound = (int)stride;
    const int seg_size = (pc_bound + SEG - 1) / SEG;
    const int jbegin = seg * seg_size;
    const int jend = min(max_pc, jbegin + seg_size);

    __shared__ __nv_bfloat16 kt[16 * HDT];
    __shared__ __nv_bfloat16 vt2[HDT * 24];
    __shared__ __nv_bfloat16 pt[6 * 16 * 24];   // p3: 16 key cols + pad

    const int lane = threadIdx.x & 31;
    const int w    = threadIdx.x >> 5;              // 0..5 — owns REAL rows w*8..w*8+7
    const int g    = lane >> 2, tq = lane & 3;      // mma frag coords
    const int qrow0 = w * 8 + blockIdx.z * 48;      // this lane's real row; grid.z = row group

    // This row's causal bound (pos[b]+1); rows >= NQ park everything.
    const int rbnd = (qrow0 + g < NQ) ? pos[(qrow0 + g) / QG] + 1 : 0;

    // Q row-fragments from GLOBAL (once; ~24 KB total, L2-resident): row (qrow0+g) of this
    // kvh's group, k-chunk ks16. A-frag layout: a[0],a[2] = row g cols 4tq..4tq+3; a[1],a[3]
    // = row g+8 (PARKED rows: zero — their scores stay -1e30, never emitted).
    unsigned qa[KSTEPS][4];
    #pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        qa[ks][0] = qa[ks][2] = 0u;
        qa[ks][1] = qa[ks][3] = 0u;
        if (qrow0 + g < NQ) {
            const __nv_bfloat16* q0 = q + (long long)((qrow0 + g) / QG) * q_pitch
                                     + (long long)(kvh * QG + ((qrow0 + g) % QG)) * HDT
                                     + ks * 16 + 4 * tq;
            qa[ks][0] = *(const unsigned*)q0;
            qa[ks][2] = *(const unsigned*)(q0 + 2);
        }
    }
    float m = -1e30f, l = 0.f;
    float o[NTILES][2];                             // real row g only (cols 2tq, 2tq+1)
    #pragma unroll
    for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] = 0.f; o[nt][1] = 0.f; }

    const long long kvbase = ((long long)slot_ids[0] * nkv + kvh) * (long long)stride;
    const __nv_bfloat16* kbase = k_cache + kvbase * HDT;
    const __nv_bfloat16* vbase = v_cache + kvbase * HDT;

    uint4 p3_kreg[3]; uint4 p3_vreg[3]; int p3_row[3]; int p3_col[3];
    {
        const int tv0 = min(16, max(0, jend - jbegin));
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            const int i = threadIdx.x * 8 + r * blockDim.x * 8;
            const int row = i / HDT, col = i % HDT;
            p3_row[r] = row; p3_col[r] = col;
            if (row < tv0) {
                p3_kreg[r] = *(const uint4*)&kbase[(long long)(jbegin + row) * HDT + col];
                *(uint4*)&p3_vreg[r] = *(const uint4*)&vbase[(long long)(jbegin + row) * HDT + col];
            }
        }
    }
    for (int t0 = jbegin; t0 < jend; t0 += 16) {
        const int tv = min(16, jend - t0);
        __syncthreads();                            // tile t-1 consumed
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            const int row = p3_row[r], col = p3_col[r];
            if (row < tv) {
                *(uint4*)&kt[row * HDT + col] = p3_kreg[r];
                const __nv_bfloat16* v8 = reinterpret_cast<const __nv_bfloat16*>(&p3_vreg[r]);
                #pragma unroll
                for (int k = 0; k < 8; ++k) vt2[(col + k) * 24 + row] = v8[k];
            }
        }
        if (tv < 16) {                              // 0*NaN guard (see header)
            for (int i = threadIdx.x; i < (16 - tv) * HDT; i += blockDim.x) {
                const int key = tv + i / HDT, col = i % HDT;
                vt2[col * 24 + key] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();                            // tile t staged
        {
            const int t1 = t0 + 16;
            const int tv1 = min(16, max(0, jend - t1));
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                const int i = threadIdx.x * 8 + r * blockDim.x * 8;
                const int row = i / HDT, col = i % HDT;
                p3_row[r] = row; p3_col[r] = col;
                if (row < tv1) {
                    p3_kreg[r] = *(const uint4*)&kbase[(long long)(t1 + row) * HDT + col];
                    *(uint4*)&p3_vreg[r] = *(const uint4*)&vbase[(long long)(t1 + row) * HDT + col];
                }
            }
        }
        __nv_bfloat16* mypt = pt + w * 16 * 24;
        // ---- scores + single-fold-per-tile softmax (4 key-groups of 8) ----
        float sraw[2][2];
        float rmx = m;
        #pragma unroll
        for (int jg = 0; jg < 2; ++jg) {
            float s[4] = { 0.f, 0.f, 0.f, 0.f };
            #pragma unroll
            for (int ks = 0; ks < KSTEPS; ++ks) {
                const __nv_bfloat16* krow = kt + (jg * 8 + g) * HDT + ks * 16 + 4 * tq;
                unsigned bK[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)krow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(krow + 2)) };
                gve_mma(s, qa[ks], bK);
            }
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = jg * 8 + 2 * tq + e;
                sraw[jg][e] = (key < tv && (t0 + key) < rbnd) ? s[e] * scale : -1e30f;
            }
            float rh = fmaxf(sraw[jg][0], sraw[jg][1]);
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 1));
            rh = fmaxf(rh, __shfl_xor_sync(0xffffffffu, rh, 2));
            rmx = fmaxf(rmx, rh);
        }
        {                                           // one rescale to the tile-final basis
            const float rs = __expf(m - rmx);       // m=-1e30 first -> 0, no NaN
            l *= rs; m = rmx;
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) { o[nt][0] *= rs; o[nt][1] *= rs; }
        }
        float pacc = 0.f;
        #pragma unroll
        for (int jg = 0; jg < 2; ++jg) {
            const float p0 = __expf(sraw[jg][0] - rmx);
            const float p1 = __expf(sraw[jg][1] - rmx);
            pacc += p0 + p1;
            mypt[g * 24 + jg * 8 + 2 * tq]     = __float2bfloat16(p0);
            mypt[g * 24 + jg * 8 + 2 * tq + 1] = __float2bfloat16(p1);
        }
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 1);
        pacc += __shfl_xor_sync(0xffffffffu, pacc, 2);
        l += pacc;
        // NOTE: rows 8..15 of this warp's P tile are never written — the PV A-frag's second
        // row-half reads that stale garbage, but its products land ONLY in the C-frag halves
        // this kernel discards (o keeps row g's columns); a NaN there cannot cross rows.
        #pragma unroll
        for (int kp = 0; kp < 16; kp += 16) {
            const __nv_bfloat16* prow = mypt + g * 24 + kp + 4 * tq;
            unsigned aP[4] = {
                *(const unsigned*)prow,
                *(const unsigned*)(prow + 8 * 24),
                *(const unsigned*)(prow + 2),
                *(const unsigned*)(prow + 8 * 24 + 2) };
            #pragma unroll
            for (int nt = 0; nt < NTILES; ++nt) {
                const __nv_bfloat16* vrow = vt2 + (nt * 8 + g) * 24 + kp + 4 * tq;
                unsigned bV[2] = {
                    gve_bf162_b32(*(const __nv_bfloat162*)vrow),
                    gve_bf162_b32(*(const __nv_bfloat162*)(vrow + 2)) };
                float dd[4] = { 0.f, 0.f, 0.f, 0.f };
                gve_mma(dd, aP, bV);
                o[nt][0] += dd[0]; o[nt][1] += dd[1];   // keep row g's cols only
            }
        }
        __syncwarp();
    }
    // Emit partials in the SPLITK layout for gqa_attn_reduce_e.
    const int qi = qrow0 + g;
    if (qi < NQ) {
        const int b = qi / QG, qh = kvh * QG + (qi % QG);
        const long long base = ((long long)b * nh + qh) * ns_grid + seg;
        if (tq == 0) { out_m[base] = m; out_l[base] = l; }
        #pragma unroll
        for (int nt = 0; nt < NTILES; ++nt) {
            out_acc[base * HDT + nt * 8 + 2 * tq]     = o[nt][0];
            out_acc[base * HDT + nt * 8 + 2 * tq + 1] = o[nt][1];
        }
    }
}



