// gpu_batch/06_splitk_k8v4_k8v8.cuh: gqa_attn_splitk k8v4 / k8v8 and the GQA-packed twins.
// SAME split structure, SAME reduction order, SAME merge as the q4 kernel: the lossless-MTP contract (decode == verify col-0).
// Section 6 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================================================================================================
// gqa_attn_splitk_k8v4 — the k8v4 twin of gqa_attn_splitk_q4: SAME split structure, SAME
// reduction order, SAME merge — the lossless-MTP contract (decode == verify col-0, every split
// produces the same fp32 score for a (kvh, pos)). Only the K source changes: each lane reads its
// 4 consecutive int8 codes as one aligned u32 (the 20 B/16 block is 4-B-aligned throughout, and
// DPL divides 16 so a lane never straddles blocks) and multiplies by the fp32(fp16 scale) —
// kdq[i] = (float)((int8_t)(codes >> 8i)) * ksc, the doc's dequant formula with both upcasts
// exact and one rounding. The V path is verbatim from the q4 kernel (same nibble reader, same
// 12 B/16 row stride — the V cache layout is unchanged).
extern "C" __global__ void gqa_attn_splitk_k8v4(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;   // F0: q row pitch (mtot fused view / nh*hd split)

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
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
    const int DPL = hd >> 5;

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows diverged: 20 B/16
    const int v_rb = KVQ_ROW_BYTES(hd);              // V rows: the unchanged q4 12 B/16
    const int lane_blk = (lane * DPL) / KVQ_BLK;     // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;     // its first code within the block (4-aligned)
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 12;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd];
        const unsigned char* krow = kb + (long long)t * k_rb;
        const float ksc = __half2float(__ushort_as_half(*(const unsigned short*)(krow + KVQ_BLK)));
        const uint32_t* kcodes = (const uint32_t*)(krow + lane_off);   // 4 aligned int8 codes
        float kdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) kdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const uint32_t codes = kcodes[u];
            kdq[u * 4 + 0] = (float)((int8_t)(codes >> 0)) * ksc;
            kdq[u * 4 + 1] = (float)((int8_t)(codes >> 8)) * ksc;
            kdq[u * 4 + 2] = (float)((int8_t)(codes >> 16)) * ksc;
            kdq[u * 4 + 3] = (float)((int8_t)(codes >> 24)) * ksc;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * kdq[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const unsigned char* vrow = vb + (long long)t * v_rb;
        const float vsc = e4m3_f(vrow[8]);
        const unsigned short* vcodes = (const unsigned short*)(vrow + lane_off / 2);
        float vdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) vdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const int pk = (int)vcodes[u];   // SIGNED: `(pk & 0xF) - 7` in unsigned wraps nibble<7 to ~4.3e9
            vdq[u * 4 + 0] = (float)((pk & 0xF) - 7) * vsc;
            vdq[u * 4 + 1] = (float)(((pk >> 4) & 0xF) - 7) * vsc;
            vdq[u * 4 + 2] = (float)(((pk >> 8) & 0xF) - 7) * vsc;
            vdq[u * 4 + 3] = (float)(((pk >> 12) & 0xF) - 7) * vsc;
        }
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * vdq[i];
        m = m_new;
        l = l * a_old + a_cur;
    }

    extern __shared__ float sh[];
    float* sacc = sh;
    float* sm   = sh + NW * hd;
    float* sl   = sm + NW;
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) sacc[warp * hd + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

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
}

// ===================================================================================================
// gqa_attn_splitk_q4_gq — GQA-PACKED twin of gqa_attn_splitk_q4: one block per (kv head, split)
// reads the K/V chunk ONCE and applies the whole GQA group, instead of one block per QUERY head
// re-reading the same chunk gqa_ratio times. The per-head kernel streams nh × the KV bytes; at
// Hy3's 8:1 GQA and ≥26K ctx that is 13 GB/token at 242 GB/s ≈ 53.6 ms/token of pure re-reads
// (L2 dedupes the group reads at ≤16K — 12 MB/layer — but not at 20+ MB/layer). This kernel cuts
// decode-attention DRAM traffic by gqa_ratio×: the "32K anomaly" (E1) dies here.
//
// BIT-IDENTICAL to gqa_attn_splitk_q4 per head, by construction: same split semantics
// (sk_nsplits on the column's own pc), same row→warp assignment (r = start+warp, += NW), same
// per-lane dequant of its DPL slice, same shfl halving tree per head, same per-row online-softmax
// update per head, same fixed-warp-order merge — the K/V row is simply dequantized once per
// GROUP instead of once per head. out_m/out_l/out_acc slots match the per-head kernel to the
// last bit, so decode == verify col-0 (the lossless-MTP contract) is preserved, and mixing this
// kernel with the per-head one across a dispatch boundary is numerically safe.
//
// Scope: hd == 128 (DPL 4) and gqa_ratio ≤ SK_GQA_MAX — the q4 KV path is Hy3-only today; other
// geometries fall back to the per-head kernel at the launch site. Same args/layout.
#define SK_GQA_MAX 8

template<int DPL_T>
__device__ __forceinline__ void gqa_splitk_q4_gq_impl(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
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

    // Per-lane q slices for the whole group, in registers. g >= gqa_ratio is clamped to a valid
    // (duplicated) row and predicated to exact +0.0f — its outputs are never read.
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

    const int rb = KVQ_ROW_BYTES(hd);
    const int lane_blk = (lane * DPL_T) / KVQ_BLK;
    const int lane_off = (lane * DPL_T) % KVQ_BLK;
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride * rb;
    const unsigned char* kb = k_cache + kvbase + lane_blk * 12;
    const unsigned char* vb = v_cache + kvbase + lane_blk * 12;
    // E1b 2-row software pipeline: at long context this loop is LATENCY-bound (each row's K-then-V
    // dependent loads ~1.9K cycles back-to-back, only 4 warps/SM to hide them). Process rows in
    // pairs with ALL loads issued up front — one memory round-trip per pair instead of two —
    // then apply the two online-softmax updates sequentially in the same ascending order as
    // before. Every per-head FP expression keeps its exact form and order (dot, tree, scale,
    // m/a, acc), so the result is BIT-IDENTICAL to the unpipelined loop (and to gqa_attn_splitk_q4).
    for (int r = start + warp; r < end; r += 2 * NW) {
        const int r2 = r + NW;
        const bool has2 = r2 < end;
        const int dd = r - pos_start;
        const int t  = (!path || dd < 0) ? r  : pos_start + (int)path[b * MAX_VERIFY + dd];
        const int t2 = has2 ? ((!path || (r2 - pos_start) < 0) ? r2 : pos_start + (int)path[b * MAX_VERIFY + (r2 - pos_start)]) : 0;
        // up-front loads for BOTH rows (independent — latency paid once)
        const unsigned char* krow  = kb + (long long)t * rb;
        const unsigned char* krow2 = kb + (long long)t2 * rb;
        const unsigned char* vrow  = vb + (long long)t * rb;
        const unsigned char* vrow2 = vb + (long long)t2 * rb;
        const float ksc  = e4m3_f(krow[8]);
        const float ksc2 = has2 ? e4m3_f(krow2[8]) : 0.0f;
        const float vsc  = e4m3_f(vrow[8]);
        const float vsc2 = has2 ? e4m3_f(vrow2[8]) : 0.0f;
        const unsigned short* kcodes  = (const unsigned short*)(krow  + lane_off / 2);
        const unsigned short* kcodes2 = (const unsigned short*)(krow2 + lane_off / 2);
        const unsigned short* vcodes  = (const unsigned short*)(vrow  + lane_off / 2);
        const unsigned short* vcodes2 = (const unsigned short*)(vrow2 + lane_off / 2);
        float kdq[DPL_T], kdq2[DPL_T], vdq[DPL_T], vdq2[DPL_T];
        #pragma unroll
        for (int u = 0; u < DPL_T / 4; u++) {
            const int pk  = (int)kcodes[u];    // SIGNED: unsigned nibble math wraps nibble<7 to ~4.3e9
            const int pk2 = has2 ? (int)kcodes2[u] : 0;
            const int pv  = (int)vcodes[u];
            const int pv2 = has2 ? (int)vcodes2[u] : 0;
            kdq[u * 4 + 0]  = (float)((pk & 0xF) - 7) * ksc;
            kdq[u * 4 + 1]  = (float)(((pk >> 4) & 0xF) - 7) * ksc;
            kdq[u * 4 + 2]  = (float)(((pk >> 8) & 0xF) - 7) * ksc;
            kdq[u * 4 + 3]  = (float)(((pk >> 12) & 0xF) - 7) * ksc;
            kdq2[u * 4 + 0] = (float)((pk2 & 0xF) - 7) * ksc2;
            kdq2[u * 4 + 1] = (float)(((pk2 >> 4) & 0xF) - 7) * ksc2;
            kdq2[u * 4 + 2] = (float)(((pk2 >> 8) & 0xF) - 7) * ksc2;
            kdq2[u * 4 + 3] = (float)(((pk2 >> 12) & 0xF) - 7) * ksc2;
            vdq[u * 4 + 0]  = (float)((pv & 0xF) - 7) * vsc;
            vdq[u * 4 + 1]  = (float)(((pv >> 4) & 0xF) - 7) * vsc;
            vdq[u * 4 + 2]  = (float)(((pv >> 8) & 0xF) - 7) * vsc;
            vdq[u * 4 + 3]  = (float)(((pv >> 12) & 0xF) - 7) * vsc;
            vdq2[u * 4 + 0] = (float)((pv2 & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 1] = (float)(((pv2 >> 4) & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 2] = (float)(((pv2 >> 8) & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 3] = (float)(((pv2 >> 12) & 0xF) - 7) * vsc2;
        }
        // dots for both rows (independent of the running softmax state)
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
        // then the two state updates, ascending — identical sequence to the unpipelined loop.
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

    // Merge per head into one smem buffer, sequentially — each head's merge is the per-head
    // kernel's exact fixed-warp-order one.
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

extern "C" __global__ void gqa_attn_splitk_q4_gq(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128) {
        gqa_splitk_q4_gq_impl<4>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                 pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    } else if (hd == 256) {
        gqa_splitk_q4_gq_impl<8>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                 pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    }
    // other hd: never launched — attn_dispatch falls back to the per-head kernel.
}

// ===================================================================================================
// gqa_attn_splitk_k8v4_gq — the k8v4 twin of gqa_attn_splitk_q4_gq: one block per (kv head, split)
// reads the K/V chunk ONCE for the whole GQA group (the E1 win that ships hy3's 8:1 GQA), with the
// E1b 2-row software pipeline (all loads up front, the two softmax updates applied ascending).
// BIT-IDENTICAL to gqa_attn_splitk_k8v4 per head by the same construction argument as the q4 pair:
// same split semantics, same row->warp assignment, same per-lane dequant of its DPL slice, same
// shfl halving tree per head, same online-softmax update order, same fixed-warp-order merge.
// K loads become u32-per-lane (4 aligned int8 codes, fp16 scale); the V path is verbatim from
// the q4_gq kernel. K and V caches diverge in row size (20 B/16 vs 12 B/16), so the two bases
// are computed separately.
#define SK_GQA_MAX 8

template<int DPL_T>
__device__ __forceinline__ void gqa_splitk_k8v4_gq_impl(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
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
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;   // F0: q row pitch (mtot fused view / nh*hd split)

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

    // Per-lane q slices for the whole group, in registers. g >= gqa_ratio is clamped to a valid
    // (duplicated) row and predicated to exact +0.0f — its outputs are never read.
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

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows diverged: 20 B/16
    const int v_rb = KVQ_ROW_BYTES(hd);              // V rows: the unchanged q4 12 B/16
    const int lane_blk = (lane * DPL_T) / KVQ_BLK;
    const int lane_off = (lane * DPL_T) % KVQ_BLK;   // 4-aligned (DPL_T divides 16)
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 12;
    // E1b 2-row software pipeline: at long context this loop is LATENCY-bound (each row's K-then-V
    // dependent loads ~1.9K cycles back-to-back, only 4 warps/SM to hide them). Process rows in
    // pairs with ALL loads issued up front — one memory round-trip per pair instead of two —
    // then apply the two online-softmax updates sequentially in the same ascending order as
    // before. Every per-head FP expression keeps its exact form and order (dot, tree, scale,
    // m/a, acc), so the result is BIT-IDENTICAL to the unpipelined loop (and to gqa_attn_splitk_k8v4).
    for (int r = start + warp; r < end; r += 2 * NW) {
        const int r2 = r + NW;
        const bool has2 = r2 < end;
        const int dd = r - pos_start;
        const int t  = (!path || dd < 0) ? r  : pos_start + (int)path[b * MAX_VERIFY + dd];
        const int t2 = has2 ? ((!path || (r2 - pos_start) < 0) ? r2 : pos_start + (int)path[b * MAX_VERIFY + (r2 - pos_start)]) : 0;
        // up-front loads for BOTH rows (independent — latency paid once)
        const unsigned char* krow  = kb + (long long)t * k_rb;
        const unsigned char* krow2 = kb + (long long)t2 * k_rb;
        const unsigned char* vrow  = vb + (long long)t * v_rb;
        const unsigned char* vrow2 = vb + (long long)t2 * v_rb;
        const float ksc  = __half2float(__ushort_as_half(*(const unsigned short*)(krow  + KVQ_BLK)));
        const float ksc2 = has2 ? __half2float(__ushort_as_half(*(const unsigned short*)(krow2 + KVQ_BLK))) : 0.0f;
        const float vsc  = e4m3_f(vrow[8]);
        const float vsc2 = has2 ? e4m3_f(vrow2[8]) : 0.0f;
        const uint32_t* kcodes  = (const uint32_t*)(krow  + lane_off);
        const uint32_t* kcodes2 = (const uint32_t*)(krow2 + lane_off);
        const unsigned short* vcodes  = (const unsigned short*)(vrow  + lane_off / 2);
        const unsigned short* vcodes2 = (const unsigned short*)(vrow2 + lane_off / 2);
        float kdq[DPL_T], kdq2[DPL_T], vdq[DPL_T], vdq2[DPL_T];
        #pragma unroll
        for (int u = 0; u < DPL_T / 4; u++) {
            const uint32_t w  = kcodes[u];     // 4 aligned int8 codes, element j at byte j (LE)
            const uint32_t w2 = has2 ? kcodes2[u] : 0;
            const int pv  = (int)vcodes[u];    // SIGNED: unsigned nibble math wraps nibble<7 to ~4.3e9
            const int pv2 = has2 ? (int)vcodes2[u] : 0;
            kdq[u * 4 + 0]  = (float)((int8_t)(w >> 0)) * ksc;
            kdq[u * 4 + 1]  = (float)((int8_t)(w >> 8)) * ksc;
            kdq[u * 4 + 2]  = (float)((int8_t)(w >> 16)) * ksc;
            kdq[u * 4 + 3]  = (float)((int8_t)(w >> 24)) * ksc;
            kdq2[u * 4 + 0] = (float)((int8_t)(w2 >> 0)) * ksc2;
            kdq2[u * 4 + 1] = (float)((int8_t)(w2 >> 8)) * ksc2;
            kdq2[u * 4 + 2] = (float)((int8_t)(w2 >> 16)) * ksc2;
            kdq2[u * 4 + 3] = (float)((int8_t)(w2 >> 24)) * ksc2;
            vdq[u * 4 + 0]  = (float)((pv & 0xF) - 7) * vsc;
            vdq[u * 4 + 1]  = (float)(((pv >> 4) & 0xF) - 7) * vsc;
            vdq[u * 4 + 2]  = (float)(((pv >> 8) & 0xF) - 7) * vsc;
            vdq[u * 4 + 3]  = (float)(((pv >> 12) & 0xF) - 7) * vsc;
            vdq2[u * 4 + 0] = (float)((pv2 & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 1] = (float)(((pv2 >> 4) & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 2] = (float)(((pv2 >> 8) & 0xF) - 7) * vsc2;
            vdq2[u * 4 + 3] = (float)(((pv2 >> 12) & 0xF) - 7) * vsc2;
        }
        // dots for both rows (independent of the running softmax state)
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
        // then the two state updates, ascending — identical sequence to the unpipelined loop.
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

    // Merge per head into one smem buffer, sequentially — each head's merge is the per-head
    // kernel's exact fixed-warp-order one.
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

extern "C" __global__ void gqa_attn_splitk_k8v4_gq(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128) {
        gqa_splitk_k8v4_gq_impl<4>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                   pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    } else if (hd == 256) {
        gqa_splitk_k8v4_gq_impl<8>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                   pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    }
    // other hd: never launched — attn_dispatch falls back to the per-head kernel.
}

// k8v8 reader dequant, the contract EVERY k8v8 reader shares (the `_e` tensor-core kernel stages bf16(code*scale)
// into its tiles; the prefill dequant writes the same): ONE bf16 rounding of fp32(int8 code) * fp32(fp16 scale), then
// back to fp32 for the FMA. Skipping the rounding made a lane's bytes depend on which kernel class served it.
__device__ __forceinline__ float kv8_dq(int8_t code, float sc) {
    return __bfloat162float(__float2bfloat16_rn((float)code * sc));
}

// ===================================================================================================
// gqa_attn_splitk_k8v8 — the k8v8 twin of gqa_attn_splitk_k8v4 (the multi-lane / tree reader of the int8 K+V
// cache): SAME split structure (sk_nsplits on the column's own pc), SAME reduction order, SAME merge; per-column
// slot_ids[b] and pos[b], so lanes at different slots/positions share one launch. The `_e` tensor-core kernel
// (gqa_attn_verify_e_k8v8) stays the single-request lane (it reads ONE slot for every column); this kernel serves
// every dispatch that lane is not host-gated to — plain batched decode of several lanes, FOREST / tree verify.
// Only the V source differs from k8v4: V rows are the K layout (20 B/16: 16 B int8 codes + fp16 scale), read as one
// aligned u32 per 4 codes and multiplied by fp32(fp16 scale) — the same dequant formula as K, both upcasts exact.
extern "C" __global__ void gqa_attn_splitk_k8v8(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;   // F0: q row pitch (mtot fused view / nh*hd split)

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
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
    const int DPL = hd >> 5;

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows diverged: 20 B/16
    const int v_rb = KV8_ROW_BYTES(hd);              // V rows: the K layout, 20 B/16
    const int lane_blk = (lane * DPL) / KVQ_BLK;     // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;     // its first code within the block (4-aligned)
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 20;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd];
        const unsigned char* krow = kb + (long long)t * k_rb;
        const float ksc = __half2float(__ushort_as_half(*(const unsigned short*)(krow + KVQ_BLK)));
        const uint32_t* kcodes = (const uint32_t*)(krow + lane_off);   // 4 aligned int8 codes
        float kdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) kdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const uint32_t codes = kcodes[u];
            kdq[u * 4 + 0] = kv8_dq((int8_t)(codes >> 0), ksc);
            kdq[u * 4 + 1] = kv8_dq((int8_t)(codes >> 8), ksc);
            kdq[u * 4 + 2] = kv8_dq((int8_t)(codes >> 16), ksc);
            kdq[u * 4 + 3] = kv8_dq((int8_t)(codes >> 24), ksc);
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * kdq[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const unsigned char* vrow = vb + (long long)t * v_rb;
        const float vsc = __half2float(__ushort_as_half(*(const unsigned short*)(vrow + KVQ_BLK)));
        const uint32_t* vcodes = (const uint32_t*)(vrow + lane_off);   // 4 aligned int8 codes
        float vdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) vdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const uint32_t codes = vcodes[u];
            vdq[u * 4 + 0] = kv8_dq((int8_t)(codes >> 0), vsc);
            vdq[u * 4 + 1] = kv8_dq((int8_t)(codes >> 8), vsc);
            vdq[u * 4 + 2] = kv8_dq((int8_t)(codes >> 16), vsc);
            vdq[u * 4 + 3] = kv8_dq((int8_t)(codes >> 24), vsc);
        }
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * vdq[i];
        m = m_new;
        l = l * a_old + a_cur;
    }

    extern __shared__ float sh[];
    float* sacc = sh;
    float* sm   = sh + NW * hd;
    float* sl   = sm + NW;
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) sacc[warp * hd + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

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
}

// ===================================================================================================
// gqa_attn_splitk_k8v8_gq — the GQA-packed twin of gqa_attn_splitk_k8v8: one block per (kv head, split, column)
// reads the K/V chunk ONCE for the whole GQA group, with the same 2-row software pipeline. BIT-IDENTICAL to
// gqa_attn_splitk_k8v8 per head by the same construction argument as the k8v4 pair. V is read exactly like K
// (20 B/16, aligned u32 codes + fp16 scale); the launch contract is the k8v4 pair's (hd 128|256, ratio 2..8).
template<int DPL_T>
__device__ __forceinline__ void gqa_splitk_k8v8_gq_impl(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
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
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;   // F0: q row pitch (mtot fused view / nh*hd split)

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

    // Per-lane q slices for the whole group, in registers. g >= gqa_ratio is clamped to a valid
    // (duplicated) row and predicated to exact +0.0f — its outputs are never read.
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

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows diverged: 20 B/16
    const int v_rb = KV8_ROW_BYTES(hd);              // V rows: the K layout, 20 B/16
    const int lane_blk = (lane * DPL_T) / KVQ_BLK;
    const int lane_off = (lane * DPL_T) % KVQ_BLK;   // 4-aligned (DPL_T divides 16)
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 20;
    // E1b 2-row software pipeline: at long context this loop is LATENCY-bound (each row's K-then-V
    // dependent loads ~1.9K cycles back-to-back, only 4 warps/SM to hide them). Process rows in
    // pairs with ALL loads issued up front — one memory round-trip per pair instead of two —
    // then apply the two online-softmax updates sequentially in the same ascending order as
    // before. Every per-head FP expression keeps its exact form and order (dot, tree, scale,
    // m/a, acc), so the result is BIT-IDENTICAL to the unpipelined loop (and to gqa_attn_splitk_k8v4).
    for (int r = start + warp; r < end; r += 2 * NW) {
        const int r2 = r + NW;
        const bool has2 = r2 < end;
        const int dd = r - pos_start;
        const int t  = (!path || dd < 0) ? r  : pos_start + (int)path[b * MAX_VERIFY + dd];
        const int t2 = has2 ? ((!path || (r2 - pos_start) < 0) ? r2 : pos_start + (int)path[b * MAX_VERIFY + (r2 - pos_start)]) : 0;
        // up-front loads for BOTH rows (independent — latency paid once)
        const unsigned char* krow  = kb + (long long)t * k_rb;
        const unsigned char* krow2 = kb + (long long)t2 * k_rb;
        const unsigned char* vrow  = vb + (long long)t * v_rb;
        const unsigned char* vrow2 = vb + (long long)t2 * v_rb;
        const float ksc  = __half2float(__ushort_as_half(*(const unsigned short*)(krow  + KVQ_BLK)));
        const float ksc2 = has2 ? __half2float(__ushort_as_half(*(const unsigned short*)(krow2 + KVQ_BLK))) : 0.0f;
        const float vsc  = __half2float(__ushort_as_half(*(const unsigned short*)(vrow  + KVQ_BLK)));
        const float vsc2 = has2 ? __half2float(__ushort_as_half(*(const unsigned short*)(vrow2 + KVQ_BLK))) : 0.0f;
        const uint32_t* kcodes  = (const uint32_t*)(krow  + lane_off);
        const uint32_t* kcodes2 = (const uint32_t*)(krow2 + lane_off);
        const uint32_t* vcodes  = (const uint32_t*)(vrow  + lane_off);
        const uint32_t* vcodes2 = (const uint32_t*)(vrow2 + lane_off);
        float kdq[DPL_T], kdq2[DPL_T], vdq[DPL_T], vdq2[DPL_T];
        #pragma unroll
        for (int u = 0; u < DPL_T / 4; u++) {
            const uint32_t w  = kcodes[u];     // 4 aligned int8 codes, element j at byte j (LE)
            const uint32_t w2 = has2 ? kcodes2[u] : 0;
            const uint32_t pv  = vcodes[u];    // 4 aligned int8 codes, element j at byte j (LE)
            const uint32_t pv2 = has2 ? vcodes2[u] : 0;
            kdq[u * 4 + 0]  = kv8_dq((int8_t)(w >> 0), ksc);
            kdq[u * 4 + 1]  = kv8_dq((int8_t)(w >> 8), ksc);
            kdq[u * 4 + 2]  = kv8_dq((int8_t)(w >> 16), ksc);
            kdq[u * 4 + 3]  = kv8_dq((int8_t)(w >> 24), ksc);
            kdq2[u * 4 + 0] = kv8_dq((int8_t)(w2 >> 0), ksc2);
            kdq2[u * 4 + 1] = kv8_dq((int8_t)(w2 >> 8), ksc2);
            kdq2[u * 4 + 2] = kv8_dq((int8_t)(w2 >> 16), ksc2);
            kdq2[u * 4 + 3] = kv8_dq((int8_t)(w2 >> 24), ksc2);
            vdq[u * 4 + 0]  = kv8_dq((int8_t)(pv >> 0), vsc);
            vdq[u * 4 + 1]  = kv8_dq((int8_t)(pv >> 8), vsc);
            vdq[u * 4 + 2]  = kv8_dq((int8_t)(pv >> 16), vsc);
            vdq[u * 4 + 3]  = kv8_dq((int8_t)(pv >> 24), vsc);
            vdq2[u * 4 + 0] = kv8_dq((int8_t)(pv2 >> 0), vsc2);
            vdq2[u * 4 + 1] = kv8_dq((int8_t)(pv2 >> 8), vsc2);
            vdq2[u * 4 + 2] = kv8_dq((int8_t)(pv2 >> 16), vsc2);
            vdq2[u * 4 + 3] = kv8_dq((int8_t)(pv2 >> 24), vsc2);
        }
        // dots for both rows (independent of the running softmax state)
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
        // then the two state updates, ascending — identical sequence to the unpipelined loop.
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

    // Merge per head into one smem buffer, sequentially — each head's merge is the per-head
    // kernel's exact fixed-warp-order one.
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

extern "C" __global__ void gqa_attn_splitk_k8v8_gq(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128) {
        gqa_splitk_k8v8_gq_impl<4>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                   pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    } else if (hd == 256) {
        gqa_splitk_k8v8_gq_impl<8>(out_m, out_l, out_acc, q, k_cache, v_cache,
                                   pos, bs_packed, nh_packed, slot_ids, path, col_pos_start);
    }
    // other hd: never launched — attn_dispatch falls back to the per-head kernel.
}

// ===================================================================================================
// gqa_attn_splitk_tq — the TurboQuant twin of gqa_attn_splitk_q4: SAME split structure, SAME
// reduction order, SAME merge. The query was rotated ONCE per (token, head) by rotate_q_tq into an
// interleaved f32 buffer (qr = Pi·q at 2j, qs = S·q at 2j+1) — the reference's "rotate once, never
// rotate back" principle. Per row the kernel dequantizes the packed K in-register (2-bit codes +
// QJL sign bits + fp16 norms) and computes the DUAL-DOT score
//   s = kn * ( <Pi q, cb2[idx]> + sqrt(pi/2)/d * rn * <S q, sign> )
// — no 1/sqrt(hd): this estimator IS <q,k> (REPORT §4.1, validated against scores_tq.bin). PV
// accumulates in the ROTATED domain (vn·cb3[idx]) and the merge applies the Pi^T epilogue ONCE
// per (token, head): out_acc = Pi^T·(sum_t a_t·vn_t·cb3[idx_t]) with out_l = sum a_t and
// out_m = max — gqa_attn_reduce divides out_acc by out_l as usual (Pi^T is linear, so the
// normalized output is softmax-weighted Pi^T·v exactly).
//
// Scope: hd == 128 only (the TQ layout is d=128). TQ is Hy3-only today and Hy3 is a CHAIN
// (col_pos_start == None at every call site), so the FOREST col_pos_start arg of the q4 twin is
// dropped: pos_start = pos[0], exactly what the q4 kernels compute when col_pos_start is NULL.
extern "C" __global__ void gqa_attn_splitk_tq(
    float* out_m, float* out_l, float* out_acc,
    const float* qrqs, const unsigned char* k_cache, const unsigned char* v_cache,
    const float* tables, const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path) {
    const int nh  = nh_packed >> 20;
    const int nkv = nh_packed & 0x3FF;
    const int hd  = TQ_HD;                              // fixed by the layout contract
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos[b] + 1;
    const int pos_start = pos[0];                       // TQ: chain only (no FOREST lanes)
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);
    if (split >= ns) return;
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;
    const int DPL = hd >> 5;

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    const float* qr = qrqs + ((long long)b * nh + qh) * 2 * hd + 2 * lane * DPL;  // interleaved: coord j at floats 2j/2j+1
    float qrv[SK_DPL_MAX], qsv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) {
        qrv[i] = (i < DPL) ? qr[2 * i] : 0.0f;
        qsv[i] = (i < DPL) ? qr[2 * i + 1] : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    const float qjl_scale = tables[TQ_TAB_SCALE];
    const int rb = TQ_ROW_BYTES;
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride * rb;
    const unsigned char* kb = k_cache + kvbase;
    const unsigned char* vb = v_cache + kvbase;
    // lane's coord window: [4L, 4L+4). K codes (2-bit: byte L, 4 x 2-bit LSB-first; 3-bit:
    // LSB-first bitstream, bits [12L+3u, 12L+3u+3) — the u16 at bytes (12L)>>3..+1, shift
    // RELATIVE to the u16: 12L+3u - 8*kc_byte); QJL signs: byte TQ_SIGN_OFF+(L>>1), bits
    // 4*(L&1)+u; V codes: 3-bit LSB-first, bits [12L+3u, 12L+3u+3) — the u16 at bytes
    // (12L)>>3..+1 (byte-unaligned safe), shift RELATIVE to the u16: 12L+3u - 8*vc_byte.
#if TQ_K_BITS == 3
    const int kc_byte = ((lane * DPL) * 3) >> 3;
    const int kc_shift0 = (lane * DPL) * 3 - 8 * kc_byte;   // {0,4}: the lane's first K coord in the u16
#else
    const int kc_byte = (lane * DPL) >> 2;
#endif
    const int sc_byte = TQ_SIGN_OFF + ((lane * DPL) >> 3);
    const int sc_shift = (lane * DPL) & 7;
    const int vc_byte = ((lane * DPL) * 3) >> 3;
    const int vc_shift0 = (lane * DPL) * 3 - 8 * vc_byte;   // {0,4}: the lane's first coord in the u16
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd];
        const unsigned char* krow = kb + (long long)t * rb;
        const unsigned char* vrow = vb + (long long)t * rb;
        const float rn = tq_half_at(krow, TQ_RN_OFF), kn = tq_half_at(krow, TQ_KN_OFF);
        const float vn = tq_half_at(vrow, 48);
#if TQ_K_BITS == 3
        const unsigned short kw = (unsigned short)krow[kc_byte] | ((unsigned short)krow[kc_byte + 1] << 8);
#else
        const unsigned char kcb = krow[kc_byte];
#endif
        const unsigned char scb = krow[sc_byte];
        const unsigned short vw = (unsigned short)vrow[vc_byte] | ((unsigned short)vrow[vc_byte + 1] << 8);
        float kdq[SK_DPL_MAX], vdq[SK_DPL_MAX], sg[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) { kdq[i] = 0.0f; vdq[i] = 0.0f; sg[i] = 0.0f; }
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX; u++) {
            if (u >= DPL) break;
#if TQ_K_BITS == 3
            kdq[u] = cb3[(kw >> (kc_shift0 + 3 * u)) & 7];
#else
            kdq[u] = cb2[(kcb >> (2 * u)) & 3];
#endif
            sg[u]  = ((scb >> (sc_shift + u)) & 1) ? 1.0f : -1.0f;
            vdq[u] = cb3[(vw >> (vc_shift0 + 3 * u)) & 7];
        }
        float s_code = 0.0f, s_qjl = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) { s_code += qrv[i] * kdq[i]; s_qjl += qsv[i] * sg[i]; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s_code += __shfl_xor_sync(0xffffffffu, s_code, off);
            s_qjl  += __shfl_xor_sync(0xffffffffu, s_qjl, off);
        }
        const float s = kn * (s_code + qjl_scale * rn * s_qjl);

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * (vn * vdq[i]);
        m = m_new;
        l = l * a_old + a_cur;
    }

    // Merge (the per-head kernel's exact fixed-warp-order merge) + the Pi^T epilogue: the
    // 128-dim rotated-domain partial (threads 0..127, staged in smem) unrotates once per head.
    extern __shared__ float sh[];
    float* sacc = sh;                       // NW * hd
    float* sm   = sh + NW * hd;             // NW
    float* sl   = sm + NW;                  // NW
    float* tacc = sl + NW;                  // hd
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) sacc[warp * hd + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

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
        tacc[d] = num;                       // rotated-domain partial (unnormalized)
        if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        const int d = threadIdx.x;
        const float* pi = tables + TQ_TAB_PI + d;     // Pi[i][d] = tables[i*128 + d]
        float o = 0.0f;
        #pragma unroll
        for (int i = 0; i < TQ_HD; i++) o += pi[i * TQ_HD] * tacc[i];
        out_acc[idx * hd + d] = o;           // Pi^T · partial; the reduce normalizes by out_l
    }
}

