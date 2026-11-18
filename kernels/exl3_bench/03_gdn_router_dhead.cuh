// exl3_bench/03_gdn_router_dhead.cuh: GDN gated-delta-rule step, router top-k, MoE gate/combine, DHEAD draft lm_head diet.
// GDN: engine reduction orders kept. DHEAD is DRAFT-ONLY (verification is untouched).
// Section 3 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- GDN gated-delta-rule step (port of delta_step_b + gdn_token):
// qkv_conv [B][qkv_stride]: q at 0, k at key_dim, v at 2*key_dim + head*vd.
// state [B][nh][kd][vd] f32. beta = sigmoid(b); gt = exp(-exp(A_log)*softplus(a+dt_bias)).
// grid (B*nh*vd/32), block kd. Engine reduction orders kept.
#define XQ_GDN_C 32
#define XQ_GDN_SP (XQ_GDN_C + 1)
// 2026-09-26: coalesced state copy (8 lanes x float4 per 128-B row segment) replacing the
// thread-per-row scalar loop (a warp touched 32 rows = 32 segments per element step). The
// VALUES placed in S_sh / S are identical — only the access pattern changes (bitwise).
__device__ __forceinline__ void xq_gdn_ld_state(float* S_sh, const float* S, int kd, int vd, int bb0) {
    const int n4 = kd * (XQ_GDN_C / 4);
    for (int e = threadIdx.x; e < n4; e += blockDim.x) {
        const int r = e / (XQ_GDN_C / 4), c4 = (e % (XQ_GDN_C / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(S + (long long)r * vd + bb0 + c4);
        float* d = S_sh + r * XQ_GDN_SP + c4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
}
__device__ __forceinline__ void xq_gdn_st_state(float* S, const float* S_sh, int kd, int vd, int bb0) {
    const int n4 = kd * (XQ_GDN_C / 4);
    for (int e = threadIdx.x; e < n4; e += blockDim.x) {
        const int r = e / (XQ_GDN_C / 4), c4 = (e % (XQ_GDN_C / 4)) * 4;
        const float* d = S_sh + r * XQ_GDN_SP + c4;
        *reinterpret_cast<float4*>(S + (long long)r * vd + bb0 + c4) = make_float4(d[0], d[1], d[2], d[3]);
    }
}

__device__ __forceinline__ void xq_gdn_token(float* S_sh, int kd, int bb0,
                                             const __half* q_in, const __half* k_in, const __half* v_in,
                                             __half* coreb, int key_head, float beta, float gt,
                                             float* Srow, float* kv_mem, float* vbuf, float* delta,
                                             float* qrow, float* krow) {
    const int a = threadIdx.x;
    float qv = xq_h2f(q_in[key_head * kd + a]);
    float kv = xq_h2f(k_in[key_head * kd + a]);
    Srow[a] = qv * qv; __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float qn = rsqrtf(Srow[0] + 1e-6f); __syncthreads();
    qv *= qn;
    Srow[a] = kv * kv; __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) { if (a < s2) Srow[a] += Srow[a + s2]; __syncthreads(); }
    float kn = rsqrtf(Srow[0] + 1e-6f); __syncthreads();
    kv *= kn;
    float scale = 1.0f / sqrtf((float)kd);
    qv *= scale; qrow[a] = qv; krow[a] = kv; __syncthreads();
    #pragma unroll
    for (int c = 0; c < XQ_GDN_C; c++) S_sh[a * XQ_GDN_SP + c] *= gt;
    __syncthreads();
    if (a < XQ_GDN_C) {
        float km = 0.0f;
        for (int aa = 0; aa < kd; aa++) km += S_sh[aa * XQ_GDN_SP + a] * krow[aa];
        kv_mem[a] = km;
        vbuf[a] = xq_h2f(v_in[bb0 + a]);
    }
    __syncthreads();
    if (a < XQ_GDN_C) delta[a] = (vbuf[a] - kv_mem[a]) * beta;
    __syncthreads();
    const float kk = krow[a];
    #pragma unroll
    for (int c = 0; c < XQ_GDN_C; c++) S_sh[a * XQ_GDN_SP + c] += kk * delta[c];
    __syncthreads();
    if (a < XQ_GDN_C) {
        float o = 0.0f;
        for (int aa = 0; aa < kd; aa++) o += S_sh[aa * XQ_GDN_SP + a] * qrow[aa];
        coreb[bb0 + a] = xq_f2h(o);
    }
    __syncthreads();
}

extern "C" __global__ void xq_gdn_step(__half* __restrict__ core, const __half* __restrict__ qkv,
                                       float* __restrict__ state, const __half* __restrict__ b_in,
                                       const __half* __restrict__ a_in, int nh, int n_k_heads,
                                       int kd, int vd, const float* __restrict__ a_log,
                                       const float* __restrict__ dt_bias,
                                       const int* __restrict__ slot_ids) {
    XQ_PDL_ENTRY();
    int blk = blockIdx.x;
    int nchunk = vd / XQ_GDN_C;
    const int qkv_stride = 2 * n_k_heads * kd + nh * vd;  // conv_dim (arity limit: derive in-kernel)
    int chunk = blk % nchunk; blk /= nchunk;
    int head = blk % nh; blk /= nh;
    int b = blk;
    // S-A3-e FIX: the recurrent state is PER-SLOT (a request keeps its slot for its lifetime);
    // the batch row b is the launch position. Offline b==slot always; serving recycles lanes.
    const int slot = slot_ids[b * 2];
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
    const __half* col = qkv + (long long)b * qkv_stride;
    float* S = state + ((long long)slot * nh + head) * kd * vd;
    float beta = 1.0f / (1.0f + __expf(-xq_h2f(b_in[(long long)b * nh + head])));
    float sp = xq_h2f(a_in[(long long)b * nh + head]) + dt_bias[head];
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    float gt = __expf(-__expf(a_log[head]) * sp);
    xq_gdn_ld_state(S_sh, S, kd, vd, bb0);
    __syncthreads();
    xq_gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                 core + (long long)b * (nh * vd) + (long long)head * vd,
                 key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
    xq_gdn_st_state(S, S_sh, kd, vd, bb0);
}

// ---- GDN gated output norm (port of rmsnorm_gated_sig_b): per (b, head):
// out = x * rsqrt(mean(x^2)+eps) * w[i] * sigmoid(z). grid B*nh, block vd.
extern "C" __global__ void xq_gdn_gate(__half* __restrict__ out, const __half* __restrict__ x,
                                       const __half* __restrict__ z, const __half* __restrict__ w,
                                       int vd, int nh, int B, float eps) {
    XQ_PDL_ENTRY();
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? xq_h2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) {
        float g = xq_h2f(z[(long long)b * (nh * vd) + (long long)head * vd + tid]);
        out[base + tid] = xq_f2h(v * inv * xq_h2f(w[tid]) * xq_sig(g));
    }
}

// EXL27B: xq_gdn_gate for the Qwen3.5-class packs (output_gate_type swish / absent): the gated RMSNorm
// with silu(z) instead of sigmoid(z) — a copy; xq_gdn_gate (FN, sigmoid) is untouched.
// out = x * rsqrt(mean(x^2)+eps) * w[i] * silu(z). grid B*nh, block vd.
extern "C" __global__ void xq_gdn_gate_silu(__half* __restrict__ out, const __half* __restrict__ x,
                                            const __half* __restrict__ z, const __half* __restrict__ w,
                                            int vd, int nh, int B, float eps) {
    XQ_PDL_ENTRY();
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? xq_h2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) {
        float g = xq_h2f(z[(long long)b * (nh * vd) + (long long)head * vd + tid]);
        out[base + tid] = xq_f2h(v * inv * xq_h2f(w[tid]) * (g * xq_sig(g)));
    }
}

// ---- MoE router: logits [B][ne] fp32 -> softmax -> top-k (ties -> lowest id) -> renorm.
// smem floats: [0,bs) reduce scratch, [bs, bs+ne) probs, [bs+ne, bs+ne+2*bs) candidate scratch.
extern "C" __global__ void xq_router_topk(int* __restrict__ ids, float* __restrict__ wts,
                                          const float* __restrict__ logits, int ne, int k, int B) {
    XQ_PDL_ENTRY();
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float* prob = sm + bs;
    int* cand_i = (int*)(sm + bs + ne);
    const float* lg = logits + (long long)b * ne;
    float mx = -INFINITY;
    for (int i = tid; i < ne; i += bs) mx = fmaxf(mx, lg[i]);
    sm[tid] = mx; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] = fmaxf(sm[tid], sm[tid + s2]); __syncthreads(); }
    float gmax = sm[0];
    __syncthreads();  // S-A3-e FIX (expert-confirmed): all threads must read gmax before sm is reused.
    float part = 0.0f;
    for (int i = tid; i < ne; i += bs) part += __expf(lg[i] - gmax);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float denom = sm[0];
    for (int i = tid; i < ne; i += bs) prob[i] = __expf(lg[i] - gmax) / denom;
    __syncthreads();
    float* cand_v = sm;  // reuse the reduce scratch for candidate values
    for (int j = 0; j < k; j++) {
        float bv = -2.0f; int bi = ne;
        for (int i = tid; i < ne; i += bs)
            if (prob[i] > bv || (prob[i] == bv && i < bi)) { bv = prob[i]; bi = i; }
        cand_v[tid] = bv; cand_i[tid] = bi;
        __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) {
                float v0 = cand_v[tid], v1 = cand_v[tid + s2];
                int i0 = cand_i[tid], i1 = cand_i[tid + s2];
                if (v1 > v0 || (v1 == v0 && i1 < i0)) { cand_v[tid] = v1; cand_i[tid] = i1; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            // NaN/Inf logits leave every prob comparison false and bi at its
            // `ne` sentinel; clamp so host counting stays in-bounds (S-A3-f-b).
            if (cand_i[0] < 0 || cand_i[0] >= ne) cand_i[0] = 0;
            ids[(long long)b * k + j] = cand_i[0];
            wts[(long long)b * k + j] = cand_v[0];
            prob[cand_i[0]] = -2.0f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) sum += wts[(long long)b * k + j];
        for (int j = 0; j < k; j++) wts[(long long)b * k + j] /= sum;
    }
}

// ---- MoE gate*up: din[em,d] = silu(gate)*up; y_gu row em = [gate(640) | up(640)] interleaved.
// S-A3-f Item 2: grid padded to the topk cap; row640 = m*mi rows live.
extern "C" __global__ void xq_moe_gate_mul(__half* __restrict__ din, const __half* __restrict__ y_gu,
                                           long long row640, const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)(*esel) * row640) return;
    long long em = i / 640;          // which (slot, m) row
    long long d = i - em * 640;      // column within the 640-wide half
    float g = xq_h2f(y_gu[em * 1280 + d]);
    float u = xq_h2f(y_gu[em * 1280 + 640 + d]);
    din[i] = xq_f2h(xq_silu(g) * u);
}

// ---- MoE combine: out[m,i] = sig(sg.x)*y_sh[m,i] + sum_j wts[m,j]*y_d[slot(m,j)][m,i].
// slotmap [ne] i32: expert id -> slot in the grouped buffers this step, -1 = absent.
extern "C" __global__ void xq_moe_combine(__half* __restrict__ out, const __half* __restrict__ y_d,
                                          const __half* __restrict__ y_sh,
                                          const int* __restrict__ ids, const float* __restrict__ wts,
                                          const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                          const __half* __restrict__ x, int k, int h, int M) {
    XQ_PDL_ENTRY();
    int m = blockIdx.x;
    if (m >= M) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float part = 0.0f;
    for (int c = tid; c < h; c += bs) part += xq_h2f(sg[c]) * xq_h2f(x[(long long)m * h + c]);
    sm[tid] = part; __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float sgv = xq_sig(sm[0]);
    for (int i = tid; i < h; i += bs) {
        float acc = sgv * xq_h2f(y_sh[(long long)m * h + i]);
        for (int j = 0; j < k; j++) {
            int e = ids[(long long)m * k + j];
            int slot = slotmap[e];
            if (slot >= 0) acc += wts[(long long)m * k + j] * xq_h2f(y_d[((long long)slot * M + m) * h + i]);
        }
        out[(long long)m * h + i] = xq_f2h(acc);
    }
}

// ---- greedy argmax over vocab (fp32 compare, lowest-id ties). One block.
extern "C" __global__ void xq_argmax(int* __restrict__ out_ids, int lane,
                                     const __half* __restrict__ logits, long long start, int V) {
    XQ_PDL_ENTRY();
    extern __shared__ float sm[];
    long long* si = (long long*)(sm + blockDim.x);
    int tid = threadIdx.x, bs = blockDim.x;
    float best = -INFINITY; long long bi = 0;
    for (int i = tid; i < V; i += bs) {
        float v = xq_h2f(logits[start + i]);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sm[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            float v0 = sm[tid], v1 = sm[tid + s2];
            long long i0 = si[tid], i1 = si[tid + s2];
            if (v1 > v0 || (v1 == v0 && i1 < i0)) { sm[tid] = v1; si[tid] = i1; }
        }
        __syncthreads();
    }
    if (tid == 0) out_ids[lane] = (int)si[0];
}

// ---- WP09: ONE-launch multi-block greedy argmax over m rows (replaces m single-block 1024-thread
// xq_argmax launches, ~21 us each). grid (NB <= 32, m), block = multiple of 32.
// EXACTNESS: every thread folds its elements with xq_argmax's comparator verbatim
// (v > best || (v == best && i < bi)) from xq_argmax's init (-inf, 0). NaN never compares true,
// so the fold is the lexicographic max over {(-inf, 0)} U {(x_i, i) : x_i not NaN} under
// (value desc by fp32 compare — +0 == -0 — then id asc). That order is total on the candidate
// set, so the winner does not depend on the partition or the reduction tree: xq_argmax's
// one-block tree and this kernel's (per-thread fold -> warp -> block -> cross-block) return the
// SAME id, including the ties, the +-0, all--inf (-> 0) and all-NaN (-> 0) rows (probe:
// EXL3-ARGMAX-ROWS). Cross-block: (best, bi) packs into an order-preserving u64 key (canonical
// +0, sign-flip map on the float bits; low word = ~id so a larger key = a lower id), each block
// publishes its key, __threadfence + per-row counter; the LAST block of row r reduces the NB keys
// (L2 reads) and writes out_ids[lane0 + r], then re-arms cnt[r] = 0 (one real zero at allocation,
// graph replays reuse the buffers — the single-stream contract, as hc_bar / done_cnt).
__device__ __forceinline__ void xq_am_take(float& best, int& bi, float v, int i) {
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
}
__device__ __forceinline__ unsigned long long xq_am_key(float v, int i) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) == 0u) u = 0u;                  // -0 == +0 under the fp32 compare
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);       // order-preserving on non-NaN floats
    return ((unsigned long long)u << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned)i);
}
__device__ __forceinline__ unsigned long long xq_am_wmax(unsigned long long k) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long ok = __shfl_xor_sync(0xffffffffu, k, o);
        k = ok > k ? ok : k;
    }
    return k;
}
extern "C" __global__ void xq_argmax_rows(int* __restrict__ out_ids, int lane0,
                                          const __half* __restrict__ logits, long long start0,
                                          long long rstride, int V,
                                          unsigned long long* __restrict__ part,
                                          unsigned int* __restrict__ cnt) {
    XQ_PDL_ENTRY();
    __shared__ unsigned long long wk[32];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const __half* row = logits + start0 + (long long)r * rstride;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;     // 8-aligned slices -> row+lo keeps row's alignment
    const int lo = min(b * chunk, V), hi = min(lo + chunk, V);
    float best = -INFINITY; int bi = 0;
    if ((reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull) {
        const int nv = (hi - lo) >> 3;
        const uint4* rv = reinterpret_cast<const uint4*>(row + lo);
        #pragma unroll 4
        for (int q = tid; q < nv; q += bs) {
            const uint4 u = __ldg(rv + q);
            const __half* hv = reinterpret_cast<const __half*>(&u);
            #pragma unroll
            for (int j = 0; j < 8; j++) xq_am_take(best, bi, xq_h2f(hv[j]), lo + q * 8 + j);
        }
        for (int i = lo + nv * 8 + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    } else {
        for (int i = lo + tid; i < hi; i += bs) xq_am_take(best, bi, xq_h2f(row[i]), i);
    }
    unsigned long long key = xq_am_wmax(xq_am_key(best, bi));
    if ((tid & 31) == 0) wk[tid >> 5] = key;
    __syncthreads();
    if (tid < 32) {
        key = xq_am_wmax(tid < (bs >> 5) ? wk[tid] : 0ull);   // 0 < every real key (ord(-inf) > 0)
        if (tid == 0) {
            part[(long long)r * nb + b] = key;
            __threadfence();
            last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
        }
    }
    __syncthreads();
    if (!last) return;
    if (tid < 32) {
        __threadfence();
        const unsigned long long k2 = xq_am_wmax(tid < nb ? __ldcg(part + (long long)r * nb + tid) : 0ull);
        if (tid == 0) {
            out_ids[lane0 + r] = (int)(0xFFFFFFFFu - (unsigned)(k2 & 0xFFFFFFFFull));
            cnt[r] = 0u;                                      // re-arm for the next launch / replay
        }
    }
}

// ---- API-parity DDS: max logit of one fp16 row -> out[ooff] (f32). The draft head's argmax
// LOGIT is the reference implementation's dynamic-draft confidence score (exllamav3 DraftConfidenceCalibrator).
extern "C" __global__ void xq_rowmax_f16(float* __restrict__ out, long long ooff,
                                         const __half* __restrict__ x, long long start, int V) {
    XQ_PDL_ENTRY();
    __shared__ float sm[32];
    float best = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) best = fmaxf(best, __half2float(x[start + i]));
    for (int o = 16; o; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : -INFINITY;
        for (int o = 16; o; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (threadIdx.x == 0) out[ooff] = w;
    }
}

// =============================================================================
// DHEAD (w2, 2026-09-26): draft lm_head diet — DRAFT-ONLY (verification is untouched).
// The draft pass's pruned lm_head (EXL3 5-bit column slice, 65,536 x 2560 = 104.9 MB/pass) ran as
// had_suh + exl3_hmma_gemm (0.52 ms, ~85% of its byte floor) + had_svh + argmax_rows + rowmax
// + 2 copies. Replacement, three launches (target: baseline sm_121 — dp4a, mma.sync.m16n8k16 f16,
// ld.global.cs; nothing family/arch-specific is needed for a byte-bound GEMV):
//  (1) xq_dh_prep    one CTA: xh = had128(x * suh) with exl3_had_suh_dev's ops verbatim (the f16
//                    xh is BITWISE the old chain's), then a per-vector int8 quantization of xh in
//                    the screen's byte order + {sx, sum q, screen noise sigma}.
//  (2) xq_dh_screen  B-bit (1|2|4) SCREEN of the SAME rotated weights. Codes are built at load from
//                    the decoded trellis with one GLOBAL uniform quantizer (the mul1 codebook is
//                    unit-variance by construction, so group scales buy nothing). One CTA per
//                    128-col Hadamard block (512 CTAs, all co-resident at 12/SM), dp4a over
//                    streamed (ld.global.cs, single-use) 16-B code chunks, in-CTA had128 x svh
//                    epilogue -> approximate block max bmax[b]. 2-bit = 41.9 MB/pass.
//  (3) xq_dh_rescore every CTA selects the SAME candidate set (blocks whose approximate max is
//                    within delta * sigma * (bsig_b + bsig_max) of the best, at most tmax, ties and
//                    truncation by (value desc, index asc)); CTA (r, s) recomputes candidate r's
//                    128 logits EXACTLY from the EXL3 trellis over K-split s (exl3_gemm_body's
//                    decode + m16n8k16 mma per k16 step, staged through smem); the LAST CTA sums
//                    the S fp32 partials in ascending s -> f16 yraw -> had_svh's ops verbatim ->
//                    (max, lowest id) over the candidates -> argmax, dconf[i], d_dev[i], toks[0]
//                    (replacing argmax_rows + xq_rowmax_f16 + 2 x xq_copy1_i32).
// Exactness: when the old argmax's block is a candidate, the draft id and dconf equal the old
// head's up to the split-K fp32 re-association before the f16 yraw rounding (a 1-ulp tie class);
// otherwise the draft is the best in-candidate token. Greedy output never depends on the draft.
// Workspace `ws` (u32 words; mirrored by src/exl3_forward/dhead.rs ws_layout):
//   [0, K/4)            screen x (int8, permuted);  [K/4 + {0,1,2}] = sx, sum q, sigma
//   [off_bmax, +nblk)   approximate block maxima (f32)
//   [off_cnt]           rescore last-CTA counter (self re-arming; one real zero at build)
//   [off_dbg, +TCAP+4)  {ns, cand[0..ns), .., nq} (XCHECK readback)
//   [off_part, ...)     fp32 partials [TCAP][S][128]
// =============================================================================
#define DH_KPT 8     // approximate block maxima per rescore thread: nblk <= 256 * DH_KPT
#define DH_TCAP 32   // candidate cap (tmax <= DH_TCAP)
#define DH_RMAX 16   // k16 rows per rescore K-split (host: (K/16) / S <= DH_RMAX)

__device__ __forceinline__ int dh_off_bmax(int K) { return ((K >> 2) + 4 + 3) & ~3; }
__device__ __forceinline__ int dh_off_cnt(int K, int nblk) { return dh_off_bmax(K) + ((nblk + 3) & ~3); }
__device__ __forceinline__ int dh_off_dbg(int K, int nblk) { return dh_off_cnt(K, nblk) + 4; }
__device__ __forceinline__ int dh_off_part(int K, int nblk) { return dh_off_dbg(K, nblk) + DH_TCAP + 4; }

// order-preserving u32 of a non-NaN float (xq_am_key's high word) and its inverse
__device__ __forceinline__ unsigned dh_ord(float v) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) == 0u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float dh_unord(unsigned o) {
    return (o & 0x80000000u) ? __uint_as_float(o & 0x7FFFFFFFu) : __uint_as_float(~o);
}

// Load-time build: one thread per u32 code word. Word wi <-> (block nb, chunk kc, column cl,
// word u): wi = ((nb * KC + kc) * 128 + cl) * 4 + u, so a warp's 16-B loads in the screen are
// 32 consecutive columns (512 B) and a CTA's whole block is one contiguous stream. Word u of
// chunk kc covers k-group g = 4 kc + u (G = 32/B consecutive k); code(k = g G + b cpb + j) sits
// at bits [8 b + B j, +B), cpb = 8/B, so (word >> B j) & mask puts codes j of the 4 bytes in the
// byte lanes that xq_dh_prep's x word (g, j) holds. Value of code c = dq * (c - (2^B - 1) / 2).
extern "C" __global__ void __launch_bounds__(256) xq_dh_build(
        const uint16_t* __restrict__ tr, unsigned* __restrict__ codes,
        int K, int N, int bits, int B, float inv_dq, long long nwords) {
    const long long wi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= nwords) return;
    const int G = 32 / B, cpb = 8 / B, KC = K / (128 / B), L = 1 << B;
    const int u = (int)(wi & 3);
    const int cl = (int)((wi >> 2) & 127);
    const long long rest = wi >> 9;
    const int kc = (int)(rest % KC);
    const int nb = (int)(rest / KC);
    const int c = nb * 128 + cl, g = kc * 4 + u;
    const int nb16 = N >> 4, n16 = c >> 4, cc = c & 15;
    unsigned word = 0u;
    for (int kl = 0; kl < G; ++kl) {
        const int k = g * G + kl, kb = k >> 4, r = k & 15;
        const uint32_t* ring = reinterpret_cast<const uint32_t*>(tr + ((size_t)kb * nb16 + n16) * (8 * bits));
        const int l = 8 * ((cc >> 1) & 3) + ((r & 7) >> 1);                          // tmap(r, cc)
        const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (cc >> 3) + 32 * (cc & 1);
        const float w = __half2float(__ushort_as_half(exl3_dq_k2(ring, t, bits)));
        int q = (int)floorf(w * inv_dq + 0.5f * (float)L);
        q = max(0, min(L - 1, q));
        const int b = kl / cpb, j = kl - b * cpb;
        word |= (unsigned)q << (8 * b + B * j);
    }
    codes[wi] = word;
}

// (1) prep: blockDim = 32 * (K/128) (one warp per 128-block of K, K <= 4096).
extern "C" __global__ void __launch_bounds__(1024) xq_dh_prep(
        const __half* __restrict__ x, const __half* __restrict__ suh, __half* __restrict__ xh,
        unsigned* __restrict__ ws, int K, int B, float mse) {
    XQ_PDL_ENTRY();
    __shared__ float s_am[32];
    __shared__ float s_ss[32];
    __shared__ int s_qs[32];
    __shared__ __align__(16) unsigned char s_q[4096];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nw = blockDim.x >> 5;
    const int base = warp * 128 + lane * 4;          // exl3_had_suh_dev at m = 0, kb = warp
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __half2float(x[base + r]) * __half2float(suh[base + r]);
    exl3_had128(v, lane);
    float xf[4], am = 0.f, ss = 0.f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const __half hv = __float2half_rn(v[r]);
        xh[base + r] = hv;                           // == exl3_had_suh's xh, bit for bit
        xf[r] = __half2float(hv);
        am = fmaxf(am, fabsf(xf[r]));
        ss = fmaf(xf[r], xf[r], ss);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) { s_am[warp] = am; s_ss[warp] = ss; }
    __syncthreads();
    float amax = 0.f, sst = 0.f;
    for (int w = 0; w < nw; ++w) { amax = fmaxf(amax, s_am[w]); sst += s_ss[w]; }
    const float sx = amax > 0.f ? amax * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sx;
    const int cpb = 8 / B, G = 32 / B;
    int qs = 0;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int q = max(-127, min(127, __float2int_rn(xf[r] * inv)));
        qs += q;
        const int k = base + r, g = k / G, kl = k - g * G, b = kl / cpb, j = kl - b * cpb;
        s_q[(g * cpb + j) * 4 + b] = (unsigned char)(q & 0xFF);
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o);
    if (lane == 0) s_qs[warp] = qs;
    __syncthreads();
    const int KW = K >> 2;
    for (int i = tid; i < KW; i += blockDim.x) ws[i] = reinterpret_cast<const unsigned*>(s_q)[i];
    if (tid == 0) {
        int S = 0;
        for (int w = 0; w < nw; ++w) S += s_qs[w];
        ws[KW] = __float_as_uint(sx);
        ws[KW + 1] = (unsigned)S;
        // yraw noise sigma: weight-quantization MSE * |xh|^2 + x-quantization (sx^2/12 per term, K
        // unit-variance weights). The logit noise of column j is |svh_j| * sigma (had128 is orthonormal).
        ws[KW + 2] = __float_as_uint(sqrtf(mse * sst + (float)K * sx * sx * (1.0f / 12.0f)));
        ws[KW + 3] = 0u;
    }
}

// dp4a over one 16-B code chunk (4 words x 32/B codes) against its 32/B x words in smem.
template <int B>
__device__ __forceinline__ int dh_dot4(const uint4 w, const unsigned* xs, int acc) {
    constexpr int CPB = 8 / B;
    constexpr unsigned MASK = ((1u << B) - 1u) * 0x01010101u;
    const unsigned wv[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int j = 0; j < CPB; ++j)
            acc = __dp4a((int)((wv[u] >> (B * j)) & MASK), (int)xs[u * CPB + j], acc);
    }
    return acc;
}

template <int B>
__device__ __forceinline__ void dh_screen_body(const uint4* __restrict__ codes, unsigned* __restrict__ ws,
                                               const __half* __restrict__ svh, int K, float dq,
                                               unsigned* xs, float* s_y,
                                               float* __restrict__ bdst, int nb_off) {
    constexpr int XPU = 32 / B;                      // x words per 16-B code chunk (= 4 * cpb)
    // nb = the GLOBAL 128-column block: nb_off = 0 for the replicated screen, rank * nblk / world
    // for TP-H #3's sharded one. bdst[nb] = the approximate block max (ws bmax for the replicated
    // screen, the zero-padded exchange buffer for the sharded one).
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nb = nb_off + blockIdx.x;
    const int KW = K >> 2, KC = K / (128 / B);
    for (int i = tid; i < KW; i += 128) xs[i] = ws[i];
    __syncthreads();
    // 4 independent 16-B streaming loads in flight per thread (x 12 CTAs x 128 threads = 96 KB
    // per SM, far above the ~5 KB/SM the DRAM queue needs); one accumulator — the other 47
    // resident warps hide the dp4a chain. Nothing but the pointer/acc lives across the loop.
    const uint4* __restrict__ p = codes + (size_t)nb * KC * 128 + tid;
    int acc = 0;
    int kc = 0;
    for (; kc + 4 <= KC; kc += 4) {
        const uint4 w0 = __ldcs(p + (size_t)(kc + 0) * 128);
        const uint4 w1 = __ldcs(p + (size_t)(kc + 1) * 128);
        const uint4 w2 = __ldcs(p + (size_t)(kc + 2) * 128);
        const uint4 w3 = __ldcs(p + (size_t)(kc + 3) * 128);
        acc = dh_dot4<B>(w0, xs + (kc + 0) * XPU, acc);
        acc = dh_dot4<B>(w1, xs + (kc + 1) * XPU, acc);
        acc = dh_dot4<B>(w2, xs + (kc + 2) * XPU, acc);
        acc = dh_dot4<B>(w3, xs + (kc + 3) * XPU, acc);
    }
    for (; kc < KC; ++kc) acc = dh_dot4<B>(__ldcs(p + (size_t)kc * 128), xs + kc * XPU, acc);
    const float sx = __uint_as_float(ws[KW]);
    const int S = (int)ws[KW + 1];
    const float z = 0.5f * (float)((1 << B) - 1);
    s_y[tid] = dq * sx * ((float)acc - z * (float)S);   // approximate yraw of column tid
    __syncthreads();
    if (warp == 0) {
        float v[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = s_y[lane * 4 + r];
        exl3_had128(v, lane);
        float m = -INFINITY;
        #pragma unroll
        for (int r = 0; r < 4; ++r) m = fmaxf(m, v[r] * __half2float(svh[nb * 128 + lane * 4 + r]));
        #pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) bdst[nb] = m;
    }
}

// (2) screen: grid nblk = N/128, block 128. K <= 4096, K % (128/B) == 0.
extern "C" __global__ void __launch_bounds__(128, 12) xq_dh_screen(
        const uint4* __restrict__ codes, unsigned* __restrict__ ws, const __half* __restrict__ svh,
        int K, int B, float dq) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned xs[1024];
    __shared__ float s_y[128];
    float* const bdst = reinterpret_cast<float*>(ws + dh_off_bmax(K));
    switch (B) {
        case 1: dh_screen_body<1>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        case 2: dh_screen_body<2>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        case 4: dh_screen_body<4>(codes, ws, svh, K, dq, xs, s_y, bdst, 0); break;
        default: break;                               // the loader refuses other widths
    }
}

// Exact per-split partials of one candidate block: exl3_gemm_body's slot constants, decode and
// m16n8k16 mma (A = the one live row, zero elsewhere), ring words from smem.
template <int BITS, bool HALF = false>
__device__ __forceinline__ void dh_rescore_mma(const uint32_t* __restrict__ ringw,
                                               const unsigned* __restrict__ sx2, int R, int lane,
                                               float (&acc)[2][4]) {
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const bool live = (lane >> 2) == 0;              // gid 0 = row 0, the only live A row
    const int tig = lane & 3;
    for (int row = 0; row < R; ++row) {
        const uint32_t* ring = ringw + row * WRING;
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        // A fragments: {a0,a1} = A[gid][2tig..], {a2,a3} = A[gid+8][..] (0), {a4,a5} = A[gid][2tig+8..]
        const uint32_t a0 = live ? sx2[row * 8 + tig] : 0u;
        const uint32_t a2 = live ? sx2[row * 8 + 4 + tig] : 0u;
        const uint32_t z = 0u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(z), "r"(a2), "r"(z),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }
}

// (3) rescore: grid (S, tmax) — split-fastest, so the ns * S working CTAs are the FIRST ns * S in
// dispatch order and the (tmax - ns) * S early-exit CTAs never delay them; block 256; dynamic
// smem = R * 8 * 32 * hbits bytes (<= 32 KB). geom = pass_i | hbits << 8 | S << 16 | tmax << 24 | K << 32.
// tr / svh = the EXACT pruned head (Quad lm_head_draft: N columns, nb16 = N/16 trellis stride).
extern "C" __global__ void __launch_bounds__(256, 3) xq_dh_rescore(
        const uint16_t* __restrict__ tr, const __half* __restrict__ xh, const __half* __restrict__ svh,
        const float* __restrict__ bsig, unsigned* __restrict__ ws,
        int* __restrict__ out_id, float* __restrict__ dconf, int* __restrict__ d_dev, int* __restrict__ toks,
        long long geom, int N, float delta) {
    XQ_PDL_ENTRY();
    extern __shared__ __align__(16) uint4 s_ring4[];            // [8 warps][R rows][2 * bits]
    __shared__ unsigned s_x2[DH_RMAX * 8];
    __shared__ unsigned s_om[8];
    __shared__ int s_nq[8];
    __shared__ int s_bs[2][8];
    __shared__ int s_wh[DH_KPT * 8];                 // per (round i, warp w) selected counts
    __shared__ int s_we[DH_KPT * 8];
    __shared__ int s_ph[DH_KPT * 8];                 // their exclusive prefixes in (i, w) order
    __shared__ int s_pe[DH_KPT * 8];
    __shared__ int s_tot[2];
    __shared__ int s_cand[DH_TCAP];
    __shared__ unsigned long long s_best[8];
    __shared__ int s_last;
    const int pass_i = (int)(geom & 0xFF), bits = (int)((geom >> 8) & 0xFF);
    const int S = (int)((geom >> 16) & 0xFF), tmax = (int)((geom >> 24) & 0xFF), K = (int)(geom >> 32);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N >> 7, KW = K >> 2;
    const float* bmax = reinterpret_cast<const float*>(ws + dh_off_bmax(K));
    unsigned* cnt = ws + dh_off_cnt(K, nblk);
    int* dbg = reinterpret_cast<int*>(ws + dh_off_dbg(K, nblk));
    float* part = reinterpret_cast<float*>(ws + dh_off_part(K, nblk));
    const float sig = __uint_as_float(ws[KW + 2]);
    const float bsmax = bsig[nblk];                   // max over blocks (host-appended)

    // ---- 1. candidate selection: identical in every CTA (pure function of bmax, sig, bsig) ----
    unsigned ob[DH_KPT];                              // ordered block maxima (0 = no block)
    unsigned om = 0u;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        float v = -INFINITY;
        if (b < nblk) { v = bmax[b]; if (!(v == v)) v = -INFINITY; }
        ob[i] = b < nblk ? dh_ord(v) : 0u;
        om = max(om, ob[i]);
    }
    om = __reduce_max_sync(0xffffffffu, om);
    if (lane == 0) s_om[warp] = om;
    __syncthreads();
    unsigned go = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) go = max(go, s_om[w]);
    const float gmax = dh_unord(go);
    unsigned qm = 0u;                                 // qualified: within the noise margin of the best
    int qc = 0;
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const int b = tid + 256 * i;
        if (b < nblk) {
            const float marg = fmaxf(delta * sig * (bsig[b] + bsmax), 0.f);   // NaN -> 0
            if (dh_unord(ob[i]) + marg >= gmax) { qm |= 1u << i; ++qc; }
        }
    }
    qc = __reduce_add_sync(0xffffffffu, qc);
    if (lane == 0) s_nq[warp] = qc;
    __syncthreads();
    int nq = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) nq += s_nq[w];
    unsigned hm = qm, em = 0u;                        // ranked first (ascending id), then tie fill
    if (nq > tmax) {                                  // uniform branch
        // tau = the tmax-th largest ordered value among the qualified (32-step bisection)
        unsigned tau = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned t = tau | (1u << bit);
            int c = 0;
            #pragma unroll
            for (int i = 0; i < DH_KPT; ++i) c += (((qm >> i) & 1u) && ob[i] >= t) ? 1 : 0;
            c = __reduce_add_sync(0xffffffffu, c);
            if (lane == 0) s_bs[bit & 1][warp] = c;
            __syncthreads();
            int tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += s_bs[bit & 1][w];
            if (tot >= tmax) tau = t;
        }
        hm = 0u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            if ((qm >> i) & 1u) {
                if (ob[i] > tau) hm |= 1u << i;
                else if (ob[i] == tau) em |= 1u << i;
            }
        }
    }
    // ascending-index ranks: all of hm (count < tmax when truncating), then em fills the rest.
    // Block b = tid + 256 i, so ascending b = (i, warp, lane) order.
    #pragma unroll
    for (int i = 0; i < DH_KPT; ++i) {
        const unsigned bh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
        const unsigned be = __ballot_sync(0xffffffffu, (em >> i) & 1u);
        if (lane == 0) { s_wh[i * 8 + warp] = __popc(bh); s_we[i * 8 + warp] = __popc(be); }
    }
    __syncthreads();
    if (tid < 2 * DH_KPT * 8) {                       // 64 prefix entries per mask
        const int e = tid & (DH_KPT * 8 - 1);
        const int* src = tid < DH_KPT * 8 ? s_wh : s_we;
        int pfx = 0;
        for (int f = 0; f < e; ++f) pfx += src[f];
        (tid < DH_KPT * 8 ? s_ph : s_pe)[e] = pfx;
        if (e == DH_KPT * 8 - 1) s_tot[tid < DH_KPT * 8 ? 0 : 1] = pfx + src[e];
    }
    __syncthreads();
    const int nh = s_tot[0];
    const int ns = min(min(nh + s_tot[1], tmax), DH_TCAP);
    {
        const unsigned lt = (1u << lane) - 1u;
        #pragma unroll
        for (int i = 0; i < DH_KPT; ++i) {
            const unsigned mh = __ballot_sync(0xffffffffu, (hm >> i) & 1u);
            const unsigned me = __ballot_sync(0xffffffffu, (em >> i) & 1u);
            const int b = tid + 256 * i;
            if ((hm >> i) & 1u) {
                const int rk = s_ph[i * 8 + warp] + __popc(mh & lt);
                if (rk < ns) s_cand[rk] = b;
            }
            if ((em >> i) & 1u) {
                const int rk = nh + s_pe[i * 8 + warp] + __popc(me & lt);
                if (rk < ns) s_cand[rk] = b;
            }
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0) {         // XCHECK readback (a few words)
        if (tid < ns) dbg[1 + tid] = s_cand[tid];
        if (tid == 0) { dbg[0] = ns; dbg[DH_TCAP + 1] = nq; }
    }
    const int rc = blockIdx.y, sp = blockIdx.x;
    if (rc >= ns) return;                             // uniform per CTA
    const int cb = s_cand[rc];

    // ---- 2. exact partial of candidate cb over k16 rows [sp R, (sp + 1) R) ----
    const int R = (K >> 4) / S, kb0 = sp * R;
    const int nb16 = N >> 4;
    const int U4 = bits; // k2: uint4 chunks per trellis row (== 2*BITS integer, == 2KA+1 half)                          // 16-B chunks per trellis block (32 * bits B)
    const int rowu4 = 8 * U4;                         // one k16 row of the 8 column blocks
    for (int idx = tid; idx < R * rowu4; idx += 256) {
        const int row = idx / rowu4, rem = idx - row * rowu4;
        const int w = rem / U4, q = rem - w * U4;
        const uint4* src = reinterpret_cast<const uint4*>(tr + ((size_t)(kb0 + row) * nb16 + (size_t)cb * 8) * (8 * bits));
        s_ring4[(w * R + row) * U4 + q] = __ldcs(src + rem);
    }
    {
        const unsigned* xh2 = reinterpret_cast<const unsigned*>(xh) + kb0 * 8;
        for (int i = tid; i < R * 8; i += 256) s_x2[i] = xh2[i];
    }
    __syncthreads();
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.f;
    const uint32_t* ringw = reinterpret_cast<const uint32_t*>(s_ring4 + warp * R * U4);
    switch (bits) {
        case 4: dh_rescore_mma<2>(ringw, s_x2, R, lane, acc); break;
        case 3: dh_rescore_mma<1, true>(ringw, s_x2, R, lane, acc); break;
        case 6: dh_rescore_mma<3>(ringw, s_x2, R, lane, acc); break;
        case 8: dh_rescore_mma<4>(ringw, s_x2, R, lane, acc); break;
        case 10: dh_rescore_mma<5>(ringw, s_x2, R, lane, acc); break;
        case 12: dh_rescore_mma<6>(ringw, s_x2, R, lane, acc); break;
        default: break;
    }
    if ((lane >> 2) == 0) {                           // row 0 of the two n8 tiles: cols 16w + 8h + 2tig
        float* pp = part + ((size_t)(rc * S + sp) * 128) + warp * 16 + 2 * (lane & 3);
        pp[0] = acc[0][0]; pp[1] = acc[0][1]; pp[8] = acc[1][0]; pp[9] = acc[1][1];
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = (atomicAdd(cnt, 1u) == (unsigned)(ns * S) - 1u) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) return;

    // ---- 3. last CTA: ascending-s combine -> f16 yraw -> had_svh -> (max, lowest id) ----
    __threadfence();
    unsigned long long best = 0ull;
    for (int rr = warp; rr < ns; rr += 8) {
        const int bb = s_cand[rr];
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* pc = part + (size_t)rr * S * 128 + lane * 4 + q;
            float a = 0.f;
            for (int s2 = 0; s2 < S; ++s2) a += __ldcg(pc + (size_t)s2 * 128);
            v[q] = __half2float(__float2half_rn(a));   // yraw (exl3_gemm_body's f16 RN store)
        }
        exl3_had128(v, lane);                         // exl3_had_svh_dev's ops
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = bb * 128 + lane * 4 + q;
            const float y = __half2float(__float2half_rn(v[q] * __half2float(svh[col])));
            const unsigned long long kk = xq_am_key(y, col);
            best = kk > best ? kk : best;
        }
    }
    best = xq_am_wmax(best);
    if (lane == 0) s_best[warp] = best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long k = 0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) k = s_best[w] > k ? s_best[w] : k;
        const int id = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
        out_id[0] = id;                               // sc.argmax (== argmax_rows' lane 0)
        dconf[pass_i] = dh_unord((unsigned)(k >> 32)); // == xq_rowmax_f16 of the in-candidate max
        d_dev[pass_i] = id;                           // == the old xq_copy1_i32 pair
        toks[0] = id;
        cnt[0] = 0u;                                  // re-arm for the next launch / replay
    }
}

// ---- API-parity G2: device sampler for sampled (temperature > 0) lanes.
// temperature -> top-k -> top-p -> multinomial over fp16 logits; one 1024-thread block per row.
// Per-row params samp[row*8 + {0 flag, 1 T(f32 bits), 2 top_p(f32), 3 top_k(0=off), 4 seed_lo,
// 5 seed_hi, 6 counter, 7 min_p(f32, 0=off)}]. flag == 0 -> the row keeps xq_argmax's id
// untouched (greedy rows stay bit-identical). Overwrites out_ids[row] with the sample.
// Semantics: top-k keeps every logit >= the k-th largest (ties included, the HF warper rule);
// top-p keeps the smallest descending-logit prefix whose mass >= top_p (ties at the boundary
// included); min_p keeps p >= min_p * p_max. Sampling = inverse CDF over the survivors in a
// fixed (warp-chunk, index) order with a counter-hashed uniform — a pure function of
// (logits, params), so a fixed seed reproduces. Used for plain steps AND MTP verify rows:
// with a deterministic (argmax) draft, "sample the target at each verify row, accept while
// sample == draft" is distribution-exact (the reference implementation's EXL3_BATCH_VERIFY rule).
__device__ __forceinline__ unsigned int xq_skey(unsigned short b) {
    return (b & 0x8000u) ? ((~(unsigned int)b) & 0xFFFFu) : ((unsigned int)b | 0x8000u);
}
__device__ __forceinline__ float xq_skey_val(unsigned int k) {
    unsigned short b = (k & 0x8000u) ? (unsigned short)(k & 0x7FFFu) : (unsigned short)((~k) & 0xFFFFu);
    return __half2float(__ushort_as_half(b));
}
__device__ __forceinline__ unsigned long long xq_mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// fixed-order block reductions (deterministic: same shuffle/tree every launch)
__device__ __forceinline__ unsigned int xq_bmax_u32(unsigned int v, unsigned int* sh) {
    for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        unsigned int w = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0u;
        for (int o = 16; o; o >>= 1) w = max(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (threadIdx.x == 0) sh[0] = w;
    }
    __syncthreads();
    unsigned int r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float xq_bsum_f32(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
        for (int o = 16; o; o >>= 1) w += __shfl_xor_sync(0xffffffffu, w, o);
        if (threadIdx.x == 0) sh[0] = w;
    }
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}
extern "C" __global__ void __launch_bounds__(1024) xq_sample_rows(
        int* __restrict__ out_ids, const __half* __restrict__ logits, int V,
        const unsigned int* __restrict__ samp) {
    XQ_PDL_ENTRY();
    const int row = blockIdx.x;
    const unsigned int* P = samp + row * 8;
    if (P[0] == 0u) return;
    const float T = fmaxf(__uint_as_float(P[1]), 1e-6f);
    const float invT = 1.0f / T;
    const float top_p = __uint_as_float(P[2]);
    const unsigned int top_k = P[3];
    const unsigned long long seed = ((unsigned long long)P[5] << 32) | (unsigned long long)P[4];
    const unsigned int ctr = P[6];
    const float min_p = __uint_as_float(P[7]);
    const unsigned short* L = reinterpret_cast<const unsigned short*>(logits) + (long long)row * V;
    const int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, wid = tid >> 5, nw = bs >> 5;

    __shared__ unsigned int ush[32];
    __shared__ float fsh[32];
    __shared__ unsigned int hist[256];
    __shared__ float fh[256];
    __shared__ unsigned int s_u[4];
    __shared__ float s_f[2];

    // 1. max key -> lmax (softmax shift)
    unsigned int mk = 0u;
    for (int i = tid; i < V; i += bs) mk = max(mk, xq_skey(L[i]));
    mk = xq_bmax_u32(mk, ush);
    const float lmax = xq_skey_val(mk);

    // 2. top-k key threshold tau (radix select over the 16-bit key, ties included)
    unsigned int tau = 0u;
    if (top_k > 0u && top_k < (unsigned int)V) {
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) atomicAdd(&hist[xq_skey(L[i]) >> 8], 1u);
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, hb = 0u, need = 1u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= top_k) { hb = (unsigned int)b; need = top_k - cum; break; }
                cum += hist[b];
            }
            s_u[0] = hb; s_u[1] = need;
        }
        __syncthreads();
        const unsigned int hb = s_u[0], need = s_u[1];
        for (int i = tid; i < 256; i += bs) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if ((k >> 8) == hb) atomicAdd(&hist[k & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u, lb = 0u;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= need) { lb = (unsigned int)b; break; }
                cum += hist[b];
            }
            s_u[2] = (hb << 8) | lb;
        }
        __syncthreads();
        tau = s_u[2];
    }
    // min_p as a logit floor: p >= min_p * p_max  <=>  l >= lmax + T * ln(min_p)
    const float mfloor = (min_p > 0.0f) ? lmax + T * __logf(min_p) : -INFINITY;

    // 3. candidate mass S (deterministic)
    float part = 0.0f;
    for (int i = tid; i < V; i += bs) {
        unsigned int k = xq_skey(L[i]);
        if (k >= tau) {
            float x = xq_skey_val(k);
            if (x >= mfloor) part += __expf((x - lmax) * invT);
        }
    }
    const float S = xq_bsum_f32(part, fsh);

    // 4. top-p key threshold theta (mass radix select; boundary ties included)
    unsigned int thr = tau;
    if (top_p < 1.0f && S > 0.0f) {
        const float target = top_p * S;
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k >> 8], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int hb = 0u; float rem = 0.0f; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= target) { hb = (unsigned int)b; rem = target - cum; found = 1; break; }
                cum += fh[b];
            }
            if (!found) { hb = (unsigned int)max(last, 0); rem = 3.0e38f; }
            s_u[0] = hb; s_f[0] = rem;
        }
        __syncthreads();
        const unsigned int hb = s_u[0];
        const float rem = s_f[0];
        for (int i = tid; i < 256; i += bs) fh[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < V; i += bs) {
            unsigned int k = xq_skey(L[i]);
            if (k >= tau && (k >> 8) == hb) {
                float x = xq_skey_val(k);
                if (x >= mfloor) atomicAdd(&fh[k & 255u], __expf((x - lmax) * invT));
            }
        }
        __syncthreads();
        if (tid == 0) {
            float cum = 0.0f; unsigned int lb = 0u; int last = -1, found = 0;
            for (int b = 255; b >= 0; b--) {
                if (fh[b] > 0.0f) last = b;
                if (cum + fh[b] >= rem) { lb = (unsigned int)b; found = 1; break; }
                cum += fh[b];
            }
            if (!found) lb = (unsigned int)max(last, 0);
            s_u[3] = max((hb << 8) | lb, tau);
        }
        __syncthreads();
        thr = s_u[3];
    }

    // 5. inverse CDF over survivors: warp w owns [w*chunk, (w+1)*chunk), lanes stride 32.
    const int chunk = (V + nw - 1) / nw;
    const int w0 = wid * chunk, w1 = min(V, w0 + chunk);
    float lsum = 0.0f;
    for (int i = w0 + lane; i < w1; i += 32) {
        unsigned int k = xq_skey(L[i]);
        if (k >= thr) {
            float x = xq_skey_val(k);
            if (x >= mfloor) lsum += __expf((x - lmax) * invT);
        }
    }
    for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
    if (lane == 0) fsh[wid] = lsum;
    __syncthreads();
    if (tid == 0) {
        const float u = ((float)(xq_mix64(seed ^ xq_mix64((unsigned long long)ctr + 1ull)) >> 40) + 0.5f)
                        * (1.0f / 16777216.0f);
        float tot = 0.0f;
        for (int w = 0; w < nw; w++) tot += fsh[w];
        const float target = u * tot;
        float cum = 0.0f; int sel = -1, last = 0;
        for (int w = 0; w < nw; w++) {
            if (fsh[w] > 0.0f) last = w;
            if (sel < 0 && cum + fsh[w] > target) { sel = w; break; }
            cum += fsh[w];
        }
        if (sel < 0) { sel = last; cum = 0.0f; for (int w = 0; w < sel; w++) cum += fsh[w]; }
        s_u[0] = (unsigned int)sel; s_f[0] = target - cum;
    }
    __syncthreads();
    if (wid == (int)s_u[0]) {
        const float want = s_f[0];
        float base = 0.0f;
        int pick = -1, lastidx = -1;
        for (int i0 = w0; i0 < w1 && pick < 0; i0 += 32) {
            const int i = i0 + lane;
            float wgt = 0.0f;
            if (i < w1) {
                unsigned int k = xq_skey(L[i]);
                if (k >= thr) {
                    float x = xq_skey_val(k);
                    if (x >= mfloor) wgt = __expf((x - lmax) * invT);
                }
            }
            float incl = wgt;
            for (int o = 1; o < 32; o <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            unsigned int hit = __ballot_sync(0xffffffffu, wgt > 0.0f && base + incl > want);
            unsigned int any = __ballot_sync(0xffffffffu, wgt > 0.0f);
            if (any) lastidx = i0 + 31 - __clz(any);
            if (hit) pick = i0 + __ffs(hit) - 1;
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
        // no survivor at all (non-finite row) -> keep xq_argmax's id
        if (lane == 0 && (pick >= 0 || lastidx >= 0)) out_ids[row] = (pick >= 0) ? pick : lastidx;
    }
}

