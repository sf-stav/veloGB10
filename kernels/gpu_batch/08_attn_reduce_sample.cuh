// gpu_batch/08_attn_reduce_sample.cuh: register-staged attention tiles, split merge (gqa_attn_reduce), concat/sample/spec-verify/nll kernels.
// The merge recomputes ns from this column's pc exactly as gqa_attn_splitk did.
// Section 8 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// p3: 16-key tiles + register-staged pipeline (oracle-validated +18% bf16 @30K, SEG 56;
// numerics parity rel(f32-ref) 2.29e-3 == stock kernel class). See STEP3_PLAN round-20/21.
extern "C" __global__ void __launch_bounds__(192) gqa_attn_verify_e_p3(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ pos, long long bs_packed, int nh_packed,
    const int* __restrict__ slot_ids, const int* __restrict__ col_pos_start) {
    (void)col_pos_start;                             // identity path: logical == physical
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128)      gve_p3_impl<128>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
    else if (hd == 256) gve_p3_impl<256>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
}

extern "C" __global__ void __launch_bounds__(192) gqa_attn_verify_e(
    float* __restrict__ out_m, float* __restrict__ out_l, float* __restrict__ out_acc,
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ pos, long long bs_packed, int nh_packed,
    const int* __restrict__ slot_ids, const int* __restrict__ col_pos_start) {
    (void)col_pos_start;                             // identity path: logical == physical
    const int hd = (nh_packed >> 10) & 0x3FF;
    if (hd == 128)      gve_impl<128>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
    else if (hd == 256) gve_impl<256>(out_m, out_l, out_acc, q, k_cache, v_cache, pos, bs_packed, nh_packed, slot_ids);
}

extern "C" __global__ void gqa_attn_reduce_e(
    __nv_bfloat16* out,
    const float* in_m, const float* in_l, const float* in_acc,
    const int* pos, int ns_grid, int B, int nh_packed) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int blk = blockIdx.x;
    const int b = blk / nh;
    if (b >= B) return;
    const int qh = blk % nh;
    const int d = threadIdx.x;
    const int ns = ns_grid;                        // _e: SEG is the authority, no recompute

    float m = -1e30f;
    for (int s = 0; s < ns; s++) {
        const long long idx = ((long long)b * nh + qh) * ns_grid + s;
        m = fmaxf(m, in_m[idx]);
    }
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < ns; s++) {                 // FIXED order -> deterministic
        const long long idx = ((long long)b * nh + qh) * ns_grid + s;
        const float alpha = __expf(in_m[idx] - m);
        l   += in_l[idx] * alpha;
        acc += in_acc[idx * hd + d] * alpha;
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = f2b(l > 0.0f ? acc / l : 0.0f);
}

// Merge a column's partial softmaxes. It recomputes ns from THIS COLUMN's pc, exactly as
// gqa_attn_splitk did -- the two must agree or the reduction reads partials that were never written.
// `ns_grid` is only the stride of the partial buffer, never the loop bound.
extern "C" __global__ void gqa_attn_reduce(
    __nv_bfloat16* out,
    const float* in_m, const float* in_l, const float* in_acc,
    const int* pos, int ns_grid, int B, int nh_packed) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int blk = blockIdx.x;
    const int b = blk / nh;
    if (b >= B) return;
    const int qh = blk % nh;
    const int d = threadIdx.x;
    const int ns = sk_nsplits(pos[b] + 1);

    float m = -1e30f;
    for (int s = 0; s < ns; s++) {
        const long long idx = ((long long)b * nh + qh) * ns_grid + s;
        m = fmaxf(m, in_m[idx]);
    }

    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < ns; s++) {                 // FIXED order -> deterministic
        const long long idx = ((long long)b * nh + qh) * ns_grid + s;
        const float alpha = __expf(in_m[idx] - m);
        l   += in_l[idx] * alpha;
        acc += in_acc[idx * hd + d] * alpha;
    }

    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = f2b(l > 0.0f ? acc / l : 0.0f);
}

#ifndef SAMPLE_K_MAX
#define SAMPLE_K_MAX 64
#endif

// ---- concat_b: interleave two [h, batch] bf16 tensors into one [2h, batch] ----
// out[b*2h + i] = (i < h) ? a[b*h + i] : b_in[b*h + (i - h)]. Used by the MTP FC fusion layer
// to build [norm(h_t), norm(e_{t+1})] without a host round-trip.
extern "C" __global__ void concat_b(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b_in,
                                    int h, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * h * batch;
    if (idx >= total) return;
    int brow = idx / (2 * h);
    int i = idx % (2 * h);
    out[idx] = (i < h) ? a[brow * h + i] : b_in[brow * h + (i - h)];
}

// ---- sample_b: on-GPU multinomial sampling (temperature -> top-k -> softmax -> top-p -> sample)
// logits:    [batch, v] bf16, row-major (lane i at i*v). MODIFIED in place: selected entries are
//            masked to -inf so the iterative top-k selection never repeats a token. Safe because the
//            logits buffer is freshly produced each decode step and released after this call.
// token_ids: [batch] i32 output. temps/top_ks/top_ps/seeds: per-lane [batch].
// One block per lane; blockDim.x must be a power of two. Fully on-GPU (no host sync) so it is
// graph-capturable. Shared memory layout: topv[K_MAX] floats, topi[K_MAX] ints, then a
// pair-reduction scratch of 2*blockDim.x floats.
extern "C" __global__ void sample_b(int* token_ids, __nv_bfloat16* logits,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int batch) {
    int lane = blockIdx.x;
    if (lane >= batch) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    __nv_bfloat16* row = logits + (long long)lane * v;
    float temp = temps[lane];
    int topk = top_ks[lane];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[lane];

    extern __shared__ float sh[];
    float* topv = sh;                                  // SAMPLE_K_MAX scaled logits (descending)
    int*   topi = (int*)(sh + SAMPLE_K_MAX);           // SAMPLE_K_MAX matching token indices
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;    // red[2*tid]=value, red[2*tid+1]=index

    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int j = tid; j < v; j += nthr) { float val = b2f(row[j]); if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) token_ids[lane] = (int)red[1];
        return;
    }

    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int j = tid; j < v; j += nthr) { float val = b2f(row[j]); if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            row[(int)red[1]] = f2b(-1e30f);   // mask so the next pass skips it
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int j = 0; j < topk; j++) { probs[j] = __expf(topv[j] - mx); sum += probs[j]; }
        // nucleus (top-p): smallest prefix whose normalized cumsum reaches topp
        float cum = 0.0f; int nc = topk - 1;
        for (int j = 0; j < topk; j++) { cum += probs[j]; if (cum >= topp * sum) { nc = j; break; } }
        unsigned int s = seeds[lane];
        s = s * 1664525u + 1013904223u;
        float r = (s >> 8) * (1.0f / 16777216.0f);   // [0,1)
        r *= cum;                                    // scale to the nucleus mass
        float acc = 0.0f; int chosen = topi[nc];
        for (int j = 0; j <= nc; j++) { acc += probs[j]; if (r < acc) { chosen = topi[j]; break; } }
        token_ids[lane] = chosen;
    }
}

// f32 twin of sample_b for hy_v3's `enable_lm_head_fp32` contract: IDENTICAL selection (same top-k
// passes, same in-place masking, same LCG draw) but over the UNROUNDED fp32 logits — a bf16 round of
// the row perturbs near-tie selections, which at temp>0 is a measurable distribution bias. `logits`
// stays DESTRUCTIVE (masked in place); the buffer is freshly produced each decode step and released
// after this call, exactly as sample_b's is.
extern "C" __global__ void sample_f32_b(int* token_ids, float* logits,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int batch) {
    int lane = blockIdx.x;
    if (lane >= batch) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    float* row = logits + (long long)lane * v;
    float temp = temps[lane];
    int topk = top_ks[lane];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[lane];

    extern __shared__ float sh[];
    float* topv = sh;                                  // SAMPLE_K_MAX scaled logits (descending)
    int*   topi = (int*)(sh + SAMPLE_K_MAX);           // SAMPLE_K_MAX matching token indices
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;    // red[2*tid]=value, red[2*tid+1]=index

    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int j = tid; j < v; j += nthr) { float val = row[j]; if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) token_ids[lane] = (int)red[1];
        return;
    }

    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int j = tid; j < v; j += nthr) { float val = row[j]; if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            row[(int)red[1]] = -1e30f;   // mask so the next pass skips it
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int j = 0; j < topk; j++) { probs[j] = __expf(topv[j] - mx); sum += probs[j]; }
        // nucleus (top-p): smallest prefix whose normalized cumsum reaches topp
        float cum = 0.0f; int nc = topk - 1;
        for (int j = 0; j < topk; j++) { cum += probs[j]; if (cum >= topp * sum) { nc = j; break; } }
        unsigned int s = seeds[lane];
        s = s * 1664525u + 1013904223u;
        float r = (s >> 8) * (1.0f / 16777216.0f);   // [0,1)
        r *= cum;                                    // scale to the nucleus mass
        float acc = 0.0f; int chosen = topi[nc];
        for (int j = 0; j <= nc; j++) { acc += probs[j]; if (r < acc) { chosen = topi[j]; break; } }
        token_ids[lane] = chosen;
    }
}

// ===================== BATCH-INVARIANT GEMM =====================
// C[M, N] (col-major, ld=M) = W[M,K] (row-major) ^T-style @ X[K,N] (col-major, ld=K):
//   C[n*M + m] = sum_k W[m*K + k] * X[n*K + k]   (fp32 accumulate, bf16 round on write)
//
// BATCH-INVARIANT: C[m,0] is bit-identical for N=1 and any N, because each output element's
// k-reduction is performed by a fixed set of threads in a fixed order (one block per output row m;
// threads split K into a fixed strided pattern; tree-reduce in a fixed shape). No Split-K across
// blocks, no N-dependent tiling. This is what makes the MTP verify (N=K) numerically match the
// decode (N=1) — fixing the cuBLAS N=1-vs-N=2 divergence that broke 9B/27B MTP.
//
// Grid: (M,1,1). Block: (T,1,1) with T a fixed thread count (e.g. 256). Shared mem: Nmax*T*4 bytes.
// Nmax must be >= the largest N the kernel is launched with (acc is statically Nmax-sized).
#define GEMM_BINV_NMAX 16

// Output store: bf16 rounds (the serving path); f32 keeps the unrounded accumulator — that, and only
// that, is what hy_v3's `enable_lm_head_fp32` asks of the logits GEMM (the accumulation was always
// fp32). The bf16 instantiation inlines to exactly the old epilogue, so qwen output is unchanged.
__device__ __forceinline__ void cstore(__nv_bfloat16* p, float v) { *p = f2b(v); }
__device__ __forceinline__ void cstore(float* p, float v) { *p = v; }

// The body is templated on N so that `acc[]` is a compile-time-sized register array.
//
// With a RUNTIME N, `acc[GEMM_BINV_NMAX]` is dynamically indexed, so the compiler is forced to place
// it in LOCAL memory (a 64-byte stack frame). Every `acc[n] +=` then becomes a local load+store, and
// that traffic scales with N while the (dominant, bandwidth-bound) W read does not. The result was a
// kernel that hit full bandwidth at N=1 but was 23% slower at N=2 — precisely the width the MTP
// verify runs at, so it silently ate most of the speculative-decoding win. Compile-time N keeps acc
// in registers and makes N=2 cost what N=1 costs.
//
// The arithmetic is UNCHANGED: same strided k-loop, same fixed tree-reduce shape, same order. So
// column 0 stays bit-identical to the N=1 decode and the greedy-lossless guarantee is preserved.
template<int NC, typename CT>
__device__ __forceinline__ void gemm_binv_impl(CT* C, const __nv_bfloat16* W,
                                               const __nv_bfloat16* X, int M, int K) {
    int m = blockIdx.x;
    int t = threadIdx.x;
    int T = blockDim.x;
    const __nv_bfloat16* Wrow = W + (long long)m * K;
    float acc[NC];
    #pragma unroll
    for (int n = 0; n < NC; n++) acc[n] = 0.0f;
    // Strided k-loop: thread t handles k = t, t+T, t+2T, ... (consecutive threads -> coalesced).
    for (int k = t; k < K; k += T) {
        float w = b2f(Wrow[k]);
        #pragma unroll
        for (int n = 0; n < NC; n++) acc[n] += w * b2f(X[(long long)n * K + k]);
    }
    // Tree-reduce each column across the T threads (fixed shape -> N-independent reduction order).
    extern __shared__ float sh[];  // [N][T]
    #pragma unroll
    for (int n = 0; n < NC; n++) sh[n * T + t] = acc[n];
    __syncthreads();
    for (int stride = T >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            #pragma unroll
            for (int n = 0; n < NC; n++) sh[n * T + t] += sh[n * T + t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        #pragma unroll
        for (int n = 0; n < NC; n++) cstore(C + (long long)n * M + m, sh[n * T + 0]);
    }
}

// Generic fallback (runtime N, acc in local memory) for widths without a specialization —
// e.g. a future batched multi-lane verify. Same reduction order as the templated path.
template<typename CT>
__device__ __forceinline__ void gemm_binv_fallback(CT* C, const __nv_bfloat16* W,
                                                   const __nv_bfloat16* X, int M, int K, int N) {
    int m = blockIdx.x;
    int t = threadIdx.x;
    int T = blockDim.x;
    const __nv_bfloat16* Wrow = W + (long long)m * K;
    float acc[GEMM_BINV_NMAX];
    for (int n = 0; n < N; n++) acc[n] = 0.0f;
    for (int k = t; k < K; k += T) {
        float w = b2f(Wrow[k]);
        for (int n = 0; n < N; n++) acc[n] += w * b2f(X[(long long)n * K + k]);
    }
    extern __shared__ float sh[];
    for (int n = 0; n < N; n++) sh[n * T + t] = acc[n];
    __syncthreads();
    for (int stride = T >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            for (int n = 0; n < N; n++) sh[n * T + t] += sh[n * T + t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int n = 0; n < N; n++) cstore(C + (long long)n * M + m, sh[n * T + 0]);
    }
}

// ===================== E19: fused draft-head argmax (FR-Spec) =====================

// Draft-head GEMV with a FUSED argmax partial epilogue: the per-row dot is the SAME fixed-order
// instruction sequence as gemm_binv_impl<1> (one block per row, 256-thread strided k-loop, same
// tree shape), so every row's value is bit-identical to the gemm_binv_b materialization at N=1 —
// but instead of writing the [M] logits it writes ONE (value, row) partial pair per row, and a tiny
// final reduce picks the winner. The draft chain's per-token logits traffic (write vocab + re-read
// vocab for argmax_b) disappears; the partials are written once and read once. bf16 head only (the
// hy_v3 FR-Spec shape); NVFP4/FP8 draft heads keep the materialized path.
extern "C" __global__ void gemm_binv_argmax_b(float* part, const __nv_bfloat16* W,
                                              const __nv_bfloat16* X, int M, int K) {
    int m = blockIdx.x;
    int t = threadIdx.x;
    int T = blockDim.x;
    const __nv_bfloat16* Wrow = W + (long long)m * K;
    float acc = 0.0f;
    for (int k = t; k < K; k += T) acc += b2f(Wrow[k]) * b2f(X[k]);
    extern __shared__ float sh[];
    sh[t] = acc;
    __syncthreads();
    for (int stride = T >> 1; stride > 0; stride >>= 1) {
        if (t < stride) sh[t] += sh[t + stride];
        __syncthreads();
    }
    if (t == 0) { part[2 * m] = sh[0]; part[2 * m + 1] = (float)m; }
}

// Final reduce of the (value, row) partials: strictly-greater comparison in the same scan + tree
// shape as argmax_b, so the winner (including tie order) is identical to argmax_b over the same
// values — deterministic by construction, no atomics anywhere.
extern "C" __global__ void argmax_part_b(int* token_ids, const float* part, int M) {
    int t = threadIdx.x;
    int T = blockDim.x;
    float lmx = -1e30f; int lidx = 0;
    for (int j = t; j < M; j += T) { float val = part[2 * j]; if (val > lmx) { lmx = val; lidx = (int)part[2 * j + 1]; } }
    extern __shared__ float red[];
    red[2 * t] = lmx; red[2 * t + 1] = (float)lidx;
    __syncthreads();
    for (int s2 = T / 2; s2 > 0; s2 >>= 1) {
        if (t < s2 && red[2 * (t + s2)] > red[2 * t]) { red[2 * t] = red[2 * (t + s2)]; red[2 * t + 1] = red[2 * (t + s2) + 1]; }
        __syncthreads();
    }
    if (t == 0) token_ids[0] = (int)red[1];
}

template<typename CT>
__device__ __forceinline__ void gemm_binv_dispatch(CT* C, const __nv_bfloat16* W,
                                                   const __nv_bfloat16* X, int M, int K, int N) {
    // N is uniform across the block, so this switch never diverges.
    switch (N) {
        case 1: gemm_binv_impl<1>(C, W, X, M, K); return;
        case 2: gemm_binv_impl<2>(C, W, X, M, K); return;
        case 3: gemm_binv_impl<3>(C, W, X, M, K); return;
        case 4: gemm_binv_impl<4>(C, W, X, M, K); return;
        case 5: gemm_binv_impl<5>(C, W, X, M, K); return;
        case 6: gemm_binv_impl<6>(C, W, X, M, K); return;
        case 7: gemm_binv_impl<7>(C, W, X, M, K); return;
        case 8: gemm_binv_impl<8>(C, W, X, M, K); return;
        default: gemm_binv_fallback(C, W, X, M, K, N); return;
    }
}

extern "C" __global__ void gemm_binv_b(__nv_bfloat16* C, const __nv_bfloat16* W,
                                       const __nv_bfloat16* X, int M, int K, int N) {
    if (blockIdx.x >= M) return;
    gemm_binv_dispatch(C, W, X, M, K, N);
}

// f32-output twin of gemm_binv_b — hy_v3's enable_lm_head_fp32 logits path. IDENTICAL reduction
// structure (same strided-k, same tree, same templated N), so it is batch-invariant for exactly the
// same reason; only the epilogue store skips the bf16 round.
extern "C" __global__ void gemm_binv_f32_b(float* C, const __nv_bfloat16* W,
                                           const __nv_bfloat16* X, int M, int K, int N) {
    if (blockIdx.x >= M) return;
    gemm_binv_dispatch(C, W, X, M, K, N);
}

// ===================== STOCHASTIC MTP KERNELS =====================

// sample_prob_b: like sample_b but also writes the chosen token's normalized probability q(x)
// under the temperature→topk→softmax→topp nucleus, so the MTP accept loop can compute
// min(1, p_target(x) / q_draft(x)).
extern "C" __global__ void sample_prob_b(int* token_ids, float* qprobs, __nv_bfloat16* logits,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int batch) {
    int lane = blockIdx.x;
    if (lane >= batch) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    __nv_bfloat16* row = logits + (long long)lane * v;
    float temp = temps[lane];
    int topk = top_ks[lane];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[lane];

    extern __shared__ float sh[];
    float* topv = sh;
    int*   topi = (int*)(sh + SAMPLE_K_MAX);
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;

    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int j = tid; j < v; j += nthr) { float val = b2f(row[j]); if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) { token_ids[lane] = (int)red[1]; qprobs[lane] = 1.0f; }
        return;
    }

    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int j = tid; j < v; j += nthr) { float val = b2f(row[j]); if (val > lmx) { lmx = val; lidx = j; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) { topv[s] = red[0] * inv_temp; topi[s] = (int)red[1]; row[(int)red[1]] = f2b(-1e30f); }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int j = 0; j < topk; j++) { probs[j] = __expf(topv[j] - mx); sum += probs[j]; }
        float cum = 0.0f; int nc = topk - 1;
        for (int j = 0; j < topk; j++) { cum += probs[j]; if (cum >= topp * sum) { nc = j; break; } }
        unsigned int s = seeds[lane];
        s = s * 1664525u + 1013904223u;
        float r = (s >> 8) * (1.0f / 16777216.0f);
        r *= cum;
        float acc = 0.0f; int chosen = topi[nc]; int chosen_j = nc;
        for (int j = 0; j <= nc; j++) { acc += probs[j]; if (r < acc) { chosen = topi[j]; chosen_j = j; break; } }
        token_ids[lane] = chosen;
        qprobs[lane] = probs[chosen_j] / cum;
    }
}

// spec_verify_b: per-column (grid=verify-depth) target-distribution logic for stochastic MTP.
// One block per verify column j. Computes the nucleus-normalized target distribution p_j, then:
//   for j < depth-1: writes p_of_draft[j] = p_j(draft_tokens[j]) and resid_tok[j] = sample from p_j
//                    with the drafted token's mass zeroed (pragmatic residual).
//   for j == depth-1: writes bonus_tok = sample from full p_j.
// resid_tok is [depth]: columns 0..depth-2 hold the residual resample for each drafted position,
// and column depth-1 holds the all-accepted BONUS token. Folding the bonus in here (rather than a
// separate scalar buffer) saves an allocation and a device->host readback on every decode step.
// NOTE: `logits` is DESTRUCTIVE — the top-k selection masks each chosen entry to -inf in place, so
// the caller must not read the logits afterwards (verify_forward_sample releases them immediately).
// It previously took logits as const and, unable to mask, re-scanned the already-picked list for
// every vocab element on every pass: O(topk^2 * vocab) ~ 50M comparisons vs sample_b's O(topk*vocab).
// That cost ~10 ms/step -- as much as the GDN rollback, draft and re-prime combined -- purely
// because of the const. Masking in place makes it identical in cost (and in nucleus) to sample_b.
extern "C" __global__ void spec_verify_b(
    float* p_of_draft, int* resid_tok,
    __nv_bfloat16* logits, const int* draft_tokens, const float* draft_qprobs,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int depth) {
    int j = blockIdx.x;
    if (j >= depth) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    __nv_bfloat16* col = logits + (long long)j * v;
    float temp = temps[j];
    int topk = top_ks[j];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[j];

    extern __shared__ float sh[];
    float* topv = sh;
    int*   topi = (int*)(sh + SAMPLE_K_MAX);
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;

    // Greedy path: target is a point mass at the argmax.
    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int k = tid; k < v; k += nthr) { float val = b2f(col[k]); if (val > lmx) { lmx = val; lidx = k; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            int argmax_tok = (int)red[1];
            if (j < depth - 1) {
                p_of_draft[j] = (argmax_tok == draft_tokens[j]) ? 1.0f : 0.0f;
            }
            resid_tok[j] = argmax_tok;   // j == depth-1 => the bonus slot
        }
        return;
    }

    // Stochastic path: top-k → softmax → top-p nucleus. Selection is now identical to sample_b's,
    // in-place mask included, so the two kernels agree on the nucleus by construction.
    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int k = tid; k < v; k += nthr) {
            float val = b2f(col[k]); if (val > lmx) { lmx = val; lidx = k; }
        }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            col[(int)red[1]] = f2b(-1e30f);   // mask so the next pass skips it
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int s = 0; s < topk; s++) { probs[s] = __expf(topv[s] - mx); sum += probs[s]; }
        float cum = 0.0f; int nc = topk - 1;
        for (int s = 0; s < topk; s++) { cum += probs[s]; if (cum >= topp * sum) { nc = s; break; } }

        if (j < depth - 1) {
            int draft_tok = draft_tokens[j];
            float p_draft = 0.0f;
            int draft_idx = -1;
            for (int s = 0; s <= nc; s++) {
                if (topi[s] == draft_tok) { draft_idx = s; p_draft = probs[s] / cum; break; }
            }
            p_of_draft[j] = p_draft;

            // Resample from residual (p \ {draft}) — pragmatic variant: zero out draft's mass.
            unsigned int sr = seeds[j];
            sr = sr * 1664525u + 1013904223u;
            float ru = (sr >> 8) * (1.0f / 16777216.0f);

            float resid_cum = (draft_idx >= 0) ? cum - probs[draft_idx] : cum;
            if (resid_cum <= 0.0f) {
                // All nucleus mass on the draft token → fall back to full distribution.
                resid_cum = cum; ru *= cum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) { acc += probs[s]; if (ru < acc) { chosen = topi[s]; break; } }
                resid_tok[j] = chosen;
            } else {
                ru *= resid_cum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) {
                    if (s == draft_idx) continue;
                    acc += probs[s];
                    if (ru < acc) { chosen = topi[s]; break; }
                }
                resid_tok[j] = chosen;
            }
        } else {
            // Bonus column: sample from full target p_{depth-1}.
            unsigned int sb = seeds[j];
            sb = sb * 1664525u + 1013904223u;
            float rb = (sb >> 8) * (1.0f / 16777216.0f);
            rb *= cum;
            float acc = 0.0f; int chosen = topi[nc];
            for (int s = 0; s <= nc; s++) { acc += probs[s]; if (rb < acc) { chosen = topi[s]; break; } }
            resid_tok[j] = chosen;   // j == depth-1 => the bonus slot
        }
    }
}

// spec_verify_realq_b: spec_verify_b with the REAL-q rejection-sampling residual (S5F2 L2 —
// the SGLang `speculative_sampling_classic_kernel` semantics, reject_sampling.py:112-115).
// IDENTICAL nucleus selection + p_of_draft + bonus (pure p) logic; the difference is the
// residual on rejection: the EXACT relu(p - q) renormalized over the nucleus, where q is the
// selector's candidate-table weight at every candidate token (0 elsewhere), instead of the
// pragmatic p \ {draft}. With q = 1 (deterministic drafts) the two residuals coincide, so this
// is the q≠1 generalization the real-q accept (u*q < p) requires for distribution exactness.
// The candidate table (cand_tok/cand_q, [depth][16] each) is per-column. Degenerate residual
// (p == q on the support — numerically measure-zero) falls back to a pure-p sample.
extern "C" __global__ void spec_verify_realq_b(
    float* p_of_draft, int* resid_tok,
    __nv_bfloat16* logits, const int* draft_tokens, const float* draft_qprobs,
    const unsigned long long* cand,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int depth) {
    int j = blockIdx.x;
    if (j >= depth) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    __nv_bfloat16* col = logits + (long long)j * v;
    float temp = temps[j];
    int topk = top_ks[j];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[j];

    extern __shared__ float sh[];
    float* topv = sh;
    int*   topi = (int*)(sh + SAMPLE_K_MAX);
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;

    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int k = tid; k < v; k += nthr) { float val = b2f(col[k]); if (val > lmx) { lmx = val; lidx = k; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            int argmax_tok = (int)red[1];
            if (j < depth - 1) {
                p_of_draft[j] = (argmax_tok == draft_tokens[j]) ? 1.0f : 0.0f;
            }
            resid_tok[j] = argmax_tok;   // j == depth-1 => the bonus slot
        }
        return;
    }

    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int k = tid; k < v; k += nthr) {
            float val = b2f(col[k]); if (val > lmx) { lmx = val; lidx = k; }
        }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            col[(int)red[1]] = f2b(-1e30f);   // mask so the next pass skips it
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int s = 0; s < topk; s++) { probs[s] = __expf(topv[s] - mx); sum += probs[s]; }
        float cum = 0.0f; int nc = topk - 1;
        for (int s = 0; s < topk; s++) { cum += probs[s]; if (cum >= topp * sum) { nc = s; break; } }

        if (j < depth - 1) {
            int draft_tok = draft_tokens[j];
            float p_draft = 0.0f;
            int draft_idx = -1;
            for (int s = 0; s <= nc; s++) {
                if (topi[s] == draft_tok) { draft_idx = s; p_draft = probs[s] / cum; break; }
            }
            p_of_draft[j] = p_draft;

            // Resample from the EXACT residual relu(p - q)+ renormalized (SGLang classic
            // kernel): p_s = the renormalized nucleus probability, q_s = the candidate table's
            // weight when the nucleus token is one of the 16 candidates, else 0.
            unsigned int sr = seeds[j];
            sr = sr * 1664525u + 1013904223u;
            float ru = (sr >> 8) * (1.0f / 16777216.0f);

            // Packed candidate pairs: (q bits << 32) | token id.
            float resid[SAMPLE_K_MAX];
            float resid_sum = 0.0f;
            for (int s = 0; s <= nc; s++) {
                float ps = probs[s] / cum;
                float qs = 0.0f;
                for (int k = 0; k < 16; k++) {
                    unsigned long long pair = cand[j * 16 + k];
                    if (topi[s] == (int)(pair & 0xffffffffull)) {
                        qs = __uint_as_float((unsigned)(pair >> 32));
                        break;
                    }
                }
                float r = ps - qs;
                resid[s] = (r > 0.0f) ? r : 0.0f;
                resid_sum += resid[s];
            }
            if (resid_sum <= 0.0f) {
                // Degenerate (p == q on the support): fall back to a pure-p sample.
                ru *= cum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) { acc += probs[s]; if (ru < acc) { chosen = topi[s]; break; } }
                resid_tok[j] = chosen;
            } else {
                ru *= resid_sum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) {
                    acc += resid[s];
                    if (ru < acc) { chosen = topi[s]; break; }
                }
                resid_tok[j] = chosen;
            }
        } else {
            // Bonus column: sample from full target p_{depth-1}.
            unsigned int sb = seeds[j];
            sb = sb * 1664525u + 1013904223u;
            float rb = (sb >> 8) * (1.0f / 16777216.0f);
            rb *= cum;
            float acc = 0.0f; int chosen = topi[nc];
            for (int s = 0; s <= nc; s++) { acc += probs[s]; if (rb < acc) { chosen = topi[s]; break; } }
            resid_tok[j] = chosen;   // j == depth-1 => the bonus slot
        }
    }
}

// df2_topk20_dump_b: S5F3 — dump-only. For every verify column j (the target distribution at
// position pos+1+j), compute the top-k-20 selection + softmax + top-p nucleus EXACTLY as
// spec_verify_b / spec_verify_realq_b do, and write the (token, renorm-p) table to
// `table_out[j * 20 + s]` packed as `(p_bits << 32) | tok` (renorm p = probs[s]/cum inside the
// nucleus, 0 outside). The selection runs on a per-column COPY of the logits (the scratch), so
// the original logits stay pristine for the verify kernel that follows. The offline analysis
// cross-checks p_of_draft against this table (the S3 p-fidelity check) and uses the table to
// score arbitrary tokens (the oracle-replay drafts). Writes nothing when the column is greedy
// (temp < 1e-6 — the target is a point mass; the analysis handles that case from p_of_draft).
extern "C" __global__ void df2_topk20_dump_b(
    unsigned long long* table_out,
    __nv_bfloat16* col_scratch,
    const __nv_bfloat16* logits,
    const float* temps, const int* top_ks, const float* top_ps,
    int v, int depth) {
    int j = blockIdx.x;
    if (j >= depth) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    const __nv_bfloat16* col = logits + (long long)j * v;
    __nv_bfloat16* scr = col_scratch + (long long)j * v;
    float temp = temps[j];
    int topk = top_ks[j];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    if (topk > 20) topk = 20;      // the dump table is 20 wide (the protocol's k20)
    float topp = top_ps[j];

    // Copy the column (bit-identical values; the selection then masks the copy).
    for (int k = tid; k < v; k += nthr) { scr[k] = col[k]; }
    __syncthreads();

    extern __shared__ float sh[];
    float* topv = sh;
    int*   topi = (int*)(sh + SAMPLE_K_MAX);
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;

    // Greedy column: nothing to dump (point mass — the analysis reads p_of_draft).
    if (temp < 1e-6f) { if (tid == 0) table_out[j * 20] = 0ull; return; }

    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int k = tid; k < v; k += nthr) {
            float val = b2f(scr[k]); if (val > lmx) { lmx = val; lidx = k; }
        }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            scr[(int)red[1]] = f2b(-1e30f);   // mask so the next pass skips it (same as spec_verify_b)
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int s = 0; s < topk; s++) { probs[s] = __expf(topv[s] - mx); sum += probs[s]; }
        float cum = 0.0f; int nc = topk - 1;
        for (int s = 0; s < topk; s++) { cum += probs[s]; if (cum >= topp * sum) { nc = s; break; } }
        for (int s = 0; s < topk; s++) {
            float p = (s <= nc) ? probs[s] / cum : 0.0f;
            unsigned long long word = ((unsigned long long)__float_as_uint(p) << 32)
                                    | (unsigned)topi[s];
            table_out[j * 20 + s] = word;
        }
        if (topk < 20) {
            for (int s = topk; s < 20; s++) { table_out[j * 20 + s] = 0ull; }
        }
    }
}

// f32 twin of spec_verify_b for hy_v3's `enable_lm_head_fp32` contract: IDENTICAL nucleus selection,
// p_of_draft, residual-resample and bonus logic (same top-k passes, same in-place masking, same LCG
// draws) but over the UNROUNDED fp32 logits — a bf16 round of the column perturbs near-tie selections
// and the p(x)/q(x) accept ratio, which at temp>0 is a measurable distribution bias. `logits` stays
// DESTRUCTIVE (masked in place); verify_forward_sample releases the buffer immediately, as before.
extern "C" __global__ void spec_verify_f32_b(
    float* p_of_draft, int* resid_tok,
    float* logits, const int* draft_tokens, const float* draft_qprobs,
    const float* temps, const int* top_ks, const float* top_ps, const unsigned int* seeds,
    int v, int depth) {
    int j = blockIdx.x;
    if (j >= depth) return;
    int tid = threadIdx.x;
    int nthr = blockDim.x;
    float* col = logits + (long long)j * v;
    float temp = temps[j];
    int topk = top_ks[j];
    if (topk < 1) topk = 1;
    if (topk > SAMPLE_K_MAX) topk = SAMPLE_K_MAX;
    float topp = top_ps[j];

    extern __shared__ float sh[];
    float* topv = sh;
    int*   topi = (int*)(sh + SAMPLE_K_MAX);
    float* red  = sh + SAMPLE_K_MAX + SAMPLE_K_MAX;

    // Greedy path: target is a point mass at the argmax.
    if (temp < 1e-6f) {
        float lmx = -1e30f; int lidx = 0;
        for (int k = tid; k < v; k += nthr) { float val = col[k]; if (val > lmx) { lmx = val; lidx = k; } }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            int argmax_tok = (int)red[1];
            if (j < depth - 1) {
                p_of_draft[j] = (argmax_tok == draft_tokens[j]) ? 1.0f : 0.0f;
            }
            resid_tok[j] = argmax_tok;   // j == depth-1 => the bonus slot
        }
        return;
    }

    // Stochastic path: top-k → softmax → top-p nucleus. Selection is identical to sample_f32_b's,
    // in-place mask included, so the two kernels agree on the nucleus by construction.
    float inv_temp = 1.0f / temp;
    for (int s = 0; s < topk; s++) {
        float lmx = -1e30f; int lidx = -1;
        for (int k = tid; k < v; k += nthr) {
            float val = col[k]; if (val > lmx) { lmx = val; lidx = k; }
        }
        red[2*tid] = lmx; red[2*tid+1] = (float)lidx;
        __syncthreads();
        for (int s2 = nthr/2; s2 > 0; s2 >>= 1) {
            if (tid < s2 && red[2*(tid+s2)] > red[2*tid]) { red[2*tid] = red[2*(tid+s2)]; red[2*tid+1] = red[2*(tid+s2)+1]; }
            __syncthreads();
        }
        if (tid == 0) {
            topv[s] = red[0] * inv_temp;
            topi[s] = (int)red[1];
            col[(int)red[1]] = -1e30f;   // mask so the next pass skips it
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = topv[0];
        float probs[SAMPLE_K_MAX];
        float sum = 0.0f;
        for (int s = 0; s < topk; s++) { probs[s] = __expf(topv[s] - mx); sum += probs[s]; }
        float cum = 0.0f; int nc = topk - 1;
        for (int s = 0; s < topk; s++) { cum += probs[s]; if (cum >= topp * sum) { nc = s; break; } }

        if (j < depth - 1) {
            int draft_tok = draft_tokens[j];
            float p_draft = 0.0f;
            int draft_idx = -1;
            for (int s = 0; s <= nc; s++) {
                if (topi[s] == draft_tok) { draft_idx = s; p_draft = probs[s] / cum; break; }
            }
            p_of_draft[j] = p_draft;

            // Resample from residual (p \ {draft}) — pragmatic variant: zero out draft's mass.
            unsigned int sr = seeds[j];
            sr = sr * 1664525u + 1013904223u;
            float ru = (sr >> 8) * (1.0f / 16777216.0f);

            float resid_cum = (draft_idx >= 0) ? cum - probs[draft_idx] : cum;
            if (resid_cum <= 0.0f) {
                // All nucleus mass on the draft token → fall back to full distribution.
                resid_cum = cum; ru *= cum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) { acc += probs[s]; if (ru < acc) { chosen = topi[s]; break; } }
                resid_tok[j] = chosen;
            } else {
                ru *= resid_cum;
                float acc = 0.0f; int chosen = topi[nc];
                for (int s = 0; s <= nc; s++) {
                    if (s == draft_idx) continue;
                    acc += probs[s];
                    if (ru < acc) { chosen = topi[s]; break; }
                }
                resid_tok[j] = chosen;
            }
        } else {
            // Bonus column: sample from full target p_{depth-1}.
            unsigned int sb = seeds[j];
            sb = sb * 1664525u + 1013904223u;
            float rb = (sb >> 8) * (1.0f / 16777216.0f);
            rb *= cum;
            float acc = 0.0f; int chosen = topi[nc];
            for (int s = 0; s <= nc; s++) { acc += probs[s]; if (rb < acc) { chosen = topi[s]; break; } }
            resid_tok[j] = chosen;   // j == depth-1 => the bonus slot
        }
    }
}

// ===================== QUALITY EVALUATION =====================
// Per-position negative log-likelihood of a target token: nll = -(logit[t] - max - log(sum exp)).
// One block per position; the full-vocab softmax stays on device so the [vocab, N] logits never
// have to cross to the host (at 248k vocab that would be ~0.5 GB per window).
extern "C" __global__ void nll_b(float* out, const __nv_bfloat16* logits, const int* targets,
                                 int v, int n) {
    int j = blockIdx.x;
    if (j >= n) return;
    int tid = threadIdx.x, nthr = blockDim.x;
    const __nv_bfloat16* col = logits + (long long)j * v;
    extern __shared__ float red[];

    float mx = -1e30f;
    for (int k = tid; k < v; k += nthr) mx = fmaxf(mx, b2f(col[k]));
    red[tid] = mx;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float sm = 0.0f;
    for (int k = tid; k < v; k += nthr) sm += __expf(b2f(col[k]) - mx);
    red[tid] = sm;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        int t = targets[j];
        float lt = b2f(col[t]);
        out[j] = -(lt - mx - __logf(red[0]));
    }
}

// ===================== FUSED DEQUANT GEMV (the quantized decode critical path) =====================
//
// These are `gemm_binv_b` with the weight read swapped for packed 4-bit / 8-bit + inline dequant.
// Decode is a bandwidth-bound GEMV (N <= 16, zero arithmetic intensity), so CUDA cores dequantizing
// in registers hit the same roofline tensor cores would — the FP4 tensor cores are a prefill lever,
// not a decode one, and are deliberately not used here.
//
// EVERY constraint that made gemm_binv_b work carries over, and two bite harder:
//
//  1. BATCH-INVARIANT. One block per output row m; each thread owns a fixed strided slice of K;
//     fixed-shape tree reduce. N must never change the reduction order, or column 0 stops matching a
//     single-token decode and greedy-MTP losslessness silently dies. Dequant is a pure per-element
//     function, so it cannot break this — provided no N-dependent tiling is added.
//
//  2. COMPILE-TIME N, or ptxas spills `acc[]` to LOCAL memory and the entire bandwidth win is lost.
//     This is the bug that cost this project the most (see AGENTS.md §4.1): a runtime-indexed
//     per-thread array cannot be register-allocated, and the resulting load/store traffic scales with
//     N while the weight read does not. The nibble-unpack loop must also be fully unrolled.
//     GATE: `-Xptxas -v` must report `0 bytes stack frame` for both kernels.

// E2M1 decode, ARITHMETIC — deliberately not a __constant__ lookup table.
//
// A 16-entry __constant__ LUT is the obvious way to write this and it is a trap: constant memory is
// optimized for BROADCAST (all threads in a warp reading the same address). Here every thread decodes
// a different nibble, so the access diverges and the constant cache serializes it into one
// transaction per distinct address — up to 8-way, on the hottest line of the kernel. Measured: the
// LUT version delivered only 1.15x over bf16 where the bytes promised ~3x.
//
// A per-thread `const float lut[8]` indexed dynamically is worse still: it cannot be
// register-allocated and lands in LOCAL memory (see AGENTS.md §4.1).
//
// So: build the value from the bits. E2M1 is s|ee|m with magnitudes {0,.5,1,1.5,2,3,4,6}:
//   e == 0 -> 0.5*m                       (0, 0.5)
//   e >  0 -> (1 + 0.5*m) * 2^(e-1)       (1,1.5 | 2,3 | 4,6)
__device__ __forceinline__ float e2m1_f(uint8_t c) {
    unsigned e    = (c >> 1) & 0x3u;
    unsigned m    = c & 0x1u;
    unsigned sign = (unsigned)(c & 0x8) << 28;
    // e>0: f32 exponent = e + 126 (E2M1 bias 1 -> f32 bias 127), mantissa bit -> bit 22.
    //   e=1: 1.0/1.5   e=2: 2.0/3.0   e=3: 4.0/6.0
    // e=0: 0 or 0.5 (0x3F000000).
    unsigned bits = e ? (sign | ((e + 126u) << 23) | (m << 22))
                      : (sign | (m ? 0x3F000000u : 0u));
    return __uint_as_float(bits);
}

// E4M3 -> f32 by BIT SURGERY, not arithmetic.
//
// E4M3 is a float (never integer-cast it — that was the old prototype's core bug), but the naive
// decode reaches for exp2f/powf, and a transcendental on the hot path is brutal: it is what made the
// byte-granular mapping (which decodes the scale 8x more often) *slower* than a worse-utilised one.
// Both formats are IEEE-shaped, so the conversion is just a re-lay of the exponent and mantissa
// fields: E4M3 has bias 7 and 3 mantissa bits, f32 has bias 127 and 23.
__device__ __forceinline__ float e4m3_f(uint8_t b) {
    unsigned sign = (unsigned)(b & 0x80) << 24;
    int      e    = (b >> 3) & 0x0F;
    unsigned m    = (unsigned)(b & 0x07);
    if (e == 0) {                                   // subnormal: m * 2^-9
        float v = (float)m * 0.001953125f;          // 2^-9
        return (b & 0x80) ? -v : v;
    }
    unsigned bits = sign | ((unsigned)(e - 7 + 127) << 23) | (m << 20);
    return __uint_as_float(bits);
}

// UE8M0 -> f32: 2^(b-127), an exact power of two (dsv4_load::e8m0_to_f32's device twin).
// Plain `b<<23` covers b in [1,254]; b=0 means 2^-127, an f32 SUBNORMAL (0x00400000), not 0.0.
// 0xFF is NaN in the spec — the host loader asserts it never appears; here it decodes to +inf.
__device__ __forceinline__ float e8m0_f(uint8_t b) {
    return b ? __uint_as_float((unsigned)b << 23) : 0x1p-127f;
}

// e4m3 encoder (round-to-nearest-even, saturate to ±448, the exact inverse of e4m3_f for every
// representable value — pack/unpack of a previously-representable input is identity).
__device__ __forceinline__ uint8_t f32_to_e4m3(float f) {
    const unsigned b = __float_as_uint(f);
    const unsigned sgn = (b >> 24) & 0x80;
    const float a = fabsf(f);
    if (a != a) return (uint8_t)(sgn | 0x7F);        // NaN -> max NaN pattern
    if (a >= 448.0f) return (uint8_t)(sgn | 0x7E);   // saturate to max 448
    if (a == 0.0f) return (uint8_t)sgn;
    // normal or subnormal: find exponent/mantissa with round-to-nearest-even on 3 mantissa bits.
    int e;
    const float m = frexpf(a, &e);                   // a = m * 2^e, m in [0.5, 1)
    // e4m3 normal range: value = (1 + m3/8) * 2^(E-7), E in [1, 15]; subnormal: m3 * 2^-9.
    if (e - 1 < -6) {                                // subnormal territory: quantum = 2^-9
        int m3 = (int)lrintf(a * 512.0f);            // / 2^-9
        if (m3 > 7) m3 = 7;                          // 8 * 2^-9 = 2^-6 rounds up to the smallest normal
        return (uint8_t)(sgn | m3);
    }
    int E = e - 1 + 7;                               // value = (1+frac) * 2^(e-1), E = e-1+7
    float frac = m * 2.0f - 1.0f;                    // in [0, 1)
    int m3 = (int)lrintf(frac * 8.0f);
    if (m3 == 8) { m3 = 0; E += 1; }                 // mantissa overflow -> next exponent
    if (E > 15) return (uint8_t)(sgn | 0x7E);
    return (uint8_t)(sgn | (E << 3) | m3);
}


