// exl3_bench/06_router_fold_r16.cuh: S-B9-CF-R16 routing fold at 9..16 rows, MoE route, grouped GEMM entries.
// Same per-row warp-local chain as xq_router_fold_c (bitwise).
// Section 6 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// =============================================================================
// S-B9-CF-R16 (PLAN/CF_R16_REPORT.md): the routing fold at 9..16 rows.
// xq_router_fold_c_r16 == xq_router_fold_c (A5-K2 COAL) with (a) the smem partial
// table 16 rows deep and (b) the tail covering two rows per warp (warp w serves
// rows w and w+8 with the same warp-local chain), so a forward with m = 9..16
// rows takes the ONE-launch fold instead of falling back to
// xq_router_ks + xq_router_combine + xq_router_topk + xq_moe_route (4 launches;
// 4.8 ms at m = 10 against 1.3 ms for the fold at m = 8).
// The launch geometry is the m<=8 fold's exactly — grid (ne/4) x 256 threads — so
// the GEMV (warps 0..7 are the whole block) and the grid-barrier election are the
// m<=8 code verbatim. BIT-IDENTITY vs the m<=8 fold and vs the legacy chain:
//   logits  — the same per-thread 40-wide slice chain (float4 groups of 8,
//             ascending) and the same one-thread ascending s = 0..63 fold; only
//             the partial table depth (16) and the x load placement change, and
//             neither is math.
//   softmax — the per-row chain is the SAME warp-local replay of xq_router_topk's
//             256-thread code (lane l holds the virtual threads tid = l + 32q over
//             elements i = tid + 256p; the halving tree replays in the same order
//             with the same (low, high) operand order), run twice per warp.
//   top-k   — the COAL lane tree (4 levels) + redux.sync, unchanged; (prob desc,
//             id asc) is a total order, so the pick sequence is unique.
//   renorm  — sum = ((0 + w0) + w1) + ... ascending j, w_j / sum, unchanged.
//   route   — first-seen over ids row-major: s_first = smem atomicMin, slot =
//             #{first positions < first(e)} (per-warp popc prefix over the 8 warp
//             ballots; BK = m*k <= 160 spans at most 5 warps and a warp with no
//             flags contributes 0), the same serial-counter result in the same
//             row-major order.
// m <= 8 dispatches to the untouched xq_router_fold_c; this entry is only ever
// launched for m > 8 (host gate; --exl3-r16=0 restores today's dispatch).
// Dynamic smem: 8 warps x 4 experts x 40 float4 = XQ_RFC_SMEM_R16 (20 KB, the
// m<=8 fold's exact carve-out — no opt-in needed). Static: part[16 * 289] (18 KB).
// =============================================================================
#define XQ_RFC_SMEM_R16 (8 * 4 * 40 * 16)
template <bool REMAP, bool COAL = false, bool EP = false>
__device__ __forceinline__ void xq_router_fold_body_r16(
        float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
        int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
        int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
        unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
        unsigned long long gu_words, unsigned long long d_words, const int* __restrict__ ep_lut = nullptr) {
    constexpr int MAXM = 16;   // S-B9-CF-R16 row bound (the m<=8 body's MAXM is 8)
    constexpr int PS = REMAP ? XQ_RFD_PS : XQ_RF_KS;
    constexpr int RS = REMAP ? XQ_RFD_RS : XQ_RF_EPB * XQ_RF_KS;
    static_assert(MAXM * RS >= XQ_RFD_NEMAX + 2 * MAXM * XQ_RTR_MAXK + 8, "tail smem alias");
    __shared__ float part[MAXM * RS];
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int el = REMAP ? (lane >> 3) : (tid / XQ_RF_KS);
    const int s = REMAP ? ((warp << 3) | (lane & 7)) : (tid % XQ_RF_KS);
    const int n = blockIdx.x * XQ_RF_EPB + el;
    // ---- GEMV: xq_router_fused's per-thread body, verbatim ----
    const int slice = K / XQ_RF_KS;          // host: K % 64 == 0 && slice % 8 == 0
    const int k0 = s * slice;
    const __half* wr = w + (size_t)n * K + k0;
    float acc[MAXM];
    #pragma unroll
    for (int r = 0; r < MAXM; r++) acc[r] = 0.0f;
    if constexpr (COAL) {
        static_assert(REMAP, "COAL uses the REMAP lane map (warp = 4 experts x 8 slices)");
        extern __shared__ __align__(16) float4 xq_rfc_ws[];          // [8 warps][4 experts][40]
        if (slice == 40) {                                           // host: always (K == 2560)
            float4* ws = xq_rfc_ws + warp * (XQ_RF_EPB * 40);
            #pragma unroll
            for (int q = 0; q < 5; ++q) {
                const int f = (q << 5) | lane, ef = f / 40, c = f - ef * 40;
                const int nf = blockIdx.x * XQ_RF_EPB + ef;
                if (nf < N)
                    lmh_cp16((unsigned)__cvta_generic_to_shared(ws + f),
                             w + (size_t)nf * K + warp * (8 * 40) + (c << 3));
            }
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 0;\n" ::: "memory");
            __syncwarp();
            if (n < N) {
                float4 wv5[5];
                #pragma unroll
                for (int q = 0; q < 5; ++q) wv5[q] = ws[el * 40 + (lane & 7) * 5 + q];
                #pragma unroll
                for (int q = 0; q < 5; ++q) {
                    const __half* wh = (const __half*)&wv5[q];
                    // r16: the x load is per row (xv[16] would pin 64 registers);
                    // loads are not math — the per-(row, q, j) FFMA order is the same.
                    #pragma unroll
                    for (int r = 0; r < MAXM; r++) {
                        if (r < M) {
                            float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                            const __half* xh_ = (const __half*)&xv;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                        }
                    }
                }
            }
        }
    } else {
    if (n < N && slice == 40) {
        float4 wv5[5];
        #pragma unroll
        for (int q = 0; q < 5; ++q) wv5[q] = *(const float4*)(wr + (q << 3));
        #pragma unroll
        for (int q = 0; q < 5; ++q) {
            const __half* wh = (const __half*)&wv5[q];
            #pragma unroll
            for (int r = 0; r < MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
    } else if (n < N) {
        const int s8 = slice >> 3;
        for (int q = 0; q < s8; ++q) {
            float4 wv = *(const float4*)(wr + (q << 3));
            const __half* wh = (const __half*)&wv;
            #pragma unroll
            for (int r = 0; r < MAXM; r++) {
                if (r < M) {
                    float4 xv = *(const float4*)(x + (size_t)r * K + k0 + (q << 3));
                    const __half* xh_ = (const __half*)&xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += __half2float(wh[j]) * __half2float(xh_[j]);
                }
            }
        }
        for (int kk = (s8 << 3); kk < slice; ++kk) {
            #pragma unroll
            for (int r = 0; r < MAXM; r++)
                if (r < M) acc[r] += __half2float(wr[kk]) * __half2float(x[(size_t)r * K + k0 + kk]);
        }
    }
    }   // !COAL
    #pragma unroll
    for (int r = 0; r < MAXM; r++) if (r < M) part[r * RS + el * PS + s] = acc[r];
    __syncthreads();
    if (tid < M * XQ_RF_EPB) {
        const int r = tid / XQ_RF_EPB, e = tid % XQ_RF_EPB;
        const int nn = blockIdx.x * XQ_RF_EPB + e;
        if (nn < N) {
            float a = 0.0f;
            for (int q = 0; q < XQ_RF_KS; ++q) a += part[r * RS + e * PS + q];
            out[(size_t)r * N + nn] = a;
        }
    }
    // ---- publish + elect the last block (every block's logits visible before its ticket) ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int last = (atomicAdd(done_cnt, 1) == (int)gridDim.x - 1);
        if (last) __threadfence();
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    // ---- tail (last block only). `part` is dead: reuse it for the route scratch ----
    int* s_first = reinterpret_cast<int*>(part);                 // [ne]  first position of expert e
    int* s_ids = s_first + XQ_RFD_NEMAX;                         // [M*k] picked ids, row-major
    int* s_slot = s_ids + MAXM * XQ_RTR_MAXK;              // [M*k] slot of a first position
    int* s_wc = s_slot + MAXM * XQ_RTR_MAXK;               // [8]   per-warp first counts
    for (int e = tid; e < N; e += blockDim.x) s_first[e] = 0x7fffffff;
    // S-B9-CF-R16: 8 warps cover 16 rows — warp w runs the SAME warp-local chain
    // for rows w and w+8 (sequential replays; per-row bits identical to the
    // m<=8 tail's warp r <-> row r).
    for (int rr = 0; rr < 2; ++rr) {
      const int r = warp + (rr << 3);
      if (r < M) {
          const float* lg = out + (size_t)r * N;
          float v[16];
          #pragma unroll
          for (int t = 0; t < 16; ++t) { const int i = lane + 32 * t; v[t] = (i < N) ? __ldcg(lg + i) : 0.0f; }
          // max: virtual thread q folds i = l+32q, then l+32q+256 (ascending), then the 256-tree.
          float mq[8];
          #pragma unroll
          for (int q = 0; q < 8; ++q) {
              mq[q] = -INFINITY;
              if (lane + 32 * q < N) mq[q] = fmaxf(mq[q], v[q]);
              if (lane + 32 * q + 256 < N) mq[q] = fmaxf(mq[q], v[q + 8]);
          }
          #pragma unroll
          for (int q = 0; q < 4; ++q) mq[q] = fmaxf(mq[q], mq[q + 4]);           // s2 = 128
          #pragma unroll
          for (int q = 0; q < 2; ++q) mq[q] = fmaxf(mq[q], mq[q + 2]);           // s2 = 64
          float mx = fmaxf(mq[0], mq[1]);                                          // s2 = 32
          #pragma unroll
          for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
          const float gmax = __shfl_sync(0xffffffffu, mx, 0);
          // denominator: the same per-virtual-thread partials and the same halving tree.
          float ev[16];
          #pragma unroll
          for (int t = 0; t < 16; ++t) ev[t] = __expf(v[t] - gmax);
          float pq[8];
          #pragma unroll
          for (int q = 0; q < 8; ++q) {
              pq[q] = 0.0f;
              if (lane + 32 * q < N) pq[q] += ev[q];
              if (lane + 32 * q + 256 < N) pq[q] += ev[q + 8];
          }
          #pragma unroll
          for (int q = 0; q < 4; ++q) pq[q] += pq[q + 4];                          // s2 = 128
          #pragma unroll
          for (int q = 0; q < 2; ++q) pq[q] += pq[q + 2];                          // s2 = 64
          float sm_ = pq[0] + pq[1];                                               // s2 = 32
          #pragma unroll
          for (int o = 16; o > 0; o >>= 1) sm_ += __shfl_down_sync(0xffffffffu, sm_, o);
          const float denom = __shfl_sync(0xffffffffu, sm_, 0);
          float p[16];
          #pragma unroll
          for (int t = 0; t < 16; ++t) p[t] = (lane + 32 * t < N) ? ev[t] / denom : -4.0f;  // -4: never picked
          // top-k under (prob desc, id asc); lane j keeps pick j; every lane folds the same sum.
          float myw = 0.0f, sum = 0.0f;
          int myid = 0;
          for (int j = 0; j < k; ++j) {
              float bv = -2.0f; int bi = N;
              if constexpr (COAL) {
                  // A5-K2: the same pick, shorter chains. (prob desc, id asc) is a total order on the
                  // non-NaN probs and every candidate here is a prob in [0, 1] (ev <= 1 after the max
                  // shift; NaN / picked -3 / padding -4 are never picked: they never beat the (-2, N)
                  // seed). So: map non-candidates (!(p >= 0)) to -5, take the lane's best by a 4-level
                  // tree (ids ascend with t, so a pair keeps its lower id unless the higher one is
                  // strictly greater), then the warp's best as (max key, min id among max-key lanes)
                  // with redux.sync, key = bits(p) + 1 (order-preserving for p >= 0; 0 = no
                  // candidate). bv carries the exact prob bits back. == the linear scan + butterfly.
                  float tv[16]; int ti[16];
                  #pragma unroll
                  for (int t = 0; t < 16; ++t) { tv[t] = (p[t] >= 0.0f) ? p[t] : -5.0f; ti[t] = lane + 32 * t; }
                  #pragma unroll
                  for (int st = 1; st < 16; st <<= 1) {
                      #pragma unroll
                      for (int t = 0; t < 16; t += 2 * st)
                          if (tv[t + st] > tv[t]) { tv[t] = tv[t + st]; ti[t] = ti[t + st]; }
                  }
                  const unsigned key = (tv[0] >= 0.0f) ? __float_as_uint(tv[0]) + 1u : 0u;
                  const unsigned kmax = __reduce_max_sync(0xffffffffu, key);
                  const unsigned imin = __reduce_min_sync(0xffffffffu, (key == kmax) ? (unsigned)ti[0] : 0xffffffffu);
                  if (kmax != 0u) { bv = __uint_as_float(kmax - 1u); bi = (int)imin; }
              } else {
              #pragma unroll
              for (int t = 0; t < 16; ++t) {
                  const int i = lane + 32 * t;
                  if (p[t] > bv || (p[t] == bv && i < bi)) { bv = p[t]; bi = i; }
              }
              #pragma unroll
              for (int o = 16; o > 0; o >>= 1) {
                  const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                  const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                  if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
              }
              }   // !COAL
              #pragma unroll
              for (int t = 0; t < 16; ++t) if (lane + 32 * t == bi) p[t] = -3.0f;
              const int id = (bi < 0 || bi >= N) ? 0 : bi;   // topk_route's NaN/Inf clamp
              if (lane == j) { myw = bv; myid = id; }
              sum += bv;
          }
        if (lane < k) {
            ids[(size_t)r * k + lane] = myid;
            wts[(size_t)r * k + lane] = myw / sum;
            s_ids[r * k + lane] = myid;
        }
      }
    }
    __syncthreads();
    // ---- first-seen route (xq_moe_route semantics) ----
    const int BK = M * k;                                        // host: <= 160 <= blockDim (r16)
    if (tid < BK) { const int e = s_ids[tid]; if (e >= 0 && e < N) atomicMin(&s_first[e], tid); }
    __syncthreads();
    int e_t = -1; bool first = false;
    if (tid < BK) { e_t = s_ids[tid]; first = (e_t >= 0 && e_t < N) && s_first[e_t] == tid; }
    int le_t = e_t;                                              // the stacked index idxmap/offs carry
    if constexpr (EP) {
        if (first) { le_t = __ldg(ep_lut + e_t); first = le_t >= 0; }   // a remote expert takes no slot
    }
    const unsigned bal = __ballot_sync(0xffffffffu, first);
    if (lane == 0) s_wc[warp] = __popc(bal);
    __syncthreads();
    int base = 0, total = 0;
    #pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) { const int c = s_wc[w8]; total += c; if (w8 < warp) base += c; }
    if (first) {
        const int slot = base + __popc(bal & ((1u << lane) - 1u));
        s_slot[tid] = slot;
        idxmap[slot] = le_t;
        offs_gu[slot] = (unsigned long long)le_t * gu_words;
        offs_d[slot] = (unsigned long long)le_t * d_words;
    }
    if (tid == 0) { esel_dev[0] = total; *done_cnt = 0; }   // re-arm for the next launch / graph replay
    __syncthreads();
    for (int e = tid; e < N; e += blockDim.x) {
        const int f = s_first[e];
        bool live = f != 0x7fffffff;
        if constexpr (EP) live = live && __ldg(ep_lut + e) >= 0;       // s_slot[f] is unwritten for a remote e
        slotmap[e] = live ? s_slot[f] : -1;
    }
}

extern "C" __global__ void __launch_bounds__(256, 3)   // 3 blocks/SM: all ne/4 = 128 blocks in ONE wave
xq_router_fold(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
               int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
               int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
               unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
               unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                              offs_gu, offs_d, gu_words, d_words);
}
// A5-K2: the coalesced-weight twin (dynamic smem 8 x 2560 B = XQ_RFC_SMEM; host: K == 2560).
#define XQ_RFC_SMEM (8 * XQ_RF_EPB * 40 * 16)
extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_c(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                 int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                 int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                 unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                 unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true, true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                                    offs_gu, offs_d, gu_words, d_words);
}
// TP-G (T2 lever #5): xq_router_fold_c with xq_moe_route_ep folded into its route tail (EP ranks).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_c_ep(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                    int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                    int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                    unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                    unsigned long long gu_words, unsigned long long d_words, const int* __restrict__ ep_lut) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<true, true, true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                                          offs_gu, offs_d, gu_words, d_words, ep_lut);
}

// S-B9-CF-R16: the 9..16-row COAL routing fold (see xq_router_fold_body_r16).
extern "C" __global__ void __launch_bounds__(256, 2)   // 39 KB smem/CTA: 2 CTAs/SM
xq_router_fold_c_r16(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                     int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                     int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                     unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                     unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body_r16<true, true>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                                        offs_gu, offs_d, gu_words, d_words);
}


extern "C" __global__ void __launch_bounds__(256, 3)
xq_router_fold_m0(float* out, const __half* __restrict__ w, const __half* __restrict__ x, int M, int N, int K,
                  int* __restrict__ done_cnt, int* __restrict__ ids, float* __restrict__ wts, int k,
                  int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel_dev,
                  unsigned long long* __restrict__ offs_gu, unsigned long long* __restrict__ offs_d,
                  unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    xq_router_fold_body<false>(out, w, x, M, N, K, done_cnt, ids, wts, k, slotmap, idxmap, esel_dev,
                               offs_gu, offs_d, gu_words, d_words);
}

// ---- S-A3-f Item 2: device-side MoE routing. Replaces the per-layer host round-trip
// (sync + dtoh ids + host first-seen dedup + 4 uploads) with ONE kernel launch.
// Semantics replicate the old host scan EXACTLY: slots are assigned first-seen over
// ids[0..m*topk] row-major; offs_* are the compacted trellis word offsets; esel_dev[0]
// is the live-expert count the padded consumer grids early-exit on. Grid (1,1,1).
extern "C" __global__ void xq_moe_route(const int* __restrict__ ids, int m, int topk, int ne,
                                        int* __restrict__ slotmap, int* __restrict__ idxmap,
                                        int* __restrict__ esel_dev,
                                        unsigned long long* __restrict__ offs_gu,
                                        unsigned long long* __restrict__ offs_d,
                                        unsigned long long gu_words, unsigned long long d_words) {
    XQ_PDL_ENTRY();
    __shared__ int cnt;
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < m * topk; t++) {
            int e = ids[t];
            if (e >= 0 && e < ne && slotmap[e] == -1) {
                slotmap[e] = c;
                idxmap[c] = e;
                c++;
            }
        }
        cnt = c;
        esel_dev[0] = c;
    }
    __syncthreads();
    for (int s = tid; s < cnt; s += blockDim.x) {
        int e = idxmap[s];
        offs_gu[s] = (unsigned long long)e * gu_words;
        offs_d[s] = (unsigned long long)e * d_words;
    }
}

// S-A3-s diagnostic: scatter per-row QSA selection lists into a position-indexed buffer
// (dst_sel[pos][sel_max], dst_psel[pos]) — grid rows, block 256; rows with pos >= cap skipped.
extern "C" __global__ void xq_dbg_scatter_sel(int* __restrict__ dst_sel, int* __restrict__ dst_psel,
                                              const int* __restrict__ src_sel, const int* __restrict__ src_psel,
                                              const int* __restrict__ pos, int sel_max, int cap) {
    XQ_PDL_ENTRY();
    const int r = blockIdx.x;
    const int p = pos[r];
    if (p < 0 || p >= cap) return;
    for (int i = threadIdx.x; i < sel_max; i += blockDim.x)
        dst_sel[(long long)p * sel_max + i] = src_sel[(long long)r * sel_max + i];
    if (threadIdx.x == 0) dst_psel[p] = src_psel[r];
}

// S-A3-o diagnostic: hist[m*w + min(esel, w-1)] += 1 (GB10_EXL3_ESEL_HIST only).
extern "C" __global__ void xq_esel_hist(const int* __restrict__ esel, int* __restrict__ hist, int m, int w) {
    XQ_PDL_ENTRY();
    int e = *esel;
    atomicAdd(&hist[m * w + (e < w - 1 ? e : w - 1)], 1);
}

// CF-P1e step 0 diagnostic (--exl3-route-log): append one record per decode/verify MoE call to a device ring:
// [layer, m, ids[0..m*topk] padded with -1 to 16*16 slots]. Graph-safe (fixed pointers, an atomic counter);
// dumped and reset at every request finish. Diagnostic only.
#define XQ_ROUTE_REC (2 + 16 * 16)
extern "C" __global__ void xq_route_log(const int* __restrict__ ids, int m, int topk, int layer,
                                        int* __restrict__ buf, int cap, unsigned* __restrict__ ctr) {
    XQ_PDL_ENTRY();
    __shared__ unsigned r;
    if (threadIdx.x == 0) r = atomicAdd(ctr, 1u);
    __syncthreads();
    if (r >= (unsigned)cap) return;
    int* rec = buf + (size_t)r * XQ_ROUTE_REC;
    if (threadIdx.x == 0) { rec[0] = layer; rec[1] = m; }
    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) rec[2 + t] = t < m * topk ? ids[t] : -1;
}

// TP-I #6 diagnostic (GB10_TP_EP_HIST): per-(layer, expert) routing histogram of every decode /
// verify MoE call (m <= 16). hist[e] += 1 once per call when expert e is live in the call (the
// weight-streaming unit of the expert GEMMs), hist[ne + e] += its row picks. Diagnostic only.
extern "C" __global__ void xq_expert_hist(const int* __restrict__ ids, int m, int topk, int ne, int* __restrict__ hist) {
    XQ_PDL_ENTRY();
    __shared__ unsigned char seen[1024];
    const int tid = threadIdx.x;
    for (int e = tid; e < ne && e < 1024; e += blockDim.x) seen[e] = 0;
    __syncthreads();
    for (int t = tid; t < m * topk; t += blockDim.x) {
        int e = ids[t];
        if (e >= 0 && e < ne && e < 1024) { seen[e] = 1; atomicAdd(&hist[ne + e], 1); }
    }
    __syncthreads();
    for (int e = tid; e < ne && e < 1024; e += blockDim.x) if (seen[e]) atomicAdd(&hist[e], 1);
}

// ---- per-expert output Hadamard: y[e,m,:] = H(yraw[e,m,:]) * svh_all[idxmap[e],:]).
// S-A3-f Item 2: grids are PADDED to the topk cap; blocks of absent experts exit on
// the device-routed count before touching memory.
extern "C" __global__ void xq_had_svh_multi(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                            const int* __restrict__ idxmap,
                                            __half* __restrict__ y, int M, int N,
                                            const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    int nblk = N >> 7;
    int blk = blockIdx.x % nblk;
    long long em = blockIdx.x / nblk;
    int m = (int)(em % M);
    int e = (int)(em / M);
    if (e >= *esel) return;
    int lane = threadIdx.x & 31;
    long long base = ((long long)e * M + m) * N + (long long)blk * 128 + lane * 4;
    const __half* sr = svh_all + (long long)idxmap[e] * N + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = xq_h2f(yraw[base + r]);
    exl3_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) y[base + r] = xq_f2h(v[r] * xq_h2f(sr[r]));
}

// ---- grouped trellis GEMM with PER-EXPERT xh ([E][M][K]). Same tile contract as
// exl3_hmma_gemm_grouped; only the A-operand indexing changes.
// S-A3-f Item 2: grid padded to the topk cap; absent experts exit on the device count.
extern "C" __global__ void xq_gemm_grouped_xh(const uint16_t* __restrict__ base,
                                              const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh,
                                              __half* __restrict__ y,
                                              int M, int K, int N, int bits,
                                              const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    __half* ye = y + (long long)expert * M * N;
    // K35 (2026-10-31), after LEG-B 2026-10-30: the fractional widths K = 1.5 / 2.5 / 3.5 arrive as the
    // EXL3_FRAC_BITS(KA) code (17 / 18 / 19; the host's MoeLayer::bits). The integer a1/xh bodies below
    // cannot express the half-width ring (W16 = 8*(2*KA+1) u16 per tile), and a `default: break` here
    // was a REACHABLE SILENT NO-OP once (LEG-B: k2=3 experts wrote NOTHING from L4 on, sentinel proof
    // 5120/5120, tests/k23_golden_l4e7up.json). The HALF twins: A-once (dyn smem M*(K+8)*2, the
    // launcher passes it exactly as for the integer widths) when it fits, else the barrier body (which
    // stages A in the static exl3_gemm_body_sa; the dyn smem is simply unused).
#define XQ_GXH_FRAC(KA) case EXL3_FRAC_BITS(KA): \
        if (exl3_a1_fits(M, K)) exl3_gemm_body_a1<KA, true>(tr, xe, ye, M, K, N, tile); \
        else                    exl3_gemm_body<KA, true>(tr, xe, ye, M, K, N, tile); \
        return;
    switch (bits) { XQ_GXH_FRAC(1) XQ_GXH_FRAC(2) XQ_GXH_FRAC(3) default: break; }
#undef XQ_GXH_FRAC
    // S-A3-n F3: A staged once when it fits (the launcher passes M*(K+8)*2 dyn smem).
    if (exl3_a1_fits(M, K)) {
        // P7 B3 leg 2: bits 2/7 join the switch — the old default: break was a REACHABLE silent
        // no-op (the host K-generic fallback routes 2/7-bit packs here; the [K=2] rung-1 gate
        // caught it as zeros-vs-a1b2, with the down GEMM a 0-vs-0 empty pass).
        switch (bits) {
            case 2: exl3_gemm_body_a1<2>(tr, xe, ye, M, K, N, tile); break;
            case 3: exl3_gemm_body_a1<3>(tr, xe, ye, M, K, N, tile); break;
            case 4: exl3_gemm_body_a1<4>(tr, xe, ye, M, K, N, tile); break;
            case 5: exl3_gemm_body_a1<5>(tr, xe, ye, M, K, N, tile); break;
            case 6: exl3_gemm_body_a1<6>(tr, xe, ye, M, K, N, tile); break;
            case 7: exl3_gemm_body_a1<7>(tr, xe, ye, M, K, N, tile); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 2: exl3_gemm_body<2>(tr, xe, ye, M, K, N, tile); break;
        case 3: exl3_gemm_body<3>(tr, xe, ye, M, K, N, tile); break;
        case 4: exl3_gemm_body<4>(tr, xe, ye, M, K, N, tile); break;
        case 5: exl3_gemm_body<5>(tr, xe, ye, M, K, N, tile); break;
        case 6: exl3_gemm_body<6>(tr, xe, ye, M, K, N, tile); break;
        case 7: exl3_gemm_body<7>(tr, xe, ye, M, K, N, tile); break;
        default: break;
    }
}

template <int BITS>
__device__ __forceinline__ void exl3_gemm_body_a1_fh(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ x_in,
                                                  const __half* __restrict__ suh,
                                                  const __half* __restrict__ svh,
                                                  __half* __restrict__ y,
                                                  int M, int K, int N, int cta_tile) {
    extern __shared__ __half a1_sa[];        // [M][K+8]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int SK = K + 8;
    {
        // fused xq_had_suh_multi: A = f16(H128(x * suh)) per (row, 128-block), the SAME loads,
        // product, exl3_had128 butterfly and fp16 rounding => the bytes xq_had_suh_multi wrote.
        const int nblk = K >> 7;
        for (int bi = warp; bi < M * nblk; bi += (blockDim.x >> 5)) {
            const int r = bi / nblk, blk = bi - r * nblk;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                v[j] = xq_h2f(x_in[(size_t)r * K + blk * 128 + lane * 4 + j]) * xq_h2f(suh[blk * 128 + lane * 4 + j]);
            exl3_had128(v, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j) a1_sa[(size_t)r * SK + blk * 128 + lane * 4 + j] = xq_f2h(v[j]);
        }
    }
    __syncthreads();

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;

    // S-A3-n F4: keep DRAM busy without registers — lanes 0..2 prefetch the
    // 16x16 block window (8*BITS words <= 160 B) PF_D steps ahead into L2
    // (one predicated instruction/step; no value change). The in-round expert
    // stream is DRAM-latency-bound at 24 warps/SM with one 96-B window in flight.
    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < BITS * 32;          // 3/4/5 lanes x 32 B
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;              // bytes per k16 step
    // S-A3-o O1: prologue — the first PF_D windows were never prefetched (every CTA
    // started with PF_D serial DRAM misses, all CTAs of a wave in lock-step). Issue
    // them up front; value-transparent (L2 hint only).
    if (pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (d < KB) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    for (int kb = 0; kb < KB; ++kb) {
        if (pf_lane && kb + PF_D < KB)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
        pf_ptr += pf_step;
        uint32_t bfrag[2][2];
#if A1_SHFL
        // S-A3-n F7: ONE coalesced load per lane per step (window words
        // lane, lane+32 < 8*BITS), decode words delivered by __shfl_sync —
        // replaces 16 scalar L1 loads per lane per step. Same words => same bits.
        constexpr int WW = 8 * BITS;
        const unsigned w0 = (lane < WW) ? __ldg(ring + lane) : 0u;
        const unsigned w1 = (WW > 32 && lane + 32 < WW) ? __ldg(ring + lane + 32) : 0u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint16_t rv[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = h * 4 + q;
                unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                if (WW > 32) {
                    const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                    const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                    lo = s_i1m[j] >= 32 ? lo2 : lo;
                    hi = s_i0m[j] >= 32 ? hi2 : hi;
                }
                const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                const unsigned sum = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                            __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
            }
            bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
            bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
        }
#else
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
#endif
        ring += ring_kstride;
        const int col = kb * 16;
        const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
        const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
        uint32_t a1 = 0u, a3 = 0u;
        if (hi_rows) {
            a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
            a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
    }

    // fused xq_had_svh_multi: the fp16 GEMM tile (the exact yraw values) staged in smem, then
    // y = f16(H128(yraw) * svh) per row with the SAME butterfly and rounding.
    __syncthreads();                          // every warp is done reading A from a1_sa
    __half* yt = a1_sa;                       // M x 128 halves (<= M*(K+8))
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lc = 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) *(__half2*)&yt[row0 * 128 + lc] = __floats2half2_rn(acc[h][0], acc[h][1]);
        if (row1 < M) *(__half2*)&yt[row1 * 128 + lc] = __floats2half2_rn(acc[h][2], acc[h][3]);
    }
    __syncthreads();
    for (int r = warp; r < M; r += (blockDim.x >> 5)) {
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = xq_h2f(yt[r * 128 + lane * 4 + j]);
        exl3_had128(v, lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            y[(size_t)r * N + cta_n + lane * 4 + j] = xq_f2h(v[j] * xq_h2f(svh[cta_n + lane * 4 + j]));
    }
}

// 2026-09-26: the served expert GEMM with BOTH per-expert Hadamards fused (A5 D3, experts):
// xq_had_suh_multi -> xq_gemm_grouped_a1b3 -> xq_had_svh_multi become ONE launch per GEMM, bit
// for bit the same bytes (same load/product/butterfly/rounding; same K loop and mma sequence).
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3_fh(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                        const __half* __restrict__ x_in, int per_expert_in,
                        const __half* __restrict__ suh_all, const __half* __restrict__ svh_all,
                        const int* __restrict__ idxmap, __half* __restrict__ y,
                        int M, int K, int N, const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ix = idxmap[expert];
    exl3_gemm_body_a1_fh<3>(base + (size_t)offs[expert],
                            x_in + (per_expert_in ? (long long)expert * M * K : 0LL),
                            suh_all + (long long)ix * K, svh_all + (long long)ix * N,
                            y + (long long)expert * M * N, M, K, N, tile);
}

// S-A3-o O3: the expert stream's served body (A-once, 3-bit) as its OWN entry, so
// its register budget is not the max over the barrier/b5 bodies sharing
// xq_gemm_grouped_xh (96 regs -> 2 CTAs/SM). Same body, same instruction sequence
// per output => bitwise identical to xq_gemm_grouped_xh (probe-exl3-binv).
// Preconditions (launcher-checked): exl3_a1_fits(M, K), bits == 3.
extern "C" __global__ void __launch_bounds__(256, 3)
xq_gemm_grouped_a1b3(const uint16_t* __restrict__ base,
                     const uint64_t* __restrict__ offs,
                     const __half* __restrict__ xh,
                     __half* __restrict__ y,
                     int M, int K, int N, int bits,
                     const int* __restrict__ esel) {
    XQ_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    exl3_gemm_body_a1<3>(base + (size_t)offs[expert], xh + (long long)expert * M * K,
                         y + (long long)expert * M * N, M, K, N, tile);
}

