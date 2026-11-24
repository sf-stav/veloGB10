// exl3_bench/23_tp_allreduce.cuh: TP-A..TP-4H2 transports: decode all-reduce with K1 folded, K2m, vocab-parallel lm_head, sharded DHEAD screen.
// Same doorbell protocol as gpu_batch.cu K1/K2 (native/tp_doorbell.h I1-I9): only WHO runs each step moves.
// Section 23 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// =============================================================================
// TP-G (T2 lever #2, PLAN/notes_2026-09-28/TP-G.md): the DECODE all-reduce (<= 16 rows) with K1
// FOLDED into the producing kernel's epilogue and a multi-block K2 with an optional GPU-side
// receive. Same doorbell protocol as gpu_batch.cu's K1/K2 (native/tp_doorbell.h I1-I9: same rings,
// same len/tail tags, same proxy, same wire bytes, fp32 payload) — only WHO runs each step moves:
//   producer+K1 : every block computes its share of the fp32 partial EXACTLY as the unfused producer
//                 (xq_ks_combine_f32 / xq_moe_combine_ep / xq_cvt_f16_f32, same expressions) into the
//                 device-memory local partial and arrives on a counter; the LAST block runs K1: the I3
//                 reuse gate (plain loads, I6), the partial -> send slot copy, fence.sc.sys, tail epoch +
//                 len tag, c->epoch, RELEASE gpu_ready — the single-block K1's publish sequence.
//   K2 (dec)    : every block gates itself: the proxy's validated release (cpu_done >= e, I5) OR, with
//                 io bit 3, the GPU-side placement proof tp_wait_add_g uses (peer_committed >= e hint +
//                 the slot's generation-tagged tail == e, then fence.acquire.sys); publishes RX_DONE = e
//                 (the proxy then skips validating epochs the GPU already proved — the slot may hold a
//                 later generation by the time the proxy looks). Sum = tp_reduce_body mode 2's canonical
//                 lower + upper fp32 add, rounded ONCE to f16 with __float2half_rn (= xq_cvt_f32_f16)
//                 when io bit 1 — the separate cvt launch is gone. Bitwise equal to K1 + K2 + cvt.
// World-general gates (world>2: partner(e-R)'s per-QP tx_retired, partner(e)'s per-peer cpu_done +
// the R9 tail equality, per-round recv rings, the round-k canonical order); the host takes this path
// at world 2 only (f16 out would round a world>2 intermediate round sum — TP-F.md §7 item 1).
// =============================================================================
__device__ __forceinline__ unsigned long long xtp_ld(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void xtp_st_rel(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void xtp_fence_acq() { asm volatile("fence.acquire.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long xtp_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned long long* xtp_flag(tp_dev_ctx* c, int off) {
    return (unsigned long long*)((char*)c->flags + off);
}
__device__ __forceinline__ int xtp_partner(tp_dev_ctx* c, unsigned long long e) {
    return (int)c->rank ^ (1 << (int)(e % (unsigned long long)c->rounds));
}
__device__ __forceinline__ unsigned long long* xtp_nway(tp_dev_ctx* c, int peer, size_t sub) {
    return (unsigned long long*)((char*)c->nway_flags + (size_t)peer * TP_CL + sub);
}
__device__ __forceinline__ unsigned char* xtp_send_slot(tp_dev_ctx* c, unsigned long long e) {
    return c->send_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
__device__ __forceinline__ const unsigned char* xtp_recv_slot(tp_dev_ctx* c, unsigned long long e) {
    if (c->world > 2) {
        const unsigned long long r = (unsigned long long)c->rounds;
        const unsigned long long round = e % r;
        const unsigned long long slot = (e / r) & (TP_RING_SLOTS - 1);
        return c->nway_recv + (round * (unsigned long long)TP_RING_SLOTS + slot)
                              * (unsigned long long)c->slot_stride;
    }
    return c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}

// TP-4F2 (--tp-reduce single): the I3 reuse gate of the uniform one-shot ctx. Every epoch of that ctx is pushed
// to ALL THREE peers, so the send slot's previous owner (epoch e-R) is free only when EVERY peer QP retired it:
// warp 0, lane l < 3 polls tx_retired[rank ^ (l+1)] >= e-R (three independent host-memory loads in flight at
// once, not three serial round trips on the critical path), the warp votes, same bounded deadline / abort
// word / status 11 as the tree gate. Reached only when c->oneshot (the dedicated ctx; rails 1 and 2 have 0).
// World 4 only (the host refuses the flag elsewhere). All threads call; returns 1 = aborted.
__device__ __forceinline__ int xtp_k1_gate_os(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab_os;
    if (threadIdx.x < 32) {
        const int lane = (int)threadIdx.x;
        if (lane == 0) s_ab_os = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = (lane < 3) ? xtp_nway(c, (int)c->rank ^ (lane + 1), TP_NWAY_TX_OFF) : nullptr;
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            auto retired = [&]() -> bool { return ret == nullptr || xtp_ld(ret) >= tgt; };
            bool ok = __all_sync(0xffffffffu, retired());
            if (!ok) {
                if (lane == 0) c->gate_waits += 1;
                const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                for (;;) {
                    if (__all_sync(0xffffffffu, retired())) break;
                    int stop = 0;
                    if (lane == 0) {
                        if (xtp_ld(ab)) stop = 1;
                        else if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); stop = 1; }
                    }
                    stop = __shfl_sync(0xffffffffu, stop, 0);
                    if (stop) { if (lane == 0) s_ab_os = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            xtp_fence_acq();                        // every lane (no lane reads lane 0's shared flag mid-warp)
        }
    }
    __syncthreads();
    return s_ab_os;
}

// The I3 reuse gate (tp_gate_copy_signal_mb's, verbatim): thread 0 waits tx_retired >= e-R, bounded
// (K1_GATE_WAIT_NS -> device status 11, I9). All threads call; returns 1 = aborted (store nothing).
__device__ __forceinline__ int xtp_k1_gate(tp_dev_ctx* c, unsigned long long e) {
    if (c->oneshot) return xtp_k1_gate_os(c, e);        // TP-4F2: the uniform one-shot ctx only (rails 1/2: 0)
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = (c->world > 2)
                ? xtp_nway(c, xtp_partner(c, tgt), TP_NWAY_TX_OFF) : xtp_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            // TP-I (audit C7): the only caller is xtp_k1_last, where exactly ONE block (the last to
            // arrive, an arbitrary blockIdx) runs the gate, so count on its thread 0. The old
            // `blockIdx.x == 0` guard (inherited from the multi-block K1m, gpu_batch.cu, where every
            // block gates) made the folded decode path under-report binds at grid > 1. Diagnostic only.
            if (xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) {
                    // device status 11 (I9; no printf: it would put a stack frame on the producer —
                    // the host reports the abort loudly at its next agree / sync)
                    xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11);
                    s_ab = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

// TP-G v2 fold (the "last-block" K1): the producer blocks store ONLY their device-memory partial
// (no system-scope traffic per block: the v1 fold, where every block wrote the host-memory send slot
// and ran fence.sc.sys, cost ~+11 us per 80 KiB barrier on the transport bench), arrive on the counter
// (fence.gpu), and the LAST block runs K1 itself: the I3 gate, the whole partial -> send slot copy
// (16 B vectors, L2-hot), fence.sc.sys, tail + len tag, release gpu_ready. One launch fewer than
// producer + K1, same bytes on the wire, same publish sequence. nbytes % 16 == 0 (host).
__device__ __forceinline__ void xtp_k1_last(tp_dev_ctx* c, unsigned* arrive, const float* part, unsigned nbytes) {
    __shared__ int s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();                                // this block's partial stores, device scope
        const unsigned prev = atomicAdd(arrive, 1u);
        s_last = (prev == gridDim.x * gridDim.y * gridDim.z - 1u);
        if (s_last) __threadfence();                    // acquire side of the last-block pattern
    }
    __syncthreads();
    if (!s_last) return;
    const unsigned long long e = c->epoch + 1;          // the pre-kernel counter (only the last block reads it)
    const int ab = xtp_k1_gate(c, e);
    if (!ab) {
        uint4* dst = (uint4*)xtp_send_slot(c, e);
        const uint4* s4 = (const uint4*)part;
        // TP-4Z16a: the copy is latency-bound (ONE block, one load + one store per thread per trip): four loads in flight per
        // trip halve it (80 KiB: 4.0 -> 2.0 us, 160 KiB: 7.5 -> 3.3 us, kernel-only microbenchmark). Same bytes, same slot.
        const unsigned n16 = nbytes >> 4;
        unsigned i = threadIdx.x;
        for (; i + 3u * blockDim.x < n16; i += 4u * blockDim.x) {
            const uint4 v0 = __ldcg(s4 + i), v1 = __ldcg(s4 + i + blockDim.x),
                        v2 = __ldcg(s4 + i + 2u * blockDim.x), v3 = __ldcg(s4 + i + 3u * blockDim.x);
            dst[i] = v0; dst[i + blockDim.x] = v1; dst[i + 2u * blockDim.x] = v2; dst[i + 3u * blockDim.x] = v3;
        }
        for (; i < n16; i += blockDim.x) dst[i] = __ldcg(s4 + i);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // the slot stores, system scope, before the tail
        *arrive = 0u;
        c->epoch = e;
        if (c->world > 2 && c->qp_mask)
            c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << xtp_partner(c, e);
        if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
            unsigned char* slot = xtp_send_slot(c, e);
            const unsigned wire = (nbytes + 7u) & ~7u;
            *(unsigned long long*)(slot + wire) = e;
            c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
            xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
        }
    }
}

extern "C" __global__ void xq_ks_combine_f32_k1l(tp_dev_ctx* c, const float* __restrict__ ws, float* __restrict__ out,
                                                 int M, int N, int KS, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        const int m = idx / N;
        const int n = idx - m * N;
        float acc = 0.0f;
        for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
        out[idx] = acc;
    }
    xtp_k1_last(c, arrive, out, (unsigned)(M * N) * 4u);
}

extern "C" __global__ void xq_moe_combine_ep_k1l(tp_dev_ctx* c, float* __restrict__ out, const __half* __restrict__ y_d,
                                                 const __half* __restrict__ y_sh,
                                                 const int* __restrict__ ids, const float* __restrict__ wts,
                                                 const int* __restrict__ slotmap, const __half* __restrict__ sg,
                                                 const __half* __restrict__ x, int k, int h, int M,
                                                 int add_shared, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int m = blockIdx.x;                           // host: grid == M
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    float sgv = 0.0f;
    if (add_shared) {
        float part = 0.0f;
        for (int cc = tid; cc < h; cc += bs) part += xq_h2f(sg[cc]) * xq_h2f(x[(long long)m * h + cc]);
        sm[tid] = part; __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        sgv = xq_sig(sm[0]);
    }
    for (int i = tid; i < h; i += bs) {
        float acc = add_shared ? __fmul_rn(sgv, xq_h2f(y_sh[(long long)m * h + i])) : 0.0f;
        for (int j = 0; j < k; j++) {
            int ee = ids[(long long)m * k + j];
            int slot = slotmap[ee];
            if (slot >= 0)
                acc = __fmaf_rn(wts[(long long)m * k + j], xq_h2f(y_d[((long long)slot * M + m) * h + i]), acc);
        }
        out[(long long)m * h + i] = acc;
    }
    xtp_k1_last(c, arrive, out, (unsigned)(M * h) * 4u);
}

// The ks == 1 decode chain: its partial is the GEMM's f16 y_raw — widen it (exact, xq_cvt_f16_f32)
// into the fp32 local partial, then the last block runs K1.
extern "C" __global__ void xq_cvt_f16_f32_k1l(tp_dev_ctx* c, float* __restrict__ out, const __half* __restrict__ in,
                                              int n, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
    xtp_k1_last(c, arrive, out, (unsigned)n * 4u);
}

// TP-4Z2: L2 run-ahead of the NEXT kernel's weights from the otherwise idle threads of a K2 block. A K2 spends tens of
// microseconds waiting for the peer's payload with the DRAM bus idle, and the kernel behind it (the hc mixer, 6.5 MB of int8
// weights) is weight-bound and streams cold from DRAM. The weights never depend on this reduce's output, so they can be
// pulled into L2 now. Pure prefetch: no data path, no output byte, and a 0 byte count (the default) is a no-op. The threads
// below `skip` (the spinners) never take part, so the wait is not delayed; the rest of the grid stripes the two ranges by
// 128-byte line.
__device__ __forceinline__ void xq_l2_prefetch2(unsigned long long p0, unsigned n0, unsigned long long p1, unsigned n1,
                                                unsigned skip) {
    if (n0 == 0u && n1 == 0u) return;
    if (threadIdx.x < skip) return;
    const unsigned per = blockDim.x - skip;
    const unsigned t = blockIdx.x * per + (threadIdx.x - skip);
    const unsigned nt = gridDim.x * per;
    for (unsigned off = t * 128u; off < n0; off += nt * 128u) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + off));
    for (unsigned off = t * 128u; off < n1; off += nt * 128u) asm volatile("prefetch.global.L2 [%0];" :: "l"(p1 + off));
}

// K2 (decode): gate per block (see the section header), then out = lower + upper (fp32, the canonical
// order), f16 (__float2half_rn) when io bit 1. io bit 0: `local` is f16 (widened exactly); bit 3: the
// GPU-side receive. n % 4 == 0 (host). out may alias local (same-thread read-then-write).
extern "C" __global__ void xq_tp_wait_add_dec(tp_dev_ctx* c, void* out, const void* local, int n, int io,
                                              unsigned long long pf0, unsigned pfn0, unsigned long long pf1, unsigned pfn1) {
    XQ_PDL_ENTRY();                                     // the producer+K1 wrote c->epoch: wait for it
    xq_l2_prefetch2(pf0, pfn0, pf1, pfn1, 1u);          // TP-4Z2: thread 0 spins; every other thread run-ahead-prefetches
    const unsigned long long e = c->epoch;
    const unsigned char* slot = xtp_recv_slot(c, e);
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool nw = c->world > 2, g = (io & 8) != 0;
        const unsigned long long* done = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF) : xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF) : xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* tail = (const unsigned long long*)(slot + (((unsigned)n * 4u + 7u) & ~7u));
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            // v1: the proxy's validated release (world>2 adds R9's tail equality: cpu_done is a watermark)
            if (xtp_ld(done) >= e && (!nw || xtp_ld(tail) == e)) break;
            // GPU-side receive: the commit hint arms, the payload's own generation-tagged tail proves
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }        // the proxy's tail/len guards abort a lost payload
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    if (s_ab) return;
    const int lower = (c->world > 2) ? ((c->rank & (1 << (int)(e % (unsigned long long)c->rounds))) == 0) : (c->rank == 0);
    const float4* pe = (const float4*)slot;
    const int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 l4;
        if (io & 1) {
            const uint2 u = ((const uint2*)local)[i];
            const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            l4 = make_float4(f01.x, f01.y, f23.x, f23.y);
        } else {
            l4 = ((const float4*)local)[i];
        }
        const float4 p4 = pe[i];
        const float4 a = lower ? l4 : p4;
        const float4 b = lower ? p4 : l4;
        float4 r;
        r.x = a.x + b.x; r.y = a.y + b.y; r.z = a.z + b.z; r.w = a.w + b.w;
        if (io & 2) {
            __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
            uint2 u;
            u.x = *reinterpret_cast<unsigned*>(&h01);
            u.y = *reinterpret_cast<unsigned*>(&h23);
            ((uint2*)out)[i] = u;
        } else {
            ((float4*)out)[i] = r;
        }
    }
}

// TP-4F2 K2 (--tp-reduce single): the ONE-STAGE world-4 decode/verify reduce. The dedicated uniform one-shot
// ctx carries ONE epoch per logical reduce (the producer's folded K1 published this rank's fp32 partial to all
// three peers); here every block gates on all three peers (warp 0, lane l < 3 = peer rank ^ (l+1): the proxy's
// validated release cpu_done[p] >= e AND the sender-indexed slot's generation-tagged tail == e, R9 — the same two
// conditions the rd K2 gates on, per peer), then forms
//     out = f16( (p0 + p2) + (p1 + p3) )          p_r = rank r's fp32 partial (own = `local`)
// which is EXACTLY the rd two-stage association (stage 0 pairs {0,2},{1,3} — epoch parity, PLAN/TP-4F2_REPORT.md
// section 1 — stage 1 adds the pair sums; fp32 add is commutative so the grouping is rank-independent), rounded
// ONCE with __floats2half2_rn like the rd last stage. n % 4 == 0; out must not alias local (the host passes
// y_raw / moe_out vs the fp32 `part`).
extern "C" __global__ void xq_tp_wait_add_dec_single(tp_dev_ctx* c, __half* out, const float* local, int n,
                                                     unsigned long long pf0, unsigned pfn0, unsigned long long pf1, unsigned pfn1) {
    XQ_PDL_ENTRY();                                     // the producer+K1 wrote c->epoch: wait for it
    xq_l2_prefetch2(pf0, pfn0, pf1, pfn1, 32u);         // TP-4Z2: warp 0 spins; the other warps run-ahead-prefetch
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    const int rank = (int)c->rank;
    const size_t ring_e = (size_t)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x < 32) {
        const int lane = (int)threadIdx.x;
        if (lane == 0) s_ab = 0;
        const int peer = (lane < 3) ? (rank ^ (lane + 1)) : -1;
        const unsigned long long* done = nullptr;
        const unsigned long long* tail = nullptr;
        if (peer >= 0) {
            done = xtp_nway(c, peer, TP_NWAY_CPU_OFF);
            tail = (const unsigned long long*)(c->oneshot_recv
                   + ((size_t)peer * TP_RING_SLOTS + ring_e) * (size_t)c->slot_stride + wire);
        }
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            const bool mine = (peer < 0) || (xtp_ld(done) >= e && xtp_ld(tail) == e);
            if (__all_sync(0xffffffffu, mine)) break;
            int stop = (lane == 0) ? (int)(xtp_ld(ab) != 0) : 0;   // the proxy's guards abort a lost payload
            stop = __shfl_sync(0xffffffffu, stop, 0);
            if (stop) { if (lane == 0) s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        xtp_fence_acq();                                // every lane, then the block barrier publishes it
    }
    __syncthreads();
    if (s_ab) return;
    const float4* pr[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q)
        pr[q] = (const float4*)(c->oneshot_recv + ((size_t)q * TP_RING_SLOTS + ring_e) * (size_t)c->slot_stride);
    const float4* lo = (const float4*)local;
    const int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = (q == rank) ? lo[i] : pr[q][i];
        float4 r;
        r.x = (v[0].x + v[2].x) + (v[1].x + v[3].x);
        r.y = (v[0].y + v[2].y) + (v[1].y + v[3].y);
        r.z = (v[0].z + v[2].z) + (v[1].z + v[3].z);
        r.w = (v[0].w + v[2].w) + (v[1].w + v[3].w);
        __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
        uint2 u;
        u.x = *reinterpret_cast<unsigned*>(&h01);
        u.y = *reinterpret_cast<unsigned*>(&h23);
        ((uint2*)out)[i] = u;
    }
}

// =============================================================================
// TP-I3 (c1): K2m FOLDED into the prefill consumers (world 2, pipelined transport, row-aligned
// epochs: an epoch carries whole rows [r0, r1) of the chunk). Each kernel takes K2m's place in the
// schedule (same stream position, same `lag`, so e = c->epoch - lag is K2m's epoch), gates like K2m
// (world 2: the proxy's validated cpu_done >= e, abort-aware, the critical-spin shape of
// tp_spin_until_ge), reads the peer's partial straight from the recv slot and the local partial, forms
// the canonical lower + upper fp32 sum and the single __float2half_rn — K2m's exact element ops — and
// hands that f16 value to the consumer body instead of writing it to memory. Bitwise the K2m +
// consumer pair; the f16 reduce output is never written or re-read. K2m publishes no flag in prefill
// (the recv-slot rule is stream order, R >= 2G), so neither do these.
// =============================================================================
__device__ __forceinline__ int xtp_k2m_gate(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const unsigned long long* done = xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        while (xtp_ld(done) < e) {
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) xtp_fence_acq();
    }
    __syncthreads();
    return s_ab;
}

// MoE sites: K2m (fp32 local partial, fp32 wire, f16 out) + xq_pf_hc_inj_norm over the epoch's rows.
// grid = rows of the epoch, block 512 = 4 groups of 128 threads; group s is xq_pf_hc_inj_norm's
// (row, stream s) CTA verbatim (same thread -> element map, same fma chain, same trees) reading the
// row's f16 y from shared memory, where the whole CTA staged it once (K2m's element ops). Pointers are
// offset to the epoch's first row by the host: local [rows][h] f32, out/x [rows][hc][h], inj [rows][hc].
// hc == 4 (the served model; host-checked), h % 512 == 0, h <= 4096.
extern "C" __global__ void __launch_bounds__(512)
xq_pf_hc_inj_norm_k2(tp_dev_ctx* c, unsigned lag, const float* __restrict__ local, __half* __restrict__ out,
                     float* __restrict__ x, const float* __restrict__ inj, const __half* __restrict__ w,
                     int h, float eps) {
    XQ_PDL_ENTRY();
    constexpr int HC = 4;
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    if (xtp_k2m_gate(c, e)) return;
    const int b = blockIdx.x;
    const float* peer = (const float*)xtp_recv_slot(c, e) + (long long)b * h;
    const float* loc = local + (long long)b * h;
    const bool lower = (c->rank == 0);
    __shared__ __align__(16) __half ys[4096];
    for (int i4 = threadIdx.x; i4 < (h >> 2); i4 += 512) {
        const float4 l4 = ((const float4*)loc)[i4];
        const float4 p4 = ((const float4*)peer)[i4];
        const float4 a = lower ? l4 : p4;
        const float4 bb = lower ? p4 : l4;
        float4 r;
        r.x = a.x + bb.x; r.y = a.y + bb.y; r.z = a.z + bb.z; r.w = a.w + bb.w;
        const __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&h01);
        u.y = *reinterpret_cast<const unsigned*>(&h23);
        *reinterpret_cast<uint2*>(ys + 4 * i4) = u;
    }
    __syncthreads();
    // ---- xq_pf_hc_inj_norm's body for (row b, stream s = group)
    const int s = threadIdx.x >> 7;
    const int blk = b * HC + s;
    __shared__ float smg[HC][512];
    __shared__ float s_invg[HC];
    float* sm = smg[s];
    const int T = threadIdx.x & 127;
    float* xs = x + (long long)blk * h;
    const float g = inj[b * HC + s];
    const int nk = h >> 9;
    constexpr int KM = 8;
    float v[KM][4];
    float pa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, pb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float4 t4 = *reinterpret_cast<const float4*>(xs + i);
            float t[4] = {t4.x, t4.y, t4.z, t4.w};
            float yf[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ys + i), yf);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) t[e2] += g * yf[e2];          // xq_hc_inject's fma
            *reinterpret_cast<float4*>(xs + i) = make_float4(t[0], t[1], t[2], t[3]);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                v[k][e2] = t[e2];
                if (k & 1) pb[e2] += t[e2] * t[e2]; else pa[e2] += t[e2] * t[e2];   // xq_hc_norm's sum_sq
            }
        }
    }
    #pragma unroll
    for (int e2 = 0; e2 < 4; ++e2) sm[4 * T + e2] = pa[e2] + pb[e2];              // level 512
    __syncthreads();
    if (T < 32) {
        const float r = xq_pf_tree512(sm, T);
        if (T == 0) s_invg[s] = rsqrtf(r / (float)h + eps);
    }
    __syncthreads();
    const float inv = s_invg[s];
    const __half* ws = w + (long long)s * h;
    __half* os = out + (long long)blk * h;
    #pragma unroll
    for (int k = 0; k < KM; ++k) {
        if (k < nk) {
            const int i = 512 * k + 4 * T;
            float wf[4], o[4];
            xq_pf_unpack4(*reinterpret_cast<const uint2*>(ws + i), wf);
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) o[e2] = v[k][e2] * inv * (1.0f + wf[e2]);
            *reinterpret_cast<uint2*>(os + i) = xq_pf_pack4(o);
        }
    }
}

// Mixer sites: K2m (f16 local partial = the chain's y_raw, f16 wire, f16 out) + exl3_had_svh over the
// epoch's rows. A persistent grid (block 256 = 8 warps; grid <= 96): ONE gate per CTA (v1 gave every
// (row, 128-block) its own 32-thread CTA, i.e. ~4,000 CTAs per epoch each spinning on the system-scope
// cpu_done line and fencing — 839 us per epoch in the trace), then each warp runs exl3_had_svh's warp
// body on tasks (row m, block nb) t = cta*8 + warp, += grid*8: per lane 4 elements formed as K2m io 7
// (widen both f16 partials exactly, canonical fp32 add, __floats2half2_rn) — then exl3_had_svh_dev's
// arithmetic on those values. local / y are offset to the epoch's first row by the host.
extern "C" __global__ void __launch_bounds__(256)
exl3_had_svh_k2(tp_dev_ctx* c, unsigned lag, const __half* __restrict__ local,
                const __half* __restrict__ svh, __half* __restrict__ y, int n, int rows) {
    XQ_PDL_ENTRY();
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    if (xtp_k2m_gate(c, e)) return;
    const __half* peer = (const __half*)xtp_recv_slot(c, e);
    const bool lower = (c->rank == 0);
    const int nbk = n >> 7;
    const int tasks = rows * nbk;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int t = blockIdx.x * 8 + warp; t < tasks; t += gridDim.x * 8) {     // warp-uniform
        const int nb = t % nbk;
        const int m = t / nbk;
        const size_t base = (size_t)m * n + (size_t)nb * 128 + lane * 4;
        float v[4];
        {
            const uint2 ul = *reinterpret_cast<const uint2*>(local + base);
            const uint2 up = *reinterpret_cast<const uint2*>(peer + base);
            const float2 l01 = __half22float2(*reinterpret_cast<const __half2*>(&ul.x));
            const float2 l23 = __half22float2(*reinterpret_cast<const __half2*>(&ul.y));
            const float2 p01 = __half22float2(*reinterpret_cast<const __half2*>(&up.x));
            const float2 p23 = __half22float2(*reinterpret_cast<const __half2*>(&up.y));
            const float4 l4 = make_float4(l01.x, l01.y, l23.x, l23.y);
            const float4 p4 = make_float4(p01.x, p01.y, p23.x, p23.y);
            const float4 a = lower ? l4 : p4;
            const float4 bb = lower ? p4 : l4;
            float4 r;
            r.x = a.x + bb.x; r.y = a.y + bb.y; r.z = a.z + bb.z; r.w = a.w + bb.w;
            const __half2 h01 = __floats2half2_rn(r.x, r.y), h23 = __floats2half2_rn(r.z, r.w);
            v[0] = __half2float(__low2half(h01)); v[1] = __half2float(__high2half(h01));
            v[2] = __half2float(__low2half(h23)); v[3] = __half2float(__high2half(h23));
        }
        exl3_had128(v, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int j = lane * 4 + r;
            y[base + r] = __float2half_rn(v[r] * __half2float(svh[nb * 128 + j]));
        }
    }
}

// =============================================================================
// TP-H #4: vocab-parallel greedy lm_head tail (world 2). Each rank computes ONLY its 124160-column
// shard of the logits (lmh GEMM on the shard Quad, fused Hadamard: bitwise the replicated head's
// columns), reduces each row to ONE u64 key (xq_am_key: value desc, then lowest id — a TOTAL
// order), and exchanges the m keys through the doorbell transport; both ranks then take the
// max-merge of the two keys per row. Because the key order is total, max(shard keys with GLOBAL
// ids) == the replicated xq_argmax_rows result bit for bit (NaN loses in both; the degenerate
// all(-inf/NaN) row resolves to id 0 in both, since rank 0's sentinel outranks rank 1's).
//   xq_argmax_rows_vp : xq_argmax_rows' body over the shard (NO early return — every block must
//                       reach the fold), the row-last block writes keys[r] with the GLOBAL id, then
//                       xtp_k1_last (the TP-G v2 last-block K1) over the whole grid publishes keys.
//   xq_tp_wait_keys   : K2 — the shared GPU-side gate, then out_ids[r] = id(max(own, peer)).
// nbytes = 8 * roundup_even(m) (a multiple of 16: xtp_k1_last copies uint4). The padding keys are
// never written (zero on both ranks).
// =============================================================================
extern "C" __global__ void xq_argmax_rows_vp(tp_dev_ctx* c, unsigned long long* __restrict__ keys,
                                             const __half* __restrict__ logits, long long rstride,
                                             int V, int id_base,
                                             unsigned long long* __restrict__ part,
                                             unsigned int* __restrict__ cnt,
                                             unsigned* __restrict__ arrive, unsigned nbytes) {
    XQ_PDL_ENTRY();
    __shared__ unsigned long long wk[32];
    __shared__ bool last;
    const int tid = threadIdx.x, bs = blockDim.x;
    const int r = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
    const __half* row = logits + (long long)r * rstride;
    const int chunk = (((V + nb - 1) / nb) + 7) & ~7;
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
        key = xq_am_wmax(tid < (bs >> 5) ? wk[tid] : 0ull);
        if (tid == 0) {
            part[(long long)r * nb + b] = key;
            __threadfence();
            last = (atomicAdd(cnt + r, 1u) == (unsigned)nb - 1u);
        }
    }
    __syncthreads();
    if (last && tid < 32) {
        __threadfence();
        const unsigned long long k2 = xq_am_wmax(tid < nb ? __ldcg(part + (long long)r * nb + tid) : 0ull);
        if (tid == 0) {
            const unsigned lid = 0xFFFFFFFFu - (unsigned)(k2 & 0xFFFFFFFFull);
            keys[r] = (k2 & 0xFFFFFFFF00000000ull) | (unsigned long long)(0xFFFFFFFFu - (lid + (unsigned)id_base));
            cnt[r] = 0u;                                      // re-arm for the next launch / replay
        }
    }
    xtp_k1_last(c, arrive, reinterpret_cast<const float*>(keys), nbytes);
}

// The K2 gate of xq_tp_wait_add_dec (same three release conditions, same fence / RX_DONE), as a
// single-block function: returns 1 = aborted (store nothing).
__device__ __forceinline__ int xtp_k2_gate1(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool nw = c->world > 2, g = (io & 8) != 0;
        const unsigned long long* done = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF) : xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = nw ? xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF) : xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e && (!nw || xtp_ld(tail) == e)) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void xq_tp_wait_keys(tp_dev_ctx* c, unsigned long long* keys, int* out_ids,
                                           int m, int nbytes, int io) {
    XQ_PDL_ENTRY();                                     // the argmax+K1 wrote c->epoch: wait for it
    const unsigned long long e = c->epoch;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const unsigned long long* tail = (const unsigned long long*)(slot + (((unsigned)nbytes + 7u) & ~7u));
    if (xtp_k2_gate1(c, e, tail, io)) return;
    const unsigned long long* pk = (const unsigned long long*)slot;
    for (int r = threadIdx.x; r < m; r += blockDim.x) {
        const unsigned long long a = keys[r], b = pk[r];
        const unsigned long long k = a > b ? a : b;     // total order: rank-order independent
        keys[r] = k;
        out_ids[r] = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
    }
}

// =============================================================================
// TP-H2: vocab-parallel lm_head for SAMPLED / PENALIZED / RATIO-RULE rows (world 2; --tp-vp-sampled).
// The sampler needs the full logits row, so each rank's shard rows (the TP-H #4 shard GEMM: every column
// bitwise the replicated head's, ks = 1) are ALL-GATHERED into sc.logits [m][V] through the decode
// doorbell, and the UNCHANGED xq_pen_rows / xq_argmax_rows / xq_sample_rows(_rq) run on the reassembled
// rows: bitwise the replicated head's buffer by construction (the gather moves f16 bits unchanged).
// One epoch carries rows [r0, r0 + nr) (nr <= 4: 4 x 248,320 B fits one 1 MiB ring slot); the host
// launches the K1s of every epoch, then the K2s (lag = epochs - 1 - j; G <= 4 = R/2: the TP-F recv-slot
// rule). Payload layout in the slot: [nr][n_sh] f16, row-contiguous.
//   xq_vp_gather_k1 : K1m's shape (tp_gate_copy_signal_mb): every block waits the I3 reuse gate itself,
//                     copies its grid-stride share of the shard rows to BOTH the send slot and this rank's
//                     columns [col_own, col_own + n_sh) of the logits rows, fence.sc.sys, arrives; the LAST
//                     block publishes (c->epoch, tail epoch, len tag, gpu_ready release) — K1m's sequence.
//   xq_vp_gather_k2 : the receive gate of xq_tp_wait_add_dec per block (cpu_done, or with io bit 3 the
//                     commit hint + the payload's generation-tagged tail; RX_DONE from block 0), then the
//                     peer's rows -> its columns [col_peer, col_peer + n_sh). No arithmetic anywhere.
// All row / column offsets are 16 B aligned (n_sh % 8 == 0, V % 8 == 0: host-checked). World 2 only.
// =============================================================================
// The I3 reuse gate of K1m (every block gates itself; the diagnostic count on block 0 only).
__device__ __forceinline__ int xtp_k1_gate_mb(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = xtp_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather_k1(
        tp_dev_ctx* c, const __half* __restrict__ shard, __half* __restrict__ logits, long long ldl,
        int n_sh, int col_own, int r0, int nr, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    // every block reads the PRE-kernel counter (the last block stores it only after all arrived)
    const unsigned long long e = c->epoch + 1;
    const int ab = xtp_k1_gate_mb(c, e);
    unsigned char* slot = xtp_send_slot(c, e);
    const unsigned nbytes = (unsigned)nr * (unsigned)n_sh * 2u;
    if (!ab) {
        const int rq = n_sh >> 3;                                   // uint4 per shard row
        const int tot = nr * rq;
        const uint4* src = reinterpret_cast<const uint4*>(shard) + (long long)r0 * rq;
        uint4* dsl = reinterpret_cast<uint4*>(slot);
        const long long lq = ldl >> 3;                              // uint4 per logits row
        uint4* dlg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq + (col_own >> 3);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
            const int r = i / rq;
            const int j = i - r * rq;
            const uint4 v = __ldcg(src + i);
            dsl[i] = v;
            dlg[(long long)r * lq + j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's slot stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // re-armed for the next launch / replay
            c->epoch = e;                               // advances even on abort (I9 no-op), like K1m
            if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e; // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

// K2 gate, multi-block: xq_tp_wait_add_dec's three release conditions (cpu_done; with io bit 3 the commit
// hint + the payload tail; abort) per block, RX_DONE published by block 0 only. Returns 1 = aborted.
__device__ __forceinline__ int xtp_k2_gate_mb(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool g = (io & 8) != 0;
        const unsigned long long* done = xtp_flag(c, TP_F_CPU_DONE);
        const unsigned long long* hint = xtp_flag(c, TP_F_PEER_COMMITTED);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather_k2(
        tp_dev_ctx* c, __half* __restrict__ logits, long long ldl, int n_sh, int col_peer,
        int r0, int nr, unsigned lag, int io) {
    XQ_PDL_ENTRY();                                     // the K1s wrote c->epoch: wait for them
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const unsigned nbytes = (unsigned)nr * (unsigned)n_sh * 2u;
    const unsigned long long* tail = (const unsigned long long*)(slot + ((nbytes + 7u) & ~7u));
    if (xtp_k2_gate_mb(c, e, tail, io)) return;
    const int rq = n_sh >> 3;
    const int tot = nr * rq;
    const uint4* src = reinterpret_cast<const uint4*>(slot);
    const long long lq = ldl >> 3;
    uint4* dlg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq + (col_peer >> 3);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
        const int r = i / rq;
        const int j = i - r * rq;
        dlg[(long long)r * lq + j] = src[i];
    }
}

// =============================================================================
// TP-4H2: WORLD-4 vocab-parallel row all-gather (--tp-vp-sampled at world 4). ADDITIVE: nothing above
// (xq_vp_gather_k1/k2, xtp_k1_gate_mb, xtp_k2_gate_mb — the world-2 path) is touched; the host takes
// these kernels at world 4 only (`vp_gather_rows_w4`, src/exl3_forward/xtp.rs). The schedule, launch
// tables and the index math below are mirrored in src/tp_vpgather.rs and pinned by its CPU tests.
//
// Recursive doubling over the decode doorbell, TWO epochs per group of nr <= 4 rows (partner(e) = rank ^
// (1 << (e % rounds)), rounds = 2; shard s = logits columns [s * n_sh, (s + 1) * n_sh)):
//   stage 0 (epoch e)   : payload = the OWN shard rows [nr][n_sh] f16 (from the shard GEMM output); K1 also
//                         copies them into this rank's columns of the logits rows. K2 writes the partner's
//                         shard (partner = rank ^ m0, m0 = 1 << (e % rounds)) into ITS columns.
//   stage 1 (epoch e+1) : payload = TWO shards, [blk 0 = shard rank][blk 1 = shard rank ^ m0] (m0 = the
//                         PREVIOUS epoch's mask: 1 << ((e + rounds - 1) % rounds)), each [nr][n_sh], read
//                         from the logits rows stage 0 assembled. K2 writes partner p's two shards
//                         (p, p ^ m0) — exactly the two this rank lacks. Not a contiguous column block
//                         when m0 = 2 (the odd-first-epoch phase): shards are indexed, never assumed adjacent.
// Payload unit i (uint4 = 8 f16) -> (blk, row, j): i = (blk * nr + row) * rq + j, rq = n_sh / 8. Payload
// bytes = nr * n_sh * 2 * (stage + 1) (<= 993,280 at nr = 4, stage 1), the 8 B generation tail at align8.
// The host launches per group K1(0) K2(0) K1(1) K2(1), every K2 with lag 0 (stage 1's K1 reads columns
// stage 0's K2 wrote; serial lookahead 1 = the proven recv-slot rule). No arithmetic anywhere: bit copies.
// All offsets are 16 B aligned (n_sh % 8 == 0, V % 8 == 0: host-checked). World 4 only (host-checked).
// =============================================================================
// The I3 reuse gate, multi-block, world > 2: every block waits partner(e - R)'s per-QP tx_retired (the
// same peer as partner(e): R is even). xtp_k1_gate's body with xtp_k1_gate_mb's per-block shape.
__device__ __forceinline__ int xtp_k1_gate_mb_nw(tp_dev_ctx* c, unsigned long long e) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = xtp_nway(c, xtp_partner(c, tgt), TP_NWAY_TX_OFF);
            const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && xtp_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = xtp_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (xtp_ld(ret) < tgt) {
                if (xtp_ld(ab)) { s_ab = 1; break; }
                if (xtp_now() >= deadline) { xtp_st_rel(xtp_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) xtp_fence_acq();
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather4_k1(
        tp_dev_ctx* c, const __half* __restrict__ shard, __half* __restrict__ logits, long long ldl,
        int n_sh, int r0, int nr, int stage, unsigned* __restrict__ arrive) {
    XQ_PDL_ENTRY();
    // every block reads the PRE-kernel counter (the last block stores it only after all arrived)
    const unsigned long long e = c->epoch + 1;
    const int ab = xtp_k1_gate_mb_nw(c, e);
    unsigned char* slot = xtp_send_slot(c, e);
    const int rq = n_sh >> 3;                                       // uint4 per shard row
    const int per = nr * rq;                                        // uint4 per shard block
    const int tot = per * (stage + 1);
    const unsigned nbytes = (unsigned)tot * 16u;
    if (!ab) {
        const int rank = (int)c->rank;
        const long long lq = ldl >> 3;                              // uint4 per logits row
        uint4* dsl = reinterpret_cast<uint4*>(slot);
        uint4* lg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq;
        if (stage == 0) {
            const uint4* src = reinterpret_cast<const uint4*>(shard) + (long long)r0 * rq;
            uint4* own = lg + (long long)rank * rq;                 // this rank's columns
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
                const int r = i / rq;
                const int j = i - r * rq;
                const uint4 v = __ldcg(src + i);
                dsl[i] = v;
                own[(long long)r * lq + j] = v;
            }
        } else {
            // the two shards held after stage 0: own + the stage-0 partner's (mask of the previous epoch)
            const int m0 = 1 << (int)((e + (unsigned long long)c->rounds - 1ull) % (unsigned long long)c->rounds);
            const int s1 = rank ^ m0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
                const int blk = i / per;
                const int rem = i - blk * per;
                const int r = rem / rq;
                const int j = rem - r * rq;
                const int s = blk ? s1 : rank;
                dsl[i] = __ldcg(lg + (long long)r * lq + (long long)s * rq + j);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's slot stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // re-armed for the next launch / replay
            c->epoch = e;                               // advances even on abort (I9 no-op), like K1m
            if (c->qp_mask) c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << xtp_partner(c, e);
            if (!ab && !xtp_ld(xtp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e; // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                xtp_st_rel(xtp_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

// K2 gate, multi-block, world > 2: xtp_k2_gate1's conditions (partner(e)'s per-peer cpu_done AND the R9
// tail equality; with io bit 3 the commit hint + the tail; abort) per block, RX_DONE by block 0 only.
__device__ __forceinline__ int xtp_k2_gate_mb_nw(tp_dev_ctx* c, unsigned long long e, const unsigned long long* tail, int io) {
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        const bool g = (io & 8) != 0;
        const unsigned long long* done = xtp_nway(c, xtp_partner(c, e), TP_NWAY_CPU_OFF);
        const unsigned long long* hint = xtp_nway(c, xtp_partner(c, e), TP_NWAY_PEER_OFF);
        const unsigned long long* ab = xtp_flag(c, TP_F_ABORT);
        const unsigned long long tight = xtp_now() + 2000ull;
        unsigned ns = 64;
        for (;;) {
            if (xtp_ld(done) >= e && xtp_ld(tail) == e) break;
            if (g && xtp_ld(hint) >= e && xtp_ld(tail) == e) break;
            if (xtp_ld(ab)) { s_ab = 1; break; }
            if (xtp_now() < tight) continue;
            __nanosleep(ns);
            if (ns < 512u) ns <<= 1;
        }
        if (!s_ab) {
            xtp_fence_acq();
            if (g && blockIdx.x == 0) xtp_st_rel(xtp_flag(c, TP_F_RX_DONE), e);
        }
    }
    __syncthreads();
    return s_ab;
}

extern "C" __global__ void __launch_bounds__(256) xq_vp_gather4_k2(
        tp_dev_ctx* c, __half* __restrict__ logits, long long ldl, int n_sh,
        int r0, int nr, int stage, unsigned lag, int io) {
    XQ_PDL_ENTRY();                                     // the K1s wrote c->epoch: wait for them
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    const unsigned char* slot = xtp_recv_slot(c, e);
    const int rq = n_sh >> 3;
    const int per = nr * rq;
    const int tot = per * (stage + 1);
    const unsigned nbytes = (unsigned)tot * 16u;
    const unsigned long long* tail = (const unsigned long long*)(slot + ((nbytes + 7u) & ~7u));
    if (xtp_k2_gate_mb_nw(c, e, tail, io)) return;
    // the partner's payload: stage 0 = its own shard; stage 1 = {partner, partner ^ (the previous epoch's mask)}
    const int p0 = (int)c->rank ^ (1 << (int)(e % (unsigned long long)c->rounds));
    const int p1 = p0 ^ (1 << (int)((e + (unsigned long long)c->rounds - 1ull) % (unsigned long long)c->rounds));
    const uint4* src = reinterpret_cast<const uint4*>(slot);
    const long long lq = ldl >> 3;
    uint4* lg = reinterpret_cast<uint4*>(logits) + (long long)r0 * lq;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += gridDim.x * blockDim.x) {
        const int blk = i / per;
        const int rem = i - blk * per;
        const int r = rem / rq;
        const int j = rem - r * rq;
        const int s = blk ? p1 : p0;
        lg[(long long)r * lq + (long long)s * rq + j] = src[i];
    }
}

// =============================================================================
// TP-H #3: sharded DHEAD screen (world 2). The draft head's 2-bit screen is 42 MB/pass — the biggest
// byte-bound item of a draft pass. Each rank streams ONLY its half of the 512 128-column blocks
// (grid = nblk / world, nb_off = rank * grid; the codes buffer stays whole, indexed by the GLOBAL
// block), writes the approximate block maxima of its blocks into the ZERO-PADDED exchange buffer
// `part` [nblk] f32 (the peer's half stays zero forever: only the owned half is ever written), and
// the last block publishes the whole buffer (xtp_k1_last, nbytes = 4 * nblk, a multiple of 16).
// K2 is the existing xq_tp_wait_add_dec on fp32 (out = ws bmax): lower + upper of a value and an
// exact 0.0 is the value, so every rank ends up with the replicated screen's bmax array BIT FOR
// BIT (a -0 max canonicalizes in dh_ord), and the replicated candidate selection + exact rescore
// then produce the replicated head's draft, dconf, d_dev and toks unchanged.
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 12) xq_dh_screen_k1(
        tp_dev_ctx* c, const uint4* __restrict__ codes, unsigned* __restrict__ ws,
        const __half* __restrict__ svh, float* __restrict__ part,
        int K, int B, float dq, int nb_off, unsigned* __restrict__ arrive, unsigned nbytes) {
    XQ_PDL_ENTRY();
    __shared__ __align__(16) unsigned xs[1024];
    __shared__ float s_y[128];
    switch (B) {
        case 1: dh_screen_body<1>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        case 2: dh_screen_body<2>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        case 4: dh_screen_body<4>(codes, ws, svh, K, dq, xs, s_y, part, nb_off); break;
        default: break;
    }
    xtp_k1_last(c, arrive, part, nbytes);
}
