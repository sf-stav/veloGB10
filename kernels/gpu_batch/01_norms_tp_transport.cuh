// gpu_batch/01_norms_tp_transport.cuh: batched RMSNorm / residual kernels and the TP=2 K1/K2 doorbell transport kernels.
// Protocol and invariants: native/tp_doorbell.h.
// Section 1 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ---- batched RMSNorm (shared weight w[n]), one block per sequence column ----
// E9: programmatic dependent — publish its own completion edge and gate on the previous kernel
// (K2 tp_wait_add) BEFORE reading x. The two instructions are no-ops on the plain path (the
// implicit stream serialization already satisfied the dependency).
extern "C" __global__ void rmsnorm_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w, int n, int B, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    const __nv_bfloat16* xb = x + (long long)b * n;
    __nv_bfloat16* ob = out + (long long)b * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float v = b2f(xb[i]);
        sum_sq += v * v;
    }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += bs) {
        float v = b2f(xb[i]);
        ob[i] = f2b(v * inv * (1.0f + w[i]));
    }
}

// ---- fused: residual += mixer; out = rmsnorm(residual,w) per column ----
// E9: programmatic dependent (same prologue as rmsnorm_b — gate on K2 before reading residual).
extern "C" __global__ void fused_res_rmsnorm_b(__nv_bfloat16* out, __nv_bfloat16* residual, const __nv_bfloat16* mixer,
                                                const float* w, int n, int B, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long off = (long long)b * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float v = b2f(residual[off + i]) + b2f(mixer[off + i]);
        residual[off + i] = f2b(v);
        sum_sq += v * v;
    }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += bs) {
        float v = b2f(residual[off + i]);
        out[off + i] = f2b(v * inv * (1.0f + w[i]));
    }
}

// ---- F5 (EXPERT_FUSION_PASSES_RESPONSE §3.5): flavor-Q twin — BIT-EXACT to the two-kernel
// chain add_residual_b + rmsnorm_b. The shipped kernel above (flavor S) accumulates sum_sq
// from the UNROUNDED fp32 add; the two-kernel path's rmsnorm reads the bf16-ROUNDED residual,
// so only flavor Q reproduces it bit-for-bit (sum_sq over b2f(f2b(v))). QUALITY-mode sites
// (the FFN-side epilogue on both paths) use THIS kernel; the mixer post-norm site keeps the
// shipped flavor-S kernel (which is the default there today).
extern "C" __global__ void fused_res_rmsnorm_q_b(__nv_bfloat16* out, __nv_bfloat16* residual, const __nv_bfloat16* mixer,
                                                const float* w, int n, int B, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long off = (long long)b * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float v = b2f(residual[off + i]) + b2f(mixer[off + i]);
        __nv_bfloat16 r = f2b(v);
        residual[off + i] = r;
        float vr = b2f(r);
        sum_sq += vr * vr;
    }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    for (int i = tid; i < n; i += bs) {
        float v = b2f(residual[off + i]);
        out[off + i] = f2b(v * inv * (1.0f + w[i]));
    }
}

// ---- elementwise over n*B elements ----
// E9: programmatic dependent (gate on K2 before reading a — the MLP output is the barrier-reduced
// hidden; the residual add consumes it).
extern "C" __global__ void add_residual_b(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int total) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = f2b(b2f(a[i]) + b2f(b[i]));
}
extern "C" __global__ void silu_mul_b(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = f2b(silu_f(b2f(gate[i])) * b2f(up[i]));
}

// ================= TP=2 doorbell all-reduce — GPU side (K1/K2) =================
// Protocol + invariants: native/tp_doorbell.h. Two launches per barrier (down from four):
//   K1 tp_gate_copy_signal : wait tx_retired >= e-R (reuse gate, I3) -> copy the local partial into
//                            send[e%R] -> write the 8 B tail epoch at slot+align8(len) -> publish the
//                            generation-tagged length in len_local -> RELEASE gpu_ready = e (I1)
//   K2 tp_wait_add         : wait cpu_done >= e (I5) -> out = rank0_partial + rank1_partial
//
// CAPTURE HYGIENE (round-3): both kernels take the ctx POINTER and derive the epoch/slot on-device from
// c->epoch. No host-precomputed epoch or slot address is ever a kernel arg, so CUDA-graph capture is a
// no-op wrap of the decode sequence rather than a protocol rewrite.
//
// The old four-kernel handshake (tp_copy_bf16/tp_signal/tp_wait/tp_add_bf16) is gone: it had one slot
// (no ring, so no reuse discipline), used gpu_ready as the RDMA source for the epoch (post->DMA race),
// and polled the NIC-written epoch on the GPU — which is not sound here, because GB10 reports
// CAN_FLUSH_REMOTE_WRITES = 0, so NIC payload writes need not be GPU-visible when the flag is.

// Poll loops use plain (relaxed) loads plus backoff, never an atomic RMW (I6) — an RMW would ping-pong
// cache-line ownership across the C2C fabric that weights, NIC and CPU all share. One acquire fence is
// taken once the condition holds, which is what orders the payload reads that follow.
__device__ __forceinline__ unsigned long long tp_ld_relaxed(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void tp_st_release(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void tp_fence_acquire() {
    asm volatile("fence.acquire.sys;" ::: "memory");
}
__device__ __forceinline__ unsigned long long tp_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned long long* tp_flag(tp_dev_ctx* c, int byte_off) {
    return (unsigned long long*)((char*)c->flags + byte_off);
}
__device__ __forceinline__ void tp_stamp(tp_dev_ctx* c, unsigned long long e, int slot) {
    if (c->gpu_ts) c->gpu_ts[(e % TP_GTS_EPOCHS) * TP_GTS_STRIDE + slot] = tp_globaltimer();
}

// Wait until *p >= tgt. Returns 0 on success, 1 if the cooperative abort status went nonzero (I9).
// `critical` picks the backoff shape (R2d): the cpu_done wait sits on the wire-RTT critical path, so it
// spins tight for ~2 us before backing off, capped at 512 ns — a 2 us sleep quantum there was a large
// fraction of the barrier floor. The reuse gate is rarely hit and can afford the lazy 2 us cap.
// Thread 0 does the polling (single spinner); __syncthreads() then orders every other thread's payload
// access after t0's acquire, and broadcasts the abort verdict without a second flag read.
__device__ __forceinline__ int tp_spin_until_ge(tp_dev_ctx* c, const unsigned long long* p,
                                                unsigned long long tgt, int critical) {
    __shared__ int s_abort;
    if (threadIdx.x == 0) {
        s_abort = 0;
        const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
        unsigned long long tight_until = critical ? tp_globaltimer() + 2000ull : 0ull;
        unsigned ns = 64, cap = critical ? 512u : 2048u;
        while (tp_ld_relaxed(p) < tgt) {
            if (tp_ld_relaxed(ab)) { s_abort = 1; break; }
            if (critical && tp_globaltimer() < tight_until) continue;   // tight spin, no sleep
            __nanosleep(ns);
            if (ns < cap) ns <<= 1;
        }
        if (!s_abort) tp_fence_acquire();
    }
    __syncthreads();
    return s_abort;
}

// ---- N-way (world > 2) round/partner resolution, derived from the DEVICE epoch (capture-safe). ----
// world==2 leaves these unused: the single-QP arm below is byte-for-byte the pre-P3 K1/K2.
__device__ __forceinline__ int tp_rounds_of(tp_dev_ctx* c) { return (int)c->rounds; }
__device__ __forceinline__ int tp_round_of(tp_dev_ctx* c, unsigned long long e) {
    return (int)(e % (unsigned)tp_rounds_of(c));
}
__device__ __forceinline__ int tp_partner_of(tp_dev_ctx* c, unsigned long long e) {
    return (int)c->rank ^ (1 << tp_round_of(c, e));
}
// Per-peer flag entry: peer_committed / tx_retired / cpu_done, each on its own 64 B line, indexed by
// PEER RANK (see tp_doorbell.h TP_NWAY_*). The base abort word (TP_F_ABORT) stays in the shared block.
__device__ __forceinline__ unsigned long long* tp_nway_flagp(tp_dev_ctx* c, int peer, size_t sub_off) {
    return (unsigned long long*)((char*)c->nway_flags + (size_t)peer * TP_CL + sub_off);
}
// The recv-ring slot for epoch e. world==2: the LEGACY shared ring, recv_ring + (e%R)*stride
// (byte-identical). world>2: round(e)'s OWN recv ring, nway_recv + (round(e)*R + e%R)*stride — so a
// round's payload can never alias another round's slot (the cross-round staleness fix; round(e) =
// e % c->rounds, computed identically on every rank from the device epoch).
__device__ __forceinline__ unsigned char* tp_recv_slot_ptr(tp_dev_ctx* c, unsigned long long e) {
    if (c->world > 2) {
        unsigned long long round = e % (unsigned long long)c->rounds;
        return c->nway_recv + (round * (unsigned long long)TP_RING_SLOTS
                               + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride;
    }
    return c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
// The peer's inline-commit hint line for the v2/K2' stage-A spin. world==2: the shared
// TP_F_PEER_COMMITTED (byte-identical). world>2: partner(e)'s per-peer peer_committed line — the
// shared TP_F_PEER_COMMITTED word is never written at world>2 (the commit WR targets the per-peer
// region), so reading it would spin on a permanently-zero flag.
__device__ __forceinline__ const unsigned long long* tp_peer_committed_ptr(tp_dev_ctx* c,
                                                                            unsigned long long e) {
    return (c->world > 2)
        ? tp_nway_flagp(c, tp_partner_of(c, e), TP_NWAY_PEER_OFF)
        : tp_flag(c, TP_F_PEER_COMMITTED);
}

// K1 — reuse gate, copy the local partial into the send ring, publish the watermark.
// `src` is the raw local partial (bf16 or fp32 — the copy is byte-agnostic, payload_bytes words of 4).
// `nbytes` = how much of the slot this barrier actually carries (0 = the default c->payload_bytes).
// The all-reduce is CHUNKED: a reduction larger than one ring slot (prefill reduces hidden * prompt_len,
// which is unbounded) is split into several barriers, and chunks are sized to nearly fill a slot, so the
// length VARIES per epoch (last chunk of a reduction, decode-sized barriers). The tail epoch therefore
// sits at align8(len), not at a fixed offset, and K1 publishes the generation-tagged length into
// len_local BEFORE releasing gpu_ready (the same release orders both) so the proxy can size the RDMA
// write and the peer can find the tail. A fixed length per call site is capture-safe — it is the EPOCH
// and the slot address that must never be kernel arguments, not the payload size.
extern "C" __global__ void tp_gate_copy_signal(tp_dev_ctx* c, const unsigned int* src, unsigned int nbytes) {
    if (threadIdx.x == 0) c->epoch += 1;      // single block on a single stream: no race
    __syncthreads();
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K1_IN);

    // P3-1/expert Change 2: record the QP conflict set for THIS epoch so the reuse gate at epoch
    // e+R can wait on exactly Q(e) instead of a superset. Bit p set iff this epoch's commit was
    // posted to peer p's QP: tree epoch -> just partner(e)'s bit; one-shot -> all peers' bits.
    // Derived from the same unified predicate the proxy uses (wire_len <= payload_bytes), so every
    // rank computes the same mask. world==2 has no qp_mask (single-QP arm never reads it).
    const unsigned this_len = nbytes ? nbytes : c->payload_bytes;
    if (c->world > 2 && c->qp_mask && threadIdx.x == 0) {
        unsigned mask;
        if (c->oneshot && this_len <= c->payload_bytes) {
            mask = ((1u << c->world) - 1u) & ~(1u << c->rank);   // all peers
        } else {
            mask = 1u << tp_partner_of(c, e);                     // tree partner only
        }
        c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = mask;
    }

    // I3 reuse gate: do not overwrite send[s] until the WR that shipped it R barriers ago has retired.
    // Only meaningful once the ring has wrapped once.
    // N-way (world>2): the slot at epoch e-R was shipped by partner(e-R)'s QP, so the reuse gate waits on
    // THAT peer's per-QP tx_retired (NOT the current partner — a different QP with independent in-order
    // retirement). world==2: partner(e-R) == rank^1 == the single peer -> tx_retired (byte-identical).
    if (e > TP_RING_SLOTS) {
        // P3-1/expert Change 2: gate on EXACTLY the QPs that epoch e-R was posted to (read from the
        // qp_mask ring K1 wrote at epoch e-R). Tree e-R -> its single partner's QP; one-shot e-R ->
        // all 3 peers' QPs. The old one-shot arm waited on ALL peers regardless of Q(e-R), which
        // demanded credit from QPs the conflicting epoch never touched — with the per-QP tx_retired
        // only advancing on that QP's own (now always-signaled) CQEs, a no-traffic QP's watermark
        // could sit below e-R forever under pattern-locked mixed traffic. The target stays e-R; only
        // the WAIT SET narrows to the true conflict set.
        const unsigned long long tgt = e - TP_RING_SLOTS;
        const unsigned qmask = (c->world > 2 && c->qp_mask)
            ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)]
            : 0u;
        if (c->oneshot && qmask && this_len <= c->payload_bytes) {
            if (threadIdx.x == 0 && [&]() {
                const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
                for (int p = 0; p < (int)c->world; p++) {
                    if (!(qmask & (1u << p))) continue;   // only the QPs e-R was posted to
                    const unsigned long long* ret = tp_nway_flagp(c, p, TP_NWAY_TX_OFF);
                    if (tp_ld_relaxed(ret) < tgt) c->gate_waits += 1;
                    unsigned long long deadline = tp_globaltimer() + K1_GATE_WAIT_NS;
                    unsigned ns = 64, cap = 2048u;
                    while (tp_ld_relaxed(ret) < tgt) {
                        if (tp_ld_relaxed(ab)) return 1;
                        if (tp_globaltimer() >= deadline) {
                            tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                            printf("[K1-gate-timeout/oneshot] rank=%d epoch=%llu peer=%d tgt=%llu qmask=%x\n",
                                   c->rank, e, p, (unsigned long long)tgt, qmask);
                            return 1;
                        }
                        __nanosleep(ns);
                        if (ns < cap) ns <<= 1;
                    }
                }
                return 0;
            }()) { /* aborted or timed out: no-op through the stream */ }
            __syncthreads();
            if (tp_ld_relaxed(tp_flag(c, TP_F_ABORT))) return;
            tp_fence_acquire();
            goto copy_payload;
        }
        const unsigned long long* ret = (c->world > 2)
            ? tp_nway_flagp(c, tp_partner_of(c, e - TP_RING_SLOTS), TP_NWAY_TX_OFF)
            : tp_flag(c, TP_F_TX_RETIRED);
        // Count the times the gate ACTUALLY binds. Without this the bench can only assume it was
        // exercised — and the rendezvous bounds skew to ~1 barrier, so under normal traffic it never
        // binds and the gate would ship untested (see the cq_hold hook in net_shim.c).
        if (threadIdx.x == 0 && tp_ld_relaxed(ret) < e - TP_RING_SLOTS)
            c->gate_waits += 1;
        // Bounded gate (freeze hardening): a stalled send DMA retires no CQE, so tx_retired never
        // advances and this spin would bind forever (the 10 s watchdog on the skew-1 side was the
        // only bound, and it produced a SILENT abort in the acceptance gates). On expiry, publish
        // device status 11 (the same cooperative status the K2' tail deadline uses, I9) and no-op —
        // both ranks see it at their next flag poll and the host sees a loud abort.
        __shared__ int s_gate_to;
        if (threadIdx.x == 0) {
            s_gate_to = 0;
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            const unsigned long long tgt = e - TP_RING_SLOTS;
            unsigned long long deadline = tp_globaltimer() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (tp_ld_relaxed(ret) < tgt) {
                if (tp_ld_relaxed(ab)) { s_gate_to = 1; break; }   // someone else aborted — no-op
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    // DIAG: the stalled epoch + the tx_retired value the gate was waiting on. The
                    // rounds count distinguishes an epoch/round drift from a pure latency stall.
                    printf("[K1-gate-timeout] rank=%d epoch=%llu rounds=%u tgt=%llu txret=%llu\n",
                           c->rank, e, c->rounds, tgt, tp_ld_relaxed(ret));
                    s_gate_to = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_gate_to) tp_fence_acquire();
        }
        __syncthreads();
        if (s_gate_to) return;                     // I9: no-op through the stream
    }

copy_payload:                                       // P3-1 one-shot gate jumps here (gates passed)
    const unsigned len = nbytes ? nbytes : c->payload_bytes;
    const unsigned wire = (len + 7u) & ~7u;         // tail lands on an 8 B boundary (<= 4 B pad)
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    unsigned int* dst = (unsigned int*)slot;
    const unsigned nw = len >> 2;
    for (unsigned i = threadIdx.x; i < nw; i += blockDim.x) dst[i] = src[i];

    __syncthreads();
    if (threadIdx.x == 0) {
        // Tail-epoch guard (R2d): written LAST, and shipped as the trailing 8 B of the same RDMA write,
        // so the peer proxy can prove placement order held before it releases its GPU.
        *(unsigned long long*)(slot + wire) = e;
        // Per-epoch length publish: the proxy reads this after acquiring gpu_ready >= e (the release
        // below orders it), tags it onto the wire ahead of the payload, and the peer's RECV path
        // bounded-waits on the epoch bits before it can even locate the tail.
        c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
        tp_st_release(tp_flag(c, TP_F_GPU_READY), e);   // watermark publish (I1) — release orders the above
        tp_stamp(c, e, TP_GTS_K1_OUT);
    }
}

// K2 — wait for the CPU-bounced release, then reduce. Canonical rank0 + rank1 add order on BOTH ranks
// (round-3 R3c): cross-rank bit-identity is automatic since IEEE add commutes, but fixing the order also
// removes NaN-payload and signed-zero ambiguity, for free.
//   fp32_mode = 0 : `local` and `out` are the same bf16 buffer; peer partial is bf16; fp32 accumulate, one round.
//   fp32_mode = 1 : `local` is the GEMV's fp32 accumulator, peer partial is fp32, `out` is bf16 — the
//                   FP32-preserving production path (a single rounding boundary for the whole reduction).
//   fp32_mode = 2 : `local` and `out` are the same FP32 buffer; peer partial is fp32; NO rounding —
//                   the vocab-parallel LM-head gather (each rank's rows are exact +0.0 for the peer,
//                   so the sum IS the gather and stays bitwise-identical to the replicated head).
// The mode is a PER-BARRIER kernel argument (constant per call site, so capture-safe — the round-3 rule
// forbids only epoch/slot args): decode/verify reductions run FP32-preserving while wide prefill
// reductions stay bf16-chunked, in the same run, on one ring. (A global mode in the ctx made that
// mix impossible: K2 would read bf16 prefill partials as fp32.)
// The three reduce bodies shared verbatim by tp_wait_add (v1, CPU-bounced gate) and
// tp_wait_add_g (v2, GPU-direct two-stage gate) — the arithmetic exists exactly once in the
// tree (EXPERT_GPU_ALLREDUCE §3.2/§4). `peer` = the recv-ring slot for epoch e; the mode is a
// per-barrier constant as documented on tp_wait_add.
__device__ __forceinline__ void tp_reduce_body(tp_dev_ctx* c, __nv_bfloat16* out,
                                               const void* local, const unsigned char* peer,
                                               int n, int fp32_mode) {
    // Canonical reduction order, rank-identical on every rank. world==2: rank0 + rank1 (round-0
    // instance of the rule). world>2: round k adds (lower-half partial) + (upper-half partial), where
    // "lower" is the rank whose bit k is 0 — both partners compute the SAME sum in the SAME order, so
    // the FP result telescopes to a globally identical value across all N ranks (IEEE add is
    // non-associative; fixing the ORDER, not just commutativity, is what makes it bit-identical).
    const unsigned long long ce = c->epoch;
    const int lower = (c->world > 2)
        ? ((c->rank & (1 << tp_round_of(c, ce))) == 0)
        : (c->rank == 0);
    if (fp32_mode == 2) {
        // f32 -> f32, no rounding boundary: the vocab-parallel head gather. `out` aliases `local`.
        float* of = (float*)out;
        const float* lo = (const float*)local;
        const float* pe = (const float*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = lower ? lo[i] : pe[i];
            float b = lower ? pe[i] : lo[i];
            of[i] = a + b;
        }
    } else if (fp32_mode == 3) {
        // R9 (world>2 only): bf16 local partial + bf16 peer partial, summed in fp32 into a FRESH fp32
        // `out` (NOT the bf16 send buffer). The recursive-doubling intermediate stays fp32 across
        // rounds — no in-place bf16 re-round of the running sum (the world=2 mode-0 path's rounding
        // is bit-identical only because there is exactly one exchange). The final round's caller
        // rounds this fp32 accumulator to bf16 once. `out` never aliases `local` here.
        float* of = (float*)out;
        const __nv_bfloat16* lo = (const __nv_bfloat16*)local;
        const __nv_bfloat16* pe = (const __nv_bfloat16*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = b2f(lower ? lo[i] : pe[i]);
            float b = b2f(lower ? pe[i] : lo[i]);
            of[i] = a + b;
        }
    } else if (fp32_mode) {
        const float* lo = (const float*)local;
        const float* pe = (const float*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = lower ? lo[i] : pe[i];   // lower half's partial
            float b = lower ? pe[i] : lo[i];   // upper half's partial
            out[i] = f2b(a + b);
        }
    } else {
        const __nv_bfloat16* lo = (const __nv_bfloat16*)local;
        const __nv_bfloat16* pe = (const __nv_bfloat16*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = b2f(lower ? lo[i] : pe[i]);
            float b = b2f(lower ? pe[i] : lo[i]);
            out[i] = f2b(a + b);
        }
    }
}

// K2 — wait for the CPU-bounced release, then reduce. Canonical rank0 + rank1 add order on BOTH ranks
// (round-3 R3c): cross-rank bit-identity is automatic since IEEE add commutes, but fixing the order also
// removes NaN-payload and signed-zero ambiguity, for free.
//   fp32_mode = 0 : `local` and `out` are the same bf16 buffer; peer partial is bf16; fp32 accumulate, one round.
//   fp32_mode = 1 : `local` is the GEMV's fp32 accumulator, peer partial is fp32, `out` is bf16 — the
//                   FP32-preserving production path (a single rounding boundary for the whole reduction).
//   fp32_mode = 2 : `local` and `out` are the same FP32 buffer; peer partial is fp32; NO rounding —
//                   the vocab-parallel LM-head gather (each rank's rows are exact +0.0 for the peer,
//                   so the sum IS the gather and stays bitwise-identical to the replicated head).
// The mode is a PER-BARRIER kernel argument (constant per call site, so capture-safe — the round-3 rule
// forbids only epoch/slot args): decode/verify reductions run FP32-preserving while wide prefill
// reductions stay bf16-chunked, in the same run, on one ring. (A global mode in the ctx made that
// mix impossible: K2 would read bf16 prefill partials as fp32.)
extern "C" __global__ void tp_wait_add(tp_dev_ctx* c, __nv_bfloat16* out, const void* local, int n,
                                       int fp32_mode) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);

    // E9: publish the launch-completion edge BEFORE the spin — the programmatic dependents
    // (rmsnorm_b / fused_res_rmsnorm_b / add_residual_b / the pdl GEMM, launched with the
    // stream-serialization attribute) become resident and run their weight-prefetch preambles
    // while this kernel still waits for the peer's partial. When E9 is off the attribute is never
    // set, so no dependent is resident early and this instruction is a no-op — the plain barrier
    // path is behaviorally byte-for-byte.
    asm volatile("griddepcontrol.launch_dependents;");

    // I5 gate: world==2 waits on the single cpu_done; world>2 waits on partner(e)'s per-peer cpu_done
    // (the proxy released THAT peer's line after the fence). abort => no-op (I9).
    const unsigned long long* done = (c->world > 2)
        ? tp_nway_flagp(c, tp_partner_of(c, e), TP_NWAY_CPU_OFF)
        : tp_flag(c, TP_F_CPU_DONE);
    if (tp_spin_until_ge(c, done, e, 1)) return;
    tp_stamp(c, e, TP_GTS_K2_GO);

    const unsigned char* peer = tp_recv_slot_ptr(c, e);
    // R9 (world>2): the cpu_done `>=` gate is a per-partner WATERMARK, so it can pass for epoch e
    // before e's own slot is the validated one (a backlog drain releasing a later epoch of the same
    // partner). Validate the slot's generation-tagged TAIL by equality before consuming — the exact
    // placement proof the v2 GPU-direct path already relies on. mode 3 ships bf16 (2 B/elem).
    if (c->world > 2) {
        const unsigned len = n * ((fp32_mode == 1 || fp32_mode == 2) ? 4u : 2u);
        const unsigned wire = (len + 7u) & ~7u;
        __shared__ int s_tail;
        if (threadIdx.x == 0) {
            s_tail = 0;
            const unsigned long long* tail = (const unsigned long long*)(peer + wire);
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            // B8 §1.5-3: adaptive stage-B deadline — wide (slot-filling) epochs get the 5 s bound,
            // decode-width keep the 1 ms fast path. Numerics-neutral (wait bound only).
            unsigned long long deadline = tp_globaltimer()
                + (wire > c->payload_bytes ? TP_TAIL_LONG_WAIT_NS : TP_TAIL_WAIT_NS);
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(tail) != e) {
                if (tp_ld_relaxed(ab)) { s_tail = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    printf("[K2-tail-timeout] rank=%d epoch=%llu wire=%u fp32_mode=%d\n",
                           c->rank, e, wire, fp32_mode);
                    s_tail = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_tail) tp_fence_acquire();
        }
        __syncthreads();
        if (s_tail) return;                    // I9: no-op through the stream
    }
    tp_reduce_body(c, out, local, peer, n, fp32_mode);
}

// K2' — v2 receive (EXPERT_GPU_ALLREDUCE §3.2): the GPU waits directly on the NIC-written
// payload's generation-tagged tail instead of the CPU-bounced `cpu_done`. The proxy still posts
// the wire (send side is CPU-only) and validates nothing on the receive path when v2 is active.
//   Stage A — non-trusted hint: spin until peer_committed >= e (the inline commit's epoch; the
//   ~4.36M incident proved a bare flag is NOT placement proof, so this only arms the deadline).
//   Stage B — the commit: spin until *(u64*)(recv_slot + wire) == e — the payload tail, the same
//   generation-tagged value the CPU receiver validates today. Deadline TP_TAIL_WAIT_NS from
//   stage-A completion (the payload was posted before the commit on a reliable QP, so a tail
//   lagging the commit past that is lost, not slow): write device status 11 to TP_F_ABORT and
//   no-op through the stream (I9) — the host sync observes the abort and re-inits (I8).
//   The length is DERIVED (n × elem_bytes, align8), never read from len_peer — the chunking is a
//   pure SPMD function both ranks compute identically.
//   After consumption: publish the rx_done watermark (the watchdog's v2 debt signal).
extern "C" __global__ void tp_wait_add_g(tp_dev_ctx* c, __nv_bfloat16* out, const void* local, int n,
                                         int fp32_mode) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);

    // E9 edge BEFORE the spins, exactly as tp_wait_add (C8 — a fused successor must keep it).
    asm volatile("griddepcontrol.launch_dependents;");

    // Stage A: the peer's inline commit hint (non-trusted, arms the deadline). No deadline of its
    // own — a slow peer is legitimate; the 10 s watchdog bounds it.
    if (tp_spin_until_ge(c, tp_peer_committed_ptr(c, e), e, 1)) return;

    // Stage B: the payload tail — the actual placement proof. Deadline-armed from stage-A completion.
    // mode 3 (R9) ships a BF16 payload (same 2 B/elem as mode 0) even though it accumulates fp32.
    const unsigned len = n * ((fp32_mode == 1 || fp32_mode == 2) ? 4u : 2u);
    const unsigned wire = (len + 7u) & ~7u;
    const unsigned char* peer = tp_recv_slot_ptr(c, e);
    // N-way receive-side clean (world>2 only): a previous occupant of this round-keyed slot left
    // its own generation-tagged tail behind. A stale tail can never EQUAL e (the prior generation
    // is e - rounds*R), but clear it defensively so no torn/early word from a past generation can
    // satisfy the stage-B equality for THIS epoch. Guarded `!= e`: if the fresh tail already
    // landed (the peer posted ahead of us — legitimate), leave it untouched; erasing it would
    // deadlock stage B. world==2 keeps the legacy single-ring path byte-identical.
    if (c->world > 2 && threadIdx.x == 0) {
        volatile unsigned long long* tw = (volatile unsigned long long*)(peer + wire);
        if (tp_ld_relaxed((const unsigned long long*)tw) != e) *tw = 0ull;
    }
    __syncthreads();
    __shared__ int s_timeout;
    if (threadIdx.x == 0) {
        s_timeout = 0;
        const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
        const unsigned long long* tail =
            (const unsigned long long*)(peer + wire);
        // B8 §1.5-3: adaptive stage-B deadline — wide (slot-filling) epochs get the 5 s bound,
        // decode-width keep the 1 ms fast path. Numerics-neutral (wait bound only).
        unsigned long long deadline = tp_globaltimer()
            + (wire > c->payload_bytes ? TP_TAIL_LONG_WAIT_NS : TP_TAIL_WAIT_NS);
        unsigned ns = 64, cap = 512u;
        while (tp_ld_relaxed(tail) != e) {
            if (tp_ld_relaxed(ab)) { s_timeout = 1; break; }   // someone else aborted — no-op
            if (tp_globaltimer() >= deadline) {
                // Tail never landed: the commit was observed but the payload is lost (the 4.36M
                // class). Cooperative abort with the device status code — the host re-inits (I8).
                tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                printf("[K2-stageB-timeout] rank=%d epoch=%llu wire=%u fp32_mode=%d\n",
                       c->rank, e, wire, fp32_mode);
                s_timeout = 1;
                break;
            }
            __nanosleep(ns);
            if (ns < cap) ns <<= 1;
        }
        if (!s_timeout) tp_fence_acquire();
    }
    __syncthreads();
    if (s_timeout) return;                     // I9: no-op through the stream
    tp_stamp(c, e, TP_GTS_K2_GO);

    tp_reduce_body(c, out, local, peer, n, fp32_mode);
    __syncthreads();
    if (threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);   // consumed watermark
}

// K2 one-shot (P3-1, world==4 only): ALL-PEERS exchange in a single serialized hop. Each rank's
// K1 published its partial to all 3 peers (the proxy posts the same 3-WR chain to every peer QP,
// payload landing in SENDER-indexed recv rings — layout selector c->rounds == 4). This kernel
// waits for all 3 inbound tails (stage A: per-peer committed hints arm the deadlines; stage B:
// per-sender-slot tails by equality — the placement proof), fences, then reduces the 4 partials
// in CANONICAL ABSOLUTE RANK ORDER ((p0+p1)+p2)+p3, fp32 accumulate, ONE bf16 rounding — every
// rank reads the same 4 buffers in the same order and computes the same fp32 telescoping sum, so
// cross-rank bit-identity holds by construction (the same discipline as tp_reduce_body).
// Abort discipline identical to tp_wait_add_g (I9 no-op through the stream, device status 11).
// The length is DERIVED (n × 2 for the bf16 wire), matching the sender's tail by SPMD
// construction. Only valid when c->rounds == 4 (the sender-indexed layout); the doubling tree
// keeps tp_wait_add_g otherwise.
extern "C" __global__ void tp_wait_add_4way(tp_dev_ctx* c, __nv_bfloat16* out, const void* local, int n,
                                            int fp32_mode) {
    const unsigned long long e = c->epoch;
    tp_stamp(c, e, TP_GTS_K2_IN);

    asm volatile("griddepcontrol.launch_dependents;");   // E9 edge, exactly as tp_wait_add_g

    const unsigned len = n * 2u;            // one-shot ships the bf16 partial (2 B/elem)
    const unsigned wire = (len + 7u) & ~7u;
    const unsigned long long* ab = tp_flag(c, TP_F_ABORT);

    // Stage A: all 3 peers' commit hints >= e. The hint is NON-TRUSTED (I5): it only arms the
    // stage-B short deadline. P3-1 fix (b): give each hint a bounded TP_HINT_WAIT_NS deadline, but
    // on expiry DON'T abort — stop waiting on the hint and let the stage-B tail poll prove
    // placement (a slow/lost hint is legitimate; the payload tail is the real proof). A peer whose
    // hint never arrived simply forces this rank into stage B with the longer tail deadline. The
    // ONLY abort path in stage A is a peer's abort flag (cooperative, I9).
    __shared__ int s_any_timeout;
    if (threadIdx.x == 0) {
        s_any_timeout = 0;
        for (int p = 0; p < (int)c->world; p++) {
            if (p == (int)c->rank) continue;
            const unsigned long long* hint = tp_nway_flagp(c, p, TP_NWAY_PEER_OFF);
            unsigned long long deadline = tp_globaltimer() + TP_HINT_WAIT_NS;
            unsigned long long tight_until = tp_globaltimer() + 2000ull;  // tight spin first (µs-scale arrivals)
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(hint) < e) {
                if (tp_ld_relaxed(ab)) { s_any_timeout = 1; break; }   // someone aborted — no-op
                if (tp_globaltimer() >= deadline) break;              // hint lost/slow: stage B proves placement
                if (tp_globaltimer() < tight_until) continue;         // tight spin, no sleep
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (s_any_timeout) break;
        }
    }
    __syncthreads();
    if (s_any_timeout) return;              // I9 no-op (a peer aborted)

    // Stage B: each sender's slot tail == e (the placement proof), deadline-armed per slot. If the
    // hint never arrived the tail may still be legitimately in flight, so use the longer
    // TP_TAIL_LONG_WAIT_NS (the same bound the K1 reuse gate tolerates); only a REAL tail stall
    // past that is a lost payload -> status 11 (I9).
    if (threadIdx.x == 0) {
        for (int p = 0; p < (int)c->world; p++) {
            if (p == (int)c->rank) continue;
            // sender-indexed slot: ring p, slot e % R — the sender-indexed layout selector
            const unsigned char* slot = c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride;
            const unsigned long long* tail = (const unsigned long long*)(slot + wire);
            // receive-side clean, guarded != e (identical discipline to tp_wait_add_g)
            if (tp_ld_relaxed(tail) != e) *(volatile unsigned long long*)(slot + wire) = 0ull;
            unsigned long long deadline = tp_globaltimer() + TP_TAIL_LONG_WAIT_NS;
            unsigned long long tight_until = tp_globaltimer() + 4000ull;  // tight spin: arrivals are µs-scale
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed((const unsigned long long*)(slot + wire)) != e) {
                if (tp_ld_relaxed(ab)) { s_any_timeout = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    printf("[4way-stageB-timeout] rank=%d epoch=%llu peer=%d\n", c->rank, e, p);
                    s_any_timeout = 1;
                    break;
                }
                if (tp_globaltimer() < tight_until) continue;           // tight spin, no sleep
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (s_any_timeout) break;
        }
        if (!s_any_timeout) tp_fence_acquire();
    }
    __syncthreads();
    if (s_any_timeout) return;
    tp_stamp(c, e, TP_GTS_K2_GO);

    // Canonical 4-way reduce: ((p0+p1)+p2)+p3 in fp32, one bf16 rounding. `local` is THIS rank's
    // partial (rank r's element value == what sits in every peer's ring r). Modes 1/2 (fp32
    // partials) are NOT valid here (the one-shot ships bf16 only) — callers use mode 0/3 class.
    const __nv_bfloat16* parts[4];
    for (int p = 0; p < 4; p++) {
        parts[p] = (p == (int)c->rank)
            ? (const __nv_bfloat16*)local
            : (const __nv_bfloat16*)(c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride);
    }
    if (fp32_mode == 2) {
        // mode 2: fp32 out, no rounding (the LM-head gather class) — partials are still bf16 on
        // the wire; upconvert each and sum in canonical order.
        float* of = (float*)out;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float acc = b2f(parts[0][i]);
            for (int p = 1; p < (int)c->world; p++) acc += b2f(parts[p][i]);
            of[i] = acc;
        }
    } else {
        // mode 0/3 class: bf16 out, fp32 accumulate, ONE rounding.
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float acc = b2f(parts[0][i]);
            for (int p = 1; p < (int)c->world; p++) acc += b2f(parts[p][i]);
            out[i] = f2b(acc);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);   // consumed watermark
}

// K1 one-shot, MULTI-BLOCK (NAIVE lever 1c, the uniform one-shot ctx only): the single-block tp_gate_copy_signal
// spends ~14 us per decode reduce copying 16 KB into the mapped host send ring from ONE block with 4 B stores (the
// 2026-10-20 TP4 census: 105 launches per token = 1.5 ms of a 22 ms step, on every rank, on the critical path of
// every reduce). Same protocol, same bytes, same order of publication: every block waits the I3 reuse gate on
// exactly the QPs epoch e-R was posted to (the qp_mask ring; a uniform ctx posts every epoch to all peers),
// copies its grid-stride share with 16 B stores, fences at system scope and arrives on a device counter; the LAST
// block records this epoch's qp_mask (all peers), writes the generation-tagged tail LAST, publishes the len tag
// and releases gpu_ready (I1) -- K1's publish sequence after ALL blocks' payload stores. c->epoch is advanced by
// the last block (every block computed e from the pre-kernel value), so the K2 that follows in stream order reads
// the new epoch. nbytes % 16 == 0 (the host passes fp32 payloads of n % 4 == 0 floats). Bitwise-neutral: the wire
// bytes and their placement are those of K1.
extern "C" __global__ void tp_gate_copy_signal_os_mb(tp_dev_ctx* c, const void* src, unsigned nbytes, unsigned* arrive) {
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;              // only the QPs e-R was posted to
                const unsigned long long* ret = tp_nway_flagp(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && tp_ld_relaxed(ret) < tgt) c->gate_waits += 1;
                unsigned long long deadline = tp_globaltimer() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (tp_ld_relaxed(ret) < tgt) {
                    if (tp_ld_relaxed(ab)) { s_ab = 1; break; }
                    if (tp_globaltimer() >= deadline) {
                        tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                        if (blockIdx.x == 0)
                            printf("[K1os-mb-gate-timeout] rank=%d epoch=%llu peer=%d tgt=%llu qmask=%x\n",
                                   c->rank, e, p, (unsigned long long)tgt, qmask);
                        s_ab = 1;
                        break;
                    }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) tp_fence_acquire();
        }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        const unsigned n16 = nbytes >> 4;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = s4[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's payload stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // the next launch is stream-ordered behind us
            c->epoch = e;                               // advances even on abort, like K1 (I9 no-op)
            if (c->world > 2 && c->qp_mask)             // a one-shot epoch: posted to ALL peers' QPs
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!tp_ld_relaxed(tp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;                 // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                tp_st_release(tp_flag(c, TP_F_GPU_READY), e);            // watermark (I1)
            }
        }
    }
}

// K2 one-shot, fp32 wire (NAIVE lever 1, `--tp-reduce single` on the Naive TP=4 path): the all-peers single
// hop of a UNIFORM one-shot ctx (net_shim oneshot_uniform), gated exactly like the EXL3 path's
// xq_tp_wait_add_dec_single: per peer, the proxy's validated RELEASE `cpu_done[p] >= e` (the proxy checked
// that peer's commit hint, len tag and generation-tagged tail at the sender-indexed slot, then fenced and
// released) AND the slot's tail == e (the R9 watermark check: cpu_done is a per-peer watermark, so it can
// pass for e before e's own slot is the validated one). This kernel never writes a tail: on this ctx the
// proxy owns tail validation, and a GPU-side "receive-side clean" of a stale tail races the proxy's
// (possibly lagging) validation of the previous occupant of the slot — that race is exactly what the first
// served boot of this lever hit (TAIL-EPOCH GUARD (nway): tail=0 expected 1351, abort 2).
// The partials are fp32 on the wire (4 B/elem, what the Naive path reduces: MoE / attention partial sums,
// the sampler's exchange buffers, the drafter's hidden) and the result stays fp32, IN PLACE (`out` may alias
// `local`: every thread reads its own element before it writes it). The sum is formed as
//     out = (p0 + p2) + (p1 + p3)          p_r = rank r's fp32 partial (own = `local`)
// which is EXACTLY the association of the Naive path's R10-aligned recursive-doubling tree (the first
// K1 of a logical reduce pre-increments the epoch to an odd value -> round 1 -> partner rank^2, so stage 0
// pairs {0,2},{1,3}; stage 1 (partner rank^1) adds the two pair sums; fp32 add is commutative, so the order
// inside a pair -- which differs per rank -- cannot change a bit). Every rank reads the same four buffers
// and forms the same two pair sums, so the result is bit-identical across ranks AND bit-identical to the
// two-stage tree it replaces (tp_xport::single_tree_reference is the host-side statement of this).
// Only valid on a uniform one-shot ctx: EVERY epoch is an all-peers push, so K1's qp_mask and the proxy's
// dispatch agree by construction as long as n*4 <= payload_bytes -- the host enforces that bound and
// routes wider reduces to the tree on the primary ctx.
extern "C" __global__ void tp_wait_add_4way_f32(tp_dev_ctx* c, float* out, const float* local, int n) {
    const unsigned long long e = c->epoch;
    tp_stamp(c, e, TP_GTS_K2_IN);

    asm volatile("griddepcontrol.launch_dependents;");   // E9 edge, exactly as tp_wait_add_4way

    const unsigned len = (unsigned)n * 4u;  // fp32 partial (4 B/elem)
    const unsigned wire = (len + 7u) & ~7u;
    const unsigned long long* ab = tp_flag(c, TP_F_ABORT);

    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        for (int p = 0; p < (int)c->world && !s_ab; p++) {
            if (p == (int)c->rank) continue;
            const unsigned long long* done = tp_nway_flagp(c, p, TP_NWAY_CPU_OFF);
            const unsigned char* slot = c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride;
            const unsigned long long* tail = (const unsigned long long*)(slot + wire);
            unsigned long long tight_until = tp_globaltimer() + 2000ull;   // tight spin first (us-scale arrivals)
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(done) < e || tp_ld_relaxed(tail) != e) {
                if (tp_ld_relaxed(ab)) { s_ab = 1; break; }   // the proxy's guards abort a lost payload (I9)
                if (tp_globaltimer() < tight_until) continue;
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
        }
        if (!s_ab) tp_fence_acquire();
    }
    __syncthreads();
    if (s_ab) return;                       // I9 no-op through the stream
    tp_stamp(c, e, TP_GTS_K2_GO);

    const float* parts[4];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        parts[p] = (p == (int)c->rank)
            ? local
            : (const float*)(c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride);
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float a0 = parts[0][i], a1 = parts[1][i], a2 = parts[2][i], a3 = parts[3][i];
        out[i] = (a0 + a2) + (a1 + a3);
    }
}

// K2 one-shot, fp32 wire, GPU-VALIDATED receive (NAIVE lever 1b, `--tp-gpu-recv on` on the Naive single-stage
// ctx): the same reduce as tp_wait_add_4way_f32, but on a uniform one-shot ctx whose proxy runs in the v2 receive
// mode (net_set_recv_mode gpu=1: it posts the wire and never validates or releases on the receive side), so this
// kernel is the receive side -- exactly tp_wait_add_4way's discipline on an fp32 wire: stage A arms the deadlines
// on the 3 peers' non-trusted commit hints (I5; a lost / slow hint only forces the long stage-B bound), stage B
// proves placement by each sender-indexed slot's generation-tagged tail == e (a stale tail is cleaned first, guarded
// != e -- safe here because nothing on the host reads these tails in this mode), fence, reduce, publish rx_done
// (the watchdog's v2 debt signal). Removes the CPU bounce (proxy validation -> cpu_done release -> GPU) from every
// decode-width reduce; the sum and its association are the same as tp_wait_add_4way_f32 (bitwise the tree).
extern "C" __global__ void tp_wait_add_4way_f32g(tp_dev_ctx* c, float* out, const float* local, int n) {
    const unsigned long long e = c->epoch;
    tp_stamp(c, e, TP_GTS_K2_IN);

    asm volatile("griddepcontrol.launch_dependents;");   // E9 edge, exactly as tp_wait_add_4way

    const unsigned len = (unsigned)n * 4u;  // fp32 partial (4 B/elem)
    const unsigned wire = (len + 7u) & ~7u;
    const unsigned long long* ab = tp_flag(c, TP_F_ABORT);

    __shared__ int s_any_timeout;
    if (threadIdx.x == 0) {
        s_any_timeout = 0;
        for (int p = 0; p < (int)c->world; p++) {
            if (p == (int)c->rank) continue;
            const unsigned long long* hint = tp_nway_flagp(c, p, TP_NWAY_PEER_OFF);
            unsigned long long deadline = tp_globaltimer() + TP_HINT_WAIT_NS;
            unsigned long long tight_until = tp_globaltimer() + 2000ull;
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(hint) < e) {
                if (tp_ld_relaxed(ab)) { s_any_timeout = 1; break; }
                if (tp_globaltimer() >= deadline) break;              // hint lost/slow: stage B proves placement
                if (tp_globaltimer() < tight_until) continue;
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (s_any_timeout) break;
        }
    }
    __syncthreads();
    if (s_any_timeout) return;              // I9 no-op (a peer aborted)

    if (threadIdx.x == 0) {
        for (int p = 0; p < (int)c->world; p++) {
            if (p == (int)c->rank) continue;
            const unsigned char* slot = c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride;
            // NO receive-side clean here (tp_wait_add_4way / tp_wait_add_g store 0 over a tail != e "defensively"):
            // that store races the NIC's write of THIS epoch's tail -- load sees the old generation, the NIC lands
            // tail == e, the GPU's 0 lands last, and stage B waits 5 s on a tail that will never reappear. Seen
            // once in ~450K epochs on the served TP4 cluster (2026-10-20, epoch 170380, peer 2, abort 11). The
            // equality test needs no clean: a stale tail is a different generation (e - 8k) and can never equal e.
            unsigned long long deadline = tp_globaltimer() + TP_TAIL_LONG_WAIT_NS;
            unsigned long long tight_until = tp_globaltimer() + 4000ull;
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed((const unsigned long long*)(slot + wire)) != e) {
                if (tp_ld_relaxed(ab)) { s_any_timeout = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    printf("[4way-f32g-stageB-timeout] rank=%d epoch=%llu peer=%d\n", c->rank, e, p);
                    s_any_timeout = 1;
                    break;
                }
                if (tp_globaltimer() < tight_until) continue;
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (s_any_timeout) break;
        }
        if (!s_any_timeout) tp_fence_acquire();
    }
    __syncthreads();
    if (s_any_timeout) return;
    tp_stamp(c, e, TP_GTS_K2_GO);

    const float* parts[4];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        parts[p] = (p == (int)c->rank)
            ? local
            : (const float*)(c->oneshot_recv
                + ((unsigned long long)p * (unsigned long long)TP_RING_SLOTS
                   + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride);
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float a0 = parts[0][i], a1 = parts[1][i], a2 = parts[2][i], a3 = parts[3][i];
        out[i] = (a0 + a2) + (a1 + a3);
    }
    __syncthreads();
    if (threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);   // consumed watermark (the v2 debt signal)
}

