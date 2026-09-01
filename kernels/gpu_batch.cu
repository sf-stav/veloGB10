// Batched (continuous-batching) kernels for Qwen3.5-0.8B decode.
// ALL activations are bf16 (__nv_bfloat16) — eliminates f32↔bf16 conversion kernels.
// Internal computation is f32. State arrays (KV cache, conv, recurrent) stay f32.
// Activations are column-major [feat, B] (B sequences = B columns; seq j at offset j*feat).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "tp_doorbell.h"

// ---- Build-ID stamp: makes a stale PTX impossible to run silently ----
// build.rs hashes the .cu bytes and passes the result as -DKERNEL_BUILD_ID. GpuModel::load reads this
// global back out of the loaded module and asserts it equals the ID compiled into the BINARY. A fresh
// binary next to old kernels then fails loudly at startup instead of launching a kernel whose ABI it no
// longer agrees with -- which is how we once got CUDA_ERROR_ILLEGAL_ADDRESS out of correct code.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }


__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }

// Attention head-dim envelope: the decode/prefill attention kernels take hd as a runtime argument and
// size their per-lane register slices for hd/32 <= SK_DPL_MAX, i.e. any hd that is a positive multiple
// of 32 up to SK_HD_MAX (qwen3.5: 256; hy_v3: 128; DeepSeek: 512). A bigger hd needs only these raised.
#define SK_HD_MAX 512
#define SK_DPL_MAX (SK_HD_MAX / 32)

#define GRID1(total) ((int)(((total) + 255) / 256))

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

// K2'' — AR landing 2: fused reduce + residual-add + rmsnorm (EXPERT_GPU_ALLREDUCE §5, §9).
// Replaces the K2' (or K2) + fused_res_rmsnorm_b / fused_res_rmsnorm_q_b two-launch chain at the
// mixer/FFN epilogue sites. Same GPU-direct two-stage gate as tp_wait_add_g (stage A = the
// peer_committed hint, stage B = the slot tail with the TP_TAIL_WAIT_NS deadline -> device status
// 11), then per element the K2 rounding feeds the fused_res_rmsnorm_b body VERBATIM, so the
// rounding placement of the current two-kernel chain is replicated exactly (§4 — the trap):
//   r1 = f2b(a + b)            — the K2 round (mode 0: bf16 partials, mode 1: fp32 partials),
//                                materialized nowhere, rounded at exactly the same point
//   v  = b2f(residual) + b2f(r1); residual = f2b(v)
//   flavor 0 (S, mixer site):  sum_sq += v*v            — sum_sq from the UNROUNDED fp32 add
//   flavor 1 (Q, FFN site):    sum_sq += b2f(f2b(v))²   — sum_sq from the ROUNDED residual
// Both flavors then run the identical halving tree and pass 2 (re-reads the ROUNDED residual).
//
// Multi-block (one block per batch column, like fused_res_rmsnorm_b): each block spins its own
// thread 0 on the SAME two lines (read-only polls, I6 — no RMW), then reduces its column against
// the shared recv slot. `local` is the K1-published LOCAL partial (bf16 mode 0 / fp32 mode 1) and
// `peer` the recv-ring slot — the same operands tp_reduce_body reads, in the same canonical
// rank0+rank1 order (cross-rank bit-identity by IEEE commutativity, round-3 R3c).
//
// The length is DERIVED (n x elem_bytes, align8), never read from len_peer — the same SPMD
// derivation tp_wait_add_g uses; K1 at these sites ships exactly n*elem_bytes, so the derived
// wire offset matches the sender's tail by construction.
//
// Launched PLAIN (blaunch!) — K1 -> this kernel is a hard stream edge (§5 rule 2: it reads
// c->epoch). Its own launch_dependents edge releases the downstream PSS dependents (the pdl
// GEMM's weight prefetch) during the wait, exactly as K2/tp_wait_add_g. Grid (batch,1,1) x 1024
// threads x 4096 B smem. rx_done published from block 0 after a final __syncthreads() — the
// watchdog's v2 receive watermark (it covers block 0's consumption; the watchdog's 10 s debt
// tolerance absorbs the other blocks' skew).
extern "C" __global__ void tp_reduce_resnorm_b(tp_dev_ctx* c, __nv_bfloat16* out, __nv_bfloat16* residual,
                                               const void* local, const float* w, int n, int B,
                                               float eps, int fp32_mode, int flavor) {
    const unsigned long long e = c->epoch;
    const unsigned slot = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);

    // E9 edge BEFORE the spins, exactly as tp_wait_add_g (C8 — the fused successor keeps it).
    asm volatile("griddepcontrol.launch_dependents;");

    // Stage A: the peer's inline commit hint (non-trusted, arms the deadline; no deadline of its
    // own — a slow peer is legitimate, the 10 s watchdog bounds it).
    if (tp_spin_until_ge(c, tp_peer_committed_ptr(c, e), e, 1)) return;

    // Stage B: the payload tail — the actual placement proof. Deadline-armed from stage-A
    // completion: the payload was posted before the commit on a reliable QP, so a tail lagging
    // the commit past TP_TAIL_WAIT_NS is lost, not slow -> cooperative status 11 (I9), the
    // host re-inits (I8). Body verbatim from tp_wait_add_g.
    // mode 3 (R9) ships BF16 (2 B/elem, like mode 0); modes 1/2 ship fp32 (4 B/elem).
    const unsigned len = n * ((fp32_mode == 1 || fp32_mode == 2) ? 4u : 2u);   // (mode 2 never reaches a norm site)
    const unsigned wire = (len + 7u) & ~7u;
    const unsigned char* peer = tp_recv_slot_ptr(c, e);
    // N-way receive-side clean (world>2 only), same discipline as tp_wait_add_g: erase a PREVIOUS
    // generation's tail before the stage-B wait, but never a fresh one (guarded != e).
    if (c->world > 2 && threadIdx.x == 0) {
        volatile unsigned long long* tw = (volatile unsigned long long*)(peer + wire);
        if (tp_ld_relaxed((const unsigned long long*)tw) != e) *tw = 0ull;
    }
    __syncthreads();
    __shared__ int s_timeout;
    if (threadIdx.x == 0) {
        s_timeout = 0;
        const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
        const unsigned long long* tail = (const unsigned long long*)(peer + wire);
        // B8 §1.5-3: adaptive stage-B deadline — wide (slot-filling) epochs get the 5 s bound,
        // decode-width keep the 1 ms fast path. Numerics-neutral (wait bound only).
        unsigned long long deadline = tp_globaltimer()
            + (wire > c->payload_bytes ? TP_TAIL_LONG_WAIT_NS : TP_TAIL_WAIT_NS);
        unsigned ns = 64, cap = 512u;
        while (tp_ld_relaxed(tail) != e) {
            if (tp_ld_relaxed(ab)) { s_timeout = 1; break; }   // someone else aborted — no-op
            if (tp_globaltimer() >= deadline) {
                tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                printf("[K2-resnorm-stageB-timeout] rank=%d epoch=%llu wire=%u fp32_mode=%d\n",
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

    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long off = (long long)b * n;
    // Canonical lower+upper order, identical to tp_reduce_body: world==2 -> rank0 first; world>2 ->
    // the rank whose bit `round(e)` is 0 is the lower half. The bare `rank == 0` was the pre-N-way
    // world==2-only rule and is WRONG at world>2 (it broke rank-identity for ranks 1 and 3).
    const int r0 = (c->world > 2) ? ((c->rank & (1 << tp_round_of(c, e))) == 0) : (c->rank == 0);
    const float* lo_f = (const float*)local + off;
    const float* pe_f = (const float*)peer + off;
    const __nv_bfloat16* lo_b = (const __nv_bfloat16*)local + off;
    const __nv_bfloat16* pe_b = (const __nv_bfloat16*)peer + off;

    // Pass 1: the K2 rounding fused into the residual add (fused_res_rmsnorm_b body verbatim).
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += bs) {
        float a, bb;
        if (fp32_mode) { a = r0 ? lo_f[i] : pe_f[i]; bb = r0 ? pe_f[i] : lo_f[i]; }
        else           { a = b2f(r0 ? lo_b[i] : pe_b[i]); bb = b2f(r0 ? pe_b[i] : lo_b[i]); }
        __nv_bfloat16 r1 = f2b(a + bb);          // the K2 round — rounded at exactly the same point
        float v = b2f(residual[off + i]) + b2f(r1);
        if (flavor) {                            // Q (FFN site): sum_sq from the ROUNDED residual
            __nv_bfloat16 r = f2b(v);
            residual[off + i] = r;
            float vr = b2f(r);
            sum_sq += vr * vr;
        } else {                                 // S (mixer site): sum_sq from the UNROUNDED fp32 add
            residual[off + i] = f2b(v);
            sum_sq += v * v;
        }
    }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)n + eps);
    // Pass 2 re-reads the ROUNDED residual (both flavors — the pass-1 store is the shared state).
    for (int i = tid; i < n; i += bs) {
        float v = b2f(residual[off + i]);
        out[off + i] = f2b(v * inv * (1.0f + w[i]));
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);
}

// ---- bench-only kernels (--tp-barrier-bench). Kept OUT of K1/K2 so the production path the gate
// proves is byte-for-byte the production path that ships. All derive the epoch from c->epoch. ----

// Self-describing payload: word0 = epoch, interior = LFSR keyed on the epoch, last word = XOR checksum.
// Detects stale slot reuse, epoch skew, and partially-visible DMA in one check.
__device__ __forceinline__ unsigned tp_bench_word(unsigned long long e, unsigned i) {
    unsigned x = (unsigned)e * 2654435761u + i * 2246822519u;
    x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return x;
}
extern "C" __global__ void tp_bench_fill(tp_dev_ctx* c, unsigned int* src) {
    const unsigned long long e = c->epoch + 1;      // K1 has not incremented yet — fill for the NEXT epoch
    const unsigned nw = c->payload_bytes >> 2;
    unsigned chk = 0;
    for (unsigned i = threadIdx.x; i < nw - 1; i += blockDim.x) {
        unsigned v = (i == 0) ? (unsigned)e : tp_bench_word(e, i);
        src[i] = v;
        chk ^= v;
    }
    // block-wide XOR reduction into the final word
    __shared__ unsigned s_chk[512];
    s_chk[threadIdx.x] = chk;
    __syncthreads();
    for (unsigned st = blockDim.x >> 1; st; st >>= 1) {
        if (threadIdx.x < st) s_chk[threadIdx.x] ^= s_chk[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) src[nw - 1] = s_chk[0];
}

// Validate the peer's slot for this epoch, then POISON it so any later stale read is unmistakable.
// err[0] = error count, err[1] = first bad epoch, err[2] = first bad word index, err[3] = observed word.
extern "C" __global__ void tp_bench_validate(tp_dev_ctx* c, unsigned long long* err, int poison) {
    const unsigned long long e = c->epoch;
    // P3-1: in one-shot mode the peers' payloads land in the SENDER-indexed block (sender p's
    // epoch-e payload at oneshot_recv + p*R + e%R), not the round-keyed recv ring. Every rank
    // fills the SAME deterministic payload (tp_bench_word is rank-independent), so each sender
    // ring must independently hold the expected bytes. Validate ALL world-1 sender rings.
    const unsigned char* slot0;
    unsigned nrings;
    if (c->oneshot && c->oneshot_recv) {
        slot0 = c->oneshot_recv;
        nrings = c->world;                 // validate every sender's ring except our own
    } else {
        slot0 = tp_recv_slot_ptr(c, e);
        nrings = 1;
    }
    const unsigned nw = c->payload_bytes >> 2;

    for (unsigned r = 0; r < nrings; r++) {
        if (nrings > 1 && (int)r == (int)c->rank) continue;   // our own ring: we are the sender
        unsigned int* slot = (unsigned int*)(slot0
            + ((unsigned long long)r * (unsigned long long)TP_RING_SLOTS
               + (nrings > 1 ? (e & (TP_RING_SLOTS - 1)) : 0ull))
              * (unsigned long long)c->slot_stride);
        unsigned chk = 0;
        for (unsigned i = threadIdx.x; i < nw - 1; i += blockDim.x) {
            unsigned want = (i == 0) ? (unsigned)e : tp_bench_word(e, i);
            unsigned got  = slot[i];
            chk ^= got;
            if (got != want) {
                if (atomicAdd(err, 1ull) == 0) { err[1] = e; err[2] = i; err[3] = got; }
            }
        }
        __shared__ unsigned s_chk[512];
        s_chk[threadIdx.x] = chk;
        __syncthreads();
        for (unsigned st = blockDim.x >> 1; st; st >>= 1) {
            if (threadIdx.x < st) s_chk[threadIdx.x] ^= s_chk[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (s_chk[0] != slot[nw - 1] && atomicAdd(err, 1ull) == 0) {
                err[1] = e; err[2] = 0xFFFFFFFFu; err[3] = slot[nw - 1];
            }
            // the tail guard the receive proxy also checks — verify it GPU-side too
            unsigned long long tail = *(unsigned long long*)((char*)slot + c->payload_bytes);
            if (tail != e && atomicAdd(err, 1ull) == 0) { err[1] = e; err[2] = 0xFFFFFFFEu; err[3] = tail; }
        }
        __syncthreads();
        if (poison) {
            for (unsigned i = threadIdx.x; i < nw; i += blockDim.x) slot[i] = 0xDEADBEEFu;
            if (threadIdx.x == 0) *(unsigned long long*)((char*)slot + c->payload_bytes) = 0xDEADBEEFDEADBEEFull;
        }
        __syncthreads();
    }
}

// Drive the system to ring-full so the reuse gate and the S<=R invariant are exercised deliberately.
// Nothing else in the bench reaches ring depth, so without this the gate would ship untested.
// Sample the GPU's %globaltimer so the host can estimate the GPU<->CPU clock offset (the two domains
// have different epochs, so any cross-domain stage timing is meaningless without it).
extern "C" __global__ void tp_bench_now(unsigned long long* out) {
    if (threadIdx.x == 0) *out = tp_globaltimer();
}

extern "C" __global__ void tp_bench_stall(tp_dev_ctx* c, unsigned int every, unsigned long long ns) {
    if (threadIdx.x != 0 || !every) return;
    if ((c->epoch % every) != 0) return;
    unsigned long long until = tp_globaltimer() + ns;
    while (tp_globaltimer() < until) __nanosleep(1000);
}

// ---- TP=2 masked-replicated FFN (Stage 3 Proof v0) ----
// Both ranks compute the FULL intermediate; each then ZEROS the rows it does not own so the (replicated)
// down-proj becomes this rank's partial sum over its half of the intermediate, and a cross-rank
// all-reduce of the down-proj output reconstructs the full FFN. The buffer is token-major [batch, im]
// (each token's im-vector is contiguous — see gemm_mma_fp4_b's X indexing X[col*K+k]), so an element's
// intermediate-row index is (i % im). Zeros every element whose row is outside [lo, hi).
extern "C" __global__ void tp_mask_rows(__nv_bfloat16* buf, int im, int total, int lo, int hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i % im;
    if (row < lo || row >= hi) buf[i] = f2b(0.0f);
}

// ---- batched per-head RMSNorm: x is [nh*hd, B]; one block per (seq, head) ----
extern "C" __global__ void rmsnorm_perhead_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w, int nh, int hd, int B, float eps) {
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * (nh * hd) + (long long)head * hd;
    float v = (tid < hd) ? b2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) out[base + tid] = f2b(v * inv * (1.0f + w[tid]));
}

// ---- batched gated RMSNorm (linear attn): core,z are [nh*vd, B]; one block per (seq,head) ----
// F0: z_off_z_stride packs the z row geometry — (z_off<<15)|z_stride — (0, nh*vd) packed, or
// (conv_dim, mtot) for the z VIEW inside the fused GDN output [qkv|z|b|a].
extern "C" __global__ void rmsnorm_gated_b(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z, const float* w, int vd, int nh, int B, float eps, int z_off_z_stride) {
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const int z_off = z_off_z_stride & 0x7FFF;
    const int z_stride = (unsigned)z_off_z_stride >> 15;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? b2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) out[base + tid] = f2b(v * inv * w[tid] * silu_f(b2f(z[z_off + (long long)b * z_stride + (long long)head * vd + tid])));
}

// ---- batched rotate_half RoPE with per-seq cos/sin tables [B, rdim] ----
extern "C" __global__ void rope_b(__nv_bfloat16* x, const float* cos, const float* sin, int nh, int hd, int rdim, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rdim / 2;
    int per_seq = nh * half;
    int total = B * per_seq;
    if (idx >= total) return;
    int b = idx / per_seq;
    int rem = idx % per_seq;
    int head = rem / half;
    int pair = rem % half;
    long long base = (long long)b * (nh * hd) + (long long)head * hd;
    long long cb = (long long)b * rdim + pair;
    float x1 = b2f(x[base + pair]);
    float x2 = b2f(x[base + pair + half]);
    float c = cos[cb], s = sin[cb];
    x[base + pair] = f2b(x1 * c - x2 * s);
    x[base + pair + half] = f2b(x2 * c + x1 * s);
}

// ---- batched split q proj output [nh*hd*2, B] into q[nh*hd,B] and gate[nh*hd,B] ----
// F0: `qg_stride` is the qg row pitch (nh*hd*2 packed, or the fused qkv mtot for an offset view).
extern "C" __global__ void split_qgate_b(__nv_bfloat16* q, __nv_bfloat16* gate, const __nv_bfloat16* qg, int nh, int hd, int B, int qg_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nh * hd;
    if (idx >= total) return;
    int b = idx / (nh * hd);
    int rem = idx % (nh * hd);
    int head = rem / hd;
    int d = rem % hd;
    long long qg_base = (long long)b * qg_stride + (long long)head * (hd * 2);
    q[idx] = qg[qg_base + d];
    gate[idx] = qg[qg_base + hd + d];
}

// ---- batched depthwise causal conv1d step ----
// x: [conv_dim, B] bf16 (in/out); state: [B, conv_dim, k] f32; w: [conv_dim, k] f32
// F0: `row_stride` is the x row pitch (conv_dim packed, or the fused GDN mtot for a qkv view).
extern "C" __global__ void conv1d_b(__nv_bfloat16* x, float* state, const float* w, int conv_dim, int k, int B, const int* slot_ids, int row_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * conv_dim;
    if (idx >= total) return;
    int b = idx / conv_dim;
    int c = idx % conv_dim;
    int slot = slot_ids[b];
    float* st = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 1; j < k; j++) st[j - 1] = st[j];
    st[k - 1] = b2f(x[(long long)b * row_stride + c]);
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
    x[(long long)b * row_stride + c] = f2b(silu_f(acc));
}

// ---- batched gated-delta-rule recurrent step ----
// qkv: [conv_dim, B] bf16; state: [B, nh, kd, vd] f32; core out: [nh*vd, B] bf16
// b_in,a_in: [nh, B] bf16; a_log,dt_bias: [nh] f32
// ===================== GATED DELTA-NET: the recurrent scan =====================
//
// The delta rule touches its recurrent state S six times per token: decay it, dot it with k, rank-1
// update it, dot it with q. Done through GLOBAL memory (as this kernel used to) that is 6 x |S| of
// traffic per token — 288 MB per token across 9B's 24 GDN layers, which measured at 20% of ALL GPU
// time and, worse, was the ENTIRE per-column cost of a speculative verify. The GEMM is flat in N now;
// this was what still scaled.
//
// So: load S into SHARED memory once, run the whole token loop there, store it back once. Global
// traffic drops from 6|S| per token to 2|S| per FORWARD, regardless of N.
//
// Blocked over `vd` (the value dim) because every column of S is independent — the two dot products
// are per-column, delta is per-column, and the rank-1 update is per-(row,column). Only the q/k norms
// reduce over the key dim, and those are cheap enough to recompute in each block. Chunking is what
// gets us from 32 blocks (nh) to 128, i.e. from two thirds of the SMs idle to a full wave.
//
// THE ARITHMETIC BELOW IS BIT-FOR-BIT WHAT THE GLOBAL-MEMORY VERSION DID, and that is load-bearing,
// not cosmetic: decode (`delta_step_b`) and verify (`delta_step_prefill`) both call this, and if they
// disagree by one ulp then column 0 of a verify stops matching a decode and greedy MTP is silently no
// longer lossless. Hence: same halving tree-reduce for the norms, and the same ASCENDING sequential
// sum over `aa` for both dot products. Do not "improve" either into a parallel reduction without
// changing both paths together and re-running --probe-binv and --bench-mtp.

#define GDN_C 32          // vd columns of S per block. 128/32 = 4 blocks per head.
#define GDN_SP (GDN_C+1)  // padded row stride: kills the 32-way bank conflict on the row passes.

/// One delta-rule step, entirely inside the shared-memory tile `S_sh` [kd][GDN_SP].
/// `blockDim.x == kd`; `bb0` is this block's first vd column.
__device__ __forceinline__ void gdn_token(
    float* S_sh, int kd, int bb0,
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
    __nv_bfloat16* coreb, int key_head, float beta, float gt,
    float* Srow, float* kv_mem, float* vbuf, float* delta, float* qrow, float* krow)
{
    const int a = threadIdx.x;

    float qv = b2f(q_in[key_head * kd + a]);
    float kv = b2f(k_in[key_head * kd + a]);
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

    // decay — thread a owns ROW a of the tile
    #pragma unroll
    for (int c = 0; c < GDN_C; c++) S_sh[a * GDN_SP + c] *= gt;
    __syncthreads();

    // thread a owns COLUMN a of the tile (only the first GDN_C threads have one)
    if (a < GDN_C) {
        float km = 0.0f;
        for (int aa = 0; aa < kd; aa++) km += S_sh[aa * GDN_SP + a] * krow[aa];   // ASCENDING — fixed
        kv_mem[a] = km;
        vbuf[a]   = b2f(v_in[bb0 + a]);
    }
    __syncthreads();
    if (a < GDN_C) delta[a] = (vbuf[a] - kv_mem[a]) * beta;
    __syncthreads();

    // rank-1 update — thread a owns ROW a again
    const float kk = krow[a];
    #pragma unroll
    for (int c = 0; c < GDN_C; c++) S_sh[a * GDN_SP + c] += kk * delta[c];
    __syncthreads();

    if (a < GDN_C) {
        float o = 0.0f;
        for (int aa = 0; aa < kd; aa++) o += S_sh[aa * GDN_SP + a] * qrow[aa];    // ASCENDING — fixed
        coreb[bb0 + a] = f2b(o);
    }
    __syncthreads();
}

/// Cooperatively move this block's [kd][GDN_C] slice of S between global and shared.
/// A warp covers exactly one row's GDN_C floats = 128 contiguous bytes.
__device__ __forceinline__ void gdn_tile_load(float* S_sh, const float* S, int kd, int vd, int bb0) {
    for (int i = threadIdx.x; i < kd * GDN_C; i += blockDim.x) {
        int a = i / GDN_C, c = i - a * GDN_C;
        S_sh[a * GDN_SP + c] = S[(long long)a * vd + bb0 + c];
    }
    __syncthreads();
}
__device__ __forceinline__ void gdn_tile_store(float* S, const float* S_sh, int kd, int vd, int bb0) {
    __syncthreads();
    for (int i = threadIdx.x; i < kd * GDN_C; i += blockDim.x) {
        int a = i / GDN_C, c = i - a * GDN_C;
        S[(long long)a * vd + bb0 + c] = S_sh[a * GDN_SP + c];
    }
}

/// DECODE: one token per sequence, `B` sequences in flight. grid = B * nh * (vd/GDN_C).
// `visits` (proof build only, else NULL): one counter per value head, incremented once per block-chunk.
// Catches the ALIASING failure the red zones cannot see — a grid oversized such that surplus blocks fold
// back onto valid local heads via `blk % nh`, which is redundant work with identical output and no
// out-of-range write. Checked == nchunk per head at end of run. ~24 atomics/layer/token: negligible, and
// it only exists under the proof flag.
extern "C" __global__ void delta_step_b(__nv_bfloat16* core, const __nv_bfloat16* qkv, float* state,
                                         const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
                                         int nh_packed, int kd_vd,
                                         const float* a_log, const float* dt_bias,
                                         const int* slot_ids, unsigned long long* visits,
                                         int qkv_ba_stride) {
    // nh_packed = n_value_heads | (n_key_heads << 16). cudarc caps kernel launches at 12 arguments and
    // the proof-build `visits` pointer took the last slot, so these two small counts share one.
    const int nh = nh_packed & 0xFFFF;
    const int n_k_heads = (nh_packed >> 16) & 0xFFFF;
    const int kd = kd_vd & 0xFFFF;
    const int vd = (unsigned)kd_vd >> 16;
    // F0: qkv_ba_stride = (qkv_stride<<15)|ba_stride — the qkv row pitch (conv_dim packed / fused
    // mtot) and the b/a row pitch (nh packed / fused mtot); the b/a base pointers are pre-offset
    // by the caller (segment offset + this rank's h0 head range) in the fused-view layout.
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int nchunk = vd / GDN_C;
    int blk = blockIdx.x;
    const int chunk = blk % nchunk;  blk /= nchunk;
    const int head  = blk % nh;      blk /= nh;
    const int b     = blk;
    if (visits && threadIdx.x == 0) atomicAdd(&visits[head], 1ull);
    const int key_head = head * n_k_heads / nh;   // map value head -> key head
    const int key_dim = n_k_heads * kd;
    const int bb0 = chunk * GDN_C;

    extern __shared__ float sh[];
    float* S_sh   = sh;                       // [kd][GDN_SP]
    float* Srow   = S_sh + kd * GDN_SP;       // [kd]
    float* kv_mem = Srow + kd;                // [GDN_C]
    float* vbuf   = kv_mem + GDN_C;
    float* delta  = vbuf + GDN_C;
    float* qrow   = delta + GDN_C;            // [kd]
    float* krow   = qrow + kd;                // [kd]

    const __nv_bfloat16* col = qkv + (long long)b * qkv_stride;
    float* S = state + ((long long)slot_ids[b] * nh + head) * kd * vd;

    float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)b * ba_stride + head])));
    float sp = b2f(a_in[(long long)b * ba_stride + head]) + dt_bias[head];
    sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
    float gt = __expf(-__expf(a_log[head]) * sp);

    gdn_tile_load(S_sh, S, kd, vd, bb0);
    gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
              core + (long long)b * (nh * vd) + (long long)head * vd,
              key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);
    gdn_tile_store(S, S_sh, kd, vd, bb0);
}

/// PREFILL / VERIFY: N tokens of ONE sequence, scanned sequentially. grid = nh * (vd/GDN_C).
///
/// `mid_s` snapshots the state after each of the first N-1 tokens, so a speculative verify that
/// accepts `nacc < N` drafts can restore S_nacc without a second forward. It must be PER COLUMN:
/// snapshotting only after the first token silently corrupts the recurrent state at any depth > 2,
/// which is what made "acceptance collapses with depth" look like a property of the model.
// parent (tree drafting): parent[t] = the node whose recurrent state node t continues (its DFS parent).
// A CHAIN is parent[t] = t-1 (with parent[0] = -1), so the reload below never fires and the kernel is
// byte-identical to the pre-tree scan. Visited in DFS order, so the resident S_sh is usually already the
// parent's state; only a genuine branch (parent[t] != t-1) reloads it from that parent's checkpoint.
// nullptr => plain chain (the main prefill). kd,vd packed into one arg to fit the 12-arg launch cap.
extern "C" __global__ void delta_step_prefill(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh, float* mid_s,
    const int* parent) {
    // `parent` is PACKED per column: low 16 bits = DFS parent (two's-complement int16, root = -1),
    // high 16 bits = the column's lane slot. This keeps the launch at 12 args (cudarc's tuple ceiling)
    // while routing per-column slot into the forest scan. null `parent` (prefill) => pre-offset `state`.
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    // F0: qkv_ba_stride = (qkv_stride<<15)|ba_stride — the qkv row pitch (conv_dim packed / fused
    // mtot) and the b/a row pitch (nh packed / fused mtot); the b/a base pointers are pre-offset
    // by the caller (segment offset + this rank's h0 head range) in the fused-view layout. `nh`
    // was a launch arg and is now derived from the grid (gridDim.x == nh*nchunk).
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int nchunk = vd / GDN_C;
    const int chunk = blockIdx.x % nchunk;
    const int head  = blockIdx.x / nchunk;
    const int nh = (int)(gridDim.x / nchunk);
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int bb0 = chunk * GDN_C;

    extern __shared__ float sh[];
    float* S_sh   = sh;
    float* Srow   = S_sh + kd * GDN_SP;
    float* kv_mem = Srow + kd;
    float* vbuf   = kv_mem + GDN_C;
    float* delta  = vbuf + GDN_C;
    float* qrow   = delta + GDN_C;
    float* krow   = qrow + kd;

    const long long head_off = (long long)head * kd * vd;
    const long long slot_stride = (long long)nh * kd * vd;

    // FOREST scan. Columns are packed lanes: each lane is a chain whose first column is a ROOT
    // (packed parent low-16 == -1). A root (re)initialises S_sh from ITS lane's committed state
    // (packed parent high-16 == the lane slot); interior columns continue the resident state, or on a
    // genuine tree branch reload the parent's mid_s checkpoint. A lane's LAST column commits S_sh back
    // to its slot. Single lane: one root at t=0, one commit at t=N-1 -> byte-identical to the old chain.
    // null `parent` (prefill): p_t==t-1, one root at t=0 with slot 0 (state pre-offset), commit at N-1.
    for (int t = 0; t < N; t++) {
        const int slot_t = parent ? (int)((unsigned)parent[t] >> 16) : 0;
        const int p_t    = parent ? (int)(short)((unsigned)parent[t] & 0xFFFF) : (t - 1);
        if (p_t == -1) {
            // Lane root: load this lane's committed recurrent state. t==0 needs no barrier (S_sh fresh);
            // an interior root must wait for the previous lane's last readers of S_sh.
            if (t != 0) __syncthreads();
            gdn_tile_load(S_sh, state + (long long)slot_t * slot_stride + head_off, kd, vd, bb0);
        } else if (p_t != t - 1) {
            __syncthreads();   // every thread past its last S_sh use before we overwrite it
            gdn_tile_load(S_sh, mid_s + (long long)p_t * slot_stride + head_off, kd, vd, bb0);
        }
        const __nv_bfloat16* col = qkv + (long long)t * qkv_stride;

        float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)t * ba_stride + head])));
        float sp = b2f(a_in[(long long)t * ba_stride + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        float gt = __expf(-__expf(a_log[head]) * sp);

        gdn_token(S_sh, kd, bb0, col, col + key_dim, col + 2 * key_dim + head * vd,
                  core + (long long)t * (nh * vd) + (long long)head * vd,
                  key_head, beta, gt, Srow, kv_mem, vbuf, delta, qrow, krow);

        if (t < N - 1 && mid_s) {
            float* mid_S = mid_s + (long long)t * slot_stride + head_off;
            gdn_tile_store(mid_S, S_sh, kd, vd, bb0);
            __syncthreads();
        }
        // Lane boundary: commit this lane's final state to its slot if the next column is a new root
        // (or this is the last column). Reads S_sh; the next root's load is separated by __syncthreads.
        bool last_of_lane = (t == N - 1);
        if (!last_of_lane && parent) {
            const int next_p = (int)(short)((unsigned)parent[t + 1] & 0xFFFF);
            last_of_lane = (next_p == -1);
        }
        if (last_of_lane) {
            gdn_tile_store(state + (long long)slot_t * slot_stride + head_off, S_sh, kd, vd, bb0);
        }
    }
}

// ===========================================================================
// P4 B3 (2026-08-17): CHUNKED GDN prefill scan — WY/UT form, prefill-ONLY.
//
// Replaces the per-token serial loop for N >= GDN_CHUNK_MIN when the caller sets
// GB10_GDN_CHUNK. Decode/verify keep delta_step_prefill (bit-exact contract, §2.4/2.8);
// prefill sits OUTSIDE the batch-invariance contract (same rule as chunked prefill
// reassociation, batch.rs), so the chunked reassociation is acceptable — validated
// o/S rel-L2 ~7.6e-5 vs the sequential kernel on metal (tool_probe/b3_gdn_chunk.cu).
//
// Math (per head; C=32 tokens/chunk; gamma_t = prod_{r<=t} g_r within chunk, log-space):
//   A[t,s] = (gamma_{t-1}/gamma_s)(k_t.k_s)  strictly lower
//   T = (I + diag(beta)A)^-1 diag(beta)   (fwd-substitution, f32)
//   U = T (V - Gamma K S_in),  Gamma_t = gamma_t   [gamma_t NOT gamma_{t-1} — validated]
//   S_out = gamma_C S_in + K_w^T U,  K_w[s] = (gamma_C/gamma_s) k_s
//   O = D U + Gamma (Q S_in),  D[t,s] = (gamma_t/gamma_s)(q_t.k_s), s<=t
// q/k are normalized + 1/sqrt(kd)-scaled IN KERNEL (same formula as gdn_token).
// Grid: nh * (vd/64); block 256; dynamic smem (see Rust dispatch for the size).
// ===========================================================================
#define GDN_CHUNK 32
#define GDN_VCB   64            // value columns per block (validated standalone config; needs smem opt-in)


// ===========================================================================
// P0 (F-report step 3): one coalesced, full-occupancy prep pass per GDN layer.
// Grid: (N); block: 128 threads == kd for one head... expanded: blockIdx.y = head.
// blockDim.x = kd; grid (N, nh). Each block: halving-tree norms for its (t, head),
// reads the packed qkv ONCE coalesced (per head), writes DENSE scratch:
//   Qs [N][nh][KDS] f32 (normalized, *1/sqrt(kd); KDS=kd+4 pad keeps the scan's
//      smem loads conflict-free when copied 1:1),
//   Ks [N][nh][KDS] f32 (normalized, NO extra scale — gdn_token convention),
//   Vs [N][nh][vd] f32,
//   Ps [N][nh][2]  f32 (beta, log_g).
// The scan kernel then stages by pure coalesced f32 copies — no norms, no pitched bf16.
// Numerics: same formulas as the in-kernel staging it replaces (which themselves mirror
// gdn_token; the reduction order here is a halving tree like gdn_token's).
// ===========================================================================
extern "C" __global__ void gdn_prep_b(const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ b_in, const __nv_bfloat16* __restrict__ a_in,
    int qkv_ba_stride, int kd_vd, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, int N_nkh,
    float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs,
    float* __restrict__ Ps)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int nh = (int)gridDim.y;
    const int t = blockIdx.x;                 // token
    const int head = blockIdx.y;              // value head
    if (t >= N) return;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int a = threadIdx.x;                // 0..kd-1
    const int KDS = kd + 4;

    const __nv_bfloat16* col = qkv + (long long)t * qkv_stride;
    const float qv = b2f(col[key_head * kd + a]);
    const float kv = b2f(col[key_dim + key_head * kd + a]);

    // halving-tree norms (gdn_token's pattern; thread a of kd)
    __shared__ float nrm[2][128];
    nrm[0][a] = qv * qv; nrm[1][a] = kv * kv;
    __syncthreads();
    for (int s2 = kd / 2; s2 > 0; s2 >>= 1) {
        if (a < s2) { nrm[0][a] += nrm[0][a + s2]; nrm[1][a] += nrm[1][a + s2]; }
        __syncthreads();
    }
    const float qn = rsqrtf(nrm[0][0] + 1e-6f) * rsqrtf((float)kd);   // q: L2 * 1/sqrt(kd)
    const float kn = rsqrtf(nrm[1][0] + 1e-6f);                       // k: L2 ONLY
    __syncthreads();

    const long long base = ((long long)t * nh + head);
    Qs[base * KDS + a] = qv * qn;
    Ks[base * KDS + a] = kv * kn;

    // v: threads beyond kd copy columns (vd=128=kd here; general: strided)
    if (a < vd) Vs[base * vd + a] = b2f(col[2 * key_dim + head * vd + a]);

    if (a == 0) {
        float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)t * ba_stride + head])));
        float sp = b2f(a_in[(long long)t * ba_stride + head]) + dt_bias[head];
        sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
        Ps[base * 2 + 0] = beta;
        Ps[base * 2 + 1] = -__expf(a_log[head]) * sp;   // raw log-g (scan cumsums)
    }
}

// GB10_GDN_CHUNK2 (2026-08-26): tensor-core chunked GDN prefill scan — the chunk math of
// gdn_chunk_prefill_b (o/S rel-L2 7.6e-5 scalar) with every GEMM phase on mma.m16n8k16
// bf16 (probe: tool_probe/gdn_chunk_tc.cu, 9.4 ms/layer at N=8192 vs 198 sequential = 21x;
// o/S rel-L2 ~2.2e-2 vs the f32 seq oracle = the bf16-operand envelope, non-compounding
// over N=32..8192). Consumes the SAME gdn_prep_b P0 scratch; S carried in bf16 across
// chunks (FLA precedent). Host dispatches only for kd==128 && vd==128 && n>=256.
__device__ __forceinline__ unsigned gtc_pack2(float lo, float hi) {
    __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void gtc_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#define GTC_C 32
#define GTC_VC 64
#define GTC_KD2 (128 + 8)
#define GTC_CD2 (GTC_C + 8)
#define GTC_VD2 (GTC_VC + 8)
#define GTC_THR 256
extern "C" __global__ __launch_bounds__(GTC_THR, 1) void gdn_chunk_tc_b(
    __nv_bfloat16* __restrict__ core, float* __restrict__ state,
    const float* __restrict__ Qs, const float* __restrict__ Ks,
    const float* __restrict__ Vs, const float* __restrict__ Ps,
    int kd_vd, int N_nkh)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    (void)kd; (void)vd;              // host-guaranteed 128/128; tiles below are compile-time
    const int N = N_nkh & 0xFFFFFF;
    const int ncb = GTC_VC == 64 ? 2 : 1;
    const int NH_ = (int)(gridDim.x / ncb);
    const int C = GTC_C, VC = GTC_VC, KD = 128, VD = 128, KD2 = GTC_KD2, CD2 = GTC_CD2, VD2 = GTC_VD2;
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * VC;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const int g = lane >> 2, tq = lane & 3;      // fragment coords

    extern __shared__ unsigned char gtc_dyn[];
    __nv_bfloat16* Qb = (__nv_bfloat16*)gtc_dyn;                        // [C][KD2]
    __nv_bfloat16* Kb = Qb + C * KD2;                      // [C][KD2]
    __nv_bfloat16* Kt = Kb + C * KD2;                      // [KD][CD2] (transposed K)
    __nv_bfloat16* Vb = Kt + KD * CD2;                     // [C][VD2]
    __nv_bfloat16* Db = Vb + C * VD2;                      // [C][CD2]
    __nv_bfloat16* Ub = Db + C * CD2;                      // [C][VD2]  (unscaled U)
    __nv_bfloat16* Ulb = Ub + C * VD2;                     // [C][VD2]  (lambda-scaled U)
    __nv_bfloat16* Sb = Ulb + C * VD2;                     // [KD][VD2]
    float* Af = (float*)(Sb + KD * VD2);          // [C][C]
    float* Wf = Af + C * C;                       // [C][VC]
    float* Of = Wf + C * VC;                      // [C][VC]
    float* bt = Of + C * VC;                      // [C]
    float* lg = bt + C;                           // [C+1]

    const float* S_in = state + ((long long)head * KD) * VD + bb0;
    for (int i = t; i < KD * VC; i += GTC_THR) {
        int r = i / VC, c = i % VC;
        Sb[r * VD2 + c] = f2b(S_in[(long long)r * VD + c]);
    }
    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += C) {
        const int n = min(C, N - c0);
        // ---- stage: prep f32 -> bf16 tiles; pads zero ----
        for (int i = t; i < n * KD; i += GTC_THR) {
            int tt = i / KD, r = i % KD;
            const long long base = ((long long)(c0 + tt) * NH_ + head);
            Qb[tt * KD2 + r] = f2b(Qs[base * 132 + r]);
            const float kv = Ks[base * 132 + r];
            Kb[tt * KD2 + r] = f2b(kv);
            Kt[r * CD2 + tt] = f2b(kv);
        }
        for (int i = t; i < n * VC; i += GTC_THR) {
            int tt = i / VC, c = i % VC;
            const long long base = ((long long)(c0 + tt) * NH_ + head);
            Vb[tt * VD2 + c] = f2b(Vs[base * 128 + bb0 + c]);
        }
        __syncthreads();
        for (int i = t; i < (C - n) * KD; i += GTC_THR) {
            int tt = n + i / KD, r = i % KD;
            Qb[tt * KD2 + r] = f2b(0.f); Kb[tt * KD2 + r] = f2b(0.f); Kt[r * CD2 + tt] = f2b(0.f);
        }
        for (int i = t; i < (C - n) * VC; i += GTC_THR) {
            Vb[(n + i / VC) * VD2 + (i % VC)] = f2b(0.f);
        }
        __syncthreads();
        for (int i = t; i < C; i += GTC_THR) {
            if (i < n) {
                const long long base = ((long long)(c0 + i) * NH_ + head);
                bt[i] = Ps[base * 2 + 0]; lg[i + 1] = Ps[base * 2 + 1];
            } else { bt[i] = 0.f; lg[i + 1] = 0.f; }
        }
        __syncthreads();
        if (warp == 0 && lane < C) {              // inclusive scan (one warp)
            float v = lg[lane + 1];
#pragma unroll
            for (int o = 1; o < C; o <<= 1) {
                float u = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if ((int)lane >= o) v += u;
            }
            lg[lane + 1] = v;
        }
        if (t == 0) lg[0] = 0.f;
        __syncthreads();

        // ---- A / D grams (mma, out 32x32 = 8 tiles, one per warp; k=128) ----
        {
            const int mt = warp >> 2, nt = warp & 3;
            float accA[4] = {0,0,0,0}, accD[4] = {0,0,0,0};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq, s_row = nt * 8 + g;
                unsigned aK[4] = {
                    *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                    *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
                unsigned aQ[4] = {
                    *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                    *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
                unsigned bK[2] = {
                    gtc_pack2(b2f(Kb[s_row * KD2 + cc]), b2f(Kb[s_row * KD2 + cc + 1])),
                    gtc_pack2(b2f(Kb[s_row * KD2 + cc + 2]), b2f(Kb[s_row * KD2 + cc + 3]))};
                gtc_mma(accA, aK, bK);
                gtc_mma(accD, aQ, bK);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = mt * 16 + g + 8 * (e >= 2);
                const int scol = nt * 8 + 2 * tq + (e & 1);
                Af[row * C + scol] = (scol < row) ? accA[e] * __expf(lg[row] - lg[scol + 1]) : 0.f;
                Db[row * CD2 + scol] =
                    (scol <= row && scol < n) ? f2b(accD[e] * __expf(lg[row + 1] - lg[scol + 1])) : f2b(0.f);
            }
        }
        __syncthreads();

        // ---- W = V - exp(lg[tt+1]) * (K S) (mma, out 32x64 = 2 tiles/warp; k=128) ----
        {
            const int mt = warp >> 2, nt0 = (warp & 3) * 2;
            float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aK[4] = {
                    *(const unsigned*)(Kb + r0 * KD2 + cc), *(const unsigned*)(Kb + r1 * KD2 + cc),
                    *(const unsigned*)(Kb + r0 * KD2 + cc + 2), *(const unsigned*)(Kb + r1 * KD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bS[2] = {
                        gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aK, bS);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                    const float v = (row < n) ? b2f(Vb[row * VD2 + col]) : 0.f;
                    Wf[row * VC + col] = v - __expf(lg[row + 1]) * acc[j][e];
                }
        }
        __syncthreads();

        // ---- U = (I + diag(beta)A)^-1 (beta . W): fwd-subst in place, parallel cols ----
        for (int c = t; c < VC; c += GTC_THR) {
            for (int tt = 0; tt < n; tt++) {
                const float b = bt[tt];
                float acc = b * Wf[tt * VC + c];
                for (int s = 0; s < tt; s++) acc -= b * Af[tt * C + s] * Wf[s * VC + c];
                Wf[tt * VC + c] = acc;
            }
        }
        __syncthreads();

        // ---- materialize Ub (unscaled) and Ulb (lambda_s = exp(lg[n]-lg[s+1])) ----
        {
            const float gn = __expf(lg[n]);
            for (int i = t; i < C * VC; i += GTC_THR) {
                int s = i / VC, c = i % VC;
                const float u = (s < n) ? Wf[s * VC + c] : 0.f;
                Ub[s * VD2 + c] = f2b(u);
                Ulb[s * VD2 + c] = f2b(u * __expf(lg[n] - lg[s + 1]));
            }
            (void)gn;
        }
        __syncthreads();

        // ---- O = exp(lg[tt+1]) * (Q S) + (D U) (mma; QS k=128 then DU k=C accumulates) ----
        {
            const int mt = warp >> 2, nt0 = (warp & 3) * 2;
            float acc[2][4] = {{0,0,0,0},{0,0,0,0}};
            for (int ks = 0; ks < KD; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aQ[4] = {
                    *(const unsigned*)(Qb + r0 * KD2 + cc), *(const unsigned*)(Qb + r1 * KD2 + cc),
                    *(const unsigned*)(Qb + r0 * KD2 + cc + 2), *(const unsigned*)(Qb + r1 * KD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bS[2] = {
                        gtc_pack2(b2f(Sb[(ks + 4 * tq) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Sb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Sb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aQ, bS);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                    acc[j][e] = __expf(lg[row + 1]) * acc[j][e];
                }
            for (int ks = 0; ks < C; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aD[4] = {
                    *(const unsigned*)(Db + r0 * CD2 + cc), *(const unsigned*)(Db + r1 * CD2 + cc),
                    *(const unsigned*)(Db + r0 * CD2 + cc + 2), *(const unsigned*)(Db + r1 * CD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int col = (nt0 + j) * 8 + g;
                    unsigned bU[2] = {
                        gtc_pack2(b2f(Ub[(ks + 4 * tq) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Ub[(ks + 4 * tq + 2) * VD2 + col]), b2f(Ub[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aD, bU);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    if (row < n) {
                        const int col = (nt0 + j) * 8 + 2 * tq + (e & 1);
                        core[(long long)(c0 + row) * (NH_ * 128) + head * 128 + bb0 + col] = f2b(acc[j][e]);
                    }
                }
        }
        __syncthreads();

        // ---- S' = gamma S + Kt . Ulb  (mma out KD x VC = 8 tiles/warp; k=C) ----
        {
            const int mt = warp;                   // 8 m-tiles of 16 kd-rows
            const float gam = __expf(lg[n]);
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    acc[j][e] = gam * b2f(Sb[row * VD2 + col]);
                }
            for (int ks = 0; ks < C; ks += 16) {
                const int r0 = mt * 16 + g, r1 = r0 + 8, cc = ks + 4 * tq;
                unsigned aKt[4] = {
                    *(const unsigned*)(Kt + r0 * CD2 + cc), *(const unsigned*)(Kt + r1 * CD2 + cc),
                    *(const unsigned*)(Kt + r0 * CD2 + cc + 2), *(const unsigned*)(Kt + r1 * CD2 + cc + 2)};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int col = j * 8 + g;
                    unsigned bU[2] = {
                        gtc_pack2(b2f(Ulb[(ks + 4 * tq) * VD2 + col]), b2f(Ulb[(ks + 4 * tq + 1) * VD2 + col])),
                        gtc_pack2(b2f(Ulb[(ks + 4 * tq + 2) * VD2 + col]), b2f(Ulb[(ks + 4 * tq + 3) * VD2 + col]))};
                    gtc_mma(acc[j], aKt, bU);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = mt * 16 + g + 8 * (e >= 2);
                    const int col = j * 8 + 2 * tq + (e & 1);
                    Sb[row * VD2 + col] = f2b(acc[j][e]);
                }
        }
        __syncthreads();
    }

    // writeback final state (bf16 -> f32)
    float* S_out = state + ((long long)head * KD) * VD + bb0;
    for (int i = t; i < KD * VC; i += GTC_THR) {
        int r = i / VC, c = i % VC;
        S_out[(long long)r * VD + c] = b2f(Sb[r * VD2 + c]);
    }
}


// Template body: OLDSTAGE=false is the serving path (P0 dense-scratch staging from gdn_prep_b);
// OLDSTAGE=true is a DIAGNOSTIC variant with the pre-P0 in-kernel staging (normalize + serial
// dots from qkv/b_in/a_in, scratch args ignored) — selected by GB10_GDN_OLDSTAGE at load, used
// with GB10_GDN_XCHECK to attribute chunked-vs-seq divergence to the staging rework vs the chunk
// algorithm itself. Dead branch compiles out; the serving kernel is unchanged.
template <bool OLDSTAGE>
__device__ __forceinline__ void gdn_chunk_prefill_body(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    const int qkv_stride = (unsigned)qkv_ba_stride >> 15;
    const int ba_stride = qkv_ba_stride & 0x7FFF;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int ncb = vd / GDN_VCB;                    // value-column blocks (2 at vd=128)
    const int head = blockIdx.x / ncb;
    const int bb0 = (blockIdx.x % ncb) * GDN_VCB;
    const int VC = GDN_VCB;
    const int nh = (int)(gridDim.x / ncb);
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int t = threadIdx.x;
    const float kds = rsqrtf((float)kd);

    // F-padded row stride for Kt/Qt: kd(=128) is ~0 mod 32 banks => 32-way conflicts in the
    // A/D Gram loops. kd+4 padding: measured -35% kernel time, bit-identical output (F report).
    const int KDS = kd + 4;
    extern __shared__ float dyn[];
    float* Ssh = dyn;                                // kd*VC
    float* Kt  = Ssh + kd * VC;                      // C*KDS (normed, scaled)
    float* Qt  = Kt + GDN_CHUNK * KDS;               // C*KDS
    float* Vt  = Qt + GDN_CHUNK * KDS;               // C*VC
    float* Wsh = Vt + GDN_CHUNK * VC;                // C*VC (W then U in place)
    float* Ash = Wsh + GDN_CHUNK * VC;               // C*C
    float* Dsh = Ash + GDN_CHUNK * GDN_CHUNK;        // C*C
    float* bt  = Dsh + GDN_CHUNK * GDN_CHUNK;        // C   (beta)
    float* lg  = bt + GDN_CHUNK;                     // C+1 (log gamma cumsum)

    const long long head_off = (long long)head * kd * vd;
    float* S_ = state + head_off;                    // pre-offset (prefill: slot 0)

    // load this block's state columns
    if (t == 0) lg[0] = 0.f;   // log-gamma cumsum base (uninitialized smem = NaN otherwise)
    for (int i = t; i < kd * VC; i += blockDim.x) {
        int r = i / VC, c = i % VC;
        Ssh[i] = S_[(long long)r * vd + bb0 + c];
    }
    __syncthreads();

    for (int c0 = 0; c0 < N; c0 += GDN_CHUNK) {
        const int n = min(GDN_CHUNK, N - c0);
        if (OLDSTAGE) {
            // pre-P0 staging (diagnostic reference): in-kernel normalize, serial dots
            for (int tt = t; tt < n; tt += blockDim.x) {
                const __nv_bfloat16* col = qkv + (long long)(c0 + tt) * qkv_stride;
                const __nv_bfloat16* q_src = col + key_head * kd;
                const __nv_bfloat16* k_src = col + key_dim + key_head * kd;
                const __nv_bfloat16* v_src = col + 2 * key_dim + head * vd + bb0;
                float qs2 = 0.f, ks2 = 0.f;
                for (int r = 0; r < kd; r++) {
                    float qv = b2f(q_src[r]), kv = b2f(k_src[r]);
                    Qt[tt * KDS + r] = qv; Kt[tt * KDS + r] = kv;
                    qs2 += qv * qv; ks2 += kv * kv;
                }
                float qn = rsqrtf(qs2 + 1e-6f) * kds, kn = rsqrtf(ks2 + 1e-6f);
                for (int r = 0; r < kd; r++) {
                    Qt[tt * KDS + r] *= qn; Kt[tt * KDS + r] *= kn;
                }
                for (int c = 0; c < VC; c++) Vt[tt * VC + c] = b2f(v_src[c]);
                float sp = b2f(a_in[(long long)(c0 + tt) * ba_stride + head]) + dt_bias[head];
                sp = (sp > 20.0f) ? sp : __logf(1.0f + __expf(sp));
                bt[tt] = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)(c0 + tt) * ba_stride + head])));
                lg[tt + 1] = -__expf(a_log[head]) * sp;   // raw log-g (cumsum below — SERIAL dep)
            }
        } else {
        // P0 staging: dense scratch from gdn_prep_b — coalesced f32 copies, no math.
        for (int i = t; i < n * kd; i += blockDim.x) {
            int tt = i / kd, r = i % kd;
            long long base = ((long long)(c0 + tt) * nh + head);
            Qt[tt * KDS + r] = Qs[base * KDS + r];
            Kt[tt * KDS + r] = Ks[base * KDS + r];
        }
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            long long base = ((long long)(c0 + tt) * nh + head);
            Vt[tt * VC + c] = Vs[base * vd + bb0 + c];
            if (c == 0) {
                bt[tt]       = Ps[base * 2 + 0];
                lg[tt + 1]   = Ps[base * 2 + 1];        // raw log-g (cumsum below)
            }
        }
        }
        __syncthreads();
        if (t == 0) {                                   // exclusive cumsum, single thread
            for (int i = 0; i < n; i++) lg[i + 1] += lg[i];
        }
        __syncthreads();
        // A = decay-weighted (k.k) Gram; D = decay-weighted (q.k) Gram — DIFFERENT left operands
        // (B3 port bug found by expert review: D must use Qt, not Kt).
        for (int i = t; i < n * n; i += blockDim.x) {
            int tt = i / n, s = i % n;
            float a = 0.f, d = 0.f;
            if (s < tt) {
                for (int r = 0; r < kd; r++) a += Kt[tt * KDS + r] * Kt[s * KDS + r];
                for (int r = 0; r < kd; r++) d += Qt[tt * KDS + r] * Kt[s * KDS + r];
                a *= __expf(lg[tt] - lg[s + 1]);
            } else if (s == tt) {
                for (int r = 0; r < kd; r++) d += Qt[tt * KDS + r] * Kt[s * KDS + r];
            }
            Ash[tt * GDN_CHUNK + s] = a;
            if (s <= tt) Dsh[tt * GDN_CHUNK + s] = d * __expf(lg[tt + 1] - lg[s + 1]);
            else        Dsh[tt * GDN_CHUNK + s] = 0.f;
        }
        __syncthreads();
        // W = V - Gamma K S_in (parallel over (t,c))
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            float ks = 0.f;
            for (int r = 0; r < kd; r++) ks += Kt[tt * KDS + r] * Ssh[r * VC + c];
            Wsh[i] = Vt[tt * VC + c] - __expf(lg[tt + 1]) * ks;
        }
        __syncthreads();
        // U = (I + diag(beta)A)^-1 diag(beta) W (fwd-subst, parallel over c)
        for (int c = t; c < VC; c += blockDim.x) {
            for (int tt = 0; tt < n; tt++) {
                float b = bt[tt];
                float acc = b * Wsh[tt * VC + c];
                for (int s = 0; s < tt; s++) acc -= Ash[tt * GDN_CHUNK + s] * b * Wsh[s * VC + c];
                Wsh[tt * VC + c] = acc;
            }
        }
        __syncthreads();
        // O = D U + Gamma (Q S_in)
        for (int i = t; i < n * VC; i += blockDim.x) {
            int tt = i / VC, c = i % VC;
            float qs = 0.f;
            for (int r = 0; r < kd; r++) qs += Qt[tt * KDS + r] * Ssh[r * VC + c];
            float acc = __expf(lg[tt + 1]) * qs;
            for (int s = 0; s <= tt; s++) acc += Dsh[tt * GDN_CHUNK + s] * Wsh[s * VC + c];
            core[(long long)(c0 + tt) * (nh * vd) + (long long)head * vd + bb0 + c] = f2b(acc);
        }
        __syncthreads();
        // S = gamma_C S + sum_s K_w[s] U[s]^T
        {
            float gC = __expf(lg[n]);
            for (int i = t; i < kd * VC; i += blockDim.x) {
                int r = i / VC, c = i % VC;
                float acc = gC * Ssh[i];
                for (int s = 0; s < n; s++)
                    acc += __expf(lg[n] - lg[s + 1]) * Kt[s * KDS + r] * Wsh[s * VC + c];
                Ssh[i] = acc;
            }
            __syncthreads();
        }
    }
    for (int i = t; i < kd * VC; i += blockDim.x) {
        int r = i / VC, c = i % VC;
        S_[(long long)r * vd + bb0 + c] = Ssh[i];
    }
}

extern "C" __global__ void gdn_chunk_prefill_b(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    gdn_chunk_prefill_body<false>(core, qkv, state, b_in, a_in, qkv_ba_stride, kd_vd,
                                  a_log, dt_bias, N_nkh, Qs, Ks, Vs, Ps);
}

// Diagnostic twin (same signature/launch config; scratch args ignored): pre-P0 staging.
extern "C" __global__ void gdn_chunk_prefill_b_oldstage(__nv_bfloat16* core, const __nv_bfloat16* qkv,
    float* state, const __nv_bfloat16* b_in, const __nv_bfloat16* a_in,
    int qkv_ba_stride, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh,
    const float* Qs, const float* Ks, const float* Vs, const float* Ps)
{
    gdn_chunk_prefill_body<true>(core, qkv, state, b_in, a_in, qkv_ba_stride, kd_vd,
                                 a_log, dt_bias, N_nkh, Qs, Ks, Vs, Ps);
}

extern "C" __global__ void compact_kv_b(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    __nv_bfloat16* ks, __nv_bfloat16* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = len * nkv * hd;
    if (idx >= total) return;
    int k = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int dv = rem % hd;
    long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * hd + dv;
    long long soff = ((long long)k * nkv + h) * hd + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

// ===================================================================================================
// 4-bit KV cache (GB10_KV_QUANT=1) — per-16-element affine quantization, deterministic per position.
//
// Layout per (slot, kvh, position): hd/16 blocks × 9 B = [8 B codes (16×4b)] [1 B e4m3 scale].
// The SAME e4m3 codec the NVFP4 weights use (round to nearest, low nibble = element 2j, high = 2j+1).
// Per-block scale = e4m3(amax/7), codes = round(x / e4m3(amax/7)) clamped to [-7, 7]. Deterministic and
// position-local: the packed form of position p depends only on (kvh, p), never on batch or the
// reduction structure — so decode, verify, and the prefill scratch all dequantize to the SAME
// values and the lossless-MTP contract (bitwise verify==decode) is preserved by construction.
//
// q = round(x * 7/amax); x' = q * amax/7. Error ~ amax/14 RMS per block — the P7 trade: 3.56x
// fewer KV bytes for a small attention-input perturbation (gated vs bf16 KV; see HY3 docs).
// ===================================================================================================
#define KVQ_BLK 16
#define KVQ_ROW_BYTES(hd) (((hd) / KVQ_BLK) * 12)   // [8 B codes][1 B e4m3 scale][3 B pad] — keeps u16/u32 alignment

// e4m3 codec (defined with the other quantization helpers further down; the KV pack/unpack
// and the 4-bit attention reader need it here).
__device__ __forceinline__ float e4m3_f(uint8_t b);
__device__ __forceinline__ uint8_t f32_to_e4m3(float f);

__device__ __forceinline__ void kvq16_pack(const __nv_bfloat16* x, unsigned char* out) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) amax = fmaxf(amax, fabsf(b2f(x[i])));
    // Round the scale to e4m3 FIRST and code with the rounded value (the weight quantizer's
    // convention, quant.rs). Coding with the exact amax/7 while the readers dequantize with the
    // e4m3-rounded byte gave every block a common-mode gain error of up to +-6.25% (3 mantissa
    // bits) on top of the +-0.5 LSB code error: +2..6% relL2 for nothing. f32_to_e4m3 is the exact
    // inverse of e4m3_f on representable values, so the stored byte is the scale the codes used.
    const float s = amax > 0.f ? e4m3_f(f32_to_e4m3(amax / 7.0f)) : 0.f;
    const float inv_s = s > 0.f ? 1.0f / s : 0.f;
    unsigned char c[KVQ_BLK / 2];
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) {
        int lo = (int)lrintf(b2f(x[2 * j]) * inv_s);
        int hi = (int)lrintf(b2f(x[2 * j + 1]) * inv_s);
        lo = max(-7, min(7, lo)) + 7;
        hi = max(-7, min(7, hi)) + 7;
        c[j] = (unsigned char)(lo | (hi << 4));
    }
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) out[j] = c[j];
    out[KVQ_BLK / 2] = f32_to_e4m3(s);
}

__device__ __forceinline__ void kvq16_unpack(const unsigned char* in, float* x) {
    const float s = e4m3_f(in[KVQ_BLK / 2]);
    #pragma unroll
    for (int j = 0; j < KVQ_BLK / 2; j++) {
        const int lo = (in[j] & 0xF) - 7;
        const int hi = (in[j] >> 4) - 7;
        x[2 * j] = (float)lo * s;
        x[2 * j + 1] = (float)hi * s;
    }
}

// write_kv_b_q4 — quantize + write one token's K/V (decode/verify path).
// One block per (b, kvh, block16): reads its 16 values, packs, writes 9 B. grid(B*nkv*(hd/16)).
extern "C" __global__ void write_kv_b_q4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = B * nkv * nb;
    if (idx >= total) return;
    const int b = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, k_cache + coff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + coff);
}

// write_kv_prefill_q4 — quantize + append N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_q4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        int stride, int nkv, int hd, int N, int pos_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = N * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, k_cache + coff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + coff);
}

// compact_kv_q4 — verbatim copy of packed rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; the packed form is position-local, so a byte copy suffices).
extern "C" __global__ void compact_kv_q4(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    const int rb = KVQ_ROW_BYTES(hd);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * rb;
    if (idx >= total) return;
    const int k = idx / (nkv * rb);
    const int rem = idx % (nkv * rb);
    const int h = rem / rb;
    const int dv = rem % rb;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * rb + dv;
    const long long soff = ((long long)k * nkv + h) * rb + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

// dequant_kv_q4 — expand packed KV to bf16 scratch for the prefill paths, in CACHE layout
// ([kvh][pos][hd], the layout attn_prefill_tiled / gqa_attn_prefill index). The formula is
// IDENTICAL to the splitk reader's, so prefill and decode see the same values.
// `out_stride` is the OUTPUT head stride: n_pos for a one-shot scratch (the old behavior — every
// call packs its own [0, n_pos) range), or the persistent mirror's position budget for the
// incremental path (E2 Fix 2), where the host shifts `out` by pos_start*hd so rows land at
// (h*out_stride + pos_start + t) — the shift is head-uniform, so a base shift suffices.
extern "C" __global__ void dequant_kv_q4(__nv_bfloat16* out, const unsigned char* cache,
        int nkv, int stride, int hd, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = n_pos * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const unsigned char* in = cache + crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    float x[KVQ_BLK];
    kvq16_unpack(in, x);
    __nv_bfloat16* o = out + ((long long)h * out_stride + t) * hd + blk * KVQ_BLK;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) o[i] = f2b(x[i]);
}

// ===================================================================================================
// k8v4 KV cache (GB10_KV_K8V4=1) — int8 K + the q4 V, per-16 affine blocks, deterministic per
// position. K: per-block scale s = amax/127 stored as fp16 (RNE), codes = lrintf(x*127/amax)
// clamped to [-127, 127] (the -128 code unused, mirroring q4's ±7-of-±8), 16 B codes + 2 B fp16
// scale = 18 meaningful B in a 20 B/16 stride (2 B pad — keeps every row 4-B/u32-aligned). The
// V cache reuses the q4 signed-nibble layout byte-for-byte (kvq16_pack, the 12 B/16 stride).
// The int8 grid is 127 steps vs q4's 7 -> ~18x quieter on the score dot; the fp16 scale
// removes q4's ~3% e4m3 common-mode scale rounding. Same determinism contract as q4: the packed
// form of position p depends only on (kvh, p), so decode, verify, and every split dequantize to
// the SAME values — the lossless-MTP contract is preserved by construction.
// ===================================================================================================
#define KV8_ROW_BYTES(hd) (((hd) / KVQ_BLK) * 20)   // [16 B codes][2 B fp16 scale][2 B pad] — u32-aligned

__device__ __forceinline__ void kv8_pack16(const __nv_bfloat16* x, unsigned char* out) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) amax = fmaxf(amax, fabsf(b2f(x[i])));
    // Same convention as kvq16_pack: round the scale to its stored precision (fp16) first, then
    // code with the rounded value, so the coder and the readers agree on the scale exactly.
    const float s = amax > 0.f ? __half2float(__float2half(amax / 127.0f)) : 0.f;   // amax==0 -> s=0, codes 0
    const float inv_s = s > 0.f ? 1.0f / s : 0.f;
    signed char c[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) {
        int q = (int)lrintf(b2f(x[i]) * inv_s);              // RNE, same arithmetic order as kvq16_pack
        q = max(-127, min(127, q));
        c[i] = (signed char)q;
    }
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) out[i] = (unsigned char)c[i];
    *(unsigned short*)(out + KVQ_BLK) = __half_as_ushort(__float2half(s));   // fp16 scale, RNE, LE
    out[KVQ_BLK + 2] = 0;                                     // pad
    out[KVQ_BLK + 3] = 0;
}

// write_kv_b_k8v4 — quantize + write one token's K/V (decode/verify path).
// One block per (b, kvh, block16), the q4 launch shape: K block via kv8_pack16 into the K cache
// at the 20 B/16 stride, V block via the EXISTING kvq16_pack into the V cache at the 12 B/16 stride.
extern "C" __global__ void write_kv_b_k8v4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = B * nkv * nb;
    if (idx >= total) return;
    const int b = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + vcoff);
}

// write_kv_prefill_k8v4 — quantize + append N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_k8v4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        int stride, int nkv, int hd, int N, int pos_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = N * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + vcoff);
}

// compact_kv_k8v4 — verbatim copy of packed rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; the packed form is position-local, so a byte copy suffices).
// The K and V caches diverge in row size for the first time: K rows are (hd/16)*20 B, V rows
// stay (hd/16)*12 B. Each byte of the combined grid belongs to exactly one cache: the first
// k_rb bytes of a (k, h) row's scratch are the K row, the remaining v_rb the V row. The host
// sizes the two scratch buffers separately (len*nkv*k_rb and len*nkv*v_rb bytes).
extern "C" __global__ void compact_kv_k8v4(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    const int k_rb = KV8_ROW_BYTES(hd);
    const int v_rb = KVQ_ROW_BYTES(hd);
    const int row_b = k_rb + v_rb;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * row_b;
    if (idx >= total) return;
    const int k = idx / (nkv * row_b);
    const int rem = idx % (nkv * row_b);
    const int h = rem / row_b;
    const int dv = rem % row_b;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long cbase = (((long long)slot * nkv + h) * stride + cache_pos);
    const long long sbase = ((long long)k * nkv + h);
    if (dv < k_rb) {
        const long long coff = cbase * k_rb + dv;
        const long long soff = sbase * k_rb + dv;
        if (dir == 0) { ks[soff] = k_cache[coff]; } else { k_cache[coff] = ks[soff]; }
    } else {
        const int vdv = dv - k_rb;
        const long long coff = cbase * v_rb + vdv;
        const long long soff = sbase * v_rb + vdv;
        if (dir == 0) { vs[soff] = v_cache[coff]; } else { v_cache[coff] = vs[soff]; }
    }
}

// dequant_kv_k8v4 — expand packed K (k8v4) to bf16 scratch for the prefill paths, in CACHE layout
// ([kvh][pos][hd]). The formula is IDENTICAL to the splitk reader's — bf16(fp32(int8 code) ×
// fp32(fp16 scale)), both upcasts exact, one rounding, position-stateless — so prefill and decode
// see the same values. K CHANNEL ONLY: the V cache reuses dequant_kv_q4 unchanged (the call sites
// already launch the dequant twice, K then V). Same `out_stride` semantics as dequant_kv_q4.
extern "C" __global__ void dequant_kv_k8v4(__nv_bfloat16* out, const unsigned char* cache,
        int nkv, int stride, int hd, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = n_pos * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const unsigned char* in = cache + crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const float s = __half2float(__ushort_as_half(*(const unsigned short*)(in + KVQ_BLK)));
    float x[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) x[i] = (float)((signed char)in[i]) * s;
    __nv_bfloat16* o = out + ((long long)h * out_stride + t) * hd + blk * KVQ_BLK;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) o[i] = f2b(x[i]);
}

// ===================================================================================================
// TurboQuant KV (E4, GB10_KV_TQ=1) — 3.5-bit pack-at-write rotated-domain KV cache.
// Reference: /tmp/tq_ref2 (REPORT.md + tq.py + goldens) + HY3_TURBOQUANT_KV_PLAN.md.
//
// Layout per (slot, kvh, position), hd == 128 ONLY (the d=128 reference layout):
//   K row — 52 B: [0,32)   2-bit Lloyd-Max codes (coord j at bits [2j, 2j+2), LSB-first bitstream)
//                 [32,48)  QJL signs (coord j = bit j; 1 -> +1, 0 -> -1)
//                 [48,50)  ||r|| fp16 LE (unit-key residual norm, fp16-rounded at pack)
//                 [50,52)  ||k|| fp16 LE
//   V row — 52 B:  [0,48)  3-bit Lloyd-Max codes (coord j at bits [3j, 3j+3), LSB-first)
//                  [48,50) ||v|| fp16 LE
//                  [50,52) pad (uniform 52-B row stride so K and V share one row size)
// Row index = (slot*nkv + kvh) * stride + pos, row-major; stride in positions.
//
// Math (row vectors; "apply Pi" = x @ Pi^T == Pi·x as a column-vector matvec):
//   K:  kn = ||k||; xhat = k/kn; y = Pi·xhat; idx = argmin |y - cb2| (ties -> higher index, the
//       reference's searchsorted-right pair rule); r = xhat - Pi^T cb2[idx];
//       sign = sign(S·r) with sign(0) := +1; rn = ||r||.  Pack (idx, sign, fp16 rn, fp16 kn).
//       Unbiased reconstruction: k~ = kn·(Pi^T cb2[idx] + sqrt(pi/2)/d · rn · S^T sign).
//   V:  vn = ||v||; y = Pi·(v/vn); idx = argmin |y - cb3|;  v~ = vn·Pi^T cb3[idx] (MSE).
//   Score (the dual dot; the engine's attention): s = kn·( <Pi q, cb2[idx]> + sqrt(pi/2)/d·rn·<S q, sign> )
//       — NO 1/sqrt(hd) factor: this estimator IS <q,k> (REPORT §4.1).
//   PV:  acc in the ROTATED domain (vn·cb3[idx]); the splitk merge applies out = Pi^T·acc once
//       per (token, head).
//
// Determinism: every row is a pure function of its (kvh, pos) inputs computed by ONE block with a
// FIXED thread mapping and fixed-order reductions (no atomics) — batch-invariant by construction.
// The fp16 norms are the __float2half (round-to-nearest-even) values, exactly the reference's
// numpy '<f2' pack.
// ===================================================================================================
#define TQ_HD 128
// TurboQuant K-channel bit width, selected at compile time by -DTQ_B3 (the b=3 variant:
// GB10_KV_TQ=3). The b=2 build (no define) is the golden-anchored E4 layout; the b=3 build
// packs the K codes at 3 bits (48 B) -> K row 68 B = codes[0,48) | signs[48,64) | rn[64,66)
// | kn[66,68); V rows stay 3-bit codes + fp16 norm (50 meaningful B) padded to the shared row
// size so K and V caches keep one uniform stride. Both layouts share the TQ_TAB_* tables (the
// b=3 K codebook IS the existing cb3 block — 8 ascending centroids).
#ifdef TQ_B3
#define TQ_K_BITS     3
#define TQ_ROW_BYTES  68
#define TQ_SIGN_OFF   48
#define TQ_RN_OFF     64
#define TQ_KN_OFF     66
#else
#define TQ_K_BITS     2
#define TQ_ROW_BYTES  52
#define TQ_SIGN_OFF   32
#define TQ_RN_OFF     48
#define TQ_KN_OFF     50
#endif
// Host table buffer (f32, uploaded once — src/gpu.rs build_tq_tables). Offsets in floats:
//   [0, 16384)       Pi    row-major Pi[j][i]
//   [16384, 32768)   PiT   row-major PiT[j][i] = Pi[i][j]
//   [32768, 49152)   S     row-major S[j][i]
//   [49152, 65536)   ST    row-major ST[j][i] = S[i][j]
//   [65536, 65540)   cb2   4 ascending 2-bit centroids
//   [65540, 65548)   cb3   8 ascending 3-bit centroids  <- K codebook for the b=3 build
//   [65548]          qjl_scale  sqrt(pi/2)/128 (== scale.bin)
#define TQ_TAB_PI    0
#define TQ_TAB_PIT   16384
#define TQ_TAB_S     32768
#define TQ_TAB_ST    49152
#define TQ_TAB_CB2   65536
#define TQ_TAB_CB3   65540
#define TQ_TAB_SCALE 65548
#if TQ_K_BITS == 3
#define TQ_ENCODE_SMEM_BYTES 4176   // tq_encode_rows dynamic smem (must match the host launch)
#else
#define TQ_ENCODE_SMEM_BYTES 4160   // tq_encode_rows dynamic smem (must match the host launch)
#endif

__device__ __forceinline__ float tq_half_at(const unsigned char* row, int off) {
    return __half2float(__ushort_as_half(*(const unsigned short*)(row + off)));
}

// Nearest centroid, ties -> HIGHER index (the reference's searchsorted-right + `<=` pair rule).
__device__ __forceinline__ int tq_quant2(const float y, const float* cb2) {
    const float d0 = fabsf(y - cb2[0]), d1 = fabsf(y - cb2[1]),
                d2 = fabsf(y - cb2[2]), d3 = fabsf(y - cb2[3]);
    int i = 0; float bd = d0;
    if (d1 <= bd) { i = 1; bd = d1; }
    if (d2 <= bd) { i = 2; bd = d2; }
    if (d3 <= bd) { i = 3; bd = d3; }
    return i;
}
__device__ __forceinline__ int tq_quant3(const float y, const float* cb3) {
    int i = 0; float bd = fabsf(y - cb3[0]);
    #pragma unroll
    for (int k = 1; k < 8; k++) { const float d = fabsf(y - cb3[k]); if (d <= bd) { i = k; bd = d; } }
    return i;
}

// One block = one (kvh, position); blockDim.x = 128 (one thread per coord). Encodes the K row
// (b-bit Lloyd-Max + QJL signs + fp16 ||r||/||k||, b = TQ_K_BITS) and the V row (3-bit + fp16
// ||v||) from bf16 inputs. Dynamic smem layout (floats):
//   xk[128] | xv[128] | wk[128] | wv[128] | ridx[128] | vidx[128] | sgn[128] | wpk[4] | wpv[4]
//   | bkn[1] | bvn[1] | brn[1]  — then bytes: kcs[128] | kss[128] | vcs[128] | krow[TQ_ROW_BYTES] | vrow[52].
__device__ __forceinline__ void tq_encode_rows(
        const float* tables,
        const __nv_bfloat16* krow_in, const __nv_bfloat16* vrow_in,
        unsigned char* kout, unsigned char* vout, float* sh) {
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const float* pi  = tables + TQ_TAB_PI;
    const float* pit = tables + TQ_TAB_PIT;
    const float* s   = tables + TQ_TAB_S;
    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    float* xk = sh;            float* xv = sh + 128;
    float* wk = sh + 256;      float* wv = sh + 384;
    float* ridx = sh + 512;    float* vidx = sh + 640;
    float* sgn = sh + 768;
    float* wpk = sh + 896;     float* wpv = sh + 900;
    float* bkn = sh + 904;     float* bvn = sh + 908;    float* brn = sh + 912;
    unsigned char* kcs = (unsigned char*)(sh + 916);    // 128 B
    unsigned char* kss = kcs + 128;                     // 128 B
    unsigned char* vcs = kss + 128;                     // 128 B
    unsigned char* krow = vcs + 128;                    // TQ_ROW_BYTES
    unsigned char* vrow = krow + TQ_ROW_BYTES;          // 52 B (50 meaningful + 2 pad)

    xk[t] = b2f(krow_in[t]);
    xv[t] = b2f(vrow_in[t]);
    // per-warp sums of squares (deterministic halving tree), then fixed warp-order combine
    float pk = xk[t] * xk[t], pv = xv[t] * xv[t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { pk += __shfl_xor_sync(0xffffffffu, pk, off); pv += __shfl_xor_sync(0xffffffffu, pv, off); }
    __syncthreads();
    if (lane == 0) { wpk[warp] = pk; wpv[warp] = pv; }
    __syncthreads();
    if (t == 0) {
        bkn[0] = sqrtf(wpk[0] + wpk[1] + wpk[2] + wpk[3]);
        bvn[0] = sqrtf(wpv[0] + wpv[1] + wpv[2] + wpv[3]);
    }
    __syncthreads();
    const float kn = bkn[0], vn = bvn[0];
    xk[t] *= 1.0f / fmaxf(kn, 1e-30f);                 // unit key/value (clamp like the reference)
    xv[t] *= 1.0f / fmaxf(vn, 1e-30f);
    __syncthreads();
    // rotate: y = Pi·x  (y[t] = sum_i Pi[t][i]·x[i], fp32 ascending i)
    float acc = 0.0f;
    const float* pirow = pi + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * xk[i];
    wk[t] = acc;
    acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * xv[i];
    wv[t] = acc;
    __syncthreads();
#if TQ_K_BITS == 3
    ridx[t] = (float)tq_quant3(wk[t], cb3);
#else
    ridx[t] = (float)tq_quant2(wk[t], cb2);
#endif
    vidx[t] = (float)tq_quant3(wv[t], cb3);
    __syncthreads();
    // residual in the ORIGINAL domain: r[t] = xhat[t] - (Pi^T cb)[t], (Pi^T cb)[t] = sum_j Pi[j][t]·cb[j]
    acc = 0.0f;
    const float* pitrow = pit + t * TQ_HD;
#if TQ_K_BITS == 3
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb3[(int)ridx[j]];
#else
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb2[(int)ridx[j]];
#endif
    const float r = xk[t] - acc;
    sgn[t] = r;                                        // stage r for the S·r matvec
    __syncthreads();
    acc = 0.0f;
    const float* srow = s + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += srow[i] * sgn[i];
    const float sv = (acc >= 0.0f) ? 1.0f : -1.0f;    // sign(0) := +1; kept in a register —
    // sgn still holds the staged r and is NEVER overwritten (a cross-thread write here would race
    // the other warps' S·r reads).
    // rn = ||r|| (same fixed reduction)
    pk = r * r;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) pk += __shfl_xor_sync(0xffffffffu, pk, off);
    __syncthreads();
    if (lane == 0) wpk[warp] = pk;
    __syncthreads();
    if (t == 0) brn[0] = sqrtf(wpk[0] + wpk[1] + wpk[2] + wpk[3]);
    __syncthreads();
    const float rn = brn[0];
    // ---- stage + pack K row ----
    kcs[t] = (unsigned char)(int)ridx[t];
    kss[t] = (sv > 0.0f) ? 1u : 0u;
    __syncthreads();
#if TQ_K_BITS == 3
    if (t < 48) {                                      // 3-bit LSB-first bitstream (like the V pack)
        const int g = t;
        unsigned char v = 0;
        for (int j = (8 * g) / 3; j <= (8 * g + 7) / 3; j++) {
            if (j >= TQ_HD) break;
            const int shift = 3 * j;
            const unsigned char c = (unsigned char)(kcs[j] & 7u);
            if (shift >= 8 * g) {                      // starts in this byte
                const int lo = shift - 8 * g;          // 0..7; a 3-bit code may straddle
                const int take = min(3, 8 - lo);       // bits of c that fit in this byte
                v |= (unsigned char)((c & ((1u << take) - 1u)) << lo);
            } else {                                   // started in the previous byte
                const int hi = 8 * g - shift;          // 1..2
                v |= (unsigned char)(c >> hi);
            }
        }
        krow[g] = v;
    }
#else
    if (t < 32) {                                      // 4 two-bit codes per byte (LSB-first)
        unsigned char v = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) v |= (unsigned char)((kcs[4 * t + j] & 3u) << (2 * j));
        krow[t] = v;
    }
#endif
    if (t < 16) {                                      // 8 sign bits per byte (bit j = coord j)
        unsigned char v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) v |= (unsigned char)(kss[8 * t + j] << j);
        krow[TQ_SIGN_OFF + t] = v;
    }
    if (t == 0) {                                      // fp16 LE norms (round-to-nearest-even)
        *(unsigned short*)(krow + TQ_RN_OFF) = __half_as_ushort(__float2half(rn));
        *(unsigned short*)(krow + TQ_KN_OFF) = __half_as_ushort(__float2half(kn));
    }
    // ---- stage + pack V row ----
    vcs[t] = (unsigned char)(int)vidx[t];
    __syncthreads();
    if (t < 48) {                                      // 3-bit LSB-first bitstream
        const int g = t;
        unsigned char v = 0;
        for (int j = (8 * g) / 3; j <= (8 * g + 7) / 3; j++) {
            if (j >= TQ_HD) break;
            const int shift = 3 * j;
            const unsigned char c = (unsigned char)(vcs[j] & 7u);
            if (shift >= 8 * g) {                      // starts in this byte
                const int lo = shift - 8 * g;          // 0..7; a 3-bit code may straddle
                const int take = min(3, 8 - lo);       // bits of c that fit in this byte
                v |= (unsigned char)((c & ((1u << take) - 1u)) << lo);
            } else {                                   // started in the previous byte
                const int hi = 8 * g - shift;          // 1..2
                v |= (unsigned char)(c >> hi);
            }
        }
        vrow[g] = v;
    }
    if (t == 0) *(unsigned short*)(vrow + 48) = __half_as_ushort(__float2half(vn));
    __syncthreads();
    // K rows are TQ_ROW_BYTES (52 for b=2, 68 for b=3); V rows carry 50 meaningful bytes + pad.
    if (t < TQ_ROW_BYTES) kout[t] = krow[t];
    if (t < 52) vout[t] = vrow[t];
}

// write_kv_b_tq — pack-at-write one token's K/V (decode/verify path). One block per (b, kvh).
// grid(B*nkv), blockDim 128, smem TQ_ENCODE_SMEM_BYTES.
extern "C" __global__ void write_kv_b_tq(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, const float* tables,
        const int* pos, int stride, int nkv, int B, const int* slot_ids) {
    const int idx = blockIdx.x;
    if (idx >= B * nkv) return;
    const int b = idx / nkv;
    const int h = idx % nkv;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    extern __shared__ float sh[];
    tq_encode_rows(tables,
        k_new + ((long long)b * nkv + h) * TQ_HD,
        v_new + ((long long)b * nkv + h) * TQ_HD,
        k_cache + crow * TQ_ROW_BYTES, v_cache + crow * TQ_ROW_BYTES, sh);
}

// write_kv_prefill_tq — pack-at-write N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_tq(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, const float* tables,
        int stride, int nkv, int N, int pos_start) {
    const int idx = blockIdx.x;
    if (idx >= N * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    const long long crow = ((long long)h * stride) + (pos_start + t);
    extern __shared__ float sh[];
    tq_encode_rows(tables,
        k_new + ((long long)t * nkv + h) * TQ_HD,
        v_new + ((long long)t * nkv + h) * TQ_HD,
        k_cache + crow * TQ_ROW_BYTES, v_cache + crow * TQ_ROW_BYTES, sh);
}

// rotate_q_tq — the "rotate once per (token, head)" step: qr = Pi·q and qs = S·q into an
// interleaved f32 buffer qrqs[b*nh*2*hd + qh*2*hd + {2j, 2j+1}] = (qr_j, qs_j). One block per
// (b, qh), 128 threads. Deterministic per (b, qh) — fixed mapping, ascending-order dots.
// F0: `q_pitch` is the q row pitch (nh*hd packed, or the fused qkv mtot for an offset view).
extern "C" __global__ void rotate_q_tq(float* qrqs, const __nv_bfloat16* q,
        const float* tables, int nh, int hd, int B, int q_pitch) {
    const int idx = blockIdx.x;
    if (idx >= B * nh) return;
    const int b = idx / nh;
    const int qh = idx % nh;
    const float* pi = tables + TQ_TAB_PI;
    const float* s  = tables + TQ_TAB_S;
    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd;
    float* out = qrqs + ((long long)b * nh + qh) * 2 * hd;
    extern __shared__ float sh[];                      // hd floats (hd == 128 == blockDim)
    sh[threadIdx.x] = b2f(qrow[threadIdx.x]);
    __syncthreads();
    const int t = threadIdx.x;
    float acc = 0.0f;
    const float* pirow = pi + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * sh[i];
    out[2 * t] = acc;
    acc = 0.0f;
    const float* srow = s + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += srow[i] * sh[i];
    out[2 * t + 1] = acc;
}

// Dequant rows [pos_start, pos_start+n_pos) of the packed cache into scratch in CACHE layout
// ([kvh][pos][hd]): row (h, pos_start+t) lands at (h*out_stride + pos_start + t)*hd. K dequant is
// the reference's decode_k (unbiased reconstruction): k~ = kn·(Pi^T cb2[idx] + qjl·rn·S^T sign);
// V dequant is decode_v: v~ = vn·Pi^T cb3[idx]. One block per (t, kvh), 128 threads, smem 1536 B.
template<bool F32>
__device__ __forceinline__ void tq_dequant_rows(
        const float* tables, const unsigned char* kcache, const unsigned char* vcache,
        long long row, long long out_base, void* out_k, void* out_v, float* sh) {
    const int t = threadIdx.x;
    const float* pit = tables + TQ_TAB_PIT;
    const float* st  = tables + TQ_TAB_ST;
    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    const float qjl_scale = tables[TQ_TAB_SCALE];
    const unsigned char* krow = kcache + row * TQ_ROW_BYTES;
    const unsigned char* vrow = vcache + row * TQ_ROW_BYTES;
    float* kcode = sh;          // [0,128)
    float* ksgn  = sh + 128;    // [128,256)
    float* vcode = sh + 256;    // [256,384)
#if TQ_K_BITS == 3
    {   // 3-bit coord t: bits [3t, 3t+3) — two bytes read separately (byte-unaligned safe)
        const int b0 = (3 * t) >> 3;
        const int off = (3 * t) & 7;
        const unsigned short kw = (unsigned short)krow[b0] | ((unsigned short)krow[b0 + 1] << 8);
        kcode[t] = (float)((kw >> off) & 7);
    }
#else
    kcode[t] = (float)((krow[t >> 2] >> (2 * (t & 3))) & 3);
#endif
    ksgn[t]  = ((krow[TQ_SIGN_OFF + (t >> 3)] >> (t & 7)) & 1) ? 1.0f : -1.0f;
    {   // 3-bit coord t: bits [3t, 3t+3) — two bytes read separately (byte-unaligned safe)
        const int b0 = (3 * t) >> 3;
        const int off = (3 * t) & 7;
        const unsigned short vw = (unsigned short)vrow[b0] | ((unsigned short)vrow[b0 + 1] << 8);
        vcode[t] = (float)((vw >> off) & 7);
    }
    __syncthreads();
    const float rn = tq_half_at(krow, TQ_RN_OFF), kn = tq_half_at(krow, TQ_KN_OFF);
    const float vn = tq_half_at(vrow, 48);
    const float* pitrow = pit + t * TQ_HD;
    float acc = 0.0f;
#if TQ_K_BITS == 3
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb3[(int)kcode[j]];
#else
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb2[(int)kcode[j]];
#endif
    float acc2 = 0.0f;
    const float* strow = st + t * TQ_HD;
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc2 += strow[j] * ksgn[j];
    const float kval = kn * (acc + qjl_scale * rn * acc2);
    float accv = 0.0f;
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) accv += pitrow[j] * cb3[(int)vcode[j]];
    const float vval = vn * accv;
    if (F32) {
        ((float*)out_k)[out_base + t] = kval;
        ((float*)out_v)[out_base + t] = vval;
    } else {
        ((__nv_bfloat16*)out_k)[out_base + t] = f2b(kval);
        ((__nv_bfloat16*)out_v)[out_base + t] = f2b(vval);
    }
}

extern "C" __global__ void dequant_kv_tq(__nv_bfloat16* out_k, __nv_bfloat16* out_v,
        const unsigned char* kcache, const unsigned char* vcache, const float* tables,
        int nkv, int stride, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x;
    if (idx >= n_pos * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    extern __shared__ float sh[];
    const long long row = (long long)h * stride + (pos_start + t);
    // OUT row = the launch's t (the q4 convention): the one-shot scratch passes pos_start=0,
    // the incremental mirror shifts the OUT base pointer by the watermark rows instead.
    const long long out_base = ((long long)h * out_stride + t) * TQ_HD;
    tq_dequant_rows<false>(tables, kcache, vcache, row, out_base, out_k, out_v, sh);
}

// dequant_kv_tq_full — debug/oracle twin with f32 output (host-checkable full dequant).
extern "C" __global__ void dequant_kv_tq_full(float* out_k, float* out_v,
        const unsigned char* kcache, const unsigned char* vcache, const float* tables,
        int nkv, int stride, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x;
    if (idx >= n_pos * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    extern __shared__ float sh[];
    const long long row = (long long)h * stride + (pos_start + t);
    const long long out_base = ((long long)h * out_stride + t) * TQ_HD;
    tq_dequant_rows<true>(tables, kcache, vcache, row, out_base, out_k, out_v, sh);
}

// compact_kv_tq — verbatim copy of packed 52-B rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; position-addressable rows, so a byte copy suffices).
extern "C" __global__ void compact_kv_tq(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int dir) {
    const int rb = TQ_ROW_BYTES;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * rb;
    if (idx >= total) return;
    const int k = idx / (nkv * rb);
    const int rem = idx % (nkv * rb);
    const int h = rem / rb;
    const int dv = rem % rb;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * rb + dv;
    const long long soff = ((long long)k * nkv + h) * rb + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

extern "C" __global__ void write_kv_b(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                                       const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nkv * hd;
    if (idx >= total) return;
    int b = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int d = rem % hd;
    int slot = slot_ids[b];
    long long coff = ((long long)slot * nkv + h) * stride + pos[b];
    k_cache[coff * hd + d] = k_new[(long long)b * nkv * hd + h * hd + d];
    v_cache[coff * hd + d] = v_new[(long long)b * nkv * hd + h * hd + d];
}


// gqa_attn_flash USED TO LIVE HERE and it is deliberately gone.
//
// It was the "n_splits == 1" fast path for decode attention. Having two kernels that both compute
// decode attention is precisely what broke the lossless-MTP contract: a decode and a verify at the
// same position could pick DIFFERENT ONES (see the note on gqa_attn_splitk), and they did not agree
// to the last bit -- this one divided in fp32, the other round-tripped the numerator through bf16.
// One kernel, one code path. gqa_attn_splitk with ns=1 is this kernel, and costs the same.

// ---- batched sigmoid gate (in place): attn[b,nh*hd] *= sigmoid(gate[b,nh*hd]) ----
extern "C" __global__ void sigmoid_gate_b(__nv_bfloat16* attn, const __nv_bfloat16* gate, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) attn[i] = f2b(b2f(attn[i]) * (1.0f / (1.0f + __expf(-b2f(gate[i])))));
}

// ---- gather RoPE cos/sin from pre-computed tables [max_pos, rdim] ----
extern "C" __global__ void gather_rope_b(float* out_cos, float* out_sin,
    const float* cos_table, const float* sin_table,
    const int* pos, int rdim, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * rdim) return;
    int b = idx / rdim;
    int d = idx % rdim;
    int p = pos[b];
    out_cos[idx] = cos_table[(long long)p * rdim + d];
    out_sin[idx] = sin_table[(long long)p * rdim + d];
}

// ---- gather embedding rows from bf16 table → bf16 output ----
// embed_table: [vocab, h] bf16 (row-major); tokens: [B]; out: [h*B] bf16 (col-major)
extern "C" __global__ void embed_gather_b(__nv_bfloat16* out, const void* embed_table_v,
    const int* tokens, int h, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * h) return;
    int b = idx / h;
    int d = idx % h;
    int tok = tokens[b];
    const __nv_bfloat16* embed_table = (const __nv_bfloat16*)embed_table_v;
    out[(long long)b * h + d] = embed_table[(long long)tok * h + d];
}

// ---- logits penalty: repetition (multiplicative) + presence (flat) + frequency (count) ----
// pen_tokens: [max_pen, B] i32 (-1 = unused); pen_counts: [max_pen, B] i16
// rep_factor > 1.0 penalizes repeated tokens multiplicatively (HF/CTRL formula)
// presence > 0 subtracts flat from any token that appeared
// frequency > 0 subtracts frequency * count from each token
extern "C" __global__ void rep_penalty_b(__nv_bfloat16* logits, const int* pen_tokens,
    const short* pen_counts, int n_pen,
    const float* rep_factors, const float* presences, const float* frequencies,
    int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    __nv_bfloat16* col = logits + (long long)b * vocab;
    float rep_factor = rep_factors[b];
    float presence = presences[b];
    float frequency = frequencies[b];
    for (int i = tid; i < n_pen; i += blockDim.x) {
        int tok = pen_tokens[(long long)b * n_pen + i];
        if (tok < 0 || tok >= vocab) continue;
        short count = pen_counts[(long long)b * n_pen + i];
        float v = b2f(col[tok]);
        if (rep_factor > 1.0f) v = v > 0.0f ? v / rep_factor : v * rep_factor;
        v -= presence;
        v -= frequency * (float)count;
        col[tok] = f2b(v);
    }
}

// ---- rep_penalty over FP32 logits (hy_v3 enable_lm_head_fp32): same formula, float in/out ----
extern "C" __global__ void rep_penalty_f32_b(float* logits, const int* pen_tokens,
    const short* pen_counts, int n_pen,
    const float* rep_factors, const float* presences, const float* frequencies,
    int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    float* col = logits + (long long)b * vocab;
    float rep_factor = rep_factors[b];
    float presence = presences[b];
    float frequency = frequencies[b];
    for (int i = tid; i < n_pen; i += blockDim.x) {
        int tok = pen_tokens[(long long)b * n_pen + i];
        if (tok < 0 || tok >= vocab) continue;
        short count = pen_counts[(long long)b * n_pen + i];
        float v = col[tok];
        if (rep_factor > 1.0f) v = v > 0.0f ? v / rep_factor : v * rep_factor;
        v -= presence;
        v -= frequency * (float)count;
        col[tok] = v;
    }
}

// ---- batched argmax: logits [vocab, B] bf16 col-major → token_ids [B] ----
extern "C" __global__ void argmax_b(int* token_ids, const __nv_bfloat16* logits, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    int* s_idxs = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    const __nv_bfloat16* col = logits + (long long)b * vocab;
    float my_max = -1e30f;
    int my_idx = 0;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = b2f(col[i]);
        if (v > my_max) { my_max = v; my_idx = i; }
    }
    s_vals[tid] = my_max;
    s_idxs[tid] = my_idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            if (s_vals[tid + s2] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s2];
                s_idxs[tid] = s_idxs[tid + s2];
            }
        }
        __syncthreads();
    }
    if (tid == 0) token_ids[b] = s_idxs[0];
}

// ---- argmax over FP32 logits (hy_v3 enable_lm_head_fp32): same scan + tree, float reads ----
extern "C" __global__ void argmax_f32_b(int* token_ids, const float* logits, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    int* s_idxs = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    const float* col = logits + (long long)b * vocab;
    float my_max = -1e30f;
    int my_idx = 0;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = col[i];
        if (v > my_max) { my_max = v; my_idx = i; }
    }
    s_vals[tid] = my_max;
    s_idxs[tid] = my_idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            if (s_vals[tid + s2] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s2];
                s_idxs[tid] = s_idxs[tid + s2];
            }
        }
        __syncthreads();
    }
    if (tid == 0) token_ids[b] = s_idxs[0];
}

// ===================== DEVICE-RESIDENT TOKEN LOOP (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE) =========
// Four small kernels that move the per-step host bookkeeping into the captured graphs so a clean
// decode step becomes: replay graph -> dtoh [batch] i32 -> bookkeeping -> replay again. All are
// pool-free (persistent DecodeBuffers only), host-sync-free, allocation-free: capture-safe by
// construction. Bit-exactness strategy: the selection kernels (argmax_*/sample_*) and the penalty
// APPLICATION kernels (rep_penalty_*) are NOT touched — only the window/seed BOOKKEEPING moves
// on-device, and it is pure integer set-membership, so the rebuilt arrays are byte-identical to
// today's host rebuild (src/batch.rs batched_decode). Keep the tie-break-relevant kernels' block
// sizes and comparison order untouched — block size is part of the argmax tie-break.

// ---- graph-head id flow: device copy + position advance. grid (1,1,1), block (B,1,1), B<=16.
// tokens[b] <- token_ids[b] (last step's selected id becomes this step's embed input);
// pos[b] += 1 (every Phase-B lane advances exactly one position per step).
extern "C" __global__ void ids_advance_b(int* tokens, const int* token_ids, int* pos, int B) {
    int b = threadIdx.x;
    if (b >= B) return;
    tokens[b] = token_ids[b];
    pos[b] += 1;
}

// ---- penalty ring push (graph epilogue, after the selection kernel).
// grid (B,1,1), block (1,1,1). ring layout: ring[b*n_pen .. b*n_pen+n_pen), most-recent at head-1.
// ring_state[b] packs (head << 8) | len; len saturates at n_pen (64 = MAX_PEN_TOKENS).
extern "C" __global__ void penalty_ring_push_b(int* ring, int* ring_state,
                                               const int* token_ids, int n_pen, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int st = ring_state[b];
    int head = st >> 8, len = st & 0xff;
    ring[b * n_pen + head] = token_ids[b];
    ring_state[b] = (((head + 1) % n_pen) << 8) | (len < n_pen ? len + 1 : n_pen);
}

// ---- penalty window rebuild: ring (duplicates allowed, MRU first) -> deduped (token,count) list
// in pen_tokens/pen_counts, BYTE-IDENTICAL to the host rebuild (MRU->LRU; first occurrence keeps
// the slot; count = total occurrences inside the window). grid (B,1,1), block (n_pen,1,1).
// -1 sentinel: rep_penalty_*_b skips it.
extern "C" __global__ void penalty_window_b(const int* ring, const int* ring_state,
                                            int* pen_tokens, short* pen_counts,
                                            int n_pen, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int st = ring_state[b];
    int head = st >> 8, len = st & 0xff;
    int i = threadIdx.x;                       // slot i == i-th most-recent entry
    int tok = -1; short cnt = 0;
    if (i < len) {
        int pos_i = (head - 1 - i + n_pen) % n_pen;
        tok = ring[b * n_pen + pos_i];
        bool first = true;
        for (int j = 0; j < i; j++) {          // any more-recent duplicate? then not first
            int pos_j = (head - 1 - j + n_pen) % n_pen;
            if (ring[b * n_pen + pos_j] == tok) { first = false; break; }
        }
        if (first) {
            cnt = 1;
            for (int j = i + 1; j < len; j++) {
                int pos_j = (head - 1 - j + n_pen) % n_pen;
                if (ring[b * n_pen + pos_j] == tok) cnt++;
            }
        } else { tok = -1; cnt = 0; }
    }
    pen_tokens[b * n_pen + i] = tok;
    pen_counts[b * n_pen + i] = cnt;
}

// ---- device seed schedule (sample graph only), bit-identical to the host schedule
// (splitmix64 + rng_u32 with RNG_DOM_SAMPLE at idx 0, src/batch.rs:367-383).
// grid (1,1,1), block (B,1,1). keys advance exactly once per EXECUTED step; capture warmups
// never see real keys (they are uploaded at admission, after capture).
__device__ __forceinline__ unsigned long long splitmix64_dev(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
#define RNG_DOM_SAMPLE_DEV 0x53414D504C450001ULL   // == RNG_DOM_SAMPLE, src/batch.rs:377
extern "C" __global__ void seed_advance_b(unsigned int* seeds_out,
                                          unsigned long long* keys, int B) {
    int b = threadIdx.x;
    if (b >= B) return;
    unsigned long long k = keys[b];
    // rng_u32(key, DOM, idx=0): the idx term is 0*GOLDEN == 0, so key ^ DOM ^ 0.
    seeds_out[b] = (unsigned int)(splitmix64_dev(k ^ RNG_DOM_SAMPLE_DEV) >> 32);
    keys[b] = splitmix64_dev(k);               // advance exactly once per executed step
}

// ===================== PREFILL KERNELS (sequential over N positions) =====================

// ---- write_kv_prefill: write N positions of K/V into ONE slot's cache ----
// k_new,v_new: [nkv*hd, N] bf16; k_cache,v_cache: [nkv, stride, hd] bf16.
// pos_start: absolute write offset (0 for from-scratch prefill; = current decode position for
// causal-append / MTP verify, so the N new K/V vectors land at positions pos_start..pos_start+N-1).
extern "C" __global__ void write_kv_prefill(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, int stride, int nkv, int hd, int N, int pos_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * nkv * hd;
    if (idx >= total) return;
    int t = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int d = rem % hd;
    // The cache holds `stride` positions per head. A prompt longer than the cache used to write straight
    // past the end of it — silently, corrupting whatever allocation came next, which surfaced as two
    // identical prefills producing different answers. The caller must reject an over-long prompt; this
    // guard is the backstop that turns a heap corruption into a dropped write.
    if (pos_start + t >= stride) return;
    long long pos = (long long)h * stride + (pos_start + t);
    k_cache[pos * hd + d] = k_new[(long long)t * nkv * hd + h * hd + d];
    v_cache[pos * hd + d] = v_new[(long long)t * nkv * hd + h * hd + d];
}

// ---- gqa_attn_prefill: causal attention for N positions of one sequence ----
// q: [nh*hd, N] bf16; k_cache,v_cache: [nkv, stride, hd] bf16; out: [nh*hd, N] bf16
// ONE BLOCK PER (QUERY TILE OF hd/32, query_head); blockDim.x = hd. Each warp owns ONE query of the
// tile (hd=256 → 8 queries/block, hd=128 → 4). Works for any hd that is a positive multiple of 32
// and <= 512 (SK_DPL_MAX register cap). pos_start: absolute position of the first of the N tokens
// (0 for a from-scratch prompt prefill; for a causal-append -- the MTP head's prompt prime -- = the
// position the append starts at, so column i attends to KV[0 .. pos_start+i]).
//
// TWO BUGS DIED HERE, and they are worth remembering separately.
//
// (1) The ORIGINAL did a full 256-thread tree reduction, WITH __syncthreads(), FOR EVERY SINGLE KEY:
//     ~11 barriers per key on a loop whose length is the context. Measured on an 8K prompt: 2.27 s
//     per layer. Fixed by making the q.k dot product a WARP SHUFFLE -- no shared memory, no barriers.
//
// (2) That fix took it to 348 ms/layer and I nearly stopped, because it "looked like flash attention"
//     -- but it still gave EVERY QUERY ITS OWN BLOCK, and every block re-read the entire K/V history.
//     At 8K that is nh * N^2/2 = 536M (query,key) pairs x 1024 B = 549 GB of L1/L2 traffic per layer.
//     Against the measured 348 ms that is ~1.6 TB/s: pinned to the L2 bandwidth ceiling. The arithmetic
//     says only ~46 ms of it was instruction issue, so it was never compute -- it was re-reading K/V
//     16-thousand times. STATIC INSPECTION CANNOT SEE THIS; only the traffic arithmetic can.
//
// So: a block now owns a TILE of query positions (warp w takes query tile*QT + w) and the warps sweep
// the SAME keys together. One K/V row fetch from L2 now serves QT queries instead of 1 -- an 8x cut
// (at hd=256) in the binding resource. Each warp carries the complete running softmax (m, l, acc)
// for ITS OWN query in registers across the whole key range, so no warp ever needs another warp's
// partial: the cross-warp merge, and all of the kernel's shared memory, are simply GONE.
//
// Causality: warps in a block are within QT positions of each other, so they diverge only over the
// last handful of keys -- masked with a predicate, not a branch out of the loop.
extern "C" __global__ void gqa_attn_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh, int nkv, int hd, float scale, int N, int pos_start) {
    const int QT = blockDim.x >> 5;              // query positions per block == warps per block (hd/32)
    const int blk = blockIdx.x;
    const int tile = blk / nh, qh = blk % nh;
    const int kvh = qh / (nh / nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;                   // head dims per lane (8 when hd=256, 4 when hd=128)

    const int t = tile * QT + warp;            // THIS warp's query position
    const bool active = (t < N);
    const int pc = pos_start + t + 1;          // keys this warp's query attends to
    // Keys ANY warp in this block needs: the block sweeps to the last query's horizon.
    const int tlast = min(tile * QT + QT - 1, N - 1);
    const int pc_blk = pos_start + tlast + 1;

    // Per-lane register slices: compile-time trip count SK_DPL_MAX keeps qv/acc in registers
    // (AGENTS.md §4.1); lanes i >= DPL are predicated off and contribute exact +0.0f terms, so an
    // hd=256 launch is bit-identical to the old hardcoded-256 kernel.
    const __nv_bfloat16* qrow = q + (long long)(active ? t : 0) * (nh * hd) + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;

    // ===================== WHAT THIS KERNEL IS ACTUALLY BOUND BY: STILL UNKNOWN =====================
    //
    // Measured, 9B, 8K prompt, per attention layer (nsys, 16 launches = 2 prefills x 8 layers):
    //
    //   barrier-per-key (the original)            2270 ms
    //   warp-shuffle, one block per QUERY          348 ms
    //   + query tiling, 8 queries share each key   310 ms   <-- 11%, NOT the 7x predicted
    //   + unroll 4 keys per iteration              312 ms   <-- 0%
    //
    // TWO HYPOTHESES, BOTH WRONG, BOTH KILLED BY MEASUREMENT:
    //
    //   1. "L2-bandwidth bound." One block per query re-read the whole K/V: nh*N^2/2 = 536M
    //      (query,key) pairs x 1024 B = 549 GB/layer, which against 348 ms is ~1.6 TB/s -- right at the
    //      L2 ceiling. Query tiling cuts that traffic 8x. It bought 11%. So L2 bandwidth was NOT the
    //      binding constraint, and the fact that the arithmetic *landed on* the L2 ceiling was a
    //      coincidence that read like a diagnosis.
    //
    //   2. "Then it's latency: sharing a key across 8 warps collapsed memory-level parallelism 8x."
    //      Plausible, and it predicts that giving each warp 4 independent K/V rows in flight recovers
    //      it. Implemented (KU=4). It bought ZERO. So it is not MLP either.
    //
    // What is left is instruction issue: ~536M warp-key iterations, each doing 16 scalar FMA, 16 bf16
    // -> f32 converts, a 5-step shuffle reduction and 2 transcendentals. That is ~25e9 warp-instructions
    // for one layer, and the arithmetic says an issue-bound kernel should take ~80 ms, not 310 -- so
    // even that does not close. `ncu` would answer it in one run (stall reasons, issue efficiency), but
    // GPU performance counters are locked on this box (ERR_NVGPUCTRPERM -- see PROFILING.md), so this
    // stays honestly open rather than getting a third confident story.
    //
    // The structural answer is almost certainly tensor cores: an mma.sync QK^T + PV replaces ~16 scalar
    // FMAs and 16 converts per lane per key with a handful of instructions on bf16 fragments. That is
    // the FlashAttention-2 design and it is the open question for the expert.
    //
    // Keeping the query tiling (it is a real 11% and it is strictly less DRAM traffic); NOT keeping the
    // key unroll (it bought nothing and cost readability).
    for (int tt = 0; tt < pc_blk; tt++) {
        // All warps issue this same address: one fetch, QT queries served.
        const __nv_bfloat16* krow = kb + (long long)tt * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;                              // now uniform across the warp

        if (active && tt < pc) {                 // causal mask (differs only over the last <QT keys)
            const float m_new = fmaxf(m, s);
            const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
            const __nv_bfloat16* vrow = vb + (long long)tt * hd;
            #pragma unroll
            for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
            m = m_new;
            l = l * a_old + a_cur;
        }
    }

    // A warp owns its query's ENTIRE softmax -- nothing to merge, nothing to share.
    // U2 class-1 (PLAN/U2_FIX_DRAFT.md): normalise with a single division acc/l, matching
    // gqa_attn_reduce's acc/l — acc*(1/l) rounds differently and a near-tie argmax can flip.
    if (active) {
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) orow[i] = f2b(l > 0.0f ? acc[i] / l : 0.0f);
    }
}

// ===================== TILED PREFILL ATTENTION (materialised S, cuBLAS GEMMs) =====================
//
// The scalar gqa_attn_prefill is bound by instruction issue, not by memory: query tiling cut its L1/L2
// traffic 8x for 11%, and putting 4 independent K/V rows in flight per warp bought 0%. A PERFECT scalar
// kernel bottoms out around 25 ms/layer (549 GFLOP of QK^T+PV against ~22 TFLOPS of f32 CUDA-core
// issue); a MEDIOCRE tensor-core one starts at ~9 ms. That is a choice between hardware units, not
// between tunings -- so the fix is to route the two GEMMs through tensor cores.
//
// This is the cuBLAS-materialised form: S = Q^T K into a scratch tile, online softmax, O += P V^T.
// It is slower than a fused mma flash kernel (it pays to write and re-read S), but it (a) captures most
// of the win, and (b) is numerically simple enough to be the ORACLE that a later mma kernel is fuzzed
// against. This kernel has produced two confident wrong diagnoses already; the next version gets a
// reference to check against.
//
// PREFILL IS OUTSIDE THE BATCH-INVARIANCE CONTRACT. That contract binds decode <-> verify; prefill
// feeds both identically. So we may tile, split and reorder here freely -- the discipline that governs
// every other kernel in this file does not apply. (Still deterministic: fixed tiling, fixed order.)
//
// Layouts, all column-major so cuBLAS can eat them directly:
//   Q_h  [hd x N ] ld = nh*hd   base = q  + qh*hd            (head-major within a token)
//   K_h  [hd x pc] ld = hd      base = kc + kvh*stride*hd
//   S^T  [Bc x Br] f32, ld = Bc, one slab per kv head   <-- TRANSPOSED, deliberately
//   O    [Br x hd] f32, ld = Br, one slab per query head -- carried across key tiles
//
// S IS STORED TRANSPOSED FOR ONE REASON: COALESCING. The softmax runs one block per query row, walking
// that row's keys. With S as [Br x Bc] column-major, adjacent threads (adjacent keys j) are Br floats
// = 4 KB apart -- every lane's load is its own 32-byte sector, and the kernel ate 85% of the attention
// time despite the GEMMs being fast. Storing S^T makes a query's row contiguous. Same DRAM-sector
// lesson as the GEMM scale reads; it does not stop being true because cuBLAS wrote the buffer.
//
// ---- attn_softmax_tile: the online-softmax step between the two GEMMs ----
// One block per (query row i, kv head). Applies scale + causal mask, updates the running (m, l) for the
// row, rescales that row's O accumulator by exp(m_old - m_new), and writes P = exp(s - m_new) in bf16
// for the PV GEMM.
//
// `r` is the index of the query head WITHIN its GQA group: the caller issues one batched GEMM per r
// (batched over kv heads), because a GQA group's 4 query heads share one K/V and cuBLAS cannot express
// "stride 0 for 4, then jump".
//
// The argument list is packed because cudarc's launch tuple caps at 12. `scale` is folded into the
// QK^T GEMM's alpha (free), nh/gqa/r share one int, and (s0, pc) share one i64.
extern "C" __global__ void attn_softmax_tile(
    const float* __restrict__ S, __nv_bfloat16* __restrict__ P,
    float* __restrict__ O, float* __restrict__ mrun, float* __restrict__ lrun,
    int Br, int Bc, int hd, int nh_gqa_r, int rows, int qpos0, long long s0_pc) {
    const int nh  = nh_gqa_r / 10000;
    const int gqa = (nh_gqa_r / 100) % 100;
    const int r   = nh_gqa_r % 100;
    const int s0  = (int)(s0_pc >> 32);
    const int pc  = (int)(s0_pc & 0xffffffffLL);
    (void)nh;

    const int i   = blockIdx.x;          // query row within the tile
    const int kvh = blockIdx.y;          // kv head
    const int qh  = kvh * gqa + r;       // query head
    const int tid = threadIdx.x, nt = blockDim.x;

    const bool live = (i < rows);
    const int qpos = qpos0 + i;                      // absolute position of the query
    const int kmax = min(pc - 1, qpos);              // causal: last key this query may see

    const float* Ss = S + (long long)kvh * Br * Bc;  // this kv head's S^T slab
    __nv_bfloat16* Ps = P + (long long)kvh * Br * Bc;
    float* Os = O + (long long)qh * Br * hd;
    float* mr = mrun + (long long)qh * Br;
    float* lr = lrun + (long long)qh * Br;

    extern __shared__ float red[];

    // ---- pass 1: row max over the valid keys of this tile ----
    float mx = -1e30f;
    if (live) {
        for (int j = tid; j < Bc; j += nt) {
            const int key = s0 + j;
            if (key <= kmax) mx = fmaxf(mx, Ss[(long long)i * Bc + j]);
        }
    }
    red[tid] = mx;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float tile_max = red[0];
    __syncthreads();

    const float m_old = live ? mr[i] : 0.0f;
    const float m_new = fmaxf(m_old, tile_max);
    // A tile that is entirely masked (or a dead row) must not touch (m, l) or O -- and exp(-inf - -inf)
    // is NaN, so guard rather than rely on arithmetic.
    const bool any = live && (tile_max > -1e29f);
    const float alpha = (any && m_old > -1e29f) ? __expf(m_old - m_new) : (any ? 0.0f : 1.0f);

    // ---- pass 2: P = exp(s - m_new), and the row sum ----
    float sum = 0.0f;
    if (live) {
        for (int j = tid; j < Bc; j += nt) {
            const int key = s0 + j;
            float p = 0.0f;
            if (any && key <= kmax) p = __expf(Ss[(long long)i * Bc + j] - m_new);
            Ps[(long long)i * Bc + j] = f2b(p);      // masked -> exactly 0, so PV adds nothing
            sum += p;
        }
    } else {
        for (int j = tid; j < Bc; j += nt) Ps[(long long)i * Bc + j] = f2b(0.0f);
    }
    red[tid] = sum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float tile_sum = red[0];

    // ---- rescale this row's O accumulator, then commit (m, l) ----
    // alpha == 1.0f exactly when the running max did not move this tile (the common case once the
    // max stabilizes), and x * 1.0f is bit-identical to x for every finite accumulator — so skipping
    // the pass is value-exact and avoids a fully strided (8x-amplified) read-modify-write of O.
    if (live && any && alpha != 1.0f) {
        for (int d = tid; d < hd; d += nt) Os[(long long)d * Br + i] *= alpha;
    }
    if (live && tid == 0) {
        mr[i] = any ? m_new : m_old;
        lr[i] = any ? (lr[i] * alpha + tile_sum) : lr[i];
    }
}

// ---- attn_finalize: O / l -> the [nh*hd, N] output layout ----
extern "C" __global__ void attn_finalize(
    __nv_bfloat16* __restrict__ out, const float* __restrict__ O, const float* __restrict__ lrun,
    int Br, int hd, int nh, int t0, int N) {
    // Coalesced on both sides via a 32-row smem transpose. The old mapping (grid (br, nh),
    // thread-per-d) read the column-major O slab at O[d*Br + i] — one 32 B sector per 4 B element,
    // ~8x read waste per row tile. Load phase: warp lanes read consecutive ROWS for a fixed d
    // (contiguous 128 B). Write phase: one warp per row writes the contiguous bf16 run. Values are
    // untouched — the same per-element o / l in the same order.
    __shared__ float tile[32][256 + 1];                  // +1 kills bank conflicts; hd <= 256 asserted by the caller
    const int i0 = blockIdx.x * 32;
    const int qh = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;        // block is (32, 8)
    const float* Os = O + (long long)qh * Br * hd;

    for (int k = 0; k < 32; k++) {
        const int d = ty * 32 + k;
        const int i = i0 + tx;
        tile[tx][d] = (i < Br && d < hd) ? Os[(long long)d * Br + i] : 0.0f;
    }
    __syncthreads();

    const int warp = ty, lane = tx;                      // 8 warps x 32 lanes; 4 rows per warp
    for (int r = 0; r < 4; r++) {
        const int i = i0 + warp * 4 + r;
        const int t = t0 + i;
        if (i >= Br || t >= N) continue;
        const float l = lrun[(long long)qh * Br + i];
        for (int d = lane; d < hd; d += 32)
            out[(long long)t * (nh * hd) + (long long)qh * hd + d] =
                f2b(l > 0.0f ? tile[warp * 4 + r][d] / l : 0.0f);
    }
}

// ---- attn_tile_init: reset the per-tile running softmax state ----
extern "C" __global__ void attn_tile_init(float* O, float* mrun, float* lrun, int Br, int hd, int nh) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long no = (long long)nh * Br * hd;
    if (idx < no) O[idx] = 0.0f;
    const long long nml = (long long)nh * Br;
    if (idx < nml) { mrun[idx] = -1e30f; lrun[idx] = 0.0f; }
}

// ---- attn_prefill_fa_b: fused flash-attention prefill (ONE launch replaces the QK^T/softmax/PV
// tile walk of attn_prefill_tiled; no S/P global materialization, no per-tile init/finalize) ----
//
// Prefill ONLY — gated by GB10_FA_PREFILL=1 and n >= PF_MIN at the Rust call site; the tiled path
// below it stays the default and the verify path is untouched (verify never reaches PF_MIN).
//
// Semantics = the attn_softmax_tile/attn_finalize numerics, one pass in registers:
//   score = (Q[t,hh,:] . K[kvh,k,:]) * rsqrt(hd);  online softmax with running (m, l, O);
//   Out = O / l.  Keys 0..pos_start+t inclusive (causal, chunked prefill — the prefix is already
//   in the cache from earlier chunks + write_kv_prefill).
//
// Layouts (identical to the cuBLAS calls it replaces):
//   Q  bf16 row-major  [N, nh*hd]        Q[t,hh,d]  = q + (t*nh+hh)*hd + d
//   K,V bf16 head-major [nkv, stride, hd] K[h,k,d]  = kc + (h*stride+k)*hd + d
//   Out bf16 [N, nh*hd]                  (same as Q)
//
// THIS BODY IS THE SCALAR REFERENCE ORACLE (warp-per-row), validated relL2 2.3e-05 vs a host
// reference across (N,pos_start) = (256,0),(130,0),(256,512),(130,501) on .13. It is the
// correctness fallback for geometries the tensor-core kernel does not cover.
extern "C" __global__ void attn_prefill_fa_sc_b(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int hd, int stride, int pos_start)
{
    const int WARPS = 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int qhead = blockIdx.y;
    const int kvh  = qhead / gqa;
    const int t = blockIdx.x * WARPS + warp;
    if (t >= N) return;
    const int qpos = pos_start + t;
    const int dpl = hd / 32;                       // dims per lane (8 at hd=256)
    const __nv_bfloat16* Qp = Q + ((long long)t * nh + qhead) * hd;
    float qv[8];
    for (int i = 0; i < dpl; i++) qv[i] = b2f(Qp[lane * dpl + i]);
    float m = -1e30f, l = 0.0f;
    float ov[8];
    for (int i = 0; i < dpl; i++) ov[i] = 0.0f;
    const float inv_sqrt_hd = rsqrtf((float)hd);
    const __nv_bfloat16* Kb = K + (long long)kvh * stride * hd;
    const __nv_bfloat16* Vb = V + (long long)kvh * stride * hd;
    for (int k = 0; k <= qpos; k++) {
        const __nv_bfloat16* Kp = Kb + (long long)k * hd;
        const __nv_bfloat16* Vp = Vb + (long long)k * hd;
        float kv[8], vv[8];
        for (int i = 0; i < dpl; i++) { kv[i] = b2f(Kp[lane * dpl + i]); vv[i] = b2f(Vp[lane * dpl + i]); }
        float dot = 0.f;
        for (int i = 0; i < dpl; i++) dot += qv[i] * kv[i];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        float score = dot * inv_sqrt_hd;
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * alpha + p;
        for (int i = 0; i < dpl; i++) ov[i] = ov[i] * alpha + p * vv[i];
        m = m_new;
    }
    for (int i = 0; i < dpl; i++) {
        const int d = lane * dpl + i;
        if (d < hd)
            Out[((long long)t * nh + qhead) * hd + d] = f2b(l > 0.0f ? ov[i] / l : 0.0f);
    }
}

// ---- attn_prefill_fa_b: FUSED FLASH-ATTENTION PREFILL, TENSOR-CORE BODY ----
// Implements PLAN/FA_KERNEL_EXPERT_REPORT.md (measured on .13: 15.3/44.7/74.0/103.5 ms at
// N=8192, pc=8K/16K/24K/32K — 4.0x under the 3-kernel path's per-call bar at every geometry,
// ~54 TF/s effective vs the ~75 TF/s bf16 mma ceiling).
//
// Config (report §0): block 192 thr (6 warps) x M=96 rows = 16 tokens x 6 GQA heads
// (token-major; the g heads share the K/V smem tiles = guaranteed 6x reuse), Bc=32 keys
// double-buffered via cp.async (64 KiB dynamic smem, opt-in from Rust), Q+O in registers
// (~255 regs, 4 B spill at hd=256), K ldmatrix NO-transpose (K[key][d] IS B .col), V .trans,
// P->A repack in-register (zero permutations — D-frag col pattern == A-frag k pattern).
// Online-softmax guards replicate attn_softmax_tile exactly (sentinels, any-guard,
// alpha==1.0f rescale skip, l>0 epilogue). P is rounded to bf16 before PV — the SAME rounding
// point as the production tiled path's bf16 p_buf.
//
// Correctness (tool_probe/fa_tc.cu harness, .13): relL2 1.9e-3 vs an fp32 host ref = the bf16-P
// quantization envelope (2^-9), verified by a peaky-softmax structural test (error collapses to
// 1-output-ulp flips concentrated on few-key rows; hd=128 identical). Layouts identical to the
// tiled path; hd must be 128 or 256 (host dispatches, other hd -> scalar/old path); g must
// divide 96 (g=6 -> 16 tokens/block).
__device__ __forceinline__ unsigned fa_cvt2(float lo, float hi) {
    __nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void fa_mma_m16n8k16(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void fa_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fa_ldm_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fa_cp16(__nv_bfloat16* dst, const __nv_bfloat16* src, bool full) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    const int sz = full ? 16 : 0;               // src-size 0 => 16 B zero-fill (cache bound)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(src), "r"(sz));
}

#define FA_BC 32
#define FA_THREADS 192

template <int HD>
__device__ __forceinline__ void fa_tc_body(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int stride, int pos_start, int g, __nv_bfloat16* smem)
{
    const int Br_t = 96 / g;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = (int)threadIdx.x;
    const int kvh  = blockIdx.y;
    const int t0   = (int)(gridDim.x - 1 - blockIdx.x) * Br_t;   // longest blocks first
    const int pc   = pos_start + N;
    const int kmax = min(pos_start + t0 + Br_t - 1, pc - 1);

    const int rr0 = 16*warp + (lane >> 2), rr1 = rr0 + 8;
    const int ttok0 = t0 + rr0 / g, ttok1 = t0 + rr1 / g;
    const int hed0 = kvh*g + rr0 % g, hed1 = kvh*g + rr1 % g;
    const int ctok0 = min(ttok0, N-1), ctok1 = min(ttok1, N-1);   // dead rows: safe loads, no stores
    const long long qb0 = ((long long)ctok0 * nh + hed0) * HD;
    const long long qb1 = ((long long)ctok1 * nh + hed1) * HD;
    const int qpos0 = pos_start + ttok0, qpos1 = pos_start + ttok1;

    unsigned qf[HD/16][4];
    #pragma unroll
    for (int ks = 0; ks < HD/16; ks++) {
        const int doff = 16*ks + 2*(lane & 3);
        qf[ks][0] = *(const unsigned*)(Q + qb0 + doff);
        qf[ks][1] = *(const unsigned*)(Q + qb1 + doff);
        qf[ks][2] = *(const unsigned*)(Q + qb0 + doff + 8);
        qf[ks][3] = *(const unsigned*)(Q + qb1 + doff + 8);
    }

    float o[HD/8][4];
    #pragma unroll
    for (int j = 0; j < HD/8; j++) o[j][0]=o[j][1]=o[j][2]=o[j][3]=0.f;
    float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;
    const float rsqrt_hd = rsqrtf((float)HD);

    auto load_tile = [&](int stage, int s0g) {
        #pragma unroll 4
        for (int c = tid; c < FA_BC * (HD >> 3); c += FA_THREADS) {
            const int key = c / (HD >> 3), chunk = c % (HD >> 3);
            const int gkey = s0g + key;
            const bool valid = gkey < pc;
            const int gk = min(gkey, pc - 1);
            const int cs = (chunk & 24) | ((chunk & 7) ^ (key & 7));   // Swizzle<3,3,3>
            __nv_bfloat16* kd = smem + (stage * FA_BC + key) * HD + (cs << 3);
            __nv_bfloat16* vd = smem + ((2 + stage) * FA_BC + key) * HD + (cs << 3);
            const __nv_bfloat16* gkP = K + ((long long)kvh * stride + gk) * HD + (chunk << 3);
            const __nv_bfloat16* gvP = V + ((long long)kvh * stride + gk) * HD + (chunk << 3);
            fa_cp16(kd, gkP, valid);
            fa_cp16(vd, gvP, valid);
        }
        asm volatile("cp.async.commit_group;");
    };

    load_tile(0, 0);
    if (FA_BC <= kmax) load_tile(1, FA_BC);

    for (int s0 = 0; s0 <= kmax; s0 += FA_BC) {
        const int stage = (s0 / FA_BC) & 1;
        if (s0 + FA_BC <= kmax) asm volatile("cp.async.wait_group 1;");   // retire tile s0's group
        else                    asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // ===== QK^T =====
        float s[4][4];
        #pragma unroll
        for (int j = 0; j < 4; j++) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f;
        {
            const __nv_bfloat16* Kb = smem + stage * FA_BC * HD;
            #pragma unroll
            for (int ks = 0; ks < HD/16; ks++) {
                #pragma unroll
                for (int j2 = 0; j2 < 2; j2++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16*j2 + lane;         ldim = 16*ks; }
                    else if (lane < 16) { lkey = 16*j2 + lane - 8;     ldim = 16*ks + 8; }
                    else if (lane < 24) { lkey = 16*j2 + 8 + lane-16;  ldim = 16*ks; }
                    else                { lkey = 16*j2 + 8 + lane-24;  ldim = 16*ks + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    const unsigned addr = (unsigned)__cvta_generic_to_shared(Kb + lkey*HD + (cs << 3));
                    unsigned r0, r1, r2, r3;
                    fa_ldm_x4(r0, r1, r2, r3, addr);
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    fa_mma_m16n8k16(s[2*j2],   qf[ks], ba);
                    fa_mma_m16n8k16(s[2*j2+1], qf[ks], bb);
                }
            }
        }

        // ===== online softmax (attn_softmax_tile guards) =====
        const bool masked_tile = (s0 + FA_BC > pos_start + t0);
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * rsqrt_hd;
                if (masked_tile) {
                    const int key = s0 + 8*j + 2*(lane&3) + (e&1);
                    if (key > (e < 2 ? qpos0 : qpos1)) v = -1e30f;
                }
                s[j][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 2));
        const bool any0 = mx0 > -1e29f, any1 = mx1 > -1e29f;
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = (any0 && m0 > -1e29f) ? __expf(m0 - mn0) : (any0 ? 0.f : 1.f);
        const float al1 = (any1 && m1 > -1e29f) ? __expf(m1 - mn1) : (any1 ? 0.f : 1.f);
        float su0 = 0.f, su1 = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = (s[j][e] > -1e29f) ? __expf(s[j][e] - (e < 2 ? mn0 : mn1)) : 0.f;
                s[j][e] = p;
                if (e < 2) su0 += p; else su1 += p;
            }
        }
        su0 += __shfl_xor_sync(~0u, su0, 1); su0 += __shfl_xor_sync(~0u, su0, 2);
        su1 += __shfl_xor_sync(~0u, su1, 1); su1 += __shfl_xor_sync(~0u, su1, 2);
        l0 = l0*al0 + su0;  m0 = mn0;
        l1 = l1*al1 + su1;  m1 = mn1;

        if (al0 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD/8; j++) { o[j][0] *= al0; o[j][1] *= al0; }
        }
        if (al1 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD/8; j++) { o[j][2] *= al1; o[j][3] *= al1; }
        }

        // ===== P->A in-register repack + PV =====
        unsigned pa[2][4];
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            pa[ks][0] = fa_cvt2(s[2*ks  ][0], s[2*ks  ][1]);
            pa[ks][1] = fa_cvt2(s[2*ks  ][2], s[2*ks  ][3]);
            pa[ks][2] = fa_cvt2(s[2*ks+1][0], s[2*ks+1][1]);
            pa[ks][3] = fa_cvt2(s[2*ks+1][2], s[2*ks+1][3]);
        }
        {
            const __nv_bfloat16* Vb = smem + (2 + stage) * FA_BC * HD;
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                #pragma unroll
                for (int jn = 0; jn < HD/16; jn++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16*ks + lane;         ldim = 16*jn; }
                    else if (lane < 16) { lkey = 16*ks + 8 + lane - 8; ldim = 16*jn; }
                    else if (lane < 24) { lkey = 16*ks + lane - 16;    ldim = 16*jn + 8; }
                    else                { lkey = 16*ks + 8 + lane-24;  ldim = 16*jn + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    const unsigned addr = (unsigned)__cvta_generic_to_shared(Vb + lkey*HD + (cs << 3));
                    unsigned r0, r1, r2, r3;
                    fa_ldm_x4_trans(r0, r1, r2, r3, addr);
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    fa_mma_m16n8k16(o[2*jn],   pa[ks], ba);
                    fa_mma_m16n8k16(o[2*jn+1], pa[ks], bb);
                }
            }
        }

        __syncthreads();                                  // warps done reading buf[stage]
        if (s0 + 2*FA_BC <= kmax) load_tile(stage, s0 + 2*FA_BC);
    }

    const float il0 = (l0 > 0.f) ? 1.f/l0 : 0.f;
    const float il1 = (l1 > 0.f) ? 1.f/l1 : 0.f;
    if (ttok0 < N) {
        const long long ob = ((long long)ttok0 * nh + hed0) * HD;
        #pragma unroll
        for (int j = 0; j < HD/8; j++)
            *(__nv_bfloat162*)(Out + ob + 8*j + 2*(lane&3)) =
                __float22bfloat162_rn(make_float2(o[j][0]*il0, o[j][1]*il0));
    }
    if (ttok1 < N) {
        const long long ob = ((long long)ttok1 * nh + hed1) * HD;
        #pragma unroll
        for (int j = 0; j < HD/8; j++)
            *(__nv_bfloat162*)(Out + ob + 8*j + 2*(lane&3)) =
                __float22bfloat162_rn(make_float2(o[j][2]*il1, o[j][3]*il1));
    }
}

extern "C" __global__ void __launch_bounds__(FA_THREADS, 1) attn_prefill_fa_b(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ Out,
    int N, int nh, int nkv, int hd, int stride, int pos_start, int g)
{
    extern __shared__ __nv_bfloat16 fa_smem[];    // K[2][32][hd] then V[2][32][hd], opt-in >48 KiB
    if (hd == 256)      fa_tc_body<256>(Q, K, V, Out, N, nh, nkv, stride, pos_start, g, fa_smem);
    else if (hd == 128) fa_tc_body<128>(Q, K, V, Out, N, nh, nkv, stride, pos_start, g, fa_smem);
    // other hd: caller must route to attn_prefill_fa_sc_b / the tiled path
}



// ---- conv1d prefill: FULLY PARALLEL over (channel, position) ----
// in: [conv_dim, N] bf16 col-major; out: [conv_dim, N] bf16; state: [conv_dim, k] f32; w: [conv_dim, k] f32
//
// THIS IS NOT A RECURRENCE, AND TREATING IT AS ONE COST 163 ms OF EVERY 8K PREFILL.
//
// The old kernel ran one thread per channel and walked N positions sequentially, shuffling a k-wide
// window as if the convolution had a carried dependence. It does not. A width-k CAUSAL DEPTHWISE conv
// is a stencil: output t reads inputs t-(k-1) .. t and nothing else. Written out, with `xf` the virtual
// input = (the k-1 tail carried in `state`) followed by this chunk's N inputs:
//
//     xf[i]           = state[c][i + 1]              for i < k-1        (x[-(k-1)] .. x[-1])
//     xf[(k-1) + t]   = in[t][c]                     for t in [0, N)
//     out[t][c]       = silu( sum_{j<k} w[c][j] * xf[t + j] )
//
// so every (c, t) is independent. One thread each; conv_dim*N threads instead of conv_dim.
//
// THE ACCUMULATION ORDER IS LOAD-BEARING. This kernel also runs on the MTP verify path, where column 0
// must stay BIT-IDENTICAL to a 1-token decode (which uses conv1d_b). Same values, same ascending-j FMA
// order => same bits. Do not "optimise" this into a tree reduction.
//
// It had to stop being in-place: thread (c,t) reads inputs t-(k-1)..t, which other threads are writing.
// Hence separate in/out buffers -- the caller swaps them.
//
// mid_state: optional PER-COLUMN state checkpoints. Column t's post-state is written to
// mid_state + t*(conv_dim*k), for t in [0, N-2] -- one snapshot per position we might roll back to.
// NULL disables snapshotting (regular prefill passes NULL). The post-state after token t is just the
// window ending at t: state_after_t[j] = xf[t + j]. Which is exactly what this thread already read.
//
// (Snapshotting used to cover only column 0 -- "the committed token is always accepted". True ONLY at
// depth 2. At depth >= 3, accepting nacc=1 and rejecting the rest rolled the recurrent state back past
// the accepted draft, so the GDN state went stale and every later draft was built on it.)
// win_src (tree drafting): per-node window sources. win_src[t*k + j] = the j-th window source for node
// t, encoded as: v >= 0 -> draft-block input in[v*conv_dim + c]; v < 0 -> carried state st0[v + k].
// nullptr => the CHAIN formula v = t + j - (k-1) (consecutive positions) -- byte-identical to the
// pre-tree kernel, and what the main prefill always uses. Only the verify passes a real table.
// F0: `in_stride` is the `in` row pitch (conv_dim packed, or the fused GDN mtot for a qkv view).
extern "C" __global__ void conv1d_prefill(__nv_bfloat16* out, const __nv_bfloat16* in,
                                          const float* state, const float* w,
                                          int conv_dim, int k, int N, float* mid_state,
                                          const int* win_src, const int* slot_ids, int in_stride) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)conv_dim * N;
    if (idx >= total) return;
    const int t = (int)(idx / conv_dim);        // position
    const int c = (int)(idx % conv_dim);        // channel -- adjacent threads => coalesced over c

    // Forest: a negative win_src entry reads the committed conv tail from `state`. slot_ids non-null
    // (verify) => `state` is the layer BASE and this column's lane is slot_ids[t]; null (prefill) =>
    // `state` is pre-offset to the single slot. Single-lane verify: slot_ids[t]==slot, identical.
    const long long cslot = slot_ids ? (long long)slot_ids[t] : 0;
    const float* st0 = state + cslot * ((long long)conv_dim * k) + (long long)c * k;
    const float* wc  = w + (long long)c * k;

    // k == 4 specialization: the window lives in SCALAR REGISTERS. A runtime-indexed local array
    // (float win[8] under #pragma unroll 1, the old code) cannot be register-allocated — ptxas
    // reports a 32 B stack frame and every access is a DRAM-backed load+store, in the kernel that
    // runs conv_dim*N threads per GDN layer. Same values, same ascending-j FMA order => bit-identical
    // to the generic path (and to conv1d_b, as the note above requires).
    if (k == 4) {
        const int v0 = win_src ? win_src[t * 4 + 0] : (t - 3);
        const int v1 = win_src ? win_src[t * 4 + 1] : (t - 2);
        const int v2 = win_src ? win_src[t * 4 + 2] : (t - 1);
        const int v3 = win_src ? win_src[t * 4 + 3] : (t - 0);
        const float w0 = (v0 < 0) ? st0[v0 + 4] : b2f(in[(long long)v0 * in_stride + c]);
        const float w1 = (v1 < 0) ? st0[v1 + 4] : b2f(in[(long long)v1 * in_stride + c]);
        const float w2 = (v2 < 0) ? st0[v2 + 4] : b2f(in[(long long)v2 * in_stride + c]);
        const float w3 = (v3 < 0) ? st0[v3 + 4] : b2f(in[(long long)v3 * in_stride + c]);
        float acc = wc[0] * w0;
        acc += wc[1] * w1;
        acc += wc[2] * w2;
        acc += wc[3] * w3;                          // ASCENDING — see the bit-identity note
        out[(long long)t * conv_dim + c] = f2b(silu_f(acc));
        if (mid_state && t < N - 1) {
            float* ms = mid_state + (long long)t * conv_dim * 4 + (long long)c * 4;
            ms[0] = w0; ms[1] = w1; ms[2] = w2; ms[3] = w3;
        }
        return;
    }

    float win[8];                                // generic fallback (k != 4; unused in production)
    #pragma unroll 1
    for (int j = 0; j < k; j++) {
        const int v = win_src ? win_src[t * k + j] : (t + j - (k - 1));
        win[j] = (v < 0) ? st0[v + k]
                         : b2f(in[(long long)v * in_stride + c]);
    }
    float acc = 0.0f;
    #pragma unroll 1
    for (int j = 0; j < k; j++) acc += wc[j] * win[j];   // ASCENDING -- see the bit-identity note
    out[(long long)t * conv_dim + c] = f2b(silu_f(acc));

    if (mid_state && t < N - 1) {
        float* ms = mid_state + (long long)t * conv_dim * k + (long long)c * k;
        for (int j = 0; j < k; j++) ms[j] = win[j];
    }
}

// ---- conv1d prefill: carry the final window back into `state` ----
// Separate launch, because conv1d_prefill READS `state` from every thread: writing it there would race.
// One thread per channel; reads the whole final window before writing any of it, so the in-place update
// is safe even when N < k (short verify widths, where the window still straddles the old state).
// F0: `in_stride` is the `in` row pitch (conv_dim packed, or the fused GDN mtot for a qkv view).
extern "C" __global__ void conv1d_prefill_state(float* state, const __nv_bfloat16* in,
                                                int conv_dim, int k, int last_t, int lane_len,
                                                const int* slot_ids, int in_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    // Carry ONE lane's final conv window into its committed slot. The lane ends at column `last_t` and
    // spans `lane_len` columns; a window position before the lane start (offset < 0) falls back to the
    // lane's own committed tail (via slot_ids[last_t]). Launched once per lane by the host. Single lane
    // (last_t==N-1, lane_len==N; slot_ids null=>pre-offset): byte-identical to the pre-forest carry.
    const long long lslot = slot_ids ? (long long)slot_ids[last_t] : 0;
    float* st = state + lslot * ((long long)conv_dim * k) + (long long)c * k;

    if (k == 4) {
        // Scalar registers — same local-memory trap as conv1d_prefill (32 B stack frame via ptxas).
        const int off0 = lane_len - 4;
        const long long base = (long long)last_t - 3;
        const float n0 = (off0 + 0 < 0) ? st[off0 + 4] : b2f(in[(base + 0) * in_stride + c]);
        const float n1 = (off0 + 1 < 0) ? st[off0 + 5] : b2f(in[(base + 1) * in_stride + c]);
        const float n2 = (off0 + 2 < 0) ? st[off0 + 6] : b2f(in[(base + 2) * in_stride + c]);
        const float n3 = (off0 + 3 < 0) ? st[off0 + 7] : b2f(in[(base + 3) * in_stride + c]);
        st[0] = n0; st[1] = n1; st[2] = n2; st[3] = n3;   // read-all-then-write-all
        return;
    }

    float nx[8];
    for (int j = 0; j < k; j++) {
        const int off = (lane_len - k) + j;      // offset from lane start; < 0 => committed tail
        nx[j] = (off < 0) ? st[off + k]
                          : b2f(in[(long long)(last_t - (k - 1) + j) * in_stride + c]);
    }
    for (int j = 0; j < k; j++) st[j] = nx[j];   // read-all-then-write-all: no self-overlap hazard
}

// ===================== SPLIT-K ATTENTION (Flash-Decoding) =====================
// Parameterized for any head_dim that is a positive multiple of 32 and <= SK_HD_MAX
// (qwen3.5: 256; hy_v3: 128; DeepSeek: 512). hd arrives inside nh_packed =
// (nh << 20) | (hd << 10) | nkv  (nh < 2048, hd/nkv <= 1023).

#define MAX_VERIFY 16   // MUST match gpu.rs MAX_VERIFY (verify width cap; path table row stride)

// One block per (batch, query_head, split); blockDim.x = 256 = 8 warps. Each block owns a contiguous
// slice of the key range and emits a partial softmax (m, l, acc) that gqa_attn_reduce merges.
//
// THE PER-KEY BARRIER: this used to run a 256-thread __syncthreads() tree reduction for EVERY key --
// ~90k serialised barriers per block at an 8K context. Now: warp-shuffle dot with 8 warps striding the
// split's keys, each carrying its own register-resident (m, l, acc[8]).
//
// ===================== THE LOSSLESS-MTP CONTRACT LIVES HERE =====================
//
// Column k of an N-wide verify MUST be bit-identical to a 1-token decode at the same position. That
// forces attention for a query at position p to be a pure function of (q, KV[0..p], pc) -- it may not
// depend on B, on the other columns, or on anything derived from them. Two violations lived here:
//
//   1. n_splits was derived from batch*nh ("split more when there are too few CTAs"), so a decode
//      split the keys 6 ways and a 4-wide verify split them 2 ways.
//   2. Deriving it from max_pc instead was ALSO WRONG, and this is the subtle one: a decode has
//      max_pc = pos+1 while a verify has max_pc = pos_start+N. Whenever those straddle a multiple of
//      256 the two disagree -- and if one lands on n_splits==1 it took a DIFFERENT KERNEL entirely
//      (gqa_attn_flash, which keeps the numerator in fp32) while the other round-tripped the
//      numerator through bf16 here. End-to-end MTP was silently non-lossless; it passed anyway most
//      runs, because a 1-ulp difference rarely flips an argmax. That is the worst kind of bug: it
//      fails as a coin toss, so a green gate proves nothing.
//
// So `ns` is computed HERE, from THIS COLUMN's OWN pc, and nothing else. The caller sizes the grid
// with an upper bound (pc_b <= max_pc => ns_b <= ns_grid) and surplus blocks return without writing.
// gqa_attn_flash is gone: one kernel, one code path, no way for decode and verify to take different
// ones. out_acc is fp32 for the same reason -- the bf16 round-trip was pure lossy noise.
__device__ __forceinline__ int sk_nsplits(int pc) { return min(max(pc / 256, 1), 32); }

// RANK-SPACE split-K (tree drafting, review §2). A tree column must execute a DECODE's exact iteration
// space over its LOGICAL prefix; only the rank->slot address map changes. So `pos` now carries the
// LOGICAL position (prefix + ancestor path incl. self); it governs `pc` and the split structure. The
// keys enter via `path`: rank r < pos_start is a prefix key at slot r; rank r >= pos_start is the
// (r-pos_start)-th on-block ancestor, at slot pos_start + path[b*MAX_VERIFY + (r-pos_start)].
// path == nullptr => identity (t = r): the decode path and any plain chain, byte-identical to pre-tree.
// This keeps `ns` a pure function of each column's own logical key count, so column 0 (and every
// ACCEPTABLE node, whose emitted logits must match a decode) stays bit-identical -- unlike a
// slot-derived pc, which drifts by ulps on a 256-straddle (the third n_splits bug). ONE loop, ONE
// warp stride, address indirection only: splitting into prefix+draft loops would regroup the per-warp
// partials and break bit-identity even for a chain.
// stride is packed into bs_packed to fit the 12-arg launch cap: bits 0-18 stride, 19-24 ns_grid, 25-30 B.
extern "C" __global__ void gqa_attn_splitk(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos, long long bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    // scale = 1/sqrt(hd), computed in-kernel to free a launch slot for `col_pos_start` (12-arg cudarc
    // ceiling). For hd=256, sqrtf(256)=16 exactly, so 1/16 = 0.0625f is bit-identical to the old
    // constant — the qwen3.5 decode is unchanged to the last bit. FOREST: `col_pos_start[b]` is column
    // b's lane prefix boundary; null => pos[0] (single lane / tree / decode) — byte-identical to pre-forest.
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);   // grid fan-out (an UPPER BOUND on every column's ns)
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    // F0: the q row pitch (bits 31-49) — the fused qkv mtot for offset views, nh*hd for packed.
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    // R2.1: b INNERMOST => the B verify columns of one (qh, split) are co-scheduled and share the
    // L2 read of the same K/V chunk (was: b-major, re-reading the chunk from DRAM B times). Arithmetic
    // and reduction order unchanged => bit-identical. Bijection over the exact nh*ns_grid*B grid.
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos[b] + 1;                  // LOGICAL key count (chain: unchanged)
    // Prefix boundary: per-column (its lane's committed length) for a forest; pos[0] otherwise (used iff path).
    const int pos_start = col_pos_start ? col_pos_start[b] : pos[0];
    const int slot = slot_ids[b];

    const int ns = sk_nsplits(pc);             // THIS COLUMN's split count -- from ITS OWN pc alone
    if (split >= ns) return;                   // surplus block: a wider column needed the fan-out
    const int split_size = (pc + ns - 1) / ns;
    const int start = split * split_size;
    const int end = min(start + split_size, pc);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NW = blockDim.x >> 5;            // hd/32 warps (8 at hd=256)
    const int DPL = hd >> 5;                   // head dims per lane (8 at hd=256, 4 at hd=128)

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {                          // this split has no keys
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < hd) out_acc[idx * hd + threadIdx.x] = 0.0f;
        return;
    }

    // Per-lane register slices, compile-time trip count SK_DPL_MAX so qv/acc stay in registers
    // (AGENTS.md §4.1); lanes i >= DPL are predicated off and contribute exact +0.0f terms, so an
    // hd=256 launch executes bit-identically to the old hardcoded-256 kernel.
    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;

    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;

    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd]; // rank -> slot
        const __nv_bfloat16* krow = kb + (long long)t * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)t * hd;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
        m = m_new;
        l = l * a_old + a_cur;
    }

    // Merge this block's warp-partials into one partial softmax, in FIXED warp order.
    extern __shared__ float sh[];
    float* sacc = sh;                     // NW * hd
    float* sm   = sh + NW * hd;           // NW
    float* sl   = sm + NW;                // NW
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
        out_acc[idx * hd + d] = num;              // UNNORMALISED fp32, paired with (mg, den)
        if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
    }
}

// ===================================================================================================
// gqa_attn_splitk_q4 — the 4-bit-KV twin of gqa_attn_splitk, SAME split structure, SAME reduction
// order, SAME merge. Only the K/V source changes: packed per-16 affine blocks (see kvq16_pack)
// instead of bf16 rows. Each lane dequantizes its DPL-dim slice from the block that holds it
// (DPL divides 16, so a lane never straddles blocks): scale = e4m3 byte, codes = its DPL nibbles.
// A (kvh, position, dim) dequantizes to the same fp32 value in decode, verify, and every split —
// the lossless-MTP contract is preserved by construction (see the note on the bf16 original).
extern "C" __global__ void gqa_attn_splitk_q4(
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
    // F0: the q row pitch (bits 31-49) — the fused qkv mtot for offset views, nh*hd for packed.
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

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

    const int rb = KVQ_ROW_BYTES(hd);
    const int lane_blk = (lane * DPL) / KVQ_BLK;       // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;       // its first nibble within the block
    const long long kvbase = ((long long)slot * nkv + kvh) * (long long)stride * rb;
    const unsigned char* kb = k_cache + kvbase + lane_blk * 12;
    const unsigned char* vb = v_cache + kvbase + lane_blk * 12;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd];
        const unsigned char* krow = kb + (long long)t * rb;
        const float ksc = e4m3_f(krow[8]);
        const unsigned short* kcodes = (const unsigned short*)(krow + lane_off / 2);
        float kdq[SK_DPL_MAX];
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) kdq[i] = 0.0f;
        #pragma unroll
        for (int u = 0; u < SK_DPL_MAX / 4; u++) {
            if (u >= DPL / 4) break;
            const int pk = (int)kcodes[u];   // SIGNED: `(pk & 0xF) - 7` in unsigned wraps nibble<7 to ~4.3e9
            kdq[u * 4 + 0] = (float)((pk & 0xF) - 7) * ksc;
            kdq[u * 4 + 1] = (float)(((pk >> 4) & 0xF) - 7) * ksc;
            kdq[u * 4 + 2] = (float)(((pk >> 8) & 0xF) - 7) * ksc;
            kdq[u * 4 + 3] = (float)(((pk >> 12) & 0xF) - 7) * ksc;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * kdq[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const unsigned char* vrow = vb + (long long)t * rb;
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


// ================== TENSOR-CORE QUANTIZED GEMM — one fixed shape, flat in N ==================
//
// The ONLY GEMM the quantized serving path uses, at every decode/verify width. The SIMT dequant-GEMV
// it replaced is deleted, not kept as a fallback: it needed a different weight layout (row-major),
// which cannot carry a fused multi-tensor weight, and a second layout is exactly the kind of trap
// that has bitten this project repeatedly. The problem it solves, measured on 9B:
//
//     9B         decode (N=1)   verify (N=4)
//     bf16        73.3 ms        83.9 ms      <- cuBLAS at N=4: tensor cores, flat in N
//     NVFP4       31.6 ms        84.4 ms      <- our SIMT GEMV: 2.7x, and lands on bf16's time
//
// Quantization bought NOTHING at the width speculation actually needs. Per 16-element K-block a SIMT
// thread pays ~16 weight-decode ops (constant in N) + N*16 FMAs (linear in N). At N=1 the kernel sat
// at 83% of the bandwidth roofline — i.e. with almost no compute headroom — so the linear-in-N FMA
// term tipped it compute-bound almost immediately. Landing exactly on bf16's tensor-core time is what
// "the SIMT FMA pipe became the roofline" looks like. The fix is not a better GEMV. It is to move the
// FMAs off the SIMT pipe entirely.
//
// THE DESIGN (Marlin's, in essence):
//
//   * Weights stay PACKED in VRAM and are permuted offline into mma-fragment order (quant.rs).
//     One contiguous aligned load per lane per k-step; a warp's tile is one contiguous byte run.
//   * Dequantize in REGISTERS to bf16 fragments. This cost is per-WEIGHT, so it is constant in N.
//     It is the term that must not scale, and it doesn't.
//   * Feed `mma.sync.m16n8k16`. The N*K products now cost ~nothing: one HMMA covers 8 columns of N.
//     The kernel goes back to being bound by the packed-weight bytes — flat in N until activation
//     traffic matters, which at N<=16 it does not.
//
// WHY THIS IS STILL BITWISE BATCH-INVARIANT (the guarantee greedy-MTP losslessness rests on):
//
// The instinct is that tensor cores must cost us determinism, because cuBLAS does. But cuBLAS breaks
// invariance by SELECTING DIFFERENT KERNELS AND TILINGS PER SHAPE, not because `mma` is inherently
// unstable — an mma instruction is a deterministic function of its inputs. So:
//
//   1. Inside one mma: hardware reduction order is fixed for a given instruction shape, and the 8
//      output columns are INDEPENDENT dot products. Columns 1..15 cannot perturb column 0.
//   2. Across k-slices: the K loop is a fixed stride with no dependence on N.
//   3. Across the 8 warps' split-K partials: summed in warp-index order in shared memory. Fixed.
//      (Not atomicAdd — note vLLM's Marlin path sets VLLM_MARLIN_USE_ATOMIC_ADD=1, i.e. the
//      competition's reduction order is scheduler-dependent. That is a choice, not a necessity.)
//
// Then the move that makes invariance TRIVIAL rather than merely argued: N IS ALWAYS PADDED TO 16.
// Decode (N=1) and verify (N=2..16) execute the identical instruction sequence; the padded columns
// are separate accumulators that are computed and thrown away. Column 0 is bit-identical at every N
// BY CONSTRUCTION. The padding is free where it counts — the kernel is bound by packed-weight bytes,
// which are the same at N=1 and N=16, and the wasted HMMA slots ride a pipe the GEMV left idle.
//
// This also collapses the engineering surface: ONE kernel, no N-dispatch boundary. The old
// bf16-falls-to-cuBLAS-above-N=2 split was both a perf cliff and, historically, how invariance broke.

#define MMA_NW 8                       // warps per block; they split K and reduce in fixed order
#define MMA_SMEM (MMA_NW * 32 * 8)     // [8 acc slots][8 warps][32 lanes] f32

// D[16x8] += A[16x16] * B[16x8], bf16 in, f32 accumulate. A row-major, B col-major (= our X layout:
// X[n][k] with k contiguous IS "col" for B, so no transpose anywhere).
__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two E2M1 nibbles -> a bf16x2 register, each PRE-SCALED by the block's E4M3 scale.
//
// Folding the scale in here is exact, and that is not a lucky accident: E2M1 magnitudes carry 1
// mantissa bit, E4M3 carries 3, so the product needs at most 5 — and bf16 has 7. Every scaled weight
// is therefore representable with ZERO rounding, and the mma's f32 accumulate is at least as accurate
// as the SIMT kernel's. The f32 TENSOR scale (`inv_gs`, arbitrary bits) is NOT folded here — it is
// constant over the whole tensor and is applied once to the f32 accumulator at the end.
__device__ __forceinline__ uint32_t fp4_pair_bf16(uint32_t byte, float s) {
    __nv_bfloat162 v = __floats2bfloat162_rn(e2m1_f(byte & 0x0F) * s, e2m1_f((byte >> 4) & 0x0F) * s);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t fp8_pair_bf16(uint32_t lo, uint32_t hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(e4m3_f((uint8_t)lo), e4m3_f((uint8_t)hi));
    return *reinterpret_cast<uint32_t*>(&v);
}

// Merge the 8 warps' fragment accumulators in FIXED warp order (shared memory, no atomics) and
// return this thread's element of the 16x16 tile: thread tid owns (rlane, rslot) = (tid&31, tid>>5).
// This is the whole cross-warp reduction of both the plain epilogue and the split-K partial; the
// caller owns the (m, n) fragment map and whatever it does with the value.
__device__ __forceinline__ float mma_warp_reduce(float* sh, float acc[2][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    // Re-slice the block: 256 threads, 256 (lane, acc-slot) pairs, one each. Keeping `lane` in the
    // low bits keeps both the store above and the load below bank-conflict-free.
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order
    return v;
}

// Reduce the 8 warps' fragment accumulators in fixed warp order, scale, and scatter to C[n][m].
// `rs` is the FP8 per-row scale (nullptr for NVFP4, where `gs` carries the tensor scale instead).
// `Cf` (optional): write the FP32 accumulator instead of rounding to bf16. This is the whole of the
// FP32-preserving TP=2 reduction — the row-parallel partial leaves the GEMM UNROUNDED, crosses the wire
// in FP32, is summed in FP32 on both ranks, and is rounded to bf16 exactly ONCE in tp_wait_add. The only
// remaining difference from a single-node full-K accumulation is FP32 addition association, which is
// reassociation-class, not a precision loss.
__device__ __forceinline__ void mma_epilogue(float* sh, float acc[2][4], __nv_bfloat16* C,
                                             const float* rs, const float* gs, int mt, int M, int N,
                                             float* Cf = nullptr) {
    const float v = mma_warp_reduce(sh, acc);

    // Invert the mma C-fragment map: c_i is row (g + 8*(i>=2)), col (2t + (i&1)) of the 16x8 subtile.
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    // FUSED weights hold several source tensors stacked along M, each with its own NVFP4 tensor
    // scale. Every segment boundary is 16-aligned, so a tile lies wholly inside one segment and the
    // scale is a per-TILE lookup — read once per block, no requantization, no precision loss.
    if (n < N && m < M) {
        const float o = v * (rs ? rs[m] : gs[mt]);
        if (Cf) Cf[(long long)n * M + m] = o;          // FP32-preserving partial (no round here)
        else    C[(long long)n * M + m] = f2b(o);
    }
}

// mma_epilogue with NO output-side scale: the DSV4 block-scale kernel promotes sa·sb into the
// accumulator inside the k-loop, so the fixed-order cross-warp reduce is written out as-is.
__device__ __forceinline__ void mma_epilogue_prescaled(float* sh, float acc[2][4], __nv_bfloat16* C,
                                                       int mt, int M, int N, float* Cf = nullptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order

    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N && m < M) {
        if (Cf) Cf[(long long)n * M + m] = v;
        else    C[(long long)n * M + m] = f2b(v);
    }
}

// One warp's pass over k-block PAIRS p in [p_lo, p_hi) stepping MMA_NW, accumulating into acc —
// the exact body of gemm_mma_fp4_b's k-loop, shared verbatim with the split-K partial kernel
// (gemm_mma_fp4_splitk_b) so the two can never drift. xr0/xr1 are this lane's two X rows (already
// N-clamped by the caller); the rest of the geometry is the caller's.
//
// A warp takes an ADJACENT PAIR of k-blocks per iteration. The pairing is not for unrolling —
// it is to fix a DRAM sector waste that ncu found and that nothing else could see.
//
// The scale array holds 16 E4M3 bytes per 16x16 tile. Reading them the obvious way,
//
//     const uint8_t* sct = Sct + tile*16;   s_lo = sct[g];   s_hi = sct[g+8];   // g = lane>>2
//
// makes 32 lanes ask for only 16 DISTINCT BYTES -- half of a 32-byte DRAM sector. The weight load
// sitting next to it is perfect (128 contiguous bytes, 4 full sectors), but every scale fetch threw
// away half its sector. Measured: **18.7 of 32 bytes utilized** across the 64.6% of sectors that
// miss L2, on a kernel that ncu confirms is latency-starved (14 active warps per scheduler but only
// 0.82 ELIGIBLE -- they are all parked on `long_scoreboard`, waiting for memory).
//
// Consecutive tiles' scales are contiguous, so a warp that takes tiles 2p and 2p+1 reads
// Sct[..2p*16 .. 2p*16+32) -- 32 contiguous bytes, ONE FULL SECTOR, 32/32 utilized.
//
// NB: an earlier attempt at this failed and taught the lesson. It paired kb with kb+MMA_NW (a
// STRIDED pair), which gives the same instruction-level parallelism but leaves the two scale reads
// on DIFFERENT sectors -- still 50% each. It measured no faster. Adjacency is the entire point;
// "unroll by 2" is not.
//
// The k-visit order changes (2w, 2w+1, 2w+16, ... instead of w, w+8, ...) but it is still FIXED and
// N-INDEPENDENT, so column 0 is bit-identical at every N and batch-invariance is untouched. Gate:
// --probe-binv. K % 32 == 0 for every tensor in this family (asserted host-side).
__device__ __forceinline__ void mma_fp4_pairs(float acc[2][4], int p_lo, int p_hi,
                                              const uint32_t* Wt32, const uint8_t* Sct,
                                              const __nv_bfloat16* X, long long xr0, long long xr1,
                                              int mt, int nblk, int g, int t) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // S9F L1 (the ~15 ms verify lever): cp.async double-buffering of the weight pair + the scale
    // sector, so the k-loop overlaps the (L2/DRAM) load latency with the mma stream — ncu showed
    // the M=8 kernel latency-starved (0.82 eligible warps parked on `long_scoreboard`). Each lane
    // stages its OWN 3 u32s (wq0, wq1, and one 4-B scale word) into per-lane smem slots; the scale
    // bytes come back via __shfl (the lane holding the word extracts its byte). The k-visit order
    // and EVERY arithmetic op are unchanged — only the memory path differs (async global->smem
    // instead of a synchronous global read), so column 0 stays bit-identical at every N and
    // batch-invariance is untouched. Gate: --probe-binv. The adjacent-pair sector trick (2p, 2p+1)
    // and the fixed k-visit order are preserved verbatim.
    //
    // Layout: wstage[2][MMA_NW*32*3] — [buffer][warp*32+lane][wq0 | wq1 | scale-word]. Each lane's
    // scale word is the 4-B ALIGNED fetch Sct[tile*16 + (lane&~3)] (cp.async needs naturally
    // aligned sources); the warp's 8 distinct words cover the whole 32-B scale sector (one
    // DRAM sector, 32/32 utilized — the scale sector is never split, exactly like the direct-load
    // pairing). Byte g of the sector lives in lane (g>>2)*4's word at byte (g&3). A/B measured:
    // the full staging (weights + scales) beat both the reference and a weights-only variant
    // (verify 96.2 vs 98.4/98.9 ms) — the scale path replaced 4 strided global byte-loads per
    // lane with 1 cp.async + 4 shuffles, fewer memory instructions on the dequant chain.
    __shared__ uint32_t wstage[2][MMA_NW * 32 * 3];
    const int sid = (warp * 32 + lane) * 3;

    auto stage = [&](int p, int buf) {
        const long long tile = (long long)mt * nblk + (p << 1);
        const uint32_t* wsrc = Wt32 + tile * 32;
        const uint8_t* ssrc = Sct + tile * 16;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 0])), "l"(wsrc + lane));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 1])), "l"(wsrc + 32 + lane));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&wstage[buf][sid + 2])),
                        "l"(ssrc + (lane & ~3)));
        asm volatile("cp.async.commit_group;");
    };

    // Preload the first pair, then pipeline: issue pair p+MMA_NW while computing pair p. wait_group
    // 1 after each issue leaves exactly the just-issued prefetch outstanding — the current pair's
    // group (committed earlier) has completed. The per-thread staging needs NO barrier: every lane
    // reads only its own slots (cp.async completion is per-thread).
    stage(p_lo, 0);
    int i = 0;
    for (int p = p_lo; p < p_hi; p += MMA_NW, i++) {
        const int cur = i & 1;
        const bool more = (p + MMA_NW) < p_hi;
        if (more) { stage(p + MMA_NW, cur ^ 1); }
        if (more) { asm volatile("cp.async.wait_group 1;"); }
        else      { asm volatile("cp.async.wait_group 0;"); }

        const uint32_t wq0 = wstage[cur][sid + 0];              // this lane's 4 B of tile 2p's weight row
        const uint32_t wq1 = wstage[cur][sid + 1];              // tile 2p+1's
        const uint32_t sw  = wstage[cur][sid + 2];              // this lane's 4-B scale-sector word
        // The scale bytes via shuffles (same bytes the direct-load path read — bit-identical).
        // Byte b of the sector sits in lane (b>>2)*4's word at byte (b&3).
        const uint32_t wb = __shfl_sync(0xffffffffu, sw, (g >> 2) * 4);
        const uint32_t wb8 = __shfl_sync(0xffffffffu, sw, ((g + 8) >> 2) * 4);
        const uint32_t wb16 = __shfl_sync(0xffffffffu, sw, ((g + 16) >> 2) * 4);
        const uint32_t wb24 = __shfl_sync(0xffffffffu, sw, ((g + 24) >> 2) * 4);
        const float s0lo = e4m3_f((uint8_t)((wb   >> ((g & 3) * 8)) & 0xFF));    // sct[g]
        const float s0hi = e4m3_f((uint8_t)((wb8  >> (((g + 8) & 3) * 8)) & 0xFF));   // sct[g+8]
        const float s1lo = e4m3_f((uint8_t)((wb16 >> (((g + 16) & 3) * 8)) & 0xFF));  // sct[g+16]
        const float s1hi = e4m3_f((uint8_t)((wb24 >> (((g + 24) & 3) * 8)) & 0xFF));  // sct[g+24]

        const int k0 = (p << 5);                                  // 2 k-blocks = 32 elements of K
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);

        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);        // block 2p,   columns 0..7
        mma_m16n8k16(acc[1], ra, rb1);        // block 2p,   columns 8..15

        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        uint32_t rb3[2] = { Xh[t + 8], Xh[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);        // block 2p+1, columns 0..7
        mma_m16n8k16(acc[1], ra, rb3);        // block 2p+1, columns 8..15
    }
}

// ---- NVFP4. One mma k-step consumes exactly one 16-element scale block, so the block scale is
// constant over the step and folds into the A-fragment for free. That alignment is the whole trick.
// Cf != nullptr => write the FP32 accumulator to Cf and leave C untouched (TP=2 FP32-preserving path).
//
// E9: the LAST argument is the PDL (programmatic dependent launch) flag. pdl==1 means this grid was
// launched as the programmatic-stream-serialization SECONDARY of a TP barrier chain: it publishes
// its own launch-completion edge, streams this block's weight bytes into L2 (read-only — no
// numerics change), and only then griddepcontrol.wait gates the (barrier-reduced) activation. pdl==0
// (the plain path, and every non-E9 caller) skips all three — byte-identical to the pre-E9 kernel.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp4_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, int M, int K, int N,
    float* Cf, int pdl)
{
    const int ntm = M >> 4, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    if (pdl) {
        // E9 prologue: publish, prefetch this block's tiles' weights, then wait. The prefetch
        // issues exactly the weight/scale loads the k-loop below will perform (same tile range,
        // same lanes), so the compute hits L2; the loads are kept alive by the sink checksum and
        // the never-true store (a store under a false condition cannot affect C).
        asm volatile("griddepcontrol.launch_dependents;");
        const uint32_t* Wt32p = reinterpret_cast<const uint32_t*>(Wt);
        const int npairp = nblk >> 1;
        unsigned sink = 0u;
        for (int mtp = blockIdx.x; mtp < ntm; mtp += gridDim.x) {
            for (int p = warp; p < npairp; p += MMA_NW) {
                const long long tile = (long long)mtp * nblk + (p << 1);
                sink ^= Wt32p[tile * 32 + lane];
                sink ^= Wt32p[tile * 32 + 32 + lane];
                sink ^= (unsigned)Sct[tile * 16 + g] << 8;
                sink ^= (unsigned)Sct[tile * 16 + g + 8] << 16;
            }
        }
        if (sink == 0xDEADBEEFu) C[threadIdx.x] = f2b(0.f);   // never true — keeps the loads alive
        asm volatile("griddepcontrol.wait;");
    }

    // The two X rows this lane's B-fragments read. Columns >= N are padding: clamp them onto a valid
    // row so the load stays in bounds. They feed independent accumulators that are never written, so
    // the garbage cannot reach column 0 — see the invariance argument above.
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;

    __shared__ float sh[MMA_SMEM];
    // Persistent output tiles: the grid is capped at the resident-block capacity (48 SMs x the 6
    // blocks/SM __launch_bounds__ asserts) and block b computes tiles b, b+gridDim.x, ... . Every
    // tile's arithmetic is self-contained — the k-loop visits k-blocks in the same fixed order and
    // mma_epilogue reduces the warps in the same fixed order — so WHICH block runs a tile, and in
    // what order tiles are consumed, cannot change any tile's result. The tile->block map is a
    // function of the weight shape (ntm, gridDim.x) only, never of N: column 0 stays bit-identical
    // at every N and batch-invariance is untouched. Gate: --probe-binv.
    for (int mt = blockIdx.x; mt < ntm; mt += gridDim.x) {
        // `sh` is reused across tiles: mma_epilogue syncs write->read but NOT read->next-write, so
        // barrier here or a fast warp could overwrite sh while a slow one still reads the last tile.
        __syncthreads();
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        mma_fp4_pairs(acc, warp, npair, Wt32, Sct, X, xr0, xr1, mt, nblk, g, t);

        mma_epilogue(sh, acc, C, nullptr, gs, mt, M, N, Cf);
    }
}

// ---- SPLIT-K VARIANT for the shapes the persistent grid cannot fill: long-K, small-M (the FFN
// down-proj, M=5120 K=17408). There ntm=320 tiles is 1.11 waves of the 288 resident blocks AND each
// warp sits on a ~136-block serial k-chain, so the machine is both underfilled and latency-starved.
// Split the npair k-block-pair range into `nsplit` CONTIGUOUS chunks: work item (mt, s) accumulates
// chunk s of tile mt exactly as gemm_mma_fp4_b would, warp-reduces in the same fixed order, and
// writes the UNSCALED fp32 partial to P[(mt*nsplit+s)*256 + tid]. gemm_mma_splitk_reduce_b then
// sums the nsplit partials in FIXED s order and applies the epilogue.
//
// BATCH-INVARIANCE: nsplit and the chunk boundaries are functions of the weight shape (K, and M via
// the host-side tile cap) ONLY — never of N — and every reduction is fixed-order (no atomics, ever;
// see the Marlin note above). A decode (N=1) and a verify (N<=16) see identical split geometry and
// identical per-column arithmetic, so column 0 is bit-identical at every N. Gate: --probe-binv;
// force the path on small models with GB10_GEMM_SPLITK=<S>.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp4_splitk_b(
    float* P, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const __nv_bfloat16* __restrict__ X, int M, int K, int N, int nsplit)
{
    const int ntm = M >> 4, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    const int chunk = (npair + nsplit - 1) / nsplit;   // pair-range per split; K-only

    __shared__ float sh[MMA_SMEM];
    for (int work = blockIdx.x; work < ntm * nsplit; work += gridDim.x) {
        // `sh` is reused across work items — barrier here, same reason as the unsplit tile loop.
        __syncthreads();
        const int mt = work / nsplit, s = work - mt * nsplit;
        const int p0 = s * chunk;
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        mma_fp4_pairs(acc, p0 + warp, min(p0 + chunk, npair), Wt32, Sct, X, xr0, xr1, mt, nblk, g, t);

        P[(long long)work * 256 + threadIdx.x] = mma_warp_reduce(sh, acc);   // UNSCALED fp32 partial
    }
}

// The second pass: one block per output tile, 256 threads = the 256 (m, n) fragment slots. Sum the
// nsplit partials in FIXED s order, then the usual epilogue — per-tile gs scale, bf16 store, or the
// unrounded fp32 store to Cf for the FP32-preserving TP=2 path (exactly mma_epilogue's semantics).
extern "C" __global__ void gemm_mma_splitk_reduce_b(
    __nv_bfloat16* C, const float* __restrict__ P, const float* __restrict__ gs,
    int M, int N, int nsplit, float* Cf)
{
    const int mt = blockIdx.x;
    const long long base = (long long)mt * nsplit * 256 + threadIdx.x;
    float v = 0.0f;
    for (int s = 0; s < nsplit; s++) v += P[base + (long long)s * 256];   // FIXED order s=0..S-1

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N && m < M) {
        const float o = v * gs[mt];
        if (Cf) Cf[(long long)n * M + m] = o;          // FP32-preserving partial (no round here)
        else    C[(long long)n * M + m] = f2b(o);
    }
}

// ---- FP8 E4M3. Scales are per output ROW, constant over K, so nothing folds into the fragment and
// the row scale is applied once to the f32 accumulator in the epilogue.
extern "C" __global__ __launch_bounds__(256) void gemm_mma_fp8_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const float* __restrict__ RowScale,
    const __nv_bfloat16* __restrict__ X, int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nblk; kb += MMA_NW) {
        const long long tile = (long long)mt * nblk + kb;
        const uint2 w8 = Wt64[tile * 32 + lane];            // ONE 8-byte load = the whole A-fragment

        uint32_t ra[4];
        ra[0] = fp8_pair_bf16(w8.x,       w8.x >>  8);      // row g,   cols 2t, 2t+1
        ra[1] = fp8_pair_bf16(w8.x >> 16, w8.x >> 24);      // row g+8, cols 2t, 2t+1
        ra[2] = fp8_pair_bf16(w8.y,       w8.y >>  8);      // row g,   cols 2t+8, 2t+9
        ra[3] = fp8_pair_bf16(w8.y >> 16, w8.y >> 24);      // row g+8, cols 2t+8, 2t+9

        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + (kb << 4));
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + (kb << 4));
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };

        mma_m16n8k16(acc[0], ra, rb0);
        mma_m16n8k16(acc[1], ra, rb1);
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, C, RowScale, nullptr, mt, M, N, Cf);
}

// ================== DSV4 FP8 BLOCK-SCALE EPILOGUE GEMM (§12.A.2, §C.3) ==================
//
// C[n,m] (bf16, or f32 into Cf for the TP path) =
//     Σ over K-128 blocks kb of ( raw fp32 block GEMM over codes ) · sa[n,kb] · sb[m/128,kb]
//
// DeepSeek-V4's FP8 weights are NOT our per-row-scaled FP8: the scales are per-128×128-block
// UE8M0 (sb), and the activations arrive pre-quantized as e4m3 codes [N,K] + per-128 UE8M0
// scales sa [N,K/128] (activation quant is a different kernel's job). The weight bytes are the
// SAME MMA-repacked 16x16 tiles as gemm_mma_fp8_b (quant.rs::repack_fp8_mma, 256 B/tile).
//
// EXACTNESS (why the only CPU-reference divergence is f32 addition ORDER):
//   * e4m3 -> bf16 is EXACT for both operands (4 significand bits <= bf16's 8), so the mma's
//     raw block GEMM is the reference's raw code GEMM bit-for-bit (fixed hw add order).
//   * UE8M0 scales are decoded IN-KERNEL as bare powers of two (bit shift, exact — no host
//     pre-decode pass, 4x less scale traffic, and the sa·sb product stays exact). b=0 is
//     2^-127, an f32 SUBNORMAL — the naive `b<<23` would give 0.0; the one-instruction fixup
//     keeps the decode equal to dsv4_load::e8m0_to_f32 on every byte.
//   * Promotion multiplies the raw block partial by sa·sb (pow2 x pow2 = exact in f32), so it
//     introduces NO rounding of its own; pow2 scaling commutes exactly with the f32 sums.
//
// THE WARP-OWNERSHIP RESTRUCTURE (the reason this is not just gemm_mma_fp8_b + a scale):
// the per-128-K promotion must happen on a whole 128-K block's raw partial, so a warp must
// OWN whole 128-K blocks. gemm_mma_fp8_b strides single 16-K blocks across the 8 warps and
// gemm_mma_fp4_b strides PAIRS — under either, one 128-K block (8 blocks = 4 pairs) is split
// across 4 warps and no warp can promote. Here warp w takes 128-blocks w, w+8, w+16, ... and
// walks the block's 4 ADJACENT k-block pairs in order. The pair step keeps fp4_b's DRAM-sector
// argument (:3128-3147): two adjacent 256 B tiles = 512 contiguous bytes per warp per step,
// every load a full sector run. The visit order (block-stride outside, ascending pairs inside)
// is FIXED and N-independent, each warp promotes its own partial in ascending owned-block
// order, and the cross-warp reduce is the existing fixed-order smem one — column 0 is
// bit-identical at every N (N padded to 16 by the usual row-clamps; no atomics anywhere).
//
// Host contract: M % 128 == 0 && K % 128 == 0 && 1 <= N <= 16 (asserted host-side).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns — the
    // padding feeds independent accumulators that are never written (invariance: :3120-3124).
    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
            const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
            const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
            ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
            ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
            ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
            uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
            mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

            ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
            ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
            ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
            ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
            uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
            mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
        }

        // Promote the warp-local raw partial: sb is per (128-row, 128-K) block — constant over
        // this whole tile — and sa is per token column. All pow2, so every product is exact.
        const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
        const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
        const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
        acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
        acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
        acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
        acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb2 — TWO adjacent 16-row tiles per CTA (R3A.1 E1b).
// Identical per-element contract to gemm_dsv4_fp8_bsb (above): same K-partition (warp w owns
// 128-K blocks w, w+8, ...), same per-block 8 ascending-k mma into a zero-inited raw, same
// per-block promotion by sa*sb (pow2, exact), same fixed ascending-warp epilogue reduce, same
// clamp construction. The two tiles' chains are INDEPENDENT accumulators interleaved in the
// instruction stream (contract-sanctioned ILP); the X fragments are loaded once and shared
// (same values, same columns) — activation reads halve and each warp has two independent
// weight streams in flight, which is the whole point: the single-tile kernel's per-warp
// stream is too short to cover LPDDR latency at the decode shapes (143-165 GB/s ncu).
// Grid: one block per PAIR of 16-row tiles, (M+31)/32 CTAs; M % 128 == 0 in production so the
// has1 guard never fires (kept for generality — a lone-tile CTA's chain is unchanged).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt0 = 2 * blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < M;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
            const uint2 w00 = Wt64[tile0 * 32 + lane];
            const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
            const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
            ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
            ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
            ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
            mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
            mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
            ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
            ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
            ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
            ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
            mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
            mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

            if (has1) {
                const long long tile1 = (long long)mt1 * nblk + (kb << 3) + 2 * q;
                const uint2 w10 = Wt64[tile1 * 32 + lane];
                const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                mma_m16n8k16(raw1[1], ra, rb1);
                ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                mma_m16n8k16(raw1[1], ra, rb3);
            }
        }

        // Promote each tile's raw partial by its own (sa*sb) — the same two-operand products
        // in the same ascending owned-block order as the single-tile kernel (all pow2, exact).
        const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
        const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
        {
            const float sb = e8m0_f(Sb[(long long)(mt0 >> 3) * nkb + kb]);
            const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
            acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
            acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
            acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
            acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
        }
        if (has1) {
            const float sb = e8m0_f(Sb[(long long)(mt1 >> 3) * nkb + kb]);
            const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
            acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
            acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
            acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
            acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
        }
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc0, C, mt0, M, N, Cf);
    __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
    if (has1) mma_epilogue_prescaled(sh, acc1, C, mt1, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb_tma — Tier-1 item 1.3 order-faithfulness PROBE: DeepGEMM-class
// (1d1d) memory structure — one cp.async.bulk (TMA) per 16 KB weight sweep into an
// FP8_TMA_NS-stage smem pipeline with mbarrier completion — hosting our LOCKED reduction
// order. PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb: same warp ownership of
// whole 128-K blocks (warp w owns kb = w, w+8, ...), same per-block 4 ascending adjacent
// k-block pairs of mma into a zeroed raw, same per-block sa*sb promotion in ascending
// owned-block order, same clamp construction, same fixed-order cross-warp epilogue. The
// ONLY change is the weight data path: the tile's repacked weight region is consumed in
// 16 KB sweeps (8 128-K blocks = kb 8j..8j+7, CONTIGUOUS in the repacked layout), staged
// by bulk TMA; the fragment loads then read the same bytes from smem that bsb reads from
// global. Values, visit order, promotion order: untouched. Bitwise gate = bsb (the probe
// verdict in DSV4_SESSION_OPT_FINAL_LIST.md 1.3 decides port vs capped).
//
// Host contract: same as bsb PLUS K % 1024 == 0 (whole sweeps), dynamic smem =
// FP8_TMA_NS * (16384 + 8) bytes, grid = M/16. NS sweep (RUN 15): NS=5 → 1 CTA/SM
// (GB10 smem opt-in cap ~99 KB incl. the 8 KB static epilogue buffer) → 149–164 GB/s;
// NS=2 → 2 CTAs/SM → 191–230 GB/s ≈ parity with bsb2, never above — hence CAPPED at
// probe status, NOT wired into production (verdict: DSV4_SESSION_OPT_FINAL_LIST.md 1.3).
#define FP8_TMA_NS 2
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_tma(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int N, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nkb = K >> 7;
    const int sweeps = nkb >> 3;             // 8 128-K blocks (16 KB) per sweep

    extern __shared__ __align__(128) uint8_t tsmem[];   // NS x 16 KB stages, then NS barriers
    uint64_t* bars = reinterpret_cast<uint64_t*>(tsmem + FP8_TMA_NS * 16384);

    if (threadIdx.x < FP8_TMA_NS) {
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[threadIdx.x]);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(baddr));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    // TMA issue for sweep j into stage j%NS (one thread; the barrier's single arrival is
    // the issuing thread's arrive.expect_tx; completion needs the 16384 tx bytes).
    const long long region = (long long)mt * nkb * 2048;    // this tile's weight region base
    auto issue = [&](int j) {
        const int s = j % FP8_TMA_NS;
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[s]);
        const uint32_t saddr = (uint32_t)__cvta_generic_to_shared(tsmem + s * 16384);
        const uint8_t* src = Wt + region + (long long)j * 16384;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(baddr), "r"(16384));
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "r"(saddr), "l"(src), "r"(16384), "r"(baddr) : "memory");
    };
    if (threadIdx.x == 0) {
        const int pre = sweeps < FP8_TMA_NS ? sweeps : FP8_TMA_NS;
        for (int j = 0; j < pre; j++) issue(j);
    }

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns —
    // verbatim from bsb (activation path is unchanged).
    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    for (int j = 0; j < sweeps; j++) {
        const int s = j % FP8_TMA_NS;
        const uint32_t baddr = (uint32_t)__cvta_generic_to_shared(&bars[s]);
        const int phase = (j / FP8_TMA_NS) & 1;
        uint32_t done = 0;
        while (!done) {
            asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                         " selp.u32 %0, 1, 0, p;}"
                         : "=r"(done) : "r"(baddr), "r"(phase));
        }

        // warp's 128-K block this sweep is kb = warp + 8j — the warp's 2 KB run inside the
        // staged 16 KB (offset warp*2048), then the verbatim bsb fragment walk.
        const int kb = warp + 8 * j;
        const uint2* Wt64 = reinterpret_cast<const uint2*>(tsmem + s * 16384 + warp * 2048);
        float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const uint2 w0 = Wt64[q * 64 + lane];           // tile 2q   (256 B, 8 B/lane)
            const uint2 w1 = Wt64[q * 64 + 32 + lane];      // tile 2q+1, back-to-back

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
            ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
            ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
            ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
            uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
            mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

            ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
            ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
            ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
            ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
            uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
            mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
        }

        // Verbatim bsb promotion: sb per (128-row, 128-K) block, sa per token column, pow2.
        const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
        const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
        const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
        acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
        acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
        acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
        acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;

        __syncthreads();        // stage fully consumed before it is reissued
        if (threadIdx.x == 0 && j + FP8_TMA_NS < sweeps) issue(j + FP8_TMA_NS);
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb2q — TWO-OP pair (R3A.1 E2 first rung): ONE launch computes two
// INDEPENDENT projections that share the activation (production: wq_a + wkv per layer, both
// read the same x codes/scales). CTAs [0, packs0) compute op0's tiles (2 per CTA), CTAs
// [packs0, packs0+packs1) compute op1's. Per-element contract per tile is IDENTICAL to
// gemm_dsv4_fp8_bsb2 on that op's weight (same K-partition, same per-block 8 ascending-k mma
// into zero-inited raw, same per-block sa*sb promotion, same fixed ascending-warp epilogue
// reduce) — the op select only routes tile indices; the gate compares against the two
// separate bsb2 launches bitwise. M0 % 128 == 0 && M1 % 128 == 0 in production (guard kept).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb2q(
    __nv_bfloat16* C0, const uint8_t* __restrict__ Wt0, const uint8_t* __restrict__ Sb0,
    __nv_bfloat16* C1, const uint8_t* __restrict__ Wt1, const uint8_t* __restrict__ Sb1,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    long long m01, long long kn, float* Cf)
{
    const int M0 = (int)(m01 & 0xffffffff), M1 = (int)(m01 >> 32);
    const int K = (int)(kn & 0xffffffff), N = (int)(kn >> 32);
    const int packs0 = (M0 + 31) / 32;
    const int op = (blockIdx.x >= packs0) ? 1 : 0;
    const int b = op ? (int)blockIdx.x - packs0 : (int)blockIdx.x;
    const int mt0 = 2 * b;
    const int M = op ? M1 : M0;
    const bool has1 = (mt0 + 1) * 16 < M;
    __nv_bfloat16* C = op ? C1 : C0;
    const uint8_t* Wt = op ? Wt1 : Wt0;
    const uint8_t* Sb = op ? Sb1 : Sb0;
    const int mt1 = mt0 + (has1 ? 1 : 0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
            const uint2 w00 = Wt64[tile0 * 32 + lane];
            const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
            const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
            ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
            ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
            ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
            mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
            mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
            ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
            ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
            ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
            ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
            mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
            mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

            if (has1) {
                const long long tile1 = (long long)mt1 * nblk + (kb << 3) + 2 * q;
                const uint2 w10 = Wt64[tile1 * 32 + lane];
                const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                mma_m16n8k16(raw1[1], ra, rb1);
                ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                mma_m16n8k16(raw1[1], ra, rb3);
            }
        }

        // Promote each tile's raw partial by its own (sa*sb) — the same two-operand products
        // in the same ascending owned-block order as the single-tile kernel (all pow2, exact).
        const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
        const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
        {
            const float sb = e8m0_f(Sb[(long long)(mt0 >> 3) * nkb + kb]);
            const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
            acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
            acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
            acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
            acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
        }
        if (has1) {
            const float sb = e8m0_f(Sb[(long long)(mt1 >> 3) * nkb + kb]);
            const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
            acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
            acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
            acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
            acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
        }
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc0, C, mt0, M, N, Cf);
    __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
    if (has1) mma_epilogue_prescaled(sh, acc1, C, mt1, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb1q — TWO-OP pair, ONE 16-row tile per CTA (Tier-1 item 1.4, RUN 16).
// Same two-independent-ops-sharing-the-activation fusion as gemm_dsv4_fp8_bsb2q (production:
// wq_a + wkv per layer), but with the single-tile bsb body per CTA instead of two tiles.
// Motivation (measured, RUN 16): at the decode pair shapes (M0=1024, M1=512, K=4096) the
// two-tile fused launch fields only 48 CTAs (~1 CTA/SM, 8/64 warps — latency-bound ramp);
// single-tile bsb BEATS bsb2 at these shapes in isolation (wq_a 26.7 vs 36.6 µs; wkv 14.4 vs
// 25.4 µs — the opposite of the big shapes where bsb2 won). This kernel keeps the one-launch
// fusion (one ramp for both ops) at 96 CTAs. PER-ELEMENT CONTRACT IS IDENTICAL TO
// gemm_dsv4_fp8_bsb on each op's weight (same K-partition, same per-block 8 ascending-k mma
// into zero-inited raw, same per-block sa*sb promotion, same fixed ascending-warp epilogue
// reduce, same clamp construction) — the op select only routes tile indices; the gate
// compares against the separate bsb launches bitwise. Host contract: M0,M1 % 128 == 0,
// K % 128 == 0, 1 <= N <= 16; grid = M0/16 + M1/16 CTAs.
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb1q(
    __nv_bfloat16* C0, const uint8_t* __restrict__ Wt0, const uint8_t* __restrict__ Sb0,
    __nv_bfloat16* C1, const uint8_t* __restrict__ Wt1, const uint8_t* __restrict__ Sb1,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    long long m01, long long kn, float* Cf)
{
    const int M0 = (int)(m01 & 0xffffffff);
    const int tiles0 = M0 >> 4;
    const int op = (blockIdx.x >= tiles0) ? 1 : 0;
    const int mt = op ? (int)blockIdx.x - tiles0 : (int)blockIdx.x;
    const int M = op ? (int)(m01 >> 32) : M0;
    __nv_bfloat16* C = op ? C1 : C0;
    const uint8_t* Wt = op ? Wt1 : Wt0;
    const uint8_t* Sb = op ? Sb1 : Sb0;
    const int K = (int)(kn & 0xffffffff), N = (int)(kn >> 32);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;

    // X rows / sa rows this lane reads, clamped onto valid rows for padding columns —
    // verbatim from bsb (activation path is unchanged).
    const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
    const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    const uint8_t* sa0 = Sa + n0 * nkb; const uint8_t* sa1 = Sa + n1 * nkb;
    const uint8_t* sa2 = Sa + n2 * nkb; const uint8_t* sa3 = Sa + n3 * nkb;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);

    for (int kb = warp; kb < nkb; kb += MMA_NW) {
        float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
            const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
            const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
            const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back

            const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
            const uint16_t* Xl = reinterpret_cast<const uint16_t*>(X + xr0 + k0);
            const uint16_t* Xh = reinterpret_cast<const uint16_t*>(X + xr1 + k0);
            const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
            const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

            uint32_t ra[4];
            ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
            ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
            ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
            ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
            uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
            uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
            mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

            ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
            ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
            ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
            ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
            uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
            uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
            mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
            mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
        }

        // Verbatim bsb promotion: sb per (128-row, 128-K) block, sa per token column, pow2.
        const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
        const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
        const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
        acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
        acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
        acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
        acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;
    }

    __shared__ float sh[MMA_SMEM];
    mma_epilogue_prescaled(sh, acc, C, mt, M, N, Cf);
}

// ---- gemm_dsv4_fp8_bsb_pf — PREFILL weight-stationary variant (R3A.4 P1).
// The serving path used to chunk prefill into <=16-row LAUNCHES (the G2 N<=16 regime), so at
// s=2048 every projection's full weight matrix was re-read from DRAM 128x (nsys: 2.5 s of a
// 21 s prefill). Here the token-chunk loop moves INSIDE the kernel: one launch per
// projection, each CTA's 16-row weight tile is read from DRAM once and stays L2-hot across
// the chunk loop. PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb: for every output
// (row m, token n) the chain is the same — warp w owns 128-K blocks w, w+8, ... ascending;
// per block the same 8 ascending-k mma into a zero-inited raw; promote by sa*sb per block in
// ascending owned-kb order; the same fixed ascending-warp epilogue reduce; the same clamps
// for the partial last chunk. The chunk loop only sequences INDEPENDENT accumulator sets
// (one per chunk) — the gate compares this kernel against the <=16-row decomposition bitwise.
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    // grid (M/16, G): blockIdx.y selects a GROUP of up to PF_G chunks — the chunk loop is
    // split across G CTA groups so chunks run in parallel again (a single CTA serially
    // walking all chunks serialized the latency chain; each weight tile is re-read at most
    // G times, and stays L2-hot inside each group). Per-element chains are untouched: every
    // (row, token) still gets the same ascending-K chain as the <=16-row decomposition.
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + (long long)nc * 16 * K;
        const uint8_t* Sac = Sa + (long long)nc * 16 * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
        const uint8_t* sa0 = Sac + n0 * nkb; const uint8_t* sa1 = Sac + n1 * nkb;
        const uint8_t* sa2 = Sac + n2 * nkb; const uint8_t* sa3 = Sac + n3 * nkb;

        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
                const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
                const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
                ra[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
                ra[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
                ra[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
                uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                mma_m16n8k16(raw[0], ra, rb0);      // block 2q,   token columns 0..7
                mma_m16n8k16(raw[1], ra, rb1);      // block 2q,   token columns 8..15

                ra[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
                ra[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
                ra[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
                ra[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);
                uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
                mma_m16n8k16(raw[0], ra, rb2);      // block 2q+1, token columns 0..7
                mma_m16n8k16(raw[1], ra, rb3);      // block 2q+1, token columns 8..15
            }

            // Promote the warp-local raw partial: sb is per (128-row, 128-K) block — constant over
            // this whole tile — and sa is per token column. All pow2, so every product is exact.
            const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
            const float s0 = e8m0_f(sa0[kb]) * sb, s1 = e8m0_f(sa1[kb]) * sb;
            const float s2 = e8m0_f(sa2[kb]) * sb, s3 = e8m0_f(sa3[kb]) * sb;
            acc[0][0] += raw[0][0] * s0;  acc[0][1] += raw[0][1] * s1;   // cols 2t, 2t+1
            acc[0][2] += raw[0][2] * s0;  acc[0][3] += raw[0][3] * s1;
            acc[1][0] += raw[1][0] * s2;  acc[1][1] += raw[1][1] * s3;   // cols 2t+8, 2t+9
            acc[1][2] += raw[1][2] * s2;  acc[1][3] += raw[1][3] * s3;
        }

        mma_epilogue_prescaled(sh, acc, C + (long long)nc * 16 * M, mt, M, N,
                               Cf ? Cf + (long long)nc * 16 * M : nullptr);
        __syncthreads();             // sh reuse barrier between chunks
    }
}

// ---- gemm_dsv4_fp8_bsb_pf4 — PREFILL width variant: FOUR 16-token sub-chunks per pass
// (Tier-2 item 2.2, session 6). pf loops 16-token chunks; at width every (kb, q) weight
// fragment is re-loaded from L2 once per 16-token chunk and each warp's mma stream has
// only 2 independent accumulator chains (raw[0]/raw[1]) — issue-starved: ~4–8% of fp8
// peak in the session-5 decomposition (18.0% of prefill busy). Here the CTA walks
// 64-token GROUPS: the weight fragments for a (kb, q) are loaded ONCE and feed four
// sub-chunks' mma — 4 independent raw chains per warp (the bsb2 ILP trick applied
// across token chunks), quartering the weight L2 request rate per output.
//
// PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb_pf (and hence to the <=16-row
// bsb decomposition): for every output (row m, token n) — warp w owns 128-K blocks
// w, w+8, ... ascending; per block the same 8 ascending-k mma into a zero-inited raw
// (within a q step the c-inner loop issues sub-chunk c's block-2q mma pair then its
// block-2q+1 pair — the same ascending-k sequence bsb issues, with the four
// sub-chunks' INDEPENDENT chains interleaved in the instruction stream, the
// contract-sanctioned ILP argument of bsb2); per-block promotion by sa*sb in ascending
// owned-kb order; the same fixed ascending-warp epilogue reduce per sub-chunk; the same
// clamps for partial sub-chunks. Sub-chunks entirely past S are skipped wholesale —
// their outputs do not exist and no live chain reads their bytes. Gate: bitwise vs the
// <=16-row bsb decomposition at S in {17, 63, 64, 65, 130, 2048} (tests/dsv4_fp8_bsb_test.rs).
//
// Grid: (M/16, G) — blockIdx.y walks 64-token groups (~2 groups = 8 chunks per CTA,
// the same grouping target as pf).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int ngroups = (S + 63) >> 6;
    const int per_g = (ngroups + gridDim.y - 1) / gridDim.y;
    const int ng0 = blockIdx.y * per_g, ng1 = min(ngroups, ng0 + per_g);

    for (int ng = ng0; ng < ng1; ng++) {
        const int base = ng << 6;
        const int nlive = min(4, (S - base + 15) >> 4);   // live 16-token sub-chunks
        // Per-sub-chunk clamps — verbatim pf. Dead sub-chunks clamp onto row 0/1; their
        // loads, mma, promotions and epilogues are all skipped by the c < nlive guards.
        int Nn[4]; long long xr0[4], xr1[4];
        const uint8_t *sa0[4], *sa1[4], *sa2[4], *sa3[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int N = max(1, min(16, S - base - 16 * c));
            const uint8_t* Sac = Sa + (long long)(base + 16 * c) * nkb;
            const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
            const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
            Nn[c] = N;
            xr0[c] = (long long)(g     < N ? g     : N - 1) * K;
            xr1[c] = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
            sa0[c] = Sac + n0 * nkb; sa1[c] = Sac + n1 * nkb;
            sa2[c] = Sac + n2 * nkb; sa3[c] = Sac + n3 * nkb;
        }

        float acc[4][2][4];
        #pragma unroll
        for (int c = 0; c < 4; c++)
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[c][i][j] = 0.f;

        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw[4][2][4];
            #pragma unroll
            for (int c = 0; c < 4; c++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) raw[c][i][j] = 0.f;

            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile = (long long)mt * nblk + (kb << 3) + 2 * q;
                const uint2 w0 = Wt64[tile * 32 + lane];        // 256 B tile, one 8 B lane run
                const uint2 w1 = Wt64[tile * 32 + 32 + lane];   // the next tile, back-to-back
                uint32_t ra0[4], ra1[4];
                ra0[0] = fp8_pair_bf16(w0.x,       w0.x >>  8);
                ra0[1] = fp8_pair_bf16(w0.x >> 16, w0.x >> 24);
                ra0[2] = fp8_pair_bf16(w0.y,       w0.y >>  8);
                ra0[3] = fp8_pair_bf16(w0.y >> 16, w0.y >> 24);
                ra1[0] = fp8_pair_bf16(w1.x,       w1.x >>  8);
                ra1[1] = fp8_pair_bf16(w1.x >> 16, w1.x >> 24);
                ra1[2] = fp8_pair_bf16(w1.y,       w1.y >>  8);
                ra1[3] = fp8_pair_bf16(w1.y >> 16, w1.y >> 24);

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    if (c >= nlive) continue;
                    const uint16_t* Xl = reinterpret_cast<const uint16_t*>(
                        X + (long long)(base + 16 * c) * K + xr0[c] + k0);
                    const uint16_t* Xh = reinterpret_cast<const uint16_t*>(
                        X + (long long)(base + 16 * c) * K + xr1[c] + k0);
                    const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                    const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                    const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                    const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                    const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                    const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };
                    mma_m16n8k16(raw[c][0], ra0, rb0);      // block 2q,   token columns 0..7
                    mma_m16n8k16(raw[c][1], ra0, rb1);      // block 2q,   token columns 8..15
                    mma_m16n8k16(raw[c][0], ra1, rb2);      // block 2q+1, token columns 0..7
                    mma_m16n8k16(raw[c][1], ra1, rb3);      // block 2q+1, token columns 8..15
                }
            }

            // Promote each sub-chunk's raw partial by its own (sa*sb) — the same two-operand
            // products in the same ascending owned-block order as pf (all pow2, exact).
            const float sb = e8m0_f(Sb[(long long)(mt >> 3) * nkb + kb]);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (c >= nlive) continue;
                const float s0 = e8m0_f(sa0[c][kb]) * sb, s1 = e8m0_f(sa1[c][kb]) * sb;
                const float s2 = e8m0_f(sa2[c][kb]) * sb, s3 = e8m0_f(sa3[c][kb]) * sb;
                acc[c][0][0] += raw[c][0][0] * s0;  acc[c][0][1] += raw[c][0][1] * s1;
                acc[c][0][2] += raw[c][0][2] * s0;  acc[c][0][3] += raw[c][0][3] * s1;
                acc[c][1][0] += raw[c][1][0] * s2;  acc[c][1][1] += raw[c][1][1] * s3;
                acc[c][1][2] += raw[c][1][2] * s2;  acc[c][1][3] += raw[c][1][3] * s3;
            }
        }

        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (c >= nlive) continue;
            const long long off = (long long)(base + 16 * c) * M;
            mma_epilogue_prescaled(sh, acc[c], C + off, mt, M, Nn[c],
                                   Cf ? Cf + off : nullptr);
            __syncthreads();             // sh reuse barrier between sub-chunks
        }
    }
}

// ---- gemm_dsv4_fp8_bsb_pf2 — PREFILL width variant: TWO 16-row weight tiles per CTA
// (Tier-2 item 2.2, session 6). pf walks 16-token chunks with ONE tile per CTA; at
// width the X fragment loads (8 LDG per (kb, q)) and their e4m3->bf16 conversions feed
// only 4 mma, and every warp's mma stream has only 2 independent accumulator chains.
// Here each CTA computes TWO adjacent 16-row tiles (bsb2's decode trick moved to the
// prefill chunk loop): the X fragments for a (chunk, kb, q) are loaded and converted
// ONCE and feed both tiles' mma — halving the activation LDG + conversion cost per
// mma and giving each warp 4 independent raw chains, at ~pf's register footprint.
//
// PER-ELEMENT CONTRACT IS IDENTICAL TO gemm_dsv4_fp8_bsb_pf (and hence to the <=16-row
// bsb decomposition): for every output (row m, token n) — warp w owns 128-K blocks
// w, w+8, ... ascending; per block the same 8 ascending-k mma into a zero-inited raw
// (the two tiles' INDEPENDENT chains interleave in the instruction stream — the
// contract-sanctioned ILP argument of bsb2); per-block promotion by sa*sb in ascending
// owned-kb order (each tile by its OWN sb — (mt0>>3) vs (mt1>>3) differ across a
// 128-row boundary, exactly as bsb2); the same fixed ascending-warp epilogue reduce
// per tile; the same clamps for the partial last chunk. M % 128 == 0 in production so
// the has1 guard never fires (kept for generality — a lone-tile CTA's chain is
// unchanged). Gate: bitwise vs the <=16-row bsb decomposition (tests/dsv4_fp8_bsb_test.rs).
//
// Grid: ((M+31)/32, G) — blockIdx.y walks 16-token chunk groups (~8 chunks per CTA,
// the same grouping target as pf).
extern "C" __global__ __launch_bounds__(256) void gemm_dsv4_fp8_bsb_pf2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int M, int K, int S, float* Cf)
{
    const int mt0 = 2 * blockIdx.x, mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < M;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + (long long)nc * 16 * K;
        const uint8_t* Sac = Sa + (long long)nc * 16 * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
        const uint8_t* sa0 = Sac + n0 * nkb; const uint8_t* sa1 = Sac + n1 * nkb;
        const uint8_t* sa2 = Sac + n2 * nkb; const uint8_t* sa3 = Sac + n3 * nkb;

        float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile0 = (long long)mt0 * nblk + (kb << 3) + 2 * q;
                const uint2 w00 = Wt64[tile0 * 32 + lane];
                const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
                ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
                ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
                ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
                ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
                ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
                ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
                ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

                if (has1) {
                    const long long tile1 = (long long)mt1 * nblk + (kb << 3) + 2 * q;
                    const uint2 w10 = Wt64[tile1 * 32 + lane];
                    const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                    ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                    ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                    ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                    ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                    mma_m16n8k16(raw1[1], ra, rb1);
                    ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                    ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                    ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                    ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                    mma_m16n8k16(raw1[1], ra, rb3);
                }
            }

            // Promote each tile's raw partial by its own (sa*sb) — the same two-operand
            // products in the same ascending owned-block order as pf (all pow2, exact).
            const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
            const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
            {
                const float sb = e8m0_f(Sb[(long long)(mt0 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
                acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
                acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
                acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
            }
            if (has1) {
                const float sb = e8m0_f(Sb[(long long)(mt1 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
                acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
                acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
                acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
            }
        }

        const long long off = (long long)nc * 16 * M;
        float* Cfc = Cf ? Cf + off : nullptr;
        mma_epilogue_prescaled(sh, acc0, C + off, mt0, M, N, Cfc);
        __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
        if (has1) mma_epilogue_prescaled(sh, acc1, C + off, mt1, M, N, Cfc);
        __syncthreads();                 // sh reuse barrier between chunks
    }
}

// ---- dsv4_olo_einsum_fp8_b — item 2.5: wo_a as an fp8 einsum-class per-group BMM.
// out[t, grp*R + c] = Σ_k o[t, grp*K + k] · wo_aq[grp*R + c, k] for t < S — the DeepGEMM
// fp8_einsum "bhr,hdr->bhd" class (n_groups = G local, group = 8 heads × 512, o_lora_rank R),
// with fp8 e4m3 + UE8M0 128-block scales on BOTH operands. The wo_a_q quantizer lays the
// groups as contiguous R-row bands (per-head-group tiles): the group offset enters the
// weight-tile index, the scale index, the X/Sa base pointers, and the output column base.
//
// TOLERANCE-CLASS by contract (item 2.5 / §6-a): reduction order is scheduler-chosen,
// K-split/atomics allowed, no N-invariance requirement — the ONLY bitwise obligations are
// the caller's (quantizers must be deterministic). The per-element instruction chain is the
// bsb-family one (fp8_pair_bf16 → mma m16n8k16 → per-128-block sa·sb promotion in ascending
// owned-kb order → fixed-order cross-warp reduce), so the OUTPUT is a bounded-rel-L2
// approximation of dsv4_olo_proj_tc_b/tc4_b (fp8 inputs vs bf16) — gated, not bitwise.
//
// Grid: ((R/32)*G, GY) — blockIdx.x = (group, 16-row tile pair), blockIdx.y walks 16-token
// chunk groups (pf2's ~8-chunks-per-CTA grouping + the 2-wave fill floor). R % 128 == 0 and
// K % 128 == 0 asserted host-side (production R=1024, K=4096). Two adjacent tiles per CTA
// share each chunk's X fragments (pf2's schedule: halved activation LDG + 4 independent
// accumulator chains per warp, ~80 regs, 3 CTA/SM).
// Epilogue for dsv4_olo_einsum_fp8_b: the bsb fixed-order reduce + fragment map, but the
// output tile's columns are offset by the group's R-row band (col = col_base + mt*16 + ...)
// inside a [S, G*R] row-major buffer (stride = G*R). Prescaled (the sa·sb promotion already
// happened in the k-loop). bf16 RNE round on store, same as mma_epilogue_prescaled.
__device__ __forceinline__ void mma_epilogue_olo(float* sh, float acc[2][4], __nv_bfloat16* C,
                                                 int mt, int col_base, int stride, int N) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 8; i++) sh[i * 256 + warp * 32 + lane] = acc[i >> 2][i & 3];
    __syncthreads();

    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    float v = 0.0f;
    #pragma unroll
    for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];   // FIXED order

    const int g = rlane >> 2, t = rlane & 3, sub = rslot >> 2, i = rslot & 3;
    const int m = col_base + mt * 16 + g + ((i >= 2) ? 8 : 0);
    const int n = sub * 8 + 2 * t + (i & 1);
    if (n < N) C[(long long)n * stride + m] = f2b(v);
}
extern "C" __global__ __launch_bounds__(256) void dsv4_olo_einsum_fp8_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sb,
    const uint8_t* __restrict__ X, const uint8_t* __restrict__ Sa,
    int R, int K, int S, int G, float* Cf)
{
    const int gx_pairs = R / 32;
    const int grp = blockIdx.x / gx_pairs;
    const int mt0 = 2 * (blockIdx.x % gx_pairs), mt1 = mt0 + 1;
    const bool has1 = mt1 * 16 < R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, nkb = K >> 7;
    const uint2* Wt64 = reinterpret_cast<const uint2*>(Wt);
    __shared__ float sh[MMA_SMEM];
    const int stride = G * R;
    const int nchunks = (S + 15) / 16;
    const int per_g = (nchunks + gridDim.y - 1) / gridDim.y;
    const int nc0 = blockIdx.y * per_g, nc1 = min(nchunks, nc0 + per_g);
    const long long wt_grp = (long long)grp * (R / 16) * nblk;
    const int sb_grp = grp * (R / 128) * nkb;

    for (int nc = nc0; nc < nc1; nc++) {
        const int N = min(16, S - nc * 16);              // this chunk's live rows (clamps below)
        const uint8_t* Xc = X + ((long long)nc * 16 * G + grp) * K;
        const uint8_t* Sac = Sa + ((long long)nc * 16 * G + grp) * nkb;
        // X rows / sa rows this lane reads, clamped onto valid rows for padding columns.
        // NB: the X token ROW stride is G*K (all groups' columns) and the per-token sa row
        // is G*nkb wide (one nkb-scale run per group) — stride both by the FULL row or
        // tokens > 0 read the wrong group's columns/scales (bitwise-broken at N>1).
        const int n0 = min(2 * t,     N - 1), n1 = min(2 * t + 1, N - 1);
        const int n2 = min(2 * t + 8, N - 1), n3 = min(2 * t + 9, N - 1);
        const long long xr0 = (long long)(g     < N ? g     : N - 1) * G * K;
        const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * G * K;
        const uint8_t* sa0 = Sac + n0 * (G * nkb); const uint8_t* sa1 = Sac + n1 * (G * nkb);
        const uint8_t* sa2 = Sac + n2 * (G * nkb); const uint8_t* sa3 = Sac + n3 * (G * nkb);

        float acc0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        float acc1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int kb = warp; kb < nkb; kb += MMA_NW) {
            float raw0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            float raw1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 4; q++) {                  // the block's 4 adjacent k-block pairs
                const long long tile0 = wt_grp + (long long)mt0 * nblk + (kb << 3) + 2 * q;
                const uint2 w00 = Wt64[tile0 * 32 + lane];
                const uint2 w01 = Wt64[tile0 * 32 + 32 + lane];

                const int k0 = (kb << 7) + (q << 5);            // 2 k-blocks = 32 codes of K
                const uint16_t* Xl = reinterpret_cast<const uint16_t*>(Xc + xr0 + k0);
                const uint16_t* Xh = reinterpret_cast<const uint16_t*>(Xc + xr1 + k0);
                const uint32_t x0 = Xl[t], x1 = Xl[t + 4], x2 = Xh[t], x3 = Xh[t + 4];
                const uint32_t y0 = Xl[t + 8], y1 = Xl[t + 12], y2 = Xh[t + 8], y3 = Xh[t + 12];
                const uint32_t rb0[2] = { fp8_pair_bf16(x0, x0 >> 8), fp8_pair_bf16(x1, x1 >> 8) };
                const uint32_t rb1[2] = { fp8_pair_bf16(x2, x2 >> 8), fp8_pair_bf16(x3, x3 >> 8) };
                const uint32_t rb2[2] = { fp8_pair_bf16(y0, y0 >> 8), fp8_pair_bf16(y1, y1 >> 8) };
                const uint32_t rb3[2] = { fp8_pair_bf16(y2, y2 >> 8), fp8_pair_bf16(y3, y3 >> 8) };

                uint32_t ra[4];
                ra[0] = fp8_pair_bf16(w00.x,       w00.x >>  8);
                ra[1] = fp8_pair_bf16(w00.x >> 16, w00.x >> 24);
                ra[2] = fp8_pair_bf16(w00.y,       w00.y >>  8);
                ra[3] = fp8_pair_bf16(w00.y >> 16, w00.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb0);      // tile0 block 2q,   token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb1);      // tile0 block 2q,   token columns 8..15
                ra[0] = fp8_pair_bf16(w01.x,       w01.x >>  8);
                ra[1] = fp8_pair_bf16(w01.x >> 16, w01.x >> 24);
                ra[2] = fp8_pair_bf16(w01.y,       w01.y >>  8);
                ra[3] = fp8_pair_bf16(w01.y >> 16, w01.y >> 24);
                mma_m16n8k16(raw0[0], ra, rb2);      // tile0 block 2q+1, token columns 0..7
                mma_m16n8k16(raw0[1], ra, rb3);      // tile0 block 2q+1, token columns 8..15

                if (has1) {
                    const long long tile1 = wt_grp + (long long)mt1 * nblk + (kb << 3) + 2 * q;
                    const uint2 w10 = Wt64[tile1 * 32 + lane];
                    const uint2 w11 = Wt64[tile1 * 32 + 32 + lane];
                    ra[0] = fp8_pair_bf16(w10.x,       w10.x >>  8);
                    ra[1] = fp8_pair_bf16(w10.x >> 16, w10.x >> 24);
                    ra[2] = fp8_pair_bf16(w10.y,       w10.y >>  8);
                    ra[3] = fp8_pair_bf16(w10.y >> 16, w10.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb0);  // tile1 block 2q — INDEPENDENT chain
                    mma_m16n8k16(raw1[1], ra, rb1);
                    ra[0] = fp8_pair_bf16(w11.x,       w11.x >>  8);
                    ra[1] = fp8_pair_bf16(w11.x >> 16, w11.x >> 24);
                    ra[2] = fp8_pair_bf16(w11.y,       w11.y >>  8);
                    ra[3] = fp8_pair_bf16(w11.y >> 16, w11.y >> 24);
                    mma_m16n8k16(raw1[0], ra, rb2);  // tile1 block 2q+1
                    mma_m16n8k16(raw1[1], ra, rb3);
                }
            }

            // Promote each tile's raw partial by its own (sa*sb) in ascending owned-block
            // order — same two-operand pow2 products as the bsb family (exact).
            const float s0 = e8m0_f(sa0[kb]), s1 = e8m0_f(sa1[kb]);
            const float s2 = e8m0_f(sa2[kb]), s3 = e8m0_f(sa3[kb]);
            {
                const float sb = e8m0_f(Sb[sb_grp + (mt0 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc0[0][0] += raw0[0][0] * t0;  acc0[0][1] += raw0[0][1] * t1;
                acc0[0][2] += raw0[0][2] * t0;  acc0[0][3] += raw0[0][3] * t1;
                acc0[1][0] += raw0[1][0] * t2;  acc0[1][1] += raw0[1][1] * t3;
                acc0[1][2] += raw0[1][2] * t2;  acc0[1][3] += raw0[1][3] * t3;
            }
            if (has1) {
                const float sb = e8m0_f(Sb[sb_grp + (mt1 >> 3) * nkb + kb]);
                const float t0 = s0 * sb, t1 = s1 * sb, t2 = s2 * sb, t3 = s3 * sb;
                acc1[0][0] += raw1[0][0] * t0;  acc1[0][1] += raw1[0][1] * t1;
                acc1[0][2] += raw1[0][2] * t0;  acc1[0][3] += raw1[0][3] * t1;
                acc1[1][0] += raw1[1][0] * t2;  acc1[1][1] += raw1[1][1] * t3;
                acc1[1][2] += raw1[1][2] * t2;  acc1[1][3] += raw1[1][3] * t3;
            }
        }

        __nv_bfloat16* Cc = C + (long long)nc * 16 * stride;
        mma_epilogue_olo(sh, acc0, Cc, mt0, grp * R, stride, N);
        __syncthreads();                 // sh reuse barrier between the two tiles' epilogues
        if (has1) mma_epilogue_olo(sh, acc1, Cc, mt1, grp * R, stride, N);
        __syncthreads();                 // sh reuse barrier between chunks
    }
}

// ---- The tiled layout is now the ONLY layout a quantized weight is stored in, so the two consumers
// that read weights element-wise (prefill dequant, embedding gather) must invert the permutation.
// This mirrors `fp4_tile_slot` / `fp8_tile_slot` in quant.rs — the Rust unit test proves the map is a
// bijection; these must agree with it or the model is quietly, subtly wrong.
__device__ __forceinline__ float fp4_tiled_at(const uint8_t* Wt, const uint8_t* Sct, const float* gs,
                                              int nblk, int row, int c) {
    const int r = row & 15, cc = c & 15;
    const int lane = (r & 7) * 4 + ((cc & 7) >> 1);
    const int j    = (r >> 3) | ((cc >> 3) << 1);
    const long long tile = (long long)(row >> 4) * nblk + (c >> 4);
    const uint8_t byte = Wt[tile * 128 + lane * 4 + j];
    const uint8_t nib  = (cc & 1) ? (byte >> 4) : (byte & 0x0F);
    return e2m1_f(nib) * e4m3_f(Sct[tile * 16 + r]) * gs[row >> 4];
}
__device__ __forceinline__ float fp8_tiled_at(const uint8_t* Wt, int nblk, int row, int c) {
    const int r = row & 15, cc = c & 15;
    const int lane = (r & 7) * 4 + ((cc & 7) >> 1);
    const int j    = (cc & 1) | ((r >> 3) << 1) | ((cc >> 3) << 2);
    const long long tile = (long long)(row >> 4) * nblk + (c >> 4);
    return e4m3_f(Wt[tile * 256 + lane * 8 + j]);
}

extern "C" __global__ void dequant_fp4_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                               const uint8_t* Sct, const float* gs, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    out[i] = f2b(fp4_tiled_at(Wt, Sct, gs, K >> 4, (int)(i / K), (int)(i % K)));
}
extern "C" __global__ void dequant_fp8_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                               const float* RowScale, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    int row = (int)(i / K);
    out[i] = f2b(fp8_tiled_at(Wt, K >> 4, row, (int)(i % K)) * RowScale[row]);
}
extern "C" __global__ void embed_gather_fp4_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                                    const uint8_t* Sct, const float* gs,
                                                    const int* tokens, int h, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * batch) return;
    out[i] = f2b(fp4_tiled_at(Wt, Sct, gs, h >> 4, tokens[i / h], i % h));
}
extern "C" __global__ void embed_gather_fp8_tiled_b(__nv_bfloat16* out, const uint8_t* Wt,
                                                    const float* RowScale, const int* tokens,
                                                    int h, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * batch) return;
    int row = tokens[i / h];
    out[i] = f2b(fp8_tiled_at(Wt, h >> 4, row, i % h) * RowScale[row]);
}

// ---- STREAM-style read-bandwidth probe. Settles "what is the roofline, actually?".
//
// Two of our own documents disagreed (248 GB/s "measured sustained" vs 216 GB/s observed by the best
// kernel), and that 15% decides whether the mid-size GEMMs have 10% left in them or 25%. It also
// decides whether a competitor's claimed tok/s is physically possible on this part. So: measure it,
// with the simplest possible pure-read kernel — 16-byte vectorized loads, grid-stride, no writes
// except one guarded sink the compiler cannot fold away.
extern "C" __global__ void bw_read_b(float* sink, const uint4* __restrict__ src, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0xFFFFFFFFu) sink[0] = 1.0f;   // never taken; keeps the loads live
}

// ================== SPLIT the fused projections back into their consumers ==================
//
// The fused GEMM writes [M_tot, N] with M contiguous within a column (C[n*M + m]). Its consumers
// (conv1d, the GDN scan, rope, attention) each want their own [m_i, N] buffer. Rather than thread a
// column stride through every one of them -- including the GDN scan, which carries the bitwise
// losslessness guarantee -- scatter once here. It is pure activation traffic: ~200 KB per GDN layer,
// under 0.2% of a decode step, against a GEMM win of 4.7%.

/// GDN: fused [conv_dim + value_dim + nh + nh, N] -> qkv, z, b, a.
// `nh_src` is the number of b/a rows PRESENT in the fused tensor, `nh` the number this rank consumes,
// and `h0` the first head it owns. Under TP=2 GDN sharding the b/a segments stay REPLICATED at full
// width: they are one row per value head (48), and NVFP4 packs output rows in 16-row tiles, so a 48-row
// segment cannot be halved at tile granularity. They are 0.6 % of in_proj, so replicating the bytes is
// free and the slice happens here instead. Unsharded callers pass nh_src == nh and h0 == 0.
extern "C" __global__ void split_gdn_b(__nv_bfloat16* qkv, __nv_bfloat16* z, __nv_bfloat16* bb,
                                       __nv_bfloat16* aa, const __nv_bfloat16* fused,
                                       int conv_dim, int value_dim, int nh, int batch,
                                       int nh_src, int h0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int mtot = conv_dim + value_dim + 2 * nh_src;
    if (i >= mtot * batch) return;
    const int n = i / mtot, r = i - n * mtot;
    const __nv_bfloat16 v = fused[(long long)n * mtot + r];
    if (r < conv_dim)                  { qkv[(long long)n * conv_dim + r] = v; return; }
    if (r < conv_dim + value_dim)      { z[(long long)n * value_dim + (r - conv_dim)] = v; return; }
    // b/a rows: keep only this rank's head range [h0, h0+nh)
    if (r < conv_dim + value_dim + nh_src) {
        const int hsrc = r - conv_dim - value_dim;
        if (hsrc >= h0 && hsrc < h0 + nh) bb[(long long)n * nh + (hsrc - h0)] = v;
    } else {
        const int hsrc = r - conv_dim - value_dim - nh_src;
        if (hsrc >= h0 && hsrc < h0 + nh) aa[(long long)n * nh + (hsrc - h0)] = v;
    }
}

/// Attention: fused [qg_dim + kv_dim + kv_dim, N] -> qg (q|gate, split later), k, v.
extern "C" __global__ void split_qkv_b(__nv_bfloat16* qg, __nv_bfloat16* k, __nv_bfloat16* v,
                                       const __nv_bfloat16* fused, int qg_dim, int kv_dim, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int mtot = qg_dim + 2 * kv_dim;
    if (i >= mtot * batch) return;
    const int n = i / mtot, r = i - n * mtot;
    const __nv_bfloat16 val = fused[(long long)n * mtot + r];
    if (r < qg_dim)                qg[(long long)n * qg_dim + r] = val;
    else if (r < qg_dim + kv_dim)  k[(long long)n * kv_dim + (r - qg_dim)] = val;
    else                           v[(long long)n * kv_dim + (r - qg_dim - kv_dim)] = val;
}

// ===================================================================================================
// MoE (qwen3_5_moe). Correctness-first bf16 path. Router (softmax→top-k→renorm), a per-token grouped
// expert MLP over the STACKED fused weights, and the sigmoid-gated shared expert. Not yet fused-dequant
// or tuned — this is the oracle-correct reference the NVFP4 grouped kernel will be validated against.
// ===================================================================================================

// Router: logits [E, B] col-major (logit[e + b*E]). Per token: softmax(fp32) over E → top-K → RENORM the
// K probs. Emits ids [K, B] (int) and wts [K, B] (float), col-major. One block per token; smem = E floats.
extern "C" __global__ void moe_router_topk_b(int* ids, float* wts, const __nv_bfloat16* logits,
                                             int E, int K, int B) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E]
    for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = __bfloat162float(logits[e + (long)b * E]);
    __syncthreads();
    // WARP 0 does a parallel top-K: 32 lanes scan E/32 experts each, K rounds of warp-argmax+remove.
    // softmax is monotonic → top-K of probs == top-K of logits, and the full-softmax denominator Z
    // CANCELS in the renorm, so only the K winners need exp().
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            float mx = __bfloat162float(logits[ids[b * K] + (long)b * E]);  // top-1 logit = global max
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = __bfloat162float(logits[ids[j + b * K] + (long)b * E]);
                ev[j] = __expf(l - mx); wsum += ev[j];
            }
            float winv = 1.f / wsum; for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv;
        }
    }
}

// Router (hy_v3): logits [E, B] col-major, bias [E] fp32. Per token: score_e = sigmoid(logit_e) in
// fp32; top-K selected by score_e + bias_e (the learned noaux balancing bias — SELECTION ONLY); the
// combine weights are the UN-biased sigmoid scores at the selected indices, renormalized
// (route_norm) and multiplied by `scaling` (router_scaling_factor = 2.826). Mirrors the reference
// (HYV3TopKRouter: routing_weights.gather(top_k_index) / (sum + 1e-20) * router_scaling_factor).
// Emits ids [K, B] (int) and wts [K, B] (float), col-major — the SAME contract as moe_router_topk_b,
// so every downstream expert path (bf16 moe_experts_b, NVFP4 gemm_moe_mma_fp4/grouped) is shared.
// One block per token; smem = E floats.
extern "C" __global__ void moe_router_topk_sigmoid_b(int* ids, float* wts, const __nv_bfloat16* logits,
                                                     const float* bias, int E, int K, int B,
                                                     int route_norm, float scaling) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E] selection scores
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float l = __bfloat162float(logits[e + (long)b * E]);
        s[e] = 1.f / (1.f + __expf(-l)) + bias[e];     // sigmoid + bias, for selection ONLY
    }
    __syncthreads();
    // WARP 0 does a parallel top-K: 32 lanes scan E/32 experts each, K rounds of warp-argmax+remove.
    // Same scan structure as moe_router_topk_b (deterministic tie-break toward the lower stride slot).
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            // Weights: the UN-biased sigmoid scores, recomputed from the logits so the bias and the
            // in-place removals above never touch them. Same sigmoid formula => same bits as s[] had.
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = __bfloat162float(logits[ids[j + b * K] + (long)b * E]);
                ev[j] = 1.f / (1.f + __expf(-l));
                wsum += ev[j];
            }
            if (route_norm) {
                float winv = 1.f / (wsum + 1e-20f);      // the reference's +1e-20 guard
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv * scaling;
            } else {
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * scaling;
            }
        }
    }
}

// FP32-logits twin of moe_router_topk_sigmoid_b. The hy_v3 reference computes router logits in
// FP32 (F.linear(hidden.float(), gate.float())) and selects on them; rounding the logits to bf16
// first perturbs Hy3's tightly-clustered sigmoid scores by ~1e-3 and flips near-tie top-8
// selections (~10-20% of tokens) — an O(1) output change per flipped token. Identical
// selection/weight math otherwise.
extern "C" __global__ void moe_router_topk_sigmoid_f32_b(int* ids, float* wts, const float* logits,
                                                         const float* bias, int E, int K, int B,
                                                         int route_norm, float scaling) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float s[];                       // [E] selection scores
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float l = logits[e + (long)b * E];
        s[e] = 1.f / (1.f + __expf(-l)) + bias[e];     // sigmoid + bias, for selection ONLY
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = -1;
            for (int e = lane; e < E; e += 32) if (s[e] > lv) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {           // warp-reduce argmax
                float ov = __shfl_down_sync(0xffffffff, lv, o);
                int   oi = __shfl_down_sync(0xffffffff, li, o);
                if (ov > lv) { lv = ov; li = oi; }
            }
            int best = __shfl_sync(0xffffffff, li, 0);
            if (lane == 0) ids[j + b * K] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;             // remove for next round
            __syncwarp();
        }
        if (lane == 0) {
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                float l = logits[ids[j + b * K] + (long)b * E];
                ev[j] = 1.f / (1.f + __expf(-l));
                wsum += ev[j];
            }
            if (route_norm) {
                float winv = 1.f / (wsum + 1e-20f);
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * winv * scaling;
            } else {
                for (int j = 0; j < K; j++) wts[j + b * K] = ev[j] * scaling;
            }
        }
    }
}


// Stacked: gate_up [E, 2I, H] (row-major per expert; rows 0..I=gate, I..2I=up), down [E, H, I]. x,out [H,B]
// col-major. One block per token; smem = (2I + I + H) floats. Correctness-first (scalar dots).
extern "C" __global__ void moe_experts_b(__nv_bfloat16* out, const __nv_bfloat16* x,
                                         const int* ids, const float* wts,
                                         const __nv_bfloat16* gate_up, const __nv_bfloat16* down,
                                         int H, int I, int K, int B) {
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float sm[];
    float* gu  = sm;              // [2I]
    float* hh  = sm + 2 * I;      // [I]
    float* acc = sm + 3 * I;      // [H]
    for (int i = threadIdx.x; i < H; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();
    const __nv_bfloat16* xb = x + (long)b * H;
    for (int j = 0; j < K; j++) {
        int e = ids[j + b * K]; float w = wts[j + b * K];
        const __nv_bfloat16* gW = gate_up + (long)e * (2 * I) * H;   // [2I, H]
        const __nv_bfloat16* dW = down    + (long)e * H * I;         // [H, I]
        for (int r = threadIdx.x; r < 2 * I; r += blockDim.x) {
            float acc0 = 0.f; const __nv_bfloat16* wr = gW + (long)r * H;
            for (int c = 0; c < H; c++) acc0 += __bfloat162float(wr[c]) * __bfloat162float(xb[c]);
            gu[r] = acc0;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < I; r += blockDim.x) {
            float g = gu[r], u = gu[I + r]; hh[r] = (g / (1.f + __expf(-g))) * u;   // silu(gate)*up
        }
        __syncthreads();
        for (int r = threadIdx.x; r < H; r += blockDim.x) {
            float acc0 = 0.f; const __nv_bfloat16* wr = dW + (long)r * I;
            for (int c = 0; c < I; c++) acc0 += __bfloat162float(wr[c]) * hh[c];
            acc[r] += w * acc0;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[i + (long)b * H] = __float2bfloat16(acc[i]);
}

// Shared-expert combine: out[h,b] += sigmoid(gate[b]) * shared[h,b].  gate [1,B], shared/out [H,B].
extern "C" __global__ void moe_shared_combine_b(__nv_bfloat16* out, const __nv_bfloat16* shared,
                                                const __nv_bfloat16* gate, int H, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)H * B) return;
    int b = idx / H;
    float g = __bfloat162float(gate[b]); float sig = 1.f / (1.f + __expf(-g));
    out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + sig * __bfloat162float(shared[idx]));
}

// NVFP4 fused-dequant grouped expert MLP — same structure as moe_experts_b, but the stacked expert
// weights are RAW NVFP4: gate_up_q [E*2I, H/2] nibbles + gate_up_s [E*2I, H/16] E4M3; down_q [E*H, I/2]
// + down_s [E*H, I/16]. Dequant per element: e2m1_f(nib)*e4m3_f(blockscale)*gs (gs = 1/global_scale).
// 4-bit reads = ~4x less weight bandwidth than the bf16 kernel. Uses e2m1_f/e4m3_f defined above.
extern "C" __global__ void moe_experts_fp4_b(
        __nv_bfloat16* out, const __nv_bfloat16* x, const int* ids, const float* wts,
        const uint8_t* gu_q, const uint8_t* gu_s, float gu_gs,
        const uint8_t* dn_q, const uint8_t* dn_s, float dn_gs,
        int hi, int kb) {                                  // packed (H<<16|I), (K<<16|B) — cudarc arity
    const int H = hi >> 16, I = hi & 0xffff, K = kb >> 16, B = kb & 0xffff;
    int b = blockIdx.x; if (b >= B) return;
    extern __shared__ float sm[];
    float* gu  = sm;              // [2I]
    float* hh  = sm + 2 * I;      // [I]
    float* acc = sm + 3 * I;      // [H]
    const int Hb = H >> 1, Hs = H >> 4, Ib = I >> 1, Is = I >> 4;   // bytes/scales per row
    for (int i = threadIdx.x; i < H; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();
    const __nv_bfloat16* xb = x + (long)b * H;
    for (int j = 0; j < K; j++) {
        int e = ids[j + b * K]; float w = wts[j + b * K];
        for (int r = threadIdx.x; r < 2 * I; r += blockDim.x) {     // gate_up: row e*2I + r, len H
            long grow = (long)e * (2 * I) + r;
            const uint8_t* q = gu_q + grow * Hb;
            const uint8_t* s = gu_s + grow * Hs;
            float acc0 = 0.f;
            for (int c = 0; c < H; c++) {
                uint8_t byte = q[c >> 1];
                uint8_t nib  = (c & 1) ? (byte >> 4) : (byte & 0x0F);
                acc0 += (e2m1_f(nib) * e4m3_f(s[c >> 4]) * gu_gs) * __bfloat162float(xb[c]);
            }
            gu[r] = acc0;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < I; r += blockDim.x) {
            float g = gu[r], u = gu[I + r]; hh[r] = (g / (1.f + __expf(-g))) * u;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < H; r += blockDim.x) {          // down: row e*H + r, len I
            long grow = (long)e * H + r;
            const uint8_t* q = dn_q + grow * Ib;
            const uint8_t* s = dn_s + grow * Is;
            float acc0 = 0.f;
            for (int c = 0; c < I; c++) {
                uint8_t byte = q[c >> 1];
                uint8_t nib  = (c & 1) ? (byte >> 4) : (byte & 0x0F);
                acc0 += (e2m1_f(nib) * e4m3_f(s[c >> 4]) * dn_gs) * hh[c];
            }
            acc[r] += w * acc0;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[i + (long)b * H] = __float2bfloat16(acc[i]);
}

// ===================================================================================================
// MoE NVFP4 expert MLP — OPTIMIZED (warp-per-output-row GEMV). Fixes the scalar kernel's two problems:
// (1) occupancy — grid is B*K*2I / B*H warps (fills all SMs, vs 1 block/token); (2) coalescing — a warp's
// 32 lanes read consecutive bytes of one weight row. Raw NVFP4 layout: gate_up_q [E*2I, H/2] + [E*2I,H/16]
// E4M3; down_q [E*H, I/2] + [E*H, I/16]. Same math as moe_experts_fp4_b → validated by the same oracle.
// Intermediate gate_up written to a [B*K, 2I] float scratch (silu barrier splits gate_up from down).
// ===================================================================================================

// gate_up: gu_out[(b*K+slot), r] = Σ_c dequant(gate_up[e·2I+r, c]) · x[b, c].  One warp per (b, slot, r).
extern "C" __global__ void moe_gate_up_fp4_warp(
        float* gu_out, const __nv_bfloat16* x, const int* ids,
        const uint8_t* gu_q, const uint8_t* gu_s, float gs, int H, int I, int K, int B) {
    int wid = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (wid >= B * K * (2 * I)) return;
    int r = wid % (2 * I); int t = wid / (2 * I); int slot = t % K; int b = t / K;
    int e = ids[b * K + slot];
    long ROW = (long)e * (2 * I) + r;
    int Hb = H >> 1;
    const uint32_t* q = reinterpret_cast<const uint32_t*>(gu_q + ROW * Hb);   // 4 bytes = 8 nibbles
    const uint8_t* s = gu_s + ROW * (H >> 4);
    const __nv_bfloat16* xb = x + (long)b * H;
    float acc = 0.f;
    for (int u = lane; u < (Hb >> 2); u += 32) {          // coalesced 128 B/warp; 8 K-elements/lane
        uint32_t w = q[u]; int c0 = u << 3;
        float sc = e4m3_f(s[c0 >> 4]) * gs;               // one scale decode per 8 elements (same block)
        #pragma unroll
        for (int n = 0; n < 8; n++)
            acc += e2m1_f((w >> (4 * n)) & 0xF) * sc * __bfloat162float(xb[c0 + n]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) gu_out[(long)(b * K + slot) * (2 * I) + r] = acc;
}

// silu: h[(bk), r] = silu(gu[bk, r]) * gu[bk, I+r].   idx over B*K*I.
extern "C" __global__ void moe_silu_b(float* h_out, const float* gu, int I, int BK) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    float g = gu[(long)bk * 2 * I + r], u = gu[(long)bk * 2 * I + I + r];
    h_out[idx] = (g / (1.f + __expf(-g))) * u;
}

// down: out[b, r] = Σ_slot w · Σ_c dequant(down[e·H+r, c]) · h[(b*K+slot), c].  One warp per (b, r),
// looping the K slots (each output row written once → no atomics).
extern "C" __global__ void moe_down_fp4_warp(
        __nv_bfloat16* out, const float* h, const int* ids, const float* wts,
        const uint8_t* dn_q, const uint8_t* dn_s, float gs, int H, int I, int K, int B) {
    int wid = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (wid >= B * H) return;
    int r = wid % H, b = wid / H;
    int Ib = I >> 1;
    // Accumulate the K experts' weighted partials PER LANE, then reduce ONCE (vs one reduce/expert).
    float lane_acc = 0.f;
    for (int slot = 0; slot < K; slot++) {
        int e = ids[b * K + slot]; float w = wts[b * K + slot];
        long ROW = (long)e * H + r;
        const uint32_t* q = reinterpret_cast<const uint32_t*>(dn_q + ROW * Ib);
        const uint8_t* s = dn_s + ROW * (I >> 4);
        const float* hb = h + (long)(b * K + slot) * I;
        float p = 0.f;
        for (int u = lane; u < (Ib >> 2); u += 32) {
            uint32_t wv = q[u]; int c0 = u << 3;
            float sc = e4m3_f(s[c0 >> 4]) * gs;
            #pragma unroll
            for (int n = 0; n < 8; n++)
                p += e2m1_f((wv >> (4 * n)) & 0xF) * sc * hb[c0 + n];
        }
        lane_acc += w * p;
    }
    for (int o = 16; o > 0; o >>= 1) lane_acc += __shfl_down_sync(0xffffffff, lane_acc, o);
    if (lane == 0) out[(long)b * H + r] = __float2bfloat16(lane_acc);
}

// ===================================================================================================
// TENSOR-CORE grouped MoE GEMV (the perf lever, ~marlin-style). Reuses gemm_mma_fp4_b's tuned inner
// loop + epilogue verbatim, but the weight tile is chosen by the ON-DEVICE routing (expert = ids[bslot])
// and N=1 (one token per slot). fp32 MMA accumulate → quality preserved. Weights are the REPACKED
// (MMA-layout) stacked experts. Used for gate_up (M=2I, K=H, x_by_slot=0 → X row=token b) and down
// (M=H, K=I, x_by_slot=1 → X row=the per-slot h). Output → per-slot bf16 scratch C[bslot, M].
// grid = (M/16 tiles, B*Kslots), block = 256 (8 warps). The stacked experts are one NVFP4 segment, so
// the global scale is uniform → the epilogue's per-tile gs lookup stays correct with the local tile.
// TP=2 expert-parallel: Wt/Sct/gs hold only this rank's expert band, so the router's GLOBAL id is
// rebased by expert_base. A (token,slot) whose expert is remote (local id outside [0,e_span)) gets
// EXPLICIT zero output rows — C is pool scratch (stale, not zeroed) and moe_combine_experts_b sums
// all K slots unconditionally. Owned-expert math is byte-identical to the unsharded path (base=0, span=ne).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = __float2bfloat16(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;                // N=1: every fragment reads this row
    const long long mt_g = (long long)e * (M >> 4) + mt;               // expert e's (local) weight tile

    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sct = Sct + tile * 16;
        const float s0lo = e4m3_f(sct[g]),      s0hi = e4m3_f(sct[g + 8]);
        const float s1lo = e4m3_f(sct[g + 16]), s1hi = e4m3_f(sct[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);   // N=1 → Xh == Xl
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        // acc[1] holds the n=8..15 output columns; for this N=1 GEMV those are never written by the
        // epilogue (n < N=1), so the second MMA per K-step is pure dead compute — dropped. This roughly
        // halves the tensor-core issue in the hot loop (73%→ closer to the fp8 GEMV's 87% roofline).
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, 1);
}

// Combine the K experts' per-slot down outputs: out[b, r] = Σ_slot wts[b*K+slot] · down_s[(b*K+slot), r].
extern "C" __global__ void moe_combine_experts_b(__nv_bfloat16* out, const __nv_bfloat16* down_s,
                                                 const float* wts, int H, int K, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < K; slot++)
        acc += wts[b * K + slot] * __bfloat162float(down_s[(long)(b * K + slot) * H + r]);
    out[idx] = __float2bfloat16(acc);
}

// silu for the MMA path: h[bk, r] = silu(gu[bk, r]) * gu[bk, I+r], gu bf16 interleaved [B*K, 2I].
// BS(b): the grouped arm passes its poff[ne] bound so rows past the real padded total early-exit
// (the GEMM downstream exits on the same device bound); the slot arm passes poff=NULL (all rows
// real there).
extern "C" __global__ void moe_silu_bf16_b(__nv_bfloat16* h_out, const __nv_bfloat16* gu, int I, int BK,
                                           const int* poff, int ne) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    if (poff && bk >= poff[ne]) return;
    float g = __bfloat162float(gu[(long)bk * 2 * I + r]), u = __bfloat162float(gu[(long)bk * 2 * I + I + r]);
    h_out[idx] = __float2bfloat16((g / (1.f + __expf(-g))) * u);
}

// ===================================================================================================
// TOKEN-GATHER grouped MoE (marlin-style) for batch>1 (prefill/verify). The N=1 kernel re-reads each
// expert's weights once PER TOKEN → catastrophic for prefill (~64 tokens/expert → 64× redundant reads).
// Here: counting-sort the (token,slot) pairs by expert, permute activations so an expert's tokens are
// contiguous, then ONE grouped GEMM reads each expert's weight ONCE for all its tokens (N>1). Pairs are
// padded per expert to a multiple of 8 (one MMA n-tile) so a tile never straddles two experts.
// P = batch*k pairs; ids[p] = expert of pair p (p = token*k + slot).

// TP=2: [e_lo, e_hi) is this rank's expert band; remote pairs are not counted, so no padded group is
// ever allocated for a remote expert (whose weight tiles this rank does not hold).
extern "C" __global__ void moe_count_b(int* count, const int* ids, int P, int e_lo, int e_hi) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    int e = ids[p];
    if (e < e_lo || e >= e_hi) return;
    atomicAdd(&count[e], 1);
}
// Padded prefix offsets (single block): poff[e] = start row of expert e; poff[ne] = total padded rows.
// Also seeds cursor[e]=poff[e] for the scatter (avoids a separate copy).
extern "C" __global__ void moe_offsets_b(int* poff, int* cursor, const int* count, int ne) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    // Pad each expert to a multiple of 16 (was 8) so the grouped GEMM's 16-token blocks never straddle
    // two experts (see gemm_moe_grouped_mma_fp4's weight-reuse fix).
    for (int e = 0; e < ne; e++) { poff[e] = acc; cursor[e] = acc; acc += ((count[e] + 15) / 16) * 16; }
    poff[ne] = acc;
}
// Scatter each pair to its expert's contiguous block. cursor[] starts = poff[] (copied host-side).
// perm_tok[pos]=token, perm_wt[pos]=weight, inv_pos[p]=pos (for the no-atomics combine).
// TP=2: a pair whose expert is remote ([e_lo,e_hi) misses) is NOT enqueued — it contributes an exact
// zero. inv_pos[p] = -1 marks it; moe_combine_grouped_b skips -1 explicitly (its down_perm row would
// be indeterminate — no group was allocated for the remote expert).
extern "C" __global__ void moe_scatter_b(int* perm_tok, float* perm_wt, int* inv_pos, int* cursor,
                                         const int* ids, const float* wts, int P, int k,
                                         int e_lo, int e_hi) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    int e = ids[p];
    if (e < e_lo || e >= e_hi) { inv_pos[p] = -1; return; }
    int pos = atomicAdd(&cursor[e], 1);
    perm_tok[pos] = p / k;         // token index
    perm_wt[pos]  = wts[p];
    inv_pos[p]    = pos;
}
// Per-n-tile expert id: tiles [poff[e]/8, poff[e+1]/8) belong to expert e.
extern "C" __global__ void moe_tilemap_b(int* tile_e, const int* poff, int ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= ne) return;
    for (int nt = poff[e] >> 3; nt < poff[e + 1] >> 3; nt++) tile_e[nt] = e;
}
// Gather activations into permuted order: x_perm[pos, :] = x[:, perm_tok[pos]] (0 for padding pos=-1).
extern "C" __global__ void moe_gather_x_b(__nv_bfloat16* x_perm, const __nv_bfloat16* x,
                                          const int* perm_tok, int H, int Ppad,
                                          const int* poff, int ne) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)Ppad * H) return;
    int pos = idx / H, c = idx % H;
    // BS(b) group-bound exit (EXPERT_BATCH_SCALING §5b): rows past the real padded total are never
    // consumed — the grouped GEMM / quant / silu early-exit on the SAME poff[ne] device bound — so
    // skip writing them (they hold stale pool data; nothing reads past poff[ne]). The fold arm
    // passes ne+1 (its GEMMs bound on poff[ne+1] — the routed+shared total).
    if (poff && pos >= poff[ne]) return;
    int t = perm_tok[pos];
    x_perm[idx] = (t >= 0) ? x[(long)t * H + c] : __float2bfloat16(0.f);
}
// Grouped MMA: same tuned loop as gemm_mma_fp4_b, but the weight tile is expert tile_e[nt] and the 8
// columns are this n-tile's 8 permuted tokens. C[bslot-block, M] = C_perm. N=16 (padding tokens masked
// out downstream by inv_pos). X_perm [Ppad, K], C_perm [Ppad, M], both row-major.
// E16 device-only form: the launch grid is a STATIC host-known upper bound (verify: ppad_max/16;
// prefill: (p + e_span*16)/16 groups) — there is NO host readback of the padded total. Each block
// early-exits against poff[ne] read ON DEVICE (both call sites now launch the bound, so the exit
// fires for every 16-token group past the real padded total).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_grouped_mma_fp4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    // WEIGHT-REUSE FIX: each block now covers 16 tokens (two 8-token n-tiles), reading the expert weight
    // ONCE and MMA-ing it into acc[0] (tokens 0-7) AND acc[1] (tokens 8-15). Previously N=8 fed acc[1]
    // clamped/dead data, so the weight was re-read once per 8 tokens; for prefill (~256 tok/expert) that
    // meant ~32× weight re-reads from HBM (~28.8 GB/layer, the 59%-of-prefill bottleneck). At N=16 the
    // re-reads halve. Requires experts padded to a multiple of 16 (moe_offsets_b) so a block never spans
    // two experts; blockIdx.y now indexes 16-token GROUPS (2 of the per-8-tile tile_e entries).
    // TP=2: tile_e holds GLOBAL ids but only OWNED experts emit tiles (remote counts are zero), so
    // e = tile_e - expert_base always lands inside this rank's band — weight reads stay in bounds.
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mt_g = (long long)e * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sct = Sct + tile * 16;
        const float s0lo = e4m3_f(sct[g]),      s0hi = e4m3_f(sct[g + 8]);
        const float s1lo = e4m3_f(sct[g + 16]), s1hi = e4m3_f(sct[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        uint32_t ra[4];
        ra[0]=fp4_pair_bf16(wq0,s0lo); ra[1]=fp4_pair_bf16(wq0>>8,s0hi); ra[2]=fp4_pair_bf16(wq0>>16,s0lo); ra[3]=fp4_pair_bf16(wq0>>24,s0hi);
        uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
        mma_m16n8k16(acc[0], ra, rb0); mma_m16n8k16(acc[1], ra, rb1);
        ra[0]=fp4_pair_bf16(wq1,s1lo); ra[1]=fp4_pair_bf16(wq1>>8,s1hi); ra[2]=fp4_pair_bf16(wq1>>16,s1lo); ra[3]=fp4_pair_bf16(wq1>>24,s1hi);
        uint32_t rb2[2]={Xl[t+8],Xl[t+12]}, rb3[2]={Xh[t+8],Xh[t+12]};
        mma_m16n8k16(acc[0], ra, rb2); mma_m16n8k16(acc[1], ra, rb3);
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, N);
}
// ===================================================================================================
// u4 burst-load variants (queue #3 probe, 2026-07-31): the fp8 family's winning load structure —
// FOUR pair-iterations' worth of global loads hoisted before the mmas consume them (the
// gemm_dsv4_fp8_bsb1q inner `#pragma unroll` pattern), applied to the fp4 MoE loop. The per-element
// K chains are UNTOUCHED: the warp still owns pairs p = warp + 8k ascending; the unrolled body just
// issues the loads of k, k+1, k+2, k+3 together and executes the mmas in locked ascending order
// (the acc dependency chain serializes them). DSV4 K ∈ {4096, 2048} → npair % 32 == 0, so the
// 4-pair groups are exact (no remainder order change).
extern "C" __global__ __launch_bounds__(256, 3) void gemm_moe_mma_fp4_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = __float2bfloat16(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;
    const long long mt_g = (long long)e * (M >> 4) + mt;

    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p0 = warp; p0 < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2];
        const uint8_t* sct[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + 8 * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sct[u] = Sct + tile * 16;
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sct[u][g]),      s0hi = e4m3_f(sct[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            const int k0 = (p0 + 8 * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
            uint32_t r0[2] = { Xl[t], Xl[t + 4] };
            mma_m16n8k16(acc[0], ra, r0);                              // block 2p,   columns 0..7
            const float s1lo = e4m3_f(sct[u][g + 16]), s1hi = e4m3_f(sct[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { Xl[t + 8], Xl[t + 12] };
            mma_m16n8k16(acc[0], ra, r2);                              // block 2p+1, columns 0..7
        }
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, 1);
}

// ===================================================================================================
// x2-tile streaming variants (queue #3, 2026-07-31): TWO 16-row weight tiles per CTA share the X
// fragments and interleave two INDEPENDENT K-chains in one loop body, doubling the global loads in
// flight per warp at the same occupancy (the pf2-class schedule from the fp8 family, bsb2's decode
// trick). The bind this attacks is the measured one: both single-tile kernels are long-scoreboard
// (global-load-latency) bound at ~62-72% of the 255 GB/s roofline (ncu: stalled_long_scoreboard 8.8-
// 12.6 per issue-active, issue 49-63%, occupancy 94-98%, zero stack/spills).
// BITWISE CONTRACT: each output element's K chain is UNTOUCHED — same pair partition (p = warp + 8k,
// ascending), same per-pair block order (2p then 2p+1), same per-warp fp32 accumulation, same
// warp-order epilogue reduction (mma_epilogue). The x2 kernels differ from the single-tile kernels
// ONLY in which block computes which mt pair and in instruction scheduling, so every existing gate
// (N=1 vs grouped at every N in 2..=16, wide-vs-chunked, serving hashes) remains the contract.
// grid.x = M/32 (was M/16); launch_bounds(256,4) keeps ptxas at <=64 regs (4 CTAs/SM).
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, int Kslots, int x_by_slot, int expert_base, int e_span)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;
    const int e = ids[bslot] - expert_base;                            // LOCAL expert id
    __nv_bfloat16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                                        // remote expert: contribute 0
        if (threadIdx.x < 16) {
            Cb[mt0 * 16 + threadIdx.x] = __float2bfloat16(0.f);
            Cb[mt1 * 16 + threadIdx.x] = __float2bfloat16(0.f);
        }
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / Kslots);
    const __nv_bfloat16* Xtok = X + (long long)xrow * K;                // N=1: every fragment reads this row
    const long long mtA = (long long)e * (M >> 4) + mt0;
    const long long mtB = mtA + 1;                                     // adjacent tile of the same expert

    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = tileA + nblk;                          // +1 mt tile, same k-blocks
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = Sct + tileA * 16;
        const uint8_t* sctB = Sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);   // shared by both tiles
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                                // A block 2p,   columns 0..7
        mma_m16n8k16(accB[0], raB, rb0);                                // B block 2p,   columns 0..7
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                                // A block 2p+1, columns 0..7
        mma_m16n8k16(accB[0], raB, rb2);                                // B block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, accA, Cb, nullptr, gs + (long long)e * (M >> 4), mt0, M, 1);
    __syncthreads();                                                    // sh read->next-write hazard (the dense kernel's documented barrier)
    mma_epilogue(sh, accB, Cb, nullptr, gs + (long long)e * (M >> 4), mt1, M, 1);
}

extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mtA = (long long)e * (M >> 4) + mt0;
    const long long mtB = mtA + 1;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = tileA + nblk;                          // +1 mt tile, same k-blocks
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = Sct + tileA * 16;
        const uint8_t* sctB = Sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        uint32_t rb1[2] = { Xh[t], Xh[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                                // A block 2p,   columns 0..7
        mma_m16n8k16(accA[1], raA, rb1);                                // A block 2p,   columns 8..15
        mma_m16n8k16(accB[0], raB, rb0);                                // B block 2p,   columns 0..7
        mma_m16n8k16(accB[1], raB, rb1);                                // B block 2p,   columns 8..15
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        uint32_t rb3[2] = { Xh[t + 8], Xh[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                                // A block 2p+1, columns 0..7
        mma_m16n8k16(accA[1], raA, rb3);                                // A block 2p+1, columns 8..15
        mma_m16n8k16(accB[0], raB, rb2);                                // B block 2p+1, columns 0..7
        mma_m16n8k16(accB[1], raB, rb3);                                // B block 2p+1, columns 8..15
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, accA, Cb, nullptr, gs + (long long)e * (M >> 4), mt0, M, N);
    __syncthreads();                                                    // sh read->next-write hazard (the dense kernel's documented barrier)
    mma_epilogue(sh, accB, Cb, nullptr, gs + (long long)e * (M >> 4), mt1, M, N);
}

// ---- gemm_moe_grouped_mma_fp4_u4 — the same burst-load structure for the grouped kernel: FOUR
// pair-iterations' loads hoisted (weights, scales, both n-tile X fragments) before the mmas, which
// execute in the locked per-element order (per pair: block 2p cols 0..7, 2p cols 8..15, 2p+1 cols
// 0..7, 2p+1 cols 8..15, ascending pairs). DSV4 K % 1024 == 0 → npair % 32 == 0, groups exact.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 16;
    const int e = tile_e[nt * 2] - expert_base;                        // LOCAL expert id
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K;  // this group's 16 tokens (rows 0..15)
    const long long mt_g = (long long)e * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p0 = warp; p0 < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2], rb[4][4], rh[4][4];
        const uint8_t* sct[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + 8 * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sct[u] = Sct + tile * 16;
            const int k0 = (p0 + 8 * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
            const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
            rb[u][0] = Xl[t];      rb[u][1] = Xl[t + 4];
            rb[u][2] = Xl[t + 8];  rb[u][3] = Xl[t + 12];
            rh[u][0] = Xh[t];      rh[u][1] = Xh[t + 4];
            rh[u][2] = Xh[t + 8];  rh[u][3] = Xh[t + 12];
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sct[u][g]),      s0hi = e4m3_f(sct[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            uint32_t r0[2] = { rb[u][0], rb[u][1] };
            uint32_t r1[2] = { rh[u][0], rh[u][1] };
            mma_m16n8k16(acc[0], ra, r0);                              // block 2p,   columns 0..7
            mma_m16n8k16(acc[1], ra, r1);                              // block 2p,   columns 8..15
            const float s1lo = e4m3_f(sct[u][g + 16]), s1hi = e4m3_f(sct[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { rb[u][2], rb[u][3] };
            uint32_t r3[2] = { rh[u][2], rh[u][3] };
            mma_m16n8k16(acc[0], ra, r2);                              // block 2p+1, columns 0..7
            mma_m16n8k16(acc[1], ra, r3);                              // block 2p+1, columns 8..15
        }
    }
    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, Cb, nullptr, gs + (long long)e * (M >> 4), mt, M, N);
}

// Combine: out[:, t] = Σ_slot perm_wt[inv_pos[t*k+slot]] · down_perm[inv_pos[t*k+slot], :].  No atomics.
// TP=2: inv_pos == -1 marks a remote-expert pair (moe_scatter_b) — skip it; it contributes an exact
// zero (skipping vs adding 0.0f is the same fp32 sum, slot order preserved).
extern "C" __global__ void moe_combine_grouped_b(__nv_bfloat16* out, const __nv_bfloat16* down_perm,
                                                 const float* perm_wt, const int* inv_pos,
                                                 int H, int k, int B) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++) {
        int pos = inv_pos[b * k + slot];
        if (pos < 0) continue;
        acc += perm_wt[pos] * __bfloat162float(down_perm[(long)pos * H + r]);
    }
    out[idx] = __float2bfloat16(acc);
}

// verify_chain_params_b — E13: expand the chain-identity verify inputs ON DEVICE from 3 scalars
// (pos_start, slot, n), so a captured verify graph needs only a tiny params write per replay instead
// of six host->device array uploads + a pipeline sync. Fills exactly the arrays the eager host path
// builds in verify_forward_core_topo (chain identity: KV pos == RoPE pos == pos_start+t; parent[t] =
// t-1 packed with the lane slot; winsrc the width-ck stencil; path[b][d] = d).
extern "C" __global__ void verify_chain_params_b(
    const int* params,   // [pos_start, slot, n, ck]
    int* pos, int* rope, int* slot_ids, int* winsrc, int* parent, unsigned char* path,
    int max_verify) {
    const int pos_start = params[0];
    const int slot = params[1];
    const int n = params[2];
    const int ck = params[3];
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    pos[t] = pos_start + t;
    rope[t] = pos_start + t;
    slot_ids[t] = slot;
    for (int j = 0; j < ck; j++) winsrc[t * ck + j] = t + j - (ck - 1);
    parent[t] = ((slot & 0xFFFF) << 16) | ((t - 1) & 0xFFFF);
    for (int d = 0; d < max_verify; d++) path[t * max_verify + d] = (unsigned char)d;
}

// ---- R3.2 (L2): vocab-parallel LM head under TP=2 — the 8 B maxloc exchange ----
// dsv4_argmax_pair_b: single-block (val, idx) argmax over n fp32 logits, total order (val
// desc, idx asc) — the same order as the host dsv4_argmax (sequential ascending, strict >) and
// the engine's argmax_pass1/2 (halving tree keeps the LOWER index on ties). Each thread strides
// n tracking its local best; a fixed halving tree reduces the 1024 partials. Writes
// pair = {val f32 bits, GLOBAL idx} (idx_base = the rank's vocab-half offset). Grid (1,1,1),
// block (1024,1,1) — one launch per token.
extern "C" __global__ void dsv4_argmax_pair_b(const float* __restrict__ logits, int n,
                                              unsigned int* __restrict__ pair, int idx_base) {
    const int tid = threadIdx.x;
    float bv = -1e30f; unsigned bi = 0xffffffffu;
    for (int i = tid; i < n; i += 1024) {
        float v = logits[i];
        if (v > bv || (v == bv && (unsigned)i < bi)) { bv = v; bi = (unsigned)i; }
    }
    __shared__ float sv[1024];
    __shared__ unsigned si[1024];
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s2 = 512; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            float ov = sv[tid + s2]; unsigned oi = si[tid + s2];
            if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { pair[0] = __float_as_uint(sv[0]); pair[1] = (unsigned)idx_base + si[0]; }
}

// tp_wait_maxloc_b — the K2 of the maxloc barrier: each rank shipped its 8 B (val, idx) local
// winner with K1 (tp_gate_copy_signal, nbytes=8); wait for the peer's, then write the GLOBAL
// winner's index. Total order (val desc, idx asc) — deterministic and identical on both ranks,
// so SPMD lockstep holds without a token broadcast. NaN loses every comparison (matches the
// host argmax's strict-> skip).
extern "C" __global__ void tp_wait_maxloc_b(tp_dev_ctx* c, const unsigned int* local_pair, int* out_idx) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);
    if (tp_spin_until_ge(c, tp_flag(c, TP_F_CPU_DONE), e, 1)) return;   // I5 gate; abort => no-op
    tp_stamp(c, e, TP_GTS_K2_GO);
    if (threadIdx.x == 0) {
        const unsigned int* peer = (const unsigned int*)(c->recv_ring + (size_t)s * c->slot_stride);
        const float lv = __uint_as_float(local_pair[0]);
        const float pv = __uint_as_float(peer[0]);
        const unsigned li = local_pair[1], pi = peer[1];
        const bool local_wins = (lv > pv) || (lv == pv && li < pi);
        out_idx[0] = (int)(local_wins ? li : pi);
    }
}

// v2 twin of tp_wait_maxloc_b: the same two-stage gate as tp_wait_add_g (hint, then the 8 B
// payload's tail at slot+8 with the deadline -> status 11), winner logic byte-identical.
extern "C" __global__ void tp_wait_maxloc_g(tp_dev_ctx* c, const unsigned int* local_pair, int* out_idx) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);
    asm volatile("griddepcontrol.launch_dependents;");
    if (tp_spin_until_ge(c, tp_flag(c, TP_F_PEER_COMMITTED), e, 1)) return;
    const unsigned char* peer = c->recv_ring + (size_t)s * c->slot_stride;
    __shared__ int s_timeout;
    if (threadIdx.x == 0) {
        s_timeout = 0;
        const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
        const unsigned long long* tail =
            (const unsigned long long*)(peer + 8);   // 8 B payload, tail at slot+8
        // maxloc always ships exactly 8 B (never a slot-filling "wide" barrier), so the short
        // stage-B deadline is correct here — no adaptive bound needed (B8 §1.5-3).
        unsigned long long deadline = tp_globaltimer() + TP_TAIL_WAIT_NS;
        unsigned ns = 64, cap = 512u;
        while (tp_ld_relaxed(tail) != e) {
            if (tp_ld_relaxed(ab)) { s_timeout = 1; break; }
            if (tp_globaltimer() >= deadline) {
                tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                printf("[maxloc-stageB-timeout] rank=%d epoch=%llu\n", c->rank, e);
                s_timeout = 1;
                break;
            }
            __nanosleep(ns);
            if (ns < cap) ns <<= 1;
        }
        if (!s_timeout) tp_fence_acquire();
    }
    __syncthreads();
    if (s_timeout) return;
    tp_stamp(c, e, TP_GTS_K2_GO);
    if (threadIdx.x == 0) {
        const unsigned int* p = (const unsigned int*)peer;
        const float lv = __uint_as_float(local_pair[0]);
        const float pv = __uint_as_float(p[0]);
        const unsigned li = local_pair[1], pi = p[1];
        const bool local_wins = (lv > pv) || (lv == pv && li < pi);
        out_idx[0] = (int)(local_wins ? li : pi);
    }
    __syncthreads();
    if (threadIdx.x == 0) tp_st_release(tp_flag(c, TP_F_RX_DONE), e);
}

// ===================================================================================================
// E12/E8/E9 re-implementation (2026-08-11) — the fold kernels + the E8 fp32 tail + the fused
// norm+rope WIP kernel. Everything below matches the surviving orchestration (gpu.rs's moe_batch
// fold paths + E8 tail + the rmsnorm_rope_b call sites) exactly — the launch args, smem, grids.
// ===================================================================================================

// ---- fused per-head RMSNorm + RoPE (the lost WIP kernel; replaces the split rmsnorm_b + rope_b
// pair at the full_attn_batch / prefill per-head sites). One block per (seq, head), IN-PLACE on
// `out` (x == out — the call sites pass 11 args: out, w, cos, sin, nh, hd, rdim, B, eps, off,
// row_stride). The norm math is rmsnorm_perhead_b's (v = out[base+tid], tree-reduce s[tid] = v^2,
// inv = rsqrtf(s[0]/hd + eps), out = v*inv*(1+w[tid])), then the rope is rope_b's per-head rotary
// pair on the normed row — byte-identical to running rmsnorm_perhead_b then rope_b on the same row.
// F0: `off`/`row_stride` give the row pitch of the buffer — packed buffers pass (0, nh*hd) /
// (0, nkv*hd), the fused qkv VIEW passes (0, mtot) for q / (qg_dim, mtot) for k — so the SAME
// kernel serves both layouts with identical arithmetic (base = off + b*row_stride + head*hd).
extern "C" __global__ void rmsnorm_rope_b(__nv_bfloat16* out, const float* w,
                                          const float* cos, const float* sin, int nh, int hd,
                                          int rdim, int B, float eps, int off, int row_stride) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * row_stride + off + (long long)head * hd;
    float v = (tid < hd) ? b2f(out[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) out[base + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(out[base + tid]);
        float x2 = b2f(out[base + tid + half]);
        float c = cos[cb], sn = sin[cb];
        out[base + tid] = f2b(x1 * c - x2 * sn);
        out[base + tid + half] = f2b(x2 * c + x1 * sn);
    }
}

// ===================================================================================================
// F4 (EXPERT_FUSION_PASSES §3.4): fold write_kv_b into the k-side rope kernel, with the raw v copy
// riding the same launch. Grid (b·nkv,1,1), block 256 (pad for the v-copy phase), smem hd fp32.
// Phase 1: norm+rope the k row IN PLACE exactly as rmsnorm_rope_b (same v^2 tree — with blockDim
// 256 the threads tid >= hd contribute exact +0.0f leaves, so s[0] and every f2b round are
// bit-identical to a 128-thread launch — same rsqrtf, rope reads the ROUNDED values). Phase 2:
// write the roped k row AND the raw v row to the cache at (slot_ids[b], kvh, pos[b]) with
// write_kv_b's indexing. Bit-exact by construction: norm/rope math untouched, the write is a copy.
// F0: the k/v rows are VIEWS into the fused qkv GEMM output [token, [q|k|v]] — k at
// k_off + b*mtot + head*hd, v at k_off + nkv*hd + b*mtot + head*hd, mtot = k_off + 2*nkv*hd.
// FUSED-VIEW LAYOUT ONLY: the GB10_NO_QKV_VIEW / GB10_NO_FUSED_PERHEAD_ROPE escapes restore the
// un-fused write path (write_kv_b / write_kv_b_q4 on packed buffers). 12-arg launch (packed):
//   stride_nkv = nkv<<19 | stride        (stride = cache positions per slot, 19 bits)
//   hd_rdim    = hd<<16  | rdim
//   k_off      = the q|gate segment width (qg_dim) — derives mtot and the v segment offset.
extern "C" __global__ void rmsnorm_rope_kvwrite_b(
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    __nv_bfloat16* qkv, const float* w,
    const float* cos, const float* sin,
    const int* pos, const int* slot_ids,
    int stride_nkv, int hd_rdim, int k_off, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    const int stride = stride_nkv & 0x7FFFF;
    const int nkv = (unsigned)stride_nkv >> 19;
    const int hd = (unsigned)hd_rdim >> 16;
    const int rdim = hd_rdim & 0xFFFF;
    int blk = blockIdx.x;
    int b = blk / nkv;
    int head = blk % nkv;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const long long row_stride = (long long)k_off + 2LL * nkv * hd;
    long long kbase = (long long)b * row_stride + k_off + (long long)head * hd;
    // Phase 1: norm+rope k in place — rmsnorm_rope_b's math verbatim (tree over hd, exact-zero
    // padding for tid >= hd; the tree level hd/2..1 is identical to a blockDim==hd launch).
    float v = (tid < hd) ? b2f(qkv[kbase + tid]) : 0.0f;
    if (tid < hd) s[tid] = v * v;
    __syncthreads();
    for (int s2 = hd / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) qkv[kbase + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(qkv[kbase + tid]);
        float x2 = b2f(qkv[kbase + tid + half]);
        float c = cos[cb], sn = sin[cb];
        qkv[kbase + tid] = f2b(x1 * c - x2 * sn);
        qkv[kbase + tid + half] = f2b(x2 * c + x1 * sn);
    }
    __syncthreads();   // phase 2 reads the roped row
    // Phase 2: write the roped k row + the RAW v row (write_kv_b indexing: (slot*nkv+h)*stride+pos).
    const long long vbase = kbase + (long long)nkv * hd;
    const int slot = slot_ids[b];
    const long long coff = ((long long)slot * nkv + head) * (long long)stride + pos[b];
    if (tid < hd) {
        k_cache[coff * hd + tid] = qkv[kbase + tid];
        v_cache[coff * hd + tid] = qkv[vbase + tid];
    }
}

// ---- F4 q4 twin: identical phase 1; phase 2 packs the roped k row + the raw v row with
// kvq16_pack into the q4 cache — write_kv_b_q4's quantize-at-write op order preserved byte-for-byte
// (per-16 amax ascending, s = amax/7 e4m3, lrintf clamp ±7, kvq16_pack byte order): the pack is a
// pure function of the 16 input values, and those values are exactly what write_kv_b_q4 reads
// after rmsnorm_rope_b — same inputs, same kvq16_pack => same 9 B per block.
extern "C" __global__ void rmsnorm_rope_kvwrite_q4_b(
    unsigned char* k_cache, unsigned char* v_cache,
    __nv_bfloat16* qkv, const float* w,
    const float* cos, const float* sin,
    const int* pos, const int* slot_ids,
    int stride_nkv, int hd_rdim, int k_off, float eps) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;");
    const int stride = stride_nkv & 0x7FFFF;
    const int nkv = (unsigned)stride_nkv >> 19;
    const int hd = (unsigned)hd_rdim >> 16;
    const int rdim = hd_rdim & 0xFFFF;
    int blk = blockIdx.x;
    int b = blk / nkv;
    int head = blk % nkv;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const long long row_stride = (long long)k_off + 2LL * nkv * hd;
    long long kbase = (long long)b * row_stride + k_off + (long long)head * hd;
    float v = (tid < hd) ? b2f(qkv[kbase + tid]) : 0.0f;
    if (tid < hd) s[tid] = v * v;
    __syncthreads();
    for (int s2 = hd / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)hd + eps);
    if (tid < hd) qkv[kbase + tid] = f2b(v * inv * (1.0f + w[tid]));
    __syncthreads();
    int half = rdim / 2;
    if (tid < half) {
        long long cb = (long long)b * rdim + tid;
        float x1 = b2f(qkv[kbase + tid]);
        float x2 = b2f(qkv[kbase + tid + half]);
        float c = cos[cb], sn = sin[cb];
        qkv[kbase + tid] = f2b(x1 * c - x2 * sn);
        qkv[kbase + tid + half] = f2b(x2 * c + x1 * sn);
    }
    __syncthreads();
    // Phase 2: pack the roped k row + the raw v row (write_kv_b_q4 indexing: crow =
    // (slot*nkv+h)*stride + pos, coff = crow*KVQ_ROW_BYTES(hd) + blk*12).
    const long long vbase = kbase + (long long)nkv * hd;
    const int nb = hd / KVQ_BLK;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + head) * (long long)stride + pos[b];
    const long long coff = crow * (long long)KVQ_ROW_BYTES(hd);
    if (tid < nb) {
        kvq16_pack(qkv + kbase + (long long)tid * KVQ_BLK, k_cache + coff + (long long)tid * 12);
        kvq16_pack(qkv + vbase + (long long)tid * KVQ_BLK, v_cache + coff + (long long)tid * 12);
    }
}

// ---- E8 fp32-partial tail: accumulate the rank's bf16 routed partial into the fp32 shared partial
// (exact in fp32 — the routed sum is the bf16 `out`, whose value in fp32 is exact). The fp32 sum
// then crosses the wire and rounds to bf16 exactly ONCE in tp_wait_add mode 1 (tp_all_reduce_fp32).
extern "C" __global__ void add_f32_bf16_b(float* acc, const __nv_bfloat16* routed, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) acc[i] += b2f(routed[i]);
}

// ---- moe_offsets_fold_b: moe_offsets_b + the E12 shared-expert region. poff grows to ne+2:
// poff[ne] = the routed padded total, poff[ne+1] = poff[ne] + pad32(ns) (the shared region holds
// pad32(ns) rows — one per token, slot k of every token, in FIXED token order). Seeds cursor[e] =
// poff[e] for the scatter, like the plain kernel.
extern "C" __global__ void moe_offsets_fold_b(int* poff, int* cursor, const int* count, int ne, int ns) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < ne; e++) { poff[e] = acc; cursor[e] = acc; acc += ((count[e] + 15) / 16) * 16; }
    poff[ne] = acc;
    poff[ne + 1] = acc + ((ns + 31) / 32) * 32;
}

// ---- moe_tilemap_fold_b: moe_tilemap_b over ne+1 blocks — the routed experts' tiles plus the
// shared region's tiles ([poff[ne]>>3, poff[ne+1]>>3), marked with the pseudo-expert id ne — the
// fold GEMMs branch on the ROW range, so the value only distinguishes the region).
extern "C" __global__ void moe_tilemap_fold_b(int* tile_e, const int* poff, int ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x; if (e > ne) return;
    if (e < ne) {
        for (int nt = poff[e] >> 3; nt < poff[e + 1] >> 3; nt++) tile_e[nt] = e;
    } else {
        for (int nt = poff[ne] >> 3; nt < poff[ne + 1] >> 3; nt++) tile_e[nt] = ne;
    }
}

// ---- moe_shared_scatter_b: the shared pairs (slot k of every token) land in the dedicated region
// in FIXED token order — position of token b is poff[ne]+b, so the combine needs no per-pair
// bookkeeping and the fp32 add order is deterministic. perm_wt = sigmoid(sgate[b]) (the old
// moe_shared_combine_b weight) or exactly 1.0 for hy_v3's ungated shared expert.
extern "C" __global__ void moe_shared_scatter_b(int* perm_tok, float* perm_wt, const int* poff,
                                                const __nv_bfloat16* sgate, int ne, int ns, int gated) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= ns) return;
    int pos = poff[ne] + b;
    perm_tok[pos] = b;
    perm_wt[pos] = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
}

// ---- moe_silu_fold_b: the E12/E8 fold-aware silu. Routed rows (bk in the routed band, or
// bk%(k+1) != k in the slot path) are byte-identical to moe_silu_bf16_b. The shared rows write the
// RANK's half of the act row: sk_off = sdesc[3]&0xffffffff (rank·sdn_k), sl = sdesc[3]>>32 (sdn_k —
// the gu half width; the replicated path has sk_off = 0, sl = si, i.e. the full row). The gu reads
// are the rank's bands of the slot's gu rows (the fold GU GEMM wrote them there).
// Slot path: poff = NULL, ne = k -> the shared slot is row % (k+1) == k.
// Grouped path: poff = poff, ne = ne -> the shared region is [poff[ne], poff[ne+1]).
extern "C" __global__ void moe_silu_fold_b(__nv_bfloat16* h_out, const __nv_bfloat16* gu,
                                           int I, int BK, const int* poff, int ne,
                                           const long long* sdesc) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
    // E13 fold-race fix: the GU GEMMs write only rows [0, poff[ne+1]) (device early-exit on the
    // fold bound) while the host grid bound BK (ppad_cap/ppad_max) exceeds it — the tail rows
    // held STALE pool memory and this kernel READ them, making the wide-fold prefill
    // nondeterministic run-to-run. Exit at the device-computed real total (slot path: BK is
    // exact, no poff). Rows [poff[ne+1], ceil32) that the GEMM wrote are padding — never read
    // downstream (down GEMM + combine both exit at poff[ne+1]).
    if (poff) {
        const int real = poff[ne + 1];
        // E13: the down GEMM's grid covers rows < ceil32(real) (its tail half reads h_p there) —
        // WRITE ZEROS through that range so the tail reads are deterministic (its outputs are
        // padding, never combined); rows >= ceil32(real) are never read by any kernel.
        const int tail_end = (real + 31) & ~31;
        if (bk >= real) {
            if (bk < tail_end) h_out[idx] = f2b(0.f);
            return;
        }
    }
    const int sk_off = (int)(sdesc[3] & 0xffffffffu);
    const int sl = (int)(sdesc[3] >> 32);
    bool shared;
    if (poff) shared = (bk >= poff[ne] && bk < poff[ne + 1]);
    else      shared = (bk % (ne + 1) == ne);
    int rp = r - sk_off;
    if (!shared || (rp >= 0 && rp < sl)) {
        float g = b2f(gu[(long)bk * 2 * I + r]), u = b2f(gu[(long)bk * 2 * I + I + r]);
        h_out[idx] = f2b(silu_f(g) * u);
    }
}

// ---- moe_combine_experts_fold_b: the slot-path fold combine. Routed slots via wts (slot order
// unchanged — routed slots first), then the shared slot (row (b·(k+1)+k) of down_s) with the
// sigmoid gate weight. fold_shared==1 (wide prefill, E8 shard): the shared slot joins the SAME
// fp32 sum — one rounding (prefill reassociation accepted). fold_shared==0 (the decode/verify
// regime): routed-only, bf16 round here, then the shared add rounds again in
// moe_shared_slot_combine_b post-reduce — the old two-stage rounding, bit-identical to the
// separate-launch path.
extern "C" __global__ void moe_combine_experts_fold_b(__nv_bfloat16* out, const __nv_bfloat16* down_s,
                                                      const float* wts, const __nv_bfloat16* sgate,
                                                      int H, int k, int B, int fold_shared, int gated) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++)
        acc += wts[b * k + slot] * b2f(down_s[(long)(b * (k + 1) + slot) * H + r]);
    if (fold_shared) {
        float w = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
        acc += w * b2f(down_s[(long)(b * (k + 1) + k) * H + r]);
    }
    out[idx] = f2b(acc);
}

// ---- moe_combine_grouped_fold_b: the grouped-path fold combine. Routed slots via inv_pos (skip
// -1 — remote experts under TP, exactly moe_combine_grouped_b), then — fold_shared only — the
// shared slot at poff[ne]+b (fixed token order, perm_wt pre-filled by moe_shared_scatter_b). With
// fold_shared==0 the shared add happens post-reduce in moe_shared_slot_combine_b (two-stage
// rounding); the reduce placement mirrors the old path (per-sub-batch reduces are elementwise and
// token-independent, so sub-batching is combine-invariant).
extern "C" __global__ void moe_combine_grouped_fold_b(__nv_bfloat16* out, const __nv_bfloat16* down_perm,
                                                      const float* perm_wt, const int* inv_pos,
                                                      const int* poff, int H, int k, int B, int ne,
                                                      int fold_shared) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    float acc = 0.f;
    for (int slot = 0; slot < k; slot++) {
        int pos = inv_pos[b * k + slot];
        if (pos < 0) continue;
        acc += perm_wt[pos] * b2f(down_perm[(long)pos * H + r]);
    }
    if (fold_shared) {
        int pos = poff[ne] + b;
        acc += perm_wt[pos] * b2f(down_perm[(long)pos * H + r]);
    }
    out[idx] = f2b(acc);
}

// ---- moe_shared_slot_combine_b: the replicated-shared post-reduce add — out[b,r] =
// f2b(b2f(out[b,r]) + w·b2f(down[row(b),r])), the SECOND stage of the two-stage rounding (the
// routed sum was already bf16). mode==0 (slot path): row = b·stride + slot; mode==1 (grouped
// path): row = poff[ne] + b (the fixed-token-order shared region). gated: w = sigmoid(sgate[b]),
// else exactly 1.0.
extern "C" __global__ void moe_shared_slot_combine_b(__nv_bfloat16* out, const __nv_bfloat16* down,
                                                     const int* poff, const __nv_bfloat16* sgate,
                                                     int H, int stride, int slot, int B, int ne,
                                                     int mode, int gated) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)B * H) return;
    int r = idx % H, b = idx / H;
    int row = mode ? (poff[ne] + b) : (b * stride + slot);
    float w = gated ? (1.f / (1.f + __expf(-b2f(sgate[b])))) : 1.f;
    out[idx] = f2b(b2f(out[idx]) + w * b2f(down[(long)row * H + r]));
}

// ===================================================================================================
// THE SHARED-SLOT DESCRIPTOR (device i64[16], written once at load; the fold GEMMs take its
// POINTER as their single extra arg — cudarc launch tuples cap at 12 elements):
//   [0..4)  gate_up { qweight, scales, gs, (SK<<32)|sk_off }   SK = h, sk_off = 0 (col-shard: M)
//   [4..8)  down    { qweight, scales, gs, (sdn_k<<32)|rank·sdn_k }
//   [8]     the shared slot's gu M (2·si, or 2·si_local = si under the shard)
//   [12]    the down's M (= h, uniform no-op; keeps the uniform desc[8] read in-bounds)
// A launch passes the pointer of the descriptor it reads: the GU GEMMs pass the BASE (their own
// desc at [0..4), the down's sk at [7] for the rank's band offset), the DOWN GEMMs pass base+32
// (their own desc at [0..4) with sk_off = rank·sdn_k, the down's M at [8]).
// ===================================================================================================

// The slot-path fold GEMM: gemm_moe_mma_fp4 with Kslots = k+1 columns (the last is the shared
// slot). Routed slots are byte-identical to gemm_moe_mma_fp4 (same per-element k-chain, same
// N=1 epilogue). The shared slot reads its weight geometry from the descriptor: kk = desc[3]>>32,
// X k-offset = desc[3]&0xffffffff, its own M = desc[8] (== the launch M -> identity C rows; the
// sharded gu half -> the C rows map into the rank's bands of the full-width slot: gate rows at
// [rank_off, rank_off+sl), up rows at [2·sl+rank_off, 2·sl+rank_off+sl) with sl = desc[8]/2 and
// rank_off = the down desc's sk_off — the weight row m' lands at C row m' + rank_off for the gate
// band and m' + sl + rank_off for the up band, which is exactly the paired-ColSegs layout).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        // Sharded-GU epilogue: the C row is the launch tile's row; the weight row = m + m_shift
        // (the rank's band). The whole block shares one band, so the per-tile scale is gs_w[mt_s].
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        // gs_w is ALREADY the per-expert base (gs + e·(M>>4), or the shared tensor's own gs); mt_g
        // is the full tile index used for the WEIGHT reads and must NOT be added again — the
        // epilogue indexes gs_w[mt] itself. (The old gs_w + mt_g read the scale at
        // 2·(e·(M>>4) + mt) — double the expert AND tile offsets — garbage/OOB for e>0, mt>0.)
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ===================================================================================================
// E11 — expert-GEMM tuning variants of gemm_moe_mma_fp4_fold (branch work/e11). BITWISE CONTRACT:
// every variant preserves the exact per-element k-chain — same pair partition (p = warp + 8k
// ascending), same per-pair block order (2p then 2p+1), same per-warp fp32 accumulation, same
// warp-order epilogue reduction. Variants differ ONLY in block→tile mapping or instruction
// scheduling, so col-0 batch-invariance and the oracle-identical MoE gate are untouched
// (gates: --probe-binv + bench_mtp LOSSLESS + acceptance).

// ---- gemm_moe_mma_fp4_fold_u4 — the u4 burst-load structure (gemm_moe_mma_fp4_u4's winning
// load schedule) applied to the fold kernel: FOUR pair-iterations' global loads hoisted per loop
// step, then the MMAs execute in locked ascending order (the acc dependency chain serializes
// them). Same pair ownership as the plain fold (p = warp + 8k ascending), so bit-identical.
// npair % 4 == 0 is NOT required — a tail guard runs the leftover pairs in the plain ascending
// loop (same order). launch_bounds(256,3) mirrors the u4 probe kernel's occupancy.
extern "C" __global__ __launch_bounds__(256, 3) void gemm_moe_mma_fp4_fold_u4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    int p0;
    for (p0 = warp; p0 + 3 * MMA_NW < npair; p0 += MMA_NW * 4) {
        uint32_t wq[4][2];
        const uint8_t* sctv[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const long long tile = mt_g * nblk + ((p0 + MMA_NW * u) << 1);
            wq[u][0] = Wt32[tile * 32 + lane];
            wq[u][1] = Wt32[tile * 32 + 32 + lane];
            sctv[u] = sct + tile * 16;
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float s0lo = e4m3_f(sctv[u][g]),      s0hi = e4m3_f(sctv[u][g + 8]);
            uint32_t ra[4];
            ra[0] = fp4_pair_bf16(wq[u][0],        s0lo);
            ra[1] = fp4_pair_bf16(wq[u][0] >>  8,  s0hi);
            ra[2] = fp4_pair_bf16(wq[u][0] >> 16,  s0lo);
            ra[3] = fp4_pair_bf16(wq[u][0] >> 24,  s0hi);
            const int k0 = (p0 + MMA_NW * u) << 5;
            const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
            uint32_t r0[2] = { Xl[t], Xl[t + 4] };
            mma_m16n8k16(acc[0], ra, r0);                            // block 2p,   columns 0..7
            const float s1lo = e4m3_f(sctv[u][g + 16]), s1hi = e4m3_f(sctv[u][g + 24]);
            ra[0] = fp4_pair_bf16(wq[u][1],        s1lo);
            ra[1] = fp4_pair_bf16(wq[u][1] >>  8,  s1hi);
            ra[2] = fp4_pair_bf16(wq[u][1] >> 16,  s1lo);
            ra[3] = fp4_pair_bf16(wq[u][1] >> 24,  s1hi);
            uint32_t r2[2] = { Xl[t + 8], Xl[t + 12] };
            mma_m16n8k16(acc[0], ra, r2);                            // block 2p+1, columns 0..7
        }
    }
    // Tail (npair not a multiple of 32): the leftover pairs in the SAME ascending order.
    for (int p = p0; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_x2 — the x2 structure (TWO 16-row weight tiles per CTA share the X
// fragments and interleave two INDEPENDENT K-chains in one loop body) applied to the fold kernel.
// Per-tile k-chains are UNTOUCHED (each tile's pairs in the same ascending order, epilogue reduce
// in the same warp order) — only which block computes which mt PAIR changes. M % 32 == 0 required
// (hy3: gate_up M=3072, down M=4096 — both even). The band (sharded-GU) mapping is computed per
// tile. launch_bounds(256,4) mirrors the x2 probe kernel.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_fold_x2(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt0 = blockIdx.x << 1, mt1 = mt0 + 1, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g0, mt_g1;
    bool band0 = false, band1 = false;
    int m_shift0 = 0, m_shift1 = 0, mt_s0 = mt0, mt_s1 = mt1, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) {
                Cb[mt0 * 16 + threadIdx.x] = f2b(0.f);
                Cb[mt1 * 16 + threadIdx.x] = f2b(0.f);
            }
            return;
        }
        mt_g0 = (long long)e * (M >> 4) + mt0;
        mt_g1 = mt_g0 + 1;                                           // adjacent tile of the same expert
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cb0 = mt0 << 4;
            if (cb0 < sl + rank_off) { mt_s0 = (cb0 - rank_off) >> 4;  m_shift0 = -rank_off;      band0 = true; }
            else                     { mt_s0 = (cb0 - sl - rank_off) >> 4; m_shift0 = -(sl + rank_off); band0 = true; }
            const int cb1 = mt1 << 4;
            if (cb1 < sl + rank_off) { mt_s1 = (cb1 - rank_off) >> 4;  m_shift1 = -rank_off;      band1 = true; }
            else                     { mt_s1 = (cb1 - sl - rank_off) >> 4; m_shift1 = -(sl + rank_off); band1 = true; }
        } else { mt_s0 = mt0; mt_s1 = mt1; }
        mt_g0 = mt_s0; mt_g1 = mt_s1;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mt_g0 * nblk + (p << 1);
        const long long tileB = mt_g1 * nblk + (p << 1);
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = sct + tileA * 16;
        const uint8_t* sctB = sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(accA[0], raA, rb0);                             // A block 2p,   columns 0..7
        mma_m16n8k16(accB[0], raB, rb0);                             // B block 2p,   columns 0..7
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(accA[0], raA, rb2);                             // A block 2p+1, columns 0..7
        mma_m16n8k16(accB[0], raB, rb2);                             // B block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band0) {
        const float v = mma_warp_reduce(sh, accA);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt0 * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift0;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s0]);
    } else {
        mma_epilogue(sh, accA, Cb, nullptr, gs_w, mt_s0, M, 1);
    }
    __syncthreads();
    if (band1) {
        const float v = mma_warp_reduce(sh, accB);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt1 * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift1;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s1]);
    } else {
        mma_epilogue(sh, accB, Cb, nullptr, gs_w, mt_s1, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_rast — the rasterization candidate: the SAME fold kernel with the
// grid axes swapped (blockIdx.x = bslot, blockIdx.y = mt; grid = (bkf, M/16)). Pure scheduling —
// the block→tile map changes which (mt, slot) blocks are co-resident, the per-element math is
// untouched.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold_rast(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int bslot = blockIdx.x, mt = blockIdx.y;                   // AXES SWAPPED vs the fold
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {                                  // remote expert: contribute 0
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- occupancy twins of the plain fold (launch_bounds 256,5 and 256,4): more registers per
// thread for the scheduler at lower co-resident-block counts — the launch_bounds knob the E11
// "safe cluster/PDL/occupancy" candidate list wants measured at Hy3's shapes. Identical body.
extern "C" __global__ __launch_bounds__(256, 5) void gemm_moe_mma_fp4_fold_lb5(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_mma_fp4_fold_lb4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- gemm_moe_mma_fp4_fold_pdl — the E9 PDL (programmatic dependent launch) twin of the fold
// kernel. pdl==1: publish the launch-completion edge, stream THIS BLOCK's weight tiles into L2
// (read-only — the loads are the same tile range the k-loop performs; kept alive by a never-true
// sink store), then griddepcontrol.wait gates the X reads. The weight SELECTION (ids for routed
// slots, sdesc for the shared slot) is final by the time this kernel launches (the up GEMM and
// silu — which consumed ids — run between the router and the down GEMM; the shared slot's
// descriptor is a kernel argument), so the prologue is dependency-safe: it must only wait before
// touching X. pdl==0 skips the preamble — byte-identical to the plain fold. This is the E11
// "safe PDL" candidate: the down GEMM's weight stream overlaps the silu (+ the router tail), and
// the shared slot's blocks overlap the routed up GEMM. Numerics are untouched (the sink store is
// never executed; the loads are the same values the compute reads).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_mma_fp4_fold_pdl(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids,
    int M, int K, long long kx, int expert_base, int e_span,
    const long long* __restrict__ sdesc, int pdl)
{
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int Kslots = (int)(kx >> 32);
    const int x_by_slot = (int)(kx & 0xffffffffu);
    const int b = bslot / Kslots, s = bslot - b * Kslots;
    __nv_bfloat16* Cb = C + (long long)bslot * M;

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    const int xrow = x_by_slot ? bslot : b;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (s < Kslots - 1) {
        const int e = ids[b * (Kslots - 1) + s] - expert_base;
        if (e < 0 || e >= e_span) {
            if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
            return;
        }
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;

    if (pdl) {
        // E9 prologue: publish, prefetch this block's tile, wait. The prefetch issues exactly the
        // weight/scale loads the k-loop performs (same tile, same pairs, same lanes) — the compute
        // then hits L2. The shared slot's descriptor is a kernel argument (constant); the routed
        // slot's ids were consumed by the up GEMM before this down GEMM launched, so both are safe
        // to read before the wait. The never-true store keeps the loads alive.
        asm volatile("griddepcontrol.launch_dependents;");
        const uint32_t* Wt32p = reinterpret_cast<const uint32_t*>(wt);
        unsigned sink = 0u;
        for (int p = warp; p < npair; p += MMA_NW) {
            const long long tile = mt_g * nblk + (p << 1);
            sink ^= Wt32p[tile * 32 + lane];
            sink ^= Wt32p[tile * 32 + 32 + lane];
            sink ^= (unsigned)sct[tile * 16 + g] << 8;
            sink ^= (unsigned)sct[tile * 16 + g + 8] << 16;
        }
        if (sink == 0xDEADBEEFu) C[threadIdx.x] = f2b(0.f);   // never true — keeps the loads alive
        asm volatile("griddepcontrol.wait;");
    }

    const __nv_bfloat16* Xtok = X + (long long)xrow * K + sk_off;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(Xtok + k0);
        uint32_t ra[4];
        ra[0] = fp4_pair_bf16(wq0,        s0lo);
        ra[1] = fp4_pair_bf16(wq0 >>  8,  s0hi);
        ra[2] = fp4_pair_bf16(wq0 >> 16,  s0lo);
        ra[3] = fp4_pair_bf16(wq0 >> 24,  s0hi);
        uint32_t rb0[2] = { Xl[t], Xl[t + 4] };
        mma_m16n8k16(acc[0], ra, rb0);                                // block 2p, columns 0..7
        ra[0] = fp4_pair_bf16(wq1,        s1lo);
        ra[1] = fp4_pair_bf16(wq1 >>  8,  s1hi);
        ra[2] = fp4_pair_bf16(wq1 >> 16,  s1lo);
        ra[3] = fp4_pair_bf16(wq1 >> 24,  s1hi);
        uint32_t rb2[2] = { Xl[t + 8], Xl[t + 12] };
        mma_m16n8k16(acc[0], ra, rb2);                                // block 2p+1, columns 0..7
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, 1);
    }
}

// ---- E11 probe fill kernel: writes a deterministic pattern over `n` uint32s (probe-only — stands
// in for the silu/router as the PDL predecessor so the overlap mechanism can be measured in
// isolation). Content-irrelevant; the fold GEMM reads X AFTER griddepcontrol.wait in the pdl arm.
extern "C" __global__ void e11_fill_x_b(unsigned int* X, int n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    X[idx] = (unsigned int)(idx * 2654435761u);
}

// The grouped-path fold GEMM: gemm_moe_grouped_mma_fp4 with the early-exit bound poff[nef]
// (nef = ne+1 — the routed band PLUS the shared region) and the descriptor-driven shared region
// [poff[nef-1], poff[nef]). Routed groups are byte-identical to gemm_moe_grouped_mma_fp4 (same
// weight-reuse structure, same per-element k-chain); a shared-region group computes the shared
// weight from the descriptor — the GU writes the rank's bands (C rows per the paired layout), the
// down writes its full h rows reading the rank's act half at the k-offset (bit-identical to the
// old act_half slice feeding the down GEMM).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_grouped_mma_fp4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int nef,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[nef]) return;   // the fold bound = poff[ne+1] (routed + shared total)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, N = 16;
    const bool is_shared = (nt * 16 >= poff[nef - 1]);

    const uint8_t* wt = Wt; const uint8_t* sct = Sct;
    const float* gs_w = gs;
    int kk = K, sk_off = 0;
    long long mt_g;
    bool band = false; int m_shift = 0, mt_s = mt, m_local = M;

    if (!is_shared) {
        const int e = tile_e[nt * 2] - expert_base;
        mt_g = (long long)e * (M >> 4) + mt;
        gs_w = gs + (long long)e * (M >> 4);
    } else {
        const long long sk = sdesc[3];
        kk = (int)(sk >> 32);
        sk_off = (int)(sk & 0xffffffffu);
        m_local = (int)sdesc[8];
        wt = (const uint8_t*)(uintptr_t)sdesc[0];
        sct = (const uint8_t*)(uintptr_t)sdesc[1];
        gs_w = (const float*)(uintptr_t)sdesc[2];
        if (m_local != M) {
            const int sl = m_local >> 1;
            const int rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { mt_s = (cbase - rank_off) >> 4;  m_shift = -rank_off;      band = true; }
            else                       { mt_s = (cbase - sl - rank_off) >> 4; m_shift = -(sl + rank_off); band = true; }
        } else mt_s = mt;
        mt_g = mt_s;
    }

    const int nblk = kk >> 4, npair = nblk >> 1;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 16) * K + sk_off;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    __nv_bfloat16* Cb = C + (long long)(nt * 16) * M;
    float acc[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(wt);
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = mt_g * nblk + (p << 1);
        const uint32_t wq0 = Wt32[tile * 32 + lane];
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];
        const uint8_t* sctp = sct + tile * 16;
        const float s0lo = e4m3_f(sctp[g]),      s0hi = e4m3_f(sctp[g + 8]);
        const float s1lo = e4m3_f(sctp[g + 16]), s1hi = e4m3_f(sctp[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        uint32_t ra[4];
        ra[0]=fp4_pair_bf16(wq0,s0lo); ra[1]=fp4_pair_bf16(wq0>>8,s0hi); ra[2]=fp4_pair_bf16(wq0>>16,s0lo); ra[3]=fp4_pair_bf16(wq0>>24,s0hi);
        uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
        mma_m16n8k16(acc[0], ra, rb0); mma_m16n8k16(acc[1], ra, rb1);
        ra[0]=fp4_pair_bf16(wq1,s1lo); ra[1]=fp4_pair_bf16(wq1>>8,s1hi); ra[2]=fp4_pair_bf16(wq1>>16,s1lo); ra[3]=fp4_pair_bf16(wq1>>24,s1hi);
        uint32_t rb2[2]={Xl[t+8],Xl[t+12]}, rb3[2]={Xh[t+8],Xh[t+12]};
        mma_m16n8k16(acc[0], ra, rb2); mma_m16n8k16(acc[1], ra, rb3);
    }
    __shared__ float sh[MMA_SMEM];
    if (band) {
        const float v = mma_warp_reduce(sh, acc);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, tt = rlane & 3, sub = rslot >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + m_shift;
        if (mw >= 0 && mw < m_local) Cb[m] = f2b(v * gs_w[mt_s]);
    } else {
        // gs_w is the per-expert base; mt_g already carries the expert offset (weight reads) and
        // must not be re-added — the epilogue indexes gs_w[mt_s] itself (see the slot-fold kernel).
        mma_epilogue(sh, acc, Cb, nullptr, gs_w, mt_s, M, N);
    }
}

// ---- gemm_moe_grouped_mma_fp4_x4 — the LOST 32-token wide grouped kernel (the E23 wide prefill
// path's default). ONE block per 32-token group; the block treats the group as TWO 16-token halves
// (moe_offsets_b pads experts to 16, not 32, so a 32-token group may straddle two experts) and
// resolves each half's expert independently — when the halves share an expert, the weight loads hit
// the same addresses (L2-cached), halving the expert-weight DRAM re-reads vs the 16-token kernel;
// the X fragments are shared by all four 8-token n-tiles. Per-element contract is identical to
// gemm_moe_grouped_mma_fp4: same pair partition (p = warp + 8k ascending), same per-pair block
// order (2p then 2p+1), same per-element fp32 accumulation, same fixed-order warp epilogue per
// 16-token half — only which block covers which group and the instruction scheduling change.
// grid = (M/16, ppad/32); the early-exit bound is poff[ne] read ON DEVICE (E16 device-only form).
// A half whose rows fall past the real padded total (the last group's tail) clamps its expert to
// the other half's — its outputs are padding rows (never combined), and the read stays in-bounds.
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int ne)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 32 >= poff[ne]) return;   // E16: static-grid upper bound; real tile count is device-side
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4, N = 32;
    // Per-half experts (8-tile index nt*4 + 2*h). The last group's tail half clamps to the first.
    const int ntiles = poff[ne] >> 3;
    const int e0 = tile_e[nt * 4] - expert_base;
    const int e1 = (nt * 4 + 2 < ntiles) ? (tile_e[nt * 4 + 2] - expert_base) : e0;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 32) * K;  // this group's 32 tokens
    const long long mtA = (long long)e0 * (M >> 4) + mt;
    const long long mtB = (long long)e1 * (M >> 4) + mt;
    __nv_bfloat16* Cb = C + (long long)(nt * 32) * M;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tileA = mtA * nblk + (p << 1);
        const long long tileB = mtB * nblk + (p << 1);
        const uint32_t wA0 = Wt32[tileA * 32 + lane];
        const uint32_t wA1 = Wt32[tileA * 32 + 32 + lane];
        const uint32_t wB0 = Wt32[tileB * 32 + lane];
        const uint32_t wB1 = Wt32[tileB * 32 + 32 + lane];
        const uint8_t* sctA = Sct + tileA * 16;
        const uint8_t* sctB = Sct + tileB * 16;
        const float a0lo = e4m3_f(sctA[g]),      a0hi = e4m3_f(sctA[g + 8]);
        const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
        const float b0lo = e4m3_f(sctB[g]),      b0hi = e4m3_f(sctB[g + 8]);
        const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + k0);
        const uint32_t* Xl2 = reinterpret_cast<const uint32_t*>(X + xr0 + 16 * K + k0);
        const uint32_t* Xh2 = reinterpret_cast<const uint32_t*>(X + xr1 + 16 * K + k0);
        uint32_t raA[4], raB[4];
        raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
        raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
        uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
        uint32_t rb2[2]={Xl2[t],Xl2[t+4]}, rb3[2]={Xh2[t],Xh2[t+4]};
        mma_m16n8k16(accA[0], raA, rb0); mma_m16n8k16(accA[1], raA, rb1); // half 0, block 2p
        mma_m16n8k16(accB[0], raB, rb2); mma_m16n8k16(accB[1], raB, rb3); // half 1, block 2p
        raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
        raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
        uint32_t rb4[2]={Xl[t+8],Xl[t+12]}, rb5[2]={Xh[t+8],Xh[t+12]};
        uint32_t rb6[2]={Xl2[t+8],Xl2[t+12]}, rb7[2]={Xh2[t+8],Xh2[t+12]};
        mma_m16n8k16(accA[0], raA, rb4); mma_m16n8k16(accA[1], raA, rb5); // half 0, block 2p+1
        mma_m16n8k16(accB[0], raB, rb6); mma_m16n8k16(accB[1], raB, rb7); // half 1, block 2p+1
    }
    __shared__ float sh[MMA_SMEM];
    const float* gsA = gs + (long long)e0 * (M >> 4);
    const float* gsB = gs + (long long)e1 * (M >> 4);
    mma_epilogue(sh, accA, Cb, nullptr, gsA, mt, M, 16);
    __syncthreads();                                                    // sh read->next-write hazard
    mma_epilogue(sh, accB, Cb + (long long)16 * M, nullptr, gsB, mt, M, 16);
}

// ---- gemm_moe_grouped_mma_fp4_x4_fold — the wide-prefill fold twin of the x4 kernel: 32-token
// groups, the fold bound poff[nef] = poff[ne+1], and the descriptor-driven shared region. Each
// 16-token half is resolved independently: routed iff its base < poff[nef-1] (= poff[ne], 16-
// aligned so a half never straddles the routed/shared boundary); the shared half reads the shared
// weight from the descriptor (kk/sk_off at desc[3], M at desc[8]) with the band/identity C-row
// mapping — the GU writes the rank's bands, the down writes its full h rows reading the rank's act
// half at the k-offset. A tail half past the real total (never in the fold: the shared region is
// pad32(ns) and the routed band 16-aligned, but the group-level exit covers poff[nef] mod 32)
// clamps its expert to the other half's; its rows are padding (never combined).
extern "C" __global__ __launch_bounds__(256, 4) void gemm_moe_grouped_mma_fp4_x4_fold(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base, const int* __restrict__ poff, int nef,
    const long long* __restrict__ sdesc)
{
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 32 >= poff[nef]) return;   // the fold bound = poff[ne+1]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, N = 32;
    const int shared_start = poff[nef - 1];
    const int ntiles = poff[nef] >> 3;

    // Per-half geometry: {weight ptrs, gs base, kk, sk_off, mt_g, band, m_shift, mt_s, m_local}.
    const uint8_t* wt0 = Wt; const uint8_t* sct0 = Sct; const float* gs0 = gs;
    const uint8_t* wt1 = Wt; const uint8_t* sct1 = Sct; const float* gs1 = gs;
    int kk0 = K, kk1 = K, so0 = 0, so1 = 0;
    long long mg0, mg1;
    bool band0 = false, band1 = false;
    int ms0 = mt, ms1 = mt, msh0 = 0, msh1 = 0, ml0 = M, ml1 = M;

    const bool sh0 = (nt * 32     >= shared_start);
    const bool sh1 = (nt * 32 + 16 >= shared_start);
    // Route each half: shared -> descriptor; routed -> tile_e (clamp a tail half to the other's).
    if (sh0) {
        const long long sk = sdesc[3];
        kk0 = (int)(sk >> 32); so0 = (int)(sk & 0xffffffffu);
        ml0 = (int)sdesc[8];
        wt0 = (const uint8_t*)(uintptr_t)sdesc[0];
        sct0 = (const uint8_t*)(uintptr_t)sdesc[1];
        gs0 = (const float*)(uintptr_t)sdesc[2];
        if (ml0 != M) {
            const int sl = ml0 >> 1, rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { ms0 = (cbase - rank_off) >> 4; msh0 = -rank_off; band0 = true; }
            else                       { ms0 = (cbase - sl - rank_off) >> 4; msh0 = -(sl + rank_off); band0 = true; }
        } else ms0 = mt;
        mg0 = ms0;
    } else {
        const int e = (nt * 4 < ntiles) ? (tile_e[nt * 4] - expert_base) : 0;
        mg0 = (long long)e * (M >> 4) + mt;
        gs0 = gs + (long long)e * (M >> 4);
    }
    if (sh1) {
        const long long sk = sdesc[3];
        kk1 = (int)(sk >> 32); so1 = (int)(sk & 0xffffffffu);
        ml1 = (int)sdesc[8];
        wt1 = (const uint8_t*)(uintptr_t)sdesc[0];
        sct1 = (const uint8_t*)(uintptr_t)sdesc[1];
        gs1 = (const float*)(uintptr_t)sdesc[2];
        if (ml1 != M) {
            const int sl = ml1 >> 1, rank_off = (int)(sdesc[7] & 0xffffffffu);
            const int cbase = mt << 4;
            if (cbase < sl + rank_off) { ms1 = (cbase - rank_off) >> 4; msh1 = -rank_off; band1 = true; }
            else                       { ms1 = (cbase - sl - rank_off) >> 4; msh1 = -(sl + rank_off); band1 = true; }
        } else ms1 = mt;
        mg1 = ms1;
    } else {
        const int e = (nt * 4 + 2 < ntiles) ? (tile_e[nt * 4 + 2] - expert_base) : (tile_e[nt * 4] - expert_base);
        mg1 = (long long)e * (M >> 4) + mt;
        gs1 = gs + (long long)e * (M >> 4);
    }

    const int nb0 = kk0 >> 4, nb1 = kk1 >> 4, np0 = nb0 >> 1, np1 = nb1 >> 1;
    const __nv_bfloat16* X = Xperm + (long long)(nt * 32) * K;
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;
    __nv_bfloat16* Cb = C + (long long)(nt * 32) * M;
    float accA[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    float accB[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    const uint32_t* W0 = reinterpret_cast<const uint32_t*>(wt0);
    const uint32_t* W1 = reinterpret_cast<const uint32_t*>(wt1);
    // The k-loop: both halves take the same pairs p = warp + 8k ascending; half 0 reads its own
    // weight, half 1 its own (the same addresses when both are routed with the same expert).
    const int npair = max(np0, np1);
    for (int p = warp; p < npair; p += MMA_NW) {
        const int k0 = (p << 5);
        const uint32_t* Xl = reinterpret_cast<const uint32_t*>(X + xr0 + so0 + k0);
        const uint32_t* Xh = reinterpret_cast<const uint32_t*>(X + xr1 + so0 + k0);
        const uint32_t* Xl2 = reinterpret_cast<const uint32_t*>(X + xr0 + so1 + 16 * K + k0);
        const uint32_t* Xh2 = reinterpret_cast<const uint32_t*>(X + xr1 + so1 + 16 * K + k0);
        if (p < np0) {
            const long long tileA = mg0 * nb0 + (p << 1);
            const uint32_t wA0 = W0[tileA * 32 + lane];
            const uint32_t wA1 = W0[tileA * 32 + 32 + lane];
            const uint8_t* sctA = sct0 + tileA * 16;
            const float a0lo = e4m3_f(sctA[g]), a0hi = e4m3_f(sctA[g + 8]);
            const float a1lo = e4m3_f(sctA[g + 16]), a1hi = e4m3_f(sctA[g + 24]);
            uint32_t raA[4];
            raA[0]=fp4_pair_bf16(wA0,a0lo); raA[1]=fp4_pair_bf16(wA0>>8,a0hi); raA[2]=fp4_pair_bf16(wA0>>16,a0lo); raA[3]=fp4_pair_bf16(wA0>>24,a0hi);
            uint32_t rb0[2]={Xl[t],Xl[t+4]}, rb1[2]={Xh[t],Xh[t+4]};
            mma_m16n8k16(accA[0], raA, rb0); mma_m16n8k16(accA[1], raA, rb1);
            raA[0]=fp4_pair_bf16(wA1,a1lo); raA[1]=fp4_pair_bf16(wA1>>8,a1hi); raA[2]=fp4_pair_bf16(wA1>>16,a1lo); raA[3]=fp4_pair_bf16(wA1>>24,a1hi);
            uint32_t rb4[2]={Xl[t+8],Xl[t+12]}, rb5[2]={Xh[t+8],Xh[t+12]};
            mma_m16n8k16(accA[0], raA, rb4); mma_m16n8k16(accA[1], raA, rb5);
        }
        if (p < np1) {
            const long long tileB = mg1 * nb1 + (p << 1);
            const uint32_t wB0 = W1[tileB * 32 + lane];
            const uint32_t wB1 = W1[tileB * 32 + 32 + lane];
            const uint8_t* sctB = sct1 + tileB * 16;
            const float b0lo = e4m3_f(sctB[g]), b0hi = e4m3_f(sctB[g + 8]);
            const float b1lo = e4m3_f(sctB[g + 16]), b1hi = e4m3_f(sctB[g + 24]);
            uint32_t raB[4];
            raB[0]=fp4_pair_bf16(wB0,b0lo); raB[1]=fp4_pair_bf16(wB0>>8,b0hi); raB[2]=fp4_pair_bf16(wB0>>16,b0lo); raB[3]=fp4_pair_bf16(wB0>>24,b0hi);
            uint32_t rb2[2]={Xl2[t],Xl2[t+4]}, rb3[2]={Xh2[t],Xh2[t+4]};
            mma_m16n8k16(accB[0], raB, rb2); mma_m16n8k16(accB[1], raB, rb3);
            raB[0]=fp4_pair_bf16(wB1,b1lo); raB[1]=fp4_pair_bf16(wB1>>8,b1hi); raB[2]=fp4_pair_bf16(wB1>>16,b1lo); raB[3]=fp4_pair_bf16(wB1>>24,b1hi);
            uint32_t rb6[2]={Xl2[t+8],Xl2[t+12]}, rb7[2]={Xh2[t+8],Xh2[t+12]};
            mma_m16n8k16(accB[0], raB, rb6); mma_m16n8k16(accB[1], raB, rb7);
        }
    }
    __shared__ float sh[MMA_SMEM];
    // Epilogue per 16-token half: band mapping (sharded GU) or the identity mma_epilogue. gs0/gs1
    // are the per-expert bases (mg0/mg1 already carry the expert offset for the weight reads and
    // must not be re-added — the epilogue indexes gs[mt] itself).
    const float* gse0 = gs0;
    const float* gse1 = gs1;
    if (band0) {
        const float v = mma_warp_reduce(sh, accA);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + msh0;
        if (mw >= 0 && mw < ml0) Cb[m] = f2b(v * gs0[ms0]);
    } else {
        mma_epilogue(sh, accA, Cb, nullptr, gse0, mt, M, 16);
    }
    __syncthreads();                                                    // sh read->next-write hazard
    __nv_bfloat16* Cb1 = Cb + (long long)16 * M;
    if (band1) {
        const float v = mma_warp_reduce(sh, accB);
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
        const int gg = rlane >> 2, i = rslot & 3;
        const int m = mt * 16 + gg + ((i >= 2) ? 8 : 0);
        const int mw = m + msh1;
        if (mw >= 0 && mw < ml1) Cb1[m] = f2b(v * gs1[ms1]);
    } else {
        mma_epilogue(sh, accB, Cb1, nullptr, gse1, mt, M, 16);
    }
}

// ---- E0: fused GDN slot restore (rollback). Replaces the host loop in GpuModel::copy_gdn_slot that
// issued 2*n_gdn cudaMemcpyDtoDAsync calls per rollback (48 layers x {conv_state, s_state} on the
// 27B = 96 driver calls, ~25% of the rollback span idle in issue gaps — E1's nsys inter-graph
// budget). `table` is [n_gdn] conv_state base pointers followed by [n_gdn] s_state base pointers,
// both rank-local (TP-aware strides are the caller's, exactly as the old per-layer dtod math).
// gridDim.y partitions each layer's flat byte range [0, cb+sb) so the copy spreads over ~4 blocks
// per layer (192 blocks at n_gdn=48 — the one-block-per-layer shape leaves the GPU ~17% occupied
// and cannot hold the 238 GB/s roofline). PURE BYTE COPY: no arithmetic, so it is bit-identical to
// the dtod memcpys by construction. 16B vector path when both strides are 16B multiples (every
// current model: conv 40,960 B, s 786,432 B at rank-local 12 v-heads), u32 path otherwise (f32
// tensors are always 4B aligned).
extern "C" __global__ void gdn_rollback_b(const unsigned long long* __restrict__ table,
                                          int n_gdn, unsigned long long cb, unsigned long long sb,
                                          int src, int dst)
{
    const int li = blockIdx.x;
    if (li >= n_gdn) return;
    const unsigned char* conv_src = (const unsigned char*)table[li]            + (unsigned long long)src * cb;
    unsigned char*       conv_dst = (unsigned char*)table[li]            + (unsigned long long)dst * cb;
    const unsigned char* s_src    = (const unsigned char*)table[n_gdn + li] + (unsigned long long)src * sb;
    unsigned char*       s_dst    = (unsigned char*)table[n_gdn + li] + (unsigned long long)dst * sb;
    // Independent per-region partitions across gridDim.y — a single flat split could span the
    // conv/s_state boundary and copy neighbor-slot bytes; two ranges cannot. Partition widths
    // round up to 16B so the uint4 path's addresses stay aligned (both strides are 16B multiples
    // in that branch, hence every boundary too).
    unsigned long long cc = (cb + gridDim.y - 1) / gridDim.y;
    unsigned long long cs = (sb + gridDim.y - 1) / gridDim.y;
    cc = (cc + 15ULL) & ~15ULL;
    cs = (cs + 15ULL) & ~15ULL;
    const unsigned long long clo = (unsigned long long)blockIdx.y * cc;
    const unsigned long long slo = (unsigned long long)blockIdx.y * cs;
    const long long cn = (clo < cb) ? (long long)min(cb - clo, cc) : 0LL;
    const long long sn = (slo < sb) ? (long long)min(sb - slo, cs) : 0LL;
    const bool vec = (cb % 16 == 0) && (sb % 16 == 0);
    #define GDN_CPY(T, sp, dp, n) do {                                     \
        T const* s_ = (T const*)((sp)); T* d_ = (T*)(dp);                  \
        for (long long i = threadIdx.x; i < (n); i += blockDim.x) d_[i] = s_[i]; \
    } while (0)
    if (cn > 0) {
        if (vec) GDN_CPY(uint4,  conv_src + clo, conv_dst + clo, cn >> 4);
        else     GDN_CPY(unsigned int, conv_src + clo, conv_dst + clo, cn >> 2);
    }
    if (sn > 0) {
        if (vec) GDN_CPY(uint4,  s_src + slo, s_dst + slo, sn >> 4);
        else     GDN_CPY(unsigned int, s_src + slo, s_dst + slo, sn >> 2);
    }
    #undef GDN_CPY
}

// =====================================================================================================
// ===================== qwen4_exp (Qwen3.8-Flash-Next): hyper-connections + PLE =====================
// =====================================================================================================
//
// The residual stream is `hc` copies of the hidden ("hyper-connections", rw = hc*h): every sublayer
// reads a gated MIX of the streams and writes back a per-stream-weighted INJECTION of its output.
// Reference: Qwen4ExpTextGatedResidual (modeling_qwen4_exp.py). Layout stays column-major [feat, B].
// The PLE layer (Qwen4ExpTextPLELayer) adds hashed n-gram embeddings into every stream once, at one
// trunk layer. All norms here are the qwen (1+w) RMSNorm, GROUPED per stream where noted.

// out[s*h+i, b] = in[i, b] for every stream s (the embedding replicated into the hc streams).
extern "C" __global__ void hc_expand_b(__nv_bfloat16* out, const __nv_bfloat16* in, int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int r = (int)(idx % rw);
    out[idx] = in[(long long)b * h + (r % h)];
}

// Grouped RMSNorm: one block per (b, s); rmsnorm over the s-th group of h, times (1 + w[s*h+i]).
extern "C" __global__ void hc_norm_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w,
                                     int h, int hc, int B, float eps) {
    int blk = blockIdx.x;
    if (blk >= B * hc) return;
    int b = blk / hc, s = blk % hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long off = (long long)b * ((long long)h * hc) + (long long)s * h;
    float sum_sq = 0.0f;
    for (int i = tid; i < h; i += bs) { float v = b2f(x[off + i]); sum_sq += v * v; }
    sm[tid] = sum_sq;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(sm[0] / (float)h + eps);
    for (int i = tid; i < h; i += bs) {
        float v = b2f(x[off + i]);
        out[off + i] = f2b(v * inv * (1.0f + w[s * h + i]));
    }
}

// x = silu(x / div), in place (the hc lowrank activation: silu(down(hn) / hc)).
extern "C" __global__ void silu_div_b(__nv_bfloat16* x, float div, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = f2b(silu_f(b2f(x[idx]) / div));
}

// The stream MIX + the injection weights, one block per column b:
//   x[i, b]   = (1/hc) * sum_s sigmoid(u[s*h+i, b]) * hn[s*h+i, b]
//   inj[s, b] = 2 * sigmoid( (sum_c winj[s, c] * hn[c, b]) / hc )     (winj bf16 [hc, rw]; skipped if null)
extern "C" __global__ void hc_mix_b(__nv_bfloat16* x, float* inj, const __nv_bfloat16* hn, const __nv_bfloat16* u,
                                    const __nv_bfloat16* winj, int h, int hc, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long rw = (long long)h * hc;
    const __nv_bfloat16* hb = hn + (long long)b * rw;
    const __nv_bfloat16* ub = u + (long long)b * rw;
    for (int i = tid; i < h; i += bs) {
        float acc = 0.0f;
        for (int s = 0; s < hc; s++) {
            long long o = (long long)s * h + i;
            float g = 1.0f / (1.0f + __expf(-b2f(ub[o])));
            acc += g * b2f(hb[o]);
        }
        x[(long long)b * h + i] = f2b(acc / (float)hc);
    }
    if (winj == nullptr || inj == nullptr) return;
    // injection logits: hc dot products over rw, block-reduced.
    for (int s = 0; s < hc; s++) {
        const __nv_bfloat16* ws = winj + (long long)s * rw;
        float part = 0.0f;
        for (long long c = tid; c < rw; c += bs) part += b2f(ws[c]) * b2f(hb[c]);
        sm[tid] = part;
        __syncthreads();
        for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
        if (tid == 0) inj[(long long)b * hc + s] = 2.0f / (1.0f + __expf(-(sm[0] / (float)hc)));
        __syncthreads();
    }
}

// resid[s*h+i, b] += out[i, b] * inj[s, b]
extern "C" __global__ void hc_inject_b(__nv_bfloat16* resid, const __nv_bfloat16* out, const float* inj,
                                       int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int r = (int)(idx % rw);
    int s = r / h, i = r % h;
    float v = b2f(resid[idx]) + b2f(out[(long long)b * h + i]) * inj[(long long)b * hc + s];
    resid[idx] = f2b(v);
}

// GDN output norm with a SIGMOID gate (qwen4_exp `output_gate_type: sigmoid`); twin of rmsnorm_gated_b.
extern "C" __global__ void rmsnorm_gated_sig_b(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z,
                                               const float* w, int vd, int nh, int B, float eps, int z_off_z_stride) {
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    const int z_off = z_off_z_stride & 0x7FFF;
    const int z_stride = (unsigned)z_off_z_stride >> 15;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? b2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) {
        float g = b2f(z[z_off + (long long)b * z_stride + (long long)head * vd + tid]);
        out[base + tid] = f2b(v * inv * w[tid] * (1.0f / (1.0f + __expf(-g))));
    }
}

// ---- PLE: ancestry helpers ----
// Column t's k-th ancestor in a verify tree (`parent` packed: low 16 bits = DFS parent, 0xFFFF = root;
// null parent = chain t-1). Returns the column index, or -(m) where m >= 1 is how many steps
// remain past the root (those come from the slot's committed state / token ring).
// `chain`: with a null `parent`, 1 = columns form a chain (t-1), 0 = every column is its own root
// (a decode batch: independent lanes, each continuing from its own slot's state).
__device__ __forceinline__ int ple_ancestor(const int* parent, int chain, int t, int k) {
    int c = t;
    for (int i = 0; i < k; i++) {
        int p;
        if (parent) { int v = parent[c] & 0xFFFF; p = (v == 0xFFFF) ? -1 : v; }
        else p = chain ? (c - 1) : -1;
        if (p < 0) return -(k - i);
        c = p;
    }
    return c;
}
__device__ __forceinline__ int ple_slot_of(const int* slot_ids, int slot0, int t) { return slot_ids ? slot_ids[t] : slot0; }

// ---- PLE hash: row ids [heads, n] (row-major per column: ids[t*heads + j]) ----
// tab = [mult[ngram] | head_vocab[heads] | head_offset[heads]] (i64). ring = [slots, ngram-1] tokens
// oldest first (the committed history of the slot). EOS rule: the token p back is EOS if any token
// in the p positions before the current one is EOS (Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos).
// geom = ngram | (hpn << 8) | (heads << 16) | (chain << 24)   (cudarc launch tuples cap at 12 args)
extern "C" __global__ void ple_hash_b(long long* ids, const int* tokens, const int* ring, const int* slot_ids, int slot0,
                                      const int* parent, const long long* tab, int geom, int eos, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int ngram = geom & 0xFF, hpn = (geom >> 8) & 0xFF, heads = (geom >> 16) & 0xFF, chain = (geom >> 24) & 1;
    long long shifted[8];
    int slot = ple_slot_of(slot_ids, slot0, t);
    shifted[0] = tokens[t];
    bool blocked = false;
    for (int p = 1; p < ngram; p++) {
        int a = ple_ancestor(parent, chain, t, p);
        int tok;
        if (a >= 0) tok = tokens[a];
        else { int m = -a; tok = ring[(long long)slot * (ngram - 1) + (ngram - 1 - m)]; }   // m=1 → newest ring entry
        if (tok == eos) blocked = true;
        shifted[p] = blocked ? (long long)eos : (long long)tok;
    }
    const long long* mult = tab;
    const long long* hv = tab + ngram;
    const long long* ho = tab + ngram + heads;
    for (int ng = 2; ng <= ngram; ng++) {
        int start = (ng - 2) * hpn;
        long long mixed = shifted[0] * mult[0];
        for (int p = 1; p < ng; p++) mixed ^= shifted[p] * mult[p];
        for (int j = 0; j < hpn; j++) {
            int hh = start + j;
            long long r = mixed % hv[hh]; if (r < 0) r += hv[hh];
            ids[(long long)t * heads + hh] = r + ho[hh];
        }
    }
}

// Commit the token ring for slot(s): dst_ring[slot(dst)] = the last (ngram-1) tokens along column t's
// path (older ones from the source ring). tsel >= 0: only column tsel → dst slot dst_slot0; tsel < 0:
// every column t → dst slot dst_slot0 + t (per-column checkpoints). dst may alias src only when the
// written slot differs from every read slot (the caller guarantees it: main-slot commits go through
// a scratch slot).
// tsel >= 0: column tsel → dst slot dst_slot0. tsel == -1: column t → dst slot dst_slot0 + t (per-column
// checkpoints). tsel == -2: column t → ITS OWN slot (decode lanes, in place: read-all-then-write).
// One block per column, blockDim >= ngram-1.
extern "C" __global__ void ple_ring_commit_b(int* dst_ring, int dst_slot0, const int* src_ring, const int* tokens,
                                             const int* slot_ids, int slot0, const int* parent, int chain,
                                             int ngram, int n, int tsel) {
    int col = blockIdx.x;
    int t = (tsel >= 0) ? tsel : col;
    if (tsel < 0 && col >= n) return;
    if (tsel >= 0 && col > 0) return;
    int L = ngram - 1;
    int src_slot = ple_slot_of(slot_ids, slot0, t);
    int dst_slot = (tsel >= 0) ? dst_slot0 : (tsel == -1 ? dst_slot0 + t : src_slot);
    int l = threadIdx.x;
    int tok = 0;
    if (l < L) {
        int back = L - 1 - l;        // l = L-1 → the token itself (0 back)
        int a = ple_ancestor(parent, chain, t, back);
        tok = (a >= 0) ? tokens[a] : src_ring[(long long)src_slot * L + (L - (-a))];
    }
    __syncthreads();
    if (l < L) dst_ring[(long long)dst_slot * L + l] = tok;
}

// ---- PLE table records: gather (device-resident table) and dequant ----
// stage[r*96 ..] = table[ids[r]*96 ..], 24 u32 words per record.
extern "C" __global__ void ple_gather_rows_b(unsigned char* stage, const unsigned char* table, const long long* ids, int nrec) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nrec * 24) return;
    long long r = idx / 24; int w = (int)(idx % 24);
    const unsigned int* src = (const unsigned int*)(table + ids[r] * 96);
    unsigned int* dst = (unsigned int*)(stage + r * 96);
    dst[w] = src[w];
}

__device__ __forceinline__ float ple_e4m3(unsigned char b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0x0F;
    float man = (float)(b & 0x07);
    if (exp == 0) return sign * (man / 8.0f) * 0.015625f;          // 2^-6
    return sign * (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
}
__constant__ float PLE_E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// emb[(head*160 + blk*16 + i), col] for record r = col*heads + head; one thread per (record, 16-block).
// gs[shard] is the RECIPROCAL global scale of the record's source shard: w = e2m1 * e4m3 / gs.
extern "C" __global__ void ple_dequant_rows_b(__nv_bfloat16* emb, const unsigned char* stage, const long long* ids,
                                              const float* gs, int rows_per_shard, int heads, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nrec = (long long)n * heads;
    if (idx >= nrec * 10) return;
    long long r = idx / 10; int blk = (int)(idx % 10);
    int col = (int)(r / heads), head = (int)(r % heads);
    const unsigned char* rec = stage + r * 96;
    float st = 1.0f / gs[(int)(ids[r] / rows_per_shard)];
    float s = ple_e4m3(rec[80 + blk]) * st;
    __nv_bfloat16* o = emb + (long long)col * (heads * 160) + (long long)head * 160 + blk * 16;
    for (int i = 0; i < 16; i++) {
        int e = blk * 16 + i;
        unsigned char byte = rec[e >> 1];
        unsigned char code = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        float v = PLE_E2M1[code & 7];
        if (code & 8) v = -v;
        o[i] = f2b(v * s);
    }
}

// ---- PLE gate: gv[s*h+i, b] = sigmoid(gate[s,b]) * value[i, b] ----
//   gate = (kn[s] . qn[s]) / sqrt(h);  gate = sign(gate) * sqrt(max(|gate|, 1e-6))
// one block per (b, s).
extern "C" __global__ void ple_gate_b(__nv_bfloat16* gv, const __nv_bfloat16* kn, const __nv_bfloat16* qn,
                                      const __nv_bfloat16* value, int h, int hc, int B) {
    int blk = blockIdx.x;
    if (blk >= B * hc) return;
    int b = blk / hc, s = blk % hc;
    extern __shared__ float sm[];
    int tid = threadIdx.x, bs = blockDim.x;
    long long off = (long long)b * ((long long)h * hc) + (long long)s * h;
    float part = 0.0f;
    for (int i = tid; i < h; i += bs) part += b2f(kn[off + i]) * b2f(qn[off + i]);
    sm[tid] = part;
    __syncthreads();
    for (int s2 = bs / 2; s2 > 0; s2 >>= 1) { if (tid < s2) sm[tid] += sm[tid + s2]; __syncthreads(); }
    float g = sm[0] * rsqrtf((float)h);
    float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    g = (g < 0.0f) ? -mag : mag;
    float sg = 1.0f / (1.0f + __expf(-g));
    for (int i = tid; i < h; i += bs) gv[off + i] = f2b(sg * b2f(value[(long long)b * h + i]));
}

// ---- PLE dilated depthwise causal conv (kernel K, dilation dil, state length L = (K-1)*dil) ----
// The conv input is gvn (the norm_conv output); its output is silu(conv) ADDED to gv, and the sum is
// the PLE output ADDED to the residual: resid += gv + silu(conv(gvn)).
// state[slot][l][c], l = 0 oldest ... L-1 newest (the last L conv inputs of the slot).
// DECODE step (one token per column, its own slot): taps come from the state; then the state shifts
// and takes gvn as its newest row.
extern "C" __global__ void ple_dconv_decode_b(__nv_bfloat16* resid, const __nv_bfloat16* gv, const __nv_bfloat16* gvn,
                                              float* state, const float* w, const int* slot_ids,
                                              int rw, int L, int K, int dil, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * B) return;
    int b = (int)(idx / rw);
    int c = (int)(idx % rw);
    int slot = slot_ids[b];
    float* st = state + (long long)slot * L * rw;
    float x = b2f(gvn[idx]);
    float acc = w[c * K + (K - 1)] * x;
    for (int j = 0; j < K - 1; j++) {
        int back = (K - 1 - j) * dil;          // positions back
        acc += w[c * K + j] * st[(long long)(L - back) * rw + c];
    }
    for (int l = 1; l < L; l++) st[(long long)(l - 1) * rw + c] = st[(long long)l * rw + c];
    st[(long long)(L - 1) * rw + c] = x;
    float v = b2f(resid[idx]) + b2f(gv[idx]) + silu_f(acc);
    resid[idx] = f2b(v);
}

// PREFILL / VERIFY (n columns along a chain or tree): taps at `back` positions come from the
// ancestor columns, or from the slot's committed state past the root. Pure read of `state` — the
// commit is a separate launch (ple_dconv_state_b) so nothing races.
// geom = L | (K << 8) | (dil << 16) | (chain << 24)
extern "C" __global__ void ple_dconv_prefill_b(__nv_bfloat16* resid, const __nv_bfloat16* gv, const __nv_bfloat16* gvn,
                                               const float* state, const int* slot_ids, int slot0, const int* parent,
                                               const float* w, int rw, int geom, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * n) return;
    const int L = geom & 0xFF, K = (geom >> 8) & 0xFF, dil = (geom >> 16) & 0xFF, chain = (geom >> 24) & 1;
    int t = (int)(idx / rw);
    int c = (int)(idx % rw);
    int slot = ple_slot_of(slot_ids, slot0, t);
    const float* st = state + (long long)slot * L * rw;
    float acc = w[c * K + (K - 1)] * b2f(gvn[idx]);
    for (int j = 0; j < K - 1; j++) {
        int back = (K - 1 - j) * dil;
        int a = ple_ancestor(parent, chain, t, back);
        float xv = (a >= 0) ? b2f(gvn[(long long)a * rw + c]) : st[(long long)(L - (-a)) * rw + c];
        acc += w[c * K + j] * xv;
    }
    float v = b2f(resid[idx]) + b2f(gv[idx]) + silu_f(acc);
    resid[idx] = f2b(v);
}

// State after column t: rows l = 0..L-1 hold the conv input (L-1-l) positions back from t (ancestor
// columns, else the SOURCE slot's state). tsel >= 0: column tsel → dst slot dst_slot0 (blockIdx.y
// unused); tsel < 0: column t = blockIdx.y → dst slot dst_slot0 + t (per-column checkpoints).
// dst must not alias the source rows being read (main-slot commits go through a scratch slot).
extern "C" __global__ void ple_dconv_state_b(float* dst, int dst_slot0, const float* src, const int* slot_ids, int slot0,
                                             const int* parent, int chain, const __nv_bfloat16* gvn, int rw, int L, int n, int tsel) {
    int t = (tsel >= 0) ? tsel : (int)blockIdx.y;
    if (t >= n) return;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rw * L) return;
    int l = (int)(idx / rw);
    int c = (int)(idx % rw);
    int dst_slot = (tsel >= 0) ? dst_slot0 : dst_slot0 + t;
    int src_slot = ple_slot_of(slot_ids, slot0, t);
    int back = L - 1 - l;
    int a = ple_ancestor(parent, chain, t, back);
    float v = (a >= 0) ? b2f(gvn[(long long)a * rw + c]) : src[((long long)src_slot * L + (L - (-a))) * rw + c];
    dst[((long long)dst_slot * L + l) * rw + c] = v;
}

// Byte copy of one PLE state slot + ring entry (rollback / snapshot), same contract as gdn_rollback_b.
// dst_ids != null: the destination slot is dst_ids[dst_idx] (read on device — capture-legal).
extern "C" __global__ void ple_slot_copy_b(float* state, int* ring, unsigned long long state_floats, int ring_ints, int src, int dst,
                                           const int* dst_ids, int dst_idx) {
    if (dst_ids) dst = dst_ids[dst_idx];
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (long long)state_floats) {
        state[(long long)dst * state_floats + idx] = state[(long long)src * state_floats + idx];
    }
    if (idx < ring_ints) ring[(long long)dst * ring_ints + idx] = ring[(long long)src * ring_ints + idx];
}

// qwen4_exp MTP input fusion: streams[s*h+i, b] += x[i, b] (the fc_embedding term broadcast to every stream).
extern "C" __global__ void hc_add_bcast_b(__nv_bfloat16* streams, const __nv_bfloat16* x, int h, int hc, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long rw = (long long)h * hc;
    if (idx >= rw * B) return;
    int b = (int)(idx / rw);
    int i = (int)((idx % rw) % h);
    streams[idx] = f2b(b2f(streams[idx]) + b2f(x[(long long)b * h + i]));
}

// ===================== qwen4_exp QSA sparse-attention indexer (Qwen4ExpTextQSAIndexer) =====================
//
// Reference (modeling_qwen4_exp.py): per full-attention layer, index_qk_proj(hidden) → q (heads×hd,
// q_layernorm (1+w), RoPE on the first rdim dims at the query position) and one RAW key (hd, cached per
// token pre-norm). The query's visible ranks [0, pc) are cut into complete blocks of `ratio` consecutive
// ranks; a block's key = k_layernorm(mean of its raw keys) with RoPE at the block-start rank; the score
// is Σ_heads relu(q_h·k_j)/√hd; the top-min(budget/ratio, nblocks) blocks plus the tail ranks
// (pc mod ratio) are the ONLY keys the layer's attention sees. Below budget+ratio-1 visible tokens every
// block is selected (dense attention is exact there — the engine keeps its dense kernels for that case).
//
// Engine mapping. Raw keys live in a per-layer `[slot][stride][hd]` bf16 cache addressed EXACTLY like
// the KV cache (write column = pos[b], slot = slot_ids[b]); block keys are recomputed on the fly from
// raw keys (four 256-B reads per block), so there is no derived cache to invalidate on an MTP rollback,
// a prefix-cache clamp or a tree compaction — whatever holds for the KV cache holds here. Selection
// runs in the SAME rank space as gqa_attn_splitk (rank r < pos_start is prefix column r; rank r >=
// pos_start is on-block ancestor `path[...]`), so a verify column's selection is a pure function of
// (its q, the committed raw keys, its ancestors) — identical to the decode at that position, which
// is what the lossless-MTP contract needs. The top-k is a deterministic radix select (ties → lowest
// block index) and the selected ranks are emitted ascending, mapped to cache columns, so the attention
// kernels below are the dense kernels with one address indirection (an identity list is bit-identical
// to the dense kernel by construction).
//
// `QsaParams` (a 64-B device struct built once per indexer at load) carries what would not fit the
// 12-argument launch cap: the rope tables, the two norm weights, eps and the geometry.
struct QsaParams {
    const float* cos_t;     // [max_pos][rdim] (duplicated halves, see build_rope_tables)
    const float* sin_t;
    const float* kw;        // k_layernorm weight [hd] (gemma style: scale = 1 + w)
    const float* qw;        // q_layernorm weight [hd] (unused on device; the q norm runs rmsnorm_rope_b)
    float eps;
    int hd;                 // indexer head dim (128; 32 on the tiny model)
    int heads;              // indexer query heads (4)
    int rdim;               // rotary dims (64; 16 on the tiny model)
    int ratio;              // compress ratio (4)
    int topk;               // block budget = indexer_budget / ratio (512)
    int sel_max;            // indexer_budget + ratio - 1 (2051): row pitch of the selection lists
    int pad;
};
#define QSA_HD_MAX 128
#define QSA_HEADS_MAX 8
#define QSA_PF_QT 8            // queries per block in qsa_score_prefill_b

__device__ __forceinline__ float qsa_bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

// rank -> cache column, gqa_attn_splitk's rule (path == nullptr => identity).
__device__ __forceinline__ int qsa_col(int rank, int pos_start, const unsigned char* path, int b) {
    if (!path || rank < pos_start) return rank;
    return pos_start + (int)path[b * MAX_VERIFY + (rank - pos_start)];
}

// 1. raw key write. keys[(slot*stride + col)*hd + d] = qk[b*pitch + k_off + d].
//    pos == nullptr: col = pos_start + b and slot 0 (prefill chain into a slot base); else col = pos[b],
//    slot = slot_ids[b] (decode / verify, base pointer = slot 0).
extern "C" __global__ void qsa_key_write_b(__nv_bfloat16* keys, const __nv_bfloat16* qk, const int* pos,
                                           const int* slot_ids, int stride, int pos_start, int pitch, int k_off,
                                           int hd, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * hd) return;
    int b = (int)(idx / hd), d = (int)(idx % hd);
    int col = pos ? pos[b] : pos_start + b;
    int slot = (pos && slot_ids) ? slot_ids[b] : 0;
    keys[((long long)slot * stride + col) * hd + d] = qk[(long long)b * pitch + k_off + d];
}

// Block key (warp-cooperative): mean of the `ratio` raw keys at cache rows `rows[]` (f32 sum / ratio →
// bf16, HF's `.float().mean().to(bf16)`), k_layernorm (f32 from the bf16 mean, ×(1+w), → bf16), RoPE at
// rank `pos_rope` on dims [0, rdim) (f32, → bf16). Lane owns dims [lane*DPL, lane*DPL+DPL); `sm` is a
// per-warp hd-float scratch used for the rotate-half pairing.
__device__ __forceinline__ void qsa_block_key(float* v, const __nv_bfloat16* keys, const long long* rows,
                                              const QsaParams* p, int pos_rope, float* sm, int lane) {
    const int hd = p->hd, DPL = hd >> 5, rdim = p->rdim, half = rdim >> 1;
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = 0.0f;
    for (int r = 0; r < p->ratio; r++) {
        const __nv_bfloat16* row = keys + rows[r] * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < DPL) v[i] += b2f(row[i]);
    }
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) { v[i] = qsa_bf16r(v[i] / (float)p->ratio); ss += v[i] * v[i]; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float inv = rsqrtf(ss / (float)hd + p->eps);
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) {
        v[i] = qsa_bf16r(v[i] * inv * (1.0f + p->kw[lane * DPL + i]));
        sm[lane * DPL + i] = v[i];
    }
    __syncwarp();
    const float* ct = p->cos_t + (long long)pos_rope * rdim;
    const float* st = p->sin_t + (long long)pos_rope * rdim;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) {
        const int d = lane * DPL + i;
        if (d < half)       v[i] = qsa_bf16r(sm[d] * ct[d] - sm[d + half] * st[d]);
        else if (d < rdim)  v[i] = qsa_bf16r(sm[d] * ct[d] + sm[d - half] * st[d]);
    }
    __syncwarp();
}

// 2. decode / verify scores: one warp per (column b = blockIdx.y, block j). scores[b*nblk_stride + j] =
//    Σ_h relu(q_h·K_j)/√hd for j < pc/ratio (pc = pos[b]+1, `pos` = LOGICAL positions). q rows are the
//    normed+roped queries at q + b*q_pitch (heads*hd contiguous). geom = stride | q_pitch<<20 | nblk_stride<<40.
extern "C" __global__ void qsa_score_b(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* keys,
                                       const int* pos, const int* slot_ids, const unsigned char* path,
                                       const int* cps, const QsaParams* p, long long geom) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int stride = (int)(geom & 0xFFFFF);
    const int q_pitch = (int)((geom >> 20) & 0xFFFFF);
    const int nblk_stride = (int)((geom >> 40) & 0xFFFFF);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    const int pc = pos[b] + 1;
    const int ratio = p->ratio;
    const int nb = pc / ratio;
    if (j >= nb) return;
    const int pos_start = cps ? cps[b] : pos[0];
    const long long slot_base = (long long)slot_ids[b] * stride;
    long long rows[8];
    for (int r = 0; r < ratio; r++) rows[r] = slot_base + qsa_col(j * ratio + r, pos_start, path, b);
    float v[4];
    qsa_block_key(v, keys, rows, p, j * ratio, sm + warp * QSA_HD_MAX, lane);
    const int hd = p->hd, DPL = hd >> 5;
    float s = 0.0f;
    for (int h = 0; h < p->heads; h++) {
        const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)h * hd + lane * DPL;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < DPL) dot += v[i] * b2f(qrow[i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        s += fmaxf(dot, 0.0f);
    }
    if (lane == 0) {
        s = s / sqrtf((float)hd);
        if (s == 0.0f) s = 0.0f;              // never -0.0: the radix select keys on the float bits
        scores[(long long)b * nblk_stride + j] = s;
    }
}

// 3. prefill block keys: blocks[j] for j < nblk from the slot's committed raw keys (identity ranks).
extern "C" __global__ void qsa_block_keys_b(__nv_bfloat16* blocks, const __nv_bfloat16* keys, const QsaParams* p, int nblk) {
    __shared__ float sm[8 * QSA_HD_MAX];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= nblk) return;
    long long rows[8];
    for (int r = 0; r < p->ratio; r++) rows[r] = (long long)j * p->ratio + r;
    float v[4];
    qsa_block_key(v, keys, rows, p, j * p->ratio, sm + warp * QSA_HD_MAX, lane);
    const int DPL = p->hd >> 5;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (i < DPL) blocks[(long long)j * p->hd + lane * DPL + i] = f2b(v[i]);
}

// 4. prefill scores: QSA_PF_QT queries (positions pos_start+t, their normed+roped q in smem as f32)
//    against one block per thread; scores[t*nblk_stride + j] for j < (pos_start+t+1)/ratio.
//    grid (ceil(nblk/256), ceil(n/QT)), block 256, smem QT*heads*hd floats.
extern "C" __global__ void qsa_score_prefill_b(float* scores, const __nv_bfloat16* q, const __nv_bfloat16* blocks,
                                               const QsaParams* p, int pos_start, int n, int nblk_stride, int q_pitch) {
    extern __shared__ float sq[];
    const int hd = p->hd, heads = p->heads, ratio = p->ratio;
    const int qd = heads * hd;
    const int t0 = blockIdx.y * QSA_PF_QT;
    for (int i = threadIdx.x; i < QSA_PF_QT * qd; i += blockDim.x) {
        const int qq = i / qd, e = i % qd;
        const int t = t0 + qq;
        sq[i] = (t < n) ? b2f(q[(long long)t * q_pitch + e]) : 0.0f;
    }
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nblk_stride) return;
    // Only blocks some query of this tile can see.
    const int tlast = min(t0 + QSA_PF_QT - 1, n - 1);
    if (j >= (pos_start + tlast + 1) / ratio) return;
    float acc[QSA_PF_QT][QSA_HEADS_MAX];
    #pragma unroll
    for (int qq = 0; qq < QSA_PF_QT; qq++)
        #pragma unroll
        for (int h = 0; h < QSA_HEADS_MAX; h++) acc[qq][h] = 0.0f;
    const __nv_bfloat16* krow = blocks + (long long)j * hd;
    for (int d = 0; d < hd; d += 4) {
        const float k0 = b2f(krow[d]), k1 = b2f(krow[d + 1]), k2 = b2f(krow[d + 2]), k3 = b2f(krow[d + 3]);
        #pragma unroll
        for (int qq = 0; qq < QSA_PF_QT; qq++) {
            #pragma unroll
            for (int h = 0; h < QSA_HEADS_MAX; h++) if (h < heads) {   // compile-time indices: acc stays in registers
                const float4 qv = *reinterpret_cast<const float4*>(sq + (qq * heads + h) * hd + d);
                acc[qq][h] += k0 * qv.x + k1 * qv.y + k2 * qv.z + k3 * qv.w;
            }
        }
    }
    #pragma unroll
    for (int qq = 0; qq < QSA_PF_QT; qq++) {
        const int t = t0 + qq;
        if (t >= n) break;
        if (j >= (pos_start + t + 1) / ratio) continue;
        float s = 0.0f;
        #pragma unroll
        for (int h = 0; h < QSA_HEADS_MAX; h++) if (h < heads) s += fmaxf(acc[qq][h], 0.0f);
        s = s / sqrtf((float)hd);
        if (s == 0.0f) s = 0.0f;
        scores[(long long)t * nblk_stride + j] = s;
    }
}

// Block-wide exclusive scan of two int flags (1024 threads = 32 warps). Returns the exclusive prefix
// of each and the block totals.
__device__ __forceinline__ void qsa_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc, int* wa, int* wc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ia = a, ic = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int xa = __shfl_up_sync(0xffffffffu, ia, off), xc = __shfl_up_sync(0xffffffffu, ic, off);
        if (lane >= off) { ia += xa; ic += xc; }
    }
    if (lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    int ba = 0, bc = 0, sa = 0, sc = 0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; w++) { if (w < warp) { ba += wa[w]; bc += wc[w]; } sa += wa[w]; sc += wc[w]; }
    pa = ba + ia - a; pc = bc + ic - c; ta = sa; tc = sc;
    __syncthreads();
}

// 5. top-k + selection list. One 1024-thread block per column b. pc = pos ? pos[b]+1 : pos_start+b+1;
//    nb = pc/ratio blocks, k = min(topk, nb). Radix select (4 passes, MSB first) over the non-negative
//    score bits finds the k-th largest key T and how many key==T blocks to keep (`kp`); the stable
//    compaction keeps every block > T and the first kp blocks == T in block order, emitting their
//    ratio ranks (→ cache columns via qsa_col) ascending, then the tail ranks [nb*ratio, pc).
//    sel row pitch = p->sel_max; pos_sel[b] = nsel-1 (the splitk/reduce "pos" of the selection).
extern "C" __global__ void __launch_bounds__(1024)
qsa_topk_b(int* sel, int* pos_sel, const float* scores, const int* pos, const unsigned char* path,
           const int* cps, const QsaParams* p, int nblk_stride, int pos_start_chain) {
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int ratio = p->ratio, K = p->topk, sel_max = p->sel_max;
    const int pc = pos ? pos[b] + 1 : pos_start_chain + b + 1;
    const int nb = pc / ratio;
    const int tail = pc - nb * ratio;
    const int pos_start = pos ? (cps ? cps[b] : pos[0]) : 0;
    const float* sc = scores + (long long)b * nblk_stride;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (nb > K) {
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int j = tid; j < nb; j += blockDim.x) {
                const unsigned key = __float_as_uint(sc[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0, bin = 0, rem = kp;
                for (int i = 255; i >= 0; i--) {
                    const int c = hist[i];
                    if (cum + c >= kp) { bin = i; rem = kp - cum; break; }
                    cum += c;
                }
                s_bin = bin; s_kp = rem;
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            __syncthreads();
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0;
    int* srow = sel + (long long)b * sel_max;
    for (int base = 0; base < nb; base += blockDim.x) {
        const int j = base + tid;
        const bool valid = j < nb;
        const unsigned key = valid ? __float_as_uint(sc[j]) : 0u;
        const bool gt = valid && key > T;
        const bool eq = valid && key == T;
        int p_sel0, p_eq, t_sel, t_eq;
        // first scan: equal-rank (needed to decide selection), second: selection rank
        qsa_exscan2(eq ? 1 : 0, 0, p_eq, p_sel0, t_eq, t_sel, wa, wc);
        const bool selected = gt || (eq && (run_eq + p_eq) < kp);
        int p_s, p_d, t_s, t_d;
        qsa_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) {
            const int r = run_sel + p_s;
            for (int i = 0; i < ratio; i++) srow[r * ratio + i] = qsa_col(j * ratio + i, pos_start, path, b);
        }
        run_sel += t_s; run_eq += t_eq;
    }
    const int nsel = run_sel * ratio;
    for (int i = tid; i < tail; i += blockDim.x) srow[nsel + i] = qsa_col(nb * ratio + i, pos_start, path, b);
    if (tid == 0) pos_sel[b] = nsel + tail - 1;
}

// 6. gqa_attn_sel_splitk — gqa_attn_splitk with the key set replaced by the column's selection list:
//    pc = pos_sel[b]+1 keys, rank r → cache column sel[b*sel_max + r]. Split structure, warp stride,
//    reduction order and merge are the dense kernel's, verbatim: an identity list is bit-identical to
//    gqa_attn_splitk, and gqa_attn_reduce (with pos = pos_sel) finishes it unchanged.
extern "C" __global__ void gqa_attn_sel_splitk(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* pos_sel, long long bs_packed, int nh_packed, const int* slot_ids,
    const int* sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos_sel[b] + 1;
    const int slot = slot_ids[b];
    const int* srow = sel + (long long)b * sel_max;

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

    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r = start + warp; r < end; r += NW) {
        const int t = srow[r];
        const __nv_bfloat16* krow = kb + (long long)t * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)t * hd;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
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

// 6b. gqa_attn_sel_splitk_k8v4 — gqa_attn_sel_splitk over the k8v4 packed cache: K rows of 20 B/16
//     (int8 codes + fp16 block scale), V rows of 12 B/16 (q4 nibbles + e4m3 block scale). The row
//     dequant is gqa_attn_splitk_k8v4's, the key set is the column's selection list; split structure,
//     warp stride, reduction order and merge are the bf16 kernel's verbatim, so gqa_attn_reduce (with
//     pos = pos_sel) finishes it unchanged and decode == verify col-0 holds as for the dense pair.
extern "C" __global__ void gqa_attn_sel_splitk_k8v4(
    float* out_m, float* out_l, float* out_acc,
    const __nv_bfloat16* q, const unsigned char* k_cache, const unsigned char* v_cache,
    const int* pos_sel, long long bs_packed, int nh_packed, const int* slot_ids,
    const int* sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const float scale = 1.0f / sqrtf((float)hd);
    const int gqa_ratio = nh / nkv;
    const int stride  = (int)(bs_packed & 0x7FFFF);
    const int ns_grid = (int)((bs_packed >> 19) & 0x3F);
    const int B       = (int)((bs_packed >> 25) & 0x3F);
    const long long q_pitch = (bs_packed >> 31) & 0x7FFFF;

    const int blk = blockIdx.x;
    const int qh = blk / (ns_grid * B);
    const int rem = blk % (ns_grid * B);
    const int split = rem / B;
    const int b = rem % B;
    const int kvh = qh / gqa_ratio;
    const int pc = pos_sel[b] + 1;
    const int slot = slot_ids[b];
    const int* srow = sel + (long long)b * sel_max;

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

    const int k_rb = KV8_ROW_BYTES(hd);              // K rows: 20 B/16
    const int v_rb = KVQ_ROW_BYTES(hd);              // V rows: 12 B/16
    const int lane_blk = (lane * DPL) / KVQ_BLK;     // the 16-block holding this lane's slice
    const int lane_off = (lane * DPL) % KVQ_BLK;     // its first code within the block (4-aligned)
    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const unsigned char* kb = k_cache + kvbase * k_rb + lane_blk * 20;
    const unsigned char* vb = v_cache + kvbase * v_rb + lane_blk * 12;
    for (int r = start + warp; r < end; r += NW) {
        const int t = srow[r];
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

// 7. gqa_attn_sel_prefill — causal prefill attention over per-query selection lists (one warp per
//    query, its whole softmax in registers, gqa_attn_prefill's arithmetic). Prefill is outside the
//    batch-invariance contract, so no split structure is needed. k/v pointers are the SLOT base.
//    grid (ceil(N/8) * nh), block 256.
extern "C" __global__ void gqa_attn_sel_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh_packed, float scale,
    int N, const int* sel, const int* pos_sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int QT = blockDim.x >> 5;
    const int blk = blockIdx.x;
    const int tile = blk / nh, qh = blk % nh;
    const int kvh = qh / (nh / nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;
    const int t = tile * QT + warp;
    if (t >= N) return;
    const int pc = pos_sel[t] + 1;
    const int* srow = sel + (long long)t * sel_max;

    const __nv_bfloat16* qrow = q + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
    float qv[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) qv[i] = (i < DPL) ? b2f(qrow[i]) : 0.0f;
    float m = -1e30f, l = 0.0f;
    float acc[SK_DPL_MAX];
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = 0.0f;
    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r = 0; r < pc; r++) {
        const int tt = srow[r];
        const __nv_bfloat16* krow = kb + (long long)tt * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) s += qv[i] * ((i < DPL) ? b2f(krow[i]) : 0.0f);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)tt * hd;
        #pragma unroll
        for (int i = 0; i < SK_DPL_MAX; i++) acc[i] = acc[i] * a_old + a_cur * ((i < DPL) ? b2f(vrow[i]) : 0.0f);
        m = m_new;
        l = l * a_old + a_cur;
    }
    __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
    #pragma unroll
    for (int i = 0; i < SK_DPL_MAX; i++) if (i < DPL) orow[i] = f2b(l > 0.0f ? acc[i] / l : 0.0f);
}

// 8. raw-key compaction twin of compact_kv_b (an accepted tree path moved to contiguous columns).
extern "C" __global__ void qsa_compact_b(__nv_bfloat16* keys, __nv_bfloat16* scratch, const int* src_pos,
                                         int len, int pos_start, int slot, int stride, int hd, int dir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len * hd) return;
    int k = idx / hd, dv = idx % hd;
    long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    long long coff = ((long long)slot * stride + cache_pos) * hd + dv;
    long long soff = (long long)k * hd + dv;
    if (dir == 0) scratch[soff] = keys[coff]; else keys[coff] = scratch[soff];
}

// 4b. prefill score combine after the cuBLAS head GEMMs: G[h][t*nblk + j] = (q_{t,h} · K_j)/√hd
//     (alpha folded in), scores[t*nblk + j] = Σ_h relu(G) for j < (pos_start+t+1)/ratio.
extern "C" __global__ void qsa_score_combine_b(float* scores, const float* G, const QsaParams* p,
                                               int pos_start, int nch, int nblk) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nch * nblk) return;
    const int t = (int)(idx / nblk), j = (int)(idx % nblk);
    if (j >= (pos_start + t + 1) / p->ratio) return;
    const long long hs = (long long)nch * nblk;
    float s = 0.0f;
    for (int h = 0; h < p->heads; h++) s += fmaxf(G[h * hs + idx], 0.0f);
    if (s == 0.0f) s = 0.0f;
    scores[idx] = s;
}

// 7b. gqa_attn_sel_prefill2 — the selected-list prefill attention, restructured for gathered rows:
//     one warp per (query, kv head, group of SEL_GS query heads) so a K/V row is fetched once per
//     group instead of once per head, and SEL_KU keys in flight per iteration (the selected rows have
//     no locality — the one-key loop of gqa_attn_sel_prefill was latency-bound: 250 ms/layer on a
//     7.3K prompt). Per-query arithmetic and order are unchanged (keys ascending, same lane dot /
//     xor-tree / online softmax), so the output is bit-identical to gqa_attn_sel_prefill.
//     hd <= 256. grid (ceil(N/8) * nkv * ngroups), block 256.
#define SEL_GS 4
#define SEL_KU 4
extern "C" __global__ void __launch_bounds__(256) gqa_attn_sel_prefill2(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh_packed, float scale,
    int N, const int* sel, const int* pos_sel, int sel_max) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int G = nh / nkv;
    const int ngroups = (G + SEL_GS - 1) / SEL_GS;
    const int QT = blockDim.x >> 5;
    const int blk = blockIdx.x;
    const int tile = blk / (nkv * ngroups);
    const int rem = blk % (nkv * ngroups);
    const int kvh = rem / ngroups, grp = rem % ngroups;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;                    // <= 8
    const int t = tile * QT + warp;
    if (t >= N) return;
    const int pc = pos_sel[t] + 1;
    const int* srow = sel + (long long)t * sel_max;
    const int h0 = grp * SEL_GS;
    const int nhg = min(SEL_GS, G - h0);

    float qv[SEL_GS][8];
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) {
        const int qh = kvh * G + h0 + g;
        const __nv_bfloat16* qrow = q + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 8; i++) qv[g][i] = (g < nhg && i < DPL) ? b2f(qrow[i]) : 0.0f;
    }
    float m[SEL_GS], l[SEL_GS], acc[SEL_GS][8];
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) { m[g] = -1e30f; l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[g][i] = 0.0f; }

    const long long kvbase = (long long)kvh * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * hd + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * hd + lane * DPL;
    for (int r0 = 0; r0 < pc; r0 += SEL_KU) {
        float kr[SEL_KU][8], vr[SEL_KU][8];
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++) {
            const int r = min(r0 + u, pc - 1);          // clamp: surplus lanes recompute the last key, masked below
            const int tt = srow[r];
            const __nv_bfloat16* krow = kb + (long long)tt * hd;
            const __nv_bfloat16* vrow = vb + (long long)tt * hd;
            #pragma unroll
            for (int i = 0; i < 8; i++) { kr[u][i] = (i < DPL) ? b2f(krow[i]) : 0.0f; vr[u][i] = (i < DPL) ? b2f(vrow[i]) : 0.0f; }
        }
        float s[SEL_KU][SEL_GS];
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++)
            #pragma unroll
            for (int g = 0; g < SEL_GS; g++) {
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) d += qv[g][i] * kr[u][i];
                s[u][g] = d;
            }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            #pragma unroll
            for (int u = 0; u < SEL_KU; u++)
                #pragma unroll
                for (int g = 0; g < SEL_GS; g++) s[u][g] += __shfl_xor_sync(0xffffffffu, s[u][g], off);
        #pragma unroll
        for (int u = 0; u < SEL_KU; u++) {
            if (r0 + u >= pc) break;
            #pragma unroll
            for (int g = 0; g < SEL_GS; g++) {
                const float sc = s[u][g] * scale;
                const float m_new = fmaxf(m[g], sc);
                const float a_old = __expf(m[g] - m_new), a_cur = __expf(sc - m_new);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = acc[g][i] * a_old + a_cur * vr[u][i];
                m[g] = m_new;
                l[g] = l[g] * a_old + a_cur;
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < SEL_GS; g++) {
        if (g >= nhg) break;
        const int qh = kvh * G + h0 + g;
        __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 8; i++) if (i < DPL) orow[i] = f2b(l[g] > 0.0f ? acc[g][i] / l[g] : 0.0f);
    }
}

// ===================== GPTQ → NVFP4 (src/gptq.rs, the --gptq quantizer) =====================
// W is f32 row-major [M, K] on device (a working copy of the bf16 source weight, optionally
// micro-rotated). U is the upper Cholesky factor of the damped inverse Hessian (f32 row-major
// [K, K]; only j <= k is read). The sweep quantizes one 128-column block for ALL rows at once —
// one thread per row, columns strictly in order with the GPTQ error feedback inside the block —
// and writes the NVFP4 codes (nibbles, [M, K/2]) and the per-16 E4M3 block scales ([M, K/16]) in
// the artifact's layout; the cross-block propagation W[:, c1:] -= Err · U[c0:c1, c1:] is a cuBLAS
// GEMM issued by the host between sweeps.

__device__ __forceinline__ unsigned char gptq_e2m1(float x) {   // == quant::f32_to_e2m1 (RNE, ties to even code)
    const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    unsigned char sign = x < 0.f ? 8 : 0;
    float a = fminf(fabsf(x), 6.f);
    unsigned char best = 0; float be = 1e30f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float e = fabsf(grid[i] - a);
        if (e < be - 1e-9f || (fabsf(e - be) <= 1e-9f && (i % 2) == 0)) { be = e; best = (unsigned char)i; }
    }
    return sign | best;
}
__device__ __forceinline__ float gptq_e2m1_val(unsigned char c) {
    const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = grid[c & 7]; return (c & 8) ? -v : v;
}

extern "C" __global__ void gptq_bf16_to_f32_b(float* out, const __nv_bfloat16* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// |x| max over n floats (atomicMax on the non-negative float bit pattern; *out must start at 0).
extern "C" __global__ void gptq_absmax_b(unsigned int* out, const __nv_bfloat16* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? fabsf(__bfloat162float(x[i])) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(v));
}

// NVFP4 activation headroom calibration. One logical sample is the amax of a contiguous
// 16-element activation block, matching the dynamic UE4M3 block scale consumed by W4A4. The
// histogram is logarithmic so corpora with very different activation ranges can be merged by
// adding counts instead of merging already-rounded global scales.
//
// stats layout: [512 log2 histogram bins | zero-block count | non-finite-block count], all u64.
// `running_max` is the positive f32 bit-pattern, so integer atomicMax has the desired ordering.
extern "C" __global__ void igs_hist_b(unsigned long long* stats, unsigned int* running_max,
                                      const __nv_bfloat16* x, long long n) {
    constexpr int NBINS = 512;
    constexpr float LOG2_MIN = -40.0f;
    constexpr float LOG2_MAX = 40.0f;
    __shared__ unsigned int shist[NBINS];
    __shared__ unsigned int sextra[2];
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) shist[i] = 0;
    if (threadIdx.x < 2) sextra[threadIdx.x] = 0;
    __syncthreads();

    const long long block = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nblocks = (n + 15) / 16;
    if (block < nblocks) {
        const long long base = block * 16;
        float amax = 0.0f;
        bool invalid = false;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const long long i = base + j;
            if (i < n) {
                const float value = __bfloat162float(x[i]);
                invalid |= !isfinite(value);
                if (!invalid) amax = fmaxf(amax, fabsf(value));
            }
        }
        if (invalid) {
            atomicAdd(&sextra[1], 1u);
        } else if (amax == 0.0f) {
            atomicAdd(&sextra[0], 1u);
        } else {
            atomicMax(running_max, __float_as_uint(amax));
            const float frac = (log2f(amax) - LOG2_MIN) / (LOG2_MAX - LOG2_MIN);
            int bin = (int)floorf(frac * (float)NBINS);
            bin = max(0, min(NBINS - 1, bin));
            atomicAdd(&shist[bin], 1u);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) {
        const unsigned int count = shist[i];
        if (count) atomicAdd(&stats[i], (unsigned long long)count);
    }
    if (threadIdx.x < 2 && sextra[threadIdx.x]) {
        atomicAdd(&stats[NBINS + threadIdx.x], (unsigned long long)sextra[threadIdx.x]);
    }
}

// In-place 16-point Hadamard (orthonormal, H/4) on every 16-vector of a [M, K] f32 matrix.
// axis 0: vectors along the rows (elements row*K + 16b + i) — the K (input) dimension;
// axis 1: vectors along the columns (elements (16b+i)*K + col) — for the Hessian's row side.
extern "C" __global__ void gptq_hadamard16_b(float* W, int M, int K, int axis) {
    long long nvec = (axis == 0) ? (long long)M * (K / 16) : (long long)(M / 16) * K;
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    float* p; long long stride;
    if (axis == 0) { long long row = v / (K / 16), b = v % (K / 16); p = W + row * K + b * 16; stride = 1; }
    else           { long long b = v / K, col = v % K; p = W + (b * 16) * K + col; stride = K; }
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = p[i * stride];
    #pragma unroll
    for (int len = 1; len < 16; len <<= 1)
        #pragma unroll
        for (int i = 0; i < 16; i += 2 * len)
            #pragma unroll
            for (int j = i; j < i + len; j++) { float a = x[j], b = x[j + len]; x[j] = a + b; x[j + len] = a - b; }
    #pragma unroll
    for (int i = 0; i < 16; i++) p[i * stride] = x[i] * 0.25f;
}

// The block sweep. c0 = first column of the block, bs = block width (multiple of 16, <= 256).
// s_tensor = 1/global_scale (the per-tensor NVFP4 scale). nclip = number of clip ratios tried per
// 16-group (1 = plain amax/6). Err [M, bs] row-major receives (w - q)/U[j][j].
extern "C" __global__ void gptq_sweep_b(float* W, unsigned char* qw, unsigned char* qs, float* Err,
                                        const float* U, int M, int K, int c0, int bs, float s_tensor, int nclip) {
    const float ratios[7] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f};
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float* w = W + (long long)r * K;
    float* e = Err + (long long)r * bs;
    float s = 1.f, inv = 0.f;
    for (int j = 0; j < bs; j++) {
        const int col = c0 + j;
        if ((col & 15) == 0) {
            // group scale from the CURRENT (error-updated) values of this row's group
            float amax = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(w[col + i]));
            unsigned char best_code = 0; float best_err = 1e30f;
            for (int c = 0; c < nclip; c++) {
                float sraw = (amax > 0.f) ? amax * ratios[c] / 6.f / s_tensor : 0.f;
                unsigned char code = f32_to_e4m3(sraw);
                float sc = e4m3_f(code) * s_tensor;
                float iv = (sc > 0.f) ? 1.f / sc : 0.f;
                float err = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; i++) { float q = gptq_e2m1_val(gptq_e2m1(w[col + i] * iv)) * sc; err += (q - w[col + i]) * (q - w[col + i]); }
                if (c == 0 || err < best_err) { best_err = err; best_code = code; }
            }
            qs[(long long)r * (K / 16) + col / 16] = best_code;
            s = e4m3_f(best_code) * s_tensor;
            inv = (s > 0.f) ? 1.f / s : 0.f;
        }
        unsigned char code = gptq_e2m1(w[col] * inv);
        float q = gptq_e2m1_val(code) * s;
        long long qi = (long long)r * (K / 2) + col / 2;
        unsigned char byte = qw[qi];
        qw[qi] = (col & 1) ? ((byte & 0x0F) | (code << 4)) : ((byte & 0xF0) | code);
        const float d = U[(long long)col * K + col];
        const float err = (w[col] - q) / d;
        e[j] = err;
        const float* urow = U + (long long)col * K;
        for (int k2 = j; k2 < bs; k2++) w[c0 + k2] -= err * urow[c0 + k2];
    }
}

// Gather n token rows (each K bf16, contiguous) by index into dst [n, K].
extern "C" __global__ void gptq_gather_rows_b(__nv_bfloat16* dst, const __nv_bfloat16* src, const int* idx, int n, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * K) return;
    int t = (int)(i / K), c = (int)(i % K);
    dst[i] = src[(long long)idx[t] * K + c];
}

// act[i, t] = silu(g[i, t]) * u[i, t] from a fused [2I, n] f32 gate|up buffer (column-major per
// token: gate rows 0..I, up rows I..2I), written bf16 [I, n].
extern "C" __global__ void gptq_silu_mul_gu_b(__nv_bfloat16* act, const float* gu, int I, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)I * n) return;
    int t = (int)(i / I), r = (int)(i % I);
    float g = gu[(long long)t * 2 * I + r], u = gu[(long long)t * 2 * I + I + r];
    act[i] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
}

// The engine-side micro-rotation for MR-GPTQ artifacts: x' = H16/4 · x on every 16-block of the
// K (feature) dimension of a bf16 activation matrix [K, n] (feature-contiguous per column).
extern "C" __global__ void gptq_rotate_act_b(__nv_bfloat16* out, const __nv_bfloat16* x, long long nvec) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    const __nv_bfloat16* p = x + v * 16;
    float y[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) y[i] = __bfloat162float(p[i]);
    #pragma unroll
    for (int len = 1; len < 16; len <<= 1)
        #pragma unroll
        for (int i = 0; i < 16; i += 2 * len)
            #pragma unroll
            for (int j = i; j < i + len; j++) { float a = y[j], b = y[j + len]; y[j] = a + b; y[j + len] = a - b; }
    __nv_bfloat16* o = out + v * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = __float2bfloat16(y[i] * 0.25f);
}
extern "C" __global__ void gptq_absmax_f32_b(unsigned int* out, const float* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? fabsf(x[i]) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(v));
}

// Pick the best E4M3 local scale for one 16-value NVFP4 group. The candidates deliberately
// match gptq_sweep_b: this helper is used by the alternating tensor-scale fit and by static
// activation-order's precomputed groups, so calibration and the final sweep share one grid.
__device__ __forceinline__ unsigned char gptq_best_scale16(const float x[16], float s_tensor,
                                                            int nclip, float* best_mse) {
    const float ratios[7] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f};
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));
    unsigned char best_code = 0;
    float be = 1e30f;
    for (int c = 0; c < nclip; c++) {
        float raw = (amax > 0.f && s_tensor > 0.f) ? amax * ratios[c] / 6.f / s_tensor : 0.f;
        unsigned char code = f32_to_e4m3(raw);
        float sc = e4m3_f(code) * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        float err = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float q = gptq_e2m1_val(gptq_e2m1(x[i] * inv)) * sc;
            float d = q - x[i];
            err += d * d;
        }
        if (c == 0 || err < be) { be = err; best_code = code; }
    }
    *best_mse = be;
    return best_code;
}

// Alternating global-scale sufficient statistics. With a fixed local-scale/code assignment,
// q_i = s_tensor * z_i and the least-squares tensor scale is sum(w_i*z_i)/sum(z_i^2).
// Re-evaluating this kernel at that scale reassigns the local E4M3 scales and FP4 codes. One
// block contributes three f64 atomics, avoiding the precision loss of atomics per 16-value group.
extern "C" __global__ void gptq_scale_stats_f32_b(double* stats, const float* w,
                                                   long long ngroups, float s_tensor, int nclip) {
    __shared__ double sn[256], sd[256], se[256];
    double num = 0.0, den = 0.0, mse = 0.0;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < ngroups; g += (long long)gridDim.x * blockDim.x) {
        float x[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) x[i] = w[g * 16 + i];
        float group_mse;
        unsigned char scode = gptq_best_scale16(x, s_tensor, nclip, &group_mse);
        float es = e4m3_f(scode);
        float sc = es * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float z = es * gptq_e2m1_val(gptq_e2m1(x[i] * inv));
            num += (double)x[i] * (double)z;
            den += (double)z * (double)z;
        }
        mse += (double)group_mse;
    }
    sn[threadIdx.x] = num; sd[threadIdx.x] = den; se[threadIdx.x] = mse;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x+s]; sd[threadIdx.x] += sd[threadIdx.x+s]; se[threadIdx.x] += se[threadIdx.x+s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(stats, sn[0]); atomicAdd(stats + 1, sd[0]); atomicAdd(stats + 2, se[0]); }
}

// Same statistics directly from the bf16 source. This lets a stacked MoE tensor share one
// optimized global scale without materializing every expert as f32 at once. Because MR's H16 acts
// independently on exactly these groups, the optional rotation is lossless with respect to the
// f32 working copy produced by gptq_w32.
extern "C" __global__ void gptq_scale_stats_bf16_b(double* stats, const __nv_bfloat16* w,
                                                    long long ngroups, int rotate,
                                                    float s_tensor, int nclip) {
    __shared__ double sn[256], sd[256], se[256];
    double num = 0.0, den = 0.0, mse = 0.0;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < ngroups; g += (long long)gridDim.x * blockDim.x) {
        float x[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) x[i] = __bfloat162float(w[g * 16 + i]);
        if (rotate) {
            #pragma unroll
            for (int len = 1; len < 16; len <<= 1)
                #pragma unroll
                for (int i = 0; i < 16; i += 2 * len)
                    #pragma unroll
                    for (int j = i; j < i + len; j++) { float a=x[j], b=x[j+len]; x[j]=a+b; x[j+len]=a-b; }
            #pragma unroll
            for (int i = 0; i < 16; i++) x[i] *= 0.25f;
        }
        float group_mse;
        unsigned char scode = gptq_best_scale16(x, s_tensor, nclip, &group_mse);
        float es = e4m3_f(scode);
        float sc = es * s_tensor;
        float inv = sc > 0.f ? 1.f / sc : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float z = es * gptq_e2m1_val(gptq_e2m1(x[i] * inv));
            num += (double)x[i] * (double)z;
            den += (double)z * (double)z;
        }
        mse += (double)group_mse;
    }
    sn[threadIdx.x] = num; sd[threadIdx.x] = den; se[threadIdx.x] = mse;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x+s]; sd[threadIdx.x] += sd[threadIdx.x+s]; se[threadIdx.x] += se[threadIdx.x+s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(stats, sn[0]); atomicAdd(stats + 1, sd[0]); atomicAdd(stats + 2, se[0]); }
}

// Static groups: choose every local scale from the original, unpermuted rotated weight before
// GPTQ updates it. Static activation-order can then visit columns in any order while every final
// code remains reconstructible by the original block-of-16 scale layout.
extern "C" __global__ void gptq_static_scales_b(unsigned char* qs, const float* w,
                                                 long long ngroups, float s_tensor, int nclip) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = w[g * 16 + i];
    float mse;
    qs[g] = gptq_best_scale16(x, s_tensor, nclip, &mse);
}

// NVIDIA-style NVFP4 local-Hessian scale search. One CUDA block handles eight output rows
// for one original 16-column group. Each warp cooperatively evaluates all 126 positive finite
// E4M3 scales and minimizes dw^T H_block dw. The same scale bytes and
// packed FP4 layout are consumed by the existing runtime; this adds no inference overhead.
extern "C" __global__ void gptq_static_scales_hessian_b(
        unsigned char* qs, const float* w, const float* H,
        unsigned long long* fallback_count, int M, int K, float s_tensor) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = blockIdx.x;
    const int row = (int)blockIdx.y * 8 + warp;
    const int c0 = group * 16;
    __shared__ float h16[16 * 16];

    const int t = threadIdx.x;
    h16[t] = H[(long long)(c0 + t / 16) * K + c0 + t % 16];
    __syncthreads();
    if (row >= M) return;

    // The warp cooperates on one candidate at a time: lane i owns dw[i], then broadcasts it
    // to form one Hessian row. This performs the same arithmetic as four independent candidates
    // per lane while keeping register pressure and PTX size bounded.
    const unsigned mask = 0xffffffffu;
    const float wi = lane < 16 ? w[(long long)row * K + c0 + lane] : 0.f;
    float amax = lane < 16 ? fabsf(wi) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
    float best_loss = 3.402823466e+38F;
    int best_code = 0;
    #pragma unroll 1
    for (int code = 1; code <= 126; code++) {
        const float scale = e4m3_f((unsigned char)code) * s_tensor;
        const float inv = scale > 0.f ? 1.f / scale : 0.f;
        const float q = lane < 16 ? gptq_e2m1_val(gptq_e2m1(wi * inv)) * scale : 0.f;
        const float dw = lane < 16 ? wi - q : 0.f;
        float hd = 0.f;
        #pragma unroll 1
        for (int j = 0; j < 16; j++) {
            const float dwj = __shfl_sync(mask, dw, j);
            if (lane < 16) hd += h16[lane * 16 + j] * dwj;
        }
        float loss = dw * hd;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            loss += __shfl_down_sync(mask, loss, off);
        if (lane == 0 && isfinite(loss) && loss < best_loss) {
            best_loss = loss; best_code = code;
        }
    }

    if (lane == 0) {
        if (best_code == 0) {
            // A non-finite Hessian should never reach here; preserve a valid artifact with the
            // ordinary un-clipped amax scale and expose the event to the host diagnostic.
            best_code = (int)f32_to_e4m3(
                (amax > 0.f && s_tensor > 0.f) ? amax / 6.f / s_tensor : 0.f);
            atomicAdd(fallback_count, 1ULL);
        }
        qs[(long long)row * (K / 16) + group] = (unsigned char)best_code;
    }
}

// out[:,j] = in[:,perm[j]], and out[i,j] = in[perm[i],perm[j]]. The latter permutes the
// Hessian into the same activation-importance basis as the working weight.
extern "C" __global__ void gptq_permute_w_b(float* out, const float* in, const int* perm, int M, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * K) return;
    int r = (int)(i / K), j = (int)(i % K);
    out[i] = in[(long long)r * K + perm[j]];
}
extern "C" __global__ void gptq_permute_h_b(float* out, const float* in, const int* perm, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)K * K) return;
    int r = (int)(i / K), c = (int)(i % K);
    out[i] = in[(long long)perm[r] * K + perm[c]];
}

// GPTQ sweep in activation-importance order with precomputed static scales. W and U are already
// permuted. Codes are written directly at their ORIGINAL columns, so neither the artifact nor the
// serving kernels need a permutation table.
extern "C" __global__ void gptq_sweep_static_b(float* W, unsigned char* qw,
                                                const unsigned char* qs, float* Err,
                                                const float* U, const int* perm,
                                                int M, int K, int c0, int bs, float s_tensor) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float* w = W + (long long)r * K;
    float* e = Err + (long long)r * bs;
    for (int j = 0; j < bs; j++) {
        int col = c0 + j;
        int orig = perm[col];
        float s = e4m3_f(qs[(long long)r * (K / 16) + orig / 16]) * s_tensor;
        float inv = s > 0.f ? 1.f / s : 0.f;
        unsigned char code = gptq_e2m1(w[col] * inv);
        float q = gptq_e2m1_val(code) * s;
        long long qi = (long long)r * (K / 2) + orig / 2;
        unsigned char byte = qw[qi];
        qw[qi] = (orig & 1) ? ((byte & 0x0F) | (code << 4)) : ((byte & 0xF0) | code);
        float d = U[(long long)col * K + col];
        float err = (w[col] - q) / d;
        e[j] = err;
        const float* urow = U + (long long)col * K;
        for (int k2 = j; k2 < bs; k2++) w[c0 + k2] -= err * urow[c0 + k2];
    }
}

// Compact activation profile for COLA/ACDM. One block owns one feature channel and reduces all
// sequence positions. stats = global sum/sum-of-squares; sketch is a deterministic CountSketch of
// the mean-pooled channel vector (a sparse random projection without materializing [K,N] on host).
extern "C" __global__ void calib_profile_b(double* stats, float* sketch,
                                             const __nv_bfloat16* x, int K, int N,
                                             int sketch_dim, unsigned int seed) {
    int c = blockIdx.x;
    if (c >= K) return;
    double sum = 0.0, sumsq = 0.0;
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        float v = __bfloat162float(x[(long long)t * K + c]);
        sum += (double)v; sumsq += (double)v * (double)v;
    }
    __shared__ double calib_prof_sum[256];
    __shared__ double calib_prof_sumsq[256];
    double* rs = calib_prof_sum; double* rq = calib_prof_sumsq;
    rs[threadIdx.x] = sum; rq[threadIdx.x] = sumsq;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) { rs[threadIdx.x] += rs[threadIdx.x + off]; rq[threadIdx.x] += rq[threadIdx.x + off]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats[0], rs[0]); atomicAdd(&stats[1], rq[0]);
        unsigned int z = ((unsigned int)c ^ seed) * 0x9e3779b9u;
        z ^= z >> 16; z *= 0x85ebca6bu; z ^= z >> 13;
        int bucket = (int)(z % (unsigned int)sketch_dim);
        float sign = (z & 0x80000000u) ? -1.0f : 1.0f;
        atomicAdd(&sketch[bucket], sign * (float)(rs[0] / (double)N));
    }
}
