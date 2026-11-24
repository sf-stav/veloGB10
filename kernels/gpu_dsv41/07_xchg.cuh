// =============================================================================
// kernels/gpu_dsv41/07_xchg.cuh — band 05 (leg PX1): the dsv41 copies of the world-2 doorbell exchange pair.
// Host side: src/dsv41/tp/xchg.rs (launch, routing, trace). Protocol and invariants: native/tp_doorbell.h (I1-I9),
// unchanged: same ctx, same rings, same len / tail tags, same proxy, same wire bytes.
//
// Copied from the shared pair `tp_gate_copy_signal_mb` (kernels/gpu_batch/02_tp_prefill_conv_delta.cuh, world-2
// fp32 arm) and `tp_wait_add` mode 2 (kernels/gpu_batch/01_norms_tp_transport.cuh), which stay untouched. What
// differs is the external reference recipe's exchange mechanism (PLAN/DSV41_RX4_REPORT.md B.2 / B.5):
//   1. the stage is a PDL dependent: griddepcontrol.wait (the producer's partial is complete), then
//      launch_dependents at once, so the collect is resident while the stage copies (theirs: ACQBULK, PREEXIT);
//   2. the collect is a PDL dependent of the stage: it waits (the stage's epoch store), releases ITS dependent
//      (the consumer) at once, then polls the receive gate WITHOUT sleeping (theirs: LDG.STRONG.SYS + YIELD per
//      poll, no NANOSLEEP, the abort word every 1,024 polls);
//   3. the collect's copy-out is vectorised and spread over several blocks, every load of a thread issued before
//      its adds (theirs: LDG.E.128 uncached, four loads in flight a thread, straight into the consumer's layout).
// The sum is element-wise `lower + upper` (rank 0's partial first), the same IEEE add as `tp_wait_add` mode 2,
// in place: bit-identical output for any grid, any vector width.
//
// Trace (`tr` != 0, `--dsv41-xchg-trace`, diagnostics only): globaltimer stamps per epoch into a mapped host ring
// of D41X_XC_TR_EPOCHS x D41X_XC_TR_STRIDE u64 (layout below, mirrored by xchg.rs `TR_*`). Plain stores; nothing
// else reads them on the device.
// =============================================================================

#include "../../native/tp_doorbell.h"   // relative: the SASS check compiles without -Inative

#define D41X_XC_TR_EPOCHS 4096
#define D41X_XC_TR_STRIDE 8
#define D41X_XC_TR_S_IN 0      // stage block 0 entry (before griddepcontrol.wait)
#define D41X_XC_TR_S_GO 1      // stage block 0 after the wait and the reuse gate
#define D41X_XC_TR_S_DB 2      // the last stage block stored the doorbell (gpu_ready)
#define D41X_XC_TR_C_IN 3      // collect block 0 entry
#define D41X_XC_TR_C_GO 4      // collect block 0 after griddepcontrol.wait
#define D41X_XC_TR_C_SEEN 5    // collect block 0 saw cpu_done >= e
#define D41X_XC_TR_C_END0 6    // collect block 0 finished its share
#define D41X_XC_TR_C_ENDL 7    // collect block (grid - 1) finished its share
#define D41X_XC_POLL_ABORT 1024u   // abort word every this many polls (theirs)

static __device__ __forceinline__ unsigned long long d41x_xc_ld(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void d41x_xc_st_rel(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
static __device__ __forceinline__ unsigned long long d41x_xc_ld_gpu(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void d41x_xc_st_rel_gpu(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.gpu.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
static __device__ __forceinline__ unsigned long long d41x_xc_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
static __device__ __forceinline__ unsigned long long* d41x_xc_flag(tp_dev_ctx* c, int off) {
    return (unsigned long long*)((char*)c->flags + off);
}
static __device__ __forceinline__ void d41x_xc_stamp(unsigned long long* tr, unsigned long long e, int k,
                                                     unsigned long long t) {
    if (tr) tr[(e & (D41X_XC_TR_EPOCHS - 1)) * D41X_XC_TR_STRIDE + k] = t;
}
static __device__ __forceinline__ float4 d41x_xc_ldcv4(const float4* p) {
    float4 v;
    asm volatile("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ float d41x_xc_ldcv1(const float* p) {
    float v;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
// Slow-path abort + lazy deadline check for collect/gather spin loops: fires only every D41X_XC_POLL_ABORT
// polls, arms %globaltimer lazily on the first trigger (0 hot-path cost on <1024-poll exchanges), and publishes
// cooperative device status 11 (I9) past K1_GATE_WAIT_NS so a width/tail mismatch cannot wedge the GPU.
static __device__ __forceinline__ int d41x_xc_poll_abort(tp_dev_ctx* c, const unsigned long long* ab,
                                                         unsigned& polls, unsigned long long& t0) {
    if ((++polls & (D41X_XC_POLL_ABORT - 1)) != 0) return 0;
    if (d41x_xc_ld(ab)) return 1;
    const unsigned long long now = d41x_xc_now();
    if (t0 == 0ull) { t0 = now; return 0; }
    if (now - t0 >= K1_GATE_WAIT_NS) {
        d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11ull);
        return 1;
    }
    return 0;
}
// Phase 1A + 1D: single-line, single-CTA C2C gate on `done` (TP_F_CPU_DONE for world-2 / oneshot_uniform,
// or partner's per-peer cpu_done for tree collects). When `rel_slot` != nullptr and gridDim.x > 1,
// ONLY blockIdx.x == 0 polls C2C (eliminating the multi-CTA C2C sys-load storm during active DMA) and releases
// sibling blocks via a device-L2 word (zero C2C traffic from blocks 1..gridDim.x-1; bit 63 = cooperative abort).
// Every woken block still runs fence.acquire.sys before reading host-pinned recv slots.
static __device__ __forceinline__ int d41x_xc_gate_ptr(tp_dev_ctx* c, const unsigned long long* done,
                                                       unsigned long long e, unsigned long long* rel_slot) {
    int ab_flag = 0;
    if (rel_slot != nullptr && gridDim.x > 1 && blockIdx.x > 0) {
        unsigned long long rv;
        do { rv = d41x_xc_ld_gpu(rel_slot); } while ((rv & 0x7fffffffffffffffull) < e);
        ab_flag = (int)(rv >> 63);
        if (!ab_flag) asm volatile("fence.acquire.sys;" ::: "memory");
        return ab_flag;
    }
    const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
    unsigned polls = 0;
    unsigned long long t0 = 0ull;
    while (d41x_xc_ld(done) < e) {
        if (d41x_xc_poll_abort(c, ab, polls, t0)) { ab_flag = 1; break; }
    }
    if (!ab_flag) asm volatile("fence.acquire.sys;" ::: "memory");
    if (rel_slot != nullptr && gridDim.x > 1) {
        d41x_xc_st_rel_gpu(rel_slot, ab_flag ? (e | 0x8000000000000000ull) : e);
    }
    return ab_flag;
}
static __device__ __forceinline__ int d41x_xc_gate_single(tp_dev_ctx* c, unsigned long long e,
                                                          unsigned long long* rel_slot) {
    return d41x_xc_gate_ptr(c, d41x_xc_flag(c, TP_F_CPU_DONE), e, rel_slot);
}

// Stage (world 2, fp32 payload, nbytes a 16 B multiple): reuse gate, copy the local partial into send[e % R], last
// block publishes tail + len tag + epoch, release-stores gpu_ready (the doorbell). = tp_gate_copy_signal_mb with
// src_f16 = 0 and world 2, plus the PDL prologue. No printf (ABI rule 7): a gate timeout only sets abort 11.
extern "C" __global__ void d41x_xc_stage(tp_dev_ctx* c, const void* src, unsigned nbytes, unsigned* arrive,
                                         unsigned long long* tr) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();   // the producer's partial is complete; the collect may become resident now
    const unsigned long long e = c->epoch + 1;   // every block reads the pre-kernel counter
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n16 = nbytes >> 4;
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = s4[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's payload stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // the next stage waits for this grid (PDL) first
            c->epoch = e;                               // advances even on abort (I9 no-op)
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;                 // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);     // the doorbell (I1)
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

// Collect (world 2): wait cpu_done >= e (I5), then buf = lower + upper in place over n floats (n % 4 == 0),
// rank 0's partial first (= tp_wait_add mode 2). `per` = float4s per thread per pass (all loads before the adds).
template <int PER>
static __device__ __forceinline__ void d41x_xc_sum(float* buf, const float* peer, int n4, int lower) {
    float4* b4 = (float4*)buf;
    const float4* p4 = (const float4*)peer;
    const int stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * PER) {
        float4 lo[PER], pe[PER];
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) { lo[k] = b4[i]; pe[k] = d41x_xc_ldcv4(p4 + i); }
        }
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a = lower ? lo[k] : pe[k];
                const float4 b = lower ? pe[k] : lo[k];
                b4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }
    }
}

extern "C" __global__ void d41x_xc_collect(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                           unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();   // the stage completed (its epoch store is visible); the consumer may become resident now
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;                                   // I9: no-op through the stream
    const float* peer = (const float*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    d41x_xc_sum<4>(buf, peer, n >> 2, c->rank == 0);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// Clock calibration (trace only): one thread writes %globaltimer into a mapped host word for `ns` nanoseconds while
// the host samples it between two CLOCK_MONOTONIC_RAW reads (xchg.rs `calibrate`).
extern "C" __global__ void d41x_xc_clock(unsigned long long* out, unsigned long long ns) {
    const unsigned long long t0 = d41x_xc_now();
    unsigned long long t = t0;
    while (t - t0 < ns) {
        t = d41x_xc_now();
        asm volatile("st.relaxed.sys.b64 [%0], %1;" :: "l"(out), "l"(t) : "memory");
    }
}

// =============================================================================
// leg PX2: the world-4 pair on the uniform one-shot single-stage ctx (rdma.rs `Single`). Copies of the shared
// `tp_gate_copy_signal_os_mb` and `tp_wait_add_4way_f32` (kernels/gpu_batch/01_norms_tp_transport.cuh, untouched),
// with the world-2 pair's changes: the stage is a PDL dependent (waits for its producer, releases the collect); the
// collect is a PDL dependent of the stage (releases its consumer), polls every peer's gate without sleeping (abort
// word every 1,024 polls) and sums over several blocks with float4 loads, every load of a thread before its adds.
// Same protocol (I1-I9: reuse gate on the qp_mask peers, sender-indexed recv rings, the proxy's per-peer validated
// cpu_done + the slot tail == e watermark check, R9), same wire bytes, and THE SAME ASSOCIATION per element:
//     out = (p0 + p2) + (p1 + p3)
// so TP4 output stays bit-identical to the shared pair (and to the two-stage tree it replaced).
// =============================================================================

static __device__ __forceinline__ unsigned long long* d41x_xc_nway(tp_dev_ctx* c, int peer, size_t sub_off) {
    return (unsigned long long*)((char*)c->nway_flags + (size_t)peer * TP_CL + sub_off);
}
static __device__ __forceinline__ const unsigned char* d41x_xc_os_slot(tp_dev_ctx* c, int sender, unsigned long long e) {
    return c->oneshot_recv + ((unsigned long long)sender * (unsigned long long)TP_RING_SLOTS
                              + (e & (TP_RING_SLOTS - 1))) * (unsigned long long)c->slot_stride;
}

// Stage (world 4, uniform one-shot ctx, nbytes % 16 == 0). = tp_gate_copy_signal_os_mb + the PDL prologue; no printf.
extern "C" __global__ void d41x_xc4_stage(tp_dev_ctx* c, const void* src, unsigned nbytes, unsigned* arrive,
                                          unsigned long long* tr) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;              // only the QPs e-R was posted to
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        const unsigned n16 = nbytes >> 4;
        const unsigned stride = gridDim.x * blockDim.x;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = s4[i];
        if (nbytes & 15u) {
            const unsigned n4 = nbytes >> 2;
            unsigned* dst4 = (unsigned*)slot;
            const unsigned* src4 = (const unsigned*)src;
            for (unsigned i = (n16 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst4[i] = src4[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)             // a one-shot epoch: posted to ALL peers' QPs
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

template <int PER>
static __device__ __forceinline__ void d41x_xc4_sum(float* buf, const float* q0, const float* q1, const float* q2,
                                                    const float* q3, int n4) {
    float4* b4 = (float4*)buf;
    const float4* P[4] = {(const float4*)q0, (const float4*)q1, (const float4*)q2, (const float4*)q3};
    const int stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * PER) {
        float4 v[PER][4];
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
#pragma unroll
                for (int p = 0; p < 4; p++) v[k][p] = (P[p] == (const float4*)buf) ? b4[i] : d41x_xc_ldcv4(P[p] + i);
            }
        }
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a0 = v[k][0], a1 = v[k][1], a2 = v[k][2], a3 = v[k][3];
                b4[i] = make_float4((a0.x + a2.x) + (a1.x + a3.x), (a0.y + a2.y) + (a1.y + a3.y),
                                    (a0.z + a2.z) + (a1.z + a3.z), (a0.w + a2.w) + (a1.w + a3.w));
            }
        }
    }
}

// Collect (world 4, uniform one-shot ctx, v1 receive = the proxy validates every peer's tail under one fence
// and publishes the shared TP_F_CPU_DONE watermark): single-CTA C2C poll on TP_F_CPU_DONE (no tail hammering
// during DMA), L2 handoff to sibling blocks, then buf = (p0 + p2) + (p1 + p3) in place over n floats (n % 4 == 0).
extern "C" __global__ void d41x_xc4_collect(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                            unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;                                   // I9
    const float* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum<2>(buf, q[0], q[1], q[2], q[3], n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// =============================================================================
// leg TX1: the dual-rail world-4 pair (kernels below; host side xchg.rs `exchange_dual`, grid.y = rail).
// =============================================================================

// leg TX1: the dual-rail stage (world 4): two d41x_xc4_stage grids in one launch, one per rail (--dsv41-dual-rail).
// Copy of d41x_xc4_stage below with the per-rail parameter selection above; the trace ring is not stamped (tr = 0).
extern "C" __global__ void d41x_xc4d_stage(tp_dev_ctx* c1, tp_dev_ctx* c2, const void* src1, const void* src2, unsigned nb1, unsigned nb2,
                                           unsigned* arrive2, unsigned long long* tr) {
    // grid.y selects the rail: y = 0 stages half 1 on c1, y = 1 half 2 on c2; each y group is a complete stage grid (its own
    // arrival counter arrive2[y]), so the body below is d41x_xc4_stage verbatim per group
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const void* src = blockIdx.y ? src2 : src1;
    const unsigned nbytes = blockIdx.y ? nb2 : nb1;
    unsigned* arrive = arrive2 + blockIdx.y;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;              // only the QPs e-R was posted to
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        const unsigned n16 = nbytes >> 4;
        const unsigned stride = gridDim.x * blockDim.x;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = s4[i];
        if (nbytes & 15u) {
            const unsigned n4 = nbytes >> 2;
            unsigned* dst4 = (unsigned*)slot;
            const unsigned* src4 = (const unsigned*)src;
            for (unsigned i = (n16 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst4[i] = src4[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)             // a one-shot epoch: posted to ALL peers' QPs
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

// leg TX1 + Phase 1A: the dual-rail collect (world 4): one CTA per rail polls TP_F_CPU_DONE over C2C and wakes
// sibling blocks via rel2[blockIdx.y] in GPU L2, then buf = (p0 + p2) + (p1 + p3) in place over n floats.
extern "C" __global__ void d41x_xc4d_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2, int n1, int n2,
                                             unsigned long long* tr, unsigned long long* rel2) {
    // grid.y selects the rail: each y group collects its own half on its own ctx (the two waits and the two copy-outs overlap)
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;                                   // I9
    const float* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum<2>(buf, q[0], q[1], q[2], q[3], n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// =============================================================================
// Phase 2 / TP=8: world-8 single-stage collect (single-rail + dual-rail fused, grid.y = rail).
// Uses d41x_xc4_stage / d41x_xc4d_stage verbatim (both already parameterize on c->world).
// Canonical sum order: ((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7)) — bit-identical to
// the 3-round world-8 recursive-doubling tree (`tp_xport::reduce_pipe` / `rd_tree_reference(8, 0)`).
// =============================================================================

template <int PER>
static __device__ __forceinline__ void d41x_xc8_sum(float* buf, const float* const q[8], int n4) {
    float4* b4 = (float4*)buf;
    const float4* P[8];
#pragma unroll
    for (int p = 0; p < 8; p++) P[p] = (const float4*)q[p];
    const int stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * PER) {
        float4 v[PER][8];
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
#pragma unroll
                for (int p = 0; p < 8; p++) v[k][p] = (P[p] == (const float4*)buf) ? b4[i] : d41x_xc_ldcv4(P[p] + i);
            }
        }
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a0 = v[k][0], a1 = v[k][1], a2 = v[k][2], a3 = v[k][3];
                const float4 a4 = v[k][4], a5 = v[k][5], a6 = v[k][6], a7 = v[k][7];
                b4[i] = make_float4(
                    ((a0.x + a2.x) + (a4.x + a6.x)) + ((a1.x + a3.x) + (a5.x + a7.x)),
                    ((a0.y + a2.y) + (a4.y + a6.y)) + ((a1.y + a3.y) + (a5.y + a7.y)),
                    ((a0.z + a2.z) + (a4.z + a6.z)) + ((a1.z + a3.z) + (a5.z + a7.z)),
                    ((a0.w + a2.w) + (a4.w + a6.w)) + ((a1.w + a3.w) + (a5.w + a7.w)));
            }
        }
    }
}

extern "C" __global__ void d41x_xc8_collect(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                            unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum<1>(buf, q, n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc8d_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2, int n1, int n2,
                                             unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum<1>(buf, q, n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// leg PX2 (`--probe-dsv41-xchg`): the device's view of tp_dev_ctx (sizeof, then the offsets of the fields the probe's
// host mirror fills), so the probe refuses a mirror that drifted from native/tp_doorbell.h.
extern "C" __global__ void d41x_xc_ctx_layout(unsigned* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out[0] = (unsigned)sizeof(tp_dev_ctx);
    out[1] = (unsigned)offsetof(tp_dev_ctx, epoch);        out[2] = (unsigned)offsetof(tp_dev_ctx, flags);
    out[3] = (unsigned)offsetof(tp_dev_ctx, send_ring);    out[4] = (unsigned)offsetof(tp_dev_ctx, recv_ring);
    out[5] = (unsigned)offsetof(tp_dev_ctx, gpu_ts);       out[6] = (unsigned)offsetof(tp_dev_ctx, slot_stride);
    out[7] = (unsigned)offsetof(tp_dev_ctx, payload_bytes); out[8] = (unsigned)offsetof(tp_dev_ctx, rank);
    out[9] = (unsigned)offsetof(tp_dev_ctx, fp32_payload); out[10] = (unsigned)offsetof(tp_dev_ctx, gate_waits);
    out[11] = (unsigned)offsetof(tp_dev_ctx, len_local);   out[12] = (unsigned)offsetof(tp_dev_ctx, world);
    out[13] = (unsigned)offsetof(tp_dev_ctx, rounds);      out[14] = (unsigned)offsetof(tp_dev_ctx, nway_flags);
    out[15] = (unsigned)offsetof(tp_dev_ctx, nway_recv);   out[16] = (unsigned)offsetof(tp_dev_ctx, oneshot_recv);
    out[17] = (unsigned)offsetof(tp_dev_ctx, oneshot);     out[18] = (unsigned)offsetof(tp_dev_ctx, qp_mask);
}

// =============================================================================
// leg PX2: wide frames at world 4 as a reduce-scatter + all-gather that KEEPS the association (p0 + p2) + (p1 + p3).
// Rank r owns quarter r of the frame (half bit1(r), quarter bit0(r) inside it). Reduce-scatter = two epochs of the
// primary link's recursive-doubling tree (its partner schedule: an odd epoch pairs r with r ^ 2, the next one with
// r ^ 1; R10 keeps every reduce starting on an odd epoch): epoch A sends the half the partner keeps and adds the
// partner's copy of OUR half (lower rank of the pair first: p0 + p2, p1 + p3); epoch B sends the quarter of our half the
// partner keeps and adds into our quarter ((p0 + p2) + (p1 + p3), the lower rank first). All-gather = ONE all-peers
// epoch of the single-stage ctx: every rank sends its summed quarter, and copies (no arithmetic: -0.0 stays -0.0) the
// three other quarters into place. Per element the same two adds in the same order as the one-stage collect.
// The tree pair below is a copy of the shared tp_gate_copy_signal_mb / tp_wait_add_mb world>2 arms (fp32, lag 0) with
// the PDL prologue, the no-sleep gate poll and the float4 multi-block body.
// =============================================================================

static __device__ __forceinline__ int d41x_xct_partner(tp_dev_ctx* c, unsigned long long e) {
    return (int)c->rank ^ (1 << (int)(e % (unsigned long long)c->rounds));
}
static __device__ __forceinline__ const unsigned char* d41x_xct_slot(tp_dev_ctx* c, unsigned long long e) {
    const unsigned long long r = (unsigned long long)c->rounds;
    const unsigned long long round = e % r;
    const unsigned long long slot = (e / r) & (TP_RING_SLOTS - 1);
    return c->nway_recv + (round * (unsigned long long)TP_RING_SLOTS + slot) * (unsigned long long)c->slot_stride;
}

// Tree stage (world > 2, primary link): = tp_gate_copy_signal_mb's world>2 arm (fp32 bytes) + the PDL prologue.
extern "C" __global__ void d41x_xct_stage(tp_dev_ctx* c, const void* src, unsigned nbytes, unsigned* arrive,
                                          unsigned long long* tr) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_nway(c, d41x_xct_partner(c, tgt), TP_NWAY_TX_OFF);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        const unsigned n16 = nbytes >> 4;
        const unsigned stride = gridDim.x * blockDim.x;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = s4[i];
        if (nbytes & 15u) {
            const unsigned n4 = nbytes >> 2;
            unsigned* dst4 = (unsigned*)slot;
            const unsigned* src4 = (const unsigned*)src;
            for (unsigned i = (n16 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst4[i] = src4[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->qp_mask) c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << d41x_xct_partner(c, e);   // a tree epoch
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

// Tree collect (world > 2, primary link): proxy validates partner(e)'s tail under SEQ_CST before storing
// cpu_done[partner] = e, so single-CTA C2C polling on cpu_done[partner] + L2 handoff via `rel` suffices.
// Then dst[i] = lower + upper over n floats in place (lower = the pair's rank whose bit round(e) is 0).
extern "C" __global__ void d41x_xct_collect(tp_dev_ctx* c, float* dst, int n, unsigned long long* tr,
                                            unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned char* slot = d41x_xct_slot(c, e);
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_ptr(c, d41x_xc_nway(c, d41x_xct_partner(c, e), TP_NWAY_CPU_OFF), e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int lower = ((int)c->rank & (1 << (int)(e % (unsigned long long)c->rounds))) == 0;
    d41x_xc_sum<4>(dst, (const float*)slot, n >> 2, lower);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// Phase 1D: dual-rail tree stage (world > 2, grid.y = rail: y=0 on c1, y=1 on c2).
extern "C" __global__ void d41x_xctd_stage(tp_dev_ctx* c1, tp_dev_ctx* c2, const void* src1, const void* src2,
                                           unsigned nb1, unsigned nb2, unsigned* arrive2, unsigned long long* tr) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const void* src = blockIdx.y ? src2 : src1;
    const unsigned nbytes = blockIdx.y ? nb2 : nb1;
    unsigned* arrive = arrive2 + blockIdx.y;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_nway(c, d41x_xct_partner(c, tgt), TP_NWAY_TX_OFF);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        const unsigned n16 = nbytes >> 4;
        const unsigned stride = gridDim.x * blockDim.x;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = s4[i];
        if (nbytes & 15u) {
            const unsigned n4 = nbytes >> 2;
            unsigned* dst4 = (unsigned*)slot;
            const unsigned* src4 = (const unsigned*)src;
            for (unsigned i = (n16 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst4[i] = src4[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->qp_mask) c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << d41x_xct_partner(c, e);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

// Phase 1D: dual-rail tree collect (world > 2, grid.y = rail, single-CTA C2C gate + L2 handoff via rel2[y]).
extern "C" __global__ void d41x_xctd_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* dst1, float* dst2,
                                             int n1, int n2, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* dst = blockIdx.y ? dst2 : dst1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned char* slot = d41x_xct_slot(c, e);
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_ptr(c, d41x_xc_nway(c, d41x_xct_partner(c, e), TP_NWAY_CPU_OFF), e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int lower = ((int)c->rank & (1 << (int)(e % (unsigned long long)c->rounds))) == 0;
    d41x_xc_sum<4>(dst, (const float*)slot, n >> 2, lower);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// All-gather collect (world 4, single-stage ctx): single-CTA C2C gate on TP_F_CPU_DONE + L2 handoff, then a plain
// copy of peer p's nq floats into buf + p * nq (no arithmetic). The stage is d41x_xc4_stage with src = buf + rank * nq.
extern "C" __global__ void d41x_xc4_gather(tp_dev_ctx* c, float* buf, int nq, unsigned long long* tr,
                                           unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < (int)c->world; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)p * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// Phase 1D: dual-rail all-gather collect (world 4, grid.y = rail, single-CTA C2C gate + L2 handoff via rel2[y]).
// Each rail y gathers its half's 3 remote quarters into buf_y + p * nq_y (no arithmetic: -0.0 preserved).
extern "C" __global__ void d41x_xc4d_gather(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                            int nq1, int nq2, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int nq = blockIdx.y ? nq2 : nq1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < (int)c->world; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)p * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// =============================================================================
// Phase A: Zero-CPU-bounce GPU-validated collects (TP2 / TP4 / TP8, single & dual rail),
// World-8 RS+AG gather (involution slot_w8(p)), and opt-in BF16 wire stage/collect.
// =============================================================================

static __device__ __forceinline__ int d41x_xc_slot_w8(int p) {
    return ((p & 2) << 1) | ((p & 4) >> 1) | (p & 1);
}

// Zero-CPU-bounce gate for world 2 (hybrid-compatible with Pipe: publishes TP_F_RX_DONE so the proxy skips CPU validation).
static __device__ __forceinline__ int d41x_xc_gate_gpu_w2(tp_dev_ctx* c, unsigned long long e, unsigned wire,
                                                          unsigned long long* rel_slot) {
    int ab_flag = 0;
    if (rel_slot != nullptr && gridDim.x > 1 && blockIdx.x > 0) {
        unsigned long long rv;
        do { rv = d41x_xc_ld_gpu(rel_slot); } while ((rv & 0x7fffffffffffffffull) < e);
        ab_flag = (int)(rv >> 63);
        if (!ab_flag) asm volatile("fence.acquire.sys;" ::: "memory");
        return ab_flag;
    }
    const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
    const unsigned char* slot = c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
    const unsigned long long* tail = (const unsigned long long*)(slot + wire);
    unsigned polls = 0;
    unsigned long long t0 = 0ull;
    while (d41x_xc_ld(tail) != e) {
        if (d41x_xc_poll_abort(c, ab, polls, t0)) { ab_flag = 1; break; }
    }
    if (!ab_flag) {
        asm volatile("fence.acquire.sys;" ::: "memory");
        d41x_xc_st_rel(d41x_xc_flag(c, TP_F_RX_DONE), e);
    }
    if (rel_slot != nullptr && gridDim.x > 1) {
        d41x_xc_st_rel_gpu(rel_slot, ab_flag ? (e | 0x8000000000000000ull) : e);
    }
    return ab_flag;
}

// Zero-CPU-bounce gate for oneshot_uniform (world 4 & world 8): CTA 0 polls all world-1 peer slot tails directly,
// runs fence.acquire.sys, publishes TP_F_RX_DONE for the watchdog, and wakes sibling CTAs via device-L2 `rel_slot`.
static __device__ __forceinline__ int d41x_xc_gate_gpu_os(tp_dev_ctx* c, unsigned long long e, unsigned wire,
                                                          unsigned long long* rel_slot) {
    int ab_flag = 0;
    if (rel_slot != nullptr && gridDim.x > 1 && blockIdx.x > 0) {
        unsigned long long rv;
        do { rv = d41x_xc_ld_gpu(rel_slot); } while ((rv & 0x7fffffffffffffffull) < e);
        ab_flag = (int)(rv >> 63);
        if (!ab_flag) asm volatile("fence.acquire.sys;" ::: "memory");
        return ab_flag;
    }
    const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
    unsigned polls = 0;
    unsigned long long t0 = 0ull;
    for (int p = 0; p < (int)c->world && !ab_flag; p++) {
        if (p == (int)c->rank) continue;
        const unsigned long long* tail = (const unsigned long long*)(d41x_xc_os_slot(c, p, e) + wire);
        while (d41x_xc_ld(tail) != e) {
            if (d41x_xc_poll_abort(c, ab, polls, t0)) { ab_flag = 1; break; }
        }
    }
    if (!ab_flag) {
        asm volatile("fence.acquire.sys;" ::: "memory");
        d41x_xc_st_rel(d41x_xc_flag(c, TP_F_RX_DONE), e);
    }
    if (rel_slot != nullptr && gridDim.x > 1) {
        d41x_xc_st_rel_gpu(rel_slot, ab_flag ? (e | 0x8000000000000000ull) : e);
    }
    return ab_flag;
}

extern "C" __global__ void d41x_xc_collect_g(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                             unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_w2(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* peer = (const float*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    d41x_xc_sum<4>(buf, peer, n >> 2, c->rank == 0);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

// R3: Fused dual-rail world-2 stage & collect (grid.y = 2: rail 0 / rail 1 in one launch, removing serial PDL dep).
extern "C" __global__ void d41x_xcd_stage(tp_dev_ctx* c1, tp_dev_ctx* c2, const void* src1, const void* src2,
                                          unsigned nb1, unsigned nb2, unsigned* arrive2, unsigned long long* tr) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const void* src = blockIdx.y ? src2 : src1;
    const unsigned nbytes = blockIdx.y ? nb2 : nb1;
    unsigned* arrive = arrive2 + blockIdx.y;
    unsigned long long* tr_r = blockIdx.y ? nullptr : tr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n16 = nbytes >> 4;
        uint4* dst = (uint4*)slot;
        const uint4* s4 = (const uint4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = s4[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

extern "C" __global__ void d41x_xcd_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                            int n1, int n2, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    unsigned long long* tr_r = blockIdx.y ? nullptr : tr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* peer = (const float*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    d41x_xc_sum<4>(buf, peer, n >> 2, c->rank == 0);
    if (tr_r) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xcd_collect_g(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                              int n1, int n2, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    unsigned long long* tr_r = blockIdx.y ? nullptr : tr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_w2(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* peer = (const float*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    d41x_xc_sum<4>(buf, peer, n >> 2, c->rank == 0);
    if (tr_r) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc4_collect_g(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                              unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum<2>(buf, q[0], q[1], q[2], q[3], n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc4d_collect_g(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2, int n1, int n2,
                                               unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum<2>(buf, q[0], q[1], q[2], q[3], n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc8_collect_g(tp_dev_ctx* c, float* buf, int n, unsigned long long* tr,
                                              unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum<1>(buf, q, n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc8d_collect_g(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2, int n1, int n2,
                                               unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const float* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum<1>(buf, q, n >> 2);
    if (tr) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_END0, d41x_xc_now());
        if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) d41x_xc_stamp(tr, e, D41X_XC_TR_C_ENDL, d41x_xc_now());
    }
}

extern "C" __global__ void d41x_xc4_gather_g(tp_dev_ctx* c, float* buf, int nq, unsigned long long* tr,
                                             unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)nq * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < (int)c->world; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)p * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
}

extern "C" __global__ void d41x_xc4d_gather_g(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                              int nq1, int nq2, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int nq = blockIdx.y ? nq2 : nq1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)nq * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = d41x_xc_gate_gpu_os(c, e, wire, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < (int)c->world; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)p * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
}

extern "C" __global__ void d41x_xc8_gather(tp_dev_ctx* c, float* buf, int nq, int gpu_recv,
                                           unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)nq * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < 8; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)d41x_xc_slot_w8(p) * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
}

extern "C" __global__ void d41x_xc8d_gather(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                            int nq1, int nq2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int nq = blockIdx.y ? nq2 : nq1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)nq * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int nq4 = nq >> 2;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < 8; p++) {
        if (p == (int)c->rank) continue;
        const float4* src = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dst = (float4*)(buf + (size_t)d41x_xc_slot_w8(p) * nq);
        for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nq4; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) v[k] = d41x_xc_ldcv4(src + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const int i = i0 + k * stride; if (i < nq4) dst[i] = v[k]; }
        }
    }
}

// =============================================================================
// Phase A (opt-in --dsv41-wire-bf16 on): BF16 wire compression for TP2 / TP4 / TP8.
// =============================================================================

static __device__ __forceinline__ uint2 d41x_f4_to_b16x4(float4 v) {
    __nv_bfloat162 a = __float22bfloat162_rn(make_float2(v.x, v.y));
    __nv_bfloat162 b = __float22bfloat162_rn(make_float2(v.z, v.w));
    uint2 u;
    u.x = *reinterpret_cast<unsigned*>(&a);
    u.y = *reinterpret_cast<unsigned*>(&b);
    return u;
}
static __device__ __forceinline__ float4 d41x_b16x4_to_f4(uint2 u) {
    __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
    __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    return make_float4(fa.x, fa.y, fb.x, fb.y);
}
static __device__ __forceinline__ uint2 d41x_xc_ldcv2_u32(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.cv.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

extern "C" __global__ void d41x_xc_stage_b16(tp_dev_ctx* c, const float* src, unsigned nfloats, unsigned* arrive,
                                             unsigned long long* tr) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n4 = nfloats >> 2;
        uint2* dst = (uint2*)slot;
        const float4* s4 = (const float4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
            dst[i] = d41x_f4_to_b16x4(s4[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = ((nfloats * 2u) + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

extern "C" __global__ void d41x_xc4_stage_b16(tp_dev_ctx* c, const float* src, unsigned nfloats, unsigned* arrive,
                                              unsigned long long* tr) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n4 = nfloats >> 2;
        uint2* dst = (uint2*)slot;
        const float4* s4 = (const float4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
            dst[i] = d41x_f4_to_b16x4(s4[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = ((nfloats * 2u) + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

extern "C" __global__ void d41x_xc4d_stage_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, const float* src1, const float* src2,
                                               unsigned nf1, unsigned nf2, unsigned* arrive2, unsigned long long* tr) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const float* src = blockIdx.y ? src2 : src1;
    const unsigned nfloats = blockIdx.y ? nf2 : nf1;
    unsigned* arrive = arrive2 + blockIdx.y;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n4 = nfloats >> 2;
        uint2* dst = (uint2*)slot;
        const float4* s4 = (const float4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
            dst[i] = d41x_f4_to_b16x4(s4[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = ((nfloats * 2u) + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

extern "C" __global__ void d41x_xc_collect_b16(tp_dev_ctx* c, float* buf, int n, int gpu_recv,
                                               unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_w2(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* p2 = (const uint2*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    float4* b4 = (float4*)buf;
    const int n4 = n >> 2, lower = (c->rank == 0), stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * 4) {
        float4 lo[4], pe[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * stride;
            if (i < n4) { lo[k] = d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i])); pe[k] = d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(p2 + i)); }
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a = lower ? lo[k] : pe[k];
                const float4 b = lower ? pe[k] : lo[k];
                b4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }
    }
}

extern "C" __global__ void d41x_xcd_stage_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, const float* src1, const float* src2,
                                              unsigned nf1, unsigned nf2, unsigned* arrive2, unsigned long long* tr) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const float* src = blockIdx.y ? src2 : src1;
    const unsigned nfloats = blockIdx.y ? nf2 : nf1;
    unsigned* arrive = arrive2 + blockIdx.y;
    unsigned long long* tr_r = blockIdx.y ? nullptr : tr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned long long* ret = d41x_xc_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
            const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (d41x_xc_ld(ret) < tgt) {
                if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
        if (blockIdx.x == 0) { d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_IN, t_in); d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_GO, d41x_xc_now()); }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n4 = nfloats >> 2;
        uint2* dst = (uint2*)slot;
        const float4* s4 = (const float4*)src;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
            dst[i] = d41x_f4_to_b16x4(s4[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                const unsigned wire = ((nfloats * 2u) + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
            d41x_xc_stamp(tr_r, e, D41X_XC_TR_S_DB, d41x_xc_now());
        }
    }
}

extern "C" __global__ void d41x_xcd_collect_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                                int n1, int n2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    unsigned long long* tr_r = blockIdx.y ? nullptr : tr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_w2(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr_r, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* p2 = (const uint2*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    float4* b4 = (float4*)buf;
    const int n4 = n >> 2, lower = (c->rank == 0), stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * 4) {
        float4 lo[4], pe[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * stride;
            if (i < n4) { lo[k] = d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i])); pe[k] = d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(p2 + i)); }
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a = lower ? lo[k] : pe[k];
                const float4 b = lower ? pe[k] : lo[k];
                b4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }
    }
}

static __device__ __forceinline__ void d41x_xc4_sum_b16(float* buf, const uint2* const P[4], int rank, int n4) {
    float4* b4 = (float4*)buf;
    const int stride = gridDim.x * blockDim.x;
    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n4; i0 += stride * 2) {
        float4 v[2][4];
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
#pragma unroll
                for (int p = 0; p < 4; p++)
                    v[k][p] = (p == rank) ? d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i])) : d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(P[p] + i));
            }
        }
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int i = i0 + k * stride;
            if (i < n4) {
                const float4 a0 = v[k][0], a1 = v[k][1], a2 = v[k][2], a3 = v[k][3];
                b4[i] = make_float4((a0.x + a2.x) + (a1.x + a3.x), (a0.y + a2.y) + (a1.y + a3.y),
                                    (a0.z + a2.z) + (a1.z + a3.z), (a0.w + a2.w) + (a1.w + a3.w));
            }
        }
    }
}

static __device__ __forceinline__ void d41x_xc8_sum_b16(float* buf, const uint2* const P[8], int rank, int n4) {
    float4* b4 = (float4*)buf;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 a[8];
#pragma unroll
        for (int p = 0; p < 8; p++)
            a[p] = (p == rank) ? d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i])) : d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(P[p] + i));
        b4[i] = make_float4(
            ((a[0].x + a[2].x) + (a[4].x + a[6].x)) + ((a[1].x + a[3].x) + (a[5].x + a[7].x)),
            ((a[0].y + a[2].y) + (a[4].y + a[6].y)) + ((a[1].y + a[3].y) + (a[5].y + a[7].y)),
            ((a[0].z + a[2].z) + (a[4].z + a[6].z)) + ((a[1].z + a[3].z) + (a[5].z + a[7].z)),
            ((a[0].w + a[2].w) + (a[4].w + a[6].w)) + ((a[1].w + a[3].w) + (a[5].w + a[7].w)));
    }
}

extern "C" __global__ void d41x_xc4_collect_b16(tp_dev_ctx* c, float* buf, int n, int gpu_recv,
                                                unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum_b16(buf, q, (int)c->rank, n >> 2);
}

extern "C" __global__ void d41x_xc4d_collect_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                                 int n1, int n2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* q[4];
#pragma unroll
    for (int p = 0; p < 4; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xc4_sum_b16(buf, q, (int)c->rank, n >> 2);
}

extern "C" __global__ void d41x_xc8_collect_b16(tp_dev_ctx* c, float* buf, int n, int gpu_recv,
                                                unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum_b16(buf, q, (int)c->rank, n >> 2);
}


extern "C" __global__ void d41x_xc8d_collect_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                                 int n1, int n2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const uint2* q[8];
#pragma unroll
    for (int p = 0; p < 8; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xc8_sum_b16(buf, q, (int)c->rank, n >> 2);
}

// =============================================================================
// Velo-Fabric Arbitrary-W (TP3 / TP6 / general W in 3..=8) + 1-RTT All-to-All
// (DeepSpeed-Ulysses sequence-parallel attention, Jacobs et al. arXiv:2309.14509,
//  and 1-RTT direct Reduce-Scatter / All-Gather for non-power-of-two worlds).
// =============================================================================

static __device__ __forceinline__ void d41x_xcn_sum_b16(float* buf, const uint2* const P[8], int world, int rank, int n4) {
    float4* b4 = (float4*)buf;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 acc = (0 == rank) ? d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i]))
                                 : d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(P[0] + i));
        for (int p = 1; p < world; p++) {
            float4 v = (p == rank) ? d41x_b16x4_to_f4(d41x_f4_to_b16x4(b4[i]))
                                   : d41x_b16x4_to_f4(d41x_xc_ldcv2_u32(P[p] + i));
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        b4[i] = acc;
    }
}

extern "C" __global__ void d41x_xcn_collect_b16(tp_dev_ctx* c, float* buf, int n, int gpu_recv,
                                                unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int w = (int)c->world;
    const uint2* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_b16(buf, q, w, (int)c->rank, n >> 2);
}

extern "C" __global__ void d41x_xcnd_collect_b16(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                                 int n1, int n2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (((unsigned)n * 2u) + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int w = (int)c->world;
    const uint2* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? nullptr : (const uint2*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_b16(buf, q, w, (int)c->rank, n >> 2);
}

static __device__ __forceinline__ void d41x_xcn_sum_f32(float* buf, const float* const P[8], int world, int rank, int n) {
    float4* b4 = (float4*)buf;
    const int n4 = n >> 2;
    const int stride = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n4; i += stride) {
        float4 acc = (0 == rank) ? b4[i] : d41x_xc_ldcv4((const float4*)P[0] + i);
        for (int p = 1; p < world; p++) {
            float4 v = (p == rank) ? b4[i] : d41x_xc_ldcv4((const float4*)P[p] + i);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        b4[i] = acc;
    }
    if (n & 3) {
        for (int i = (n4 << 2) + tid; i < n; i += stride) {
            float acc = (0 == rank) ? buf[i] : d41x_xc_ldcv1(P[0] + i);
            for (int p = 1; p < world; p++) {
                float v = (p == rank) ? buf[i] : d41x_xc_ldcv1(P[p] + i);
                acc += v;
            }
            buf[i] = acc;
        }
    }
}

extern "C" __global__ void d41x_xcn_collect(tp_dev_ctx* c, float* buf, int n, int gpu_recv,
                                            unsigned long long* tr, unsigned long long* rel) {
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int w = (int)c->world;
    const float* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_f32(buf, q, w, (int)c->rank, n);
}

extern "C" __global__ void d41x_xcnd_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* buf1, float* buf2,
                                             int n1, int n2, int gpu_recv, unsigned long long* tr, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* buf = blockIdx.y ? buf2 : buf1;
    const int n = blockIdx.y ? n2 : n1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    const unsigned long long t_in = d41x_xc_now();
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = ((unsigned)n * 4u + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) { d41x_xc_stamp(tr, e, D41X_XC_TR_C_IN, t_in); d41x_xc_stamp(tr, e, D41X_XC_TR_C_GO, d41x_xc_now()); }
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
        if (blockIdx.x == 0) d41x_xc_stamp(tr, e, D41X_XC_TR_C_SEEN, d41x_xc_now());
    }
    __syncthreads();
    if (s_ab) return;
    const int w = (int)c->world;
    const float* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? (const float*)buf : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_f32(buf, q, w, (int)c->rank, n);
}

// 1-RTT All-to-All stage (world >= 3, uniform oneshot ctx): packs peer p's slice from `src + p * peer_stride + off`
// (chunk_bytes, 16B aligned) into `send_slot(e) + p * a2a_stride`, copies own slice to `dst_self` if non-null, and
// publishes TP_LEN_TAG(e, wire) | TP_LEN_FLAG_A2A so the proxy RDMA-writes each peer's distinct slice directly into
// its sender-indexed recv slot.
static __device__ __forceinline__ void d41x_a2a_stage_body(tp_dev_ctx* c, const unsigned char* src, unsigned char* dst_self,
                                                           unsigned long long peer_stride, unsigned long long off,
                                                           unsigned chunk_bytes, unsigned* arrive) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    const unsigned wire = (chunk_bytes + 7u) & ~7u;
    const unsigned a2a_stride = (wire + TP_TAIL_BYTES + TP_CL - 1u) & ~(TP_CL - 1u);
    if (!s_ab) {
        const unsigned n16 = chunk_bytes >> 4;
        const unsigned stride = gridDim.x * blockDim.x;
        const int w = (int)c->world;
        const int rk = (int)c->rank;
        for (int p = 0; p < w; p++) {
            const uint4* sp = (const uint4*)(src + (size_t)p * peer_stride + off);
            uint4* dp = (p == rk) ? (uint4*)dst_self : (uint4*)(slot + (size_t)p * a2a_stride);
            if (!dp || dp == sp) continue;
            for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dp[i] = sp[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                for (int p = 0; p < (int)c->world; p++) {
                    if (p == (int)c->rank) continue;
                    *(unsigned long long*)(slot + (size_t)p * a2a_stride + wire) = e;
                }
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire) | TP_LEN_FLAG_A2A;
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

extern "C" __global__ void d41x_a2a_stage(tp_dev_ctx* c, const unsigned char* src, unsigned char* dst_self,
                                          unsigned long long peer_stride, unsigned long long off,
                                          unsigned chunk_bytes, unsigned* arrive) {
    d41x_a2a_stage_body(c, src, dst_self, peer_stride, off, chunk_bytes, arrive);
}

extern "C" __global__ void d41x_a2ad_stage(tp_dev_ctx* c1, tp_dev_ctx* c2, const unsigned char* src,
                                           unsigned char* dst_self1, unsigned char* dst_self2,
                                           unsigned long long peer_stride, unsigned long long off1, unsigned long long off2,
                                           unsigned ch1, unsigned ch2, unsigned* arrive2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    unsigned char* ds = blockIdx.y ? dst_self2 : dst_self1;
    unsigned long long off = blockIdx.y ? off2 : off1;
    unsigned ch = blockIdx.y ? ch2 : ch1;
    d41x_a2a_stage_body(c, src, ds, peer_stride, off, ch, arrive2 + blockIdx.y);
}

// 1-RTT All-to-All / All-Gather collect (world >= 3): copies each peer's recv slot into `dst + p * peer_stride + off`.
static __device__ __forceinline__ void d41x_a2a_collect_body(tp_dev_ctx* c, unsigned char* dst,
                                                             unsigned long long peer_stride, unsigned long long off,
                                                             unsigned chunk_bytes, int gpu_recv, unsigned long long* rel) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (chunk_bytes + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab) return;
    const unsigned n16 = chunk_bytes >> 4;
    const unsigned stride = gridDim.x * blockDim.x;
    const int w = (int)c->world;
    const int rk = (int)c->rank;
    for (int p = 0; p < w; p++) {
        if (p == rk) continue;
        const float4* sp = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dp = (float4*)(dst + (size_t)p * peer_stride + off);
        for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n16; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) v[k] = d41x_xc_ldcv4(sp + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) dp[i] = v[k]; }
        }
    }
}

extern "C" __global__ void d41x_a2a_collect(tp_dev_ctx* c, unsigned char* dst, unsigned long long peer_stride,
                                            unsigned long long off, unsigned chunk_bytes, int gpu_recv,
                                            unsigned long long* rel) {
    d41x_a2a_collect_body(c, dst, peer_stride, off, chunk_bytes, gpu_recv, rel);
}

extern "C" __global__ void d41x_a2ad_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, unsigned char* dst,
                                             unsigned long long peer_stride, unsigned long long off1, unsigned long long off2,
                                             unsigned ch1, unsigned ch2, int gpu_recv, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    unsigned long long off = blockIdx.y ? off2 : off1;
    unsigned ch = blockIdx.y ? ch2 : ch1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    d41x_a2a_collect_body(c, dst, peer_stride, off, ch, gpu_recv, rel);
}

// World-2 1-RTT All-to-All collect (copies peer's recv_ring slot directly to dst_peer without arithmetic).
extern "C" __global__ void d41x_a2a2_collect(tp_dev_ctx* c, unsigned char* dst_peer, unsigned chunk_bytes,
                                             int gpu_recv, unsigned long long* rel) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (chunk_bytes + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_w2(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab) return;
    const float4* sp = (const float4*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    float4* dp = (float4*)dst_peer;
    const unsigned n16 = chunk_bytes >> 4;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n16; i0 += stride * 4) {
        float4 v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) v[k] = d41x_xc_ldcv4(sp + i); }
#pragma unroll
        for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) dp[i] = v[k]; }
    }
}

extern "C" __global__ void d41x_a2a2d_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, unsigned char* dst1, unsigned char* dst2,
                                              unsigned nb1, unsigned nb2, int gpu_recv, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    unsigned char* dst_peer = blockIdx.y ? dst2 : dst1;
    const unsigned chunk_bytes = blockIdx.y ? nb2 : nb1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (chunk_bytes + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_w2(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab) return;
    const float4* sp = (const float4*)(c->recv_ring + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride);
    float4* dp = (float4*)dst_peer;
    const unsigned n16 = chunk_bytes >> 4;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n16; i0 += stride * 4) {
        float4 v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) v[k] = d41x_xc_ldcv4(sp + i); }
#pragma unroll
        for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) dp[i] = v[k]; }
    }
}

// 1-RTT Direct Reduce-Scatter collect (world >= 3, after d41x_a2a_stage / d41x_a2a_var_stage):
// sums all W ranks' slices for this rank in canonical order 0..W directly into `dst_my` (FP32).
// `wire` is the uniform slot wire length (align8(max_ch)) while `n` is this rank's actual float count (n*4 <= wire).
extern "C" __global__ void d41x_rsn_collect(tp_dev_ctx* c, float* dst_my, int n, unsigned wire, int gpu_recv,
                                            unsigned long long* rel) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab || n <= 0) return;
    const int w = (int)c->world;
    const float* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? (const float*)dst_my : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_f32(dst_my, q, w, (int)c->rank, n);
}

extern "C" __global__ void d41x_rsnd_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, float* dst1, float* dst2,
                                             int n1, int n2, unsigned w1, unsigned w2, int gpu_recv, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    float* dst_my = blockIdx.y ? dst2 : dst1;
    const int n = blockIdx.y ? n2 : n1;
    const unsigned wire = blockIdx.y ? w2 : w1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab || n <= 0) return;
    const int w = (int)c->world;
    const float* q[8];
    for (int p = 0; p < w; p++) q[p] = (p == (int)c->rank) ? (const float*)dst_my : (const float*)d41x_xc_os_slot(c, p, e);
    d41x_xcn_sum_f32(dst_my, q, w, (int)c->rank, n);
}

// Variable-block (uneven row counts, e.g. m % W != 0 at TP3 / TP6) 1-RTT All-to-All / RS / AG kernels.
struct TpSpan8 {
    unsigned long long off[8];
    unsigned len[8];
};

static __device__ __forceinline__ void d41x_a2a_var_stage_body(tp_dev_ctx* c, const unsigned char* src,
                                                               unsigned char* dst_self, TpSpan8 spn,
                                                               unsigned sub_off, unsigned max_ch, unsigned* arrive) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    const unsigned wire = (max_ch + 7u) & ~7u;
    const unsigned a2a_stride = (wire + TP_TAIL_BYTES + TP_CL - 1u) & ~(TP_CL - 1u);
    if (!s_ab) {
        const unsigned stride = gridDim.x * blockDim.x;
        const int w = (int)c->world;
        const int rk = (int)c->rank;
        for (int p = 0; p < w; p++) {
            const unsigned plen = spn.len[p];
            const unsigned avail = (sub_off < plen) ? (plen - sub_off) : 0u;
            const unsigned ch = (avail < max_ch) ? avail : max_ch;
            const unsigned n16 = ch >> 4;
            const uint4* sp = (const uint4*)(src + spn.off[p] + sub_off);
            uint4* dp = (p == rk) ? (uint4*)dst_self : (uint4*)(slot + (size_t)p * a2a_stride);
            if (!dp || dp == sp) continue;
            for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dp[i] = sp[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                for (int p = 0; p < (int)c->world; p++) {
                    if (p == (int)c->rank) continue;
                    *(unsigned long long*)(slot + (size_t)p * a2a_stride + wire) = e;
                }
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire) | TP_LEN_FLAG_A2A;
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

extern "C" __global__ void d41x_a2a_var_stage(tp_dev_ctx* c, const unsigned char* src, unsigned char* dst_self,
                                              TpSpan8 spn, unsigned sub_off, unsigned max_ch, unsigned* arrive) {
    d41x_a2a_var_stage_body(c, src, dst_self, spn, sub_off, max_ch, arrive);
}

extern "C" __global__ void d41x_a2ad_var_stage(tp_dev_ctx* c1, tp_dev_ctx* c2, const unsigned char* src,
                                               unsigned char* ds1, unsigned char* ds2, TpSpan8 spn,
                                               unsigned sub1, unsigned sub2, unsigned ch1, unsigned ch2, unsigned* arrive2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    unsigned char* ds = blockIdx.y ? ds2 : ds1;
    unsigned sub = blockIdx.y ? sub2 : sub1;
    unsigned ch = blockIdx.y ? ch2 : ch1;
    d41x_a2a_var_stage_body(c, src, ds, spn, sub, ch, arrive2 + blockIdx.y);
}

// Variable-block All-Gather collect (world >= 3, after d41x_xc4_stage / d41x_xc4d_stage of max_ch bytes):
// writes each peer p's actual byte count `min(max_ch, max(0, spn.len[p] - sub_off))` into `dst + spn.off[p] + sub_off`.
static __device__ __forceinline__ void d41x_ag_var_collect_body(tp_dev_ctx* c, unsigned char* dst, TpSpan8 spn,
                                                                unsigned sub_off, unsigned max_ch, int gpu_recv,
                                                                unsigned long long* rel) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch;
    const unsigned wire = (max_ch + 7u) & ~7u;
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = gpu_recv ? d41x_xc_gate_gpu_os(c, e, wire, rel) : d41x_xc_gate_single(c, e, rel);
    }
    __syncthreads();
    if (s_ab) return;
    const unsigned stride = gridDim.x * blockDim.x;
    const int w = (int)c->world;
    const int rk = (int)c->rank;
    for (int p = 0; p < w; p++) {
        if (p == rk) continue;
        const unsigned plen = spn.len[p];
        const unsigned avail = (sub_off < plen) ? (plen - sub_off) : 0u;
        const unsigned ch = (avail < max_ch) ? avail : max_ch;
        const unsigned n16 = ch >> 4;
        const float4* sp = (const float4*)d41x_xc_os_slot(c, p, e);
        float4* dp = (float4*)(dst + spn.off[p] + sub_off);
        for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n16; i0 += stride * 4) {
            float4 v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) v[k] = d41x_xc_ldcv4(sp + i); }
#pragma unroll
            for (int k = 0; k < 4; k++) { const unsigned i = i0 + k * stride; if (i < n16) dp[i] = v[k]; }
        }
    }
}

extern "C" __global__ void d41x_ag_var_collect(tp_dev_ctx* c, unsigned char* dst, TpSpan8 spn,
                                               unsigned sub_off, unsigned max_ch, int gpu_recv,
                                               unsigned long long* rel) {
    d41x_ag_var_collect_body(c, dst, spn, sub_off, max_ch, gpu_recv, rel);
}

extern "C" __global__ void d41x_agd_var_collect(tp_dev_ctx* c1, tp_dev_ctx* c2, unsigned char* dst, TpSpan8 spn,
                                                unsigned sub1, unsigned sub2, unsigned ch1, unsigned ch2,
                                                int gpu_recv, unsigned long long* rel2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    unsigned sub = blockIdx.y ? sub2 : sub1;
    unsigned ch = blockIdx.y ? ch2 : ch1;
    unsigned long long* rel = rel2 ? (rel2 + blockIdx.y) : nullptr;
    d41x_ag_var_collect_body(c, dst, spn, sub, ch, gpu_recv, rel);
}

// Stage for variable-block All-Gather: broadcasts rank's own slice (padded to max_ch bytes in send_slot when rank's
// slice is shorter than max_ch so all ranks use the identical wire length `max_ch`!).
static __device__ __forceinline__ void d41x_ag_var_stage_body(tp_dev_ctx* c, const unsigned char* src_my,
                                                              unsigned my_avail, unsigned max_ch, unsigned* arrive) {
    D41_PDL_ENTRY();
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            const unsigned qmask = c->qp_mask ? c->qp_mask[tgt & (TP_QPMASK_SLOTS - 1)] : 0u;
            const unsigned long long* ab = d41x_xc_flag(c, TP_F_ABORT);
            for (int p = 0; p < (int)c->world && !s_ab; p++) {
                if (!(qmask & (1u << p))) continue;
                const unsigned long long* ret = d41x_xc_nway(c, p, TP_NWAY_TX_OFF);
                if (blockIdx.x == 0 && d41x_xc_ld(ret) < tgt) c->gate_waits += 1;
                const unsigned long long deadline = d41x_xc_now() + K1_GATE_WAIT_NS;
                unsigned ns = 64, cap = 2048u;
                while (d41x_xc_ld(ret) < tgt) {
                    if (d41x_xc_ld(ab)) { s_ab = 1; break; }
                    if (d41x_xc_now() >= deadline) { d41x_xc_st_rel(d41x_xc_flag(c, TP_F_ABORT), 11); s_ab = 1; break; }
                    __nanosleep(ns);
                    if (ns < cap) ns <<= 1;
                }
            }
            if (!s_ab) asm volatile("fence.acquire.sys;" ::: "memory");
        }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    const unsigned wire = (max_ch + 7u) & ~7u;
    if (!s_ab) {
        const unsigned ch = (my_avail < max_ch) ? my_avail : max_ch;
        const unsigned n16 = ch >> 4;
        const uint4* sp = (const uint4*)src_my;
        uint4* dp = (uint4*)slot;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dp[i] = sp[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const unsigned prev = atomicAdd(arrive, 1u);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            *arrive = 0u;
            c->epoch = e;
            if (c->world > 2 && c->qp_mask)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = ((1u << c->world) - 1u) & ~(1u << c->rank);
            if (!d41x_xc_ld(d41x_xc_flag(c, TP_F_ABORT))) {
                *(unsigned long long*)(slot + wire) = e;
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                d41x_xc_st_rel(d41x_xc_flag(c, TP_F_GPU_READY), e);
            }
        }
    }
}

extern "C" __global__ void d41x_ag_var_stage(tp_dev_ctx* c, const unsigned char* src_my,
                                             unsigned my_avail, unsigned max_ch, unsigned* arrive) {
    d41x_ag_var_stage_body(c, src_my, my_avail, max_ch, arrive);
}

extern "C" __global__ void d41x_agd_var_stage(tp_dev_ctx* c1, tp_dev_ctx* c2,
                                              const unsigned char* s1, const unsigned char* s2,
                                              unsigned av1, unsigned av2, unsigned ch1, unsigned ch2, unsigned* arrive2) {
    tp_dev_ctx* c = blockIdx.y ? c2 : c1;
    const unsigned char* src = blockIdx.y ? s2 : s1;
    unsigned av = blockIdx.y ? av2 : av1;
    unsigned ch = blockIdx.y ? ch2 : ch1;
    d41x_ag_var_stage_body(c, src, av, ch, arrive2 + blockIdx.y);
}

// =============================================================================
// U6a: DSV4.1 LAGA Asymmetric SP Output All-to-All Pack / Unpack (BF16 mid)
// Packs head-sharded `wo_a` BF16 output `[nq, ng * olr]` into `[W][max_my_n][ng * olr]`
// for 1-RTT dual-rail A2A, and unpacks `[W][max_my_n][ng * olr]` -> `[my_n, W * ng * olr]`
// so each rank runs whole `wo_b` on `1/W` rows directly into BF16 `ao_own`.
// =============================================================================
struct D41LagaBlocks {
    int r0[8];
    int m[8];
};

struct D41LagaGroups {
    int g0[8];
    int ng[8];
};

extern "C" __global__ void d41x_laga_pack_b16(const uint4* __restrict__ mid, uint4* __restrict__ a2a_src,
                                              int max_my_n, int my_v4, int stride_v4, D41LagaBlocks blk) {
    D41_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= blk.m[p]) return;
    const uint4* src = mid + (size_t)(blk.r0[p] + t) * my_v4;
    uint4* dst = a2a_src + ((size_t)p * max_my_n + t) * stride_v4;
    for (int j = threadIdx.x; j < my_v4; j += blockDim.x) dst[j] = src[j];
}

extern "C" __global__ void d41x_laga_unpack_b16(const uint4* __restrict__ a2a_dst, uint4* __restrict__ mid_full,
                                                int max_my_n, int stride_v4, int all_v4, int olr_v4, D41LagaGroups grp) {
    D41_PDL_ENTRY();
    const int p = blockIdx.x;
    const int t = blockIdx.y;
    const int pv4 = grp.ng[p] * olr_v4;
    const int dst_off = grp.g0[p] * olr_v4;
    const uint4* src = a2a_dst + ((size_t)p * max_my_n + t) * stride_v4;
    uint4* dst = mid_full + (size_t)t * all_v4 + dst_off;
    for (int j = threadIdx.x; j < pv4; j += blockDim.x) dst[j] = src[j];
}
