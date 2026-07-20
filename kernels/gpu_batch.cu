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

#define GRID1(total) ((int)(((total) + 255) / 256))

// ---- batched RMSNorm (shared weight w[n]), one block per sequence column ----
extern "C" __global__ void rmsnorm_b(__nv_bfloat16* out, const __nv_bfloat16* x, const float* w, int n, int B, float eps) {
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
extern "C" __global__ void fused_res_rmsnorm_b(__nv_bfloat16* out, __nv_bfloat16* residual, const __nv_bfloat16* mixer,
                                                const float* w, int n, int B, float eps) {
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

// ---- elementwise over n*B elements ----
extern "C" __global__ void add_residual_b(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int total) {
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
//                            send[e%R] -> write the 8 B tail epoch -> RELEASE gpu_ready = e (I1)
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

// K1 — reuse gate, copy the local partial into the send ring, publish the watermark.
// `src` is the raw local partial (bf16 or fp32 — the copy is byte-agnostic, payload_bytes words of 4).
// `nbytes` = how much of the slot this barrier actually carries. The all-reduce is CHUNKED: a reduction
// larger than one ring slot (prefill reduces hidden * prompt_len, which is unbounded) is split into
// several barriers. A fixed length per call site is capture-safe — it is the EPOCH and the slot address
// that must never be kernel arguments, not the payload size.
extern "C" __global__ void tp_gate_copy_signal(tp_dev_ctx* c, const unsigned int* src, unsigned int nbytes) {
    if (threadIdx.x == 0) c->epoch += 1;      // single block on a single stream: no race
    __syncthreads();
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K1_IN);

    // I3 reuse gate: do not overwrite send[s] until the WR that shipped it R barriers ago has retired.
    // Only meaningful once the ring has wrapped once.
    if (e > TP_RING_SLOTS) {
        // Count the times the gate ACTUALLY binds. Without this the bench can only assume it was
        // exercised — and the rendezvous bounds skew to ~1 barrier, so under normal traffic it never
        // binds and the gate would ship untested (see the cq_hold hook in net_shim.c).
        if (threadIdx.x == 0 && tp_ld_relaxed(tp_flag(c, TP_F_TX_RETIRED)) < e - TP_RING_SLOTS)
            c->gate_waits += 1;
        if (tp_spin_until_ge(c, tp_flag(c, TP_F_TX_RETIRED), e - TP_RING_SLOTS, 0)) return;
    }

    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    unsigned int* dst = (unsigned int*)slot;
    const unsigned nw = (nbytes ? nbytes : c->payload_bytes) >> 2;
    for (unsigned i = threadIdx.x; i < nw; i += blockDim.x) dst[i] = src[i];

    __syncthreads();
    if (threadIdx.x == 0) {
        // Tail-epoch guard (R2d): written LAST, and shipped as the trailing 8 B of the same RDMA write,
        // so the peer proxy can prove placement order held before it releases its GPU.
        *(unsigned long long*)(slot + c->payload_bytes) = e;
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
// The mode is a PER-BARRIER kernel argument (constant per call site, so capture-safe — the round-3 rule
// forbids only epoch/slot args): decode/verify reductions run FP32-preserving while wide prefill
// reductions stay bf16-chunked, in the same run, on one ring. (A global mode in the ctx made that
// mix impossible: K2 would read bf16 prefill partials as fp32.)
extern "C" __global__ void tp_wait_add(tp_dev_ctx* c, __nv_bfloat16* out, const void* local, int n,
                                       int fp32_mode) {
    const unsigned long long e = c->epoch;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    tp_stamp(c, e, TP_GTS_K2_IN);

    if (tp_spin_until_ge(c, tp_flag(c, TP_F_CPU_DONE), e, 1)) return;   // I5 gate; abort => no-op (I9)
    tp_stamp(c, e, TP_GTS_K2_GO);

    const unsigned char* peer = c->recv_ring + (size_t)s * c->slot_stride;
    const int r0 = (c->rank == 0);
    if (fp32_mode) {
        const float* lo = (const float*)local;
        const float* pe = (const float*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = r0 ? lo[i] : pe[i];      // rank 0's partial
            float b = r0 ? pe[i] : lo[i];      // rank 1's partial
            out[i] = f2b(a + b);
        }
    } else {
        const __nv_bfloat16* lo = (const __nv_bfloat16*)local;
        const __nv_bfloat16* pe = (const __nv_bfloat16*)peer;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float a = b2f(r0 ? lo[i] : pe[i]);
            float b = b2f(r0 ? pe[i] : lo[i]);
            out[i] = f2b(a + b);
        }
    }
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
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    unsigned int* slot = (unsigned int*)(c->recv_ring + (size_t)s * c->slot_stride);
    const unsigned nw = c->payload_bytes >> 2;

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
extern "C" __global__ void rmsnorm_gated_b(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z, const float* w, int vd, int nh, int B, float eps) {
    int blk = blockIdx.x;
    int b = blk / nh;
    int head = blk % nh;
    extern __shared__ float s[];
    int tid = threadIdx.x;
    long long base = (long long)b * (nh * vd) + (long long)head * vd;
    float v = (tid < vd) ? b2f(x[base + tid]) : 0.0f;
    s[tid] = v * v;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) s[tid] += s[tid + s2]; __syncthreads(); }
    float inv = rsqrtf(s[0] / (float)vd + eps);
    if (tid < vd) out[base + tid] = f2b(v * inv * w[tid] * silu_f(b2f(z[base + tid])));
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
extern "C" __global__ void split_qgate_b(__nv_bfloat16* q, __nv_bfloat16* gate, const __nv_bfloat16* qg, int nh, int hd, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nh * hd;
    if (idx >= total) return;
    int b = idx / (nh * hd);
    int rem = idx % (nh * hd);
    int head = rem / hd;
    int d = rem % hd;
    long long qg_base = (long long)b * (nh * hd * 2) + (long long)head * (hd * 2);
    q[idx] = qg[qg_base + d];
    gate[idx] = qg[qg_base + hd + d];
}

// ---- batched depthwise causal conv1d step ----
// x: [conv_dim, B] bf16 (in/out); state: [B, conv_dim, k] f32; w: [conv_dim, k] f32
extern "C" __global__ void conv1d_b(__nv_bfloat16* x, float* state, const float* w, int conv_dim, int k, int B, const int* slot_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * conv_dim;
    if (idx >= total) return;
    int b = idx / conv_dim;
    int c = idx % conv_dim;
    int slot = slot_ids[b];
    float* st = state + ((long long)slot * conv_dim + c) * k;
    for (int j = 1; j < k; j++) st[j - 1] = st[j];
    st[k - 1] = b2f(x[(long long)b * conv_dim + c]);
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[c * k + j] * st[j];
    x[(long long)b * conv_dim + c] = f2b(silu_f(acc));
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
                                         int nh_packed, int kd, int vd,
                                         const float* a_log, const float* dt_bias,
                                         const int* slot_ids, unsigned long long* visits) {
    // nh_packed = n_value_heads | (n_key_heads << 16). cudarc caps kernel launches at 12 arguments and
    // the proof-build `visits` pointer took the last slot, so these two small counts share one.
    const int nh = nh_packed & 0xFFFF;
    const int n_k_heads = (nh_packed >> 16) & 0xFFFF;
    const int nchunk = vd / GDN_C;
    int blk = blockIdx.x;
    const int chunk = blk % nchunk;  blk /= nchunk;
    const int head  = blk % nh;      blk /= nh;
    const int b     = blk;
    if (visits && threadIdx.x == 0) atomicAdd(&visits[head], 1ull);
    const int key_head = head * n_k_heads / nh;   // map value head -> key head
    const int key_dim = n_k_heads * kd;
    const int conv_dim = key_dim * 2 + nh * vd;
    const int bb0 = chunk * GDN_C;

    extern __shared__ float sh[];
    float* S_sh   = sh;                       // [kd][GDN_SP]
    float* Srow   = S_sh + kd * GDN_SP;       // [kd]
    float* kv_mem = Srow + kd;                // [GDN_C]
    float* vbuf   = kv_mem + GDN_C;
    float* delta  = vbuf + GDN_C;
    float* qrow   = delta + GDN_C;            // [kd]
    float* krow   = qrow + kd;                // [kd]

    const __nv_bfloat16* col = qkv + (long long)b * conv_dim;
    float* S = state + ((long long)slot_ids[b] * nh + head) * kd * vd;

    float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)b * nh + head])));
    float sp = b2f(a_in[(long long)b * nh + head]) + dt_bias[head];
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
    int nh, int kd_vd, const float* a_log, const float* dt_bias, int N_nkh, float* mid_s,
    const int* parent) {
    // `parent` is PACKED per column: low 16 bits = DFS parent (two's-complement int16, root = -1),
    // high 16 bits = the column's lane slot. This keeps the launch at 12 args (cudarc's tuple ceiling)
    // while routing per-column slot into the forest scan. null `parent` (prefill) => pre-offset `state`.
    const int kd = kd_vd & 0xFFFF;
    const int vd = kd_vd >> 16;
    const int N = N_nkh & 0xFFFFFF;
    const int n_k_heads = (N_nkh >> 24) & 0xFF;
    const int nchunk = vd / GDN_C;
    const int chunk = blockIdx.x % nchunk;
    const int head  = blockIdx.x / nchunk;
    const int key_head = head * n_k_heads / nh;
    const int key_dim = n_k_heads * kd;
    const int conv_dim = key_dim * 2 + nh * vd;
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
        const __nv_bfloat16* col = qkv + (long long)t * conv_dim;

        float beta = 1.0f / (1.0f + __expf(-b2f(b_in[(long long)t * nh + head])));
        float sp = b2f(a_in[(long long)t * nh + head]) + dt_bias[head];
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
// ONE BLOCK PER (QUERY TILE OF 8, query_head); blockDim.x = 256 = 8 warps; hd = 256 in this family.
// pos_start: absolute position of the first of the N tokens (0 for a from-scratch prompt prefill;
// for a causal-append -- the MTP head's prompt prime -- = the position the append starts at, so
// column i attends to KV[0 .. pos_start+i]).
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
// So: a block now owns EIGHT query positions (warp w takes query tile*8 + w) and the 8 warps sweep the
// SAME keys together. One K/V row fetch from L2 now serves 8 queries instead of 1 -- an 8x cut in the
// binding resource. Each warp carries the complete running softmax (m, l, acc[8]) for ITS OWN query in
// registers across the whole key range, so no warp ever needs another warp's partial: the cross-warp
// merge, and all of the kernel's shared memory, are simply GONE.
//
// Causality: warps in a block are within 8 positions of each other, so they diverge only over the last
// handful of keys -- masked with a predicate, not a branch out of the loop.
extern "C" __global__ void gqa_attn_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, int stride, int nh, int nkv, int hd, float scale, int N, int pos_start) {
    const int QT = 8;                          // query positions per block == warps per block
    const int blk = blockIdx.x;
    const int tile = blk / nh, qh = blk % nh;
    const int kvh = qh / (nh / nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int DPL = hd >> 5;                   // head dims per lane (8 when hd=256)

    const int t = tile * QT + warp;            // THIS warp's query position
    const bool active = (t < N);
    const int pc = pos_start + t + 1;          // keys this warp's query attends to
    // Keys ANY warp in this block needs: the block sweeps to the last query's horizon.
    const int tlast = min(tile * QT + QT - 1, N - 1);
    const int pc_blk = pos_start + tlast + 1;

    // This lane's slice of q: 8 contiguous dims = one 16-byte load.
    const __nv_bfloat16* qrow = q + (long long)(active ? t : 0) * (nh * hd) + (long long)qh * hd + lane * DPL;
    float qv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) qv[i] = b2f(qrow[i]);

    float m = -1e30f, l = 0.0f;
    float acc[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

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
        // All 8 warps issue this same address: one fetch, eight queries served.
        const __nv_bfloat16* krow = kb + (long long)tt * hd;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += qv[i] * b2f(krow[i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;                              // now uniform across the warp

        if (active && tt < pc) {                 // causal mask (differs only over the last <8 keys)
            const float m_new = fmaxf(m, s);
            const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
            const __nv_bfloat16* vrow = vb + (long long)tt * hd;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * a_old + a_cur * b2f(vrow[i]);
            m = m_new;
            l = l * a_old + a_cur;
        }
    }

    // A warp owns its query's ENTIRE softmax -- nothing to merge, nothing to share.
    if (active) {
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        __nv_bfloat16* orow = out + (long long)t * (nh * hd) + (long long)qh * hd + lane * DPL;
        #pragma unroll
        for (int i = 0; i < 8; i++) orow[i] = f2b(acc[i] * inv);
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
extern "C" __global__ void conv1d_prefill(__nv_bfloat16* out, const __nv_bfloat16* in,
                                          const float* state, const float* w,
                                          int conv_dim, int k, int N, float* mid_state,
                                          const int* win_src, const int* slot_ids) {
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
        const float w0 = (v0 < 0) ? st0[v0 + 4] : b2f(in[(long long)v0 * conv_dim + c]);
        const float w1 = (v1 < 0) ? st0[v1 + 4] : b2f(in[(long long)v1 * conv_dim + c]);
        const float w2 = (v2 < 0) ? st0[v2 + 4] : b2f(in[(long long)v2 * conv_dim + c]);
        const float w3 = (v3 < 0) ? st0[v3 + 4] : b2f(in[(long long)v3 * conv_dim + c]);
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
                         : b2f(in[(long long)v * conv_dim + c]);
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
extern "C" __global__ void conv1d_prefill_state(float* state, const __nv_bfloat16* in,
                                                int conv_dim, int k, int last_t, int lane_len,
                                                const int* slot_ids) {
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
        const float n0 = (off0 + 0 < 0) ? st[off0 + 4] : b2f(in[(base + 0) * conv_dim + c]);
        const float n1 = (off0 + 1 < 0) ? st[off0 + 5] : b2f(in[(base + 1) * conv_dim + c]);
        const float n2 = (off0 + 2 < 0) ? st[off0 + 6] : b2f(in[(base + 2) * conv_dim + c]);
        const float n3 = (off0 + 3 < 0) ? st[off0 + 7] : b2f(in[(base + 3) * conv_dim + c]);
        st[0] = n0; st[1] = n1; st[2] = n2; st[3] = n3;   // read-all-then-write-all
        return;
    }

    float nx[8];
    for (int j = 0; j < k; j++) {
        const int off = (lane_len - k) + j;      // offset from lane start; < 0 => committed tail
        nx[j] = (off < 0) ? st[off + k]
                          : b2f(in[(long long)(last_t - (k - 1) + j) * conv_dim + c]);
    }
    for (int j = 0; j < k; j++) st[j] = nx[j];   // read-all-then-write-all: no self-overlap hazard
}

// ===================== SPLIT-K ATTENTION (Flash-Decoding) =====================
// Parameterized for any Qwen3.5 model. hd=256 is constant across the family.
// nh and nkv are packed into nh_packed = nh * 1000 + nkv.

#define SK_HD 256
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
    const int* pos, int bs_packed, int nh_packed, const int* slot_ids,
    const unsigned char* path, const int* col_pos_start) {
    // scale = 1/sqrt(hd) with hd == SK_HD == 256 constant across the family; sqrtf(256)=16 exactly, so
    // 1/16 = 0.0625f is bit-identical to the host's old `scale` arg. Computed here to free a launch slot
    // for `col_pos_start` (12-arg cudarc ceiling). FOREST: `col_pos_start[b]` is column b's lane prefix
    // boundary; null => pos[0] (single lane / tree / decode) — byte-identical to pre-forest.
    const float scale = 1.0f / sqrtf((float)SK_HD);
    const int nh = nh_packed / 1000;
    const int nkv = nh_packed % 1000;
    const int gqa_ratio = nh / nkv;
    const int stride  = bs_packed & 0x7FFFF;
    const int ns_grid = (bs_packed >> 19) & 0x3F;   // grid fan-out (an UPPER BOUND on every column's ns)
    const int B       = (bs_packed >> 25) & 0x3F;

    const int blk = blockIdx.x;
    const int b = blk / (nh * ns_grid);
    if (b >= B) return;
    const int rem = blk % (nh * ns_grid);
    const int qh = rem / ns_grid;
    const int split = rem % ns_grid;
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
    const int NW = blockDim.x >> 5;            // 8
    const int DPL = SK_HD >> 5;                // 8

    const long long idx = ((long long)b * nh + qh) * ns_grid + split;
    if (start >= pc) {                          // this split has no keys
        if (threadIdx.x == 0) { out_m[idx] = -1e30f; out_l[idx] = 0.0f; }
        if (threadIdx.x < SK_HD) out_acc[idx * SK_HD + threadIdx.x] = 0.0f;
        return;
    }

    const __nv_bfloat16* qrow = q + (long long)b * (nh * SK_HD) + (long long)qh * SK_HD + lane * DPL;
    float qv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) qv[i] = b2f(qrow[i]);

    float m = -1e30f, l = 0.0f;
    float acc[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    const long long kvbase = ((long long)slot * nkv + kvh) * stride;
    const __nv_bfloat16* kb = k_cache + kvbase * SK_HD + lane * DPL;
    const __nv_bfloat16* vb = v_cache + kvbase * SK_HD + lane * DPL;
    for (int r = start + warp; r < end; r += NW) {
        const int dd = r - pos_start;
        const int t = (!path || dd < 0) ? r : pos_start + (int)path[b * MAX_VERIFY + dd]; // rank -> slot
        const __nv_bfloat16* krow = kb + (long long)t * SK_HD;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += qv[i] * b2f(krow[i]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float a_old = __expf(m - m_new), a_cur = __expf(s - m_new);
        const __nv_bfloat16* vrow = vb + (long long)t * SK_HD;
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = acc[i] * a_old + a_cur * b2f(vrow[i]);
        m = m_new;
        l = l * a_old + a_cur;
    }

    // Merge this block's 8 warp-partials into one partial softmax, in FIXED warp order.
    extern __shared__ float sh[];
    float* sacc = sh;                     // NW * SK_HD
    float* sm   = sh + NW * SK_HD;        // NW
    float* sl   = sm + NW;                // NW
    #pragma unroll
    for (int i = 0; i < 8; i++) sacc[warp * SK_HD + lane * DPL + i] = acc[i];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

    if (threadIdx.x < SK_HD) {
        const int d = threadIdx.x;
        float mg = -1e30f;
        for (int w = 0; w < NW; w++) mg = fmaxf(mg, sm[w]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < NW; w++) {
            const float a = __expf(sm[w] - mg);
            num += sacc[w * SK_HD + d] * a;
            den += sl[w] * a;
        }
        out_acc[idx * SK_HD + d] = num;           // UNNORMALISED fp32, paired with (mg, den)
        if (d == 0) { out_m[idx] = mg; out_l[idx] = den; }
    }
}

// Merge a column's partial softmaxes. It recomputes ns from THIS COLUMN's pc, exactly as
// gqa_attn_splitk did -- the two must agree or the reduction reads partials that were never written.
// `ns_grid` is only the stride of the partial buffer, never the loop bound.
extern "C" __global__ void gqa_attn_reduce(
    __nv_bfloat16* out,
    const float* in_m, const float* in_l, const float* in_acc,
    const int* pos, int ns_grid, int B, int nh_packed) {
    const int nh = nh_packed / 1000;
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
        acc += in_acc[idx * SK_HD + d] * alpha;
    }

    out[(long long)b * (nh * SK_HD) + (long long)qh * SK_HD + d] = f2b(l > 0.0f ? acc / l : 0.0f);
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
template<int NC>
__device__ __forceinline__ void gemm_binv_impl(__nv_bfloat16* C, const __nv_bfloat16* W,
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
        for (int n = 0; n < NC; n++) C[(long long)n * M + m] = f2b(sh[n * T + 0]);
    }
}

extern "C" __global__ void gemm_binv_b(__nv_bfloat16* C, const __nv_bfloat16* W,
                                       const __nv_bfloat16* X, int M, int K, int N) {
    if (blockIdx.x >= M) return;
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
        default: break;
    }
    // Generic fallback (runtime N, acc in local memory) for widths without a specialization —
    // e.g. a future batched multi-lane verify. Same reduction order as the templated path.
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
        for (int n = 0; n < N; n++) C[(long long)n * M + m] = f2b(sh[n * T + 0]);
    }
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

    // Invert the mma C-fragment map: c_i is row (g + 8*(i>=2)), col (2t + (i&1)) of the 16x8 subtile.
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

// ---- NVFP4. One mma k-step consumes exactly one 16-element scale block, so the block scale is
// constant over the step and folds into the A-fragment for free. That alignment is the whole trick.
// Cf != nullptr => write the FP32 accumulator to Cf and leave C untouched (TP=2 FP32-preserving path).
extern "C" __global__ __launch_bounds__(256, 6) void gemm_mma_fp4_b(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, int M, int K, int N,
    float* Cf)
{
    const int mt = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, nblk = K >> 4;

    // The two X rows this lane's B-fragments read. Columns >= N are padding: clamp them onto a valid
    // row so the load stays in bounds. They feed independent accumulators that are never written, so
    // the garbage cannot reach column 0 — see the invariance argument above.
    const long long xr0 = (long long)(g     < N ? g     : N - 1) * K;
    const long long xr1 = (long long)(g + 8 < N ? g + 8 : N - 1) * K;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint32_t* Wt32 = reinterpret_cast<const uint32_t*>(Wt);

    // A warp takes an ADJACENT PAIR of k-blocks per iteration. The pairing is not for unrolling --
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
    const int npair = nblk >> 1;
    for (int p = warp; p < npair; p += MMA_NW) {
        const long long tile = (long long)mt * nblk + (p << 1);   // tiles 2p and 2p+1
        const uint32_t wq0 = Wt32[tile * 32 + lane];              // 128 contiguous B
        const uint32_t wq1 = Wt32[tile * 32 + 32 + lane];         // the next 128, back-to-back
        const uint8_t* sct = Sct + tile * 16;                     // 32 contiguous B = ONE sector
        const float s0lo = e4m3_f(sct[g]),      s0hi = e4m3_f(sct[g + 8]);
        const float s1lo = e4m3_f(sct[g + 16]), s1hi = e4m3_f(sct[g + 24]);

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

    __shared__ float sh[MMA_SMEM];
    mma_epilogue(sh, acc, C, nullptr, gs, mt, M, N, Cf);
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

// Per-token grouped expert MLP. out[:,b] = Σ_j w_j · down[e_j]( silu(gate[e_j]·x_b) * up[e_j]·x_b ).
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
extern "C" __global__ void moe_silu_bf16_b(__nv_bfloat16* h_out, const __nv_bfloat16* gu, int I, int BK) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    int bk = idx / I, r = idx % I;
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
                                          const int* perm_tok, int H, int Ppad) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)Ppad * H) return;
    int pos = idx / H, c = idx % H; int t = perm_tok[pos];
    x_perm[idx] = (t >= 0) ? x[(long)t * H + c] : __float2bfloat16(0.f);
}
// Grouped MMA: same tuned loop as gemm_mma_fp4_b, but the weight tile is expert tile_e[nt] and the 8
// columns are this n-tile's 8 permuted tokens. C[bslot-block, M] = C_perm. N=8 (padding tokens masked
// out downstream by inv_pos). X_perm [Ppad, K], C_perm [Ppad, M], both row-major.
extern "C" __global__ __launch_bounds__(256, 6) void gemm_moe_grouped_mma_fp4(
    __nv_bfloat16* C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ Sct,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ Xperm, const int* __restrict__ tile_e,
    int M, int K, int expert_base)
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
