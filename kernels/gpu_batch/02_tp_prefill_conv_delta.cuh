// gpu_batch/02_tp_prefill_conv_delta.cuh: TP-F prefill transport pair, TP bench kernels, conv1d_b and delta_step kernels.
// Same protocol (I1-I9) as the single-block K1/K2.
// Section 2 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ================= TP-F: the PREFILL transport pair (fp32 mode-2 payloads, any world) =================
// The single-block K1/K2 above move a 1 MiB prefill epoch at ~12 GB/s and run strictly K1(e) ->
// K2(e) -> K1(e+1): copy, wire and add never overlap (TP-E ledger §5: 590-630 ms of a 1.47 s chunk).
// These two kernels are the SAME protocol (I1-I9, same rings, same len/tail tags, same proxy, same
// wire bytes) with two changes:
//   1. MULTI-BLOCK bodies. K1m: every block waits the I3 reuse gate itself (plain loads, I6), copies
//      its grid-stride share of the payload (16 B vectors), then arrives on a device-memory counter;
//      the LAST block to arrive (fence.sc.sys on both sides of the arrive — the standard last-block
//      pattern) writes the tail epoch + len tag (+ the world>2 qp_mask entry), stores c->epoch and
//      RELEASES gpu_ready — exactly the single-block K1's publish sequence, after ALL blocks' payload
//      stores. K2m: every block acquires the receive gate itself and adds its share.
//   2. PIPELINED epochs. K2m consumes epoch c->epoch - lag, so the host may launch up to G K1m ahead
//      of the K2m that consumes the oldest (lag = a per-launch CONSTANT: capture-safe, the round-3
//      rule forbids only absolute epoch/slot args). Recv-slot safety (world 2): the peer can
//      overwrite our recv slot x%R only with epoch x+R, whose K1m follows its K2m(x+R-G), which needs
//      OUR K1m(x+R-G), which follows our K2m(x+R-2G) >= K2m(x) in stream order => R >= 2G. world>2
//      (recursive doubling, round = e % rounds): with G a multiple of rounds, round(x+R-G) = round(x)
//      so the peer's K2m(x+R-G) consumes from US and the same chain closes at one hop (host enforces
//      both; the world>2 arm is UNTESTED on hardware as of TP-F — see PLAN/notes_2026-09-28/TP-F.md).
// World-general exactly as K1/K2: world==2 uses the single QP's flags and the shared recv ring;
// world>2 gates on partner(e-R)'s per-QP tx_retired, partner(e)'s per-peer cpu_done + the slot's
// generation-tagged tail (R9), per-round recv rings, and the round-k canonical order (the partner
// whose bit k is 0 first) — so the fp32 result is rank-identical for any power-of-two world and
// bitwise equal to the single-block pair for any chunking/pipelining. One-shot (P3-1) is refused by
// the host. nbytes must be a 16 B multiple (the host falls back to the single-block pair otherwise).
// src_f16 (bit 0): `src` holds f16 values and K1m widens them to the fp32 wire payload while copying
// (exact; fuses the producer's f16 -> f32 conversion pass). Bit 2 (f16 WIRE, requires bit 0): the
// partial is ALREADY f16-exact (the row-parallel wide chain's f16 y_raw), so the wire carries the f16
// values themselves — half the bytes, and the peer widens them exactly: bitwise the same fp32 sum
// (lossless: no value is rounded that was not already f16). nbytes is the wire length either way.
extern "C" __global__ void tp_gate_copy_signal_mb(tp_dev_ctx* c, const void* src, unsigned nbytes,
                                                  unsigned* arrive, int src_f16) {
    // Every block reads the PRE-kernel counter (the last block stores it only after all arrived).
    const unsigned long long e = c->epoch + 1;
    const unsigned s = (unsigned)(e & (TP_RING_SLOTS - 1));
    __shared__ int s_ab;
    if (threadIdx.x == 0) {
        s_ab = 0;
        if (e > TP_RING_SLOTS) {
            const unsigned long long tgt = e - TP_RING_SLOTS;
            // world>2: the slot at e-R was shipped on partner(e-R)'s QP (tree, never one-shot here).
            const unsigned long long* ret = (c->world > 2)
                ? tp_nway_flagp(c, tp_partner_of(c, tgt), TP_NWAY_TX_OFF)
                : tp_flag(c, TP_F_TX_RETIRED);
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            if (blockIdx.x == 0 && tp_ld_relaxed(ret) < tgt) c->gate_waits += 1;
            unsigned long long deadline = tp_globaltimer() + K1_GATE_WAIT_NS;
            unsigned ns = 64, cap = 2048u;
            while (tp_ld_relaxed(ret) < tgt) {
                if (tp_ld_relaxed(ab)) { s_ab = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    if (blockIdx.x == 0)
                        printf("[K1m-gate-timeout] rank=%d epoch=%llu tgt=%llu txret=%llu\n",
                               c->rank, e, tgt, tp_ld_relaxed(ret));
                    s_ab = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_ab) tp_fence_acquire();
        }
    }
    __syncthreads();
    unsigned char* slot = c->send_ring + (size_t)s * c->slot_stride;
    if (!s_ab) {
        const unsigned n16 = nbytes >> 4;
        if (src_f16 & 4) {                              // f16 wire: a plain byte copy
            uint4* dst = (uint4*)slot;
            const uint4* s4 = (const uint4*)src;
            for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x)
                dst[i] = s4[i];
        } else if (src_f16 & 1) {
            float4* dst = (float4*)slot;
            const uint2* sh = (const uint2*)src;         // 4 halves per 8 B
            for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
                const uint2 u = sh[i];
                const __half2 h01 = *reinterpret_cast<const __half2*>(&u.x);
                const __half2 h23 = *reinterpret_cast<const __half2*>(&u.y);
                const float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
                dst[i] = make_float4(f01.x, f01.y, f23.x, f23.y);
            }
        } else {
            uint4* dst = (uint4*)slot;
            const uint4* s4 = (const uint4*)src;
            for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x)
                dst[i] = s4[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();                         // this block's payload stores, system scope
        const unsigned prev = atomicAdd(arrive, 1u);    // device-memory counter (GPU-only line, I6-safe)
        if (prev == gridDim.x - 1) {
            __threadfence_system();                     // acquire side of the last-block pattern
            *arrive = 0u;                               // next launch is stream-ordered behind us
            c->epoch = e;                               // advances even on abort, like K1 (I9 no-op)
            if (c->world > 2 && c->qp_mask)             // tree epoch: just partner(e)'s QP (as K1)
                c->qp_mask[e & (TP_QPMASK_SLOTS - 1)] = 1u << tp_partner_of(c, e);
            if (!tp_ld_relaxed(tp_flag(c, TP_F_ABORT))) {
                const unsigned wire = (nbytes + 7u) & ~7u;
                *(unsigned long long*)(slot + wire) = e;                 // tail epoch, written LAST
                c->len_local[e & (TP_LEN_EPOCHS - 1)] = TP_LEN_TAG(e, wire);
                tp_st_release(tp_flag(c, TP_F_GPU_READY), e);            // watermark (I1)
            }
        }
    }
}

// io bit 0: `local` holds f16 (widened exactly on read); io bit 1: `out` is f16, written as
// __float2half_rn(sum) — the same rounding as xq_cvt_f32_f16 (fuses the consumer's f32 -> f16 pass);
// io bit 2: the PEER's payload is f16 on the wire (K1m bit 2), widened exactly; n = elements.
// io 0 = the plain fp32 in-place reduce (out == local allowed in every mode: same-thread RAW).
extern "C" __global__ void tp_wait_add_mb(tp_dev_ctx* c, void* out, const void* local, int n,
                                          unsigned lag, int io) {
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    // I5 gate: world==2 the single cpu_done; world>2 partner(e)'s per-peer cpu_done. Per block; I9.
    const unsigned long long* done = (c->world > 2)
        ? tp_nway_flagp(c, tp_partner_of(c, e), TP_NWAY_CPU_OFF)
        : tp_flag(c, TP_F_CPU_DONE);
    if (tp_spin_until_ge(c, done, e, 1)) return;
    const unsigned char* slot = tp_recv_slot_ptr(c, e);
    if (c->world > 2) {
        // R9: cpu_done is a per-partner WATERMARK — validate THIS epoch's slot tail by equality.
        const unsigned wire = ((unsigned)n * ((io & 4) ? 2u : 4u) + 7u) & ~7u;
        __shared__ int s_tail;
        if (threadIdx.x == 0) {
            s_tail = 0;
            const unsigned long long* tail = (const unsigned long long*)(slot + wire);
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            unsigned long long deadline = tp_globaltimer() + TP_TAIL_LONG_WAIT_NS;
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(tail) != e) {
                if (tp_ld_relaxed(ab)) { s_tail = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    if (blockIdx.x == 0) printf("[K2m-tail-timeout] rank=%d epoch=%llu wire=%u\n", c->rank, e, wire);
                    s_tail = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_tail) tp_fence_acquire();
        }
        __syncthreads();
        if (s_tail) return;
    }
    const float4* pe = (const float4*)slot;
    // canonical order (tp_reduce_body mode 2): world 2 rank0 + rank1; world>2 round k the partner
    // whose bit k is 0 first — both partners compute the same sum in the same order.
    const int lower = (c->world > 2) ? ((c->rank & (1 << tp_round_of(c, e))) == 0) : (c->rank == 0);
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
        float4 p4;
        if (io & 4) {
            const uint2 u = ((const uint2*)slot)[i];
            const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            p4 = make_float4(f01.x, f01.y, f23.x, f23.y);
        } else {
            p4 = pe[i];
        }
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

// TP-SP (sequence-parallel prefill, src/tp_xport.rs all_gather_rows): the ALL-GATHER's receive side.
// The same per-block receive gate as tp_wait_add_mb (world 2: the single cpu_done >= e; world > 2:
// partner(e)'s per-peer cpu_done + the slot's generation-tagged tail by equality, R9; abort = no-op,
// I9), then a plain grid-stride 16 B copy of the peer's `nbytes` payload into `out`. NO arithmetic:
// the peer's bits land unchanged (a K2m add with a zero local partial would turn -0.0 into +0.0).
// Epoch e = c->epoch - lag, exactly as K2m (lag a per-launch constant; capture-safe).
extern "C" __global__ void tp_wait_copy_mb(tp_dev_ctx* c, void* out, unsigned nbytes, unsigned lag) {
    const unsigned long long e = c->epoch - (unsigned long long)lag;
    const unsigned long long* done = (c->world > 2)
        ? tp_nway_flagp(c, tp_partner_of(c, e), TP_NWAY_CPU_OFF)
        : tp_flag(c, TP_F_CPU_DONE);
    if (tp_spin_until_ge(c, done, e, 1)) return;
    const unsigned char* slot = tp_recv_slot_ptr(c, e);
    if (c->world > 2) {
        const unsigned wire = (nbytes + 7u) & ~7u;
        __shared__ int s_tail;
        if (threadIdx.x == 0) {
            s_tail = 0;
            const unsigned long long* tail = (const unsigned long long*)(slot + wire);
            const unsigned long long* ab = tp_flag(c, TP_F_ABORT);
            unsigned long long deadline = tp_globaltimer() + TP_TAIL_LONG_WAIT_NS;
            unsigned ns = 64, cap = 512u;
            while (tp_ld_relaxed(tail) != e) {
                if (tp_ld_relaxed(ab)) { s_tail = 1; break; }
                if (tp_globaltimer() >= deadline) {
                    // device status 11 (I9): the host's next flag poll reports the abort loudly (no printf
                    // here: its argument buffer would cost this kernel a stack frame)
                    tp_st_release(tp_flag(c, TP_F_ABORT), 11);
                    s_tail = 1;
                    break;
                }
                __nanosleep(ns);
                if (ns < cap) ns <<= 1;
            }
            if (!s_tail) tp_fence_acquire();
        }
        __syncthreads();
        if (s_tail) return;
    }
    const uint4* s4 = (const uint4*)slot;
    uint4* d4 = (uint4*)out;
    const unsigned n16 = nbytes >> 4;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x)
        d4[i] = s4[i];
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
//   A[t,s] = (gamma_t/gamma_s)(k_t.k_s)  strictly lower  [S-A3-w: was gamma_{t-1}; fixed in tc_b + prefill_b, split wuo_b parked (arm NaN)]
//   T = (I + diag(beta)A)^-1 diag(beta)   (fwd-substitution, f32)
//   U = T (V - Gamma K S_in),  Gamma_t = gamma_t   [gamma_t NOT gamma_{t-1} — validated]
//   S_out = gamma_C S_in + K_w^T U,  K_w[s] = (gamma_C/gamma_s) k_s
//   O = D U + Gamma (Q S_in),  D[t,s] = (gamma_t/gamma_s)(q_t.k_s), s<=t
// q/k are normalized + 1/sqrt(kd)-scaled IN KERNEL (same formula as gdn_token).
// Grid: nh * (vd/64); block 256; dynamic smem (see Rust dispatch for the size).
// ===========================================================================
#define GDN_CHUNK 32
#define GDN_VCB   64            // value columns per block (validated standalone config; needs smem opt-in)


