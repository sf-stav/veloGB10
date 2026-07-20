/* tp_doorbell.h — shared TP=2 doorbell all-reduce layout (C proxy + CUDA kernels).
 *
 * Derived from tp_doorbell_ref/doorbell_protocol.h with the ROUND-2 / ROUND-3 refinements folded in.
 * Included by BOTH native/net_shim.c (host proxy) and kernels/gpu_batch.cu (device K1/K2), so every
 * offset here is load-bearing on both sides — a change breaks the wire format, not just a build.
 *
 * INVARIANTS (keep these in the real code; the reference file carries the long-form rationale):
 *  I1  The PROXY owns the posted epoch (monotone `next_to_post`, from 1). The epoch reaches the peer as
 *      IBV_SEND_INLINE data copied into the WQE at post time — no host memory the NIC reads for it, so a
 *      post->DMA race on the epoch is structurally impossible. `gpu_ready` is only a producer WATERMARK.
 *  I2  Barrier e uses slot s = e % R in BOTH rings. The bidirectional rendezvous bounds skew to 1
 *      barrier, so R=2 suffices; R=8 is margin.
 *  I3  Reuse gate: before writing send[e%R] the GPU waits tx_retired >= e-R. RC completions are in
 *      order, so a CQE for WR n retires every WR <= n including the unsignaled ones.
 *  I4  S <= R or I3 deadlocks (the CQE that opens the gate would belong to an unpostable epoch).
 *  I5  Visibility (CAN_FLUSH_REMOTE_WRITES=0 on GB10): the GPU may NOT consume NIC-written payload
 *      directly. NIC writes payload then epoch (same RC QP => placement-ordered); the PROXY observes
 *      peer_committed, fences, then RELEASE-stores cpu_done; the GPU ACQUIRE-loads cpu_done and only
 *      then reads recv[s]. The GPU never keys off peer_committed.
 *  I6  Poll loops are plain load + backoff (yield on CPU, __nanosleep on GPU). NEVER an atomic RMW —
 *      it would ping-pong line ownership on the C2C fabric that weights/NIC/CPU all share.
 *  I7  Every flag on its own 64 B line, segregated by writer (GPU / NIC / CPU). MR registered WITHOUT
 *      relaxed ordering.
 *  I8  Counters mutate ONLY via (a) K1 and (b) the proxy following gpu_ready. Any other mutation is a
 *      FULL re-init of every counter/flag on BOTH nodes — never partial recovery.
 *  I9  Abort is COOPERATIVE (status word + return), never __trap(): downstream kernels no-op through
 *      the stream, the host discards the poisoned token and does the I8 re-init.
 *
 * CAPTURE HYGIENE (round-3): K1/K2 take ONLY the ctx pointer and derive slot = epoch % R on-device from
 * the device-side counter. NEVER pass a host-precomputed slot address or epoch value — capture freezes
 * kernel args, so a host-side epoch would freeze the protocol at capture time.
 */
#ifndef TP_DOORBELL_H
#define TP_DOORBELL_H

#define TP_RING_SLOTS    8    /* R — power of two. Tunable (bench 8 vs 16).            */
#define TP_SIGNAL_EVERY  4    /* S — MUST be <= TP_RING_SLOTS (I4).                    */
#define TP_TAIL_BYTES    8    /* trailing u64 epoch guard, written last by K1          */
#define TP_CL            64

/* Flags block — five u64s, each alone on a 64 B line (I7). Byte offsets from the flags base;
 * both the proxy and the kernels index by these, so they are the ABI. */
#define TP_F_GPU_READY        0    /* GPU-written  : producer watermark (I1)           */
#define TP_F_PEER_COMMITTED  64    /* NIC-written  : peer proxy's inline epoch lands here */
#define TP_F_CPU_DONE       128    /* CPU-written  : GPU release gate (I5)             */
#define TP_F_TX_RETIRED     192    /* CPU-written  : reuse credit from CQEs (I3)       */
#define TP_F_ABORT          256    /* status word, 0 = ok (I9)                         */
/* Lockstep agreement channel (MTP under TP). Both ranks must accept the SAME drafted tokens every step;
 * if they ever differ they desync permanently and the all-reduce starts pairing mismatched epochs. The
 * main thread publishes (step, accept_count, hash) into AGREE_OUT; the proxy ships it inline to the
 * peer's AGREE_IN using the same doorbell mechanism as the barrier — no second QP, no locking around
 * ibv_post_send, and it reuses transport that is already adversarially proven. */
#define TP_F_AGREE_OUT      320    /* this rank's (step|count|hash), main-thread written */
#define TP_F_AGREE_IN       384    /* peer's, NIC written                                */
#define TP_FLAGS_BYTES      448

/* MR-registered region layout: [flags][send_ring R*stride][recv_ring R*stride].
 * `stride` is a runtime value (align64(fp32_capacity + TP_TAIL_BYTES)) — slots are sized for the FP32
 * payload from day one so switching precision never re-addresses the rings (round-3). Within a slot the
 * active payload occupies [0, payload_bytes) and the tail u64 sits AT payload_bytes, so the RDMA write
 * length is payload_bytes + TP_TAIL_BYTES. */
#define TP_RING_BASE  TP_FLAGS_BYTES

/* Device-side context. Written once by the host at init, then read by K1/K2; `epoch` is the device-side
 * barrier counter (the source of truth — this is what makes graph capture a no-op). Lives in mapped
 * pinned memory so the host can assert epoch == gpu_ready at graph instantiation (I8 tripwire).
 * Layout is shared between nvcc and cc — keep the fields explicitly sized and 8 B aligned. */
typedef struct {
    unsigned long long  epoch;          /* device barrier counter, incremented in K1     */
    unsigned long long* flags;          /* device ptr to the flags block                 */
    unsigned char*      send_ring;      /* device ptr, R slots of `stride`               */
    unsigned char*      recv_ring;
    unsigned long long* gpu_ts;         /* device ptr to GPU timestamp ring, or NULL     */
    unsigned int        slot_stride;
    unsigned int        payload_bytes;  /* ACTIVE payload (bf16: hidden*2, fp32: hidden*4) */
    int                 rank;           /* 0/1 — canonical rank0+rank1 add order (round-3) */
    unsigned int        fp32_payload;   /* 1 = payload is fp32 (production), 0 = bf16     */
    unsigned long long  gate_waits;     /* times K1 actually blocked on the I3 reuse gate */
} tp_dev_ctx;

/* GPU timestamp ring: TP_GTS_STRIDE u64 slots per epoch, indexed by epoch % TP_GTS_EPOCHS.
 * Bench-only; production runs pass gpu_ts = NULL and the kernels skip the writes. */
#define TP_GTS_EPOCHS   4096
#define TP_GTS_STRIDE   4
#define TP_GTS_K1_IN    0   /* K1 entry                       */
#define TP_GTS_K1_OUT   1   /* K1 gpu_ready published         */
#define TP_GTS_K2_IN    2   /* K2 entry (cpu_done wait start) */
#define TP_GTS_K2_GO    3   /* K2 cpu_done observed           */

#endif /* TP_DOORBELL_H */
