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
#define TP_TAIL_WAIT_NS  1000000ull  /* K2'/maxloc_g stage-B deadline: a tail lagging the
                                        inline commit past this is lost, not slow (status 11) */
#define TP_HINT_WAIT_NS  1000000ull  /* K2' stage-A hint deadline: the inline commit hint only arms
                                        stage B's short deadline. On GB10 (CAN_FLUSH_REMOTE_WRITES=0,
                                        invariant I5) a RETIRED 8 B inline commit can sit in flush
                                        limbo indefinitely, invisible to the polling GPU — an
                                        unbounded stage-A spin froze a rank mid-reduce with device
                                        status 0 (the world=4 §4.1 wedge). Past this bound, stop
                                        waiting on the hint and let the tail poll prove placement. */
#define TP_TAIL_LONG_WAIT_NS 5000000000ull /* K2' stage-B deadline when the hint never arrived: the
                                        payload may still be legitimately in flight (a slow peer is
                                        legal); the same bound the K1 reuse gate tolerates, well
                                        inside the 10 s watchdog. On expiry: status 11 (I9). */
#define K1_GATE_WAIT_NS  5000000000ull /* K1 reuse-gate (I3) deadline: a stalled send DMA (no CQE —
                                        the C2C-freeze class) would bind the tx_retired spin forever;
                                        on expiry K1 publishes device status 11 (cooperative I9) so
                                        both ranks no-op and the host sees a loud abort instead of
                                        waiting out the 10 s watchdog for a silent one. */
#define TP_CL            64

/* R10: control-exchange ring depth per sender (net_exchange_one). The old single slot per sender
 * had a clobber race (the peer's next exchange could overwrite the tag we hadn't observed yet);
 * a pair runs at most one agree = 2 exchanges ahead, so 4 slots cannot alias. */
#define TP_CTRL_RING     4

/* Flags block — six u64s, each alone on a 64 B line (I7). Byte offsets from the flags base;
 * both the proxy and the kernels index by these, so they are the ABI. */
#define TP_F_GPU_READY        0    /* GPU-written  : producer watermark (I1)           */
#define TP_F_PEER_COMMITTED  64    /* NIC-written  : peer proxy's inline epoch lands here */
#define TP_F_CPU_DONE       128    /* CPU-written  : GPU release gate (I5 — v1 receive) */
#define TP_F_TX_RETIRED     192    /* CPU-written  : reuse credit from CQEs (I3)       */
#define TP_F_ABORT          256    /* status word, 0 = ok (I9)                         */
/* Lockstep agreement channel (MTP under TP). Both ranks must accept the SAME drafted tokens every step;
 * if they ever differ they desync permanently and the all-reduce starts pairing mismatched epochs. The
 * main thread publishes (step, accept_count, hash) into AGREE_OUT; the proxy ships it inline to the
 * peer's AGREE_IN using the same doorbell mechanism as the barrier — no second QP, no locking around
 * ibv_post_send, and it reuses transport that is already adversarially proven. */
#define TP_F_AGREE_OUT      320    /* this rank's (step|count|hash), main-thread written */
#define TP_F_AGREE_IN       384    /* peer's, NIC written                                */
#define TP_F_RX_DONE        448    /* GPU-written (K2' tp_wait_add_g): receive watermark —
                                      published AFTER the payload was consumed; feeds the watchdog
                                      debt signal in v2 receive mode (EXPERT_GPU_ALLREDUCE §3.2/§7) */
#define TP_FLAGS_BYTES      512

/* N-way per-peer flag region (world > 2 only). Lives OUTSIDE the fixed 512 B world==2 flags block so
 * world==2 keeps every existing offset (R1 byte-identity). Three sub-arrays, each indexed by PEER RANK
 * (not a dense 0..n-2 — the design indexes by partner_rank), each entry on its own 64 B line (I7):
 *   peer_committed[p]  (NIC-written)  : peer p's proxy's inline epoch lands here
 *   tx_retired[p]      (CPU-written)  : this rank's QP-to-p retirement credit (per-QP I3 gate)
 *   cpu_done[p]        (CPU-written)  : this rank's receive release gate for peer p (I5)
 * The region is appended AFTER the recv ring inside the one ibv_reg_mr buffer at world>2; its byte
 * offset is a runtime value, so the kernels address it via the c->nway_flags POINTER (never a
 * compile-time offset). world==2 sets nway_flags = NULL and never touches these offsets. */
#define TP_NWAY_MAX_WORLD   16
#define TP_NWAY_PEER_OFF    (0)
#define TP_NWAY_TX_OFF      (TP_NWAY_MAX_WORLD * TP_CL)
#define TP_NWAY_CPU_OFF     (2 * TP_NWAY_MAX_WORLD * TP_CL)
#define TP_NWAY_ABORT_OFF   (3 * TP_NWAY_MAX_WORLD * TP_CL)  /* B8 §1.5-7: per-peer abort mirror —
                                          host-polled only (the proxy reads a peer's abort here and
                                          RDMA-writes its own on abort); the device never indexes it */
#define TP_NWAY_FLAGS_BYTES (4 * TP_NWAY_MAX_WORLD * TP_CL)

/* N-way per-round receive rings (world > 2 only). Recursive doubling alternates the partner per
 * round (partner(e) = rank ^ (1 << (e % rounds))), so consecutive epochs map to DIFFERENT partners.
 * With a single shared recv ring indexed by a bare e % R, a barrier e and a barrier e + R land in
 * the SAME slot s = e % R even when they belong to different rounds/partners — and the I3 reuse
 * gate only constrains the SENDER's own send ring, not cross-partner freshness of the recv slot.
 * On GB10 (CAN_FLUSH_REMOTE_WRITES=0, relaxed Grace C2C / PCIe ordering) a round-r payload and a
 * later round-r' payload for the same slot are NOT placement-ordered across the two QPs, so K2 for
 * round r can read a slot whose bytes the OTHER round's partner already started overwriting — a
 * stale/torn payload that passes the generation-tagged tail wait on one member of a pair but not
 * the other (the measured world=4 agree() hash divergence: ranks 0==2, 1 and 3 each differ).
 *
 * The fix: give each recursive-doubling ROUND its own recv ring — `rounds` copies of the R-slot
 * ring, appended AFTER the shared [send_ring][recv_ring] block (which keeps its world==2 layout
 * byte-for-byte). The recv slot for epoch e on the world>2 arm is
 *     nway_recv_base + round(e) * R * slot_stride + ((e / rounds) % R) * slot_stride,   round(e) = e % rounds.
 * Phase 2: per-round quotient indexing `(e / rounds) % R` uses all R slots bijectively per round
 * (reuse period = R * rounds epochs for ANY rounds >= 1, including world=8 / rounds=3 -> period 24).
 * Both sides compute round(e) and slot(e) identically (SPMD, from the device epoch), so the sender's
 * post_range_nway remote address and the receiver's K2/proxy read address agree without any wire
 * change. rounds <= log2(TP_NWAY_MAX_WORLD) = 4. world==2 sets these to 0 and never uses them. */
#define TP_NWAY_MAX_ROUNDS   4   /* log2(TP_NWAY_MAX_WORLD) */

/* P3-1 (expert Change 2): device ring recording, per epoch, the QP bitmask that epoch was posted
 * to — bit p set iff epoch e's commit was posted to peer p's QP. K1 writes qp_mask[e % 128] right
 * after bumping the epoch; the reuse gate at epoch e reads qp_mask[(e-R) % 128] to learn the TRUE
 * conflict set Q(e-R) (tree epoch -> just its partner's QP; one-shot -> all peers' QPs) and waits
 * only on those tx_retired lines. The gate's target stays e-R; only the WAIT SET narrows to the
 * QPs that actually read the slot. SPMD-derived from the same per-epoch predicate, so every rank
 * computes the same mask. 128 entries >> R so the read at (e-R) can never alias epoch e. */
#define TP_QPMASK_SLOTS  128  /* power of two; >> TP_RING_SLOTS */

/* Per-epoch length rings (variable-payload barriers). The payload length varies per epoch (prefill
 * chunks fill a slot; decode barriers stay small), so the receiver must learn each epoch's byte length
 * BEFORE it can locate the tail: K1 publishes the length here, generation-tagged, and the sender proxy
 * ships it inline ahead of the payload. Both rings live INSIDE the ibv_reg_mr region:
 *   len_local[e % TP_LEN_EPOCHS] — written by the LOCAL K1 (before the gpu_ready release), read by the
 *                                  LOCAL proxy at post time. Never on the wire.
 *   len_peer [e % TP_LEN_EPOCHS] — RDMA target for the PEER proxy's inline tag write, read by the
 *                                  LOCAL CPU in the RECV path (bounded-wait, same discipline as the tail).
 * Tag encoding: val = (epoch << 20) | (wire_bytes >> 3). wire_bytes = align8(per-call nbytes) <= slot
 * capacity (128 KB) => wire_bytes>>3 <= 2^14 fits the low 20 bits. Epochs are monotone, so a stale entry
 * reads epoch e-TP_LEN_EPOCHS and never aliases e; a FUTURE generation cannot land because the sender's
 * own I3 reuse gate (plus the bidirectional rendezvous) bounds how far it can run ahead of the consumer.
 * The length itself is always an 8 B multiple (the wire length is rounded up; K2 reads exactly c elems,
 * so <= 4 pad bytes at the tail end of the payload are harmless). */
#define TP_LEN_EPOCHS     4096   /* power of two; >> R so ring reuse is never close (see net_shim.c)  */
/* Bit 19 inside the 20-bit tag carries the All-to-All mode flag (bits 0..18 encode wire_bytes>>3 up to 4 MiB). */
#define TP_LEN_FLAG_A2A   (1ull << 19)
#define TP_LEN_TAG(e, w)  (((unsigned long long)(e) << 20) | (((unsigned long long)(w)) >> 3))
#define TP_LEN_TAG_EPOCH(t)  ((t) >> 20)
#define TP_LEN_TAG_BYTES(t)  (((t) & ((1ull << 19) - 1)) << 3)
#define TP_LEN_TAG_IS_A2A(t) (((t) & TP_LEN_FLAG_A2A) != 0)

/* MR-registered region layout: [flags][len_local][len_peer][send_ring R*stride][recv_ring R*stride].
 * `stride` is a runtime value (align64(fp32_capacity + TP_TAIL_BYTES)) — slots are sized for the FP32
 * payload from day one so switching precision never re-addresses the rings (round-3). Within a slot the
 * epoch's payload occupies [0, len) and the tail u64 sits AT align8(len), so the RDMA write length is
 * align8(len) + TP_TAIL_BYTES. (len == payload_bytes for the fixed-size decode/bench barriers.) */
#define TP_LEN_LOCAL_OFF  TP_FLAGS_BYTES
#define TP_LEN_PEER_OFF   (TP_LEN_LOCAL_OFF + TP_LEN_EPOCHS * 8)
#define TP_RING_BASE      (TP_LEN_PEER_OFF + TP_LEN_EPOCHS * 8)

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
    unsigned int        payload_bytes;  /* DEFAULT payload (K1 nbytes==0 path + benches): bf16
                                         * hidden*2, fp32 hidden*4. Per-call nbytes overrides.   */
    int                 rank;           /* 0/1 — canonical rank0+rank1 add order (round-3) */
    unsigned int        fp32_payload;   /* 1 = payload is fp32 (production), 0 = bf16     */
    unsigned long long  gate_waits;     /* times K1 actually blocked on the I3 reuse gate */
    unsigned long long* len_local;      /* device ptr to the len_local ring (K1 publishes per-epoch
                                         * wire lengths here; see TP_LEN_EPOCHS above)            */
    /* ---- N-way (world > 2) fields, APPENDED so world==2 keeps the exact pre-P3 layout ----
     * world==2 leaves these at 0/NULL and K1/K2 take the unchanged single-QP arm. */
    unsigned int        world;          /* TP rank count (2 = the byte-identical legacy path)     */
    unsigned int        rounds;         /* log2(world); round = epoch % rounds on the world>2 arm   */
    unsigned long long* nway_flags;     /* device ptr to the per-peer flag region (NULL at world==2) */
    unsigned char*      nway_recv;      /* device ptr to the per-round recv rings (NULL at world==2;
                                         * rounds*R slots, epoch e at round(e)*R + e%R)            */
    unsigned char*      oneshot_recv;   /* P3-1: device ptr to the DEDICATED sender-indexed block
                                         * (after the round rings): sender s's epoch-e payload at
                                         * oneshot_recv + s*R + e%R. NULL unless oneshot.          */
    unsigned int        oneshot;        /* P3-1: 1 = one-shot all-peers push (world==4 only). The nway
                                         * region holds `world` SENDER-indexed rings (sender s's
                                         * payload lands in ring s, slot e%R, on every peer); the
                                         * proxy posts the 3-WR chain to ALL peers per epoch; K2 is
                                         * tp_wait_add_4way (all-tails wait + canonical ((p0+p1)+p2)+p3).
                                         * 0 = the recursive-doubling tree, byte-identical legacy.   */
    unsigned int*       qp_mask;        /* P3-1/expert: device ring, qp_mask[e % TP_QPMASK_SLOTS] =
                                         * bitmask of QPs epoch e was posted to (K1-written). The
                                         * reuse gate at e reads [(e-R) % TP_QPMASK_SLOTS] to gate on
                                         * exactly Q(e-R). NULL at world==2.                        */
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
