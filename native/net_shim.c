// TP=2 comm transport shim — flat C ABI over libibverbs, for src/net.rs.
//
// GB10 has NO GPUDirect -> comm buffers are cudaHostAlloc + ibv_reg_mr (coherent to GPU+CPU+NIC).
// RC QP, RoCEv2 GID index 3, TCP control-plane handshake (rdma_cm fails on this RoCE).
//
// HOT PATH = the doorbell all-reduce (see native/tp_doorbell.h for the invariants I1-I9, and
// tp_doorbell_ref/ for the long-form rationale from three expert review rounds). The short version:
//
//   send:  proxy owns a monotone epoch; per barrier it posts ONE linked WR chain —
//          RDMA_WRITE 8 B INLINE length tag -> peer len_peer[e%4096] (unsignaled), then
//          RDMA_WRITE send_ring[s] -> peer recv_ring[s], length align8(len)+tail   (unsignaled), then
//          an 8 B IBV_SEND_INLINE epoch -> peer flags.peer_committed (signaled every S). The payload
//          length VARIES per epoch (slot-filling prefill chunks, small decode barriers); the receiver
//          learns each length from the generation-tagged tag before it can locate the tail.
//          Plain WRITE consumes no peer recv WQE, so the RNR-NAK class (ms-scale bimodal stalls when
//          barriers cluster) cannot happen.
//   recv:  CAN_FLUSH_REMOTE_WRITES = 0 on GB10, so the GPU may NOT consume NIC-written payload
//          directly (the payload DMA need not be visible to the GPU when the epoch flag is). The
//          proxy bounces visibility: observe peer_committed -> full fence -> RELEASE-store cpu_done;
//          the GPU acquire-loads cpu_done and only then reads recv[s].
//
// The previous revision of this file had a single send/recv slot, used gpu_ready as the RDMA SOURCE
// for the epoch (a post->DMA race: the NIC reads that word whenever it gets round to it, not at post
// time), and had the GPU poll the NIC-written epoch directly. It is replaced wholesale.
#define _GNU_SOURCE
#include "tp_doorbell.h"
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <pthread.h>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/auxv.h>   /* CPUIDLE: getauxval(AT_HWCAP) & HWCAP_EVTSTRM */
#include <sys/prctl.h>  /* CPUIDLE: PR_SET_TIMERSLACK for the proxy's idle sleeps */

static inline void net_shim_log_prefix(void) {
    struct timespec ts;
    struct tm tm_loc;
    char tbuf[24];
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm_loc);
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", &tm_loc);
    if (isatty(STDERR_FILENO)) {
        fprintf(stderr, "\x1b[1;35m[net_shim]\x1b[0m \x1b[2m%s\x1b[0m ", tbuf);
    } else {
        fprintf(stderr, "[net_shim] %s ", tbuf);
    }
}
#define LOGE(fmt, ...) do { flockfile(stderr); net_shim_log_prefix(); fprintf(stderr, fmt "\n", ##__VA_ARGS__); funlockfile(stderr); } while (0)

#define TP_CTS_STRIDE     5
#define TP_CTS_READY      0   // proxy observed gpu_ready >= e
#define TP_CTS_POSTED     1   // ibv_post_send returned for e
#define TP_CTS_CQE        2   // CQE retired an epoch >= e
#define TP_CTS_PEERSEEN   3   // peer_committed >= e observed (receive side)
#define TP_CTS_RELEASED   4   // cpu_done stored for e

#define TP_MAX_POST_BATCH (2 * TP_RING_SLOTS)   // reuse gate bounds ready-but-unposted to R

// wr_id for net_exchange's signaled send. The startup channel shares the QP (and cq_send) with the
// hot path, so once the proxy is running (persistent server, request 2+) drain_cq can dequeue the
// exchange's send CQE first — wr_id 0 used to be silently swallowed there and net_exchange then
// waited on got_send forever (live-live stall: both ranks in broadcast_prompt, both "healthy").
// A magic wr_id lets drain_cq hand the completion across instead of eating it. UINT64_MAX collides
// with no epoch and must be excluded from tx_retired accounting.
#define TP_XCHG_WR_ID 0xFFFFFFFFFFFFFFFFull

// P3-1: a one-shot epoch is posted to ALL world-1 peer QPs, so each QP's send CQ delivers a CQE for
// the SAME epoch. drain_cq_nway must attribute each CQE to the QP it completed on (per-QP tx_retired
// credit, I3) — but the legacy tree wr_id is a bare epoch, so a one-shot CQE would be
// indistinguishable. Tag one-shot CQEs with the PEER RANK in the top 16 bits (a bare epoch is
// < 2^48, so the tag never aliases the xchg WR_ID or a real epoch).
#define TP_ONESHOT_TAG(e, p) ((((uint64_t)(p) + 1ull) << 48) | (uint64_t)(e))
#define TP_ONESHOT_TAGGED(w) ((w) >> 48)
#define TP_ONESHOT_PEER(w)   ((int)((w) >> 48) - 1)
#define TP_ONESHOT_EPOCH(w)  ((w) & ((1ull << 48) - 1ull))

// Liveness deadline for the proxy watchdog and net_agree: 10 s. A hang is forever, so any large
// value discriminates; the cost of a FALSE fire is catastrophic (a one-rank abort silently corrupts
// the peer's all-reduces), so err high. Legitimate mid-run unmatched debt is ms-scale (the
// rendezvous bounds skew to ~1 barrier); load-time skew is excluded by arming after the first
// rendezvous. Abort codes: 6 = watchdog, 7 = agree timeout, 8 = pin failure, 9 = len-ring desync,
// 10 = dead peer in exchange, 11 = device-side payload-tail timeout (K2'/maxloc_g stage B fired —
// EXPERT_GPU_ALLREDUCE §3.2; 1 user, 2 tail-guard, 3 post_send, 4 CQE, 5 agree-post already in use).
#define TP_WDOG_NS 10000000000ull
// Dead-peer probe cadence for net_exchange. The QP runs retry_cnt=7 (INFINITE retries) by design, so
// a peer that dies after ACKing our send produces NO error CQE for a passive waiter — without an
// out-of-band check, net_exchange waiting on the peer's WITH_IMM is an infinite 100%-CPU spin (the
// 131-min node incident). Legitimate one-sided waits here are LONG: the node waits out the head's
// ~100 s model load before the first request and arbitrary user idle between requests. So every
// TP_LIVE_PROBE_NS of total silence we TCP-probe the peer's liveness port (tcp_port+1): connect OK =
// peer process alive, keep waiting; connect refused/timeout (twice) = peer GONE -> abort code 10.
#define TP_LIVE_PROBE_NS 5000000000ull
// How long the RECV path waits for a slot's tail-epoch to land after the commit for that epoch was
// already observed (relaxed-PCIe reorder window) before declaring the payload lost. The payload was
// posted before the commit on a reliable QP, so the true lag is ns-us; 1 ms is ~100x the barrier
// floor and 1/10000 of the watchdog — beyond it, something is genuinely broken. Now lives in
// tp_doorbell.h (TP_TAIL_WAIT_NS) so K2'/maxloc_g share it.

/* One per-peer RC sub-context (world > 2). Indexed by PEER RANK (not a dense 0..n-2 — the doubling
 * schedule selects a partner by `rank ^ (1<<round)`, so the array is the partner rank). The ring
 * (send/recv/len) stays SHARED across peers: one epoch is in flight against one partner at a time. */
typedef struct PeerLink {
    int               valid;
    struct ibv_qp*    qp;
    struct ibv_qp_ex* qpx;
    uint64_t          remote_addr;
    uint32_t          remote_rkey;
    char              peer_ip[64];
} PeerLink;

typedef struct NetCtx {
    struct ibv_context* ctx;
    struct ibv_pd*      pd;
    struct ibv_td*      td;          // Phase 1C: lock-free single-threaded domain for oneshot_uniform QPs
    struct ibv_pd*      td_pd;       // parent domain binding pd + td (NULL when unused/unsupported)
    struct ibv_cq*      cq_send;     // hot path (R2b: separate from the startup CQ)
    struct ibv_cq*      cq_startup;  // retained WITH_IMM handshake / out-of-band channel
    struct ibv_qp*      qp;
    struct ibv_mr*      mr;
    unsigned            max_inline;  // negotiated QP max_inline_data (512 -> 256 -> 64)

    void*    hbuf;            // cudaHostAlloc host ptr: [flags][len_local][len_peer][send_ring][recv_ring]
    void*    dbuf;            // matching CUDA device ptr
    size_t   region_bytes;
    unsigned slot_stride;     // per-slot bytes, sized for the FP32 payload (round-3)
    unsigned payload_bytes;   // DEFAULT payload bytes (K1 nbytes==0: bf16 hidden*2, or fp32 hidden*4)

    tp_dev_ctx* dev_ctx_h;    // mapped pinned tp_dev_ctx (host view)
    void*       dev_ctx_d;    // device ptr to the same

    uint64_t remote_addr;     // peer MR base
    uint32_t remote_rkey;
    uint32_t gen;             // WITH_IMM exchange generation (startup path only)

    char     peer_ip[64];     // peer dotted-quad (rank 0 learns it from accept()) — liveness probes
    int      liveness_port;   // tcp_port + 1: our responder + the peer-probe target

    // --- N-way transport (world > 2). All zero/NULL at world==2 so the single-QP path is untouched. ---
    int      world;           // TP rank count (2 = legacy single-QP path)
    int      rounds;          // log2(world)
    int      oneshot;         // P3-1: 1 = one-shot all-peers push at world==4 (GB10_TP_ONESHOT)
    int      oneshot_uniform; // TP-4F2: 1 = EVERY epoch of this ctx is a one-shot all-peers push (a dedicated
                              // single-stage reduce ctx; world==4 only). Set only by net_set_oneshot_uniform_once.
    struct NetCtx* parent;    // TP-4F2: the primary link; this ctx's abort is mirrored into it (and vice versa)
    PeerLink* peers;         // array indexed by PEER RANK, size world (slot [rank] is unused/valid=0)
    uint64_t nway_flags_off;  // byte offset of the per-peer flag region inside hbuf (0 at world==2)
    uint64_t nway_flags_d;    // device ptr to the per-peer flag region (0 at world==2)
    uint64_t nway_recv_off;   // byte offset of the per-round recv rings (0 at world==2); rounds*R slots
    uint64_t oneshot_recv_off;// P3-1: byte offset of the DEDICATED sender-indexed one-shot block (world rings)
    void*    qp_mask_dev;     // P3-1/expert: device qp_mask ring (cudaMalloc'd, world>2 only)
    volatile uint64_t xchg_send_done;  // drain_cq -> net_exchange handoff: the proxy dequeued the
                                       // exchange's send CQE (cq_send is shared — see TP_XCHG_WR_ID)
    volatile uint64_t xchg_send_seq;   // B8 §1.5-4: monotone send-CQE generation (drain_cq_nway bumps
                                       // per TP_XCHG_WR_ID CQE; exchange_one snapshots-before-post and
                                       // waits for >) — a lost CQE can't be misread, and the wait is
                                       // deadline-bound by the dead-peer probe.
    volatile int      proxy_running;   // net_proxy_loop owns cq_send — net_exchange must not poll it
    uint64_t          last_xchg_gen;   // generation of the last completed exchange (recv-slot select)

    // --- P5: N-way control-plane exchange (world > 2). All zero at world==2 so net_exchange/net_agree
    //     keep the untouched single-QP path. The dedicated per-rank control receive region lives INSIDE
    //     the one ibv_reg_mr buffer, appended after the per-peer flags; rank r's control payload lands
    //     at ctrl_recv_off + r*slot_stride (a distinct full slot per rank, so node A's reply can never
    //     clobber node B's on the head). `xchg_gen[p]` is the per-peer exchange generation (the head
    //     indexes by node rank, a node indexes by 0 = its head peer); it increments once per exchange of
    //     that pair so the two sides stay gen-symmetric. ---
    uint64_t          ctrl_recv_off;
    uint64_t          ctrl_last_off;   // R10: stable per-sender "last received" slots (after the ring)
    uint64_t          ctrl_send_off;   // dedicated control SEND staging slot (world>2), separate from
                                       // the hot-path send ring and net_send_hptr (world==2 path)
    uint64_t          xchg_gen[TP_NWAY_MAX_WORLD];

    // --- R9 DIAGNOSTIC: per-epoch payload checksum rings (world>2 only, GB10_TP_DIAG=1) ---
    // The agree() hash compares the FINAL decode state — useless for localizing WHICH delivery
    // diverged. Instead every posted epoch records (e, partner, FNV-1a64 of the exact payload bytes
    // handed to the NIC) on the SEND side, and every validated epoch records the same triple over
    // the received slot bytes on the RECV side. On an agree mismatch the head gathers all ranks'
    // rings and diffs them: the first (epoch, partner) whose recv sum != the partner's send sum IS
    // the bad delivery. world==2 never touches any of this.
    int       tp_diag;
    uint64_t  diag_send;                        // hbuf offset of the send ring, [i][3] = (e, p, fnv)
    uint64_t  diag_recv;                        // same, receiver side
    uint64_t  diag_send_idx, diag_recv_idx;
    uint64_t  diag_send_fold[TP_NWAY_MAX_WORLD];  // cumulative FNV folds per partner (XOR'd in)
    uint64_t  diag_recv_fold[TP_NWAY_MAX_WORLD];

    volatile int aborted;
    int      port_num, gid_idx, rank;
    int      recv_gpu;              // v2 receive mode (GB10_TP_GPU_RECV): the GPU kernel consumes
                                    // the NIC-written payload directly; the proxy skips its RECV stage
                                    // and the watchdog keys on TP_F_RX_DONE (EXPERT_GPU_ALLREDUCE §3.2)

    // --- bench hooks (all zero in production) ---
    unsigned  inject_delay_us_max;   // random proxy sleep before each post
    unsigned  cq_hold;               // defer CQ draining until this many epochs are unretired
    uint64_t  cq_hold_ns;            // ...and then for this long, so the gate demonstrably binds
    uint64_t  hold_since;
    int       ts_on;                 // record CPU-side per-epoch timestamps
    uint64_t* cpu_ts;                // [TP_GTS_EPOCHS][TP_CTS_STRIDE]
    uint64_t* gpu_ts_h;              // mapped GPU timestamp ring (host view)
    uint64_t  tail_fires;            // tail-epoch guard fire count (payload never landed) — MUST stay 0
    uint64_t  tail_waits;            // times the tail-epoch wait engaged (reordered payload, recovered)
    uint64_t  gpu_rx_skips;          // TP-G: epochs the GPU decode K2 proved itself (RX_DONE >= e) — no CPU proof needed
    uint64_t  len_waits;             // times the len-tag wait engaged (reordered tag, recovered)
    int       tail_drill;            // GB10_TP_TAIL_DRILL: invert commit/payload order every 4096th epoch
    uint64_t  posted_epochs, retired_epochs, released_epochs;
    uint64_t  qp_posted[TP_NWAY_MAX_WORLD];  // Phase 1C: per-QP post counter for S=4 signaling on oneshot_uniform
    uint64_t  agree_last;            // last lockstep token shipped
    int       last_posted_peer;      // last QP's peer rank (world>2 self-clocking of the reuse gate)
    uint32_t  rng;
    // S-B9-REL-TP2RACE probe (--tp-race-probe <ms>, issue #10): rank 1's proxy holds every world-2
    // epoch release (cpu_done) by this many ms and hashes the withheld slot across the hold — any
    // change is a host exchange frame landing on an unconsumed epoch payload, caught at the
    // transport layer whatever it does to the drafts. 0 = off (production).
    uint32_t  probe_hold_ms;
    uint64_t  probe_hits;            // withheld slots observed changing (MUST stay 0 on a fixed transport)
} NetCtx;

#include "net_shim/01_helpers_proxy_nway.inc"
#include "net_shim/02_init_accessors_bench.inc"
#include "net_shim/03_hot_path.inc"
#include "net_shim/04_startup_audit_channel.inc"
#include "net_shim/05_bulk_rdma.inc"
