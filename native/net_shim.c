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

#define LOGE(fmt, ...) fprintf(stderr, "[net_shim] " fmt "\n", ##__VA_ARGS__)

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
    int              valid;
    struct ibv_qp*   qp;
    uint64_t         remote_addr;
    uint32_t         remote_rkey;
    char             peer_ip[64];
} PeerLink;

typedef struct NetCtx {
    struct ibv_context* ctx;
    struct ibv_pd*      pd;
    struct ibv_cq*      cq_send;     // hot path (R2b: separate from the startup CQ)
    struct ibv_cq*      cq_startup;  // retained WITH_IMM handshake / out-of-band channel
    struct ibv_qp*      qp;
    struct ibv_mr*      mr;

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

// ---------------------------------------------------------------- helpers

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline void cpu_relax(void) { __asm__ __volatile__("yield" ::: "memory"); }

// TP-E: the host-side wait policy of every CPU rendezvous wait (net_agree, net_exchange, the proxy's
// len/tail placement waits). It used to be "spin 50 us, then nanosleep(1 ms) per retry" everywhere:
// any inter-rank skew above 50 us (a GPU round finishing a few hundred us apart on the two ranks is
// normal) became up to ~1.06 ms of dead time, and the rank that slept then arrived late at the next
// collective. Now: spin (plain load + yield, I6) for a bounded budget, THEN the 1 ms sleeps, so a
// healthy mid-request wait never sleeps and a genuinely idle/one-sided wait (peer model load, a dead
// peer until the watchdog) still drops to ~0% CPU after the budget. The callers run on pinned cores
// (TP scheduler/mirror thread core 9, proxy core 19), so the spin never steals a worker's core.
// Timing only: no value, order or protocol changes. --tp-spin-us overrides the budget (diagnostic;
// 50 = the pre-TP-E behaviour for A/B). Resolved lazily at the first wait, so the node sees the head's
// shipped options (installed before the link carries any lockstep traffic).
// CLI-1: the transport's options arrive from Rust (net_set_opts, called before every net_init) —
// this shim reads no environment variables.
#define TP_SPIN_DEFAULT_NS 20000000ull   /* 20 ms */
static uint64_t g_tp_spin_ns = 0;        /* 0 = unresolved */
static uint64_t g_tp_wait_sleeps = 0;    /* sleeps taken by the wait helper (all threads) */
static unsigned long long g_opt_spin_us = 0;   /* --tp-spin-us (0 = unset / the default) */
static int g_opt_tail_drill = 0;               /* --tp-tail-drill (test-only) */
static int g_opt_oneshot = 0;                  /* --tp-oneshot (world == 4 only) */
static int g_opt_tp_diag = 0;                  /* --tp-diag */
static unsigned g_opt_probe_hold_ms = 0;   /* --tp-race-probe <ms> (0 = off; test-only, issue #10) */
void net_set_probe(unsigned hold_ms) { g_opt_probe_hold_ms = hold_ms; }

void net_set_opts(unsigned long long spin_us, int tail_drill, int oneshot, int tp_diag) {
    g_opt_spin_us = spin_us; g_opt_tail_drill = tail_drill; g_opt_oneshot = oneshot; g_opt_tp_diag = tp_diag;
}
/* TP-4F2 (--tp-reduce single): the NEXT net_init_nway (world == 4) builds a uniform one-shot ctx — every
 * epoch is the all-peers push into the sender-indexed block, the proxy releases all three peers together,
 * the K1 reuse gate waits on all three QPs. One-shot by design: consumed (and cleared) by that init, so no
 * other ctx — rail 1, rail 2 — is ever affected. The legacy --tp-oneshot global is untouched. */
static int g_oneshot_uniform_once = 0;
void net_set_oneshot_uniform_once(int on) { g_oneshot_uniform_once = on; }
static inline uint64_t tp_spin_budget_ns(void) {
    uint64_t v = __atomic_load_n(&g_tp_spin_ns, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        unsigned long long us = g_opt_spin_us;
        v = TP_SPIN_DEFAULT_NS;
        if (us >= 1) v = (uint64_t)us * 1000ull;
        __atomic_store_n(&g_tp_spin_ns, v, __ATOMIC_RELAXED);
        fprintf(stderr, "[net] TP-E host wait policy: spin %llu us, then 1 ms sleeps%s\n",
                (unsigned long long)(v / 1000ull), us >= 1 ? " (--tp-spin-us)" : " (default)");
    }
    return v;
}
// One retry step of a host wait that has been waiting `dt` ns.
static inline void tp_host_backoff(uint64_t dt) {
    if (dt >= tp_spin_budget_ns()) {
        __atomic_fetch_add(&g_tp_wait_sleeps, 1, __ATOMIC_RELAXED);   // host counter, not a flag line (I6 n/a)
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 }; nanosleep(&ts, NULL);
    } else cpu_relax();
}
uint64_t net_wait_sleeps(void) { return __atomic_load_n(&g_tp_wait_sleeps, __ATOMIC_RELAXED); }

static inline volatile uint64_t* flagp(NetCtx* c, size_t off) {
    return (volatile uint64_t*)((char*)c->hbuf + off);
}
static inline char* send_slot(NetCtx* c, uint64_t e) {
    return (char*)c->hbuf + TP_RING_BASE + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
static inline char* recv_slot(NetCtx* c, uint64_t e) {
    return (char*)c->hbuf + TP_RING_BASE + (size_t)TP_RING_SLOTS * c->slot_stride
         + (size_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
static inline uint64_t peer_recv_raddr(NetCtx* c, uint64_t e) {
    return c->remote_addr + TP_RING_BASE + (uint64_t)TP_RING_SLOTS * c->slot_stride
         + (uint64_t)(e & (TP_RING_SLOTS - 1)) * c->slot_stride;
}
static inline void stamp(NetCtx* c, uint64_t e, int which, uint64_t t) {
    if (c->ts_on) c->cpu_ts[(e % TP_GTS_EPOCHS) * TP_CTS_STRIDE + which] = t;
}
// R10 freeze-frame dump (defined below); forward-declared so tp_set_abort can call it on the FIRST
// abort regardless of code (B8 §1.5-2 — a silent code-11/2/9 abort must leave the same phase+ring
// evidence a watchdog does).
static void wdog_dump(NetCtx* c, uint64_t posted, uint64_t matched);

// Cooperative abort (I9): a status word, never a trap. Downstream kernels see it and no-op.
static void tp_set_abort(NetCtx* c, uint64_t code) {
    if (!c->aborted && code != 0) {  // log only the FIRST abort (the diagnostic signal, once)
        LOGE("ABORT rank=%d world=%d code=%llu (tp_set_abort)", c->rank, c->world, (unsigned long long)code);
        // B8 §1.5-2: freeze-frame on EVERY first abort (not just the code-6 watchdog), so a silent
        // code-11 / code-2 / code-9 abort leaves the same phase + ring evidence a watchdog would.
        wdog_dump(c, c->posted_epochs, c->retired_epochs);
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(flagp(c, TP_F_ABORT), code, __ATOMIC_RELEASE);
    c->aborted = 1;
}

// R10: freeze-frame dump on the watchdog path. The GPU stamp ring (host-pinned, written by the
// kernels BEFORE they spin) plus the flags/len rings say exactly which epoch and phase each side
// was in when the link died — without this the watchdog leaves nothing but the two numbers it
// prints, and a deadline-less K2 stage-A spin is invisible everywhere else.
static inline volatile uint64_t* peer_committed_flagp(NetCtx* c, int peer_rank);
static inline volatile uint64_t* tx_retired_flagp(NetCtx* c, int peer_rank);
static inline volatile uint64_t* cpu_done_flagp(NetCtx* c, int peer_rank);
static void wdog_dump(NetCtx* c, uint64_t posted, uint64_t matched) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/tp_wdog.rank%d.dump", c->rank);
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# rank %d world %d rounds %d slot_stride %u recv_gpu %d xchg_gen %llu (slot %llu)\n",
            c->rank, c->world, c->rounds, c->slot_stride, c->recv_gpu,
            (unsigned long long)c->gen, (unsigned long long)(c->gen & (TP_RING_SLOTS - 1)));
    fprintf(f, "# posted %llu matched %llu retired %llu released %llu device_epoch %llu\n",
            (unsigned long long)posted, (unsigned long long)matched,
            (unsigned long long)c->retired_epochs, (unsigned long long)c->released_epochs,
            (unsigned long long)(c->dev_ctx_h ? c->dev_ctx_h->epoch : 0));
    fprintf(f, "# gpu_ready %llu rx_done %llu abort %llu\n",
            (unsigned long long)*flagp(c, TP_F_GPU_READY),
            (unsigned long long)*flagp(c, TP_F_RX_DONE),
            (unsigned long long)*flagp(c, TP_F_ABORT));
    for (int p = 0; p < c->world && p < TP_NWAY_MAX_WORLD; p++) {
        fprintf(f, "# peer %d: peer_committed %llu tx_retired %llu cpu_done %llu\n", p,
                (unsigned long long)*peer_committed_flagp(c, p),
                (unsigned long long)*tx_retired_flagp(c, p),
                (unsigned long long)*cpu_done_flagp(c, p));
    }
    // len ring for the last 4096 epochs (code-path fingerprint; diff across ranks finds where a
    // rank's launch sequence diverged) + GPU stamps for the last 32 (phase of the stuck kernels).
    volatile uint64_t* len_local = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_LOCAL_OFF);
    uint64_t hi = posted > matched ? posted : matched;
    uint64_t lo = hi >= (TP_LEN_EPOCHS - 1) ? hi - (TP_LEN_EPOCHS - 1) : 1;
    for (uint64_t e = lo; e <= hi; e++) {
        uint64_t tag = len_local[e & (TP_LEN_EPOCHS - 1)];
        fprintf(f, "e %llu len %llu tag_epoch %llu", (unsigned long long)e,
                (unsigned long long)TP_LEN_TAG_BYTES(tag),
                (unsigned long long)TP_LEN_TAG_EPOCH(tag));
        if (c->gpu_ts_h && e + 32 > hi) {
            uint64_t* s = &c->gpu_ts_h[(e % TP_GTS_EPOCHS) * TP_GTS_STRIDE];
            fprintf(f, " k1in %llu k1out %llu k2in %llu k2go %llu",
                    (unsigned long long)s[TP_GTS_K1_IN], (unsigned long long)s[TP_GTS_K1_OUT],
                    (unsigned long long)s[TP_GTS_K2_IN], (unsigned long long)s[TP_GTS_K2_GO]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    LOGE("WATCHDOG dump: %s", path);
}

typedef struct { uint32_t qpn, psn; uint64_t addr; uint32_t rkey; union ibv_gid gid; } Exch
    __attribute__((packed));

// N-way-only exchange envelope (P4/R8): the same QP info as Exch PLUS the sender's rank, so the
// accept side can verify WHO connected before trusting the QP info. world==2 never uses this — its
// single-QP path keeps the untouched `tcp_exchange`/`Exch` above.
typedef struct { uint32_t rank; Exch e; } NwayExch __attribute__((packed));

static int tcp_exchange(int rank, const char* peer_ip, int port, Exch* lo, Exch* re,
                        char* peer_out, size_t peer_out_len) {
    int sock;
    if (rank == 0) {
        int ls = socket(AF_INET, SOCK_STREAM, 0); int o = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
        if (bind(ls,(struct sockaddr*)&a,sizeof(a))<0){ LOGE("bind: %s",strerror(errno)); return -1; }
        if (listen(ls,1)<0){ LOGE("listen"); return -1; }
        struct sockaddr_in pa; socklen_t plen = sizeof(pa); memset(&pa,0,sizeof(pa));
        sock = accept(ls, (struct sockaddr*)&pa, &plen); close(ls);
        if (sock<0){ LOGE("accept"); return -1; }
        if (peer_out) inet_ntop(AF_INET, &pa.sin_addr, peer_out, peer_out_len);
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,peer_ip,&a.sin_addr);
        int ok=-1; for(int i=0;i<400;i++){ if(connect(sock,(struct sockaddr*)&a,sizeof(a))==0){ok=0;break;} usleep(50000);}
        if (ok){ LOGE("connect %s:%d: %s",peer_ip,port,strerror(errno)); return -1; }
        if (peer_out) snprintf(peer_out, peer_out_len, "%s", peer_ip);
    }
    if (write(sock,lo,sizeof(*lo))!=(ssize_t)sizeof(*lo)){ LOGE("w exch"); close(sock); return -1; }
    if (read (sock,re,sizeof(*re))!=(ssize_t)sizeof(*re)){ LOGE("r exch"); close(sock); return -1; }
    close(sock);
    return 0;
}

// N-way-only, per-pair QP handshake (P4/R8). world==2 does NOT use this (it keeps the untouched
// tcp_exchange above). Two differences from tcp_exchange:
//   1. It carries an explicit sender-rank field in `NwayExch`; BOTH sides verify the received rank
//      equals the expected peer before trusting the QP info, so a mis-pair fails loudly instead of
//      silently corrupting the doubling-partner map.
//   2. Each ordered pair (min,max) has its OWN TCP port (`nway_pair_port`), derived deterministically
//      from the pair, so no two pairs contend for a listener. The LOWER rank binds/listens, the
//      HIGHER rank connects — exactly one listener per (min,max) port, no sequential rebind.
static int nway_pair_port(int tcp_port, int world, int a, int b) {
    // a,b are ranks; use the canonical min/max so both sides compute the same port. Offset +2 keeps
    // clear of the world==2 control port (tcp_port) and the liveness responder (tcp_port+1).
    int lo = a < b ? a : b;
    int hi = a < b ? b : a;
    return tcp_port + 2 + lo * world + hi;
}

// TP-4D (C8): how long a world > 2 pair listener waits for the higher rank to dial (ms).
#define NWAY_ACCEPT_TIMEOUT_MS 600000

static int nway_tcp_exchange(int my_rank, int peer_rank, const char* peer_ip, int port,
                             NwayExch* lo, NwayExch* re, char* peer_out, size_t peer_out_len) {
    int listener = (my_rank < peer_rank);
    int sock;
    if (listener) {
        int ls = socket(AF_INET, SOCK_STREAM, 0); int o = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
        if (bind(ls,(struct sockaddr*)&a,sizeof(a))<0){ LOGE("nway bind: %s",strerror(errno)); return -1; }
        if (listen(ls,1)<0){ LOGE("nway listen"); return -1; }
        struct sockaddr_in pa; socklen_t plen = sizeof(pa); memset(&pa,0,sizeof(pa));
        // TP-4D (audit C8, world > 2 ONLY — the world==2 tcp_exchange above keeps its bare accept): the listener
        // waits for the higher rank's dial at most NWAY_ACCEPT_TIMEOUT_MS. The head accepts its world-1 pairs
        // SEQUENTIALLY and a node dials only after its own pack check, so this bounds the skew of FOUR boots; a
        // peer that never dials (silent hang, wrong host) becomes a named error instead of an unbounded accept().
        // A peer that REPORTS its failure is handled earlier and faster by the head's node watch (cluster.rs).
        {
            struct pollfd pfd; memset(&pfd, 0, sizeof(pfd)); pfd.fd = ls; pfd.events = POLLIN;
            int pr;
            do { pr = poll(&pfd, 1, NWAY_ACCEPT_TIMEOUT_MS); } while (pr < 0 && errno == EINTR);
            if (pr <= 0) {
                LOGE("nway accept: rank %d never dialled rank %d on port %d within %d s (%s) — the peer did not "
                     "reach its link bring-up; check its log", peer_rank, my_rank, port, NWAY_ACCEPT_TIMEOUT_MS / 1000,
                     pr == 0 ? "timeout" : strerror(errno));
                close(ls);
                return -1;
            }
        }
        sock = accept(ls, (struct sockaddr*)&pa, &plen); close(ls);
        if (sock<0){ LOGE("nway accept"); return -1; }
        if (peer_out) inet_ntop(AF_INET, &pa.sin_addr, peer_out, peer_out_len);
    } else {
        // world>2: the head binds its per-pair RDMA listener only AFTER it has synced ALL world-1
        // nodes (multi-GB cold-cache transfer takes minutes), and it binds/accepts pairs SEQUENTIALLY
        // ((0,1) then (0,2) then (0,3) ...). A node therefore reaches this connect before the head's
        // listener for its pair exists and must keep retrying across the whole sync + handshake skew.
        // Each retry needs a FRESH socket: a socket that has seen a failed connect() cannot be
        // re-connected on Linux. Budget = 6000 * 50 ms = 300 s (the world==2 tcp_exchange path keeps
        // its proven 20 s window — this arm is only entered for world > 2).
        int ok = -1;
        for (int i = 0; i < 6000; i++) {
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) { LOGE("nway socket"); return -1; }
            struct sockaddr_in a; memset(&a,0,sizeof(a));
            a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,peer_ip,&a.sin_addr);
            if (connect(sock,(struct sockaddr*)&a,sizeof(a))==0){ ok=0; break; }
            close(sock);
            usleep(50000);
        }
        if (ok){ LOGE("nway connect %s:%d: %s",peer_ip,port,strerror(errno)); return -1; }
        if (peer_out) snprintf(peer_out, peer_out_len, "%s", peer_ip);
    }
    if (write(sock,lo,sizeof(*lo))!=(ssize_t)sizeof(*lo)){ LOGE("nway w exch"); close(sock); return -1; }
    if (read (sock,re,sizeof(*re))!=(ssize_t)sizeof(*re)){ LOGE("nway r exch"); close(sock); return -1; }
    close(sock);
    if (re->rank != (uint32_t)peer_rank) {
        LOGE("nway mis-pair: connected rank %u but expected peer rank %d (pair port %d) — refusing",
             re->rank, peer_rank, port);
        return -1;
    }
    return 0;
}

// Liveness responder (one per NetCtx, detached). The ONLY way a rank blocked in net_exchange can
// tell "peer process is dead" from "peer is slow" (model load, user idle between requests): the RC
// QP retries forever by design, so a dead peer yields no error CQE for a passive waiter — that was
// the 131-min 100%-CPU node spin. A successful connect here == this process is alive. Nothing more.
static void* liveness_responder(void* arg) {
    int port = (int)(intptr_t)arg;
    int ls = socket(AF_INET, SOCK_STREAM, 0); int o = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
    struct sockaddr_in a; memset(&a,0,sizeof(a));
    a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
    if (ls < 0 || bind(ls,(struct sockaddr*)&a,sizeof(a))<0 || listen(ls,8)<0) {
        // Non-fatal: OUR probes of the peer still work; the peer's probes of us will fail and it
        // will abort instead of hanging — the fail-safe direction.
        LOGE("WARNING: liveness responder on port %d unavailable (%s)", port, strerror(errno));
        if (ls >= 0) close(ls);
        return NULL;
    }
    for (;;) {
        int s = accept(ls, NULL, NULL);
        if (s >= 0) close(s);
        else { struct timespec ts = { .tv_sec = 0, .tv_nsec = 50000000 }; nanosleep(&ts, NULL); }
    }
    return NULL;
}

// Non-blocking TCP connect with timeout: 0 = peer process alive, nonzero = unreachable.
static int probe_once(const char* ip, int port, int timeout_ms) {
    int s = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (s < 0) return -1;
    struct sockaddr_in a; memset(&a,0,sizeof(a));
    a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,ip,&a.sin_addr);
    int rc = connect(s,(struct sockaddr*)&a,sizeof(a));
    if (rc == 0) { close(s); return 0; }
    if (errno != EINPROGRESS) { close(s); return -1; }
    struct pollfd pfd; memset(&pfd,0,sizeof(pfd)); pfd.fd = s; pfd.events = POLLOUT;
    if (poll(&pfd, 1, timeout_ms) <= 0) { close(s); return -1; }
    int err = 0; socklen_t len = sizeof(err);
    if (getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) { close(s); return -1; }
    close(s); return 0;
}

// Two attempts 500 ms apart — a single dropped SYN on an idle LAN must not kill a healthy session.
static int peer_alive(NetCtx* c) {
    if (probe_once(c->peer_ip, c->liveness_port, 2000) == 0) return 1;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000000 }; nanosleep(&ts, NULL);
    return probe_once(c->peer_ip, c->liveness_port, 2000) == 0;
}
// world>2: the same dead-peer probe, keyed on a specific peer's RoCE IP (its liveness port is shared).
static int peer_alive_at(NetCtx* c, int peer_rank) {
    const char* ip = c->peers[peer_rank].peer_ip;
    if (!ip || !ip[0]) return 0;
    if (probe_once(ip, c->liveness_port, 2000) == 0) return 1;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 500000000 }; nanosleep(&ts, NULL);
    return probe_once(ip, c->liveness_port, 2000) == 0;
}

// ---------------------------------------------------------------- N-way schedule
// Recursive doubling: rounds = log2(world), round = epoch % rounds, partner = rank ^ (1<<round).
// Deterministic and identical on every rank (SPMD lockstep). The device epoch advances once per ROUND
// (a logical all-reduce at a model site consumes `rounds` consecutive epochs). world==2 degenerates to
// rounds=1, partner = rank ^ 1 = the single peer — which is why the legacy single-QP path is unchanged.
static inline int nway_partner_rank(int rank, int round) { return rank ^ (1 << round); }
static inline int nway_round_of(NetCtx* c, uint64_t e) {
    return (int)(e % (unsigned)c->rounds);
}
// Per-peer flag entry: three sub-arrays of TP_CL each, indexed by PEER RANK.
static inline volatile uint64_t* nway_flagp(NetCtx* c, int peer_rank, size_t sub_off) {
    return (volatile uint64_t*)((char*)c->hbuf + c->nway_flags_off
                                + (size_t)peer_rank * TP_CL + sub_off);
}
// The peer whose QP owns epoch e (world>2). At world==2 this is never called.
static inline PeerLink* active_peer(NetCtx* c, uint64_t e) {
    int p = nway_partner_rank(c->rank, nway_round_of(c, e));
    return &c->peers[p];
}

// Unified accessors: at world==2 they return the legacy single-QP fields (byte-identical behavior);
// at world>2 they index the per-peer array. `peer_rank` is the PARTNER rank (rank ^ (1<<round)).
static inline int partner_rank_of(NetCtx* c, uint64_t e) {
    // world==2: rounds==1 -> round 0 -> partner = rank ^ 1 = the single peer.
    return nway_partner_rank(c->rank, nway_round_of(c, e));
}
static inline struct ibv_qp* peer_qp(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? c->qp : c->peers[peer_rank].qp;
}
static inline uint64_t peer_remote_addr(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? c->remote_addr : c->peers[peer_rank].remote_addr;
}
static inline uint32_t peer_remote_rkey(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? c->remote_rkey : c->peers[peer_rank].remote_rkey;
}
static inline volatile uint64_t* peer_committed_flagp(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? flagp(c, TP_F_PEER_COMMITTED) : nway_flagp(c, peer_rank, TP_NWAY_PEER_OFF);
}
static inline volatile uint64_t* cpu_done_flagp(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? flagp(c, TP_F_CPU_DONE) : nway_flagp(c, peer_rank, TP_NWAY_CPU_OFF);
}
static inline volatile uint64_t* tx_retired_flagp(NetCtx* c, int peer_rank) {
    return (c->world == 2) ? flagp(c, TP_F_TX_RETIRED) : nway_flagp(c, peer_rank, TP_NWAY_TX_OFF);
}
// world>2: the RDMA target of the commit WR (the peer's peer_committed slot indexed by OUR rank).
static inline uint64_t peer_committed_raddr(NetCtx* c, int peer_rank) {
    if (c->world == 2) return c->remote_addr + TP_F_PEER_COMMITTED;
    return c->peers[peer_rank].remote_addr + c->nway_flags_off
         + (size_t)c->rank * TP_CL + TP_NWAY_PEER_OFF;
}
static inline uint64_t peer_recv_raddr_at(uint64_t remote_addr, unsigned slot_stride, uint64_t e) {
    return remote_addr + TP_RING_BASE + (uint64_t)TP_RING_SLOTS * slot_stride
         + (uint64_t)(e & (TP_RING_SLOTS - 1)) * slot_stride;
}
// world>2 per-round recv ring (see TP_NWAY_MAX_ROUNDS in tp_doorbell.h): epoch e lands in
// round(e)'s own R-slot ring at slot e % R, so a round's payload can never alias another round's
// slot. Both sides compute round(e) = e % rounds identically from the (SPMD) epoch. world==2 never
// calls these (its recv path keeps the shared recv_ring + (e%R)*stride, byte-identical).
static inline uint64_t nway_recv_slot_off(NetCtx* c, uint64_t e) {
    uint64_t round = e % (unsigned)c->rounds;
    return c->nway_recv_off
         + (round * (uint64_t)TP_RING_SLOTS + (e & (TP_RING_SLOTS - 1))) * c->slot_stride;
}
static inline char* nway_recv_slot_ptr(NetCtx* c, uint64_t e) {
    return (char*)c->hbuf + nway_recv_slot_off(c, e);
}
static inline uint64_t nway_peer_recv_raddr(NetCtx* c, int peer_rank, uint64_t e) {
    return c->peers[peer_rank].remote_addr + nway_recv_slot_off(c, e);
}
// TP-4F2: the SENDER-indexed one-shot block's slot for sender p, epoch e (what post_range_oneshot writes:
// oneshot_recv_off + (sender*R + e%R)*stride). Only the oneshot_uniform ctx's RECV branch reads it on the CPU.
static inline char* oneshot_recv_slot_ptr(NetCtx* c, int p, uint64_t e) {
    return (char*)c->hbuf + c->oneshot_recv_off
         + ((uint64_t)p * (uint64_t)TP_RING_SLOTS + (e & (TP_RING_SLOTS - 1))) * c->slot_stride;
}

// ---- R9 DIAGNOSTIC helpers (world>2, tp_diag only) ----
#define TP_DIAG_RING_EPOCHS 65536ull   // power of two; mismatch aborts long before a wrap matters
static uint64_t tp_fnv64(const void* p, uint64_t len) {
    const unsigned char* b = (const unsigned char*)p;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h ? h : 1;                    // never 0 — 0 is the ring's empty sentinel
}
static void tp_diag_log_send(NetCtx* c, uint64_t e, int p, const char* payload, uint64_t len) {
    if (!c->tp_diag) return;
    uint64_t h = tp_fnv64(payload, len);
    volatile uint64_t* r = (volatile uint64_t*)((char*)c->hbuf + c->diag_send);
    uint64_t i = c->diag_send_idx++;
    uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
    r[j + 0] = e;
    r[j + 1] = (uint64_t)p;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    r[j + 2] = h;
    c->diag_send_fold[p] = c->diag_send_fold[p]
        ? (c->diag_send_fold[p] ^ h) * 1099511628211ull : h;
}
static void tp_diag_log_recv(NetCtx* c, uint64_t e, int p, const char* payload, uint64_t len) {
    if (!c->tp_diag) return;
    uint64_t h = tp_fnv64(payload, len);
    volatile uint64_t* r = (volatile uint64_t*)((char*)c->hbuf + c->diag_recv);
    uint64_t i = c->diag_recv_idx++;
    uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
    r[j + 0] = e;
    r[j + 1] = (uint64_t)p;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    r[j + 2] = h;
    c->diag_recv_fold[p] = c->diag_recv_fold[p]
        ? (c->diag_recv_fold[p] ^ h) * 1099511628211ull : h;
}
// O(1) ambiguity detectors, shipped in the agree frame's [0..8)/[8..16) words (the frame stays
// 32 B, byte-identical): the XOR of every send/recv ring entry's (epoch<<32)|partner tag. All
// ranks execute the identical SPMD barrier schedule, so equal values across ranks PROVE every
// rank sent/received the identical epoch set per direction — then the first differing 64-entry
// frame is unambiguously the first bad delivery (frame/4 -> epoch, sender, receiver).
uint64_t net_diag_send_xor(NetCtx* c) {
    if (!c->tp_diag) return 0;
    volatile uint64_t* r = (volatile uint64_t*)((char*)c->hbuf + c->diag_send);
    uint64_t x = 0, n = c->diag_send_idx;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
        x ^= (r[j] << 32) | r[j + 1];
    }
    return x;
}
uint64_t net_diag_recv_xor(NetCtx* c) {
    if (!c->tp_diag) return 0;
    volatile uint64_t* r = (volatile uint64_t*)((char*)c->hbuf + c->diag_recv);
    uint64_t x = 0, n = c->diag_recv_idx;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
        x ^= (r[j] << 32) | r[j + 1];
    }
    return x;
}
void* net_diag_send_hptr(NetCtx* c) { return (char*)c->hbuf + c->diag_send; }
void* net_diag_recv_hptr(NetCtx* c) { return (char*)c->hbuf + c->diag_recv; }
uint64_t net_diag_send_idx(NetCtx* c) { return c->diag_send_idx; }
uint64_t net_diag_recv_idx(NetCtx* c) { return c->diag_recv_idx; }
void net_diag_folds(NetCtx* c, uint64_t* send_fold, uint64_t* recv_fold) {
    for (int p = 0; p < c->world; p++) { send_fold[p] = c->diag_send_fold[p]; recv_fold[p] = c->diag_recv_fold[p]; }
}
// Dump this rank's two rings in an easily diffable text form (called on the agree-mismatch path).
void net_diag_dump(NetCtx* c, const char* path) {
    if (!c->tp_diag) return;
    char full[512];
    snprintf(full, sizeof(full), "%s.rank%d", path, c->rank);
    FILE* f = fopen(full, "w");
    if (!f) { LOGE("diag dump: fopen(%s): %s", full, strerror(errno)); return; }
    fprintf(f, "# rank %d world %d send_idx %llu recv_idx %llu\n", c->rank, c->world,
            (unsigned long long)c->diag_send_idx, (unsigned long long)c->diag_recv_idx);
    fprintf(f, "# send folds:"); for (int p = 0; p < c->world; p++) fprintf(f, " %d=%016llx", p, (unsigned long long)c->diag_send_fold[p]);
    fprintf(f, "\n# recv folds:"); for (int p = 0; p < c->world; p++) fprintf(f, " %d=%016llx", p, (unsigned long long)c->diag_recv_fold[p]);
    fprintf(f, "\n# dir epoch partner fnv64\n");
    volatile uint64_t* rs = (volatile uint64_t*)((char*)c->hbuf + c->diag_send);
    for (uint64_t i = 0; i < c->diag_send_idx; i++) {
        uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
        fprintf(f, "S %llu %llu %016llx\n", (unsigned long long)rs[j], (unsigned long long)rs[j+1], (unsigned long long)rs[j+2]);
    }
    volatile uint64_t* rr = (volatile uint64_t*)((char*)c->hbuf + c->diag_recv);
    for (uint64_t i = 0; i < c->diag_recv_idx; i++) {
        uint64_t j = (i & (TP_DIAG_RING_EPOCHS - 1)) * 3;
        fprintf(f, "R %llu %llu %016llx\n", (unsigned long long)rr[j], (unsigned long long)rr[j+1], (unsigned long long)rr[j+2]);
    }
    fclose(f);
    LOGE("diag dump written: %s (send %llu, recv %llu)", full,
         (unsigned long long)c->diag_send_idx, (unsigned long long)c->diag_recv_idx);
}
// world>2 control-plane QP (head<->node): toward rank 0, or toward rank 1 if we are the head.
static inline int control_peer_rank(NetCtx* c) { return (c->rank == 0) ? 1 : 0; }
static inline struct ibv_qp* control_qp(NetCtx* c) {
    return (c->world == 2) ? c->qp : c->peers[control_peer_rank(c)].qp;
}
static inline uint64_t control_remote_addr(NetCtx* c) {
    return (c->world == 2) ? c->remote_addr : c->peers[control_peer_rank(c)].remote_addr;
}
static inline uint32_t control_remote_rkey(NetCtx* c) {
    return (c->world == 2) ? c->remote_rkey : c->peers[control_peer_rank(c)].remote_rkey;
}
// world>2 control receive slots, RING-KEYED by the pair's generation (R10): sender `src`'s
// exchange with generation g lands at ctrl_recv_off + (src*TP_CTRL_RING + g%TP_CTRL_RING)*stride.
// The old single-slot-per-sender design had a clobber race: the peer's exchange returns as soon as
// OUR token lands on ITS side, so it could post its NEXT exchange (g+1) into our single slot
// before we observed g — the tag wait then spins forever on a slot holding g+1 and both sides
// deadlock (observed at world=4 a few steps into decode; the hang's step varied with timing).
// A pair can run at most one agree (2 exchanges) ahead, so a depth-4 ring cannot alias.
// world==2 never calls these.
static inline char* ctrl_recv_slot(NetCtx* c, int src, uint64_t g) {
    return (char*)c->hbuf + c->ctrl_recv_off
         + ((size_t)src * TP_CTRL_RING + (size_t)(g % TP_CTRL_RING)) * c->slot_stride;
}
static inline uint64_t ctrl_recv_raddr(NetCtx* c, int peer_rank, uint64_t g) {
    return c->peers[peer_rank].remote_addr + c->ctrl_recv_off
         + ((size_t)c->rank * TP_CTRL_RING + (size_t)(g % TP_CTRL_RING)) * c->slot_stride;
}
// Stable "last received from src" slot: exchange_one copies the validated ring payload here before
// returning, so callers reading via net_ctrl_recv_hptr need no generation bookkeeping.
static inline char* ctrl_last_slot(NetCtx* c, int src) {
    return (char*)c->hbuf + c->ctrl_last_off + (size_t)src * c->slot_stride;
}
// Dedicated control SEND staging slot (world>2), separate from the hot-path send ring and net_send_hptr.
static inline char* ctrl_send_slot(NetCtx* c) {
    return (char*)c->hbuf + c->ctrl_send_off;
}

// S-B9-REL-TP2RACE (issue #10): the world-2 exchange's slot arithmetic. At world 2 the frame stages
// in the DEDICATED ctrl send slot and lands in the peer's ctrl recv ring, slot (src, g%TP_CTRL_RING)
// — memory no doorbell epoch can touch, so an in-flight/unconsumed epoch payload can never be
// clobbered by a control frame and the frame can never be clobbered by an epoch. World>2 keeps the
// historical hot-ring addressing for this pairwise channel (its lockstep rides net_exchange_one).
static inline char* xchg_stage_slot(NetCtx* c) {
    return (c->world == 2) ? ctrl_send_slot(c) : (char*)c->hbuf + TP_RING_BASE;
}
static inline char* xchg_recv_slot_ptr(NetCtx* c, uint64_t g) {
    if (c->world != 2) return recv_slot(c, g);
    return ctrl_recv_slot(c, 1 - c->rank, g);
}
static inline uint64_t xchg_remote_slot(NetCtx* c, uint64_t g) {
    if (c->world != 2)
        return (uint64_t)((char*)control_remote_addr(c) + TP_RING_BASE
                          + (size_t)TP_RING_SLOTS * c->slot_stride)
               + (size_t)(g & (TP_RING_SLOTS - 1)) * c->slot_stride;
    return control_remote_addr(c)
         + (uint64_t)((char*)ctrl_recv_slot(c, c->rank, g) - (char*)c->hbuf);
}

// ---------------------------------------------------------------- init

// Forward declarations (defined below; net_init dispatches to them).
static NetCtx* net_init_world2(int rank, const char* peer_ip, int tcp_port, const char* dev_name,
                               int gid_idx, int fp32_capacity_bytes, int payload_bytes);
static NetCtx* net_init_nway(int rank, int world, const char* const* peer_ips,
                             int tcp_port, const char* dev_name, int gid_idx,
                             int fp32_capacity_bytes, int payload_bytes);
static void net_proxy_loop_world2(NetCtx* c, int core);
static void net_proxy_loop_nway(NetCtx* c, int core);
static int nway_recv_slot(NetCtx* c, uint64_t e, int p);
static int nway_recv_slot_at(NetCtx* c, uint64_t e, int p, char* slot);
static int peer_alive_at(NetCtx* c, int peer_rank);

// rank: 0 = listen (head), 1 = connect (node).
// `fp32_capacity_bytes` sizes the ring slots (allocate for the FP32 payload from day one so the
// precision switch never re-addresses the rings); `payload_bytes` is what actually ships this run.
//
// P3 N-way: `world` is the TP rank count and `peer_ips[0..world)` is the per-rank peer-IP list
// (indexed by PEER RANK; entry [rank] is this rank and unused). `n_peers` = world-1. world==2 MUST
// take the exact single-QP path it always took (R1): it dispatches to net_init_world2, whose body is
// the pre-P3 net_init verbatim (the peer is peer_ips[1-rank]).
NetCtx* net_init(int rank, int world, const char* const* peer_ips, int n_peers,
                 int tcp_port, const char* dev_name, int gid_idx,
                 int fp32_capacity_bytes, int payload_bytes) {
    if (payload_bytes <= 0 || payload_bytes > fp32_capacity_bytes) {
        LOGE("payload_bytes %d out of range (capacity %d)", payload_bytes, fp32_capacity_bytes);
        return NULL;
    }
    if (TP_SIGNAL_EVERY > TP_RING_SLOTS) { LOGE("I4 violated: S=%d > R=%d", TP_SIGNAL_EVERY, TP_RING_SLOTS); return NULL; }
    if (world < 2 || world > TP_NWAY_MAX_WORLD || (world & (world - 1)) != 0) {
        LOGE("world %d must be a power of two in [2,%d]", world, TP_NWAY_MAX_WORLD);
        return NULL;
    }
    if (n_peers != world - 1 || !peer_ips) {
        LOGE("net_init: n_peers=%d != world-1=%d, or peer_ips is NULL", n_peers, world - 1);
        return NULL;
    }

    if (world == 2) {
        // R1 hard rule: the world==2 peer for rank r is peer_ips[1-r] — identical to the old
        // single `peer_ip` (rank 0 -> peer_ips[1], rank 1 -> peer_ips[0]).
        return net_init_world2(rank, peer_ips[1 - rank], tcp_port, dev_name, gid_idx,
                               fp32_capacity_bytes, payload_bytes);
    }
    return net_init_nway(rank, world, peer_ips, tcp_port, dev_name, gid_idx,
                         fp32_capacity_bytes, payload_bytes);
}

// The pre-P3 single-QP bring-up, byte-for-byte. MUST NOT change (world==2 fast path).
// S-B9-REL-TP2RACE side item (REL-NGRAM one transient bring-up failure): the link pinned regions
// are small but a cudaHostAlloc can fail transiently right after the model + primary rail came up
// (unified-memory squeeze) — the head never reached its TCP listener and the node logged
// `connect :29600 refused`. Retry a few times before giving up; persistent failure stays loud.
static void* host_alloc_with_retry(size_t bytes) {
    void* p = NULL;
    for (int i = 0; i < 5; i++) {
        if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped|cudaHostAllocPortable) == cudaSuccess) {
            if (i) LOGE("cudaHostAlloc(%zu) succeeded on retry %d", bytes, i);
            return p;
        }
        LOGE("cudaHostAlloc(%zu) failed (attempt %d/5) — retrying in 1 s", bytes, i + 1);
        struct timespec d = { .tv_sec = 1, .tv_nsec = 0 };
        nanosleep(&d, NULL);
    }
    return NULL;
}

static NetCtx* net_init_world2(int rank, const char* peer_ip, int tcp_port, const char* dev_name,
                               int gid_idx, int fp32_capacity_bytes, int payload_bytes) {
    NetCtx* c = (NetCtx*)calloc(1, sizeof(NetCtx));
    c->port_num = 1; c->gid_idx = gid_idx; c->rank = rank; c->rng = 0x9E3779B9u ^ (unsigned)rank;
    c->payload_bytes = (unsigned)payload_bytes;
    c->world = 2; c->rounds = 1;
    c->tail_drill = g_opt_tail_drill != 0;   // test-only (--tp-tail-drill): see post_range
    if (c->tail_drill) LOGE("TAIL DRILL ON: inverting commit/payload order every 4096th epoch");
    // slot = payload capacity + 8 B tail epoch, 64 B aligned so no two slots share a line
    c->slot_stride = (unsigned)(((size_t)fp32_capacity_bytes + TP_TAIL_BYTES + TP_CL - 1) & ~(size_t)(TP_CL - 1));
    // S-B9-REL-TP2RACE (issue #10): world 2 now appends the DEDICATED control region after the
    // doorbell rings — the same region world>2 has had since P5 — and every world-2 host exchange
    // (net_exchange: pre-verify, tp_round, step-go, bring-up) stages and lands there instead of the
    // hot-path rings. The R1 "byte-for-byte world==2" rule is kept for everything the device sees:
    // no existing offset moves (flags, len rings, both rings, dev_ctx) — the MR simply grows by
    // 11 slots (~11.5 MB pinned). The pre-verify exchange frame can then never overwrite a peer
    // epoch payload that is validated but not yet consumed by its K2 (the #10 race: drafts diverge,
    // "TP pre-verify FAILED", 4 head exits over v0.7.0-0.7.2), and no drain/fence is needed for it.
    c->ctrl_recv_off = TP_RING_BASE + (size_t)2 * TP_RING_SLOTS * c->slot_stride;
    c->ctrl_last_off = c->ctrl_recv_off + (size_t)2 * TP_CTRL_RING * c->slot_stride;
    c->ctrl_send_off = c->ctrl_last_off + (size_t)2 * c->slot_stride;
    c->region_bytes  = c->ctrl_send_off + c->slot_stride;
    c->probe_hold_ms = g_opt_probe_hold_ms;
    if (c->probe_hold_ms)
        LOGE("TP RACE PROBE ON: rank %d holds every epoch release by %u ms and watches the withheld "
             "slot for a host exchange frame (issue #10 fault injection; test-only)", c->rank, c->probe_hold_ms);

    int n=0; struct ibv_device** devs = ibv_get_device_list(&n);
    if (!devs || n<=0){ LOGE("get_device_list"); return NULL; }
    struct ibv_device* dev=NULL;
    for (int i=0;i<n;i++) if(!strcmp(ibv_get_device_name(devs[i]), dev_name)) dev=devs[i];
    if (!dev){ LOGE("device %s not found", dev_name); return NULL; }
    c->ctx = ibv_open_device(dev);
    c->pd  = ibv_alloc_pd(c->ctx);
    // R2b: the hot path gets its own CQ so it never shares a wr_id/opcode namespace with the retained
    // startup WITH_IMM channel (whose recv WQEs stay posted — un-posting them would resurrect RNR).
    c->cq_send    = ibv_create_cq(c->ctx, 256, NULL, NULL, 0);
    c->cq_startup = ibv_create_cq(c->ctx, 256, NULL, NULL, 0);
    if (!c->ctx || !c->pd || !c->cq_send || !c->cq_startup){ LOGE("ctx/pd/cq"); return NULL; }

    if ((c->hbuf = (char*)host_alloc_with_retry(c->region_bytes)) == NULL){
        LOGE("cudaHostAlloc(%zu)", c->region_bytes); return NULL; }
    memset(c->hbuf, 0, c->region_bytes);
    if (cudaHostGetDevicePointer(&c->dbuf, c->hbuf, 0) != cudaSuccess){ LOGE("devptr"); return NULL; }
    // I7: NO IBV_ACCESS_RELAXED_ORDERING. The per-MR flag is the actual switch on mlx5 (PCIe DevCtl
    // RlxdOrd+ is only "permitted"); leaving it off keeps remote writes strongly ordered, which is what
    // the CPU-bounce receive relies on. The tail-epoch guard below is the runtime proof.
    c->mr = ibv_reg_mr(c->pd, c->hbuf, c->region_bytes,
        IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
    if (!c->mr){ LOGE("reg_mr on cudaHostAlloc buffer (%s)", strerror(errno)); return NULL; }

    // Device ctx: mapped pinned so the host can read the device epoch counter (I8/Q4 tripwire).
    if ((c->dev_ctx_h = (tp_dev_ctx*)host_alloc_with_retry(sizeof(tp_dev_ctx))) == NULL){
        LOGE("cudaHostAlloc(dev_ctx)"); return NULL; }
    memset(c->dev_ctx_h, 0, sizeof(tp_dev_ctx));
    if (cudaHostGetDevicePointer(&c->dev_ctx_d, c->dev_ctx_h, 0) != cudaSuccess){ LOGE("devptr ctx"); return NULL; }
    c->dev_ctx_h->epoch         = 0;
    c->dev_ctx_h->flags         = (unsigned long long*)c->dbuf;
    c->dev_ctx_h->send_ring     = (unsigned char*)c->dbuf + TP_RING_BASE;
    c->dev_ctx_h->recv_ring     = (unsigned char*)c->dbuf + TP_RING_BASE + (size_t)TP_RING_SLOTS * c->slot_stride;
    c->dev_ctx_h->gpu_ts        = NULL;
    c->dev_ctx_h->slot_stride   = c->slot_stride;
    c->dev_ctx_h->payload_bytes = c->payload_bytes;
    c->dev_ctx_h->rank          = rank;
    c->dev_ctx_h->fp32_payload  = 0;
    c->dev_ctx_h->len_local     = (unsigned long long*)((char*)c->dbuf + TP_LEN_LOCAL_OFF);
    c->dev_ctx_h->world         = 2;
    c->dev_ctx_h->rounds        = 1;
    c->dev_ctx_h->nway_flags    = NULL;

    struct ibv_qp_init_attr qia; memset(&qia,0,sizeof(qia));
    qia.send_cq=c->cq_send; qia.recv_cq=c->cq_startup; qia.qp_type=IBV_QPT_RC;
    qia.cap.max_send_wr=256; qia.cap.max_recv_wr=256; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1;
    qia.cap.max_inline_data=64;            // I1: the 8 B epoch must fit inline
    c->qp = ibv_create_qp(c->pd,&qia);
    if (!c->qp){ LOGE("create_qp"); return NULL; }

    struct ibv_qp_attr a; memset(&a,0,sizeof(a));
    a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=c->port_num;
    a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(c->qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ LOGE("INIT"); return NULL; }

    union ibv_gid mygid;
    if (ibv_query_gid(c->ctx,c->port_num,gid_idx,&mygid)){ LOGE("query_gid"); return NULL; }
    Exch lo, re; memset(&lo,0,sizeof(lo));
    lo.qpn=c->qp->qp_num; lo.psn=0x1000+rank*333; lo.addr=(uint64_t)c->hbuf; lo.rkey=c->mr->rkey; lo.gid=mygid;
    // Liveness responder BEFORE the handshake: the peer can start probing the moment it connects.
    c->liveness_port = tcp_port + 1;
    {
        pthread_t th;
        if (pthread_create(&th, NULL, liveness_responder, (void*)(intptr_t)c->liveness_port) == 0)
            pthread_detach(th);
        else
            LOGE("WARNING: liveness responder thread failed to start (%s)", strerror(errno));
    }
    if (tcp_exchange(rank, peer_ip, tcp_port, &lo, &re, c->peer_ip, sizeof(c->peer_ip))) return NULL;
    c->remote_addr=re.addr; c->remote_rkey=re.rkey;

    memset(&a,0,sizeof(a));
    a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096; a.dest_qp_num=re.qpn; a.rq_psn=re.psn;
    a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
    a.ah_attr.is_global=1; a.ah_attr.port_num=c->port_num;
    a.ah_attr.grh.dgid=re.gid; a.ah_attr.grh.sgid_index=gid_idx; a.ah_attr.grh.hop_limit=1;
    if (ibv_modify_qp(c->qp,&a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|
        IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)){ LOGE("RTR"); return NULL; }
    memset(&a,0,sizeof(a));
    a.qp_state=IBV_QPS_RTS; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.sq_psn=lo.psn; a.max_rd_atomic=1;
    if (ibv_modify_qp(c->qp,&a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|
        IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)){ LOGE("RTS"); return NULL; }

    for (int i=0;i<64;i++){ struct ibv_recv_wr wr, *bad; memset(&wr,0,sizeof(wr)); wr.num_sge=0;
        if (ibv_post_recv(c->qp,&wr,&bad)){ LOGE("post_recv init"); return NULL; } }
    return c;
}

// N-way (world > 2): world-1 per-peer RC QPs + the per-peer flag cache-lines. The send/recv/len rings
// are SHARED (one epoch in flight against one partner at a time — the round schedule is serial).
//
// TCP handshake rule: rank 0 listens for every peer (its control QP toward peer p is established by
// accepting p's connect); ranks 1..N-1 connect to the head for the CONTROL QP (rank 0, peer_ips[0]) and
// connect/listen for their doubling partners per a deterministic rule. To keep the tcp_exchange
// accept-side symmetric WITHOUT a full ordering protocol in P3 (P4 owns topology), we reuse tcp_exchange
// semantics: the LOWER rank listens, the HIGHER rank connects — so exactly one side binds each (a,b)
// pair and every QP pairs deterministically. The control QP (toward rank 0) still rides this rule:
// rank 0 (lower) listens, rank r>0 connects. peer_ip for a given partner is peer_ips[partner_rank].
static NetCtx* net_init_nway(int rank, int world, const char* const* peer_ips,
                             int tcp_port, const char* dev_name, int gid_idx,
                             int fp32_capacity_bytes, int payload_bytes) {
    NetCtx* c = (NetCtx*)calloc(1, sizeof(NetCtx));
    c->port_num = 1; c->gid_idx = gid_idx; c->rank = rank; c->rng = 0x9E3779B9u ^ (unsigned)rank;
    c->payload_bytes = (unsigned)payload_bytes;
    c->world = world; c->rounds = 0; while ((1 << c->rounds) < world) c->rounds++;
    // P3-1 one-shot push (--tp-oneshot, world==4 only, DEFAULT OFF): the nway recv region
    // holds `world` SENDER-indexed rings instead of `rounds` per-round rings (world(4) > rounds(2)
    // at world==4, so the region grows 2 slots — same allocation class). The option is the ONLY
    // selector; every rank resolves the same value at init (the head's option rides TpConfig to the
    // node before the link comes up; a mismatch aborts at the first barrier, loud by construction).
    c->oneshot_uniform = (world == 4 && g_oneshot_uniform_once) ? 1 : 0;   // TP-4F2: a dedicated uniform ctx
    g_oneshot_uniform_once = 0;                                              // consumed: never leaks to the next init
    c->oneshot = (world == 4 && (g_opt_oneshot || c->oneshot_uniform)) ? 1u : 0u;
    if (c->oneshot_uniform) LOGE("TP-4F2 SINGLE-STAGE REDUCE CTX (world=%d): uniform one-shot all-peers push, "
                                 "sender-indexed recv rings, proxy releases all peers together", world);
    else if (c->oneshot) LOGE("P3-1 ONE-SHOT PUSH ACTIVE (world=%d): sender-indexed recv rings, "
                         "all-peers post, tp_wait_add_4way", world);
    c->tail_drill = g_opt_tail_drill != 0;
    if (c->tail_drill) LOGE("TAIL DRILL ON: inverting commit/payload order every 4096th epoch");
    c->slot_stride = (unsigned)(((size_t)fp32_capacity_bytes + TP_TAIL_BYTES + TP_CL - 1) & ~(size_t)(TP_CL - 1));
    // Layout: [world==2 rings: send R + recv R][per-round recv rings rounds*R][per-peer flags]
    // [control recv world slots][control send 1 slot]. The per-round recv rings sit AFTER the
    // world==2 rings so the legacy layout is untouched; they give each recursive-doubling round its
    // own R-slot recv ring (the cross-round staleness fix — see TP_NWAY_MAX_ROUNDS in tp_doorbell.h).
    // Then the per-peer flag region (every peer's flags on their own lines), then the dedicated
    // per-rank control receive region: rank r's control payload lands at ctrl_recv_off +
    // r*slot_stride — a distinct full slot per rank, so a node's reply can never clobber another
    // node's on the head (net_exchange_one, P5).
    uint64_t rings_end   = TP_RING_BASE + (size_t)2 * TP_RING_SLOTS * c->slot_stride;
    c->nway_recv_off  = rings_end;
    // P3-1: SEPARATE REGIONS — the tree keeps its `rounds` round-keyed rings, and one-shot gets
    // its OWN `world` sender-indexed ring block appended after them. Mixed-algorithm epochs (tree
    // for wide/fp32/maxloc, one-shot for decode-width bf16) then NEVER share a slot: the two
    // addressing schemes live in disjoint memory. (The first attempt reused the same region —
    // rank 0's tree round-1 writes collided with rank 1's sender-ring-1 writes: the silent-wedge /
    // prefill-divergence class.)
    c->oneshot_recv_off = c->nway_recv_off + (size_t)c->rounds * TP_RING_SLOTS * c->slot_stride;
    c->nway_flags_off = c->oneshot_recv_off
        + (size_t)(c->oneshot ? world : 0) * TP_RING_SLOTS * c->slot_stride;
    c->ctrl_recv_off  = c->nway_flags_off + TP_NWAY_FLAGS_BYTES;
    c->ctrl_last_off  = c->ctrl_recv_off + (size_t)world * TP_CTRL_RING * c->slot_stride;
    c->ctrl_send_off  = c->ctrl_last_off + (size_t)world * c->slot_stride;
    // R9 DIAGNOSTIC (--tp-diag): two (epoch, partner, fnv64) rings appended after the control
    // slots. Send-side first, then recv-side; both sized TP_DIAG_RING_EPOCHS deep. Everything stays
    // zero/off unless the option is set, so production layout timing is unaffected at world>2 and
    // world==2 never reaches this initializer at all.
    c->tp_diag = g_opt_tp_diag != 0;
    c->diag_send = c->ctrl_send_off + c->slot_stride;
    c->diag_recv = c->diag_send + (uint64_t)(c->tp_diag ? 1 : 0) * TP_DIAG_RING_EPOCHS * 3 * 8;
    c->region_bytes = c->diag_recv + (uint64_t)(c->tp_diag ? 1 : 0) * TP_DIAG_RING_EPOCHS * 3 * 8;
    if (c->tp_diag)
        LOGE("TP DIAG ON: per-epoch payload checksum rings at +%llu/+%llu (%llu entries each)",
             (unsigned long long)c->diag_send, (unsigned long long)c->diag_recv,
             (unsigned long long)TP_DIAG_RING_EPOCHS);

    int n=0; struct ibv_device** devs = ibv_get_device_list(&n);
    if (!devs || n<=0){ LOGE("get_device_list"); return NULL; }
    struct ibv_device* dev=NULL;
    for (int i=0;i<n;i++) if(!strcmp(ibv_get_device_name(devs[i]), dev_name)) dev=devs[i];
    if (!dev){ LOGE("device %s not found", dev_name); return NULL; }
    c->ctx = ibv_open_device(dev);
    c->pd  = ibv_alloc_pd(c->ctx);
    c->cq_send    = ibv_create_cq(c->ctx, 512, NULL, NULL, 0);
    c->cq_startup = ibv_create_cq(c->ctx, 256, NULL, NULL, 0);
    if (!c->ctx || !c->pd || !c->cq_send || !c->cq_startup){ LOGE("ctx/pd/cq"); return NULL; }

    if ((c->hbuf = (char*)host_alloc_with_retry(c->region_bytes)) == NULL){
        LOGE("cudaHostAlloc(%zu)", c->region_bytes); return NULL; }
    memset(c->hbuf, 0, c->region_bytes);
    if (cudaHostGetDevicePointer(&c->dbuf, c->hbuf, 0) != cudaSuccess){ LOGE("devptr"); return NULL; }
    c->mr = ibv_reg_mr(c->pd, c->hbuf, c->region_bytes,
        IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
    if (!c->mr){ LOGE("reg_mr on cudaHostAlloc buffer (%s)", strerror(errno)); return NULL; }

    if ((c->dev_ctx_h = (tp_dev_ctx*)host_alloc_with_retry(sizeof(tp_dev_ctx))) == NULL){
        LOGE("cudaHostAlloc(dev_ctx)"); return NULL; }
    memset(c->dev_ctx_h, 0, sizeof(tp_dev_ctx));
    if (cudaHostGetDevicePointer(&c->dev_ctx_d, c->dev_ctx_h, 0) != cudaSuccess){ LOGE("devptr ctx"); return NULL; }
    c->dev_ctx_h->epoch         = 0;
    c->dev_ctx_h->flags         = (unsigned long long*)c->dbuf;
    c->dev_ctx_h->send_ring     = (unsigned char*)c->dbuf + TP_RING_BASE;
    c->dev_ctx_h->recv_ring     = (unsigned char*)c->dbuf + TP_RING_BASE + (size_t)TP_RING_SLOTS * c->slot_stride;
    c->dev_ctx_h->gpu_ts        = NULL;
    c->dev_ctx_h->slot_stride   = c->slot_stride;
    c->dev_ctx_h->payload_bytes = c->payload_bytes;
    c->dev_ctx_h->rank          = rank;
    c->dev_ctx_h->fp32_payload  = 0;
    c->dev_ctx_h->len_local     = (unsigned long long*)((char*)c->dbuf + TP_LEN_LOCAL_OFF);
    c->dev_ctx_h->world         = world;
    c->dev_ctx_h->rounds        = c->rounds;
    c->dev_ctx_h->oneshot       = c->oneshot;   // P3-1: K2 dispatch + K1 gate variant read this
    c->dev_ctx_h->oneshot_recv  = (unsigned char*)c->dbuf + c->oneshot_recv_off;   // P3-1 block
    c->dev_ctx_h->nway_flags    = (unsigned long long*)((char*)c->dbuf + c->nway_flags_off);
    c->nway_flags_d             = (uint64_t)(uintptr_t)c->dev_ctx_h->nway_flags;
    c->dev_ctx_h->nway_recv     = (unsigned char*)c->dbuf + c->nway_recv_off;
    // P3-1/expert Change 2: the qp_mask ring K1 writes per epoch and the reuse gate reads at e-R.
    // world>2 only (world==2 leaves it NULL — the single-QP gate arm never reads it). Zeroed: a
    // zero mask means "no conflict set recorded yet" (ring not wrapped), and the gate only reads
    // [(e-R)] once e > R, by which point epoch e-R has written its mask. Device-visible, persistent.
    if (world > 2) {
        void* qm = NULL;
        if (cudaMalloc(&qm, (size_t)TP_QPMASK_SLOTS * sizeof(unsigned int)) != cudaSuccess) {
            LOGE("cudaMalloc(qp_mask)"); return NULL;
        }
        cudaMemset(qm, 0, (size_t)TP_QPMASK_SLOTS * sizeof(unsigned int));
        c->qp_mask_dev = qm;
        c->dev_ctx_h->qp_mask = (unsigned int*)qm;
    }

    c->peers = (PeerLink*)calloc((size_t)world, sizeof(PeerLink));

    // Liveness responder BEFORE any handshake (tcp_port + 1). One responder, all peers probe it.
    c->liveness_port = tcp_port + 1;
    {
        pthread_t th;
        if (pthread_create(&th, NULL, liveness_responder, (void*)(intptr_t)c->liveness_port) == 0)
            pthread_detach(th);
        else
            LOGE("WARNING: liveness responder thread failed to start (%s)", strerror(errno));
    }

    union ibv_gid mygid;
    if (ibv_query_gid(c->ctx, c->port_num, gid_idx, &mygid)){ LOGE("query_gid"); return NULL; }

    // One QP per peer. The lower rank listens, the higher connects (deterministic, matches tcp_exchange).
    for (int p = 0; p < world; p++) {
        if (p == rank) continue;
        PeerLink* pl = &c->peers[p];
        const char* pip = peer_ips[p];
        if (!pip || !pip[0]) { LOGE("rank %d: peer_ips[%d] empty", rank, p); return NULL; }

        struct ibv_qp_init_attr qia; memset(&qia,0,sizeof(qia));
        qia.send_cq=c->cq_send; qia.recv_cq=c->cq_startup; qia.qp_type=IBV_QPT_RC;
        qia.cap.max_send_wr=256; qia.cap.max_recv_wr=256; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1;
        qia.cap.max_inline_data=64;
        pl->qp = ibv_create_qp(c->pd, &qia);
        if (!pl->qp){ LOGE("create_qp(peer %d)", p); return NULL; }

        struct ibv_qp_attr a; memset(&a,0,sizeof(a));
        a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=c->port_num;
        a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(pl->qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ LOGE("INIT(peer %d)", p); return NULL; }

        NwayExch lo, re; memset(&lo,0,sizeof(lo));
        lo.rank = (uint32_t)rank;
        lo.e.qpn=pl->qp->qp_num; lo.e.psn=0x1000+(rank^p)*333; lo.e.addr=(uint64_t)c->hbuf; lo.e.rkey=c->mr->rkey; lo.e.gid=mygid;
        // Per-pair deterministic schedule (P4/R8): the LOWER rank listens, the HIGHER rank connects,
        // on a port derived from the ordered pair (min,max). Each pair therefore has exactly one
        // listener, no two pairs contend for the same port, and both sides verify the sender's rank.
        int pair_port = nway_pair_port(tcp_port, world, rank, p);
        if (nway_tcp_exchange(rank, p, pip, pair_port, &lo, &re,
                              pl->peer_ip, sizeof(pl->peer_ip))) {
            LOGE("nway_tcp_exchange(peer %d) failed", p);
            return NULL;
        }
        pl->remote_addr = re.e.addr; pl->remote_rkey = re.e.rkey;
        pl->valid = 1;

        memset(&a,0,sizeof(a));
        a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096; a.dest_qp_num=re.e.qpn; a.rq_psn=re.e.psn;
        a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
        a.ah_attr.is_global=1; a.ah_attr.port_num=c->port_num;
        a.ah_attr.grh.dgid=re.e.gid; a.ah_attr.grh.sgid_index=gid_idx; a.ah_attr.grh.hop_limit=1;
        if (ibv_modify_qp(pl->qp,&a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|
            IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)){ LOGE("RTR(peer %d)", p); return NULL; }
        memset(&a,0,sizeof(a));
        a.qp_state=IBV_QPS_RTS; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.sq_psn=lo.e.psn; a.max_rd_atomic=1;
        if (ibv_modify_qp(pl->qp,&a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|
            IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)){ LOGE("RTS(peer %d)", p); return NULL; }

        for (int i=0;i<64;i++){ struct ibv_recv_wr wr, *bad; memset(&wr,0,sizeof(wr)); wr.num_sge=0;
            if (ibv_post_recv(pl->qp,&wr,&bad)){ LOGE("post_recv init(peer %d)", p); return NULL; } }
    }
    return c;
}

// ---------------------------------------------------------------- accessors

// Set the DEFAULT payload for the hot path: the length a K1 call with nbytes==0 ships (the decode /
// bench barriers; prefill passes an explicit per-call length that overrides this). Called once, after
// the model config is known and BEFORE the proxy thread starts (the proxy and K1/K2 both read these,
// and I8 forbids mutating protocol state underneath a running system). `fp32` selects the
// FP32-preserving production reduction in K2.
int net_set_payload(NetCtx* c, int payload_bytes, int fp32) {
    // 8 B multiple, not 4: the default path writes the 8 B tail epoch at slot + align8(payload_bytes),
    // and the bench kernels read it back at exactly payload_bytes — the two coincide only when the
    // default is already aligned (a 4-mod-8 default would also misalign the GPU tail store).
    if (payload_bytes <= 0 || (payload_bytes & 7) ||
        (size_t)payload_bytes + TP_TAIL_BYTES > c->slot_stride) {
        LOGE("net_set_payload(%d): must be >0, 8 B multiple, and <= slot capacity %u",
             payload_bytes, c->slot_stride - TP_TAIL_BYTES);
        return -1;
    }
    c->payload_bytes = (unsigned)payload_bytes;
    c->dev_ctx_h->payload_bytes = (unsigned)payload_bytes;
    c->dev_ctx_h->fp32_payload  = fp32 ? 1u : 0u;
    return 0;
}

// v2 receive mode (EXPERT_GPU_ALLREDUCE §3.2): with `gpu` set, the GPU's K2'/maxloc_g kernels
// consume the NIC-written payload directly (two-stage hint+tail gate with the derived length);
// the proxy skips its RECV stage entirely and the watchdog keys on the GPU's TP_F_RX_DONE
// watermark. The wire format is byte-identical to v1, so mixed ranks interwork. Callable only
// before the proxy starts (same discipline as net_set_payload). Returns 0 on success.
int net_set_recv_mode(NetCtx* c, int gpu) {
    if (c->proxy_running) {
        LOGE("net_set_recv_mode(%d): proxy already running — set before net_spawn_proxy", gpu);
        return -1;
    }
    c->recv_gpu = gpu ? 1 : 0;
    return 0;
}

// The GPU's receive watermark (TP_F_RX_DONE) — the watchdog's v2 debt signal + diagnostics.
uint64_t net_rx_done(const NetCtx* c) { return *flagp(c, TP_F_RX_DONE); }

// S-B9-REL-TP2RACE (issue #10): the exchange generation counter (for the pre-verify bail diagnostic
// — "gen & 7 next to device_epoch & 7"), the probe's alignment bump (Rust computes the delta so it
// is unit-testable; the shim owns the counter so it stays race-free), and the probe hit count.
uint64_t net_xchg_gen(const NetCtx* c) { return c->gen; }
uint64_t net_gen_add(NetCtx* c, uint64_t delta) { c->gen += delta; return c->gen; }
uint64_t net_probe_hits(const NetCtx* c) { return c->probe_hits; }

const char* net_peer_ip(NetCtx* c) { return c->peer_ip; }   // TP-F: the aux rail dials the same peer
void* net_ctx_dptr(NetCtx* c)   { return c->dev_ctx_d; }          // K1/K2 kernel arg (the ONLY one)
void* net_flags_dptr(NetCtx* c) { return c->dbuf; }
// S-B9-REL-TP2RACE: at world 2 the exchange staging/readback is the DEDICATED ctrl slot pair, off
// the hot-path rings (issue #10); world>2 keeps the legacy slot-0 staging for this pairwise channel.
void* net_send_hptr(NetCtx* c)  { return xchg_stage_slot(c); }                             // ctrl send slot (w2)
// world>2: host view of the control receive slot for sender rank `src` (net_exchange_one's receiver).
void* net_ctrl_recv_hptr(NetCtx* c, int src) { return ctrl_last_slot(c, src); }
// world>2: host view of the dedicated control SEND staging slot (net_exchange_one's sender).
void* net_ctrl_send_hptr(NetCtx* c) { return ctrl_send_slot(c); }
int   net_world(NetCtx* c) { return c->world; }
int   net_rank(NetCtx* c) { return c->rank; }
// Startup-channel read view: the slot of the LAST COMPLETED exchange generation. Exchanges are
// gen-slotted on the receive side (see net_exchange), so a fast peer's NEXT generation can never
// clobber the payload a slow reader is still consuming (the sanity->broadcast 0x0-stamp race).
void* net_recv_hptr(NetCtx* c)  { return xchg_recv_slot_ptr(c, c->last_xchg_gen); }
void* net_send_dptr(NetCtx* c)  { return (char*)c->dbuf + TP_RING_BASE; }
void* net_recv_dptr(NetCtx* c)  { return (char*)c->dbuf + TP_RING_BASE
                                       + (size_t)TP_RING_SLOTS * c->slot_stride; }
unsigned long long net_device_epoch(NetCtx* c) { return c->dev_ctx_h->epoch; }
unsigned long long net_gate_waits(NetCtx* c)   { return c->dev_ctx_h->gate_waits; }
unsigned long long net_gpu_ready(NetCtx* c)    { return *flagp(c, TP_F_GPU_READY); }
unsigned long long net_tail_fires(NetCtx* c)   { return c->tail_fires; }
unsigned long long net_gpu_rx_skips(NetCtx* c) { return c->gpu_rx_skips; }   // TP-G
unsigned long long net_abort_status(NetCtx* c) { return *flagp(c, TP_F_ABORT); }
/* TP-4F2: tie the single-stage reduce ctx's abort to the primary link's, both ways, inside its proxy loop (a
 * graph replay never returns to the host between reduces, so a host-side per-launch check cannot be the only
 * surfacing path). NULL = no parent (every other ctx). Set once, before the proxy starts. */
void net_set_parent(NetCtx* c, NetCtx* parent) { c->parent = parent; }

// Pin the CALLING thread to `core` (GB10 is big.LITTLE; a launch or poll thread parked on a little
// A725 balloons latency and drains the GPU stream mid-token). Returns 0 if the affinity read back.
int net_pin_thread(int core) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) return -1;
    cpu_set_t rb; CPU_ZERO(&rb);
    if (pthread_getaffinity_np(pthread_self(), sizeof(rb), &rb)) return -1;
    return (CPU_ISSET(core, &rb) && CPU_COUNT(&rb) == 1) ? 0 : -1;   // readback, not hope
}

// ---------------------------------------------------------------- bench hooks

// Max hold is R, not R+1: K1(e) blocks on the gate BEFORE publishing gpu_ready=e, so at that moment
// posted == e-1 and unretired == (e-1) - retired. The gate binds when retired < e-R, i.e. when
// unretired reaches exactly R. A hold of R+1 waits for an outstanding count that K1 -- now blocked --
// can never produce, so the system deadlocks instead of testing the gate. (Measured: cq_hold=9 with
// R=8 hangs; cq_hold=8 binds the gate and recovers.)
int net_bench_cq_hold(NetCtx* c, unsigned hold, unsigned hold_us) {
    if (hold > TP_RING_SLOTS) {
        LOGE("cq_hold %u > R (%d) deadlocks by construction, it does not test the gate", hold, TP_RING_SLOTS);
        return -1;
    }
    c->cq_hold = hold;
    c->cq_hold_ns = (uint64_t)hold_us * 1000ull;
    c->hold_since = 0;
    return 0;
}

void net_bench_config(NetCtx* c, unsigned inject_delay_us_max, int ts_on) {
    c->inject_delay_us_max = inject_delay_us_max;
    if (ts_on && !c->cpu_ts)
        c->cpu_ts = (uint64_t*)calloc((size_t)TP_GTS_EPOCHS * TP_CTS_STRIDE, sizeof(uint64_t));
    if (ts_on && !c->gpu_ts_h) {
        if (cudaHostAlloc((void**)&c->gpu_ts_h, (size_t)TP_GTS_EPOCHS * TP_GTS_STRIDE * sizeof(uint64_t),
                          cudaHostAllocMapped|cudaHostAllocPortable) == cudaSuccess) {
            memset(c->gpu_ts_h, 0, (size_t)TP_GTS_EPOCHS * TP_GTS_STRIDE * sizeof(uint64_t));
            void* d = NULL;
            if (cudaHostGetDevicePointer(&d, c->gpu_ts_h, 0) == cudaSuccess)
                c->dev_ctx_h->gpu_ts = (unsigned long long*)d;
        }
    }
    c->ts_on = ts_on;
}
// The SAME clock the proxy stamps with — the bench needs it to bracket a GPU %globaltimer sample and
// estimate the GPU<->CPU offset. std::time::Instant is CLOCK_MONOTONIC, which drifts from _RAW by the
// accumulated NTP frequency adjustment (milliseconds on a long-running box), so it cannot substitute.
uint64_t net_now_ns(void) { return now_ns(); }

uint64_t* net_cpu_ts(NetCtx* c) { return c->cpu_ts; }
uint64_t* net_gpu_ts(NetCtx* c) { return c->gpu_ts_h; }
void net_counters(NetCtx* c, unsigned long long* posted, unsigned long long* retired,
                  unsigned long long* released, unsigned long long* tail_fires) {
    if (posted)     *posted     = c->posted_epochs;
    if (retired)    *retired    = c->retired_epochs;
    if (released)   *released   = c->released_epochs;
    if (tail_fires) *tail_fires = c->tail_fires;
}

static inline unsigned bench_rand(NetCtx* c) {   // xorshift; no libc rand in the hot loop
    unsigned x = c->rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; c->rng = x; return x;
}
static void bench_delay_us(NetCtx* c, unsigned max_us) {
    unsigned us = bench_rand(c) % (max_us + 1);
    uint64_t deadline = now_ns() + (uint64_t)us * 1000ull;
    while (now_ns() < deadline) cpu_relax();
}

// ---------------------------------------------------------------- hot path

// Post every currently-ready epoch as ONE linked WR list (round-3 R3a: drain what is ready, never wait
// to form a batch). Per epoch, three chained WRs on the same QP:
//   WR0: RDMA_WRITE 8 B INLINE length tag -> peer len_peer[e%TP_LEN_EPOCHS]     (unsignaled)
//   WR1: RDMA_WRITE send_ring[s] -> peer recv_ring[s], length align8(len)+tail  (unsignaled)
//   WR2: RDMA_WRITE 8 B INLINE epoch -> peer flags.peer_committed  (signaled iff e % S == 0)
// The length tag comes from len_local[e%TP_LEN_EPOCHS], which K1 wrote BEFORE release-storing
// gpu_ready = e — the proxy acquire-loads gpu_ready before posting, so the tag is visible with epoch
// bits == e by construction (a mismatch is corruption/desync, never a race: abort code 9).
// GB10_TP_TAIL_DRILL (test only, read once in net_init): every 4096th epoch INVERTS the triple —
// the commit is posted FIRST, then the tag, then the payload (signaling moves to the payload WR, which
// now posts last, so the CQE still retires the whole triple in order). The receiver's tail-epoch wait
// must engage and recover; that is the end-to-end proof of the load-bearing tail guard. (The len-tag
// wait may or may not engage — the tag follows the commit within the same post — but the payload,
// posted last, is always behind the commit, exactly the bypass scenario the guard exists for.)
// Returns 0 on success.
static int post_range(NetCtx* c, uint64_t first, uint64_t last) {
    unsigned cnt = (unsigned)(last - first + 1);
    if (cnt > TP_MAX_POST_BATCH) cnt = TP_MAX_POST_BATCH;

    struct ibv_send_wr wr[3 * TP_MAX_POST_BATCH];
    struct ibv_sge     sg[3 * TP_MAX_POST_BATCH];
    uint64_t           epv[TP_MAX_POST_BATCH];    // inline epoch source; copied into the WQE at post time
    uint64_t           tagv[TP_MAX_POST_BATCH];   // inline len-tag source (same)

    volatile uint64_t* len_local = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_LOCAL_OFF);

    for (unsigned i = 0; i < cnt; i++) {
        uint64_t e = first + i;
        epv[i] = e;
        // This epoch's wire length. K1 published it before gpu_ready; verify the generation.
        uint64_t tag = len_local[e & (TP_LEN_EPOCHS - 1)];
        uint64_t len = TP_LEN_TAG_BYTES(tag);
        if (TP_LEN_TAG_EPOCH(tag) != e || len == 0 || len > c->slot_stride - TP_TAIL_BYTES) {
            LOGE("LEN-RING DESYNC at epoch %llu: len_local[%llu]=%016llx (epoch bits %llu, len %llu, capacity %u)",
                 (unsigned long long)e, (unsigned long long)(e & (TP_LEN_EPOCHS - 1)),
                 (unsigned long long)tag, (unsigned long long)TP_LEN_TAG_EPOCH(tag),
                 (unsigned long long)len, c->slot_stride - TP_TAIL_BYTES);
            tp_set_abort(c, 9);
            return -1;
        }
        tagv[i] = tag;
        // Drill: does this epoch ship commit-before-payload?
        const int invert = c->tail_drill && (e % 4096 == 0);
        const int li = invert ? 1 : 0;        // len-tag WR slot within the triple
        const int pi = invert ? 2 : 1;        // payload WR slot within the triple
        const int ci = invert ? 0 : 2;        // commit WR slot within the triple

        sg[3*i+li] = (struct ibv_sge){ .addr = (uint64_t)&tagv[i], .length = TP_TAIL_BYTES, .lkey = 0 };
        sg[3*i+pi] = (struct ibv_sge){ .addr   = (uint64_t)send_slot(c, e),
                                       .length = (unsigned)(len + TP_TAIL_BYTES),
                                       .lkey   = c->mr->lkey };
        sg[3*i+ci] = (struct ibv_sge){ .addr = (uint64_t)&epv[i], .length = TP_TAIL_BYTES, .lkey = 0 };

        memset(&wr[3*i], 0, sizeof(wr[0])); memset(&wr[3*i+1], 0, sizeof(wr[0])); memset(&wr[3*i+2], 0, sizeof(wr[0]));
        // len-tag WR (epoch's wire length -> peer len_peer ring)
        wr[3*i+li].wr_id      = e;
        wr[3*i+li].opcode     = IBV_WR_RDMA_WRITE;
        wr[3*i+li].sg_list    = &sg[3*i+li]; wr[3*i+li].num_sge = 1;
        wr[3*i+li].send_flags = IBV_SEND_INLINE;      // unsignaled
        wr[3*i+li].wr.rdma.remote_addr = c->remote_addr + TP_LEN_PEER_OFF
                                       + (uint64_t)(e & (TP_LEN_EPOCHS - 1)) * 8;
        wr[3*i+li].wr.rdma.rkey        = c->remote_rkey;

        // payload WR
        wr[3*i+pi].wr_id      = e;
        wr[3*i+pi].opcode     = IBV_WR_RDMA_WRITE;
        wr[3*i+pi].sg_list    = &sg[3*i+pi]; wr[3*i+pi].num_sge = 1;
        // signaled iff e % S == 0; in drill mode the payload WR carries the signal (it posts last,
        // so its CQE retires the commit WR that precedes it — same accounting as the normal path).
        wr[3*i+pi].send_flags = invert ? ((e % TP_SIGNAL_EVERY == 0) ? IBV_SEND_SIGNALED : 0) : 0;
        wr[3*i+pi].wr.rdma.remote_addr = peer_recv_raddr(c, e);
        wr[3*i+pi].wr.rdma.rkey        = c->remote_rkey;

        // commit WR (epoch -> peer_committed)
        wr[3*i+ci].wr_id      = e;                                 // CQE carries the retired epoch
        wr[3*i+ci].opcode     = IBV_WR_RDMA_WRITE;
        wr[3*i+ci].sg_list    = &sg[3*i+ci]; wr[3*i+ci].num_sge = 1;
        // commit WR (epoch -> peer_committed). P3-1/expert: SIGNAL EVERY epoch's commit. The old
        // e % S cadence is only safe when per-QP traffic is dense+uniform; the one-shot push makes
        // a per-QP subsequence sparse/pattern-locked (the rank^3 QP carries one-shot ONLY), so its
        // signaled-CQE supply — {e≡0 mod S} ∩ {this QP's epochs} — can be structurally empty and
        // tx_retired[p] freezes while the sends physically retire invisibly. S=1 everywhere makes
        // tx_retired[p] exact (1 CQE per posted epoch per QP). Cost is ~2 orders of magnitude under
        // measurability at our barrier rates; the WR count is unchanged, only the flag flips.
        wr[3*i+ci].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        wr[3*i+ci].wr.rdma.remote_addr = c->remote_addr + TP_F_PEER_COMMITTED;
        wr[3*i+ci].wr.rdma.rkey        = c->remote_rkey;

        // The chain is ALWAYS physical (3i -> 3i+1 -> 3i+2 -> 3i+3): the triple order on the wire is
        // decided by which WR sits in which slot, never by the links. (The first drill version linked
        // by role, which under inversion skipped every payload WR entirely — the watchdog+abort+
        // divergence path caught it loudly, exactly as designed, but the drill itself was invalid.)
        wr[3*i].next   = &wr[3*i+1];
        wr[3*i+1].next = &wr[3*i+2];
        wr[3*i+2].next = (i + 1 < cnt) ? &wr[3*(i+1)] : NULL;
    }

    struct ibv_send_wr* bad = NULL;
    int rc = ibv_post_send(c->qp, &wr[0], &bad);
    if (rc) { LOGE("post_send rc=%d (%s)", rc, strerror(rc)); tp_set_abort(c, 3); return -1; }
    c->posted_epochs += cnt;
    if (c->ts_on) { uint64_t t = now_ns(); for (unsigned i = 0; i < cnt; i++) stamp(c, first+i, TP_CTS_POSTED, t); }
    return (int)cnt;
}

// Non-blocking CQ drain. RC completions are in order, so a CQE for epoch m retires every WR <= m
// including the unsignaled payload writes -> publishing tx_retired = m opens the reuse gate (I3).
// Error CQEs are emitted for UNSIGNALED WRs too, so this check covers them.
static int drain_cq(NetCtx* c) {
    struct ibv_wc wc[8];
    int n = ibv_poll_cq(c->cq_send, 8, wc);
    if (n <= 0) return 0;
    uint64_t hi = 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            LOGE("hot-path CQE status %d (%s) wr_id %llu", wc[i].status,
                 ibv_wc_status_str(wc[i].status), (unsigned long long)wc[i].wr_id);
            tp_set_abort(c, 4);
            return -1;
        }
        if (wc[i].wr_id == TP_XCHG_WR_ID) {   // startup-channel send, NOT an epoch: hand to net_exchange
            __atomic_store_n(&c->xchg_send_done, 1, __ATOMIC_RELEASE);
            continue;
        }
        if (wc[i].wr_id > hi) hi = wc[i].wr_id;
    }
    if (hi) {
        __atomic_store_n(flagp(c, TP_F_TX_RETIRED), hi, __ATOMIC_RELEASE);
        c->retired_epochs = hi;
        if (c->ts_on) stamp(c, hi, TP_CTS_CQE, now_ns());
    }
    return n;
}

// N-way single-epoch post (world > 2): the SAME 3-WR doorbell triple, posted to partner(epoch)'s QP.
// Returns 1 on success, -1 on abort. One epoch at a time (the round schedule sends consecutive epochs
// to different partners).
static int post_range_nway(NetCtx* c, uint64_t e) {
    volatile uint64_t* len_local = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_LOCAL_OFF);
    uint64_t tag = len_local[e & (TP_LEN_EPOCHS - 1)];
    uint64_t len = TP_LEN_TAG_BYTES(tag);
    if (TP_LEN_TAG_EPOCH(tag) != e || len == 0 || len > c->slot_stride - TP_TAIL_BYTES) {
        LOGE("LEN-RING DESYNC (nway) at epoch %llu: len_local[%llu]=%016llx (epoch bits %llu, len %llu, capacity %u)",
             (unsigned long long)e, (unsigned long long)(e & (TP_LEN_EPOCHS - 1)),
             (unsigned long long)tag, (unsigned long long)TP_LEN_TAG_EPOCH(tag),
             (unsigned long long)len, c->slot_stride - TP_TAIL_BYTES);
        tp_set_abort(c, 9);
        return -1;
    }

    int p = partner_rank_of(c, e);
    PeerLink* pl = &c->peers[p];
    uint64_t epv = e, tagv = tag;

    struct ibv_sge sg[3];
    struct ibv_send_wr wr[3];
    const int invert = c->tail_drill && (e % 4096 == 0);
    const int li = invert ? 1 : 0;
    const int pi = invert ? 2 : 1;
    const int ci = invert ? 0 : 2;

    sg[li] = (struct ibv_sge){ .addr = (uint64_t)&tagv, .length = TP_TAIL_BYTES, .lkey = 0 };
    sg[pi] = (struct ibv_sge){ .addr   = (uint64_t)send_slot(c, e),
                               .length = (unsigned)(len + TP_TAIL_BYTES),
                               .lkey   = c->mr->lkey };
    sg[ci] = (struct ibv_sge){ .addr = (uint64_t)&epv, .length = TP_TAIL_BYTES, .lkey = 0 };

    memset(&wr[0], 0, sizeof(wr[0])); memset(&wr[1], 0, sizeof(wr[0])); memset(&wr[2], 0, sizeof(wr[0]));

    wr[li].wr_id      = e;
    wr[li].opcode     = IBV_WR_RDMA_WRITE;
    wr[li].sg_list    = &sg[li]; wr[li].num_sge = 1;
    wr[li].send_flags = IBV_SEND_INLINE;
    wr[li].wr.rdma.remote_addr = pl->remote_addr + TP_LEN_PEER_OFF
                               + (uint64_t)(e & (TP_LEN_EPOCHS - 1)) * 8;
    wr[li].wr.rdma.rkey        = pl->remote_rkey;

    wr[pi].wr_id      = e;
    wr[pi].opcode     = IBV_WR_RDMA_WRITE;
    wr[pi].sg_list    = &sg[pi]; wr[pi].num_sge = 1;
    wr[pi].send_flags = invert ? ((e % TP_SIGNAL_EVERY == 0) ? IBV_SEND_SIGNALED : 0) : 0;
    // Round-keyed recv slot (world>2): epoch e lands in round(e)'s own recv ring on the partner,
    // so a round's payload can never alias another round's slot (the cross-round staleness fix).
    wr[pi].wr.rdma.remote_addr = nway_peer_recv_raddr(c, p, e);
    wr[pi].wr.rdma.rkey        = pl->remote_rkey;

    wr[ci].wr_id      = e;
    wr[ci].opcode     = IBV_WR_RDMA_WRITE;
    wr[ci].sg_list    = &sg[ci]; wr[ci].num_sge = 1;
    // P3-1/expert: SIGNAL EVERY epoch's commit (see post_range). The `first_on_qp` self-clock was a
    // patch over the e % S cadence's blind spot at world>2; with S=1 it is redundant — every posted
    // epoch produces a CQE on its QP, so tx_retired[p] is exact. Keep last_posted_peer updated for
    // any other reader.
    int first_on_qp = (c->posted_epochs == 0) || (c->last_posted_peer != p) ? 1 : 0;
    if (first_on_qp) c->last_posted_peer = p;
    (void)first_on_qp;
    wr[ci].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    wr[ci].wr.rdma.remote_addr = peer_committed_raddr(c, p);
    wr[ci].wr.rdma.rkey        = pl->remote_rkey;

    wr[0].next = &wr[1];
    wr[1].next = &wr[2];
    wr[2].next = NULL;

    struct ibv_send_wr* bad = NULL;
    int rc = ibv_post_send(pl->qp, &wr[0], &bad);
    if (rc) { LOGE("post_send(nway) rc=%d (%s)", rc, strerror(rc)); tp_set_abort(c, 3); return -1; }
    c->posted_epochs++;
    if (c->ts_on) { uint64_t t = now_ns(); stamp(c, e, TP_CTS_POSTED, t); }
    // R9 DIAGNOSTIC: checksum the EXACT bytes handed to the NIC for this epoch (post returns after
    // the SGE data is captured; the GPU has released this slot, so it is stable here).
    tp_diag_log_send(c, e, p, send_slot(c, e), len);
    return 1;
}

// P3-1: Rust-side one-shot selector (reads the ctx field the kernels/proxy use — one source of truth).
int net_oneshot_on(NetCtx* c) { return c ? c->oneshot : 0; }

// P3-1: the per-epoch algorithm rule both sides derive independently (proxy: from len_local; the
// Rust dispatch: from n). One-shot iff the wire length fits the DEFAULT decode payload (h*2) —
// exactly the decode/draft lane. Verify (h*batch) and prefill chunks are wider -> the tree, whose
// kernels address round-keyed slots. SPMD-safe: both sides compute the same predicate from the
// same per-epoch length.
int net_use_oneshot(NetCtx* c, uint64_t wire_len) {
    return c && c->oneshot && wire_len && wire_len <= (uint64_t)c->payload_bytes;
}

// P3-1 one-shot (world==4, c->oneshot): post the SAME 3-WR chain (len tag / payload+tail / commit)
// to ALL world-1 peers' QPs for epoch e. The payload lands in SENDER-indexed recv rings on every
// peer (ring = c->rank, slot = e % R), so each receiver's K2 (tp_wait_add_4way) finds all 3 peers'
// partials at SPMD-derived addresses with zero per-epoch coordination. The commit hints + tails
// keep the exact placement-proof discipline of the tree path per peer. Signaled cadence unchanged;
// `first_on_qp` applies per peer QP (the nway self-clocking rule, all peers now).
static uint64_t oneshot_peer_recv_raddr(NetCtx* c, int peer_rank, uint64_t e) {
    // SENDER(c->rank)-indexed ring in the peer's DEDICATED one-shot block (after the tree's
    // round rings): oneshot_recv_off + rank*R*stride + (e%R)*stride
    return c->peers[peer_rank].remote_addr + c->oneshot_recv_off
         + ((uint64_t)c->rank * (uint64_t)TP_RING_SLOTS + (e & (TP_RING_SLOTS - 1)))
           * c->slot_stride;
}
static int post_range_oneshot(NetCtx* c, uint64_t e) {
    volatile uint64_t* len_local = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_LOCAL_OFF);
    uint64_t tag = len_local[e & (TP_LEN_EPOCHS - 1)];
    uint64_t len = TP_LEN_TAG_BYTES(tag);
    if (TP_LEN_TAG_EPOCH(tag) != e || len == 0 || len > c->slot_stride - TP_TAIL_BYTES) {
        LOGE("LEN-RING DESYNC (oneshot) at epoch %llu", (unsigned long long)e);
        tp_set_abort(c, 9);
        return -1;
    }
    uint64_t epv = e, tagv = tag;
    const int invert = c->tail_drill && (e % 4096 == 0);

    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) continue;
        PeerLink* pl = &c->peers[p];

        struct ibv_sge sg[3];
        struct ibv_send_wr wr[3];
        const int li = invert ? 1 : 0;
        const int pi = invert ? 2 : 1;
        const int ci = invert ? 0 : 2;

        sg[li] = (struct ibv_sge){ .addr = (uint64_t)&tagv, .length = TP_TAIL_BYTES, .lkey = 0 };
        sg[pi] = (struct ibv_sge){ .addr   = (uint64_t)send_slot(c, e),
                                   .length = (unsigned)(len + TP_TAIL_BYTES),
                                   .lkey   = c->mr->lkey };
        sg[ci] = (struct ibv_sge){ .addr = (uint64_t)&epv, .length = TP_TAIL_BYTES, .lkey = 0 };

        memset(&wr[0], 0, sizeof(wr[0])); memset(&wr[1], 0, sizeof(wr[0])); memset(&wr[2], 0, sizeof(wr[0]));

        wr[li].wr_id = e; wr[li].opcode = IBV_WR_RDMA_WRITE;
        wr[li].sg_list = &sg[li]; wr[li].num_sge = 1;
        wr[li].send_flags = IBV_SEND_INLINE;
        wr[li].wr.rdma.remote_addr = pl->remote_addr + TP_LEN_PEER_OFF
                                   + (uint64_t)(e & (TP_LEN_EPOCHS - 1)) * 8;
        wr[li].wr.rdma.rkey = pl->remote_rkey;

        wr[pi].wr_id = e; wr[pi].opcode = IBV_WR_RDMA_WRITE;
        wr[pi].sg_list = &sg[pi]; wr[pi].num_sge = 1;
        wr[pi].send_flags = invert ? ((e % TP_SIGNAL_EVERY == 0) ? IBV_SEND_SIGNALED : 0) : 0;
        wr[pi].wr.rdma.remote_addr = oneshot_peer_recv_raddr(c, p, e);   // SENDER-indexed ring
        wr[pi].wr.rdma.rkey = pl->remote_rkey;

        wr[ci].wr_id = TP_ONESHOT_TAG(e, p); wr[ci].opcode = IBV_WR_RDMA_WRITE;
        wr[ci].sg_list = &sg[ci]; wr[ci].num_sge = 1;
        int first_on_qp = (c->posted_epochs == 0) || (c->last_posted_peer != p) ? 1 : 0;
        if (first_on_qp) c->last_posted_peer = p;
        (void)first_on_qp;
        // P3-1/expert: SIGNAL EVERY one-shot commit on every QP it touches (the whole point of the
        // fix — the rank^3 QP carries ONLY one-shot traffic, so the e % S cadence can starve its
        // tx_retired indefinitely). 1 CQE per (epoch, peer) => exact per-QP retirement.
        wr[ci].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        wr[ci].wr.rdma.remote_addr = peer_committed_raddr(c, p);
        wr[ci].wr.rdma.rkey = pl->remote_rkey;

        wr[0].next = &wr[1];
        wr[1].next = &wr[2];
        wr[2].next = NULL;

        struct ibv_send_wr* bad = NULL;
        int rc = ibv_post_send(pl->qp, &wr[0], &bad);
        if (rc) { LOGE("post_send(oneshot,p%d) rc=%d (%s)", p, rc, strerror(rc)); tp_set_abort(c, 3); return -1; }
        tp_diag_log_send(c, e, p, send_slot(c, e), len);
    }
    c->posted_epochs++;
    if (c->ts_on) { uint64_t t = now_ns(); stamp(c, e, TP_CTS_POSTED, t); }
    return 1;
}

// N-way non-blocking CQ drain (world > 2). All per-peer QPs share cq_send; the CQE's wr_id IS the
// epoch, and the epoch maps to a peer via the round schedule. Publish each QP's retirement into its own
// `tx_retired[p]` line (per-QP in-order retirement -> the I3 reuse gate keys the PREVIOUS slot owner).
static int drain_cq_nway(NetCtx* c) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->cq_send, 32, wc);
    if (n <= 0) return 0;
    uint64_t hi[TP_NWAY_MAX_WORLD];
    memset(hi, 0, sizeof(hi));
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            LOGE("hot-path CQE status %d (%s) wr_id %llu", wc[i].status,
                 ibv_wc_status_str(wc[i].status), (unsigned long long)wc[i].wr_id);
            tp_set_abort(c, 4);
            return -1;
        }
        if (wc[i].wr_id == TP_XCHG_WR_ID) {
            __atomic_fetch_add(&c->xchg_send_seq, 1, __ATOMIC_RELEASE);   // B8 §1.5-4: monotone gen
            continue;
        }
        // P3-1: a one-shot CQE is tagged with the PEER RANK it completed on (per-QP tx_retired
        // credit). A one-shot epoch is posted to all world-1 QPs, so without this tag every
        // completion would be mis-attributed to the TREE partner for that epoch — at world=4 the
        // rank^3 peer's tx_retired would NEVER advance and the K1 all-peers reuse gate wedges at
        // the first ring wrap (the mixed-width wedge root cause). Decode the tag: attribute to the
        // tagged peer, credit the bare epoch. Tree epochs (untagged) keep the round-schedule
        // attribution, byte-identical.
        int p; uint64_t ee;
        if (TP_ONESHOT_TAGGED(wc[i].wr_id)) { p = TP_ONESHOT_PEER(wc[i].wr_id); ee = TP_ONESHOT_EPOCH(wc[i].wr_id); }
        else                                { p = partner_rank_of(c, wc[i].wr_id); ee = wc[i].wr_id; }
        if (p >= 0 && p < c->world && ee > hi[p]) hi[p] = ee;
    }
    uint64_t global_hi = 0;
    for (int p = 0; p < c->world; p++) {
        if (hi[p]) {
            __atomic_store_n(tx_retired_flagp(c, p), hi[p], __ATOMIC_RELEASE);
            if (hi[p] > global_hi) global_hi = hi[p];
        }
    }
    if (global_hi) {
        c->retired_epochs = global_hi;
        if (c->ts_on) stamp(c, global_hi, TP_CTS_CQE, now_ns());
    }
    return n;
}

// Persistent CPU proxy loop, on its own pinned thread. The main decode thread never syncs: it queues
// K1/K2 per reduction on the GPU stream and races ahead. This loop is the only mutator of tx_retired
// and cpu_done (I8).
void net_proxy_loop(NetCtx* c, int core) {
    if (c->world == 2) { net_proxy_loop_world2(c, core); return; }   // R1: single-QP fast path
    net_proxy_loop_nway(c, core);
}

// The pre-P3 world==2 proxy loop, byte-for-byte (single QP, single peer_committed/cpu_done).
static void net_proxy_loop_world2(NetCtx* c, int core) {
    // Own cq_send from here on: net_exchange polls only cq_startup + the xchg_send_done handoff.
    __atomic_store_n(&c->proxy_running, 1, __ATOMIC_RELEASE);
    // Pinning is the measurement, not a preference: an unpinned proxy costs ~40% end-to-end
    // (9.0 vs 15.1 tok/s, measured) and presents exactly like a protocol stall. The bench refuses
    // to report numbers unpinned; production must refuse to RUN unpinned. Pass core < 0 to opt out
    // explicitly. Abort code 8.
    if (core >= 0 && net_pin_thread(core)) {
        LOGE("FATAL: proxy failed to pin to core %d — aborting rather than running unpinned", core);
        tp_set_abort(c, 8);
        return;
    }

    uint64_t next_to_post = 1;   // proxy OWNS the posted epoch (I1)
    uint64_t next_release = 1;   // next peer epoch to hand to the local GPU

    // Liveness watchdog (the design's "abort/timeout from day one" rule). The QP retry machinery
    // covers a dead NIC (error CQE -> abort), but NOT a peer whose proxy+NIC stay alive while its
    // GPU or main thread hangs: then peer_committed simply never advances and both our K2 spin and
    // net_agree would wait FOREVER, silently, mid-stream. The SPMD program is symmetric, so the
    // peer owes us epoch e whenever we have POSTED e: `next_to_post > next_release` persisting for
    // TP_WDOG_NS is a dead peer by construction. Whichever side hangs, the OTHER side's watchdog
    // fires. Abort code 6.
    //   Threshold: the worst LEGITIMATE inter-post gap is one prefill layer's compute (~0.25 s at
    // 8K-chunk prefill on the 122B); 2 s is ~10x that, and a real hang is forever, so any large
    // threshold works. Idle periods have nothing outstanding and never engage it.
    uint64_t wdog_since = 0;       // when the current outstanding-debt window opened (0 = none)
    unsigned  wdog_ticks = 0;      // spin counter so the clock is read rarely

    while (!c->aborted && !__atomic_load_n(flagp(c, TP_F_ABORT), __ATOMIC_ACQUIRE)) {
        int did_work = 0;

        // -- WATCHDOG: is there posted work the peer has not matched? (cheap: no clock read) --
        // Armed ONLY after the first completed rendezvous: before that, "posted but unmatched" is a
        // LEGITIMATE state — one rank can finish model load and reach the first barriers seconds
        // before the other (load-time skew), and the reuse gate makes it wait exactly here. Firing
        // in that window aborts a healthy run. After the first rendezvous, a 10 s unmatched debt
        // cannot be load skew — it is a dead peer. (Bring-up hangs before the first rendezvous are
        // loud and killable by inspection; the watchdog covers the steady state.)
        // v2 (recv_gpu): the proxy never releases — the GPU's rx_done watermark is the matched
        // counter; the arm keys off rx_done > 0 instead of released_epochs (same load-skew
        // exclusion). `matched = max(next_release, rx_done)` keeps the watchdog mode-agnostic and
        // correct for mixed v1/v2 pairs (EXPERT_GPU_ALLREDUCE §7).
        uint64_t rx_done = c->recv_gpu ? *flagp(c, TP_F_RX_DONE) : 0;   // plain load (I6)
        // R10: rx_done is INCLUSIVE (highest epoch consumed); next_to_post is EXCLUSIVE. At a
        // quiescent point (posted == consumed == E, next_to_post == E+1) the raw max() leaves a
        // permanent one-epoch "debt" — and 10 s of idle (between requests, between bring-up and
        // the first HTTP hit, or any host-side pause > TP_WDOG_NS on a peer) then aborts a
        // perfectly healthy link. This killed every world=4 bring-up at the first quiet gap.
        // Compensate by making the v2 counter exclusive (+1); a REAL deadlock still shows debt >= 1
        // (our posted E vs rx_done stuck below E-1). v1's next_release is already exclusive.
        uint64_t matched = c->recv_gpu
            ? (next_release > rx_done + 1 ? next_release : rx_done + 1)
            : (next_release > rx_done ? next_release : rx_done);
        if ((c->released_epochs > 0 || rx_done > 0) && next_to_post > matched) {
            if (wdog_since == 0) { wdog_since = now_ns(); wdog_ticks = 0; }
            else if (((wdog_ticks++) & 0x3FF) == 0 && now_ns() - wdog_since > TP_WDOG_NS) {
                LOGE("WATCHDOG: peer has not matched posted epoch %llu for %llums (awaiting %llu)"
                     " — declaring peer dead; aborting",
                     (unsigned long long)(next_to_post - 1),
                     (unsigned long long)(TP_WDOG_NS / 1000000ull),
                     (unsigned long long)matched);
                tp_set_abort(c, 6);
                break;
            }
        } else {
            wdog_since = 0;
        }

        // -- SEND: the watermark says payload for every epoch <= w is in the send ring --
        uint64_t w = __atomic_load_n(flagp(c, TP_F_GPU_READY), __ATOMIC_ACQUIRE);
        if (w >= next_to_post) {
            if (c->ts_on) { uint64_t t = now_ns(); for (uint64_t e = next_to_post; e <= w; e++) stamp(c, e, TP_CTS_READY, t); }
            if (c->inject_delay_us_max) bench_delay_us(c, c->inject_delay_us_max);
            int posted = post_range(c, next_to_post, w);
            if (posted < 0) break;
            next_to_post += (uint64_t)posted;
            did_work = 1;
        }

        // -- CQ: non-blocking drain -> tx_retired --
        // Bench hook (round-3 "delayed CQ polling"): withhold retirement credit until `cq_hold` epochs
        // are outstanding. This is the ONLY way to make the I3 reuse gate actually bind — the
        // bidirectional rendezvous bounds inter-node skew to ~1 barrier, so a consumer stall slows
        // everything down symmetrically and never reaches ring depth. With the hold, tx_retired lags
        // past e-R, K1 blocks on the gate, the next drain opens it: bounded backpressure, not collapse.
        // cq_hold MUST be <= R+1 (net_bench_config clamps): above that the gate binds at R+1 unretired
        // and no further epoch can ever be posted to reach the hold threshold — a deliberate deadlock.
        int hold = 0;
        if (c->cq_hold) {
            uint64_t unret = c->posted_epochs - c->retired_epochs;
            if (unret < (uint64_t)c->cq_hold) {
                hold = 1; c->hold_since = 0;          // not yet at the threshold: keep withholding
            } else if (c->cq_hold_ns) {
                // Threshold reached and K1 is now blocked on the gate. Keep withholding for a fixed
                // interval so the gate DEMONSTRABLY binds every cycle — releasing the instant the
                // threshold is hit races the gate and (measured) binds it only ~once per run.
                uint64_t t = now_ns();
                if (!c->hold_since) c->hold_since = t;
                if (t - c->hold_since < c->cq_hold_ns) hold = 1; else c->hold_since = 0;
            }
        }
        if (!hold) {
            int drained = drain_cq(c);
            if (drained < 0) break;
            if (drained > 0) did_work = 1;
        }

        // -- AGREE: ship the lockstep token if the main thread published a new one --
        {
            uint64_t ao = *flagp(c, TP_F_AGREE_OUT);
            if (ao != c->agree_last) {
                struct ibv_sge sg = { .addr = (uint64_t)&ao, .length = 8, .lkey = 0 };
                struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
                wr.wr_id = 0; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sg; wr.num_sge = 1;
                wr.send_flags = IBV_SEND_INLINE;      /* unsignaled: the peer's spin is the ack */
                wr.wr.rdma.remote_addr = c->remote_addr + TP_F_AGREE_IN;
                wr.wr.rdma.rkey = c->remote_rkey;
                if (ibv_post_send(c->qp, &wr, &bad)) { tp_set_abort(c, 5); break; }
                c->agree_last = ao;
                did_work = 1;
            }
        }

        // -- RECV: CPU bounce for visibility (I5) — SKIPPED ENTIRELY in v2 receive mode, where
        // the GPU's K2'/maxloc_g kernels validate the payload tail directly (the proxy still posts
        // the wire; the receive-side validation moves GPU-side, EXPERT_GPU_ALLREDUCE §3.2/§8).
        if (c->recv_gpu) {
            // v2: nothing to do — the GPU consumed the payload; the watchdog keys on rx_done.
            // (next_release stays 1; the max() in the watchdog uses rx_done.)
        } else {
        uint64_t pc = *flagp(c, TP_F_PEER_COMMITTED);          // plain volatile load, no RMW (I6)
        if (pc >= next_release) {
            if (c->ts_on) stamp(c, pc, TP_CTS_PEERSEEN, now_ns());
            // Tail-epoch guard (R2d), now LOAD-BEARING — twice over. The old assumption — "same QP =>
            // the payload is placed before the commit" — is FALSE on this platform: at epoch ~4.36M
            // under a prefill flood the receiver observed peer_committed=e while the slot still
            // held generation e-R's tail (Grace C2C + relaxed PCIe ordering let the 8 B inline
            // commit bypass the large payload write; RC guarantees DELIVERY, not cross-address
            // placement order). So peer_committed is only a HINT of how far to check. Two
            // generation-tagged values are the actual commits, checked in order with bounded waits:
            //   1. len_peer[e%TP_LEN_EPOCHS]: this epoch's wire length (the receiver cannot even
            //      locate the tail without it). Posted ahead of the payload; the same bypass logic
            //      applies to it, so it gets the same bounded-wait discipline.
            //   2. the slot's trailing u64 at recv_slot(e)+len: the payload's own commit.
            // The payload/tag were posted before the commit on a reliable QP, so they land
            // momentarily; and the reuse-gate chain (the peer cannot post WR1(e+R) until WR1(e) has
            // retired, which requires placement here) means the slot cannot be overwritten by a later
            // generation while we wait. The same holds for the len ring at TP_LEN_EPOCHS=4096 depth:
            // the peer's K1(e+4096) runs only after its K2 chain consumed our cpu_done for e+4095,
            // which this loop released only after reading generation e's tag. Abort only if a value
            // genuinely never lands.
            int ok = 1;
            uint64_t e = next_release;
            volatile uint64_t* len_peer = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_PEER_OFF);
            // TP-G: the EXL3 decode K2 (xq_tp_wait_add_dec, GPU-side receive) validates an epoch's
            // payload tail itself and RELEASE-publishes RX_DONE = e once it passed; the GPU then runs
            // ahead of this loop, so by the time we look the slot may already hold generation e+R (the
            // peer may reuse it once our K2(e) completed). An epoch the GPU proved needs no CPU proof:
            // skip it. (RX_DONE is written in world-2 v1 mode ONLY by that kernel; every earlier epoch
            // was consumed by a cpu_done-gated K2 first — stream order — so no skipped epoch is
            // unvalidated. NVFP4 v1 never writes it: inert there.)
            volatile uint64_t* rxd = flagp(c, TP_F_RX_DONE);
            for (; e <= pc; e++) {
                if (*rxd >= e) { c->gpu_rx_skips++; continue; }
                volatile uint64_t* lp = len_peer + (e & (TP_LEN_EPOCHS - 1));
                uint64_t tag = *lp;
                int gpu_done = 0;
                if (TP_LEN_TAG_EPOCH(tag) != e) {
                    uint64_t t0 = now_ns();
                    for (;;) {
                        if (c->aborted || *flagp(c, TP_F_ABORT)) { ok = 0; break; }
                        if (*rxd >= e) { gpu_done = 1; break; }
                        tag = *lp;
                        if (TP_LEN_TAG_EPOCH(tag) == e) break;
                        uint64_t dt = now_ns() - t0;
                        if (dt > TP_TAIL_WAIT_NS) {
                            LOGE("LEN-EPOCH GUARD FIRED: len_peer[%llu]=%016llx expected epoch %llu (peer_committed=%llu)"
                                 " — length tag never landed (waited %llums)",
                                 (unsigned long long)(e & (TP_LEN_EPOCHS - 1)), (unsigned long long)tag,
                                 (unsigned long long)e, (unsigned long long)pc,
                                 (unsigned long long)(TP_TAIL_WAIT_NS / 1000000ull));
                            c->tail_fires++;
                            tp_set_abort(c, 2);
                            ok = 0;
                            break;
                        }
                        tp_host_backoff(dt);
                    }
                    if (!ok) break;
                    if (gpu_done) { c->gpu_rx_skips++; continue; }
                    c->len_waits++;
                    if (c->len_waits <= 8 || (c->len_waits & (c->len_waits - 1)) == 0)
                        LOGE("len-epoch wait engaged for epoch %llu (%lluth) — tag landed after the commit (recovered)",
                             (unsigned long long)e, (unsigned long long)c->len_waits);
                }
                uint64_t len = TP_LEN_TAG_BYTES(tag);
                if (len == 0 || len > c->slot_stride - TP_TAIL_BYTES) {
                    LOGE("LEN-EPOCH GUARD FIRED: malformed len %llu for epoch %llu (capacity %u) — aborting",
                         (unsigned long long)len, (unsigned long long)e, c->slot_stride - TP_TAIL_BYTES);
                    c->tail_fires++;
                    tp_set_abort(c, 2);
                    ok = 0;
                    break;
                }
                volatile uint64_t* tailp = (volatile uint64_t*)(recv_slot(c, e) + len);
                if (*tailp == e) continue;
                uint64_t t0 = now_ns();
                for (;;) {
                    if (c->aborted || *flagp(c, TP_F_ABORT)) { ok = 0; break; }
                    if (*tailp == e) break;
                    if (*rxd >= e) { gpu_done = 1; break; }
                    uint64_t dt = now_ns() - t0;
                    if (dt > TP_TAIL_WAIT_NS) {
                        uint64_t tail = *tailp;
                        LOGE("TAIL-EPOCH GUARD FIRED: slot %llu tail=%llu expected %llu (peer_committed=%llu)"
                             " — payload never landed (waited %llums)",
                             (unsigned long long)(e & (TP_RING_SLOTS-1)), (unsigned long long)tail,
                             (unsigned long long)e, (unsigned long long)pc,
                             (unsigned long long)(TP_TAIL_WAIT_NS / 1000000ull));
                        c->tail_fires++;
                        tp_set_abort(c, 2);
                        ok = 0;
                        break;
                    }
                    tp_host_backoff(dt);
                }
                if (!ok) break;
                if (gpu_done) { c->gpu_rx_skips++; continue; }
                c->tail_waits++;
                if (c->tail_waits <= 8 || (c->tail_waits & (c->tail_waits - 1)) == 0)
                    LOGE("tail-epoch wait engaged for epoch %llu (%lluth) — payload landed after the commit (recovered)",
                         (unsigned long long)e, (unsigned long long)c->tail_waits);
            }
            if (!ok) break;
            // S-B9-REL-TP2RACE probe (--tp-race-probe, issue #10): rank 1 holds THIS release by
            // probe_hold_ms. Nothing legitimate can write recv_slot(pc) while the epoch is withheld
            // (the lockstep bounds both ranks' K1s behind this very release), so a hash change
            // across the hold is a host exchange frame clobbering an unconsumed epoch payload —
            // the race, caught at the transport layer regardless of what it does to the drafts.
            if (c->probe_hold_ms && c->rank == 1) {
                char* pslot = recv_slot(c, pc);
                uint64_t h0 = tp_fnv64(pslot, 64);
                uint64_t pt0 = now_ns();
                while (now_ns() - pt0 < (uint64_t)c->probe_hold_ms * 1000000ull) {
                    if (c->aborted || *flagp(c, TP_F_ABORT)) break;
                    struct timespec pd = { .tv_sec = 0, .tv_nsec = 200000 };
                    nanosleep(&pd, NULL);
                }
                uint64_t h1 = tp_fnv64(pslot, 64);
                if (h0 != h1) {
                    c->probe_hits++;
                    LOGE("[tp-race-probe] CLOBBER at epoch %llu slot %llu: the withheld payload changed "
                         "(hash %016llx -> %016llx, first word %016llx, tail-elem guard had passed) — a host "
                         "exchange frame landed on an unconsumed epoch payload (issue #10 race CONFIRMED)",
                         (unsigned long long)pc, (unsigned long long)(pc & (TP_RING_SLOTS - 1)),
                         (unsigned long long)h0, (unsigned long long)h1,
                         (unsigned long long)*(volatile uint64_t*)pslot);
                }
            }
            // Full fence, then RELEASE-store cpu_done: when the GPU acquire-loads it, this core's
            // coherent view of the NIC's payload writes is ordered behind the flag read (I5).
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            __atomic_store_n(flagp(c, TP_F_CPU_DONE), pc, __ATOMIC_RELEASE);
            c->released_epochs = pc;
            if (c->ts_on) stamp(c, pc, TP_CTS_RELEASED, now_ns());
            next_release = pc + 1;
            did_work = 1;
        }
        }

        if (!did_work) cpu_relax();   // plain load + yield, never an atomic RMW (I6)
    }
}

// N-way (world > 2) proxy: the SAME doorbell invariants as the world==2 loop, per QP, with the round
// schedule selecting the partner. The rings are shared (one epoch in flight against ONE partner at a
// time), so `next_to_post`/`next_release` are still monotone GLOBAL epoch watermarks. The partner for
// epoch e is `rank ^ (1 << (e % rounds))`; the receive side publishes `cpu_done[p]` (per-peer line) and
// the send side posts to `peer_committed[rank]` in peer p's region (so each peer sees only OUR epoch).
// The I3 reuse gate keys on `tx_retired[p_prev]` where p_prev = partner(e - R) — the QP that shipped the
// slot R epochs ago (per-QP in-order retirement). drain_cq publishes each QP's retirement into its own
// `tx_retired[p]` line. `net_exchange`/`net_agree` stay pairwise on the head<->node control QP (P5 owns
// N-way agreement).
static void net_proxy_loop_nway(NetCtx* c, int core) {
    __atomic_store_n(&c->proxy_running, 1, __ATOMIC_RELEASE);
    if (core >= 0 && net_pin_thread(core)) {
        LOGE("FATAL: proxy failed to pin to core %d — aborting rather than running unpinned", core);
        tp_set_abort(c, 8);
        return;
    }

    uint64_t next_to_post = 1;   // proxy OWNS the posted epoch (I1)
    uint64_t next_release = 1;   // next peer epoch to hand to the local GPU

    // Liveness watchdog: same semantics as world==2, but the debt is global (posted vs released), since
    // every posted epoch is owed by SOME peer and the round schedule is symmetric across ranks.
    uint64_t wdog_since = 0;
    unsigned  wdog_ticks = 0;

    while (!c->aborted && !__atomic_load_n(flagp(c, TP_F_ABORT), __ATOMIC_ACQUIRE)) {
        int did_work = 0;

        // TP-4F2: a ctx with a parent (the single-stage reduce ctx) follows the primary link's abort — one
        // plain load per iteration; every other ctx has parent == NULL and skips this entirely.
        if (c->parent) {
            uint64_t pv = __atomic_load_n(flagp(c->parent, TP_F_ABORT), __ATOMIC_ACQUIRE);
            if (pv || c->parent->aborted) { tp_set_abort(c, (pv ? pv : 1ull) | (1ull << 61)); break; }
        }

        uint64_t rx_done = c->recv_gpu ? *flagp(c, TP_F_RX_DONE) : 0;
        // R10: exclusive-ize rx_done in v2 (see net_proxy_loop_world2) — the raw inclusive counter
        // leaves a phantom one-epoch debt at every quiescent point and aborts a healthy link.
        uint64_t matched = c->recv_gpu
            ? (next_release > rx_done + 1 ? next_release : rx_done + 1)
            : (next_release > rx_done ? next_release : rx_done);
        if ((c->released_epochs > 0 || rx_done > 0) && next_to_post > matched) {
            if (wdog_since == 0) { wdog_since = now_ns(); wdog_ticks = 0; }
            else if (((wdog_ticks++) & 0x3FF) == 0 && now_ns() - wdog_since > TP_WDOG_NS) {
                LOGE("WATCHDOG: peer has not matched posted epoch %llu for %llums (awaiting %llu)"
                     " — declaring peer dead; aborting",
                     (unsigned long long)(next_to_post - 1),
                     (unsigned long long)(TP_WDOG_NS / 1000000ull),
                     (unsigned long long)matched);
                tp_set_abort(c, 6);
                break;
            }
        } else {
            wdog_since = 0;
        }

        // -- SEND: the watermark says payload for every epoch <= w is in the send ring --
        uint64_t w = __atomic_load_n(flagp(c, TP_F_GPU_READY), __ATOMIC_ACQUIRE);
        if (w >= next_to_post) {
            if (c->ts_on) { uint64_t t = now_ns(); for (uint64_t e = next_to_post; e <= w; e++) stamp(c, e, TP_CTS_READY, t); }
            if (c->inject_delay_us_max) bench_delay_us(c, c->inject_delay_us_max);
            // Post ONE epoch at a time: consecutive epochs go to DIFFERENT partners (round schedule),
            // so a batched WR list would have to span QPs — post serially per epoch.
            for (uint64_t e = next_to_post; e <= w; e++) {
                // P3-1 per-epoch rule: decode-width epochs (wire <= default payload) go one-shot
                // into the sender-indexed block; wider (verify/prefill chunks) and every other
                // tree consumer (fp32 LM-head gather, maxloc) keep the round-keyed rings. The two
                // regions are disjoint, so mixed algorithms cannot collide.
                volatile uint64_t* ll = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_LOCAL_OFF);
                uint64_t wl = TP_LEN_TAG_BYTES(ll[e & (TP_LEN_EPOCHS - 1)]);
                int os = c->oneshot_uniform ? 1 : net_use_oneshot(c, wl);   // TP-4F2: uniform ctx = always one-shot
                if (c->oneshot && !c->oneshot_uniform && e <= 8)   // P3-1 DIAG: the wedge lives at the mixed-width boundary
                    LOGE("[os-dispatch] epoch=%llu wire=%llu payload_bytes=%u -> %s",
                         (unsigned long long)e, (unsigned long long)wl, c->payload_bytes,
                         os ? "ONESHOT" : "tree");
                int posted = os ? post_range_oneshot(c, e) : post_range_nway(c, e);
                if (posted < 0) break;
            }
            if (c->aborted || *flagp(c, TP_F_ABORT)) break;
            next_to_post = w + 1;
            did_work = 1;
        }

        // -- CQ: non-blocking drain -> per-peer tx_retired --
        // PLAN/10 #5 (2026-08-17): the n-way loop used an all-or-nothing gate — ANY cq_hold
        // >= 1 skipped drain_cq_nway ENTIRELY, so tx_retired[p] never advanced and the K1
        // reuse gate timed out → abort. That is why the cq-hold drill failed identically on
        // tree AND one-shot at world=4 (a drill defect, not a transport defect). Port the
        // graded hold from net_proxy_loop_world2: withhold only until `cq_hold` epochs are
        // unretired, then (optionally) for cq_hold_ns more so the gate demonstrably binds —
        // bounded backpressure, never a full stop. Counters (posted_epochs/retired_epochs)
        // are maintained by post_range_nway/post_range_oneshot and drain_cq_nway exactly as
        // in the world=2 loop, so the arithmetic is identical.
        int hold = 0;
        if (c->cq_hold) {
            uint64_t unret = c->posted_epochs - c->retired_epochs;
            if (unret < (uint64_t)c->cq_hold) {
                hold = 1; c->hold_since = 0;          // not yet at the threshold: keep withholding
            } else if (c->cq_hold_ns) {
                uint64_t t = now_ns();
                if (!c->hold_since) c->hold_since = t;
                if (t - c->hold_since < c->cq_hold_ns) hold = 1; else c->hold_since = 0;
            }
        }
        if (!hold) {
            int drained = drain_cq_nway(c);
            if (drained < 0) break;
            if (drained > 0) did_work = 1;
        }

        // -- ABORT FAN-OUT (B8 §1.5-7): mirror any peer's abort locally — one plain load per peer
        //    per iteration (I6). world>2 agree rides net_exchange_one (P5), never TP_F_AGREE_OUT, so
        //    the pairwise AGREE-shipping arm that lived here was dead code and is intentionally gone.
        {
            uint64_t peer_ab = 0;
            for (int p = 0; p < c->world; p++) {
                if (p == c->rank) continue;
                uint64_t v = *nway_flagp(c, p, TP_NWAY_ABORT_OFF);
                if (v) { peer_ab = v; break; }
            }
            if (peer_ab) { tp_set_abort(c, peer_ab | (1ull << 62)); break; }
        }

        // -- RECV, TP-4F2 uniform one-shot ctx: epoch e is owed by ALL world-1 peers (all-peers push into the
        //    sender-indexed block). The CPU bounce is the tree's, per peer: the peer's commit hint, the len
        //    tag, the slot's generation-tagged tail (R9) — each peer's slot is validated at its own
        //    sender-indexed address — then ONE fence and a RELEASE-store of cpu_done[p] = e for every peer
        //    (I5). Epochs release strictly in order; nothing of the round-keyed rings is read.
        if (c->oneshot_uniform) {
            uint64_t e = next_release;
            uint64_t advanced = 0;
            while (!c->aborted && !*flagp(c, TP_F_ABORT)) {
                int all = 1;
                for (int p = 0; p < c->world; p++) {
                    if (p == c->rank) continue;
                    if (*peer_committed_flagp(c, p) < e) { all = 0; break; }
                }
                if (!all) break;
                int ok = 1;
                for (int p = 0; p < c->world; p++) {
                    if (p == c->rank) continue;
                    if (!nway_recv_slot_at(c, e, p, oneshot_recv_slot_ptr(c, p, e))) { ok = 0; break; }
                }
                if (!ok) break;
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                for (int p = 0; p < c->world; p++) {
                    if (p == c->rank) continue;
                    __atomic_store_n(cpu_done_flagp(c, p), e, __ATOMIC_RELEASE);
                }
                advanced = e;
                e++;
            }
            if (advanced) {
                c->released_epochs = advanced;
                if (c->ts_on) stamp(c, advanced, TP_CTS_RELEASED, now_ns());
                next_release = advanced + 1;
                did_work = 1;
            }
        } else
        // -- RECV: CPU bounce, per-peer. Advance the global watermark while each epoch's partner slot
        //    releases independently (the round schedule is serial, so `next_release` advances in order).
        if (!c->recv_gpu) {
            // Find the highest contiguous prefix of epochs whose partner's commit has landed.
            uint64_t e = next_release;
            uint64_t advanced = 0;
            while (!c->aborted && !*flagp(c, TP_F_ABORT)) {
                int p = partner_rank_of(c, e);
                uint64_t pc = *peer_committed_flagp(c, p);
                if (pc < e) break;
                // Tail-epoch guard (same discipline as world==2): the payload was posted before the
                // commit on this QP; bounded-wait for the len tag + slot tail.
                if (!nway_recv_slot(c, e, p)) break;
                // Per-partner release ORDERING (the max-watermark read-before-write fix): the loop
                // advances one epoch per iteration, but with rounds > 1 consecutive epochs map to
                // DIFFERENT partners, so a single release-store would leave partner p's cpu_done line
                // at e while the pending epochs e+1 .. up-to-peer_commit (this same partner's next
                // round(s)) are not yet validated — and a K2 at one of those later epochs would see
                // cpu_done[p] = e < its target and stall while its payload sits unvalidated (the
                // max-watermark inversion, reversed). On reaching a NEW partner, drain the PREVIOUS
                // partner's backlog: validate every pending epoch of that partner (peer_committed[p]
                // is a per-QP monotone watermark, so its validated epochs are contiguous per QP) and
                // release its cpu_done to the highest validated one BEFORE advancing `e`. Then a
                // cpu_done[p] >= e observed by K2 always means epoch e's own slot was validated by
                // this proxy — never a later epoch's stand-in. The watermark (`advanced`) still moves
                // one epoch per iteration; only the cpu_done lines run ahead of it.
                int prev_p = partner_rank_of(c, e - 1);
                if (p != prev_p) {
                    uint64_t pc_prev = *peer_committed_flagp(c, prev_p);
                    uint64_t pp = e - 1 + (uint64_t)c->rounds;   // first pending epoch of prev partner
                    for (; pp <= pc_prev; pp += (uint64_t)c->rounds) {
                        if (!nway_recv_slot(c, pp, prev_p)) { advanced = e - 1; goto recv_out; }
                    }
                    uint64_t rel = pp - (uint64_t)c->rounds;     // highest validated epoch of prev_p
                    __atomic_thread_fence(__ATOMIC_SEQ_CST);
                    if (rel >= e - 1)
                        __atomic_store_n(cpu_done_flagp(c, prev_p), rel, __ATOMIC_RELEASE);
                }
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                __atomic_store_n(cpu_done_flagp(c, p), e, __ATOMIC_RELEASE);
                advanced = e;
                e++;
            }
recv_out:
            if (advanced) {
                c->released_epochs = advanced;
                if (c->ts_on) stamp(c, advanced, TP_CTS_RELEASED, now_ns());
                next_release = advanced + 1;
                did_work = 1;
            }
        }

        if (!did_work) cpu_relax();
    }

    // B8 §1.5-7: cross-rank abort fan-out epilogue. If WE aborted (code from a kernel, the watchdog,
    // a guard, or a mirrored peer), ship our code to every peer's abort slot so a one-rank abort
    // becomes an all-ranks cooperative stop — instead of 5 s dead-peer probes + supervisor re-arm.
    // Best-effort unsigned inline WRs; peers poll these slots in the loop above.
    {
        uint64_t ab = *flagp(c, TP_F_ABORT);
        // TP-4F2: this ctx's abort (a kernel's device status, a guard, the watchdog) must reach the primary
        // link too — its agree / step checks are the ones the host reads. Never overwrite an abort code the
        // parent already carries.
        if (ab && c->parent && !c->parent->aborted && !*flagp(c->parent, TP_F_ABORT))
            tp_set_abort(c->parent, ab | (1ull << 60));
        if (ab) {
            for (int p = 0; p < c->world; p++) {
                if (p == c->rank) continue;
                struct ibv_sge sg = { .addr = (uint64_t)(uintptr_t)&ab, .length = 8, .lkey = 0 };
                struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
                wr.wr_id = 0; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sg; wr.num_sge = 1;
                wr.send_flags = IBV_SEND_INLINE;
                wr.wr.rdma.remote_addr = peer_remote_addr(c, p) + c->nway_flags_off
                    + (size_t)c->rank * TP_CL + TP_NWAY_ABORT_OFF;
                wr.wr.rdma.rkey = peer_remote_rkey(c, p);
                ibv_post_send(peer_qp(c, p), &wr, &bad);
            }
        }
    }
}

// Wait (bounded) for the length tag + slot tail of epoch e from peer p, then report placement OK.
// Mirrors the world==2 RECV tail-epoch guard on a per-QP basis. Returns 1 when the slot is ready.
static int nway_recv_slot(NetCtx* c, uint64_t e, int p) {
    return nway_recv_slot_at(c, e, p, nway_recv_slot_ptr(c, e));
}
// `slot` = the receive slot's base: the round ring's (tree path) or the sender-indexed one-shot block's
// (TP-4F2 uniform ctx). Everything else is the tree path's guard, unchanged.
static int nway_recv_slot_at(NetCtx* c, uint64_t e, int p, char* slot) {
    volatile uint64_t* len_peer = (volatile uint64_t*)((char*)c->hbuf + TP_LEN_PEER_OFF);
    volatile uint64_t* lp = len_peer + (e & (TP_LEN_EPOCHS - 1));
    uint64_t tag = *lp;
    if (TP_LEN_TAG_EPOCH(tag) != e) {
        uint64_t t0 = now_ns();
        for (;;) {
            if (c->aborted || *flagp(c, TP_F_ABORT)) return 0;
            tag = *lp;
            if (TP_LEN_TAG_EPOCH(tag) == e) break;
            uint64_t dt = now_ns() - t0;
            if (dt > TP_TAIL_WAIT_NS) {
                LOGE("LEN-EPOCH GUARD FIRED (nway): len_peer[%llu]=%016llx expected epoch %llu (peer %d)",
                     (unsigned long long)(e & (TP_LEN_EPOCHS - 1)), (unsigned long long)tag,
                     (unsigned long long)e, p);
                c->tail_fires++;
                tp_set_abort(c, 2);
                return 0;
            }
            tp_host_backoff(dt);
        }
    }
    uint64_t len = TP_LEN_TAG_BYTES(tag);
    if (len == 0 || len > c->slot_stride - TP_TAIL_BYTES) {
        LOGE("LEN-EPOCH GUARD FIRED (nway): malformed len %llu for epoch %llu (peer %d)",
             (unsigned long long)len, (unsigned long long)e, p);
        c->tail_fires++;
        tp_set_abort(c, 2);
        return 0;
    }
    volatile uint64_t* tailp = (volatile uint64_t*)(slot + len);
    if (*tailp == e) {
        // R9 DIAGNOSTIC: checksum the EXACT bytes this rank's GPU is about to consume for epoch e.
        tp_diag_log_recv(c, e, p, slot, len);
        return 1;
    }
    uint64_t t0 = now_ns();
    for (;;) {
        if (c->aborted || *flagp(c, TP_F_ABORT)) return 0;
        if (*tailp == e) break;
        uint64_t dt = now_ns() - t0;
        if (dt > TP_TAIL_WAIT_NS) {
            uint64_t tail = *tailp;
            LOGE("TAIL-EPOCH GUARD FIRED (nway): slot %llu tail=%llu expected %llu (peer %d)",
                 (unsigned long long)(e & (TP_RING_SLOTS-1)), (unsigned long long)tail,
                 (unsigned long long)e, p);
            c->tail_fires++;
            tp_set_abort(c, 2);
            return 0;
        }
        tp_host_backoff(dt);
    }
    // R9 DIAGNOSTIC: same, on the recovered path (tail landed after the commit).
    tp_diag_log_recv(c, e, p, slot, len);
    return 1;
}

// R3b: forced signaled flush for quiesce / finite-bench end — post one signaled 8 B inline write and
// drain it, so every outstanding unsignaled WR becomes observably retired. Returns 0 on success.
int net_flush(NetCtx* c) {
    uint64_t v = 0;
    struct ibv_sge sg = { .addr = (uint64_t)&v, .length = 8, .lkey = 0 };
    struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sg; wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    // write into our own scratch at the peer: the last 8 B of its flags block is unused padding.
    // world>2 rides the control QP (head<->node); world==2 this is the single QP (identical).
    wr.wr.rdma.remote_addr = control_remote_addr(c) + TP_F_ABORT + 8;
    wr.wr.rdma.rkey = control_remote_rkey(c);
    if (ibv_post_send(control_qp(c), &wr, &bad)) { LOGE("flush post_send"); return -1; }
    for (;;) {
        struct ibv_wc wc; int n = ibv_poll_cq(c->cq_send, 1, &wc);
        if (n < 0) return -1;
        if (n == 0) { if (c->aborted) return -2; cpu_relax(); continue; }
        if (wc.status != IBV_WC_SUCCESS) { LOGE("flush wc %d", wc.status); return -3; }
        return 0;
    }
}

// ---------------------------------------------------------------- startup / audit channel

// One all-reduce EXCHANGE of nbytes over the retained WITH_IMM channel: ship send slot 0 to the peer's
// recv ring and block until the peer's write for the SAME generation has landed. Off the hot path —
// used by --net-test (the FP32-partial numerical audit) and as an out-of-band sanity/re-init channel.
//
// Receive-side generation discipline (both learned the hard way, same family as the R2d tail guard):
//   1. GEN-SLOTTED: generation g's payload goes to recv slot (g % TP_RING_SLOTS), never slot 0 for
//      every generation. The protocol is lockstep (each exchange pairs one send with one recv, and a
//      rank can complete at most one generation ahead of its peer), but that one-ahead generation is
//      enough to CLOBBER a slow reader: the node's broadcast_prompt zero-fill landed in the head's
//      recv slot while the head was still reading the sanity stamp from it — stamp read 0x0, bring-up
//      died (2026-07-27). With per-gen slots the protocols 1-ahead window can never overlap.
//   2. TAIL-TAGGED: the sender writes the generation into the payload's last 8 bytes, and the
//      receiver waits for THAT value before trusting the payload. A recv CQE proves delivery, not
//      cross-address placement visibility to this CPU on GB10 (Grace C2C reorder — the hot path's
//      tail-epoch guard exists for exactly this). The tag is part of the payload DMA, so it is the
//      placement proof for the bytes before it. Callers keep the last 2 words of every frame clear
//      (the "2 words headroom" in broadcast_prompt's FRAME_I32); --net-test pads its slot by 8.
int net_exchange(NetCtx* c, int nbytes) {
    if (nbytes < 16 || (size_t)nbytes > c->slot_stride - TP_TAIL_BYTES) {
        LOGE("exchange nbytes %d out of range (slot capacity %u)", nbytes, c->slot_stride - TP_TAIL_BYTES);
        return -1;
    }
    c->gen++;
    const uint64_t g = c->gen;
    // Tail tag BEFORE the post: it ships inside the same WRITE_WITH_IMM payload.
    *(uint64_t*)((char*)net_send_hptr(c) + nbytes - TP_TAIL_BYTES) = g;

    struct ibv_sge sge; memset(&sge,0,sizeof(sge));
    sge.addr=(uint64_t)net_send_hptr(c); sge.length=nbytes; sge.lkey=c->mr->lkey;
    struct ibv_send_wr wr, *bad; memset(&wr,0,sizeof(wr));
    wr.sg_list=&sge; wr.num_sge=1; wr.opcode=IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags=IBV_SEND_SIGNALED;
    wr.imm_data=(uint32_t)g;
    wr.wr_id=TP_XCHG_WR_ID;   // drain_cq hands this CQE over via xchg_send_done (see the define)
    // Control QP (head<->node): world==2 it is the single QP; world>2 it is the rank-0<->rank-1 QP
    // (net_exchange/net_agree stay pairwise on the control QP in P3 — P5 makes them N-way).
    wr.wr.rdma.remote_addr = xchg_remote_slot(c, g);   // peer's exchange slot for gen g (world 2: ctrl ring)
    wr.wr.rdma.rkey=control_remote_rkey(c);
    __atomic_store_n(&c->xchg_send_done, 0, __ATOMIC_RELEASE);
    if (ibv_post_send(control_qp(c),&wr,&bad)){ LOGE("post_send"); return -1; }

    int got_send=0, got_recv=0;
    const uint64_t t0 = now_ns();
    uint64_t last_probe = t0;
    while (!(got_send && got_recv)) {
        struct ibv_wc wc;
        // Once the proxy runs it OWNS cq_send (our send completion arrives via xchg_send_done);
        // polling it here could steal a hot-path epoch CQE from drain_cq. Pre-proxy there is no
        // hot path, so polling cq_send directly is both safe and required.
        int r = 0;
        if (!__atomic_load_n(&c->proxy_running, __ATOMIC_ACQUIRE))
            r = ibv_poll_cq(c->cq_send, 1, &wc);                    // our signaled send
        if (r == 0) r = ibv_poll_cq(c->cq_startup, 1, &wc);         // the peer's incoming WITH_IMM
        if (r<0){ LOGE("poll_cq"); return -1; }
        if (r==0){
            // The proxy may have retired our send CQE (shared cq_send) — accept its handoff.
            if (!got_send && __atomic_load_n(&c->xchg_send_done, __ATOMIC_ACQUIRE)) got_send = 1;
            if (got_send && got_recv) break;
            if (c->aborted) return -2;
            uint64_t dt = now_ns() - t0;
            // Dead-peer check (TP_LIVE_PROBE_NS comment): infinite QP retries make a dead peer
            // indistinguishable from a slow one at the verbs level — probe out-of-band instead.
            // Legitimate one-sided waits (peer model-load, user idle between requests) probe OK
            // and keep waiting; a dead peer fails twice and we abort LOUDLY (clean exit/re-arm).
            if (dt >= TP_LIVE_PROBE_NS && now_ns() - last_probe >= TP_LIVE_PROBE_NS) {
                last_probe = now_ns();
                if (c->peer_ip[0] && !peer_alive(c)) {
                    LOGE("exchange: peer %s unreachable on liveness port %d after %llus of silence"
                         " — peer is DEAD; aborting (code 10)",
                         c->peer_ip, c->liveness_port, (unsigned long long)(dt / 1000000000ull));
                    tp_set_abort(c, 10);
                    return -3;
                }
            }
            // TP-E: spin for the bounded budget, then 1 ms nanosleeps (tp_host_backoff) — a healthy
            // mid-request wait never sleeps; an idle wait still drops to ~0% CPU after the budget.
            tp_host_backoff(dt);
            continue;
        }
        if (wc.status != IBV_WC_SUCCESS){ LOGE("wc status %d op %d", wc.status, wc.opcode); return -3; }
        if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM || wc.opcode == IBV_WC_RECV) {
            if ((uint32_t)wc.imm_data != (uint32_t)g) {
                // CQEs retire in arrival order and the protocol is lockstep, so the imm for THIS
                // poll must be THIS generation. Anything else is a protocol desync — die loudly.
                LOGE("exchange: generation desync — imm %u while awaiting gen %llu",
                     (unsigned)wc.imm_data, (unsigned long long)g);
                return -3;
            }
            got_recv = 1;
            struct ibv_recv_wr rw, *rb; memset(&rw,0,sizeof(rw)); rw.num_sge=0;   // replenish
            if (ibv_post_recv(control_qp(c),&rw,&rb)){ LOGE("post_recv replenish"); return -1; }
        } else if (wc.wr_id == TP_XCHG_WR_ID) {
            got_send = 1;
        } else {
            // Unreachable by construction (pre-proxy there are no epoch WRs; post-proxy we do not
            // poll cq_send) — log, do NOT treat as ours: a foreign CQE says nothing about our send.
            LOGE("exchange: unexpected send-side CQE wr_id %llu — ignored", (unsigned long long)wc.wr_id);
        }
    }
    // Placement proof for the payload we are about to let the caller read (point 2 above). The
    // per-gen slot means this wait cannot be clobbered by the peer's next generation while we wait.
    {
        volatile uint64_t* tailp = (volatile uint64_t*)(xchg_recv_slot_ptr(c, g) + nbytes - TP_TAIL_BYTES);
        uint64_t tw = now_ns();
        while (*tailp != g) {
            if (c->aborted) return -2;
            uint64_t dt = now_ns() - tw;
            if (dt > TP_TAIL_WAIT_NS) {
                LOGE("exchange: recv payload tail for gen %llu never landed (tail=%llu, waited %llums)"
                     " — placement visibility violated; aborting",
                     (unsigned long long)g, (unsigned long long)*tailp,
                     (unsigned long long)(TP_TAIL_WAIT_NS / 1000000ull));
                c->tail_fires++;
                tp_set_abort(c, 2);
                return -3;
            }
            tp_host_backoff(dt);
        }
        __atomic_thread_fence(__ATOMIC_SEQ_CST);   // payload reads ordered behind the tag read
    }
    c->last_xchg_gen = g;
    return 0;
}

// Publish this rank's lockstep token and block until the peer's token for the SAME step arrives.
// `val` must carry the step in its high bits so a stale peer value is distinguishable from a fresh one.
// Returns the peer's token, or 0 if aborted/timed out. Called once per MTP step: ~1 wire RTT, not per barrier.
// Deadline: without one, a peer whose main thread hangs (proxy/NIC still alive, so no error CQE ever)
// hangs THIS thread forever, mid-step, silently. TP_WDOG_NS is ~10^5x the expected RTT; the wait spins
// for the TP-E budget (tp_host_backoff, default 20 ms), then 1 ms nanosleeps. Abort code 7.
uint64_t net_agree(NetCtx* c, uint64_t val, uint64_t step_mask, uint64_t step_val) {
    __atomic_store_n(flagp(c, TP_F_AGREE_OUT), val, __ATOMIC_RELEASE);
    const uint64_t t0 = now_ns();
    for (;;) {
        uint64_t in = __atomic_load_n(flagp(c, TP_F_AGREE_IN), __ATOMIC_ACQUIRE);
        // `in != 0`: zero is "no token yet", NOT a step-0 token. A fresh ctx's zeroed AGREE_IN
        // satisfies a step-0 mask (step_val == 0), so the FIRST rank to arrive used to rendezvous
        // with a phantom and return 0 (== abort) while its peer sailed through — a 50/50 startup
        // race the step-3 drill exposed. A real peer token at step 0 is always nonzero (it
        // carries count>0 and a hash), so rejecting zero costs nothing.
        if (in != 0 && (in & step_mask) == step_val) return in;
        if (c->aborted || *flagp(c, TP_F_ABORT)) return 0;
        uint64_t dt = now_ns() - t0;
        if (dt >= TP_WDOG_NS) {
            LOGE("net_agree: peer token for step %llu did not arrive within %llus — aborting",
                 (unsigned long long)step_val, (unsigned long long)(TP_WDOG_NS / 1000000000ull));
            tp_set_abort(c, 7);
            return 0;
        }
        tp_host_backoff(dt);
    }
}

// P5: one world>2 control-plane rendezvous with a SPECIFIC peer (head<->node), over the dedicated
// per-rank control receive slots. Plain RDMA_WRITE + tail-tag placement proof — NO recv CQE, NO
// WITH_IMM, so it never touches cq_startup (which net_exchange owns for the pairwise dflash channel)
// and never contends with the hot-path recv ring.
//
// Protocol (both sides of the pair run the same function with mirrored peer_rank):
//   * `g = ++xchg_gen[peer_rank]` — a per-peer generation; the pair increments in lockstep so both
//     sides agree on g for this exchange.
//   * sender writes the caller's payload into send slot 0 (already staged by the caller) and stamps
//     the tail tag `(g << 8) | my_rank` at nbytes - TP_TAIL_BYTES.
//   * sender posts ONE SIGNALED RDMA_WRITE to peer's control slot `ctrl_recv_off + my_rank*slot_stride`
//     (wr_id = TP_XCHG_WR_ID, so drain_cq_nway hands the CQE over via xchg_send_done as today).
//   * receiver waits (bounded spin -> nanosleep, with the out-of-band liveness probe) for the tail tag
//     `(g << 8) | peer_rank` in ITS control slot for sender rank `peer_rank` — the rank bits make the
//     proof unambiguous even though all ranks share the MR.
// world==2 is never routed here (the pairwise net_exchange/net_agree stay untouched).
//
// TP-4D (audit C5): `deadline_ns` (0 = none, the pre-TP-4D behaviour of every caller) bounds EACH of the two waits
// (send CQE, peer tail tag) from the moment that wait began. The dead-peer probe only proves the peer PROCESS is up
// (its liveness port answers), so a peer that is alive but never reaches this lockstep point — GPU hung in a kernel,
// host thread stuck — held the other ranks forever: the proxy watchdog (code 6) only fires with hot-path epochs
// outstanding, and between lockstep points there are none. On expiry: abort code 12, return -4. Routed only at
// world > 2 (world == 2 never reaches this function), so the pairwise wire path is untouched.
static int exchange_one_impl(NetCtx* c, int peer_rank, int nbytes, uint64_t deadline_ns) {
    if (c->world == 2) { LOGE("net_exchange_one: world==2 must use net_exchange"); return -1; }
    if (peer_rank < 0 || peer_rank >= c->world || peer_rank == c->rank) {
        LOGE("net_exchange_one: bad peer_rank %d (world %d)", peer_rank, c->world);
        return -1;
    }
    if (!c->peers[peer_rank].valid) { LOGE("net_exchange_one: peer %d QP not valid", peer_rank); return -1; }
    if (nbytes < 16 || (size_t)nbytes > c->slot_stride - TP_TAIL_BYTES) {
        LOGE("exchange_one nbytes %d out of range (slot capacity %u)", nbytes, c->slot_stride - TP_TAIL_BYTES);
        return -1;
    }

    const uint64_t g = ++c->xchg_gen[peer_rank];
    const uint64_t tag = (g << 8) | (uint64_t)c->rank;   // sender-stamped: my_rank on the wire
    // Tail tag BEFORE the post: it ships inside the same RDMA_WRITE payload.
    *(uint64_t*)(ctrl_send_slot(c) + nbytes - TP_TAIL_BYTES) = tag;

    struct ibv_sge sge; memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)ctrl_send_slot(c); sge.length = (uint32_t)nbytes; sge.lkey = c->mr->lkey;
    struct ibv_send_wr wr, *bad; memset(&wr, 0, sizeof(wr));
    wr.wr_id = TP_XCHG_WR_ID;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = ctrl_recv_raddr(c, peer_rank, g);
    wr.wr.rdma.rkey = peer_remote_rkey(c, peer_rank);

    // B8 §1.5-4: snapshot the send-CQE generation BEFORE posting; the proxy bumps xchg_send_seq on
    // every TP_XCHG_WR_ID CQE, so a monotone `>` proves OUR CQE landed even if several exchanges
    // interleave. The wait is now deadline-bound by the same TP_LIVE_PROBE_NS dead-peer probe the
    // placement wait uses — a lost CQE can no longer hang forever silently.
    const uint64_t send_seq_before = __atomic_load_n(&c->xchg_send_seq, __ATOMIC_ACQUIRE);
    if (ibv_post_send(peer_qp(c, peer_rank), &wr, &bad)) { LOGE("exchange_one post_send"); return -1; }

    int got_send = 0;
    const uint64_t s_t0 = now_ns();
    uint64_t s_last_probe = s_t0;
    for (;;) {
        if (!__atomic_load_n(&c->proxy_running, __ATOMIC_ACQUIRE)) {
            struct ibv_wc wc; int r = ibv_poll_cq(c->cq_send, 1, &wc);
            if (r < 0) { LOGE("exchange_one poll_cq"); return -1; }
            if (r > 0) {
                if (wc.status != IBV_WC_SUCCESS) { LOGE("exchange_one send wc status %d", wc.status); return -3; }
                if (wc.wr_id == TP_XCHG_WR_ID) got_send = 1;
            }
        } else if (__atomic_load_n(&c->xchg_send_seq, __ATOMIC_ACQUIRE) > send_seq_before) {
            got_send = 1;
        }
        if (got_send) break;
        if (c->aborted || *flagp(c, TP_F_ABORT)) {
            LOGE("exchange_one rc=-2: aborted (peer %d), TP_F_ABORT=%llu",
                 peer_rank, (unsigned long long)*flagp(c, TP_F_ABORT));
            return -2;
        }
        uint64_t s_dt = now_ns() - s_t0;
        if (deadline_ns && s_dt >= deadline_ns) {
            LOGE("exchange_one: lockstep deadline (send CQE to peer %d) expired after %llums — peer alive but not progressing; aborting (code 12)",
                 peer_rank, (unsigned long long)(s_dt / 1000000ull));
            tp_set_abort(c, 12);
            return -4;
        }
        if (s_dt >= TP_LIVE_PROBE_NS && now_ns() - s_last_probe >= TP_LIVE_PROBE_NS) {
            s_last_probe = now_ns();
            if (!peer_alive_at(c, peer_rank)) {
                LOGE("exchange_one: peer %d unreachable (send CQE) after %llus — DEAD; aborting (code 10)",
                     peer_rank, (unsigned long long)(s_dt / 1000000000ull));
                tp_set_abort(c, 10);
                return -3;
            }
        }
        cpu_relax();
    }

    // Placement proof: wait the peer's rank-stamped tail tag in OUR control RING slot (generation-
    // keyed, so the peer's next exchange cannot clobber this one before we observe it) for sender
    // rank `peer_rank`. Same discipline as net_exchange (bounded spin -> nanosleep -> liveness probe).
    volatile uint64_t* tailp = (volatile uint64_t*)(ctrl_recv_slot(c, peer_rank, g) + nbytes - TP_TAIL_BYTES);
    const uint64_t expect = (g << 8) | (uint64_t)peer_rank;   // sender's rank on the receive side
    const uint64_t t0 = now_ns();
    uint64_t last_probe = t0;
    for (;;) {
        if (*tailp == expect) break;
        if (c->aborted || *flagp(c, TP_F_ABORT)) return -2;
        uint64_t dt = now_ns() - t0;
        if (deadline_ns && dt >= deadline_ns) {
            LOGE("exchange_one: lockstep deadline (frame from peer %d) expired after %llums — peer alive but not progressing; aborting (code 12)",
                 peer_rank, (unsigned long long)(dt / 1000000ull));
            tp_set_abort(c, 12);
            return -4;
        }
        if (dt >= TP_LIVE_PROBE_NS && now_ns() - last_probe >= TP_LIVE_PROBE_NS) {
            last_probe = now_ns();
            if (!peer_alive_at(c, peer_rank)) {
                LOGE("exchange_one: peer %d unreachable after %llus — DEAD; aborting (code 10)",
                     peer_rank, (unsigned long long)(dt / 1000000000ull));
                tp_set_abort(c, 10);
                return -3;
            }
        }
        tp_host_backoff(dt);
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);   // payload reads ordered behind the tag read
    // R10: copy the validated payload to the stable per-sender slot — net_ctrl_recv_hptr readers
    // have no generation bookkeeping, and the ring slot may be overwritten by a later exchange.
    memcpy(ctrl_last_slot(c, peer_rank), (const void*)ctrl_recv_slot(c, peer_rank, g), (size_t)nbytes);
    // TP-4D: retire the consumed frame. The ring slot is next written at generation g + TP_CTRL_RING, never before
    // we get here (a pair runs at most one agree ahead: the peer cannot complete its exchange g+3 without OUR frame
    // g+3, which we post only after this function returned), and the tag wait above compares ONE u64 at
    // `nbytes - 8`. Frames of different wire sizes share these slots (the world > 2 lockstep ops carry several
    // sizes), so a payload word pair an older, LARGER frame left at that offset — a device epoch, a digest, a
    // counter — could equal a future expected tag `(g << 8) | rank` and make the wait return on stale data before
    // the peer posted. Zeroing every byte the consumed frame wrote leaves the slot "all zero" until the next post,
    // so a stale interior word can never be mistaken for a tag. (The slot escapes to the NIC, so the memset is not a
    // dead store; the fence orders it before our next post.)
    memset((void*)ctrl_recv_slot(c, peer_rank, g), 0, (size_t)nbytes);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return 0;
}

int net_exchange_one(NetCtx* c, int peer_rank, int nbytes) { return exchange_one_impl(c, peer_rank, nbytes, 0); }
// TP-4D: the lockstep-deadline variant (see exchange_one_impl). deadline_ns == 0 is exactly net_exchange_one.
int net_exchange_one_dl(NetCtx* c, int peer_rank, int nbytes, uint64_t deadline_ns) {
    return exchange_one_impl(c, peer_rank, nbytes, deadline_ns);
}

void net_abort(NetCtx* c){ if (c) tp_set_abort(c, 1); }

void net_shutdown(NetCtx* c) {
    if (!c) return;
    if (c->qp) ibv_destroy_qp(c->qp);
    if (c->peers) {
        for (int p = 0; p < c->world; p++) {
            if (c->peers[p].qp) ibv_destroy_qp(c->peers[p].qp);
        }
        free(c->peers);
        c->peers = NULL;
    }
    if (c->mr) ibv_dereg_mr(c->mr);
    if (c->hbuf) cudaFreeHost(c->hbuf);
    if (c->dev_ctx_h) cudaFreeHost(c->dev_ctx_h);
    if (c->gpu_ts_h) cudaFreeHost(c->gpu_ts_h);
    if (c->cpu_ts) free(c->cpu_ts);
    if (c->cq_send) ibv_destroy_cq(c->cq_send);
    if (c->cq_startup) ibv_destroy_cq(c->cq_startup);
    if (c->pd) ibv_dealloc_pd(c->pd);
    if (c->ctx) ibv_close_device(c->ctx);
    free(c);
}
