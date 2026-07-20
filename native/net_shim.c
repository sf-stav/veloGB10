// TP=2 comm transport shim — flat C ABI over libibverbs, for src/net.rs.
//
// GB10 has NO GPUDirect -> comm buffers are cudaHostAlloc + ibv_reg_mr (coherent to GPU+CPU+NIC).
// RC QP, RoCEv2 GID index 3, TCP control-plane handshake (rdma_cm fails on this RoCE).
//
// HOT PATH = the doorbell all-reduce (see native/tp_doorbell.h for the invariants I1-I9, and
// tp_doorbell_ref/ for the long-form rationale from three expert review rounds). The short version:
//
//   send:  proxy owns a monotone epoch; per barrier it posts ONE linked WR chain —
//          RDMA_WRITE send_ring[s] -> peer recv_ring[s] (unsignaled), then an 8 B IBV_SEND_INLINE
//          epoch -> peer flags.peer_committed (signaled every S). Same QP => the payload is placed
//          before the doorbell at the peer. Plain WRITE consumes no peer recv WQE, so the RNR-NAK
//          class (ms-scale bimodal stalls when barriers cluster) cannot happen.
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

// Liveness deadline for the proxy watchdog and net_agree: 10 s. A hang is forever, so any large
// value discriminates; the cost of a FALSE fire is catastrophic (a one-rank abort silently corrupts
// the peer's all-reduces), so err high. Legitimate mid-run unmatched debt is ms-scale (the
// rendezvous bounds skew to ~1 barrier); load-time skew is excluded by arming after the first
// rendezvous. Abort codes: 6 = watchdog, 7 = agree timeout, 8 = pin failure
// (1 user, 2 tail-guard, 3 post_send, 4 CQE, 5 agree-post already in use).
#define TP_WDOG_NS 10000000000ull
// How long the RECV path waits for a slot's tail-epoch to land after the commit for that epoch was
// already observed (relaxed-PCIe reorder window) before declaring the payload lost. The payload was
// posted before the commit on a reliable QP, so the true lag is ns-us; 1 ms is ~100x the barrier
// floor and 1/10000 of the watchdog — beyond it, something is genuinely broken.
#define TP_TAIL_WAIT_NS 1000000ull

typedef struct NetCtx {
    struct ibv_context* ctx;
    struct ibv_pd*      pd;
    struct ibv_cq*      cq_send;     // hot path (R2b: separate from the startup CQ)
    struct ibv_cq*      cq_startup;  // retained WITH_IMM handshake / out-of-band channel
    struct ibv_qp*      qp;
    struct ibv_mr*      mr;

    void*    hbuf;            // cudaHostAlloc host ptr: [flags][send_ring][recv_ring]
    void*    dbuf;            // matching CUDA device ptr
    size_t   region_bytes;
    unsigned slot_stride;     // per-slot bytes, sized for the FP32 payload (round-3)
    unsigned payload_bytes;   // ACTIVE payload bytes (bf16 hidden*2, or fp32 hidden*4)

    tp_dev_ctx* dev_ctx_h;    // mapped pinned tp_dev_ctx (host view)
    void*       dev_ctx_d;    // device ptr to the same

    uint64_t remote_addr;     // peer MR base
    uint32_t remote_rkey;
    uint32_t gen;             // WITH_IMM exchange generation (startup path only)

    volatile int aborted;
    int      port_num, gid_idx, rank;

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
    int       tail_drill;            // GB10_TP_TAIL_DRILL: invert commit/payload order every 4096th epoch
    uint64_t  posted_epochs, retired_epochs, released_epochs;
    uint64_t  agree_last;            // last lockstep token shipped
    uint32_t  rng;
} NetCtx;

// ---------------------------------------------------------------- helpers

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline void cpu_relax(void) { __asm__ __volatile__("yield" ::: "memory"); }

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
// Cooperative abort (I9): a status word, never a trap. Downstream kernels see it and no-op.
static void tp_set_abort(NetCtx* c, uint64_t code) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(flagp(c, TP_F_ABORT), code, __ATOMIC_RELEASE);
    c->aborted = 1;
}

typedef struct { uint32_t qpn, psn; uint64_t addr; uint32_t rkey; union ibv_gid gid; } Exch
    __attribute__((packed));

static int tcp_exchange(int rank, const char* peer_ip, int port, Exch* lo, Exch* re) {
    int sock;
    if (rank == 0) {
        int ls = socket(AF_INET, SOCK_STREAM, 0); int o = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
        if (bind(ls,(struct sockaddr*)&a,sizeof(a))<0){ LOGE("bind: %s",strerror(errno)); return -1; }
        if (listen(ls,1)<0){ LOGE("listen"); return -1; }
        sock = accept(ls, NULL, NULL); close(ls);
        if (sock<0){ LOGE("accept"); return -1; }
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,peer_ip,&a.sin_addr);
        int ok=-1; for(int i=0;i<400;i++){ if(connect(sock,(struct sockaddr*)&a,sizeof(a))==0){ok=0;break;} usleep(50000);}
        if (ok){ LOGE("connect %s:%d: %s",peer_ip,port,strerror(errno)); return -1; }
    }
    if (write(sock,lo,sizeof(*lo))!=(ssize_t)sizeof(*lo)){ LOGE("w exch"); close(sock); return -1; }
    if (read (sock,re,sizeof(*re))!=(ssize_t)sizeof(*re)){ LOGE("r exch"); close(sock); return -1; }
    close(sock);
    return 0;
}

// ---------------------------------------------------------------- init

// rank: 0 = listen (head), 1 = connect (node).
// `fp32_capacity_bytes` sizes the ring slots (allocate for the FP32 payload from day one so the
// precision switch never re-addresses the rings); `payload_bytes` is what actually ships this run.
NetCtx* net_init(int rank, const char* peer_ip, int tcp_port, const char* dev_name,
                 int gid_idx, int fp32_capacity_bytes, int payload_bytes) {
    if (payload_bytes <= 0 || payload_bytes > fp32_capacity_bytes) {
        LOGE("payload_bytes %d out of range (capacity %d)", payload_bytes, fp32_capacity_bytes);
        return NULL;
    }
    if (TP_SIGNAL_EVERY > TP_RING_SLOTS) { LOGE("I4 violated: S=%d > R=%d", TP_SIGNAL_EVERY, TP_RING_SLOTS); return NULL; }

    NetCtx* c = (NetCtx*)calloc(1, sizeof(NetCtx));
    c->port_num = 1; c->gid_idx = gid_idx; c->rank = rank; c->rng = 0x9E3779B9u ^ (unsigned)rank;
    c->payload_bytes = (unsigned)payload_bytes;
    c->tail_drill = getenv("GB10_TP_TAIL_DRILL") != NULL;   // test-only: see post_range
    if (c->tail_drill) LOGE("TAIL DRILL ON: inverting commit/payload order every 4096th epoch");
    // slot = payload capacity + 8 B tail epoch, 64 B aligned so no two slots share a line
    c->slot_stride = (unsigned)(((size_t)fp32_capacity_bytes + TP_TAIL_BYTES + TP_CL - 1) & ~(size_t)(TP_CL - 1));
    c->region_bytes = TP_RING_BASE + (size_t)2 * TP_RING_SLOTS * c->slot_stride;

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

    if (cudaHostAlloc(&c->hbuf, c->region_bytes, cudaHostAllocMapped|cudaHostAllocPortable) != cudaSuccess){
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
    if (cudaHostAlloc((void**)&c->dev_ctx_h, sizeof(tp_dev_ctx),
                      cudaHostAllocMapped|cudaHostAllocPortable) != cudaSuccess){
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
    if (tcp_exchange(rank, peer_ip, tcp_port, &lo, &re)) return NULL;
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

// ---------------------------------------------------------------- accessors

// Set the ACTIVE payload for the hot path. Called once, after the model config is known and BEFORE the
// proxy thread starts (the proxy and K1/K2 both read these, and I8 forbids mutating protocol state
// underneath a running system). `fp32` selects the FP32-preserving production reduction in K2.
int net_set_payload(NetCtx* c, int payload_bytes, int fp32) {
    // 8 B multiple, not 4: K1 writes the 8 B tail epoch at slot + payload_bytes, so a 4-mod-8
    // payload would make that a misaligned 8 B GPU store (runtime misaligned-address error).
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

void* net_ctx_dptr(NetCtx* c)   { return c->dev_ctx_d; }          // K1/K2 kernel arg (the ONLY one)
void* net_flags_dptr(NetCtx* c) { return c->dbuf; }
void* net_send_hptr(NetCtx* c)  { return (char*)c->hbuf + TP_RING_BASE; }                 // slot 0
void* net_recv_hptr(NetCtx* c)  { return (char*)c->hbuf + TP_RING_BASE
                                       + (size_t)TP_RING_SLOTS * c->slot_stride; }        // slot 0
void* net_send_dptr(NetCtx* c)  { return (char*)c->dbuf + TP_RING_BASE; }
void* net_recv_dptr(NetCtx* c)  { return (char*)c->dbuf + TP_RING_BASE
                                       + (size_t)TP_RING_SLOTS * c->slot_stride; }
unsigned long long net_device_epoch(NetCtx* c) { return c->dev_ctx_h->epoch; }
unsigned long long net_gate_waits(NetCtx* c)   { return c->dev_ctx_h->gate_waits; }
unsigned long long net_gpu_ready(NetCtx* c)    { return *flagp(c, TP_F_GPU_READY); }
unsigned long long net_tail_fires(NetCtx* c)   { return c->tail_fires; }
unsigned long long net_abort_status(NetCtx* c) { return *flagp(c, TP_F_ABORT); }

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
// to form a batch). Per epoch, two chained WRs on the same QP:
//   WR0: RDMA_WRITE send_ring[s] -> peer recv_ring[s], length payload+tail   (unsignaled)
//   WR1: RDMA_WRITE 8 B INLINE epoch -> peer flags.peer_committed  (signaled iff e % S == 0)
// GB10_TP_TAIL_DRILL (test only, read once in net_init): every 4096th epoch INVERTS the pair —
// the commit is posted first and the payload second (signaling moves to the payload WR, which now
// posts last, so the CQE still retires both in order). The receiver's tail-epoch wait must engage
// and recover; that is the end-to-end proof of the load-bearing tail guard.
// Returns 0 on success.
static int post_range(NetCtx* c, uint64_t first, uint64_t last) {
    unsigned cnt = (unsigned)(last - first + 1);
    if (cnt > TP_MAX_POST_BATCH) cnt = TP_MAX_POST_BATCH;

    struct ibv_send_wr wr[2 * TP_MAX_POST_BATCH];
    struct ibv_sge     sg[2 * TP_MAX_POST_BATCH];
    uint64_t           epv[TP_MAX_POST_BATCH];    // inline source; copied into the WQE at post time

    for (unsigned i = 0; i < cnt; i++) {
        uint64_t e = first + i;
        epv[i] = e;
        // Drill: does this epoch ship commit-before-payload?
        const int invert = c->tail_drill && (e % 4096 == 0);
        const int pi = invert ? 1 : 0;        // payload WR slot within the pair
        const int ci = invert ? 0 : 1;        // commit WR slot within the pair

        sg[2*i+pi] = (struct ibv_sge){ .addr   = (uint64_t)send_slot(c, e),
                                       .length = c->payload_bytes + TP_TAIL_BYTES,
                                       .lkey   = c->mr->lkey };
        sg[2*i+ci] = (struct ibv_sge){ .addr = (uint64_t)&epv[i], .length = TP_TAIL_BYTES, .lkey = 0 };

        memset(&wr[2*i], 0, sizeof(wr[0])); memset(&wr[2*i+1], 0, sizeof(wr[0]));
        // payload WR
        wr[2*i+pi].wr_id      = e;
        wr[2*i+pi].opcode     = IBV_WR_RDMA_WRITE;
        wr[2*i+pi].sg_list    = &sg[2*i+pi]; wr[2*i+pi].num_sge = 1;
        // signaled iff e % S == 0; in drill mode the payload WR carries the signal (it posts last,
        // so its CQE retires the commit WR that precedes it — same accounting as the normal path).
        wr[2*i+pi].send_flags = invert ? ((e % TP_SIGNAL_EVERY == 0) ? IBV_SEND_SIGNALED : 0) : 0;
        wr[2*i+pi].wr.rdma.remote_addr = peer_recv_raddr(c, e);
        wr[2*i+pi].wr.rdma.rkey        = c->remote_rkey;

        // commit WR (epoch -> peer_committed)
        wr[2*i+ci].wr_id      = e;                                 // CQE carries the retired epoch
        wr[2*i+ci].opcode     = IBV_WR_RDMA_WRITE;
        wr[2*i+ci].sg_list    = &sg[2*i+ci]; wr[2*i+ci].num_sge = 1;
        wr[2*i+ci].send_flags = IBV_SEND_INLINE |
                                (invert ? 0 : ((e % TP_SIGNAL_EVERY == 0) ? IBV_SEND_SIGNALED : 0));
        wr[2*i+ci].wr.rdma.remote_addr = c->remote_addr + TP_F_PEER_COMMITTED;
        wr[2*i+ci].wr.rdma.rkey        = c->remote_rkey;

        // The chain is ALWAYS physical (2i -> 2i+1 -> 2i+2): the pair order on the wire is decided
        // by which WR sits in which slot, never by the links. (The first drill version linked by
        // role, which under inversion skipped every payload WR entirely — the watchdog+abort+divergence
        // path caught it loudly, exactly as designed, but the drill itself was invalid.)
        wr[2*i].next   = &wr[2*i+1];
        wr[2*i+1].next = (i + 1 < cnt) ? &wr[2*(i+1)] : NULL;
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
        if (wc[i].wr_id > hi) hi = wc[i].wr_id;
    }
    if (hi) {
        __atomic_store_n(flagp(c, TP_F_TX_RETIRED), hi, __ATOMIC_RELEASE);
        c->retired_epochs = hi;
        if (c->ts_on) stamp(c, hi, TP_CTS_CQE, now_ns());
    }
    return n;
}

// Persistent CPU proxy loop, on its own pinned thread. The main decode thread never syncs: it queues
// K1/K2 per reduction on the GPU stream and races ahead. This loop is the only mutator of tx_retired
// and cpu_done (I8).
void net_proxy_loop(NetCtx* c, int core) {
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
        if (c->released_epochs > 0 && next_to_post > next_release) {
            if (wdog_since == 0) { wdog_since = now_ns(); wdog_ticks = 0; }
            else if (((wdog_ticks++) & 0x3FF) == 0 && now_ns() - wdog_since > TP_WDOG_NS) {
                LOGE("WATCHDOG: peer has not matched posted epoch %llu for %llums (awaiting %llu)"
                     " — declaring peer dead; aborting",
                     (unsigned long long)(next_to_post - 1),
                     (unsigned long long)(TP_WDOG_NS / 1000000ull),
                     (unsigned long long)next_release);
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

        // -- RECV: CPU bounce for visibility (I5) --
        uint64_t pc = *flagp(c, TP_F_PEER_COMMITTED);          // plain volatile load, no RMW (I6)
        if (pc >= next_release) {
            if (c->ts_on) stamp(c, pc, TP_CTS_PEERSEEN, now_ns());
            // Tail-epoch guard (R2d), now LOAD-BEARING. The old assumption — "same QP => the
            // payload is placed before the commit" — is FALSE on this platform: at epoch ~4.36M
            // under a prefill flood the receiver observed peer_committed=e while the slot still
            // held generation e-R's tail (Grace C2C + relaxed PCIe ordering let the 8 B inline
            // commit bypass the large payload write; RC guarantees DELIVERY, not cross-address
            // placement order). So peer_committed is only a HINT of how far to check: the slot's
            // trailing u64 is the actual commit. Wait (bounded) for each tail to read its epoch —
            // the payload was posted before the commit on a reliable QP, so it lands momentarily;
            // and the reuse-gate chain (the peer cannot post WR0(e+R) until WR0(e) has retired,
            // which requires placement here) means the slot cannot be overwritten by a later
            // generation while we wait. Abort only if a payload genuinely never lands.
            int ok = 1;
            uint64_t e = next_release;
            for (; e <= pc; e++) {
                volatile uint64_t* tailp = (volatile uint64_t*)(recv_slot(c, e) + c->payload_bytes);
                if (*tailp == e) continue;
                uint64_t t0 = now_ns();
                for (;;) {
                    if (c->aborted || *flagp(c, TP_F_ABORT)) { ok = 0; break; }
                    if (*tailp == e) break;
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
                    if (dt >= 50000ull) { struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 }; nanosleep(&ts, NULL); }
                    else cpu_relax();
                }
                if (!ok) break;
                c->tail_waits++;
                if (c->tail_waits <= 8 || (c->tail_waits & (c->tail_waits - 1)) == 0)
                    LOGE("tail-epoch wait engaged for epoch %llu (%lluth) — payload landed after the commit (recovered)",
                         (unsigned long long)e, (unsigned long long)c->tail_waits);
            }
            if (!ok) break;
            // Full fence, then RELEASE-store cpu_done: when the GPU acquire-loads it, this core's
            // coherent view of the NIC's payload writes is ordered behind the flag read (I5).
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            __atomic_store_n(flagp(c, TP_F_CPU_DONE), pc, __ATOMIC_RELEASE);
            c->released_epochs = pc;
            if (c->ts_on) stamp(c, pc, TP_CTS_RELEASED, now_ns());
            next_release = pc + 1;
            did_work = 1;
        }

        if (!did_work) cpu_relax();   // plain load + yield, never an atomic RMW (I6)
    }
}

// R3b: forced signaled flush for quiesce / finite-bench end — post one signaled 8 B inline write and
// drain it, so every outstanding unsignaled WR becomes observably retired. Returns 0 on success.
int net_flush(NetCtx* c) {
    uint64_t v = 0;
    struct ibv_sge sg = { .addr = (uint64_t)&v, .length = 8, .lkey = 0 };
    struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sg; wr.num_sge = 1;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    // write into our own scratch at the peer: the last 8 B of its flags block is unused padding
    wr.wr.rdma.remote_addr = c->remote_addr + TP_F_ABORT + 8;
    wr.wr.rdma.rkey = c->remote_rkey;
    if (ibv_post_send(c->qp, &wr, &bad)) { LOGE("flush post_send"); return -1; }
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
// recv slot 0 and block until the peer's write has landed. Off the hot path — used by --net-test (the
// FP32-partial numerical audit) and as an out-of-band sanity/re-init channel (I8).
int net_exchange(NetCtx* c, int nbytes) {
    c->gen++;
    struct ibv_sge sge; memset(&sge,0,sizeof(sge));
    sge.addr=(uint64_t)net_send_hptr(c); sge.length=nbytes; sge.lkey=c->mr->lkey;
    struct ibv_send_wr wr, *bad; memset(&wr,0,sizeof(wr));
    wr.sg_list=&sge; wr.num_sge=1; wr.opcode=IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags=IBV_SEND_SIGNALED;
    wr.imm_data=c->gen;
    wr.wr.rdma.remote_addr = c->remote_addr + TP_RING_BASE + (uint64_t)TP_RING_SLOTS * c->slot_stride;
    wr.wr.rdma.rkey=c->remote_rkey;
    if (ibv_post_send(c->qp,&wr,&bad)){ LOGE("post_send"); return -1; }

    int got_send=0, got_recv=0;
    while (!(got_send && got_recv)) {
        struct ibv_wc wc;
        int r = ibv_poll_cq(c->cq_send, 1, &wc);            // our signaled send
        if (r == 0) r = ibv_poll_cq(c->cq_startup, 1, &wc); // the peer's incoming WITH_IMM
        if (r<0){ LOGE("poll_cq"); return -1; }
        if (r==0){ if (c->aborted) return -2; cpu_relax(); continue; }
        if (wc.status != IBV_WC_SUCCESS){ LOGE("wc status %d op %d", wc.status, wc.opcode); return -3; }
        if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM || wc.opcode == IBV_WC_RECV) {
            got_recv = 1;
            struct ibv_recv_wr rw, *rb; memset(&rw,0,sizeof(rw)); rw.num_sge=0;   // replenish
            if (ibv_post_recv(c->qp,&rw,&rb)){ LOGE("post_recv replenish"); return -1; }
        } else { got_send = 1; }
    }
    return 0;
}

// Publish this rank's lockstep token and block until the peer's token for the SAME step arrives.
// `val` must carry the step in its high bits so a stale peer value is distinguishable from a fresh one.
// Returns the peer's token, or 0 if aborted/timed out. Called once per MTP step: ~1 wire RTT, not per barrier.
// Deadline: without one, a peer whose main thread hangs (proxy/NIC still alive, so no error CQE ever)
// hangs THIS thread forever, mid-step, silently. TP_WDOG_NS is ~10^5x the expected RTT; past 50 us we
// escalate from tight spin to 1 ms nanosleeps so a slow-but-alive peer costs at most ~1 ms. Abort code 7.
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
        if (dt >= 50000ull) { struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 }; nanosleep(&ts, NULL); }
        else cpu_relax();
    }
}

void net_abort(NetCtx* c){ if (c) tp_set_abort(c, 1); }

void net_shutdown(NetCtx* c) {
    if (!c) return;
    if (c->qp) ibv_destroy_qp(c->qp);
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
