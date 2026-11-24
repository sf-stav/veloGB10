// net_shim/01_helpers_proxy_nway.inc: Shim helpers (clock, abort/idle knobs, slot addressing), the N-way TCP exchange and schedule, and the diagnostic ring dump.
// File-static helpers shared by every later section: this is one translation unit, sections are included in order and must stay in order.
// Section 1 of 4 of net_shim.c: included in order by net_shim.c, one translation unit; text is moved verbatim.
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

/* ISSUES 153 (2026-10-27): liveness-tied bring-up. A cold-cache head ships its world-1 ranks
 * SEQUENTIALLY (~0.10 GB/s, minutes per rank) and opens its per-pair listeners only afterwards,
 * so the old fixed budgets (300 s dial / 600 s accept) expired mid-ship and EVERY node died with
 * "nway connect: Connection refused" on the first cold boot (warm blob cache hid it — the whole
 * sync takes seconds). The dial/accept loops below now retry as long as the head's control
 * stream is alive: the node process registers that fd (net_set_abort_fd), and a MSG_PEEK never
 * consumes protocol bytes, so the stream stays usable by the control protocol. A hard 1 h cap
 * backstops a registered-fd-less (bench) or pathological case. */
static int g_abort_fd = -1;      /* Rust-owned fd of the node's control stream to the head */
void net_set_abort_fd(int fd) { g_abort_fd = fd; }
static const char* g_abort_why = NULL;
static int bringup_aborted(void) {
    if (g_abort_fd < 0) return 0;
    unsigned char b;
    ssize_t r = recv(g_abort_fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
    if (r == 0) { g_abort_why = "head control stream closed"; return 1; }
    if (r < 0 && (errno == ECONNRESET || errno == EPIPE || errno == ETIMEDOUT)) {
        g_abort_why = "head control stream reset"; return 1;
    }
    return 0;
}
/* exposed for the Rust unit test + diagnostics: 1 iff the registered head stream is dead */
int net_abort_check(void) { return bringup_aborted(); }
static uint64_t now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}
#define BRINGUP_CAP_MS 3600000ull   /* hard cap on any bring-up retry loop */
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

/* CPUIDLE (2026-11-06): the proxy's idle tier. The proxy loops spun (plain load + YIELD) on their pinned big
 * core for the whole TP session, so a server with no request in flight still kept one X925 core per rank
 * at full clock around the clock. YIELD is only a hint: on a core without SMT it changes nothing, power
 * included. Now, after --proxy-idle-us of continuous no-work WITH NOTHING OWED (every posted epoch matched;
 * the critical-path wait for a slow peer keeps spinning), the loop waits with WFE: the core clock-gates
 * until the GPU_READY line it watches is written (the local GPU's first post of the next collective) or the
 * kernel's 10 kHz timer event stream ticks (<= 100 us), then polls everything once more. Leaving the tier
 * costs at most one tick, on the first collective after a quiet gap; at the recommended threshold (20 ms) the
 * gaps inside a request (decode steps, prefill layers, host work between rounds) do not reach it. Timing
 * only: no value, order or protocol change. The watched load is a load-exclusive (no store, no RMW: I6
 * holds). Without the event stream (HWCAP_EVTSTRM absent) the tier sleeps 50 us per poll instead.
 * NIGHT1 (2026-11-16): ON by default at 300 us (set from Rust: PROXY_IDLE_US_DEFAULT), so the proxy also parks in the
 * gaps inside a request; --proxy-idle-us 0 = this debt-free idle tier is off (the pre-CPUIDLE spin; --proxy-owed-wait is separate). */
#define TP_PROXY_IDLE_RECOMMENDED_US 300ull     /* the default (NIGHT1) */
static unsigned long long g_opt_proxy_idle_us = 0;
static uint64_t g_proxy_parks = 0;     /* entries into the idle tier (all proxies) */
static uint64_t g_proxy_park_ns = 0;   /* time spent in it */
static int g_proxy_wfe = -1;           /* -1 = unresolved, 1 = WFE + event stream, 0 = 50 us sleeps */
void net_set_proxy_idle(unsigned long long us) { g_opt_proxy_idle_us = us; }
static int g_opt_proxy_owed = 0;       /* --proxy-owed-wait: 0 = spin (default), 1 = wfet (see TP_OWED_SPIN_NS) */
void net_set_proxy_owed(int wfet) { g_opt_proxy_owed = wfet ? 1 : 0; }
uint64_t net_proxy_parks(void) { return __atomic_load_n(&g_proxy_parks, __ATOMIC_RELAXED); }
uint64_t net_proxy_park_ns(void) { return __atomic_load_n(&g_proxy_park_ns, __ATOMIC_RELAXED); }

#define TP_PROXY_DEEP_NS   1000000000ull   /* after 1 s parked: real sleeps instead of WFE */
#define TP_PROXY_SLEEP_NS  250000ull        /* the deep tier's sleep (<= ~0.25 ms on the next request) */

/* --proxy-owed-wait wfet (opt-in; default spin). The tier above parks only with NOTHING owed, so between the
 * collectives of a request, while a posted epoch waits for its match, the proxy still spins its pinned core. With
 * this option and FEAT_WFxT (HWCAP2_WFXT: the X925/A725 of GB10), a proxy that is owed an epoch and has had no work for
 * TP_OWED_SPIN_NS parks in WFET, with the exclusive monitor armed on the line it prefers to be woken by and an absolute
 * deadline TP_OWED_PARK_NS ahead, then runs the loop once more:
 *   - CPU receive (world 2, v1): the PEER_COMMITTED line, written by the NIC (the peer's inline commit);
 *   - GPU receive (v2, any world): the GPU_READY line; with no receive stage, the proxy's next job is the GPU's post.
 * The watch is a wake preference, not the only dependency: everything else the loop services (the CQ drain and so
 * tx_retired, the agree token, the exchange handoff, the abort flags, the watchdog, a GPU post while the loop waits
 * on the NIC) is polled on the following passes; the deadline bounds one WFET, not their service. Timing only: no value,
 * order or protocol change; the load-exclusive is a load (no store, no RMW: I6 holds). The N-way CPU receive (its
 * commits land on per-peer lines) keeps spinning. Both times are untuned.
 * Measured on two DGX Spark (GB10, ConnectX-7 RoCE v2, 8-byte inline RDMA writes into registered host memory, plain
 * and cudaHostAlloc): a NIC write wakes an LDXR-armed WFET-parked X925 core with ping-pong RTT p50 3.46 us vs 3.23-3.34
 * us spinning, p99 3.73-3.78 vs 3.66-3.73; unarmed (plain LDR) 63-65 / 130-133 us, i.e. the timer event stream.
 * A GPU store over NVLink-C2C wakes it the same way (p50 1.50 vs 1.43 us spinning). */
#define TP_OWED_SPIN_NS    5000ull   /* spin this long after the last work before the first owed park */
#define TP_OWED_PARK_NS   10000ull   /* each WFET's counter deadline: bounds one park, not the service of off-line work */

typedef struct { uint64_t idle_ns, idle_since, park_t0, park_start, win_t0; unsigned ticks, wakes; int parked;
                 /* --proxy-owed-wait wfet (aarch64 + FEAT_WFxT only) */
                 uint64_t owed_t0, owed_spin_ticks, owed_park_ticks; int owed_park; } ProxyIdle;

#if defined(__aarch64__)
#ifndef HWCAP2_WFXT
#define HWCAP2_WFXT (1UL << 31)
#endif
static inline uint64_t proxy_cntvct(void) {
    uint64_t t;
    __asm__ __volatile__("isb\n\tmrs %0, cntvct_el0" : "=r"(t) :: "memory");
    return t;
}
static inline uint64_t proxy_cntfrq(void) {
    uint64_t f;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}
/* Park until *p may have reached `seen`, an event arrives, or CNTVCT reaches `deadline`: the kernel's __cmpwait
 * sequence with WFET. SEVL+WFE consumes any pending event, LDXR arms the monitor on *p's line, and WFET waits only
 * while the value is still below `seen`. Returns early and spuriously by design; the caller re-reads.
 * Requires HWCAP2_WFXT. */
static inline void proxy_wait_ge(volatile uint64_t* p, uint64_t seen, uint64_t deadline) {
    uint64_t tmp;
    __asm__ __volatile__("mov x9, %[dl]\n\t"
                         "sevl\n\t"
                         "wfe\n\t"
                         "ldxr %[tmp], %[v]\n\t"
                         "cmp %[tmp], %[seen]\n\t"
                         "b.hs 1f\n\t"
                         ".inst 0xd5031009\n"    /* wfet x9 */
                         "1:"
                         : [tmp] "=&r"(tmp)
                         : [v] "Q"(*p), [seen] "r"(seen), [dl] "r"(deadline)
                         : "x9", "cc", "memory");
    (void)tmp;
}
#endif

// `owed_kind`: what this ctx's owed wait is waiting for. 0 = no owed parking (the N-way CPU receive, whose
// commits land on per-peer lines), 1 = the GPU_READY line (GPU receive), 2 = the PEER_COMMITTED line (world 2, CPU receive).
static void proxy_idle_init(ProxyIdle* s, int owed_kind) {
    memset(s, 0, sizeof(*s));
    s->idle_ns = (uint64_t)g_opt_proxy_idle_us * 1000ull;
    if (g_proxy_wfe < 0) {
#if defined(__aarch64__)
        g_proxy_wfe = (getauxval(AT_HWCAP) & HWCAP_EVTSTRM) ? 1 : 0;
#else
        g_proxy_wfe = 0;
#endif
    }
    if (s->idle_ns) prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);   // this (proxy) thread: 1 us sleep slack
    if (s->idle_ns)
        fprintf(stderr, "[net] proxy idle tier: park after %llu us with nothing owed (%s, then 250 us sleeps after 1 s)%s\n",
                (unsigned long long)g_opt_proxy_idle_us,
                g_proxy_wfe ? "WFE, <= 100 us wake" : "50 us sleeps, no event stream",
                g_opt_proxy_idle_us == TP_PROXY_IDLE_RECOMMENDED_US ? " (--proxy-idle-us default)" : " (--proxy-idle-us)");
    else
        fprintf(stderr, "[net] proxy idle tier: off (--proxy-idle-us 0): the debt-free idle tier is disabled\n");
    int wfxt = 0;
#if defined(__aarch64__)
    wfxt = (getauxval(AT_HWCAP2) & HWCAP2_WFXT) != 0;
    if (g_opt_proxy_owed && owed_kind && wfxt) {
        uint64_t hz = proxy_cntfrq();
        s->owed_spin_ticks = hz * TP_OWED_SPIN_NS / 1000000000ull;
        s->owed_park_ticks = hz * TP_OWED_PARK_NS / 1000000000ull;
        if (!s->owed_park_ticks) s->owed_park_ticks = 1;
        s->owed_park = 1;
    }
#endif
    if (s->owed_park)
        fprintf(stderr, "[net] proxy owed wait: spin %llu us, then WFET parks of <= %llu us on the %s line (--proxy-owed-wait wfet)\n",
                (unsigned long long)(TP_OWED_SPIN_NS / 1000ull), (unsigned long long)(TP_OWED_PARK_NS / 1000ull),
                owed_kind == 2 ? "peer-commit (NIC-written)" : "GPU_READY");
    else
        fprintf(stderr, "[net] proxy owed wait: spin (%s)\n",
                !g_opt_proxy_owed ? "--proxy-owed-wait spin, the default" :
                !owed_kind ? "N-way CPU receive: per-peer commit lines" : !wfxt ? "no FEAT_WFxT" : "?");
}

#if defined(__aarch64__)
// The owed wait of one loop iteration with nothing done: spin for TP_OWED_SPIN_NS after the last work, then park on `w`
// until it may have reached `seen`, an event arrives, or the deadline. The loop re-reads everything after it returns.
static inline void proxy_owed_park(ProxyIdle* s, volatile uint64_t* w, uint64_t seen) {
    uint64_t t = proxy_cntvct();
    if (!s->owed_t0) s->owed_t0 = t;
    if (t - s->owed_t0 < s->owed_spin_ticks) { cpu_relax(); return; }
    proxy_wait_ge(w, seen, t + s->owed_park_ticks);
}
#endif

// One loop iteration's ending. `did_work` resets the tier; `owed` (posted epochs the peers have not
// matched) keeps the critical-path wait: a spin, or with --proxy-owed-wait wfet the owed park on `owed_watch` until it
// reaches `owed_seen` (NULL = spin); `watch`/`seen`: the GPU_READY line and the epoch the loop waits
// for — a value already >= seen means work is pending, so never sleep on it.
static inline void proxy_idle_step(ProxyIdle* s, int did_work, int owed, volatile uint64_t* watch,
                                   uint64_t seen, volatile uint64_t* owed_watch, uint64_t owed_seen) {
    if (did_work || owed) {
        if (s->parked) {
            __atomic_fetch_add(&g_proxy_park_ns, now_ns() - s->park_t0, __ATOMIC_RELAXED);
            s->parked = 0;
        }
        s->idle_since = 0; s->ticks = 0;
        if (did_work) { s->owed_t0 = 0; return; }
#if defined(__aarch64__)
        if (s->owed_park && owed_watch) { proxy_owed_park(s, owed_watch, owed_seen); return; }
#endif
        (void)owed_watch; (void)owed_seen;
        cpu_relax();
        return;
    }
    s->owed_t0 = 0;
    if (!s->idle_ns) { cpu_relax(); return; }
    if (!s->parked) {
        // Read the clock only every 1024 idle spins: the first idle iterations after work stay a plain
        // load + yield, exactly as before.
        if (((++s->ticks) & 0x3FF) == 0) {
            uint64_t t = now_ns();
            if (!s->idle_since) s->idle_since = t;
            else if (t - s->idle_since >= s->idle_ns) {
                s->parked = 1; s->park_t0 = t; s->park_start = t; s->win_t0 = t; s->wakes = 0;
                __atomic_fetch_add(&g_proxy_parks, 1, __ATOMIC_RELAXED);
            }
        }
        cpu_relax();
        return;
    }
    // Two depths. The first second of quiet waits with WFE: the core clock-gates (no OS sleep, so /proc
    // still counts it as running) and wakes the instant the GPU posts — right for the long gaps INSIDE a
    // request (a big model's prefill layer). After a second the server is idle between requests: real
    // sleeps let the core reach its idle states, at a cost of <= ~0.25 ms on the next request's first
    // collective.
    int deep = s->park_t0 - s->park_start >= TP_PROXY_DEEP_NS;
#if defined(__aarch64__)
    if (g_proxy_wfe && !deep) {
        // The kernel's __cmpwait sequence: SEVL+WFE first consumes any event already pending (an exception
        // return, the tick that just woke us), so the second WFE really waits for the next one — the monitor
        // armed by LDXR (a write to the GPU_READY line) or the 10 kHz event stream. (The first version armed
        // the monitor without clearing a pending event; on GB10 its WFE returned at once and the "parked"
        // proxy still spun a full core.)
        uint64_t v;
        __asm__ __volatile__("sevl\n\twfe" ::: "memory");
        __asm__ __volatile__("ldxr %0, [%1]" : "=r"(v) : "r"(watch) : "memory");   // arm the monitor
        if (v < seen) __asm__ __volatile__("wfe" ::: "memory");
    } else
#endif
    {
        (void)watch; (void)seen;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = deep ? (long)TP_PROXY_SLEEP_NS : 50000L };
        nanosleep(&ts, NULL);
    }
    // Live accounting (the metric moves during a long park, not only when it ends) + a guard: 4096 wakes
    // in under 100 ms (> 40 kHz, 4x the event stream's rate) means WFE is not waiting on this machine —
    // fall back to 50 us sleeps for every proxy, once, and say so.
    uint64_t t = now_ns();
    __atomic_fetch_add(&g_proxy_park_ns, t - s->park_t0, __ATOMIC_RELAXED);
    s->park_t0 = t;
    if (((++s->wakes) & 0xFFF) == 0) {
        if (g_proxy_wfe && !deep && t - s->win_t0 < 100000000ull) {
            g_proxy_wfe = 0;
            fprintf(stderr, "[net] proxy idle tier: WFE returned %u times in %.1f ms (not waiting on this "
                            "machine) — using 50 us sleeps instead\n", 4096u, (t - s->win_t0) / 1e6);
        }
        s->win_t0 = t;
    }
}

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
        struct sockaddr_in a; memset(&a,0,sizeof(a));
        a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,peer_ip,&a.sin_addr);
        /* ISSUES 153: liveness-tied dial (was 400 x 50 ms = 20 s — a cold TP2 ship is ~12 min).
         * Fresh socket per attempt (a socket that has seen a failed connect() must not be
         * re-connected), abort the instant the head's control stream dies, 1 h hard cap. */
        sock = -1;
        for (uint64_t t0 = now_ms(), last_note = t0;;) {
            if (bringup_aborted()) { LOGE("connect %s:%d: giving up — %s", peer_ip, port, g_abort_why); return -1; }
            uint64_t el = now_ms() - t0;
            if (el > BRINGUP_CAP_MS) { LOGE("connect %s:%d: no listener within %llu s (cap)", peer_ip, port, (unsigned long long)(el/1000)); return -1; }
            if (el - last_note >= 60000) { LOGE("connect %s:%d: still retrying (%llu s — cold-cache head still shipping?)", peer_ip, port, (unsigned long long)(el/1000)); last_note += 60000; }
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) { LOGE("socket"); return -1; }
            if (connect(sock,(struct sockaddr*)&a,sizeof(a))==0) break;
            close(sock); usleep(50000);
        }
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
            /* ISSUES 153: sliced accept — the old single 600 s poll expired on cold boots where a
             * lower rank waits (world-1) x ship-time for the LAST rank to reach its dial. */
            int pr = 0;
            uint64_t t0 = now_ms(), last_note = t0;
            for (;;) {
                if (bringup_aborted()) { LOGE("nway accept: giving up on rank %d — %s", peer_rank, g_abort_why); close(ls); return -1; }
                do { pr = poll(&pfd, 1, 1000); } while (pr < 0 && errno == EINTR);
                if (pr > 0) break;
                if (pr < 0) { LOGE("nway accept poll: %s", strerror(errno)); close(ls); return -1; }
                uint64_t el = now_ms() - t0;
                if (el > BRINGUP_CAP_MS) {
                    LOGE("nway accept: rank %d never dialled rank %d on port %d within %llu s (timeout) — the peer did not "
                         "reach its link bring-up; check its log", peer_rank, my_rank, port, (unsigned long long)(el/1000));
                    close(ls);
                    return -1;
                }
                if (el - last_note >= 60000) { LOGE("nway accept: rank %d still waiting for rank %d's dial (%llu s — cold-cache ship?)", my_rank, peer_rank, (unsigned long long)(el/1000)); last_note += 60000; }
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
        // re-connected on Linux. ISSUES 153: liveness-tied dial — the old 6000 x 50 ms = 300 s
        // budget expired while a cold head was still shipping (sequential, minutes per rank);
        // now we retry while the head's control stream is alive, with a 1 h hard cap.
        int ok = -1;
        for (uint64_t t0 = now_ms(), last_note = t0;;) {
            if (bringup_aborted()) { LOGE("nway connect %s:%d: giving up — %s", peer_ip, port, g_abort_why); return -1; }
            uint64_t el = now_ms() - t0;
            if (el > BRINGUP_CAP_MS) { LOGE("nway connect %s:%d: no listener within %llu s (cap)", peer_ip, port, (unsigned long long)(el/1000)); return -1; }
            if (el - last_note >= 60000) { LOGE("nway connect %s:%d: still retrying (%llu s — cold-cache head still shipping?)", peer_ip, port, (unsigned long long)(el/1000)); last_note += 60000; }
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
// round(e)'s own R-slot ring at per-round slot (e / rounds) % R (Phase 2: bijective across all R slots
// for every round count including world=8 / rounds=3, reuse period R * rounds epochs). Both sides compute
// round(e) and slot(e) identically from the (SPMD) epoch. world==2 never calls these.
static inline uint64_t nway_recv_slot_off(NetCtx* c, uint64_t e) {
    uint64_t r = (uint64_t)c->rounds;
    uint64_t round = e % r;
    uint64_t slot = (e / r) & (TP_RING_SLOTS - 1);
    return c->nway_recv_off + (round * (uint64_t)TP_RING_SLOTS + slot) * c->slot_stride;
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

