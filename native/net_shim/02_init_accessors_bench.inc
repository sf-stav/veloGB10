// net_shim/02_init_accessors_bench.inc: net_init and its transport bring-up, the flat C ABI accessors, and the bench hooks.
// Same translation unit as the preamble in net_shim.c; the exported C ABI is mirrored by src/net.rs, do not reorder or rename.
// Section 2 of 4 of net_shim.c: included in order by net_shim.c, one translation unit; text is moved verbatim.
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
    if (world < 2 || world > TP_NWAY_MAX_WORLD) {
        LOGE("world %d must be in [2,%d]", world, TP_NWAY_MAX_WORLD);
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

// PR #15 (review R2, S-B9-REL-PRPORT3): the GID index is resolved ONCE at boot on the Rust side
// (tp::resolve_gid_idx: explicit --gid-idx wins, else the sysfs RoCE-v2 scan — which checks the
// GID *type*, not just the IPv4-mapped prefix — else the static default) and passed to every
// shim bring-up. The shim binds EXACTLY that index: if it does not query back a valid non-zero
// GID, bring-up fails LOUDLY (the callers log `query_gid` and bail). The earlier fallback scan
// here matched only the IPv4-mapped bytes without checking the GID type, so on an all-zero or
// stale index it could silently re-bind a RoCE v1 GID — a wire mismatch with the peer instead
// of a loud failure, and a contradiction of the "explicit --gid-idx skips the scan" help text.
static int get_valid_roce_gid(struct ibv_context *ctx, int port_num, int *gid_idx, union ibv_gid *out_gid) {
    if (ibv_query_gid(ctx, port_num, *gid_idx, out_gid) == 0) {
        static const unsigned char zero_gid[16] = {0};
        if (memcmp(out_gid, zero_gid, 16) != 0) {
            return 0; // the Rust-resolved index is live — bind it
        }
        LOGE("query_gid: GID at index %d reads back ALL-ZERO (stale index; resolve_gid_idx picked it — re-run bring-up)", *gid_idx);
        return -1;
    }
    LOGE("query_gid: index %d not queryable on port %d (resolve_gid_idx picked it — re-run bring-up)", *gid_idx, port_num);
    return -1;
}

// The world==2 single-QP bring-up (the P3 shape; still the world==2 fast path). The QP/ring setup
// below is unchanged; the cudaHostAlloc retry + named-CUDA-error logging in the allocator are
// deliberate hardening (S-B9-REL-TP2RACE REL-NGRAM transient bring-up failure; PR #15 error text)
// on top of it — not "byte-for-byte pre-P3" anymore, by design.
// REL-NGRAM background: a cudaHostAlloc can fail transiently right after the model + primary rail
// came up (unified-memory squeeze) — the head never reached its TCP listener and the node logged
// `connect :29600 refused`. Retry a few times before giving up; persistent failure stays loud.
static void* host_alloc_with_retry(size_t bytes) {
    void* p = NULL;
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < 5; i++) {
        err = cudaHostAlloc(&p, bytes, cudaHostAllocMapped|cudaHostAllocPortable);
        if (err == cudaSuccess) {
            if (i) LOGE("cudaHostAlloc(%zu) succeeded on retry %d", bytes, i);
            return p;
        }
        // PR #15: name the CUDA error — a bare "cudaHostAlloc(N)" line hid the actual cause.
        LOGE("cudaHostAlloc(%zu) failed (attempt %d/5): %s (code %d) — retrying in 1 s",
             bytes, i + 1, cudaGetErrorString(err), (int)err);
        struct timespec d = { .tv_sec = 1, .tv_nsec = 0 };
        nanosleep(&d, NULL);
    }
    LOGE("cudaHostAlloc(%zu) failed after 5 attempts: %s (code %d)",
         bytes, cudaGetErrorString(err), (int)err);
    return NULL;
}

static NetCtx* net_init_world2(int rank, const char* peer_ip, int tcp_port, const char* dev_name,
                               int gid_idx, int fp32_capacity_bytes, int payload_bytes) {
    // PR #15: pin the calling thread to device 0 before any cudaHostAlloc — the shim can be
    // brought up from a thread that never touched CUDA, and an unpinned context makes the
    // mapped-alloc fail (or bind to the wrong device on multi-GPU boxes).
    cudaSetDevice(0);
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
    // Phase 1C: request 512 B inline (BlueFlame zero-DMA-read on tiny head/Markov/agree frames), fall back to 256/64.
    const unsigned inl_try[3] = { 512u, 256u, 64u };
    for (int t = 0; t < 3 && !c->qp; t++) {
        qia.cap.max_inline_data = inl_try[t];
        c->qp = ibv_create_qp(c->pd, &qia);
        if (c->qp) c->max_inline = qia.cap.max_inline_data ? qia.cap.max_inline_data : inl_try[t];
    }
    if (!c->qp){ LOGE("create_qp"); return NULL; }

    struct ibv_qp_attr a; memset(&a,0,sizeof(a));
    a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=c->port_num;
    a.qp_access_flags=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(c->qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)){ LOGE("INIT"); return NULL; }

    union ibv_gid mygid;
    if (get_valid_roce_gid(c->ctx, c->port_num, &gid_idx, &mygid) != 0){ LOGE("query_gid"); return NULL; }
    c->gid_idx = gid_idx;   // PR #15: record the index actually bound (the Rust-resolved one, unchanged here)
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
    cudaSetDevice(0);   // PR #15: same device pin as world2 (see net_init_world2)
    NetCtx* c = (NetCtx*)calloc(1, sizeof(NetCtx));
    c->port_num = 1; c->gid_idx = gid_idx; c->rank = rank; c->rng = 0x9E3779B9u ^ (unsigned)rank;
    c->payload_bytes = (unsigned)payload_bytes;
    const int pow2 = (world & (world - 1)) == 0;
    c->world = world; c->rounds = 0;
    if (pow2) { while ((1 << c->rounds) < world) c->rounds++; }
    // P3-1 / Velo-Fabric arbitrary-W (TP3/TP6): non-power-of-two worlds have no XOR doubling tree
    // (rounds = 0) and always run in uniform all-peers one-shot / all-to-all mode.
    c->oneshot_uniform = ((world >= 3 && g_oneshot_uniform_once) || (!pow2 && world > 2)) ? 1 : 0;
    g_oneshot_uniform_once = 0;                                                            // consumed: never leaks to the next init
    c->oneshot = ((world == 4 && g_opt_oneshot) || c->oneshot_uniform) ? 1u : 0u;
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
    if (get_valid_roce_gid(c->ctx, c->port_num, &gid_idx, &mygid) != 0){ LOGE("query_gid"); return NULL; }
    c->gid_idx = gid_idx;   // PR #15: record the index actually bound (the Rust-resolved one, unchanged here)

    // Phase 1C: for oneshot_uniform (100% proxy-owned, no control-plane traffic on its QPs),
    // allocate a lock-free IBV_THREAD_DOMAIN parent domain so ibv_wr_* posts skip per-QP spinlocks.
    if (c->oneshot_uniform) {
        struct ibv_td_init_attr tda; memset(&tda, 0, sizeof(tda));
        c->td = ibv_alloc_td(c->ctx, &tda);
        if (c->td) {
            struct ibv_parent_domain_init_attr pda; memset(&pda, 0, sizeof(pda));
            pda.pd = c->pd; pda.td = c->td;
            c->td_pd = ibv_alloc_parent_domain(c->ctx, &pda);
            if (!c->td_pd) { ibv_dealloc_td(c->td); c->td = NULL; }
        }
    }
    struct ibv_pd* qp_pd = c->td_pd ? c->td_pd : c->pd;

    // One QP per peer. The lower rank listens, the higher connects (deterministic, matches tcp_exchange).
    for (int p = 0; p < world; p++) {
        if (p == rank) continue;
        PeerLink* pl = &c->peers[p];
        const char* pip = peer_ips[p];
        if (!pip || !pip[0]) { LOGE("rank %d: peer_ips[%d] empty", rank, p); return NULL; }

        const unsigned inl_try[3] = { 512u, 256u, 64u };
        if (c->oneshot_uniform) {
            for (int t = 0; t < 3 && !pl->qp; t++) {
                struct ibv_qp_init_attr_ex qex; memset(&qex, 0, sizeof(qex));
                qex.send_cq = c->cq_send; qex.recv_cq = c->cq_startup; qex.qp_type = IBV_QPT_RC;
                qex.cap.max_send_wr = 256; qex.cap.max_recv_wr = 256;
                qex.cap.max_send_sge = 1; qex.cap.max_recv_sge = 1;
                qex.cap.max_inline_data = inl_try[t];
                qex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
                qex.pd = qp_pd;
                qex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
                pl->qp = ibv_create_qp_ex(c->ctx, &qex);
                if (pl->qp) {
                    pl->qpx = ibv_qp_to_qp_ex(pl->qp);
                    unsigned mi = qex.cap.max_inline_data ? qex.cap.max_inline_data : inl_try[t];
                    if (c->max_inline == 0 || mi < c->max_inline) c->max_inline = mi;
                }
            }
        }
        if (!pl->qp) {
            struct ibv_qp_init_attr qia; memset(&qia,0,sizeof(qia));
            qia.send_cq=c->cq_send; qia.recv_cq=c->cq_startup; qia.qp_type=IBV_QPT_RC;
            qia.cap.max_send_wr=256; qia.cap.max_recv_wr=256; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1;
            for (int t = 0; t < 3 && !pl->qp; t++) {
                qia.cap.max_inline_data = inl_try[t];
                pl->qp = ibv_create_qp(c->pd, &qia);
                if (pl->qp) {
                    unsigned mi = qia.cap.max_inline_data ? qia.cap.max_inline_data : inl_try[t];
                    if (c->max_inline == 0 || mi < c->max_inline) c->max_inline = mi;
                }
            }
        }
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

