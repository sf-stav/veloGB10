// net_shim/03_hot_path.inc: The doorbell all-reduce hot path: post_range / post_range_oneshot and the N-way receive-slot handling.
// HOT PATH: invariants I1-I9 live in native/tp_doorbell.h; the CPU bounces visibility (peer_committed -> fence -> RELEASE cpu_done) because GB10 cannot flush remote writes to the GPU.
// Section 3 of 4 of net_shim.c: included in order by net_shim.c, one translation unit; text is moved verbatim.
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
    ProxyIdle idle; proxy_idle_init(&idle);   // CPUIDLE: park after --proxy-idle-us with nothing owed

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

        // plain load + yield (I6) while work flows or a peer owes us an epoch; WFE once quiet (CPUIDLE)
        if (did_work) proxy_idle_step(&idle, 1, 0, NULL, 0);
        else {
            uint64_t rxd = c->recv_gpu ? *flagp(c, TP_F_RX_DONE) : 0;   // the watchdog's debt, recomputed
            uint64_t m = c->recv_gpu ? (next_release > rxd + 1 ? next_release : rxd + 1)
                                     : (next_release > rxd ? next_release : rxd);
            proxy_idle_step(&idle, 0, next_to_post > m, flagp(c, TP_F_GPU_READY), next_to_post);
        }
    }
    if (idle.parked) __atomic_fetch_add(&g_proxy_park_ns, now_ns() - idle.park_t0, __ATOMIC_RELAXED);
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
    ProxyIdle idle; proxy_idle_init(&idle);   // CPUIDLE: park after --proxy-idle-us with nothing owed

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
        //    NAIVE lever 1b: under the v2 receive mode (net_set_recv_mode gpu=1) this stage is SKIPPED like the
        //    world-2 and tree receives below: the K2 (tp_wait_add_4way_f32g) validates every peer's tail itself and
        //    publishes rx_done; the watchdog above already keys on rx_done in that mode.
        if (c->oneshot_uniform && !c->recv_gpu) {
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

        // CPUIDLE: same tier as world==2 (the global debt, as the watchdog computes it)
        if (did_work) proxy_idle_step(&idle, 1, 0, NULL, 0);
        else {
            uint64_t rxd = c->recv_gpu ? *flagp(c, TP_F_RX_DONE) : 0;
            uint64_t m = c->recv_gpu ? (next_release > rxd + 1 ? next_release : rxd + 1)
                                     : (next_release > rxd ? next_release : rxd);
            proxy_idle_step(&idle, 0, next_to_post > m, flagp(c, TP_F_GPU_READY), next_to_post);
        }
    }
    if (idle.parked) __atomic_fetch_add(&g_proxy_park_ns, now_ns() - idle.park_t0, __ATOMIC_RELAXED);

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

