// net_shim/04_startup_audit_channel.inc: Startup / audit channel: net_exchange (the retained WITH_IMM all-reduce exchange used by --net-test and as an out-of-band channel) and the abort entry.
// Off the hot path; shares the NetCtx state declared in the net_shim.c preamble.
// Section 4 of 4 of net_shim.c: included in order by net_shim.c, one translation unit; text is moved verbatim.
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
// TP-D boot variant: `net_agree` with an explicit deadline. The BOOT agree is the FIRST
// cross-rank rendezvous of a session, right after two INDEPENDENT model loads (raw pack on
// the head, converted cache on a warm node): 50-400 s of LEGITIMATE skew the 10 s per-step
// watchdog cannot cover (2026-10-30 Lygodactylus 4bpw + 2.05bpw TP boot aborts, code 7
// "peer token did not arrive within 10s"). Same rationale as arming the doorbell watchdog
// only AFTER the first rendezvous. A peer that DIED (vs merely slow) is still caught by the
// out-of-band liveness probe / abort flag, exactly as in net_exchange's probe-tolerant wait.
uint64_t net_agree_dl(NetCtx* c, uint64_t val, uint64_t step_mask, uint64_t step_val,
                      uint64_t deadline_ns) {
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
        if (dt >= deadline_ns) {
            LOGE("net_agree: peer token for step %llu did not arrive within %llus — aborting",
                 (unsigned long long)step_val, (unsigned long long)(deadline_ns / 1000000000ull));
            tp_set_abort(c, 7);
            return 0;
        }
        tp_host_backoff(dt);
    }
}

// The per-step agree: the 10 s TP_WDOG_NS deadline (a mid-run hang must tear the link down
// fast; legitimate mid-run unmatched skew is ms-scale — see the comment above).
uint64_t net_agree(NetCtx* c, uint64_t val, uint64_t step_mask, uint64_t step_val) {
    return net_agree_dl(c, val, step_mask, step_val, TP_WDOG_NS);
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

    const int use_inl = ((unsigned)nbytes <= c->max_inline);
    struct ibv_sge sge; memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)ctrl_send_slot(c); sge.length = (uint32_t)nbytes; sge.lkey = use_inl ? 0 : c->mr->lkey;
    struct ibv_send_wr wr, *bad; memset(&wr, 0, sizeof(wr));
    wr.wr_id = TP_XCHG_WR_ID;
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED | (use_inl ? IBV_SEND_INLINE : 0);
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

// Phase 2B: parallel head <-> all-peers control exchange (1 RTT instead of world-1 serial RTTs).
// Rank 0 posts the staged ctrl_send_slot payload to all peers 1..world upfront (inline when <= max_inline),
// waits for all world-1 send CQEs and all world-1 peer tail tags concurrently, then copies each into
// ctrl_last_slot(r) and zeroes the consumed ring slots. Wire-identical to world-1 calls of net_exchange_one
// on the peers (nodes 1..world continue calling net_exchange_one(ctx, 0, nbytes) unchanged).
int net_exchange_all_peers(NetCtx* c, int nbytes, uint64_t deadline_ns) {
    if (c->world <= 2 || c->rank != 0) { LOGE("net_exchange_all_peers: requires world>2, rank==0"); return -1; }
    if (nbytes < 16 || (size_t)nbytes > c->slot_stride - TP_TAIL_BYTES) return -1;
    const int n_peers = c->world - 1;
    const int use_inl = ((unsigned)nbytes <= c->max_inline && nbytes <= 512);
    if (!use_inl) {
        int any_err = 0;
        for (int r = 1; r < c->world; r++) {
            int rc = exchange_one_impl(c, r, nbytes, deadline_ns);
            if (rc != 0 && any_err == 0) any_err = rc;
        }
        return any_err;
    }

    uint64_t gens[TP_NWAY_MAX_WORLD];
    unsigned char ibuf[512];
    memcpy(ibuf, ctrl_send_slot(c), (size_t)(nbytes - TP_TAIL_BYTES));
    const uint64_t send_seq_before = __atomic_load_n(&c->xchg_send_seq, __ATOMIC_ACQUIRE);

    int posted = 0;
    for (int r = 1; r < c->world; r++) {
        if (!c->peers[r].valid) { LOGE("exchange_all: peer %d QP not valid", r); return -1; }
        const uint64_t g = ++c->xchg_gen[r];
        gens[r] = g;
        const uint64_t tag = (g << 8) | (uint64_t)c->rank;
        memcpy(ibuf + nbytes - TP_TAIL_BYTES, &tag, TP_TAIL_BYTES);
        struct ibv_sge sge = { .addr = (uint64_t)(uintptr_t)ibuf, .length = (uint32_t)nbytes, .lkey = 0 };
        struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
        wr.wr_id = TP_XCHG_WR_ID;
        wr.sg_list = &sge; wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        wr.wr.rdma.remote_addr = ctrl_recv_raddr(c, r, g);
        wr.wr.rdma.rkey = peer_remote_rkey(c, r);
        if (ibv_post_send(peer_qp(c, r), &wr, &bad)) { LOGE("exchange_all post_send(%d)", r); return -1; }
        posted++;
    }

    // Wait for all `posted` send CQEs in parallel.
    int got_sends = 0;
    const uint64_t s_t0 = now_ns();
    uint64_t s_last_probe = s_t0;
    for (;;) {
        if (!__atomic_load_n(&c->proxy_running, __ATOMIC_ACQUIRE)) {
            struct ibv_wc wc[16]; int n = ibv_poll_cq(c->cq_send, 16, wc);
            if (n < 0) { LOGE("exchange_all poll_cq"); return -1; }
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) { LOGE("exchange_all send wc %d", wc[i].status); return -3; }
                if (wc[i].wr_id == TP_XCHG_WR_ID) got_sends++;
            }
        } else {
            uint64_t cur = __atomic_load_n(&c->xchg_send_seq, __ATOMIC_ACQUIRE);
            if (cur >= send_seq_before + (uint64_t)posted) got_sends = posted;
        }
        if (got_sends >= posted) break;
        if (c->aborted || *flagp(c, TP_F_ABORT)) return -2;
        uint64_t s_dt = now_ns() - s_t0;
        if (deadline_ns && s_dt >= deadline_ns) { tp_set_abort(c, 12); return -4; }
        if (s_dt >= TP_LIVE_PROBE_NS && now_ns() - s_last_probe >= TP_LIVE_PROBE_NS) {
            s_last_probe = now_ns();
            for (int r = 1; r < c->world; r++) {
                if (!peer_alive_at(c, r)) {
                    LOGE("exchange_all: peer %d unreachable (send CQE) — DEAD (code 10)", r);
                    tp_set_abort(c, 10);
                    return -3;
                }
            }
        }
        cpu_relax();
    }

    // Wait for all world-1 peer tail tags in parallel.
    unsigned done_mask = 0u;
    const unsigned full_mask = ((1u << c->world) - 1u) & ~1u;
    const uint64_t t0 = now_ns();
    uint64_t last_probe = t0;
    for (;;) {
        for (int r = 1; r < c->world; r++) {
            if (done_mask & (1u << r)) continue;
            volatile uint64_t* tailp = (volatile uint64_t*)(ctrl_recv_slot(c, r, gens[r]) + nbytes - TP_TAIL_BYTES);
            if (*tailp == ((gens[r] << 8) | (uint64_t)r)) done_mask |= (1u << r);
        }
        if (done_mask == full_mask) break;
        if (c->aborted || *flagp(c, TP_F_ABORT)) return -2;
        uint64_t dt = now_ns() - t0;
        if (deadline_ns && dt >= deadline_ns) { tp_set_abort(c, 12); return -4; }
        if (dt >= TP_LIVE_PROBE_NS && now_ns() - last_probe >= TP_LIVE_PROBE_NS) {
            last_probe = now_ns();
            for (int r = 1; r < c->world; r++) {
                if (done_mask & (1u << r)) continue;
                if (!peer_alive_at(c, r)) {
                    LOGE("exchange_all: peer %d unreachable (recv) — DEAD (code 10)", r);
                    tp_set_abort(c, 10);
                    return -3;
                }
            }
        }
        tp_host_backoff(dt);
    }

    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    for (int r = 1; r < c->world; r++) {
        char* rslot = ctrl_recv_slot(c, r, gens[r]);
        memcpy(ctrl_last_slot(c, r), (const void*)rslot, (size_t)nbytes);
        memset((void*)rslot, 0, (size_t)nbytes);
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    (void)n_peers;
    return 0;
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
    if (c->qp_mask_dev) cudaFree(c->qp_mask_dev);
    if (c->cpu_ts) free(c->cpu_ts);
    if (c->cq_send) ibv_destroy_cq(c->cq_send);
    if (c->cq_startup) ibv_destroy_cq(c->cq_startup);
    if (c->td_pd) ibv_dealloc_pd(c->td_pd);
    if (c->td) ibv_dealloc_td(c->td);
    if (c->pd) ibv_dealloc_pd(c->pd);
    if (c->ctx) ibv_close_device(c->ctx);
    free(c);
}
