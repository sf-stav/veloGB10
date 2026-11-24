// net_shim/05_bulk_rdma.inc: Velo-Fabric bulk RDMA blob transfer engine for rank shipping.
// Uses plain host-allocated memory (posix_memalign + ibv_reg_mr — zero CUDA dependency) with a
// 4-slot x 4 MiB credit-paced RDMA_WRITE_WITH_IMM ring over RC QP (RoCEv2).
// Handshake metadata (BulkEndpoint, 40 B) rides the existing TCP cluster control stream (no extra
// TCP ports). Zero changes to the on-disk blob cache format.

#define BULK_MAX_SLOTS      8
#define BULK_TIMEOUT_MS     60000ull   /* 60 s per-chunk stall bound so a dead peer never wedges */
#define BULK_IMM_SLOT_SHIFT 28
#define BULK_IMM_LEN_MASK   0x0FFFFFFFu

typedef struct {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint32_t chunk_bytes;
    uint64_t addr;
    uint8_t  gid[16];
} BulkEndpoint;

typedef struct {
    struct ibv_context* ctx;
    struct ibv_pd*      pd;
    struct ibv_cq*      cq_send;
    struct ibv_cq*      cq_recv;
    struct ibv_qp*      qp;
    struct ibv_mr*      mr;
    uint8_t*            buf;
    size_t              buf_bytes;
    uint32_t            chunk_bytes;
    uint32_t            slots;
    int                 port_num;
    int                 gid_idx;
    uint32_t            local_psn;
    uint64_t            remote_addr;
    uint32_t            remote_rkey;
    // Sender pacing state
    uint32_t            credits;
    uint32_t            send_seq;
    uint32_t            inflight_send;
    // Receiver credit-send pacing state
    uint32_t            inflight_ack;
} BulkCtx;

static int bulk_post_empty_recv(BulkCtx* b) {
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0xB01Cu;
    wr.num_sge = 0;
    return ibv_post_recv(b->qp, &wr, &bad);
}

static int bulk_poll_cq_wait(struct ibv_cq* cq, struct ibv_wc* wc, uint64_t timeout_ms) {
    uint64_t t0 = now_ms();
    unsigned spins = 0;
    for (;;) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n < 0) return -1;
        if (n > 0) {
            if (wc->status != IBV_WC_SUCCESS) {
                LOGE("bulk CQE error status=%d (%s) opcode=%d", wc->status,
                     ibv_wc_status_str(wc->status), wc->opcode);
                return -2;
            }
            return 1;
        }
        if (++spins < 2000u) {
            cpu_relax();
        } else {
            if (now_ms() - t0 > timeout_ms) {
                LOGE("bulk CQ timeout after %llu ms", (unsigned long long)timeout_ms);
                return -3;
            }
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 50000 }; /* 50 us */
            nanosleep(&ts, NULL);
        }
    }
}

void bulk_close(BulkCtx* b) {
    if (!b) return;
    if (b->inflight_ack > 0 && b->cq_send) {
        struct ibv_wc wc;
        uint64_t t0 = now_ms();
        while (b->inflight_ack > 0 && now_ms() - t0 < 5) {
            int n = ibv_poll_cq(b->cq_send, 1, &wc);
            if (n > 0) b->inflight_ack--;
            else if (n < 0) break;
            else cpu_relax();
        }
    }
    if (b->qp)      ibv_destroy_qp(b->qp);
    if (b->mr)      ibv_dereg_mr(b->mr);
    if (b->cq_send) ibv_destroy_cq(b->cq_send);
    if (b->cq_recv) ibv_destroy_cq(b->cq_recv);
    if (b->pd)      ibv_dealloc_pd(b->pd);
    if (b->ctx)     ibv_close_device(b->ctx);
    if (b->buf)     free(b->buf);
    free(b);
}

// Open a bulk RDMA endpoint on `dev_name` / `gid_idx`, allocate `slots * chunk_bytes` 4K-aligned
// registered memory, move QP to INIT, pre-post recv WQEs, and fill `out_ep`.
BulkCtx* bulk_open(const char* dev_name, int gid_idx, uint32_t chunk_bytes, uint32_t slots,
                   BulkEndpoint* out_ep) {
    if (!dev_name || !dev_name[0] || !out_ep) return NULL;
    if (slots == 0 || slots > BULK_MAX_SLOTS) return NULL;
    if (chunk_bytes == 0 || chunk_bytes > BULK_IMM_LEN_MASK) return NULL;

    int ndev = 0;
    struct ibv_device** devs = ibv_get_device_list(&ndev);
    if (!devs || ndev <= 0) return NULL;
    struct ibv_device* dev = NULL;
    for (int i = 0; i < ndev; i++) {
        if (!strcmp(ibv_get_device_name(devs[i]), dev_name)) { dev = devs[i]; break; }
    }
    struct ibv_context* ictx = dev ? ibv_open_device(dev) : NULL;
    ibv_free_device_list(devs);
    if (!ictx) { LOGE("bulk_open: device %s not found/openable", dev_name); return NULL; }

    BulkCtx* b = (BulkCtx*)calloc(1, sizeof(BulkCtx));
    if (!b) { ibv_close_device(ictx); return NULL; }
    b->ctx = ictx;
    b->port_num = 1;
    b->gid_idx = gid_idx;
    b->chunk_bytes = chunk_bytes;
    b->slots = slots;
    b->credits = slots;

    union ibv_gid mygid;
    if (get_valid_roce_gid(b->ctx, b->port_num, &b->gid_idx, &mygid) != 0) {
        LOGE("bulk_open: query_gid failed on %s idx %d", dev_name, gid_idx);
        bulk_close(b); return NULL;
    }

    b->pd = ibv_alloc_pd(b->ctx);
    b->cq_send = ibv_create_cq(b->ctx, 64, NULL, NULL, 0);
    b->cq_recv = ibv_create_cq(b->ctx, 64, NULL, NULL, 0);
    if (!b->pd || !b->cq_send || !b->cq_recv) { bulk_close(b); return NULL; }

    b->buf_bytes = (size_t)slots * (size_t)chunk_bytes;
    if (posix_memalign((void**)&b->buf, 4096, b->buf_bytes) != 0 || !b->buf) {
        LOGE("bulk_open: posix_memalign(%zu) failed", b->buf_bytes);
        bulk_close(b); return NULL;
    }
    memset(b->buf, 0, b->buf_bytes);

    b->mr = ibv_reg_mr(b->pd, b->buf, b->buf_bytes,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!b->mr) {
        LOGE("bulk_open: ibv_reg_mr(%zu) failed (%s)", b->buf_bytes, strerror(errno));
        bulk_close(b); return NULL;
    }

    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = b->cq_send;
    qia.recv_cq = b->cq_recv;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 32;
    qia.cap.max_recv_wr = 32;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 64;
    b->qp = ibv_create_qp(b->pd, &qia);
    if (!b->qp) { LOGE("bulk_open: create_qp failed"); bulk_close(b); return NULL; }

    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = b->port_num;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(b->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        LOGE("bulk_open: modify_qp INIT failed");
        bulk_close(b); return NULL;
    }

    // Pre-post 2 * slots empty recv WQEs so neither data nor credit IMMs can ever RNR-NAK.
    for (uint32_t i = 0; i < 2 * slots; i++) {
        if (bulk_post_empty_recv(b) != 0) {
            LOGE("bulk_open: post_recv failed");
            bulk_close(b); return NULL;
        }
    }

    b->local_psn = (uint32_t)((now_ns() ^ (uint64_t)(uintptr_t)b) & 0xFFFFFFu);
    memset(out_ep, 0, sizeof(*out_ep));
    out_ep->qpn = b->qp->qp_num;
    out_ep->psn = b->local_psn;
    out_ep->rkey = b->mr->rkey;
    out_ep->chunk_bytes = chunk_bytes;
    out_ep->addr = (uint64_t)(uintptr_t)b->buf;
    memcpy(out_ep->gid, mygid.raw, 16);
    return b;
}

// Transition QP INIT -> RTR -> RTS using peer's BulkEndpoint (exchanged over TCP).
int bulk_connect(BulkCtx* b, const BulkEndpoint* peer) {
    if (!b || !peer) return -1;
    if (peer->chunk_bytes != b->chunk_bytes) {
        LOGE("bulk_connect: chunk mismatch local=%u peer=%u", b->chunk_bytes, peer->chunk_bytes);
        return -1;
    }
    b->remote_addr = peer->addr;
    b->remote_rkey = peer->rkey;

    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = peer->qpn;
    a.rq_psn = peer->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = b->port_num;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    a.ah_attr.grh.sgid_index = b->gid_idx;
    a.ah_attr.grh.hop_limit = 1;
    if (ibv_modify_qp(b->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        LOGE("bulk_connect: modify_qp RTR failed (%s)", strerror(errno));
        return -1;
    }

    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 18;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = b->local_psn;
    a.max_rd_atomic = 1;
    if (ibv_modify_qp(b->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        LOGE("bulk_connect: modify_qp RTS failed (%s)", strerror(errno));
        return -1;
    }
    return 0;
}

// Sender: acquire the next free registered MR slot (waits for receiver credit + send retirement).
int bulk_send_acquire(BulkCtx* b, uint32_t* out_slot, uint8_t** out_ptr) {
    if (!b || !out_slot || !out_ptr) return -1;
    // Drain any already-completed send CQEs and credit IMMs opportunistically.
    struct ibv_wc wc;
    while (b->inflight_send > 0 && ibv_poll_cq(b->cq_send, 1, &wc) > 0) {
        if (wc.status != IBV_WC_SUCCESS) { LOGE("bulk send CQE error %d", wc.status); return -2; }
        b->inflight_send--;
    }
    while (ibv_poll_cq(b->cq_recv, 1, &wc) > 0) {
        if (wc.status != IBV_WC_SUCCESS) { LOGE("bulk credit CQE error %d", wc.status); return -2; }
        if (bulk_post_empty_recv(b) != 0) return -3;
        b->credits++;
    }
    // Wait until a peer slot credit is available.
    while (b->credits == 0) {
        if (bulk_poll_cq_wait(b->cq_recv, &wc, BULK_TIMEOUT_MS) <= 0) return -4;
        if (bulk_post_empty_recv(b) != 0) return -3;
        b->credits++;
    }
    // Ensure our local send buffer slot isn't still in flight on the NIC.
    while (b->inflight_send >= b->slots) {
        if (bulk_poll_cq_wait(b->cq_send, &wc, BULK_TIMEOUT_MS) <= 0) return -5;
        b->inflight_send--;
    }
    uint32_t s = b->send_seq % b->slots;
    *out_slot = s;
    *out_ptr = b->buf + (size_t)s * (size_t)b->chunk_bytes;
    return 0;
}

// Sender: RDMA_WRITE_WITH_IMM slot `slot` (length `len` bytes, 1..=chunk_bytes) into peer's matching slot.
int bulk_send_commit(BulkCtx* b, uint32_t slot, uint32_t len) {
    if (!b || slot >= b->slots || len == 0 || len > b->chunk_bytes) return -1;
    size_t off = (size_t)slot * (size_t)b->chunk_bytes;
    struct ibv_sge sg = {
        .addr   = (uint64_t)(uintptr_t)(b->buf + off),
        .length = len,
        .lkey   = b->mr->lkey,
    };
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = b->send_seq;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl((slot << BULK_IMM_SLOT_SHIFT) | (len & BULK_IMM_LEN_MASK));
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = b->remote_addr + off;
    wr.wr.rdma.rkey = b->remote_rkey;
    int rc = ibv_post_send(b->qp, &wr, &bad);
    if (rc) { LOGE("bulk_send_commit: post_send rc=%d (%s)", rc, strerror(rc)); return -1; }
    b->send_seq++;
    b->credits--;
    b->inflight_send++;
    return 0;
}

// Sender: wait until all posted chunks have retired on the local send CQ.
int bulk_send_drain(BulkCtx* b) {
    if (!b) return -1;
    struct ibv_wc wc;
    while (b->inflight_send > 0) {
        if (bulk_poll_cq_wait(b->cq_send, &wc, BULK_TIMEOUT_MS) <= 0) return -1;
        b->inflight_send--;
    }
    return 0;
}

// Receiver: wait for the next RDMA_WRITE_WITH_IMM chunk, re-posting a recv WQE before returning.
int bulk_recv_next(BulkCtx* b, uint32_t* out_slot, const uint8_t** out_ptr, uint32_t* out_len) {
    if (!b || !out_slot || !out_ptr || !out_len) return -1;
    struct ibv_wc wc;
    if (bulk_poll_cq_wait(b->cq_recv, &wc, BULK_TIMEOUT_MS) <= 0) return -1;
    if (bulk_post_empty_recv(b) != 0) { LOGE("bulk_recv_next: post_recv failed"); return -2; }
    uint32_t imm = ntohl(wc.imm_data);
    uint32_t slot = imm >> BULK_IMM_SLOT_SHIFT;
    uint32_t len  = imm & BULK_IMM_LEN_MASK;
    if (slot >= b->slots || len == 0 || len > b->chunk_bytes) {
        LOGE("bulk_recv_next: corrupt imm slot=%u len=%u", slot, len);
        return -3;
    }
    *out_slot = slot;
    *out_ptr  = b->buf + (size_t)slot * (size_t)b->chunk_bytes;
    *out_len  = len;
    return 0;
}

// Receiver: release `slot` back to the sender via a 0-byte RDMA_WRITE_WITH_IMM credit ack.
int bulk_recv_release(BulkCtx* b, uint32_t slot) {
    if (!b || slot >= b->slots) return -1;
    // Drain completed credit-ack CQEs so the send queue never fills.
    struct ibv_wc wc;
    while (b->inflight_ack > 0 && ibv_poll_cq(b->cq_send, 1, &wc) > 0) {
        if (wc.status != IBV_WC_SUCCESS) { LOGE("bulk ack CQE error %d", wc.status); return -2; }
        b->inflight_ack--;
    }
    while (b->inflight_ack >= 16) {
        if (bulk_poll_cq_wait(b->cq_send, &wc, BULK_TIMEOUT_MS) <= 0) return -3;
        b->inflight_ack--;
    }
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = slot;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl(slot);
    wr.num_sge = 0;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = b->remote_addr;
    wr.wr.rdma.rkey = b->remote_rkey;
    int rc = ibv_post_send(b->qp, &wr, &bad);
    if (rc) { LOGE("bulk_recv_release: post_send rc=%d (%s)", rc, strerror(rc)); return -1; }
    b->inflight_ack++;
    return 0;
}
