//! The TP transport attachment: `TpAttach` / `TpState` / `SingleXport` / `DecXport` and `impl TpState`
//! (attach, the sp all-gather / reduce-scatter helpers, the decode doorbell accessors). The knobs that pick
//! the transports live in `tp_config.rs`. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Transport attachment
// ---------------------------------------------------------------------------------------------

/// The gpu_batch doorbell kernels (K1 tp_gate_copy_signal / K2 tp_wait_add), loaded under their
/// own module name so the EXL3 engine never collides with a gpu.rs-loaded gpu_batch module.
const TP_MODULE: &str = "exl3_tp";
/// The proxy thread's core: a big X925 (the pinned-proxy rule, memory note pin-the-tp-proxy-thread).
const TP_PROXY_CORE: i32 = 19;
/// Widest all-reduce row count the partial buffer holds: decode 1 … verify 16, and (TP-B) a
/// prefill chunk (the reduce is chunked into ring slots). TP-D: 4096 — the served default C=2048
/// with WP16 C1 tail checkpoints merges chunks up to 2C - 1 = 4095 rows (psc_rows); 2048 failed the
/// first served 32K prompt ("row-parallel chain m=2778 … exceeds the TP partial buffer"). The serve
/// boot refuses a --prefill-chunk whose widest chunk exceeds this (exl3_serve::build_serve).
pub const TP_MAX_ROWS: usize = 4096;

/// What `FwdModel::load_tp` needs to attach TP: this rank, the world and the brought-up link.
pub struct TpAttach {
    pub rank: i32,
    pub world: i32,
    pub link: crate::net::TpLink,
}

pub struct TpState {
    pub rank: i32,
    pub world: i32,
    pub(super) ctx: u64,
    pub(super) part: CudaSlice<f32>,
    /// TP-F prefill transport (--tp-prefill-xport): None = the single-block serial K1/K2 pair
    /// for every reduce (the TP-B..E path).
    xport: Option<crate::tp_xport::Pipe>,
    pub(super) kern: crate::tp_xport::Kernels,
    /// the dual-rail transport's second link (its NetCtx address), for the abort check
    aux: Option<usize>,
    /// TP-G decode transport (--tp-dec-xport): None = the TP-B..F serial K1/K2 + cvt pair.
    pub(in crate::exl3_forward) dec: Option<DecXport>,
    /// TP-4F2 (`--tp-reduce single`): the dedicated single-stage reduce transport (None = rd, today's path).
    pub(super) single: Option<SingleXport>,
    /// TP-4Z2 (`--tp-k2-prefetch`): hint the NEXT decode K2 with the weights of the kernel that consumes this reduce's
    /// output, so its idle threads pull them into L2 during the wait. `pf_hint` = [p0, n0, p1, n1], set by the layer
    /// loop at capture time and consumed (swapped to zero) by the first K2 launched after it. A prefetch only.
    pf_on: bool,
    pf_hint: [std::sync::atomic::AtomicU64; 4],
}

/// TP-4F2: the uniform one-shot ctx the single-stage decode/verify reduce rides (its own proxy + epoch counter).
pub(super) struct SingleXport {
    /// device address of the ctx (the first argument of the folded K1 producers and `xq_tp_wait_add_dec_single`)
    pub(super) ctx: u64,
}

/// TP-G (T2 lever #2): the decode (<= 16 rows) all-reduce with K1 folded into the producer's last
/// block (xq_ks_combine_f32_k1l / xq_moe_combine_ep_k1l / xq_cvt_f16_f32_k1l) and a multi-block K2
/// (xq_tp_wait_add_dec) that writes the f16 sum (the cvt launch folded) and also gates GPU-side on the
/// payload tail (--tp-dec-grecv, default on), whichever of that and the proxy's cpu_done comes first.
pub(in crate::exl3_forward) struct DecXport {
    /// K1 last-block arrive counter (device memory, zeroed; re-armed by the last block)
    pub arrive: CudaSlice<u32>,
    /// GPU-side receive (xq_tp_wait_add_dec io bit 3): commit hint + payload tail, OR the cpu_done
    pub grecv: bool,
    /// K2 grid (blocks x 256 threads)
    pub blocks: u32,
}

impl TpState {
    pub fn attach(dev: &Arc<CudaDevice>, stream: &CudaStream, t: TpAttach, h: usize) -> Result<Self> {
        anyhow::ensure!(t.world >= 2, "EXL3 TP: attach needs world >= 2 (got {})", t.world);
        world_gt2_flag_refusals(t.world)?;
        world_gt2_overlap_refusal(t.world, pf_overlap_code())?;
        anyhow::ensure!(t.world == 2 || !t.link.oneshot_on(),
            "EXL3 TP world {}: the transport ctx selected the one-shot all-peers push — refused on the EXL3 path \
             (bf16 wire, tp_wait_add_4way; it wedged NVFP4 serving and measured at parity with the two-shot)", t.world);
        let kern = crate::tp_xport::Kernels::load(dev, TP_MODULE)?;
        let mut link = t.link;
        // DEFAULT payload = one fp32 hidden row (per-barrier lengths are explicit at every site).
        link.set_payload(h * 4, true).context("net_set_payload")?;
        let trace = crate::opts::var(crate::opt!("tp-trace")).is_ok() || crate::tp::tp_config().map_or(false, |c| c.trace);
        if trace {
            crate::net::trace_enable(&mut link);
            crate::dprintln!("[exl3-tp] per-barrier tracing ON");
        }
        let ctx = link.ctx_device_ptr();
        let addr = link.ctx_addr();
        let peer_ip = link.peer_ip();
        // The proxy OWNS the transport from here (gpu.rs attach_tp contract); the OS reclaims at exit.
        std::mem::forget(link);
        crate::net::spawn_proxy(addr, TP_PROXY_CORE);
        let mut part = dev.alloc_zeros::<f32>(TP_MAX_ROWS * h)?;
        let mode = prefill_xport_mode();
        // TP-I (audit C6): the FIRST cross-rank sync after each rank's independent shard load (50-220 s,
        // skewed by page-cache warmth), so it must tolerate minutes of legitimate skew: probe-tolerant,
        // no overall deadline, a dead peer is still caught by the out-of-band liveness probe (abort
        // code 10). It was a 10 s net_agree once (abort code 7). Live-round deadlines are untouched.
        // TP-4B: at world > 2 it runs in EVERY prefill mode — round partners meet in the very first
        // reduce and a load skew above the 10 s proxy watchdog would abort it (code 6). World 2 keeps
        // its old condition (mode 2 only): modes 0/1 never had the rendezvous and still do not.
        // It also rendezvouses the primary link before rank 1 dials rail 2 (mode 2), and the frame
        // carries the mode (SPMD check).
        if t.world > 2 || mode == 2 { boot_rendezvous(t.rank, mode)?; }
        if t.world > 2 {
            align_round_phase(dev, &kern, stream, ctx, &mut part, t.world, t.rank, "rail 1", &crate::net::traced_device_epoch)?;
        }
        let mut aux: Option<usize> = None;
        let xport = if mode > 0 {
            let mut rails = vec![ctx];
            if mode == 2 {
                let dev2 = rail2_dev();
                let (mut l2, how) = if t.world == 2 {
                    let port = crate::tp::TP_PORT + TP_AUX_PORT_OFF;
                    if t.rank != 0 { std::thread::sleep(std::time::Duration::from_millis(300)); }
                    let l2 = crate::net::TpLink::connect(t.rank, if t.rank == 0 { "" } else { &peer_ip }, port, &dev2,
                                                         crate::tp::resolve_gid_idx(&dev2), crate::tp::TP_SLOT_BYTES)
                        .with_context(|| format!("TP-F rail 2 ({dev2}, port {port}, peer {peer_ip})"))?;
                    (l2, format!("TCP {port}, peer {peer_ip}"))
                } else {
                    // TP-4E: rail 2 as a full N-way link. Every rank is past the boot rendezvous (rail 1, above),
                    // and the preflight agrees the second device's health over rail 1 BEFORE anyone dials, so a
                    // missing device fails every rank in seconds instead of hanging the handshake (accept has no
                    // timeout). Its own port base (`rail2_nway_base`, collision-checked against rail 1).
                    rail2_preflight(t.rank, &dev2)?;
                    let base = rail2_nway_base(t.world)?;
                    let ips = crate::tp::resolve_topology(t.world)?;
                    let l2 = crate::net::TpLink::connect_nway(t.rank, t.world, &ips, base, &dev2,
                                                              crate::tp::resolve_gid_idx(&dev2), crate::tp::TP_SLOT_BYTES)
                        .with_context(|| format!("TP-4E rail 2 N-way link ({dev2}, TCP base {base}, world {})", t.world))?;
                    (l2, format!("N-way, TCP base {base}, {} peers", t.world - 1))
                };
                l2.set_payload(h * 4, true).context("rail 2 net_set_payload")?;
                rails.push(l2.ctx_device_ptr());
                let a2 = l2.ctx_addr();
                std::mem::forget(l2);
                crate::net::spawn_proxy_aux(a2, TP_AUX_PROXY_CORE);
                aux = Some(a2);
                if t.world > 2 {
                    // the pad exchanges (none in practice: a fresh ctx is at epoch 0) need rail 2's proxy running
                    align_round_phase(dev, &kern, stream, *rails.last().unwrap(), &mut part, t.world, t.rank, "rail 2",
                                      &move || crate::net::ctx_device_epoch(a2))?;
                }
                crate::dprintln!("[exl3-tp] rank {} rail 2 UP: {dev2} ({how}), proxy pinned to core {TP_AUX_PROXY_CORE}", t.rank);
            }
            let lookahead = crate::opts::var(crate::opt!("tp-xport-lookahead")).ok().and_then(|v| v.parse().ok()).unwrap_or(4usize);
            let blocks = crate::opts::var(crate::opt!("tp-xport-blocks")).ok().and_then(|v| v.parse().ok()).unwrap_or(16u32);
            Some(crate::tp_xport::Pipe::new(dev, rails, t.world as usize, lookahead, blocks)?)
        } else { None };
        // TP-4F2 (`--tp-reduce single`): the dedicated single-stage reduce ctx. Built here — every rank is past the
        // boot rendezvous and rail 2 (same sequence on every rank, so handshake skew is bounded) — as a full world-4
        // N-way link on the PRIMARY device, a uniform one-shot ctx (net_shim `oneshot_uniform`) with its OWN epoch
        // counter, so the rail-1 R10 round phase (`align_round_phase` above) is never perturbed by it. Its abort is
        // mirrored into rail 1's inside its proxy (`set_parent`), because a graph replay never returns to the host
        // between reduces. Absent the flag nothing here runs (no ctx, no thread, no ports).
        let single = if tp_reduce_single_for(t.world) {
            tp_reduce_refusal(true, t.world, dec_xport_mode())?;
            let base = single_nway_base(t.world)?;
            let ips = crate::tp::resolve_topology(t.world)?;
            let dev1 = crate::tp::rdma_dev();
            let mut l3 = crate::net::TpLink::connect_nway_uniform(t.rank, t.world, &ips, base, &dev1,
                                                                  crate::tp::resolve_gid_idx(&dev1), XPORT_MIN_BYTES)
                .with_context(|| format!("TP-4F2 single-stage ctx ({dev1}, TCP base {base}, world {})", t.world))?;
            l3.set_payload(h * 4, true).context("single-stage ctx net_set_payload")?;
            l3.set_parent(addr);
            let ctx3 = l3.ctx_device_ptr();
            let a3 = l3.ctx_addr();
            std::mem::forget(l3);
            crate::net::spawn_proxy_aux(a3, TP_SINGLE_PROXY_CORE);
            crate::dprintln!("[exl3-tp] rank {} --tp-reduce single UP: dedicated uniform one-shot ctx on {dev1} (N-way, TCP base {base}, {} peers), \
                      proxy pinned to core {TP_SINGLE_PROXY_CORE}; decode/verify reduces = ONE all-peers fp32 stage, sum (p0+p2)+(p1+p3) \
                      (bitwise the rd two-stage tree); prefill reduces and the key/screen exchanges stay on rail 1",
                     t.rank, t.world - 1);
            Some(SingleXport { ctx: ctx3 })
        } else { None };
        let dec = if dec_xport_mode() == 1 {
            let mut arrive = dev.alloc_zeros::<u32>(1)?;
            dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2: alloc_zeros does not zero
            // TP-4C: world > 2 receives on cpu_done only (see world_gt2_flag_refusals); world 2 unchanged
            let grecv = t.world <= 2 && crate::opts::var(crate::opt!("tp-dec-grecv")).map_or(true, |v| v.trim() != "0");
            let blocks = crate::opts::var(crate::opt!("tp-dec-blocks")).ok().and_then(|v| v.parse().ok()).unwrap_or(8u32).clamp(1, 64);
            Some(DecXport { arrive, grecv, blocks })
        } else { None };
        dev.synchronize()?;
        match &dec {
            Some(d) if t.world > 2 => crate::dprintln!("[exl3-tp] decode transport: world {} = {} exchanges per logical reduce (TP-4C): the producer's \
                                                folded K1 + multi-block K2 for round 0, plain K1 + K2 for the later rounds (fp32 \
                                                intermediate, f16 only on the last round), {} K2 blocks, receive cpu_done only \
                                                (--tp-dec-grecv is world 2 only — TP-4F lever)",
                                               t.world, (t.world as u32).trailing_zeros(), d.blocks),
            Some(d) => crate::dprintln!("[exl3-tp] decode transport: K1 FOLDED into the producers + multi-block K2 ({} blocks, f16 cvt folded), \
                                 receive {} (--tp-dec-xport default; =0 = the serial pair)", d.blocks,
                                if d.grecv { "GPU-side (payload tail) or cpu_done" } else { "cpu_done only (--tp-dec-grecv=0)" }),
            None => crate::dprintln!("[exl3-tp] decode transport: serial single-block K1/K2 + cvt (--tp-dec-xport=0)"),
        }
        crate::dprintln!("[exl3-tp] rank {}/{} attached: fp32 all-reduce, proxy pinned to core {TP_PROXY_CORE}, \
                  EP deal '{}', partial buffer {} rows", t.rank, t.world, ep_deal_kind(), TP_MAX_ROWS);
        match &xport {
            Some(x) => crate::dprintln!("[exl3-tp] prefill transport: PIPELINED multi-block (--tp-prefill-xport) — \
                                 {} rail(s), lookahead {} epochs/rail, {} blocks, reduces > {} KiB",
                                 x.rails.len(), x.lookahead, x.blocks, XPORT_MIN_BYTES / 1024),
            None => crate::dprintln!("[exl3-tp] prefill transport: serial single-block K1/K2 (--tp-prefill-xport=0)"),
        }
        crate::dprintln!("[exl3-tp] prefill transport choice: {}",
                 prefill_xport_why(t.world, mode, crate::opts::var(crate::opt!("tp-prefill-xport")).is_ok()));
        match pf_overlap_rows() {
            0 => crate::dprintln!("[exl3-tp] prefill overlap: OFF (--tp-prefill-overlap off)"),
            _ if pf_overlap_fold() => crate::dprintln!("[exl3-tp] prefill overlap: FOLD — whole-row epochs, each epoch's K2m replaced \
                           by the fused consumer (xq_pf_hc_inj_norm_k2 / exl3_had_svh_k2 + inject+norm), one stream \
                           (--tp-prefill-overlap fold)"),
            r => crate::dprintln!("[exl3-tp] prefill overlap: ON — the next hc_pre's inject+norm (and the mixer's svh) run per \
                           landed {r}-row range inside each pipelined prefill reduce, {} (--tp-prefill-overlap {})",
                          if pf_overlap_dual() { "K1m/K2m on the transport's own stream (dual)" } else { "same stream (inline)" },
                          pf_overlap_desc(pf_overlap_code())),
        }
        if t.world > 2 && pf_overlap_rows() > 0 {
            crate::dprintln!("[exl3-tp] prefill overlap at world {}: the hooks ride the round-aware program (TP-4E): a row range is consumed \
                      only after its LAST round's K2m (fp32 running sum staged in the partial buffer, one f16 rounding at the end); \
                      --tp-prefill-overlap fold is refused at world > 2", t.world);
        }
        if seq_parallel_on() && t.world > 4 {
            crate::dprintln!("[exl3-tp] TP-SP sequence-parallel prefill: INACTIVE at world {} (the default ON yields: SP exists at world 2 \
                      and world 4 only; the replicated prefill runs; an explicit --tp-seq-parallel is refused here)",
                     t.world);
        } else if seq_parallel_on() {
            crate::dprintln!("[exl3-tp] TP-SP sequence-parallel prefill: ON (--tp-seq-parallel {}) at world {} — chunks of {SP_MIN_ROWS}..={SP_MAX_ROWS} \
                      rows: hc_pre on the own row block (rank r the rows [r*ceil(c/{}), (r+1)*ceil(c/{})), hc GEMM plans pinned to the full \
                      chunk's), x all-gathered before every mixer / router, reduce-scatters instead of all-reduces{}{}{}",
                     seq_parallel_desc(seq_parallel_code()), t.world, t.world, t.world,
                     if t.world == 4 { " (TP-4S world 4: quarter rows; recursive-halving reduce-scatter / recursive-doubling all-gather on the two doorbell rounds)" } else { "" },
                     if seq_parallel_router() { "; the MoE router on the own rows + one ids/weights all-gather per layer (SP-2)" } else { "" },
                     if xport.is_some() { "" } else { " — INACTIVE: needs the pipelined transport (--tp-prefill-xport != 0)" });
        }
        let pf_on = !matches!(crate::opts::var(crate::opt!("tp-k2-prefetch")).map(|v| v.trim().to_ascii_lowercase()).as_deref(),
                              Ok("off") | Ok("0") | Ok("false"));
        crate::dprintln!("[exl3-tp] rank {} --tp-k2-prefetch {} (L2 run-ahead of the next hc mixer's weights from the idle threads of the decode K2)",
                 t.rank, if pf_on { "ON" } else { "OFF" });
        Ok(TpState { rank: t.rank, world: t.world, ctx, part, xport, kern, aux, dec, single, pf_on, pf_hint: Default::default() })
    }

    /// In-place FP32 sum-all-reduce of `n` floats at device address `p` (world 2): K1 publishes
    /// the local partial, K2 (mode 2) waits for the peer's and writes rank0 + rank1 (canonical
    /// order on BOTH ranks => bit-identical results), no rounding. Both on the blocking stream.
    /// TP-B: payloads larger than one ring slot (prefill: 2048 x 2560 x 4 B = 21 MB) run as
    /// consecutive epochs of <= TP_EPOCH_FLOATS each (the existing per-epoch rings; the same
    /// elementwise canonical sum, so the result is independent of the chunking).
    pub(super) fn all_reduce_f32(&self, l: &Launcher, p: u64, n: usize) -> Result<()> {
        if let Some(x) = &self.xport {
            if let Some(a) = self.aux {
                // rail 2 is invisible to the agrees (they ride rail 1): surface its abort there (I9)
                let st = crate::net::ctx_abort_status(a);
                if st != 0 {
                    crate::net::abort_link();
                    anyhow::bail!("TP-F rail 2 transport ABORT (status {st}) — aborting the primary link");
                }
            }
            if n * 4 > XPORT_MIN_BYTES && crate::tp_xport::pipe_ok(n) {
                return crate::tp_xport::reduce_pipe(&self.kern, l.stream, x, crate::tp_xport::Io::f32_inplace(p), n);
            }
        }
        crate::tp_xport::reduce_serial(&self.kern, l.stream, self.ctx, p, n, self.rounds())
    }

    /// TP-F fused prefill reduce: the local partial at `io.src` (f16 or f32) -> canonical fp32 sum ->
    /// `io.out` (f16: one __float2half_rn rounding, exactly the xq_cvt_f32_f16 the unfused path runs).
    /// Returns false (nothing launched) when the pipelined transport does not take this reduce — the
    /// caller then runs its explicit cvt + all_reduce_f32 + cvt sequence.
    pub(super) fn all_reduce_io(&self, io: crate::tp_xport::Io, n: usize) -> Result<Option<crate::tp_xport::Io>> {
        match &self.xport {
            Some(x) if n * 4 > XPORT_MIN_BYTES && crate::tp_xport::pipe_ok(n) => {
                if x.rounds == 1 { return Ok(Some(io)); }
                // world>2 (rounds > 1): round k's local is round k-1's fp32 SUM, which an f16 in-place
                // buffer would round between rounds (output-changing). The running sum is staged in fp32
                // in `part` (the fp32 scratch both fallback chains already reduce through; at the MoE
                // site src == part, consumed in place), round 0 still ships the f16-exact partial at f16,
                // the last round writes `io.out` (one f16 rounding): bitwise the explicit cvt + fp32
                // all-reduce + cvt path, without its two conversion passes.
                anyhow::ensure!(n <= self.part.len(), "world > 2 staged reduce: {n} floats exceed the fp32 partial buffer ({})",
                                self.part.len());
                Ok(Some(crate::tp_xport::Io { mid: if io.mid != 0 { io.mid } else { self.part_ptr() }, ..io }))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn reduce_io(&self, l: &Launcher, io: crate::tp_xport::Io, n: usize) -> Result<()> {
        self.reduce_io_hooked(l, io, n, None)
    }

    /// reduce_io with the TP-I2 rows-landed hook (None = reduce_io).
    pub(super) fn reduce_io_hooked(&self, l: &Launcher, io: crate::tp_xport::Io, n: usize,
                        hook: Option<crate::tp_xport::RowHook<'_>>) -> Result<()> {
        let x = self.xport.as_ref().context("reduce_io without the pipelined transport")?;
        if let Some(a) = self.aux {
            let st = crate::net::ctx_abort_status(a);
            if st != 0 {
                crate::net::abort_link();
                anyhow::bail!("TP-F rail 2 transport ABORT (status {st}) — aborting the primary link");
            }
        }
        crate::tp_xport::reduce_pipe_hooked(&self.kern, l.stream, x, io, n, hook)
    }

    pub(super) fn part_ptr(&self) -> u64 { *self.part.device_ptr() as u64 }

    /// Recursive-doubling rounds per all-reduce: log2(world) (1 at world 2; attach admits 2 and 4 only).
    pub(super) fn rounds(&self) -> usize { (self.world as u32).trailing_zeros() as usize }

    /// TP-SP: the exchanges need the pipelined transport with one doorbell round per rank-bit: world 2 (one round)
    /// or, TP-4S, world 4 (two rounds: A = rank ^ 2, B = rank ^ 1). Other worlds have no SP schedule.
    pub(in crate::exl3_forward) fn sp_ready(&self) -> bool {
        matches!(self.world, 2 | 4) && self.xport.as_ref().map_or(false, |x| x.rounds == self.rounds())
    }

    /// The rail-2 abort surfaced on the primary link (rail 2 is invisible to the agrees, I9).
    fn rail2_check(&self) -> Result<()> {
        if let Some(a) = self.aux {
            let st = crate::net::ctx_abort_status(a);
            if st != 0 {
                crate::net::abort_link();
                anyhow::bail!("TP-F rail 2 transport ABORT (status {st}) — aborting the primary link");
            }
        }
        Ok(())
    }

    /// TP-SP: all-gather `row_bytes`-byte rows at `base` — this rank's own rows to the peer, the peer's
    /// own rows into this rank's buffer (a bit copy both ways).
    pub(in crate::exl3_forward) fn sp_all_gather(&self, l: &Launcher, base: u64, row_bytes: usize, sp: &SpRows) -> Result<()> {
        let x = self.xport.as_ref().context("TP-SP all-gather without the pipelined transport")?;
        self.rail2_check()?;
        if sp.world == 4 { return crate::tp_xport::all_gather_rows_w4(&self.kern, l.stream, x, base, row_bytes, &sp.blocks, sp.rank); }
        crate::tp_xport::all_gather_rows(&self.kern, l.stream, x, base, row_bytes, sp.own, sp.peer)
    }

    /// TP-SP2: all-gather several row buffers of the same row blocks in ONE exchange program (the
    /// router's ids + weights rows).
    pub(in crate::exl3_forward) fn sp_all_gather_multi(&self, l: &Launcher, bufs: &[(u64, usize)], sp: &SpRows) -> Result<()> {
        let x = self.xport.as_ref().context("TP-SP all-gather without the pipelined transport")?;
        self.rail2_check()?;
        if sp.world == 4 { return crate::tp_xport::all_gather_rows_multi_w4(&self.kern, l.stream, x, bufs, &sp.blocks, sp.rank); }
        crate::tp_xport::all_gather_rows_multi(&self.kern, l.stream, x, bufs, sp.own, sp.peer)
    }

    /// TP-SP: reduce-scatter a c-row partial (`io` as reduce_io, row_len elements per row): the canonical
    /// sum lands in this rank's own rows of io.out only; `hook` = the dual overlap on the own rows.
    pub(in crate::exl3_forward) fn sp_reduce_scatter(&self, l: &Launcher, io: crate::tp_xport::Io, row_len: usize, sp: &SpRows,
                                    hook: Option<crate::tp_xport::SpHook<'_>>) -> Result<()> {
        let x = self.xport.as_ref().context("TP-SP reduce-scatter without the pipelined transport")?;
        self.rail2_check()?;
        if sp.world == 4 {
            // world 4: the running sum between the two exchanges lives in an fp32 stage indexed by absolute row
            // (`part`, as `all_reduce_io` stages the world > 2 all-reduce); an fp32 in-place src stages itself
            anyhow::ensure!(sp.c * row_len <= self.part.len(), "TP-4S reduce-scatter: {} floats exceed the fp32 partial buffer ({})",
                            sp.c * row_len, self.part.len());
            let io = crate::tp_xport::Io { mid: if io.mid != 0 { io.mid } else { self.part_ptr() }, ..io };
            return crate::tp_xport::reduce_scatter_rows_w4(&self.kern, l.stream, x, io, row_len, &sp.blocks, sp.rank, hook);
        }
        crate::tp_xport::reduce_scatter_rows(&self.kern, l.stream, x, io, row_len, sp.own, sp.peer, hook)
    }

    /// TP-H #3: the device address of the doorbell context (the first argument of every folded K1 / K2).
    pub(in crate::exl3_forward) fn ctx_addr(&self) -> u64 { self.ctx }

    /// TP-H #3: the K1 arrive counter of the decode transport (None = the serial pair: no folds).
    pub(in crate::exl3_forward) fn dec_arrive(&self) -> Option<u64> { self.dec.as_ref().map(|d| *d.arrive.device_ptr() as u64) }

    /// TP-H #3 K2: out[n] (fp32) = canonical lower + upper of `local` (this rank's zero-padded fp32 block
    /// maxima) and the peer's payload of the epoch the preceding xq_dh_screen_k1 published.
    /// TP-4C: world-general (2 or 4). At world 4 it is the two-exchange screen (`dec_reduce_finish`): the
    /// intermediate is `out` itself (fp32), so the zero-padded `local` keeps its zeros for the next screen,
    /// and the sum over the ranks' disjoint spans (one non-zero contributor per entry) is bitwise the
    /// replicated array under any association.
    pub(in crate::exl3_forward) fn dh_bmax_k2(&self, l: &Launcher, out: u64, local: u64, n: usize) -> Result<()> {
        let d = self.dec.as_ref().context("the sharded DHEAD screen without the decode transport")?;
        anyhow::ensure!(n % 4 == 0 && n * 4 <= XPORT_MIN_BYTES, "the sharded DHEAD screen K2: n {n} is not a float4-clean decode-sized payload");
        anyhow::ensure!(out != local, "the sharded DHEAD screen K2: out must not alias the zero-padded local (the multi-round intermediate lives in out)");
        self.dec_reduce_finish(l, d, out, local, n, false, false)
    }

    /// TP-G decode K2: out (f16 when `out_f16`) = canonical lower + upper of `local` (f16 when
    /// `local_f16`) and the peer's fp32 payload of the epoch the preceding folded K1 published.
    fn dec_k2(&self, l: &Launcher, d: &DecXport, out: u64, local: u64, n: usize, local_f16: bool, out_f16: bool) -> Result<()> {
        let io = (local_f16 as i32) | ((out_f16 as i32) << 1) | ((d.grecv as i32) << 3);
        let pf = self.take_pf_hint();
        xqlaunch!(l, "xq_tp_wait_add_dec", (d.blocks, 1, 1), (256, 1, 1), 0,
                  (self.ctx, out, local, n as i32, io, pf[0], pf[1] as u32, pf[2], pf[3] as u32))
    }

    /// TP-4C: one decode-path LOGICAL reduce = `rounds` exchanges (`tp_xport::dec_exchanges`), exactly
    /// `rounds` device epochs (every K1 bumps the counter once, so the R10 round phase is invariant across
    /// arms). Exchange 0's K1 is the PRODUCER's folded K1 (already launched by the caller); K2 reads the
    /// caller's `local`. Exchanges k > 0 launch the plain single-block K1 over the fp32 intermediate (the
    /// round k-1 sum) and a K2 that reads it; only the last K2 writes `out` (f16 when `out_f16`).
    /// At world 2 (rounds == 1) this is exactly the single `dec_k2(out, local, ..)` launch of TP-G.
    fn dec_reduce_finish(&self, l: &Launcher, d: &DecXport, out: u64, local: u64, n: usize,
                         local_f16: bool, out_f16: bool) -> Result<()> {
        let rounds = self.rounds();
        anyhow::ensure!(rounds == 1 || !local_f16,
            "decode reduce at world {}: an f16 local operand would round between rounds", self.world);
        anyhow::ensure!(n <= crate::tp_xport::EPOCH_FLOATS, "decode reduce: {n} floats exceed one epoch");
        let inter = crate::tp_xport::dec_intermediate(local, out, out_f16);
        for x in crate::tp_xport::dec_exchanges(rounds) {
            if !x.folded_k1 { crate::tp_xport::k1_serial(&self.kern, l.stream, self.ctx, inter, n)?; }
            let a = crate::tp_xport::dec_exchange_args(&x, local, local_f16, out, out_f16);
            self.dec_k2(l, d, a.out, a.local, n, a.local_f16, a.out_f16)?;
        }
        Ok(())
    }

    /// TP-4F2: the doorbell ctx the producer-folded K1 of a decode/verify SUM reduce (chain_rowpar / ep_combine_reduce)
    /// publishes on: the single-stage ctx under `--tp-reduce single`, else rail 1 (today). The screen / key
    /// exchanges keep `ctx_addr()` / `tp.ctx` (rail 1, rd) — only the 96 per-round sum reduces move.
    pub(in crate::exl3_forward) fn dec_ctx(&self) -> u64 { self.single.as_ref().map_or(self.ctx, |s| s.ctx) }

    /// TP-4Z2: arm the L2 run-ahead of the NEXT decode K2 (two byte ranges: p0/n0, p1/n1). No-op when
    /// `--tp-k2-prefetch off`. The K2 that follows in launch (= capture) order consumes it.
    pub(in crate::exl3_forward) fn set_pf_hint(&self, r: [(u64, u32); 2]) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.pf_on { return; }
        PF_ARMED.fetch_add(1, Relaxed);
        self.pf_hint[0].store(r[0].0, Relaxed); self.pf_hint[1].store(r[0].1 as u64, Relaxed);
        self.pf_hint[2].store(r[1].0, Relaxed); self.pf_hint[3].store(r[1].1 as u64, Relaxed);
    }

    /// TP-4Z2: take (and clear) the pending hint: [p0, n0, p1, n1], all zero when none is armed.
    fn take_pf_hint(&self) -> [u64; 4] {
        use std::sync::atomic::Ordering::Relaxed;
        let h = [self.pf_hint[0].swap(0, Relaxed), self.pf_hint[1].swap(0, Relaxed), self.pf_hint[2].swap(0, Relaxed), self.pf_hint[3].swap(0, Relaxed)];
        if h[1] != 0 || h[3] != 0 { PF_USED.fetch_add(1, Relaxed); }
        h
    }

    /// TP-4F2: the K2 of a decode/verify sum reduce (fp32 `local` -> f16 `out`): `dec_reduce_finish` (rd: `rounds`
    /// exchanges) by default, or ONE `xq_tp_wait_add_dec_single` launch on the single-stage ctx whose epoch the
    /// preceding folded K1 published.
    pub(super) fn dec_sum_finish(&self, l: &Launcher, d: &DecXport, out: u64, local: u64, n: usize) -> Result<()> {
        match &self.single {
            None => self.dec_reduce_finish(l, d, out, local, n, false, true),
            Some(s) => {
                anyhow::ensure!(n % 4 == 0 && n <= crate::tp_xport::EPOCH_FLOATS && out != local,
                                "single-stage decode reduce: n {n} is not a float4-clean single-epoch payload / out aliases local");
                let pf = self.take_pf_hint();
                xqlaunch!(l, "xq_tp_wait_add_dec_single", (d.blocks, 1, 1), (256, 1, 1), 0,
                          (s.ctx, out, local, n as i32, pf[0], pf[1] as u32, pf[2], pf[3] as u32))
            }
        }
    }

    /// TP-G: this reduce takes the folded decode path (decode-sized, float4-clean). TP-4C: at every attached
    /// world (2 or 4) — `dec_reduce_finish` runs the `rounds` exchanges.
    pub(super) fn dec_ok(&self, n: usize) -> Option<&DecXport> {
        self.dec.as_ref().filter(|_| n % 4 == 0 && n * 4 <= XPORT_MIN_BYTES)
    }
}
