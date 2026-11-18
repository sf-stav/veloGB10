//! S5: the Naive family's tensor-parallel transport — the engine's RDMA doorbell all-reduce
//! (`tp_xport` K1 publish / K2 wait-and-add, serial single-block pairs on the model's own stream),
//! fp32 in place, canonical summation order on every rank (so the residual is bit-identical across
//! ranks and the head's sampled token is reproducible on the node). World 2 at S5; world 4 (the
//! recursive-doubling rounds need the round-phase alignment the EXL3 path carries) is S9.
//!
//! Lever 1 (perf review §5 #1, 2026-10-20): at world 4 every decode-width reduce (a plain step's 96
//! partial sums, a verify block's rows, the sampler's exchange phases, the sharded drafter's hidden) is ONE
//! all-peers stage on a dedicated uniform one-shot ctx (`--tp-reduce auto|single`; the EXL3 path's TP-4F2
//! mechanism), instead of the two recursive-doubling stages: half the link round-trips per reduce. The K2
//! (`tp_wait_add_4way_f32`) forms `(p0 + p2) + (p1 + p3)`, exactly the association of the aligned two-stage
//! tree, so every served number is bitwise what the tree produced. Reduces wider than the single ctx's slot
//! (prefill chunks) stay on the tree over the primary link; world 2 has one stage already and is untouched.
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// The doorbell kernels load under their own module name (never the gpu.rs or EXL3 names).
const TP_MODULE: &str = "naive_tp";
/// The proxy thread's core (the engine's pinned-proxy rule).
const TP_PROXY_CORE: i32 = 19;
/// Lever 1: the single-stage ctx's kernels load under their own module name (the K2 is not in
/// `tp_xport::KERNEL_NAMES`), and its proxy takes the core the EXL3 path's single-stage proxy uses (a big
/// X925 no other thread of this path is pinned to; 19 = the primary proxy).
const SINGLE_MODULE: &str = "naive_tp_single";
const SINGLE_K2: &str = "tp_wait_add_4way_f32";
/// Lever 1b: the K2 of the single-stage ctx under the v2 receive mode (`--tp-gpu-recv on`): the proxy skips its CPU
/// receive stage and the kernel validates every peer's NIC-written tail itself (no CPU bounce per reduce).
const SINGLE_K2_GPU: &str = "tp_wait_add_4way_f32g";
/// Lever 1c: the multi-block K1 of the single-stage ctx (`--naive-tp-k1 multi`, the default): the payload copy into
/// the mapped send ring spread over several blocks with 16 B stores (the single-block K1 took ~14 us per reduce).
const SINGLE_K1_MB: &str = "tp_gate_copy_signal_os_mb";

/// Lever 1c: `--naive-tp-k1 multi|single` (default multi): the pure resolution.
pub fn k1_multi_resolve(raw: Option<&str>) -> bool {
    !matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("single"))
}

/// Lever 1c: the multi-block K1's grid for an `n`-float payload: one 16 B store per thread per block of 256, at
/// most 32 blocks (a 16-row verify block = 64 Ki floats -> 16 Ki uint4 -> 32 blocks of 2 stores per thread).
pub fn k1_mb_blocks(n: usize) -> u32 { ((n / 4).div_ceil(256)).clamp(1, 32) as u32 }
const TP_SINGLE_PROXY_CORE: i32 = 17;
/// Lever 1: the single-stage ctx's ring-slot capacity = its payload bound: MAX_ROWS (16) rows x 4096 x 4 B =
/// 256 KiB covers a verify block, a sampler phase and the drafter's exchange; a reduce above it (a prefill
/// chunk: 1024 rows) takes the tree on the primary link. The bound is ALSO the K1 / proxy agreement rule
/// (K1's qp_mask says "all peers" only for `len <= payload_bytes`; the uniform proxy always posts to all
/// peers), so a reduce is routed here only when `n * 4 <= payload`.
pub const SINGLE_CAP_BYTES: usize = 256 * 1024;

/// Lever 1b: the PURE resolution of `--tp-gpu-recv` for the Naive single-stage ctx: an explicit on/1/true selects the
/// GPU-validated receive (the proxy never validates or releases on this ctx); anything else = the CPU bounce.
pub fn gpu_recv_resolve(raw: Option<&str>) -> bool {
    matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1") | Some("on") | Some("true"))
}

/// ISSUES 100: reduces of more than this many bytes take the pipelined transport (prefill chunks: 1024 rows = 16 MiB);
/// the decode-width ones keep the serial pair / the single stage. The EXL3 path's threshold (XPORT_MIN_BYTES).
pub const PIPE_MIN_BYTES: usize = 256 * 1024;

/// ISSUES 100: `--naive-tp-prefill-xport pipe|serial` (default pipe): the pure resolution.
pub fn prefill_pipe_resolve(raw: Option<&str>) -> bool {
    !matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("serial"))
}

/// Lever 1: the dedicated single-stage reduce ctx (world 4 only).
struct Single {
    ctx: u64,
    k2: CudaFunction,
    /// lever 1b: the K2 is the GPU-validated one (`--tp-gpu-recv on`)
    gpu: bool,
    /// lever 1c: the multi-block K1 and its device arrive counter (None = the single-block K1)
    k1m: Option<(CudaFunction, CudaSlice<u32>)>,
    /// the payload bound in bytes (= `SINGLE_CAP_BYTES`, what `set_payload` installed)
    payload: usize,
}

/// Lever 1: the PURE resolution of `--tp-reduce` for the Naive path: `(single on, why)`. `auto` (and an unset
/// flag) = single at world 4; `rd` = the tree everywhere; `single` at world 2 FALLS BACK to the tree with a
/// reason (world 2 already reduces in one stage: nothing to fuse, nothing to refuse — AGENTS §1b-bis 8).
pub fn reduce_single_resolve(raw: Option<&str>, world: i32) -> (bool, String) {
    let v = raw.map(|v| v.trim().to_ascii_lowercase());
    match (v.as_deref(), world) {
        (Some("rd"), _) => (false, "--tp-reduce rd: the recursive-doubling tree".to_string()),
        (Some("single"), 4) => (true, "--tp-reduce single".to_string()),
        (Some("single"), w) => (false, format!("--tp-reduce single asked, world {w} reduces in ONE stage already: nothing to fuse (the tree runs)")),
        (_, 4) => (true, "--tp-reduce auto = single at world 4".to_string()),
        (_, w) => (false, format!("--tp-reduce auto = the tree at world {w}")),
    }
}

pub struct NaiveTp {
    pub rank: i32,
    pub world: i32,
    kind: Kind,
    /// fp32 scratch for the partial sums ([MAX_ROWS][H] + the MoE combine)
    pub part: CudaSlice<f32>,
    /// the host addresses of the link ctxs this rank reduces over (the primary, the single-stage one): their abort
    /// words (empty = the loopback transport, which cannot abort)
    links: Vec<usize>,
}

enum Kind {
    /// the RDMA doorbell link (two or four boxes); `single` = lever 1's dedicated one-shot ctx (world 4, on);
    /// `pipe` = the pipelined multi-block transport for reduces wider than `PIPE_MIN_BYTES` (prefill chunks;
    /// ISSUES 100, `--naive-tp-prefill-xport pipe`, the EXL3 path's TP-F mechanism on the primary link)
    Rdma { ctx: u64, k1: CudaFunction, k2: CudaFunction, single: Option<Single>, pipe: Option<(crate::tp_xport::Kernels, crate::tp_xport::Pipe)>,
           /// lever 1c on the tree link (TP2, and TP4 under `--tp-reduce rd`): the multi-block K1 for decode-width epochs
           k1m_tree: Option<(CudaFunction, CudaSlice<u32>)> },
    /// one box, W threads: the partials cross host channels (a development / gate transport — the
    /// canonical rank-ascending sum, so it must give bit-identical results to the link at world 2)
    Loop { txs: Vec<std::sync::mpsc::Sender<Vec<f32>>>, rx: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<f32>>>, dev: Arc<CudaDevice> },
}

impl NaiveTp {
    /// A pair of loopback transports for two shards on ONE device (rank 0, rank 1).
    pub fn loopback_pair(dev: &Arc<CudaDevice>, h: usize, max_rows: usize) -> Result<(NaiveTp, NaiveTp)> {
        let mut g = Self::loopback_group(dev, h, max_rows, 2)?;
        let b = g.pop().unwrap(); let a = g.pop().unwrap();
        Ok((a, b))
    }

    /// `world` loopback transports for `world` shards on ONE device (ranks 0..world).
    pub fn loopback_group(dev: &Arc<CudaDevice>, h: usize, max_rows: usize, world: usize) -> Result<Vec<NaiveTp>> {
        let mut txs = Vec::new(); let mut rxs = Vec::new();
        for _ in 0..world { let (t, r) = std::sync::mpsc::channel::<Vec<f32>>(); txs.push(t); rxs.push(r); }
        let mut out = Vec::new();
        for (rank, rx) in rxs.into_iter().enumerate() {
            out.push(NaiveTp { rank: rank as i32, world: world as i32,
                               kind: Kind::Loop { txs: txs.clone(), rx: std::sync::Mutex::new(rx), dev: dev.clone() },
                               part: dev.alloc_zeros::<f32>(max_rows * h)?, links: Vec::new() });
        }
        Ok(out)
    }

    /// Attach the brought-up link: sanity + branch agree, the doorbell kernels, the proxy thread.
    pub fn attach(dev: &Arc<CudaDevice>, mut ctx: crate::tp::TpContext, h: usize, max_rows: usize) -> Result<Self> {
        ensure!(ctx.world == 2 || ctx.world == 4, "NAIVE TP: world {} (2 or 4)", ctx.world);
        ensure!(!(ctx.world > 2 && ctx.link.oneshot_on()), "NAIVE TP world {}: the one-shot all-peers transport is not taken (recursive doubling only)", ctx.world);
        ctx.sanity()?;
        ctx.branch_check(&crate::tp::TpBranch::Naive)?;
        let (rank, world, mut link) = ctx.into_parts();
        let kern = crate::tp_xport::Kernels::load(dev, TP_MODULE)?;
        let f = |n: &str| dev.get_func(TP_MODULE, n).with_context(|| format!("doorbell kernel {n} missing"));
        let (k1, k2) = (f("tp_gate_copy_signal")?, f("tp_wait_add")?);
        link.set_payload(h * 4, true).context("net_set_payload")?;
        let cptr = link.ctx_device_ptr();
        let addr = link.ctx_addr();
        // the proxy OWNS the transport from here (the engine's attach contract); the OS reclaims at exit
        std::mem::forget(link);
        crate::net::spawn_proxy(addr, TP_PROXY_CORE);
        let part = dev.alloc_zeros::<f32>(max_rows * h)?;
        dev.synchronize()?;
        let mut single_addr = 0usize;
        if world > 2 {
            // the engine's R10 rule (exl3 xtp `align_round_phase`): a logical all-reduce at world 4 is 2 epochs
            // (round 0: partner ^1, round 1: partner ^2) and epoch e runs round e % 2, so the device epoch
            // counter must be a multiple of 2 before the first reduce on EVERY rank; a fresh ctx is at 0 —
            // this is checked, and padded with empty exchanges if anything ever consumed an odd count first
            let rounds = (world as u32).trailing_zeros() as u64;
            let e0 = crate::net::traced_device_epoch();
            let pad = ((rounds - e0 % rounds) % rounds) as usize;
            if pad > 0 {
                let stream = dev.fork_default_stream()?;
                let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                let p = *part.device_ptr() as u64;
                for _ in 0..pad {
                    unsafe { k1.clone().launch_on_stream(&stream, c1, (cptr, p, 16u32)) }?;
                    unsafe { k2.clone().launch_on_stream(&stream, c1, (cptr, p, p, 4i32, 2i32)) }?;
                }
                unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
            }
            let e1 = crate::net::traced_device_epoch();
            ensure!(e1 % rounds == 0, "NAIVE TP: device epoch {e1} not a multiple of {rounds} after alignment");
            crate::dprintln!("[naive-tp] rank {rank}/{world}: round phase aligned (epoch {e0} -> {e1}, {pad} pad exchange(s))");
        }
        crate::dprintln!("[naive-tp] rank {rank}/{world} attached: serial doorbell all-reduce ({} round(s) per reduce), fp32 partials, proxy on core {TP_PROXY_CORE}",
                         (world as u32).trailing_zeros());
        // Lever 1: the single-stage ctx. Brought up AFTER the primary link's proxy runs and its round phase is aligned
        // (the same sequence on every rank, so the N-way handshakes pair), as a full world-4 N-way link on the primary
        // device, uniform one-shot (its own epoch counter: the primary's round phase never sees it), its abort mirrored
        // into the primary's inside its proxy (a graph replay never returns to the host between reduces).
        let (single_on, why) = reduce_single_resolve(crate::opts::var(crate::opt!("tp-reduce")).ok().as_deref(), world);
        crate::metrics::NAIVE_TP_PRIMARY_CTX.store(addr, std::sync::atomic::Ordering::Relaxed);
        let single = if single_on {
            let base = crate::exl3_forward::xtp::single_nway_base(world)?;
            let ips = crate::tp::resolve_topology(world)?;
            let dev1 = crate::tp::rdma_dev();
            let mut l3 = crate::net::TpLink::connect_nway_uniform(rank, world, &ips, base, &dev1,
                                                                  crate::tp::resolve_gid_idx(&dev1), SINGLE_CAP_BYTES)
                .with_context(|| format!("NAIVE TP lever 1: single-stage ctx ({dev1}, TCP base {base}, world {world})"))?;
            l3.set_payload(SINGLE_CAP_BYTES, true).context("single-stage ctx net_set_payload")?;
            let gpu = gpu_recv_resolve(crate::opts::var(crate::opt!("tp-gpu-recv")).ok().as_deref());
            if gpu { l3.set_recv_mode(true).context("single-stage ctx net_set_recv_mode")?; }
            l3.set_parent(addr);
            let ctx3 = l3.ctx_device_ptr();
            let a3 = l3.ctx_addr();
            single_addr = a3;
            std::mem::forget(l3);
            crate::net::spawn_proxy_aux(a3, TP_SINGLE_PROXY_CORE);
            if dev.get_func(SINGLE_MODULE, SINGLE_K2).is_none() {
                let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_BATCH);
                dev.load_ptx(ptx, SINGLE_MODULE, &[SINGLE_K2, SINGLE_K2_GPU, SINGLE_K1_MB]).context("load the single-stage kernels (gpu_batch.ptx)")?;
            }
            let name = if gpu { SINGLE_K2_GPU } else { SINGLE_K2 };
            let k2s = dev.get_func(SINGLE_MODULE, name).with_context(|| format!("doorbell kernel {name} missing"))?;
            let k1m = if k1_multi_resolve(crate::opts::var(crate::opt!("naive-tp-k1")).ok().as_deref()) {
                let f = dev.get_func(SINGLE_MODULE, SINGLE_K1_MB).with_context(|| format!("doorbell kernel {SINGLE_K1_MB} missing"))?;
                let mut arrive = dev.alloc_zeros::<u32>(1)?;
                dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2: alloc_zeros does not zero
                Some((f, arrive))
            } else { None };
            crate::metrics::NAIVE_TP_SINGLE_CTX.store(a3, std::sync::atomic::Ordering::Relaxed);
            crate::metrics::NAIVE_TP_SINGLE_GPU_RECV.store(gpu, std::sync::atomic::Ordering::Relaxed);
            crate::metrics::NAIVE_TP_SINGLE_K1_MB.store(k1m.is_some(), std::sync::atomic::Ordering::Relaxed);
            Some(Single { ctx: ctx3, k2: k2s, gpu, k1m, payload: SINGLE_CAP_BYTES })
        } else { None };
        // ISSUES 100: the pipelined prefill transport on the primary link (one rail; lookahead 4, 16 blocks: the EXL3
        // defaults). It keeps the R10 round phase (every device epoch has its round; fill bubbles), and its per-element
        // sums are the serial schedule's (lower + upper per exchange, the same association), so the output is bitwise.
        let pipe = if prefill_pipe_resolve(crate::opts::var(crate::opt!("naive-tp-prefill-xport")).ok().as_deref()) {
            Some((kern, crate::tp_xport::Pipe::new(dev, vec![cptr], world as usize, 4, 16).context("pipelined prefill transport")?))
        } else { None };
        if rank == 0 {
            println!("[naive-tp] rank 0/{world} prefill reduces (> {} KiB): {}", PIPE_MIN_BYTES / 1024,
                     if pipe.is_some() { "PIPELINED multi-block transport (lookahead 4, 16 blocks; --naive-tp-prefill-xport pipe)" }
                     else { "serial one-block pair, 1 MiB epochs (--naive-tp-prefill-xport serial)" });
        }
        // lever 1c on the tree link: `tp_gate_copy_signal_mb` (the TP-F multi-block K1: same reuse gate on the epoch's partner
        // QP, same tail / len / gpu_ready publication, so the unchanged K2 consumes it) for the decode-width epochs
        let k1m_tree = if k1_multi_resolve(crate::opts::var(crate::opt!("naive-tp-k1")).ok().as_deref()) {
            let f = dev.get_func(TP_MODULE, "tp_gate_copy_signal_mb").context("doorbell kernel tp_gate_copy_signal_mb missing")?;
            let mut arrive = dev.alloc_zeros::<u32>(1)?;
            dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2
            Some((f, arrive))
        } else { None };
        let links = [addr, single_addr].into_iter().filter(|a| *a != 0).collect();
        let tp = NaiveTp { rank, world, kind: Kind::Rdma { ctx: cptr, k1, k2, single, pipe, k1m_tree }, part, links };
        if single_on {
            // one reduce through the new ctx before anything else rides it: the canonical sum of small integers is
            // exact in fp32, so a single wrong float is a transport fault, not an association difference
            let stream = crate::gpu::fork_blocking_stream(dev);
            let bad = tp.self_test(dev, &stream, 4096).context("single-stage ctx self-test")?;
            ensure!(bad == 0, "NAIVE TP lever 1: single-stage ctx self-test: {bad} wrong floats of 4096 (rank {rank})");
            let gpu = matches!(&tp.kind, Kind::Rdma { single: Some(Single { gpu: true, .. }), .. });
            let line = format!("[naive-tp] rank {rank}/{world} reduce: ONE all-peers stage on a dedicated one-shot ctx ({why}): fp32 wire, \
                                sum (p0+p2)+(p1+p3) = bitwise the two-stage tree; reduces up to {} KiB take it (a plain step, a verify block, \
                                the sampler and drafter exchanges), wider ones (prefill chunks) stay on the tree; receive {}; K1 {}; proxy on core {TP_SINGLE_PROXY_CORE}, \
                                TCP base {}; self-test 4096 floats OK", SINGLE_CAP_BYTES / 1024,
                               if gpu { "GPU-validated (--tp-gpu-recv on: the K2 proves every peer's tail itself, no CPU bounce)" } else { "CPU bounce (the proxy validates the tails and releases cpu_done; --tp-gpu-recv on removes it)" },
                               if matches!(&tp.kind, Kind::Rdma { single: Some(Single { k1m: Some(_), .. }), .. }) { "multi-block (--naive-tp-k1 multi)" } else { "single-block (--naive-tp-k1 single)" },
                               crate::exl3_forward::xtp::single_nway_base(world)?);
            if rank == 0 { println!("{line}"); } else { crate::dprintln!("{line}"); }
        } else if rank == 0 {
            println!("[naive-tp] rank 0/{world} reduce: the recursive-doubling tree, {} stage(s) per reduce ({why})", (world as u32).trailing_zeros());
        }
        Ok(tp)
    }

    /// Lever 1: is the single-stage ctx live on this transport?
    pub fn single_stage(&self) -> bool { matches!(&self.kind, Kind::Rdma { single: Some(_), .. }) }

    /// Transport self-test: rank r fills `n` floats with (r + 1) * (i % 7), reduces, and checks the
    /// canonical sum sum_r (r + 1) * (i % 7). Returns the number of wrong floats.
    pub fn self_test(&self, dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream, n: usize) -> Result<usize> {
        let w = self.world as usize;
        let mine: Vec<f32> = (0..n).map(|i| (self.rank as f32 + 1.0) * (i % 7) as f32).collect();
        let want: Vec<f32> = (0..n).map(|i| (1..=w).map(|r| r as f32).sum::<f32>() * (i % 7) as f32).collect();
        let buf = dev.htod_sync_copy(&mine)?;
        self.all_reduce(stream, &buf, n)?;
        unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&buf)?;
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        crate::dprintln!("[naive-tp] rank {}: self-test {n} floats, {bad} wrong (got[0..8] {:?}, want {:?})", self.rank, &got[..8.min(n)], &want[..8.min(n)]);
        Ok(bad)
    }

    /// The transport's abort word (0 = healthy). An abort is cooperative (a status word, never a trap): the reduce
    /// kernels see it and no-op, so every reduce after it returns a PARTIAL sum and nothing fails by itself.
    pub fn abort_code(&self) -> u64 {
        self.links.iter().map(|a| crate::net::ctx_abort_status(*a)).find(|c| *c != 0).unwrap_or(0)
    }

    /// Fail loudly once the link has aborted (`what` = where it was noticed).
    pub fn check_link(&self, what: &str) -> Result<()> {
        let c = self.abort_code();
        ensure!(c == 0, "NAIVE TP rank {}: the transport ABORTED (code {c:#x}), noticed {what}: every reduce since returned a partial sum; \
                         the output would be wrong, so this rank stops (the watchdog dump is in /tmp/tp_wdog.rank{}.dump)", self.rank, self.rank);
        Ok(())
    }

    /// In-place fp32 sum all-reduce of the first `n` floats of `buf` (the model's default stream).
    pub fn all_reduce(&self, stream: &cudarc::driver::CudaStream, buf: &CudaSlice<f32>, n: usize) -> Result<()> {
        self.all_reduce_at(stream, buf, 0, n)
    }

    /// In-place fp32 sum all-reduce of buf[off .. off + n).
    pub fn all_reduce_at(&self, stream: &cudarc::driver::CudaStream, buf: &CudaSlice<f32>, off: usize, n: usize) -> Result<()> {
        ensure!(off + n <= buf.len(), "all_reduce: {off} + {n} floats exceed the buffer ({})", buf.len());
        let base = *buf.device_ptr() as u64 + (off * 4) as u64;
        match &self.kind {
            Kind::Rdma { ctx, k1, k2, single, pipe, k1m_tree } => {
                let rounds = (self.world as u32).trailing_zeros() as usize;
                let c1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                if let Some((kern, x)) = pipe {
                    if n * 4 > PIPE_MIN_BYTES && crate::tp_xport::pipe_ok(n) {
                        return crate::tp_xport::reduce_pipe(kern, stream, x, crate::tp_xport::Io::f32_inplace(base), n)
                            .context("pipelined prefill all-reduce");
                    }
                }
                if let Some(s) = single {
                    if n * 4 <= s.payload {
                        // lever 1: ONE K1 (the uniform proxy fans the slot to all 3 peers) + the fp32 4-way K2, in place
                        unsafe {
                            match &s.k1m {
                                Some((k1m, arrive)) => {
                                    ensure!(n % 4 == 0, "single-stage multi-block K1: {n} floats is not a 16 B multiple");
                                    let cm = LaunchConfig { grid_dim: (k1_mb_blocks(n), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                                    k1m.clone().launch_on_stream(stream, cm, (s.ctx, base, (n * 4) as u32, arrive)).context("single-stage K1m launch")?;
                                }
                                None => k1.clone().launch_on_stream(stream, c1, (s.ctx, base, (n * 4) as u32)).context("single-stage K1 launch")?,
                            }
                            s.k2.clone().launch_on_stream(stream, c1, (s.ctx, base, base, n as i32)).context("single-stage K2 launch")?;
                        }
                        return Ok(());
                    }
                }
                for sp in crate::tp_xport::serial_plan(n, rounds) {
                    ensure!(sp.len * 4 + 64 <= crate::tp::TP_SLOT_BYTES, "all-reduce epoch {} floats exceeds one ring slot", sp.len);
                    let q = base + (sp.off * 4) as u64;
                    unsafe {
                        match k1m_tree {
                            Some((k1m, arrive)) if sp.len * 4 <= PIPE_MIN_BYTES && sp.len % 4 == 0 => {
                                let cm = LaunchConfig { grid_dim: (k1_mb_blocks(sp.len), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                                k1m.clone().launch_on_stream(stream, cm, (*ctx, q, (sp.len * 4) as u32, arrive, 0i32)).context("K1m launch")?;
                            }
                            _ => k1.clone().launch_on_stream(stream, c1, (*ctx, q, (sp.len * 4) as u32)).context("K1 launch")?,
                        }
                        k2.clone().launch_on_stream(stream, c1, (*ctx, q, q, sp.len as i32, 2i32)).context("K2 launch")?;
                    }
                }
            }
            Kind::Loop { txs, rx, dev: _ } => {
                // a host round trip: wait for the stream, exchange with every peer, sum in rank order, write back
                unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }.context("loop sync")?;
                let w = self.world as usize;
                let mut mine = vec![0f32; n];
                unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut mine, base) }.context("loop dtoh")?;
                let mut tagged = mine.clone(); tagged.push(self.rank as f32);
                for (r, tx) in txs.iter().enumerate() { if r != self.rank as usize { tx.send(tagged.clone()).context("loopback peer gone")?; } }
                let mut parts: Vec<Option<Vec<f32>>> = vec![None; w];
                parts[self.rank as usize] = Some(mine);
                let rxg = rx.lock().unwrap();
                for _ in 0..w - 1 {
                    let mut v = rxg.recv().context("loopback peer gone")?;
                    let r = v.pop().unwrap() as usize;
                    parts[r] = Some(v);
                }
                drop(rxg);
                // canonical order: rank 0 + rank 1 + ... (the link's world-2 order; at world 4 the link sums
                // (p0+p1)+(p2+p3) per recursive doubling — the gate compares at a tolerance there)
                let mut sum = vec![0f32; n];
                for r in 0..w { for (a, b) in sum.iter_mut().zip(parts[r].as_ref().unwrap()) { *a += *b; } }
                unsafe { cudarc::driver::result::memcpy_htod_sync(base, &sum) }.context("loop htod")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lever 1: `--tp-reduce` on the Naive path: auto/unset = single at world 4 only; rd = the tree; single at world 2 = a
    /// graceful fallback with a reason (never a refusal).
    #[test]
    fn reduce_single_resolution() {
        assert!(reduce_single_resolve(None, 4).0);
        assert!(reduce_single_resolve(Some("auto"), 4).0);
        assert!(reduce_single_resolve(Some(" Single "), 4).0);
        assert!(!reduce_single_resolve(Some("rd"), 4).0);
        assert!(!reduce_single_resolve(None, 2).0);
        assert!(!reduce_single_resolve(Some("single"), 2).0);
        assert!(reduce_single_resolve(Some("single"), 2).1.contains("nothing to fuse"));
        assert!(!reduce_single_resolve(Some("rd"), 2).0);
    }

    /// Lever 1: the K2 launched by name on the served path is defined in gpu_batch.cu and present in the embedded PTX
    /// (the module load list is this file's own `[SINGLE_K2]`, so a missing symbol would only show at the first boot).
    #[test]
    fn single_stage_k2_is_in_the_embedded_ptx() {
        for k in [SINGLE_K2, SINGLE_K2_GPU, SINGLE_K1_MB] {
            assert!(src_text!("kernels/gpu_batch").contains(&format!("void {k}(")), "{k} not defined in gpu_batch.cu");
            assert!(crate::ptx_embed::GPU_BATCH.contains(&format!(".entry {k}(")), "{k} missing from the embedded gpu_batch.ptx");
        }
    }

    /// Lever 1c: the multi-block K1 is the default; only `single` selects the single-block K1. Its grid covers a plain
    /// step (4096 floats: 4 blocks) up to a 16-row verify block (32 blocks), one or two 16 B stores per thread.
    #[test]
    fn k1_multi_resolution_and_grid() {
        assert!(k1_multi_resolve(None) && k1_multi_resolve(Some("multi")) && !k1_multi_resolve(Some("single")));
        assert_eq!(k1_mb_blocks(4096), 4);
        assert_eq!(k1_mb_blocks(16 * 4096), 32);
        assert_eq!(k1_mb_blocks(32), 1);
        assert_eq!(k1_mb_blocks(8 * 4096), 32);
    }

    /// ISSUES 100: the pipelined prefill transport is the default; `serial` keeps the one-block pair. Only reduces above
    /// the decode width take it (a plain step / verify block never does).
    #[test]
    fn prefill_pipe_resolution() {
        assert!(prefill_pipe_resolve(None) && prefill_pipe_resolve(Some("pipe")) && !prefill_pipe_resolve(Some("serial")));
        assert!(crate::naive::model::MAX_ROWS * 4096 * 4 <= PIPE_MIN_BYTES);
        assert!(1024 * 4096 * 4 > PIPE_MIN_BYTES);
    }

    /// Lever 1b: only an explicit on/1/true selects the GPU-validated receive on the single-stage ctx.
    #[test]
    fn gpu_recv_resolution() {
        assert!(gpu_recv_resolve(Some("on")) && gpu_recv_resolve(Some("1")) && gpu_recv_resolve(Some(" True ")));
        assert!(!gpu_recv_resolve(None) && !gpu_recv_resolve(Some("off")) && !gpu_recv_resolve(Some("0")) && !gpu_recv_resolve(Some("auto")));
    }

    /// Lever 1: the routing bound is the payload bound (K1's qp_mask / the uniform proxy agree only under it): a plain
    /// step, a 16-row verify block and the drafter's exchange are under it at h = 4096; a 1024-row prefill chunk is not.
    #[test]
    fn single_stage_bound_covers_decode_width_only() {
        let h = 4096usize;
        assert!(h * 4 <= SINGLE_CAP_BYTES);
        assert!(crate::naive::model::MAX_ROWS * h * 4 <= SINGLE_CAP_BYTES);
        assert!(1024 * h * 4 > SINGLE_CAP_BYTES);
        // the payload must satisfy the shim's rule (8 B multiple, <= slot capacity)
        assert_eq!(SINGLE_CAP_BYTES % 8, 0);
    }
}
