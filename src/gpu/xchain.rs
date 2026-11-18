//! Cross-chain capture, state taps and the dump formats (`XChain2Dump`, ctx / fold / mtpx / prfx dumps) used by the
//! xchain probes. Inert unless armed; the default serving path is byte-identical. Neighbours: see the map in `mod.rs`.

use super::*;

/// Cross-chain capture (Phase A quality-fix probe, --probe-mxfp4-xchain): when
/// --mxfp4-xchain-capture is set, `forward_batch_dev`/`forward_decode_gpu` record per-layer
/// pre-GEMM normed inputs, per-layer residuals, the final hidden, and the raw logits into a
/// process-global sink the probe drains. Inert otherwise (a set-env check + list push per call)
/// — the default path is byte-identical.
#[derive(Default)]
pub struct XChainCapture {
    /// Number of tokens in this forward (prefill = prompt len; decode steps = 1).
    pub batch: usize,
    /// [layer] the normed hidden feeding that layer's GEMMs (bf16-origin f32).
    pub layer_inputs: Vec<Vec<f32>>,
    /// [layer] the residual stream after that layer's FFN add.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Post-final-norm hidden.
    pub final_hidden: Vec<f32>,
    /// E13: this capture is the MTP head's draft forward (mtp_forward), not a main-model pass.
    pub mtp_head: bool,
    /// Raw (pre-penalty) LM-head logits, captured in forward_decode_gpu.
    pub logits: Vec<f32>,
    /// R9: the PREFILL last-position logits (the greedy first-token input), captured in
    /// prefill_batch. A cross-rank match here proves prefill is bit-identical END TO END — which
    /// isolates the decode-side divergence to the recurrent GDN state, not the sampled first token.
    pub prefill_logits: Vec<f32>,
}

pub(super) static XCHAIN: std::sync::OnceLock<parking_lot::Mutex<Vec<XChainCapture>>> = std::sync::OnceLock::new();
/// Set by --probe-mxfp4-xchain: disable the 8-entry retention trim so a full pass's captures
/// survive to `xchain_capture_take`. Serving never sets it (R9 memory bound stays active there).
pub(super) static XCHAIN_UNBOUNDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// S-B15: count of verify-graph REPLAYS (launches of a captured verify graph). Turns "verify
/// graphs are ON" from a behavioural proof (absence of a disable flag) into a direct one: a run
/// that drafted/verified through the captured graph reports replays > 0 (tp_trace_dump prints it).
pub static VERIFY_GRAPH_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// R9: while a CUDA-graph capture is open (decode/verify graph instantiation), ANY dtoh on the
// capturing stream invalidates it (CUDA_ERROR_STREAM_CAPTURE_*), even one that early-returns —
// the enqueue attempt poisons the stream. So xchain must be fully SUPPRESSED during capture. This
// flag is set around the known capture sites; the decode-time mismatch dump runs after capture is
// long done, so localization is unaffected.
static XCHAIN_SUPPRESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn xchain_set_suppressed(on: bool) { XCHAIN_SUPPRESS.store(on, std::sync::atomic::Ordering::Relaxed); }

// ===== Phase-8 F9 localization: the per-layer residual tap (probe-only) =====
// `verify_state_diff` arms this when --state-tap=1: every `forward_batch_dev` call (decode)
// and every eager `verify_forward_core_topo` call (verify) appends one record, and each layer
// iteration snapshots the post-FFN residual (dtoh — syncs the stream; diagnostics only). The
// probe then compares verify column t against decode step t PER LAYER, so the FIRST divergent
// layer + column localizes the TP-only decode-vs-verify residual (the F9 residual defect).
// Inert unless armed (one relaxed atomic load per call / per layer).
pub struct StateTapCall {
    pub tag: &'static str,
    pub width: usize,
    /// [layer] post-FFN residual, bf16, column-major [col][h] (width columns).
    pub layers: Vec<Vec<half::bf16>>,
    /// (convo, core) per GDN layer in visit order: the post-conv qkv rows and the GDN core out.
    pub gdn: Vec<(Vec<half::bf16>, Vec<half::bf16>)>,
    /// Post-delta_step s_state snapshot per GDN layer (both slots; small widths only).
    pub s: Vec<Vec<f32>>,
}
static STATE_TAP: parking_lot::Mutex<Vec<StateTapCall>> = parking_lot::Mutex::new(Vec::new());
static STATE_TAP_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn state_tap_arm() {
    STATE_TAP_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
    STATE_TAP.lock().clear();
}
pub fn state_tap_disarm() { STATE_TAP_ARMED.store(false, std::sync::atomic::Ordering::SeqCst); }
pub fn state_tap_active() -> bool { STATE_TAP_ARMED.load(std::sync::atomic::Ordering::Relaxed) }
pub fn state_tap_call(tag: &'static str, width: usize) {
    if !state_tap_active() { return; }
    STATE_TAP.lock().push(StateTapCall { tag, width, layers: Vec::new(), gdn: Vec::new(), s: Vec::new() });
}
pub fn state_tap_layer(v: Vec<half::bf16>) {
    if !state_tap_active() { return; }
    if let Some(c) = STATE_TAP.lock().last_mut() { c.layers.push(v); }
}
/// GDN internals: (conv-output qkv rows, core output) for the GDN layer just launched.
pub fn state_tap_gdn(convo: Vec<half::bf16>, core: Vec<half::bf16>) {
    if !state_tap_active() { return; }
    if let Some(c) = STATE_TAP.lock().last_mut() { c.gdn.push((convo, core)); }
}
/// Post-delta s_state snapshot (both slots, local extent) for the GDN layer just launched.
pub fn state_tap_s(s: Vec<f32>) {
    if !state_tap_active() { return; }
    if let Some(c) = STATE_TAP.lock().last_mut() { c.s.push(s); }
}
pub fn state_tap_take() -> Vec<StateTapCall> { std::mem::take(&mut *STATE_TAP.lock()) }

pub(super) fn xchain_active() -> bool {
    if XCHAIN_SUPPRESS.load(std::sync::atomic::Ordering::Relaxed) { return false; }
    crate::opts::var(crate::opt!("mxfp4-xchain-capture")).is_ok()
        || crate::opts::var(crate::opt!("tp-diag")).is_ok()   // R9: diag arms the sink so the agree-mismatch
                                                   // dump can localize the divergent layer
}

pub(super) fn xchain_new(batch: usize) {
    let mut g = XCHAIN.get_or_init(Default::default).lock();
    // R9 serving use: the sink would otherwise grow by one entry (64 layers × h f32) per decode
    // step until the agree-mismatch dump. Keep only the FIRST prefill entry plus the FIRST decode
    // entry (enough to localize the divergent layer); the agree aborts long before more is needed.
    // R9: cap at 8 entries (prefill + the first several decode forwards) so the mismatch dump
    // localizes the divergent step without unbounded host memory. Keep the prefill entry always.
    // The --probe-mxfp4-xchain instrument needs EVERY decode step of its pass (it asserts
    // caps == max_new) — the probe flips XCHAIN_UNBOUNDED and takes the memory hit itself.
    if g.len() >= 8 && !XCHAIN_UNBOUNDED.load(std::sync::atomic::Ordering::Relaxed) {
        let mut retained: Vec<XChainCapture> = Vec::new();
        if let Some(i) = g.iter().position(|c| c.batch > 1) { retained.push(g.remove(i)); }
        for c in g.drain(..).take(7) { retained.push(c); }
        *g = retained;
    }
    g.push(XChainCapture { batch, ..Default::default() });
}
/// Apply `f` to the most recent capture entry (the forward currently running). Single-threaded
/// probe use; the lock is held only for the duration of the closure.
pub(super) fn xchain_capture<F: FnOnce(&mut XChainCapture)>(f: F) {
    let mut g = XCHAIN.get_or_init(Default::default).lock();
    if let Some(last) = g.last_mut() { f(last); }
}
/// dtoh of a bf16 device slice (bucketed pool slice) as f32, truncated to `n` — SYNCHRONIZED
/// copy on the COMPUTE stream. The old version enqueued an async copy and dropped the host
/// buffer at return while the copy was still in flight — the GPU wrote into freed heap and the
/// probe died at exit with `free(): invalid next size` (deterministic on the 27B xchain probe).
/// The sync is safe for the prefill/decode capture paths. The VERIFY path holds an implicit
/// stream capture open (its compute-stream htods run while capturing): the sync then fails with
/// "operation not permitted" — fall back to the async pattern, but KEEP the host buffer alive in
/// a static stash until the probe's final sync, so the pending copy never writes into freed heap.
pub(super) fn xchain_dtoh_f32(gpu: &GpuModel, s: &B, n: usize) -> Vec<f32> {
    let mut host: Vec<half::bf16> = vec![half::bf16::from_f32(0.0); s.len()];
    // R9: the dtoh enqueue is itself ILLEGAL while a stream capture is open (capture_decode_graph,
    // verify graphs). Rather than panic (the old .unwrap() turned that into a destructor abort),
    // return an EMPTY section — the diff then reads "not captured", never a crash.
    let enqueued = unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut host,
            *s.device_ptr() as cudarc::driver::sys::CUdeviceptr, gpu.stream.stream)
    };
    if enqueued.is_err() { return Vec::new(); }
    let synced = unsafe { cudarc::driver::result::stream::synchronize(gpu.stream.stream) }.is_ok();
    if !synced {
        // Capture-open (verify-path probe capture): the copy completes when the capture runs.
        // Convert best-effort (matches the old behavior) and keep the buffer alive.
        let v: Vec<f32> = host.iter().take(n).map(|b| half::bf16::to_f32(*b)).collect();
        static KEEP: parking_lot::Mutex<Vec<Vec<half::bf16>>> = parking_lot::Mutex::new(Vec::new());
        KEEP.lock().push(host);
        return v;
    }
    host.truncate(n);
    host.iter().map(|b| half::bf16::to_f32(*b)).collect()
}

/// R9 LOCALIZATION DUMP: write the capture sink's per-layer residuals + final hidden + logits to a
/// binary file, rank-suffixed, so the four TP ranks' sinks can be diffed offline to find the FIRST
/// divergent layer. Format: u32 count of sections, then per section (u32 tag, u64 len, f32 data).
/// tag 0 = prefill final_hidden; 1000+li = prefill layer li residual; 2000 = decode final_hidden;
/// 3000 = decode logits. (xchain dumps f32 regardless of the bf16 compute — a diff still isolates
/// the divergent layer even though the bf16 rounding is folded in.)
pub fn xchain_rank_dump(path: &str, rank: i32) -> anyhow::Result<()> {
    use std::io::Write;
    let caps = XCHAIN.get_or_init(Default::default).lock();
    let full = format!("{path}.rank{rank}");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&full)?);
    let mut nsec = 0u32;
    let mut body: Vec<u8> = Vec::new();
    let mut push = |tag: u32, data: &[f32]| {
        body.extend_from_slice(&tag.to_le_bytes());
        body.extend_from_slice(&(data.len() as u64).to_le_bytes());
        for x in data { body.extend_from_slice(&x.to_bits().to_le_bytes()); }
        nsec += 1;
    };
    for cap in caps.iter() {
        let prefill = cap.batch > 1;
        for (li, l) in cap.layer_outputs.iter().enumerate() {
            if !l.is_empty() { push(if prefill { 1000 + li as u32 } else { 4000 + li as u32 }, l); }
        }
        if !cap.final_hidden.is_empty() { push(if prefill { 0 } else { 2000 }, &cap.final_hidden); }
        if !cap.logits.is_empty() { push(3000, &cap.logits); }
        if !cap.prefill_logits.is_empty() { push(5000, &cap.prefill_logits); }   // R9 prefill logits
        // R9: per-layer INPUTS (the normed hidden feeding each layer's GEMMs) — the decode layer
        // residual diverges from layer 0, so the input is what isolates "bad input vs bad compute".
        for (li, l) in cap.layer_inputs.iter().enumerate() {
            if !l.is_empty() { push(if prefill { 6000 + li as u32 } else { 7000 + li as u32 }, l); }
        }
    }
    f.write_all(&nsec.to_le_bytes())?;
    f.write_all(&body)?;
    eprintln!("[xchain-rank-dump] rank {rank}: {nsec} sections -> {full}");
    Ok(())
}

/// R9: how many forwards the capture sink currently holds (0 = the capture never ran — wrong path).
pub fn xchain_sink_len() -> usize { XCHAIN.get_or_init(Default::default).lock().len() }

/// Probe-only: reset the capture sink between passes.
/// Flake hunt (2026-08-26): the fuzz caught identical dual prefills diverging. Compare the two
/// most recent PREFILL captures layer-by-layer (inputs and outputs) and print the first
/// divergent layer + rel-L2 profile, so the panic line names the layer instead of just the token.
pub fn dump_dual_prefill_diff(plen: usize) {
    let g = XCHAIN.get_or_init(Default::default).lock();
    let pf: Vec<&XChainCapture> = g.iter().filter(|c| c.batch == plen && !c.mtp_head).collect();
    if pf.len() < 2 { eprintln!("[dual-diff] need 2 prefill captures, have {}", pf.len()); return; }
    let (a, b) = (pf[pf.len() - 2], pf[pf.len() - 1]);
    eprintln!("[dual-diff] comparing two prefill captures (plen={plen}, {} layers)", a.layer_outputs.len());
    let rel = |x: &[f32], y: &[f32]| -> f64 {
        let mut n = 0f64; let mut d = 0f64;
        for i in 0..x.len().min(y.len()) { let dd = (x[i] - y[i]) as f64; n += dd * dd; d += (x[i] * x[i]) as f64; }
        (n / d.max(1e-300)).sqrt()
    };
    for extra in [("out-row-src", 0usize), ("post-gemv-last", 1usize)] {
        let (name, idx) = extra;
        if a.layer_inputs.len() > idx && b.layer_inputs.len() > idx {
            let eq = a.layer_inputs[idx].iter().zip(&b.layer_inputs[idx])
                .filter(|(x, y)| x.to_bits() == y.to_bits()).count();
            let nan_b = b.layer_inputs[idx].iter().filter(|f| f.to_bits() == 0x7fff0000).count();
            eprintln!("[dual-diff] {name}: bit-identical {eq}/{} b-0x7fff-count={nan_b}", a.layer_inputs[idx].len());
        }
    }
    for li in 0..a.layer_inputs.len().min(b.layer_inputs.len()) {
        let ri = rel(&a.layer_inputs[li], &b.layer_inputs[li]);
        let ro = rel(&a.layer_outputs[li], &b.layer_outputs[li]);
        if ri > 0.0 || ro > 0.0 {
            eprintln!("[dual-diff] FIRST DIVERGENCE at layer {li}: input relL2={ri:.3e} output relL2={ro:.3e}");
            for lj in li..a.layer_outputs.len().min(b.layer_outputs.len()) {
                let rj = rel(&a.layer_outputs[lj], &b.layer_outputs[lj]);
                eprintln!("[dual-diff]   layer {lj}: out relL2={rj:.3e}");
            }
            break;
        }
    }
    let rf = rel(&a.final_hidden, &b.final_hidden);
    let fb_eq = a.final_hidden.iter().zip(&b.final_hidden).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    eprintln!("[dual-diff] GEMV-input `last` row: relL2={rf:.3e} bit-identical {fb_eq}/{}", a.final_hidden.len());
    {
        let h = 5120usize;
        let mut dlo = 0usize; let mut dhi = 0usize; let mut first = None;
        for (i, (x, y)) in a.final_hidden.iter().zip(&b.final_hidden).enumerate() {
            if x.to_bits() != y.to_bits() {
                if i < h { dlo += 1; } else { dhi += 1; }
                if first.is_none() { first = Some(i); }
            }
        }
        eprintln!("[dual-diff]   `last` diffs: real[0..{h})={dlo} slack[{h}..)={dhi} first={first:?}");
        if let Some(i) = first {
            eprintln!("[dual-diff]   bits[{i}]: a={:#06x} b={:#06x}",
                      a.final_hidden.get(i).map(|f| f.to_bits()).unwrap_or(0),
                      b.final_hidden.get(i).map(|f| f.to_bits()).unwrap_or(0));
        }
    }
    let mut ldiff = None;
    if !a.prefill_logits.is_empty() && !b.prefill_logits.is_empty() {
        let rl = rel(&a.prefill_logits, &b.prefill_logits);
        let mut nbad = 0usize;
        for i in 0..a.prefill_logits.len().min(b.prefill_logits.len()) {
            if a.prefill_logits[i] != b.prefill_logits[i] { if ldiff.is_none() { ldiff = Some(i); } nbad += 1; }
        }
        eprintln!("[dual-diff] prefill_logits relL2={rl:.3e} ndiff={nbad}/{} first={ldiff:?}",
                  a.prefill_logits.len().min(b.prefill_logits.len()));
        if let Some(i) = ldiff {
            eprintln!("[dual-diff]   logits[{i}]: a={} b={}  (a[0..3]={:?} b[0..3]={:?})",
                      a.prefill_logits[i], b.prefill_logits[i],
                      &a.prefill_logits[i.min(3)..(i.min(3)+3)], &b.prefill_logits[i.min(3)..(i.min(3)+3)]);
        }
        // argmax of both — the greedy decision each call would make
        let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).map(|(i, _)| i);
        eprintln!("[dual-diff]   argmax a={:?} b={:?}", am(&a.prefill_logits), am(&b.prefill_logits));
    }
}

pub fn xchain_capture_reset() {
    XCHAIN.get_or_init(Default::default).lock().clear();
}
/// Probe-only: drain the capture sink (per forward call, prefill first).
pub fn xchain_capture_take() -> Vec<XChainCapture> {
    XCHAIN.get_or_init(Default::default).lock().drain(..).collect()
}

// Phase-8 F9 xhash: per-column FNV of the FINAL hidden from the decode path
// (forward_decode_logits -> forward_batch_dev) and the verify path (verify_forward).
// Debug-only (--f9-xhash=1); bench_mtp correlates them to find the first bit-divergent
// position (which can precede the first token flip by hundreds of tokens).
pub static F9XHASH: std::sync::Mutex<Vec<(u8, usize, usize, Vec<u64>)>> = std::sync::Mutex::new(Vec::new());
static F9XCALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(super) fn f9_xhash_active() -> bool { crate::opts::var(crate::opt!("f9-xhash")).is_ok() }
pub(super) fn f9_x_new_call() -> usize {
    let id = F9XCALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    F9X_CUR.with(|c| c.set(id)); id
}

/// One-load cross-chain dump (--probe-mxfp4-xchain2): the subset of a pass the cross-load
/// comparison needs. Compact little-endian binary on disk (no serde dep): magic "XC2D" + version
/// + counts, then hcol (h f32), tokens (max_new u32), step1/step_last per-layer outputs
/// (2 × nlayers × h f32), final logits (vocab f32).
#[derive(Clone, Default)]
pub struct XChain2Dump {
    pub plen: usize,
    pub max_new: usize,
    pub nlayers: usize,
    pub h: usize,
    pub vocab: usize,
    pub a0: u32,
    pub hcol: Vec<f32>,
    pub tokens: Vec<u32>,
    pub step1_outputs: Vec<Vec<f32>>,
    pub step_last_outputs: Vec<Vec<f32>>,
    pub final_logits: Vec<f32>,
}

pub fn xchain2_write(path: &str, d: &XChain2Dump) -> anyhow::Result<()> {
    let mut b: Vec<u8> = Vec::with_capacity(64 + d.hcol.len() * 4 + d.tokens.len() * 4
        + 2 * d.nlayers * d.h * 4 + d.final_logits.len() * 4);
    b.extend_from_slice(b"XC2D");
    for v in [1u32, d.plen as u32, d.max_new as u32, d.nlayers as u32, d.h as u32, d.vocab as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&d.a0.to_le_bytes());
    for x in &d.hcol { b.extend_from_slice(&x.to_le_bytes()); }
    for t in &d.tokens { b.extend_from_slice(&t.to_le_bytes()); }
    for l in [&d.step1_outputs, &d.step_last_outputs] {
        for li in l { for x in li { b.extend_from_slice(&x.to_le_bytes()); } }
    }
    for x in &d.final_logits { b.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(path, b)?;
    Ok(())
}

fn xc2_need(b: &[u8], o: &mut usize, n: usize, what: &str, path: &str) -> anyhow::Result<()> {
    if *o + n > b.len() { anyhow::bail!("xchain2 dump '{}' truncated at {what}", path); }
    *o += n;
    Ok(())
}
fn xc2_u32(b: &[u8], n: usize) -> u32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[n..n + 4]);
    u32::from_le_bytes(x)
}
fn xc2_f32(b: &[u8], n: usize) -> f32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[n..n + 4]);
    f32::from_le_bytes(x)
}

fn xc2_take_f(b: &[u8], o: &mut usize, n: usize, path: &str) -> anyhow::Result<Vec<f32>> {
    if *o + n * 4 > b.len() { anyhow::bail!("xchain2 dump '{}' truncated (f32 block)", path); }
    let mut v = Vec::with_capacity(n);
    for i in 0..n { v.push(xc2_f32(b, *o + i * 4)); }
    *o += n * 4;
    Ok(v)
}
fn xc2_take_u(b: &[u8], o: &mut usize, n: usize, path: &str) -> anyhow::Result<Vec<u32>> {
    if *o + n * 4 > b.len() { anyhow::bail!("xchain2 dump '{}' truncated (u32 block)", path); }
    let mut v = Vec::with_capacity(n);
    for i in 0..n { v.push(xc2_u32(b, *o + i * 4)); }
    *o += n * 4;
    Ok(v)
}

/// DFlash context-chain dump (E29-B2): per-step target hiddens at the drafter's 5 conditioning
/// layers + the token chain, written from the xchain capture sink (the hooks in
/// `forward_batch_dev` record per-layer outputs for prefill AND decode steps).
/// Format "DFCTX" (little-endian): magic "DFCTX", version u32=1, plen u32, nsteps u32,
/// h u32, nctx_layers u32=5, tokens (plen+nsteps) u32, then (nsteps+1) × 5 × h f32 —
/// feature 0 = prefill's last prompt position, features 1..nsteps = decode steps 0..nsteps-1,
/// each feature = layer outputs at {1,20,39,58,77} in that order.
pub const DFLASH_CTX_LAYERS: [usize; 5] = [1, 20, 39, 58, 77];

pub fn xchain_ctx_write(path: &str, caps: &[XChainCapture], tokens: &[u32], plen: usize) -> anyhow::Result<()> {
    let nsteps = tokens.len().saturating_sub(plen);
    let prefill = caps.first().ok_or_else(|| anyhow::anyhow!("no prefill capture"))?;
    // TP one-shot path: the prefill is token-by-token (batch==1); the recorder keeps only its
    // final step. Batch path: the prefill entry has batch==plen and the row is plen-1.
    let h = prefill.layer_outputs.first().map(|l| l.len() / prefill.batch.max(1)).unwrap_or(0);
    if h == 0 { anyhow::bail!("prefill capture has no layer outputs"); }
    let nctx = DFLASH_CTX_LAYERS.len();
    let mut b: Vec<u8> = Vec::with_capacity(24 + tokens.len() * 4 + (nsteps + 1) * nctx * h * 4);
    b.extend_from_slice(b"DFCTX");
    for v in [1u32, plen as u32, nsteps as u32, h as u32, nctx as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for t in tokens { b.extend_from_slice(&t.to_le_bytes()); }
    let mut push_feature = |cap: &XChainCapture, row: usize| -> anyhow::Result<()> {
        for &li in &DFLASH_CTX_LAYERS {
            let l = cap.layer_outputs.get(li).ok_or_else(|| anyhow::anyhow!("capture missing layer {li}"))?;
            let off = row * h;
            if off + h > l.len() { anyhow::bail!("capture layer {li} too short ({} rows)", l.len() / h.max(1)); }
            for x in &l[off..off + h] { b.extend_from_slice(&x.to_le_bytes()); }
        }
        Ok(())
    };
    // feature 0: the prefill entry's last position (batch==plen → row plen-1; batch==1 → row 0).
    let prefill_row = if prefill.batch == 1 { 0 } else { plen - 1 };
    push_feature(prefill, prefill_row)?;
    for s in caps.iter().skip(1) {
        push_feature(s, 0)?;
    }
    std::fs::write(path, b)?;
    Ok(())
}

/// E13 fold-vs-nofold cross-chain dump: per-step per-layer outputs (ALL layers) + the token
/// chain, from the xchain capture sink. Format "FOLDX" (little-endian): magic "FOLDX", version
/// u32=1, plen u32, nsteps u32, h u32, nlayers u32, tokens (plen+nsteps) u32, then (nsteps+1) ×
/// nlayers × h f32 — feature 0 = prefill's last position (row plen-1, or row 0 when the prefill
/// capture is batch==1), features 1..nsteps = decode steps 0..nsteps-1. Run the TP head twice —
/// once with --moe-no-fold=1 (reference), once with --moe-fold=1 — then compare with
/// --probe-fold-xchain-compare.
pub fn xchain_fold_write(path: &str, caps: &[XChainCapture], tokens: &[u32], plen: usize) -> anyhow::Result<()> {
    let nsteps = tokens.len().saturating_sub(plen);
    let prefill = caps.first().ok_or_else(|| anyhow::anyhow!("no prefill capture"))?;
    let h = prefill.layer_outputs.first().map(|l| l.len() / prefill.batch.max(1)).unwrap_or(0);
    if h == 0 { anyhow::bail!("prefill capture has no layer outputs"); }
    let nlayers = prefill.layer_outputs.len();
    if nlayers == 0 { anyhow::bail!("prefill capture has no layers"); }
    let mut b: Vec<u8> = Vec::with_capacity(24 + tokens.len() * 4 + (nsteps + 1) * nlayers * h * 4);
    b.extend_from_slice(b"FOLDX");
    for v in [1u32, plen as u32, nsteps as u32, h as u32, nlayers as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for t in tokens { b.extend_from_slice(&t.to_le_bytes()); }
    let mut push_feature = |cap: &XChainCapture, row: usize| -> anyhow::Result<()> {
        for l in &cap.layer_outputs {
            let off = row * h;
            if off + h > l.len() { anyhow::bail!("capture layer too short ({} rows)", l.len() / h.max(1)); }
            for x in &l[off..off + h] { b.extend_from_slice(&x.to_le_bytes()); }
        }
        Ok(())
    };
    let prefill_row = if prefill.batch == 1 { 0 } else { plen - 1 };
    push_feature(prefill, prefill_row)?;
    for s in caps.iter().skip(1) {
        push_feature(s, 0)?;
    }
    std::fs::write(path, b)?;
    Ok(())
}

/// In-memory form of a FOLDX dump.
pub struct FoldXChainDump {
    pub plen: usize,
    pub nsteps: usize,
    pub h: usize,
    pub nlayers: usize,
    pub tokens: Vec<u32>,
    /// nsteps × nlayers × h — feature 0 = prefill final row, 1.. = the decode steps.
    pub outputs: Vec<Vec<f32>>,
}

pub fn fold_xchain_read(path: &str) -> anyhow::Result<FoldXChainDump> {
    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[0..5] != b"FOLDX" { anyhow::bail!("fold dump '{}': bad magic", path); }
    let mut o = 5usize;
    let mut take_u = |n: usize| -> anyhow::Result<Vec<u32>> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n { v.push(xc2_u32(&b, o)); o += 4; }
        Ok(v)
    };
    let hdr = take_u(5)?;
    let (ver, plen, nsteps, h, nlayers) =
        (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize, hdr[3] as usize, hdr[4] as usize);
    if ver != 1 { anyhow::bail!("fold dump '{}': version {ver}", path); }
    let tokens = take_u(plen + nsteps)?;
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(nsteps * nlayers);
    for _ in 0..nsteps * nlayers {
        let mut l = Vec::with_capacity(h);
        for _ in 0..h { l.push(xc2_f32(&b, o)); o += 4; }
        outputs.push(l);
    }
    if o > b.len() { anyhow::bail!("fold dump '{}' truncated", path); }
    Ok(FoldXChainDump { plen, nsteps, h, nlayers, tokens, outputs })
}

/// E13 MTP-head cross-chain dump: per draft step, the head's MLP input + output hiddens.
/// Format "MTPX" (little-endian): magic "MTPX", version u32=1, plen u32, nsteps u32 (= generated
/// tokens), ncap u32 (= batch-1 MTP-head captures), h u32, tokens (plen+nsteps) u32, then
/// ncap × 2 × h f32 (input, output per capture).
pub fn mtpx_write(path: &str, caps: &[XChainCapture], tokens: &[u32], plen: usize) -> anyhow::Result<()> {
    // All MTP-head forwards in order (prime, per-step drafts, re-primes) — the compare aligns by
    // index; the prime phase precedes the first draft in both runs identically. The prompt-prime
    // chunks are batch>1 (len = h·chunk) — only the batch==1 draft/re-prime steps share the h
    // geometry the dump assumes.
    let steps: Vec<&XChainCapture> = caps.iter().filter(|c| c.mtp_head && c.batch == 1).collect();
    let nsteps = tokens.len().saturating_sub(plen);
    let ncap = steps.len();
    let h = steps.first().and_then(|c| c.layer_outputs.first()).map(|l| l.len()).unwrap_or(0);
    if h == 0 { anyhow::bail!("no MTP-head captures with outputs"); }
    let mut b: Vec<u8> = Vec::with_capacity(24 + tokens.len() * 4 + ncap * 2 * h * 4);
    b.extend_from_slice(b"MTPX");
    for v in [1u32, plen as u32, nsteps as u32, ncap as u32, h as u32] { b.extend_from_slice(&v.to_le_bytes()); }
    for t in tokens { b.extend_from_slice(&t.to_le_bytes()); }
    for c in &steps {
        for x in &c.layer_inputs[0] { b.extend_from_slice(&x.to_le_bytes()); }
        for x in &c.layer_outputs[0] { b.extend_from_slice(&x.to_le_bytes()); }
    }
    std::fs::write(path, b)?;
    Ok(())
}

/// In-memory form of an MTPX dump.
pub struct MtpxDump {
    pub plen: usize,
    pub nsteps: usize,
    pub ncap: usize,
    pub h: usize,
    pub tokens: Vec<u32>,
    /// ncap × 2 × h — (input, output) per MTP-head capture, in call order.
    pub io: Vec<Vec<f32>>,
}

pub fn mtpx_read(path: &str) -> anyhow::Result<MtpxDump> {
    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[0..4] != b"MTPX" { anyhow::bail!("mtpx dump '{}': bad magic", path); }
    let mut o = 4usize;
    let mut take_u = |n: usize| -> anyhow::Result<Vec<u32>> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n { v.push(xc2_u32(&b, o)); o += 4; }
        Ok(v)
    };
    let hdr = take_u(5)?;
    let (ver, plen, nsteps, ncap, h) =
        (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize, hdr[3] as usize, hdr[4] as usize);
    if ver != 1 { anyhow::bail!("mtpx dump '{}': version {ver}", path); }
    let tokens = take_u(plen + nsteps)?;
    let mut io: Vec<Vec<f32>> = Vec::with_capacity(ncap * 2);
    for _ in 0..ncap * 2 {
        let mut l = Vec::with_capacity(h);
        for _ in 0..h { l.push(xc2_f32(&b, o)); o += 4; }
        io.push(l);
    }
    if o > b.len() { anyhow::bail!("mtpx dump '{}' truncated", path); }
    Ok(MtpxDump { plen, nsteps, ncap, h, tokens, io })
}

/// E13 wide-prefill cross-chain dump: the BATCHED prefill capture's per-layer outputs at the
/// last prompt row. Format "PRFX" (little-endian): magic "PRFX", version u32=1, plen u32,
/// nsteps u32 (= generated tokens), h u32, nlayers u32, tokens (plen+nsteps) u32, then
/// nlayers × h f32 (the prefill's final-row layer outputs). The fold's wide-prefill path
/// (batch > MAX_VERIFY) is nondeterministic — two fold-on runs of the same prompt differ;
/// this dump localizes which layers race.
pub fn prfx_write(path: &str, caps: &[XChainCapture], tokens: &[u32], plen: usize) -> anyhow::Result<()> {
    let prefill = caps.iter().find(|c| c.batch > 1 && !c.mtp_head)
        .ok_or_else(|| {
            let inv: Vec<String> = caps.iter().take(8)
                .map(|c| format!("(b={},mtp={},in={},out={})", c.batch, c.mtp_head,
                                  c.layer_inputs.first().map(|l| l.len()).unwrap_or(0),
                                  c.layer_outputs.first().map(|l| l.len()).unwrap_or(0)))
                .collect();
            anyhow::anyhow!("no batched main-model prefill capture (batch > 1); sink: {}", inv.join(" "))
        })?;
    let h = prefill.layer_outputs.first().map(|l| l.len() / prefill.batch.max(1)).unwrap_or(0);
    if h == 0 { anyhow::bail!("prefill capture has no layer outputs"); }
    let nlayers = prefill.layer_outputs.len();
    let nsteps = tokens.len().saturating_sub(plen);
    let row = plen - 1;
    let mut b: Vec<u8> = Vec::with_capacity(24 + tokens.len() * 4 + nlayers * h * 4);
    b.extend_from_slice(b"PRFX");
    for v in [1u32, plen as u32, nsteps as u32, h as u32, nlayers as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for t in tokens { b.extend_from_slice(&t.to_le_bytes()); }
    for l in &prefill.layer_outputs {
        let off = row * h;
        if off + h > l.len() {
            anyhow::bail!("prefill capture layer too short: {} rows (batch={}, h={}, plen={}, row={})",
                          l.len() / h.max(1), prefill.batch, h, plen, row);
        }
        for x in &l[off..off + h] { b.extend_from_slice(&x.to_le_bytes()); }
    }
    std::fs::write(path, b)?;
    Ok(())
}

/// In-memory form of a PRFX dump.
pub struct PrfxDump {
    pub plen: usize,
    pub nsteps: usize,
    pub h: usize,
    pub nlayers: usize,
    pub tokens: Vec<u32>,
    /// nlayers × h — the prefill's final-row layer outputs.
    pub outputs: Vec<Vec<f32>>,
}

pub fn prfx_read(path: &str) -> anyhow::Result<PrfxDump> {
    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[0..4] != b"PRFX" { anyhow::bail!("prfx dump '{}': bad magic", path); }
    let mut o = 4usize;
    let mut take_u = |n: usize| -> anyhow::Result<Vec<u32>> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n { v.push(xc2_u32(&b, o)); o += 4; }
        Ok(v)
    };
    let hdr = take_u(5)?;
    let (ver, plen, nsteps, h, nlayers) =
        (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize, hdr[3] as usize, hdr[4] as usize);
    if ver != 1 { anyhow::bail!("prfx dump '{}': version {ver}", path); }
    let tokens = take_u(plen + nsteps)?;
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(nlayers);
    for _ in 0..nlayers {
        let mut l = Vec::with_capacity(h);
        for _ in 0..h { l.push(xc2_f32(&b, o)); o += 4; }
        outputs.push(l);
    }
    if o > b.len() { anyhow::bail!("prfx dump '{}' truncated", path); }
    Ok(PrfxDump { plen, nsteps, h, nlayers, tokens, outputs })
}

pub fn xchain2_read(path: &str) -> anyhow::Result<XChain2Dump> {    let b = std::fs::read(path)?;
    if b.len() < 24 || &b[0..4] != b"XC2D" { anyhow::bail!("xchain2 dump '{}': bad magic", path); }
    let mut o = 4usize;
    xc2_need(&b, &mut o, 4, "version", path)?;
    let _ver = xc2_u32(&b, o - 4);
    xc2_need(&b, &mut o, 4, "plen", path)?;
    let plen = xc2_u32(&b, o - 4) as usize;
    xc2_need(&b, &mut o, 4, "max_new", path)?;
    let max_new = xc2_u32(&b, o - 4) as usize;
    xc2_need(&b, &mut o, 4, "nlayers", path)?;
    let nlayers = xc2_u32(&b, o - 4) as usize;
    xc2_need(&b, &mut o, 4, "h", path)?;
    let h = xc2_u32(&b, o - 4) as usize;
    xc2_need(&b, &mut o, 4, "vocab", path)?;
    let vocab = xc2_u32(&b, o - 4) as usize;
    xc2_need(&b, &mut o, 4, "a0", path)?;
    let a0 = xc2_u32(&b, o - 4);
    let hcol = xc2_take_f(&b, &mut o, h, path)?;
    let tokens = xc2_take_u(&b, &mut o, max_new, path)?;
    let mut step1_outputs = Vec::with_capacity(nlayers);
    for _ in 0..nlayers { step1_outputs.push(xc2_take_f(&b, &mut o, h, path)?); }
    let mut step_last_outputs = Vec::with_capacity(nlayers);
    for _ in 0..nlayers { step_last_outputs.push(xc2_take_f(&b, &mut o, h, path)?); }
    let final_logits = xc2_take_f(&b, &mut o, vocab, path)?;
    Ok(XChain2Dump { plen, max_new, nlayers, h, vocab, a0, hcol, tokens,
                     step1_outputs, step_last_outputs, final_logits })
}
