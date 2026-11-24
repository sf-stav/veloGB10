//! DSV4.1 GPU host side of the sparse attention core (leg KA). Kernels: `kernels/gpu_dsv41/60_attn.cuh`
//! (+ 61 tile / 62 core / 63 entries). This file: the Rust mirror of the argument structs, the launch
//! wrappers (decode / verify and prefill), the decode scratch sizing, and the RoPE inverse frequencies of a
//! layer class. CPU references: `attn/reference.rs`; the probe (`--probe-dsv41-kernel --dsv41-kernel-set
//! attn`): `attn/probe.rs`, `attn/fixture.rs`, `attn/timing.rs`.
//!
//! ## What the kernels compute (inference/model.py Attention.forward 766-791, kernel.py sparse_attn)
//! For each query row (position `pos`, lane `lane`) and each of the rank's `heads` heads:
//! keys = the SWA window (positions `win_end - 127 ..= win_end`, FP8 E4M3 + E8M0/32 from the lane's ring
//! of 134 slots, or from linear rows for positions >= `lin_base` during prefill) + the selected compressed
//! rows (`sel`, positions in the layer's KV-source cache: 288-B FP4 E2M1 + E4M3/16 records) + optional
//! extra keys (the DSpark block KV); s = q.k * 512^-0.5 (bf16 operands, fp32 accumulate); online softmax
//! in blocks of 64 keys with the sink in the denominator and P cast to bf16 before P.V; o = bf16(O / L);
//! then, per `epi`, the inverse RoPE on the last 64 dims and act_quant (the WO-A input, bf16 values and
//! optionally the E4M3 codes + E8M0 bytes). See 60_attn.cuh for the canonical per-row arithmetic: a
//! row's result is the same whichever entry, row count, chunking or head range (TP) computed it.
//!
//! ## Entries
//! * `d41_attn_decode`: decode and verify rows (one CTA per row x head group x split, last-arriver merge
//!   in the same launch); needs [`DecodeScratch`].
//! * `d41_attn_prefill`: any number of rows (one CTA per row x head group, all splits in-CTA).
//! Both: 512 threads, `SMEM` bytes of dynamic shared memory (opted in on the raw handle that launches).

pub mod fixture;
pub mod probe;
pub mod reference;
pub mod timing;

use crate::dsv41::weights::{DevPtr, Launch};
use anyhow::{bail, ensure, Context, Result};

/// Entry kernels of this band; the gpu_dsv41 loader (`super::load_module`) collects them.
pub const FNS: &[&str] = &["d41_attn_decode", "d41_attn_decode_2cta", "d41_attn_prefill", "d41_attn_abi", "d41_attn_merge", "d41_attn_recipe", "d41_attn_merge_recipe"];

/// Leg PA1 (`d41_attn_args.xflags`, decode entry): the window keys L2-prefetched before the PDL wait
pub const X_WPF: u32 = 1;
/// Leg PA1: every CTA writes its split state as a slot; `d41_attn_merge` (one CTA per (row, head)) merges in split order and
/// runs the epilogue (+ PD1's wo_a rotation)
pub const X_DEFER: u32 = 2;
/// Leg PA1: q's forward RoPE (the call's `rope_freq`, the row's position) applied while loading q
pub const X_QROPE: u32 = 4;
/// Leg PA1 (`--dsv41-attn-split recipe`, opt-in): the external reference recipe's split rule (`launch_recipe`; DIFFERENT
/// output bits from the exact rule)
pub const X_RECIPE: u32 = 8;
/// Leg P7ATTN (`d41_attn_args.xflags`, decode entry): 256-thread 8-head 2-CTA/SM decode kernel (`d41_attn_decode_2cta`),
/// contiguous `[16][512]` slot layout read by `d41_attn_merge`; bit-for-bit identical to `d41_attn_decode`
pub const X_2CTA: u32 = 16;
/// Leg P7ATTN (host-only override in `launch_decode_x`): force the 512-thread 1-CTA/SM decode kernel (`--dsv41-attn-2cta off`)
pub const X_1CTA: u32 = 32;
/// Heads per CTA of `d41_attn_decode_2cta` (`66_attn_2cta.cuh` `D41_ATTN_2CTA_HG`)
pub const HG_2CTA: usize = 8;
/// Threads per CTA of `d41_attn_decode_2cta` (`66_attn_2cta.cuh` `D41_ATTN_2CTA_THREADS`)
pub const THREADS_2CTA: u32 = 256;
/// Dynamic shared memory per CTA of `d41_attn_decode_2cta` (`66_attn_2cta.cuh` `D41_ATTN_2CTA_SMEM`, <= 48 KB)
pub const SMEM_2CTA: u32 = 46_736;
/// The recipe rule's splits (their decode form: 8 for every row they serve in a decode round of <= 16 rows). OUR flag applies
/// it to EVERY row of every call (decode, verify, prefill chunks, rounds of > 16 rows), in launches of <= `RECIPE_SPLIT_ROWS`
/// rows, so a row's bits never depend on the call's width or kind (batch invariance and decode == prefill hold under the
/// flag). Their engine runs prompt chunks of > 16 rows in ONE pass with other bits; that one-pass form is NOT served here.
pub const RECIPE_SPLITS: usize = 8;
pub const RECIPE_SPLIT_ROWS: usize = 16;
/// d41_attn_recipe's dynamic shared memory (65_attn_recipe.cuh D41R_SMEM)
pub const RECIPE_SMEM: u32 = 38_976;

/// Whether the recipe rule covers a call (its specification has no DSpark extra keys and no window floor; the caller keeps
/// the exact rule otherwise).
pub fn recipe_covers(call: &AttnCall) -> bool { !call.lanes.iter().any(|l| l.ext.map_or(false, |e| e.2 > 0)) && call.sel_cap <= SEL_MAX }

/// Leg PA1: the recipe split rule (opt-in): 8 splits of 32-key blocks dealt round-robin + `d41_attn_merge_recipe`, for
/// every row, in pieces of <= 16 rows (the scratch holds 8 split slots for min(rows, 16) rows). `x` takes X_WPF / X_QROPE.
pub fn launch_recipe(l: Launch, call: &AttnCall, scratch: Option<&DecodeScratch>, rot: Option<&RotOut>, x: u32) -> Result<()> {
    ensure!(recipe_covers(call), "attn recipe rule: DSpark extra keys are not covered (the exact rule serves them)");
    let ns = RECIPE_SPLITS;
    let piece = call.rows.min(RECIPE_SPLIT_ROWS);
    let s = scratch.context("attn recipe rule: a split scratch is needed")?;
    ensure!(s.nsplit >= ns && s.rows >= piece && s.groups >= call.heads.div_ceil(HG), "attn recipe rule: the scratch holds {} rows x {} splits (needs {piece} x {ns})", s.rows, s.nsplit);
    let mut a = call.args(1, None)?;
    a.part = s.part.0; a.cnt = s.cnt.0;
    a.nsplit_max = ns as i32;
    if let Some(rot) = rot {
        for (i, &p) in rot.suh.iter().enumerate() { ensure!(p != 0 && p % 8 == 0, "attn rot_out: suh {i} {p:#x}"); a.rot_suh[i] = p; }
        a.rot_xh = rot.xh; a.rot_ld = rot.ld as i64; a.rot_hpg = rot.hpg as i32;
    }
    a.xflags = (x & (X_WPF | X_QROPE)) as i32;
    RECIPE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut r0 = 0usize;
    while r0 < call.rows {
        let n = (call.rows - r0).min(piece);
        let mut b = a;
        let off = |p: u64, stride_bytes: u64| if p == 0 { 0 } else { p + r0 as u64 * stride_bytes };
        b.rows = n as i32;
        b.q = off(a.q, a.q_stride as u64 * 2);
        b.pos = off(a.pos, 4);
        b.win_end = off(a.win_end, 4);
        b.lane = off(a.lane, 4);
        b.sel = off(a.sel, a.sel_stride as u64 * 4);
        b.n_sel = off(a.n_sel, 4);
        b.out = off(a.out, a.out_stride as u64 * 2);
        b.out_q = off(a.out_q, a.outq_stride as u64);
        b.out_s = off(a.out_s, a.outs_stride as u64);
        b.rot_xh = off(a.rot_xh, a.rot_ld as u64 * 2);
        launch_cfg2("d41_attn_recipe", l, &b, (n * b.groups as usize) as u32, ns as u32, 256, RECIPE_SMEM)?;
        launch_cfg("d41_attn_merge_recipe", l, &b, (n * b.groups as usize * HG) as u32, 128, 0)?;
        r0 += n;
    }
    Ok(())
}
static RECIPE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn recipe_calls() -> u64 { RECIPE_CALLS.load(std::sync::atomic::Ordering::Relaxed) }

/// head_dim (K = V = the 512-d latent)
pub const D: usize = 512;
/// heads per CTA (one m16 MMA tile)
pub const HG: usize = 16;
/// threads per CTA
pub const THREADS: u32 = 512;
/// dynamic shared memory per CTA (61_attn_tile.cuh D41_ATTN_SMEM)
pub const SMEM: u32 = 100_880;
/// floats of one decode split partial (O [16][512], m [16], l [16])
pub const PART_FLOATS: usize = HG * D + 2 * HG;
/// 64-key blocks per split (60_attn.cuh D41_ATTN_SPLIT_BLOCKS: part of the canonical arithmetic)
pub const SPLIT_BLOCKS: usize = 2;
/// blocks per row at most: 2 window + 8 compressed (512 / 64) + 1 extra
pub const MAX_BLOCKS: usize = 11;
/// splits per row at most
pub const MAX_SPLITS: usize = MAX_BLOCKS.div_ceil(SPLIT_BLOCKS);
/// lanes one call can address
pub const MAX_LANES: usize = 8;
/// compressed entries per row at most (index_topk)
pub const SEL_MAX: usize = 512;
/// extra keys per lane at most (one block)
pub const EXT_MAX: usize = 64;
/// epilogue flag: the inverse RoPE on dims 448..511
pub const EPI_ROPE: u32 = 1;
/// epilogue flag: act_quant (E4M3 + E8M0 per 32) -> the WO-A input
pub const EPI_AQ: u32 = 2;
/// kernel error codes (written to the optional error word)
pub const ERR_SEL_RANGE: i32 = 1;

/// One lane's caches for one layer (`d41_attn_lane`). A null pointer = that source is absent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnLane {
    pub swa_q: u64,
    pub swa_s: u64,
    pub lin_q: u64,
    pub lin_s: u64,
    pub kv: u64,
    pub ext_q: u64,
    pub ext_s: u64,
    pub lin_base: i64,
    pub slots: i32,
    pub kv_rows: i32,
    pub n_ext: i32,
    pub pad_: i32,
}

/// The kernel argument block (`d41_attn_args`), passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnArgs {
    pub lanes: [AttnLane; MAX_LANES],
    pub q: u64,
    pub q_stride: i64,
    pub sink: u64,
    pub pos: u64,
    pub win_end: u64,
    pub lane: u64,
    pub sel: u64,
    pub n_sel: u64,
    pub sel_stride: i64,
    pub sel_cap: i32,
    pub heads: i32,
    pub out: u64,
    pub out_stride: i64,
    pub out_q: u64,
    pub outq_stride: i64,
    pub out_s: u64,
    pub outs_stride: i64,
    pub part: u64,
    pub cnt: u64,
    pub err: u64,
    pub rows: i32,
    pub groups: i32,
    pub nsplit_max: i32,
    pub epi: u32,
    pub scale: f32,
    /// window keys below this position are masked (0 = none; leg CE's truncated replay, `launch_prefill_floor`)
    pub win_lo: i32,
    pub rope_freq: [f32; 32],
    /// leg PD1: wo_a's rotated input rows written by the epilogue (`launch_decode_rot`; 0 = off)
    pub rot_suh: [u64; 8],
    pub rot_xh: u64,
    pub rot_ld: i64,
    pub rot_hpg: i32,
    /// leg PA1: `X_*` flags of the decode entry (0 = KA / PV1 / PD1's behaviour)
    pub xflags: i32,
}

/// Size / offset facts in the order `d41_attn_abi` reports them (the probe checks host == device).
pub fn host_abi() -> [u32; 13] {
    let a = std::mem::MaybeUninit::<AttnArgs>::uninit();
    let p = a.as_ptr();
    let b = p as usize;
    // SAFETY: addr_of! on fields of an uninitialised value only computes addresses
    let (q, out, rows, rope, lane1) = unsafe {
        (std::ptr::addr_of!((*p).q) as usize, std::ptr::addr_of!((*p).out) as usize, std::ptr::addr_of!((*p).rows) as usize,
         std::ptr::addr_of!((*p).rope_freq) as usize, std::ptr::addr_of!((*p).lanes[1]) as usize)
    };
    [std::mem::size_of::<AttnLane>() as u32, std::mem::size_of::<AttnArgs>() as u32, (q - b) as u32, (out - b) as u32,
     (rows - b) as u32, (rope - b) as u32, SMEM, THREADS, PART_FLOATS as u32, MAX_SPLITS as u32, MAX_LANES as u32, (lane1 - b) as u32,
     SPLIT_BLOCKS as u32]
}

/// One lane's sources for a call (device pointers; `DevPtr(0)` = absent).
#[derive(Clone, Copy, Debug)]
pub struct LaneSrc {
    /// SWA ring planes (q [slots][512] E4M3, s [slots][16] E8M0), slot = position % slots
    pub swa_q: DevPtr,
    pub swa_s: DevPtr,
    pub slots: usize,
    /// prefill: linear window rows (q, s, lin_base), row i = position `lin_base + i` (same plane format)
    pub lin: Option<(DevPtr, DevPtr, i64)>,
    /// the layer's KV-source cache (288-B records) and its row count; `None` for SWA-only layers / DSpark
    pub kv: Option<(DevPtr, usize)>,
    /// extra keys (DSpark block KV, FP8 planes) and their count (<= 64)
    pub ext: Option<(DevPtr, DevPtr, usize)>,
}

/// Per-row inputs and the outputs of one call. All row arrays are device arrays of `rows` entries.
#[derive(Clone, Debug)]
pub struct AttnCall {
    pub rows: usize,
    /// heads in q / sink / out (the rank's head range: 64 / 32 / 16 / 8)
    pub heads: usize,
    /// bf16 [rows][q_stride], head h at h * 512, after RoPE
    pub q: DevPtr,
    pub q_stride: usize,
    /// f32 [heads]
    pub sink: DevPtr,
    /// i32 [rows] absolute positions (the inverse RoPE)
    pub pos: DevPtr,
    /// i32 [rows] last window position (`DevPtr(0)`: = pos; DSpark rows: the main position)
    pub win_end: DevPtr,
    /// i32 [rows] lane index into `lanes` (`DevPtr(0)`: all lane 0)
    pub lane: DevPtr,
    /// i32 [rows][sel_stride] compressed positions, -1 = none (`DevPtr(0)`: no compressed part)
    pub sel: DevPtr,
    pub sel_stride: usize,
    /// entries of sel per row to read at most (<= 512)
    pub sel_cap: usize,
    /// i32 [rows] entries to read per row (`DevPtr(0)`: sel_cap)
    pub n_sel: DevPtr,
    pub lanes: Vec<LaneSrc>,
    /// bf16 [rows][out_stride]: o after the enabled epilogue steps
    pub out: DevPtr,
    pub out_stride: usize,
    /// optional E4M3 codes [rows][outq_stride] and E8M0 bytes [rows][outs_stride] (EPI_AQ)
    pub out_q: Option<(DevPtr, usize)>,
    pub out_s: Option<(DevPtr, usize)>,
    pub epi: u32,
    /// fp32 inverse frequencies of the layer's RoPE class (`rope_inv_freq`)
    pub rope_freq: [f32; 32],
    /// optional i32 error word (0 = fine; `ERR_SEL_RANGE` when a selection is past kv_rows)
    pub err: DevPtr,
}

/// The decode / verify scratch: split partials and the self-resetting arrival counters (zero them once
/// after allocation). One scratch per stream: two concurrent calls must not share it.
#[derive(Clone, Copy, Debug)]
pub struct DecodeScratch { pub part: DevPtr, pub cnt: DevPtr, pub rows: usize, pub groups: usize, pub nsplit: usize }

impl DecodeScratch {
    /// (partial bytes, counter bytes) for `rows` rows of `heads` heads with up to `nsplit` splits.
    pub fn bytes(rows: usize, heads: usize, nsplit: usize) -> (usize, usize) {
        let g = heads.div_ceil(HG);
        (rows * g * nsplit * PART_FLOATS * 4, rows * g * 4)
    }
}

/// The softmax scale fp32(512^-0.5) the reference uses (`head_dim ** -0.5` in Python, cast once).
pub fn softmax_scale() -> f32 { (1.0f64 / D as f64).sqrt() as f32 }

/// Splits a decode grid must cover: (2 window blocks + ceil(sel_cap / 64) compressed blocks + the extra
/// block) / SPLIT_BLOCKS, rounded up. Rows needing fewer splits leave the extra CTAs idle (they return).
pub fn nsplit_for(sel_cap: usize, any_ext: bool) -> usize { (2 + sel_cap.min(SEL_MAX).div_ceil(64) + usize::from(any_ext)).div_ceil(SPLIT_BLOCKS) }

/// The fp32 inverse RoPE frequencies of a layer class: the ONE dsv41 builder (`crate::dsv41::rope`, torch on
/// CUDA, GPU-built; SL RoPE unification: this file had its own builder with the CPU's true-division ramp, 1 ulp
/// off REFb's table in entry 24 of the compressed class). `yarn` = (factor, original_seq_len, beta_fast,
/// beta_slow) for compressed layers (theta = compress_rope_theta), `None` for SWA-only layers and DSpark stages.
pub fn rope_inv_freq(theta: f64, yarn: Option<(f64, f64, f64, f64)>) -> Result<[f32; 32]> {
    crate::dsv41::rope::table(&crate::dsv41::rope::Params { theta, dim: 64, yarn })
}

/// The RoPE class of a layer of a config: compressed layers rotate at compress_rope_theta with YaRN,
/// SWA-only layers and DSpark stages at rope_theta without (inference/model.py 681-697).
pub fn rope_inv_freq_for_layer(cfg: &crate::dsv41::config::Dsv41Config, layer: usize) -> Result<[f32; 32]> {
    crate::dsv41::rope::table_for_layer(cfg, layer)
}

impl AttnCall {
    fn args(&self, nsplit_max: usize, scratch: Option<&DecodeScratch>) -> Result<AttnArgs> {
        ensure!(self.rows > 0, "attn: no rows");
        ensure!((1..=64).contains(&self.heads), "attn: heads {} (1..=64)", self.heads);
        ensure!(!self.lanes.is_empty() && self.lanes.len() <= MAX_LANES, "attn: {} lanes (1..={MAX_LANES})", self.lanes.len());
        ensure!(self.q_stride % 8 == 0 && self.q_stride >= self.heads * D, "attn: q_stride {} (multiple of 8, >= heads x 512)", self.q_stride);
        ensure!(self.out_stride % 2 == 0 && self.out_stride >= self.heads * D, "attn: out_stride {}", self.out_stride);
        ensure!(self.sel_cap <= SEL_MAX, "attn: sel_cap {} > {SEL_MAX}", self.sel_cap);
        let mut lanes = [AttnLane::default(); MAX_LANES];
        for (i, l) in self.lanes.iter().enumerate() {
            ensure!(l.slots > 0 || l.lin.is_some(), "attn: lane {i} has neither a ring nor linear window rows");
            let (lq, ls, lb) = l.lin.map(|(q, s, b)| (q.0, s.0, b)).unwrap_or((0, 0, i64::MAX));
            let (kv, kr) = l.kv.map(|(p, n)| (p.0, n)).unwrap_or((0, 0));
            let (eq, es, ne) = l.ext.map(|(q, s, n)| (q.0, s.0, n)).unwrap_or((0, 0, 0));
            ensure!(ne <= EXT_MAX, "attn: lane {i} has {ne} extra keys (<= {EXT_MAX})");
            ensure!(kr <= i32::MAX as usize, "attn: lane {i} kv rows {kr}");
            lanes[i] = AttnLane { swa_q: l.swa_q.0, swa_s: l.swa_s.0, lin_q: lq, lin_s: ls, kv, ext_q: eq, ext_s: es, lin_base: lb,
                                  slots: l.slots.max(1) as i32, kv_rows: kr as i32, n_ext: ne as i32, pad_: 0 };
        }
        let groups = self.heads.div_ceil(HG);
        // nsplit_max = partial slots | (CTAs per (row, head group) << 16) (PV1 split groups; 63_attn_entry.cuh d41a_run)
        let (part, cnt) = match scratch {
            Some(s) => {
                ensure!(s.rows >= self.rows && s.groups >= groups && s.nsplit >= (nsplit_max & 0xFFFF), "attn: decode scratch too small");
                (s.part.0, s.cnt.0)
            }
            None => (0, 0),
        };
        Ok(AttnArgs {
            lanes, q: self.q.0, q_stride: self.q_stride as i64, sink: self.sink.0, pos: self.pos.0, win_end: self.win_end.0,
            lane: self.lane.0, sel: self.sel.0, n_sel: self.n_sel.0, sel_stride: self.sel_stride as i64, sel_cap: self.sel_cap as i32,
            heads: self.heads as i32, out: self.out.0, out_stride: self.out_stride as i64,
            out_q: self.out_q.map_or(0, |p| p.0 .0), outq_stride: self.out_q.map_or(0, |p| p.1 as i64),
            out_s: self.out_s.map_or(0, |p| p.0 .0), outs_stride: self.out_s.map_or(0, |p| p.1 as i64),
            part, cnt, err: self.err.0, rows: self.rows as i32, groups: groups as i32, nsplit_max: nsplit_max as i32,
            epi: self.epi, scale: softmax_scale(), win_lo: 0, rope_freq: self.rope_freq,
            rot_suh: [0; 8], rot_xh: 0, rot_ld: 0, rot_hpg: 0, xflags: 0,
        })
    }

    /// The splits a decode grid needs for this call.
    pub fn nsplit(&self) -> usize {
        let any_ext = self.lanes.iter().any(|l| l.ext.map_or(false, |e| e.2 > 0));
        let cap = if self.sel.0 != 0 { self.sel_cap } else { 0 };
        nsplit_for(cap, any_ext)
    }
}

/// PV1: CTAs per (row, 16-head group) of a decode / verify call with `ns` splits a row: the G in 1..=ns that minimises a
/// cost model fitted on GB10 (tools/dsv41_pv1/attn_bench.cu, graph-timed, cold KV, 4K): a CTA of k splits costs
/// ~1.0 + 5.9 k us, a split group (G > 1) adds ~5.8 us (partials, counter, merge), every wave after the first ~2 us.
/// G = ns is KA's one split per CTA (best at 1-6 rows), G = 1 the prefill entry's whole row per CTA (best once rows x
/// head groups fill the SMs). Every G gives the same bits (the merges run in the same order), so this is a speed choice only.
pub fn split_group(rows: usize, groups: usize, ns: usize, sms: usize) -> usize {
    let (mut best, mut best_t) = (ns.max(1), f64::MAX);
    for g in 1..=ns.max(1) {
        let waves = (rows * groups * g).div_ceil(sms.max(1)) as f64;
        let t = waves * (1.0 + 5.9 * ns.div_ceil(g) as f64) + if g > 1 { 5.8 } else { 0.0 } + 2.0 * (waves - 1.0);
        if t < best_t - 1e-9 { best_t = t; best = g; }
    }
    best
}

/// P7ATTN: CTAs per `(row, 8-head subgroup)` of `d41_attn_decode_2cta` (`2 * sms` resident CTAs at 2 CTAs/SM).
pub fn split_group_2cta(rows: usize, heads: usize, ns: usize, sms: usize) -> usize {
    split_group(rows, heads.div_ceil(HG_2CTA), ns, 2 * sms.max(1))
}

/// P7ATTN: `--dsv41-attn-2cta auto` shape selector fitted on GB10 (48 SMs, 96 resident 2-CTA slots).
/// With `defer` (`--dsv41-attn-merge on`, production TP decode / verify), `d41_attn_decode_2cta` wins whenever 16-head
/// 1-CTA/SM blocks under-fill SMs or spill waves (`rows * sub_groups <= 32` = up to 8 rows at TP2, 16 rows at TP4, plus
/// 2..24 rows at TP8 and SWA-only up to `sms` head-subgroups). Without `defer` (in-CTA merge), `2cta` wins at `rows <= 2`
/// (`heads > 8`), `rows >= 2` (`heads <= 8`), and SWA-only up to `sms` head-subgroups.
pub fn use_2cta_auto(rows: usize, heads: usize, ns: usize, sms: usize, defer: bool) -> bool {
    let sg = heads.div_ceil(HG_2CTA);
    let sms = sms.max(1);
    if ns <= 1 {
        heads > HG_2CTA && rows * sg <= sms
    } else if heads <= HG_2CTA {
        rows >= 2
    } else if defer {
        rows * sg <= (2 * sms) / 3
    } else {
        rows <= 2 && rows * sg <= (2 * sms) / 3
    }
}

fn resolve_2cta(rows: usize, heads: usize, ns: usize, sms: usize, defer: bool) -> bool {
    match crate::opts::lookup("dsv41-attn-2cta").and_then(crate::opts::get).as_deref() {
        Some("on") => true,
        Some("off") => false,
        _ => use_2cta_auto(rows, heads, ns, sms, defer),
    }
}

/// SMs of the current device (read once).
fn sm_count() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        use cudarc::driver::sys::*;
        let (mut dev, mut n) = (0 as CUdevice, 0i32);
        let ok = unsafe { cuCtxGetDevice(&mut dev) == CUresult::CUDA_SUCCESS
            && cuDeviceGetAttribute(&mut n, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUresult::CUDA_SUCCESS };
        if ok && n > 0 { n as usize } else { 48 }
    })
}

/// PV1 / P7ATTN receipts (for /metrics): decode calls, split-grouped calls, and 2-CTA/SM decode calls
static DEC_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GROUPED_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DEC_2CTA_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn decode_calls() -> u64 { DEC_CALLS.load(std::sync::atomic::Ordering::Relaxed) }
pub fn grouped_calls() -> u64 { GROUPED_CALLS.load(std::sync::atomic::Ordering::Relaxed) }
pub fn decode_2cta_calls() -> u64 { DEC_2CTA_CALLS.load(std::sync::atomic::Ordering::Relaxed) }
/// `/metrics` rows of PV1 / P7ATTN decode attention paths (the counts are launches the host built: a graph replay re-runs a captured launch without counting).
pub fn render(o: &mut String) {
    for (n, h, v) in [
        ("velogb10_dsv41_attn_dec_calls_total", "Decode / verify attention launches built by the host (graph replays re-run captured launches without counting)", decode_calls()),
        ("velogb10_dsv41_attn_dec_grouped_calls_total", "Decode / verify attention launches that gave a CTA several splits (PV1 split groups, G < splits)", grouped_calls()),
        ("velogb10_dsv41_attn_dec_2cta_calls_total", "Decode / verify attention launches built on d41_attn_decode_2cta (256 threads, 8 heads/CTA, 2 CTAs/SM; P7ATTN)", decode_2cta_calls()),
    ] { o.push_str(&format!("# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}\n")); }
}

/// Decode / verify: dispatches `d41_attn_decode_2cta` or `d41_attn_decode` per `--dsv41-attn-2cta` (`auto` by default),
/// bit-identical to `launch_decode_x(..., None, 0)`.
pub fn launch_decode(l: Launch, call: &AttnCall, scratch: &DecodeScratch) -> Result<()> {
    launch_decode_x(l, call, scratch, None, 0)
}

/// Leg PD1: wo_a's input rotation done by the attention epilogue (the producer-side rotation of the dense decode class).
/// `suh[i]` = the suh of wo_a group i (the call's i-th group, heads [i * hpg, (i + 1) * hpg) of `call.heads`), `xh` = fp16
/// [rows][ld] (head h at column h * 512); every head of the call must belong to a listed group.
#[derive(Clone, Copy, Debug)]
pub struct RotOut<'a> { pub suh: &'a [u64], pub hpg: usize, pub xh: u64, pub ld: usize }

/// `launch_decode` with the epilogue also writing wo_a's rotated input rows (`RotOut`); the attention's own outputs and
/// bits are unchanged (the rotation reads the stored bf16 values).
pub fn launch_decode_rot(l: Launch, call: &AttnCall, scratch: &DecodeScratch, rot: &RotOut) -> Result<()> {
    launch_decode_x(l, call, scratch, Some(rot), 0)
}

/// Leg PA1 / P7ATTN: the decode entry with the `X_*` flags `x` (and PD1's wo_a rotation when `rot` is given). With `X_DEFER` a
/// second launch, `d41_attn_merge` (one 128-thread CTA per (row, head)), merges the split slots in split order and runs
/// the epilogue; the outputs (out, out_q / out_s, rot_xh) are bit-identical across `1cta` / `2cta` and `launch_decode`.
pub fn launch_decode_x(l: Launch, call: &AttnCall, scratch: &DecodeScratch, rot: Option<&RotOut>, x: u32) -> Result<()> {
    ensure!(x & !(X_WPF | X_DEFER | X_QROPE | X_RECIPE | X_2CTA | X_1CTA) == 0, "attn: unknown PA1 flags {x:#x}");
    if x & X_RECIPE != 0 && recipe_covers(call) { return launch_recipe(l, call, Some(scratch), rot, x); }
    let ns = call.nsplit();
    let sms = sm_count();
    let defer_req = x & X_DEFER != 0;
    let two_cta = if x & X_2CTA != 0 { true } else if x & X_1CTA != 0 { false } else { resolve_2cta(call.rows, call.heads, ns, sms, defer_req) };
    let x = x & !(X_RECIPE | X_2CTA | X_1CTA);
    let g = if two_cta {
        split_group_2cta(call.rows, call.heads, ns, sms)
    } else {
        split_group(call.rows, call.heads.div_ceil(HG), ns, sms)
    };
    ensure!(ns < (1 << 16) && g < (1 << 15), "attn: {ns} splits / {g} CTAs per row group");
    // the separate merge only pays where a (row, group) has several CTAs; with one (G = 1, wide calls) the in-CTA epilogue
    // stays (kernel timing: +6 us at 24 rows otherwise). Same bits either way.
    let x = if g == 1 { x & !X_DEFER } else { x };
    let dev_x = if two_cta { x | X_2CTA } else { x };
    let mut a = call.args(ns | (g << 16), Some(scratch))?;
    if let Some(rot) = rot {
        ensure!(rot.hpg > 0 && !rot.suh.is_empty() && rot.suh.len() <= 8 && rot.suh.len() * rot.hpg >= call.heads,
                "attn rot_out: {} groups of {} heads for {} heads (<= 8 groups)", rot.suh.len(), rot.hpg, call.heads);
        ensure!(rot.xh != 0 && rot.xh % 8 == 0 && rot.ld % 4 == 0 && rot.ld >= call.heads * D, "attn rot_out: xh {:#x} ld {}", rot.xh, rot.ld);
        for (i, &p) in rot.suh.iter().enumerate() { ensure!(p != 0 && p % 8 == 0, "attn rot_out: suh {i} {p:#x}"); a.rot_suh[i] = p; }
        a.rot_xh = rot.xh; a.rot_ld = rot.ld as i64; a.rot_hpg = rot.hpg as i32;
    }
    a.xflags = dev_x as i32;
    DEC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if g < ns { GROUPED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    if two_cta {
        DEC_2CTA_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sub_groups = call.heads.div_ceil(HG_2CTA);
        let grid = (call.rows * sub_groups * g) as u32;
        launch_cfg("d41_attn_decode_2cta", l, &a, grid, THREADS_2CTA, SMEM_2CTA)?;
    } else {
        let grid = (call.rows * a.groups as usize * g) as u32;
        launch_cfg("d41_attn_decode", l, &a, grid, THREADS, SMEM)?;
    }
    if x & X_DEFER != 0 { launch_cfg("d41_attn_merge", l, &a, (call.rows * a.groups as usize * HG) as u32, 128, 0)?; }
    Ok(())
}

/// Prefill: one CTA per (row, 16-head group), every split in-CTA. Any row count.
pub fn launch_prefill(l: Launch, call: &AttnCall) -> Result<()> {
    let a = call.args(1, None)?;
    launch("d41_attn_prefill", l, &a, (call.rows * a.groups as usize) as u32)
}

/// Prefill with a window floor (leg CE, the opt-in truncated replay `--dsv41-ced-keep`): window keys below
/// `win_lo` are masked for every row of the call (DeepSeek's `swa_valid_start`). `win_lo = 0` is `launch_prefill`.
pub fn launch_prefill_floor(l: Launch, call: &AttnCall, win_lo: usize) -> Result<()> {
    let mut a = call.args(1, None)?;
    ensure!(win_lo <= i32::MAX as usize, "attn: window floor {win_lo}");
    a.win_lo = win_lo as i32;
    launch("d41_attn_prefill", l, &a, (call.rows * a.groups as usize) as u32)
}

fn launch(name: &str, l: Launch, a: &AttnArgs, grid: u32) -> Result<()> { launch_cfg(name, l, a, grid, THREADS, SMEM) }

fn launch_cfg(name: &str, l: Launch, a: &AttnArgs, grid: u32, threads: u32, smem: u32) -> Result<()> { launch_cfg2(name, l, a, grid, 1, threads, smem) }

fn launch_cfg2(name: &str, l: Launch, a: &AttnArgs, grid: u32, grid_y: u32, threads: u32, smem: u32) -> Result<()> {
    use cudarc::driver::sys::*;
    if l.stream == 0 { bail!("attn: the legacy default stream is not allowed (AGENTS 2.1)"); }
    let f = super::raw_fn(name).with_context(|| format!("{name}: raw function missing (module not loaded?)"))?;
    static OPTED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    if smem > 48 * 1024 {
        let mut o = OPTED.lock().map_err(|_| anyhow::anyhow!("attn opt-in lock"))?;
        if !o.contains(&(f as usize)) {
            let r = unsafe { cuFuncSetAttribute(f, CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM as i32) };
            ensure!(r == CUresult::CUDA_SUCCESS, "{name}: smem opt-in {SMEM}: {r:?}");
            o.push(f as usize);
        }
    }
    let mut params = [a as *const AttnArgs as *mut std::ffi::c_void];
    let mut attr = CUlaunchAttribute {
        id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
    };
    let mut cfg = CUlaunchConfig {
        gridDimX: grid, gridDimY: grid_y, gridDimZ: 1, blockDimX: threads, blockDimY: 1, blockDimZ: 1,
        sharedMemBytes: smem, hStream: l.stream as CUstream, attrs: &mut attr, numAttrs: u32::from(l.pdl),
    };
    let r = unsafe { cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut()) };
    ensure!(r == CUresult::CUDA_SUCCESS, "{name}: launch ({grid} CTAs): {r:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_attn_host_abi_layout() {
        let h = host_abi();
        assert_eq!(h[0], 80, "d41_attn_lane is 80 B");
        assert_eq!(h[1], 944 + 88, "d41_attn_args is 1,032 B (944 + PD1's rot_out block of 88 B at the end)");
        assert_eq!(h[5], 944 - 128);
        assert_eq!(h[11], 80);
        assert_eq!(PART_FLOATS, 8224);
        assert_eq!(nsplit_for(512, false), 10usize.div_ceil(SPLIT_BLOCKS));
        assert_eq!(nsplit_for(15, false), 3usize.div_ceil(SPLIT_BLOCKS));
        assert_eq!(nsplit_for(0, true), 3usize.div_ceil(SPLIT_BLOCKS));
        assert!(nsplit_for(512, true) <= MAX_SPLITS);
        assert_eq!(DecodeScratch::bytes(24, 16, 5), (24 * 5 * 8224 * 4, 24 * 4));
        // PV1 split groups on 48 SMs (attn_bench, 4K, 512 picks: 5 splits a row)
        assert_eq!(split_group(1, 1, 5, 48), 5);     // KA's one split per CTA
        assert_eq!(split_group(6, 1, 5, 48), 5);
        assert_eq!(split_group(12, 1, 5, 48), 3);    // 16 heads (TP4), 12 rows
        assert_eq!(split_group(24, 1, 5, 48), 2);    // 16 heads, 24 rows
        assert_eq!(split_group(12, 2, 5, 48), 2);    // 32 heads (TP2), 12 rows
        assert_eq!(split_group(24, 2, 5, 48), 1);    // 32 heads, 24 rows: the whole row per CTA
        assert_eq!(split_group(24, 4, 5, 48), 1);    // 64 heads
        assert_eq!(split_group(3, 1, 1, 48), 1);     // one split a row
        // P7ATTN 2-CTA/SM split groups & auto dispatch on 48 SMs (96 resident CTAs)
        assert_eq!(split_group_2cta(1, 32, 5, 48), 5);   // TP2 1 row: 4 * 5 = 20 CTAs
        assert_eq!(split_group_2cta(4, 32, 5, 48), 5);   // TP2 4 rows: 16 * 5 = 80 CTAs
        assert_eq!(split_group_2cta(6, 32, 5, 48), 3);   // TP2 6 rows (1x5): 24 * 3 = 72 CTAs (1 wave!)
        assert_eq!(split_group_2cta(8, 32, 5, 48), 3);   // TP2 8 rows: 32 * 3 = 96 CTAs (1 wave!)
        assert_eq!(split_group_2cta(6, 16, 5, 48), 5);   // TP4 6 rows (1x5): 12 * 5 = 60 CTAs
        assert_eq!(split_group_2cta(12, 16, 5, 48), 3);  // TP4 12 rows (2x5): 24 * 3 = 72 CTAs (1 wave!)
        assert_eq!(split_group_2cta(16, 16, 5, 48), 3);  // TP4 16 rows (4x3): 32 * 3 = 96 CTAs (1 wave!)
        assert_eq!(split_group_2cta(24, 32, 5, 48), 1);  // TP2 24 rows: 96 CTAs
        assert!(use_2cta_auto(1, 32, 5, 48, true) && use_2cta_auto(6, 32, 5, 48, true) && use_2cta_auto(8, 32, 5, 48, true));
        assert!(!use_2cta_auto(12, 32, 5, 48, true) && !use_2cta_auto(16, 32, 5, 48, true) && !use_2cta_auto(24, 32, 5, 48, true));
        assert!(use_2cta_auto(1, 16, 5, 48, true) && use_2cta_auto(6, 16, 5, 48, true) && use_2cta_auto(12, 16, 5, 48, true) && use_2cta_auto(16, 16, 5, 48, true));
        assert!(!use_2cta_auto(20, 16, 5, 48, true) && !use_2cta_auto(24, 16, 5, 48, true));
        assert!(!use_2cta_auto(1, 8, 5, 48, true) && use_2cta_auto(6, 8, 5, 48, true) && use_2cta_auto(24, 8, 5, 48, true));
        assert!(use_2cta_auto(2, 32, 5, 48, false) && !use_2cta_auto(6, 32, 5, 48, false) && use_2cta_auto(12, 8, 5, 48, false));
        assert!(use_2cta_auto(6, 32, 1, 48, true) && use_2cta_auto(12, 32, 1, 48, true) && !use_2cta_auto(16, 32, 1, 48, true));
        assert!(use_2cta_auto(24, 16, 1, 48, true) && !use_2cta_auto(6, 8, 1, 48, true));
        assert_eq!(SMEM_2CTA, 46_736);
        assert!(SMEM_2CTA <= 48 * 1024);
        assert_eq!(softmax_scale().to_bits(), 0x3D35_04F3);
    }

    #[test]
    fn dsv41_attn_rope_freqs_match_the_reference_shape() {
        // (the host emulation of the one builder: the GPU-built table is checked against torch in dsv41::rope)
        let rope_inv_freq = |theta: f64, yarn| crate::dsv41::rope::table_host(&crate::dsv41::rope::Params { theta, dim: 64, yarn });
        // SWA-only class: theta 10000, no YaRN: f_0 = 1, monotone decreasing
        let f = rope_inv_freq(10000.0, None);
        assert_eq!(f[0], 1.0);
        assert!(f.windows(2).all(|w| w[1] < w[0]));
        // compressed class: theta 160000, YaRN factor 16 over 65536: low pairs untouched, high pairs / 16
        let g = rope_inv_freq(160000.0, Some((16.0, 65536.0, 32.0, 1.0)));
        let plain = rope_inv_freq(160000.0, None);
        assert_eq!(g[..15], plain[..15], "pairs below the ramp keep their frequency");
        for i in 25..32 { assert_eq!(g[i], plain[i] / 16.0 * 1.0 + plain[i] * 0.0, "pairs past the ramp are divided by the factor"); }
    }
}
