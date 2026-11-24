//! Leg PH1: the drafter's Markov bias cache (`d41_ds_mbias`, band 94).
//!
//! Each Markov chain step adds bias(t) = markov_head . markov_embed[t] (rank 256) to every lane's logits row: DS's
//! sequence recomputes it with the plain GEMV every step (the whole Markov head of this rank's vocabulary, 33 MB at
//! TP2, ~135 us). The external reference code keeps the bias rows of a fixed set of frequent tokens, filled once at
//! load, and skips the head read when every stream's chain input is one of them (RX6 §1.1). Here the same, with OUR
//! arithmetic kept: the cached row is the f32 `res` of the same per-column sequence the uncached step runs (no fp16
//! rounding), so a step served from the cache adds the same two f32 operands and the logits, the draws, the drafts and
//! the verify's q rows are bit-identical either way (proven at load over every cached token: [`MarkovCache::selfcheck`]).
//!
//! * the token list: `--dsv41-markov-tokens builtin` (the reference code's frequency ranking of code and English, 256 ids,
//!   `mtoks::BUILTIN`) or a file of ids (whitespace / comma separated, most frequent first; `tools/dsv41_ph1/markov_tokens.py
//!   rank` makes one from text samples); rank 0 reads it and broadcasts it, so every rank caches the same tokens;
//! * the size: `--dsv41-markov-cache N` (the first N distinct ids; 0 = off; K x vlocal x 4 B per rank);
//! * the engagement receipt: steps served from the cache / computed, counted on the device by the kernel itself and
//!   mirrored on the host from the round's ids table (`/metrics` `velogb10_dsv41_markov_*`, `scripts/owner_check.py`).

use super::mtoks;
use super::vocab::as_bytes;
use crate::dsv41::fwd::dev::{Buf, Gpu};
use crate::dsv41::weights::{Accum, Act, ActDtype, BackendKind, DenseWeight, DevPtr, Shard as WShard};
use crate::dsv41::census::classify::Role;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::DeviceRepr;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// CTAs of a `d41_ds_mbias` launch (grid-stride over the columns; measured with `tools/dsv41_ph1/mbias_bench.cu` on
/// `.11`: the uncached step at parity with or faster than the plain GEMV at 1-4 lanes and every TP shard, the cached step
/// 6-10 us).
pub const GRID: u32 = 384;
/// The Markov rank the kernel serves.
pub const RANK: usize = 256;
pub const MODE_STEP: i32 = 0;
pub const MODE_FILL: i32 = 1;
pub const MODE_STEP_MISS: i32 = 2;
/// Lanes the cached step serves (the kernel's decision covers up to 8 rows).
pub const MAX_LANES: usize = 8;
/// The default list size (leg P7.6: 512 static tokens).
pub const DEFAULT_K: usize = 512;
/// Default dynamic session-vocabulary slots written on GPU misses (1024 at TP4+, 512 at TP1/TP2).
pub fn default_dyn_k(world: usize) -> usize { if world >= 4 { 1024 } else { 512 } }

/// (weight ptr, code 0 bf16 / 1 f16) of the 1-row `DsConfidence` projection when stored plain (`K == k_expect`).
pub fn conf_weight(w: &(DenseWeight, BackendKind), k_expect: usize) -> Option<(u64, i32)> {
    if w.1 != BackendKind::Plain || !matches!(w.0.shard, WShard::Whole) || w.0.out != 1 || w.0.inn as usize != k_expect { return None; }
    let t = w.0.tensors.iter().find(|t| t.role == Role::Weight)?;
    let code = match t.dtype.as_str() { "BF16" => 0, "F16" => 1, _ => return None };
    (t.shape.len() == 2 && t.shape[0] == 1 && t.shape[1] as usize == k_expect).then_some((t.ptr.0, code))
}

/// `d41_ds_mbias_args`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MbiasArgs {
    pub w: u64, pub x: u64, pub y: u64, pub ids: u64, pub slot: u64, pub cache: u64, pub lanes: u64, pub cnt: u64,
    pub dyn_ptr: u64, pub pad_ptr: u64,
    pub wcode: i32, pub nn: i32, pub ldx: i32, pub ldy: i32, pub rows: i32, pub mode: i32, pub vocab: i32, pub pad: i32,
}
unsafe impl DeviceRepr for MbiasArgs {}

/// `d41_ds_mc_admit_args`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AdmitArgs {
    pub slot: u64, pub ids: u64, pub lanes: u64, pub dyn_ptr: u64, pub dyn_owner: u64, pub conf: u64, pub early_t: u64, pub cnt: u64,
    pub conf_offset: f64, pub lookahead_m: f64,
    pub rows: i32, pub vocab: i32, pub static_k: i32, pub dyn_cap: i32, pub step: i32, pub k_max: i32, pub conf_ld: i32, pub early_on: i32,
}
unsafe impl DeviceRepr for AdmitArgs {}

/// Step context for dynamic Markov cache admission + conservative early-stop inside `draft_body`.
#[derive(Clone, Copy, Debug)]
pub struct StepAdmit {
    pub conf: u64,
    pub early_t: u64,
    pub conf_offset: f64,
    pub lookahead_m: f64,
    pub step: usize,
    pub k_max: usize,
    pub conf_ld: usize,
    pub early_on: bool,
}

/// The cached rows of one rank.
pub struct MarkovCache {
    /// the cached tokens, row j = toks[j]
    pub toks: Vec<u32>,
    init_slot: Vec<i32>,
    /// [vocab] i32 slot table, on the host (the round's host mirror) and the device
    host_slot: Vec<i32>,
    slot: Buf,
    /// [K + dyn_cap][nn] f32
    cache: Buf,
    dyn_state: Buf,
    dyn_owner: Buf,
    pub dyn_cap: usize,
    host_ring_head: usize,
    host_pend: Vec<(i32, i32)>,
    host_dyn_owner: Vec<i32>,
    /// [K][256] bf16: the cached tokens' embedding rows (kept for the self-check)
    xemb: Buf,
    /// [4] u64: [hit, miss, dyn_fills, early_skipped] (device counters)
    cnt: Buf,
    /// [8] lanes, all active (the self-check's lane table)
    lanes_all: Buf,
    pub nn: usize,
    pub vocab: usize,
    wcode: i32,
    w: u64,
    pub fill_ms: f64,
}

/// (pointer to this rank's first row, rows, code 0 bf16 / 1 f16) of the Markov head (`plain` backend's rule).
fn head_rows(w: &DenseWeight) -> Result<(u64, usize, i32)> {
    let t = w.tensors.iter().find(|t| t.role == Role::Weight).with_context(|| format!("{}: no weight tensor", w.key))?;
    let code = match t.dtype.as_str() { "BF16" => 0, "F16" => 1, d => bail!("Markov head {}: dtype {d} (BF16 / F16 only)", w.key) };
    ensure!(t.shape.len() == 2 && t.shape[1] as usize == RANK && w.inn as usize == RANK, "Markov head {}: shape {:?} (rank {RANK} expected)", w.key, t.shape);
    let n = t.shape[0] as usize;
    Ok(match &w.shard {
        WShard::Whole => (t.ptr.0, n, code),
        WShard::Out(r) => {
            let (lo, hi) = (r.start as usize, r.end as usize);
            ensure!(hi <= w.out as usize && lo < hi, "Markov head {}: shard {lo}..{hi}", w.key);
            if n == w.out as usize { (t.ptr.0 + (lo * RANK * 2) as u64, hi - lo, code) } else { ensure!(n == hi - lo, "Markov head {}: resident rows {n}", w.key); (t.ptr.0, n, code) }
        }
        WShard::In(_) => bail!("Markov head {}: an input slice is not supported", w.key),
    })
}

/// Why the cache cannot serve these Markov pieces (None = it can).
pub fn unsupported(me: &(DenseWeight, BackendKind), mh: &(DenseWeight, BackendKind)) -> Option<String> {
    if me.1 != BackendKind::Plain || mh.1 != BackendKind::Plain { return Some(format!("Markov backends {:?} / {:?} (plain expected)", me.1, mh.1)); }
    if me.0.inn as usize != RANK { return Some(format!("Markov embedding width {} (rank {RANK} expected)", me.0.inn)); }
    head_rows(&mh.0).err().map(|e| e.to_string())
}

/// The token list the flags ask for, before the size cut: (ids, label).
pub fn list_from_flags() -> Result<(Vec<u32>, String)> {
    let spec = crate::opts::lookup("dsv41-markov-tokens").and_then(crate::opts::get).filter(|s| s != "unset").unwrap_or_else(|| "builtin".into());
    if spec == "builtin" { return Ok((mtoks::BUILTIN.to_vec(), "builtin (1024 frequency-ranked code + English tokens)".into())); }
    let text = std::fs::read_to_string(&spec).with_context(|| format!("--dsv41-markov-tokens {spec}"))?;
    Ok((parse_list(&text)?, format!("file {spec}")))
}

/// Ids of a list file: integers separated by whitespace or commas, most frequent first ('#' starts a comment line).
pub fn parse_list(text: &str) -> Result<Vec<u32>> {
    let mut v = Vec::new();
    for line in text.lines() {
        let l = line.split('#').next().unwrap_or("");
        for w in l.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()) {
            v.push(w.parse::<u32>().with_context(|| format!("token list: {w:?} is not a token id"))?);
        }
    }
    Ok(v)
}

/// The first `k` distinct ids of `ids` below `vocab` (the cached set, in order).
pub fn first_distinct(ids: &[u32], k: usize, vocab: usize) -> Vec<u32> {
    let mut seen = std::collections::HashSet::new();
    ids.iter().copied().filter(|&t| (t as usize) < vocab && seen.insert(t)).take(k).collect()
}

/// Leg P4: `--dsv41-markov-cache-lanes` (default on): a step where some active lane misses serves the lanes that hit from the cache.
pub fn perlane() -> bool { crate::opts::lookup("dsv41-markov-cache-lanes").and_then(crate::opts::get).map_or(true, |v| crate::opts::is_on_value(&v)) }

/// Leg P6: `--dsv41-draft-early-stop` (default on): skip uncached Markov head DRAM reads after 1-lane exact stopping rule halts.
pub fn early_stop_on() -> bool { crate::opts::lookup("dsv41-draft-early-stop").and_then(crate::opts::get).map_or(true, |v| crate::opts::is_on_value(&v)) }

/// Leg P6: `--dsv41-markov-dyn-cache auto|N` (dynamic session-vocabulary GPU writeback slots; 0 = off).
pub fn dyn_k_from_flags(world: usize) -> Result<usize> {
    match crate::opts::lookup("dsv41-markov-dyn-cache").and_then(crate::opts::get).filter(|s| s != "unset" && s != "auto") {
        None => Ok(default_dyn_k(world)),
        Some(v) => {
            let n = v.trim().parse::<usize>().with_context(|| format!("--dsv41-markov-dyn-cache {v}"))?;
            ensure!(n <= 4096, "--dsv41-markov-dyn-cache {v}: at most 4096");
            Ok(if n == 0 { 0 } else { n.max(MAX_LANES) })
        }
    }
}

/// The size `--dsv41-markov-cache` asks for.
pub fn k_from_flags() -> Result<usize> {
    match crate::opts::lookup("dsv41-markov-cache").and_then(crate::opts::get).filter(|s| s != "unset") {
        None => Ok(DEFAULT_K),
        Some(v) => v.trim().parse::<usize>().with_context(|| format!("--dsv41-markov-cache {v}")),
    }
}

impl MarkovCache {
    /// Gather the tokens' embedding rows and FILL their cache rows with `d41_ds_mbias` (the step's own arithmetic).
    pub fn build(g: &Gpu, me: &(DenseWeight, BackendKind), mh: &(DenseWeight, BackendKind), toks: Vec<u32>, vocab: usize) -> Result<MarkovCache> {
        Self::build_with_dyn(g, me, mh, toks, 0, vocab)
    }
    pub fn build_with_dyn(g: &Gpu, me: &(DenseWeight, BackendKind), mh: &(DenseWeight, BackendKind), toks: Vec<u32>, dyn_cap: usize, vocab: usize) -> Result<MarkovCache> {
        if let Some(why) = unsupported(me, mh) { bail!("{why}"); }
        ensure!(!toks.is_empty(), "an empty token list");
        let (w, nn, wcode) = head_rows(&mh.0)?;
        let k = toks.len();
        let t0 = std::time::Instant::now();
        let mut host_slot = vec![-1i32; vocab];
        for (j, &t) in toks.iter().enumerate() {
            ensure!((t as usize) < vocab && host_slot[t as usize] < 0, "token list: id {t} repeated or >= {vocab}");
            host_slot[t as usize] = j as i32;
        }
        let init_slot = host_slot.clone();
        let slot = g.buf(vocab * 4)?;
        g.h2d(slot.ptr(), as_bytes(&host_slot))?;
        let ids = g.buf(k * 4)?;
        g.h2d(ids.ptr(), as_bytes(&toks))?;
        let xemb = g.buf(k * RANK * 2)?;
        let out = Act { ptr: DevPtr(xemb.ptr()), rows: k, cols: RANK, ld: RANK, dtype: ActDtype::Bf16 };
        crate::dsv41::fwd::glue::backend(me.1).gather(&g.l(), &me.0, DevPtr(ids.ptr()), k, &out)?;
        let total_k = k + dyn_cap;
        let cache = g.buf(total_k * nn * 4)?;
        let dyn_state = g.buf(128)?;
        g.memset(dyn_state.ptr(), 128, 0)?;
        let dyn_owner = g.buf(dyn_cap.max(1) * 4)?;
        g.h2d(dyn_owner.ptr(), as_bytes(&vec![-1i32; dyn_cap.max(1)]))?;
        let cnt = g.buf(32)?;
        g.memset(cnt.ptr(), 32, 0)?;
        let lanes_all = g.buf(MAX_LANES * std::mem::size_of::<super::Lane>())?;
        let la: Vec<super::Lane> = (0..MAX_LANES).map(|_| super::Lane { active: 1, ..Default::default() }).collect();
        g.h2d(lanes_all.ptr(), as_bytes(&la))?;
        let mc = MarkovCache {
            toks, init_slot, host_slot, slot, cache, dyn_state, dyn_owner, dyn_cap,
            host_ring_head: 0, host_pend: Vec::new(), host_dyn_owner: vec![-1; dyn_cap],
            xemb, cnt, lanes_all, nn, vocab, wcode, w, fill_ms: 0.0,
        };
        mc.launch(g, MODE_FILL, mc.xemb.ptr(), RANK, k, mc.cache.ptr(), nn, 0, 0, false)?;
        g.sync()?;
        Ok(MarkovCache { fill_ms: t0.elapsed().as_secs_f64() * 1e3, ..mc })
    }

    /// Reset dynamic ring + counters after boot calibration so host and device mirrors start round 1 in lockstep.
    pub fn reset_dyn(&mut self, g: &Gpu) -> Result<()> {
        self.host_slot.copy_from_slice(&self.init_slot);
        self.host_ring_head = 0;
        self.host_pend.clear();
        self.host_dyn_owner.fill(-1);
        g.h2d(self.slot.ptr(), as_bytes(&self.init_slot))?;
        g.memset(self.dyn_state.ptr(), 128, 0)?;
        g.h2d(self.dyn_owner.ptr(), as_bytes(&vec![-1i32; self.dyn_cap.max(1)]))?;
        g.memset(self.cnt.ptr(), 32, 0)?;
        g.sync()
    }

    pub fn k(&self) -> usize { self.toks.len() }
    pub fn total_k(&self) -> usize { self.toks.len() + self.dyn_cap }
    /// Device address of `d41_ds_mc_dyn::stopped` (`0` = full chain ran, `1..=5` = early-stopped at step `stopped - 1`).
    pub fn stopped_ptr(&self) -> u64 { self.dyn_state.ptr() + 8 }
    pub fn dev_bytes(&self) -> u64 { [&self.slot, &self.cache, &self.dyn_state, &self.dyn_owner, &self.xemb, &self.cnt, &self.lanes_all].iter().map(|b| b.bytes as u64).sum() }

    #[allow(clippy::too_many_arguments)]
    fn launch(&self, g: &Gpu, mode: i32, x: u64, ldx: usize, rows: usize, y: u64, ldy: usize, ids: u64, lanes: u64, use_dyn: bool) -> Result<()> {
        ensure!(x % 16 == 0 && ldx % 8 == 0, "d41_ds_mbias: the embedding rows must be 16-B aligned (x {x:#x}, ld {ldx})");
        ensure!(mode == MODE_FILL || rows <= MAX_LANES, "d41_ds_mbias: {rows} lanes > {MAX_LANES}");
        let a = MbiasArgs { w: self.w, x, y, ids, slot: self.slot.ptr(), cache: self.cache.ptr(), lanes, cnt: if mode == MODE_STEP { self.cnt.ptr() } else { 0 },
                            dyn_ptr: if use_dyn { self.dyn_state.ptr() } else { 0 }, pad_ptr: 0,
                            wcode: self.wcode, nn: self.nn as i32, ldx: ldx as i32, ldy: ldy as i32, rows: rows as i32, mode, vocab: self.vocab as i32, pad: perlane() as i32 };
        let name = if mode != MODE_FILL && rows <= 4 { "d41_ds_mbias_r4" } else { "d41_ds_mbias" };
        g.launch1(name, (GRID, 1, 1), (256, 1, 1), 0, &a)
    }

    /// One chain step of `rows` lanes: `logits` rows (f32, stride `ldy`) += bias(ids[l]); `x` = the lanes' gathered
    /// embedding rows (bf16, stride `ldx`), `ids` = the chain inputs (i32 [rows]), `lanes` = the round's lane table.
    pub fn step(&self, g: &Gpu, x: u64, ldx: usize, rows: usize, logits: u64, ldy: usize, ids: u64, lanes: u64) -> Result<()> {
        self.launch(g, MODE_STEP, x, ldx, rows, logits, ldy, ids, lanes, false)
    }

    /// Leg P6: run `d41_ds_mc_admit` (dynamic cache commit/admission + conservative early-stop) followed by `d41_ds_mbias`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_admit(&self, g: &Gpu, x: u64, ldx: usize, rows: usize, logits: u64, ldy: usize, ids: u64, lanes: u64, adm: StepAdmit) -> Result<()> {
        let use_dyn = self.dyn_cap > 0 || adm.early_on;
        if use_dyn {
            let aa = AdmitArgs {
                slot: self.slot.ptr(), ids, lanes, dyn_ptr: self.dyn_state.ptr(),
                dyn_owner: if self.dyn_cap > 0 { self.dyn_owner.ptr() } else { 0 },
                conf: adm.conf, early_t: adm.early_t, cnt: self.cnt.ptr(),
                conf_offset: adm.conf_offset, lookahead_m: adm.lookahead_m,
                rows: rows as i32, vocab: self.vocab as i32, static_k: self.k() as i32, dyn_cap: self.dyn_cap as i32,
                step: adm.step as i32, k_max: adm.k_max as i32, conf_ld: adm.conf_ld as i32, early_on: adm.early_on as i32,
            };
            g.launch1("d41_ds_mc_admit", (1, 1, 1), (32, 1, 1), 0, &aa)?;
        }
        self.launch(g, MODE_STEP, x, ldx, rows, logits, ldy, ids, lanes, use_dyn)
    }

    /// Is a step of these chain inputs (active lanes only) served from the cache? (the kernel's own rule)
    pub fn hit(&self, ids: &[i32]) -> bool {
        ids.len() <= MAX_LANES && ids.iter().all(|&t| t >= 0 && (t as usize) < self.vocab && self.host_slot[t as usize] >= 0)
    }
    pub fn has(&self, t: i32) -> bool { t >= 0 && (t as usize) < self.vocab && self.host_slot[t as usize] >= 0 }

    /// Mirror `d41_ds_mc_admit` on the host for one active step's chain inputs (when not early-stopped).
    pub fn mirror_step(&mut self, active_ids: &[i32], stopped: bool) -> (u64, bool) {
        for &(pid, ps) in &self.host_pend {
            if pid >= 0 && (pid as usize) < self.vocab && ps >= 0 {
                self.host_slot[pid as usize] = ps;
            }
        }
        self.host_pend.clear();
        if stopped { return (0, true); }
        let hits = active_ids.iter().filter(|&&t| self.has(t)).count() as u64;
        let all_hit = self.hit(active_ids);
        if self.dyn_cap > 0 && active_ids.len() <= MAX_LANES {
            for &id in active_ids {
                if id < 0 || (id as usize) >= self.vocab || self.host_slot[id as usize] >= 0 { continue; }
                if self.host_pend.iter().any(|&(pid, _)| pid == id) { continue; }
                if self.host_pend.len() >= MAX_LANES || self.host_pend.len() >= self.dyn_cap { break; }
                let idx = self.host_ring_head;
                self.host_ring_head = if idx + 1 >= self.dyn_cap { 0 } else { idx + 1 };
                let s = (self.toks.len() + idx) as i32;
                let old = self.host_dyn_owner[idx];
                if old >= 0 && (old as usize) < self.vocab && self.host_slot[old as usize] == s {
                    self.host_slot[old as usize] = -1;
                }
                self.host_dyn_owner[idx] = id;
                self.host_pend.push((id, s));
                DYN_FILLS.fetch_add(1, Relaxed);
            }
        }
        (hits, all_hit)
    }

    /// The device counters (steps served from the cache, steps computed, dynamic fills, early-stopped steps).
    pub fn device_counts_all(&self, g: &Gpu) -> Result<[u64; 4]> {
        let v = g.down::<u64>(self.cnt.ptr(), 4)?;
        Ok([v[0], v[1], v[2], v[3]])
    }
    pub fn device_counts(&self, g: &Gpu) -> Result<(u64, u64)> {
        let v = self.device_counts_all(g)?;
        Ok((v[0], v[1]))
    }

    /// Over every cached token, in groups of `lanes` (1 and 4): the plain GEMV += (DS's sequence), the kernel's
    /// computed step (forced miss) and its cached step on the same logits rows, compared bit for bit. Returns
    /// (groups checked, mismatching groups).
    pub fn selfcheck(&self, g: &Gpu, mh: &(DenseWeight, BackendKind), scr: &crate::dsv41::fwd::glue::Scr) -> Result<(usize, usize)> {
        let nn = self.nn;
        let (mut groups, mut bad) = (0usize, 0usize);
        for lanes in [1usize, 4] {
            let y0: Vec<f32> = (0..lanes * nn).map(|i| match i % 997 { 3 => -0.0, 5 => 0.0, _ => ((i as u64 * 2_654_435_761 % 20_011) as f32 - 10_000.0) * 1e-3 }).collect();
            let (ya, yb, yc) = (g.buf(lanes * nn * 4)?, g.buf(lanes * nn * 4)?, g.buf(lanes * nn * 4)?);
            let ids = g.buf(lanes * 4)?;
            let mut j = 0;
            while j + lanes <= self.k() {
                let x = self.xemb.ptr() + (j * RANK * 2) as u64;
                g.h2d(ids.ptr(), as_bytes(&self.toks[j..j + lanes]))?;
                for b in [&ya, &yb, &yc] { g.h2d(b.ptr(), as_bytes(&y0))?; }
                let xa = Act { ptr: DevPtr(x), rows: lanes, cols: RANK, ld: RANK, dtype: ActDtype::Bf16 };
                let mut yy = Act { ptr: DevPtr(ya.ptr()), rows: lanes, cols: nn, ld: nn, dtype: ActDtype::F32 };
                yy.cols = nn;
                crate::dsv41::fwd::glue::linear(g, "ds.markov_cache.check (plain GEMV)", mh, &xa, &yy, Accum::Add, scr)?;
                self.launch(g, MODE_STEP_MISS, x, RANK, lanes, yb.ptr(), nn, ids.ptr(), self.lanes_all.ptr(), false)?;
                // a cached step through the counted entry would move the served counters: the check uses no counter
                let a = MbiasArgs { w: self.w, x, y: yc.ptr(), ids: ids.ptr(), slot: self.slot.ptr(), cache: self.cache.ptr(), lanes: self.lanes_all.ptr(), cnt: 0,
                                    dyn_ptr: 0, pad_ptr: 0,
                                    wcode: self.wcode, nn: nn as i32, ldx: RANK as i32, ldy: nn as i32, rows: lanes as i32, mode: MODE_STEP, vocab: self.vocab as i32, pad: 0 };
                g.launch1(if lanes <= 4 { "d41_ds_mbias_r4" } else { "d41_ds_mbias" }, (GRID, 1, 1), (256, 1, 1), 0, &a)?;
                let (va, vb, vc) = (g.down::<u32>(ya.ptr(), lanes * nn)?, g.down::<u32>(yb.ptr(), lanes * nn)?, g.down::<u32>(yc.ptr(), lanes * nn)?);
                groups += 1;
                if va != vb || vb != vc { bad += 1; }
                j += lanes;
            }
        }
        // leg P4: a step where only SOME lanes hit (per-lane hit / miss, pad bit 0) against the all-rows computed step, every subset of 4 lanes
        // missing (the missing lanes carry a token outside the cache), in several places of the list, with the lane table's active
        // flags all set (the inactive-lane rule is unchanged code and covered by the served identity runs)
        let unc: Vec<i32> = (0..self.vocab as i32).filter(|&t| self.host_slot[t as usize] < 0).take(4).collect();
        if self.k() >= 8 && unc.len() == 4 {
            let lanes = 4usize;
            let y0: Vec<f32> = (0..lanes * nn).map(|i| match i % 991 { 7 => -0.0, 11 => 0.0, _ => ((i as u64 * 40_503 % 30_011) as f32 - 15_000.0) * 1e-3 }).collect();
            let (yb, yc) = (g.buf(lanes * nn * 4)?, g.buf(lanes * nn * 4)?);
            let ids = g.buf(lanes * 4)?;
            for j in (0..=self.k() - lanes).step_by(29) {
                for miss in 1u32..16 {
                    let x = self.xemb.ptr() + (j * RANK * 2) as u64;
                    let idv: Vec<i32> = (0..lanes).map(|i| if miss >> i & 1 == 1 { unc[i] } else { self.toks[j + i] as i32 }).collect();
                    g.h2d(ids.ptr(), as_bytes(&idv))?;
                    for b in [&yb, &yc] { g.h2d(b.ptr(), as_bytes(&y0))?; }
                    self.launch(g, MODE_STEP_MISS, x, RANK, lanes, yb.ptr(), nn, ids.ptr(), self.lanes_all.ptr(), false)?;
                    let a = MbiasArgs { w: self.w, x, y: yc.ptr(), ids: ids.ptr(), slot: self.slot.ptr(), cache: self.cache.ptr(), lanes: self.lanes_all.ptr(), cnt: 0,
                                        dyn_ptr: 0, pad_ptr: 0,
                                        wcode: self.wcode, nn: nn as i32, ldx: RANK as i32, ldy: nn as i32, rows: lanes as i32, mode: MODE_STEP, vocab: self.vocab as i32, pad: 1 };
                    g.launch1("d41_ds_mbias_r4", (GRID, 1, 1), (256, 1, 1), 0, &a)?;
                    let (vb, vc) = (g.down::<u32>(yb.ptr(), lanes * nn)?, g.down::<u32>(yc.ptr(), lanes * nn)?);
                    groups += 1;
                    if vb != vc { bad += 1; }
                }
            }
        }
        Ok((groups, bad))
    }
}

// ------------------------------------------------------------------------------------------------ /metrics

/// steps served from the cache / computed, as the host mirrors the kernel's rule (rank 0, every drafted round)
pub static STEPS_HIT: AtomicU64 = AtomicU64::new(0);
pub static STEPS_MISS: AtomicU64 = AtomicU64::new(0);
/// chain inputs (lane-steps) whose token is cached / not
pub static INPUTS_HIT: AtomicU64 = AtomicU64::new(0);
pub static INPUTS_MISS: AtomicU64 = AtomicU64::new(0);
/// the kernel's own counters as last read (rank 0)
pub static DEV_HIT: AtomicU64 = AtomicU64::new(0);
pub static DEV_MISS: AtomicU64 = AtomicU64::new(0);
pub static DYN_CAP: AtomicU64 = AtomicU64::new(0);
pub static DYN_FILLS: AtomicU64 = AtomicU64::new(0);
pub static DEV_DYN_FILLS: AtomicU64 = AtomicU64::new(0);
pub static DEV_EARLY_SKIP: AtomicU64 = AtomicU64::new(0);
pub static EARLY_SKIP_STEPS: AtomicU64 = AtomicU64::new(0);
/// steps by the number of active lanes (1..=4, 5+ in the last), (hit, miss)
static BY_LANES: Mutex<[(u64, u64); 5]> = Mutex::new([(0, 0); 5]);
/// the boot's cache state: (tokens cached, MB a rank, fill ms, self-check groups, mismatches, list label) or the reason it is off
static STATE: Mutex<Option<Result<(usize, f64, f64, usize, usize, String), String>>> = Mutex::new(None);

pub fn set_state(s: Result<(usize, f64, f64, usize, usize, String), String>) { if let Ok(mut st) = STATE.lock() { *st = Some(s); } }

/// One drafted round's steps: `inputs[i]` = the active lanes' chain inputs of step i.
pub fn record_steps(mc: &mut MarkovCache, inputs: &[Vec<i32>], stopped_from: Option<usize>) {
    for (step_idx, st) in inputs.iter().enumerate() {
        let stopped = stopped_from.is_some_and(|s0| step_idx >= s0);
        let (hits, hit) = mc.mirror_step(st, stopped);
        if stopped { continue; }
        INPUTS_HIT.fetch_add(hits, Relaxed);
        INPUTS_MISS.fetch_add(st.len() as u64 - hits, Relaxed);
        if hit { STEPS_HIT.fetch_add(1, Relaxed); } else { STEPS_MISS.fetch_add(1, Relaxed); }
        if let Ok(mut b) = BY_LANES.lock() { let i = st.len().clamp(1, 5) - 1; if hit { b[i].0 += 1; } else { b[i].1 += 1; } }
    }
}

fn row(o: &mut String, name: &str, help: &str, v: u64) { o.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n")); }

/// The section (empty unless the boot set the cache state).
pub fn render(o: &mut String) {
    let Some(st) = STATE.lock().ok().and_then(|s| s.clone()) else { return };
    row(o, "velogb10_dsv41_spec_plain_finish_total", "Verifies of 0 drafts (plain rounds, first tokens) that skipped the residual pass and its exchange (leg PH1; no lane can reject).",
        super::vocab::PLAIN_FINISH.load(Relaxed));
    match st {
        Err(why) => {
            o.push_str("# HELP velogb10_dsv41_markov_cache_off The drafter's Markov bias cache is off (the reason).\n# TYPE velogb10_dsv41_markov_cache_off gauge\n");
            o.push_str(&format!("velogb10_dsv41_markov_cache_off{{reason=\"{}\"}} 1\n", why.replace('"', "'").replace('\n', " ")));
        }
        Ok((k, mb, fill, groups, bad, label)) => {
            o.push_str("# HELP velogb10_dsv41_markov_cache_tokens Tokens whose Markov bias rows the drafter caches (per rank: MB, fill ms; the load-time bit check over every cached token).\n# TYPE velogb10_dsv41_markov_cache_tokens gauge\n");
            o.push_str(&format!("velogb10_dsv41_markov_cache_tokens{{list=\"{}\",mb=\"{mb:.1}\",fill_ms=\"{fill:.1}\",check_groups=\"{groups}\",check_mismatch=\"{bad}\"}} {k}\n", label.replace('"', "'")));
            row(o, "velogb10_dsv41_markov_steps_cached_total", "Draft chain steps whose every active lane's input token was cached: the Markov head was not read (host mirror of the kernel's rule).", STEPS_HIT.load(Relaxed));
            row(o, "velogb10_dsv41_markov_steps_computed_total", "Draft chain steps that read the Markov head (some active lane's input not cached).", STEPS_MISS.load(Relaxed));
            row(o, "velogb10_dsv41_markov_inputs_cached_total", "Chain inputs (lane-steps) whose token was cached.", INPUTS_HIT.load(Relaxed));
            row(o, "velogb10_dsv41_markov_inputs_uncached_total", "Chain inputs (lane-steps) whose token was not cached.", INPUTS_MISS.load(Relaxed));
            row(o, "velogb10_dsv41_markov_device_steps_cached_total", "Steps the kernel itself served from the cache (device counter, read every 64 drafted rounds; it also counts the boot calibration's draft steps).", DEV_HIT.load(Relaxed));
            row(o, "velogb10_dsv41_markov_device_steps_computed_total", "Steps the kernel itself computed (device counter, read with the one above).", DEV_MISS.load(Relaxed));
            o.push_str(&format!("# HELP velogb10_dsv41_markov_dyn_cap Dynamic session-vocabulary Markov ring slots per rank (--dsv41-markov-dyn-cache; leg P6).\n# TYPE velogb10_dsv41_markov_dyn_cap gauge\nvelogb10_dsv41_markov_dyn_cap {}\n", DYN_CAP.load(Relaxed)));
            row(o, "velogb10_dsv41_markov_dyn_fills_total", "Dynamic session-vocabulary Markov rows filled on-GPU during misses (leg P6).", DYN_FILLS.load(Relaxed));
            row(o, "velogb10_dsv41_markov_device_dyn_fills_total", "Dynamic Markov rows filled on-GPU as counted by d41_ds_mc_admit (leg P6).", DEV_DYN_FILLS.load(Relaxed));
            row(o, "velogb10_dsv41_markov_cache_early_skip_steps", "Markov chain steps skipped by conservative exact per-step confidence early-stop (host ground-truth from device stopped state; leg P6).", EARLY_SKIP_STEPS.load(Relaxed));
            row(o, "velogb10_dsv41_markov_device_early_skipped_total", "Markov head steps skipped on-GPU after conservative exact early-stop (device counter; leg P6).", DEV_EARLY_SKIP.load(Relaxed));
            if let Ok(b) = BY_LANES.lock() {
                o.push_str("# HELP velogb10_dsv41_markov_steps_by_lanes_total Chain steps by active lanes (5 = 5 or more) and outcome.\n# TYPE velogb10_dsv41_markov_steps_by_lanes_total counter\n");
                for (i, (h, m)) in b.iter().enumerate() {
                    o.push_str(&format!("velogb10_dsv41_markov_steps_by_lanes_total{{lanes=\"{}\",outcome=\"cached\"}} {h}\nvelogb10_dsv41_markov_steps_by_lanes_total{{lanes=\"{}\",outcome=\"computed\"}} {m}\n", i + 1, i + 1));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ph1_token_lists() {
        assert_eq!(parse_list("14 201,270\n# a comment\n223\t16").unwrap(), vec![14, 201, 270, 223, 16]);
        assert!(parse_list("14 x").is_err());
        assert_eq!(first_distinct(&[5, 5, 9, 200_000, 7, 9, 3], 3, 129_280), vec![5, 9, 7]);
        let b = first_distinct(&mtoks::BUILTIN, 1024, 129_280);
        assert_eq!(b.len(), 1024, "the built-in list: 1024 distinct ids below the vocabulary");
        assert_eq!(&b[..4], &[14, 201, 270, 223]);
    }
    #[test]
    fn dsv41_ph1_mbias_args_layout() {
        assert_eq!(std::mem::size_of::<MbiasArgs>(), 10 * 8 + 8 * 4);
        assert_eq!(std::mem::size_of::<AdmitArgs>(), 8 * 8 + 2 * 8 + 8 * 4);
    }
    #[test]
    fn dsv41_ph1_markov_metrics_render() {
        set_state(Ok((256, 66.2, 30.0, 320, 0, "builtin".into())));
        STEPS_HIT.fetch_add(3, Relaxed);
        let mut o = String::new();
        render(&mut o);
        for want in ["velogb10_dsv41_markov_cache_tokens{list=\"builtin\"", "velogb10_dsv41_markov_steps_cached_total ", "velogb10_dsv41_markov_steps_computed_total ",
                     "velogb10_dsv41_markov_device_steps_cached_total ", "velogb10_dsv41_markov_steps_by_lanes_total{lanes=\"1\",outcome=\"cached\"}"] {
            assert!(o.contains(want), "{want} missing:\n{o}");
        }
    }
}
