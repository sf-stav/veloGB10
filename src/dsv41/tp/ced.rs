//! DSV4.1 TP (leg FW): leg CE's exact CED prefill at TP, chunked, for one lane (`TpModel::prefill`).
//!
//! The schedule is CE's (`ced::schedule::Schedule`, a pure function of the config, the prompt length and the mode;
//! chunk boundaries never enter it): per layer L the query rows `[q0, P)`, the window rows `[w0, P)` and, for the
//! first decoder layer's KV source, the compressor + index keys on every row. Per chunk `[c0, c1)` every layer runs
//! the rows of its ranges inside the chunk, on the TP pieces of the decode forward (`fwd.rs`):
//! * the stream of a layer covers the rows `[q0 of the layer below, c1)` (the encoder: the whole chunk);
//! * attention-norm input on `[a0, c1)` (a0 = 0 for the KV-source pass, else w0), the KV-source pass, the window
//!   K/V (ring + linear rows; rows below w0 are never read by a query), then the query rows: mix (K slice riding the
//!   reduce), q path, indexer, attention over the rank's heads, wo -> one reduce over the query rows, the FFN side
//!   (inner-width experts, Engram partial) -> one reduce;
//! * a layer whose ranges miss the chunk ends the chunk (the cones shrink upwards).
//! Every prefill linear runs in the wide row family (`Fwd::wide_min` = 33: a shorter piece is padded), so the decode
//! state and the first-token logits do not depend on the chunking nor on the mode (CED == the every-layer prefill,
//! bit for bit: gated by the `ced` part of the program). `Mode::Exact` runs every layer on every row (the oracle and
//! the score mode); `Mode::Replay { keep }` is DeepSeek's opt-in bounded replay (approximate below 2,541 rows).
//! The DSpark taps of the prompt's last `TAP_ROWS` rows (inputs of layers 37-39, inside the cone by construction)
//! are written when asked, for the drafter's rings.

use super::fwd::{HeadWant, Seg, TpModel};
use super::frame::MIX_STRIDE;
use super::comm::{Dsv41Comm, TpOut};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::ced::schedule::{chunks_elastic, n_of, Mode, Schedule};
use crate::dsv41::fwd::glue;
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

/// The wide row family's minimum rows (leg CE: the backends' dense linears switch families above 32 rows).
pub const WIDE_MIN: usize = 33;

/// Whether `--dsv41-sp-prefill` is active (`auto` / `on` at `world` in {2, 3, 4, 6, 8} when SP RDMA transport is ready).
pub fn sp_prefill_mode_on(raw: Option<&str>, world: usize, comm_ready: bool) -> bool {
    let want = !matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("off"));
    want && matches!(world, 2 | 3 | 4 | 6 | 8) && comm_ready
}

/// U6a: whether LAGA BF16 attention-output All-to-All + local whole `wo_b` is active (`world` in {2, 3, 4, 6, 8}).
/// Default ON (`auto`) when `a2a_ready()`; `--dsv41-sp-laga off` disables.
pub fn sp_laga_mode_on(raw: Option<&str>, world: usize, a2a_ready: bool) -> bool {
    let want = !matches!(raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("off"));
    want && matches!(world, 2 | 3 | 4 | 6 | 8) && a2a_ready
}

/// Contiguous per-rank row slices `[r0, r1)` covering `0..n`, with internal boundaries aligned to 128-row M-tiles
/// and every slice `>= 64` rows (`> WIDE_MIN = 33`), so sharded wide GEMMs, norms, routers and mixers match non-SP bit-for-bit.
pub fn sp_row_blocks(n: usize, world: usize) -> Option<Vec<(usize, usize)>> {
    if !matches!(world, 2 | 3 | 4 | 6 | 8) || n < world * 128 { return None; }
    let tiles = n / 128;
    let base = tiles / world;
    let rem = tiles % world;
    let mut out = Vec::with_capacity(world);
    let mut cur = 0usize;
    for r in 0..world {
        let t = base + usize::from(r < rem);
        let next = if r + 1 == world { n } else { cur + t * 128 };
        if next <= cur || next - cur < 64 { return None; }
        out.push((cur, next));
        cur = next;
    }
    Some(out)
}

/// 4-row super-block ranges of `sp_row_blocks` (where internal boundaries are multiples of 128 and the tail is padded to `ceil(n/4)*4`).
pub fn sp_blocks_div4(blocks: &[(usize, usize)]) -> Vec<(usize, usize)> {
    blocks.iter().map(|&(a, b)| (a / 4, b.div_ceil(4))).collect()
}

/// Complementary `2 * world` mirror-zigzag stripes covering `0..nq`:
/// `front[r] = S_r` and `back[r] = S_{2W - 1 - r}`, partitioning the `ceil(nq / 8)` 8-row indexer blocks evenly across
/// `2 * world` stripes (`<= 1` 8-row block difference between any two ranks, `0` skew when `ceil(nq / 8)` is a multiple
/// of `2 * world`) while keeping every stripe `>= 56` rows (`> WIDE_MIN = 33`) and internal boundaries on 8-row multiples
/// (and 128-row multiples automatically whenever `nq` is a multiple of `2 * world * 128`).
pub fn sp_mirror_stripes(nq: usize, world: usize) -> Option<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
    let ns = 2 * world;
    if !matches!(world, 2 | 3 | 4 | 6 | 8) || nq < ns * 64 { return None; }
    let units = nq.div_ceil(8);
    let base = units / ns;
    let rem = units % ns;
    let mut s = Vec::with_capacity(ns);
    let mut cur = 0usize;
    for i in 0..ns {
        let prio = if i >= world { ns - 1 - i } else { world + i };
        let u = base + usize::from(prio < rem);
        let next = if i + 1 == ns { nq } else { (cur + u * 8).min(nq) };
        if next <= cur || next - cur < 56 { return None; }
        s.push((cur, next));
        cur = next;
    }
    let front = (0..world).map(|r| s[r]).collect();
    let back = (0..world).map(|r| s[ns - 1 - r]).collect();
    Some((front, back))
}

/// What a prefill did.
#[derive(Clone, Debug, Default)]
pub struct Prefilled {
    pub secs: f64,
    /// the last prompt row's row index in the final stream (`TpModel::head_logits(row, 1, ..)`)
    pub last_row: usize,
    /// layer-rows computed (query rows) and what the every-layer prefill computes
    pub layer_rows: u64,
    pub exact_rows: u64,
    /// the head exchange of the last prompt row, when asked
    pub head: Option<TpOut>,
}

/// Where the prefill writes the DSpark taps: (buffer bf16 [TAP_ROWS][ld], ld).
pub type TapDst = Option<(u64, usize)>;

impl TpModel<'_> {
    /// Prefill lane `lane` with the prompt `toks` in chunks of `chunk` rows under `mode`. A cold lane (position 0)
    /// computes every row; a lane holding a restored prefix state at k = its position (`restore`: the state's tokens
    /// must equal `toks[..k]`) resumes there (`Schedule::resumed`: an encoder-only state only past its decoder cone).
    pub fn prefill(&mut self, lane: usize, toks: &[u32], chunk: usize, mode: Mode, taps: TapDst, want_head: bool) -> Result<Prefilled> {
        self.prefill_part(lane, toks, toks.len(), None, None, chunk, mode, taps, want_head)
    }

    /// One call of a prefill that may span several calls (leg SE2: a long prompt is prefilled a few chunks per call so the
    /// other lanes keep decoding between calls). `toks` = the prompt's rows known so far (the lane's own `pos` rows plus the
    /// rows this call computes), `total` = the prompt's full length (the CED schedule is a function of it, so every call
    /// of one prefill uses the same one), `origin` = None on the first call (the lane's state at entry is the origin), or
    /// the (position, kind) the first call started from. A call whose rows end below `total` leaves the lane at its last
    /// chunk boundary (`pos`, `hist`, `kind`) and returns no final row. `full_at` (leg CE2, `Schedule::new_with`): the lowest
    /// mid-prompt position whose state must be full (the serve shell's lowest grid snapshot); every call of one prefill passes
    /// the same value, so they all run the same schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_part(&mut self, lane: usize, toks: &[u32], total: usize, origin: Option<(usize, crate::dsv41::ced::schedule::StateKind)>,
                        full_at: Option<usize>, chunk: usize, mode: Mode, taps: TapDst, want_head: bool) -> Result<Prefilled> {
        ensure!(self.layers.len() >= self.cfg.layers && self.layers[..self.cfg.layers].iter().all(|x| x.is_some()), "the prefill needs all trunk layers");
        ensure!(!toks.is_empty(), "an empty prompt");
        ensure!(toks.len() <= total, "prefill: {} rows known of a {total}-row prompt", toks.len());
        let k = self.lanes[lane].pos;
        ensure!(k < toks.len(), "lane {lane} holds {k} positions of the {} rows given: nothing to compute", toks.len());
        ensure!(self.lanes[lane].hist[..] == toks[..k], "lane {lane}: the state's {k} tokens are not a prefix of the prompt");
        let (k0, kind0) = origin.unwrap_or((k, self.lanes[lane].kind));
        let sched = if k0 == 0 { Schedule::new_with(self.cfg, total, mode, full_at)? } else { Schedule::resumed_with(self.cfg, total, mode, k0, kind0, full_at)? };
        let wm = std::mem::replace(&mut self.f.wide_min, WIDE_MIN);
        let t = std::time::Instant::now();
        let r = self.prefill_sched(lane, toks, total, k, chunk.max(1).min(self.max_rows), &sched, taps, want_head);
        self.f.wide_min = wm;
        self.win_lo = 0;
        let mut out = r?;
        out.secs = t.elapsed().as_secs_f64();
        // what the every-layer prefill computes for the rows of this call (a first call also counts the rows its restored state saved)
        let exact = ((toks.len() - if origin.is_some() { k } else { 0 }) * self.cfg.layers) as u64;
        super::metrics::ced_rows(out.layer_rows, exact);
        out.exact_rows = exact;
        Ok(out)
    }

    fn prefill_sched(&mut self, lane: usize, toks: &[u32], p: usize, start: usize, chunk: usize, sched: &Schedule, taps: TapDst, want_head: bool) -> Result<Prefilled> {
        let cfg = self.cfg;
        let (d, hc) = (cfg.hidden, cfg.hc);
        let t0 = p.saturating_sub(super::spec::TAP_ROWS);
        let (fa, fb) = (self.frames[0].ptr(), self.frames[1].ptr());
        let sp_raw = crate::opts::lookup("dsv41-sp-prefill").and_then(crate::opts::get);
        let sp_on = sp_prefill_mode_on(sp_raw.as_deref(), self.world, self.comm.sp_ready());
        let laga_raw = crate::opts::lookup("dsv41-sp-laga").and_then(crate::opts::get);
        let laga_on = sp_on && sp_laga_mode_on(laga_raw.as_deref(), self.world, self.comm.a2a_ready());
        let hc_whole = if sp_on { self.hc_whole_ok()? } else { false };
        if matches!(self.world, 2 | 3 | 4 | 6 | 8) {
            static SAID_SP: std::sync::Once = std::sync::Once::new();
            let (rk, wd) = (self.rank, self.world);
            SAID_SP.call_once(|| crate::logq::send(false, format!(
                "[dsv41] P7SP sequence-parallel prefill: {} (--dsv41-sp-prefill {}, rank {rk}/{wd}, stream-shard hc_whole={hc_whole}, LAGA={laga_on})",
                if sp_on { "ON" } else { "off" }, sp_raw.as_deref().unwrap_or("auto")
            )));
        }
        let mut out = Prefilled::default();
        let mut final_row = None;
        // leg PP1: the rows after this call's last chunk (the next call's chunk, sent with this call by the head)
        let ahead = std::mem::take(&mut self.pf_ahead);
        let plan = chunks_elastic(start, toks.len(), chunk, self.tail_merge);
        if self.tail_merge > 0 && plan.len() < crate::dsv41::ced::schedule::chunks(start, toks.len(), chunk).len() { super::metrics::PF_TAIL_MERGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let has_vl = !self.lanes[lane].vl_types.is_empty();
        for (c0, c1) in plan {
            let rows = c1 - c0;
            self.swap_bump();
            self.segs = vec![Seg { lane, row0: 0, n: rows, p0: c0 }];
            self.upload_rows(&toks[c0..c1])?;
            let (pos_all, ids) = (self.pos_dev, self.ids_dev);
            // leg PP1: the next chunk's Engram rows are read while this chunk computes (its step then finds them in the cache);
            // the reads are planned once layer 0 is queued, so the GPU has work while the host plans them
            let mut next: Option<&[u32]> = Some(if c1 < toks.len() { &toks[c1..(c1 + chunk).min(toks.len())] } else { &ahead });
            let (erows, mut next_live, chunk_has_vl, mut h) = if !has_vl {
                // Zero-overhead text-only path: 100% identical calls and collectives to pre-vision master.
                let er = self.engram_rows(&toks[c0..c1])?;
                let h0 = self.embed(ids, rows)?;
                (er, None, false, h0)
            } else {
                let live_chunk: Vec<bool> = (c0..c1).map(|pos| self.lanes[lane].vl_types.get(pos).copied().unwrap_or(-1) < 0).collect();
                let ch_vl = live_chunk.iter().any(|&lv| !lv);
                let er = self.engram_rows_vl(&toks[c0..c1], Some(&live_chunk))?;
                let nl: Option<Vec<bool>> = next.map(|n| (c1..c1 + n.len()).map(|pos| self.lanes[lane].vl_types.get(pos).copied().unwrap_or(-1) < 0).collect());
                let h0 = if ch_vl {
                    let mask_u8: Vec<u8> = live_chunk.iter().map(|&lv| (!lv) as u8).collect();
                    ensure!(mask_u8.len() <= self.vl_mask_buf.bytes, "vl_mask_buf {} B < {rows} rows", self.vl_mask_buf.bytes);
                    self.f.g.h2d_async(self.vl_mask_buf.ptr(), &mask_u8)?;
                    self.embed_vl_chunk(lane, ids, c0, rows)?
                } else {
                    self.embed(ids, rows)?
                };
                (er, nl, ch_vl, h0)
            };
            let mut pre = self.f.b.take(rows * 24 * 4)?;
            self.k.fill_pre(self.cs, pre, 24, rows, hc)?;
            let mut sharded: Option<Vec<(usize, usize)>> = if sp_on && hc_whole { sp_row_blocks(rows, self.world) } else { None };
            let mut hrow0 = c0;
            let mut kv_next: Option<u64> = None;
            let mut sel_of: HashMap<usize, (u64, usize)> = HashMap::new();
            let mut cands: Option<CandRows> = None;
            let mut last_q = None;
            for l in 0..cfg.layers {
                // NIGHT1: inside a prefill chunk, every PASS_EVERY_LAYERS layers (one atomic load on a cool box; a hot box
                // drains the queued work, then paces / holds, so a long chunk cannot push the board past the hard limit)
                if l > 0 && l % crate::thermal::PASS_EVERY_LAYERS == 0 { let g = self.f.g; crate::thermal::pace_pass(self.gm, || g.sync())?; }
                if l == 1 {
                    if let Some(n) = next.take() {
                        if !has_vl { self.engram_ahead(lane, n)?; } else { self.engram_ahead_vl(lane, n, next_live.take().as_deref())?; }
                    }
                }
                self.swap_bump();
                let cr = sched.chunk(l, c0, c1);
                let (a0, na, nw, nq) = (cr.a.0, n_of(cr.a), n_of(cr.w), n_of(cr.q));
                if na == 0 { break; }
                ensure!(a0 >= hrow0, "layer {l}: needs rows from {a0}, the stream starts at {hrow0}");
                let want_tap = taps.is_some() && cfg.dspark_targets.contains(&l) && t0 < c1;
                let want_last = c1 == p && l + 1 == cfg.layers;
                let aligned_ranges = a0 == hrow0
                    && cr.kv.map_or(true, |kv| kv.0 == a0 && n_of(kv) == na)
                    && (nw == 0 || (cr.w.0 == a0 && nw == na))
                    && (nq == 0 || (cr.q.0 == a0 && nq == na));
                let layer_blocks = if sp_on && hc_whole && aligned_ranges && !want_tap && !want_last {
                    sp_row_blocks(na, self.world)
                } else {
                    None
                };
                if sharded.as_ref() != layer_blocks.as_ref() {
                    if let Some(prev) = sharded.take() {
                        let mut bufs = vec![(h, hc * d * 2), (pre, 24 * 4)];
                        if let Some(kvb) = kv_next { bufs.push((kvb, (hc + 1) * d * 2)); }
                        self.comm.sp_all_gather_multi(self.cs, &bufs, &prev)?;
                    }
                    sharded = layer_blocks;
                }
                if let Some(ref blocks) = sharded {
                    use crate::dsv41::gpu::indexer as ki;
                    super::xchg::sp_layer();
                    let (r0, r1) = blocks[self.rank];
                    let my_n = r1 - r0;
                    if let Some(kvb) = kv_next.take() {
                        ensure!(hrow0 == c0, "an Engram layer inside a decoder cone");
                        let h_new = self.f.b.take(na * hc * d * 2)?;
                        self.engram_gate_into(l, h + (r0 * hc * d * 2) as u64, kvb + (r0 * (hc + 1) * d * 2) as u64, my_n, h_new + (r0 * hc * d * 2) as u64)?;
                        h = h_new;
                    }
                    let ratio = cfg.ratio(l);
                    let compressed = ratio > 0;
                    let hd = cfg.head_dim;
                    let ql = cfg.q_lora_rank;
                    let an = self.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
                    let ain_own = self.f.b.take(my_n * d * 2)?;
                    self.hc_pre_norm_into(h + (r0 * hc * d * 2) as u64, pre + (r0 * 24 * 4) as u64, 24, &an, my_n, ain_own)?;
                    let mut front_gathers: Vec<(u64, usize)> = Vec::with_capacity(5);
                    let comp_proj = if let Some(kv) = cr.kv.filter(|&kv| n_of(kv) > 0) {
                        let nkv = n_of(kv);
                        if ratio >= 2 {
                            let (kvp, scp) = (self.f.b.take(nkv * hd * 4)?, self.f.b.take(nkv * hd * 4)?);
                            let lw = &self.layers[l].as_ref().unwrap().w;
                            self.f.lin_rows("comp.wkv", lw, C::CompWkv, None, ain_own, d, d, my_n, kvp + (r0 * hd * 4) as u64, hd, true)?;
                            let lw = &self.layers[l].as_ref().unwrap().w;
                            self.f.lin_rows("comp.wgate", lw, C::CompWgate, None, ain_own, d, d, my_n, scp + (r0 * hd * 4) as u64, hd, true)?;
                            front_gathers.push((kvp, hd * 4));
                            front_gathers.push((scp, hd * 4));
                            Some((kv.0, nkv, kvp, scp, 4usize))
                        } else {
                            let kvp = self.f.b.take(nkv * hd * 2)?;
                            let lw = &self.layers[l].as_ref().unwrap().w;
                            self.f.lin_rows("comp.wkv", lw, C::CompWkv, None, ain_own, d, d, my_n, kvp + (r0 * hd * 2) as u64, hd, false)?;
                            front_gathers.push((kvp, hd * 2));
                            Some((kv.0, nkv, kvp, 0u64, 2usize))
                        }
                    } else {
                        None
                    };
                    let win_proj = if nw > 0 {
                        let raw = self.f.b.take(nw * hd * 2)?;
                        let lw = &self.layers[l].as_ref().unwrap().w;
                        self.f.lin_rows("attn.wkv", lw, C::AttnWkv, None, ain_own, d, d, my_n, raw + (r0 * hd * 2) as u64, hd, false)?;
                        front_gathers.push((raw, hd * 2));
                        Some((cr.w.0, raw))
                    } else {
                        None
                    };
                    if nq == 0 {
                        if !front_gathers.is_empty() {
                            self.comm.sp_all_gather_multi(self.cs, &front_gathers, blocks)?;
                        }
                        if let Some((kv0, nkv, kvp, scp, es)) = comp_proj {
                            self.segs = vec![Seg { lane, row0: 0, n: nkv, p0: kv0 }];
                            self.compressor_write_proj(l, kvp, scp, es, nkv)?;
                        }
                        if let Some((w0, raw)) = win_proj {
                            self.segs = vec![Seg { lane, row0: 0, n: nw, p0: w0 }];
                            self.window_write_raw(l, raw, nw, compressed)?;
                        }
                        break;
                    }
                    let q0 = cr.q.0;
                    out.layer_rows += nq as u64;
                    let rec_a_own = self.f.b.take(my_n * 24 * 4)?;
                    self.mix_local_into(l, true, h + (r0 * hc * d * 2) as u64, my_n, rec_a_own)?;
                    let qr = self.f.b.take(nq * ql * 2)?;
                    self.q_lora_into(l, ain_own, my_n, qr + (r0 * ql * 2) as u64)?;
                    front_gathers.push((qr, ql * 2));
                    let wp_pre = if compressed && cfg.index_source_of(l) == Some(l) {
                        let wp = self.f.b.take(nq * ki::HEADS * 2)?;
                        let lw = &self.layers[l].as_ref().unwrap().w;
                        self.f.lin_rows("tp.prefill.idx.weights_proj", lw, C::IdxWeightsProj, None, ain_own, d, d, my_n, wp + (r0 * ki::HEADS * 2) as u64, ki::HEADS, false)?;
                        front_gathers.push((wp, ki::HEADS * 2));
                        Some(wp)
                    } else {
                        None
                    };
                    self.comm.sp_all_gather_multi(self.cs, &front_gathers, blocks)?;
                    if let Some((kv0, nkv, kvp, scp, es)) = comp_proj {
                        self.segs = vec![Seg { lane, row0: 0, n: nkv, p0: kv0 }];
                        self.compressor_write_proj(l, kvp, scp, es, nkv)?;
                    }
                    if let Some((w0, raw)) = win_proj {
                        self.segs = vec![Seg { lane, row0: 0, n: nw, p0: w0 }];
                        self.window_write_raw(l, raw, nw, compressed)?;
                    }
                    self.segs = vec![Seg { lane, row0: 0, n: nq, p0: q0 }];
                    let posq = pos_all + ((q0 - c0) * 4) as u64;
                    let mut sel = None;
                    let mut src = None;
                    if compressed {
                        let s = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
                        src = Some(s);
                        let isrc = cfg.index_source_of(l).context("no index source")?;
                        if isrc == l {
                            let keep = self.indexer_prefill_sp(l, s, qr, 0, wp_pre, posq, nq, q0, &mut cands)?;
                            sel_of.insert(l, (keep, q0));
                        }
                        let (sl, s0) = *sel_of.get(&isrc).with_context(|| format!("layer {l}: index source {isrc} has no selection in this chunk"))?;
                        ensure!(q0 >= s0, "layer {l}: query rows from {q0} before its index source's rows ({s0})");
                        sel = Some((sl + ((q0 - s0) * 512 * 4) as u64, 512usize, 512usize));
                    }
                    self.win_lo = cr.win_lo;
                    let ao_own = self.f.b.take(my_n * d * 2)?;
                    let use_laga = laga_on && self.layers[l].as_ref().unwrap().w.lin(C::AttnWoB)
                        .is_ok_and(|w| matches!(w.0.shard, crate::dsv41::weights::Shard::Whole));
                    if use_laga {
                        // U6a (LAGA): 1-RTT BF16 All-to-All on `wo_a` output + local whole `wo_b` on `my_n` rows (TP2/TP3/TP4/TP6/TP8)
                        let mid = self.attn_core_mid(l, qr, posq, nq, src, sel)?;
                        self.win_lo = 0;
                        let max_my_n = blocks.iter().map(|&(a, b)| b - a).max().unwrap_or(0);
                        let (grp, max_ng) = super::xchg::D41LagaGroups::for_world(cfg, self.world);
                        let my_cols = self.rg.ng * cfg.o_lora_rank;
                        let stride_cols = max_ng * cfg.o_lora_rank;
                        let all_cols = cfg.o_groups * cfg.o_lora_rank;
                        let slice_bytes = max_my_n * stride_cols * 2;
                        let a2a_src = self.f.b.take(self.world * slice_bytes)?;
                        let a2a_dst = self.f.b.take(self.world * slice_bytes)?;
                        let mid_full = self.f.b.take(my_n * all_cols * 2)?;
                        super::xchg::laga_pack_b16(self.cs.stream as u64, mid, a2a_src, max_my_n, self.world, my_cols, stride_cols, blocks)?;
                        self.comm.all_to_all(self.cs, a2a_src, a2a_dst, slice_bytes)?;
                        super::xchg::laga_unpack_b16(self.cs.stream as u64, a2a_dst, mid_full, max_my_n, my_n, self.world, stride_cols, all_cols, cfg.o_lora_rank, grp)?;
                        self.wo_b_whole(l, mid_full, my_n, ao_own)?;
                        super::xchg::sp_laga_layer();
                    } else {
                        self.attn_core(l, qr, posq, nq, src, sel, fa)?;
                        self.win_lo = 0;
                        self.comm.sp_reduce_scatter(self.cs, fa, d, blocks)?;
                        glue::add_round(self.f.g, fa + (r0 * d * 4) as u64, 0, ao_own, my_n, d)?;
                    }
                    let xm_own = self.f.b.take(my_n * hc * d * 2)?;
                    self.hc_post_into(ao_own, h + (r0 * hc * d * 2) as u64, rec_a_own, my_n, xm_own)?;
                    let engram_next = cfg.engram_layers.contains(&(l + 1));
                    let extra = if engram_next { (hc + 1) * d } else { 0 };
                    let rec_f = self.f.b.take(nq * 24 * 4)?;
                    self.mix_local_into(l, false, xm_own, my_n, rec_f + (r0 * 24 * 4) as u64)?;
                    let fnm = self.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
                    let nq4 = nq.div_ceil(4) * 4;
                    let fin = self.f.b.take(nq4 * d * 2)?;
                    self.hc_pre_norm_into(xm_own, rec_a_own, 24, &fnm, my_n, fin + (r0 * d * 2) as u64)?;
                    let (rid, rw) = (self.f.b.take(nq4 * cfg.top_k * 4)?, self.f.b.take(nq4 * cfg.top_k * 4)?);
                    let vmask = if chunk_has_vl { self.vl_mask_buf.ptr() + (q0 - c0 + r0) as u64 } else { 0 };
                    self.router_vl_into(l, fin + (r0 * d * 2) as u64, my_n, vmask, rid + (r0 * cfg.top_k * 4) as u64, rw + (r0 * cfg.top_k * 4) as u64)?;
                    let b4 = sp_blocks_div4(blocks);
                    self.comm.sp_all_gather_multi(self.cs, &[(fin, 4 * d * 2), (rid, 16 * cfg.top_k), (rw, 16 * cfg.top_k)], &b4)?;
                    self.ffn_core(l, fin, nq, rid, rw, fb)?;
                    self.comm.sp_reduce_scatter(self.cs, fb, d, blocks)?;
                    if engram_next {
                        ensure!(q0 == c0, "an Engram layer inside a decoder cone");
                        let (e, stride) = erows.context("an Engram layer without the Engram runtime")?;
                        let li = cfg.engram_layers.iter().position(|&x| x == l + 1).unwrap();
                        let fb_eng = fb + (nq * (d + MIX_STRIDE) * 4) as u64;
                        self.engram_partial(l + 1, e + (li * stride) as u64, nq, fb_eng)?;
                        self.comm.sp_reduce_scatter(self.cs, fb_eng, extra, blocks)?;
                        let kvb = self.f.b.take(nq * extra * 2)?;
                        glue::add_round(self.f.g, fb_eng + (r0 * extra * 4) as u64, 0, kvb + (r0 * extra * 2) as u64, my_n, extra)?;
                        kv_next = Some(kvb);
                    }
                    let fo_own = self.f.b.take(my_n * d * 2)?;
                    glue::add_round(self.f.g, fb + (r0 * d * 4) as u64, 0, fo_own, my_n, d)?;
                    let h_next = self.f.b.take(nq * hc * d * 2)?;
                    self.hc_post_into(fo_own, xm_own, rec_f + (r0 * 24 * 4) as u64, my_n, h_next + (r0 * hc * d * 2) as u64)?;
                    h = h_next;
                    pre = rec_f;
                    hrow0 = q0;
                    if l + 1 == cfg.layers { last_q = Some(q0); }
                    continue;
                }
                let hoff = |r: usize| ((r - hrow0) * hc * d * 2) as u64;
                let poff = |r: usize| ((r - hrow0) * 24 * 4) as u64;
                if let Some(kvb) = kv_next.take() {
                    ensure!(hrow0 == c0, "an Engram layer inside a decoder cone");
                    h = self.engram_gate(l, h, kvb, c1 - hrow0)?;
                }
                if let (Some((tb, tld)), Some(ti)) = (taps, cfg.dspark_targets.iter().position(|&x| x == l)) {
                    let lo = t0.max(hrow0);
                    if lo < c1 { self.k.tap(self.cs, h, hc, d, lo - hrow0, c1 - lo, tb, lo - t0, tld, ti * d)?; }
                }
                let ratio = cfg.ratio(l);
                let compressed = ratio > 0;
                let an = self.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
                let ain = self.f.hc_pre_norm(h + hoff(a0), pre + poff(a0), 24, &an, na)?;
                let arow = |r: usize| ain + ((r - a0) * d * 2) as u64;
                if let Some(kv) = cr.kv {
                    if n_of(kv) > 0 {
                        self.segs = vec![Seg { lane, row0: 0, n: n_of(kv), p0: kv.0 }];
                        self.compressor(l, arow(kv.0), n_of(kv))?;
                    }
                }
                if nw > 0 {
                    self.segs = vec![Seg { lane, row0: 0, n: nw, p0: cr.w.0 }];
                    self.window(l, arow(cr.w.0), nw, compressed)?;
                }
                if nq == 0 { break; }
                let q0 = cr.q.0;
                out.layer_rows += nq as u64;
                self.segs = vec![Seg { lane, row0: 0, n: nq, p0: q0 }];
                let posq = pos_all + ((q0 - c0) * 4) as u64;
                let hq = h + hoff(q0);
                self.mix_partial(self.layers[l].as_ref().unwrap().mix_a, hq, nq, fa + (nq * d * 4) as u64)?;
                let qa = arow(q0);
                let qblocks = if sp_on { sp_row_blocks(nq, self.world) } else { None };
                if qblocks.is_some() { super::xchg::sp_layer(); }
                let qr = if let Some(ref qb) = qblocks {
                    let ql = cfg.q_lora_rank;
                    let (r0, r1) = qb[self.rank];
                    let qr = self.f.b.take(nq * ql * 2)?;
                    self.q_lora_into(l, qa + (r0 * d * 2) as u64, r1 - r0, qr + (r0 * ql * 2) as u64)?;
                    self.comm.sp_all_gather_multi(self.cs, &[(qr, ql * 2)], qb)?;
                    qr
                } else {
                    self.q_lora(l, qa, nq)?
                };
                let mut sel = None;
                let mut src = None;
                if compressed {
                    let s = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
                    src = Some(s);
                    let isrc = cfg.index_source_of(l).context("no index source")?;
                    if isrc == l {
                        let keep = if sp_on && sp_mirror_stripes(nq, self.world).is_some() {
                            self.indexer_prefill_sp(l, s, qr, qa, None, posq, nq, q0, &mut cands)?
                        } else {
                            let sl = self.indexer_prefill(l, s, qr, qa, posq, nq, q0, &mut cands)?;
                            let keep = self.sel_keep(l, nq)?;
                            self.f.g.d2d(keep, sl, nq * 512 * 4)?;
                            keep
                        };
                        sel_of.insert(l, (keep, q0));
                    }
                    let (sl, s0) = *sel_of.get(&isrc).with_context(|| format!("layer {l}: index source {isrc} has no selection in this chunk"))?;
                    ensure!(q0 >= s0, "layer {l}: query rows from {q0} before its index source's rows ({s0})");
                    sel = Some((sl + ((q0 - s0) * 512 * 4) as u64, 512usize, 512usize));
                }
                self.win_lo = cr.win_lo;
                self.attn_core(l, qr, posq, nq, src, sel, fa)?;
                self.win_lo = 0;
                self.reduce_act(fa, nq * (d + MIX_STRIDE))?;
                let rec_a = self.mix_finish(l, true, fa + (nq * d * 4) as u64, nq)?;
                let ao = self.f.b.take(nq * d * 2)?;
                glue::add_round(self.f.g, fa, 0, ao, nq, d)?;
                let xm = self.f.hc_post(ao, hq, rec_a, nq)?;
                let engram_next = cfg.engram_layers.contains(&(l + 1));
                let extra = if engram_next { (hc + 1) * d } else { 0 };
                self.mix_partial(self.layers[l].as_ref().unwrap().mix_f, xm, nq, fb + (nq * d * 4) as u64)?;
                let fnm = self.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
                let fin = self.f.hc_pre_norm(xm, rec_a, 24, &fnm, nq)?;
                let (rid, rw) = if let Some(ref qb) = qblocks {
                    let (r0, r1) = qb[self.rank];
                    let nq4 = nq.div_ceil(4) * 4;
                    let (rid, rw) = (self.f.b.take(nq4 * cfg.top_k * 4)?, self.f.b.take(nq4 * cfg.top_k * 4)?);
                    let vmask = if chunk_has_vl { self.vl_mask_buf.ptr() + (q0 - c0 + r0) as u64 } else { 0 };
                    self.router_vl_into(l, fin + (r0 * d * 2) as u64, r1 - r0, vmask, rid + (r0 * cfg.top_k * 4) as u64, rw + (r0 * cfg.top_k * 4) as u64)?;
                    let b4 = sp_blocks_div4(qb);
                    self.comm.sp_all_gather_multi(self.cs, &[(rid, 16 * cfg.top_k), (rw, 16 * cfg.top_k)], &b4)?;
                    (rid, rw)
                } else if chunk_has_vl {
                    self.f.router_vl(&self.layers[l].as_ref().unwrap().w, fin, nq, self.vl_mask_buf.ptr() + (q0 - c0) as u64)?
                } else {
                    self.f.router(&self.layers[l].as_ref().unwrap().w, fin, nq)?
                };
                self.ffn_core(l, fin, nq, rid, rw, fb)?;
                if engram_next {
                    ensure!(q0 == c0, "an Engram layer inside a decoder cone");
                    let (e, stride) = erows.context("an Engram layer without the Engram runtime")?;
                    let li = cfg.engram_layers.iter().position(|&x| x == l + 1).unwrap();
                    self.engram_partial(l + 1, e + (li * stride) as u64, nq, fb + (nq * (d + MIX_STRIDE) * 4) as u64)?;
                }
                self.reduce_act(fb, nq * (d + MIX_STRIDE + extra))?;
                let rec_f = self.mix_finish(l, false, fb + (nq * d * 4) as u64, nq)?;
                let fo = self.f.b.take(nq * d * 2)?;
                glue::add_round(self.f.g, fb, 0, fo, nq, d)?;
                h = self.f.hc_post(fo, xm, rec_f, nq)?;
                if engram_next {
                    let kvb = self.f.b.take(nq * extra * 2)?;
                    glue::add_round(self.f.g, fb + (nq * (d + MIX_STRIDE) * 4) as u64, 0, kvb, nq, extra)?;
                    kv_next = Some(kvb);
                }
                pre = rec_f;
                hrow0 = q0;
                if l + 1 == cfg.layers { last_q = Some(q0); }
            }
            if let Some(n) = next.take() {
                if !has_vl { self.engram_ahead(lane, n)?; } else { self.engram_ahead_vl(lane, n, next_live.take().as_deref())?; }
            }
            self.engram_check()?;
            let ln = &mut self.lanes[lane];
            ln.pos = c1;
            ln.hist.extend_from_slice(&toks[c0..c1]);
            ln.kind = sched.state_kind_at(c1);
            if c1 == p {
                ln.vl_aligner = None;
                ln.vl_row_ops.clear();
                let q0 = last_q.context("the last chunk did not reach the last layer")?;
                self.last = Some((h, pre));
                final_row = Some(p - 1 - q0);
                self.segs = vec![Seg { lane, row0: 0, n: p - q0, p0: q0 }];
            }
        }
        if toks.len() < p { return Ok(out); }
        out.last_row = final_row.context("no final row")?;
        if want_head {
            let (h, pre) = self.last.unwrap();
            let r = out.last_row;
            let (d, hc) = (self.cfg.hidden, self.cfg.hc);
            let mark = self.f.b.off;
            let o = self.head(h + (r * hc * d * 2) as u64, pre + (r * 24 * 4) as u64, 24, 1, &[r], &[-1])?;
            self.f.b.off = mark;
            out.head = o.first().copied();
        }
        self.comm.check("after a prefill")?;
        let _ = HeadWant::None;
        Ok(out)
    }

    /// The selection buffer of an index source, sized for `rows` (the prefill's query rows may exceed the decode size).
    fn sel_keep(&mut self, l: usize, rows: usize) -> Result<u64> {
        let b = self.sel_bufs.get(&l).context("index source without a selection buffer")?;
        ensure!(b.bytes >= rows * 512 * 4, "layer {l}: selection buffer {} B < {} rows", b.bytes, rows);
        Ok(b.ptr())
    }

    /// The indexer of one prefill layer (one lane, `nq` query rows at positions q0..): 8-row blocks, one key chunk
    /// (KI's prefill shape), as CE's chunk driver: the candidate source keeps its rows' candidate bitmaps; a candidate
    /// user builds the union of ITS OWN row blocks from them (`d41_idx_union` on its rows), so any cone start works and
    /// a row's selection never depends on which rows shared its call.
    #[allow(clippy::too_many_arguments)]
    fn indexer_prefill(&mut self, l: usize, s: usize, qr: u64, x: u64, pos: u64, nq: usize, q0: usize, cands: &mut Option<CandRows>) -> Result<u64> {
        use crate::dsv41::gpu::indexer as ki;
        let cfg = self.cfg;
        let (d, ql) = (cfg.hidden, cfg.q_lora_rank);
        let cand_out = cfg.candidate_source == Some(l);
        let cand_in = cfg.uses_candidates(l);
        let w = &self.layers[l].as_ref().unwrap().w;
        let iq = self.f.b.take(nq * ki::HEADS * ki::HEAD_DIM * 2)?;
        self.f.lin_rows("tp.prefill.idx.wq_b", w, C::IdxWqB, None, qr, ql, ql, nq, iq, ki::HEADS * ki::HEAD_DIM, false)?;
        let wp = self.f.b.take(nq * ki::HEADS * 2)?;
        let w = &self.layers[l].as_ref().unwrap().w;
        self.f.lin_rows("tp.prefill.idx.weights_proj", w, C::IdxWeightsProj, None, x, d, d, nq, wp, ki::HEADS, false)?;
        let lane = self.segs[0].lane;
        let st = self.lanes[lane].src.get(&s).context("index source state")?;
        let ratio = st.ratio;
        let keys_at_end = (q0 + nq) / ratio.max(1) as usize;
        ensure!(keys_at_end <= st.n_groups, "layer {l}: index keys up to {keys_at_end} needed, the source holds {}", st.n_groups);
        let (kq, ks) = (st.iq.ptr(), st.is.ptr());
        let r = 8usize;
        let rbs: Vec<(i32, i32)> = (0..nq).step_by(r).map(|i| (i as i32, (nq - i).min(r) as i32)).collect();
        let n_rb = rbs.len();
        let cap = keys_at_end.max(1);
        let ucap = r * ki::CAND_BLOCKS;
        let sz = ki::sizes(n_rb, r, 1, nq, cap, ucap);
        let rbv: Vec<u64> = rbs.iter().flat_map(|&(r0, n)| [kq, ks, (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
        let rbd = self.f.b.take(rbv.len() * 8)?;
        self.f.g.h2d_async(rbd, &rbv)?;
        let sel = self.f.b.take(sz.sel_i32 * 4)?;
        let cnt = self.f.b.take(nq * 4)?;
        let part = self.f.b.take(sz.part_u64.max(1) * 8)?;
        let pcnt = self.f.b.take(sz.pcnt_i32.max(1) * 4)?;
        let bmax = if cand_out { self.f.b.take(sz.bmax_u16.max(1) * 2)? } else { 0 };
        let cand = self.f.b.take(sz.cand_i32.max(1) * 4)?;
        let ccnt = self.f.b.take(nq * 4)?;
        let err = self.f.b.take(4)?;
        self.f.g.memset(err, 4, 0)?;
        let mut a = ki::IdxArgs::default();
        a.iq = iq; a.w = wp; a.pos = pos; a.rbs = rbd; a.sel = sel; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt; a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt;
        a.err = err; a.tick = 0;
        a.n_rb = n_rb as i32; a.rows_total = nq as i32; a.iq_ld = (ki::HEADS * ki::HEAD_DIM) as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
        a.ratio = ratio as i32; a.write_bmax = cand_out as i32;
        a.n_keys_cap = cap as i32; a.nb_cap = cap.div_ceil(ki::CAND_BLOCK) as i32; a.nbw_cap = (a.nb_cap as usize).div_ceil(32) as i32;
        a.g = 1; a.min_tiles = 8; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = self.f.rope.ki;
        let path;
        if cand_in {
            let c = cands.as_ref().with_context(|| format!("layer {l}: no candidate rows of layer {:?} in this chunk", cfg.candidate_source))?;
            ensure!(q0 >= c.row0 && q0 + nq <= c.row0 + c.n, "layer {l}: rows {q0}..{} outside the candidate rows {}..{}", q0 + nq, c.row0, c.row0 + c.n);
            let off = q0 - c.row0;
            let uni = self.f.b.take(n_rb * ucap * 4)?;
            let ucnt = self.f.b.take(n_rb * 4)?;
            let mut u = a;
            u.pos = c.pos.ptr() + (off * 4) as u64;
            u.bm = c.bm.ptr() + (off * c.nbw_cap * 4) as u64;
            u.n_keys_cap = c.n_keys_cap as i32; u.nb_cap = c.nb_cap as i32; u.nbw_cap = c.nbw_cap as i32; u.ratio = c.ratio as i32;
            u.uni = uni; u.ucnt = ucnt;
            ki::union(&self.f.g.l(), &u)?;
            a.uni = uni; a.ucnt = ucnt; a.mode = 1;
            a.n_keys_cap = c.n_keys_cap as i32; a.nb_cap = c.nb_cap as i32; a.nbw_cap = c.nbw_cap as i32;
            ki::score(&self.f.g.l(), &a)?;
            path = "tp.prefill.idx (union of own rows + mode 1)";
        } else if cand_out {
            let bm = self.f.g.buf(sz.bm_u32.max(1) * 4)?;
            let pb = self.f.g.buf(nq.max(1) * 4)?;
            self.f.g.d2d(pb.ptr(), pos, nq * 4)?;
            a.bm = bm.ptr();
            a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
            ki::score(&self.f.g.l(), &a)?;
            ki::cand(&self.f.g.l(), &a)?;
            *cands = Some(CandRows { row0: q0, n: nq, bm, pos: pb, nbw_cap: a.nbw_cap as usize, nb_cap: a.nb_cap as usize, n_keys_cap: cap, ratio });
            path = "tp.prefill.idx (scores + candidate blocks)";
        } else {
            a.bm = self.f.b.take(4)?; a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
            ki::score(&self.f.g.l(), &a)?;
            path = "tp.prefill.idx (prefill shape)";
        }
        crate::dsv41::fwd::receipts::kernel(path, "d41_idx_score_r8", nq);
        if self.f.checks {
            let e = self.f.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "indexer layer {l}: kernel error word {e}");
        }
        Ok(sel)
    }

    /// Leg P7SP: Mirror-Zigzag causal indexer prefill across `self.world` ranks (`sp_mirror_stripes`):
    /// rank `r` computes `idx.wq_b`, (`idx.weights_proj` unless `wp_pre` was already gathered), and `d41_idx_score_r8`
    /// (+ `d41_idx_cand` on `L20` / `d41_idx_union` on `L24, 28, 32, 36`) ONLY on its two complementary stripes
    /// `front[r]` and `back[r]`, writing directly into `sel_keep(l, nq)` (and `bm` on `L20`), then all-gathering
    /// `sel` (+ `bm`) over the pipelined RDMA transport. 100% bit-identical to `indexer_prefill`.
    #[allow(clippy::too_many_arguments)]
    fn indexer_prefill_sp(&mut self, l: usize, s: usize, qr: u64, x: u64, wp_pre: Option<u64>, pos: u64, nq: usize, q0: usize,
                          cands: &mut Option<CandRows>) -> Result<u64> {
        use crate::dsv41::gpu::indexer as ki;
        let (front, back) = sp_mirror_stripes(nq, self.world).context("indexer_prefill_sp called below threshold")?;
        super::xchg::sp_idx_layer();
        let cfg = self.cfg;
        let (d, ql) = (cfg.hidden, cfg.q_lora_rank);
        let cand_out = cfg.candidate_source == Some(l);
        let cand_in = cfg.uses_candidates(l);
        let (fa0, fa1) = front[self.rank];
        let (ba0, ba1) = back[self.rank];
        let (s0, s1) = if fa0 <= ba0 { ((fa0, fa1), (ba0, ba1)) } else { ((ba0, ba1), (fa0, fa1)) };
        let spans: Vec<(usize, usize)> = if s0.1 == s1.0 { vec![(s0.0, s1.1)] } else { vec![s0, s1] };
        let iq_cols = ki::HEADS * ki::HEAD_DIM;
        let iq = self.f.b.take(nq * iq_cols * 2)?;
        for &(r0, r1) in &spans {
            let lw = &self.layers[l].as_ref().unwrap().w;
            self.f.lin_rows("tp.prefill.idx.wq_b (SP)", lw, C::IdxWqB, None, qr + (r0 * ql * 2) as u64, ql, ql, r1 - r0, iq + (r0 * iq_cols * 2) as u64, iq_cols, false)?;
        }
        let wp = match wp_pre {
            Some(p) => p,
            None => {
                ensure!(x != 0, "layer {l}: indexer_prefill_sp needs x when wp_pre is None");
                let wp = self.f.b.take(nq * ki::HEADS * 2)?;
                for &(r0, r1) in &spans {
                    let lw = &self.layers[l].as_ref().unwrap().w;
                    self.f.lin_rows("tp.prefill.idx.weights_proj (SP)", lw, C::IdxWeightsProj, None, x + (r0 * d * 2) as u64, d, d, r1 - r0, wp + (r0 * ki::HEADS * 2) as u64, ki::HEADS, false)?;
                }
                wp
            }
        };
        let lane = self.segs[0].lane;
        let st = self.lanes[lane].src.get(&s).context("index source state")?;
        let ratio = st.ratio;
        let keys_at_end = (q0 + nq) / ratio.max(1) as usize;
        ensure!(keys_at_end <= st.n_groups, "layer {l}: index keys up to {keys_at_end} needed, the source holds {}", st.n_groups);
        let (kq, ks) = (st.iq.ptr(), st.is.ptr());
        let r = 8usize;
        let mut rbs: Vec<(i32, i32)> = Vec::new();
        for &(r0, r1) in &spans {
            for i in (r0..r1).step_by(r) {
                rbs.push((i as i32, (r1 - i).min(r) as i32));
            }
        }
        let n_rb = rbs.len();
        let cap = keys_at_end.max(1);
        let ucap = r * ki::CAND_BLOCKS;
        let nb_cap = cap.div_ceil(ki::CAND_BLOCK);
        // Pad `nbw_cap` to a 4-u32 (16-byte) multiple so `bm` rows can be all-gathered directly over `tp_xport`.
        let nbw_cap = nb_cap.div_ceil(32).div_ceil(4) * 4;
        let rbv: Vec<u64> = rbs.iter().flat_map(|&(r0, n)| [kq, ks, (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
        let rbd = self.f.b.take(rbv.len() * 8)?;
        self.f.g.h2d_async(rbd, &rbv)?;
        let keep = self.sel_keep(l, nq)?;
        let cnt = self.f.b.take(nq * 4)?;
        let part = self.f.b.take((n_rb * r * ki::TOPK).max(1) * 8)?;
        let pcnt = self.f.b.take((n_rb * r).max(1) * 4)?;
        let bmax = if cand_out { self.f.b.take((nq * nb_cap).max(1) * 2)? } else { 0 };
        let cand = self.f.b.take((nq * ki::CAND_BLOCKS).max(1) * 4)?;
        let ccnt = self.f.b.take(nq * 4)?;
        let err = self.f.b.take(4)?;
        self.f.g.memset(err, 4, 0)?;
        let mut a = ki::IdxArgs::default();
        a.iq = iq; a.w = wp; a.pos = pos; a.rbs = rbd; a.sel = keep; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt; a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt;
        a.err = err; a.tick = 0;
        a.n_rb = n_rb as i32; a.rows_total = nq as i32; a.iq_ld = iq_cols as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
        a.ratio = ratio as i32; a.write_bmax = cand_out as i32;
        a.n_keys_cap = cap as i32; a.nb_cap = nb_cap as i32; a.nbw_cap = nbw_cap as i32;
        a.g = 1; a.min_tiles = 8; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = self.f.rope.ki;
        let path;
        let mut bm_gather: Option<(crate::dsv41::fwd::dev::Buf, crate::dsv41::fwd::dev::Buf)> = None;
        if cand_in {
            let c = cands.as_ref().with_context(|| format!("layer {l}: no candidate rows of layer {:?} in this chunk", cfg.candidate_source))?;
            ensure!(q0 >= c.row0 && q0 + nq <= c.row0 + c.n, "layer {l}: rows {q0}..{} outside the candidate rows {}..{}", q0 + nq, c.row0, c.row0 + c.n);
            let off = q0 - c.row0;
            let uni = self.f.b.take((n_rb * ucap).max(1) * 4)?;
            let ucnt = self.f.b.take(n_rb.max(1) * 4)?;
            let mut u = a;
            u.pos = c.pos.ptr() + (off * 4) as u64;
            u.bm = c.bm.ptr() + (off * c.nbw_cap * 4) as u64;
            u.n_keys_cap = c.n_keys_cap as i32; u.nb_cap = c.nb_cap as i32; u.nbw_cap = c.nbw_cap as i32; u.ratio = c.ratio as i32;
            u.uni = uni; u.ucnt = ucnt;
            ki::union(&self.f.g.l(), &u)?;
            a.uni = uni; a.ucnt = ucnt; a.mode = 1;
            a.n_keys_cap = c.n_keys_cap as i32; a.nb_cap = c.nb_cap as i32; a.nbw_cap = c.nbw_cap as i32;
            ki::score(&self.f.g.l(), &a)?;
            path = "tp.prefill.idx.sp (mirror-zigzag union + mode 1)";
        } else if cand_out {
            let bm = self.f.g.buf((nq * nbw_cap).max(1) * 4)?;
            let pb = self.f.g.buf(nq.max(1) * 4)?;
            self.f.g.d2d(pb.ptr(), pos, nq * 4)?;
            a.bm = bm.ptr();
            a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
            ki::score(&self.f.g.l(), &a)?;
            ki::cand(&self.f.g.l(), &a)?;
            bm_gather = Some((bm, pb));
            path = "tp.prefill.idx.sp (mirror-zigzag scores + candidate blocks)";
        } else {
            a.bm = self.f.b.take(4)?; a.uni = self.f.b.take(4)?; a.ucnt = self.f.b.take(n_rb.max(1) * 4)?;
            ki::score(&self.f.g.l(), &a)?;
            path = "tp.prefill.idx.sp (mirror-zigzag prefill shape)";
        }
        let mut gbufs = vec![(keep, ki::TOPK * 4)];
        if let Some((ref bm, _)) = bm_gather {
            gbufs.push((bm.ptr(), nbw_cap * 4));
        }
        self.comm.sp_all_gather_multi(self.cs, &gbufs, &front)?;
        self.comm.sp_all_gather_multi(self.cs, &gbufs, &back)?;
        if let Some((bm, pb)) = bm_gather {
            *cands = Some(CandRows { row0: q0, n: nq, bm, pos: pb, nbw_cap, nb_cap, n_keys_cap: cap, ratio });
        }
        crate::dsv41::fwd::receipts::kernel(path, "d41_idx_score_r8", n_rb * r);
        if self.f.checks {
            let e = self.f.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "indexer layer {l}: kernel error word {e}");
        }
        Ok(keep)
    }
}

/// Layer 20's per-row candidate bitmaps of one chunk's query rows `[row0, row0 + n)` (the candidate users build their
/// own row-block unions from them).
struct CandRows { row0: usize, n: usize, bm: crate::dsv41::fwd::dev::Buf, pos: crate::dsv41::fwd::dev::Buf, nbw_cap: usize, nb_cap: usize, n_keys_cap: usize, ratio: u32 }

/// A lane's prefix state (CE's `LaneSnap` for the TP lane): every layer's ring, every KV source's caches up to its groups
/// and its frontier ring, the token history (the Engram hash state is rebuilt from it). `kind` says which resumes are
/// exact. Leg MEM1: every piece lives in ONE device allocation (256-B aligned offsets), so a kept state is one block of
/// the memory pool instead of ~100 small ones scattered between transient allocations, and `bytes` is what it holds.
pub struct TpSnap {
    pub pos: usize,
    pub kind: crate::dsv41::ced::schedule::StateKind,
    pub hist: Vec<u32>,
    pub vl_types: Vec<i8>,
    mem: crate::dsv41::fwd::dev::Buf,
    /// per layer: (ring q offset, ring s offset)
    rings: Vec<(usize, usize)>,
    /// per KV source: (layer, groups, kv offset, index-K q offset, index-K s offset, frontier (a, b) offsets)
    srcs: Vec<(usize, usize, usize, usize, usize, Option<(usize, usize)>)>,
    pub bytes: usize,
    /// leg CE3: a prompt-end state's DSpark taps (the prompt's last `min(pos, TAP_ROWS)` rows, as the taps buffer holds them),
    /// set by the serve program when it takes the state right after the prompt's last prefill call
    pub tail: Option<crate::dsv41::fwd::dev::Buf>,
}

impl TpSnap {
    /// Rows a resumed prefill must recompute at least from this state (the prefix pool's cap: 134 full, 2,541 encoder).
    pub fn min_suffix(&self, cfg: &crate::dsv41::config::Dsv41Config, prompt_len: usize) -> usize {
        Schedule::new(cfg, prompt_len.max(1), Mode::Ced).map(|s| s.min_suffix(self.kind)).unwrap_or(crate::dsv41::ced::schedule::MIN_KEEP)
    }
}

impl TpModel<'_> {
    /// Snapshot lane `lane` (device copies on the stream into one allocation).
    pub fn snapshot(&mut self, lane: usize) -> Result<TpSnap> {
        let g = self.f.g;
        let ln = &self.lanes[lane];
        let a256 = |n: usize| (n + 255) / 256 * 256;
        let mut off = 0usize;
        let mut at = |n: usize| { let o = off; off += a256(n); o };
        let rings: Vec<(usize, usize)> = ln.win.iter().map(|w| (at(w.ring_q.bytes), at(w.ring_s.bytes))).collect();
        let mut keys: Vec<usize> = ln.src.keys().copied().collect();
        keys.sort_unstable();
        let mut srcs = Vec::with_capacity(keys.len());
        for s in keys {
            let st = &ln.src[&s];
            let n = st.n_groups;
            let (kv, iq, is_) = (at(n * 288), at(n * 64), at(n * 4));
            let front = st.front.as_ref().map(|(a, b)| (at(a.bytes), at(b.bytes)));
            srcs.push((s, n, kv, iq, is_, front));
        }
        let mem = g.buf(off)?;
        let base = mem.ptr();
        for (w, &(q, sc)) in ln.win.iter().zip(&rings) {
            g.d2d(base + q as u64, w.ring_q.ptr(), w.ring_q.bytes)?;
            g.d2d(base + sc as u64, w.ring_s.ptr(), w.ring_s.bytes)?;
        }
        for &(s, n, kv, iq, is_, front) in &srcs {
            let st = &ln.src[&s];
            g.d2d(base + kv as u64, st.kv_main.ptr(), n * 288)?;
            g.d2d(base + iq as u64, st.iq.ptr(), n * 64)?;
            g.d2d(base + is_ as u64, st.is.ptr(), n * 4)?;
            if let (Some((a, b)), Some((x, y))) = (&st.front, front) {
                g.d2d(base + x as u64, a.ptr(), a.bytes)?;
                g.d2d(base + y as u64, b.ptr(), b.bytes)?;
            }
        }
        g.sync()?;
        let bytes = mem.bytes;
        let vl_types = if ln.vl_types.is_empty() {
            Vec::new()
        } else {
            ln.vl_types[..ln.vl_types.len().min(ln.pos)].to_vec()
        };
        Ok(TpSnap { pos: ln.pos, kind: ln.kind, hist: ln.hist.clone(), vl_types, mem, rings, srcs, bytes, tail: None })
    }

    /// Restore a snapshot into lane `lane` (its state becomes the snapshot's; the Engram hash state from the tokens).
    pub fn restore(&mut self, lane: usize, s: &TpSnap) -> Result<()> {
        let g = self.f.g;
        let ln = &mut self.lanes[lane];
        let base = s.mem.ptr();
        for (w, &(q, sc)) in ln.win.iter().zip(&s.rings) {
            g.d2d(w.ring_q.ptr(), base + q as u64, w.ring_q.bytes)?;
            g.d2d(w.ring_s.ptr(), base + sc as u64, w.ring_s.bytes)?;
        }
        for &(l, n, kv, iq, is_, front) in &s.srcs {
            let st = ln.src.get_mut(&l).context("snapshot source not in the lane")?;
            g.d2d(st.kv_main.ptr(), base + kv as u64, n * 288)?;
            g.d2d(st.iq.ptr(), base + iq as u64, n * 64)?;
            g.d2d(st.is.ptr(), base + is_ as u64, n * 4)?;
            if let (Some((a, b)), Some((x, y))) = (&st.front, front) { g.d2d(a.ptr(), base + x as u64, a.bytes)?; g.d2d(b.ptr(), base + y as u64, b.bytes)?; }
            st.n_groups = n;
        }
        ln.pos = s.pos;
        ln.kind = s.kind;
        ln.hist = s.hist.clone();
        ln.vl_types = s.vl_types.clone();
        ln.vl_row_ops.clear();
        ln.vl_aligner = None;
        ln.hash = match self.hasher.as_ref() {
            Some(hs) => {
                let t0 = s.hist.len().saturating_sub(self.cfg.engram_max_ngram - 1);
                let tail = &s.hist[t0..];
                if s.vl_types.is_empty() {
                    hs.state_at(s.pos as u64, tail, None)?
                } else {
                    let live_tail: Vec<bool> = (t0..s.hist.len()).map(|p| s.vl_types.get(p).copied().unwrap_or(-1) < 0).collect();
                    hs.state_at(s.pos as u64, tail, Some(&live_tail))?
                }
            }
            None => crate::dsv41::engram::hash::NgramHashState::new(),
        };
        g.sync()
    }
}

#[cfg(test)]
mod sp_tests {
    use super::*;

    #[test]
    fn sp_row_blocks_align_to_128_and_cover_every_row() {
        assert!(sp_row_blocks(255, 2).is_none());
        assert!(sp_row_blocks(383, 3).is_none());
        assert!(sp_row_blocks(511, 4).is_none());
        assert!(sp_row_blocks(767, 6).is_none());
        assert!(sp_row_blocks(1023, 8).is_none());
        assert!(sp_row_blocks(2048, 1).is_none());
        for (world, min_n) in [(2usize, 256usize), (3, 384), (4, 512), (6, 768), (8, 1024)] {
            for n in [min_n, min_n + 1, min_n + 73, 1024, 2048, 2049, 2560, 4096, 8192, 8195] {
                let b = sp_row_blocks(n, world).unwrap();
                assert_eq!(b.len(), world);
                assert_eq!(b[0].0, 0);
                assert_eq!(b[world - 1].1, n);
                for r in 0..world {
                    assert!(b[r].1 - b[r].0 >= 64, "world {world} n {n} rank {r}: {:?}", b[r]);
                    if r + 1 < world {
                        assert_eq!(b[r].1, b[r + 1].0);
                        assert_eq!(b[r].1 % 128, 0, "internal boundary must align to 128-row M-tile");
                    }
                }
                let b4 = sp_blocks_div4(&b);
                assert_eq!(b4[0].0, 0);
                assert_eq!(b4[world - 1].1, n.div_ceil(4));
                for r in 0..world - 1 { assert_eq!(b4[r].1, b4[r + 1].0); }
            }
        }
    }

    #[test]
    fn sp_mirror_stripes_partition_exact_8row_blocks_and_balance_causal_work() {
        assert!(sp_mirror_stripes(255, 2).is_none());
        assert!(sp_mirror_stripes(383, 3).is_none());
        assert!(sp_mirror_stripes(511, 4).is_none());
        assert!(sp_mirror_stripes(767, 6).is_none());
        assert!(sp_mirror_stripes(1023, 8).is_none());
        for (world, min_nq) in [(2usize, 256usize), (3, 384), (4, 512), (6, 768), (8, 1024)] {
            for nq in [min_nq, 382, 512, 890, 1398, 1906, 2048, 2049, 2560, 8192] {
                if nq < 2 * world * 64 { continue; }
                let (front, back) = sp_mirror_stripes(nq, world).unwrap();
                assert_eq!(front.len(), world);
                assert_eq!(back.len(), world);
                // Every row 0..nq covered once across front and back
                let mut seen = vec![0u32; nq];
                let mut rank_keys = vec![0u64; world];
                let mut rank_rbs = Vec::new();
                for r in 0..world {
                    assert!(front[r].1 - front[r].0 >= 56 && back[r].1 - back[r].0 >= 56);
                    for &(a, b) in &[front[r], back[r]] {
                        assert_eq!(a % 8, 0);
                        if b < nq { assert_eq!(b % 8, 0); }
                        for row in a..b {
                            seen[row] += 1;
                            rank_keys[r] += (10_000 + row as u64 + 1) / 2;
                        }
                        for i in (a..b).step_by(8) {
                            rank_rbs.push((i as i32, (b - i).min(8) as i32));
                        }
                    }
                }
                assert!(seen.iter().all(|&c| c == 1), "world {world} nq {nq}: rows not covered once");
                // Sorted 8-row blocks equal non-SP's exact 8-row blocks
                rank_rbs.sort_by_key(|x| x.0);
                let ref_rbs: Vec<(i32, i32)> = (0..nq).step_by(8).map(|i| (i as i32, (nq - i).min(8) as i32)).collect();
                assert_eq!(rank_rbs, ref_rbs, "world {world} nq {nq}: 8-row blocks must match non-SP");
                // Causal key-scoring load skew < 1% on even chunks (0.000% when divisible by 2 * world * 128)
                let min_k = *rank_keys.iter().min().unwrap() as f64;
                let max_k = *rank_keys.iter().max().unwrap() as f64;
                if nq % (2 * world * 128) == 0 {
                    assert_eq!(min_k as u64, max_k as u64, "exact 0 skew on aligned chunk {nq} world {world}");
                } else {
                    // Discrete 8-row block quantum: e.g. world=3 nq=512 -> 64 blocks = 22+21+21 (4.58% irreducible).
                    let lim = if !world.is_power_of_two() || world >= 6 { 0.06 } else { 0.03 };
                    assert!((max_k - min_k) / max_k < lim, "world {world} nq {nq}: skew {:.4}", (max_k - min_k) / max_k);
                }
            }
        }
        let id = crate::opts::lookup("dsv41-sp-prefill").expect("--dsv41-sp-prefill registered");
        assert_eq!(id.def().scope, crate::opts::Scope::Spmd);
        assert_eq!(id.def().env, "");
        let lid = crate::opts::lookup("dsv41-sp-laga").expect("--dsv41-sp-laga registered");
        assert_eq!(lid.def().scope, crate::opts::Scope::Spmd);
        assert_eq!(lid.def().env, "");
        for w in [2usize, 3, 4, 6, 8] {
            assert!(sp_laga_mode_on(None, w, true), "LAGA auto ON for world={w}");
            assert!(sp_laga_mode_on(Some("on"), w, true));
            assert!(!sp_laga_mode_on(Some("off"), w, true));
            assert!(!sp_laga_mode_on(None, w, false));
        }
        assert!(!sp_laga_mode_on(None, 1, true));
    }
}
