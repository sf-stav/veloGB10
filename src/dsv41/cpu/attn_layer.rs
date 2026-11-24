//! DSV4.1 attention oracle (leg OA): one attention sublayer of the backbone, as `model.py` computes it
//! (Attention 766-791, Compressor 429-485, Indexer 488-581, select_candidate_blocks 584-611,
//! sparse_attn = kernel.py 311-403 in REFb's emulation), with the cross-layer state the reference
//! keeps in `shared_attn` and the per-layer caches.
//!
//! The oracle is driven row-wise: `forward_layer` takes the attention input (`attn_in` =
//! attn_norm(hc_pre(x, pre_mix))) of rows [s0, s1) and computes
//! * the window K/V of every row (kept for all positions: the decode ring is a view of it),
//! * at a KV source, the compressor over the rows (an incomplete group waits in `kv_state` /
//!   `score_state` exactly as the reference's buffers hold it), the index keys (owners) and the
//!   compressed KV,
//! * for the requested query rows only: q, the indexer (index sources) or the published top-k
//!   (reusers), the sparse attention, the inverse RoPE, `act_quant`, `wo_a`, `wo_b`.
//!
//! `Order::Prefill { p_total }` is the reference's single-shot prefill (start_pos 0, window index list
//! `[max(0,t-127) ..]` of width min(p_total, window)); REFb proved a chunked prefill with the same
//! p_total bit-identical to it, which is what calling this with s0 > 0 computes. `Order::Decode` is
//! one token at start_pos > 0 with the reference's ring order (`get_window_topk_idxs` decode branch:
//! slots oldest first, slots past start_pos = -1), which places the masked slots FIRST while the ring
//! fills, so its online-softmax blocks group differently from a prefill row (fp32-rounding-level
//! differences only).

use super::num::{act_quant_inplace, bf16, dot_f64, fp4_quant_e4m3_inplace, fp4_quant_e8m0_inplace, par_map, rmsnorm, variant, Fp8Lin};
use super::rope::{Rope, Yarn};
use super::weights::AttnWeights;
use crate::dsv41::config::Dsv41Config;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// The online-softmax block of `sparse_attn` (kernel.py: 64 gather indices per step).
pub const ATTN_BLOCK: usize = 64;

/// The attention shape and wiring the oracle needs (from `Dsv41Config`, or hand-built in tests).
#[derive(Clone, Debug)]
pub struct OracleCfg {
    pub layers: usize,
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub window: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub cand_source: Option<usize>,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub eps: f32,
    pub ratios: Vec<u32>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    /// ratio-0 layers: theta 10,000, no YaRN
    pub rope_plain: Rope,
    /// ratio > 0 layers: compress theta 160,000 with YaRN
    pub rope_yarn: Rope,
}

impl OracleCfg {
    pub fn from_config(c: &Dsv41Config) -> OracleCfg {
        let yarn = Yarn {
            original_seq_len: c.rope.original_max_position_embeddings as f64,
            factor: c.rope.factor,
            beta_fast: c.rope.beta_fast,
            beta_slow: c.rope.beta_slow,
        };
        OracleCfg {
            layers: c.layers,
            dim: c.hidden,
            heads: c.heads,
            head_dim: c.head_dim,
            rope_dim: c.rope_dim,
            q_lora: c.q_lora_rank,
            o_groups: c.o_groups,
            o_lora: c.o_lora_rank,
            window: c.window,
            index_heads: c.index_heads,
            index_head_dim: c.index_head_dim,
            index_topk: c.index_topk,
            cand_source: c.candidate_source,
            cand_topk_blocks: c.candidate_topk_blocks,
            cand_block: c.candidate_block,
            eps: c.rms_eps as f32,
            ratios: c.compress_ratios.clone(),
            kv_sources: c.kv_sources.clone(),
            index_sources: c.index_sources.clone(),
            rope_plain: Rope::new(c.rope_dim, c.rope.theta, None),
            rope_yarn: Rope::new(c.rope_dim, c.rope.compress_theta, Some(yarn).filter(|_| !variant::is(variant::NO_YARN))),
        }
    }
    pub fn ratio(&self, l: usize) -> usize { self.ratios.get(l).copied().unwrap_or(0) as usize }
    pub fn rope(&self, l: usize) -> &Rope { if self.ratio(l) > 0 { &self.rope_yarn } else { &self.rope_plain } }
    pub fn is_kv_source(&self, l: usize) -> bool { self.ratio(l) > 0 && self.kv_sources.contains(&l) }
    pub fn is_index_source(&self, l: usize) -> bool { self.ratio(l) > 0 && self.index_sources.contains(&l) }
    /// The KV source whose compressed cache (and index keys) layer `l` reads.
    pub fn kv_source_of(&self, l: usize) -> Option<usize> {
        if self.ratio(l) == 0 { return None; }
        self.kv_sources.iter().copied().filter(|&s| s <= l).max()
    }
    /// The index source whose top-k layer `l` attends with (itself at an index source).
    pub fn index_source_of(&self, l: usize) -> Option<usize> {
        if self.ratio(l) == 0 { return None; }
        self.index_sources.iter().copied().filter(|&s| s <= l).max()
    }
    pub fn is_cand_source(&self, l: usize) -> bool { self.cand_source == Some(l) }
    pub fn uses_candidates(&self, l: usize) -> bool { self.is_index_source(l) && self.cand_source.map_or(false, |c| c < l) }
    /// Human name of the layer's attention class (the report groups by it).
    pub fn class_name(&self, l: usize) -> String {
        let r = self.ratio(l);
        if r == 0 { return "SWA-only".into(); }
        let mut s = format!("ratio-{r}");
        s += if self.is_kv_source(l) { " KV source" } else { " KV reuser" };
        if self.is_cand_source(l) { s += ", candidate source"; }
        if self.is_index_source(l) { s += if self.uses_candidates(l) { ", index source (candidates)" } else { ", index source" }; }
        else { s += ", top-k reuser"; }
        s
    }
}

/// How the query rows' window index lists are ordered (it decides the online-softmax blocks).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    /// start_pos-0 prefill (or a chunk of one, REFb semantics) of a prompt of `p_total` rows
    Prefill { p_total: usize },
    /// one token at start_pos > 0 (reference decode: ring order, ring of `window` slots)
    Decode,
}

/// The state a KV-source layer leaves (the reference's `compress_kv_cache`, `k_cache`, `kv_state`,
/// `score_state`).
#[derive(Clone, Debug, Default)]
pub struct SrcState {
    pub ratio: usize,
    /// groups x head_dim: the RoPE'd, FP4(E4M3/16)-rounded compressed latents (bf16 values)
    pub compress_kv: Vec<f32>,
    /// groups x index_head_dim: index keys (owners), RoPE'd and FP4(E8M0/32)-rounded
    pub index_k: Vec<f32>,
    /// ratio x head_dim: the partial group (fp32), slot = position % ratio (zeros initially)
    pub kv_state: Vec<f32>,
    /// ratio x head_dim (-inf initially)
    pub score_state: Vec<f32>,
    /// positions fed so far
    pub seen: usize,
}

impl SrcState {
    /// Same decode-relevant state: equal compressed KV and index keys, and equal partial-group rows
    /// for the positions of the group still open (the other slots may hold stale rows of a completed
    /// group, exactly as the reference's buffers do, and are never read).
    pub fn same_live_state(&self, o: &SrcState, head_dim: usize) -> bool {
        if self.compress_kv != o.compress_kv || self.index_k != o.index_k || self.seen != o.seen || self.ratio != o.ratio {
            return false;
        }
        let r = self.ratio.max(1);
        (self.seen / r * r..self.seen).all(|pos| {
            let sl = (pos % r) * head_dim..(pos % r + 1) * head_dim;
            self.kv_state[sl.clone()] == o.kv_state[sl.clone()] && self.score_state[sl.clone()] == o.score_state[sl]
        })
    }
}

/// One index source's verdict for one query row.
#[derive(Clone, Debug, Default)]
pub struct TopK {
    /// selected compressed positions in ascending order, then -1 padding (width min(topk, end_c))
    pub pos: Vec<i32>,
    /// the full score row over the reachable positions (bf16 values; candidate-masked = -inf);
    /// kept for tie analysis
    pub scores: Vec<f32>,
}

/// What `forward_layer` returns for the query rows (in the order asked).
#[derive(Clone, Debug, Default)]
pub struct LayerOut {
    /// rows x dim (bf16 values): the attention sublayer output after `wo_b`
    pub attn_out: Vec<f32>,
    /// rows x heads x head_dim (bf16 values): `sparse_attn`'s output after the inverse RoPE, before
    /// WO-A's `act_quant` (the comparison point for a GPU attention kernel)
    pub attn_o: Vec<f32>,
    /// rows x (o_groups * o_lora) (bf16 values): `wo_a` outputs = `wo_b`'s input
    pub wo_b_in: Vec<f32>,
    /// index sources only: the top-k per query row
    pub topk: Vec<TopK>,
}

/// The attention oracle: per-layer window history, per-source compressed state, published top-k
/// and candidate blocks (keyed by query position, as `shared_attn` is per forward call).
#[derive(Clone, Debug)]
pub struct Oracle {
    pub cfg: OracleCfg,
    /// layer -> positions x head_dim window K/V (bf16 values after RoPE + act_quant)
    pub window_kv: BTreeMap<usize, Vec<f32>>,
    pub src: BTreeMap<usize, SrcState>,
    /// index source -> query position -> top-k
    pub topk: BTreeMap<usize, BTreeMap<usize, TopK>>,
    /// query position -> kept candidate blocks (from the candidate source)
    pub cand: BTreeMap<usize, Vec<bool>>,
    /// query position -> the candidate source's block scores (max over each block, newest block
    /// +inf), kept for near-tie analysis of the candidate cut
    pub cand_scores: BTreeMap<usize, Vec<f32>>,
}

/// The input of one `forward_layer` call.
pub struct LayerIn<'a> {
    pub layer: usize,
    /// first absolute position of `x`
    pub s0: usize,
    /// (s1 - s0) x dim: attn_in rows (bf16 values)
    pub x: &'a [f32],
    /// absolute positions in [s0, s1) whose attention output is wanted
    pub queries: &'a [usize],
    pub order: Order,
    /// index sources only: attend with (and publish) these compressed-position lists instead of the
    /// oracle's own selection (query position -> list). The oracle's own top-k is still computed and
    /// returned. Lets a per-layer check use the reference's selection while the selection itself is
    /// compared separately, up to ties.
    pub topk_override: Option<&'a BTreeMap<usize, Vec<i32>>>,
}

impl Oracle {
    pub fn new(cfg: OracleCfg) -> Oracle {
        Oracle { cfg, window_kv: BTreeMap::new(), src: BTreeMap::new(), topk: BTreeMap::new(), cand: BTreeMap::new(), cand_scores: BTreeMap::new() }
    }

    /// The decode ring of a layer as the reference holds it after `seen` positions: slot =
    /// position % window, never-written slots zero ([window][head_dim]).
    pub fn window_ring(&self, l: usize) -> Vec<f32> {
        let (w, hd) = (self.cfg.window, self.cfg.head_dim);
        let mut ring = vec![0f32; w * hd];
        if let Some(kv) = self.window_kv.get(&l) {
            let n = kv.len() / hd;
            for p in n.saturating_sub(w)..n {
                ring[(p % w) * hd..(p % w + 1) * hd].copy_from_slice(&kv[p * hd..(p + 1) * hd]);
            }
        }
        ring
    }

    /// Run one attention sublayer over rows [s0, s1) (see the module doc).
    pub fn forward_layer(&mut self, w: &AttnWeights, inp: LayerIn) -> Result<LayerOut> {
        let c = self.cfg.clone();
        let l = inp.layer;
        ensure!(w.layer == l, "weights of layer {} given for layer {l}", w.layer);
        let (d, hd, rd) = (c.dim, c.head_dim, c.rope_dim);
        ensure!(inp.x.len() % d == 0, "attn_in is not a whole number of rows");
        let rows = inp.x.len() / d;
        let (s0, s1) = (inp.s0, inp.s0 + rows);
        if inp.order == Order::Decode {
            ensure!(rows == 1 && s0 > 0, "Order::Decode is one token at start_pos > 0");
        }
        for &q in inp.queries {
            ensure!(q >= s0 && q < s1, "query row {q} is outside the rows [{s0}, {s1}) given");
        }
        let rope = c.rope(l).clone();

        // ---- window K/V for every row (model.py 706-708)
        let mut kv = w.wkv.forward(inp.x, rows);
        let mut tmp = vec![0f32; hd];
        for r in 0..rows {
            let row = &mut kv[r * hd..(r + 1) * hd];
            tmp.copy_from_slice(row);
            rmsnorm(&tmp, &w.kv_norm, c.eps, row);
            rope.apply(&mut row[hd - rd..], s0 + r, false);
            act_quant_inplace(row, 32);
        }
        let hist = self.window_kv.entry(l).or_default();
        ensure!(hist.len() / hd == s0, "layer {l}: window history holds {} positions, the call starts at {s0}", hist.len() / hd);
        hist.extend_from_slice(&kv);

        // ---- compressor, index keys, compressed KV (KV sources; model.py 740-762, 458-485, 537-549)
        if c.is_kv_source(l) {
            self.compress(w, l, inp.x, s0, &rope, inp.order)?;
        }

        // ---- queries
        let nq = inp.queries.len();
        if nq == 0 {
            return Ok(LayerOut::default());
        }
        let mut xq = Vec::with_capacity(nq * d);
        for &q in inp.queries {
            xq.extend_from_slice(&inp.x[(q - s0) * d..(q - s0 + 1) * d]);
        }
        let mut qr = w.wq_a.forward(&xq, nq);
        let mut tmpq = vec![0f32; c.q_lora];
        for r in 0..nq {
            let row = &mut qr[r * c.q_lora..(r + 1) * c.q_lora];
            tmpq.copy_from_slice(row);
            rmsnorm(&tmpq, &w.q_norm, c.eps, row);
        }
        let mut q = w.wq_b.forward(&qr, nq);
        for (r, &pos) in inp.queries.iter().enumerate() {
            let cis = rope.cis(pos);
            for h in 0..c.heads {
                let o = r * c.heads * hd + h * hd;
                rope.apply_cis(&mut q[o + hd - rd..o + hd], &cis, false);
            }
        }

        // ---- compressed indices: own indexer or the published top-k
        let ratio = c.ratio(l);
        let mut topk_out = Vec::new();
        let comp_lists: Vec<Vec<i32>> = if ratio == 0 {
            vec![Vec::new(); nq]
        } else if c.is_index_source(l) {
            topk_out = self.indexer(w, l, &xq, &qr, inp.queries, s1)?;
            let mut lists = Vec::with_capacity(nq);
            let pubm = self.topk.entry(l).or_default();
            for (&pos, t) in inp.queries.iter().zip(&topk_out) {
                let used = match inp.topk_override {
                    Some(ov) => TopK { pos: ov.get(&pos).with_context(|| format!("layer {l}: top-k override has no row {pos}"))?.clone(), scores: Vec::new() },
                    None => t.clone(),
                };
                lists.push(used.pos.clone());
                pubm.insert(pos, used);
            }
            lists
        } else {
            let isrc = c.index_source_of(l).context("a compressing layer without an index source")?;
            let m = self.topk.get(&isrc).with_context(|| format!("layer {l}: index source {isrc} has published no top-k"))?;
            inp.queries
                .iter()
                .map(|p| m.get(p).map(|t| t.pos.clone()).with_context(|| format!("layer {l}: no top-k from {isrc} for row {p}")))
                .collect::<Result<_>>()?
        };

        // ---- sparse attention over [window | compressed] (kernel.py 311-403)
        let win = &self.window_kv[&l];
        let comp_kv: &[f32] = match c.kv_source_of(l) {
            Some(s) => &self.src.get(&s).with_context(|| format!("layer {l}: KV source {s} has no state"))?.compress_kv,
            None => &[],
        };
        let lists: Vec<Vec<Option<&[f32]>>> = inp
            .queries
            .iter()
            .zip(&comp_lists)
            .map(|(&t, cl)| {
                let mut v: Vec<Option<&[f32]>> = window_positions(c.window, t, inp.order)
                    .into_iter()
                    .map(|p| p.map(|p| &win[p * hd..(p + 1) * hd]))
                    .collect();
                v.extend(cl.iter().map(|&g| if g < 0 { None } else { Some(&comp_kv[g as usize * hd..(g as usize + 1) * hd]) }));
                v
            })
            .collect();
        let scale = (1.0f64 / (hd as f64).sqrt()) as f32;
        let heads = c.heads;
        let o_rows: Vec<f32> = par_map(nq * heads, 1, |rg| {
            let mut out = Vec::with_capacity(rg.len() * hd);
            for i in rg {
                let (r, h) = (i / heads, i % heads);
                let qh = &q[r * heads * hd + h * hd..r * heads * hd + (h + 1) * hd];
                out.extend(sparse_attn_head(qh, &lists[r], w.sink[h], scale));
            }
            out
        });
        drop(lists);

        // ---- inverse RoPE, act_quant on WO-A's input, wo_a per group, wo_b (model.py 781-790)
        let mut o = o_rows;
        let gin = heads * hd / c.o_groups;
        for (r, &pos) in inp.queries.iter().enumerate() {
            let cis = rope.cis(pos);
            for h in 0..heads {
                let base = r * heads * hd + h * hd;
                if !variant::is(variant::NO_INV_ROPE) {
                    rope.apply_cis(&mut o[base + hd - rd..base + hd], &cis, true);
                }
            }
        }
        // the attention output itself (bf16, after the inverse RoPE): the point a GPU attention kernel
        // compares at (leg KA: per-layer attn_out amplifies block-order noise 5-10x through wo_a / wo_b)
        let attn_o = o.clone();
        for r in 0..nq {
            if !variant::is(variant::NO_WOA_QUANT) {
                act_quant_inplace(&mut o[r * heads * hd..(r + 1) * heads * hd], 32);
            }
        }
        let (g, ol) = (c.o_groups, c.o_lora);
        let woa = &w.wo_a;
        let wo_b_in: Vec<f32> = par_map(nq * g * ol, 64, |rg| {
            rg.map(|i| {
                let (r, gr) = (i / (g * ol), i % (g * ol));
                let gi = gr / ol;
                let orow = &o[r * heads * hd + gi * gin..r * heads * hd + (gi + 1) * gin];
                bf16(dot_f64(orow, &woa[gr * gin..(gr + 1) * gin]) as f32)
            })
            .collect()
        });
        let attn_out = w.wo_b.forward(&wo_b_in, nq);
        Ok(LayerOut { attn_out, attn_o, wo_b_in, topk: topk_out })
    }

    /// The compressor of KV source `l` over rows [s0, s0 + rows): pools complete groups (the open
    /// group carried in `kv_state` / `score_state`), then index keys (owners) and compressed KV.
    /// The two state buffers are written as `model.py` writes them: a prefill (start_pos 0, or a
    /// chunk of one, REFb's convention) stores only the rows of the group left open at its end
    /// (slots 0..remainder; an aligned prefill leaves the buffers untouched: zeros / -inf), a decode
    /// step stores its row's slot `pos % ratio`. Pooling reads the carried rows from them.
    fn compress(&mut self, w: &AttnWeights, l: usize, x: &[f32], s0: usize, rope: &Rope, order: Order) -> Result<()> {
        let c = self.cfg.clone();
        let cw = w.comp.as_ref().with_context(|| format!("layer {l} is a KV source but has no compressor weights"))?;
        let (d, hd, rd) = (c.dim, c.head_dim, c.rope_dim);
        let rows = x.len() / d;
        let ratio = cw.ratio;
        let st = self.src.entry(l).or_insert_with(|| SrcState {
            ratio,
            kv_state: vec![0f32; ratio * hd],
            score_state: vec![f32::NEG_INFINITY; ratio * hd],
            ..Default::default()
        });
        ensure!(st.seen == s0, "layer {l}: compressor has seen {} positions, the call starts at {s0}", st.seen);
        // latents (before RoPE) of the groups completed by these rows, with their group index
        let mut latents: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut tmp = vec![0f32; hd];
        if ratio == 1 {
            let kv = cw.wkv.forward_bf16(x, rows);
            for r in 0..rows {
                let mut lat = vec![0f32; hd];
                rmsnorm(&kv[r * hd..(r + 1) * hd], &cw.norm, c.eps, &mut lat);
                latents.push((s0 + r, lat));
            }
        } else {
            let kv = cw.wkv.forward_f32(x, rows);
            let sc = cw.wgate.as_ref().context("ratio > 1 compressor without wgate")?.forward_f32(x, rows);
            // the open group: the carried rows come from the buffers, this call's rows are added
            let (mut gkv, mut gsc) = (st.kv_state.clone(), st.score_state.clone());
            for r in 0..rows {
                let pos = s0 + r;
                let slot = pos % ratio;
                gkv[slot * hd..(slot + 1) * hd].copy_from_slice(&kv[r * hd..(r + 1) * hd]);
                gsc[slot * hd..(slot + 1) * hd].copy_from_slice(&sc[r * hd..(r + 1) * hd]);
                if (pos + 1) % ratio == 0 {
                    // softmax over the ratio rows of the group, per channel (exp(x - max) / sum, as
                    // torch's softmax), then (kv * p).sum over the rows: products rounded, then added
                    for ch in 0..hd {
                        let mut m = f32::NEG_INFINITY;
                        for j in 0..ratio {
                            m = m.max(gsc[j * hd + ch]);
                        }
                        let mut e = vec![0f32; ratio];
                        let mut sum = 0f32;
                        for j in 0..ratio {
                            e[j] = (gsc[j * hd + ch] - m).exp();
                            sum += e[j];
                        }
                        let mut acc = 0f32;
                        for j in 0..ratio {
                            acc += gkv[j * hd + ch] * (e[j] / sum);
                        }
                        tmp[ch] = bf16(acc);
                    }
                    let mut lat = vec![0f32; hd];
                    rmsnorm(&tmp, &cw.norm, c.eps, &mut lat);
                    latents.push((pos / ratio, lat));
                }
            }
            match order {
                Order::Decode => {
                    st.kv_state = gkv;
                    st.score_state = gsc;
                }
                Order::Prefill { .. } => {
                    for pos in (s0 + rows) / ratio * ratio..s0 + rows {
                        let sl = (pos % ratio) * hd..(pos % ratio + 1) * hd;
                        st.kv_state[sl.clone()].copy_from_slice(&gkv[sl.clone()]);
                        st.score_state[sl.clone()].copy_from_slice(&gsc[sl]);
                    }
                }
            }
        }
        st.seen = s0 + rows;
        let ih = c.index_head_dim;
        for (gi, lat) in latents {
            ensure!(st.compress_kv.len() / hd == gi, "layer {l}: group {gi} out of order");
            let gpos = gi * ratio;
            let cis = rope.cis(gpos);
            if let Some((wk, kn)) = w.idx.as_ref().and_then(|iw| Some((iw.wk.as_ref()?, iw.k_norm.as_ref()?))) {
                let k0 = wk.forward_bf16(&lat, 1);
                let mut k = vec![0f32; ih];
                rmsnorm(&k0, kn, c.eps, &mut k);
                rope.apply_cis(&mut k[ih - rd..], &cis, false);
                fp4_quant_e8m0_inplace(&mut k, 32);
                st.index_k.extend_from_slice(&k);
            }
            let mut ck = lat;
            rope.apply_cis(&mut ck[hd - rd..], &cis, false);
            fp4_quant_e4m3_inplace(&mut ck, 16);
            st.compress_kv.extend_from_slice(&ck);
        }
        Ok(())
    }

    /// The indexer of index source `l` for the query rows (model.py 551-581; REFb `fp32acc` scores).
    fn indexer(&mut self, w: &AttnWeights, l: usize, xq: &[f32], qr: &[f32], queries: &[usize], s1: usize) -> Result<Vec<TopK>> {
        let c = self.cfg.clone();
        let iw = w.idx.as_ref().with_context(|| format!("layer {l} is an index source but has no indexer weights"))?;
        let ratio = c.ratio(l);
        let owner = c.kv_source_of(l).context("index source without a KV source")?;
        let (ih, ihd, rd) = (c.index_heads, c.index_head_dim, c.rope_dim);
        let nq = queries.len();
        let rope = c.rope(l);
        let mut qi = iw.wq_b.forward(qr, nq);
        for (r, &pos) in queries.iter().enumerate() {
            let cis = rope.cis(pos);
            for h in 0..ih {
                let o = r * ih * ihd + h * ihd;
                rope.apply_cis(&mut qi[o + ihd - rd..o + ihd], &cis, false);
            }
            fp4_quant_e8m0_inplace(&mut qi[r * ih * ihd..(r + 1) * ih * ihd], 32);
        }
        // weights_proj(x) * (index_head_dim^-0.5 * index_heads^-0.5): a bf16 tensor times an fp32 scalar
        let wscale = ((ihd as f64).powf(-0.5) * (ih as f64).powf(-0.5)) as f32;
        let wts: Vec<f32> = iw.weights_proj.forward_bf16(xq, nq).into_iter().map(|v| bf16(v * wscale)).collect();
        let ik = &self.src.get(&owner).with_context(|| format!("layer {l}: index-key owner {owner} has no state"))?.index_k;
        let end_c = s1 / ratio;
        ensure!(ik.len() / ihd >= end_c, "layer {l}: {} index keys, {end_c} needed", ik.len() / ihd);
        let topk = c.index_topk.min(end_c);
        let cand_src = c.is_cand_source(l);
        let uses = c.uses_candidates(l);
        let mut out = Vec::with_capacity(nq);
        for (r, &pos) in queries.iter().enumerate() {
            let cl = (pos + 1) / ratio; // reachable compressed positions (model.py 563-564; decode: end_pos // ratio)
            let qrow = &qi[r * ih * ihd..(r + 1) * ih * ihd];
            let wrow = &wts[r * ih..(r + 1) * ih];
            let f32s = variant::is(variant::FP32_INDEX_SCORE);
            let mut scores: Vec<f32> = par_map(cl, 256, |rg| {
                rg.map(|t| {
                    let krow = &ik[t * ihd..(t + 1) * ihd];
                    let mut s = 0f64;
                    for h in 0..ih {
                        if f32s {
                            s += ((dot_f64(&qrow[h * ihd..(h + 1) * ihd], krow) as f32).max(0.0) * wrow[h]) as f64;
                            continue;
                        }
                        let v = bf16(dot_f64(&qrow[h * ihd..(h + 1) * ihd], krow) as f32).max(0.0);
                        s += bf16(v * wrow[h]) as f64;
                    }
                    if f32s { s as f32 } else { bf16(s as f32) }
                })
                .collect()
            });
            if cand_src {
                let keep = select_candidate_blocks(&scores, cl, c.cand_topk_blocks, c.cand_block);
                self.cand.insert(pos, keep);
                self.cand_scores.insert(pos, block_scores(&scores, cl, c.cand_block));
            } else if uses {
                let keep = self.cand.get(&pos).with_context(|| format!("layer {l}: no candidate blocks for row {pos}"))?;
                for (t, s) in scores.iter_mut().enumerate() {
                    if !keep.get(t / c.cand_block).copied().unwrap_or(false) {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
            let pos_list = select_topk(&scores, topk);
            out.push(TopK { pos: pos_list, scores });
        }
        Ok(out)
    }
}

/// The window positions a query at `t` attends to, in the reference's index order (None = -1).
pub fn window_positions(window: usize, t: usize, order: Order) -> Vec<Option<usize>> {
    match order {
        Order::Prefill { p_total } => {
            // get_window_topk_idxs prefill branch: (t - window + 1).clamp(0) + arange(min(P, window)), > t -> -1
            let base = (t + 1).saturating_sub(window);
            (0..p_total.min(window)).map(|j| Some(base + j).filter(|&p| p <= t)).collect()
        }
        Order::Decode => {
            // decode branch: slots oldest first; slot s holds the latest position = s (mod window); slot > t -> -1
            let oldest = t % window + 1;
            (oldest..window).chain(0..oldest).map(|s| if s > t { None } else { Some(t - ((t - s) % window)) }).collect()
        }
    }
}

/// The reference's top-k over one score row (`topk(k, sorted=False)` then `.sort()`): the k largest
/// scores, ties to the LOWER position (torch's behaviour on the fixtures, REFb §4), returned in
/// ascending position order. `scores` covers the reachable positions only; when fewer than k are
/// reachable the rest of the list is -1 (the reference picks unreachable -inf positions there and
/// maps them to -1).
pub fn select_topk(scores: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let take = k.min(scores.len());
    let mut sel: Vec<i32> = idx[..take].iter().map(|&i| i as i32).collect();
    sel.sort_unstable();
    sel.resize(k, -1);
    sel
}

/// `select_candidate_blocks` (model.py 584-611) for one query whose reachable positions are
/// `scores` (length `cl` = compress_lens): block score = max over `block` positions, the block of
/// the newest reachable position pinned to +inf, top min(topk_blocks, n_blocks) blocks kept when
/// finite. Returns the keep flag per block. Blocks wholly past the reach are -inf in the reference
/// and never kept, so only reachable blocks are scored here.
pub fn select_candidate_blocks(scores: &[f32], cl: usize, topk_blocks: usize, block: usize) -> Vec<bool> {
    let bs = block_scores(scores, cl, block);
    let nb = bs.len();
    let mut idx: Vec<usize> = (0..nb).collect();
    idx.sort_by(|&a, &b| bs[b].partial_cmp(&bs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let mut keep = vec![false; nb];
    for &b in idx.iter().take(topk_blocks.min(nb)) {
        keep[b] = bs[b] > f32::NEG_INFINITY;
    }
    keep
}

/// The candidate blocks' scores for one query: max over each block of `block` reachable positions,
/// the block holding the newest reachable position pinned to +inf.
pub fn block_scores(scores: &[f32], cl: usize, block: usize) -> Vec<f32> {
    let nb = cl.div_ceil(block);
    let mut bs: Vec<f32> = (0..nb).map(|b| scores[b * block..((b + 1) * block).min(cl)].iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
    if cl > 0 {
        bs[(cl - 1) / block] = f32::INFINITY;
    }
    bs
}

/// `sparse_attn` for one (query, head), REFb's emulation op for op: index list in the given order,
/// padded to a multiple of 64; per block the running max (init -1e30), p = exp(s - m) in fp32, the
/// denominator from the fp32 p (l = fp32(l * corr + sum p), formed in f64), P.V from bf16(p), the
/// sink in the denominator only, bf16 output. Scores = fp32(q . kv) * fp32(head_dim^-0.5).
pub fn sparse_attn_head(q: &[f32], list: &[Option<&[f32]>], sink: f32, scale: f32) -> Vec<f32> {
    let d = q.len();
    let nblk = list.len().div_ceil(ATTN_BLOCK);
    let s: Vec<f32> = (0..nblk * ATTN_BLOCK)
        .map(|j| match list.get(j).copied().flatten() {
            Some(kv) => (dot_f64(q, kv) as f32) * scale,
            None => f32::NEG_INFINITY,
        })
        .collect();
    let no_bf16_p = variant::is(variant::NO_BF16_P);
    let mut m = -1e30f32;
    let mut l = 0f32;
    let mut acc = vec![0f32; d];
    let mut pv = vec![0f64; d];
    for b in 0..nblk {
        let sb = &s[b * ATTN_BLOCK..(b + 1) * ATTN_BLOCK];
        let m_prev = m;
        m = sb.iter().copied().fold(m_prev, f32::max);
        let corr = (m_prev - m).exp();
        let p: Vec<f32> = sb.iter().map(|&v| (v - m).exp()).collect();
        let psum: f64 = p.iter().map(|&v| v as f64).sum();
        l = ((l as f64) * (corr as f64) + (psum as f32) as f64) as f32;
        pv.iter_mut().for_each(|v| *v = 0.0);
        for (j, &pj) in p.iter().enumerate() {
            let pb = if no_bf16_p { pj as f64 } else { bf16(pj) as f64 };
            if pb == 0.0 {
                continue;
            }
            let kv = list[b * ATTN_BLOCK + j].expect("a finite p comes from a valid index");
            for (a, &v) in pv.iter_mut().zip(kv) {
                *a += pb * v as f64;
            }
        }
        for (a, &v) in acc.iter_mut().zip(&pv) {
            *a = *a * corr + v as f32;
        }
    }
    if !variant::is(variant::NO_SINK) {
        l += (sink - m).exp();
    }
    acc.iter().map(|&a| bf16(a / l)).collect()
}

/// TP check helper: `wo_b` as N ranks would compute it when its input (o_groups x o_lora) is split
/// into N equal K ranges (one o_group per rank at TP8): each rank's fp32 partial over its K blocks,
/// summed in rank order in fp32, one bf16 rounding (T0 §3.3). Returns rows x dim.
pub fn wo_b_split(wo_b: &Fp8Lin, x: &[f32], rows: usize, ranks: usize) -> Vec<f32> {
    let kb = wo_b.k / Fp8Lin::BLOCK;
    assert!(kb % ranks == 0, "wo_b K blocks {kb} do not split into {ranks} ranks");
    let per = kb / ranks;
    let mut sum = vec![0f32; rows * wo_b.n];
    for r in 0..ranks {
        let part = wo_b.forward_f32(x, rows, r * per..(r + 1) * per);
        for (s, p) in sum.iter_mut().zip(part) {
            *s += p;
        }
    }
    sum.into_iter().map(bf16).collect()
}
