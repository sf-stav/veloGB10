//! DSV4.1 leg OD: the DSpark draft forward on the CPU (model.py `DSparkBlock` 1103-1159,
//! `DSparkAttention` 1034-1077, `forward_spec` 1277-1285; REF_MAP §3.8 and §4 items 1, 19-22).
//!
//! The drafter, as the reference runs it:
//! * **Prefill** (`start_pos == 0`): `main_x = main_norm(main_proj(main_hidden))` for the prompt rows,
//!   and every stage seeds ITS window ring with `act_quant(rope(kv_norm(wkv(main_x)), pos))` of the last
//!   `window` rows (slot = pos % window). Nothing else runs.
//! * **Decode step** at position `p` (the main model just ran the token at `p` and sampled `t` for
//!   `p + 1`): `main_x` of row `p` goes into slot `p % window` of every stage's ring; the block
//!   `[t, noise x (block - 1)]` at rope positions `p + 1 .. p + block` runs through the three stages
//!   (each a full mHC block: window-only MQA over `[ring slots 0 .. min(window, p + 1)) ++ the block's own
//!   kv]`, bidirectional inside the block; a 128-expert top-3 MoE + shared expert), the pre-mix one
//!   sublayer late and threaded stage to stage; then `hc_pre`, the stage-2 norm, the tied head (fp32),
//!   the sequential Markov chain (`logits[i] += markov(ids[i])`, `ids[i + 1] = argmax`) and the
//!   confidence head on `[hc_pre output (un-normed); markov embed]` (a raw logit).
//! * The block's own kv is NOT written to the ring (REF_MAP §4 item 20).

use super::dspark_num::*;
use super::dspark_weights::{DsparkW, ExpertW, StageW};
use anyhow::Result;

/// One stage's window ring: slot -> (absolute position, kv row). Unwritten slots read as zeros, like
/// the reference's zero-initialised `window_kv_cache`.
#[derive(Clone)]
pub struct Ring { pub slots: Vec<Option<(usize, Vec<f32>)>> }

impl Ring {
    pub fn new(window: usize) -> Ring { Ring { slots: vec![None; window] } }
    pub fn put(&mut self, pos: usize, kv: Vec<f32>) { let w = self.slots.len(); self.slots[pos % w] = Some((pos, kv)); }
}

/// What one stage did at one decode step (for debugging and for the GPU legs' comparisons).
#[derive(Clone, Debug, Default)]
pub struct StageTrace {
    /// the stage's input stream `[hc * dim]` per block row, and the pre-mix its attention uses
    pub x_in: Vec<Vec<f32>>,
    pub pre_mix_in: Vec<Vec<f32>>,
    /// attention input (`attn_norm(hc_pre(x_in, pre_mix_in))`) and output (after `wo_b`)
    pub attn_x: Vec<Vec<f32>>,
    pub attn_out: Vec<Vec<f32>>,
    /// the stream after the attention `hc_post`
    pub hc_mid: Vec<Vec<f32>>,
    /// MoE input (`ffn_norm(hc_pre(hc_mid, attn_pre))`) and output
    pub ffn_x: Vec<Vec<f32>>,
    pub ffn_out: Vec<Vec<f32>>,
    pub x_out: Vec<Vec<f32>>,
    /// this stage's `ffn_pre` (the next stage's pre-mix)
    pub ffn_pre: Vec<Vec<f32>>,
    /// attention internals: q after RoPE `[heads * head_dim]`, the block's kv rows, the sparse-attention
    /// output before the inverse RoPE
    pub attn_q: Vec<Vec<f32>>,
    pub attn_kv: Vec<Vec<f32>>,
    pub attn_o: Vec<Vec<f32>>,
    /// the stage's whole window ring at this step `[window][head_dim]` (slot order; unwritten slots zero)
    pub ring: Vec<Vec<f32>>,
    /// the hc coefficients `[pre (hc), post (hc), comb (hc x hc, comb[j][k] at j * hc + k)]` of the attention
    /// and the FFN half (fp32)
    pub attn_mix: Vec<Vec<f32>>,
    pub ffn_mix: Vec<Vec<f32>>,
    /// `q_norm(wq_a(attn_x))` (the input of `wq_b`, before its act_quant)
    pub attn_qr: Vec<Vec<f32>>,
    /// the bf16 input of each routed expert's `w2` (`weight * silu(gate) * up`), row-major over (row, top-k slot),
    /// and of the shared expert's `w2`
    pub moe_h: Vec<Vec<f32>>,
    pub shared_h: Vec<Vec<f32>>,
    pub router_ids: Vec<Vec<usize>>,
    pub router_w: Vec<Vec<f32>>,
    /// the four highest biased gate scores per row (to see near-ties at the top-3 boundary)
    pub router_top4: Vec<Vec<f32>>,
    /// ring positions the block attended to (slot order)
    pub ring_positions: Vec<usize>,
}

/// The drafts of one decode step.
#[derive(Clone, Debug, Default)]
pub struct DraftOut {
    /// `[token at p + 1, drafts for p + 2 .. p + 1 + block]` (the reference's `output_ids`)
    pub ids: Vec<u32>,
    /// raw confidence logits `[block]`
    pub confidence: Vec<f32>,
    /// top-8 token ids of each row's Markov-biased logits, descending
    pub top8: Vec<Vec<u32>>,
    /// top-1 minus top-2 logit of each row (a near-tie warning when small)
    pub margin: Vec<f32>,
    /// with a probe chain (`Dspark::chain`'s `probe`): per row, the probe token's rank in this row's
    /// logits (0 = argmax) and its gap below the argmax logit
    pub probe: Vec<(usize, f32)>,
}

/// The CPU DSpark oracle.
pub struct Dspark {
    pub w: DsparkW,
    rope: Rope,
    scale: f32,
    /// keep every stage's intermediates in `StageTrace` (the router choices are always kept)
    pub trace: bool,
}

impl Dspark {
    pub fn new(w: DsparkW) -> Dspark {
        // DSpark stages have compress ratio 0: rope_theta without YaRN (model.py 681-697)
        let rope = Rope::plain(w.cfg.rope_dim, w.cfg.rope.theta);
        let scale = ((w.cfg.head_dim as f64).powf(-0.5)) as f32;
        Dspark { w, rope, scale, trace: false }
    }

    pub fn window(&self) -> usize { self.w.cfg.window }
    pub fn new_rings(&self) -> Vec<Ring> { (0..self.w.stages.len()).map(|_| Ring::new(self.window())).collect() }
    fn eps(&self) -> f32 { self.w.cfg.rms_eps as f32 }

    /// `main_norm(main_proj(main_hidden))` for one row (`forward_embed`, 1133). `mh`: [targets x dim] bf16 values.
    pub fn main_x(&self, mh: &[f32]) -> Vec<f32> {
        let y = qlinear(&[mh.to_vec()], &self.w.main_proj).pop().unwrap();
        rms_norm(&y, &self.w.main_norm, self.eps())
    }

    /// One stage's ring row for position `pos`: `act_quant(rope(kv_norm(wkv(main_x)), pos))` (1044-1046).
    pub fn main_kv(&self, stage: usize, main_x: &[f32], pos: usize) -> Vec<f32> {
        let st = &self.w.stages[stage];
        let y = qlinear(&[main_x.to_vec()], &st.wkv).pop().unwrap();
        let mut kv = rms_norm(&y, &st.kv_norm, self.eps());
        self.rope.apply(&mut kv, pos, false);
        act_quant_inplace(&mut kv, 32);
        kv
    }

    /// The prefill side of `forward_spec`: seed every stage's ring with the prompt row at `pos`.
    pub fn seed(&self, rings: &mut [Ring], main_x: &[f32], pos: usize) {
        for (s, r) in rings.iter_mut().enumerate() { r.put(pos, self.main_kv(s, main_x, pos)); }
    }

    // -------------------------------------------------------------------------------------- mHC pieces

    /// `Block.hc_mixes` (950-957) for each row `[hc * dim]`: (pre, post, comb).
    fn hc_mixes(&self, x: &[Vec<f32>], f: &F32Mat, scale: &[f32], base: &[f32]) -> Vec<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
        let c = &self.w.cfg;
        let mixes = linear_f32(x, f);
        x.iter().zip(mixes).map(|(xr, m)| {
            let r = rsqrt32(mean_sq(xr) + self.eps());
            let m: Vec<f32> = m.iter().map(|v| v * r).collect();
            sinkhorn(&m, scale, base, c.hc, c.sinkhorn_iters, c.hc_eps as f32)
        }).collect()
    }

    /// `hc_pre` (959-962): `bf16( sum_j pre[j] * x[j] )`, fp32 products and sum over the copies.
    fn hc_pre(&self, x: &[f32], pre: &[f32]) -> Vec<f32> {
        let d = self.w.cfg.hidden;
        (0..d).map(|e| {
            let mut s = 0f32;
            for (j, p) in pre.iter().enumerate() { s += p * x[j * d + e]; }
            bf16r(s)
        }).collect()
    }

    /// `hc_post` (964-968): `new[k] = bf16( post[k] * out + sum_j comb[j][k] * residual[j] )`.
    fn hc_post(&self, out: &[f32], res: &[f32], post: &[f32], comb: &[Vec<f32>]) -> Vec<f32> {
        let (d, hc) = (self.w.cfg.hidden, self.w.cfg.hc);
        let mut y = vec![0f32; hc * d];
        for k in 0..hc {
            for e in 0..d {
                let a = post[k] * out[e];
                let mut s = 0f32;
                for (j, cj) in comb.iter().enumerate() { s += cj[k] * res[j * d + e]; }
                y[k * d + e] = bf16r(a + s);
            }
        }
        y
    }

    // -------------------------------------------------------------------------------------- attention

    /// `DSparkAttention.forward` at `start_pos = p > 0` for the block rows `xs` (attn_norm outputs),
    /// after `main_kv(p)` has been written into `ring` (1056-1077).
    fn attention(&self, st: &StageW, xs: &[Vec<f32>], p: usize, ring: &Ring, tr: &mut StageTrace) -> Vec<Vec<f32>> {
        let c = &self.w.cfg;
        let (h, hd, rd) = (c.heads, c.head_dim, c.rope_dim);
        let eps = self.eps();
        let qr: Vec<Vec<f32>> = qlinear(xs, &st.wq_a).iter().map(|r| rms_norm(r, &st.q_norm, eps)).collect();
        if self.trace { tr.attn_qr = qr.clone(); }
        let mut q = qlinear(&qr, &st.wq_b);
        for (r, qrow) in q.iter_mut().enumerate() {
            for hh in 0..h { self.rope.apply(&mut qrow[hh * hd..(hh + 1) * hd], p + 1 + r, false); }
        }
        let mut kv: Vec<Vec<f32>> = qlinear(xs, &st.wkv).iter().map(|r| rms_norm(r, &st.kv_norm, eps)).collect();
        for (r, k) in kv.iter_mut().enumerate() {
            self.rope.apply(k, p + 1 + r, false);
            act_quant_inplace(k, 32);
        }
        // get_dspark_topk_idxs (1021-1031): ring slots 0 .. min(window, p + 1), then window + block rows
        let win = ring.slots.len();
        let nring = win.min(p + 1);
        let mut all: Vec<Vec<f32>> = ring.slots.iter().map(|s| s.as_ref().map(|(_, v)| v.clone()).unwrap_or_else(|| vec![0f32; hd])).collect();
        tr.ring_positions = ring.slots[..nring].iter().map(|s| s.as_ref().map_or(usize::MAX, |(pp, _)| *pp)).collect();
        all.extend(kv.iter().cloned());
        let list: Vec<i32> = (0..nring as i32).chain((0..xs.len() as i32).map(|j| win as i32 + j)).collect();
        let idx = vec![list; xs.len()];
        let mut o = sparse_attn(&q, h, hd, &all, &idx, &st.sink, self.scale);
        if self.trace {
            tr.attn_q = q.clone();
            tr.attn_kv = kv.clone();
            tr.attn_o = o.clone();
            tr.ring = all[..win].to_vec();
        }
        for (r, orow) in o.iter_mut().enumerate() {
            for hh in 0..h { self.rope.apply(&mut orow[hh * hd..(hh + 1) * hd], p + 1 + r, true); }
            debug_assert_eq!(rd, 2 * self.rope.freqs.len());
            act_quant_inplace(orow, 32);
        }
        // wo_a: block-diagonal over o_groups (einsum "bsgd,grd->bsgr"), bf16 with fp32 accumulation
        let (g, gin) = (c.o_groups, h * hd / c.o_groups);
        let gout = st.wo_a.n / g;
        let ya = par_cols(st.wo_a.n, o.len(), |col, out| {
            let grp = col / gout;
            let wr = &st.wo_a.w[col * gin..(col + 1) * gin];
            for (ri, orow) in o.iter().enumerate() {
                let x = &orow[grp * gin..(grp + 1) * gin];
                let mut d = 0f64;
                for j in 0..gin { d += x[j] as f64 * bf16_bits(wr[j]) as f64; }
                out[ri] = bf16r(d as f32);
            }
        });
        qlinear(&ya, &st.wo_b)
    }

    // -------------------------------------------------------------------------------------- MoE

    fn expert_fwd(&self, e: &ExpertW, xs: &[Vec<f32>], weights: Option<&[f32]>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let lim = self.w.cfg.swiglu_limit as f32;
        let g = qlinear(xs, &e.w1);
        let u = qlinear(xs, &e.w3);
        let hmid: Vec<Vec<f32>> = g.iter().zip(&u).enumerate().map(|(r, (gr, ur))| {
            gr.iter().zip(ur).map(|(&gv, &uv)| {
                let (gv, uv) = if lim > 0.0 { (gv.min(lim), uv.clamp(-lim, lim)) } else { (gv, uv) };
                let mut a = silu32(gv) * uv;
                if let Some(w) = weights { a *= w[r]; }
                bf16r(a)
            }).collect()
        }).collect();
        (qlinear(&hmid, &e.w2), hmid)
    }

    /// `MoE.forward` (891-906) with the stage's `Gate` (803-829): fp32 scores, sqrt(softplus), bias for
    /// the selection only, renormalised weights x route_scale; routed outputs (each bf16) summed in fp32
    /// in ascending expert id, then the shared expert, one bf16 rounding.
    fn moe(&self, st: &StageW, xs: &[Vec<f32>], tr: &mut StageTrace) -> Result<Vec<Vec<f32>>> {
        let c = &self.w.cfg;
        let k = c.dspark_top_k;
        let scores = linear_f32_bf16w(xs, &st.gate_w);
        let mut sel: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
        for sr in &scores {
            let sp: Vec<f32> = sr.iter().map(|&v| softplus32(v).sqrt()).collect();
            let biased: Vec<f32> = sp.iter().zip(&st.gate_bias).map(|(a, b)| a + b).collect();
            let ids = topk_desc(&biased, k);
            let mut w: Vec<f32> = ids.iter().map(|&i| sp[i]).collect();
            if c.norm_topk_prob && k > 1 {
                let mut sum = 0f32;
                for v in &w { sum += v; }
                let den = sum + 1e-20f32;
                for v in w.iter_mut() { *v /= den; }
            }
            for v in w.iter_mut() { *v *= c.routed_scale as f32; }
            tr.router_top4.push(topk_desc(&biased, k + 1).iter().map(|&i| biased[i]).collect());
            sel.push((ids, w));
        }
        let d = c.hidden;
        let mut y = vec![vec![0f32; d]; xs.len()];
        let mut hs: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); k]; xs.len()];
        let mut experts: Vec<usize> = sel.iter().flat_map(|(ids, _)| ids.iter().copied()).collect();
        experts.sort_unstable();
        experts.dedup();
        for e in experts {
            let rows: Vec<(usize, usize, f32)> = sel.iter().enumerate().filter_map(|(r, (ids, w))| ids.iter().position(|&x| x == e).map(|t| (r, t, w[t]))).collect();
            let ew = st.expert(&self.w.ckpt, c, e)?;
            let xin: Vec<Vec<f32>> = rows.iter().map(|(r, _, _)| xs[*r].clone()).collect();
            let wts: Vec<f32> = rows.iter().map(|(_, _, w)| *w).collect();
            let (out, hmid) = self.expert_fwd(&ew, &xin, Some(&wts));
            for (((r, t, _), o), h) in rows.iter().zip(out).zip(hmid) {
                for (a, b) in y[*r].iter_mut().zip(o) { *a += b; }
                if self.trace { hs[*r][*t] = h; }
            }
        }
        let (sh, shh) = self.expert_fwd(&st.shared, xs, None);
        for (yr, sr) in y.iter_mut().zip(sh) { for (a, b) in yr.iter_mut().zip(sr) { *a += b; } }
        if self.trace {
            tr.moe_h = hs.into_iter().flatten().collect();
            tr.shared_h = shh;
        }
        tr.router_ids = sel.iter().map(|(i, _)| i.clone()).collect();
        tr.router_w = sel.iter().map(|(_, w)| w.clone()).collect();
        Ok(y.into_iter().map(|r| r.into_iter().map(bf16r).collect()).collect())
    }

    // -------------------------------------------------------------------------------------- one stage

    /// `Block.forward` (970-996) for a DSpark stage at `start_pos = p > 0`. `x`: block rows `[hc * dim]`;
    /// `pre_mix`: per row `[hc]` (the previous stage's `ffn_pre`, one-hot at stage 0). Returns the new
    /// stream and this stage's `ffn_pre`.
    pub fn stage(&self, st: &StageW, x: Vec<Vec<f32>>, pre_mix: &[Vec<f32>], p: usize, ring: &Ring, tr: &mut StageTrace) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let eps = self.eps();
        let residual = x;
        let am = self.hc_mixes(&residual, &st.hc_attn_fn, &st.hc_attn_scale, &st.hc_attn_base);
        let xa: Vec<Vec<f32>> = residual.iter().zip(pre_mix).map(|(r, pm)| rms_norm(&self.hc_pre(r, pm), &st.attn_norm, eps)).collect();
        let ao = self.attention(st, &xa, p, ring, tr);
        let x1: Vec<Vec<f32>> = (0..residual.len()).map(|r| self.hc_post(&ao[r], &residual[r], &am[r].1, &am[r].2)).collect();
        let fm = self.hc_mixes(&x1, &st.hc_ffn_fn, &st.hc_ffn_scale, &st.hc_ffn_base);
        let xf: Vec<Vec<f32>> = x1.iter().zip(&am).map(|(r, m)| rms_norm(&self.hc_pre(r, &m.0), &st.ffn_norm, eps)).collect();
        let fo = self.moe(st, &xf, tr)?;
        let x2: Vec<Vec<f32>> = (0..x1.len()).map(|r| self.hc_post(&fo[r], &x1[r], &fm[r].1, &fm[r].2)).collect();
        let ffn_pre: Vec<Vec<f32>> = fm.iter().map(|m| m.0.clone()).collect();
        if self.trace {
            let flat = |m: &(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)| -> Vec<f32> { m.0.iter().chain(&m.1).chain(m.2.iter().flatten()).copied().collect() };
            tr.attn_mix = am.iter().map(flat).collect();
            tr.ffn_mix = fm.iter().map(flat).collect();
            tr.x_in = residual;
            tr.pre_mix_in = pre_mix.to_vec();
            tr.attn_x = xa;
            tr.attn_out = ao;
            tr.hc_mid = x1;
            tr.ffn_x = xf;
            tr.ffn_out = fo;
            tr.x_out = x2.clone();
            tr.ffn_pre = ffn_pre.clone();
        }
        Ok((x2, ffn_pre))
    }

    // -------------------------------------------------------------------------------------- one draft step

    /// The stage part of one decode step of `forward_spec` at `start_pos = p`: writes `main_x` (row `p`)
    /// into every stage's ring, runs the block `[token, noise ...]` through the stages and the tied head.
    /// The Markov chain runs in `chain` (it does not touch the stages).
    pub fn draft_body(&self, p: usize, token: u32, main_x: &[f32], rings: &mut [Ring]) -> Result<DraftBody> {
        assert!(p > 0, "the decode-side forward_spec needs start_pos > 0");
        let c = &self.w.cfg;
        let (d, hc, blk) = (c.hidden, c.hc, c.dspark_block);
        for (s, r) in rings.iter_mut().enumerate() { r.put(p, self.main_kv(s, main_x, p)); }
        let ids0: Vec<u32> = std::iter::once(token).chain(std::iter::repeat(c.dspark_noise_token).take(blk - 1)).collect();
        let mut x: Vec<Vec<f32>> = Vec::with_capacity(blk);
        for &t in &ids0 {
            let e = self.w.embed_row(t)?;
            let mut row = Vec::with_capacity(hc * d);
            for _ in 0..hc { row.extend_from_slice(&e); }
            x.push(row);
        }
        let mut pre_mix: Vec<Vec<f32>> = vec![{ let mut v = vec![0f32; hc]; v[0] = 1.0; v }; blk];
        let mut traces = Vec::new();
        for (s, st) in self.w.stages.iter().enumerate() {
            let mut tr = StageTrace::default();
            let (nx, npm) = self.stage(st, x, &pre_mix, p, &rings[s], &mut tr)?;
            x = nx;
            pre_mix = npm;
            traces.push(tr);
        }
        // forward_head (1140-1159): hc_pre with the last stage's ffn_pre, the stage-2 norm, the tied head
        let xh: Vec<Vec<f32>> = x.iter().zip(&pre_mix).map(|(r, pm)| self.hc_pre(r, pm)).collect();
        let xn: Vec<Vec<f32>> = xh.iter().map(|r| rms_norm(r, &self.w.norm, self.eps())).collect();
        let logits = linear_f32_bf16w(&xn, &self.w.head);
        Ok(DraftBody { token, logits, x_head: xh, stages: traces })
    }

    /// The Markov chain + confidence head (1149-1158). `forced`: when given, the chain is fed these ids
    /// (`[token, d1 .. d_block]`, e.g. the reference's drafts) instead of its own argmax, so one flipped
    /// draft does not change the bias of the later rows.
    pub fn chain(&self, body: &DraftBody, forced: Option<&[u32]>, probe: Option<&[u32]>) -> DraftOut {
        let c = &self.w.cfg;
        let (blk, r) = (c.dspark_block, c.dspark_markov_rank);
        let mut logits = body.logits.clone();
        let mut ids = vec![body.token];
        let mut membeds = Vec::with_capacity(blk);
        for i in 0..blk {
            let prev = match forced { Some(f) => f[i], None => ids[i] };
            let e: Vec<f32> = self.w.markov_embed.w[prev as usize * r..(prev as usize + 1) * r].iter().map(|b| bf16_bits(*b)).collect();
            let bias = linear_f32_bf16w(&[e.clone()], &self.w.markov_head).pop().unwrap();
            for (l, b) in logits[i].iter_mut().zip(&bias) { *l += b; }
            ids.push(argmax_first(&logits[i]) as u32);
            membeds.push(e);
        }
        let cin: Vec<Vec<f32>> = body.x_head.iter().zip(&membeds).map(|(a, b)| { let mut v = a.clone(); v.extend_from_slice(b); v }).collect();
        let confidence: Vec<f32> = linear_f32_bf16w(&cin, &self.w.confidence).into_iter().map(|v| v[0]).collect();
        let mut top8 = Vec::with_capacity(blk);
        let mut margin = Vec::with_capacity(blk);
        let mut pr = Vec::new();
        for (i, l) in logits.iter().enumerate() {
            let t = topk_desc(l, 8);
            margin.push(l[t[0]] - l[t[1]]);
            if let Some(pv) = probe {
                let lt = l[pv[i + 1] as usize];
                pr.push((l.iter().filter(|&&v| v > lt).count(), l[t[0]] - lt));
            }
            top8.push(t.into_iter().map(|v| v as u32).collect());
        }
        DraftOut { ids, confidence, top8, margin, probe: pr }
    }
}

/// The stage pass of one draft step (input to `Dspark::chain`).
pub struct DraftBody {
    pub token: u32,
    /// the tied head's fp32 logits `[block][vocab]`, before the Markov bias
    pub logits: Vec<Vec<f32>>,
    /// the un-normed `hc_pre` output after the last stage `[block][dim]` (the confidence head's input)
    pub x_head: Vec<Vec<f32>>,
    pub stages: Vec<StageTrace>,
}
