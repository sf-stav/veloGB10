//! S1 — the exact f32 CPU reference forward of the looped Inkling-2 text model, op for op after
//! transformers `models/inkling2/modeling_inkling2.py` + `models/inkling2_loop/modeling_inkling2_loop.py`
//! (the oracle for every kernel that follows).
//!
//! Weights are read in the CHECKPOINT naming (`model.llm.*`), BF16/F32 tensors widened to f32,
//! routed experts dequantized on demand per visit (`pack::expert_f32`, bit-exact to the reference
//! loader). All dot products accumulate in f64; every activation is stored as f32.
//!
//! State (prefill then decode): per LOGICAL layer slot (`cfg.visits()` of them) the K/V cache and
//! the four short-conv input histories (k, v, attn-out, mlp-out). AttnRes block states are
//! per-token (depth-wise) and need no cache. The sliding window is applied by masking (the full
//! history is kept), which is exactly the reference's `kv > q - window` rule.
//!
//! Per-op notes the plan did not spell out (verified against the reference):
//!   * static short conv: weight `[C, 1, k]`, tap `k-1` weights the CURRENT token; dynamic conv
//!     `static_w [k, C]`: tap 0 weights the current token; both add the input back and divide by
//!     `sqrt(k + 1)` (static) / `sqrt(2k + 1)` (dynamic).
//!   * K/V: linear -> short conv -> (K: L2 clip with NO learned scale; V: unweighted RMSNorm).
//!     Q: linear -> learned scale -> L2 clip (-> log scale on full layers). The dynamic conv's
//!     per-token taps come from the (normed) key / value INPUT, not from the projected states.
//!   * relative bias, sinks and the output gate all read the QUERY input (`qx`).
//!   * MessageGate: `down` on concat(rms(h), rms(m)); forget / gain are tanh of rank-r projections.

use super::config::Inkling2Config;
use super::pack::{Inkling2Pack, P};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

// ------------------------------------------------------------------------------------------------
// numeric helpers
// ------------------------------------------------------------------------------------------------

fn nthreads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16) }

/// `y[t][o] = sum_i x[t][i] * w[o][i]` (w row-major `[n_out][n_in]`), f64 accumulation.
pub fn matmul(x: &[f32], t: usize, w: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    assert_eq!(x.len(), t * n_in);
    assert_eq!(w.len(), n_out * n_in, "weight {} vs {n_out}x{n_in}", w.len());
    let mut y = vec![0f32; t * n_out];
    let work = t * n_out * n_in;
    let nt = if work < 1 << 20 { 1 } else { nthreads() };
    let chunk = n_out.div_ceil(nt);
    // y laid out [t][n_out]; parallelize over output columns: compute into per-thread buffers
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt).map(|ti| {
            let (o0, o1) = (ti * chunk, ((ti + 1) * chunk).min(n_out));
            s.spawn(move || {
                let mut b = vec![0f32; t * o1.saturating_sub(o0)];
                for o in o0..o1 {
                    let wr = &w[o * n_in..(o + 1) * n_in];
                    for tt in 0..t {
                        let xr = &x[tt * n_in..(tt + 1) * n_in];
                        let mut acc = 0f64;
                        for i in 0..n_in { acc += xr[i] as f64 * wr[i] as f64; }
                        b[tt * (o1 - o0) + (o - o0)] = acc as f32;
                    }
                }
                (o0, b)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (o0, b) in parts {
        if b.is_empty() { continue; } // a thread past the last column (n_out < threads * chunk)
        let w_ = b.len() / t;
        for tt in 0..t { y[tt * n_out + o0..tt * n_out + o0 + w_].copy_from_slice(&b[tt * w_..(tt + 1) * w_]); }
    }
    y
}

pub(crate) fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    (1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt()) as f32
}
/// Unweighted RMSNorm of each `d`-row.
pub(crate) fn rms_rows(x: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut o = x.to_vec();
    for r in o.chunks_mut(d) { let s = rms_inv(r, eps); for v in r.iter_mut() { *v *= s; } }
    o
}
/// Weighted RMSNorm (`w * x * rsqrt(mean(x^2) + eps)`, the reference's `Inkling2RMSNorm`).
pub(crate) fn rmsw_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut o = x.to_vec();
    for r in o.chunks_mut(d) { let s = rms_inv(r, eps); for (v, g) in r.iter_mut().zip(w) { *v = g * (*v * s); } }
    o
}
pub(crate) fn l2clip(v: &mut [f32], thr: f32) {
    let n = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() as f32;
    let f = (thr / (n + 1e-6)).min(1.0);
    for x in v.iter_mut() { *x *= f; }
}
pub(crate) fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softcap(x: f32, c: f32) -> f32 { (x / c).tanh() * c }
pub(crate) fn log_scale(pos: usize, cfg: &Inkling2Config) -> f32 {
    1.0 + cfg.log_alpha * (((pos + 1) as f32) / cfg.log_floor).max(1.0).ln()
}
fn softmax_inplace(v: &mut [f64]) {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for x in v.iter_mut() { *x = (*x - m).exp(); s += *x; }
    for x in v.iter_mut() { *x /= s; }
}

// ------------------------------------------------------------------------------------------------
// weights
// ------------------------------------------------------------------------------------------------

pub(crate) struct Gate { pub(crate) down: Vec<f32>, pub(crate) forget: Vec<f32>, pub(crate) gain: Vec<f32> }

struct Layer {
    sliding: bool,
    h: usize, kv: usize, hd: usize, ext: usize,
    attn_norm: Option<Vec<f32>>,
    qkv_norm: Option<[Vec<f32>; 3]>,
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wr: Vec<f32>, wo: Vec<f32>, wg: Vec<f32>, sink: Vec<f32>,
    q_scale: Vec<f32>, r_scale: Vec<f32>, rel_proj: Vec<f32>,
    /// static: `[C][k]`; dynamic: (`static_w [k][C]`, `weight_proj [kv*k][D]`)
    k_conv: Conv, v_conv: Conv,
    attn_conv: Vec<f32>, mlp_conv: Vec<f32>,
    mlp_norm: Vec<f32>,
    router: Vec<f32>, router_bias: Vec<f32>,
    gates: Option<[Gate; 4]>, // q(attn), k, v, mlp
}
enum Conv { Static(Vec<f32>), Dynamic(Vec<f32>, Vec<f32>) }

pub struct CpuModel {
    pub cfg: Inkling2Config,
    pack: Inkling2Pack,
    layers: Vec<Layer>,
    attn_q: Vec<Vec<f32>>,  // index l-1 for layer l (1 or 3 queries)
    mlp_q: Vec<Vec<f32>>,
    out_q: Vec<f32>,
    msg_q: Vec<[Vec<f32>; 4]>, // per loop layer: q, k, v, mlp
    embed_norm: Vec<f32>,
    norm: Vec<f32>,
    unembed: Vec<f32>, // [unpadded][D]
}

/// Per-sequence decode state.
pub struct State {
    pub pos: usize,
    pub(crate) k: Vec<Vec<f32>>,          // [slot] -> [pos][kv*hd]
    pub(crate) v: Vec<Vec<f32>>,
    pub(crate) conv: Vec<[Vec<f32>; 4]>,  // [slot] -> [k_in, v_in, attn_out, mlp_out] histories, [pos][C]
}

pub type Taps = BTreeMap<String, Vec<f32>>;

impl CpuModel {
    pub fn load(dir: &Path) -> Result<CpuModel> {
        let pack = Inkling2Pack::open(dir)?;
        let cfg = pack.cfg.clone();
        let g = |n: &str| -> Result<Vec<f32>> { pack.plain(&format!("{P}{n}")) };
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let (h, kv, hd) = cfg.attn_geom(l);
            let sliding = cfg.is_sliding[l];
            let conv = |s: &str| -> Result<Conv> {
                Ok(if sliding { Conv::Static(g(&format!("{p}attn.{s}_sconv.weight"))?) }
                   else { Conv::Dynamic(g(&format!("{p}attn.{s}_sconv.static_w"))?, g(&format!("{p}attn.{s}_sconv.weight_proj"))?) })
            };
            let gate = |s: &str| -> Result<Gate> {
                Ok(Gate { down: g(&format!("{p}controllers.{s}.down_weight"))?, forget: g(&format!("{p}controllers.{s}.forget_weight"))?,
                          gain: g(&format!("{p}controllers.{s}.gain_weight"))? })
            };
            layers.push(Layer {
                sliding, h, kv, hd, ext: cfg.rel_ext(l),
                attn_norm: if sliding { Some(g(&format!("{p}attn_norm.weight"))?) } else { None },
                qkv_norm: if sliding { None } else {
                    Some([g(&format!("{p}q_input_norm.weight"))?, g(&format!("{p}k_input_norm.weight"))?, g(&format!("{p}v_input_norm.weight"))?])
                },
                wq: g(&format!("{p}attn.wq_du.weight"))?, wk: g(&format!("{p}attn.wk_dv.weight"))?, wv: g(&format!("{p}attn.wv_dv.weight"))?,
                wr: g(&format!("{p}attn.wr_du.weight"))?, wo: g(&format!("{p}attn.wo_ud.weight"))?, wg: g(&format!("{p}attn.gate_attn.weight"))?,
                sink: g(&format!("{p}attn.attn_sink_proj.weight"))?,
                q_scale: g(&format!("{p}attn.q_l2_clip.scale"))?, r_scale: g(&format!("{p}attn.rel_l2_clip.scale"))?,
                rel_proj: g(&format!("{p}attn.rel_logits_proj.proj"))?,
                k_conv: conv("k")?, v_conv: conv("v")?,
                attn_conv: g(&format!("{p}attn_sconv.weight"))?, mlp_conv: g(&format!("{p}mlp_sconv.weight"))?,
                mlp_norm: g(&format!("{p}mlp_norm.weight"))?,
                router: g(&format!("{p}mlp.gate.weight"))?, router_bias: g(&format!("{p}mlp.gate.bias"))?,
                gates: if cfg.is_loop(l) { Some([gate("attn")?, gate("k")?, gate("v")?, gate("mlp")?]) } else { None },
            });
        }
        let attn_q = (0..cfg.layers - 1).map(|i| g(&format!("attn_res.attn_queries.{i}"))).collect::<Result<Vec<_>>>()?;
        let mlp_q = (0..cfg.layers).map(|i| g(&format!("attn_res.mlp_queries.{i}"))).collect::<Result<Vec<_>>>()?;
        let d = cfg.hidden;
        let qm = g("middle_cycle_cross_pass_queries")?;
        let kvm = g("middle_cycle_cross_pass_kv_queries")?;
        let row = |m: &[f32], r: usize| m[r * d..(r + 1) * d].to_vec();
        // reference Inkling2LoopMergeMessageQueries: [q, k, v, mlp] = [queries[2k], kv[2k], kv[2k+1], queries[2k+1]]
        let msg_q = (0..cfg.loop_len()).map(|k| [row(&qm, 2 * k), row(&kvm, 2 * k), row(&kvm, 2 * k + 1), row(&qm, 2 * k + 1)]).collect();
        let unembed = pack.st.plain_f32_range(&format!("{P}unembed.weight"), 0, cfg.unpadded_vocab * d)?;
        Ok(CpuModel {
            out_q: g("attn_res.output_query")?, embed_norm: g("embed_norm.weight")?, norm: g("norm.weight")?,
            attn_q, mlp_q, msg_q, unembed, layers, cfg, pack,
        })
    }

    /// Probe access (the GPU kernel gates read the oracle's own tensors).
    pub(crate) fn pack(&self) -> &Inkling2Pack { &self.pack }
    pub(crate) fn gate(&self, l: usize, i: usize) -> Option<&Gate> { self.layers[l].gates.as_ref().map(|g| &g[i]) }
    /// AttnRes queries: layer input (`l >= 1`), MLP input, final, cross-pass messages (loop layer `k`).
    pub(crate) fn attn_queries(&self, l: usize) -> &[f32] { &self.attn_q[l - 1] }
    pub(crate) fn mlp_queries(&self, l: usize) -> &[f32] { &self.mlp_q[l] }
    pub(crate) fn msg_queries(&self, k: usize) -> Vec<f32> { self.msg_q[k].concat() }

    pub fn new_state(&self) -> State {
        let n = self.cfg.visits();
        State { pos: 0, k: vec![Vec::new(); n], v: vec![Vec::new(); n], conv: (0..n).map(|_| [vec![], vec![], vec![], vec![]]).collect() }
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let d = self.cfg.hidden;
        let mut x = Vec::with_capacity(ids.len() * d);
        for &t in ids {
            ensure!((t as usize) < self.cfg.vocab, "token {t} >= vocab");
            x.extend(self.pack.st.plain_f32_range(&format!("{P}embed.weight"), t as usize * d, (t as usize + 1) * d)?);
        }
        Ok(rmsw_rows(&x, &self.embed_norm, self.cfg.eps))
    }

    /// AttnRes over depth: for each token, values = rms(blocks ++ [partial]); out_q = softmax(q.V/sqrt(D)) V.
    pub(crate) fn attn_res(&self, blocks: &[Vec<f32>], partial: &[f32], queries: &[f32], t: usize) -> Vec<Vec<f32>> {
        let d = self.cfg.hidden;
        let nq = queries.len() / d;
        let mut outs = vec![vec![0f32; t * d]; nq];
        let scale = (d as f64).powf(-0.5);
        for tt in 0..t {
            let mut vals: Vec<Vec<f32>> = blocks.iter().map(|b| b[tt * d..(tt + 1) * d].to_vec()).collect();
            vals.push(partial[tt * d..(tt + 1) * d].to_vec());
            for v in vals.iter_mut() { let s = rms_inv(v, 1e-6); for x in v.iter_mut() { *x *= s; } }
            for qi in 0..nq {
                let q = &queries[qi * d..(qi + 1) * d];
                let mut sc: Vec<f64> = vals.iter().map(|v| v.iter().zip(q).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale).collect();
                softmax_inplace(&mut sc);
                let o = &mut outs[qi][tt * d..(tt + 1) * d];
                for c in 0..d { o[c] = vals.iter().zip(&sc).map(|(v, &p)| v[c] as f64 * p).sum::<f64>() as f32; }
            }
        }
        outs
    }

    pub(crate) fn message_gate(&self, g: &Gate, h: &[f32], m: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.hidden;
        let r = self.cfg.gate_rank;
        let mut cat = Vec::with_capacity(t * 2 * d);
        let (hn, mn) = (rms_rows(h, d, self.cfg.eps), rms_rows(m, d, self.cfg.eps));
        for tt in 0..t { cat.extend_from_slice(&hn[tt * d..(tt + 1) * d]); cat.extend_from_slice(&mn[tt * d..(tt + 1) * d]); }
        let z = matmul(&cat, t, &g.down, r, 2 * d);
        let f = matmul(&z, t, &g.forget, d, r);
        let gn = matmul(&z, t, &g.gain, d, r);
        (0..t * d).map(|i| (1.0 - f[i].tanh()) * h[i] + gn[i].tanh() * m[i]).collect()
    }

    /// Static depthwise causal conv over `hist` (all inputs so far, `[n][C]`) for the last `t` rows.
    pub(crate) fn conv_static(w: &[f32], hist: &[f32], c: usize, k: usize, t: usize) -> Vec<f32> {
        let n = hist.len() / c;
        let mut o = vec![0f32; t * c];
        for tt in 0..t {
            let pos = n - t + tt;
            for ch in 0..c {
                let mut acc = 0f64;
                for j in 0..k {
                    let lag = k - 1 - j; // w[j] weights token pos - (k-1-j)
                    if pos >= lag { acc += w[ch * k + j] as f64 * hist[(pos - lag) * c + ch] as f64; }
                }
                let x = hist[pos * c + ch];
                o[tt * c + ch] = ((acc as f32) + x) / ((k + 1) as f32).sqrt();
            }
        }
        o
    }

    pub(crate) fn conv_dynamic(&self, sw: &[f32], wp: &[f32], dyn_in: &[f32], hist: &[f32], kvh: usize, c: usize, t: usize) -> Vec<f32> {
        let k = self.cfg.conv_k;
        let d = self.cfg.hidden;
        let dw = matmul(dyn_in, t, wp, kvh * k, d); // [t][kvh][k]
        let hd = c / kvh;
        let n = hist.len() / c;
        let mut o = vec![0f32; t * c];
        for tt in 0..t {
            let pos = n - t + tt;
            for ch in 0..c {
                let head = ch / hd;
                let mut acc = 0f64;
                for lag in 0..k {
                    if pos >= lag {
                        let wgt = sw[lag * c + ch] + dw[tt * kvh * k + head * k + lag];
                        acc += hist[(pos - lag) * c + ch] as f64 * wgt as f64;
                    }
                }
                let x = hist[pos * c + ch];
                o[tt * c + ch] = ((acc as f32) + x) / ((2 * k + 1) as f32).sqrt();
            }
        }
        o
    }

    /// Q / relative-state post-processing of the `t` rows at positions `p0..`: learned scale, L2 clip,
    /// LogScale on full layers (D1).
    pub(crate) fn attn_qr(&self, l: usize, mut q: Vec<f32>, mut r: Vec<f32>, p0: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let ly = &self.layers[l];
        let (nh, hd) = (ly.h, ly.hd);
        for tt in 0..t {
            for hh in 0..nh {
                let v = &mut q[(tt * nh + hh) * hd..(tt * nh + hh + 1) * hd];
                for i in 0..hd { v[i] *= ly.q_scale[hh * hd + i]; }
                l2clip(v, c.qk_clip);
                if !ly.sliding { let s = log_scale(p0 + tt, c); for x in v.iter_mut() { *x *= s; } }
                let rv = &mut r[(tt * nh + hh) * c.d_rel..(tt * nh + hh + 1) * c.d_rel];
                for i in 0..c.d_rel { rv[i] *= ly.r_scale[hh * c.d_rel + i]; }
                l2clip(rv, c.rel_clip);
                if !ly.sliding { let s = log_scale(p0 + tt, c); for x in rv.iter_mut() { *x *= s; } }
            }
        }
        (q, r)
    }

    /// Short convs on the projected K / V (histories in the state), K L2 clip, V unweighted RMSNorm,
    /// append to the slot's K / V cache (D2). `kx` / `vx`: the normed key / value inputs (dynamic taps, D4).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kv_append(&self, l: usize, st: &mut State, slot: usize, kl: &[f32], vl: &[f32], kx: &[f32], vx: &[f32], t: usize) {
        let c = &self.cfg;
        let ly = &self.layers[l];
        let (kvh, hd) = (ly.kv, ly.hd);
        let k = c.conv_k;
        let ckv = kvh * hd;
        st.conv[slot][0].extend_from_slice(kl);
        st.conv[slot][1].extend_from_slice(vl);
        let (mut kc, mut vc) = match (&ly.k_conv, &ly.v_conv) {
            (Conv::Static(wk), Conv::Static(wv)) => (Self::conv_static(wk, &st.conv[slot][0], ckv, k, t), Self::conv_static(wv, &st.conv[slot][1], ckv, k, t)),
            (Conv::Dynamic(sk, pk), Conv::Dynamic(sv, pv)) => (
                self.conv_dynamic(sk, pk, kx, &st.conv[slot][0], kvh, ckv, t),
                self.conv_dynamic(sv, pv, vx, &st.conv[slot][1], kvh, ckv, t)),
            _ => unreachable!(),
        };
        for tt in 0..t {
            for hh in 0..kvh {
                l2clip(&mut kc[(tt * kvh + hh) * hd..(tt * kvh + hh + 1) * hd], c.qk_clip);
                let vv = &mut vc[(tt * kvh + hh) * hd..(tt * kvh + hh + 1) * hd];
                let s = rms_inv(vv, c.eps);
                for x in vv.iter_mut() { *x *= s; }
            }
        }
        st.k[slot].extend_from_slice(&kc);
        st.v[slot].extend_from_slice(&vc);
    }

    /// Attention of the `t` rows at `p0..` over the slot's cache (relative bias, sink, sigmoid gate):
    /// the gated `[t][nh*hd]` input of `wo`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend(&self, l: usize, st: &State, slot: usize, q: &[f32], r: &[f32], sink: &[f32], gate: &[f32], p0: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let ly = &self.layers[l];
        let (nh, kvh, hd) = (ly.h, ly.kv, ly.hd);
        let ckv = kvh * hd;
        let nkv = st.k[slot].len() / ckv;
        // --- attention with relative bias + sinks
        let grp = nh / kvh;
        let scale = (hd as f64).powf(-0.5);
        let mut att = vec![0f32; t * nh * hd];
        for tt in 0..t {
            let qp = p0 + tt;
            for hh in 0..nh {
                let qv = &q[(tt * nh + hh) * hd..(tt * nh + hh + 1) * hd];
                let rv = &r[(tt * nh + hh) * c.d_rel..(tt * nh + hh + 1) * c.d_rel];
                let kh = hh / grp;
                let mut keys = Vec::new();
                let mut lg = Vec::new();
                for kp in 0..nkv.min(qp + 1) {
                    let dist = qp - kp;
                    if ly.sliding && dist >= c.sliding_window { continue; }
                    let kv_ = &st.k[slot][(kp * kvh + kh) * hd..(kp * kvh + kh + 1) * hd];
                    let mut s = qv.iter().zip(kv_).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale;
                    if dist < ly.ext {
                        // rel_logits[h][dist] = sum_d r[h][d] * proj[d][dist]
                        let mut b = 0f64;
                        for dd in 0..c.d_rel { b += rv[dd] as f64 * ly.rel_proj[dd * ly.ext + dist] as f64; }
                        s += b as f32 as f64;
                    }
                    keys.push(kp);
                    lg.push(s);
                }
                lg.push(sink[tt * nh + hh] as f64);
                softmax_inplace(&mut lg);
                let o = &mut att[(tt * nh + hh) * hd..(tt * nh + hh + 1) * hd];
                for i in 0..hd {
                    let mut a = 0f64;
                    for (j, &kp) in keys.iter().enumerate() { a += lg[j] * st.v[slot][(kp * kvh + kh) * hd + i] as f64; }
                    o[i] = a as f32;
                }
            }
        }
        for i in 0..att.len() { att[i] *= sigmoid(gate[i]); }
        att
    }

    fn moe(&self, l: usize, x: &[f32], t: usize, taps: &mut Option<&mut Taps>, tag: &str) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, ne, ii) = (c.hidden, c.n_experts, c.inter);
        let ly = &self.layers[l];
        let logits = matmul(x, t, &ly.router, ne, d);
        let mut sel: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t);
        for tt in 0..t {
            let sc: Vec<f32> = logits[tt * ne..(tt + 1) * ne].iter().map(|&v| sigmoid(v)).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| (sc[b] + ly.router_bias[b]).partial_cmp(&(sc[a] + ly.router_bias[a])).unwrap().then(a.cmp(&b)));
            let top = &idx[..c.topk];
            let s: f32 = top.iter().map(|&e| sc[e]).sum();
            sel.push(top.iter().map(|&e| (e, sc[e] / (s + 1e-20) * c.route_scale)).collect());
        }
        if let Some(tp) = taps.as_deref_mut() {
            tp.insert(format!("{tag}.router_logits"), logits.clone());
        }
        let mut uniq: Vec<usize> = sel.iter().flat_map(|v| v.iter().map(|p| p.0)).collect();
        uniq.sort();
        uniq.dedup();
        // dequantize the union in parallel
        let nt = nthreads();
        let deq: Vec<Result<(usize, Vec<f32>, Vec<f32>)>> = std::thread::scope(|s| {
            let chunks: Vec<Vec<usize>> = (0..nt).map(|i| uniq.iter().cloned().skip(i).step_by(nt).collect()).collect();
            let hs: Vec<_> = chunks.into_iter().map(|ch| {
                let pack = &self.pack;
                s.spawn(move || ch.into_iter().map(|e| {
                    let (_, _, w13) = pack.expert_f32(l, e, "w13")?;
                    let (_, _, w2) = pack.expert_f32(l, e, "w2")?;
                    Ok((e, w13, w2))
                }).collect::<Vec<_>>())
            }).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut ex: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
        for r in deq { let (e, a, b) = r?; ex.insert(e, (a, b)); }
        let mut out = vec![0f32; t * d];
        for tt in 0..t {
            let xr = &x[tt * d..(tt + 1) * d];
            let mut acc = vec![0f64; d];
            for &(e, wgt) in &sel[tt] {
                let (w13, w2) = &ex[&e];
                let gu = matmul(xr, 1, w13, 2 * ii, d);
                let a: Vec<f32> = (0..ii).map(|i| softcap(gu[i], c.gate_softcap) * sigmoid(gu[i]) * softcap(gu[ii + i], c.up_softcap)).collect();
                let y = matmul(&a, 1, w2, d, ii);
                for j in 0..d { acc[j] += (y[j] * wgt) as f64; }
            }
            for j in 0..d { out[tt * d + j] = acc[j] as f32; }
        }
        Ok(out)
    }

    /// One decoder-layer call (physical `l`, loop pass `pass`) on the `t` new tokens.
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, st: &mut State, l: usize, pass: usize, h: &[f32], blocks: &[Vec<f32>], msgs: Option<&[Vec<f32>]>,
             t: usize, taps: &mut Option<&mut Taps>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let d = c.hidden;
        let ly = &self.layers[l];
        let slot = c.cache_slot(l, pass);
        let tag = format!("s{slot}");
        let (nh, kvh, hd) = (ly.h, ly.kv, ly.hd);
        let k = c.conv_k;
        let g = ly.gates.as_ref();
        // --- attention inputs
        let (qx, kx, vx);
        if ly.sliding {
            let x = if l > 0 { self.attn_res(blocks, h, &self.attn_q[l - 1], t).remove(0) } else { h.to_vec() };
            let w = ly.attn_norm.as_ref().unwrap();
            match msgs {
                None => { let n = rmsw_rows(&x, w, c.eps); qx = n.clone(); kx = n.clone(); vx = n; }
                Some(m) => {
                    let g = g.context("messages on a non-loop layer")?;
                    qx = rmsw_rows(&self.message_gate(&g[0], &x, &m[0], t), w, c.eps);
                    kx = rmsw_rows(&self.message_gate(&g[1], &x, &m[1], t), w, c.eps);
                    vx = rmsw_rows(&self.message_gate(&g[2], &x, &m[2], t), w, c.eps);
                }
            }
        } else {
            let (mut a, mut b, mut cc) = if l > 0 {
                let mut o = self.attn_res(blocks, h, &self.attn_q[l - 1], t);
                ensure!(o.len() == 3, "full layer {l}: {} attn queries", o.len());
                let z = o.pop().unwrap(); let y = o.pop().unwrap(); let x = o.pop().unwrap();
                (x, y, z)
            } else { (h.to_vec(), h.to_vec(), h.to_vec()) };
            if let Some(m) = msgs {
                let g = g.context("messages on a non-loop layer")?;
                a = self.message_gate(&g[0], &a, &m[0], t);
                b = self.message_gate(&g[1], &b, &m[1], t);
                cc = self.message_gate(&g[2], &cc, &m[2], t);
            }
            let w = ly.qkv_norm.as_ref().unwrap();
            qx = rmsw_rows(&a, &w[0], c.eps); kx = rmsw_rows(&b, &w[1], c.eps); vx = rmsw_rows(&cc, &w[2], c.eps);
        }
        // --- projections
        let q = matmul(&qx, t, &ly.wq, nh * hd, d);
        let kl = matmul(&kx, t, &ly.wk, kvh * hd, d);
        let vl = matmul(&vx, t, &ly.wv, kvh * hd, d);
        let r = matmul(&qx, t, &ly.wr, nh * c.d_rel, d);
        let sink = matmul(&qx, t, &ly.sink, nh, d);
        let gate = matmul(&qx, t, &ly.wg, nh * hd, d);
        let p0 = st.pos;
        let (q, r) = self.attn_qr(l, q, r, p0, t);
        self.kv_append(l, st, slot, &kl, &vl, &kx, &vx, t);
        let att = self.attend(l, st, slot, &q, &r, &sink, &gate, p0, t);
        let ao = matmul(&att, t, &ly.wo, d, nh * hd);
        st.conv[slot][2].extend_from_slice(&ao);
        let y = Self::conv_static(&ly.attn_conv, &st.conv[slot][2], d, k, t);
        if let Some(tp) = taps.as_deref_mut() { tp.insert(format!("{tag}.attn_out"), y.clone()); }
        // --- residual / block boundary
        let mut blocks_l: Vec<Vec<f32>> = blocks.to_vec();
        let residual: Vec<f32> = if c.is_boundary(l) { blocks_l.push(h.to_vec()); y } else { h.iter().zip(&y).map(|(a, b)| a + b).collect() };
        let mut mi = self.attn_res(&blocks_l, &residual, &self.mlp_q[l], t).remove(0);
        if let Some(m) = msgs { mi = self.message_gate(&g.unwrap()[3], &mi, &m[3], t); }
        let mn = rmsw_rows(&mi, &ly.mlp_norm, c.eps);
        let mo = self.moe(l, &mn, t, taps, &tag)?;
        st.conv[slot][3].extend_from_slice(&mo);
        let y2 = Self::conv_static(&ly.mlp_conv, &st.conv[slot][3], d, k, t);
        if let Some(tp) = taps.as_deref_mut() { tp.insert(format!("{tag}.mlp_out"), y2.clone()); }
        Ok(residual.iter().zip(&y2).map(|(a, b)| a + b).collect())
    }

    /// Run `ids` (the next tokens of the sequence in `st`). Returns the final normed hidden `[t][D]`.
    /// `taps` (if given) receives: `embed`, `v{i}` = output of the i-th decoder-layer call (visit
    /// order), per-slot attention / MoE outputs and router logits, `final`.
    pub fn forward(&self, st: &mut State, ids: &[u32], taps: Option<&mut Taps>) -> Result<Vec<f32>> {
        let x = self.embed(ids)?;
        self.forward_x(st, x, ids.len(), taps)
    }

    /// The same from the (normed) input embeddings `x` `[t][D]` (S5: image features already spliced in).
    pub fn forward_x(&self, st: &mut State, x: Vec<f32>, t: usize, mut taps: Option<&mut Taps>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if let Some(tp) = taps.as_deref_mut() { tp.insert("embed".into(), x.clone()); }
        let mut h = x;
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut visit = 0usize;
        let rec = |taps: &mut Option<&mut Taps>, h: &Vec<f32>, visit: &mut usize| {
            if let Some(tp) = taps.as_deref_mut() { tp.insert(format!("v{}", *visit), h.clone()); }
            *visit += 1;
        };
        for l in 0..c.loop_start {
            let hn = self.layer(st, l, 0, &h, &blocks, None, t, &mut taps)?;
            if c.is_boundary(l) { blocks.push(h.clone()); }
            h = hn;
            rec(&mut taps, &h, &mut visit);
        }
        let (pre_h, pre_b) = (h.clone(), blocks.clone());
        let npre = pre_b.len();
        for l in c.loop_start..c.loop_end {
            let hn = self.layer(st, l, 0, &h, &blocks, None, t, &mut taps)?;
            if c.is_boundary(l) { blocks.push(h.clone()); }
            h = hn;
            rec(&mut taps, &h, &mut visit);
        }
        for pass in 1..c.passes {
            let prev_b: Vec<Vec<f32>> = blocks[(npre + 1).min(blocks.len())..].to_vec();
            let prev_h = h.clone();
            h = pre_h.clone();
            blocks = pre_b.clone();
            for (kk, l) in (c.loop_start..c.loop_end).enumerate() {
                let msgs: Vec<Vec<f32>> = {
                    let q = &self.msg_q[kk];
                    let mut qs = Vec::with_capacity(4 * c.hidden);
                    for x in q { qs.extend_from_slice(x); }
                    self.attn_res(&prev_b, &prev_h, &qs, t)
                };
                let hn = self.layer(st, l, pass, &h, &blocks, Some(&msgs), t, &mut taps)?;
                if c.is_boundary(l) { blocks.push(h.clone()); }
                h = hn;
                rec(&mut taps, &h, &mut visit);
            }
        }
        for l in c.loop_end..c.layers {
            let hn = self.layer(st, l, 0, &h, &blocks, None, t, &mut taps)?;
            if c.is_boundary(l) { blocks.push(h.clone()); }
            h = hn;
            rec(&mut taps, &h, &mut visit);
        }
        let f = self.attn_res(&blocks, &h, &self.out_q, t).remove(0);
        let f = rmsw_rows(&f, &self.norm, c.eps);
        if let Some(tp) = taps.as_deref_mut() { tp.insert("final".into(), f.clone()); }
        st.pos += t;
        Ok(f)
    }

    /// Softcapped logits `[unpadded_vocab]` of one final hidden row.
    pub fn logits(&self, f: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        matmul(f, 1, &self.unembed, c.unpadded_vocab, c.hidden).into_iter().map(|v| softcap(v, c.logit_softcap)).collect()
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { num += (*x as f64 - *y as f64).powi(2); den += (*y as f64).powi(2); }
    (num / den.max(1e-300)).sqrt()
}

fn top_k(v: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx[..k.min(v.len())].iter().map(|&i| (i, v[i])).collect()
}

/// `--probe-inkling2-cpu --model-dir <pack> --oracle-dir <ref>`: runs every case of
/// `<ref>/cases.json` (written by tools/inkling2/ref_forward.py) and compares with
/// `<ref>/<case>.safetensors` taps. Case kinds: `taps` (every tap, rel-L2 bound), `logits`
/// (next token + top-k logits of the last prompt position, then teacher-forced decode steps).
pub fn probe_cpu(dir: &Path, ref_dir: &str, tol: Option<f64>) -> Result<()> {
    let t0 = std::time::Instant::now();
    let m = CpuModel::load(dir)?;
    println!("INK2_CPU loaded {} layers ({} visits) in {:.1}s", m.cfg.layers, m.cfg.visits(), t0.elapsed().as_secs_f64());
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(ref_dir).join("cases.json"))?)?;
    let mut fails = 0usize;
    for case in cases.as_array().context("cases.json is not a list")? {
        let name = case["name"].as_str().context("case name")?;
        let kind = case["kind"].as_str().unwrap_or("taps");
        let steps: Vec<Vec<u32>> = case["steps"].as_array().context("steps")?.iter()
            .map(|s| s.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()).collect();
        let r = super::pack::open_single(&Path::new(ref_dir).join(format!("{name}.safetensors")))?;
        let mut st = m.new_state();
        let tol = tol.unwrap_or(case["tol"].as_f64().unwrap_or(1e-5));
        println!("INK2_CASE {name} kind={kind} steps={:?} tol={tol:e}", steps.iter().map(|s| s.len()).collect::<Vec<_>>());
        for (si, ids) in steps.iter().enumerate() {
            let ts = std::time::Instant::now();
            let mut taps = Taps::new();
            let want_taps = kind == "taps" || r.has(&format!("step{si}.final"));
            let f = m.forward(&mut st, ids, if want_taps { Some(&mut taps) } else { None })?;
            let last = &f[(ids.len() - 1) * m.cfg.hidden..];
            let lg = m.logits(last);
            if kind == "taps" {
                let mut worst = (0f64, String::new());
                let mut n = 0;
                let mut all: Vec<String> = Vec::new();
                for (k, v) in &taps {
                    let rn = format!("step{si}.{k}");
                    if !r.has(&rn) { continue; }
                    let rv = r.plain_f32(&rn)?;
                    ensure!(rv.len() == v.len(), "{rn}: ref {} vs ours {}", rv.len(), v.len());
                    let e = rel_l2(v, &rv);
                    n += 1;
                    if e > worst.0 { worst = (e, rn.clone()); }
                    all.push(format!("{k}={e:.1e}"));
                    if e > tol { println!("  TAP_FAIL {rn} rel-L2 {e:.3e}"); fails += 1; }
                }
                let rl = format!("step{si}.logits");
                let el = rel_l2(&lg, &r.plain_f32(&rl)?);
                if el > tol { fails += 1; }
                println!("  step {si}: {n} taps compared, worst {} rel-L2 {:.3e}; last-row logits rel-L2 {el:.3e} ({:.1}s)", worst.1, worst.0, ts.elapsed().as_secs_f64());
                ensure!(n > 0, "no taps matched — harness failure, not a pass");
                println!("    all taps: {}", all.join(" "));
            } else {
                let rl = r.plain_f32(&format!("step{si}.logits"))?;
                let el = rel_l2(&lg, &rl);
                if want_taps {
                    // info only: per-visit hidden rel-L2 over the whole prefill
                    let mut v: Vec<(String, f64)> = Vec::new();
                    for (k, x) in &taps {
                        let rn = format!("step{si}.{k}");
                        if r.has(&rn) { v.push((k.clone(), rel_l2(x, &r.plain_f32(&rn)?))); }
                    }
                    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let pick = |n: &str| v.iter().find(|x| x.0 == n).map(|x| x.1).unwrap_or(f64::NAN);
                    println!("  step {si} taps (info): {} compared; embed {:.2e} v0 {:.2e} v5 {:.2e} v23 {:.2e} v47 {:.2e} final {:.2e}; worst {} {:.2e}",
                             v.len(), pick("embed"), pick("v0"), pick("v5"), pick("v23"), pick("v47"), pick("final"),
                             v.first().map(|x| x.0.as_str()).unwrap_or("-"), v.first().map(|x| x.1).unwrap_or(f64::NAN));
                }
                let ours = top_k(&lg, 20);
                let theirs = top_k(&rl, 20);
                let maxd = theirs.iter().map(|&(i, v)| (lg[i] - v).abs()).fold(0f32, f32::max);
                let same_top1 = ours[0].0 == theirs[0].0;
                let same_set = ours.iter().filter(|x| theirs.iter().any(|y| y.0 == x.0)).count();
                println!("  step {si} (+{} tok, pos {}): next ours {} ({:.4}) ref {} ({:.4}); top-20 overlap {same_set}/20, max |dlogit| on ref top-20 {maxd:.3e}; logits rel-L2 {el:.3e} ({:.1}s)",
                         ids.len(), st.pos, ours[0].0, ours[0].1, theirs[0].0, theirs[0].1, ts.elapsed().as_secs_f64());
                if !same_top1 || el > tol { fails += 1; println!("  LOGITS_FAIL"); }
            }
        }
    }
    if fails > 0 { anyhow::bail!("INK2_CPU_FAIL: {fails} comparisons above tolerance"); }
    println!("INK2_CPU_OK every case within tolerance");
    Ok(())
}
