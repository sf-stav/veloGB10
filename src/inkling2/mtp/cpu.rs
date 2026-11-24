//! S6: the f32 CPU oracle of the bundled drafter, op for op after the external SGLang reference
//! (`models/inkling2_mtp.py` + the DFlash draft model it extends + `kernels/ops/speculative/inkling2_mtp.py`):
//!
//!   * context feature of layer l at position t: `rmsnorm(sum_i alpha[l][i] * tap_i(t)) * dflare_norm_l`,
//!     alpha = softmax(fuse_weight) rounded to bf16; taps = the residual ENTERING target visits 5 / 23 / 41 and the
//!     final-norm output (visit 48 is the reference's post-final-norm sentinel), all bf16;
//!   * context K = RoPE(k_norm_ctx(k_proj_ctx(feature))), V = v_proj_ctx(feature) at the absolute position t;
//!   * block rows: [embed_norm(embed(anchor)), mask_embed x 7] at positions P..P+7; pre-norm layers (input_layernorm,
//!     q / k head norms + RoPE, NON-causal attention over context ++ block, post_attention_layernorm, SwiGLU), final norm;
//!     local layers (0, 2) see context positions >= P - 1024 (the FA4 block-anchored window), layer 1 all of them;
//!   * base logits of block rows 1..7 = unembed . hidden (no cap); Markov step j: `[state | markov_w1[prev] |
//!     hidden_{j+1} | f_rnn]` -> rnn_proj -> gates (each op rounded to bf16) -> latent; step logits =
//!     `30 tanh((base + unembed . (hidden_out(latent))) / 30)` (the reference folds hidden_out into the head; the
//!     oracle keeps it unfolded, in f64).
//!
//! Linear inputs are rounded to bf16 (as the bf16 reference feeds them); everything else is f32 / f64.

use super::super::cpu::matmul;
use super::super::gpu::k1::{bf16_to_f32, f32_to_bf16};
use super::super::pack::{Inkling2Pack, P as PFX};
use super::gpu::fuse_alpha;
use anyhow::{Context, Result};

const D: usize = 2304;
const NH: usize = 24;
const NKV: usize = 8;
const HD: usize = 128;

fn bf(x: f32) -> f32 { bf16_to_f32(f32_to_bf16(x)) }
fn bfv(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| bf(v)).collect() }

fn rmsnorm_w(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let r = 1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| g * (v * r)).collect()
}

fn rope(x: &mut [f32], pos: usize, theta: f32) {
    for i in 0..HD / 2 {
        let inv = 1.0f32 / theta.powf((2 * i) as f32 / HD as f32);
        let ang = pos as f32 * inv;
        let (s, c) = ang.sin_cos();
        let (a, b) = (x[i], x[i + HD / 2]);
        x[i] = a * c - b * s;
        x[i + HD / 2] = b * c + a * s;
    }
}

struct Layer {
    q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, kc: Vec<f32>, vc: Vec<f32>, o: Vec<f32>, gate: Vec<f32>, up: Vec<f32>, down: Vec<f32>,
    in_norm: Vec<f32>, post_norm: Vec<f32>, q_norm: Vec<f32>, k_norm: Vec<f32>, k_norm_ctx: Vec<f32>, dflare: Vec<f32>, local: bool,
}

pub struct MtpCpu {
    layers: Vec<Layer>,
    alpha: Vec<f32>,
    norm: Vec<f32>,
    mask: Vec<f32>,
    mk_dflare: Vec<f32>,
    w1: Vec<f32>,
    rnn_w: Vec<f32>,
    rnn_b: Vec<f32>,
    ho_w: Vec<f32>,
    ho_b: Vec<f32>,
    embed_norm: Vec<f32>,
    eps: f32,
    target_eps: f32,
    theta: f32,
    window: usize,
    cap: f32,
    pub n: usize,
}

/// One drafted block (host): final hidden rows [8][D], base logits [7][n], step logits [7][n] (teacher-forced on `prev`).
pub struct BlockOut { pub fin: Vec<f32>, pub base: Vec<f32>, pub step: Vec<f32> }

/// Context K / V per layer, [t][8][128] each.
pub struct Ctx { k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, frnn: Vec<f32>, p: usize }

impl MtpCpu {
    pub fn load(pack: &Inkling2Pack) -> Result<MtpCpu> {
        let cfg = pack.cfg.mtp.clone().context("no mtp_config")?;
        let st = pack.mtp.as_ref().context("no mtp.safetensors")?;
        let f = |n: &str| st.plain_f32(n);
        let mut layers = Vec::new();
        for l in 0..3 {
            let p = format!("layers.{l}.");
            let a = format!("{p}self_attn.");
            layers.push(Layer {
                q: f(&format!("{a}q_proj.weight"))?, k: f(&format!("{a}k_proj.weight"))?, v: f(&format!("{a}v_proj.weight"))?,
                kc: f(&format!("{a}k_proj_ctx.weight"))?, vc: f(&format!("{a}v_proj_ctx.weight"))?, o: f(&format!("{a}o_proj.weight"))?,
                gate: f(&format!("{p}mlp.gate_proj.weight"))?, up: f(&format!("{p}mlp.up_proj.weight"))?, down: f(&format!("{p}mlp.down_proj.weight"))?,
                in_norm: f(&format!("{p}input_layernorm.weight"))?, post_norm: f(&format!("{p}post_attention_layernorm.weight"))?,
                q_norm: f(&format!("{a}q_norm.weight"))?, k_norm: f(&format!("{a}k_norm.weight"))?, k_norm_ctx: f(&format!("{a}k_norm_ctx.weight"))?,
                dflare: f(&format!("{p}dflare_norm.weight"))?, local: cfg.local_layer_ids.contains(&l),
            });
        }
        Ok(MtpCpu {
            layers, alpha: fuse_alpha(&f("fuse_weight")?), norm: f("norm.weight")?, mask: f("mask_embed")?,
            mk_dflare: f("markov_head.dflare_norm.weight")?, w1: f("markov_head.markov_w1.weight")?, rnn_w: f("markov_head.rnn_proj.weight")?,
            rnn_b: f("markov_head.rnn_proj.bias")?, ho_w: f("markov_head.hidden_out.weight")?, ho_b: f("markov_head.hidden_out.bias")?,
            embed_norm: pack.plain(&format!("{PFX}embed_norm.weight"))?, eps: cfg.eps, target_eps: pack.cfg.eps, theta: cfg.rope_theta as f32,
            window: cfg.sliding_window, cap: pack.cfg.logit_softcap, n: pack.cfg.unpadded_vocab,
        })
    }

    fn mix(&self, tap: &[f32], row: usize, w: &[f32]) -> Vec<f32> {
        let a = &self.alpha[row * 4..row * 4 + 4];
        let m: Vec<f32> = (0..D).map(|c| { let mut s = tap[c] * a[0]; for i in 1..4 { s = tap[i * D + c].mul_add(a[i], s); } s }).collect();
        rmsnorm_w(&m, w, self.eps)
    }

    /// Context from the taps of positions 0..p (`taps` [p][4][D], bf16 values).
    pub fn context(&self, taps: &[f32], p: usize) -> Ctx {
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for ly in &self.layers {
            let feat: Vec<f32> = (0..p).flat_map(|t| bfv(&self.mix(&taps[t * 4 * D..(t + 1) * 4 * D], ks.len(), &ly.dflare))).collect();
            let k = matmul(&feat, p, &ly.kc, NKV * HD, D);
            let v = matmul(&feat, p, &ly.vc, NKV * HD, D);
            let mut ko = vec![0f32; p * NKV * HD];
            for t in 0..p {
                for h in 0..NKV {
                    let o = (t * NKV + h) * HD;
                    let mut x = bfv(&rmsnorm_w(&bfv(&k[o..o + HD]), &ly.k_norm_ctx, self.eps));
                    rope(&mut x, t, self.theta);
                    ko[o..o + HD].copy_from_slice(&bfv(&x));
                }
            }
            ks.push(ko);
            vs.push(bfv(&v));
        }
        let frnn = bfv(&self.mix(&taps[(p - 1) * 4 * D..p * 4 * D], 3, &self.mk_dflare));
        Ctx { k: ks, v: vs, frnn, p }
    }

    /// The block for anchor `anchor_emb` (the target's raw bf16 embedding row) with the Markov chain teacher-forced on
    /// `prev` (prev[0] = anchor, prev[j] = draft j); `unembed` = [n][D] f32.
    pub fn block(&self, ctx: &Ctx, anchor_emb: &[f32], prev: &[u32], unembed: &[f32]) -> BlockOut {
        let p = ctx.p;
        let mut h = vec![0f32; 8 * D];
        h[..D].copy_from_slice(&rmsnorm_w(anchor_emb, &self.embed_norm, self.target_eps));
        for r in 1..8 { h[r * D..(r + 1) * D].copy_from_slice(&self.mask); }
        let scale = 1.0 / (HD as f32).sqrt();
        for (l, ly) in self.layers.iter().enumerate() {
            let x: Vec<f32> = (0..8).flat_map(|r| bfv(&rmsnorm_w(&h[r * D..(r + 1) * D], &ly.in_norm, self.eps))).collect();
            let q = matmul(&x, 8, &ly.q, NH * HD, D);
            let k = matmul(&x, 8, &ly.k, NKV * HD, D);
            let v = matmul(&x, 8, &ly.v, NKV * HD, D);
            let mut qs = vec![0f32; 8 * NH * HD];
            let mut bk = vec![0f32; 8 * NKV * HD];
            for r in 0..8 {
                for hh in 0..NH {
                    let o = (r * NH + hh) * HD;
                    let mut y = bfv(&rmsnorm_w(&bfv(&q[o..o + HD]), &ly.q_norm, self.eps));
                    rope(&mut y, p + r, self.theta);
                    qs[o..o + HD].copy_from_slice(&bfv(&y));
                }
                for hh in 0..NKV {
                    let o = (r * NKV + hh) * HD;
                    let mut y = bfv(&rmsnorm_w(&bfv(&k[o..o + HD]), &ly.k_norm, self.eps));
                    rope(&mut y, p + r, self.theta);
                    bk[o..o + HD].copy_from_slice(&bfv(&y));
                }
            }
            let bv = bfv(&v);
            let lo = if ly.local { p.saturating_sub(self.window) } else { 0 };
            let mut att = vec![0f32; 8 * NH * HD];
            for r in 0..8 {
                for hh in 0..NH {
                    let kvh = hh / (NH / NKV);
                    let qv = &qs[(r * NH + hh) * HD..(r * NH + hh + 1) * HD];
                    let keys: Vec<(&[f32], &[f32])> = (lo..p).map(|t| (&ctx.k[l][(t * NKV + kvh) * HD..(t * NKV + kvh + 1) * HD], &ctx.v[l][(t * NKV + kvh) * HD..(t * NKV + kvh + 1) * HD]))
                        .chain((0..8).map(|b| (&bk[(b * NKV + kvh) * HD..(b * NKV + kvh + 1) * HD], &bv[(b * NKV + kvh) * HD..(b * NKV + kvh + 1) * HD]))).collect();
                    let sc: Vec<f64> = keys.iter().map(|(kk, _)| qv.iter().zip(kk.iter()).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale as f64).collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|&s| (s - mx).exp()).collect();
                    let tot: f64 = e.iter().sum();
                    let o = &mut att[(r * NH + hh) * HD..(r * NH + hh + 1) * HD];
                    for d in 0..HD { o[d] = (keys.iter().zip(&e).map(|((_, vv), &w)| w * vv[d] as f64).sum::<f64>() / tot) as f32; }
                }
            }
            let ao = matmul(&bfv(&att), 8, &ly.o, D, NH * HD);
            for i in 0..8 * D { h[i] += ao[i]; }
            let x2: Vec<f32> = (0..8).flat_map(|r| bfv(&rmsnorm_w(&h[r * D..(r + 1) * D], &ly.post_norm, self.eps))).collect();
            let g = matmul(&x2, 8, &ly.gate, 9216, D);
            let u = matmul(&x2, 8, &ly.up, 9216, D);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| { let a = bf(a); bf(bf(a / (1.0 + (-a).exp())) * bf(b)) }).collect();
            let dn = matmul(&act, 8, &ly.down, D, 9216);
            for i in 0..8 * D { h[i] += dn[i]; }
        }
        let fin: Vec<f32> = (0..8).flat_map(|r| rmsnorm_w(&h[r * D..(r + 1) * D], &self.norm, self.eps)).collect();
        let n = self.n;
        let base = matmul(&fin[D..], 7, unembed, n, D);
        let mut step = vec![0f32; 7 * n];
        let mut state = vec![0f32; 256];
        for j in 0..7 {
            let mut inp = Vec::with_capacity(5120);
            inp.extend_from_slice(&state);
            inp.extend_from_slice(&self.w1[prev[j] as usize * 256..(prev[j] as usize + 1) * 256]);
            inp.extend(bfv(&fin[(j + 1) * D..(j + 2) * D]));
            inp.extend_from_slice(&ctx.frnn);
            let y = matmul(&inp, 1, &self.rnn_w, 768, 5120);
            let mut lat = vec![0f32; 256];
            for c in 0..256 {
                let gr = bf(y[c] + self.rnn_b[c]);
                let cr = bf(y[256 + c] + self.rnn_b[256 + c]);
                let or = bf(y[512 + c] + self.rnn_b[512 + c]);
                let gate = bf(1.0 / (1.0 + (-gr).exp()));
                let kept = bf(gate * state[c]);
                let blend = bf(bf(1.0 - gate) * bf(cr.tanh()));
                state[c] = bf(kept + blend);
                lat[c] = bf(or.tanh());
            }
            let hv: Vec<f32> = (0..D).map(|d| (self.ho_b[d] as f64 + (0..256).map(|r| self.ho_w[d * 256 + r] as f64 * lat[r] as f64).sum::<f64>()) as f32).collect();
            let bias = matmul(&hv, 1, unembed, n, D);
            for i in 0..n { step[j * n + i] = (((base[j * n + i] + bias[i]) / self.cap).tanh()) * self.cap; }
        }
        BlockOut { fin, base, step }
    }
}
