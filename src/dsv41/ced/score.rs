//! CE: the score mode (GA's G1, teacher-forced NLL): an EXACT prefill (`Mode::Exact`: every layer on every row,
//! since CED computes the decoder only on its cone and has no logits for the other rows; DeepSeek's engine also
//! turns `prompt_logprobs` off under its truncation) followed by the head on EVERY row:
//! `logits = head(norm(hc_pre(x_39, ffn_pre_39)))` in fp32 (model.py 1271-1272, ParallelHead), then a
//! full-vocabulary log-softmax; `logprobs[i] = log p(ids[i + 1] | ids[..= i])`.
//!
//! On the slice (no layers 3-13, 15-19, 22-39) the trunk part is the exact schedule the gate checks on every slice
//! layer; the head here is fed REFb's dumped final hidden (`G.h_final` = hc_pre of the last layer, every dumped
//! row) and compared with REFb's teacher-forced numbers (`G.tf_lse`, `G.tf_target_logit` on every prompt row) and
//! its full logits (`G.logits` rows): our norm + head + log-softmax against the reference's.
//! `--dsv41-score-out FILE` writes the g1 file format for every fixture whose dumped final hidden covers every row.

use super::probe::{bf16_bits, Gate};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::fwd::cmp::Pass;
use crate::dsv41::fwd::dev::dp;
use crate::dsv41::fwd::glue;
use crate::dsv41::fwd::layer::fmt_of;
use crate::dsv41::fwd::weights::{load_globals, Pack};
use crate::dsv41::gpu::mixers as km;
use crate::dsv41::weights::{ActDtype, Act, DevPtr, Scratch};
use anyhow::{ensure, Context, Result};
use sha2::Digest;
use std::path::Path;

/// g1's `ids_sha`: sha256 of the JSON list with no spaces.
pub fn ids_sha(ids: &[u32]) -> String {
    let s = format!("[{}]", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    sha2::Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// log-softmax of one row: (log p of every id as f64 lse, argmax).
fn lse_argmax(v: &[f32]) -> (f64, usize) {
    let (mut mx, mut am) = (f32::NEG_INFINITY, 0usize);
    for (i, &x) in v.iter().enumerate() { if x > mx { mx = x; am = i; } }
    let s: f64 = v.iter().map(|&x| ((x - mx) as f64).exp()).sum();
    (mx as f64 + s.ln(), am)
}

impl<'a> Gate<'a> {
    /// The score head on every fixture; returns the g1 file body when asked.
    pub fn score_gate(&mut self, pack: &Pack, fixes: &[super::probe::Fix], out: Option<&Path>) -> Result<()> {
        let g = self.f.g;
        let hw = load_globals(g, pack, &[C::FinalNorm, C::Head])?;
        let (head, backend) = hw.lin(C::Head)?;
        let norm = hw.param(C::FinalNorm)?;
        let d = self.f.cfg.hidden;
        let vocab = head.out as usize;
        let mut docs = serde_json::Map::new();
        for fx in fixes {
            let h = fx.d.global("h_final")?;
            let rows: Vec<usize> = h.positions().into_iter().map(|x| x as usize).collect();
            let ids: Vec<u32> = fx.d.global("prompt_ids")?.i.iter().map(|&x| x as u32).collect();
            let (tl, tt) = (fx.d.global("tf_lse")?, fx.d.global("tf_target_logit")?);
            let ref_full = fx.d.global_opt("logits")?;
            let t0 = std::time::Instant::now();
            let n = rows.len();
            let hd = g.up(&bf16_bits(&h.f))?;
            let mut ours_lp = Vec::with_capacity(n);   // (pos, logprob of the next id, argmax, lse)
            let mut kl = (0f64, 0usize, 0usize);          // sum KL(ref || ours), rows, top-1 agreements
            const PIECE: usize = 64;
            let mut r0 = 0;
            while r0 < n {
                let m = (n - r0).min(PIECE);
                self.f.g.sync()?;
                self.f.b.reset();
                let y = self.f.b.take(m * d * 2)?;
                let a = km::RmsNormArgs { x: dp(&hd) + (r0 * d * 2) as u64, x_ld: d as i64, w: norm.ptr.0, y, y_ld: d as i64, w_fmt: fmt_of(norm)?, dim: d as i32,
                                          rows: m as i32, eps: self.f.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
                g.launch1("d41_rmsnorm", (m as u32, 1, 1), (256, 1, 1), 0, &a)?;
                let lo = self.f.b.take(m * vocab * 4)?;
                let x = Act { ptr: DevPtr(y), rows: m, cols: d, ld: d, dtype: ActDtype::Bf16 };
                let o = Act { ptr: DevPtr(lo), rows: m, cols: vocab, ld: vocab, dtype: ActDtype::F32 };
                let be = glue::backend(*backend);
                ensure!(be.scratch_bytes(head, m) <= self.f.scr.bytes, "head scratch");
                be.logits(&g.l(), head, &x, &o, Scratch { ptr: DevPtr(self.f.scr.ptr), bytes: self.f.scr.bytes })?;
                crate::dsv41::fwd::receipts::kernel("ced.score_head", "d41_rmsnorm + backend logits (fp32)", m);
                let lg = g.down::<f32>(lo, m * vocab)?;
                for i in 0..m {
                    let pos = rows[r0 + i];
                    let row = &lg[i * vocab..(i + 1) * vocab];
                    let (lse, am) = lse_argmax(row);
                    let lp = if pos + 1 < ids.len() { row[ids[pos + 1] as usize] as f64 - lse } else { f64::NAN };
                    ours_lp.push((pos, lp, am, lse));
                    if let Some(rf) = &ref_full {
                        if let Some(j) = rf.index_of(pos as i64) {
                            let rr = rf.row_f(j);
                            let (rlse, ram) = lse_argmax(rr);
                            let mut k = 0f64;
                            for v in 0..vocab {
                                let pr = (rr[v] as f64 - rlse).exp();
                                if pr > 0.0 { k += pr * ((rr[v] as f64 - rlse) - (row[v] as f64 - lse)); }
                            }
                            kl.0 += k; kl.1 += 1; if ram == am { kl.2 += 1; }
                        }
                    }
                }
                r0 += m;
            }
            // against REFb's teacher-forced log-probs on every scored row with a next token
            let (mut d_sum, mut d_max, mut nll_o, mut nll_r, mut cnt) = (0f64, 0f64, 0f64, 0f64, 0usize);
            for &(pos, lp, _, _) in &ours_lp {
                if !lp.is_finite() { continue; }
                let i = tl.index_of(pos as i64).context("tf row")?;
                let rlp = tt.f[i] as f64 - tl.f[i] as f64;
                d_sum += (lp - rlp).abs(); d_max = d_max.max((lp - rlp).abs());
                nll_o -= lp; nll_r -= rlp; cnt += 1;
            }
            let (mean_d, dn) = (d_sum / cnt.max(1) as f64, (nll_o - nll_r) / cnt.max(1) as f64);
            let ok = match self.pass { Pass::Native => mean_d <= 1e-3 && d_max <= 2e-2, Pass::Exl3 => dn.abs() <= 0.05 };
            self.push(&fx.name, self.f.cfg.layers, "score: logprobs vs REFb tf", Some(ok && cnt > 0),
                      format!("{cnt} rows: mean |d logprob| {mean_d:.2e} max {d_max:.2e}; mean NLL ours {:.4} ref {:.4} (excess {dn:+.4} nats) [{}]",
                              nll_o / cnt.max(1) as f64, nll_r / cnt.max(1) as f64,
                              match self.pass { Pass::Native => "native: mean |d| <= 1e-3, max <= 2e-2", Pass::Exl3 => "EXL3 head: |excess| <= 0.05 nats" }));
            if kl.1 > 0 {
                self.push(&fx.name, self.f.cfg.layers, "score: full logits vs REFb", None,
                          format!("{} rows: mean KL(ref || ours) {:.3e}, top-1 agreement {:.4}", kl.1, kl.0 / kl.1 as f64, kl.2 as f64 / kl.1 as f64));
            }
            self.push(&fx.name, self.f.cfg.layers, "score: time", None, format!("{n} rows (norm + head + log-softmax on the host) in {:.1} s", t0.elapsed().as_secs_f64()));
            // the g1 file: only when every row of the prompt is scored
            if rows.len() == ids.len() && rows.iter().enumerate().all(|(i, &p)| i == p) {
                let lps: Vec<f64> = ours_lp.iter().filter(|x| x.1.is_finite()).map(|x| x.1).collect();
                ensure!(lps.len() == ids.len() - 1, "{}: {} log-probs for {} ids", fx.name, lps.len(), ids.len());
                docs.insert(fx.name.clone(), serde_json::json!({"ids_sha256": ids_sha(&ids), "logprobs": lps}));
            }
        }
        if let Some(path) = out {
            let n = docs.len();
            let body = serde_json::json!({"docs": docs, "source": "gb10_inference --probe-dsv41-ced score (slice: head on REFb's dumped final hidden)"});
            std::fs::write(path, serde_json::to_vec_pretty(&body)?).with_context(|| format!("write {}", path.display()))?;
            self.push("-", self.f.cfg.layers, "score: g1 file", None, format!("{} ({n} docs with every row scored)", path.display()));
        }
        Ok(())
    }
}
