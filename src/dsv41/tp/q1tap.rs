//! Leg Q1: per-row teacher-forced scores and full-logit taps of the TP forward, for the same-method fidelity table
//! (`tools/dsv41_q1/tp_logits.py` stitches the ranks' vocabulary slices, `tools/dsv41_q1/compare.py` compares).
//!
//! `--dsv41-tp-run q1score --dsv41-q1-score-out DIR`: for every fixture of `--dsv41-tp-fixtures`, the prompt followed
//! by REFb's greedy continuation (teacher-forced) runs through the full non-CED chunked prefill; every rank writes
//! `DIR/<fixture>.rank<r>.safetensors` (on its own box) with the per-row greedy token, log-sum-exp and the next
//! token's log-probability (`Q1.greedy`, `Q1.lse`, `Q1.probe_lp`, rows 0..n-1) and its vocabulary slice of the full
//! logits at the tapped rows (`Q1.rows`, `Q1.logits` [rows][vocab slice] f32, `Q1.vocab` = [lo, hi)).

use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::st::{write_file, StOut, StSource};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Rows (absolute positions) whose full logits slice is downloaded in `TpModel::head`.
pub struct LogitTap {
    pub want: BTreeSet<usize>,
    pub rows: Vec<usize>,
    pub logits: Vec<f32>,
    pub vl: usize,
}

impl LogitTap {
    pub fn new(want: BTreeSet<usize>) -> LogitTap { LogitTap { want, rows: Vec::new(), logits: Vec::new(), vl: 0 } }
    /// Called by the head with its fp32 logits [n][vl] of rows pos0.. (before the exchange touches anything).
    pub fn take(&mut self, g: &Gpu, lg: u64, n: usize, vl: usize, pos0: usize) -> Result<()> {
        for i in 0..n {
            let p = pos0 + i;
            if self.want.contains(&p) {
                let v: Vec<f32> = g.down(lg + (i * vl * 4) as u64, vl)?;
                self.rows.push(p);
                self.logits.extend_from_slice(&v);
                self.vl = vl;
            }
        }
        Ok(())
    }
}

fn t<'a>(name: &str, dtype: &str, shape: Vec<u64>, b: &'a [u8]) -> StOut<'a> {
    StOut { name: name.into(), dtype: dtype.into(), shape, src: StSource::Bytes(b) }
}

fn le<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    let mut out = vec![0u8; n];
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.as_mut_ptr(), n) };
    out
}

/// Write one rank's file.
#[allow(clippy::too_many_arguments)]
pub fn write(path: &Path, ids: &[u32], prompt_len: usize, greedy: &[i64], lse: &[f32], probe_lp: &[f32], tap: &LogitTap, vocab: (usize, usize)) -> Result<()> {
    let ids64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let rows64: Vec<i64> = tap.rows.iter().map(|&x| x as i64).collect();
    let (b_ids, b_g, b_l, b_p, b_r, b_lg) = (le(&ids64), le(greedy), le(lse), le(probe_lp), le(&rows64), le(&tap.logits));
    let b_pl = le(&[prompt_len as i64]);
    let b_v = le(&[vocab.0 as i64, vocab.1 as i64]);
    let ts = vec![
        t("G.prompt_ids", "I64", vec![ids.len() as u64], &b_ids),
        t("Q1.prompt_len", "I64", vec![1], &b_pl),
        t("Q1.vocab", "I64", vec![2], &b_v),
        t("Q1.greedy", "I64", vec![greedy.len() as u64], &b_g),
        t("Q1.lse", "F32", vec![lse.len() as u64], &b_l),
        t("Q1.probe_lp", "F32", vec![probe_lp.len() as u64], &b_p),
        t("Q1.rows", "I64", vec![tap.rows.len() as u64], &b_r),
        t("Q1.logits", "F32", vec![tap.rows.len() as u64, tap.vl as u64], &b_lg),
    ];
    write_file(path, &ts, &BTreeMap::new(), &mut |_| {})?;
    Ok(())
}
