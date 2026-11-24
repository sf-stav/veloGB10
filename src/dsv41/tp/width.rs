//! DSV4.1 TP (leg SE): the verify-width probe GA's G4 reads (`--probe-dsv41-tp --dsv41-tp-program model --dsv41-tp-run width`).
//! Batch invariance at every width a speculative verify uses: a lane prefilled with a chat prompt, then ONE call of
//! 24 rows (fixed tokens at consecutive positions) is the reference; for every width w = 1..=24 the lane is restored to
//! the same state and the first w rows are run as a w-row call; every row r < w must equal row r of the 24-row call in
//! its head output bits (greedy id, log-sum, max logit). Success line: `WIDTH PROBE (G4): widths 1-24 ... PASS`.

use super::fwd::{HeadWant, TpModel};
use super::prog::Guard;
use crate::dsv41::ced::schedule::Mode;
use crate::tokenizer::QwenTokenizer;
use anyhow::Result;

pub fn run(m: &mut TpModel, guard: &mut Guard, tok: &QwenTokenizer, chunk: usize, mode: Mode, say: &mut dyn FnMut(String)) -> Result<bool> {
    let ids = super::fwprog::encode(tok, "Explain, step by step, how a refrigerator moves heat out of its interior.", true)?;
    m.reset_lane_n(0)?;
    super::fwprog::prefill(m, None, 0, &ids, chunk, mode)?;
    let snap = m.snapshot(0)?;
    let toks: Vec<u32> = (0..24).map(|i| 1000 + ((i * 37) % 50_000) as u32).collect();
    let key = |o: &super::comm::TpOut| (o.greedy, o.lse.to_bits(), o.max_x.to_bits());
    let reference: Vec<_> = m.forward_segs(&[(0, &toks[..])], HeadWant::All, None)?.iter().map(key).collect();
    guard.agree("width reference", reference.iter().fold(0u32, |a, r| a.wrapping_mul(31) ^ r.0 as u32))?;
    anyhow::ensure!(reference.len() == 24, "the 24-row reference call returned {} rows", reference.len());
    let (mut rows, mut bad) = (0usize, Vec::new());
    for w in 1..=24usize {
        m.restore(0, &snap)?;
        let outs = m.forward_segs(&[(0, &toks[..w])], HeadWant::All, None)?;
        anyhow::ensure!(outs.len() == w, "the {w}-row call returned {} rows", outs.len());
        for (r, o) in outs.iter().enumerate() {
            rows += 1;
            if key(o) != reference[r] { bad.push(format!("w{w}r{r}")); }
        }
        guard.agree("width call", outs.iter().fold(w as u32, |a, o| a.wrapping_mul(31) ^ o.greedy as u32))?;
    }
    m.reset_lane_n(0)?;
    let ok = bad.is_empty();
    say(format!("WIDTH PROBE (G4): widths 1-24 on a {}-token context, {rows} head rows compared with the 24-row call's: {} {}", ids.len(),
                if ok { "all bit-identical PASS".to_string() } else { format!("{} DIFFER FAIL", bad.len()) }, bad.iter().take(8).cloned().collect::<Vec<_>>().join(" ")));
    Ok(ok)
}
