//! DSV4.1 TP (leg SE): presence / frequency / repetition penalties for the serving rounds. Every rank keeps the same
//! per-lane counts of the request's generated tokens (the emitted tokens are identical on every rank) and applies the
//! penalty to its own vocabulary slice of each verify row's target logits, after the head and before DS's sampler
//! (`Round::verify`). Row j of a lane's chain sees the counts plus the drafts before it (d1..dj), as the host sampler
//! sees the history. The draft distribution is not penalised (any proposal distribution is exact in the acceptance rule).

use super::tpk::{PenArgs, TpfKernels};
use crate::dsv41::fwd::dev::{Buf, Gpu};
use anyhow::Result;
use cudarc::driver::CudaStream;

pub struct Pen {
    k: TpfKernels,
    counts: Buf,
    host: Vec<Vec<u16>>,
    dirty: Vec<bool>,
    /// (presence, frequency, repetition) per lane; None = the lane's request has none
    par: Vec<Option<[f32; 3]>>,
    v: usize,
    v0: usize,
    vlocal: usize,
}

impl Pen {
    pub fn new(g: &Gpu, lanes: usize, vocab: usize, v0: usize, vlocal: usize) -> Result<Pen> {
        Ok(Pen { k: TpfKernels::load(&g.dev)?, counts: g.buf(lanes * vocab * 2)?, host: vec![Vec::new(); lanes], dirty: vec![true; lanes], par: vec![None; lanes], v: vocab, v0, vlocal })
    }
    /// A new request on `lane` (empty counts); `par` = its penalties when it has any.
    pub fn set(&mut self, lane: usize, par: Option<[f32; 3]>) {
        self.host[lane].clear();
        self.par[lane] = par.filter(|p| p[0] != 0.0 || p[1] != 0.0 || (p[2] - 1.0).abs() > 1e-9);
        self.dirty[lane] = true;
    }
    pub fn any(&self) -> bool { self.par.iter().any(|p| p.is_some()) }
    /// Device bytes of the counts (leg MEM1's ledger).
    pub fn dev_bytes(&self) -> u64 { self.counts.bytes as u64 }
    pub fn active(&self, lane: usize) -> bool { self.par[lane].is_some() }
    /// The lane emitted `toks`.
    pub fn add(&mut self, lane: usize, toks: &[u32]) {
        if self.par[lane].is_none() { return; }
        let h = &mut self.host[lane];
        if h.is_empty() { h.resize(self.v, 0); }
        for &t in toks { if let Some(c) = h.get_mut(t as usize) { *c = c.saturating_add(1); } }
        self.dirty[lane] = true;
    }
    /// Penalise the `k + 1` rows of `lane`'s chain at `rows` (row j at `rows + j * vlocal * 4`); `drafts` = d1..dk.
    pub fn apply(&mut self, g: &Gpu, cs: &CudaStream, lane: usize, rows: u64, k: usize, drafts: &[u32]) -> Result<()> {
        let Some(p) = self.par[lane] else { return Ok(()) };
        let dev = self.counts.ptr() + (lane * self.v * 2) as u64;
        if self.dirty[lane] {
            if self.host[lane].is_empty() { g.memset(dev, self.v * 2, 0)?; } else { g.h2d_async(dev, &self.host[lane])?; }
            self.dirty[lane] = false;
        }
        for j in 0..=k {
            let mut e = [0i32; 5];
            let n = j.min(5).min(drafts.len());
            for i in 0..n { e[i] = drafts[i] as i32; }
            let a = PenArgs { logits: rows + (j * self.vlocal * 4) as u64, counts: dev, v0: self.v0 as i64, vlocal: self.vlocal as i32, n_extra: n as i32, e,
                              pres: p[0], freq: p[1], rep: p[2], ..Default::default() };
            self.k.penalty(cs, &a)?;
        }
        Ok(())
    }
}
