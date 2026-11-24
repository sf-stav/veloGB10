//! DSV4.1 TP (leg TPC): the TP smoke without the real forward (`--probe-dsv41-tp`). Every rank runs a synthetic
//! layer that writes partials of the REAL frame shapes (`frame::Shape::round`: 80 layer reduces with the mix rider,
//! the Engram and `main_proj` riders, the head exchange, the drafter's 6 reduces and 6 head-style exchanges), and
//! proves, on every rank:
//! 1. **reduces**: each reduced frame equals, bit for bit, the sum of every rank's partial in the transport's
//!    association, recomputed on this GPU (a single-rank computation of the same sums);
//! 2. **head exchange**: greedy / sample / log-sum / probe log-prob of the distributed exchange
//!    equal, bit for bit, the same kernels run over every rank's shard on this one GPU; the token ids also equal a
//!    one-shard (world 1) pass over the whole vocabulary, and planted exact ties resolve to the lowest id;
//! 3. **rank agreement**: one agree word over every output, equal on all ranks;
//! then times every collective of the round (eager, GPU-bound behind a spin gate; and the round as one captured
//! CUDA graph), with the per-step agree guard before every round. RESULT: `DSV41_TP_SMOKE_OK`.

use super::comm::{Dsv41Comm, HeadCall, TpComm, TpOut, TpRow};
use super::frame::{lanes_of, Class, Site, HEAD_REC};
use super::kern::LocalArgs;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice, CudaStream, DevicePtr};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SmokeOpts {
    pub rows: Vec<usize>,
    pub rounds: usize,
    pub graph: bool,
    pub seed: u64,
}

impl SmokeOpts {
    /// From the option registry (`--dsv41-tp-rows`, `--dsv41-tp-rounds`, `--dsv41-tp-graph`; SPMD options, so the
    /// node resolves the head's values from the shipped config).
    pub fn from_registry() -> Result<Self> {
        let rows_s = crate::opts::get(crate::opt!("dsv41-tp-rows")).unwrap_or_else(|| "1,6,12,24".into());
        let rows = rows_s.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>().with_context(|| format!("--dsv41-tp-rows {rows_s}"))?;
        ensure!(!rows.is_empty() && rows.iter().all(|&r| (1..=super::frame::MAX_ROWS).contains(&r)), "--dsv41-tp-rows: 1..{} each", super::frame::MAX_ROWS);
        let rounds = crate::opts::get(crate::opt!("dsv41-tp-rounds")).map(|s| s.parse::<usize>()).transpose().context("--dsv41-tp-rounds")?.unwrap_or(200);
        let graph = crate::opts::get(crate::opt!("dsv41-tp-graph")).map_or(true, |v| v != "off");
        Ok(SmokeOpts { rows, rounds: rounds.max(1), graph, seed: 0xD541_7C00_5EED })
    }
}

fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn fnv(h: &mut u64, x: u64) { for b in x.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100_0000_01B3); } }
fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }
fn sync(s: &CudaStream) -> Result<()> { unsafe { cudarc::driver::result::stream::synchronize(s.stream) }.context("stream sync")?; Ok(()) }

/// Raw timing events (cudarc 0.9 has none).
struct Ev(sys::CUevent);
impl Ev {
    fn new() -> Result<Ev> {
        let mut e: sys::CUevent = std::ptr::null_mut();
        let r = unsafe { sys::cuEventCreate(&mut e, 0) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
        Ok(Ev(e))
    }
    fn record(&self, s: &CudaStream) -> Result<()> {
        let r = unsafe { sys::cuEventRecord(self.0, s.stream) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r:?}");
        Ok(())
    }
    fn ms_to(&self, b: &Ev) -> Result<f32> {
        let mut ms = 0f32;
        let r = unsafe { sys::cuEventElapsedTime(&mut ms, self.0, b.0) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
        Ok(ms)
    }
}
impl Drop for Ev { fn drop(&mut self) { unsafe { sys::cuEventDestroy_v2(self.0); } } }

/// The per-step agree guard (AGENTS §2.10): every rank presents the same step token, or the run stops.
struct Guard { step: u64, world: usize }
impl Guard {
    fn agree(&mut self, what: &str, hash: u32) -> Result<()> {
        self.step += 1;
        if self.world < 2 { return Ok(()); }
        match crate::net::agree_ext(self.step, 0, 0, hash) {
            Some((_, w)) if w == crate::net::agree_word(0, hash) => { super::metrics::agreed(); Ok(()) }
            Some((_, w)) => bail!("dsv41 TP agree MISMATCH at step {} ({what}): peer word {w:#x}, mine {:#x}", self.step, crate::net::agree_word(0, hash)),
            None => bail!("dsv41 TP agree: the link aborted or the peer did not arrive (step {}, {what})", self.step),
        }
    }
}

pub struct Smoke<'a> {
    dev: Arc<CudaDevice>,
    comm: &'a TpComm,
    s: CudaStream,
    o: SmokeOpts,
    guard: Guard,
    pub lines: Vec<String>,
    /// every output's bits, folded: the rank-agreement word
    out_hash: u64,
    failures: Vec<String>,
    /// the eager head outputs per verify width (drafter, target), for the graph check
    head_eager: BTreeMap<usize, (Vec<TpOut>, Vec<TpOut>)>,
}

/// The head exchange's synthetic inputs at one verify width (identical on every rank: keyed by the width only).
struct HeadInputs {
    rows: usize,
    full: CudaSlice<f32>,
    dfull: CudaSlice<f32>,
    trows: CudaSlice<TpRow>,
    drows: CudaSlice<TpRow>,
    tr: Vec<TpRow>,
    dout: CudaSlice<TpOut>,
    out: CudaSlice<TpOut>,
}

impl<'a> Smoke<'a> {
    pub fn new(dev: &Arc<CudaDevice>, comm: &'a TpComm, o: SmokeOpts) -> Self {
        Smoke { dev: dev.clone(), comm, s: crate::gpu::fork_blocking_stream(dev), o, guard: Guard { step: 0, world: comm.world },
                lines: Vec::new(), out_hash: 0xCBF2_9CE4_8422_2325, failures: Vec::new(), head_eager: BTreeMap::new() }
    }

    fn head_inputs(&self, rows: usize) -> Result<HeadInputs> {
        let c = self.comm;
        let v = c.shape.vocab;
        let full = self.dev.alloc_zeros::<f32>(rows * v)?;
        let dfull = self.dev.alloc_zeros::<f32>(rows * v)?;
        self.synth_head_logits(rows, &full, &dfull)?;
        let lo = |r: usize| 17 + (r * 131) % 1000;
        let hi = |r: usize| v - 1 - (r * 977) % 1000;
        let mut tr = Vec::new();
        for r in 0..rows {
            let temp = [0.0f32, 1.0, 0.7, 1.0][r % 4];
            let probe = match r % 3 { 0 => lo(r), 1 => hi(r), _ => (r * 7919 + 12345 + (r % 2) * (v / 2)) % v } as i32;
            tr.push(TpRow { seed: self.o.seed ^ 0x5A5A, pos: 1000 + r as u64, temp, probe, flags: 0, pad: 0 });
        }
        let dr: Vec<TpRow> = tr.iter().map(|t| TpRow { flags: 1, ..*t }).collect();
        let (trows, drows) = (self.dev.htod_sync_copy(&tr)?, self.dev.htod_sync_copy(&dr)?);
        let dout = self.dev.htod_sync_copy(&vec![TpOut::default(); rows])?;
        let out = self.dev.htod_sync_copy(&vec![TpOut::default(); rows])?;
        Ok(HeadInputs { rows, full, dfull, trows, drows, tr, dout, out })
    }

    fn synth_head_logits(&self, rows: usize, full: &CudaSlice<f32>, dfull: &CudaSlice<f32>) -> Result<()> {
        let c = self.comm;
        let v = c.shape.vocab;
        let key = mix64(self.o.seed ^ 0x4EAD ^ ((rows as u64) << 20));
        c.kern.synth_logits(&self.s, dptr(full), v, rows, v, key)?;
        c.kern.synth_logits(&self.s, dptr(dfull), v, rows, v, mix64(key))
    }

    /// The drafter exchange and the target exchange (accept / residual are DS's sampler's since leg SE2).
    fn head_calls(&self, h: &HeadInputs) -> Result<()> {
        let c = self.comm;
        let (v, rows) = (c.shape.vocab, h.rows);
        let voff = (c.vocab().start * 4) as u64;
        c.head(&self.s, &HeadCall { site: Site::DraftHead, logits: dptr(&h.dfull) + voff, ld: v, rows, rows_p: dptr(&h.drows), out: dptr(&h.dout) })?;
        c.head(&self.s, &HeadCall { site: Site::Head, logits: dptr(&h.full) + voff, ld: v, rows, rows_p: dptr(&h.trows), out: dptr(&h.out) })
    }

    /// The same work as the correctness passes (every sum site synthesized, reduced and checked; the three head
    /// exchanges), captured as ONE CUDA graph and replayed: graph capture must not change a bit.
    fn graph_check(&mut self, rows: usize) -> Result<()> {
        let c = self.comm;
        let first = super::frame::first_round_bit(c.world);
        let sum_sites = self.sum_sites(rows);
        let mut bad = self.dev.alloc_zeros::<u32>(sum_sites.len() * 2)?;
        self.dev.htod_sync_copy_into(&sum_sites.iter().flat_map(|_| [0u32, u32::MAX]).collect::<Vec<_>>(), &mut bad)?;
        let hin = self.head_inputs(rows)?;
        sync(&self.s)?;
        let raw = self.s.stream;
        let r = unsafe { sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture {r:?}");
        let body = (|| -> Result<()> {
            for (i, (site, r)) in sum_sites.iter().enumerate() {
                let f = c.frame(*site, *r)?;
                let key = self.site_key(rows, i);
                c.kern.synth_frame(&self.s, f.ptr, f.spec.floats(), *r, f.spec.hidden, f.spec.mix_used, key, c.rank)?;
                c.reduce(&self.s, *site, *r)?;
                c.kern.synth_check(&self.s, f.ptr, f.spec.floats(), *r, f.spec.hidden, f.spec.mix_used, key, c.world, first, dptr(&bad) + (i * 8) as u64)?;
            }
            self.synth_head_logits(rows, &hin.full, &hin.dfull)?;
            self.head_calls(&hin)
        })();
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        body?;
        ensure!(r2 == sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture {r2:?}");
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r3 = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        ensure!(r3 == sys::CUresult::CUDA_SUCCESS, "cuGraphInstantiate {r3:?}");
        let replays = 3;
        for k in 0..replays {
            self.guard.agree("graph check replay", 0x6C4E_0000 | ((rows as u32) << 4) | k)?;
            let r = unsafe { sys::cuGraphLaunch(exec, raw) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch {r:?}");
            sync(&self.s)?;
        }
        unsafe { sys::cuGraphExecDestroy(exec); }
        super::metrics::graph_replays(replays as u64);
        c.check(&format!("the graph check at {rows} rows"))?;
        let b = self.dev.dtoh_sync_copy(&bad)?;
        let mism: u64 = b.chunks(2).map(|x| x[0] as u64).sum();
        let (gd, gt) = (self.dev.dtoh_sync_copy(&hin.dout)?, self.dev.dtoh_sync_copy(&hin.out)?);
        let head_same = self.head_eager.get(&rows).map_or(false, |(ed, et)| ed.iter().zip(&gd).chain(et.iter().zip(&gt)).all(|(a, b)| a.bits() == b.bits()));
        if mism != 0 { self.failures.push(format!("graph replay at {rows} rows: {mism} reduced floats differ from the single-rank sum")); }
        if !head_same { self.failures.push(format!("graph replay at {rows} rows: the head exchange outputs differ from the eager ones")); }
        fnv(&mut self.out_hash, mism);
        self.lines.push(format!("graph check at {rows} verify rows: {} reduces (synth + reduce + check) and the 2 head exchanges captured as one CUDA graph, \
                                 {replays} replays: {mism} mismatching floats, head outputs {} the eager run{}",
                                sum_sites.len(), if head_same { "bit-identical to" } else { "DIFFERENT from" }, if mism == 0 && head_same { " (GRAPH-CAPTURABLE, BITWISE EQUAL)" } else { "" }));
        Ok(())
    }

    fn sum_sites(&self, rows: usize) -> Vec<(Site, usize)> {
        let c = self.comm;
        c.shape.round(rows, false).sites.into_iter().filter(|(s, r)| !c.shape.frame(*s, *r, c.world).is_gather()).collect()
    }

    fn site_key(&self, rows: usize, i: usize) -> u64 { mix64(self.o.seed ^ ((rows as u64) << 40) ^ ((i as u64) << 8)) }

    pub fn run(mut self) -> Result<Vec<String>> {
        let c = self.comm;
        self.lines.push(format!("{}", c.describe()));
        self.lines.push(format!("rank {} of {}: vocabulary ids {}..{} ({} rows); rows {:?}; {} timed rounds per row count; graph {}",
                                c.rank, c.world, c.vocab().start, c.vocab().end, c.vocab().len(), self.o.rows, self.o.rounds, if self.o.graph { "on" } else { "off" }));
        crate::net::boot_barrier(0xD41_0001)?;
        self.reduces()?;
        self.heads()?;
        if self.o.graph { for &r in &self.o.rows.clone() { self.graph_check(r)?; } }
        c.check("after the correctness passes")?;
        let mut table = Vec::new();
        for &r in &self.o.rows.clone() { table.push(self.timing(r)?); }
        c.check("after the timing rounds")?;
        self.timing_table(&table);
        // rank agreement over every output and every mismatch count
        let mut h = self.out_hash;
        fnv(&mut h, self.failures.len() as u64);
        self.guard.agree("rank agreement over every output", (h ^ (h >> 32)) as u32)
            .map_err(|e| e.context("the ranks' outputs (or mismatch counts) differ"))?;
        self.lines.push(format!("rank agreement: output word {:#010x} equal on all {} rank(s)", (h ^ (h >> 32)) as u32, c.world));
        if !self.failures.is_empty() {
            for f in &self.failures { self.lines.push(format!("FAIL {f}")); }
            bail!("{} check(s) failed (listed above)", self.failures.len());
        }
        self.lines.push("DSV41_TP_SMOKE_OK".into());
        Ok(self.lines)
    }

    /// 1. every sum site of the round at every row count: synth, reduce, recompute-and-compare.
    fn reduces(&mut self) -> Result<()> {
        let c = self.comm;
        let first = super::frame::first_round_bit(c.world);
        for &rows in &self.o.rows {
            let sum_sites = self.sum_sites(rows);
            let mut bad = self.dev.alloc_zeros::<u32>(sum_sites.len() * 2)?;
            self.dev.htod_sync_copy_into(&sum_sites.iter().flat_map(|_| [0u32, u32::MAX]).collect::<Vec<_>>(), &mut bad)?;
            let mut floats = 0usize;
            for (i, (site, r)) in sum_sites.iter().enumerate() {
                let f = c.frame(*site, *r)?;
                let n = f.spec.floats();
                floats += n;
                let key = self.site_key(rows, i);
                c.kern.synth_frame(&self.s, f.ptr, n, *r, f.spec.hidden, f.spec.mix_used, key, c.rank)?;
                c.reduce(&self.s, *site, *r)?;
                c.kern.synth_check(&self.s, f.ptr, n, *r, f.spec.hidden, f.spec.mix_used, key, c.world, first, dptr(&bad) + (i * 8) as u64)?;
            }
            sync(&self.s)?;
            c.check(&format!("the reduce pass at {rows} rows"))?;
            let b = self.dev.dtoh_sync_copy(&bad)?;
            let mut mism = 0u64;
            for (i, (site, r)) in sum_sites.iter().enumerate() {
                fnv(&mut self.out_hash, b[2 * i] as u64);
                if b[2 * i] != 0 {
                    mism += b[2 * i] as u64;
                    self.failures.push(format!("reduce {site:?} at {r} rows: {} of {} floats differ from the single-rank sum (first index {})",
                                               b[2 * i], c.shape.frame(*site, *r, c.world).floats(), b[2 * i + 1]));
                }
            }
            self.lines.push(format!("reduces at {rows} verify rows: {} sites, {:.2} MB per rank, {} mismatching floats against the single-rank sum{}",
                                    sum_sites.len(), floats as f64 * 4.0 / 1e6, mism, if mism == 0 { " (BITWISE EQUAL)" } else { "" }));
        }
        Ok(())
    }

    /// 2. the head exchange, drafter + target, at every row count.
    fn heads(&mut self) -> Result<()> {
        let c = self.comm;
        let (v, w) = (c.shape.vocab, c.world);
        for &rows in &self.o.rows.clone() {
            let lo = |r: usize| 17 + (r * 131) % 1000;
            let hi = |r: usize| v - 1 - (r * 977) % 1000;
            let hin = self.head_inputs(rows)?;
            let (full, dfull, trows, drows, tr, dout, out) = (&hin.full, &hin.dfull, &hin.trows, &hin.drows, &hin.tr, &hin.dout, &hin.out);
            let mk = |dev: &Arc<CudaDevice>| -> Result<CudaSlice<TpOut>> { Ok(dev.htod_sync_copy(&vec![TpOut::default(); rows])?) };
            // the distributed exchange
            let e = [Ev::new()?, Ev::new()?];
            e[0].record(&self.s)?;
            self.head_calls(&hin)?;
            e[1].record(&self.s)?;
            sync(&self.s)?;
            let head_ms = e[0].ms_to(&e[1])?;
            c.check(&format!("the head pass at {rows} rows"))?;
            // the single-GPU reference: the same kernels over every rank's shard, then over one whole-vocabulary shard
            let sim = |world: usize| -> Result<(Vec<TpOut>, Vec<TpOut>)> {
                let mut rec = self.dev.alloc_zeros::<f32>(rows * world * HEAD_REC)?;
                self.dev.memset_zeros(&mut rec)?;   // AGENTS §2.2: alloc_zeros does not zero
                let (rd, rt) = (mk(&self.dev)?, mk(&self.dev)?);
                let shard = |q: usize| crate::dsv41::load::splits::vocab_range(v, world, q);
                let local = |rows_p: u64, logits: &CudaSlice<f32>, rec: u64| -> Result<()> {
                    for q in 0..world {
                        let sh = shard(q);
                        c.kern.head_local(&self.s, rows, LocalArgs { logits: dptr(logits) + (sh.start * 4) as u64, ld: v as i64, rows_p, rec, part: c.part_ptr(),
                                                                    vlen: sh.len() as i32, voff: sh.start as i32, rank: q as i32, world: world as i32,
                                                                    zero_others: 0, ..Default::default() })?;
                    }
                    Ok(())
                };
                local(dptr(drows), dfull, dptr(&rec))?;
                c.kern.head_combine(&self.s, dptr(&rec), rows, world, dptr(drows), dptr(&rd))?;
                local(dptr(trows), full, dptr(&rec))?;
                c.kern.head_combine(&self.s, dptr(&rec), rows, world, dptr(trows), dptr(&rt))?;
                sync(&self.s)?;
                Ok((self.dev.dtoh_sync_copy(&rd)?, self.dev.dtoh_sync_copy(&rt)?))
            };
            let (got_d, got_t) = (self.dev.dtoh_sync_copy(dout)?, self.dev.dtoh_sync_copy(out)?);
            let (ref_d, ref_t) = sim(w)?;
            let (one_d, one_t) = sim(1)?;
            let mut bit_bad = 0usize;
            let mut id_bad = 0usize;
            let mut lse_rel = 0f64;
            for r in 0..rows {
                for (g, rf) in [(&got_d[r], &ref_d[r]), (&got_t[r], &ref_t[r])] {
                    for b in g.bits() { fnv(&mut self.out_hash, b as u64); }
                    if g.bits() != rf.bits() {
                        bit_bad += 1;
                        self.failures.push(format!("head row {r} at {rows} rows: distributed {g:?} != single-GPU {rf:?}"));
                    }
                }
                for (g, one) in [(&got_d[r], &one_d[r]), (&got_t[r], &one_t[r])] {
                    if g.greedy != one.greedy || g.sample != one.sample {
                        id_bad += 1;
                        self.failures.push(format!("head row {r} at {rows} rows: token ids {:?} != one-shard {:?}",
                                                   (g.greedy, g.sample), (one.greedy, one.sample)));
                    }
                    if g.lse.is_finite() { lse_rel = lse_rel.max(((g.lse - one.lse) as f64 / one.lse as f64).abs()); }
                }
                let t = &got_t[r];
                let want = match r % 3 { 0 => Some(lo(r)), 1 => Some(hi(r)), _ => None };
                if let Some(wid) = want {
                    if t.greedy != wid as i32 { self.failures.push(format!("planted row {r} at {rows} rows: greedy {} != {wid} (lowest id of the maximum)", t.greedy)); }
                }
                if tr[r].temp == 0.0 && t.sample != t.greedy {
                    self.failures.push(format!("greedy row {r} at {rows} rows: sample {} inconsistent with argmax {}", t.sample, t.greedy));
                }
            }
            self.head_eager.insert(rows, (got_d.clone(), got_t.clone()));
            self.lines.push(format!("head exchange at {rows} rows: drafter + target in {:.1} us; {} of {} row outputs differ in any bit from the \
                                     single-GPU computation{}; token ids vs one whole-vocabulary shard: {} differ (log-sum last-bit rounding only); \
                                     max |lse rel diff| vs one shard {:.1e}",
                                    head_ms * 1000.0, bit_bad, 2 * rows, if bit_bad == 0 { " (BITWISE EQUAL)" } else { "" }, id_bad, lse_rel));
        }
        Ok(())
    }

    /// One collective of the round, as the timing runs it (a head-style site = its gather all-reduce only).
    fn collective(&self, site: Site, rows: usize) -> Result<()> {
        let c = self.comm;
        let f = c.shape.frame(site, rows, c.world);
        if f.is_gather() {
            ensure!(f.rec == HEAD_REC, "{site:?}: only the head exchange's gather is run here");
            let base = c.head_rec_ptr();
            c.all_reduce_raw(&self.s, base, f.floats())
        } else {
            c.reduce(&self.s, site, rows)
        }
    }

    /// 3. timing at verify width `rows`: eager per-collective events behind a spin gate, then the round as a graph.
    fn timing(&mut self, rows: usize) -> Result<TimingRow> {
        let c = self.comm;
        let round = c.shape.round(rows, false);
        let n = round.len();
        let evs: Vec<Ev> = (0..=n).map(|_| Ev::new()).collect::<Result<_>>()?;
        let mut per_site: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut totals = Vec::new();
        let mut agree_us = Vec::new();
        let warm = 3usize;
        for k in 0..warm + self.o.rounds {
            let mut h = 0xCBF2_9CE4_8422_2325u64;
            fnv(&mut h, rows as u64); fnv(&mut h, k as u64); fnv(&mut h, n as u64);
            let t0 = std::time::Instant::now();
            self.guard.agree("timing round", (h ^ (h >> 32)) as u32)?;
            agree_us.push(t0.elapsed().as_secs_f64() * 1e6);
            c.kern.spin(&self.s, 3000)?;
            // the skew absorber: one tiny collective in lockstep before the first timed one
            c.all_reduce_raw(&self.s, c.head_rec_ptr(), 4)?;
            evs[0].record(&self.s)?;
            for (i, (site, r)) in round.sites.iter().enumerate() {
                self.collective(*site, *r)?;
                evs[i + 1].record(&self.s)?;
            }
            sync(&self.s)?;
            super::metrics::round();
            if k < warm { continue; }
            let mut tot = 0f32;
            for i in 0..n {
                let ms = evs[i].ms_to(&evs[i + 1])?;
                per_site[i].push(ms * 1000.0);
                tot += ms;
            }
            totals.push(tot * 1000.0);
        }
        c.check(&format!("the eager timing at {rows} rows"))?;
        let graph_us = if self.o.graph { Some(self.graph_round(&round.sites, rows)?) } else { None };
        // per class
        let mut by: BTreeMap<Class, (usize, Vec<f32>, usize, &'static str)> = BTreeMap::new();
        for (i, (site, r)) in round.sites.iter().enumerate() {
            let f = c.shape.frame(*site, *r, c.world);
            let e = by.entry(c.shape.class(*site)).or_insert((0, Vec::new(), f.bytes(), c.path_name(f.bytes())));
            e.0 += 1;
            e.1.extend_from_slice(&per_site[i]);
        }
        let t0_sites: Vec<usize> = round.sites.iter().enumerate().filter(|(_, (s, _))| matches!(s, Site::Attn(_) | Site::Ffn(_) | Site::Head)).map(|(i, _)| i).collect();
        let t93: Vec<usize> = round.sites.iter().enumerate().filter(|(_, (s, _))| !matches!(s, Site::HeadResidual)).map(|(i, _)| i).collect();
        let sub = |idx: &[usize]| -> f32 {
            let mut v: Vec<f32> = (0..totals.len()).map(|k| idx.iter().map(|&i| per_site[i][k]).sum()).collect();
            pct(&mut v, 0.5)
        };
        Ok(TimingRow { rows, classes: by.into_iter().map(|(cl, (cnt, mut v, b, p))| (cl, cnt, b, p, pct(&mut v, 0.5), pct(&mut v, 0.99))).collect(),
                       t0_81: sub(&t0_sites), t0_93: sub(&t93), all_p50: pct(&mut totals.clone(), 0.5), all_p99: pct(&mut totals, 0.99),
                       graph_us, agree_p50: pct(&mut agree_us.iter().map(|x| *x as f32).collect(), 0.5) })
    }

    /// The round's collectives captured once into a CUDA graph and replayed back to back: us per round.
    fn graph_round(&mut self, sites: &[(Site, usize)], rows: usize) -> Result<f32> {
        let c = self.comm;
        sync(&self.s)?;
        let raw = self.s.stream;
        let r = unsafe { sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture {r:?}");
        let body = sites.iter().try_for_each(|(s, r)| self.collective(*s, *r));
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        body?;
        ensure!(r2 == sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture {r2:?}");
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r3 = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        ensure!(r3 == sys::CUresult::CUDA_SUCCESS, "cuGraphInstantiate {r3:?}");
        let launch = |s: &CudaStream| -> Result<()> {
            let r = unsafe { sys::cuGraphLaunch(exec, s.stream) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch {r:?}");
            Ok(())
        };
        let (a, b) = (Ev::new()?, Ev::new()?);
        let batch = 10usize;
        let mut per = Vec::new();
        let reps = (self.o.rounds / batch).max(3);
        for k in 0..reps + 1 {
            let mut h = 0x6AF0u64; fnv(&mut h, rows as u64); fnv(&mut h, k as u64);
            self.guard.agree("graph batch", (h ^ (h >> 32)) as u32)?;
            c.kern.spin(&self.s, 1000)?;
            c.all_reduce_raw(&self.s, c.head_rec_ptr(), 4)?;
            a.record(&self.s)?;
            for _ in 0..batch { launch(&self.s)?; }
            b.record(&self.s)?;
            sync(&self.s)?;
            super::metrics::graph_replays(batch as u64);
            if k > 0 { per.push(a.ms_to(&b)? * 1000.0 / batch as f32); }
        }
        unsafe { sys::cuGraphExecDestroy(exec); }
        c.check(&format!("the graph replays at {rows} rows"))?;
        // the captured round's arithmetic is the eager one: re-run the reduce check through one replay of a captured
        // synth + reduce of the largest frame
        Ok(pct(&mut per, 0.5))
    }

    fn timing_table(&mut self, t: &[TimingRow]) {
        let c = self.comm;
        let (a, b) = match c.world { 2 => (22.0, 14.0), 4 => (30.0, 18.0), _ => (40.0, 25.0) };
        self.lines.push(String::new());
        self.lines.push(format!("Per-collective latency, TP{} (GPU events, eager, p50 / p99 us; payload per rank; transport path)", c.world));
        self.lines.push("| verify rows | class | per round | bytes / rank | path | p50 us | p99 us | T0 A / B us |".into());
        self.lines.push("|---|---|---|---|---|---|---|---|".into());
        for row in t {
            for (cl, cnt, bytes, path, p50, p99) in &row.classes {
                let t0 = match cl { Class::Layer | Class::LayerEngram | Class::LayerMainProj | Class::Head =>
                                        format!("{:.0} / {:.0}", a + row.rows as f64, b + row.rows as f64), _ => "-".into() };
                self.lines.push(format!("| {} | {} | {} | {} | {} | {:.1} | {:.1} | {} |", row.rows, cl.name(), cnt, bytes, path, p50, p99, t0));
            }
        }
        self.lines.push(String::new());
        self.lines.push(format!("Per-round comm total, TP{} (us; p50 over rounds; T0 ledger = 81 collectives x ({a:.0} | {b:.0} + rows) us)", c.world));
        self.lines.push("| verify rows | lanes | T0's 81 (80 layer + head) | T0 A / B for 81 | all 93 of T0 | + residual (94) eager p50 / p99 | 94 as one graph | agree guard p50 |".into());
        self.lines.push("|---|---|---|---|---|---|---|---|".into());
        for row in t {
            let ta = 81.0 * (a + row.rows as f64);
            let tb = 81.0 * (b + row.rows as f64);
            self.lines.push(format!("| {} | {} | {:.0} | {:.0} / {:.0} | {:.0} | {:.0} / {:.0} | {} | {:.0} |", row.rows, lanes_of(row.rows), row.t0_81, ta, tb, row.t0_93,
                                    row.all_p50, row.all_p99, row.graph_us.map_or("-".into(), |g| format!("{g:.0}")), row.agree_p50));
        }
    }
}

struct TimingRow {
    rows: usize,
    /// (class, count per round, bytes per rank, path, p50 us, p99 us)
    classes: Vec<(Class, usize, usize, &'static str, f32, f32)>,
    t0_81: f32,
    t0_93: f32,
    all_p50: f32,
    all_p99: f32,
    graph_us: Option<f32>,
    agree_p50: f32,
}

fn pct(v: &mut Vec<f32>, q: f64) -> f32 {
    if v.is_empty() { return f32::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[((v.len() - 1) as f64 * q).round() as usize]
}
