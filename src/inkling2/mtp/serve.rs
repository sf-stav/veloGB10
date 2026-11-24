//! S6 / S8 / PV1: speculative decoding with the bundled drafter inside the serve worker (`model::serve::worker`, which
//! keeps its queue, interleaved prefill and prefix cache). Every decode iteration is one round over the decoding lanes:
//!   1. each lane that drafts this round runs the drafter (8 block rows) and picks its count c_l from the drafter's
//!      confidences (exact look-ahead stop, `policy.rs`), capped so the verify rows fit `GpuModel::verify_lanes`
//!      (<= 48 rows, <= 16 a lane: 7 drafts a lane at 1..4 lanes), by the served context and the request's budget;
//!   2. ONE `verify_lanes` call (a CUDA graph per row count) over every lane's [next, d_1..d_c] (a lane that does not
//!      draft verifies just its next token = a plain decode step; every row is bit-identical to its 1-row decode);
//!   3. per lane the exact accept on the device, the drafter commit of the accepted rows (from the verify's taps),
//!      emission.
//! Whether an iteration speculates at all is decided per lane count from the realised throughput of past
//! speculative vs plain iterations (`Ctl`, with periodic probes); `--mtp on` always speculates.
//! Requests with top_p < 1 or top_k > 0 draft too (S8): their accept runs on the host against the truncated target rows
//! (`policy::trunc_probs`, the law `cpu_sample` draws from) with the same exact rule; counted (`spec_trunc_rounds`).
//! Prefix cache: the drafter's lane state is saved / restored with the target's at prompt ends (`MtpSnap`).

use super::super::model::serve::{cpu_sample, emit_tok, mix, needs_cpu, Active};
use super::super::model::{GpuModel, Lane, Work, D};
use super::super::serve::metrics as mt;
use super::gpu::{DraftReq, LMAX};
use super::policy::Policy;
use super::spec::Spec;
use super::super::gpu::dp;

#[derive(Clone, Copy, Debug)]
pub struct SpecOpts {
    /// `--mtp on`: draft every round (no back-off to plain decode when speculation loses)
    pub force: bool,
    /// most drafts per round for a lane alone (`--inkling2-mtp-max-drafts`, 1..7)
    pub max_drafts: usize,
}

/// Most verify rows of one round (`GpuModel::verify_lanes`: <= 48 rows, <= 16 per lane; every row bit-identical to
/// its 1-row decode).
const ROWS: usize = 48;
const LANE_ROWS: usize = 16;

/// Whether an iteration speculates, per number of active lanes: the realised throughput (tokens emitted by all lanes
/// per ms of the iteration) of speculative vs plain iterations, each a ratio of decayed sums (tokens / ms, so a noisy
/// per-round count does not bias it), with a probe of the other mode every PROBE iterations. Decided from past
/// iterations only (exact for sampled requests, `policy.rs`).
struct Ctl { tok: [[f64; 2]; 17], ms: [[f64; 2]; 17], n: [[u32; 2]; 17], since: [u32; 17] }
const PROBE: u32 = 48;
const KEEP: f64 = 0.97;
impl Ctl {
    fn new() -> Ctl { Ctl { tok: [[0.0; 2]; 17], ms: [[0.0; 2]; 17], n: [[0; 2]; 17], since: [0; 17] } }
    fn rate(&self, l: usize, k: usize) -> f64 { self.tok[l][k] / self.ms[l][k].max(1e-9) }
    /// true = speculate this iteration
    fn decide(&mut self, lanes: usize) -> bool {
        let l = lanes.min(16);
        // warm-up: 8 speculative and 4 plain iterations at this lane count
        if self.n[l][1] < 8 { return true; }
        if self.n[l][0] < 4 { return false; }
        let best = self.rate(l, 1) >= self.rate(l, 0);
        self.since[l] += 1;
        if self.since[l] >= PROBE { self.since[l] = 0; return !best; }
        best
    }
    fn learn(&mut self, lanes: usize, spec: bool, tokens: usize, ms: f64) {
        let (l, k) = (lanes.min(16), spec as usize);
        self.tok[l][k] = self.tok[l][k] * KEEP + tokens as f64;
        self.ms[l][k] = self.ms[l][k] * KEEP + ms;
        self.n[l][k] += 1;
    }
}

/// The speculative side of the serve worker (`model::serve::worker`): the drafter, its per-lane state, the per-lane
/// count policies and the speculate-or-plain controller. The worker keeps its queue, lanes, interleaved prefill and
/// prefix cache; it calls `commit_prefill` after every prefill chunk, `admit` when a lane takes a request, the snapshot
/// functions at prompt ends / prefix hits, and `step` for each decode iteration.
pub struct SpecState { sp: Spec, o: SpecOpts, pol: Vec<Policy>, ctl: Ctl }

impl SpecState {
    pub fn new(g: &GpuModel, pack: &super::super::pack::Inkling2Pack, ws: &mut Work, lanes: usize, max_ctx: usize, o: SpecOpts)
               -> anyhow::Result<SpecState> {
        let sp = Spec::new(g, pack, ws, lanes, max_ctx)?;
        Ok(SpecState { sp, o, pol: (0..lanes).map(|_| Policy::new(31.0, 2.5, 12.0, o.max_drafts)).collect(), ctl: Ctl::new() })
    }
    /// A lane takes a new request: its count policy restarts from the lane's learned costs.
    pub fn admit(&mut self, li: usize) {
        let p = &self.pol[li];
        self.pol[li] = Policy::new(p.verify_ms[1].max(1.0), 2.5, p.draft_ms, self.o.max_drafts);
    }
    /// The drafter context of the `m` prefill rows just run for lane `li` at `p0..` (taps of that forward).
    pub fn commit_prefill(&mut self, g: &GpuModel, ws: &Work, li: usize, m: usize, p0: usize) -> anyhow::Result<()> {
        if g.tp.is_some() {
            g.mirror_op(super::super::tp::Op::MtpCommit { lane: li, tap_off: 0, rows: m, p0 })?;
        }
        self.sp.dev.commit_ctx(g.k, &mut self.sp.w[li], &self.sp.lanes[li], dp(ws.taps.as_ref().unwrap()), m, p0)
    }
    pub fn new_snap(&self, g: &GpuModel, li: usize) -> anyhow::Result<super::gpu::MtpSnap> { self.sp.dev.new_snap(&g.dev, &self.sp.lanes[li]) }
    pub fn snap_save(&self, _g: &GpuModel, li: usize, s: &mut super::gpu::MtpSnap) -> anyhow::Result<()> { self.sp.dev.snap_copy(&self.sp.lanes[li], s, true) }
    pub fn snap_restore(&self, _g: &GpuModel, li: usize, s: &super::gpu::MtpSnap) -> anyhow::Result<()> { self.sp.dev.snap_copy(&self.sp.lanes[li], s, false) }
    /// One decode iteration over the decoding lanes `rows`: a speculative round, or (controller) a plain step through
    /// the same verify entry. Emits the tokens and finishes requests.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(&mut self, g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], act: &mut [Option<Active>], rows: &[usize], v: usize, eos: &[u32])
                -> anyhow::Result<()> {
        let spec = self.o.force || self.ctl.decide(rows.len());
        let t0 = std::time::Instant::now();
        let tokens = round(g, ws, lanes, &mut self.sp, &mut self.pol, act, rows, self.o, spec, v, eos)?;
        self.ctl.learn(rows.len(), spec, tokens, t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn round(g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], sp: &mut Spec, pol: &mut [Policy], act: &mut [Option<Active>], rows: &[usize],
         o: SpecOpts, spec: bool, v: usize, eos: &[u32]) -> anyhow::Result<usize> {
    let t0 = std::time::Instant::now();
    let cap = (ROWS / rows.len()).min(LANE_ROWS).saturating_sub(1).min(o.max_drafts);
    // 1. drafts
    let mut plan: Vec<(usize, Vec<u32>, Vec<f32>)> = vec![(0, vec![], vec![]); rows.len()]; // (c, drafts, conf)
    let mut reqs: Vec<(usize, DraftReq)> = Vec::new(); // (index in rows, request)
    for (k, &i) in rows.iter().enumerate() {
        let a = act[i].as_ref().unwrap();
        // S8: top_p / top_k requests draft too (exact: their accept runs on the host against the truncated rows)
        // room left in the served context and in the request's budget (a draft past either is never verified)
        let room = lanes[i].max_ctx.saturating_sub(lanes[i].pos + 1).min(a.req.max_new.saturating_sub(a.gen + 1));
        let cap = cap.min(room);
        if !(spec && cap > 0) { continue; }
        let seed = mix(a.req.seed ^ mix(a.step ^ 0x5bec));
        reqs.push((k, DraftReq { lane: i, anchor: a.next, p: lanes[i].pos, t: a.req.temperature, seed, cap }));
    }
    for grp in reqs.chunks(LMAX) {
        let rq: Vec<DraftReq> = grp.iter().map(|x| x.1).collect();
        if g.tp.is_some() {
            g.mirror_op(super::super::tp::Op::DraftMulti {
                reqs: rq.iter().map(|r| (r.lane, r.anchor, r.p, r.t, r.seed, r.cap)).collect(),
            })?;
        }
        let ds = {
            let pr = &*pol;
            sp.dev.draft_multi(g, &mut sp.batch, &mut sp.w, &sp.lanes, &rq, &mut |l, j, cf| pr[rq[l].lane].more(cf, j, rq[l].cap))?
        };
        for ((k, r), d) in grp.iter().zip(ds) {
            let c = pol[r.lane].choose(&d.conf, r.cap.min(d.conf.len()));
            plan[*k] = (c, d.toks, d.conf);
        }
    }
    let t1 = std::time::Instant::now();
    // 2. one verify forward
    let mut toks_l: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
    for (k, &i) in rows.iter().enumerate() {
        let c = plan[k].0;
        let mut t = vec![act[i].as_ref().unwrap().next];
        t.extend_from_slice(&plan[k].1[..c]);
        toks_l.push(t);
    }
    let segs: Vec<(usize, &[u32])> = rows.iter().zip(&toks_l).map(|(&i, t)| (i, &t[..])).collect();
    let nrows: usize = toks_l.iter().map(|t| t.len()).sum();
    g.verify_lanes(ws, lanes, &segs, true)?;
    let ids_len = nrows;
    mt::inc(&mt::FWD_DECODE_STEPS);
    mt::add(&mt::FWD_DECODE_ROWS, ids_len as u64);
    // 3. accept, commit, emit
    let taps = dp(ws.taps.as_ref().unwrap());
    let mut r0 = 0usize;
    let mut results: Vec<(usize, Vec<u32>)> = Vec::new();
    // Drop 1: GPU-sampled lanes (greedy and T > 0) run vocab-parallel accept with ZERO full-vocab gather!
    // Full-vocab gather is ONLY performed when a CPU-sampled (top_p / top_k) lane verified drafts.
    let need_host_trunc = rows.iter().enumerate().any(|(k, &i)| plan[k].0 > 0 && needs_cpu(&act[i].as_ref().unwrap().req));
    let host_rows: Option<Vec<f32>> = if need_host_trunc {
        let lfull = g.logits_full(ws, ids_len)?;
        g.k.sync()?;
        let mut h = vec![0f32; ids_len * v];
        let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut std::ffi::c_void, lfull, ids_len * v * 4) };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "verify rows to host: {r:?}");
        Some(h)
    } else { None };
    let vn = sp.dev.n;
    for (k, &i) in rows.iter().enumerate() {
        let c = plan[k].0;
        let a = act[i].as_ref().unwrap();
        let p0 = lanes[i].pos;
        let (acc, toks) = if needs_cpu(&a.req) && c > 0 {
            // S8: exact accept against the TRUNCATED target (the law cpu_sample draws from)
            let l = host_rows.as_ref().unwrap();
            let (t, tp, tk) = (a.req.temperature, a.req.top_p, a.req.top_k);
            let mut st = vec![0f32; c * v];
            if g.tp.is_some() {
                g.mirror_op(super::super::tp::Op::MtpGatherStep { lane: i, c })?;
                let sfull = g.gather_vocab_rows(dp(&sp.w[i].step), c)?;
                g.k.sync()?;
                let r = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(st.as_mut_ptr() as *mut std::ffi::c_void, sfull, c * v * 4) };
                anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "draft step rows to host: {r:?}");
            } else {
                g.dev.dtoh_sync_copy_into(&sp.w[i].step.slice(0..c * v), &mut st)?;
            }
            let seed = mix(a.req.seed ^ mix(a.step ^ 0x7c0b));
            let uni = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            let u: Vec<f64> = (0..c).map(|j| uni(seed ^ (j as u64 + 1))).collect();
            let d: Vec<usize> = plan[k].1[..c].iter().map(|&x| x as usize).collect();
            let (acc, nt) = super::policy::accept_lazy(c, &d, &mut |j| super::policy::trunc_probs(&l[(r0 + j) * v..(r0 + j + 1) * v], t, tp, tk),
                                                       &mut |j| super::policy::softmax_t(&st[j * v..(j + 1) * v], t), &u, uni(seed ^ 0xff));
            mt::inc(&mt::SPEC_TRUNC_ROUNDS);
            mt::add(&mt::FWD_CPU_SAMPLED, (acc + 1) as u64);
            let mut tt = plan[k].1[..acc].to_vec();
            tt.push(nt as u32);
            if g.tp.is_some() {
                g.mirror_op(super::super::tp::Op::MtpCommit { lane: i, tap_off: r0, rows: acc + 1, p0 })?;
            }
            sp.dev.commit_ctx(g.k, &mut sp.w[i], &sp.lanes[i], taps + (r0 * 4 * D * 2) as u64, acc + 1, p0)?;
            (acc, tt)
        } else if needs_cpu(&a.req) {
            mt::inc(&mt::FWD_CPU_SAMPLED);
            let l = g.logits_host(ws, r0 + 1)?;
            let seed = mix(a.req.seed ^ mix(a.step));
            let nt = cpu_sample(&l[r0 * v..(r0 + 1) * v], a.req.temperature, a.req.top_p, a.req.top_k, seed);
            if g.tp.is_some() {
                g.mirror_op(super::super::tp::Op::MtpCommit { lane: i, tap_off: r0, rows: 1, p0 })?;
            }
            sp.dev.commit_ctx(g.k, &mut sp.w[i], &sp.lanes[i], taps + (r0 * 4 * D * 2) as u64, 1, p0)?;
            (0, vec![nt])
        } else {
            let seed = mix(a.req.seed ^ mix(a.step ^ 0xacc));
            if g.tp.is_some() {
                g.mirror_op(super::super::tp::Op::MtpAccept { lane: i, r0, c, t: a.req.temperature, seed, p0 })?;
            }
            let (acc, nt) = sp.dev.accept_tp(g, &mut sp.w[i], dp(&ws.logits) + (r0 * vn * 4) as u64, c, a.req.temperature, seed)?;
            sp.dev.commit_ctx(g.k, &mut sp.w[i], &sp.lanes[i], taps + (r0 * 4 * D * 2) as u64, acc + 1, p0)?;
            mt::add(&mt::FWD_GPU_SAMPLED, (acc + 1) as u64);
            let mut t = plan[k].1[..acc].to_vec();
            t.push(nt);
            (acc, t)
        };
        lanes[i].pos += acc + 1;
        r0 += c + 1;
        results.push((acc, toks));
    }
    let draft_ms = (t1 - t0).as_secs_f64() * 1e3;
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let tokens: usize = results.iter().map(|r| r.1.len()).sum();
    for (k, &i) in rows.iter().enumerate() {
        let (c, acc) = (plan[k].0, results[k].0);
        if plan[k].1.is_empty() {
            pol[i].learn_plain(verify_ms);
            mt::inc(&mt::SPEC_PLAIN_ROWS);
        } else {
            pol[i].learn(&plan[k].2, c, acc, verify_ms, draft_ms / rows.len() as f64);
            mt::inc(&mt::SPEC_ROUNDS);
            mt::add(&mt::SPEC_DRAFTED, c as u64);
            mt::add(&mt::SPEC_ACCEPTED, acc as u64);
            if acc < mt::SPEC_ACCEPT_HIST.len() { mt::inc(&mt::SPEC_ACCEPT_HIST[acc]); }
        }
        let mut a = act[i].take().unwrap();
        let mut alive = true;
        for &t in &results[k].1 {
            a.next = t;
            a.step += 1;
            if !emit_tok(&mut a, eos, lanes[i].max_ctx, lanes[i].pos) { alive = false; break; }
        }
        if alive { act[i] = Some(a); }
    }
    Ok(tokens)
}
