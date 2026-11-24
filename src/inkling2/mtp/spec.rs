//! S6: speculative rounds on the GPU model (one lane per call; the serve worker and the probe drive it).
//!
//! The target state commits by position (K1 / S3 design: `lane.pos += accepted + 1`; the K / V rows and conv-ring
//! rows written for rejected drafts are overwritten by the next forward and never read before that). The drafter
//! commits the same rows from the verify's taps.

use super::gpu::{Draft, DraftReq, MtpBatch, MtpDev, MtpLane, MtpWork, GMAX};
use super::policy::Policy;
use super::super::gpu::dp;
use super::super::model::{GpuModel, Lane, Seg, Work, CHUNK, D};
use anyhow::{ensure, Result};

pub struct Spec {
    pub dev: MtpDev,
    /// one drafter work set per lane (a lane's drafts / step logits live until its verify)
    pub w: Vec<MtpWork>,
    pub lanes: Vec<MtpLane>,
    /// multi-lane draft scratch
    pub batch: MtpBatch,
}

/// One round's outcome.
#[derive(Clone, Debug, Default)]
pub struct Round {
    /// drafts verified, accepted
    pub c: usize,
    pub a: usize,
    /// the emitted tokens (a drafts + the target's next token)
    pub toks: Vec<u32>,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub conf: Vec<f32>,
    pub drafts: Vec<u32>,
}

impl Spec {
    pub fn new(g: &GpuModel, pack: &super::super::pack::Inkling2Pack, ws: &mut Work, lanes: usize, max_ctx: usize) -> Result<Spec> {
        let dev = MtpDev::load(g, pack)?;
        let w = (0..lanes).map(|_| dev.new_work(&g.dev)).collect::<Result<Vec<_>>>()?;
        let ls = (0..lanes).map(|_| dev.new_lane(&g.dev, max_ctx)).collect::<Result<Vec<_>>>()?;
        let mut t = g.dev.alloc_zeros::<u16>(CHUNK * 4 * D)?;
        g.dev.memset_zeros(&mut t)?;
        ws.taps = Some(t);
        ws.tap_visits = dev.tap_visits();
        let batch = dev.new_batch(&g.dev)?;
        Ok(Spec { dev, w, lanes: ls, batch })
    }

    fn taps(ws: &Work) -> u64 { dp(ws.taps.as_ref().unwrap()) }

    /// Prefill `ids` into lane `li` from its position (chunks), committing the drafter context of every chunk; the
    /// logits of the last row land in `ws.logits[0]`.
    pub fn prefill(&mut self, g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, ids: &[u32]) -> Result<()> {
        let mut i = 0;
        while i < ids.len() {
            let m = (ids.len() - i).min(CHUNK);
            let last = i + m == ids.len();
            let p0 = lanes[li].pos;
            let outr = [m - 1];
            {
                let refs: Vec<&Lane> = lanes.iter().collect();
                g.forward(ws, &refs, &[Seg { lane: li, p0, m }], &ids[i..i + m], if last { &outr[..] } else { &[][..] })?;
            }
            self.dev.commit_ctx(g.k, &mut self.w[li], &self.lanes[li], Self::taps(ws), m, p0)?;
            lanes[li].pos += m;
            i += m;
        }
        Ok(())
    }

    /// Draft for lane `li` (anchor `x` at its position).
    pub fn draft(&mut self, g: &GpuModel, lanes: &[Lane], li: usize, x: u32, t: f32, seed: u64) -> Result<Draft> {
        self.dev.draft(g, &mut self.w[li], &self.lanes[li], x, lanes[li].pos, t, seed)
    }

    /// Verify the first `c` drafts of the last `draft` (one lane), accept, commit. `t` <= 0: greedy.
    pub fn verify(&mut self, g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, x: u32, d: &Draft, c: usize, t: f32, seed: u64)
                  -> Result<(usize, Vec<u32>)> {
        ensure!(c <= GMAX);
        let p0 = lanes[li].pos;
        let mut ids = vec![x];
        ids.extend_from_slice(&d.toks[..c]);
        // the served verify entry (S7 `verify_lanes`: every row bit-identical to its 1-row decode)
        g.verify_lanes(ws, lanes, &[(li, &ids[..])], false)?;
        let (a, nt) = self.dev.accept(g, &mut self.w[li], dp(&ws.logits), c, t, seed)?;
        self.dev.commit_ctx(g.k, &mut self.w[li], &self.lanes[li], Self::taps(ws), a + 1, p0)?;
        lanes[li].pos += a + 1;
        let mut out = d.toks[..a].to_vec();
        out.push(nt);
        Ok((a, out))
    }

    /// One full round (draft, policy count, verify) for lane `li`; `cap` = most drafts (0 = a plain step).
    #[allow(clippy::too_many_arguments)]
    pub fn round(&mut self, g: &GpuModel, ws: &mut Work, lanes: &mut [Lane], li: usize, x: u32, t: f32, seed: u64, pol: &mut Policy,
                 cap: usize) -> Result<Round> {
        let t0 = std::time::Instant::now();
        if cap == 0 {
            let empty = Draft { toks: vec![], conf: vec![] };
            // the drafter's tok[0] must hold the anchor for nothing here; verify reads only w.tok[1..=c] (c = 0)
            let (a, toks) = self.verify(g, ws, lanes, li, x, &empty, 0, t, seed)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            pol.learn_plain(ms);
            return Ok(Round { c: 0, a, toks, draft_ms: 0.0, verify_ms: ms, conf: vec![], drafts: vec![] });
        }
        let d = {
            let p = &*pol;
            let req = [DraftReq { lane: li, anchor: x, p: lanes[li].pos, t, seed, cap }];
            self.dev.draft_multi(g, &mut self.batch, &mut self.w, &self.lanes, &req, &mut |_, j, cf| p.more(cf, j, cap))?.remove(0)
        };
        let t1 = std::time::Instant::now();
        let c = pol.choose(&d.conf, cap.min(d.conf.len()));
        let (a, toks) = self.verify(g, ws, lanes, li, x, &d, c, t, seed)?;
        let draft_ms = (t1 - t0).as_secs_f64() * 1e3;
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
        pol.learn(&d.conf, c, a, verify_ms, draft_ms);
        Ok(Round { c, a, toks, draft_ms, verify_ms, conf: d.conf, drafts: d.toks })
    }
}
