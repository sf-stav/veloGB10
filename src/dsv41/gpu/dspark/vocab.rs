//! The vocab passes of the DSpark round on the GPU (band 91) and the round's device state.
//!
//! A pass = `d41_ds_vpart` over this rank's vocab shard (blocks of [`CHUNK`] ids; the block count depends on the
//! shard only, never on the row count, so a row's record is the same in a 1-row and a 24-row call: batch
//! invariance) -> `d41_ds_vrank` (blocks merged in order: this rank's record per row) -> the exchange (all-gather
//! of the records, [`super::xchg`]) -> `d41_ds_vmerge` (ranks merged in order) with the call's epilogue:
//! * PICK (drafter rows): the drafted id (greedy argmax / keyed Gumbel draw from the processed distribution), its
//!   processed log-probability and the row's log-sum-exp, kept for the verify;
//! * STATS (verify rows): the merged records (the target's log-sum-exp, argmax, the bonus draw, the draft's value);
//! * FINISH (after the residual pass): the token after a rejection and the lane's round result.
//! `d41_ds_accept` sits between the verify's STATS and the residual pass. Every launch reads its per-round values
//! from device memory ([`Round::lanes`]): a whole round is capturable in one CUDA graph per shape.

use super::xchg::VocabExchange;
use super::{AcceptArgs, Lane, VRec, VmergeArgs, VpartArgs, VrankArgs, BLOCK, IDS_ROWS, OUT_TOK};
use crate::dsv41::fwd::dev::{Buf, Gpu};
use anyhow::{ensure, Result};

/// Leg PH1: verifies of 0 drafts that skipped the residual pass and its exchange (`Round::verify_k`).
pub static PLAIN_FINISH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Vocab ids per block of a pass (fixed: the reduction order of a row never depends on the call's row count).
pub const CHUNK: usize = 2048;
pub const REC: usize = std::mem::size_of::<VRec>();

/// This rank's slice of the vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct Shard { pub v0: usize, pub vlocal: usize }
impl Shard {
    pub fn nblk(&self) -> usize { self.vlocal.div_ceil(CHUNK).max(1) }
}

/// A logits operand: row (l, j) at `ptr + (l * lane_stride + j * row_stride) * 4` (f32, this rank's shard columns).
#[derive(Clone, Copy, Debug, Default)]
pub struct Rows { pub ptr: u64, pub lane_stride: usize, pub row_stride: usize }

/// `d41_ds_mstep_l*`'s CTAs per vocab block (`D41_DS_MS_SUB`: 256 columns each).
pub const MS_SUB: usize = 8;

/// The operands of one Markov chain step ([`Round::mstep`]): the Markov embedding [vocab][256] and this rank's
/// Markov head rows [vlocal][256] (bf16 / f16: `mh_code` 0 / 1), the chain's input ids table ([6][lanes]), the confidence input (row
/// (l, step) columns d .. d + 256 at `cin + (l * cin_ls + step * ld + d) * 2`) and the logits rows.
#[derive(Clone, Copy, Debug)]
pub struct MstepIn { pub me: u64, pub mh: u64, pub mh_code: i32, pub table: u64, pub cin: u64, pub cin_ls: usize, pub ld: usize, pub d: usize, pub logits: Rows }

/// An optional per-row cut (kept iff a > cut_a || (a == cut_a && id <= cut_id)): device arrays [lanes][rpl].
#[derive(Clone, Copy, Debug, Default)]
pub struct Cut { pub a: u64, pub id: u64, pub rpl: usize }

/// The round's device state for up to `cap` lanes.
pub struct Round {
    pub cap: usize,
    pub rpl_max: usize,
    pub rows_max: usize,
    pub nblk_max: usize,
    pub ranks: usize,
    /// [cap] `Lane`
    pub lanes: Buf,
    /// [6][cap] i32: row 0 = the block token, rows 1..=5 = the drafts (the drafter's picks)
    pub ids: Buf,
    /// [6][cap] i32: the Markov chain's inputs when forced (the gate's forced chain); otherwise unused
    pub fids: Buf,
    /// [cap][5] f32: the drafts' processed log-probabilities / the draft rows' log-sum-exp
    pub dlp: Buf,
    pub dlse: Buf,
    /// [cap][8] i32: the tokens a round emits
    pub out_tok: Buf,
    /// [cap][rpl_max] merged target-row records (the verify's STATS)
    pub tres: Buf,
    /// [cap * rpl_max] merged records of any STATS call (control draws, probes)
    pub sres: Buf,
    part: Buf,
    send: Buf,
    recv: Buf,
    /// [min(cap, 4)][nblk_max * CHUNK] f32: `d41_ds_mstep`'s Gumbel scores
    gsc: Buf,
    /// leg P7.2: model EOS / stop token ids (-1 = unused), masked on rows `0..lane.suppress_eos`
    pub stop_ids: [i32; 4],
}

impl Round {
    /// Device bytes of the round's buffers (leg MEM1's ledger).
    pub fn dev_bytes(&self) -> u64 {
        [&self.lanes, &self.ids, &self.fids, &self.dlp, &self.dlse, &self.out_tok, &self.tres, &self.sres, &self.part, &self.send, &self.recv, &self.gsc]
            .iter().map(|b| b.bytes as u64).sum()
    }

    /// Buffers for `cap` lanes, `rpl_max` rows per lane (verify: k + 1), a shard of `nblk_max` blocks and `ranks` ranks.
    pub fn new(g: &Gpu, cap: usize, rpl_max: usize, nblk_max: usize, ranks: usize) -> Result<Round> {
        ensure!(cap > 0 && rpl_max >= 1 && rpl_max <= OUT_TOK, "round: {cap} lanes x {rpl_max} rows");
        let rows_max = cap * rpl_max.max(BLOCK);
        let ids_rows = IDS_ROWS.max(rpl_max);
        Ok(Round {
            cap, rpl_max, rows_max, nblk_max, ranks,
            lanes: g.buf(cap * std::mem::size_of::<Lane>())?,
            ids: g.buf(ids_rows * cap * 4)?,
            fids: g.buf(IDS_ROWS * cap * 4)?,
            dlp: g.buf(cap * BLOCK * 4)?,
            dlse: g.buf(cap * BLOCK * 4)?,
            out_tok: g.buf(cap * OUT_TOK * 4)?,
            tres: g.buf(cap * rpl_max * REC)?,
            sres: g.buf(rows_max * REC)?,
            part: g.buf(rows_max * nblk_max * REC)?,
            send: g.buf(rows_max * REC)?,
            recv: g.buf(ranks * rows_max * REC)?,
            gsc: g.buf(cap.min(4) * nblk_max * CHUNK * 4)?,
            stop_ids: [-1; 4],
        })
    }

    /// Lane states to the device (between rounds, outside any graph, on the compute stream).
    pub fn set_lanes(&self, g: &Gpu, lanes: &[Lane]) -> Result<()> {
        ensure!(lanes.len() <= self.cap, "{} lanes > {}", lanes.len(), self.cap);
        g.h2d_async(self.lanes.ptr(), as_bytes(lanes))
    }
    pub fn get_lanes(&self, g: &Gpu, n: usize) -> Result<Vec<Lane>> {
        let raw = g.down::<u8>(self.lanes.ptr(), n * std::mem::size_of::<Lane>())?;
        Ok(raw.chunks_exact(std::mem::size_of::<Lane>()).map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const Lane) }).collect())
    }
    pub fn out_tokens(&self, g: &Gpu, n: usize) -> Result<Vec<[i32; OUT_TOK]>> {
        let v = g.down::<i32>(self.out_tok.ptr(), n * OUT_TOK)?;
        Ok(v.chunks_exact(OUT_TOK).map(|c| { let mut a = [0; OUT_TOK]; a.copy_from_slice(c); a }).collect())
    }
    /// Download both `lanes` and `out_tokens` with a single stream sync.
    pub fn get_lanes_and_out(&self, g: &Gpu, n: usize) -> Result<(Vec<Lane>, Vec<[i32; OUT_TOK]>)> {
        let (raw, v) = g.down2::<u8, i32>(self.lanes.ptr(), n * std::mem::size_of::<Lane>(), self.out_tok.ptr(), n * OUT_TOK)?;
        let lanes = raw.chunks_exact(std::mem::size_of::<Lane>()).map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const Lane) }).collect();
        let out = v.chunks_exact(OUT_TOK).map(|c| { let mut a = [0; OUT_TOK]; a.copy_from_slice(c); a }).collect();
        Ok((lanes, out))
    }
    /// The ids table ([6][n] i32, row-major by table row).
    pub fn ids_table(&self, g: &Gpu) -> Result<Vec<i32>> { g.down::<i32>(self.ids.ptr(), IDS_ROWS * self.cap) }
    pub fn recs(&self, g: &Gpu, ptr: u64, n: usize) -> Result<Vec<VRec>> {
        let raw = g.down::<u8>(ptr, n * REC)?;
        Ok(raw.chunks_exact(REC).map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const VRec) }).collect())
    }

    fn base_args(&self, sh: Shard, mode: i32, nlanes: usize, rpl: usize, j0: usize, t: Rows) -> VpartArgs {
        VpartArgs {
            logits: t.ptr, lane_stride: t.lane_stride as i64, row_stride: t.row_stride as i64, lanes: self.lanes.ptr(),
            nlanes: nlanes as i32, rpl: rpl as i32, j0: j0 as i32, vlocal: sh.vlocal as i32, v0: sh.v0 as i32, chunk: CHUNK as i32, mode,
            rpl_t: self.rpl_max as i32, alpha: super::alpha(), part: self.part.ptr(), stop_ids: self.stop_ids, ..Default::default()
        }
    }

    /// One vocab pass: vpart over `rows` rows -> this rank's records -> the exchange. Returns the gathered records.
    fn pass(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, a: &VpartArgs, rows: usize) -> Result<u64> {
        ensure!(rows <= self.rows_max && sh.nblk() <= self.nblk_max && x.ranks() == self.ranks, "vocab pass: {rows} rows / {} blocks / {} ranks over the round's size", sh.nblk(), x.ranks());
        let nblk = sh.nblk();
        g.launch1("d41_ds_vpart", (nblk as u32, rows as u32, 1), (256, 1, 1), 0, a)?;
        let local = if x.ranks() == 1 { self.recv.ptr() } else { self.send.ptr() };
        g.launch1("d41_ds_vrank", ((rows as u32).div_ceil(32), 1, 1), (32, 1, 1), 0, &VrankArgs { part: self.part.ptr(), out: local, rows: rows as i32, nblk: nblk as i32 })?;
        x.all_gather(g, local, rows * REC, self.recv.ptr())?;
        Ok(self.recv.ptr())
    }

    fn merge(&self, g: &Gpu, m: &VmergeArgs) -> Result<()> {
        g.launch1("d41_ds_vmerge", ((m.rows as u32).div_ceil(32), 1, 1), (32, 1, 1), 0, m)
    }

    /// Drafter rows j0 .. j0 + rpl of `nlanes` lanes: draws (greedy argmax / sampled in the draft namespace),
    /// written to the ids table row j + 1, with the processed log-probability and the row's log-sum-exp.
    pub fn draft_pick(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, rows: Rows, nlanes: usize, j0: usize, rpl: usize, cut: Option<Cut>) -> Result<()> {
        ensure!(j0 + rpl <= BLOCK, "draft rows {j0}..{}", j0 + rpl);
        let mut a = self.base_args(sh, super::MODE_DRAFT, nlanes, rpl, j0, rows);
        if let Some(c) = cut { a.cut_a = c.a; a.cut_id = c.id; a.cut_rpl = c.rpl as i32; }
        let n = nlanes * rpl;
        let recs = self.pass(g, x, sh, &a, n)?;
        self.merge(g, &VmergeArgs { recs, res: 0, lanes: self.lanes.ptr(), ids: self.ids.ptr(), dlp: self.dlp.ptr(), dlse: self.dlse.ptr(), nranks: x.ranks() as i32,
                                    rows: n as i32, nlanes: nlanes as i32, rpl: rpl as i32, j0: j0 as i32, kind: super::KIND_PICK, rpl_t: self.rpl_max as i32, ..Default::default() })
    }

    /// One Markov chain step of `nlanes` lanes (leg PK5): `d41_ds_mstep_l{1,2,4}` (the embedding rows of the chain
    /// inputs, the Markov bias added to the logits rows `step`, every id's Gumbel score), `d41_ds_vscan` (this rank's
    /// block records), then at TP1 `d41_ds_vfin` (block merge + pick), otherwise `d41_ds_vrank`, the exchange and
    /// `d41_ds_vmerge`'s PICK. The same bits as [`Round::draft_pick`] after a gather and a plain GEMV.
    pub fn mstep(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, m: &MstepIn, nlanes: usize, step: usize) -> Result<()> {
        ensure!(nlanes >= 1 && nlanes <= 4 && nlanes <= self.cap && step < BLOCK, "mstep: {nlanes} lanes, step {step}");
        let nblk = sh.nblk();
        ensure!(nblk <= self.nblk_max && x.ranks() == self.ranks, "mstep: {nblk} blocks / {} ranks over the round's size", x.ranks());
        let a = super::MstepArgs {
            me: m.me, mh: m.mh, table: m.table, cin: m.cin, logits: m.logits.ptr, lane_stride: m.logits.lane_stride as i64, row_stride: m.logits.row_stride as i64,
            cin_ls: m.cin_ls as i64, lanes: self.lanes.ptr(), gsc: self.gsc.ptr(), nlanes: nlanes as i32, step: step as i32, ld: m.ld as i32, d: m.d as i32,
            vlocal: sh.vlocal as i32, v0: sh.v0 as i32, mh_code: m.mh_code, pad: 0,
        };
        // the 2-lane instance also serves one lane (the 1-lane one runs at higher occupancy and streams a DRAM-sized
        // shard slower: TP1 / TP2, measured with tools/dsv41_pk5/chain_bench.cu)
        let name = if nlanes <= 2 { "d41_ds_mstep_l2" } else { "d41_ds_mstep_l4" };
        g.launch1(name, ((nblk * MS_SUB) as u32, 1, 1), (256, 1, 1), 0, &a)?;
        g.launch1("d41_ds_vscan", (nblk as u32, nlanes as u32, 1), (256, 1, 1), 0,
                  &super::VscanArgs { logits: m.logits.ptr, lane_stride: m.logits.lane_stride as i64, row_stride: m.logits.row_stride as i64, gsc: self.gsc.ptr(),
                                      lanes: self.lanes.ptr(), part: self.part.ptr(), step: step as i32, vlocal: sh.vlocal as i32, v0: sh.v0 as i32, pad: 0 })?;
        if x.ranks() == 1 {
            ensure!(nlanes * nblk <= 4 * 64, "mstep: {nlanes} lanes x {nblk} blocks > d41_ds_vfin's 256 records");
            return g.launch1("d41_ds_vfin", (1, 1, 1), (256, 1, 1), 0,
                             &super::VfinArgs { part: self.part.ptr(), recs: self.recv.ptr(), lanes: self.lanes.ptr(), ids: self.ids.ptr(), dlp: self.dlp.ptr(),
                                                dlse: self.dlse.ptr(), nlanes: nlanes as i32, nblk: nblk as i32, step: step as i32, pick: 1 });
        }
        g.launch1("d41_ds_vrank", ((nlanes as u32).div_ceil(32), 1, 1), (32, 1, 1), 0, &VrankArgs { part: self.part.ptr(), out: self.send.ptr(), rows: nlanes as i32, nblk: nblk as i32 })?;
        x.all_gather(g, self.send.ptr(), nlanes * REC, self.recv.ptr())?;
        self.merge(g, &VmergeArgs { recs: self.recv.ptr(), res: 0, lanes: self.lanes.ptr(), ids: self.ids.ptr(), dlp: self.dlp.ptr(), dlse: self.dlse.ptr(),
                                    nranks: x.ranks() as i32, rows: nlanes as i32, nlanes: nlanes as i32, rpl: 1, j0: step as i32, kind: super::KIND_PICK,
                                    rpl_t: self.rpl_max as i32, ..Default::default() })
    }

    /// Statistics of rows 0 .. rpl of `nlanes` lanes (verify mode: probes = the drafts, the bonus draw at stream 0)
    /// merged into `sres` ([nlanes * rpl] records). Control draws and probes use this.
    pub fn stats(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, rows: Rows, nlanes: usize, rpl: usize, cut: Option<Cut>) -> Result<u64> {
        let mut a = self.base_args(sh, super::MODE_VERIFY, nlanes, rpl, 0, rows);
        a.rpl_t = rpl as i32;
        a.ids = self.ids.ptr();
        if let Some(c) = cut { a.cut_a = c.a; a.cut_id = c.id; a.cut_rpl = c.rpl as i32; }
        let n = nlanes * rpl;
        let recs = self.pass(g, x, sh, &a, n)?;
        self.merge(g, &VmergeArgs { recs, res: self.sres.ptr(), lanes: self.lanes.ptr(), nranks: x.ranks() as i32, rows: n as i32, nlanes: nlanes as i32,
                                    rpl: rpl as i32, kind: super::KIND_STATS, rpl_t: rpl as i32, ..Default::default() })?;
        Ok(self.sres.ptr())
    }

    /// The verify of `nlanes` lanes: target rows (k + 1 per lane, `rpl_max` stride) against the drafts in the ids
    /// table and the drafter's rows (`d`, the logits the drafts were drawn from, after the Markov bias). Leaves
    /// each lane's emitted tokens in `out_tok` and its round result in `lanes` (n_acc, n_emit, wend, tok, counters).
    #[allow(clippy::too_many_arguments)]
    pub fn verify(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, t: Rows, d: Rows, nlanes: usize, tcut: Option<Cut>, dcut: Option<Cut>, test_flags: i32) -> Result<()> {
        self.verify_k(g, x, sh, t, d, nlanes, tcut, dcut, test_flags, usize::MAX)
    }

    /// [`Round::verify`] when the caller knows the most drafts any lane of the call verifies (`kmax`; every rank passes the
    /// same value). Leg PH1: at `kmax == 0` no lane can reject (`d41_ds_accept` takes the bonus draw and clears
    /// `need_res`), so the residual pass draws nothing and the FINISH merge reads no record: the pass and its exchange
    /// are skipped and FINISH runs alone. The same tokens and lane results, one exchange fewer per plain round.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_k(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, t: Rows, d: Rows, nlanes: usize, tcut: Option<Cut>, dcut: Option<Cut>, test_flags: i32,
                    kmax: usize) -> Result<()> {
        self.verify_rpl(g, x, sh, t, d, nlanes, self.rpl_max, tcut, dcut, test_flags, kmax)
    }

    /// [`Round::verify_k`] over `rpl` (`1..=self.rpl_max`) target rows per lane so normal `<= 6`-row rounds never scan
    /// unused wide-copy slots while `WIDEVERIFY1` (`7..=16` rows) scans the active block width.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_rpl(&self, g: &Gpu, x: &dyn VocabExchange, sh: Shard, t: Rows, d: Rows, nlanes: usize, rpl: usize, tcut: Option<Cut>, dcut: Option<Cut>,
                      test_flags: i32, kmax: usize) -> Result<()> {
        ensure!(rpl >= 1 && rpl <= self.rpl_max, "verify_rpl: {rpl} rows per lane (max {})", self.rpl_max);
        let mut a = self.base_args(sh, super::MODE_VERIFY, nlanes, rpl, 0, t);
        a.rpl_t = rpl as i32;
        a.ids = self.ids.ptr();
        if let Some(c) = tcut { a.cut_a = c.a; a.cut_id = c.id; a.cut_rpl = c.rpl as i32; }
        let n = nlanes * rpl;
        let recs = self.pass(g, x, sh, &a, n)?;
        self.merge(g, &VmergeArgs { recs, res: self.tres.ptr(), lanes: self.lanes.ptr(), nranks: x.ranks() as i32, rows: n as i32, nlanes: nlanes as i32,
                                    rpl: rpl as i32, kind: super::KIND_STATS, rpl_t: rpl as i32, ..Default::default() })?;
        g.launch1("d41_ds_accept", (nlanes as u32, 1, 1), (32, 1, 1), 0, &AcceptArgs { lanes: self.lanes.ptr(), tres: self.tres.ptr(), ids: self.ids.ptr(), dlp: self.dlp.ptr(),
                                                                                       out_tok: self.out_tok.ptr(), nlanes: nlanes as i32, rpl_t: rpl as i32, test_flags, pad: 0.0 })?;
        if kmax == 0 {
            PLAIN_FINISH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return self.merge(g, &VmergeArgs { recs: self.recv.ptr(), res: 0, lanes: self.lanes.ptr(), out_tok: self.out_tok.ptr(), tres: self.tres.ptr(), nranks: x.ranks() as i32,
                                               rows: nlanes as i32, nlanes: nlanes as i32, rpl: 1, kind: super::KIND_FINISH, rpl_t: rpl as i32, test_flags, ..Default::default() });
        }
        let mut r = self.base_args(sh, super::MODE_RESID, nlanes, 1, 0, t);
        r.rpl_t = rpl as i32;
        r.dlogits = d.ptr;
        r.d_lane_stride = d.lane_stride as i64;
        r.d_row_stride = d.row_stride as i64;
        r.dlse = self.dlse.ptr();
        r.tres = self.tres.ptr();
        if let Some(c) = tcut { r.cut_a = c.a; r.cut_id = c.id; r.cut_rpl = c.rpl as i32; }
        if let Some(c) = dcut { r.dcut_a = c.a; r.dcut_id = c.id; }
        let recs = self.pass(g, x, sh, &r, nlanes)?;
        self.merge(g, &VmergeArgs { recs, res: 0, lanes: self.lanes.ptr(), out_tok: self.out_tok.ptr(), tres: self.tres.ptr(), nranks: x.ranks() as i32, rows: nlanes as i32,
                                    nlanes: nlanes as i32, rpl: 1, kind: super::KIND_FINISH, rpl_t: rpl as i32, test_flags, ..Default::default() })
    }

    /// The round's commit: the next verify block starts at the drafter's block token (vbase = wend + 1).
    pub fn commit(&self, g: &Gpu, nlanes: usize) -> Result<()> {
        g.launch1("d41_ds_main", ((nlanes as u32).div_ceil(32), 1, 1), (32, 1, 1), 0, &super::MainArgs { lanes: self.lanes.ptr(), rows: nlanes as i32, mode: 1, ..Default::default() })
    }

    /// A pass over a vocabulary split into `shards` slices of the same rows, each slice's records placed in its rank
    /// slot as an all-gather would deliver them (the TP-invariance check of the probe; [`super::xchg::Simulated`]).
    pub fn stats_split(&self, g: &Gpu, shards: &[Shard], rows: Rows, nlanes: usize, rpl: usize) -> Result<u64> {
        ensure!(shards.len() == self.ranks, "{} shards for a {}-rank round", shards.len(), self.ranks);
        let n = nlanes * rpl;
        for (r, sh) in shards.iter().enumerate() {
            let mut a = self.base_args(*sh, super::MODE_VERIFY, nlanes, rpl, 0, Rows { ptr: rows.ptr + (sh.v0 * 4) as u64, ..rows });
            a.ids = self.ids.ptr();
            g.launch1("d41_ds_vpart", (sh.nblk() as u32, n as u32, 1), (256, 1, 1), 0, &a)?;
            g.launch1("d41_ds_vrank", ((n as u32).div_ceil(32), 1, 1), (32, 1, 1), 0,
                      &VrankArgs { part: self.part.ptr(), out: self.recv.ptr() + (r * n * REC) as u64, rows: n as i32, nblk: sh.nblk() as i32 })?;
        }
        self.merge(g, &VmergeArgs { recs: self.recv.ptr(), res: self.sres.ptr(), lanes: self.lanes.ptr(), nranks: shards.len() as i32, rows: n as i32, nlanes: nlanes as i32,
                                    rpl: rpl as i32, kind: super::KIND_STATS, rpl_t: self.rpl_max as i32, ..Default::default() })?;
        Ok(self.sres.ptr())
    }
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) } }

// ------------------------------------------------------------------------------------------------ host twins

/// splitmix64 finaliser (= SV's `mix`, = `d41_ds_mix`).
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
/// The row key of (seed, counter) (`d41_ds_key`).
pub fn key(seed: u64, ctr: u64) -> u64 { mix(mix(seed) ^ ctr.wrapping_mul(0xD6E8_FEB8_6659_FD93)) }
/// SV's 53-bit uniform (`d41_ds_uniform`; the same value as `serve::sampler::uniform`).
pub fn uniform(seed: u64, ctr: u64) -> f64 { (key(seed, ctr) >> 11) as f64 / (1u64 << 53) as f64 }
/// The Gumbel noise of `id` under `key` in f64 (the device computes the same u in f32; the twin is for tests).
pub fn gumbel(key: u64, id: u32) -> f64 {
    let r = mix(key.wrapping_add((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let u = ((r as f32) * 5.421_010_862_427_522e-20f32).max(f32::MIN_POSITIVE) as f64;
    -(-(-u).ln_1p()).ln()
}
/// The counter of a draw: position * 3 + stream.
pub fn counter(pos: i64, stream: u64) -> u64 { (pos as u64).wrapping_mul(3).wrapping_add(stream) }
/// The draft namespace's seed of a lane's round `rounds` (`d41_ds_draft_seed`, leg PS1: fresh draft noise every round).
pub fn draft_seed(seed: u64, rounds: u32) -> u64 { seed.wrapping_add(1_000_000_007).wrapping_add((rounds as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) }

/// Host twin of one row's record (f64 sums: a tolerance reference for the device's f32 record).
pub fn host_record(row: &[f32], v0: usize, inv_t: f32, cut: Option<(f32, i32)>, key_: u64, probe: Option<u32>) -> (f32, u32, f64, Option<u32>, Option<f64>) {
    let (ca, cid) = cut.unwrap_or((f32::NEG_INFINITY, i32::MAX));
    let (mut amax, mut aid) = (f32::NEG_INFINITY, u32::MAX);
    let (mut m, mut s) = (f64::NEG_INFINITY, 0f64);
    let (mut gm, mut gid) = (f64::NEG_INFINITY, None);
    let mut px = None;
    for (v, &a0) in row.iter().enumerate() {
        let a = if a0.is_nan() { f32::NEG_INFINITY } else { a0 };
        let id = (v0 + v) as u32;
        if a > amax { amax = a; aid = id; }
        let kept = a > ca || (a == ca && id as i32 <= cid);
        if inv_t > 0.0 && kept && a != f32::NEG_INFINITY {
            let x = (a * inv_t) as f64;
            if x > m { s = s * (m - x).exp() + 1.0; m = x; } else { s += (x - m).exp(); }
            let sc = x + gumbel(key_, id);
            if sc > gm { gm = sc; gid = Some(id); }
        }
        if probe == Some(id) { px = Some(if inv_t > 0.0 { if kept { (a * inv_t) as f64 } else { f64::NEG_INFINITY } } else { a as f64 }); }
    }
    (amax, aid, if m == f64::NEG_INFINITY { m } else { m + s.ln() }, gid, px)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ds_uniform_equals_the_serve_sampler() {
        for (s, c) in [(42u64, 0u64), (7, 123_456), (u64::MAX, 3 * 1_000_001 + 2)] {
            assert_eq!(uniform(s, c), crate::dsv41::serve::sampler::uniform(s, c));
        }
        assert_eq!(counter(1000, 2), 3002);
    }
    #[test]
    fn dsv41_ps1_draft_noise_is_fresh_every_round() {
        // round 0 keeps the position-only key of every earlier build; each later round draws from another stream
        assert_eq!(draft_seed(42, 0), 42 + 1_000_000_007);
        let keys: std::collections::BTreeSet<u64> = (0..64u32).map(|r| key(draft_seed(42, r), counter(1000, 0))).collect();
        assert_eq!(keys.len(), 64);
    }
    #[test]
    fn dsv41_ds_gumbel_noise_is_standard_gumbel() {
        // mean of Gumbel(0, 1) = Euler-Mascheroni 0.5772, variance pi^2 / 6
        let k = key(1, 2);
        let n = 200_000;
        let v: Vec<f64> = (0..n).map(|i| gumbel(k, i)).collect();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 0.5772).abs() < 0.01, "mean {mean}");
        assert!((var - std::f64::consts::PI.powi(2) / 6.0).abs() < 0.03, "var {var}");
        // the tail is not cut at 2^-24: values far above ln(2^24) = 16.6 are possible in principle; check the cap
        assert!(gumbel(k, 0).is_finite());
    }
    #[test]
    fn dsv41_ds_host_record_gumbel_max_samples_softmax() {
        let row: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0];
        let z: f64 = row.iter().map(|&x| (x as f64).exp()).sum();
        let mut cnt = [0usize; 4];
        let n = 60_000;
        for t in 0..n {
            let (_, _, lse, gid, _) = host_record(&row, 0, 1.0, None, key(9, t), None);
            assert!((lse - z.ln()).abs() < 1e-9);
            cnt[gid.unwrap() as usize] += 1;
        }
        for (i, c) in cnt.iter().enumerate() {
            let p = (row[i] as f64).exp() / z;
            assert!((*c as f64 / n as f64 - p).abs() < 0.01, "token {i}: {} vs {p}", *c as f64 / n as f64);
        }
    }
    #[test]
    fn dsv41_ds_cut_keeps_a_prefix_with_lowest_id_ties() {
        let row = [3.0f32, 3.0, 3.0, 1.0];
        // cut at (3.0, id 1): ids 0, 1 kept, 2 (tie, higher id) and 3 not
        let (_, aid, lse, _, px) = host_record(&row, 0, 1.0, Some((3.0, 1)), key(1, 1), Some(2));
        assert_eq!(aid, 0);
        assert!((lse - (2.0 * 3f64.exp()).ln()).abs() < 1e-9);
        assert_eq!(px, Some(f64::NEG_INFINITY));
    }
}
