//! DSV4.1 GPU host side of the DSpark round (leg DS). Kernels: band 90-99, `kernels/gpu_dsv41/90_dspark.cuh`
//! (+ `91_ds_vocab.cuh`, `92_ds_draft.cuh`); the structs below are their byte-exact twins (`d41_ds_sizes` checks).
//!
//! * `xchg` — the vocab-parallel exchange the round needs from the TP layer (one 64-B record per row and rank);
//!   `Local` is the TP1 implementation. Leg TPC provides the multi-rank one.
//! * `vocab` — the vocab passes (statistics, keyed Gumbel draws, the residual draw, accept / finish) and the
//!   round's device state (`Round`); host twins of the record semantics for the unit tests.
//! * `drafter` — the GPU drafter: `main_proj` over the layer 37-39 taps, the per-stage window rings, the 3 stages
//!   (SL's per-layer pieces), the tied head, the sequential Markov chain (rank 256), the confidence head.
//! * `gate` — `--probe-dsv41-dspark`: the drafter against REFb's drafts with OD's floor rule and against OD's
//!   traces per op (head side), graph capture, step timing.
//! * `chisq` — the sampled-verification gate: reference-free two-sample chi-square on synthetic distributions.
//! * `policy` — the draft-length policy priced from measured round costs (cached per binary + pack), the
//!   confidence-scheduled variable length (opt-in lever).
//! * `metrics` — `/metrics` counters, the once-per-boot line naming the speculation source.
//! * `prefetch` — the draft-token Engram prefetch through EIO's `warm`.

pub mod alloc;
pub mod chain;
pub mod chisq;
pub mod drafter;
#[cfg(test)]
mod exact_tests;
pub mod dsattn;
pub mod gate;
pub mod mcache;
pub mod metrics;
pub mod mtoks;
pub mod policy;
pub mod prefetch;
pub mod roundcost;
pub mod tpbench;
pub mod vocab;
pub mod xchg;

use cudarc::driver::DeviceRepr;

/// Entry kernels of band 90 (the gpu_dsv41 loader collects them).
pub const FNS: &[&str] = &[
    "d41_ds_sizes", "d41_ds_vpart", "d41_ds_vrank", "d41_ds_vmerge", "d41_ds_accept", "d41_ds_kv_rows", "d41_ds_kv_rows3", "d41_ds_setup", "d41_ds_rep", "d41_ds_main",
    "d41_ds_mstep_l2", "d41_ds_mstep_l4", "d41_ds_vscan", "d41_ds_vfin", "d41_ds_conf",
    "d41_ds_plain_r8", "d41_ds_plain_r16", "d41_ds_plain_r24", "d41_ds_plain_r32",
    "d41_ds_mbias", "d41_ds_mbias_r4", "d41_ds_mc_admit",
];

/// DSpark block rows per lane (the trained block; DeepSeek's `DEEPINFER_USE_NEXTN_DECODE` default).
pub const BLOCK: usize = 5;
/// Rows of the DSpark draft ids table: [0] = the block token, [1..=5] = the drafts.
pub const IDS_ROWS: usize = BLOCK + 1;
/// Maximum verify rows per lane under wide decode copy (`WIDEVERIFY1`: [0] = pending + up to 15 copied tokens).
pub const WIDE_RPL: usize = 16;
/// Emitted-token slots per lane (`D41_DS_OUT_TOK`).
pub const OUT_TOK: usize = 16;
/// alpha = exp(-5e-5): DeepSeek's draft-probability penalty of the residual (DRAFT_PROB_PENALTY).
pub fn alpha() -> f32 { (-5e-5f64).exp() as f32 }

pub const MODE_VERIFY: i32 = 0;
pub const MODE_DRAFT: i32 = 1;
pub const MODE_RESID: i32 = 2;
pub const KIND_STATS: i32 = 0;
pub const KIND_PICK: i32 = 1;
pub const KIND_FINISH: i32 = 2;
/// accept test flag 1: the swapped rule u < min(1, p / q) (a negative control of the chi-square gate only)
pub const TEST_SWAP_RULE: i32 = 1;
/// finish test flag 2: a fresh draw from q instead of the residual after a rejection (negative control only)
pub const TEST_NO_RESIDUAL: i32 = 2;

/// One (row, part) record of a vocab pass (`d41_ds_vrec`, 64 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VRec {
    pub amax: f32, pub aid: i32,
    pub m: f32, pub s: f32,
    pub gmax: f32, pub gid: i32, pub gx: f32,
    pub px: f32, pub pheld: i32,
    pub rsum: f32, pub rmax: f32, pub rid: i32,
    pub pad: [i32; 4],
}
unsafe impl DeviceRepr for VRec {}
impl VRec {
    /// log-sum-exp of the kept set (x units)
    pub fn lse(&self) -> f32 { if self.m == f32::NEG_INFINITY { f32::NEG_INFINITY } else { self.m + self.s.ln() } }
}

/// Per-lane round state (`d41_ds_lane`, 72 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lane {
    pub seed: u64,
    pub inv_t: f32,
    pub vbase: i32,
    pub k: i32,
    pub wend: i32,
    pub tok: i32,
    pub n_acc: i32,
    pub n_emit: i32,
    pub need_res: i32,
    pub rounds: u32,
    pub drafted: u32,
    pub accepted: u32,
    pub active: i32,
    pub zmass: i32,
    /// leg CD1: this round's drafts of the lane are a copy proposal (the accept kernel compares with the target's own draw)
    pub copy: i32,
    /// leg P7.2: rows 0..suppress_eos mask stop_ids to -inf (0 = none)
    pub suppress_eos: i32,
    /// leg P7.4: 1 + packed target row0 in tlog (0 = strided layout)
    pub trow0_p1: i32,
}
unsafe impl DeviceRepr for Lane {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VpartArgs {
    pub logits: u64, pub lane_stride: i64, pub row_stride: i64,
    pub dlogits: u64, pub d_lane_stride: i64, pub d_row_stride: i64,
    pub lanes: u64,
    pub cut_a: u64, pub cut_id: u64, pub dcut_a: u64, pub dcut_id: u64, pub dlse: u64,
    pub tres: u64, pub ids: u64, pub part: u64,
    pub nlanes: i32, pub rpl: i32, pub j0: i32, pub cut_rpl: i32,
    pub vlocal: i32, pub v0: i32, pub chunk: i32, pub mode: i32,
    pub rpl_t: i32, pub alpha: f32, pub stop_ids: [i32; 4],
}
unsafe impl DeviceRepr for VpartArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VrankArgs { pub part: u64, pub out: u64, pub rows: i32, pub nblk: i32 }
unsafe impl DeviceRepr for VrankArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VmergeArgs {
    pub recs: u64, pub res: u64, pub lanes: u64, pub ids: u64, pub dlp: u64, pub dlse: u64, pub out_tok: u64, pub tres: u64,
    pub nranks: i32, pub rows: i32, pub nlanes: i32, pub rpl: i32,
    pub j0: i32, pub kind: i32, pub rpl_t: i32, pub test_flags: i32,
}
unsafe impl DeviceRepr for VmergeArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptArgs { pub lanes: u64, pub tres: u64, pub ids: u64, pub dlp: u64, pub out_tok: u64, pub nlanes: i32, pub rpl_t: i32, pub test_flags: i32, pub pad: f32 }
unsafe impl DeviceRepr for AcceptArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvArgs {
    pub kv_raw: u64, pub norm_w: u64, pub inv_freq: u64, pub pos: u64, pub lane: u64,
    pub ring_q: u64, pub ring_s: u64, pub ring_lq: i64, pub ring_ls: i64,
    pub chunk_q: u64, pub chunk_s: u64,
    pub rows: i32, pub norm_fmt: u32, pub eps: f32, pub pad: i32,
}
unsafe impl DeviceRepr for KvArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Kv3Args {
    pub kv_raw: u64, pub norm_w: [u64; 3], pub inv_freq: u64, pub pos: u64, pub lane: u64,
    pub ring_q: u64, pub ring_s: u64, pub ring_lq: i64, pub ring_ls: i64,
    pub rows: i32, pub cap: i32, pub norm_fmt: [u32; 3], pub eps: f32,
}
unsafe impl DeviceRepr for Kv3Args {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SetupArgs { pub lanes: u64, pub eids: u64, pub bpos: u64, pub bwend: u64, pub blane: u64, pub ids: u64, pub pre: u64, pub nlanes: i32, pub noise: i32 }
unsafe impl DeviceRepr for SetupArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RepArgs { pub e: u64, pub x: u64, pub rows: i32, pub dim: i32 }
unsafe impl DeviceRepr for RepArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MainArgs { pub lanes: u64, pub rlane: u64, pub rj: u64, pub mpos: u64, pub mlane: u64, pub rows: i32, pub mode: i32 }
unsafe impl DeviceRepr for MainArgs {}

/// `d41_ds_mstep_args` (leg PK5): the chain step's fused gather + Markov GEMV + Gumbel scores.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MstepArgs {
    pub me: u64, pub mh: u64, pub table: u64, pub cin: u64, pub logits: u64,
    pub lane_stride: i64, pub row_stride: i64, pub cin_ls: i64,
    pub lanes: u64, pub gsc: u64,
    pub nlanes: i32, pub step: i32, pub ld: i32, pub d: i32,
    pub vlocal: i32, pub v0: i32, pub mh_code: i32, pub pad: i32,
}
unsafe impl DeviceRepr for MstepArgs {}

/// `d41_ds_vscan_args` (leg PK5): the draft rows' block records with the scores read back.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VscanArgs { pub logits: u64, pub lane_stride: i64, pub row_stride: i64, pub gsc: u64, pub lanes: u64, pub part: u64, pub step: i32, pub vlocal: i32, pub v0: i32, pub pad: i32 }
unsafe impl DeviceRepr for VscanArgs {}

/// `d41_ds_vfin_args` (leg PK5): TP1 block merge + PICK.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VfinArgs { pub part: u64, pub recs: u64, pub lanes: u64, pub ids: u64, pub dlp: u64, pub dlse: u64, pub nlanes: i32, pub nblk: i32, pub step: i32, pub pick: i32 }
unsafe impl DeviceRepr for VfinArgs {}

/// `d41_ds_conf_args` (leg PK5): the confidence head, one warp per row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ConfArgs { pub w: u64, pub x: u64, pub y: u64, pub wcode: i32, pub k: i32, pub ldx: i32, pub ldy: i32, pub rows: i32, pub pad: i32 }
unsafe impl DeviceRepr for ConfArgs {}

/// `d41_ds_plain_args` (leg PK5): a plain GEMV over up to 32 rows in one pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlainArgs { pub w: u64, pub x: u64, pub y: u64, pub wcode: i32, pub k: i32, pub nn: i32, pub xcode: i32, pub ldx: i32, pub rows: i32, pub ycode: i32, pub ldy: i32, pub accum: i32, pub pad: i32 }
unsafe impl DeviceRepr for PlainArgs {}

/// The host sizes the device's `d41_ds_sizes` must report, in its order.
pub fn host_sizes() -> [usize; 16] {
    use std::mem::size_of;
    [size_of::<VRec>(), size_of::<Lane>(), size_of::<VpartArgs>(), size_of::<VrankArgs>(), size_of::<VmergeArgs>(), size_of::<AcceptArgs>(),
     size_of::<KvArgs>(), size_of::<SetupArgs>(), size_of::<RepArgs>(), size_of::<MainArgs>(), size_of::<MstepArgs>(), size_of::<ConfArgs>(), size_of::<PlainArgs>(),
     size_of::<VscanArgs>(), size_of::<VfinArgs>(), size_of::<mcache::MbiasArgs>()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ds_struct_layouts() {
        let s = host_sizes();
        assert_eq!(s[0], 64);
        assert_eq!(s[1], 72);
        assert_eq!(s[2], 15 * 8 + 14 * 4);
        assert_eq!(std::mem::size_of::<Kv3Args>(), 11 * 8 + 6 * 4);
        assert_eq!(s[3], 2 * 8 + 2 * 4);
        assert_eq!(s[4], 8 * 8 + 8 * 4);
        assert_eq!(s[5], 5 * 8 + 4 * 4);
        assert_eq!(s[6], 11 * 8 + 4 * 4);
        assert_eq!(s[7], 7 * 8 + 2 * 4);
        assert_eq!(s[8], 2 * 8 + 2 * 4);
        assert_eq!(s[9], 5 * 8 + 2 * 4);
        assert_eq!(s[10], 10 * 8 + 8 * 4);
        assert_eq!(s[11], 3 * 8 + 6 * 4);
        assert_eq!(s[12], 3 * 8 + 10 * 4);
        assert_eq!(s[13], 6 * 8 + 4 * 4);
        assert_eq!(s[14], 6 * 8 + 4 * 4);
        assert!((alpha() as f64 - (-5e-5f64).exp()).abs() < 1e-7 && alpha() < 1.0);
    }
}
