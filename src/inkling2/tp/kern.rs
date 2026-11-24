//! Host side of the TP kernels (kernels/gpu_inkling2/d0_tp_attn.cuh, d8_tp_glue.cuh): the attention launches at a
//! rank's head width (`nq` query heads; `nq == 24` = the TP1 kernels of k1.rs, unchanged) and the vocab-parallel
//! sampler. Every launch goes through `Ink2Kernels::launch` (the module's stream: graph-capturable).

use crate::inkling2::gpu::k1::{attn_smem, attn_split, AttnArgs, AttnSplit, PrepArgs, ATTN_MR, ATTN_SP, ATTN_TC2_SMEM, ATTN_TC2_SP_SMEM, LANES_MAXS};
use crate::inkling2::gpu::{Ink2Kernels, KArgs};
use anyhow::{ensure, Result};

/// Head widths the TP copies are built for (24 = TP1; 12 = TP2; 9 = TP3; 6 = TP3/TP4/TP6; 3 = TP6/TP8).
pub const WIDTHS: [usize; 5] = [24, 12, 9, 6, 3];
const KINDS: [&str; 9] = ["attn_prep", "attn", "attn_combine", "attn_tc2", "attn_tc2_sp", "attn_prep_lanes", "attn_lanes", "attn_combine_lanes", "attn_lanes_mr"];

/// Every TP entry (loaded with the module: `gpu::FNS` + this list).
pub fn fns() -> Vec<String> {
    let mut v: Vec<String> = WIDTHS.iter().flat_map(|n| KINDS.iter().map(move |k| format!("ink_tp{n}_{k}"))).collect();
    v.extend([
        "ink_tp_gumbel_part", "ink_tp_gumbel_rec", "ink_tp_rank_pick",
        "ink_tp_logits_place", "ink_tp_logits_pack", "ink_tp_logits_unpack",
        "ink_tp_mtp_step_pack", "ink_tp_mtp_step_fin",
        "ink_tp_accept_pack", "ink_tp_accept_decide", "ink6_tp_final_part",
        "ink_sconv_sp",
        "ink_ulysses_y_pack", "ink_ulysses_y_unpack",
        "ink_ulysses_att_pack", "ink_ulysses_att_unpack",
        "ink_ulysses_y_pack_b16", "ink_ulysses_y_unpack_b16",
        "ink_ulysses_att_pack_b16", "ink_ulysses_att_unpack_b16",
    ].map(String::from));
    v
}
/// (entry, dynamic smem bytes) that need the opt-in above 48 KB.
pub fn smem_optins() -> Vec<(String, i32)> {
    let mut v = Vec::new();
    for n in WIDTHS {
        v.push((format!("ink_tp{n}_attn"), attn_smem(16) as i32));
        v.push((format!("ink_tp{n}_attn_tc2"), ATTN_TC2_SMEM as i32));
        v.push((format!("ink_tp{n}_attn_lanes"), attn_smem(1) as i32));
        v.push((format!("ink_tp{n}_attn_tc2_sp"), ATTN_TC2_SP_SMEM as i32));
        v.push((format!("ink_tp{n}_attn_lanes_mr"), attn_smem(16) as i32));
    }
    v
}

thread_local! {
    /// The identity gate (`--probe-inkling2-tp-sim`): route the TP1 (24-head) launches to d0's NQ = 24 copies.
    pub static FORCE_TP24: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
fn tp1(nq: usize) -> bool { nq == 24 && !FORCE_TP24.with(|c| c.get()) }

fn name(nq: usize, k: &str) -> Result<String> {
    ensure!(WIDTHS.contains(&nq), "inkling2 TP: no attention kernels for {nq} heads per rank");
    Ok(format!("ink_tp{nq}_{k}"))
}
fn prep_threads(nq: usize) -> u32 { ((nq + nq / 3) * 32) as u32 }

/// `k1::attn_prep` at `nq` heads.
pub fn attn_prep(k: &Ink2Kernels, nq: usize, a: &PrepArgs) -> Result<()> {
    if tp1(nq) { return k.attn_prep(a); }
    k.launch(&name(nq, "attn_prep")?, (a.m as u32, 1, 1), (prep_threads(nq), 1, 1), 0, KArgs::new().v(*a))
}

/// `k1::attn` at `nq` heads (prefill rows > 16: the two-tile tensor-core kernel; else the CUDA-core split kernel).
pub fn attn(k: &Ink2Kernels, nq: usize, a: &AttnArgs, prefill: bool, tc: bool) -> Result<AttnSplit> {
    if tp1(nq) { return k.attn(a, prefill, tc); }
    let m = a.m as usize;
    let nkv = nq / 3;
    let mut a = *a;
    let sp = if prefill || m > 16 { AttnSplit { chunk: 1 << 30, c0: 0, nsplit: 1 } } else { attn_split(a.p0 as usize, m, a.window as usize) };
    a.chunk = sp.chunk as i32;
    a.c0 = sp.c0 as i32;
    if tc && sp.nsplit == 1 && m > 16 {
        let (kind, sm) = if ATTN_SP.with(|c| c.get()) { ("attn_tc2_sp", ATTN_TC2_SP_SMEM) } else { ("attn_tc2", ATTN_TC2_SMEM) };
        k.launch(&name(nq, kind)?, (1, nkv as u32, m.div_ceil(32) as u32), (192, 1, 1), sm as u32, KArgs::new().v(a))?;
    } else {
        ensure!(!tc || m <= 16, "inkling2 TP attention: the one-tile tensor-core kernel is not built for TP (m {m})");
        k.launch(&name(nq, "attn")?, (sp.nsplit as u32, nkv as u32, m.div_ceil(16) as u32), (256, 1, 1), attn_smem(m) as u32, KArgs::new().v(a))?;
    }
    if sp.nsplit > 1 {
        k.launch(&name(nq, "attn_combine")?, (m as u32, nq as u32, 1), (128, 1, 1), 0, KArgs::new().v(a).v(sp.nsplit as i32))?;
    }
    Ok(sp)
}

pub fn attn_prep_lanes(k: &Ink2Kernels, nq: usize, a: &PrepArgs, tab: u64, slot: usize, nslots: usize, rinfo: u64, n: usize) -> Result<()> {
    if tp1(nq) { return k.attn_prep_lanes(a, tab, slot, nslots, rinfo, n); }
    k.launch(&name(nq, "attn_prep_lanes")?, (n as u32, 1, 1), (prep_threads(nq), 1, 1), 0,
             KArgs::new().v(*a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))
}

/// `k1::attn_lanes` at `nq` heads (`runs`: some lane carries several rows: the multi-row entry serves those runs).
#[allow(clippy::too_many_arguments)]
pub fn attn_lanes(k: &Ink2Kernels, nq: usize, a: &AttnArgs, tab: u64, slot: usize, nslots: usize, rinfo: u64, n: usize, runs: bool) -> Result<()> {
    if tp1(nq) { return k.attn_lanes(a, tab, slot, nslots, rinfo, n, runs); }
    let runs = runs && ATTN_MR.with(|c| c.get());
    let mut a = *a;
    a.c0 = if runs { 0 } else { 1 };
    k.launch(&name(nq, "attn_lanes")?, (LANES_MAXS as u32, (nq / 3) as u32, n as u32), (256, 1, 1), attn_smem(1) as u32,
             KArgs::new().v(a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))?;
    if runs {
        k.launch(&name(nq, "attn_lanes_mr")?, (LANES_MAXS as u32 + 1, (nq / 3) as u32, n as u32), (256, 1, 1), attn_smem(16) as u32,
                 KArgs::new().v(a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))?;
    }
    let a = &a;
    k.launch(&name(nq, "attn_combine_lanes")?, (n as u32, nq as u32, 1), (128, 1, 1), 0, KArgs::new().v(*a).v(rinfo))
}

/// The vocab-parallel sampler, stage 1: per row the rank's best (perturbed score, global id) into its slot of `rec`
/// ([m][world][2] fp32, the other slots zeroed). `g` CTAs per row over the rank's `n` logits at `voff`.
#[allow(clippy::too_many_arguments)]
pub fn gumbel_rec(k: &Ink2Kernels, logits: u64, ld: usize, n: usize, voff: usize, seeds: u64, inv_t: u64, pmax: u64, pidx: u64,
                  g: usize, m: usize, rec: u64, rank: usize, world: usize) -> Result<()> {
    k.launch("ink_tp_gumbel_part", (g as u32, m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(logits).v(ld as i32).v(n as i32).v(voff as i32).v(seeds).v(inv_t).v(pmax).v(pidx))?;
    k.launch("ink_tp_gumbel_rec", (m as u32, 1, 1), (256, 1, 1), 0,
             KArgs::new().v(pmax).v(pidx).v(g as i32).v(rec).v(rank as i32).v(world as i32))
}
/// Stage 2 (after the all-reduce of `rec`): the winning id of every row into `out` [m] i32.
pub fn rank_pick(k: &Ink2Kernels, rec: u64, world: usize, m: usize, out: u64) -> Result<()> {
    ensure!(m <= 32);
    k.launch("ink_tp_rank_pick", (1, 1, 1), (32, 1, 1), 0, KArgs::new().v(rec).v(world as i32).v(m as i32).v(out))
}
/// This rank's logits rows into a zeroed full-vocabulary buffer `full` [m][v] at column `voff`.
pub fn logits_place(k: &Ink2Kernels, logits: u64, ld: usize, n: usize, voff: usize, full: u64, v: usize, m: usize) -> Result<()> {
    k.launch("ink_tp_logits_place", (v.div_ceil(256).min(256) as u32, m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(logits).v(ld as i32).v(n as i32).v(voff as i32).v(full).v(v as i32))
}
/// Phase B: pack this rank's `[m][vn]` logits into `gather[(rank * m + row) * per ..]` for 2-rail all-gather.
#[allow(clippy::too_many_arguments)]
pub fn logits_pack(k: &Ink2Kernels, logits: u64, ld: usize, vn: usize, gather: u64, per: usize, rank: usize, m: usize) -> Result<()> {
    k.launch("ink_tp_logits_pack", (per.div_ceil(256).min(256) as u32, m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(logits).v(ld as i32).v(vn as i32).v(gather).v(per as i32).v(rank as i32).v(m as i32))
}
/// Phase B: unpack all-gathered `[world][m][per]` into `full` `[m][v]`.
pub fn logits_unpack(k: &Ink2Kernels, gather: u64, per: usize, full: u64, v: usize, m: usize, world: usize) -> Result<()> {
    k.launch("ink_tp_logits_unpack", (v.div_ceil(256).min(256) as u32, m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(gather).v(per as i32).v(full).v(v as i32).v(m as i32).v(world as i32))
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MtpStepPackArgs {
    pub fpm: [u64; 4],
    pub fps: [u64; 4],
    pub gpm: [u64; 4],
    pub gpi: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MtpStepFinArgs {
    pub lse_q: [u64; 4],
    pub conf: [u64; 4],
    pub tok: [u64; 4],
}

/// Drop 1: pack local `(max_q, sum_q, best_gumbel, global_id)` for `nl` active lanes into `rec` `[nl][world][4]`.
pub fn mtp_step_pack(k: &Ink2Kernels, a: MtpStepPackArgs, nl: usize, fold_g: usize, smp_g: usize, rec: u64, rank: usize, world: usize) -> Result<()> {
    k.launch("ink_tp_mtp_step_pack", (nl as u32, 1, 1), (256, 1, 1), 0,
             KArgs::new().v(a).v(fold_g as i32).v(smp_g as i32).v(rec).v(rank as i32).v(world as i32))
}

/// Drop 1: finalize global `lse_q`, `conf`, and global winning draft token for `nl` active lanes from all-reduced `rec`.
pub fn mtp_step_fin(k: &Ink2Kernels, rec: u64, world: usize, o: MtpStepFinArgs, nl: usize) -> Result<()> {
    k.launch("ink_tp_mtp_step_fin", (nl as u32, 1, 1), (32, 1, 1), 0,
             KArgs::new().v(rec).v(world as i32).v(o))
}

/// Drop 1: pack local target LSE + draft-token `(tl_d, dl_d)` across `c + 1` verify rows into `rec` `[c+1][world][4]`.
#[allow(clippy::too_many_arguments)]
pub fn accept_pack(k: &Ink2Kernels, lpm: u64, lps: u64, lse_g: usize, tok: u64, c: usize,
                   tl: u64, ldl: usize, dl: u64, ldd: usize, voff: usize, vn: usize,
                   rec: u64, rank: usize, world: usize) -> Result<()> {
    k.launch("ink_tp_accept_pack", ((c + 1) as u32, 1, 1), (32, 1, 1), 0,
             KArgs::new().v(lpm).v(lps).v(lse_g as i32).v(tok).v(c as i32)
                 .v(tl).v(ldl as i32).v(dl).v(ldd as i32)
                 .v(voff as i32).v(vn as i32).v(rec).v(rank as i32).v(world as i32))
}

/// Drop 1: decide exact global `lse_p[0..=c]` and accepted count `res[0]` from all-reduced `rec`.
#[allow(clippy::too_many_arguments)]
pub fn accept_decide(k: &Ink2Kernels, rec: u64, world: usize, c: usize, lse_q: u64, lse_p: u64, it: f32, seed: u64, res: u64) -> Result<()> {
    k.launch("ink_tp_accept_decide", (1, 1, 1), (32, 1, 1), 0,
             KArgs::new().v(rec).v(world as i32).v(c as i32).v(lse_q).v(lse_p).v(it).v(seed).v(res))
}

/// Drop 1: vocab-parallel final draw `(p_a - q_a)+` or `p_c` over this rank's `[voff, voff + vn)`.
#[allow(clippy::too_many_arguments)]
pub fn final_part(k: &Ink2Kernels, tl: u64, ldl: usize, dl: u64, ldd: usize, vn: usize, voff: usize,
                  lse_p: u64, lse_q: u64, it: f32, res: u64, c: usize, seed: u64,
                  g: usize, pmax: u64, pidx: u64) -> Result<()> {
    k.launch("ink6_tp_final_part", (g as u32, 1, 1), (256, 1, 1), 0,
             KArgs::new().v(tl).v(ldl as i32).v(dl).v(ldd as i32).v(vn as i32).v(voff as i32)
                 .v(lse_p).v(lse_q).v(it).v(res).v(c as i32).v(seed).v(pmax).v(pidx))
}

/// Drop 2 (#2): sequence-parallel short causal conv (`k = 4`) + residual over `[my_r0 .. my_r1)` (`m` rows).
/// `halo` = previous rank's last 3 rows `[3][c]` fp32 (`0` on rank 0 -> reads `ring`).
#[allow(clippy::too_many_arguments)]
pub fn sconv_sp(k: &Ink2Kernels, x: u64, halo: u64, ring: u64, w: u64, base: u64, out: u64, c: usize, p0: usize, m: usize) -> Result<()> {
    k.launch("ink_sconv_sp", (c.div_ceil(256) as u32, m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(x).v(halo).v(ring).v((crate::inkling2::gpu::k1::RING - 1) as i32)
                 .v(w).v(base).v(out).v(c as i32).v(p0 as i32).v(1.0f32 / 5f32.sqrt()))
}

/// DeepSpeed-Ulysses (Jacobs et al., arXiv:2309.14509) per-rank sequence block descriptor passed by value.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct InkUlyBlocks {
    pub r0: [i32; 8],
    pub m: [i32; 8],
}

impl InkUlyBlocks {
    pub fn from_blocks(blocks: &[(usize, usize)]) -> Self {
        let mut out = Self::default();
        for (p, &(a, b)) in blocks.iter().enumerate().take(8) {
            out.r0[p] = a as i32;
            out.m[p] = (b - a) as i32;
        }
        out
    }
}

/// Arbitrary-W (TP2/TP3/TP4/TP6/TP8) per-rank head partition passed by value to Ulysses pack/unpack kernels.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct InkUlyHeads {
    pub h0: [i32; 8],
    pub nq: [i32; 8],
    pub g0: [i32; 8],
    pub nkv: [i32; 8],
}

impl InkUlyHeads {
    pub fn for_world(world: usize, inner: usize, vocab: usize) -> Result<(Self, usize, usize)> {
        let mut out = Self::default();
        let (mut max_nq, mut max_nkv) = (0usize, 0usize);
        for p in 0..world.min(8) {
            let r = super::shard::Rank::new(p, world, inner, vocab)?;
            out.h0[p] = r.h0 as i32;
            out.nq[p] = r.nq as i32;
            out.g0[p] = r.g0 as i32;
            out.nkv[p] = r.nkv as i32;
            max_nq = max_nq.max(r.nq);
            max_nkv = max_nkv.max(r.nkv);
        }
        Ok((out, max_nq, max_nkv))
    }
}

/// DeepSpeed-Ulysses (arXiv:2309.14509) forward pack: `y_full[my_m, ldy_f]` -> `a2a_src[world][max_my_m][y_stride]` (without `gate_attn`).
#[allow(clippy::too_many_arguments)]
pub fn ulysses_y_pack(
    k: &Ink2Kernels, y_full: u64, ldy_f: usize, a2a_src: u64, y_stride: usize,
    max_my_m: usize, my_m: usize, world: usize, hds: InkUlyHeads,
    oq: usize, orr: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
) -> Result<()> {
    k.launch("ink_ulysses_y_pack", (world as u32, my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(y_full).v(ldy_f as i32).v(a2a_src).v(y_stride as i32).v(max_my_m as i32)
                 .v(hds)
                 .v(oq as i32).v(orr as i32).v(os as i32)
                 .v(ok as i32).v(okd).v(ov as i32).v(ovd))
}

/// DeepSpeed-Ulysses (arXiv:2309.14509) forward unpack: `a2a_dst[world][max_my_m][y_stride]` -> `ws_y[m, ldy_tp]` (without `gate_attn`).
#[allow(clippy::too_many_arguments)]
pub fn ulysses_y_unpack(
    k: &Ink2Kernels, a2a_dst: u64, y_stride: usize, max_my_m: usize, ws_y: u64, ldy_tp: usize,
    world: usize, nq: usize, nkv: usize,
    oq: usize, orr: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
    blk: InkUlyBlocks,
) -> Result<()> {
    k.launch("ink_ulysses_y_unpack", (world as u32, max_my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(a2a_dst).v(y_stride as i32).v(max_my_m as i32).v(ws_y).v(ldy_tp as i32)
                 .v(nq as i32).v(nkv as i32)
                 .v(oq as i32).v(orr as i32).v(os as i32)
                 .v(ok as i32).v(okd).v(ov as i32).v(ovd).v(blk))
}

/// DeepSpeed-Ulysses (arXiv:2309.14509) reverse pack: ungated FP32 `att_f32[m, nq_hd]` -> `a2a_src[world][max_my_m][att_stride]`.
pub fn ulysses_att_pack(
    k: &Ink2Kernels, att_f32: u64, a2a_src: u64, max_my_m: usize, world: usize, nq_hd: usize, att_stride: usize, blk: InkUlyBlocks,
) -> Result<()> {
    k.launch("ink_ulysses_att_pack", (world as u32, max_my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(att_f32).v(a2a_src).v(max_my_m as i32).v(nq_hd as i32).v(att_stride as i32).v(blk))
}

/// DeepSpeed-Ulysses (arXiv:2309.14509) reverse unpack + local `sigmoid(gate_attn)` epilogue (TP2/TP3/TP4/TP6/TP8).
#[allow(clippy::too_many_arguments)]
pub fn ulysses_att_unpack(
    k: &Ink2Kernels, a2a_dst: u64, y_full: u64, ldy_f: usize, og_f: usize, att_full: u64,
    max_my_m: usize, my_m: usize, world: usize, att_stride: usize, all_hd: usize, hds: InkUlyHeads,
) -> Result<()> {
    k.launch("ink_ulysses_att_unpack", (world as u32, my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(a2a_dst).v(y_full).v(ldy_f as i32).v(og_f as i32).v(att_full)
                 .v(max_my_m as i32).v(att_stride as i32).v(all_hd as i32).v(hds))
}

/// U7a: opt-in BF16 Ulysses forward pack (Q/K/V BF16, r/s/kd/vd FP32, arbitrary W).
#[allow(clippy::too_many_arguments)]
pub fn ulysses_y_pack_b16(
    k: &Ink2Kernels, y_full: u64, ldy_f: usize, a2a_src: u64, row_bytes: usize,
    max_my_m: usize, my_m: usize, world: usize, hds: InkUlyHeads,
    oq: usize, orr: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
) -> Result<()> {
    k.launch("ink_ulysses_y_pack_b16", (world as u32, my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(y_full).v(ldy_f as i32).v(a2a_src).v(row_bytes as i32).v(max_my_m as i32)
                 .v(hds)
                 .v(oq as i32).v(orr as i32).v(os as i32)
                 .v(ok as i32).v(okd).v(ov as i32).v(ovd))
}

/// U7a: opt-in BF16 Ulysses forward unpack.
#[allow(clippy::too_many_arguments)]
pub fn ulysses_y_unpack_b16(
    k: &Ink2Kernels, a2a_dst: u64, row_bytes: usize, max_my_m: usize, ws_y: u64, ldy_tp: usize,
    world: usize, nq: usize, nkv: usize,
    oq: usize, orr: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
    blk: InkUlyBlocks,
) -> Result<()> {
    k.launch("ink_ulysses_y_unpack_b16", (world as u32, max_my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(a2a_dst).v(row_bytes as i32).v(max_my_m as i32).v(ws_y).v(ldy_tp as i32)
                 .v(nq as i32).v(nkv as i32)
                 .v(oq as i32).v(orr as i32).v(os as i32)
                 .v(ok as i32).v(okd).v(ov as i32).v(ovd).v(blk))
}

/// U7a: opt-in BF16 Ulysses reverse pack (ungated att_f32 -> BF16).
pub fn ulysses_att_pack_b16(
    k: &Ink2Kernels, att_f32: u64, a2a_src: u64, max_my_m: usize, world: usize, nq_hd: usize, att_stride: usize,
    blk: InkUlyBlocks,
) -> Result<()> {
    k.launch("ink_ulysses_att_pack_b16", (world as u32, max_my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(att_f32).v(a2a_src).v(max_my_m as i32).v(nq_hd as i32).v(att_stride as i32).v(blk))
}

/// U7a: opt-in BF16 Ulysses reverse unpack + local `sigmoid(gate_attn)` -> BF16 `att_full` (TP2/TP3/TP4/TP6/TP8).
#[allow(clippy::too_many_arguments)]
pub fn ulysses_att_unpack_b16(
    k: &Ink2Kernels, a2a_dst: u64, y_full: u64, ldy_f: usize, og_f: usize, att_full: u64,
    max_my_m: usize, my_m: usize, world: usize, att_stride: usize, all_hd: usize, hds: InkUlyHeads,
) -> Result<()> {
    k.launch("ink_ulysses_att_unpack_b16", (world as u32, my_m as u32, 1), (256, 1, 1), 0,
             KArgs::new().v(a2a_dst).v(y_full).v(ldy_f as i32).v(og_f as i32).v(att_full)
                 .v(max_my_m as i32).v(att_stride as i32).v(all_hd as i32).v(hds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulysses_heads_partition_all_topologies() {
        for (w, exp_nq, exp_nkv) in [
            (2usize, vec![12, 12], vec![4, 4]),
            (3, vec![9, 9, 6], vec![3, 3, 2]),
            (4, vec![6, 6, 6, 6], vec![2, 2, 2, 2]),
            (6, vec![6, 6, 3, 3, 3, 3], vec![2, 2, 1, 1, 1, 1]),
            (8, vec![3; 8], vec![1; 8]),
        ] {
            let (h, max_nq, max_nkv) = InkUlyHeads::for_world(w, 1152, 131072).unwrap();
            assert_eq!(max_nq, exp_nq[0] as usize);
            assert_eq!(max_nkv, exp_nkv[0] as usize);
            let (mut h0, mut g0) = (0i32, 0i32);
            for p in 0..w {
                assert_eq!(h.h0[p], h0);
                assert_eq!(h.nq[p], exp_nq[p]);
                assert_eq!(h.g0[p], g0);
                assert_eq!(h.nkv[p], exp_nkv[p]);
                h0 += h.nq[p];
                g0 += h.nkv[p];
            }
            assert_eq!(h0, 24);
            assert_eq!(g0, 8);
        }
    }
}
