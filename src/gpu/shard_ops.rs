//! Host-side TP shard-at-load ops: `LoadShardOp`, per-family shard-factor / op selection (hy_v3, qwen), the
//! `shard_raw_*` and `host_shard_*` byte-band helpers. Twins of the GPU-side sharders in `tp_shard.rs`: the byte math must stay identical.
//! Map: `mod.rs`.

use super::*;

// -----------------------------------------------------------------------------------------------
// TP=2 SHARD-AT-LOAD (hy_v3): host-side twins of the GPU-side sharders below (`shard_nvfp4_col_segs`,
// `shard_nvfp4_row`). They run between `repack_nvfp4_mma` and the upload inside the loader, so the
// byte math is IDENTICAL to the post-load path it replaces — 16-row tile bands for column-parallel,
// per-tile k-block ranges for row-parallel. hy_v3's 169.6 GB cannot be loaded whole and sharded at
// attach; qwen families never take these (they fit, and keep the proven post-load path).
// -----------------------------------------------------------------------------------------------

/// Which shard (if any) a tensor gets at load. Mirrors `tp_shard_weights`'s per-site choices.
#[derive(Debug)]
pub(crate) enum LoadShardOp {
    None,
    /// Column-parallel, whole-M single segment (dense FFN gate/up, quantized LM head).
    Col,
    /// Column-parallel over output-row segments (the fused attention qkv).
    ColSegs(Vec<(usize, usize, bool)>),
    /// Row-parallel (attention o_proj, dense FFN down).
    Row,
}

// -----------------------------------------------------------------------------------------------
// Per-tensor-class shard-factor predicates. These are the SINGLE source of truth shared by the
// post-load `tp_shard_weights`, the load-time shard ops, and the `eff_*` head/dim accessors, so
// the weight sharder, the runtime kernel geometry, and the at-load slicer can never drift.
//
// "sharding is a property of the tensor, not the process" (TP_M4_NOTES): a tensor class shards by
// `/world` IFF its natural split dim is divisible by `world` at the required granularity; otherwise
// it stays REPLICATED (factor 1). `mixers` gates only the attn/GDN mixer classes — dense FFN shards
// on `world > 1` alone.
// -----------------------------------------------------------------------------------------------

/// Dense-FFN factor (col gate/up `intermediate_size`, row down K = `intermediate_size`). The
/// gate/up col split needs `intermediate_size % (16*world) == 0`; the down ROW split is the stronger
/// constraint — `shard_nvfp4_row` asserts the LOCAL K is 32-aligned (`(k/world) % 32 == 0`, the fp4
/// adjacent-pair loop reads 32 contiguous k-blocks per tile) — so the combined predicate is
/// `intermediate_size % (32*world) == 0`. Using only 16*world here would let a future family pass the
/// factor check and then PANIC in the row sharder instead of replicating. All current Qwen
/// `intermediate_size` values are 512-aligned, so this is byte-identical for them (and reproduces the
/// original world=2 `k % 64 == 0` assert exactly).
pub(super) fn ffn_shard_factor_for(cfg: &Config, world: usize, _mixers: bool) -> usize {
    if world > 1 && cfg.intermediate_size % (32 * world) == 0 { world } else { 1 }
}

/// Attention factor (fused qkv per-head col + o_proj row). Bound by BOTH head counts; the binding
/// constraint on the flagships is `num_kv_heads` (2 or 4).
pub(super) fn attn_shard_factor_for(cfg: &Config, world: usize, mixers: bool) -> usize {
    if world > 1 && mixers && cfg.num_heads % world == 0 && cfg.num_kv_heads % world == 0 {
        world
    } else {
        1
    }
}

/// GDN factor (fused in_proj per-head col + conv/state + out_proj row). Bound by both GDN head counts.
pub(super) fn gdn_shard_factor_for(cfg: &Config, world: usize, mixers: bool) -> usize {
    if world > 1 && mixers && cfg.lin_num_k_heads % world == 0 && cfg.lin_num_v_heads % world == 0 {
        world
    } else {
        1
    }
}

/// LM-head vocab-parallel factor (quantized heads only on the post-load path; the at-load hy_v3 E7
/// bf16 head shard uses the same vocab divisibility). Gated on `mixers` like attention/GDN — the
/// post-load path keeps the shipped `[tp-shard-mixers]` flag as the outer gate for LM-head.
pub(super) fn vocab_shard_factor_for(cfg: &Config, world: usize, mixers: bool) -> usize {
    if world > 1 && mixers && cfg.vocab_size % (16 * world) == 0 { world } else { 1 }
}

/// MoE expert-band factor (the stacked [ne·M_e, K] column band). Gated on `mixers` — the post-load
/// path keeps the shipped flag as the outer gate for experts.
pub(super) fn experts_shard_factor_for(cfg: &Config, world: usize, mixers: bool) -> usize {
    if world > 1 && mixers && cfg.num_experts % world == 0 { world } else { 1 }
}

/// E8 shared-expert factor (the fused [2·si, h] gate+up col + [h, si] down row). Uses
/// `shared_expert_intermediate_size` — NOT `intermediate_size` (the routed expert width). Col check
/// `si % (16*world) == 0` subsumes the row check `(si/16) % world == 0`.
pub(super) fn shared_expert_shard_factor_for(cfg: &Config, world: usize, mixers: bool) -> usize {
    if world > 1 && mixers && cfg.shared_expert_intermediate_size % (16 * world) == 0 { world } else { 1 }
}

/// Key for the parallel-assembly results map: the `gwn` name-group joined by a control char that
/// never appears in tensor names. One entry per fused (or single) weight the pipeline produced.
pub(super) fn akey(names: &[String]) -> String { names.join("\u{1f}") }

/// hy_v3 main-layer stacked expert tensor? (`model.layers.<i>.mlp.experts.*`, i < num_layers).
/// The MTP block at `model.layers.80` is deliberately excluded — the loader skips it wholesale.
pub(super) fn is_hy3_main_expert(stem: &str, cfg: &Config) -> bool {
    let Some(rest) = stem.strip_prefix("model.layers.") else { return false };
    let Some((idx, tail)) = rest.split_once('.') else { return false };
    let Ok(i) = idx.parse::<usize>() else { return false };
    i < cfg.num_layers && tail.starts_with("mlp.experts.")
}

/// The load-time shard for a tensor, derived from its name(s) — the loader's `gwn` sees exactly
/// what the site requested (one name, or the three q/k/v names of a fused attention qkv).
/// Only ever called with `sal.is_some()` (hy_v3 + TP + mixer sharding); anything the plan does
/// NOT shard (router, embed, norms, expert_bias, bf16 LM head) maps to `None` and stays
/// replicated on both ranks. `quantized` tells it the tensor's actual format (the LM head
/// shards only when quantized — attach_tp's rule; hy_v3's is bf16 and stays whole).
///
/// E8 (revised, shared expert only): the shared_mlp gate/up/down are sharded like dense L0 —
/// the FUSED [2·si, h] gate+up column-parallel, the down row-parallel — so each rank reads HALF
/// the shared weights and the rank-local partial folds into the (already paid) MoE all-reduce
/// (moe_batch's fold_shared path, composed with E12's fold). The fused gate+up cannot take a
/// plain M-split (rank 0 would get all gates, rank 1 all ups, breaking the gate[i]·up[i] pairing
/// SiLU needs), so the ColSegs split PAIRS the rows: rank r keeps gate rows [r·si/2, (r+1)·si/2)
/// AND up rows [si + r·si/2, si + (r+1)·si/2) — one rank owns gate[i]·up[i] for its i-half, and
/// the local tensor is the standard [gate_local | up_local] fused layout with si_local = si/2.
/// --e8-no-shard=1 restores the replicated shared expert (the fold then runs the replicated
/// path — byte-identical to the pre-E8 bf16 output).
pub(super) fn hy3_load_shard_op(names: &[String], cfg: &Config, quantized: bool, world: usize) -> LoadShardOp {
    // E8 escape: --e8-no-shard=1 keeps the shared expert replicated (byte-identical path).
    let e8_on = crate::opts::var(crate::opt!("e8-no-shard")).map_or(true, |v| v != "1");
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    // The MTP block (layers.num_layers) is REPLICATED under TP — barrier-free drafting requires
    // full-width weights on both ranks, so no shard op applies to it.
    let layer_idx = names[0].trim_start_matches("model.layers.").split('.').next()
        .and_then(|i| i.parse::<usize>().ok());
    if layer_idx.map_or(false, |i| i >= cfg.num_layers) { return LoadShardOp::None; }
    // Fused attention qkv: [q (nh·hd·qmul) | k (nkv·hd) | v (nkv·hd)] — head-split per segment.
    // Replicate (None) when the head counts do not divide `world` (kv_heads=2 at world=4).
    if names.len() == 3 && names[0].starts_with("model.layers.")
        && names[0].ends_with("self_attn.q_proj.weight") {
        if attn_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        let qmul = if cfg.attn_out_gate() { 2 } else { 1 };
        return LoadShardOp::ColSegs(vec![
            (0, nh * hd * qmul, true),
            (nh * hd * qmul, nkv * hd, true),
            (nh * hd * qmul + nkv * hd, nkv * hd, true)]);
    }
    // E8: the SHARED expert's FUSED gate+up ([gate (si) | up (si)] rows, fused by the loader from
    // the two names) — column-parallel with PAIRED segments so gate[i]·up[i] stay on one rank
    // (see the fn comment). The local tensor keeps the fused layout with si_local = si/world rows of
    // gate + si/world of up. Quantized only: the bf16 loader path cannot slice.
    if e8_on && quantized && names.len() == 2
        && names[0].ends_with("mlp.shared_mlp.gate_proj.weight") {
        if shared_expert_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        let si = cfg.shared_expert_intermediate_size;
        return LoadShardOp::ColSegs(vec![(0, si, true), (si, si, true)]);
    }
    if names.len() != 1 { return LoadShardOp::None; }
    let n = &names[0];
    if !n.starts_with("model.layers.") {
        // Outside the layer stack: a QUANTIZED LM head shards vocab-parallel (attach_tp's rule);
        // the hy3 recipe keeps it bf16, so it stays replicated here.
        if n == "lm_head.weight" && quantized && vocab_shard_factor_for(cfg, world, true) > 1 {
            return LoadShardOp::Col;
        }
        return LoadShardOp::None;
    }
    if n.ends_with("self_attn.o_proj.weight") {
        return if attn_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    if n.ends_with(".mlp.gate_proj.weight") || n.ends_with(".mlp.up_proj.weight") {
        return if ffn_shard_factor_for(cfg, world, false) > 1 { LoadShardOp::Col } else { LoadShardOp::None };
    }
    if n.ends_with(".mlp.down_proj.weight") {
        return if ffn_shard_factor_for(cfg, world, false) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    // E8: the shared expert's DOWN is row-parallel (each rank computes 1/world of the shared output's
    // K). Quantized only; --e8-no-shard=1 keeps it replicated too.
    if e8_on && quantized && n.ends_with("mlp.shared_mlp.down_proj.weight") {
        return if shared_expert_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    if n.contains("linear_attn") {
        panic!("shard-at-load: GDN mixer sharding is not implemented at load (hy_v3 has none — \
                this is a qwen tensor reaching the hy3-only path; use the post-load shard)");
    }
    LoadShardOp::None   // router, embed, norms, expert_bias: replicated
}

/// TP=2 shard-at-load op for the QWEN family (the hy_v3 twin is `hy3_load_shard_op`). Mirrors
/// attach_tp's `tp_shard_weights` site map so the loader's host-side slicing is byte-identical
/// to the in-place shard it replaces: dense FFN gate/up col-parallel + down row-parallel; MoE
/// experts col (the contiguous expert band); fused attention qkv per-head col-segs + o_proj row;
/// fused GDN in_proj per-head col-segs + out_proj row; the LM head vocab-parallel (quantized);
/// everything else (router, shared expert, norms, embed, the MTP head) replicated.
pub(crate) fn qwen_load_shard_op(names: &[String], cfg: &Config, quantized: bool, world: usize) -> LoadShardOp {
    let op = qwen_load_shard_op_inner(names, cfg, quantized, world);
    if crate::opts::var(crate::opt!("load-shard-trace")).is_ok() {
        eprintln!("[shard-op] names0={} n_names={} quant={} world={} -> {:?}",
                  names[0], names.len(), quantized, world, op);
    }
    op
}

fn qwen_load_shard_op_inner(names: &[String], cfg: &Config, quantized: bool, world: usize) -> LoadShardOp {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let n = &names[0];
    // The MTP head is REPLICATED under TP (barrier-free drafting needs full-width weights).
    if n.starts_with("mtp.") { return LoadShardOp::None; }
    // Fused attention qkv: [q (nh·hd·qmul) | k (nkv·hd) | v (nkv·hd)] — head-split per segment.
    // Replicate (None) when the head counts do not divide `world`.
    if names.len() == 3 && n.ends_with("self_attn.q_proj.weight") {
        if attn_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        let qmul = if cfg.attn_out_gate() { 2 } else { 1 };
        return LoadShardOp::ColSegs(vec![
            (0, nh * hd * qmul, true),
            (nh * hd * qmul, nkv * hd, true),
            (nh * hd * qmul + nkv * hd, nkv * hd, true)]);
    }
    // Fused GDN in_proj: [qkv | z | b | a] — per-head col-segs; b/a carried WHOLE (their rows
    // won't split at the 16-row tile granularity — split_gdn_b slices them to the local range).
    // Replicate the whole GDN (in_proj + conv/state + out_proj) when the head counts don't divide.
    if names.len() == 4 && n.ends_with("linear_attn.in_proj_qkv.weight") {
        if gdn_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        let (lin_nk, lin_nv) = (cfg.lin_num_k_heads, cfg.lin_num_v_heads);
        let (kdim, vdim) = (cfg.key_dim(), cfg.value_dim());
        let cdim = kdim * 2 + vdim;
        let _ = lin_nk;
        return LoadShardOp::ColSegs(vec![
            (0, kdim, true), (kdim, kdim, true), (2 * kdim, vdim, true),
            (cdim, vdim, true), (cdim + vdim, lin_nv, false), (cdim + vdim + lin_nv, lin_nv, false)]);
    }
    // Mixed-dtype GDN site (foreign checkpoints: in_proj_qkv/z FP8, in_proj_b/a bf16 — the
    // [lin_nv, k] b/a rows don't tile into 128-blocks): the 4-name FUSED rule above can never
    // assemble it, so the site runs the legacy SPLIT arm and each part arrives here as a 1-name
    // group. Under TP the split arm computes LOCAL dims, so the quantized parts must shard by
    // head band exactly like the fused site — a replicated full weight would make EVERY rank
    // read the FIRST head band and the out_proj all-reduce would sum the same half twice.
    // b/a stay replicated-full: the arm GEMMs them at the full head count and the kernel reads
    // the rank's band via a bf16 pointer offset (no bf16 shard pipeline needed).
    if n.ends_with("linear_attn.in_proj_qkv.weight") && quantized {
        if gdn_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        let (kdim, vdim) = (cfg.key_dim(), cfg.value_dim());
        return LoadShardOp::ColSegs(vec![
            (0, kdim, true), (kdim, kdim, true), (2 * kdim, vdim, true)]);
    }
    if n.ends_with("linear_attn.in_proj_z.weight") && quantized {
        if gdn_shard_factor_for(cfg, world, true) <= 1 { return LoadShardOp::None; }
        return LoadShardOp::Col;
    }
    if names.len() != 1 { return LoadShardOp::None; }
    if n == "lm_head.weight" {
        return if quantized && vocab_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Col } else { LoadShardOp::None };
    }
    if n.ends_with("embed_tokens.weight") { return LoadShardOp::None; }
    if n.ends_with("self_attn.o_proj.weight") {
        return if attn_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    if n.ends_with("linear_attn.out_proj.weight") {
        return if gdn_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    if n.contains(".mlp.experts.gate_up_proj.weight") || n.contains(".mlp.experts.down_proj.weight") {
        return if experts_shard_factor_for(cfg, world, true) > 1 { LoadShardOp::Col } else { LoadShardOp::None };
    }
    if n.ends_with(".mlp.gate_proj.weight") || n.ends_with(".mlp.up_proj.weight") {
        return if ffn_shard_factor_for(cfg, world, false) > 1 { LoadShardOp::Col } else { LoadShardOp::None };
    }
    if n.ends_with(".mlp.down_proj.weight") {
        return if ffn_shard_factor_for(cfg, world, false) > 1 { LoadShardOp::Row } else { LoadShardOp::None };
    }
    // router, shared expert, norms, conv1d, a_log, dt_bias, expert_bias: replicated.
    LoadShardOp::None
}

/// Slice a FUSED RAW NVFP4 tensor (packed [m, k/2] + scales [m, k/16] + per-tile gs) to rank r's
/// TP shard by `op` — the RAW-side twin of the post-load repacked-layout sharders, byte-identical
/// to repacking then slicing (the 16-row-tile independence of both repacks; the OMMA shard
/// invariant is gated by the CPU test in src/mxfp4.rs). Economy mode NEEDS this: it never
/// materializes the standard MMA repack, so the old post-repack slice had nothing to slice.
pub(super) fn shard_raw_nvfp4(qw: &[u8], sc: &[u8], gsv: &[f32], m: usize, k: usize,
                   op: &LoadShardOp, rank: usize, world: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>, usize, usize) {
    let pw = k / 2;                 // packed bytes per row
    let nblk = k / 16;              // scale bytes per row
    match op {
        LoadShardOp::None => (qw.to_vec(), sc.to_vec(), gsv.to_vec(), m, k),
        LoadShardOp::Col => {
            let (r0, take) = col_band(m, rank, world);
            (qw[r0*pw..(r0+take)*pw].to_vec(), sc[r0*nblk..(r0+take)*nblk].to_vec(),
             gsv[r0/16..(r0+take)/16].to_vec(), take, k)
        }
        LoadShardOp::ColSegs(segs) => {
            let mut q = Vec::new(); let mut s = Vec::new(); let mut g = Vec::new();
            let mut mo = 0usize;
            for &(off, cnt, split) in segs {
                let take = if split { cnt / world } else { cnt };
                let r0 = off + if split { rank * take } else { 0 };
                q.extend_from_slice(&qw[r0*pw..(r0+take)*pw]);
                s.extend_from_slice(&sc[r0*nblk..(r0+take)*nblk]);
                g.extend_from_slice(&gsv[r0/16..(r0+take)/16]);
                mo += take;
            }
            (q, s, g, mo, k)
        }
        LoadShardOp::Row => {
            let kblk = nblk / world;
            let (kb0, kb1) = (rank * kblk, (rank + 1) * kblk);
            let mut q = Vec::with_capacity(m * (k / 2 / world));
            let mut s = Vec::with_capacity(m * kblk);
            for r in 0..m {
                q.extend_from_slice(&qw[r*pw + kb0*8..r*pw + kb1*8]);   // 16 elements = 8 packed bytes
                s.extend_from_slice(&sc[r*nblk + kb0..r*nblk + kb1]);
            }
            (q, s, gsv.to_vec(), m, k / world)   // ROW: K splits by world
        }
    }
}

/// The rows [r0, r0 + take) a column-parallel (`LoadShardOp::Col`) shard gives rank `rank` of `world`
/// out of `m` output rows — the ONE definition the load-time slicer (`shard_raw_nvfp4`) and the B32
/// shard shipper (`shard_plan`, pre-sliced NVFP4 expert bands) share.
pub(crate) fn col_band(m: usize, rank: usize, world: usize) -> (usize, usize) {
    let take = m / world;
    (rank * take, take)
}

/// RAW-side FP8 twin of `shard_raw_nvfp4` (data [m, k] + per-row f32 scales).
pub(super) fn shard_raw_fp8(qw: &[u8], rs: &[f32], m: usize, k: usize,
                 op: &LoadShardOp, rank: usize, world: usize) -> (Vec<u8>, Vec<f32>, usize, usize) {
    match op {
        LoadShardOp::None => (qw.to_vec(), rs.to_vec(), m, k),
        LoadShardOp::Col => {
            let take = m / world;
            let r0 = rank * take;
            (qw[r0*k..(r0+take)*k].to_vec(), rs[r0..r0+take].to_vec(), take, k)
        }
        LoadShardOp::ColSegs(segs) => {
            // per-segment slices, exactly like the NVFP4 twin (the fused FP8 qkv/in_proj)
            let mut q = Vec::new(); let mut r2 = Vec::new();
            let mut mo = 0usize;
            for &(off, cnt, split) in segs {
                let take = if split { cnt / world } else { cnt };
                let r0 = off + if split { rank * take } else { 0 };
                q.extend_from_slice(&qw[r0*k..(r0+take)*k]);
                r2.extend_from_slice(&rs[r0..r0+take]);
                mo += take;
            }
            (q, r2, mo, k)
        }
        LoadShardOp::Row => {
            let k0 = rank * (k / world);
            let k_local = k / world;
            let mut q = Vec::with_capacity(m * k_local);
            for r in 0..m { q.extend_from_slice(&qw[r*k + k0..r*k + k0 + k_local]); }
            (q, rs.to_vec(), m, k_local)
        }
    }
}

/// RAW-side shard twin for Qwen block-128 FP8: slices the fused RAW [M,K] codes AND the flat
/// [(M/128)*nb_k] scale matrix, preserving the w = code·scale_inv[row-band, k-block] pairing.
/// Col/ColSegs split M → scale BANDS split (asserts 128-row alignment: a 16-row tile must never
/// straddle a band, which holds for every Qwen tensor — segment offsets are head multiples of
/// 256 and rank splits are 128-multiples). Row splits K → scale k-blocks split. Byte-identical
/// contract to shard_raw_fp8: the caller repacks to MMA order afterwards.
pub(super) fn shard_raw_fp8_blk(qw: &[u8], sc: &[f32], m: usize, k: usize, nb_k: usize,
                     op: &LoadShardOp, rank: usize, world: usize)
                     -> (Vec<u8>, Vec<f32>, usize, usize, usize) {
    let bands = m / 128;
    let seg_pick = |rows: &[(usize, usize)]| -> (Vec<u8>, Vec<f32>, usize) {
        let mut q = Vec::new(); let mut s = Vec::new(); let mut mo = 0usize;
        for &(r0, take) in rows {
            assert!(r0 % 128 == 0 && take % 128 == 0,
                    "fp8-blk shard segment (off {r0}, take {take}) not 128-aligned — tiles would straddle scale bands");
            q.extend_from_slice(&qw[r0 * k..(r0 + take) * k]);
            s.extend_from_slice(&sc[(r0 / 128) * nb_k..((r0 + take) / 128) * nb_k]);
            mo += take;
        }
        (q, s, mo)
    };
    match op {
        LoadShardOp::None => (qw.to_vec(), sc.to_vec(), m, k, nb_k),
        LoadShardOp::Col => {
            let take = m / world;
            let (q, s, mo) = seg_pick(&[(rank * take, take)]);
            (q, s, mo, k, nb_k)
        }
        LoadShardOp::ColSegs(segs) => {
            let mut rows = Vec::new();
            for &(off, cnt, split) in segs {
                let take = if split { cnt / world } else { cnt };
                let r0 = off + if split { rank * take } else { 0 };
                rows.push((r0, take));
            }
            let (q, s, mo) = seg_pick(&rows);
            (q, s, mo, k, nb_k)
        }
        LoadShardOp::Row => {
            assert!(nb_k % world == 0, "fp8-blk row shard: nb_k={nb_k} won't split across world {world}");
            let k0 = rank * (k / world);
            let k_local = k / world;
            let nb_local = nb_k / world;
            let kb0 = rank * nb_local;
            let mut q = Vec::with_capacity(m * k_local);
            for r in 0..m { q.extend_from_slice(&qw[r * k + k0..r * k + k0 + k_local]); }
            let mut s = Vec::with_capacity(bands * nb_local);
            for b in 0..bands { s.extend_from_slice(&sc[b * nb_k + kb0..b * nb_k + kb0 + nb_local]); }
            (q, s, m, k_local, nb_local)
        }
    }
}

/// Host-side twin of `shard_nvfp4_col_segs`: slice 16-row tile bands out of the MMA-repacked
/// (wt, st, gsv) triple. Returns the sharded triple + local M.
pub(crate) fn host_shard_nvfp4_col_segs(wt: &[u8], st: &[u8], gsv: &[f32], k: usize,
                             segs: &[(usize, usize, bool)], rank: usize, world: usize)
    -> (Vec<u8>, Vec<u8>, Vec<f32>, usize) {
    let nblk = k / 16;
    let qpt = nblk * 128;            // qweight bytes per output tile
    let spt = nblk * 16;             // scale bytes per output tile
    let (mut q_new, mut s_new, mut g_new) = (Vec::new(), Vec::new(), Vec::new());
    let mut m_local = 0usize;
    for &(off, cnt, split) in segs {
        assert!(off % 16 == 0, "load col-shard seg (off {off}) not tile-aligned");
        let take = if split { cnt / world } else { cnt };
        if split { assert!(cnt % (16 * world) == 0, "load col-shard seg (off {off}, cnt {cnt}) won't split at 16-row tiles across world {world}"); }
        else     { assert!(cnt % 16 == 0, "load col-shard unsplit seg (off {off}, cnt {cnt}) not tile-aligned"); }
        let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
        let ntl = take / 16;
        q_new.extend_from_slice(&wt[tile_lo * qpt..(tile_lo + ntl) * qpt]);
        s_new.extend_from_slice(&st[tile_lo * spt..(tile_lo + ntl) * spt]);
        g_new.extend_from_slice(&gsv[tile_lo..tile_lo + ntl]);
        m_local += take;
    }
    (q_new, s_new, g_new, m_local)
}

/// Host-side twin of `shard_nvfp4_row`: keep k-blocks [rank·nblk/world, +nblk/world) of every output tile.
/// gs is per output tile (K-independent) → copied unchanged. Returns (wt, st, k_local).
pub(crate) fn host_shard_nvfp4_row(wt: &[u8], st: &[u8], m: usize, k: usize, rank: usize, world: usize)
    -> (Vec<u8>, Vec<u8>, usize) {
    assert!((k / 16) % world == 0 && (k / world) % 32 == 0, "load row-shard: k={k} won't split into 32-aligned slices across world {world}");
    let k_local = k / world;
    let nblk_full = k / 16;
    let nblk_local = k_local / 16;
    let tiles = m / 16;
    let kb_lo = rank * nblk_local;
    let mut q_new = Vec::with_capacity(tiles * nblk_local * 128);
    let mut s_new = Vec::with_capacity(tiles * nblk_local * 16);
    for mt in 0..tiles {
        let q_src = (mt * nblk_full + kb_lo) * 128;
        q_new.extend_from_slice(&wt[q_src..q_src + nblk_local * 128]);
        let s_src = (mt * nblk_full + kb_lo) * 16;
        s_new.extend_from_slice(&st[s_src..s_src + nblk_local * 16]);
    }
    (q_new, s_new, k_local)
}

// ---- Launch-argument bit-packings (public PR #21) -------------------------------------------------
// Several kernels take a geometry tuple packed into one word (the 12-argument launch cap). The fields
// are fixed-width and the kernels unpack them with fixed masks, so a value one past its width lands in
// the NEIGHBOURING field: a wrong cache stride, i.e. reads and writes at the wrong device addresses.
// These checks were debug_assert! (compiled out of every served build); they are release asserts now,
// and each packing has ONE function so the layout and its bound cannot drift apart.
