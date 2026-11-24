//! DSV4.1 CPU attention oracle (leg OA): DeepSeek-V4.1's attention sublayer in Rust f32 / f64,
//! following `inference/model.py` and REFb's emulation of `kernel.py` op by op, checked layer by
//! layer against REFb's reference dumps. Reference-grade and single-box: it is the yardstick the
//! GPU kernels (KC, KI, KA) and the slice / TP models compare against, never a served path.
//!
//! What it covers (PLAN/DSV41_REF_MAP.md §2.2-2.8, §3.3-3.5, §4):
//! * the q LoRA path (`wq_a` 5120 -> 1280, RMSNorm, `wq_b` 1280 -> 64 x 512) and the window KV path
//!   (`wkv`, RMSNorm, RoPE on the last 64 dims, FP8 round trip over all 512 dims);
//! * RoPE: theta 10,000 for ratio-0 layers, compress theta 160,000 with YaRN x16 elsewhere;
//! * the Compressor (ratio 2: fp32 `wkv` / `wgate`, softmax pooling over each group per channel,
//!   the open group held in `kv_state` / `score_state`; ratio 1: bf16 `wkv`), RMSNorm, RoPE at the
//!   group's first position, FP4 E2M1 with E4M3 scales per 16 (round-to-nearest-even);
//! * the KV-source / reuse structure (`kv_source_layer_ids` 2 / 8 / 14 / 20) and the index-key
//!   owners (FP4 with E8M0 scales per 32);
//! * the Indexer (32 heads x 128, top-512, bf16 score chain) at the 8 index sources, the layer-20
//!   candidate blocks (top 2,048 blocks of 8, newest block pinned) and their use at 24 / 28 / 32 / 36;
//! * SWA 128 (prefill window lists and the decode ring order), the sparse attention (online softmax
//!   in blocks of 64 in index order, P cast to bf16 before P.V, the sink in the denominator);
//! * the `o_groups` 8 output: inverse RoPE, `act_quant` on WO-A's input, `wo_a` per group (bf16),
//!   `wo_b` (FP8), and the TP form of `wo_b` (fp32 partials per o_group range, rank-order sum).
//!
//! Files: `attn_num.rs` (formats, quantisers, linears, threads), `attn_rope.rs`, `attn_weights.rs`
//! (native checkpoint reader), `attn_layer.rs` (the layer and its state), `attn_fixture.rs` (REFb
//! dumps, comparisons, the probe), `attn_tests.rs`.
//!
//! Mode: `--probe-dsv41-oracle --dsv41-oracle-part attn --model-dir <native checkpoint>
//! --dsv41-dump-ref <REFb fixture dir> [--layers 0,1,2,...]`. Prints one block per layer and
//! `DSV41_ORACLE_ATTN_OK` when every layer is within the stated tolerances.

#[path = "attn_num.rs"]
pub mod num;
#[path = "attn_rope.rs"]
pub mod rope;
#[path = "attn_weights.rs"]
pub mod weights;
#[path = "attn_layer.rs"]
pub mod layer;
#[path = "attn_fixture.rs"]
pub mod fixture;
#[cfg(test)]
#[path = "attn_tests.rs"]
mod tests;

pub use layer::{LayerIn, LayerOut, Oracle, OracleCfg, Order, SrcState, TopK};

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Parse a layer list: `0,1,2`, `20-24`, `all` (= every layer the dump has).
pub fn parse_layers(s: &str) -> Result<Option<Vec<usize>>> {
    if s == "all" {
        return Ok(None);
    }
    let mut v = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (usize, usize) = (a.trim().parse().context("--layers")?, b.trim().parse().context("--layers")?);
            v.extend(a..=b);
        } else {
            v.push(part.trim().parse().context("--layers")?);
        }
    }
    Ok(Some(v))
}

/// The `--dsv41-oracle-part attn` mode. Returns Ok(()) only when every layer passes.
pub fn probe_cli(args: &[String]) -> Result<()> {
    let model = crate::arg_value(args, "--model-dir").map(PathBuf::from).context("--model-dir <native checkpoint> is required")?;
    let dump = crate::opts::lookup("dsv41-dump-ref").and_then(crate::opts::get).map(PathBuf::from)
        .context("--dsv41-dump-ref <REFb fixture dir> is required (e.g. /mnt/models/dsv41/ref/short_chat)")?;
    let layers = match crate::arg_value(args, "--layers") { Some(s) => parse_layers(s)?, None => None };
    let r = fixture::probe(&model, &dump, layers)?;
    for l in &r.lines {
        println!("{l}");
    }
    anyhow::ensure!(r.ok, "at least one layer is out of tolerance (see above)");
    println!("DSV41_ORACLE_ATTN_OK");
    Ok(())
}
