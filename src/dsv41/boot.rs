//! DSV4.1 (leg I0): the rows the shared startup configuration block (src/startup.rs) prints for a
//! V4.1 pack — model structure (attention classes, CSA sources, Engram, DSpark), the weight
//! formats and bits from the census, and the TP split from the rank map. `startup::config_lines`
//! calls `config_rows` once (one line there); every other family's block is unchanged.

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::rankmap::{self, Posture};
use std::path::Path;

fn ranges(v: &[usize]) -> String {
    let u: Vec<u16> = v.iter().map(|&x| x as u16).collect();
    crate::dsv41::census::report::layer_ranges(&u)
}

/// The structure rows (pure: from the config alone).
pub fn structure_rows(c: &Dsv41Config) -> Vec<(String, String)> {
    let by_ratio = |r: u32| -> Vec<usize> { (0..c.layers).filter(|&l| c.ratio(l) == r).collect() };
    let mut att = format!("SWA-only {}", ranges(&by_ratio(0)));
    for r in [2u32, 1] {
        let ls = by_ratio(r);
        if ls.is_empty() { continue; }
        let src: Vec<usize> = c.kv_sources.iter().copied().filter(|s| c.ratio(*s) == r).collect();
        att.push_str(&format!("; ratio-{r} {} (KV sources {:?})", ranges(&ls), src));
    }
    att.push_str(&format!("; index sources {:?} top-{}; candidates from layer {} ({} blocks x {}); window {} (ring {} with backtracking); {} output groups",
                          c.index_sources, c.index_topk, c.candidate_source.map_or("-".into(), |x| x.to_string()),
                          c.candidate_topk_blocks, c.candidate_block, c.window, c.swa_slots(), c.o_groups));
    if let Some(d) = c.first_decoder_layer() {
        att.push_str(&format!("; CED: encoder 0-{}, decoder {d}-{} (layer {d}'s compressor + index K on every prompt row)", d - 1, c.layers - 1));
    }
    let tables = match &c.engram_table_dir {
        Some(d) => format!("tables from {d}"),
        None => "tables: DeepSeek's FP8 tables (native shards), owned by head per rank".into(),
    };
    let mut rows = vec![
        ("dsv41 model".into(), c.summary()),
        ("attention".into(), att),
        ("engram".into(), format!("layers {:?}, {} heads x n-gram orders 2..={} ({} rows per token per layer), {}",
                                  c.engram_layers, c.engram_heads, c.engram_max_ngram, c.engram_cols(), tables)),
        ("dspark".into(), format!("{} stages, block {}, targets {:?}, {} experts top-{}, Markov rank {}",
                                  c.dspark_stages, c.dspark_block, c.dspark_targets, c.dspark_routed, c.dspark_top_k, c.dspark_markov_rank)),
        ("hc / precision".into(), format!("hc {} (Sinkhorn {}, eps {:e}), rms eps {:e}, SwiGLU limit {}, routed scale {}",
                                          c.hc, c.sinkhorn_iters, c.hc_eps, c.rms_eps, c.swiglu_limit, c.routed_scale)),
    ];
    if let Some(v) = c.vision.as_ref().filter(|v| v.layers > 0) {
        let state = if crate::dsv41::vision::VisionMode::from_registry() == crate::dsv41::vision::VisionMode::Off {
            format!("ViT {}L + {}x{} aligner (OFF via --dsv41-vision off)", v.layers, v.downsample, v.downsample)
        } else {
            format!("ViT {}L + {}x{} aligner (BF16 on rank 0, max {} tokens/image)", v.layers, v.downsample, v.downsample, v.max_image_tokens)
        };
        rows.push(("vision".into(), state));
    }
    rows
}

/// The split row for `--tp W` (None for TP=1 or an unknown width): T0's split (PLAN/DSV41_TP_DESIGN.md §1) as the TP forward
/// runs it, the expert split from `--dsv41-expert-split` (inner width by default, D-DSV41-3; `ep` = whole experts).
pub fn split_row(c: &Dsv41Config, world: usize) -> Option<(String, String)> {
    use crate::dsv41::load::splits;
    let p = Posture::from_world(world)?;
    if p == Posture::Tp1 { return None; }
    let m = rankmap::rank_map(c, p);
    let heads: Vec<String> = m.iter().map(|s| format!("{:?}", s.engram_heads)).collect();
    let ep = crate::dsv41::load::ship::deal() == "ep";
    let blocks: Vec<String> = (0..world).map(|r| splits::part(c.moe_inter as u64 / 128, world, r, 0).1.to_string()).collect();
    let experts = if ep { format!("EP {} whole experts / rank (DSpark {}; --dsv41-expert-split ep)", m[0].experts.len(), m[0].dspark_experts.len()) }
                  else { format!("experts at inner width 1/{world} (every expert on every rank, 128-blocks {} for even ids; shared expert the same)", blocks.join("+")) };
    let (_, ng) = splits::part(c.o_groups as u64, world, 0, 0);
    let nh = ng * (c.heads / c.o_groups) as u64;
    let v = splits::vocab_range(c.vocab, world, 0);
    // what the plan holds whole on every rank and the forward cuts at call time (TP4 / TP8 by default: `--dsv41-attn-whole auto`)
    let aw = crate::opts::lookup("dsv41-attn-whole").and_then(crate::opts::get).unwrap_or_else(|| "auto".into());
    let whole = crate::dsv41::load::plan::PlanOpts::resolve(world, crate::dsv41::load::plan::ExpertSplit::Inner, &aw, false)
        .map(|o| held_whole_note(o.attn_whole, o.engram_wkv_whole, !o.embed_rank0_only)).unwrap_or_default();
    Some(("dsv41 split".into(), format!("{}: {experts}; attention {nh} heads / {ng} o_groups per rank (FP32 wo_b partials); hc_fn K / {world} riding the sublayer reduce; \
        vocab {} / rank (whole 128-row blocks); Engram heads {} (wkv partials ride the FFN reduce of the layer before){whole}", p.name(), v.len(), heads.join(" "))))
}

/// The split row's note on the weights every rank holds WHOLE (the forward cuts them to the rank's heads / groups / K
/// slice / Engram columns at call time), and where the embedding lives.
pub fn held_whole_note(attn_whole: bool, wkv_whole: bool, embed_everywhere: bool) -> String {
    let mut w = Vec::new();
    if attn_whole { w.push("wq_b / wo_a / wo_b / sinks / hc_fn"); }
    if wkv_whole { w.push("Engram wkv"); }
    let mut s = String::new();
    if !w.is_empty() { s.push_str(&format!("; held whole on every rank and cut at call time: {}", w.join(", "))); }
    if embed_everywhere { s.push_str("; embedding on every rank"); } else { s.push_str("; embedding on rank 0 (its rows ride the fp32 reduce)"); }
    s
}

/// All dsv41 rows for the configuration block (None when `dir` is not a V4.1 pack). Reads the
/// config and the census of the pack's headers (formats and bits); a census error becomes a row.
pub fn config_rows(dir: &Path, tp_world: Option<usize>) -> Option<Vec<(String, String)>> {
    if !crate::dsv41::config::is_dsv41_dir(&dir.to_string_lossy()) { return None; }
    let mut rows = Vec::new();
    match Dsv41Config::load(dir) {
        Err(e) => { rows.push(("dsv41 config".into(), format!("UNREADABLE: {e:#}"))); return Some(rows); }
        Ok(c) => {
            rows.extend(structure_rows(&c));
            if let Err(e) = c.validate() { rows.push(("dsv41 config".into(), format!("INVALID: {e:#}"))); }
            match crate::dsv41::census::Census::build(dir) {
                Ok(cs) => {
                    let probs = if cs.problems.is_empty() { String::new() } else { format!(" ({} census problems: run --dsv41-census)", cs.problems.len()) };
                    rows.push(("weight formats".into(), format!("{}{probs}", cs.weights_line())));
                }
                Err(e) => rows.push(("weight formats".into(), format!("census failed: {e:#}"))),
            }
            if let Some(r) = tp_world.and_then(|w| split_row(&c, w)) { rows.push(r); }
        }
    }
    Some(rows)
}

/// The "weights" row for a V4.1 pack: the shared description with its "MTP head N bits" named for what it is here. The pack's
/// `mtp_bits` covers the 46 MTP / DSpark dense tensors, and this engine speculates with the DSpark drafter (no MTP head runs).
pub fn weights_row(dir: &Path, desc: String) -> String {
    if !crate::dsv41::config::is_dsv41_dir(&dir.to_string_lossy()) { return desc; }
    desc.replace("MTP head ", "DSpark drafter ")
}

/// Why speculation is off on this boot, if it is: `--mtp off` (also the older `0` / `false`), or `--dsv41-draft-len 0`.
pub fn spec_off_reason(mtp: Option<&str>, draft_len: Option<&str>) -> Option<&'static str> {
    if crate::dsv41::gpu::dspark::policy::effective_len("auto", mtp) == "0" { return Some("--mtp off"); }
    if draft_len.map(str::trim) == Some("0") { return Some("--dsv41-draft-len 0"); }
    None
}

/// The "speculation" row of the configuration block for a V4.1 pack (None for any other model): the source is DSpark (the
/// pack's own drafter, never an MTP head or a DFlash2 drafter), the draft-length policy, and the state. Pure of the flags.
pub fn speculation_row(dir: &Path, mtp: Option<&str>, draft_len: Option<&str>) -> Option<String> {
    if !crate::dsv41::config::is_dsv41_dir(&dir.to_string_lossy()) { return None; }
    let c = Dsv41Config::load(dir).ok();
    let drafter = c.as_ref().map_or("DSpark drafter".to_string(), |c| format!("DSpark drafter ({} stages, block {}, in the checkpoint)", c.dspark_stages, c.dspark_block));
    let policy = match draft_len.map(str::trim).unwrap_or("auto") {
        "auto" => "draft length auto (the length that maximises lanes x expected tokens per round time, from round costs measured at boot and acceptance measured online)".to_string(),
        "0" => "draft length 0".to_string(),
        k => format!("draft length pinned at {k} (--dsv41-draft-len)"),
    };
    let state = match spec_off_reason(mtp, draft_len) {
        Some(why) => format!("OFF ({why}): plain decoding, no drafts"),
        None => "ON (greedy lossless; sampled requests distribution-exact)".to_string(),
    };
    Some(format!("source {drafter}; {policy}; {state}. --mtp off or --dsv41-draft-len 0 turns it off; --spec-source and --draft-dir do not apply"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::test_config_json;

    #[test]
    fn dsv41_boot_rows() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        let r = structure_rows(&c);
        let att = &r.iter().find(|(k, _)| k == "attention").unwrap().1;
        assert!(att.starts_with("SWA-only 0-1; ratio-2 2-19 (KV sources [2, 8, 14]); ratio-1 20-39 (KV sources [20])"), "{att}");
        let vis = &r.iter().find(|(k, _)| k == "vision").unwrap().1;
        assert_eq!(vis, "ViT 32L + 3x3 aligner (BF16 on rank 0, max 1024 tokens/image)");
        assert!(r.iter().all(|(k, _)| k.len() <= 14), "keys fit the block's column");
        let s = split_row(&c, 4).unwrap().1;
        assert!(s.starts_with("TP4: experts at inner width 1/4 (every expert on every rank, 128-blocks 5+4+5+4 for even ids"), "{s}");
        assert!(s.contains("attention 16 heads / 2 o_groups per rank") && s.contains("vocab 32384 / rank") && s.contains("Engram heads 0..2 2..4 4..6 6..8"), "{s}");
        assert!(s.contains("held whole on every rank and cut at call time: wq_b / wo_a / wo_b / sinks / hc_fn, Engram wkv; embedding on every rank"), "TP4 holds attention and wkv whole: {s}");
        let s2 = split_row(&c, 2).unwrap().1;
        assert!(!s2.contains("held whole") && s2.ends_with("embedding on rank 0 (its rows ride the fp32 reduce)"), "TP2 cuts attention at load: {s2}");
        assert_eq!(held_whole_note(false, true, true), "; held whole on every rank and cut at call time: Engram wkv; embedding on every rank");
        // TP1 returns None by definition; every supported shared world (2,3,4,6,8) has a row.
        assert!(split_row(&c, 1).is_none() && split_row(&c, 5).is_none() && split_row(&c, 7).is_none());
        assert!(config_rows(Path::new("/nonexistent-dsv41"), None).is_none());
    }

    #[test]
    fn uxf_speculation_is_described_truthfully() {
        // a non-V4.1 dir is untouched
        assert!(speculation_row(Path::new("/nonexistent-dsv41"), None, None).is_none());
        assert_eq!(weights_row(Path::new("/nonexistent-dsv41"), "EXL3 2.9 bpw (lm_head 6 bits, MTP head 4 bits)".into()), "EXL3 2.9 bpw (lm_head 6 bits, MTP head 4 bits)");
        assert_eq!(spec_off_reason(None, None), None);
        assert_eq!(spec_off_reason(Some("auto"), Some("auto")), None);
        assert_eq!(spec_off_reason(Some("on"), Some("3")), None);
        assert_eq!(spec_off_reason(Some("off"), None), Some("--mtp off"));
        assert_eq!(spec_off_reason(Some("off"), Some("3")), Some("--mtp off"));
        assert_eq!(spec_off_reason(None, Some("0")), Some("--dsv41-draft-len 0"));
    }
}
