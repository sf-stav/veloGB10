//! Weight format enum, tensor groups, and the `--fake-quant` recipe / layer-rule parsing.
//! Siblings: `mod.rs` has the codecs the formats name; `mod.rs` holds the directory map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Tensor grouping + the --fake-quant spec
// ---------------------------------------------------------------------------------------------

/// Which family a tensor belongs to, for the "can this survive 4 bits?" experiment.
///
/// NOTE for the 122B MoE (Phase G): the same "must stay high precision" folklore is told about MoE
/// **router** layers. It does not apply to this dense family — there is no router — but when the MoE
/// lands, add a `Router` group here and put the claim through the same test rather than inheriting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group { Mlp, Attn, Gdn, LmHead, Embed, Mtp, Router, Expert, Hc, Ple, PleTable, Other }

pub fn group_of(name: &str) -> Group {
    // Routing gates FIRST (before the mtp/embed/mlp catch-alls) so `-router` holds BOTH the main AND the
    // MTP-head routers/shared-gates bf16. A quantized MTP router mis-routes the DRAFTER → tanks
    // speculative acceptance while output quality stays perfect (verify corrects it) — the invisible bug.
    if name.contains(".mlp.gate.weight") || name.contains(".shared_expert_gate.") { return Group::Router; }
    if name.contains(".router.") { return Group::Router; }   // hy_v3: mlp.router.gate.weight
    if name.starts_with("mtp.") { return Group::Mtp; }
    if name.contains(".eh_proj") { return Group::Mtp; }   // hy_v3: layer-80 MTP embed→hidden projection
    if name.contains("lm_head") { return Group::LmHead; }
    if name.contains("embed_tokens") { return Group::Embed; }
    // qwen4_exp (Qwen3.8-Flash-Next). The PLE n-gram table is NOT a GEMM weight (an embedding
    // gathered by row) — its own group so the quantizer can route it to the row-record codec. The
    // PLE projections (key_proj [hc*h, ple_dim], value_proj [h, ple_dim]) and the hyper-connection
    // mixers (input_mix_weight_down [lowrank, hc*h] / _up [hc*h, lowrank]) are ordinary GEMMs.
    // Checked BEFORE `.mlp.`: `mlp_hyper_connection` must land in Hc, not Mlp.
    if name.contains(".ngram_embedding.") { return Group::PleTable; }
    if name.contains(".ple.") { return Group::Ple; }
    if name.contains("hyper_connection") { return Group::Hc; }
    // MoE — test BEFORE the generic `.mlp.`: the stacked routed experts are their own group (the
    // sparse-quant risk lands here).
    if name.contains(".mlp.experts.") { return Group::Expert; }
    if name.contains(".mlp.") { return Group::Mlp; }   // dense MLP + the shared_expert MLP
    if name.contains(".self_attn.") { return Group::Attn; }
    if name.contains(".linear_attn.") {
        // Only the projections. conv1d / A_log / dt_bias / norm take the f32 path and are tiny.
        if name.contains("in_proj") || name.contains("out_proj") { return Group::Gdn; }
    }
    Group::Other
}

pub fn group_name(g: Group) -> &'static str {
    match g {
        Group::Mlp => "mlp", Group::Attn => "attn", Group::Gdn => "gdn",
        Group::LmHead => "lmhead", Group::Embed => "embed", Group::Mtp => "mtp",
        Group::Router => "router", Group::Expert => "expert", Group::Other => "other",
        Group::Hc => "hc", Group::Ple => "ple", Group::PleTable => "pletable",
    }
}

/// Parse `--fake-quant` into a per-group format map.
///
/// Tokens are `group[:fmt]`, or `-group` to drop one. `fmt` is `nvfp4` (default) or `fp8`.
/// ```text
///   all                 every group at NVFP4
///   all,gdn:fp8         NVFP4 everywhere, but FP8 for the sensitive GDN projections  <-- the recipe
///   all,-gdn            NVFP4 everywhere, GDN left in bf16
///   mlp,attn            what the HF checkpoints actually do
/// ```
pub fn fake_quant_spec() -> Option<Vec<(Group, Fmt)>> {
    let spec = crate::opts::var(crate::opt!("fake-quant")).ok()?;
    parse_recipe(&spec)
}

/// The per-LAYER override half of a recipe spec (the same env var / `--recipe` string): tokens
/// `layers:<sel>:<fmt>` where `<sel>` is `lo-hi` (inclusive range) or a comma list of indices.
/// Layer rules WIN over the group map for the trunk layers they cover (last rule wins on
/// overlap). Used by the S5F2 clean-tap ladder (L1a: `layers:5,19,33,47,61:fp8` on top of
/// `all`; L1b: `layers:0-61:bf16` on top of `all` — the bf16 override = no fake-quant on the
/// clean prefix). `fmt:bf16` in a layer rule means "keep this layer at the raw bf16 weights".
pub fn fake_quant_layer_rules() -> Vec<LayerRule> {
    match crate::opts::var(crate::opt!("fake-quant")) {
        Ok(spec) => parse_layer_rules(&spec),
        Err(_) => Vec::new(),
    }
}

/// One per-layer precision override: trunk layers `[lo, hi]` (inclusive) at `fmt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRule { pub lo: usize, pub hi: usize, pub fmt: Fmt }

/// Parse the `layers:` tokens out of a recipe spec string, in spec order (later wins).
/// `layers:0-61:bf16` (range) or `layers:5,19,33,47,61:fp8` (comma list).
pub fn parse_layer_rules(spec: &str) -> Vec<LayerRule> {
    let mut rules = Vec::new();
    for tok in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let tok = tok.strip_prefix('-').unwrap_or(tok);
        let Some(rest) = tok.strip_prefix("layers:") else { continue };
        let (sel, fmt) = match rest.rsplit_once(':') {
            Some((s, f)) => (s, f),
            None => continue,
        };
        let fmt = match fmt {
            "fp8" => Fmt::Fp8,
            "nvfp4" => Fmt::Nvfp4,
            "bf16" => Fmt::Bf16,
            _ => { eprintln!("--fake-quant: unknown layer format {:?}", fmt); std::process::exit(1); }
        };
        // The selector is `lo-hi` (range) or a list separated by `,` or `+` (the `+` form
        // keeps the whole token comma-free so the recipe spec tokenizes cleanly).
        for part in sel.split([',', '+']) {
            let part = part.trim();
            if let Some((lo, hi)) = part.split_once('-') {
                let (lo, hi): (usize, usize) = match (lo.parse(), hi.parse()) {
                    (Ok(a), Ok(b)) if a <= b => (a, b),
                    _ => { eprintln!("--fake-quant: bad layer range {:?}", part); std::process::exit(1); }
                };
                rules.push(LayerRule { lo, hi, fmt });
            } else if let Ok(li) = part.parse::<usize>() {
                rules.push(LayerRule { lo: li, hi: li, fmt });
            } else {
                eprintln!("--fake-quant: bad layer selector {:?}", part);
                std::process::exit(1);
            }
        }
    }
    rules
}

/// The trunk layer index in a weight name (`...layers.<N>.<rest>` — both the qwen3.5
/// `model.language_model.layers.N.*` and the hy_v3 `model.layers.N.*` roots), or None.
pub fn layer_index_of(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let idx = stem.rfind("layers.")?;
    let rest = &stem[idx + "layers.".len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() { return None; }
    digits.parse().ok()
}

/// Layer-aware format selection: a matching per-layer rule wins over the group map; otherwise
/// the group map decides (`fmt_for`). `fmt:bf16` from a rule returns Bf16 = "no fake-quant".
pub fn fmt_for_layer(map: &[(Group, Fmt)], rules: &[LayerRule], name: &str) -> Fmt {
    if let Some(li) = layer_index_of(name) {
        for r in rules.iter().rev() {
            if li >= r.lo && li <= r.hi { return r.fmt; }
        }
    }
    fmt_for(map, name)
}

/// Parse a recipe string into a per-group format map. `None` means "no quantization".
pub fn parse_recipe(spec: &str) -> Option<Vec<(Group, Fmt)>> {
    let spec = spec.trim().to_string();
    if spec.is_empty() || spec == "off" || spec == "none" { return None; }
    let all = [Group::Mlp, Group::Attn, Group::Gdn, Group::LmHead, Group::Embed, Group::Mtp,
               Group::Router, Group::Expert, Group::Hc, Group::Ple, Group::PleTable];
    let mut map: Vec<(Group, Fmt)> = Vec::new();
    for tok in spec.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if tok.strip_prefix('-').unwrap_or(tok).starts_with("layers:") { continue; } // per-layer overrides
        let (neg, tok) = match tok.strip_prefix('-') { Some(r) => (true, r), None => (false, tok) };
        let (name, fmt) = match tok.split_once(':') {
            Some((n, "fp8")) => (n, Fmt::Fp8),
            Some((n, "nvfp4")) => (n, Fmt::Nvfp4),
            Some((n, "bf16")) => (n, Fmt::Bf16),
            Some((_, f)) => { eprintln!("--fake-quant: unknown format {:?}", f); std::process::exit(1); }
            None => (tok, Fmt::Nvfp4),
        };
        let groups: Vec<Group> = match name {
            "all" => all.to_vec(),
            "mlp" => vec![Group::Mlp],
            "attn" => vec![Group::Attn],
            "gdn" => vec![Group::Gdn],
            "lmhead" => vec![Group::LmHead],
            "embed" => vec![Group::Embed],
            "mtp" => vec![Group::Mtp],
            "router" => vec![Group::Router],
            "expert" => vec![Group::Expert],
            "hc" => vec![Group::Hc],
            "ple" => vec![Group::Ple],
            "pletable" => vec![Group::PleTable],
            other => { eprintln!("--fake-quant: unknown group {:?}", other); std::process::exit(1); }
        };
        for g in groups {
            map.retain(|&(x, _)| x != g);
            if !neg && fmt != Fmt::Bf16 { map.push((g, fmt)); }
        }
    }
    Some(map)
}

/// The format chosen for a tensor, or Bf16 if it is not in scope.
pub fn fmt_for(map: &[(Group, Fmt)], name: &str) -> Fmt {
    let g = group_of(name);
    map.iter().find(|&&(x, _)| x == g).map(|&(_, f)| f).unwrap_or(Fmt::Bf16)
}

pub fn fmt_name(f: Fmt) -> &'static str {
    match f { Fmt::Bf16 => "bf16", Fmt::Fp8 => "fp8", Fmt::Nvfp4 => "nvfp4" }
}

