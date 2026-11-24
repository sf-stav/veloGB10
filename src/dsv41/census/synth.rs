//! Test-only: synthetic V4.1 packs (header-only safetensors with sparse data holes), in any mix of
//! formats, so the census / slice / weight-plan tests run without a model directory.

use super::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::st::{self, StOut, StSource};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// The format a synthetic linear is written in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SFmt { Exl3(u8), Plain(&'static str), Fp8, Mxfp4 }

/// A reduced-expert V4.1 config (real shapes otherwise): `n_routed` / `dspark_routed` experts.
pub fn small_config(n_routed: usize, dspark_routed: usize) -> Value {
    let mut v = crate::dsv41::config::test_config_json();
    v["text_config"]["n_routed_experts"] = n_routed.into();
    v["text_config"]["dspark_n_routed_experts"] = dspark_routed.into();
    v["text_config"]["num_experts_per_tok"] = n_routed.min(6).into();
    v["text_config"]["dspark_num_experts_per_tok"] = dspark_routed.min(3).into();
    v.as_object_mut().unwrap().remove("vision_config"); // synthetic packs carry no vision tower
    v
}

/// Every module of layer slot `l` as (key, class, sub). Mirrors the census completeness rules.
pub fn layer_modules(cfg: &Dsv41Config, l: usize, wo_a_slices: bool) -> Vec<(String, C, Option<u16>)> {
    let p = if cfg.is_dspark(l) { format!("mtp.{}.", l - cfg.layers) } else { format!("layers.{l}.") };
    let mut v: Vec<(String, C, Option<u16>)> = Vec::new();
    let mut add = |k: &str, c: C, s: Option<u16>| v.push((format!("{p}{k}"), c, s));
    for (k, c) in [("attn.attn_sink", C::AttnSink), ("attn_norm", C::AttnNorm), ("attn.q_norm", C::AttnQNorm), ("attn.kv_norm", C::AttnKvNorm),
                   ("attn.wq_a", C::AttnWqA), ("attn.wq_b", C::AttnWqB), ("attn.wkv", C::AttnWkv), ("attn.wo_b", C::AttnWoB), ("ffn_norm", C::FfnNorm),
                   ("hc_attn_fn", C::HcAttnFn), ("hc_attn_base", C::HcAttnBase), ("hc_attn_scale", C::HcAttnScale), ("hc_ffn_fn", C::HcFfnFn),
                   ("hc_ffn_base", C::HcFfnBase), ("hc_ffn_scale", C::HcFfnScale), ("ffn.gate", C::GateWeight), ("ffn.gate.bias", C::GateBias),
                   ("ffn.gate.bias_vl", C::GateBiasVl), ("ffn.shared_experts.w1", C::SharedW1), ("ffn.shared_experts.w2", C::SharedW2),
                   ("ffn.shared_experts.w3", C::SharedW3)] { add(k, c, None); }
    if wo_a_slices { for g in 0..cfg.o_groups { add(&format!("attn.wo_a.slice.{g}"), C::AttnWoA, Some(g as u16)); } } else { add("attn.wo_a", C::AttnWoA, None); }
    for e in 0..cfg.moe_of(l).0 {
        for (w, c) in [("w1", C::ExpertW1), ("w2", C::ExpertW2), ("w3", C::ExpertW3)] { add(&format!("ffn.experts.{e}.{w}"), c, Some(e as u16)); }
    }
    if cfg.kv_sources.contains(&l) {
        for (k, c) in [("attn.compressor.wkv", C::CompWkv), ("attn.compressor.norm", C::CompNorm), ("attn.indexer.wk", C::IdxWk), ("attn.indexer.k_norm", C::IdxKNorm)] { add(k, c, None); }
        if cfg.ratio(l) > 1 { add("attn.compressor.wgate", C::CompWgate, None); }
    }
    if cfg.index_sources.contains(&l) { add("attn.indexer.wq_b", C::IdxWqB, None); add("attn.indexer.weights_proj", C::IdxWeightsProj, None); }
    if cfg.engram_layers.contains(&l) { for (k, c) in [("engram.wkv", C::EngramWkv), ("engram.q_weight", C::EngramQ), ("engram.k_weight", C::EngramK)] { add(k, c, None); } }
    if l == cfg.layers { add("main_proj", C::DsMainProj, None); add("main_norm", C::DsMainNorm, None); }
    if l + 1 == cfg.total_layers() {
        for (k, c) in [("norm", C::DsNorm), ("markov_head.embed", C::DsMarkovEmbed), ("markov_head.head", C::DsMarkovHead), ("confidence_head.proj", C::DsConfidence)] { add(k, c, None); }
    }
    v
}

/// The tensors (name, dtype, shape) of one module in format `f` (params ignore `f`).
pub fn module_tensors(cfg: &Dsv41Config, key: &str, c: C, layer: Option<u16>, sliced: bool, f: SFmt) -> Vec<(String, String, Vec<u64>)> {
    use super::classify::{logical, Logical};
    let lg = logical(c, cfg, layer, sliced).expect("logical");
    let t = |s: &str, d: &str, sh: Vec<u64>| (s.to_string(), d.to_string(), sh);
    if !c.is_linear() {
        let shape = match lg { Logical::Vector(n) => vec![n], Logical::Matrix(a, b) => vec![a, b], Logical::Scalar => vec![], Logical::Linear { out, inn } => vec![out, inn] };
        let dtype = if matches!(c, C::AttnSink | C::GateBias | C::GateBiasVl) || c.name().starts_with("hc_") { "F32" } else { "BF16" };
        let name = match c { C::AttnSink | C::GateBias | C::GateBiasVl | C::EngramQ | C::EngramK | C::ImageSep => key.to_string(),
                             _ if c.name().starts_with("hc_") => key.to_string(), _ => format!("{key}.weight") };
        return vec![t(&name, dtype, shape)];
    }
    let Logical::Linear { out, inn } = lg else { panic!("linear") };
    match f {
        SFmt::Exl3(k) => vec![t(&format!("{key}.trellis"), "I16", vec![inn / 16, out / 16, 16 * k as u64]),
                              t(&format!("{key}.suh"), "F16", vec![inn]), t(&format!("{key}.svh"), "F16", vec![out]), t(&format!("{key}.mul1"), "I32", vec![])],
        SFmt::Plain(d) => vec![t(&format!("{key}.weight"), d, vec![out, inn])],
        SFmt::Fp8 => vec![t(&format!("{key}.weight"), "F8_E4M3", vec![out, inn]), t(&format!("{key}.scale"), "F8_E8M0", vec![out.div_ceil(32), inn.div_ceil(32)])],
        SFmt::Mxfp4 => vec![t(&format!("{key}.weight"), "I8", vec![out, inn / 2]), t(&format!("{key}.scale"), "F8_E8M0", vec![out, inn / 32])],
    }
}

/// Write a synthetic pack: `config` JSON, the given layer slots (+ globals), each linear in the
/// format `policy` picks. `exl3` = EXL3 naming (per-group wo_a slices, a quantization_config.json
/// with tensor_storage for the EXL3 modules); `engram_tables` = put the native tables in.
pub fn write_pack(dir: &Path, config: &Value, layers: &[usize], exl3: bool, engram_tables: bool,
                  policy: &dyn Fn(C, Option<u16>, Option<u16>) -> SFmt) -> Dsv41Config {
    std::fs::create_dir_all(dir).unwrap();
    let mut config = config.clone();
    let cfg0 = Dsv41Config::from_value(&config, None).unwrap();
    if layers.len() != cfg0.total_layers() { config["dsv41_slice"] = serde_json::json!({"layers": layers}); }
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let cfg = Dsv41Config::from_value(&config, None).unwrap();
    let mut files: Vec<(String, Vec<(String, String, Vec<u64>)>)> = Vec::new();
    let mut storage = serde_json::Map::new();
    let mut emit = |key: &str, c: C, layer: Option<u16>, sub: Option<u16>, out: &mut Vec<(String, String, Vec<u64>)>| {
        let sliced = c == C::AttnWoA && sub.is_some();
        let f = policy(c, layer, sub);
        let ts = module_tensors(&cfg, key, c, layer, sliced, f);
        if let (true, SFmt::Exl3(k)) = (c.is_linear(), f) {
            let mut stored = serde_json::Map::new();
            for (n, d, sh) in &ts {
                let torch = match d.as_str() { "I16" => "torch.int16", "F16" => "torch.float16", "I32" => "torch.int32", _ => "?" };
                let bytes = sh.iter().product::<u64>() * st::dtype_size(d).unwrap();
                stored.insert(n.clone(), serde_json::json!({"shape": sh, "n_bytes": bytes, "dtype": torch}));
            }
            storage.insert(key.to_string(), serde_json::json!({"stored_tensors": stored, "quant_format": "exl3", "bits_per_weight": k, "mul1_multiplier": 2212286765u64}));
        }
        out.extend(ts);
    };
    let mut g = Vec::new();
    for (k, c) in [("embed", C::Embed), ("head", C::Head), ("norm", C::FinalNorm), ("image_start", C::ImageSep), ("image_end", C::ImageSep), ("image_newline", C::ImageSep)] {
        emit(k, c, None, None, &mut g);
    }
    files.push(("model-globals.safetensors".into(), g));
    for &l in layers {
        let mut v = Vec::new();
        for (key, c, sub) in layer_modules(&cfg, l, exl3) { emit(&key, c, Some(l as u16), sub, &mut v); }
        if engram_tables && cfg.engram_layers.contains(&l) {
            let li = cfg.engram_layers.iter().position(|&x| x == l).unwrap();
            let rows = cfg.engram_rows[li];
            v.push((format!("layers.{l}.engram.embed.weight"), "F8_E4M3".into(), vec![rows, 256]));
            v.push((format!("layers.{l}.engram.embed.scale"), "F8_E8M0".into(), vec![rows, 8]));
        }
        files.push((format!("model-layer{l:02}.safetensors"), v));
    }
    let mut wm = serde_json::Map::new();
    let mut total = 0u64;
    for (f, ts) in &files {
        let outs: Vec<StOut> = ts.iter().map(|(n, d, sh)| StOut { name: n.clone(), dtype: d.clone(), shape: sh.clone(),
            src: StSource::Hole(sh.iter().product::<u64>() * st::dtype_size(d).unwrap()) }).collect();
        for o in &outs { wm.insert(o.name.clone(), Value::String(f.clone())); }
        total += ts.iter().map(|(_, d, sh)| sh.iter().product::<u64>() * st::dtype_size(d).unwrap()).sum::<u64>();
        st::write_file(&dir.join(f), &outs, &BTreeMap::new(), &mut |_| {}).unwrap();
    }
    std::fs::write(dir.join("model.safetensors.index.json"),
                   serde_json::to_string(&serde_json::json!({"metadata": {"total_size": total}, "weight_map": wm})).unwrap()).unwrap();
    if exl3 {
        std::fs::write(dir.join("quantization_config.json"),
                       serde_json::to_string(&serde_json::json!({"quant_method": "exl3", "bits": 2.9, "tensor_storage": storage})).unwrap()).unwrap();
    }
    cfg
}

/// A fresh, empty temp dir for one test.
pub fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("dsv41_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
