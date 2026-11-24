//! DSV4.1 census (leg I0): tensor names -> (class, role, layer, sub-index, module), the logical
//! shape every class must have (from the config, never per pack), and the preliminary TP split
//! of every class. Pack-generic: names in the DeepSeek inference layout (`layers.N.attn.wq_a.*`)
//! and in the HF layout (`model.layers.N.self_attn.*`, `weight_scale_inv`, `e_score_correction_bias`)
//! classify alike (the same normalization as the reference's `convert.py`).

use crate::dsv41::config::Dsv41Config;

/// Every V4.1 tensor class (one per logical module kind), DSpark and vision included.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorClass {
    // globals
    Embed, Head, FinalNorm, ImageSep,
    // attention (trunk layers and DSpark stages)
    AttnSink, AttnNorm, AttnQNorm, AttnKvNorm, AttnWqA, AttnWqB, AttnWkv, AttnWoA, AttnWoB,
    // compressor (KV sources)
    CompWkv, CompWgate, CompNorm,
    // indexer (index sources; wk / k_norm on KV sources)
    IdxWqB, IdxWeightsProj, IdxWk, IdxKNorm,
    // hyper-connections
    HcAttnFn, HcAttnBase, HcAttnScale, HcFfnFn, HcFfnBase, HcFfnScale,
    // FFN
    FfnNorm, GateWeight, GateBias, GateBiasVl, ExpertW1, ExpertW2, ExpertW3, SharedW1, SharedW2, SharedW3,
    // Engram
    EngramTable, EngramWkv, EngramQ, EngramK,
    // DSpark-only
    DsMainProj, DsMainNorm, DsNorm, DsMarkovEmbed, DsMarkovHead, DsConfidence, DsTiedEmbed, DsTiedHead,
    // vision
    VisPatchEmbed, VisNorm, VisQkv, VisWo, VisMlpW1, VisMlpW2, AlignerW1, AlignerW2,
}

use TensorClass as C;

impl TensorClass {
    pub const ALL: [TensorClass; 56] = [
        C::Embed, C::Head, C::FinalNorm, C::ImageSep, C::AttnSink, C::AttnNorm, C::AttnQNorm, C::AttnKvNorm, C::AttnWqA,
        C::AttnWqB, C::AttnWkv, C::AttnWoA, C::AttnWoB, C::CompWkv, C::CompWgate, C::CompNorm, C::IdxWqB, C::IdxWeightsProj,
        C::IdxWk, C::IdxKNorm, C::HcAttnFn, C::HcAttnBase, C::HcAttnScale, C::HcFfnFn, C::HcFfnBase, C::HcFfnScale,
        C::FfnNorm, C::GateWeight, C::GateBias, C::GateBiasVl, C::ExpertW1, C::ExpertW2, C::ExpertW3, C::SharedW1,
        C::SharedW2, C::SharedW3, C::EngramTable, C::EngramWkv, C::EngramQ, C::EngramK, C::DsMainProj, C::DsMainNorm,
        C::DsNorm, C::DsMarkovEmbed, C::DsMarkovHead, C::DsConfidence, C::DsTiedEmbed, C::DsTiedHead, C::VisPatchEmbed,
        C::VisNorm, C::VisQkv, C::VisWo, C::VisMlpW1, C::VisMlpW2, C::AlignerW1, C::AlignerW2,
    ];

    pub fn name(self) -> &'static str {
        match self {
            C::Embed => "embed", C::Head => "head", C::FinalNorm => "final_norm", C::ImageSep => "image_sep",
            C::AttnSink => "attn.sink", C::AttnNorm => "attn_norm", C::AttnQNorm => "attn.q_norm", C::AttnKvNorm => "attn.kv_norm",
            C::AttnWqA => "attn.wq_a", C::AttnWqB => "attn.wq_b", C::AttnWkv => "attn.wkv", C::AttnWoA => "attn.wo_a", C::AttnWoB => "attn.wo_b",
            C::CompWkv => "compressor.wkv", C::CompWgate => "compressor.wgate", C::CompNorm => "compressor.norm",
            C::IdxWqB => "indexer.wq_b", C::IdxWeightsProj => "indexer.weights_proj", C::IdxWk => "indexer.wk", C::IdxKNorm => "indexer.k_norm",
            C::HcAttnFn => "hc_attn_fn", C::HcAttnBase => "hc_attn_base", C::HcAttnScale => "hc_attn_scale",
            C::HcFfnFn => "hc_ffn_fn", C::HcFfnBase => "hc_ffn_base", C::HcFfnScale => "hc_ffn_scale",
            C::FfnNorm => "ffn_norm", C::GateWeight => "gate.weight", C::GateBias => "gate.bias", C::GateBiasVl => "gate.bias_vl",
            C::ExpertW1 => "experts.w1", C::ExpertW2 => "experts.w2", C::ExpertW3 => "experts.w3",
            C::SharedW1 => "shared.w1", C::SharedW2 => "shared.w2", C::SharedW3 => "shared.w3",
            C::EngramTable => "engram.table", C::EngramWkv => "engram.wkv", C::EngramQ => "engram.q_weight", C::EngramK => "engram.k_weight",
            C::DsMainProj => "dspark.main_proj", C::DsMainNorm => "dspark.main_norm", C::DsNorm => "dspark.norm",
            C::DsMarkovEmbed => "dspark.markov_embed", C::DsMarkovHead => "dspark.markov_head", C::DsConfidence => "dspark.confidence",
            C::DsTiedEmbed => "dspark.tied_embed", C::DsTiedHead => "dspark.tied_head",
            C::VisPatchEmbed => "vision.patch_embed", C::VisNorm => "vision.norm", C::VisQkv => "vision.wqkv", C::VisWo => "vision.wo",
            C::VisMlpW1 => "vision.mlp.w1", C::VisMlpW2 => "vision.mlp.w2", C::AlignerW1 => "aligner.w1", C::AlignerW2 => "aligner.w2",
        }
    }

    /// A weight matrix the forward multiplies by (a "linear": any backend can hold it).
    pub fn is_linear(self) -> bool {
        matches!(self, C::Embed | C::Head | C::AttnWqA | C::AttnWqB | C::AttnWkv | C::AttnWoA | C::AttnWoB | C::CompWkv | C::CompWgate
            | C::IdxWqB | C::IdxWeightsProj | C::IdxWk | C::GateWeight | C::ExpertW1 | C::ExpertW2 | C::ExpertW3 | C::SharedW1
            | C::SharedW2 | C::SharedW3 | C::EngramTable | C::EngramWkv | C::DsMainProj | C::DsMarkovEmbed | C::DsMarkovHead
            | C::DsConfidence | C::DsTiedEmbed | C::DsTiedHead | C::VisPatchEmbed | C::VisQkv | C::VisWo | C::VisMlpW1
            | C::VisMlpW2 | C::AlignerW1 | C::AlignerW2)
    }

    pub fn is_routed_expert(self) -> bool { matches!(self, C::ExpertW1 | C::ExpertW2 | C::ExpertW3) }

    /// The per-layer byte group (census "bytes per layer" columns).
    pub fn group(self) -> &'static str {
        match self {
            C::AttnSink | C::AttnNorm | C::AttnQNorm | C::AttnKvNorm | C::AttnWqA | C::AttnWqB | C::AttnWkv | C::AttnWoA | C::AttnWoB => "attention",
            C::CompWkv | C::CompWgate | C::CompNorm | C::IdxWqB | C::IdxWeightsProj | C::IdxWk | C::IdxKNorm => "indexer+compressor",
            C::HcAttnFn | C::HcAttnBase | C::HcAttnScale | C::HcFfnFn | C::HcFfnBase | C::HcFfnScale | C::FfnNorm => "hc+norms",
            C::GateWeight | C::GateBias | C::GateBiasVl => "router",
            C::ExpertW1 | C::ExpertW2 | C::ExpertW3 => "routed experts",
            C::SharedW1 | C::SharedW2 | C::SharedW3 => "shared expert",
            C::EngramTable => "engram tables",
            C::EngramWkv | C::EngramQ | C::EngramK => "engram module",
            C::DsMainProj | C::DsMainNorm | C::DsNorm | C::DsMarkovEmbed | C::DsMarkovHead | C::DsConfidence | C::DsTiedEmbed | C::DsTiedHead => "dspark extras",
            C::Embed | C::Head | C::FinalNorm | C::ImageSep => "global",
            C::VisPatchEmbed | C::VisNorm | C::VisQkv | C::VisWo | C::VisMlpW1 | C::VisMlpW2 | C::AlignerW1 | C::AlignerW2 => "vision",
        }
    }
}

/// The role of a tensor inside its module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// the weight values (plain / FP8 / packed FP4)
    Weight,
    /// a block scale tensor (FP8 / FP4 blocks; HF `weight_scale_inv` / NVFP4 `weight_scale`)
    Scale,
    /// a second-level (per-tensor) scale (NVFP4 `weight_scale_2`)
    Scale2,
    /// an activation scale stored with the weight (NVFP4 `input_scale`)
    InputScale,
    /// EXL3: the packed trellis codes
    Trellis,
    /// EXL3: the input-side Hadamard sign/scale vector
    Suh,
    /// EXL3: the output-side Hadamard sign/scale vector
    Svh,
    /// EXL3: the codebook multiplier (mul1 codebook)
    Mul1,
    /// a linear's additive bias
    Bias,
    /// a module that is one tensor (norm weights stored as `.weight` are `Weight`; hc params, sinks)
    Param,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Weight => "weight", Role::Scale => "scale", Role::Scale2 => "scale2", Role::InputScale => "input_scale",
            Role::Trellis => "trellis", Role::Suh => "suh", Role::Svh => "svh", Role::Mul1 => "mul1", Role::Bias => "bias", Role::Param => "param",
        }
    }
}

/// One classified tensor name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    pub class: TensorClass,
    pub role: Role,
    /// model layer slot: 0..layers = trunk, layers.. = DSpark stages (None = global / vision)
    pub layer: Option<u16>,
    /// expert index (routed experts), wo_a slice index (EXL3 per-group slices), vision block index
    pub sub: Option<u16>,
    /// the module key: the name without its role suffix (all tensors of one linear share it)
    pub module: String,
}

/// convert.py's normalization (HF layout -> inference layout).
pub fn normalize(name: &str) -> String {
    let mut n = name.strip_prefix("model.").unwrap_or(name).to_string();
    n = n.replace("self_attn", "attn");
    if !n.starts_with("vision.") { n = n.replace(".mlp.", ".ffn."); }
    n = n.replace("weight_scale_inv", "scale");
    n = n.replace("e_score_correction_bias", "bias");
    n
}

fn role_of(last: &str) -> Option<Role> {
    Some(match last {
        "weight" => Role::Weight, "scale" | "weight_scale" => Role::Scale, "weight_scale_2" => Role::Scale2,
        "input_scale" => Role::InputScale, "trellis" => Role::Trellis, "suh" => Role::Suh, "svh" => Role::Svh,
        "mul1" => Role::Mul1, "bias" => Role::Bias,
        _ => return None,
    })
}

/// Classify one tensor name (None = not a V4.1 tensor: the census reports it as a problem).
pub fn classify(raw: &str, n_layers: usize) -> Option<Classified> {
    let name = normalize(raw);
    let parts: Vec<&str> = name.split('.').collect();
    // prefix -> (layer, rest)
    let (layer, vblock, rest): (Option<u16>, Option<u16>, &[&str]) = match parts.as_slice() {
        ["layers", l, rest @ ..] => (Some(l.parse::<u16>().ok()?), None, rest),
        ["mtp", s, rest @ ..] => (Some(n_layers as u16 + s.parse::<u16>().ok()?), None, rest),
        ["vision", "blocks", b, rest @ ..] => (None, Some(b.parse::<u16>().ok()?), rest),
        _ => (None, None, &parts[..]),
    };
    let (role, mpath): (Role, &[&str]) = match rest.split_last() {
        Some((last, head)) if !head.is_empty() => match role_of(last) { Some(r) => (r, head), None => (Role::Param, rest) },
        _ => (Role::Param, rest),
    };
    let module_of = |m: &[&str]| -> String {
        let keep = name.len() - rest.join(".").len();
        format!("{}{}", &name[..keep], m.join("."))
    };
    let mut sub = vblock;
    let is_ds = layer.map_or(false, |l| l as usize >= n_layers);
    let class = match (layer.is_some(), vblock.is_some(), mpath) {
        (true, _, ["attn", "attn_sink"]) => C::AttnSink,
        (true, _, ["attn", "compressor", "norm"]) => C::CompNorm,
        (true, _, ["attn", "compressor", "wgate"]) => C::CompWgate,
        (true, _, ["attn", "compressor", "wkv"]) => C::CompWkv,
        (true, _, ["attn", "indexer", "k_norm"]) => C::IdxKNorm,
        (true, _, ["attn", "indexer", "weights_proj"]) => C::IdxWeightsProj,
        (true, _, ["attn", "indexer", "wk"]) => C::IdxWk,
        (true, _, ["attn", "indexer", "wq_b"]) => C::IdxWqB,
        (true, _, ["attn", "kv_norm"]) => C::AttnKvNorm,
        (true, _, ["attn", "q_norm"]) => C::AttnQNorm,
        (true, _, ["attn", "wkv"]) => C::AttnWkv,
        (true, _, ["attn", "wq_a"]) => C::AttnWqA,
        (true, _, ["attn", "wq_b"]) => C::AttnWqB,
        (true, _, ["attn", "wo_a"]) => C::AttnWoA,
        (true, _, ["attn", "wo_a", "slice", s]) => { sub = Some(s.parse().ok()?); C::AttnWoA }
        (true, _, ["attn", "wo_b"]) => C::AttnWoB,
        (true, _, ["attn_norm"]) => C::AttnNorm,
        (true, _, ["ffn_norm"]) => C::FfnNorm,
        (true, _, ["engram", "embed"]) => C::EngramTable,
        (true, _, ["engram", "wkv"]) => C::EngramWkv,
        (true, _, ["engram", "q_weight"]) => C::EngramQ,
        (true, _, ["engram", "k_weight"]) => C::EngramK,
        (true, _, ["ffn", "experts", e, w]) => {
            sub = Some(e.parse().ok()?);
            match *w { "w1" => C::ExpertW1, "w2" => C::ExpertW2, "w3" => C::ExpertW3, _ => return None }
        }
        (true, _, ["ffn", "gate"]) if role == Role::Weight => C::GateWeight,
        (true, _, ["ffn", "gate"]) if role == Role::Bias => C::GateBias,
        (true, _, ["ffn", "gate", "bias_vl"]) => C::GateBiasVl,
        (true, _, ["ffn", "shared_experts", w]) => match *w { "w1" => C::SharedW1, "w2" => C::SharedW2, "w3" => C::SharedW3, _ => return None },
        (true, _, ["hc_attn_fn"]) => C::HcAttnFn,
        (true, _, ["hc_attn_base"]) => C::HcAttnBase,
        (true, _, ["hc_attn_scale"]) => C::HcAttnScale,
        (true, _, ["hc_ffn_fn"]) => C::HcFfnFn,
        (true, _, ["hc_ffn_base"]) => C::HcFfnBase,
        (true, _, ["hc_ffn_scale"]) => C::HcFfnScale,
        (true, _, ["main_proj"]) if is_ds => C::DsMainProj,
        (true, _, ["main_norm"]) if is_ds => C::DsMainNorm,
        (true, _, ["norm"]) if is_ds => C::DsNorm,
        (true, _, ["markov_head", "embed"]) if is_ds => C::DsMarkovEmbed,
        (true, _, ["markov_head", "head"]) if is_ds => C::DsMarkovHead,
        (true, _, ["confidence_head", "proj"]) if is_ds => C::DsConfidence,
        (true, _, ["embed"]) if is_ds => C::DsTiedEmbed,
        (true, _, ["head"]) if is_ds => C::DsTiedHead,
        (false, true, ["norm1"]) | (false, true, ["norm2"]) => C::VisNorm,
        (false, true, ["attn", "wqkv"]) => C::VisQkv,
        (false, true, ["attn", "wo"]) => C::VisWo,
        (false, true, ["mlp", "w1"]) => C::VisMlpW1,
        (false, true, ["mlp", "w2"]) => C::VisMlpW2,
        (false, false, ["embed"]) => C::Embed,
        (false, false, ["head"]) => C::Head,
        (false, false, ["norm"]) => C::FinalNorm,
        (false, false, ["image_start"]) | (false, false, ["image_end"]) | (false, false, ["image_newline"]) => C::ImageSep,
        (false, false, ["vision", "patch_embed", "proj"]) => C::VisPatchEmbed,
        (false, false, ["vision", "norm"]) => C::VisNorm,
        (false, false, ["aligner", "w1"]) => C::AlignerW1,
        (false, false, ["aligner", "w2"]) => C::AlignerW2,
        _ => return None,
    };
    // the gate's bias is its own (selection-only) parameter, not the linear's additive bias
    let module = if class == C::GateBias { name.clone() } else { module_of(mpath) };
    Some(Classified { class, role, layer, sub, module })
}

/// The logical shape a class must have, from the config (never from a pack).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logical {
    /// y = x W^T with W [out, in]
    Linear { out: u64, inn: u64 },
    Vector(u64),
    Matrix(u64, u64),
    Scalar,
}

/// `layer` is the module's layer slot; `sliced` = an EXL3 per-group `wo_a.slice.N` module.
pub fn logical(class: TensorClass, cfg: &Dsv41Config, layer: Option<u16>, sliced: bool) -> Option<Logical> {
    let d = cfg.hidden as u64;
    let h = cfg.heads as u64;
    let hd = cfg.head_dim as u64;
    let g = cfg.o_groups as u64;
    let lin = |out: u64, inn: u64| Some(Logical::Linear { out, inn });
    let v = |n: u64| Some(Logical::Vector(n));
    let n_exp = layer.map(|l| cfg.moe_of(l as usize).0).unwrap_or(cfg.n_routed) as u64;
    Some(match class {
        C::Embed | C::Head | C::DsTiedEmbed | C::DsTiedHead => return lin(cfg.vocab as u64, d),
        C::FinalNorm | C::AttnNorm | C::FfnNorm | C::DsMainNorm | C::DsNorm | C::ImageSep => return v(d),
        C::AttnSink => return v(h),
        C::AttnQNorm => return v(cfg.q_lora_rank as u64),
        C::AttnKvNorm | C::CompNorm => return v(hd),
        C::AttnWqA => return lin(cfg.q_lora_rank as u64, d),
        C::AttnWqB => return lin(h * hd, cfg.q_lora_rank as u64),
        C::AttnWkv | C::CompWkv | C::CompWgate => return lin(hd, d),
        C::AttnWoA => return if sliced { lin(cfg.o_lora_rank as u64, h * hd / g) } else { lin(g * cfg.o_lora_rank as u64, h * hd / g) },
        C::AttnWoB => return lin(d, g * cfg.o_lora_rank as u64),
        C::IdxWqB => return lin((cfg.index_heads * cfg.index_head_dim) as u64, cfg.q_lora_rank as u64),
        C::IdxWeightsProj => return lin(cfg.index_heads as u64, d),
        C::IdxWk => return lin(cfg.index_head_dim as u64, hd),
        C::IdxKNorm => return v(cfg.index_head_dim as u64),
        C::HcAttnFn | C::HcFfnFn => Logical::Matrix(cfg.hc_mix() as u64, cfg.hc as u64 * d),
        C::HcAttnBase | C::HcFfnBase => Logical::Vector(cfg.hc_mix() as u64),
        C::HcAttnScale | C::HcFfnScale => Logical::Vector(3),
        C::GateWeight => return lin(n_exp, d),
        C::GateBias | C::GateBiasVl => return v(n_exp),
        C::ExpertW1 | C::ExpertW3 | C::SharedW1 | C::SharedW3 => return lin(cfg.moe_inter as u64, d),
        C::ExpertW2 | C::SharedW2 => return lin(d, cfg.moe_inter as u64),
        C::EngramTable => {
            let li = cfg.engram_layers.iter().position(|&l| Some(l as u16) == layer)?;
            return lin(cfg.engram_rows[li], cfg.engram_head_dim as u64);
        }
        C::EngramWkv => return lin(d * (cfg.hc as u64 + 1), (cfg.engram_cols() * cfg.engram_head_dim) as u64),
        C::EngramQ | C::EngramK => Logical::Matrix(cfg.hc as u64, d),
        C::DsMainProj => return lin(d, d * cfg.dspark_targets.len() as u64),
        C::DsMarkovEmbed | C::DsMarkovHead => return lin(cfg.vocab as u64, cfg.dspark_markov_rank as u64),
        C::DsConfidence => return lin(1, d + cfg.dspark_markov_rank as u64),
        C::VisPatchEmbed | C::VisNorm | C::VisQkv | C::VisWo | C::VisMlpW1 | C::VisMlpW2 | C::AlignerW1 | C::AlignerW2 => {
            let vc = cfg.vision.as_ref()?;
            let vd = vc.hidden as u64;
            match class {
                C::VisPatchEmbed => return lin(vd, 3 * (vc.patch * vc.patch) as u64),
                C::VisNorm => return v(vd),
                C::VisQkv => return lin(3 * vd, vd),
                C::VisWo => return lin(vd, vd),
                C::VisMlpW1 => return lin(2 * vc.inter as u64, vd),
                C::VisMlpW2 => return lin(vd, vc.inter as u64),
                C::AlignerW1 => return lin(d, vd * (vc.downsample * vc.downsample) as u64),
                _ => return lin(d, d),
            }
        }
    })
}

/// How a class is laid out over TP ranks (the plan's PRELIMINARY split; leg T0 owns the final one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// every rank holds all of it
    Replicated,
    /// expert parallel: rank r holds experts [r*E/W, (r+1)*E/W)
    Expert,
    /// column-parallel: the output rows are divided (heads, groups, vocab)
    Out,
    /// row-parallel: the input columns are divided (the reduce follows)
    In,
    /// Engram tables: rank r holds its Engram heads' columns
    EngramHeads,
}

pub fn split_of(class: TensorClass) -> Split {
    match class {
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 => Split::Expert,
        C::AttnWqB | C::AttnSink | C::AttnWoA | C::IdxWqB | C::IdxWeightsProj | C::Head | C::DsMarkovHead => Split::Out,
        C::AttnWoB => Split::In,
        C::EngramTable => Split::EngramHeads,
        _ => Split::Replicated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_classify_covers_both_layouts() {
        let c = |n: &str| classify(n, 40).unwrap_or_else(|| panic!("{n}"));
        let x = c("layers.3.attn.wo_a.slice.5.trellis");
        assert_eq!((x.class, x.role, x.layer, x.sub, x.module.as_str()), (C::AttnWoA, Role::Trellis, Some(3), Some(5), "layers.3.attn.wo_a.slice.5"));
        let x = c("mtp.2.ffn.experts.127.w2.scale");
        assert_eq!((x.class, x.role, x.layer, x.sub), (C::ExpertW2, Role::Scale, Some(42), Some(127)));
        assert_eq!(c("layers.0.ffn.gate.bias").class, C::GateBias);
        assert_eq!(c("layers.0.ffn.gate.bias").module, "layers.0.ffn.gate.bias");
        assert_eq!(c("layers.0.ffn.gate.weight").module, "layers.0.ffn.gate");
        assert_eq!(c("layers.0.ffn.gate.bias_vl").class, C::GateBiasVl);
        assert_eq!(c("layers.0.hc_attn_scale").class, C::HcAttnScale);
        assert_eq!(c("layers.0.hc_attn_scale").role, Role::Param);
        assert_eq!(c("mtp.2.norm.weight").class, C::DsNorm);
        assert_eq!(c("norm.weight").class, C::FinalNorm);
        assert_eq!(c("vision.blocks.31.attn.wqkv.bias").role, Role::Bias);
        assert_eq!(c("vision.blocks.31.attn.wqkv.bias").sub, Some(31));
        assert_eq!(c("image_newline").class, C::ImageSep);
        // HF layout
        let x = c("model.layers.7.self_attn.wq_a.weight_scale_inv");
        assert_eq!((x.class, x.role, x.layer), (C::AttnWqA, Role::Scale, Some(7)));
        assert_eq!(c("model.layers.7.mlp.gate.e_score_correction_bias").class, C::GateBias);
        assert_eq!(c("layers.9.ffn.experts.3.w1.weight_scale_2").role, Role::Scale2);
        assert!(classify("layers.0.attn.mystery", 40).is_none());
        assert!(classify("lm_head.weight", 40).is_none());
    }

    #[test]
    fn dsv41_class_table_is_exact() {
        let mut v: Vec<TensorClass> = TensorClass::ALL.to_vec();
        v.sort(); v.dedup();
        assert_eq!(v.len(), TensorClass::ALL.len(), "every class exactly once in ALL");
        for c in v { assert!(!c.name().is_empty() && !c.group().is_empty()); }
    }
}
