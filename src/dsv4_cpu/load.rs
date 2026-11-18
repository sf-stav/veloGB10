//! Weight extraction (Lane A `Dsv4Layer` -> `CpuLayer`, replay path only), trunk-top extraction and the
//! packed-cache dequant oracle helpers (single source of truth for the R5 gates).
//! Siblings: `block.rs` consumes the extracted layers; `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------
// Weight extraction (Lane A Dsv4Layer -> CpuLayer; replay path only — unit
// tests construct CpuLayer directly)
// ---------------------------------------------------------------------------

/// Build a trunk CpuLayer from Lane A's strict-loaded layer.
pub fn cpu_layer_from_dsv4(layer: Dsv4Layer, cfg: &Dsv4Config, kind: LayerKind) -> Result<CpuLayer> {
    let Dsv4Layer { mut tensors, experts_w1, experts_w2, experts_w3 } = layer;
    cpu_layer_core(&mut tensors, (experts_w1, experts_w2, experts_w3), cfg, kind)
}

/// Shared extraction core: pulls every tensor a Block needs out of `map`
/// (stripped keys, §A), leaving any extras (DSpark stage-specifics) behind.
fn cpu_layer_core(
    mut map: &mut HashMap<String, HostTensor>,
    experts: (Vec<Nvfp4Tensor>, Vec<Nvfp4Tensor>, Vec<Nvfp4Tensor>),
    cfg: &Dsv4Config,
    kind: LayerKind,
) -> Result<CpuLayer> {
    let dim = cfg.dim;
    let qlr = cfg.q_lora_rank;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let g = cfg.o_groups;
    let r = cfg.o_lora_rank;
    let ne = cfg.n_routed_experts;
    let inter = cfg.moe_inter_dim;
    let compressor = match kind {
        LayerKind::Swa => None,
        LayerKind::Csa => Some(CompressorWeights {
            wkv: take_f32(&mut map, "attn.compressor.wkv.weight", 2 * hd * dim)?,
            wgate: take_f32(&mut map, "attn.compressor.wgate.weight", 2 * hd * dim)?,
            norm: take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
            ape: take_f32(&mut map, "attn.compressor.ape", 4 * 2 * hd)?,
            ratio: 4,
            head_dim: hd,
            rope_dim: cfg.rope_head_dim,
            overlap: true,
            rotate: false,
            sim_group: 64,
            dim,
        }),
        LayerKind::Hca => Some(CompressorWeights {
            wkv: take_f32(&mut map, "attn.compressor.wkv.weight", hd * dim)?,
            wgate: take_f32(&mut map, "attn.compressor.wgate.weight", hd * dim)?,
            norm: take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
            ape: take_f32(&mut map, "attn.compressor.ape", 128 * hd)?,
            ratio: 128,
            head_dim: hd,
            rope_dim: cfg.rope_head_dim,
            overlap: false,
            rotate: false,
            sim_group: 64,
            dim,
        }),
    };
    let indexer = if matches!(kind, LayerKind::Csa) {
        let ihd = cfg.index_head_dim;
        Some(IndexerWeights {
            wq_b: take_f32(&mut map, "attn.indexer.wq_b.weight", cfg.index_n_heads * ihd * qlr)?,
            weights_proj: take_bf16_as_f32(&mut map, "attn.indexer.weights_proj.weight", cfg.index_n_heads * dim)?,
            compressor: CompressorWeights {
                wkv: take_f32(&mut map, "attn.indexer.compressor.wkv.weight", 2 * ihd * dim)?,
                wgate: take_f32(&mut map, "attn.indexer.compressor.wgate.weight", 2 * ihd * dim)?,
                norm: take_f32(&mut map, "attn.indexer.compressor.norm.weight", ihd)?,
                ape: take_f32(&mut map, "attn.indexer.compressor.ape", 4 * 2 * ihd)?,
                ratio: 4,
                head_dim: ihd,
                rope_dim: cfg.rope_head_dim,
                overlap: true,
                rotate: true,
                sim_group: 32,
                dim,
            },
        })
    } else {
        None
    };
    let attn = AttnWeights {
        wq_a: take_f32(&mut map, "attn.wq_a.weight", qlr * dim)?,
        q_norm: take_f32(&mut map, "attn.q_norm.weight", qlr)?,
        wq_b: take_f32(&mut map, "attn.wq_b.weight", nh * hd * qlr)?,
        wkv: take_f32(&mut map, "attn.wkv.weight", hd * dim)?,
        kv_norm: take_f32(&mut map, "attn.kv_norm.weight", hd)?,
        sink: take_f32(&mut map, "attn.attn_sink", nh)?,
        wo_a: take_bf16_as_f32(&mut map, "attn.wo_a.weight", g * r * nh * hd / g)?,
        wo_b: take_f32(&mut map, "attn.wo_b.weight", dim * g * r)?,
        compressor,
        indexer,
        kind,
    };
    let hc = |prefix: &str, map: &mut HashMap<String, HostTensor>| -> Result<HcParams> {
        Ok(HcParams {
            hc_fn: take_f32(map, &format!("{prefix}_fn"), 24 * cfg.hc_mult * dim)?,
            hc_base: take_f32(map, &format!("{prefix}_base"), 24)?,
            hc_scale: {
                let v = take_f32(map, &format!("{prefix}_scale"), 3)?;
                [v[0], v[1], v[2]]
            },
        })
    };
    let hc_attn = hc("hc_attn", &mut map)?;
    let hc_ffn = hc("hc_ffn", &mut map)?;
    let moe = MoeWeights {
        gate_w: take_bf16_as_f32(&mut map, "ffn.gate.weight", ne * dim)?,
        gate_bias: opt_f32(&mut map, "ffn.gate.bias", ne)?,
        tid2eid: {
            let key = "ffn.gate.tid2eid";
            if map.contains_key(key) {
                Some(take_i32(&mut map, key, cfg.vocab_size * cfg.n_activated_experts)?)
            } else {
                None
            }
        },
        shared: ExpertF32 {
            w1: take_f32(&mut map, "ffn.shared_experts.w1.weight", inter * dim)?,
            w2: take_f32(&mut map, "ffn.shared_experts.w2.weight", dim * inter)?,
            w3: take_f32(&mut map, "ffn.shared_experts.w3.weight", inter * dim)?,
        },
        experts: ExpertBank::from_nvfp4(experts.0, experts.1, experts.2),
    };
    Ok(CpuLayer {
        kind,
        attn,
        attn_norm: take_f32(&mut map, "attn_norm.weight", dim)?,
        ffn_norm: take_f32(&mut map, "ffn_norm.weight", dim)?,
        hc_attn,
        hc_ffn,
        moe,
        main_proj: None,
        main_norm: None,
        norm: None,
        hc_head: None,
        markov_w1: None,
        markov_w2: None,
        confidence: None,
    })
}

/// Build a DSpark stage CpuLayer from Lane A's `load_mtp_stage` output
/// (mtp.{S}.* — a full Block minus compressor/indexer, plus stage extras).
pub fn cpu_stage_from_dsv4(layer: Dsv4Layer, cfg: &Dsv4Config, stage: usize) -> Result<CpuLayer> {
    let Dsv4Layer { mut tensors, experts_w1, experts_w2, experts_w3 } = layer;
    let mut base = cpu_layer_core(&mut tensors, (experts_w1, experts_w2, experts_w3), cfg, LayerKind::Swa)?;
    let dim = cfg.dim;
    if stage == 0 {
        base.main_proj = Some(take_f32(&mut tensors, "main_proj.weight", dim * 3 * dim)?);
        base.main_norm = Some(take_f32(&mut tensors, "main_norm.weight", dim)?);
    }
    if stage == cfg.n_mtp_layers - 1 {
        let rank = cfg.dspark_markov_rank;
        base.norm = Some(take_f32(&mut tensors, "norm.weight", dim)?);
        base.hc_head = Some(HcHeadParams {
            hc_fn: take_f32(&mut tensors, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * dim)?,
            hc_base: take_f32(&mut tensors, "hc_head_base", cfg.hc_mult)?,
            hc_scale: take_f32(&mut tensors, "hc_head_scale", 1)?[0],
        });
        base.markov_w1 = Some(take_bf16_as_f32(&mut tensors, "markov_head.markov_w1.weight", cfg.vocab_size * rank)?);
        base.markov_w2 = Some(take_f32(&mut tensors, "markov_head.markov_w2.weight", cfg.vocab_size * rank)?);
        base.confidence = Some(take_f32(&mut tensors, "confidence_head.proj.weight", dim + rank)?);
    }
    Ok(base)
}

/// Trunk top level (embed/norm/head/hc_head) from `load_trunk_top`.
pub struct TrunkTop {
    pub embed: Vec<f32>, // [vocab, dim] bf16-valued
    pub norm: Vec<f32>,
    pub head: Vec<f32>, // fp32
    pub hc_head: HcHeadParams,
}

pub fn trunk_top_from(mut map: HashMap<String, HostTensor>, cfg: &Dsv4Config) -> Result<TrunkTop> {
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    Ok(TrunkTop {
        embed: take_bf16_as_f32(&mut map, "embed.weight", vocab * dim)?,
        norm: take_f32(&mut map, "norm.weight", dim)?,
        head: take_f32(&mut map, "head.weight", vocab * dim)?,
        hc_head: HcHeadParams {
            hc_fn: take_f32(&mut map, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * dim)?,
            hc_base: take_f32(&mut map, "hc_head_base", cfg.hc_mult)?,
            hc_scale: take_f32(&mut map, "hc_head_scale", 1)?[0],
        },
    })
}

// ---------------------------------------------------------------------------
// R5 dequant oracle helpers (single source of truth for the packed-cache gates +
// the future packed gather oracle). These mirror the device decoders EXACTLY
// (gpu_dsv4.cu §C.2 / gpu_dsv4_comp.cu §5): dequant(pack(v)) == the QAT-simmed
// bf16 value bit-for-bit — that identity is the whole R5 premise.
// ---------------------------------------------------------------------------

/// FP4 E2M1 decode (device: `dsv4_fp4_to_f32`). sign<<3 | exp<<1 | man.
pub fn fp4_e2m1_to_f32(c: u8) -> f32 {
    let e = (c >> 1) & 3;
    let m = c & 1;
    let mag = if e == 0 {
        if m == 1 { 0.5 } else { 0.0 }
    } else {
        (1.0 + 0.5 * m as f32) * (1u32 << (e - 1)) as f32
    };
    if c & 8 != 0 { -mag } else { mag }
}

/// FP8 E4M3 decode (device: `__nv_cvt_fp8_to_halfraw`). sign<<7 | exp(4,bias 7)<<3 | man(3).
pub fn fp8_e4m3_to_f32(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 0xF) as i32;
    let m = (c & 0x7) as f32;
    let mag = if e == 0 { (m / 8.0) * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    sign * mag
}

/// UE8M0 scale byte → f32 (e8m0 byte = biased pow2 exponent; b >= 1 by the amax floors).
pub fn e8m0_to_f32(b: u8) -> f32 {
    ((b as i32 - 127) as f32).exp2()
}

/// Unpack one FP4-g32 row (the indexer cache form): n dims from n/2 code bytes + n/32
/// scale bytes — returns the bf16 values the QAT-sim cache holds (bf16(fp4*sc)).
pub fn unpack_fp4_g32_row(codes: &[u8], scales: &[u8], n: usize) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(n);
    for g in 0..(n / 32) {
        let sc = e8m0_to_f32(scales[g]);
        for k in 0..32 {
            let byte = codes[g * 16 + (k >> 1)];
            let nib = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
            out.push(half::bf16::from_f32(fp4_e2m1_to_f32(nib) * sc));
        }
    }
    out
}

/// Unpack one FP8-g64 span (the attn cache's nope form): n dims from n code bytes +
/// n/64 scale bytes — bf16(fp8*sc).
pub fn unpack_fp8_g64_span(codes: &[u8], scales: &[u8], n: usize) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(n);
    for g in 0..(n / 64) {
        let sc = e8m0_to_f32(scales[g]);
        for k in 0..64 {
            out.push(half::bf16::from_f32(fp8_e4m3_to_f32(codes[g * 64 + k]) * sc));
        }
    }
    out
}
