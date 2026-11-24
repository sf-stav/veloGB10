//! The stock Inkling-2 pack: census of every tensor (`model-*.safetensors` via the index, plus the
//! bundled drafter `mtp.safetensors`), shape/dtype checks against the config, and the exact host
//! dequantization of the routed experts.
//!
//! Expert storage (verified against transformers `core_model_loading.Inkling2LoopDequantizeExperts`
//! + `Interleave(dim=1)`):
//!
//! | format | tensors | dequant |
//! |---|---|---|
//! | NVFP4 (`amax` present) | `w{13,2}_weight` U8 `[E, N, K/2]` (low nibble = even k), `_scale` F8_E4M3 `[E, N, K/16]`, `_amax` F32 `[E]`, `_outer_scale_denominator` F32 `[1]` | `(e2m1 * e4m3(s)) * (amax[e] / denom)` in f32 |
//! | MXFP8 | `w{13,2}_weight` F8_E4M3 `[E, N, K]`, `_scale` U8 `[E, N, K/32]` | `e4m3(w) * 2^(s - 127)` |
//!
//! `w13` rows are INTERLEAVED (row `2i` = gate row `i`, row `2i+1` = up row `i`); the reference
//! de-interleaves to `gate_up_proj = [gate; up]` and `chunk(2)` gives gate first.
//!
//! The safetensors reader is `fp4l::pack::StDir`, used read-only.

use super::config::Inkling2Config;
use crate::fp4l::pack::{e4m3, Dt, StDir, TRef, E2M1};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpFmt { Nvfp4, Mxfp8 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    ExpertsNvfp4,
    ExpertsMxfp8,
    EmbedUnembed,
    AttnRouterConvNorm,
    AttnResQueries,
    Controllers,
    Vision,
    Mtp,
}

pub struct Inkling2Pack {
    pub st: StDir,
    pub mtp: Option<StDir>,
    pub cfg: Inkling2Config,
    pub exp_fmt: Vec<ExpFmt>,
    pub quant_json: Option<serde_json::Value>,
}

fn dt_parse(s: &str) -> Result<Dt> {
    Ok(match s {
        "BF16" => Dt::Bf16, "F16" => Dt::F16, "F32" => Dt::F32, "U8" => Dt::U8, "U32" => Dt::U32,
        "I64" => Dt::I64, "F8_E4M3" => Dt::F8e4m3,
        o => bail!("unsupported safetensors dtype {o}"),
    })
}

/// A single safetensors file as a `StDir` (the drafter file is not in the index).
pub fn open_single(path: &Path) -> Result<StDir> {
    let fh = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len8 = [0u8; 8];
    fh.read_exact_at(&mut len8, 0)?;
    let n = u64::from_le_bytes(len8);
    ensure!(n < (1 << 31), "{}: implausible header length", path.display());
    let mut hb = vec![0u8; n as usize];
    fh.read_exact_at(&mut hb, 8)?;
    let h: serde_json::Value = serde_json::from_slice(&hb)?;
    let mut t = HashMap::new();
    for (name, e) in h.as_object().context("header")? {
        if name == "__metadata__" { continue; }
        let dtype = dt_parse(e["dtype"].as_str().unwrap_or("?"))?;
        let shape: Vec<usize> = e["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let o = e["data_offsets"].as_array().context("data_offsets")?;
        let (a, b) = (o[0].as_u64().unwrap_or(0), o[1].as_u64().unwrap_or(0));
        ensure!(b - a == shape.iter().product::<usize>() as u64 * dtype.size() as u64, "{name}: byte length vs shape");
        t.insert(name.clone(), TRef { file: 0, dtype, shape, off: 8 + n + a, len: b - a });
    }
    Ok(StDir {
        dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        files: vec![fh],
        file_names: vec![path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()],
        t,
    })
}

/// One expected tensor.
struct Want { dt: Dt, shape: Vec<usize>, class: Class }

pub const P: &str = "model.llm.";

impl Inkling2Pack {
    pub fn open(dir: &Path) -> Result<Inkling2Pack> {
        let cfg = Inkling2Config::load(dir)?;
        let st = StDir::open(dir)?;
        let mp = dir.join("mtp.safetensors");
        let mtp = if mp.is_file() { Some(open_single(&mp)?) } else { None };
        let mut exp_fmt = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let base = format!("{P}layers.{l}.mlp.experts.w13_weight");
            ensure!(st.has(&base), "missing {base}");
            exp_fmt.push(if st.has(&format!("{base}_amax")) { ExpFmt::Nvfp4 } else { ExpFmt::Mxfp8 });
        }
        let qp = dir.join("hf_quant_config.json");
        let quant_json = if qp.is_file() { Some(serde_json::from_slice(&std::fs::read(&qp)?)?) } else { None };
        Ok(Inkling2Pack { st, mtp, cfg, exp_fmt, quant_json })
    }

    /// Every tensor the text model, the vision tower and the drafter must have, with dtype and shape.
    fn wanted(&self) -> (BTreeMap<String, Want>, BTreeMap<String, Want>) {
        let c = &self.cfg;
        let d = c.hidden;
        let mut w = BTreeMap::new();
        let put = |m: &mut BTreeMap<String, Want>, n: String, dt: Dt, shape: Vec<usize>, class: Class| { m.insert(n, Want { dt, shape, class }); };
        use Class::*;
        let bf = Dt::Bf16;
        for l in 0..c.layers {
            let p = format!("{P}layers.{l}.");
            let (h, kv, hd) = c.attn_geom(l);
            let k = c.conv_k;
            let a = AttnRouterConvNorm;
            put(&mut w, format!("{p}attn.attn_sink_proj.weight"), bf, vec![h, d], a);
            put(&mut w, format!("{p}attn.gate_attn.weight"), bf, vec![h * hd, d], a);
            put(&mut w, format!("{p}attn.q_l2_clip.scale"), bf, vec![h, hd], a);
            put(&mut w, format!("{p}attn.rel_l2_clip.scale"), bf, vec![h, c.d_rel], a);
            put(&mut w, format!("{p}attn.rel_logits_proj.proj"), bf, vec![c.d_rel, c.rel_ext(l)], a);
            put(&mut w, format!("{p}attn.wq_du.weight"), bf, vec![h * hd, d], a);
            put(&mut w, format!("{p}attn.wk_dv.weight"), bf, vec![kv * hd, d], a);
            put(&mut w, format!("{p}attn.wv_dv.weight"), bf, vec![kv * hd, d], a);
            put(&mut w, format!("{p}attn.wr_du.weight"), bf, vec![h * c.d_rel, d], a);
            put(&mut w, format!("{p}attn.wo_ud.weight"), bf, vec![d, h * hd], a);
            if c.is_sliding[l] {
                put(&mut w, format!("{p}attn.k_sconv.weight"), bf, vec![kv * hd, 1, k], a);
                put(&mut w, format!("{p}attn.v_sconv.weight"), bf, vec![kv * hd, 1, k], a);
                put(&mut w, format!("{p}attn_norm.weight"), bf, vec![d], a);
            } else {
                for s in ["k", "v"] {
                    put(&mut w, format!("{p}attn.{s}_sconv.static_w"), bf, vec![k, kv * hd], a);
                    put(&mut w, format!("{p}attn.{s}_sconv.weight_proj"), bf, vec![kv * k, d], a);
                }
                for s in ["q", "k", "v"] { put(&mut w, format!("{p}{s}_input_norm.weight"), bf, vec![d], a); }
            }
            put(&mut w, format!("{p}attn_sconv.weight"), bf, vec![d, 1, k], a);
            put(&mut w, format!("{p}mlp_sconv.weight"), bf, vec![d, 1, k], a);
            put(&mut w, format!("{p}mlp_norm.weight"), bf, vec![d], a);
            put(&mut w, format!("{p}mlp.gate.weight"), bf, vec![c.n_experts, d], a);
            put(&mut w, format!("{p}mlp.gate.bias"), Dt::F32, vec![c.n_experts], a);
            if c.is_loop(l) {
                for s in ["attn", "k", "v", "mlp"] {
                    put(&mut w, format!("{p}controllers.{s}.down_weight"), bf, vec![c.gate_rank, 2 * d], Controllers);
                    put(&mut w, format!("{p}controllers.{s}.forget_weight"), bf, vec![d, c.gate_rank], Controllers);
                    put(&mut w, format!("{p}controllers.{s}.gain_weight"), bf, vec![d, c.gate_rank], Controllers);
                }
            }
            let e = c.n_experts;
            let (i2, ii) = (2 * c.inter, c.inter);
            for (proj, n, kk) in [("w13", i2, d), ("w2", d, ii)] {
                let b = format!("{p}mlp.experts.{proj}_weight");
                match self.exp_fmt[l] {
                    ExpFmt::Nvfp4 => {
                        put(&mut w, b.clone(), Dt::U8, vec![e, n, kk / 2], ExpertsNvfp4);
                        put(&mut w, format!("{b}_scale"), Dt::F8e4m3, vec![e, n, kk / 16], ExpertsNvfp4);
                        put(&mut w, format!("{b}_amax"), Dt::F32, vec![e], ExpertsNvfp4);
                        put(&mut w, format!("{b}_outer_scale_denominator"), Dt::F32, vec![1], ExpertsNvfp4);
                    }
                    ExpFmt::Mxfp8 => {
                        put(&mut w, b.clone(), Dt::F8e4m3, vec![e, n, kk], ExpertsMxfp8);
                        put(&mut w, format!("{b}_scale"), Dt::U8, vec![e, n, kk / 32], ExpertsMxfp8);
                    }
                }
            }
        }
        for i in 0..c.layers - 1 {
            let shape = if c.is_sliding[i + 1] { vec![d] } else { vec![3, d] };
            put(&mut w, format!("{P}attn_res.attn_queries.{i}"), Dt::F32, shape, Class::AttnResQueries);
        }
        for i in 0..c.layers { put(&mut w, format!("{P}attn_res.mlp_queries.{i}"), Dt::F32, vec![d], Class::AttnResQueries); }
        put(&mut w, format!("{P}attn_res.output_query"), Dt::F32, vec![d], Class::AttnResQueries);
        put(&mut w, format!("{P}middle_cycle_cross_pass_queries"), Dt::F32, vec![2 * c.loop_len(), d], Class::AttnResQueries);
        put(&mut w, format!("{P}middle_cycle_cross_pass_kv_queries"), Dt::F32, vec![2 * c.loop_len(), d], Class::AttnResQueries);
        put(&mut w, format!("{P}embed.weight"), bf, vec![c.vocab, d], Class::EmbedUnembed);
        put(&mut w, format!("{P}unembed.weight"), bf, vec![c.vocab, d], Class::EmbedUnembed);
        put(&mut w, format!("{P}embed_norm.weight"), bf, vec![d], Class::AttnRouterConvNorm);
        put(&mut w, format!("{P}norm.weight"), bf, vec![d], Class::AttnRouterConvNorm);
        // vision tower (stock geometry; checked by name + dtype + shape)
        if self.st.has("model.visual.proj_out.weight") {
            let v = "model.visual.";
            let (vh, vm) = (1024usize, 4096usize);
            put(&mut w, format!("{v}backbone.embeddings.class_embedding"), bf, vec![vh], Class::Vision);
            put(&mut w, format!("{v}backbone.embeddings.cls_position_embedding"), bf, vec![vh], Class::Vision);
            put(&mut w, format!("{v}backbone.embeddings.patch_embedding.weight"), bf, vec![vh, 1200], Class::Vision);
            put(&mut w, format!("{v}backbone.embeddings.position_embedding"), bf, vec![1024, vh], Class::Vision);
            put(&mut w, format!("{v}backbone.ln_pre.bias"), bf, vec![vh], Class::Vision);
            put(&mut w, format!("{v}backbone.ln_pre.weight"), bf, vec![vh], Class::Vision);
            put(&mut w, format!("{v}proj_in.weight"), bf, vec![4 * vh, 4 * vh], Class::Vision);
            put(&mut w, format!("{v}proj_out.weight"), bf, vec![d, 4 * vh], Class::Vision);
            for l in 0..23 {
                let p = format!("{v}backbone.layers.{l}.");
                for s in ["ln_1", "ln_2"] { for x in ["bias", "weight"] { put(&mut w, format!("{p}{s}.{x}"), bf, vec![vh], Class::Vision); } }
                for s in ["ls_1", "ls_2"] { put(&mut w, format!("{p}{s}.gamma"), bf, vec![vh], Class::Vision); }
                put(&mut w, format!("{p}mlp.c_fc.bias"), bf, vec![vm], Class::Vision);
                put(&mut w, format!("{p}mlp.c_fc.weight"), bf, vec![vm, vh], Class::Vision);
                put(&mut w, format!("{p}mlp.c_proj.bias"), bf, vec![vh], Class::Vision);
                put(&mut w, format!("{p}mlp.c_proj.weight"), bf, vec![vh, vm], Class::Vision);
                for s in ["q", "k", "v", "out"] {
                    put(&mut w, format!("{p}self_attn.{s}_proj.bias"), bf, vec![vh], Class::Vision);
                    put(&mut w, format!("{p}self_attn.{s}_proj.weight"), bf, vec![vh, vh], Class::Vision);
                }
            }
        }
        let mut m = BTreeMap::new();
        if let Some(mc) = &c.mtp {
            let (md, mh, mkv, mhd, mi, r) = (mc.hidden, mc.heads, mc.kv_heads, mc.head_dim, mc.inter, mc.markov_rank);
            let nt = mc.target_layer_ids.len();
            put(&mut m, "fuse_weight".into(), bf, vec![mc.layers + 1, nt], Class::Mtp);
            for l in 0..mc.layers {
                let p = format!("layers.{l}.");
                for s in ["dflare_norm", "input_layernorm", "post_attention_layernorm"] { put(&mut m, format!("{p}{s}.weight"), bf, vec![md], Class::Mtp); }
                put(&mut m, format!("{p}mlp.gate_proj.weight"), bf, vec![mi, md], Class::Mtp);
                put(&mut m, format!("{p}mlp.up_proj.weight"), bf, vec![mi, md], Class::Mtp);
                put(&mut m, format!("{p}mlp.down_proj.weight"), bf, vec![md, mi], Class::Mtp);
                put(&mut m, format!("{p}self_attn.q_proj.weight"), bf, vec![mh * mhd, md], Class::Mtp);
                for s in ["k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"] { put(&mut m, format!("{p}self_attn.{s}.weight"), bf, vec![mkv * mhd, md], Class::Mtp); }
                put(&mut m, format!("{p}self_attn.o_proj.weight"), bf, vec![md, mh * mhd], Class::Mtp);
                for s in ["q_norm", "k_norm", "k_norm_ctx"] { put(&mut m, format!("{p}self_attn.{s}.weight"), bf, vec![mhd], Class::Mtp); }
            }
            put(&mut m, "markov_head.dflare_norm.weight".into(), bf, vec![md], Class::Mtp);
            put(&mut m, "markov_head.hidden_out.weight".into(), bf, vec![md, r], Class::Mtp);
            put(&mut m, "markov_head.hidden_out.bias".into(), bf, vec![md], Class::Mtp);
            put(&mut m, "markov_head.markov_w1.weight".into(), bf, vec![c.vocab, r], Class::Mtp);
            put(&mut m, "markov_head.rnn_proj.weight".into(), bf, vec![3 * r, 2 * r + 2 * md], Class::Mtp);
            put(&mut m, "markov_head.rnn_proj.bias".into(), bf, vec![3 * r], Class::Mtp);
            put(&mut m, "mask_embed".into(), bf, vec![md], Class::Mtp);
            put(&mut m, "norm.weight".into(), bf, vec![md], Class::Mtp);
        }
        (w, m)
    }

    /// Classify every tensor; any unknown, missing, mistyped or misshaped tensor is an error.
    /// Returns per-class (count, bytes) for the main file and the drafter file.
    pub fn census(&self) -> Result<(BTreeMap<Class, (usize, u64)>, BTreeMap<Class, (usize, u64)>)> {
        let (want, want_m) = self.wanted();
        let check = |st: &StDir, want: &BTreeMap<String, Want>, what: &str| -> Result<BTreeMap<Class, (usize, u64)>> {
            let mut errs = Vec::new();
            let mut out: BTreeMap<Class, (usize, u64)> = BTreeMap::new();
            for (n, t) in &st.t {
                match want.get(n) {
                    None => errs.push(format!("{what}: unclassified tensor {n} {:?} {:?}", t.dtype, t.shape)),
                    Some(w) => {
                        if w.dt != t.dtype || w.shape != t.shape {
                            errs.push(format!("{what}: {n}: {:?} {:?}, expected {:?} {:?}", t.dtype, t.shape, w.dt, w.shape));
                        }
                        let e = out.entry(w.class).or_default();
                        e.0 += 1;
                        e.1 += t.len;
                    }
                }
            }
            for n in want.keys() { if !st.t.contains_key(n) { errs.push(format!("{what}: missing tensor {n}")); } }
            if !errs.is_empty() {
                let k = errs.len();
                errs.truncate(20);
                bail!("{k} census errors:\n  {}", errs.join("\n  "));
            }
            Ok(out)
        };
        let main = check(&self.st, &want, "model")?;
        let mtp = match &self.mtp { Some(m) => check(m, &want_m, "mtp")?, None => BTreeMap::new() };
        // hf_quant_config cross-check: the nvfp4 / mxfp8 groups name exactly the expert stems we classified.
        if let Some(q) = &self.quant_json {
            if let Some(groups) = q.pointer("/model/parameter_groups").and_then(|x| x.as_array()) {
                let mut nv = Vec::new();
                let mut mx = Vec::new();
                for g in groups {
                    let cd = g.get("compute_dtype").and_then(|x| x.as_str()).unwrap_or("");
                    let names: Vec<String> = g.get("tensors").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|y| y.as_str().map(String::from)).collect()).unwrap_or_default();
                    for n in &names {
                        ensure!(self.st.has(n), "hf_quant_config names {n}, not in the pack");
                    }
                    if cd == "nvfp4" { nv.extend(names); } else if cd == "mxfp8" { mx.extend(names); }
                }
                for l in 0..self.cfg.layers {
                    for proj in ["w13", "w2"] {
                        let n = format!("{P}layers.{l}.mlp.experts.{proj}_weight");
                        let (inv, inm) = (nv.contains(&n), mx.contains(&n));
                        match self.exp_fmt[l] {
                            ExpFmt::Nvfp4 => ensure!(inv && !inm, "hf_quant_config: {n} is not in the nvfp4 group"),
                            ExpFmt::Mxfp8 => ensure!(inm && !inv, "hf_quant_config: {n} is not in the mxfp8 group"),
                        }
                    }
                }
            }
        }
        Ok((main, mtp))
    }

    /// Expert `e` of layer `l`, `proj` = "w13" or "w2", dequantized to f32, row-major `[n][k]`.
    /// `w13` is returned DE-INTERLEAVED (`[gate rows; up rows]`, the reference's `gate_up_proj`).
    pub fn expert_f32(&self, l: usize, e: usize, proj: &str) -> Result<(usize, usize, Vec<f32>)> {
        let b = format!("{P}layers.{l}.mlp.experts.{proj}_weight");
        let t = self.st.get(&b)?;
        let (ne, n) = (t.shape[0], t.shape[1]);
        ensure!(e < ne, "{b}: expert {e} >= {ne}");
        let mut out;
        let k;
        match self.exp_fmt[l] {
            ExpFmt::Nvfp4 => {
                let kh = t.shape[2];
                k = kh * 2;
                let packed = self.st.raw_range(&b, (e * n * kh) as u64, ((e + 1) * n * kh) as u64)?;
                let kb = k / 16;
                let sc = self.st.raw_range(&format!("{b}_scale"), (e * n * kb) as u64, ((e + 1) * n * kb) as u64)?;
                let amax = self.st.plain_f32(&format!("{b}_amax"))?;
                let den = self.st.plain_f32(&format!("{b}_outer_scale_denominator"))?[0];
                let fac = amax[e] / den;
                out = vec![0f32; n * k];
                for r in 0..n {
                    for c in 0..k {
                        let byte = packed[r * kh + c / 2];
                        let q = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
                        out[r * k + c] = (E2M1[q as usize] * e4m3(sc[r * kb + c / 16])) * fac;
                    }
                }
            }
            ExpFmt::Mxfp8 => {
                k = t.shape[2];
                let codes = self.st.raw_range(&b, (e * n * k) as u64, ((e + 1) * n * k) as u64)?;
                let kb = k / 32;
                let sc = self.st.raw_range(&format!("{b}_scale"), (e * n * kb) as u64, ((e + 1) * n * kb) as u64)?;
                out = vec![0f32; n * k];
                for r in 0..n {
                    for c in 0..k {
                        let s = sc[r * kb + c / 32];
                        out[r * k + c] = e4m3(codes[r * k + c]) * exp2_ue8m0(s);
                    }
                }
            }
        }
        if proj == "w13" { out = deinterleave_rows(&out, n, k); }
        Ok((n, k, out))
    }

    /// Raw expert bytes `(codes, scales)` of one expert (for the dequant cache).
    pub fn plain(&self, name: &str) -> Result<Vec<f32>> { self.st.plain_f32(name) }
    pub fn mtp_plain(&self, name: &str) -> Result<Vec<f32>> { self.mtp.as_ref().context("no mtp.safetensors")?.plain_f32(name) }
}

/// `torch.exp2(float(s) - 127)` (s = 255 -> +inf, s = 0 -> 2^-127 subnormal; both exact in f32).
pub fn exp2_ue8m0(s: u8) -> f32 {
    if s == 255 { return f32::INFINITY; }
    if s == 0 { return f32::from_bits(1 << 22); }
    f32::from_bits((s as u32) << 23)
}

/// Row `2i` -> row `i`, row `2i+1` -> row `n/2 + i` (the reference's `Interleave(dim=1)`).
pub fn deinterleave_rows(x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let h = n / 2;
    let mut o = vec![0f32; n * k];
    for i in 0..h {
        o[i * k..(i + 1) * k].copy_from_slice(&x[(2 * i) * k..(2 * i + 1) * k]);
        o[(h + i) * k..(h + i + 1) * k].copy_from_slice(&x[(2 * i + 1) * k..(2 * i + 2) * k]);
    }
    o
}

fn gb(b: u64) -> f64 { b as f64 / 1e9 }

/// `--probe-inkling2-pack --model-dir <dir> [--oracle-dir <ref>]`: census + dequant check against a
/// reference dump (`tools/inkling2/ref_dequant.py` writes `<ref>/dequant_ref.safetensors`).
pub fn probe_pack(dir: &Path, ref_dir: Option<&str>) -> Result<()> {
    let p = Inkling2Pack::open(dir)?;
    let c = &p.cfg;
    println!("INK2_CONFIG hidden={} layers={} visits={} vocab={}/{} heads={}/{}x{} swa={}/{}x{} window={} d_rel={} rel_extent={} experts={} top{} inter={} route_scale={} loop=[{},{})x{} gate_rank={} block={} conv_k={} eos={:?}",
             c.hidden, c.layers, c.visits(), c.unpadded_vocab, c.vocab, c.heads, c.kv_heads, c.head_dim, c.swa_heads, c.swa_kv_heads, c.swa_head_dim,
             c.sliding_window, c.d_rel, c.rel_extent, c.n_experts, c.topk, c.inter, c.route_scale, c.loop_start, c.loop_end, c.passes, c.gate_rank, c.block, c.conv_k, c.eos);
    let full: Vec<usize> = (0..c.layers).filter(|&l| !c.is_sliding[l]).collect();
    println!("INK2_LAYERS full={:?} nvfp4={:?} mxfp8={:?}", full,
             (0..c.layers).filter(|&l| p.exp_fmt[l] == ExpFmt::Nvfp4).collect::<Vec<_>>(),
             (0..c.layers).filter(|&l| p.exp_fmt[l] == ExpFmt::Mxfp8).collect::<Vec<_>>());
    if let Some(m) = &c.mtp {
        println!("INK2_MTP_CONFIG layers={} hidden={} inter={} heads={}/{}x{} local={:?} window={} block={} markov={}x{} taps={:?} rope_theta={}",
                 m.layers, m.hidden, m.inter, m.heads, m.kv_heads, m.head_dim, m.local_layer_ids, m.sliding_window, m.block_size,
                 m.markov_head_type, m.markov_rank, m.target_layer_ids, m.rope_theta);
    }
    let (main, mtp) = p.census()?;
    let (mut n0, mut b0) = (0usize, 0u64);
    for (k, (n, b)) in &main { println!("INK2_CENSUS model {k:?}: {n} tensors, {:.3} GB", gb(*b)); n0 += n; b0 += b; }
    println!("INK2_CENSUS model TOTAL {n0} tensors, {:.3} GB (every tensor classified, dtype+shape checked)", gb(b0));
    let (mut n1, mut b1) = (0usize, 0u64);
    for (k, (n, b)) in &mtp { println!("INK2_CENSUS mtp {k:?}: {n} tensors, {:.3} GB", gb(*b)); n1 += n; b1 += b; }
    println!("INK2_CENSUS mtp TOTAL {n1} tensors, {:.3} GB", gb(b1));
    // MXFP8 scale structure: the quant config says weight_block_shape [32, 32]; the file stores one
    // scale per (row, 32 columns). Count rows whose scales equal their 32-row block's first row.
    if let Some(l) = (0..c.layers).find(|&l| p.exp_fmt[l] == ExpFmt::Mxfp8) {
        let b = format!("{P}layers.{l}.mlp.experts.w2_weight_scale");
        let t = p.st.get(&b)?;
        let (n, kb) = (t.shape[1], t.shape[2]);
        let sc = p.st.raw_range(&b, 0, (n * kb) as u64)?;
        let same = (0..n).filter(|&r| sc[r * kb..(r + 1) * kb] == sc[(r - r % 32) * kb..(r - r % 32 + 1) * kb]).count();
        println!("INK2_MX8_SCALES layer {l} w2 expert 0: {same}/{n} rows equal their 32-row block's first row (stored per row; 32x32 blocks would make this {n}/{n})");
    }
    // NVFP4 amax / denominator ranges
    for l in [0usize, c.layers / 2] {
        if p.exp_fmt[l] != ExpFmt::Nvfp4 { continue; }
        for proj in ["w13", "w2"] {
            let b = format!("{P}layers.{l}.mlp.experts.{proj}_weight");
            let a = p.plain(&format!("{b}_amax"))?;
            let dn = p.plain(&format!("{b}_outer_scale_denominator"))?;
            let (lo, hi) = a.iter().fold((f32::MAX, f32::MIN), |(x, y), &v| (x.min(v), y.max(v)));
            println!("INK2_NV4 layer {l} {proj}: amax [{lo:.4e}, {hi:.4e}] denominator {:?}", dn);
        }
    }
    if let Some(rd) = ref_dir {
        let rp = Path::new(rd).join("dequant_ref.safetensors");
        let r = open_single(&rp)?;
        let mut names: Vec<&String> = r.t.keys().collect();
        names.sort();
        let mut all_ok = true;
        for n in names {
            // name: L{l}.E{e}.{w13|w2}
            let parts: Vec<&str> = n.split('.').collect();
            ensure!(parts.len() == 3, "ref name {n}");
            let l: usize = parts[0][1..].parse()?;
            let e: usize = parts[1][1..].parse()?;
            let (rows, cols, ours) = p.expert_f32(l, e, parts[2])?;
            let theirs = r.plain_f32(n)?;
            ensure!(r.t[n].shape == vec![rows, cols], "{n}: ref shape {:?} vs ours {rows}x{cols}", r.t[n].shape);
            let mut nbad = 0usize;
            let mut maxd = 0f32;
            for (a, b) in ours.iter().zip(theirs.iter()) {
                if a.to_bits() != b.to_bits() && !(*a == 0.0 && *b == 0.0) { nbad += 1; maxd = maxd.max((a - b).abs()); }
            }
            let fmt = format!("{:?}", p.exp_fmt[l]);
            println!("INK2_DEQUANT {n} ({fmt}) {rows}x{cols}: {} / {} values differ bitwise (max |d| {maxd:.3e})", nbad, ours.len());
            if nbad > 0 { all_ok = false; }
        }
        if all_ok { println!("INK2_DEQUANT_OK every checked expert matches the reference bit for bit"); } else { bail!("INK2_DEQUANT_FAIL"); }
    }
    println!("INK2_PACK_OK");
    Ok(())
}
